//! What machine code calls and reads: the context (`Ctx`, one per entry from the interpreter), the runtime functions
//! (`j8_helper`, `j8_deopt`, `j8_poll`), the trampolines calls between compiled functions go through, and the
//! executable memory.
//!
//! ```text
//! Ctx  +0 budget  +8 pending  +16 global  +24 script scope  +32 Object.prototype  +40 ret  +48 ret pc  +56 acc
//!      +64 depth  +72 depth limit  +128 ARGS (16 words)  +256 BUF (deopt values)
//! trampoline  jmp [rip + cell]   one per function: its native code, or the interpreter until it has some
//! ```
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cell::RefCell;

use j8executor::{DeoptFrame, Entry, Executor};
use j8framework::{Agent, Error, ObjectData, Slot as PropSlot, Value, ValueBits};

use crate::build::{CONTEXT_TAG, OBJECT_TAG};
use crate::ir::{Helper, Rep, Slot};
use crate::codegen::{Reason, RESUMED, THREW};
use crate::Jit;

/// How many values a deopt can hand over through the buffer
pub const BUF_WORDS: usize = 2048;
/// Operands of runtime calls with more than four
pub const ARGS_WORDS: usize = 16;
/// Loop iterations between polls
pub const BUDGET_CHUNK: i64 = 1024;
/// Native frames at most (deeper hands over to the interpreter, which throws when it is too deep)
pub const DEPTH_LIMIT: u64 = 2000;
/// Stack native code may use below where the JIT first ran (machine code past it hands over)
pub const NATIVE_STACK: u64 = 1 << 20;

/// The context machine code runs with (see the module docs for the layout).
#[repr(C)]
pub struct Ctx {
    pub budget: i64,
    pub pending: u64,
    pub global: u64,
    pub script_scope: u64,
    pub object_proto: u64,
    pub ret: u64,
    pub ret_pc: u64,
    pub acc: u64,
    pub depth: u64,
    pub limit: u64,
    /// Changes whenever code other than compiled code may have run (global cache sites compare it)
    pub epoch: u64,
    /// A global cache site's slow path found another function than expected
    pub miss: u64,
    /// Native code doesn't go below this stack address (deeper calls hand over to the interpreter)
    pub stack_floor: u64,
    pad: [u64; 3],
    pub args: [u64; ARGS_WORDS],
    pub buf: [u64; BUF_WORDS],
    // ---- not read by machine code ----
    /// The entry (lifetimes erased: only used while the entry is on the stack); null for a call from the executor
    /// without an interpreter frame
    pub entry: *mut core::ffi::c_void,
    /// The executor and the agent (lifetimes erased likewise)
    pub exe: *const core::ffi::c_void,
    pub agent: *mut core::ffi::c_void,
    pub jit: *const Jit,
    pub error: Option<Error>,
}

impl Ctx {
    pub const BUDGET: i32 = 0;
    pub const PENDING: i32 = 8;
    pub const GLOBAL: i32 = 16;
    pub const SCRIPT_SCOPE: i32 = 24;
    pub const OBJECT_PROTO: i32 = 32;
    pub const RET: i32 = 40;
    pub const RET_PC: i32 = 48;
    pub const ACC: i32 = 56;
    pub const EPOCH: i32 = 80;
    pub const MISS: i32 = 88;
    pub const STACK_FLOOR: i32 = 96;
    pub const ARGS: i32 = 128;
    pub const BUF: i32 = 256;

    /// A context for running machine code on `e`: a recycled one (see [`Jit::recycle`]) or a new one
    pub fn new(e: &mut Entry<'_, '_, '_, '_>, jit: &Jit) -> Box<Ctx> {
        let acc = e.acc().bits();
        let entry: *mut core::ffi::c_void = core::ptr::from_mut(e).cast();
        let exe = e.executor();
        let mut c = Self::bare(exe, e.agent(), jit);
        c.acc = acc;
        c.entry = entry;
        c
    }

    /// A context for a call from the executor (no entry frame)
    pub fn bare(exe: &Executor, a: &mut Agent<'_, '_>, jit: &Jit) -> Box<Ctx> {
        let addr = |v: Value<'_>| v.bits() & ValueBits::PAYLOAD;
        let (global, scope, proto) = (addr(Value::Object(a.realm.global)), addr(Value::Object(a.realm.script_scope)), addr(Value::Object(a.realm.object_proto)));
        let exe: *const core::ffi::c_void = core::ptr::from_ref(exe).cast();
        let agent: *mut core::ffi::c_void = core::ptr::from_mut(a).cast();
        if let Some(mut c) = jit.spare_ctx() {
            c.budget = BUDGET_CHUNK;
            c.pending = 0;
            c.global = global;
            c.script_scope = scope;
            c.object_proto = proto;
            c.acc = ValueBits::UNDEFINED;
            c.epoch = jit.bump();
            c.miss = 0;
            c.stack_floor = jit.stack_floor();
            c.entry = core::ptr::null_mut();
            c.exe = exe;
            c.agent = agent;
            c.error = None;
            return c;
        }
        Box::new(Ctx {
            budget: BUDGET_CHUNK,
            pending: 0,
            global,
            script_scope: scope,
            object_proto: proto,
            ret: 0,
            ret_pc: 0,
            acc: ValueBits::UNDEFINED,
            depth: 0,
            limit: DEPTH_LIMIT,
            epoch: jit.bump(),
            miss: 0,
            stack_floor: jit.stack_floor(),
            pad: [0; 3],
            args: [0; ARGS_WORDS],
            buf: [0; BUF_WORDS],
            entry: core::ptr::null_mut(),
            exe,
            agent,
            jit,
            error: None,
        })
    }

    /// The executor and the agent (erased lifetimes: `'gc` is the running mutate's, for the duration of the call)
    ///
    /// # Safety
    /// Only while the code this context was made for runs.
    pub unsafe fn parts<'a>(&mut self) -> (&'a Executor, &'a mut Agent<'a, 'a>) {
        // SAFETY: the caller's contract
        unsafe { (&*self.exe.cast(), &mut *self.agent.cast()) }
    }

    /// The entry and agent (erased lifetimes: `'gc` is the running mutate's, for the duration of the call)
    ///
    /// # Safety
    /// Only while the entry this context was made for is running machine code.
    pub unsafe fn entry<'a>(&mut self) -> &'a mut Entry<'a, 'a, 'a, 'a> {
        // SAFETY: the caller's contract
        unsafe { &mut *self.entry.cast() }
    }

    fn jit<'a>(&self) -> &'a Jit {
        // SAFETY: the JIT outlives every entry it runs
        unsafe { &*self.jit }
    }

    fn fail(&mut self, e: Error) -> u64 {
        self.pending = 1;
        self.error = Some(e);
        0
    }
}

/// A word ⇒ a value of this mutate
///
/// # Safety
/// The word came from this mutate's heap (or is a scalar).
unsafe fn val<'gc>(w: u64) -> Value<'gc> {
    // SAFETY: the caller's contract
    unsafe { Value::from_bits(w) }
}

/// The value a deopt slot holds in representation `rep` with raw bits `bits`
///
/// # Safety
/// As [`val`].
#[expect(clippy::cast_possible_truncation, reason = "an i32 sign-extended to 64 bits")]
unsafe fn boxed<'gc>(rep: Rep, bits: u64) -> Value<'gc> {
    match rep {
        Rep::I32 => Value::Number(f64::from(bits as i64 as i32)),
        Rep::F64 => Value::Number(f64::from_bits(bits)),
        Rep::Bool => Value::Bool(bits & 1 != 0),
        // SAFETY: an address of this mutate's heap
        Rep::Ptr => unsafe { val(bits | OBJECT_TAG) },
        // SAFETY: as above
        Rep::Ctx => unsafe { val(bits | CONTEXT_TAG) },
        // SAFETY: the caller's contract
        Rep::Word | Rep::None => unsafe { val(bits) },
    }
}

/// A runtime call: `id` names the helper (see `lower`); `a … d` its operands (more through `Ctx::ARGS`: then `a` is
/// their count)
#[unsafe(no_mangle)]
pub extern "C" fn j8_helper(ctx: *mut Ctx, id: u64, a: u64, b: u64, c: u64, d: u64) -> u64 {
    // SAFETY: machine code passes its context
    let ctx = unsafe { &mut *ctx };
    let jit = ctx.jit();
    let (h, n) = jit.call(id);
    let mut buf = [0u64; ARGS_WORDS];
    if n > 4 {
        buf[..n].copy_from_slice(&ctx.args[..n]);
    } else {
        buf[..4].copy_from_slice(&[a, b, c, d]);
    }
    let ops = &buf[..n.min(ARGS_WORDS)];
    // SAFETY: the code is running
    let (exe, a) = unsafe { ctx.parts() };
    let r = helper(jit, exe, a, h, ops, ctx.epoch);
    if !matches!(h, Helper::ModF | Helper::ToBoolean | Helper::StoreCell | Helper::Alloc(_) | Helper::CreateClosure(..) | Helper::CreateContext(..) | Helper::ObjectLiteral(..) | Helper::GlobalProbe(_)) {
        ctx.epoch = jit.bump();
    }
    if let (Helper::GlobalProbe(_), Ok(GLOBAL_MISS)) = (h, &r) {
        ctx.miss = 1;
        return 0;
    }
    match r {
        Ok(w) => w,
        Err(err) => ctx.fail(err),
    }
}

/// What a global cache probe gives when the function isn't the expected one
const GLOBAL_MISS: u64 = u64::MAX;

fn helper<'gc>(jit: &Jit, exe: &Executor, a: &mut Agent<'gc, '_>, h: Helper, ops: &[u64], epoch: u64) -> Result<u64, Error> {
    // SAFETY: operands are words / addresses of this mutate
    let v = |i: usize| unsafe { val::<'gc>(ops.get(i).copied().unwrap_or(ValueBits::UNDEFINED)) };
    Ok(match h {
        Helper::Call => {
            let mut args = [Value::Undefined; ARGS_WORDS];
            let n = ops.len().saturating_sub(2);
            for (i, x) in args.iter_mut().enumerate().take(n) {
                *x = v(2 + i);
            }
            a.call(v(0), v(1), &args[..n])?.bits()
        }
        Helper::Construct => {
            let args: Vec<Value<'gc>> = (2..ops.len()).map(v).collect();
            a.construct(v(0), &args, v(1))?.bits()
        }
        Helper::GetNamed(f, k) => {
            let name = exe.jit_function(f).name(i64::from(k));
            a.get_named(v(0), name)?.bits()
        }
        Helper::SetNamed(f, k, assign) => {
            let name = exe.jit_function(f).name(i64::from(k));
            match (assign, v(0).as_object()) {
                (false, Some(o)) => a.define(o, name, j8framework::Property::data(v(1)))?,
                // As the interpreter: a failed assignment throws in strict code only
                _ => a.put_named(v(0), name, v(1), exe.jit_function(f).strict())?,
            }
            0
        }
        Helper::CreateClosure(f, k) => Value::Object(exe.jit_create_closure(a, f, i64::from(k), v(0).as_context())).bits(),
        Helper::CreateContext(shape, n) => Value::context(j8framework::Env::context(n as usize, Value::Undefined, v(0).as_context(), Some(shape), None).alloc(a.mc)).bits(),
        Helper::ObjectLiteral(f, k) => Value::Object(exe.jit_object_literal(a, f, i64::from(k))).bits(),
        Helper::ModF => (f64::from_bits(ops[0]) % f64::from_bits(ops[1])).to_bits(),
        Helper::ToBoolean => u64::from(a.to_boolean(v(0))),
        Helper::StoreCell => {
            // SAFETY: the address of a value cell of this mutate's heap
            let cell = unsafe { &*(ops[0] as usize as *const j8framework::ValueCell<'gc>) };
            cell.set(a.mc, v(1));
            0
        }
        Helper::LoadGlobal(f, k, slot) => exe.jit_global_load(a, f, i64::from(k), i64::from(slot))?.bits(),
        Helper::GlobalProbe(k) => {
            let (f, name, slot, expect) = jit.site(k);
            let v = exe.jit_global_load(a, f, i64::from(name), i64::from(slot))?;
            if let Some(want) = expect
                && v.as_object().and_then(|o| o.closure()).is_none_or(|c| c.func != want)
            {
                return Ok(GLOBAL_MISS);
            }
            // SAFETY: the site's cache: two words the code reads
            unsafe {
                let cache = ops[0] as usize as *mut u64;
                cache.write(epoch);
                cache.add(1).write(v.bits());
            }
            v.bits()
        }
        Helper::Binary(op) => a.binary(op, v(0), v(1))?.bits(),
        Helper::GetKeyed => a.get_keyed(v(0), v(1))?.bits(),
        Helper::SetKeyed(strict) => {
            a.put_keyed(v(0), v(1), v(2), strict)?;
            0
        }
        Helper::Unary(op) => a.unary(op, v(0))?.bits(),
        Helper::LooseEquals => u64::from(a.loose_equals(v(0), v(1))?),
        Helper::InstanceOf => u64::from(a.instance_of(v(0), v(1))?),
        Helper::Compare(op) => u64::from(a.compare(op, v(0), v(1))?),
        Helper::StrictEquals => u64::from(a.compare(j8framework::CompareOp::StrictEq, v(0), v(1))?),
        Helper::Alloc(k) => {
            let shape = jit.shape(k);
            // SAFETY: an object address of this mutate
            let proto = unsafe { val::<'gc>(ops[0] | OBJECT_TAG) }.as_object();
            let o = ObjectData::ordinary(a.mc, proto, ops.len() - 1);
            if let Some(s) = &shape {
                o.fill(a.mc, s, (1..ops.len()).map(|i| PropSlot::Data(v(i))));
            }
            Value::Object(o).bits() & ValueBits::PAYLOAD
        }
    })
}

/// A loop back edge spent its budget: `0` go on, `1` hand over (out of budget, or terminated)
#[unsafe(no_mangle)]
pub extern "C" fn j8_poll(ctx: *mut Ctx) -> u64 {
    // SAFETY: machine code passes its context
    let ctx = unsafe { &mut *ctx };
    ctx.budget = BUDGET_CHUNK;
    if ctx.entry.is_null() {
        // No interpreter frame below to pause: only a termination request stops it
        // SAFETY: the code is running
        let (_, a) = unsafe { ctx.parts() };
        return u64::from(a.stop.observe());
    }
    // SAFETY: the entry is running
    let e = unsafe { ctx.entry() };
    #[expect(clippy::cast_sign_loss, reason = "positive")]
    u64::from(e.spend(BUDGET_CHUNK as usize))
}

/// A guard failed (or the code hands over): rebuilds the interpreter frames of deopt `id` and continues there
#[unsafe(no_mangle)]
pub extern "C" fn j8_deopt(ctx: *mut Ctx, id: u64) -> Ret {
    Ret { value: deopt(ctx, id), int: 0 }
}

/// What native code returns: the value's word in rax, and in rdx [`Ret::INT`] | the int32 when it is one (0 else)
#[repr(C)]
pub struct Ret {
    pub value: u64,
    pub int: u64,
}

impl Ret {
    /// rdx's mark: the low 32 bits hold the value as an int32
    pub const INT: u64 = 1 << 32;
}

fn deopt(ctx: *mut Ctx, id: u64) -> u64 {
    // SAFETY: machine code passes its context
    let ctx = unsafe { &mut *ctx };
    let jit = ctx.jit();
    let d = jit.deopt(id);
    jit.learn(&d);
    // SAFETY: the code is running
    let (exe, a) = unsafe { ctx.parts() };
    let frames = frames(a, ctx, &d);
    ctx.epoch = jit.bump();
    let throw = if d.reason == Reason::Throw { ctx.error.take() } else { None };
    if d.native {
        ctx.pending = 0;
        let thrown = match throw {
            Some(Error::Exception(g)) => Some(a.local(&g)),
            Some(other) => return ctx.fail(other),
            None => None,
        };
        let saved = exe.jit_depth().get();
        let r = exe.resume_frames(a, frames, thrown);
        exe.jit_depth().set(saved);
        return match r {
            Ok(v) => v.bits(),
            Err(err) => ctx.fail(err),
        };
    }
    // SAFETY: interpreter-entry code runs on an entry
    let e = unsafe { ctx.entry() };
    e.resume(frames);
    match throw {
        Some(err) => {
            ctx.error = Some(err);
            THREW as u64
        }
        None => RESUMED as u64,
    }
}

/// The interpreter frames a deopt's state stands for
fn frames<'gc>(a: &mut Agent<'gc, '_>, ctx: &Ctx, d: &crate::Deopt) -> Vec<DeoptFrame<'gc>> {
    let st = d.state();
    let mut objs: Vec<Option<Value<'gc>>> = alloc::vec![None; st.objects.len()];
    let get = |v: u32| -> Value<'gc> {
        let (rep, src) = d.value(v);
        let bits = match src {
            crate::Src::Buf(i) => ctx.buf[i as usize],
            crate::Src::Const(b) => b,
        };
        // SAFETY: words / addresses of this mutate
        unsafe { boxed(rep, bits) }
    };
    // Objects first (fields may be objects too: those come first in the list, see `Builder::slot`; materialize in
    // reverse so inner ones exist)
    let cell = RefCell::new(&mut objs);
    let slot = |s: &Slot| -> Value<'gc> {
        match s {
            Slot::Value(v) => get(*v),
            Slot::Virtual(i) => cell.borrow()[*i as usize].unwrap_or(Value::Undefined),
            Slot::Closure => Value::Uninit,
        }
    };
    for i in (0..st.objects.len()).rev() {
        let o = &st.objects[i];
        let proto = get(o.proto).as_object();
        let fields: Vec<Value<'gc>> = o.fields.iter().map(slot).collect();
        let ob = ObjectData::ordinary(a.mc, proto, fields.len());
        if let Some(s) = &o.shape {
            ob.fill(a.mc, s, fields.into_iter().map(PropSlot::Data));
        }
        cell.borrow_mut()[i] = Some(Value::Object(ob));
    }
    for (c, idx, s) in &st.writes {
        let env = get(*c);
        // SAFETY: a context address of this mutate
        let env = unsafe { val::<'gc>((env.bits() & ValueBits::PAYLOAD) | CONTEXT_TAG) }.as_context();
        if let Some(env) = env {
            j8framework::Env::set(env, a.mc, *idx as usize, slot(s));
        }
    }
    st.frames.iter().map(|f| DeoptFrame {
        func: f.func,
        callee: slot(&f.callee),
        params: f.params.iter().map(slot).collect(),
        argc: if f.argc == u32::MAX { f.params.len().saturating_sub(1) } else { f.argc as usize },
        regs: f.regs.iter().map(slot).collect(),
        context: {
            let c = slot(&f.context);
            if c.as_context().is_some() || c.bits() == ValueBits::UNDEFINED || c.bits() == ValueBits::UNINIT { c } else {
                // SAFETY: a context address of this mutate
                unsafe { val::<'gc>((c.bits() & ValueBits::PAYLOAD) | CONTEXT_TAG) }
            }
        },
        acc: slot(&f.acc),
        pc: f.pc as usize,
        construct: f.construct.as_ref().map(|(n, t)| (slot(n), slot(t))),
    }).collect()
}

/// A call through a trampoline that has no native code yet: run it in the interpreter (and count it)
#[unsafe(no_mangle)]
pub extern "C" fn j8_interp(ctx: *mut Ctx, callee: u64, this: u64, a0: u64, a1: u64, a2: u64) -> Ret {
    Ret { value: interp(ctx, callee, this, a0, a1, a2), int: 0 }
}

fn interp(ctx: *mut Ctx, callee: u64, this: u64, a0: u64, a1: u64, a2: u64) -> u64 {
    // SAFETY: machine code passes its context
    let ctx = unsafe { &mut *ctx };
    let jit = ctx.jit();
    // SAFETY: the code is running
    let (exe, a) = unsafe { ctx.parts() };
    // SAFETY: words of this mutate
    let (f, t, args) = unsafe { (val(callee), val(this), [val(a0), val(a1), val(a2)]) };
    if let Some(c) = f.as_object().and_then(|o| o.closure()) {
        jit.native_hot(exe, a, c.func);
    }
    let n = f.as_object().and_then(|o| o.closure()).map_or(3, |c| exe.jit_function(c.func).shared().formal_parameters as usize).min(3);
    let r = a.call(f, t, &args[..n]);
    ctx.epoch = jit.bump();
    match r {
        Ok(v) => v.bits(),
        Err(err) => ctx.fail(err),
    }
}

/// Executable memory that pieces of code are cut from (no permission changes per piece), kept for as long as the JIT
/// lives. On Linux a chunk is an anonymous file mapped shared twice, writable at one address and executable at
/// another. Elsewhere both views are one private anonymous mapping, readable, writable and executable.
#[derive(Default)]
pub struct Arena {
    /// (writable view, executable view)
    chunks: Vec<(Mapped, Mapped)>,
    /// Bytes used in the last chunk
    used: usize,
}

impl Arena {
    /// Bytes per chunk
    const CHUNK: usize = 1 << 20;
    /// Code starts at multiples of this
    const ALIGN: usize = 16;

    /// A chunk of `len` bytes: its two views
    #[cfg(target_os = "linux")]
    fn chunk(len: usize) -> Option<(Mapped, Mapped)> {
        // SAFETY: a fresh anonymous file, sized, mapped shared twice
        unsafe {
            let fd = libc::memfd_create(c"j8jit".as_ptr(), libc::MFD_CLOEXEC);
            if fd < 0 {
                return None;
            }
            let ok = libc::ftruncate(fd, libc::off_t::try_from(len).ok()?) == 0;
            let map = |prot| libc::mmap(core::ptr::null_mut(), len, prot, libc::MAP_SHARED, fd, 0);
            let (w, x) = if ok { (map(libc::PROT_READ | libc::PROT_WRITE), map(libc::PROT_READ | libc::PROT_EXEC)) } else { (libc::MAP_FAILED, libc::MAP_FAILED) };
            libc::close(fd);
            if w == libc::MAP_FAILED || x == libc::MAP_FAILED {
                return None;
            }
            Some((Mapped { at: w.cast(), len }, Mapped { at: x.cast(), len }))
        }
    }

    /// A chunk of `len` bytes: both views of one private anonymous mapping
    #[cfg(not(target_os = "linux"))]
    fn chunk(len: usize) -> Option<(Mapped, Mapped)> {
        // SAFETY: a fresh private anonymous mapping, readable, writable and executable
        unsafe {
            let at = libc::mmap(core::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC, libc::MAP_PRIVATE | libc::MAP_ANON, -1, 0);
            if at == libc::MAP_FAILED {
                return None;
            }
            let at = at.cast();
            Some((Mapped { at, len }, Mapped { at, len }))
        }
    }

    /// `bytes` placed executable (`patch` fixes it up, given its executable address): that address
    pub fn place(&mut self, bytes: &[u8], patch: impl FnOnce(&mut [u8], usize)) -> Option<usize> {
        let len = bytes.len().max(1);
        let at = self.used.next_multiple_of(Self::ALIGN);
        if self.chunks.last().is_none_or(|(w, _)| at + len > w.len) {
            self.chunks.push(Self::chunk(len.next_multiple_of(Self::CHUNK))?);
            self.used = 0;
            return self.place(bytes, patch);
        }
        let (w, x) = self.chunks.last()?;
        // SAFETY: [at, at + len) of the writable view, not used by any other piece
        let buf = unsafe { core::slice::from_raw_parts_mut(w.at.add(at), len) };
        buf[..bytes.len()].copy_from_slice(bytes);
        let addr = x.at as usize + at;
        patch(buf, addr);
        self.used = at + len;
        Some(addr)
    }
}

/// Executable memory: one mapping.
pub struct Mapped {
    pub at: *mut u8,
    pub len: usize,
}


/// Trampolines: `jmp [rip + d]` to a cell in a writable page next to them. One page of stubs, one of cells.
pub struct Trampolines {
    at: *mut u8,
    used: usize,
}

impl Trampolines {
    const PAGE: usize = 4096;
    const STUB: usize = 8;
    pub const PER_PAGE: usize = Self::PAGE / Self::STUB;

    pub fn new() -> Option<Self> {
        // SAFETY: a fresh mapping: stubs page read-execute, cells page read-write
        unsafe {
            let at = libc::mmap(core::ptr::null_mut(), 2 * Self::PAGE, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0);
            if at == libc::MAP_FAILED {
                return None;
            }
            let at = at.cast::<u8>();
            for i in 0..Self::PER_PAGE {
                let stub = at.add(i * Self::STUB);
                let cell = at.add(Self::PAGE + i * 8);
                // jmp [rip + disp32]: rip is after the 6-byte instruction
                let disp = i32::try_from(cell as isize - (stub as isize + 6)).unwrap_or(0);
                let mut code = [0xff, 0x25, 0, 0, 0, 0, 0xcc, 0xcc];
                code[2..6].copy_from_slice(&disp.to_le_bytes());
                core::ptr::copy_nonoverlapping(code.as_ptr(), stub, Self::STUB);
            }
            if libc::mprotect(at.cast(), Self::PAGE, libc::PROT_READ | libc::PROT_EXEC) != 0 {
                return None;
            }
            Some(Trampolines { at, used: 0 })
        }
    }

    /// A free stub pointing at `target`: its address
    pub fn take(&mut self, target: usize) -> Option<usize> {
        if self.used == Self::PER_PAGE {
            return None;
        }
        let i = self.used;
        self.used += 1;
        self.set(i, target);
        Some(self.at as usize + i * Self::STUB)
    }

    /// Repoints stub `stub` (an address from `take`)
    pub fn set_at(&self, stub: usize, target: usize) {
        let i = (stub - self.at as usize) / Self::STUB;
        self.set(i, target);
    }

    fn set(&self, i: usize, target: usize) {
        // SAFETY: cell `i` of the writable page
        unsafe { core::ptr::write_volatile(self.at.add(Self::PAGE + i * 8).cast::<usize>(), target) };
    }

    pub fn owns(&self, stub: usize) -> bool {
        stub >= self.at as usize && stub < self.at as usize + Self::PAGE
    }
}

impl Drop for Trampolines {
    fn drop(&mut self) {
        // SAFETY: the mapping made in `new`
        unsafe { libc::munmap(self.at.cast(), 2 * Self::PAGE) };
    }
}
