// Copyright 2018-2022 the Deno authors. All rights reserved. MIT license.

use std::ffi::c_void;
use std::iter::once;

use deno_core::v8::fast_api;
use dynasmrt::dynasm;
use dynasmrt::DynasmApi;
use dynasmrt::ExecutableBuffer;

use crate::NativeType;
use crate::Symbol;

pub(crate) fn is_compatible(sym: &Symbol) -> bool {
  cfg!(all(target_arch = "x86_64", target_family = "unix"))
    && !sym.can_callback
    && is_fast_api_rv(sym.result_type)
}

pub(crate) fn compile_trampoline(sym: &Symbol) -> Trampoline {
  #[cfg(all(target_arch = "x86_64", target_family = "unix"))]
  return SysVAmd64::compile(sym);
  #[cfg(target_arch = "aarch64")]
  return Aarch64::compile(sym);
  todo!()
}

pub(crate) fn make_template(sym: &Symbol, trampoline: &Trampoline) -> Template {
  let args = once(fast_api::Type::V8Value)
    .chain(sym.parameter_types.iter().map(|t| t.into()))
    .collect::<Vec<_>>();

  Template {
    args: args.into_boxed_slice(),
    ret: (&fast_api::Type::from(&sym.result_type)).into(),
    symbol_ptr: trampoline.ptr(),
  }
}

/// Trampoline for fast-call FFI functions
///
/// Calls the FFI function without the first argument (the receiver)
pub(crate) struct Trampoline(ExecutableBuffer);

impl Trampoline {
  fn ptr(&self) -> *const c_void {
    &self.0[0] as *const u8 as *const c_void
  }
}

pub(crate) struct Template {
  args: Box<[fast_api::Type]>,
  ret: fast_api::CType,
  symbol_ptr: *const c_void,
}

impl fast_api::FastFunction for Template {
  fn function(&self) -> *const c_void {
    self.symbol_ptr
  }

  fn args(&self) -> &'static [fast_api::Type] {
    Box::leak(self.args.clone())
  }

  fn return_type(&self) -> fast_api::CType {
    self.ret
  }
}

impl From<&NativeType> for fast_api::Type {
  fn from(native_type: &NativeType) -> Self {
    match native_type {
      NativeType::U8 | NativeType::U16 | NativeType::U32 => {
        fast_api::Type::Uint32
      }
      NativeType::I8 | NativeType::I16 | NativeType::I32 => {
        fast_api::Type::Int32
      }
      NativeType::F32 => fast_api::Type::Float32,
      NativeType::F64 => fast_api::Type::Float64,
      NativeType::Void => fast_api::Type::Void,
      NativeType::I64 => fast_api::Type::Int64,
      NativeType::U64 => fast_api::Type::Uint64,
      NativeType::ISize => fast_api::Type::Int64,
      NativeType::USize | NativeType::Function | NativeType::Pointer => {
        fast_api::Type::Uint64
      }
    }
  }
}

fn is_fast_api_rv(rv: NativeType) -> bool {
  !matches!(
    rv,
    NativeType::Function
      | NativeType::Pointer
      | NativeType::I64
      | NativeType::ISize
      | NativeType::U64
      | NativeType::USize
  )
}

struct SysVAmd64 {
  assembler: dynasmrt::x64::Assembler,
  // As defined in section 3.2.3 of the SysV ABI spec, arguments are classified in the following classes:
  // - INTEGER:
  //    > Arguments of types (signed and unsigned) _Bool, char, short, int,
  //    > long, long long, and pointers are in the INTEGER class.
  // - SSE:
  //    > Arguments of types float, double, _Decimal32, _Decimal64 and
  //    > __m64 are in class SSE.
  //
  // See https://refspecs.linuxfoundation.org/elf/x86_64-abi-0.99.pdf
  integer_args: i32,
  sse_args: i32,

  allocated_stack: u16,
}

impl SysVAmd64 {
  // Integer arguments go to the following GPR, in order: rdi, rsi, rdx, rcx, r8, r9
  const INTEGER_REG: i32 = 6;
  // SSE arguments go to the first 8 SSE registers: xmm0-xmm7
  const SSE_REG: i32 = 8;

  fn new() -> Self {
    Self {
      assembler: dynasmrt::x64::Assembler::new().unwrap(),
      integer_args: 0,
      sse_args: 0,
      allocated_stack: 0,
    }
  }

  fn compile(sym: &Symbol) -> Trampoline {
    // TODO: Apple Silicon & windows x64 support
    let mut compiler = SysVAmd64::new();

    let can_tailcall = !compiler.must_cast_return_value(sym.result_type);
    if !can_tailcall {
      compiler.allocate_stack(&sym.parameter_types);
    }

    for argument in &sym.parameter_types {
      compiler.move_left(argument)
    }
    if !compiler.integer_args_have_moved() {
      // the receiver object should never be expected. Avoid its unexpected or deliverated leak
      compiler.zero_first_arg();
    }

    if !can_tailcall {
      compiler.call(sym.ptr.as_ptr());
      if compiler.must_cast_return_value(sym.result_type) {
        compiler.cast_return_value(sym.result_type);
      }
      compiler.deallocate_stack();
      compiler.ret();
    } else {
      compiler.tailcall(sym.ptr.as_ptr());
    }

    Trampoline(compiler.finalize())
  }

  fn move_left(&mut self, arg: &NativeType) {
    match arg {
      NativeType::F32 => self.move_sse(Single),
      NativeType::F64 => self.move_sse(Double),
      NativeType::U8 => self.move_integer(Unsigned(B)),
      NativeType::U16 => self.move_integer(Unsigned(W)),
      NativeType::U32 | NativeType::Void => self.move_integer(Unsigned(DW)),
      NativeType::U64
      | NativeType::USize
      | NativeType::Function
      | NativeType::Pointer => self.move_integer(Unsigned(QW)),
      NativeType::I8 => self.move_integer(Signed(B)),
      NativeType::I16 => self.move_integer(Signed(W)),
      NativeType::I32 => self.move_integer(Signed(DW)),
      NativeType::I64 | NativeType::ISize => self.move_integer(Signed(QW)),
    }
  }

  fn move_sse(&mut self, float: Float) {
    // Section 3.2.3 of the SysV AMD64 ABI:
    // > If the class is SSE, the next available vector register is used, the registers
    // > are taken in the order from %xmm0 to %xmm7.
    // [...]
    // > Once registers are assigned, the arguments passed in memory are pushed on
    // > the stack in reversed (right-to-left) order

    let arg_i = self.sse_args + 1;
    self.sse_args = arg_i;
    // floats are only moved to accomodate integer movement in the stack
    let is_in_stack = arg_i > Self::SSE_REG;
    let stack_has_moved =
      self.allocated_stack > 0 || self.integer_args >= Self::INTEGER_REG;

    if is_in_stack && stack_has_moved {
      // adding 1 to the integer amount to account for the receiver
      let pos_in_stack = (arg_i - Self::SSE_REG)
        + (1 + self.integer_args - Self::INTEGER_REG).max(0);
      let new_pos_in_stack = pos_in_stack - 1;

      let rsp_offset;
      let new_rsp_offset;

      if self.allocated_stack > 0 {
        rsp_offset = pos_in_stack * 8 + self.allocated_stack as i32;
        // creating a new stack frame for the to be called FFI function
        // substract 8 bytes because this new stack frame does not yet have return address
        new_rsp_offset = new_pos_in_stack * 8 - 8;
      } else {
        rsp_offset = pos_in_stack * 8;
        new_rsp_offset = new_pos_in_stack * 8;
      }

      debug_assert!(
        self.allocated_stack == 0
          || new_rsp_offset <= self.allocated_stack as i32
      );

      // SSE registers remain untouch. Only when the stack is modified, the floats in the stack need to be accomodated
      match float {
        Single => dynasm!(self.assembler
          ; movss xmm8, [rsp + rsp_offset]
          ; movss [rsp + new_rsp_offset], xmm8
        ),
        Double => dynasm!(self.assembler
          ; movsd xmm8, [rsp + rsp_offset]
          ; movsd [rsp + new_rsp_offset], xmm8
        ),
      }
    }
  }

  fn move_integer(&mut self, arg: Integer) {
    // Section 3.2.3 of the SysV AMD64 ABI:
    // > If the class is INTEGER, the next available register of the sequence %rdi,
    // > %rsi, %rdx, %rcx, %r8 and %r9 is used
    // [...]
    // > Once registers are assigned, the arguments passed in memory are pushed on
    // > the stack in reversed (right-to-left) order

    let arg_i = self.integer_args + 1;
    self.integer_args = arg_i;

    // adding 1 to the integer amount to account for the receiver
    let pos_in_stack =
      (1 + arg_i - Self::INTEGER_REG) + (self.sse_args - Self::SSE_REG).max(0);
    let new_pos_in_stack = pos_in_stack - 1;

    let rsp_offset;
    let new_rsp_offset;

    if self.allocated_stack > 0 {
      rsp_offset = pos_in_stack * 8 + self.allocated_stack as i32;
      // creating a new stack frame for the to be called FFI function
      // substract 8 bytes because this new stack frame does not yet have return address
      new_rsp_offset = new_pos_in_stack * 8 - 8;
    } else {
      rsp_offset = pos_in_stack * 8;
      new_rsp_offset = new_pos_in_stack * 8;
    }

    debug_assert!(
      self.allocated_stack == 0
        || new_rsp_offset <= self.allocated_stack as i32
    );

    // move each argument one position to the left. The first argument in the stack moves to the last register (r9).
    // If the FFI function is called with a new stack frame, the arguments remaining in the stack are copied to the new stack frame.
    // Otherwise, they are copied 8 bytes lower
    match (arg_i, arg) {
      // Conventionally, many compilers expect 8 and 16 bit arguments to be sign/zero extended to 32 bits
      // See https://stackoverflow.com/a/36760539/2623340
      (1, Unsigned(B)) => dynasm!(self.assembler; movzx edi, sil),
      (1, Signed(B)) => dynasm!(self.assembler; movsx edi, sil),
      (1, Unsigned(W)) => dynasm!(self.assembler; movzx edi, si),
      (1, Signed(W)) => dynasm!(self.assembler; movsx edi, si),
      (1, Unsigned(DW) | Signed(DW)) => dynasm!(self.assembler; mov edi, esi),
      (1, Unsigned(QW) | Signed(QW)) => dynasm!(self.assembler; mov rdi, rsi),

      (2, Unsigned(B)) => dynasm!(self.assembler; movzx esi, dl),
      (2, Signed(B)) => dynasm!(self.assembler; movsx esi, dl),
      (2, Unsigned(W)) => dynasm!(self.assembler; movzx esi, dx),
      (2, Signed(W)) => dynasm!(self.assembler; movsx esi, dx),
      (2, Unsigned(DW) | Signed(DW)) => dynasm!(self.assembler; mov esi, edx),
      (2, Unsigned(QW) | Signed(QW)) => dynasm!(self.assembler; mov rsi, rdx),

      (3, Unsigned(B)) => dynasm!(self.assembler; movzx edx, cl),
      (3, Signed(B)) => dynasm!(self.assembler; movsx edx, cl),
      (3, Unsigned(W)) => dynasm!(self.assembler; movzx edx, cx),
      (3, Signed(W)) => dynasm!(self.assembler; movsx edx, cx),
      (3, Unsigned(DW) | Signed(DW)) => dynasm!(self.assembler; mov edx, ecx),
      (3, Unsigned(QW) | Signed(QW)) => dynasm!(self.assembler; mov rdx, rcx),

      (4, Unsigned(B)) => dynasm!(self.assembler; movzx ecx, r8b),
      (4, Signed(B)) => dynasm!(self.assembler; movsx ecx, r8b),
      (4, Unsigned(W)) => dynasm!(self.assembler; movzx ecx, r8w),
      (4, Signed(W)) => dynasm!(self.assembler; movsx ecx, r8w),
      (4, Unsigned(DW) | Signed(DW)) => dynasm!(self.assembler; mov ecx, r8d),
      (4, Unsigned(QW) | Signed(QW)) => dynasm!(self.assembler; mov rcx, r8),

      (5, Unsigned(B)) => dynasm!(self.assembler; movzx r8d, r9b),
      (5, Signed(B)) => dynasm!(self.assembler; movsx r8d, r9b),
      (5, Unsigned(W)) => dynasm!(self.assembler; movzx r8d, r9w),
      (5, Signed(W)) => dynasm!(self.assembler; movsx r8d, r9w),
      (5, Unsigned(DW) | Signed(DW)) => dynasm!(self.assembler; mov r8d, r9d),
      (5, Unsigned(QW) | Signed(QW)) => dynasm!(self.assembler; mov r8, r9),

      (6, Unsigned(B)) => {
        dynasm!(self.assembler; movzx r9d, BYTE [rsp + rsp_offset])
      }
      (6, Signed(B)) => {
        dynasm!(self.assembler; movsx r9d, BYTE [rsp + rsp_offset])
      }
      (6, Unsigned(W)) => {
        dynasm!(self.assembler; movzx r9d, WORD [rsp + rsp_offset])
      }
      (6, Signed(W)) => {
        dynasm!(self.assembler; movsx r9d, WORD [rsp + rsp_offset])
      }
      (6, Unsigned(DW) | Signed(DW)) => {
        dynasm!(self.assembler; mov r9d, [rsp + rsp_offset])
      }
      (6, Unsigned(QW) | Signed(QW)) => {
        dynasm!(self.assembler; mov r9, [rsp + rsp_offset])
      }

      (_, Unsigned(B)) => dynasm!(self.assembler
        ; movzx eax, BYTE [rsp + rsp_offset]
        ; mov [rsp + new_rsp_offset], eax
      ),
      (_, Signed(B)) => dynasm!(self.assembler
        ; movsx eax, BYTE [rsp + rsp_offset]
        ; mov [rsp + new_rsp_offset], eax
      ),
      (_, Unsigned(W)) => dynasm!(self.assembler
        ; movzx eax, WORD [rsp + rsp_offset]
        ; mov [rsp + new_rsp_offset], eax
      ),
      (_, Signed(W)) => dynasm!(self.assembler
        ; movsx eax, WORD [rsp + rsp_offset]
        ; mov [rsp + new_rsp_offset], eax
      ),
      (_, Unsigned(DW) | Signed(DW)) => dynasm!(self.assembler
        ; mov eax, [rsp + rsp_offset]
        ; mov [rsp + new_rsp_offset], eax
      ),
      (_, Unsigned(QW) | Signed(QW)) => dynasm!(self.assembler
        ; mov rax, [rsp + rsp_offset]
        ; mov [rsp + new_rsp_offset], rax
      ),
    }
  }

  fn zero_first_arg(&mut self) {
    dynasm!(self.assembler
      ; xor rdi, rdi
    );
  }

  fn cast_return_value(&mut self, rv: NativeType) {
    // 8 and 16 bit integers are extended to 32 bits
    match rv {
      NativeType::U8 => dynasm!(self.assembler; movzx eax, al),
      NativeType::I8 => dynasm!(self.assembler; movsx eax, al),
      NativeType::U16 => dynasm!(self.assembler; movzx eax, ax),
      NativeType::I16 => dynasm!(self.assembler; movsx eax, ax),
      _ => (),
    }
  }

  fn allocate_stack(&mut self, params: &[NativeType]) {
    let mut stack_size = {
      let mut int = -Self::INTEGER_REG;
      let mut sse = -Self::SSE_REG;
      for param in params {
        match param {
          NativeType::F32 | NativeType::F64 => sse += 1,
          _ => int += 1,
        }
      }
      ((int.max(0) + sse.max(0)) * 8) as u16
    };

    // Align stack frame
    if (stack_size + 8) % 16 != 0 {
      // Section 3.2.2 of the SysV AMD64 ABI:
      // > The end of the input argument area shall be aligned on a 16 (32 or 64, if
      // > __m256 or __m512 is passed on stack) byte boundary. In other words, the value
      // > (%rsp + 8) is always a multiple of 16 (32 or 64) when control is transferred to
      // > the function entry point. The stack pointer, %rsp, always points to the end of the
      // > latest allocated stack frame.
      stack_size += 8;
    }

    dynasm!(self.assembler
      ; sub rsp, stack_size as i32
    );
    self.allocated_stack = stack_size;
  }

  fn deallocate_stack(&mut self) {
    dynasm!(self.assembler
      ; add rsp, self.allocated_stack as i32
    );
  }

  fn call(&mut self, ptr: *const c_void) {
    // the stack has been aligned during stack allocation
    dynasm!(self.assembler
      ; mov rax, QWORD ptr as _
      ; call rax
    );
  }

  fn tailcall(&mut self, ptr: *const c_void) {
    // stack pointer is never modified and remains aligned
    // return address remains the one provided by the trampoline's caller (V8)
    dynasm!(self.assembler
      ; mov rax, QWORD ptr as _
      ; jmp rax
    );
  }

  fn ret(&mut self) {
    // the stack has been deallocated before ret is called
    dynasm!(self.assembler
      ; ret
    );
  }

  fn integer_args_have_moved(&self) -> bool {
    self.integer_args > 0
  }

  fn must_cast_return_value(&self, rv: NativeType) -> bool {
    // V8 only supports i32 and u32 return types for integers
    // We support 8 and 16 bit integers by extending them to 32 bits in the trampoline before returning
    matches!(
      rv,
      NativeType::U8 | NativeType::I8 | NativeType::U16 | NativeType::I16
    )
  }

  fn finalize(self) -> ExecutableBuffer {
    self.assembler.finalize().unwrap()
  }
}

#[cfg(target_arch = "aarch64")]
struct Aarch64 {
  assembler: dynasmrt::aarch64::Assembler,
  // As defined in section 6.4.2 of the Aarch64 Procedure Call Standard (PCS) spec, arguments are classified as follows:
  // - INTEGRAL or POINTER:
  //    > If the argument is an Integral or Pointer Type, the size of the argument is less than or equal to 8 bytes
  //    > and the NGRN is less than 8, the argument is copied to the least significant bits in x[NGRN].
  //
  // - Floating-Point or Vector:
  //    > If the argument is a Half-, Single-, Double- or Quad- precision Floating-point or short vector type
  //    > and the NSRN is less than 8, then the argument is allocated to the least significant bits of register v[NSRN]
  //
  // See https://github.com/ARM-software/abi-aa/blob/60a8eb8c55e999d74dac5e368fc9d7e36e38dda4/aapcs64/aapcs64.rst#642parameter-passing-rules
  // counters
  integer_args: i32,
  float_args: i32,

  stack_trampoline: u16,
  stack_original: u16,

  stack_allocated: u16,
}

#[cfg(target_arch = "aarch64")]
impl Aarch64 {
  // Integer arguments go to the first 8 GPR: x0-x7
  const INTEGER_REG: i32 = 8;
  // Floating-point arguments go to the first 8 SIMD & Floating-Point registers: v0-v1
  const FLOAT_REG: i32 = 8;

  fn new() -> Self {
    Self {
      assembler: dynasmrt::aarch64::Assembler::new().unwrap(),
      integer_args: 0,
      float_args: 0,
      stack_allocated: 0,
      stack_trampoline: 0,
      stack_original: 0,
    }
  }

  fn compile(sym: &Symbol) -> Trampoline {
    // TODO: Apple Silicon & windows x64 support
    let mut compiler = SysVAmd64::new();

    let can_tailcall = !compiler.must_cast_return_value(sym.result_type);
    if !can_tailcall {
      compiler.allocate_stack(&sym.parameter_types);
    }

    for argument in &sym.parameter_types {
      compiler.move_left(argument)
    }
    if !compiler.integer_args_have_moved() {
      // the receiver object should never be expected. Avoid its unexpected or deliverated leak
      compiler.zero_first_arg();
    }

    if !can_tailcall {
      compiler.call(sym.ptr.as_ptr());
      if compiler.must_cast_return_value(sym.result_type) {
        compiler.cast_return_value(sym.result_type);
      }
      compiler.deallocate_stack();
      compiler.ret();
    } else {
      compiler.tailcall(sym.ptr.as_ptr());
    }

    Trampoline(compiler.finalize())
  }

  fn move_left(&mut self, arg: &NativeType) {
    match arg {
      NativeType::F32 => self.move_float(Single),
      NativeType::F64 => self.move_float(Double),
      NativeType::U8 => self.move_integer(Unsigned(B)),
      NativeType::U16 => self.move_integer(Unsigned(W)),
      NativeType::U32 | NativeType::Void => self.move_integer(Unsigned(DW)),
      NativeType::U64
      | NativeType::USize
      | NativeType::Function
      | NativeType::Pointer => self.move_integer(Unsigned(QW)),
      NativeType::I8 => self.move_integer(Signed(B)),
      NativeType::I16 => self.move_integer(Signed(W)),
      NativeType::I32 => self.move_integer(Signed(DW)),
      NativeType::I64 | NativeType::ISize => self.move_integer(Signed(QW)),
    }
  }

  fn move_float(&mut self, float: Float) {
    // Section 3.2.3 of the SysV AMD64 ABI:
    // > If the class is SSE, the next available vector register is used, the registers
    // > are taken in the order from %xmm0 to %xmm7.
    // [...]
    // > Once registers are assigned, the arguments passed in memory are pushed on
    // > the stack in reversed (right-to-left) order

    let arg_i = self.float_args + 1;
    self.float_args = arg_i;

    let is_in_stack = arg_i > Self::FLOAT_REG;
    if is_in_stack {
      // 6.4.2 Aarch64 PCS:
      // > The NSAA [Next Stacked Argument Address] is rounded up to the larger of 8 or the Natural Alignment of the argument’s type.
      // > The argument is copied to memory at the adjusted NSAA. The NSAA is incremented by the size of the argument.
      //
      // https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms:
      // > Function arguments may consume slots on the stack that are not multiples of 8 bytes.
      // > If the total number of bytes for stack-based arguments is not a multiple of 8 bytes,
      // > insert padding on the stack to maintain the 8-byte alignment requirements.
      #[cfg(target_vendor = "apple")]
      let size = float as u16;
      #[cfg(not(target_vendor = "apple"))]
      let size = (float as u16).max(8);
      // The trampoline and the orignal function always have the same amount of floats in the stack
      self.stack_trampoline += size;
      self.stack_original += size;
    }
    // floats are only moved to accomodate integer movement in the stack
    let stack_has_moved =
      self.stack_allocated > 0 || self.integer_args >= Self::INTEGER_REG;

    if is_in_stack && stack_has_moved {
      let rsp_offset;
      let new_rsp_offset;
      if self.stack_allocated > 0 {
        rsp_offset = self.stack_trampoline as u32 + self.stack_allocated as u32;
        new_rsp_offset = self.stack_original as u32;
      } else {
        rsp_offset = self.stack_trampoline as u32;
        new_rsp_offset = self.stack_original as u32;
      }

      debug_assert!(
        self.stack_allocated == 0
          || new_rsp_offset <= self.stack_allocated as u32
      );

      match float {
        Single => dynasm!(self.assembler
          ; .arch aarch64
          // 6.1.2 Aarch64 PCS:
          // > Registers v8-v15 must be preserved by a callee across subroutine calls;
          // > the remaining registers (v0-v7, v16-v31) do not need to be preserved (or should be preserved by the caller).
          ; ldr s16, [sp, rsp_offset]
          ; str s16, [sp, new_rsp_offset]
        ),
        Double => dynasm!(self.assembler
          ; .arch aarch64
          ; ldr d16, [sp, rsp_offset]
          ; str d16, [sp, new_rsp_offset]
        ),
      }
    }
  }

  fn move_integer(&mut self, arg: Integer) {
    // > If the argument is an Integral or Pointer Type, the size of the argument is less than or equal to 8 bytes and the NGRN is less than 8,
    // > the argument is copied to the least significant bits in x[NGRN]. The NGRN is incremented by one. The argument has now been allocated.
    // > [if NGRN is equal or more than 8]
    // > The argument is copied to memory at the adjusted NSAA. The NSAA is incremented by the size of the argument. The argument has now been allocated.

    let arg_i = self.integer_args + 1;
    self.integer_args = arg_i;

    let (Unsigned(size) | Signed(size)) = arg;

    // 6.4.2 Aarch64 PCS:
    // > The NSAA [Next Stacked Argument Address] is rounded up to the larger of 8 or the Natural Alignment of the argument’s type.
    // > The argument is copied to memory at the adjusted NSAA. The NSAA is incremented by the size of the argument.
    //
    // https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms:
    // > Function arguments may consume slots on the stack that are not multiples of 8 bytes.
    // > If the total number of bytes for stack-based arguments is not a multiple of 8 bytes,
    // > insert padding on the stack to maintain the 8-byte alignment requirements.
    #[cfg(target_vendor = "apple")]
    let bytes = size as u16;
    #[cfg(not(target_vendor = "apple"))]
    let bytes = (size as u16).max(8);

    // move each argument one position to the left. The first argument in the stack moves to the last register.
    // If the FFI function is called with a new stack frame, the arguments remaining in the stack are copied to the new stack frame.
    // Otherwise, they are copied 8 bytes lower
    match (arg_i, arg) {
      // From https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms:
      // > The caller of a function is responsible for signing or zero-extending any argument with fewer than 32 bits.
      // > The standard ABI expects the callee to sign or zero-extend those arguments.
      // (this applies to register parameters, as stack parameters are not padded in Apple)
      (1, Signed(B)) => dynasm!(self.assembler; .arch aarch64; sxtb w0, w1),
      (1, Unsigned(B)) => {
        dynasm!(self.assembler; .arch aarch64; and w0, w1, 0xFF)
      }
      (1, Signed(W)) => dynasm!(self.assembler; .arch aarch64; sxth w0, w1),
      (1, Unsigned(W)) => {
        dynasm!(self.assembler; .arch aarch64; and w0, w1, 0xFFFF)
      }
      (1, Signed(DW) | Unsigned(DW)) => {
        dynasm!(self.assembler; .arch aarch64; mov w0, w1)
      }
      (1, Signed(QW) | Unsigned(QW)) => {
        dynasm!(self.assembler; .arch aarch64; mov x0, x1)
      }

      (2, Signed(B)) => dynasm!(self.assembler; .arch aarch64; sxtb w1, w2),
      (2, Unsigned(B)) => {
        dynasm!(self.assembler; .arch aarch64; and w1, w2, 0xFF)
      }
      (2, Signed(W)) => dynasm!(self.assembler; .arch aarch64; sxth w1, w2),
      (2, Unsigned(W)) => {
        dynasm!(self.assembler; .arch aarch64; and w1, w2, 0xFFFF)
      }
      (2, Signed(DW) | Unsigned(DW)) => {
        dynasm!(self.assembler; .arch aarch64; mov w1, w2)
      }
      (2, Signed(QW) | Unsigned(QW)) => {
        dynasm!(self.assembler; .arch aarch64; mov x1, x2)
      }

      (3, Signed(B)) => dynasm!(self.assembler; .arch aarch64; sxtb w2, w3),
      (3, Unsigned(B)) => {
        dynasm!(self.assembler; .arch aarch64; and w2, w3, 0xFF)
      }
      (3, Signed(W)) => dynasm!(self.assembler; .arch aarch64; sxth w2, w3),
      (3, Unsigned(W)) => {
        dynasm!(self.assembler; .arch aarch64; and w2, w3, 0xFFFF)
      }
      (3, Signed(DW) | Unsigned(DW)) => {
        dynasm!(self.assembler; .arch aarch64; mov w2, w3)
      }
      (3, Signed(QW) | Unsigned(QW)) => {
        dynasm!(self.assembler; .arch aarch64; mov x2, x3)
      }

      (4, Signed(B)) => dynasm!(self.assembler; .arch aarch64; sxtb w3, w4),
      (4, Unsigned(B)) => {
        dynasm!(self.assembler; .arch aarch64; and w3, w4, 0xFF)
      }
      (4, Signed(W)) => dynasm!(self.assembler; .arch aarch64; sxth w3, w4),
      (4, Unsigned(W)) => {
        dynasm!(self.assembler; .arch aarch64; and w3, w4, 0xFFFF)
      }
      (4, Signed(DW) | Unsigned(DW)) => {
        dynasm!(self.assembler; .arch aarch64; mov w3, w4)
      }
      (4, Signed(QW) | Unsigned(QW)) => {
        dynasm!(self.assembler; .arch aarch64; mov x3, x4)
      }

      (5, Signed(B)) => dynasm!(self.assembler; .arch aarch64; sxtb w4, w5),
      (5, Unsigned(B)) => {
        dynasm!(self.assembler; .arch aarch64; and w4, w5, 0xFF)
      }
      (5, Signed(W)) => dynasm!(self.assembler; .arch aarch64; sxth w4, w5),
      (5, Unsigned(W)) => {
        dynasm!(self.assembler; .arch aarch64; and w4, w5, 0xFFFF)
      }
      (5, Signed(DW) | Unsigned(DW)) => {
        dynasm!(self.assembler; .arch aarch64; mov w4, w5)
      }
      (5, Signed(QW) | Unsigned(QW)) => {
        dynasm!(self.assembler; .arch aarch64; mov x4, x5)
      }

      (6, Signed(B)) => dynasm!(self.assembler; .arch aarch64; sxtb w5, w6),
      (6, Unsigned(B)) => {
        dynasm!(self.assembler; .arch aarch64; and w5, w6, 0xFF)
      }
      (6, Signed(W)) => dynasm!(self.assembler; .arch aarch64; sxth w5, w6),
      (6, Unsigned(W)) => {
        dynasm!(self.assembler; .arch aarch64; and w5, w6, 0xFFFF)
      }
      (6, Signed(DW) | Unsigned(DW)) => {
        dynasm!(self.assembler; .arch aarch64; mov w5, w6)
      }
      (6, Signed(QW) | Unsigned(QW)) => {
        dynasm!(self.assembler; .arch aarch64; mov x5, x6)
      }

      (7, Signed(B)) => dynasm!(self.assembler; .arch aarch64; sxtb w6, w7),
      (7, Unsigned(B)) => {
        dynasm!(self.assembler; .arch aarch64; and w6, w7, 0xFF)
      }
      (8, Signed(W)) => dynasm!(self.assembler; .arch aarch64; sxth w5, w7),
      (8, Unsigned(W)) => {
        dynasm!(self.assembler; .arch aarch64; and w5, w7, 0xFFFF)
      }
      (7, Signed(DW) | Unsigned(DW)) => {
        dynasm!(self.assembler; .arch aarch64; mov w6, w7)
      }
      (7, Signed(QW) | Unsigned(QW)) => {
        dynasm!(self.assembler; .arch aarch64; mov x6, x7)
      }

      (8, arg) => {
        self.stack_trampoline += bytes;
        match arg {
          Signed(B) => {
            dynasm!(self.assembler; .arch aarch64; ldrsb w7, [sp, self.input_sp_offset()])
          }
          Unsigned(B) => {
            dynasm!(self.assembler; .arch aarch64; ldrb w7, [sp, self.input_sp_offset()])
          }
          Signed(W) => {
            dynasm!(self.assembler; .arch aarch64; ldrsh w7, [sp, self.input_sp_offset()])
          }
          Unsigned(W) => {
            dynasm!(self.assembler; .arch aarch64; ldrh w7, [sp, self.input_sp_offset()])
          }
          Signed(DW) | Unsigned(DW) => {
            dynasm!(self.assembler; .arch aarch64; ldr w7, [sp, self.input_sp_offset()])
          }
          Signed(QW) | Unsigned(QW) => {
            dynasm!(self.assembler; .arch aarch64; ldr x7, [sp, self.input_sp_offset()])
          }
        }
      }

      (_, arg) => {
        self.stack_trampoline += bytes;
        self.stack_original += bytes;
        match arg {
          Signed(B) | Unsigned(B) => dynasm!(self.assembler
            ; .arch aarch64
            ; ldrb w8, [sp, self.input_sp_offset()]
            ; strb w8, [sp, self.output_sp_offset()]
          ),
          Signed(W) | Unsigned(W) => dynasm!(self.assembler
            ; .arch aarch64
            ; ldrh w8, [sp, self.input_sp_offset()]
            ; strh w8, [sp, self.output_sp_offset()]
          ),
          Signed(DW) | Unsigned(DW) => dynasm!(self.assembler
            ; .arch aarch64
            ;  ldr w8, [sp, self.input_sp_offset()]
            ; str w8, [sp, self.output_sp_offset()]
          ),
          Signed(QW) | Unsigned(QW) => dynasm!(self.assembler
            ; .arch aarch64
            ; ldr x8, [sp, self.input_sp_offset()]
            ; str x8, [sp, self.output_sp_offset()]
          ),
        }
      }
    };

    debug_assert!(
      self.stack_allocated == 0
        || self.output_sp_offset() <= self.stack_allocated as u32 - 16
    );
  }

  fn input_sp_offset(&self) -> u32 {
    (if self.stack_allocated > 0 {
      self.stack_trampoline + self.stack_allocated
    } else {
      self.stack_trampoline
    }) as u32
  }

  fn output_sp_offset(&self) -> u32 {
    self.stack_original as u32
  }

  fn zero_first_arg(&mut self) {
    dynasm!(self.assembler
      ; .arch aarch64
      ; mov x0, xzr
    );
  }

  fn cast_return_value(&mut self, rv: NativeType) {
    // return values follow the same rules as register arguments. Therefore, in Apple the RV is sign/zero extended by the callee,
    // whereas standard ARM64 dictates the upper bits are undefined.
    // At the time of writing Rust has an outstanding bug when targetting aarch64-unknown-linux-gnu,
    // where integer return values smaller than 32 bits are assumed to be sign extended by the callee:
    // https://github.com/rust-lang/rust/issues/97463
    #[cfg(not(target_vendor = "apple"))]
    match rv {
      NativeType::U8 => {
        dynasm!(self.assembler; .arch aarch64; and w0, w0, 0xFF)
      }
      NativeType::I8 => dynasm!(self.assembler; .arch aarch64; sxtb w0, w0),
      NativeType::U16 => {
        dynasm!(self.assembler; .arch aarch64; and w0, w0, 0xFFFF)
      }
      NativeType::I16 => dynasm!(self.assembler; .arch aarch64; sxth w0, w0),
      _ => (),
    }
  }

  fn allocate_stack(&mut self, params: &[NativeType]) {
    let mut stack_size = 0u32;
    let mut int = -Self::INTEGER_REG;
    let mut sse = -Self::FLOAT_REG;
    #[cfg(not(target_vendor = "apple"))]
    {
      for param in params {
        match param {
          NativeType::F32 | NativeType::F64 => sse += 1,
          _ => int += 1,
        }
      }
      stack_size = ((int.max(0) + sse.max(0)) * 8) as u32
    }
    #[cfg(target_vendor = "apple")]
    for param in params {
      match param {
        NativeType::F32 => {
          sse += 1;
          if sse > 0 {
            stack_size += 4;
          }
        }
        NativeType::F64 => {
          sse += 1;
          if sse > 0 {
            stack_size += 8;
          }
        }
        NativeType::I8 | NativeType::U8 => {
          int += 1;
          if int > 0 {
            stack_size += 1;
          }
        }
        NativeType::I16 | NativeType::U16 => {
          int += 1;
          if int > 0 {
            stack_size += 2;
          }
        }
        NativeType::I32 | NativeType::U32 => {
          int += 1;
          if int > 0 {
            stack_size += 4;
          }
        }
        NativeType::I64 | NativeType::U64 => {
          int += 1;
          if int > 0 {
            stack_size += 8;
          }
        }
      }
    }

    // 6.2.3 Aarch64 PCS:
    // > Each frame shall link to the frame of its caller by means of a frame record
    // > of two 64-bit values on the stack (independent of the data model).
    stack_size += 16;

    // Align stack frame
    // 6.2.2 Aarch PCS:
    // > at any point at which memory is accessed via SP, the hardware requires that:
    // > SP mod 16 = 0. The stack must be quad-word aligned.
    stack_size += stack_size % 16;

    dynasm!(self.assembler
      ; .arch aarch64
      ; sub sp, sp, stack_size
      // 6.2.3 Aarch64 PCS:
      // > The frame record for the innermost frame (belonging to the most recent routine invocation)
      // > shall be pointed to by the frame pointer register (FP [x29]). The lowest addressed double-word shall point
      // > to the previous frame record and the highest addressed double-word shall contain the value passed
      // > in LR [x30] on entry to the current function.
      ; stp x29, x30, [sp, stack_size - 16]
      ; add x29, sp, stack_size - 16
    );
    self.stack_allocated = stack_size as u16;
  }

  fn deallocate_stack(&mut self) {
    dynasm!(self.assembler
      ; .arch aarch64
      ; add sp, sp, self.stack_allocated as u32
    );
  }

  fn call(&mut self, ptr: *const c_void) {
    // the stack has been aligned during stack allocation
    dynasm!(self.assembler
      ; .arch aarch64
      ; bl ptr as u32
    );
  }

  fn tailcall(&mut self, ptr: *const c_void) {
    // stack pointer is never modified and remains aligned
    // frame pointer remains the one provided by the trampoline's caller (V8)
    dynasm!(self.assembler
      ; .arch aarch64
      ; b ptr as u32
    );
  }

  fn ret(&mut self) {
    // the stack has been deallocated before ret is called
    dynasm!(self.assembler
      ; .arch aarch64
      ; ret
    );
  }

  fn integer_args_have_moved(&self) -> bool {
    self.integer_args > 0
  }

  fn must_cast_return_value(&self, rv: NativeType) -> bool {
    // V8 only supports i32 and u32 return types for integers
    // We support 8 and 16 bit integers by extending them to 32 bits in the trampoline before returning
    matches!(
      rv,
      NativeType::U8 | NativeType::I8 | NativeType::U16 | NativeType::I16
    )
  }

  fn finalize(self) -> ExecutableBuffer {
    self.assembler.finalize().unwrap()
  }
}

#[derive(Clone, Copy, Debug)]
enum Float {
  Single = 4,
  Double = 8,
}
use Float::*;

#[derive(Clone, Copy, Debug)]
enum Integer {
  Signed(Size),
  Unsigned(Size),
}
use Integer::*;

// TODO: aarch64 uses B, H, W, D
#[derive(Clone, Copy, Debug)]
enum Size {
  B = 1,
  W = 2,
  DW = 4,
  QW = 8,
}
use Size::*;
