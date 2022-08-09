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
  cfg!(any(
    all(target_arch = "x86_64", target_family = "unix"),
    all(target_arch = "aarch64", target_vendor = "apple")
  )) && !sym.can_callback
    && is_fast_api_rv(sym.result_type)
}

pub(crate) fn compile_trampoline(sym: &Symbol) -> Trampoline {
  #[cfg(all(target_arch = "x86_64", target_family = "unix"))]
  return SysVAmd64::compile(sym);
  #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
  return Aarch64Apple::compile(sym);
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
          ; .arch x64
          ; movss xmm8, [rsp + rsp_offset]
          ; movss [rsp + new_rsp_offset], xmm8
        ),
        Double => dynasm!(self.assembler
          ; .arch x64
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
      (1, Unsigned(B)) => dynasm!(self.assembler; .arch x64; movzx edi, sil),
      (1, Signed(B)) => dynasm!(self.assembler; .arch x64; movsx edi, sil),
      (1, Unsigned(W)) => dynasm!(self.assembler; .arch x64; movzx edi, si),
      (1, Signed(W)) => dynasm!(self.assembler; .arch x64; movsx edi, si),
      (1, Unsigned(DW) | Signed(DW)) => {
        dynasm!(self.assembler; .arch x64; mov edi, esi)
      }
      (1, Unsigned(QW) | Signed(QW)) => {
        dynasm!(self.assembler; .arch x64; mov rdi, rsi)
      }

      (2, Unsigned(B)) => dynasm!(self.assembler; .arch x64; movzx esi, dl),
      (2, Signed(B)) => dynasm!(self.assembler; .arch x64; movsx esi, dl),
      (2, Unsigned(W)) => dynasm!(self.assembler; .arch x64; movzx esi, dx),
      (2, Signed(W)) => dynasm!(self.assembler; .arch x64; movsx esi, dx),
      (2, Unsigned(DW) | Signed(DW)) => {
        dynasm!(self.assembler; .arch x64; mov esi, edx)
      }
      (2, Unsigned(QW) | Signed(QW)) => {
        dynasm!(self.assembler; .arch x64; mov rsi, rdx)
      }

      (3, Unsigned(B)) => dynasm!(self.assembler; .arch x64; movzx edx, cl),
      (3, Signed(B)) => dynasm!(self.assembler; .arch x64; movsx edx, cl),
      (3, Unsigned(W)) => dynasm!(self.assembler; .arch x64; movzx edx, cx),
      (3, Signed(W)) => dynasm!(self.assembler; .arch x64; movsx edx, cx),
      (3, Unsigned(DW) | Signed(DW)) => {
        dynasm!(self.assembler; .arch x64; mov edx, ecx)
      }
      (3, Unsigned(QW) | Signed(QW)) => {
        dynasm!(self.assembler; .arch x64; mov rdx, rcx)
      }

      (4, Unsigned(B)) => dynasm!(self.assembler; .arch x64; movzx ecx, r8b),
      (4, Signed(B)) => dynasm!(self.assembler; .arch x64; movsx ecx, r8b),
      (4, Unsigned(W)) => dynasm!(self.assembler; .arch x64; movzx ecx, r8w),
      (4, Signed(W)) => dynasm!(self.assembler; .arch x64; movsx ecx, r8w),
      (4, Unsigned(DW) | Signed(DW)) => {
        dynasm!(self.assembler; .arch x64; mov ecx, r8d)
      }
      (4, Unsigned(QW) | Signed(QW)) => {
        dynasm!(self.assembler; .arch x64; mov rcx, r8)
      }

      (5, Unsigned(B)) => dynasm!(self.assembler; .arch x64; movzx r8d, r9b),
      (5, Signed(B)) => dynasm!(self.assembler; .arch x64; movsx r8d, r9b),
      (5, Unsigned(W)) => dynasm!(self.assembler; .arch x64; movzx r8d, r9w),
      (5, Signed(W)) => dynasm!(self.assembler; .arch x64; movsx r8d, r9w),
      (5, Unsigned(DW) | Signed(DW)) => {
        dynasm!(self.assembler; .arch x64; mov r8d, r9d)
      }
      (5, Unsigned(QW) | Signed(QW)) => {
        dynasm!(self.assembler; .arch x64; mov r8, r9)
      }

      (6, Unsigned(B)) => {
        dynasm!(self.assembler; .arch x64; movzx r9d, BYTE [rsp + rsp_offset])
      }
      (6, Signed(B)) => {
        dynasm!(self.assembler; .arch x64; movsx r9d, BYTE [rsp + rsp_offset])
      }
      (6, Unsigned(W)) => {
        dynasm!(self.assembler; .arch x64; movzx r9d, WORD [rsp + rsp_offset])
      }
      (6, Signed(W)) => {
        dynasm!(self.assembler; .arch x64; movsx r9d, WORD [rsp + rsp_offset])
      }
      (6, Unsigned(DW) | Signed(DW)) => {
        dynasm!(self.assembler; .arch x64; mov r9d, [rsp + rsp_offset])
      }
      (6, Unsigned(QW) | Signed(QW)) => {
        dynasm!(self.assembler; .arch x64; mov r9, [rsp + rsp_offset])
      }

      (_, Unsigned(B)) => dynasm!(self.assembler
        ; .arch x64
        ; movzx eax, BYTE [rsp + rsp_offset]
        ; mov [rsp + new_rsp_offset], eax
      ),
      (_, Signed(B)) => dynasm!(self.assembler
        ; .arch x64
        ; movsx eax, BYTE [rsp + rsp_offset]
        ; mov [rsp + new_rsp_offset], eax
      ),
      (_, Unsigned(W)) => dynasm!(self.assembler
        ; .arch x64
        ; movzx eax, WORD [rsp + rsp_offset]
        ; mov [rsp + new_rsp_offset], eax
      ),
      (_, Signed(W)) => dynasm!(self.assembler
        ; .arch x64
        ; movsx eax, WORD [rsp + rsp_offset]
        ; mov [rsp + new_rsp_offset], eax
      ),
      (_, Unsigned(DW) | Signed(DW)) => dynasm!(self.assembler
        ; .arch x64
        ; mov eax, [rsp + rsp_offset]
        ; mov [rsp + new_rsp_offset], eax
      ),
      (_, Unsigned(QW) | Signed(QW)) => dynasm!(self.assembler
        ; .arch x64
        ; mov rax, [rsp + rsp_offset]
        ; mov [rsp + new_rsp_offset], rax
      ),
    }
  }

  fn zero_first_arg(&mut self) {
    dynasm!(self.assembler
      ; .arch x64
      ; xor edi, edi
    );
  }

  fn cast_return_value(&mut self, rv: NativeType) {
    // 8 and 16 bit integers are extended to 32 bits
    match rv {
      NativeType::U8 => dynasm!(self.assembler; .arch x64; movzx eax, al),
      NativeType::I8 => dynasm!(self.assembler; .arch x64; movsx eax, al),
      NativeType::U16 => dynasm!(self.assembler; .arch x64; movzx eax, ax),
      NativeType::I16 => dynasm!(self.assembler; .arch x64; movsx eax, ax),
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
      ; .arch x64
      ; sub rsp, stack_size as i32
    );
    self.allocated_stack = stack_size;
  }

  fn deallocate_stack(&mut self) {
    dynasm!(self.assembler
      ; .arch x64
      ; add rsp, self.allocated_stack as i32
    );
  }

  fn call(&mut self, ptr: *const c_void) {
    // the stack has been aligned during stack allocation
    dynasm!(self.assembler
      ; .arch x64
      ; mov rax, QWORD ptr as _
      ; call rax
    );
  }

  fn tailcall(&mut self, ptr: *const c_void) {
    // stack pointer is never modified and remains aligned
    // return address remains the one provided by the trampoline's caller (V8)
    dynasm!(self.assembler
      ; .arch x64
      ; mov rax, QWORD ptr as _
      ; jmp rax
    );
  }

  fn ret(&mut self) {
    // the stack has been deallocated before ret is called
    dynasm!(self.assembler
      ; .arch x64
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

struct Aarch64Apple {
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
  integer_args: u32,
  float_args: u32,

  stack_trampoline: u32,
  stack_original: u32,

  stack_allocated: u32,
}

// trait Abi {
//   const INTEGER_REG: u32;
//   const FLOAT_REG: u32;

//   fn compile(sym: &Symbol) -> Trampoline where Self: Default {
//     let mut compiler = Self::default();

//     let can_tailcall = !compiler.must_cast_return_value(sym.result_type);
//     if !can_tailcall {
//       compiler.allocate_stack(&sym.parameter_types);
//     }

//     for argument in &sym.parameter_types {
//       compiler.move_left(argument)
//     }
//     if !compiler.integer_args_have_moved() {
//       // the receiver object should never be expected. Avoid its unexpected or deliverated leak
//       compiler.zero_first_arg();
//     }

//     if !can_tailcall {
//       compiler.call(sym.ptr.as_ptr());
//       if compiler.must_cast_return_value(sym.result_type) {
//         compiler.cast_return_value(sym.result_type);
//       }
//       compiler.deallocate_stack();
//       compiler.ret();
//     } else {
//       compiler.tailcall(sym.ptr.as_ptr());
//     }

//     Trampoline(compiler.finalize())
//   }

//   fn move_left(&mut self, arg: &NativeType) {
//     match arg {
//       NativeType::F32 => self.move_float(Single),
//       NativeType::F64 => self.move_float(Double),
//       NativeType::U8 => self.move_integer(Unsigned(B)),
//       NativeType::U16 => self.move_integer(Unsigned(W)),
//       NativeType::U32 | NativeType::Void => self.move_integer(Unsigned(DW)),
//       NativeType::U64
//       | NativeType::USize
//       | NativeType::Function
//       | NativeType::Pointer => self.move_integer(Unsigned(QW)),
//       NativeType::I8 => self.move_integer(Signed(B)),
//       NativeType::I16 => self.move_integer(Signed(W)),
//       NativeType::I32 => self.move_integer(Signed(DW)),
//       NativeType::I64 | NativeType::ISize => self.move_integer(Signed(QW)),
//     }
//   }

//   fn move_float(&mut self, arg_i: usize, arg: Float) {
//     let is_in_stack = arg_i > Self::FLOAT_REG;

//     if is_in_stack {
//       let size = arg as u32;
//       let padding_trampoline = (size - self.stack_trampoline % size) % size;
//       let padding_original = (size - self.stack_original % size) % size;

//       println!(
//         "input offset: {}",
//         self.stack_trampoline + padding_trampoline
//       );
//       println!("output offset: {}", self.stack_original + padding_original);
//       // floats are only moved to accomodate integer movement in the stack
//       let stack_has_moved =
//         self.stack_allocated > 0 || self.integer_args >= Self::INTEGER_REG;
//       if stack_has_moved {
//         match arg {
//           Single => dynasm!(self.assembler
//             ; .arch aarch64
//             // 6.1.2 Aarch64 PCS:
//             // > Registers v8-v15 must be preserved by a callee across subroutine calls;
//             // > the remaining registers (v0-v7, v16-v31) do not need to be preserved (or should be preserved by the caller).
//             ; ldr s16, [sp, self.stack_trampoline + padding_trampoline]
//             ; str s16, [sp, self.stack_original + padding_original]
//           ),
//           Double => dynasm!(self.assembler
//             ; .arch aarch64
//             ; ldr d16, [sp, self.stack_trampoline + padding_trampoline]
//             ; str d16, [sp, self.stack_original + padding_original]
//           ),
//         }
//       }
//       // The trampoline and the orignal function always have the same amount of floats in the stack
//       self.stack_trampoline += size + padding_trampoline;
//       self.stack_original += size + padding_original;
//     }
//   }

//   fn move_integer(&mut self, arg: Integer) {
//     // > If the argument is an Integral or Pointer Type, the size of the argument is less than or equal to 8 bytes and the NGRN is less than 8,
//     // > the argument is copied to the least significant bits in x[NGRN]. The NGRN is incremented by one. The argument has now been allocated.
//     // > [if NGRN is equal or more than 8]
//     // > The argument is copied to memory at the adjusted NSAA. The NSAA is incremented by the size of the argument. The argument has now been allocated.

//     let arg_i = self.integer_args + 1;
//     self.integer_args = arg_i;

//     let (Unsigned(size) | Signed(size)) = arg;

//     // https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms:
//     // > Function arguments may consume slots on the stack that are not multiples of 8 bytes.
//     // > If the total number of bytes for stack-based arguments is not a multiple of 8 bytes,
//     // > insert padding on the stack to maintain the 8-byte alignment requirements.
//     let bytes = size as u16;

//     // move each argument one position to the left. The first argument in the stack moves to the last register.
//     // If the FFI function is called with a new stack frame, the arguments remaining in the stack are copied to the new stack frame.
//     // Otherwise, they are copied 8 bytes lower
//     match (arg_i, arg) {
//       // From https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms:
//       // > The caller of a function is responsible for signing or zero-extending any argument with fewer than 32 bits.
//       // > The standard ABI expects the callee to sign or zero-extend those arguments.
//       // (this applies to register parameters, as stack parameters are not padded in Apple)
//       (1, Signed(B)) => dynasm!(self.assembler; .arch aarch64; sxtb w0, w1),
//       (1, Unsigned(B)) => {
//         dynasm!(self.assembler; .arch aarch64; and w0, w1, 0xFF)
//       }
//       (1, Signed(W)) => dynasm!(self.assembler; .arch aarch64; sxth w0, w1),
//       (1, Unsigned(W)) => {
//         dynasm!(self.assembler; .arch aarch64; and w0, w1, 0xFFFF)
//       }
//       (1, Signed(DW) | Unsigned(DW)) => {
//         dynasm!(self.assembler; .arch aarch64; mov w0, w1)
//       }
//       (1, Signed(QW) | Unsigned(QW)) => {
//         dynasm!(self.assembler; .arch aarch64; mov x0, x1)
//       }

//       (2, Signed(B)) => dynasm!(self.assembler; .arch aarch64; sxtb w1, w2),
//       (2, Unsigned(B)) => {
//         dynasm!(self.assembler; .arch aarch64; and w1, w2, 0xFF)
//       }
//       (2, Signed(W)) => dynasm!(self.assembler; .arch aarch64; sxth w1, w2),
//       (2, Unsigned(W)) => {
//         dynasm!(self.assembler; .arch aarch64; and w1, w2, 0xFFFF)
//       }
//       (2, Signed(DW) | Unsigned(DW)) => {
//         dynasm!(self.assembler; .arch aarch64; mov w1, w2)
//       }
//       (2, Signed(QW) | Unsigned(QW)) => {
//         dynasm!(self.assembler; .arch aarch64; mov x1, x2)
//       }

//       (3, Signed(B)) => dynasm!(self.assembler; .arch aarch64; sxtb w2, w3),
//       (3, Unsigned(B)) => {
//         dynasm!(self.assembler; .arch aarch64; and w2, w3, 0xFF)
//       }
//       (3, Signed(W)) => dynasm!(self.assembler; .arch aarch64; sxth w2, w3),
//       (3, Unsigned(W)) => {
//         dynasm!(self.assembler; .arch aarch64; and w2, w3, 0xFFFF)
//       }
//       (3, Signed(DW) | Unsigned(DW)) => {
//         dynasm!(self.assembler; .arch aarch64; mov w2, w3)
//       }
//       (3, Signed(QW) | Unsigned(QW)) => {
//         dynasm!(self.assembler; .arch aarch64; mov x2, x3)
//       }

//       (4, Signed(B)) => dynasm!(self.assembler; .arch aarch64; sxtb w3, w4),
//       (4, Unsigned(B)) => {
//         dynasm!(self.assembler; .arch aarch64; and w3, w4, 0xFF)
//       }
//       (4, Signed(W)) => dynasm!(self.assembler; .arch aarch64; sxth w3, w4),
//       (4, Unsigned(W)) => {
//         dynasm!(self.assembler; .arch aarch64; and w3, w4, 0xFFFF)
//       }
//       (4, Signed(DW) | Unsigned(DW)) => {
//         dynasm!(self.assembler; .arch aarch64; mov w3, w4)
//       }
//       (4, Signed(QW) | Unsigned(QW)) => {
//         dynasm!(self.assembler; .arch aarch64; mov x3, x4)
//       }

//       (5, Signed(B)) => dynasm!(self.assembler; .arch aarch64; sxtb w4, w5),
//       (5, Unsigned(B)) => {
//         dynasm!(self.assembler; .arch aarch64; and w4, w5, 0xFF)
//       }
//       (5, Signed(W)) => dynasm!(self.assembler; .arch aarch64; sxth w4, w5),
//       (5, Unsigned(W)) => {
//         dynasm!(self.assembler; .arch aarch64; and w4, w5, 0xFFFF)
//       }
//       (5, Signed(DW) | Unsigned(DW)) => {
//         dynasm!(self.assembler; .arch aarch64; mov w4, w5)
//       }
//       (5, Signed(QW) | Unsigned(QW)) => {
//         dynasm!(self.assembler; .arch aarch64; mov x4, x5)
//       }

//       (6, Signed(B)) => dynasm!(self.assembler; .arch aarch64; sxtb w5, w6),
//       (6, Unsigned(B)) => {
//         dynasm!(self.assembler; .arch aarch64; and w5, w6, 0xFF)
//       }
//       (6, Signed(W)) => dynasm!(self.assembler; .arch aarch64; sxth w5, w6),
//       (6, Unsigned(W)) => {
//         dynasm!(self.assembler; .arch aarch64; and w5, w6, 0xFFFF)
//       }
//       (6, Signed(DW) | Unsigned(DW)) => {
//         dynasm!(self.assembler; .arch aarch64; mov w5, w6)
//       }
//       (6, Signed(QW) | Unsigned(QW)) => {
//         dynasm!(self.assembler; .arch aarch64; mov x5, x6)
//       }

//       (7, Signed(B)) => dynasm!(self.assembler; .arch aarch64; sxtb w6, w7),
//       (7, Unsigned(B)) => {
//         dynasm!(self.assembler; .arch aarch64; and w6, w7, 0xFF)
//       }
//       (8, Signed(W)) => dynasm!(self.assembler; .arch aarch64; sxth w5, w7),
//       (8, Unsigned(W)) => {
//         dynasm!(self.assembler; .arch aarch64; and w5, w7, 0xFFFF)
//       }
//       (7, Signed(DW) | Unsigned(DW)) => {
//         dynasm!(self.assembler; .arch aarch64; mov w6, w7)
//       }
//       (7, Signed(QW) | Unsigned(QW)) => {
//         dynasm!(self.assembler; .arch aarch64; mov x6, x7)
//       }

//       (8, arg) => {
//         match arg {
//           Signed(B) => {
//             dynasm!(self.assembler; .arch aarch64; ldrsb w7, [sp, self.input_sp_offset()])
//           }
//           Unsigned(B) => {
//             dynasm!(self.assembler; .arch aarch64; ldrb w7, [sp, self.input_sp_offset()])
//           }
//           Signed(W) => {
//             dynasm!(self.assembler; .arch aarch64; ldrsh w7, [sp, self.input_sp_offset()])
//           }
//           Unsigned(W) => {
//             dynasm!(self.assembler; .arch aarch64; ldrh w7, [sp, self.input_sp_offset()])
//           }
//           Signed(DW) | Unsigned(DW) => {
//             dynasm!(self.assembler; .arch aarch64; ldr w7, [sp, self.input_sp_offset()])
//           }
//           Signed(QW) | Unsigned(QW) => {
//             dynasm!(self.assembler; .arch aarch64; ldr x7, [sp, self.input_sp_offset()])
//           }
//         }
//         // 16 and 8 bit integers are 32 bit integers in v8
//         self.stack_trampoline += arg.size().max(4);
//       }

//       (_, arg) => {
//         let size_original = arg.size();
//         // 16 and 8 bit integers are 32 bit integers in v8
//         let size_trampoline = size_original.max(4);
//         let padding_trampoline = (size_trampoline
//           - self.stack_trampoline % size_trampoline)
//           % size_trampoline;
//         let padding_original =
//           (size_original - self.stack_original % size_original) % size_original;
//         println!(
//           "input offset: {}",
//           self.stack_trampoline + padding_trampoline
//         );
//         println!("output offset: {}", self.stack_original + padding_original);
//         match arg {
//           Signed(B) | Unsigned(B) => dynasm!(self.assembler
//             ; .arch aarch64
//             ; ldr w8, [sp, self.stack_trampoline + padding_trampoline]
//             ; strb w8, [sp, self.stack_original + padding_original]
//           ),
//           Signed(W) | Unsigned(W) => dynasm!(self.assembler
//             ; .arch aarch64
//             ; ldr w8, [sp, self.stack_trampoline + padding_trampoline]
//             ; strh w8, [sp, self.stack_original + padding_original]
//           ),
//           Signed(DW) | Unsigned(DW) => dynasm!(self.assembler
//             ; .arch aarch64
//             ;  ldr w8, [sp, self.stack_trampoline + padding_trampoline]
//             ; str w8, [sp, self.stack_original + padding_original]
//           ),
//           Signed(QW) | Unsigned(QW) => dynasm!(self.assembler
//             ; .arch aarch64
//             ; ldr x8, [sp, self.stack_trampoline + padding_trampoline]
//             ; str x8, [sp, self.stack_original + padding_original]
//           ),
//         }
//         self.stack_trampoline += padding_trampoline + size_trampoline;
//         self.stack_original += padding_original + size_original;
//       }
//     };
//   }

//   fn input_sp_offset(&self) -> u32 {
//     (if self.stack_allocated > 0 {
//       self.stack_trampoline + self.stack_allocated
//     } else {
//       self.stack_trampoline
//     }) as u32
//   }

//   fn output_sp_offset(&self) -> u32 {
//     self.stack_original as u32
//   }

//   fn zero_first_arg(&mut self) {
//     dynasm!(self.assembler
//       ; .arch aarch64
//       ; mov x0, xzr
//     );
//   }

//   fn cast_return_value(&mut self, rv: NativeType) {
//     // Apple does not need to cast the return value
//     unreachable!()
//   }

//   fn allocate_stack(&mut self, params: &[NativeType]) {
//     // Apple always tail-calls
//     unreachable!()
//   }

//   fn deallocate_stack(&mut self) {
//     // Apple always tail-calls
//     unreachable!()
//   }

//   fn call(&mut self, ptr: *const c_void) {
//     // Apple always tail-calls
//     unreachable!()
//   }

//   fn tailcall(&mut self, ptr: *const c_void) {
//     // stack pointer is never modified and remains aligned
//     // frame pointer remains the one provided by the trampoline's caller (V8)

//     let mut address = ptr as u64;
//     let mut imm16 = address & 0xFFFF;
//     dynasm!(self.assembler
//       ; .arch aarch64
//       ; movz x8, imm16 as u32
//     );
//     address >>= 16;
//     let mut shift = 16;
//     while address > 0 {
//       imm16 = address & 0xFFFF;
//       dynasm!(self.assembler
//         ; .arch aarch64
//         ; movk x8, imm16 as u32, lsl shift
//       );
//       address >>= 16;
//       shift += 16;
//     }
//     dynasm!(self.assembler
//         ; .arch aarch64
//         ; br x8
//     );
//   }

//   fn ret(&mut self) {
//     // Apple always tail-calls
//     unreachable!()
//   }

//   fn integer_args_have_moved(&self) -> bool {
//     self.integer_args > 0
//   }

//   fn must_cast_return_value(&self, _rv: NativeType) -> bool {
//     // V8 only supports i32 and u32 return types for integers
//     // We support 8 and 16 bit integers by extending them to 32 bits in the trampoline before returning

//     // return values follow the same rules as register arguments. Therefore, in Apple the RV is sign/zero extended by the callee.
//     false
//   }

//   fn finalize(self) -> ExecutableBuffer {
//     self.assembler.finalize().unwrap()
//   }
// }
impl Aarch64Apple {
  // Integer arguments go to the first 8 GPR: x0-x7
  const INTEGER_REG: u32 = 8;
  // Floating-point arguments go to the first 8 SIMD & Floating-Point registers: v0-v1
  const FLOAT_REG: u32 = 8;

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
    let mut compiler = Self::new();

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

  fn move_float(&mut self, arg: Float) {
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
      let size = arg as u32;
      let padding_trampoline = (size - self.stack_trampoline % size) % size;
      let padding_original = (size - self.stack_original % size) % size;

      println!(
        "input offset: {}",
        self.stack_trampoline + padding_trampoline
      );
      println!("output offset: {}", self.stack_original + padding_original);
      // floats are only moved to accomodate integer movement in the stack
      let stack_has_moved =
        self.stack_allocated > 0 || self.integer_args >= Self::INTEGER_REG;
      if stack_has_moved {
        match arg {
          Single => dynasm!(self.assembler
            ; .arch aarch64
            // 6.1.2 Aarch64 PCS:
            // > Registers v8-v15 must be preserved by a callee across subroutine calls;
            // > the remaining registers (v0-v7, v16-v31) do not need to be preserved (or should be preserved by the caller).
            ; ldr s16, [sp, self.stack_trampoline + padding_trampoline]
            ; str s16, [sp, self.stack_original + padding_original]
          ),
          Double => dynasm!(self.assembler
            ; .arch aarch64
            ; ldr d16, [sp, self.stack_trampoline + padding_trampoline]
            ; str d16, [sp, self.stack_original + padding_original]
          ),
        }
      }
      // The trampoline and the orignal function always have the same amount of floats in the stack
      self.stack_trampoline += size + padding_trampoline;
      self.stack_original += size + padding_original;
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

    // https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms:
    // > Function arguments may consume slots on the stack that are not multiples of 8 bytes.
    // > If the total number of bytes for stack-based arguments is not a multiple of 8 bytes,
    // > insert padding on the stack to maintain the 8-byte alignment requirements.
    let bytes = size as u16;

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
      (7, Signed(W)) => dynasm!(self.assembler; .arch aarch64; sxth w6, w7),
      (7, Unsigned(W)) => {
        dynasm!(self.assembler; .arch aarch64; and w6, w7, 0xFFFF)
      }
      (7, Signed(DW) | Unsigned(DW)) => {
        dynasm!(self.assembler; .arch aarch64; mov w6, w7)
      }
      (7, Signed(QW) | Unsigned(QW)) => {
        dynasm!(self.assembler; .arch aarch64; mov x6, x7)
      }

      (8, arg) => {
        match arg {
          Signed(B) => {
            dynasm!(self.assembler; .arch aarch64; ldrsb w7, [sp, self.stack_trampoline])
          }
          Unsigned(B) => {
            dynasm!(self.assembler; .arch aarch64; ldrb w7, [sp, self.stack_trampoline])
          }
          Signed(W) => {
            dynasm!(self.assembler; .arch aarch64; ldrsh w7, [sp, self.stack_trampoline])
          }
          Unsigned(W) => {
            dynasm!(self.assembler; .arch aarch64; ldrh w7, [sp, self.stack_trampoline])
          }
          Signed(DW) | Unsigned(DW) => {
            dynasm!(self.assembler; .arch aarch64; ldr w7, [sp, self.stack_trampoline])
          }
          Signed(QW) | Unsigned(QW) => {
            dynasm!(self.assembler; .arch aarch64; ldr x7, [sp, self.stack_trampoline])
          }
        }
        // 16 and 8 bit integers are 32 bit integers in v8
        self.stack_trampoline += arg.size().max(4);
      }

      (_, arg) => {
        let size_original = arg.size();
        // 16 and 8 bit integers are 32 bit integers in v8
        let size_trampoline = size_original.max(4);
        let padding_trampoline = (size_trampoline
          - self.stack_trampoline % size_trampoline)
          % size_trampoline;
        let padding_original =
          (size_original - self.stack_original % size_original) % size_original;
        println!(
          "input offset: {}",
          self.stack_trampoline + padding_trampoline
        );
        println!("output offset: {}", self.stack_original + padding_original);
        match arg {
          Signed(B) | Unsigned(B) => dynasm!(self.assembler
            ; .arch aarch64
            ; ldr w8, [sp, self.stack_trampoline + padding_trampoline]
            ; strb w8, [sp, self.stack_original + padding_original]
          ),
          Signed(W) | Unsigned(W) => dynasm!(self.assembler
            ; .arch aarch64
            ; ldr w8, [sp, self.stack_trampoline + padding_trampoline]
            ; strh w8, [sp, self.stack_original + padding_original]
          ),
          Signed(DW) | Unsigned(DW) => dynasm!(self.assembler
            ; .arch aarch64
            ;  ldr w8, [sp, self.stack_trampoline + padding_trampoline]
            ; str w8, [sp, self.stack_original + padding_original]
          ),
          Signed(QW) | Unsigned(QW) => dynasm!(self.assembler
            ; .arch aarch64
            ; ldr x8, [sp, self.stack_trampoline + padding_trampoline]
            ; str x8, [sp, self.stack_original + padding_original]
          ),
        }
        self.stack_trampoline += padding_trampoline + size_trampoline;
        self.stack_original += padding_original + size_original;
      }
    };
  }

  fn zero_first_arg(&mut self) {
    dynasm!(self.assembler
      ; .arch aarch64
      ; mov x0, xzr
    );
  }

  fn cast_return_value(&mut self, rv: NativeType) {
    // Apple does not need to cast the return value
    unreachable!()
  }

  fn allocate_stack(&mut self, params: &[NativeType]) {
    // Apple always tail-calls
    unreachable!()
  }

  fn deallocate_stack(&mut self) {
    // Apple always tail-calls
    unreachable!()
  }

  fn call(&mut self, ptr: *const c_void) {
    // Apple always tail-calls
    unreachable!()
  }

  fn tailcall(&mut self, ptr: *const c_void) {
    // stack pointer is never modified and remains aligned
    // frame pointer remains the one provided by the trampoline's caller (V8)

    let mut address = ptr as u64;
    let mut imm16 = address & 0xFFFF;
    dynasm!(self.assembler
      ; .arch aarch64
      ; movz x8, imm16 as u32
    );
    address >>= 16;
    let mut shift = 16;
    while address > 0 {
      imm16 = address & 0xFFFF;
      dynasm!(self.assembler
        ; .arch aarch64
        ; movk x8, imm16 as u32, lsl shift
      );
      address >>= 16;
      shift += 16;
    }
    dynasm!(self.assembler
        ; .arch aarch64
        ; br x8
    );
  }

  fn ret(&mut self) {
    // Apple always tail-calls
    unreachable!()
  }

  fn integer_args_have_moved(&self) -> bool {
    self.integer_args > 0
  }

  fn must_cast_return_value(&self, _rv: NativeType) -> bool {
    // V8 only supports i32 and u32 return types for integers
    // We support 8 and 16 bit integers by extending them to 32 bits in the trampoline before returning

    // return values follow the same rules as register arguments. Therefore, in Apple the RV is sign/zero extended by the callee.
    false
  }

  fn finalize(self) -> ExecutableBuffer {
    self.assembler.finalize().unwrap()
  }
}

struct X64Windows {
  assembler: dynasmrt::x64::Assembler,
  args: i32,

  allocated_stack: u16,
}

impl X64Windows {
  // > Integer valued arguments in the leftmost four positions are passed in left-to-right order in RCX, RDX, R8, and R9
  // > [...]
  // > Any floating-point and double-precision arguments in the first four parameters are passed in XMM0 - XMM3, depending on position
  const REGISTER_ARGS: i32 = 4;

  fn new() -> Self {
    Self {
      assembler: dynasmrt::x64::Assembler::new().unwrap(),
      args: 0,
      allocated_stack: 0,
    }
  }

  fn compile(sym: &Symbol) -> Trampoline {
    // TODO: Apple Silicon & windows x64 support
    let mut compiler = Self::new();

    let can_tailcall = !compiler.must_cast_return_value(sym.result_type);
    if !can_tailcall {
      compiler.allocate_stack(&sym.parameter_types);
    }

    for argument in &sym.parameter_types {
      compiler.move_left(argument)
    }
    if !compiler.is_recv_overriden() {
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

  fn move_sse(&mut self, arg: Float) {
    self.move_arg(F(arg))
  }

  fn move_integer(&mut self, arg: Integer) {
    let (Signed(size) | Unsigned(size)) = arg;
    self.move_arg(I(size))
  }

  fn move_arg(&mut self, arg: Argument) {
    let arg_i = self.args + 1;
    self.args = arg_i;

    // adding 1 to the integer amount to account for the receiver
    // Not substrating REGISTER-ARGS because of shadow space for register parameters
    let pos_in_stack = 1 + arg_i;
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
      (1, I(B | W | DW)) => dynasm!(self.assembler; .arch x64; mov ecx, edx),
      (1, I(QW)) => dynasm!(self.assembler; .arch x64; mov rcx, rdx),
      (1, F(_)) => {
        // Use movaps for doubles as well, benefits of smaller encoding outweight those of using the correct instruction for the type (movapd)
        dynasm!(self.assembler; .arch x64; xor ecx, ecx; movaps xmm0, xmm1)
      }

      (2, I(B | W | DW)) => dynasm!(self.assembler; .arch x64; mov edx, r8d),
      (2, I(QW)) => dynasm!(self.assembler; .arch x64; mov rdx, r8),
      (2, F(_)) => {
        dynasm!(self.assembler; .arch x64; movaps xmm1, xmm2)
      }

      (3, I(B | W | DW)) => dynasm!(self.assembler; .arch x64; mov r8d, r9d),
      (3, I(QW)) => dynasm!(self.assembler; .arch x64; mov r8, r9),
      (3, F(_)) => {
        dynasm!(self.assembler; .arch x64; movaps xmm2, xmm3)
      }

      (4, I(B | W | DW)) => {
        dynasm!(self.assembler; .arch x64; mov r9d, [rsp + rsp_offset])
      }
      (4, I(QW)) => dynasm!(self.assembler; .arch x64; mov r9, [rsp + rsp_offset]),
      (4, F(_)) => {
        // parameter 4 is always 16-byte aligned, so we can use movaps instead of movups
        dynasm!(self.assembler; .arch x64; movaps xmm3, [rsp + rsp_offset])
      }
      (_, I(B | W | DW)) => {
        dynasm!(self.assembler
          ; .arch x64
          ; mov eax, [rsp + rsp_offset]
          ; mov [rsp + new_rsp_offset], eax
        )
      }
      (_, I(QW)) => {
        dynasm!(self.assembler
          ; .arch x64
          ; mov rax, [rsp + rsp_offset]
          ; mov [rsp + new_rsp_offset], rax
        )
      }
      (_, F(_)) => {
        dynasm!(self.assembler
          ; .arch x64
          ; movups xmm4, [rsp + rsp_offset]
          ; movups [rsp + new_rsp_offset], xmm4
        )
      } //   (1, I(B)) => dynasm!(self.assembler; .arch x64; mov cl, dl),
        //   (1, I(W)) => dynasm!(self.assembler; .arch x64; mov cx, dx),
        //   (1, I(DW)) => dynasm!(self.assembler; .arch x64; mov ecx, edx),
        //   (1, I(QW)) => dynasm!(self.assembler; .arch x64; mov rcx, rdx),
        //   (1, F(Single)) => {
        //     dynasm!(self.assembler; .arch x64; xor ecx, ecx; movaps xmm0, xmm1)
        //   }
        //   (1, F(Double)) => {
        //     dynasm!(self.assembler; .arch x64; xor ecx, ecx; movapd xmm0, xmm1)
        //   }

        //   (2, I(B)) => dynasm!(self.assembler; .arch x64; mov dl, r8b),
        //   (2, I(W)) => dynasm!(self.assembler; .arch x64; mov dx, r8w),
        //   (2, I(DW)) => dynasm!(self.assembler; .arch x64; mov edx, r8d),
        //   (2, I(QW)) => dynasm!(self.assembler; .arch x64; mov rdx, r8),
        //   (2, F(Single)) => dynasm!(self.assembler; .arch x64; movaps xmm1, xmm2),
        //   (2, F(Double)) => dynasm!(self.assembler; .arch x64; movapd xmm1, xmm2),

        //   (3, I(B)) => dynasm!(self.assembler; .arch x64; mov r8b, r9b),
        //   (3, I(W)) => dynasm!(self.assembler; .arch x64; mov r8w, r9w),
        //   (3, I(DW)) => dynasm!(self.assembler; .arch x64; mov r8d, r9d),
        //   (3, I(QW)) => dynasm!(self.assembler; .arch x64; mov r8, r9),
        //   (3, F(Single)) => dynasm!(self.assembler; .arch x64; movaps xmm2, xmm3),
        //   (3, F(Double)) => dynasm!(self.assembler; .arch x64; movapd xmm2, xmm3),

        //   (4, I(B)) => {
        //     dynasm!(self.assembler; .arch x64; mov r9b, [rsp + rsp_offset])
        //   }
        //   (4, I(W)) => {
        //     dynasm!(self.assembler; .arch x64; mov r9w, [rsp + rsp_offset])
        //   }
        //   (4, I(DW)) => {
        //     dynasm!(self.assembler; .arch x64; mov r9d, [rsp + rsp_offset])
        //   }
        //   (4, I(QW)) => {
        //     dynasm!(self.assembler; .arch x64; mov r9, [rsp + rsp_offset])
        //   }
        //   (4, F(Single)) => {
        //     dynasm!(self.assembler; .arch x64; movaps xmm3, [rsp + rsp_offset])
        //   }
        //   (4, F(Double)) => {
        //     dynasm!(self.assembler; .arch x64; movapd xmm3, [rsp + rsp_offset])
        //   }

        //   (_, I(B)) => {
        //     dynasm!(self.assembler
        //       ;.arch x64
        //       ; mov eax, [rsp + rsp_offset]  // Document register renaming
        //       ; mov [rsp + new_rsp_offset], al
        //     )
        //   }
        //   (_, I(W)) => {
        //     dynasm!(self.assembler
        //       ;.arch x64
        //       ; mov eax, [rsp + rsp_offset]
        //       ; mov [rsp + new_rsp_offset], ax
        //     )
        //   }
        //   (_, I(DW)) => {
        //     dynasm!(self.assembler
        //       ;.arch x64
        //       ; mov eax, [rsp + rsp_offset]
        //       ; mov [rsp + new_rsp_offset], eax
        //     )
        //   }
        //   (_, I(QW)) => {
        //     dynasm!(self.assembler
        //       ;.arch x64
        //       ; mov rax, [rsp + rsp_offset]
        //       ; mov [rsp + new_rsp_offset], rax
        //     )
        //   }
        //   (_, F(Single) | F(Double)) => {
        //     dynasm!(self.assembler
        //       ; .arch x64
        //       ; movups xmm4, [rsp + rsp_offset]
        //       ; movups [rsp + new_rsp_offset], xmm4
        //     )
        //   }
    }
  }

  fn zero_first_arg(&mut self) {
    dynasm!(self.assembler
      ; .arch x64
      ; xor ecx, ecx
    );
  }

  fn cast_return_value(&mut self, rv: NativeType) {
    // 8 and 16 bit integers are extended to 32 bits
    match rv {
      NativeType::U8 => dynasm!(self.assembler; .arch x64; movzx eax, al),
      NativeType::I8 => dynasm!(self.assembler; .arch x64; movsx eax, al),
      NativeType::U16 => dynasm!(self.assembler; .arch x64; movzx eax, ax),
      NativeType::I16 => dynasm!(self.assembler; .arch x64; movsx eax, ax),
      _ => (),
    }
  }

  fn allocate_stack(&mut self, params: &[NativeType]) {
    let mut stack_size = 0u16;
    stack_size += 32; // Shadow space
    stack_size +=
      params.iter().skip(Self::REGISTER_ARGS as usize).count() as u16 * 8;

    // Align stack (considering 8 byte caller return address)
    stack_size += (16 - (stack_size + 8) % 16) % 16;

    dynasm!(self.assembler
      ; .arch x64
      ; sub rsp, stack_size as i32
    );
    self.allocated_stack = stack_size;
  }

  fn deallocate_stack(&mut self) {
    dynasm!(self.assembler
      ; .arch x64
      ; add rsp, self.allocated_stack as i32
    );
  }

  fn call(&mut self, ptr: *const c_void) {
    // the stack has been aligned during stack allocation
    dynasm!(self.assembler
      ; .arch x64
      ; mov rax, QWORD ptr as _
      ; call rax
    );
  }

  fn tailcall(&mut self, ptr: *const c_void) {
    // stack pointer is never modified and remains aligned
    // return address remains the one provided by the trampoline's caller (V8)
    dynasm!(self.assembler
      ; .arch x64
      ; mov rax, QWORD ptr as _
      ; jmp rax
    );
  }

  fn ret(&mut self) {
    // the stack has been deallocated before ret is called
    dynasm!(self.assembler
      ; .arch x64
      ; ret
    );
  }

  fn is_recv_overriden(&self) -> bool {
    self.args > 0
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
    let mut compiler = Aarch64::new();

    let can_tailcall = !compiler.must_cast_return_value(sym.result_type);
    if !can_tailcall {
      compiler.allocate_stack(&sym.parameter_types);
    }

    for argument in &sym.parameter_types {
      compiler.move_left(argument)
    }
    if !compiler.is_recv_overriden() {
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
      // The trampoline and the orignal function always have the same amount of floats in the stack
      self.stack_trampoline += float as u16;
      self.stack_original += float as u16;
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
        println!("input offset: {}", self.input_sp_offset());
        println!("output offset: {}", self.output_sp_offset());
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
    // TODO: REFACTOR, AS APPLE ALWAYS TAILCALLS
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
        NativeType::I32 | NativeType::U32 | NativeType::Void => {
          int += 1;
          if int > 0 {
            stack_size += 4;
          }
        }
        NativeType::I64
        | NativeType::U64
        | NativeType::Function
        | NativeType::USize
        | NativeType::ISize
        | NativeType::Pointer => {
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
    // TODO: CORRECT MODULO LOGIC
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
      ; ldp x29, x30, [sp, self.stack_allocated - 16]
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

  fn is_recv_overriden(&self) -> bool {
    self.integer_args > 0
  }

  fn must_cast_return_value(&self, rv: NativeType) -> bool {
    // V8 only supports i32 and u32 return types for integers
    // We support 8 and 16 bit integers by extending them to 32 bits in the trampoline before returning

    // return values follow the same rules as register arguments. Therefore, in Apple the RV is sign/zero extended by the callee,
    // whereas standard ARM64 dictates the upper bits are undefined.
    // At the time of writing Rust has an outstanding bug when targetting aarch64-unknown-linux-gnu,
    // where integer return values smaller than 32 bits are assumed to be sign extended by the callee:
    // https://github.com/rust-lang/rust/issues/97463
    cfg!(not(target_vendor = "apple"))
      && matches!(
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

impl Integer {
  fn size(self) -> u32 {
    match self {
      Signed(size) | Unsigned(size) => size as u32,
    }
  }
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

#[derive(Clone, Copy, Debug)]
enum Argument {
  I(Size),
  F(Float),
}

use Argument::*;

#[cfg(test)]
mod tests {
  use std::ptr::null_mut;

  use libffi::middle::Type;

  use crate::NativeType;
  use crate::Symbol;

  fn symbol(parameters: Vec<NativeType>, ret: NativeType) -> Symbol {
    Symbol {
      cif: libffi::middle::Cif::new(vec![], Type::void()),
      ptr: libffi::middle::CodePtr(null_mut()),
      parameter_types: parameters,
      result_type: ret,
      can_callback: false,
    }
  }

  mod aarch64_apple {
    use std::ops::Deref;

    use dynasmrt::dynasm;

    use super::super::Aarch64Apple;
    use super::symbol;
    use crate::NativeType::*;

    #[test]
    fn tailcall() {
      let trampoline = Aarch64Apple::compile(&symbol(
        vec![
          U8, U16, I16, I8, U32, U64, Pointer, Function, I64, I32, I16, I8,
          F32, F32, F32, F32, F64, F64, F64, F64, F32, F64,
        ],
        Void,
      ));

      let mut assembler = dynasmrt::aarch64::Assembler::new().unwrap();
      // See https://godbolt.org/z/Gr1Mcbch5
      dynasm!(assembler
        ; .arch aarch64
        ; and w0, w1, 0xFF   // u8
        ; and w1, w2, 0xFFFF // u16
        ; sxth w2, w3        // i16
        ; sxtb w3, w4        // i8
        ; mov w4, w5         // u32
        ; mov x5, x6         // u64
        ; mov x6, x7         // Pointer
        ; ldr x7, [sp]       // Function
        ; ldr x8, [sp, 8]    // i64
        ; str x8, [sp]       // ..
        ; ldr w8, [sp, 16]   // i32
        ; str w8, [sp, 8]    // ..
        ; ldr w8, [sp, 20]   // i16
        ; strh w8, [sp, 12]   // ..
        ; ldr w8, [sp, 24]   // i8
        ; strb w8, [sp, 14]   // ..
        ; ldr s16, [sp, 28]  // f32
        ; str s16, [sp, 16]  // ..
        ; ldr d16, [sp, 32]  // f64
        ; str d16, [sp, 24]  // ..
        ; movz x8, 0
        ; br x8
      );
      let expected = assembler.finalize().unwrap();
      assert_eq!(trampoline.0.deref(), expected.deref());
    }
  }

  mod x64_windows {
    use std::ops::Deref;

    use dynasmrt::{dynasm, DynasmApi};

    use super::super::X64Windows;
    use super::symbol;
    use crate::NativeType::*;

    #[test]
    fn tailcall() {
      let trampoline = X64Windows::compile(&symbol(
        vec![U8, I16, F64, F32, U32, I8, Pointer],
        Void,
      ));

      let mut assembler = dynasmrt::x64::Assembler::new().unwrap();
      dynasm!(assembler
        ; .arch x64
        ; mov ecx, edx          // u8
        ; mov edx, r8d         // i16
        ; movaps xmm2, xmm3      // f64
        ; movaps xmm3, [DWORD rsp + 40] // f32
        ; mov eax, [DWORD rsp + 48]    // u32
        ; mov [DWORD rsp + 40], eax    // ..
        ; mov eax, [DWORD rsp + 56]    // i8
        ; mov [DWORD rsp + 48], eax     // ..
        ; mov rax, [DWORD rsp + 64]    // Pointer
        ; mov [DWORD rsp + 56], rax    // ..
        ; mov rax, QWORD 0
        ; jmp rax
      );
      let expected = assembler.finalize().unwrap();
      assert_eq!(trampoline.0.deref(), expected.deref());
    }
  }
}
