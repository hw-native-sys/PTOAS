// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// C wrapper TU for pto.print / pto.tprint support on the VPTO backend.
//
// The VPTO LLVM emitter cannot instantiate the cce::printf template itself
// (the CCE frontend owns print.h), so this file provides plain "C" shims that
// delegate to cce::printf.  It is NOT part of the ptoas host build: the
// ObjectEmission pipeline compiles it with bisheng (CC1, no -cce-enable-mix)
// into bitcode and llvm-links it with the kernel bitcode before device
// compilation.
//
// The format string arrives as a runtime pointer to a module global defined
// in the kernel TU — cce::printf treats fmt as an opaque byte stream and
// never resolves it by address, so external globals work.
//
// The DebugTunnelData pointer (DTData) is passed by the kernel as a hidden
// parameter; pto_print_init/finish set up / tear down the fix-stack object
// that cce::printf's GetKernelInstance() reads back.  This mirrors what the
// CCE driver's -cce-aicore-enable-print-init-finish pass injects for kernel
// entries; that pass crashes on -x ir input, so the kernel calls these
// directly instead.

#include <ccelib/__ccelib.h>
#include <stdint.h>

extern "C" [aicore] void *pto_get_fix_stack_object()
    asm("llvm.hivm.get.sycl.fix.stack.object");

// Literal text node (no conversion): used for tprint headers and shapes.
extern "C" [aicore] __attribute__((always_inline)) void
pto_print_str(__gm__ const char *fmt) {
  cce::printf(fmt);
}

// Float value: f16/bf16/f64 operands are converted to f32 by the emitter
// before the call (the DebugTunnel FLOAT node carries 4 bytes).
extern "C" [aicore] __attribute__((always_inline)) void
pto_print_f32(__gm__ const char *fmt, float v) {
  cce::printf(fmt, v);
}

// Signed integer: the INT node carries 8 bytes, so narrow operands are
// sign-extended to i64 by the emitter.
extern "C" [aicore] __attribute__((always_inline)) void
pto_print_i64(__gm__ const char *fmt, int64_t v) {
  cce::printf(fmt, v);
}

// Unsigned integer: zero-extended to i64 by the emitter so %u/%x/%o format
// the widened value correctly on the host.  cce::printf's Support<> list has
// no 64-bit unsigned type (it stops at uint32_t), so the value is forwarded
// as long long — the INT node still carries the same 8 bytes and the host
// format string reinterprets them.
extern "C" [aicore] __attribute__((always_inline)) void
pto_print_u64(__gm__ const char *fmt, uint64_t v) {
  cce::printf(fmt, (long long)v);
}

extern "C" [aicore] __attribute__((always_inline)) void
pto_print_init(__gm__ void *dt) {
  cce::internal::DebugTunnel::OnKernelInitialize(
      (__gm__ cce::internal::DebugTunnelData *)dt);
  *(uint64_t *)pto_get_fix_stack_object() = (uint64_t)dt;
}

extern "C" [aicore] __attribute__((always_inline)) void
pto_print_finish(__gm__ void *dt) {
  cce::internal::DebugTunnel::OnKernelFinish(
      (__gm__ cce::internal::DebugTunnelData *)dt);
}
