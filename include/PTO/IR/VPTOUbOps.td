// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOUbOps.td - PTO raw ubuf ops (a2a3) ----------------*- tablegen -*-===//

#ifndef MLIR_DIALECT_PTO_IR_VPTOUBOPS
#define MLIR_DIALECT_PTO_IR_VPTOUBOPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "PTO/IR/VPTOInterfaces.td"

class PTO_UBufBinaryOp<string mnemonic, list<Trait> traits = []>
    : PTO_Op<mnemonic,
      !listconcat(traits, [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>])> {
  let summary = "Raw ubuf binary operation mapping to llvm.hivm intrinsics.";
  let description = [{
    Lowers to `llvm.hivm` intrinsics for the a2a3 (dav-c220-vec) backend.
    Operates directly on ubuf pointers rather than vreg values.

    The repeat / block-stride / repeat-stride operands are packed into a single
    `i64` config during lowering, following the bit layout defined in
    `cce_aicore_intrinsics_3101.h`.
  }];

  let arguments = (ins
    PTO_BufferType:$dst,
    PTO_BufferType:$src0,
    PTO_BufferType:$src1,
    I64:$repeat,
    I64:$dstBlockStride,
    I64:$src0BlockStride,
    I64:$src1BlockStride,
    I64:$dstRepeatStride,
    I64:$src0RepeatStride,
    I64:$src1RepeatStride
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $dst `,` $src0 `,` $src1 `,`
    $repeat `,` $dstBlockStride `,` $src0BlockStride `,`
    $src1BlockStride `,` $dstRepeatStride `,` $src0RepeatStride `,`
    $src1RepeatStride
    attr-dict `:` type($dst) `,` type($src0) `,` type($src1) `,`
    type($repeat) `,` type($dstBlockStride) `,` type($src0BlockStride) `,`
    type($src1BlockStride) `,` type($dstRepeatStride) `,`
    type($src0RepeatStride) `,` type($src1RepeatStride)
  }];
}

def PTO_UBVaddOp : PTO_UBufBinaryOp<"ub.vadd">;
def PTO_UBVsubOp : PTO_UBufBinaryOp<"ub.vsub">;
def PTO_UBVmulOp : PTO_UBufBinaryOp<"ub.vmul">;
def PTO_UBVdivOp : PTO_UBufBinaryOp<"ub.vdiv">;

def PTO_UBSetMaskOp : PTO_Op<"ub.set_mask", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Set vector mask lanes for UB buffer pipe operations";
  let description = [{
    Configures the hardware vector mask registers. Subsequent `pto.ub.*`
    operations in the same basic block execute under this mask.

    Each bit in the mask corresponds to one element lane. `mask0` covers
    lanes 0–63, `mask1` covers lanes 64–127.
    Use `mask0=-1, mask1=-1` to restore full-write mode (all lanes active).
  }];

  let arguments = (ins I64:$mask0, I64:$mask1);
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $mask0 `,` $mask1 attr-dict `:` type($mask0) `,` type($mask1)
  }];
}

def PTO_UBSetMaskCountOp : PTO_Op<"ub.set_mask_count", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Set count mode for UB buffer pipe operations";
  let description = [{
    Toggles the hardware control register into count mode (bit 56 set).
    In count mode, subsequent `pto.ub.*` operations use `MASK[0]` as an
    element count rather than a per-lane bitmask. `repeat=0` signals the
    hardware to use the count register.

    Use `pto.ub.set_mask_norm` to restore normal mask mode.
  }];

  let arguments = (ins);
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    attr-dict
  }];
}

def PTO_UBSetMaskNormOp : PTO_Op<"ub.set_mask_norm", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Restore normal mask mode for UB buffer pipe operations";
  let description = [{
    Toggles the hardware control register out of count mode (bit 56 cleared).
    Subsequent `pto.ub.*` operations interpret `MASK[0..1]` as per-lane
    bitmasks (normal mode).

    Always pair with a preceding `pto.ub.set_mask_count` or restore after
    count-mode operations.
  }];

  let arguments = (ins);
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    attr-dict
  }];
}

#endif // MLIR_DIALECT_PTO_IR_VPTOUBOPS
