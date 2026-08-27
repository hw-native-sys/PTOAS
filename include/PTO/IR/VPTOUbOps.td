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
def PTO_UBVmaxOp : PTO_UBufBinaryOp<"ub.vmax">;
def PTO_UBVminOp : PTO_UBufBinaryOp<"ub.vmin">;
def PTO_UBVandOp : PTO_UBufBinaryOp<"ub.vand">;
def PTO_UBVorOp  : PTO_UBufBinaryOp<"ub.vor">;
def PTO_UBVaddReluOp : PTO_UBufBinaryOp<"ub.vaddrelu">;

//===----------------------------------------------------------------------===//
// PTO_UBufUnaryOp — unary elementwise (VNOT)
//===----------------------------------------------------------------------===//

class PTO_UBufUnaryOp<string mnemonic, list<Trait> traits = []>
    : PTO_Op<mnemonic,
      !listconcat(traits, [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>])> {
  let summary = "Raw ubuf unary operation mapping to llvm.hivm intrinsics.";
  let description = [{
    Lowers to `llvm.hivm` unary intrinsics (e.g. VNOT) for the a2a3
    (dav-c220-vec) backend. Operates directly on ubuf pointers.

    Config word uses unary layout: repeat[63:56], dstBlkStride[15:0],
    srcBlkStride[31:16], dstRepStride[39:32], srcRepStride[51:40].
  }];

  let arguments = (ins
    PTO_BufferType:$dst,
    PTO_BufferType:$src,
    I64:$repeat,
    I64:$dstBlockStride,
    I64:$srcBlockStride,
    I64:$dstRepeatStride,
    I64:$srcRepeatStride
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $dst `,` $src `,`
    $repeat `,` $dstBlockStride `,` $srcBlockStride `,`
    $dstRepeatStride `,` $srcRepeatStride
    attr-dict `:` type($dst) `,` type($src) `,`
    type($repeat) `,` type($dstBlockStride) `,` type($srcBlockStride) `,`
    type($dstRepeatStride) `,` type($srcRepeatStride)
  }];
}

def PTO_UBVnotOp : PTO_UBufUnaryOp<"ub.vnot">;
def PTO_UBVabsOp : PTO_UBufUnaryOp<"ub.vabs">;
def PTO_UBVreluOp : PTO_UBufUnaryOp<"ub.vrelu">;
def PTO_UBVexpOp : PTO_UBufUnaryOp<"ub.vexp">;
def PTO_UBVlnOp : PTO_UBufUnaryOp<"ub.vln">;
def PTO_UBVsqrtOp : PTO_UBufUnaryOp<"ub.vsqrt">;
def PTO_UBVrsqrtOp : PTO_UBufUnaryOp<"ub.vrsqrt">;

//===----------------------------------------------------------------------===//
// PTO_UBufShiftOp — unary shift with scalar distance
//===----------------------------------------------------------------------===//

class PTO_UBufShiftOp<string mnemonic, list<Trait> traits = []>
    : PTO_Op<mnemonic,
      !listconcat(traits, [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>])> {
  let summary = "Raw ubuf shift operation mapping to llvm.hivm.VSHL/VSHR.";
  let description = [{
    Lowers to `llvm.hivm.VSHL.u16` / `llvm.hivm.VSHR.u16` for the a2a3
    (dav-c220-vec) backend. Operates directly on ubuf pointers.

    Unlike binary ops, shift takes a single source pointer and a scalar
    `shiftDist` (i64) — the hardware applies a uniform shift to all elements.
  }];

  let arguments = (ins
    PTO_BufferType:$dst,
    PTO_BufferType:$src,
    I64:$shiftDist,
    I64:$repeat,
    I64:$dstBlockStride,
    I64:$srcBlockStride,
    I64:$dstRepeatStride,
    I64:$srcRepeatStride
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $dst `,` $src `,` $shiftDist `,`
    $repeat `,` $dstBlockStride `,` $srcBlockStride `,`
    $dstRepeatStride `,` $srcRepeatStride
    attr-dict `:` type($dst) `,` type($src) `,` type($shiftDist) `,`
    type($repeat) `,` type($dstBlockStride) `,` type($srcBlockStride) `,`
    type($dstRepeatStride) `,` type($srcRepeatStride)
  }];
}

def PTO_UBVshlOp : PTO_UBufShiftOp<"ub.vshl">;
def PTO_UBVshrOp : PTO_UBufShiftOp<"ub.vshr">;
def PTO_UBVmulSOp : PTO_UBufShiftOp<"ub.vmuls">;
def PTO_UBVaddSOp : PTO_UBufShiftOp<"ub.vadds">;
def PTO_UBVmaxSOp : PTO_UBufShiftOp<"ub.vmaxs">;
def PTO_UBVminSOp : PTO_UBufShiftOp<"ub.vmins">;

def PTO_UBVdupOp : PTO_Op<"ub.vdup", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Raw ubuf scalar duplicate mapping to llvm.hivm.MOVEV.*";

  let arguments = (ins
    PTO_BufferType:$dst,
    I64:$scalar,
    I64:$repeat,
    I64:$dstBlockStride,
    I64:$srcBlockStride,
    I64:$dstRepeatStride,
    I64:$srcRepeatStride
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $dst `,` $scalar `,`
    $repeat `,` $dstBlockStride `,` $srcBlockStride `,`
    $dstRepeatStride `,` $srcRepeatStride
    attr-dict `:` type($dst) `,` type($scalar) `,`
    type($repeat) `,` type($dstBlockStride) `,` type($srcBlockStride) `,`
    type($dstRepeatStride) `,` type($srcRepeatStride)
  }];
}

def PTO_UBVgatherbOp : PTO_Op<"ub.vgatherb", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Raw ubuf byte-offset gather mapping to the CCE vgatherb intrinsic (a2a3).";
  let description = [{
    Lowers to the raw CCE `vgatherb` intrinsic for the a2a3 (dav-c220-vec)
    backend. The bishem-emitted intrinsic is the packed 3-operand form
    `llvm.hivm.VGATHERB.b16/.b32(ptr dst, ptr offset, i64 config)`, where the
    2nd pointer is the offset buffer and the source data base address is packed
    into `config[31:0]`. Config layout (decoded from bisheng IR — see
    docs/designs/a2a3-vpto-gather.md): srcAddr[31:0], dstRepeatStride[39:32],
    dstBlockStride[47:40], reserved[55:48]=0, repeat[63:56]. This is the A2/A3
    UB-pointer gather primitive used to lower `pto.tgatherb`; do not confuse
    with the A5 vreg `vgather2`.
  }];

  let arguments = (ins
    PTO_BufferType:$dst,
    PTO_BufferType:$offset,
    PTO_BufferType:$src,
    I64:$dstRepeatStride,
    I64:$dstBlockStride,
    I64:$repeat
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $dst `,` $offset `,` $src `,`
    $dstRepeatStride `,` $dstBlockStride `,` $repeat
    attr-dict `:` type($dst) `,` type($offset) `,` type($src) `,`
    type($dstRepeatStride) `,` type($dstBlockStride) `,` type($repeat)
  }];
}

def PTO_UBVgatherOp : PTO_Op<"ub.vgather", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Raw ubuf element gather mapping to the CCE vgather intrinsic (a2a3).";
  let description = [{
    Lowers to `llvm.hivm.VGATHER.b16/.b32(dst_ptr, src_ptr, i64 config)`.
    The 2nd pointer is the address buffer; `offsetAddr` is a base offset added
    to each address in the buffer. Config layout: offsetAddr[31:0],
    dstRepeatStride[39:32], repeat[63:56].
  }];

  let arguments = (ins
    PTO_BufferType:$dst,
    PTO_BufferType:$src,
    I64:$offsetAddr,
    I64:$dstRepeatStride,
    I64:$repeat
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $dst `,` $src `,` $offsetAddr `,` $dstRepeatStride `,` $repeat
    attr-dict `:` type($dst) `,` type($src) `,` type($offsetAddr) `,`
    type($dstRepeatStride) `,` type($repeat)
  }];
}

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
