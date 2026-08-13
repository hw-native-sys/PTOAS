// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOControlOps - semantic operation family -*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_VPTOCONTROLOPS
#define MLIR_DIALECT_PTO_IR_VPTOCONTROLOPS

include "PTO/IR/VPTOOpCommon.td"

def TensorViewAddrOp : PTO_Op<"tensor_view_addr", [Pure]> {
  let summary = "Extract address from a tensor view.";
  let description = [{
    Returns the address view carried by a `tensor_view` or
    `partition_tensor_view` value. The result may be either a memref view or a
    typed PTO pointer, depending on the requested destination type.

    In authoring-form IR this op preserves the descriptor-style surface;
    tile-native inputs remain unchanged through memory planning and sync.
    This op may also accept a memref operand after earlier view lowering, in
    which case it behaves as an identity marker and is removed by lowering.
  }];

  let arguments = (ins AnyTypeOf<[TensorViewType, PartitionTensorViewType, AnyMemRef],
                                "TensorViewLike or MemRef">:$src);
  let results = (outs PtrOrMemRef:$dst);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src attr-dict `:` qualified(type($src)) `->` type($dst)
  }];
}

def TileBufAddrOp : PTO_Op<"tile_buf_addr", [Pure]> {
  let summary = "Extract address from a tile_buf.";
  let description = [{
    Returns the address view of the data region of a `tile_buf` value.
    The result may be either a memref view or a typed PTO pointer, depending
    on the requested destination type. Memref results use the tile's static
    shape and address space.

    This op is emitted by TileLang DSL templates and resolved by the
    FoldTileBufIntrinsics pass after inlining. Hand-written `.pto` may also
    use it directly on lowered memref/tile handles.
  }];

  let arguments = (ins AnyTypeOf<[TileBufType, AnyMemRef],
                                 "tile_buf or tile-bound memref">:$src);
  let results = (outs PtrOrMemRef:$dst);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src attr-dict `:` qualified(type($src)) `->` type($dst)
  }];
}

def VecScopeOp : PTO_Op<"vecscope", [SingleBlock, NoTerminator,
                                      VectorMicroOpInterface]> {
  let summary = "Structured region container for one VPTO vector scope";
  let description = [{
    `pto.vecscope` marks a structured vector-scope interval without overloading
    a dummy carrier loop with scope metadata. Lowering and emission passes may
    use the region boundary to preserve loop shape while treating the enclosed
    body as one VPTO vector interval.
  }];

  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let assemblyFormat = "$body attr-dict";
}

def StrictVecScopeOp : PTO_Op<"strict_vecscope", [SingleBlock, NoTerminator,
                                                   IsolatedFromAbove,
                                                   VectorMicroOpInterface]> {
  let summary = "Structured VPTO vector scope with explicit captures only";
  let description = [{
    `pto.strict_vecscope` is the strict form of `pto.vecscope`. Values used by
    the body must be passed explicitly through op operands and corresponding
    block arguments; implicit SSA capture from the surrounding scope is
    rejected.
  }];

  let arguments = (ins Variadic<AnyType>:$captures);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` $captures `)` $body attr-dict `:` functional-type($captures, results)
  }];
}

def PTO_MemBarOp : PTO_MicroOp<"mem_bar"> {
  let summary = "Low-level VPTO memory barrier";
  let description = [{
    Low-level memory ordering barrier that lowers to one of the
    `llvm.hivm.mem.bar.*` intrinsics exposed by Bisheng.
  }];

  let arguments = (ins PTO_MemBarAttr:$kind);
  let results = (outs);

  let hasCustomAssemblyFormat = 1;
}

def DsbOp : PTO_MicroOp<"dsb"> {
  let summary = "Issue a low-level DSB memory barrier";
  let description = [{
    Lowers to `llvm.hivm.DSB(i64)`. The memory domain maps directly to the CCE
    `mem_dsb_t` enum values: `ALL`, `DDR`, `UB`, and `SEQ`.
  }];

  let arguments = (ins PTO_DsbMemAttr:$mem);
  let results = (outs);

  let assemblyFormat = [{
    custom<LegacyOrAttrDsbMem>($mem) attr-dict
  }];
}

def DcciOp : PTO_MicroOp<"dcci"> {
  let summary = "Issue a low-level DCCI cache maintenance operation";
  let description = [{
    Lowers to the `llvm.hivm.DCCI*` intrinsic family. The pointer address space
    selects the intrinsic variant: GM pointers use `DCCI` / `DCCI.DST`; UB
    pointers use `DCCI.UB` / `DCCI.DST.UB`.
  }];

  let arguments = (ins
    PTO_BufferType:$ptr,
    PTO_DcciCacheLineAttr:$cache,
    OptionalAttr<PTO_DcciDstAttr>:$dst
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $ptr custom<LegacyOrAttrDcciCacheLine>($cache)
    custom<OptionalDcciDst>($dst) attr-dict `:` type($ptr)
  }];
}


class PTO_BinaryI64ConfigOp<string mnemonic> : PTO_Op<mnemonic> {
  let arguments = (ins
    I64:$first,
    I64:$second
  );

  let results = (outs);

  let assemblyFormat = [{
    $first `,` $second attr-dict `:` type($first) `,` type($second)
  }];
}

class PTO_BinaryI64PureOp<string mnemonic> : PTO_Op<mnemonic, [Pure]> {
  let arguments = (ins
    I64:$first,
    I64:$second
  );

  let results = (outs I64:$result);

  let assemblyFormat = [{
    $first `,` $second attr-dict `:` type($first) `,` type($second) `->` type($result)
  }];
}

class PTO_UnaryI64ConfigOp<string mnemonic> : PTO_Op<mnemonic> {
  let arguments = (ins I64:$value);
  let results = (outs);

  let assemblyFormat = [{
    $value attr-dict `:` type($value)
  }];
}

class PTO_NullaryI64PureOp<string mnemonic> : PTO_Op<mnemonic, [Pure]> {
  let arguments = (ins);
  let results = (outs I64:$result);

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

class PTO_NullaryI32PureOp<string mnemonic> : PTO_Op<mnemonic, [Pure]> {
  let arguments = (ins);
  let results = (outs I32:$result);

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

class PTO_NullaryConfigOp<string mnemonic> : PTO_Op<mnemonic> {
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = [{ attr-dict }];
}

def PTO_SetMovPadValOp : PTO_Op<"set_mov_pad_val"> {
  let arguments = (ins AnyTypeOf<[AnyInteger, AnyFloat],
                                "integer/float scalar">:$value);
  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $value attr-dict `:` type($value)
  }];
}

def PTO_SetLoop2StrideOutToUbOp : PTO_BinaryI64ConfigOp<"set_loop2_stride_outtoub">;
def PTO_SetLoop1StrideOutToUbOp : PTO_BinaryI64ConfigOp<"set_loop1_stride_outtoub">;
def PTO_SetLoopSizeOutToUbOp : PTO_BinaryI64ConfigOp<"set_loop_size_outtoub">;
def PTO_SetLoop2StrideOutToL1Op : PTO_UnaryI64ConfigOp<"set_loop2_stride_outtol1">;
def PTO_SetLoop1StrideOutToL1Op : PTO_UnaryI64ConfigOp<"set_loop1_stride_outtol1">;
def PTO_SetLoopSizeOutToL1Op : PTO_UnaryI64ConfigOp<"set_loop_size_outtol1">;
def PTO_SetLoop2StrideUbToOutOp : PTO_BinaryI64ConfigOp<"set_loop2_stride_ubtoout">;
def PTO_SetLoop1StrideUbToOutOp : PTO_BinaryI64ConfigOp<"set_loop1_stride_ubtoout">;
def PTO_SetLoopSizeUbToOutOp : PTO_BinaryI64ConfigOp<"set_loop_size_ubtoout">;
def PTO_SetLoop3ParaOp : PTO_BinaryI64ConfigOp<"set_loop3_para">;
def PTO_SetChannelParaOp : PTO_BinaryI64ConfigOp<"set_channel_para">;
def PTO_SetMte2NzParaOp : PTO_UnaryI64ConfigOp<"set_mte2_nz_para">;
def PTO_SetPadValOutToL1Op : PTO_UnaryI64ConfigOp<"set_pad_val_outtol1">;
// Reading CTRL is ordered with respect to set_ctrl. It must not be modeled as
// Pure, otherwise CSE can reuse a value read before an intervening write.
def PTO_GetCtrlOp : PTO_Op<"get_ctrl"> {
  let arguments = (ins);
  let results = (outs I64:$result);
  let assemblyFormat = [{ attr-dict `:` type($result) }];
}
def PTO_GetVms4SrOp : PTO_Op<"get_vms4_sr", [Pure]> {
  let arguments = (ins);
  let results = (outs I16:$list0, I16:$list1, I16:$list2, I16:$list3);

  let assemblyFormat = [{
    attr-dict `:` type($list0) `,` type($list1) `,` type($list2) `,` type($list3)
  }];
}
def PTO_SetCtrlOp : PTO_UnaryI64ConfigOp<"set_ctrl">;
def PTO_SetStoreAtomicCfgOp : PTO_UnaryI64ConfigOp<"set_store_atomic_cfg">;
def PTO_Sbitset0Op : PTO_BinaryI64PureOp<"sbitset0">;
def PTO_Sbitset1Op : PTO_BinaryI64PureOp<"sbitset1">;
def PTO_SetQuantPreOp : PTO_UnaryI64ConfigOp<"set_quant_pre">;
def PTO_SetReluAlphaOp : PTO_UnaryI64ConfigOp<"set_relu_alpha">;
def PTO_SetFixClipReluOp : PTO_UnaryI64ConfigOp<"set_fix_clip_relu">;
def PTO_SetFpcOp : PTO_UnaryI64ConfigOp<"set_fpc">;
def PTO_SetAtomicAddOp : PTO_NullaryConfigOp<"set_atomic_add">;
def PTO_SetAtomicMaxOp : PTO_NullaryConfigOp<"set_atomic_max">;
def PTO_SetAtomicMinOp : PTO_NullaryConfigOp<"set_atomic_min">;
def PTO_SetAtomicNoneOp : PTO_NullaryConfigOp<"set_atomic_none">;
def PTO_SetAtomicF32Op : PTO_NullaryConfigOp<"set_atomic_f32">;
def PTO_SetAtomicF16Op : PTO_NullaryConfigOp<"set_atomic_f16">;
def PTO_SetAtomicBF16Op : PTO_NullaryConfigOp<"set_atomic_bf16">;
def PTO_SetAtomicS32Op : PTO_NullaryConfigOp<"set_atomic_s32">;
def PTO_SetAtomicS16Op : PTO_NullaryConfigOp<"set_atomic_s16">;
def PTO_SetAtomicS8Op : PTO_NullaryConfigOp<"set_atomic_s8">;

#endif // MLIR_DIALECT_PTO_IR_VPTOCONTROLOPS
