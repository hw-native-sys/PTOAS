// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOOps.td - PTO low-level operations ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_VPTOOPS
#define MLIR_DIALECT_PTO_IR_VPTOOPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "PTO/IR/VPTOInterfaces.td"

def PTO_VectorType : Type<CPred<"::llvm::isa<::mlir::pto::VRegType>($_self)">,
                           "PTO low-level vector type">;
def PTO_MaskTypeConstraint : Type<CPred<"::llvm::isa<::mlir::pto::MaskType>($_self)">,
                                   "PTO low-level mask type">;
def PTO_B8MaskTypeConstraint : Type<
  CPred<"::llvm::isa<::mlir::pto::MaskType>($_self) && ::llvm::cast<::mlir::pto::MaskType>($_self).isB8()">,
  "PTO low-level b8 mask type">;
def PTO_B16MaskTypeConstraint : Type<
  CPred<"::llvm::isa<::mlir::pto::MaskType>($_self) && ::llvm::cast<::mlir::pto::MaskType>($_self).isB16()">,
  "PTO low-level b16 mask type">;
def PTO_B32MaskTypeConstraint : Type<
  CPred<"::llvm::isa<::mlir::pto::MaskType>($_self) && ::llvm::cast<::mlir::pto::MaskType>($_self).isB32()">,
  "PTO low-level b32 mask type">;
def PTO_AlignTypeConstraint : Type<CPred<"::llvm::isa<::mlir::pto::AlignType>($_self)">,
                                    "PTO low-level align type">;

def PTO_BufferType : Type<
  CPred<"::llvm::isa<::mlir::pto::PtrType>($_self)">,
  "pointer-like buffer type">;
def PTO_ScalingPtrType : Type<
  CPred<"::llvm::isa<::mlir::pto::PtrType>($_self) && "
        "::llvm::cast<::mlir::pto::PtrType>($_self).getMemorySpace().getAddressSpace() == ::mlir::pto::AddressSpace::SCALING">,
  "scaling pointer type">;
def PTO_BufferLikeType : AnyTypeOf<[AnyMemRef, PTO_BufferType],
                                    "memref or pointer-like buffer type">;
def PTO_AccStorePayloadType : AnyTypeOf<[AnySignlessInteger, AnyFloat, PTO_ScalingPtrType],
                                        "signless integer, float scalar, or scaling pointer">;
def PTO_AccStoreClipPayloadType : AnyTypeOf<[AnyInteger, AnyFloat],
                                            "integer or float clip scalar">;
def PTO_V2F16Type : Type<
  CPred<"::llvm::isa<::mlir::VectorType>($_self) && "
        "::llvm::cast<::mlir::VectorType>($_self).getRank() == 1 && "
        "::llvm::cast<::mlir::VectorType>($_self).getDimSize(0) == 2 && "
        "::llvm::cast<::mlir::VectorType>($_self).getElementType().isF16()">,
  "vector<2xf16>">;
def PTO_V2BF16Type : Type<
  CPred<"::llvm::isa<::mlir::VectorType>($_self) && "
        "::llvm::cast<::mlir::VectorType>($_self).getRank() == 1 && "
        "::llvm::cast<::mlir::VectorType>($_self).getDimSize(0) == 2 && "
        "::llvm::cast<::mlir::VectorType>($_self).getElementType().isBF16()">,
  "vector<2xbf16>">;
def PTO_V2F32Type : Type<
  CPred<"::llvm::isa<::mlir::VectorType>($_self) && "
        "::llvm::cast<::mlir::VectorType>($_self).getRank() == 1 && "
        "::llvm::cast<::mlir::VectorType>($_self).getDimSize(0) == 2 && "
        "::llvm::cast<::mlir::VectorType>($_self).getElementType().isF32()">,
  "vector<2xf32>">;
def PTO_V2F8E4M3FNType : Type<
  CPred<"::llvm::isa<::mlir::VectorType>($_self) && "
        "::llvm::cast<::mlir::VectorType>($_self).getRank() == 1 && "
        "::llvm::cast<::mlir::VectorType>($_self).getDimSize(0) == 2 && "
        "::llvm::isa<::mlir::Float8E4M3FNType>(::llvm::cast<::mlir::VectorType>($_self).getElementType())">,
  "vector<2xf8E4M3FN>">;
def PTO_V4F8E4M3FNType : Type<
  CPred<"::llvm::isa<::mlir::VectorType>($_self) && "
        "::llvm::cast<::mlir::VectorType>($_self).getRank() == 1 && "
        "::llvm::cast<::mlir::VectorType>($_self).getDimSize(0) == 4 && "
        "::llvm::isa<::mlir::Float8E4M3FNType>(::llvm::cast<::mlir::VectorType>($_self).getElementType())">,
  "vector<4xf8E4M3FN>">;
def PTO_V8F8E4M3FNType : Type<
  CPred<"::llvm::isa<::mlir::VectorType>($_self) && "
        "::llvm::cast<::mlir::VectorType>($_self).getRank() == 1 && "
        "::llvm::cast<::mlir::VectorType>($_self).getDimSize(0) == 8 && "
        "::llvm::isa<::mlir::Float8E4M3FNType>(::llvm::cast<::mlir::VectorType>($_self).getElementType())">,
  "vector<8xf8E4M3FN>">;
def PTO_V2F8E5M2Type : Type<
  CPred<"::llvm::isa<::mlir::VectorType>($_self) && "
        "::llvm::cast<::mlir::VectorType>($_self).getRank() == 1 && "
        "::llvm::cast<::mlir::VectorType>($_self).getDimSize(0) == 2 && "
        "::llvm::isa<::mlir::Float8E5M2Type>(::llvm::cast<::mlir::VectorType>($_self).getElementType())">,
  "vector<2xf8E5M2>">;
def PTO_V4F8E5M2Type : Type<
  CPred<"::llvm::isa<::mlir::VectorType>($_self) && "
        "::llvm::cast<::mlir::VectorType>($_self).getRank() == 1 && "
        "::llvm::cast<::mlir::VectorType>($_self).getDimSize(0) == 4 && "
        "::llvm::isa<::mlir::Float8E5M2Type>(::llvm::cast<::mlir::VectorType>($_self).getElementType())">,
  "vector<4xf8E5M2>">;
def PTO_V8F8E5M2Type : Type<
  CPred<"::llvm::isa<::mlir::VectorType>($_self) && "
        "::llvm::cast<::mlir::VectorType>($_self).getRank() == 1 && "
        "::llvm::cast<::mlir::VectorType>($_self).getDimSize(0) == 8 && "
        "::llvm::isa<::mlir::Float8E5M2Type>(::llvm::cast<::mlir::VectorType>($_self).getElementType())">,
  "vector<8xf8E5M2>">;
def PTO_V2HiF8Type : Type<
  CPred<"::llvm::isa<::mlir::pto::HiF8x2Type>($_self)">,
  "!pto.hif8x2">;
def PTO_LowPrecisionPackedType : AnyTypeOf<[PTO_V2F8E4M3FNType,
                                            PTO_V4F8E4M3FNType,
                                            PTO_V8F8E4M3FNType,
                                            PTO_V2F8E5M2Type,
                                            PTO_V4F8E5M2Type,
                                            PTO_V8F8E5M2Type,
                                            PTO_V2HiF8Type],
                                           "supported FP8/HiF8 packed type">;
def PTO_SimtF4ValueType : Type<
  CPred<"::llvm::isa<::mlir::pto::F4E1M2x2Type, ::mlir::pto::F4E2M1x2Type>($_self)">,
  "!pto.f4E1M2x2 or !pto.f4E2M1x2">;
def PTO_SimtShuffleValueType : AnyTypeOf<[I32, I64, F16, F32, PTO_V2F16Type],
                                         "i32, i64, f16, f32 or vector<2xf16>">;
def PTO_SimtUnaryFloatValueType : AnyTypeOf<[F16, F32, BF16, PTO_V2F16Type, PTO_V2BF16Type],
                                           "f16, f32, bf16, vector<2xf16> or vector<2xbf16>">;
def PTO_SimtUnaryF16F32ValueType : AnyTypeOf<[F16, F32, PTO_V2F16Type],
                                            "f16, f32 or vector<2xf16>">;
def PTO_SimtBinaryFloatValueType : AnyTypeOf<
  [F16, F32, BF16, PTO_V2F16Type, PTO_V2BF16Type],
  "f16, f32, bf16, vector<2xf16> or vector<2xbf16>">;
def PTO_SimtBinaryF16F32ValueType : AnyTypeOf<[F16, F32, PTO_V2F16Type],
                                             "f16, f32 or vector<2xf16>">;
def PTO_SimtConvertValueType : AnyTypeOf<[I32, I64, F16, BF16, F32,
                                          PTO_V2F16Type, PTO_V2BF16Type,
                                          PTO_V2F32Type, PTO_LowPrecisionPackedType,
                                          PTO_SimtF4ValueType],
                                         "standard scalar, low-precision scalar payload, or supported vector<2xT> conversion type">;
def PTO_AtomicValueType : AnyTypeOf<[I32, I64, F16, BF16, F32, PTO_V2F16Type, PTO_V2BF16Type],
                                    "i32, i64, f16, bf16, f32, vector<2xf16> or vector<2xbf16>">;
def PTO_DevScalarValueType : AnyTypeOf<[I8, I16, I32, I64],
                                       "i8, i16, i32 or i64">;

class PTO_MicroOp<string mnemonic, list<Trait> traits = []>
    : PTO_Op<mnemonic, !listconcat(traits, [MicroOpInterface,
                                            VPTOSchedulingOpInterface])>;

class PTO_VectorMicroOp<string mnemonic, list<Trait> traits = []>
    : PTO_MicroOp<mnemonic,
                  !listconcat(traits, [VectorMicroOpInterface])>;

class PTO_CubeMicroOp<string mnemonic, list<Trait> traits = []>
    : PTO_MicroOp<mnemonic,
                  !listconcat(traits, [CubeMicroOpInterface])>;

class PTO_MteOp<string mnemonic, list<Trait> traits = []>
    : PTO_MicroOp<mnemonic, !listconcat(traits, [MteOpInterface])>;

class PTO_SimtOp<string mnemonic, list<Trait> traits = []>
    : PTO_MicroOp<mnemonic, !listconcat(traits, [SimtOpInterface])>;

def PTOLoadOp : PTO_SimtOp<"load", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Load one scalar element from a VPTO pointer-like operand.";

  let arguments = (ins
    PTO_BufferLikeType:$ptr,
    Index:$offset
  );

  let results = (outs AnyType:$value);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $ptr `[` $offset `]` attr-dict `:` type($ptr) `->` type($value)
  }];
}

def PTOStoreOp : PTO_SimtOp<"store", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Store one scalar element to a VPTO pointer-like operand.";

  let arguments = (ins
    PTO_BufferLikeType:$ptr,
    Index:$offset,
    AnyType:$value
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $ptr `[` $offset `]` attr-dict `:` type($ptr) `,` type($value)
  }];
}

def PTOLdgOp : PTO_SimtOp<"ldg", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Load one element from GM with cache controls.";
  let description = [{
    Load a scalar or packed vector value from GM (Global Memory).

    Supported value types:
    - Scalar: i8/i16/i32/i64, f16/bf16/f32/f64, fp8/hif8
    - Packed: vector<2xf16>, vector<2xbf16>, vector<2xf32>,
      vector<2/4/8xf8E4M3FN>, vector<2/4/8xf8E5M2>,
      vector<2xi8>, vector<2xi16>, vector<2xi32>,
      and !pto.hif8x2

    The offset operand is an element-level ``index`` (not byte offset).
    For ``!pto.ptr<vector<2xf32>, gm>``, offset 1 advances by 8 bytes
    (2 × sizeof(f32)).  The effective address for ``vector<2xf32>`` must
    satisfy 8-byte alignment.
  }];

  let arguments = (ins
    PTO_BufferType:$ptr,
    Index:$offset,
    OptionalAttr<PTO_L1CacheAttr>:$l1cache,
    OptionalAttr<PTO_LdL2CacheAttr>:$l2cache
  );

  let results = (outs AnyType:$value);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $ptr `[` $offset `]` custom<L1Cache>($l1cache)
    custom<LdL2Cache>($l2cache) attr-dict `:` type($ptr) `->` type($value)
  }];
}

def PTOStgOp : PTO_SimtOp<"stg", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Store one element to GM with cache controls.";
  let description = [{
    Store a scalar or packed vector value to GM (Global Memory).

    Supported value types and offset/alignment semantics are the same as
    ``pto.ldg``.  See ``pto.ldg`` for details.
  }];

  let arguments = (ins
    PTO_BufferType:$ptr,
    Index:$offset,
    AnyType:$value,
    OptionalAttr<PTO_L1CacheAttr>:$l1cache,
    OptionalAttr<PTO_StL2CacheAttr>:$l2cache
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $ptr `[` $offset `]` custom<L1Cache>($l1cache)
    custom<StL2Cache>($l2cache) attr-dict `:` type($ptr) `,` type($value)
  }];
}

def PTOLdDevOp : PTO_MicroOp<"ld_dev", [
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Load one scalar GM element while bypassing the local L1 data cache.";
  let description = [{
    Load one integer scalar element from GM through the AICore device-memory
    path while bypassing the local L1 data cache.  The offset operand is an
    element offset, not a byte offset.  The operation does not provide
    atomicity, synchronization, memory-ordering, or L2 cache-policy semantics.
  }];

  let arguments = (ins
    PTO_BufferType:$ptr,
    Index:$offset
  );

  let results = (outs PTO_DevScalarValueType:$value);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $ptr `[` $offset `]` attr-dict `:` type($ptr) `->` type($value)
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_S; }
  }];
}

def PTOStDevOp : PTO_MicroOp<"st_dev", [
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Store one scalar GM element while bypassing the local L1 data cache.";
  let description = [{
    Store one integer scalar element to GM through the AICore device-memory
    path while bypassing the local L1 data cache.  The offset operand is an
    element offset, not a byte offset.  The operation does not provide
    atomicity, synchronization, memory-ordering, or L2 cache-policy semantics.
  }];

  let arguments = (ins
    PTO_DevScalarValueType:$value,
    PTO_BufferType:$ptr,
    Index:$offset
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $ptr `[` $offset `]` attr-dict `:` type($ptr) `,` type($value)
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_S; }
  }];
}

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
    body as one VPTO vector interval. Pipeline, system, cache, SIMT, and
    cross-core synchronization operations delimit vector scopes and must remain
    outside the region. `pto.mem_bar` is the exception: it orders vector memory
    operations within the region.
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
    rejected. It has the same synchronization boundary rules as `pto.vecscope`.
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

def DsbOp : PTO_Op<"dsb", [MicroOpInterface]> {
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

def DcciOp : PTO_Op<"dcci", [MicroOpInterface]> {
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

class PTO_UnaryI64ConfigOp<string mnemonic, list<Trait> traits = []>
    : PTO_Op<mnemonic, traits> {
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

class PTO_DirectSimtNullaryI64PureOp<string mnemonic>
    : PTO_SimtOp<mnemonic, [Pure]> {
  let arguments = (ins);
  let results = (outs I64:$result);

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

class PTO_DirectSimtNullaryI32PureOp<string mnemonic>
    : PTO_SimtOp<mnemonic, [Pure]> {
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
def PTO_GetCtrlOp : PTO_Op<"get_ctrl", [VPTOSchedulingOpInterface]> {
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
def PTO_SetCtrlOp : PTO_UnaryI64ConfigOp<
    "set_ctrl", [VPTOSchedulingOpInterface]>;
def PTO_SetStoreAtomicCfgOp : PTO_UnaryI64ConfigOp<"set_store_atomic_cfg">;
def PTO_StoreVfSimtInfoOp : PTO_SimtOp<"store_vfsimt_info"> {
  let summary = "Configure SIMT VF launch dimensions";
  let arguments = (ins I32:$dimZ, I32:$dimY, I32:$dimX);
  let results = (outs);
  let assemblyFormat = [{
    $dimZ `,` $dimY `,` $dimX attr-dict `:` type($dimZ) `,` type($dimY) `,` type($dimX)
  }];
}
def PTO_SimtLaunchOp : PTO_Op<"simt_launch"> {
  let summary = "Launch a SIMT entry with VF launch dimensions";
  let description = [{
    `pto.simt_launch` is source-level sugar for configuring SIMT VF launch
    dimensions and calling a function marked with `pto.simt_entry`.

    The textual dimension order follows the familiar launch order
    `<<<x, y, z>>>`; the canonical `pto.store_vfsimt_info` operation stores
    the same values in `z, y, x` order for the backend.
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       I32:$dimX,
                       I32:$dimY,
                       I32:$dimZ,
                       Variadic<AnyType>:$args);
  let results = (outs);
  let assemblyFormat = [{
    $callee `<` `<` `<` $dimX `,` $dimY `,` $dimZ `>` `>` `>`
    `` `(` $args `)` attr-dict `:` functional-type($args, results)
  }];
  let hasVerifier = 1;
}
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
def PTO_GetTidXOp : PTO_DirectSimtNullaryI32PureOp<"get_tid_x">;
def PTO_GetTidYOp : PTO_DirectSimtNullaryI32PureOp<"get_tid_y">;
def PTO_GetTidZOp : PTO_DirectSimtNullaryI32PureOp<"get_tid_z">;
def PTO_GetBlockDimXOp : PTO_DirectSimtNullaryI32PureOp<"get_block_dim_x">;
def PTO_GetBlockDimYOp : PTO_DirectSimtNullaryI32PureOp<"get_block_dim_y">;
def PTO_GetBlockDimZOp : PTO_DirectSimtNullaryI32PureOp<"get_block_dim_z">;
def PTO_GetGridDimXOp : PTO_DirectSimtNullaryI32PureOp<"get_grid_dim_x">;
def PTO_GetGridDimYOp : PTO_DirectSimtNullaryI32PureOp<"get_grid_dim_y">;
def PTO_GetGridDimZOp : PTO_DirectSimtNullaryI32PureOp<"get_grid_dim_z">;
def PTO_GetBlockIdxXOp : PTO_DirectSimtNullaryI32PureOp<"get_block_idx_x">;
def PTO_GetBlockIdxYOp : PTO_DirectSimtNullaryI32PureOp<"get_block_idx_y">;
def PTO_GetBlockIdxZOp : PTO_DirectSimtNullaryI32PureOp<"get_block_idx_z">;
def PTO_GetVecCoreIdOp : PTO_DirectSimtNullaryI32PureOp<"get_veccoreid">;
def PTO_GetLaneIdOp : PTO_DirectSimtNullaryI32PureOp<"get_laneid">;
def PTO_GetClock32Op : PTO_DirectSimtNullaryI32PureOp<"get_clock32">;
def PTO_GetClock64Op : PTO_DirectSimtNullaryI64PureOp<"get_clock64">;
def PTO_GetLaneMaskEqOp : PTO_DirectSimtNullaryI32PureOp<"get_lanemask_eq">;
def PTO_GetLaneMaskLeOp : PTO_DirectSimtNullaryI32PureOp<"get_lanemask_le">;
def PTO_GetLaneMaskLtOp : PTO_DirectSimtNullaryI32PureOp<"get_lanemask_lt">;
def PTO_GetLaneMaskGeOp : PTO_DirectSimtNullaryI32PureOp<"get_lanemask_ge">;
def PTO_GetLaneMaskGtOp : PTO_DirectSimtNullaryI32PureOp<"get_lanemask_gt">;

def PTO_VoteAllOp : PTO_SimtOp<"vote_all", [Pure]> {
  let arguments = (ins I1:$pred);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $pred attr-dict `:` type($pred) `->` type($result)
  }];
}

def PTO_VoteAnyOp : PTO_SimtOp<"vote_any", [Pure]> {
  let arguments = (ins I1:$pred);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $pred attr-dict `:` type($pred) `->` type($result)
  }];
}

def PTO_VoteUniOp : PTO_SimtOp<"vote_uni", [Pure]> {
  let arguments = (ins I1:$pred);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $pred attr-dict `:` type($pred) `->` type($result)
  }];
}

def PTO_VoteBallotOp : PTO_SimtOp<"vote_ballot", [Pure]> {
  let arguments = (ins I1:$pred);
  let results = (outs I32:$result);

  let assemblyFormat = [{
    $pred attr-dict `:` type($pred) `->` type($result)
  }];
}

def PTO_ShuffleIdxOp : PTO_SimtOp<"shuffle_idx", [Pure, AllTypesMatch<["value", "result"]>]> {
  let arguments = (ins
    PTO_SimtShuffleValueType:$value,
    I32:$index,
    DefaultValuedAttr<I32Attr, "32">:$width
  );
  let results = (outs PTO_SimtShuffleValueType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $value `,` $index attr-dict `:` type($value) `,` type($index) `->` type($result)
  }];
}

def PTO_ShuffleUpOp : PTO_SimtOp<"shuffle_up", [Pure, AllTypesMatch<["value", "result"]>]> {
  let arguments = (ins
    PTO_SimtShuffleValueType:$value,
    I32:$offset,
    DefaultValuedAttr<I32Attr, "32">:$width
  );
  let results = (outs PTO_SimtShuffleValueType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $value `,` $offset attr-dict `:` type($value) `,` type($offset) `->` type($result)
  }];
}

def PTO_ShuffleDownOp : PTO_SimtOp<"shuffle_down", [Pure, AllTypesMatch<["value", "result"]>]> {
  let arguments = (ins
    PTO_SimtShuffleValueType:$value,
    I32:$offset,
    DefaultValuedAttr<I32Attr, "32">:$width
  );
  let results = (outs PTO_SimtShuffleValueType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $value `,` $offset attr-dict `:` type($value) `,` type($offset) `->` type($result)
  }];
}

def PTO_ShuffleBflyOp : PTO_SimtOp<"shuffle_bfly", [Pure, AllTypesMatch<["value", "result"]>]> {
  let arguments = (ins
    PTO_SimtShuffleValueType:$value,
    I32:$mask,
    DefaultValuedAttr<I32Attr, "32">:$width
  );
  let results = (outs PTO_SimtShuffleValueType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $value `,` $mask attr-dict `:` type($value) `,` type($mask) `->` type($result)
  }];
}

def PTO_ReduxAddOp : PTO_SimtOp<"redux_add", [Pure, AllTypesMatch<["value", "result"]>]> {
  let arguments = (ins
    AnyTypeOf<[I32, F16, F32], "integer/float scalar">:$value,
    OptionalAttr<PTO_SignednessAttr>:$signedness
  );
  let results = (outs AnyTypeOf<[I32, F16, F32], "integer/float scalar">:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $value (custom<OptionalSignedness>($signedness)^)? attr-dict
    `:` type($value) `->` type($result)
  }];
}

def PTO_ReduxMaxOp : PTO_SimtOp<"redux_max", [Pure, AllTypesMatch<["value", "result"]>]> {
  let arguments = (ins
    AnyTypeOf<[I32, F16, F32], "integer/float scalar">:$value,
    OptionalAttr<PTO_SignednessAttr>:$signedness
  );
  let results = (outs AnyTypeOf<[I32, F16, F32], "integer/float scalar">:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $value (custom<OptionalSignedness>($signedness)^)? attr-dict
    `:` type($value) `->` type($result)
  }];
}

def PTO_ReduxMinOp : PTO_SimtOp<"redux_min", [Pure, AllTypesMatch<["value", "result"]>]> {
  let arguments = (ins
    AnyTypeOf<[I32, F16, F32], "integer/float scalar">:$value,
    OptionalAttr<PTO_SignednessAttr>:$signedness
  );
  let results = (outs AnyTypeOf<[I32, F16, F32], "integer/float scalar">:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $value (custom<OptionalSignedness>($signedness)^)? attr-dict
    `:` type($value) `->` type($result)
  }];
}

class PTO_AtomicBinaryOp<string mnemonic>
    : PTO_SimtOp<mnemonic, [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins
    PTO_BufferType:$ptr,
    PTO_AtomicValueType:$value,
    OptionalAttr<PTO_StL2CacheAttr>:$l2cache,
    OptionalAttr<PTO_SignednessAttr>:$signedness
  );
  let results = (outs PTO_AtomicValueType:$old);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $ptr `,` $value custom<StL2Cache>($l2cache)
    (custom<OptionalSignedness>($signedness)^)? attr-dict
    `:` type($ptr) `,` type($value) `->` type($old)
  }];
}

class PTO_AtomicCasLikeOp<string mnemonic>
    : PTO_SimtOp<mnemonic, [
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        AllTypesMatch<["compare", "value", "old"]>
      ]> {
  let arguments = (ins
    PTO_BufferType:$ptr,
    PTO_AtomicValueType:$compare,
    PTO_AtomicValueType:$value,
    OptionalAttr<PTO_StL2CacheAttr>:$l2cache,
    OptionalAttr<PTO_SignednessAttr>:$signedness
  );
  let results = (outs PTO_AtomicValueType:$old);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $ptr `,` $compare `,` $value custom<StL2Cache>($l2cache)
    (custom<OptionalSignedness>($signedness)^)? attr-dict
    `:` type($ptr) `,` type($value) `->` type($old)
  }];
}

def PTO_AtomicCasOp : PTO_AtomicCasLikeOp<"atomic_cas">;
def PTO_AtomicExchOp : PTO_AtomicBinaryOp<"atomic_exch">;
def PTO_AtomicAddOp : PTO_AtomicBinaryOp<"atomic_add">;
def PTO_AtomicSubOp : PTO_AtomicBinaryOp<"atomic_sub">;
def PTO_AtomicMinOp : PTO_AtomicBinaryOp<"atomic_min">;
def PTO_AtomicMaxOp : PTO_AtomicBinaryOp<"atomic_max">;
def PTO_AtomicAndOp : PTO_AtomicBinaryOp<"atomic_and">;
def PTO_AtomicOrOp : PTO_AtomicBinaryOp<"atomic_or">;
def PTO_AtomicXorOp : PTO_AtomicBinaryOp<"atomic_xor">;

def PTO_PrmtOp : PTO_SimtOp<"prmt", [Pure]> {
  let arguments = (ins I32:$lhs, I32:$rhs, I32:$selector);
  let results = (outs I32:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $selector attr-dict `:` type($lhs) `,` type($rhs) `,` type($selector) `->` type($result)
  }];
}

def PTO_MulhiOp : PTO_SimtOp<"mulhi", [
    Pure,
    AllTypesMatch<["lhs", "rhs", "result"]>
  ]> {
  let arguments = (ins
    AnyTypeOf<[I32, I64], "i32 or i64">:$lhs,
    AnyTypeOf<[I32, I64], "i32 or i64">:$rhs,
    PTO_SignednessAttr:$signedness
  );
  let results = (outs AnyTypeOf<[I32, I64], "i32 or i64">:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs custom<Signedness>($signedness) attr-dict
    `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def PTO_MulI32ToI64Op : PTO_SimtOp<"mul_i32toi64", [Pure]> {
  let arguments = (ins
    I32:$lhs,
    I32:$rhs,
    PTO_SignednessAttr:$signedness
  );
  let results = (outs I64:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs custom<Signedness>($signedness) attr-dict
    `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def PTO_SqrtOp
    : PTO_SimtOp<"sqrt", [Pure, AllTypesMatch<["value", "result"]>]> {
  let arguments = (ins PTO_SimtUnaryF16F32ValueType:$value);
  let results = (outs PTO_SimtUnaryF16F32ValueType:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result)
  }];
}

class PTO_UnaryFloatScalarOp<string mnemonic>
    : PTO_SimtOp<mnemonic, [Pure, AllTypesMatch<["value", "result"]>]> {
  let arguments = (ins AnyTypeOf<[F16, F32, BF16], "floating scalar">:$value);
  let results = (outs AnyTypeOf<[F16, F32, BF16], "floating scalar">:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result)
  }];
}

class PTO_BinaryFloatScalarOp<string mnemonic>
    : PTO_SimtOp<mnemonic, [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let arguments = (ins
    AnyTypeOf<[F16, F32, BF16], "floating scalar">:$lhs,
    AnyTypeOf<[F16, F32, BF16], "floating scalar">:$rhs
  );
  let results = (outs AnyTypeOf<[F16, F32, BF16], "floating scalar">:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def PTO_AbsFOp
    : PTO_SimtOp<"absf", [Pure, AllTypesMatch<["value", "result"]>]> {
  let arguments = (ins AnyTypeOf<[F16, F32, PTO_V2F16Type, PTO_V2BF16Type],
                                 "f16, f32, vector<2xf16> or vector<2xbf16>">:$value);
  let results = (outs AnyTypeOf<[F16, F32, PTO_V2F16Type, PTO_V2BF16Type],
                                "f16, f32, vector<2xf16> or vector<2xbf16>">:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result)
  }];
}

class PTO_UnaryF16F32ScalarOp<string mnemonic>
    : PTO_SimtOp<mnemonic, [Pure, AllTypesMatch<["value", "result"]>]> {
  let arguments = (ins PTO_SimtUnaryF16F32ValueType:$value);
  let results = (outs PTO_SimtUnaryF16F32ValueType:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result)
  }];
}

class PTO_UnaryF32ScalarOp<string mnemonic>
    : PTO_SimtOp<mnemonic, [Pure, AllTypesMatch<["value", "result"]>]> {
  let arguments = (ins F32:$value);
  let results = (outs F32:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result)
  }];
}

def PTO_ExpOp : PTO_UnaryF16F32ScalarOp<"exp">;
def PTO_LogOp : PTO_UnaryF16F32ScalarOp<"log">;
def PTO_SinOp : PTO_UnaryF32ScalarOp<"sin">;
def PTO_CosOp : PTO_UnaryF32ScalarOp<"cos">;
def PTO_CeilOp : PTO_UnaryFloatScalarOp<"ceil">;
def PTO_FloorOp : PTO_UnaryFloatScalarOp<"floor">;
def PTO_RintOp : PTO_UnaryFloatScalarOp<"rint">;
def PTO_RoundOp : PTO_UnaryFloatScalarOp<"round">;

class PTO_BinaryFloatValueOp<string mnemonic>
    : PTO_SimtOp<mnemonic, [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let arguments = (ins
    PTO_SimtBinaryFloatValueType:$lhs,
    PTO_SimtBinaryFloatValueType:$rhs
  );
  let results = (outs PTO_SimtBinaryFloatValueType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

class PTO_BinaryF16F32ScalarOp<string mnemonic>
    : PTO_SimtOp<mnemonic, [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let arguments = (ins
    PTO_SimtBinaryF16F32ValueType:$lhs,
    PTO_SimtBinaryF16F32ValueType:$rhs
  );
  let results = (outs PTO_SimtBinaryF16F32ValueType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def PTO_FMinOp : PTO_BinaryFloatValueOp<"fmin">;
def PTO_FMaxOp : PTO_BinaryFloatValueOp<"fmax">;
def PTO_PowOp : PTO_BinaryF16F32ScalarOp<"pow">;

def PTO_FmaOp
    : PTO_SimtOp<"fma", [Pure, AllTypesMatch<["lhs", "rhs", "acc", "result"]>]> {
  let arguments = (ins
    PTO_SimtUnaryFloatValueType:$lhs,
    PTO_SimtUnaryFloatValueType:$rhs,
    PTO_SimtUnaryFloatValueType:$acc
  );
  let results = (outs PTO_SimtUnaryFloatValueType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $acc attr-dict `:` type($lhs) `,` type($rhs) `,` type($acc) `->` type($result)
  }];
}

def PTO_ConvertOp : PTO_SimtOp<"convert", [Pure]> {
  let arguments = (ins
    PTO_SimtConvertValueType:$src,
    PTO_RoundingAttr:$rounding,
    PTO_SaturationAttr:$saturation,
    OptionalAttr<PTO_SignednessAttr>:$signedness
  );
  let results = (outs PTO_SimtConvertValueType:$dst);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $src custom<ConvertRounding>($rounding)
    custom<ConvertSaturation>($saturation)
    (custom<OptionalSignedness>($signedness)^)? attr-dict
    `:` type($src) `->` type($dst)
  }];
}

def PTO_SyncthreadsOp : PTO_Op<"syncthreads", [
    MicroOpInterface,
    SimtOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Synchronize all workitems in a SIMT VF entry";
  let arguments = (ins);
  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{ attr-dict }];
}

def PTO_ThreadfenceOp : PTO_Op<"threadfence", [
    MicroOpInterface,
    SimtOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Issue a SIMT workitem memory fence";
  let arguments = (ins);
  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{ attr-dict }];
}

def PTO_ThreadfenceBlockOp : PTO_Op<"threadfence_block", [
    MicroOpInterface,
    SimtOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Issue a SIMT block-scoped workitem memory fence";
  let arguments = (ins);
  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{ attr-dict }];
}

def PTO_KeepOp : PTO_SimtOp<"keep"> {
  let summary = "Preserve one scalar payload across adjacent SIMT VF entries";
  let arguments = (ins AnyType:$payload, I64Attr:$slot);
  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $payload attr-dict `:` type($payload)
  }];
}

def PTO_ResumeOp : PTO_SimtOp<"resume"> {
  let summary = "Restore one scalar payload preserved by a previous SIMT VF entry";
  let arguments = (ins I64Attr:$slot);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

def PTO_CopyGmToUbufOp : PTO_MteOp<"copy_gm_to_ubuf", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    I64:$sid,
    I64:$n_burst,
    I64:$len_burst,
    I64:$left_padding_count,
    I64:$right_padding_count,
    I1:$data_select_bit,
    I64:$l2_cache_ctl,
    I64:$gm_stride,
    I64:$ub_stride
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $sid `,` $n_burst `,` $len_burst `,`
    $left_padding_count `,` $right_padding_count `,` $data_select_bit `,` $l2_cache_ctl `,` $gm_stride `,` $ub_stride
    attr-dict `:` type($source) `,` type($destination) `,` type($sid) `,` type($n_burst) `,`
    type($len_burst) `,` type($left_padding_count) `,`
    type($right_padding_count) `,` type($data_select_bit) `,` type($l2_cache_ctl) `,` type($gm_stride) `,` type($ub_stride)
  }];
}

def PTO_MteGmUbOp : PTO_MteOp<"mte_gm_ub", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    I64:$l2_cache_ctl,
    I64:$len_burst,
    I64:$n_burst,
    I64:$nburst_src_stride,
    I64:$nburst_dst_stride,
    Variadic<I64>:$loop_counts,
    Variadic<I64>:$loop_src_strides,
    Variadic<I64>:$loop_dst_strides,
    Optional<AnyTypeOf<[AnyInteger, AnyFloat], "integer/float scalar">>:$pad_value,
    Optional<I64>:$left_padding_count,
    Optional<I64>:$right_padding_count
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::Value":$destination,
      "::mlir::Value":$l2CacheCtl,
      "::mlir::Value":$lenBurst,
      "::mlir::pto::DmaLoopConfig":$nburst,
      "::llvm::ArrayRef<::mlir::pto::DmaLoopConfig>":$loops,
      "::std::optional<::mlir::pto::DmaPadConfig>":$pad
    )>,
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::Value":$destination,
      "::mlir::Value":$l2CacheCtl,
      "::mlir::Value":$lenBurst,
      "::mlir::pto::DmaLoopConfig":$nburst,
      "::std::optional<::mlir::pto::DmaLoopConfig>":$loop1,
      "::std::optional<::mlir::pto::DmaLoopConfig>":$loop2,
      "::std::optional<::mlir::pto::DmaPadConfig>":$pad
    )>
  ];

}

def PTO_CopyUbufToUbufOp : PTO_MteOp<"copy_ubuf_to_ubuf"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$sid,
    I64:$n_burst,
    I64:$len_burst,
    I64:$src_stride,
    I64:$dst_stride
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $sid `,` $n_burst `,` $len_burst `,` $src_stride `,` $dst_stride
    attr-dict `:` type($source) `,` type($destination) `,` type($sid) `,` type($n_burst) `,`
    type($len_burst) `,` type($src_stride) `,` type($dst_stride)
  }];
}

def PTO_CopyCbufToUbufOp : PTO_MteOp<"copy_cbuf_to_ubuf", [
    CubeMicroOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$sid,
    I64:$n_burst,
    I64:$len_burst,
    I64:$src_stride,
    I64:$dst_stride
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $sid `,` $n_burst `,` $len_burst `,` $src_stride `,` $dst_stride
    attr-dict `:` type($source) `,` type($destination) `,` type($sid) `,` type($n_burst) `,`
    type($len_burst) `,` type($src_stride) `,` type($dst_stride)
  }];
}

def PTO_CopyUbufToCbufOp : PTO_MteOp<"copy_ubuf_to_cbuf", [
    CubeMicroOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$sid,
    I64:$n_burst,
    I64:$len_burst,
    I64:$src_stride,
    I64:$dst_stride
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $sid `,` $n_burst `,` $len_burst `,` $src_stride `,` $dst_stride
    attr-dict `:` type($source) `,` type($destination) `,` type($sid) `,` type($n_burst) `,`
    type($len_burst) `,` type($src_stride) `,` type($dst_stride)
  }];
}

def PTO_MteUbUbOp : PTO_MteOp<"mte_ub_ub", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$n_burst,
    I64:$len_burst,
    I64:$src_stride,
    I64:$dst_stride
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $len_burst
    `nburst` `(` $n_burst `,` $src_stride `,` $dst_stride `)`
    attr-dict `:` type($source) `,` type($destination) `,` type($n_burst) `,`
    type($len_burst) `,` type($src_stride) `,`
    type($dst_stride)
  }];
}

def PTO_MteUbL1Op : PTO_MteOp<"mte_ub_l1", [
    CubeMicroOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$n_burst,
    I64:$len_burst,
    I64:$src_stride,
    I64:$dst_stride
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $len_burst
    `nburst` `(` $n_burst `,` $src_stride `,` $dst_stride `)`
    attr-dict `:` type($source) `,` type($destination) `,` type($n_burst) `,`
    type($len_burst) `,` type($src_stride) `,`
    type($dst_stride)
  }];
}

def PTO_CopyGmToCbufOp : PTO_MteOp<"copy_gm_to_cbuf"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$n_burst,
    I64:$len_burst,
    I64:$src_stride,
    I64:$dst_stride
  );

  let results = (outs);

  let assemblyFormat = [{
    $source `,` $destination `,` $n_burst `,` $len_burst `,` $src_stride `,` $dst_stride
    attr-dict `:` type($source) `,` type($destination) `,` type($n_burst) `,` type($len_burst) `,` type($src_stride) `,` type($dst_stride)
  }];
}

def PTO_CopyGmToCbufMultiNd2NzOp : PTO_MteOp<"copy_gm_to_cbuf_multi_nd2nz"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$sid,
    I64:$loop1_src_stride,
    I64:$l2_cache_ctrl,
    I64:$n_value,
    I64:$d_value,
    I64:$loop4_src_stride,
    I1:$smallc0_en
  );

  let results = (outs);

  let assemblyFormat = [{
    $source `,` $destination `,` $sid `,` $loop1_src_stride `,` $l2_cache_ctrl `,`
    $n_value `,` $d_value `,` $loop4_src_stride `,` $smallc0_en
    attr-dict `:` type($source) `,` type($destination) `,` type($sid) `,`
    type($loop1_src_stride) `,` type($l2_cache_ctrl) `,` type($n_value) `,`
    type($d_value) `,` type($loop4_src_stride) `,` type($smallc0_en)
  }];
}

def PTO_CopyGmToCbufMultiDn2NzOp : PTO_MteOp<"copy_gm_to_cbuf_multi_dn2nz"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$sid,
    I64:$loop1_src_stride,
    I64:$l2_cache_ctrl,
    I64:$n_value,
    I64:$d_value,
    I64:$loop4_src_stride,
    I1:$smallc0_en
  );

  let results = (outs);

  let assemblyFormat = [{
    $source `,` $destination `,` $sid `,` $loop1_src_stride `,` $l2_cache_ctrl `,`
    $n_value `,` $d_value `,` $loop4_src_stride `,` $smallc0_en
    attr-dict `:` type($source) `,` type($destination) `,` type($sid) `,`
    type($loop1_src_stride) `,` type($l2_cache_ctrl) `,` type($n_value) `,`
    type($d_value) `,` type($loop4_src_stride) `,` type($smallc0_en)
  }];
}

def PTO_CopyCbufToBtOp : PTO_MteOp<"copy_cbuf_to_bt"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I1:$conv_control,
    I64:$n_burst,
    I64:$len_burst,
    I64:$source_gap,
    I64:$dst_gap
  );
  let results = (outs);

  let assemblyFormat = [{
    $source `,` $destination `,` $conv_control `,` $n_burst `,` $len_burst `,`
    $source_gap `,` $dst_gap
    attr-dict `:` type($source) `,` type($destination) `,` type($conv_control) `,`
    type($n_burst) `,` type($len_burst) `,` type($source_gap) `,` type($dst_gap)
  }];
}

def PTO_CopyCbufToFbufOp : PTO_MteOp<"copy_cbuf_to_fbuf"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$n_burst,
    I64:$len_burst,
    I64:$source_gap,
    I64:$dst_gap
  );
  let results = (outs);

  let assemblyFormat = [{
    $source `,` $destination `,` $n_burst `,` $len_burst `,`
    $source_gap `,` $dst_gap
    attr-dict `:` type($source) `,` type($destination) `,` type($n_burst) `,`
    type($len_burst) `,` type($source_gap) `,` type($dst_gap)
  }];
}

def PTO_LoadCbufToCaOp : PTO_MteOp<"load_cbuf_to_ca"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$m_start,
    I64:$k_start,
    I64:$m_step,
    I64:$k_step,
    I64:$src_stride,
    I64:$dst_stride,
    DefaultValuedAttr<BoolAttr, "false">:$transpose
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $m_start `,` $k_start `,` $m_step `,`
    $k_step `,` $src_stride `,` $dst_stride attr-dict `:` type($source) `,`
    type($destination) `,` type($m_start) `,` type($k_start) `,`
    type($m_step) `,` type($k_step) `,` type($src_stride) `,`
    type($dst_stride)
  }];
}

def PTO_LoadCbufToCaS4Op : PTO_MteOp<"load_cbuf_to_ca_s4"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$m_start,
    I64:$k_start,
    I64:$m_step,
    I64:$k_step,
    I64:$src_stride,
    I64:$dst_stride,
    I64:$transpose
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $m_start `,` $k_start `,` $m_step `,`
    $k_step `,` $src_stride `,` $dst_stride `,` $transpose
    attr-dict `:` type($source) `,` type($destination) `,` type($m_start) `,`
    type($k_start) `,` type($m_step) `,` type($k_step) `,` type($src_stride) `,`
    type($dst_stride) `,` type($transpose)
  }];
}

def PTO_LoadCbufToCbOp : PTO_MteOp<"load_cbuf_to_cb"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$m_start,
    I64:$k_start,
    I64:$m_step,
    I64:$k_step,
    I64:$src_stride,
    I64:$dst_stride,
    DefaultValuedAttr<BoolAttr, "false">:$transpose
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $m_start `,` $k_start `,` $m_step `,`
    $k_step `,` $src_stride `,` $dst_stride attr-dict `:` type($source) `,`
    type($destination) `,` type($m_start) `,` type($k_start) `,`
    type($m_step) `,` type($k_step) `,` type($src_stride) `,`
    type($dst_stride)
  }];
}

def PTO_LoadCbufToCbS4Op : PTO_MteOp<"load_cbuf_to_cb_s4"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$m_start,
    I64:$k_start,
    I64:$m_step,
    I64:$k_step,
    I64:$src_stride,
    I64:$dst_stride,
    I64:$transpose
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $m_start `,` $k_start `,` $m_step `,`
    $k_step `,` $src_stride `,` $dst_stride `,` $transpose
    attr-dict `:` type($source) `,` type($destination) `,` type($m_start) `,`
    type($k_start) `,` type($m_step) `,` type($k_step) `,` type($src_stride) `,`
    type($dst_stride) `,` type($transpose)
  }];
}

def PTO_LoadCbufToCaMxOp : PTO_MteOp<"load_cbuf_to_ca_mx"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$x_start_position,
    I64:$y_start_position,
    I64:$x_step,
    I64:$y_step,
    I64:$src_stride,
    I64:$dst_stride
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $x_start_position `,` $y_start_position `,`
    $x_step `,` $y_step `,` $src_stride `,` $dst_stride attr-dict `:`
    type($source) `,` type($destination) `,` type($x_start_position) `,`
    type($y_start_position) `,` type($x_step) `,` type($y_step) `,`
    type($src_stride) `,` type($dst_stride)
  }];
}

def PTO_LoadCbufToCbMxOp : PTO_MteOp<"load_cbuf_to_cb_mx"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$x_start_position,
    I64:$y_start_position,
    I64:$x_step,
    I64:$y_step,
    I64:$src_stride,
    I64:$dst_stride
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $x_start_position `,` $y_start_position `,`
    $x_step `,` $y_step `,` $src_stride `,` $dst_stride attr-dict `:`
    type($source) `,` type($destination) `,` type($x_start_position) `,`
    type($y_start_position) `,` type($x_step) `,` type($y_step) `,`
    type($src_stride) `,` type($dst_stride)
  }];
}

def PTO_CopyMatrixCcToGmOp : PTO_MteOp<"copy_matrix_cc_to_gm"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$xm,
    I64:$xt
  );

  let results = (outs);

  let assemblyFormat = [{
    $source `,` $destination `,` $xm `,` $xt attr-dict `:` type($source) `,` type($destination) `,` type($xm) `,` type($xt)
  }];
}

def PTO_CopyMatrixCcToCbufOp : PTO_MteOp<"copy_matrix_cc_to_cbuf"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$config0,
    I64:$config1
  );

  let results = (outs);

  let assemblyFormat = [{
    $source `,` $destination `,` $config0 `,` $config1 attr-dict `:` type($source) `,`
    type($destination) `,` type($config0) `,` type($config1)
  }];
}

def PTO_CopyMatrixCcToUbOp : PTO_MteOp<"copy_matrix_cc_to_ub"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$config0,
    I64:$config1
  );

  let results = (outs);

  let assemblyFormat = [{
    $source `,` $destination `,` $config0 `,` $config1 attr-dict `:` type($source) `,`
    type($destination) `,` type($config0) `,` type($config1)
  }];
}

def PTO_VldsOp : PTO_VectorMicroOp<"vlds", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    VPTOAddressSemanticsOpInterface
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    Index:$offset,
    OptionalAttr<StrAttr>:$dist
  );

  let results = (outs PTO_VectorType:$result,
                      Optional<PTO_BufferLikeType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $offset `]` attr-dict `:` type($source) `->` type($result)
    (`,` type($updated_base)^)?
  }];
}

def PTO_Vldsx2Op : PTO_VectorMicroOp<"vldsx2", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    VPTOAddressSemanticsOpInterface
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    Index:$offset,
    StrAttr:$dist
  );
  let results = (outs PTO_VectorType:$low,
                      PTO_VectorType:$high,
                      Optional<PTO_BufferLikeType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $offset `]` `,` $dist attr-dict `:` type($source) `,` type($offset) `->` type($low) `,` type($high)
    (`,` type($updated_base)^)?
  }];
}

def PTO_VldasOp : PTO_VectorMicroOp<"vldas", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$source
  );

  let results = (outs PTO_AlignTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` type($result)
  }];
}

def PTO_InitAlignOp : PTO_VectorMicroOp<"init_align", []> {
  let arguments = (ins);

  let results = (outs PTO_AlignTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

def PTO_SprclrOp : PTO_VectorMicroOp<"sprclr", []> {
  let arguments = (ins StrAttr:$spr);

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $spr attr-dict
  }];
}

def PTO_SprstiOp : PTO_VectorMicroOp<"sprsti", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    VPTOAddressSemanticsOpInterface
  ]> {
  let arguments = (ins
    StrAttr:$spr,
    PTO_BufferType:$destination,
    I32:$offset
  );

  let results = (outs Optional<PTO_BufferType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $spr `,` $destination `[` $offset `]` attr-dict `:` type($destination) `,` type($offset)
    (`->` type($updated_base)^)?
  }];
}

def PTO_SprstsOp : PTO_VectorMicroOp<"sprsts", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    VPTOAddressSemanticsOpInterface
  ]> {
  let arguments = (ins
    StrAttr:$spr,
    PTO_BufferType:$destination,
    I32:$offset
  );

  let results = (outs Optional<PTO_BufferType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $spr `,` $destination `[` $offset `]` attr-dict `:` type($destination) `,` type($offset)
    (`->` type($updated_base)^)?
  }];
}

def PTO_VldusOp : PTO_VectorMicroOp<"vldus", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    VPTOAddressSemanticsOpInterface
  ]> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_AlignTypeConstraint:$align,
    Optional<Index>:$increment
  );

  let results = (outs
    PTO_VectorType:$result,
    PTO_AlignTypeConstraint:$updated_align,
    Optional<PTO_BufferType>:$updated_base
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $align (`,` $increment^)? attr-dict `:` type($source) `,` type($align)
    (`,` type($increment)^)? `->` type($result) `,` type($updated_align)
    (`,` type($updated_base)^)?
  }];
}

def PTO_UvldOp : PTO_VectorMicroOp<"uvld", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    Index:$offset
  );

  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $offset `]` attr-dict `:` type($source) `->` type($result)
  }];
}

def PTO_VbrOp : PTO_VectorMicroOp<"vbr", [Pure]> {
  let arguments = (ins AnyType:$value);
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result)
  }];
}

def PTO_VdupOp : PTO_VectorMicroOp<"vdup", [Pure]> {
  let arguments = (ins
    AnyType:$input,
    PTO_MaskTypeConstraint:$mask,
    OptionalAttr<StrAttr>:$position
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $mask attr-dict `:` type($input) `,` type($mask) `->` type($result)
  }];
}

def PTO_PsetB8Op : PTO_VectorMicroOp<"pset_b8", [Pure]> {
  let arguments = (ins StrAttr:$pattern);
  let results = (outs PTO_B8MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $pattern attr-dict `:` type($result)
  }];
}

def PTO_PsetB16Op : PTO_VectorMicroOp<"pset_b16", [Pure]> {
  let arguments = (ins StrAttr:$pattern);
  let results = (outs PTO_B16MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $pattern attr-dict `:` type($result)
  }];
}

// NOTE: The op families introduced below are intentionally marked as
// unvalidated scaffolding. They are added to preserve missing CCE builtin
// semantics at the dialect layer, but they have not yet been validated through
// PTO lowering or end-to-end sample execution.
def PTO_PsetB32Op : PTO_VectorMicroOp<"pset_b32", [Pure]> {
  let arguments = (ins StrAttr:$pattern);
  let results = (outs PTO_B32MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $pattern attr-dict `:` type($result)
  }];
}

def PTO_PgeB8Op : PTO_VectorMicroOp<"pge_b8", [Pure]> {
  let arguments = (ins StrAttr:$pattern);
  let results = (outs PTO_B8MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $pattern attr-dict `:` type($result)
  }];
}

def PTO_PgeB16Op : PTO_VectorMicroOp<"pge_b16", [Pure]> {
  let arguments = (ins StrAttr:$pattern);
  let results = (outs PTO_B16MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $pattern attr-dict `:` type($result)
  }];
}

def PTO_PgeB32Op : PTO_VectorMicroOp<"pge_b32", [Pure]> {
  let arguments = (ins StrAttr:$pattern);
  let results = (outs PTO_B32MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $pattern attr-dict `:` type($result)
  }];
}

def PTO_PltB8Op : PTO_VectorMicroOp<"plt_b8", [Pure]> {
  let arguments = (ins I32:$scalar);
  let results = (outs PTO_B8MaskTypeConstraint:$mask, I32:$scalar_out);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $scalar attr-dict `:` type($scalar) `->` type($mask) `,` type($scalar_out)
  }];
}

def PTO_PltB16Op : PTO_VectorMicroOp<"plt_b16", [Pure]> {
  let arguments = (ins I32:$scalar);
  let results = (outs PTO_B16MaskTypeConstraint:$mask, I32:$scalar_out);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $scalar attr-dict `:` type($scalar) `->` type($mask) `,` type($scalar_out)
  }];
}

def PTO_PltB32Op : PTO_VectorMicroOp<"plt_b32", [Pure]> {
  let arguments = (ins I32:$scalar);
  let results = (outs PTO_B32MaskTypeConstraint:$mask, I32:$scalar_out);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $scalar attr-dict `:` type($scalar) `->` type($mask) `,` type($scalar_out)
  }];
}

def PTO_PltmB8Op : PTO_VectorMicroOp<"pltm_b8", [Pure]> {
  let arguments = (ins I16:$loop, I32:$bound);
  let results = (outs PTO_B8MaskTypeConstraint:$mask);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $loop `,` $bound attr-dict `:` type($loop) `,` type($bound) `->` type($mask)
  }];
}

def PTO_PltmB16Op : PTO_VectorMicroOp<"pltm_b16", [Pure]> {
  let arguments = (ins I16:$loop, I32:$bound);
  let results = (outs PTO_B16MaskTypeConstraint:$mask);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $loop `,` $bound attr-dict `:` type($loop) `,` type($bound) `->` type($mask)
  }];
}

def PTO_PltmB32Op : PTO_VectorMicroOp<"pltm_b32", [Pure]> {
  let arguments = (ins I16:$loop, I32:$bound);
  let results = (outs PTO_B32MaskTypeConstraint:$mask);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $loop `,` $bound attr-dict `:` type($loop) `,` type($bound) `->` type($mask)
  }];
}

class PTO_MaskUnaryOp<string mnemonic> : PTO_VectorMicroOp<mnemonic, [Pure]> {
  let arguments = (ins PTO_MaskTypeConstraint:$input);
  let results = (outs PTO_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

def PTO_PpackOp : PTO_MaskUnaryOp<"ppack"> {
  let arguments = (ins PTO_MaskTypeConstraint:$input, StrAttr:$part);
  let assemblyFormat = [{
    $input `,` $part attr-dict `:` type($input) `->` type($result)
  }];
}

def PTO_PunpackOp : PTO_MaskUnaryOp<"punpack"> {
  let arguments = (ins PTO_MaskTypeConstraint:$input, StrAttr:$part);
  let assemblyFormat = [{
    $input `,` $part attr-dict `:` type($input) `->` type($result)
  }];
}

def PTO_PbitcastOp : PTO_MaskUnaryOp<"pbitcast">;

def PTO_PnotOp : PTO_VectorMicroOp<"pnot", [Pure]> {
  let arguments = (ins
    PTO_MaskTypeConstraint:$input,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $mask attr-dict `:` type($input) `,` type($mask) `->` type($result)
  }];
}

def PTO_PselOp : PTO_VectorMicroOp<"psel", [Pure]> {
  let arguments = (ins
    PTO_MaskTypeConstraint:$src0,
    PTO_MaskTypeConstraint:$src1,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 `,` $mask attr-dict `:` type($src0) `,` type($src1) `,` type($mask) `->` type($result)
  }];
}

def PTO_PandOp : PTO_VectorMicroOp<"pand", [Pure]> {
  let arguments = (ins
    PTO_MaskTypeConstraint:$src0,
    PTO_MaskTypeConstraint:$src1,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 `,` $mask attr-dict `:` type($src0) `,` type($src1) `,` type($mask) `->` type($result)
  }];
}

def PTO_PorOp : PTO_VectorMicroOp<"por", [Pure]> {
  let arguments = (ins
    PTO_MaskTypeConstraint:$src0,
    PTO_MaskTypeConstraint:$src1,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 `,` $mask attr-dict `:` type($src0) `,` type($src1) `,` type($mask) `->` type($result)
  }];
}

def PTO_PxorOp : PTO_VectorMicroOp<"pxor", [Pure]> {
  let arguments = (ins
    PTO_MaskTypeConstraint:$src0,
    PTO_MaskTypeConstraint:$src1,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 `,` $mask attr-dict `:` type($src0) `,` type($src1) `,` type($mask) `->` type($result)
  }];
}

def PTO_PldsOp : PTO_VectorMicroOp<"plds", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    VPTOAddressSemanticsOpInterface
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    Index:$offset,
    StrAttr:$dist
  );
  let results = (outs PTO_MaskTypeConstraint:$result,
                      Optional<PTO_BufferLikeType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $offset `]` `,` $dist attr-dict `:` type($source) `,` type($offset) `->` type($result)
    (`,` type($updated_base)^)?
  }];
}

def PTO_PldiOp : PTO_VectorMicroOp<"pldi", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    VPTOAddressSemanticsOpInterface
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    Index:$offset,
    StrAttr:$dist
  );
  let results = (outs PTO_MaskTypeConstraint:$result,
                      Optional<PTO_BufferLikeType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $offset `]` `,` $dist attr-dict `:` type($source) `,` type($offset) `->` type($result)
    (`,` type($updated_base)^)?
  }];
}

def PTO_PstiOp : PTO_VectorMicroOp<"psti", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    VPTOAddressSemanticsOpInterface
  ]> {
  let arguments = (ins
    PTO_MaskTypeConstraint:$value,
    PTO_BufferLikeType:$destination,
    Index:$offset,
    StrAttr:$dist
  );
  let results = (outs Optional<PTO_BufferLikeType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `[` $offset `]` `,` $dist attr-dict `:` type($value) `,` type($destination) `,` type($offset)
    (`->` type($updated_base)^)?
  }];
}

def PTO_VabsOp : PTO_VectorMicroOp<"vabs", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$input,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $mask attr-dict `:` type($input) `,` type($mask) `->` type($result)
  }];
}

class PTO_UnaryVecOp<string mnemonic> : PTO_VectorMicroOp<mnemonic, [Pure]> {
  let arguments = (ins
    PTO_VectorType:$input,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $mask attr-dict `:` type($input) `,` type($mask) `->` type($result)
  }];
}

def PTO_VexpOp : PTO_UnaryVecOp<"vexp">;
def PTO_VlnOp : PTO_UnaryVecOp<"vln">;
def PTO_VsqrtOp : PTO_UnaryVecOp<"vsqrt">;
def PTO_VnegOp : PTO_UnaryVecOp<"vneg">;
def PTO_VreluOp : PTO_UnaryVecOp<"vrelu">;
def PTO_VnotOp : PTO_UnaryVecOp<"vnot">;
def PTO_VcaddOp : PTO_UnaryVecOp<"vcadd">;

def PTO_MadOp : PTO_CubeMicroOp<"mad", [
    DeclareOpInterfaceMethods<PTO_MadSemanticOpInterface, [
      "isMadMxFamily", "hasBiasOperand", "readsAccumulator",
      "supportsTf32Mode", "getBiasOrNull"
    ]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$lhs,
    PTO_BufferType:$rhs,
    PTO_BufferType:$dst,
    I64:$m,
    I64:$n,
    I64:$k,
    OptionalAttr<PTO_MadUnitFlagModeAttr>:$unit_flag_mode,
    UnitAttr:$disable_gemv,
    OptionalAttr<PTO_MadSatModeAttr>:$sat_mode,
    OptionalAttr<PTO_Tf32ModeAttr>:$tf32_mode,
    UnitAttr:$n_dir
  );
  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MadAccOp : PTO_CubeMicroOp<"mad_acc", [
    DeclareOpInterfaceMethods<PTO_MadSemanticOpInterface, [
      "isMadMxFamily", "hasBiasOperand", "readsAccumulator",
      "supportsTf32Mode", "getBiasOrNull"
    ]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$lhs,
    PTO_BufferType:$rhs,
    PTO_BufferType:$dst,
    I64:$m,
    I64:$n,
    I64:$k,
    OptionalAttr<PTO_MadUnitFlagModeAttr>:$unit_flag_mode,
    UnitAttr:$disable_gemv,
    OptionalAttr<PTO_MadSatModeAttr>:$sat_mode,
    OptionalAttr<PTO_Tf32ModeAttr>:$tf32_mode,
    UnitAttr:$n_dir
  );
  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MadBiasOp : PTO_CubeMicroOp<"mad_bias", [
    DeclareOpInterfaceMethods<PTO_MadSemanticOpInterface, [
      "isMadMxFamily", "hasBiasOperand", "readsAccumulator",
      "supportsTf32Mode", "getBiasOrNull"
    ]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$lhs,
    PTO_BufferType:$rhs,
    PTO_BufferType:$dst,
    PTO_BufferType:$bias,
    I64:$m,
    I64:$n,
    I64:$k,
    OptionalAttr<PTO_MadUnitFlagModeAttr>:$unit_flag_mode,
    UnitAttr:$disable_gemv,
    OptionalAttr<PTO_MadSatModeAttr>:$sat_mode,
    OptionalAttr<PTO_Tf32ModeAttr>:$tf32_mode,
    UnitAttr:$n_dir
  );
  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MadMxOp : PTO_CubeMicroOp<"mad_mx", [
    DeclareOpInterfaceMethods<PTO_MadSemanticOpInterface, [
      "isMadMxFamily", "hasBiasOperand", "readsAccumulator",
      "supportsTf32Mode", "getBiasOrNull", "getTf32ModeAttr"
    ]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$lhs,
    PTO_BufferType:$rhs,
    PTO_BufferType:$dst,
    I64:$m,
    I64:$n,
    I64:$k,
    OptionalAttr<PTO_MadUnitFlagModeAttr>:$unit_flag_mode,
    UnitAttr:$disable_gemv,
    OptionalAttr<PTO_MadSatModeAttr>:$sat_mode,
    UnitAttr:$n_dir
  );
  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MadMxAccOp : PTO_CubeMicroOp<"mad_mx_acc", [
    DeclareOpInterfaceMethods<PTO_MadSemanticOpInterface, [
      "isMadMxFamily", "hasBiasOperand", "readsAccumulator",
      "supportsTf32Mode", "getBiasOrNull", "getTf32ModeAttr"
    ]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$lhs,
    PTO_BufferType:$rhs,
    PTO_BufferType:$dst,
    I64:$m,
    I64:$n,
    I64:$k,
    OptionalAttr<PTO_MadUnitFlagModeAttr>:$unit_flag_mode,
    UnitAttr:$disable_gemv,
    OptionalAttr<PTO_MadSatModeAttr>:$sat_mode,
    UnitAttr:$n_dir
  );
  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MadMxBiasOp : PTO_CubeMicroOp<"mad_mx_bias", [
    DeclareOpInterfaceMethods<PTO_MadSemanticOpInterface, [
      "isMadMxFamily", "hasBiasOperand", "readsAccumulator",
      "supportsTf32Mode", "getBiasOrNull", "getTf32ModeAttr"
    ]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$lhs,
    PTO_BufferType:$rhs,
    PTO_BufferType:$dst,
    PTO_BufferType:$bias,
    I64:$m,
    I64:$n,
    I64:$k,
    OptionalAttr<PTO_MadUnitFlagModeAttr>:$unit_flag_mode,
    UnitAttr:$disable_gemv,
    OptionalAttr<PTO_MadSatModeAttr>:$sat_mode,
    UnitAttr:$n_dir
  );
  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MadRawOp : PTO_CubeMicroOp<"mad_raw", [
    DeclareOpInterfaceMethods<PTO_MadRawOpInterface, [
      "isMadMxFamily", "hasBiasOperand", "getBiasOrNull"
    ]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$lhs,
    PTO_BufferType:$rhs,
    PTO_BufferType:$dst,
    I64:$xt
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $dst `,` $xt attr-dict `:` type($lhs) `,` type($rhs) `,` type($dst) `,` type($xt)
  }];
}

def PTO_MadBiasRawOp : PTO_CubeMicroOp<"mad_bias_raw", [
    DeclareOpInterfaceMethods<PTO_MadRawOpInterface, [
      "isMadMxFamily", "hasBiasOperand", "getBiasOrNull"
    ]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$lhs,
    PTO_BufferType:$rhs,
    PTO_BufferType:$dst,
    PTO_BufferType:$bias,
    I64:$xt
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $dst `,` $bias `,` $xt attr-dict `:` type($lhs) `,` type($rhs) `,` type($dst) `,` type($bias) `,` type($xt)
  }];
}

def PTO_MadMxRawOp : PTO_CubeMicroOp<"mad_mx_raw", [
    DeclareOpInterfaceMethods<PTO_MadRawOpInterface, [
      "isMadMxFamily", "hasBiasOperand", "getBiasOrNull"
    ]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$lhs,
    PTO_BufferType:$rhs,
    PTO_BufferType:$dst,
    I64:$xt
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $dst `,` $xt attr-dict `:` type($lhs) `,` type($rhs) `,` type($dst) `,` type($xt)
  }];
}

def PTO_MadMxBiasRawOp : PTO_CubeMicroOp<"mad_mx_bias_raw", [
    DeclareOpInterfaceMethods<PTO_MadRawOpInterface, [
      "isMadMxFamily", "hasBiasOperand", "getBiasOrNull"
    ]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$lhs,
    PTO_BufferType:$rhs,
    PTO_BufferType:$dst,
    PTO_BufferType:$bias,
    I64:$xt
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $dst `,` $bias `,` $xt attr-dict `:` type($lhs) `,` type($rhs) `,` type($dst) `,` type($bias) `,` type($xt)
  }];
}

def PTO_VcmaxOp : PTO_UnaryVecOp<"vcmax">;
def PTO_VcminOp : PTO_UnaryVecOp<"vcmin">;
def PTO_VcgaddOp : PTO_UnaryVecOp<"vcgadd">;
def PTO_VcgmaxOp : PTO_UnaryVecOp<"vcgmax">;
def PTO_VcgminOp : PTO_UnaryVecOp<"vcgmin">;
def PTO_VcpaddOp : PTO_UnaryVecOp<"vcpadd">;

class PTO_HistogramOp<string mnemonic> : PTO_VectorMicroOp<mnemonic, [Pure]> {
  let arguments = (ins
    PTO_VectorType:$acc,
    PTO_VectorType:$source,
    PTO_B8MaskTypeConstraint:$mask,
    I32:$bin
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $acc `,` $source `,` $mask `,` $bin attr-dict `:` type($acc) `,` type($source) `,` type($mask) `,` type($bin) `->` type($result)
  }];
}

def PTO_Chistv2Op : PTO_HistogramOp<"chistv2">;
def PTO_Dhistv2Op : PTO_HistogramOp<"dhistv2">;

class PTO_ExtremaPredicateOp<string mnemonic>
    : PTO_VectorMicroOp<mnemonic, [Pure]> {
  let arguments = (ins
    PTO_VectorType:$input,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs
    PTO_VectorType:$value,
    PTO_MaskTypeConstraint:$predicate
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $mask attr-dict `:` type($input) `,` type($mask) `->` type($value) `,` type($predicate)
  }];
}

def PTO_VcbmaxOp : PTO_ExtremaPredicateOp<"vcbmax">;
def PTO_VcbminOp : PTO_ExtremaPredicateOp<"vcbmin">;

class PTO_BinaryVecOp<string mnemonic> : PTO_VectorMicroOp<mnemonic, [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $mask attr-dict `:` type($lhs) `,` type($rhs) `,` type($mask) `->` type($result)
  }];
}

def PTO_VaddOp : PTO_BinaryVecOp<"vadd">;
def PTO_VsubOp : PTO_BinaryVecOp<"vsub">;
def PTO_VmulOp : PTO_BinaryVecOp<"vmul">;
def PTO_VdivOp : PTO_BinaryVecOp<"vdiv">;
def PTO_VmaxOp : PTO_BinaryVecOp<"vmax">;
def PTO_VminOp : PTO_BinaryVecOp<"vmin">;
def PTO_VandOp : PTO_BinaryVecOp<"vand">;
def PTO_VorOp : PTO_BinaryVecOp<"vor">;
def PTO_VxorOp : PTO_BinaryVecOp<"vxor">;

class PTO_TernaryVecOp<string mnemonic> : PTO_VectorMicroOp<mnemonic, [Pure]> {
  let arguments = (ins
    PTO_VectorType:$acc,
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $acc `,` $lhs `,` $rhs `,` $mask attr-dict `:` type($acc) `,` type($lhs) `,` type($rhs) `,` type($mask) `->` type($result)
  }];
}

def PTO_VmaddOp : PTO_TernaryVecOp<"vmadd">;

def PTO_VaddcOp : PTO_VectorMicroOp<"vaddc", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs
    PTO_VectorType:$result,
    PTO_MaskTypeConstraint:$carry
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $mask attr-dict `:` type($lhs) `,` type($rhs) `,` type($mask) `->` type($result) `,` type($carry)
  }];
}

def PTO_VsubcOp : PTO_VectorMicroOp<"vsubc", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs
    PTO_VectorType:$result,
    PTO_MaskTypeConstraint:$carry
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $mask attr-dict `:` type($lhs) `,` type($rhs) `,` type($mask) `->` type($result) `,` type($carry)
  }];
}

def PTO_VaddcsOp : PTO_VectorMicroOp<"vaddcs", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs,
    PTO_MaskTypeConstraint:$carry_in,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs
    PTO_VectorType:$result,
    PTO_MaskTypeConstraint:$carry
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $carry_in `,` $mask attr-dict `:` type($lhs) `,` type($rhs) `,` type($carry_in) `,` type($mask) `->` type($result) `,` type($carry)
  }];
}

def PTO_VsubcsOp : PTO_VectorMicroOp<"vsubcs", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs,
    PTO_MaskTypeConstraint:$carry_in,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs
    PTO_VectorType:$result,
    PTO_MaskTypeConstraint:$carry
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $carry_in `,` $mask attr-dict `:` type($lhs) `,` type($rhs) `,` type($carry_in) `,` type($mask) `->` type($result) `,` type($carry)
  }];
}

def PTO_VshlOp : PTO_BinaryVecOp<"vshl">;
def PTO_VshrOp : PTO_BinaryVecOp<"vshr">;

def PTO_VselOp : PTO_VectorMicroOp<"vsel", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$src0,
    PTO_VectorType:$src1,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 `,` $mask attr-dict `:` type($src0) `,` type($src1) `,` type($mask) `->` type($result)
  }];
}

def PTO_VcmpOp : PTO_VectorMicroOp<"vcmp", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$src0,
    PTO_VectorType:$src1,
    PTO_MaskTypeConstraint:$mask,
    StrAttr:$cmp_mode
  );
  let results = (outs PTO_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 `,` $mask `,` $cmp_mode attr-dict `:` type($src0) `,` type($src1) `,` type($mask) `->` type($result)
  }];
}

def PTO_VcmpsOp : PTO_VectorMicroOp<"vcmps", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$src,
    AnyType:$scalar,
    PTO_MaskTypeConstraint:$mask,
    StrAttr:$cmp_mode
  );
  let results = (outs PTO_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src `,` $scalar `,` $mask `,` $cmp_mode attr-dict `:` type($src) `,` type($scalar) `,` type($mask) `->` type($result)
  }];
}

class PTO_PredicatePairReorderOp<string mnemonic, Type operandTy>
    : PTO_VectorMicroOp<mnemonic, [Pure]> {
  let arguments = (ins
    operandTy:$lhs,
    operandTy:$rhs
  );
  let results = (outs
    operandTy:$low,
    operandTy:$high
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($low) `,` type($high)
  }];
}

def PTO_PdintlvB8Op : PTO_PredicatePairReorderOp<"pdintlv_b8",
    PTO_B8MaskTypeConstraint>;
def PTO_PdintlvB16Op : PTO_PredicatePairReorderOp<"pdintlv_b16",
    PTO_B16MaskTypeConstraint>;
def PTO_PdintlvB32Op : PTO_PredicatePairReorderOp<"pdintlv_b32",
    PTO_B32MaskTypeConstraint>;

def PTO_PintlvB8Op : PTO_PredicatePairReorderOp<"pintlv_b8",
    PTO_B8MaskTypeConstraint>;
def PTO_PintlvB16Op : PTO_PredicatePairReorderOp<"pintlv_b16",
    PTO_B16MaskTypeConstraint>;
def PTO_PintlvB32Op : PTO_PredicatePairReorderOp<"pintlv_b32",
    PTO_B32MaskTypeConstraint>;

class PTO_VecScalarOp<string mnemonic> : PTO_VectorMicroOp<mnemonic, [Pure]> {
  let arguments = (ins
    PTO_VectorType:$input,
    AnyType:$scalar,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $scalar `,` $mask attr-dict `:` type($input) `,` type($scalar) `,` type($mask) `->` type($result)
  }];
}

class PTO_VecScalarMaskedOp<string mnemonic>
    : PTO_VectorMicroOp<mnemonic, [Pure]> {
  let arguments = (ins
    PTO_VectorType:$input,
    AnyType:$scalar,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $scalar `,` $mask attr-dict `:` type($input) `,` type($scalar) `,` type($mask) `->` type($result)
  }];
}

def PTO_VtrcOp : PTO_VectorMicroOp<"vtrc", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$input,
    PTO_MaskTypeConstraint:$mask,
    StrAttr:$round_mode
  );
  let results = (outs PTO_VectorType:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def PTO_VcvtOp : PTO_VectorMicroOp<"vcvt", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$input,
    PTO_MaskTypeConstraint:$mask,
    OptionalAttr<StrAttr>:$rnd,
    OptionalAttr<StrAttr>:$sat,
    OptionalAttr<StrAttr>:$part
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_VmulscvtOp : PTO_VectorMicroOp<"vmulscvt", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$input,
    AnyType:$scalar,
    PTO_MaskTypeConstraint:$mask,
    StrAttr:$rnd,
    StrAttr:$part
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $scalar `,` $mask `,` $rnd `,` $part attr-dict `:` type($input) `,` type($scalar) `,` type($mask) `->` type($result)
  }];
}

def PTO_VbitcastOp : PTO_VectorMicroOp<"vbitcast", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$input
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

def PTO_VciOp : PTO_VectorMicroOp<"vci", [Pure]> {
  let arguments = (ins
    AnyTypeOf<[AnyInteger, AnyFloat], "integer/float scalar">:$index,
    OptionalAttr<StrAttr>:$order
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $index attr-dict `:` type($index) `->` type($result)
  }];
}

def PTO_VbitsortOp : PTO_VectorMicroOp<"vbitsort", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$destination,
    PTO_BufferType:$source,
    PTO_BufferType:$indices,
    Index:$repeat_times
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $destination `,` $source `,` $indices `,` $repeat_times attr-dict `:` type($destination) `,`
    type($source) `,` type($indices) `,` type($repeat_times)
  }];
}

def PTO_Vmrgsort4Op : PTO_VectorMicroOp<"vmrgsort4"> {
  let arguments = (ins
    PTO_BufferType:$destination,
    PTO_BufferType:$source0,
    PTO_BufferType:$source1,
    PTO_BufferType:$source2,
    PTO_BufferType:$source3,
    I64:$count,
    I64:$config
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $destination `,` $source0 `,` $source1 `,` $source2 `,` $source3 `,` $count `,` $config
    attr-dict `:` type($destination) `,` type($source0) `,` type($source1) `,` type($source2) `,`
    type($source3) `,` type($count) `,` type($config)
  }];
}

def PTO_Vgather2Op : PTO_VectorMicroOp<"vgather2", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_VectorType:$offsets,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $offsets `,` $mask attr-dict `:` type($source) `,` type($offsets) `,` type($mask) `->` type($result)
  }];
}

def PTO_VgatherbOp : PTO_VectorMicroOp<"vgatherb", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_VectorType:$offsets,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $offsets `,` $mask attr-dict `:` type($source) `,` type($offsets) `,` type($mask) `->` type($result)
  }];
}

// NOTE: Unvalidated new gather/select/interleave-family abstractions. Added to
// cover CCE builtin families not yet exercised through end-to-end PTO seams.
def PTO_Vgather2BcOp : PTO_VectorMicroOp<"vgather2_bc", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_VectorType:$offsets,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $offsets `,` $mask attr-dict `:` type($source) `,` type($offsets) `,` type($mask) `->` type($result)
  }];
}

def PTO_VmulsOp : PTO_VecScalarMaskedOp<"vmuls">;
def PTO_VaddsOp : PTO_VecScalarMaskedOp<"vadds">;
def PTO_VmaxsOp : PTO_VecScalarMaskedOp<"vmaxs">;
def PTO_VminsOp : PTO_VecScalarMaskedOp<"vmins">;
def PTO_VlreluOp : PTO_VecScalarMaskedOp<"vlrelu">;
def PTO_VshlsOp : PTO_VecScalarMaskedOp<"vshls">;
def PTO_VshrsOp : PTO_VecScalarMaskedOp<"vshrs">;

def PTO_VstsOp : PTO_VectorMicroOp<"vsts", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    VPTOAddressSemanticsOpInterface
  ]> {
  let arguments = (ins
    PTO_VectorType:$value,
    PTO_BufferLikeType:$destination,
    Index:$offset,
    OptionalAttr<StrAttr>:$dist,
    PTO_MaskTypeConstraint:$mask
  );

  let results = (outs Optional<PTO_BufferLikeType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `[` $offset `]` `,` $mask attr-dict `:` type($value) `,` type($destination) `,` type($mask)
    (`->` type($updated_base)^)?
  }];
}

def PTO_VscatterOp : PTO_VectorMicroOp<"vscatter", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_VectorType:$value,
    PTO_BufferType:$destination,
    PTO_VectorType:$offsets,
    PTO_MaskTypeConstraint:$mask
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `,` $offsets `,` $mask attr-dict `:` type($value) `,` type($destination) `,` type($offsets) `,` type($mask)
  }];
}

def PTO_PstsOp : PTO_VectorMicroOp<"psts", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    VPTOAddressSemanticsOpInterface
  ]> {
  let arguments = (ins
    PTO_MaskTypeConstraint:$value,
    PTO_BufferLikeType:$destination,
    Index:$offset,
    StrAttr:$dist
  );

  let results = (outs Optional<PTO_BufferLikeType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `[` $offset `]` `,` $dist attr-dict `:` type($value) `,` type($destination) `,` type($offset)
    (`->` type($updated_base)^)?
  }];
}

def PTO_CopyUbufToGmOp : PTO_MteOp<"copy_ubuf_to_gm", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    I64:$sid,
    I64:$n_burst,
    I64:$len_burst,
    I64:$l2_cache_ctl,
    I64:$burst_dst_stride,
    I64:$burst_src_stride
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $sid `,` $n_burst `,` $len_burst `,`
    $l2_cache_ctl `,` $burst_dst_stride `,` $burst_src_stride
    attr-dict `:` type($source) `,` type($destination) `,` type($sid) `,` type($n_burst) `,`
    type($len_burst) `,` type($l2_cache_ctl) `,`
    type($burst_dst_stride) `,` type($burst_src_stride)
  }];
}

def PTO_MteUbGmOp : PTO_MteOp<"mte_ub_gm", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    I64:$len_burst,
    I64:$n_burst,
    I64:$nburst_src_stride,
    I64:$nburst_dst_stride,
    Optional<I64>:$l2_cache_ctl,
    Variadic<I64>:$loop_counts,
    Variadic<I64>:$loop_src_strides,
    Variadic<I64>:$loop_dst_strides
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::Value":$destination,
      "::mlir::Value":$lenBurst,
      "::mlir::pto::DmaLoopConfig":$nburst,
      "::mlir::Value":$l2CacheCtl,
      "::llvm::ArrayRef<::mlir::pto::DmaLoopConfig>":$loops
    )>,
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::Value":$destination,
      "::mlir::Value":$lenBurst,
      "::mlir::pto::DmaLoopConfig":$nburst,
      "::mlir::Value":$l2CacheCtl,
      "::std::optional<::mlir::pto::DmaLoopConfig>":$loop1,
      "::std::optional<::mlir::pto::DmaLoopConfig>":$loop2
    )>
  ];
}

// Cube bridge wrappers: fuse common register config sequences with the core
// cube transfer/load/store ops, similar to dma_load/dma_store wrappers.
def PTO_MteGmL1Op : PTO_MteOp<"mte_gm_l1", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    I64:$len_burst,
    I64:$n_burst,
    I64:$nburst_src_stride,
    I64:$nburst_dst_stride,
    Variadic<I64>:$loop_counts,
    Variadic<I64>:$loop_src_strides,
    Variadic<I64>:$loop_dst_strides
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::Value":$destination,
      "::mlir::Value":$lenBurst,
      "::mlir::pto::DmaLoopConfig":$nburst,
      "::llvm::ArrayRef<::mlir::pto::DmaLoopConfig>":$loops
    )>,
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::Value":$destination,
      "::mlir::Value":$lenBurst,
      "::mlir::pto::DmaLoopConfig":$nburst,
      "::std::optional<::mlir::pto::DmaLoopConfig>":$loop1,
      "::std::optional<::mlir::pto::DmaLoopConfig>":$loop2
    )>
  ];
}

def PTO_MteL1UbOp : PTO_MteOp<"mte_l1_ub", [
    CubeMicroOpInterface,
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    I64:$len_burst,
    I64:$n_burst,
    I64:$nburst_src_stride,
    I64:$nburst_dst_stride,
    Variadic<I64>:$loop_counts,
    Variadic<I64>:$loop_src_strides,
    Variadic<I64>:$loop_dst_strides
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::Value":$destination,
      "::mlir::Value":$lenBurst,
      "::mlir::pto::DmaLoopConfig":$nburst,
      "::llvm::ArrayRef<::mlir::pto::DmaLoopConfig>":$loops
    )>,
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::Value":$destination,
      "::mlir::Value":$lenBurst,
      "::mlir::pto::DmaLoopConfig":$nburst,
      "::std::optional<::mlir::pto::DmaLoopConfig>":$loop1,
      "::std::optional<::mlir::pto::DmaLoopConfig>":$loop2
    )>
  ];
}

def PTO_MteL1BtOp : PTO_MteOp<"mte_l1_bt", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    I64:$len_burst,
    I64:$n_burst,
    I64:$nburst_src_gap,
    I64:$nburst_dst_gap
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::Value":$destination,
      "::mlir::Value":$lenBurst,
      "::mlir::pto::DmaLoopConfig":$nburst
    )>
  ];
}

def PTO_MteL1FbOp : PTO_MteOp<"mte_l1_fb", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    I64:$len_burst,
    I64:$n_burst,
    I64:$nburst_src_gap,
    I64:$nburst_dst_gap
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::Value":$destination,
      "::mlir::Value":$lenBurst,
      "::mlir::pto::DmaLoopConfig":$nburst
    )>
  ];
}

def PTO_MteGmL1FracOp : PTO_MteOp<"mte_gm_l1_frac", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    I64:$n_value,
    I64:$d_value,
    I64:$src_inner_stride,
    I64:$group_count,
    I64:$dst_loop2_stride,
    I64:$dst_loop3_stride,
    I64:$dst_loop4_stride,
    I64:$l2_cache_ctrl,
    I1:$smallc0_en,
    Optional<I64>:$src_outer_stride,
    PTO_CubeLoadFracModeAttr:$mode
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::Value":$destination,
      "::mlir::pto::CubeLoadFracMode":$mode,
      "::mlir::pto::CubeLoadFracShapeConfig":$shape,
      "::mlir::pto::CubeLoadFracSrcLayoutConfig":$srcLayout,
      "::mlir::pto::CubeLoadFracDstGroupConfig":$dstGroup,
      "::mlir::pto::CubeLoadFracCtrlConfig":$ctrl
    )>
  ];
}

def PTO_RawFillL1Op : PTO_MicroOp<"raw_fill_l1", [
    CubeMicroOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Fill a strided L1/MAT region with a raw bit pattern";
  let description = [{
    Public wrapper for filling a strided L1 matrix-buffer region with a raw
    bit pattern. `dst` may be a memref or `!pto.ptr` buffer-like value and is
    expanded to the canonical `pto.create_cbuf_matrix` micro-op by the
    wrapper-expansion pass.

    Fields (in order):
    - `dst`: destination buffer; after expansion must resolve to the MAT/L1
      address space.
    - `byte_offset`: non-negative byte distance applied to `dst` before the
      fill; it must be a multiple of 32 bytes.
    - `raw_value`: raw fill pattern. For a 16-bit fill, the low 16 bits are
      repeated in every 16-bit word; for a 32-bit fill, all 32 bits are used.
    - `repeat_times`: number of fill repetitions.
    - `block_num_32b`: number of 32-byte blocks per repeat.
    - `dst_gap_32b`: destination gap between repeats, in 32-byte blocks.
    - `fill_word_bits`: static i64 attribute selecting the compile-time fill
      word width, exactly 16 or 32.

    `repeat_times`, `block_num_32b`, and `dst_gap_32b` must each be in the
    inclusive range [0, 32767], including when dynamically computed. The
    verifier diagnoses known constants outside that range.

    The effective destination address (`dst + byte_offset`) must be aligned to
    32 bytes. The verifier diagnoses known constant offsets that are not
    multiples of 32; dynamic offsets are a caller precondition.
  }];

  let arguments = (ins
    PTO_BufferLikeType:$dst,
    I64:$byte_offset,
    I32:$raw_value,
    I64:$repeat_times,
    I64:$block_num_32b,
    I64:$dst_gap_32b,
    I64Attr:$fill_word_bits
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $dst `,` $byte_offset `,` $raw_value `,` $repeat_times `,` $block_num_32b `,`
    $dst_gap_32b attr-dict `:` type($dst) `,` type($byte_offset) `,`
    type($raw_value) `,` type($repeat_times) `,` type($block_num_32b) `,`
    type($dst_gap_32b)
  }];
}

def PTO_CreateCbufMatrixOp : PTO_CubeMicroOp<"create_cbuf_matrix", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Canonical L1 raw-pattern fill micro-op";
  let description = [{
    Canonical micro-op for filling a strided L1/CBUF matrix region with a raw
    bit pattern. Produced by the `pto.raw_fill_l1` wrapper expansion.

    `dst` must be a 32-byte-aligned
    `!pto.ptr<ui16|ui32, #pto.address_space<mat>>` pointer that already
    includes `byte_offset`. For a 16-bit fill, the low 16 bits of
    `raw_value` are repeated in every 16-bit word; for a 32-bit fill, all 32
    bits are used. The static `fill_word_bits` i64 attribute selects the fill
    word width, exactly 16 or 32, and must match the destination view width.
    `repeat_times` counts repetitions; `block_num_32b` and `dst_gap_32b`
    count 32-byte blocks. All three geometry operands must be in the
    inclusive range [0, 32767].
  }];

  let arguments = (ins
    PTO_BufferType:$dst,
    I32:$raw_value,
    I64:$repeat_times,
    I64:$block_num_32b,
    I64:$dst_gap_32b,
    I64Attr:$fill_word_bits
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $dst `,` $raw_value `,` $repeat_times `,` $block_num_32b `,`
    $dst_gap_32b attr-dict `:` type($dst) `,` type($raw_value) `,`
    type($repeat_times) `,` type($block_num_32b) `,` type($dst_gap_32b)
  }];
}

def PTO_MteL1L0aOp : PTO_MteOp<"mte_l1_l0a", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    Optional<I64>:$m,
    Optional<I64>:$k,
    Optional<I64>:$start_row,
    Optional<I64>:$start_col,
    Optional<I64>:$m_start,
    Optional<I64>:$k_start,
    Optional<I64>:$m_step,
    Optional<I64>:$k_step,
    Optional<I64>:$src_stride,
    Optional<I64>:$dst_stride,
    DefaultValuedAttr<BoolAttr, "false">:$transpose
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MteL1L0bOp : PTO_MteOp<"mte_l1_l0b", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    Optional<I64>:$k,
    Optional<I64>:$n,
    Optional<I64>:$start_row,
    Optional<I64>:$start_col,
    Optional<I64>:$m_start,
    Optional<I64>:$k_start,
    Optional<I64>:$m_step,
    Optional<I64>:$k_step,
    Optional<I64>:$src_stride,
    Optional<I64>:$dst_stride,
    DefaultValuedAttr<BoolAttr, "false">:$transpose
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MteL1L0aMxOp : PTO_MteOp<"mte_l1_l0a_mx", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    Optional<I64>:$m,
    Optional<I64>:$k,
    Optional<I64>:$start_row,
    Optional<I64>:$start_col,
    Optional<I64>:$x_start,
    Optional<I64>:$y_start,
    Optional<I64>:$x_step,
    Optional<I64>:$y_step,
    Optional<I64>:$src_stride,
    Optional<I64>:$dst_stride
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MteL1L0bMxOp : PTO_MteOp<"mte_l1_l0b_mx", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    Optional<I64>:$k,
    Optional<I64>:$n,
    Optional<I64>:$start_row,
    Optional<I64>:$start_col,
    Optional<I64>:$x_start,
    Optional<I64>:$y_start,
    Optional<I64>:$x_step,
    Optional<I64>:$y_step,
    Optional<I64>:$src_stride,
    Optional<I64>:$dst_stride
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MteL0cL1Op : PTO_MteOp<"mte_l0c_l1", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    I64:$m,
    I64:$n,
    I64:$src_stride,
    I64:$dst_stride,
    Optional<PTO_AccStorePayloadType>:$pre_quant,
    Optional<PTO_AccStorePayloadType>:$pre_relu,
    Optional<PTO_AccStoreClipPayloadType>:$clip_value,
    Optional<I64>:$split,
    Optional<I64>:$loop0_src_stride,
    Optional<I64>:$loop3_count,
    Optional<I64>:$loop3_src_stride,
    Optional<I64>:$loop3_dst_stride,
    OptionalAttr<PTO_AccStoreModeAttr>:$mode,
    OptionalAttr<PTO_UnitFlagCtrlAttr>:$unit_flag,
    OptionalAttr<PTO_AccStoreQuantPreModeAttr>:$pre_quant_mode,
    OptionalAttr<PTO_ReluPreModeAttr>:$pre_relu_mode,
    OptionalAttr<PTO_AccStoreAtomicTypeAttr>:$atomic_type,
    OptionalAttr<PTO_AccStoreAtomicOpAttr>:$atomic_op,
    OptionalAttr<PTO_AccStoreSatModeAttr>:$sat_mode
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MteL0cGmOp : PTO_MteOp<"mte_l0c_gm", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    I64:$m,
    I64:$n,
    I64:$src_stride,
    I64:$dst_stride,
    Optional<PTO_AccStorePayloadType>:$pre_quant,
    Optional<PTO_AccStorePayloadType>:$pre_relu,
    Optional<PTO_AccStoreClipPayloadType>:$clip_value,
    I64:$sid,
    I64:$l2_cache_ctrl,
    Optional<I64>:$split,
    Optional<I64>:$loop0_src_stride,
    Optional<I64>:$loop3_count,
    Optional<I64>:$loop3_src_stride,
    Optional<I64>:$loop3_dst_stride,
    OptionalAttr<PTO_AccStoreModeAttr>:$mode,
    OptionalAttr<PTO_UnitFlagCtrlAttr>:$unit_flag,
    OptionalAttr<PTO_AccStoreQuantPreModeAttr>:$pre_quant_mode,
    OptionalAttr<PTO_ReluPreModeAttr>:$pre_relu_mode,
    OptionalAttr<PTO_AccStoreAtomicTypeAttr>:$atomic_type,
    OptionalAttr<PTO_AccStoreAtomicOpAttr>:$atomic_op,
    OptionalAttr<PTO_AccStoreSatModeAttr>:$sat_mode
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MteL0cUbOp : PTO_MteOp<"mte_l0c_ub", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    I64:$m,
    I64:$n,
    I64:$src_stride,
    I64:$dst_stride,
    Optional<PTO_AccStorePayloadType>:$pre_quant,
    Optional<PTO_AccStorePayloadType>:$pre_relu,
    Optional<PTO_AccStoreClipPayloadType>:$clip_value,
    Optional<I64>:$sub_blockid,
    Optional<I64>:$split,
    Optional<I64>:$loop0_src_stride,
    Optional<I64>:$loop3_count,
    Optional<I64>:$loop3_src_stride,
    Optional<I64>:$loop3_dst_stride,
    PTO_AccStoreUbDstModeAttr:$dst_mode,
    OptionalAttr<PTO_AccStoreModeAttr>:$mode,
    OptionalAttr<PTO_UnitFlagCtrlAttr>:$unit_flag,
    OptionalAttr<PTO_AccStoreQuantPreModeAttr>:$pre_quant_mode,
    OptionalAttr<PTO_ReluPreModeAttr>:$pre_relu_mode,
    OptionalAttr<PTO_AccStoreSatModeAttr>:$sat_mode
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

// NOTE: Unvalidated new x2 / pair / align-store-family abstractions. Added to
// reflect CCE builtin families but not yet end-to-end validated.
def PTO_VselrOp : PTO_VectorMicroOp<"vselr", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$src0,
    PTO_VectorType:$src1
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 attr-dict `:` type($src0) `,` type($src1) `->` type($result)
  }];
}

def PTO_VsqzOp : PTO_UnaryVecOp<"vsqz">;

def PTO_VusqzOp : PTO_VectorMicroOp<"vusqz", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$src,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src `,` $mask attr-dict `:` type($src) `,` type($mask) `->` type($result)
  }];
}

def PTO_VpackOp : PTO_VectorMicroOp<"vpack", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$src,
    StrAttr:$part
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src `,` $part attr-dict `:` type($src) `->` type($result)
  }];
}

def PTO_VsunpackOp : PTO_VectorMicroOp<"vsunpack", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$src,
    Index:$part
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src `,` $part attr-dict `:` type($src) `->` type($result)
  }];
}

def PTO_VzunpackOp : PTO_VectorMicroOp<"vzunpack", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$src,
    Index:$part
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src `,` $part attr-dict `:` type($src) `->` type($result)
  }];
}

def PTO_Vselrv2Op : PTO_VectorMicroOp<"vselrv2", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$src0,
    PTO_VectorType:$src1
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 attr-dict `:` type($src0) `,` type($src1) `->` type($result)
  }];
}

def PTO_VintlvOp : PTO_VectorMicroOp<"vintlv", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs
  );
  let results = (outs PTO_VectorType:$low, PTO_VectorType:$high);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($low) `,` type($high)
  }];
}

def PTO_VdintlvOp : PTO_VectorMicroOp<"vdintlv", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs
  );
  let results = (outs PTO_VectorType:$low, PTO_VectorType:$high);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($low) `,` type($high)
  }];
}

def PTO_Vintlvv2Op : PTO_VectorMicroOp<"vintlvv2", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs,
    StrAttr:$part
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $part attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def PTO_Vdintlvv2Op : PTO_VectorMicroOp<"vdintlvv2", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs,
    StrAttr:$part
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $part attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def PTO_VmullOp : PTO_VectorMicroOp<"vmull", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$low, PTO_VectorType:$high);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $mask attr-dict `:` type($lhs) `,` type($rhs) `,` type($mask) `->` type($low) `,` type($high)
  }];
}

def PTO_VmulaOp : PTO_VectorMicroOp<"vmula", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$acc,
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $acc `,` $lhs `,` $rhs `,` $mask attr-dict `:` type($acc) `,` type($lhs) `,` type($rhs) `,` type($mask) `->` type($result)
  }];
}

class PTO_UnmaskedBinaryVecOp<string mnemonic>
    : PTO_VectorMicroOp<mnemonic, [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

class PTO_BinaryVecMaskedOp<string mnemonic>
    : PTO_VectorMicroOp<mnemonic, [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $mask attr-dict `:` type($lhs) `,` type($rhs) `,` type($mask) `->` type($result)
  }];
}

def PTO_VpreluOp : PTO_BinaryVecMaskedOp<"vprelu">;
def PTO_VexpdifOp : PTO_VectorMicroOp<"vexpdif", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$input,
    PTO_VectorType:$max,
    PTO_MaskTypeConstraint:$mask,
    StrAttr:$part
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $max `,` $mask `,` $part attr-dict `:` type($input) `,` type($max) `,` type($mask) `->` type($result)
  }];
}

def PTO_VaxpyOp : PTO_VectorMicroOp<"vaxpy", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$src0,
    PTO_VectorType:$src1,
    AnyType:$alpha,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 `,` $alpha `,` $mask attr-dict `:` type($src0) `,` type($src1) `,` type($alpha) `,` type($mask) `->` type($result)
  }];
}

def PTO_VaddreluconvOp : PTO_VectorMicroOp<"vaddreluconv", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def PTO_VmulconvOp : PTO_VectorMicroOp<"vmulconv", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def PTO_Vstsx2Op : PTO_VectorMicroOp<"vstsx2", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_VectorType:$low,
    PTO_VectorType:$high,
    PTO_BufferLikeType:$destination,
    Index:$offset,
    StrAttr:$dist,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $low `,` $high `,` $destination `[` $offset `]` `,` $dist `,` $mask attr-dict `:` type($low) `,` type($high) `,` type($destination) `,` type($offset) `,` type($mask)
  }];
}

def PTO_VsldbOp : PTO_VectorMicroOp<"vsldb", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    VPTOAddressSemanticsOpInterface
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    I16:$block_stride,
    I16:$repeat_stride,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result,
                      Optional<PTO_BufferLikeType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $block_stride `,` $repeat_stride `,` $mask attr-dict `:` type($source) `,` type($block_stride) `,` type($repeat_stride) `,` type($mask) `->` type($result)
    (`,` type($updated_base)^)?
  }];
}

def PTO_VsstbOp : PTO_VectorMicroOp<"vsstb", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    VPTOAddressSemanticsOpInterface
  ]> {
  let arguments = (ins
    PTO_VectorType:$value,
    PTO_BufferLikeType:$destination,
    I16:$block_stride,
    I16:$repeat_stride,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs Optional<PTO_BufferLikeType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `,` $block_stride `,` $repeat_stride `,` $mask attr-dict `:` type($value) `,` type($destination) `,` type($block_stride) `,` type($repeat_stride) `,` type($mask)
    (`->` type($updated_base)^)?
  }];
}

def PTO_VstasOp : PTO_VectorMicroOp<"vstas", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    VPTOAddressSemanticsOpInterface
  ]> {
  let arguments = (ins
    PTO_AlignTypeConstraint:$value,
    PTO_BufferLikeType:$destination,
    I32:$offset
  );
  let results = (outs Optional<PTO_BufferLikeType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `,` $offset attr-dict `:` type($value) `,` type($destination) `,` type($offset)
    (`->` type($updated_base)^)?
  }];
}

def PTO_VstarOp : PTO_VectorMicroOp<"vstar", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_AlignTypeConstraint:$value,
    PTO_BufferLikeType:$destination
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination attr-dict `:` type($value) `,` type($destination)
  }];
}

// NOTE: Unvalidated stateful store-family abstractions. These preserve
// align/base/offset update results explicitly in SSA form instead of relying on
// implicit CCE reference updates.
// Keep `base/base_out` pointer-only (`PTO_BufferType`): memref semantics for
// stateful post-update addresses are intentionally out of scope in this change.
def PTO_PstuOp : PTO_VectorMicroOp<"pstu", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_AlignTypeConstraint:$align_in,
    PTO_MaskTypeConstraint:$value,
    PTO_BufferType:$base
  );
  let results = (outs PTO_AlignTypeConstraint:$align_out, PTO_BufferType:$base_out);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $align_in `,` $value `,` $base attr-dict `:` type($align_in) `,` type($value) `,` type($base) `->` type($align_out) `,` type($base_out)
  }];
}

def PTO_VstusOp : PTO_VectorMicroOp<"vstus", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    VPTOAddressSemanticsOpInterface
  ]> {
  let arguments = (ins
    PTO_AlignTypeConstraint:$align_in,
    I32:$offset,
    PTO_VectorType:$value,
    PTO_BufferType:$base
  );
  let results = (outs PTO_AlignTypeConstraint:$align_out,
                      Optional<PTO_BufferType>:$base_out);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $align_in `,` $offset `,` $value `,` $base attr-dict `:` type($align_in) `,` type($offset) `,` type($value) `,` type($base) `->` type($align_out)
    (`,` type($base_out)^)?
  }];
}

def PTO_VsturOp : PTO_VectorMicroOp<"vstur", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_AlignTypeConstraint:$align_in,
    PTO_VectorType:$value,
    PTO_BufferType:$base,
    StrAttr:$mode
  );
  let results = (outs PTO_AlignTypeConstraint:$align_out);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $align_in `,` $value `,` $base `,` $mode attr-dict `:` type($align_in) `,` type($value) `,` type($base) `->` type($align_out)
  }];
}

#endif // MLIR_DIALECT_PTO_IR_VPTOOPS
