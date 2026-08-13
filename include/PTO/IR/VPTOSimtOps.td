// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details.

//===- VPTOSimtOps - semantic operation family -*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_VPTOSIMTOPS
#define MLIR_DIALECT_PTO_IR_VPTOSIMTOPS

include "PTO/IR/VPTOOpCommon.td"

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

def PTO_ExpOp : PTO_UnaryF16F32ScalarOp<"exp">;
def PTO_LogOp : PTO_UnaryF16F32ScalarOp<"log">;
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

def PTO_SyncthreadsOp : PTO_SimtOp<"syncthreads", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Synchronize all workitems in a SIMT VF entry";
  let arguments = (ins);
  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{ attr-dict }];
}

def PTO_ThreadfenceOp : PTO_SimtOp<"threadfence", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Issue a SIMT workitem memory fence";
  let arguments = (ins);
  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{ attr-dict }];
}

def PTO_ThreadfenceBlockOp : PTO_SimtOp<"threadfence_block", [
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

#endif // MLIR_DIALECT_PTO_IR_VPTOSIMTOPS
