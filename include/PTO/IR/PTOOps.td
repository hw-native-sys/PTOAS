// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOOps.td - Pattern descriptor operations -----------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// This file declares the PTO dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTOOPS
#define MLIR_DIALECT_PTO_IR_PTOOPS

include "PTO/IR/PTODialect.td"
include "PTO/IR/PTOAttrs.td"
include "PTO/IR/PTOTypeDefs.td"
include "PTO/IR/PTOInterfaces.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

def TensorOrMemref :
  AnyTypeOf<[AnyMemRef, AnyRankedTensor], "Tensor or Memref">;

def PTODpsType :
  AnyTypeOf<[AnyRankedTensor, AnyMemRef, PartitionTensorViewType, TileBufType]>;

def PtrOrMemRef :
  AnyTypeOf<[PtrType, AnyMemRef], "Ptr or MemRef">;

def TileBufOrMemRef :
  AnyTypeOf<[TileBufType, AnyMemRef], "TileBuf or MemRef">;

def ScalarPtrOrMemRef :
  TypeConstraint<
    CPred<"::mlir::pto::isScalarPtrOrMemRef($_self)">,
    "Ptr or MemRef in GM">;

def ScalarType :
  AnyTypeOf<[AnySignlessInteger, AnyFloat], "numeric (integer/float)">;

//===----------------------------------------------------------------------===//
// Op Class
//===----------------------------------------------------------------------===//
class PTO_TOp<string mnemonic, list<Trait> traits = []>
    : Op<PTO_Dialect, mnemonic, !listconcat(traits, [ ])>;

class PTO_DpsOp<string mnemonic, list<Trait> traits = []>
    : Op<PTO_Dialect, mnemonic, traits # [DestinationStyleOpInterface]> {
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getDstMutable();
    }
  }];
}

class PTO_Op<string mnemonic, list<Trait> traits = []>
    : Op<PTO_Dialect, mnemonic, !listconcat(traits, [ ])>;

//===----------------------------------------------------------------------===//
// Pointer/View Ops (for your front-end IR)
//===----------------------------------------------------------------------===//

def AddPtrOp : PTO_Op<"addptr", [
    Pure,
    AllTypesMatch<["ptr", "result"]>,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Add an element offset to a !pto.ptr";
  let description = [{
    Computes a new pointer by adding an element offset to the base pointer.
    The offset is in elements (not bytes).
  }];

  let arguments = (ins
    PtrType:$ptr,
    Index:$offset
  );

  let results = (outs PtrType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $ptr `,` $offset attr-dict `:` type($ptr) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Scalar pointer load/store
//===----------------------------------------------------------------------===//

def LoadScalarOp : PTO_Op<"load_scalar", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Load a single scalar element from a pointer at offset.";

  let arguments = (ins
    ScalarPtrOrMemRef:$ptr,
    Index:$offset
  );

  let results = (outs AnyType:$value);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $ptr `[` $offset `]` attr-dict `:` type($ptr) `->` type($value)
  }];
}

def StoreScalarOp : PTO_Op<"store_scalar", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Store a single scalar element to a pointer at offset.";

  let arguments = (ins
    ScalarPtrOrMemRef:$ptr,
    Index:$offset,
    AnyType:$value
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $ptr `[` $offset `]` attr-dict `:` type($ptr) `,` type($value)
  }];
}

def MakeTensorViewOp : PTO_Op<"make_tensor_view", [AttrSizedOperandSegments]> {
  let summary = "Wrap a pointer as a tensor_view descriptor (no allocation, no copy).";

  let arguments = (ins
    AnyType:$ptr,
    Variadic<Index>:$shape,
    Variadic<Index>:$strides,
    OptionalAttr<PTO_LayoutAttr>:$layout
  );

  let results = (outs AnyType:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// =============================================================================
// PartitionViewOp
// =============================================================================
def PartitionViewOp : PTO_Op<"partition_view", [AttrSizedOperandSegments]> {
  let summary = "Partition a tensor view into a smaller logical view (logical slicing).";
  let description = [{
    Captures a specific calculation region from a large view.
    It carries offsets (where to read) and sizes (how much to read).
    
    IR Example:
      %1 = pto.partition_view %0, offsets=[...], sizes=[...] 
           : !pto.tensor_view -> !pto.partition_tensor_view
  }];

  let arguments = (ins
    TensorViewType:$source,    // 输入: 物理大底座 (MakeTensorViewOp 的结果)
    Variadic<Index>:$offsets,  // 动态 offsets
    Variadic<Index>:$sizes     // 动态 sizes
  );

  let results = (outs PartitionTensorViewType:$result); // 输出: 逻辑切片

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` `offsets` `=` `[` $offsets `]` `,` `sizes` `=` `[` $sizes `]`
    attr-dict `:` qualified(type($source)) `->` qualified(type($result))
  }];
}

// Helper: tensor_view or memref (after lowering tensor_view to memref).
def TensorViewOrMemRef :
  AnyTypeOf<[TensorViewType, AnyMemRef], "TensorView or MemRef">;

// Get the size of a dimension of a tensor_view or its lowered memref view.
// Result type: Index (use arith.index_cast if i32 is needed).
def GetTensorViewDimOp : PTO_Op<"get_tensor_view_dim", [Pure]> {
  let summary = "Get the size of a dimension of a tensor_view.";
  let description = [{
    Returns the size of the given dimension of a logical tensor view.
    This op accepts either !pto.tensor_view or the memref it is lowered to.
    IR (tensor_view form):
      %dim_size = pto.get_tensor_view_dim %tv, %dim_index
        : !pto.tensor_view, index -> index
    IR (memref form, after lowering):
      %dim_size = pto.get_tensor_view_dim %mr, %dim_index
        : memref<...>, index -> index
  }];
  let arguments = (ins
    TensorViewOrMemRef:$tensor_view,
    Index:$dim_index
  );
  let results = (outs Index:$result);
  let assemblyFormat = [{
    $tensor_view `,` $dim_index `:` qualified(type($tensor_view)) `->` qualified(type($result))
    attr-dict
  }];
}

def AllocTileOp : PTO_Op<"alloc_tile", [AttrSizedOperandSegments]> {
  let summary = "Allocates a tile buffer (logical buffer).";

  let arguments = (ins
    Optional<I64>:$addr,
    Optional<Index>:$valid_row,
    Optional<Index>:$valid_col
  );

  let results = (outs TileBufType:$result);

  let assemblyFormat = [{
    (`addr` `=` $addr^)?
    (`valid_row` `=` $valid_row^)? 
    (`valid_col` `=` $valid_col^)?
     attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::LogicalResult verify();
  }];
}


// ============================================================================
// BindTileOp: 将 Config 和 Valid Dims 绑定到 MemRef 上
// ============================================================================
def BindTileOp : PTO_Op<"bind_tile", [
    Pure,
    AttrSizedOperandSegments
    // 允许输入 offset:0 -> 输出 offset:?
  ]> {
  let summary = "Binds metadata and implicitly casts layout";
  let description = [{
    Wraps a memref with PTO metadata (valid dimensions and config).
  }];

  // [修改] 以前是 Variadic<Index>:$valid_dims
  // 现在改为明确的 Optional Row/Col，与 PointerCastOp 保持一致
  let arguments = (ins
    AnyMemRef:$source,
    Optional<Index>:$valid_row,
    Optional<Index>:$valid_col,
    TileBufConfigAttr:$config
  );

  let results = (outs AnyMemRef:$result);

  // [修改] assemblyFormat: 去掉 valid_dims，改为 ($row, $col)?
  let assemblyFormat = [{
    $source (`,` $valid_row^ `,` $valid_col)? attr-dict `:` qualified(type($source)) `->` qualified(type($result))
  }];
}

def SubsetOp : PTO_Op<"subset", [
    Pure,
    ViewLikeOpInterface,
    DeclareOpInterfaceMethods<InferTypeOpInterface> // 启用 C++ 推导
  ]> {
  
  let summary = "Create a strided view (subset) from a parent tile.";
  let description = [{
    Creates a view into the source tile.
    - Result Shape: Defined by static `sizes`.
    - Result Strides: Inherited from `source`.
    - Result Offset: Represented as multi-dimensional symbols (s0, s1...) in the layout map.
  }];

  let arguments = (ins
    TileBufType:$source,
    Variadic<Index>:$offsets, // 运行时动态偏移 [i, j]
    I64ArrayAttr:$sizes       // 静态形状 [32, 32]
  );

  let results = (outs TileBufType:$result);
  let hasVerifier = 1;

  // 语法示例: %sub = pto.subset %src[%i, %j] sizes [32, 32] : !type
  // 注意：没有 -> qualified(type($result))
  let assemblyFormat = [{
    $source `[` $offsets `]` `sizes` $sizes attr-dict `:` qualified(type($source))
  }];

  // [新增] 显式实现 ViewLikeOpInterface 缺失的方法
  let extraClassDeclaration = [{
    // 接口要求 getViewSource，我们转发给自动生成的 getSource
    ::mlir::Value getViewSource() { return getSource(); }
    
    // ViewLikeOpInterface 可能还需要 getOffsets (如果 Variadic 不自动匹配)
    // 但通常 Variadic<Index>:$offsets 会生成 getOffsets()，这应该没问题。
    // 如果后续报 getOffsets 错，也可以在这里加。
  }];
}

def SetValidShapeOp : PTO_Op<"set_validshape", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Update valid row/col metadata in-place on a dynamic tile";
  let description = [{
    Updates the runtime valid_row/valid_col metadata on an existing dynamic
    tile_buf handle in place.

    This op does not allocate or alias a new tile handle. It is only valid for
    tile_buf types whose valid_row and valid_col are both dynamic (`?`), and
    the source must be a locally bound tile handle (for example `alloc_tile`
    or a view derived from it). Function arguments/results are not supported.
  }];

  let arguments = (ins
    TileBufOrMemRef:$source,
    Index:$valid_row,
    Index:$valid_col
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $valid_row `,` $valid_col attr-dict `:`
    qualified(type($source))
  }];
}

// ============================================================================
// SSA TileBuf Config Ops (aliasing views)
// ============================================================================

def BitcastOp : PTO_Op<"bitcast", [
    Pure,
    ViewLikeOpInterface
  ]> {
  let summary = "SSA dtype reinterpretation of a tile buffer view (aliases src storage)";
  let description = [{
    Returns a tile buffer view with a different element type (dtype) while
    reusing the same underlying storage as the source.

    This op is a metadata/config rewrite only (no data movement).
  }];

  let arguments = (ins
    TileBufType:$src
  );

  let results = (outs
    TileBufType:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = "$src attr-dict `:` qualified(type($src)) `->` qualified(type($result))";

  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSrc(); }
  }];
}

//===----------------------------------------------------------------------===//
// DMA Ops
//===----------------------------------------------------------------------===//
// -------------------------
// DPS versions in tile world
// -------------------------
def TLoadOp  : PTO_TOp<"tload", [
  PTO_DpsInitOpInterface,
  AttrSizedOperandSegments,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO data load operation (Dimension Collapse: PartitionView -> TileBuf).";
  let description = [{
    Loads data from a high-dimensional logical partition view into a 2D physical tile buffer.
    Constraint: The product of partition view sizes must match the product of tile buffer valid dims.
    DPS form:
      pto.tload ins(%partition_view) outs(%tile_buf)
  }];

  let arguments = (ins 
    PTODpsType:$src,
    PTODpsType:$dst,
    OptionalAttr<PTO_PadModeAttr>:$pad_mode,
    Optional<AnyType>:$pad_value,
    Optional<Index>:$left_padding_num,
    Optional<AnyType>:$right_padding_num,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$init_out_buffer,
    Optional<AnyType>:$init_condition
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let hasVerifier = 1;

  let builders = [
    // 1. Basic: (src, dst)
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst), [{
      build($_builder, $_state, res, src, dst, 
            /*pad_mode=*/nullptr, /*pad_value=*/nullptr, 
            /*left=*/nullptr, /*right=*/nullptr, 
            /*init_out=*/nullptr, /*init_cond=*/nullptr); 
    }]>,

    // 2. With left_padding_num
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst,
                   "Value":$left_padding_num), [{
      build($_builder, $_state, res, src, dst, 
            nullptr, nullptr, left_padding_num, nullptr, nullptr, nullptr);
    }]>,

    // 3. With pad_mode, pad_value
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst,
                   "pto::PadModeAttr":$pad_mode, "Value":$pad_value), [{
      build($_builder, $_state, res, src, dst, 
            pad_mode, pad_value, nullptr, nullptr, nullptr, nullptr);
    }]>,

    // 4. ... + left
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst,
                   "pto::PadModeAttr":$pad_mode, "Value":$pad_value,
                   "Value":$left_padding_num), [{
      build($_builder, $_state, res, src, dst, 
            pad_mode, pad_value, left_padding_num, nullptr, nullptr, nullptr);
    }]>,

    // 5. ... + left + right
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst,
                   "pto::PadModeAttr":$pad_mode, "Value":$pad_value,
                   "Value":$left_padding_num, "Value":$right_padding_num), [{
      build($_builder, $_state, res, src, dst, 
            pad_mode, pad_value, left_padding_num, right_padding_num, nullptr, nullptr);
    }]>,

    // 6. ... + left + right + bool
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst,
                   "pto::PadModeAttr":$pad_mode, "Value":$pad_value,
                   "Value":$left_padding_num, "Value":$right_padding_num,
                   "bool":$init_out_buffer), [{
      build($_builder, $_state, res, src, dst, 
            pad_mode, pad_value, left_padding_num, right_padding_num, 
            init_out_buffer, nullptr);
    }]>
  ];

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`pad_mode` `=` $pad_mode^)?
    (`pad_value` `=` $pad_value^ `:` type($pad_value))?
    (`left_padding_num` `=` $left_padding_num^ `:` type($left_padding_num))?
    (`init_out_buffer` `=` $init_out_buffer^ )?
    (`right_padding_num` `=` $right_padding_num^ `:` type($right_padding_num))?
    (`init_condition` `=` $init_condition^ `:` type($init_condition))?
    (`->` qualified(type($result))^)?
  }];
  
  let extraClassDeclaration = [{
    static StringRef getOpName() { return "pto_load"; }
    ShapedType getSrcOperandType() {
       return cast<ShapedType>(getSrc().getType());
    }
    ShapedType getDstOperandType() {
       return cast<ShapedType>(getDst().getType());
    }
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_MTE2;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TPrefetchOp : PTO_TOp<"tprefetch", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Prefetch global data into a local tile buffer.";
  let description = [{
    Prefetches data from a global partition view into a temporary local tile
    buffer. Unlike most PTO intrinsics, TPREFETCH does not carry implicit
    TSYNC(wait_events...) semantics in PTO-ISA.

    DPS form:
      pto.tprefetch ins(%src) outs(%dst)
  }];

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst)) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE2; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TStoreOp: PTO_TOp<"tstore", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO data store operation (TileBuf -> PartitionView).";
  let description = [{
    Stores data from a 2D physical tile buffer back to a high-dimensional logical partition view.
    
    DPS form:
      pto.tstore ins(%tile_buf) outs(%partition_view)
  }];

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    Optional<I64>:$preQuantScalar,
    DefaultValuedAttr<PTO_STPhaseAttr, "::mlir::pto::STPhase::Unspecified">:$stPhase,
    DefaultValuedAttr<PTO_AtomicTypeAttr, "::mlir::pto::AtomicType::AtomicNone">:$atomicType,
    DefaultValuedAttr<PTO_ReluPreModeAttr, "::mlir::pto::ReluPreMode::NoRelu">:$reluPreMode
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src))
           (`,` $preQuantScalar^ `:` type($preQuantScalar))? `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      // pto-isa lowers TSTORE differently depending on the source tile domain:
      //   - ACC (L0C) -> GM uses the FIX pipe (copy_matrix_cc_to_gm).
      //   - VEC/UB and MAT -> GM use the MTE3 pipe.
      //
      // Sync insertion must use the correct pipe, otherwise the generated
      // set_flag/wait_flag pairs won't order the actual hardware instructions.
      auto isAcc = [](Type ty) -> bool {
        if (auto tb = ::mlir::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = ::mlir::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace() == ::mlir::pto::AddressSpace::ACC;
          return false;
        }
        if (auto mr = ::mlir::dyn_cast<::mlir::MemRefType>(ty)) {
          if (auto ms = mr.getMemorySpace()) {
            if (auto as = ::mlir::dyn_cast<::mlir::pto::AddressSpaceAttr>(ms))
              return as.getAddressSpace() == ::mlir::pto::AddressSpace::ACC;
          }
          return false;
        }
        return false;
      };
      if (isAcc(getSrc().getType()))
        return ::mlir::pto::PIPE::PIPE_FIX;
      return ::mlir::pto::PIPE::PIPE_MTE3;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TPackOp : PTO_TOp<"tpack", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Pack a wider vec tile into a narrower vec tile (A5).";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst)) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TTransOp : PTO_TOp<"ttrans", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO matrix trans operation (destination-style, DPS).";
  let description = [{
    DPS form:
      tile buf:  pto.ttrans ins(%src %tmp) outs(%dst)
  }];
  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$tmp,
    PTODpsType:$dst
  );
  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $src `,` $tmp `:` qualified(type($src)) `,` qualified(type($tmp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_V; // Vector Pipe
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getDstMutable();
    }
  }];
}
//===----------------------------------------------------------------------===//
// TMATMUL_BIAS / TMATMUL_MX family
//   - DPS: AnyMemRef
//   - TOp: PTODpsType
//   - All in ins/outs form
//===----------------------------------------------------------------------===//

// -------------------------
// DPS versions (AnyMemRef)
// -------------------------

// -------------------------
// Tile-world TOp versions (PTODpsType)
// -------------------------

def TMatmulBiasOp : PTO_TOp<"tmatmul.bias", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Matmul with bias add (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$a,
    PTODpsType:$b,
    PTODpsType:$bias,
    PTODpsType:$dst
  );

  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;

  //let builders = [
  //  OpBuilder<(ins "TypeRange":$resultTypes, "Value":$a, "Value":$b, "Value":$bias, "Value":$dst), [{
  //    build($_builder, $_state, resultTypes, a, b, bias, dst, ValueRange{});
  //  }]>
  //];

  let assemblyFormat = [{
    `ins` `(` $a `,` $b `,` $bias `:` qualified(type($a)) `,` qualified(type($b)) `,` qualified(type($bias)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TMATMUL_BIAS"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

// pto.tmatmul.mx ins(%a,%a_scale,%b,%b_scale) outs(%dst) -> tensor?
def TMatmulMxOp : PTO_TOp<"tmatmul.mx", [
    PTO_DpsInitOpInterface,
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Matmul mx (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$a,
    PTODpsType:$a_scale,
    PTODpsType:$b,
    PTODpsType:$b_scale,
    PTODpsType:$dst);

  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $a `,` $a_scale `,` $b `,` $b_scale
          `:` type($a) `,` type($a_scale) `,` type($b) `,` type($b_scale) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TMATMUL_MX"; }
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_M;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

// pto.tmatmul.mx.acc ins(%c_in,%a,%a_scale,%b,%b_scale) outs(%dst) -> tensor?
def TMatmulMxAccOp : PTO_TOp<"tmatmul.mx.acc", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Matmul mx accumulate (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$c_in,
    PTODpsType:$a,
    PTODpsType:$a_scale,
    PTODpsType:$b,
    PTODpsType:$b_scale,
    PTODpsType:$dst);

  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $c_in `,` $a `,` $a_scale `,` $b `,` $b_scale
          `:` type($c_in) `,` type($a) `,` type($a_scale) `,` type($b) `,` type($b_scale) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TMATMUL_MX_ACC"; }
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_M;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

// pto.tmatmul.mx.bias ins(%a,%a_scale,%b,%b_scale,%bias) outs(%dst)  -> tensor?
def TMatmulMxBiasOp : PTO_TOp<"tmatmul.mx.bias",[
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Matmul mx with bias (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$a,
    PTODpsType:$a_scale,
    PTODpsType:$b,
    PTODpsType:$b_scale,
    PTODpsType:$bias,
    PTODpsType:$dst);

  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;


  let assemblyFormat = [{
    `ins` `(` $a `,` $a_scale `,` $b `,` $b_scale `,` $bias
          `:` type($a) `,` type($a_scale) `,` type($b) `,` type($b_scale) `,` qualified(type($bias)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TMATMUL_MX_BIAS"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TMatmulOp : PTO_TOp<"tmatmul", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO matrix multiplication operation (optional bias), destination-style (DPS) in tile world.";

  let arguments = (ins
    PTODpsType:$lhs,
    PTODpsType:$rhs,
    PTODpsType:$dst
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` qualified(type($lhs)) `,` qualified(type($rhs)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let extraClassDeclaration = [{
    static StringRef getOpName() { return "matmul_dps"; }
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_M; 
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TMatmulAccOp : PTO_TOp<"tmatmul.acc", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO matrix multiplication accumulate operation, destination-style (DPS, no bias).";

  let arguments = (ins
    PTODpsType:$acc_in,
    PTODpsType:$lhs,
    PTODpsType:$rhs,
    PTODpsType:$dst
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $acc_in `,` $lhs `,` $rhs `:` qualified(type($acc_in) ) `,` qualified(type($lhs)) `,` qualified(type($rhs)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let extraClassDeclaration = [{
      ::mlir::pto::PIPE getPipe() {
        return ::mlir::pto::PIPE::PIPE_M; 
      }
      ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGemvOp : PTO_TOp<"tgemv", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO matrix-vector multiplication operation (optional bias), destination-style (DPS) in tile world.";

  let arguments = (ins
    PTODpsType:$lhs,
    PTODpsType:$rhs,
    PTODpsType:$dst
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` qualified(type($lhs)) `,` qualified(type($rhs)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let extraClassDeclaration = [{
    static StringRef getOpName() { return "gemv"; }
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_M; 
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGemvAccOp : PTO_TOp<"tgemv.acc", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO matrix-vector multiplication accumulate operation, destination-style (DPS, no bias).";

  let arguments = (ins
    PTODpsType:$acc_in,
    PTODpsType:$lhs,
    PTODpsType:$rhs,
    PTODpsType:$dst
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $acc_in `,` $lhs `,` $rhs `:` qualified(type($acc_in) ) `,` qualified(type($lhs)) `,` qualified(type($rhs)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let extraClassDeclaration = [{
      ::mlir::pto::PIPE getPipe() {
        return ::mlir::pto::PIPE::PIPE_M; 
      }
      ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGemvBiasOp : PTO_TOp<"tgemv.bias", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "GEMV with bias add (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType :$a,
    PTODpsType :$b,
    PTODpsType :$bias,
    PTODpsType :$dst
  );

  let results = (outs Optional<AnyRankedTensor>:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $a `,` $b `,` $bias `:` qualified(type($a)) `,` qualified(type($b)) `,` qualified(type($bias)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TGEMV_BIAS"; }
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_M; 
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGemvMxOp : PTO_TOp<"tgemv.mx", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Mixed-precision GEMV with scale tiles (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$a,
    PTODpsType:$a_scale,
    PTODpsType:$b,
    PTODpsType:$b_scale,
    PTODpsType:$dst
  );

  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $a `,` $a_scale `,` $b `,` $b_scale
          `:` type($a) `,` type($a_scale) `,` type($b) `,` type($b_scale) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TGEMV_MX"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGemvMxAccOp : PTO_TOp<"tgemv.mx.acc", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Mixed-precision GEMV accumulate with scale tiles (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$c_in,
    PTODpsType:$a,
    PTODpsType:$a_scale,
    PTODpsType:$b,
    PTODpsType:$b_scale,
    PTODpsType:$dst
  );

  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $c_in `,` $a `,` $a_scale `,` $b `,` $b_scale
          `:` type($c_in) `,` type($a) `,` type($a_scale) `,` type($b) `,` type($b_scale) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TGEMV_MX"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGemvMxBiasOp : PTO_TOp<"tgemv.mx.bias", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Mixed-precision GEMV with bias and scale tiles (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$a,
    PTODpsType:$a_scale,
    PTODpsType:$b,
    PTODpsType:$b_scale,
    PTODpsType:$bias,
    PTODpsType:$dst
  );

  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $a `,` $a_scale `,` $b `,` $b_scale `,` $bias
          `:` type($a) `,` type($a_scale) `,` type($b) `,` type($b_scale) `,` qualified(type($bias)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TGEMV_MX"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TMovOp  : PTO_TOp<"tmov", [
  AttrSizedOperandSegments,
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Move data between domains (DPS version).";
  
  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    Optional<PTODpsType>:$fp,
    Optional<I64>:$preQuantScalar,
    OptionalAttr<PTO_AccToVecModeAttr>:$accToVecMode,
    DefaultValuedAttr<PTO_ReluPreModeAttr, "::mlir::pto::ReluPreMode::NoRelu">:$reluPreMode
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src))
    (`,` $fp^ `:` qualified(type($fp)))?
    (`,` $preQuantScalar^ `:` type($preQuantScalar))?
    `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      // TMOV spans multiple hardware pipelines depending on the source/dest
      // domains. Most tile-domain moves are executed by MTE1 (e.g. MAT->L0),
      // while UB->UB copies are vector-pipe operations.
      auto getASFromType = [](Type ty)
          -> std::optional<::mlir::pto::AddressSpace> {
        // Pre-lowering: tile_buf carries the address space in its memorySpace.
        if (auto tb = llvm::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = llvm::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace();
          return std::nullopt;
        }
        // Post PTOViewToMemref: tile_buf is erased to memref but memorySpace is
        // preserved in memref's memorySpace attribute.
        if (auto mr = llvm::dyn_cast<::mlir::MemRefType>(ty)) {
          if (auto ms = mr.getMemorySpace()) {
            if (auto as =
                    llvm::dyn_cast<::mlir::pto::AddressSpaceAttr>(ms))
              return as.getAddressSpace();
          }
          return std::nullopt;
        }
        return std::nullopt;
      };

      auto sOpt = getASFromType(getSrc().getType());
      auto dOpt = getASFromType(getDst().getType());
      if (!sOpt.has_value() || !dOpt.has_value())
        return ::mlir::pto::PIPE::PIPE_V;

      const auto s = sOpt.value();
      const auto d = dOpt.value();

      // UB -> UB copy is vector pipe.
      if (s == ::mlir::pto::AddressSpace::VEC && d == ::mlir::pto::AddressSpace::VEC) {
        return ::mlir::pto::PIPE::PIPE_V;
      }

      // MAT -> L0 (Left/Right/Bias/Scaling) are MTE1 moves.
      if ((s == ::mlir::pto::AddressSpace::MAT &&
           (d == ::mlir::pto::AddressSpace::LEFT || d == ::mlir::pto::AddressSpace::RIGHT ||
            d == ::mlir::pto::AddressSpace::BIAS || d == ::mlir::pto::AddressSpace::SCALING))) {
        return ::mlir::pto::PIPE::PIPE_MTE1;
      }

      // ACC -> MAT / VEC use FIX in PTO-ISA sync mapping (TMOV_A2M/TMOV_A2V).
      if (s == ::mlir::pto::AddressSpace::ACC &&
          (d == ::mlir::pto::AddressSpace::MAT ||
           d == ::mlir::pto::AddressSpace::VEC)) {
        return ::mlir::pto::PIPE::PIPE_FIX;
      }

      // Fallback: treat as vector pipe (safe default for most intra-domain moves).
      return ::mlir::pto::PIPE::PIPE_V;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}


//===----------------------------------------------------------------------===//
// Pointer Cast Op (existing)
//===----------------------------------------------------------------------===//

def PointerCastOp : PTO_Op<"pointer_cast", [AttrSizedOperandSegments, Pure]> {
  let summary = "Casts an integer address to a MemRef with optional valid dims";
  
  // 参数定义 (保持 Optional)
  let arguments = (ins 
    Variadic<I64>:$addrs,
    Optional<Index>:$valid_row,
    Optional<Index>:$valid_col,
    OptionalAttr<TileBufConfigAttr>:$config
  );
  
  let results = (outs Res<AnyMemRef, "", [MemAlloc]>:$result);

  // Assembly Format (去掉了 [])
  let assemblyFormat = [{
    `(` $addrs `)` ($valid_row^ `,` $valid_col)? attr-dict `:` qualified(type($result))
  }];

  // Builder (修复 addOperand -> addOperands)
  let builders = [
    OpBuilder<(ins "Type":$result, "ValueRange":$addrs, "Value":$vRow, "Value":$vCol, "Attribute":$config), [{
      $_state.addTypes(result);
      $_state.addOperands(addrs);
      // [关键修复] addOperand -> addOperands
      if (vRow) $_state.addOperands(vRow);
      if (vCol) $_state.addOperands(vCol);
      if (config) $_state.addAttribute("config", config);
      
      int32_t addrsSize = addrs.size();
      int32_t vRowSize = vRow ? 1 : 0;
      int32_t vColSize = vCol ? 1 : 0;
      $_state.addAttribute("operandSegmentSizes", 
        $_builder.getDenseI32ArrayAttr({addrsSize, vRowSize, vColSize}));
    }]>
  ];
}

// =============================================================================
// System/Runtime Query Ops
// =============================================================================

def GetBlockIdxOp : PTO_Op<"get_block_idx", [Pure]> {
  let summary = "Get the current block index (core ID).";
  let description = [{
    Returns the linear index of the current compute unit (CUBE Core) within the task.
    The return value is in the range [0, BlockNum - 1].
  }];

  let arguments = (ins);
  let results = (outs I64:$result);

  let assemblyFormat = "attr-dict";
}

def GetSubBlockIdxOp : PTO_Op<"get_subblock_idx", [Pure]> {
  let summary = "Get the current vector core ID.";
  let description = [{
    Returns the ID of the current compute unit (Vector Core).
    The return value is in the range [0, 1].
  }];

  let arguments = (ins);
  let results = (outs I64:$result);

  let assemblyFormat = "attr-dict";
}

def GetBlockNumOp : PTO_Op<"get_block_num", [Pure]> {
  let summary = "Get the total number of blocks (cores).";
  let description = [{
    Returns the total number of compute units (Blocks) configured for the current task.
  }];

  let arguments = (ins);
  let results = (outs I64:$result);

  let assemblyFormat = "attr-dict";
}

def GetSubBlockNumOp : PTO_Op<"get_subblock_num", [Pure]> {
  let summary = "Get the number of vector cores.";
  let description = [{
    Returns the total number of vector compute units.
  }];

  let arguments = (ins);
  let results = (outs I64:$result);

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// High-Level Synchronization Ops
//===----------------------------------------------------------------------===//

def PTO_PipeEventTypeLikeAttr : AnyAttrOf<[PTO_PipeEventTypeAttr, PTO_SyncOpTypeAttr]>;

def RecordEventOp : PTO_Op<"record_event"> {
  let summary = "Record an event for synchronization (High Level)";
  let description = [{
    Records an event from a source operation type to a destination operation type.
    Will be lowered to `pto.set_flag` based on op-to-pipe mapping.
  }];
  let arguments = (ins
    PTO_PipeEventTypeLikeAttr:$src_op,
    PTO_PipeEventTypeLikeAttr:$dst_op,
    PTO_EventAttr:$event_id
  );
  let results = (outs);
  let assemblyFormat = [{
    `[` $src_op `,` $dst_op `,` $event_id `]` attr-dict
  }];
}

def WaitEventOp : PTO_Op<"wait_event"> {
  let summary = "Wait for an event (High Level)";
  let description = [{
    Waits for an event from a source operation type to a destination operation type.
    Will be lowered to `pto.wait_flag` based on op-to-pipe mapping.
  }];
  let arguments = (ins
    PTO_PipeEventTypeLikeAttr:$src_op,
    PTO_PipeEventTypeLikeAttr:$dst_op,
    PTO_EventAttr:$event_id
  );
  let results = (outs);
  let assemblyFormat = [{
    `[` $src_op `,` $dst_op `,` $event_id `]` attr-dict
  }];
}

// High-Level Barrier (single pipe) with op type mapping
def BarrierSyncOp : PTO_Op<"barrier_sync"> {
  let summary = "High-level barrier mapped from SyncOpType to PIPE";
  let description = [{
    A convenience barrier that specifies a SyncOpType instead of PIPE. The lowering
    pass maps the op type to the corresponding hardware pipe and emits `pto.barrier`.
  }];
  let arguments = (ins
    PTO_SyncOpTypeAttr:$op_type
  );
  let results = (outs);
  let assemblyFormat = [{
    `[` $op_type `]` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Section Ops (Macros Containers)
//===----------------------------------------------------------------------===//

class PTO_SectionOp<string mnemonic>
    : PTO_Op<mnemonic, [SingleBlock, NoTerminator]> {
  let summary = "Container for core-specific code guarded by macros";
  let description = [{
    During conversion to EmitC, this op is lowered to:
    emitc.verbatim("#if defined(MACRO)")
    ... inlined body ...
    emitc.verbatim("#endif")
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
}

def SectionCubeOp : PTO_SectionOp<"section.cube">;
def SectionVectorOp : PTO_SectionOp<"section.vector">;

//===----------------------------------------------------------------------===//
// Frontend TPUSH/TPOP Pipe Communication Ops
//===----------------------------------------------------------------------===//

def AicInitializePipeOp : PTO_Op<"aic_initialize_pipe"> {
  let summary = "Frontend Cube-kernel pipe initialization";

  let arguments = (ins
      I8Attr:$dir_mask,
      I32Attr:$slot_size,
      Optional<PtrType>:$gm_slot_buffer,
      I32:$c2v_consumer_buf,
      I32:$v2c_consumer_buf
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `{` `dir_mask` `=` $dir_mask `,`
        `slot_size` `=` $slot_size `}`
    `(`
      (`gm_slot_buffer` `=` $gm_slot_buffer^ `:` type($gm_slot_buffer) `,`)?
      `c2v_consumer_buf` `=` $c2v_consumer_buf `:` type($c2v_consumer_buf) `,`
      `v2c_consumer_buf` `=` $v2c_consumer_buf `:` type($v2c_consumer_buf)
    `)`
    attr-dict
  }];
}

def AivInitializePipeOp : PTO_Op<"aiv_initialize_pipe"> {
  let summary = "Frontend Vector-kernel pipe initialization";

  let arguments = (ins
      I8Attr:$dir_mask,
      I32Attr:$slot_size,
      Optional<PtrType>:$gm_slot_buffer,
      I32:$c2v_consumer_buf,
      I32:$v2c_consumer_buf
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `{` `dir_mask` `=` $dir_mask `,`
        `slot_size` `=` $slot_size `}`
    `(`
      (`gm_slot_buffer` `=` $gm_slot_buffer^ `:` type($gm_slot_buffer) `,`)?
      `c2v_consumer_buf` `=` $c2v_consumer_buf `:` type($c2v_consumer_buf) `,`
      `v2c_consumer_buf` `=` $v2c_consumer_buf `:` type($v2c_consumer_buf)
    `)`
    attr-dict
  }];
}

def TPushToAivOp : PTO_TOp<"tpush_to_aiv"> {
  let summary = "Frontend C2V producer push from Cube to Vector";

  let arguments = (ins
      PTODpsType:$tile,
      I8Attr:$split
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $tile `:` qualified(type($tile)) `)`
    `{` `split` `=` $split `}` attr-dict
  }];
}

def TPushToAicOp : PTO_TOp<"tpush_to_aic"> {
  let summary = "Frontend V2C producer push from Vector to Cube";

  let arguments = (ins
      PTODpsType:$tile,
      I8Attr:$split
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $tile `:` qualified(type($tile)) `)`
    `{` `split` `=` $split `}` attr-dict
  }];
}

def TPopFromAicOp : PTO_TOp<"tpop_from_aic"> {
  let summary = "Frontend C2V consumer pop in Vector kernel";

  let arguments = (ins
      I8Attr:$split
  );

  let results = (outs PTODpsType:$tile);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `{` `split` `=` $split `}` attr-dict `->` qualified(type($tile))
  }];
}

def TPopFromAivOp : PTO_TOp<"tpop_from_aiv"> {
  let summary = "Frontend V2C consumer pop in Cube kernel";

  let arguments = (ins
      I8Attr:$split
  );

  let results = (outs PTODpsType:$tile);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `{` `split` `=` $split `}` attr-dict `->` qualified(type($tile))
  }];
}

def TFreeFromAicOp : PTO_TOp<"tfree_from_aic"> {
  let summary = "Frontend C2V consumer free in Vector kernel";

  let arguments = (ins
      I8Attr:$split
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `{` `split` `=` $split `}` attr-dict
  }];
}

def TFreeFromAivOp : PTO_TOp<"tfree_from_aiv"> {
  let summary = "Frontend V2C consumer free in Cube kernel";

  let arguments = (ins
      I8Attr:$split
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `{` `split` `=` $split `}` attr-dict
  }];
}

def ReserveBufferOp : PTO_Op<"reserve_buffer"> {
  let summary = "Reserve a local consumer slot buffer";

  let arguments = (ins
      StrAttr:$name,
      I32Attr:$size,
      PTO_AddressSpaceAttr:$location,
      BoolAttr:$autoAlloc,
      OptionalAttr<I32Attr>:$base
  );

  let results = (outs I32:$addr);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `{` `name` `=` $name `,`
        `size` `=` $size `,`
        `location` `=` $location `,`
        `auto` `=` $autoAlloc
        (`,` `base` `=` $base^)?
        `}` attr-dict `->` type($addr)
  }];
}

def ImportReservedBufferOp : PTO_Op<"import_reserved_buffer"> {
  let summary = "Import a peer reserved local consumer slot buffer";

  let arguments = (ins
      StrAttr:$name,
      FlatSymbolRefAttr:$peer_func
  );

  let results = (outs I32:$addr);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `{` `name` `=` $name `,`
        `peer_func` `=` $peer_func
        `}` attr-dict `->` type($addr)
  }];
}

//===----------------------------------------------------------------------===//
// TPUSH/TPOP Pipe Communication Ops
//===----------------------------------------------------------------------===//

def BuildAsyncSessionOp : PTO_Op<"build_async_session", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Build an async DMA session handle for TPUT_ASYNC/TGET_ASYNC";

  let arguments = (ins
      TileBufOrMemRef:$scratch,
      ScalarPtrOrMemRef:$workspace,
      OptionalAttr<I32Attr>:$sync_id,
      OptionalAttr<I64Attr>:$block_bytes,
      OptionalAttr<I64Attr>:$comm_block_offset,
      OptionalAttr<I32Attr>:$queue_num,
      OptionalAttr<I64Attr>:$channel_group_idx
  );

  let results = (outs AsyncSessionType:$session);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $scratch `,` $workspace `:` qualified(type($scratch)) `,` type($workspace) `)`
    attr-dict `->` qualified(type($session))
  }];
}

def TPutAsyncOp : PTO_Op<"tput_async", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Asynchronous remote write from local GM to remote GM";

  let arguments = (ins
      PTODpsType:$dst,
      PTODpsType:$src,
      AsyncSessionType:$session
  );

  let results = (outs AsyncEventType:$event);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $dst `,` $src `,` $session `:`
        type($dst) `,` type($src) `,` qualified(type($session)) `)`
    attr-dict `->` qualified(type($event))
  }];
}

def TGetAsyncOp : PTO_Op<"tget_async", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Asynchronous remote read from remote GM to local GM";

  let arguments = (ins
      PTODpsType:$dst,
      PTODpsType:$src,
      AsyncSessionType:$session
  );

  let results = (outs AsyncEventType:$event);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $dst `,` $src `,` $session `:`
        type($dst) `,` type($src) `,` qualified(type($session)) `)`
    attr-dict `->` qualified(type($event))
  }];
}

def WaitAsyncEventOp : PTO_Op<"wait_async_event", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Block until an async DMA event completes";

  let arguments = (ins
      AsyncEventType:$event,
      AsyncSessionType:$session
  );

  let results = (outs I1:$completed);

  let assemblyFormat = [{
    `(` $event `,` $session `:`
        qualified(type($event)) `,` qualified(type($session)) `)`
    attr-dict `->` type($completed)
  }];
}

def TestAsyncEventOp : PTO_Op<"test_async_event", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Non-blocking completion test for an async DMA event";

  let arguments = (ins
      AsyncEventType:$event,
      AsyncSessionType:$session
  );

  let results = (outs I1:$completed);

  let assemblyFormat = [{
    `(` $event `,` $session `:`
        qualified(type($event)) `,` qualified(type($session)) `)`
    attr-dict `->` type($completed)
  }];
}

def TPutOp : PTO_Op<"tput", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Synchronous remote write from local GM to remote GM";
  let arguments = (ins
      PTODpsType:$dst,
      PTODpsType:$src,
      PTODpsType:$ping,
      Optional<PTODpsType>:$pong,
      DefaultValuedAttr<PTO_AtomicTypeAttr, "::mlir::pto::AtomicType::AtomicNone">:$atomicType
  );
  let results = (outs);
  let hasVerifier = 1;
}

def TGetOp : PTO_Op<"tget", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Synchronous remote read from remote GM to local GM";
  let arguments = (ins
      PTODpsType:$dst,
      PTODpsType:$src,
      PTODpsType:$ping,
      Optional<PTODpsType>:$pong
  );
  let results = (outs);
  let hasVerifier = 1;
}

def TNotifyOp : PTO_Op<"tnotify", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Send a signal notification to remote GM";
  let arguments = (ins
      PTODpsType:$signal,
      AnySignlessInteger:$value,
      PTO_NotifyOpAttr:$notifyOp
  );
  let results = (outs);
  let hasVerifier = 1;
}

def TWaitOp : PTO_Op<"twait", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Block until signal(s) satisfy a comparison";
  let arguments = (ins
      PTODpsType:$signal,
      AnySignlessInteger:$cmpValue,
      PTO_WaitCmpAttr:$cmp
  );
  let results = (outs);
  let hasVerifier = 1;
}

def TTestOp : PTO_Op<"ttest", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Non-blocking signal comparison test";
  let arguments = (ins
      PTODpsType:$signal,
      AnySignlessInteger:$cmpValue,
      PTO_WaitCmpAttr:$cmp
  );
  let results = (outs I1:$result);
  let hasVerifier = 1;
}

def TBroadcastOp : PTO_Op<"tbroadcast", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Broadcast local GM data to all group members";
  let arguments = (ins
      PTODpsType:$src,
      PTODpsType:$ping,
      Optional<PTODpsType>:$pong,
      Variadic<PTODpsType>:$group,
      I32Attr:$root
  );
  let results = (outs);
  let hasVerifier = 1;
}

def CommTGatherOp : PTO_Op<"comm_tgather", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Gather remote GM data from a communication group";
  let arguments = (ins
      PTODpsType:$dst,
      PTODpsType:$ping,
      Optional<PTODpsType>:$pong,
      Variadic<PTODpsType>:$group,
      I32Attr:$root
  );
  let results = (outs);
  let hasVerifier = 1;
}

def CommTScatterOp : PTO_Op<"comm_tscatter", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Scatter local GM data to a communication group";
  let arguments = (ins
      PTODpsType:$src,
      PTODpsType:$ping,
      Optional<PTODpsType>:$pong,
      Variadic<PTODpsType>:$group,
      I32Attr:$root
  );
  let results = (outs);
  let hasVerifier = 1;
}

def TReduceOp : PTO_Op<"treduce", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Reduce remote GM data from a communication group";
  let arguments = (ins
      PTODpsType:$dst,
      PTODpsType:$acc,
      PTODpsType:$recvPing,
      Optional<PTODpsType>:$recvPong,
      Variadic<PTODpsType>:$group,
      PTO_ReduceOpAttr:$reduceOp,
      I32Attr:$root
  );
  let results = (outs);
  let hasVerifier = 1;
}

def InitializeL2G2LPipeOp : PTO_Op<"initialize_l2g2l_pipe", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Initialize a local-to-global-to-local pipe handle";

  let arguments = (ins
      I8Attr:$dir_mask,
      I32Attr:$slot_size,
      I32Attr:$slot_num,
      OptionalAttr<I32Attr>:$local_slot_num,
      OptionalAttr<I32Attr>:$flag_base,
      AnyType:$gm_addr,
      AnyType:$local_addr,
      Optional<AnyType>:$peer_local_addr
  );

  let results = (outs PipeType:$pipe);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `{` `dir_mask` `=` $dir_mask `,`
        `slot_size` `=` $slot_size `,`
        `slot_num` `=` $slot_num
        (`,` `local_slot_num` `=` $local_slot_num^)?
        (`,` `flag_base` `=` $flag_base^)?
        `}`
    `(` $gm_addr `:` type($gm_addr) `,` $local_addr `:` type($local_addr)
        (`,` $peer_local_addr^ `:` type($peer_local_addr))? `)`
    attr-dict `->` qualified(type($pipe))
  }];
}

def InitializeL2LPipeOp : PTO_Op<"initialize_l2l_pipe", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Initialize a local-to-local pipe handle";

  let arguments = (ins
      I8Attr:$dir_mask,
      I32Attr:$slot_size,
      I32Attr:$slot_num,
      OptionalAttr<I32Attr>:$flag_base,
      AnyType:$local_addr,
      Optional<AnyType>:$peer_local_addr
  );

  let results = (outs PipeType:$pipe);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `{` `dir_mask` `=` $dir_mask `,`
        `slot_size` `=` $slot_size `,`
        `slot_num` `=` $slot_num
        (`,` `flag_base` `=` $flag_base^)?
        `}`
    `(` $local_addr `:` type($local_addr)
        (`,` $peer_local_addr^ `:` type($peer_local_addr))? `)`
    attr-dict `->` qualified(type($pipe))
  }];
}

def TPushOp : PTO_TOp<"tpush", [
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Push tile data through a unified pipe handle";

  let arguments = (ins
      PTODpsType:$tile,
      PipeType:$pipe_handle,
      I8Attr:$split
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $tile `,` $pipe_handle `:` qualified(type($tile)) `,` qualified(type($pipe_handle)) `)`
    `{` `split` `=` $split `}` attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      auto getAddressSpace = [](Type ty) -> std::optional<::mlir::pto::AddressSpace> {
        if (auto tb = ::mlir::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = ::mlir::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace();
          return std::nullopt;
        }
        if (auto mr = ::mlir::dyn_cast<::mlir::MemRefType>(ty)) {
          if (auto ms = mr.getMemorySpace()) {
            if (auto as = ::mlir::dyn_cast<::mlir::pto::AddressSpaceAttr>(ms))
              return as.getAddressSpace();
          }
          return std::nullopt;
        }
        return std::nullopt;
      };

      auto as = getAddressSpace(getTile().getType());
      if (!as)
        return ::mlir::pto::PIPE::PIPE_UNASSIGNED;
      if (*as == ::mlir::pto::AddressSpace::ACC)
        return ::mlir::pto::PIPE::PIPE_FIX;
      if (*as == ::mlir::pto::AddressSpace::VEC)
        return ::mlir::pto::PIPE::PIPE_MTE3;
      return ::mlir::pto::PIPE::PIPE_UNASSIGNED;
    }
  }];
}

def DeclareTileOp : PTO_Op<"declare_tile", [Pure]> {
  let summary = "Declare a tile value whose address will be assigned later";

  let results = (outs PTODpsType:$tile);

  let assemblyFormat = [{
    attr-dict `->` qualified(type($tile))
  }];
}

def TAssignOp : PTO_Op<"tassign", [Pure]> {
  let summary = "Rebind a declared tile handle to a runtime address";
  let description = [{
    Rebinds an existing tile handle to a new address. This op is intended for
    manual synchronization workflows (for example, rebinding inside loops while
    keeping the tile declaration outside the loop).
  }];

  let arguments = (ins
    PTODpsType:$tile,
    I64:$addr
  );
  let results = (outs PTODpsType:$result);

  let assemblyFormat = [{
    $tile `,` $addr attr-dict `:` qualified(type($tile)) `->` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::LogicalResult verify();
  }];
}

def DeclareEventIdArrayOp : PTO_Op<"declare_eventid_array"> {
  let summary = "Declare a manual-only local event-id array";
  let description = [{
    Declares a local array used to store dynamic event ids in manual sync code.
    This is a frontend/EmitC convenience feature and is not intended to
    participate in PlanMemory or automatic synchronization passes.
  }];

  let results = (outs EventIdArrayType:$array);

  let assemblyFormat = [{
    attr-dict `->` qualified(type($array))
  }];
}

def EventIdArrayGetOp : PTO_Op<"eventid_array_get"> {
  let summary = "Read one element from a manual event-id array";

  let arguments = (ins
    EventIdArrayType:$array,
    Index:$index
  );
  let results = (outs Index:$result);

  let assemblyFormat = [{
    $array `[` $index `]` attr-dict `:` qualified(type($array))
  }];
}

def EventIdArraySetOp : PTO_Op<"eventid_array_set"> {
  let summary = "Write one element into a manual event-id array";

  let arguments = (ins
    EventIdArrayType:$array,
    Index:$index,
    Index:$value
  );
  let results = (outs);

  let assemblyFormat = [{
    $array `[` $index `]` `,` $value attr-dict `:` qualified(type($array))
  }];
}

def DeclareTileMemRefOp : PTO_Op<"declare_tile_memref", [Pure]> {
  let summary = "Internal memref placeholder for a tile whose address is assigned later";
  let description = [{
    Internal lowering op used by PTOViewToMemref. This op does not allocate
    storage; it only provides a memref-typed SSA handle so later passes can
    attach tile metadata through pto.bind_tile before the address is filled by
    pipe operations such as pto.tpop.
  }];

  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    attr-dict `->` qualified(type($result))
  }];
}

def TPopOp : PTO_TOp<"tpop", [
    PTO_DpsInitOpInterface,
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Bind a declared consumer tile to the next FIFO slot";

  let arguments = (ins
      PTODpsType:$tile,
      PipeType:$pipe_handle,
      I8Attr:$split
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $tile `,` $pipe_handle `:` qualified(type($tile)) `,` qualified(type($pipe_handle)) `)`
    `{` `split` `=` $split `}` attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      auto getAddressSpace = [](Type ty) -> std::optional<::mlir::pto::AddressSpace> {
        if (auto tb = ::mlir::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = ::mlir::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace();
          return std::nullopt;
        }
        if (auto mr = ::mlir::dyn_cast<::mlir::MemRefType>(ty)) {
          if (auto ms = mr.getMemorySpace()) {
            if (auto as = ::mlir::dyn_cast<::mlir::pto::AddressSpaceAttr>(ms))
              return as.getAddressSpace();
          }
          return std::nullopt;
        }
        return std::nullopt;
      };

      auto isA5Target = [&]() -> bool {
        auto moduleOp = getOperation()->getParentOfType<::mlir::ModuleOp>();
        if (!moduleOp)
          return false;

        if (auto arch =
                moduleOp->getAttrOfType<::mlir::StringAttr>("pto.target_arch")) {
          if (arch.getValue().equals_insensitive("a5"))
            return true;
        }
        return false;
      };

      auto as = getAddressSpace(getTile().getType());
      if (!as)
        return ::mlir::pto::PIPE::PIPE_UNASSIGNED;

      switch (*as) {
      case ::mlir::pto::AddressSpace::VEC:
      case ::mlir::pto::AddressSpace::MAT:
        return isA5Target() ? ::mlir::pto::PIPE::PIPE_S
                            : ::mlir::pto::PIPE::PIPE_MTE2;
      default:
        break;
      }
      return ::mlir::pto::PIPE::PIPE_UNASSIGNED;
    }

    ::mlir::MutableOperandRange getDpsInitsMutable() { return getTileMutable(); }
  }];
}

def TFreeOp : PTO_TOp<"tfree", [
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Release the current consumer slot for a pipe";

  let arguments = (ins
      PipeType:$pipe_handle,
      I8Attr:$split
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $pipe_handle `:` qualified(type($pipe_handle)) `)`
    `{` `split` `=` $split `}` attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_S; }
  }];
}

//===----------------------------------------------------------------------===//
// Synchronization Ops
//===----------------------------------------------------------------------===//

def SetFlagOp : PTO_Op<"set_flag"> {
  let summary = "Set synchronization flag between pipes";
  let arguments = (ins
    PTO_PipeAttr:$src_pipe,
    PTO_PipeAttr:$dst_pipe,
    PTO_EventAttr:$event_id
  );
  let results = (outs);
  let assemblyFormat = [{
    `[` $src_pipe `,` $dst_pipe `,` $event_id `]` attr-dict
  }];
}

def WaitFlagOp : PTO_Op<"wait_flag"> {
  let summary = "Wait for synchronization flag";
  let arguments = (ins
    PTO_PipeAttr:$src_pipe,
    PTO_PipeAttr:$dst_pipe,
    PTO_EventAttr:$event_id
  );
  let results = (outs);
  let assemblyFormat = [{
    `[` $src_pipe `,` $dst_pipe `,` $event_id `]` attr-dict
  }];
}

def SetFlagDynOp : PTO_Op<"set_flag_dyn"> {
  let summary = "Set synchronization flag between pipes (dynamic event id)";
  let arguments = (ins
    PTO_PipeAttr:$src_pipe,
    PTO_PipeAttr:$dst_pipe,
    Index:$event_id
  );
  let results = (outs);
  let assemblyFormat = [{
    `[` $src_pipe `,` $dst_pipe `,` $event_id `]` attr-dict
  }];
}

def WaitFlagDynOp : PTO_Op<"wait_flag_dyn"> {
  let summary = "Wait for synchronization flag (dynamic event id)";
  let arguments = (ins
    PTO_PipeAttr:$src_pipe,
    PTO_PipeAttr:$dst_pipe,
    Index:$event_id
  );
  let results = (outs);
  let assemblyFormat = [{
    `[` $src_pipe `,` $dst_pipe `,` $event_id `]` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Buffer-ID Synchronization (A5)
//===----------------------------------------------------------------------===//

def GetBufOp : PTO_Op<"get_buf"> {
  let summary = "Acquire a buffer-id token for a sync op type (A5)";
  let description = [{
    `pto.get_buf` participates in a buffer-id based ordering model. Operations
    mapped to the same pipe and guarded by the same buffer-id are enforced to
    execute in program order relative to other mapped pipes using the same
    buffer-id.

    Like `pto.record_event` / `pto.wait_event`, this op accepts a high-level
    op type endpoint (`pipe_event_type` / `sync_op_type`) instead of a concrete
    pipe name.

    This op is intended to be lowered to the CCEC builtin intrinsic `get_buf`.
  }];

  let arguments = (ins
    PTO_PipeEventTypeLikeAttr:$op_type,
    I32Attr:$buf_id,
    DefaultValuedAttr<I32Attr, "0">:$mode
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `[` $op_type `,` $buf_id `]` attr-dict
  }];
}

def RlsBufOp : PTO_Op<"rls_buf"> {
  let summary = "Release a buffer-id token for a sync op type (A5)";
  let description = [{
    Releases the previously acquired buffer-id token for the given op type
    (mapped to a concrete pipe).

    This op is intended to be lowered to the CCEC builtin intrinsic `rls_buf`.
  }];

  let arguments = (ins
    PTO_PipeEventTypeLikeAttr:$op_type,
    I32Attr:$buf_id,
    DefaultValuedAttr<I32Attr, "0">:$mode
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `[` $op_type `,` $buf_id `]` attr-dict
  }];
}

def SyncSetOp : PTO_Op<"sync.set"> {
  let summary = "Set a synchronization signal (trigger) between cube and vector.";
  let description = [{
    Sets a synchronization signal on the specified pipeline stage.
    Corresponds to `ffts_cross_core_sync` (A3) or `set_intra_block` (A5).
    On A3, the optional `ffts_mode` attribute selects the first argument passed
    to `getFFTSMsg`; if omitted, the legacy `FFTS_MODE_VAL` path is preserved.
  }];

  let arguments = (ins
    PTO_PipeAttr:$pipe,
    OptionalAttr<I32Attr>:$event_id,
    OptionalAttr<I32Attr>:$ffts_mode,
    Optional<Index>:$event_id_dyn
  );

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def SyncWaitOp : PTO_Op<"sync.wait"> {
  let summary = "Wait for a synchronization signal (barrier) between cube and vector.";
  let description = [{
    Waits for a synchronization signal on the specified pipeline stage.
    Corresponds to `wait_flag_dev` (A3) or `wait_intra_block` (A5).
  }];

  let arguments = (ins
    PTO_PipeAttr:$pipe,
    OptionalAttr<I32Attr>:$event_id,
    Optional<Index>:$event_id_dyn
  );

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def BarrierOp : PTO_Op<"barrier"> {
  let summary = "Intra-pipeline memory barrier";
  let arguments = (ins PTO_PipeAttr:$pipe);
  let assemblyFormat = "$pipe attr-dict";
}


//===----------------------------------------------------------------------===//
// FFT Configuration Operation
//===----------------------------------------------------------------------===//

def SetFFTsOp : PTO_Op<"set_ffts", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Set FFTS/flags pointer for runtime (side-effecting).";
  let arguments = (ins AnyMemRef:$ffts);
  let results = (outs);

  let assemblyFormat = "$ffts attr-dict `:` type($ffts)";

  let hasVerifier = 1;
}


def PrintOp : PTO_Op<"print", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Print debug: format string (attribute) and scalar value.";
  let description = [{
    Debug print op. First argument is the format string (string attribute), second is a scalar value.
    Format in IR: `pto.print ins("format", %scalar : type(%scalar))`
  }];
  let arguments = (ins StrAttr:$format, ScalarType:$scalar);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $format `,` $scalar `:` type($scalar) `)`
    attr-dict
  }];
}

def TrapOp : PTO_Op<"trap"> {
  let summary = "Trap: abort execution (no operands).";
  let description = [{
    Inserts a trap to stop execution. No arguments, no results.
    Format in IR: pto.trap
  }];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

// ---- tile-world TOp version (with 't') ----
// pto.tmgather ins(%mem, %idx) outs(%dst) [ ...] -> tensor?
def MGatherOp : PTO_TOp<"mgather", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Gather-load elements from memory into a tile using per-element indices (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$mem,
    PTODpsType:$idx,
    PTODpsType:$dst);

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $mem `,` $idx `:` type($mem) `,` type($idx) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "MGATHER"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE2; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}



// ---- tile-world TOp version ----
// pto.tsetval ins(%offset, %val) outs(%dst) : PTODpsType, index, T
def TSetValOp : PTO_TOp<"tsetval", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Write a scalar value into a single element of dst at offset (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$dst,
    Index:$offset,
    ScalarType:$val
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $offset `,` $val `:` type($offset) `,` type($val) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "SETVAL"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

// ---- tile-world TOp version ----
// pto.tgetval ins(%src, %offset) outs(%dst) : PTODpsType, index -> T
def TGetValOp : PTO_TOp<"tgetval", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Read a single element from tile-like src at offset into a scalar (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$src,
    Index:$offset
  );

  let results = (outs ScalarType:$dst);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $offset `:` qualified(type($src)) `,` type($offset) `)`
    `outs` `:` qualified(type($dst) )
    attr-dict
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "GETVAL"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}


// ---- tile-world TOp version (with 't') ----
// pto.mscatter ins(%src, %idx) outs(%mem) [ ...]
def MScatterOp : PTO_TOp<"mscatter", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Scatter-store elements from a tile into memory using per-element indices (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$idx,
    PTODpsType:$mem               // outs target
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $idx `:` qualified(type($src)) `,` type($idx) `)`
    `outs` `(` $mem `:` type($mem) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "MSCATTER"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE3; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getMemMutable(); }
  }];
}

//===----------------------------------------------------------------------===//
// PTO_TOPs(Tilebuffer, DPS)
//===----------------------------------------------------------------------===//

def TAbsOp : PTO_TOp<"tabs", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise absolute value of a tile ";
  let description = [{
    For each element (i, j): dst[i,j] = |src[i,j]|.
  }];

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs); 

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)` `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_V; // Vector pipe for elementwise ops
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TAddOp : PTO_TOp<"tadd", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise add of two tiles ";
  let description = [{
    For each element (i, j): dst[i,j] = src0[i,j] + src1[i,j].
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_V;
    }

    ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getDstMutable();
    }
  }];
}

def TAddCOp : PTO_TOp<"taddc", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise ternary add of tiles ";
  let description = [{
    For each element (i, j): dst[i,j] = src0[i,j] + src1[i,j] + src2[i,j].
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$src2,
    PTODpsType:$dst
  );

  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `,` $src2 `:`
          qualified(type($src0)) `,` qualified(type($src1)) `,` qualified(type($src2)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TAddSOp : PTO_TOp<"tadds", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise add a scalar to a tile ";
  let description = [{
    For each element (i, j): dst[i,j] = src[i,j] + scalar.
    Tile operands are tilebuffers; scalar is a builtin scalar type (e.g. f32).
  }];

  let arguments = (ins
    PTODpsType:$src,
    ScalarType:$scalar,
    PTODpsType:$dst
  );

  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TAddSCOp : PTO_TOp<"taddsc", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "dst = src0 + scalar + src1 ";
  let arguments = (ins
    PTODpsType:$src0,
    ScalarType:$scalar,
    PTODpsType:$src1,
    PTODpsType:$dst
  );
  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $scalar `,` $src1 `:` qualified(type($src0)) `,` type($scalar) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}


def TAndOp : PTO_TOp<"tand", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TAND(dst, src0, src1) bitwise-and on tiles";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1, PTODpsType:$dst);
  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TConcatOp: PTO_TOp<"tconcat", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TCONCAT: Concatenate two tiles along the column dimension.";
  let description = [{
    Semantics (per row):
      dst[:, 0:validCol0] = src0
      dst[:, validCol0:validCol0+validCol1] = src1

    Constraints are implementation-defined; see verifier for current checks.
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TAndSOp : PTO_TOp<"tands", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TANDS(dst, src, scalar) bitwise-and tile with scalar";
  let arguments = (ins PTODpsType:$src, AnySignlessInteger:$scalar, PTODpsType:$dst);
  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}


def TCIOp : PTO_TOp<"tci", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Generate contiguous integer sequence into dst tile.";

  let arguments = (ins
    AnyInteger:$S,
    PTODpsType:$dst,
    DefaultValuedAttr<BoolAttr, "false">:$descending
  );
  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $S 
    attr-dict
    `:` type($S) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TTriOp : PTO_TOp<"ttri", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Generate a lower/upper triangular mask tile";

  let arguments = (ins
    AnyInteger:$diagonal,
    PTODpsType:$dst,
    DefaultValuedAttr<I32Attr, "0">:$upperOrLower
  );
  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $diagonal
    attr-dict
    `:` type($diagonal) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TCmpOp : PTO_TOp<"tcmp", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Compare two tiles and write a packed predicate mask";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst,
    OptionalAttr<PTO_CmpModeAttr>:$cmpMode
  );
  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1
            attr-dict
          `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`   
  }];

   let builders = [
    OpBuilder<(ins "TypeRange":$res, "Value":$src0, "Value":$src1, "Value":$dst), [{
      build($_builder, $_state, res, src0, src1, dst, /*cmpMode=*/nullptr); 
    }]>];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TCmpSOp : PTO_TOp<"tcmps", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Compare scalar value against a tile and write a packed predicate mask";

  let arguments = (ins
    PTODpsType:$src,
    ScalarType:$scalar,
    DefaultValuedAttr<PTO_CmpModeAttr, "::mlir::pto::CmpMode::EQ">:$cmpMode,
    PTODpsType:$dst
  );

  let results = (outs);
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar
          attr-dict
          `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    
  }];
}

def TColExpandOp : PTO_TOp<"tcolexpand", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Broadcast src(0, j) to all rows in column j ";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TColExpandAddOp : PTO_TOp<"tcolexpandadd", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Column-wise broadcast add: add a per-column scalar vector src1 to src0 ";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TColExpandMulOp : PTO_TOp<"tcolexpandmul", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Column-wise broadcast multiply: multiply each element of src0 by a per-column scalar vector src1 ";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TColExpandDivOp : PTO_TOp<"tcolexpanddiv", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Column-wise broadcast divide: divide each element of src0 by a per-column scalar vector src1 ";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TColExpandSubOp : PTO_TOp<"tcolexpandsub", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Column-wise broadcast subtract: subtract a per-column scalar vector src1 from src0 ";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TColExpandExpdifOp : PTO_TOp<"tcolexpandexpdif", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Column-wise broadcast expdif: compute exp(src0 - src1) using a per-column scalar vector src1 ";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TColExpandMaxOp : PTO_TOp<"tcolexpandmax", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Column-wise broadcast max: take max(src0, src1) using a per-column scalar vector src1 ";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TColExpandMinOp : PTO_TOp<"tcolexpandmin", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Column-wise broadcast min: take min(src0, src1) using a per-column scalar vector src1 ";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TColMaxOp : PTO_TOp<"tcolmax", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Reduce each column by taking the maximum across rows ";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TColMinOp : PTO_TOp<"tcolmin", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Reduce each column by taking the minimum across rows ";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TColSumOp : PTO_TOp<"tcolsum", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Reduce each column by summing across rows (tilebuf-based, explicit tmp)";

  let arguments = (ins
    PTODpsType:$src,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$isBinary
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return ::mlir::MutableOperandRange(getOperation(), 1, getOperation()->getNumOperands());
    }
  }];
}

def TColProdOp : PTO_TOp<"tcolprod", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Reduce each column by multiplying across rows";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TCvtOp : PTO_TOp<"tcvt", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise type conversion with rounding mode (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_RoundModeAttr, "::mlir::pto::RoundMode::CAST_RINT">:$rmode
  );

  let results = (outs);
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src 
    attr-dict
    `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
  }];
}

def TDivOp : PTO_TOp<"tdiv", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise division of two tiles (tilebuf, DPS)";
  let description = [{
    For each element (i, j): dst[i,j] = src0[i,j] / src1[i,j].
    Division-by-zero behavior is target-defined.
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

}

def TDivSOp : PTO_TOp<"tdivs", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise division with a scalar (tilebuf, DPS)";

  let arguments = (ins
    AnyType:$src,
    ScalarType:$scalar,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

}

def TFModOp : PTO_TOp<"tfmod", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise fmod/remainder of two tiles (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TFModSOp : PTO_TOp<"tfmods", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise fmod/remainder with a scalar (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    ScalarType:$scalar,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TExpOp : PTO_TOp<"texp", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise exponential (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TExpandsOp : PTO_TOp<"texpands", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Broadcast scalar into dst (tilebuf, DPS)";

  let arguments = (ins
    ScalarType:$scalar,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $scalar `:` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TExtractOp : PTO_TOp<"textract", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Extract sub-tile window from src into dst (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    Index:$indexRow,
    Index:$indexCol,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $indexRow `,` $indexCol `:` qualified(type($src)) `,` type($indexRow) `,` type($indexCol) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      // Align with pto-isa TEXTRACT op classes:
      //   - TEXTRACT_M2LR : MAT -> LEFT/RIGHT (and scaling-like targets) (MTE1)
      //   - TEXTRACT_V2M  : VEC -> MAT       (FIX)
      //   - TEXTRACT_A2M  : ACC -> MAT       (FIX)
      // For pure UB slices (VEC -> VEC), treat as vector pipe.
      auto getASFromType = [](Type ty)
          -> std::optional<::mlir::pto::AddressSpace> {
        if (auto tb = llvm::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = llvm::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace();
          return std::nullopt;
        }
        if (auto mr = llvm::dyn_cast<::mlir::MemRefType>(ty)) {
          if (auto ms = mr.getMemorySpace()) {
            if (auto as =
                    llvm::dyn_cast<::mlir::pto::AddressSpaceAttr>(ms))
              return as.getAddressSpace();
          }
          return std::nullopt;
        }
        return std::nullopt;
      };

      auto sOpt = getASFromType(getSrc().getType());
      auto dOpt = getASFromType(getDst().getType());
      if (!sOpt.has_value() || !dOpt.has_value())
        return ::mlir::pto::PIPE::PIPE_V;

      const auto s = sOpt.value();
      const auto d = dOpt.value();

      if (s == ::mlir::pto::AddressSpace::MAT &&
          (d == ::mlir::pto::AddressSpace::LEFT ||
           d == ::mlir::pto::AddressSpace::RIGHT ||
           d == ::mlir::pto::AddressSpace::BIAS ||
           d == ::mlir::pto::AddressSpace::SCALING)) {
        return ::mlir::pto::PIPE::PIPE_MTE1;
      }

      if ((s == ::mlir::pto::AddressSpace::VEC &&
           d == ::mlir::pto::AddressSpace::MAT) ||
          (s == ::mlir::pto::AddressSpace::ACC &&
           d == ::mlir::pto::AddressSpace::MAT)) {
        return ::mlir::pto::PIPE::PIPE_FIX;
      }

      if (s == ::mlir::pto::AddressSpace::VEC &&
          d == ::mlir::pto::AddressSpace::VEC) {
        return ::mlir::pto::PIPE::PIPE_V;
      }

      // Default to vector pipe for unmatched combinations.
      return ::mlir::pto::PIPE::PIPE_V;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TExtractFPOp : PTO_TOp<"textract_fp", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Extract acc tile window into dst using fp/scaling tile (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$fp,
    Index:$indexRow,
    Index:$indexCol,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $fp `,` $indexRow `,` $indexCol `:` qualified(type($src)) `,` qualified(type($fp)) `,` type($indexRow) `,` type($indexCol) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_FIX; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TInsertOp : PTO_TOp<"tinsert", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Insert src sub-tile into dst at (indexRow, indexCol) (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    Index:$indexRow,
    Index:$indexCol,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $indexRow `,` $indexCol `:` qualified(type($src)) `,` type($indexRow) `,` type($indexCol) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    // TINSERT runs on different pipes depending on target/path:
    // - A5 Vec(UB)->Vec(UB): PIPE_V  (ND_VEC path).
    // - A5 Vec(UB)->Mat(L1): PIPE_MTE3 (custom UB->L1 copy path).
    // - Acc->Mat (A2/A3/A5 regular path): PIPE_FIX.
    ::mlir::pto::PIPE getPipe() {
      auto isA5Target = [&]() -> bool {
        auto moduleOp = getOperation()->getParentOfType<::mlir::ModuleOp>();
        if (!moduleOp)
          return false;

        if (auto arch =
                moduleOp->getAttrOfType<::mlir::StringAttr>("pto.target_arch")) {
          if (arch.getValue().equals_insensitive("a5"))
            return true;
        }

        if (auto spec =
                moduleOp->getAttrOfType<::mlir::StringAttr>("pto.device-spec")) {
          auto s = spec.getValue();
          if (s.starts_with("Ascend950") || s.starts_with("Ascend910_95"))
            return true;
        }
        return false;
      };

      auto getASFromType = [](Type ty)
          -> std::optional<::mlir::pto::AddressSpace> {
        if (auto tb = ::mlir::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = ::mlir::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace();
          return std::nullopt;
        }
        if (auto mr = ::mlir::dyn_cast<::mlir::MemRefType>(ty)) {
          if (auto ms = mr.getMemorySpace()) {
            if (auto as = ::mlir::dyn_cast<::mlir::pto::AddressSpaceAttr>(ms))
              return as.getAddressSpace();
          }
          return std::nullopt;
        }
        return std::nullopt;
      };

      auto sOpt = getASFromType(getSrc().getType());
      auto dOpt = getASFromType(getDst().getType());
      if (isA5Target() && sOpt.has_value() && dOpt.has_value() &&
          sOpt.value() == ::mlir::pto::AddressSpace::VEC &&
          dOpt.value() == ::mlir::pto::AddressSpace::VEC)
        return ::mlir::pto::PIPE::PIPE_V;
      if (isA5Target() && sOpt.has_value() && dOpt.has_value() &&
          sOpt.value() == ::mlir::pto::AddressSpace::VEC &&
          dOpt.value() == ::mlir::pto::AddressSpace::MAT)
        return ::mlir::pto::PIPE::PIPE_MTE3;

      // Default and Acc->Mat path.
      return ::mlir::pto::PIPE::PIPE_FIX;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TInsertFPOp : PTO_TOp<"tinsert_fp", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Insert acc tile window into dst using fp/scaling tile (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$fp,
    Index:$indexRow,
    Index:$indexCol,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $fp `,` $indexRow `,` $indexCol `:` qualified(type($src)) `,` qualified(type($fp)) `,` type($indexRow) `,` type($indexCol) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_FIX; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TFillPadOp : PTO_TOp<"tfillpad", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Copy src into dst and fill padded elements using dst PadVal (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins`  `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TFillPadExpandOp : PTO_TOp<"tfillpad_expand", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Copy src into dst and fill padded elements using dst PadVal, allowing dst to be larger (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins`  `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGatherOp : PTO_TOp<"tgather", [
  AttrSizedOperandSegments,
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Gather/select elements using index+tmp, scalar-compare+tmp, or a mask pattern (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    Optional<PTODpsType>:$cdst,
    Optional<PTODpsType>:$indices,
    Optional<PTODpsType>:$tmp,
    Optional<ScalarType>:$kValue,
    OptionalAttr<PTO_MaskPatternAttr>:$maskPattern,
    OptionalAttr<PTO_CmpModeAttr>:$cmpMode,
    OptionalAttr<I32Attr>:$offset
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    bool hasCompareForm() { return static_cast<bool>(getCdst()) || static_cast<bool>(getKValue()); }
    bool hasIndexForm() { return static_cast<bool>(getIndices()); }
    bool hasMaskForm() { return static_cast<bool>(getMaskPatternAttr()); }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return ::mlir::MutableOperandRange(getOperation(), 1, 1 + (getCdst() ? 1 : 0));
    }
  }];
}

def TGatherBOp : PTO_TOp<"tgatherb", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Gather elements using byte offsets (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$offsets,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $offsets `:` qualified(type($src)) `,` type($offsets) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TLogOp : PTO_TOp<"tlog", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise natural logarithm (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TLReluOp : PTO_TOp<"tlrelu", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Leaky ReLU with a scalar slope (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    F32:$slope,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $slope `:` qualified(type($src)) `,` type($slope) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TMaxOp : PTO_TOp<"tmax", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise maximum of two tiles (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TMaxSOp : PTO_TOp<"tmaxs", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise max of a tile and a scalar (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    ScalarType:$scalar,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TMinOp : PTO_TOp<"tmin", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise minimum of two tiles (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TMinSOp : PTO_TOp<"tmins", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise minimum of a tile and a scalar (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    ScalarType:$scalar,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TMovFPOp : PTO_TOp<"tmov.fp", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TMOV_FP: move/convert using fp (scaling) tile (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$fp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $fp `:` qualified(type($src)) `,` qualified(type($fp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    // TMOV_FP is an ACC->MAT move (Cc->Cb) with vector quant parameters in
    // SCALING (fbuf). Treat it as a data-movement op for sync insertion.
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE1; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TMrgSortOp: PTO_TOp<"tmrgsort", [
  AttrSizedOperandSegments,
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TMRGSORT: Merge sort (format1: ins(src,blockLen) out(dst); format2: ins(2 to 4 srcs), outs(dst,tmp,executed)).";

  let arguments = (ins
    Variadic<PTODpsType>:$srcs,
    Optional<AnyInteger>:$blockLen,
    Variadic<PTODpsType>:$dsts,
    Optional<AnyType>:$excuted,
    DefaultValuedAttr<BoolAttr, "false">:$exhausted
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    bool isFormat1() { return getSrcs().size() == 1u && getBlockLen() && getDsts().size() == 1u; }
    bool isFormat2() { return getSrcs().size() >= 2u && getSrcs().size() <= 4u && getDsts().size() == 2u && getExcuted(); }
    Value getSrc() { return getSrcs().front(); }
    Value getDst() { return getDsts().front(); }
    Value getTmp() { return getDsts()[1]; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstsMutable(); }
    void print(::mlir::OpAsmPrinter &p);
    static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TMulOp: PTO_TOp<"tmul", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TMUL: Elementwise multiply of two tiles.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

//===----------------------------------------------------------------------===//
// PTOOps.td  (add TMULS DPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TMulSOp: PTO_TOp<"tmuls", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TMULS: Elementwise multiply a tile by a scalar.";

  let arguments = (ins
    PTODpsType:$src0,
    ScalarType:$scalar,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $scalar `:` qualified(type($src0)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TNEG TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TNegOp: PTO_TOp<"tneg", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TNEG: Elementwise negation of a tile.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TNOT TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TNotOp: PTO_TOp<"tnot", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TNOT: Elementwise bitwise NOT of a tile.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TOR TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TOrOp: PTO_TOp<"tor", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TOR: Elementwise bitwise OR of two tiles.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TORS TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TOrSOp: PTO_TOp<"tors", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TORS: Elementwise bitwise OR of a tile and a scalar.";

  let arguments = (ins
    PTODpsType:$src,
    AnySignlessInteger:$scalar,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TPARTADD TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TPartAddOp: PTO_TOp<"tpartadd", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TPARTADD: Partial elementwise add with implementation-defined handling of mismatched valid regions.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TPARTMAX TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TPartMaxOp: PTO_TOp<"tpartmax", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Partial elementwise max with implementation-defined handling of mismatched valid regions.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TPARTMIN TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TPartMinOp: PTO_TOp<"tpartmin", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Partial elementwise min with implementation-defined handling of mismatched valid regions.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TPARTMUL TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TPartMulOp: PTO_TOp<"tpartmul", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Partial elementwise mul with implementation-defined handling of mismatched valid regions.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TPRELU TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TPReluOp: PTO_TOp<"tprelu", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TPRELU: Elementwise PReLU (parametric ReLU) with a per-element slope tile.";

  let description = [{
    DPS form:
      tile buf:  pto.tprelu ins(%src0, %src1, %tmp) outs(%dst)
  }];
  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `,` $tmp `:` qualified(type($src0)) `,` qualified(type($src1)) `,` qualified(type($tmp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// TQuantOp
//===----------------------------------------------------------------------===//

def TQuantOp : PTO_TOp<"tquant", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TQUANT: Quantize a tile using a scaling (fp) tile.";

  let description = [{
    Converts src tile elements to a lower-precision format using the scaling
    parameters held in the fp tile, writing results into dst.

    For INT8_ASYM an additional offset tile must be supplied in ins.

    DPS form (INT8_SYM):
      pto.tquant ins(%src, %fp : <src_type>, <fp_type>)
                 outs(%dst : <dst_type>) {quant_type = #pto<quant_type INT8_SYM>}
    DPS form (INT8_ASYM):
      pto.tquant ins(%src, %fp, %offset : <src_type>, <fp_type>, <offset_type>)
                 outs(%dst : <dst_type>) {quant_type = #pto<quant_type INT8_ASYM>}
  }];

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$fp,
    Optional<PTODpsType>:$offset,
    PTODpsType:$dst,
    PTO_QuantTypeAttr:$quant_type
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// TDequantOp
//===----------------------------------------------------------------------===//

def TDequantOp : PTO_TOp<"tdequant", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TDEQUANT: Dequantize an integer tile using per-row scale and offset tiles.";

  let description = [{
    Converts an integer (int8 or int16) src tile to float32, applying per-row
    scale and offset parameters held in the scale and offset tiles.

    Formula: dst[i][j] = (float(src[i][j]) - offset[i][0]) * scale[i][0]

    The scale and offset tiles are row-vectors: one f32 value per row is
    broadcast across all columns of that row.

    DPS form:
      pto.tdequant ins(%src, %scale, %offset : <src_type>, <scale_type>, <offset_type>)
                   outs(%dst : <dst_type>)
  }];

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$scale,
    PTODpsType:$offset,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

//===----------------------------------------------------------------------===//
// PTOOps.td  (add TRECIP TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TRecipOp: PTO_TOp<"trecip", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TRECIP: Elementwise reciprocal of a tile.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TRELU TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TReluOp: PTO_TOp<"trelu", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TRELU: Elementwise ReLU of a tile.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TREM TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TRemOp: PTO_TOp<"trem", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TREM: Elementwise remainder of two tiles.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `,` $tmp `:` qualified(type($src0)) `,` qualified(type($src1)) `,` qualified(type($tmp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TREMS TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TRemSOp: PTO_TOp<"trems", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TREMS: Elementwise remainder with a scalar";

  let arguments = (ins
    PTODpsType:$src,
    ScalarType:$scalar,
    PTODpsType:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `,` $tmp `:` qualified(type($src)) `,` type($scalar) `,` qualified(type($tmp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (TRESHAPE: SSA view op; aliases src storage)
//===----------------------------------------------------------------------===//

def TReshapeOp: PTO_TOp<"treshape", [
  OpPipeInterface,
  Pure,
  ViewLikeOpInterface
]> {
  let summary = "TRESHAPE: Reinterpret a tile buffer view (SSA; aliases src storage)";

  let arguments = (ins
    PTODpsType:$src
  );

  let results = (outs
    PTODpsType:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = "$src attr-dict `:` qualified(type($src)) `->` qualified(type($result))";

  let extraClassDeclaration = [{
    // Keep consistent with pto-isa opPipeList: TRESHAPE is modeled on PIPE_S.
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_S; }
    ::mlir::Value getViewSource() { return getSrc(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWEXPAND TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TRowExpandOp: PTO_TOp<"trowexpand", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWEXPAND: Broadcast the first element of each source row across the destination row.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWEXPANDDIV TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TRowExpandDivOp: PTO_TOp<"trowexpanddiv", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWEXPANDDIV: Row-wise broadcast divide: divide each row of src0 by a per-row scalar vector src1.";
  let description = [{
    pto-isa has overloads with/without tmp; optional tmp for scratch.
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWEXPANDMUL TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TRowExpandMulOp: PTO_TOp<"trowexpandmul", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWEXPANDMUL: Row-wise broadcast multiply with per-row scalar vector.";
  let description = [{
    pto-isa has overloads with/without tmp; optional tmp for scratch.
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWEXPANDSUB TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TRowExpandSubOp: PTO_TOp<"trowexpandsub", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWEXPANDSUB: Row-wise broadcast subtract: subtract a per-row scalar vector src1 from each row of src0.";
  let description = [{
    pto-isa has overloads with/without tmp; optional tmp for scratch.
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TRowExpandAddOp: PTO_TOp<"trowexpandadd", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWEXPANDADD: Row-wise broadcast add: add a per-row scalar vector src1 to each row of src0.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TRowExpandExpdifOp: PTO_TOp<"trowexpandexpdif", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWEXPANDEXPDIF: Row-wise broadcast expdif with per-row scalar vector.";
  let description = [{
    pto-isa has overloads with/without tmp on A2/A3; A5 supports the 3-operand form only.
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TRowExpandMaxOp: PTO_TOp<"trowexpandmax", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWEXPANDMAX: Row-wise broadcast max with per-row scalar vector.";
  let description = [{
    pto-isa has overloads with/without tmp on A2/A3; A5 supports the 3-operand form only.
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TRowExpandMinOp: PTO_TOp<"trowexpandmin", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWEXPANDMIN: Row-wise broadcast min with per-row scalar vector.";
  let description = [{
    pto-isa has overloads with/without tmp on A2/A3; A5 supports the 3-operand form only.
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWMAX TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TRowMaxOp: PTO_TOp<"trowmax", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWMAX: Reduce each row by taking the maximum across columns.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $tmp `:` qualified(type($src)) `,` qualified(type($tmp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWMIN TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TRowMinOp: PTO_TOp<"trowmin", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWMIN: Reduce each row by taking the minimum across columns.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $tmp `:` qualified(type($src)) `,` qualified(type($tmp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWSUM TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TRowSumOp: PTO_TOp<"trowsum", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWSUM: Reduce each row by summing across columns.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $tmp `:` qualified(type($src)) `,` qualified(type($tmp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TRowProdOp: PTO_TOp<"trowprod", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWPROD: Reduce each row by multiplying across columns.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $tmp `:` qualified(type($src)) `,` qualified(type($tmp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TRSQRT TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TRsqrtOp: PTO_TOp<"trsqrt", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TRSQRT: Elementwise reciprocal square root.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    Optional<PTODpsType>:$tmp
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSCATTER TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TScatterOp: PTO_TOp<"tscatter", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSCATTER: Scatter elements of a source tile into a destination tile using per-element indices.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$indexes,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $indexes `:` qualified(type($src)) `,` qualified(type($indexes) ) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      // NOTE:
      // - On A2/A3, pto-isa TSCATTER is a scalar UB loop (PIPE_S).
      // - On A5, pto-isa TSCATTER is vector scatter (PIPE_V).
      // We accept either `pto.target_arch=a5` or A5 `pto.device-spec`.
      auto isA5Target = [&]() -> bool {
        auto moduleOp = getOperation()->getParentOfType<::mlir::ModuleOp>();
        if (!moduleOp)
          return false;

        if (auto arch =
                moduleOp->getAttrOfType<::mlir::StringAttr>("pto.target_arch")) {
          if (arch.getValue().equals_insensitive("a5"))
            return true;
        }

        if (auto spec =
                moduleOp->getAttrOfType<::mlir::StringAttr>("pto.device-spec")) {
          auto s = spec.getValue();
          if (s.starts_with("Ascend950") || s.starts_with("Ascend910_95"))
            return true;
        }

        return false;
      };

      return isA5Target() ? ::mlir::pto::PIPE::PIPE_V
                          : ::mlir::pto::PIPE::PIPE_S;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSEL TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TSelOp: PTO_TOp<"tsel", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSEL: Select between two tiles using a mask tile (per-element selection).";
  let description = [{
    DPS form (pto-isa TSEL requires tmp): ins(%mask, %src0, %src1, %tmp) outs(%dst)
  }];

  let arguments = (ins
    PTODpsType:$mask,
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $mask `,` $src0 `,` $src1 `,` $tmp `:` qualified(type($mask)) `,` qualified(type($src0)) `,` qualified(type($src1)) `,` qualified(type($tmp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSELS TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TSelSOp: PTO_TOp<"tsels", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSELS: Select one of two source tiles using a scalar selectMode (global select).";
  let description = [{
    DPS form (pto-isa TSELS requires tmp): ins(%mask, %src, %tmp, %scalar) outs(%dst)
  }];

  let arguments = (ins
    PTODpsType:$mask,
    PTODpsType:$src,
    PTODpsType:$tmp,
    ScalarType:$scalar,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $mask `,` $src `,` $tmp `,` $scalar `:` qualified(type($mask)) `,` qualified(type($src)) `,` qualified(type($tmp)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSHL TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TShlOp: PTO_TOp<"tshl", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSHL: Elementwise shift-left of two tiles.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSHR TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TShrOp: PTO_TOp<"tshr", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSHR: Elementwise shift-right of two tiles.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSHLS/TSHRS TBDPS: shift-left/right by scalar)
//===----------------------------------------------------------------------===//

def TShlSOp : PTO_TOp<"tshls", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSHLS: Elementwise shift-left of a tile by a scalar (shift count).";
  let arguments = (ins
    PTODpsType:$src,
    AnySignlessInteger:$scalar,
    PTODpsType:$dst
  );
  let results = (outs);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TShrSOp : PTO_TOp<"tshrs", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSHRS: Elementwise shift-right of a tile by a scalar (shift count).";
  let arguments = (ins
    PTODpsType:$src,
    AnySignlessInteger:$scalar,
    PTODpsType:$dst
  );
  let results = (outs);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSORT32 TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TSort32Op: PTO_TOp<"tsort32", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSORT32: Sort fixed-size 32-element blocks using explicit input indices.";
  let description = [{
    DPS form matching pto-isa overloads: (dst, src, idx) or (dst, src, idx, tmp).
    In PTO IR, `idx` is modeled as an input operand in `ins(...)`, while `dst`
    remains the sole DPS init/output buffer in `outs(...)`.
  }];

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$idx,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSQRT TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TSqrtOp: PTO_TOp<"tsqrt", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,  
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSQRT: Elementwise square root.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSTORE_FP TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TStoreFPOp: PTO_TOp<"tstore_fp", [
  PTO_DpsInitOpInterface,
]> {
  let summary = "TSTORE_FP: Store an accumulator tile into global memory using a scaling (fp) tile for vector quantization parameters.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$fp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $fp `:` qualified(type($src)) `,` qualified(type($fp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSUB TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TSubOp: PTO_TOp<"tsub", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSUB: Elementwise subtract of two tiles.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSUBC TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TSubCOp: PTO_TOp<"tsubc", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSUBC: Elementwise ternary op: src0 - src1 + src2.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$src2,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `,` $src2 `:`
            qualified(type($src0)) `,` qualified(type($src1)) `,` qualified(type($src2)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];


}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSUBS TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TSubSOp: PTO_TOp<"tsubs", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSUBS: Elementwise subtract a scalar from a tile.";

  let arguments = (ins
    PTODpsType:$src,
    ScalarType:$scalar,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];


}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSUBSC TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TSubSCOp: PTO_TOp<"tsubsc", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSUBSC: Elementwise fused op: src0 - scalar + src1.";

  let arguments = (ins
    PTODpsType:$src0,
    ScalarType:$scalar,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $scalar `,` $src1 `:`
            qualified(type($src0)) `,` type($scalar) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

}

//===----------------------------------------------------------------------===//
// PTOOps.td  (add TXORS TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TXorSOp: PTO_TOp<"txors", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TXORS: Elementwise bitwise XOR of a tile and a scalar.";
  let description = [{
    DPS form (pto-isa TXORS requires tmp): ins(%src, %scalar, %tmp) outs(%dst)
  }];

  let arguments = (ins
    PTODpsType:$src,
    AnySignlessInteger:$scalar,
    PTODpsType:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `,` $tmp `:` qualified(type($src)) `,` type($scalar) `,` qualified(type($tmp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TXOR TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TXorOp: PTO_TOp<"txor", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TXOR: Elementwise bitwise XOR of two tiles.";
  let description = [{
    DPS form:
      without tmp: pto.txor ins(%src0, %src1) outs(%dst)
      with tmp:    pto.txor ins(%src0, %src1, %tmp) outs(%dst)
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1  `,` $tmp `:` qualified(type($src0)) `,` qualified(type($src1)) `,` qualified(type($tmp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

}

def TPrintOp: PTO_TOp<"tprint", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TPRINT: Print the contents of a Tile or GlobalTensor for debugging purposes directly from device code..";

  let arguments = (ins
    PTODpsType:$src
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
 }

#endif // MLIR_DIALECT_PTO_IR_PTOOPS
