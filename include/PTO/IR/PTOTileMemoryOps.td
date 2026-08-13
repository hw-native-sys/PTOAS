// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOTileMemoryOps - semantic operation family -*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTOTILEMEMORYOPS
#define MLIR_DIALECT_PTO_IR_PTOTILEMEMORYOPS

include "PTO/IR/PTOOpCommon.td"

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

//===----------------------------------------------------------------------===//
// Multi-buffer tile allocation and slot selection
//===----------------------------------------------------------------------===//
//
// `pto.alloc_multi_tile` declares an N-slot logical tile buffer. The actual
// physical allocation of N slots is performed directly by PTOPlanMemory.
// The op remains tile-native until PTOResolveBufferSelect materializes an
// addressed `pto.alloc_tile` for each selected slot.
//
// `pto.multi_tile_get` is the *only* way to consume a `multi_tile_buf`: it
// returns a regular `tile_buf` view onto the chosen slot. The slot index
// may be a constant or any SSA `index` value; PTOAS does not synthesize
// `iv mod N` automatically -- the user expression IS the slot selector.
//

def AllocMultiTileOp : PTO_Op<"alloc_multi_tile", [AttrSizedOperandSegments]> {
  let summary = "Allocate an N-slot multi-buffer tile";
  let description = [{
    Produces a `!pto.multi_tile_buf<S, count=N>` value owning N physical
    slots of the inner tile shape `S`.

    In the default pipeline (`--pto-level=level1|level2`) no `addr` operand
    is given and the physical addresses of the N slots are decided by
    PTOPlanMemory.

    Under `--pto-level=level3` the local memory is owned by the upper-level
    caller and PTOPlanMemory does not run, so an explicit base address
    operand `$addr` is required (mirroring `pto.alloc_tile`). The N slots
    are laid out contiguously as the region `[addr, addr + N * slotBytes)`
    with slot `k` at `addr + k * slotBytes`, where `slotBytes` is the byte
    footprint of one slot tile (`product(shape) * element_size`). The caller
    is responsible for having reserved `N * slotBytes` bytes at `addr`.

    `valid_row` / `valid_col` operands follow the same semantics as
    `pto.alloc_tile`: required when the inner tile's valid shape is `?`,
    absent when both valid dims are static.
  }];

  let arguments = (ins
    Optional<I64>:$addr,
    Optional<Index>:$valid_row,
    Optional<Index>:$valid_col
  );

  let results = (outs MultiTileBufType:$result);

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

def MultiTileGetOp : PTO_Op<"multi_tile_get", [
    Pure,
    ViewLikeOpInterface
  ]> {
  let summary = "Pick one physical slot of a multi_tile_buf";
  let description = [{
    Returns a single-slot view of a `multi_tile_buf`. The result `tile_buf`
    must equal the source's per-slot type; the only thing this op selects
    is which of the N physical slots is referred to.

    The slot index is an `index` value in `[0, count)`. If it is a constant,
    the verifier checks the range. If it is a runtime SSA value, downstream
    sync analysis treats this use as a slot-indexed dynamic access; ptoas
    does NOT rewrite the user expression into `iv mod N`.

    This op is metadata-only (no data movement). PlanMemory preserves it,
    sync analysis consumes its slot index directly, and
    PTOResolveBufferSelect replaces it with an addressed `pto.alloc_tile`.
  }];

  let arguments = (ins
    MultiTileBufType:$source,
    Index:$slot
  );

  let results = (outs TileBufType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $slot `]` attr-dict
    `:` qualified(type($source)) `->` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSource(); }
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
    TileBufType:$source,
    Index:$valid_row,
    Index:$valid_col
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $valid_row `,` $valid_col attr-dict `:`
    qualified(type($source))
  }];
}

def GetValidShapeOp : PTO_Op<"get_validshape", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Read runtime valid row/col metadata from a tile";
  let description = [{
    Reads the current runtime valid_row/valid_col metadata from a tile_buf
    handle. This op observes prior set_validshape updates and returns the
    current values as index-typed SSA values.
  }];

  let arguments = (ins
    TileBufType:$source
  );

  let results = (outs
    Index:$valid_row,
    Index:$valid_col
  );

  let hasVerifier = 1;

  let assemblyFormat = "$source attr-dict `:` qualified(type($source))";
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

def TPrefetchAsyncOp : PTO_Op<"tprefetch_async"> {
  let summary = "Asynchronously prefetch global data into L2 cache via SDMA CMO.";
  let description = [{
    Starts one asynchronous prefetch operation.
  }];

  let arguments = (ins
    PTODpsType:$src,
    PrefetchAsyncContextType:$ctx
  );

  let results = (outs
    AsyncEventType:$event
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $src `,` $ctx `:`
        qualified(type($src)) `,` qualified(type($ctx)) `)`
    attr-dict `->` qualified(type($event))
  }];
}

def TStoreOp: PTO_TOp<"tstore", [
  AttrSizedOperandSegments,
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
    Optional<PTODpsType>:$fp,
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
           (`fp` $fp^ `:` qualified(type($fp)))?
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
        return false;
      };
      if (isAcc(getSrc().getType()))
        return ::mlir::pto::PIPE::PIPE_FIX;
      return ::mlir::pto::PIPE::PIPE_MTE3;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

#endif // MLIR_DIALECT_PTO_IR_PTOTILEMEMORYOPS
