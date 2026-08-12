// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOBaseOps.td - PTO base and infrastructure operations -*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// This file owns the PTO operations in this semantic family.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTOBASEOPS
#define MLIR_DIALECT_PTO_IR_PTOBASEOPS

include "PTO/IR/PTOOpCommon.td"

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

def PtrToIntOp : PTO_Op<"ptrtoint", [Pure]> {
  let summary = "Convert a global pointer to a byte address";
  let description = [{
    Converts a global pointer value to an i64 byte address. Pointer arithmetic
    such as `pto.addptr` remains explicit until backend lowering.
  }];

  let arguments = (ins
    PtrType:$ptr
  );

  let results = (outs
    I64:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $ptr attr-dict `:` type($ptr) `->` type($result)
  }];
}

def IntToPtrOp : PTO_Op<"inttoptr", [Pure]> {
  let summary = "Convert an i64 byte address to a global pointer";
  let description = [{
    Converts an i64 byte address to a global pointer of the requested element
    type. This is an escape hatch for explicit byte-address arithmetic and
    cross-element-type pointer reinterpretation. To keep byte-address-derived
    pointers from escaping into general pointer IR, the result may only be used
    directly as the pointer operand of `pto.load_scalar` or `pto.store_scalar`.
  }];

  let arguments = (ins
    AnyInteger:$addr
  );

  let results = (outs
    PtrType:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $addr attr-dict `:` type($addr) `->` type($result)
  }];
}

def CastPtrOp : PTO_Op<"castptr", [Pure]> {
  let summary = "Cast between integer and !pto.ptr, or between !pto.ptr types";
  let description = [{
    Performs an explicit pointer-domain cast.

    Supported cases:
    - integer -> !pto.ptr<elem, space>
    - !pto.ptr<elem, space> -> integer
    - !pto.ptr<srcElem, space> -> !pto.ptr<dstElem, space>
    - memref<..., space> -> !pto.ptr<elem, space> (extract the aligned base ptr)

    Pointer-to-pointer casts must stay within the same PTO memory space. Cross
    space casts such as gm <-> ub are rejected by the verifier.
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Scalar pointer load/store
//===----------------------------------------------------------------------===//

def LoadScalarOp : PTO_Op<"load_scalar", [
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Load a single scalar element from a pointer at offset.";

  let arguments = (ins
    PtrType:$ptr,
    Index:$offset
  );

  let results = (outs AnyType:$value);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $ptr `[` $offset `]` attr-dict `:` type($ptr) `->` type($value)
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_S; }
  }];
}

def StoreScalarOp : PTO_Op<"store_scalar", [
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Store a single scalar element to a pointer at offset.";

  let arguments = (ins
    PtrType:$ptr,
    Index:$offset,
    AnyType:$value
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

def MakeTensorViewOp : PTO_Op<"make_tensor_view", [AttrSizedOperandSegments,
                                                    Pure]> {
  let summary = "Wrap a pointer as a tensor_view descriptor (no allocation, no copy).";

  let arguments = (ins
    PtrType:$ptr,
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

def PartitionViewOp : PTO_Op<"partition_view", [AttrSizedOperandSegments,
                                                Pure]> {
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
  let hasCustomAssemblyFormat = 1;
}

// Get the size of a dimension of a tensor_view.
// Result type: Index (use arith.index_cast if i32 is needed).

def GetTensorViewDimOp : PTO_Op<"get_tensor_view_dim", [Pure]> {
  let summary = "Get the size of a dimension of a tensor_view.";
  let description = [{
    Returns the size of the given dimension of a logical tensor view.
    This op accepts !pto.tensor_view or !pto.partition_tensor_view.
    IR:
      %dim_size = pto.get_tensor_view_dim %tv, %dim_index
        : !pto.tensor_view, index -> index
  }];
  let arguments = (ins
    TensorViewLike:$tensor_view,
    Index:$dim_index
  );
  let results = (outs Index:$result);
  let assemblyFormat = [{
    $tensor_view `,` $dim_index `:` qualified(type($tensor_view)) `->` qualified(type($result))
    attr-dict
  }];
}

// Get the logical stride of a tensor_view dimension in elements.
// Result type: Index (use arith.index_cast if i32 is needed).

def GetTensorViewStrideOp : PTO_Op<"get_tensor_view_stride", [Pure]> {
  let summary = "Get the stride of a dimension of a tensor_view.";
  let description = [{
    Returns the stride, measured in elements, of the given dimension of a
    logical tensor view. This op accepts !pto.tensor_view or
    !pto.partition_tensor_view.
  }];
  let arguments = (ins
    TensorViewLike:$tensor_view,
    Index:$dim_index
  );
  let results = (outs Index:$result);
  let assemblyFormat = [{
    $tensor_view `,` $dim_index `:` qualified(type($tensor_view)) `->` qualified(type($result))
    attr-dict
  }];
}

def SubViewOp : PTO_Op<"subview", [
    Pure,
    ViewLikeOpInterface,
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<InferTypeOpInterface> // 启用 C++ 推导
  ]> {

  let summary = "Create a tile subview from a parent tile (logical size + valid dims).";
  let description = [{
    Creates a view into the source tile.
    - Logical subview window: defined by `offsets` + static `sizes`.
    - Result tile type shape: equals subview `sizes` (logical shape).
    - Effective subview extent: represented by inferred `valid_row/valid_col`
      (i.e. `valid_shape`), clipped by subview `sizes`.
    - Lowering keeps parent physical stride/base semantics for non-compact
      access, while IR type exposes logical subview size for readability.
  }];

  let arguments = (ins
    TileBufType:$source,
    Variadic<Index>:$offsets, // 运行时动态偏移 [i, j]
    I64ArrayAttr:$sizes,      // 静态形状 [32, 32]
    Optional<Index>:$valid_row,
    Optional<Index>:$valid_col
  );

  let results = (outs TileBufType:$result);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  // [新增] 显式实现 ViewLikeOpInterface 缺失的方法
  let extraClassDeclaration = [{
    // 接口要求 getViewSource，我们转发给自动生成的 getSource
    ::mlir::Value getViewSource() { return getSource(); }

    // ViewLikeOpInterface 可能还需要 getOffsets (如果 Variadic 不自动匹配)
    // 但通常 Variadic<Index>:$offsets 会生成 getOffsets()，这应该没问题。
    // 如果后续报 getOffsets 错，也可以在这里加。

    // Override the InferTypeOpInterface compatibility check so a declared empty
    // valid region (valid_row/valid_col = 0) is accepted in place of the size-
    // inferred valid extent on the operand-omitted path (PyPTO dual-AIV no-op
    // replay). inferReturnTypes' default impl dispatches to this static method.
    static bool isCompatibleReturnTypes(::mlir::TypeRange l, ::mlir::TypeRange r);
  }];
}

def MakePrefetchAsyncContextOp : PTO_Op<"make_prefetch_async_context"> {
  let summary = "Build a PrefetchAsyncContext from GM workspace.";
  let description = [{
    Materializes a first-class PTO IR `PrefetchAsyncContext` value from a GM
    workspace pointer.
  }];

  let arguments = (ins
    PtrType:$workspace
  );

  let results = (outs
    PrefetchAsyncContextType:$ctx
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $workspace `:` qualified(type($workspace)) `)`
    attr-dict `->` qualified(type($ctx))
  }];
}

def GetPrefetchAsyncSessionOp : PTO_Op<"get_prefetch_async_session", [Pure]> {
  let summary = "Project session from a PrefetchAsyncContext.";

  let arguments = (ins
    PrefetchAsyncContextType:$ctx
  );

  let results = (outs
    AsyncSessionType:$session
  );

  let assemblyFormat = [{
    $ctx attr-dict `:` qualified(type($ctx)) `->` qualified(type($session))
  }];
}

def GetBlockIdxOp : PTO_Op<"get_block_idx", [
    Pure,
    MicroOpInterface,
    SimtOpInterface
  ]> {
  let summary = "Get the current block index (core ID).";
  let description = [{
    Returns the linear index of the current compute unit (CUBE Core) within the task.
    The return value is in the range [0, BlockNum - 1].
  }];

  let arguments = (ins);
  let results = (outs I64:$result);

  let assemblyFormat = "attr-dict";
}

def GetSubBlockIdxOp : PTO_Op<"get_subblock_idx", [
    Pure,
    MicroOpInterface,
    SimtOpInterface
  ]> {
  let summary = "Get the current vector core ID.";
  let description = [{
    Returns the ID of the current compute unit (Vector Core).
    The return value is in the range [0, 1].
  }];

  let arguments = (ins);
  let results = (outs I64:$result);

  let assemblyFormat = "attr-dict";
}

def GetBlockNumOp : PTO_Op<"get_block_num", [
    Pure,
    MicroOpInterface,
    SimtOpInterface
  ]> {
  let summary = "Get the total number of blocks (cores).";
  let description = [{
    Returns the total number of compute units (Blocks) configured for the current task.
  }];

  let arguments = (ins);
  let results = (outs I64:$result);

  let assemblyFormat = "attr-dict";
}

def GetSubBlockNumOp : PTO_Op<"get_subblock_num", [
    Pure,
    MicroOpInterface,
    SimtOpInterface
  ]> {
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

def SectionCubeOp : PTO_SectionOp<"section.cube">;

def SectionVectorOp : PTO_SectionOp<"section.vector">;

def SectionSimtOp : PTO_Op<"section.simt", [SingleBlock, NoTerminator]> {
  let summary = "Container for inline SIMT code";
  let description = [{
    `pto.section.simt` keeps an inline SIMT body in the enclosing kernel
    function until downstream PTOAS lowering consumes it.  It is a
    source-level carrier for transformations that need to observe the lexical
    relationship between surrounding code and SIMT regions.

    Launch dimensions are explicit static i32 attributes.  The textual launch
    dimension order matches `pto.simt_launch`:
    `<<<dimX, dimY, dimZ>>>`.
  }];
  let arguments = (ins I32Attr:$dimX, I32Attr:$dimY, I32Attr:$dimZ);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    `<` `<` `<` $dimX `,` $dimY `,` $dimZ `>` `>` `>` $body attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Tile Fusion Region Ops
//===----------------------------------------------------------------------===//

def FusionRegionOp : PTO_Op<"fusion_region", [SingleBlock]> {
  let summary = "Structured region container for one scheduled tile fusion group";
  let description = [{
    `pto.fusion_region` wraps one already-scheduled fusion group inside the
    current function and exposes explicit escaping values as results.
    The body is a lightweight single-block container and may implicitly capture
    surrounding SSA values from the parent scope.
  }];

  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $body attr-dict `:` type(results)
  }];
}

def YieldOp : PTO_Op<"yield", [Terminator, ParentOneOf<["FusionRegionOp"]>]> {
  let summary = "Terminate a pto.fusion_region and return its visible outputs";
  let description = [{
    `pto.yield` is the dedicated terminator for `pto.fusion_region`. Its
    operands enumerate the tile/value set that remains externally visible after
    encapsulation. They define the region outputs in the same order as the
    parent `pto.fusion_region` results, giving downstream passes a compact
    summary of which internal tiles still escape the region boundary.
  }];

  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` $values `)` attr-dict `:` functional-type($values, results)
  }];
}

//===----------------------------------------------------------------------===//
// Frontend TPUSH/TPOP Pipe Communication Ops
//===----------------------------------------------------------------------===//

def AicInitializePipeOp : PTO_Op<"aic_initialize_pipe",
                                 [AttrSizedOperandSegments]> {
  let summary = "Frontend Cube-kernel pipe initialization";

  let arguments = (ins
      DefaultValuedOptionalAttr<I32Attr, "0">:$id,
      I8Attr:$dir_mask,
      I32Attr:$slot_size,
      OptionalAttr<I32Attr>:$slot_num,
      OptionalAttr<I32Attr>:$local_slot_num,
      OptionalAttr<BoolAttr>:$nosplit,
      OptionalAttr<PTO_AccPushEpilogueAttr>:$acc_push_epilogue,
      Optional<PtrType>:$gm_slot_buffer,
      Optional<TensorViewType>:$gm_slot_tensor,
      Optional<I32>:$c2v_consumer_buf,
      Optional<I32>:$v2c_consumer_buf
  );

  let results = (outs);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def AivInitializePipeOp : PTO_Op<"aiv_initialize_pipe",
                                 [AttrSizedOperandSegments]> {
  let summary = "Frontend Vector-kernel pipe initialization";

  let arguments = (ins
      DefaultValuedOptionalAttr<I32Attr, "0">:$id,
      I8Attr:$dir_mask,
      I32Attr:$slot_size,
      OptionalAttr<I32Attr>:$slot_num,
      OptionalAttr<I32Attr>:$local_slot_num,
      OptionalAttr<BoolAttr>:$nosplit,
      OptionalAttr<PTO_AccPushEpilogueAttr>:$acc_push_epilogue,
      Optional<PtrType>:$gm_slot_buffer,
      Optional<TensorViewType>:$gm_slot_tensor,
      Optional<I32>:$c2v_consumer_buf,
      Optional<I32>:$v2c_consumer_buf
  );

  let results = (outs);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def BuildAsyncSessionOp : PTO_Op<"comm.build_async_session", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Build an async DMA session handle for TPUT_ASYNC/TGET_ASYNC";

  let arguments = (ins
      TileBufType:$scratch,
      PtrType:$workspace,
      OptionalAttr<I32Attr>:$sync_id,
      OptionalAttr<I64Attr>:$block_bytes,
      OptionalAttr<I64Attr>:$comm_block_offset,
      OptionalAttr<I32Attr>:$queue_num,
      OptionalAttr<I64Attr>:$channel_group_idx
  );

  let results = (outs AsyncSessionType:$session);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $scratch `,` $workspace `:` qualified(type($scratch)) `,` qualified(type($workspace)) `)`
    attr-dict `->` qualified(type($session))
  }];
}

def WaitAsyncEventOp : PTO_Op<"comm.wait_async_event", [
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

def TestAsyncEventOp : PTO_Op<"comm.test_async_event", [
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

def TNotifyOp : PTO_Op<"comm.tnotify", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Send a signal notification to remote GM";
  let arguments = (ins
      PTOCommType:$signal,
      AnySignlessInteger:$value,
      PTO_NotifyOpAttr:$notifyOp
  );
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` $signal `,` $value `:` type($signal) `,` type($value) `)`
    attr-dict
  }];
}

def TWaitOp : PTO_Op<"comm.twait", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Block until signal(s) satisfy a comparison";
  let arguments = (ins
      PTOCommType:$signal,
      AnySignlessInteger:$cmpValue,
      PTO_WaitCmpAttr:$cmp
  );
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` $signal `,` $cmpValue `:` type($signal) `,` type($cmpValue) `)`
    attr-dict
  }];
}

def TTestOp : PTO_Op<"comm.ttest", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Non-blocking signal comparison test";
  let arguments = (ins
      PTOCommType:$signal,
      AnySignlessInteger:$cmpValue,
      PTO_WaitCmpAttr:$cmp
  );
  let results = (outs I1:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` $signal `,` $cmpValue `:` type($signal) `,` type($cmpValue) `)`
    attr-dict `->` type($result)
  }];
}

def InitializeL2G2LPipeOp : PTO_Op<"initialize_l2g2l_pipe", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Initialize a local-to-global-to-local pipe handle";

  let arguments = (ins
      I8Attr:$dir_mask,
      I32Attr:$slot_size,
      I32Attr:$slot_num,
      OptionalAttr<I32Attr>:$local_slot_num,
      OptionalAttr<I32Attr>:$flag_base,
      OptionalAttr<BoolAttr>:$nosplit,
      OptionalAttr<PTO_AccPushEpilogueAttr>:$acc_push_epilogue,
      AnyType:$gm_addr,
      Optional<AnyType>:$local_addr,
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
        (`,` `nosplit` `=` $nosplit^)?
        `}`
    `(` $gm_addr `:` type($gm_addr)
        (`,` $local_addr^ `:` type($local_addr))?
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
      OptionalAttr<BoolAttr>:$nosplit,
      OptionalAttr<PTO_AccPushEpilogueAttr>:$acc_push_epilogue,
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
        (`,` `nosplit` `=` $nosplit^)?
        `}`
    `(` $local_addr `:` type($local_addr)
        (`,` $peer_local_addr^ `:` type($peer_local_addr))? `)`
    attr-dict `->` qualified(type($pipe))
  }];
}

def DeclareGlobalOp : PTO_Op<"declare_global"> {
  let summary = "Declare a GlobalTensor descriptor whose address is assigned later";

  let results = (outs TensorViewType:$entry);

  let assemblyFormat = [{
    attr-dict `->` qualified(type($entry))
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

//===----------------------------------------------------------------------===//
// Stack-local array (minimum set)
//===----------------------------------------------------------------------===//
//
// Mirrors the `eventid_array` triad (declare / get / set) for a generic
// statically-shaped scalar stack array. Disjoint from the tile-buf / scratch
// memory world: the array's address is decided by the host C++ compiler at
// codegen time, never by PTO memory planning.

def DeclareLocalArrayOp : PTO_Op<"declare_local_array"> {
  let summary = "Declare a stack-local statically-shaped array";
  let description = [{
    Lowers to a plain C++ stack variable like `T a[d1][d2]...;`. The element
    type is restricted to scalars (integer / float); no nested aggregates.

    This op does not participate in PTO memory planning. The resulting array's
    address is determined by the host C++ compiler.
  }];

  let results = (outs LocalArrayType:$array);

  let assemblyFormat = [{
    attr-dict `->` qualified(type($array))
  }];
}

def LocalArrayGetOp : PTO_Op<"local_array_get"> {
  let summary = "Read one element from a stack-local array";
  let description = [{
    Lowers to a scalar snapshot initialized from `a[i0][i1]...[iN-1]` in the
    emitted C++. This preserves SSA value semantics if the array is mutated
    later. The number of indices must equal the array's rank.
  }];

  let arguments = (ins
    LocalArrayType:$array,
    Variadic<Index>:$indices
  );
  let results = (outs ScalarType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $array `[` $indices `]` attr-dict
      `:` qualified(type($array)) `->` type($result)
  }];
}

def LocalArraySetOp : PTO_Op<"local_array_set"> {
  let summary = "Write one element into a stack-local array";
  let description = [{
    Lowers to `a[i0][i1]...[iN-1] = v;` in the emitted C++. The number of
    indices must equal the array's rank.
  }];

  let arguments = (ins
    LocalArrayType:$array,
    Variadic<Index>:$indices,
    ScalarType:$value
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $array `[` $indices `]` `,` $value attr-dict
      `:` qualified(type($array)) `,` type($value)
  }];
}

//===----------------------------------------------------------------------===//
// Stack-local heterogeneous aggregate (struct)
//===----------------------------------------------------------------------===//
//
// Mirrors the `local_array` triad (declare / get / set) for a generic
// heterogeneous scalar-unit aggregate. The struct's address is decided by the
// host C++ compiler at codegen time, never by PTO memory planning or
// `pto.pointer_cast`. Field positions are constant (struct semantics), so
// `get` / `set` carry a constant `path` of field indices that may descend
// through nested structs. The path always ends at a scalar; a nested struct is
// traversed by a longer path, never handed back as a value.

def DeclareStructOp : PTO_Op<"declare_struct"> {
  let summary = "Declare a stack-local heterogeneous aggregate";
  let description = [{
    Lowers to a plain C++ stack variable like `PtoStruct_X s;`. Field types are
    restricted to scalars (integers of width 8/16/32/64, f16/bf16/f32/f64) or a
    nested `!pto.struct`.

    This op does not participate in PTO memory planning. The resulting struct's
    address is determined by the host C++ compiler.

    The struct is stack storage owned by the scope that declares it, and a
    `!pto.struct` value lowers to a pointer to that storage. Handing the value
    to a terminator would publish the address outside the owning scope, so it is
    rejected: returning it would leave the caller holding a pointer to a dead
    frame. Module-level provenance validation also rejects function results and
    derived operation results of this type, so aliases cannot launder that
    address past the direct-use check. Struct storage originates from this op
    or is received through a function argument.
    Since `pto.struct_set` mutates in place, a struct never needs to be returned
    or yielded — declare it in the outer scope and pass it down as a function
    argument instead.
  }];

  let results = (outs StructType:$s);

  let hasVerifier = 1;

  let assemblyFormat = [{
    attr-dict `->` qualified(type($s))
  }];
}

def StructGetOp : PTO_Op<"struct_get"> {
  let summary = "Read a field from a stack-local struct by constant path";
  let description = [{
    Reads the field reached by following the constant `path` of field indices
    from the struct, descending through nested structs. Lowers to `s.fA.fB...`
    member access, read into a fresh variable so the SSA value keeps its value
    if the struct is mutated later (mirrors `pto.local_array_get`).

    The path must end at a scalar field, and the result type must equal that
    field's type. A path stopping on a nested `!pto.struct` is rejected —
    extend it to reach a scalar inside.
  }];

  let arguments = (ins
    StructType:$s,
    DenseI64ArrayAttr:$path
  );
  let results = (outs AnyType:$value);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $s custom<StructPath>($path) attr-dict
      `:` qualified(type($s)) `->` type($value)
  }];
}

def StructSetOp : PTO_Op<"struct_set"> {
  let summary = "Write a field into a stack-local struct by constant path";
  let description = [{
    Writes `value` into the field reached by following the constant `path` of
    field indices, descending through nested structs. Lowers to
    `s.fA.fB... = value;`.

    The path must end at a scalar field, and the value type must equal that
    field's type. A path stopping on a nested `!pto.struct` is rejected —
    extend it to reach a scalar inside.
  }];

  let arguments = (ins
    StructType:$s,
    DenseI64ArrayAttr:$path,
    AnyType:$value
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $s custom<StructPath>($path) `,` $value attr-dict
      `:` qualified(type($s)) `,` type($value)
  }];
}

def SetFlagOp : PTO_Op<"set_flag"> {
  let summary = "Set synchronization flag between pipes";
  let arguments = (ins
    PTO_PipeAttr:$src_pipe,
    PTO_PipeAttr:$dst_pipe,
    PTO_EventAttr:$event_id
  );
  let results = (outs);
  let extraClassDeclaration = [{
    void print(::mlir::OpAsmPrinter &p);
    static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser,
                                     ::mlir::OperationState &result);
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
  let extraClassDeclaration = [{
    void print(::mlir::OpAsmPrinter &p);
    static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser,
                                     ::mlir::OperationState &result);
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

def CmoCacheInvalidOp : PTO_Op<"cmo.cacheinvalid"> {
  let summary = "Perform cache maintenance for a memory space";
  let description = [{
    Cache maintenance operation for the specified memory space.

    The whole-cache form lowers to PTO-ISA's two-argument
    `dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE)`.

    The single-cache-line form carries the GM payload address selected by the
    producer DSL/user and lowers to
    `dcci(addr, cache_line_t::SINGLE_CACHE_LINE)`.

    This op is an explicit CMO boundary. PTOAS no longer runs the historical
    MemoryConsistency analysis pass that consumed marker-only CMO ops or inferred
    signal/payload pipe drains, so precise placement and coverage are the
    producer DSL/user's responsibility.
  }];

  let arguments = (ins
    Optional<AnyType>:$addr,
    PTO_AddressSpaceAttr:$space
  );

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def FenceBarrierAllOp : PTO_Op<"fence.barrier_all"> {
  let summary = "All-direction memory fence for a visibility scope";
  let description = [{
    Memory fence for a visibility scope. Around signal/payload communication it
    can be used as the producer-side publish barrier or the consumer-side
    acquire barrier depending on placement. In EmitC, `scope = gm` and
    `scope = all` drain all vector-kernel pipes with `PIPE_ALL` before lowering
    to `dsb(DSB_DDR)`. Other kernel kinds conservatively drain MTE2, MTE3, and
    FIX before the DDR barrier. Users therefore do not need to separately model
    producer pipe classes before a GM visibility fence.
  }];

  let arguments = (ins PTO_FenceScopeAttr:$scope);
  let assemblyFormat = "$scope attr-dict";
}

def SyncAllOp : PTO_Op<"syncall", [AttrSizedOperandSegments]> {
  let summary = "Cross-core all-participant synchronization.";
  let description = [{
    Models `pto::SYNCALL`.

    Supported IR forms:
    - hard sync: no operands, `mode = hard`
    - soft sync: `gm_workspace`, optional `used_cores`

    The soft operand ABI is identical for AIV-only, AIC-only, and mixed
    synchronization. The GM workspace must reserve at least 16 `i32` elements
    (64 bytes), be zero-initialized before first use, and occupy an exclusive
    cache line.
  }];

  let arguments = (ins
    Optional<AnyTypeOf<[PtrType, TensorViewLike],
                       "Ptr, TensorView, or PartitionTensorView">>:$gm_workspace,
    Optional<I32>:$used_cores,
    PTO_SyncAllModeAttr:$mode,
    PTO_SyncCoreTypeAttr:$core_type
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

// ---------------------------------------------------------------------------
// TileBuf intrinsics — used in TileLang DSL-generated template functions.
// These ops extract memref address and valid shape from tile_buf parameters.
// After inline, FoldTileBufIntrinsics resolves them to concrete values.
// ---------------------------------------------------------------------------

def SetFFTsOp : PTO_Op<"set_ffts", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Set FFTS/flags pointer for runtime (side-effecting).";
  let arguments = (ins PtrType:$ffts);
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
  let arguments = (ins StrAttr:$format, PrintScalarType:$scalar);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $format `,` $scalar `:` type($scalar) `)`
    attr-dict
  }];
}

def TrapOp : PTO_Op<"trap", [MicroOpInterface, SimtOpInterface]> {
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

#endif // MLIR_DIALECT_PTO_IR_PTOBASEOPS
