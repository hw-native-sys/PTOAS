// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOTilePipelineOps - semantic operation family -*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTOTILEPIPELINEOPS
#define MLIR_DIALECT_PTO_IR_PTOTILEPIPELINEOPS

include "PTO/IR/PTOOpCommon.td"

def TAllocToAivOp : PTO_TOp<"talloc_to_aiv"> {
  let summary = "Frontend C2V producer alloc of a GlobalTensor FIFO entry";

  let arguments = (ins
      DefaultValuedOptionalAttr<I32Attr, "0">:$id,
      I8Attr:$split
  );

  let results = (outs TensorViewType:$entry);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `{` (`id` `=` $id^ `,`)? `split` `=` $split `}` attr-dict
    `->` qualified(type($entry))
  }];
}

def TAllocToAicOp : PTO_TOp<"talloc_to_aic"> {
  let summary = "Frontend V2C producer alloc of a GlobalTensor FIFO entry";

  let arguments = (ins
      DefaultValuedOptionalAttr<I32Attr, "0">:$id,
      I8Attr:$split
  );

  let results = (outs TensorViewType:$entry);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `{` (`id` `=` $id^ `,`)? `split` `=` $split `}` attr-dict
    `->` qualified(type($entry))
  }];
}

def TPushToAivOp : PTO_TOp<"tpush_to_aiv", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Frontend C2V producer push from Cube to Vector";

  let arguments = (ins
      PTOPipeEntryType:$tile,
      DefaultValuedOptionalAttr<I32Attr, "0">:$id,
      I8Attr:$split
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $tile `:` qualified(type($tile)) `)`
    `{` (`id` `=` $id^ `,`)? `split` `=` $split `}` attr-dict
  }];
}

def TPushToAicOp : PTO_TOp<"tpush_to_aic"> {
  let summary = "Frontend V2C producer push from Vector to Cube";

  let arguments = (ins
      PTOPipeEntryType:$tile,
      Optional<I64>:$aiv_subblockid,
      DefaultValuedOptionalAttr<I32Attr, "0">:$id,
      I8Attr:$split
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $tile `:` qualified(type($tile)) `)`
    `{` (`id` `=` $id^ `,`)? `split` `=` $split `}`
    (`aiv_subblockid` `(` $aiv_subblockid^ `)`)?
    attr-dict
  }];
}

def TPopFromAicOp : PTO_TOp<"tpop_from_aic", [AttrSizedOperandSegments]> {
  let summary = "Frontend C2V consumer pop in Vector kernel";

  let arguments = (ins
      Optional<Index>:$valid_row,
      Optional<Index>:$valid_col,
      Optional<I64>:$aiv_subblockid,
      DefaultValuedOptionalAttr<I32Attr, "0">:$id,
      I8Attr:$split
  );

  let results = (outs PTOPipeEntryType:$tile);
  let hasVerifier = 1;

  let assemblyFormat = [{
    (`(` $valid_row^ `,` $valid_col `)`)?
    `{` (`id` `=` $id^ `,`)? `split` `=` $split `}`
    (`aiv_subblockid` `(` $aiv_subblockid^ `)`)?
    attr-dict
    `->` qualified(type($tile))
  }];
}

def TPopFromAivOp : PTO_TOp<"tpop_from_aiv", [AttrSizedOperandSegments]> {
  let summary = "Frontend V2C consumer pop in Cube kernel";

  let arguments = (ins
      Optional<Index>:$valid_row,
      Optional<Index>:$valid_col,
      DefaultValuedOptionalAttr<I32Attr, "0">:$id,
      I8Attr:$split
  );

  let results = (outs PTOPipeEntryType:$tile);
  let hasVerifier = 1;

  let assemblyFormat = [{
    (`(` $valid_row^ `,` $valid_col `)`)? `{` (`id` `=` $id^ `,`)? `split` `=` $split `}` attr-dict
    `->` qualified(type($tile))
  }];
}

def TFreeFromAicOp : PTO_TOp<"tfree_from_aic"> {
  let summary = "Frontend C2V consumer free in Vector kernel";

  let arguments = (ins
      Optional<PTOPipeEntryType>:$entry,
      DefaultValuedOptionalAttr<I32Attr, "0">:$id,
      I8Attr:$split
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    (`(` $entry^ `:` qualified(type($entry)) `)`)?
    `{` (`id` `=` $id^ `,`)? `split` `=` $split `}` attr-dict
  }];
}

def TFreeFromAivOp : PTO_TOp<"tfree_from_aiv"> {
  let summary = "Frontend V2C consumer free in Cube kernel";

  let arguments = (ins
      Optional<PTOPipeEntryType>:$entry,
      DefaultValuedOptionalAttr<I32Attr, "0">:$id,
      I8Attr:$split
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    (`(` $entry^ `:` qualified(type($entry)) `)`)?
    `{` (`id` `=` $id^ `,`)? `split` `=` $split `}` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Fixpipe Quant Config Ops
//===----------------------------------------------------------------------===//

def SetQuantScalarOp : PTO_Op<"set_quant_scalar", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Set scalar quantization parameters for fixpipe producer";

  let description = [{
    Producer-side scalar quantization configuration for fixpipe TPUSH.
    Binds an f32 scalar quant payload to a logical pipe id.
    This binding persists until overwritten by another set_quant_scalar with
    the same id. Used by scalar quant modes: deqf16_scalar, req8_scalar,
    qf322b8_pre_scalar, qf322f16_pre_scalar, qf322bf16_pre_scalar,
    qs322bf16_pre_scalar, qf322hif8_pre_scalar, qf322fp8_pre_scalar.
  }];

  let arguments = (ins
      F32:$scale,
      DefaultValuedOptionalAttr<I32Attr, "0">:$id
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $scale `:` type($scale) `)` `{` `id` `=` $id `}` attr-dict
  }];
}

def SetQuantVectorOp : PTO_Op<"set_quant_vector", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Set vector quantization parameters for fixpipe producer";

  let description = [{
    Producer-side vector quantization configuration for fixpipe TPUSH.
    Binds a vector quant payload (scaling tile) to a logical pipe id.
    The input tile must have loc=scaling. A5 consumes a direct scaling
    payload with an element type in the f16/bf16/f32 family; A3 consumes
    a packed 64-bit integer payload (i64/ui64). This binding persists
    until overwritten by another set_quant_vector with the same id.
    Used by vector quant modes: deqf16_vec, req8_vec, qf322b8_pre_vec,
    qs322bf16_pre_vec.
  }];

  let arguments = (ins
      PTOPipeEntryType:$scaling_tile,
      DefaultValuedOptionalAttr<I32Attr, "0">:$id
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $scaling_tile `:` qualified(type($scaling_tile)) `)`
    `{` `id` `=` $id `}` attr-dict
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

def TPutAsyncOp : PTO_Op<"comm.tput_async", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Asynchronous remote write from local GM to remote GM";

  let arguments = (ins
      PTOCommType:$dst,
      PTOCommType:$src,
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

def TGetAsyncOp : PTO_Op<"comm.tget_async", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Asynchronous remote read from remote GM to local GM";

  let arguments = (ins
      PTOCommType:$dst,
      PTOCommType:$src,
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

def TPutOp : PTO_Op<"comm.tput", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Synchronous remote write from local GM to remote GM";
  let arguments = (ins
      PTOCommType:$dst,
      PTOCommType:$src,
      PTOCommType:$ping,
      Optional<PTOCommType>:$pong,
      DefaultValuedAttr<PTO_AtomicTypeAttr, "::mlir::pto::AtomicType::AtomicNone">:$atomicType
  );
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` $dst `,` $src `,` `buf` `(` $ping (`,` $pong^)? `)`
    `:` type($dst) `,` type($src) `,` type($ping) (`,` type($pong)^)? `)`
    attr-dict
  }];
}

def TGetOp : PTO_Op<"comm.tget", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Synchronous remote read from remote GM to local GM";
  let arguments = (ins
      PTOCommType:$dst,
      PTOCommType:$src,
      PTOCommType:$ping,
      Optional<PTOCommType>:$pong
  );
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` $dst `,` $src `,` `buf` `(` $ping (`,` $pong^)? `)`
    `:` type($dst) `,` type($src) `,` type($ping) (`,` type($pong)^)? `)`
    attr-dict
  }];
}

def TBroadcastOp : PTO_Op<"comm.tbroadcast", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Broadcast local GM data to all group members";
  let arguments = (ins
      PTOCommType:$src,
      PTOCommType:$ping,
      Optional<PTOCommType>:$pong,
      Variadic<PTOCommType>:$group,
      I32Attr:$root
  );
  let results = (outs);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def CommTGatherOp : PTO_Op<"comm.tgather", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Gather remote GM data from a communication group";
  let arguments = (ins
      PTOCommType:$dst,
      PTOCommType:$ping,
      Optional<PTOCommType>:$pong,
      Variadic<PTOCommType>:$group,
      I32Attr:$root
  );
  let results = (outs);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def CommTScatterOp : PTO_Op<"comm.tscatter", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Scatter local GM data to a communication group";
  let arguments = (ins
      PTOCommType:$src,
      PTOCommType:$ping,
      Optional<PTOCommType>:$pong,
      Variadic<PTOCommType>:$group,
      I32Attr:$root
  );
  let results = (outs);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def TReduceOp : PTO_Op<"comm.treduce", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Reduce remote GM data from a communication group";
  let arguments = (ins
      PTOCommType:$dst,
      PTOCommType:$acc,
      PTOCommType:$recvPing,
      Optional<PTOCommType>:$recvPong,
      Variadic<PTOCommType>:$group,
      PTO_ReduceOpAttr:$reduceOp,
      I32Attr:$root
  );
  let results = (outs);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def TPushOp : PTO_TOp<"tpush", [
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Push tile data through a unified pipe handle";

  let arguments = (ins
      PTOPipeEntryType:$tile,
      PipeType:$pipe_handle,
      Optional<I64>:$aiv_subblockid,
      I8Attr:$split
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $tile `,` $pipe_handle `:` qualified(type($tile)) `,` qualified(type($pipe_handle)) `)`
    `{` `split` `=` $split `}`
    (`aiv_subblockid` `(` $aiv_subblockid^ `)`)?
    attr-dict
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

def DeclareTileOp : PTO_Op<"declare_tile"> {
  let summary = "Declare a tile value whose address will be assigned later";

  let results = (outs PTODpsType:$tile);

  let assemblyFormat = [{
    attr-dict `->` qualified(type($tile))
  }];
}

def TAllocOp : PTO_TOp<"talloc", [
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Bind a declared GlobalTensor descriptor to the next producer FIFO slot";

  let arguments = (ins
      TensorViewType:$entry,
      PipeType:$pipe_handle,
      I8Attr:$split
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $entry `,` $pipe_handle `:` qualified(type($entry)) `,` qualified(type($pipe_handle)) `)`
    `{` `split` `=` $split `}` attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE3; }
  }];
}

def TAssignOp : PTO_Op<"tassign", [Pure, AllTypesMatch<["tile", "result"]>]> {
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

def TPopOp : PTO_TOp<"tpop", [
    PTO_DpsInitOpInterface,
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Bind a declared consumer tile to the next FIFO slot";

  let arguments = (ins
      PTOPipeEntryType:$tile,
      PipeType:$pipe_handle,
      Optional<I64>:$aiv_subblockid,
      I8Attr:$split
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $tile `,` $pipe_handle `:` qualified(type($tile)) `,` qualified(type($pipe_handle)) `)`
    `{` `split` `=` $split `}`
    (`aiv_subblockid` `(` $aiv_subblockid^ `)`)?
    attr-dict
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
      Optional<PTOPipeEntryType>:$entry,
      PipeType:$pipe_handle,
      I8Attr:$split
  );

  let results = (outs);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_S; }
  }];
}

//===----------------------------------------------------------------------===//
// Synchronization Ops
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
    PTO_PipeLikeAttr:$op_type,
    I32Attr:$buf_id,
    DefaultValuedAttr<I32Attr, "0">:$mode
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    void print(::mlir::OpAsmPrinter &p);
    static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser,
                                     ::mlir::OperationState &result);
  }];
}

def GetBufDynOp : PTO_Op<"get_buf_dyn"> {
  let summary = "Acquire a buffer-id token (dynamic buf-id)";
  let description = [{
    Dynamic variant of `pto.get_buf` where the buffer-id is provided as an
    SSA value (index) instead of a static integer attribute. This enables
    runtime-computed buf_id patterns such as SIMT ping-pong buffering
    (e.g. `iter & 1`).
  }];

  let arguments = (ins
    PTO_PipeLikeAttr:$op_type,
    Index:$buf_id,
    DefaultValuedAttr<I32Attr, "0">:$mode
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def RlsBufOp : PTO_Op<"rls_buf"> {
  let summary = "Release a buffer-id token for a sync op type (A5)";
  let description = [{
    Releases the previously acquired buffer-id token for the given op type
    (mapped to a concrete pipe).

    This op is intended to be lowered to the CCEC builtin intrinsic `rls_buf`.
  }];

  let arguments = (ins
    PTO_PipeLikeAttr:$op_type,
    I32Attr:$buf_id,
    DefaultValuedAttr<I32Attr, "0">:$mode
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    void print(::mlir::OpAsmPrinter &p);
    static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser,
                                     ::mlir::OperationState &result);
  }];
}

def RlsBufDynOp : PTO_Op<"rls_buf_dyn"> {
  let summary = "Release a buffer-id token (dynamic buf-id)";
  let description = [{
    Dynamic variant of `pto.rls_buf` where the buffer-id is provided as an
    SSA value (index) instead of a static integer attribute.
  }];

  let arguments = (ins
    PTO_PipeLikeAttr:$op_type,
    Index:$buf_id,
    DefaultValuedAttr<I32Attr, "0">:$mode
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def TSyncOp : PTO_TOp<"tsync"> {
  let summary = "Direct TSYNC mapping (variadic operands).";
  let description = [{
    Lowers directly to a `TSYNC(...)` intrinsic call with the same operand list.
    This op is intentionally a thin bridge for explicit synchronization usage.
  }];

  let arguments = (ins Variadic<PTODpsType>:$events);
  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $events `:` type($events) `)` attr-dict
  }];
}

#endif // MLIR_DIALECT_PTO_IR_PTOTILEPIPELINEOPS
