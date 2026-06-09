// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===-- Passes.td - PTO pass definition file ---------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
//
// Pass registration notes
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_PASSES
#define MLIR_DIALECT_PTO_PASSES

include "mlir/Pass/PassBase.td"

// Define PTOInsertSync Pass
def PTOInsertSync : Pass<"pto-insert-sync", "func::FuncOp"> {
  let summary = "Insert synchronization flags for PTO dialect";
  let description = [{
    Analyzes data dependencies between Cube, Vector and MTE pipelines
    and inserts explicit synchronization instructions.
  }];
 
  // Declare the constructor function name
  let constructor = "mlir::pto::createPTOInsertSyncPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect" 
  ];
}

def PTOInjectBarrierAllSync : Pass<"pto-inject-barrier-all-sync", "func::FuncOp"> {
  let summary = "Conservatively insert PIPE_ALL barriers for PTO dialect";
  let description = [{
    Inserts a `pto.barrier <PIPE_ALL>` before each PTO pipe operation that has
    MemoryEffects read/write side effects. This mode does not reuse set/wait
    dependency analysis and does not replace the internal synchronization
    protocol of composite pipe ops such as `pto.tpush`, `pto.tpop`, and
    `pto.tfree`; it only adds conservative barriers at operation boundaries.
  }];

  let constructor = "mlir::pto::createPTOInjectBarrierAllSyncPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect"
  ];
}

// Define PTOGraphSyncSolver Pass (graph-based intra-core sync solver).
// Coexists with PTOInsertSync; selectable via the driver flag
// `--enable-graph-sync-solver`. Ported from bishengir's GraphSyncSolver core
// hierarchical SyncSolver IR, conflict graph pruning, event-id coloring and
// barrier fallback, with PTO-specific IR translation and sync op codegen.
def PTOGraphSyncSolver : Pass<"pto-graph-sync-solver", "func::FuncOp"> {
  let summary = "Graph-based intra-core synchronization solver for PTO";
  let description = [{
    Alternative to PTOInsertSync. Builds a hierarchical SyncSolver IR and
    decides which set/wait/barrier ops to emit by combining a reachability
    analysis (so transitively-covered candidates are skipped) with graph
    coloring over the pool of EVENT_IDs.
  }];

  let constructor = "mlir::pto::createPTOGraphSyncSolverPass()";

  let options = [
    Option<"eventIdNumMax", "event-id-num-max", "int64_t",
           /*default=*/"8",
           "Maximum EVENT_ID slots usable by coloring; this caps the hardware "
           "available event-id budget used by the graph solver.">,
  ];

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect"
  ];
}

def ConvertToPTOOp : Pass<"convert-to-pto-op"> {
  let summary = "Convert Ops from other dialects to PTO Ops";
  let constructor = "mlir::pto::createConvertToPTOOpPass()";
  let dependentDialects = ["arith::ArithDialect", "pto::PTODialect",
                           "tensor::TensorDialect",
  ];
}

def InferPTOLayout : Pass<"pto-infer-layout", "func::FuncOp"> {
  let summary = "Infer GlobalTensor layout (ND/DN/NZ) for make_tensor_view";
  let description = [{
    For each pto.make_tensor_view with fully static shape/stride, compute layout
    (ND/DN/NZ) per design rules and attach the optional `layout` attribute.
    Dynamic stride/shape are left unset.
  }];
  let constructor = "mlir::pto::createInferPTOLayoutPass()";
  let dependentDialects = ["pto::PTODialect", "arith::ArithDialect"];
}

def PTOA5NormalizeTMov : Pass<"pto-a5-normalize-tmov", "func::FuncOp"> {
  let summary = "Normalize risky A5 vec->vec col_major TMOV into row-major reinterpret + TMOV";
  let description = [{
    Rewrites A5 `pto.tmov` patterns that use vec->vec col_major/none_box tiles:
      src(col_major) -> pto.treshape(row_major) -> pto.tmov(row_major->row_major)
      dst(col_major) -> pto.treshape(row_major)
    This avoids unsupported A5 PTO-ISA vec->vec col_major TMOV paths while
    preserving alias semantics via SSA treshape (no real data movement).
  }];
  let constructor = "mlir::pto::createPTOA5NormalizeTMovPass()";
  let dependentDialects = ["pto::PTODialect", "func::FuncDialect"];
}


def InferPTOMemScope : Pass<"pto-infer-mem-scope"> {
  let summary = "Infer memory scope for PTO Ops";
  let constructor = "mlir::pto::createInferPTOMemScopePass()";
  let dependentDialects = ["pto::PTODialect", "memref::MemRefDialect", ];
}

def PlanMemory : Pass<"pto-plan-memory", "ModuleOp"> {
  let summary = "Plan memory for PTO Ops";
  let constructor = "mlir::pto::createPlanMemoryPass()";
  let dependentDialects = ["pto::PTODialect", ];
  let options = [
    Option<"memMode", "mem-plan-mode", "pto::MemPlanMode",
           "pto::MemPlanMode::LOCAL_MEM_PLAN",
           "plan mem mode (default is LOCAL_MEM_PLAN)",
           [{::llvm::cl::values(
               clEnumValN(pto::MemPlanMode::LOCAL_MEM_PLAN, "local-mem-plan",
                          "plan mem mode is for memref.alloc"),
               clEnumValN(
                   pto::MemPlanMode::GLOBAL_WORKSPACE_PLAN,
                   "global-work-space-plan",
                   "plan mem mode is for memref_ext.alloc_workspace"))}]>,
    Option<"enableGlobalReuse", "enable-global-workspace-reuse", "bool",
           /*default=*/"false",
           "Enable global workspace reuse ,default : false">,
    Option<"enablePrintMemoryAllocatedSize", "enable-print-memory-allocated-size", "bool",
           /*default=*/"false",
           "Enable print memory allocated size, default : false">,
    Option<"restrictInplaceAsISA", "restrict-inplace-as-isa", "bool",
           /*default=*/"false",
           "restrict memory inplace as isa, default : false">,
  ];
}
def PTOLoweringSyncToPipe : Pass<"pto-lowering-sync-to-pipe", "func::FuncOp"> {
  let summary = "Lower high-level sync ops to low-level pipe ops";
  let description = [{
    Converts high-level `pto.record_event` and `pto.wait_event` operations
    into low-level `pto.set_flag` and `pto.wait_flag` operations based on
    hardware pipe mapping logic.
  }];

  let constructor = "mlir::pto::createLoweringSyncToPipePass()";
  let dependentDialects = [
    "mlir::pto::PTODialect"
  ];
}

def PTOWrapFunctionsInSections : Pass<"pto-wrap-functions-in-sections", "func::FuncOp"> {
  let summary = "Wrap attributed single-core functions in PTO cube/vector sections";
  let description = [{
    Rewrites frontend-marked functions into the existing PTO section form:
    - `pto.kernel_kind = #pto.kernel_kind<cube>` -> wrap the function body in
      `pto.section.cube`
    - `pto.kernel_kind = #pto.kernel_kind<vector>` -> wrap the function body
      in `pto.section.vector`
  }];

  let constructor = "mlir::pto::createPTOWrapFunctionsInSectionsPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def PTOLowerFrontendPipeOps : Pass<"pto-lower-frontend-pipe-ops", "func::FuncOp"> {
  let summary = "Lower frontend TPUSH/TPOP pipe ops to internal pipe ops";
  let description = [{
    Converts frontend `aic/aiv_initialize_pipe`, `tpush_to_*`,
    `tpop_from_*`, and `tfree_from_*` ops into the internal unified pipe IR:
    `initialize_l2l_pipe` / `initialize_l2g2l_pipe`, `tpush`,
    `declare_tile`, `tpop`, and `tfree`.
  }];

  let constructor = "mlir::pto::createPTOLowerFrontendPipeOpsPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def PTOAssignDefaultFrontendPipeId : Pass<"pto-assign-default-frontend-pipe-id", "func::FuncOp"> {
  let summary = "Materialize default id = 0 on frontend TPUSH/TPOP pipe ops";
  let description = [{
    Preserves backward compatibility with older frontend pipe syntax that
    omitted `id`. This pass rewrites missing `id` attrs on
    `aic/aiv_initialize_pipe`, `tpush_to_*`, `tpop_from_*`, and
    `tfree_from_*` to an explicit `id = 0` before frontend pipe lowering.
  }];

  let constructor = "mlir::pto::createPTOAssignDefaultFrontendPipeIdPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def PTOInferValidatePipeInit : Pass<"pto-infer-validate-pipe-init", "ModuleOp"> {
  let summary = "Infer and validate internal pipe init nosplit configuration";
  let description = [{
    Runs after frontend pipe lowering and before memory planning. For each
    logical pipe, this pass:
    - validates that downstream `pto.tpush` / `pto.tpop` / `pto.tfree` users
      do not mix `split = 0` with `split = 1/2`
    - preserves explicit `nosplit` attrs on internal pipe init ops and rejects
      conflicts across peer pipe init pairs
    - infers missing `nosplit` attrs from downstream split usage for backward
      compatibility with older IR that omitted init-level `nosplit`
    - propagates the resolved `nosplit` value across peer pipe init pairs so
      both ends of one logical pipe agree before EmitC lowering
  }];

  let constructor = "mlir::pto::createPTOInferValidatePipeInitPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def PTOResolveReservedBuffers : Pass<"pto-resolve-reserved-buffers", "ModuleOp"> {
  let summary = "Resolve reserved local buffer addresses and peer pipe flag bases";
  let description = [{
    Runs after `pto-plan-memory`. Assumes `pto.reserve_buffer` base addresses
    have already been planned, then:
    - aligns missing `flag_base` attrs for peer internal pipe init ops
    - rejects internal pipe init ops without explicit `flag_base` when their
      `local_addr` cannot be traced back to `pto.reserve_buffer` /
      `pto.import_reserved_buffer`
    - replaces `pto.reserve_buffer` / `pto.import_reserved_buffer` results with
      resolved constant local addresses
    - erases the frontend-only reserve/import ops before EmitC lowering
  }];

  let constructor = "mlir::pto::createPTOResolveReservedBuffersPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect"
  ];
}

def PTOVerifyTFree : Pass<"pto-verify-tfree", "func::FuncOp"> {
  let summary = "Verify explicit matching pto.tfree placement for pto.tpop";
  let description = [{
    For each `pto.tpop` in section.cube / section.vector, verify there is a
    matching later `pto.tfree(pipe_handle)` in the same block, the borrowed
    tile is not used after the free, and the same pipe does not accumulate
    multiple outstanding pops before the matching free.
  }];

  let constructor = "mlir::pto::createPTOVerifyTFreePass()";

  let dependentDialects = [
    "mlir::pto::PTODialect"
  ];
}

def PTOViewToMemref : Pass<"pto-view-to-memref", "ModuleOp"> {
  let summary = "Lower PTO views to memref with Metadata Binding";
  let description = [{
    Lower PTO tile/view operations to memref-based IR while preserving tile
    metadata through binding ops and SSA backtracking.
  }];

  let constructor = "mlir::pto::createPTOViewToMemrefPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect"
  ];
}

def PTOValidateIntToPtrUses : Pass<"pto-validate-inttoptr-uses", "func::FuncOp"> {
  let summary = "Validate restricted uses of PTO inttoptr results";
  let description = [{
    Verifies that `pto.inttoptr` results do not escape into general pointer IR.
    The result may only be used directly as the pointer operand of
    `pto.load_scalar` or `pto.store_scalar`.
  }];

  let constructor = "mlir::pto::createPTOValidateIntToPtrUsesPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def PTOMaterializeTileHandles : Pass<"pto-materialize-tile-handles", "ModuleOp"> {
  let summary = "Materialize tile_buf handles from planned memrefs before EmitC";
  let description = [{
    Re-wraps local memref values used by tile operations into explicit
    pto.materialize_tile ops. This keeps PlanMemory and InjectSync in memref IR
    while making the tile handle boundary visible to EmitC lowering.
  }];

  let constructor = "mlir::pto::createPTOMaterializeTileHandlesPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect"
  ];
}

#endif // MLIR_DIALECT_PTO_PASSES
