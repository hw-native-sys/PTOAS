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
// TODO
//
//===----------------------------------------------------------------------===//

// The VPTO backend is emitted from tools/ptoas rather than a TableGen pass;
// these registrations continue to describe the shared pre-backend pipeline.

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

def PTOBufidSync : Pass<"pto-bufid-sync", "func::FuncOp"> {
  let summary = "Insert get_buf/rls_buf synchronization for PTO dialect";
  let description = [{
    Analyzes cross-pipeline local buffer dependencies and wraps dependent
    operations with A5 buffer-id synchronization. Unlike pto-insert-sync, this
    pass emits get_buf/rls_buf pairs instead of set_flag/wait_flag events.
  }];

  let constructor = "mlir::pto::createPTOBufidSyncPass()";

  let options = [
    Option<"enableBufidSyncDebug", "enable-bufid-sync-debug",
           "bool", "false",
           "Enable verbose debug printing for the bufid_sync pass">
  ];

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect"
  ];
}

// Define PTOGraphSyncSolver Pass (graph-based intra-core sync solver).
// Coexists with PTOInsertSync; selectable via the driver flag
// `--enable-graph-sync-solver`. Ported from bishengir's GraphSyncSolver core:
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

def PTORemoveIdentityTMov : Pass<"pto-remove-identity-tmov", "func::FuncOp"> {
  let summary = "Remove plain PTO TMOV operations that copy a tile onto itself";
  let description = [{
    Removes side-effect-free `pto.tmov` operations whose source and destination
    are proven to refer to the same physical tile address/range. Except for
    exact same-SSA self-copies, the proof goes through InsertSync's
    `BaseMemInfo` exact-range analysis for statically addressable values.
    Same-address copies are removed only when the TMOV destination/result is
    dead and no later same-address alias use is visible, so a live copy cannot
    be the sole synchronization bridge. This pass runs before InsertSync so
    deleting the identity TMOV also prevents
    synchronization from being generated only for that no-op copy.
  }];
  let constructor = "mlir::pto::createPTORemoveIdentityTMovPass()";
  let dependentDialects = [
    "pto::PTODialect",
    "arith::ArithDialect",
    "func::FuncDialect",
    "memref::MemRefDialect"
  ];
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
    Option<"orderBySize", "order-by-size", "bool",
           /*default=*/"false",
           "Process buffers largest-first (first-fit-decreasing order) during "
           "local memory planning instead of the default DMA-first order. "
           "Decreasing-size order packs heterogeneous-size buffers tighter "
           "(matches XLA/TVM/SOMAS). default : false">,
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

def PreFusionAnalysis : Pass<"pto-pre-fusion-analysis", "func::FuncOp"> {
  let summary = "Analyze block-local tile-fusion candidates before planning";
  let description = [{
    Builds reusable pre-planning analysis state for tile fusion in tile-native
    PTO IR. This pass is analysis-only and does not emit grouping or
    scheduling decisions.
  }];
  let constructor = "mlir::pto::createPreFusionAnalysisPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def PrintPreFusionAnalysis
    : Pass<"pto-print-pre-fusion-analysis", "func::FuncOp"> {
  let summary = "Print pre-fusion analysis results for testing and debugging";
  let description = [{
    Computes the pre-fusion analysis result and prints a stable textual dump of
    block-local compute nodes, edges, liveness, iteration-domain classes, and
    local boundaries.
  }];
  let constructor = "mlir::pto::createPrintPreFusionAnalysisPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def FusionPlan : Pass<"pto-fusion-plan", "func::FuncOp"> {
  let summary = "Build conservative tile-fusion planning groups from block-local analysis";
  let description = [{
    Consumes PreFusionAnalysis results, forms conservative block-local fusion
    groups for currently supported tile-native compute ops, and annotates
    accepted group members with:
      - pto.fusion.group_id
      - pto.fusion.order
  }];
  let constructor = "mlir::pto::createFusionPlanPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
  let options = [
    Option<"enableShapeInference", "enable-shape-inference",
           "bool", /*default=*/"false",
           "Enable shape inference (ShapeConstraintSolver) for tile fusion. "
           "When disabled (default), fall back to static/direct-bound "
           "iteration-domain inference.  Only FusionPlan consumes this option; "
           "FusionRegionGen consumes only the shared pre-fusion dataflow graph "
           "(cached by the analysis manager) and never consults iteration-domain "
           "classes, so it does not need the option.">
  ];
}

def OpScheduling : Pass<"pto-op-scheduling", "func::FuncOp"> {
  let summary = "Compact planned fusion groups into block-local contiguous spans";
  let description = [{
    Consumes fusion planning metadata emitted by FusionPlanPass and performs
    block-local instruction scheduling to make each accepted fusion group a
    contiguous span without redefining group membership.
  }];
  let constructor = "mlir::pto::createOpSchedulingPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def PTOMarkLastUse : Pass<"pto-mark-last-use", "func::FuncOp"> {
  let summary = "Mark scheduled tile-fusion operand last-use slots";
  let description = [{
    Walks scheduled tile-fusion spans identified by pto.fusion.group_id /
    pto.fusion.order and annotates each op with a stable per-input last-use
    bit mask. The analysis considers both later in-span uses and later
    post-span uses in the enclosing block.
  }];
  let constructor = "mlir::pto::createPTOMarkLastUsePass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def PTOFusionRegionGen : Pass<"pto-fusion-region-gen", "func::FuncOp"> {
  let summary = "Wrap one scheduled fusion span into one pto.fusion_region";
  let description = [{
    Consumes scheduled tile-fusion metadata and encapsulates one contiguous
    fusion group span into one pto.fusion_region without redefining group
    membership or physical order.
  }];
  let constructor = "mlir::pto::createPTOFusionRegionGenPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
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

def PTONormalizeUncoveredTileSections
    : Pass<"pto-normalize-uncovered-tile-sections", "ModuleOp"> {
  let summary = "Collect uncovered top-level PTO tile segments before section normalization";
  let description = [{
    Scans function top-level blocks for contiguous operation segments that are
    not already covered by `pto.section.cube` / `pto.section.vector`.

    This pass is the shared entry point for inferred-section normalization:
    explicit sections remain in place, while uncovered top-level segments that
    contain tile operations are identified for later wrapping.
  }];

  let constructor = "mlir::pto::createPTONormalizeUncoveredTileSectionsPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def PTOInferTileOpSummary
    : Pass<"pto-infer-tileop-summary", "func::FuncOp"> {
  let summary = "Infer phase summary attributes for PTODSL tileop helpers";
  let description = [{
    Scans functions marked with canonical `pto.tileop.helper` (while still
    accepting legacy `pto.ptodsl.subkernel_helper = "tileop"`) and
    derives the backend-owned summary attributes:
    - `pto.tileop.primary_domain`
    - `pto.tileop.phases`
    - `pto.tileop.operand_effects`

    This pass is intentionally summary-only. It does not materialize sections,
    inline helpers, or change InsertSync modeling.
  }];

  let constructor = "mlir::pto::createPTOInferTileOpSummaryPass()";

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::pto::PTODialect"
  ];
}

def PTOMaterializeTileOpSections
    : Pass<"pto-materialize-tileop-sections", "func::FuncOp"> {
  let summary = "Materialize one primary PTO section for PTODSL tileop helpers";
  let description = [{
    Consumes backend-owned tileop summary attributes on functions marked with
    canonical `pto.tileop.helper` (while still accepting legacy
    `pto.ptodsl.subkernel_helper = "tileop"`) and wraps one contiguous
    primary-domain compute span in `pto.section.vector` or `pto.section.cube`.

    Unlike `pto-normalize-uncovered-tile-sections`, this pass is helper-local
    and runs after tileop phase summary inference. It uses those inferred phases
    to preserve non-primary MTE/S/sync operations outside the materialized
    primary-domain section.

    The MVP implementation expects one contiguous primary compute span in the
    helper body. Leading and trailing MTE phases remain top-level so late
    helper inlining can expose VPTO section sugar to `vpto-split-cv-module`.
  }];

  let constructor = "mlir::pto::createPTOMaterializeTileOpSectionsPass()";

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::pto::PTODialect"
  ];
}

def PTOVerifyTileOpContract
    : Pass<"pto-verify-tileop-contract", "func::FuncOp"> {
  let summary = "Verify backend-owned PTODSL tileop helper contracts";
  let description = [{
    Verifies functions marked with canonical `pto.tileop.helper` (while still
    accepting legacy `pto.ptodsl.subkernel_helper = "tileop"`) after summary
    inference and section materialization.

    Current MVP contract:
    - results must be PTO scalar values only
    - helper-local `pto.alloc_tile`, `pto.reserve_buffer`, and `pto.talloc`
      are rejected
    - nested tileop helper calls are rejected
    - SIMT-only PTO ops are rejected
    - at least one primary compute op must exist, and all primary compute ops
      must belong to exactly one domain (`vector` or `cube`)
    - scalar/MTE/sync phases may coexist, but `pto.tileop.primary_domain`,
      `pto.tileop.phases`, and `pto.tileop.operand_effects` must remain
      consistent with the helper body
  }];

  let constructor = "mlir::pto::createPTOVerifyTileOpContractPass()";

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::pto::PTODialect"
  ];
}

def VPTOSplitCVModule : Pass<"vpto-split-cv-module", "ModuleOp"> {
  let summary = "Split a VPTO module with cube/vector sections into kernel modules";
  let description = [{
    Rewrites a single-module VPTO input where one `pto.kernel` function
    contains `pto.section.cube` / `pto.section.vector` regions into the
    normalized fatobj input form with one child module per kernel kind.

    The pass clones the original module for each target kind, removes the
    opposite section kind, and inlines the target section body. It does not
    analyze cross-section SSA dependencies; normal MLIR verification catches
    illegal uses after the non-target section is removed.
  }];

  let constructor = "mlir::pto::createVPTOSplitCVModulePass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def VPTONormalizeContainer : Pass<"vpto-normalize-container", "ModuleOp"> {
  let summary = "Normalize and verify VPTO kernel container shape";
  let description = [{
    Normalizes a VPTO kernel module carrying `pto.kernel_kind` into the canonical
    outer container form. Existing containers are left in place. The pass
    verifies that the resulting top-level module contains only child modules and
    that every child module carries `pto.kernel_kind`.
  }];

  let constructor = "mlir::pto::createVPTONormalizeContainerPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect"
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

def ExpandTileOp : Pass<"pto-expand-tile-op", "ModuleOp"> {
  let summary = "Expand tile ops into calls to TileLang DSL template functions";
  let description = [{
    Expands tile-level operations (pto.tadd, pto.tsub, etc.) by invoking the
    TileLang Python DSL to instantiate template libraries. The generated
    template functions use tile_buf parameters and contain vector-level
    implementations (pto.vecscope, pto.vlds, pto.vadd, pto.vsts, etc.).

    Each tile op is replaced by a func.call to the generated template function,
    with tile_buf operands passed directly (no type bridging).

    After this pass, the Inline pass inlines template bodies, and
    FoldTileBufIntrinsics resolves tile_buf_addr / tile_valid_rows /
    tile_valid_cols.
  }];
  let constructor = "mlir::pto::createExpandTileOpPass()";
  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect"
  ];
  let options = [
    Option<"tilelangPath", "tilelang-path", "std::string",
           /*default=*/"\"\"",
           "Path to directory of .py tilelang DSL template files">,
    Option<"tilelangPkgPath", "tilelang-pkg-path", "std::string",
           /*default=*/"\"\"",
           "PYTHONPATH for tilelang_dsl package (added to env)">,
    Option<"pythonExe", "python-exe", "std::string",
           /*default=*/"\"python3\"",
           "Python executable for tilelang DSL invocation">,
    Option<"daemonSocketPath", "daemon-socket-path", "std::string",
           /*default=*/"\"\"",
           "Path to Unix domain socket for daemon RPC (if empty, uses subprocess)">
  ];
}

def FoldTileBufIntrinsics : Pass<"pto-fold-tile-buf-intrinsics", "mlir::func::FuncOp"> {
  let summary = "Fold structured-view intrinsics after template inlining";
  let description = [{
    After TileLang DSL template functions are inlined, the IR contains
    structured-view intrinsics whose operands are now bound to concrete values.

    This pass resolves them:
    - pto.tile_buf_addr → replaced by the memref recovered from the active
      materialized tile handle, or by pto.castptr when the requested result
      type is !pto.ptr
    - pto.tile_valid_rows → folded to arith.constant if v_row is static,
      or replaced with the dynamic valid_row operand carried by the
      materialized tile handle
    - pto.tile_valid_cols → same as above for v_col

    tensor_view family:
    - pto.tensor_view_addr → traces through unrealized_conversion_cast →
      subview → reinterpret_cast, then folds to the base memref or to
      pto.castptr/pto.addptr on the base memref
    - pto.get_tensor_view_dim → folded to arith.constant for static subview
      sizes, or to the subview size SSA operand for dynamic dims
    - pto.get_tensor_view_stride → folded to the reinterpret_cast stride
      operand, multiplied by the subview stride when needed

    Dead unrealized_conversion_cast, memref.subview, and
    memref.reinterpret_cast ops exposed by folding are cleaned up after the
    rewrite.
  }];
  let constructor = "mlir::pto::createFoldTileBufIntrinsicsPass()";
  let options = [
    Option<"foldMode", "fold-mode", "std::string",
           /*default=*/"\"all\"",
           "Select which intrinsic family to fold: all, shape-only, or addr-only">
  ];
  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect"
  ];
}

def PTOCanonicalizeIR : Pass<"pto-canonicalize-ir", "func::FuncOp"> {
  let summary = "Canonicalize rank-2 view descriptors into rank-5 form (VPTO-only)";
  let description = [{
    Rewrites rank-2 tensor_view / partition_tensor_view descriptors into the
    canonical right-aligned rank-5 form [1, 1, 1, R, C], matching the 5D
    descriptor layout expected by VPTO lowering. Stride expansion uses the
    same cumulative-product rule as rightAlignTo5D and
    buildGlobalTensorShapeAndStride: stride[i] = shape[i+1] * stride[i+1].

    Currently gated on --pto-backend=vpto to limit blast radius. A3/A5
    EmitC codegen already pads strides to rank-5 via InferPTOLayout and
    buildGlobalTensorShapeAndStride, so it does not need this pass at the
    IR level. The gate can be lifted once the pass is proven stable.

    A post-canonicalization verification detects any surviving rank-2 view
    types to prevent silent failures when new view-consuming ops are added.
  }];
  let constructor = "mlir::pto::createPTOCanonicalizeIRPass()";
  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::arith::ArithDialect"
  ];
}

def PTOInlineLibCall : Pass<"pto-inline-libcall", "ModuleOp"> {
  let summary = "Materialize OP-Lib instance bodies and inline OP-Lib calls";
  let description = [{
    Resolves OP-Lib instance declarations generated by OP-Lib lowering,
    materializes instance bodies, and inlines OP-Lib calls into caller/fused
    helper functions. Function signatures stay in !pto.tile_buf form.
  }];
  let constructor = "mlir::pto::createPTOInlineLibCallPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
  let options = [Option<
      "debug", "debug", "bool",
      /*default=*/"false",
      "Enable verbose debug logging for OP-Lib instantiation/inlining">];
}

def PTOInlineBackendHelpers
    : Pass<"pto-inline-backend-helpers", "ModuleOp"> {
  let summary = "Inline backend helper functions shared by VPTO and EmitC";
  let description = [{
    Force-inlines backend helper functions that should not survive to backend-
    specific lowering:
    - PTODSL subkernel helpers marked with `pto.tileop.helper` or legacy
      `pto.ptodsl.subkernel_helper`

    This pass runs on the shared mainline before backend-specific pipelines so
    both VPTO and EmitC consume the same helper-inlined IR.
  }];
  let constructor = "mlir::pto::createPTOInlineBackendHelpersPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
  let options = [Option<
      "debug", "debug", "bool",
      /*default=*/"false",
      "Enable verbose debug logging for backend helper inlining">];
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

def PTOVerifySubkernelPipeContract
    : Pass<"pto-verify-subkernel-pipe-contract", "func::FuncOp"> {
  let summary = "Verify PTODSL subkernel helpers stay within one InsertSync pipe contract";
  let description = [{
    Verifies legacy PTODSL subkernel helpers marked with
    `pto.ptodsl.subkernel_helper` and their same-module local call closure stay
    within one role-consistent PTO compute contract before InsertSync models
    the call boundary as one compound node.

    Current contract:
    - `simd` helpers may only contain PTO ops on `PIPE_V`
    - `cube` helpers may only contain PTO ops on `PIPE_M`
    - both reject role-mismatched PTO pipe ops (including MTE/FIX/S) and
      nested PTODSL subkernel calls
    - local helper `func.call` is allowed only when the transitive callee closure
      satisfies the same contract
  }];

  let constructor = "mlir::pto::createPTOVerifySubkernelPipeContractPass()";

  let dependentDialects = [
    "mlir::func::FuncDialect",
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

def PTOResolveBufferSelect : Pass<"pto-resolve-buffer-select", "ModuleOp"> {
  let summary = "Lower pto.slot_marker to single-slot pointer_cast (multi-buffer)";
  let description = [{
    Consumes `pto.slot_marker %src[%k]` ops written by PTOViewToMemref to
    thread multi-buffer slot selection through the memref layer.

    The op is replaced by a fresh single-address `pto.pointer_cast` that
    refers to the chosen physical slot:

    - constant slot `k`: a single `pto.pointer_cast(addrK)` is emitted at
      the use site, using slot k from the underlying multi-address
      pointer_cast created by PTOPlanMemory / AllocToPointerCast.
    - dynamic slot `%k`: per-slot single-address `pto.pointer_cast`s are
      created and an N-way `arith.select` chain picks one according to the
      user-supplied SSA. This pass does NOT rewrite `%k` into `iv mod N`;
      the frontend expression is the slot selector.

    The original multi-address `pto.pointer_cast` is kept in IR as the
    "alloc anchor" so future sync passes can recognize the multi-buffer
    geometry.
  }];

  let constructor = "mlir::pto::createPTOResolveBufferSelectPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect"
  ];
}

def PTOUnrollSIMTFor : Pass<"pto-unroll-simt-for", "func::FuncOp"> {
  let summary =
      "Unroll explicitly annotated scf.for loops in pto.simt_entry functions";
  let description = [{
    Walks functions with `pto.simt_entry` attribute and fully unrolls `scf.for`
    loops that carry the `{pto.unroll = "full"}` attribute.

    Only loops with constant lower bound, upper bound, and positive step are
    unrolled.  The annotation is opt-in: loops without the attribute are
    left unchanged regardless of trip count.

    Full unrolling enables subsequent canonicalize/sccp/cse to constant-fold
    induction-variable-dependent conditionals (e.g. `scf.if i < 10`), eliminating
    divergent branches inside SIMTVF kernels.

    This pass must run before `createConvertSCFToCFPass` (i.e. before LLVM
    lowering) so that the unrolled structured control flow can be canonicalized.
  }];
  let constructor = "mlir::pto::createPTOUnrollSIMTForPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::pto::PTODialect",
    "mlir::arith::ArithDialect"
  ];
}

def PTOInferVPTOVecScope
    : Pass<"pto-infer-vpto-vecscope", "func::FuncOp"> {
  let summary =
      "Infer missing pto.vecscope regions for VPTO vector operation clusters";
  let description = [{
    Runs near the VPTO emission boundary after inlining, canonicalization,
    CSE, pointer normalization, and wrapper-op expansion have exposed the final
    SSA shape. The pass greedily clusters contiguous VPTO vector operations
    into `pto.vecscope` regions while preserving explicit vector-scope carriers
    and treating DMA/copy/sync, unresolved calls, terminators, and forbidden
    operations as boundaries.

    The inferred `pto.vecscope` form remains resultless. Values whose type is
    `!pto.vreg`, `!pto.mask`, or `!pto.align` must not escape the inferred
    scope.
  }];
  let constructor = "mlir::pto::createPTOInferVPTOVecScopePass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect"];
}

def PTOValidateVPTOIR : Pass<"pto-validate-vpto-ir", "ModuleOp"> {
  let summary =
      "Validate authoring-stage VPTO legality before ptr-boundary canonicalization";
  let description = [{
    Runs the authoring-stage VPTO legality verifier on post-mainline VPTO IR.
    This stage keeps the memref-first authoring surface legal, while checking
    the shared structural contracts that must hold before emission-boundary
    canonicalization.
  }];
  let constructor = "mlir::pto::createPTOValidateVPTOIRPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect"];
}

def PTOValidateVPTOEmissionIR
    : Pass<"pto-validate-vpto-emission-ir", "ModuleOp"> {
  let summary =
      "Validate emission-stage VPTO legality after ptr-boundary canonicalization";
  let description = [{
    Runs the emission-stage VPTO legality verifier on ptr-form VPTO IR after
    `PTOVPTOPtrBoundary`. This stage re-checks the shared authoring contracts
    and confirms the final emission surface no longer carries memref boundary
    state or residual wrapper scaffold.
  }];
  let constructor = "mlir::pto::createPTOValidateVPTOEmissionIRPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect"];
}

def VPTOExpandWrapperOps
    : Pass<"vpto-expand-wrapper-ops", "func::FuncOp"> {
  let summary =
      "Expand VPTO wrapper ops to emission-ready low-level VPTO IR";
  let description = [{
    Expand higher-level VPTO wrapper operations, including bridge, DMA, cube,
    accumulator-store, and MAD semantic forms, to the low-level pointer/raw VPTO
    operations consumed by backend emission.
  }];
  let constructor = "mlir::pto::createVPTOExpandWrapperOpsPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::LLVM::LLVMDialect",
                           "mlir::pto::PTODialect"];
}

def PTOVPTOPtrBoundary
    : Pass<"pto-vpto-ptr-boundary", "ModuleOp"> {
  let summary =
      "Canonicalize the final VPTO emission boundary from memref-first IR to ptr ABI";
  let description = [{
    Runs the final emission-boundary ptr canonicalization after the backend
    mainline has finished its memref-first optimization pipeline. This pass
    rewrites eligible memref function arguments to same-space `!pto.ptr`,
    rejects memref function results, canonicalizes supported body-level VPTO
    buffer-like ops to ptr-form, and drops dead boundary/view scaffold such as
    trivial `pto.castptr`, `pto.bind_tile`, `memref.subview`,
    `memref.reinterpret_cast`, and `memref.memory_space_cast` once they become
    unused.
  }];
  let constructor = "mlir::pto::createPTOVPTOPtrBoundaryPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect"];
}

def PTOLowLevelLoopFusion : Pass<"pto-low-level-loop-fusion", "ModuleOp"> {
  let summary =
      "Fuse adjacent VPTO post-lowering loop nests inside pto.fusion_region";
  let description = [{
    Conservatively fuses adjacent post-lowering loop stages for already-planned
    fusion groups after the tile-native frontend fusion core. The formal input
    contract for this pass is low-level `scf.for + pto.*` structure that
    remains inside `pto.fusion_region` until explicit flatten.
  }];
  let constructor = "mlir::pto::createPTOLowLevelLoopFusionPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::vector::VectorDialect"];
  let options = [Option<
      "debug", "debug", "bool",
      /*default=*/"false",
      "Enable verbose debug logging for low-level loop fusion">];
}

def PTOFusionPredicateElision
    : Pass<"pto-fusion-predicate-elision", "func::FuncOp"> {
  let summary =
      "Elide redundant fusion-local VPTO plt predicate materialization";
  let description = [{
    Scans `pto.fusion_region` bodies after VPTO post-lowering CSE and prepares
    fusion-local redundant `pto.plt_*` predicate elimination before
    load/store cleanup and explicit flatten.
  }];
  let constructor = "mlir::pto::createPTOFusionPredicateElisionPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
}

def PTOFusionLoadStoreElision
    : Pass<"pto-fusion-load-store-elision", "func::FuncOp"> {
  let summary =
      "Elide fusion-local VPTO load/store round trips before flattening";
  let description = [{
    Removes local store materialization inside canonical fused loop bodies
    before explicit flatten. It also performs conservative frontier-aware
    cleanup for non-escaping fusion-region internal buffers.
  }];
  let constructor = "mlir::pto::createPTOFusionLoadStoreElisionPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
}

def PTOFlattenFusionRegion : Pass<"pto-flatten-fusion-region", "func::FuncOp"> {
  let summary = "Flatten pto.fusion_region wrappers back into the parent block";
  let description = [{
    Eliminates residual `pto.fusion_region` wrappers after region-local
    lowering is complete by moving body operations back to the parent block,
    replacing region results with `pto.yield` operands, and erasing both
    `pto.yield` and the wrapper op.
  }];
  let constructor = "mlir::pto::createPTOFlattenFusionRegionPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def VPTOPtrNormalize
    : Pass<"vpto-ptr-normalize", "ModuleOp"> {
  let summary =
      "Normalize VPTO ptr-like values and users into a uniform !pto.ptr form";
  let description = [{
    Uses MLIR's conversion framework to normalize VPTO ptr-related forms before
    the existing VPTO ptr-boundary canonicalization runs. This pass rewrites
    supported tile-buffer and memref view producers such as `pto.tile_buf_addr`
    and `memref.subview`, and updates VPTO memory ops to consume the normalized
    ptr-form consistently.
  }];
  let constructor = "mlir::pto::createVPTOPtrNormalizePass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::arith::ArithDialect"];
}

def VPTOPtrCastCleanup
    : Pass<"vpto-ptr-cast-cleanup", "ModuleOp"> {
  let summary = "Collapse transient ptr-memref-ptr bridge casts after VPTO ptr normalization";
  let description = [{
    Eliminates bridge chains such as
    `!pto.ptr -> builtin.unrealized_conversion_cast -> memref.cast ->
     builtin.unrealized_conversion_cast -> !pto.ptr`
    when the outer ptr types already match.
  }];
  let constructor = "mlir::pto::createVPTOPtrCastCleanupPass()";
  let dependentDialects = ["mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect"];
}

#endif // MLIR_DIALECT_PTO_PASSES
