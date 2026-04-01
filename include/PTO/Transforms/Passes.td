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

#endif // MLIR_DIALECT_PTO_PASSES
