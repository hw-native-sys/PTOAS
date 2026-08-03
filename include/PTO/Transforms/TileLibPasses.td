// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// TileLib passes require a live compiler-session service and are intentionally
// excluded from the global textual pass registry. Their constructor strings
// suppress generated default factories; PTOAS creates them through the
// service-injecting factories declared in Passes.h.

#ifndef MLIR_DIALECT_PTO_TILELIB_PASSES
#define MLIR_DIALECT_PTO_TILELIB_PASSES

include "mlir/Pass/PassBase.td"

def InsertTemplateAttributes
    : Pass<"pto-insert-template-attributes", "ModuleOp"> {
  let summary = "Attach legal PTODSL template candidates to tile operations";
  let description = [{
    Queries the compiler's in-process PTODSL TileLib service for legal template
    candidates and stores the compact candidate list on each tile operation as
    the `candidates` attribute. Each candidate contains only id, name,
    loop_depth, postupdate, and tail metadata.
  }];
  let constructor = "mlir::pto::createInsertTemplateAttributesPass()";
  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def ExpandTileOp : Pass<"pto-expand-tile-op", "ModuleOp"> {
  let summary = "Expand tile ops into calls to TileLib template functions";
  let description = [{
    Expands tile-level operations (pto.tadd, pto.tsub, etc.) by asking the
    compiler's in-process PTODSL TileLib service to instantiate template
    libraries in the current MLIRContext. The generated template functions use
    tile_buf parameters and contain vector-level implementations (pto.vecscope,
    pto.vlds, pto.vadd, pto.vsts, etc.).

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
}

#endif // MLIR_DIALECT_PTO_TILELIB_PASSES
