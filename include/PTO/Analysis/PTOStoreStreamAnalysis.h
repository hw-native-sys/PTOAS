// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Restricted certificate for a single, isolated store stream in a fixed loop.
#ifndef PTO_ANALYSIS_PTOSTORESTREAMANALYSIS_H
#define PTO_ANALYSIS_PTOSTORESTREAMANALYSIS_H
#include "PTO/Analysis/PTOMemoryAccessAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::pto {
struct PTOIsolatedStoreStream {
    VstusOp store;
    VstasOp flush;
    PTOMemoryFootprint footprint;
};
// Includes a conservative 32-byte transaction envelope. Unknown operations,
// nested regions, intermediate consumers and noncontinuous addresses fail.
std::optional<PTOIsolatedStoreStream> getIsolatedStoreStream(scf::ForOp loop);
} // namespace mlir::pto
#endif
