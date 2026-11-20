// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_ANALYSIS_STORE_STREAM_RANGE_H
#define PTO_ANALYSIS_STORE_STREAM_RANGE_H
#include "PTO/Analysis/PTOIntegerRange.h"
#include "PTO/Analysis/PTOMemoryAccessAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
namespace mlir::pto {
// Include enclosing domains when proving a nested stream. Only the current
// loop is collapsed; enclosing IVs remain in the exported footprint.
struct StreamDomains {
    SmallVector<Value> ivs;
    SmallVector<integer_range::PTOLoopDomain> bounds;

    explicit StreamDomains(scf::ForOp loop)
    {
        for (Operation* op = loop; op; op = op->getParentOp()) {
            if (auto parent = dyn_cast<scf::ForOp>(op)) {
                ivs.push_back(parent.getInductionVar());
                bounds.push_back(integer_range::getLoopEnvelope(parent));
            }
        }
    }
};

bool addressIsBounded(Value pointer, scf::ForOp loop, unsigned depth = 0);
std::optional<PTOMemoryFootprint> boundFootprint(PTOMemoryFootprint footprint, scf::ForOp loop);
std::optional<PTOMemoryFootprint> closeEnclosingDomains(PTOMemoryFootprint footprint, scf::ForOp loop);
std::optional<PTOMemoryFootprint> transactionEnvelope(VstusOp store, scf::ForOp loop, int64_t bytes);
}
#endif
