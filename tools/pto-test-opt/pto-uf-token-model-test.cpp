// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Exercise the structural proof directly: final event placement can hide a
// rejected pair behind another already-proven same-iteration handoff.
#include "PTO/Transforms/InsertSync/UFTokenModel.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::pto;

static void appendAction(SyncIRs& ir, MLIRContext& context, bool publish)
{
    auto node = std::make_unique<CompoundInstanceElement>(
        ir.size(), SmallVector<const BaseMemInfo*>{}, SmallVector<const BaseMemInfo*>{},
        publish ? PipelineType::PIPE_M : PipelineType::PIPE_FIX,
        OperationName(publish ? "pto.tmatmul" : "pto.tstore", &context));
    node->ufInfo.role = publish ? UFRole::MatmulProducer : UFRole::AccStore;
    node->ufInfo.accPhase = AccPhase::Final;
    node->ufInfo.stPhase = STPhase::Final;
    ir.push_back(std::move(node));
}

static bool checkPublishFirst(MLIRContext& context)
{
    SyncIRs ir;
    ir.push_back(std::make_unique<LoopInstanceElement>(0, 0, 3));
    appendAction(ir, context, true);
    appendAction(ir, context, false);
    ir.push_back(std::make_unique<LoopInstanceElement>(3, 0, 3, KindOfLoop::LOOP_END));
    UFTokenModel model(ir);
    model.build();
    // The circular RAW span contains store_i and matmul_(i+1).
    return model.isHandoffAdjacent(1, 2, false, 0) &&
           model.isHandoffAdjacent(2, 1, true, 3) &&
           !model.isHandoffAdjacent(1, 2, true, 3);
}

static unsigned appendConsumeLoop(SyncIRs& ir, MLIRContext& context)
{
    const unsigned begin = ir.size();
    ir.push_back(std::make_unique<LoopInstanceElement>(begin, begin, begin + 3));
    appendAction(ir, context, false);
    appendAction(ir, context, true);
    ir.push_back(std::make_unique<LoopInstanceElement>(begin + 3, begin, begin + 3, KindOfLoop::LOOP_END));
    return begin;
}

static bool checkConsumeFirst(MLIRContext& context, bool prologue, bool precedingLoop = false)
{
    SyncIRs ir;
    if (prologue) {
        appendAction(ir, context, true);
    }
    if (precedingLoop) {
        appendConsumeLoop(ir, context);
    }
    const unsigned begin = appendConsumeLoop(ir, context);
    appendAction(ir, context, false);
    UFTokenModel model(ir);
    model.build();
    const bool forward = model.isHandoffAdjacent(begin + 1, begin + 2, false, 0);
    const bool carried = model.isHandoffAdjacent(begin + 2, begin + 1, true, begin + 3);
    const bool wrongCarried = model.isHandoffAdjacent(begin + 1, begin + 2, true, begin + 3);
    const bool boundary = model.isPipelineBoundaryOrdered(0, begin + 4);
    return forward == prologue && carried == prologue && !wrongCarried &&
           boundary == (prologue && !precedingLoop);
}

int main()
{
    MLIRContext context;
    const bool valid = checkPublishFirst(context) && checkConsumeFirst(context, true) &&
                       checkConsumeFirst(context, false) && checkConsumeFirst(context, false, true) &&
                       checkConsumeFirst(context, true, true);
    if (!valid) {
        llvm::errs() << "UF token loop proof regression\n";
        return 1;
    }
    return 0;
}
