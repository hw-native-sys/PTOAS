// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_VECSCOPEMEMBAR_VECSCOPEMEMORYFOOTPRINT_H
#define PTO_TRANSFORMS_VECSCOPEMEMBAR_VECSCOPEMEMORYFOOTPRINT_H
#include "PTO/Analysis/PTOMemoryAccessAnalysis.h"

namespace mlir::pto::vecscopemembar {
struct AccessOccurrence {
    unsigned id = 0;
    Operation* op = nullptr;
    PTOMemoryAccessKind kind = PTOMemoryAccessKind::Load;
    PTOMemoryAccessUnit unit = PTOMemoryAccessUnit::Vector;
    PTOMemoryFootprint footprint;
    SmallVector<unsigned, 2> loopNest;
    SmallVector<unsigned, 2> schedulePath;
    unsigned lexicalOrder = 0;
};


}
#endif
