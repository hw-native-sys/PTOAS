// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_VPTOEXPANDMADOPS_H
#define PTO_TRANSFORMS_VPTOEXPANDMADOPS_H

#include "mlir/IR/PatternMatch.h"

namespace mlir::pto::expand_mad {

// Adds the patterns lowering the mad semantic ops (mad/mad_acc/mad_bias and
// their mx variants) to the matching raw ops inside a ctrl_state_guard.
void populateExpandMadPatterns(RewritePatternSet &patterns);

} // namespace mlir::pto::expand_mad

#endif // PTO_TRANSFORMS_VPTOEXPANDMADOPS_H
