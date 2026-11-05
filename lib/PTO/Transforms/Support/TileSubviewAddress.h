// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_SUPPORT_TILESUBVIEWADDRESS_H
#define PTO_TRANSFORMS_SUPPORT_TILESUBVIEWADDRESS_H

#include "mlir/IR/Builders.h"

namespace mlir::pto {
class SubViewOp;
class TileBufType;

// Apply the window offset using the source tile's physical strides. The
// caller supplies the source address, including any earlier view offsets.
Value buildTileSubviewAddress(SubViewOp subview, Value base,
                              TileBufType parentType, OpBuilder &builder,
                              Location loc);
TileBufType getTileSubviewPhysicalType(SubViewOp subview, TileBufType parentType);
} // namespace mlir::pto

#endif
