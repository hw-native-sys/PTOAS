// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOTile.h - PTO tile operation declarations -----------*- C++ -*-===//

#ifndef MLIR_DIALECT_PTO_IR_PTOTILE_H_
#define MLIR_DIALECT_PTO_IR_PTOTILE_H_

#include "PTO/IR/PTOCommon.h"
#include "PTO/IR/PTOTileMemory.h"
#include "PTO/IR/PTOTileCube.h"
#include "PTO/IR/PTOTilePipeline.h"
#include "PTO/IR/PTOTileVectorData.h"
#include "PTO/IR/PTOTileVectorElementwise.h"
#include "PTO/IR/PTOTileVectorReduction.h"

#define GET_OP_CLASSES
#include "PTO/IR/PTOTileMemoryOps.h.inc"
#include "PTO/IR/PTOTileCubeOps.h.inc"
#include "PTO/IR/PTOTilePipelineOps.h.inc"
#include "PTO/IR/PTOTileVectorDataOps.h.inc"
#include "PTO/IR/PTOTileVectorElementwiseOps.h.inc"
#include "PTO/IR/PTOTileVectorReductionOps.h.inc"

#endif // MLIR_DIALECT_PTO_IR_PTOTILE_H_
