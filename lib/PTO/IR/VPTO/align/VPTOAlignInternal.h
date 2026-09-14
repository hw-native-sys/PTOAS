// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOAlignInternal.h - per-instruction align-chain hooks ------------===//
//===----------------------------------------------------------------------===//
//
// Per-instruction policy hooks for the !pto.align chain verification. Each
// hook is defined in align/VPTO<Instruction>.cpp and stitched into the shared
// store/load chain policies by the driver in align/VPTOAlign.cpp. A hook
// returns false/nullopt when its instruction does not match, so the
// composition order is irrelevant.
//
// This header is internal to lib/PTO/IR/VPTO/align and not installed.

#ifndef PTO_IR_VPTO_ALIGN_INTERNAL_H
#define PTO_IR_VPTO_ALIGN_INTERNAL_H

#include "VPTOInternal.h"

// pto.init_align: accepted store-chain root (backward walk sinks here).
bool isInitAlignStoreRoot(mlir::Operation *def);

// pto.pstu: store-state advance (align_in / align_out) and accepted root.
bool isPstuStoreRoot(mlir::Operation *def);
std::optional<mlir::Value> pstuStoreStateIn(mlir::Operation *def);
std::optional<mlir::Value> pstuStoreStateOut(mlir::Operation *def);

// pto.vstus: store-state advance (align_in / align_out) and accepted root.
bool isVstusStoreRoot(mlir::Operation *def);
std::optional<mlir::Value> vstusStoreStateIn(mlir::Operation *def);
std::optional<mlir::Value> vstusStoreStateOut(mlir::Operation *def);

// pto.vstur: store-state advance (align_in / align_out) and accepted root.
bool isVsturStoreRoot(mlir::Operation *def);
std::optional<mlir::Value> vsturStoreStateIn(mlir::Operation *def);
std::optional<mlir::Value> vsturStoreStateOut(mlir::Operation *def);

// pto.vstas: terminal store-chain sink (consumes without advancing).
bool isVstasStoreSink(mlir::Operation *def);

// pto.vstar: terminal store-chain sink (consumes without advancing).
bool isVstarStoreSink(mlir::Operation *def);

// pto.vldas: accepted load-chain root (backward walk sinks here).
bool isVldasLoadRoot(mlir::Operation *def);

// pto.vldus: load-state advance (align / updated_align) and accepted root.
bool isVldusLoadRoot(mlir::Operation *def);
std::optional<mlir::Value> vldusLoadStateIn(mlir::Operation *def);
std::optional<mlir::Value> vldusLoadStateOut(mlir::Operation *def);

#endif // PTO_IR_VPTO_ALIGN_INTERNAL_H
