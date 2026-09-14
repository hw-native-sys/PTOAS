// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOCubeBridgeInternal.h - LoadCbufTo* verify templates ------------===//
//===----------------------------------------------------------------------===//
//
// Shared verification templates for the pto.load_cbuf_to_* cube bridge load
// instructions. Each instruction's verify() entry point lives in its own file
// (cbridge/VPTOLoadCbufTo*.cpp) and instantiates the regular/S4 verifier from
// here. This header is internal to lib/PTO/IR/VPTO/cbridge and not installed.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_IR_VPTO_CBRIDGE_INTERNAL_H
#define PTO_IR_VPTO_CBRIDGE_INTERNAL_H

#include "VPTOInternal.h"

template <typename OpTy>
static mlir::LogicalResult verifyExplicitCubeBridgeLoadControls(OpTy op) {
  constexpr int64_t kU16Max = 65535;
  constexpr int64_t kU8Max = 255;
  mlir::Operation *operation = op.getOperation();
  if (mlir::failed(verifyStaticControlRange(operation, op.getMStart(), "m_start", 0,
                                            kU16Max)) ||
      mlir::failed(verifyStaticControlRange(operation, op.getKStart(), "k_start", 0,
                                            kU16Max)) ||
      mlir::failed(verifyStaticControlRange(operation, op.getMStep(), "m_step", 1,
                                            kU8Max)) ||
      mlir::failed(verifyStaticControlRange(operation, op.getKStep(), "k_step", 1,
                                            kU8Max)) ||
      mlir::failed(verifyStaticControlRange(operation, op.getSrcStride(),
                                            "src_stride", 1, kU16Max)) ||
      mlir::failed(verifyStaticControlRange(operation, op.getDstStride(),
                                            "dst_stride", 1, kU16Max))) {
    return mlir::failure();
  }
  return mlir::success();
}

template <typename OpTy>
static mlir::LogicalResult
verifyS4CubeBridgeLoad(OpTy op, mlir::pto::AddressSpace expectedDstSpace,
                       llvm::StringRef dstName) {
  if (mlir::failed(verifyCubeBridgeLoadLikeOp(op, expectedDstSpace, dstName))) {
    return mlir::failure();
  }

  mlir::Type sourceElem = getBufferElementType(op.getSource().getType());
  mlir::Type destinationElem = getBufferElementType(op.getDestination().getType());
  if (!mlir::pto::isPTOFloat4PackedType(sourceElem)) {
    return op.emitOpError(
        "requires packed FP4 source element type f4e1m2x2 or f4e2m1x2");
  }
  if (!mlir::pto::isPTOFloat4PackedType(destinationElem)) {
    return op.emitOpError(
        "requires packed FP4 destination element type f4e1m2x2 or f4e2m1x2");
  }
  if (sourceElem != destinationElem) {
    return op.emitOpError(
        "requires source and destination packed FP4 element types to match");
  }
  if (mlir::failed(verifyExplicitCubeBridgeLoadControls(op))) {
    return mlir::failure();
  }
  return verifyStaticControlRange(op.getOperation(), op.getTranspose(),
                                  "transpose", 0, 1);
}

template <typename OpTy>
static mlir::LogicalResult
verifyRegularCubeBridgeLoad(OpTy op, mlir::pto::AddressSpace expectedDstSpace,
                            llvm::StringRef dstName) {
  if (mlir::failed(verifyCubeBridgeLoadLikeOp(op, expectedDstSpace, dstName))) {
    return mlir::failure();
  }
  mlir::Type sourceElem = getBufferElementType(op.getSource().getType());
  mlir::Type destinationElem = getBufferElementType(op.getDestination().getType());
  if (mlir::pto::isPTOFloat4PackedType(sourceElem)) {
    return op.emitOpError("packed FP4 source requires the S4 load operation");
  }
  if (mlir::pto::isPTOFloat4PackedType(destinationElem)) {
    return op.emitOpError("packed FP4 destination requires the S4 load operation");
  }
  return verifyExplicitCubeBridgeLoadControls(op);
}

#endif // PTO_IR_VPTO_CBRIDGE_INTERNAL_H
