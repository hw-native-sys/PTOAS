// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOVerifyCore.cpp; kept as a fragment included by PTOVerifyCore.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

LogicalResult mlir::pto::SyncWaitOp::verify() {
  bool hasStatic = getEventIdAttr() != nullptr;
  bool hasDynamic = static_cast<bool>(getEventIdDyn());
  if (hasStatic == hasDynamic)
    return emitOpError()
           << "expects exactly one event-id form: static attr or dynamic index operand";

  auto verifyA2A3 = []() -> LogicalResult { return success(); };
  auto verifyA5 = [this]() -> LogicalResult {
    switch (getPipe().getPipe()) {
    case PIPE::PIPE_FIX:
    case PIPE::PIPE_MTE1:
    case PIPE::PIPE_MTE2:
    case PIPE::PIPE_MTE3:
    case PIPE::PIPE_V:
      return success();
    default:
      return emitOpError() << "A5 sync.wait expects pipe to be one of "
                              "<PIPE_FIX>, <PIPE_MTE1>, <PIPE_MTE2>, "
                              "<PIPE_MTE3>, <PIPE_V>";
    }
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TStoreOp::verify() {
  bool hasPreQuant = static_cast<bool>(getPreQuantScalar());
  auto reluMode = getReluPreMode();

  auto verifyA2A3 = [this, hasPreQuant, reluMode]() -> LogicalResult {
    auto common =
        verifyTStoreCommon(*this, getSrc(), getDst(), /*allowLowPrecision=*/false);
    if (failed(common))
      return failure();
    return verifyTStoreA2A3(*this, *common, hasPreQuant, reluMode);
  };

  auto verifyA5 = [this, hasPreQuant, reluMode]() -> LogicalResult {
    auto common =
        verifyTStoreCommon(*this, getSrc(), getDst(), /*allowLowPrecision=*/true);
    if (failed(common))
      return failure();
    return verifyTStoreA5(*this, *common, hasPreQuant, reluMode);
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
