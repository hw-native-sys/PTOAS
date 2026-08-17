// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_SOFTLIBSERVICE_H
#define MLIR_DIALECT_PTO_TRANSFORMS_SOFTLIBSERVICE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>

namespace mlir::pto {

struct SoftLibMaterializationRequest {
  std::string target;
  std::string op;
  std::string operandSpecsJson;
};

using SoftLibMaterializationCallback =
    llvm::function_ref<LogicalResult(ModuleOp source, StringRef entrySymbol)>;

class SoftLibService {
public:
  virtual ~SoftLibService() = default;

  virtual LogicalResult
  materialize(const SoftLibMaterializationRequest &request,
              MLIRContext &context,
              SoftLibMaterializationCallback callback) = 0;
};

class SoftLibRuntime {
public:
  static void install(std::shared_ptr<SoftLibService> service);
  static void uninstall(SoftLibService *service);
  static std::shared_ptr<SoftLibService> getService();
};

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_SOFTLIBSERVICE_H
