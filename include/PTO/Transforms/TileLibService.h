// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_TILELIBSERVICE_H
#define MLIR_DIALECT_PTO_TRANSFORMS_TILELIBSERVICE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>

namespace mlir::pto {

/// Pure-data request used by the in-process TileLib materializer. The JSON
/// fields are request data only; generated MLIR never crosses this interface as
/// text. Keeping this interface independent of pybind11 allows transform passes
/// to remain usable from native tests and non-Python hosts.
struct TileLibMaterializationRequest {
  std::string target;
  std::string op;
  std::string operandSpecsJson;
  std::string contextAttrsJson;
  std::string candidateId;
};

using TileLibMaterializationCallback =
    llvm::function_ref<LogicalResult(ModuleOp source, StringRef entrySymbol)>;

/// Synchronous handoff for a materialized TileLib implementation. The source
/// module is borrowed and remains owned by the service for the duration of the
/// callback. Consumers must clone/import any operations they need before the
/// callback returns.
class TileLibService {
public:
  virtual ~TileLibService() = default;

  virtual FailureOr<std::string>
  getMetadata(const TileLibMaterializationRequest &request) = 0;

  virtual LogicalResult
  materialize(const TileLibMaterializationRequest &request,
              MLIRContext &context,
              TileLibMaterializationCallback callback) = 0;
};

/// Process-wide access to the host TileLib implementation. The runtime owns no
/// compilation context: passes obtain the current MLIRContext from their
/// operation and pass it to TileLibService::materialize. A host binding installs
/// one service implementation for the lifetime of that runtime.
class TileLibRuntime {
public:
  static void install(std::shared_ptr<TileLibService> service);
  static void uninstall(TileLibService *service);
  static std::shared_ptr<TileLibService> getService();
};

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_TILELIBSERVICE_H
