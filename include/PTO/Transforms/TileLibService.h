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

struct TileLibMaterialization {
  OwningOpRef<ModuleOp> module;
  std::string entrySymbol;
};

/// C++ ownership boundary for a TileLib implementation. Implementations return
/// a C++-owned source ModuleOp in the requested context. The caller may then
/// clone/import its generated functions into the caller module.
class TileLibService {
public:
  virtual ~TileLibService() = default;

  virtual FailureOr<std::string>
  getMetadata(const TileLibMaterializationRequest &request) = 0;

  virtual FailureOr<TileLibMaterialization>
  materialize(const TileLibMaterializationRequest &request,
              MLIRContext &context) = 0;
};

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_TILELIBSERVICE_H
