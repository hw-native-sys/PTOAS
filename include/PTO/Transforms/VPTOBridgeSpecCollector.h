// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.huawei.com/
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOBridgeSpecCollector.h - bridge spec collection -------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// Per-function collection of the VPTO bridge specialization fields that the
// wrapper generation pass merges into the module spec. Both the declarative
// channel and the pipe family pass write through this collector so the
// same-key policies and the kBridgeFuncSpecAttrName attribute shape live in
// one place.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOBRIDGESPECCOLLECTOR_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOBRIDGESPECCOLLECTOR_H

#include "PTO/Transforms/VPTOBridgeTokens.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <utility>

namespace mlir {
namespace pto {

/// Collects the per-function bridge specialization fields written as the
/// kBridgeFuncSpecAttrName function attribute.
class BridgeSpecCollector {
public:
  /// Adds a spec field with same-value dedup: several ops sharing a tile
  /// shape or a wrapper entry write the same token under one key, which is
  /// harmless; a different token for a key already written is diagnosed as
  /// a conflict on `op` (the wrapper renders one token per spec field).
  void addField(Operation *op, llvm::StringRef key, llvm::StringRef token);

  /// Adds a spec field that may be written at most once per function (the
  /// pipe family's single producer/consumer pair); any repeat is diagnosed
  /// on `op`.
  void addUniqueField(Operation *op, llvm::StringRef key,
                      llvm::StringRef token);

  /// Returns whether any conflict was diagnosed.
  bool hadError() const { return hadError_; }

  /// Stores the collected fields as the kBridgeFuncSpecAttrName function
  /// attribute. No-op when nothing was collected.
  void store(func::FuncOp func) const;

private:
  llvm::SmallVector<std::pair<std::string, std::string>> fields;
  llvm::StringMap<std::string> written;
  bool hadError_ = false;
};

inline void BridgeSpecCollector::addField(Operation *op, llvm::StringRef key,
                                          llvm::StringRef token) {
  auto inserted = written.try_emplace(key, token);
  if (inserted.second) {
    fields.emplace_back(key.str(), token.str());
    return;
  }
  if (inserted.first->second != token) {
    op->emitError() << "VPTO bridge spec field '" << key
                    << "' was already collected as '"
                    << inserted.first->second
                    << "'; the wrapper renders one token per spec field";
    hadError_ = true;
  }
}

inline void BridgeSpecCollector::addUniqueField(Operation *op,
                                                llvm::StringRef key,
                                                llvm::StringRef token) {
  if (written.count(key)) {
    op->emitError()
        << "VPTO bridge spec field '" << key
        << "' was already collected; only one bridged producer/consumer pair "
           "per function is supported";
    hadError_ = true;
    return;
  }
  written.try_emplace(key, token);
  fields.emplace_back(key.str(), token.str());
}

inline void BridgeSpecCollector::store(func::FuncOp func) const {
  if (fields.empty()) {
    return;
  }
  SmallVector<NamedAttribute> specAttrs;
  specAttrs.reserve(fields.size());
  for (const auto &field : fields) {
    specAttrs.push_back({StringAttr::get(func.getContext(), field.first),
                         StringAttr::get(func.getContext(), field.second)});
  }
  func->setAttr(kBridgeFuncSpecAttrName,
                DictionaryAttr::get(func.getContext(), specAttrs));
}

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOBRIDGESPECCOLLECTOR_H
