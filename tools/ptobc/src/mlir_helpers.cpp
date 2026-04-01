// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "ptobc/mlir_helpers.h"

#include <mlir/AsmParser/AsmParser.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

#include <llvm/Support/raw_ostream.h>

#include <stdexcept>

namespace ptobc {

std::string printType(mlir::Type t) {
  std::string s;
  llvm::raw_string_ostream os(s);
  t.print(os);
  os.flush();
  return s;
}

std::string printAttr(mlir::Attribute a) {
  std::string s;
  llvm::raw_string_ostream os(s);
  a.print(os);
  os.flush();
  return s;
}

std::string printAttrDict(mlir::DictionaryAttr a) {
  return printAttr(a);
}

mlir::Type parseType(mlir::MLIRContext& ctx, const std::string& s) {
  mlir::Type t = mlir::parseType(s, &ctx);
  if (!t) throw std::runtime_error("failed to parse type: " + s);
  return t;
}

mlir::Attribute parseAttr(mlir::MLIRContext& ctx, const std::string& s) {
  mlir::Attribute a = mlir::parseAttribute(s, &ctx);
  if (!a) throw std::runtime_error("failed to parse attr: " + s);
  return a;
}

mlir::DictionaryAttr parseAttrDict(mlir::MLIRContext& ctx, const std::string& s) {
  auto a = parseAttr(ctx, s);
  auto d = mlir::dyn_cast<mlir::DictionaryAttr>(a);
  if (!d) throw std::runtime_error("attr is not a dictionary: " + s);
  return d;
}

} // namespace ptobc
