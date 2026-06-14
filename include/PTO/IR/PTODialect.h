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

#ifndef MLIR_DIALECT_PTO_IR_PTODIALECT_H
#define MLIR_DIALECT_PTO_IR_PTODIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace pto {

class PTODialect : public ::mlir::Dialect {
  explicit PTODialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;

public:
  ~PTODialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("pto");
  }

  ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                                   ::mlir::Type type) const override;

  void printAttribute(::mlir::Attribute attr,
                      ::mlir::DialectAsmPrinter &printer) const override;

  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

  void printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &printer) const override;
};

} // namespace pto
} // namespace mlir

MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::pto::PTODialect)

#endif // MLIR_DIALECT_PTO_IR_PTODIALECT_H
