// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOAttrInterfaces.td - PTO attribute interfaces ------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// This is the definition file for PTO dialect attribute interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTOATTRINTERFACES
#define MLIR_DIALECT_PTO_IR_PTOATTRINTERFACES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Enum token reflection
//===----------------------------------------------------------------------===//

// Implemented by PTO enum attributes that can feed a VPTO bridge template
// token. This MLIR build has no generic C++ runtime base for TableGen enum
// attributes, so the declarative bridge lowering reflects the enum case
// through this interface instead of knowing each attribute class: it
// assembles `EnumType::Case` spellings from the whitelist `enum_type`
// prefix and the enumerator symbol exposed here.
def PTO_EnumTokenAttr : AttrInterface<"EnumTokenAttr"> {
  let cppNamespace = "::mlir::pto";
  let description = [{
    An attribute whose value is one case of a PTO enum and can be rendered
    as a C++ template token by the declarative VPTO bridge lowering.
  }];
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the C++ enumerator symbol of the attribute's enum case
        (e.g. "Partial" for `pto::AccPhase::Partial`).
      }],
      /*retTy=*/"llvm::StringRef",
      /*methodName=*/"getEnumCaseSymbol",
      /*args=*/(ins)
    >,
  ];
}

#endif // MLIR_DIALECT_PTO_IR_PTOATTRINTERFACES
