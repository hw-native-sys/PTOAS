// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOTypeDefs.td ---------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_VPTOTYPEDEFS
#define MLIR_DIALECT_PTO_IR_VPTOTYPEDEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def VRegType : TypeDef<PTO_Dialect, "VReg"> {
  let mnemonic = "vreg";
  let summary = "A 256-byte PTO low-level vector";

  let parameters = (ins
    "int64_t":$elementCount,
    "Type":$elementType
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def MaskType : TypeDef<PTO_Dialect, "Mask"> {
  let mnemonic = "mask";
  let summary = "A PTO low-level predicate/mask register";

  let parameters = (ins
    StringRefParameter<"mask granularity view">:$granularity
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    static bool isSupportedGranularity(::llvm::StringRef granularity);

    bool isB8() const { return getGranularity() == "b8"; }
    bool isB16() const { return getGranularity() == "b16"; }
    bool isB32() const { return getGranularity() == "b32"; }
  }];
}

def VAddrType : TypeDef<PTO_Dialect, "VAddr"> {
  let mnemonic = "vaddr";
  let summary = "A PTO low-level vector_address offset token";

  let parameters = (ins
    StringRefParameter<"address granularity view">:$granularity
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    static bool isSupportedGranularity(::llvm::StringRef granularity);

    bool isB8() const { return getGranularity() == "b8"; }
    bool isB16() const { return getGranularity() == "b16"; }
    bool isB32() const { return getGranularity() == "b32"; }
  }];
}

def AlignType : TypeDef<PTO_Dialect, "Align"> {
  let mnemonic = "align";
  let summary = "A PTO low-level vector_align carrier";
}

#endif // MLIR_DIALECT_PTO_IR_VPTOTYPEDEFS
