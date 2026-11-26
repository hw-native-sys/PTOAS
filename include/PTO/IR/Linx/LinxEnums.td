// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_LINX_ENUMS_TD
#define PTO_LINX_ENUMS_TD
include "mlir/IR/EnumAttr.td"

def LinxPTO_TileLocation : I32EnumAttr<"TileLocation", "Linx tile storage location", [
  I32EnumAttrCase<"Local", 0, "local">,
  I32EnumAttrCase<"Shared", 1, "shared">
]> {
  let cppNamespace = "::mlir::pto::linx";
  let genSpecializedAttr = 0;
}
def LinxPTO_TileLayout : I32EnumAttr<"TileLayout", "Linx tile layout", [
  I32EnumAttrCase<"RowMajor", 0, "row_major">,
  I32EnumAttrCase<"M16", 1, "m16">,
  I32EnumAttrCase<"M32", 2, "m32">,
  I32EnumAttrCase<"N8", 3, "n8">
]> {
  let cppNamespace = "::mlir::pto::linx";
  let genSpecializedAttr = 0;
}
#endif
