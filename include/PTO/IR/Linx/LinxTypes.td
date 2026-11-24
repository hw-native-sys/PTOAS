// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_LINX_TYPES_TD
#define PTO_LINX_TYPES_TD
include "PTO/IR/Linx/LinxDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "PTO/IR/Linx/LinxEnums.td"
class LinxPTO_ShapedType<string name, string spelling> : TypeDef<PTO_Dialect, name> {
  let mnemonic = spelling;
  let parameters = (ins "int64_t":$rows, "int64_t":$cols, "::mlir::Type":$elementType);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}
def LinxPTO_TileBuf : LinxPTO_ShapedType<"TileBuf", "tile_buf"> {
  let summary = "Linx static tile buffer handle";
  let parameters = (ins EnumParameter<LinxPTO_TileLocation>:$location,
                        EnumParameter<LinxPTO_TileLayout>:$layout,
                        "int64_t":$rows, "int64_t":$cols, "::mlir::Type":$elementType);
}
def LinxPTO_GMView : LinxPTO_ShapedType<"GMView", "partition_tensor_view"> {
  let summary = "Linx static two-dimensional external memory view";
}
#endif
