// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOAddressSemantics.h - VPTO current-access contract ---*- C++ -*-===//

#ifndef PTO_IR_VPTOADDRESSSEMANTICS_H
#define PTO_IR_VPTOADDRESSSEMANTICS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::pto {

enum class VPTOAddressUnit {
  Element,
  Block,
  Byte,
  Alignment,
};

struct VPTOAddressOffset {
  Value value;
  VPTOAddressUnit unit = VPTOAddressUnit::Element;
};

struct VPTOAddressAccess {
  Value base;
  std::optional<VPTOAddressOffset> offset;
};

/// Default implementation used by operations carrying
/// VPTOAddressSemanticsOpInterface.  It reports only the address of the current
/// access; post-access advances are intentionally excluded.
SmallVector<VPTOAddressAccess>
getDefaultVPTOAddressAccesses(Operation *operation);

llvm::StringRef stringifyVPTOAddressUnit(VPTOAddressUnit unit);

} // namespace mlir::pto

#endif // PTO_IR_VPTOADDRESSSEMANTICS_H
