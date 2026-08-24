// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOAddressSemantics.h - VPTO addressing contract --------*- C++ -*-===//

#ifndef PTO_IR_VPTOADDRESSSEMANTICS_H
#define PTO_IR_VPTOADDRESSSEMANTICS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir::pto {

enum class VPTOAddressUnit {
  Element,
  Block,
  Byte,
  Alignment,
};

/// Target encoding restriction on a post-access pointer advance.
enum class VPTOAdvanceConstraint {
  Dynamic,
  Constant,
  SignedI8,
};

struct VPTOAddressOffset {
  OpOperand *operand;
  VPTOAddressUnit unit = VPTOAddressUnit::Element;
  /// Value whose element type defines an Element unit. This can differ from
  /// the base pointer for payload-denominated operations.
  Value elementTypeSource;
};

struct VPTOAddressAccess {
  OpOperand *baseOperand;
  std::optional<VPTOAddressOffset> offset;
};

/// Post-access advance semantics are separate from the current-access address.
/// A null advanceOperand denotes an optional trailing operand that is absent in
/// the normal form and is materialized when building the post-update form.
/// updatedBase is null for a normal-form operation and names the ODS result for
/// an operation that is already in post-update form.
struct VPTOPostUpdateSemantics {
  OpOperand *baseOperand;
  OpOperand *advanceOperand;
  VPTOAddressUnit advanceUnit = VPTOAddressUnit::Element;
  VPTOAdvanceConstraint constraint = VPTOAdvanceConstraint::Dynamic;
  Value updatedBase;
  /// Value whose element type defines an Element advance unit.
  Value elementTypeSource;
};

/// Complete VPTO addressing contract. Current accesses describe where an
/// operation reads or writes now. For an operation already in post-update form,
/// the current access is the base alone because its offset denotes only the
/// after-access advance. postUpdate independently describes that advance; its
/// unit may differ intentionally from a normal form's current offset unit.
struct VPTOAddressSemantics {
  SmallVector<VPTOAddressAccess> currentAccesses;
  std::optional<VPTOPostUpdateSemantics> postUpdate;
};

/// Default implementation used by VPTOAddressSemanticsOpInterface.
VPTOAddressSemantics getDefaultVPTOAddressSemantics(Operation *operation);

/// Resolve the byte width of one address unit from the operation's addressing
/// contract. For Element units, elementTypeSource may be a base pointer or a
/// vector payload value.
std::optional<int64_t>
getVPTOAddressUnitBytes(Operation *operation, VPTOAddressUnit unit,
                        Value elementTypeSource);

llvm::StringRef stringifyVPTOAddressUnit(VPTOAddressUnit unit);
llvm::StringRef stringifyVPTOAdvanceConstraint(VPTOAdvanceConstraint value);

} // namespace mlir::pto

#endif // PTO_IR_VPTOADDRESSSEMANTICS_H
