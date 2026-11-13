// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef VMILAYOUTSUPPORTBITCAST_INC
#define VMILAYOUTSUPPORTBITCAST_INC

//===- VMILayoutSupportBitcast.inc - width-changing bitcast queries -*- C++ -*-=//
// Textual include unit of VMILayoutSupport.cpp, included at the end of the
// translation unit.  It holds the width-changing bitcast rule table and the
// queries that read it: the relation is a single contiguous row whose only
// consumer is getBitcastLayoutFact*, so the table belongs next to that query
// rather than in the shared tables.
//===----------------------------------------------------------------------===//

namespace {

struct WidthChangingBitcastLayoutPattern {
  LayoutPattern layout;
};

constexpr WidthChangingBitcastLayoutPattern
    kWidthChangingBitcastLayoutPatterns[] = {
    {c()},
};

} // namespace

// A width-changing bitcast keeps the register footprint and reinterprets the
// element type, so its only legal relation is the contiguous row the table
// above holds.
FailureOr<VMIBitcastLayoutFact>
VMILayoutSupport::getBitcastLayoutFact(VMIBitcastOp op,
                                       std::string *reason) const {
  auto fail =
      [reason](const Twine &message) -> FailureOr<VMIBitcastLayoutFact> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !resultLayout) {
    return fail("requires assigned source and result layouts");
  }
  if (sourceLayout != resultLayout) {
    return fail("requires matching source and result layouts");
  }

  int64_t numGroups =
      sourceLayout.isGroupSlots() ? sourceLayout.getNumGroups() : 0;
  unsigned sourceElementBits =
      pto::getPTOStorageElemBitWidth(sourceType.getElementType());
  unsigned resultElementBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (sourceElementBits == 0 || resultElementBits == 0) {
    return fail("requires source and result with known storage element width");
  }
  // Equal-width bitcast is layout-transparent for any identical layout.  Only
  // width-changing bitcast needs a table row because not every layout has a
  // representation-preserving carrier reinterpretation across element widths.
  if (sourceElementBits != resultElementBits) {
    bool matchedLayout = false;
    MLIRContext *ctx = op.getContext();
    for (const WidthChangingBitcastLayoutPattern &pattern :
         kWidthChangingBitcastLayoutPatterns) {
      if (matchesLayoutPattern(ctx, pattern.layout, sourceLayout, numGroups) &&
          matchesLayoutPattern(ctx, pattern.layout, resultLayout, numGroups)) {
        matchedLayout = true;
        break;
      }
    }
    if (!matchedLayout) {
      return fail("width-changing bitcast layout does not match a bitcast "
                  "layout table row");
    }
  }

  return VMIBitcastLayoutFact{sourceLayout, resultLayout};
}

FailureOr<SmallVector<VMIBitcastLayoutFact, mlir::pto::kValue4>>
VMILayoutSupport::getBitcastLayoutFactsForLayout(
    VMIVRegType sourceType, VMIVRegType resultType, VMICastLayoutPort port,
    VMILayoutAttr layout, std::string *reason) const {
  auto fail = [reason](const Twine &message)
      -> FailureOr<SmallVector<VMIBitcastLayoutFact, 4>> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  if (!layout) {
    return fail("requires an assigned bitcast query layout");
  }

  unsigned sourceElementBits =
      pto::getPTOStorageElemBitWidth(sourceType.getElementType());
  unsigned resultElementBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (sourceElementBits == 0 || resultElementBits == 0) {
    return fail("requires source and result with known storage element width");
  }

  if (sourceElementBits == resultElementBits) {
    return SmallVector<VMIBitcastLayoutFact, mlir::pto::kValue4>{
        VMIBitcastLayoutFact{layout, layout}};
  }

  int64_t numGroups = layout.isGroupSlots() ? layout.getNumGroups() : 0;
  MLIRContext *ctx = sourceType.getContext();
  SmallVector<VMIBitcastLayoutFact, mlir::pto::kValue4> facts;
  for (const WidthChangingBitcastLayoutPattern &pattern :
       kWidthChangingBitcastLayoutPatterns) {
    VMILayoutAttr candidate =
        materializeLayoutPattern(ctx, pattern.layout, numGroups);
    if (!candidate) {
      continue;
    }
    if (port == VMICastLayoutPort::Source && candidate != layout) {
      continue;
    }
    if (port == VMICastLayoutPort::Result && candidate != layout) {
      continue;
    }
    facts.push_back(VMIBitcastLayoutFact{candidate, candidate});
  }

  if (facts.empty()) {
    if (port == VMICastLayoutPort::Source) {
      return fail(
          "requires a legal width-changing bitcast source layout relation");
    }
    return fail(
        "requires a legal width-changing bitcast result layout relation");
  }
  return facts;
}

LogicalResult VMILayoutSupport::getBitcastSupport(VMIBitcastOp op,
                                                  std::string *reason) const {
  return getBitcastLayoutFact(op, reason);
}

#endif // VMILAYOUTSUPPORTBITCAST_INC
