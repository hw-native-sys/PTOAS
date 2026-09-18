// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef VMILAYOUTSUPPORTSPINESCOPED_INC
#define VMILAYOUTSUPPORTSPINESCOPED_INC

//===- VMILayoutSupportSpineScoped.inc - spine-scoped cast queries -*- C++ -*-=//
// Textual include unit of VMILayoutSupport.cpp, included at the end of the
// translation unit so the definitions below can use the file-local cast
// helpers.  It holds the cast-layout queries the direction-spine peephole in
// VMILayoutAssignment reaches through getSpineScopedCastLayoutFact*; the rows
// they read are not part of the generic preferred/legal tables.
//===----------------------------------------------------------------------===//

FailureOr<VMICastLayoutFact>
VMILayoutSupport::getSpineScopedCastLayoutFact(VMIVRegType sourceType,
                                               VMIVRegType resultType,
                                               std::string *reason) const {
  FailureOr<VMICastLayoutFact> fact = getPreferredCastLayoutFactImpl(
      kSpineScopedCastLayoutPatterns,
      {sourceType, resultType, "preferred spine-scoped cast layout table",
       reason, VMICastLayoutPriority::High});
  if (failed(fact)) {
    return failure();
  }
  if (failed(getSpineScopedCastLayoutFactForLayouts(
          sourceType, resultType, fact->sourceLayout, fact->resultLayout,
          reason))) {
    return failure();
  }
  return fact;
}

FailureOr<VMICastLayoutFact>
VMILayoutSupport::getSpineScopedCastLayoutFactForLayouts(
    VMIVRegType sourceType, VMIVRegType resultType, VMILayoutAttr sourceLayout,
    VMILayoutAttr resultLayout, std::string *reason) const {
  FailureOr<SmallVector<VMICastLayoutFact, mlir::pto::kValue4>> facts =
      getSpineScopedCastLayoutFactsForLayout(
          sourceType, resultType, VMICastLayoutPort::Source, sourceLayout,
          reason);
  if (failed(facts)) {
    return failure();
  }
  return selectCastLayoutFactForResultLayout(
      *facts, resultLayout,
      "spine-scoped cast layout query produced ambiguous facts",
      "source/result layouts do not match a spine-scoped cast row", reason);
}

FailureOr<SmallVector<VMICastLayoutFact, mlir::pto::kValue4>>
VMILayoutSupport::getSpineScopedCastLayoutFactsForLayout(
    VMIVRegType sourceType, VMIVRegType resultType, VMICastLayoutPort port,
    VMILayoutAttr layout, std::string *reason) const {
  // The generic legal table stays the base relation set: the scoped query is a
  // strict superset so that a scoped op propagates exactly like an unscoped one
  // everywhere outside the two composite rows.
  std::pair<int64_t, int64_t> elementBits =
      getCastElementBits(sourceType, resultType);
  int64_t sourceBits = elementBits.first;
  int64_t resultBits = elementBits.second;
  int64_t numGroups =
      layout && layout.isGroupSlots() ? layout.getNumGroups() : 0;
  SmallVector<VMICastLayoutFact, mlir::pto::kValue4> facts;
  collectMatchingCastFacts(kLegalCastLayoutPatterns, sourceType, resultType,
                               port, layout, sourceBits, resultBits, numGroups,
                               facts);
  collectMatchingCastFacts(kSpineScopedLegalCastLayoutPatterns, sourceType,
                               resultType, port, layout, sourceBits,
                               resultBits, numGroups, facts);
  if (facts.empty()) {
    if (reason) {
      *reason = port == VMICastLayoutPort::Source
                    ? "requires a spine-scoped legal cast relation for the "
                      "source layout"
                    : "requires a spine-scoped legal cast relation for the "
                      "result layout";
    }
    return failure();
  }
  return facts;
}

#endif // VMILAYOUTSUPPORTSPINESCOPED_INC
