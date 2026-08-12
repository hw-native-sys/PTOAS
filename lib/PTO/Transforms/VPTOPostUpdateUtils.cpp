// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/VPTOPostUpdateUtils.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include <limits>

using namespace mlir;

namespace mlir::pto {

static constexpr int64_t kBlockSizeBytes = 32;

const PostUpdateOpTable &getPostUpdateOpTable() {
  static const PostUpdateOpTable table = [] {
    PostUpdateOpTable t;
    //                         base stride current-address unit/domain results
    t["pto.vlds"] = {0, 1, true, PostUpdateAddressUnit::Element, 1};
    t["pto.vldsx2"] = {0, 1, true, PostUpdateAddressUnit::Element, 2};
    t["pto.plds"] = {0, 1, true, PostUpdateAddressUnit::Byte, 1};
    t["pto.pldi"] = {0,
                     1,
                     true,
                     PostUpdateAddressUnit::Alignment,
                     1,
                     PostUpdateAddressDomain::Signed,
                     PostUpdateStrideConstraint::Constant};
    t["pto.vsts"] = {1, 2, true, PostUpdateAddressUnit::Element, 0};
    t["pto.psts"] = {1, 2, true, PostUpdateAddressUnit::Byte, 0};
    t["pto.psti"] = {1,
                     2,
                     true,
                     PostUpdateAddressUnit::Alignment,
                     0,
                     PostUpdateAddressDomain::Signed,
                     PostUpdateStrideConstraint::Constant};
    t["pto.sprsts"] = {0, 1, true, PostUpdateAddressUnit::Byte, 0};
    t["pto.sprsti"] = {0,
                       1,
                       true,
                       PostUpdateAddressUnit::Alignment,
                       0,
                       PostUpdateAddressDomain::Signed,
                       PostUpdateStrideConstraint::SignedI8};
    t["pto.vstas"] = {1, 2, true, PostUpdateAddressUnit::Element, 0};
    t["pto.vsldb"] = {0,    2,
                      true, PostUpdateAddressUnit::Block,
                      1,    PostUpdateAddressDomain::Unsigned};
    t["pto.vsstb"] = {1,    3,
                      true, PostUpdateAddressUnit::Block,
                      0,    PostUpdateAddressDomain::Unsigned};
    t["pto.vldus"] = {0, std::nullopt, false, PostUpdateAddressUnit::Element,
                      2};
    t["pto.vstus"] = {3, 1, false, PostUpdateAddressUnit::Element, 1};
    return t;
  }();
  return table;
}

const PostUpdateOpInfo *getPostUpdateOpInfo(Operation *op) {
  auto it = getPostUpdateOpTable().find(op->getName().getStringRef());
  return it == getPostUpdateOpTable().end() ? nullptr : &it->second;
}

std::optional<int64_t>
getCanonicalAddressRecurrenceStep(Value value, scf::ForOp forOp,
                                  PostUpdateAddressDomain domain) {
  auto type = dyn_cast<IntegerType>(value.getType());
  auto iterArg = dyn_cast<BlockArgument>(value);
  if (!type || type.getWidth() != 16 || !iterArg ||
      iterArg.getOwner() != forOp.getBody() || iterArg.getArgNumber() == 0)
    return std::nullopt;

  unsigned index = iterArg.getArgNumber() - 1;
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  Value yielded = yieldOp.getOperand(index);
  APInt bits;

  if (auto add = yielded.getDefiningOp<arith::AddIOp>()) {
    bool hasRequiredFlag = domain == PostUpdateAddressDomain::Signed
                               ? add.hasNoSignedWrap()
                               : add.hasNoUnsignedWrap();
    if (!hasRequiredFlag)
      return std::nullopt;
    Value step;
    if (add.getLhs() == value)
      step = add.getRhs();
    else if (add.getRhs() == value)
      step = add.getLhs();
    if (!step || !matchPattern(step, m_ConstantInt(&bits)) ||
        bits.getBitWidth() > 64)
      return std::nullopt;
    return domain == PostUpdateAddressDomain::Signed
               ? bits.getSExtValue()
               : static_cast<int64_t>(bits.getZExtValue());
  }

  if (auto sub = yielded.getDefiningOp<arith::SubIOp>()) {
    bool hasRequiredFlag = domain == PostUpdateAddressDomain::Signed
                               ? sub.hasNoSignedWrap()
                               : sub.hasNoUnsignedWrap();
    if (!hasRequiredFlag || sub.getLhs() != value ||
        !matchPattern(sub.getRhs(), m_ConstantInt(&bits)) ||
        bits.getBitWidth() > 64)
      return std::nullopt;
    uint64_t magnitude = domain == PostUpdateAddressDomain::Signed
                             ? static_cast<uint64_t>(bits.getSExtValue())
                             : bits.getZExtValue();
    if (magnitude > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return std::nullopt;
    return -static_cast<int64_t>(magnitude);
  }

  return std::nullopt;
}

std::optional<int64_t> getPostUpdateBaseUnitBytes(Value base) {
  Type elementType;
  if (auto ptrType = dyn_cast<PtrType>(base.getType()))
    elementType = ptrType.getElementType();
  else if (auto memrefType = dyn_cast<MemRefType>(base.getType()))
    elementType = memrefType.getElementType();
  else
    return std::nullopt;

  if (!elementType || !elementType.isIntOrFloat())
    return std::nullopt;
  unsigned bits = elementType.getIntOrFloatBitWidth();
  if (bits == 0 || bits % 8 != 0)
    return std::nullopt;
  return static_cast<int64_t>(bits / 8);
}

std::optional<int64_t> getPostUpdateAddressUnitBytes(Operation *op,
                                                     PostUpdateAddressUnit unit,
                                                     int64_t elementBytes) {
  switch (unit) {
  case PostUpdateAddressUnit::Element:
    return elementBytes;
  case PostUpdateAddressUnit::Block:
    return kBlockSizeBytes;
  case PostUpdateAddressUnit::Alignment:
    return getLoadStoreVecAlignmentSize(op);
  case PostUpdateAddressUnit::Byte:
    return 1;
  }
  llvm_unreachable("unhandled post-update address unit");
}

bool satisfiesPostUpdateStrideConstraint(
    PostUpdateStrideConstraint constraint,
    std::optional<int64_t> constantStride) {
  if (constraint == PostUpdateStrideConstraint::Dynamic)
    return true;
  if (!constantStride)
    return false;
  return constraint == PostUpdateStrideConstraint::Constant ||
         (*constantStride >= -128 && *constantStride <= 127);
}

} // namespace mlir::pto
