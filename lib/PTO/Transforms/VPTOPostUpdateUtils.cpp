// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/VPTOPostUpdateUtils.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include <limits>

using namespace mlir;

namespace mlir::pto {

static constexpr int64_t kBlockSizeBytes = 32;

const PostUpdateOpTable &getPostUpdateOpTable() {
  static const PostUpdateOpTable table = [] {
    PostUpdateOpTable t;
    //                         base stride current-address unit/domain results
    t["pto.vlds"] = {0,
                      1,
                      true,
                      PostUpdateAddressUnit::Element,
                      1,
                      PostUpdateAddressDomain::Signed,
                      PostUpdateStrideConstraint::Dynamic,
                      PostUpdateElementTypeSource::Result,
                      0};
    t["pto.vldsx2"] = {0,
                        1,
                        true,
                        PostUpdateAddressUnit::Element,
                        2,
                        PostUpdateAddressDomain::Signed,
                        PostUpdateStrideConstraint::Dynamic,
                        PostUpdateElementTypeSource::Result,
                        0};
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
    t["pto.vldus"] = {0,
                       std::nullopt,
                       false,
                       PostUpdateAddressUnit::Element,
                       2,
                       PostUpdateAddressDomain::Signed,
                       PostUpdateStrideConstraint::Dynamic,
                       PostUpdateElementTypeSource::Result,
                       0};
    t["pto.vstus"] = {3,
                       1,
                       false,
                       PostUpdateAddressUnit::Element,
                       1,
                       PostUpdateAddressDomain::Signed,
                       PostUpdateStrideConstraint::Dynamic,
                       PostUpdateElementTypeSource::Operand,
                       2};
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

scf::ForOp pruneDeadLoopCarriedValues(scf::ForOp forOp,
                                      OpBuilder &builder) {
  unsigned numIterArgs = forOp.getInitArgs().size();
  if (numIterArgs == 0) {
    return forOp;
  }

  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  DenseSet<Value> liveValues;
  SmallVector<Value> worklist;
  SmallVector<bool> keepIterArg(numIterArgs, false);

  auto markLive = [&](Value value) {
    if (value && liveValues.insert(value).second) {
      worklist.push_back(value);
    }
  };

  for (auto [idx, result] : llvm::enumerate(forOp.getResults())) {
    if (result.use_empty()) {
      continue;
    }
    keepIterArg[idx] = true;
    markLive(yieldOp.getOperand(idx));
  }

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!isPure(&op)) {
      for (Value operand : op.getOperands()) {
        markLive(operand);
      }
    }
    bool hasRegions = op.getNumRegions() != 0;
    if (hasRegions) {
      op.walk([&](Operation *nested) {
        for (Value operand : nested->getOperands()) {
          markLive(operand);
        }
      });
    }
  }

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      bool belongsToAnotherRegion = blockArg.getOwner() != forOp.getBody();
      bool isInductionVariable = blockArg.getArgNumber() == 0;
      if (belongsToAnotherRegion || isInductionVariable) {
        continue;
      }
      unsigned idx = blockArg.getArgNumber() - 1;
      if (!keepIterArg[idx]) {
        keepIterArg[idx] = true;
        markLive(yieldOp.getOperand(idx));
      }
      continue;
    }

    Operation *defOp = value.getDefiningOp();
    if (!defOp || !forOp->isAncestor(defOp)) {
      continue;
    }
    for (Value operand : defOp->getOperands()) {
      markLive(operand);
    }
  }

  if (llvm::all_of(keepIterArg, [](bool keep) { return keep; })) {
    return forOp;
  }

  SmallVector<Value> newInitArgs;
  newInitArgs.reserve(numIterArgs);
  for (auto [idx, init] : llvm::enumerate(forOp.getInitArgs())) {
    if (keepIterArg[idx]) {
      newInitArgs.push_back(init);
    }
  }

  builder.setInsertionPoint(forOp);
  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newInitArgs);
  newForOp->setAttrs(forOp->getAttrs());
  bool hasPlaceholderYield = !newForOp.getBody()->empty() &&
                             isa<scf::YieldOp>(newForOp.getBody()->back());
  if (hasPlaceholderYield) {
    newForOp.getBody()->back().erase();
  }

  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  unsigned newArgIdx = 0;
  for (auto [idx, oldArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
    if (keepIterArg[idx]) {
      mapping.map(oldArg, newForOp.getRegionIterArgs()[newArgIdx++]);
    }
  }

  builder.setInsertionPointToStart(newForOp.getBody());
  for (Operation &op : forOp.getBody()->without_terminator()) {
    bool hasLiveResult = llvm::any_of(op.getResults(), [&](Value result) {
      return liveValues.contains(result);
    });
    bool mustClone = !isPure(&op) || op.getNumRegions() != 0 || hasLiveResult;
    if (mustClone) {
      builder.clone(op, mapping);
    }
  }

  SmallVector<Value> newYields;
  newYields.reserve(newInitArgs.size());
  for (auto [idx, yielded] : llvm::enumerate(yieldOp.getOperands())) {
    if (keepIterArg[idx]) {
      newYields.push_back(mapping.lookupOrDefault(yielded));
    }
  }
  builder.setInsertionPointToEnd(newForOp.getBody());
  builder.create<scf::YieldOp>(yieldOp.getLoc(), newYields);

  unsigned newResultIdx = 0;
  for (auto [idx, oldResult] : llvm::enumerate(forOp.getResults())) {
    if (keepIterArg[idx]) {
      oldResult.replaceAllUsesWith(newForOp.getResult(newResultIdx++));
      continue;
    }
    if (!oldResult.use_empty()) {
      llvm_unreachable("cannot drop a used scf.for result");
    }
  }

  forOp.erase();
  return newForOp;
}

std::optional<int64_t> getPostUpdateBaseUnitBytes(Value base) {
  Type elementType;
  if (auto ptrType = dyn_cast<PtrType>(base.getType())) {
    elementType = ptrType.getElementType();
  } else if (auto memrefType = dyn_cast<MemRefType>(base.getType())) {
    elementType = memrefType.getElementType();
  } else {
    return std::nullopt;
  }

  unsigned bytes = getPTOStorageElemByteSize(elementType);
  return bytes == 0 ? std::nullopt
                    : std::optional<int64_t>(static_cast<int64_t>(bytes));
}

static std::optional<int64_t> getVectorElementBytes(Type type) {
  auto vectorType = dyn_cast<VRegType>(type);
  if (!vectorType) {
    return std::nullopt;
  }
  unsigned bytes = getPTOStorageElemByteSize(vectorType.getElementType());
  return bytes == 0 ? std::nullopt
                    : std::optional<int64_t>(static_cast<int64_t>(bytes));
}

std::optional<int64_t>
getPostUpdateAddressUnitBytes(Operation *op, const PostUpdateOpInfo &info,
                              int64_t baseElementBytes) {
  switch (info.addressUnit) {
  case PostUpdateAddressUnit::Element:
    switch (info.elementTypeSource) {
    case PostUpdateElementTypeSource::Base:
      return baseElementBytes;
    case PostUpdateElementTypeSource::Operand:
      if (info.elementTypeIndex >= op->getNumOperands()) {
        return std::nullopt;
      }
      return getVectorElementBytes(
          op->getOperand(info.elementTypeIndex).getType());
    case PostUpdateElementTypeSource::Result:
      if (info.elementTypeIndex >= op->getNumResults()) {
        return std::nullopt;
      }
      return getVectorElementBytes(
          op->getResult(info.elementTypeIndex).getType());
    }
    llvm_unreachable("unhandled post-update element type source");
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
