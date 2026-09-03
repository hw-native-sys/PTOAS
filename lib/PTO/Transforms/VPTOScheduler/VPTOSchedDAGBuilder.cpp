// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedDAGBuilder.cpp - VPTO scheduling DAG builder -------------===//

#include "PTO/Transforms/VPTOScheduler/VPTOSchedDAGBuilder.h"

#include "PTO/Analysis/PTOAddressAnalysis.h"
#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MathExtras.h"

#include <array>
#include <limits>

#include "../Utils.h"

using namespace mlir;
using namespace mlir::pto;

FailureOr<std::unique_ptr<VPTOSchedDAG>>
VPTOSchedDAGBuilder::build(const VPTOSchedRegion &region) const {
  VPTOScheduleFailure failure;
  return build(region, failure);
}

FailureOr<std::unique_ptr<VPTOSchedDAG>> VPTOSchedDAGBuilder::build(
    const VPTOSchedRegion &region, VPTOScheduleFailure &failure) const {
  uint64_t nodeCount = region.operations.size();
  if (limits && nodeCount > limits->maxNodes) {
    failure = {VPTOScheduleFailureKind::Budget, "nodes", nodeCount,
               limits->maxNodes, {}};
    return mlir::failure();
  }

  auto dag = std::make_unique<VPTOSchedDAG>(region);
  if (failed(buildSSAEdges(*dag, failure))) {
    return mlir::failure();
  }
  if (failed(buildMemoryEdges(*dag, failure))) {
    return mlir::failure();
  }
  if (failed(buildImplicitAndSyncEdges(*dag, failure))) {
    return mlir::failure();
  }
  if (failed(buildModelFallbackEdges(*dag, failure))) {
    return mlir::failure();
  }
  if (failed(consumeWork(failure, dag->getUnits().size()))) {
    return mlir::failure();
  }
  if (failed(consumeWork(failure, dag->getEdges().size()))) {
    return mlir::failure();
  }
  if (failed(dag->computeCriticalPaths())) {
    failure.kind = VPTOScheduleFailureKind::Scheduling;
    failure.detail = "dependency DAG contains a Must-edge cycle";
    return mlir::failure();
  }
  dag->resetDependencyCounts();
  return std::move(dag);
}

LogicalResult
VPTOSchedDAGBuilder::consumeWork(VPTOScheduleFailure &failure,
                                 uint64_t amount) const {
  if (!budget || budget->consume(amount)) {
    return success();
  }
  uint64_t limit = budget->getLimit();
  uint64_t actual = limit == std::numeric_limits<uint64_t>::max()
                        ? limit
                        : limit + 1;
  failure = {VPTOScheduleFailureKind::Budget, "work-units", actual, limit, {}};
  return mlir::failure();
}

LogicalResult VPTOSchedDAGBuilder::addEdge(
    VPTOSchedDAG &dag, VPTOSUnit &predecessor, VPTOSUnit &successor,
    VPTOSchedEdgeKind kind, VPTOSchedEdgeStrength strength, unsigned latency,
    Twine reason, VPTOScheduleFailure &failure) const {
  uint64_t edgeCount = dag.getEdges().size();
  if (limits && edgeCount >= limits->maxEdges) {
    uint64_t actual = edgeCount == std::numeric_limits<uint64_t>::max()
                          ? edgeCount
                          : edgeCount + 1;
    failure = {VPTOScheduleFailureKind::Budget, "edges", actual,
               limits->maxEdges, {}};
    return mlir::failure();
  }
  if (failed(consumeWork(failure))) {
    return mlir::failure();
  }
  dag.addEdge(predecessor, successor, kind, strength, latency, reason);
  return success();
}

namespace {
struct IntegerRange {
  int64_t lowerInclusive = 0;
  int64_t upperInclusive = 0;
};

struct ResolvedMemoryAccess {
  VPTOMemoryAccess semantics;
  Value aliasRoot;
  std::optional<int64_t> absoluteByteOffset;
  std::optional<IntegerRange> absoluteByteRange;
  std::optional<PTOAddressExpr> addressExpression;
};

struct FrontierAccess {
  VPTOSUnit *unit = nullptr;
  ResolvedMemoryAccess access;
};

static std::optional<int64_t> getConstantInteger(Value value) {
  IntegerAttr constant;
  bool invalidConstant = !value ||
                         !matchPattern(value, m_Constant(&constant)) ||
                         !constant.getValue().isSignedIntN(64);
  if (invalidConstant) {
    return std::nullopt;
  }
  return constant.getValue().getSExtValue();
}

static std::optional<ConstantIntRanges> getIntegerRanges(Value value,
                                                         unsigned depth = 0);

static std::optional<ConstantIntRanges>
getLoopInductionRange(BlockArgument argument, unsigned bitWidth) {
  auto forOp = dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
  bool isLoopInduction = forOp && forOp.getInductionVar() == argument;
  if (!isLoopInduction) {
    return std::nullopt;
  }
  std::optional<int64_t> lower = getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
  if (!lower || !step || *lower < 0 || *step <= 0) {
    return std::nullopt;
  }
  APInt lowerBound(bitWidth, static_cast<uint64_t>(*lower),
                   /*isSigned=*/true);
  return ConstantIntRanges::fromSigned(lowerBound,
                                       APInt::getSignedMaxValue(bitWidth));
}

static std::optional<SmallVector<ConstantIntRanges>>
getOperandRanges(Operation &operation, unsigned depth) {
  SmallVector<ConstantIntRanges> ranges;
  ranges.reserve(operation.getNumOperands());
  for (Value operand : operation.getOperands()) {
    std::optional<ConstantIntRanges> range =
        getIntegerRanges(operand, depth + 1);
    if (!range) {
      return std::nullopt;
    }
    ranges.push_back(std::move(*range));
  }
  return ranges;
}

static std::optional<ConstantIntRanges> getIntegerRanges(Value value,
                                                         unsigned depth) {
  constexpr unsigned maxDepth = 32;
  if (!value || depth > maxDepth) {
    return std::nullopt;
  }
  IntegerAttr constant;
  if (matchPattern(value, m_Constant(&constant))) {
    return ConstantIntRanges::constant(constant.getValue());
  }
  unsigned bitWidth = ConstantIntRanges::getStorageBitwidth(value.getType());
  if (bitWidth == 0) {
    return std::nullopt;
  }
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    return getLoopInductionRange(argument, bitWidth);
  }
  Operation *definingOp = value.getDefiningOp();
  auto rangeInterface = dyn_cast_or_null<InferIntRangeInterface>(definingOp);
  if (!rangeInterface) {
    return std::nullopt;
  }
  std::optional<SmallVector<ConstantIntRanges>> ranges =
      getOperandRanges(*definingOp, depth);
  if (!ranges) {
    return std::nullopt;
  }
  std::optional<ConstantIntRanges> result;
  rangeInterface.inferResultRanges(
      *ranges, [&](Value resultValue, const ConstantIntRanges &range) {
        if (resultValue == value) {
          result = range;
        }
      });
  return result;
}

static std::optional<IntegerRange> getIntegerRange(Value value) {
  std::optional<ConstantIntRanges> range = getIntegerRanges(value);
  bool invalidRange = !range || !range->smin().isSignedIntN(64) ||
                      !range->smax().isSignedIntN(64);
  if (invalidRange) {
    return std::nullopt;
  }
  return IntegerRange{range->smin().getSExtValue(),
                      range->smax().getSExtValue()};
}

static std::optional<std::pair<__int128, __int128>>
getMathematicalResultRange(Operation *operation) {
  bool isAdd = isa_and_nonnull<arith::AddIOp>(operation);
  bool isSub = isa_and_nonnull<arith::SubIOp>(operation);
  bool isMul = isa_and_nonnull<arith::MulIOp>(operation);
  if (!isAdd && !isSub && !isMul) {
    return std::nullopt;
  }
  std::optional<IntegerRange> lhs = getIntegerRange(operation->getOperand(0));
  std::optional<IntegerRange> rhs = getIntegerRange(operation->getOperand(1));
  if (!lhs || !rhs) {
    return std::nullopt;
  }

  if (isAdd) {
    return std::make_pair(
        static_cast<__int128>(lhs->lowerInclusive) + rhs->lowerInclusive,
        static_cast<__int128>(lhs->upperInclusive) + rhs->upperInclusive);
  }
  if (isSub) {
    return std::make_pair(
        static_cast<__int128>(lhs->lowerInclusive) - rhs->upperInclusive,
        static_cast<__int128>(lhs->upperInclusive) - rhs->lowerInclusive);
  }
  std::array<__int128, 4> products = {
      static_cast<__int128>(lhs->lowerInclusive) * rhs->lowerInclusive,
      static_cast<__int128>(lhs->lowerInclusive) * rhs->upperInclusive,
      static_cast<__int128>(lhs->upperInclusive) * rhs->lowerInclusive,
      static_cast<__int128>(lhs->upperInclusive) * rhs->upperInclusive};
  auto bounds = std::minmax_element(products.begin(), products.end());
  return std::make_pair(*bounds.first, *bounds.second);
}

static bool isProvenNoSignedWrap(Value value) {
  Operation *operation = value.getDefiningOp();
  auto overflow =
      dyn_cast_or_null<arith::ArithIntegerOverflowFlagsInterface>(operation);
  if (overflow && overflow.hasNoSignedWrap()) {
    return true;
  }
  unsigned bitWidth = ConstantIntRanges::getStorageBitwidth(value.getType());
  auto resultRange = getMathematicalResultRange(operation);
  if (bitWidth == 0 || bitWidth > 64 || !resultRange) {
    return false;
  }
  __int128 signedMinimum = -(__int128{1} << (bitWidth - 1));
  __int128 signedMaximum = (__int128{1} << (bitWidth - 1)) - 1;
  return signedMinimum <= resultRange->first &&
         resultRange->second <= signedMaximum;
}

/// Remove finite-width source boundaries only when range inference proves
/// that the source operation cannot wrap. Remaining source values are kept as
/// opaque atoms, so equal dynamic terms can cancel without reassociating
/// unproven arithmetic.
static PTOTypedExprRef expandProvenPointExpression(
    const PTOTypedExprRef &expression) {
  if (!expression) {
    return {};
  }
  if (expression->kind == PTOTypedExpr::Kind::Constant ||
      expression->kind == PTOTypedExpr::Kind::Opaque ||
      expression->kind == PTOTypedExpr::Kind::Cast) {
    return expression;
  }
  if (expression->sourceValue &&
      !isProvenNoSignedWrap(expression->sourceValue)) {
    return makePTOOpaqueExpr(expression->sourceValue);
  }

  PTOTypedExprRef lhs = expandProvenPointExpression(expression->lhs);
  PTOTypedExprRef rhs = expandProvenPointExpression(expression->rhs);
  switch (expression->kind) {
  case PTOTypedExpr::Kind::Add:
    return makePTOAddExpr(lhs, rhs, expression->type);
  case PTOTypedExpr::Kind::Sub:
    return makePTOSubExpr(lhs, rhs, expression->type);
  case PTOTypedExpr::Kind::Mul:
    return makePTOMulExpr(lhs, rhs, expression->type);
  default:
    return expression;
  }
}

static FailureOr<PTOTypedExprRef>
getOperationOffsetBytes(const PTOAddressExpr &address, int64_t scale) {
  if (!address.offset) {
    return makePTOConstantExpr(0);
  }
  if (!address.offset->unitBytes || *address.offset->unitBytes <= 0) {
    return failure();
  }
  int64_t coefficient;
  if (llvm::MulOverflow(scale, *address.offset->unitBytes, coefficient)) {
    return failure();
  }
  PTOTypedExprRef offset =
      expandProvenPointExpression(address.offset->value);
  Type type = offset ? offset->type : Type();
  return makePTOMulExpr(offset, makePTOConstantExpr(coefficient, type), type);
}

static std::optional<int64_t>
getExactByteDifference(const ResolvedMemoryAccess &from,
                       const ResolvedMemoryAccess &to) {
  if (!from.addressExpression || !to.addressExpression) {
    return std::nullopt;
  }
  const PTOAddressExpr &fromAddress = *from.addressExpression;
  const PTOAddressExpr &toAddress = *to.addressExpression;
  if (fromAddress.rootOrBase != toAddress.rootOrBase ||
      fromAddress.elementBytes <= 0 ||
      fromAddress.elementBytes != toAddress.elementBytes) {
    return std::nullopt;
  }

  PTOTypedExprRef fromElements =
      expandProvenPointExpression(fromAddress.elementOffset);
  PTOTypedExprRef toElements =
      expandProvenPointExpression(toAddress.elementOffset);
  Type expressionType = toElements ? toElements->type : Type();
  PTOTypedExprRef difference = makePTOMulExpr(
      makePTOSubExpr(toElements, fromElements, expressionType),
      makePTOConstantExpr(toAddress.elementBytes, expressionType),
      expressionType);

  FailureOr<PTOTypedExprRef> fromOperation =
      getOperationOffsetBytes(fromAddress, -1);
  FailureOr<PTOTypedExprRef> toOperation =
      getOperationOffsetBytes(toAddress, 1);
  bool operationOffsetsUnavailable =
      failed(fromOperation) || failed(toOperation);
  if (operationOffsetsUnavailable) {
    return std::nullopt;
  }
  difference = makePTOAddExpr(difference, *fromOperation, expressionType);
  difference = makePTOAddExpr(difference, *toOperation, expressionType);

  std::optional<PTOLinearExpr> linear = normalizePTOLinearExpr(difference);
  if (!linear || !linear->terms.empty()) {
    return std::nullopt;
  }
  return linear->constant;
}

/// Resolve a conservative byte-address range for pointer-producing operations
/// explicit at the VPTO scheduling boundary. Unknown transforms and possible
/// integer overflow deliberately fall back to may-alias.
static std::optional<IntegerRange>
addDisplacement(int64_t address, const IntegerRange &displacement) {
  IntegerRange result;
  if (llvm::AddOverflow(address, displacement.lowerInclusive,
                        result.lowerInclusive) ||
      llvm::AddOverflow(address, displacement.upperInclusive,
                        result.upperInclusive)) {
    return std::nullopt;
  }
  return result;
}

static std::optional<int64_t> getAddPtrElementByteSize(AddPtrOp addPtr) {
  auto pointerType = dyn_cast<PtrType>(addPtr.getPtr().getType());
  if (!pointerType) {
    return std::nullopt;
  }
  Type elementType = pointerType.getElementType();
  bool invalidElementType = !elementType.isIntOrFloat() ||
                            elementType.getIntOrFloatBitWidth() % 8 != 0;
  if (invalidElementType) {
    return std::nullopt;
  }
  return static_cast<int64_t>(elementType.getIntOrFloatBitWidth() / 8);
}

static LogicalResult accumulateAddPtrDisplacement(AddPtrOp addPtr,
                                                  IntegerRange &displacement) {
  std::optional<IntegerRange> elementOffset =
      getIntegerRange(addPtr.getOffset());
  std::optional<int64_t> elementByteSize = getAddPtrElementByteSize(addPtr);
  if (!elementOffset || !elementByteSize) {
    return failure();
  }
  int64_t minimumByteOffset = 0;
  int64_t maximumByteOffset = 0;
  return failure(
      llvm::MulOverflow(elementOffset->lowerInclusive, *elementByteSize,
                        minimumByteOffset) ||
      llvm::MulOverflow(elementOffset->upperInclusive, *elementByteSize,
                        maximumByteOffset) ||
      llvm::AddOverflow(displacement.lowerInclusive, minimumByteOffset,
                        displacement.lowerInclusive) ||
      llvm::AddOverflow(displacement.upperInclusive, maximumByteOffset,
                        displacement.upperInclusive));
}

static std::optional<IntegerRange> getPointerAddressRange(Value value) {
  SmallPtrSet<Operation *, 8> visited;
  IntegerRange displacement;
  while (Operation *definingOp = value.getDefiningOp()) {
    if (!visited.insert(definingOp).second) {
      return std::nullopt;
    }
    if (auto cast = dyn_cast<CastPtrOp>(definingOp)) {
      Value input = cast.getInput();
      if (std::optional<int64_t> address = getConstantInteger(input)) {
        return addDisplacement(*address, displacement);
      }
      if (!isa<PtrType>(input.getType())) {
        return std::nullopt;
      }
      value = input;
      continue;
    }
    if (auto intToPtr = dyn_cast<IntToPtrOp>(definingOp)) {
      std::optional<int64_t> address = getConstantInteger(intToPtr.getAddr());
      if (!address) {
        return std::nullopt;
      }
      return addDisplacement(*address, displacement);
    }
    if (auto addPtr = dyn_cast<AddPtrOp>(definingOp)) {
      if (failed(accumulateAddPtrDisplacement(addPtr, displacement))) {
        return std::nullopt;
      }
      value = addPtr.getPtr();
      continue;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

static Value getAliasRoot(Value value) {
  SmallPtrSet<Operation *, 8> visited;
  while (Operation *definingOp = value.getDefiningOp()) {
    if (!visited.insert(definingOp).second)
      break;
    std::optional<std::pair<Value, Value>> alias =
        getOperationAliasInfo(definingOp);
    if (!alias || alias->first != value || !alias->second)
      break;
    value = alias->second;
  }
  return value;
}

static const PTOAddressExpr *
findUniqueAddressExpression(Value address,
                            ArrayRef<PTOAddressExpr> expressions) {
  const PTOAddressExpr *matching = nullptr;
  for (const PTOAddressExpr &expression : expressions) {
    if (expression.currentBase != address) {
      continue;
    }
    if (matching) {
      return nullptr;
    }
    matching = &expression;
  }
  return matching;
}

static void resolveAbsoluteByteRange(ResolvedMemoryAccess &access) {
  std::optional<IntegerRange> pointerRange =
      getPointerAddressRange(access.semantics.address);
  if (!pointerRange || !access.semantics.byteOffset) {
    return;
  }
  IntegerRange byteRange;
  bool overflow =
      llvm::AddOverflow(pointerRange->lowerInclusive,
                        *access.semantics.byteOffset,
                        byteRange.lowerInclusive) ||
      llvm::AddOverflow(pointerRange->upperInclusive,
                        *access.semantics.byteOffset, byteRange.upperInclusive);
  if (overflow) {
    return;
  }
  access.absoluteByteRange = byteRange;
  if (byteRange.lowerInclusive == byteRange.upperInclusive) {
    access.absoluteByteOffset = byteRange.lowerInclusive;
  }
}

static ResolvedMemoryAccess
resolveMemoryAccess(const VPTOMemoryAccess &memoryAccess,
                    ArrayRef<PTOAddressExpr> addressExpressions) {
  ResolvedMemoryAccess access{
      memoryAccess, {}, std::nullopt, std::nullopt, std::nullopt};
  if (!access.semantics.address) {
    return access;
  }
  if (const PTOAddressExpr *expression = findUniqueAddressExpression(
          access.semantics.address, addressExpressions)) {
    access.addressExpression = *expression;
  }
  resolveAbsoluteByteRange(access);
  access.aliasRoot = getAliasRoot(access.semantics.address);
  if (access.aliasRoot != access.semantics.address) {
    access.semantics.byteOffset.reset();
  }
  return access;
}

static SmallVector<ResolvedMemoryAccess>
resolveMemoryAccesses(Operation *operation,
                      const VPTOSchedulingSemantics &semantics,
                      PTOAddressAnalysis *addressAnalysis) {
  SmallVector<PTOAddressExpr> addressExpressions;
  if (addressAnalysis) {
    auto result = addressAnalysis->getAddresses(operation);
    if (result) {
      addressExpressions = std::move(*result.value);
    }
  }
  SmallVector<ResolvedMemoryAccess> accesses;
  accesses.reserve(semantics.memoryAccesses.size());
  for (const VPTOMemoryAccess &memoryAccess : semantics.memoryAccesses) {
    accesses.push_back(resolveMemoryAccess(memoryAccess, addressExpressions));
  }
  return accesses;
}

static std::optional<bool>
getExactDifferenceOverlap(const ResolvedMemoryAccess &lhs,
                          const ResolvedMemoryAccess &rhs) {
  bool hasPositiveSizes = lhs.semantics.byteSize &&
                          *lhs.semantics.byteSize > 0 &&
                          rhs.semantics.byteSize && *rhs.semantics.byteSize > 0;
  if (!hasPositiveSizes) {
    return std::nullopt;
  }
  std::optional<int64_t> rhsBegin = getExactByteDifference(lhs, rhs);
  int64_t rhsEnd = 0;
  if (!rhsBegin ||
      llvm::AddOverflow(*rhsBegin, *rhs.semantics.byteSize, rhsEnd)) {
    return std::nullopt;
  }
  return *rhsBegin < *lhs.semantics.byteSize && rhsEnd > 0;
}

static bool hasDisjointAbsoluteRanges(const ResolvedMemoryAccess &lhs,
                                      const ResolvedMemoryAccess &rhs) {
  if (!lhs.absoluteByteRange || !lhs.semantics.byteSize ||
      !rhs.absoluteByteRange || !rhs.semantics.byteSize) {
    return false;
  }
  int64_t lhsLatestEnd = 0;
  int64_t rhsLatestEnd = 0;
  bool overflow = llvm::AddOverflow(lhs.absoluteByteRange->upperInclusive,
                                    *lhs.semantics.byteSize, lhsLatestEnd) ||
                  llvm::AddOverflow(rhs.absoluteByteRange->upperInclusive,
                                    *rhs.semantics.byteSize, rhsLatestEnd);
  return !overflow && (lhsLatestEnd <= rhs.absoluteByteRange->lowerInclusive ||
                       rhsLatestEnd <= lhs.absoluteByteRange->lowerInclusive);
}

static std::optional<bool> getOffsetOverlap(std::optional<int64_t> lhsOffset,
                                            std::optional<int64_t> lhsSize,
                                            std::optional<int64_t> rhsOffset,
                                            std::optional<int64_t> rhsSize) {
  if (!lhsOffset || !lhsSize || !rhsOffset || !rhsSize) {
    return std::nullopt;
  }
  int64_t lhsEnd = 0;
  int64_t rhsEnd = 0;
  bool overflow = llvm::AddOverflow(*lhsOffset, *lhsSize, lhsEnd) ||
                  llvm::AddOverflow(*rhsOffset, *rhsSize, rhsEnd);
  if (overflow) {
    return std::nullopt;
  }
  return *lhsOffset < rhsEnd && *rhsOffset < lhsEnd;
}

static bool mayAlias(const ResolvedMemoryAccess &lhs,
                     const ResolvedMemoryAccess &rhs) {
  bool differentAddressSpaces =
      lhs.semantics.addressSpace && rhs.semantics.addressSpace &&
      lhs.semantics.addressSpace != rhs.semantics.addressSpace;
  if (differentAddressSpaces) {
    return false;
  }
  if (!lhs.semantics.address || !rhs.semantics.address) {
    return true;
  }
  if (std::optional<bool> overlap = getExactDifferenceOverlap(lhs, rhs)) {
    return *overlap;
  }
  if (hasDisjointAbsoluteRanges(lhs, rhs)) {
    return false;
  }
  if (std::optional<bool> overlap =
          getOffsetOverlap(lhs.absoluteByteOffset, lhs.semantics.byteSize,
                           rhs.absoluteByteOffset, rhs.semantics.byteSize)) {
    return *overlap;
  }
  bool hasSameAliasRoot = lhs.aliasRoot == rhs.aliasRoot;
  if (hasSameAliasRoot) {
    if (std::optional<bool> overlap = getOffsetOverlap(
            lhs.semantics.byteOffset, lhs.semantics.byteSize,
            rhs.semantics.byteOffset, rhs.semantics.byteSize)) {
      return *overlap;
    }
  }
  // Different roots in the same physical space remain conservative: memory
  // planning may have assigned overlapping ranges to distinct SSA roots.
  return true;
}

static bool needsMemoryOrder(const ResolvedMemoryAccess &lhs,
                             const ResolvedMemoryAccess &rhs) {
  if (!mayAlias(lhs, rhs))
    return false;
  if (lhs.semantics.ordered || rhs.semantics.ordered || lhs.semantics.unknown ||
      rhs.semantics.unknown)
    return true;
  return lhs.semantics.writes || rhs.semantics.writes;
}

struct MemoryRange {
  int64_t begin = 0;
  int64_t end = 0;
};

static std::optional<MemoryRange>
getMemoryRange(std::optional<int64_t> byteOffset,
               std::optional<int64_t> byteSize) {
  if (!byteOffset || !byteSize) {
    return std::nullopt;
  }
  int64_t end;
  if (llvm::AddOverflow(*byteOffset, *byteSize, end)) {
    return std::nullopt;
  }
  return MemoryRange{*byteOffset, end};
}

static bool containsMemoryRange(const MemoryRange &outer,
                                const MemoryRange &inner) {
  return outer.begin <= inner.begin && inner.end <= outer.end;
}

static bool coversAddressSpace(const ResolvedMemoryAccess &prior,
                               const ResolvedMemoryAccess &current) {
  if (!current.semantics.addressSpace) {
    return true;
  }
  return prior.semantics.addressSpace &&
         current.semantics.addressSpace == prior.semantics.addressSpace;
}

static bool containsMemoryRange(const ResolvedMemoryAccess &prior,
                                const ResolvedMemoryAccess &current) {
  if (prior.semantics.byteSize && *prior.semantics.byteSize > 0 &&
      current.semantics.byteSize && *current.semantics.byteSize > 0) {
    if (std::optional<int64_t> priorBegin =
            getExactByteDifference(current, prior)) {
      int64_t priorEnd;
      if (!llvm::AddOverflow(*priorBegin, *prior.semantics.byteSize,
                             priorEnd) &&
          *priorBegin >= 0 && priorEnd <= *current.semantics.byteSize) {
        return true;
      }
    }
  }

  std::optional<MemoryRange> currentAbsolute = getMemoryRange(
      current.absoluteByteOffset, current.semantics.byteSize);
  std::optional<MemoryRange> priorAbsolute =
      getMemoryRange(prior.absoluteByteOffset, prior.semantics.byteSize);
  if (currentAbsolute && priorAbsolute &&
      containsMemoryRange(*currentAbsolute, *priorAbsolute)) {
    return true;
  }

  std::optional<MemoryRange> currentRelative = getMemoryRange(
      current.semantics.byteOffset, current.semantics.byteSize);
  std::optional<MemoryRange> priorRelative =
      getMemoryRange(prior.semantics.byteOffset, prior.semantics.byteSize);
  return current.aliasRoot && current.aliasRoot == prior.aliasRoot &&
         currentRelative && priorRelative &&
         containsMemoryRange(*currentRelative, *priorRelative);
}

static bool hasMemoryRange(const ResolvedMemoryAccess &access) {
  return (access.absoluteByteRange && access.semantics.byteSize) ||
         getMemoryRange(access.absoluteByteOffset, access.semantics.byteSize) ||
         getMemoryRange(access.semantics.byteOffset,
                        access.semantics.byteSize);
}

/// Return whether replacing `prior` with `current` preserves every future
/// ordering edge that `prior` could require. Merely overlapping ranges is not
/// sufficient: a later access may overlap the part of `prior` that lies
/// outside `current`, leaving no transitive path through `current`.
static bool subsumesMemoryFrontierAccess(
    const ResolvedMemoryAccess &prior,
    const ResolvedMemoryAccess &current) {
  bool closesFrontier = current.semantics.writes ||
                        current.semantics.ordered || current.semantics.unknown;
  if (!closesFrontier || !mayAlias(prior, current)) {
    return false;
  }

  if (!coversAddressSpace(prior, current)) {
    return false;
  }
  if (containsMemoryRange(prior, current)) {
    return true;
  }

  // Without any usable interval the alias model treats `current` as covering
  // its complete compatible address space, so it safely represents every
  // prior entry in that space. If it has a bounded interval, retain any prior
  // entry that was not proven to be contained above.
  return !hasMemoryRange(current);
}

static bool isPostUpdateAddress(const VPTOSUnit &producer, Value value) {
  return llvm::any_of(
      producer.getSemantics().effects, [&](const VPTOSchedulingEffect &effect) {
        return effect.kind == VPTOSchedulingEffectKind::PostUpdate &&
               effect.value == value;
      });
}

template <typename ConsumeWork>
static LogicalResult
collectMemoryPredecessors(ArrayRef<FrontierAccess> frontier,
                          ArrayRef<ResolvedMemoryAccess> currentAccesses,
                          ConsumeWork &&consumeWork,
                          SmallPtrSetImpl<VPTOSUnit *> &predecessors) {
  for (const FrontierAccess &prior : frontier) {
    for (const ResolvedMemoryAccess &current : currentAccesses) {
      if (failed(consumeWork())) {
        return mlir::failure();
      }
      if (needsMemoryOrder(prior.access, current)) {
        predecessors.insert(prior.unit);
        break;
      }
    }
  }
  return success();
}

template <typename ConsumeWork>
static FailureOr<SmallVector<SmallVector<ResolvedMemoryAccess>>>
collectResolvedMemoryAccesses(const VPTOSchedDAG &dag,
                              PTOAddressAnalysis *addressAnalysis,
                              ConsumeWork &&consumeWork) {
  SmallVector<SmallVector<ResolvedMemoryAccess>> accesses;
  accesses.reserve(dag.getUnits().size());
  for (const std::unique_ptr<VPTOSUnit> &unit : dag.getUnits()) {
    if (failed(consumeWork(unit->getSemantics().memoryAccesses.size()))) {
      return failure();
    }
    accesses.push_back(resolveMemoryAccesses(
        unit->getOperation(), unit->getSemantics(), addressAnalysis));
  }
  return accesses;
}

static void
updateMemoryFrontier(SmallVectorImpl<FrontierAccess> &frontier, VPTOSUnit &unit,
                     ArrayRef<ResolvedMemoryAccess> currentAccesses) {
  llvm::erase_if(frontier, [&](const FrontierAccess &prior) {
    return llvm::any_of(
        currentAccesses, [&](const ResolvedMemoryAccess &current) {
          return subsumesMemoryFrontierAccess(prior.access, current);
        });
  });
  for (const ResolvedMemoryAccess &current : currentAccesses) {
    frontier.push_back({&unit, current});
  }
}

template <typename AddEdge>
static LogicalResult addBarrierPredecessors(const VPTOSchedDAG &dag,
                                            VPTOSUnit &barrier,
                                            AddEdge &&addEdge) {
  for (const std::unique_ptr<VPTOSUnit> &prior : dag.getUnits()) {
    bool reachedBarrier =
        prior->getOriginalIndex() >= barrier.getOriginalIndex();
    if (reachedBarrier) {
      break;
    }
    if (failed(addEdge(*prior, barrier, VPTOSchedEdgeKind::Sync, 0,
                       "before scheduling barrier"))) {
      return mlir::failure();
    }
  }
  return success();
}

template <typename AddEdge>
static LogicalResult
addImplicitReadEdge(const VPTOSchedulingEffect &effect, VPTOSUnit &unit,
                    llvm::StringMap<VPTOSUnit *> &lastWrite,
                    llvm::StringMap<SmallVector<VPTOSUnit *>> &readsSinceWrite,
                    AddEdge &&addEdge) {
  if (VPTOSUnit *writer = lastWrite.lookup(effect.resource)) {
    if (failed(addEdge(*writer, unit, VPTOSchedEdgeKind::Data, 1,
                       Twine("implicit read of ") + effect.resource))) {
      return mlir::failure();
    }
  }
  readsSinceWrite[effect.resource].push_back(&unit);
  return success();
}

template <typename AddEdge>
static LogicalResult addImplicitWriteEdges(
    const VPTOSchedulingEffect &effect, VPTOSUnit &unit,
    llvm::StringMap<VPTOSUnit *> &lastWrite,
    llvm::StringMap<SmallVector<VPTOSUnit *>> &readsSinceWrite,
    AddEdge &&addEdge) {
  if (VPTOSUnit *writer = lastWrite.lookup(effect.resource)) {
    if (failed(addEdge(*writer, unit, VPTOSchedEdgeKind::Output, 0,
                       Twine("implicit write of ") + effect.resource))) {
      return mlir::failure();
    }
  }
  for (VPTOSUnit *reader : readsSinceWrite[effect.resource]) {
    if (failed(
            addEdge(*reader, unit, VPTOSchedEdgeKind::Anti, 0,
                    Twine("implicit anti-dependence on ") + effect.resource))) {
      return mlir::failure();
    }
  }
  readsSinceWrite[effect.resource].clear();
  lastWrite[effect.resource] = &unit;
  return success();
}

template <typename ConsumeWork>
static FailureOr<bool>
liveInEscapesRegion(const VPTOSchedDAG &dag, Operation &lastRegionOperation,
                    Value value, ConsumeWork &&consumeWork) {
  for (Operation *user : value.getUsers()) {
    if (failed(consumeWork())) {
      return failure();
    }
    if (dag.lookup(user)) {
      continue;
    }
    bool isInAnotherBlock = user->getBlock() != dag.getRegion().block;
    bool isAfterRegion =
        !isInAnotherBlock && lastRegionOperation.isBeforeInBlock(user);
    if (isInAnotherBlock || isAfterRegion) {
      return true;
    }
  }
  return false;
}

template <typename ConsumeWork, typename AddEdge>
static LogicalResult
addSSAOperandEdges(VPTOSchedDAG &dag, VPTOSUnit &unit,
                   const VPTOSchedModel *model, Operation &lastRegionOperation,
                   DenseSet<Value> &checkedLiveIns, ConsumeWork &&consumeWork,
                   AddEdge &&addEdge) {
  for (auto [operandIndex, operand] :
       llvm::enumerate(unit.getOperation()->getOperands())) {
    if (failed(consumeWork())) {
      return mlir::failure();
    }
    Operation *definingOp = operand.getDefiningOp();
    VPTOSUnit *predecessor = definingOp ? dag.lookup(definingOp) : nullptr;
    if (!predecessor) {
      dag.addLiveIn(operand);
      if (!checkedLiveIns.insert(operand).second) {
        continue;
      }
      FailureOr<bool> escapes =
          liveInEscapesRegion(dag, lastRegionOperation, operand, consumeWork);
      if (failed(escapes)) {
        return mlir::failure();
      }
      if (*escapes) {
        dag.addLiveOut(operand);
      }
      continue;
    }
    unsigned latency =
        model ? model->getSchedParameters(predecessor->getOperation())
                    .writeLatency
              : 1;
    std::string reason =
        (isPostUpdateAddress(*predecessor, operand)
             ? Twine("post-update address operand #") + Twine(operandIndex)
             : Twine("ssa operand #") + Twine(operandIndex))
            .str();
    if (failed(addEdge(*predecessor, unit, latency, reason))) {
      return mlir::failure();
    }
  }
  return success();
}

template <typename ConsumeWork>
static LogicalResult addSSAResultLiveOuts(VPTOSchedDAG &dag, VPTOSUnit &unit,
                                          ConsumeWork &&consumeWork) {
  for (Value result : unit.getOperation()->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (failed(consumeWork())) {
        return mlir::failure();
      }
      if (!dag.lookup(user)) {
        dag.addLiveOut(result);
        break;
      }
    }
  }
  return success();
}

} // namespace

LogicalResult VPTOSchedDAGBuilder::buildMemoryEdges(
    VPTOSchedDAG &dag, VPTOScheduleFailure &failure) const {
  std::unique_ptr<PTOValueEvolutionAnalysis> valueEvolution;
  std::unique_ptr<PTOAddressAnalysis> addressAnalysis;
  if (!dag.getUnits().empty()) {
    func::FuncOp function =
        dag.getUnits().front()->getOperation()->getParentOfType<func::FuncOp>();
    if (function) {
      valueEvolution =
          std::make_unique<PTOValueEvolutionAnalysis>(function);
      addressAnalysis =
          std::make_unique<PTOAddressAnalysis>(function, *valueEvolution);
    }
  }

  auto consume = [&](uint64_t amount) { return consumeWork(failure, amount); };
  auto resolved =
      collectResolvedMemoryAccesses(dag, addressAnalysis.get(), consume);
  if (failed(resolved)) {
    return mlir::failure();
  }

  SmallVector<FrontierAccess> frontier;
  for (auto indexedAccesses : llvm::enumerate(*resolved)) {
    size_t unitIndex = indexedAccesses.index();
    ArrayRef<ResolvedMemoryAccess> currentAccesses = indexedAccesses.value();
    if (currentAccesses.empty()) {
      continue;
    }
    VPTOSUnit *unit = dag.getUnits()[unitIndex].get();
    SmallPtrSet<VPTOSUnit *, 8> predecessors;
    auto consumeOne = [&]() { return consumeWork(failure); };
    if (failed(collectMemoryPredecessors(frontier, currentAccesses, consumeOne,
                                         predecessors))) {
      return mlir::failure();
    }
    for (VPTOSUnit *predecessor : predecessors) {
      if (failed(addEdge(dag, *predecessor, *unit,
                         VPTOSchedEdgeKind::Memory,
                         VPTOSchedEdgeStrength::Must,
                         /*latency=*/0,
                         "may-alias memory frontier in original order",
                         failure))) {
        return mlir::failure();
      }
    }

    // A write (or ordered/unknown access) replaces an earlier entry only when
    // its may-alias coverage contains that entry. Partial overlap establishes
    // an edge but cannot remove the old entry: a future access may touch only
    // the uncovered portion. Read-only accesses remain side by side until a
    // later closing access safely subsumes them.
    updateMemoryFrontier(frontier, *unit, currentAccesses);
  }
  return success();
}

LogicalResult VPTOSchedDAGBuilder::buildImplicitAndSyncEdges(
    VPTOSchedDAG &dag, VPTOScheduleFailure &failure) const {
  llvm::StringMap<VPTOSUnit *> lastWrite;
  llvm::StringMap<SmallVector<VPTOSUnit *>> readsSinceWrite;
  VPTOSUnit *lastBarrier = nullptr;

  for (const std::unique_ptr<VPTOSUnit> &unitOwner : dag.getUnits()) {
    VPTOSUnit &unit = *unitOwner;
    auto add = [&](VPTOSUnit &predecessor, VPTOSUnit &successor,
                   VPTOSchedEdgeKind kind, unsigned latency, Twine reason) {
      return addEdge(dag, predecessor, successor, kind,
                     VPTOSchedEdgeStrength::Must, latency, reason, failure);
    };
    if (lastBarrier && lastBarrier != &unit &&
        failed(add(*lastBarrier, unit, VPTOSchedEdgeKind::Sync, 0,
                   "after scheduling barrier"))) {
      return mlir::failure();
    }

    for (const VPTOSchedulingEffect &effect : unit.getSemantics().effects) {
      if (failed(consumeWork(failure))) {
        return mlir::failure();
      }
      if (effect.kind == VPTOSchedulingEffectKind::Barrier) {
        if (failed(addBarrierPredecessors(dag, unit, add))) {
          return mlir::failure();
        }
        lastBarrier = &unit;
        continue;
      }
      if (effect.resource.empty()) {
        continue;
      }
      if (effect.kind == VPTOSchedulingEffectKind::ImplicitRead) {
        if (failed(addImplicitReadEdge(effect, unit, lastWrite, readsSinceWrite,
                                       add))) {
          return mlir::failure();
        }
        continue;
      }
      if (effect.kind != VPTOSchedulingEffectKind::ImplicitWrite) {
        continue;
      }
      if (failed(addImplicitWriteEdges(effect, unit, lastWrite, readsSinceWrite,
                                       add))) {
        return mlir::failure();
      }
    }
  }
  return success();
}

LogicalResult VPTOSchedDAGBuilder::buildSSAEdges(
    VPTOSchedDAG &dag, VPTOScheduleFailure &failure) const {
  for (Value value : dag.getRegion().liveThroughs) {
    if (failed(consumeWork(failure))) {
      return mlir::failure();
    }
    dag.addLiveIn(value);
    dag.addLiveOut(value);
  }

  DenseSet<Value> checkedLiveIns;
  Operation *lastRegionOperation = dag.getRegion().operations.back();
  for (const std::unique_ptr<VPTOSUnit> &unitOwner : dag.getUnits()) {
    VPTOSUnit &unit = *unitOwner;
    auto consume = [&]() { return consumeWork(failure); };
    auto add = [&](VPTOSUnit &predecessor, VPTOSUnit &successor,
                   unsigned latency, Twine reason) {
      return addEdge(dag, predecessor, successor, VPTOSchedEdgeKind::Data,
                     VPTOSchedEdgeStrength::Must, latency, reason, failure);
    };
    if (failed(addSSAOperandEdges(dag, unit, model, *lastRegionOperation,
                                  checkedLiveIns, consume, add)) ||
        failed(addSSAResultLiveOuts(dag, unit, consume))) {
      return mlir::failure();
    }
  }
  return success();
}

LogicalResult VPTOSchedDAGBuilder::buildModelFallbackEdges(
    VPTOSchedDAG &dag, VPTOScheduleFailure &failure) const {
  if (!model)
    return success();
  ArrayRef<std::unique_ptr<VPTOSUnit>> units = dag.getUnits();
  for (size_t index = 0; index < units.size(); ++index) {
    if (failed(consumeWork(failure))) {
      return mlir::failure();
    }
    VPTOSUnit &unit = *units[index];
    if (model->getSchedClass(unit.getOperation()).known)
      continue;
    if (index != 0 &&
        failed(addEdge(dag, *units[index - 1], unit,
                       VPTOSchedEdgeKind::Artificial,
                       VPTOSchedEdgeStrength::Must, 0,
                       "unknown sched class preserves predecessor order",
                       failure))) {
      return mlir::failure();
    }
    if (index + 1 != units.size()) {
      if (failed(addEdge(dag, unit, *units[index + 1],
                         VPTOSchedEdgeKind::Artificial,
                         VPTOSchedEdgeStrength::Must, 0,
                         "unknown sched class preserves successor order",
                         failure))) {
        return mlir::failure();
      }
    }
  }
  return success();
}
