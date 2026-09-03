// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/VPTOStatefulStreamFusion.h"
#include "PTO/Analysis/PTOAddressAnalysis.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::pto;

namespace {
struct StatefulStoreStream {
  InitAlignOp init;
  VstusOp firstStore;
  VstasOp flush;
  Value initialBase;
  Value finalAlign;
  Value finalBase;
  int64_t totalAdvance = 0;
};

static std::optional<int64_t> getConstantInt64(Value value) {
  APInt constant;
  bool isConstant = matchPattern(value, m_ConstantInt(&constant)) &&
                    constant.isSignedIntN(64);
  if (!isConstant) {
    return std::nullopt;
  }
  return constant.getSExtValue();
}

static std::optional<StatefulStoreStream>
parseStatefulStoreStream(InitAlignOp init) {
  Value initialAlign = init.getResult();
  if (!initialAlign.hasOneUse()) {
    return std::nullopt;
  }
  auto firstStore = dyn_cast<VstusOp>(*initialAlign.getUsers().begin());
  bool isFirstStore = firstStore && firstStore.getAlignIn() == initialAlign &&
                      firstStore->getBlock() == init->getBlock();
  if (!isFirstStore) {
    return std::nullopt;
  }

  StatefulStoreStream stream{init, firstStore, {}, firstStore.getBase(),
                             {},   {},         0};
  VstusOp store = firstStore;
  while (true) {
    Value baseOut = store.getBaseOut();
    std::optional<int64_t> advance = getConstantInt64(store.getOffset());
    if (!baseOut || !advance ||
        llvm::AddOverflow(stream.totalAdvance, *advance, stream.totalAdvance) ||
        !store.getAlignOut().hasOneUse() || !baseOut.hasOneUse()) {
      return std::nullopt;
    }

    Operation *alignUser = *store.getAlignOut().getUsers().begin();
    Operation *baseUser = *baseOut.getUsers().begin();
    if (auto nextStore = dyn_cast<VstusOp>(alignUser)) {
      if (baseUser != nextStore ||
          nextStore.getAlignIn() != store.getAlignOut() ||
          nextStore.getBase() != baseOut ||
          nextStore->getBlock() != init->getBlock()) {
        return std::nullopt;
      }
      store = nextStore;
      continue;
    }

    auto flush = dyn_cast<VstasOp>(alignUser);
    std::optional<int64_t> flushOffset;
    if (flush) {
      flushOffset = getConstantInt64(flush.getOffset());
    }
    bool isValidFlush =
        flush && baseUser == flush && flush.getValue() == store.getAlignOut() &&
        flush.getDestination() == baseOut && !flush.getUpdatedBase() &&
        flushOffset && *flushOffset == 0 &&
        flush->getBlock() == init->getBlock();
    if (!isValidFlush) {
      return std::nullopt;
    }

    stream.flush = flush;
    stream.finalAlign = store.getAlignOut();
    stream.finalBase = baseOut;
    return stream;
  }
}

static bool areContiguousStatefulAddresses(Value first, int64_t advanceElements,
                                           Value second) {
  auto firstType = dyn_cast<PtrType>(first.getType());
  if (!firstType || firstType != second.getType()) {
    return false;
  }
  unsigned elementBits = getPTOStorageElemBitWidth(firstType.getElementType());
  if (elementBits == 0 || elementBits % 8 != 0) {
    return false;
  }
  int64_t elementBytes = static_cast<int64_t>(elementBits / 8);
  int64_t expectedBytes;
  if (llvm::MulOverflow(advanceElements, elementBytes, expectedBytes)) {
    return false;
  }
  auto difference = getKnownAddressDifferenceBytes(first, second);
  return difference && *difference == expectedBytes;
}

static bool canMoveStatefulStoreFlush(StatefulStoreStream first,
                                      StatefulStoreStream second) {
  bool isOrdered = first.flush->getBlock() == second.firstStore->getBlock() &&
                   first.flush->isBeforeInBlock(second.firstStore);
  if (!isOrdered) {
    return false;
  }
  bool sawSecondInit = false;
  for (Operation *op = first.flush->getNextNode(); op != second.firstStore;
       op = op->getNextNode()) {
    if (!op) {
      return false;
    }
    if (op == second.init) {
      sawSecondInit = true;
      continue;
    }
    bool isSafeInterveningOp = isa<InitAlignOp>(op) == false &&
                               op->getNumRegions() == 0 &&
                               isMemoryEffectFree(op);
    if (!isSafeInterveningOp) {
      return false;
    }
  }
  return sawSecondInit;
}

static bool tryFuseStatefulStoreStreams(StatefulStoreStream first,
                                        StatefulStoreStream second) {
  bool matchingTypes =
      first.initialBase.getType() == second.initialBase.getType();
  if (!matchingTypes || !canMoveStatefulStoreFlush(first, second)) {
    return false;
  }
  bool isContiguous = areContiguousStatefulAddresses(
      first.initialBase, first.totalAdvance, second.initialBase);
  if (!isContiguous) {
    return false;
  }

  second.firstStore.getAlignInMutable().set(first.finalAlign);
  second.firstStore.getBaseMutable().set(first.finalBase);
  first.flush.erase();
  second.init.erase();
  return true;
}

static void fuseStatefulStoreStreams(ModuleOp module) {
  while (true) {
    SmallVector<InitAlignOp> initializers;
    module.walk([&](InitAlignOp init) { initializers.push_back(init); });
    DenseMap<Block *, StatefulStoreStream> previousStreams;
    bool changed = false;
    for (InitAlignOp init : initializers) {
      std::optional<StatefulStoreStream> stream =
          parseStatefulStoreStream(init);
      Block *block = init->getBlock();
      if (!stream) {
        previousStreams.erase(block);
        continue;
      }
      auto previous = previousStreams.find(block);
      bool canFuse = previous != previousStreams.end() &&
                     tryFuseStatefulStoreStreams(previous->second, *stream);
      if (canFuse) {
        changed = true;
        break;
      }
      previousStreams[block] = *stream;
    }
    if (!changed) {
      return;
    }
  }
}

struct StatefulLoadStream {
  VldasOp init;
  VldusOp firstLoad;
  VldusOp lastLoad;
  Value initialSource;
  Value finalAlign;
  Value finalBase;
  int64_t totalElements = 0;
};

static std::optional<StatefulLoadStream> parseStatefulLoadStream(VldasOp init) {
  Value initialAlign = init.getResult();
  if (!initialAlign.hasOneUse()) {
    return std::nullopt;
  }
  auto firstLoad = dyn_cast<VldusOp>(*initialAlign.getUsers().begin());
  bool isFirstLoad = firstLoad && firstLoad.getAlign() == initialAlign &&
                     firstLoad->getBlock() == init->getBlock() &&
                     firstLoad.getIncrement() && firstLoad.getUpdatedBase();
  if (!isFirstLoad) {
    return std::nullopt;
  }

  StatefulLoadStream stream{init, firstLoad, {}, firstLoad.getSource(),
                            {},   {},        0};
  VldusOp load = firstLoad;
  while (true) {
    auto resultType = dyn_cast<VRegType>(load.getResult().getType());
    auto increment = getConstantInt64(load.getIncrement());
    if (!resultType || !increment ||
        *increment != resultType.getElementCount() ||
        llvm::AddOverflow(stream.totalElements, *increment,
                          stream.totalElements)) {
      return std::nullopt;
    }
    Value nextAlign = load.getUpdatedAlign();
    Value nextBase = load.getUpdatedBase();
    if (!nextAlign.hasOneUse()) {
      stream.lastLoad = load;
      stream.finalAlign = nextAlign;
      stream.finalBase = nextBase;
      return stream;
    }
    auto nextLoad = dyn_cast<VldusOp>(*nextAlign.getUsers().begin());
    bool isNextLoad = nextLoad && nextLoad.getAlign() == nextAlign &&
                      nextBase.hasOneUse() &&
                      *nextBase.getUsers().begin() == nextLoad &&
                      nextLoad.getSource() == nextBase &&
                      nextLoad->getBlock() == init->getBlock() &&
                      nextLoad.getIncrement() && nextLoad.getUpdatedBase();
    if (!isNextLoad) {
      return std::nullopt;
    }
    load = nextLoad;
  }
}

static bool canMoveLoadStream(StatefulLoadStream first,
                              StatefulLoadStream second) {
  bool isOrdered = first.lastLoad->getBlock() == second.firstLoad->getBlock() &&
                   first.lastLoad->isBeforeInBlock(second.init);
  if (!isOrdered) {
    return false;
  }
  for (Operation *op = first.lastLoad->getNextNode(); op != second.init;
       op = op->getNextNode()) {
    bool isSafeInterveningOp =
        op && op->getNumRegions() == 0 && isMemoryEffectFree(op);
    if (!isSafeInterveningOp) {
      return false;
    }
  }
  return true;
}

static bool tryFuseStatefulLoadStreams(StatefulLoadStream first,
                                       StatefulLoadStream second) {
  bool matchingTypes =
      first.initialSource.getType() == second.initialSource.getType();
  if (!matchingTypes || !canMoveLoadStream(first, second)) {
    return false;
  }
  bool isContiguous = areContiguousStatefulAddresses(
      first.initialSource, first.totalElements, second.initialSource);
  if (!isContiguous) {
    return false;
  }

  second.firstLoad.getSourceMutable().set(first.finalBase);
  second.firstLoad.getAlignMutable().set(first.finalAlign);
  second.init.erase();
  return true;
}

static void fuseStatefulLoadStreams(ModuleOp module) {
  while (true) {
    SmallVector<VldasOp> initializers;
    module.walk([&](VldasOp init) { initializers.push_back(init); });
    DenseMap<Block *, StatefulLoadStream> previousStreams;
    bool changed = false;
    for (VldasOp init : initializers) {
      auto stream = parseStatefulLoadStream(init);
      Block *block = init->getBlock();
      if (!stream) {
        previousStreams.erase(block);
        continue;
      }
      auto previous = previousStreams.find(block);
      bool canFuse = previous != previousStreams.end() &&
                     tryFuseStatefulLoadStreams(previous->second, *stream);
      if (canFuse) {
        changed = true;
        break;
      }
      previousStreams[block] = *stream;
    }
    if (!changed) {
      return;
    }
  }
}

static std::optional<int64_t> getLoopTripCount(scf::ForOp loop) {
  auto lower = getConstantInt64(loop.getLowerBound());
  auto upper = getConstantInt64(loop.getUpperBound());
  auto step = getConstantInt64(loop.getStep());
  if (!lower || !upper || !step || *step <= 0 || *lower >= *upper) {
    return std::nullopt;
  }
  __int128 distance = static_cast<__int128>(*upper) - *lower;
  __int128 count = (distance + static_cast<__int128>(*step) - 1) / *step;
  if (count > std::numeric_limits<int64_t>::max()) {
    return std::nullopt;
  }
  return static_cast<int64_t>(count);
}

static std::optional<int64_t>
getLoopAddressCoefficient(Value value, scf::ForOp loop,
                          DenseMap<Value, int64_t> &cache,
                          DenseSet<Value> &failed) {
  bool isOutside =
      loop.isDefinedOutsideOfLoop(value) || matchPattern(value, m_Constant());
  if (isOutside) {
    return 0;
  }
  if (value == loop.getInductionVar()) {
    return 1;
  }
  if (auto it = cache.find(value); it != cache.end()) {
    return it->second;
  }
  bool isFailed = failed.contains(value) || isa<BlockArgument>(value);
  if (isFailed) {
    return std::nullopt;
  }

  Operation *def = value.getDefiningOp();
  auto coefficient = [&](Value operand) {
    return getLoopAddressCoefficient(operand, loop, cache, failed);
  };
  std::optional<int64_t> result;
  int64_t combined;
  if (auto add = dyn_cast_or_null<arith::AddIOp>(def)) {
    auto lhs = coefficient(add.getLhs());
    auto rhs = coefficient(add.getRhs());
    if (lhs && rhs && !llvm::AddOverflow(*lhs, *rhs, combined)) {
      result = combined;
    }
  } else if (auto sub = dyn_cast_or_null<arith::SubIOp>(def)) {
    auto lhs = coefficient(sub.getLhs());
    auto rhs = coefficient(sub.getRhs());
    if (lhs && rhs && !llvm::SubOverflow(*lhs, *rhs, combined)) {
      result = combined;
    }
  } else if (auto mul = dyn_cast_or_null<arith::MulIOp>(def)) {
    auto lhsConstant = getConstantInt64(mul.getLhs());
    auto rhsConstant = getConstantInt64(mul.getRhs());
    Value varying = lhsConstant ? mul.getRhs() : mul.getLhs();
    std::optional<int64_t> constant = lhsConstant ? lhsConstant : rhsConstant;
    auto varyingCoefficient = constant ? coefficient(varying) : std::nullopt;
    if (constant && varyingCoefficient &&
        !llvm::MulOverflow(*constant, *varyingCoefficient, combined)) {
      result = combined;
    }
  } else if (auto addPtr = dyn_cast_or_null<AddPtrOp>(def)) {
    auto pointer = coefficient(addPtr.getPtr());
    auto offset = coefficient(addPtr.getOffset());
    if (pointer && offset && !llvm::AddOverflow(*pointer, *offset, combined)) {
      result = combined;
    }
  } else if (auto castPtr = dyn_cast_or_null<CastPtrOp>(def)) {
    result = coefficient(castPtr.getInput());
  } else if (isa_and_nonnull<arith::IndexCastOp, arith::IndexCastUIOp>(def)) {
    result = coefficient(def->getOperand(0));
  }

  if (!result) {
    failed.insert(value);
    return std::nullopt;
  }
  cache[value] = *result;
  return result;
}

static bool isLoopContinuousAddress(Value address, int64_t streamAdvance,
                                    scf::ForOp loop) {
  auto tripCount = getLoopTripCount(loop);
  auto step = getConstantInt64(loop.getStep());
  if (!tripCount || *tripCount < 2 || !step) {
    return false;
  }
  DenseMap<Value, int64_t> cache;
  DenseSet<Value> failed;
  auto coefficient = getLoopAddressCoefficient(address, loop, cache, failed);
  int64_t iterationAdvance;
  return coefficient &&
         !llvm::MulOverflow(*coefficient, *step, iterationAdvance) &&
         iterationAdvance == streamAdvance;
}

static Value materializeAtLoopEntry(Value value, scf::ForOp loop,
                                    IRRewriter &rewriter,
                                    DenseMap<Value, Value> &cache) {
  if (loop.isDefinedOutsideOfLoop(value)) {
    return value;
  }
  if (value == loop.getInductionVar()) {
    return loop.getLowerBound();
  }
  if (auto it = cache.find(value); it != cache.end()) {
    return it->second;
  }
  if (isa<BlockArgument>(value)) {
    return nullptr;
  }

  Operation *def = value.getDefiningOp();
  bool canMaterialize = def && def->getParentOp() == loop &&
                        def->getNumRegions() == 0 && isMemoryEffectFree(def);
  if (!canMaterialize) {
    return nullptr;
  }
  IRMapping mapping;
  for (Value operand : def->getOperands()) {
    Value mapped = materializeAtLoopEntry(operand, loop, rewriter, cache);
    if (!mapped) {
      return nullptr;
    }
    mapping.map(operand, mapped);
  }
  Operation *clone = rewriter.clone(*def, mapping);
  for (auto [original, materialized] :
       llvm::zip_equal(def->getResults(), clone->getResults())) {
    cache[original] = materialized;
  }
  return cache.lookup(value);
}

static DenseSet<Operation *>
getStoreStreamOperations(StatefulStoreStream stream) {
  DenseSet<Operation *> operations{stream.init, stream.flush};
  VstusOp store = stream.firstStore;
  while (store) {
    operations.insert(store);
    bool isFinalStore = store.getAlignOut() == stream.finalAlign;
    if (isFinalStore) {
      break;
    }
    store = dyn_cast<VstusOp>(*store.getAlignOut().getUsers().begin());
  }
  return operations;
}

static DenseSet<Operation *>
getLoadStreamOperations(StatefulLoadStream stream) {
  DenseSet<Operation *> operations{stream.init};
  VldusOp load = stream.firstLoad;
  while (load) {
    operations.insert(load);
    if (load == stream.lastLoad) {
      break;
    }
    load = dyn_cast<VldusOp>(*load.getUpdatedAlign().getUsers().begin());
  }
  return operations;
}

static bool hasOtherStatefulOperation(scf::ForOp loop,
                                      const DenseSet<Operation *> &streamOps) {
  bool found = false;
  loop.walk([&](Operation *op) {
    bool isOtherStateful =
        !streamOps.contains(op) &&
        isa<InitAlignOp, VldasOp, VldusOp, VstusOp, VstasOp>(op);
    if (isOtherStateful) {
      found = true;
    }
  });
  return found;
}

static Value getPointerRoot(Value pointer) {
  while (auto addPtr = pointer.getDefiningOp<AddPtrOp>()) {
    pointer = addPtr.getPtr();
  }
  return pointer;
}

static std::optional<int64_t> getStaticPointerRootAddress(Value pointer) {
  pointer = getPointerRoot(pointer);
  auto castPtr = pointer.getDefiningOp<CastPtrOp>();
  if (!castPtr) {
    return std::nullopt;
  }
  Value address = castPtr.getInput();
  if (auto cast = address.getDefiningOp<UnrealizedConversionCastOp>()) {
    bool hasSingleOperand = cast->getNumOperands() == 1;
    if (!hasSingleOperand) {
      return std::nullopt;
    }
    address = cast->getOperand(0);
  }
  return getConstantInt64(address);
}

static bool mayAliasPointerRoots(Value lhs, Value rhs) {
  Value lhsRoot = getPointerRoot(lhs);
  Value rhsRoot = getPointerRoot(rhs);
  if (lhsRoot == rhsRoot) {
    return true;
  }
  auto lhsAddress = getStaticPointerRootAddress(lhsRoot);
  auto rhsAddress = getStaticPointerRootAddress(rhsRoot);
  return !lhsAddress || !rhsAddress || *lhsAddress == *rhsAddress;
}

static bool
hasPotentiallyAliasingEffect(scf::ForOp loop, Value streamAddress,
                             const DenseSet<Operation *> &streamOps) {
  bool found = false;
  loop.walk([&](Operation *op) {
    bool skipOperation =
        found || op == loop || streamOps.contains(op) || isMemoryEffectFree(op);
    if (skipOperation) {
      return;
    }

    bool hasPointerOperand = false;
    for (Value operand : op->getOperands()) {
      if (!isa<PtrType>(operand.getType())) {
        continue;
      }
      hasPointerOperand = true;
      if (mayAliasPointerRoots(streamAddress, operand)) {
        found = true;
        return;
      }
    }
    // Unknown side effects cannot be moved across safely.
    if (!hasPointerOperand) {
      found = true;
    }
  });
  return found;
}

static bool fuseLoopCarriedStoreStream(StatefulStoreStream stream,
                                       IRRewriter &rewriter) {
  auto loop = stream.init->getParentOfType<scf::ForOp>();
  DenseSet<Operation *> streamOps = getStoreStreamOperations(stream);
  bool canFuse =
      loop && stream.init->getParentOp() == loop &&
      isLoopContinuousAddress(stream.initialBase, stream.totalAdvance, loop) &&
      !hasOtherStatefulOperation(loop, streamOps) &&
      !hasPotentiallyAliasingEffect(loop, stream.initialBase, streamOps);
  if (!canFuse) {
    return false;
  }

  rewriter.setInsertionPoint(loop);
  Location loopLoc = loop.getLoc();
  unsigned originalResults = loop.getNumResults();
  DenseMap<Value, Value> cache;
  Value initialBase =
      materializeAtLoopEntry(stream.initialBase, loop, rewriter, cache);
  if (!initialBase) {
    return false;
  }
  Value initialAlign =
      rewriter.create<InitAlignOp>(loopLoc, stream.finalAlign.getType())
          .getResult();
  SmallVector<BlockArgument> newArguments;
  NewYieldValuesFn yields = [&](OpBuilder &, Location,
                                ArrayRef<BlockArgument> arguments) {
    newArguments.assign(arguments.begin(), arguments.end());
    return SmallVector<Value>{stream.finalBase, stream.finalAlign};
  };
  auto replacement = loop.replaceWithAdditionalYields(
      rewriter, ValueRange{initialBase, initialAlign}, false, yields);
  if (failed(replacement)) {
    return false;
  }
  auto newLoop = cast<scf::ForOp>(replacement->getOperation());
  stream.firstStore.getBaseMutable().set(newArguments[0]);
  stream.firstStore.getAlignInMutable().set(newArguments[1]);
  stream.init.erase();
  stream.flush.erase();

  rewriter.setInsertionPointAfter(newLoop);
  Value zero = rewriter.create<arith::ConstantIntOp>(loopLoc, 0, 32);
  rewriter.create<VstasOp>(loopLoc, /*updated_base=*/Type{},
                           newLoop.getResult(originalResults + 1),
                           newLoop.getResult(originalResults), zero);
  return true;
}

static bool fuseLoopCarriedLoadStream(StatefulLoadStream stream,
                                      IRRewriter &rewriter) {
  auto loop = stream.init->getParentOfType<scf::ForOp>();
  DenseSet<Operation *> streamOps = getLoadStreamOperations(stream);
  bool canFuse =
      loop && stream.init->getParentOp() == loop &&
      isLoopContinuousAddress(stream.initialSource, stream.totalElements,
                              loop) &&
      !hasOtherStatefulOperation(loop, streamOps) &&
      !hasPotentiallyAliasingEffect(loop, stream.initialSource, streamOps);
  if (!canFuse) {
    return false;
  }

  rewriter.setInsertionPoint(loop);
  DenseMap<Value, Value> cache;
  Value initialBase =
      materializeAtLoopEntry(stream.initialSource, loop, rewriter, cache);
  if (!initialBase) {
    return false;
  }
  Value initialAlign =
      rewriter
          .create<VldasOp>(loop.getLoc(), stream.finalAlign.getType(),
                           initialBase)
          .getResult();
  SmallVector<BlockArgument> newArguments;
  NewYieldValuesFn yields = [&](OpBuilder &, Location,
                                ArrayRef<BlockArgument> arguments) {
    newArguments.assign(arguments.begin(), arguments.end());
    return SmallVector<Value>{stream.finalBase, stream.finalAlign};
  };
  auto replacement = loop.replaceWithAdditionalYields(
      rewriter, ValueRange{initialBase, initialAlign}, false, yields);
  if (failed(replacement)) {
    return false;
  }
  stream.firstLoad.getSourceMutable().set(newArguments[0]);
  stream.firstLoad.getAlignMutable().set(newArguments[1]);
  stream.init.erase();
  return true;
}

static void fuseLoopCarriedStatefulStreams(ModuleOp module) {
  IRRewriter rewriter(module.getContext());
  while (true) {
    bool changed = false;
    SmallVector<InitAlignOp> storeInitializers;
    module.walk([&](InitAlignOp init) { storeInitializers.push_back(init); });
    for (InitAlignOp init : storeInitializers) {
      auto stream = parseStatefulStoreStream(init);
      if (stream && fuseLoopCarriedStoreStream(*stream, rewriter)) {
        changed = true;
        break;
      }
    }
    if (changed) {
      continue;
    }

    SmallVector<VldasOp> loadInitializers;
    module.walk([&](VldasOp init) { loadInitializers.push_back(init); });
    for (VldasOp init : loadInitializers) {
      auto stream = parseStatefulLoadStream(init);
      if (stream && fuseLoopCarriedLoadStream(*stream, rewriter)) {
        changed = true;
        break;
      }
    }
    if (!changed) {
      return;
    }
  }
}

static void runVPTOStatefulStreamFusionImpl(ModuleOp module) {
  fuseStatefulStoreStreams(module);
  fuseStatefulLoadStreams(module);
  fuseLoopCarriedStatefulStreams(module);
}
} // namespace

namespace mlir::pto {
#define GEN_PASS_DEF_VPTOSTATEFULSTREAMFUSION
#include "PTO/Transforms/Passes.h.inc"

struct VPTOStatefulStreamFusionPass
    : public impl::VPTOStatefulStreamFusionBase<
          VPTOStatefulStreamFusionPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VPTOStatefulStreamFusionPass)

  void runOnOperation() override {
    runVPTOStatefulStreamFusion(getOperation());
  }
};

std::unique_ptr<Pass> createVPTOStatefulStreamFusionPass() {
  return std::make_unique<VPTOStatefulStreamFusionPass>();
}

void runVPTOStatefulStreamFusion(ModuleOp module) {
  runVPTOStatefulStreamFusionImpl(module);
}
} // namespace mlir::pto
