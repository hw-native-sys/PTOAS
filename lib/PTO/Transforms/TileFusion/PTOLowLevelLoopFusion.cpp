// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdlib>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOLOWLEVELLOOPFUSION
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

struct LoopLevelInfo {
  scf::ForOp loop;
  SmallVector<Operation *, mlir::pto::kValue4> preludeOps;
  SmallVector<Operation *, mlir::pto::kValue4> epilogueOps;
};

struct StageInfo {
  SmallVector<Operation *, mlir::pto::kValue4> setupOps;
  SmallVector<LoopLevelInfo, mlir::pto::kValue4> levels;
  SmallVector<Operation *, mlir::pto::kValue8> leafOps;

  scf::ForOp getOuterLoop() const { return levels.front().loop; }
  unsigned getDepth() const { return levels.size(); }
};

static bool areEquivalentValues(Value lhs, Value rhs);

static Value mapValueOrSelf(Value value, IRMapping &mapping) {
  return mapping.lookupOrDefault(value);
}

static bool sameForHeader(scf::ForOp lhs, scf::ForOp rhs) {
  return areEquivalentValues(lhs.getLowerBound(), rhs.getLowerBound()) &&
         areEquivalentValues(lhs.getUpperBound(), rhs.getUpperBound()) &&
         areEquivalentValues(lhs.getStep(), rhs.getStep()) &&
         lhs->getAttrs() == rhs->getAttrs();
}

static bool isPureNoRegionOp(Operation *op) {
  return op->getNumRegions() == 0 && isMemoryEffectFree(op);
}

static bool isMovableMemoryPreludeOp(Operation *op) {
  return op->getNumRegions() == 0 && isa<MemoryEffectOpInterface>(op);
}

static bool isSupportedPreludeOp(Operation *op) {
  return isPureNoRegionOp(op) || isMovableMemoryPreludeOp(op);
}

static bool isSupportedLeafOp(Operation *op) { return op->getNumRegions() == 0; }

static bool isInterstageSetupOp(Operation *op) {
  if (isPureNoRegionOp(op)) {
    return true;
  }

  return false;
}

static bool areEquivalentOperations(Operation *lhs, Operation *rhs) {
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->getName() != rhs->getName()) {
    return false;
  }
  if (lhs->getNumRegions() != 0 || rhs->getNumRegions() != 0) {
    return false;
  }
  if (lhs->getNumResults() != rhs->getNumResults()) {
    return false;
  }
  if (lhs->getNumOperands() != rhs->getNumOperands()) {
    return false;
  }
  if (lhs->getAttrDictionary() != rhs->getAttrDictionary()) {
    return false;
  }
  if (!llvm::equal(lhs->getResultTypes(), rhs->getResultTypes())) {
    return false;
  }

  if (auto lhsDim = dyn_cast<memref::DimOp>(lhs)) {
    auto rhsDim = cast<memref::DimOp>(rhs);
    return lhsDim.getSource().getType() == rhsDim.getSource().getType() &&
           areEquivalentValues(lhsDim.getIndex(), rhsDim.getIndex());
  }

  for (auto [lhsOperand, rhsOperand] :
       llvm::zip(lhs->getOperands(), rhs->getOperands())) {
    if (!areEquivalentValues(lhsOperand, rhsOperand)) {
      return false;
    }
  }
  return true;
}

static bool areEquivalentValues(Value lhs, Value rhs) {
  if (lhs == rhs) {
    return true;
  }
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs.getType() != rhs.getType()) {
    return false;
  }

  auto lhsArg = dyn_cast<BlockArgument>(lhs);
  auto rhsArg = dyn_cast<BlockArgument>(rhs);
  if (lhsArg || rhsArg) {
    return lhsArg && rhsArg && lhsArg.getOwner() == rhsArg.getOwner() &&
           lhsArg.getArgNumber() == rhsArg.getArgNumber();
  }

  return areEquivalentOperations(lhs.getDefiningOp(), rhs.getDefiningOp());
}

static Value traceBlockArgumentAlias(Value value) {
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg) {
    return {};
  }
  auto forOp = dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp());
  bool hasMatchingInit = forOp && arg.getArgNumber() > 0 &&
                         forOp.getInitArgs().size() >= arg.getArgNumber();
  if (hasMatchingInit) {
    return forOp.getInitArgs()[arg.getArgNumber() - 1];
  }
  return {};
}

static Value traceMemRefAlias(Operation *def) {
  if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
    return subview.getSource();
  }
  if (auto cast = dyn_cast<memref::CastOp>(def)) {
    return cast.getSource();
  }
  if (auto cast = dyn_cast<memref::ReinterpretCastOp>(def)) {
    return cast.getSource();
  }
  if (auto cast = dyn_cast<memref::MemorySpaceCastOp>(def)) {
    return cast.getSource();
  }
  if (auto transpose = dyn_cast<memref::TransposeOp>(def)) {
    return transpose.getIn();
  }
  return {};
}

static Value tracePTOAlias(Operation *def) {
  if (auto subview = dyn_cast<pto::SubViewOp>(def)) {
    return subview.getSource();
  }
  if (auto bitcast = dyn_cast<pto::BitcastOp>(def)) {
    return bitcast.getSrc();
  }
  if (auto reshape = dyn_cast<pto::TReshapeOp>(def)) {
    return reshape.getSrc();
  }
  return {};
}

static Value traceUnrealizedAlias(Value value,
                                  UnrealizedConversionCastOp cast) {
  if (cast.getInputs().empty()) {
    return {};
  }
  if (auto result = dyn_cast<OpResult>(value)) {
    unsigned resultNumber = result.getResultNumber();
    if (resultNumber < cast.getInputs().size()) {
      return cast.getInputs()[resultNumber];
    }
  }
  return cast.getInputs().size() == 1 ? cast.getInputs().front() : Value();
}

static Value traceLoopResultAlias(Value value, scf::ForOp forOp) {
  auto result = dyn_cast<OpResult>(value);
  bool hasMatchingInit =
      result && result.getResultNumber() < forOp.getInitArgs().size();
  if (hasMatchingInit) {
    return forOp.getInitArgs()[result.getResultNumber()];
  }
  return {};
}

static Value traceAliasRootOneStep(Value value) {
  if (Value alias = traceBlockArgumentAlias(value)) {
    return alias;
  }

  Operation *def = value.getDefiningOp();
  if (!def) {
    return {};
  }
  if (Value alias = traceMemRefAlias(def)) {
    return alias;
  }
  if (Value alias = tracePTOAlias(def)) {
    return alias;
  }
  if (auto cast = dyn_cast<UnrealizedConversionCastOp>(def)) {
    return traceUnrealizedAlias(value, cast);
  }
  if (auto forOp = dyn_cast<scf::ForOp>(def)) {
    return traceLoopResultAlias(value, forOp);
  }
  return {};
}

static Value traceAliasRoot(Value value) {
  int loopBound = 256;
  while (value) {
    Value upward = traceAliasRootOneStep(value);
    if (!upward) {
      break;
    }
    value = upward;
    if (loopBound-- <= 0) {
      break;
    }
  }
  return value;
}

static LogicalResult collectAliasRelevantRoots(
    Operation *op, SmallVectorImpl<Value> &roots) {
  if (isMemoryEffectFree(op)) {
    return success();
  }

  auto effectsOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectsOp) {
    return failure();
  }

  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, mlir::pto::kValue4> effects;
  effectsOp.getEffects(effects);
  for (const auto &effect : effects) {
    Value effectValue = effect.getValue();
    if (!effectValue) {
      return failure();
    }

    Type effectType = effectValue.getType();
    if (!isa<BaseMemRefType, pto::PtrType>(effectType)) {
      if (isa<MemoryEffects::Write>(effect.getEffect())) {
        return failure();
      }
      continue;
    }

    Value root = traceAliasRoot(effectValue);
    if (!root) {
      return failure();
    }
    roots.push_back(root);
  }
  return success();
}

static bool containsEquivalentRoot(ArrayRef<Value> roots, Value candidate) {
  return llvm::any_of(roots, [&](Value root) {
    return areEquivalentValues(root, candidate);
  });
}

/// Check that \p movableOp has no aliasing conflict with the memory ops in
/// \p crossStages.  \p crossStages are stages whose leaf and epilogue ops the
/// movableOp would be reordered across in the fused loop.
static bool canMoveAcrossOperations(Operation *movableOp,
                                    ArrayRef<Value> movableRoots,
                                    ArrayRef<Operation *> crossedOps,
                                    StringRef crossedKind,
                                    llvm::raw_ostream *debugOS) {
  for (Operation *op : crossedOps) {
    SmallVector<Value, mlir::pto::kValue4> opRoots;
    if (failed(collectAliasRelevantRoots(op, opRoots))) {
      if (debugOS) {
        *debugOS << "[op-fusion] reject movable op " << movableOp->getName()
                 << " at " << movableOp->getLoc() << ": crossed effects of "
                 << op->getName() << " are not alias-analyzable\n";
      }
      return false;
    }
    if (llvm::any_of(movableRoots, [&](Value root) {
          return containsEquivalentRoot(opRoots, root);
        })) {
      if (debugOS) {
        *debugOS << "[op-fusion] reject movable op " << movableOp->getName()
                 << " at " << movableOp->getLoc()
                 << ": touched root may alias a crossed " << crossedKind
                 << "\n";
      }
      return false;
    }
  }
  return true;
}

static bool canMoveAcrossStages(Operation *movableOp,
                                ArrayRef<StageInfo> crossStages,
                                llvm::raw_ostream *debugOS) {
  SmallVector<Value, mlir::pto::kValue4> roots;
  if (failed(collectAliasRelevantRoots(movableOp, roots))) {
    if (debugOS) {
      *debugOS << "[op-fusion] reject movable op " << movableOp->getName()
               << " at " << movableOp->getLoc()
               << ": touched roots are not alias-analyzable\n";
    }
    return false;
  }
  for (const StageInfo &crossStage : crossStages) {
    if (!canMoveAcrossOperations(movableOp, roots, crossStage.leafOps,
                                 "stage memory op", debugOS)) {
      return false;
    }
    for (const LoopLevelInfo &level : crossStage.levels) {
      if (!canMoveAcrossOperations(movableOp, roots, level.epilogueOps,
                                   "stage epilogue op", debugOS)) {
        return false;
      }
    }
  }

  return true;
}

static bool arePreludeReordersLegal(ArrayRef<StageInfo> stages,
                                    llvm::raw_ostream *debugOS) {
  for (size_t stageIndex = 1; stageIndex < stages.size(); ++stageIndex) {
    ArrayRef<StageInfo> priorStages(stages.data(), stageIndex);

    // Prelude ops of the current stage are moved before all prior-stage
    // leaf and epilogue ops in the fused loop.  Check they don't alias.
    for (const LoopLevelInfo &level : stages[stageIndex].levels) {
      for (Operation *op : level.preludeOps) {
        if (!canMoveAcrossStages(op, priorStages, debugOS)) {
          return false;
        }
      }
    }

    // Epilogue ops of prior stages are reordered to execute after the
    // current stage's leaf ops in the fused loop.  Check that each prior
    // stage's epilogue ops don't alias the current stage's leaf ops.
    for (const StageInfo &priorStage : priorStages) {
      for (const LoopLevelInfo &priorLevel : priorStage.levels) {
        for (Operation *epilogueOp : priorLevel.epilogueOps) {
          if (!canMoveAcrossStages(epilogueOp,
                                   ArrayRef<StageInfo>(&stages[stageIndex], 1),
                                   debugOS)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

static LogicalResult analyzeStage(scf::ForOp outerLoop, StageInfo &stage) {
  scf::ForOp currentLoop = outerLoop;
  while (currentLoop) {
    stage.levels.push_back(LoopLevelInfo{currentLoop, {}, {}});
    LoopLevelInfo &currentLevel = stage.levels.back();

    SmallVector<Operation *, mlir::pto::kValue8> bodyOps;
    scf::ForOp childLoop;
    for (Operation &op : currentLoop.getBody()->without_terminator()) {
      bodyOps.push_back(&op);
      if (auto nestedLoop = dyn_cast<scf::ForOp>(op)) {
        if (childLoop) {
          return failure();
        }
        childLoop = nestedLoop;
      }
    }

    if (!childLoop) {
      for (Operation *op : bodyOps) {
        if (!isSupportedLeafOp(op)) {
          return failure();
        }
        stage.leafOps.push_back(op);
      }
      return failure(stage.leafOps.empty());
    }

    bool seenChildLoop = false;
    for (Operation *op : bodyOps) {
      if (op == childLoop.getOperation()) {
        seenChildLoop = true;
        continue;
      }
      if (!seenChildLoop) {
        // Ops before the child loop are prelude ops.
        if (!isSupportedPreludeOp(op)) {
          return failure();
        }
        currentLevel.preludeOps.push_back(op);
      } else {
        // Ops after the child loop are epilogue ops (e.g. row-reduction
        // result stores in trowmax/trowsum).  They must be supported
        // leaf-like ops (no regions) so we can clone them into the fused
        // loop after all inner body ops.
        if (!isSupportedPreludeOp(op)) {
          return failure();
        }
        currentLevel.epilogueOps.push_back(op);
      }
    }

    currentLoop = childLoop;
  }

  return failure();
}

static bool appendStage(scf::ForOp loop,
                        SmallVectorImpl<Operation *> &pendingSetup,
                        SmallVectorImpl<StageInfo> &stages,
                        llvm::raw_ostream *debugOS) {
  StageInfo stage;
  stage.setupOps.assign(pendingSetup.begin(), pendingSetup.end());
  pendingSetup.clear();
  if (succeeded(analyzeStage(loop, stage))) {
    stages.push_back(std::move(stage));
    return true;
  }
  if (debugOS) {
    *debugOS << "[op-fusion] stop stage run before " << loop.getLoc()
             << ": next stage analysis failed\n";
  }
  return false;
}

static SmallVector<StageInfo, mlir::pto::kValue8> collectStageRunFrom(scf::ForOp firstLoop,
                                                     llvm::raw_ostream *debugOS) {
  SmallVector<StageInfo, mlir::pto::kValue8> stages;

  StageInfo firstStage;
  if (failed(analyzeStage(firstLoop, firstStage))) {
    if (debugOS) {
      *debugOS << "[op-fusion] reject loop stage at " << firstLoop.getLoc()
               << ": stage analysis failed\n";
    }
    return stages;
  }
  stages.push_back(std::move(firstStage));

  SmallVector<Operation *, mlir::pto::kValue4> pendingSetup;
  for (Operation *op = firstLoop->getNextNode(); op; op = op->getNextNode()) {
    if (auto nextLoop = dyn_cast<scf::ForOp>(op)) {
      if (!appendStage(nextLoop, pendingSetup, stages, debugOS)) {
        break;
      }
      continue;
    }

    if (!isInterstageSetupOp(op)) {
      if (debugOS) {
        *debugOS << "[op-fusion] stop stage run at op " << op->getName()
                 << "\n";
      }
      break;
    }
    pendingSetup.push_back(op);
  }

  return stages;
}

static bool sameLoopNestShape(const StageInfo &lhs, const StageInfo &rhs) {
  if (lhs.getDepth() != rhs.getDepth()) {
    return false;
  }
  return llvm::all_of(llvm::zip(lhs.levels, rhs.levels), [](auto pair) {
    return sameForHeader(std::get<0>(pair).loop, std::get<1>(pair).loop);
  });
}

static void cloneOpAndMapResults(OpBuilder &builder, Operation *op,
                                 IRMapping &mapping) {
  Operation *cloned = builder.clone(*op, mapping);
  for (auto [oldRes, newRes] :
       llvm::zip(op->getResults(), cloned->getResults())) {
    mapping.map(oldRes, newRes);
  }
}

static void appendMappedValues(ValueRange values, IRMapping &mapping,
                               SmallVectorImpl<Value> &mappedValues) {
  for (Value value : values) {
    mappedValues.push_back(mapValueOrSelf(value, mapping));
  }
}

static void mapFusedLoopArguments(scf::ForOp fusedLoop,
                                  MutableArrayRef<StageInfo> stages,
                                  MutableArrayRef<IRMapping> mappings,
                                  unsigned levelIndex) {
  unsigned iterArgOffset = 0;
  for (auto [stageIndex, stage] : llvm::enumerate(stages)) {
    scf::ForOp originalLoop = stage.levels[levelIndex].loop;
    mappings[stageIndex].map(originalLoop.getInductionVar(),
                             fusedLoop.getInductionVar());
    for (auto [argIndex, originalArg] :
         llvm::enumerate(originalLoop.getRegionIterArgs())) {
      mappings[stageIndex].map(
          originalArg, fusedLoop.getRegionIterArgs()[iterArgOffset + argIndex]);
    }
    iterArgOffset += originalLoop.getRegionIterArgs().size();
  }
}

static SmallVector<Value, mlir::pto::kValue8>
collectFusedYieldOperands(MutableArrayRef<StageInfo> stages,
                          MutableArrayRef<IRMapping> mappings,
                          unsigned levelIndex) {
  SmallVector<Value, mlir::pto::kValue8> operands;
  for (auto [stageIndex, stage] : llvm::enumerate(stages)) {
    auto originalYield = cast<scf::YieldOp>(
        stage.levels[levelIndex].loop.getBody()->getTerminator());
    appendMappedValues(ValueRange(originalYield.getOperands()),
                       mappings[stageIndex], operands);
  }
  return operands;
}

static void setFusedYieldOperands(scf::ForOp fusedLoop, Location location,
                                  ValueRange operands) {
  Block *body = fusedLoop.getBody();
  Operation *terminator =
      !body->empty() && body->back().hasTrait<OpTrait::IsTerminator>()
          ? &body->back()
          : nullptr;
  if (auto yield = dyn_cast_or_null<scf::YieldOp>(terminator)) {
    yield->setOperands(operands);
    return;
  }
  OpBuilder::atBlockEnd(body).create<scf::YieldOp>(location, operands);
}

static void mapFusedLoopResults(scf::ForOp fusedLoop,
                                MutableArrayRef<StageInfo> stages,
                                MutableArrayRef<IRMapping> mappings,
                                unsigned levelIndex) {
  unsigned resultOffset = 0;
  for (auto [stageIndex, stage] : llvm::enumerate(stages)) {
    scf::ForOp originalLoop = stage.levels[levelIndex].loop;
    for (Value originalResult : originalLoop.getResults()) {
      mappings[stageIndex].map(originalResult,
                               fusedLoop.getResults()[resultOffset++]);
    }
  }
}

static scf::ForOp buildFusedLoopNestAtLevel(OpBuilder &builder,
                                            MutableArrayRef<StageInfo> stages,
                                            MutableArrayRef<IRMapping> mappings,
                                            unsigned levelIndex) {
  scf::ForOp firstLoop = stages.front().levels[levelIndex].loop;

  SmallVector<Value, mlir::pto::kValue8> fusedInitArgs;
  for (auto [stageIndex, stage] : llvm::enumerate(stages)) {
    appendMappedValues(ValueRange(stage.levels[levelIndex].loop.getInitArgs()),
                       mappings[stageIndex], fusedInitArgs);
  }

  auto fusedLoop = builder.create<scf::ForOp>(
      firstLoop.getLoc(),
      mapValueOrSelf(firstLoop.getLowerBound(), mappings.front()),
      mapValueOrSelf(firstLoop.getUpperBound(), mappings.front()),
      mapValueOrSelf(firstLoop.getStep(), mappings.front()), fusedInitArgs);
  fusedLoop->setAttrs(firstLoop->getAttrs());
  mapFusedLoopArguments(fusedLoop, stages, mappings, levelIndex);

  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(fusedLoop.getBody());
  for (auto [stageIndex, stage] : llvm::enumerate(stages)) {
    for (Operation *op : stage.levels[levelIndex].preludeOps) {
      cloneOpAndMapResults(bodyBuilder, op, mappings[stageIndex]);
    }
  }

  if (levelIndex + 1 < stages.front().getDepth()) {
    (void)buildFusedLoopNestAtLevel(bodyBuilder, stages, mappings,
                                    levelIndex + 1);
  } else {
    for (auto [stageIndex, stage] : llvm::enumerate(stages)) {
      for (Operation *op : stage.leafOps) {
        cloneOpAndMapResults(bodyBuilder, op, mappings[stageIndex]);
      }
    }
  }

  // Clone epilogue ops after the inner loop / leaf ops for each stage.
  // Epilogue ops appear after the child loop in the original stage and
  // must come after all inner-level body ops in the fused loop too.
  for (auto [stageIndex, stage] : llvm::enumerate(stages)) {
    for (Operation *op : stage.levels[levelIndex].epilogueOps) {
      cloneOpAndMapResults(bodyBuilder, op, mappings[stageIndex]);
    }
  }

  SmallVector<Value, mlir::pto::kValue8> fusedYieldOperands =
      collectFusedYieldOperands(stages, mappings, levelIndex);
  setFusedYieldOperands(fusedLoop, firstLoop.getLoc(), fusedYieldOperands);
  mapFusedLoopResults(fusedLoop, stages, mappings, levelIndex);

  return fusedLoop;
}

static bool fuseStageRun(SmallVectorImpl<StageInfo> &stages,
                         llvm::raw_ostream *debugOS) {
  if (stages.size() < mlir::pto::kValue2) {
    if (debugOS) {
      *debugOS << "[op-fusion] reject loop run: need at least 2 stages, got "
               << stages.size() << "\n";
    }
    return false;
  }

  StageInfo &first = stages.front();
  for (StageInfo &stage : llvm::drop_begin(stages)) {
    if (!sameLoopNestShape(first, stage)) {
      if (debugOS) {
        *debugOS << "[op-fusion] reject loop run: loop nest shape mismatch\n";
      }
      return false;
    }
  }
  if (!arePreludeReordersLegal(stages, debugOS)) {
    return false;
  }

  OpBuilder blockBuilder(first.getOuterLoop());
  SmallVector<IRMapping, mlir::pto::kValue8> stageMappings(stages.size());
  auto fusedOuterLoop =
      buildFusedLoopNestAtLevel(blockBuilder, stages, stageMappings, 0);

  for (StageInfo &stage : llvm::drop_begin(stages)) {
    for (Operation *setupOp : stage.setupOps) {
      setupOp->moveBefore(fusedOuterLoop);
    }
  }

  for (StageInfo &stage : llvm::reverse(stages)) {
    stage.getOuterLoop().erase();
  }

  return true;
}

static bool fuseStageRunsInBlock(Block &block, llvm::raw_ostream *debugOS) {
  bool changed = false;
  bool localChange = true;

  while (localChange) {
    localChange = false;
    for (Operation &op : block) {
      auto firstLoop = dyn_cast<scf::ForOp>(op);
      if (!firstLoop) {
        continue;
      }

      SmallVector<StageInfo, mlir::pto::kValue8> stages =
          collectStageRunFrom(firstLoop, debugOS);
      if (!fuseStageRun(stages, debugOS)) {
        continue;
      }

      changed = true;
      localChange = true;
      break;
    }
  }

  return changed;
}

struct PTOLowLevelLoopFusionPass
    : public pto::impl::PTOLowLevelLoopFusionBase<
          PTOLowLevelLoopFusionPass> {
  using pto::impl::PTOLowLevelLoopFusionBase<
      PTOLowLevelLoopFusionPass>::PTOLowLevelLoopFusionBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    const bool traceEnabled =
        debug || (std::getenv("PTO_LL_LOOP_FUSION_TRACE") != nullptr);
    llvm::raw_ostream *traceOS = traceEnabled ? &llvm::errs() : nullptr;

    int fusedFuncs = 0;
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (func.isExternal()) {
        continue;
      }
      if (func.getSymName().starts_with("__pto_oplib_")) {
        continue;
      }
      if (func.empty()) {
        continue;
      }

      bool changed = false;
      func.walk([&](pto::FusionRegionOp fusionRegion) {
        changed |= fuseStageRunsInBlock(fusionRegion.getBody().front(), traceOS);
      });
      if (changed) {
        ++fusedFuncs;
      }
    }

    if (traceEnabled) {
      llvm::errs() << "[op-fusion] low-level loop fusion changed " << fusedFuncs
                   << " function(s)\n";
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOLowLevelLoopFusionPass(
    const PTOLowLevelLoopFusionOptions &options) {
  return std::make_unique<PTOLowLevelLoopFusionPass>(options);
}
