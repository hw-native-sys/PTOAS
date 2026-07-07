// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
namespace func = ::mlir::func;
#define GEN_PASS_DEF_PTOMATERIALIZETILEOPSECTIONS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static constexpr llvm::StringLiteral kTileOpPrimaryDomainAttr =
    "pto.tileop.primary_domain";
static constexpr llvm::StringLiteral kTileOpPhasesAttr = "pto.tileop.phases";

static bool isTileOpSubkernelHelper(func::FuncOp funcOp) {
  return pto::isPTODSLTileOpHelper(funcOp);
}

static bool isTileOpBodyOp(Operation *op) {
  if (!op || isa<func::ReturnOp>(op))
    return false;
  if (op->getName().getDialectNamespace() != PTODialect::getDialectNamespace())
    return false;
  return true;
}

static std::optional<PIPE> getTileOpBodyPipe(Operation *op) {
  if (!isTileOpBodyOp(op))
    return std::nullopt;

  if (auto pipeOp = dyn_cast<OpPipeInterface>(op)) {
    PIPE pipe = pipeOp.getPipe();
    if (pipe != PIPE::PIPE_UNASSIGNED)
      return pipe;
  }

  if (isa<VecScopeOp, StrictVecScopeOp>(op))
    return std::nullopt;

  StringRef name = op->getName().getStringRef();
  if (name.starts_with("pto.v"))
    return PIPE::PIPE_V;
  if (name.starts_with("pto.mad"))
    return PIPE::PIPE_M;
  if (name == "pto.plt_b8" || name == "pto.plt_b16" ||
      name == "pto.plt_b32" || name == "pto.pltm_b8" ||
      name == "pto.pltm_b16" || name == "pto.pltm_b32" ||
      name == "pto.load" || name == "pto.store" || name == "pto.ldg" ||
      name == "pto.stg")
    return PIPE::PIPE_S;
  if (name == "pto.copy_gm_to_ubuf" || name == "pto.mte_gm_ub" ||
      name == "pto.mte_gm_l1" || name == "pto.mte_gm_l1_frac")
    return PIPE::PIPE_MTE2;
  if (name == "pto.mte_ub_gm" || name == "pto.mte_l0c_gm")
    return PIPE::PIPE_MTE3;
  if (name == "pto.mte_l1_l0a" || name == "pto.mte_l1_l0b" ||
      name == "pto.mte_l1_l0a_mx" || name == "pto.mte_l1_l0b_mx")
    return PIPE::PIPE_MTE1;
  return std::nullopt;
}

static bool hasExistingSection(func::FuncOp funcOp) {
  bool found = false;
  funcOp.walk([&](Operation *op) {
    if (isa<SectionCubeOp, SectionVectorOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static bool isPrimaryVectorPipe(PIPE pipe) {
  return pipe == PIPE::PIPE_V || pipe == PIPE::PIPE_V2;
}

static bool isPrimaryCubePipe(PIPE pipe) {
  return pipe == PIPE::PIPE_M;
}

static bool isPrimaryPipeForKind(PIPE pipe, FunctionKernelKind kind) {
  switch (kind) {
  case FunctionKernelKind::Vector:
    return isPrimaryVectorPipe(pipe);
  case FunctionKernelKind::Cube:
    return isPrimaryCubePipe(pipe);
  }
  llvm_unreachable("unexpected kernel kind");
}

static FailureOr<FunctionKernelKind> getPrimaryDomain(func::FuncOp funcOp) {
  auto primaryAttr =
      funcOp->getAttrOfType<FunctionKernelKindAttr>(kTileOpPrimaryDomainAttr);
  if (!primaryAttr)
    return funcOp.emitOpError("requires ")
           << kTileOpPrimaryDomainAttr << " before tileop section materialization";
  return primaryAttr.getKernelKind();
}

static FailureOr<unsigned> findFirstPrimaryPhase(func::FuncOp funcOp,
                                                 ArrayAttr phases,
                                                 FunctionKernelKind kind) {
  for (auto [index, attr] : llvm::enumerate(phases)) {
    auto dict = dyn_cast<DictionaryAttr>(attr);
    auto pipeAttr = dict ? dyn_cast_or_null<PipeAttr>(dict.get("pipe")) : PipeAttr();
    if (!dict || !pipeAttr) {
      return funcOp.emitOpError("expects ")
             << kTileOpPhasesAttr << " entries to carry a pipe attr";
    }
    if (isPrimaryPipeForKind(pipeAttr.getPipe(), kind))
      return index;
  }
  return funcOp.emitOpError("requires at least one primary compute phase in ")
         << kTileOpPhasesAttr;
}

static FailureOr<unsigned> findLastPrimaryPhase(func::FuncOp funcOp,
                                                ArrayAttr phases,
                                                FunctionKernelKind kind) {
  for (int index = static_cast<int>(phases.size()) - 1; index >= 0; --index) {
    auto dict = dyn_cast<DictionaryAttr>(phases[index]);
    auto pipeAttr = dict ? dyn_cast_or_null<PipeAttr>(dict.get("pipe")) : PipeAttr();
    if (!dict || !pipeAttr) {
      return funcOp.emitOpError("expects ")
             << kTileOpPhasesAttr << " entries to carry a pipe attr";
    }
    if (isPrimaryPipeForKind(pipeAttr.getPipe(), kind))
      return static_cast<unsigned>(index);
  }
  return funcOp.emitOpError("requires at least one primary compute phase in ")
         << kTileOpPhasesAttr;
}

static SmallVector<Operation *, 8> collectTileOpBodyOps(Block &block) {
  SmallVector<Operation *, 8> ops;
  for (Operation &op : block.without_terminator()) {
    if (!getTileOpBodyPipe(&op))
      continue;
    ops.push_back(&op);
  }
  return ops;
}

static FailureOr<std::optional<std::pair<unsigned, unsigned>>>
findPrimaryOpRange(func::FuncOp funcOp, ArrayRef<Operation *> ops,
                   FunctionKernelKind kind) {
  int first = -1;
  int last = -1;
  for (auto [index, op] : llvm::enumerate(ops)) {
    std::optional<PIPE> maybePipe = getTileOpBodyPipe(op);
    if (!maybePipe)
      continue;
    PIPE pipe = *maybePipe;
    if (!isPrimaryPipeForKind(pipe, kind))
      continue;
    if (first < 0)
      first = static_cast<int>(index);
    last = static_cast<int>(index);
  }

  if (first < 0 || last < 0)
    return std::optional<std::pair<unsigned, unsigned>>();

  for (int index = first; index <= last; ++index) {
    std::optional<PIPE> maybePipe = getTileOpBodyPipe(ops[index]);
    if (!maybePipe)
      continue;
    PIPE pipe = *maybePipe;
    if (!isPrimaryPipeForKind(pipe, kind)) {
      return ops[index]->emitError()
             << "tileop primary compute span is not contiguous; MVP materializer "
                "supports one contiguous primary-domain span";
    }
  }

  return std::optional<std::pair<unsigned, unsigned>>(
      std::make_pair(static_cast<unsigned>(first),
                     static_cast<unsigned>(last)));
}

static bool isVectorScopeLocalType(Type type) {
  return isa<MaskType, VRegType>(type);
}

static bool hasVectorScopeLocalResult(Operation *op) {
  return llvm::any_of(op->getResultTypes(), isVectorScopeLocalType);
}

static Operation *expandStartForLocalProducers(Block &block, Operation *firstOp,
                                               Operation *lastOp) {
  Operation *expandedFirst = firstOp;
  bool changed = true;
  while (changed) {
    changed = false;
    auto expandedFirstIt = Block::iterator(expandedFirst);
    auto afterLastIt = std::next(Block::iterator(lastOp));
    for (auto it = expandedFirstIt; it != afterLastIt; ++it) {
      Operation &op = *it;
      for (Value operand : op.getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def || def->getBlock() != &block || !def->isBeforeInBlock(expandedFirst))
          continue;
        if (!hasVectorScopeLocalResult(def))
          continue;
        expandedFirst = def;
        changed = true;
        break;
      }
      if (changed)
        break;
    }
  }
  return expandedFirst;
}

template <typename SectionOpT>
static void wrapOperationRange(Block &block, Operation *firstOp,
                               Operation *lastOp) {
  OpBuilder builder(firstOp);
  auto sectionOp = builder.create<SectionOpT>(firstOp->getLoc());
  if (!sectionOp.getBody().empty())
    sectionOp.getBody().dropAllReferences();
  while (!sectionOp.getBody().empty())
    delete &sectionOp.getBody().front();
  sectionOp.getBody().push_back(new Block());
  Block &sectionBlock = sectionOp.getBody().front();

  auto firstIt = Block::iterator(firstOp);
  auto afterLastIt = std::next(Block::iterator(lastOp));
  sectionBlock.getOperations().splice(sectionBlock.end(),
                                      block.getOperations(), firstIt,
                                      afterLastIt);
}

static LogicalResult materializePrimarySectionsInBlock(func::FuncOp funcOp,
                                                       Block &block,
                                                       FunctionKernelKind kind,
                                                       bool &materializedAny) {
  SmallVector<Operation *, 16> topLevelOps;
  for (Operation &op : block.without_terminator())
    topLevelOps.push_back(&op);

  SmallVector<Operation *, 8> bodyOps = collectTileOpBodyOps(block);
  if (!bodyOps.empty()) {
    FailureOr<std::optional<std::pair<unsigned, unsigned>>> primaryRange =
        findPrimaryOpRange(funcOp, bodyOps, kind);
    if (failed(primaryRange))
      return failure();

    if (*primaryRange) {
      Operation *firstPrimaryOp = bodyOps[(*primaryRange)->first];
      Operation *lastPrimaryOp = bodyOps[(*primaryRange)->second];
      firstPrimaryOp =
          expandStartForLocalProducers(block, firstPrimaryOp, lastPrimaryOp);
      switch (kind) {
      case FunctionKernelKind::Vector:
        wrapOperationRange<SectionVectorOp>(block, firstPrimaryOp,
                                            lastPrimaryOp);
        break;
      case FunctionKernelKind::Cube:
        wrapOperationRange<SectionCubeOp>(block, firstPrimaryOp, lastPrimaryOp);
        break;
      }
      materializedAny = true;
    }
  }

  for (Operation *op : topLevelOps) {
    if (!op || getTileOpBodyPipe(op))
      continue;
    for (Region &region : op->getRegions())
      for (Block &nestedBlock : region)
        if (failed(materializePrimarySectionsInBlock(funcOp, nestedBlock, kind,
                                                     materializedAny)))
          return failure();
  }
  return success();
}

static LogicalResult materializeTileOpSection(func::FuncOp funcOp) {
  if (!isTileOpSubkernelHelper(funcOp) || funcOp.isDeclaration())
    return success();

  if (hasExistingSection(funcOp))
    return success();

  auto phases = funcOp->getAttrOfType<ArrayAttr>(kTileOpPhasesAttr);
  if (!phases || phases.empty())
    return success();

  FailureOr<FunctionKernelKind> primaryDomain = getPrimaryDomain(funcOp);
  if (failed(primaryDomain))
    return failure();

  FailureOr<unsigned> firstPhase =
      findFirstPrimaryPhase(funcOp, phases, *primaryDomain);
  if (failed(firstPhase))
    return failure();
  FailureOr<unsigned> lastPhase =
      findLastPrimaryPhase(funcOp, phases, *primaryDomain);
  if (failed(lastPhase))
    return failure();
  (void)firstPhase;
  (void)lastPhase;

  bool materializedAny = false;
  if (failed(materializePrimarySectionsInBlock(funcOp, funcOp.getBody().front(),
                                               *primaryDomain, materializedAny)))
    return failure();
  if (!materializedAny) {
    return funcOp.emitOpError(
        "requires at least one primary compute op in helper body");
  }
  return success();
}

struct PTOMaterializeTileOpSectionsPass
    : public mlir::pto::impl::PTOMaterializeTileOpSectionsBase<
          PTOMaterializeTileOpSectionsPass> {
  void runOnOperation() override {
    if (failed(materializeTileOpSection(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOMaterializeTileOpSectionsPass() {
  return std::make_unique<PTOMaterializeTileOpSectionsPass>();
}
