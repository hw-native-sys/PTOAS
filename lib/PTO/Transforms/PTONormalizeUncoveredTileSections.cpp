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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTONORMALIZEUNCOVEREDTILESECTIONS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

enum class InferredSectionKind {
  Vector,
  Cube,
};

struct UncoveredTopLevelSegment {
  Operation *firstOp = nullptr;
  Operation *lastOp = nullptr;
  Operation *firstTileCarrierOp = nullptr;
  bool containsTileOp = false;
  bool containsNestedExplicitSection = false;
  unsigned vectorTileOpCount = 0;
  unsigned cubeTileOpCount = 0;
  SmallVector<Operation *, 4> ambiguousTileOps;
};

static void mergeSegmentSummary(UncoveredTopLevelSegment &dst,
                                const UncoveredTopLevelSegment &src) {
  if (!src.firstOp)
    return;
  if (!dst.firstOp)
    dst.firstOp = src.firstOp;
  dst.lastOp = src.lastOp;
  if (!dst.firstTileCarrierOp)
    dst.firstTileCarrierOp = src.firstTileCarrierOp;
  dst.containsTileOp |= src.containsTileOp;
  dst.containsNestedExplicitSection |= src.containsNestedExplicitSection;
  dst.vectorTileOpCount += src.vectorTileOpCount;
  dst.cubeTileOpCount += src.cubeTileOpCount;
  dst.ambiguousTileOps.append(src.ambiguousTileOps.begin(),
                              src.ambiguousTileOps.end());
}

static bool isExplicitSection(Operation *op) {
  return isa<SectionCubeOp, SectionVectorOp>(op);
}

static bool isTileLikeOp(Operation *op) {
  if (!op)
    return false;
  return isa<OpPipeInterface>(op) &&
         op->getName().getStringRef().starts_with("pto.t");
}

static bool isPipeLikeOp(Operation *op) {
  return op && isa<OpPipeInterface>(op);
}

static bool isRawVPTOVectorTransientType(Type type) {
  return isa<VRegType, MaskType, AlignType>(type);
}

static bool isRawVPTOVectorLikeOp(Operation *op) {
  if (!op)
    return false;
  for (Value operand : op->getOperands()) {
    if (isRawVPTOVectorTransientType(operand.getType()))
      return true;
  }
  for (Value result : op->getResults()) {
    if (isRawVPTOVectorTransientType(result.getType()))
      return true;
  }
  return false;
}

static bool hasAnySection(func::FuncOp funcOp) {
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

static bool hasExplicitFunctionKernelKind(func::FuncOp funcOp) {
  return funcOp &&
         funcOp->hasAttrOfType<FunctionKernelKindAttr>(FunctionKernelKindAttr::name);
}

static bool isTileOpSubkernelHelper(func::FuncOp funcOp) {
  return pto::isPTODSLTileOpHelper(funcOp);
}

static bool isInsideKernelKindModule(func::FuncOp funcOp) {
  if (!funcOp)
    return false;
  ModuleOp owner = funcOp->getParentOfType<ModuleOp>();
  return owner && owner->hasAttr(FunctionKernelKindAttr::name);
}

static bool hasKnownKernelKindContext(func::FuncOp funcOp) {
  return isInsideKernelKindModule(funcOp) ||
         hasExplicitFunctionKernelKind(funcOp) ||
         isTileOpSubkernelHelper(funcOp);
}

static std::optional<AddressSpace> getBufferAddressSpace(Type type) {
  if (auto tileType = dyn_cast<TileBufType>(type)) {
    if (auto attr =
            dyn_cast_or_null<AddressSpaceAttr>(tileType.getMemorySpace()))
      return attr.getAddressSpace();
    return std::nullopt;
  }
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    if (auto attr =
            dyn_cast_or_null<AddressSpaceAttr>(memrefType.getMemorySpace()))
      return attr.getAddressSpace();
    return std::nullopt;
  }
  return std::nullopt;
}

static void collectTileAddressSpaces(Type type,
                                     SmallVectorImpl<AddressSpace> &spaces) {
  if (std::optional<AddressSpace> addressSpace = getBufferAddressSpace(type))
    spaces.push_back(*addressSpace);
}

static std::optional<int8_t> getPipeHandleDirMask(Value pipeHandle) {
  if (!pipeHandle)
    return std::nullopt;
  if (auto init = pipeHandle.getDefiningOp<InitializeL2LPipeOp>())
    return init.getDirMask();
  if (auto init = pipeHandle.getDefiningOp<InitializeL2G2LPipeOp>())
    return init.getDirMask();
  return std::nullopt;
}

static std::optional<InferredSectionKind>
classifyTileSectionByAddressSpace(std::optional<AddressSpace> space) {
  if (!space)
    return std::nullopt;

  switch (*space) {
  case AddressSpace::VEC:
    return InferredSectionKind::Vector;
  case AddressSpace::MAT:
  case AddressSpace::LEFT:
  case AddressSpace::RIGHT:
  case AddressSpace::ACC:
  case AddressSpace::BIAS:
  case AddressSpace::SCALING:
    return InferredSectionKind::Cube;
  default:
    return std::nullopt;
  }
}

static std::optional<InferredSectionKind>
classifyInternalPipeTileOp(Operation *op) {
  if (auto push = dyn_cast<TPushOp>(op)) {
    std::optional<int8_t> dirMask = getPipeHandleDirMask(push.getPipeHandle());
    if (!dirMask)
      return std::nullopt;
    if (*dirMask == 1)
      return InferredSectionKind::Cube;
    if (*dirMask == 2)
      return InferredSectionKind::Vector;
    return classifyTileSectionByAddressSpace(
        getBufferAddressSpace(push.getTile().getType()));
  }

  if (auto pop = dyn_cast<TPopOp>(op)) {
    std::optional<int8_t> dirMask = getPipeHandleDirMask(pop.getPipeHandle());
    if (!dirMask)
      return std::nullopt;
    if (*dirMask == 1)
      return InferredSectionKind::Vector;
    if (*dirMask == 2)
      return InferredSectionKind::Cube;
    return classifyTileSectionByAddressSpace(
        getBufferAddressSpace(pop.getTile().getType()));
  }

  if (auto free = dyn_cast<TFreeOp>(op)) {
    std::optional<int8_t> dirMask = getPipeHandleDirMask(free.getPipeHandle());
    if (!dirMask)
      return std::nullopt;
    if (*dirMask == 1)
      return InferredSectionKind::Vector;
    if (*dirMask == 2)
      return InferredSectionKind::Cube;
    if (!free.getEntry())
      return std::nullopt;
    return classifyTileSectionByAddressSpace(
        getBufferAddressSpace(free.getEntry().getType()));
  }

  if (auto alloc = dyn_cast<TAllocOp>(op)) {
    std::optional<int8_t> dirMask = getPipeHandleDirMask(alloc.getPipeHandle());
    if (!dirMask)
      return std::nullopt;
    if (*dirMask == 1)
      return InferredSectionKind::Cube;
    if (*dirMask == 2)
      return InferredSectionKind::Vector;
    return std::nullopt;
  }

  return std::nullopt;
}

static std::optional<InferredSectionKind> classifyTileOpByName(Operation *op) {
  StringRef name = op->getName().getStringRef();
  if (name.starts_with("pto.tmatmul") || name.starts_with("pto.tgemv"))
    return InferredSectionKind::Cube;
  if (name == "pto.talloc_to_aiv" || name == "pto.tpush_to_aic" ||
      name == "pto.tpop_from_aic" || name == "pto.tfree_from_aic")
    return InferredSectionKind::Vector;
  if (name == "pto.talloc_to_aic" || name == "pto.tpush_to_aiv" ||
      name == "pto.tpop_from_aiv" || name == "pto.tfree_from_aiv")
    return InferredSectionKind::Cube;
  if (name.ends_with("_to_aiv") || name.ends_with("_from_aiv"))
    return InferredSectionKind::Vector;
  return std::nullopt;
}

static std::optional<InferredSectionKind>
classifyTileOpByPipe(Operation *op) {
  auto pipeOp = dyn_cast<OpPipeInterface>(op);
  if (!pipeOp)
    return std::nullopt;

  switch (pipeOp.getPipe()) {
  case PIPE::PIPE_M:
  case PIPE::PIPE_MTE1:
    return InferredSectionKind::Cube;
  case PIPE::PIPE_V:
  case PIPE::PIPE_V2:
  case PIPE::PIPE_S:
    return InferredSectionKind::Vector;
  default:
    break;
  }
  return std::nullopt;
}

static std::optional<InferredSectionKind>
classifyTileOpByAddressSpace(Operation *op) {
  SmallVector<AddressSpace, 8> spaces;
  for (Value operand : op->getOperands())
    collectTileAddressSpaces(operand.getType(), spaces);
  for (Value result : op->getResults())
    collectTileAddressSpaces(result.getType(), spaces);

  bool sawVec = false;
  bool sawMat = false;
  bool sawCubeOnly = false;
  for (AddressSpace space : spaces) {
    switch (space) {
    case AddressSpace::LEFT:
    case AddressSpace::RIGHT:
    case AddressSpace::ACC:
    case AddressSpace::BIAS:
    case AddressSpace::SCALING:
      sawCubeOnly = true;
      break;
    case AddressSpace::VEC:
      sawVec = true;
      break;
    case AddressSpace::MAT:
      sawMat = true;
      break;
    default:
      break;
    }
  }

  if (sawCubeOnly)
    return InferredSectionKind::Cube;
  if (sawVec)
    return InferredSectionKind::Vector;
  if (sawMat)
    return classifyTileOpByPipe(op);
  return std::nullopt;
}

static std::optional<InferredSectionKind>
classifyTLoadByDestinationAddressSpace(Operation *op) {
  if (!isa<TLoadOp>(op))
    return std::nullopt;

  auto tload = cast<TLoadOp>(op);
  std::optional<AddressSpace> dstSpace =
      getBufferAddressSpace(tload.getDst().getType());
  if (!dstSpace)
    return std::nullopt;

  switch (*dstSpace) {
  case AddressSpace::VEC:
    return InferredSectionKind::Vector;
  case AddressSpace::MAT:
  case AddressSpace::LEFT:
  case AddressSpace::RIGHT:
  case AddressSpace::ACC:
  case AddressSpace::BIAS:
  case AddressSpace::SCALING:
    return InferredSectionKind::Cube;
  default:
    return std::nullopt;
  }
}

static std::optional<InferredSectionKind>
classifyTStoreBySourceAddressSpace(Operation *op) {
  if (!isa<TStoreOp>(op))
    return std::nullopt;

  auto tstore = cast<TStoreOp>(op);
  std::optional<AddressSpace> srcSpace =
      getBufferAddressSpace(tstore.getSrc().getType());
  if (!srcSpace)
    return std::nullopt;

  switch (*srcSpace) {
  case AddressSpace::VEC:
    return InferredSectionKind::Vector;
  case AddressSpace::MAT:
  case AddressSpace::LEFT:
  case AddressSpace::RIGHT:
  case AddressSpace::ACC:
  case AddressSpace::BIAS:
  case AddressSpace::SCALING:
    return InferredSectionKind::Cube;
  default:
    return std::nullopt;
  }
}

static std::optional<InferredSectionKind> classifyTileOp(Operation *op) {
  if (std::optional<InferredSectionKind> kind = classifyTileOpByName(op))
    return kind;
  if (std::optional<InferredSectionKind> kind = classifyInternalPipeTileOp(op))
    return kind;
  if (std::optional<InferredSectionKind> kind =
          classifyTLoadByDestinationAddressSpace(op))
    return kind;
  if (std::optional<InferredSectionKind> kind =
          classifyTStoreBySourceAddressSpace(op))
    return kind;
  if (std::optional<InferredSectionKind> kind = classifyTileOpByAddressSpace(op))
    return kind;
  return classifyTileOpByPipe(op);
}

struct ModuleKindSummary {
  unsigned vectorCount = 0;
  unsigned cubeCount = 0;
  SmallVector<Operation *, 4> ambiguousOps;
};

enum class FunctionKindCacheState : uint8_t {
  Unknown = 0,
  Vector = 1,
  Cube = 2,
  InProgress = 3,
};

static void inspectModuleKindOperation(Operation *op, ModuleKindSummary &summary) {
  if (!op)
    return;
  if (isExplicitSection(op))
    return;

  if (isPipeLikeOp(op)) {
    if (std::optional<InferredSectionKind> kind = classifyTileOp(op)) {
      if (*kind == InferredSectionKind::Vector)
        ++summary.vectorCount;
      else
        ++summary.cubeCount;
    } else if (isTileLikeOp(op)) {
      summary.ambiguousOps.push_back(op);
    }
  } else if (isRawVPTOVectorLikeOp(op)) {
    ++summary.vectorCount;
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &nested : block.getOperations())
        inspectModuleKindOperation(&nested, summary);
    }
  }
}

static FunctionKindCacheState
encodeFunctionKind(std::optional<InferredSectionKind> kind) {
  if (!kind)
    return FunctionKindCacheState::Unknown;
  return *kind == InferredSectionKind::Vector ? FunctionKindCacheState::Vector
                                              : FunctionKindCacheState::Cube;
}

static std::optional<InferredSectionKind>
decodeFunctionKind(FunctionKindCacheState state) {
  switch (state) {
  case FunctionKindCacheState::Vector:
    return InferredSectionKind::Vector;
  case FunctionKindCacheState::Cube:
    return InferredSectionKind::Cube;
  case FunctionKindCacheState::Unknown:
  case FunctionKindCacheState::InProgress:
    return std::nullopt;
  }
  llvm_unreachable("unexpected function kind cache state");
}

static func::CallOp getTransparentWrapperCall(func::FuncOp funcOp) {
  if (!funcOp || funcOp.isDeclaration() || !funcOp.getBody().hasOneBlock())
    return nullptr;

  Block &entryBlock = funcOp.getBody().front();
  func::CallOp callOp;
  func::ReturnOp returnOp;
  for (Operation &op : entryBlock.getOperations()) {
    if (auto ret = dyn_cast<func::ReturnOp>(op)) {
      returnOp = ret;
      continue;
    }
    if (callOp)
      return nullptr;
    callOp = dyn_cast<func::CallOp>(op);
    if (!callOp)
      return nullptr;
  }

  if (!callOp || !returnOp)
    return nullptr;
  if (returnOp.getNumOperands() != callOp.getNumResults())
    return nullptr;
  for (auto [returned, forwarded] :
       llvm::zip(returnOp.getOperands(), callOp.getResults())) {
    if (returned != forwarded)
      return nullptr;
  }
  return callOp;
}

static std::optional<InferredSectionKind>
inferWholeFunctionKind(func::FuncOp funcOp,
                       llvm::DenseMap<Operation *, FunctionKindCacheState> &cache) {
  if (!funcOp || funcOp.isDeclaration())
    return std::nullopt;

  auto cacheIt = cache.find(funcOp.getOperation());
  if (cacheIt != cache.end()) {
    if (cacheIt->second == FunctionKindCacheState::InProgress)
      return std::nullopt;
    return decodeFunctionKind(cacheIt->second);
  }
  cache[funcOp.getOperation()] = FunctionKindCacheState::InProgress;

  ModuleKindSummary summary;
  inspectModuleKindOperation(funcOp.getOperation(), summary);
  std::optional<InferredSectionKind> inferredKind;
  if (summary.ambiguousOps.empty() && !(summary.vectorCount && summary.cubeCount)) {
    if (summary.vectorCount)
      inferredKind = InferredSectionKind::Vector;
    else if (summary.cubeCount)
      inferredKind = InferredSectionKind::Cube;
  }

  if (!inferredKind) {
    if (func::CallOp callOp = getTransparentWrapperCall(funcOp)) {
      auto callee =
          SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(funcOp, callOp.getCalleeAttr());
      if (callee && callee != funcOp)
        inferredKind = inferWholeFunctionKind(callee, cache);
    }
  }

  cache[funcOp.getOperation()] = encodeFunctionKind(inferredKind);
  return inferredKind;
}

static void assignModuleKernelKind(ModuleOp module, InferredSectionKind kind) {
  FunctionKernelKind kernelKind =
      kind == InferredSectionKind::Vector ? FunctionKernelKind::Vector
                                          : FunctionKernelKind::Cube;
  module->setAttr(FunctionKernelKindAttr::name,
                  FunctionKernelKindAttr::get(module.getContext(), kernelKind));
}

static void assignFunctionKernelKind(func::FuncOp funcOp,
                                     InferredSectionKind kind) {
  if (!funcOp)
    return;

  FunctionKernelKind kernelKind =
      kind == InferredSectionKind::Vector ? FunctionKernelKind::Vector
                                          : FunctionKernelKind::Cube;
  funcOp->setAttr(FunctionKernelKindAttr::name,
                  FunctionKernelKindAttr::get(funcOp.getContext(), kernelKind));
}

static LogicalResult tryAssignWholeModuleKernelKind(ModuleOp module) {
  if (!module || module->hasAttr(FunctionKernelKindAttr::name))
    return success();

  SmallVector<func::FuncOp> defs;
  for (auto funcOp : module.getOps<func::FuncOp>()) {
    if (!funcOp.isDeclaration())
      defs.push_back(funcOp);
  }
  if (defs.empty())
    return success();

  llvm::DenseMap<Operation *, FunctionKindCacheState> cache;
  std::optional<InferredSectionKind> commonKind;
  for (func::FuncOp funcOp : defs) {
    if (hasAnySection(funcOp))
      return success();
    std::optional<InferredSectionKind> funcKind =
        inferWholeFunctionKind(funcOp, cache);
    if (!funcKind)
      return success();
    if (!commonKind) {
      commonKind = funcKind;
      continue;
    }
    if (*commonKind != *funcKind)
      return success();
  }

  if (!commonKind)
    return success();
  assignModuleKernelKind(module, *commonKind);
  return success();
}

static LogicalResult tryAssignWholeFunctionKernelKind(func::FuncOp funcOp) {
  if (!funcOp || funcOp.isDeclaration() || hasAnySection(funcOp) ||
      hasKnownKernelKindContext(funcOp))
    return success();

  llvm::DenseMap<Operation *, FunctionKindCacheState> cache;
  std::optional<InferredSectionKind> kind = inferWholeFunctionKind(funcOp, cache);
  if (!kind)
    return success();

  assignFunctionKernelKind(funcOp, *kind);
  return success();
}

static void inspectSegmentOperation(Operation *op,
                                    UncoveredTopLevelSegment &segment) {
  if (!op)
    return;

  if (isTileLikeOp(op)) {
    segment.containsTileOp = true;
    if (std::optional<InferredSectionKind> kind = classifyTileOp(op)) {
      if (*kind == InferredSectionKind::Vector)
        ++segment.vectorTileOpCount;
      else
        ++segment.cubeTileOpCount;
    } else {
      segment.ambiguousTileOps.push_back(op);
    }
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &nested : block.getOperations()) {
        if (isExplicitSection(&nested)) {
          segment.containsNestedExplicitSection = true;
          continue;
        }
        inspectSegmentOperation(&nested, segment);
      }
    }
  }
}

static std::optional<InferredSectionKind>
inferSegmentKind(const UncoveredTopLevelSegment &segment) {
  if (!segment.ambiguousTileOps.empty())
    return std::nullopt;
  if (segment.vectorTileOpCount && segment.cubeTileOpCount)
    return std::nullopt;
  if (segment.vectorTileOpCount)
    return InferredSectionKind::Vector;
  if (segment.cubeTileOpCount)
    return InferredSectionKind::Cube;
  return std::nullopt;
}

static UncoveredTopLevelSegment summarizeTopLevelOperation(Operation *op) {
  UncoveredTopLevelSegment summary;
  if (!op || isa<func::ReturnOp>(op) || isExplicitSection(op))
    return summary;

  summary.firstOp = op;
  summary.lastOp = op;
  inspectSegmentOperation(op, summary);
  if (summary.containsTileOp)
    summary.firstTileCarrierOp = op;
  return summary;
}

static void collectUncoveredTopLevelSegments(
    func::FuncOp funcOp, SmallVectorImpl<UncoveredTopLevelSegment> &segments) {
  if (!funcOp || funcOp.isDeclaration() || !funcOp.getBody().hasOneBlock())
    return;

  Block &entryBlock = funcOp.getBody().front();
  UncoveredTopLevelSegment current;

  auto flushCurrent = [&]() {
    if (!current.firstOp)
      return;
    segments.push_back(current);
    current = {};
  };

  for (Operation &op : entryBlock.getOperations()) {
    if (isa<func::ReturnOp>(op)) {
      flushCurrent();
      continue;
    }

    if (isExplicitSection(&op)) {
      flushCurrent();
      continue;
    }

    UncoveredTopLevelSegment opSummary = summarizeTopLevelOperation(&op);
    if (!opSummary.containsTileOp) {
      flushCurrent();
      continue;
    }

    if (!current.firstOp) {
      current = std::move(opSummary);
      continue;
    }

    std::optional<InferredSectionKind> currentKind = inferSegmentKind(current);
    std::optional<InferredSectionKind> opKind = inferSegmentKind(opSummary);
    bool mustSplit = current.containsNestedExplicitSection ||
                     opSummary.containsNestedExplicitSection || !currentKind ||
                     !opKind || *currentKind != *opKind;
    if (mustSplit) {
      flushCurrent();
      current = std::move(opSummary);
      continue;
    }

    mergeSegmentSummary(current, opSummary);
  }

  flushCurrent();
}

template <typename SectionOpT>
static void wrapUncoveredTopLevelSegment(func::FuncOp funcOp,
                                         const UncoveredTopLevelSegment &segment) {
  Block &entryBlock = funcOp.getBody().front();
  Operation *firstOp = segment.firstOp;
  Operation *lastOp = segment.lastOp;
  if (!firstOp || !lastOp)
    return;

  OpBuilder builder(firstOp);
  auto sectionOp = builder.create<SectionOpT>(firstOp->getLoc());
  sectionOp.getBody().push_back(new Block());
  Block *sectionBlock = &sectionOp.getBody().front();

  auto firstIt = Block::iterator(firstOp);
  auto afterLastIt = std::next(Block::iterator(lastOp));
  sectionBlock->getOperations().splice(sectionBlock->end(),
                                       entryBlock.getOperations(), firstIt,
                                       afterLastIt);
}

static LogicalResult emitSegmentInferenceError(func::FuncOp funcOp,
                                               const UncoveredTopLevelSegment &segment) {
  InFlightDiagnostic diag =
      funcOp.emitOpError("contains an uncovered top-level TileOp segment whose "
                         "section kind cannot be inferred uniquely");
  if (segment.vectorTileOpCount && segment.cubeTileOpCount) {
    diag << "; saw both vector-like and cube-like TileOps in the same segment";
  } else if (!segment.ambiguousTileOps.empty()) {
    diag << "; ambiguous TileOp(s): ";
    for (size_t i = 0, e = segment.ambiguousTileOps.size(); i < e && i < 3; ++i) {
      if (i)
        diag << ", ";
      diag << '\'' << segment.ambiguousTileOps[i]->getName().getStringRef()
           << '\'';
    }
  }
  return failure();
}

static LogicalResult emitResidualUncoveredTileSegmentError(
    func::FuncOp funcOp, const UncoveredTopLevelSegment &segment) {
  InFlightDiagnostic diag = funcOp.emitOpError(
      "still contains an uncovered top-level TileOp segment after section "
      "normalization");
  if (segment.containsNestedExplicitSection) {
    diag << "; a top-level op mixes nested explicit pto.section.* with sibling "
            "TileOps outside those sections";
  }
  diag << "; first uncovered TileOp segment starts at '"
       << (segment.firstTileCarrierOp ? segment.firstTileCarrierOp
                                      : segment.firstOp)
              ->getName()
              .getStringRef()
       << '\'';
  return failure();
}

static LogicalResult normalizeFunction(func::FuncOp funcOp) {
  if (hasKnownKernelKindContext(funcOp))
    return success();

  SmallVector<UncoveredTopLevelSegment, 4> segments;
  collectUncoveredTopLevelSegments(funcOp, segments);
  for (const UncoveredTopLevelSegment &segment : llvm::reverse(segments)) {
    if (!segment.containsTileOp || segment.containsNestedExplicitSection)
      continue;

    std::optional<InferredSectionKind> kind = inferSegmentKind(segment);
    if (!kind)
      return emitSegmentInferenceError(funcOp, segment);

    switch (*kind) {
    case InferredSectionKind::Cube:
      wrapUncoveredTopLevelSegment<SectionCubeOp>(funcOp, segment);
      break;
    case InferredSectionKind::Vector:
      wrapUncoveredTopLevelSegment<SectionVectorOp>(funcOp, segment);
      break;
    }
  }
  return success();
}

static LogicalResult verifyFunctionHasNoResidualUncoveredTileSegments(
    func::FuncOp funcOp) {
  if (hasKnownKernelKindContext(funcOp))
    return success();

  SmallVector<UncoveredTopLevelSegment, 4> segments;
  collectUncoveredTopLevelSegments(funcOp, segments);
  for (const UncoveredTopLevelSegment &segment : segments) {
    if (!segment.containsTileOp)
      continue;
    return emitResidualUncoveredTileSegmentError(funcOp, segment);
  }
  return success();
}

static LogicalResult scanModuleForUncoveredTileSegments(ModuleOp module) {
  LogicalResult status = success();
  module.walk([&](ModuleOp nestedModule) {
    if (failed(status))
      return WalkResult::interrupt();
    status = tryAssignWholeModuleKernelKind(nestedModule);
    return failed(status) ? WalkResult::interrupt() : WalkResult::advance();
  });
  if (failed(status))
    return status;

  module.walk([&](func::FuncOp funcOp) {
    if (failed(status))
      return WalkResult::interrupt();
    status = tryAssignWholeFunctionKernelKind(funcOp);
    if (failed(status))
      return WalkResult::interrupt();
    status = normalizeFunction(funcOp);
    if (succeeded(status))
      status = verifyFunctionHasNoResidualUncoveredTileSegments(funcOp);
    return failed(status) ? WalkResult::interrupt() : WalkResult::advance();
  });
  return status;
}

struct PTONormalizeUncoveredTileSectionsPass
    : public mlir::pto::impl::PTONormalizeUncoveredTileSectionsBase<
          PTONormalizeUncoveredTileSectionsPass> {
  void runOnOperation() override {
    if (failed(scanModuleForUncoveredTileSegments(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTONormalizeUncoveredTileSectionsPass() {
  return std::make_unique<PTONormalizeUncoveredTileSectionsPass>();
}
