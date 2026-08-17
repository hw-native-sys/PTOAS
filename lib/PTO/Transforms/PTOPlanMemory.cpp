// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PlanMemory.cpp ----Plan Buffer Memory Address ----------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "PTOPlanMemory.h"

#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#define DEBUG_TYPE "pto-plan-memory"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X)

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PLANMEMORY
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace pto;

namespace {

constexpr int64_t kBitsPerByte = 8;
constexpr unsigned kI32BitWidth = 32;
constexpr unsigned kMemoryEffectReserveSize = 8;
constexpr int kSingleBufferCount = 1;
constexpr int kDoubleBufferCount = 2;
constexpr int64_t kA5VecLocalMemBits = 2031616;
constexpr int64_t kA3VecLocalMemBits = 1572864;
constexpr int64_t kMatLocalMemBits = 4194304;
constexpr int64_t kLocalMemAlignmentBytes = 256;

struct LocalMemSpec {
  int64_t capacityBits = 0;
  int64_t alignBytes = 1;
};

static std::optional<int64_t> getTileBufferFootprintBytes(TileBufType type) {
  ArrayRef<int64_t> shape = type.getShape();
  unsigned elemBytes = getPTOStorageElemByteSize(type.getElementType());
  if (elemBytes == 0) {
    return std::nullopt;
  }

  if (type.getCompactModeI32() !=
      static_cast<int32_t>(pto::CompactMode::RowPlusOne)) {
    std::optional<int64_t> totalStaticSize = getStaticTotalSize(shape);
    if (!totalStaticSize.has_value()) {
      return std::nullopt;
    }
    return totalStaticSize.value() * static_cast<int64_t>(elemBytes);
  }

  if (shape.size() != mlir::pto::kValue2 || llvm::is_contained(shape, ShapedType::kDynamic)) {
    return std::nullopt;
  }

  bool rowMajor =
      type.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
  int64_t major = rowMajor ? shape[0] : shape[1];
  int64_t minor = rowMajor ? shape[1] : shape[0];
  if (major == 0 || minor == 0) {
    return 0;
  }
  return ((major - 1) * (minor + 1) + minor) *
         static_cast<int64_t>(elemBytes);
}

static int64_t ceilDivBitsToBytes(int64_t bits) {
  return (bits + kBitsPerByte - 1) / kBitsPerByte;
}

static int64_t alignUpBytes(int64_t value, int64_t align) {
  int64_t safeAlign = std::max<int64_t>(align, 1);
  if (safeAlign == 1) {
    return value;
  }
  int64_t rem = value % safeAlign;
  if (rem == 0) {
    return value;
  }
  return value + (safeAlign - rem);
}

static size_t plannerAlignBitsFromBytes(size_t alignBytes) {
  return std::max<size_t>(alignBytes, 1) * kBitsPerByte;
}

static LocalMemSpec getLocalMemSpec(Operation *op, AddressSpace as) {
  switch (as) {
  case AddressSpace::VEC:
    return isTargetArchA5(op)
               ? LocalMemSpec{kA5VecLocalMemBits, kLocalMemAlignmentBytes}
               : LocalMemSpec{kA3VecLocalMemBits, kLocalMemAlignmentBytes};
  case AddressSpace::MAT:
    return LocalMemSpec{kMatLocalMemBits, kLocalMemAlignmentBytes};
  default:
    return LocalMemSpec{};
  }
}

static bool isNameIn(StringRef name, ArrayRef<StringRef> names) {
  return llvm::is_contained(names, name);
}

static bool isIgnoredA5TmpOperandUse(OpOperand &use) {
  Operation *owner = use.getOwner();
  unsigned operandNo = use.getOperandNumber();
  StringRef name = owner->getName().getStringRef();

  if (auto dpsOp = dyn_cast<pto::PTO_DpsInitOpInterface>(owner)) {
    if (llvm::is_contained(dpsOp.getDpsInits(), use.get())) {
      return false;
    }
  } else if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(owner)) {
    if (llvm::is_contained(dpsOp.getDpsInits(), use.get())) {
      return false;
    }
  }

  if (isNameIn(name, {"pto.trowargmax", "pto.trowargmin", "pto.trowmax",
                      "pto.trowmin", "pto.trowsum", "pto.trowprod"})) {
    return operandNo == 1;
  }

  if (name == "pto.txors") {
    return operandNo == mlir::pto::kValue2;
  }

  if (isNameIn(name, {"pto.tprelu", "pto.txor", "pto.tsels",
                      "pto.trowexpand", "pto.tcolexpand",
                      "pto.trowexpandadd", "pto.trowexpanddiv",
                      "pto.trowexpandexpdif", "pto.trowexpandmax",
                      "pto.trowexpandmin", "pto.trowexpandmul",
                      "pto.trowexpandsub", "pto.tcolexpandadd",
                      "pto.tcolexpanddiv", "pto.tcolexpandexpdif",
                      "pto.tcolexpandmax", "pto.tcolexpandmin",
                      "pto.tcolexpandmul", "pto.tcolexpandsub"})) {
    return operandNo == mlir::pto::kValue2;
  }

  if (name == "pto.tsel") {
    return operandNo == mlir::pto::kValue3;
  }

  return false;
}

static bool isA5IgnoredTmpAlloc(pto::AllocTileOp allocTile) {
  if (getTargetArch(allocTile.getOperation()) != PTOArch::A5) {
    return false;
  }

  Value value = allocTile.getResult();
  if (value.use_empty()) {
    return false;
  }

  for (OpOperand &use : value.getUses()) {
    if (!isIgnoredA5TmpOperandUse(use)) {
      return false;
    }
  }
  return true;
}

static void collectStableValueOrder(Region &region,
                                    AsmState &asmState,
                                    DenseMap<Value, std::string> &stableValueKeys,
                                    SmallVectorImpl<Value> &seenValues) {
  auto recordValue = [&asmState, &seenValues, &stableValueKeys](Value value) {
    if (stableValueKeys.find(value) != stableValueKeys.end()) {
      return;
    }
    std::string key;
    llvm::raw_string_ostream os(key);
    value.printAsOperand(os, asmState);
    stableValueKeys[value] = os.str();
    seenValues.push_back(value);
  };

  for (Block &block : region) {
    for (BlockArgument blockArg : block.getArguments()) {
      recordValue(blockArg);
    }
    for (Operation &op : block) {
      for (Value result : op.getResults()) {
        recordValue(result);
      }
      for (Region &nestedRegion : op.getRegions()) {
        collectStableValueOrder(nestedRegion, asmState, stableValueKeys,
                                seenValues);
      }
    }
  }
}

static StableValueOrderMap buildStableValueOrder(func::FuncOp func) {
  DenseMap<Value, std::string> stableValueKeys;
  SmallVector<Value> seenValues;
  AsmState asmState(func);
  collectStableValueOrder(func.getBody(), asmState, stableValueKeys, seenValues);

  llvm::sort(seenValues, [&](Value lhs, Value rhs) {
    const std::string &lhsKey = stableValueKeys.find(lhs)->second;
    const std::string &rhsKey = stableValueKeys.find(rhs)->second;
    if (lhsKey != rhsKey) {
      return lhsKey < rhsKey;
    }
    return isLessValue(lhs, rhs);
  });

  StableValueOrderMap stableValueOrder;
  for (auto [index, value] : llvm::enumerate(seenValues)) {
    stableValueOrder[value] = index;
  }
  return stableValueOrder;
}

static uint32_t lookupStableValueOrder(
    Value value, const StableValueOrderMap &stableValueOrder) {
  auto it = stableValueOrder.find(value);
  if (it != stableValueOrder.end()) {
    return it->second;
  }
  return std::numeric_limits<uint32_t>::max();
}

static void sortValuesByStableOrder(
    SmallVectorImpl<Value> &values,
    const StableValueOrderMap &stableValueOrder) {
  llvm::sort(values, [&](Value lhs, Value rhs) {
    uint32_t lhsOrder = lookupStableValueOrder(lhs, stableValueOrder);
    uint32_t rhsOrder = lookupStableValueOrder(rhs, stableValueOrder);
    if (lhsOrder != rhsOrder) {
      return lhsOrder < rhsOrder;
    }
    return isLessValue(lhs, rhs);
  });
}

static SmallVector<Value> getScratchBuffersFromEffects(Operation *op,
                                                       ValueRange dpsInits,
                                                       const StableValueOrderMap &stableValueOrder) {
  SmallVector<Value> scratchBuffers;
  auto memEffect = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffect) {
    return scratchBuffers;
  }

  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>,
              kMemoryEffectReserveSize>
      effects;
  memEffect.getEffects(effects);
  for (const auto &effect : effects) {
    if (!isa<MemoryEffects::Write>(effect.getEffect())) {
      continue;
    }
    Value value = effect.getValue();
    if (!value) {
      continue;
    }
    if (!llvm::is_contained(op->getOperands(), value)) {
      continue;
    }
    if (llvm::is_contained(dpsInits, value)) {
      continue;
    }
    if (!llvm::is_contained(scratchBuffers, value)) {
      scratchBuffers.push_back(value);
    }
  }
  sortValuesByStableOrder(scratchBuffers, stableValueOrder);
  return scratchBuffers;
}

static SmallVector<Value>
getMemoryEffectBufferOperands(Operation *op,
                              const StableValueOrderMap &stableValueOrder) {
  SmallVector<Value> buffers;
  auto memEffect = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffect) {
    return buffers;
  }

  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>,
              kMemoryEffectReserveSize>
      effects;
  memEffect.getEffects(effects);
  for (const auto &effect : effects) {
    if (!isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect())) {
      continue;
    }
    Value value = effect.getValue();
    if (!value || !GetBufferSpaceAttr(value)) {
      continue;
    }
    if (!llvm::is_contained(buffers, value)) {
      buffers.push_back(value);
    }
  }
  sortValuesByStableOrder(buffers, stableValueOrder);
  return buffers;
}

static SmallVector<ValuePair>
getScratchConflictPairsFromEffects(Operation *op, ValueRange dpsInits,
                                   const StableValueOrderMap &stableValueOrder) {
  SmallVector<ValuePair> conflictPairs;
  SmallVector<Value> scratchBuffers =
      getScratchBuffersFromEffects(op, dpsInits, stableValueOrder);
  for (Value scratch : scratchBuffers) {
    for (Value dst : dpsInits) {
      if (!scratch || !dst || scratch == dst) {
        continue;
      }
      conflictPairs.emplace_back(scratch, dst);
    }
  }
  return conflictPairs;
}

enum class ReserveBufferMode {
  None,
  Auto,
  Manual,
};

struct ReserveBufferPlan {
  ReserveBufferMode mode = ReserveBufferMode::None;
  ReserveBufferOp reserveOp;
  AddressSpace addressSpace = AddressSpace::Zero;
  int64_t sizeBytes = 0;
  int64_t capacityBytes = 0;
  int64_t alignBytes = 1;
};

using ReserveBufferPlans = SmallVector<ReserveBufferPlan>;

static LogicalResult analyzeReserveBufferPlans(func::FuncOp funcOp,
                                               ReserveBufferPlans &plans) {
  SmallVector<ReserveBufferOp> reserveOps;
  funcOp.walk(
      [&](ReserveBufferOp reserveOp) { reserveOps.push_back(reserveOp); });

  if (reserveOps.empty()) {
    return success();
  }

  for (ReserveBufferOp reserveOp : reserveOps) {
    AddressSpace as = reserveOp.getLocation().getAddressSpace();
    auto spec = getLocalMemSpec(reserveOp.getOperation(), as);
    if (spec.capacityBits <= 0 || spec.alignBytes <= 0) {
      return reserveOp.emitOpError("unsupported reserve_buffer location");
    }

    int64_t capacityBytes = spec.capacityBits / kBitsPerByte;
    int64_t sizeBytes = reserveOp.getSize();
    bool autoAlloc = reserveOp.getAutoAlloc();

    ReserveBufferPlan &plan = plans.emplace_back();
    plan.mode = autoAlloc ? ReserveBufferMode::Auto : ReserveBufferMode::Manual;
    plan.reserveOp = reserveOp;
    plan.addressSpace = as;
    plan.sizeBytes = sizeBytes;
    plan.capacityBytes = capacityBytes;
    plan.alignBytes = spec.alignBytes;

    // Auto mode only declares that one contiguous region must be reserved.
    // The concrete base is filled later from a hole in the target local space.
    if (autoAlloc) {
      if (reserveOp.getBaseAttr()) {
        return reserveOp.emitOpError(
            "expects 'base' to be absent when 'auto' is true");
      }
      continue;
    }

    // In manual mode, reserve_buffer.base is already fixed by the frontend or
    // an earlier stage. Only basic validation is needed here.
    auto baseAttr = reserveOp.getBaseAttr();
    if (!baseAttr) {
      return reserveOp.emitOpError("expects 'base' when 'auto' is false");
    }

    int64_t baseBytes = baseAttr.getInt();
    if (baseBytes % spec.alignBytes != 0) {
      return reserveOp.emitOpError(
          "expects 'base' to satisfy the address-space alignment");
    }
    if (baseBytes + sizeBytes > capacityBytes) {
      return reserveOp.emitOpError("exceeds available local memory capacity");
    }
  }

  return success();
}

struct OccupiedByteRange {
  int64_t begin = 0;
  int64_t end = 0;
};

static LogicalResult assignAutoReserveBufferBases(
    ReserveBufferPlans &plans,
    const std::map<Value, BufferInfo, ValueComparator> &bufferInfos,
    const DenseMap<Value, SmallVector<uint64_t>> &buffer2Offsets) {
  std::map<AddressSpace, SmallVector<OccupiedByteRange>> occupiedByAddressSpace;
  for (const auto &it : bufferInfos) {
    Value buffer = it.first;
    const BufferInfo &bufferInfo = it.second;

    auto offsetsIt = buffer2Offsets.find(buffer);
    if (offsetsIt == buffer2Offsets.end()) {
      continue;
    }

    // Reserve-buffer allocation intentionally happens after normal MemPlan.
    // Reconstruct the already occupied byte ranges from the planned local
    // buffers, then place reserve_buffer into the first aligned hole.
    int64_t occupiedSizeBytes = ceilDivBitsToBytes(bufferInfo.constBits);
    if (bufferInfo.operation) {
      auto spec = getLocalMemSpec(bufferInfo.operation, bufferInfo.bufferScope);
      occupiedSizeBytes =
          alignUpBytes(occupiedSizeBytes, std::max<int64_t>(spec.alignBytes, 1));
    }
    for (uint64_t offsetBytes : offsetsIt->second) {
      occupiedByAddressSpace[bufferInfo.bufferScope].push_back(
          OccupiedByteRange{static_cast<int64_t>(offsetBytes),
                            static_cast<int64_t>(offsetBytes) + occupiedSizeBytes});
    }
  }

  auto normalizeRanges = [](SmallVector<OccupiedByteRange> &ranges) {
    llvm::sort(ranges,
               [](const OccupiedByteRange &lhs, const OccupiedByteRange &rhs) {
                 return lhs.begin < rhs.begin;
               });

    SmallVector<OccupiedByteRange> merged;
    for (const OccupiedByteRange &range : ranges) {
      if (merged.empty() || range.begin > merged.back().end) {
        merged.push_back(range);
        continue;
      }
      merged.back().end = std::max(merged.back().end, range.end);
    }
    ranges.swap(merged);
  };

  for (auto &it : occupiedByAddressSpace) {
    normalizeRanges(it.second);
  }

  for (ReserveBufferPlan &plan : plans) {
    if (plan.mode != ReserveBufferMode::Auto || !plan.reserveOp) {
      continue;
    }

    SmallVector<OccupiedByteRange> &occupied =
        occupiedByAddressSpace[plan.addressSpace];

    // First-fit search: try address 0 first, then keep moving the candidate to
    // the end of the current occupied interval until a large-enough aligned
    // hole is found.
    int64_t candidateBase = 0;
    for (const OccupiedByteRange &range : occupied) {
      candidateBase = alignUpBytes(candidateBase, plan.alignBytes);
      if (candidateBase + plan.sizeBytes <= range.begin) {
        break;
      }
      candidateBase = std::max(candidateBase, range.end);
    }
    candidateBase = alignUpBytes(candidateBase, plan.alignBytes);
    if (candidateBase + plan.sizeBytes > plan.capacityBytes) {
      return plan.reserveOp.emitOpError(
          "failed to allocate local memory hole for reserve_buffer");
    }

    plan.reserveOp->setAttr(
        "base",
        IntegerAttr::get(
            IntegerType::get(plan.reserveOp.getContext(), kI32BitWidth),
            candidateBase));
    occupied.push_back(
        OccupiedByteRange{candidateBase, candidateBase + plan.sizeBytes});
    normalizeRanges(occupied);
  }
  return success();
}

} // namespace

void MemLivenessAnalysis::build() {
  Region &funcRegion = func_.getBody();
  stableValueOrder = buildStableValueOrder(func_);
  Liveness live(func_);
  // Recursively obtaining IR information.
  RecursionIR(&funcRegion, live);
  // the lifetime of the buffer.
  GenerateBufferLife();
}

bool MemLivenessAnalysis::isLocalMemPlan() const {
  return planMode == MemPlanMode::LOCAL_MEM_PLAN;
}

bool MemLivenessAnalysis::isGlobalWorkSpaceMemPlan() const {
  return planMode == MemPlanMode::GLOBAL_WORKSPACE_PLAN;
}

void MemLivenessAnalysis::RecursionIR(Region *region, Liveness live) {
  auto result = region->walk<WalkOrder::PreOrder>([&](Operation *op) {
    // recursive control flow
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      RecursiveIfOp(ifOp, live);
      return WalkResult::skip();
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      RecursiveForOp(forOp, live);
      return WalkResult::skip();
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      RecursiveWhileOp(whileOp, live);
      return WalkResult::skip();
    } else if (auto fusionRegion = dyn_cast<pto::FusionRegionOp>(op)) {
      RecursiveFusionRegionOp(fusionRegion, live);
      return WalkResult::skip();
    } else if (auto section = dyn_cast<pto::SectionCubeOp>(op)) {
      // Sections are transparent containers. Analyze tileop-local cube work
      // in place so it contributes to liveness.
      UpdateLinearOperation(section.getOperation());
      RecursionIR(&section.getBody(), live);
      auto sectionEnd = UpdateLinearOperation(section.getOperation());
      OpKillHandle(sectionEnd, live, section->getBlock());
      return WalkResult::skip();
    } else if (auto section = dyn_cast<pto::SectionVectorOp>(op)) {
      // Keep vector sections transparent for the same reason as cube sections.
      UpdateLinearOperation(section.getOperation());
      RecursionIR(&section.getBody(), live);
      auto sectionEnd = UpdateLinearOperation(section.getOperation());
      OpKillHandle(sectionEnd, live, section->getBlock());
      return WalkResult::skip();
    }

    // process operation
    auto curOpInfo = UpdateLinearOperation(op);
    auto mayAliasOp = getOperationAliasInfo(op);
    if (mayAliasOp.has_value()) {
      auto aliasPair = mayAliasOp.value();
      UpdateBufferAlias(aliasPair.first, aliasPair.second);
    } else if (isa<pto::DeclareTileOp>(op)) {
      // Runtime-bound tile handles do not allocate static local storage.
      return WalkResult::advance();
    } else if (isLocalMemPlan() && dyn_cast<pto::AllocTileOp>(op)) {
      auto allocTileOp = cast<pto::AllocTileOp>(op);
      if (allocTileOp.getAddr()) {
        return WalkResult::advance();
      }
      if (isA5IgnoredTmpAlloc(allocTileOp)) {
        return WalkResult::advance();
      }
      auto memorySpaceAttr = GetBufferSpaceAttr(allocTileOp.getResult());
      if (!isLocalBuffer(memorySpaceAttr)) {
        allocTileOp.emitError("Alloc tile buffer not at local space");
        return WalkResult::interrupt();
      }
      if (auto attr = allocTileOp->getAttrOfType<IntegerAttr>(
              mlir::pto::kPtoMultiBufferAttrName)) {
        uint64_t n = attr.getValue().getZExtValue();
        if (n < mlir::pto::kPtoMultiBufferMinNum ||
            n > mlir::pto::kPtoMultiBufferMaxNum) {
          allocTileOp.emitError()
              << "pto.multi_buffer must be in ["
              << mlir::pto::kPtoMultiBufferMinNum << ", "
              << mlir::pto::kPtoMultiBufferMaxNum << "] (got " << n << ")";
          return WalkResult::interrupt();
        }
        buffer2MultiNum[allocTileOp.getResult()] = static_cast<uint32_t>(n);
      }
      UpdateOpBufferInfo(op, op->getResults());
      return WalkResult::advance();
    } else if (isLocalMemPlan() && dyn_cast<pto::AllocMultiTileOp>(op)) {
      auto allocMultiOp = cast<pto::AllocMultiTileOp>(op);
      if (allocMultiOp.getAddr()) {
        return WalkResult::advance();
      }
      auto memorySpaceAttr = GetBufferSpaceAttr(allocMultiOp.getResult());
      if (!isLocalBuffer(memorySpaceAttr)) {
        allocMultiOp.emitError("Alloc multi tile buffer not at local space");
        return WalkResult::interrupt();
      }
      uint32_t count = allocMultiOp.getResult().getType().getCount();
      buffer2MultiNum[allocMultiOp.getResult()] = count;
      UpdateOpBufferInfo(op, op->getResults());
      return WalkResult::advance();
    } else if (auto tprintOp = dyn_cast<pto::TPrintOp>(op)) {
      // TPrintOp only reads from buffer, similar to LoadOp
      UpdateOpGenInfo(curOpInfo, llvm::to_vector(op->getOperands()));
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto tgetvalOp = dyn_cast<pto::TGetValOp>(op)) {
      (void)tgetvalOp;
      UpdateOpGenInfo(curOpInfo, llvm::to_vector(op->getOperands()));
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto setValidShapeOp = dyn_cast<pto::SetValidShapeOp>(op)) {
      (void)setValidShapeOp;
      // Metadata-only update on an existing tile handle. Keep the source buffer
      // alive through this operation, but do not model it as producing a new
      // alias/result buffer.
      UpdateOpGenInfo(curOpInfo, ValueRange{op->getOperand(0)});
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto getValidShapeOp = dyn_cast<pto::GetValidShapeOp>(op)) {
      (void)getValidShapeOp;
      // Metadata-only read from an existing tile handle. It touches the source
      // buffer for liveness, but the scalar row/col results are not buffers.
      UpdateOpGenInfo(curOpInfo, ValueRange{op->getOperand(0)});
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto ptoDpsOp = dyn_cast<pto::PTO_DpsInitOpInterface>(op)) {
      // PTO ops with destination (tile_buf, partition_view, etc.).
      SmallVector<Value> genBuffers = llvm::to_vector(op->getOperands());
      auto scratchBuffers = getScratchBuffersFromEffects(
          op, ptoDpsOp.getDpsInits(), stableValueOrder);
      genBuffers.append(scratchBuffers.begin(), scratchBuffers.end());
      UpdateOpGenInfo(curOpInfo, genBuffers);
      for (const auto &conflictPair :
           getScratchConflictPairsFromEffects(op, ptoDpsOp.getDpsInits(),
                                              stableValueOrder)) {
        RecordSemanticConflict(conflictPair.first, conflictPair.second);
      }
      for (const auto &[lhs, rhs] : getSemanticNoAliasPairs(op))
        RecordSemanticConflict(lhs, rhs);
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto dstStyleOp = dyn_cast<DestinationStyleOpInterface>(op)) {
      // Preserve generic destination-style aliasing for non-tile tensors.
      UpdateInitAndResAlias(dstStyleOp);
      UpdateOpGenInfo(curOpInfo, llvm::to_vector(dstStyleOp.getDpsInits()));
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
      UpdateBufferAlias(selectOp.getResult(), selectOp.getTrueValue(), true);
      UpdateBufferAlias(selectOp.getResult(), selectOp.getFalseValue(), true);
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto tileBufAddrOp = dyn_cast<pto::TileBufAddrOp>(op)) {
      UpdateOpGenInfo(curOpInfo, ValueRange{tileBufAddrOp.getSrc()});
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
      UpdateOpGenInfo(curOpInfo, llvm::to_vector(callOp->getOperands()));
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto simtLaunch = dyn_cast<pto::SimtLaunchOp>(op)) {
      // A launched SIMT helper reads its explicit argument list. Treat it as
      // a call for liveness so tile-derived local memrefs remain live through
      // the launch.
      UpdateOpGenInfo(curOpInfo, llvm::to_vector(simtLaunch->getOperands()));
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (isa<pto::TAllocOp, pto::TPushOp, pto::TFreeOp,
                   pto::SetQuantScalarOp, pto::SetQuantVectorOp,
                   pto::InitializeL2LPipeOp, pto::InitializeL2G2LPipeOp,
                   pto::BuildAsyncSessionOp,
                   pto::TPutAsyncOp, pto::TGetAsyncOp, pto::TPutOp,
                   pto::TGetOp, pto::TNotifyOp, pto::TWaitOp, pto::TTestOp,
                   pto::SyncAllOp,
                   pto::TBroadcastOp, pto::CommTGatherOp,
                   pto::CommTScatterOp, pto::TReduceOp>(op)) {
      UpdateOpGenInfo(curOpInfo, llvm::to_vector(op->getOperands()));
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto gpuLaunchOp = dyn_cast<gpu::LaunchFuncOp>(op)) {
      UpdateOpGenInfo(curOpInfo, llvm::to_vector(gpuLaunchOp->getOperands()));
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (isa<MemoryEffectOpInterface>(op)) {
      // Vector/Cube micro-ops such as vlds/vsts are not destination-style
      // ops, but their explicit memory effects identify the local buffers
      // that must remain live through the instruction.
      SmallVector<Value> buffers =
          getMemoryEffectBufferOperands(op, stableValueOrder);
      UpdateOpGenInfo(curOpInfo, buffers);
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (failed(CheckIfUnknownOpTouchBuffer(op))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result == WalkResult::interrupt()) {
    llvm_unreachable("PlanMemory Traverse IR Failed! ");
  }
}

void MemLivenessAnalysis::UpdateInitAndResAlias(
    DestinationStyleOpInterface dstStyleOp) {
  auto results = dstStyleOp->getResults();
  if (results.empty()) {
    return;
  }
  for (auto [res, init] :
       llvm::zip(dstStyleOp->getResults(), dstStyleOp.getDpsInits())) {
    auto iter = buffer2AliasVec.find(init);
    if (iter == buffer2AliasVec.end()) {
      continue;
    }
    auto tensorType = dyn_cast_or_null<TensorType>(res.getType());
    if (tensorType) {
      UpdateBufferAlias(res, init);
    }
  }
}

OpInfo *MemLivenessAnalysis::UpdateLinearOperation(Operation *op) {
  auto opInfo = std::make_unique<OpInfo>(op, seqIndex++);
  auto curOpInfo = opInfo.get();
  linearOperation.push_back(std::move(opInfo));
  return curOpInfo;
}

void MemLivenessAnalysis::UpdateForOpBufferAlias(scf::ForOp forOp) {
  if (forOp.getResults().empty()) {
    return;
  }
  if (!forOp.getRegionIterArgs().empty()) {
    if (forOp.getYieldedValues().size() != forOp.getRegionIterArgs().size() ||
        forOp.getInitArgs().size() != forOp.getRegionIterArgs().size()) {
      llvm::report_fatal_error("scf.for alias sizes are inconsistent");
    }
    for (auto [i, arg] : llvm::enumerate(forOp.getRegionIterArgs())) {
      // yielded values alias region iter args.
      UpdateBufferAlias(forOp.getYieldedValues()[i], arg);
    }
  }
  if (forOp->getResults().size() != forOp.getYieldedValues().size()) {
    llvm::report_fatal_error("scf.for result/yield sizes are inconsistent");
  }
  for (auto [i, arg] : llvm::enumerate(forOp.getYieldedValues())) {
    // forOp result values alias region iter yielded values.
    UpdateBufferAlias(forOp->getResult(i), arg);
  }
}

void MemLivenessAnalysis::UpdateWhileOpBufferAlias(scf::WhileOp whileOp) {
  if (whileOp.getResults().size() != whileOp.getYieldedValues().size())
    llvm::report_fatal_error("scf.while result/yield sizes are inconsistent");

  // The before region receives the initial values and the after region
  // receives the values forwarded by scf.condition.  The after-region yield
  // is the back-edge which becomes the next before-region argument and the
  // scf.while result on exit.
  for (auto [i, init] : llvm::enumerate(whileOp.getInits())) {
    if (i < whileOp.getBeforeArguments().size())
      UpdateBufferAlias(whileOp.getBeforeArguments()[i], init);
  }

  auto conditionArgs = whileOp.getConditionOp().getArgs();
  for (auto [i, arg] : llvm::enumerate(whileOp.getAfterArguments())) {
    if (i < conditionArgs.size())
      UpdateBufferAlias(arg, conditionArgs[i]);
  }

  for (auto [i, yielded] : llvm::enumerate(whileOp.getYieldedValues())) {
    if (i < whileOp.getBeforeArguments().size())
      UpdateBufferAlias(whileOp.getBeforeArguments()[i], yielded);
    if (i < whileOp.getConditionOp().getArgs().size())
      UpdateBufferAlias(whileOp.getResult(i),
                        whileOp.getConditionOp().getArgs()[i]);
    if (i < whileOp.getResults().size())
      UpdateBufferAlias(whileOp.getResult(i), yielded);
  }
}

void MemLivenessAnalysis::RecursiveForOp(scf::ForOp forOp, Liveness live) {
  // Model loop-carried tile handles as aliases of the yielded roots.
  auto forBeginSeq = UpdateLinearOperation(forOp.getOperation());
  UpdateOpGenInfo(forBeginSeq, GetLiveBuffersInLoop(forOp, live));
  UpdateForOpInitArgsAlias(forOp);
  RecursionIR(&forOp.getRegion(), live);
  UpdateForOpBufferAlias(forOp);
  auto forEndSeq = UpdateLinearOperation(forOp.getOperation());
  OpKillHandle(forEndSeq, live, forOp->getBlock());
}

void MemLivenessAnalysis::RecursiveWhileOp(scf::WhileOp whileOp,
                                           Liveness live) {
  auto whileBeginSeq = UpdateLinearOperation(whileOp.getOperation());
  UpdateOpGenInfo(whileBeginSeq, GetLiveBuffersInLoop(whileOp.getOperation(), live));

  UpdateWhileOpBufferAlias(whileOp);
  RecursionIR(&whileOp.getBefore(), live);
  RecursionIR(&whileOp.getAfter(), live);
  UpdateWhileOpBufferAlias(whileOp);

  auto whileEndSeq = UpdateLinearOperation(whileOp.getOperation());
  OpKillHandle(whileEndSeq, live, whileOp->getBlock());
}

void MemLivenessAnalysis::UpdateForOpInitArgsAlias(scf::ForOp forOp) {
  if (forOp.getInitArgs().empty()) {
    return;
  }
  if (forOp.getInitArgs().size() != forOp.getRegionIterArgs().size()) {
    llvm::report_fatal_error("scf.for init/iter-arg sizes are inconsistent");
  }
  for (auto [i, arg] : llvm::enumerate(forOp.getInitArgs())) {
    // init args alias region iter args.
    UpdateBufferAlias(forOp.getRegionIterArgs()[i], arg);
  }
}

void MemLivenessAnalysis::UpdateIfOpBufferAlias(scf::IfOp ifOp,
                                                scf::YieldOp yieldOp) {
  if (ifOp.getResults().empty()) {
    return;
  }
  if (ifOp->getResults().size() != yieldOp->getOperands().size()) {
    llvm::report_fatal_error("scf.if result/yield sizes are inconsistent");
  }
  for (auto [i, arg] : llvm::enumerate(yieldOp->getOperands())) {
    // Multiple buffers involved, requiring one-to-one correspondence.
    UpdateBufferAlias(ifOp->getResult(i), arg);
  }
}

void MemLivenessAnalysis::RecursiveIfOp(scf::IfOp ifOp, Liveness live) {
  // Join the tile roots yielded by each branch into the if result.
  UpdateLinearOperation(ifOp.getOperation());
  RecursionIR(&ifOp.getThenRegion(), live);
  auto curIfElse = UpdateLinearOperation(ifOp.getOperation());
  UpdateIfOpBufferAlias(ifOp, ifOp.thenYield());

  auto curIfEnd = curIfElse;
  if (ifOp.elseBlock()) {
    RecursionIR(&ifOp.getElseRegion(), live);
    curIfEnd = UpdateLinearOperation(ifOp.getOperation());
    UpdateIfOpBufferAlias(ifOp, ifOp.elseYield());
  }
  OpKillHandle(curIfEnd, live, ifOp->getBlock());
}

void MemLivenessAnalysis::UpdateFusionRegionBufferAlias(
    pto::FusionRegionOp fusionRegion, pto::YieldOp yieldOp) {
  if (fusionRegion.getResults().empty()) {
    return;
  }
  if (fusionRegion->getResults().size() != yieldOp->getOperands().size()) {
    llvm::report_fatal_error(
        "pto.fusion_region result/yield sizes are inconsistent");
  }
  for (auto [i, yielded] : llvm::enumerate(yieldOp->getOperands())) {
    UpdateBufferAlias(fusionRegion->getResult(i), yielded);
  }
}

void MemLivenessAnalysis::RecursiveFusionRegionOp(pto::FusionRegionOp fusionRegion,
                                                  Liveness live) {
  UpdateLinearOperation(fusionRegion.getOperation());
  RecursionIR(&fusionRegion.getBody(), live);

  auto yieldOp =
      dyn_cast<pto::YieldOp>(fusionRegion.getBody().front().getTerminator());
  if (!yieldOp) {
    llvm::report_fatal_error("pto.fusion_region must terminate with pto.yield");
  }
  UpdateFusionRegionBufferAlias(fusionRegion, yieldOp);

  auto regionEnd = UpdateLinearOperation(fusionRegion.getOperation());
  OpKillHandle(regionEnd, live, fusionRegion->getBlock());
}

SmallVector<Value> MemLivenessAnalysis::GetLiveBuffersInLoop(Operation *loopOp,
                                                             Liveness live) {
  SmallVector<Value> allocBeforeLoopBuffers;
  const auto *liveBlockInfo = live.getLiveness(loopOp->getBlock());
  auto currentLiveValues =
      liveBlockInfo->currentlyLiveValues(loopOp);
  if (currentLiveValues.empty()) {
    return allocBeforeLoopBuffers;
  }
  // The gen buffer of the same operation must ensure the order of priority.
  SetVector<Value> currentLiveValuesOrder;
  for (auto buffer : currentLiveValues) {
    currentLiveValuesOrder.insert(buffer);
  }
  for (const Value &operand : currentLiveValuesOrder) {
    auto aliasBuffers = GetAliasBuffers(operand);
    aliasBuffers.insert(operand);
    for (auto Buffer : aliasBuffers) {
      auto iter = buffer2status.find(Buffer);
      if (iter != buffer2status.end()) {
        allocBeforeLoopBuffers.push_back(Buffer);
      }
    }
  }
  sortValuesByStableOrder(allocBeforeLoopBuffers, stableValueOrder);
  return allocBeforeLoopBuffers;
}

bool MemLivenessAnalysis::isSkippableOp(Operation *op) const {
  // Call-like ops are still modeled explicitly. Only pure terminators and
  // dim queries are skipped here.
  return isa<func::ReturnOp, scf::YieldOp, pto::YieldOp>(op);
}

LogicalResult
MemLivenessAnalysis::CheckIfUnknownOpTouchBuffer(Operation *op) const {
  if (isSkippableOp(op) || isGlobalWorkSpaceMemPlan()) {
    // This scene can be ignored.
    return success();
  }
  if (isOpTouchLocalBuffer(op)) {
    op->emitError("PlanMemory Fail : Unrecognized type of Operation touches "
                  "local buffer!");
    return failure();
  }
  return success();
}

void MemLivenessAnalysis::UpdateBufferAlias(Value buffer, Value aliasBuffer,
                                            bool isIgnoreInplace) {
  // union all alias buffers about `aliasBuffer` and `buffer`
  auto unionAliasSet =
      Union(GetAliasBuffers(aliasBuffer), GetAliasBuffers(buffer));
  unionAliasSet.insert(buffer);
  unionAliasSet.insert(aliasBuffer);

  // update alias map info for each buffer
  // e.g. if A alias B, C alias D, now update:
  // A alias B,C,D; B alias A,C,D; C alias A,B,D; D alias A,B,C
  for (auto buf : unionAliasSet) {
    // remove buf self from union alias set
    auto clonedAliasSet = unionAliasSet;
    clonedAliasSet.remove(buf);

    buffer2AliasVec[buf] = clonedAliasSet;
  }

  // Mark aliases that must not participate in inplace merging.
  auto it = bufferInfos.find(aliasBuffer);
  if (isIgnoreInplace && it != bufferInfos.end()) {
    it->second.ignoreInplace = true;
  }
}

SetVector<Value> MemLivenessAnalysis::Union(SetVector<Value> set1,
                                            SetVector<Value> set2) {
  SetVector<Value> unionSet;
  unionSet.insert(set1.begin(), set1.end());
  unionSet.insert(set2.begin(), set2.end());
  return unionSet;
}

SetVector<Value> MemLivenessAnalysis::GetAliasBuffers(Value aliasBuffer) {
  if (!aliasBuffer) {
    return {};
  }

  auto trueVar = buffer2AliasVec.find(aliasBuffer);
  if (trueVar != buffer2AliasVec.end()) {
    return trueVar->second;
  }
  return {};
}

void MemLivenessAnalysis::UpdateOpBufferInfo(Operation *op,
                                             const ValueRange &results) {
  for (const Value &operand : results) {
    auto it = buffer2status.find(operand);
    if (it != buffer2status.end()) {
      continue;
    }
    bufferInfos[operand] = GenerateBufferInfo(op, operand);
    buffer2status[operand] = BufferStatus::DEFFINED;
  }
}

void MemLivenessAnalysis::UpdateOpGenInfo(OpInfo *opInfo,
                                          const ValueRange &results) {
  if (results.empty()) {
    return;
  }
  for (Value operand : results) {
    auto aliasBuffers = GetAliasBuffers(operand);
    aliasBuffers.insert(operand);
    for (auto buffer : aliasBuffers) {
      UpdateOperandGenInfo(opInfo, buffer);
    }
  }
}

void MemLivenessAnalysis::UpdateOperandGenInfo(OpInfo *opInfo, Value operand) {
  auto iter_buffer = buffer2status.find(operand);
  if (iter_buffer == buffer2status.end()) {
    return;
  }
  if (iter_buffer->second == BufferStatus::DEFFINED) {
    genKillMap[opInfo].gen.push_back(operand);
    buffer2status[iter_buffer->first] = BufferStatus::GENED;
  } else if (iter_buffer->second == BufferStatus::KILLED) {
    llvm_unreachable("The buffer memory has been released and cannot be used "
                     "again! ");
  }
}

void MemLivenessAnalysis::OpKillHandle(OpInfo *opInfo, Liveness live,
                                       Block *block) {
  const auto *liveBlockInfo = live.getLiveness(block);
  auto currentLiveValues =
      liveBlockInfo->currentlyLiveValues(opInfo->operation);
  if (currentLiveValues.empty()) {
    return;
  }
  SmallVector<Value> liveValues(currentLiveValues.begin(),
                                currentLiveValues.end());
  sortValuesByStableOrder(liveValues, stableValueOrder);
  for (const Value &operand : liveValues) {
    UpdateOpKillInfo(opInfo, operand, live);
  }
}

void MemLivenessAnalysis::UpdateOpKillInfo(OpInfo *opInfo, Value operand,
                                           Liveness live) {
  auto aliasBuffers = GetAliasBuffers(operand);
  aliasBuffers.insert(operand);
  for (Value aliasBuffer : aliasBuffers) {
    auto iterBuffer = buffer2status.find(aliasBuffer);
    if (iterBuffer == buffer2status.end()) {
      return;
    }
    if (iterBuffer->second == BufferStatus::GENED &&
        IsInSameBlock(iterBuffer->first.getDefiningOp(), opInfo->operation) &&
        AllDeadAfter(opInfo->operation, aliasBuffers, live)) {
      genKillMap[opInfo].kill.push_back(aliasBuffer);
      buffer2status[iterBuffer->first] = BufferStatus::KILLED;
    }
  }
}

bool MemLivenessAnalysis::IsInSameBlock(Operation *op1, Operation *op2) const {
  return op1->getBlock() == op2->getBlock();
}

bool MemLivenessAnalysis::AllDeadAfter(Operation *op, SetVector<Value> aliasVec,
                                       Liveness live) const {
  for (auto aliasBuffer : aliasVec) {
    if (!live.isDeadAfter(aliasBuffer, op)) {
      return false;
    }
  }
  return true;
}

void MemLivenessAnalysis::RecordSemanticConflict(Value lhs, Value rhs) {
  SetVector<Value> lhsAliases = GetAliasBuffers(lhs);
  lhsAliases.insert(lhs);
  SetVector<Value> rhsAliases = GetAliasBuffers(rhs);
  rhsAliases.insert(rhs);

  auto appendUniquePair = [this](Value a, Value b) {
    if (!a || !b || a == b) {
      return;
    }
    ValuePair pair = isLessValue(a, b) ? ValuePair(a, b) : ValuePair(b, a);
    if (!llvm::is_contained(semanticConflictPairs, pair)) {
      semanticConflictPairs.push_back(pair);
    }
  };

  for (Value a : lhsAliases) {
    for (Value b : rhsAliases) {
      appendUniquePair(a, b);
    }
  }
}

BufferInfo MemLivenessAnalysis::GenerateBufferInfo(Operation *op,
                                                   Value operand) {
  auto memorySpaceAttr = GetBufferSpaceAttr(operand);
  if (isLocalMemPlan() && isLocalBuffer(memorySpaceAttr)) {
    if (!memorySpaceAttr.has_value()) {
      llvm::report_fatal_error("local buffer must have memory space");
    }
    return GetBufferInfo(op, operand,
                         memorySpaceAttr.value().getAddressSpace());
  }
  llvm_unreachable("buffer must has BufferInfo !");
}

BufferInfo MemLivenessAnalysis::GetBufferInfo(Operation *op, Value operand,
                                              pto::AddressSpace bufferScope) {
  BufferInfo bufferInfo;
  bufferInfo.operation = op;
  bufferInfo.bufferScope = bufferScope;
  // get buffer size, now for static shape
  Type elementType;
  std::optional<int64_t> footprintBytes;
  if (auto tileType = dyn_cast<TileBufType>(operand.getType())) {
    elementType = tileType.getElementType();
    footprintBytes = getTileBufferFootprintBytes(tileType);
  } else if (auto multiType = dyn_cast<MultiTileBufType>(operand.getType())) {
    TileBufType slotType = multiType.getSlotType();
    elementType = slotType.getElementType();
    footprintBytes = getTileBufferFootprintBytes(slotType);
  } else {
    llvm_unreachable("local memory planner expects tile buffer roots");
  }
  bufferInfo.bufferType = elementType;
  if (!footprintBytes.has_value()) {
    llvm::report_fatal_error(
        "failed to obtain buffer static physical footprint");
  }
  bufferInfo.constBits = footprintBytes.value() * kBitsPerByte;
  return bufferInfo;
}

void MemLivenessAnalysis::GenerateBufferLife() {
  int scopeTime = 0;
  for (size_t i = 0; i < linearOperation.size(); ++i) {
    auto it = genKillMap.find(linearOperation[i].get());
    if (it == genKillMap.end()) {
      scopeTime++;
      continue;
    }
    // Time given to buffer start.
    for (const Value &genBuffer : it->second.gen) {
      std::unique_ptr<BufferLife> bufferLife =
          std::make_unique<BufferLife>(genBuffer);
      bufferLife->allocTime = scopeTime;
      buffer2Life[genBuffer] = std::move(bufferLife);
    }
    // Time given to buffer end.
    for (const Value &killBuffer : it->second.kill) {
      auto iter = buffer2Life.find(killBuffer);
      if (iter == buffer2Life.end()) {
        llvm::report_fatal_error("buffer lifetime killed before generation");
      }
      iter->second->freeTime = scopeTime;
    }
    scopeTime++;
  }
}

std::shared_ptr<BufferLife>
StorageEntry::GetBufferLifeByValue(const Value v) const {
  auto find = std::find_if(
      bufferLifeVec.begin(), bufferLifeVec.end(),
      [v](std::shared_ptr<BufferLife> life) { return life->buffer == v; });
  if (find != bufferLifeVec.end()) {
    return *find;
  }
  return nullptr;
}

bool MemPlan::IsReusePTOOp(Operation *op) const {
  if (restrictInplaceAsISA) {
    return false;
  }

  // not in ISA but confirmed with hardware developers:
  // elementwise ops with the same shape and the same bitwidth operands can also
  // do memory inplace for src and dst
  return false;
}

SmallVector<ValuePair> MemPlan::GenerateInplaceList() {
  SmallVector<ValuePair> inplaceList;
  DenseMap<Operation *, bool> hasTouchOp;
  inplaceList.insert(inplaceList.end(), inplacePairList.begin(),
                     inplacePairList.end());
  for (auto &operationSeq : linearOperation) {
    auto it = genKillMap.find(operationSeq.get());
    if (it == genKillMap.end()) {
      continue;
    }
    if (hasTouchOp[operationSeq->operation]) {
      continue;
    }

    SmallVector<Value> genBuffers(it->second.gen.begin(), it->second.gen.end());
    SmallVector<Value> killBuffers(it->second.kill.begin(), it->second.kill.end());
    sortValuesByStableOrder(genBuffers, stableValueOrder);
    sortValuesByStableOrder(killBuffers, stableValueOrder);

    for (const Value &genBuffer : genBuffers) {
      auto genBufferIter = bufferInfos.find(genBuffer);
      if (genBufferIter == bufferInfos.end()) {
        llvm::report_fatal_error("gen buffer missing from buffer info map");
      }
      if (genBufferIter->second.ignoreInplace) {
        continue;
      }

      for (const Value &killBuffer : killBuffers) {
        auto killBufferIter = bufferInfos.find(killBuffer);
        if (killBufferIter == bufferInfos.end()) {
          llvm::report_fatal_error("kill buffer missing from buffer info map");
        }
        if (killBufferIter->second.ignoreInplace) {
          continue;
        }

        bool bufferSizeMatch =
            killBufferIter->second.constBits >= genBufferIter->second.constBits;
        bool isResuableOp = IsReusePTOOp(it->first->operation);
        bool canInplace = bufferSizeMatch && isResuableOp;
        if (canInplace) {
          inplaceList.emplace_back(std::make_pair(genBuffer, killBuffer));
          break;
        }
      }
    }
    // Nodes in inplace are only processed once.
    hasTouchOp[operationSeq->operation] = true;
  }
  return inplaceList;
}

void MemPlan::EmitPlanMemoryFailureInfo() {
  if (failApplyBufferInfo.empty()) {
    return;
  }
  for (auto &iter : failApplyBufferInfo) {
    AddressSpace space = iter.first;
    func_.emitError() << stringifyEnum(space) << " overflow, requires "
                      << iter.second << " bits while "
                      << GetBufferSpaceInfo(space).second << " bits avaliable!";
  }
}

bool MemPlan::RecordOverflowIfAny() {
  if (!failApplyBufferInfo.empty()) {
    return true;
  }
  if (planMode != MemPlanMode::LOCAL_MEM_PLAN ||
      memscope2rootStorageEntry.empty()) {
    return false;
  }

  for (auto &it : memscope2rootStorageEntry) {
    auto *rootStorageEntry = it.second;
    if (!rootStorageEntry) {
      continue;
    }
    auto bufferSpaceInfo =
        GetBufferSpaceInfo(rootStorageEntry->bufInfo->bufferScope);
    size_t maxBits = bufferSpaceInfo.second;
    uint64_t maxAllocBits = rootStorageEntry->alignedConstBits;
    for (auto *child : rootStorageEntry->mergedChildren) {
      maxAllocBits =
          std::max(maxAllocBits, child->bitsOffset + child->alignedConstBits);
    }
    if (maxAllocBits > maxBits) {
      failApplyBufferInfo[rootStorageEntry->bufInfo->bufferScope] =
          maxAllocBits;
    }
  }

  return !failApplyBufferInfo.empty();
}

bool MemPlan::HasSemanticConflict(const StorageEntry *entry,
                                  const BufferLifeVec &bufferLives) const {
  if (!entry || semanticConflictPairs.empty() || bufferLives.empty()) {
    return false;
  }

  auto containsPair = [this](Value lhs, Value rhs) {
    ValuePair pair = isLessValue(lhs, rhs) ? ValuePair(lhs, rhs)
                                           : ValuePair(rhs, lhs);
    return llvm::is_contained(semanticConflictPairs, pair);
  };

  for (Value entryBuffer : entry->inplaceBuffers) {
    for (const auto &life : bufferLives) {
      if (!life) {
        continue;
      }
      Value otherBuffer = life->buffer;
      if (!otherBuffer || entryBuffer == otherBuffer) {
        continue;
      }
      if (containsPair(entryBuffer, otherBuffer)) {
        return true;
      }
    }
  }
  return false;
}

// Plan Memory algorithm.
LogicalResult MemPlan::plan() {
  // Construct StorageEntry structure.
  GenerateStorageEntry();
  // Plan memory address.
  PlanStatus as = planMode == MemPlanMode::LOCAL_MEM_PLAN
                      ? PlanLocalMemAddress()
                      : PlanWorkSpaceMemAddress();
  if (as == PlanStatus::PLAN_FAILED) {
    EmitPlanMemoryFailureInfo();
    return failure();
  }
  if (RecordOverflowIfAny()) {
    EmitPlanMemoryFailureInfo();
    return failure();
  }
  auto hasAddressOverlap = [](const StorageEntry *lhs, const StorageEntry *rhs) {
    uint64_t lhsBegin = lhs->bitsOffset;
    uint64_t lhsEnd = lhs->bitsOffset + lhs->alignedConstBits;
    uint64_t rhsBegin = rhs->bitsOffset;
    uint64_t rhsEnd = rhs->bitsOffset + rhs->alignedConstBits;
    return lhsBegin < rhsEnd && rhsBegin < lhsEnd;
  };
  SmallVector<const StorageEntry *> plannedEntries;
  plannedEntries.reserve(StorageEntryVec.size() + pingEntry2RelationPongEntry.size());
  for (const auto &entry : StorageEntryVec) {
    plannedEntries.push_back(entry.get());
  }
  for (const auto &entry : pingEntry2RelationPongEntry) {
    plannedEntries.push_back(entry.second.get());
  }
  for (size_t i = 0; i < plannedEntries.size(); ++i) {
    for (size_t j = i + 1; j < plannedEntries.size(); ++j) {
      const StorageEntry *lhs = plannedEntries[i];
      const StorageEntry *rhs = plannedEntries[j];
      if (!lhs || !rhs) {
        continue;
      }
      if (lhs->bufInfo->bufferScope != rhs->bufInfo->bufferScope) {
        continue;
      }
      if (!hasAddressOverlap(lhs, rhs)) {
        continue;
      }
      bool lifeOverlap =
          !GetOverlapBufferLife(lhs->bufferLifeVec, rhs->bufferLifeVec).empty();
      bool semanticConflict = HasSemanticConflict(lhs, rhs->bufferLifeVec);
      if (!lifeOverlap && !semanticConflict) {
        continue;
      }
      func_.emitError()
          << "PlanMemory produced overlapping local buffers in "
          << stringifyEnum(lhs->bufInfo->bufferScope)
          << " at offsets " << lhs->bitsOffset << " and " << rhs->bitsOffset;
      return failure();
    }
  }
  // Update the address information of each buffer after memory buffer.
  UpdateBuffer2Offsets();
  if (enablePrintMemoryAllocatedSize) {
    PrintSuccessfulAllocatedMaxBits();
  }
  return success();
}

void MemPlan::GenerateStorageEntry() {
  // create new storage entry.
  for (auto &operation : linearOperation) {
    auto it = genKillMap.find(operation.get());
    if (it == genKillMap.end()) {
      continue;
    }
    SmallVector<Value> genBuffers(it->second.gen.begin(), it->second.gen.end());
    sortValuesByStableOrder(genBuffers, stableValueOrder);
    for (const Value &genBuffer : genBuffers) {
      auto iter = bufferInfos.find(genBuffer);
      if (iter == bufferInfos.end()) {
        continue;
      }
      const std::shared_ptr<BufferLife> &bufLife = buffer2Life.at(genBuffer);
      std::unique_ptr<StorageEntry> entry = std::make_unique<StorageEntry>();
      entry->bufInfo = &iter->second;
      entry->bufferLifeVec.emplace_back(bufLife);
      entry->inplaceBuffers.emplace_back(iter->first);
      auto multiBuffer = buffer2MultiNum.find(genBuffer);
      if (multiBuffer != buffer2MultiNum.end()) {
        entry->multiBufferNum = multiBuffer->second;
      }
      buffer2storageEntry[genBuffer] = entry.get();
      // Verify the validity of parameters after initialization.
      ValidateParameters(entry);
      StorageEntryVec.emplace_back(std::move(entry));
    }
  }
}

void MemPlan::PrintSuccessfulAllocatedMaxBits() {
  auto it = memscope2rootStorageEntry.find(pto::AddressSpace::VEC);
  if (it != memscope2rootStorageEntry.end()) {
    if (!it->second) {
      llvm::report_fatal_error("missing root storage entry for VEC scope");
    }
    uint64_t ubAllocBits = it->second->alignedConstBits + it->second->bitsOffset;
    for (auto& child : it->second->mergedChildren) {
      ubAllocBits = std::max(ubAllocBits, child->bitsOffset + child->alignedConstBits);
    }
    llvm::outs() << "[PTOPlanMemory] Allocated UB size = " << ubAllocBits
                 << " bits\n";
  }
}

void MemPlan::ValidateParameters(std::unique_ptr<StorageEntry> &e) const {
  if (!e->bufInfo->operation) {
    llvm::report_fatal_error("storage entry missing defining operation");
  }
  if (e->bufInfo->constBits < 0U) {
    llvm::report_fatal_error("storage entry has invalid memory size");
  }
  if (e->bufferLifeVec.empty()) {
    llvm::report_fatal_error("storage entry missing lifetime information");
  }
}

void MemPlan::UpdateBuffer2Offsets() {
  for (auto &e : StorageEntryVec) {
    // Skip sibling (slot >= 1) entries -- their offsets are written via the
    // primary entry's `relationOtherBuffers` traversal below. Without this
    // skip the sibling offsets would be appended in StorageEntryVec order
    // rather than slot order, breaking the runtime contract that
    // `buffer2Offsets[buffer][k]` is slot k's physical offset.
    if (e->isMultiBufferSlot) {
      continue;
    }
    for (Value &buffer : e->inplaceBuffers) {
      buffer2Offsets[buffer].push_back(
          (e->bitsOffset + kBitsToByte - 1) / kBitsToByte);
      // Multi-buffer primary: append sibling offsets in slot order so the
      // final offsets list is [slot0, slot1, ..., slotN-1].
      for (auto *sibling : e->relationOtherBuffers) {
        if (!sibling) {
          continue;
        }
        buffer2Offsets[buffer].push_back(
            (sibling->bitsOffset + kBitsToByte - 1) / kBitsToByte);
      }
    }
  }
  // In the MultiBuffer scenario, single reuse db will result in additional
  // storageEntry.
  UpdateMultiBufferReuseExtraOffset();
}

void MemPlan::UpdateMultiBufferReuseExtraOffset() {
  if (pingEntry2RelationPongEntry.empty()) {
    return;
  }

  for (auto &relationEntry : pingEntry2RelationPongEntry) {
    for (Value &buffer : relationEntry.second->inplaceBuffers) {
      // MultiBuffer can cause multiple addrs.
      buffer2Offsets[buffer].push_back(
          (relationEntry.second->bitsOffset + kBitsToByte - 1) /
          kBitsToByte);
    }
  }
}

void MemPlan::MergeInplaceSE() {
  // get the list of inplace value pair.
  SmallVector<ValuePair> inplaceList = GenerateInplaceList();
  // try to merge storage entries. genSE is replaced by KillSE.
  for (const auto &pairIter : inplaceList) {
    const StorageEntry *genSE = buffer2storageEntry[pairIter.first];
    StorageEntry *killSE = buffer2storageEntry[pairIter.second];
    if (genSE == killSE) {
      // already same storageEntry, no need to inplace.
      continue;
    }
    if (genSE == nullptr || killSE == nullptr) {
      llvm::report_fatal_error("invalid storage entry during inplace merge");
    }
    BufferLifeVec mergedBufferLifeVec;
    mergedBufferLifeVec.insert(mergedBufferLifeVec.end(),
                               genSE->bufferLifeVec.begin(),
                               genSE->bufferLifeVec.end());
    mergedBufferLifeVec.insert(mergedBufferLifeVec.end(),
                               killSE->bufferLifeVec.begin(),
                               killSE->bufferLifeVec.end());
    MergeBufferVec(mergedBufferLifeVec);
    killSE->bufferLifeVec.swap(mergedBufferLifeVec);
    //  merge allocs of two storage entry and update inplaceBuffers.
    killSE->inplaceBuffers.insert(killSE->inplaceBuffers.begin(),
                                  genSE->inplaceBuffers.begin(),
                                  genSE->inplaceBuffers.end());

    // Take the maximum value for the inplace scene.
    killSE->multiBufferNum =
        std::max(genSE->multiBufferNum, killSE->multiBufferNum);

    // all buffers have same storage entry after merging
    for (auto &buffer : genSE->inplaceBuffers) {
      buffer2storageEntry[buffer] = killSE;
    }
    // remove the alloc info of dst after successful merging
    auto e = std::find_if(StorageEntryVec.begin(), StorageEntryVec.end(),
                          [genSE](std::unique_ptr<StorageEntry> &se) {
                            return se.get() == genSE;
                          });
    StorageEntryVec.erase(e);
  }
}

PlanStatus MemPlan::PlanLocalMemAddress() {
  // merge from the first storage entry
  MergeInplaceSE();
  dmaFirstPipelineOpt.build(func_);
  ExpandMultiBufferStorageEntry();
  MergeSameScopeSE();
  return PlanMemAddressOfWholeLocalBuffer();
}

PlanStatus MemPlan::PlanWorkSpaceMemAddress() {
  // merge from the first storage entry
  MergeInplaceSE();
  ExpandMultiBufferStorageEntry();
  return PlanMemOffsetOfWholeWorkSpace();
}

PlanStatus MemPlan::PlanMemOffsetOfWholeWorkSpace() {
  for (auto &it : workSpaceArg2rootStorageEntry) {
    StorageEntry *rootStorageEntry = it.second;
    if (!enableGlobalReuse) {
      GlobalWorkspaceNoReuse(rootStorageEntry);
      continue;
    }
    MemBoundList outline;
    PlanRecHis history;
    SpecInfo si;
    // Can be reuse without conflicting life intervals.
    si.specLevel = si.minLevel;
    int childrenNum = static_cast<int>(rootStorageEntry->mergedChildren.size());
    outline.push_back(std::make_shared<MemoryBound>(
        BufferLifeVec(), 0, std::numeric_limits<uint64_t>::max(), nullptr));

    // The initial value is rootStorageEntry.
    StorageEntry *curEntry = rootStorageEntry;
    while (si.childIdx < childrenNum) {
      curEntry->alignedConstBits =
          static_cast<uint64_t>(curEntry->bufInfo->constBits);
      curEntry->childIdx = si.childIdx;
      LogicalResult planResult = MultiSpecPlan(si, outline, history, curEntry);
      if (failed(planResult)) {
        return PlanStatus::PLAN_FAILED;
      }
      if (si.childIdx >= childrenNum) {
        break;
      }
      curEntry = rootStorageEntry->mergedChildren[si.childIdx];
    }
  }
  planStatus = PlanStatus::PLAN_SUCCESS;
  return planStatus;
}

void MemPlan::GlobalWorkspaceNoReuse(StorageEntry *rootStorageEntry) {
  rootStorageEntry->bitsOffset = 0;
  uint64_t offset = static_cast<uint64_t>(rootStorageEntry->bufInfo->constBits);
  for (StorageEntry *child : rootStorageEntry->mergedChildren) {
    child->bitsOffset = offset;
    offset += static_cast<uint64_t>(child->bufInfo->constBits);
  }
}

void MemPlan::ExpandMultiBufferStorageEntry() {
  // For each multi-buffer primary entry, create (N - 1) sibling entries so
  // the planner can lay out one physical slot per sibling. Siblings are
  // pushed into `StorageEntryVec` and participate in normal Stage0/Stage2
  // address allocation. The primary keeps `relationOtherBuffers` pointing
  // at the siblings in slot order (slot 1..N-1), and `relationPongEntry`
  // aliases the first sibling so existing N == 2 codepaths keep working.
  size_t size = StorageEntryVec.size();
  for (size_t i = 0; i < size; i++) {
    auto *primary = StorageEntryVec[i].get();
    if (primary->multiBufferNum <= 1) {
      continue;
    }
    uint32_t n = primary->multiBufferNum;
    for (uint32_t slot = 1; slot < n; ++slot) {
      auto entry = std::make_unique<StorageEntry>();
      entry->bufInfo = primary->bufInfo;
      entry->bufferLifeVec = primary->bufferLifeVec;
      entry->alignedConstBits = primary->alignedConstBits;
      entry->inplaceBuffers = primary->inplaceBuffers;
      entry->multiBufferNum = n;
      entry->isMultiBufferSlot = true;
      primary->relationOtherBuffers.push_back(entry.get());
      StorageEntryVec.push_back(std::move(entry));
    }
    if (!primary->relationOtherBuffers.empty()) {
      primary->relationPongEntry = primary->relationOtherBuffers.front();
    }
  }
}

bool MemPlan::IsEnoughForBuffersNoReuse(StorageEntry *rootStorageEntry,
                                        size_t restBufferSize,
                                        size_t alignUnit) {
  auto iter =
      bufferScope2RequiredSize.find(rootStorageEntry->bufInfo->bufferScope);
  if (iter == bufferScope2RequiredSize.end()) {
    llvm::report_fatal_error("missing required-size entry for buffer scope");
  }
  if (iter->second < restBufferSize) {
    // Even when the scope fits without reuse (no peak to save), honor
    // largest-first placement so the option means the same thing on both paths:
    // a deterministic decreasing-size layout regardless of whether reuse kicks
    // in. Stable sort keeps uniform-size scopes byte-identical to the default.
    if (orderBySize) {
      rootStorageEntry = GetSizeOrderedRootStorageEntry(rootStorageEntry);
    }
    PlanBuffersWithoutReuse(rootStorageEntry, alignUnit);
    return true;
  }
  return false;
}

void MemPlan::PlanBuffersWithoutReuse(StorageEntry *rootStorageEntry,
                                      size_t alignUnit) {
  uint offset = 0;
  rootStorageEntry->bitsOffset = offset;
  offset = AlignUp(rootStorageEntry->bufInfo->constBits, alignUnit);
  rootStorageEntry->alignedConstBits = offset;
  for (StorageEntry *child : rootStorageEntry->mergedChildren) {
    child->bitsOffset = offset;
    uint64_t alignedBits = AlignUp(child->bufInfo->constBits, alignUnit);
    offset += alignedBits;
    child->alignedConstBits = alignedBits;
  }
}

void MemPlan::MergeSameScopeSE() {
  // Construct root StorageEntry and collect the same scope StorageEntry
  for (auto &iter : StorageEntryVec) {
    auto iter_scope =
        memscope2rootStorageEntry.find(iter->bufInfo->bufferScope);
    if (iter_scope == memscope2rootStorageEntry.end()) {
      memscope2rootStorageEntry[iter->bufInfo->bufferScope] = iter.get();
    } else {
      iter_scope->second->mergedChildren.push_back(iter.get());
    }
  }

  // set bufferScope2RequiredSize for all StorageEntry
  for (auto &rootStorageEntry : memscope2rootStorageEntry) {
    auto bufferSpaceInfo = GetBufferSpaceInfo(rootStorageEntry.first);
    size_t accumulateSize = AlignUp(rootStorageEntry.second->bufInfo->constBits,
                                    bufferSpaceInfo.first);
    for (auto &childrenStorageEntry : rootStorageEntry.second->mergedChildren) {
      size_t curStorageSize = AlignUp(childrenStorageEntry->bufInfo->constBits,
                                      bufferSpaceInfo.first);
      accumulateSize = accumulateSize + curStorageSize;
    }
    bufferScope2RequiredSize[rootStorageEntry.first] = accumulateSize;
  }
}

void MemPlan::PlanMemAddressForLevel0(
    StorageEntry *rootStorageEntry) {
  // get the buffer info for a given scope.
  auto bufferSpaceInfo =
      GetBufferSpaceInfo(rootStorageEntry->bufInfo->bufferScope);
  size_t align = bufferSpaceInfo.first;
  size_t maxBits = UINT64_MAX;
  rootStorageEntry = GetReorderRootStorageEntry(rootStorageEntry);
  // memory outline in a given buffer scope.
  MemBoundList outline;
  PlanRecHis history;
  SpecInfo si;
  si.specLevel = SPEC_LEVEL_0;
  si.maxLevel = SPEC_LEVEL_0;
  int childrenNum = static_cast<int>(rootStorageEntry->mergedChildren.size());
  outline.push_back(
      std::make_shared<MemoryBound>(BufferLifeVec(), 0, maxBits, nullptr));

  // The initial value is rootStorageEntry.
  StorageEntry *curEntry = rootStorageEntry;
  while (si.childIdx < childrenNum) {
    uint64_t needBits = static_cast<uint64_t>(curEntry->bufInfo->constBits);
    curEntry->alignedConstBits = AlignUp(needBits, align);
    curEntry->childIdx = si.childIdx;
    (void)MultiSpecPlan(si, outline, history, curEntry);
    if (si.childIdx >= childrenNum) {
      break;
    }
    curEntry = rootStorageEntry->mergedChildren[si.childIdx];
  }
  // Find the max appled bits from all children and root, which is the max
  // memory applied in this buffer space.
  uint64_t maxAllocBits = rootStorageEntry->alignedConstBits;
  auto children = rootStorageEntry->mergedChildren;
  for (auto *child : children) {
    maxAllocBits =
        std::max(maxAllocBits, child->bitsOffset + child->alignedConstBits);
  }
  failApplyBufferInfo[rootStorageEntry->bufInfo->bufferScope] = maxAllocBits;
}

PlanStatus MemPlan::PlanMemAddressOfWholeLocalBuffer() {
  // Start plan
  for (auto &it : memscope2rootStorageEntry) {
    StorageEntry *rootStorageEntry = it.second;
    // get the buffer info for a given scope.
    auto bufferSpaceInfo =
        GetBufferSpaceInfo(rootStorageEntry->bufInfo->bufferScope);
    size_t align = bufferSpaceInfo.first;
    size_t maxBits = bufferSpaceInfo.second;
    if (rootStorageEntry->mergedChildren.empty()) {
      PlanStatus status = PlanSingleLocalBuffer(rootStorageEntry, align, maxBits);
      if (status != PlanStatus::PLAN_SUCCESS) {
        return status;
      }
      continue;
    }
    if (IsEnoughForBuffersNoReuse(rootStorageEntry, maxBits, align)) {
      continue;
    }
    PlanStatus status = PlanReusableLocalBuffer(rootStorageEntry, align, maxBits);
    if (status != PlanStatus::PLAN_SUCCESS) {
      return status;
    }
  }
  planStatus = PlanStatus::PLAN_SUCCESS;
  return planStatus;
}

PlanStatus MemPlan::PlanSingleLocalBuffer(StorageEntry *rootStorageEntry,
                                          size_t align, size_t maxBits) {
  uint64_t needAlignedBits = AlignUp(rootStorageEntry->bufInfo->constBits, align);
  if (needAlignedBits > maxBits) {
    failApplyBufferInfo[rootStorageEntry->bufInfo->bufferScope] =
        needAlignedBits;
    return PlanStatus::PLAN_FAILED;
  }
  rootStorageEntry->bitsOffset = 0;
  rootStorageEntry->alignedConstBits = needAlignedBits;
  return PlanStatus::PLAN_SUCCESS;
}

PlanStatus MemPlan::PlanReusableLocalBuffer(StorageEntry *rootStorageEntry,
                                            size_t align, size_t maxBits) {
  rootStorageEntry = GetReorderRootStorageEntry(rootStorageEntry);
  ReportMemLifeDebugInfo(rootStorageEntry);

  MemBoundList outline;
  PlanRecHis history;
  SpecInfo si;
  si.specLevel = si.maxLevel;
  int childrenNum = static_cast<int>(rootStorageEntry->mergedChildren.size());
  outline.push_back(
      std::make_shared<MemoryBound>(BufferLifeVec(), 0, maxBits, nullptr));

  StorageEntry *curEntry = rootStorageEntry;
  while (si.childIdx < childrenNum) {
    uint64_t needBits = static_cast<uint64_t>(curEntry->bufInfo->constBits);
    curEntry->alignedConstBits = AlignUp(needBits, align);
    curEntry->childIdx = si.childIdx;
    LDBG("\n");
    LDBG("----------Need-Plan-CurEntry---------\n");
    ReportCurEntryDebugInfo(curEntry);
    LDBG("\n");
    LogicalResult planResult = MultiSpecPlan(si, outline, history, curEntry);
    if (failed(planResult)) {
      StatusWrapper statusWrapper = {false,   curEntry->alignedConstBits,
                                     &si,     outline,
                                     history, rootStorageEntry};
      LDBG("\n");
      LDBG("----------ApplyFailStrategy---------\n");
      ReportCurEntryDebugInfo(curEntry);
      LDBG("\n");
      PlanStatus status = ApplyFailStrategy(statusWrapper, maxBits);
      if (status == PlanStatus::RESTART_NEW_PLAN) {
        si = SpecInfo();
        curEntry = rootStorageEntry;
        continue;
      }
      if (status == PlanStatus::PLAN_FAILED) {
        ReportAllocatedEntryDebugInfo(rootStorageEntry);
        PlanMemAddressForLevel0(rootStorageEntry);
        return status;
      }
    }
    if (si.childIdx >= childrenNum) {
      break;
    }
    curEntry = rootStorageEntry->mergedChildren[si.childIdx];
  }
  return PlanStatus::PLAN_SUCCESS;
}

void MemPlan::ReportMemLifeDebugInfo(StorageEntry *rootStorageEntry) {
  LDBG("-------------------------- Buffer2Life --------------------------\n");
  MemLifeDebugInfo(rootStorageEntry);
  for (auto &StorageEntry : rootStorageEntry->mergedChildren) {
    MemLifeDebugInfo(StorageEntry);
  }
}

void MemPlan::MemLifeDebugInfo(StorageEntry *storageEntry) {
  for (auto &buffer : storageEntry->inplaceBuffers) {
    (void)buffer;
    LDBG("Buffer : " << buffer << "\n");
  }
  for (auto &bufferLife : storageEntry->bufferLifeVec) {
    (void)bufferLife;
    LDBG("bufferLife : "
         << "allocTime : " << bufferLife->allocTime
         << " , freeTime : " << bufferLife->freeTime << "\n");
  }
  LDBG("\n");
}

void MemPlan::ReportCurEntryDebugInfo(const StorageEntry *curEntry) {
  for (auto &buffer : curEntry->inplaceBuffers) {
    (void)buffer;
    LDBG("buffer : " << buffer);
  }
}

StorageEntry *
MemPlan::GetReorderRootStorageEntry(StorageEntry *rootStorageEntry) {
  if (orderBySize) {
    return GetSizeOrderedRootStorageEntry(rootStorageEntry);
  }
  if (rootStorageEntry->bufInfo->bufferScope != pto::AddressSpace::VEC) {
    return rootStorageEntry;
  }
  SmallVector<StorageEntry *> origStorageEntryVec = {rootStorageEntry};
  origStorageEntryVec.insert(origStorageEntryVec.end(),
                             rootStorageEntry->mergedChildren.begin(),
                             rootStorageEntry->mergedChildren.end());

  // reorder storage entrys: dma touched buffers + other buffers + scalar
  // touched buffers
  SmallVector<StorageEntry *> reorderedStorageEntryVec;
  SmallVector<StorageEntry *> touchPipeScalarStorageEntryVec;
  for (auto &storageEntry : origStorageEntryVec) {
    for (auto &buffer : storageEntry->inplaceBuffers) {
      if (dmaFirstPipelineOpt.IsDmaBuffer(buffer)) {
        reorderedStorageEntryVec.push_back(storageEntry);
        break;
      }
      if (dmaFirstPipelineOpt.IsScalarBuffer(buffer)) {
        touchPipeScalarStorageEntryVec.push_back(storageEntry);
        break;
      }
    }
  }
  for (auto &storageEntry : origStorageEntryVec) {
    auto it1 = std::find(reorderedStorageEntryVec.begin(),
                         reorderedStorageEntryVec.end(), storageEntry);
    auto it2 = std::find(touchPipeScalarStorageEntryVec.begin(),
                         touchPipeScalarStorageEntryVec.end(), storageEntry);
    if (it1 == reorderedStorageEntryVec.end() &&
        it2 == touchPipeScalarStorageEntryVec.end()) {
      reorderedStorageEntryVec.push_back(storageEntry);
    }
  }

  reorderedStorageEntryVec.insert(reorderedStorageEntryVec.end(),
                                  touchPipeScalarStorageEntryVec.begin(),
                                  touchPipeScalarStorageEntryVec.end());

  // Ensure that ping pong is continuously plan mem in the multi buffer.
  ReorderContinuousPingPongEntry(reorderedStorageEntryVec);
  StorageEntry *reorderedRootStorageEntry = reorderedStorageEntryVec[0];
  reorderedRootStorageEntry->mergedChildren.clear();
  for (size_t j = 1; j < reorderedStorageEntryVec.size(); ++j) {
    reorderedRootStorageEntry->mergedChildren.push_back(
        reorderedStorageEntryVec[j]);
  }
  return reorderedRootStorageEntry;
}

void MemPlan::ReorderContinuousPingPongEntry(
    SmallVector<StorageEntry *> &storageEntryVec) {
  SmallVector<StorageEntry *> reorderedStorageEntryVec;
  for (auto &storageEntry : storageEntryVec) {
    auto it = std::find(reorderedStorageEntryVec.begin(),
                        reorderedStorageEntryVec.end(), storageEntry);
    if (it == reorderedStorageEntryVec.end()) {
      reorderedStorageEntryVec.push_back(storageEntry);
      if (storageEntry->multiBufferNum == kDoubleBufferCount &&
          storageEntry->relationPongEntry) {
        // Ping Pong continuous save.
        reorderedStorageEntryVec.push_back(storageEntry->relationPongEntry);
      }
    }
  }
  reorderedStorageEntryVec.swap(storageEntryVec);
}

StorageEntry *
MemPlan::GetSizeOrderedRootStorageEntry(StorageEntry *rootStorageEntry) {
  // First-fit-decreasing: place the largest buffers first. For the heterogeneous
  // buffer sizes that real kernels produce, decreasing-size order packs tighter
  // than an arbitrary/DMA-first order (this is the ordering XLA, TVM and SOMAS
  // all use). Applies to every memory space, unlike the DMA-first reorder which
  // is VEC-only.
  SmallVector<StorageEntry *> entries = {rootStorageEntry};
  entries.insert(entries.end(), rootStorageEntry->mergedChildren.begin(),
                 rootStorageEntry->mergedChildren.end());

  // Stable sort by decreasing buffer size. Stable keeps the original order among
  // equal-size buffers, so uniform-size instances (e.g. the plan_memory_* tests)
  // are left untouched.
  std::stable_sort(entries.begin(), entries.end(),
                   [](const StorageEntry *a, const StorageEntry *b) {
                     return a->bufInfo->constBits > b->bufInfo->constBits;
                   });

  // Keep ping-pong (double-buffer) pairs contiguous so double-buffering is
  // preserved (same post-processing the DMA-first path applies).
  ReorderContinuousPingPongEntry(entries);

  // Rebuild the flat root -> children structure around the new (largest) root.
  // Clear every entry's child list first: when the root changes, the previous
  // root would otherwise keep its stale child list (forming a cycle), and only
  // the new root should carry the flat list of the others.
  StorageEntry *reorderedRootStorageEntry = entries[0];
  for (StorageEntry *entry : entries) {
    entry->mergedChildren.clear();
  }
  for (size_t j = 1; j < entries.size(); ++j) {
    reorderedRootStorageEntry->mergedChildren.push_back(entries[j]);
  }
  // Keep the scope -> root map consistent so later consumers (RecordOverflowIfAny,
  // PrintSuccessfulAllocatedMaxBits) read the new root and its full child list.
  // This must accompany the clear above: clearing children without updating the
  // map would leave the stale root pointing at an empty child list.
  memscope2rootStorageEntry[reorderedRootStorageEntry->bufInfo->bufferScope] =
      reorderedRootStorageEntry;
  return reorderedRootStorageEntry;
}

std::pair<size_t, size_t>
MemPlan::GetBufferSpaceInfo(pto::AddressSpace &space) const {
  switch (space) {
  case pto::AddressSpace::VEC:
    return std::make_pair(plannerAlignBitsFromBytes(ubAlignSize), ubSpaceSize);
  case pto::AddressSpace::MAT:
    return std::make_pair(plannerAlignBitsFromBytes(l1AlignSize), l1SpaceSize);
  case pto::AddressSpace::ACC:
    return std::make_pair(plannerAlignBitsFromBytes(l0cAlignSize),
                          l0cSpaceSize);
  case pto::AddressSpace::LEFT:
    return std::make_pair(plannerAlignBitsFromBytes(l0aAlignSize),
                          l0aSpaceSize);
  case pto::AddressSpace::RIGHT:
    return std::make_pair(plannerAlignBitsFromBytes(l0bAlignSize),
                          l0bSpaceSize);
  case pto::AddressSpace::BIAS:
    return std::make_pair(plannerAlignBitsFromBytes(biasAlignSize),
                          biasSpaceSize);
  case pto::AddressSpace::SCALING:
    return std::make_pair(plannerAlignBitsFromBytes(scalingAlignSize),
                          scalingSpaceSize);
  case pto::AddressSpace::Zero:
  case pto::AddressSpace::GM:
    return std::make_pair(size_t{0}, size_t{0});
  }

  llvm_unreachable("Temporarily unsupported memory buffer space !");
}

LogicalResult MemPlan::MultiSpecPlan(SpecInfo &si, MemBoundList &outline,
                                     PlanRecHis &history, StorageEntry *entry) {
  LogicalResult planResult = failure();
  for (int i = si.specLevel; i >= si.minLevel; i--) {
    planResult = SpecAlloc(outline, history, entry, si, i);
    if (succeeded(planResult)) {
      if (si.childIdx == si.specStartIdx) {
        // In roll back plan, when the specified specStartIdx is reached,
        // the subsequent plan still adopts the maxLevel strategy.
        si.specLevel = si.maxLevel;
      }
      si.childIdx++;
      break;
    }
  }
  return planResult;
}

LogicalResult MemPlan::SpecAlloc(MemBoundList &outline, PlanRecHis &his,
                                 StorageEntry *e, const SpecInfo &si,
                                 int localLevel) {
  if (std::any_of(his.begin(), his.end(),
                  [e](PlanRecord &r) { return r.entry && r.entry == e; })) {
    // If the plan has already been completed, return success directly.
    return success();
  }
  for (MemBoundListConstIter start = outline.begin(); start != outline.end();
       ++start) {
    uint64_t size = 0;
    uint64_t allocOffset = (*start)->offset;
    for (MemBoundListConstIter end = start; end != outline.end(); ++end) {
      std::shared_ptr<MemoryBound> last = *end;
      size += last->extent;
      // if index & addr are as same as last rollback result,
      // continue to find next result
      if (IsSamePlanAsLastRollBack(allocOffset, e->childIdx, si) ||
          VerifyConflictStage0(e, last)) {
        start = end;
        break;
      }
      if (size < e->alignedConstBits) {
        continue;
      }
      // If SPEC_LEVEL_1, then the address of pong Offset address needs to be
      // allocated.
      uint64_t pongOffset{0};
      if (localLevel == SPEC_LEVEL_1 &&
          VerifyConflictStage1(outline, his, e,
                               OutlineSectionInfo(start, end, size, false),
                               pongOffset)) {
        break;
      }

      if (VerifyConflictStage2(his, e, localLevel, start, outline)) {
        break;
      }
      e->bitsOffset = allocOffset;
      UpdateOutline(outline, his, e,
                    OutlineSectionInfo(start, end, size, false), localLevel);

      if (localLevel == SPEC_LEVEL_1) {
        // There is no conflict with the historical plan of buffer life, and
        // the address of the Pong can be assigned.
        PlanRelationPongEntryAddress(pongOffset, e);
        SpecAllocRelationPongEntry(outline, his, e, pongOffset);
      }
      LDBG("APPLY_SPEC_LEVEL:  " << localLevel << "\n");
      bool needRecord =
          allocatedEntry.end() ==
          std::find(allocatedEntry.begin(), allocatedEntry.end(), e);
      if (needRecord) {
        allocatedEntry.push_back(e);
      }
      return success();
    }
  }
  return failure();
}

LoopLikeOpInterface
MemPlan::GetBufferParentLoop(const SmallVector<Value> &buffers) {
  llvm::SmallSet<LoopLikeOpInterface, 1> parentLoopVec;
  for (auto buffer : buffers) {
    if (!buffer.getDefiningOp()) {
      if (!isa<scf::ForOp, scf::WhileOp>(
              buffer.getParentBlock()->getParentOp())) {
        llvm::report_fatal_error("expected loop-carried block argument");
      }
      // Init args and region iter arg are inplace, ignore Region Iter Arg
      // without DefineOp.
      continue;
    }
    LoopLikeOpInterface bufferParentLoop = getParentLoop(buffer);
    if (bufferParentLoop) {
      parentLoopVec.insert(bufferParentLoop);
    } else {
      return nullptr;
    }
  }
  if (parentLoopVec.size() == 1) {
    return *parentLoopVec.begin();
  }
  return nullptr;
}

bool MemPlan::VerifyConflictStage1(MemBoundList &outline, PlanRecHis &his,
                                   StorageEntry *e,
                                   const OutlineSectionInfo &outlineInfo,
                                   uint64_t &pongOffset) {
  if (outlineInfo.mem_start != outlineInfo.mem_end) {
    return true;
  }
  auto reuseBoundStorageEntry = (*outlineInfo.mem_start)->lastStorageEntry;
  if (!reuseBoundStorageEntry) {
    // This area has not been planed, so there is no need to consider it.
    return true;
  }

  StorageEntry *multiRelationPongEntry =
      GetMultiRelationPongEntry(reuseBoundStorageEntry);
  if (multiRelationPongEntry) {
    if (e->multiBufferNum == kSingleBufferCount ||
        (e->multiBufferNum == kDoubleBufferCount && e->relationPongEntry &&
         (e->relationPongEntry->bitsOffset != 0))) {
      auto parentLoop1 = GetBufferParentLoop(e->inplaceBuffers);
      auto parentLoop2 =
          GetBufferParentLoop(reuseBoundStorageEntry->inplaceBuffers);
      if (!(parentLoop1 != nullptr && parentLoop2 != nullptr &&
            parentLoop1 == parentLoop2)) {
        // Cannot be reused under the same for.
        return true;
      }
      // There are two situations:
      // 1. Single reuse DB.
      // 2. DB reuse DB.
      pongOffset = multiRelationPongEntry->bitsOffset;
      bool conflict = std::any_of(
          his.begin(), his.end(), [pongOffset, e, this](PlanRecord &r) {
            return this->IsBufferLifeVecConflict(r, pongOffset, e);
          });
      if (!conflict) {
        return false;
      }
    }
  }
  return true;
}

StorageEntry *
MemPlan::GetMultiRelationPongEntry(const StorageEntry *reuseBoundStorageEntry) {
  if (reuseBoundStorageEntry->multiBufferNum == kDoubleBufferCount &&
      reuseBoundStorageEntry->relationPongEntry &&
      (reuseBoundStorageEntry->relationPongEntry->bitsOffset != 0)) {
    // If the reuseBoundStorageEntry itself requires db, directly match and
    // return relationPongEntry.
    return reuseBoundStorageEntry->relationPongEntry;
  }
  auto iter = pingEntry2RelationPongEntry.find(reuseBoundStorageEntry);
  if (iter != pingEntry2RelationPongEntry.end()) {
    // If the reuseBoundStorageEntry itself is single, but has already been
    // reused with db and has an extra pong StorageEntry is added.
    return iter->second.get();
  }
  return nullptr;
}

void MemPlan::SpecAllocRelationPongEntry(MemBoundList &outline, PlanRecHis &his,
                                         StorageEntry *e, uint64_t offset) {
  for (MemBoundListConstIter start = outline.begin(); start != outline.end();
       ++start) {
    uint64_t size = 0;
    // Find the MemBound corresponding to the Pong offset.
    if ((*start)->offset != offset) {
      continue;
    }
    for (MemBoundListConstIter end = start; end != outline.end(); ++end) {
      std::shared_ptr<MemoryBound> last = *end;
      size += last->extent;
      if (size < e->alignedConstBits) {
        continue;
      }
      StorageEntry *pongStorageEntry = nullptr;
      auto iter = pingEntry2RelationPongEntry.find(e);
      if (iter != pingEntry2RelationPongEntry.end()) {
        pongStorageEntry = iter->second.get();
      }
      if (e->multiBufferNum == kDoubleBufferCount && e->relationPongEntry) {
        pongStorageEntry = e->relationPongEntry;
      }
      if (!pongStorageEntry) {
        llvm::report_fatal_error("pong storage entry not found");
      }
      UpdateOutline(outline, his, pongStorageEntry,
                    OutlineSectionInfo(start, end, size, true), SPEC_LEVEL_1);
      return;
    }
  }
}

bool MemPlan::IsBufferLifeVecConflict(PlanRecord &r, uint64_t offset,
                                      const StorageEntry *e) const {
  if ((r.firstMemBound->offset + r.allExtent > offset) &&
      (r.firstMemBound->offset < offset + e->alignedConstBits)) {
    if (HasSemanticConflict(e, r.firstMemBound->bufferLifeVec)) {
      return true;
    }
    DenseMap<ValuePair, BufferLife> intersection =
        GetOverlapBufferLife(r.entry->bufferLifeVec, e->bufferLifeVec);
    return !intersection.empty();
  }
  return false;
}

void MemPlan::PlanRelationPongEntryAddress(uint64_t offset, StorageEntry *e) {
  if (e->multiBufferNum == kSingleBufferCount) {
    std::unique_ptr<StorageEntry> entry = std::make_unique<StorageEntry>();
    entry->bufInfo = e->bufInfo;
    entry->bufferLifeVec = e->bufferLifeVec;
    entry->alignedConstBits = e->alignedConstBits;
    entry->inplaceBuffers = e->inplaceBuffers;
    entry->multiBufferNum = e->multiBufferNum;
    entry->bitsOffset = offset;
    pingEntry2RelationPongEntry[e] = std::move(entry);
  } else if (e->multiBufferNum == kDoubleBufferCount) {
    e->relationPongEntry->bitsOffset = offset;
  }
  // N > 2: the Stage1 "place ping next to a free pong slot" optimization is
  // not modeled for the general N-way case in this release. Sibling entries
  // get their own addresses via the normal Stage0/Stage2 paths in
  // `PlanReusableLocalBuffer` / `PlanSingleLocalBuffer`. This branch is a
  // no-op rather than an unreachable so the planner can keep making forward
  // progress on N > 2 inputs.
}

bool MemPlan::VerifyConflictStage2(PlanRecHis &his, const StorageEntry *e,
                                   int specLevel, MemBoundListConstIter &start,
                                   const MemBoundList &outline) {
  if (specLevel != SPEC_LEVEL_2) {
    return false;
  }
  bool touchMemCanUse = false;
  MemBoundListConstIter foundMem;

  for (auto iter = start; iter != outline.end(); ++iter) {
    uint64_t offset = (*iter)->offset;
    bool conflict =
        std::any_of(his.begin(), his.end(), [offset, e, this](PlanRecord &r) {
          return (r.firstMemBound->offset + r.allExtent > offset) &&
                 (r.firstMemBound->offset < offset + e->alignedConstBits) &&
                 this->PipeConflict(r.entry, e, this->pipeDmaConflictMap);
        });
    // if conflict, continue finding the first bound that has no conflict
    // if last bound do not meet the size, continue
    if (conflict ||
        ((*iter == outline.back()) && (*iter)->extent < e->alignedConstBits)) {
      continue;
    }
    touchMemCanUse = true;
    foundMem = iter;
    break;
  }

  if (touchMemCanUse) {
    bool conflict = (foundMem != start);
    start = conflict ? --foundMem : start;
    return conflict;
  }
  // if cannot find a bound that has no conflict with current entry,
  return true;
}

bool MemPlan::PipeConflict(const StorageEntry *e1, const StorageEntry *e2,
                           DenseMap<StorageEntryPair, bool> &conflictMap) {
  if (e1 == nullptr || e2 == nullptr) {
    return false;
  }
  auto sePair = std::make_pair(e1, e2);
  auto [iter, isInserted] = conflictMap.try_emplace(sePair, false);
  if (!isInserted) {
    return iter->second;
  }

  for (const Value var1 : e1->inplaceBuffers) {
    for (const Value var2 : e2->inplaceBuffers) {
      bool conflict = dmaFirstPipelineOpt.BufferPipeConflict(var1, var2);
      if (conflict) {
        iter->second = true;
        return true;
      }
    }
  }
  return false;
}

void MemPlan::UpdateOutline(MemBoundList &outline, PlanRecHis &his,
                            StorageEntry *e,
                            const OutlineSectionInfo &outlineInfo,
                            int localLevel) const {
  auto start = outlineInfo.mem_start;
  MemBoundListConstIter end = outlineInfo.mem_end;
  // outline:
  // |-------start+end-------------|
  // |--head--|--split e--|--tail--|
  uint64_t res = outlineInfo.size - e->alignedConstBits;
  std::shared_ptr<MemoryBound> last = *end;
  ++end;
  std::shared_ptr<MemoryBound> bound;
  SmallVector<std::shared_ptr<MemoryBound>> splitBound;
  // split e, to get Boundbound
  if (splitOutline) {
    // add splitBound by splitting e to section
    AddMemBoundInSectionalWay(e, start, end, splitBound);
  } else {
    // origin outline
    BufferLifeVec life(e->bufferLifeVec.begin(), e->bufferLifeVec.end());
    MergeBufferLife(start, end, life);
    splitBound.emplace_back(std::make_shared<MemoryBound>(
        life, e->bitsOffset, e->alignedConstBits, e));
  }

  // insert tail memory bound
  if (res > 0) {
    bound = std::make_shared<MemoryBound>(last->bufferLifeVec,
                                          last->offset + last->extent - res,
                                          res, last->lastStorageEntry);
    end = outline.insert(end, bound);
  }
  // insert split e memory bound
  for (int i = static_cast<int>(splitBound.size()) - 1; i >= 0; --i) {
    end = outline.insert(end, splitBound[i]);
  }
  // record the current plan of first split entry in his
  his.emplace_back(PlanRecord{localLevel,
                              e->childIdx,
                              res > 0,
                              false,
                              splitBound.size(),
                              e,
                              e->alignedConstBits,
                              splitBound[0],
                              {},
                              outlineInfo.isDirectlyRollback});
  PlanRecord &r = his.back();
  r.replaced.splice(r.replaced.begin(), outline, start, end);
}

void MemPlan::AddMemBoundInSectionalWay(
    StorageEntry *e, MemBoundListConstIter start, MemBoundListConstIter end,
    SmallVector<std::shared_ptr<MemoryBound>> &splitBound) const {
  // |--outline1--|--outline2--|--outline3--|
  // |---------e------------ |
  // |--split e1 -|-split e2-|
  for (auto iter = start; iter != end; ++iter) {
    BufferLifeVec life(e->bufferLifeVec.begin(), e->bufferLifeVec.end());
    life.insert(life.end(), (*iter)->bufferLifeVec.begin(),
                (*iter)->bufferLifeVec.end());
    // merge the buffer life
    MergeBufferVec(life);
    // get the extent
    uint64_t size = 0;
    if (std::distance(start, iter) == std::distance(start, end) - 1) {
      // deal with the last split e2
      size = e->bitsOffset + e->alignedConstBits - (*iter)->offset;
    } else {
      size = (*iter)->extent;
    }
    splitBound.emplace_back(
        std::make_shared<MemoryBound>(life, (*iter)->offset, size, e));
  }
}

inline void MemPlan::MergeBufferLife(MemBoundList::const_iterator start,
                                     MemBoundList::const_iterator end,
                                     BufferLifeVec &newLife) const {
  size_t size = 0;
  for (auto it = start; it != end; ++it) {
    size += (*it)->bufferLifeVec.size();
  }
  newLife.reserve(size);
  for (auto it = start; it != end; ++it) {
    newLife.insert(newLife.end(), (*it)->bufferLifeVec.begin(),
                   (*it)->bufferLifeVec.end());
  }
  MergeBufferVec(newLife);
}

void MemPlan::MergeBufferVec(BufferLifeVec &bufferLife) const {
  if (bufferLife.empty()) {
    return;
  }
  BufferLifeVec mergedLife;
  mergedLife.reserve(bufferLife.size());
  // sort life by alloc and free time
  std::sort(bufferLife.begin(), bufferLife.end(), CompareBufferLife());
  int start = bufferLife[0]->allocTime;
  int end = bufferLife[0]->freeTime;
  auto buffer = bufferLife[0]->buffer;
  // merge life
  for (size_t i = 1; i < bufferLife.size(); i++) {
    auto &life = bufferLife[i];
    if (life->allocTime <= end + 1) {
      end = end < life->freeTime ? life->freeTime : end;
    } else {
      mergedLife.emplace_back(std::make_unique<BufferLife>(buffer, start, end));
      buffer = life->buffer;
      start = life->allocTime;
      end = life->freeTime;
    }
  }
  mergedLife.emplace_back(std::make_unique<BufferLife>(buffer, start, end));
  bufferLife.swap(mergedLife);
}

bool MemPlan::IsSamePlanAsLastRollBack(uint64_t allocOffset, int curChildIdx,
                                       const SpecInfo &si) const {
  return curChildIdx == si.rollbackIdx && allocOffset == si.rollbackAddr;
}

// spec_level == SPEC_LEVEL_0
inline bool
MemPlan::VerifyConflictStage0(StorageEntry *e,
                              const std::shared_ptr<MemoryBound> &last) {
  if (HasSemanticConflict(e, last->bufferLifeVec)) {
    return true;
  }
  // level_0: offset = 0, offset means life distance
  DenseMap<ValuePair, BufferLife> intersection =
      GetOverlapBufferLife(e->bufferLifeVec, last->bufferLifeVec);
  return !intersection.empty();
}

// verify two buffer life vectors is conflict or not
// The key pair looks like the following diagram
// indicate that var1 is generated later than var2.
//                                     buffer2
//                               --- PLAN_time
//   buffer1       intersected   | |
//                 buffer_life   | |
// PLAN_time ---    lo ---      ---
//            | |       ---      | |
//            | |       ---      | |
//            ---    hi ---      --- free_time
//            | |
//            | |
// free_time  ---
// Meantime, the overlap is the intersected buffer_life.
DenseMap<ValuePair, BufferLife>
MemPlan::GetOverlapBufferLife(const BufferLifeVec &b1,
                              const BufferLifeVec &b2) const {
  DenseMap<ValuePair, BufferLife> intersection;
  size_t i = 0;
  size_t j = 0;
  size_t b1Len = b1.size();
  size_t b2Len = b2.size();
  if (b1Len == 0 || b2Len == 0) {
    return intersection;
  }
  while (i < b1Len && j < b2Len) {
    auto lo = std::max(b1[i]->allocTime, b2[j]->allocTime);
    auto hi = std::min(b1[i]->freeTime, b2[j]->freeTime);
    if (lo <= hi) {
      BufferLife bufferLife(nullptr, lo, hi);
      ValuePair key =
          lo == b1[i]->allocTime && hi == b2[j]->freeTime
              ? std::make_pair(b1[i]->buffer,
                               b2[j]->buffer) // case in the diagram
              : std::make_pair(b2[j]->buffer, b1[i]->buffer); // opposing case
      intersection.try_emplace(key, bufferLife);
    }
    if (b1[i]->freeTime < b2[j]->freeTime) {
      i += 1;
    } else {
      j += 1;
    }
  }
  return intersection;
}

PlanStatus MemPlan::ApplyFailStrategy(StatusWrapper &statusWrapper,
                                      const size_t maxBits) {
  RollBackForAllocFail(statusWrapper, maxBits);
  // second class rollback, level 1 --> 0
  if (statusWrapper.si->specLevel > SPEC_LEVEL_0 &&
      statusWrapper.si->childIdx >= 0) {
    statusWrapper.si->specLevel--;
    return PlanStatus::CONTINUE_PLAN;
  }
  if (!splitOutline) {
    // roll back to origin again, enable split outline.
    splitOutline = true;
    return PlanStatus::RESTART_NEW_PLAN;
  }
  return PlanStatus::PLAN_FAILED;
}

void MemPlan::ReportAllocatedEntryDebugInfo(StorageEntry *rootStorageEntry) {
  auto printRecord = [this](const StorageEntry *entry) {
    uint64_t needByte =
        (entry->alignedConstBits + kBitsToByte - 1) / kBitsToByte;
    uint64_t offsetByte =
        (entry->bitsOffset + kBitsToByte - 1) / kBitsToByte;
    (void)needByte;
    (void)offsetByte;
    ReportCurEntryDebugInfo(entry);
    LDBG(", offset: " << offsetByte);
    LDBG(", extent: " << needByte);
    LDBG(", buffer life: ");
    for (auto &bufferLife : entry->bufferLifeVec) {
      (void)bufferLife;
      LDBG("[" << bufferLife->allocTime << "-" << bufferLife->freeTime
               << "], ");
    }
  };
  LDBG("--------------------------BUFFER ALLOCATE "
       "START-------------------------- "
       << "\n"
       << "\n");
  LDBG("  BUFFER ALLOCATE START: UB"
       << "\n");
  if (!allocatedEntry.empty()) {
    for (auto &entry : allocatedEntry) {
      printRecord(entry);
      LDBG("\n");
    }
    size_t num = allocatedEntry.size() - 1;
    if (rootStorageEntry->mergedChildren.size() <= num) {
      llvm::report_fatal_error("missing failed storage entry");
    }
    const StorageEntry *failedSe = rootStorageEntry->mergedChildren[num];
    printRecord(failedSe);
    LDBG("alloc fail,because exceed bound of memory \n"
         << "  BUFFER ALLOCATE END \n");
    LDBG("\n"
         << "--------------------------BUFFER ALLOCATE "
            "END-------------------------- "
         << "\n");
  }
}

LogicalResult MemPlan::InitMemSpecsFromModule(func::FuncOp funcOp) {
  struct MemSpec {
    int ubSpaceSize;
    int l1SpaceSize;
    int l0aSpaceSize;
    int l0bSpaceSize;
    int l0cSpaceSize;
    int ubAlignSize;
    int l1AlignSize;
    int l0cAlignSize;
    int l0aAlignSize;
    int l0bAlignSize;
    int biasAlignSize;
    int biasSpaceSize;
    int scalingAlignSize;
    int scalingSpaceSize;
  };

  const MemSpec kA3 = {
      1572864, 4194304, 524288, 524288, 1048576, 256, 256,
      4096,    4096,    4096,   256,    524288, 256, 1572864};
  const MemSpec kA5 = {
      2031616, 4194304, 524288, 524288, 2097152, 256, 256,
      4096,    4096,    4096,   256,    524288, 256, 2031616};

  auto applySpec = [this](const MemSpec &spec) {
    ubSpaceSize = spec.ubSpaceSize;
    l1SpaceSize = spec.l1SpaceSize;
    l0aSpaceSize = spec.l0aSpaceSize;
    l0bSpaceSize = spec.l0bSpaceSize;
    l0cSpaceSize = spec.l0cSpaceSize;
    ubAlignSize = spec.ubAlignSize;
    l1AlignSize = spec.l1AlignSize;
    l0cAlignSize = spec.l0cAlignSize;
    l0aAlignSize = spec.l0aAlignSize;
    l0bAlignSize = spec.l0bAlignSize;
    biasAlignSize = spec.biasAlignSize;
    biasSpaceSize = spec.biasSpaceSize;
    scalingAlignSize = spec.scalingAlignSize;
    scalingSpaceSize = spec.scalingSpaceSize;
  };

  // Default to a3.
  applySpec(kA3);

  // --pto-arch options:
  // a3 -> default memory spec
  // a5 -> override memory spec
  if (isTargetArchA5(getTopLevelModuleOp(funcOp))) {
    applySpec(kA5);
  }
  return success();
}

void MemPlan::RollBackForAllocFail(StatusWrapper &statusWrapper,
                                   const size_t maxBits) {
  while (ContinueRollBack(statusWrapper)) {
    RollBackForAllocFailInner(statusWrapper, maxBits);
  }
}

bool MemPlan::ContinueRollBack(const StatusWrapper &statusWrapper) const {
  return (!statusWrapper.hasEnoughRollBackSize) &&
         (!statusWrapper.history.empty() && (!statusWrapper.outline.empty()));
}

void MemPlan::RollBackForAllocFailInner(StatusWrapper &statusWrapper,
                                        const size_t maxBits) {
  auto &si = statusWrapper.si;
  if (si->childIdx > si->specStartIdx) {
    si->specStartIdx = si->childIdx;
  }
  // Check whether the container is empty before accessing "history"
  while (!statusWrapper.history.empty()) {
    PlanRecord r =
        RollbackOutline(statusWrapper.history, statusWrapper.outline);
    auto iter = pingEntry2RelationPongEntry.find(r.entry);
    if (iter != pingEntry2RelationPongEntry.end()) {
      pingEntry2RelationPongEntry.erase(iter);
    }
    if (r.isDirectlyRollback ||
        (r.entry->multiBufferNum == kDoubleBufferCount &&
         !r.entry->relationPongEntry)) {
      continue;
    }
    si->childIdx = r.childIdx;
    si->specLevel = r.specLevel;
    if (si->specLevel > si->minLevel) {
      // record rollback info: index and address
      si->rollbackAddr =
          si->childIdx == -1
              ? UINT64_MAX
              : statusWrapper.RootE->mergedChildren[si->childIdx]->bitsOffset;
      si->rollbackIdx = si->childIdx;
      if (statusWrapper.si->rollbackAddr + statusWrapper.alignedConstBits >
          maxBits) {
        continue;
      }
      statusWrapper.hasEnoughRollBackSize = true;
      break;
    }
  }
}

PlanRecord MemPlan::RollbackOutline(PlanRecHis &history,
                                    MemBoundList &outline) const {
  auto r = history.back();
  auto it = std::find(outline.begin(), outline.end(), r.firstMemBound);
  // |--head--|--split entry--|--tail--|
  // erase head
  if (r.headed) {
    it--;
    it = outline.erase(it);
  }
  // erase split entry
  for (size_t i = 0; i < r.splitNums; i++) {
    it = outline.erase(it);
  }
  // erase tail
  if (r.tailed) {
    it = outline.erase(it);
  }
  // restore outline and replaced
  outline.splice(it, r.replaced);
  history.pop_back();
  return r;
}

namespace {

class LegacyAllocTileOpAddPlannedAddressPattern
    : public OpRewritePattern<pto::AllocTileOp> {
public:
  explicit LegacyAllocTileOpAddPlannedAddressPattern(
      MLIRContext *context,
      DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets)
      : OpRewritePattern<pto::AllocTileOp>(context),
        buffer2Offsets(std::move(buffer2Offsets)) {}

  LogicalResult matchAndRewrite(pto::AllocTileOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getAddr()) {
      return failure();
    }

    auto tileType = dyn_cast<TileBufType>(op.getResult().getType());
    if (!tileType) {
      return failure();
    }

    auto it = buffer2Offsets.find(op.getResult());
    if (it == buffer2Offsets.end() || it->second.empty()) {
      return failure();
    }

    if (it->second.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "single alloc_tile root expects exactly one planned address");
    }

    Value addr = rewriter.create<arith::ConstantIntOp>(op.getLoc(),
                                                       it->second.front(), 64);
    auto planned = rewriter.create<pto::AllocTileOp>(
        op.getLoc(), tileType, addr,
        op.getValidRow() ? op.getValidRow() : Value(),
        op.getValidCol() ? op.getValidCol() : Value());
    for (NamedAttribute attr : op->getAttrs()) {
      if (attr.getName().getValue() == "operandSegmentSizes") {
        continue;
      }
      planned->setAttr(attr.getName(), attr.getValue());
    }

    rewriter.replaceOp(op, planned.getResult());
    return success();
  }

private:
  DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets;
};

class LegacyAllocMultiTileOpAddPlannedAddressesPattern
    : public OpRewritePattern<pto::AllocMultiTileOp> {
public:
  explicit LegacyAllocMultiTileOpAddPlannedAddressesPattern(
      MLIRContext *context,
      DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets)
      : OpRewritePattern<pto::AllocMultiTileOp>(context),
        buffer2Offsets(std::move(buffer2Offsets)) {}

  LogicalResult matchAndRewrite(pto::AllocMultiTileOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getAddr() || op->hasAttr(pto::kPtoMultiBufferAddrsAttrName)) {
      return failure();
    }
    auto it = buffer2Offsets.find(op.getResult());
    if (it == buffer2Offsets.end() || it->second.empty()) {
      return failure();
    }
    if (it->second.size() != op.getResult().getType().getCount()) {
      return rewriter.notifyMatchFailure(
          op, "planned address count does not match multi_tile_buf count");
    }

    SmallVector<int64_t> addrs;
    addrs.reserve(it->second.size());
    for (uint64_t offset : it->second) {
      addrs.push_back(static_cast<int64_t>(offset));
    }
    rewriter.modifyOpInPlace(op, [&] {
      op->setAttr(pto::kPtoMultiBufferAddrsAttrName,
                  rewriter.getDenseI64ArrayAttr(addrs));
    });
    return success();
  }

private:
  DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets;
};

static FailureOr<MemPlanMode> parseLegacyMemPlanMode(func::FuncOp func,
                                                     llvm::StringRef memMode) {
  if (memMode.equals_insensitive("local") ||
      memMode.equals_insensitive("local-mem-plan")) {
    return MemPlanMode::LOCAL_MEM_PLAN;
  }
  if (memMode.equals_insensitive("global-work-space-plan")) {
    return MemPlanMode::GLOBAL_WORKSPACE_PLAN;
  }
  func.emitError("unsupported mem-mode '")
      << memMode << "'; only 'local' is supported by the PTOAS pipeline";
  return failure();
}

struct PlanMemoryPass : public mlir::pto::impl::PlanMemoryBase<PlanMemoryPass> {
public:
  PlanMemoryPass() = default;
  explicit PlanMemoryPass(const mlir::pto::PlanMemoryOptions &planMemoryOption)
      : PlanMemoryBase(planMemoryOption) {}

  void runOnOperation() override;

private:
  void populateBufferAddressToAllocOp(
      RewritePatternSet &patterns, MemPlanMode mode,
      DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets) {
    if (mode == MemPlanMode::LOCAL_MEM_PLAN) {
      patterns.add<LegacyAllocTileOpAddPlannedAddressPattern>(
          patterns.getContext(), buffer2Offsets);
      patterns.add<LegacyAllocMultiTileOpAddPlannedAddressesPattern>(
          patterns.getContext(), buffer2Offsets);
    }
  }
};
} // namespace

void PlanMemoryPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  SmallVector<func::FuncOp> funcs;
  moduleOp.walk([&](func::FuncOp funcOp) {
    // TileOp helpers only contain compute code and deliberately do not own
    // alloc_tile/reserve_buffer lifetimes.  All other functions, including
    // ordinary functions in backend child modules, must be planned.
    if (!funcOp->hasAttr("pto.tileop.helper")) {
      funcs.push_back(funcOp);
    }
  });

  for (func::FuncOp funcOp : funcs) {
    auto parsedMode = parseLegacyMemPlanMode(funcOp, this->memMode);
    if (failed(parsedMode)) {
      return signalPassFailure();
    }
    MemPlanMode mode = *parsedMode;
    ReserveBufferPlans reservePlans;
    if (mode == MemPlanMode::LOCAL_MEM_PLAN &&
        failed(analyzeReserveBufferPlans(funcOp, reservePlans))) {
      return signalPassFailure();
    }
    if (mode == MemPlanMode::LOCAL_MEM_PLAN) {
      for (ReserveBufferPlan &reservePlan : reservePlans) {
        if (reservePlan.mode != ReserveBufferMode::Manual) {
          continue;
        }
        reservePlan.reserveOp.emitOpError(
            "pto.reserve_buffer with explicit 'base' (auto = false) is not "
            "supported in PlanMemory; use --pto-level=level3 or set auto = true");
        return signalPassFailure();
      }
    }

    MemLivenessAnalysis memLiveness(funcOp, mode);
    memLiveness.build();

    constexpr bool enableGlobalReuse = false;
    constexpr bool enablePrintMemoryAllocatedSize = false;
    constexpr bool restrictInplaceAsISA = false;
    MemPlan memPlan(mode, enableGlobalReuse, enablePrintMemoryAllocatedSize,
                    restrictInplaceAsISA, this->orderBySize);
    if (failed(memPlan.InitMemSpecsFromModule(funcOp))) {
      return signalPassFailure();
    }
    memPlan.func_ = funcOp;
    memPlan.SetLinearOperation(memLiveness.linearOperation);
    memPlan.SetBufferInfos(memLiveness.bufferInfos);
    memPlan.SetBuffer2Life(memLiveness.buffer2Life);
    memPlan.SetGenKillMap(memLiveness.genKillMap);
    memPlan.SetBuffer2MultiNum(memLiveness.buffer2MultiNum);
    memPlan.SetInplacePairList(memLiveness.inplacePairList);
    memPlan.SetSemanticConflictPairs(memLiveness.semanticConflictPairs);
    memPlan.SetStableValueOrder(std::move(memLiveness.stableValueOrder));
    if (failed(memPlan.plan())) {
      return signalPassFailure();
    }
    // Keep reserve_buffer allocation outside the core MemPlan algorithm:
    // normal local buffers are planned first, then reserve_buffer claims one
    // aligned hole in its target address space.
    if (mode == MemPlanMode::LOCAL_MEM_PLAN &&
        failed(assignAutoReserveBufferBases(
            reservePlans, memLiveness.bufferInfos, memPlan.GetBuffer2Offsets()))) {
      return signalPassFailure();
    }

    RewritePatternSet patterns(&getContext());
    populateBufferAddressToAllocOp(patterns, mode, memPlan.GetBuffer2Offsets());
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
    if (failed(verifySemanticNoAliasRanges(funcOp)))
      return signalPassFailure();

    bool hasUnplannedAllocTile = false;
    funcOp.walk([&](pto::AllocTileOp op) {
      if (op.getAddr()) {
        return;
      }
      if (op->use_empty()) {
        return;
      }
      if (isA5IgnoredTmpAlloc(op)) {
        return;
      }
      op.emitError(
          "PTOPlanMemory failed to assign an address to pto.alloc_tile");
      hasUnplannedAllocTile = true;
    });
    if (hasUnplannedAllocTile) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<Pass>
mlir::pto::createPlanMemoryPass(const PlanMemoryOptions &options) {
  return std::make_unique<PlanMemoryPass>(options);
}
