// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMILowerUnifiedToLegacy.cpp - Lower unified v-ops to legacy ops ----===//
//
// Lowers unified v-prefixed VMI ops to their legacy equivalents under the
// opt-in --vmi-two-stage-lowering flag.
//
// Category A — pure syntactic renames (4 ops):
//   vci             → iota
//   vinterpret_cast → bitcast
//   vsel            → select
//   vbrc            → broadcast  (skipped when num_groups is present)
//
// Category B — elementwise arithmetic / bitwise:
//   vreg operations remain unified and lower directly to masked VPTO ops.
//   Predicate-only vand/vor/vxor/vnot still use the legacy mask operations.
//
// Category C1 — compare + seed (2 ops):
//   vcmp  → cmpf/cmpi + mask_and
//   vcmps → broadcast scalar + cmpf/cmpi + mask_and
//
// Category C2 — unified type conversion (1 op):
//   vcvt → type-dispatch to extf/truncf/fptosi/sitofp/extsi/extui/trunci
//   For fp narrowing, unified saturate=SAT is normalized away because the
//   legacy truncf -> VPTO lowering already materializes saturating low-level
//   vcvt forms for supported narrowing result families.
//
// Category C3 — unified load/store (2 ops):
//   vload  → load / group_load / group_slot_load / stride_load
//   vstore → store / masked_store / group_store / stride_store
//   Contiguous 1/2/4/8-lane values alias unit-stride group-slot operations.
//
// Category C4 — static mask creation (3 ops):
//   pset → create_mask(all lanes)
//   pge  → create_mask(N lanes)
//   plt  → create_mask(min(rem, L))
//
// Category C6 — unified reduce (3 ops):
//   vcadd → reduce_addf/reduce_addi or group_reduce_addf/group_reduce_addi
//   vcmax → reduce_maxf/reduce_maxi or group_reduce_maxf/group_reduce_maxi
//   vcmin → reduce_minf/reduce_mini or group_reduce_minf/group_reduce_mini
//
// Category C8 — indexed gather/scatter → legacy gather/scatter (2 ops):
//   vgather  → gather   (pmode="zero": passthru = zero constant)
//   vscatter → scatter
//
// Category C9 — masked and fused activation / softmax operations remain
// unified for direct VMI-to-VPTO lowering.
//
// Category D — no legacy equivalent (explicitly skipped, 13 ops):
//   vadds/vmuls/vmaxs/vmins/vshls/vshrs
//   vaddc vaddcs standalone vintlv/vdintlv vselr vgatherb vmull
//   Adjacent contiguous load/rearrange and rearrange/store compositions are
//   first combined into the corresponding legacy memory operation, and
//   small-shape f16 vexpdif (below one f16 physical part) is widened to the
//   f32 form with two extf ops (see expandSmallShapeF16Vexpdif) before the
//   worklist runs.
//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Transforms/Passes.h"
#include "VMI/VMIIndexUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <limits>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMILOWERUNIFIEDTOLEGACY
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {
constexpr unsigned kIndexBitWidth = 64;
constexpr int64_t kSingleGroupCount = 1;
constexpr int64_t kDecimalRadix = 10;
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Create a zero-valued VMIConstantOp with the same type as \p vmiType.
static Value createZeroConstant(OpBuilder &builder, Location loc,
                                VMIVRegType vmiType) {
  Type elemType = vmiType.getElementType();
  int64_t laneCount = vmiType.getElementCount();
  auto shapedType = RankedTensorType::get({laneCount}, elemType);

  DenseElementsAttr zeroAttr;
  if (auto floatType = dyn_cast<FloatType>(elemType)) {
    zeroAttr = DenseElementsAttr::get(
        shapedType, APFloat::getZero(floatType.getFloatSemantics()));
  } else if (auto intType = dyn_cast<IntegerType>(elemType)) {
    zeroAttr = DenseElementsAttr::get(
        shapedType, APInt::getZero(intType.getWidth()));
  } else {
    llvm_unreachable("unsupported VMI element type for zero constant");
  }
  return builder.create<VMIConstantOp>(loc, vmiType, zeroAttr).getResult();
}


/// Map a unified vcmp `cmp` mode to the predicate string for legacy
/// cmpf/cmpi. Float operands use ordered predicates (olt, oeq, ...);
/// integer operands select signedness from the element type.
static std::string mapCmpPredicate(StringRef cmp, Type elemType,
                                   bool isFloat) {
  if (isFloat) {
    // Already ordered/unordered — pass through.
    if (cmp.starts_with("o") || cmp.starts_with("u")) {
      return cmp.str();
    }
    return ("o" + cmp).str(); // e.g. "lt" → "olt"
  }
  // Integer.
  if (cmp.starts_with("s") || cmp.starts_with("u")) {
    return cmp.str();
  }
  // eq/ne are valid for both fp and int without prefix.
  if (cmp == "eq" || cmp == "ne") {
    return cmp.str();
  }
  auto intType = dyn_cast<IntegerType>(elemType);
  if (intType && !intType.isSigned()) {
    return ("u" + cmp).str(); // e.g. "lt" -> "ult"
  }
  return ("s" + cmp).str();   // e.g. "lt" -> "slt"
}

/// Return true when \p elemType is a floating-point type.
/// Return true for MLIR FloatType and PTO low-precision float-like types
/// (hif8, f8, f4, etc.).
static bool isFloatType(Type elemType) {
  return isa<FloatType>(elemType) || pto::isPTOLowPrecisionType(elemType);
}

/// Return the element type of a VMIVRegType.
static Type getVMIElementType(Value v) {
  return cast<VMIVRegType>(v.getType()).getElementType();
}

/// Return the storage bit width for VMI element types (float / float-like / int).
static unsigned getVMIElementBitWidth(Type type) {
  if (isa<IndexType>(type)) {
    return kIndexBitWidth;
  }
  return pto::getPTOStorageElemBitWidth(type);
}

/// Inspect the source and result element types of a vcvt and classify the
/// conversion direction.  Returns one of:
///   "widen_fp", "narrow_fp", "fptosi", "fptoui",
///   "sitofp", "widen_int", "narrow_int"
static StringRef classifyCvtDirection(Type srcElem, Type dstElem) {
  bool srcFp = isFloatType(srcElem);
  bool dstFp = isFloatType(dstElem);
  unsigned srcBits = getVMIElementBitWidth(srcElem);
  unsigned dstBits = getVMIElementBitWidth(dstElem);

  if (srcFp && dstFp) {
    return dstBits > srcBits ? "widen_fp" : "narrow_fp";
  }
  if (srcFp && !dstFp) {
    if (auto intTy = dyn_cast<IntegerType>(dstElem))
      return intTy.isSigned() ? "fptosi" : "fptoui";
    return "fptosi";
  }
  if (!srcFp && dstFp) {
    auto intTy = dyn_cast<IntegerType>(srcElem);
    if (!intTy || !intTy.isSigned()) {
      return "unsupported";
    }
    return "sitofp";
  }
  // int → int
  return dstBits > srcBits ? "widen_int" : "narrow_int";
}

//===----------------------------------------------------------------------===//
// Category C1 helpers: vcmp / vcmps
//===----------------------------------------------------------------------===//

/// Returns true if `seed` is provably an all-active mask (every lane active),
/// so `mask_and(x, seed)` is the identity and the AND can be skipped. Covers a
/// `pset` (all lanes active by definition) and a `create_mask` whose
/// active_lanes is a constant >= the mask lane count.
static bool isAllActiveSeed(Value seed) {
  Operation *def = seed.getDefiningOp();
  if (!def) {
    return false;
  }
  if (isa<VMIPsetOp>(def)) {
    return true;
  }
  if (auto cm = dyn_cast<VMICreateMaskOp>(def)) {
    auto maskTy = cast<VMIMaskType>(cm.getResult().getType());
    if (auto cst = cm.getActiveLanes().getDefiningOp<arith::ConstantOp>()) {
      if (auto ia = dyn_cast<IntegerAttr>(cst.getValue())) {
        return ia.getInt() >= maskTy.getElementCount();
      }
    }
  }
  return false;
}

static bool isCompactGroupCount(int64_t count) {
  return count == kSingleGroupCount || count == mlir::pto::kValue2 ||
         count == mlir::pto::kValue4 || count == mlir::pto::kValue8;
}

/// Lower vcmp to cmpf/cmpi + mask_and.
static LogicalResult lowerVCmp(VMIVcmpOp op, OpBuilder &builder) {
  Location loc = op.getLoc();
  Type elemType = getVMIElementType(op.getLhs());
  bool isFloat = isFloatType(elemType);
  StringRef cmpMode = op.getCmp();
  std::string predicate = mapCmpPredicate(cmpMode, elemType, isFloat);

  // Build legacy cmpf or cmpi.
  Value rawMask;
  if (isFloat) {
    rawMask = builder
                  .create<VMICmpFOp>(loc, op.getResult().getType(),
                                     builder.getStringAttr(predicate),
                                     op.getLhs(), op.getRhs())
                  .getResult();
  } else {
    rawMask = builder
                  .create<VMICmpIOp>(loc, op.getResult().getType(),
                                     builder.getStringAttr(predicate),
                                     op.getLhs(), op.getRhs())
                  .getResult();
  }

  // mask_and with seed — skipped when the seed is all-active (identity AND).
  Value result = rawMask;
  if (!isAllActiveSeed(op.getSeed())) {
    result = builder
                 .create<VMIMaskAndOp>(loc, op.getResult().getType(), rawMask,
                                       op.getSeed())
                 .getResult();
  }

  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

/// Lower vcmps to broadcast scalar + cmpf/cmpi + mask_and.
static LogicalResult lowerVCmps(VMIVcmpsOp op, OpBuilder &builder) {
  Location loc = op.getLoc();
  Type srcVmiType = op.getSrc().getType();
  Value scalar = op.getScalar();
  Type elemType = getVMIElementType(op.getSrc());
  bool isFloat = isFloatType(elemType);
  StringRef cmpMode = op.getCmp();
  std::string predicate = mapCmpPredicate(cmpMode, elemType, isFloat);

  // 1. Broadcast scalar to vector.
  Value brc = builder.create<VMIBroadcastOp>(loc, srcVmiType, scalar)
                  .getResult();

  // 2. Legacy cmpf or cmpi.
  Value rawMask;
  if (isFloat) {
    rawMask = builder
                  .create<VMICmpFOp>(loc, op.getResult().getType(),
                                     builder.getStringAttr(predicate),
                                     op.getSrc(), brc)
                  .getResult();
  } else {
    rawMask = builder
                  .create<VMICmpIOp>(loc, op.getResult().getType(),
                                     builder.getStringAttr(predicate),
                                     op.getSrc(), brc)
                  .getResult();
  }

  // 3. mask_and with seed — skipped when the seed is all-active (identity AND).
  Value result = rawMask;
  if (!isAllActiveSeed(op.getSeed())) {
    result = builder
                 .create<VMIMaskAndOp>(loc, op.getResult().getType(), rawMask,
                                       op.getSeed())
                 .getResult();
  }

  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Category C2 helper: vcvt
//===----------------------------------------------------------------------===//

/// Lower vcvt by dispatching on src→dst element types.
static LogicalResult lowerVCvt(VMICvtOp op, OpBuilder &builder) {
  Type srcElem = getVMIElementType(op.getSource());
  Type dstElem = getVMIElementType(op.getResult());
  StringRef direction = classifyCvtDirection(srcElem, dstElem);
  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  Value source = op.getSource();
  Value result;

  StringAttr saturateAttr = op.getSaturateAttr();

  if (direction == "widen_fp") {
    result = builder.create<VMIExtFOp>(loc, resultType, source).getResult();
  } else if (direction == "narrow_fp") {
    StringAttr roundingAttr = op.getRoundingAttr();
    result = builder
                 .create<VMITruncFOp>(loc, resultType, source, roundingAttr,
                                     saturateAttr)
            .getResult();
  } else if (direction == "fptosi") {
    result = builder
            .create<VMIFPToSIOp>(loc, resultType, source,
                                 op.getRoundingAttr(), saturateAttr)
            .getResult();
  } else if (direction == "fptoui") {
    result = builder
            .create<VMIFPToUIOp>(loc, resultType, source,
                                 op.getRoundingAttr(), saturateAttr)
            .getResult();
  } else if (direction == "sitofp") {
    result = builder.create<VMISIToFPOp>(loc, resultType, source).getResult();
  } else if (direction == "widen_int") {
    // Use source type signedness to decide signed vs unsigned extension.
    bool useSigned = true;
    if (auto intTy = dyn_cast<IntegerType>(srcElem)) {
      useSigned = intTy.isSigned();
    }
    if (useSigned) {
      result =
          builder.create<VMIExtSIOp>(loc, resultType, source).getResult();
    } else {
      result =
          builder.create<VMIExtUIOp>(loc, resultType, source).getResult();
}
  } else if (direction == "narrow_int") {
    result =
        builder.create<VMITruncIOp>(loc, resultType, source, saturateAttr)
            .getResult();
  } else {
    return failure();
  }

  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Category C3 helpers: vload / vstore
//===----------------------------------------------------------------------===//

static StringAttr getMaskGranularity(Type elementType, OpBuilder &builder) {
  // Surface phase: masks must use pred granularity so PTOValidateVMIIR
  // passes.  Concrete b8/b16/b32 granularity is assigned later by
  // VMIMaskGranularityAssignment.
  (void)elementType;
  return builder.getStringAttr("pred");
}

static Value createAllActiveMask(VMIVRegType valueType, Location loc,
                                 OpBuilder &builder) {
  auto maskType = VMIMaskType::get(
      builder.getContext(), valueType.getElementCount(),
      getMaskGranularity(valueType.getElementType(), builder),
      valueType.getLayout());
  Value activeLanes = builder.create<arith::ConstantOp>(
      loc, builder.getIndexAttr(valueType.getElementCount()));
  return builder.create<VMICreateMaskOp>(loc, maskType, activeLanes).getResult();
}

static LogicalResult lowerGroupedLoad(VMIvLoadOp op, OpBuilder &builder) {
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  int64_t numGroups = op.getGroupAttr().getInt();
  Value replacement;
  bool hasOneSlotPerGroup = resultType.getElementCount() == numGroups;
  if (hasOneSlotPerGroup) {
    replacement = builder
                      .create<VMIGroupSlotLoadOp>(
                          op.getLoc(), resultType, op.getSource(),
                          op.getOffset(), op.getStride(), op.getGroupAttr())
                      .getResult();
  } else {
    replacement = builder
                      .create<VMIGroupLoadOp>(
                          op.getLoc(), resultType, op.getSource(),
                          op.getOffset(), op.getStride(), op.getGroupAttr())
                      .getResult();
  }
  op.getResult().replaceAllUsesWith(replacement);
  return success();
}

static LogicalResult lowerBlockStrideLoad(VMIvLoadOp op,
                                          OpBuilder &builder) {
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  Value mask = createAllActiveMask(resultType, op.getLoc(), builder);
  Value replacement =
      builder
          .create<VMIStrideLoadOp>(op.getLoc(), resultType, op.getSource(),
                                   op.getOffset(), op.getBlockStride(), mask)
          .getResult();
  op.getResult().replaceAllUsesWith(replacement);
  return success();
}

// A continuous vload is a dense access whatever its length.  Short vectors are
// not group packets: how few elements a load touches is a property of the
// physical access VMIToVPTO plans for it, not of the logical lane layout, so
// the length must not change which legacy op -- and therefore which layout
// contract -- the value carries.
static LogicalResult lowerContinuousLoad(VMIvLoadOp op,
                                         OpBuilder &builder) {
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  Value replacement = builder
                          .create<VMILoadOp>(op.getLoc(), resultType,
                                             op.getSource(), op.getOffset())
                          .getResult();
  op.getResult().replaceAllUsesWith(replacement);
  return success();
}

static LogicalResult lowerVLoad(VMIvLoadOp op, OpBuilder &builder) {
  LogicalResult result = success();
  if (op.getGroupAttr()) {
    result = lowerGroupedLoad(op, builder);
  } else if (op.getBlockStride()) {
    result = lowerBlockStrideLoad(op, builder);
  } else {
    result = lowerContinuousLoad(op, builder);
  }
  if (failed(result)) {
    return failure();
  }
  op.erase();
  return success();
}

static LogicalResult lowerBlockStrideStore(VMIvStoreOp op,
                                           OpBuilder &builder) {
  auto valueType = cast<VMIVRegType>(op.getValue().getType());
  Value mask = op.getMask().empty()
                   ? createAllActiveMask(valueType, op.getLoc(), builder)
                   : op.getMask()[0];
  builder.create<VMIStrideStoreOp>(
      op.getLoc(), op.getValue(), op.getDestination(), op.getOffset(),
      op.getBlockStride(), mask);
  return success();
}

static LogicalResult lowerContinuousStore(VMIvStoreOp op,
                                           OpBuilder &builder) {
  Value value = op.getValue();
  Value mask = op.getMask().empty() ? Value() : op.getMask().front();
  auto valueType = cast<VMIVRegType>(value.getType());
  int64_t numGroups = valueType.getElementCount();
  bool canUseGroupStore = isCompactGroupCount(numGroups) &&
                          (!mask || isAllActiveSeed(mask));
  if (canUseGroupStore) {
    Value unitStride = builder.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    builder.create<VMIGroupStoreOp>(
        op.getLoc(), value, op.getDestination(), op.getOffset(),
        unitStride, builder.getI64IntegerAttr(numGroups));
    return success();
  }
  if (mask) {
    builder.create<VMIMaskedStoreOp>(op.getLoc(), value,
                                     op.getDestination(), op.getOffset(), mask);
  } else {
    builder.create<VMIStoreOp>(op.getLoc(), value,
                               op.getDestination(), op.getOffset());
  }
  return success();
}

/// A group store whose row stride equals the per-group element count writes the
/// rows back to back, so the whole access is one dense contiguous store.
/// kGroupStoreLayoutPatterns only covers a few group shapes, so recognizing the
/// degenerate row stride here (instead of registering every group size) keeps
/// the store on the dense path.  One-lane groups are the slot form: the compact
/// and group_slots paths already lower those, so they keep the group form, and
/// so does any group store that carries a mask.
static bool isDenseAliasedGroupStore(VMIvStoreOp op) {
  if (!op.getMask().empty()) {
    return false;
  }
  auto valueType = dyn_cast<VMIVRegType>(op.getValue().getType());
  if (!valueType) {
    return false;
  }
  // The dense store keeps the value and destination element types equal; the
  // group store is also the packed-narrowing store (e.g. i32 slots to a u8
  // row), so a mismatched row must keep the group form.
  Type destinationElementType;
  if (auto destinationPtr = dyn_cast<PtrType>(op.getDestination().getType())) {
    destinationElementType = destinationPtr.getElementType();
  } else if (auto destinationMemRef =
                 dyn_cast<MemRefType>(op.getDestination().getType())) {
    destinationElementType = destinationMemRef.getElementType();
  }
  if (destinationElementType != valueType.getElementType()) {
    return false;
  }
  int64_t numGroups = op.getGroupAttr().getInt();
  int64_t elementCount = valueType.getElementCount();
  if (numGroups <= 0 || elementCount <= 0 || elementCount % numGroups != 0) {
    return false;
  }
  int64_t groupSize = elementCount / numGroups;
  if (groupSize < mlir::pto::kValue2) {
    return false;
  }
  std::optional<int64_t> rowStride = getConstantIndexValue(op.getStride());
  return rowStride && *rowStride == groupSize;
}

static LogicalResult lowerVStore(VMIvStoreOp op, OpBuilder &builder) {
  // Stores are mask-governed by contract: the unified op carries no pmode, so
  // there is no merge/zero selection to translate.
  LogicalResult result = success();
  if (op.getGroupAttr()) {
    if (isDenseAliasedGroupStore(op)) {
      // Rows are packed, so the dense store aliases the group store exactly.
      result = lowerContinuousStore(op, builder);
    } else {
      builder.create<VMIGroupStoreOp>(
          op.getLoc(), op.getValue(), op.getDestination(), op.getOffset(),
          op.getStride(), op.getGroupAttr());
    }
  } else if (op.getBlockStride()) {
    result = lowerBlockStrideStore(op, builder);
  } else {
    result = lowerContinuousStore(op, builder);
  }
  if (failed(result)) {
    return failure();
  }
  op.erase();
  return success();
}

static bool areMemoryEffectsFreeBetween(Operation *first, Operation *second) {
  Block *firstBlock = first ? first->getBlock() : nullptr;
  Block *secondBlock = second ? second->getBlock() : nullptr;
  if (!first || !second || firstBlock != secondBlock) {
    return false;
  }
  for (Operation *op = first->getNextNode(); op && op != second;
       op = op->getNextNode()) {
    if (!isMemoryEffectFree(op)) {
      return false;
    }
  }
  return first->isBeforeInBlock(second);
}

static bool isOffsetPlusElements(Value base, Value candidate,
                                 int64_t elementCount) {
  if (auto baseValue = getConstantIndexValue(base)) {
    if (auto candidateValue = getConstantIndexValue(candidate)) {
      return *baseValue <= std::numeric_limits<int64_t>::max() - elementCount &&
             *candidateValue == *baseValue + elementCount;
    }
  }
  auto add = candidate.getDefiningOp<arith::AddIOp>();
  if (!add) {
    return false;
  }
  auto lhsOffset = getConstantIndexValue(add.getLhs());
  auto rhsOffset = getConstantIndexValue(add.getRhs());
  return (add.getLhs() == base && rhsOffset == elementCount) ||
         (add.getRhs() == base && lhsOffset == elementCount);
}

static bool isContiguousVMIStore(VMIvStoreOp store) {
  return !store.getStride() && !store.getBlockStride() && !store.getGroup() &&
         store.getMask().empty();
}

static bool combineDeinterleaveLoads(VMIVdintlvOp deinterleave) {
  if (!isAllActiveSeed(deinterleave.getMask())) {
    return false;
  }
  auto firstLoad = deinterleave.getLhs().getDefiningOp<VMIvLoadOp>();
  auto secondLoad = deinterleave.getRhs().getDefiningOp<VMIvLoadOp>();
  if (!firstLoad || !secondLoad || firstLoad == secondLoad ||
      firstLoad.getSource() != secondLoad.getSource() ||
      firstLoad.getGroup() || secondLoad.getGroup() || firstLoad.getStride() ||
      secondLoad.getStride() || firstLoad.getBlockStride() ||
      secondLoad.getBlockStride() || !firstLoad.getResult().hasOneUse() ||
      !secondLoad.getResult().hasOneUse() ||
      *firstLoad.getResult().user_begin() != deinterleave.getOperation() ||
      *secondLoad.getResult().user_begin() != deinterleave.getOperation()) {
    return false;
  }

  auto firstType = cast<VMIVRegType>(firstLoad.getResult().getType());
  int64_t laneCount = firstType.getElementCount();
  if (laneCount <= 0 ||
      !isOffsetPlusElements(firstLoad.getOffset(), secondLoad.getOffset(),
                            laneCount) ||
      !areMemoryEffectsFreeBetween(firstLoad, secondLoad)) {
    return false;
  }

  OpBuilder builder(firstLoad);
  auto fusedLoad = builder.create<VMIDeinterleaveLoadOp>(
      deinterleave.getLoc(), deinterleave.getLow().getType(),
      deinterleave.getHigh().getType(), firstLoad.getSource(),
      firstLoad.getOffset());
  deinterleave.getLow().replaceAllUsesWith(fusedLoad.getLow());
  deinterleave.getHigh().replaceAllUsesWith(fusedLoad.getHigh());
  deinterleave.erase();
  firstLoad.erase();
  secondLoad.erase();
  return true;
}

static bool combineInterleaveStores(VMIVintlvOp interleave) {
  bool hasAllActiveMask = isAllActiveSeed(interleave.getMask());
  bool lowHasOneUse = interleave.getLow().hasOneUse();
  bool highHasOneUse = interleave.getHigh().hasOneUse();
  if (!hasAllActiveMask || !lowHasOneUse || !highHasOneUse) {
    return false;
  }
  auto firstStore = dyn_cast<VMIvStoreOp>(*interleave.getLow().user_begin());
  auto secondStore = dyn_cast<VMIvStoreOp>(*interleave.getHigh().user_begin());
  if (!firstStore || !secondStore || firstStore == secondStore ||
      firstStore.getValue() != interleave.getLow() ||
      secondStore.getValue() != interleave.getHigh() ||
      !isContiguousVMIStore(firstStore) ||
      !isContiguousVMIStore(secondStore) ||
      firstStore.getDestination() != secondStore.getDestination()) {
    return false;
  }

  auto valueType = cast<VMIVRegType>(interleave.getLhs().getType());
  int64_t laneCount = valueType.getElementCount();
  if (laneCount <= 0 ||
      !isOffsetPlusElements(firstStore.getOffset(), secondStore.getOffset(),
                            laneCount) ||
      !areMemoryEffectsFreeBetween(firstStore, secondStore)) {
    return false;
  }

  OpBuilder builder(firstStore);
  builder.create<VMIInterleaveStoreOp>(
      interleave.getLoc(), interleave.getLhs(), interleave.getRhs(),
      firstStore.getDestination(), firstStore.getOffset());
  firstStore.erase();
  secondStore.erase();
  interleave.erase();
  return true;
}

static void combineInterleavedMemoryOps(ModuleOp module) {
  SmallVector<VMIVdintlvOp, mlir::pto::kValue8> deinterleaves;
  SmallVector<VMIVintlvOp, mlir::pto::kValue8> interleaves;
  module.walk([&](Operation *op) {
    if (auto deinterleave = dyn_cast<VMIVdintlvOp>(op)) {
      deinterleaves.push_back(deinterleave);
    } else if (auto interleave = dyn_cast<VMIVintlvOp>(op)) {
      interleaves.push_back(interleave);
    }
  });

  for (VMIVdintlvOp op : deinterleaves) {
    if (op->getBlock()) {
      (void)combineDeinterleaveLoads(op);
    }
  }
  for (VMIVintlvOp op : interleaves) {
    if (op->getBlock()) {
      (void)combineInterleaveStores(op);
    }
  }
}

// Category C4 helpers: pset / pge
//===----------------------------------------------------------------------===//

/// Lower pset "PAT_ALL" → create_mask(all_lanes).
static LogicalResult lowerPset(VMIPsetOp op, OpBuilder &builder) {
  // If an all-active consumer (e.g. vcmp) elided its use, drop the seed
  // entirely instead of materialising a dead create_mask.
  if (op.use_empty()) {
    op->erase();
    return success();
  }
  Location loc = op.getLoc();
  auto maskType = cast<VMIMaskType>(op.getResult().getType());
  int64_t laneCount = maskType.getElementCount();
  auto indexType = IndexType::get(builder.getContext());
  Value activeLanes = builder.create<arith::ConstantOp>(
      loc, indexType, builder.getIndexAttr(laneCount));
  Value result =
      builder.create<VMICreateMaskOp>(loc, maskType, activeLanes).getResult();
  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

/// Lower pge "PAT_VLN" → create_mask(N).
/// When {group = C} is present → create_group_mask(N, num_groups=C,
/// group_size = total_lanes / C).
static LogicalResult lowerPge(VMIPgeOp op, OpBuilder &builder) {
  StringRef pattern = op.getPattern();
  // Parse "PAT_VL<num>" or fall back to "PAT_VL16".
  int64_t numLanes = 16;
  if (pattern.starts_with("PAT_VL")) {
    StringRef numStr = pattern.drop_front(6); // strlen("PAT_VL")
    int64_t parsed = 0;
    bool hasValidLaneCount =
        !numStr.empty() && !numStr.getAsInteger(kDecimalRadix, parsed) &&
        parsed > 0;
    if (hasValidLaneCount) {
      numLanes = parsed;
    }
  }

  Location loc = op.getLoc();
  auto maskType = cast<VMIMaskType>(op.getResult().getType());
  auto indexType = IndexType::get(builder.getContext());
  Value activeLanes = builder.create<arith::ConstantOp>(
      loc, indexType, builder.getIndexAttr(numLanes));

  if (auto groupAttr = op.getGroupAttr()) {
    // Grouped tail mask → create_group_mask
    int64_t numGroups = groupAttr.getInt();
    int64_t totalLanes = maskType.getElementCount();
    int64_t groupSize = totalLanes / numGroups;
    Value result =
        builder
            .create<VMICreateGroupMaskOp>(loc, maskType, activeLanes,
                           builder.getI64IntegerAttr(numGroups),
                           builder.getI64IntegerAttr(groupSize))
                       .getResult();
    op.getResult().replaceAllUsesWith(result);
  } else {
    Value result =
        builder.create<VMICreateMaskOp>(loc, maskType, activeLanes).getResult();
    op.getResult().replaceAllUsesWith(result);
  }
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Category C6 helpers: vcadd / vcmax / vcmin
//===----------------------------------------------------------------------===//

enum class LegacyReductionKind { Unsupported, Integer, Float };

/// Classify supported legacy reductions before any lowering creates IR.
template <typename ReductionOp>
static LegacyReductionKind getLegacyReductionKind(ReductionOp op) {
  if (auto pmode = op.getPmode(); pmode && *pmode != "zero") {
    return LegacyReductionKind::Unsupported;
  }
  Type elementType = cast<VMIVRegType>(op.getSource().getType()).getElementType();
  if (elementType.isF16() || elementType.isF32()) {
    return LegacyReductionKind::Float;
  }
  if (auto integerType = dyn_cast<IntegerType>(elementType)) {
    unsigned width = integerType.getWidth();
    if (width == 16 || width == 32) {
      return LegacyReductionKind::Integer;
    }
  }
  return LegacyReductionKind::Unsupported;
}

template <typename ReductionOp>
static std::optional<int64_t> getReductionNumGroups(ReductionOp op) {
  if (auto groupAttr = op.getGroupAttr()) {
    return groupAttr.getInt();
  }

  // A full reduction is one logical group. Keep the alias decision local to
  // the reduction instead of relying on a downstream store to mutate it.
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (!resultType.getLayoutAttr()) {
    return 1;
  }
  return std::nullopt;
}

/// Share type/group dispatch and replacement across the legacy reductions.
/// Only floating add needs an extra attribute (reassoc).
template <typename FloatOp, typename IntegerOp, typename GroupFloatOp,
          typename GroupIntegerOp, typename ReductionOp, typename... FloatAttrs>
static LogicalResult lowerLegacyReduction(ReductionOp op, OpBuilder &builder,
                                          FloatAttrs... floatAttrs) {
  LegacyReductionKind kind = getLegacyReductionKind(op);
  if (kind == LegacyReductionKind::Unsupported) {
    return failure();
  }

  bool isFloat = kind == LegacyReductionKind::Float;
  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  Value source = op.getSource();
  Value mask = op.getMask();
  Value result;
  if (std::optional<int64_t> numGroups = getReductionNumGroups(op)) {
    auto groupAttr = builder.getI64IntegerAttr(*numGroups);
    if (isFloat) {
      result = builder
                   .create<GroupFloatOp>(loc, resultType, source, mask,
                                         groupAttr, floatAttrs...)
                   .getResult();
    } else {
      result =
          builder
              .create<GroupIntegerOp>(loc, resultType, source, mask, groupAttr)
              .getResult();
    }
  } else if (isFloat) {
    result =
        builder.create<FloatOp>(loc, resultType, source, mask, floatAttrs...)
            .getResult();
  } else {
    result =
        builder.create<IntegerOp>(loc, resultType, source, mask).getResult();
  }
  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

/// Lower vcadd to legacy full or grouped float/integer addition reduction.
static LogicalResult lowerVCadd(VMIvcaddOp op, OpBuilder &builder) {
  return lowerLegacyReduction<VMIReduceAddFOp, VMIReduceAddIOp,
                              VMIGroupReduceAddFOp, VMIGroupReduceAddIOp>(
      op, builder, op.getReassocAttr());
}

/// Lower vcmax to legacy full or grouped float/integer maximum reduction.
static LogicalResult lowerVcmax(VMIvcmaxOp op, OpBuilder &builder) {
  return lowerLegacyReduction<VMIReduceMaxFOp, VMIReduceMaxIOp,
                              VMIGroupReduceMaxFOp, VMIGroupReduceMaxIOp>(
      op, builder);
}

/// Lower vcmin to legacy full or grouped float/integer minimum reduction.
static LogicalResult lowerVcmin(VMIvcminOp op, OpBuilder &builder) {
  return lowerLegacyReduction<VMIReduceMinFOp, VMIReduceMinIOp,
                              VMIGroupReduceMinFOp, VMIGroupReduceMinIOp>(
      op, builder);
}

/// Lower plt(rem:i32) -> create_mask(min(rem, L)) + arith remainder chain.
///   %act  = arith.minsi %rem, %cL         // min(rem, L)
///   %aidx = arith.index_cast %act          // i32 -> index
///   %mask = vmi.create_mask %aidx
///   %next = arith.subi %rem, %act          // rem - min(rem, L) = max(rem-L, 0)
static LogicalResult lowerPlt(VMIPltOp op, OpBuilder &builder) {
  Location loc = op.getLoc();
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  int64_t laneCount = maskType.getElementCount();

  auto i32Type = builder.getIntegerType(32);
  Value cL = builder.create<arith::ConstantOp>(
      loc, i32Type, builder.getIntegerAttr(i32Type, laneCount));
  Value act = builder.create<arith::MinSIOp>(loc, i32Type, op.getScalar(), cL);
  Value aidx = builder.create<arith::IndexCastOp>(
      loc, builder.getIndexType(), act);
  Value mask = builder.create<VMICreateMaskOp>(loc, maskType, aidx).getResult();
  Value next = builder.create<arith::SubIOp>(loc, i32Type, op.getScalar(), act);

  op.getMask().replaceAllUsesWith(mask);
  op.getScalarOut().replaceAllUsesWith(next);
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Category C8 helpers: vgather / vscatter
//===----------------------------------------------------------------------===//

/// Lower vgather to legacy gather.  Legacy gather carries an explicit passthru
/// operand for inactive lanes; the zero predicate mode is modelled with a zero
/// passthru.
static LogicalResult lowerVgather(VMIVgatherOp op, OpBuilder &builder) {
  Location loc = op.getLoc();
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  // pmode="zero" (default): inactive lanes are zeroed. Legacy gather models
  // inactive lanes with an explicit passthru whose element type must match the
  // result, so synthesise a zero constant of the result type — the offsets
  // vector cannot be reused because its element type (e.g. i32) generally
  // differs from the result element type (e.g. f32).
  Value passthru = createZeroConstant(builder, loc, resultType);
  Value result =
      builder
          .create<VMIGatherOp>(loc, resultType, op.getSource(), op.getOffsets(),
                               op.getMask(), passthru)
                     .getResult();
  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

/// Lower vscatter to legacy scatter.  Legacy scatter only writes active lanes
/// (mask-governed), matching vscatter's zero predicate mode.
static LogicalResult lowerVscatter(VMIVscatterOp op, OpBuilder &builder) {
  Location loc = op.getLoc();
  builder.create<VMIScatterOp>(loc, op.getValue(), op.getDestination(),
                               op.getOffsets(), op.getMask());
  op->erase();
  return success();
}

/// Splits one dual-form bitwise op onto its two interfaces.
///
/// Mask operands go to the mask interface (`MaskOp`); vreg operands go to the
/// vreg interface (`DataOp`), which keeps the governed predicate mask and pmode
/// so the VMI-to-VPTO lowering can forward them as the predication operand.
template <typename MaskOp, typename DataOp, typename UnifiedOp>
static void lowerBitwiseOnVRegOrMask(UnifiedOp op, OpBuilder &builder) {
  builder.setInsertionPoint(op);
  Value result;
  if (isa<VMIMaskType>(op.getLhs().getType())) {
    result = builder.create<MaskOp>(op.getLoc(), op.getResult().getType(),
                                    op.getLhs(), op.getRhs())
                 .getResult();
  } else {
    result = builder
                 .create<DataOp>(op.getLoc(), op.getResult().getType(),
                                 op.getLhs(), op.getRhs(), op.getMask(),
                                 op.getPmodeAttr())
                 .getResult();
  }
  op.getResult().replaceAllUsesWith(result);
  op.erase();
}

/// Unary counterpart of `lowerBitwiseOnVRegOrMask` for `pto.vmi.vnot`.
template <typename MaskOp, typename DataOp, typename UnifiedOp>
static void lowerBitwiseNotOnVRegOrMask(UnifiedOp op, OpBuilder &builder) {
  builder.setInsertionPoint(op);
  Value result;
  if (isa<VMIMaskType>(op.getSource().getType())) {
    result = builder
                 .create<MaskOp>(op.getLoc(), op.getResult().getType(),
                                 op.getSource())
                 .getResult();
  } else {
    result = builder
                 .create<DataOp>(op.getLoc(), op.getResult().getType(),
                                 op.getSource(), op.getMask(),
                                 op.getPmodeAttr())
                 .getResult();
  }
  op.getResult().replaceAllUsesWith(result);
  op.erase();
}

//===----------------------------------------------------------------------===//
// Small-shape f16 vexpdif expansion
//===----------------------------------------------------------------------===//

// The f16 vexpdif lowering needs one packed f16 source part to become
// EVEN/ODD f32 result parts, a contiguous -> deinterleave(2) relation that
// only materializes once the source fills whole f16 physical parts: the
// interleave hardware entry consumes a whole v128f16 part (this threshold
// must stay in sync with kVexpdifInterleaveLanes in
// VPTOCANN900LLVMEmitterInternal.h), and the 32-bit c <-> d(2)
// ensure_layout bridge in VMILayoutSupportTables.inc carries the matching
// N<128, 256> bounds for the public lane counts (1, 2, 4, 8, 64, 128, 256;
// see docs/isa/vmi-isa/00-architecture-overview.md). Below one part the
// layout tables pin the result to a 1:1 part relation, the 2x arity check
// in the VMIToVPTO pattern fails, and the op survives as a residual. Widen
// x and max to f32 first and keep the op on the 1:1 f32 path, which every
// public lane count lowers. f16 values are exact in f32, so the documented
// per-lane semantics exp(x - max) is unchanged.
static void expandSmallShapeF16Vexpdif(ModuleOp module) {
  FailureOr<int64_t> f16LanesPerPart =
      getDataLanesPerPart(Float16Type::get(module.getContext()));
  if (failed(f16LanesPerPart)) {
    return;
  }
  module.walk([&](VMIVexpdifOp op) {
    auto xType = cast<VMIVRegType>(op.getX().getType());
    bool needsWidening = xType.getElementType().isF16() &&
                         xType.getElementCount() < *f16LanesPerPart;
    if (!needsWidening) {
      return;
    }
    OpBuilder builder(op);
    // The widened type stays layout-less: layout assignment derives the
    // widening cast relation itself (ls(2) f16 -> contiguous f32), and the
    // source layout would only synthesize a relation the legal table does
    // not carry (e.g. ls(2) -> ls(2) for a 16 -> 32 cast).
    auto wideType = VMIVRegType::get(op.getContext(), xType.getElementCount(),
                                     builder.getF32Type(), Attribute());
    Value x = builder.create<VMIExtFOp>(op.getLoc(), wideType, op.getX())
                   .getResult();
    Value max = builder.create<VMIExtFOp>(op.getLoc(), wideType, op.getMax())
                    .getResult();
    op.getXMutable().assign(x);
    op.getMaxMutable().assign(max);
  });
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace {

struct VMILowerUnifiedToLegacyPass
    : public mlir::pto::impl::VMILowerUnifiedToLegacyBase<
          VMILowerUnifiedToLegacyPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMILowerUnifiedToLegacyPass)

  void runOnOperation() override;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
  }
};

} // namespace

void VMILowerUnifiedToLegacyPass::runOnOperation() {
  ModuleOp module = getOperation();
  combineInterleavedMemoryOps(module);
  expandSmallShapeF16Vexpdif(module);
  SmallVector<Operation *, mlir::pto::kValue128> worklist;

  // Collect all unified VMI ops (walk encounters them in IR order).
  module.walk([&](Operation *op) {
    // Category A
    if (isa<VMIVciOp, VMIVinterpretCastOp, VMIvSelOp, VMIVbrcOp>(op) ||
        // Category B — dual-form bitwise ops are split onto the mask interface
        // (mask operands) or the vreg interface (vreg operands, governed mask
        // kept) by the dispatch below, so both forms join the worklist.
        isa<VMIVandOp, VMIVorOp, VMIVxorOp, VMIVnotOp>(op) ||
        // Category C1
        isa<VMIVcmpOp, VMIVcmpsOp>(op) ||
        // Category C2
        isa<VMICvtOp>(op) ||
        // Category C3
        isa<VMIvLoadOp, VMIvStoreOp, VMIVsstbOp>(op) ||
        // Category C4
        isa<VMIPsetOp, VMIPgeOp, VMIPltOp>(op) ||
        // Category C6 — unified reduce (partial coverage)
        isa<VMIvcaddOp, VMIvcmaxOp, VMIvcminOp>(op) ||
        // Category C8 — indexed gather / scatter
        isa<VMIVgatherOp, VMIVscatterOp>(op)) {
      worklist.push_back(op);
    }

    // Category D — no legacy equivalent (require direct VMIToVPTO lowering):
    //   vector-scalar ops, vaddc/vaddcs, vintlv, vdintlv, vselr,
    //   vgatherb, vmull
    // These are intentionally NOT added to the worklist — they flow through
    // to VMIToVPTO which must provide direct 1:N lowering patterns.
    // (`plt` belongs to Category C4 above and does have a legacy lowering.)
    if (isa<VMIAddSOp, VMIMulSOp, VMIMaxSOp, VMIMinSOp, VMIShlSOp, VMIShrSOp,
            VMIVaddcOp, VMIVsubcOp, VMIVaddcsOp, VMIVsubcsOp,
            VMIVintlvOp, VMIVdintlvOp, VMIVselrOp, VMIVgatherbOp, VMIVmullOp>(
            op)) {
      op->emitRemark("VMI unified op has no legacy equivalent — "
                     "requires direct VMIToVPTO 1:N lowering");
    }
  });

  // Process consumers before producers to avoid stale producer uses.
  for (Operation *op : llvm::reverse(worklist)) {
    if (!op->getBlock()) {
      continue;
    }
    OpBuilder builder(op);

    // ---- Category A: pure syntactic renames ----

    if (auto vop = dyn_cast<VMIVciOp>(op)) {
      // Public vci without grouping (or group=1) is ordinary continuous iota.
      // group>1 lowers to the internal contiguous-only group_iota producer.
      builder.setInsertionPoint(op);
      StringAttr orderAttr;
      if (auto order = vop.getOrder()) {
        orderAttr = builder.getStringAttr(*order);
      }

      Type resultType = vop.getResult().getType();
      IntegerAttr groupAttr = vop.getGroupAttr();
      if (groupAttr && groupAttr.getInt() > 1) {
        if (auto vmiTy = dyn_cast<VMIVRegType>(resultType)) {
          VMILayoutAttr layout = vmiTy.getLayoutAttr();
          if (layout && !layout.isContiguous()) {
            Type contigType = VMIVRegType::get(
                op->getContext(), vmiTy.getElementCount(),
                vmiTy.getElementType(),
                VMILayoutAttr::getContiguous(op->getContext()));
            Value contig =
                builder
                    .create<VMIGroupIotaOp>(op->getLoc(), contigType,
                                            vop.getBase(), orderAttr, groupAttr)
                    .getResult();
            Value converted =
                builder
                    .create<VMIEnsureLayoutOp>(op->getLoc(), vmiTy, contig)
                    .getResult();
            vop.getResult().replaceAllUsesWith(converted);
            op->erase();
            continue;
          }
        }
        Value grouped =
            builder
                .create<VMIGroupIotaOp>(op->getLoc(), resultType, vop.getBase(),
                                        orderAttr, groupAttr)
                .getResult();
        vop.getResult().replaceAllUsesWith(grouped);
        op->erase();
        continue;
      }

      Value iota = builder
                       .create<VMIIotaOp>(op->getLoc(), resultType,
                                          vop.getBase(), orderAttr)
              .getResult();
      vop.getResult().replaceAllUsesWith(iota);
      op->erase();
      continue;
    }

    if (auto vop = dyn_cast<VMIVinterpretCastOp>(op)) {
      // vinterpret_cast -> bitcast
      builder.setInsertionPoint(op);
      Value result =
          builder
              .create<VMIBitcastOp>(op->getLoc(), vop.getResult().getType(),
                                    vop.getSource())
              .getResult();
      vop.getResult().replaceAllUsesWith(result);
      op->erase();
      continue;
    }

    if (auto vop = dyn_cast<VMIvSelOp>(op)) {
      // vsel -> select
      builder.setInsertionPoint(op);
      Value result =
          builder
              .create<VMISelectOp>(op->getLoc(), vop.getResult().getType(),
                                   vop.getMask(), vop.getTrueValue(),
                                   vop.getFalseValue())
              .getResult();
      vop.getResult().replaceAllUsesWith(result);
      op->erase();
      continue;
    }

    if (auto vop = dyn_cast<VMIVbrcOp>(op)) {
      // vbrc -> broadcast; vbrc{group} -> group_broadcast
      builder.setInsertionPoint(op);
      Value result;
      if (vop.getGroupAttr()) {
        result = builder
                     .create<VMIGroupBroadcastOp>(
                         op->getLoc(), vop.getResult().getType(),
                                             vop.getValue(), vop.getGroupAttr())
                .getResult();
      } else {
        result =
            builder
                .create<VMIBroadcastOp>(op->getLoc(), vop.getResult().getType(),
                                        vop.getValue())
                .getResult();
      }
      vop.getResult().replaceAllUsesWith(result);
      op->erase();
      continue;
    }

    // ---- Category C4: static mask creation ----

    if (auto vop = dyn_cast<VMIPsetOp>(op)) {
      (void)lowerPset(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIPgeOp>(op)) {
      (void)lowerPge(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIPltOp>(op)) {
      (void)lowerPlt(vop, builder);
      continue;
    }

    // ---- Category C1: vcmp / vcmps ----

    if (auto vop = dyn_cast<VMIVcmpOp>(op)) {
      (void)lowerVCmp(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIVcmpsOp>(op)) {
      (void)lowerVCmps(vop, builder);
      continue;
    }

    // ---- Category C2: vcvt ----

    if (auto vop = dyn_cast<VMICvtOp>(op)) {
      (void)lowerVCvt(vop, builder);
      continue;
    }

    // ---- Category C3: vload / vstore ----

    if (auto vop = dyn_cast<VMIvLoadOp>(op)) {
      (void)lowerVLoad(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIvStoreOp>(op)) {
      (void)lowerVStore(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIVsstbOp>(op)) {
      builder.create<VMIStrideStoreOp>(vop.getLoc(), vop.getValue(),
                                       vop.getDestination(), vop.getOffset(),
          vop.getBlockStride(), vop.getMask());
      vop->erase();
      continue;
    }

    // ---- Category C6: unified reduce ----

    if (auto vop = dyn_cast<VMIvcaddOp>(op)) {
      (void)lowerVCadd(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIvcmaxOp>(op)) {
      (void)lowerVcmax(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIvcminOp>(op)) {
      (void)lowerVcmin(vop, builder);
      continue;
    }

    // ---- Category C8: indexed gather / scatter ----

    if (auto vop = dyn_cast<VMIVgatherOp>(op)) {
      (void)lowerVgather(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIVscatterOp>(op)) {
      (void)lowerVscatter(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIVandOp>(op)) {
      lowerBitwiseOnVRegOrMask<VMIMaskAndOp, VMIAndIOp>(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIVorOp>(op)) {
      lowerBitwiseOnVRegOrMask<VMIMaskOrOp, VMIOrIOp>(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIVxorOp>(op)) {
      lowerBitwiseOnVRegOrMask<VMIMaskXOrOp, VMIXOrIOp>(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIVnotOp>(op)) {
      lowerBitwiseNotOnVRegOrMask<VMIMaskNotOp, VMINotOp>(vop, builder);
      continue;
    }
  }
}

std::unique_ptr<Pass> mlir::pto::createVMILowerUnifiedToLegacyPass() {
  return std::make_unique<VMILowerUnifiedToLegacyPass>();
}
