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
//   vload  → dispatch by dist_mode/group/block_stride to
//            load / deinterleave_load / group_broadcast_load{num_groups=1} / ...
//   vstore → dispatch to store / masked_store / interleave_store / group_store / ...
//   Continuous 1/2/4/8-lane values alias unit-stride
//   group_slot_load/group_store operations.
//   Skipped: dist_mode "unpack" (physical widening, no legacy equivalent).
//
// Category C4 — static mask creation (3 ops):
//   pset → create_mask(all lanes)
//   pge  → create_mask(N lanes)
//   plt  → create_mask(min(rem, L))
//
// Category C3 — unified load/store (2 ops, dispatch by dist_mode/group):
//   vload → load / deinterleave_load / group_load
//   vstore → store / masked_store / interleave_store / group_store
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
//   vaddc vaddcs vintlv vdintlv vselr vgatherb vmull
//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"

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

/// Returns the string name of a predicate mode, defaulting to "zero".
static StringRef getPmodeOrDefault(Operation *op, StringRef attrName = "pmode") {
  if (auto attr = op->getAttrOfType<StringAttr>(attrName)) {
    return attr.getValue();
  }
  return "zero";
}

/// Returns true when the pmode on `op` is "merge" — these ops must be
/// skipped because merge semantic (inactive lane preserves OLD_DEST) cannot
/// be expressed in VMI SSA IR.
static bool hasMergePmode(Operation *op) {
  return getPmodeOrDefault(op) == "merge";
}

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
  if (hasMergePmode(op)) {
    return failure();
  }

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
  if (hasMergePmode(op)) {
    return failure();
  }

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
  if (hasMergePmode(op)) {
    return failure();
  }

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
  unsigned bits = mlir::pto::kValue32;
  if (auto integerType = dyn_cast<IntegerType>(elementType)) {
    bits = integerType.getWidth();
  } else if (auto floatType = dyn_cast<FloatType>(elementType)) {
    bits = floatType.getWidth();
  }
  StringRef granularity = bits <= mlir::pto::kValue8
                              ? "b8"
                              : bits <= mlir::pto::kValue16 ? "b16" : "b32";
  return builder.getStringAttr(granularity);
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
  auto resultType = cast<VMIVRegType>(op.getResults().front().getType());
  int64_t numGroups = op.getGroupAttr().getInt();
  bool isBroadcast = op.getDistMode() && op.getDistMode() == "brc";
  Value replacement;
  if (isBroadcast) {
    replacement = builder
                      .create<VMIGroupBroadcastLoadOp>(
                          op.getLoc(), resultType, op.getSource(),
                          op.getOffset(), op.getStride(), op.getGroupAttr())
                      .getResult();
  } else if (resultType.getElementCount() == numGroups) {
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
  op.getResults().front().replaceAllUsesWith(replacement);
  return success();
}

static LogicalResult lowerBlockStrideLoad(VMIvLoadOp op,
                                          OpBuilder &builder) {
  auto resultType = cast<VMIVRegType>(op.getResults().front().getType());
  Value mask = createAllActiveMask(resultType, op.getLoc(), builder);
  Value replacement =
      builder
          .create<VMIStrideLoadOp>(op.getLoc(), resultType, op.getSource(),
                                   op.getOffset(), op.getBlockStride(), mask)
          .getResult();
  op.getResults().front().replaceAllUsesWith(replacement);
  return success();
}

// A continuous vload is a dense access whatever its length.  Short vectors are
// not group packets: how few elements a load touches is a property of the
// physical access VMIToVPTO plans for it, not of the logical lane layout, so
// the length must not change which legacy op -- and therefore which layout
// contract -- the value carries.
static LogicalResult lowerContinuousLoad(VMIvLoadOp op,
                                         OpBuilder &builder) {
  auto resultType = cast<VMIVRegType>(op.getResults().front().getType());
  Value replacement = builder
                          .create<VMILoadOp>(op.getLoc(), resultType,
                                             op.getSource(), op.getOffset())
                          .getResult();
  op.getResults().front().replaceAllUsesWith(replacement);
  return success();
}

static LogicalResult lowerDeinterleaveLoad(VMIvLoadOp op,
                                           OpBuilder &builder) {
  auto load = builder.create<VMIDeinterleaveLoadOp>(
      op.getLoc(), op.getResults()[0].getType(), op.getResults()[1].getType(),
      op.getSource(), op.getOffset());
  op.getResults()[0].replaceAllUsesWith(load.getLow());
  op.getResults()[1].replaceAllUsesWith(load.getHigh());
  return success();
}

static LogicalResult lowerBroadcastLoad(VMIvLoadOp op, OpBuilder &builder) {
  Location loc = op.getLoc();
  Type resultType = op.getResults().front().getType();
  Value stride = builder.create<arith::ConstantOp>(
      loc, builder.getIndexType(), builder.getIndexAttr(0));
  auto load = builder.create<VMIGroupBroadcastLoadOp>(
      loc, resultType, op.getSource(), op.getOffset(), stride,
      builder.getI64IntegerAttr(1));
  op.getResults().front().replaceAllUsesWith(load.getResult());
  return success();
}

static LogicalResult lowerDistributedLoad(VMIvLoadOp op,
                                          OpBuilder &builder) {
  StringAttr modeAttr = op.getDistModeAttr();
  StringRef mode = modeAttr ? modeAttr.getValue() : "continuous";
  if (mode == "continuous") {
    return lowerContinuousLoad(op, builder);
  }
  if (mode == "dintlv") {
    return lowerDeinterleaveLoad(op, builder);
  }
  if (mode == "brc") {
    return lowerBroadcastLoad(op, builder);
  }
  return failure();
}

static LogicalResult lowerVLoad(VMIvLoadOp op, OpBuilder &builder) {
  LogicalResult result = success();
  if (op.getGroupAttr()) {
    result = lowerGroupedLoad(op, builder);
  } else if (op.getBlockStride()) {
    result = lowerBlockStrideLoad(op, builder);
  } else if (hasMergePmode(op)) {
    return failure();
  } else {
    result = lowerDistributedLoad(op, builder);
  }
  if (failed(result)) {
    return failure();
  }
  op.erase();
  return success();
}

static LogicalResult lowerBlockStrideStore(VMIvStoreOp op,
                                           OpBuilder &builder) {
  auto valueType = cast<VMIVRegType>(op.getValues()[0].getType());
  Value mask = op.getMask().empty()
                   ? createAllActiveMask(valueType, op.getLoc(), builder)
                   : op.getMask()[0];
  builder.create<VMIStrideStoreOp>(
      op.getLoc(), op.getValues()[0], op.getDestination(), op.getOffset(),
      op.getBlockStride(), mask);
  return success();
}

static LogicalResult lowerContinuousStore(VMIvStoreOp op,
                                           OpBuilder &builder) {
  ValueRange values = op.getValues();
  if (values.empty()) {
    return failure();
  }
  Value mask = op.getMask().empty() ? Value() : op.getMask().front();
  auto valueType = cast<VMIVRegType>(values.front().getType());
  int64_t numGroups = valueType.getElementCount();
  bool canUseGroupStore = isCompactGroupCount(numGroups) &&
                          (!mask || isAllActiveSeed(mask));
  if (canUseGroupStore) {
    Value unitStride = builder.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    builder.create<VMIGroupStoreOp>(
        op.getLoc(), values.front(), op.getDestination(), op.getOffset(),
        unitStride, builder.getI64IntegerAttr(numGroups));
    return success();
  }
  if (mask) {
    builder.create<VMIMaskedStoreOp>(op.getLoc(), values.front(),
                                     op.getDestination(), op.getOffset(), mask);
  } else {
    builder.create<VMIStoreOp>(op.getLoc(), values.front(),
                               op.getDestination(), op.getOffset());
  }
  return success();
}

static LogicalResult lowerDistributedStore(VMIvStoreOp op,
                                           OpBuilder &builder) {
  StringAttr modeAttr = op.getDistModeAttr();
  StringRef mode = modeAttr ? modeAttr.getValue() : "continuous";
  if (mode == "continuous") {
    return lowerContinuousStore(op, builder);
  }
  if (mode == "intlv") {
    ValueRange values = op.getValues();
    if (values.size() < mlir::pto::kValue2) {
      return failure();
    }
    builder.create<VMIInterleaveStoreOp>(
        op.getLoc(), values[0], values[1], op.getDestination(), op.getOffset());
    return success();
  }
  return failure();
}

static LogicalResult lowerVStore(VMIvStoreOp op, OpBuilder &builder) {
  if (hasMergePmode(op)) {
    return failure();
  }
  LogicalResult result = success();
  if (op.getGroupAttr()) {
    builder.create<VMIGroupStoreOp>(
        op.getLoc(), op.getValues()[0], op.getDestination(), op.getOffset(),
        op.getStride(), op.getGroupAttr());
  } else if (op.getBlockStride()) {
    result = lowerBlockStrideStore(op, builder);
  } else {
    result = lowerDistributedStore(op, builder);
  }
  if (failed(result)) {
    return failure();
  }
  op.erase();
  return success();
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

/// Lower vcadd to legacy reduce_addf/reduce_addi or
/// group_reduce_addf/group_reduce_addi.  Always succeeds for valid input
/// (vcadd verifier guarantees reassoc for float, and group 整除 source lanes).
static LogicalResult lowerVCadd(VMIvcaddOp op, OpBuilder &builder) {
  if (hasMergePmode(op)) {
    return failure();
  }

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  Type elemType = sourceType.getElementType();
  bool isFloat = isa<FloatType>(elemType);
  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  Value source = op.getSource();
  Value mask = op.getMask();

  if (std::optional<int64_t> numGroups = getReductionNumGroups(op)) {
    // Group reduce path
    Value result;
    if (isFloat) {
      result =
          builder
              .create<VMIGroupReduceAddFOp>(loc, resultType, source, mask,
                                            builder.getI64IntegerAttr(*numGroups),
                                            op.getReassocAttr())
              .getResult();
    } else {
      result =
          builder
              .create<VMIGroupReduceAddIOp>(loc, resultType, source, mask,
                                            builder.getI64IntegerAttr(*numGroups))
              .getResult();
}
    op.getResult().replaceAllUsesWith(result);
  } else {
    // Full reduce path
    Value result;
    if (isFloat) {
      result =
          builder
              .create<VMIReduceAddFOp>(loc, resultType, source, mask,
                                       op.getReassocAttr())
              .getResult();
    } else {
      result =
          builder
              .create<VMIReduceAddIOp>(loc, resultType, source, mask)
              .getResult();
}
    op.getResult().replaceAllUsesWith(result);
  }
  op->erase();
  return success();
}

/// Lower vcmax to legacy full or grouped float/integer maximum reduction.
static LogicalResult lowerVcmax(VMIvcmaxOp op, OpBuilder &builder) {
  if (hasMergePmode(op)) {
    return failure();
  }

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  Type elemType = sourceType.getElementType();
  bool isFloat = isa<FloatType>(elemType);
  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  Value source = op.getSource();
  Value mask = op.getMask();

  if (std::optional<int64_t> numGroups = getReductionNumGroups(op)) {
    // Group reduce path
    Value result;
    if (isFloat) {
      result =
          builder
              .create<VMIGroupReduceMaxFOp>(loc, resultType, source, mask,
                                            builder.getI64IntegerAttr(*numGroups))
              .getResult();
    } else {
      result =
          builder
              .create<VMIGroupReduceMaxIOp>(loc, resultType, source, mask,
                                            builder.getI64IntegerAttr(*numGroups))
              .getResult();
}
    op.getResult().replaceAllUsesWith(result);
    op->erase();
    return success();
  }

  Value result;
  if (isFloat) {
    result = builder
                 .create<VMIReduceMaxFOp>(loc, resultType, source, mask)
                 .getResult();
  } else {
    result = builder
                 .create<VMIReduceMaxIOp>(loc, resultType, source, mask)
                 .getResult();
}
  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

/// Lower vcmin to legacy full or grouped float/integer minimum reduction.
static LogicalResult lowerVcmin(VMIvcminOp op, OpBuilder &builder) {
  if (hasMergePmode(op)) {
    return failure();
  }

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  Type elemType = sourceType.getElementType();
  bool isFloat = isa<FloatType>(elemType);
  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  Value source = op.getSource();
  Value mask = op.getMask();

  if (std::optional<int64_t> numGroups = getReductionNumGroups(op)) {
    Value result;
    if (isFloat) {
      result = builder
                   .create<VMIGroupReduceMinFOp>(
                       loc, resultType, source, mask,
                       builder.getI64IntegerAttr(*numGroups))
                   .getResult();
    } else {
      result = builder
                   .create<VMIGroupReduceMinIOp>(
                       loc, resultType, source, mask,
                       builder.getI64IntegerAttr(*numGroups))
                   .getResult();
}
    op.getResult().replaceAllUsesWith(result);
    op->erase();
    return success();
  }

  Value result;
  if (isFloat) {
    result = builder
                 .create<VMIReduceMinFOp>(loc, resultType, source, mask)
                 .getResult();
  } else {
    result = builder
                 .create<VMIReduceMinIOp>(loc, resultType, source, mask)
                 .getResult();
}
  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
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
/// operand for inactive lanes; pmode="zero" is modelled with a zero passthru.
/// pmode="merge" (preserve OLD_DEST) has no SSA passthru and is skipped.
static LogicalResult lowerVgather(VMIVgatherOp op, OpBuilder &builder) {
  if (hasMergePmode(op)) {
    return failure();
  }

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
/// (mask-governed), matching vscatter's default/zero pmode; merge is skipped.
static LogicalResult lowerVscatter(VMIVscatterOp op, OpBuilder &builder) {
  if (hasMergePmode(op)) {
    return failure();
  }

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
      // pmode="merge" cannot be expressed by the legacy stride store; leave
      // the op for VMIToVPTO (which has no vsstb pattern) so the conversion
      // fails loudly instead of silently dropping the attribute.
      if (hasMergePmode(vop)) {
        continue;
      }
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
