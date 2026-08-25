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
// Category B — elementwise arithmetic / bitwise (18 ops):
//   vadd/vsub/vmul/vdiv/vmin/vmax → legacy type-specific binary op
//   vneg/vabs/vsqrt/vexp/vln/vrelu → legacy unary op
//   vand/vor/vxor/vshl/vshr/vnot → legacy bitwise op
//   vshr selects shrui for unsigned/signless elements and shrsi for
//   explicitly signed elements.
//   Mask/pmode synthesis is intentionally bypassed here so two-stage lowering
//   does not introduce select chains before layout assignment.
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
// Category C7 — fused multiply-add family → legacy fma (2 ops):
//   vmula → fma               (float only; mask discarded; int → skipped, no legacy int fma)
//   vaxpy → broadcast + fma   (float only)
//
// Category C8 — indexed gather/scatter → legacy gather/scatter (2 ops):
//   vgather  → gather   (pmode="zero": passthru = zero constant)
//   vscatter → scatter
//
// Category C9 — fused activation / softmax (3 ops):
//   vexpdif → kept unified for direct VMI-to-VPTO fused lowering
//   vlrelu  → maxf + minf + broadcast + mulf + addf
//   vprelu  → maxf + minf + mulf + addf
//   Lowered Category C7/C8/C9 ops bypass mask/pmode synthesis here and skip
//   pmode="merge".
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
// Category B: binary elementwise → legacy compute, mask/pmode discarded
//===----------------------------------------------------------------------===//

/// Lower a BINARY unified op (vadd, vsub, ...) to a legacy compute op.
/// Unified mask and pmode are intentionally discarded.
///
/// \p createLegacy is a callable `(Location, Type, Value, Value) -> Value`
/// that emits the legacy binary op.
template <typename UnifiedOp>
static LogicalResult
lowerBinaryIgnoringMask(
    UnifiedOp op,
    function_ref<Value(Location, Type, Value, Value)> createLegacy) {
  if (hasMergePmode(op)) {
    return failure();
  }
  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();

  Value raw = createLegacy(loc, resultType, lhs, rhs);
  op.getResult().replaceAllUsesWith(raw);
  op->erase();
  return success();
}

/// Lower a UNARY unified op (vneg, vabs, …) to its legacy counterpart.
template <typename UnifiedOp>
static LogicalResult
lowerMaskedUnary(UnifiedOp op, OpBuilder &builder,
                 function_ref<Value(Location, Type, Value)> createLegacy) {
  if (hasMergePmode(op)) {
    return failure();
  }

  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  Value source = op.getSource();

  Value raw = createLegacy(loc, resultType, source);
  op.getResult().replaceAllUsesWith(raw);
  op->erase();
  return success();
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
    result =
        builder.create<VMITruncFOp>(loc, resultType, source, roundingAttr,
                                     saturateAttr)
            .getResult();
  } else if (direction == "fptosi") {
    result =
        builder
            .create<VMIFPToSIOp>(loc, resultType, source,
                                 op.getRoundingAttr(), saturateAttr)
            .getResult();
  } else if (direction == "fptoui") {
    result =
        builder
            .create<VMIFPToUIOp>(loc, resultType, source,
                                 op.getRoundingAttr(), saturateAttr)
            .getResult();
  } else if (direction == "sitofp") {
    result =
        builder.create<VMISIToFPOp>(loc, resultType, source).getResult();
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

/// Lower vload by dispatching on dist_mode.
static LogicalResult lowerVLoad(VMIvLoadOp op, OpBuilder &builder) {
  // Group mode: vload {group=C} → group_load or group_slot_load
  //   elementCount == num_groups → group_slot_load (compact, 1 scalar per group)
  //   elementCount  > num_groups → group_load (full groups)
  // Broadcast-style slot loads (dense output, stride=1) should be decomposed
  // at the VMI level: vload(compact) + vbrc.
  if (op.getGroupAttr()) {
    auto resultType = cast<VMIVRegType>(op.getResults().front().getType());
    int64_t numGroups = op.getGroupAttr().getInt();
    // {group, dist_mode="brc"}: group broadcast — one scalar per group
    // broadcast within each group (e.g. E2B).
    if (op.getDistMode() && op.getDistMode() == "brc") {
      auto gbl = builder.create<VMIGroupBroadcastLoadOp>(
          op->getLoc(), resultType, op.getSource(), op.getOffset(),
          op.getStride(), op.getGroupAttr());
      op.getResults().front().replaceAllUsesWith(gbl.getResult());
    } else if (resultType.getElementCount() == numGroups) {
      auto slotLoad = builder.create<VMIGroupSlotLoadOp>(
          op->getLoc(), resultType, op.getSource(), op.getOffset(),
          op.getStride(), op.getGroupAttr());
      op.getResults().front().replaceAllUsesWith(slotLoad.getResult());
    } else {
      auto groupLoad = builder.create<VMIGroupLoadOp>(
          op->getLoc(), resultType, op.getSource(), op.getOffset(),
          op.getStride(), op.getGroupAttr());
      op.getResults().front().replaceAllUsesWith(groupLoad.getResult());
    }
    op->erase();
    return success();
  }

  // Block-stride mode: vload {block_stride} → stride_load
  if (op.getBlockStride()) {
    auto resultType = op.getResults().front().getType();
    // Create default all-active mask via create_mask with proper granularity
    // and layout, so the legacy stride_load passes both-or-neither layout
    // checks.  pset cannot be used here because pred masks cannot carry
    // layout.
    auto resultVMIType = cast<VMIVRegType>(resultType);
    auto elemType = resultVMIType.getElementType();
    unsigned bits = 32;
    if (auto it = dyn_cast<IntegerType>(elemType)) {
      bits = it.getWidth();
    } else if (auto ft = dyn_cast<FloatType>(elemType)) {
      bits = ft.getWidth();
    }
    auto gran = StringAttr::get(builder.getContext(),
                                bits <= 8 ? "b8" : bits <= 16 ? "b16" : "b32");
    auto maskType = VMIMaskType::get(builder.getContext(),
                                     resultVMIType.getElementCount(), gran,
                                     resultVMIType.getLayout());
    auto fullLanes = builder.create<arith::ConstantOp>(
        op->getLoc(), builder.getIndexAttr(resultVMIType.getElementCount()));
    auto mask = builder.create<VMICreateMaskOp>(op->getLoc(), maskType,
                                                fullLanes.getResult());
    Value bs = op.getBlockStride();
    auto strideLoad = builder.create<VMIStrideLoadOp>(
        op->getLoc(), resultType, op.getSource(), op.getOffset(), bs,
        mask.getResult());
    op.getResults().front().replaceAllUsesWith(strideLoad.getResult());
    op->erase();
    return success();
  }

  // pmode="merge" cannot be expressed by legacy load + select — skip.
  if (hasMergePmode(op)) {
    return failure();
  }

  StringAttr distModeAttr = op.getDistModeAttr();
  StringRef distMode =
      distModeAttr ? distModeAttr.getValue() : "continuous";

  Location loc = op.getLoc();
  Value source = op.getSource();
  Value offset = op.getOffset();

  if (distMode == "continuous") {
    auto resultType = cast<VMIVRegType>(op.getResults().front().getType());
    int64_t numGroups = resultType.getElementCount();
    if (isCompactGroupCount(numGroups)) {
      Value unitStride = builder.create<arith::ConstantIndexOp>(loc, 1);
      auto loadOp = builder.create<VMIGroupSlotLoadOp>(
          loc, resultType, source, offset, unitStride,
          builder.getI64IntegerAttr(numGroups));
      op.getResults().front().replaceAllUsesWith(loadOp.getResult());
    } else {
      auto loadOp = builder.create<VMILoadOp>(loc, resultType, source, offset);
      op.getResults().front().replaceAllUsesWith(loadOp.getResult());
    }
  } else if (distMode == "dintlv") {
    auto dloadOp = builder.create<VMIDeinterleaveLoadOp>(
        loc, op.getResults()[0].getType(), op.getResults()[1].getType(),
        source, offset);
    op.getResults()[0].replaceAllUsesWith(dloadOp.getLow());
    op.getResults()[1].replaceAllUsesWith(dloadOp.getHigh());
  } else if (distMode == "brc") {
    // vload {dist_mode="brc"} -> group_broadcast_load.
    //   - no group attr: all-lane scalar broadcast (num_groups=1, stride=0).
    //   - {group = C, stride}: per-group scalar broadcast — one scalar loaded
    //     per group and broadcast within each group (e.g. E2B).
    auto resultType = op.getResults().front().getType();
    if (auto groupAttr = op.getGroupAttr()) {
      int64_t numGroups = groupAttr.getInt();
      Value stride = op.getStride();
      auto gbl = builder.create<VMIGroupBroadcastLoadOp>(
          loc, resultType, source, offset, stride,
          builder.getI64IntegerAttr(numGroups));
      op.getResults().front().replaceAllUsesWith(gbl.getResult());
    } else {
      Value zeroStride = builder.create<arith::ConstantOp>(
          loc, builder.getIndexType(), builder.getIndexAttr(0));
      auto gbl = builder.create<VMIGroupBroadcastLoadOp>(
          loc, resultType, source, offset, zeroStride,
          builder.getI64IntegerAttr(1));
      op.getResults().front().replaceAllUsesWith(gbl.getResult());
    }
  } else {
    // "unpack" has no legacy equivalent (physical widening, lane count changes).
    return failure();
  }

  op->erase();
  return success();
}

/// Lower vstore by dispatching on dist_mode.
static LogicalResult lowerVStore(VMIvStoreOp op, OpBuilder &builder) {
  // pmode="merge" (inactive lanes retain the prior destination contents)
  // cannot be expressed by the legacy store family, whose writes are governed
  // purely by the mask. Skip instead of silently dropping the attribute.
  if (hasMergePmode(op)) {
    return failure();
  }

  // Group mode: vstore {group=C} → group_store
  if (op.getGroupAttr()) {
    builder.create<VMIGroupStoreOp>(
        op->getLoc(), op.getValues()[0], op.getDestination(), op.getOffset(),
        op.getStride(), op.getGroupAttr());
    op->erase();
    return success();
  }

  // Block-stride mode: vstore {block_stride} → stride_store
  if (op.getBlockStride()) {
    auto valueType = cast<VMIVRegType>(op.getValues()[0].getType());
    // Use existing mask or create default all-active via create_mask
    // (pset cannot be used — pred masks cannot carry layout).
    Value mask;
    if (!op.getMask().empty()) {
      mask = op.getMask()[0];
    } else {
      auto elemType = valueType.getElementType();
      unsigned bits = 32;
      if (auto it = dyn_cast<IntegerType>(elemType)) {
        bits = it.getWidth();
      } else if (auto ft = dyn_cast<FloatType>(elemType)) {
        bits = ft.getWidth();
      }
      auto gran = StringAttr::get(builder.getContext(),
                                  bits <= 8 ? "b8" : bits <= 16 ? "b16" : "b32");
      auto maskType = VMIMaskType::get(builder.getContext(),
                                       valueType.getElementCount(), gran,
                                       valueType.getLayout());
      auto fullLanes = builder.create<arith::ConstantOp>(
          op->getLoc(), builder.getIndexAttr(valueType.getElementCount()));
      mask = builder.create<VMICreateMaskOp>(op->getLoc(), maskType,
                                             fullLanes.getResult())
                 .getResult();
    }
    Value bs = op.getBlockStride();
    builder.create<VMIStrideStoreOp>(op->getLoc(), op.getValues()[0],
                                    op.getDestination(), op.getOffset(), bs,
                                    mask);
    op->erase();
    return success();
  }

  StringAttr distModeAttr = op.getDistModeAttr();
  StringRef distMode =
      distModeAttr ? distModeAttr.getValue() : "continuous";

  Location loc = op.getLoc();
  Value dest = op.getDestination();
  Value offset = op.getOffset();
  auto values = op.getValues();

  if (distMode == "continuous") {
    if (values.empty()) {
      return failure();
    }
    Value mask = op.getMask().empty() ? Value() : op.getMask().front();
    auto valueType = cast<VMIVRegType>(values[0].getType());
    int64_t numGroups = valueType.getElementCount();

    // A compact 1/2/4/8-lane value contains one scalar per logical group.
    // Keep masked stores unchanged unless their mask is provably all-active:
    // group_store currently has no dynamic predication operand.
    bool allActive = !mask || isAllActiveSeed(mask);
    bool compact = isCompactGroupCount(numGroups);
    if (compact && allActive) {
      Value unitStride = builder.create<arith::ConstantIndexOp>(loc, 1);
      builder.create<VMIGroupStoreOp>(loc, values[0], dest, offset, unitStride,
                                      builder.getI64IntegerAttr(numGroups));
      op->erase();
      return success();
    }
    if (mask) {
      // Masked store path.
      builder.create<VMIMaskedStoreOp>(loc, values[0], dest, offset, mask);
    } else {
      builder.create<VMIStoreOp>(loc, values[0], dest, offset);
    }
  } else if (distMode == "dintlv") {
    if (values.size() < mlir::pto::kValue2) {
      return failure();
    }
    builder.create<VMIInterleaveStoreOp>(loc, values[0], values[1], dest,
                                         offset);
  } else {
    return failure();
  }

  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
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
    if (!numStr.empty()) {
      int64_t parsed = 0;
      for (char c : numStr) {
        if (c < '0' || c > '9') {
          break;
        }
        parsed = parsed * kDecimalRadix + (c - '0');
      }
      if (parsed > 0) {
        numLanes = parsed;
      }
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
    Value result = builder
                       .create<VMICreateGroupMaskOp>(
                           loc, maskType, activeLanes,
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

//===----------------------------------------------------------------------===//
// Category C7 helpers: vmula / vaxpy (fused multiply-add → legacy fma)
//===----------------------------------------------------------------------===//

/// Lower vmula (acc = acc + lhs*rhs) to legacy fma (lhs*rhs + acc).
/// The mask operand (if present) is discarded — legacy fma has no mask.
/// Legacy fma is floating-point only; integer vmula has no legacy equivalent
/// and is skipped (falls through to VMIToVPTO).
static LogicalResult lowerVmula(VMIVmulaOp op, OpBuilder &builder) {
  if (hasMergePmode(op)) {
    return failure();
  }

  Type resultType = op.getResult().getType();
  auto vmiType = cast<VMIVRegType>(resultType);
  if (!isFloatType(vmiType.getElementType())) {
    return failure();
  }

  Location loc = op.getLoc();
  // fma computes lhs*rhs + acc, matching vmula's acc + lhs*rhs.
  Value result = builder
                     .create<VMIFmaOp>(loc, resultType, op.getLhs(), op.getRhs(),
                                       op.getAcc())
                     .getResult();
  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

/// Lower vaxpy (alpha*x + y) to broadcast(alpha) + legacy fma.
/// alpha is a scalar float, broadcast to a vector before the fma.
static LogicalResult lowerVaxpy(VMIVaxpyOp op, OpBuilder &builder) {
  if (hasMergePmode(op)) {
    return failure();
  }

  Type resultType = op.getResult().getType();
  auto vmiType = cast<VMIVRegType>(resultType);
  if (!isFloatType(vmiType.getElementType())) {
    return failure();
  }

  Location loc = op.getLoc();
  Value alphaVec = builder
                       .create<VMIBroadcastOp>(loc, resultType, op.getAlpha())
                       .getResult();
  // fma(alpha, x, y) == alpha*x + y.
  Value raw = builder
                  .create<VMIFmaOp>(loc, resultType, alphaVec, op.getX(),
                                    op.getAcc())
                  .getResult();
  op.getResult().replaceAllUsesWith(raw);
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
  Value result = builder
                     .create<VMIGatherOp>(loc, resultType, op.getSource(),
                                          op.getOffsets(), op.getMask(),
                                          passthru)
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

//===----------------------------------------------------------------------===//
// Category C9 helpers: vlrelu / vprelu (fused → legacy chains)
//===----------------------------------------------------------------------===//

/// Lower vlrelu (x>0 ? x : slope*x) to max(x,0) + slope*min(x,0).
/// slope is a scalar float broadcast to a vector.
static LogicalResult lowerVlrelu(VMIVlreluOp op, OpBuilder &builder) {
  if (hasMergePmode(op)) {
    return failure();
  }

  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  auto vmiType = cast<VMIVRegType>(resultType);
  Value x = op.getX();

  Value zeroConst = createZeroConstant(builder, loc, vmiType);
  Value pos =
      builder.create<VMIMaxFOp>(loc, resultType, x, zeroConst).getResult();
  Value neg =
      builder.create<VMIMinFOp>(loc, resultType, x, zeroConst).getResult();
  Value slopeVec = builder
                       .create<VMIBroadcastOp>(loc, resultType, op.getSlope())
                       .getResult();
  Value scaledNeg =
      builder.create<VMIMulFOp>(loc, resultType, slopeVec, neg).getResult();
  Value raw =
      builder.create<VMIAddFOp>(loc, resultType, pos, scaledNeg).getResult();
  op.getResult().replaceAllUsesWith(raw);
  op->erase();
  return success();
}

/// Lower vprelu (max(x,0) + alpha*min(x,0)) to legacy max/min/mul/add.
/// alpha is a per-lane vector (no broadcast needed).
static LogicalResult lowerVprelu(VMIVpreluOp op, OpBuilder &builder) {
  if (hasMergePmode(op)) {
    return failure();
  }

  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  auto vmiType = cast<VMIVRegType>(resultType);
  Value x = op.getX();

  Value zeroConst = createZeroConstant(builder, loc, vmiType);
  Value pos =
      builder.create<VMIMaxFOp>(loc, resultType, x, zeroConst).getResult();
  Value neg =
      builder.create<VMIMinFOp>(loc, resultType, x, zeroConst).getResult();
  Value scaledNeg =
      builder.create<VMIMulFOp>(loc, resultType, op.getAlpha(), neg).getResult();
  Value raw =
      builder.create<VMIAddFOp>(loc, resultType, pos, scaledNeg).getResult();
  op.getResult().replaceAllUsesWith(raw);
  op->erase();
  return success();
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
        // Category B — binary
        isa<VMIVaddOp, VMIVsubOp, VMIVmulOp, VMIVdivOp, VMIVminOp, VMIVmaxOp,
            VMIVandOp, VMIVorOp, VMIVxorOp, VMIVshlOp, VMIVshrOp>(op) ||
        // Category B — unary
        isa<VMIVnegOp, VMIVabsOp, VMIVsqrtOp, VMIVexpOp, VMIVlnOp, VMIVreluOp,
            VMIVnotOp>(op) ||
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
        // Category C7 — fused multiply-add family → legacy fma
        isa<VMIVmulaOp, VMIVaxpyOp>(op) ||
        // Category C8 — indexed gather / scatter
        isa<VMIVgatherOp, VMIVscatterOp>(op) ||
        // Category C9 — fused activation / softmax (legacy chains)
        isa<VMIVlreluOp, VMIVpreluOp>(op)) {
      worklist.push_back(op);
    }

    // Category D — no legacy equivalent (require direct VMIToVPTO lowering):
    //   plt, vector-scalar ops, vaddc/vaddcs, vintlv, vdintlv, vselr,
    //   vgatherb, vmull
    // These are intentionally NOT added to the worklist — they flow through
    // to VMIToVPTO which must provide direct 1:N lowering patterns.
    if (isa<VMIAddSOp, VMIMulSOp, VMIMaxSOp, VMIMinSOp, VMIShlSOp, VMIShrSOp,
            VMIVaddcOp, VMIVaddcsOp,
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

      Value iota =
          builder
              .create<VMIIotaOp>(op->getLoc(), resultType, vop.getBase(),
                                 orderAttr)
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
        result =
            builder
                .create<VMIGroupBroadcastOp>(op->getLoc(), vop.getResult().getType(),
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
      builder.create<VMIStrideStoreOp>(
          vop.getLoc(), vop.getValue(), vop.getDestination(), vop.getOffset(),
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

    // ---- Category C7: fused multiply-add family ----

    if (auto vop = dyn_cast<VMIVmulaOp>(op)) {
      (void)lowerVmula(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIVaxpyOp>(op)) {
      (void)lowerVaxpy(vop, builder);
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

    // ---- Category C9: fused activation / softmax ----

    if (auto vop = dyn_cast<VMIVlreluOp>(op)) {
      (void)lowerVlrelu(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIVpreluOp>(op)) {
      (void)lowerVprelu(vop, builder);
      continue;
    }

    // ---- Category B: binary elementwise, mask/pmode discarded ----

    if (auto vop = dyn_cast<VMIVaddOp>(op)) {
      Type elemType = getVMIElementType(vop.getResult());
      auto createLegacy = [&](Location loc, Type ty, Value lhs, Value rhs) -> Value {
        if (isFloatType(elemType)) {
          return builder.create<VMIAddFOp>(loc, ty, lhs, rhs).getResult();
        }
        return builder.create<VMIAddIOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerBinaryIgnoringMask(vop, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVsubOp>(op)) {
      Type elemType = getVMIElementType(vop.getResult());
      auto createLegacy = [&](Location loc, Type ty, Value lhs, Value rhs) -> Value {
        if (isFloatType(elemType)) {
          return builder.create<VMISubFOp>(loc, ty, lhs, rhs).getResult();
        }
        return builder.create<VMISubIOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerBinaryIgnoringMask(vop, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVmulOp>(op)) {
      Type elemType = getVMIElementType(vop.getResult());
      auto createLegacy = [&](Location loc, Type ty, Value lhs, Value rhs) -> Value {
        if (isFloatType(elemType)) {
          return builder.create<VMIMulFOp>(loc, ty, lhs, rhs).getResult();
        }
        return builder.create<VMIMulIOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerBinaryIgnoringMask(vop, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVdivOp>(op)) {
      // Float-only; no legacy integer divide.
      auto createLegacy = [&](Location loc, Type ty, Value lhs,
                              Value rhs) -> Value {
        return builder.create<VMIDivFOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerBinaryIgnoringMask(vop, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVminOp>(op)) {
      Type elemType = getVMIElementType(vop.getResult());
      auto createLegacy = [&](Location loc, Type ty, Value lhs,
                              Value rhs) -> Value {
        if (isFloatType(elemType)) {
          return builder.create<VMIMinFOp>(loc, ty, lhs, rhs).getResult();
        }
        return builder.create<VMIMinIOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerBinaryIgnoringMask(vop, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVmaxOp>(op)) {
      Type elemType = getVMIElementType(vop.getResult());
      auto createLegacy = [&](Location loc, Type ty, Value lhs,
                              Value rhs) -> Value {
        if (isFloatType(elemType)) {
          return builder.create<VMIMaxFOp>(loc, ty, lhs, rhs).getResult();
        }
        return builder.create<VMIMaxIOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerBinaryIgnoringMask(vop, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVandOp>(op)) {
      builder.setInsertionPoint(op);
      if (isa<VMIMaskType>(vop.getLhs().getType())) {
        // Mask logic path: lower to mask_and.
        Value result = builder.create<VMIMaskAndOp>(
            vop.getLoc(), vop.getResult().getType(),
            vop.getLhs(), vop.getRhs()).getResult();
        vop.getResult().replaceAllUsesWith(result);
        op->erase();
        continue;
      }
      auto createLegacy = [&](Location loc, Type ty, Value lhs,
                              Value rhs) -> Value {
        return builder.create<VMIAndIOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerBinaryIgnoringMask(vop, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVorOp>(op)) {
      builder.setInsertionPoint(op);
      if (isa<VMIMaskType>(vop.getLhs().getType())) {
        // Mask logic path: lower to mask_or.
        Value result = builder.create<VMIMaskOrOp>(
            vop.getLoc(), vop.getResult().getType(),
            vop.getLhs(), vop.getRhs()).getResult();
        vop.getResult().replaceAllUsesWith(result);
        op->erase();
        continue;
      }
      auto createLegacy = [&](Location loc, Type ty, Value lhs,
                              Value rhs) -> Value {
        return builder.create<VMIOrIOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerBinaryIgnoringMask(vop, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVxorOp>(op)) {
      builder.setInsertionPoint(op);
      if (isa<VMIMaskType>(vop.getLhs().getType())) {
        // Mask logic path: lower to mask_xor.
        Value result = builder.create<VMIMaskXOrOp>(
            vop.getLoc(), vop.getResult().getType(),
            vop.getLhs(), vop.getRhs()).getResult();
        vop.getResult().replaceAllUsesWith(result);
        op->erase();
        continue;
      }
      auto createLegacy = [&](Location loc, Type ty, Value lhs,
                              Value rhs) -> Value {
        return builder.create<VMIXOrIOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerBinaryIgnoringMask(vop, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVshlOp>(op)) {
      auto createLegacy = [&](Location loc, Type ty, Value lhs,
                              Value rhs) -> Value {
        return builder.create<VMIShLIOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerBinaryIgnoringMask(vop, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVshrOp>(op)) {
      Type elemType = getVMIElementType(vop.getLhs());
      auto createLegacy = [&](Location loc, Type ty, Value lhs,
                              Value rhs) -> Value {
        auto intType = cast<IntegerType>(elemType);
        if (!intType.isSigned()) {
          return builder.create<VMIShRUIOp>(loc, ty, lhs, rhs).getResult();
        }
        return builder.create<VMIShRSIOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerBinaryIgnoringMask(vop, createLegacy);
      continue;
    }

    // ---- Category B: masked elementwise — unary ----

    if (auto vop = dyn_cast<VMIVnegOp>(op)) {
      Type elemType = getVMIElementType(vop.getResult());
      auto createLegacy = [&](Location loc, Type ty, Value src) -> Value {
        if (isFloatType(elemType)) {
          return builder.create<VMINegFOp>(loc, ty, src).getResult();
        }
        return builder.create<VMINegIOp>(loc, ty, src).getResult();
      };
      (void)lowerMaskedUnary(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVabsOp>(op)) {
      Type elemType = getVMIElementType(vop.getResult());
      auto createLegacy = [&](Location loc, Type ty, Value src) -> Value {
        // bf16 has no vector absf; clear the sign bit (ASC: vand with 0x7FFF).
        if (elemType.isBF16()) {
          auto srcTy = cast<VMIVRegType>(src.getType());
          Type i16 = builder.getIntegerType(16);
          auto iTy = VMIVRegType::get(builder.getContext(),
                                      srcTy.getElementCount(), i16,
                                      srcTy.getLayout());
          Value asI = builder.create<VMIBitcastOp>(loc, iTy, src);
          Value c = builder.create<arith::ConstantOp>(
              loc, i16, builder.getIntegerAttr(i16, 0x7FFF));
          Value maskVec =
              builder.create<VMIBroadcastOp>(loc, iTy, c).getResult();
          Value cleared =
              builder.create<VMIAndIOp>(loc, iTy, asI, maskVec).getResult();
          return builder.create<VMIBitcastOp>(loc, ty, cleared).getResult();
        }
        if (isFloatType(elemType)) {
          return builder.create<VMIAbsFOp>(loc, ty, src).getResult();
        }
        return builder.create<VMIAbsIOp>(loc, ty, src).getResult();
      };
      (void)lowerMaskedUnary(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVsqrtOp>(op)) {
      auto createLegacy = [&](Location loc, Type ty, Value src) -> Value {
        return builder.create<VMISqrtOp>(loc, ty, src).getResult();
      };
      (void)lowerMaskedUnary(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVexpOp>(op)) {
      auto createLegacy = [&](Location loc, Type ty, Value src) -> Value {
        return builder.create<VMIExpOp>(loc, ty, src).getResult();
      };
      (void)lowerMaskedUnary(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVlnOp>(op)) {
      auto createLegacy = [&](Location loc, Type ty, Value src) -> Value {
        return builder.create<VMILnOp>(loc, ty, src).getResult();
      };
      (void)lowerMaskedUnary(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVreluOp>(op)) {
      auto createLegacy = [&](Location loc, Type ty, Value src) -> Value {
        return builder.create<VMIReluOp>(loc, ty, src).getResult();
      };
      (void)lowerMaskedUnary(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVnotOp>(op)) {
      builder.setInsertionPoint(op);
      if (isa<VMIMaskType>(vop.getSource().getType())) {
        // Mask logic path: lower to mask_not.
        Value result = builder.create<VMIMaskNotOp>(
            vop.getLoc(), vop.getResult().getType(),
            vop.getSource()).getResult();
        vop.getResult().replaceAllUsesWith(result);
        op->erase();
        continue;
      }
      auto createLegacy = [&](Location loc, Type ty, Value src) -> Value {
        return builder.create<VMINotOp>(loc, ty, src).getResult();
      };
      (void)lowerMaskedUnary(vop, builder, createLegacy);
      continue;
    }
  }
}

std::unique_ptr<Pass> mlir::pto::createVMILowerUnifiedToLegacyPass() {
  return std::make_unique<VMILowerUnifiedToLegacyPass>();
}
