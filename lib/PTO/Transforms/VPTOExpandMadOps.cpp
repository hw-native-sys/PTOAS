// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "VPTOExpandMadOps.h"

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;

namespace {

static Value castIntegerLikeTo(Location loc, Value value, Type targetType,
                               PatternRewriter &rewriter) {
  if (value.getType() == targetType) {
    return value;
  }

  auto targetInt = dyn_cast<IntegerType>(targetType);
  if (value.getType().isIndex() && targetInt) {
    return rewriter.create<arith::IndexCastOp>(loc, targetType, value);
  }
  if (auto sourceInt = dyn_cast<IntegerType>(value.getType())) {
    if (targetInt) {
      if (sourceInt.getWidth() < targetInt.getWidth()) {
        return rewriter.create<arith::ExtUIOp>(loc, targetType, value);
      }
      if (sourceInt.getWidth() > targetInt.getWidth()) {
        return rewriter.create<arith::TruncIOp>(loc, targetType, value);
      }
      return value;
    }
    if (targetType.isIndex()) {
      return rewriter.create<arith::IndexCastOp>(loc, targetType, value);
    }
  }

  return {};
}

struct MadXtConfig {
  Value m;
  Value n;
  Value k;
  std::optional<pto::MadUnitFlagMode> unitFlagMode;
  bool disableGemv;
  bool cmatrixSource;
  bool cmatrixInit;
  // Runtime-operand overrides (take precedence over the static fields above).
  Value unitFlagValue;   // i32, expected domain 0/2/3; packed to bits 55-56
  Value accInitValue;    // i1, packed to bit 63 (zero-Cmatrix)
  Value disableGemvValue;// i1, packed to bit 61
  Value biasInitValue;   // i1, packed to bit 62 (BTbuf)
};

// Shared i64 bit arithmetic used to assemble an `xt` word.
struct MadXtBitPacker {
  Location loc;
  PatternRewriter &rewriter;

  Value constant(uint64_t value) const {
    return rewriter.create<arith::ConstantIntOp>(loc, value,
                                                 mlir::pto::kValue64);
  }
  Value shl(Value value, uint64_t amount) const {
    return rewriter.create<arith::ShLIOp>(loc, value, constant(amount));
  }
  Value bitOr(Value lhs, Value rhs) const {
    return rewriter.create<arith::OrIOp>(loc, lhs, rhs);
  }
};

// One optional flag bit: the runtime operand when present, otherwise the
// statically known value, packed at `shift`.
struct MadXtFlagBit {
  Value value;
  bool staticValue;
  uint64_t shift;
};

// Coerce a runtime flag value to i64. The operands are signless-only in ODS;
// the PTODSL frontend strips signedness (reconciled before expansion), so no
// reinterpret cast is needed here.
static Value coerceMadFlagToI64(Location loc, Value value, Type i64Ty,
                                PatternRewriter &rewriter) {
  return castIntegerLikeTo(loc, value, i64Ty, rewriter);
}

// Truncate a coerced flag value to its `width`-bit field so an out-of-domain
// operand cannot spill into the neighbouring xt bits. `width` is a
// compile-time constant of the xt layout (1 for the single-bit flags, 2 for
// unit_flag), so the mask is folded at build time.
template <uint64_t width>
static Value truncateMadFlagField(const MadXtBitPacker &packer,
                                  Value flagI64) {
  static_assert(width > 0 && width < 64, "field width must fit in a uint64");
  constexpr uint64_t mask = (uint64_t(1) << width) - 1;
  return packer.rewriter.create<arith::AndIOp>(
      packer.loc, flagI64, packer.constant(mask));
}

// Or-in one flag bit at `bit.shift`, preferring the runtime operand over the
// statically known value (1 packs the bit, 0 skips it). The runtime operand is
// masked to one bit first: the operand type admits any i32 value, so a value
// above 1 must not leak into the neighbouring flag fields.
static FailureOr<Value> packMadFlagBit(const MadXtBitPacker &packer, Value xt,
                                       const MadXtFlagBit &bit, Type i64Ty) {
  if (bit.value) {
    Value flagI64 =
        coerceMadFlagToI64(packer.loc, bit.value, i64Ty, packer.rewriter);
    if (!flagI64) {
      return failure();
    }
    flagI64 = truncateMadFlagField<1>(packer, flagI64);
    return packer.bitOr(xt, packer.shl(flagI64, bit.shift));
  }
  if (bit.staticValue) {
    return packer.bitOr(xt, packer.shl(packer.constant(1), bit.shift));
  }
  return xt;
}

// Pack the m/k/n shape fields, which always come from operands.
static FailureOr<Value> packMadShapeXt(const MadXtBitPacker &packer,
                                       const MadXtConfig &config, Type i64Ty) {
  Location loc = packer.loc;
  PatternRewriter &rewriter = packer.rewriter;
  Value mI64 = castIntegerLikeTo(loc, config.m, i64Ty, rewriter);
  Value nI64 = castIntegerLikeTo(loc, config.n, i64Ty, rewriter);
  Value kI64 = castIntegerLikeTo(loc, config.k, i64Ty, rewriter);
  if (!mI64 || !nI64 || !kI64) {
    return failure();
  }
  Value xt = packer.bitOr(mI64, packer.shl(kI64, mlir::pto::kValue12));
  return packer.bitOr(xt, packer.shl(nI64, mlir::pto::kValue24));
}

// unit_flag occupies two bits, so it does not fit the single-bit helper: the
// runtime operand carries the frontend's 0/2/3 domain directly, while the
// attribute path maps the mode enum onto the same domain.
static FailureOr<Value> packMadUnitFlagXt(const MadXtBitPacker &packer,
                                          Value xt, const MadXtConfig &config,
                                          Type i64Ty) {
  if (config.unitFlagValue) {
    Value flagI64 = coerceMadFlagToI64(packer.loc, config.unitFlagValue, i64Ty,
                                       packer.rewriter);
    if (!flagI64) {
      return failure();
    }
    flagI64 = truncateMadFlagField<mlir::pto::kValue2>(packer, flagI64);
    return packer.bitOr(xt, packer.shl(flagI64, mlir::pto::kValue55));
  }
  if (!config.unitFlagMode) {
    return xt;
  }
  uint64_t unitFlagCtrl = *config.unitFlagMode == pto::MadUnitFlagMode::CheckOnly
                              ? mlir::pto::kValue2
                              : mlir::pto::kValue3;
  return packer.bitOr(
      xt, packer.shl(packer.constant(unitFlagCtrl), mlir::pto::kValue55));
}

static FailureOr<Value> packMadXt(Location loc, const MadXtConfig &config,
                                  PatternRewriter &rewriter) {
  Type i64Ty = rewriter.getI64Type();
  MadXtBitPacker packer{loc, rewriter};
  FailureOr<Value> xt = packMadShapeXt(packer, config, i64Ty);
  if (failed(xt)) {
    return failure();
  }
  xt = packMadUnitFlagXt(packer, *xt, config, i64Ty);
  if (failed(xt)) {
    return failure();
  }

  const MadXtFlagBit flagBits[] = {
      {config.disableGemvValue, config.disableGemv, mlir::pto::kValue61},
      {config.biasInitValue, config.cmatrixSource, mlir::pto::kValue62},
      {config.accInitValue, config.cmatrixInit, mlir::pto::kValue63},
  };
  for (const MadXtFlagBit &bit : flagBits) {
    xt = packMadFlagBit(packer, *xt, bit, i64Ty);
    if (failed(xt)) {
      return failure();
    }
  }
  return *xt;
}

struct MadCtrlConfig {
  bool isHif8;
  std::optional<pto::Tf32Mode> tf32Mode;
  std::optional<pto::MadSatMode> satMode;
  bool hasNDir;
};

// Statically computed temporary CTRL requirement of one semantic MAD, per the
// ctrl_state_guard contract: bits in controlledBits are overridden (to 1 when
// also in requiredBits, else 0); bits outside controlledBits inherit the entry
// logical CTRL. Unspecified sat_mode leaves bit 48 uncontrolled; HiF8, TF32,
// and n_dir always control their fields.
struct MadCtrlRequirement {
  uint64_t controlledBits;
  uint64_t requiredBits;
};

static MadCtrlRequirement buildMadCtrlRequirement(const MadCtrlConfig &config) {
  uint64_t controlled = 0;
  uint64_t required = 0;
  auto setBit = [&controlled, &required](int bit, bool value) {
    controlled |= (uint64_t(1) << bit);
    if (value) {
      required |= (uint64_t(1) << bit);
    }
  };
  setBit(mlir::pto::kValue45, config.isHif8);
  if (config.tf32Mode) {
    setBit(mlir::pto::kValue46, true);
    setBit(mlir::pto::kValue47,
           *config.tf32Mode == pto::Tf32Mode::RoundAway);
  } else {
    setBit(mlir::pto::kValue46, false);
    setBit(mlir::pto::kValue47, false);
  }
  if (config.satMode) {
    setBit(mlir::pto::kValue48,
           *config.satMode == pto::MadSatMode::NoSat);
  }
  setBit(mlir::pto::kValue51, config.hasNDir);
  return {controlled, required};
}

enum class MadRawKind { Ordinary, OrdinaryBias, Mx, MxBias };

static MadRawKind deriveMadRawKind(pto::MadSemanticOpInterface op) {
  if (op.isMadMxFamily()) {
    return op.hasBiasOperand() ? MadRawKind::MxBias : MadRawKind::Mx;
  }
  return op.hasBiasOperand() ? MadRawKind::OrdinaryBias
                             : MadRawKind::Ordinary;
}

static LogicalResult emitMadRawOp(pto::MadSemanticOpInterface op,
                                  MadRawKind kind, Value xt,
                                  PatternRewriter &rewriter) {
  Location loc = op->getLoc();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  Value dst = op.getDst();
  switch (kind) {
  case MadRawKind::Ordinary:
    rewriter.create<pto::MadRawOp>(loc, lhs, rhs, dst, xt);
    return success();
  case MadRawKind::OrdinaryBias:
    rewriter.create<pto::MadBiasRawOp>(loc, lhs, rhs, dst, op.getBiasOrNull(),
                                       xt);
    return success();
  case MadRawKind::Mx:
    rewriter.create<pto::MadMxRawOp>(loc, lhs, rhs, dst, xt);
    return success();
  case MadRawKind::MxBias:
    rewriter.create<pto::MadMxBiasRawOp>(loc, lhs, rhs, dst,
                                         op.getBiasOrNull(), xt);
    return success();
  }
  return failure();
}

// Collect the compile-time CTRL inputs of one semantic MAD from its attributes
// and operand types.
static MadCtrlConfig readMadCtrlConfig(pto::MadSemanticOpInterface op) {
  MadCtrlConfig config{};
  if (op.supportsTf32Mode()) {
    if (auto tf32ModeAttr =
            dyn_cast_or_null<pto::Tf32ModeAttr>(op.getTf32ModeAttr())) {
      config.tf32Mode = tf32ModeAttr.getValue();
    }
  }
  if (auto satModeAttr =
          dyn_cast_or_null<pto::MadSatModeAttr>(op.getSatModeAttr())) {
    config.satMode = satModeAttr.getValue();
  }
  if (auto lhsPtr = dyn_cast<pto::PtrType>(op.getLhs().getType())) {
    config.isHif8 = pto::isPTOHiFloat8Type(lhsPtr.getElementType());
  }
  config.hasNDir = op.getNDir();
  return config;
}

// Assemble the xt immediate of one semantic MAD, preferring a runtime operand
// over the corresponding attribute for every control flag that has one.
static FailureOr<Value> buildMadXt(pto::MadSemanticOpInterface op,
                                   PatternRewriter &rewriter) {
  std::optional<pto::MadUnitFlagMode> unitFlagMode;
  if (auto unitFlagModeAttr =
          dyn_cast_or_null<pto::MadUnitFlagModeAttr>(op.getUnitFlagModeAttr())) {
    unitFlagMode = unitFlagModeAttr.getValue();
  }
  return packMadXt(
      op->getLoc(),
      {op.getM(), op.getN(), op.getK(), unitFlagMode, op.getDisableGemv(),
       op.initializesAccumulatorWithBias(), op.initializesAccumulatorWithZero(),
       op.getUnitFlagValueOrNull(), op.getAccInitValueOrNull(),
       op.getDisableGemvValueOrNull(), op.getBiasInitValueOrNull()},
      rewriter);
}

static LogicalResult lowerMadSemanticOp(pto::MadSemanticOpInterface op,
                                        PatternRewriter &rewriter) {
  Location loc = op->getLoc();
  FailureOr<Value> xt = buildMadXt(op, rewriter);
  if (failed(xt)) {
    return rewriter.notifyMatchFailure(op, "failed to pack mad xt");
  }

  // Represent the temporary CTRL requirement structurally instead of
  // materializing get_ctrl/bit-update/set_ctrl around the raw op. The CTRL
  // state optimization pass analyzes all guards and emits the minimal set of
  // hardware CTRL accesses.
  MadCtrlRequirement requirement =
      buildMadCtrlRequirement(readMadCtrlConfig(op));
  auto guard = rewriter.create<pto::CtrlStateGuardOp>(
      loc, rewriter.getI64IntegerAttr(
               static_cast<int64_t>(requirement.controlledBits)),
      rewriter.getI64IntegerAttr(
           static_cast<int64_t>(requirement.requiredBits)));
  {
    OpBuilder::InsertionGuard insertionGuard(rewriter);
    rewriter.setInsertionPointToStart(&guard.getBody().emplaceBlock());
    if (failed(emitMadRawOp(op, deriveMadRawKind(op), *xt, rewriter))) {
      return rewriter.notifyMatchFailure(op, "failed to emit mad raw op");
    }
  }
  rewriter.eraseOp(op);
  return success();
}

template <typename SemanticOp>
class ExpandMadSemanticPattern final : public OpRewritePattern<SemanticOp> {
public:
  explicit ExpandMadSemanticPattern(MLIRContext *context)
      : OpRewritePattern<SemanticOp>(context) {}

  LogicalResult matchAndRewrite(SemanticOp op,
                                PatternRewriter &rewriter) const override {
    auto semantic = dyn_cast<pto::MadSemanticOpInterface>(op.getOperation());
    if (!semantic) {
      return failure();
    }
    return lowerMadSemanticOp(semantic, rewriter);
  }
};

} // namespace

namespace mlir::pto::expand_mad {

void populateExpandMadPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<ExpandMadSemanticPattern<pto::MadOp>,
               ExpandMadSemanticPattern<pto::MadAccOp>,
               ExpandMadSemanticPattern<pto::MadBiasOp>,
               ExpandMadSemanticPattern<pto::MadMxOp>,
               ExpandMadSemanticPattern<pto::MadMxAccOp>,
               ExpandMadSemanticPattern<pto::MadMxBiasOp>>(context);
}

} // namespace mlir::pto::expand_mad
