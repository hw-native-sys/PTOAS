// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- ExpandTileOp.cpp ---------------------------------------------------===//
//===----------------------------------------------------------------------===//
//
// Expand tile-level ops (pto.tadd, pto.tsub, ...) by materializing PTODSL
// template libraries in the compiler's host Python interpreter.
//
// The generated template functions use tile_buf parameters. After this pass,
// the Inline pass inlines the template body, and FoldTileBufIntrinsics
// resolves tile_buf_addr / tile_valid_rows / tile_valid_cols.
//
// Workflow per tile op:
//   1. Extract SpecKey from ALL operands' tile_buf types.
//   2. For PTODSL, read candidates attached by InsertTemplateAttributes and
//      select the first candidate still present.
//   3. Ask the in-process TileLib service to build a source module in the same
//      MLIRContext.
//   4. Clone its entry/helper functions into the caller module.
//   5. Replace the original tile op with func.call, passing tile_buf
//      operands directly (no type bridging needed).
//

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/TileLibService.h"
#include "PTO/Transforms/TileOpExpansionUtils.h"
#include "Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

using namespace mlir;

namespace mlir {
namespace pto {
  namespace func = ::mlir::func;

#define GEN_PASS_DEF_EXPANDTILEOP
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

namespace {

constexpr llvm::StringLiteral kCandidatesAttr = "candidates";

// ============================================================================
// OperandTypeInfo: describes one operand for template specialization.
//
// Four kinds of operands:
//   Tile   — from TileBufType.  dtype + shape + memorySpace + config
//            all participate in the specialization key (SpecKey).
//   View   — from MemRefType (lowered PartitionTensorViewType). The element
//            dtype, shape, strides, memory space, and optional explicit layout
//            participate in SpecKey. PTODSL templates compile ViewSpec metadata
//            into helper bodies, so helpers with different view strides must not
//            share one cached specialization.
//   Vector — from builtin VectorType. The element dtype and vector shape
//            participate in SpecKey so helper-side schema filtering can
//            distinguish auxiliary vector operands such as tmrgsort's
//            `excuted : vector<4xi16>`.
//   Scalar — from a scalar element type.  Only dtype participates in SpecKey.
// ============================================================================
enum class OperandKind { Tile, View, Vector, Scalar };

struct OperandTypeInfo {
  OperandKind kind = OperandKind::Tile;
  std::string dtype; // all kinds: element type string (e.g. "f32")

  // --- Tile-only (TileBufType) ---
  SmallVector<int64_t, mlir::pto::kValue2> tileShape;
  SmallVector<int64_t, mlir::pto::kValue2> tileValidShape;
  std::string tileMemorySpace; // e.g. "ub", "gm", "mat", "left", "right", "acc", "bias"
  int32_t blayout = 0;
  int32_t slayout = 0;
  int32_t fractal = 0;
  uint64_t pad = 0;
  int32_t compact = 0;

  // --- View-only (MemRefType) — for JSON / constraint checking only ---
  SmallVector<int64_t> viewShape;
  SmallVector<int64_t> viewStrides;
  std::string viewMemorySpace; // "gm" or "ub"
  std::optional<pto::Layout> viewLayout;

  // --- Vector-only (builtin VectorType) ---
  SmallVector<int64_t> vectorShape;

  // --- Scalar-only ---
  std::optional<int64_t> scalarValue;

  /// Equality for SpecKey caching — only compares fields relevant to each kind.
  bool operator==(const OperandTypeInfo &rhs) const {
    if (kind != rhs.kind || dtype != rhs.dtype) {
      return false;
    }
    if (kind == OperandKind::Tile) {
      return tileShape == rhs.tileShape &&
             tileValidShape == rhs.tileValidShape &&
             tileMemorySpace == rhs.tileMemorySpace &&
             blayout == rhs.blayout && slayout == rhs.slayout &&
             fractal == rhs.fractal && pad == rhs.pad &&
             compact == rhs.compact;
    }
    if (kind == OperandKind::Vector) {
      return vectorShape == rhs.vectorShape;
    }
    if (kind == OperandKind::Scalar) {
      return scalarValue == rhs.scalarValue;
    }
    return viewShape == rhs.viewShape &&
           viewStrides == rhs.viewStrides &&
           viewMemorySpace == rhs.viewMemorySpace &&
           viewLayout == rhs.viewLayout;
  }
};

// ============================================================================
// SpecKey: identifies a specialized template instance using ALL operands.
// ============================================================================
struct SpecKey {
  std::string opName;
  std::string targetArch;
  SmallVector<OperandTypeInfo, mlir::pto::kValue4> operands;
  SmallVector<std::pair<std::string, std::string>, mlir::pto::kValue4> contextAttrs;

  bool operator==(const SpecKey &rhs) const {
    return opName == rhs.opName && targetArch == rhs.targetArch &&
           operands == rhs.operands && contextAttrs == rhs.contextAttrs;
  }
};

struct SpecKeyInfo : public llvm::DenseMapInfo<SpecKey> {
  static inline SpecKey getEmptyKey() { return {"", "", {}, {}}; }
  static inline SpecKey getTombstoneKey() {
    return {"__tombstone__", "", {}, {}};
  }
  static unsigned getHashValue(const SpecKey &key) {
    unsigned h = llvm::hash_combine(key.opName, key.targetArch);
    for (const auto &op : key.operands) {
      h = llvm::hash_combine(h, static_cast<int>(op.kind), op.dtype);
      if (op.kind == OperandKind::Tile) {
        h = llvm::hash_combine(h, op.tileMemorySpace, op.blayout,
                               op.slayout, op.fractal, op.pad, op.compact);
        for (int64_t d : op.tileShape) {
          h = llvm::hash_combine(h, d);
        }
        for (int64_t d : op.tileValidShape) {
          h = llvm::hash_combine(h, d);
        }
      } else if (op.kind == OperandKind::Vector) {
        for (int64_t d : op.vectorShape) {
          h = llvm::hash_combine(h, d);
        }
      } else if (op.kind == OperandKind::Scalar) {
        h = llvm::hash_combine(h, op.scalarValue.has_value());
        if (op.scalarValue) {
          h = llvm::hash_combine(h, *op.scalarValue);
        }
      }
      if (op.kind == OperandKind::View) {
        h = llvm::hash_combine(h, op.viewMemorySpace);
        for (int64_t d : op.viewShape) {
          h = llvm::hash_combine(h, d);
        }
        for (int64_t d : op.viewStrides) {
          h = llvm::hash_combine(h, d);
        }
        h = llvm::hash_combine(h, op.viewLayout.has_value());
        if (op.viewLayout) {
          h = llvm::hash_combine(h, static_cast<int>(*op.viewLayout));
        }
      }
    }
    for (const auto &[attrName, attrValue] : key.contextAttrs) {
      h = llvm::hash_combine(h, attrName, attrValue);
    }
    return h;
  }
  static bool isEqual(const SpecKey &lhs, const SpecKey &rhs) {
    return lhs == rhs;
  }
};
// ============================================================================
// Helpers
// ============================================================================
static std::string getDtypeString(Type elemTy) {
  if (elemTy.isIndex()) {
    return "i32";
  }
  if (elemTy.isInteger(1)) {
    return "i1";
  }
  if (elemTy.isF32()) {
    return "f32";
  }
  if (elemTy.isF16()) {
    return "f16";
  }
  if (elemTy.isBF16()) {
    return "bf16";
  }
  if (isa<Float8E4M3FNType>(elemTy)) {
    return "f8e4m3";
  }
  if (isa<Float8E5M2Type>(elemTy)) {
    return "f8e5m2";
  }
  if (isa<pto::HiF8Type>(elemTy)) {
    return "hif8";
  }
  if (isa<pto::F4E1M2x2Type>(elemTy)) {
    return "f4e1m2x2";
  }
  if (isa<pto::F4E2M1x2Type>(elemTy)) {
    return "f4e2m1x2";
  }
  if (elemTy.isUnsignedInteger(mlir::pto::kValue64)) {
    return "ui64";
  }
  if (elemTy.isUnsignedInteger(mlir::pto::kValue32)) {
    return "ui32";
  }
  if (elemTy.isUnsignedInteger(mlir::pto::kValue16)) {
    return "ui16";
  }
  if (elemTy.isUnsignedInteger(mlir::pto::kValue8)) {
    return "ui8";
  }
  if (elemTy.isSignedInteger(mlir::pto::kValue64)) {
    return "si64";
  }
  if (elemTy.isSignedInteger(mlir::pto::kValue32)) {
    return "si32";
  }
  if (elemTy.isSignedInteger(mlir::pto::kValue16)) {
    return "si16";
  }
  if (elemTy.isSignedInteger(mlir::pto::kValue8)) {
    return "si8";
  }
  if (elemTy.isSignlessInteger(mlir::pto::kValue64)) {
    return "i64";
  }
  if (elemTy.isSignlessInteger(mlir::pto::kValue32)) {
    return "i32";
  }
  if (elemTy.isSignlessInteger(mlir::pto::kValue16)) {
    return "i16";
  }
  if (elemTy.isSignlessInteger(mlir::pto::kValue8)) {
    return "i8";
  }
  return "";
}

// Cast `operand` to `dstTy`, preferring semantically precise ops over the
// generic unrealized cast so later lowering passes don't get stuck.
static Value bridgeOperandToType(OpBuilder &builder, Location loc,
                                 Value operand, Type dstTy) {
  Type srcTy = operand.getType();
  if (srcTy == dstTy) {
    return operand;
  }
  if (srcTy.isIndex() && isa<IntegerType>(dstTy)) {
    return builder.create<arith::IndexCastOp>(loc, dstTy, operand);
  }
  return builder.create<UnrealizedConversionCastOp>(loc, dstTy, operand)
      .getResult(0);
}

static StringRef getTileOpName(Operation *op) {
  return op->getName().stripDialect();
}

static std::string getTargetArchString(Operation *op) {
  if (!op) {
    return "";
  }
  for (ModuleOp current = op->getParentOfType<ModuleOp>(); current;
       current = current->getParentOfType<ModuleOp>()) {
    if (auto targetAttr = current->getAttrOfType<StringAttr>("pto.target_arch")) {
      return targetAttr.getValue().str();
    }
  }
  return "";
}

static std::string stringifyMemorySpace(pto::AddressSpace space) {
  switch (space) {
  case pto::AddressSpace::GM:
    return "gm";
  case pto::AddressSpace::MAT:
    return "mat";
  case pto::AddressSpace::LEFT:
    return "left";
  case pto::AddressSpace::RIGHT:
    return "right";
  case pto::AddressSpace::ACC:
    return "acc";
  case pto::AddressSpace::BIAS:
    return "bias";
  case pto::AddressSpace::SCALING:
    return "scaling";
  case pto::AddressSpace::VEC:
  case pto::AddressSpace::Zero:
    return "ub";
  }
  return "ub";
}

static std::string getMemorySpaceString(pto::TileBufType tbTy) {
  auto msAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(tbTy.getMemorySpace());
  return msAttr ? stringifyMemorySpace(msAttr.getAddressSpace()) : "ub";
}

static std::string getMemorySpaceString(MemRefType mrTy) {
  auto msAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(mrTy.getMemorySpace());
  return msAttr ? stringifyMemorySpace(msAttr.getAddressSpace()) : "gm";
}

static std::string getBLayoutString(int32_t blayout) {
  if (blayout == static_cast<int32_t>(pto::BLayout::ColMajor)) {
    return "col_major";
  }
  return "row_major";
}

static std::string getSLayoutString(int32_t slayout) {
  if (slayout == static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return "row_major";
  }
  if (slayout == static_cast<int32_t>(pto::SLayout::ColMajor)) {
    return "col_major";
  }
  return "none_box";
}

static constexpr llvm::StringLiteral kLayoutAttrName = "layout";

static std::optional<pto::Layout> getLayoutAttrFromOp(Operation *op) {
  if (!op) {
    return std::nullopt;
  }
  if (auto attr = op->getAttrOfType<pto::LayoutAttr>(kLayoutAttrName)) {
    return attr.getLayout();
  }
  return std::nullopt;
}

static std::optional<pto::Layout> resolveViewLayout(Value value) {
  if (!value) {
    return std::nullopt;
  }

  Operation *def = value.getDefiningOp();
  while (def) {
    if (auto layout = getLayoutAttrFromOp(def)) {
      return layout;
    }
    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      value = subview.getSource();
      def = value.getDefiningOp();
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      value = cast.getSource();
      def = value.getDefiningOp();
      continue;
    }
    if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(def)) {
      value = reinterpret.getSource();
      def = value.getDefiningOp();
      continue;
    }
    if (auto partition = dyn_cast<pto::PartitionViewOp>(def)) {
      value = partition.getSource();
      def = value.getDefiningOp();
      continue;
    }
    break;
  }
  return std::nullopt;
}

static std::optional<std::string> getViewLayoutString(std::optional<pto::Layout> layout) {
  if (!layout) {
    return std::nullopt;
  }
  return stringifyLayout(*layout).str();
}

static std::optional<std::string> getTCvtRoundModeString(pto::TCvtOp op) {
  switch (op.getRmode()) {
  case pto::RoundMode::NONE:
  case pto::RoundMode::RINT:
  case pto::RoundMode::CAST_RINT:
    return "RINT";
  case pto::RoundMode::ROUND:
    return "ROUND";
  case pto::RoundMode::FLOOR:
    return "FLOOR";
  case pto::RoundMode::CEIL:
    return "CEIL";
  case pto::RoundMode::TRUNC:
    return "TRUNC";
  case pto::RoundMode::ODD:
    return "ODD";
  }
  return std::nullopt;
}

static StringRef getPrecisionTypeString(pto::DivPrecision precision) {
  switch (precision) {
  case pto::DivPrecision::Default:
    return "default";
  case pto::DivPrecision::HighPrecision:
    return "high_precision";
  }
  llvm_unreachable("unknown DivPrecision");
}

static StringRef getPrecisionTypeString(pto::ExpPrecision precision) {
  switch (precision) {
  case pto::ExpPrecision::Default:
    return "default";
  case pto::ExpPrecision::HighPrecision:
    return "high_precision";
  }
  llvm_unreachable("unknown ExpPrecision");
}

static StringRef getPrecisionTypeString(pto::LogPrecision precision) {
  switch (precision) {
  case pto::LogPrecision::Default:
    return "default";
  case pto::LogPrecision::HighPrecision:
    return "high_precision";
  }
  llvm_unreachable("unknown LogPrecision");
}

static StringRef getPrecisionTypeString(pto::RecipPrecision precision) {
  switch (precision) {
  case pto::RecipPrecision::Default:
    return "default";
  case pto::RecipPrecision::HighPrecision:
    return "high_precision";
  }
  llvm_unreachable("unknown RecipPrecision");
}

static StringRef getPrecisionTypeString(pto::RsqrtPrecision precision) {
  switch (precision) {
  case pto::RsqrtPrecision::Default:
    return "default";
  case pto::RsqrtPrecision::HighPrecision:
    return "high_precision";
  }
  llvm_unreachable("unknown RsqrtPrecision");
}

static StringRef getPrecisionTypeString(pto::SqrtPrecision precision) {
  switch (precision) {
  case pto::SqrtPrecision::Default:
    return "default";
  case pto::SqrtPrecision::HighPrecision:
    return "high_precision";
  }
  llvm_unreachable("unknown SqrtPrecision");
}

// MUST stay in sync with template behavior. Adding an op here without a real
// high_precision code path would silence the warning while preserving default
// behavior.
static const llvm::StringSet<> &highPrecisionImplementedOps() {
  static const llvm::StringSet<> kImplementedOps{
    "pto.tlog",
    "pto.tdiv",
    "pto.tdivs",
    "pto.trecip",
    "pto.trowexpanddiv",
    "pto.tcolexpanddiv",
    "pto.texp",
    "pto.tsqrt",
  };
  return kImplementedOps;
}

template <typename OpT, typename PrecisionT>
static bool tryAppendPrecisionType(
    Operation *op,
    SmallVectorImpl<std::pair<std::string, std::string>> &attrs,
    PrecisionT highPrecision) {
  auto typed = dyn_cast<OpT>(op);
  if (!typed) {
    return false;
  }

  PrecisionT precision = typed.getPrecisionType();
  attrs.emplace_back("precisionType", getPrecisionTypeString(precision).str());

  if (precision == highPrecision &&
      !highPrecisionImplementedOps().contains(op->getName().getStringRef())) {
    StringRef opName = op->getName().getStringRef();
    llvm::errs() << "warning: '" << opName << "' op " << opName
                 << ": precisionType = high_precision requested but not yet "
                    "implemented; falling back to default behavior\n";
  }
  return true;
}

static std::string getTRandomRoundsString(pto::TRandomOp op) {
  return std::to_string(op.getRounds());
}

static LogicalResult appendOpContextAttrs(
    Operation *op,
    SmallVectorImpl<std::pair<std::string, std::string>> &attrs) {
  if (auto tcvt = dyn_cast<pto::TCvtOp>(op)) {
    std::optional<std::string> roundMode = getTCvtRoundModeString(tcvt);
    if (roundMode) {
      attrs.emplace_back("round_mode", *roundMode);
    }
  }
  if (auto trandom = dyn_cast<pto::TRandomOp>(op)) {
    attrs.emplace_back("rounds", getTRandomRoundsString(trandom));
  }
  if (auto tcmp = dyn_cast<pto::TCmpOp>(op)) {
    if (auto cmpModeAttr = tcmp.getCmpModeAttr()) {
      attrs.emplace_back("cmp_mode",
                         stringifyCmpMode(cmpModeAttr.getValue()).str());
    }
  }
  if (auto tcmps = dyn_cast<pto::TCmpSOp>(op)) {
    if (auto cmpModeAttr = tcmps.getCmpModeAttr()) {
      attrs.emplace_back("cmp_mode",
                         stringifyCmpMode(cmpModeAttr.getValue()).str());
    }
  }
  if (auto tinsert = dyn_cast<pto::TInsertOp>(op)) {
    if (auto modeAttr = tinsert.getAccToVecModeAttr()) {
      attrs.emplace_back("acc_to_vec_mode",
                         stringifyAccToVecMode(modeAttr.getValue()).str());
    }
    attrs.emplace_back("relu_pre_mode",
                       stringifyReluPreMode(tinsert.getReluPreMode()).str());
  }
  if (auto tgather = dyn_cast<pto::TGatherOp>(op)) {
    if (auto maskPatternAttr = tgather.getMaskPatternAttr()) {
      attrs.emplace_back(
          "mask_pattern",
          stringifyMaskPattern(maskPatternAttr.getValue()).str());
    }
    if (auto axisAttr = tgather.getAxisAttr()) {
      attrs.emplace_back("axis_value", axisAttr.getValue().str());
    }
  }
  if (auto ttri = dyn_cast<pto::TTriOp>(op)) {
    attrs.emplace_back("upper_or_lower", std::to_string(ttri.getUpperOrLower()));
  }
  if (auto thistogram = dyn_cast<pto::THistogramOp>(op)) {
    int byte = 1;
    if (auto byteAttr = thistogram.getByteAttr()) {
      byte = byteAttr.getInt();
    }
    attrs.emplace_back("byte", std::to_string(byte));
  }
  if (auto tci = dyn_cast<pto::TCIOp>(op)) {
    attrs.emplace_back("descending", tci.getDescending() ? "true" : "false");
  }
  if (auto tfillpad = dyn_cast<pto::TFillPadOp>(op)) {
    auto kind = pto::inferTFillPadLoweringKindAfterMemoryPlanning(tfillpad);
    if (failed(kind)) {
      return tfillpad.emitOpError(
          "cannot infer a supported lowering; expand and in-place forms "
          "require loc=vec, statically comparable physical shapes, and "
          "resolved planned addresses");
    }
    StringRef token;
    switch (*kind) {
    case pto::TFillPadLoweringKind::Normal:
      token = "normal";
      break;
    case pto::TFillPadLoweringKind::InPlace:
      token = "in_place";
      break;
    case pto::TFillPadLoweringKind::Expand:
      token = "expand";
      break;
    }
    attrs.emplace_back("lowering_kind", token.str());
  }
  if (auto tscatter = dyn_cast<pto::TScatterOp>(op)) {
    if (auto maskPatternAttr = tscatter.getMaskPatternAttr()) {
      attrs.emplace_back(
          "mask_pattern",
          stringifyMaskPattern(maskPatternAttr.getValue()).str());
    }
    if (auto axisAttr = tscatter.getAxisAttr()) {
      attrs.emplace_back("axis_value", axisAttr.getValue().str());
    }
  }
  (void)(tryAppendPrecisionType<pto::TExpOp>(
             op, attrs, pto::ExpPrecision::HighPrecision) ||
         tryAppendPrecisionType<pto::TLogOp>(
             op, attrs, pto::LogPrecision::HighPrecision) ||
         tryAppendPrecisionType<pto::TSqrtOp>(
             op, attrs, pto::SqrtPrecision::HighPrecision) ||
         tryAppendPrecisionType<pto::TRecipOp>(
             op, attrs, pto::RecipPrecision::HighPrecision) ||
         tryAppendPrecisionType<pto::TRsqrtOp>(
             op, attrs, pto::RsqrtPrecision::HighPrecision) ||
         tryAppendPrecisionType<pto::TDivOp>(
             op, attrs, pto::DivPrecision::HighPrecision) ||
         tryAppendPrecisionType<pto::TDivSOp>(
             op, attrs, pto::DivPrecision::HighPrecision) ||
         tryAppendPrecisionType<pto::TRowExpandDivOp>(
             op, attrs, pto::DivPrecision::HighPrecision) ||
         tryAppendPrecisionType<pto::TColExpandDivOp>(
             op, attrs, pto::DivPrecision::HighPrecision));
  return success();
}

static bool getStaticIntFromValue(Value value, int64_t &out) {
  if (auto cOp = value.getDefiningOp<arith::ConstantIndexOp>()) {
    out = cOp.value();
    return true;
  }
  if (auto cInt = value.getDefiningOp<arith::ConstantIntOp>()) {
    out = cInt.value();
    return true;
  }
  return false;
}

static int64_t getStaticIntOrDynamic(OpFoldResult ofr) {
  if (isa<Attribute>(ofr)) {
    Attribute attr = cast<Attribute>(ofr);
    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      return intAttr.getInt();
    }
    return ShapedType::kDynamic;
  }
  Value value = cast<Value>(ofr);
  int64_t result = ShapedType::kDynamic;
  if (getStaticIntFromValue(value, result)) {
    return result;
  }
  return ShapedType::kDynamic;
}

static void recordStaticSizes(ArrayRef<OpFoldResult> inputs,
                              SmallVectorImpl<int64_t> &out) {
  out.clear();
  out.reserve(inputs.size());
  for (OpFoldResult ofr : inputs) {
    out.push_back(getStaticIntOrDynamic(ofr));
  }
}

static SmallVector<int64_t> combineSubviewStrides(ArrayRef<int64_t> baseStrides,
                                                  ArrayRef<OpFoldResult> steps) {
  SmallVector<int64_t> result;
  result.reserve(baseStrides.size());
  for (auto [baseStride, step] : llvm::zip(baseStrides, steps)) {
    int64_t stepValue = getStaticIntOrDynamic(step);
    if (baseStride == ShapedType::kDynamic ||
        stepValue == ShapedType::kDynamic) {
      result.push_back(ShapedType::kDynamic);
      continue;
    }
    result.push_back(baseStride * stepValue);
  }
  return result;
}

static void populateViewShapeAndStrides(Value value,
                                        SmallVectorImpl<int64_t> &shape,
                                        SmallVectorImpl<int64_t> &strides) {
  if (!value) {
    return;
  }

  if (auto partition = value.getDefiningOp<pto::PartitionViewOp>()) {
    populateViewShapeAndStrides(partition.getSource(), shape, strides);
    SmallVector<int64_t> partitionShape;
    partitionShape.reserve(partition.getSizes().size());
    for (Value size : partition.getSizes()) {
      int64_t staticSize = ShapedType::kDynamic;
      (void)getStaticIntFromValue(size, staticSize);
      partitionShape.push_back(staticSize);
    }
    shape = std::move(partitionShape);
    return;
  }

  if (auto makeView = value.getDefiningOp<pto::MakeTensorViewOp>()) {
    shape.clear();
    strides.clear();
    for (Value dim : makeView.getShape()) {
      int64_t staticDim = ShapedType::kDynamic;
      (void)getStaticIntFromValue(dim, staticDim);
      shape.push_back(staticDim);
    }
    for (Value stride : makeView.getStrides()) {
      int64_t staticStride = ShapedType::kDynamic;
      (void)getStaticIntFromValue(stride, staticStride);
      strides.push_back(staticStride);
    }
    return;
  }

  if (auto subview = value.getDefiningOp<memref::SubViewOp>()) {
    populateViewShapeAndStrides(subview.getSource(), shape, strides);
    SmallVector<int64_t> subviewShape;
    recordStaticSizes(subview.getMixedSizes(), subviewShape);
    if (!subviewShape.empty()) {
      shape = subviewShape;
    }
    if (!strides.empty()) {
      strides = combineSubviewStrides(strides, subview.getMixedStrides());
    }
    return;
  }

  if (auto reinterpret = value.getDefiningOp<memref::ReinterpretCastOp>()) {
    if (shape.empty()) {
      SmallVector<int64_t> reinterpretShape;
      recordStaticSizes(reinterpret.getMixedSizes(), reinterpretShape);
      if (!reinterpretShape.empty()) {
        shape = reinterpretShape;
      }
    }
    if (strides.empty()) {
      recordStaticSizes(reinterpret.getMixedStrides(), strides);
    }
    return;
  }

  if (auto cast = value.getDefiningOp<memref::CastOp>()) {
    populateViewShapeAndStrides(cast.getSource(), shape, strides);
    return;
  }

  if (auto memrefTy = dyn_cast<MemRefType>(value.getType())) {
    if (shape.empty()) {
      shape.assign(memrefTy.getShape().begin(), memrefTy.getShape().end());
    }
    if (strides.empty()) {
      int64_t offset = ShapedType::kDynamic;
      if (succeeded(
              mlir::pto::getPTOMemRefStridesAndOffset(memrefTy, strides, offset))) {
        // strides populated — dynamic dims remain ShapedType::kDynamic.
      }
    }
  }
}

static std::optional<OperandTypeInfo> buildOperandTypeInfo(Value value) {
  Type ty = value.getType();
  // Tile operand — from TileBufType.
  if (auto tbTy = dyn_cast<pto::TileBufType>(ty)) {
    OperandTypeInfo info;
    info.kind = OperandKind::Tile;
    info.dtype = getDtypeString(tbTy.getElementType());
    if (info.dtype.empty()) {
      return std::nullopt;
    }
    info.tileShape.assign(tbTy.getShape().begin(), tbTy.getShape().end());
    auto validShape = tbTy.getValidShape();
    if (validShape.empty()) {
      info.tileValidShape.assign(tbTy.getShape().begin(), tbTy.getShape().end());
    }
    else {
      info.tileValidShape.assign(validShape.begin(), validShape.end());
}
    info.tileMemorySpace = getMemorySpaceString(tbTy);
    if (auto config = tbTy.getConfigAttr()) {
      info.blayout = static_cast<int32_t>(config.getBLayout().getValue());
      info.slayout = static_cast<int32_t>(config.getSLayout().getValue());
      info.fractal = config.getSFractalSize()
                         ? static_cast<int32_t>(config.getSFractalSize().getInt())
                         : 0;
      info.pad = static_cast<uint64_t>(config.getPad().getValue());
      info.compact =
          static_cast<int32_t>(config.getCompactMode().getValue());
    }
    return info;
  }

  // View operand — from MemRefType (lowered PartitionTensorViewType).
  if (auto mrTy = dyn_cast<MemRefType>(ty)) {
    OperandTypeInfo info;
    info.kind = OperandKind::View;
    info.dtype = getDtypeString(mrTy.getElementType());
    if (info.dtype.empty()) {
      return std::nullopt;
    }
    info.viewMemorySpace = getMemorySpaceString(mrTy);
    info.viewLayout = resolveViewLayout(value);
    populateViewShapeAndStrides(value, info.viewShape, info.viewStrides);
    if (info.viewShape.empty()) {
      info.viewShape.assign(mrTy.getShape().begin(), mrTy.getShape().end());
    }
    if (info.viewStrides.empty()) {
      int64_t offset = ShapedType::kDynamic;
      if (succeeded(mlir::pto::getPTOMemRefStridesAndOffset(
              mrTy, info.viewStrides, offset))) {
        // strides populated — dynamic dims remain ShapedType::kDynamic.
      }
    }
    return info;
  }

  if (auto viewTy = dyn_cast<pto::PartitionTensorViewType>(ty)) {
    OperandTypeInfo info;
    info.kind = OperandKind::View;
    info.dtype = getDtypeString(viewTy.getElementType());
    if (info.dtype.empty()) {
      return std::nullopt;
    }
    info.viewMemorySpace = "gm";
    info.viewLayout = resolveViewLayout(value);
    populateViewShapeAndStrides(value, info.viewShape, info.viewStrides);
    if (info.viewShape.empty()) {
      info.viewShape.assign(viewTy.getShape().begin(), viewTy.getShape().end());
    }
    if (info.viewStrides.empty()) {
      info.viewStrides.assign(viewTy.getRank(), ShapedType::kDynamic);
    }
    return info;
  }

  // Auxiliary vector operand — from builtin VectorType (e.g. vector<4xi16>).
  if (auto vecTy = dyn_cast<VectorType>(ty)) {
    OperandTypeInfo info;
    info.kind = OperandKind::Vector;
    info.dtype = getDtypeString(vecTy.getElementType());
    if (info.dtype.empty()) {
      return std::nullopt;
    }
    info.vectorShape.assign(vecTy.getShape().begin(), vecTy.getShape().end());
    return info;
  }

  // Scalar operand — from a scalar element type.
  OperandTypeInfo info;
  info.kind = OperandKind::Scalar;
  info.dtype = getDtypeString(ty);
  if (info.dtype.empty()) {
    return std::nullopt;
  }
  int64_t scalarValue = 0;
  if (getStaticIntFromValue(value, scalarValue)) {
    info.scalarValue = scalarValue;
  }
  return info;
}

static FailureOr<SpecKey> buildSpecKey(Operation *op) {
  SpecKey key;
  key.opName = getTileOpName(op).str();
  key.targetArch = getTargetArchString(op);

  for (unsigned i = 0; i < op->getNumOperands(); ++i) {
    auto info = buildOperandTypeInfo(op->getOperand(i));
    if (!info) {
      op->emitError("ExpandTileOp: cannot build specialization key for this "
                    "operand schema");
      return failure();
    }
    key.operands.push_back(*info);
  }
  if (key.operands.empty()) {
    op->emitError(
        "ExpandTileOp: cannot build a specialization key without operands");
    return failure();
  }

  if (failed(appendOpContextAttrs(op, key.contextAttrs))) {
    return failure();
  }
  return key;
}

// ============================================================================
// ExpandState: runtime state for a single pass invocation.
// ============================================================================
struct ExpandState {
  std::shared_ptr<pto::TileLibService> tileLibService;

  func::FuncOp invokeTileLib(const SpecKey &key, Operation *tileOp,
                             ModuleOp mod, MLIRContext *ctx);
  func::FuncOp invokeInProcessTileLib(const SpecKey &key,
                                      StringRef candidateId,
                                      const std::string &uniqueName,
                                      ModuleOp mod, MLIRContext *ctx);

  LogicalResult expandTileOpsInFunction(func::FuncOp func, ModuleOp mod,
                                        MLIRContext *ctx);
};

// ============================================================================
// The Pass
// ============================================================================
struct ExpandTileOpPass
    : public mlir::pto::impl::ExpandTileOpBase<ExpandTileOpPass> {
  using ExpandTileOpBase::ExpandTileOpBase;

  void runOnOperation() override;
};

/// Serialize a JSON array of integers.
static void appendJsonIntArray(std::string &json, ArrayRef<int64_t> arr) {
  json += "[";
  for (size_t i = 0; i < arr.size(); ++i) {
    if (i > 0) {
      json += ",";
    }
    json += std::to_string(arr[i]);
  }
  json += "]";
}

/// Serialize a JSON array where dynamic dimensions become `null`.
static void appendJsonDimArray(std::string &json, ArrayRef<int64_t> arr,
                               bool negativeIsDynamic = false) {
  json += "[";
  for (size_t i = 0; i < arr.size(); ++i) {
    if (i > 0) {
      json += ",";
    }
    int64_t dim = arr[i];
    if (ShapedType::isDynamic(dim) || (negativeIsDynamic && dim < 0)) {
      json += "null";
      continue;
    }
    json += std::to_string(dim);
  }
  json += "]";
}

static std::string buildOperandSpecsJson(const SpecKey &key) {
  std::string json = "[";
  for (size_t i = 0; i < key.operands.size(); ++i) {
    const auto &op = key.operands[i];
    if (i > 0) {
      json += ",";
    }

    if (op.kind == OperandKind::Tile) {
      json += "{\"kind\":\"tile\",\"dtype\":\"" + op.dtype + "\",\"shape\":";
      appendJsonIntArray(json, op.tileShape);
      json += ",\"valid_shape\":";
      appendJsonDimArray(json, op.tileValidShape, /*negativeIsDynamic=*/true);
      json += ",\"memory_space\":\"";
      json += op.tileMemorySpace;
      json += "\",\"config\":{";
      json += "\"b_layout\":\"";
      json += getBLayoutString(op.blayout);
      json += "\",\"s_layout\":\"";
      json += getSLayoutString(op.slayout);
      json += "\",\"s_fractal_size\":";
      json += std::to_string(op.fractal);
      json += ",\"pad_value\":\"0x";
      json += llvm::utohexstr(op.pad, /*LowerCase=*/false);
      json += "\",\"compact_mode\":";
      json += std::to_string(op.compact);
      json += "}}";
      continue;
    }

    if (op.kind == OperandKind::View) {
      json += "{\"kind\":\"view\",\"dtype\":\"" + op.dtype + "\",\"shape\":";
      appendJsonDimArray(json, op.viewShape);
      if (!op.viewStrides.empty()) {
        json += ",\"strides\":[";
        for (size_t dim = 0; dim < op.viewStrides.size(); ++dim) {
          if (dim > 0) {
            json += ",";
          }
          if (ShapedType::isDynamic(op.viewStrides[dim])) {
            json += "null";
          }
          else {
            json += std::to_string(op.viewStrides[dim]);
}
        }
        json += "]";
      }
      json += ",\"memory_space\":\"" + op.viewMemorySpace + "\"";
      if (auto layout = getViewLayoutString(op.viewLayout)) {
        json += ",\"config\":{\"layout\":\"";
        json += *layout;
        json += "\"}";
      }
      json += "}";
      continue;
    }

    if (op.kind == OperandKind::Vector) {
      json += "{\"kind\":\"vector\",\"dtype\":\"" + op.dtype + "\",\"shape\":";
      appendJsonIntArray(json, op.vectorShape);
      json += "}";
      continue;
    }

    // Scalar
    json += "{\"kind\":\"scalar\",\"dtype\":\"" + op.dtype + "\"";
    if (op.scalarValue) {
      json += ",\"value\":";
      json += std::to_string(*op.scalarValue);
    }
    json += "}";
  }
  json += "]";
  return json;
}

static std::string dimSuffix(int64_t dim) {
  if (ShapedType::isDynamic(dim)) {
    return "d";
  }
  return std::to_string(dim);
}

static std::string buildUniqueFunctionBaseName(const SpecKey &key) {
  std::string uniqueName = "__pto_tilelang_" + key.targetArch + "_" + key.opName;
  for (const auto &op : key.operands) {
    uniqueName += op.kind == OperandKind::Tile   ? "_tile"
                 : op.kind == OperandKind::View ? "_view"
                 : op.kind == OperandKind::Vector ? "_vector"
                                                  : "_scalar";
    uniqueName += "_" + op.dtype;
    if (op.kind == OperandKind::Tile) {
      for (int64_t d : op.tileShape) {
        uniqueName += "_" + std::to_string(d);
      }
      for (int64_t d : op.tileValidShape) {
        uniqueName += "_v" + std::to_string(d);
      }
      uniqueName += "_bl" + std::to_string(op.blayout);
      uniqueName += "_sl" + std::to_string(op.slayout);
      uniqueName += "_fr" + std::to_string(op.fractal);
      uniqueName += "_pd" + llvm::utohexstr(op.pad, /*LowerCase=*/false);
      uniqueName += "_cm" + std::to_string(op.compact);
    } else if (op.kind == OperandKind::View) {
      uniqueName += "_ms_" + op.viewMemorySpace;
      uniqueName += "_shape";
      for (int64_t d : op.viewShape) {
        uniqueName += "_" + dimSuffix(d);
      }
      uniqueName += "_strides";
      for (int64_t d : op.viewStrides) {
        uniqueName += "_" + dimSuffix(d);
      }
      if (op.viewLayout) {
        uniqueName += "_vl_" + stringifyLayout(*op.viewLayout).str();
      }
    } else if (op.kind == OperandKind::Vector) {
      for (int64_t d : op.vectorShape) {
        uniqueName += "_" + std::to_string(d);
      }
    } else if (op.kind == OperandKind::Scalar && op.scalarValue) {
      uniqueName += "_sv" + std::to_string(*op.scalarValue);
    }
  }
  for (const auto &[attrName, attrValue] : key.contextAttrs) {
    uniqueName += "_ctx_" + attrName + "_" + attrValue;
  }
  return uniqueName;
}

static std::string buildUniqueFunctionName(const SpecKey &key,
                                           StringRef candidateId) {
  std::string uniqueName = buildUniqueFunctionBaseName(key);
  if (!candidateId.empty()) {
    uniqueName += "__" + candidateId.str();
  }
  return uniqueName;
}

static std::string buildContextAttrsJson(const SpecKey &key) {
  std::string json = "{";
  for (size_t i = 0; i < key.contextAttrs.size(); ++i) {
    const auto &[attrName, attrValue] = key.contextAttrs[i];
    if (i > 0) {
      json += ",";
    }
    json += "\"";
    json += attrName;
    json += "\":\"";
    json += attrValue;
    json += "\"";
  }
  json += "}";
  return json;
}

// ============================================================================
// Materialize PTODSL in the host Python interpreter and import its functions.
// The service borrows the source module only for the synchronous callback;
// this pass clones the required functions into the caller module there.
// ============================================================================
func::FuncOp ExpandState::invokeInProcessTileLib(const SpecKey &key,
                                                 StringRef candidateId,
                                                 const std::string &uniqueName,
                                                 ModuleOp mod,
                                                 MLIRContext *ctx) {
  if (!tileLibService) {
    return nullptr;
  }

  pto::TileLibMaterializationRequest request;
  request.target = key.targetArch;
  request.op = "pto." + key.opName;
  request.operandSpecsJson = buildOperandSpecsJson(key);
  request.contextAttrsJson = buildContextAttrsJson(key);
  request.candidateId = candidateId.str();

  func::FuncOp importedEntry;
  LogicalResult materializationResult = tileLibService->materialize(
      request, *ctx, [&](ModuleOp sourceModule, StringRef entrySymbol) {
    if (!sourceModule || sourceModule.getContext() != ctx) {
      llvm::errs() << "ExpandTileOp: in-process PTODSL returned a module from "
                      "a different MLIRContext\n";
      return failure();
    }

    auto sourceEntry = sourceModule.lookupSymbol<func::FuncOp>(entrySymbol);
    if (!sourceEntry) {
      llvm::errs() << "ExpandTileOp: in-process PTODSL entry symbol @"
                   << entrySymbol << " was not found\n";
      return failure();
    }

    SmallVector<func::FuncOp, 4> sourceFuncs;
    for (func::FuncOp fn : sourceModule.getOps<func::FuncOp>()) {
      sourceFuncs.push_back(fn);
    }
    if (sourceFuncs.empty()) {
      llvm::errs() << "ExpandTileOp: in-process PTODSL returned no func.func\n";
      return failure();
    }

    SymbolTable targetSymTable(mod);
    llvm::StringMap<std::string> plannedSymbols;
    for (func::FuncOp fn : sourceFuncs) {
      std::string newName = fn == sourceEntry
                                ? uniqueName
                                : uniqueName + "__" + std::string(fn.getSymName());
      if (targetSymTable.lookup(newName)) {
        llvm::errs() << "ExpandTileOp: imported PTODSL symbol collision at @"
                     << newName << "\n";
        return failure();
      }
      plannedSymbols[fn.getSymName()] = std::move(newName);
    }

    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(mod.getBody());
    SmallVector<func::FuncOp, 4> clonedFuncs;
    for (func::FuncOp fn : sourceFuncs) {
      IRMapping mapping;
      auto cloned = cast<func::FuncOp>(builder.clone(*fn, mapping));
      cloned.setName(plannedSymbols.lookup(fn.getSymName()));
      cloned.setVisibility(SymbolTable::Visibility::Private);
      clonedFuncs.push_back(cloned);
    }

    for (func::FuncOp fn : clonedFuncs) {
      for (const auto &renamed : plannedSymbols) {
        if (failed(SymbolTable::replaceAllSymbolUses(
                StringAttr::get(ctx, renamed.getKey()),
                StringAttr::get(ctx, renamed.getValue()), fn))) {
          llvm::errs() << "ExpandTileOp: failed to rewrite imported symbol @"
                       << renamed.getKey() << " in @" << fn.getSymName()
                       << "\n";
          for (func::FuncOp imported : clonedFuncs) {
            imported.erase();
          }
          return failure();
        }
      }
    }

    importedEntry = mod.lookupSymbol<func::FuncOp>(uniqueName);
    if (!importedEntry) {
      llvm::errs() << "ExpandTileOp: failed to import PTODSL entry @"
                   << entrySymbol << "\n";
      return failure();
    }
    if (!importedEntry->hasAttr("pto.tilelang.instance")) {
      llvm::errs() << "ExpandTileOp: warning: in-process PTODSL entry @"
                   << importedEntry.getSymName()
                   << " missing pto.tilelang.instance attribute\n";
    }
    return success();
  });
  if (failed(materializationResult)) {
    llvm::errs() << "ExpandTileOp: in-process PTODSL materialization failed\n";
    return nullptr;
  }
  return importedEntry;
}

// ============================================================================
// Invoke the selected TileLib backend to generate a specialized template.
// ============================================================================
func::FuncOp ExpandState::invokeTileLib(const SpecKey &key,
                                        Operation *tileOp, ModuleOp mod,
                                        MLIRContext *ctx) {
  if (!tileLibService) {
    tileOp->emitError(
        "ExpandTileOp PTODSL backend requires an in-process service");
    return nullptr;
  }

  auto candidates = tileOp->getAttrOfType<ArrayAttr>(kCandidatesAttr);
  if (!candidates || candidates.empty()) {
    tileOp->emitError("ExpandTileOp requires at least one template candidate");
    return nullptr;
  }

  auto selected = dyn_cast<DictionaryAttr>(candidates[0]);
  if (!selected) {
    tileOp->emitError("ExpandTileOp candidate 0 must be a dictionary");
    return nullptr;
  }
  auto selectedName = selected.getAs<StringAttr>("name");
  if (!selectedName) {
    tileOp->emitError("ExpandTileOp candidate 0 requires a string name");
    return nullptr;
  }

  std::string uniqueName =
      buildUniqueFunctionName(key, selectedName.getValue());
  if (auto existing = mod.lookupSymbol<func::FuncOp>(uniqueName)) {
    return existing;
  }

  return invokeInProcessTileLib(key, selectedName.getValue(), uniqueName, mod,
                                ctx);
}

// ============================================================================
// Expand tile ops in a single function.
// ============================================================================
LogicalResult ExpandState::expandTileOpsInFunction(func::FuncOp func,
                                                   ModuleOp mod,
                                                   MLIRContext *ctx) {
  OpBuilder builder(ctx);

  // Collect tile ops first (avoid modifying while iterating).
  SmallVector<Operation *, mlir::pto::kValue16> tileOps;
  func.walk([&](Operation *op) {
    if (pto::isTileLibExpandableOp(op)) {
      tileOps.push_back(op);
    }
  });

  for (auto *op : tileOps) {
    auto specKey = buildSpecKey(op);
    if (failed(specKey)) {
      return failure();
    }

    // Materialize the selected PTODSL template in-process.
    func::FuncOp dslFn = invokeTileLib(*specKey, op, mod, ctx);
    if (!dslFn) {
      StringRef opName = getTileOpName(op);
      op->emitError("ExpandTileOp: failed to instantiate TileLib template for " +
                    opName);
      return failure();
    }

    // Replace tile op with func.call.  For view operands whose caller type
    // (memref) differs from the template parameter type (tensor_view /
    // partition_tensor_view), insert an unrealized_conversion_cast bridge.
    // FoldTileBufIntrinsics will later resolve these casts.
    builder.setInsertionPoint(op);
    SmallVector<Value> operands;
    auto fnArgTypes = dslFn.getArgumentTypes();
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      Value operand = op->getOperand(i);
      if (i < fnArgTypes.size() && operand.getType() != fnArgTypes[i]) {
        operand = bridgeOperandToType(builder, op->getLoc(), operand,
                                      fnArgTypes[i]);
      }
      operands.push_back(operand);
    }
    builder.create<func::CallOp>(op->getLoc(), dslFn, operands);
    op->erase();
  }

  return success();
}

// ============================================================================
// Main entry point.
// ============================================================================
void ExpandTileOpPass::runOnOperation() {
  ModuleOp mod = getOperation();
  MLIRContext *ctx = &getContext();

  bool hasExpandableOps = false;
  mod.walk([&](Operation *op) {
    if (pto::isTileLibExpandableOp(op)) {
      hasExpandableOps = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!hasExpandableOps) {
    return;
  }

  std::shared_ptr<pto::TileLibService> tileLibService =
      pto::TileLibRuntime::getService();
  if (!tileLibService) {
    mod.emitError("ExpandTileOp requires an initialized PTODSL runtime");
    signalPassFailure();
    return;
  }

  ExpandState state;
  state.tileLibService = tileLibService;

  for (auto func : mod.getOps<func::FuncOp>()) {
    if (func.isExternal()) {
      continue;
    }
    if (failed(state.expandTileOpsInFunction(func, mod, ctx))) {
      return signalPassFailure();
    }
  }
}

} // namespace

namespace mlir {
namespace pto {

std::unique_ptr<Pass> createExpandTileOpPass() {
  return std::make_unique<ExpandTileOpPass>();
}

} // namespace pto
} // namespace mlir
