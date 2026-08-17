// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/TileOpExpansionUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <vector>

using namespace mlir;

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_INSERTTEMPLATEATTRIBUTES
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

namespace {

constexpr llvm::StringLiteral kCandidatesAttr = "candidates";

struct CandidateMetadata {
  int64_t id;
  std::string name;
  int64_t priority;
  int64_t loopDepth;
  bool postUpdate;
  bool tail;
};

static std::string getDtypeString(Type elementType) {
  if (elementType.isIndex()) {
    return "i32";
  }
  if (elementType.isInteger(1)) {
    return "i1";
  }
  if (elementType.isF32()) {
    return "f32";
  }
  if (elementType.isF16()) {
    return "f16";
  }
  if (elementType.isBF16()) {
    return "bf16";
  }
  if (isa<Float8E4M3FNType>(elementType)) {
    return "f8e4m3";
  }
  if (isa<Float8E5M2Type>(elementType)) {
    return "f8e5m2";
  }
  if (isa<pto::HiF8Type>(elementType)) {
    return "hif8";
  }
  if (isa<pto::F4E1M2x2Type>(elementType)) {
    return "f4e1m2x2";
  }
  if (isa<pto::F4E2M1x2Type>(elementType)) {
    return "f4e2m1x2";
  }
  if (elementType.isUnsignedInteger(mlir::pto::kValue64)) {
    return "ui64";
  }
  if (elementType.isUnsignedInteger(mlir::pto::kValue32)) {
    return "ui32";
  }
  if (elementType.isUnsignedInteger(mlir::pto::kValue16)) {
    return "ui16";
  }
  if (elementType.isUnsignedInteger(mlir::pto::kValue8)) {
    return "ui8";
  }
  if (elementType.isSignedInteger(mlir::pto::kValue64)) {
    return "si64";
  }
  if (elementType.isSignedInteger(mlir::pto::kValue32)) {
    return "si32";
  }
  if (elementType.isSignedInteger(mlir::pto::kValue16)) {
    return "si16";
  }
  if (elementType.isSignedInteger(mlir::pto::kValue8)) {
    return "si8";
  }
  if (elementType.isSignlessInteger(mlir::pto::kValue64)) {
    return "i64";
  }
  if (elementType.isSignlessInteger(mlir::pto::kValue32)) {
    return "i32";
  }
  if (elementType.isSignlessInteger(mlir::pto::kValue16)) {
    return "i16";
  }
  if (elementType.isSignlessInteger(mlir::pto::kValue8)) {
    return "i8";
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

static std::string getMemorySpaceString(pto::TileBufType tileType) {
  auto memorySpace =
      dyn_cast_or_null<pto::AddressSpaceAttr>(tileType.getMemorySpace());
  return memorySpace ? stringifyMemorySpace(memorySpace.getAddressSpace())
                     : "ub";
}

static std::string getMemorySpaceString(MemRefType memrefType) {
  auto memorySpace =
      dyn_cast_or_null<pto::AddressSpaceAttr>(memrefType.getMemorySpace());
  return memorySpace ? stringifyMemorySpace(memorySpace.getAddressSpace())
                     : "gm";
}

static std::string getMemorySpaceString(pto::PartitionTensorViewType) {
  return "gm";
}

static std::string getMemorySpaceString(pto::PtrType ptrType) {
  return stringifyMemorySpace(ptrType.getMemorySpace().getAddressSpace());
}

static StringRef getBLayoutString(pto::BLayout layout) {
  return layout == pto::BLayout::ColMajor ? "col_major" : "row_major";
}

static StringRef getSLayoutString(pto::SLayout layout) {
  if (layout == pto::SLayout::RowMajor) {
    return "row_major";
  }
  if (layout == pto::SLayout::ColMajor) {
    return "col_major";
  }
  return "none_box";
}

static void appendJsonIntArray(std::string &json, ArrayRef<int64_t> values) {
  json += "[";
  for (auto [index, value] : llvm::enumerate(values)) {
    if (index != 0) {
      json += ",";
    }
    json += std::to_string(value);
  }
  json += "]";
}

static void appendJsonDimArray(std::string &json, ArrayRef<int64_t> values) {
  json += "[";
  for (auto [index, value] : llvm::enumerate(values)) {
    if (index != 0) {
      json += ",";
    }
    if (ShapedType::isDynamic(value)) {
      json += "null";
      continue;
    }
    json += std::to_string(value);
  }
  json += "]";
}

static bool getStaticIntFromValue(Value value, int64_t &out) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>()) {
    out = constant.value();
    return true;
  }
  if (auto constant = value.getDefiningOp<arith::ConstantIntOp>()) {
    out = constant.value();
    return true;
  }
  return false;
}

static int64_t getStaticIntOrDynamic(OpFoldResult value) {
  if (isa<Attribute>(value)) {
    Attribute attr = cast<Attribute>(value);
    if (auto integer = dyn_cast<IntegerAttr>(attr)) {
      return integer.getInt();
    }
    return ShapedType::kDynamic;
  }

  int64_t result = ShapedType::kDynamic;
  if (getStaticIntFromValue(cast<Value>(value), result)) {
    return result;
  }
  return ShapedType::kDynamic;
}

static void recordStaticSizes(ArrayRef<OpFoldResult> values,
                              SmallVectorImpl<int64_t> &out) {
  out.clear();
  out.reserve(values.size());
  for (OpFoldResult value : values) {
    out.push_back(getStaticIntOrDynamic(value));
  }
}

static SmallVector<int64_t>
combineSubviewStrides(ArrayRef<int64_t> baseStrides,
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

  Operation *definingOp = value.getDefiningOp();
  while (definingOp) {
    if (auto part = dyn_cast<pto::PartitionViewOp>(definingOp)) {
      value = part.getSource();
      definingOp = value.getDefiningOp();
      continue;
    }
    if (auto layout = getLayoutAttrFromOp(definingOp)) {
      return layout;
    }
    if (auto subview = dyn_cast<memref::SubViewOp>(definingOp)) {
      value = subview.getSource();
      definingOp = value.getDefiningOp();
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(definingOp)) {
      value = cast.getSource();
      definingOp = value.getDefiningOp();
      continue;
    }
    if (auto reinterpret =
            dyn_cast<memref::ReinterpretCastOp>(definingOp)) {
      value = reinterpret.getSource();
      definingOp = value.getDefiningOp();
      continue;
    }
    break;
  }
  return std::nullopt;
}

static void populatePTOViewShapeAndStrides(Value value,
                                           SmallVectorImpl<int64_t> &shape,
                                           SmallVectorImpl<int64_t> &strides) {
  if (!value) {
    return;
  }

  if (auto part = value.getDefiningOp<pto::PartitionViewOp>()) {
    if (shape.empty()) {
      shape.reserve(part.getSizes().size());
      for (Value sizeValue : part.getSizes()) {
        int64_t size = ShapedType::kDynamic;
        (void)getStaticIntFromValue(sizeValue, size);
        shape.push_back(size);
      }
      if (shape.empty()) {
        auto partTy =
            dyn_cast<pto::PartitionTensorViewType>(part.getResult().getType());
        if (partTy) {
          shape.assign(partTy.getShape().begin(), partTy.getShape().end());
        }
      }
    }
    SmallVector<int64_t> sourceShape;
    SmallVector<int64_t> sourceStrides;
    populatePTOViewShapeAndStrides(part.getSource(), sourceShape,
                                   sourceStrides);
    if (strides.empty() && !sourceStrides.empty()) {
      strides = sourceStrides;
    }
    return;
  }

  if (auto make = value.getDefiningOp<pto::MakeTensorViewOp>()) {
    if (shape.empty()) {
      auto viewTy = dyn_cast<pto::TensorViewType>(make.getResult().getType());
      if (viewTy) {
        shape.assign(viewTy.getShape().begin(), viewTy.getShape().end());
      }
    }
    if (strides.empty()) {
      strides.reserve(make.getStrides().size());
      for (Value strideValue : make.getStrides()) {
        int64_t stride = ShapedType::kDynamic;
        (void)getStaticIntFromValue(strideValue, stride);
        strides.push_back(stride);
      }
    }
    return;
  }

  if (auto viewTy = dyn_cast<pto::TensorViewType>(value.getType())) {
    if (shape.empty()) {
      shape.assign(viewTy.getShape().begin(), viewTy.getShape().end());
    }
  }
}

static void populateViewShapeAndStrides(Value value,
                                        SmallVectorImpl<int64_t> &shape,
                                        SmallVectorImpl<int64_t> &strides) {
  if (!value) {
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

  if (auto reinterpret =
          value.getDefiningOp<memref::ReinterpretCastOp>()) {
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

  if (auto memrefType = dyn_cast<MemRefType>(value.getType())) {
    if (shape.empty()) {
      shape.assign(memrefType.getShape().begin(), memrefType.getShape().end());
    }
    if (strides.empty()) {
      int64_t offset = ShapedType::kDynamic;
      (void)mlir::pto::getPTOMemRefStridesAndOffset(memrefType, strides,
                                                     offset);
    }
  }
}

static std::optional<std::string>
getViewLayoutString(std::optional<pto::Layout> layout) {
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

template <typename OpT>
static bool tryAppendPrecisionType(
    Operation *op, SmallVectorImpl<std::pair<std::string, std::string>> &attrs) {
  auto typed = dyn_cast<OpT>(op);
  if (!typed) {
    return false;
  }
  attrs.emplace_back("precisionType",
                     getPrecisionTypeString(typed.getPrecisionType()).str());
  return true;
}

// Candidate discovery runs before memory planning, so address-dependent
// context such as tfillpad's lowering_kind intentionally belongs only to the
// post-planning specialization key built by ExpandTileOp.
static void appendOpContextAttrs(
    Operation *op, SmallVectorImpl<std::pair<std::string, std::string>> &attrs) {
  if (auto tcvt = dyn_cast<pto::TCvtOp>(op)) {
    if (auto roundMode = getTCvtRoundModeString(tcvt)) {
      attrs.emplace_back("round_mode", *roundMode);
    }
  }
  if (auto trandom = dyn_cast<pto::TRandomOp>(op)) {
    attrs.emplace_back("rounds", std::to_string(trandom.getRounds()));
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
  if (auto tmrgsort = dyn_cast<pto::TMrgSortOp>(op)) {
    attrs.emplace_back("exhausted",
                       tmrgsort.getExhausted() ? "1" : "0");
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
  (void)(tryAppendPrecisionType<pto::TExpOp>(op, attrs) ||
         tryAppendPrecisionType<pto::TLogOp>(op, attrs) ||
         tryAppendPrecisionType<pto::TSqrtOp>(op, attrs) ||
         tryAppendPrecisionType<pto::TRecipOp>(op, attrs) ||
         tryAppendPrecisionType<pto::TRsqrtOp>(op, attrs) ||
         tryAppendPrecisionType<pto::TDivOp>(op, attrs) ||
         tryAppendPrecisionType<pto::TDivSOp>(op, attrs) ||
         tryAppendPrecisionType<pto::TRowExpandDivOp>(op, attrs) ||
         tryAppendPrecisionType<pto::TColExpandDivOp>(op, attrs));
}

static std::string buildContextAttrsJson(Operation *operation) {
  SmallVector<std::pair<std::string, std::string>, mlir::pto::kValue4> attrs;
  appendOpContextAttrs(operation, attrs);

  std::string json = "{";
  for (auto [index, attr] : llvm::enumerate(attrs)) {
    if (index != 0) {
      json += ",";
    }
    json += "\"";
    json += attr.first;
    json += "\":\"";
    json += attr.second;
    json += "\"";
  }
  json += "}";
  return json;
}

static void appendTileOperandSpecJson(std::string &json,
                                      pto::TileBufType tileType) {
  std::string dtype = getDtypeString(tileType.getElementType());
  json += "{\"kind\":\"tile\",\"dtype\":\"" + dtype + "\",\"shape\":";
  appendJsonIntArray(json, tileType.getShape());
  json += ",\"valid_shape\":";
  auto validShape = tileType.getValidShape();
  appendJsonIntArray(json, validShape.empty() ? tileType.getShape()
                                              : validShape);
  json += ",\"memory_space\":\"";
  json += getMemorySpaceString(tileType);

  pto::BLayout bLayout = pto::BLayout::RowMajor;
  pto::SLayout sLayout = pto::SLayout::NoneBox;
  int64_t fractalSize = 0;
  uint64_t padValue = 0;
  int32_t compactMode = static_cast<int32_t>(pto::CompactMode::Null);
  if (auto config = tileType.getConfigAttr()) {
    bLayout = config.getBLayout().getValue();
    sLayout = config.getSLayout().getValue();
    if (config.getSFractalSize()) {
      fractalSize = config.getSFractalSize().getInt();
    }
    padValue = static_cast<uint64_t>(config.getPad().getValue());
    compactMode = static_cast<int32_t>(config.getCompactMode().getValue());
  }

  json += "\",\"config\":{\"b_layout\":\"";
  json += getBLayoutString(bLayout);
  json += "\",\"s_layout\":\"";
  json += getSLayoutString(sLayout);
  json += "\",\"s_fractal_size\":";
  json += std::to_string(fractalSize);
  json += ",\"pad_value\":\"0x";
  json += llvm::utohexstr(padValue, /*LowerCase=*/false);
  json += "\",\"compact_mode\":";
  json += std::to_string(compactMode);
  json += "}}";
}

static void appendViewOperandSpecJson(std::string &json, Value operand,
                                      MemRefType memrefType) {
  std::string dtype = getDtypeString(memrefType.getElementType());
  json += "{\"kind\":\"view\",\"dtype\":\"" + dtype + "\",\"shape\":";
  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;
  populateViewShapeAndStrides(operand, shape, strides);
  if (shape.empty()) {
    shape.assign(memrefType.getShape().begin(), memrefType.getShape().end());
  }
  appendJsonDimArray(json, shape);
  if (!strides.empty()) {
    json += ",\"strides\":";
    appendJsonDimArray(json, strides);
  }
  json += ",\"memory_space\":\"";
  json += getMemorySpaceString(memrefType);
  json += "\"";
  if (auto layout = getViewLayoutString(resolveViewLayout(operand))) {
    json += ",\"config\":{\"layout\":\"";
    json += *layout;
    json += "\"}";
  }
  json += "}";
}

static void appendViewOperandSpecJson(std::string &json, Value operand,
                                      pto::PartitionTensorViewType viewType) {
  std::string dtype = getDtypeString(viewType.getElementType());
  json += "{\"kind\":\"view\",\"dtype\":\"" + dtype + "\",\"shape\":";
  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;
  populatePTOViewShapeAndStrides(operand, shape, strides);
  if (shape.empty()) {
    shape.assign(viewType.getShape().begin(), viewType.getShape().end());
  }
  appendJsonDimArray(json, shape);
  if (!strides.empty()) {
    json += ",\"strides\":";
    appendJsonDimArray(json, strides);
  }
  json += ",\"memory_space\":\"";
  json += getMemorySpaceString(viewType);
  json += "\"";
  if (auto layout = getViewLayoutString(resolveViewLayout(operand))) {
    json += ",\"config\":{\"layout\":\"";
    json += *layout;
    json += "\"}";
  }
  json += "}";
}

static void appendVectorOperandSpecJson(std::string &json,
                                        VectorType vectorType) {
  std::string dtype = getDtypeString(vectorType.getElementType());
  json += "{\"kind\":\"vector\",\"dtype\":\"" + dtype + "\",\"shape\":";
  appendJsonIntArray(json, vectorType.getShape());
  json += "}";
}

static void appendScalarOperandSpecJson(std::string &json, Value operand) {
  std::string dtype = getDtypeString(operand.getType());
  json += "{\"kind\":\"scalar\",\"dtype\":\"" + dtype + "\"";
  int64_t scalarValue = 0;
  if (getStaticIntFromValue(operand, scalarValue)) {
    json += ",\"value\":";
    json += std::to_string(scalarValue);
  }
  json += "}";
}

static void appendPtrOperandSpecJson(std::string &json, pto::PtrType ptrType) {
  json += "{\"kind\":\"pointer\",\"dtype\":\"";
  json += getDtypeString(ptrType.getElementType());
  json += "\",\"memory_space\":\"";
  json += getMemorySpaceString(ptrType);
  json += "\"}";
}

static std::optional<std::string>
buildOperandSpecsJson(Operation *operation) {
  std::string json = "[";
  for (auto [index, operand] : llvm::enumerate(operation->getOperands())) {
    if (index != 0) {
      json += ",";
    }

    Type type = operand.getType();
    if (auto tileType = dyn_cast<pto::TileBufType>(type)) {
      if (getDtypeString(tileType.getElementType()).empty()) {
        operation->emitError(
            "InsertTemplateAttributes encountered an unsupported tile dtype");
        return std::nullopt;
      }
      appendTileOperandSpecJson(json, tileType);
      continue;
    }

    if (auto memrefType = dyn_cast<MemRefType>(type)) {
      if (getDtypeString(memrefType.getElementType()).empty()) {
        operation->emitError(
            "InsertTemplateAttributes encountered an unsupported view dtype");
        return std::nullopt;
      }
      appendViewOperandSpecJson(json, operand, memrefType);
      continue;
    }

    if (auto viewType = dyn_cast<pto::PartitionTensorViewType>(type)) {
      if (getDtypeString(viewType.getElementType()).empty()) {
        operation->emitError(
            "InsertTemplateAttributes encountered an unsupported view dtype");
        return std::nullopt;
      }
      appendViewOperandSpecJson(json, operand, viewType);
      continue;
    }

    if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
      if (getDtypeString(ptrType.getElementType()).empty()) {
        operation->emitError(
            "InsertTemplateAttributes encountered an unsupported pointer dtype");
        return std::nullopt;
      }
      appendPtrOperandSpecJson(json, ptrType);
      continue;
    }

    if (auto vectorType = dyn_cast<VectorType>(type)) {
      if (getDtypeString(vectorType.getElementType()).empty()) {
        operation->emitError(
            "InsertTemplateAttributes encountered an unsupported vector dtype");
        return std::nullopt;
      }
      appendVectorOperandSpecJson(json, vectorType);
      continue;
    }

    if (!getDtypeString(type).empty()) {
      appendScalarOperandSpecJson(json, operand);
      continue;
    }

    operation->emitError(
        "InsertTemplateAttributes encountered an unsupported operand type ")
        << type;
    return std::nullopt;
  }
  json += "]";
  return json;
}

static std::optional<std::string>
getTargetArch(Operation *operation) {
  auto module = operation->getParentOfType<ModuleOp>();
  if (!module) {
    operation->emitError(
        "InsertTemplateAttributes requires a parent module");
    return std::nullopt;
  }

  for (ModuleOp current = module; current;
       current = current->getParentOfType<ModuleOp>()) {
    if (auto target = current->getAttrOfType<StringAttr>("pto.target_arch")) {
      return target.getValue().str();
    }
  }

  operation->emitError(
      "InsertTemplateAttributes requires pto.target_arch");
  return std::nullopt;
}

static FailureOr<ArrayAttr>
parseCandidateAttributes(Operation *operation, StringRef metadataJson) {
  auto parsed = llvm::json::parse(metadataJson);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    operation->emitError(
        "InsertTemplateAttributes received invalid metadata JSON");
    return failure();
  }

  auto *root = parsed->getAsObject();
  auto *candidates = root ? root->getArray("candidates") : nullptr;
  if (!candidates || candidates->empty()) {
    operation->emitError("InsertTemplateAttributes found no legal template "
                         "candidates for ")
        << operation->getName();
    return failure();
  }

  SmallVector<CandidateMetadata> parsedCandidates;
  parsedCandidates.reserve(candidates->size());
  llvm::DenseSet<int64_t> candidateIds;
  for (const llvm::json::Value &entry : *candidates) {
    auto *metadata = entry.getAsObject();
    if (!metadata) {
      operation->emitError(
          "InsertTemplateAttributes candidate metadata must be an object");
      return failure();
    }

    auto name = metadata->getString("name");
    auto id = metadata->getInteger("id");
    auto priority = metadata->getInteger("priority");
    auto loopDepth = metadata->getInteger("loop_depth");
    auto postUpdate = metadata->getBoolean("is_post_update");
    auto tail = metadata->getBoolean("has_tail");
    if (!name || !priority || !loopDepth || !postUpdate || !tail) {
      operation->emitError(
          "InsertTemplateAttributes candidate metadata is missing name, "
          "priority, loop_depth, is_post_update, or has_tail");
      return failure();
    }
    if (!id && candidates->size() != 1) {
      operation->emitError(
          "InsertTemplateAttributes requires an id for every "
          "multi-candidate template");
      return failure();
    }

    int64_t candidateId = id.value_or(0);
    if (!candidateIds.insert(candidateId).second) {
      operation->emitError(
          "InsertTemplateAttributes candidate ids must be unique");
      return failure();
    }

    parsedCandidates.push_back(CandidateMetadata{
        candidateId,
        name->str(),
        *priority,
        *loopDepth,
        *postUpdate,
        *tail,
    });
  }

  llvm::sort(parsedCandidates,
             [](const CandidateMetadata &left,
                const CandidateMetadata &right) {
               if (left.priority != right.priority) {
                 return left.priority > right.priority;
               }
               return left.name < right.name;
             });
  if (parsedCandidates.size() > 1 &&
      parsedCandidates[0].priority == parsedCandidates[1].priority) {
    operation->emitError(
        "InsertTemplateAttributes found multiple legal templates tied at "
        "the highest priority: ")
        << parsedCandidates[0].name << " and " << parsedCandidates[1].name
        << " at priority " << parsedCandidates[0].priority
        << "; assign distinct priorities or make their constraints mutually "
           "exclusive";
    return failure();
  }

  Builder builder(operation->getContext());
  SmallVector<Attribute> attributes;
  attributes.reserve(parsedCandidates.size());
  for (const CandidateMetadata &candidate : parsedCandidates) {
    attributes.push_back(DictionaryAttr::get(
        operation->getContext(),
        {
            builder.getNamedAttr("id", builder.getI64IntegerAttr(candidate.id)),
            builder.getNamedAttr("name",
                                 builder.getStringAttr(candidate.name)),
            builder.getNamedAttr(
                "loop_depth",
                builder.getI64IntegerAttr(candidate.loopDepth)),
            builder.getNamedAttr(
                "postupdate",
                builder.getI64IntegerAttr(candidate.postUpdate ? 1 : 0)),
            builder.getNamedAttr(
                "tail", builder.getI64IntegerAttr(candidate.tail ? 1 : 0)),
        }));
  }
  return builder.getArrayAttr(attributes);
}

struct InsertTemplateAttributesPass
    : public pto::impl::InsertTemplateAttributesBase<
          InsertTemplateAttributesPass> {
  using InsertTemplateAttributesBase::InsertTemplateAttributesBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<Operation *> tileOperations;
    module.walk([&](Operation *operation) {
      if (pto::isTileLibExpandableOp(operation)) {
        tileOperations.push_back(operation);
      }
    });
    if (tileOperations.empty()) {
      return;
    }
    std::shared_ptr<pto::TileLibService> tileLibService =
        pto::TileLibRuntime::getService();
    if (!tileLibService) {
      module.emitError(
          "InsertTemplateAttributes requires an initialized PTODSL runtime");
      return signalPassFailure();
    }

    for (Operation *operation : tileOperations) {
      auto target = getTargetArch(operation);
      auto operandSpecs = buildOperandSpecsJson(operation);
      if (!target || !operandSpecs) {
        return signalPassFailure();
      }
      pto::TileLibMaterializationRequest request;
      request.target = std::move(*target);
      request.op = operation->getName().getStringRef().str();
      request.operandSpecsJson = std::move(*operandSpecs);
      request.contextAttrsJson = buildContextAttrsJson(operation);
      FailureOr<std::string> metadata = tileLibService->getMetadata(request);
      if (failed(metadata)) {
        operation->emitError("in-process PTODSL metadata query failed");
        return signalPassFailure();
      }

      auto candidates = parseCandidateAttributes(operation, *metadata);
      if (failed(candidates)) {
        return signalPassFailure();
      }
      operation->setAttr(kCandidatesAttr, *candidates);
    }
  }
};

} // namespace

namespace mlir {
namespace pto {

std::unique_ptr<Pass> createInsertTemplateAttributesPass() {
  return std::make_unique<InsertTemplateAttributesPass>();
}

} // namespace pto
} // namespace mlir
