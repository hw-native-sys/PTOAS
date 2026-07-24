// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Support/PythonExecutable.h"
#include "PTO/Transforms/Passes.h"
#include "Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <unistd.h>
#include <vector>

extern "C" {
extern char **environ;
}

using namespace mlir;

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_INSERTTEMPLATEATTRIBUTES
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

namespace {

constexpr llvm::StringLiteral kCandidatesAttr = "candidates";

static bool isUnifiedInternalPipeOperation(Operation *operation) {
  return isa<pto::TAllocOp, pto::TPushOp, pto::TPopOp, pto::TFreeOp,
             pto::TDrainOp>(operation);
}

struct CandidateMetadata {
  int64_t id;
  std::string name;
  int64_t loopDepth;
  bool postUpdate;
  bool tail;
};

static std::string getDtypeString(Type elementType) {
  if (elementType.isIndex())
    return "i32";
  if (elementType.isInteger(1))
    return "i1";
  if (elementType.isF32())
    return "f32";
  if (elementType.isF16())
    return "f16";
  if (elementType.isBF16())
    return "bf16";
  if (isa<Float8E4M3FNType>(elementType))
    return "f8e4m3";
  if (isa<Float8E5M2Type>(elementType))
    return "f8e5m2";
  if (isa<pto::HiF8Type>(elementType))
    return "hif8";
  if (isa<pto::F4E1M2x2Type>(elementType))
    return "f4e1m2x2";
  if (isa<pto::F4E2M1x2Type>(elementType))
    return "f4e2m1x2";
  if (elementType.isUnsignedInteger(64))
    return "ui64";
  if (elementType.isUnsignedInteger(32))
    return "ui32";
  if (elementType.isUnsignedInteger(16))
    return "ui16";
  if (elementType.isUnsignedInteger(8))
    return "ui8";
  if (elementType.isSignedInteger(64))
    return "si64";
  if (elementType.isSignedInteger(32))
    return "si32";
  if (elementType.isSignedInteger(16))
    return "si16";
  if (elementType.isSignedInteger(8))
    return "si8";
  if (elementType.isSignlessInteger(64))
    return "i64";
  if (elementType.isSignlessInteger(32))
    return "i32";
  if (elementType.isSignlessInteger(16))
    return "i16";
  if (elementType.isSignlessInteger(8))
    return "i8";
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

static std::string getMemorySpaceString(pto::TensorViewType) { return "gm"; }

static StringRef getBLayoutString(pto::BLayout layout) {
  return layout == pto::BLayout::ColMajor ? "col_major" : "row_major";
}

static StringRef getSLayoutString(pto::SLayout layout) {
  if (layout == pto::SLayout::RowMajor)
    return "row_major";
  if (layout == pto::SLayout::ColMajor)
    return "col_major";
  return "none_box";
}

static void appendJsonIntArray(std::string &json, ArrayRef<int64_t> values) {
  json += "[";
  for (auto [index, value] : llvm::enumerate(values)) {
    if (index != 0)
      json += ",";
    json += std::to_string(value);
  }
  json += "]";
}

static void appendJsonDimArray(std::string &json, ArrayRef<int64_t> values) {
  json += "[";
  for (auto [index, value] : llvm::enumerate(values)) {
    if (index != 0)
      json += ",";
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
    if (auto integer = dyn_cast<IntegerAttr>(attr))
      return integer.getInt();
    return ShapedType::kDynamic;
  }

  int64_t result = ShapedType::kDynamic;
  if (getStaticIntFromValue(cast<Value>(value), result))
    return result;
  return ShapedType::kDynamic;
}

static void recordStaticSizes(ArrayRef<OpFoldResult> values,
                              SmallVectorImpl<int64_t> &out) {
  out.clear();
  out.reserve(values.size());
  for (OpFoldResult value : values)
    out.push_back(getStaticIntOrDynamic(value));
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
  if (!op)
    return std::nullopt;
  if (auto attr = op->getAttrOfType<pto::LayoutAttr>(kLayoutAttrName))
    return attr.getLayout();
  return std::nullopt;
}

static std::optional<pto::Layout> resolveViewLayout(Value value) {
  if (!value)
    return std::nullopt;

  Operation *definingOp = value.getDefiningOp();
  while (definingOp) {
    if (auto part = dyn_cast<pto::PartitionViewOp>(definingOp)) {
      value = part.getSource();
      definingOp = value.getDefiningOp();
      continue;
    }
    if (auto layout = getLayoutAttrFromOp(definingOp))
      return layout;
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
  if (!value)
    return;

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
        if (partTy)
          shape.assign(partTy.getShape().begin(), partTy.getShape().end());
      }
    }
    SmallVector<int64_t> sourceShape;
    SmallVector<int64_t> sourceStrides;
    populatePTOViewShapeAndStrides(part.getSource(), sourceShape,
                                   sourceStrides);
    if (strides.empty() && !sourceStrides.empty())
      strides = sourceStrides;
    return;
  }

  if (auto make = value.getDefiningOp<pto::MakeTensorViewOp>()) {
    if (shape.empty()) {
      auto viewTy = dyn_cast<pto::TensorViewType>(make.getResult().getType());
      if (viewTy)
        shape.assign(viewTy.getShape().begin(), viewTy.getShape().end());
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
    if (shape.empty())
      shape.assign(viewTy.getShape().begin(), viewTy.getShape().end());
  }
}

static void populateViewShapeAndStrides(Value value,
                                        SmallVectorImpl<int64_t> &shape,
                                        SmallVectorImpl<int64_t> &strides) {
  if (!value)
    return;

  if (auto subview = value.getDefiningOp<memref::SubViewOp>()) {
    populateViewShapeAndStrides(subview.getSource(), shape, strides);
    SmallVector<int64_t> subviewShape;
    recordStaticSizes(subview.getMixedSizes(), subviewShape);
    if (!subviewShape.empty())
      shape = subviewShape;
    if (!strides.empty())
      strides = combineSubviewStrides(strides, subview.getMixedStrides());
    return;
  }

  if (auto reinterpret =
          value.getDefiningOp<memref::ReinterpretCastOp>()) {
    if (shape.empty()) {
      SmallVector<int64_t> reinterpretShape;
      recordStaticSizes(reinterpret.getMixedSizes(), reinterpretShape);
      if (!reinterpretShape.empty())
        shape = reinterpretShape;
    }
    if (strides.empty())
      recordStaticSizes(reinterpret.getMixedStrides(), strides);
    return;
  }

  if (auto cast = value.getDefiningOp<memref::CastOp>()) {
    populateViewShapeAndStrides(cast.getSource(), shape, strides);
    return;
  }

  if (auto memrefType = dyn_cast<MemRefType>(value.getType())) {
    if (shape.empty())
      shape.assign(memrefType.getShape().begin(), memrefType.getShape().end());
    if (strides.empty()) {
      int64_t offset = ShapedType::kDynamic;
      (void)mlir::pto::getPTOMemRefStridesAndOffset(memrefType, strides,
                                                     offset);
    }
  }
}

static std::optional<std::string>
getViewLayoutString(std::optional<pto::Layout> layout) {
  if (!layout)
    return std::nullopt;
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
  if (!typed)
    return false;
  attrs.emplace_back("precisionType",
                     getPrecisionTypeString(typed.getPrecisionType()).str());
  return true;
}

static void appendOpContextAttrs(
    Operation *op, SmallVectorImpl<std::pair<std::string, std::string>> &attrs) {
  if (isUnifiedInternalPipeOperation(op)) {
    auto func = op->getParentOfType<mlir::func::FuncOp>();
    auto kernelKind =
        func ? func->getAttrOfType<pto::FunctionKernelKindAttr>(
                   pto::FunctionKernelKindAttr::name)
             : pto::FunctionKernelKindAttr();
    if (!kernelKind) {
      op->emitError(
          "requires a cube or vector kernel_kind for TileLib pipe expansion");
      return;
    }
    attrs.emplace_back(
        "kernel_kind",
        kernelKind.getKernelKind() == pto::FunctionKernelKind::Cube ? "cube"
                                                                    : "vector");
  }
  if (auto tcvt = dyn_cast<pto::TCvtOp>(op)) {
    if (auto roundMode = getTCvtRoundModeString(tcvt))
      attrs.emplace_back("round_mode", *roundMode);
  }
  if (auto trandom = dyn_cast<pto::TRandomOp>(op))
    attrs.emplace_back("rounds", std::to_string(trandom.getRounds()));
  if (auto tcmp = dyn_cast<pto::TCmpOp>(op)) {
    if (auto cmpModeAttr = tcmp.getCmpModeAttr())
      attrs.emplace_back("cmp_mode",
                         stringifyCmpMode(cmpModeAttr.getValue()).str());
  }
  if (auto tcmps = dyn_cast<pto::TCmpSOp>(op)) {
    if (auto cmpModeAttr = tcmps.getCmpModeAttr())
      attrs.emplace_back("cmp_mode",
                         stringifyCmpMode(cmpModeAttr.getValue()).str());
  }
  if (auto tmrgsort = dyn_cast<pto::TMrgSortOp>(op))
    attrs.emplace_back("exhausted",
                       tmrgsort.getExhausted() ? "1" : "0");
  if (auto tgather = dyn_cast<pto::TGatherOp>(op)) {
    if (auto maskPatternAttr = tgather.getMaskPatternAttr()) {
      attrs.emplace_back(
          "mask_pattern",
          stringifyMaskPattern(maskPatternAttr.getValue()).str());
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
  SmallVector<std::pair<std::string, std::string>, 4> attrs;
  appendOpContextAttrs(operation, attrs);

  std::string json = "{";
  for (auto [index, attr] : llvm::enumerate(attrs)) {
    if (index != 0)
      json += ",";
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
  if (auto config = tileType.getConfigAttr()) {
    bLayout = config.getBLayout().getValue();
    sLayout = config.getSLayout().getValue();
    if (config.getSFractalSize())
      fractalSize = config.getSFractalSize().getInt();
    padValue = static_cast<uint64_t>(config.getPad().getValue());
  }

  json += "\",\"config\":{\"b_layout\":\"";
  json += getBLayoutString(bLayout);
  json += "\",\"s_layout\":\"";
  json += getSLayoutString(sLayout);
  json += "\",\"s_fractal_size\":";
  json += std::to_string(fractalSize);
  json += ",\"pad_value\":\"0x";
  json += llvm::utohexstr(padValue, /*LowerCase=*/false);
  json += "\"}}";
}

static std::optional<pto::TileBufType> getBoundTileType(Value value) {
  auto bind = value.getDefiningOp<pto::BindTileOp>();
  if (!bind)
    return std::nullopt;

  auto memrefType = dyn_cast<MemRefType>(value.getType());
  if (!memrefType || memrefType.getRank() != 2)
    return std::nullopt;

  SmallVector<int64_t, 2> validShape(memrefType.getShape().begin(),
                                     memrefType.getShape().end());
  auto updateValidDim = [&](Value validDim, unsigned dimension) {
    if (!validDim)
      return;
    int64_t staticValue = 0;
    validShape[dimension] = getStaticIntFromValue(validDim, staticValue)
                                ? staticValue
                                : ShapedType::kDynamic;
  };
  updateValidDim(bind.getValidRow(), 0);
  updateValidDim(bind.getValidCol(), 1);

  return pto::TileBufType::get(value.getType().getContext(),
                                memrefType.getShape(),
                                memrefType.getElementType(),
                                memrefType.getMemorySpace(), validShape,
                                bind.getConfig());
}

static void appendViewOperandSpecJson(std::string &json, Value operand,
                                      MemRefType memrefType) {
  std::string dtype = getDtypeString(memrefType.getElementType());
  json += "{\"kind\":\"view\",\"dtype\":\"" + dtype + "\",\"shape\":";
  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;
  populateViewShapeAndStrides(operand, shape, strides);
  if (shape.empty())
    shape.assign(memrefType.getShape().begin(), memrefType.getShape().end());
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
  if (shape.empty())
    shape.assign(viewType.getShape().begin(), viewType.getShape().end());
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

static void appendViewOperandSpecJson(std::string &json, Value operand,
                                      pto::TensorViewType viewType,
                                      StringRef kind = "view") {
  std::string dtype = getDtypeString(viewType.getElementType());
  json += "{\"kind\":\"" + kind.str() + "\",\"dtype\":\"" +
          dtype + "\",\"shape\":";
  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;
  populatePTOViewShapeAndStrides(operand, shape, strides);
  if (shape.empty())
    shape.assign(viewType.getShape().begin(), viewType.getShape().end());
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

static LogicalResult appendOperandSpecJson(std::string &json,
                                           Operation *operation,
                                           Value operand) {
  Type type = operand.getType();
  if (auto tileType = dyn_cast<pto::TileBufType>(type)) {
    if (getDtypeString(tileType.getElementType()).empty())
      return operation->emitError(
          "InsertTemplateAttributes encountered an unsupported tile dtype");
    appendTileOperandSpecJson(json, tileType);
    return success();
  }

  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    if (getDtypeString(memrefType.getElementType()).empty())
      return operation->emitError(
          "InsertTemplateAttributes encountered an unsupported view dtype");
    appendViewOperandSpecJson(json, operand, memrefType);
    return success();
  }

  if (auto viewType = dyn_cast<pto::PartitionTensorViewType>(type)) {
    if (getDtypeString(viewType.getElementType()).empty())
      return operation->emitError(
          "InsertTemplateAttributes encountered an unsupported view dtype");
    appendViewOperandSpecJson(json, operand, viewType);
    return success();
  }

  if (auto viewType = dyn_cast<pto::TensorViewType>(type)) {
    if (getDtypeString(viewType.getElementType()).empty())
      return operation->emitError(
          "InsertTemplateAttributes encountered an unsupported view dtype");
    appendViewOperandSpecJson(json, operand, viewType);
    return success();
  }

  if (auto vectorType = dyn_cast<VectorType>(type)) {
    if (getDtypeString(vectorType.getElementType()).empty())
      return operation->emitError(
          "InsertTemplateAttributes encountered an unsupported vector dtype");
    appendVectorOperandSpecJson(json, vectorType);
    return success();
  }

  if (!getDtypeString(type).empty()) {
    appendScalarOperandSpecJson(json, operand);
    return success();
  }

  return operation->emitError(
      "InsertTemplateAttributes encountered an unsupported operand type ")
         << type;
}

struct PipeWireInfo {
  bool isL2G2L = false;
  int64_t dirMask = 0;
  int64_t slotSize = 0;
  int64_t slotNum = 0;
  std::optional<int64_t> localSlotNum;
  int64_t flagBase = 0;
  bool noSplit = false;
  int64_t split = 0;
  SmallVector<std::pair<std::string, Value>> resources;
};

static FailureOr<PipeWireInfo> getPipeWireInfo(Operation *operation) {
  Value pipe;
  Value state;
  int64_t split = 0;
  if (auto push = dyn_cast<pto::TPushOp>(operation)) {
    pipe = push.getPipeHandle();
    state = push.getPipeState();
    split = push.getSplit();
  } else if (auto pop = dyn_cast<pto::TPopOp>(operation)) {
    pipe = pop.getPipeHandle();
    state = pop.getPipeState();
    split = pop.getSplit();
  } else if (auto alloc = dyn_cast<pto::TAllocOp>(operation)) {
    pipe = alloc.getPipeHandle();
    state = alloc.getPipeState();
    split = alloc.getSplit();
  } else if (auto free = dyn_cast<pto::TFreeOp>(operation)) {
    pipe = free.getPipeHandle();
    state = free.getPipeState();
    split = free.getSplit();
  } else if (auto drain = dyn_cast<pto::TDrainOp>(operation)) {
    pipe = drain.getPipeHandle();
    state = drain.getPipeState();
    split = drain.getSplit();
  } else {
    return operation->emitError("is not a unified internal pipe operation");
  }

  if (!state)
    return operation->emitError(
        "requires a materialized !pto.struct<i32, i32> pipe_state");

  Operation *init = mlir::pto::getPipeInitDef(pipe);
  if (!init)
    return operation->emitError("cannot resolve its pipe initializer");
  if (mlir::pto::getPipeInitAccPushEpilogue(init))
    return operation->emitError(
        "does not support acc_push_epilogue with --enable-pipe-tilelib-expand");

  PipeWireInfo info;
  info.split = split;
  auto addResource = [&](StringRef name, Value value) {
    if (value)
      info.resources.emplace_back(name.str(), value);
  };
  if (auto l2l = dyn_cast<pto::InitializeL2LPipeOp>(init)) {
    if (!l2l.getFlagBaseAttr())
      return operation->emitError(
          "requires resolved pipe flag_base before TileLib expansion");
    info.dirMask = l2l.getDirMask();
    info.slotSize = l2l.getSlotSize();
    info.slotNum = l2l.getSlotNum();
    info.flagBase = l2l.getFlagBaseAttr().getInt();
    info.noSplit = l2l.getNosplitAttr() && l2l.getNosplitAttr().getValue();
    addResource("local_addr", l2l.getLocalAddr());
    addResource("peer_local_addr", l2l.getPeerLocalAddr());
    return info;
  }

  auto l2g2l = dyn_cast<pto::InitializeL2G2LPipeOp>(init);
  if (!l2g2l)
    return operation->emitError("requires an l2l or l2g2l pipe initializer");
  if (!l2g2l.getFlagBaseAttr())
    return operation->emitError(
        "requires resolved pipe flag_base before TileLib expansion");
  info.isL2G2L = true;
  info.dirMask = l2g2l.getDirMask();
  info.slotSize = l2g2l.getSlotSize();
  info.slotNum = l2g2l.getSlotNum();
  if (auto attr = l2g2l.getLocalSlotNumAttr())
    info.localSlotNum = attr.getInt();
  info.flagBase = l2g2l.getFlagBaseAttr().getInt();
  info.noSplit = l2g2l.getNosplitAttr() && l2g2l.getNosplitAttr().getValue();
  addResource("gm_addr", l2g2l.getGmAddr());
  addResource("local_addr", l2g2l.getLocalAddr());
  addResource("peer_local_addr", l2g2l.getPeerLocalAddr());
  return info;
}

static void appendPipeSpecJson(std::string &json, const PipeWireInfo &info) {
  json += "{\"kind\":\"pipe\",\"init_kind\":\"";
  json += info.isL2G2L ? "l2g2l" : "l2l";
  json += "\",\"dir_mask\":" + std::to_string(info.dirMask);
  json += ",\"slot_size\":" + std::to_string(info.slotSize);
  json += ",\"slot_num\":" + std::to_string(info.slotNum);
  json += ",\"local_slot_num\":";
  json += info.localSlotNum ? std::to_string(*info.localSlotNum) : "null";
  json += ",\"flag_base\":" + std::to_string(info.flagBase);
  json += ",\"nosplit\":";
  json += info.noSplit ? "true" : "false";
  json += ",\"split\":" + std::to_string(info.split);
  json += ",\"resource_names\":[";
  for (auto [index, resource] : llvm::enumerate(info.resources)) {
    if (index != 0)
      json += ",";
    json += "\"" + resource.first + "\"";
  }
  json += "]}";
}

static std::optional<std::string>
buildPipeOperandSpecsJson(Operation *operation) {
  FailureOr<PipeWireInfo> info = getPipeWireInfo(operation);
  if (failed(info))
    return std::nullopt;

  std::string json = "[";
  auto appendComma = [&]() {
    if (json.size() != 1)
      json += ",";
  };
  auto appendPipeEntry = [&](Value value) -> LogicalResult {
    appendComma();
    if (auto tileType = getBoundTileType(value)) {
      if (getDtypeString(tileType->getElementType()).empty())
        return operation->emitError(
            "InsertTemplateAttributes encountered an unsupported bound pipe tile dtype");
      appendTileOperandSpecJson(json, *tileType);
      return success();
    }
    auto viewType = dyn_cast<pto::TensorViewType>(value.getType());
    if (!viewType)
      return appendOperandSpecJson(json, operation, value);
    if (!value.getDefiningOp<pto::DeclareGlobalOp>())
      return operation->emitError(
          "requires a pto.declare_global tensor_view entry for pipe TileLib expansion");
    if (getDtypeString(viewType.getElementType()).empty())
      return operation->emitError(
          "InsertTemplateAttributes encountered an unsupported pipe entry dtype");
    appendViewOperandSpecJson(json, value, viewType, "pipe_entry");
    return success();
  };

  if (auto push = dyn_cast<pto::TPushOp>(operation)) {
    if (failed(appendPipeEntry(push.getTile())))
      return std::nullopt;
  } else if (auto pop = dyn_cast<pto::TPopOp>(operation)) {
    if (failed(appendPipeEntry(pop.getTile())))
      return std::nullopt;
  } else if (auto alloc = dyn_cast<pto::TAllocOp>(operation)) {
    if (failed(appendPipeEntry(alloc.getEntry())))
      return std::nullopt;
  } else if (auto free = dyn_cast<pto::TFreeOp>(operation)) {
    if (free.getEntry() && failed(appendPipeEntry(free.getEntry())))
      return std::nullopt;
  }

  appendComma();
  appendPipeSpecJson(json, *info);
  appendComma();
  json += "{\"kind\":\"pipe_resources\",\"names\":[";
  for (auto [index, resource] : llvm::enumerate(info->resources)) {
    if (index != 0)
      json += ",";
    json += "\"" + resource.first + "\"";
  }
  json += "],\"values\":[";
  for (auto [index, resource] : llvm::enumerate(info->resources)) {
    if (index != 0)
      json += ",";
    if (failed(appendOperandSpecJson(json, operation, resource.second)))
      return std::nullopt;
  }
  json += "]}";
  appendComma();
  json += "{\"kind\":\"pipe_state\",\"fields\":[\"i32\",\"i32\"]}";

  if (auto push = dyn_cast<pto::TPushOp>(operation)) {
    appendComma();
    if (Value subblock = push.getAivSubblockid()) {
      if (failed(appendOperandSpecJson(json, operation, subblock)))
        return std::nullopt;
    } else {
      json += "{\"kind\":\"scalar\",\"dtype\":\"i64\",\"value\":0}";
    }
  } else if (auto pop = dyn_cast<pto::TPopOp>(operation)) {
    appendComma();
    if (Value subblock = pop.getAivSubblockid()) {
      if (failed(appendOperandSpecJson(json, operation, subblock)))
        return std::nullopt;
    } else {
      json += "{\"kind\":\"scalar\",\"dtype\":\"i64\",\"value\":0}";
    }
  }
  json += "]";
  return json;
}

static std::optional<std::string>
buildOperandSpecsJson(Operation *operation) {
  if (isUnifiedInternalPipeOperation(operation))
    return buildPipeOperandSpecsJson(operation);

  std::string json = "[";
  for (auto [index, operand] : llvm::enumerate(operation->getOperands())) {
    if (index != 0)
      json += ",";
    if (failed(appendOperandSpecJson(json, operation, operand)))
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
    if (auto target = current->getAttrOfType<StringAttr>("pto.target_arch"))
      return target.getValue().str();
  }

  operation->emitError(
      "InsertTemplateAttributes requires pto.target_arch");
  return std::nullopt;
}

static std::optional<std::string>
invokeMetadataHelper(Operation *operation, StringRef pythonExe,
                     StringRef daemonSocketPath, StringRef tileLibPkgPath,
                     StringRef daemonHelperModule) {
  auto pythonPath = pto::resolvePythonExecutable(pythonExe);
  if (!pythonPath) {
    operation->emitError("InsertTemplateAttributes cannot find Python '")
        << pythonExe << "'";
    return std::nullopt;
  }

  auto target = getTargetArch(operation);
  auto operandSpecs = buildOperandSpecsJson(operation);
  if (!target || !operandSpecs)
    return std::nullopt;
  std::string contextAttrs = buildContextAttrsJson(operation);

  llvm::SmallString<128> outputPath;
  int outputFd;
  if (auto error = llvm::sys::fs::createTemporaryFile(
          "tilelib_metadata", "json", outputFd, outputPath)) {
    operation->emitError("InsertTemplateAttributes cannot create temporary "
                         "metadata output: ")
        << error.message();
    return std::nullopt;
  }
  ::close(outputFd);

  llvm::SmallString<128> errorPath;
  int errorFd;
  if (auto error = llvm::sys::fs::createTemporaryFile(
          "tilelib_metadata", "err", errorFd, errorPath)) {
    llvm::sys::fs::remove(outputPath);
    operation->emitError("InsertTemplateAttributes cannot create temporary "
                         "metadata error output: ")
        << error.message();
    return std::nullopt;
  }
  ::close(errorFd);

  std::string opName = operation->getName().getStringRef().str();
  SmallVector<StringRef> args = {
      *pythonPath,       "-m",            daemonHelperModule,
      "--method",        "get_metadata",  "--socket",
      daemonSocketPath,  "--target",      *target,
      "--op",            opName,          "--operand-specs",
      *operandSpecs,
  };
  if (contextAttrs != "{}") {
    args.push_back("--context-attrs");
    args.push_back(contextAttrs);
  }

  std::optional<StringRef> redirects[] = {
      std::nullopt,
      StringRef(outputPath),
      StringRef(errorPath),
  };

  SmallVector<StringRef> environment;
  std::string pythonPathEnvironment;
  std::vector<std::string> environmentStorage;
  bool hasPythonPath = !tileLibPkgPath.empty();
  if (hasPythonPath) {
    const char *existingPath = ::getenv("PYTHONPATH");
    pythonPathEnvironment = "PYTHONPATH=" + tileLibPkgPath.str();
    if (existingPath && existingPath[0] != '\0')
      pythonPathEnvironment += ":" + std::string(existingPath);

    for (char **entry = environ; *entry; ++entry) {
      StringRef value(*entry);
      if (!value.starts_with("PYTHONPATH="))
        environmentStorage.push_back(value.str());
    }
    environmentStorage.push_back(pythonPathEnvironment);
    for (std::string &value : environmentStorage)
      environment.push_back(value);
  }

  std::string errorMessage;
  int result = llvm::sys::ExecuteAndWait(
      *pythonPath, args,
      hasPythonPath
          ? std::optional<llvm::ArrayRef<StringRef>>(environment)
          : std::nullopt,
      redirects, /*secondsToWait=*/30, /*memoryLimit=*/0, &errorMessage);
  if (result != 0) {
    auto errorOutput = llvm::MemoryBuffer::getFile(errorPath);
    llvm::sys::fs::remove(outputPath);
    llvm::sys::fs::remove(errorPath);

    std::string detail;
    if (errorOutput)
      detail = errorOutput.get()->getBuffer().trim().str();
    if (detail.empty())
      detail = errorMessage;
    if (detail.empty())
      detail = "helper exited with status " + std::to_string(result);

    operation->emitError("InsertTemplateAttributes metadata RPC failed: ")
        << detail;
    return std::nullopt;
  }

  auto output = llvm::MemoryBuffer::getFile(outputPath);
  llvm::sys::fs::remove(outputPath);
  llvm::sys::fs::remove(errorPath);
  if (!output) {
    operation->emitError(
        "InsertTemplateAttributes cannot read metadata output");
    return std::nullopt;
  }
  return (*output)->getBuffer().str();
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
  auto *candidates = root ? root->getObject("candidates") : nullptr;
  if (!candidates || candidates->empty()) {
    operation->emitError("InsertTemplateAttributes found no legal template "
                         "candidates for ")
        << operation->getName();
    return failure();
  }

  SmallVector<CandidateMetadata> parsedCandidates;
  parsedCandidates.reserve(candidates->size());
  for (const auto &entry : *candidates) {
    auto *metadata = entry.second.getAsObject();
    if (!metadata) {
      operation->emitError(
          "InsertTemplateAttributes candidate metadata must be an object");
      return failure();
    }

    auto name = metadata->getString("name");
    auto id = metadata->getInteger("id");
    auto loopDepth = metadata->getInteger("loop_depth");
    auto postUpdate = metadata->getBoolean("is_post_update");
    auto tail = metadata->getBoolean("has_tail");
    if (!name || !loopDepth || !postUpdate || !tail) {
      operation->emitError(
          "InsertTemplateAttributes candidate metadata is missing name, "
          "loop_depth, is_post_update, or has_tail");
      return failure();
    }
    if (!id && candidates->size() != 1) {
      operation->emitError(
          "InsertTemplateAttributes requires an id for every "
          "multi-candidate template");
      return failure();
    }

    parsedCandidates.push_back(CandidateMetadata{
        id.value_or(0),
        name->str(),
        *loopDepth,
        *postUpdate,
        *tail,
    });
  }

  llvm::sort(parsedCandidates,
             [](const CandidateMetadata &left,
                const CandidateMetadata &right) {
               if (left.id != right.id)
                 return left.id < right.id;
               return left.name < right.name;
             });
  for (auto [index, candidate] : llvm::enumerate(parsedCandidates)) {
    if (index != 0 && candidate.id == parsedCandidates[index - 1].id) {
      operation->emitError(
          "InsertTemplateAttributes candidate ids must be unique");
      return failure();
    }
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
    if (daemonSocketPath.empty()) {
      module.emitError(
          "InsertTemplateAttributes requires a PTODSL daemon socket");
      return signalPassFailure();
    }

    SmallVector<Operation *> tileOperations;
    module.walk([&](Operation *operation) {
      if (isa<pto::TReshapeOp>(operation))
        return;
      const bool isPipeOperation = isUnifiedInternalPipeOperation(operation);
      if (skipPipeOps && isPipeOperation)
        return;
      if (pipeOnly != isPipeOperation)
        return;
      if (isa<pto::OpPipeInterface>(operation))
        tileOperations.push_back(operation);
    });

    for (Operation *operation : tileOperations) {
      auto metadata = invokeMetadataHelper(
          operation, pythonExe, daemonSocketPath, tileLibPkgPath,
          daemonHelperModule);
      if (!metadata)
        return signalPassFailure();

      auto candidates = parseCandidateAttributes(operation, *metadata);
      if (failed(candidates))
        return signalPassFailure();
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

std::unique_ptr<Pass> createInsertTemplateAttributesPass(
    const InsertTemplateAttributesOptions &options) {
  return std::make_unique<InsertTemplateAttributesPass>(options);
}

} // namespace pto
} // namespace mlir
