// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIToVPTO.cpp - Convert VMI to physical VPTO IR -------------------===//
//===----------------------------------------------------------------------===//

// https://discourse.llvm.org/t/matchandrewrite-hiding-virtual-functions/84933/8
#pragma GCC diagnostic ignored "-Woverloaded-virtual"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VMILayoutSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <numeric>
#include <type_traits>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMITOVPTO
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

std::optional<std::string> getX2MemoryDistToken(Type elementType,
                                                StringRef prefix);
std::optional<std::string> getDenseLaneStrideLoadDistToken(VMIVRegType type);
std::optional<std::string> getDenseLaneStrideStoreDistToken(VMIVRegType type);
std::optional<std::string> getPointStoreDistToken(Type elementType);

bool isVMIType(Type type) { return isa<VMIVRegType, VMIMaskType>(type); }

bool containsVMIType(Type type) {
  if (isVMIType(type))
    return true;

  if (auto functionType = dyn_cast<FunctionType>(type))
    return llvm::any_of(functionType.getInputs(),
                        [](Type input) { return containsVMIType(input); }) ||
           llvm::any_of(functionType.getResults(),
                        [](Type result) { return containsVMIType(result); });

  if (auto shapedType = dyn_cast<ShapedType>(type))
    return containsVMIType(shapedType.getElementType());

  return false;
}

bool hasVMIType(TypeRange types) {
  return llvm::any_of(types, [](Type type) { return containsVMIType(type); });
}

struct VMISupportResult {
  bool supported = true;
  std::string reason;

  static VMISupportResult success() { return {}; }

  static VMISupportResult failure(const Twine &reason) {
    VMISupportResult result;
    result.supported = false;
    result.reason = reason.str();
    return result;
  }

  bool isSupported() const { return supported; }

  LogicalResult toLogicalResult(std::string *outReason = nullptr) const {
    if (supported)
      return mlir::success();
    if (outReason)
      *outReason = reason;
    return mlir::failure();
  }
};

bool hasVMIType(FunctionType type) {
  return hasVMIType(type.getInputs()) || hasVMIType(type.getResults());
}

bool hasVMIType(Attribute attr) {
  if (!attr)
    return false;

  if (auto typeAttr = dyn_cast<TypeAttr>(attr))
    if (containsVMIType(typeAttr.getValue()))
      return true;

  if (auto typedAttr = dyn_cast<TypedAttr>(attr))
    if (containsVMIType(typedAttr.getType()))
      return true;

  if (auto arrayAttr = dyn_cast<ArrayAttr>(attr))
    return llvm::any_of(arrayAttr,
                        [](Attribute element) { return hasVMIType(element); });

  if (auto dictAttr = dyn_cast<DictionaryAttr>(attr))
    return llvm::any_of(dictAttr, [](NamedAttribute namedAttr) {
      return hasVMIType(namedAttr.getValue());
    });

  return false;
}

bool hasVMIType(Operation *op) {
  if (auto func = dyn_cast<func::FuncOp>(op))
    if (hasVMIType(func.getFunctionType()))
      return true;
  if (hasVMIType(op->getOperandTypes()) || hasVMIType(op->getResultTypes()))
    return true;
  for (Region &region : op->getRegions())
    for (Block &block : region)
      if (hasVMIType(block.getArgumentTypes()))
        return true;
  for (NamedAttribute attr : op->getAttrs())
    if (hasVMIType(attr.getValue()))
      return true;
  return false;
}

bool isVMIOp(Operation *op) {
  return op->getName().getStringRef().starts_with("pto.vmi.");
}

StringRef getTruncFRoundModeForResult(Type resultElementType) {
  return pto::isPTOHiFloat8Type(resultElementType) ? "A" : "R";
}

StringRef getTruncFRoundMode(VMITruncFOp op, Type resultElementType) {
  if (auto roundingAttr = op->getAttrOfType<StringAttr>("rounding"))
    return roundingAttr.getValue();
  return getTruncFRoundModeForResult(resultElementType);
}

bool isLayoutAssignedVMIType(Type type) {
  if (auto vregType = dyn_cast<VMIVRegType>(type))
    return static_cast<bool>(vregType.getLayoutAttr());
  if (auto maskType = dyn_cast<VMIMaskType>(type))
    return maskType.getLayoutAttr() &&
           VMIMaskType::isConcreteGranularity(maskType.getGranularity());
  return true;
}

LogicalResult verifyLayoutAssignedVMITypeTree(Operation *op, Type type) {
  if (!isLayoutAssignedVMIType(type))
    return op->emitError() << kVMIDiagPassInvariantPrefix
                           << "vmi-to-vpto requires layout-assigned VMI types";

  if (auto functionType = dyn_cast<FunctionType>(type)) {
    for (Type input : functionType.getInputs())
      if (failed(verifyLayoutAssignedVMITypeTree(op, input)))
        return failure();
    for (Type result : functionType.getResults())
      if (failed(verifyLayoutAssignedVMITypeTree(op, result)))
        return failure();
  }

  if (auto shapedType = dyn_cast<ShapedType>(type))
    return verifyLayoutAssignedVMITypeTree(op, shapedType.getElementType());

  return success();
}

LogicalResult verifyVMIToVPTOInputAttribute(Operation *op, Attribute attr) {
  if (!attr)
    return success();

  if (auto typeAttr = dyn_cast<TypeAttr>(attr))
    if (failed(verifyLayoutAssignedVMITypeTree(op, typeAttr.getValue())))
      return failure();

  if (auto typedAttr = dyn_cast<TypedAttr>(attr))
    if (failed(verifyLayoutAssignedVMITypeTree(op, typedAttr.getType())))
      return failure();

  if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
    for (Attribute element : arrayAttr)
      if (failed(verifyVMIToVPTOInputAttribute(op, element)))
        return failure();
  }

  if (auto dictAttr = dyn_cast<DictionaryAttr>(attr)) {
    for (NamedAttribute namedAttr : dictAttr)
      if (failed(verifyVMIToVPTOInputAttribute(op, namedAttr.getValue())))
        return failure();
  }

  return success();
}

LogicalResult verifyVMIToVPTOInputTypes(Operation *op) {
  for (Type type : op->getOperandTypes())
    if (failed(verifyLayoutAssignedVMITypeTree(op, type)))
      return failure();
  for (Type type : op->getResultTypes())
    if (failed(verifyLayoutAssignedVMITypeTree(op, type)))
      return failure();
  if (auto func = dyn_cast<func::FuncOp>(op)) {
    FunctionType functionType = func.getFunctionType();
    for (Type type : functionType.getInputs())
      if (failed(verifyLayoutAssignedVMITypeTree(op, type)))
        return failure();
    for (Type type : functionType.getResults())
      if (failed(verifyLayoutAssignedVMITypeTree(op, type)))
        return failure();
  }
  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (Type type : block.getArgumentTypes())
        if (failed(verifyLayoutAssignedVMITypeTree(op, type)))
          return failure();
  for (NamedAttribute attr : op->getAttrs())
    if (failed(verifyVMIToVPTOInputAttribute(op, attr.getValue())))
      return failure();
  return success();
}

LogicalResult verifyVMIToVPTOInputIR(ModuleOp module) {
  WalkResult result = module.walk([&](Operation *op) {
    if (failed(verifyVMIToVPTOInputTypes(op)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

static Value materializeVPTOToVMI(OpBuilder &builder, Type resultType,
                                  ValueRange inputs, Location loc) {
  if (!isVMIType(resultType))
    return {};
  return builder.create<VMIPackOp>(loc, resultType, inputs).getResult();
}

static SmallVector<Value> materializeVMIToVPTO(OpBuilder &builder,
                                               TypeRange resultTypes,
                                               ValueRange inputs,
                                               Location loc) {
  if (inputs.size() != 1 || !isVMIType(inputs.front().getType()))
    return {};
  auto unpackOp = builder.create<VMIUnpackOp>(loc, resultTypes, inputs.front());
  return SmallVector<Value>(unpackOp->getResults());
}

static FailureOr<Type> getVMIVRegPhysicalElementType(VMIVRegType type) {
  Type elementType = type.getElementType();
  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout || !layout.hasGroupSlotLaneStride())
    return elementType;

  auto integerType = dyn_cast<IntegerType>(elementType);
  if (!integerType && isa<FloatType>(elementType))
    return elementType;
  if (!integerType)
    return failure();
  if (!integerType.isUnsigned())
    return failure();
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  int64_t laneStride = layout.getLaneStride();
  if (elementBits == 0 || laneStride <= 1)
    return failure();
  if (elementBits == 8 && laneStride == 2)
    return elementType;
  int64_t physicalBits = static_cast<int64_t>(elementBits) * laneStride;
  if (physicalBits != 16 && physicalBits != 32)
    return failure();
  return IntegerType::get(type.getContext(), physicalBits);
}

static int64_t getMaskGranularityBits(StringRef granularity) {
  if (granularity == "b8")
    return 8;
  if (granularity == "b16")
    return 16;
  if (granularity == "b32")
    return 32;
  return 0;
}

static StringRef getMaskGranularityForBits(int64_t bits) {
  switch (bits) {
  case 8:
    return "b8";
  case 16:
    return "b16";
  case 32:
    return "b32";
  default:
    return "";
  }
}

static FailureOr<StringRef> getVMIMaskPhysicalGranularity(VMIMaskType type) {
  int64_t bits = getMaskGranularityBits(type.getGranularity());
  if (bits == 0)
    return failure();

  VMILayoutAttr layout = type.getLayoutAttr();
  int64_t laneStride = layout && layout.hasLaneStride() ? layout.getLaneStride()
                                                        : 1;
  int64_t physicalBits = bits * laneStride;
  StringRef physicalGranularity = getMaskGranularityForBits(physicalBits);
  if (physicalGranularity.empty())
    return failure();
  return physicalGranularity;
}

class VMIToVPTOTypeConverter final : public TypeConverter {
public:
  VMIToVPTOTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](VMIVRegType type,
                     SmallVectorImpl<Type> &results) -> LogicalResult {
      FailureOr<int64_t> arity = getVMIPhysicalArity(type);
      FailureOr<Type> physicalElementType = getVMIVRegPhysicalElementType(type);
      if (failed(arity) || failed(physicalElementType))
        return failure();
      FailureOr<int64_t> lanesPerPart =
          getDataLanesPerPart(*physicalElementType);
      if (failed(lanesPerPart))
        return failure();
      for (int64_t i = 0; i < *arity; ++i)
        results.push_back(VRegType::get(type.getContext(), *lanesPerPart,
                                        *physicalElementType));
      return success();
    });
    addConversion(
        [](VMIMaskType type, SmallVectorImpl<Type> &results) -> LogicalResult {
          FailureOr<int64_t> arity = getVMIPhysicalArity(type);
          FailureOr<StringRef> physicalGranularity =
              getVMIMaskPhysicalGranularity(type);
          if (failed(arity) || failed(physicalGranularity))
            return failure();
          for (int64_t i = 0; i < *arity; ++i)
            results.push_back(
                MaskType::get(type.getContext(), *physicalGranularity));
          return success();
        });
    TypeConverter::addSourceMaterialization(materializeVPTOToVMI);
    TypeConverter::addTargetMaterialization(materializeVMIToVPTO);
  }
};

FailureOr<SmallVector<Type>>
getConvertedResultTypes(Operation *op, unsigned resultIndex,
                        const TypeConverter &typeConverter) {
  if (resultIndex >= op->getNumResults())
    return failure();
  SmallVector<Type> resultTypes;
  if (failed(typeConverter.convertType(op->getResult(resultIndex).getType(),
                                       resultTypes)))
    return failure();
  return resultTypes;
}

FailureOr<SmallVector<Type>>
getConvertedResultTypes(Operation *op, const TypeConverter &typeConverter) {
  SmallVector<Type> resultTypes;
  if (failed(typeConverter.convertTypes(op->getResultTypes(), resultTypes)))
    return failure();
  return resultTypes;
}

FailureOr<SmallVector<Type>>
getConvertedVRegTypesWithLayout(VMIVRegType type, VMILayoutAttr layout,
                                const TypeConverter &typeConverter) {
  auto relayoutType = VMIVRegType::get(type.getContext(), type.getElementCount(),
                                       type.getElementType(), layout);
  SmallVector<Type> convertedTypes;
  if (failed(typeConverter.convertType(relayoutType, convertedTypes)))
    return failure();
  return convertedTypes;
}

FailureOr<int64_t> getVRegPhysicalFootprintBytes(TypeRange types) {
  int64_t totalBytes = 0;
  for (Type type : types) {
    auto vregType = dyn_cast<VRegType>(type);
    if (!vregType)
      return failure();
    unsigned elementBits =
        pto::getPTOStorageElemBitWidth(vregType.getElementType());
    if (elementBits == 0)
      return failure();
    int64_t chunkBits = vregType.getElementCount() * elementBits;
    if (chunkBits % 8 != 0)
      return failure();
    totalBytes += chunkBits / 8;
  }
  return totalBytes;
}

FailureOr<bool> hasNoWiderFootprintThanContiguous(TypeRange assignedTypes,
                                                  TypeRange contiguousTypes) {
  FailureOr<int64_t> assignedBytes =
      getVRegPhysicalFootprintBytes(assignedTypes);
  FailureOr<int64_t> contiguousBytes =
      getVRegPhysicalFootprintBytes(contiguousTypes);
  if (failed(assignedBytes) || failed(contiguousBytes))
    return failure();
  return *assignedBytes <= *contiguousBytes;
}

void replaceOpWithFlatConvertedValues(
    ConversionPatternRewriter &rewriter, Operation *op, ValueRange flatValues,
    const TypeConverter &typeConverter) {
  SmallVector<SmallVector<Value>> replacements;
  replacements.reserve(op->getNumResults());

  auto valueIt = flatValues.begin();
  for (OpResult result : op->getResults()) {
    SmallVector<Type> convertedTypes;
    LogicalResult converted =
        typeConverter.convertType(result.getType(), convertedTypes);
    assert(succeeded(converted) && "expected converted result types");
    (void)converted;
    assert(std::distance(valueIt, flatValues.end()) >=
               static_cast<ptrdiff_t>(convertedTypes.size()) &&
           "not enough replacement values for converted results");
    replacements.emplace_back(valueIt, valueIt + convertedTypes.size());
    valueIt += convertedTypes.size();
  }
  assert(valueIt == flatValues.end() &&
         "too many replacement values for converted results");

  rewriter.replaceOpWithMultiple(op, std::move(replacements));
}

SmallVector<Value>
flattenOneToNOperands(ArrayRef<ValueRange> operands) {
  SmallVector<Value> flat;
  for (ValueRange operand : operands)
    llvm::append_range(flat, operand);
  return flat;
}

bool isIdentityOneToNValueMapping(ValueRange originalValues,
                                  ArrayRef<ValueRange> convertedValues) {
  if (originalValues.size() != convertedValues.size())
    return false;
  for (auto [original, converted] :
       llvm::zip_equal(originalValues, convertedValues)) {
    if (converted.size() != 1 || converted.front() != original)
      return false;
  }
  return true;
}

TypeRange getConvertedSignatureTypes(
    const TypeConverter::SignatureConversion &conversion,
    unsigned originalIndex) {
  TypeRange convertedTypes = conversion.getConvertedTypes();
  if (auto mapping = conversion.getInputMapping(originalIndex))
    return convertedTypes.slice(mapping->inputNo, mapping->size);
  return {};
}

bool hasNonIdentitySignatureConversion(
    TypeRange originalTypes,
    const TypeConverter::SignatureConversion &conversion) {
  for (auto [index, originalType] : llvm::enumerate(originalTypes)) {
    TypeRange convertedTypes = getConvertedSignatureTypes(conversion, index);
    if (convertedTypes.size() != 1 || convertedTypes.front() != originalType)
      return true;
  }
  return false;
}

FailureOr<Value> createAllTrueMaskForVReg(Location loc, VRegType vregType,
                                          PatternRewriter &rewriter) {
  MLIRContext *ctx = rewriter.getContext();
  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(vregType.getElementType());
  if (elementBits == 8)
    return rewriter
        .create<PsetB8Op>(loc, MaskType::get(ctx, "b8"),
                          rewriter.getStringAttr("PAT_ALL"))
        .getResult();
  if (elementBits == 16)
    return rewriter
        .create<PsetB16Op>(loc, MaskType::get(ctx, "b16"),
                           rewriter.getStringAttr("PAT_ALL"))
        .getResult();
  if (elementBits == 32)
    return rewriter
        .create<PsetB32Op>(loc, MaskType::get(ctx, "b32"),
                           rewriter.getStringAttr("PAT_ALL"))
        .getResult();
  return failure();
}

FailureOr<MaskType> getMaskTypeForVReg(VRegType vregType, MLIRContext *ctx) {
  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(vregType.getElementType());
  if (elementBits == 8)
    return MaskType::get(ctx, "b8");
  if (elementBits == 16)
    return MaskType::get(ctx, "b16");
  if (elementBits == 32)
    return MaskType::get(ctx, "b32");
  return failure();
}

FailureOr<Value> createAllTrueMask(Location loc, MaskType maskType,
                                   PatternRewriter &rewriter) {
  StringAttr pattern = rewriter.getStringAttr("PAT_ALL");
  MLIRContext *ctx = rewriter.getContext();
  if (maskType.isB8())
    return rewriter.create<PsetB8Op>(loc, MaskType::get(ctx, "b8"), pattern)
        .getResult();
  if (maskType.isB16())
    return rewriter.create<PsetB16Op>(loc, MaskType::get(ctx, "b16"), pattern)
        .getResult();
  if (maskType.isB32())
    return rewriter.create<PsetB32Op>(loc, MaskType::get(ctx, "b32"), pattern)
        .getResult();
  return failure();
}

FailureOr<Value> createPatternMask(Location loc, MaskType maskType,
                                   StringRef pattern,
                                   PatternRewriter &rewriter) {
  StringAttr patternAttr = rewriter.getStringAttr(pattern);
  MLIRContext *ctx = rewriter.getContext();
  if (maskType.isB8())
    return rewriter.create<PsetB8Op>(loc, MaskType::get(ctx, "b8"), patternAttr)
        .getResult();
  if (maskType.isB16())
    return rewriter
        .create<PsetB16Op>(loc, MaskType::get(ctx, "b16"), patternAttr)
        .getResult();
  if (maskType.isB32())
    return rewriter
        .create<PsetB32Op>(loc, MaskType::get(ctx, "b32"), patternAttr)
        .getResult();
  return failure();
}

FailureOr<Value> createPrefixMask(Location loc, MaskType maskType,
                                  StringRef pattern,
                                  PatternRewriter &rewriter) {
  StringAttr patternAttr = rewriter.getStringAttr(pattern);
  MLIRContext *ctx = rewriter.getContext();
  if (maskType.isB8())
    return rewriter.create<PgeB8Op>(loc, MaskType::get(ctx, "b8"), patternAttr)
        .getResult();
  if (maskType.isB16())
    return rewriter
        .create<PgeB16Op>(loc, MaskType::get(ctx, "b16"), patternAttr)
        .getResult();
  if (maskType.isB32())
    return rewriter
        .create<PgeB32Op>(loc, MaskType::get(ctx, "b32"), patternAttr)
        .getResult();
  return failure();
}

bool areEquivalentReductionMasks(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  if (lhs.getType() != rhs.getType())
    return false;

  Operation *lhsOp = lhs.getDefiningOp();
  Operation *rhsOp = rhs.getDefiningOp();
  if (!lhsOp || !rhsOp || lhsOp->getName() != rhsOp->getName())
    return false;

  bool isPatternMask =
      isa<PsetB8Op, PsetB16Op, PsetB32Op, PgeB8Op, PgeB16Op, PgeB32Op>(
          lhsOp);
  return isPatternMask && lhsOp->getAttr("pattern") == rhsOp->getAttr("pattern");
}

bool haveEquivalentReductionMasks(ValueRange masks) {
  return !masks.empty() &&
         llvm::all_of(masks.drop_front(), [&](Value mask) {
           return areEquivalentReductionMasks(masks.front(), mask);
         });
}

template <typename CombineOpTy>
FailureOr<Value> combineEquivalentMaskedParts(
    Location loc, ValueRange sources, ValueRange masks, VRegType resultType,
    PatternRewriter &rewriter) {
  if (sources.empty() || sources.size() != masks.size() ||
      !haveEquivalentReductionMasks(masks))
    return failure();

  Value combined = sources.front();
  for (Value source : sources.drop_front())
    combined =
        rewriter
            .create<CombineOpTy>(loc, resultType, combined, source,
                                 masks.front())
            .getResult();
  return combined;
}

FailureOr<std::pair<Value, Value>>
createRuntimePrefixMask(Location loc, MaskType maskType, Value activeLanes,
                        PatternRewriter &rewriter) {
  MLIRContext *ctx = rewriter.getContext();
  Type scalarType = activeLanes.getType();
  if (maskType.isB8()) {
    auto op = rewriter.create<PltB8Op>(loc, MaskType::get(ctx, "b8"),
                                       scalarType, activeLanes);
    return std::make_pair(Value(op.getMask()), Value(op.getScalarOut()));
  }
  if (maskType.isB16()) {
    auto op = rewriter.create<PltB16Op>(loc, MaskType::get(ctx, "b16"),
                                        scalarType, activeLanes);
    return std::make_pair(Value(op.getMask()), Value(op.getScalarOut()));
  }
  if (maskType.isB32()) {
    auto op = rewriter.create<PltB32Op>(loc, MaskType::get(ctx, "b32"),
                                        scalarType, activeLanes);
    return std::make_pair(Value(op.getMask()), Value(op.getScalarOut()));
  }
  return failure();
}

LogicalResult
checkSupportedMaskableVReg(VMIVRegType type, std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(type.getElementType());
  FailureOr<int64_t> arity = getVMIPhysicalArity(type);
  if (failed(lanesPerPart) || failed(arity) || *arity < 1)
    return fail("requires computable non-empty physical vreg parts");

  return success();
}

Value createI32Constant(Location loc, int64_t value,
                        PatternRewriter &rewriter) {
  return rewriter.create<arith::ConstantIntOp>(loc, value, 32);
}

Value createI16Constant(Location loc, int64_t value,
                        PatternRewriter &rewriter) {
  return rewriter.create<arith::ConstantIntOp>(loc, value, 16);
}

FailureOr<Value> createPrefixMaskForActiveLanes(Location loc, MaskType maskType,
                                                int64_t activeLanes,
                                                PatternRewriter &rewriter) {
  if (activeLanes <= 0)
    return createPrefixMask(loc, maskType, "PAT_ALLF", rewriter);

  switch (activeLanes) {
  case 1:
  case 2:
  case 3:
  case 4:
  case 8:
  case 16:
  case 32:
  case 64:
  case 128:
    return createPrefixMask(
        loc, maskType, (Twine("PAT_VL") + Twine(activeLanes)).str(), rewriter);
  default: {
    FailureOr<std::pair<Value, Value>> dynamicMask = createRuntimePrefixMask(
        loc, maskType, createI32Constant(loc, activeLanes, rewriter), rewriter);
    if (failed(dynamicMask))
      return failure();
    return dynamicMask->first;
  }
  }
}

Value clampDynamicActiveLanes(Location loc, Value activeLanes,
                              int64_t maxActiveLanes,
                              PatternRewriter &rewriter) {
  Value activeI32 = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getI32Type(), activeLanes);
  Value zeroI32 = createI32Constant(loc, 0, rewriter);
  Value nonNegative = rewriter.create<arith::MaxSIOp>(loc, activeI32, zeroI32);
  Value maxI32 = createI32Constant(loc, maxActiveLanes, rewriter);
  return rewriter.create<arith::MinUIOp>(loc, nonNegative, maxI32);
}

Value createPartitionActiveLanes(Location loc, Value activeLanesI32,
                                 int64_t factor, int64_t part,
                                 PatternRewriter &rewriter) {
  if (factor == 1)
    return activeLanesI32;
  int64_t bias = factor - 1 - part;
  Value biased = activeLanesI32;
  if (bias != 0)
    biased = rewriter.create<arith::AddIOp>(
        loc, biased, createI32Constant(loc, bias, rewriter));
  return rewriter.create<arith::DivUIOp>(
      loc, biased, createI32Constant(loc, factor, rewriter));
}

std::optional<int64_t> getPowerOfTwoLog2(int64_t value) {
  if (value <= 0 || (value & (value - 1)) != 0)
    return std::nullopt;
  int64_t log2 = 0;
  while (value > 1) {
    value >>= 1;
    ++log2;
  }
  return log2;
}

std::optional<std::string> getPrefixPattern(int64_t activeLanes,
                                            int64_t lanesPerPart) {
  if (activeLanes <= 0)
    return std::string("PAT_ALLF");
  if (activeLanes >= lanesPerPart)
    return std::string("PAT_ALL");
  switch (activeLanes) {
  case 1:
  case 2:
  case 3:
  case 4:
  case 8:
  case 16:
  case 32:
  case 64:
  case 128:
    return std::string("PAT_VL") + std::to_string(activeLanes);
  default:
    return std::nullopt;
  }
}

FailureOr<Value> getSingleValue(Operation *op, ValueRange values,
                                StringRef description,
                                PatternRewriter &rewriter) {
  if (values.size() != 1) {
    (void)rewriter.notifyMatchFailure(op, description);
    return failure();
  }
  return values.front();
}

static int64_t ceilDivNonNegative(int64_t lhs, int64_t rhs) {
  return (lhs + rhs - 1) / rhs;
}

FailureOr<int64_t> getDataLayoutFactor(VMIVRegType type) {
  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout)
    return failure();
  return layout.isDenseSplit() ? layout.getFactor() : 1;
}

FailureOr<int64_t> getDataChunksInPart(VMIVRegType type, int64_t part) {
  FailureOr<int64_t> factor = getDataLayoutFactor(type);
  FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(type.getElementType());
  if (failed(factor) || failed(lanesPerPart) || part < 0 || part >= *factor)
    return failure();

  int64_t logicalLanesInPart =
      (type.getElementCount() + *factor - 1 - part) / *factor;
  return ceilDivNonNegative(logicalLanesInPart, *lanesPerPart);
}

FailureOr<int64_t> getDataFlatPartIndex(VMIVRegType type, int64_t part,
                                        int64_t chunk) {
  FailureOr<int64_t> factor = getDataLayoutFactor(type);
  if (failed(factor) || part < 0 || part >= *factor || chunk < 0)
    return failure();

  int64_t flatIndex = 0;
  for (int64_t currentPart = 0; currentPart < part; ++currentPart) {
    FailureOr<int64_t> chunks = getDataChunksInPart(type, currentPart);
    if (failed(chunks))
      return failure();
    flatIndex += *chunks;
  }

  FailureOr<int64_t> chunks = getDataChunksInPart(type, part);
  if (failed(chunks) || chunk >= *chunks)
    return failure();
  return flatIndex + chunk;
}

FailureOr<int64_t> checkFullDataPhysicalChunks(VMIVRegType type,
                                               std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<int64_t> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(type.getElementType());
  if (failed(lanesPerPart))
    return fail("requires known physical lanes per part");

  FailureOr<int64_t> factor = getDataLayoutFactor(type);
  if (failed(factor))
    return fail("requires assigned layout");

  for (int64_t part = 0; part < *factor; ++part) {
    FailureOr<int64_t> chunks = getDataChunksInPart(type, part);
    if (failed(chunks))
      return fail("requires known physical chunks");
    for (int64_t chunk = 0; chunk < *chunks; ++chunk) {
      for (int64_t lane = 0; lane < *lanesPerPart; ++lane) {
        FailureOr<bool> padding = isPaddingLane(type, part, chunk, lane);
        if (failed(padding))
          return fail("failed to map physical padding lane");
        if (*padding)
          return fail("found padding lane in physical chunk");
      }
    }
  }

  return *lanesPerPart;
}

FailureOr<int64_t> getVMITypeLayoutFactor(Type type) {
  Attribute layout;
  if (auto vregType = dyn_cast<VMIVRegType>(type))
    layout = vregType.getLayout();
  else if (auto maskType = dyn_cast<VMIMaskType>(type))
    layout = maskType.getLayout();
  else
    return failure();

  auto layoutAttr = dyn_cast_or_null<VMILayoutAttr>(layout);
  if (!layoutAttr)
    return failure();
  return layoutAttr.isDenseSplit() ? layoutAttr.getFactor() : 1;
}

FailureOr<int64_t> getVMITypeElementCount(Type type) {
  if (auto vregType = dyn_cast<VMIVRegType>(type))
    return vregType.getElementCount();
  if (auto maskType = dyn_cast<VMIMaskType>(type))
    return maskType.getElementCount();
  return failure();
}

FailureOr<int64_t> getVMITypeLanesPerPart(Type type) {
  if (auto vregType = dyn_cast<VMIVRegType>(type)) {
    FailureOr<Type> physicalElementType =
        getVMIVRegPhysicalElementType(vregType);
    if (failed(physicalElementType))
      return failure();
    return getDataLanesPerPart(*physicalElementType);
  }
  if (auto maskType = dyn_cast<VMIMaskType>(type)) {
    FailureOr<StringRef> physicalGranularity =
        getVMIMaskPhysicalGranularity(maskType);
    if (failed(physicalGranularity))
      return failure();
    return getMaskLanesPerPart(*physicalGranularity);
  }
  return failure();
}

FailureOr<int64_t> getVMITypeChunksInPart(Type type, int64_t part) {
  FailureOr<int64_t> elementCount = getVMITypeElementCount(type);
  FailureOr<int64_t> factor = getVMITypeLayoutFactor(type);
  FailureOr<int64_t> lanesPerPart = getVMITypeLanesPerPart(type);
  if (failed(elementCount) || failed(factor) || failed(lanesPerPart) ||
      part < 0 || part >= *factor)
    return failure();

  VMILayoutAttr layout;
  if (auto vregType = dyn_cast<VMIVRegType>(type))
    layout = vregType.getLayoutAttr();
  else if (auto maskType = dyn_cast<VMIMaskType>(type))
    layout = maskType.getLayoutAttr();
  if (!layout)
    return failure();

  int64_t logicalLanesInPart = (*elementCount + *factor - 1 - part) / *factor;
  int64_t laneStride = 1;
  if (isa<VMIVRegType>(type) && layout.isDense())
    laneStride = layout.getLaneStride();
  int64_t physicalLanes =
      logicalLanesInPart == 0 ? 0 : (logicalLanesInPart - 1) * laneStride + 1;
  return ceilDivNonNegative(physicalLanes, *lanesPerPart);
}

LogicalResult checkFullVMIPhysicalChunks(Type type, std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  FailureOr<int64_t> factor = getVMITypeLayoutFactor(type);
  FailureOr<int64_t> lanesPerPart = getVMITypeLanesPerPart(type);
  if (failed(factor) || failed(lanesPerPart))
    return fail("requires assigned layout with known physical lanes per part");

  for (int64_t part = 0; part < *factor; ++part) {
    FailureOr<int64_t> chunks = getVMITypeChunksInPart(type, part);
    if (failed(chunks))
      return fail("requires known physical chunks");
    for (int64_t chunk = 0; chunk < *chunks; ++chunk) {
      for (int64_t lane = 0; lane < *lanesPerPart; ++lane) {
        FailureOr<bool> padding = isPaddingLane(type, part, chunk, lane);
        if (failed(padding))
          return fail("failed to map physical padding lane");
        if (*padding)
          return fail("found padding lane in physical chunk");
      }
    }
  }

  return success();
}

FailureOr<int64_t> getContiguousMaterializationPartCount(Type type,
                                                         std::string *reason);

FailureOr<int64_t> getContiguousMaterializationPartCount(Type type,
                                                         std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<int64_t> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  FailureOr<int64_t> arity = getVMIPhysicalArity(type);
  FailureOr<int64_t> factor = getVMITypeLayoutFactor(type);
  if (failed(arity) || failed(factor))
    return fail("requires computable physical arity and assigned layout");

  Attribute layoutAttr;
  if (auto vregType = dyn_cast<VMIVRegType>(type))
    layoutAttr = vregType.getLayout();
  else if (auto maskType = dyn_cast<VMIMaskType>(type))
    layoutAttr = maskType.getLayout();
  else
    return fail("requires VMI data or mask type");

  auto layout = dyn_cast_or_null<VMILayoutAttr>(layoutAttr);
  if (!layout)
    return fail("requires assigned layout");
  if (layout.isContiguous() && layout.getLaneStride() == 1)
    return *arity;
  if (!layout.isDenseSplit() ||
      (layout.getFactor() != 2 && layout.getFactor() != 4))
    return fail("requires contiguous, deinterleaved=2/4, or "
                "block_deinterleaved=2/4 layout");

  FailureOr<int64_t> chunksPerGroup = getVMITypeChunksInPart(type, 0);
  if (failed(chunksPerGroup))
    return fail("requires known physical chunks per part");
  if (*chunksPerGroup == 0)
    return fail("requires at least one physical chunk per part");

  for (int64_t part = 1; part < *factor; ++part) {
    FailureOr<int64_t> chunks = getVMITypeChunksInPart(type, part);
    if (failed(chunks))
      return fail("requires known physical chunks per part");
    if (layout.getFactor() == 2 && *chunks != *chunksPerGroup)
      return fail("requires every deinterleaved part to have the same "
                  "physical chunk count");
  }

  VMILayoutAttr contiguous = VMILayoutAttr::getContiguous(type.getContext());
  Type contiguousType;
  if (auto vregType = dyn_cast<VMIVRegType>(type)) {
    contiguousType =
        VMIVRegType::get(type.getContext(), vregType.getElementCount(),
                         vregType.getElementType(), contiguous);
  } else {
    auto maskType = cast<VMIMaskType>(type);
    contiguousType =
        VMIMaskType::get(type.getContext(), maskType.getElementCount(),
                         maskType.getGranularity(), contiguous);
  }
  return getVMIPhysicalArity(contiguousType);
}

LogicalResult checkCanMaterializeToContiguous(Type type, std::string *reason) {
  return succeeded(getContiguousMaterializationPartCount(type, reason))
             ? success()
             : failure();
}

std::optional<int64_t> getConstantIndexValue(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto integerAttr = dyn_cast<IntegerAttr>(constant.getValue()))
      return integerAttr.getInt();
  }
  return std::nullopt;
}

static int64_t normalizeRemainder(int64_t value, int64_t modulus) {
  int64_t remainder = value % modulus;
  return remainder < 0 ? remainder + modulus : remainder;
}

std::optional<int64_t> getKnownIndexRemainder(Value value, int64_t modulus,
                                              int depth = 0) {
  if (modulus <= 1)
    return 0;
  if (depth > 6)
    return std::nullopt;
  if (std::optional<int64_t> constant = getConstantIndexValue(value))
    return normalizeRemainder(*constant, modulus);

  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    return getKnownIndexRemainder(cast.getIn(), modulus, depth + 1);
  if (auto cast = value.getDefiningOp<arith::ExtSIOp>())
    return getKnownIndexRemainder(cast.getIn(), modulus, depth + 1);
  if (auto cast = value.getDefiningOp<arith::ExtUIOp>())
    return getKnownIndexRemainder(cast.getIn(), modulus, depth + 1);
  if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() == 1 && cast->getNumResults() == 1)
      return getKnownIndexRemainder(cast.getOperand(0), modulus, depth + 1);
  }

  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    std::optional<int64_t> lhs =
        getKnownIndexRemainder(add.getLhs(), modulus, depth + 1);
    std::optional<int64_t> rhs =
        getKnownIndexRemainder(add.getRhs(), modulus, depth + 1);
    if (lhs && rhs)
      return normalizeRemainder(*lhs + *rhs, modulus);
    return std::nullopt;
  }
  if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    std::optional<int64_t> lhs =
        getKnownIndexRemainder(sub.getLhs(), modulus, depth + 1);
    std::optional<int64_t> rhs =
        getKnownIndexRemainder(sub.getRhs(), modulus, depth + 1);
    if (lhs && rhs)
      return normalizeRemainder(*lhs - *rhs, modulus);
    return std::nullopt;
  }
  if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    std::optional<int64_t> lhs =
        getKnownIndexRemainder(mul.getLhs(), modulus, depth + 1);
    std::optional<int64_t> rhs =
        getKnownIndexRemainder(mul.getRhs(), modulus, depth + 1);
    if ((lhs && *lhs == 0) || (rhs && *rhs == 0))
      return 0;
    if (lhs && rhs)
      return normalizeRemainder(*lhs * *rhs, modulus);
    return std::nullopt;
  }

  return std::nullopt;
}

std::optional<int64_t> getKnownPointerByteRemainder(Value pointer,
                                                    int64_t alignmentBytes,
                                                    int depth = 0) {
  if (alignmentBytes <= 1)
    return 0;
  if (depth > 6)
    return std::nullopt;

  // A raw PTO pointer block argument is a base address. Its required
  // address-space alignment is an ABI precondition; derived pointers must
  // prove that their element offsets preserve that alignment.
  if (isa<BlockArgument>(pointer))
    return 0;

  if (auto cast = pointer.getDefiningOp<CastPtrOp>()) {
    Value input = cast.getInput();
    if (isa<PtrType>(input.getType()))
      return getKnownPointerByteRemainder(input, alignmentBytes, depth + 1);
    if (isa<IntegerType>(input.getType()))
      return getKnownIndexRemainder(input, alignmentBytes, depth + 1);
    return std::nullopt;
  }

  if (auto cast = pointer.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() == 1 && cast->getNumResults() == 1)
      return getKnownPointerByteRemainder(cast.getOperand(0), alignmentBytes,
                                          depth + 1);
    return std::nullopt;
  }

  if (auto add = pointer.getDefiningOp<AddPtrOp>()) {
    std::optional<int64_t> base =
        getKnownPointerByteRemainder(add.getPtr(), alignmentBytes, depth + 1);
    auto pointerType = dyn_cast<PtrType>(add.getPtr().getType());
    if (!base || !pointerType)
      return std::nullopt;
    unsigned elementBits =
        pto::getPTOStorageElemBitWidth(pointerType.getElementType());
    if (elementBits == 0 || elementBits % 8 != 0)
      return std::nullopt;
    int64_t elementBytes = elementBits / 8;
    int64_t offsetModulus =
        alignmentBytes / std::gcd(alignmentBytes, elementBytes);
    std::optional<int64_t> offset =
        getKnownIndexRemainder(add.getOffset(), offsetModulus, depth + 1);
    if (!offset)
      return std::nullopt;
    return normalizeRemainder(*base + *offset * elementBytes, alignmentBytes);
  }

  return std::nullopt;
}

bool isKnown32ByteAlignedAddress(Value pointer, Value elementOffset,
                                 Type elementType) {
  constexpr int64_t alignmentBytes = 32;
  std::optional<int64_t> base =
      getKnownPointerByteRemainder(pointer, alignmentBytes);
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  if (!base || elementBits == 0 || elementBits % 8 != 0)
    return false;

  int64_t elementBytes = elementBits / 8;
  int64_t offsetModulus =
      alignmentBytes / std::gcd(alignmentBytes, elementBytes);
  std::optional<int64_t> offset =
      getKnownIndexRemainder(elementOffset, offsetModulus);
  if (!offset)
    return false;
  return normalizeRemainder(*base + *offset * elementBytes, alignmentBytes) ==
         0;
}

FailureOr<int64_t> getStaticMemRefElementCount(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  if (!memrefType || !memrefType.hasStaticShape())
    return failure();

  int64_t elements = 1;
  for (int64_t dim : memrefType.getShape())
    elements *= dim;
  return elements;
}

static Type getMemoryElementType(Type type) {
  if (auto ptrType = dyn_cast<PtrType>(type))
    return ptrType.getElementType();
  if (auto memrefType = dyn_cast<MemRefType>(type))
    return memrefType.getElementType();
  return {};
}

static bool isPackedByteGroupStore(Type destinationType, VRegType valueType) {
  Type destinationElementType = getMemoryElementType(destinationType);
  auto destinationIntegerType =
      dyn_cast_or_null<IntegerType>(destinationElementType);
  auto valueIntegerType = dyn_cast<IntegerType>(valueType.getElementType());
  return destinationIntegerType && valueIntegerType &&
         pto::getPTOStorageElemBitWidth(destinationIntegerType) == 8 &&
         pto::getPTOStorageElemBitWidth(valueIntegerType) == 32;
}

enum class VMIMemoryValidMaskKind {
  AllTrue,
  ExplicitMask,
};

enum class VMIMemoryWriteMaskKind {
  AllTrue,
  ExplicitMask,
};

enum class VMIMemoryPermutationKind {
  Identity,
};

enum class VMIMemoryFallbackDecisionKind {
  NotRequired,
  RequiredUnavailable,
};

struct VMIMemoryLogicalShape {
  int64_t elementCount = 0;
};

struct VMIMemoryLaneAddressMap {
  VMIMemoryPermutationKind permutation = VMIMemoryPermutationKind::Identity;
  int64_t baseElementOffset = 0;
  int64_t elementStride = 1;
  int64_t physicalLaneFootprint = 0;

  int64_t getExclusiveEndElement() const {
    return baseElementOffset + physicalLaneFootprint * elementStride;
  }
};

struct VMIMemoryFallbackDecision {
  VMIMemoryFallbackDecisionKind kind =
      VMIMemoryFallbackDecisionKind::NotRequired;
  std::string reason = "not required";

  static VMIMemoryFallbackDecision notRequired() { return {}; }

  static VMIMemoryFallbackDecision requiredUnavailable(const Twine &reason) {
    VMIMemoryFallbackDecision decision;
    decision.kind = VMIMemoryFallbackDecisionKind::RequiredUnavailable;
    decision.reason = reason.str();
    return decision;
  }
};

struct VMIMemorySafeReadProof {
  bool proven = false;
  std::string reason;
  std::optional<int64_t> constantOffset;
  std::optional<int64_t> staticElementCount;
  std::optional<VMIMemoryLaneAddressMap> laneAddressMap;
  int64_t physicalFootprint = 0;
};

struct VMIMemoryAccessPlan {
  Type baseType;
  VMIVRegType valueType;
  std::optional<int64_t> constantOffset;
  VMIMemoryLogicalShape logicalShape;
  VMIMemoryValidMaskKind validMask = VMIMemoryValidMaskKind::AllTrue;
  VMIMemoryPermutationKind permutation = VMIMemoryPermutationKind::Identity;
  std::optional<VMIMemoryLaneAddressMap> laneAddressMap;
  Attribute paddingValue;
  VMIMemoryWriteMaskKind writeMask = VMIMemoryWriteMaskKind::AllTrue;
  VMIMemorySafeReadProof safeReadProof;
  VMISupportResult layoutSupport;
  VMIMemoryFallbackDecision fallbackDecision;
};

FailureOr<VMIMemoryLaneAddressMap>
buildContiguousIdentityLaneAddressMap(int64_t constantOffset,
                                      VMIVRegType resultType,
                                      std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) -> FailureOr<VMIMemoryLaneAddressMap> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(resultType.getElementType());
  FailureOr<int64_t> arity = getVMIPhysicalArity(resultType);
  if (failed(lanesPerPart) || failed(arity))
    return fail("requires computable physical read footprint");

  VMIMemoryLaneAddressMap map;
  map.baseElementOffset = constantOffset;
  map.physicalLaneFootprint = *arity * *lanesPerPart;
  return map;
}

VMISupportResult requireIdentityMemRefLayout(Type memoryType, StringRef role,
                                             Value memoryValue = {}) {
  auto memrefType = dyn_cast<MemRefType>(memoryType);
  if (!memrefType || memrefType.getLayout().isIdentity())
    return VMISupportResult::success();
  std::string reason =
      (Twine(role) +
       " memref layout is non-identity; current VMI memory access plan "
       "supports only contiguous identity lane-to-address maps")
          .str();
  if (memoryValue && memoryValue.getDefiningOp<memref::SubViewOp>())
    reason += "; memref.subview requires normalized base/offset/stride "
              "lane-to-address planning";
  return VMISupportResult::failure(reason);
}

VMIMemorySafeReadProof
computeSafeFullReadProof(Type sourceType, std::optional<int64_t> constantOffset,
                         VMIVRegType resultType) {
  VMIMemorySafeReadProof proof;
  proof.constantOffset = constantOffset;

  auto fail = [&](const Twine &message) {
    proof.proven = false;
    proof.reason = message.str();
    return proof;
  };

  if (!constantOffset)
    return fail("requires constant index offset");

  FailureOr<int64_t> staticElements = getStaticMemRefElementCount(sourceType);
  if (failed(staticElements))
    return fail("requires statically shaped memref source");
  int64_t elements = *staticElements;
  proof.staticElementCount = elements;

  if (*constantOffset < 0)
    return fail("requires non-negative offset");

  std::string addressMapReason;
  FailureOr<VMIMemoryLaneAddressMap> addressMap =
      buildContiguousIdentityLaneAddressMap(*constantOffset, resultType,
                                            &addressMapReason);
  if (failed(addressMap))
    return fail(addressMapReason);
  proof.laneAddressMap = *addressMap;

  proof.physicalFootprint = addressMap->physicalLaneFootprint;
  if (addressMap->getExclusiveEndElement() > elements)
    return fail(Twine("full physical read footprint [") +
                Twine(addressMap->baseElementOffset) + ", " +
                Twine(addressMap->getExclusiveEndElement()) +
                ") exceeds static memref element count " + Twine(elements));

  proof.proven = true;
  return proof;
}

VMIMemoryAccessPlan
buildReadAccessPlan(Value source, Type sourceType, VMIVRegType resultType,
                    std::optional<int64_t> constantOffset,
                    VMIMemoryValidMaskKind validMask) {
  VMIMemoryAccessPlan plan;
  plan.baseType = sourceType;
  plan.valueType = resultType;
  plan.constantOffset = constantOffset;
  plan.logicalShape.elementCount = resultType.getElementCount();
  plan.validMask = validMask;
  plan.permutation = VMIMemoryPermutationKind::Identity;
  plan.writeMask = VMIMemoryWriteMaskKind::AllTrue;
  plan.safeReadProof =
      computeSafeFullReadProof(sourceType, constantOffset, resultType);
  plan.laneAddressMap = plan.safeReadProof.laneAddressMap;
  plan.layoutSupport =
      requireIdentityMemRefLayout(sourceType, "source", source);
  return plan;
}

VMIMemoryAccessPlan
buildWriteAccessPlan(Value destination, Type destinationType,
                     VMIVRegType valueType, VMIMemoryWriteMaskKind writeMask) {
  VMIMemoryAccessPlan plan;
  plan.baseType = destinationType;
  plan.valueType = valueType;
  plan.logicalShape.elementCount = valueType.getElementCount();
  plan.validMask = VMIMemoryValidMaskKind::AllTrue;
  plan.permutation = VMIMemoryPermutationKind::Identity;
  plan.writeMask = writeMask;
  plan.layoutSupport =
      requireIdentityMemRefLayout(destinationType, "destination", destination);
  return plan;
}

void requireUnavailableReadFallback(VMIMemoryAccessPlan &plan) {
  std::string maskedLoadReason;
  if (plan.validMask == VMIMemoryValidMaskKind::ExplicitMask)
    maskedLoadReason =
        "; target true masked/non-faulting load is unavailable because the "
        "current VPTO pto.vlds surface has no mask operand";
  std::string scratchReason =
      "; scratch memory fallback resource allocation is not implemented";
  std::string guardedReason =
      "; guarded memory fallback control-flow lowering is not implemented";
  plan.fallbackDecision = VMIMemoryFallbackDecision::requiredUnavailable(
      Twine("partial/tail read needs a scratch, guarded, or true "
            "masked/non-faulting load fallback, but no such fallback resource "
            "plan is implemented") +
      maskedLoadReason + scratchReason + guardedReason);
}

FailureOr<int64_t> verifyFullOrSafeReadVRegChunks(Operation *op,
                                                  VMIVRegType type,
                                                  Type sourceType, Value offset,
                                                  PatternRewriter &rewriter) {
  std::string fullChunkReason;
  FailureOr<int64_t> lanesPerPart =
      checkFullDataPhysicalChunks(type, &fullChunkReason);
  if (succeeded(lanesPerPart))
    return *lanesPerPart;

  VMIMemorySafeReadProof safeReadProof =
      computeSafeFullReadProof(sourceType, getConstantIndexValue(offset), type);
  if (safeReadProof.proven) {
    lanesPerPart = getDataLanesPerPart(type.getElementType());
    if (succeeded(lanesPerPart))
      return *lanesPerPart;
  }

  lanesPerPart = getDataLanesPerPart(type.getElementType());
  if (succeeded(lanesPerPart))
    return *lanesPerPart;

  (void)rewriter.notifyMatchFailure(
      op, Twine("memory lowering ") + fullChunkReason +
              "; safe full-read proof failed: " + safeReadProof.reason);
  return failure();
}

LogicalResult
checkSupportedLoadShape(VMIVRegType type, Value source, Type sourceType,
                        std::optional<int64_t> constantOffset,
                        std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMIMemoryAccessPlan accessPlan =
      buildReadAccessPlan(source, sourceType, type,
                          constantOffset, VMIMemoryValidMaskKind::AllTrue);
  if (!accessPlan.layoutSupport.isSupported())
    return fail(accessPlan.layoutSupport.reason);

  VMILayoutSupport supports;
  if (failed(supports.getLoadLayoutFact(type, reason)))
    return failure();

  if (getDenseLaneStrideLoadDistToken(type))
    return success();

  if (failed(getDataLanesPerPart(type.getElementType())))
    return fail("requires element type with known physical lane width");
  return success();
}

LogicalResult checkSupportedDeinterleaveLoadShape(
    VMIDeinterleaveLoadOp op,
    std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto lowType = cast<VMIVRegType>(op.getLow().getType());
  auto highType = cast<VMIVRegType>(op.getHigh().getType());
  VMILayoutSupport supports;
  if (failed(supports.getDeinterleaveLoadLayoutFactForLayouts(
          lowType, highType, reason)))
    return failure();
  if (!getX2MemoryDistToken(lowType.getElementType(), "DINTLV"))
    return fail("requires 8/16/32-bit element type for vldsx2 DINTLV");

  VMIMemoryAccessPlan accessPlan = buildReadAccessPlan(op.getSource(), op.getSource().getType(), lowType,
      getConstantIndexValue(op.getOffset()), VMIMemoryValidMaskKind::AllTrue);
  if (!accessPlan.layoutSupport.isSupported())
    return fail(accessPlan.layoutSupport.reason);

  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(lowType, &fullChunkReason)))
    return fail(Twine("requires full physical chunks; ") + fullChunkReason);
  return success();
}

LogicalResult
checkSupportedStoreShape(VMIVRegType type, Value destination,
                         Type destinationType, std::string *reason) {
  VMIMemoryAccessPlan accessPlan =
      buildWriteAccessPlan(destination, destinationType, type,
                           VMIMemoryWriteMaskKind::AllTrue);
  if (!accessPlan.layoutSupport.isSupported()) {
    if (reason)
      *reason = accessPlan.layoutSupport.reason;
    return failure();
  }

  if (failed(checkSupportedMaskableVReg(type, reason)))
    return failure();

  VMILayoutSupport supports;
  if (failed(supports.getStoreLayoutFact(type, reason)))
    return failure();

  if (getDenseLaneStrideStoreDistToken(type))
    return success();

  std::string fullChunkReason;
  if (succeeded(checkFullDataPhysicalChunks(type, &fullChunkReason)))
    return success();

  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout)
    return fail("requires assigned layout");
  if (failed(getDataLanesPerPart(type.getElementType())))
    return fail("requires known physical lanes per part");
  if (layout.isContiguous() && layout.getLaneStride() == 1)
    return success();

  std::string materializationReason;
  if (succeeded(checkCanMaterializeToContiguous(type, &materializationReason)))
    return success();
  return fail(Twine("partial/tail store requires contiguous layout or "
                    "deinterleaved layout that can materialize to contiguous; "
                    "value ") +
              fullChunkReason + ", materialization " + materializationReason);
}

LogicalResult checkSupportedInterleaveStoreShape(
    VMIInterleaveStoreOp op,
    std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto lowType = cast<VMIVRegType>(op.getLow().getType());
  auto highType = cast<VMIVRegType>(op.getHigh().getType());
  VMILayoutAttr lowLayout = lowType.getLayoutAttr();
  VMILayoutAttr highLayout = highType.getLayoutAttr();
  if (!lowLayout || !highLayout || !lowLayout.isContiguous() ||
      !highLayout.isContiguous())
    return fail("requires assigned contiguous low/high input layouts");
  if (lowType.getElementCount() != highType.getElementCount() ||
      lowType.getElementType() != highType.getElementType())
    return fail("requires matching low/high input shape and element type");
  if (!getX2MemoryDistToken(lowType.getElementType(), "INTLV"))
    return fail("requires 8/16/32-bit element type for vstsx2 INTLV");

  VMIMemoryAccessPlan accessPlan = buildWriteAccessPlan(op.getDestination(), op.getDestination().getType(), lowType,
      VMIMemoryWriteMaskKind::AllTrue);
  if (!accessPlan.layoutSupport.isSupported())
    return fail(accessPlan.layoutSupport.reason);
  if (failed(checkSupportedMaskableVReg(lowType, reason)))
    return failure();

  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(lowType, &fullChunkReason)))
    return fail(Twine("requires full physical chunks; ") + fullChunkReason);
  return success();
}

FailureOr<int64_t> getGroupSizeFromNumGroups(VMIVRegType type,
                                             int64_t numGroups,
                                             std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) -> FailureOr<int64_t> {
    if (reason)
      *reason = message.str();
    return failure();
  };
  if (numGroups <= 0)
    return fail("requires num_groups to be positive");
  if (type.getElementCount() % numGroups != 0)
    return fail("requires num_groups to evenly divide logical lane count");
  return type.getElementCount() / numGroups;
}

LogicalResult checkSupportedGroupChunkShape(VMIVRegType type, int64_t groupSize,
                                            std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout || !layout.isContiguous())
    return fail("requires assigned contiguous layout");
  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(type, &fullChunkReason)))
    return fail(Twine("requires full physical chunks; ") + fullChunkReason);
  FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(type.getElementType());
  if (failed(lanesPerPart))
    return fail("requires known physical lanes per part");
  if (groupSize <= 0 || type.getElementCount() % groupSize != 0)
    return fail("requires derived group size to evenly divide logical lane "
                "count");
  if (groupSize % *lanesPerPart != 0)
    return fail("currently requires group size to be a multiple of physical "
                "lanes per part");
  return success();
}

LogicalResult checkDeinterleaved2GroupStoreChunkShape(
    VMIVRegType type, int64_t groupSize, int64_t *lanesPerPart,
    int64_t *groupCount, int64_t *chunksPerGroupPerPart,
    std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout || !layout.isDeinterleaved() || layout.getFactor() != 2 ||
      layout.getLaneStride() != 1)
    return fail("requires deinterleaved=2 value layout");
  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(type, &fullChunkReason)))
    return fail(Twine("requires full physical chunks; ") + fullChunkReason);
  FailureOr<int64_t> lanes = getDataLanesPerPart(type.getElementType());
  if (failed(lanes))
    return fail("requires known physical lanes per part");
  if (!getX2MemoryDistToken(type.getElementType(), "INTLV"))
    return fail("requires 8/16/32-bit element type for vstsx2 INTLV");
  if (groupSize <= 0 || type.getElementCount() % groupSize != 0)
    return fail("requires derived group size to evenly divide logical lane "
                "count");
  int64_t pairLanes = 2 * *lanes;
  if (groupSize % pairLanes != 0)
    return fail("requires group size to be a multiple of two physical chunks");

  FailureOr<int64_t> part0Chunks = getDataChunksInPart(type, /*part=*/0);
  FailureOr<int64_t> part1Chunks = getDataChunksInPart(type, /*part=*/1);
  if (failed(part0Chunks) || failed(part1Chunks) ||
      *part0Chunks != *part1Chunks)
    return fail("requires matching deinterleaved part chunk counts");

  *lanesPerPart = *lanes;
  *groupCount = type.getElementCount() / groupSize;
  *chunksPerGroupPerPart = groupSize / pairLanes;
  if (*part0Chunks != *groupCount * *chunksPerGroupPerPart)
    return fail("requires deinterleaved chunks to align with group rows");
  return success();
}

LogicalResult
checkSupportedGroupLoadShape(VMIGroupLoadOp op, std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!resultLayout)
    return fail("requires assigned result layout");
  FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
      resultType, op.getNumGroupsAttr().getInt(), reason);
  if (failed(groupSize))
    return failure();

  if (resultLayout.isContiguous()) {
    VMILayoutSupport supports;
    if (failed(supports.getGroupLoadLayoutFact(op, reason)))
      return failure();
    if (failed(checkSupportedLoadShape(resultType, op.getSource(),
                                       op.getSource().getType(), std::nullopt,
                                       reason)))
      return failure();
    std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
    if (rowStride && *rowStride == *groupSize)
      return success();
    return checkSupportedGroupChunkShape(resultType, *groupSize, reason);
  }

  if (resultLayout.isBlockDeinterleaved() &&
      resultType.getElementType().isF32()) {
    VMILayoutSupport supports;
    if (failed(supports.getGroupLoadLayoutFact(op, reason)))
      return failure();
    VMIMemoryAccessPlan accessPlan = buildReadAccessPlan(op.getSource(), op.getSource().getType(), resultType,
        getConstantIndexValue(op.getOffset()), VMIMemoryValidMaskKind::AllTrue);
    if (!accessPlan.layoutSupport.isSupported())
      return fail(accessPlan.layoutSupport.reason);
    if (!isa<PtrType>(op.getSource().getType()))
      return fail(
          "block_deinterleaved group_load requires !pto.ptr source");
    if (op.getNumGroupsAttr().getInt() % 8 != 0)
      return fail(
          "block_deinterleaved group_load requires num_groups multiple of 8");
    std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
    if (!rowStride || *rowStride <= 0 || *rowStride % 8 != 0)
      return fail("block_deinterleaved group_load requires constant positive "
                  "row_stride divisible by 8 f32 elements");
    std::string fullChunkReason;
    if (failed(checkFullDataPhysicalChunks(resultType, &fullChunkReason)))
      return fail(Twine("block_deinterleaved group_load requires full physical "
                        "result chunks; ") +
                  fullChunkReason);
    return success();
  }

  return fail(
      "requires contiguous or block_deinterleaved f32 result layout");
}

LogicalResult checkSupportedGroupSlotLoadShape(
    VMIGroupSlotLoadOp op,
    std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutSupport supports;
  FailureOr<VMIGroupSlotLayoutFact> fact = supports.getGroupSlotLoadLayoutFact(
      resultType, op.getNumGroupsAttr().getInt(), reason);
  if (failed(fact))
    return failure();

  VMIMemoryAccessPlan accessPlan = buildReadAccessPlan(op.getSource(), op.getSource().getType(), resultType,
      getConstantIndexValue(op.getOffset()), VMIMemoryValidMaskKind::AllTrue);
  if (!accessPlan.layoutSupport.isSupported())
    return fail(accessPlan.layoutSupport.reason);
  if (!isa<PtrType>(op.getSource().getType()))
    return fail("group_slot_load requires !pto.ptr source");

  if (fact->slots == 8) {
    std::optional<int64_t> sourceGroupStride =
        getConstantIndexValue(op.getSourceGroupStride());
    if (!sourceGroupStride || *sourceGroupStride != 1)
      return fail("slots=8 group_slot_load requires constant unit "
                  "source_group_stride");
    return success();
  }

  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (elementBits == 0 || 256 % elementBits != 0)
    return fail("slots=1 group_slot_load requires supported element width");
  int64_t alignedStrideElems = 256 / elementBits;
  std::optional<int64_t> sourceGroupStride =
      getConstantIndexValue(op.getSourceGroupStride());
  if (!sourceGroupStride || *sourceGroupStride <= 0 ||
      *sourceGroupStride % alignedStrideElems != 0)
    return fail(Twine("slots=1 group_slot_load currently lowers as one "
                      "lane-0 vsldb per group and requires constant "
                      "positive source_group_stride divisible by ") +
                Twine(alignedStrideElems) +
                " elements for 32B load alignment; packed or unaligned "
                "scalar load lowering is not implemented");
  return success();
}

LogicalResult checkSupportedGroupBroadcastLoadShape(
    VMIGroupBroadcastLoadOp op,
    std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMILayoutSupport supports;
  if (failed(supports.getGroupBroadcastLoadSupport(op, reason)))
    return failure();
  VMIMemoryAccessPlan accessPlan = buildReadAccessPlan(op.getSource(), op.getSource().getType(),
      cast<VMIVRegType>(op.getResult().getType()),
      getConstantIndexValue(op.getOffset()), VMIMemoryValidMaskKind::AllTrue);
  if (!accessPlan.layoutSupport.isSupported())
    return fail(accessPlan.layoutSupport.reason);
  if (!isa<PtrType>(op.getSource().getType()))
    return fail("group_broadcast_load requires !pto.ptr source");
  return success();
}

static bool isCompactSmallGroupStore(VMILayoutAttr layout,
                                     VMIVRegType valueType, int64_t numGroups,
                                     std::optional<int64_t> rowStride) {
  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(valueType.getElementType());
  int64_t payloadBits =
      valueType.getElementCount() * static_cast<int64_t>(elementBits);
  return layout && layout.isGroupSlots() && layout.getNumGroups() == numGroups &&
         layout.getSlots() == 8 &&
         (layout.getLaneStride() == 1 || layout.getLaneStride() == 2 ||
          layout.getLaneStride() == 4) &&
         (valueType.getElementCount() == 4 ||
          valueType.getElementCount() == 8) &&
         numGroups == valueType.getElementCount() && elementBits > 0 &&
         payloadBits > 0 && payloadBits < 256 && payloadBits % 32 == 0 &&
         rowStride && *rowStride == 1;
}

LogicalResult
checkSupportedGroupStoreShape(VMIGroupStoreOp op, std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto valueType = cast<VMIVRegType>(op.getValue().getType());
  VMILayoutAttr layout = valueType.getLayoutAttr();
  std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
  if (isCompactSmallGroupStore(layout, valueType,
                               op.getNumGroupsAttr().getInt(), rowStride)) {
    if (!isa<PtrType>(op.getDestination().getType()))
      return fail("compact small group_store requires !pto.ptr destination");
    VMIMemoryAccessPlan accessPlan = buildWriteAccessPlan(
        op.getDestination(), op.getDestination().getType(), valueType,
        VMIMemoryWriteMaskKind::AllTrue);
    if (!accessPlan.layoutSupport.isSupported())
      return fail(accessPlan.layoutSupport.reason);
    return success();
  }
  if (layout && layout.isGroupSlots()) {
    VMILayoutSupport supports;
    FailureOr<VMIGroupSlotLayoutFact> fact = supports.getGroupStoreLayoutFact(
        valueType, op.getNumGroupsAttr().getInt(), reason);
    if (failed(fact))
      return failure();

    VMIMemoryAccessPlan accessPlan = buildWriteAccessPlan(op.getDestination(), op.getDestination().getType(),
        valueType, VMIMemoryWriteMaskKind::AllTrue);
    if (!accessPlan.layoutSupport.isSupported())
      return fail(accessPlan.layoutSupport.reason);

    if (fact->slots == 1) {
      unsigned elementBits =
          pto::getPTOStorageElemBitWidth(valueType.getElementType());
      if (elementBits == 0 || 256 % elementBits != 0)
        return fail("slots=1 group_store requires supported element width");
      std::optional<int64_t> rowStride =
          getConstantIndexValue(op.getRowStride());
      if (rowStride && *rowStride <= 0)
        return fail("slots=1 group_store requires positive row_stride when "
                    "row_stride is constant");
      if (!getPointStoreDistToken(valueType.getElementType()))
        return fail("slots=1 group_store requires 1PT_B8/B16/B32 store "
                    "support");
      return success();
    }

    if (!rowStride || *rowStride != 1)
      return fail("slots=8 group_store currently requires constant unit "
                  "row_stride");
    return success();
  }

  FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
      valueType, op.getNumGroupsAttr().getInt(), reason);
  if (failed(groupSize))
    return failure();
  if (failed(checkSupportedStoreShape(valueType,
                                      op.getDestination(),
                                      op.getDestination().getType(), reason)))
    return failure();
  if (succeeded(checkSupportedGroupChunkShape(valueType, *groupSize, reason)))
    return success();

  int64_t lanesPerPart = 0;
  int64_t groupCount = 0;
  int64_t chunksPerGroupPerPart = 0;
  return checkDeinterleaved2GroupStoreChunkShape(
      valueType, *groupSize, &lanesPerPart, &groupCount,
      &chunksPerGroupPerPart, reason);
}

LogicalResult
checkSupportedMaskedLoadShape(VMIMaskedLoadOp op, std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  auto passthruType = cast<VMIVRegType>(op.getPassthru().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  VMILayoutAttr passthruLayout = passthruType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMIMemoryAccessPlan accessPlan = buildReadAccessPlan(op.getSource(), op.getSource().getType(), resultType,
      getConstantIndexValue(op.getOffset()),
      VMIMemoryValidMaskKind::ExplicitMask);
  if (!accessPlan.layoutSupport.isSupported())
    return fail(accessPlan.layoutSupport.reason);
  if (!resultLayout || !passthruLayout || !maskLayout)
    return fail("requires assigned result, passthru, and mask layouts");
  if (!resultLayout.isContiguous() || !passthruLayout.isContiguous() ||
      !maskLayout.isContiguous())
    return fail("requires contiguous result, passthru, and mask layouts");

  std::string fullChunkReason;
  if (succeeded(checkFullDataPhysicalChunks(resultType, &fullChunkReason)))
    return success();

  if (accessPlan.safeReadProof.proven)
    return success();
  requireUnavailableReadFallback(accessPlan);
  return fail(Twine("partial/tail masked_load requires statically safe "
                    "full-read footprint; value ") +
              fullChunkReason + ", safe-read proof " +
              accessPlan.safeReadProof.reason +
              "; fallback decision: " + accessPlan.fallbackDecision.reason);
}

LogicalResult
checkSupportedGatherShape(VMIGatherOp op, std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  auto indicesType = cast<VMIVRegType>(op.getIndices().getType());
  auto passthruType = cast<VMIVRegType>(op.getPassthru().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  VMILayoutAttr indicesLayout = indicesType.getLayoutAttr();
  VMILayoutAttr passthruLayout = passthruType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  if (!resultLayout || !indicesLayout || !passthruLayout || !maskLayout)
    return fail("requires assigned result, indices, passthru, and mask "
                "layouts");
  if (!resultLayout.isContiguous() || !indicesLayout.isContiguous() ||
      !passthruLayout.isContiguous() || !maskLayout.isContiguous())
    return fail("requires contiguous result, indices, passthru, and mask "
                "layouts");

  if (!isa<PtrType>(op.getSource().getType()))
    return fail("requires !pto.ptr source because pto.vgather2_bc is "
                "pointer-only");

  unsigned resultBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  auto indexElementType = dyn_cast<IntegerType>(indicesType.getElementType());
  if (!indexElementType || indexElementType.isSigned())
    return fail("requires signless or unsigned integer indices");
  bool isU16Gather = resultBits == 16 && indexElementType.isUnsigned() &&
                     indexElementType.getWidth() == 16 &&
                     maskType.getGranularity() == "b16";
  bool isB32Gather = resultBits == 32 && indexElementType.getWidth() == 32 &&
                     maskType.getGranularity() == "b32";
  if (!isU16Gather && !isB32Gather)
    return fail("requires either 32-bit results with 32-bit indices and b32 "
                "mask, or ui16 results with ui16 indices and b16 mask");

  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  FailureOr<int64_t> indicesArity = getVMIPhysicalArity(indicesType);
  FailureOr<int64_t> passthruArity = getVMIPhysicalArity(passthruType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  if (failed(resultArity) || failed(indicesArity) || failed(passthruArity) ||
      failed(maskArity))
    return fail("requires computable physical arity");
  if (*resultArity != *indicesArity || *resultArity != *passthruArity ||
      *resultArity != *maskArity)
    return fail("requires result, indices, passthru, and mask to have the "
                "same physical arity");

  if (isB32Gather) {
    std::string resultReason;
    std::string indicesReason;
    std::string passthruReason;
    std::string maskReason;
    if (failed(checkFullDataPhysicalChunks(resultType, &resultReason)))
      return fail(Twine("result requires full physical chunks; ") +
                  resultReason);
    if (failed(checkFullDataPhysicalChunks(indicesType, &indicesReason)))
      return fail(Twine("indices require full physical chunks; ") +
                  indicesReason);
    if (failed(checkFullDataPhysicalChunks(passthruType, &passthruReason)))
      return fail(Twine("passthru requires full physical chunks; ") +
                  passthruReason);
    if (failed(checkFullVMIPhysicalChunks(maskType, &maskReason)))
      return fail(Twine("mask requires full physical chunks; ") + maskReason);
  } else if (*resultArity != 1) {
    return fail("ui16 gather currently supports one physical chunk");
  }

  return success();
}

LogicalResult
checkSupportedScatterShape(VMIScatterOp op, std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto valueType = cast<VMIVRegType>(op.getValue().getType());
  auto indicesType = cast<VMIVRegType>(op.getIndices().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  VMILayoutAttr valueLayout = valueType.getLayoutAttr();
  VMILayoutAttr indicesLayout = indicesType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  if (!valueLayout || !indicesLayout || !maskLayout)
    return fail("requires assigned value, indices, and mask layouts");
  if (!valueLayout.isContiguous() || !indicesLayout.isContiguous() ||
      !maskLayout.isContiguous())
    return fail("requires contiguous value, indices, and mask layouts");

  if (!isa<PtrType>(op.getDestination().getType()))
    return fail("requires !pto.ptr destination because pto.vscatter is "
                "pointer-only");

  if (pto::getPTOStorageElemBitWidth(valueType.getElementType()) != 32)
    return fail("currently requires 32-bit value element type so physical "
                "index and value lane counts match pto.vscatter");
  auto indexElementType = dyn_cast<IntegerType>(indicesType.getElementType());
  if (!indexElementType || indexElementType.getWidth() != 32 ||
      indexElementType.isSigned())
    return fail("requires signless or unsigned 32-bit indices");
  if (maskType.getGranularity() != "b32")
    return fail("requires b32 mask granularity");

  FailureOr<int64_t> valueArity = getVMIPhysicalArity(valueType);
  FailureOr<int64_t> indicesArity = getVMIPhysicalArity(indicesType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  if (failed(valueArity) || failed(indicesArity) || failed(maskArity))
    return fail("requires computable physical arity");
  if (*valueArity != *indicesArity || *valueArity != *maskArity)
    return fail("requires value, indices, and mask to have the same physical "
                "arity");

  std::string valueReason;
  std::string indicesReason;
  std::string maskReason;
  if (failed(checkFullDataPhysicalChunks(valueType, &valueReason)))
    return fail(Twine("value requires full physical chunks; ") + valueReason);
  if (failed(checkFullDataPhysicalChunks(indicesType, &indicesReason)))
    return fail(Twine("indices require full physical chunks; ") +
                indicesReason);
  if (failed(checkFullVMIPhysicalChunks(maskType, &maskReason)))
    return fail(Twine("mask requires full physical chunks; ") + maskReason);

  return success();
}

LogicalResult
checkSupportedStrideStoreShape(VMIStrideStoreOp op, std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto valueType = cast<VMIVRegType>(op.getValue().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  VMILayoutAttr valueLayout = valueType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  if (!valueLayout || !maskLayout)
    return fail("requires assigned value and mask layouts");
  if (!valueLayout.isContiguous() || !maskLayout.isContiguous())
    return fail("requires contiguous value and mask layouts");

  if (!isa<PtrType>(op.getDestination().getType()))
    return fail("requires !pto.ptr destination because pto.vsstb is "
                "pointer-only");
  if (failed(checkSupportedStoreShape(valueType,
                                      op.getDestination(),
                                      op.getDestination().getType(), reason)))
    return failure();

  FailureOr<int64_t> valueArity = getVMIPhysicalArity(valueType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  if (failed(valueArity) || failed(maskArity))
    return fail("requires computable physical arity");
  if (*valueArity != 1 || *maskArity != 1)
    return fail("currently supports one physical value/mask chunk");
  return success();
}

LogicalResult
checkSupportedStrideLoadShape(VMIStrideLoadOp op, std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  if (!resultLayout || !maskLayout)
    return fail("requires assigned result and mask layouts");
  if (!resultLayout.isContiguous() || !maskLayout.isContiguous())
    return fail("requires contiguous result and mask layouts");

  if (!isa<PtrType>(op.getSource().getType()))
    return fail("requires !pto.ptr source because pto.vsldb is pointer-only");

  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  if (failed(resultArity) || failed(maskArity))
    return fail("requires computable physical arity");
  if (*resultArity != 1 || *maskArity != 1)
    return fail("currently supports one physical result/mask chunk");
  return success();
}

Value stripMaskMaterialization(Value value) {
  while (true) {
    if (auto ensure = value.getDefiningOp<VMIEnsureMaskLayoutOp>()) {
      value = ensure.getSource();
      continue;
    }
    if (auto ensure = value.getDefiningOp<VMIEnsureMaskGranularityOp>()) {
      value = ensure.getSource();
      continue;
    }
    return value;
  }
}

bool isStaticAllActiveMask(Value mask, int64_t expectedLanes,
                           std::string *reason = nullptr) {
  mask = stripMaskMaterialization(mask);
  auto fail = [&](const Twine &message) {
    if (reason)
      *reason = message.str();
    return false;
  };

  if (auto createMask = mask.getDefiningOp<VMICreateMaskOp>()) {
    auto activeConstant =
        createMask.getActiveLanes().getDefiningOp<arith::ConstantOp>();
    if (!activeConstant)
      return fail("create_mask active_lanes is dynamic");
    auto activeAttr = dyn_cast<IntegerAttr>(activeConstant.getValue());
    if (!activeAttr)
      return fail("create_mask active_lanes is not an integer constant");
    return activeAttr.getInt() >= expectedLanes
               ? true
               : fail("create_mask active_lanes is smaller than the logical "
                      "lane count");
  }

  if (auto constantMask = mask.getDefiningOp<VMIConstantMaskOp>()) {
    auto denseAttr = dyn_cast<DenseIntElementsAttr>(constantMask.getValue());
    if (!denseAttr)
      return fail("constant_mask is not a dense integer mask");
    if (denseAttr.getNumElements() != expectedLanes)
      return fail("constant_mask element count does not match the logical "
                  "lane count");
    auto values = denseAttr.getValues<bool>();
    for (bool value : values)
      if (!value)
        return fail("constant_mask contains an inactive lane");
    return true;
  }

  return fail("mask is not a static all-active create_mask or constant_mask");
}

LogicalResult
checkSupportedExpandLoadShape(VMIExpandLoadOp op, std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  auto passthruType = cast<VMIVRegType>(op.getPassthru().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  VMILayoutAttr passthruLayout = passthruType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMIMemoryAccessPlan accessPlan = buildReadAccessPlan(op.getSource(), op.getSource().getType(), resultType,
      getConstantIndexValue(op.getOffset()),
      VMIMemoryValidMaskKind::ExplicitMask);
  if (!accessPlan.layoutSupport.isSupported())
    return fail(accessPlan.layoutSupport.reason);
  if (!resultLayout || !passthruLayout || !maskLayout)
    return fail("requires assigned result, passthru, and mask layouts");
  if (!resultLayout.isContiguous() || !passthruLayout.isContiguous() ||
      !maskLayout.isContiguous())
    return fail("requires contiguous result, passthru, and mask layouts");

  std::string maskReason;
  bool staticAllActive = isStaticAllActiveMask(
      op.getMask(), resultType.getElementCount(), &maskReason);

  std::string fullChunkReason;
  if (staticAllActive &&
      succeeded(checkFullDataPhysicalChunks(resultType, &fullChunkReason)))
    return success();

  if (staticAllActive && accessPlan.safeReadProof.proven)
    return success();

  std::string allActivePathReason;
  if (!staticAllActive) {
    allActivePathReason =
        maskReason.empty() ? "requires static all-active mask" : maskReason;
  } else {
    requireUnavailableReadFallback(accessPlan);
    allActivePathReason =
        (Twine("requires full physical chunks or statically safe full-read "
               "footprint; value ") +
         fullChunkReason + ", safe-read proof " +
         accessPlan.safeReadProof.reason +
         "; fallback decision: " + accessPlan.fallbackDecision.reason)
            .str();
  }

  if (!isa<PtrType>(op.getSource().getType()))
    return fail(Twine("runtime-mask path requires !pto.ptr source because "
                      "pto.vgather2_bc is pointer-only; all-active path ") +
                allActivePathReason);
  if (pto::getPTOStorageElemBitWidth(resultType.getElementType()) != 32)
    return fail("runtime-mask path currently requires 32-bit result element "
                "type so prefix indices and gather result lane counts match");
  if (maskType.getGranularity() != "b32")
    return fail("runtime-mask path requires b32 mask granularity");

  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  FailureOr<int64_t> passthruArity = getVMIPhysicalArity(passthruType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  if (failed(resultArity) || failed(passthruArity) || failed(maskArity))
    return fail("runtime-mask path requires computable physical arity");
  if (*resultArity != 1 || *passthruArity != 1 || *maskArity != 1)
    return fail("runtime-mask path currently supports only one physical "
                "chunk because prefix indices must not reset across chunks");

  std::string passthruReason;
  std::string maskFullReason;
  if (failed(checkFullDataPhysicalChunks(resultType, &fullChunkReason)))
    return fail(Twine("runtime-mask result requires full physical chunks; ") +
                fullChunkReason);
  if (failed(checkFullDataPhysicalChunks(passthruType, &passthruReason)))
    return fail(Twine("runtime-mask passthru requires full physical chunks; ") +
                passthruReason);
  if (failed(checkFullVMIPhysicalChunks(maskType, &maskFullReason)))
    return fail(Twine("runtime-mask mask requires full physical chunks; ") +
                maskFullReason);

  return success();
}

LogicalResult
checkSupportedMaskedStoreShape(VMIVRegType valueType, VMIMaskType maskType,
                               Value destination, Type destinationType,
                               std::string *reason) {
  VMIMemoryAccessPlan accessPlan =
      buildWriteAccessPlan(destination, destinationType,
                           valueType, VMIMemoryWriteMaskKind::ExplicitMask);
  if (!accessPlan.layoutSupport.isSupported()) {
    if (reason)
      *reason = accessPlan.layoutSupport.reason;
    return failure();
  }

  std::string valueReason;
  std::string maskReason;
  if (succeeded(checkFullDataPhysicalChunks(valueType, &valueReason)) &&
      succeeded(checkFullVMIPhysicalChunks(maskType, &maskReason)))
    return success();

  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMILayoutAttr valueLayout = valueType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  if (!valueLayout || !maskLayout)
    return fail("requires assigned value and mask layouts");

  FailureOr<int64_t> valueArity = getVMIPhysicalArity(valueType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  if (failed(valueArity) || failed(maskArity) || *valueArity != *maskArity)
    return fail("requires matching value/mask physical arity");

  if (valueLayout.hasDenseLaneStride()) {
    VMILayoutSupport supports;
    if (succeeded(
            supports.getMaskedStoreLayoutFact(valueType, maskType, reason)))
      return success();
  }

  std::string valueMaterializationReason;
  FailureOr<int64_t> valueParts = getContiguousMaterializationPartCount(
      valueType, &valueMaterializationReason);
  if (failed(valueParts))
    return fail(Twine("value cannot materialize to contiguous; value ") +
                valueReason + ", materialization " +
                valueMaterializationReason);

  std::string maskMaterializationReason;
  FailureOr<int64_t> maskParts = getContiguousMaterializationPartCount(
      maskType, &maskMaterializationReason);
  if (failed(maskParts))
    return fail(Twine("mask cannot materialize to contiguous; mask ") +
                maskReason + ", materialization " + maskMaterializationReason);
  if (*valueParts != *maskParts)
    return fail(
        "requires value/mask contiguous materialization arity to match");
  return success();
}

FailureOr<int64_t> getContiguousActiveDataLanes(VMIVRegType vmiType,
                                                int64_t chunk) {
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(vmiType.getElementType());
  if (failed(lanesPerPart))
    return failure();

  int64_t remaining = vmiType.getElementCount() - chunk * *lanesPerPart;
  return std::clamp<int64_t>(remaining, 0, *lanesPerPart);
}

FailureOr<int64_t> getActiveDataLanesInPhysicalChunk(VMIVRegType vmiType,
                                                     int64_t chunk) {
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(vmiType.getElementType());
  if (failed(lanesPerPart))
    return failure();

  int64_t active = 0;
  for (int64_t lane = 0; lane < *lanesPerPart; ++lane) {
    FailureOr<bool> padding = isPaddingLane(vmiType, /*part=*/0, chunk, lane);
    if (failed(padding))
      return failure();
    if (!*padding)
      ++active;
  }
  return active;
}

FailureOr<Value> createContiguousStoreMask(Location loc, VMIVRegType vmiType,
                                           int64_t chunk, VRegType vregType,
                                           PatternRewriter &rewriter) {
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(vmiType.getElementType());
  if (failed(lanesPerPart))
    return failure();

  FailureOr<int64_t> activeLanes = getContiguousActiveDataLanes(vmiType, chunk);
  if (failed(activeLanes))
    return failure();
  if (*activeLanes == *lanesPerPart)
    return createAllTrueMaskForVReg(loc, vregType, rewriter);

  FailureOr<MaskType> maskType =
      getMaskTypeForVReg(vregType, rewriter.getContext());
  if (failed(maskType))
    return failure();
  FailureOr<std::pair<Value, Value>> maskAndRemaining = createRuntimePrefixMask(
      loc, *maskType, createI32Constant(loc, *activeLanes, rewriter), rewriter);
  if (failed(maskAndRemaining))
    return failure();
  return maskAndRemaining->first;
}

FailureOr<Value> createMaskedStorePredicate(Location loc, VMIVRegType vmiType,
                                            int64_t chunk, Value userMask,
                                            VRegType vregType,
                                            PatternRewriter &rewriter) {
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(vmiType.getElementType());
  if (failed(lanesPerPart))
    return failure();

  FailureOr<int64_t> activeLanes = getContiguousActiveDataLanes(vmiType, chunk);
  if (failed(activeLanes))
    return failure();
  if (*activeLanes == *lanesPerPart)
    return userMask;

  auto maskType = dyn_cast<MaskType>(userMask.getType());
  if (!maskType)
    return failure();
  FailureOr<Value> tailMask =
      createContiguousStoreMask(loc, vmiType, chunk, vregType, rewriter);
  FailureOr<Value> allTrue = createAllTrueMask(loc, maskType, rewriter);
  if (failed(tailMask) || failed(allTrue))
    return failure();
  return rewriter.create<PandOp>(loc, maskType, userMask, *tailMask, *allTrue)
      .getResult();
}

FailureOr<Value> createDenseLaneStrideStorePredicate(
    Location loc, VMIVRegType vmiType, int64_t chunk, Value userMask,
    StringRef targetGranularity, PatternRewriter &rewriter) {
  auto sourceMaskType = dyn_cast<MaskType>(userMask.getType());
  if (!sourceMaskType)
    return failure();
  auto targetMaskType = MaskType::get(rewriter.getContext(), targetGranularity);
  Value compactMask = userMask;
  VMILayoutAttr layout = vmiType.getLayoutAttr();
  if (!layout)
    return failure();

  auto lower = rewriter.getStringAttr("LOWER");
  StringRef sourceGranularity = sourceMaskType.getGranularity();
  if (sourceGranularity == targetGranularity) {
    compactMask = userMask;
  } else if (layout.getLaneStride() == 2) {
    compactMask =
        rewriter.create<PunpackOp>(loc, targetMaskType, compactMask, lower)
            .getResult();
  } else if (layout.getLaneStride() == 4 && sourceGranularity == "b8" &&
             targetGranularity == "b32") {
    auto b16MaskType = MaskType::get(rewriter.getContext(), "b16");
    compactMask =
        rewriter.create<PunpackOp>(loc, b16MaskType, compactMask, lower)
            .getResult();
    compactMask =
        rewriter.create<PunpackOp>(loc, targetMaskType, compactMask, lower)
            .getResult();
  } else {
    return failure();
  }

  FailureOr<int64_t> activeLanes =
      getActiveDataLanesInPhysicalChunk(vmiType, chunk);
  FailureOr<int64_t> maskLanes = getMaskLanesPerPart(targetGranularity);
  if (failed(activeLanes) || failed(maskLanes))
    return failure();
  if (*activeLanes == *maskLanes)
    return compactMask;

  FailureOr<Value> tailMask = createPrefixMaskForActiveLanes(
      loc, targetMaskType, *activeLanes, rewriter);
  FailureOr<Value> allTrue = createAllTrueMask(loc, targetMaskType, rewriter);
  if (failed(tailMask) || failed(allTrue))
    return failure();
  return rewriter
      .create<PandOp>(loc, targetMaskType, compactMask, *tailMask, *allTrue)
      .getResult();
}

FailureOr<SmallVector<int64_t>>
computeShuffleForwardingSourceParts(VMIShuffleOp op, std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<SmallVector<int64_t>> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(sourceType.getElementType());
  if (failed(lanesPerPart))
    return fail("requires known lanes per physical part");

  ArrayRef<int64_t> indices = op.getIndices();
  if (indices.empty())
    return fail("requires non-empty indices");

  FailureOr<int64_t> resultFactor = getDataLayoutFactor(resultType);
  if (failed(resultFactor))
    return fail("requires assigned result layout");

  SmallVector<int64_t> sourceFlatIndices;
  for (int64_t resultPart = 0; resultPart < *resultFactor; ++resultPart) {
    FailureOr<int64_t> resultChunks =
        getDataChunksInPart(resultType, resultPart);
    if (failed(resultChunks))
      return fail("requires known result physical chunks");

    for (int64_t resultChunk = 0; resultChunk < *resultChunks; ++resultChunk) {
      std::optional<int64_t> sourcePart;
      std::optional<int64_t> sourceChunk;
      for (int64_t lane = 0; lane < *lanesPerPart; ++lane) {
        FailureOr<bool> padding =
            isPaddingLane(resultType, resultPart, resultChunk, lane);
        if (failed(padding))
          return fail("failed to classify result padding lanes");
        if (*padding)
          continue;

        FailureOr<int64_t> resultLogicalLane =
            mapPhysicalLaneToLogical(resultType, resultPart, resultChunk, lane);
        if (failed(resultLogicalLane) ||
            *resultLogicalLane >= static_cast<int64_t>(indices.size()))
          return fail("failed to map result lane");

        FailureOr<VMIPhysicalLane> sourcePhysical =
            mapLogicalLaneToPhysical(sourceType, indices[*resultLogicalLane]);
        if (failed(sourcePhysical))
          return fail("failed to map source lane");
        if (sourcePhysical->lane != lane)
          return fail("requires same-lane physical chunks");

        if (!sourcePart) {
          sourcePart = sourcePhysical->part;
          sourceChunk = sourcePhysical->chunk;
          continue;
        }
        if (*sourcePart != sourcePhysical->part ||
            *sourceChunk != sourcePhysical->chunk)
          return fail("requires one source chunk per result chunk");
      }

      if (!sourcePart || !sourceChunk)
        return fail("requires at least one logical lane per result chunk");
      FailureOr<int64_t> sourceFlatIndex =
          getDataFlatPartIndex(sourceType, *sourcePart, *sourceChunk);
      if (failed(sourceFlatIndex))
        return fail("source part range is out of bounds");
      sourceFlatIndices.push_back(*sourceFlatIndex);
    }
  }

  return sourceFlatIndices;
}

struct ShuffleVselrPlan {
  int64_t sourceFlatIndex = 0;
  int64_t baseLane = 0;
  bool descending = false;
};

FailureOr<int64_t> computeShuffleLane0SplatSourcePart(VMIShuffleOp op,
                                                      std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<int64_t> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  ArrayRef<int64_t> indices = op.getIndices();
  if (indices.empty())
    return fail("requires non-empty indices");
  if (!llvm::all_of(indices, [](int64_t index) { return index == 0; }))
    return fail("requires every result lane to select source lane 0");

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  FailureOr<VMIPhysicalLane> sourceLane =
      mapLogicalLaneToPhysical(sourceType, 0);
  if (failed(sourceLane))
    return fail("failed to map source lane 0");
  FailureOr<int64_t> sourceFlatIndex =
      getDataFlatPartIndex(sourceType, sourceLane->part, sourceLane->chunk);
  if (failed(sourceFlatIndex))
    return fail("source lane 0 part range is out of bounds");
  return *sourceFlatIndex;
}

FailureOr<SmallVector<ShuffleVselrPlan>>
computeShuffleVselrPlans(VMIShuffleOp op, std::string *reason) {
  auto fail =
      [&](const Twine &message) -> FailureOr<SmallVector<ShuffleVselrPlan>> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(sourceType.getElementType());
  if (failed(lanesPerPart))
    return fail("requires known lanes per physical part");

  ArrayRef<int64_t> indices = op.getIndices();
  if (indices.empty())
    return fail("requires non-empty indices");

  FailureOr<int64_t> resultFactor = getDataLayoutFactor(resultType);
  if (failed(resultFactor))
    return fail("requires assigned result layout");

  SmallVector<ShuffleVselrPlan> plans;
  for (int64_t resultPart = 0; resultPart < *resultFactor; ++resultPart) {
    FailureOr<int64_t> resultChunks =
        getDataChunksInPart(resultType, resultPart);
    if (failed(resultChunks))
      return fail("requires known result physical chunks");

    for (int64_t resultChunk = 0; resultChunk < *resultChunks; ++resultChunk) {
      std::optional<int64_t> sourcePart;
      std::optional<int64_t> sourceChunk;
      std::optional<int64_t> baseLane;
      std::optional<bool> descending;
      for (int64_t lane = 0; lane < *lanesPerPart; ++lane) {
        FailureOr<bool> padding =
            isPaddingLane(resultType, resultPart, resultChunk, lane);
        if (failed(padding) || *padding)
          return fail("requires full physical result chunks");

        FailureOr<int64_t> resultLogicalLane =
            mapPhysicalLaneToLogical(resultType, resultPart, resultChunk, lane);
        if (failed(resultLogicalLane) ||
            *resultLogicalLane >= static_cast<int64_t>(indices.size()))
          return fail("failed to map result lane");

        FailureOr<VMIPhysicalLane> sourcePhysical =
            mapLogicalLaneToPhysical(sourceType, indices[*resultLogicalLane]);
        if (failed(sourcePhysical))
          return fail("failed to map source lane");

        if (!sourcePart) {
          sourcePart = sourcePhysical->part;
          sourceChunk = sourcePhysical->chunk;
          baseLane = sourcePhysical->lane;
          continue;
        }

        if (*sourcePart != sourcePhysical->part ||
            *sourceChunk != sourcePhysical->chunk)
          return fail("requires one source chunk per result chunk");

        int64_t ascExpected = *baseLane + lane;
        int64_t descExpected = *baseLane - lane;
        bool asc = sourcePhysical->lane == ascExpected;
        bool desc = sourcePhysical->lane == descExpected;
        if (!asc && !desc)
          return fail("requires ASC or DESC affine source lane indices");

        bool laneDescending = desc && !asc;
        if (!descending) {
          descending = laneDescending;
          continue;
        }
        if (*descending != laneDescending)
          return fail("requires one index order per result chunk");
      }

      FailureOr<int64_t> sourceFlatIndex =
          getDataFlatPartIndex(sourceType, *sourcePart, *sourceChunk);
      if (failed(sourceFlatIndex))
        return fail("source part range is out of bounds");
      plans.push_back(ShuffleVselrPlan{*sourceFlatIndex, *baseLane,
                                       descending.value_or(false)});
    }
  }

  return plans;
}

struct ConstantMaskChunkMaterialization {
  SmallVector<int8_t> activeLanes;
};

FailureOr<SmallVector<ConstantMaskChunkMaterialization>>
computeConstantMaskMaterialization(VMIConstantMaskOp op, std::string *reason) {
  auto fail = [&](const Twine &message)
      -> FailureOr<SmallVector<ConstantMaskChunkMaterialization>> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto denseAttr = dyn_cast<DenseIntElementsAttr>(op.getValue());
  if (!denseAttr)
    return fail("only dense integer mask constants are supported");

  auto resultVMIType = cast<VMIMaskType>(op.getResult().getType());
  VMILayoutAttr layout = resultVMIType.getLayoutAttr();
  if (!layout ||
      !VMIMaskType::isConcreteGranularity(resultVMIType.getGranularity()))
    return fail("requires concrete layout and granularity");

  FailureOr<StringRef> physicalGranularity =
      getVMIMaskPhysicalGranularity(resultVMIType);
  FailureOr<int64_t> lanesPerPart =
      failed(physicalGranularity)
          ? FailureOr<int64_t>(failure())
          : getMaskLanesPerPart(*physicalGranularity);
  if (failed(lanesPerPart))
    return fail("requires known physical mask lanes per part");

  auto boolValues = denseAttr.getValues<bool>();
  int64_t factor = layout.isDenseSplit() ? layout.getFactor() : 1;
  SmallVector<ConstantMaskChunkMaterialization> materializations;
  for (int64_t part = 0; part < factor; ++part) {
    for (int64_t chunk = 0;; ++chunk) {
      bool anyLane = false;
      ConstantMaskChunkMaterialization materialization;
      materialization.activeLanes.reserve(*lanesPerPart);
      for (int64_t lane = 0; lane < *lanesPerPart; ++lane) {
        FailureOr<bool> padding =
            isPaddingLane(resultVMIType, part, chunk, lane);
        if (failed(padding))
          return fail("failed to map physical padding lane");
        if (*padding) {
          materialization.activeLanes.push_back(0);
          continue;
        }
        anyLane = true;

        FailureOr<int64_t> logicalLane =
            mapPhysicalLaneToLogical(resultVMIType, part, chunk, lane);
        if (failed(logicalLane))
          return fail("failed to map physical lane");
        materialization.activeLanes.push_back(boolValues[*logicalLane] ? 1 : 0);
      }
      if (!anyLane)
        break;
      materializations.push_back(std::move(materialization));
    }
  }

  return materializations;
}

FailureOr<SmallVector<ConstantMaskChunkMaterialization>>
computeGroupMaskMaterializationForType(VMICreateGroupMaskOp op,
                                       VMIMaskType resultVMIType,
                                       std::string *reason) {
  auto fail = [&](const Twine &message)
      -> FailureOr<SmallVector<ConstantMaskChunkMaterialization>> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto activeConstant =
      op.getActiveElemsPerGroup().getDefiningOp<arith::ConstantOp>();
  if (!activeConstant)
    return fail("requires constant active_elems_per_group");
  auto activeAttr = dyn_cast<IntegerAttr>(activeConstant.getValue());
  if (!activeAttr)
    return fail("active_elems_per_group must be an integer constant");

  VMILayoutAttr layout = resultVMIType.getLayoutAttr();
  if (!layout ||
      !VMIMaskType::isConcreteGranularity(resultVMIType.getGranularity()))
    return fail("requires concrete layout and granularity");

  FailureOr<StringRef> physicalGranularity =
      getVMIMaskPhysicalGranularity(resultVMIType);
  FailureOr<int64_t> lanesPerPart =
      failed(physicalGranularity)
          ? FailureOr<int64_t>(failure())
          : getMaskLanesPerPart(*physicalGranularity);
  if (failed(lanesPerPart))
    return fail("requires known physical mask lanes per part");

  int64_t numGroups = op.getNumGroupsAttr().getInt();
  int64_t groupSize = op.getGroupSizeAttr().getInt();
  if (numGroups <= 0 || groupSize <= 0 ||
      resultVMIType.getElementCount() != numGroups * groupSize)
    return fail("requires result lane count to match num_groups * group_size");

  int64_t activeElems = activeAttr.getInt();
  if (activeElems < 0)
    activeElems = 0;
  if (activeElems > groupSize)
    activeElems = groupSize;

  int64_t factor = layout.isDenseSplit() ? layout.getFactor() : 1;
  SmallVector<ConstantMaskChunkMaterialization> materializations;
  for (int64_t part = 0; part < factor; ++part) {
    for (int64_t chunk = 0;; ++chunk) {
      bool anyLane = false;
      ConstantMaskChunkMaterialization materialization;
      materialization.activeLanes.reserve(*lanesPerPart);
      for (int64_t lane = 0; lane < *lanesPerPart; ++lane) {
        FailureOr<bool> padding =
            isPaddingLane(resultVMIType, part, chunk, lane);
        if (failed(padding))
          return fail("failed to map physical padding lane");
        if (*padding) {
          materialization.activeLanes.push_back(0);
          continue;
        }
        anyLane = true;

        FailureOr<int64_t> logicalLane =
            mapPhysicalLaneToLogical(resultVMIType, part, chunk, lane);
        if (failed(logicalLane))
          return fail("failed to map physical lane");
        int64_t laneInGroup = *logicalLane % groupSize;
        materialization.activeLanes.push_back(laneInGroup < activeElems ? 1
                                                                        : 0);
      }
      if (!anyLane)
        break;
      materializations.push_back(std::move(materialization));
    }
  }

  return materializations;
}

FailureOr<SmallVector<ConstantMaskChunkMaterialization>>
computeGroupMaskMaterialization(VMICreateGroupMaskOp op, std::string *reason) {
  return computeGroupMaskMaterializationForType(
      op, cast<VMIMaskType>(op.getResult().getType()), reason);
}

FailureOr<Value> materializeConstantMaskChunk(Location loc, MaskType maskType,
                                              ArrayRef<int8_t> activeLanes,
                                              PatternRewriter &rewriter);

FailureOr<Value> createPowerOfTwoRemainder(Location loc, Value value,
                                           int64_t modulus, Value allMask,
                                           PatternRewriter &rewriter) {
  if (modulus <= 0)
    return failure();

  auto vectorType = dyn_cast<VRegType>(value.getType());
  if (!vectorType)
    return failure();

  std::optional<int64_t> shift = getPowerOfTwoLog2(modulus);
  if (!shift)
    return failure();
  if (*shift == 0) {
    Value zero = createI32Constant(loc, 0, rewriter);
    return rewriter.create<VdupOp>(loc, vectorType, zero, allMask,
                                   /*position=*/nullptr)
        .getResult();
  }

  Value shiftScalar = createI16Constant(loc, *shift, rewriter);
  Value quotient =
      rewriter.create<VshrsOp>(loc, vectorType, value, shiftScalar, allMask)
          .getResult();
  Value base =
      rewriter.create<VshlsOp>(loc, vectorType, quotient, shiftScalar, allMask)
          .getResult();
  return rewriter.create<VsubOp>(loc, vectorType, value, base, allMask)
      .getResult();
}

FailureOr<SmallVector<Value>> materializeDynamicGroupMaskForType(
    VMICreateGroupMaskOp op, Value activeElemsPerGroup,
    VMIMaskType resultVMIType, TypeRange resultTypes,
    PatternRewriter &rewriter) {
  auto fail = [&](const Twine &message) -> FailureOr<SmallVector<Value>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };

  VMILayoutAttr layout = resultVMIType.getLayoutAttr();
  if (!layout)
    return fail("dynamic create_group_mask requires assigned layout");
  if (layout.getLaneStride() != 1)
    return fail("dynamic create_group_mask requires lane_stride=1 layout");
  if (resultVMIType.getGranularity() != "b32")
    return fail("dynamic create_group_mask currently requires b32 "
                "granularity");

  int64_t numGroups = op.getNumGroupsAttr().getInt();
  int64_t groupSize = op.getGroupSizeAttr().getInt();
  if (numGroups <= 0 || groupSize <= 0 ||
      resultVMIType.getElementCount() != numGroups * groupSize)
    return fail("dynamic create_group_mask requires result lane count to "
                "match num_groups * group_size");

  FailureOr<StringRef> physicalGranularity =
      getVMIMaskPhysicalGranularity(resultVMIType);
  FailureOr<int64_t> lanesPerPart =
      failed(physicalGranularity)
          ? FailureOr<int64_t>(failure())
          : getMaskLanesPerPart(*physicalGranularity);
  FailureOr<int64_t> arity = getVMIPhysicalArity(resultVMIType);
  if (failed(lanesPerPart) || failed(arity) || *arity < 1)
    return fail("dynamic create_group_mask requires computable physical "
                "mask chunks");
  if (static_cast<int64_t>(resultTypes.size()) != *arity)
    return fail("dynamic create_group_mask physical result count mismatch");

  std::optional<int64_t> groupShift = getPowerOfTwoLog2(groupSize);
  if (!groupShift)
    return fail("dynamic create_group_mask currently requires power-of-two "
                "group_size");

  int64_t factor = layout.isDenseSplit() ? layout.getFactor() : 1;
  FailureOr<int64_t> blockElems = getVMILayoutBlockElems(resultVMIType);
  if (factor <= 0 || failed(blockElems) || *blockElems <= 0 ||
      static_cast<int64_t>(resultTypes.size()) % factor != 0)
    return fail("dynamic create_group_mask physical result count does not "
                "match layout factor");
  if (!getPowerOfTwoLog2(*blockElems))
    return fail("dynamic create_group_mask requires a power-of-two physical "
                "block element count");

  Location loc = op.getLoc();
  MLIRContext *ctx = rewriter.getContext();
  Type i32 = rewriter.getI32Type();
  auto indexVectorType = VRegType::get(ctx, *lanesPerPart, i32);
  Value activeI32 =
      clampDynamicActiveLanes(loc, activeElemsPerGroup, groupSize, rewriter);

  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  int64_t chunksPerPart = resultTypes.size() / factor;
  for (int64_t part = 0; part < factor; ++part) {
    for (int64_t chunk = 0; chunk < chunksPerPart; ++chunk) {
      Type resultType = resultTypes[part * chunksPerPart + chunk];
      auto maskType = dyn_cast<MaskType>(resultType);
      if (!maskType || !maskType.isB32())
        return fail("dynamic create_group_mask result must be b32 mask");

      FailureOr<Value> allMask = createAllTrueMask(loc, maskType, rewriter);
      if (failed(allMask))
        return fail("failed to create dynamic create_group_mask all mask");

      Value chunkBase = createI32Constant(loc, chunk * *lanesPerPart, rewriter);
      Value indexInPart =
          rewriter.create<VciOp>(loc, indexVectorType, chunkBase, StringAttr{})
              .getResult();

      Value partBlock = indexInPart;
      Value inBlockLane = createI32Constant(loc, 0, rewriter);
      if (blockElems != 1) {
        Value blockShift = createI16Constant(loc, *getPowerOfTwoLog2(*blockElems),
                                             rewriter);
        partBlock =
            rewriter
                .create<VshrsOp>(loc, indexVectorType, indexInPart, blockShift,
                                 *allMask)
                .getResult();
        Value blockBase =
            rewriter
                .create<VshlsOp>(loc, indexVectorType, partBlock, blockShift,
                                 *allMask)
                .getResult();
        inBlockLane =
            rewriter
                .create<VsubOp>(loc, indexVectorType, indexInPart, blockBase,
                                *allMask)
                .getResult();
      }

      Value factorScalar = createI32Constant(loc, factor, rewriter);
      Value logicalBlock =
          rewriter
              .create<VmulsOp>(loc, indexVectorType, partBlock, factorScalar,
                               *allMask)
              .getResult();
      if (part != 0) {
        Value partScalar = createI32Constant(loc, part, rewriter);
        logicalBlock =
            rewriter
                .create<VaddsOp>(loc, indexVectorType, logicalBlock,
                                 partScalar, *allMask)
                .getResult();
      }

      Value logicalLane = logicalBlock;
      if (blockElems != 1) {
        Value blockShift = createI16Constant(loc, *getPowerOfTwoLog2(*blockElems),
                                             rewriter);
        Value logicalBlockBase =
            rewriter
                .create<VshlsOp>(loc, indexVectorType, logicalBlock,
                                 blockShift, *allMask)
                .getResult();
        logicalLane =
            rewriter
                .create<VaddOp>(loc, indexVectorType, logicalBlockBase,
                                inBlockLane, *allMask)
                .getResult();
      }

      FailureOr<Value> laneInGroup = createPowerOfTwoRemainder(
          loc, logicalLane, groupSize, *allMask, rewriter);
      if (failed(laneInGroup))
        return fail("failed to compute dynamic create_group_mask lane index");

      Value predicate =
          rewriter
              .create<VcmpsOp>(loc, maskType, *laneInGroup, activeI32,
                               *allMask, rewriter.getStringAttr("lt"))
              .getResult();

      SmallVector<int8_t> validLanes;
      validLanes.reserve(*lanesPerPart);
      bool hasPadding = false;
      for (int64_t lane = 0; lane < *lanesPerPart; ++lane) {
        FailureOr<bool> padding =
            isPaddingLane(resultVMIType, part, chunk, lane);
        if (failed(padding))
          return fail("failed to classify dynamic create_group_mask padding");
        validLanes.push_back(*padding ? 0 : 1);
        hasPadding |= *padding;
      }
      if (hasPadding) {
        FailureOr<Value> validMask =
            materializeConstantMaskChunk(loc, maskType, validLanes, rewriter);
        if (failed(validMask))
          return fail("failed to materialize dynamic create_group_mask padding "
                      "mask");
        predicate =
            rewriter
                .create<PandOp>(loc, maskType, predicate, *validMask, *allMask)
                .getResult();
      }

      results.push_back(predicate);
    }
  }

  return results;
}

std::optional<int64_t> getPrefixActiveLaneCount(ArrayRef<int8_t> activeLanes) {
  bool seenInactive = false;
  int64_t activeCount = 0;
  for (int8_t active : activeLanes) {
    if (active) {
      if (seenInactive)
        return std::nullopt;
      ++activeCount;
      continue;
    }
    seenInactive = true;
  }
  return activeCount;
}

FailureOr<Value> materializePrefixMask(Location loc, MaskType maskType,
                                       int64_t activeLanes,
                                       int64_t lanesPerPart,
                                       PatternRewriter &rewriter) {
  std::optional<std::string> pattern =
      getPrefixPattern(activeLanes, lanesPerPart);
  if (pattern)
    return createPatternMask(loc, maskType, *pattern, rewriter);

  FailureOr<std::pair<Value, Value>> maskAndRemaining = createRuntimePrefixMask(
      loc, maskType, createI32Constant(loc, activeLanes, rewriter), rewriter);
  if (failed(maskAndRemaining))
    return failure();
  return maskAndRemaining->first;
}

FailureOr<Value> materializeConstantMaskChunk(Location loc, MaskType maskType,
                                              ArrayRef<int8_t> activeLanes,
                                              PatternRewriter &rewriter) {
  FailureOr<int64_t> lanesPerPart =
      getMaskLanesPerPart(maskType.getGranularity());
  if (failed(lanesPerPart) ||
      static_cast<int64_t>(activeLanes.size()) != *lanesPerPart)
    return failure();

  if (std::optional<int64_t> prefixCount =
          getPrefixActiveLaneCount(activeLanes))
    return materializePrefixMask(loc, maskType, *prefixCount, *lanesPerPart,
                                 rewriter);

  FailureOr<Value> allTrue = createAllTrueMask(loc, maskType, rewriter);
  if (failed(allTrue))
    return failure();

  Value result;
  int64_t lane = 0;
  while (lane < *lanesPerPart) {
    while (lane < *lanesPerPart && !activeLanes[lane])
      ++lane;
    if (lane >= *lanesPerPart)
      break;

    int64_t runBegin = lane;
    while (lane < *lanesPerPart && activeLanes[lane])
      ++lane;
    int64_t runEnd = lane;

    FailureOr<Value> prefixEnd =
        materializePrefixMask(loc, maskType, runEnd, *lanesPerPart, rewriter);
    if (failed(prefixEnd))
      return failure();

    Value runMask = *prefixEnd;
    if (runBegin != 0) {
      FailureOr<Value> prefixBegin = materializePrefixMask(
          loc, maskType, runBegin, *lanesPerPart, rewriter);
      if (failed(prefixBegin))
        return failure();
      Value notPrefixBegin =
          rewriter.create<PnotOp>(loc, maskType, *prefixBegin, *allTrue)
              .getResult();
      runMask = rewriter
                    .create<PandOp>(loc, maskType, *prefixEnd, notPrefixBegin,
                                    *allTrue)
                    .getResult();
    }

    if (!result) {
      result = runMask;
      continue;
    }
    result = rewriter.create<PorOp>(loc, maskType, result, runMask, *allTrue)
                 .getResult();
  }

  if (result)
    return result;
  return materializePrefixMask(loc, maskType, 0, *lanesPerPart, rewriter);
}

FailureOr<Value> createScalarOffsetConstant(Location loc, Type type,
                                            int64_t value,
                                            PatternRewriter &rewriter);

Value createChunkOffset(Location loc, Value baseOffset, int64_t laneOffset,
                        PatternRewriter &rewriter) {
  if (laneOffset == 0)
    return baseOffset;
  Value delta = rewriter.create<arith::ConstantIndexOp>(loc, laneOffset);
  return rewriter.create<arith::AddIOp>(loc, baseOffset, delta).getResult();
}

Value createGroupChunkOffset(Location loc, Value baseOffset, Value rowStride,
                             int64_t group, int64_t inGroupLaneOffset,
                             PatternRewriter &rewriter) {
  Value offset = baseOffset;
  if (group != 0) {
    Value groupIndex = rewriter.create<arith::ConstantIndexOp>(loc, group);
    Value rowOffset =
        rewriter.create<arith::MulIOp>(loc, rowStride, groupIndex).getResult();
    offset = rewriter.create<arith::AddIOp>(loc, offset, rowOffset).getResult();
  }
  return createChunkOffset(loc, offset, inGroupLaneOffset, rewriter);
}

LogicalResult checkContiguousFullGroupChunks(
    Operation *op, VMIVRegType type, int64_t groupSize, int64_t *lanesPerPart,
    int64_t *groupCount, int64_t *chunksPerGroup, PatternRewriter &rewriter) {
  auto fail = [&](const Twine &message) {
    return rewriter.notifyMatchFailure(op, message);
  };

  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout || !layout.isContiguous())
    return fail("group op requires contiguous VMI layout");
  if (failed(checkFullDataPhysicalChunks(type, nullptr)))
    return fail("group op requires full physical chunks");
  FailureOr<int64_t> lanes = getDataLanesPerPart(type.getElementType());
  if (failed(lanes))
    return fail("group op requires known physical lanes per part");
  if (groupSize <= 0 || type.getElementCount() % groupSize != 0)
    return fail("group op requires derived group size to evenly divide lane "
                "count");
  if (groupSize % *lanes != 0)
    return fail("group op currently requires group size to be a multiple of "
                "physical lanes per part");

  *lanesPerPart = *lanes;
  *groupCount = type.getElementCount() / groupSize;
  *chunksPerGroup = groupSize / *lanes;
  return success();
}

LogicalResult checkFullGroupSlotSourceShape(
    Operation *op, VMIVRegType type, int64_t groupSize, int64_t numGroups,
    int64_t *lanesPerPart, int64_t *groupCount, PatternRewriter &rewriter) {
  auto fail = [&](const Twine &message) {
    return rewriter.notifyMatchFailure(op, message);
  };

  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout || !layout.isGroupSlots() || layout.getNumGroups() != numGroups)
    return fail("group slot op requires matching num_groups VMI layout");
  if (type.getElementCount() != numGroups)
    return fail("group slot op requires one logical lane per group");
  FailureOr<int64_t> lanes = getDataLanesPerPart(type.getElementType());
  if (failed(lanes))
    return fail("group slot op requires known physical lanes per part");
  if (groupSize <= 0)
    return fail("group slot op requires positive derived group size");
  if (*lanes % groupSize != 0 && groupSize % *lanes != 0)
    return fail("group slot op requires group size to divide or be a "
                "multiple of physical lanes per part");

  *lanesPerPart = *lanes;
  *groupCount = numGroups;
  return success();
}

LogicalResult checkFullGroupBroadcastResultShape(
    Operation *op, VMIVRegType type, int64_t groupSize, int64_t lanesPerPart,
    int64_t *layoutFactor, int64_t *groupCount, PatternRewriter &rewriter) {
  auto fail = [&](const Twine &message) {
    return rewriter.notifyMatchFailure(op, message);
  };

  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout)
    return fail("group_broadcast result requires assigned VMI layout");
  if (layout.isGroupSlots())
    return fail("group_broadcast result requires a dense VMI layout");
  bool laneStridedDense = layout.isDense() && layout.getLaneStride() > 1;
  if (!laneStridedDense && failed(checkFullDataPhysicalChunks(type, nullptr)))
    return fail("group_broadcast result requires full physical chunks");
  FailureOr<int64_t> resultLanes = getDataLanesPerPart(type.getElementType());
  if (failed(resultLanes) || *resultLanes != lanesPerPart)
    return fail("group_broadcast result requires matching physical lanes");
  if (groupSize <= 0 || type.getElementCount() % groupSize != 0)
    return fail("group_broadcast result requires derived group size to evenly "
                "divide lane count");
  FailureOr<int64_t> factor = getDataLayoutFactor(type);
  if (failed(factor))
    return fail("group_broadcast result requires known layout factor");

  if (*factor == 1) {
    if (lanesPerPart % groupSize != 0 && groupSize % lanesPerPart != 0)
      return fail("group_broadcast contiguous result requires group size to "
                  "divide or be a multiple of physical lanes per part");
  } else {
    FailureOr<int64_t> blockElems = getVMILayoutBlockElems(type);
    bool blockFragmentSmallGroup =
        layout.isBlockDeinterleaved() && succeeded(blockElems) &&
        groupSize < lanesPerPart && lanesPerPart % *blockElems == 0;
    bool deinterleavedSmallGroup =
        layout.isDeinterleaved() &&
        groupSize < lanesPerPart && groupSize >= *factor &&
        groupSize % *factor == 0 && lanesPerPart % (groupSize / *factor) == 0;
    int64_t logicalSpanPerResultChunk = lanesPerPart * *factor;
    if (!blockFragmentSmallGroup && !deinterleavedSmallGroup &&
        (groupSize < lanesPerPart ||
         groupSize % logicalSpanPerResultChunk != 0))
      return fail("group_broadcast deinterleaved result requires every "
                  "physical result chunk to stay within one logical group");
  }

  *layoutFactor = *factor;
  *groupCount = type.getElementCount() / groupSize;
  return success();
}

FailureOr<Value> createZeroVector(Location loc, VRegType type,
                                  PatternRewriter &rewriter) {
  FailureOr<Value> zero =
      createScalarOffsetConstant(loc, type.getElementType(), 0, rewriter);
  FailureOr<Value> mask = createAllTrueMaskForVReg(loc, type, rewriter);
  if (failed(zero) || failed(mask))
    return failure();
  return rewriter
      .create<VdupOp>(loc, type, *zero, *mask,
                      /*position=*/nullptr)
      .getResult();
}

FailureOr<Value> createLaneRangeMask(Location loc, MaskType maskType,
                                     int64_t begin, int64_t end,
                                     PatternRewriter &rewriter) {
  FailureOr<int64_t> lanesPerPart =
      getMaskLanesPerPart(maskType.getGranularity());
  if (failed(lanesPerPart) || begin < 0 || begin > end || end > *lanesPerPart)
    return failure();
  SmallVector<int8_t> active(*lanesPerPart, 0);
  for (int64_t lane = begin; lane < end; ++lane)
    active[lane] = 1;
  return materializeConstantMaskChunk(loc, maskType, active, rewriter);
}

FailureOr<Value> createGroupSlotIndexVector(Location loc, VRegType indexType,
                                            int64_t groupSize,
                                            int64_t baseGroupSlot,
                                            PatternRewriter &rewriter,
                                            int64_t slotLaneStride = 1) {
  int64_t lanesPerPart = indexType.getElementCount();
  FailureOr<Value> baseScalar = createScalarOffsetConstant(
      loc, indexType.getElementType(), baseGroupSlot * slotLaneStride,
      rewriter);
  FailureOr<MaskType> maskType =
      getMaskTypeForVReg(indexType, rewriter.getContext());
  FailureOr<Value> allMask = createAllTrueMaskForVReg(loc, indexType, rewriter);
  if (failed(baseScalar) || failed(maskType) || failed(allMask))
    return failure();
  Value result = rewriter
                     .create<VdupOp>(loc, indexType, *baseScalar, *allMask,
                                     /*position=*/nullptr)
                     .getResult();
  if (groupSize >= lanesPerPart)
    return result;
  if (lanesPerPart % groupSize != 0)
    return failure();

  int64_t groupsPerChunk = lanesPerPart / groupSize;
  for (int64_t localGroup = 1; localGroup < groupsPerChunk; ++localGroup) {
    FailureOr<Value> groupScalar = createScalarOffsetConstant(
        loc, indexType.getElementType(),
        (baseGroupSlot + localGroup) * slotLaneStride, rewriter);
    FailureOr<Value> laneMask =
        createLaneRangeMask(loc, *maskType, localGroup * groupSize,
                            (localGroup + 1) * groupSize, rewriter);
    if (failed(groupScalar) || failed(laneMask))
      return failure();
    Value splat = rewriter
                      .create<VdupOp>(loc, indexType, *groupScalar, *allMask,
                                      /*position=*/nullptr)
                      .getResult();
    result = rewriter.create<VselOp>(loc, indexType, splat, result, *laneMask)
                 .getResult();
  }
  return result;
}

FailureOr<Value> createMappedGroupSlotIndexVector(
    Location loc, VMIVRegType resultVMIType, int64_t part, int64_t chunk,
    VRegType indexType, int64_t groupSize, int64_t slots, int64_t &sourceChunk,
    PatternRewriter &rewriter, int64_t slotLaneStride = 1) {
  if (groupSize <= 0 || slots <= 0)
    return failure();

  int64_t lanesPerPart = indexType.getElementCount();
  SmallVector<int64_t> slotByLane;
  slotByLane.reserve(lanesPerPart);
  std::optional<int64_t> resolvedSourceChunk;
  for (int64_t lane = 0; lane < lanesPerPart; ++lane) {
    FailureOr<bool> padding =
        isPaddingLane(resultVMIType, part, chunk, lane);
    if (failed(padding))
      return failure();
    if (*padding) {
      slotByLane.push_back(0);
      continue;
    }
    FailureOr<int64_t> logicalLane =
        mapPhysicalLaneToLogical(resultVMIType, part, chunk, lane);
    if (failed(logicalLane))
      return failure();
    int64_t group = *logicalLane / groupSize;
    int64_t candidateSourceChunk = group / slots;
    if (resolvedSourceChunk && *resolvedSourceChunk != candidateSourceChunk)
      return failure();
    resolvedSourceChunk = candidateSourceChunk;
    slotByLane.push_back((group % slots) * slotLaneStride);
  }
  if (!resolvedSourceChunk)
    return failure();
  sourceChunk = *resolvedSourceChunk;

  FailureOr<Value> baseScalar = createScalarOffsetConstant(
      loc, indexType.getElementType(), slotByLane.front(), rewriter);
  FailureOr<MaskType> maskType =
      getMaskTypeForVReg(indexType, rewriter.getContext());
  FailureOr<Value> allMask = createAllTrueMaskForVReg(loc, indexType, rewriter);
  if (failed(baseScalar) || failed(maskType) || failed(allMask))
    return failure();

  Value result = rewriter
                     .create<VdupOp>(loc, indexType, *baseScalar, *allMask,
                                     /*position=*/nullptr)
                     .getResult();
  int64_t rangeBegin = 0;
  while (rangeBegin < lanesPerPart) {
    int64_t slot = slotByLane[rangeBegin];
    int64_t rangeEnd = rangeBegin + 1;
    while (rangeEnd < lanesPerPart && slotByLane[rangeEnd] == slot)
      ++rangeEnd;
    if (rangeBegin != 0 || slot != slotByLane.front()) {
      FailureOr<Value> slotScalar = createScalarOffsetConstant(
          loc, indexType.getElementType(), slot, rewriter);
      FailureOr<Value> laneMask =
          createLaneRangeMask(loc, *maskType, rangeBegin, rangeEnd, rewriter);
      if (failed(slotScalar) || failed(laneMask))
        return failure();
      Value splat = rewriter
                        .create<VdupOp>(loc, indexType, *slotScalar, *allMask,
                                        /*position=*/nullptr)
                        .getResult();
      result = rewriter.create<VselOp>(loc, indexType, splat, result, *laneMask)
                   .getResult();
    }
    rangeBegin = rangeEnd;
  }
  return result;
}

std::optional<std::string> getX2MemoryDistToken(Type elementType,
                                                StringRef prefix) {
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  if (elementBits != 8 && elementBits != 16 && elementBits != 32)
    return std::nullopt;
  return (Twine(prefix) + "_B" + Twine(elementBits)).str();
}

std::optional<std::string> getDenseLaneStrideLoadDistToken(VMIVRegType type) {
  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout || !layout.isContiguous())
    return std::nullopt;
  unsigned elementBits = pto::getPTOStorageElemBitWidth(type.getElementType());
  if (layout.getLaneStride() == 2 &&
      (elementBits == 8 || elementBits == 16 || elementBits == 32))
    return (Twine("UNPK_B") + Twine(elementBits)).str();
  if (layout.getLaneStride() == 4 && elementBits == 8)
    return std::string("UNPK4");
  return std::nullopt;
}

std::optional<std::string>
getLaneStrideStoreDistToken(VMILayoutAttr layout, Type elementType) {
  if (!layout || !layout.hasLaneStride())
    return std::nullopt;
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  if (layout.getLaneStride() == 2 && elementBits == 8)
    return std::string("PK_B16");
  if (layout.getLaneStride() == 2 && elementBits == 16)
    return std::string("PK_B32");
  if (layout.getLaneStride() == 2 && elementBits == 32)
    return std::string("PK_B64");
  if (layout.getLaneStride() == 4 && elementBits == 8)
    return std::string("PK4_B32");
  return std::nullopt;
}

std::optional<std::string> getDenseLaneStrideStoreDistToken(VMIVRegType type) {
  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout || !layout.isContiguous())
    return std::nullopt;
  return getLaneStrideStoreDistToken(layout, type.getElementType());
}

std::optional<StringRef>
getLaneStrideStoreMaskGranularity(VMILayoutAttr layout, Type elementType) {
  if (!layout || !layout.hasLaneStride())
    return std::nullopt;
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  if (layout.getLaneStride() == 2 && elementBits == 8)
    return StringRef("b16");
  if (layout.getLaneStride() == 2 &&
      (elementBits == 16 || elementBits == 32))
    return StringRef("b32");
  if (layout.getLaneStride() == 4 && elementBits == 8)
    return StringRef("b32");
  return std::nullopt;
}

std::optional<StringRef>
getDenseLaneStrideStoreMaskGranularity(VMIVRegType type) {
  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout || !layout.isContiguous())
    return std::nullopt;
  return getLaneStrideStoreMaskGranularity(layout, type.getElementType());
}

std::optional<StringRef>
getDenseLaneStrideMaskedStoreMaskGranularity(VMIVRegType type) {
  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout || !layout.isContiguous())
    return std::nullopt;
  unsigned elementBits = pto::getPTOStorageElemBitWidth(type.getElementType());
  if (layout.getLaneStride() == 2 && elementBits == 8)
    return StringRef("b16");
  if (layout.getLaneStride() == 2 && elementBits == 16)
    return StringRef("b32");
  if (layout.getLaneStride() == 4 && elementBits == 8)
    return StringRef("b32");
  return std::nullopt;
}

std::optional<std::string> getPointStoreDistToken(Type elementType) {
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  if (elementBits != 8 && elementBits != 16 && elementBits != 32)
    return std::nullopt;
  return (Twine("1PT_B") + Twine(elementBits)).str();
}

struct VPTOCmpMode {
  StringRef mode;
  std::optional<IntegerType::SignednessSemantics> signedness;
};

std::optional<VPTOCmpMode> getVPTOCmpFMode(StringRef predicate) {
  if (predicate == "eq" || predicate == "ne" || predicate == "lt" ||
      predicate == "le" || predicate == "gt" || predicate == "ge")
    return VPTOCmpMode{predicate, std::nullopt};
  if (predicate == "oeq")
    return VPTOCmpMode{StringRef("eq"), std::nullopt};
  if (predicate == "one")
    return VPTOCmpMode{StringRef("ne"), std::nullopt};
  if (predicate == "olt")
    return VPTOCmpMode{StringRef("lt"), std::nullopt};
  if (predicate == "ole")
    return VPTOCmpMode{StringRef("le"), std::nullopt};
  if (predicate == "ogt")
    return VPTOCmpMode{StringRef("gt"), std::nullopt};
  if (predicate == "oge")
    return VPTOCmpMode{StringRef("ge"), std::nullopt};
  return std::nullopt;
}

std::optional<VPTOCmpMode> getVPTOCmpIMode(StringRef predicate) {
  if (predicate == "eq" || predicate == "ne")
    return VPTOCmpMode{predicate, std::nullopt};
  if (predicate == "ult")
    return VPTOCmpMode{
        StringRef("lt"), IntegerType::SignednessSemantics::Unsigned};
  if (predicate == "ule")
    return VPTOCmpMode{
        StringRef("le"), IntegerType::SignednessSemantics::Unsigned};
  if (predicate == "ugt")
    return VPTOCmpMode{
        StringRef("gt"), IntegerType::SignednessSemantics::Unsigned};
  if (predicate == "uge")
    return VPTOCmpMode{
        StringRef("ge"), IntegerType::SignednessSemantics::Unsigned};
  if (predicate == "slt")
    return VPTOCmpMode{
        StringRef("lt"), IntegerType::SignednessSemantics::Signed};
  if (predicate == "sle")
    return VPTOCmpMode{
        StringRef("le"), IntegerType::SignednessSemantics::Signed};
  if (predicate == "sgt")
    return VPTOCmpMode{
        StringRef("gt"), IntegerType::SignednessSemantics::Signed};
  if (predicate == "sge")
    return VPTOCmpMode{
        StringRef("ge"), IntegerType::SignednessSemantics::Signed};
  return std::nullopt;
}

template <typename SourceOp>
std::optional<VPTOCmpMode> getVPTOCmpMode(StringRef predicate) {
  if constexpr (std::is_same_v<SourceOp, VMICmpIOp>)
    return getVPTOCmpIMode(predicate);
  else
    return getVPTOCmpFMode(predicate);
}

template <typename SourceOp>
StringRef getSupportedComparePredicateMessage() {
  if constexpr (std::is_same_v<SourceOp, VMICmpIOp>)
    return "eq/ne, unsigned integer forms ult/ule/ugt/uge, and signed "
           "integer forms slt/sle/sgt/sge";
  else
    return "eq/ne/lt/le/gt/ge and ordered FP forms oeq/one/olt/ole/ogt/oge";
}

template <typename SourceOp>
LogicalResult checkSupportedComparePredicate(Operation *op,
                                             StringRef predicate) {
  if (getVPTOCmpMode<SourceOp>(predicate))
    return success();
  return op->emitError()
         << kVMIDiagUnsupportedPrefix << "compare predicate " << predicate
         << " cannot be lowered to pto.vcmp; supported predicates are "
         << getSupportedComparePredicateMessage<SourceOp>();
}

struct OneToNVMIUnpackOpPattern : OpConversionPattern<VMIUnpackOp> {
  using OpConversionPattern<VMIUnpackOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIUnpackOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    if (sourceParts.size() != op->getNumResults())
      return rewriter.notifyMatchFailure(
          op, "converted source part count must match unpack results");
    replaceOpWithFlatConvertedValues(rewriter, op, sourceParts,
                                     *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIPackOpPattern : OpConversionPattern<VMIPackOp> {
  using OpConversionPattern<VMIPackOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIPackOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<int64_t> arity = getVMIPhysicalArity(op.getResult().getType());
    SmallVector<Value> flatOperands = flattenOneToNOperands(adaptor.getOperands());
    if (failed(arity) || static_cast<int64_t>(flatOperands.size()) != *arity)
      return rewriter.notifyMatchFailure(
          op, "pack part count must match converted VMI result arity");
    replaceOpWithFlatConvertedValues(rewriter, op, flatOperands,
                                     *this->getTypeConverter());
    return success();
  }
};

LogicalResult verifyIdentityPartForwarding(Operation *op,
                                           ValueRange sourceParts,
                                           TypeRange resultTypes,
                                           PatternRewriter &rewriter) {
  if (sourceParts.size() != resultTypes.size())
    return rewriter.notifyMatchFailure(
        op, "source and result physical arity mismatch");
  for (auto [part, resultType] : llvm::zip_equal(sourceParts, resultTypes)) {
    if (part.getType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "helper requires non-identity physical materialization");
  }
  return success();
}

FailureOr<VRegType> getUnsignedCarrierVRegType(MLIRContext *ctx,
                                               unsigned elementBits) {
  if (elementBits != 8 && elementBits != 16 && elementBits != 32)
    return failure();
  auto elementType = IntegerType::get(
      ctx, elementBits, IntegerType::SignednessSemantics::Unsigned);
  return VRegType::get(ctx, 2048 / elementBits, elementType);
}

FailureOr<VRegType>
getSignednessCarrierVRegType(VRegType inputType,
                             IntegerType::SignednessSemantics signedness) {
  auto inputElementType = dyn_cast<IntegerType>(inputType.getElementType());
  if (!inputElementType)
    return failure();
  if ((signedness == IntegerType::SignednessSemantics::Signed &&
       !inputElementType.isUnsigned()) ||
      (signedness == IntegerType::SignednessSemantics::Unsigned &&
       inputElementType.isUnsigned()))
    return inputType;
  auto carrierElementType = IntegerType::get(
      inputType.getContext(), inputElementType.getWidth(), signedness);
  return VRegType::get(inputType.getContext(), inputType.getElementCount(),
                       carrierElementType);
}

FailureOr<Value> bitcastVReg(Location loc, Value value, Type resultType,
                             PatternRewriter &rewriter) {
  if (value.getType() == resultType)
    return value;
  auto inputType = dyn_cast<VRegType>(value.getType());
  auto outputType = dyn_cast<VRegType>(resultType);
  if (!inputType || !outputType)
    return failure();
  return rewriter.create<VbitcastOp>(loc, outputType, value).getResult();
}

FailureOr<VRegType> getVcaddResultType(VRegType inputType) {
  auto inputIntegerType = dyn_cast<IntegerType>(inputType.getElementType());
  if (!inputIntegerType || inputIntegerType.getWidth() == 32)
    return inputType;
  unsigned inputWidth = inputIntegerType.getWidth();
  if (inputWidth != 8 && inputWidth != 16)
    return failure();
  auto resultElementType = IntegerType::get(
      inputType.getContext(), inputWidth * 2,
      inputIntegerType.getSignedness());
  return VRegType::get(inputType.getContext(),
                       inputType.getElementCount() / 2, resultElementType);
}

FailureOr<Value> unpackToNextCarrier(Location loc, Value source,
                                     unsigned sourceBits, int64_t partIndex,
                                     PatternRewriter &rewriter) {
  FailureOr<VRegType> resultType =
      getUnsignedCarrierVRegType(rewriter.getContext(), sourceBits * 2);
  if (failed(resultType))
    return failure();
  Value part = rewriter.create<arith::ConstantIndexOp>(loc, partIndex);
  return rewriter.create<VzunpackOp>(loc, *resultType, source, part)
      .getResult();
}

FailureOr<Value> packToPreviousCarrier(Location loc, Value source,
                                       unsigned resultBits,
                                       PatternRewriter &rewriter) {
  FailureOr<VRegType> resultType =
      getUnsignedCarrierVRegType(rewriter.getContext(), resultBits);
  if (failed(resultType))
    return failure();
  return rewriter
      .create<VpackOp>(loc, *resultType, source,
                       rewriter.getStringAttr("LOWER"))
      .getResult();
}

FailureOr<SmallVector<Value>> materializeContiguousToLaneStride(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    Type elementType, int64_t laneStride, PatternRewriter &rewriter) {
  if (sourceParts.size() != resultTypes.size()) {
    (void)rewriter.notifyMatchFailure(
        op, "dense lane_stride unpack materialization requires matching "
            "source/result physical arity");
    return failure();
  }

  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  if ((laneStride != 2 && laneStride != 4) ||
      (laneStride == 4 && elementBits != 8) ||
      (elementBits != 8 && elementBits != 16)) {
    (void)rewriter.notifyMatchFailure(
        op, "unsupported dense lane_stride unpack carrier shape");
    return failure();
  }

  MLIRContext *ctx = rewriter.getContext();
  FailureOr<VRegType> inputCarrier =
      getUnsignedCarrierVRegType(ctx, elementBits);
  if (failed(inputCarrier))
    return failure();

  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (auto [resultIndex, resultType] : llvm::enumerate(resultTypes)) {
    int64_t sourceIndex = resultIndex / laneStride;
    if (sourceIndex >= static_cast<int64_t>(sourceParts.size()))
      return failure();
    Value source = sourceParts[sourceIndex];
    FailureOr<Value> current =
        bitcastVReg(op->getLoc(), source, *inputCarrier, rewriter);
    if (failed(current))
      return failure();
    int64_t part = resultIndex % laneStride;
    FailureOr<Value> unpacked =
        unpackToNextCarrier(op->getLoc(), *current, elementBits,
                            laneStride == 4 ? part / 2 : part, rewriter);
    if (failed(unpacked))
      return failure();
    current = *unpacked;
    if (laneStride == 4) {
      unpacked = unpackToNextCarrier(op->getLoc(), *current, elementBits * 2,
                                     part % 2, rewriter);
      if (failed(unpacked))
        return failure();
      current = *unpacked;
    }
    FailureOr<Value> result =
        bitcastVReg(op->getLoc(), *current, resultType, rewriter);
    if (failed(result))
      return failure();
    results.push_back(*result);
  }
  return results;
}

FailureOr<SmallVector<Value>> materializeLaneStrideToContiguous(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    Type elementType, int64_t laneStride, PatternRewriter &rewriter) {
  if (sourceParts.size() != resultTypes.size()) {
    (void)rewriter.notifyMatchFailure(
        op, "dense lane_stride pack materialization requires matching "
            "source/result physical arity");
    return failure();
  }

  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  if ((laneStride != 2 && laneStride != 4) ||
      (laneStride == 4 && elementBits != 8) ||
      (elementBits != 8 && elementBits != 16)) {
    (void)rewriter.notifyMatchFailure(
        op, "unsupported dense lane_stride pack carrier shape");
    return failure();
  }

  unsigned carrierBits =
      static_cast<unsigned>(elementBits * static_cast<unsigned>(laneStride));
  FailureOr<VRegType> sourceCarrier =
      getUnsignedCarrierVRegType(rewriter.getContext(), carrierBits);
  if (failed(sourceCarrier))
    return failure();

  SmallVector<Value> results;
  results.reserve(sourceParts.size());
  for (auto [source, resultType] : llvm::zip_equal(sourceParts, resultTypes)) {
    FailureOr<Value> current =
        bitcastVReg(op->getLoc(), source, *sourceCarrier, rewriter);
    if (failed(current))
      return failure();
    FailureOr<Value> packed = packToPreviousCarrier(op->getLoc(), *current,
                                                    carrierBits / 2, rewriter);
    if (failed(packed))
      return failure();
    current = *packed;
    if (laneStride == 4) {
      packed =
          packToPreviousCarrier(op->getLoc(), *current, elementBits, rewriter);
      if (failed(packed))
        return failure();
      current = *packed;
    }
    FailureOr<Value> result =
        bitcastVReg(op->getLoc(), *current, resultType, rewriter);
    if (failed(result))
      return failure();
    results.push_back(*result);
  }
  return results;
}

FailureOr<SmallVector<Value>> materializeGroupSlotLaneStride(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    Type elementType, int64_t sourceStride, int64_t resultStride,
    PatternRewriter &rewriter) {
  auto fail = [&](const Twine &message) -> FailureOr<SmallVector<Value>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };

  if (sourceParts.size() != resultTypes.size() || sourceParts.empty())
    return fail("group-slot lane_stride materialization requires matching "
                "non-empty source/result physical arity");
  if ((sourceStride != 1 && sourceStride != 2 && sourceStride != 4) ||
      (resultStride != 1 && resultStride != 2 && resultStride != 4))
    return fail("unsupported group-slot lane_stride factor");

  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  int64_t maxStride = std::max(sourceStride, resultStride);
  if ((elementBits != 8 && elementBits != 16) ||
      elementBits * maxStride > 32)
    return fail("unsupported group-slot lane_stride carrier shape");

  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (auto [source, resultType] :
       llvm::zip_equal(sourceParts, resultTypes)) {
    unsigned carrierBits = elementBits * sourceStride;
    FailureOr<VRegType> carrierType =
        getUnsignedCarrierVRegType(rewriter.getContext(), carrierBits);
    if (failed(carrierType))
      return fail("failed to derive group-slot source carrier type");
    FailureOr<Value> current =
        bitcastVReg(op->getLoc(), source, *carrierType, rewriter);
    if (failed(current))
      return fail("failed to bitcast group-slot source carrier");

    int64_t currentStride = sourceStride;
    while (currentStride < resultStride) {
      FailureOr<Value> unpacked = unpackToNextCarrier(
          op->getLoc(), *current, carrierBits, /*partIndex=*/0, rewriter);
      if (failed(unpacked))
        return fail("failed to unpack group-slot lane_stride carrier");
      current = *unpacked;
      currentStride *= 2;
      carrierBits *= 2;
    }
    while (currentStride > resultStride) {
      FailureOr<Value> packed = packToPreviousCarrier(
          op->getLoc(), *current, carrierBits / 2, rewriter);
      if (failed(packed))
        return fail("failed to pack group-slot lane_stride carrier");
      current = *packed;
      currentStride /= 2;
      carrierBits /= 2;
    }

    FailureOr<Value> result =
        bitcastVReg(op->getLoc(), *current, resultType, rewriter);
    if (failed(result))
      return fail("failed to bitcast group-slot result carrier");
    results.push_back(*result);
  }
  return results;
}

FailureOr<SmallVector<Value>> materializeDataLayoutConversion(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    Type sourceVMIElementType, PatternRewriter &rewriter) {
  if (!sourceLayout || !resultLayout) {
    (void)rewriter.notifyMatchFailure(
        op, "layout materialization requires assigned source/result layouts");
    return failure();
  }

  if (sourceLayout == resultLayout) {
    if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                            rewriter)))
      return failure();
    return SmallVector<Value>(sourceParts.begin(), sourceParts.end());
  }

  if (sourceLayout.isGroupSlots() && resultLayout.isGroupSlots() &&
      sourceLayout.getNumGroups() == resultLayout.getNumGroups() &&
      sourceLayout.getSlots() == 8 && resultLayout.getSlots() == 8) {
    return materializeGroupSlotLaneStride(
        op, sourceParts, resultTypes, sourceVMIElementType,
        sourceLayout.getLaneStride(), resultLayout.getLaneStride(), rewriter);
  }

  auto isElementDeinterleaved = [](VMILayoutAttr layout, int64_t factor) {
    return layout.isDeinterleaved() && layout.getFactor() == factor &&
           layout.getLaneStride() == 1;
  };
  auto isBlockDeinterleaved = [](VMILayoutAttr layout, int64_t factor) {
    return layout.isBlockDeinterleaved() && layout.getFactor() == factor;
  };

  bool contiguousToBlock =
      sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 &&
      (isBlockDeinterleaved(resultLayout, 2) ||
       isBlockDeinterleaved(resultLayout, 4));
  bool blockToContiguous =
      resultLayout.isContiguous() && resultLayout.getLaneStride() == 1 &&
      (isBlockDeinterleaved(sourceLayout, 2) ||
       isBlockDeinterleaved(sourceLayout, 4));
  if (contiguousToBlock || blockToContiguous) {
    if (sourceParts.size() == 1) {
      if (auto cast =
              sourceParts.front().getDefiningOp<UnrealizedConversionCastOp>()) {
        ValueRange inputs = cast.getInputs();
        if (inputs.size() == resultTypes.size()) {
          bool typesMatch = true;
          for (auto [input, resultType] : llvm::zip_equal(inputs, resultTypes))
            if (input.getType() != resultType) {
              typesMatch = false;
              break;
            }
          if (typesMatch)
            return SmallVector<Value>(inputs.begin(), inputs.end());
        }
      }
    }
    if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                            rewriter)))
      return failure();
    return SmallVector<Value>(sourceParts.begin(), sourceParts.end());
  }

  bool deint2ToContiguous = sourceLayout.isDeinterleaved() &&
                            isElementDeinterleaved(sourceLayout, 2) &&
                            resultLayout.isContiguous() &&
                            resultLayout.getLaneStride() == 1;
  bool contiguousToDeint2 =
      sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 &&
      resultLayout.isDeinterleaved() && isElementDeinterleaved(resultLayout, 2);
  if (deint2ToContiguous || contiguousToDeint2) {
    SmallVector<Value> results;
    if (deint2ToContiguous) {
      if (sourceParts.empty() || sourceParts.size() % 2 != 0 ||
          resultTypes.empty()) {
        (void)rewriter.notifyMatchFailure(
            op, "deinterleaved=2 to contiguous materialization requires "
                "2*N source parts and at least one result part");
        return failure();
      }
      int64_t groups = sourceParts.size() / 2;
      if (resultTypes.size() > static_cast<size_t>(2 * groups)) {
        (void)rewriter.notifyMatchFailure(
            op, "deinterleaved=2 to contiguous materialization result arity "
                "exceeds source footprint");
        return failure();
      }

      results.reserve(resultTypes.size());
      for (int64_t i = 0; i < groups && results.size() < resultTypes.size();
           ++i) {
        Value lhs = sourceParts[i];
        Value rhs = sourceParts[groups + i];
        if (lhs.getType() != rhs.getType())
          return rewriter.notifyMatchFailure(
              op, "vintlv requires matching source part types");
        Type lowType = resultTypes[results.size()];
        Type highType = results.size() + 1 < resultTypes.size()
                            ? resultTypes[results.size() + 1]
                            : lowType;
        if (lhs.getType() != lowType || lhs.getType() != highType)
          return rewriter.notifyMatchFailure(
              op, "vintlv requires operands and results to share one type");
        auto materialize = rewriter.create<VintlvOp>(
            op->getLoc(), lowType, highType, lhs, rhs);
        results.push_back(materialize.getLow());
        if (results.size() < resultTypes.size())
          results.push_back(materialize.getHigh());
      }
    } else {
      if (sourceParts.empty() || resultTypes.empty() ||
          resultTypes.size() % 2 != 0) {
        (void)rewriter.notifyMatchFailure(
            op, "contiguous to deinterleaved=2 materialization requires "
                "at least one source part and 2*N result parts");
        return failure();
      }
      int64_t groups = resultTypes.size() / 2;
      if (sourceParts.size() > static_cast<size_t>(2 * groups)) {
        (void)rewriter.notifyMatchFailure(
            op, "contiguous to deinterleaved=2 materialization source "
                "footprint exceeds result arity");
        return failure();
      }

      SmallVector<Value> part0;
      SmallVector<Value> part1;
      part0.reserve(groups);
      part1.reserve(groups);
      for (int64_t i = 0; i < groups; ++i) {
        size_t lhsIndex = 2 * i;
        if (lhsIndex >= sourceParts.size())
          return rewriter.notifyMatchFailure(
              op, "contiguous to deinterleaved=2 materialization missing "
                  "source part");
        size_t rhsIndex = lhsIndex + 1 < sourceParts.size() ? lhsIndex + 1
                                                            : lhsIndex;
        Value lhs = sourceParts[lhsIndex];
        Value rhs = sourceParts[rhsIndex];
        if (lhs.getType() != rhs.getType() ||
            lhs.getType() != resultTypes[i] ||
            lhs.getType() != resultTypes[groups + i])
          return rewriter.notifyMatchFailure(
              op, "vdintlv requires operands and results to share one type");
        auto materialize = rewriter.create<VdintlvOp>(
            op->getLoc(), resultTypes[i], resultTypes[groups + i],
            lhs, rhs);
        part0.push_back(materialize.getLow());
        part1.push_back(materialize.getHigh());
      }
      results.reserve(resultTypes.size());
      results.append(part0);
      results.append(part1);
    }
    return results;
  }

  bool deint4ToContiguous = sourceLayout.isDeinterleaved() &&
                            isElementDeinterleaved(sourceLayout, 4) &&
                            resultLayout.isContiguous() &&
                            resultLayout.getLaneStride() == 1;
  bool contiguousToDeint4 =
      sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 &&
      resultLayout.isDeinterleaved() && isElementDeinterleaved(resultLayout, 4);
  if (deint4ToContiguous || contiguousToDeint4) {
    auto getPartCounts = [](size_t totalParts,
                            int64_t factor) -> SmallVector<size_t> {
      SmallVector<size_t> counts;
      counts.reserve(factor);
      size_t base = totalParts / static_cast<size_t>(factor);
      size_t remainder = totalParts % static_cast<size_t>(factor);
      for (int64_t part = 0; part < factor; ++part)
        counts.push_back(base + (static_cast<size_t>(part) < remainder ? 1 : 0));
      return counts;
    };
    auto getPartOffsets = [](ArrayRef<size_t> counts) -> SmallVector<size_t> {
      SmallVector<size_t> offsets;
      offsets.reserve(counts.size());
      size_t offset = 0;
      for (size_t count : counts) {
        offsets.push_back(offset);
        offset += count;
      }
      return offsets;
    };

    SmallVector<Value> results;
    if (deint4ToContiguous) {
      if (sourceParts.empty() || resultTypes.empty()) {
        (void)rewriter.notifyMatchFailure(
            op, "deinterleaved=4 to contiguous materialization requires "
                "at least one source and result part");
        return failure();
      }
      if (resultTypes.size() > sourceParts.size()) {
        (void)rewriter.notifyMatchFailure(
            op, "deinterleaved=4 to contiguous materialization result arity "
                "exceeds source footprint");
        return failure();
      }

      SmallVector<size_t> sourceCounts = getPartCounts(sourceParts.size(), 4);
      SmallVector<size_t> sourceOffsets = getPartOffsets(sourceCounts);
      auto getSourcePart = [&](size_t part, size_t group) -> Value {
        if (group < sourceCounts[part])
          return sourceParts[sourceOffsets[part] + group];
        return sourceParts.back();
      };

      results.reserve(resultTypes.size());
      size_t groups = (resultTypes.size() + 3) / 4;
      for (size_t i = 0; i < groups && results.size() < resultTypes.size();
           ++i) {
        Value p0 = getSourcePart(0, i);
        Value p1 = getSourcePart(1, i);
        Value p2 = getSourcePart(2, i);
        Value p3 = getSourcePart(3, i);
        Type chunkType = p0.getType();
        if (p1.getType() != chunkType || p2.getType() != chunkType ||
            p3.getType() != chunkType)
          return rewriter.notifyMatchFailure(
              op, "vintlv deinterleaved=4 requires matching source part "
                  "types");
        for (size_t resultIndex = results.size();
             resultIndex < resultTypes.size() && resultIndex < results.size() + 4;
             ++resultIndex) {
          if (resultTypes[resultIndex] != chunkType)
            return rewriter.notifyMatchFailure(
                op, "vintlv requires operands and results to share one type");
        }

        auto even = rewriter.create<VintlvOp>(op->getLoc(), chunkType,
                                              chunkType, p0, p2);
        auto odd = rewriter.create<VintlvOp>(op->getLoc(), chunkType,
                                             chunkType, p1, p3);
        auto low = rewriter.create<VintlvOp>(op->getLoc(), chunkType, chunkType,
                                             even.getLow(), odd.getLow());
        auto high = rewriter.create<VintlvOp>(op->getLoc(), chunkType,
                                              chunkType, even.getHigh(),
                                              odd.getHigh());
        Value groupResults[] = {low.getLow(), low.getHigh(), high.getLow(),
                                high.getHigh()};
        for (Value result : groupResults) {
          if (results.size() >= resultTypes.size())
            break;
          results.push_back(result);
        }
      }
    } else {
      if (sourceParts.empty() || resultTypes.empty()) {
        (void)rewriter.notifyMatchFailure(
            op, "contiguous to deinterleaved=4 materialization requires "
                "at least one source and result part");
        return failure();
      }
      if (sourceParts.size() > resultTypes.size()) {
        (void)rewriter.notifyMatchFailure(
            op, "contiguous to deinterleaved=4 materialization source "
                "footprint exceeds result arity");
        return failure();
      }

      SmallVector<size_t> resultCounts = getPartCounts(resultTypes.size(), 4);
      SmallVector<size_t> resultOffsets = getPartOffsets(resultCounts);
      auto getContiguousSourcePart = [&](size_t index) {
        return sourceParts[std::min(index, sourceParts.size() - 1)];
      };
      SmallVector<Value> part0;
      SmallVector<Value> part1;
      SmallVector<Value> part2;
      SmallVector<Value> part3;
      part0.reserve(resultCounts[0]);
      part1.reserve(resultCounts[1]);
      part2.reserve(resultCounts[2]);
      part3.reserve(resultCounts[3]);
      size_t groups =
          *std::max_element(resultCounts.begin(), resultCounts.end());
      for (size_t i = 0; i < groups; ++i) {
        Value s0 = getContiguousSourcePart(4 * i);
        Value s1 = getContiguousSourcePart(4 * i + 1);
        Value s2 = getContiguousSourcePart(4 * i + 2);
        Value s3 = getContiguousSourcePart(4 * i + 3);
        Type chunkType = s0.getType();
        if (s0.getType() != s1.getType() || s0.getType() != s2.getType() ||
            s0.getType() != s3.getType())
          return rewriter.notifyMatchFailure(
              op, "vdintlv deinterleaved=4 requires matching source part "
                  "types");
        for (int64_t part = 0; part < 4; ++part) {
          if (i < resultCounts[part] &&
              resultTypes[resultOffsets[part] + i] != chunkType)
            return rewriter.notifyMatchFailure(
                op, "vdintlv requires operands and results to share one type");
        }

        auto low = rewriter.create<VdintlvOp>(
            op->getLoc(), chunkType, chunkType, s0, s1);
        auto high = rewriter.create<VdintlvOp>(op->getLoc(), chunkType,
                                               chunkType, s2, s3);
        auto even = rewriter.create<VdintlvOp>(
            op->getLoc(), chunkType, chunkType, low.getLow(), high.getLow());
        auto odd = rewriter.create<VdintlvOp>(
            op->getLoc(), chunkType, chunkType, low.getHigh(), high.getHigh());
        if (i < resultCounts[0])
          part0.push_back(even.getLow());
        if (i < resultCounts[1])
          part1.push_back(odd.getLow());
        if (i < resultCounts[2])
          part2.push_back(even.getHigh());
        if (i < resultCounts[3])
          part3.push_back(odd.getHigh());
      }
      results.reserve(resultTypes.size());
      results.append(part0);
      results.append(part1);
      results.append(part2);
      results.append(part3);
    }
    return results;
  }

  if (sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 &&
      resultLayout.isContiguous() && resultLayout.getLaneStride() != 1) {
    return materializeContiguousToLaneStride(
        op, sourceParts, resultTypes, sourceVMIElementType,
        resultLayout.getLaneStride(), rewriter);
  }

  if (sourceLayout.isContiguous() && sourceLayout.getLaneStride() != 1 &&
      resultLayout.isContiguous() && resultLayout.getLaneStride() == 1) {
    return materializeLaneStrideToContiguous(
        op, sourceParts, resultTypes, sourceVMIElementType,
        sourceLayout.getLaneStride(), rewriter);
  }

  if (sourceLayout.isDeinterleaved() && resultLayout.isDeinterleaved() &&
      sourceLayout.getLaneStride() == 1 && resultLayout.getLaneStride() == 1 &&
      (sourceLayout.getFactor() == 2 || sourceLayout.getFactor() == 4) &&
      (resultLayout.getFactor() == 2 || resultLayout.getFactor() == 4)) {
    VMILayoutAttr contiguous =
        VMILayoutAttr::getContiguous(rewriter.getContext());
    FailureOr<SmallVector<Value>> dense = materializeDataLayoutConversion(
        op, sourceParts, resultTypes, sourceLayout, contiguous,
        sourceVMIElementType, rewriter);
    if (failed(dense))
      return failure();
    return materializeDataLayoutConversion(op, *dense, resultTypes, contiguous,
                                           resultLayout, sourceVMIElementType,
                                           rewriter);
  }

  (void)rewriter.notifyMatchFailure(
      op, "unsupported VMI data layout materialization");
  return failure();
}

FailureOr<SmallVector<Value>> materializeEnsureLayoutConversion(
    Operation *op, ValueRange sourceParts, VMIVRegType sourceType,
    VMIVRegType resultType, const TypeConverter &typeConverter,
    PatternRewriter &rewriter) {
  VMILayoutSupport supports;
  std::string supportReason;
  if (failed(supports.getEnsureLayoutFact(sourceType, resultType,
                                          &supportReason))) {
    (void)rewriter.notifyMatchFailure(
        op, Twine("ensure_layout has no registered materialization support: ") +
                supportReason);
    return failure();
  }

  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !resultLayout) {
    (void)rewriter.notifyMatchFailure(
        op, "ensure_layout requires assigned source/result layouts");
    return failure();
  }

  SmallVector<Type> resultTypes;
  if (failed(typeConverter.convertType(resultType, resultTypes)))
    return failure();
  return materializeDataLayoutConversion(op, sourceParts, resultTypes,
                                         sourceLayout, resultLayout,
                                         sourceType.getElementType(), rewriter);
}

FailureOr<std::pair<Value, Value>>
createPredicateDintlv(Location loc, Type lowType, Type highType, Value lhs,
                      Value rhs, PatternRewriter &rewriter) {
  auto maskType = dyn_cast<MaskType>(lowType);
  if (!maskType || highType != lowType)
    return failure();
  if (maskType.isB8()) {
    auto op = rewriter.create<PdintlvB8Op>(loc, lowType, highType, lhs, rhs);
    return std::make_pair(op.getLow(), op.getHigh());
  }
  if (maskType.isB16()) {
    auto op = rewriter.create<PdintlvB16Op>(loc, lowType, highType, lhs, rhs);
    return std::make_pair(op.getLow(), op.getHigh());
  }
  if (maskType.isB32()) {
    auto op = rewriter.create<PdintlvB32Op>(loc, lowType, highType, lhs, rhs);
    return std::make_pair(op.getLow(), op.getHigh());
  }
  return failure();
}

FailureOr<std::pair<Value, Value>>
createPredicateIntlv(Location loc, Type lowType, Type highType, Value lhs,
                     Value rhs, PatternRewriter &rewriter) {
  auto maskType = dyn_cast<MaskType>(lowType);
  if (!maskType || highType != lowType)
    return failure();
  if (maskType.isB8()) {
    auto op = rewriter.create<PintlvB8Op>(loc, lowType, highType, lhs, rhs);
    return std::make_pair(op.getLow(), op.getHigh());
  }
  if (maskType.isB16()) {
    auto op = rewriter.create<PintlvB16Op>(loc, lowType, highType, lhs, rhs);
    return std::make_pair(op.getLow(), op.getHigh());
  }
  if (maskType.isB32()) {
    auto op = rewriter.create<PintlvB32Op>(loc, lowType, highType, lhs, rhs);
    return std::make_pair(op.getLow(), op.getHigh());
  }
  return failure();
}

FailureOr<SmallVector<Value>> materializeMaskLayoutConversion(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    PatternRewriter &rewriter) {
  if (!sourceLayout || !resultLayout) {
    (void)rewriter.notifyMatchFailure(
        op, "mask layout materialization requires assigned source/result "
            "layouts");
    return failure();
  }

  if (sourceLayout == resultLayout) {
    if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                            rewriter)))
      return failure();
    return SmallVector<Value>(sourceParts.begin(), sourceParts.end());
  }

  auto isElementDeinterleaved = [](VMILayoutAttr layout, int64_t factor) {
    return layout.isDeinterleaved() && layout.getFactor() == factor &&
           layout.getLaneStride() == 1;
  };

  bool deint2ToContiguous = sourceLayout.isDeinterleaved() &&
                            isElementDeinterleaved(sourceLayout, 2) &&
                            resultLayout.isContiguous() &&
                            resultLayout.getLaneStride() == 1;
  bool contiguousToDeint2 =
      sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 &&
      resultLayout.isDeinterleaved() && isElementDeinterleaved(resultLayout, 2);
  if (deint2ToContiguous || contiguousToDeint2) {
    if (sourceParts.size() != resultTypes.size() || sourceParts.empty() ||
        sourceParts.size() % 2 != 0) {
      (void)rewriter.notifyMatchFailure(
          op, "deinterleaved=2 mask layout materialization requires 2*N "
              "parts");
      return failure();
    }
    if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                            rewriter)))
      return failure();

    int64_t groups = sourceParts.size() / 2;
    SmallVector<Value> results;
    results.reserve(sourceParts.size());
    if (deint2ToContiguous) {
      for (int64_t i = 0; i < groups; ++i) {
        FailureOr<std::pair<Value, Value>> materialize = createPredicateIntlv(
            op->getLoc(), resultTypes[2 * i], resultTypes[2 * i + 1],
            sourceParts[i], sourceParts[groups + i], rewriter);
        if (failed(materialize))
          return rewriter.notifyMatchFailure(
              op, "unsupported predicate intlv mask type");
        results.append({materialize->first, materialize->second});
      }
    } else {
      SmallVector<Value> part0;
      SmallVector<Value> part1;
      part0.reserve(groups);
      part1.reserve(groups);
      for (int64_t i = 0; i < groups; ++i) {
        FailureOr<std::pair<Value, Value>> materialize = createPredicateDintlv(
            op->getLoc(), resultTypes[i], resultTypes[groups + i],
            sourceParts[2 * i], sourceParts[2 * i + 1], rewriter);
        if (failed(materialize))
          return rewriter.notifyMatchFailure(
              op, "unsupported predicate dintlv mask type");
        part0.push_back(materialize->first);
        part1.push_back(materialize->second);
      }
      results.append(part0);
      results.append(part1);
    }
    return results;
  }

  bool deint4ToContiguous = sourceLayout.isDeinterleaved() &&
                            isElementDeinterleaved(sourceLayout, 4) &&
                            resultLayout.isContiguous() &&
                            resultLayout.getLaneStride() == 1;
  bool contiguousToDeint4 =
      sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 &&
      resultLayout.isDeinterleaved() && isElementDeinterleaved(resultLayout, 4);
  if (deint4ToContiguous || contiguousToDeint4) {
    if (sourceParts.size() != resultTypes.size() || sourceParts.empty() ||
        sourceParts.size() % 4 != 0) {
      (void)rewriter.notifyMatchFailure(
          op, "deinterleaved=4 mask layout materialization requires 4*N "
              "parts");
      return failure();
    }
    if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                            rewriter)))
      return failure();

    SmallVector<Value> results;
    results.reserve(sourceParts.size());
    int64_t groups = sourceParts.size() / 4;
    if (deint4ToContiguous) {
      for (int64_t i = 0; i < groups; ++i) {
        Value p0 = sourceParts[i];
        Value p1 = sourceParts[groups + i];
        Value p2 = sourceParts[2 * groups + i];
        Value p3 = sourceParts[3 * groups + i];
        FailureOr<std::pair<Value, Value>> even =
            createPredicateIntlv(op->getLoc(), resultTypes[4 * i],
                                 resultTypes[4 * i + 1], p0, p2, rewriter);
        FailureOr<std::pair<Value, Value>> odd =
            createPredicateIntlv(op->getLoc(), resultTypes[4 * i],
                                 resultTypes[4 * i + 1], p1, p3, rewriter);
        if (failed(even) || failed(odd))
          return rewriter.notifyMatchFailure(
              op, "unsupported predicate intlv mask type");
        FailureOr<std::pair<Value, Value>> low = createPredicateIntlv(
            op->getLoc(), resultTypes[4 * i], resultTypes[4 * i + 1],
            even->first, odd->first, rewriter);
        FailureOr<std::pair<Value, Value>> high = createPredicateIntlv(
            op->getLoc(), resultTypes[4 * i + 2], resultTypes[4 * i + 3],
            even->second, odd->second, rewriter);
        if (failed(low) || failed(high))
          return rewriter.notifyMatchFailure(
              op, "unsupported predicate intlv mask type");
        results.append({low->first, low->second, high->first, high->second});
      }
    } else {
      SmallVector<Value> part0;
      SmallVector<Value> part1;
      SmallVector<Value> part2;
      SmallVector<Value> part3;
      part0.reserve(groups);
      part1.reserve(groups);
      part2.reserve(groups);
      part3.reserve(groups);
      for (int64_t i = 0; i < groups; ++i) {
        FailureOr<std::pair<Value, Value>> low = createPredicateDintlv(
            op->getLoc(), resultTypes[i], resultTypes[groups + i],
            sourceParts[4 * i], sourceParts[4 * i + 1], rewriter);
        FailureOr<std::pair<Value, Value>> high = createPredicateDintlv(
            op->getLoc(), resultTypes[2 * groups + i],
            resultTypes[3 * groups + i], sourceParts[4 * i + 2],
            sourceParts[4 * i + 3], rewriter);
        if (failed(low) || failed(high))
          return rewriter.notifyMatchFailure(
              op, "unsupported predicate dintlv mask type");
        FailureOr<std::pair<Value, Value>> even = createPredicateDintlv(
            op->getLoc(), resultTypes[i], resultTypes[2 * groups + i],
            low->first, high->first, rewriter);
        FailureOr<std::pair<Value, Value>> odd = createPredicateDintlv(
            op->getLoc(), resultTypes[groups + i], resultTypes[3 * groups + i],
            low->second, high->second, rewriter);
        if (failed(even) || failed(odd))
          return rewriter.notifyMatchFailure(
              op, "unsupported predicate dintlv mask type");
        part0.push_back(even->first);
        part1.push_back(odd->first);
        part2.push_back(even->second);
        part3.push_back(odd->second);
      }
      results.append(part0);
      results.append(part1);
      results.append(part2);
      results.append(part3);
    }
    return results;
  }

  if (sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 &&
      resultLayout.isContiguous() && resultLayout.getLaneStride() != 1) {
    int64_t laneStride = resultLayout.getLaneStride();
    if (laneStride != 2 && laneStride != 4)
      return rewriter.notifyMatchFailure(
          op, "unsupported dense mask lane_stride unpack factor");
    if (static_cast<int64_t>(resultTypes.size()) >
        static_cast<int64_t>(sourceParts.size()) * laneStride)
      return rewriter.notifyMatchFailure(
          op, "dense mask lane_stride unpack materialization result arity "
              "does not fit source arity");
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    auto lower = rewriter.getStringAttr("LOWER");
    auto higher = rewriter.getStringAttr("HIGHER");
    for (auto [resultIndex, resultType] : llvm::enumerate(resultTypes)) {
      auto maskType = dyn_cast<MaskType>(resultType);
      if (!maskType)
        return rewriter.notifyMatchFailure(
            op, "dense mask lane_stride unpack requires mask result type");
      int64_t sourceIndex = resultIndex / laneStride;
      int64_t part = resultIndex % laneStride;
      Value source = sourceParts[sourceIndex];
      StringAttr firstPart = laneStride == 4 ? (part >= 2 ? higher : lower)
                                             : (part == 1 ? higher : lower);
      Value current =
          rewriter.create<PunpackOp>(op->getLoc(), maskType, source, firstPart);
      if (laneStride == 4)
        current = rewriter.create<PunpackOp>(op->getLoc(), maskType, current,
                                             part % 2 == 0 ? lower : higher);
      results.push_back(current);
    }
    return results;
  }

  if (sourceLayout.isContiguous() && sourceLayout.getLaneStride() != 1 &&
      resultLayout.isContiguous() && resultLayout.getLaneStride() == 1) {
    if (sourceParts.empty())
      return rewriter.notifyMatchFailure(
          op, "dense mask lane_stride pack materialization requires source "
              "parts");
    int64_t laneStride = sourceLayout.getLaneStride();
    if (laneStride != 2 && laneStride != 4)
      return rewriter.notifyMatchFailure(
          op, "unsupported dense mask lane_stride pack factor");
    if (static_cast<int64_t>(sourceParts.size()) >
        static_cast<int64_t>(resultTypes.size()) * laneStride)
      return rewriter.notifyMatchFailure(
          op, "dense mask lane_stride pack materialization source arity does "
              "not fit result arity");
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    auto lower = rewriter.getStringAttr("LOWER");
    auto higher = rewriter.getStringAttr("HIGHER");
    Value allTrue;
    auto mergeMasks = [&](Value lhs, Value rhs) -> FailureOr<Value> {
      if (!allTrue) {
        FailureOr<Value> mask = createAllTrueMask(
            op->getLoc(), cast<MaskType>(lhs.getType()), rewriter);
        if (failed(mask))
          return failure();
        allTrue = *mask;
      }
      return rewriter.create<PorOp>(op->getLoc(), lhs.getType(), lhs, rhs,
                                    allTrue)
          .getResult();
    };
    auto packPair = [&](Value lowSource, std::optional<Value> highSource,
                        MaskType maskType) -> FailureOr<Value> {
      Value packed =
          rewriter.create<PpackOp>(op->getLoc(), maskType, lowSource, lower);
      if (!highSource)
        return packed;
      Value higherPacked = rewriter.create<PpackOp>(
          op->getLoc(), maskType, *highSource, higher);
      return mergeMasks(packed, higherPacked);
    };
    for (auto [resultIndex, resultType] :
         llvm::enumerate(resultTypes)) {
      auto maskType = dyn_cast<MaskType>(resultType);
      if (!maskType)
        return rewriter.notifyMatchFailure(
            op, "dense mask lane_stride pack requires mask result type");
      size_t base = resultIndex * static_cast<size_t>(laneStride);
      if (base >= sourceParts.size())
        break;

      std::optional<Value> source1;
      if (base + 1 < sourceParts.size())
        source1 = sourceParts[base + 1];
      FailureOr<Value> lowHalf = packPair(sourceParts[base], source1, maskType);
      if (failed(lowHalf))
        return failure();
      Value current = *lowHalf;
      if (laneStride == 4) {
        current =
            rewriter.create<PpackOp>(op->getLoc(), maskType, current, lower);
        if (base + 2 < sourceParts.size()) {
          std::optional<Value> source3;
          if (base + 3 < sourceParts.size())
            source3 = sourceParts[base + 3];
          FailureOr<Value> highHalf =
              packPair(sourceParts[base + 2], source3, maskType);
          if (failed(highHalf))
            return failure();
          Value higherPacked = rewriter.create<PpackOp>(
              op->getLoc(), maskType, *highHalf, higher);
          FailureOr<Value> merged = mergeMasks(current, higherPacked);
          if (failed(merged))
            return failure();
          current = *merged;
        }
      }
      results.push_back(current);
    }
    if (results.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(
          op, "dense mask lane_stride pack materialization result arity "
              "mismatch");
    return results;
  }

  (void)rewriter.notifyMatchFailure(
      op, "unsupported VMI mask layout materialization");
  return failure();
}

int getMaskGranularityRank(StringRef granularity) {
  if (granularity == "b8")
    return 0;
  if (granularity == "b16")
    return 1;
  if (granularity == "b32")
    return 2;
  return -1;
}

StringRef getMaskGranularityForRank(int rank) {
  switch (rank) {
  case 0:
    return "b8";
  case 1:
    return "b16";
  case 2:
    return "b32";
  default:
    return "";
  }
}

LogicalResult checkSupportedMaskGranularityMaterialization(
    VMIMaskType sourceType,
    VMIMaskType resultType, std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  if (sourceType.getElementCount() != resultType.getElementCount())
    return fail("requires source and result mask lane counts to match");
  if (sourceType.getLayoutAttr() != resultType.getLayoutAttr())
    return fail("requires source and result mask layouts to match");

  if (!VMIMaskType::isConcreteGranularity(sourceType.getGranularity()) ||
      !VMIMaskType::isConcreteGranularity(resultType.getGranularity()))
    return fail("requires concrete b8/b16/b32 source and result "
                "granularities");

  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (failed(sourceArity) || failed(resultArity))
    return fail("requires computable source/result physical arity");
  if (*sourceArity < 1 || *resultArity < 1)
    return fail("requires non-empty source/result physical arity");

  return success();
}

FailureOr<SmallVector<Value>> materializeAdjacentMaskGranularityConversion(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, PatternRewriter &rewriter) {
  auto fail = [&](const Twine &message) -> FailureOr<SmallVector<Value>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };

  int sourceRank = getMaskGranularityRank(sourceType.getGranularity());
  int resultRank = getMaskGranularityRank(resultType.getGranularity());
  if (std::abs(sourceRank - resultRank) != 1)
    return fail("mask granularity conversion must be adjacent");

  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> factor = getVMITypeLayoutFactor(sourceType);
  if (failed(sourceArity) || failed(factor) ||
      static_cast<int64_t>(sourceParts.size()) != *sourceArity)
    return fail("source mask part count does not match source VMI type");

  MLIRContext *ctx = op->getContext();
  auto partAttr = [&](StringRef part) { return StringAttr::get(ctx, part); };
  auto resultMaskType = MaskType::get(ctx, resultType.getGranularity());
  SmallVector<Value> results;

  int64_t sourceOffset = 0;
  for (int64_t part = 0; part < *factor; ++part) {
    FailureOr<int64_t> sourceChunks = getVMITypeChunksInPart(sourceType, part);
    FailureOr<int64_t> resultChunks = getVMITypeChunksInPart(resultType, part);
    if (failed(sourceChunks) || failed(resultChunks))
      return fail("requires computable source/result chunks per layout part");

    if (resultRank > sourceRank) {
      int64_t produced = 0;
      for (int64_t chunk = 0; chunk < *sourceChunks && produced < *resultChunks;
           ++chunk) {
        Value source = sourceParts[sourceOffset + chunk];
        results.push_back(rewriter
                              .create<PunpackOp>(op->getLoc(), resultMaskType,
                                                 source, partAttr("LOWER"))
                              .getResult());
        ++produced;
        if (produced >= *resultChunks)
          break;
        results.push_back(rewriter
                              .create<PunpackOp>(op->getLoc(), resultMaskType,
                                                 source, partAttr("HIGHER"))
                              .getResult());
        ++produced;
      }
      if (produced != *resultChunks)
        return fail("widening mask granularity conversion produced the wrong "
                    "number of result chunks");
    } else {
      Value allTrue;
      int64_t consumed = 0;
      for (int64_t chunk = 0; chunk < *resultChunks; ++chunk) {
        if (consumed >= *sourceChunks)
          return fail("narrowing mask granularity conversion ran out of "
                      "source chunks");
        Value lowerSource = sourceParts[sourceOffset + consumed++];
        Value packed = rewriter
                           .create<PpackOp>(op->getLoc(), resultMaskType,
                                            lowerSource, partAttr("LOWER"))
                           .getResult();
        if (consumed < *sourceChunks) {
          Value higherSource = sourceParts[sourceOffset + consumed++];
          Value higher = rewriter
                             .create<PpackOp>(op->getLoc(), resultMaskType,
                                              higherSource, partAttr("HIGHER"))
                             .getResult();
          if (!allTrue) {
            FailureOr<Value> mask =
                createAllTrueMask(op->getLoc(), resultMaskType, rewriter);
            if (failed(mask))
              return fail("failed to create all-true mask for ppack merge");
            allTrue = *mask;
          }
          packed = rewriter
                       .create<PorOp>(op->getLoc(), resultMaskType, packed,
                                      higher, allTrue)
                       .getResult();
        }
        results.push_back(packed);
      }
      if (consumed != *sourceChunks)
        return fail("narrowing mask granularity conversion left unused source "
                    "chunks");
    }

    sourceOffset += *sourceChunks;
  }

  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (failed(resultArity) ||
      static_cast<int64_t>(results.size()) != *resultArity)
    return fail("mask granularity conversion result count mismatch");
  return results;
}

FailureOr<SmallVector<Value>> materializeMaskGranularityConversion(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType, ValueRange sourceParts,
    PatternRewriter &rewriter) {
  std::string reason;
  if (failed(checkSupportedMaskGranularityMaterialization(sourceType, resultType, &reason))) {
    (void)rewriter.notifyMatchFailure(op, reason);
    return failure();
  }

  int currentRank = getMaskGranularityRank(sourceType.getGranularity());
  int resultRank = getMaskGranularityRank(resultType.getGranularity());
  if (std::abs(currentRank - resultRank) == 1)
    return materializeAdjacentMaskGranularityConversion(
        op, sourceType, resultType, sourceParts, rewriter);

  VMIMaskType currentType = sourceType;
  SmallVector<Value> currentParts(sourceParts.begin(), sourceParts.end());
  while (currentRank != resultRank) {
    currentRank += currentRank < resultRank ? 1 : -1;
    StringRef nextGranularity = getMaskGranularityForRank(currentRank);
    if (nextGranularity.empty()) {
      (void)rewriter.notifyMatchFailure(op,
                                        "invalid target mask granularity rank");
      return failure();
    }
    VMIMaskType nextType =
        VMIMaskType::get(op->getContext(), currentType.getElementCount(),
                         nextGranularity, currentType.getLayoutAttr());
    FailureOr<SmallVector<Value>> nextParts =
        materializeAdjacentMaskGranularityConversion(op, currentType, nextType,
                                                     currentParts, rewriter);
    if (failed(nextParts))
      return failure();
    currentType = nextType;
    currentParts = std::move(*nextParts);
  }

  return currentParts;
}

FailureOr<SmallVector<Type>> getConvertedMaskPartTypes(VMIMaskType type) {
  FailureOr<int64_t> arity = getVMIPhysicalArity(type);
  FailureOr<StringRef> physicalGranularity =
      getVMIMaskPhysicalGranularity(type);
  if (failed(arity) || failed(physicalGranularity) || *arity < 0)
    return failure();
  SmallVector<Type> types;
  types.reserve(*arity);
  Type partType = MaskType::get(type.getContext(), *physicalGranularity);
  for (int64_t i = 0; i < *arity; ++i)
    types.push_back(partType);
  return types;
}

static FailureOr<VMILayoutAttr>
getVMIMaskPhysicalCarrierLayout(VMIMaskType type) {
  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout)
    return failure();
  MLIRContext *ctx = type.getContext();
  if (layout.isContiguous())
    return VMILayoutAttr::getContiguous(ctx);
  if (layout.isDeinterleaved())
    return VMILayoutAttr::getDeinterleaved(ctx, layout.getFactor());
  if (layout.isBlockDeinterleaved())
    return VMILayoutAttr::getBlockDeinterleaved(ctx, layout.getFactor());
  if (layout.isGroupSlots())
    return VMILayoutAttr::getGroupSlots(ctx, layout.getNumGroups(),
                                        layout.getSlots());
  return failure();
}

static FailureOr<VMIMaskType>
getVMIMaskPhysicalCarrierType(VMIMaskType type) {
  FailureOr<StringRef> physicalGranularity =
      getVMIMaskPhysicalGranularity(type);
  FailureOr<VMILayoutAttr> physicalLayout =
      getVMIMaskPhysicalCarrierLayout(type);
  if (failed(physicalGranularity) || failed(physicalLayout))
    return failure();
  return VMIMaskType::get(type.getContext(), type.getElementCount(),
                          *physicalGranularity, *physicalLayout);
}

static bool isElementDeinterleavedLayout(VMILayoutAttr layout,
                                         int64_t factor) {
  return layout && layout.isDeinterleaved() && layout.getFactor() == factor &&
         layout.getLaneStride() == 1;
}

FailureOr<Value> createAllFalseMaskLike(Location loc, Value value,
                                        PatternRewriter &rewriter) {
  auto maskType = dyn_cast<MaskType>(value.getType());
  if (!maskType)
    return failure();
  return createPrefixMask(loc, maskType, "PAT_ALLF", rewriter);
}

FailureOr<SmallVector<Value>> materializeStagingDeintToContiguousMaskLayout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    int64_t factor, PatternRewriter &rewriter) {
  auto fail = [&](const Twine &message) -> FailureOr<SmallVector<Value>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  if ((factor != 2 && factor != 4) || sourceParts.empty() ||
      sourceParts.size() % factor != 0)
    return fail("staging deinterleaved mask layout requires grouped source "
                "parts");

  int64_t groups = sourceParts.size() / factor;
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (int64_t i = 0; i < groups && results.size() < resultTypes.size(); ++i) {
    auto nextType = [&](int64_t offset) -> Type {
      size_t index = results.size() + offset;
      return index < resultTypes.size() ? resultTypes[index]
                                        : resultTypes[results.size()];
    };
    if (factor == 2) {
      FailureOr<std::pair<Value, Value>> materialized = createPredicateIntlv(
          op->getLoc(), nextType(0), nextType(1), sourceParts[i],
          sourceParts[groups + i], rewriter);
      if (failed(materialized))
        return fail("unsupported predicate intlv staging mask type");
      results.push_back(materialized->first);
      if (results.size() < resultTypes.size())
        results.push_back(materialized->second);
      continue;
    }

    Value p0 = sourceParts[i];
    Value p1 = sourceParts[groups + i];
    Value p2 = sourceParts[2 * groups + i];
    Value p3 = sourceParts[3 * groups + i];
    FailureOr<std::pair<Value, Value>> even =
        createPredicateIntlv(op->getLoc(), nextType(0), nextType(1), p0, p2,
                             rewriter);
    FailureOr<std::pair<Value, Value>> odd =
        createPredicateIntlv(op->getLoc(), nextType(0), nextType(1), p1, p3,
                             rewriter);
    if (failed(even) || failed(odd))
      return fail("unsupported predicate intlv staging mask type");
    FailureOr<std::pair<Value, Value>> low = createPredicateIntlv(
        op->getLoc(), nextType(0), nextType(1), even->first, odd->first,
        rewriter);
    FailureOr<std::pair<Value, Value>> high = createPredicateIntlv(
        op->getLoc(), nextType(2), nextType(3), even->second, odd->second,
        rewriter);
    if (failed(low) || failed(high))
      return fail("unsupported predicate intlv staging mask type");
    results.push_back(low->first);
    if (results.size() < resultTypes.size())
      results.push_back(low->second);
    if (results.size() < resultTypes.size())
      results.push_back(high->first);
    if (results.size() < resultTypes.size())
      results.push_back(high->second);
  }
  if (results.size() != resultTypes.size())
    return fail("staging deinterleaved mask layout result arity mismatch");
  return results;
}

FailureOr<SmallVector<Value>> materializeStagingContiguousToDeintMaskLayout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    int64_t factor, PatternRewriter &rewriter) {
  auto fail = [&](const Twine &message) -> FailureOr<SmallVector<Value>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  if ((factor != 2 && factor != 4) || sourceParts.empty() ||
      resultTypes.size() % factor != 0)
    return fail("staging contiguous mask layout requires grouped result parts");

  int64_t groups = resultTypes.size() / factor;
  if (sourceParts.size() > static_cast<size_t>(groups * factor))
    return fail("staging contiguous mask layout has too many source parts");

  SmallVector<SmallVector<Value, 4>, 4> parts(factor);
  for (int64_t part = 0; part < factor; ++part)
    parts[part].reserve(groups);

  for (int64_t i = 0; i < groups; ++i) {
    size_t sourceBase = static_cast<size_t>(i * factor);
    if (sourceBase >= sourceParts.size())
      return fail("staging contiguous mask layout ran out of source parts");

    SmallVector<Value, 4> sources;
    sources.reserve(factor);
    for (int64_t lane = 0; lane < factor; ++lane) {
      size_t index = sourceBase + lane;
      if (index < sourceParts.size()) {
        sources.push_back(sourceParts[index]);
        continue;
      }
      FailureOr<Value> zero =
          createAllFalseMaskLike(op->getLoc(), sourceParts[sourceBase],
                                 rewriter);
      if (failed(zero))
        return fail("failed to create all-false staging mask");
      sources.push_back(*zero);
    }

    if (factor == 2) {
      FailureOr<std::pair<Value, Value>> materialized =
          createPredicateDintlv(op->getLoc(), resultTypes[i],
                                resultTypes[groups + i], sources[0],
                                sources[1], rewriter);
      if (failed(materialized))
        return fail("unsupported predicate dintlv staging mask type");
      parts[0].push_back(materialized->first);
      parts[1].push_back(materialized->second);
      continue;
    }

    FailureOr<std::pair<Value, Value>> low = createPredicateDintlv(
        op->getLoc(), resultTypes[i], resultTypes[groups + i], sources[0],
        sources[1], rewriter);
    FailureOr<std::pair<Value, Value>> high = createPredicateDintlv(
        op->getLoc(), resultTypes[2 * groups + i],
        resultTypes[3 * groups + i], sources[2], sources[3], rewriter);
    if (failed(low) || failed(high))
      return fail("unsupported predicate dintlv staging mask type");
    FailureOr<std::pair<Value, Value>> even = createPredicateDintlv(
        op->getLoc(), resultTypes[i], resultTypes[2 * groups + i], low->first,
        high->first, rewriter);
    FailureOr<std::pair<Value, Value>> odd = createPredicateDintlv(
        op->getLoc(), resultTypes[groups + i], resultTypes[3 * groups + i],
        low->second, high->second, rewriter);
    if (failed(even) || failed(odd))
      return fail("unsupported predicate dintlv staging mask type");
    parts[0].push_back(even->first);
    parts[1].push_back(odd->first);
    parts[2].push_back(even->second);
    parts[3].push_back(odd->second);
  }

  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (int64_t part = 0; part < factor; ++part) {
    if (parts[part].size() != static_cast<size_t>(groups))
      return fail("staging contiguous mask layout result arity mismatch");
    results.append(parts[part]);
  }
  return results;
}

FailureOr<SmallVector<Value>> materializeMaskGranularityCastLayoutConversion(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, TypeRange resultTypes, PatternRewriter &rewriter);

FailureOr<SmallVector<Value>>
materializeMaskGranularityCastLayoutConversionViaContiguous(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, TypeRange resultTypes, PatternRewriter &rewriter) {
  VMILayoutAttr contiguous = VMILayoutAttr::getContiguous(op->getContext());
  VMIMaskType contiguousType =
      VMIMaskType::get(op->getContext(), sourceType.getElementCount(),
                       sourceType.getGranularity(), contiguous);
  FailureOr<SmallVector<Type>> contiguousTypes =
      getConvertedMaskPartTypes(contiguousType);
  if (failed(contiguousTypes))
    return failure();
  FailureOr<SmallVector<Value>> contiguousParts =
      materializeMaskGranularityCastLayoutConversion(
          op, sourceType, contiguousType, sourceParts, *contiguousTypes,
          rewriter);
  if (failed(contiguousParts))
    return failure();
  return materializeMaskGranularityCastLayoutConversion(
      op, contiguousType, resultType, *contiguousParts, resultTypes, rewriter);
}

FailureOr<SmallVector<Value>> materializeMaskGranularityCastLayoutConversion(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, TypeRange resultTypes, PatternRewriter &rewriter) {
  auto fail = [&](const Twine &message) -> FailureOr<SmallVector<Value>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };

  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !resultLayout)
    return fail("mask granularity cast layout conversion requires layouts");

  if (sourceLayout == resultLayout) {
    if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                            rewriter)))
      return failure();
    return SmallVector<Value>(sourceParts.begin(), sourceParts.end());
  }

  FailureOr<SmallVector<Value>> layoutParts = materializeMaskLayoutConversion(
      op, sourceParts, resultTypes, sourceLayout, resultLayout, rewriter);
  if (succeeded(layoutParts))
    return layoutParts;

  bool sourceC = sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1;
  bool resultC = resultLayout.isContiguous() && resultLayout.getLaneStride() == 1;
  if (isElementDeinterleavedLayout(sourceLayout, 2) && resultC)
    return materializeStagingDeintToContiguousMaskLayout(
        op, sourceParts, resultTypes, /*factor=*/2, rewriter);
  if (sourceC && isElementDeinterleavedLayout(resultLayout, 2))
    return materializeStagingContiguousToDeintMaskLayout(
        op, sourceParts, resultTypes, /*factor=*/2, rewriter);
  if (isElementDeinterleavedLayout(sourceLayout, 4) && resultC)
    return materializeStagingDeintToContiguousMaskLayout(
        op, sourceParts, resultTypes, /*factor=*/4, rewriter);
  if (sourceC && isElementDeinterleavedLayout(resultLayout, 4))
    return materializeStagingContiguousToDeintMaskLayout(
        op, sourceParts, resultTypes, /*factor=*/4, rewriter);

  if (sourceLayout.isDenseSplit() || resultLayout.isDenseSplit())
    return materializeMaskGranularityCastLayoutConversionViaContiguous(
        op, sourceType, resultType, sourceParts, resultTypes, rewriter);

  return fail("unsupported mask granularity cast layout conversion");
}

FailureOr<SmallVector<Value>> materializeMaskGranularityCastConversion(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, TypeRange resultTypes, PatternRewriter &rewriter) {
  auto fail = [&](const Twine &message) -> FailureOr<SmallVector<Value>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };

  if (sourceType.getElementCount() != resultType.getElementCount())
    return fail("requires source and result mask lane counts to match");

  FailureOr<VMIMaskType> physicalSourceType =
      getVMIMaskPhysicalCarrierType(sourceType);
  FailureOr<VMIMaskType> physicalResultType =
      getVMIMaskPhysicalCarrierType(resultType);
  if (failed(physicalSourceType) || failed(physicalResultType))
    return fail("requires source/result mask physical carrier types");

  if (*physicalSourceType == *physicalResultType) {
    if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                            rewriter)))
      return failure();
    return SmallVector<Value>(sourceParts.begin(), sourceParts.end());
  }

  if (physicalSourceType->getLayoutAttr() == physicalResultType->getLayoutAttr())
    return materializeMaskGranularityConversion(op, *physicalSourceType,
                                                *physicalResultType,
                                                sourceParts, rewriter);

  VMIMaskType granularityType =
      VMIMaskType::get(op->getContext(), sourceType.getElementCount(),
                       physicalResultType->getGranularity(),
                       physicalSourceType->getLayoutAttr());
  FailureOr<SmallVector<Value>> granularityParts =
      materializeMaskGranularityConversion(op, *physicalSourceType,
                                           granularityType, sourceParts,
                                           rewriter);
  if (failed(granularityParts))
    return failure();
  return materializeMaskGranularityCastLayoutConversion(
      op, granularityType, *physicalResultType, *granularityParts, resultTypes,
      rewriter);
}

struct OneToNVMIEnsureLayoutOpPattern
    : OpConversionPattern<VMIEnsureLayoutOp> {
  using OpConversionPattern<VMIEnsureLayoutOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIEnsureLayoutOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType = cast<VMIVRegType>(op.getSource().getType());
    auto resultType = cast<VMIVRegType>(op.getResult().getType());
    FailureOr<SmallVector<Value>> results = materializeEnsureLayoutConversion(
        op, adaptor.getSource(), sourceType, resultType,
        *this->getTypeConverter(), rewriter);
    if (failed(results))
      return failure();
    replaceOpWithFlatConvertedValues(rewriter, op, *results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIEnsureMaskLayoutOpPattern
    : OpConversionPattern<VMIEnsureMaskLayoutOp> {
  using OpConversionPattern<
      VMIEnsureMaskLayoutOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIEnsureMaskLayoutOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType = cast<VMIMaskType>(op.getSource().getType());
    auto resultType = cast<VMIMaskType>(op.getResult().getType());
    VMILayoutSupport supports;
    std::string supportReason;
    if (failed(supports.getEnsureMaskLayoutFact(sourceType, resultType,
                                                &supportReason)))
      return rewriter.notifyMatchFailure(
          op, Twine("ensure_mask_layout has no registered materialization "
                    "support: ") +
                  supportReason);
    if (sourceType.getGranularity() != resultType.getGranularity())
      return rewriter.notifyMatchFailure(
          op, "mask layout helper cannot also change granularity");
    VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
    VMILayoutAttr resultLayout = resultType.getLayoutAttr();

    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    FailureOr<SmallVector<Value>> results = materializeMaskLayoutConversion(
        op, sourceParts, resultTypes, sourceLayout, resultLayout, rewriter);
    if (failed(results))
      return failure();
    replaceOpWithFlatConvertedValues(rewriter, op, *results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIEnsureMaskGranularityOpPattern
    : OpConversionPattern<VMIEnsureMaskGranularityOp> {
  using OpConversionPattern<
      VMIEnsureMaskGranularityOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIEnsureMaskGranularityOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType = cast<VMIMaskType>(op.getSource().getType());
    auto resultType = cast<VMIMaskType>(op.getResult().getType());
    VMILayoutSupport supports;
    bool identity = sourceType.getGranularity() == resultType.getGranularity() &&
                    sourceType.getLayoutAttr() == resultType.getLayoutAttr();
    if (!identity) {
      std::string reason;
      if (failed(supports.getMaskGranularityCastLayoutFactForLayouts(
              sourceType, resultType, sourceType.getLayoutAttr(),
              resultType.getLayoutAttr(), &reason)))
        return rewriter.notifyMatchFailure(
            op, "unsupported mask granularity cast layout relation: " + reason);
    }

    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);

    FailureOr<SmallVector<Value>> results =
        materializeMaskGranularityCastConversion(
            op, sourceType, resultType, sourceParts, resultTypes, rewriter);
    if (failed(results))
      return failure();
    if (results->size() != resultTypes.size())
      return rewriter.notifyMatchFailure(
          op, "mask granularity cast result arity mismatch");
    for (auto [result, type] : llvm::zip_equal(*results, resultTypes))
      if (result.getType() != type)
        return rewriter.notifyMatchFailure(
            op, "mask granularity cast result type mismatch");
    replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                     *this->getTypeConverter());
    return success();
  }

private:
  ;
};

struct OneToNVMIBroadcastOpPattern : OpConversionPattern<VMIBroadcastOp> {
  using OpConversionPattern<VMIBroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIBroadcastOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange inputParts = adaptor.getValue();
    if (inputParts.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "broadcast input must convert to one value");
    bool inputIsVReg = isa<VMIVRegType>(op.getValue().getType());

    FailureOr<SmallVector<Type>> maybe_resultTypes =

        getConvertedResultTypes(op, 0, *this->getTypeConverter());

    if (failed(maybe_resultTypes))

      return failure();

    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto vregType = dyn_cast<VRegType>(resultType);
      if (!vregType)
        return rewriter.notifyMatchFailure(op, "broadcast result must be vreg");
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for broadcast mask");
      StringAttr position =
          inputIsVReg ? rewriter.getStringAttr("LOWEST") : StringAttr{};
      results.push_back(rewriter
                            .create<VdupOp>(op.getLoc(), resultType,
                                            inputParts.front(), *mask, position)
                            .getResult());
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

FailureOr<Value> createScalarOffsetConstant(Location loc, Type type,
                                            int64_t value,
                                            PatternRewriter &rewriter) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return rewriter
        .create<arith::ConstantOp>(loc, IntegerAttr::get(intType, value))
        .getResult();
  }
  if (auto floatType = dyn_cast<FloatType>(type)) {
    return rewriter
        .create<arith::ConstantOp>(
            loc, rewriter.getFloatAttr(floatType, static_cast<double>(value)))
        .getResult();
  }
  return failure();
}

FailureOr<Value> createIotaChunkBase(Location loc, Value base,
                                     int64_t laneOffset, StringRef order,
                                     PatternRewriter &rewriter) {
  if (laneOffset == 0)
    return base;

  FailureOr<Value> offset =
      createScalarOffsetConstant(loc, base.getType(), laneOffset, rewriter);
  if (failed(offset))
    return failure();

  if (isa<IntegerType>(base.getType())) {
    if (order == "DESC")
      return rewriter.create<arith::SubIOp>(loc, base, *offset).getResult();
    return rewriter.create<arith::AddIOp>(loc, base, *offset).getResult();
  }
  if (isa<FloatType>(base.getType())) {
    if (order == "DESC")
      return rewriter.create<arith::SubFOp>(loc, base, *offset).getResult();
    return rewriter.create<arith::AddFOp>(loc, base, *offset).getResult();
  }

  return failure();
}

FailureOr<Value> createIotaContiguousChunk(Location loc, Type resultType,
                                           Value base, int64_t laneOffset,
                                           StringAttr orderAttr,
                                           PatternRewriter &rewriter) {
  StringRef order = orderAttr ? orderAttr.getValue() : StringRef("ASC");
  FailureOr<Value> chunkBase =
      createIotaChunkBase(loc, base, laneOffset, order, rewriter);
  if (failed(chunkBase))
    return failure();
  return rewriter.create<VciOp>(loc, resultType, *chunkBase, orderAttr)
      .getResult();
}

FailureOr<Value> createIotaDeinterleavedChunk(Location loc, Type resultType,
                                              Value base, int64_t factor,
                                              int64_t part, int64_t chunk,
                                              int64_t lanesPerPart,
                                              StringAttr orderAttr,
                                              PatternRewriter &rewriter) {
  auto vregType = dyn_cast<VRegType>(resultType);
  if (!vregType)
    return failure();

  FailureOr<Value> mask = createAllTrueMaskForVReg(loc, vregType, rewriter);
  FailureOr<Value> zero =
      createScalarOffsetConstant(loc, base.getType(), 0, rewriter);
  FailureOr<Value> factorScalar =
      createScalarOffsetConstant(loc, base.getType(), factor, rewriter);
  if (failed(mask) || failed(zero) || failed(factorScalar))
    return failure();

  Value local =
      rewriter.create<VciOp>(loc, resultType, *zero, StringAttr{}).getResult();
  Value scaled =
      rewriter.create<VmulsOp>(loc, resultType, local, *factorScalar, *mask)
          .getResult();

  StringRef order = orderAttr ? orderAttr.getValue() : StringRef("ASC");
  int64_t partOffset = part + factor * chunk * lanesPerPart;
  FailureOr<Value> biasedBase =
      createIotaChunkBase(loc, base, partOffset, order, rewriter);
  if (failed(biasedBase))
    return failure();

  if (order == "DESC") {
    Value baseVector = rewriter
                           .create<VdupOp>(loc, resultType, *biasedBase, *mask,
                                           /*position=*/nullptr)
                           .getResult();
    return rewriter.create<VsubOp>(loc, resultType, baseVector, scaled, *mask)
        .getResult();
  }

  return rewriter.create<VaddsOp>(loc, resultType, scaled, *biasedBase, *mask)
      .getResult();
}

struct OneToNVMIIotaOpPattern : OpConversionPattern<VMIIotaOp> {
  using OpConversionPattern<VMIIotaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIIotaOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    VMILayoutAttr layout = resultVMIType.getLayoutAttr();
    if (!layout)
      return rewriter.notifyMatchFailure(op, "iota requires assigned layout");

    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(resultVMIType.getElementType());
    if (failed(lanesPerPart))
      return rewriter.notifyMatchFailure(
          op, "iota requires known physical lanes per part");

    FailureOr<Value> base = getSingleValue(
        op, adaptor.getBase(), "iota base must convert to one value", rewriter);
    if (failed(base))
      return failure();

    FailureOr<SmallVector<Type>> maybe_resultTypes =

        getConvertedResultTypes(op, 0, *this->getTypeConverter());

    if (failed(maybe_resultTypes))

      return failure();

    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    SmallVector<Value> results;
    results.reserve(resultTypes.size());

    if (layout.isContiguous()) {
      for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
        if (!isa<VRegType>(resultType))
          return rewriter.notifyMatchFailure(op, "iota result must be vreg");
        FailureOr<Value> result = createIotaContiguousChunk(
            op.getLoc(), resultType, *base,
            static_cast<int64_t>(index) * *lanesPerPart, op.getOrderAttr(),
            rewriter);
        if (failed(result))
          return rewriter.notifyMatchFailure(
              op, "failed to materialize contiguous iota chunk");
        results.push_back(*result);
      }
      replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
      return success();
    }

    int64_t factor = layout.getFactor();
    if (resultTypes.size() % factor != 0)
      return rewriter.notifyMatchFailure(
          op, "deinterleaved iota physical result count does not match "
              "layout factor");
    int64_t chunksPerPart = resultTypes.size() / factor;
    for (int64_t part = 0; part < factor; ++part) {
      for (int64_t chunk = 0; chunk < chunksPerPart; ++chunk) {
        Type resultType = resultTypes[part * chunksPerPart + chunk];
        FailureOr<Value> result = createIotaDeinterleavedChunk(
            op.getLoc(), resultType, *base, factor, part, chunk, *lanesPerPart,
            op.getOrderAttr(), rewriter);
        if (failed(result))
          return rewriter.notifyMatchFailure(
              op, "failed to materialize deinterleaved iota chunk");
        results.push_back(*result);
      }
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIConstantOpPattern : OpConversionPattern<VMIConstantOp> {
  using OpConversionPattern<VMIConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIConstantOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!denseAttr || !denseAttr.isSplat())
      return rewriter.notifyMatchFailure(
          op, "only splat dense data constants are supported");
    auto splatAttr = dyn_cast<TypedAttr>(denseAttr.getSplatValue<Attribute>());
    if (!splatAttr)
      return rewriter.notifyMatchFailure(op, "splat constant must be typed");

    // arith.constant only accepts signless integer types, whereas VMI vregs may
    // carry signed/unsigned element types (e.g. ui16). Remap an unsigned/signed
    // integer splat to its signless equivalent; the downstream pto.vdup accepts
    // a signless scalar for a signed/unsigned result element.
    if (auto intAttr = dyn_cast<IntegerAttr>(splatAttr)) {
      if (auto intTy = dyn_cast<IntegerType>(intAttr.getType());
          intTy && !intTy.isSignless())
        splatAttr = IntegerAttr::get(rewriter.getIntegerType(intTy.getWidth()),
                                     intAttr.getValue());
    }

    Value scalar =
        rewriter.create<arith::ConstantOp>(op.getLoc(), splatAttr).getResult();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto vregType = dyn_cast<VRegType>(resultType);
      if (!vregType)
        return rewriter.notifyMatchFailure(op, "constant result must be vreg");
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for constant mask");
      results.push_back(rewriter
                            .create<VdupOp>(op.getLoc(), resultType, scalar,
                                            *mask,
                                            /*position=*/nullptr)
                            .getResult());
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIConstantMaskOpPattern
    : OpConversionPattern<VMIConstantMaskOp> {
  using OpConversionPattern<VMIConstantMaskOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIConstantMaskOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    std::string reason;
    FailureOr<SmallVector<ConstantMaskChunkMaterialization>> materializations =
        computeConstantMaskMaterialization(op, &reason);
    if (failed(materializations))
      return rewriter.notifyMatchFailure(op, Twine("constant_mask ") + reason);

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (const ConstantMaskChunkMaterialization &materialization :
         *materializations) {
      if (results.size() >= resultTypes.size())
        return rewriter.notifyMatchFailure(
            op, "constant_mask produced too many physical masks");
      auto maskType = dyn_cast<MaskType>(resultTypes[results.size()]);
      if (!maskType)
        return rewriter.notifyMatchFailure(op,
                                           "constant_mask result must be mask");
      FailureOr<Value> mask = materializeConstantMaskChunk(
          op.getLoc(), maskType, materialization.activeLanes, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(
            op, "failed to materialize constant_mask physical chunk");
      results.push_back(*mask);
    }

    if (results.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(
          op, "constant_mask physical result count mismatch");
    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMICreateMaskOpPattern
    : OpConversionPattern<VMICreateMaskOp> {
  using OpConversionPattern<VMICreateMaskOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMICreateMaskOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto activeConstant =
        op.getActiveLanes().getDefiningOp<arith::ConstantOp>();
    auto resultVMIType = cast<VMIMaskType>(op.getResult().getType());
    VMILayoutAttr layout = resultVMIType.getLayoutAttr();
    if (!layout ||
        !VMIMaskType::isConcreteGranularity(resultVMIType.getGranularity()))
      return rewriter.notifyMatchFailure(
          op, "create_mask requires concrete layout and granularity");
    FailureOr<StringRef> physicalGranularity =
        getVMIMaskPhysicalGranularity(resultVMIType);
    FailureOr<int64_t> lanesPerPart =
        failed(physicalGranularity)
            ? FailureOr<int64_t>(failure())
            : getMaskLanesPerPart(*physicalGranularity);
    if (failed(lanesPerPart))
      return rewriter.notifyMatchFailure(
          op, "create_mask requires known physical mask lanes per part");

    if (!activeConstant) {
      FailureOr<Value> active = getSingleValue(
          op, adaptor.getActiveLanes(),
          "create_mask active_lanes must convert to one value", rewriter);
      if (failed(active))
        return failure();

      FailureOr<SmallVector<Type>> maybe_resultTypes =

          getConvertedResultTypes(op, 0, *this->getTypeConverter());

      if (failed(maybe_resultTypes))

        return failure();

      SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
      int64_t factor = layout.isDenseSplit() ? layout.getFactor() : 1;
      if (resultTypes.size() % factor != 0)
        return rewriter.notifyMatchFailure(
            op, "dynamic create_mask physical result count does not match "
                "layout factor");
      int64_t chunksPerPart = resultTypes.size() / factor;
      Value activeI32 = clampDynamicActiveLanes(
          op.getLoc(), *active, resultVMIType.getElementCount(), rewriter);

      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      for (int64_t part = 0; part < factor; ++part) {
        Value remaining = createPartitionActiveLanes(op.getLoc(), activeI32,
                                                     factor, part, rewriter);
        for (int64_t chunk = 0; chunk < chunksPerPart; ++chunk) {
          Type resultType = resultTypes[part * chunksPerPart + chunk];
          auto maskType = dyn_cast<MaskType>(resultType);
          if (!maskType)
            return rewriter.notifyMatchFailure(
                op, "create_mask result must be mask");
          FailureOr<std::pair<Value, Value>> maskAndRemaining =
              createRuntimePrefixMask(op.getLoc(), maskType, remaining,
                                      rewriter);
          if (failed(maskAndRemaining))
            return rewriter.notifyMatchFailure(
                op, "unsupported mask type for dynamic create_mask");
          results.push_back(maskAndRemaining->first);
          remaining = maskAndRemaining->second;
        }
      }

      replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
      return success();
    }

    auto activeAttr = dyn_cast<IntegerAttr>(activeConstant.getValue());
    if (!activeAttr)
      return rewriter.notifyMatchFailure(
          op, "create_mask active_lanes must be an integer constant");

    int64_t activeLanes = activeAttr.getInt();
    if (activeLanes < 0)
      activeLanes = 0;
    if (activeLanes > resultVMIType.getElementCount())
      activeLanes = resultVMIType.getElementCount();

    FailureOr<SmallVector<Type>> maybe_resultTypes =

        getConvertedResultTypes(op, 0, *this->getTypeConverter());

    if (failed(maybe_resultTypes))

      return failure();

    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    int64_t factor = layout.isDenseSplit() ? layout.getFactor() : 1;
    SmallVector<Value> results;
    results.reserve(resultTypes.size());

    for (int64_t part = 0; part < factor; ++part) {
      for (int64_t chunk = 0;; ++chunk) {
        bool anyLane = false;
        int64_t activeInChunk = 0;
        for (int64_t lane = 0; lane < *lanesPerPart; ++lane) {
          FailureOr<bool> padding =
              isPaddingLane(resultVMIType, part, chunk, lane);
          if (failed(padding))
            return rewriter.notifyMatchFailure(
                op, "failed to map create_mask physical padding lane");
          if (*padding)
            continue;
          anyLane = true;
          FailureOr<int64_t> logicalLane =
              mapPhysicalLaneToLogical(resultVMIType, part, chunk, lane);
          if (failed(logicalLane))
            return rewriter.notifyMatchFailure(
                op, "failed to map create_mask physical lane");
          if (*logicalLane < activeLanes)
            ++activeInChunk;
        }
        if (!anyLane)
          break;

        if (results.size() >= resultTypes.size())
          return rewriter.notifyMatchFailure(
              op, "create_mask produced too many physical masks");
        auto maskType = dyn_cast<MaskType>(resultTypes[results.size()]);
        if (!maskType)
          return rewriter.notifyMatchFailure(op,
                                             "create_mask result must be mask");
        std::optional<std::string> pattern =
            getPrefixPattern(activeInChunk, *lanesPerPart);
        if (pattern) {
          FailureOr<Value> mask =
              createPrefixMask(op.getLoc(), maskType, *pattern, rewriter);
          if (failed(mask))
            return rewriter.notifyMatchFailure(
                op, "unsupported mask type for create_mask");
          results.push_back(*mask);
          continue;
        }

        FailureOr<std::pair<Value, Value>> maskAndRemaining =
            createRuntimePrefixMask(
                op.getLoc(), maskType,
                createI32Constant(op.getLoc(), activeInChunk, rewriter),
                rewriter);
        if (failed(maskAndRemaining))
          return rewriter.notifyMatchFailure(
              op, "unsupported mask type for create_mask plt fallback");
        results.push_back(maskAndRemaining->first);
      }
    }

    if (results.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(
          op, "create_mask physical result count mismatch");
    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMICreateGroupMaskOpPattern
    : OpConversionPattern<VMICreateGroupMaskOp> {
  using OpConversionPattern<
      VMICreateGroupMaskOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMICreateGroupMaskOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    auto resultVMIType = cast<VMIMaskType>(op.getResult().getType());
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    if (resultLayout && resultLayout.isBlockDeinterleaved() &&
        resultLayout.getFactor() == 4) {
      VMILayoutAttr contiguousLayout =
          VMILayoutAttr::getContiguous(op.getContext());
      auto contiguousType =
          VMIMaskType::get(op.getContext(), resultVMIType.getElementCount(),
                           resultVMIType.getGranularity(), contiguousLayout);
      SmallVector<Value> contiguousParts;
      auto activeConstant =
          op.getActiveElemsPerGroup().getDefiningOp<arith::ConstantOp>();
      if (activeConstant) {
        std::string contiguousReason;
        FailureOr<SmallVector<ConstantMaskChunkMaterialization>>
            contiguousMaterializations = computeGroupMaskMaterializationForType(
                op, contiguousType, &contiguousReason);
        if (failed(contiguousMaterializations))
          return rewriter.notifyMatchFailure(op, Twine("create_group_mask ") +
                                                     contiguousReason);

        contiguousParts.reserve(contiguousMaterializations->size());
        for (const ConstantMaskChunkMaterialization &materialization :
             *contiguousMaterializations) {
          if (contiguousParts.size() >= resultTypes.size())
            return rewriter.notifyMatchFailure(
                op, "create_group_mask produced too many contiguous masks");
          auto maskType =
              dyn_cast<MaskType>(resultTypes[contiguousParts.size()]);
          if (!maskType)
            return rewriter.notifyMatchFailure(
                op, "create_group_mask result must be mask");
          FailureOr<Value> mask = materializeConstantMaskChunk(
              op.getLoc(), maskType, materialization.activeLanes, rewriter);
          if (failed(mask))
            return rewriter.notifyMatchFailure(
                op, "failed to materialize create_group_mask contiguous chunk");
          contiguousParts.push_back(*mask);
        }
      } else {
        FailureOr<Value> active = getSingleValue(
            op, adaptor.getActiveElemsPerGroup(),
            "create_group_mask active_elems_per_group must convert to one "
            "value",
            rewriter);
        if (failed(active))
          return failure();
        FailureOr<SmallVector<Value>> dynamicParts =
            materializeDynamicGroupMaskForType(op, *active, contiguousType,
                                               resultTypes, rewriter);
        if (failed(dynamicParts))
          return failure();
        contiguousParts = std::move(*dynamicParts);
      }

      if (contiguousParts.size() != resultTypes.size())
        return rewriter.notifyMatchFailure(
            op, "create_group_mask contiguous physical result count mismatch");
      FailureOr<SmallVector<Value>> results = materializeMaskLayoutConversion(
          op, contiguousParts, resultTypes, contiguousLayout, resultLayout,
          rewriter);
      if (failed(results))
        return failure();
      replaceOpWithFlatConvertedValues(rewriter, op, *results, *this->getTypeConverter());
      return success();
    }

    auto activeConstant =
        op.getActiveElemsPerGroup().getDefiningOp<arith::ConstantOp>();
    if (!activeConstant) {
      FailureOr<Value> active = getSingleValue(
          op, adaptor.getActiveElemsPerGroup(),
          "create_group_mask active_elems_per_group must convert to one value",
          rewriter);
      if (failed(active))
        return failure();

      VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
      if (resultLayout && resultLayout.isDeinterleaved()) {
        VMILayoutAttr contiguousLayout =
            VMILayoutAttr::getContiguous(op.getContext());
        auto contiguousType =
            VMIMaskType::get(op.getContext(), resultVMIType.getElementCount(),
                             resultVMIType.getGranularity(), contiguousLayout);
        FailureOr<SmallVector<Value>> contiguousParts =
            materializeDynamicGroupMaskForType(op, *active, contiguousType,
                                               resultTypes, rewriter);
        if (failed(contiguousParts))
          return failure();
        FailureOr<SmallVector<Value>> results = materializeMaskLayoutConversion(
            op, *contiguousParts, resultTypes, contiguousLayout, resultLayout,
            rewriter);
        if (failed(results))
          return failure();
        replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                         *this->getTypeConverter());
        return success();
      }

      FailureOr<SmallVector<Value>> results =
          materializeDynamicGroupMaskForType(op, *active, resultVMIType,
                                             resultTypes, rewriter);
      if (failed(results))
        return failure();
      replaceOpWithFlatConvertedValues(rewriter, op, *results, *this->getTypeConverter());
      return success();
    }

    std::string reason;
    FailureOr<SmallVector<ConstantMaskChunkMaterialization>> materializations =
        computeGroupMaskMaterialization(op, &reason);
    if (failed(materializations))
      return rewriter.notifyMatchFailure(op,
                                         Twine("create_group_mask ") + reason);

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (const ConstantMaskChunkMaterialization &materialization :
         *materializations) {
      if (results.size() >= resultTypes.size())
        return rewriter.notifyMatchFailure(
            op, "create_group_mask produced too many physical masks");
      auto maskType = dyn_cast<MaskType>(resultTypes[results.size()]);
      if (!maskType)
        return rewriter.notifyMatchFailure(
            op, "create_group_mask result must be mask");
      FailureOr<Value> mask = materializeConstantMaskChunk(
          op.getLoc(), maskType, materialization.activeLanes, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(
            op, "failed to materialize create_group_mask physical chunk");
      results.push_back(*mask);
    }

    if (results.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(
          op, "create_group_mask physical result count mismatch");
    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMILoadOpPattern : OpConversionPattern<VMILoadOp> {
  using OpConversionPattern<VMILoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMILoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    FailureOr<Value> source =
        getSingleValue(op, adaptor.getSource(),
                       "load source must convert to one value", rewriter);
    FailureOr<Value> offset =
        getSingleValue(op, adaptor.getOffset(),
                       "load offset must convert to one value", rewriter);
    if (failed(source) || failed(offset))
      return failure();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    if (std::optional<std::string> dist =
            getDenseLaneStrideLoadDistToken(resultVMIType)) {
      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      int64_t semanticOffset = 0;
      for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
        if (!isa<VRegType>(resultType))
          return rewriter.notifyMatchFailure(op, "load result must be vreg");
        Value chunkOffset =
            createChunkOffset(op.getLoc(), *offset, semanticOffset, rewriter);
        results.push_back(rewriter
                              .create<VldsOp>(op.getLoc(), resultType,
                                              /*updated_base=*/Type{}, *source,
                                              chunkOffset,
                                              rewriter.getStringAttr(*dist))
                              .getResult());
        FailureOr<int64_t> activeLanes =
            getActiveDataLanesInPhysicalChunk(resultVMIType, index);
        if (failed(activeLanes))
          return rewriter.notifyMatchFailure(
              op, "failed to compute lane_stride load active lanes");
        semanticOffset += *activeLanes;
      }
      replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
      return success();
    }

    FailureOr<int64_t> lanesPerPart = verifyFullOrSafeReadVRegChunks(
        op, resultVMIType, op.getSource().getType(), *offset, rewriter);
    if (failed(lanesPerPart))
      return failure();

    VMILayoutAttr contiguousLayout =
        VMILayoutAttr::getContiguous(rewriter.getContext());
    FailureOr<SmallVector<Type>> maybeContiguousTypes =
        getConvertedVRegTypesWithLayout(resultVMIType, contiguousLayout,
                                        *this->getTypeConverter());
    if (failed(maybeContiguousTypes))
      return rewriter.notifyMatchFailure(
          op, "failed to compute contiguous load footprint");
    SmallVector<Type> contiguousTypes = std::move(*maybeContiguousTypes);
    FailureOr<bool> noWiderThanContiguous =
        hasNoWiderFootprintThanContiguous(resultTypes, contiguousTypes);
    if (failed(noWiderThanContiguous))
      return rewriter.notifyMatchFailure(
          op, "failed to compare load physical footprint");

    if (resultLayout && resultLayout.isDeinterleaved() &&
        resultLayout.getFactor() == 2 && *noWiderThanContiguous) {
      std::optional<std::string> dist =
          getX2MemoryDistToken(resultVMIType.getElementType(), "DINTLV");
      if (dist && !resultTypes.empty() && resultTypes.size() % 2 == 0) {
        int64_t groups = resultTypes.size() / 2;
        SmallVector<Value> lows;
        SmallVector<Value> highs;
        lows.reserve(groups);
        highs.reserve(groups);
        for (int64_t group = 0; group < groups; ++group) {
          Type lowType = resultTypes[group];
          Type highType = resultTypes[groups + group];
          if (lowType != highType)
            return rewriter.notifyMatchFailure(
                op, "vldsx2 requires matching low/high result types");
          Value chunkOffset = createChunkOffset(
              op.getLoc(), *offset, group * 2 * *lanesPerPart, rewriter);
          auto load = rewriter.create<Vldsx2Op>(op.getLoc(), lowType, highType,
                                                *source, chunkOffset,
                                                rewriter.getStringAttr(*dist));
          lows.push_back(load.getLow());
          highs.push_back(load.getHigh());
        }
        SmallVector<Value> results;
        results.reserve(resultTypes.size());
        results.append(lows);
        results.append(highs);
        replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
        return success();
      }
    }

    if (resultLayout && resultLayout.isDeinterleaved() &&
        resultLayout.getFactor() == 4 &&
        *noWiderThanContiguous) {
      std::optional<std::string> dist =
          getX2MemoryDistToken(resultVMIType.getElementType(), "DINTLV");
      if (dist && !resultTypes.empty() && resultTypes.size() % 4 == 0) {
        int64_t groups = resultTypes.size() / 4;
        SmallVector<Value> part0;
        SmallVector<Value> part1;
        SmallVector<Value> part2;
        SmallVector<Value> part3;
        part0.reserve(groups);
        part1.reserve(groups);
        part2.reserve(groups);
        part3.reserve(groups);
        for (int64_t group = 0; group < groups; ++group) {
          Type part0Type = resultTypes[group];
          Type part1Type = resultTypes[groups + group];
          Type part2Type = resultTypes[2 * groups + group];
          Type part3Type = resultTypes[3 * groups + group];
          if (part0Type != part1Type || part0Type != part2Type ||
              part0Type != part3Type)
            return rewriter.notifyMatchFailure(
                op, "vldsx2 deinterleaved=4 load requires matching part "
                    "types");

          Value firstOffset = createChunkOffset(
              op.getLoc(), *offset, group * 4 * *lanesPerPart, rewriter);
          Value secondOffset = createChunkOffset(
              op.getLoc(), *offset, (group * 4 + 2) * *lanesPerPart, rewriter);
          auto first = rewriter.create<Vldsx2Op>(
              op.getLoc(), part0Type, part1Type, *source, firstOffset,
              rewriter.getStringAttr(*dist));
          auto second = rewriter.create<Vldsx2Op>(
              op.getLoc(), part2Type, part3Type, *source, secondOffset,
              rewriter.getStringAttr(*dist));

          auto even =
              rewriter.create<VdintlvOp>(op.getLoc(), part0Type, part2Type,
                                         first.getLow(), second.getLow());
          auto odd =
              rewriter.create<VdintlvOp>(op.getLoc(), part1Type, part3Type,
                                         first.getHigh(), second.getHigh());
          part0.push_back(even.getLow());
          part1.push_back(odd.getLow());
          part2.push_back(even.getHigh());
          part3.push_back(odd.getHigh());
        }

        SmallVector<Value> results;
        results.reserve(resultTypes.size());
        results.append(part0);
        results.append(part1);
        results.append(part2);
        results.append(part3);
        replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
        return success();
      }
    }

    SmallVector<Value> contiguousParts;
    contiguousParts.reserve(contiguousTypes.size());
    for (auto [index, resultType] : llvm::enumerate(contiguousTypes)) {
      auto vregType = dyn_cast<VRegType>(resultType);
      if (!vregType)
        return rewriter.notifyMatchFailure(op, "load result must be vreg");
      Value chunkOffset = createChunkOffset(op.getLoc(), *offset,
                                            index * *lanesPerPart, rewriter);
      contiguousParts.push_back(rewriter
                                    .create<VldsOp>(op.getLoc(), resultType,
                                                    /*updated_base=*/Type{},
                                                    *source, chunkOffset,
                                                    /*dist=*/nullptr)
                                    .getResult());
    }

    FailureOr<SmallVector<Value>> results = materializeDataLayoutConversion(
        op, contiguousParts, resultTypes, contiguousLayout,
        resultVMIType.getLayoutAttr(), resultVMIType.getElementType(),
        rewriter);
    if (failed(results))
      return failure();

    replaceOpWithFlatConvertedValues(rewriter, op, *results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIDeinterleaveLoadOpPattern
    : OpConversionPattern<VMIDeinterleaveLoadOp> {
  using OpConversionPattern<
      VMIDeinterleaveLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIDeinterleaveLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lowVMIType = cast<VMIVRegType>(op.getLow().getType());
    FailureOr<Value> source = getSingleValue(
        op, adaptor.getSource(),
        "deinterleave_load source must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(),
        "deinterleave_load offset must convert to one value", rewriter);
    if (failed(source) || failed(offset))
      return failure();

    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(lowVMIType.getElementType());
    if (failed(lanesPerPart))
      return rewriter.notifyMatchFailure(
          op, "deinterleave_load requires known physical lanes per part");

    std::optional<std::string> dist =
        getX2MemoryDistToken(lowVMIType.getElementType(), "DINTLV");
    if (!dist)
      return rewriter.notifyMatchFailure(
          op, "deinterleave_load requires vldsx2 DINTLV element support");

    FailureOr<SmallVector<Type>> maybe_lowTypes =

        getConvertedResultTypes(op, 0, *this->getTypeConverter());

    if (failed(maybe_lowTypes))

      return failure();

    SmallVector<Type> lowTypes = std::move(*maybe_lowTypes);
    FailureOr<SmallVector<Type>> maybe_highTypes =
        getConvertedResultTypes(op, 1, *this->getTypeConverter());
    if (failed(maybe_highTypes))
      return failure();
    SmallVector<Type> highTypes = std::move(*maybe_highTypes);
    if (lowTypes.size() != highTypes.size())
      return rewriter.notifyMatchFailure(
          op, "deinterleave_load requires matching low/high physical arity");

    SmallVector<Value> lows;
    SmallVector<Value> highs;
    lows.reserve(lowTypes.size());
    highs.reserve(highTypes.size());
    for (size_t index = 0, e = lowTypes.size(); index < e; ++index) {
      Type lowType = lowTypes[index];
      Type highType = highTypes[index];
      if (lowType != highType)
        return rewriter.notifyMatchFailure(
            op, "deinterleave_load requires matching low/high physical types");
      Value chunkOffset = createChunkOffset(
          op.getLoc(), *offset, static_cast<int64_t>(index) * 2 * *lanesPerPart,
          rewriter);
      auto load =
          rewriter.create<Vldsx2Op>(op.getLoc(), lowType, highType, *source,
                                    chunkOffset, rewriter.getStringAttr(*dist));
      lows.push_back(load.getLow());
      highs.push_back(load.getHigh());
    }

    SmallVector<Value> results;
    results.reserve(lows.size() + highs.size());
    results.append(lows);
    results.append(highs);
    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIGroupLoadOpPattern : OpConversionPattern<VMIGroupLoadOp> {
  using OpConversionPattern<VMIGroupLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIGroupLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    FailureOr<Value> source =
        getSingleValue(op, adaptor.getSource(),
                       "group_load source must convert to one value", rewriter);
    FailureOr<Value> offset =
        getSingleValue(op, adaptor.getOffset(),
                       "group_load offset must convert to one value", rewriter);
    FailureOr<Value> rowStride = getSingleValue(
        op, adaptor.getRowStride(),
        "group_load row_stride must convert to one value", rewriter);
    if (failed(source) || failed(offset) || failed(rowStride))
      return failure();

    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    if (resultLayout && resultLayout.isBlockDeinterleaved() &&
        resultVMIType.getElementType().isF32()) {
      FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
          resultVMIType, op.getNumGroupsAttr().getInt());
      if (failed(groupSize))
        return rewriter.notifyMatchFailure(
            op, "group_load requires num_groups to evenly divide lane count");
      if ((*groupSize != 16 || resultLayout.getFactor() != 2) &&
          (*groupSize != 32 || resultLayout.getFactor() != 4))
        return rewriter.notifyMatchFailure(
            op, "block_deinterleaved group_load requires S=16/factor=2 or "
                "S=32/factor=4");
      if (op.getNumGroupsAttr().getInt() % 8 != 0)
        return rewriter.notifyMatchFailure(
            op, "block_deinterleaved group_load requires num_groups multiple "
                "of 8");
      std::optional<int64_t> constantRowStride =
          getConstantIndexValue(op.getRowStride());
      if (!constantRowStride || *constantRowStride <= 0 ||
          *constantRowStride % 8 != 0)
        return rewriter.notifyMatchFailure(
            op, "block_deinterleaved group_load requires constant positive "
                "row_stride "
                "divisible by 8 f32 elements");
      if (!isa<PtrType>((*source).getType()))
        return rewriter.notifyMatchFailure(
            op, "block_deinterleaved group_load requires !pto.ptr source");

      FailureOr<SmallVector<Type>> maybe_resultTypes =

          getConvertedResultTypes(op, 0, *this->getTypeConverter());

      if (failed(maybe_resultTypes))

        return failure();

      SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
      int64_t factor = resultLayout.getFactor();
      FailureOr<int64_t> blockElems =
          getVMILayoutBlockElems(resultVMIType);
      FailureOr<int64_t> chunksPerPart = getDataChunksInPart(resultVMIType, 0);
      if (failed(blockElems) || failed(chunksPerPart) ||
          *chunksPerPart <= 0)
        return rewriter.notifyMatchFailure(
            op, "block_deinterleaved group_load requires known block and "
                "chunks per part");
      for (int64_t part = 1; part < factor; ++part) {
        FailureOr<int64_t> currentChunks =
            getDataChunksInPart(resultVMIType, part);
        if (failed(currentChunks) || *currentChunks != *chunksPerPart)
          return rewriter.notifyMatchFailure(
              op, "block_deinterleaved group_load requires uniform chunks "
                  "per part");
      }
      if (static_cast<int64_t>(resultTypes.size()) != factor * *chunksPerPart)
        return rewriter.notifyMatchFailure(op,
                                           "block_deinterleaved group_load "
                                           "arity mismatch");

      auto makeI16 = [&](int64_t value) -> Value {
        return rewriter.create<arith::ConstantIntOp>(op.getLoc(), value, 16);
      };
      Value blockStride = makeI16(*constantRowStride / 8);
      Value zeroI16 = makeI16(0);
      auto makePtr = [&](Value elementOffset) -> Value {
        return rewriter
            .create<AddPtrOp>(op.getLoc(), (*source).getType(), *source,
                              elementOffset)
            .getResult();
      };

      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      constexpr int64_t kGroupsPerBlockLoad = 8;
      for (int64_t part = 0; part < factor; ++part) {
        for (int64_t chunk = 0; chunk < *chunksPerPart; ++chunk) {
          int64_t flatIndex = part * *chunksPerPart + chunk;
          auto vregType = dyn_cast<VRegType>(resultTypes[flatIndex]);
          if (!vregType)
            return rewriter.notifyMatchFailure(
                op, "block_deinterleaved group_load result must be vreg");
          FailureOr<Value> allMask =
              createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
          if (failed(allMask))
            return rewriter.notifyMatchFailure(
                op, "failed to create block_deinterleaved group_load mask");
          Value chunkOffset = createGroupChunkOffset(
              op.getLoc(), *offset, *rowStride, chunk * kGroupsPerBlockLoad,
              part * *blockElems, rewriter);
          Value chunkBase = makePtr(chunkOffset);
          results.push_back(rewriter
                                .create<VsldbOp>(op.getLoc(), vregType,
                                                 chunkBase, blockStride,
                                                 zeroI16, *allMask)
                                .getResult());
        }
      }

      replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
      return success();
    }

    if (resultLayout && resultLayout.isContiguous()) {
      FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
          resultVMIType, op.getNumGroupsAttr().getInt());
      if (failed(groupSize))
        return rewriter.notifyMatchFailure(
            op, "group_load requires num_groups to evenly divide lane count");
      std::optional<int64_t> constantRowStride =
          getConstantIndexValue(op.getRowStride());
      if (constantRowStride && *constantRowStride == *groupSize) {
        FailureOr<int64_t> lanesPerPart =
            getDataLanesPerPart(resultVMIType.getElementType());
        if (failed(lanesPerPart))
          return rewriter.notifyMatchFailure(
              op, "contiguous group_load requires known physical lanes");

        FailureOr<SmallVector<Type>> maybe_resultTypes =
            getConvertedResultTypes(op, 0, *this->getTypeConverter());
        if (failed(maybe_resultTypes))
          return failure();
        SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
        SmallVector<Value> results;
        results.reserve(resultTypes.size());
        for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
          auto vregType = dyn_cast<VRegType>(resultType);
          if (!vregType)
            return rewriter.notifyMatchFailure(
                op, "contiguous group_load result must be vreg");
          Value chunkOffset = createChunkOffset(
              op.getLoc(), *offset, static_cast<int64_t>(index) * *lanesPerPart,
              rewriter);
          results.push_back(rewriter
                                .create<VldsOp>(op.getLoc(), resultType,
                                                /*updated_base=*/Type{},
                                                *source, chunkOffset,
                                                /*dist=*/nullptr)
                                .getResult());
        }
        replaceOpWithFlatConvertedValues(rewriter, op, results,
                                         *this->getTypeConverter());
        return success();
      }
    }

    int64_t lanesPerPart = 0;
    int64_t groupCount = 0;
    int64_t chunksPerGroup = 0;
    FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
        resultVMIType, op.getNumGroupsAttr().getInt());
    if (failed(groupSize))
      return rewriter.notifyMatchFailure(
          op, "group_load requires num_groups to evenly divide lane count");
    if (failed(checkContiguousFullGroupChunks(op, resultVMIType, *groupSize,
                                              &lanesPerPart, &groupCount,
                                              &chunksPerGroup, rewriter)))
      return failure();

    FailureOr<SmallVector<Type>> maybe_resultTypes =

        getConvertedResultTypes(op, 0, *this->getTypeConverter());

    if (failed(maybe_resultTypes))

      return failure();

    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (static_cast<int64_t>(resultTypes.size()) != groupCount * chunksPerGroup)
      return rewriter.notifyMatchFailure(op, "group_load arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
      auto vregType = dyn_cast<VRegType>(resultType);
      if (!vregType)
        return rewriter.notifyMatchFailure(op,
                                           "group_load result must be vreg");
      int64_t group = index / chunksPerGroup;
      int64_t chunkInGroup = index % chunksPerGroup;
      Value chunkOffset =
          createGroupChunkOffset(op.getLoc(), *offset, *rowStride, group,
                                 chunkInGroup * lanesPerPart, rewriter);
      results.push_back(rewriter
                            .create<VldsOp>(op.getLoc(), resultType,
                                            /*updated_base=*/Type{}, *source,
                                            chunkOffset,
                                            /*dist=*/nullptr)
                            .getResult());
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

static LogicalResult lowerGroupSlotLoadParts(
    Operation *op, Value source, Value offset, Value sourceGroupStride,
    VMIVRegType resultVMIType, TypeRange resultTypes, int64_t numGroups,
    ConversionPatternRewriter &rewriter, SmallVectorImpl<Value> &results) {
  VMILayoutAttr layout = resultVMIType.getLayoutAttr();
  if (!layout || !layout.isGroupSlots() || layout.getSlots() <= 0)
    return rewriter.notifyMatchFailure(
        op, "group_slot_load requires explicit group_slots layout");
  if (!isa<PtrType>(source.getType()))
    return rewriter.notifyMatchFailure(
        op, "group_slot_load requires !pto.ptr source");

  int64_t slots = layout.getSlots();
  int64_t expectedArity = ceilDivNonNegative(numGroups, slots);
  if (static_cast<int64_t>(resultTypes.size()) != expectedArity)
    return rewriter.notifyMatchFailure(op, "group_slot_load arity mismatch");

  auto makeI16 = [&](int64_t value) -> Value {
    return rewriter.create<arith::ConstantIntOp>(op->getLoc(), value, 16);
  };
  Value zeroI16 = makeI16(0);
  auto makePtr = [&](Value elementOffset) -> Value {
    return rewriter
        .create<AddPtrOp>(op->getLoc(), source.getType(), source, elementOffset)
        .getResult();
  };

  results.reserve(results.size() + resultTypes.size());

  if (slots == 8) {
    std::optional<int64_t> stride = getConstantIndexValue(sourceGroupStride);
    if (!stride || *stride != 1)
      return rewriter.notifyMatchFailure(
          op, "slots=8 group_slot_load requires constant unit stride");
    for (auto [chunk, resultType] : llvm::enumerate(resultTypes)) {
      auto vregType = dyn_cast<VRegType>(resultType);
      if (!vregType)
        return rewriter.notifyMatchFailure(
            op, "group_slot_load result must be vreg");
      FailureOr<MaskType> maskType =
          getMaskTypeForVReg(vregType, rewriter.getContext());
      if (failed(maskType))
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for group_slot_load mask");
      int64_t groupBegin = static_cast<int64_t>(chunk) * slots;
      int64_t activeGroups = std::min<int64_t>(slots, numGroups - groupBegin);
      if (activeGroups <= 0)
        return rewriter.notifyMatchFailure(
            op, "slots=8 group_slot_load has no active groups for chunk");
      std::string pattern = (Twine("PAT_VL") + Twine(activeGroups)).str();
      FailureOr<Value> slotMask =
          createPrefixMask(op->getLoc(), *maskType, pattern, rewriter);
      if (failed(slotMask))
        return rewriter.notifyMatchFailure(
            op, "failed to create slots=8 group_slot_load mask");
      Value groupOffset =
          createChunkOffset(op->getLoc(), offset, groupBegin, rewriter);
      Value slotBase = makePtr(groupOffset);
      results.push_back(rewriter
                            .create<VsldbOp>(op->getLoc(), vregType, slotBase,
                                             zeroI16, zeroI16, *slotMask)
                            .getResult());
    }
    return success();
  }

  if (slots != 1)
    return rewriter.notifyMatchFailure(
        op, "group_slot_load supports only slots=8 or slots=1");
  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(resultVMIType.getElementType());
  if (elementBits == 0 || 256 % elementBits != 0)
    return rewriter.notifyMatchFailure(
        op, "slots=1 group_slot_load requires supported element width");
  int64_t alignedStrideElems = 256 / elementBits;
  std::optional<int64_t> constantStride =
      getConstantIndexValue(sourceGroupStride);
  if (!constantStride || *constantStride <= 0 ||
      *constantStride % alignedStrideElems != 0)
    return rewriter.notifyMatchFailure(
        op, Twine("slots=1 group_slot_load requires constant positive "
                  "source_group_stride divisible by ") +
                Twine(alignedStrideElems) +
                " elements for 32B lane-0 vsldb alignment");

  for (auto [group, resultType] : llvm::enumerate(resultTypes)) {
    auto vregType = dyn_cast<VRegType>(resultType);
    if (!vregType)
      return rewriter.notifyMatchFailure(op,
                                         "group_slot_load result must be vreg");
    FailureOr<MaskType> maskType =
        getMaskTypeForVReg(vregType, rewriter.getContext());
    if (failed(maskType))
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for group_slot_load mask");
    FailureOr<Value> oneBlockMask =
        createPrefixMask(op->getLoc(), *maskType, "PAT_VL1", rewriter);
    if (failed(oneBlockMask))
      return rewriter.notifyMatchFailure(
          op, "failed to create group_slot_load mask");
    Value groupOffset = offset;
    if (group != 0) {
      Value groupIndex =
          rewriter.create<arith::ConstantIndexOp>(op->getLoc(), group);
      Value rowOffset = rewriter
                            .create<arith::MulIOp>(
                                op->getLoc(), sourceGroupStride, groupIndex)
                            .getResult();
      groupOffset =
          rewriter.create<arith::AddIOp>(op->getLoc(), groupOffset, rowOffset)
              .getResult();
    }
    Value slotBase = makePtr(groupOffset);
    results.push_back(rewriter
                          .create<VsldbOp>(op->getLoc(), vregType, slotBase,
                                           zeroI16, zeroI16, *oneBlockMask)
                          .getResult());
  }
  return success();
}

static LogicalResult lowerGroupBroadcastParts(
    Operation *op, ValueRange sourceParts, VMIVRegType sourceVMIType,
    VMIVRegType resultVMIType, TypeRange resultTypes, int64_t numGroups,
    ConversionPatternRewriter &rewriter, SmallVectorImpl<Value> &results) {
  FailureOr<int64_t> groupSize =
      getGroupSizeFromNumGroups(resultVMIType, numGroups);
  if (failed(groupSize))
    return rewriter.notifyMatchFailure(
        op, "group_broadcast requires num_groups to evenly divide lane count");
  int64_t lanesPerPart = 0;
  int64_t groupCount = 0;
  if (failed(checkFullGroupSlotSourceShape(op, sourceVMIType, *groupSize,
                                           numGroups, &lanesPerPart,
                                           &groupCount, rewriter)))
    return failure();
  int64_t resultLayoutFactor = 0;
  int64_t resultGroupCount = 0;
  if (failed(checkFullGroupBroadcastResultShape(
          op, resultVMIType, *groupSize, lanesPerPart, &resultLayoutFactor,
          &resultGroupCount, rewriter)))
    return failure();
  if (resultGroupCount != groupCount)
    return rewriter.notifyMatchFailure(
        op, "group_broadcast requires matching source/result group slots");

  if (sourceParts.empty() || resultTypes.empty())
    return rewriter.notifyMatchFailure(op, "group_broadcast arity mismatch");

  auto firstSourceType = dyn_cast<VRegType>(sourceParts.front().getType());
  if (!firstSourceType)
    return rewriter.notifyMatchFailure(op,
                                       "group_broadcast source must be vreg");
  unsigned indexBits =
      pto::getPTOStorageElemBitWidth(firstSourceType.getElementType());
  if (indexBits != 8 && indexBits != 16 && indexBits != 32)
    return rewriter.notifyMatchFailure(
        op, "group_broadcast requires 8/16/32-bit index elements");
  auto indexElementType = IntegerType::get(rewriter.getContext(), indexBits);
  auto indexType =
      VRegType::get(rewriter.getContext(), firstSourceType.getElementCount(),
                    indexElementType);
  FailureOr<Value> allMask =
      createAllTrueMaskForVReg(op->getLoc(), firstSourceType, rewriter);
  if (failed(allMask))
    return rewriter.notifyMatchFailure(
        op, "failed to create group_broadcast all mask");
  VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
  VMILayoutAttr sourceLayout = sourceVMIType.getLayoutAttr();
  FailureOr<int64_t> resultBlockElems =
      getVMILayoutBlockElems(resultVMIType);
  if (failed(resultBlockElems))
    return rewriter.notifyMatchFailure(
        op, "group_broadcast requires a computable result block size");
  int64_t selectionGroupSize = *groupSize;
  if (resultLayoutFactor != 1 && resultLayout &&
      resultLayout.isBlockDeinterleaved() &&
      *groupSize < lanesPerPart)
    selectionGroupSize = *resultBlockElems;
  auto resolveLargeGroupSource = [&](int64_t group, int64_t chunksPerGroup,
                                     int64_t &sourceChunk,
                                     int64_t &baseGroupSlot) {
    int64_t slots = sourceLayout.getSlots();
    if (slots > 0) {
      sourceChunk = group / slots;
      baseGroupSlot = group % slots;
      return;
    }
    sourceChunk = group * chunksPerGroup;
    baseGroupSlot = 0;
  };

  results.clear();
  results.resize(resultTypes.size());
  for (auto [flatIndex, resultType] : llvm::enumerate(resultTypes)) {
    auto resultVRegType = dyn_cast<VRegType>(resultType);
    if (!resultVRegType || resultVRegType != firstSourceType)
      return rewriter.notifyMatchFailure(
          op, "group_broadcast requires uniform physical vreg types");
    int64_t sourceChunk = flatIndex;
    int64_t baseGroupSlot = 0;
    Value mappedGroupSlotIndex;
    if (resultLayoutFactor == 1) {
      bool laneStridedDense =
          resultLayout && resultLayout.isDense() &&
          resultLayout.getLaneStride() > 1;
      if (laneStridedDense) {
        VMILayoutAttr sourceLayout = sourceVMIType.getLayoutAttr();
        int64_t slots = sourceLayout.getSlots();
        if (slots <= 0) {
          if (sourceParts.empty() ||
              groupCount % static_cast<int64_t>(sourceParts.size()) != 0)
            return rewriter.notifyMatchFailure(
                op, "group_broadcast lane-stride source requires explicit "
                    "group_slots slots or derivable legacy slot count");
          slots = groupCount / sourceParts.size();
        }
        FailureOr<Value> index = createMappedGroupSlotIndexVector(
            op->getLoc(), resultVMIType, /*part=*/0, flatIndex, indexType,
            *groupSize, slots, sourceChunk, rewriter,
            sourceLayout.getLaneStride());
        if (failed(index))
          return rewriter.notifyMatchFailure(
              op, "failed to create group_broadcast lane-stride group-slot "
                  "index vector");
        mappedGroupSlotIndex = *index;
      } else if (*groupSize >= lanesPerPart) {
        int64_t chunksPerGroup = *groupSize / lanesPerPart;
        int64_t group = flatIndex / chunksPerGroup;
        resolveLargeGroupSource(group, chunksPerGroup, sourceChunk,
                                baseGroupSlot);
      } else {
        VMILayoutAttr sourceLayout = sourceVMIType.getLayoutAttr();
        int64_t slots = sourceLayout.getSlots();
        if (slots <= 0) {
          if (sourceParts.empty() ||
              groupCount % static_cast<int64_t>(sourceParts.size()) != 0)
            return rewriter.notifyMatchFailure(
                op, "group_broadcast small-group source requires explicit "
                    "group_slots slots or derivable legacy slot count");
          slots = groupCount / sourceParts.size();
        }
        int64_t groupsPerResultChunk = lanesPerPart / *groupSize;
        int64_t firstGroup = flatIndex * groupsPerResultChunk;
        sourceChunk = firstGroup / slots;
        baseGroupSlot = firstGroup % slots;
      }
    } else {
      bool blockFragmentSmallGroup =
          resultLayout && resultLayout.isBlockDeinterleaved() &&
          *groupSize < lanesPerPart;
      bool deinterleavedSmallGroup =
          resultLayout && resultLayout.isDeinterleaved() &&
          *groupSize < lanesPerPart;
      if (blockFragmentSmallGroup) {
        int64_t runningFlatIndex = 0;
        bool found = false;
        for (int64_t part = 0; part < resultLayoutFactor && !found; ++part) {
          FailureOr<int64_t> chunks = getDataChunksInPart(resultVMIType, part);
          if (failed(chunks))
            return rewriter.notifyMatchFailure(
                op, "group_broadcast failed to enumerate result chunks");
          for (int64_t chunk = 0; chunk < *chunks;
               ++chunk, ++runningFlatIndex) {
            if (runningFlatIndex != static_cast<int64_t>(flatIndex))
              continue;
            int64_t groupsPerResultChunk =
                lanesPerPart / *resultBlockElems;
            int64_t firstGroup = chunk * groupsPerResultChunk;
            int64_t slots = sourceLayout.getSlots();
            if (slots <= 0) {
              if (sourceParts.empty() ||
                  groupCount % static_cast<int64_t>(sourceParts.size()) != 0)
                return rewriter.notifyMatchFailure(
                    op,
                    "group_broadcast block-fragment source requires explicit "
                    "group_slots slots or derivable legacy slot count");
              slots = groupCount / sourceParts.size();
            }
            sourceChunk = firstGroup / slots;
            baseGroupSlot = firstGroup % slots;
            found = true;
            break;
          }
        }
        if (!found)
          return rewriter.notifyMatchFailure(
              op, "group_broadcast result chunk index is out of range");
      } else if (deinterleavedSmallGroup) {
        int64_t runningFlatIndex = 0;
        bool found = false;
        for (int64_t part = 0; part < resultLayoutFactor && !found; ++part) {
          FailureOr<int64_t> chunks = getDataChunksInPart(resultVMIType, part);
          if (failed(chunks))
            return rewriter.notifyMatchFailure(
                op, "group_broadcast failed to enumerate result chunks");
          for (int64_t chunk = 0; chunk < *chunks;
               ++chunk, ++runningFlatIndex) {
            if (runningFlatIndex != static_cast<int64_t>(flatIndex))
              continue;
            int64_t slots = sourceLayout.getSlots();
            if (slots <= 0) {
              if (sourceParts.empty() ||
                  groupCount % static_cast<int64_t>(sourceParts.size()) != 0)
                return rewriter.notifyMatchFailure(
                    op, "group_broadcast deinterleaved small-group source "
                        "requires explicit group_slots slots or derivable "
                        "legacy slot count");
              slots = groupCount / sourceParts.size();
            }
            FailureOr<Value> index = createMappedGroupSlotIndexVector(
                op->getLoc(), resultVMIType, part, chunk, indexType, *groupSize,
                slots, sourceChunk, rewriter, sourceLayout.getLaneStride());
            if (failed(index))
              return rewriter.notifyMatchFailure(
                  op,
                  "failed to create group_broadcast mapped group-slot index "
                  "vector");
            mappedGroupSlotIndex = *index;
            found = true;
            break;
          }
        }
        if (!found)
          return rewriter.notifyMatchFailure(
              op, "group_broadcast result chunk index is out of range");
      } else {
        int64_t runningFlatIndex = 0;
        bool found = false;
        for (int64_t part = 0; part < resultLayoutFactor && !found; ++part) {
          FailureOr<int64_t> chunks = getDataChunksInPart(resultVMIType, part);
          if (failed(chunks))
            return rewriter.notifyMatchFailure(
                op, "group_broadcast failed to enumerate result chunks");
          for (int64_t chunk = 0; chunk < *chunks;
               ++chunk, ++runningFlatIndex) {
            if (runningFlatIndex != static_cast<int64_t>(flatIndex))
              continue;
            FailureOr<int64_t> firstLogical =
                mapPhysicalLaneToLogical(resultVMIType, part, chunk, 0);
            FailureOr<int64_t> lastLogical = mapPhysicalLaneToLogical(
                resultVMIType, part, chunk, lanesPerPart - 1);
            if (failed(firstLogical) || failed(lastLogical))
              return rewriter.notifyMatchFailure(
                  op, "group_broadcast failed to map result chunk lanes");
            int64_t firstGroup = *firstLogical / *groupSize;
            int64_t lastGroup = *lastLogical / *groupSize;
            if (firstGroup != lastGroup)
              return rewriter.notifyMatchFailure(
                  op, "group_broadcast result chunk crosses logical groups");
            int64_t chunksPerGroup = *groupSize / lanesPerPart;
            resolveLargeGroupSource(firstGroup, chunksPerGroup, sourceChunk,
                                    baseGroupSlot);
            found = true;
            break;
          }
        }
        if (!found)
          return rewriter.notifyMatchFailure(
              op, "group_broadcast result chunk index is out of range");
      }
    }
    if (*groupSize >= lanesPerPart) {
      if (sourceChunk < 0 ||
          sourceChunk >= static_cast<int64_t>(sourceParts.size()))
        return rewriter.notifyMatchFailure(
            op, "group_broadcast source chunk is out of range");
      if (sourceLayout.getSlots() > 1) {
        FailureOr<Value> groupSlotIndex = createGroupSlotIndexVector(
            op->getLoc(), indexType, selectionGroupSize, baseGroupSlot,
            rewriter, sourceLayout.getLaneStride());
        if (failed(groupSlotIndex))
          return rewriter.notifyMatchFailure(
              op, "failed to create group_broadcast group-slot index vector");
        results[flatIndex] =
            rewriter
                .create<VselrOp>(op->getLoc(), resultType,
                                 sourceParts[sourceChunk], *groupSlotIndex)
                .getResult();
      } else {
        results[flatIndex] =
            rewriter
                .create<VdupOp>(op->getLoc(), resultType,
                                sourceParts[sourceChunk], *allMask,
                                rewriter.getStringAttr("LOWEST"))
                .getResult();
      }
    } else {
      bool blockFragmentSmallGroup =
          resultLayout && resultLayout.isBlockDeinterleaved();
      bool deinterleavedSmallGroup = resultLayout &&
                                     resultLayout.isDeinterleaved();
      if (resultLayoutFactor != 1 && !blockFragmentSmallGroup &&
          !deinterleavedSmallGroup)
        return rewriter.notifyMatchFailure(
            op, "group_broadcast small-group deinterleaved result is not "
                "supported");
      if (sourceChunk < 0 ||
          sourceChunk >= static_cast<int64_t>(sourceParts.size()))
        return rewriter.notifyMatchFailure(
            op, "group_broadcast source chunk is out of range");
      FailureOr<Value> groupSlotIndex =
          mappedGroupSlotIndex
              ? FailureOr<Value>(mappedGroupSlotIndex)
              : createGroupSlotIndexVector(op->getLoc(), indexType,
                                           selectionGroupSize, baseGroupSlot,
                                           rewriter,
                                           sourceLayout.getLaneStride());
      if (failed(groupSlotIndex))
        return rewriter.notifyMatchFailure(
            op, "failed to create group_broadcast group-slot index vector");
      results[flatIndex] =
          rewriter
              .create<VselrOp>(op->getLoc(), resultType,
                               sourceParts[sourceChunk], *groupSlotIndex)
              .getResult();
    }
  }
  return success();
}

struct OneToNVMIGroupSlotLoadOpPattern
    : OpConversionPattern<VMIGroupSlotLoadOp> {
  using OpConversionPattern<
      VMIGroupSlotLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIGroupSlotLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    VMILayoutAttr layout = resultVMIType.getLayoutAttr();
    if (!layout || !layout.isGroupSlots() || layout.getSlots() <= 0)
      return rewriter.notifyMatchFailure(
          op, "group_slot_load requires explicit group_slots layout");

    FailureOr<Value> source = getSingleValue(
        op, adaptor.getSource(),
        "group_slot_load source must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(),
        "group_slot_load offset must convert to one value", rewriter);
    FailureOr<Value> sourceGroupStride = getSingleValue(
        op, adaptor.getSourceGroupStride(),
        "group_slot_load source_group_stride must convert to one value",
        rewriter);
    if (failed(source) || failed(offset) || failed(sourceGroupStride))
      return failure();

    FailureOr<SmallVector<Type>> maybe_resultTypes =

        getConvertedResultTypes(op, 0, *this->getTypeConverter());

    if (failed(maybe_resultTypes))

      return failure();

    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    int64_t numGroups = op.getNumGroupsAttr().getInt();

    SmallVector<Value> results;
    if (failed(lowerGroupSlotLoadParts(op, *source, *offset, *sourceGroupStride,
                                       resultVMIType, resultTypes, numGroups,
                                       rewriter, results)))
      return failure();
    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIMaskedLoadOpPattern
    : OpConversionPattern<VMIMaskedLoadOp> {
  using OpConversionPattern<VMIMaskedLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIMaskedLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    FailureOr<Value> source = getSingleValue(
        op, adaptor.getSource(), "masked_load source must convert to one value",
        rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(), "masked_load offset must convert to one value",
        rewriter);
    if (failed(source) || failed(offset))
      return failure();

    FailureOr<int64_t> lanesPerPart = verifyFullOrSafeReadVRegChunks(
        op, resultVMIType, (*source).getType(), *offset, rewriter);
    if (failed(lanesPerPart))
      return failure();

    ValueRange maskParts = adaptor.getMask();
    ValueRange passthruParts = adaptor.getPassthru();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (maskParts.size() != passthruParts.size() ||
        passthruParts.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(op,
                                         "masked_load physical arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [index, maskPassthruAndType] : llvm::enumerate(
             llvm::zip_equal(maskParts, passthruParts, resultTypes))) {
      auto [mask, passthru, resultType] = maskPassthruAndType;
      if (!isa<MaskType>(mask.getType()) || passthru.getType() != resultType ||
          !isa<VRegType>(resultType))
        return rewriter.notifyMatchFailure(
            op, "masked_load physical part type mismatch");

      Value chunkOffset = createChunkOffset(op.getLoc(), *offset,
                                            index * *lanesPerPart, rewriter);
      Value loaded =
          rewriter
              .create<VldsOp>(op.getLoc(), resultType,
                              /*updated_base=*/Type{}, *source, chunkOffset,
                              /*dist=*/nullptr)
              .getResult();
      results.push_back(
          rewriter
              .create<VselOp>(op.getLoc(), resultType, loaded, passthru, mask)
              .getResult());
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIGatherOpPattern : OpConversionPattern<VMIGatherOp> {
  using OpConversionPattern<VMIGatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIGatherOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> source =
        getSingleValue(op, adaptor.getSource(),
                       "gather source must convert to one value", rewriter);
    if (failed(source))
      return failure();

    ValueRange indicesParts = adaptor.getIndices();
    ValueRange maskParts = adaptor.getMask();
    ValueRange passthruParts = adaptor.getPassthru();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (indicesParts.size() != maskParts.size() ||
        indicesParts.size() != passthruParts.size() ||
        indicesParts.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(op, "gather physical arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [indices, mask, passthru, resultType] :
         llvm::zip_equal(indicesParts, maskParts, passthruParts, resultTypes)) {
      if (!isa<VRegType>(indices.getType()) || !isa<MaskType>(mask.getType()) ||
          passthru.getType() != resultType || !isa<VRegType>(resultType))
        return rewriter.notifyMatchFailure(
            op, "gather physical part type mismatch");

      unsigned resultBits = pto::getPTOStorageElemBitWidth(
          cast<VRegType>(resultType).getElementType());
      Value gathered = resultBits == 16
                           ? rewriter
                                 .create<Vgather2Op>(op.getLoc(), resultType,
                                                     *source, indices, mask)
                                 .getResult()
                           : rewriter
                                 .create<Vgather2BcOp>(op.getLoc(), resultType,
                                                       *source, indices, mask)
                                 .getResult();
      results.push_back(
          rewriter
              .create<VselOp>(op.getLoc(), resultType, gathered, passthru, mask)
              .getResult());
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIExpandLoadOpPattern
    : OpConversionPattern<VMIExpandLoadOp> {
  using OpConversionPattern<VMIExpandLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIExpandLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    FailureOr<Value> source = getSingleValue(
        op, adaptor.getSource(), "expand_load source must convert to one value",
        rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(), "expand_load offset must convert to one value",
        rewriter);
    if (failed(source) || failed(offset))
      return failure();

    FailureOr<SmallVector<Type>> maybe_resultTypes =

        getConvertedResultTypes(op, 0, *this->getTypeConverter());

    if (failed(maybe_resultTypes))

      return failure();

    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (isStaticAllActiveMask(op.getMask(), resultVMIType.getElementCount())) {
      FailureOr<int64_t> lanesPerPart = verifyFullOrSafeReadVRegChunks(
          op, resultVMIType, (*source).getType(), *offset, rewriter);
      if (failed(lanesPerPart))
        return failure();

      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
        if (!isa<VRegType>(resultType))
          return rewriter.notifyMatchFailure(op,
                                             "expand_load result must be vreg");
        Value chunkOffset = createChunkOffset(op.getLoc(), *offset,
                                              index * *lanesPerPart, rewriter);
        results.push_back(rewriter
                              .create<VldsOp>(op.getLoc(), resultType,
                                              /*updated_base=*/Type{}, *source,
                                              chunkOffset,
                                              /*dist=*/nullptr)
                              .getResult());
      }

      replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
      return success();
    }

    ValueRange maskParts = adaptor.getMask();
    ValueRange passthruParts = adaptor.getPassthru();
    if (resultTypes.size() != 1 || maskParts.size() != 1 ||
        passthruParts.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "runtime expand_load supports only one physical chunk");

    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType ||
        passthruParts.front().getType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "runtime expand_load requires physical result/passthru/mask");

    auto baseType = dyn_cast<PtrType>((*source).getType());
    if (!baseType)
      return rewriter.notifyMatchFailure(op,
                                         "runtime expand_load requires ptr");
    Value gatherBase = rewriter
                           .create<AddPtrOp>(op.getLoc(), (*source).getType(),
                                             *source, *offset)
                           .getResult();
    auto indexType =
        VRegType::get(rewriter.getContext(), resultType.getElementCount(),
                      rewriter.getI32Type());
    FailureOr<Value> indexSeedMask =
        createAllTrueMaskForVReg(op.getLoc(), indexType, rewriter);
    if (failed(indexSeedMask))
      return rewriter.notifyMatchFailure(
          op, "failed to create runtime expand_load index seed mask");
    Value zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 32);
    Value carrier =
        rewriter
            .create<VdupOp>(op.getLoc(), indexType, zero, *indexSeedMask,
                            /*position=*/nullptr)
            .getResult();
    Value indices =
        rewriter
            .create<VusqzOp>(op.getLoc(), indexType, carrier, maskParts.front())
            .getResult();
    Value gathered =
        rewriter
            .create<Vgather2BcOp>(op.getLoc(), resultType, gatherBase, indices,
                                  maskParts.front())
            .getResult();
    Value result = rewriter
                       .create<VselOp>(op.getLoc(), resultType, gathered,
                                       passthruParts.front(), maskParts.front())
                       .getResult();
    replaceOpWithFlatConvertedValues(rewriter, op, SmallVector<Value>{result},
                                     *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIStoreOpPattern : OpConversionPattern<VMIStoreOp> {
  using OpConversionPattern<VMIStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIStoreOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto valueVMIType = cast<VMIVRegType>(op.getValue().getType());
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(valueVMIType.getElementType());
    if (failed(lanesPerPart))
      return rewriter.notifyMatchFailure(
          op, "store requires known physical lanes per part");
    bool fullPhysicalChunks =
        succeeded(checkFullDataPhysicalChunks(valueVMIType, nullptr));
    FailureOr<Value> destination =
        getSingleValue(op, adaptor.getDestination(),
                       "store destination must convert to one value", rewriter);
    FailureOr<Value> offset =
        getSingleValue(op, adaptor.getOffset(),
                       "store offset must convert to one value", rewriter);
    if (failed(destination) || failed(offset))
      return failure();

    ValueRange valueParts = adaptor.getValue();
    if (std::optional<std::string> dist =
            getDenseLaneStrideStoreDistToken(valueVMIType)) {
      std::optional<StringRef> maskGranularity =
          getDenseLaneStrideStoreMaskGranularity(valueVMIType);
      if (!maskGranularity)
        return rewriter.notifyMatchFailure(
            op, "unsupported lane_stride store mask granularity");
      int64_t semanticOffset = 0;
      for (auto [index, value] : llvm::enumerate(valueParts)) {
        auto vregType = dyn_cast<VRegType>(value.getType());
        if (!vregType)
          return rewriter.notifyMatchFailure(op, "store value must be vreg");
        FailureOr<int64_t> activeLanes =
            getActiveDataLanesInPhysicalChunk(valueVMIType, index);
        if (failed(activeLanes))
          return rewriter.notifyMatchFailure(
              op, "failed to compute lane_stride store active lanes");
        if (*activeLanes == 0)
          continue;
        auto maskType = MaskType::get(rewriter.getContext(), *maskGranularity);
        FailureOr<Value> mask = createPrefixMaskForActiveLanes(
            op.getLoc(), maskType, *activeLanes, rewriter);
        if (failed(mask))
          return rewriter.notifyMatchFailure(
              op, "failed to create lane_stride store mask");
        Value chunkOffset =
            createChunkOffset(op.getLoc(), *offset, semanticOffset, rewriter);
        rewriter.create<VstsOp>(op.getLoc(),
                                /*updated_base=*/Type{}, value, *destination,
                                chunkOffset, rewriter.getStringAttr(*dist),
                                *mask);
        semanticOffset += *activeLanes;
      }
      rewriter.eraseOp(op);
      return success();
    }

    VMILayoutAttr contiguousLayout =
        VMILayoutAttr::getContiguous(rewriter.getContext());
    FailureOr<SmallVector<Type>> maybeContiguousTypes =
        getConvertedVRegTypesWithLayout(valueVMIType, contiguousLayout,
                                        *this->getTypeConverter());
    if (failed(maybeContiguousTypes))
      return rewriter.notifyMatchFailure(
          op, "failed to compute contiguous store footprint");
    SmallVector<Type> contiguousTypes = std::move(*maybeContiguousTypes);
    SmallVector<Type> valuePartTypes;
    valuePartTypes.reserve(valueParts.size());
    for (Value value : valueParts)
      valuePartTypes.push_back(value.getType());
    FailureOr<bool> noWiderThanContiguous =
        hasNoWiderFootprintThanContiguous(valuePartTypes, contiguousTypes);
    if (failed(noWiderThanContiguous))
      return rewriter.notifyMatchFailure(
          op, "failed to compare store physical footprint");

    VMILayoutSupport localSupports;
    FailureOr<VMIStoreLayoutFact> storeFact =
        localSupports.getStoreLayoutFact(valueVMIType);
    if (succeeded(storeFact) && storeFact->valueLayout.isDeinterleaved() &&
        storeFact->valueLayout.getFactor() == 2 && fullPhysicalChunks &&
        *noWiderThanContiguous) {
      std::optional<std::string> dist =
          getX2MemoryDistToken(valueVMIType.getElementType(), "INTLV");
      if (dist && !valueParts.empty() && valueParts.size() % 2 == 0) {
        int64_t groups = valueParts.size() / 2;
        for (int64_t group = 0; group < groups; ++group) {
          Value low = valueParts[group];
          Value high = valueParts[groups + group];
          if (low.getType() != high.getType())
            return rewriter.notifyMatchFailure(
                op, "vstsx2 requires matching low/high value types");
          auto vregType = dyn_cast<VRegType>(low.getType());
          if (!vregType)
            return rewriter.notifyMatchFailure(op, "store value must be vreg");
          FailureOr<Value> mask =
              createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
          if (failed(mask))
            return rewriter.notifyMatchFailure(
                op, "unsupported element type for store mask");
          Value chunkOffset = createChunkOffset(
              op.getLoc(), *offset, group * 2 * *lanesPerPart, rewriter);
          rewriter.create<Vstsx2Op>(op.getLoc(), low, high, *destination,
                                    chunkOffset, rewriter.getStringAttr(*dist),
                                    *mask);
        }
        rewriter.eraseOp(op);
        return success();
      }
    }

    FailureOr<SmallVector<Value>> storeParts = materializeDataLayoutConversion(
        op, valueParts, contiguousTypes, valueVMIType.getLayoutAttr(),
        contiguousLayout, valueVMIType.getElementType(), rewriter);
    if (failed(storeParts))
      return failure();

    for (auto [index, value] : llvm::enumerate(*storeParts)) {
      auto vregType = dyn_cast<VRegType>(value.getType());
      if (!vregType)
        return rewriter.notifyMatchFailure(op, "store value must be vreg");
      if (!fullPhysicalChunks) {
        FailureOr<int64_t> activeLanes =
            getContiguousActiveDataLanes(valueVMIType, index);
        if (failed(activeLanes))
          return rewriter.notifyMatchFailure(
              op, "failed to compute store active lanes");
        if (*activeLanes == 0)
          continue;
      }
      FailureOr<Value> mask =
          fullPhysicalChunks
              ? createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter)
              : createContiguousStoreMask(op.getLoc(), valueVMIType, index,
                                          vregType, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for store mask");
      Value chunkOffset = createChunkOffset(op.getLoc(), *offset,
                                            index * *lanesPerPart, rewriter);
      rewriter.create<VstsOp>(op.getLoc(),
                              /*updated_base=*/Type{}, value, *destination,
                              chunkOffset, /*dist=*/nullptr, *mask);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct OneToNVMIInterleaveStoreOpPattern
    : OpConversionPattern<VMIInterleaveStoreOp> {
  using OpConversionPattern<
      VMIInterleaveStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIInterleaveStoreOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lowVMIType = cast<VMIVRegType>(op.getLow().getType());
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(lowVMIType.getElementType());
    if (failed(lanesPerPart))
      return rewriter.notifyMatchFailure(
          op, "interleave_store requires known physical lanes per part");

    std::optional<std::string> dist =
        getX2MemoryDistToken(lowVMIType.getElementType(), "INTLV");
    if (!dist)
      return rewriter.notifyMatchFailure(
          op, "interleave_store requires vstsx2 INTLV element support");

    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "interleave_store destination must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(),
        "interleave_store offset must convert to one value", rewriter);
    if (failed(destination) || failed(offset))
      return failure();

    ValueRange lowParts = adaptor.getLow();
    ValueRange highParts = adaptor.getHigh();
    if (lowParts.size() != highParts.size())
      return rewriter.notifyMatchFailure(
          op, "interleave_store requires matching low/high physical arity");

    for (size_t index = 0, e = lowParts.size(); index < e; ++index) {
      Value low = lowParts[index];
      Value high = highParts[index];
      if (low.getType() != high.getType())
        return rewriter.notifyMatchFailure(
            op, "interleave_store requires matching low/high physical types");
      auto vregType = dyn_cast<VRegType>(low.getType());
      if (!vregType)
        return rewriter.notifyMatchFailure(
            op, "interleave_store value must be vreg");
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for interleave_store mask");
      Value chunkOffset = createChunkOffset(
          op.getLoc(), *offset, static_cast<int64_t>(index) * 2 * *lanesPerPart,
          rewriter);
      rewriter.create<Vstsx2Op>(op.getLoc(), low, high, *destination,
                                chunkOffset, rewriter.getStringAttr(*dist),
                                *mask);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct OneToNVMIGroupStoreOpPattern
    : OpConversionPattern<VMIGroupStoreOp> {
  using OpConversionPattern<VMIGroupStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIGroupStoreOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto valueVMIType = cast<VMIVRegType>(op.getValue().getType());
    VMILayoutAttr layout = valueVMIType.getLayoutAttr();

    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "group_store destination must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(), "group_store offset must convert to one value",
        rewriter);
    FailureOr<Value> rowStride = getSingleValue(
        op, adaptor.getRowStride(),
        "group_store row_stride must convert to one value", rewriter);
    if (failed(destination) || failed(offset) || failed(rowStride))
      return failure();

    unsigned elementBits =
        pto::getPTOStorageElemBitWidth(valueVMIType.getElementType());
    bool compactSmallGroupStore = isCompactSmallGroupStore(
        layout, valueVMIType, op.getNumGroupsAttr().getInt(),
        getConstantIndexValue(op.getRowStride()));
    if (compactSmallGroupStore) {
      // The VMI input remains group_slots(num_groups=8, slots=8). Its active
      // group values occupy the leading physical lanes. Materialize the same
      // lane_stride=1 carrier as ensure_layout before selecting either an
      // aligned NORM store or unaligned 32-bit point stores.
      ValueRange valueParts = adaptor.getValue();
      if (valueParts.size() != 1)
        return rewriter.notifyMatchFailure(
            op, "compact small group_store requires one physical value part");
      auto valueType = dyn_cast<VRegType>(valueParts.front().getType());
      auto destinationType = dyn_cast<PtrType>((*destination).getType());
      if (!valueType || !destinationType)
        return rewriter.notifyMatchFailure(
            op, "compact small group_store requires vreg and ptr operands");

      Value compactValue = valueParts.front();
      if (layout.getLaneStride() != 1) {
        VMILayoutAttr compactLayout = VMILayoutAttr::getGroupSlots(
            rewriter.getContext(), layout.getNumGroups(), layout.getSlots());
        auto compactVMIType = VMIVRegType::get(
            rewriter.getContext(), valueVMIType.getElementCount(),
            valueVMIType.getElementType(), compactLayout);
        FailureOr<SmallVector<Value>> packed =
            materializeEnsureLayoutConversion(
                op, valueParts, valueVMIType, compactVMIType,
                *this->getTypeConverter(), rewriter);
        if (failed(packed) || packed->size() != 1)
          return rewriter.notifyMatchFailure(
              op, "failed to materialize compact group_store layout");
        compactValue = packed->front();
      }

      if (isKnown32ByteAlignedAddress(*destination, *offset,
                                      valueVMIType.getElementType())) {
        auto compactType = dyn_cast<VRegType>(compactValue.getType());
        std::optional<std::string> normalDist =
            getX2MemoryDistToken(valueVMIType.getElementType(), "NORM");
        if (!compactType || !normalDist)
          return rewriter.notifyMatchFailure(
              op, "aligned compact group_store requires a supported vreg "
                  "element type");
        FailureOr<MaskType> maskType =
            getMaskTypeForVReg(compactType, rewriter.getContext());
        if (failed(maskType))
          return rewriter.notifyMatchFailure(
              op, "failed to derive aligned compact group_store mask type");
        FailureOr<Value> storeMask = createPrefixMaskForActiveLanes(
            op.getLoc(), *maskType, valueVMIType.getElementCount(), rewriter);
        if (failed(storeMask))
          return rewriter.notifyMatchFailure(
              op, "failed to create aligned compact group_store mask");
        rewriter.create<VstsOp>(
            op.getLoc(), /*updated_base=*/Type{}, compactValue, *destination,
            *offset, rewriter.getStringAttr(*normalDist), *storeMask);
        rewriter.eraseOp(op);
        return success();
      }

      FailureOr<VRegType> wordType =
          getUnsignedCarrierVRegType(rewriter.getContext(), 32);
      if (failed(wordType))
        return rewriter.notifyMatchFailure(
            op, "failed to derive compact group-store word carrier type");
      FailureOr<Value> words =
          bitcastVReg(op.getLoc(), compactValue, *wordType, rewriter);
      FailureOr<Value> wordMask = createPrefixMask(
          op.getLoc(), MaskType::get(rewriter.getContext(), "b32"), "PAT_VL1",
          rewriter);
      if (failed(words) || failed(wordMask))
        return rewriter.notifyMatchFailure(
            op, "failed to materialize compact group-store words");

      Value elementBase =
          rewriter
              .create<AddPtrOp>(op.getLoc(), (*destination).getType(),
                                *destination, *offset)
              .getResult();
      auto wordPtrType =
          PtrType::get(rewriter.getContext(), wordType->getElementType(),
                       destinationType.getMemorySpace());
      Value wordBase =
          rewriter.create<CastPtrOp>(op.getLoc(), wordPtrType, elementBase)
              .getResult();
      FailureOr<Value> allWordMask =
          createAllTrueMaskForVReg(op.getLoc(), *wordType, rewriter);
      if (failed(allWordMask))
        return rewriter.notifyMatchFailure(
            op, "failed to materialize compact group-store word selector mask");
      auto indexType =
          VRegType::get(rewriter.getContext(), wordType->getElementCount(),
                        rewriter.getI32Type());

      int64_t wordCount = valueVMIType.getElementCount() *
                          static_cast<int64_t>(elementBits) / 32;
      for (int64_t wordIndex = 0; wordIndex < wordCount; ++wordIndex) {
        Value word = *words;
        if (wordIndex != 0) {
          FailureOr<Value> index = createScalarOffsetConstant(
              op.getLoc(), indexType.getElementType(), wordIndex, rewriter);
          if (failed(index))
            return rewriter.notifyMatchFailure(
                op, "failed to materialize compact group-store word index");
          Value indices =
              rewriter
                  .create<VdupOp>(op.getLoc(), indexType, *index, *allWordMask,
                                  /*position=*/nullptr)
                  .getResult();
          word = rewriter
                     .create<VselrOp>(op.getLoc(), *wordType, *words, indices)
                     .getResult();
        }
        Value wordOffset =
            rewriter.create<arith::ConstantIndexOp>(op.getLoc(), wordIndex);
        rewriter.create<VstsOp>(
            op.getLoc(), /*updated_base=*/Type{}, word, wordBase, wordOffset,
            rewriter.getStringAttr("1PT_B32"), *wordMask);
      }

      rewriter.eraseOp(op);
      return success();
    }

    if (layout && layout.isGroupSlots() && layout.getSlots() == 1 &&
        layout.getNumGroups() == op.getNumGroupsAttr().getInt()) {
      ValueRange valueParts = adaptor.getValue();
      if (static_cast<int64_t>(valueParts.size()) != layout.getNumGroups())
        return rewriter.notifyMatchFailure(
            op, "slots=1 group_store arity mismatch");
      unsigned elementBits =
          pto::getPTOStorageElemBitWidth(valueVMIType.getElementType());
      if (elementBits == 0 || 256 % elementBits != 0)
        return rewriter.notifyMatchFailure(
            op, "slots=1 group_store requires supported element width");
      std::optional<int64_t> constantRowStride =
          getConstantIndexValue(op.getRowStride());
      FailureOr<int64_t> lanesPerPart =
          getDataLanesPerPart(valueVMIType.getElementType());
      if (constantRowStride && *constantRowStride == 1 &&
          succeeded(lanesPerPart) && layout.getNumGroups() <= *lanesPerPart &&
          isKnown32ByteAlignedAddress(*destination, *offset,
                                      valueVMIType.getElementType())) {
        auto firstType = dyn_cast<VRegType>(valueParts.front().getType());
        if (!firstType)
          return rewriter.notifyMatchFailure(op,
                                             "group_store value must be vreg");
        FailureOr<MaskType> maskType =
            getMaskTypeForVReg(firstType, rewriter.getContext());
        FailureOr<Value> allMask =
            createAllTrueMaskForVReg(op.getLoc(), firstType, rewriter);
        if (failed(maskType) || failed(allMask))
          return rewriter.notifyMatchFailure(
              op, "unsupported element type for packed group_store mask");

        Value packed =
            rewriter
                .create<VdupOp>(op.getLoc(), firstType, valueParts.front(),
                                *allMask, rewriter.getStringAttr("LOWEST"))
                .getResult();
        for (int64_t group = 1; group < layout.getNumGroups(); ++group) {
          auto vregType = dyn_cast<VRegType>(valueParts[group].getType());
          if (!vregType || vregType != firstType)
            return rewriter.notifyMatchFailure(
                op, "packed group_store requires uniform vreg parts");
          Value splat =
              rewriter
                  .create<VdupOp>(op.getLoc(), firstType, valueParts[group],
                                  *allMask, rewriter.getStringAttr("LOWEST"))
                  .getResult();
          FailureOr<Value> laneMask = createLaneRangeMask(
              op.getLoc(), *maskType, group, group + 1, rewriter);
          if (failed(laneMask))
            return rewriter.notifyMatchFailure(
                op, "failed to create packed group_store lane mask");
          packed = rewriter
                       .create<VselOp>(op.getLoc(), firstType, splat, packed,
                                       *laneMask)
                       .getResult();
        }

        FailureOr<Value> storeMask = createPrefixMaskForActiveLanes(
            op.getLoc(), *maskType, layout.getNumGroups(), rewriter);
        if (failed(storeMask))
          return rewriter.notifyMatchFailure(
              op, "failed to create packed group_store store mask");
        rewriter.create<VstsOp>(op.getLoc(),
                                /*updated_base=*/Type{}, packed, *destination,
                                *offset, /*dist=*/nullptr, *storeMask);
        rewriter.eraseOp(op);
        return success();
      }
      if (constantRowStride && *constantRowStride <= 0)
        return rewriter.notifyMatchFailure(
            op, "slots=1 group_store requires positive row_stride when "
                "row_stride is constant");
      std::optional<std::string> pointDist =
          getPointStoreDistToken(valueVMIType.getElementType());
      if (!pointDist)
        return rewriter.notifyMatchFailure(
            op, "slots=1 group_store requires 1PT_B8/B16/B32 store support");

      for (auto [group, value] : llvm::enumerate(valueParts)) {
        auto vregType = dyn_cast<VRegType>(value.getType());
        if (!vregType)
          return rewriter.notifyMatchFailure(op,
                                             "group_store value must be vreg");
        FailureOr<MaskType> maskType =
            getMaskTypeForVReg(vregType, rewriter.getContext());
        if (failed(maskType))
          return rewriter.notifyMatchFailure(
              op, "unsupported element type for group_store mask");
        FailureOr<Value> mask =
            createPrefixMask(op.getLoc(), *maskType, "PAT_VL1", rewriter);
        if (failed(mask))
          return rewriter.notifyMatchFailure(
              op, "failed to create slots=1 group_store mask");
        Value groupOffset =
            createGroupChunkOffset(op.getLoc(), *offset, *rowStride, group,
                                   /*chunkLaneOffset=*/0, rewriter);
        rewriter.create<VstsOp>(op.getLoc(),
                                /*updated_base=*/Type{}, value, *destination,
                                groupOffset, rewriter.getStringAttr(*pointDist),
                                *mask);
      }

      rewriter.eraseOp(op);
      return success();
    }

    if (layout && layout.isGroupSlots() && layout.getSlots() == 8 &&
        layout.getNumGroups() == op.getNumGroupsAttr().getInt()) {
      int64_t numGroups = layout.getNumGroups();
      std::optional<int64_t> constantRowStride =
          getConstantIndexValue(op.getRowStride());
      if (!constantRowStride || *constantRowStride != 1)
        return rewriter.notifyMatchFailure(
            op, "slots=8 group_store requires constant unit row_stride");

      ValueRange valueParts = adaptor.getValue();
      if (static_cast<int64_t>(valueParts.size()) !=
          ceilDivNonNegative(numGroups, 8))
        return rewriter.notifyMatchFailure(
            op, "slots=8 group_store arity mismatch");

      if (!valueParts.empty()) {
        auto firstVRegType = dyn_cast<VRegType>(valueParts.front().getType());
        if (!firstVRegType)
          return rewriter.notifyMatchFailure(op,
                                             "group_store value must be vreg");
        bool packedByteStore = isPackedByteGroupStore(
            op.getDestination().getType(), firstVRegType);
        if (packedByteStore) {
          bool laneStridedPackedByteStore = layout.hasLaneStride();
          for (Value value : valueParts) {
            auto vregType = dyn_cast<VRegType>(value.getType());
            if (!vregType || vregType != firstVRegType)
              return rewriter.notifyMatchFailure(
                  op, "packed slots=8 group_store requires uniform vreg parts");
          }

          FailureOr<MaskType> maskType =
              getMaskTypeForVReg(firstVRegType, rewriter.getContext());
          if (failed(maskType))
            return rewriter.notifyMatchFailure(
                op, "unsupported element type for packed group_store mask");
          if (!laneStridedPackedByteStore && numGroups == 8 &&
              valueParts.size() == 1 &&
              isKnown32ByteAlignedAddress(*destination, *offset,
                                          valueVMIType.getElementType())) {
            MLIRContext *ctx = rewriter.getContext();
            auto ui16 = IntegerType::get(
                ctx, 16, IntegerType::SignednessSemantics::Unsigned);
            auto ui8 = IntegerType::get(
                ctx, 8, IntegerType::SignednessSemantics::Unsigned);
            auto packed16Type = VRegType::get(ctx, 128, ui16);
            auto packed8Type = VRegType::get(ctx, 256, ui8);
            Value packed16 =
                rewriter
                    .create<VpackOp>(op.getLoc(), packed16Type,
                                     valueParts.front(),
                                     rewriter.getStringAttr("LOWER"))
                    .getResult();
            Value packed8 =
                rewriter
                    .create<VpackOp>(op.getLoc(), packed8Type, packed16,
                                     rewriter.getStringAttr("LOWER"))
                    .getResult();
            FailureOr<MaskType> packedMaskType =
                getMaskTypeForVReg(packed8Type, ctx);
            if (failed(packedMaskType))
              return rewriter.notifyMatchFailure(
                  op, "failed to create packed byte group_store mask type");
            FailureOr<Value> storeMask = createPrefixMaskForActiveLanes(
                op.getLoc(), *packedMaskType, numGroups, rewriter);
            if (failed(storeMask))
              return rewriter.notifyMatchFailure(
                  op, "failed to create packed byte group_store mask");
            rewriter.create<VstsOp>(
                op.getLoc(), /*updated_base=*/Type{}, packed8, *destination,
                *offset, rewriter.getStringAttr("NORM_B8"), *storeMask);
            rewriter.eraseOp(op);
            return success();
          }

          auto indexElementType = IntegerType::get(
              rewriter.getContext(),
              pto::getPTOStorageElemBitWidth(firstVRegType.getElementType()));
          auto indexType =
              VRegType::get(rewriter.getContext(),
                            firstVRegType.getElementCount(), indexElementType);
          FailureOr<Value> slotIndex = createGroupSlotIndexVector(
              op.getLoc(), indexType, /*groupSize=*/8, /*baseGroupSlot=*/0,
              rewriter);
          FailureOr<Value> allMask =
              createAllTrueMaskForVReg(op.getLoc(), firstVRegType, rewriter);
          if (failed(slotIndex) || failed(allMask))
            return rewriter.notifyMatchFailure(
                op, "failed to create packed group_store lane selector");

          for (int64_t blockStart = 0; blockStart < numGroups;
               blockStart += 32) {
            FailureOr<Value> zero =
                createZeroVector(op.getLoc(), firstVRegType, rewriter);
            if (failed(zero))
              return rewriter.notifyMatchFailure(
                  op, "failed to create packed group_store accumulator");
            Value merged = *zero;
            for (int64_t localPart = 0; localPart < 4; ++localPart) {
              int64_t partIndex = blockStart / 8 + localPart;
              if (partIndex >= static_cast<int64_t>(valueParts.size()))
                break;
              int64_t remainingGroups = numGroups - partIndex * 8;
              int64_t activeGroups = std::min<int64_t>(8, remainingGroups);
              if (activeGroups <= 0)
                break;
              Value selected =
                  rewriter
                      .create<VselrOp>(op.getLoc(), firstVRegType,
                                       valueParts[partIndex], *slotIndex)
                      .getResult();
              FailureOr<Value> laneMask =
                  createLaneRangeMask(op.getLoc(), *maskType, localPart * 8,
                                      localPart * 8 + activeGroups, rewriter);
              if (failed(laneMask))
                return rewriter.notifyMatchFailure(
                    op, "failed to create packed group_store lane mask");
              merged = rewriter
                           .create<VselOp>(op.getLoc(), firstVRegType, selected,
                                           merged, *laneMask)
                           .getResult();
            }

            int64_t activeGroups =
                std::min<int64_t>(32, numGroups - blockStart);
            FailureOr<Value> storeMask = createPrefixMaskForActiveLanes(
                op.getLoc(), *maskType, activeGroups, rewriter);
            if (failed(storeMask))
              return rewriter.notifyMatchFailure(
                  op, "failed to create packed group_store store mask");
            Value groupOffset = createGroupChunkOffset(
                op.getLoc(), *offset, *rowStride, blockStart / 4,
                /*chunkLaneOffset=*/0, rewriter);
            rewriter.create<VstsOp>(
                op.getLoc(), /*updated_base=*/Type{}, merged, *destination,
                groupOffset, rewriter.getStringAttr("PK4_B32"), *storeMask);
          }

          rewriter.eraseOp(op);
          return success();
        }
      }

      if (layout.hasLaneStride()) {
        std::optional<std::string> dist = getLaneStrideStoreDistToken(
            layout, valueVMIType.getElementType());
        std::optional<StringRef> maskGranularity =
            getLaneStrideStoreMaskGranularity(
                layout, valueVMIType.getElementType());
        if (!dist || !maskGranularity)
          return rewriter.notifyMatchFailure(
              op, "unsupported slots=8 lane_stride group_store packing");

        auto maskType =
            MaskType::get(rewriter.getContext(), *maskGranularity);
        for (auto [slotBlock, value] : llvm::enumerate(valueParts)) {
          if (!isa<VRegType>(value.getType()))
            return rewriter.notifyMatchFailure(
                op, "group_store value must be vreg");
          int64_t activeGroups =
              std::min<int64_t>(8, numGroups - slotBlock * 8);
          FailureOr<Value> mask = createPrefixMaskForActiveLanes(
              op.getLoc(), maskType, activeGroups, rewriter);
          if (failed(mask))
            return rewriter.notifyMatchFailure(
                op, "failed to create packed slots=8 group_store mask");
          Value groupOffset = createGroupChunkOffset(
              op.getLoc(), *offset, *rowStride, slotBlock * 8,
              /*chunkLaneOffset=*/0, rewriter);
          rewriter.create<VstsOp>(
              op.getLoc(), /*updated_base=*/Type{}, value, *destination,
              groupOffset, rewriter.getStringAttr(*dist), *mask);
        }

        rewriter.eraseOp(op);
        return success();
      }

      for (auto [slotBlock, value] : llvm::enumerate(valueParts)) {
        auto vregType = dyn_cast<VRegType>(value.getType());
        if (!vregType)
          return rewriter.notifyMatchFailure(op,
                                             "group_store value must be vreg");
        FailureOr<MaskType> maskType =
            getMaskTypeForVReg(vregType, rewriter.getContext());
        if (failed(maskType))
          return rewriter.notifyMatchFailure(
              op, "unsupported element type for group_store mask");
        int64_t activeGroups = std::min<int64_t>(8, numGroups - slotBlock * 8);
        FailureOr<Value> mask = createPrefixMaskForActiveLanes(
            op.getLoc(), *maskType, activeGroups, rewriter);
        if (failed(mask))
          return rewriter.notifyMatchFailure(
              op, "failed to create slots=8 group_store mask");
        Value groupOffset = createGroupChunkOffset(
            op.getLoc(), *offset, *rowStride, slotBlock * 8,
            /*chunkLaneOffset=*/0, rewriter);
        rewriter.create<VstsOp>(op.getLoc(),
                                /*updated_base=*/Type{}, value, *destination,
                                groupOffset, /*dist=*/nullptr, *mask);
      }

      rewriter.eraseOp(op);
      return success();
    }

    int64_t lanesPerPart = 0;
    int64_t groupCount = 0;
    int64_t chunksPerGroup = 0;
    FailureOr<int64_t> groupSize =
        getGroupSizeFromNumGroups(valueVMIType, op.getNumGroupsAttr().getInt());
    if (failed(groupSize))
      return rewriter.notifyMatchFailure(
          op, "group_store requires num_groups to evenly divide lane count");

    int64_t d2LanesPerPart = 0;
    int64_t d2GroupCount = 0;
    int64_t d2ChunksPerGroupPerPart = 0;
    std::string d2Reason;
    if (succeeded(checkDeinterleaved2GroupStoreChunkShape(
            valueVMIType, *groupSize, &d2LanesPerPart, &d2GroupCount,
            &d2ChunksPerGroupPerPart, &d2Reason))) {
      std::optional<std::string> dist =
          getX2MemoryDistToken(valueVMIType.getElementType(), "INTLV");
      if (!dist)
        return rewriter.notifyMatchFailure(
            op, "group_store requires vstsx2 INTLV element support");

      ValueRange valueParts = adaptor.getValue();
      int64_t chunksPerPart = d2GroupCount * d2ChunksPerGroupPerPart;
      if (static_cast<int64_t>(valueParts.size()) != 2 * chunksPerPart)
        return rewriter.notifyMatchFailure(
            op, "deinterleaved=2 group_store arity mismatch");

      for (int64_t group = 0; group < d2GroupCount; ++group) {
        for (int64_t chunk = 0; chunk < d2ChunksPerGroupPerPart; ++chunk) {
          int64_t lowIndex = group * d2ChunksPerGroupPerPart + chunk;
          int64_t highIndex = chunksPerPart + lowIndex;
          Value low = valueParts[lowIndex];
          Value high = valueParts[highIndex];
          if (low.getType() != high.getType())
            return rewriter.notifyMatchFailure(
                op, "vstsx2 group_store requires matching low/high types");
          auto vregType = dyn_cast<VRegType>(low.getType());
          if (!vregType)
            return rewriter.notifyMatchFailure(op,
                                               "group_store value must be vreg");
          FailureOr<Value> mask =
              createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
          if (failed(mask))
            return rewriter.notifyMatchFailure(
                op, "unsupported element type for group_store mask");
          Value chunkOffset = createGroupChunkOffset(
              op.getLoc(), *offset, *rowStride, group,
              chunk * 2 * d2LanesPerPart, rewriter);
          rewriter.create<Vstsx2Op>(op.getLoc(), low, high, *destination,
                                    chunkOffset, rewriter.getStringAttr(*dist),
                                    *mask);
        }
      }

      rewriter.eraseOp(op);
      return success();
    }

    if (failed(checkContiguousFullGroupChunks(op, valueVMIType, *groupSize,
                                              &lanesPerPart, &groupCount,
                                              &chunksPerGroup, rewriter)))
      return failure();

    ValueRange valueParts = adaptor.getValue();
    if (static_cast<int64_t>(valueParts.size()) != groupCount * chunksPerGroup)
      return rewriter.notifyMatchFailure(op, "group_store arity mismatch");

    for (auto [index, value] : llvm::enumerate(valueParts)) {
      auto vregType = dyn_cast<VRegType>(value.getType());
      if (!vregType)
        return rewriter.notifyMatchFailure(op,
                                           "group_store value must be vreg");
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for group_store mask");
      int64_t group = index / chunksPerGroup;
      int64_t chunkInGroup = index % chunksPerGroup;
      Value chunkOffset =
          createGroupChunkOffset(op.getLoc(), *offset, *rowStride, group,
                                 chunkInGroup * lanesPerPart, rewriter);
      rewriter.create<VstsOp>(op.getLoc(),
                              /*updated_base=*/Type{}, value, *destination,
                              chunkOffset, /*dist=*/nullptr, *mask);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct OneToNVMIMaskedStoreOpPattern
    : OpConversionPattern<VMIMaskedStoreOp> {
  using OpConversionPattern<VMIMaskedStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIMaskedStoreOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto valueVMIType = cast<VMIVRegType>(op.getValue().getType());
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(valueVMIType.getElementType());
    if (failed(lanesPerPart))
      return rewriter.notifyMatchFailure(
          op, "masked_store requires known physical lanes per part");

    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "masked_store destination must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(),
        "masked_store offset must convert to one value", rewriter);
    if (failed(destination) || failed(offset))
      return failure();

    ValueRange valueParts = adaptor.getValue();
    ValueRange maskParts = adaptor.getMask();
    if (valueParts.size() != maskParts.size())
      return rewriter.notifyMatchFailure(
          op, "masked_store value/mask physical arity mismatch");

    auto maskVMIType = cast<VMIMaskType>(op.getMask().getType());
    if (std::optional<std::string> dist =
            getDenseLaneStrideStoreDistToken(valueVMIType)) {
      std::optional<StringRef> maskGranularity =
          getDenseLaneStrideMaskedStoreMaskGranularity(valueVMIType);
      VMILayoutAttr valueLayout = valueVMIType.getLayoutAttr();
      VMILayoutAttr maskLayout = maskVMIType.getLayoutAttr();
      if (maskGranularity && valueLayout && maskLayout &&
          valueLayout == maskLayout) {
        int64_t semanticOffset = 0;
        for (auto [index, valueAndMask] :
             llvm::enumerate(llvm::zip_equal(valueParts, maskParts))) {
          auto [value, mask] = valueAndMask;
          auto vregType = dyn_cast<VRegType>(value.getType());
          if (!vregType || !isa<MaskType>(mask.getType()))
            return rewriter.notifyMatchFailure(
                op, "lane_stride masked_store parts must be vreg/mask");
          FailureOr<int64_t> activeLanes =
              getActiveDataLanesInPhysicalChunk(valueVMIType, index);
          if (failed(activeLanes))
            return rewriter.notifyMatchFailure(
                op, "failed to compute lane_stride masked_store active lanes");
          if (*activeLanes == 0)
            continue;
          FailureOr<Value> storeMask = createDenseLaneStrideStorePredicate(
              op.getLoc(), valueVMIType, index, mask, *maskGranularity,
              rewriter);
          if (failed(storeMask))
            return rewriter.notifyMatchFailure(
                op, "failed to compact lane_stride masked_store predicate");
          Value chunkOffset =
              createChunkOffset(op.getLoc(), *offset, semanticOffset, rewriter);
          rewriter.create<VstsOp>(op.getLoc(),
                                  /*updated_base=*/Type{}, value, *destination,
                                  chunkOffset, rewriter.getStringAttr(*dist),
                                  *storeMask);
          semanticOffset += *activeLanes;
        }

        rewriter.eraseOp(op);
        return success();
      }
    }

    SmallVector<Type> contiguousValueTypes;
    contiguousValueTypes.reserve(valueParts.size());
    for (Value value : valueParts)
      contiguousValueTypes.push_back(value.getType());
    FailureOr<SmallVector<Value>> storeParts = materializeDataLayoutConversion(
        op, valueParts, contiguousValueTypes, valueVMIType.getLayoutAttr(),
        VMILayoutAttr::getContiguous(rewriter.getContext()),
        valueVMIType.getElementType(), rewriter);
    if (failed(storeParts))
      return failure();

    SmallVector<Type> contiguousMaskTypes;
    contiguousMaskTypes.reserve(maskParts.size());
    for (Value mask : maskParts)
      contiguousMaskTypes.push_back(mask.getType());
    FailureOr<SmallVector<Value>> storeMasks = materializeMaskLayoutConversion(
        op, maskParts, contiguousMaskTypes, maskVMIType.getLayoutAttr(),
        VMILayoutAttr::getContiguous(rewriter.getContext()), rewriter);
    if (failed(storeMasks))
      return failure();

    if (storeParts->size() != storeMasks->size())
      return rewriter.notifyMatchFailure(
          op, "masked_store converted value/mask arity mismatch");

    for (auto [index, valueAndMask] :
         llvm::enumerate(llvm::zip_equal(*storeParts, *storeMasks))) {
      auto [value, mask] = valueAndMask;
      auto vregType = dyn_cast<VRegType>(value.getType());
      if (!vregType || !isa<MaskType>(mask.getType()))
        return rewriter.notifyMatchFailure(
            op, "masked_store converted parts must be vreg/mask");
      FailureOr<int64_t> activeLanes =
          getContiguousActiveDataLanes(valueVMIType, index);
      if (failed(activeLanes))
        return rewriter.notifyMatchFailure(
            op, "failed to compute masked_store active lanes");
      if (*activeLanes == 0)
        continue;
      FailureOr<Value> storeMask = createMaskedStorePredicate(
          op.getLoc(), valueVMIType, index, mask, vregType, rewriter);
      if (failed(storeMask))
        return rewriter.notifyMatchFailure(
            op, "failed to materialize masked_store predicate");
      Value chunkOffset = createChunkOffset(op.getLoc(), *offset,
                                            index * *lanesPerPart, rewriter);
      rewriter.create<VstsOp>(op.getLoc(),
                              /*updated_base=*/Type{}, value, *destination,
                              chunkOffset, /*dist=*/nullptr, *storeMask);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct OneToNVMIGroupBroadcastLoadOpPattern
    : OpConversionPattern<VMIGroupBroadcastLoadOp> {
  using OpConversionPattern<VMIGroupBroadcastLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIGroupBroadcastLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    int64_t numGroups = op.getNumGroupsAttr().getInt();
    FailureOr<Value> source = getSingleValue(
        op, adaptor.getSource(),
        "group_broadcast_load source must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(),
        "group_broadcast_load offset must convert to one value", rewriter);
    FailureOr<Value> sourceGroupStride = getSingleValue(
        op, adaptor.getSourceGroupStride(),
        "group_broadcast_load source_group_stride must convert to one value",
        rewriter);
    if (failed(source) || failed(offset) || failed(sourceGroupStride))
      return failure();

    VMILayoutSupport supports;
    std::string supportReason;
    FailureOr<VMIGroupBroadcastLoadLayoutFact> loadFact =
        supports.getGroupBroadcastLoadLayoutFact(op, &supportReason);
    if (failed(loadFact))
      return rewriter.notifyMatchFailure(
          op, Twine("group_broadcast_load has no registered support: ") +
                  supportReason);

    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();

    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    FailureOr<VMIGroupBroadcastLoadDirectFact> directFact =
        supports.getGroupBroadcastLoadDirectFact(op);
    auto getBRCDist = [&]() -> std::optional<StringRef> {
      unsigned elementBits =
          pto::getPTOStorageElemBitWidth(resultVMIType.getElementType());
      if (elementBits == 8)
        return StringRef("BRC_B8");
      if (elementBits == 16)
        return StringRef("BRC_B16");
      if (elementBits == 32)
        return StringRef("BRC_B32");
      return std::nullopt;
    };

    if (succeeded(directFact) &&
        directFact->kind == VMIGroupBroadcastLoadDirectKind::BRC) {
      int64_t chunksPerGroup =
          directFact->layout.groupSize / directFact->layout.lanesPerPart;
      std::optional<StringRef> brcDist = getBRCDist();
      if (!brcDist)
        return rewriter.notifyMatchFailure(
            op, "group_broadcast_load BRC lowering requires b8/b16/b32 "
                "element type");
      if (!isa<PtrType>((*source).getType()))
        return rewriter.notifyMatchFailure(
            op, "group_broadcast_load BRC lowering requires !pto.ptr source");
      if (chunksPerGroup <= 0 ||
          static_cast<int64_t>(resultTypes.size()) !=
              numGroups * chunksPerGroup)
        return rewriter.notifyMatchFailure(
            op, "group_broadcast_load BRC physical arity mismatch");

      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
        auto vregType = dyn_cast<VRegType>(resultType);
        if (!vregType)
          return rewriter.notifyMatchFailure(
              op, "group_broadcast_load BRC result must be vreg");
        int64_t group = static_cast<int64_t>(index) / chunksPerGroup;
        Value groupOffset =
            createGroupChunkOffset(op.getLoc(), *offset, *sourceGroupStride,
                                   group, /*inGroupLaneOffset=*/0, rewriter);
        results.push_back(rewriter
                              .create<VldsOp>(op.getLoc(), resultType,
                                              /*updated_base=*/Type{}, *source,
                                              groupOffset,
                                              rewriter.getStringAttr(*brcDist))
                              .getResult());
      }
      replaceOpWithFlatConvertedValues(rewriter, op, results,
                                       *this->getTypeConverter());
      return success();
    }

    if (failed(directFact) ||
        directFact->kind != VMIGroupBroadcastLoadDirectKind::E2B) {
      std::optional<int64_t> stride =
          getConstantIndexValue(op.getSourceGroupStride());
      int64_t slots = (stride && *stride == 1) ? 8 : 1;
      auto sourceVMIType = VMIVRegType::get(
          rewriter.getContext(), numGroups, resultVMIType.getElementType(),
          VMILayoutAttr::getGroupSlots(rewriter.getContext(), numGroups,
                                       slots));

      FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceVMIType);
      FailureOr<Type> sourceElementType =
          getVMIVRegPhysicalElementType(sourceVMIType);
      if (failed(sourceArity) || failed(sourceElementType))
        return rewriter.notifyMatchFailure(
            op, "group_broadcast_load fallback cannot derive physical types");

      SmallVector<Type> sourceTypes;
      sourceTypes.reserve(*sourceArity);
      FailureOr<int64_t> sourceLanesPerPart =
          getDataLanesPerPart(*sourceElementType);
      if (failed(sourceLanesPerPart))
        return rewriter.notifyMatchFailure(
            op, "group_broadcast_load fallback cannot derive source lanes");
      for (int64_t i = 0; i < *sourceArity; ++i)
        sourceTypes.push_back(VRegType::get(
            rewriter.getContext(), *sourceLanesPerPart, *sourceElementType));

      SmallVector<Value> sourceParts;
      if (failed(lowerGroupSlotLoadParts(
              op, *source, *offset, *sourceGroupStride, sourceVMIType,
              sourceTypes, numGroups, rewriter, sourceParts)))
        return failure();

      SmallVector<Value> results;
      if (failed(lowerGroupBroadcastParts(op, sourceParts, sourceVMIType,
                                          resultVMIType, resultTypes, numGroups,
                                          rewriter, results)))
        return failure();
      replaceOpWithFlatConvertedValues(rewriter, op, results,
                                       *this->getTypeConverter());
      return success();
    }

    VMILayoutAttr layout = resultVMIType.getLayoutAttr();
    bool contiguousPacketLayout = layout && layout.isContiguous();
    bool splitPacketLayout = layout && layout.isDeinterleaved() &&
                             (layout.getFactor() == 2 ||
                              layout.getFactor() == 4) &&
                             layout.getLaneStride() == 1;
    if (!contiguousPacketLayout && !splitPacketLayout)
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load E2B lowering requires "
              "contiguous result layout for direct group size or "
              "deinterleaved=2/4 result layout for split "
              "group size");

    unsigned elementBits = directFact->layout.elementBits;
    if (elementBits != 16 && elementBits != 32)
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load E2B lowering requires b16 or b32 "
              "element type");
    StringRef e2bDist = elementBits == 16 ? "E2B_B16" : "E2B_B32";

    std::optional<int64_t> stride =
        getConstantIndexValue(op.getSourceGroupStride());
    if (!stride || *stride != 1)
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load E2B lowering requires constant unit "
              "source_group_stride");

    if (!isa<PtrType>((*source).getType()))
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load E2B lowering requires !pto.ptr source");

    if (numGroups != 8)
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load E2B lowering requires num_groups = 8");

    FailureOr<int64_t> chunksPerPart = getDataChunksInPart(resultVMIType, 0);
    if (failed(chunksPerPart) || *chunksPerPart <= 0)
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load requires known chunks per part");
    int64_t factor = layout.getFactor();
    for (int64_t part = 1; part < factor; ++part) {
      FailureOr<int64_t> currentChunks =
          getDataChunksInPart(resultVMIType, part);
      if (failed(currentChunks) || *currentChunks != *chunksPerPart)
        return rewriter.notifyMatchFailure(
            op, "group_broadcast_load requires uniform chunks per part");
    }
    if (static_cast<int64_t>(resultTypes.size()) != factor * *chunksPerPart)
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load physical arity mismatch");
    if (*chunksPerPart != 1)
      return rewriter.notifyMatchFailure(
          op,
          "group_broadcast_load expected one E2B packet in each part");

    SmallVector<Value> packets;
    packets.reserve(*chunksPerPart);
    for (int64_t chunk = 0; chunk < *chunksPerPart; ++chunk) {
      Type packetType = resultTypes[chunk];
      auto vregType = dyn_cast<VRegType>(packetType);
      if (!vregType)
        return rewriter.notifyMatchFailure(
            op, "group_broadcast_load result must be vreg");
      Value packetOffset =
          createChunkOffset(op.getLoc(), *offset, chunk * 8, rewriter);
      packets.push_back(rewriter
                            .create<VldsOp>(op.getLoc(), packetType,
                                            /*updated_base=*/Type{}, *source,
                                            packetOffset,
                                            rewriter.getStringAttr(e2bDist))
                            .getResult());
    }

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (int64_t part = 0; part < factor; ++part) {
      for (int64_t chunk = 0; chunk < *chunksPerPart; ++chunk) {
        int64_t flatIndex = part * *chunksPerPart + chunk;
        if (resultTypes[flatIndex] != resultTypes[chunk])
          return rewriter.notifyMatchFailure(
              op, "group_broadcast_load E2B reused packet type mismatch");
        results.push_back(packets[chunk]);
      }
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }

private:
  ;
};

struct OneToNVMIStrideLoadOpPattern
    : OpConversionPattern<VMIStrideLoadOp> {
  using OpConversionPattern<VMIStrideLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIStrideLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> source = getSingleValue(
        op, adaptor.getSource(), "stride_load source must convert to one value",
        rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(), "stride_load offset must convert to one value",
        rewriter);
    FailureOr<Value> blockStride = getSingleValue(
        op, adaptor.getBlockStride(),
        "stride_load block_stride must convert to one value", rewriter);
    FailureOr<Value> repeatStride = getSingleValue(
        op, adaptor.getRepeatStride(),
        "stride_load repeat_stride must convert to one value", rewriter);
    if (failed(source) || failed(offset) || failed(blockStride) ||
        failed(repeatStride))
      return failure();

    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (resultTypes.size() != 1 || maskParts.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "stride_load supports one physical result/mask chunk");
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    if (!resultType || !isa<MaskType>(maskParts.front().getType()))
      return rewriter.notifyMatchFailure(
          op, "stride_load requires physical vreg/mask parts");

    Value base = rewriter
                     .create<AddPtrOp>(op.getLoc(), (*source).getType(),
                                       *source, *offset)
                     .getResult();
    Value loaded =
        rewriter
            .create<VsldbOp>(op.getLoc(), resultType, base, *blockStride,
                             *repeatStride, maskParts.front())
            .getResult();
    replaceOpWithFlatConvertedValues(rewriter, op, SmallVector<Value>{loaded},
                                     *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIStrideStoreOpPattern
    : OpConversionPattern<VMIStrideStoreOp> {
  using OpConversionPattern<VMIStrideStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIStrideStoreOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "stride_store destination must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(),
        "stride_store offset must convert to one value", rewriter);
    FailureOr<Value> blockStride = getSingleValue(
        op, adaptor.getBlockStride(),
        "stride_store block_stride must convert to one value", rewriter);
    FailureOr<Value> repeatStride = getSingleValue(
        op, adaptor.getRepeatStride(),
        "stride_store repeat_stride must convert to one value", rewriter);
    if (failed(destination) || failed(offset) || failed(blockStride) ||
        failed(repeatStride))
      return failure();

    ValueRange valueParts = adaptor.getValue();
    ValueRange maskParts = adaptor.getMask();
    if (valueParts.size() != 1 || maskParts.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "stride_store supports one physical value/mask chunk");
    if (!isa<VRegType>(valueParts.front().getType()) ||
        !isa<MaskType>(maskParts.front().getType()))
      return rewriter.notifyMatchFailure(
          op, "stride_store requires physical vreg/mask parts");

    Value base = rewriter
                     .create<AddPtrOp>(op.getLoc(), (*destination).getType(),
                                       *destination, *offset)
                     .getResult();
    rewriter.create<VsstbOp>(op.getLoc(), base.getType(), valueParts.front(),
                             base, *blockStride, *repeatStride,
                             maskParts.front());
    rewriter.eraseOp(op);
    return success();
  }
};

struct OneToNVMIScatterOpPattern : OpConversionPattern<VMIScatterOp> {
  using OpConversionPattern<VMIScatterOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIScatterOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "scatter destination must convert to one value", rewriter);
    if (failed(destination))
      return failure();

    ValueRange valueParts = adaptor.getValue();
    ValueRange indicesParts = adaptor.getIndices();
    ValueRange maskParts = adaptor.getMask();
    if (valueParts.size() != indicesParts.size() ||
        valueParts.size() != maskParts.size())
      return rewriter.notifyMatchFailure(op, "scatter physical arity mismatch");

    for (auto [value, indices, mask] :
         llvm::zip_equal(valueParts, indicesParts, maskParts)) {
      if (!isa<VRegType>(value.getType()) ||
          !isa<VRegType>(indices.getType()) || !isa<MaskType>(mask.getType()))
        return rewriter.notifyMatchFailure(
            op, "scatter physical part type mismatch");
      rewriter.create<VscatterOp>(op.getLoc(), value, *destination, indices,
                                  mask);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct OneToNVMIBinaryOpPattern : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OpConversionPattern<SourceOp>::OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ValueRange lhsParts = adaptor.getLhs();
    ValueRange rhsParts = adaptor.getRhs();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (lhsParts.size() != rhsParts.size() ||
        lhsParts.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(op, "physical binary arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [lhs, rhs, resultType] :
         llvm::zip_equal(lhsParts, rhsParts, resultTypes)) {
      auto vregType = dyn_cast<VRegType>(resultType);
      if (!vregType || lhs.getType() != resultType ||
          rhs.getType() != resultType)
        return rewriter.notifyMatchFailure(
            op, "physical binary part type mismatch");
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for all-true binary mask");
      results.push_back(
          rewriter.create<TargetOp>(op.getLoc(), resultType, lhs, rhs, *mask)
              .getResult());
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIVmullOpPattern : OpConversionPattern<VMIVmullOp> {
  using OpConversionPattern<VMIVmullOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIVmullOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange aParts = adaptor.getA();
    ValueRange bParts = adaptor.getB();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybeLowTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    FailureOr<SmallVector<Type>> maybeHighTypes =
        getConvertedResultTypes(op, 1, *this->getTypeConverter());
    if (failed(maybeLowTypes) || failed(maybeHighTypes))
      return failure();
    SmallVector<Type> lowTypes = std::move(*maybeLowTypes);
    SmallVector<Type> highTypes = std::move(*maybeHighTypes);

    size_t arity = aParts.size();
    if (arity == 0 || bParts.size() != arity || maskParts.size() != arity ||
        lowTypes.size() != arity || highTypes.size() != arity)
      return rewriter.notifyMatchFailure(
          op, "physical vmull arity mismatch across a, b, mask, low, and high");

    SmallVector<Value> lows;
    SmallVector<Value> highs;
    lows.reserve(arity);
    highs.reserve(arity);
    for (size_t index = 0; index < arity; ++index) {
      Type lowType = lowTypes[index];
      Type highType = highTypes[index];
      auto dataType = dyn_cast<VRegType>(lowType);
      auto maskType = dyn_cast<MaskType>(maskParts[index].getType());
      if (!dataType || dataType.getElementCount() != 64 ||
          lowType != highType || aParts[index].getType() != lowType ||
          bParts[index].getType() != lowType)
        return rewriter.notifyMatchFailure(
            op, "vmull requires matching 64-lane physical data part types");
      auto elementType = dyn_cast<IntegerType>(dataType.getElementType());
      if (!elementType || elementType.getWidth() != 32 ||
          (!elementType.isSignless() && !elementType.isUnsigned()))
        return rewriter.notifyMatchFailure(
            op, "vmull requires physical i32 or ui32 data parts");
      if (!maskType || !maskType.isB32())
        return rewriter.notifyMatchFailure(
            op, "vmull requires a corresponding b32 mask part");

      auto vmull = rewriter.create<VmullOp>(op.getLoc(), lowType, highType,
                                            aParts[index], bParts[index],
                                            maskParts[index]);
      lows.push_back(vmull.getLow());
      highs.push_back(vmull.getHigh());
    }

    SmallVector<Value> results;
    results.reserve(lows.size() + highs.size());
    results.append(lows);
    results.append(highs);
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct OneToNVMIInterleaveOpPattern : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OpConversionPattern<SourceOp>::OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ValueRange lhsParts = adaptor.getLhs();
    ValueRange rhsParts = adaptor.getRhs();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybeLowTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    FailureOr<SmallVector<Type>> maybeHighTypes =
        getConvertedResultTypes(op, 1, *this->getTypeConverter());
    if (failed(maybeLowTypes) || failed(maybeHighTypes))
      return failure();
    SmallVector<Type> lowTypes = std::move(*maybeLowTypes);
    SmallVector<Type> highTypes = std::move(*maybeHighTypes);
    if (lhsParts.size() != rhsParts.size() ||
        lhsParts.size() != lowTypes.size() ||
        lhsParts.size() != highTypes.size())
      return rewriter.notifyMatchFailure(op,
                                         "physical interleave arity mismatch");
    if (!maskParts.empty() && maskParts.size() != lhsParts.size())
      return rewriter.notifyMatchFailure(
          op, "physical interleave mask arity mismatch");

    auto lhsType = cast<VMIVRegType>(op.getLhs().getType());
    auto rhsType = cast<VMIVRegType>(op.getRhs().getType());
    auto maskType = cast<VMIMaskType>(op.getMask().getType());
    auto lowType = cast<VMIVRegType>(op.getLow().getType());
    auto highType = cast<VMIVRegType>(op.getHigh().getType());
    VMILayoutSupport supports;
    FailureOr<VMIInterleaveLayoutFact> fact;
    if constexpr (std::is_same_v<SourceOp, VMIVintlvOp>) {
      fact = supports.getVintlvLayoutFactForLayouts(
          lhsType, rhsType, maskType, lowType, highType);
    } else {
      fact = supports.getVdintlvLayoutFactForLayouts(
          lhsType, rhsType, maskType, lowType, highType);
    }
    if (failed(fact))
      return rewriter.notifyMatchFailure(
          op, "unsupported interleave layout relation");

    auto isContiguous = [](VMILayoutAttr layout) {
      return layout && layout.isContiguous() && layout.getLaneStride() == 1;
    };
    auto getElementDeintFactor = [](VMILayoutAttr layout) -> int64_t {
      if (layout && layout.isContiguous() && layout.getLaneStride() == 1)
        return 1;
      if (layout && layout.isDeinterleaved() && layout.getLaneStride() == 1)
        return layout.getFactor();
      return 0;
    };

    bool allSameLaneStride = fact->lhsLayout == fact->rhsLayout &&
                             fact->lhsLayout == fact->maskLayout &&
                             fact->lhsLayout == fact->lowLayout &&
                             fact->lhsLayout == fact->highLayout &&
                             fact->lhsLayout.isContiguous() &&
                             fact->lhsLayout.getLaneStride() > 1;
    if (allSameLaneStride) {
      if (lhsParts.size() != 1 || rhsParts.size() != 1 ||
          lowTypes.size() != 1 || highTypes.size() != 1)
        return rewriter.notifyMatchFailure(
            op, "lane-stride interleave expects one physical carrier part");

      unsigned elementBits =
          pto::getPTOStorageElemBitWidth(lhsType.getElementType());
      int64_t laneStride = fact->lhsLayout.getLaneStride();
      int64_t carrierBits = static_cast<int64_t>(elementBits) * laneStride;
      if (elementBits == 0 || laneStride <= 1 || carrierBits <= 0 ||
          carrierBits > 32)
        return rewriter.notifyMatchFailure(
            op, "invalid lane-stride interleave carrier width");
      FailureOr<VRegType> carrierType =
          getUnsignedCarrierVRegType(rewriter.getContext(),
                                     static_cast<unsigned>(carrierBits));
      if (failed(carrierType))
        return rewriter.notifyMatchFailure(
            op, "unsupported lane-stride interleave carrier width");

      FailureOr<Value> carrierLhs = bitcastVReg(
          op.getLoc(), lhsParts.front(), *carrierType, rewriter);
      FailureOr<Value> carrierRhs = bitcastVReg(
          op.getLoc(), rhsParts.front(), *carrierType, rewriter);
      if (failed(carrierLhs) || failed(carrierRhs))
        return rewriter.notifyMatchFailure(
            op, "failed to bitcast lane-stride interleave inputs");

      auto interleave = rewriter.create<TargetOp>(
          op.getLoc(), *carrierType, *carrierType, *carrierLhs, *carrierRhs);
      FailureOr<Value> low = bitcastVReg(
          op.getLoc(), interleave.getLow(), lowTypes.front(), rewriter);
      FailureOr<Value> high = bitcastVReg(
          op.getLoc(), interleave.getHigh(), highTypes.front(), rewriter);
      if (failed(low) || failed(high))
        return rewriter.notifyMatchFailure(
            op, "failed to bitcast lane-stride interleave results");

      SmallVector<Value, 2> results = {*low, *high};
      replaceOpWithFlatConvertedValues(rewriter, op, results,
                                       *this->getTypeConverter());
      return success();
    }

    bool allContiguous = isContiguous(fact->lhsLayout) &&
                         isContiguous(fact->rhsLayout) &&
                         isContiguous(fact->maskLayout) &&
                         isContiguous(fact->lowLayout) &&
                         isContiguous(fact->highLayout);
    if (allContiguous) {
      if (lhsParts.size() != 1 || rhsParts.size() != 1 ||
          lowTypes.size() != 1 || highTypes.size() != 1)
        return rewriter.notifyMatchFailure(
            op, "single-chunk interleave expects one physical part");
      if (!maskParts.empty() && !isa<MaskType>(maskParts.front().getType()))
        return rewriter.notifyMatchFailure(
            op, "single-chunk interleave mask part type mismatch");
      if (!isa<VRegType>(lowTypes.front()) ||
          !isa<VRegType>(highTypes.front()) ||
          lhsParts.front().getType() != lowTypes.front() ||
          rhsParts.front().getType() != lowTypes.front() ||
          highTypes.front() != lowTypes.front())
        return rewriter.notifyMatchFailure(
            op, "single-chunk interleave part type mismatch");
      auto interleave = rewriter.create<TargetOp>(
          op.getLoc(), lowTypes.front(), highTypes.front(), lhsParts.front(),
          rhsParts.front());
      SmallVector<Value, 2> directResults = {interleave.getLow(),
                                             interleave.getHigh()};
      replaceOpWithFlatConvertedValues(rewriter, op, directResults,
                                       *this->getTypeConverter());
      return success();
    }

    int64_t inputFactor = getElementDeintFactor(fact->lhsLayout);
    int64_t outputFactor = getElementDeintFactor(fact->lowLayout);
    bool zeroCopyVintlv = std::is_same_v<SourceOp, VMIVintlvOp> &&
                          inputFactor > 0 &&
                          fact->rhsLayout == fact->lhsLayout &&
                          fact->maskLayout == fact->lhsLayout &&
                          fact->highLayout == fact->lowLayout &&
                          outputFactor == 2 * inputFactor;
    bool zeroCopyVdintlv = std::is_same_v<SourceOp, VMIVdintlvOp> &&
                           inputFactor > 0 &&
                           fact->rhsLayout == fact->lhsLayout &&
                           fact->maskLayout == fact->lhsLayout &&
                           fact->highLayout == fact->lowLayout &&
                           inputFactor == 2 * outputFactor;
    if (!zeroCopyVintlv && !zeroCopyVdintlv)
      return rewriter.notifyMatchFailure(
          op, "unsupported interleave physical layout relation");

    SmallVector<Value> results;
    results.reserve(lhsParts.size() + rhsParts.size());
    if (zeroCopyVintlv) {
      if (lhsParts.empty() || lhsParts.size() % (2 * inputFactor) != 0)
        return rewriter.notifyMatchFailure(
            op, "zero-copy vintlv expects input groups with even chunk count");
      size_t groupChunks = lhsParts.size() / inputFactor;
      size_t halfGroupChunks = groupChunks / 2;
      for (int64_t group = 0; group < inputFactor; ++group) {
        size_t offset = group * groupChunks;
        llvm::append_range(
            results,
            lhsParts.slice(offset, halfGroupChunks));
        llvm::append_range(
            results,
            rhsParts.slice(offset, halfGroupChunks));
      }
      for (int64_t group = 0; group < inputFactor; ++group) {
        size_t offset = group * groupChunks + halfGroupChunks;
        llvm::append_range(
            results,
            lhsParts.slice(offset, halfGroupChunks));
        llvm::append_range(
            results,
            rhsParts.slice(offset, halfGroupChunks));
      }
    } else {
      if (lhsParts.empty() || lhsParts.size() % inputFactor != 0)
        return rewriter.notifyMatchFailure(
            op, "zero-copy vdintlv expects complete input layout groups");
      size_t groupChunks = lhsParts.size() / inputFactor;
      for (int64_t group = 0; group < outputFactor; ++group) {
        size_t offset = 2 * group * groupChunks;
        llvm::append_range(results, lhsParts.slice(offset, groupChunks));
        llvm::append_range(results, rhsParts.slice(offset, groupChunks));
      }
      for (int64_t group = 0; group < outputFactor; ++group) {
        size_t offset = (2 * group + 1) * groupChunks;
        llvm::append_range(results, lhsParts.slice(offset, groupChunks));
        llvm::append_range(results, rhsParts.slice(offset, groupChunks));
      }
    }

    SmallVector<Type> resultTypes;
    resultTypes.reserve(lowTypes.size() + highTypes.size());
    llvm::append_range(resultTypes, lowTypes);
    llvm::append_range(resultTypes, highTypes);
    if (results.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(
          op, "zero-copy interleave result arity mismatch");
    for (auto [value, resultType] : llvm::zip_equal(results, resultTypes)) {
      if (value.getType() != resultType)
        return rewriter.notifyMatchFailure(
            op, "zero-copy interleave part type mismatch");
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIFmaOpPattern : OpConversionPattern<VMIFmaOp> {
  using OpConversionPattern<VMIFmaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIFmaOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange lhsParts = adaptor.getLhs();
    ValueRange rhsParts = adaptor.getRhs();
    ValueRange accParts = adaptor.getAcc();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (lhsParts.size() != rhsParts.size() ||
        lhsParts.size() != accParts.size() ||
        lhsParts.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(op, "fma physical arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [lhs, rhs, acc, resultType] :
         llvm::zip_equal(lhsParts, rhsParts, accParts, resultTypes)) {
      auto vregType = dyn_cast<VRegType>(resultType);
      if (!vregType || lhs.getType() != resultType ||
          rhs.getType() != resultType || acc.getType() != resultType)
        return rewriter.notifyMatchFailure(
            op, "fma requires matching physical vreg parts");
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(op,
                                           "unsupported element type for fma");
      results.push_back(
          rewriter
              .create<VmulaOp>(op.getLoc(), resultType, acc, lhs, rhs, *mask)
              .getResult());
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct OneToNVMIUnaryOpPattern : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OpConversionPattern<SourceOp>::OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (sourceParts.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(op, "physical unary arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [source, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      auto vregType = dyn_cast<VRegType>(resultType);
      if (!vregType || source.getType() != resultType)
        return rewriter.notifyMatchFailure(op,
                                           "physical unary part type mismatch");
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for all-true unary mask");
      results.push_back(
          rewriter.create<TargetOp>(op.getLoc(), resultType, source, *mask)
              .getResult());
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct OneToNVMIMaskBinaryOpPattern : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OpConversionPattern<SourceOp>::OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ValueRange lhsParts = adaptor.getLhs();
    ValueRange rhsParts = adaptor.getRhs();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (lhsParts.size() != rhsParts.size() ||
        lhsParts.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(op,
                                         "physical mask binary arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [lhs, rhs, resultType] :
         llvm::zip_equal(lhsParts, rhsParts, resultTypes)) {
      auto maskType = dyn_cast<MaskType>(resultType);
      if (!maskType || lhs.getType() != resultType ||
          rhs.getType() != resultType)
        return rewriter.notifyMatchFailure(
            op, "physical mask binary part type mismatch");
      FailureOr<Value> seedMask =
          createAllTrueMask(op.getLoc(), maskType, rewriter);
      if (failed(seedMask))
        return rewriter.notifyMatchFailure(
            op, "unsupported mask type for all-true mask binary seed");
      results.push_back(
          rewriter
              .create<TargetOp>(op.getLoc(), resultType, lhs, rhs, *seedMask)
              .getResult());
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct OneToNVMIMaskUnaryOpPattern : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OpConversionPattern<SourceOp>::OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (sourceParts.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(op,
                                         "physical mask unary arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [source, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      auto maskType = dyn_cast<MaskType>(resultType);
      if (!maskType || source.getType() != resultType)
        return rewriter.notifyMatchFailure(
            op, "physical mask unary part type mismatch");
      FailureOr<Value> seedMask =
          createAllTrueMask(op.getLoc(), maskType, rewriter);
      if (failed(seedMask))
        return rewriter.notifyMatchFailure(
            op, "unsupported mask type for all-true mask unary seed");
      results.push_back(
          rewriter.create<TargetOp>(op.getLoc(), resultType, source, *seedMask)
              .getResult());
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

template <typename SourceOp>
struct OneToNVMICmpOpPattern : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OpConversionPattern<SourceOp>::OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    std::optional<VPTOCmpMode> cmpMode =
        getVPTOCmpMode<SourceOp>(op.getPredicate());
    if (!cmpMode)
      return op.emitOpError()
             << kVMIDiagUnsupportedPrefix << "compare predicate "
             << op.getPredicate()
             << " cannot be lowered to pto.vcmp; supported predicates are "
             << getSupportedComparePredicateMessage<SourceOp>();

    ValueRange lhsParts = adaptor.getLhs();
    ValueRange rhsParts = adaptor.getRhs();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (lhsParts.size() != rhsParts.size() ||
        lhsParts.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(op, "physical cmp arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [lhs, rhs, resultType] :
         llvm::zip_equal(lhsParts, rhsParts, resultTypes)) {
      auto maskType = dyn_cast<MaskType>(resultType);
      auto lhsType = dyn_cast<VRegType>(lhs.getType());
      if (!maskType || lhs.getType() != rhs.getType() || !lhsType)
        return rewriter.notifyMatchFailure(op,
                                           "physical cmp part type mismatch");
      FailureOr<Value> seedMask =
          createAllTrueMask(op.getLoc(), maskType, rewriter);
      if (failed(seedMask))
        return rewriter.notifyMatchFailure(
            op, "unsupported mask type for all-true cmp seed");
      if (cmpMode->signedness) {
        FailureOr<VRegType> carrierType =
            getSignednessCarrierVRegType(lhsType, *cmpMode->signedness);
        if (failed(carrierType))
          return rewriter.notifyMatchFailure(
              op, "unsupported integer compare signedness carrier");
        FailureOr<Value> carrierLhs =
            bitcastVReg(op.getLoc(), lhs, *carrierType, rewriter);
        FailureOr<Value> carrierRhs =
            bitcastVReg(op.getLoc(), rhs, *carrierType, rewriter);
        if (failed(carrierLhs) || failed(carrierRhs))
          return rewriter.notifyMatchFailure(
              op, "failed to materialize integer compare signedness carrier");
        lhs = *carrierLhs;
        rhs = *carrierRhs;
      }
      results.push_back(rewriter
                            .create<VcmpOp>(op.getLoc(), resultType, lhs, rhs,
                                            *seedMask,
                                            rewriter.getStringAttr(cmpMode->mode))
                            .getResult());
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMISelectOpPattern : OpConversionPattern<VMISelectOp> {
  using OpConversionPattern<VMISelectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMISelectOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange maskParts = adaptor.getMask();
    ValueRange trueParts = adaptor.getTrueValue();
    ValueRange falseParts = adaptor.getFalseValue();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (maskParts.size() != trueParts.size() ||
        trueParts.size() != falseParts.size() ||
        trueParts.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(op, "physical select arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [mask, trueValue, falseValue, resultType] :
         llvm::zip_equal(maskParts, trueParts, falseParts, resultTypes)) {
      if (!isa<MaskType>(mask.getType()) || trueValue.getType() != resultType ||
          falseValue.getType() != resultType || !isa<VRegType>(resultType))
        return rewriter.notifyMatchFailure(
            op, "physical select part type mismatch");
      results.push_back(rewriter
                            .create<VselOp>(op.getLoc(), resultType, trueValue,
                                            falseValue, mask)
                            .getResult());
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIActivePrefixIndexOpPattern
    : OpConversionPattern<VMIActivePrefixIndexOp> {
  using OpConversionPattern<
      VMIActivePrefixIndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIActivePrefixIndexOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (maskParts.size() != 1 || resultTypes.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "active_prefix_index supports only one physical part");

    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType)
      return rewriter.notifyMatchFailure(
          op, "active_prefix_index requires physical vreg/mask parts");

    auto intType = dyn_cast<IntegerType>(resultType.getElementType());
    if (!intType || !intType.isSignless())
      return rewriter.notifyMatchFailure(
          op, "active_prefix_index requires signless integer result part");

    FailureOr<Value> seedMask =
        createAllTrueMaskForVReg(op.getLoc(), resultType, rewriter);
    if (failed(seedMask))
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for active_prefix_index seed mask");

    Value zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0,
                                                       intType.getWidth());
    Value carrier =
        rewriter
            .create<VdupOp>(op.getLoc(), resultType, zero, *seedMask,
                            /*position=*/nullptr)
            .getResult();
    Value result = rewriter
                       .create<VusqzOp>(op.getLoc(), resultType, carrier,
                                        maskParts.front())
                       .getResult();
    replaceOpWithFlatConvertedValues(rewriter, op, SmallVector<Value>{result},
                                     *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMICompressOpPattern : OpConversionPattern<VMICompressOp> {
  using OpConversionPattern<VMICompressOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMICompressOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (sourceParts.size() != 1 || maskParts.size() != 1 ||
        resultTypes.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "compress supports only one physical part");

    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    if (!resultType || sourceParts.front().getType() != resultType ||
        !isa<MaskType>(maskParts.front().getType()))
      return rewriter.notifyMatchFailure(
          op, "compress requires physical source/mask/result parts");

    Value result = rewriter
                       .create<VsqzOp>(op.getLoc(), resultType,
                                       sourceParts.front(), maskParts.front())
                       .getResult();
    replaceOpWithFlatConvertedValues(rewriter, op, SmallVector<Value>{result},
                                     *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMICompressStoreOpPattern
    : OpConversionPattern<VMICompressStoreOp> {
  using OpConversionPattern<
      VMICompressStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMICompressStoreOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "compress_store destination must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(),
        "compress_store offset must convert to one value", rewriter);
    if (failed(destination) || failed(offset))
      return failure();

    ValueRange valueParts = adaptor.getValue();
    ValueRange maskParts = adaptor.getMask();
    if (valueParts.size() != 1 || maskParts.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "compress_store supports only one physical part");

    auto valueType = dyn_cast<VRegType>(valueParts.front().getType());
    if (!valueType || !isa<MaskType>(maskParts.front().getType()) ||
        !isa<PtrType>((*destination).getType()))
      return rewriter.notifyMatchFailure(
          op, "compress_store requires physical value/mask and ptr "
              "destination");

    Value storeBase =
        rewriter
            .create<AddPtrOp>(op.getLoc(), (*destination).getType(),
                              *destination, *offset)
            .getResult();
    Value squeezed = rewriter
                         .create<VsqzOp>(op.getLoc(), valueType,
                                         valueParts.front(), maskParts.front())
                         .getResult();
    auto align = rewriter.create<InitAlignOp>(
        op.getLoc(), AlignType::get(rewriter.getContext()));
    auto store = rewriter.create<VsturOp>(
        op.getLoc(), align.getResult().getType(), align.getResult(), squeezed,
        storeBase, rewriter.getStringAttr("POST_UPDATE"));
    rewriter.create<VstarOp>(op.getLoc(), store.getAlignOut(), storeBase);
    rewriter.eraseOp(op);
    return success();
  }
};

struct OneToNVMIReduceAddIOpPattern
    : OpConversionPattern<VMIReduceAddIOp> {
  using OpConversionPattern<VMIReduceAddIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIReduceAddIOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    ValueRange initParts = adaptor.getInit();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (sourceParts.empty() || sourceParts.size() != maskParts.size() ||
        initParts.size() != 1 || resultTypes.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "reduce_addi requires matching source/mask chunks and one "
              "init/result chunk");

    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType || initParts.front().getType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "reduce_addi requires matching physical source/init/result "
              "vregs and one mask");

    for (Value sourcePart : sourceParts)
      if (sourcePart.getType() != resultType)
        return rewriter.notifyMatchFailure(
            op, "reduce_addi requires every source chunk to match result "
                "vreg type");
    for (Value maskPart : maskParts)
      if (maskPart.getType() != maskType)
        return rewriter.notifyMatchFailure(
            op, "reduce_addi requires every mask chunk to have the same "
                "predicate type");

    FailureOr<Value> firstLaneMask =
        createPrefixMask(op.getLoc(), maskType, "PAT_VL1", rewriter);
    if (failed(firstLaneMask))
      return rewriter.notifyMatchFailure(
          op, "failed to create reduce_addi first-lane mask");

    FailureOr<Value> combined = combineEquivalentMaskedParts<VaddOp>(
        op.getLoc(), sourceParts, maskParts, resultType, rewriter);
    if (succeeded(combined)) {
      Value reduced =
          rewriter
              .create<VcaddOp>(op.getLoc(), resultType, *combined,
                               maskParts.front())
              .getResult();
      Value result =
          rewriter
              .create<VaddOp>(op.getLoc(), resultType, reduced,
                              initParts.front(), *firstLaneMask)
              .getResult();
      replaceOpWithFlatConvertedValues(
          rewriter, op, SmallVector<Value>{result},
          *this->getTypeConverter());
      return success();
    }

    Value accumulator = initParts.front();
    for (auto [sourcePart, maskPart] :
         llvm::zip_equal(sourceParts, maskParts)) {
      Value reduced =
          rewriter
              .create<VcaddOp>(op.getLoc(), resultType, sourcePart, maskPart)
              .getResult();
      accumulator = rewriter
                        .create<VaddOp>(op.getLoc(), resultType, reduced,
                                        accumulator, *firstLaneMask)
                        .getResult();
    }

    replaceOpWithFlatConvertedValues(
            rewriter, op, SmallVector<Value>{accumulator},
            *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIReduceAddFOpPattern
    : OpConversionPattern<VMIReduceAddFOp> {
  using OpConversionPattern<VMIReduceAddFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIReduceAddFOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    ValueRange initParts = adaptor.getInit();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (sourceParts.empty() || sourceParts.size() != maskParts.size() ||
        initParts.size() != 1 || resultTypes.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "reduce_addf requires matching source/mask chunks and one "
              "init/result chunk");

    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType || initParts.front().getType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "reduce_addf requires matching physical source/init/result "
              "vregs and one mask");

    for (Value sourcePart : sourceParts)
      if (sourcePart.getType() != resultType)
        return rewriter.notifyMatchFailure(
            op, "reduce_addf requires every source chunk to match result "
                "vreg type");
    for (Value maskPart : maskParts)
      if (maskPart.getType() != maskType)
        return rewriter.notifyMatchFailure(
            op, "reduce_addf requires every mask chunk to have the same "
                "predicate type");

    FailureOr<Value> firstLaneMask =
        createPrefixMask(op.getLoc(), maskType, "PAT_VL1", rewriter);
    if (failed(firstLaneMask))
      return rewriter.notifyMatchFailure(
          op, "failed to create reduce_addf first-lane mask");

    FailureOr<Value> combined = combineEquivalentMaskedParts<VaddOp>(
        op.getLoc(), sourceParts, maskParts, resultType, rewriter);
    if (succeeded(combined)) {
      Value reduced =
          rewriter
              .create<VcaddOp>(op.getLoc(), resultType, *combined,
                               maskParts.front())
              .getResult();
      Value result =
          rewriter
              .create<VaddOp>(op.getLoc(), resultType, reduced,
                              initParts.front(), *firstLaneMask)
              .getResult();
      replaceOpWithFlatConvertedValues(
          rewriter, op, SmallVector<Value>{result},
          *this->getTypeConverter());
      return success();
    }

    Value accumulator = initParts.front();
    for (auto [sourcePart, maskPart] :
         llvm::zip_equal(sourceParts, maskParts)) {
      Value reduced =
          rewriter
              .create<VcaddOp>(op.getLoc(), resultType, sourcePart, maskPart)
              .getResult();
      accumulator = rewriter
                        .create<VaddOp>(op.getLoc(), resultType, reduced,
                                        accumulator, *firstLaneMask)
                        .getResult();
    }

    replaceOpWithFlatConvertedValues(
            rewriter, op, SmallVector<Value>{accumulator},
            *this->getTypeConverter());
    return success();
  }
};

enum class GroupReduceLoweringPlan {
  OneBlockVcgadd,
  TwoBlockDeinterleaved2VcgaddVadd,
  FourBlockDeinterleaved4VcgaddTree,
  FullDeinterleaved2VcaddRows,
  ContiguousVcaddRows,
};

FailureOr<GroupReduceLoweringPlan>
classifyGroupReduceLoweringPlan(VMIVRegType sourceType, VMIMaskType maskType,
                                VMIVRegType resultType, int64_t numGroups,
                                std::string *reason = nullptr) {
  VMILayoutSupport supports;
  FailureOr<VMIGroupReduceLayoutFact> fact =
      supports.getGroupReduceLayoutFactForLayouts(
          sourceType, maskType, resultType, numGroups, reason);
  if (failed(fact))
    return failure();

  switch (fact->blockClass) {
  case VMIGroupBlockClass::QuarterBlock:
  case VMIGroupBlockClass::HalfBlock:
  case VMIGroupBlockClass::OneBlock:
    return GroupReduceLoweringPlan::OneBlockVcgadd;
  case VMIGroupBlockClass::TwoBlock:
    return GroupReduceLoweringPlan::TwoBlockDeinterleaved2VcgaddVadd;
  case VMIGroupBlockClass::FourBlock:
    return GroupReduceLoweringPlan::FourBlockDeinterleaved4VcgaddTree;
  case VMIGroupBlockClass::FullPartMultiple:
    if (fact->sourceLayout && fact->sourceLayout.isDeinterleaved() &&
        fact->sourceLayout.getFactor() == 2)
      return GroupReduceLoweringPlan::FullDeinterleaved2VcaddRows;
    return GroupReduceLoweringPlan::ContiguousVcaddRows;
  }
  llvm_unreachable("unknown group block class");
}

template <typename OpTy, typename GroupReduceOpTy, typename RowReduceOpTy,
          typename CombineOpTy>
struct OneToNVMIGroupReduceOpPattern : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op,
                  typename OpConversionPattern<OpTy>::OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    ValueRange sourceParts = adaptor.getSource();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);

    VMILayoutSupport supports;
    std::string supportReason;
    if (failed(getSupport(supports, op, &supportReason)))
      return rewriter.notifyMatchFailure(
          op, Twine(op->getName().getStringRef()) +
                  " has no layout support: " + supportReason);
    auto maskVMIType = cast<VMIMaskType>(op.getMask().getType());
    FailureOr<GroupReduceLoweringPlan> plan = classifyGroupReduceLoweringPlan(
        sourceVMIType, maskVMIType, resultVMIType,
        op.getNumGroupsAttr().getInt(), &supportReason);
    if (failed(plan))
      return rewriter.notifyMatchFailure(
          op, Twine(op->getName().getStringRef()) +
                  " has no lowering plan: " + supportReason);

    FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
        sourceVMIType, op.getNumGroupsAttr().getInt());
    if (failed(groupSize))
      return rewriter.notifyMatchFailure(
          op, "group reduce requires num_groups to evenly divide lane count");

    if (*plan == GroupReduceLoweringPlan::OneBlockVcgadd) {
      if (sourceParts.size() != maskParts.size() ||
          sourceParts.size() != resultTypes.size() || sourceParts.empty())
        return rewriter.notifyMatchFailure(
            op, "vcg group_reduce path requires matching physical "
                "arity");
      auto resultType = dyn_cast<VRegType>(resultTypes.front());
      auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
      if (!resultType || !maskType)
        return rewriter.notifyMatchFailure(
            op, "vcg group_reduce path requires physical vreg/mask");
      for (auto [sourcePart, maskPart, physicalResultType] :
           llvm::zip_equal(sourceParts, maskParts, resultTypes)) {
        if (sourcePart.getType() != resultType ||
            maskPart.getType() != maskType || physicalResultType != resultType)
          return rewriter.notifyMatchFailure(
              op, "vcg group_reduce path requires uniform physical "
                  "chunk types");
      }

      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      for (auto [sourceIndex, sourcePart] : llvm::enumerate(sourceParts)) {
        results.push_back(rewriter
                              .create<GroupReduceOpTy>(op.getLoc(), resultType,
                                                       sourcePart,
                                                       maskParts[sourceIndex])
                              .getResult());
      }

      replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
      return success();
    }

    if (*plan == GroupReduceLoweringPlan::TwoBlockDeinterleaved2VcgaddVadd) {
      int64_t resultPartCount = resultTypes.size();
      if (static_cast<int64_t>(sourceParts.size()) != resultPartCount * 2 ||
          maskParts.size() != sourceParts.size())
        return rewriter.notifyMatchFailure(
            op, "two-block group_reduce arity mismatch");

      SmallVector<Value> results;
      results.reserve(resultPartCount);
      auto resultType = dyn_cast<VRegType>(resultTypes.front());
      auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
      if (!resultType || !maskType)
        return rewriter.notifyMatchFailure(
            op, "two-block group_reduce requires physical vreg/mask");
      int64_t numGroups = op.getNumGroupsAttr().getInt();

      for (int64_t resultIndex = 0; resultIndex < resultPartCount;
           ++resultIndex) {
        Value loSource = sourceParts[resultIndex];
        Value hiSource = sourceParts[resultPartCount + resultIndex];
        Value loMask = maskParts[resultIndex];
        Value hiMask = maskParts[resultPartCount + resultIndex];
        Type physicalResultType = resultTypes[resultIndex];
        if (physicalResultType != resultType ||
            loSource.getType() != resultType ||
            hiSource.getType() != resultType || loMask.getType() != maskType ||
            hiMask.getType() != maskType)
          return rewriter.notifyMatchFailure(
              op, "two-block group_reduce requires uniform physical "
                  "types");

        SmallVector<Value, 2> sources{loSource, hiSource};
        SmallVector<Value, 2> masks{loMask, hiMask};
        FailureOr<Value> combined = combineEquivalentMaskedParts<CombineOpTy>(
            op.getLoc(), sources, masks, resultType, rewriter);
        if (succeeded(combined)) {
          results.push_back(
              rewriter
                  .create<GroupReduceOpTy>(op.getLoc(), resultType, *combined,
                                           loMask)
                  .getResult());
          continue;
        }

        int64_t activeGroups =
            std::min<int64_t>(8, numGroups - resultIndex * 8);
        FailureOr<Value> combineMask = createPrefixMaskForActiveLanes(
            op.getLoc(), maskType, activeGroups, rewriter);
        if (failed(combineMask))
          return rewriter.notifyMatchFailure(
              op, "failed to create two-block group_reduce combine mask");
        Value lo = rewriter
                       .create<GroupReduceOpTy>(op.getLoc(), resultType,
                                                loSource, loMask)
                       .getResult();
        Value hi = rewriter
                       .create<GroupReduceOpTy>(op.getLoc(), resultType,
                                                hiSource, hiMask)
                       .getResult();
        results.push_back(rewriter
                              .create<CombineOpTy>(op.getLoc(), resultType, lo,
                                                   hi, *combineMask)
                              .getResult());
      }

      replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
      return success();
    }

    if (*plan == GroupReduceLoweringPlan::FourBlockDeinterleaved4VcgaddTree) {
      int64_t resultPartCount = resultTypes.size();
      if (static_cast<int64_t>(sourceParts.size()) != resultPartCount * 4 ||
          maskParts.size() != sourceParts.size())
        return rewriter.notifyMatchFailure(
            op, "four-block group_reduce arity mismatch");

      SmallVector<Value> results;
      results.reserve(resultPartCount);
      auto resultType = dyn_cast<VRegType>(resultTypes.front());
      auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
      if (!resultType || !maskType)
        return rewriter.notifyMatchFailure(
            op, "four-block group_reduce requires physical vreg/mask");
      int64_t numGroups = op.getNumGroupsAttr().getInt();

      for (int64_t resultIndex = 0; resultIndex < resultPartCount;
           ++resultIndex) {
        SmallVector<Value, 4> sources;
        SmallVector<Value, 4> masks;
        sources.reserve(4);
        masks.reserve(4);
        SmallVector<Value, 4> partials;
        partials.reserve(4);
        for (int64_t part = 0; part < 4; ++part) {
          int64_t sourceIndex = part * resultPartCount + resultIndex;
          Value source = sourceParts[sourceIndex];
          Value mask = maskParts[sourceIndex];
          Type physicalResultType = resultTypes[resultIndex];
          if (physicalResultType != resultType ||
              source.getType() != resultType || mask.getType() != maskType)
            return rewriter.notifyMatchFailure(
                op, "four-block group_reduce requires uniform physical "
                    "types");
          sources.push_back(source);
          masks.push_back(mask);
        }

        FailureOr<Value> combined = combineEquivalentMaskedParts<CombineOpTy>(
            op.getLoc(), sources, masks, resultType, rewriter);
        if (succeeded(combined)) {
          results.push_back(
              rewriter
                  .create<GroupReduceOpTy>(op.getLoc(), resultType, *combined,
                                           masks.front())
                  .getResult());
          continue;
        }

        int64_t activeGroups =
            std::min<int64_t>(8, numGroups - resultIndex * 8);
        FailureOr<Value> combineMask = createPrefixMaskForActiveLanes(
            op.getLoc(), maskType, activeGroups, rewriter);
        if (failed(combineMask))
          return rewriter.notifyMatchFailure(
              op, "failed to create four-block group_reduce combine mask");
        for (auto [source, mask] : llvm::zip_equal(sources, masks))
          partials.push_back(rewriter
                                 .create<GroupReduceOpTy>(
                                     op.getLoc(), resultType, source, mask)
                                 .getResult());
        Value sum01 =
            rewriter
                .create<CombineOpTy>(op.getLoc(), resultType, partials[0],
                                     partials[1], *combineMask)
                .getResult();
        Value sum23 =
            rewriter
                .create<CombineOpTy>(op.getLoc(), resultType, partials[2],
                                     partials[3], *combineMask)
                .getResult();
        results.push_back(rewriter
                              .create<CombineOpTy>(op.getLoc(), resultType,
                                                   sum01, sum23, *combineMask)
                              .getResult());
      }

      replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
      return success();
    }

    if (*plan == GroupReduceLoweringPlan::FullDeinterleaved2VcaddRows) {
      VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
      bool rowLocalSlots1Result = resultLayout && resultLayout.isGroupSlots() &&
                                  resultLayout.getSlots() == 1;
      if (!rowLocalSlots1Result)
        return rewriter.notifyMatchFailure(
            op, "deinterleaved=2 full group_reduce requires slots=1 result");

      FailureOr<int64_t> lanesPerPart =
          getDataLanesPerPart(sourceVMIType.getElementType());
      if (failed(lanesPerPart))
        return rewriter.notifyMatchFailure(
            op, "deinterleaved=2 group_reduce requires known physical lanes");
      if (*groupSize % (2 * *lanesPerPart) != 0)
        return rewriter.notifyMatchFailure(
            op, "deinterleaved=2 group_reduce requires group size to be a "
                "multiple of two physical chunks");
      int64_t groupCount = sourceVMIType.getElementCount() / *groupSize;
      int64_t chunksPerGroupPerPart = *groupSize / (2 * *lanesPerPart);
      int64_t chunksPerPart = groupCount * chunksPerGroupPerPart;
      if (sourceParts.size() != maskParts.size() ||
          static_cast<int64_t>(sourceParts.size()) != 2 * chunksPerPart ||
          static_cast<int64_t>(resultTypes.size()) != groupCount)
        return rewriter.notifyMatchFailure(
            op, "deinterleaved=2 group_reduce arity mismatch");

      SmallVector<Value> results(resultTypes.size());
      for (Type resultType : resultTypes) {
        if (!isa<VRegType>(resultType))
          return rewriter.notifyMatchFailure(
              op, "deinterleaved=2 group_reduce result must be vreg");
      }

      auto resultType = dyn_cast<VRegType>(resultTypes.front());
      auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
      if (!resultType || !maskType)
        return rewriter.notifyMatchFailure(
            op, "deinterleaved=2 group_reduce requires physical vreg/mask");
      auto sourcePartType = dyn_cast<VRegType>(sourceParts.front().getType());
      if (!sourcePartType)
        return rewriter.notifyMatchFailure(
            op, "deinterleaved=2 group_reduce source must be vreg");
      FailureOr<VRegType> rowResultType =
          getRowResultType(sourcePartType, resultType);
      if (failed(rowResultType))
        return rewriter.notifyMatchFailure(
            op, "failed to derive deinterleaved=2 row-reduction type");
      FailureOr<MaskType> rowMaskType =
          getMaskTypeForVReg(*rowResultType, rewriter.getContext());
      if (failed(rowMaskType))
        return rewriter.notifyMatchFailure(
            op, "failed to derive deinterleaved=2 combine mask type");
      FailureOr<Value> firstLaneMask = createPrefixMask(
          op.getLoc(), *rowMaskType, "PAT_VL1", rewriter);
      if (failed(firstLaneMask))
        return rewriter.notifyMatchFailure(
            op, "failed to create deinterleaved=2 group_reduce lane mask");

      for (int64_t group = 0; group < groupCount; ++group) {
        SmallVector<Value> groupSources;
        SmallVector<Value> groupMasks;
        groupSources.reserve(2 * chunksPerGroupPerPart);
        groupMasks.reserve(2 * chunksPerGroupPerPart);
        for (int64_t chunk = 0; chunk < chunksPerGroupPerPart; ++chunk) {
          int64_t loIndex = group * chunksPerGroupPerPart + chunk;
          int64_t hiIndex = chunksPerPart + loIndex;
          if (sourceParts[loIndex].getType() != sourcePartType ||
              sourceParts[hiIndex].getType() != sourcePartType ||
              maskParts[loIndex].getType() != maskType ||
              maskParts[hiIndex].getType() != maskType)
            return rewriter.notifyMatchFailure(
                op, "deinterleaved=2 group_reduce requires uniform physical "
                    "chunk types");
          groupSources.push_back(sourceParts[loIndex]);
          groupSources.push_back(sourceParts[hiIndex]);
          groupMasks.push_back(maskParts[loIndex]);
          groupMasks.push_back(maskParts[hiIndex]);
        }
        FailureOr<Value> combined = combineEquivalentMaskedParts<CombineOpTy>(
            op.getLoc(), groupSources, groupMasks, sourcePartType, rewriter);
        if (succeeded(combined)) {
          Value reduced =
              rewriter
                  .create<RowReduceOpTy>(op.getLoc(), *rowResultType, *combined,
                                         groupMasks.front())
                  .getResult();
          FailureOr<Value> finalResult =
              bitcastVReg(op.getLoc(), reduced, resultType, rewriter);
          if (failed(finalResult))
            return rewriter.notifyMatchFailure(
                op, "failed to restore deinterleaved=2 group result type");
          results[group] = *finalResult;
          continue;
        }

        Value accumulator;
        for (int64_t chunk = 0; chunk < chunksPerGroupPerPart; ++chunk) {
          int64_t loIndex = group * chunksPerGroupPerPart + chunk;
          int64_t hiIndex = chunksPerPart + loIndex;
          if (sourceParts[loIndex].getType() != sourcePartType ||
              sourceParts[hiIndex].getType() != sourcePartType ||
              maskParts[loIndex].getType() != maskType ||
              maskParts[hiIndex].getType() != maskType)
            return rewriter.notifyMatchFailure(
                op, "deinterleaved=2 group_reduce requires uniform physical "
                    "chunk types");

          Value loReduced =
              rewriter
                  .create<RowReduceOpTy>(op.getLoc(), *rowResultType,
                                         sourceParts[loIndex],
                                         maskParts[loIndex])
                  .getResult();
          Value hiReduced =
              rewriter
                  .create<RowReduceOpTy>(op.getLoc(), *rowResultType,
                                         sourceParts[hiIndex],
                                         maskParts[hiIndex])
                  .getResult();
          Value pairReduced =
              rewriter
                  .create<CombineOpTy>(op.getLoc(), *rowResultType, loReduced,
                                       hiReduced, *firstLaneMask)
                  .getResult();
          if (!accumulator) {
            accumulator = pairReduced;
            continue;
          }
          accumulator =
              rewriter
                  .create<CombineOpTy>(op.getLoc(), *rowResultType, pairReduced,
                                       accumulator, *firstLaneMask)
                  .getResult();
        }
        FailureOr<Value> finalResult =
            bitcastVReg(op.getLoc(), accumulator, resultType, rewriter);
        if (failed(finalResult))
          return rewriter.notifyMatchFailure(
              op, "failed to restore deinterleaved=2 group result type");
        results[group] = *finalResult;
      }

      replaceOpWithFlatConvertedValues(rewriter, op, results,
                                       *this->getTypeConverter());
      return success();
    }

    if (*plan != GroupReduceLoweringPlan::ContiguousVcaddRows)
      return rewriter.notifyMatchFailure(op,
                                         "unknown group_reduce lowering plan");

    int64_t lanesPerPart = 0;
    int64_t groupCount = 0;
    int64_t chunksPerGroup = 0;
    if (failed(checkContiguousFullGroupChunks(op, sourceVMIType, *groupSize,
                                              &lanesPerPart, &groupCount,
                                              &chunksPerGroup, rewriter)))
      return failure();
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    bool rowLocalSlots1Result = resultLayout && resultLayout.isGroupSlots() &&
                                resultLayout.getNumGroups() == groupCount &&
                                resultLayout.getSlots() == 1;
    int64_t expectedResultParts =
        rowLocalSlots1Result ? groupCount : groupCount * chunksPerGroup;
    if (sourceParts.size() != maskParts.size() ||
        static_cast<int64_t>(sourceParts.size()) !=
            groupCount * chunksPerGroup ||
        static_cast<int64_t>(resultTypes.size()) != expectedResultParts)
      return rewriter.notifyMatchFailure(
          op, "group_reduce requires matching source/mask/result arity");

    SmallVector<Value> results(resultTypes.size());
    for (Type resultType : resultTypes) {
      if (!isa<VRegType>(resultType))
        return rewriter.notifyMatchFailure(
            op, "group_reduce result must be vreg");
    }

    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType)
      return rewriter.notifyMatchFailure(
          op, "group_reduce requires physical vreg result and mask");

    auto sourcePartType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!sourcePartType)
      return rewriter.notifyMatchFailure(op,
                                         "group_reduce source must be vreg");
    FailureOr<VRegType> rowResultType =
        getRowResultType(sourcePartType, resultType);
    if (failed(rowResultType))
      return rewriter.notifyMatchFailure(
          op, "failed to derive group row-reduction type");
    FailureOr<MaskType> rowMaskType =
        getMaskTypeForVReg(*rowResultType, rewriter.getContext());
    if (failed(rowMaskType))
      return rewriter.notifyMatchFailure(
          op, "failed to derive group combine mask type");
    FailureOr<Value> firstLaneMask = createPrefixMask(
        op.getLoc(), *rowMaskType, "PAT_VL1", rewriter);
    if (failed(firstLaneMask))
      return rewriter.notifyMatchFailure(
          op, "failed to create group_reduce masks");

    for (int64_t group = 0; group < groupCount; ++group) {
      Value accumulator;

      int64_t firstIndex = group * chunksPerGroup;
      ValueRange groupSources = sourceParts.slice(firstIndex, chunksPerGroup);
      ValueRange groupMasks = maskParts.slice(firstIndex, chunksPerGroup);
      for (auto [source, mask] : llvm::zip_equal(groupSources, groupMasks)) {
        if (source.getType() != sourcePartType || mask.getType() != maskType)
          return rewriter.notifyMatchFailure(
              op, "group_reduce requires uniform physical chunk types");
      }
      FailureOr<Value> combined = combineEquivalentMaskedParts<CombineOpTy>(
          op.getLoc(), groupSources, groupMasks, sourcePartType, rewriter);
      if (succeeded(combined))
        accumulator =
            rewriter
                .create<RowReduceOpTy>(op.getLoc(), *rowResultType, *combined,
                                       groupMasks.front())
                .getResult();

      if (!accumulator) {
        for (int64_t chunk = 0; chunk < chunksPerGroup; ++chunk) {
          int64_t index = group * chunksPerGroup + chunk;
          if (sourceParts[index].getType() != sourcePartType ||
              maskParts[index].getType() != maskType)
            return rewriter.notifyMatchFailure(
                op, "group_reduce requires uniform physical chunk types");
          Value reduced =
              rewriter
                  .create<RowReduceOpTy>(op.getLoc(), *rowResultType,
                                         sourceParts[index], maskParts[index])
                  .getResult();
          if (!accumulator) {
            accumulator = reduced;
            continue;
          }
          accumulator =
              rewriter
                  .create<CombineOpTy>(op.getLoc(), *rowResultType, reduced,
                                       accumulator, *firstLaneMask)
                  .getResult();
        }
      }

      FailureOr<Value> finalResult =
          bitcastVReg(op.getLoc(), accumulator, resultType, rewriter);
      if (failed(finalResult))
        return rewriter.notifyMatchFailure(
            op, "failed to restore group result type");

      int64_t destChunk = rowLocalSlots1Result ? group : group * chunksPerGroup;
      if (rowLocalSlots1Result) {
        results[destChunk] = *finalResult;
      } else {
        for (int64_t chunk = 0; chunk < chunksPerGroup; ++chunk)
          results[destChunk + chunk] = *finalResult;
      }
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }

private:
  FailureOr<VRegType> getRowResultType(VRegType sourceType,
                                       VRegType resultType) const {
    if constexpr (std::is_same_v<OpTy, VMIGroupReduceAddIOp>)
      return getVcaddResultType(sourceType);
    return resultType;
  }

  LogicalResult getSupport(VMILayoutSupport &supports, VMIGroupReduceAddFOp op,
                           std::string *reason) const {
    return supports.getGroupReduceAddFSupport(op, reason);
  }

  LogicalResult getSupport(VMILayoutSupport &supports, VMIGroupReduceAddIOp op,
                           std::string *reason) const {
    return supports.getGroupReduceAddISupport(op, reason);
  }

  LogicalResult getSupport(VMILayoutSupport &supports, VMIGroupReduceMaxIOp op,
                           std::string *reason) const {
    return supports.getGroupReduceMaxISupport(op, reason);
  }

  LogicalResult getSupport(VMILayoutSupport &supports, VMIGroupReduceMaxFOp op,
                           std::string *reason) const {
    return supports.getGroupReduceMaxFSupport(op, reason);
  }

  LogicalResult getSupport(VMILayoutSupport &supports, VMIGroupReduceMinFOp op,
                           std::string *reason) const {
    return supports.getGroupReduceMinFSupport(op, reason);
  }

  LogicalResult getSupport(VMILayoutSupport &supports, VMIGroupReduceMinIOp op,
                           std::string *reason) const {
    return supports.getGroupReduceMinISupport(op, reason);
  }

  ;
};

struct OneToNVMIGroupBroadcastOpPattern
    : OpConversionPattern<VMIGroupBroadcastOp> {
  using OpConversionPattern<
      VMIGroupBroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIGroupBroadcastOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
        resultVMIType, op.getNumGroupsAttr().getInt());
    if (failed(groupSize))
      return rewriter.notifyMatchFailure(
          op,
          "group_broadcast requires num_groups to evenly divide lane count");
    int64_t lanesPerPart = 0;
    int64_t groupCount = 0;
    if (failed(checkFullGroupSlotSourceShape(
            op, sourceVMIType, *groupSize, op.getNumGroupsAttr().getInt(),
            &lanesPerPart, &groupCount, rewriter)))
      return failure();
    int64_t resultLayoutFactor = 0;
    int64_t resultGroupCount = 0;
    if (failed(checkFullGroupBroadcastResultShape(
            op, resultVMIType, *groupSize, lanesPerPart, &resultLayoutFactor,
            &resultGroupCount, rewriter)))
      return failure();
    if (resultGroupCount != groupCount)
      return rewriter.notifyMatchFailure(
          op, "group_broadcast requires matching source/result group slots");

    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (sourceParts.empty() || resultTypes.empty())
      return rewriter.notifyMatchFailure(op, "group_broadcast arity mismatch");

    auto firstSourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!firstSourceType)
      return rewriter.notifyMatchFailure(op,
                                         "group_broadcast source must be vreg");
    unsigned indexBits =
        pto::getPTOStorageElemBitWidth(firstSourceType.getElementType());
    if (indexBits != 8 && indexBits != 16 && indexBits != 32)
      return rewriter.notifyMatchFailure(
          op, "group_broadcast requires 8/16/32-bit index elements");
    auto indexElementType = IntegerType::get(rewriter.getContext(), indexBits);
    auto indexType =
        VRegType::get(rewriter.getContext(), firstSourceType.getElementCount(),
                      indexElementType);
    FailureOr<Value> allMask =
        createAllTrueMaskForVReg(op.getLoc(), firstSourceType, rewriter);
    if (failed(allMask))
      return rewriter.notifyMatchFailure(
          op, "failed to create group_broadcast all mask");
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    VMILayoutAttr sourceLayout = sourceVMIType.getLayoutAttr();
    FailureOr<int64_t> resultBlockElems =
        getVMILayoutBlockElems(resultVMIType);
    if (failed(resultBlockElems))
      return rewriter.notifyMatchFailure(
          op, "group_broadcast requires a computable result block size");
    int64_t selectionGroupSize = *groupSize;
    if (resultLayoutFactor != 1 && resultLayout &&
        resultLayout.isBlockDeinterleaved() &&
        *groupSize < lanesPerPart)
      selectionGroupSize = *resultBlockElems;
    auto resolveLargeGroupSource = [&](int64_t group, int64_t chunksPerGroup,
                                       int64_t &sourceChunk,
                                       int64_t &baseGroupSlot) {
      int64_t slots = sourceLayout.getSlots();
      if (slots > 0) {
        sourceChunk = group / slots;
        baseGroupSlot = group % slots;
        return;
      }
      sourceChunk = group * chunksPerGroup;
      baseGroupSlot = 0;
    };

    SmallVector<Value> results;
    results.resize(resultTypes.size());
    for (auto [flatIndex, resultType] : llvm::enumerate(resultTypes)) {
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      if (!resultVRegType || resultVRegType != firstSourceType)
        return rewriter.notifyMatchFailure(
            op, "group_broadcast requires uniform physical vreg types");
      int64_t sourceChunk = flatIndex;
      int64_t baseGroupSlot = 0;
      Value mappedGroupSlotIndex;
      if (resultLayoutFactor == 1) {
        bool laneStridedDense =
            resultLayout && resultLayout.isDense() &&
            resultLayout.getLaneStride() > 1;
        if (laneStridedDense) {
          VMILayoutAttr sourceLayout = sourceVMIType.getLayoutAttr();
          int64_t slots = sourceLayout.getSlots();
          if (slots <= 0) {
            if (sourceParts.empty() ||
                groupCount % static_cast<int64_t>(sourceParts.size()) != 0)
              return rewriter.notifyMatchFailure(
                  op, "group_broadcast lane-stride source requires explicit "
                      "group_slots slots or derivable legacy slot count");
            slots = groupCount / sourceParts.size();
          }
          FailureOr<Value> index = createMappedGroupSlotIndexVector(
              op.getLoc(), resultVMIType, /*part=*/0, flatIndex, indexType,
              *groupSize, slots, sourceChunk, rewriter,
              sourceLayout.getLaneStride());
          if (failed(index))
            return rewriter.notifyMatchFailure(
                op, "failed to create group_broadcast lane-stride group-slot "
                    "index vector");
          mappedGroupSlotIndex = *index;
        } else if (*groupSize >= lanesPerPart) {
          int64_t chunksPerGroup = *groupSize / lanesPerPart;
          int64_t group = flatIndex / chunksPerGroup;
          resolveLargeGroupSource(group, chunksPerGroup, sourceChunk,
                                  baseGroupSlot);
        } else {
          VMILayoutAttr sourceLayout = sourceVMIType.getLayoutAttr();
          int64_t slots = sourceLayout.getSlots();
          if (slots <= 0) {
            if (sourceParts.empty() ||
                groupCount % static_cast<int64_t>(sourceParts.size()) != 0)
              return rewriter.notifyMatchFailure(
                  op, "group_broadcast small-group source requires explicit "
                      "group_slots slots or derivable legacy slot count");
            slots = groupCount / sourceParts.size();
          }
          int64_t groupsPerResultChunk = lanesPerPart / *groupSize;
          int64_t firstGroup = flatIndex * groupsPerResultChunk;
          sourceChunk = firstGroup / slots;
          baseGroupSlot = firstGroup % slots;
        }
      } else {
        bool blockFragmentSmallGroup =
            resultLayout && resultLayout.isBlockDeinterleaved() &&
            *groupSize < lanesPerPart;
        bool deinterleavedSmallGroup =
            resultLayout && resultLayout.isDeinterleaved() &&
            *groupSize < lanesPerPart;
        if (blockFragmentSmallGroup) {
          int64_t runningFlatIndex = 0;
          bool found = false;
          for (int64_t part = 0; part < resultLayoutFactor && !found; ++part) {
            FailureOr<int64_t> chunks =
                getDataChunksInPart(resultVMIType, part);
            if (failed(chunks))
              return rewriter.notifyMatchFailure(
                  op, "group_broadcast failed to enumerate result chunks");
            for (int64_t chunk = 0; chunk < *chunks;
                 ++chunk, ++runningFlatIndex) {
              if (runningFlatIndex != static_cast<int64_t>(flatIndex))
                continue;
              int64_t groupsPerResultChunk =
                  lanesPerPart / *resultBlockElems;
              int64_t firstGroup = chunk * groupsPerResultChunk;
              int64_t slots = sourceLayout.getSlots();
              if (slots <= 0) {
                if (sourceParts.empty() ||
                    groupCount % static_cast<int64_t>(sourceParts.size()) != 0)
                  return rewriter.notifyMatchFailure(
                      op,
                      "group_broadcast block-fragment source requires explicit "
                      "group_slots slots or derivable legacy slot count");
                slots = groupCount / sourceParts.size();
              }
              sourceChunk = firstGroup / slots;
              baseGroupSlot = firstGroup % slots;
              found = true;
              break;
            }
          }
          if (!found)
            return rewriter.notifyMatchFailure(
                op, "group_broadcast result chunk index is out of range");
        } else if (deinterleavedSmallGroup) {
          int64_t runningFlatIndex = 0;
          bool found = false;
          for (int64_t part = 0; part < resultLayoutFactor && !found; ++part) {
            FailureOr<int64_t> chunks =
                getDataChunksInPart(resultVMIType, part);
            if (failed(chunks))
              return rewriter.notifyMatchFailure(
                  op, "group_broadcast failed to enumerate result chunks");
            for (int64_t chunk = 0; chunk < *chunks;
                 ++chunk, ++runningFlatIndex) {
              if (runningFlatIndex != static_cast<int64_t>(flatIndex))
                continue;
              int64_t slots = sourceLayout.getSlots();
              if (slots <= 0) {
                if (sourceParts.empty() ||
                    groupCount % static_cast<int64_t>(sourceParts.size()) != 0)
                  return rewriter.notifyMatchFailure(
                      op, "group_broadcast deinterleaved small-group source "
                          "requires explicit group_slots slots or derivable "
                          "legacy slot count");
                slots = groupCount / sourceParts.size();
              }
              FailureOr<Value> index = createMappedGroupSlotIndexVector(
                  op.getLoc(), resultVMIType, part, chunk, indexType,
                  *groupSize, slots, sourceChunk, rewriter,
                  sourceLayout.getLaneStride());
              if (failed(index))
                return rewriter.notifyMatchFailure(
                    op,
                    "failed to create group_broadcast mapped group-slot index "
                    "vector");
              mappedGroupSlotIndex = *index;
              found = true;
              break;
            }
          }
          if (!found)
            return rewriter.notifyMatchFailure(
                op, "group_broadcast result chunk index is out of range");
        } else {
          int64_t runningFlatIndex = 0;
          bool found = false;
          for (int64_t part = 0; part < resultLayoutFactor && !found; ++part) {
            FailureOr<int64_t> chunks =
                getDataChunksInPart(resultVMIType, part);
            if (failed(chunks))
              return rewriter.notifyMatchFailure(
                  op, "group_broadcast failed to enumerate result chunks");
            for (int64_t chunk = 0; chunk < *chunks;
                 ++chunk, ++runningFlatIndex) {
              if (runningFlatIndex != static_cast<int64_t>(flatIndex))
                continue;
              FailureOr<int64_t> firstLogical =
                  mapPhysicalLaneToLogical(resultVMIType, part, chunk, 0);
              FailureOr<int64_t> lastLogical = mapPhysicalLaneToLogical(
                  resultVMIType, part, chunk, lanesPerPart - 1);
              if (failed(firstLogical) || failed(lastLogical))
                return rewriter.notifyMatchFailure(
                    op, "group_broadcast failed to map result chunk lanes");
              int64_t firstGroup = *firstLogical / *groupSize;
              int64_t lastGroup = *lastLogical / *groupSize;
              if (firstGroup != lastGroup)
                return rewriter.notifyMatchFailure(
                    op, "group_broadcast result chunk crosses logical groups");
              int64_t chunksPerGroup = *groupSize / lanesPerPart;
              resolveLargeGroupSource(firstGroup, chunksPerGroup, sourceChunk,
                                      baseGroupSlot);
              found = true;
              break;
            }
          }
          if (!found)
            return rewriter.notifyMatchFailure(
                op, "group_broadcast result chunk index is out of range");
        }
      }
      if (*groupSize >= lanesPerPart) {
        if (sourceChunk < 0 ||
            sourceChunk >= static_cast<int64_t>(sourceParts.size()))
          return rewriter.notifyMatchFailure(
              op, "group_broadcast source chunk is out of range");
        if (sourceLayout.getSlots() > 1) {
          FailureOr<Value> groupSlotIndex = createGroupSlotIndexVector(
              op.getLoc(), indexType, selectionGroupSize, baseGroupSlot,
              rewriter, sourceLayout.getLaneStride());
          if (failed(groupSlotIndex))
            return rewriter.notifyMatchFailure(
                op, "failed to create group_broadcast group-slot index vector");
          results[flatIndex] =
              rewriter
                  .create<VselrOp>(op.getLoc(), resultType,
                                   sourceParts[sourceChunk], *groupSlotIndex)
                  .getResult();
        } else {
          results[flatIndex] =
              rewriter
                  .create<VdupOp>(op.getLoc(), resultType,
                                  sourceParts[sourceChunk], *allMask,
                                  rewriter.getStringAttr("LOWEST"))
                  .getResult();
        }
      } else {
        bool blockFragmentSmallGroup =
            resultLayout && resultLayout.isBlockDeinterleaved();
        bool deinterleavedSmallGroup = resultLayout &&
                                       resultLayout.isDeinterleaved();
        if (resultLayoutFactor != 1 && !blockFragmentSmallGroup &&
            !deinterleavedSmallGroup)
          return rewriter.notifyMatchFailure(
              op, "group_broadcast small-group deinterleaved result is not "
                  "supported");
        if (sourceChunk < 0 ||
            sourceChunk >= static_cast<int64_t>(sourceParts.size()))
          return rewriter.notifyMatchFailure(
              op, "group_broadcast source chunk is out of range");
        FailureOr<Value> groupSlotIndex =
            mappedGroupSlotIndex
                ? FailureOr<Value>(mappedGroupSlotIndex)
                : createGroupSlotIndexVector(op.getLoc(), indexType,
                                             selectionGroupSize, baseGroupSlot,
                                             rewriter,
                                             sourceLayout.getLaneStride());
        if (failed(groupSlotIndex))
          return rewriter.notifyMatchFailure(
              op, "failed to create group_broadcast group-slot index vector");
        results[flatIndex] =
            rewriter
                .create<VselrOp>(op.getLoc(), resultType,
                                 sourceParts[sourceChunk], *groupSlotIndex)
                .getResult();
      }
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// VMI vdhist / vchist → VPTO dhistv2 / chistv2 lowering (shared template)
//===----------------------------------------------------------------------===//

template <typename VMIOp, typename VPTOHistOp>
static LogicalResult
lowerVMIHistogramToVPTO(VMIOp op,
                        typename OpConversionPattern<VMIOp>::OneToNOpAdaptor
                            adaptor,
                        const TypeConverter *typeConverter,
                        ConversionPatternRewriter &rewriter) {
  ValueRange accParts    = adaptor.getAcc();
  ValueRange sourceParts = adaptor.getSource();
  ValueRange maskParts   = adaptor.getMask();

  // Allow 1 (Bin_N0-only, 128×ui16) or 2 (Bin_N0+Bin_N1, 256×ui16) halves
  size_t halfCount = accParts.size();
  if (halfCount != 1 && halfCount != 2)
    return rewriter.notifyMatchFailure(
        op, "expected one or two accumulator parts");
  if (sourceParts.empty() || sourceParts.size() != maskParts.size())
    return rewriter.notifyMatchFailure(
        op, "expected matching source/mask chunks");

  auto partType = dyn_cast<VRegType>(accParts[0].getType());
  if (!partType)
    return rewriter.notifyMatchFailure(op, "expected ui16 acc parts");
  if (halfCount == 2 && accParts[1].getType() != partType)
    return rewriter.notifyMatchFailure(op,
                                       "expected matching ui16 acc parts");

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(sourceType.getElementType());
  if (failed(lanesPerPart))
    return rewriter.notifyMatchFailure(op, "failed to compute source lanes");

  Location loc = op.getLoc();
  SmallVector<Value, 2> binConsts;
  binConsts.push_back(createI32Constant(loc, 0, rewriter));
  if (halfCount == 2)
    binConsts.push_back(createI32Constant(loc, 1, rewriter));

  SmallVector<Value, 2> halves(accParts.begin(), accParts.end());

  for (size_t index = 0, e = sourceParts.size(); index < e; ++index) {
    Value source   = sourceParts[index];
    Value userMask = maskParts[index];
    auto maskType  = dyn_cast<MaskType>(userMask.getType());
    if (!maskType || !maskType.isB8())
      return rewriter.notifyMatchFailure(op, "expected b8 source mask");

    Value chunkMask = userMask;
    int64_t firstLane   = int64_t(index) * *lanesPerPart;
    int64_t activeLanes = std::min<int64_t>(
        *lanesPerPart, sourceType.getElementCount() - firstLane);
    if (activeLanes < *lanesPerPart) {
      FailureOr<Value> validMask = createPrefixMaskForActiveLanes(
          loc, maskType, activeLanes, rewriter);
      FailureOr<Value> allMask = createAllTrueMask(loc, maskType, rewriter);
      if (failed(validMask) || failed(allMask))
        return rewriter.notifyMatchFailure(
            op, "failed to materialize tail-valid b8 mask");
      chunkMask =
          rewriter
              .create<PandOp>(loc, maskType, chunkMask, *validMask, *allMask)
              .getResult();
    }

    for (size_t h = 0; h < halfCount; ++h)
      halves[h] = rewriter.create<VPTOHistOp>(loc, partType, halves[h],
                                               source, chunkMask, binConsts[h])
                      .getResult();
  }

  replaceOpWithFlatConvertedValues(
      rewriter, op,
      SmallVector<Value>(halves.begin(), halves.begin() + halfCount),
      *typeConverter);
  return success();
}

struct OneToNVMIVdhistOpPattern : OpConversionPattern<VMIVdhistOp> {
  using OpConversionPattern<VMIVdhistOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIVdhistOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerVMIHistogramToVPTO<VMIVdhistOp, Dhistv2Op>(
        op, adaptor, this->getTypeConverter(), rewriter);
  }
};

struct OneToNVMIVchistOpPattern : OpConversionPattern<VMIVchistOp> {
  using OpConversionPattern<VMIVchistOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIVchistOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerVMIHistogramToVPTO<VMIVchistOp, Chistv2Op>(
        op, adaptor, this->getTypeConverter(), rewriter);
  }
};

template <typename SourceOp, typename ChunkReduceOp, typename CombineOp>
struct OneToNVMIReduceMinMaxOpPattern : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OpConversionPattern<SourceOp>::OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    ValueRange initParts = adaptor.getInit();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (sourceParts.empty() || sourceParts.size() != maskParts.size() ||
        initParts.size() != 1 || resultTypes.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "min/max reduction requires matching source/mask chunks "
              "and one init/result chunk");

    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType || initParts.front().getType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "min/max reduction requires matching physical source/"
              "init/result vregs and one mask");

    for (Value sourcePart : sourceParts)
      if (sourcePart.getType() != resultType)
        return rewriter.notifyMatchFailure(
            op, "min/max reduction requires every source chunk to "
                "match result vreg type");
    for (Value maskPart : maskParts)
      if (maskPart.getType() != maskType)
        return rewriter.notifyMatchFailure(
            op, "min/max reduction requires every mask chunk to have "
                "the same predicate type");

    FailureOr<Value> firstLaneMask =
        createPrefixMask(op.getLoc(), maskType, "PAT_VL1", rewriter);
    if (failed(firstLaneMask))
      return rewriter.notifyMatchFailure(
          op, "failed to create min/max reduction first-lane mask");

    FailureOr<Value> combined = combineEquivalentMaskedParts<CombineOp>(
        op.getLoc(), sourceParts, maskParts, resultType, rewriter);
    if (succeeded(combined)) {
      Value reduced =
          rewriter
              .create<ChunkReduceOp>(op.getLoc(), resultType, *combined,
                                     maskParts.front())
              .getResult();
      Value result =
          rewriter
              .create<CombineOp>(op.getLoc(), resultType, reduced,
                                 initParts.front(), *firstLaneMask)
              .getResult();
      replaceOpWithFlatConvertedValues(
          rewriter, op, SmallVector<Value>{result}, *this->getTypeConverter());
      return success();
    }

    Value accumulator = initParts.front();
    for (auto [sourcePart, maskPart] :
         llvm::zip_equal(sourceParts, maskParts)) {
      Value reduced = rewriter
                          .create<ChunkReduceOp>(op.getLoc(), resultType,
                                                 sourcePart, maskPart)
                          .getResult();
      accumulator = rewriter
                        .create<CombineOp>(op.getLoc(), resultType, reduced,
                                           accumulator, *firstLaneMask)
                        .getResult();
    }

    replaceOpWithFlatConvertedValues(
            rewriter, op, SmallVector<Value>{accumulator},
            *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIExtFOpPattern : OpConversionPattern<VMIExtFOp> {
  using OpConversionPattern<VMIExtFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIExtFOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (sourceParts.empty())
      return rewriter.notifyMatchFailure(
          op, "extf requires at least one physical source chunk");

    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!sourceType)
      return rewriter.notifyMatchFailure(op, "expected physical extf source");
    for (Value sourcePart : sourceParts) {
      auto currentSourceType = dyn_cast<VRegType>(sourcePart.getType());
      if (!currentSourceType || currentSourceType != sourceType)
        return rewriter.notifyMatchFailure(
            op, "extf source physical parts must have matching type");
    }

    SmallVector<VRegType> resultVRegTypes;
    resultVRegTypes.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      if (!resultVRegType ||
          (resultVRegTypes.empty() ? !resultVRegType.getElementType().isF32()
                                   : resultVRegType != resultVRegTypes.front()))
        return rewriter.notifyMatchFailure(
            op, "unsupported physical extf result type");
      resultVRegTypes.push_back(resultVRegType);
    }

    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(sourceType.getElementType());
    VMILayoutAttr sourceLayout = sourceVMIType.getLayoutAttr();
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    if (sourceLayout && resultLayout && sourceLayout.isContiguous() &&
        resultLayout.isContiguous() && resultLayout.getLaneStride() == 1 &&
        ((sourceBits == 16 && sourceLayout.getLaneStride() == 2) ||
         (sourceBits == 8 && sourceLayout.getLaneStride() == 4)) &&
        resultTypes.size() == sourceParts.size()) {
      StringRef part = sourceBits == 16 ? StringRef("EVEN") : StringRef("P0");
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(op,
                                           "failed to build extf seed mask");

      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      for (auto [sourcePart, resultType] :
           llvm::zip_equal(sourceParts, resultVRegTypes)) {
        results.push_back(rewriter
                              .create<VcvtOp>(op.getLoc(), resultType,
                                              sourcePart, *mask,
                                              /*rnd=*/nullptr, /*sat=*/nullptr,
                                              rewriter.getStringAttr(part))
                              .getResult());
      }
      replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
      return success();
    }

    ArrayRef<StringRef> parts;
    int64_t factor = 0;
    if (sourceBits == 16 && resultTypes.size() == 2 * sourceParts.size()) {
      static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
      parts = kEvenOddParts;
      factor = 2;
    } else if (sourceBits == 8 &&
               resultTypes.size() == 4 * sourceParts.size()) {
      static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
      parts = kPacked4Parts;
      factor = 4;
    } else {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical extf source/result width relation");
    }

    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
    if (failed(mask))
      return rewriter.notifyMatchFailure(op, "failed to build extf seed mask");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (int64_t partIndex = 0; partIndex < factor; ++partIndex) {
      for (auto [chunkIndex, sourcePart] : llvm::enumerate(sourceParts)) {
        VRegType resultType =
            resultVRegTypes[partIndex * sourceParts.size() + chunkIndex];
        results.push_back(
            rewriter
                .create<VcvtOp>(op.getLoc(), resultType, sourcePart, *mask,
                                /*rnd=*/nullptr, /*sat=*/nullptr,
                                rewriter.getStringAttr(parts[partIndex]))
                .getResult());
      }
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMITruncFOpPattern : OpConversionPattern<VMITruncFOp> {
  using OpConversionPattern<VMITruncFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMITruncFOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);

    VMILayoutAttr sourceLayout = sourceVMIType.getLayoutAttr();
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    if (sourceLayout && resultLayout && sourceLayout.isGroupSlots() &&
        resultLayout.isGroupSlots()) {
      unsigned logicalResultBits =
          pto::getPTOStorageElemBitWidth(resultVMIType.getElementType());
      if (sourceLayout.getNumGroups() != resultLayout.getNumGroups() ||
          sourceLayout.getSlots() != resultLayout.getSlots() ||
          (sourceLayout.getSlots() != 1 && sourceLayout.getSlots() != 8) ||
          !sourceVMIType.getElementType().isF32() ||
          (logicalResultBits != 16 && logicalResultBits != 8) ||
          sourceParts.size() != resultTypes.size())
        return rewriter.notifyMatchFailure(
            op, "unsupported group-slot truncf shape");

      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      const char *activeSlotPattern =
          sourceLayout.getSlots() == 1 ? "PAT_VL1" : "PAT_VL8";
      FailureOr<Value> activeSlotMask = createPrefixMask(
          op.getLoc(), MaskType::get(rewriter.getContext(), "b32"),
          activeSlotPattern, rewriter);
      if (failed(activeSlotMask))
        return rewriter.notifyMatchFailure(
            op, "failed to build group-slot truncf active slot mask");
      StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
      for (auto [sourcePart, physicalResultType] :
           llvm::zip_equal(sourceParts, resultTypes)) {
        auto sourceType = dyn_cast<VRegType>(sourcePart.getType());
        auto resultType = dyn_cast<VRegType>(physicalResultType);
        if (!sourceType || !sourceType.getElementType().isF32() ||
            !resultType)
          return rewriter.notifyMatchFailure(
              op, "unsupported group-slot truncf physical type");
        unsigned physicalResultBits =
            pto::getPTOStorageElemBitWidth(resultType.getElementType());
        StringAttr part;
        if (physicalResultBits == 16) {
          part = rewriter.getStringAttr("EVEN");
        } else if (physicalResultBits == 8) {
          part = rewriter.getStringAttr("P0");
        } else {
          return rewriter.notifyMatchFailure(
              op, "unsupported group-slot truncf physical type");
        }
        StringAttr rnd = rewriter.getStringAttr(
            getTruncFRoundMode(op, resultType.getElementType()));
        results.push_back(rewriter
                              .create<VcvtOp>(op.getLoc(), resultType,
                                              sourcePart, *activeSlotMask, rnd,
                                              sat, part)
                              .getResult());
      }
      replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
      return success();
    }

    if (resultTypes.empty())
      return rewriter.notifyMatchFailure(op, "truncf requires result chunks");

    auto sourceType0 = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!sourceType0 || !isa<FloatType>(sourceType0.getElementType()))
      return rewriter.notifyMatchFailure(op, "unsupported physical truncf source type");
    unsigned sourceBits = pto::getPTOStorageElemBitWidth(sourceType0.getElementType());
    if (sourceBits != 32 && sourceBits != 16)
      return rewriter.notifyMatchFailure(
          op, "truncf source bit width must be 32 or 16");
    // Group-slot layout for non-f32 sources is not supported yet.
    if (sourceLayout && sourceLayout.isGroupSlots())
      return rewriter.notifyMatchFailure(
          op, "group-slot layout for non-f32 truncf not supported");
    for (Value sourcePart : sourceParts) {
      auto sourceType = dyn_cast<VRegType>(sourcePart.getType());
      if (!sourceType || sourceType != sourceType0)
        return rewriter.notifyMatchFailure(
            op, "truncf source physical parts must have matching type");
    }

    SmallVector<VRegType> resultVRegTypes;
    resultVRegTypes.reserve(resultTypes.size());
    for (Type physicalResultType : resultTypes) {
      auto resultType = dyn_cast<VRegType>(physicalResultType);
      if (!resultType ||
          (resultVRegTypes.empty() ? pto::getPTOStorageElemBitWidth(
                                         resultType.getElementType()) == 0
                                   : resultType != resultVRegTypes.front()))
        return rewriter.notifyMatchFailure(
            op, "unsupported physical truncf result type");
      resultVRegTypes.push_back(resultType);
    }

    unsigned resultBits = pto::getPTOStorageElemBitWidth(
        resultVRegTypes.front().getElementType());
    if (sourceLayout && resultLayout && sourceLayout.isContiguous() &&
        sourceLayout.getLaneStride() == 1 && resultLayout.isContiguous() &&
        resultLayout.getLaneStride() != 1 &&
        sourceParts.size() == resultTypes.size()) {
      StringRef part;
      if (resultBits == 16 && resultLayout.getLaneStride() == 2)
        part = "EVEN";                                          // 32→16
      else if (resultBits == 8 && resultLayout.getLaneStride() == 4)
        part = "P0";                                            // 32→8 (f8/hif8)
      else if (resultBits == 8 && resultLayout.getLaneStride() == 2)
        part = "EVEN";                                          // 16→8
      else
        return rewriter.notifyMatchFailure(
            op, "unsupported dense lane_stride truncf result layout");

      FailureOr<Value> sourceMask =
          createAllTrueMaskForVReg(op.getLoc(), sourceType0, rewriter);
      if (failed(sourceMask))
        return rewriter.notifyMatchFailure(op, "failed to build truncf masks");

      StringAttr rnd = rewriter.getStringAttr(
          getTruncFRoundMode(op, resultVRegTypes.front().getElementType()));
      StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
      StringAttr partAttr = rewriter.getStringAttr(part);
      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      for (auto [sourcePart, resultType] :
           llvm::zip_equal(sourceParts, resultVRegTypes)) {
        results.push_back(rewriter
                              .create<VcvtOp>(op.getLoc(), resultType,
                                              sourcePart, *sourceMask, rnd, sat,
                                              partAttr)
                              .getResult());
      }
      replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
      return success();
    }

    ArrayRef<StringRef> allParts;
    int64_t factor = 0;
    if (resultBits * 2 == sourceBits) {
      // 32→16 or 16→8: EvenOdd.
      static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
      allParts = kEvenOddParts;
      factor = 2;
    } else if (resultBits * 4 == sourceBits) {
      // 32→8: Packed4.
      static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
      allParts = kPacked4Parts;
      factor = 4;
    } else {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical truncf source/result width relation");
    }

    int64_t resultLaneStride = resultLayout && resultLayout.isContiguous()
                                   ? resultLayout.getLaneStride()
                                   : 1;
    if (resultLaneStride <= 0 || factor % resultLaneStride != 0)
      return rewriter.notifyMatchFailure(
          op, "unsupported physical truncf result lane stride");
    int64_t sourceFactor = factor / resultLaneStride;
    if (sourceParts.size() != sourceFactor * resultTypes.size())
      return rewriter.notifyMatchFailure(
          op, "unsupported physical truncf source/result arity relation");

    FailureOr<Value> sourceMask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType0, rewriter);
    if (failed(sourceMask))
      return rewriter.notifyMatchFailure(op, "failed to build truncf masks");

    StringAttr rnd = rewriter.getStringAttr(
        getTruncFRoundMode(op, resultVRegTypes.front().getElementType()));
    StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [chunkIndex, resultType] : llvm::enumerate(resultVRegTypes)) {
      FailureOr<Value> resultMask =
          createAllTrueMaskForVReg(op.getLoc(), resultType, rewriter);
      if (failed(resultMask))
        return rewriter.notifyMatchFailure(
            op, "failed to build truncf result mask");

      SmallVector<Value> partials;
      partials.reserve(sourceFactor);
      for (int64_t partIndex = 0; partIndex < sourceFactor; ++partIndex) {
        Value sourcePart =
            sourceParts[partIndex * resultTypes.size() + chunkIndex];
        partials.push_back(
            rewriter
                .create<VcvtOp>(op.getLoc(), resultType, sourcePart,
                                *sourceMask, rnd, sat,
                                rewriter.getStringAttr(
                                    allParts[partIndex * resultLaneStride]))
                .getResult());
      }

      Value merged = partials.front();
      for (Value partial : llvm::drop_begin(partials))
        merged = rewriter
                     .create<VorOp>(op.getLoc(), resultType, merged, partial,
                                    *resultMask)
                     .getResult();
      results.push_back(merged);
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

template <typename OpT>
struct OneToNVMIExtIOpPattern : OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op,
                  typename OpConversionPattern<OpT>::OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (sourceParts.empty())
      return rewriter.notifyMatchFailure(
          op, "integer extension requires at least one physical source chunk");

    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!sourceType)
      return rewriter.notifyMatchFailure(
          op, "expected physical integer extension source");
    for (Value sourcePart : sourceParts) {
      auto currentSourceType = dyn_cast<VRegType>(sourcePart.getType());
      if (!currentSourceType || currentSourceType != sourceType)
        return rewriter.notifyMatchFailure(
            op, "integer extension source physical parts must have matching "
                "type");
    }

    VMILayoutAttr sourceLayout = sourceVMIType.getLayoutAttr();
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    if (sourceLayout && resultLayout && sourceLayout.isGroupSlots() &&
        resultLayout.isGroupSlots()) {
      if (sourceLayout.getNumGroups() != resultLayout.getNumGroups() ||
          sourceLayout.getSlots() != 8 || resultLayout.getSlots() != 8 ||
          sourceParts.size() != resultTypes.size())
        return rewriter.notifyMatchFailure(
            op, "unsupported group-slot integer extension shape");

      unsigned sourceBits =
          pto::getPTOStorageElemBitWidth(sourceVMIType.getElementType());
      unsigned resultBits =
          pto::getPTOStorageElemBitWidth(resultVMIType.getElementType());
      if ((sourceBits != 8 && sourceBits != 16) || resultBits != 32)
        return rewriter.notifyMatchFailure(
            op, "group-slot integer extension requires 8/16-bit source and "
                "32-bit result element widths");

      FailureOr<MaskType> maskType =
          getMaskTypeForVReg(sourceType, rewriter.getContext());
      if (failed(maskType))
        return rewriter.notifyMatchFailure(
            op, "failed to create group-slot integer extension mask type");
      FailureOr<Value> slotMask = createPrefixMaskForActiveLanes(
          op.getLoc(), *maskType, sourceLayout.getSlots(), rewriter);
      if (failed(slotMask))
        return rewriter.notifyMatchFailure(
            op, "failed to build group-slot integer extension mask");

      SmallVector<StringRef, 4> partNames;
      int64_t partFactor = 0;
      if (sourceBits == 16) {
        partNames.assign({"EVEN", "ODD"});
        partFactor = 2;
      } else {
        partNames.assign({"P0", "P1", "P2", "P3"});
        partFactor = 4;
      }

      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      for (auto [chunkIndex, sourcePart, resultType] :
           llvm::enumerate(sourceParts, resultTypes)) {
        auto resultVRegType = dyn_cast<VRegType>(resultType);
        if (!resultVRegType || pto::getPTOStorageElemBitWidth(
                                   resultVRegType.getElementType()) != 32)
          return rewriter.notifyMatchFailure(
              op, "unsupported group-slot integer extension result type");

        SmallVector<Value, 4> convertedParts;
        convertedParts.reserve(partNames.size());
        for (StringRef partName : partNames) {
          convertedParts.push_back(
              rewriter
                  .create<VcvtOp>(op.getLoc(), resultVRegType, sourcePart,
                                  *slotMask, /*rnd=*/nullptr, /*sat=*/nullptr,
                                  rewriter.getStringAttr(partName))
                  .getResult());
        }

        FailureOr<MaskType> resultMaskType =
            getMaskTypeForVReg(resultVRegType, rewriter.getContext());
        FailureOr<Value> resultAllMask =
            createAllTrueMaskForVReg(op.getLoc(), resultVRegType, rewriter);
        if (failed(resultMaskType) || failed(resultAllMask))
          return rewriter.notifyMatchFailure(
              op, "failed to build group-slot integer extension result seed");

        auto indexType = VRegType::get(
            rewriter.getContext(), resultVRegType.getElementCount(),
            IntegerType::get(rewriter.getContext(), 32));
        int64_t groupBegin =
            static_cast<int64_t>(chunkIndex) * sourceLayout.getSlots();
        int64_t activeSlots = std::min<int64_t>(
            sourceLayout.getSlots(), sourceLayout.getNumGroups() - groupBegin);
        if (activeSlots <= 0)
          return rewriter.notifyMatchFailure(
              op, "group-slot integer extension has no active slots");
        Value assembled;
        for (int64_t slot = 0; slot < activeSlots; ++slot) {
          int64_t partIndex = slot % partFactor;
          int64_t sourceLane = slot / partFactor;
          FailureOr<Value> laneIndexScalar = createScalarOffsetConstant(
              op.getLoc(), indexType.getElementType(), sourceLane, rewriter);
          FailureOr<Value> laneMask = createLaneRangeMask(
              op.getLoc(), *resultMaskType, slot, slot + 1, rewriter);
          if (failed(laneIndexScalar) || failed(laneMask))
            return rewriter.notifyMatchFailure(
                op, "failed to build group-slot integer extension slot mask");
          Value laneIndex =
              rewriter
                  .create<VdupOp>(op.getLoc(), indexType, *laneIndexScalar,
                                  *resultAllMask, /*position=*/nullptr)
                  .getResult();
          Value selected =
              rewriter
                  .create<VselrOp>(op.getLoc(), resultVRegType,
                                   convertedParts[partIndex], laneIndex)
                  .getResult();
          if (!assembled) {
            assembled = selected;
            continue;
          }
          assembled = rewriter
                          .create<VselOp>(op.getLoc(), resultVRegType, selected,
                                          assembled, *laneMask)
                          .getResult();
        }

        results.push_back(assembled);
      }

      replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
      return success();
    }

    SmallVector<VRegType> resultVRegTypes;
    resultVRegTypes.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      if (!resultVRegType ||
          !isa<IntegerType>(resultVRegType.getElementType()) ||
          (!resultVRegTypes.empty() &&
           resultVRegType != resultVRegTypes.front()))
        return rewriter.notifyMatchFailure(
            op, "unsupported physical integer extension result type");
      resultVRegTypes.push_back(resultVRegType);
    }

    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(sourceType.getElementType());
    unsigned resultBits = pto::getPTOStorageElemBitWidth(
        resultVRegTypes.front().getElementType());
    if (sourceLayout && resultLayout && sourceLayout.isContiguous() &&
        resultLayout.isContiguous() && resultLayout.getLaneStride() == 1 &&
        ((resultBits == sourceBits * 2 && sourceLayout.getLaneStride() == 2) ||
         (resultBits == sourceBits * 4 && sourceLayout.getLaneStride() == 4)) &&
        resultTypes.size() == sourceParts.size()) {
      StringRef part =
          resultBits == sourceBits * 2 ? StringRef("EVEN") : StringRef("P0");
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(
            op, "failed to build integer extension seed mask");

      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      for (auto [sourcePart, resultType] :
           llvm::zip_equal(sourceParts, resultVRegTypes)) {
        results.push_back(rewriter
                              .create<VcvtOp>(op.getLoc(), resultType,
                                              sourcePart, *mask,
                                              /*rnd=*/nullptr, /*sat=*/nullptr,
                                              rewriter.getStringAttr(part))
                              .getResult());
      }
      replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
      return success();
    }

    ArrayRef<StringRef> parts;
    int64_t factor = 0;
    if (resultBits == sourceBits * 2 &&
        resultTypes.size() == 2 * sourceParts.size()) {
      static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
      parts = kEvenOddParts;
      factor = 2;
    } else if (resultBits == sourceBits * 4 &&
               resultTypes.size() == 4 * sourceParts.size()) {
      static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
      parts = kPacked4Parts;
      factor = 4;
    } else {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical integer extension source/result width "
              "relation");
    }

    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
    if (failed(mask))
      return rewriter.notifyMatchFailure(
          op, "failed to build integer extension seed mask");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (int64_t partIndex = 0; partIndex < factor; ++partIndex) {
      for (auto [chunkIndex, sourcePart] : llvm::enumerate(sourceParts)) {
        VRegType resultType =
            resultVRegTypes[partIndex * sourceParts.size() + chunkIndex];
        results.push_back(
            rewriter
                .create<VcvtOp>(op.getLoc(), resultType, sourcePart, *mask,
                                /*rnd=*/nullptr, /*sat=*/nullptr,
                                rewriter.getStringAttr(parts[partIndex]))
                .getResult());
      }
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

// TruncI lowering support matrix
//
// Keep this comment aligned with both:
//   1. verifySupportedVMIToVPTOOps() diagnostics below, and
//   2. the actual OneToN lowering implemented in this pattern.
//
// Dense logical layouts
//   - deinterleaved factor 2/4 -> contiguous
//     Example: 32 -> 16 or 32 -> 8.
//     Lowering shape: emit vcvt parts EVEN/ODD or P0/P1/P2/P3, then merge
//     physical results when multiple source chunks contribute to one result.
//   - deinterleaved factor 4 -> deinterleaved factor 2
//     Example: 32 -> 16.
//     Lowering shape: emit vcvt EVEN/ODD per source chunk pair.
//   - contiguous lane_stride = 1 -> contiguous lane_stride = 2/4
//     Example: 16 -> 8 lane_stride=2, 32 -> 8 lane_stride=4.
//     Lowering shape: emit vcvt into the widened physical carrier selected by
//     the lane-stride result type.
//
// Group-slots logical layouts: group_slots(num_groups=G, slots=1 or 8)
//   - 32-bit integer -> 16-bit integer, same group_slots layout
//     Lowering shape: direct vcvt with part = EVEN.
//   - 32-bit integer -> 8-bit integer, same group_slots layout
//     Lowering shape: direct vcvt with part = P0.
//   - 16-bit unsigned integer -> 8-bit unsigned integer, slots = 8,
//     result lane_stride = 2
//     Lowering shape: direct vcvt with part = EVEN.
//   - 32-bit integer -> 8-bit integer, result lane_stride = 4
//     Lowering shape: no vcvt; keep/bitcast the 32-bit carrier and let the
//     later store consume it as PK4_B32.
struct OneToNVMITruncIOpPattern : OpConversionPattern<VMITruncIOp> {
  using OpConversionPattern<VMITruncIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMITruncIOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);

    VMILayoutAttr sourceLayout = sourceVMIType.getLayoutAttr();
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    if (sourceLayout && resultLayout && sourceLayout.isGroupSlots() &&
        resultLayout.isGroupSlots()) {
      unsigned sourceLogicalBits =
          pto::getPTOStorageElemBitWidth(sourceVMIType.getElementType());
      unsigned resultLogicalBits =
          pto::getPTOStorageElemBitWidth(resultVMIType.getElementType());
      bool supportsDirectGroupSlotTrunc =
          sourceLogicalBits == 32 &&
          (resultLogicalBits == 16 || resultLogicalBits == 8);
      bool supportsPackedU16ToU8GroupSlotTrunc =
          sourceLogicalBits == 16 && resultLogicalBits == 8 &&
          sourceLayout.getSlots() == 8 && resultLayout.getSlots() == 8 &&
          resultLayout.hasLaneStride() && resultLayout.getLaneStride() == 2;
      if (sourceLayout.getNumGroups() != resultLayout.getNumGroups() ||
          sourceLayout.getSlots() != resultLayout.getSlots() ||
          (sourceLayout.getSlots() != 1 && sourceLayout.getSlots() != 8) ||
          (!supportsDirectGroupSlotTrunc &&
           !supportsPackedU16ToU8GroupSlotTrunc) ||
          sourceParts.size() != resultTypes.size())
        return rewriter.notifyMatchFailure(
            op, "unsupported group-slot trunci shape");

      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
      const char *activeSlotPattern =
          sourceLayout.getSlots() == 1 ? "PAT_VL1" : "PAT_VL8";
      StringRef activeSlotGranularity = sourceLogicalBits == 16 ? "b16" : "b32";
      FailureOr<Value> activeSlotMask = createPrefixMask(
          op.getLoc(), MaskType::get(rewriter.getContext(), activeSlotGranularity),
          activeSlotPattern, rewriter);
      if (failed(activeSlotMask))
        return rewriter.notifyMatchFailure(
            op, "failed to build group-slot trunci active slot mask");
      for (auto [sourcePart, physicalResultType] :
           llvm::zip_equal(sourceParts, resultTypes)) {
        auto sourceType = dyn_cast<VRegType>(sourcePart.getType());
        auto resultType = dyn_cast<VRegType>(physicalResultType);
        if (!sourceType ||
            pto::getPTOStorageElemBitWidth(sourceType.getElementType()) !=
                sourceLogicalBits ||
            !resultType)
          return rewriter.notifyMatchFailure(
              op, "unsupported group-slot trunci physical type");

        if (supportsPackedU16ToU8GroupSlotTrunc) {
          results.push_back(rewriter
                                .create<VcvtOp>(op.getLoc(), resultType,
                                                sourcePart, *activeSlotMask,
                                                /*rnd=*/nullptr, sat,
                                                rewriter.getStringAttr("EVEN"))
                                .getResult());
          continue;
        }

        unsigned physicalResultBits =
            pto::getPTOStorageElemBitWidth(resultType.getElementType());
        if (resultLayout.hasLaneStride() && resultLayout.getLaneStride() == 4 &&
            pto::getPTOStorageElemBitWidth(resultVMIType.getElementType()) ==
                8 &&
            physicalResultBits == 32) {
          if (sourcePart.getType() == resultType) {
            results.push_back(sourcePart);
          } else {
            results.push_back(
                rewriter.create<VbitcastOp>(op.getLoc(), resultType, sourcePart)
                    .getResult());
          }
          continue;
        }

        if (physicalResultBits != 16 && physicalResultBits != 8)
          return rewriter.notifyMatchFailure(
              op, "unsupported group-slot trunci physical type");

        StringAttr part = physicalResultBits == 16
                              ? rewriter.getStringAttr("EVEN")
                              : rewriter.getStringAttr("P0");
        results.push_back(rewriter
                              .create<VcvtOp>(op.getLoc(), resultType,
                                              sourcePart, *activeSlotMask,
                                              /*rnd=*/nullptr, sat, part)
                              .getResult());
      }
      replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
      return success();
    }

    if (sourceParts.empty() || resultTypes.empty())
      return rewriter.notifyMatchFailure(
          op, "trunci requires non-empty physical source and result parts");

    auto sourceType0 = dyn_cast<VRegType>(sourceParts.front().getType());
    auto resultType0 = dyn_cast<VRegType>(resultTypes.front());
    if (!sourceType0 || !isa<IntegerType>(sourceType0.getElementType()) ||
        !resultType0 || !isa<IntegerType>(resultType0.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci source/result type");
    for (Value sourcePart : sourceParts) {
      auto sourceType = dyn_cast<VRegType>(sourcePart.getType());
      if (!sourceType || sourceType != sourceType0)
        return rewriter.notifyMatchFailure(
            op, "trunci source physical parts must have matching integer type");
    }
    for (Type resultType : resultTypes) {
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      if (!resultVRegType || resultVRegType != resultType0)
        return rewriter.notifyMatchFailure(
            op, "trunci result physical parts must have matching integer type");
    }

    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(sourceType0.getElementType());
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(resultType0.getElementType());
    if (sourceBits == 0 || resultBits == 0 || sourceBits % resultBits != 0)
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci source/result width relation");

    int64_t factor = sourceBits / resultBits;

    // s32 -> s8 alias: borrow ui32 -> ui8 physical path (NOSAT bit-pattern
    // equivalent).  Rewrite source parts si32 -> ui32 / result types si8 -> ui8
    // via vbitcast; finalize() below casts merged ui8 results back to si8.
    const bool s32ToS8Alias =
        sourceType0.getElementType().isSignedInteger(32) &&
        resultType0.getElementType().isSignedInteger(8);

    SmallVector<Type> originalResultTypes;
    SmallVector<Value> aliasedSourceStorage;
    if (s32ToS8Alias) {
      auto u32ElemTy = rewriter.getIntegerType(32, /*isSigned=*/false);
      auto u8ElemTy = rewriter.getIntegerType(8, /*isSigned=*/false);
      originalResultTypes = resultTypes;

      aliasedSourceStorage.reserve(sourceParts.size());
      for (Value sp : sourceParts) {
        auto spTy = cast<VRegType>(sp.getType());
        aliasedSourceStorage.push_back(
            rewriter.create<VbitcastOp>(
                op.getLoc(),
                VRegType::get(spTy.getContext(), spTy.getElementCount(),
                              u32ElemTy),
                sp).getResult());
      }
      sourceParts = ValueRange(aliasedSourceStorage);
      sourceType0 = cast<VRegType>(aliasedSourceStorage.front().getType());

      for (Type &rt : resultTypes) {
        auto rtVReg = cast<VRegType>(rt);
        rt = VRegType::get(rtVReg.getContext(), rtVReg.getElementCount(),
                           u8ElemTy);
      }
      resultType0 = cast<VRegType>(resultTypes.front());
    }

    StringAttr sat = op->getAttrOfType<StringAttr>("saturate");

    auto finalize = [&](SmallVector<Value> &results) {
      if (s32ToS8Alias) {
        for (auto &&[i, r] : llvm::enumerate(results))
          r = rewriter
                  .create<VbitcastOp>(op.getLoc(), originalResultTypes[i], r)
                  .getResult();
      }
      replaceOpWithFlatConvertedValues(rewriter, op, results,
                                       *this->getTypeConverter());
    };

    if (sourceLayout && resultLayout && sourceLayout.isContiguous() &&
        sourceLayout.getLaneStride() == 1 && resultLayout.isContiguous() &&
        resultLayout.getLaneStride() == factor &&
        sourceParts.size() == resultTypes.size()) {
      if (factor != 2 && factor != 4)
        return rewriter.notifyMatchFailure(
            op, "unsupported dense lane_stride trunci result layout");
      StringAttr part = rewriter.getStringAttr(factor == 2 ? "EVEN" : "P0");
      FailureOr<Value> sourceMask =
          createAllTrueMaskForVReg(op.getLoc(), sourceType0, rewriter);
      if (failed(sourceMask))
        return rewriter.notifyMatchFailure(op, "failed to build trunci masks");

      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      for (auto [sourcePart, resultType] :
           llvm::zip_equal(sourceParts, resultTypes)) {
        results.push_back(rewriter
                              .create<VcvtOp>(op.getLoc(), resultType,
                                              sourcePart, *sourceMask,
                                              /*rnd=*/nullptr, sat, part)
                              .getResult());
      }
      finalize(results);
      return success();
    }

    if ((factor != 2 && factor != 4) ||
        sourceParts.size() != resultTypes.size() * factor)
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci source/result arity relation");

    ArrayRef<StringRef> parts;
    if (factor == 2) {
      static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
      parts = kEvenOddParts;
    } else if (factor == 4) {
      static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
      parts = kPacked4Parts;
    } else {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci source/result width relation");
    }

    FailureOr<Value> sourceMask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType0, rewriter);
    FailureOr<Value> resultMask =
        createAllTrueMaskForVReg(op.getLoc(), resultType0, rewriter);
    if (failed(sourceMask) || failed(resultMask))
      return rewriter.notifyMatchFailure(op, "failed to build trunci masks");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (int64_t resultIndex = 0, resultCount = resultTypes.size();
         resultIndex < resultCount; ++resultIndex) {
      Type resultType = resultTypes[resultIndex];
      SmallVector<Value> partials;
      partials.reserve(parts.size());
      for (int64_t partIndex = 0; partIndex < factor; ++partIndex) {
        Value sourcePart = sourceParts[resultIndex * factor + partIndex];
        partials.push_back(
            rewriter
                .create<VcvtOp>(op.getLoc(), resultType, sourcePart,
                                *sourceMask, /*rnd=*/nullptr, sat,
                                rewriter.getStringAttr(parts[partIndex]))
                .getResult());
      }

      Value merged = partials.front();
      for (Value partial : llvm::drop_begin(partials))
        merged = rewriter
                     .create<VorOp>(op.getLoc(), resultType, merged, partial,
                                    *resultMask)
                     .getResult();
      results.push_back(merged);
    }

    finalize(results);
    return success();
  }
};

struct OneToNVMIFPToSIOpPattern : OpConversionPattern<VMIFPToSIOp> {
  using OpConversionPattern<VMIFPToSIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIFPToSIOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (sourceParts.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(
          op, "fptosi physical source/result arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    StringAttr rnd = rewriter.getStringAttr("R");
    StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      auto sourceType = dyn_cast<VRegType>(sourcePart.getType());
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      if (!sourceType || !sourceType.getElementType().isF32() ||
          !resultVRegType ||
          !isa<IntegerType>(resultVRegType.getElementType()) ||
          pto::getPTOStorageElemBitWidth(resultVRegType.getElementType()) != 32)
        return rewriter.notifyMatchFailure(
            op, "fptosi requires physical f32 source and 32-bit integer "
                "result chunks");

      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(op, "failed to build fptosi mask");
      results.push_back(rewriter
                            .create<VcvtOp>(op.getLoc(), resultVRegType,
                                            sourcePart, *mask, rnd, sat,
                                            /*part=*/nullptr)
                            .getResult());
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMISIToFPOpPattern : OpConversionPattern<VMISIToFPOp> {
  using OpConversionPattern<VMISIToFPOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMISIToFPOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (sourceParts.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(
          op, "sitofp physical source/result arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    StringAttr rnd = rewriter.getStringAttr("R");
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      auto sourceType = dyn_cast<VRegType>(sourcePart.getType());
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      if (!sourceType || !isa<IntegerType>(sourceType.getElementType()) ||
          pto::getPTOStorageElemBitWidth(sourceType.getElementType()) != 32 ||
          !resultVRegType || !resultVRegType.getElementType().isF32())
        return rewriter.notifyMatchFailure(
            op, "sitofp requires physical 32-bit integer source and f32 "
                "result chunks");

      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(op, "failed to build sitofp mask");
      results.push_back(rewriter
                            .create<VcvtOp>(op.getLoc(), resultVRegType,
                                            sourcePart, *mask, rnd,
                                            /*sat=*/nullptr, /*part=*/nullptr)
                            .getResult());
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIBitcastOpPattern : OpConversionPattern<VMIBitcastOp> {
  using OpConversionPattern<VMIBitcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIBitcastOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (sourceParts.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(op, "physical bitcast arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      if (!isa<VRegType>(sourcePart.getType()) || !isa<VRegType>(resultType))
        return rewriter.notifyMatchFailure(
            op, "physical bitcast part type mismatch");
      results.push_back(
          rewriter.create<VbitcastOp>(op.getLoc(), resultType, sourcePart)
              .getResult());
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIChannelSplitOpPattern
    : OpConversionPattern<VMIChannelSplitOp> {
  using OpConversionPattern<VMIChannelSplitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIChannelSplitOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int64_t channels = op.getNumResults();
    if (channels != 2 && channels != 4)
      return rewriter.notifyMatchFailure(
          op, "channel_split only supports 2 or 4 channels");

    auto sourceType = cast<VMIVRegType>(op.getSource().getType());
    VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
    auto channelLayout =
        VMILayoutAttr::getDeinterleaved(rewriter.getContext(), channels);
    if (!sourceLayout ||
        (!sourceLayout.isContiguous() && sourceLayout != channelLayout))
      return rewriter.notifyMatchFailure(
          op,
          "channel_split requires contiguous or matching deinterleaved source "
          "layout");
    for (Value result : op.getResults()) {
      auto resultType = cast<VMIVRegType>(result.getType());
      VMILayoutAttr resultLayout = resultType.getLayoutAttr();
      if (!resultLayout || !resultLayout.isContiguous())
        return rewriter.notifyMatchFailure(
            op, "channel_split requires contiguous result layouts");
    }

    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    FailureOr<SmallVector<Value>> results =
        materializeDataLayoutConversion(op, adaptor.getSource(), resultTypes,
                                        sourceLayout, channelLayout,
                                        sourceType.getElementType(), rewriter);
    if (failed(results))
      return failure();

    replaceOpWithFlatConvertedValues(rewriter, op, *results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIChannelMergeOpPattern
    : OpConversionPattern<VMIChannelMergeOp> {
  using OpConversionPattern<VMIChannelMergeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIChannelMergeOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int64_t channels = op.getInputs().size();
    if (channels != 2 && channels != 4)
      return rewriter.notifyMatchFailure(
          op, "channel_merge only supports 2 or 4 channels");

    for (Value input : op.getInputs()) {
      auto inputType = cast<VMIVRegType>(input.getType());
      VMILayoutAttr inputLayout = inputType.getLayoutAttr();
      if (!inputLayout || !inputLayout.isContiguous())
        return rewriter.notifyMatchFailure(
            op, "channel_merge requires contiguous input layouts");
    }
    auto resultType = cast<VMIVRegType>(op.getResult().getType());
    VMILayoutAttr resultLayout = resultType.getLayoutAttr();
    auto channelLayout =
        VMILayoutAttr::getDeinterleaved(rewriter.getContext(), channels);
    if (!resultLayout ||
        (!resultLayout.isContiguous() && resultLayout != channelLayout))
      return rewriter.notifyMatchFailure(
          op,
          "channel_merge requires contiguous or matching deinterleaved result "
          "layout");

    FailureOr<SmallVector<Type>> maybeResultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybeResultTypes))
      return failure();
    FailureOr<SmallVector<Value>> results =
        materializeDataLayoutConversion(
            op, flattenOneToNOperands(adaptor.getOperands()),
            *maybeResultTypes, channelLayout, resultLayout,
            resultType.getElementType(), rewriter);
    if (failed(results))
      return failure();

    replaceOpWithFlatConvertedValues(rewriter, op, *results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIShuffleOpPattern : OpConversionPattern<VMIShuffleOp> {
  using OpConversionPattern<VMIShuffleOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIShuffleOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    std::string reason;
    FailureOr<SmallVector<int64_t>> sourceFlatIndices =
        computeShuffleForwardingSourceParts(op, &reason);
    if (succeeded(sourceFlatIndices)) {
      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      for (int64_t sourceFlatIndex : *sourceFlatIndices) {
        if (sourceFlatIndex >= static_cast<int64_t>(sourceParts.size()))
          return rewriter.notifyMatchFailure(
              op, "shuffle forwarding source part range is out of bounds");
        results.push_back(sourceParts[sourceFlatIndex]);
      }

      if (failed(
              verifyIdentityPartForwarding(op, results, resultTypes, rewriter)))
        return failure();

      replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
      return success();
    }

    std::string splatReason;
    FailureOr<int64_t> splatSource =
        computeShuffleLane0SplatSourcePart(op, &splatReason);
    if (succeeded(splatSource)) {
      if (*splatSource >= static_cast<int64_t>(sourceParts.size()))
        return rewriter.notifyMatchFailure(
            op, "shuffle lane0 splat source part range is out of bounds");

      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      Value sourcePart = sourceParts[*splatSource];
      for (Type resultType : resultTypes) {
        auto sourceVRegType = dyn_cast<VRegType>(sourcePart.getType());
        auto resultVRegType = dyn_cast<VRegType>(resultType);
        if (!sourceVRegType || !resultVRegType ||
            sourceVRegType != resultVRegType)
          return rewriter.notifyMatchFailure(
              op, "shuffle lane0 splat requires matching physical vreg type");
        FailureOr<Value> mask =
            createAllTrueMaskForVReg(op.getLoc(), resultVRegType, rewriter);
        if (failed(mask))
          return rewriter.notifyMatchFailure(
              op, "failed to create shuffle lane0 splat mask");
        results.push_back(rewriter
                              .create<VdupOp>(op.getLoc(), resultType,
                                              sourcePart, *mask,
                                              rewriter.getStringAttr("LOWEST"))
                              .getResult());
      }

      replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
      return success();
    }

    std::string vselrReason;
    FailureOr<SmallVector<ShuffleVselrPlan>> vselrPlans =
        computeShuffleVselrPlans(op, &vselrReason);
    if (failed(vselrPlans))
      return rewriter.notifyMatchFailure(op,
                                         Twine("shuffle vselr ") + vselrReason);

    if (vselrPlans->size() != resultTypes.size())
      return rewriter.notifyMatchFailure(op, "shuffle vselr arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [plan, resultType] : llvm::zip_equal(*vselrPlans, resultTypes)) {
      if (plan.sourceFlatIndex >= static_cast<int64_t>(sourceParts.size()))
        return rewriter.notifyMatchFailure(
            op, "shuffle vselr source part range is out of bounds");

      auto sourceVRegType =
          dyn_cast<VRegType>(sourceParts[plan.sourceFlatIndex].getType());
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      if (!sourceVRegType || !resultVRegType ||
          sourceVRegType.getElementCount() !=
              resultVRegType.getElementCount() ||
          sourceVRegType.getElementType() != resultVRegType.getElementType())
        return rewriter.notifyMatchFailure(
            op, "shuffle vselr source/result type mismatch");

      unsigned indexBits =
          pto::getPTOStorageElemBitWidth(sourceVRegType.getElementType());
      if (indexBits != 8 && indexBits != 16 && indexBits != 32)
        return rewriter.notifyMatchFailure(
            op, "shuffle vselr requires 8/16/32-bit index elements");

      auto indexElementType =
          IntegerType::get(rewriter.getContext(), indexBits);
      Type indexType =
          VRegType::get(rewriter.getContext(), sourceVRegType.getElementCount(),
                        indexElementType);
      FailureOr<Value> base = createScalarOffsetConstant(
          op.getLoc(), indexElementType, plan.baseLane, rewriter);
      if (failed(base))
        return rewriter.notifyMatchFailure(
            op, "failed to materialize shuffle vselr index base");
      StringAttr orderAttr =
          plan.descending ? rewriter.getStringAttr("DESC") : StringAttr{};
      Value indexVector =
          rewriter.create<VciOp>(op.getLoc(), indexType, *base, orderAttr)
              .getResult();
      results.push_back(rewriter
                            .create<VselrOp>(op.getLoc(), resultType,
                                             sourceParts[plan.sourceFlatIndex],
                                             indexVector)
                            .getResult());
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

Block *convertBranchDestBlock(Block *block, ConversionPatternRewriter &rewriter,
                              const TypeConverter &typeConverter,
                              llvm::DenseMap<Block *, Block *> &converted) {
  auto [it, inserted] = converted.try_emplace(block, nullptr);
  if (!inserted)
    return it->second;

  TypeConverter::SignatureConversion argMapping(block->getNumArguments());
  if (failed(typeConverter.convertSignatureArgs(block->getArgumentTypes(),
                                                argMapping)) ||
      !hasNonIdentitySignatureConversion(block->getArgumentTypes(),
                                         argMapping)) {
    it->second = block;
    return block;
  }

  Block *newBlock =
      rewriter.applySignatureConversion(block, argMapping, &typeConverter);
  it->second = newBlock;
  return newBlock;
}

struct OneToNCFBranchOpPattern : OpConversionPattern<cf::BranchOp> {
  using OpConversionPattern<cf::BranchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *converter = this->getTypeConverter();
    llvm::DenseMap<Block *, Block *> convertedBlocks;
    Block *dest = convertBranchDestBlock(op.getDest(), rewriter, *converter,
                                         convertedBlocks);
    SmallVector<Value> destOperands =
        flattenOneToNOperands(adaptor.getDestOperands());

    if (isIdentityOneToNValueMapping(op.getDestOperands(),
                                     adaptor.getDestOperands()) &&
        dest == op.getDest())
      return failure();

    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, dest, destOperands);
    return success();
  }
};

struct OneToNCFCondBranchOpPattern
    : OpConversionPattern<cf::CondBranchOp> {
  using OpConversionPattern<cf::CondBranchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *converter = this->getTypeConverter();
    llvm::DenseMap<Block *, Block *> convertedBlocks;
    Block *trueDest = convertBranchDestBlock(op.getTrueDest(), rewriter,
                                             *converter, convertedBlocks);
    Block *falseDest = convertBranchDestBlock(op.getFalseDest(), rewriter,
                                              *converter, convertedBlocks);

    if (isIdentityOneToNValueMapping(op.getTrueDestOperands(),
                                     adaptor.getTrueDestOperands()) &&
        isIdentityOneToNValueMapping(op.getFalseDestOperands(),
                                     adaptor.getFalseDestOperands()) &&
        trueDest == op.getTrueDest() && falseDest == op.getFalseDest())
      return failure();

    ValueRange condition = adaptor.getCondition();
    if (condition.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "condition converted to multiple values");

    SmallVector<Value> trueOperands =
        flattenOneToNOperands(adaptor.getTrueDestOperands());
    SmallVector<Value> falseOperands =
        flattenOneToNOperands(adaptor.getFalseDestOperands());

    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(op, condition.front(),
                                                  trueDest, trueOperands,
                                                  falseDest, falseOperands);
    return success();
  }
};

struct OneToNCFSwitchOpPattern : OpConversionPattern<cf::SwitchOp> {
  using OpConversionPattern<cf::SwitchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::SwitchOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *converter = this->getTypeConverter();
    llvm::DenseMap<Block *, Block *> convertedBlocks;
    Block *defaultDest = convertBranchDestBlock(
        op.getDefaultDestination(), rewriter, *converter, convertedBlocks);

    SmallVector<Block *> caseDests;
    caseDests.reserve(op.getCaseDestinations().size());
    for (Block *dest : op.getCaseDestinations())
      caseDests.push_back(
          convertBranchDestBlock(dest, rewriter, *converter, convertedBlocks));

    bool changed = defaultDest != op.getDefaultDestination();
    for (auto [oldDest, newDest] :
         llvm::zip(op.getCaseDestinations(), caseDests))
      changed |= oldDest != newDest;
    changed |= !isIdentityOneToNValueMapping(op.getDefaultOperands(),
                                             adaptor.getDefaultOperands());
    for (auto [originalOperands, convertedOperands] :
         llvm::zip(op.getCaseOperands(), adaptor.getCaseOperands()))
      changed |=
          !isIdentityOneToNValueMapping(originalOperands, convertedOperands);
    if (!changed)
      return failure();

    ValueRange flag = adaptor.getFlag();
    if (flag.size() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "flag converted to multiple values");

    SmallVector<Value> defaultOperands;
    SmallVector<SmallVector<Value>> caseOperandStorage;
    SmallVector<ValueRange> caseOperands;
    defaultOperands = flattenOneToNOperands(adaptor.getDefaultOperands());

    caseOperandStorage.reserve(op.getCaseOperandSegments().size());
    caseOperands.reserve(op.getCaseOperandSegments().size());
    for (ArrayRef<ValueRange> convertedOperands : adaptor.getCaseOperands())
      caseOperandStorage.push_back(flattenOneToNOperands(convertedOperands));
    for (SmallVector<Value> &operands : caseOperandStorage)
      caseOperands.push_back(operands);

    rewriter.replaceOpWithNewOp<cf::SwitchOp>(
        op, flag.front(), defaultDest, defaultOperands, op.getCaseValuesAttr(),
        caseDests, caseOperands);
    return success();
  }
};

struct OneToNSCFExecuteRegionOpPattern
    : OpConversionPattern<scf::ExecuteRegionOp> {
  using OpConversionPattern<
      scf::ExecuteRegionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ExecuteRegionOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<SmallVector<Type>> maybeResultTypes =
        getConvertedResultTypes(op, *this->getTypeConverter());
    if (failed(maybeResultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybeResultTypes);
    if (resultTypes == op->getResultTypes())
      return failure();

    auto newOp =
        rewriter.create<scf::ExecuteRegionOp>(op.getLoc(), resultTypes);
    newOp->setAttrs(op->getAttrs());
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());
    replaceOpWithFlatConvertedValues(
            rewriter, op, newOp->getResults(), *this->getTypeConverter());
    return success();
  }
};

struct OneToNSCFIndexSwitchOpPattern
    : OpConversionPattern<scf::IndexSwitchOp> {
  using OpConversionPattern<
      scf::IndexSwitchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IndexSwitchOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange arg = adaptor.getArg();
    if (arg.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "index_switch selector converted to multiple values");

    FailureOr<SmallVector<Type>> maybeResultTypes =
        getConvertedResultTypes(op, *this->getTypeConverter());
    if (failed(maybeResultTypes))
      return failure();
    SmallVector<Type> resultTypes = std::move(*maybeResultTypes);
    if (resultTypes == op->getResultTypes())
      return failure();

    auto newOp = rewriter.create<scf::IndexSwitchOp>(
        op.getLoc(), resultTypes, arg.front(), op.getCases(), op.getNumCases());
    newOp->setAttrs(op->getAttrs());
    rewriter.inlineRegionBefore(op.getDefaultRegion(), newOp.getDefaultRegion(),
                                newOp.getDefaultRegion().end());
    for (auto [srcRegion, dstRegion] :
         llvm::zip(op.getCaseRegions(), newOp.getCaseRegions()))
      rewriter.inlineRegionBefore(srcRegion, dstRegion, dstRegion.end());
    replaceOpWithFlatConvertedValues(
            rewriter, op, newOp->getResults(), *this->getTypeConverter());
    return success();
  }
};

void populateVMIConversionPatterns(
    VMIToVPTOTypeConverter &typeConverter, RewritePatternSet &patterns) {
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
  scf::populateSCFStructuralTypeConversions(typeConverter, patterns);
  patterns.add<OneToNCFBranchOpPattern, OneToNCFCondBranchOpPattern,
               OneToNCFSwitchOpPattern>(typeConverter, patterns.getContext());
  patterns.add<OneToNSCFExecuteRegionOpPattern, OneToNSCFIndexSwitchOpPattern>(
      typeConverter, patterns.getContext());
  patterns.add<OneToNVMIPackOpPattern, OneToNVMIUnpackOpPattern>(
      typeConverter, patterns.getContext());
  patterns.add<
      OneToNVMIEnsureLayoutOpPattern, OneToNVMIEnsureMaskLayoutOpPattern,
      OneToNVMIBroadcastOpPattern, OneToNVMIIotaOpPattern,
      OneToNVMIConstantOpPattern, OneToNVMIConstantMaskOpPattern,
      OneToNVMICreateMaskOpPattern, OneToNVMICreateGroupMaskOpPattern,
      OneToNVMIMaskBinaryOpPattern<VMIMaskAndOp, PandOp>,
      OneToNVMIMaskBinaryOpPattern<VMIMaskOrOp, PorOp>,
      OneToNVMIMaskBinaryOpPattern<VMIMaskXOrOp, PxorOp>,
      OneToNVMIMaskUnaryOpPattern<VMIMaskNotOp, PnotOp>, OneToNVMILoadOpPattern,
      OneToNVMIDeinterleaveLoadOpPattern, OneToNVMIGroupLoadOpPattern,
      OneToNVMIGroupSlotLoadOpPattern, OneToNVMIStrideLoadOpPattern,
      OneToNVMIMaskedLoadOpPattern, OneToNVMIGatherOpPattern,
      OneToNVMIExpandLoadOpPattern, OneToNVMIStoreOpPattern,
      OneToNVMIInterleaveStoreOpPattern, OneToNVMIGroupStoreOpPattern,
      OneToNVMIMaskedStoreOpPattern, OneToNVMIStrideStoreOpPattern,
      OneToNVMIScatterOpPattern, OneToNVMIBinaryOpPattern<VMIAddFOp, VaddOp>,
      OneToNVMIBinaryOpPattern<VMIAddIOp, VaddOp>,
      OneToNVMIBinaryOpPattern<VMISubFOp, VsubOp>,
      OneToNVMIBinaryOpPattern<VMISubIOp, VsubOp>,
      OneToNVMIBinaryOpPattern<VMIMulFOp, VmulOp>,
      OneToNVMIBinaryOpPattern<VMIMulIOp, VmulOp>, OneToNVMIVmullOpPattern,
      OneToNVMIFmaOpPattern, OneToNVMIBinaryOpPattern<VMIDivFOp, VdivOp>,
      OneToNVMIBinaryOpPattern<VMIMinFOp, VminOp>,
      OneToNVMIBinaryOpPattern<VMIMaxFOp, VmaxOp>,
      OneToNVMIUnaryOpPattern<VMINegFOp, VnegOp>,
      OneToNVMIUnaryOpPattern<VMIAbsFOp, VabsOp>,
      OneToNVMIUnaryOpPattern<VMIAbsIOp, VabsOp>,
      OneToNVMIUnaryOpPattern<VMISqrtOp, VsqrtOp>,
      OneToNVMIUnaryOpPattern<VMIExpOp, VexpOp>,
      OneToNVMIUnaryOpPattern<VMILnOp, VlnOp>,
      OneToNVMIUnaryOpPattern<VMIReluOp, VreluOp>,
      OneToNVMIBinaryOpPattern<VMIAndIOp, VandOp>,
      OneToNVMIBinaryOpPattern<VMIOrIOp, VorOp>,
      OneToNVMIBinaryOpPattern<VMIXOrIOp, VxorOp>,
      OneToNVMIBinaryOpPattern<VMIShLIOp, VshlOp>,
      OneToNVMIBinaryOpPattern<VMIShRUIOp, VshrOp>,
      OneToNVMIBinaryOpPattern<VMIShRSIOp, VshrOp>,
      OneToNVMIUnaryOpPattern<VMINotOp, VnotOp>,
      OneToNVMICmpOpPattern<VMICmpFOp>, OneToNVMICmpOpPattern<VMICmpIOp>,
      OneToNVMISelectOpPattern, OneToNVMIActivePrefixIndexOpPattern,
      OneToNVMICompressOpPattern, OneToNVMICompressStoreOpPattern,
      OneToNVMIReduceAddIOpPattern, OneToNVMIReduceAddFOpPattern,
      OneToNVMIGroupBroadcastOpPattern, OneToNVMIVdhistOpPattern,
      OneToNVMIVchistOpPattern,
      OneToNVMIReduceMinMaxOpPattern<VMIReduceMaxFOp, VcmaxOp, VmaxOp>,
      OneToNVMIReduceMinMaxOpPattern<VMIReduceMinFOp, VcminOp, VminOp>,
      OneToNVMIReduceMinMaxOpPattern<VMIReduceMaxIOp, VcmaxOp, VmaxOp>,
      OneToNVMIReduceMinMaxOpPattern<VMIReduceMinIOp, VcminOp, VminOp>,
      OneToNVMIExtFOpPattern, OneToNVMITruncFOpPattern,
      OneToNVMIExtIOpPattern<VMIExtSIOp>, OneToNVMIExtIOpPattern<VMIExtUIOp>,
      OneToNVMITruncIOpPattern, OneToNVMIFPToSIOpPattern,
      OneToNVMISIToFPOpPattern, OneToNVMIBitcastOpPattern,
      OneToNVMIInterleaveOpPattern<VMIVintlvOp, VintlvOp>,
      OneToNVMIInterleaveOpPattern<VMIVdintlvOp, VdintlvOp>,
      OneToNVMIChannelSplitOpPattern, OneToNVMIChannelMergeOpPattern,
      OneToNVMIShuffleOpPattern>(typeConverter, patterns.getContext());
  patterns.add<OneToNVMIGroupBroadcastLoadOpPattern>(
      typeConverter, patterns.getContext());
  patterns.add<
      OneToNVMIGroupReduceOpPattern<VMIGroupReduceAddFOp, VcgaddOp, VcaddOp,
                                    VaddOp>,
      OneToNVMIGroupReduceOpPattern<VMIGroupReduceAddIOp, VcgaddOp, VcaddOp,
                                    VaddOp>,
      OneToNVMIGroupReduceOpPattern<VMIGroupReduceMaxIOp, VcgmaxOp, VcmaxOp,
                                    VmaxOp>,
      OneToNVMIGroupReduceOpPattern<VMIGroupReduceMaxFOp, VcgmaxOp, VcmaxOp,
                                    VmaxOp>,
      OneToNVMIGroupReduceOpPattern<VMIGroupReduceMinIOp, VcgminOp, VcminOp,
                                    VminOp>,
      OneToNVMIGroupReduceOpPattern<VMIGroupReduceMinFOp, VcgminOp, VcminOp,
                                    VminOp>>(typeConverter,
                                             patterns.getContext());
  patterns.add<OneToNVMIEnsureMaskGranularityOpPattern>(
      typeConverter, patterns.getContext());
}

LogicalResult verifyNoResidualVMIIR(ModuleOp module) {
  WalkResult result = module.walk([&](Operation *op) {
    if (auto createMask = dyn_cast<VMICreateMaskOp>(op)) {
      if (!createMask.getActiveLanes().getDefiningOp<arith::ConstantOp>()) {
        createMask.emitError()
            << kVMIDiagUnsupportedPrefix
            << "dynamic pto.vmi.create_mask active_lanes could not be lowered "
               "by the current runtime predicate generation plan";
        return WalkResult::interrupt();
      }
    }
    if (auto constant = dyn_cast<VMIConstantOp>(op)) {
      auto denseAttr = dyn_cast<DenseElementsAttr>(constant.getValue());
      if (denseAttr && !denseAttr.isSplat()) {
        constant.emitError()
            << kVMIDiagUnsupportedPrefix
            << "non-splat pto.vmi.constant requires a vreg immediate or "
               "scratch materialization plan";
        return WalkResult::interrupt();
      }
    }
    if (isVMIOp(op) || hasVMIType(op)) {
      op->emitError() << kVMIDiagResidualOpPrefix
                      << "failed to convert all VMI ops/types to VPTO";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

LogicalResult checkSupportedExtFShape(VMIExtFOp op,
                                      std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (failed(supports.getExtFSupport(op, reason)))
    return failure();
  return success();
}

LogicalResult checkSupportedTruncFShape(VMITruncFOp op,
                                        std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (failed(supports.getTruncFSupport(op, reason)))
    return failure();
  return success();
}

LogicalResult checkSupportedExtSIShape(VMIExtSIOp op,
                                       std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (failed(supports.getExtSISupport(op, reason)))
    return failure();
  return success();
}

LogicalResult checkSupportedExtUIShape(VMIExtUIOp op,
                                       std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (failed(supports.getExtUISupport(op, reason)))
    return failure();
  return success();
}

LogicalResult checkSupportedTruncIShape(VMITruncIOp op,
                                        std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (failed(supports.getTruncISupport(op, reason)))
    return failure();
  return success();
}

LogicalResult checkSupportedFPToSIShape(VMIFPToSIOp op,
                                        std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !resultLayout)
    return fail("requires assigned source/result layouts");
  if (sourceLayout != resultLayout)
    return fail("requires source/result layouts to match");
  if (!sourceType.getElementType().isF32())
    return fail("requires f32 source element type");
  if (!isa<IntegerType>(resultType.getElementType()) ||
      pto::getPTOStorageElemBitWidth(resultType.getElementType()) != 32)
    return fail("requires 32-bit integer result element type");
  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (failed(sourceArity) || failed(resultArity) ||
      *sourceArity != *resultArity)
    return fail("requires matching computable physical arity");
  return success();
}

LogicalResult checkSupportedSIToFPShape(VMISIToFPOp op,
                                        std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !resultLayout)
    return fail("requires assigned source/result layouts");
  if (sourceLayout != resultLayout)
    return fail("requires source/result layouts to match");
  if (!isa<IntegerType>(sourceType.getElementType()) ||
      pto::getPTOStorageElemBitWidth(sourceType.getElementType()) != 32)
    return fail("requires 32-bit integer source element type");
  if (!resultType.getElementType().isF32())
    return fail("requires f32 result element type");
  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (failed(sourceArity) || failed(resultArity) ||
      *sourceArity != *resultArity)
    return fail("requires matching computable physical arity");
  return success();
}

LogicalResult checkSupportedBitcastShape(VMIBitcastOp op, std::string *reason) {
  VMILayoutSupport supports;
  if (failed(supports.getBitcastSupport(op, reason)))
    return failure();
  return success();
}

LogicalResult
checkSupportedChannelSplitShape(VMIChannelSplitOp op,
                                std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  int64_t channels = op.getNumResults();
  if (channels != 2 && channels != 4)
    return fail("pto.vmi.channel_split supports only 2 or 4 channels");

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  if (!sourceLayout)
    return fail("requires assigned source layout");
  auto expectedLayout =
      VMILayoutAttr::getDeinterleaved(op.getContext(), channels);
  if (!sourceLayout.isContiguous() && sourceLayout != expectedLayout)
    return fail("requires source layout to be contiguous or matching "
                "deinterleaved channel layout");

  for (Value result : op.getResults()) {
    VMILayoutAttr resultLayout =
        cast<VMIVRegType>(result.getType()).getLayoutAttr();
    if (!resultLayout || !resultLayout.isContiguous())
      return fail("requires every result layout to be contiguous");
  }

  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  int64_t resultArity = 0;
  for (Value result : op.getResults()) {
    FailureOr<int64_t> arity =
        getVMIPhysicalArity(cast<VMIVRegType>(result.getType()));
    if (failed(arity))
      return fail("requires computable result physical arity");
    resultArity += *arity;
  }
  if (failed(sourceArity))
    return fail("requires computable source physical arity");
  if (*sourceArity != resultArity)
    return fail("requires source and result to have the same physical arity");

  return success();
}

LogicalResult
checkSupportedChannelMergeShape(VMIChannelMergeOp op,
                                std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  int64_t channels = op.getInputs().size();
  if (channels != 2 && channels != 4)
    return fail("pto.vmi.channel_merge supports only 2 or 4 channels");

  int64_t inputArity = 0;
  for (Value input : op.getInputs()) {
    auto inputType = cast<VMIVRegType>(input.getType());
    VMILayoutAttr inputLayout = inputType.getLayoutAttr();
    if (!inputLayout || !inputLayout.isContiguous())
      return fail("requires every input layout to be contiguous");
    FailureOr<int64_t> arity = getVMIPhysicalArity(inputType);
    if (failed(arity))
      return fail("requires computable input physical arity");
    inputArity += *arity;
  }

  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!resultLayout)
    return fail("requires assigned result layout");
  auto expectedLayout =
      VMILayoutAttr::getDeinterleaved(op.getContext(), channels);
  if (!resultLayout.isContiguous() && resultLayout != expectedLayout)
    return fail("requires result layout to be contiguous or matching "
                "deinterleaved channel layout");

  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (failed(resultArity))
    return fail("requires computable result physical arity");
  if (*resultArity != inputArity)
    return fail("requires source and result to have the same physical arity");

  return success();
}

LogicalResult
checkSupportedActivePrefixIndexShape(VMIActivePrefixIndexOp op,
                                     std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!maskLayout || !resultLayout)
    return fail("requires assigned mask and result layouts");
  if (!maskLayout.isContiguous() || !resultLayout.isContiguous())
    return fail("requires contiguous mask and result layouts");

  std::string resultFullReason;
  if (failed(checkFullDataPhysicalChunks(resultType, &resultFullReason)))
    return fail(Twine("requires full result physical chunks so padding mask "
                      "lanes cannot affect the observable prefix; ") +
                resultFullReason);

  std::string maskFullReason;
  if (failed(checkFullVMIPhysicalChunks(maskType, &maskFullReason)))
    return fail(Twine("requires full mask physical chunks so padding mask "
                      "lanes cannot affect the observable prefix; ") +
                maskFullReason);

  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (failed(maskArity) || failed(resultArity))
    return fail("requires computable mask and result physical arity");
  if (*maskArity != 1 || *resultArity != 1)
    return fail("requires a single physical chunk; multi-chunk prefix needs "
                "cross-chunk carry");

  return success();
}

LogicalResult checkSupportedCompressShape(VMICompressOp op,
                                          std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !maskLayout || !resultLayout)
    return fail("requires assigned source, mask, and result layouts");
  if (!sourceLayout.isContiguous() || !maskLayout.isContiguous() ||
      !resultLayout.isContiguous())
    return fail("requires contiguous source, mask, and result layouts");

  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(sourceType, &fullChunkReason)))
    return fail(Twine("requires full source physical chunks so padding mask "
                      "lanes cannot be squeezed into the result; ") +
                fullChunkReason);

  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (failed(sourceArity) || failed(maskArity) || failed(resultArity))
    return fail("requires computable source, mask, and result physical arity");
  if (*sourceArity != 1 || *maskArity != 1 || *resultArity != 1)
    return fail("requires a single physical chunk; multi-chunk compress needs "
                "cross-chunk compaction");

  return success();
}

LogicalResult checkSupportedCompressStoreShape(
    VMICompressStoreOp op,
    std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto valueType = cast<VMIVRegType>(op.getValue().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  VMILayoutAttr valueLayout = valueType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  if (!valueLayout || !maskLayout)
    return fail("requires assigned value and mask layouts");
  if (!valueLayout.isContiguous() || !maskLayout.isContiguous())
    return fail("requires contiguous value and mask layouts");

  if (!isa<PtrType>(op.getDestination().getType()))
    return fail("requires !pto.ptr destination because pto.vstur is "
                "pointer-only");

  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(valueType, &fullChunkReason)))
    return fail(Twine("requires full physical chunks so padding mask lanes "
                      "cannot be squeezed into memory; ") +
                fullChunkReason);

  FailureOr<int64_t> valueArity = getVMIPhysicalArity(valueType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  if (failed(valueArity) || failed(maskArity))
    return fail("requires computable value and mask physical arity");
  if (*valueArity != 1 || *maskArity != 1)
    return fail("requires a single physical chunk; multi-chunk "
                "compress_store needs cross-chunk compaction and SQZN "
                "state planning");

  return success();
}

template <typename OpTy>
LogicalResult
checkSupportedReduceShape(OpTy op, bool requiresReassoc,
                          std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  if (requiresReassoc && !op->hasAttr("reassoc"))
    return fail("requires reassoc attr for pair-wise floating-point vcadd");

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto initType = cast<VMIVRegType>(op.getInit().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr initLayout = initType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !initLayout || !maskLayout || !resultLayout)
    return fail("requires assigned source, init, mask, and result layouts");
  if (!sourceLayout.isContiguous() || !initLayout.isContiguous() ||
      !maskLayout.isContiguous() || !resultLayout.isContiguous())
    return fail("requires contiguous source, init, mask, and result layouts");

  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(sourceType, &fullChunkReason)))
    return fail(Twine("requires full source physical chunks so padding lanes "
                      "do not participate in the reduction; ") +
                fullChunkReason);

  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> initArity = getVMIPhysicalArity(initType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (failed(sourceArity) || failed(initArity) || failed(maskArity) ||
      failed(resultArity))
    return fail("requires computable physical arity");
  if (*sourceArity < 1 || *maskArity != *sourceArity)
    return fail("requires source and mask physical arity to match and be "
                "non-empty");
  if (*initArity != 1 || *resultArity != 1)
    return fail("requires one init and result physical chunk");

  return success();
}

template <typename OpTy>
LogicalResult
checkSupportedGroupReduceShape(OpTy op, std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if constexpr (std::is_same_v<OpTy, VMIGroupReduceAddFOp>) {
    if (succeeded(supports.getGroupReduceAddFSupport(op, reason)))
      return success();
  } else if constexpr (std::is_same_v<OpTy, VMIGroupReduceMaxFOp>) {
    if (succeeded(supports.getGroupReduceMaxFSupport(op, reason)))
      return success();
  } else if constexpr (std::is_same_v<OpTy, VMIGroupReduceMaxIOp>) {
    if (succeeded(supports.getGroupReduceMaxISupport(op, reason)))
      return success();
  } else if constexpr (std::is_same_v<OpTy, VMIGroupReduceMinFOp>) {
    if (succeeded(supports.getGroupReduceMinFSupport(op, reason)))
      return success();
  } else if constexpr (std::is_same_v<OpTy, VMIGroupReduceMinIOp>) {
    if (succeeded(supports.getGroupReduceMinISupport(op, reason)))
      return success();
  } else {
    if (succeeded(supports.getGroupReduceAddISupport(op, reason)))
      return success();
  }
  return failure();
}

LogicalResult checkSupportedGroupBroadcastShape(
    VMIGroupBroadcastOp op,
    std::string *reason = nullptr) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (sourceType.getElementType() != resultType.getElementType()) {
    if (reason)
      *reason = "requires source/result element type to match";
    return failure();
  }
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !resultLayout)
    return fail("requires assigned source/result layouts");
  int64_t numGroups = op.getNumGroupsAttr().getInt();
  if (numGroups <= 0)
    return fail("requires positive num_groups");
  if (sourceType.getElementCount() != numGroups)
    return fail("requires source lane count to match num_groups");
  if (resultType.getElementCount() % numGroups != 0)
    return fail("requires num_groups to evenly divide result lane count");
  if (!sourceLayout.isGroupSlots() || sourceLayout.getNumGroups() != numGroups)
    return fail("requires matching num_groups source layout");
  if (resultLayout.isGroupSlots())
    return fail("requires dense result layout");

  if (sourceLayout.getSlots() > 0 && sourceLayout.getSlots() != 8 &&
      sourceLayout.getSlots() != 1)
    return fail("supports only slots=8 or slots=1 group_broadcast source "
                "layouts");
  if (sourceLayout.getSlots() > 1 && numGroups % sourceLayout.getSlots() != 0)
    return fail("requires full source group-slot chunks");

  VMILayoutSupport supports;
  std::string supportReason;
  if (failed(supports.getGroupBroadcastSupport(op, &supportReason)))
    return fail(supportReason);

  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(sourceType.getElementType());
  FailureOr<int64_t> resultLanesPerPart =
      getDataLanesPerPart(resultType.getElementType());
  if (failed(lanesPerPart) || failed(resultLanesPerPart) ||
      *lanesPerPart != *resultLanesPerPart)
    return fail("requires matching physical lanes per part");
  FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
      resultType, numGroups, reason);
  if (failed(groupSize))
    return failure();
  if (*lanesPerPart % *groupSize != 0 && *groupSize % *lanesPerPart != 0)
    return fail("requires derived group size to divide or be a multiple of "
                "physical lanes per part");

  FailureOr<int64_t> resultFactor = getDataLayoutFactor(resultType);
  if (failed(resultFactor))
    return fail("requires known result layout factor");
  bool laneStridedDense =
      resultLayout.isDense() && resultLayout.getLaneStride() > 1;
  if (!laneStridedDense) {
    std::string fullChunkReason;
    if (failed(checkFullDataPhysicalChunks(resultType, &fullChunkReason)))
      return fail(Twine("requires full result physical chunks; ") +
                  fullChunkReason);
  }
  if (*resultFactor == 1)
    return success();
  FailureOr<int64_t> resultBlockElems =
      getVMILayoutBlockElems(resultType);
  bool blockFragmentSmallGroup =
      resultLayout.isBlockDeinterleaved() && succeeded(resultBlockElems) &&
      *groupSize < *lanesPerPart && *lanesPerPart % *resultBlockElems == 0;
  bool deinterleavedSmallGroup =
      resultLayout.isDeinterleaved() &&
      *groupSize < *lanesPerPart && *groupSize >= *resultFactor &&
      *groupSize % *resultFactor == 0 &&
      *lanesPerPart % (*groupSize / *resultFactor) == 0;
  if (blockFragmentSmallGroup || deinterleavedSmallGroup)
    return success();
  int64_t logicalSpanPerResultChunk = *lanesPerPart * *resultFactor;
  if (*groupSize < *lanesPerPart || *groupSize % logicalSpanPerResultChunk != 0)
    return fail("deinterleaved result requires every physical result chunk to "
                "stay within one logical group");
  return success();
}

LogicalResult checkSupportedVdhistShape(VMIVdhistOp op,
                                       std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (succeeded(supports.getVdhistSupport(op, reason)))
    return success();
  return failure();
}

LogicalResult checkSupportedVchistShape(VMIVchistOp op,
                                       std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (succeeded(supports.getVchistSupport(op, reason)))
    return success();
  return failure();
}

LogicalResult checkSupportedVmullShape(VMIVmullOp op,
                                       std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto aType = cast<VMIVRegType>(op.getA().getType());
  auto bType = cast<VMIVRegType>(op.getB().getType());
  auto lowType = cast<VMIVRegType>(op.getLow().getType());
  auto highType = cast<VMIVRegType>(op.getHigh().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());

  auto elementType = dyn_cast<IntegerType>(aType.getElementType());
  if (!elementType || elementType.getWidth() != 32 ||
      (!elementType.isSignless() && !elementType.isUnsigned()))
    return fail("requires element type to be exactly i32 or ui32");
  if (aType != bType || aType != lowType || aType != highType)
    return fail("requires identical a, b, low, and high VMI vreg types");

  int64_t lanes = aType.getElementCount();
  if (lanes != 64 && lanes != 128 && lanes != 256)
    return fail("requires logical lane count 64, 128, or 256");

  VMILayoutAttr layout = aType.getLayoutAttr();
  if (!layout)
    return fail("requires an assigned data layout");
  bool supportedLayout =
      layout.getLaneStride() == 1 &&
      (layout.isContiguous() ||
       (layout.isDeinterleaved() &&
        (layout.getFactor() == 2 || layout.getFactor() == 4)));
  if (!supportedLayout)
    return fail("requires contiguous or deinterleaved factor 2/4 layout with "
                "lane_stride=1");
  if (maskType.getLayoutAttr() != layout)
    return fail("requires the mask and all four data values to share one "
                "layout");
  if (maskType.getGranularity() != "b32")
    return fail("requires b32 mask granularity");

  FailureOr<int64_t> aArity = getVMIPhysicalArity(aType);
  FailureOr<int64_t> bArity = getVMIPhysicalArity(bType);
  FailureOr<int64_t> lowArity = getVMIPhysicalArity(lowType);
  FailureOr<int64_t> highArity = getVMIPhysicalArity(highType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  if (failed(aArity) || failed(bArity) || failed(lowArity) ||
      failed(highArity) || failed(maskArity) || *aArity < 1)
    return fail("requires computable non-empty physical arity on every port");
  if (*aArity != *bArity || *aArity != *lowArity || *aArity != *highArity ||
      *aArity != *maskArity)
    return fail("requires matching physical arity on a, b, mask, low, and "
                "high");

  FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(aType.getElementType());
  FailureOr<Type> physicalElementType = getVMIVRegPhysicalElementType(aType);
  FailureOr<StringRef> physicalMaskGranularity =
      getVMIMaskPhysicalGranularity(maskType);
  if (failed(lanesPerPart) || *lanesPerPart != 64 ||
      failed(physicalElementType) ||
      *physicalElementType != aType.getElementType() ||
      failed(physicalMaskGranularity) || *physicalMaskGranularity != "b32")
    return fail("requires 64xi32/ui32 data parts with corresponding b32 mask "
                "parts");

  return success();
}

LogicalResult
checkSupportedFmaShape(VMIFmaOp op, std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto lhsType = cast<VMIVRegType>(op.getLhs().getType());
  FailureOr<int64_t> arity = getVMIPhysicalArity(lhsType);
  if (failed(arity) || *arity < 1)
    return fail("requires computable non-empty physical arity");

  return success();
}

LogicalResult
checkSupportedReluShape(VMIReluOp op, std::string *reason = nullptr) {
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (failed(checkSupportedMaskableVReg(resultType, reason)))
    return failure();

  return success();
}

void emitEnsureLayoutMaterializationError(VMIEnsureLayoutOp ensure,
                                          VMIVRegType sourceType,
                                          VMIVRegType resultType,
                                          StringRef reason) {
  if (ensure.getResult().hasOneUse()) {
    OpOperand &use = *ensure.getResult().use_begin();
    Operation *requester = use.getOwner();
    InFlightDiagnostic diag =
        requester->emitError()
        << kVMIDiagUnsupportedPrefix << requester->getName() << " operand #"
        << use.getOperandNumber() << " has type " << sourceType
        << " but requires " << resultType
        << "; pto.vmi.ensure_layout cannot materialize this conversion";
    diag.attachNote(ensure.getLoc())
        << "failed helper conversion " << sourceType << " -> " << resultType
        << " (" << reason
        << "); partial/tail layout materialization requires an explicit "
           "packing plan";
    return;
  }

  ensure.emitError()
      << kVMIDiagUnsupportedPrefix
      << "pto.vmi.ensure_layout cannot materialize the requested data "
         "layout conversion ("
      << reason
      << "); partial/tail layout materialization requires an explicit "
         "packing plan";
}

LogicalResult
verifySupportedVMIToVPTOOps(ModuleOp module,
                            bool enableStableGatherMaskedLoad) {
  auto emitMemoryUnsupported =
      [&](Operation *op, StringRef opName, VMIVRegType type, Value source,
          std::optional<int64_t> constantOffset) -> WalkResult {
    std::string reason;
    if (succeeded(checkSupportedLoadShape(type, source,
                                          source.getType(), constantOffset,
                                          &reason)))
      return WalkResult::advance();

    op->emitError() << kVMIDiagUnsupportedPrefix << opName
                    << " direct lowering requires a supported memory source ("
                    << reason << ")";
    return WalkResult::interrupt();
  };

  auto emitMaskableUnsupported = [&](Operation *op, StringRef opName,
                                     VMIVRegType type) -> WalkResult {
    std::string reason;
    if (succeeded(checkSupportedMaskableVReg(type, &reason)))
      return WalkResult::advance();

    op->emitError()
        << kVMIDiagUnsupportedPrefix << opName
        << " direct lowering requires physical vreg parts with b8/b16/b32 "
           "predicate masks ("
        << reason << ")";
    return WalkResult::interrupt();
  };

  WalkResult result = module.walk([&](Operation *op) {
    if (auto constant = dyn_cast<VMIConstantOp>(op)) {
      auto denseAttr = dyn_cast<DenseElementsAttr>(constant.getValue());
      if (!denseAttr || !denseAttr.isSplat()) {
        constant.emitError()
            << kVMIDiagUnsupportedPrefix
            << "non-splat pto.vmi.constant requires a vreg immediate or "
               "scratch materialization plan";
        return WalkResult::interrupt();
      }
      return emitMaskableUnsupported(
          op, "pto.vmi.constant",
          cast<VMIVRegType>(constant.getResult().getType()));
    }

    if (auto broadcast = dyn_cast<VMIBroadcastOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.broadcast",
          cast<VMIVRegType>(broadcast.getResult().getType()));
    if (auto broadcast = dyn_cast<VMIGroupBroadcastOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedGroupBroadcastShape(broadcast,
                                                      &reason)))
        return WalkResult::advance();
      broadcast.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.group_broadcast requires full source chunks with "
             "#pto.vmi.layout<num_groups = G, slots = K>, a dense full result "
             "layout, "
             "and num_groups deriving a group size that divides or is a "
             "multiple of physical chunk lanes ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto hist = dyn_cast<VMIVdhistOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedVdhistShape(hist, &reason)))
        return WalkResult::advance();
      hist.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.vdhist requires contiguous Nx{ui8|i8} source, contiguous b8 "
             "mask, and contiguous 256x{ui16|i16} acc/result ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto hist = dyn_cast<VMIVchistOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedVchistShape(hist, &reason)))
        return WalkResult::advance();
      hist.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.vchist requires contiguous Nx{ui8|i8} source, contiguous b8 "
             "mask, and contiguous 256x{ui16|i16} acc/result ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto load = dyn_cast<VMILoadOp>(op)) {
      return emitMemoryUnsupported(
          op, "pto.vmi.load", cast<VMIVRegType>(load.getResult().getType()),
          load.getSource(), getConstantIndexValue(load.getOffset()));
    }
    if (auto load = dyn_cast<VMIDeinterleaveLoadOp>(op)) {
      std::string reason;
      if (succeeded(
              checkSupportedDeinterleaveLoadShape(load, &reason)))
        return WalkResult::advance();
      load.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.deinterleave_load lowers through pto.vldsx2 only for "
             "matching contiguous full low/high result chunks with a supported "
             "UB source and 8/16/32-bit element type ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto load = dyn_cast<VMIStrideLoadOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedStrideLoadShape(load, &reason)))
        return WalkResult::advance();
      load.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.stride_load lowers through pto.vsldb only for one "
             "contiguous physical result/mask chunk and a supported UB source ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto load = dyn_cast<VMIGroupLoadOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedGroupLoadShape(load, &reason)))
        return WalkResult::advance();
      load.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.group_load requires contiguous full result chunks, a "
             "supported UB source, and num_groups deriving a group size "
             "aligned to physical chunks ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto load = dyn_cast<VMIGroupSlotLoadOp>(op)) {
      std::string reason;
      if (succeeded(
              checkSupportedGroupSlotLoadShape(load, &reason)))
        return WalkResult::advance();
      load.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.group_slot_load requires explicit group_slots result "
             "layout matching num_groups, a supported UB pointer source, "
             "and either slots=8 with constant unit source_group_stride or "
             "slots=1 row-local lowering ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto load = dyn_cast<VMIGroupBroadcastLoadOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedGroupBroadcastLoadShape(load,
                                                          &reason)))
        return WalkResult::advance();
      load.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.group_broadcast_load requires either the BRC full-group "
             "chunk form, the E2B packet form for b16/b32 direct or split "
             "group size, or the generic group-slot-load then group-broadcast "
             "fallback with supported UB pointer source and source_group_stride ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto load = dyn_cast<VMIMaskedLoadOp>(op)) {
      if (enableStableGatherMaskedLoad) {
        load.emitError()
            << kVMIDiagUnsupportedPrefix
            << "pto.vmi.masked_load stable VGATHER-based lowering is reserved "
               "for strict masked/tail loads but is not implemented yet";
        return WalkResult::interrupt();
      }
      std::string reason;
      if (succeeded(checkSupportedMaskedLoadShape(load, &reason)))
        return WalkResult::advance();
      load.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.masked_load direct lowering requires a supported memory "
             "source, contiguous result/passthru/mask layouts, and either "
             "full physical chunks or a statically safe full-read footprint ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto gather = dyn_cast<VMIGatherOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedGatherShape(gather, &reason)))
        return WalkResult::advance();
      gather.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.gather lowers through pto.vgather2_bc + pto.vsel only "
             "for UB pointer sources, contiguous full physical chunks, "
             "32-bit result elements, i32 indices, and b32 masks ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto load = dyn_cast<VMIExpandLoadOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedExpandLoadShape(load, &reason)))
        return WalkResult::advance();
      load.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.expand_load direct lowering is currently supported for "
             "either a static all-active mask lowered as pto.vlds, or a "
             "one-full-chunk 32-bit UB runtime mask lowered through pto.vusqz "
             "+ pto.vgather2_bc + pto.vsel ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto store = dyn_cast<VMIStoreOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedStoreShape(cast<VMIVRegType>(store.getValue().getType()),
              store.getDestination(), store.getDestination().getType(),
              &reason)))
        return WalkResult::advance();
      store.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.store requires an 8/16/32-bit predicate-maskable "
             "element type and either full physical chunks or contiguous "
             "tail-store layout, with UB-backed destination ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto store = dyn_cast<VMIInterleaveStoreOp>(op)) {
      std::string reason;
      if (succeeded(
              checkSupportedInterleaveStoreShape(store, &reason)))
        return WalkResult::advance();
      store.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.interleave_store lowers through pto.vstsx2 only for "
             "matching contiguous full low/high input chunks with a supported "
             "UB destination and 8/16/32-bit element type ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto store = dyn_cast<VMIGroupStoreOp>(op)) {
      std::string reason;
      if (succeeded(
              checkSupportedGroupStoreShape(store, &reason)))
        return WalkResult::advance();
      store.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.group_store requires contiguous full value chunks, a "
             "supported UB destination, and num_groups deriving a group size "
             "aligned to physical chunks ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto store = dyn_cast<VMIMaskedStoreOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedMaskedStoreShape(cast<VMIVRegType>(store.getValue().getType()),
              cast<VMIMaskType>(store.getMask().getType()),
              store.getDestination(), store.getDestination().getType(),
              &reason)))
        return WalkResult::advance();
      store.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.masked_store requires either full physical chunks or "
             "contiguous tail-store value/mask layout, with UB-backed "
             "destination ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto store = dyn_cast<VMIStrideStoreOp>(op)) {
      std::string reason;
      if (succeeded(
              checkSupportedStrideStoreShape(store, &reason)))
        return WalkResult::advance();
      store.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.stride_store lowers through pto.vsstb only for one "
             "contiguous physical value/mask chunk and a supported UB "
             "destination ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto scatter = dyn_cast<VMIScatterOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedScatterShape(scatter, &reason)))
        return WalkResult::advance();
      scatter.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.scatter lowers through pto.vscatter only with a UB "
             "pointer destination, contiguous full physical chunks, 32-bit "
             "value elements, i32 indices, and b32 masks ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto ensure = dyn_cast<VMIEnsureLayoutOp>(op)) {
      auto sourceType = cast<VMIVRegType>(ensure.getSource().getType());
      auto resultType = cast<VMIVRegType>(ensure.getResult().getType());
      std::string reason;
      VMILayoutSupport supports;
      if (succeeded(
              supports.getEnsureLayoutFact(sourceType, resultType, &reason)))
        return WalkResult::advance();

      emitEnsureLayoutMaterializationError(ensure, sourceType, resultType,
                                           reason);
      return WalkResult::interrupt();
    }

    if (auto ensure = dyn_cast<VMIEnsureMaskLayoutOp>(op)) {
      auto sourceType = cast<VMIMaskType>(ensure.getSource().getType());
      auto resultType = cast<VMIMaskType>(ensure.getResult().getType());
      std::string reason;
      VMILayoutSupport supports;
      if (succeeded(supports.getEnsureMaskLayoutFact(sourceType, resultType,
                                                     &reason)))
        return WalkResult::advance();

      ensure.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.ensure_mask_layout cannot materialize the requested "
             "mask layout conversion ("
          << reason
          << "); partial/tail predicate layout materialization requires an "
             "explicit packing plan";
      return WalkResult::interrupt();
    }

    if (auto ensure = dyn_cast<VMIEnsureMaskGranularityOp>(op)) {
      auto sourceType = cast<VMIMaskType>(ensure.getSource().getType());
      auto resultType = cast<VMIMaskType>(ensure.getResult().getType());
      bool identity =
          sourceType.getGranularity() == resultType.getGranularity() &&
          sourceType.getLayoutAttr() == resultType.getLayoutAttr();
      if (!identity) {
        VMILayoutSupport supports;
        std::string reason;
        if (failed(supports.getMaskGranularityCastLayoutFactForLayouts(
                sourceType, resultType, sourceType.getLayoutAttr(),
                resultType.getLayoutAttr(), &reason))) {
          ensure.emitError()
              << kVMIDiagUnsupportedPrefix
              << "mask granularity cast layout relation is unsupported ("
              << reason << ")";
          return WalkResult::interrupt();
        }
      }

      return WalkResult::advance();
    }

    if (auto addf = dyn_cast<VMIAddFOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.addf", cast<VMIVRegType>(addf.getResult().getType()));
    if (auto addi = dyn_cast<VMIAddIOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.addi", cast<VMIVRegType>(addi.getResult().getType()));
    if (auto subf = dyn_cast<VMISubFOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.subf", cast<VMIVRegType>(subf.getResult().getType()));
    if (auto subi = dyn_cast<VMISubIOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.subi", cast<VMIVRegType>(subi.getResult().getType()));
    if (auto mulf = dyn_cast<VMIMulFOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.mulf", cast<VMIVRegType>(mulf.getResult().getType()));
    if (auto muli = dyn_cast<VMIMulIOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.muli", cast<VMIVRegType>(muli.getResult().getType()));
    if (auto vmull = dyn_cast<VMIVmullOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedVmullShape(vmull, &reason)))
        return WalkResult::advance();
      vmull.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.vmull requires equal 64/128/256-lane i32/ui32 data "
             "ports, a matching b32 mask, and contiguous or deinterleaved "
             "factor-2/factor-4 lane_stride=1 layout ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto divf = dyn_cast<VMIDivFOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.divf", cast<VMIVRegType>(divf.getResult().getType()));
    if (auto minf = dyn_cast<VMIMinFOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.minf", cast<VMIVRegType>(minf.getResult().getType()));
    if (auto maxf = dyn_cast<VMIMaxFOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.maxf", cast<VMIVRegType>(maxf.getResult().getType()));
    if (auto negf = dyn_cast<VMINegFOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.negf", cast<VMIVRegType>(negf.getResult().getType()));
    if (auto absf = dyn_cast<VMIAbsFOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.absf", cast<VMIVRegType>(absf.getResult().getType()));
    if (auto absi = dyn_cast<VMIAbsIOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.absi", cast<VMIVRegType>(absi.getResult().getType()));
    if (auto sqrt = dyn_cast<VMISqrtOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.sqrt", cast<VMIVRegType>(sqrt.getResult().getType()));
    if (auto exp = dyn_cast<VMIExpOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.exp", cast<VMIVRegType>(exp.getResult().getType()));
    if (auto ln = dyn_cast<VMILnOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.ln", cast<VMIVRegType>(ln.getResult().getType()));
    if (auto relu = dyn_cast<VMIReluOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedReluShape(relu, &reason)))
        return WalkResult::advance();
      relu.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.relu direct lowering requires physical vreg parts with "
             "b8/b16/b32 predicate masks and f16/f32 element type ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto andi = dyn_cast<VMIAndIOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.andi", cast<VMIVRegType>(andi.getResult().getType()));
    if (auto ori = dyn_cast<VMIOrIOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.ori", cast<VMIVRegType>(ori.getResult().getType()));
    if (auto xori = dyn_cast<VMIXOrIOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.xori", cast<VMIVRegType>(xori.getResult().getType()));
    if (auto shli = dyn_cast<VMIShLIOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.shli", cast<VMIVRegType>(shli.getResult().getType()));
    if (auto shrui = dyn_cast<VMIShRUIOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.shrui", cast<VMIVRegType>(shrui.getResult().getType()));
    if (auto shrsi = dyn_cast<VMIShRSIOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.shrsi", cast<VMIVRegType>(shrsi.getResult().getType()));
    if (auto notOp = dyn_cast<VMINotOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.not", cast<VMIVRegType>(notOp.getResult().getType()));
    if (auto select = dyn_cast<VMISelectOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.select",
          cast<VMIVRegType>(select.getResult().getType()));

    if (auto cmpf = dyn_cast<VMICmpFOp>(op)) {
      WalkResult physical = emitMaskableUnsupported(
          op, "pto.vmi.cmpf", cast<VMIVRegType>(cmpf.getLhs().getType()));
      if (physical.wasInterrupted())
        return physical;
      if (succeeded(checkSupportedComparePredicate<VMICmpFOp>(
              op, cmpf.getPredicate())))
        return WalkResult::advance();
      return WalkResult::interrupt();
    }

    if (auto cmpi = dyn_cast<VMICmpIOp>(op)) {
      WalkResult physical = emitMaskableUnsupported(
          op, "pto.vmi.cmpi", cast<VMIVRegType>(cmpi.getLhs().getType()));
      if (physical.wasInterrupted())
        return physical;
      if (succeeded(checkSupportedComparePredicate<VMICmpIOp>(
              op, cmpi.getPredicate())))
        return WalkResult::advance();
      return WalkResult::interrupt();
    }

    if (auto activePrefix = dyn_cast<VMIActivePrefixIndexOp>(op)) {
      std::string reason;
      if (succeeded(
              checkSupportedActivePrefixIndexShape(activePrefix, &reason)))
        return WalkResult::advance();
      activePrefix.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.active_prefix_index lowers through pto.vusqz only for "
             "one contiguous physical chunk ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto compress = dyn_cast<VMICompressOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedCompressShape(compress, &reason)))
        return WalkResult::advance();
      compress.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.compress lowers through pto.vsqz only for one "
             "contiguous full physical chunk ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto compressStore = dyn_cast<VMICompressStoreOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedCompressStoreShape(compressStore, &reason)))
        return WalkResult::advance();
      compressStore.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.compress_store lowers through pto.vsqz + pto.vstur "
             "only for one contiguous full physical chunk with a UB pointer "
             "destination ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto reduce = dyn_cast<VMIReduceAddIOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedReduceShape(
              reduce, /*requiresReassoc=*/false, &reason)))
        return WalkResult::advance();
      reduce.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.reduce_addi lowers through pto.vcadd only for "
             "contiguous full 32-bit integer source chunks with matching "
             "mask chunks and one init/result chunk ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto reduce = dyn_cast<VMIReduceAddFOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedReduceShape(
              reduce, /*requiresReassoc=*/true, &reason)))
        return WalkResult::advance();
      reduce.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.reduce_addf lowers through pto.vcadd only with "
             "reassoc, f32 contiguous full source chunks, matching mask "
             "chunks, and one init/result chunk ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto reduce = dyn_cast<VMIGroupReduceAddFOp>(op)) {
      std::string reason;
      if (succeeded(
              checkSupportedGroupReduceShape(reduce, &reason)))
        return WalkResult::advance();
      reduce.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.group_reduce_addf lowers through pto.vcgadd for 32B "
             "blocks or through pto.vcadd for contiguous full "
             "source/mask chunks, #pto.vmi.layout<num_groups = G, slots = K> "
             "result "
             "chunks, and num_groups deriving a group size aligned to "
             "physical chunks ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto reduce = dyn_cast<VMIGroupReduceAddIOp>(op)) {
      std::string reason;
      if (succeeded(
              checkSupportedGroupReduceShape(reduce, &reason)))
        return WalkResult::advance();
      reduce.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.group_reduce_addi lowers through pto.vcgadd/vadd for "
             "supported 32B block classes or through an internal widening "
             "pto.vcadd path for aligned full chunks ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto reduce = dyn_cast<VMIGroupReduceMaxIOp>(op)) {
      std::string reason;
      if (succeeded(
              checkSupportedGroupReduceShape(reduce, &reason)))
        return WalkResult::advance();
      reduce.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.group_reduce_maxi lowers through pto.vcgmax/vmax for "
             "supported 32B block classes or through pto.vcmax for aligned "
             "full chunks ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto reduce = dyn_cast<VMIGroupReduceMaxFOp>(op)) {
      std::string reason;
      if (succeeded(
              checkSupportedGroupReduceShape(reduce, &reason)))
        return WalkResult::advance();
      reduce.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.group_reduce_maxf lowers through pto.vcgmax/vmax for "
             "32B blocks or through pto.vcmax for contiguous full chunks, "
             "matching source/mask chunks, "
             "#pto.vmi.layout<num_groups = G, slots = K> result chunks, and "
             "num_groups deriving a group size aligned to physical chunks ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto reduce = dyn_cast<VMIGroupReduceMinFOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedGroupReduceShape(reduce, &reason)))
        return WalkResult::advance();
      reduce.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.group_reduce_minf lowers through pto.vcgmin/vmin for "
             "supported 32B block classes or through pto.vcmin for aligned "
             "full chunks ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto reduce = dyn_cast<VMIGroupReduceMinIOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedGroupReduceShape(reduce, &reason)))
        return WalkResult::advance();
      reduce.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.group_reduce_mini lowers through pto.vcgmin/vmin for "
             "supported 32B block classes or through pto.vcmin for aligned "
             "full chunks ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto reduce = dyn_cast<VMIReduceMaxFOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedReduceShape(
              reduce, /*requiresReassoc=*/false, &reason)))
        return WalkResult::advance();
      reduce.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.reduce_maxf lowers through pto.vcmax only for f16/f32 "
             "contiguous full source chunks with matching mask chunks and one "
             "init/result chunk ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto reduce = dyn_cast<VMIReduceMinFOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedReduceShape(
              reduce, /*requiresReassoc=*/false, &reason)))
        return WalkResult::advance();
      reduce.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.reduce_minf lowers through pto.vcmin only for f16/f32 "
             "contiguous full source chunks with matching mask chunks and one "
             "init/result chunk ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto reduce = dyn_cast<VMIReduceMaxIOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedReduceShape(
              reduce, /*requiresReassoc=*/false, &reason)))
        return WalkResult::advance();
      reduce.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.reduce_maxi lowers through pto.vcmax only for "
             "contiguous full integer source chunks with matching mask "
             "chunks and one init/result chunk ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto reduce = dyn_cast<VMIReduceMinIOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedReduceShape(
              reduce, /*requiresReassoc=*/false, &reason)))
        return WalkResult::advance();
      reduce.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.reduce_mini lowers through pto.vcmin only for "
             "contiguous full integer source chunks with matching mask "
             "chunks and one init/result chunk ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto fma = dyn_cast<VMIFmaOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedFmaShape(fma, &reason)))
        return WalkResult::advance();
      fma.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.fma lowers through pto.vmula only for f16/bf16/f32 "
             "element types ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto extf = dyn_cast<VMIExtFOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedExtFShape(extf, &reason)))
        return WalkResult::advance();

      extf.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.extf supports contiguous 16-bit float-like or fp8-like "
             "physical source chunks to f32 deinterleaved=2/4 results; "
             "partial/tail is allowed only when source padding maps to result "
             "padding ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto truncf = dyn_cast<VMITruncFOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedTruncFShape(truncf, &reason)))
        return WalkResult::advance();

      truncf.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.truncf supports f32/f16/bf16 source narrowing (dense "
             "EvenOdd, Packed4, or f32 group_slots(num_groups=G, slots=1) to "
             "f16 group_slots(num_groups=G, slots=1)); non-f32 sources "
             "currently require dense contiguous layouts ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto fptosi = dyn_cast<VMIFPToSIOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedFPToSIShape(fptosi, &reason)))
        return WalkResult::advance();

      fptosi.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.fptosi supports f32 source chunks to matching 32-bit "
             "integer result chunks with identical assigned layouts ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto sitofp = dyn_cast<VMISIToFPOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedSIToFPShape(sitofp, &reason)))
        return WalkResult::advance();

      sitofp.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.sitofp supports 32-bit integer source chunks to "
             "matching f32 result chunks with identical assigned layouts ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto extsi = dyn_cast<VMIExtSIOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedExtSIShape(extsi, &reason)))
        return WalkResult::advance();

      extsi.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.extsi supports contiguous signed/signless 8-bit or "
             "16-bit integer physical source chunks to 2x/4x wider integer "
             "deinterleaved results, or matching "
             "group_slots(num_groups=G, slots=8) 8/16-bit integer source to "
             "32-bit integer result ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto extui = dyn_cast<VMIExtUIOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedExtUIShape(extui, &reason)))
        return WalkResult::advance();

      extui.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.extui supports contiguous unsigned 8-bit or 16-bit "
             "integer physical source chunks to 2x/4x wider unsigned integer "
             "deinterleaved results, or matching "
             "group_slots(num_groups=G, slots=8) 8/16-bit integer source to "
             "32-bit integer result ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto trunci = dyn_cast<VMITruncIOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedTruncIShape(trunci, &reason)))
        return WalkResult::advance();

      trunci.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.trunci supports integer deinterleaved source layouts "
             "whose factor is the 2x/4x narrowing multiple of the contiguous "
             "or deinterleaved result layout factor, or 32-bit integer "
             "group_slots(num_groups=G, slots=1 or 8) to 8/16-bit integer "
             "group_slots(num_groups=G, slots=1 or 8), or 16-bit unsigned "
             "integer group_slots(num_groups=G, slots=8) to 8-bit unsigned "
             "integer group_slots(num_groups=G, slots=8, lane_stride=2) ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto bitcast = dyn_cast<VMIBitcastOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedBitcastShape(bitcast, &reason)))
        return WalkResult::advance();

      bitcast.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.bitcast requires matching source/result layouts with "
             "width-changing forms restricted to supported layout table rows ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto split = dyn_cast<VMIChannelSplitOp>(op)) {
      int64_t channels = split.getNumResults();
      std::string reason;
      if (succeeded(
              checkSupportedChannelSplitShape(split, &reason)))
        return WalkResult::advance();

      if (channels != 2 && channels != 4)
        split.emitError()
            << kVMIDiagUnsupportedPrefix
            << "pto.vmi.channel_split supports only 2 or 4 channels";
      else
        split.emitError()
            << kVMIDiagUnsupportedPrefix
            << "pto.vmi.channel_split requires source layout to be contiguous "
               "or matching deinterleaved channel layout, every result layout "
               "to be contiguous, and complete physical channel groups ("
            << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto merge = dyn_cast<VMIChannelMergeOp>(op)) {
      int64_t channels = merge.getInputs().size();
      std::string reason;
      if (succeeded(
              checkSupportedChannelMergeShape(merge, &reason)))
        return WalkResult::advance();

      if (channels != 2 && channels != 4)
        merge.emitError()
            << kVMIDiagUnsupportedPrefix
            << "pto.vmi.channel_merge supports only 2 or 4 channels";
      else
        merge.emitError()
            << kVMIDiagUnsupportedPrefix
            << "pto.vmi.channel_merge requires every input layout to be "
               "contiguous and result layout to be contiguous or matching "
               "deinterleaved channel layout, with complete physical channel "
               "groups ("
            << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto shuffle = dyn_cast<VMIShuffleOp>(op)) {
      std::string reason;
      if (succeeded(computeShuffleForwardingSourceParts(shuffle, &reason)))
        return WalkResult::advance();
      std::string splatReason;
      if (succeeded(computeShuffleLane0SplatSourcePart(shuffle, &splatReason)))
        return WalkResult::advance();
      std::string vselrReason;
      if (succeeded(computeShuffleVselrPlans(shuffle, &vselrReason)))
        return WalkResult::advance();

      shuffle.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.shuffle requires physical chunk forwarding or "
             "lane0 splat or vci-materializable vselr indices (forwarding: "
          << reason << "; lane0 splat: " << splatReason
          << "; vselr: " << vselrReason << ")";
      return WalkResult::interrupt();
    }

    if (auto constantMask = dyn_cast<VMIConstantMaskOp>(op)) {
      std::string reason;
      if (succeeded(computeConstantMaskMaterialization(constantMask, &reason)))
        return WalkResult::advance();

      constantMask.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.constant_mask requires a dense bool constant with "
             "concrete layout and b8/b16/b32 granularity ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

struct VMIToVPTOPass : public mlir::pto::impl::VMIToVPTOBase<VMIToVPTOPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMIToVPTOPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(verifyVMIToVPTOInputIR(module))) {
      signalPassFailure();
      return;
    }
    if (failed(verifySupportedVMIToVPTOOps(module,
                                           enableStableGatherMaskedLoad))) {
      signalPassFailure();
      return;
    }

    MLIRContext *context = module.getContext();
    VMIToVPTOTypeConverter typeConverter;
    RewritePatternSet patterns(context);

    populateVMIConversionPatterns(typeConverter, patterns);
    ConversionTarget target(*context);
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return !isVMIOp(op) && !hasVMIType(op);
    });
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      module.emitError() << kVMIDiagResidualOpPrefix
                         << "failed to convert all VMI ops/types to VPTO";
      signalPassFailure();
      return;
    }
    if (failed(verifyNoResidualVMIIR(module))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMIToVPTOPass() {
  return std::make_unique<VMIToVPTOPass>();
}
