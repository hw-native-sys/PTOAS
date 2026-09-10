// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIToVPTO.cpp - Convert VMI to physical VPTO IR -------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/Analysis/PTOAddressAnalysis.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/IR/VPTOMemoryDist.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VMILayoutSupport.h"
#include "PTO/Transforms/VPTOLowering.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/OneToNFuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <numeric>
#include <type_traits>
#include <tuple>
#include <variant>

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
static int64_t getElementDeinterleaveFactor(VMILayoutAttr layout);
static Type getMemoryElementType(Type type);
static Attribute getMemorySpace(Type type);

static FailureOr<SmallVector<Type>> getConvertedResultTypesOrFailure(
    Operation *op, const TypeConverter &typeConverter);
FailureOr<SmallVector<Type>> getConvertedResultTypes(
    Operation *op, unsigned resultIndex, const TypeConverter &typeConverter);
FailureOr<SmallVector<Type>> getConvertedResultTypes(
    Operation *op, const TypeConverter &typeConverter);
void replaceOpWithFlatConvertedValues(OneToNPatternRewriter &rewriter,
                                      Operation *op, ValueRange flatValues,
                                      TypeConverter &typeConverter);

template <typename ValidateFn>
static LogicalResult validateContiguousParts(
    Operation *op, ValueRange parts, StringRef failureMessage,
    OneToNPatternRewriter &rewriter, ValidateFn &&validate);

template <typename OpTy>
static LogicalResult lowerBinaryPhysicalResults(
    OpTy op, SmallVectorImpl<Value> &results, OneToNPatternRewriter &rewriter,
    TypeConverter &typeConverter);

template <typename OpTy>
static LogicalResult lowerPhysicalBinaryWithCarryResults(
    OpTy op, SmallVectorImpl<Value> &results, SmallVectorImpl<Value> &carries,
    OneToNPatternRewriter &rewriter, TypeConverter &typeConverter);

template <typename OpTy, typename LowerFn>
static LogicalResult lowerPointwisePhysicalParts(
    OpTy op, ArrayRef<Type> resultTypes, StringRef arityMessage,
    OneToNPatternRewriter &rewriter, LowerFn &&lowerFn,
    TypeConverter &typeConverter);

static bool isContiguousVMIVRegPart(Value part);

static FailureOr<std::pair<SmallVector<Type>, SmallVector<Type>>>
getConvertedResultTypesPair(Operation *op, const TypeConverter &typeConverter) {
  FailureOr<SmallVector<Type>> first =
      getConvertedResultTypes(op, 0, typeConverter);
  if (failed(first)) {
    return failure();
  }
  FailureOr<SmallVector<Type>> second =
      getConvertedResultTypes(op, 1, typeConverter);
  if (failed(second)) {
    return failure();
  }
  return std::make_pair(std::move(*first), std::move(*second));
}

template <typename LowerFn>
static LogicalResult lowerWithConvertedResultTypes(
    Operation *op, unsigned resultIndex, const TypeConverter &typeConverter,
    LowerFn &&lowerFn) {
  FailureOr<SmallVector<Type>> resultTypes =
      getConvertedResultTypes(op, resultIndex, typeConverter);
  if (failed(resultTypes)) {
    return failure();
  }
  return lowerFn(*resultTypes);
}

template <typename ResultT>
static FailureOr<ResultT> emitFailure(std::string *reason,
                                      const Twine &message) {
  if (reason) {
    *reason = message.str();
  }
  return failure();
}

static LogicalResult emitLogicalFailure(std::string *reason,
                                        const Twine &message) {
  if (reason) {
    *reason = message.str();
  }
  return failure();
}

struct AssignedValueMaskLayouts {
  VMILayoutAttr value;
  VMILayoutAttr mask;
};

static FailureOr<AssignedValueMaskLayouts> getAssignedValueMaskLayouts(
    VMIVRegType valueType, VMIMaskType maskType, std::string *reason) {
  VMILayoutAttr valueLayout = valueType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  if (!valueLayout || !maskLayout) {
    return emitFailure<AssignedValueMaskLayouts>(
        reason, "requires assigned value and mask layouts");
  }
  return AssignedValueMaskLayouts{valueLayout, maskLayout};
}

static LogicalResult replacePhysicalResults(
    OneToNPatternRewriter &rewriter, Operation *op,
    SmallVectorImpl<Value> &results, TypeConverter &typeConverter) {
  replaceOpWithFlatConvertedValues(rewriter, op, results, typeConverter);
  return success();
}

static LogicalResult replaceSinglePhysicalResult(
    OneToNPatternRewriter &rewriter, Operation *op, Value result,
    TypeConverter &typeConverter) {
  SmallVector<Value> results{result};
  return replacePhysicalResults(rewriter, op, results, typeConverter);
}

static LogicalResult replaceMaterializedResults(
    OneToNPatternRewriter &rewriter, Operation *op,
    FailureOr<SmallVector<Value>> results, TypeConverter &typeConverter) {
  if (failed(results)) {
    return failure();
  }
  return replacePhysicalResults(rewriter, op, *results, typeConverter);
}

template <typename OpTy, typename MaterializeFn>
static LogicalResult lowerMaterializedResults(
    OpTy op, TypeConverter &typeConverter,
    OneToNPatternRewriter &rewriter, MaterializeFn &&materializeFn) {
  return replaceMaterializedResults(
      rewriter, op, materializeFn(), typeConverter);
}

// Shared bundle for the common "convert source vreg -> physical parts -> flat
// result types" head of one-to-N lowering patterns whose source and result are
// both VMI vregs.
struct VMIPhysicalConversionInput {
  VMIVRegType sourceVMIType;
  VMIVRegType resultVMIType;
  ValueRange sourceParts;
  SmallVector<Type> resultTypes;
};

template <typename OpTy>
static FailureOr<VMIPhysicalConversionInput> getVMIPhysicalConversionInput(
    OpTy op, typename OneToNOpConversionPattern<OpTy>::OpAdaptor adaptor,
    const TypeConverter &typeConverter) {
  VMIPhysicalConversionInput input;
  input.sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
  input.resultVMIType = cast<VMIVRegType>(op.getResult().getType());
  input.sourceParts = adaptor.getSource();
  FailureOr<SmallVector<Type>> resultTypes =
      getConvertedResultTypes(op, 0, typeConverter);
  if (failed(resultTypes)) {
    return failure();
  }
  input.resultTypes = std::move(*resultTypes);
  return input;
}

// Resolves the (result, carry) type pair together with the result/carry value
// vectors, lets lowerFn fill them per physical part, then emits the carry
// results after the data results.
template <typename OpTy, typename LowerFn>
static LogicalResult lowerCarryResultParts(
    OpTy op, OneToNPatternRewriter &rewriter, TypeConverter &typeConverter,
    LowerFn &&lowerFn) {
  FailureOr<std::pair<SmallVector<Type>, SmallVector<Type>>> convertedTypes =
      getConvertedResultTypesPair(op, typeConverter);
  if (failed(convertedTypes)) {
    return failure();
  }
  SmallVector<Type> resultTypes = std::move(convertedTypes->first);
  SmallVector<Type> carryTypes = std::move(convertedTypes->second);
  SmallVector<Value> results;
  SmallVector<Value> carries;
  if (failed(lowerFn(resultTypes, carryTypes, results, carries))) {
    return failure();
  }
  return lowerPhysicalBinaryWithCarryResults(op, results, carries, rewriter,
                                             typeConverter);
}

static LogicalResult emitStatefulStoreStream(Operation *op, Value base,
                                              ValueRange values,
                                              ArrayRef<int64_t> advances,
                                              OneToNPatternRewriter &rewriter) {
  bool invalidStreamShape = values.empty() || values.size() != advances.size();
  if (invalidStreamShape) {
    return rewriter.notifyMatchFailure(
        op, "unaligned store stream requires matching non-empty values and "
            "advances");
  }
  if (llvm::any_of(advances, [](int64_t advance) {
        return advance <= 0 || !llvm::isInt<32>(advance);
      })) {
    return rewriter.notifyMatchFailure(
        op, "unaligned store stream requires positive 32-bit advances");
  }

  Value align = rewriter
                    .create<InitAlignOp>(op->getLoc(),
                                         AlignType::get(rewriter.getContext()))
                    .getResult();
  Value currentBase = base;
  for (auto [value, advance] : llvm::zip_equal(values, advances)) {
    Value advanceValue =
        rewriter.create<arith::ConstantIntOp>(op->getLoc(), advance, 32);
    auto store = rewriter.create<VstusOp>(op->getLoc(), align.getType(),
                                          currentBase.getType(), align,
                                          advanceValue, value, currentBase);
    align = store.getAlignOut();
    currentBase = store.getBaseOut();
  }

  Value zero = rewriter.create<arith::ConstantIntOp>(op->getLoc(), 0, 32);
  rewriter.create<VstasOp>(op->getLoc(), /*updated_base=*/Type{}, align,
                           currentBase, zero);
  return success();
}

static LogicalResult emitGroupStoreStream(Operation *op, Value destination,
                                           Value offset, ValueRange values,
                                           ArrayRef<int64_t> advances,
                                           OneToNPatternRewriter &rewriter) {
  Type destinationElementType = getMemoryElementType(destination.getType());
  Value storeBase = materializeBufferPointer(
      destination, destinationElementType,
      getMemorySpace(destination.getType()), rewriter, op->getLoc());
  if (!storeBase) {
    return rewriter.notifyMatchFailure(
        op, "unaligned group_store requires a ptr-compatible destination");
  }
  storeBase = rewriter
                  .create<AddPtrOp>(op->getLoc(), storeBase.getType(),
                                    storeBase, offset)
                  .getResult();
  return emitStatefulStoreStream(op, storeBase, values, advances, rewriter);
}

bool isVMIType(Type type) { return isa<VMIVRegType, VMIMaskType>(type); }

bool containsVMIType(Type type) {
  if (isVMIType(type)) {
    return true;
  }

  if (auto functionType = dyn_cast<FunctionType>(type)) {
    return llvm::any_of(functionType.getInputs(),
                        [](Type input) { return containsVMIType(input); }) ||
           llvm::any_of(functionType.getResults(),
                        [](Type result) { return containsVMIType(result); });
  }

  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return containsVMIType(shapedType.getElementType());
  }

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
    if (supported) {
      return mlir::success();
    }
    if (outReason) {
      *outReason = reason;
    }
    return mlir::failure();
  }
};

bool hasVMIType(FunctionType type) {
  return hasVMIType(type.getInputs()) || hasVMIType(type.getResults());
}

bool hasVMIType(Attribute attr) {
  if (!attr) {
    return false;
  }

  if (auto typeAttr = dyn_cast<TypeAttr>(attr)) {
    if (containsVMIType(typeAttr.getValue())) {
      return true;
    }
  }

  if (auto typedAttr = dyn_cast<TypedAttr>(attr)) {
    if (containsVMIType(typedAttr.getType())) {
      return true;
    }
  }

  if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
    return llvm::any_of(arrayAttr,
                        [](Attribute element) { return hasVMIType(element); });
  }

  if (auto dictAttr = dyn_cast<DictionaryAttr>(attr)) {
    return llvm::any_of(dictAttr, [](NamedAttribute namedAttr) {
      return hasVMIType(namedAttr.getValue());
    });
  }

  return false;
}

bool hasVMIType(Operation *op) {
  if (auto func = dyn_cast<func::FuncOp>(op)) {
    if (hasVMIType(func.getFunctionType())) {
      return true;
    }
  }
  bool operationHasVMIType = hasVMIType(op->getOperandTypes()) ||
                             hasVMIType(op->getResultTypes());
  if (operationHasVMIType) {
    return true;
  }
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      if (hasVMIType(block.getArgumentTypes())) {
        return true;
      }
    }
  }
  for (NamedAttribute attr : op->getAttrs()) {
    if (hasVMIType(attr.getValue())) {
      return true;
    }
  }
  return false;
}

bool isVMIPackedFloatCarrierType(Type type) {
  return pto::isPTOHiFloat8x2Type(type) ||
         pto::isPTOFloat4PackedType(type) ||
         pto::isPTOBF16x2Type(type);
}

bool isVMIOp(Operation *op) {
  return op->getName().getStringRef().starts_with("pto.vmi.");
}

StringRef getTruncFRoundModeForResult(Type resultElementType) {
  return pto::isPTOHiFloat8Type(resultElementType) ? "A" : "R";
}

StringRef getTruncFRoundMode(VMITruncFOp op, Type resultElementType) {
  if (auto roundingAttr = op->getAttrOfType<StringAttr>("rounding")) {
    return roundingAttr.getValue();
  }
  return getTruncFRoundModeForResult(resultElementType);
}

bool isLayoutAssignedVMIType(Type type) {
  if (auto vregType = dyn_cast<VMIVRegType>(type)) {
    return static_cast<bool>(vregType.getLayoutAttr());
  }
  if (auto maskType = dyn_cast<VMIMaskType>(type)) {
    return maskType.getLayoutAttr() &&
           VMIMaskType::isConcreteGranularity(maskType.getGranularity());
  }
  return true;
}

LogicalResult verifyLayoutAssignedVMITypeTree(Operation *op, Type type) {
  if (!isLayoutAssignedVMIType(type)) {
    return op->emitError() << kVMIDiagPassInvariantPrefix
                           << "vmi-to-vpto requires layout-assigned VMI types";
  }

  if (auto functionType = dyn_cast<FunctionType>(type)) {
    for (Type input : functionType.getInputs()) {
      if (failed(verifyLayoutAssignedVMITypeTree(op, input))) {
        return failure();
      }
    }
    for (Type result : functionType.getResults()) {
      if (failed(verifyLayoutAssignedVMITypeTree(op, result))) {
        return failure();
      }
    }
  }

  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return verifyLayoutAssignedVMITypeTree(op, shapedType.getElementType());
  }

  return success();
}

LogicalResult verifyVMIToVPTOInputAttribute(Operation *op, Attribute attr) {
  if (!attr) {
    return success();
  }

  if (auto typeAttr = dyn_cast<TypeAttr>(attr)) {
    if (failed(verifyLayoutAssignedVMITypeTree(op, typeAttr.getValue()))) {
      return failure();
    }
  }

  if (auto typedAttr = dyn_cast<TypedAttr>(attr)) {
    if (failed(verifyLayoutAssignedVMITypeTree(op, typedAttr.getType()))) {
      return failure();
    }
  }

  if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
    for (Attribute element : arrayAttr) {
      if (failed(verifyVMIToVPTOInputAttribute(op, element))) {
        return failure();
      }
    }
  }

  if (auto dictAttr = dyn_cast<DictionaryAttr>(attr)) {
    for (NamedAttribute namedAttr : dictAttr) {
      if (failed(verifyVMIToVPTOInputAttribute(op, namedAttr.getValue()))) {
        return failure();
      }
    }
  }

  return success();
}

LogicalResult verifyVMIToVPTOInputTypes(Operation *op) {
  for (Type type : op->getOperandTypes()) {
    if (failed(verifyLayoutAssignedVMITypeTree(op, type))) {
      return failure();
    }
  }
  for (Type type : op->getResultTypes()) {
    if (failed(verifyLayoutAssignedVMITypeTree(op, type))) {
      return failure();
    }
  }
  if (auto func = dyn_cast<func::FuncOp>(op)) {
    FunctionType functionType = func.getFunctionType();
    for (Type type : functionType.getInputs()) {
      if (failed(verifyLayoutAssignedVMITypeTree(op, type))) {
        return failure();
      }
    }
    for (Type type : functionType.getResults()) {
      if (failed(verifyLayoutAssignedVMITypeTree(op, type))) {
        return failure();
      }
    }
  }
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Type type : block.getArgumentTypes()) {
        if (failed(verifyLayoutAssignedVMITypeTree(op, type))) {
          return failure();
        }
      }
    }
  }
  for (NamedAttribute attr : op->getAttrs()) {
    if (failed(verifyVMIToVPTOInputAttribute(op, attr.getValue()))) {
      return failure();
    }
  }
  return success();
}

LogicalResult verifyVMIToVPTOInputIR(ModuleOp module) {
  WalkResult result = module.walk([](Operation *op) {
    if (auto cast = dyn_cast<UnrealizedConversionCastOp>(op)) {
      bool carriesVMIType = llvm::any_of(cast->getOperandTypes(), isVMIType) ||
                            llvm::any_of(cast->getResultTypes(), isVMIType);
      if (carriesVMIType) {
        cast.emitError()
            << kVMIDiagResidualOpPrefix
            << "unrealized_conversion_cast cannot carry VMI types into "
               "VMI-to-VPTO conversion";
        return WalkResult::interrupt();
      }
    }
    if (failed(verifyVMIToVPTOInputTypes(op))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

static std::optional<Value> materializeVPTOToVMI(OpBuilder &builder,
                                                 Type resultType,
                                                 ValueRange inputs,
                                                 Location loc) {
  if (!isVMIType(resultType)) {
    return std::nullopt;
  }
  return builder.create<VMIPackOp>(loc, resultType, inputs).getResult();
}

static std::optional<SmallVector<Value>>
materializeVMIToVPTO(OpBuilder &builder, TypeRange resultTypes, Value input,
                     Location loc) {
  if (!isVMIType(input.getType())) {
    return std::nullopt;
  }
  auto unpackOp = builder.create<VMIUnpackOp>(loc, resultTypes, input);
  return SmallVector<Value>(unpackOp->getResults());
}

static int64_t getMaskGranularityBits(StringRef granularity) {
  if (granularity == "b8") {
    return 8;
  }
  if (granularity == "b16") {
    return 16;
  }
  if (granularity == "b32") {
    return 32;
  }
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
  if (bits == 0) {
    return failure();
  }

  // VPTO masks are typed by the data element width.  VMI layouts may carry a
  // lane stride for packed data, but that stride does not widen the predicate
  // consumed by a vector instruction (e.g. i16 data still requires mask<b16>).
  StringRef physicalGranularity = getMaskGranularityForBits(bits);
  if (physicalGranularity.empty())
    return failure();
  return physicalGranularity;
}

class VMIToVPTOTypeConverter final : public OneToNTypeConverter {
public:
  VMIToVPTOTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](VMIVRegType type,
                     SmallVectorImpl<Type> &results) -> LogicalResult {
      FailureOr<int64_t> arity = getVMIPhysicalArity(type);
      Type physicalElementType = getVMIPhysicalDataElementType(type);
      if (failed(arity)) {
        return failure();
      }
      FailureOr<int64_t> lanesPerPart =
          getDataLanesPerPart(physicalElementType);
      if (failed(lanesPerPart)) {
        return failure();
      }
      for (int64_t i = 0; i < *arity; ++i) {
        results.push_back(VRegType::get(type.getContext(), *lanesPerPart,
                                        physicalElementType));
      }
      return success();
    });
    addConversion(
        [](VMIMaskType type, SmallVectorImpl<Type> &results) -> LogicalResult {
          FailureOr<int64_t> arity = getVMIPhysicalArity(type);
          FailureOr<StringRef> physicalGranularity =
              getVMIMaskPhysicalGranularity(type);
          bool invalidPhysicalType = failed(arity) || failed(physicalGranularity);
          if (invalidPhysicalType) {
            return failure();
          }
          for (int64_t i = 0; i < *arity; ++i) {
            results.push_back(
                MaskType::get(type.getContext(), *physicalGranularity));
          }
          return success();
        });
    TypeConverter::addSourceMaterialization(materializeVPTOToVMI);
    TypeConverter::addArgumentMaterialization(materializeVPTOToVMI);
    OneToNTypeConverter::addTargetMaterialization(materializeVMIToVPTO);
  }
};

FailureOr<SmallVector<Type>>
getConvertedResultTypes(Operation *op, unsigned resultIndex,
                        const TypeConverter &typeConverter) {
  if (resultIndex >= op->getNumResults()) {
    return failure();
  }
  SmallVector<Type> resultTypes;
  if (failed(typeConverter.convertType(op->getResult(resultIndex).getType(),
                                       resultTypes))) {
    return failure();
  }
  return resultTypes;
}

FailureOr<SmallVector<Type>>
getConvertedResultTypes(Operation *op, const TypeConverter &typeConverter) {
  SmallVector<Type> resultTypes;
  if (failed(typeConverter.convertTypes(op->getResultTypes(), resultTypes))) {
    return failure();
  }
  return resultTypes;
}

static FailureOr<SmallVector<Type>> getConvertedResultTypesOrFailure(
    Operation *op, const TypeConverter &typeConverter) {
  FailureOr<SmallVector<Type>> resultTypes =
      getConvertedResultTypes(op, 0, typeConverter);
  if (failed(resultTypes)) {
    return failure();
  }
  return std::move(*resultTypes);
}

FailureOr<SmallVector<Type>>
getConvertedVRegTypesWithLayout(VMIVRegType type, VMILayoutAttr layout,
                                const TypeConverter &typeConverter) {
  auto relayoutType = VMIVRegType::get(type.getContext(), type.getElementCount(),
                                       type.getElementType(), layout);
  SmallVector<Type> convertedTypes;
  if (failed(typeConverter.convertType(relayoutType, convertedTypes))) {
    return failure();
  }
  return convertedTypes;
}

FailureOr<int64_t> getVRegPhysicalFootprintBytes(TypeRange types) {
  int64_t totalBytes = 0;
  for (Type type : types) {
    auto vregType = dyn_cast<VRegType>(type);
    if (!vregType) {
      return failure();
    }
    unsigned elementBits =
        pto::getPTOStorageElemBitWidth(vregType.getElementType());
    if (elementBits == 0) {
      return failure();
    }
    int64_t chunkBits = vregType.getElementCount() * elementBits;
    if (chunkBits % 8 != 0) {
      return failure();
    }
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
  bool unavailableFootprints =
      failed(assignedBytes) || failed(contiguousBytes);
  if (unavailableFootprints) {
    return failure();
  }
  return *assignedBytes <= *contiguousBytes;
}

void replaceOpWithFlatConvertedValues(
    OneToNPatternRewriter &rewriter, Operation *op, ValueRange flatValues,
    TypeConverter &typeConverter) {
  OneToNTypeMapping resultMapping(op->getResultTypes());
  auto &oneToNTypeConverter =
      static_cast<OneToNTypeConverter &>(typeConverter);
  LogicalResult converted = oneToNTypeConverter.computeTypeMapping(
      op->getResultTypes(), resultMapping);
  assert(succeeded(converted) && "expected converted result types");
  (void)converted;
  rewriter.replaceOp(op, flatValues, resultMapping);
}

SmallVector<Value>
flattenOneToNOperands(ArrayRef<ValueRange> operands) {
  SmallVector<Value> flat;
  for (ValueRange operand : operands) {
    llvm::append_range(flat, operand);
  }
  return flat;
}

FailureOr<Value> createAllTrueMaskForVReg(Location loc, VRegType vregType,
                                          PatternRewriter &rewriter) {
  MLIRContext *ctx = rewriter.getContext();
  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(vregType.getElementType());
  if (elementBits == 8) {
    return rewriter
        .create<PsetB8Op>(loc, MaskType::get(ctx, "b8"),
                          rewriter.getStringAttr("PAT_ALL"))
        .getResult();
  }
  if (elementBits == 16) {
    return rewriter
        .create<PsetB16Op>(loc, MaskType::get(ctx, "b16"),
                           rewriter.getStringAttr("PAT_ALL"))
        .getResult();
  }
  if (elementBits == 32) {
    return rewriter
        .create<PsetB32Op>(loc, MaskType::get(ctx, "b32"),
                           rewriter.getStringAttr("PAT_ALL"))
        .getResult();
  }
  return failure();
}

FailureOr<MaskType> getMaskTypeForVReg(VRegType vregType, MLIRContext *ctx) {
  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(vregType.getElementType());
  if (elementBits == 8) {
    return MaskType::get(ctx, "b8");
  }
  if (elementBits == 16) {
    return MaskType::get(ctx, "b16");
  }
  if (elementBits == 32) {
    return MaskType::get(ctx, "b32");
  }
  return failure();
}

FailureOr<Value> createPatternMask(Location loc, MaskType maskType,
                                   StringRef pattern,
                                   PatternRewriter &rewriter);

FailureOr<Value> createAllTrueMask(Location loc, MaskType maskType,
                                   PatternRewriter &rewriter) {
  return createPatternMask(loc, maskType, "PAT_ALL", rewriter);
}

FailureOr<Value> createPatternMask(Location loc, MaskType maskType,
                                   StringRef pattern,
                                   PatternRewriter &rewriter) {
  StringAttr patternAttr = rewriter.getStringAttr(pattern);
  MLIRContext *ctx = rewriter.getContext();
  if (maskType.isB8()) {
    return rewriter.create<PsetB8Op>(loc, MaskType::get(ctx, "b8"), patternAttr)
        .getResult();
  }
  if (maskType.isB16()) {
    return rewriter
        .create<PsetB16Op>(loc, MaskType::get(ctx, "b16"), patternAttr)
        .getResult();
  }
  if (maskType.isB32()) {
    return rewriter
        .create<PsetB32Op>(loc, MaskType::get(ctx, "b32"), patternAttr)
        .getResult();
  }
  return failure();
}

FailureOr<Value> createPrefixMask(Location loc, MaskType maskType,
                                  StringRef pattern,
                                  PatternRewriter &rewriter) {
  return createPatternMask(loc, maskType, pattern, rewriter);
}

bool areEquivalentReductionMasks(Value lhs, Value rhs) {
  if (lhs == rhs) {
    return true;
  }
  bool differentMaskTypes = lhs.getType() != rhs.getType();
  if (differentMaskTypes) {
    return false;
  }

  Operation *lhsOp = lhs.getDefiningOp();
  Operation *rhsOp = rhs.getDefiningOp();
  bool differentDefiningOps =
      !lhsOp || !rhsOp || lhsOp->getName() != rhsOp->getName();
  if (differentDefiningOps) {
    return false;
  }

  bool isPatternMask =
      isa<PsetB8Op, PsetB16Op, PsetB32Op, PgeB8Op, PgeB16Op, PgeB32Op>(
          lhsOp);
  return isPatternMask && lhsOp->getAttr("pattern") == rhsOp->getAttr("pattern");
}

bool haveEquivalentReductionMasks(ValueRange masks) {
  return !masks.empty() &&
         llvm::all_of(masks.drop_front(), [&masks](Value mask) {
           return areEquivalentReductionMasks(masks.front(), mask);
         });
}

template <typename CombineOpTy>
FailureOr<Value> combineEquivalentMaskedParts(
    Location loc, ValueRange sources, ValueRange masks, VRegType resultType,
    PatternRewriter &rewriter) {
  bool invalidInputs = sources.empty() || sources.size() != masks.size() ||
                       !haveEquivalentReductionMasks(masks);
  if (invalidInputs) {
    return failure();
  }

  Value combined = sources.front();
  for (Value source : sources.drop_front()) {
    combined =
        rewriter
            .create<CombineOpTy>(loc, resultType, combined, source,
                                 masks.front())
            .getResult();
  }
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
  auto fail = [&reason](const Twine &message)
      -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(type.getElementType());
  FailureOr<int64_t> arity = getVMIPhysicalArity(type);
  bool invalidPhysicalParts =
      failed(lanesPerPart) || failed(arity) || *arity < 1;
  if (invalidPhysicalParts) {
    return fail("requires computable non-empty physical vreg parts");
  }

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

std::optional<std::string> getStaticPrefixPattern(int64_t activeLanes);

FailureOr<Value> createPrefixMaskForActiveLanes(Location loc, MaskType maskType,
                                                int64_t activeLanes,
                                                PatternRewriter &rewriter) {
  if (activeLanes <= 0) {
    return createPrefixMask(loc, maskType, "PAT_ALLF", rewriter);
  }

  std::optional<std::string> pattern = getStaticPrefixPattern(activeLanes);
  if (pattern) {
    return createPrefixMask(loc, maskType, *pattern, rewriter);
  }
  FailureOr<std::pair<Value, Value>> dynamicMask = createRuntimePrefixMask(
      loc, maskType, createI32Constant(loc, activeLanes, rewriter), rewriter);
  if (failed(dynamicMask)) {
    return failure();
  }
  return dynamicMask->first;
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
  if (factor == 1) {
    return activeLanesI32;
  }
  int64_t bias = factor - 1 - part;
  Value biased = activeLanesI32;
  if (bias != 0) {
    biased = rewriter.create<arith::AddIOp>(
        loc, biased, createI32Constant(loc, bias, rewriter));
  }
  return rewriter.create<arith::DivUIOp>(
      loc, biased, createI32Constant(loc, factor, rewriter));
}

std::optional<int64_t> getPowerOfTwoLog2(int64_t value) {
  bool isNotPowerOfTwo = value <= 0 || (value & (value - 1)) != 0;
  if (isNotPowerOfTwo) {
    return std::nullopt;
  }
  int64_t log2 = 0;
  while (value > 1) {
    value >>= 1;
    ++log2;
  }
  return log2;
}

std::optional<std::string> getStaticPrefixPattern(int64_t activeLanes) {
  if (activeLanes <= 0) {
    return std::string("PAT_ALLF");
  }
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

std::optional<std::string> getPrefixPattern(int64_t activeLanes,
                                            int64_t lanesPerPart) {
  if (activeLanes <= 0) {
    return std::string("PAT_ALLF");
  }
  if (activeLanes >= lanesPerPart) {
    return std::string("PAT_ALL");
  }
  return getStaticPrefixPattern(activeLanes);
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
  if (!layout) {
    return failure();
  }
  return layout.isDenseSplit() ? layout.getFactor() : 1;
}

FailureOr<int64_t> getDataChunksInPart(VMIVRegType type, int64_t part) {
  FailureOr<int64_t> factor = getDataLayoutFactor(type);
  FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(type.getElementType());
  bool invalidPart = failed(factor) || failed(lanesPerPart) ||
                     (succeeded(factor) && *factor <= 0) ||
                     (succeeded(lanesPerPart) && *lanesPerPart <= 0) ||
                     part < 0 || part >= *factor;
  if (invalidPart) {
    return failure();
  }

  int64_t logicalLanesInPart =
      (type.getElementCount() + *factor - 1 - part) / *factor;
  return ceilDivNonNegative(logicalLanesInPart, *lanesPerPart);
}

FailureOr<int64_t> getDataFlatPartIndex(VMIVRegType type, int64_t part,
                                        int64_t chunk) {
  FailureOr<int64_t> factor = getDataLayoutFactor(type);
  bool invalidPartOrChunk =
      failed(factor) || part < 0 || part >= *factor || chunk < 0;
  if (invalidPartOrChunk) {
    return failure();
  }

  int64_t flatIndex = 0;
  for (int64_t currentPart = 0; currentPart < part; ++currentPart) {
    FailureOr<int64_t> chunks = getDataChunksInPart(type, currentPart);
    if (failed(chunks)) {
      return failure();
    }
    flatIndex += *chunks;
  }

  FailureOr<int64_t> chunks = getDataChunksInPart(type, part);
  bool invalidChunk = failed(chunks) || chunk >= *chunks;
  if (invalidChunk) {
    return failure();
  }
  return flatIndex + chunk;
}

template <typename ChunkCountFn, typename PaddingFn>
LogicalResult validateNoPaddingPhysicalChunks(int64_t factor,
                                              int64_t lanesPerPart,
                                              ChunkCountFn getChunks,
                                              PaddingFn isPadding,
                                              std::string *reason);

FailureOr<int64_t> checkFullDataPhysicalChunks(VMIVRegType type,
                                               std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<int64_t> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(type.getElementType());
  if (failed(lanesPerPart)) {
    return fail("requires known physical lanes per part");
  }

  FailureOr<int64_t> factor = getDataLayoutFactor(type);
  if (failed(factor)) {
    return fail("requires assigned layout");
  }

  LogicalResult valid = validateNoPaddingPhysicalChunks(
      *factor, *lanesPerPart,
      [&](int64_t part) { return getDataChunksInPart(type, part); },
      [&](int64_t part, int64_t chunk, int64_t lane) {
        return isPaddingLane(type, part, chunk, lane);
      },
      reason);
  if (failed(valid)) {
    return failure();
  }
  return *lanesPerPart;
}

FailureOr<int64_t> getVMITypeLayoutFactor(Type type) {
  Attribute layout;
  if (auto vregType = dyn_cast<VMIVRegType>(type)) {
    layout = vregType.getLayout();
  } else if (auto maskType = dyn_cast<VMIMaskType>(type)) {
    layout = maskType.getLayout();
  } else {
    return failure();
  }

  auto layoutAttr = dyn_cast_or_null<VMILayoutAttr>(layout);
  if (!layoutAttr) {
    return failure();
  }
  return layoutAttr.isDenseSplit() ? layoutAttr.getFactor() : 1;
}

FailureOr<int64_t> getVMITypeElementCount(Type type) {
  if (auto vregType = dyn_cast<VMIVRegType>(type)) {
    return vregType.getElementCount();
  }
  if (auto maskType = dyn_cast<VMIMaskType>(type)) {
    return maskType.getElementCount();
  }
  return failure();
}

FailureOr<int64_t> getVMITypeLanesPerPart(Type type) {
  if (auto vregType = dyn_cast<VMIVRegType>(type)) {
    return getDataLanesPerPart(getVMIPhysicalDataElementType(vregType));
  }
  if (auto maskType = dyn_cast<VMIMaskType>(type)) {
    FailureOr<StringRef> physicalGranularity =
        getVMIMaskPhysicalGranularity(maskType);
    if (failed(physicalGranularity)) {
      return failure();
    }
    return getMaskLanesPerPart(*physicalGranularity);
  }
  return failure();
}

FailureOr<int64_t> getVMITypeChunksInPart(Type type, int64_t part) {
  FailureOr<int64_t> elementCount = getVMITypeElementCount(type);
  FailureOr<int64_t> factor = getVMITypeLayoutFactor(type);
  FailureOr<int64_t> lanesPerPart = getVMITypeLanesPerPart(type);
  bool invalidChunkQuery =
      failed(elementCount) || failed(factor) || failed(lanesPerPart) ||
      (succeeded(factor) && *factor <= 0) ||
      (succeeded(lanesPerPart) && *lanesPerPart <= 0) || part < 0 ||
      part >= *factor;
  if (invalidChunkQuery) {
    return failure();
  }

  VMILayoutAttr layout;
  if (auto vregType = dyn_cast<VMIVRegType>(type)) {
    layout = vregType.getLayoutAttr();
  } else if (auto maskType = dyn_cast<VMIMaskType>(type)) {
    layout = maskType.getLayoutAttr();
  }
  if (!layout) {
    return failure();
  }

  int64_t logicalLanesInPart = (*elementCount + *factor - 1 - part) / *factor;
  int64_t laneStride = 1;
  bool denseVRegLayout = isa<VMIVRegType>(type) && layout.isDense();
  if (denseVRegLayout) {
    laneStride = layout.getLaneStride();
  }
  int64_t physicalLanes =
      logicalLanesInPart == 0 ? 0 : (logicalLanesInPart - 1) * laneStride + 1;
  return ceilDivNonNegative(physicalLanes, *lanesPerPart);
}

template <typename ChunkCountFn, typename PaddingFn>
LogicalResult validateNoPaddingPhysicalChunks(int64_t factor,
                                              int64_t lanesPerPart,
                                              ChunkCountFn getChunks,
                                              PaddingFn isPadding,
                                              std::string *reason) {
  auto fail = [&reason](const Twine &message) {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  for (int64_t part = 0; part < factor; ++part) {
    FailureOr<int64_t> chunks = getChunks(part);
    if (failed(chunks)) {
      return fail("requires known physical chunks");
    }
    for (int64_t chunk = 0; chunk < *chunks; ++chunk) {
      for (int64_t lane = 0; lane < lanesPerPart; ++lane) {
        FailureOr<bool> padding = isPadding(part, chunk, lane);
        if (failed(padding)) {
          return fail("failed to map physical padding lane");
        }
        if (*padding) {
          return fail("found padding lane in physical chunk");
        }
      }
    }
  }
  return success();
}

LogicalResult checkFullVMIPhysicalChunks(Type type, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  FailureOr<int64_t> factor = getVMITypeLayoutFactor(type);
  FailureOr<int64_t> lanesPerPart = getVMITypeLanesPerPart(type);
  bool unavailableVMIShape = failed(factor) || failed(lanesPerPart);
  if (unavailableVMIShape) {
    return fail("requires assigned layout with known physical lanes per part");
  }

  return validateNoPaddingPhysicalChunks(
      *factor, *lanesPerPart,
      [&](int64_t part) { return getVMITypeChunksInPart(type, part); },
      [&](int64_t part, int64_t chunk, int64_t lane) {
        return isPaddingLane(type, part, chunk, lane);
      },
      reason);
}

FailureOr<int64_t> getContiguousMaterializationPartCount(Type type,
                                                         std::string *reason);

static FailureOr<VMILayoutAttr> getMaterializationLayout(
    Type type, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<VMILayoutAttr> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  Attribute layoutAttr;
  if (auto vregType = dyn_cast<VMIVRegType>(type)) {
    layoutAttr = vregType.getLayout();
  } else if (auto maskType = dyn_cast<VMIMaskType>(type)) {
    layoutAttr = maskType.getLayout();
  } else {
    return fail("requires VMI data or mask type");
  }
  auto layout = dyn_cast_or_null<VMILayoutAttr>(layoutAttr);
  if (!layout) {
    return fail("requires assigned layout");
  }
  return layout;
}

static LogicalResult verifyMaterializationPartCounts(
    Type type, VMILayoutAttr layout, int64_t factor, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<int64_t> chunksPerGroup = getVMITypeChunksInPart(type, 0);
  if (failed(chunksPerGroup)) {
    return fail("requires known physical chunks per part");
  }
  if (*chunksPerGroup == 0) {
    return fail("requires at least one physical chunk per part");
  }
  for (int64_t part = 1; part < factor; ++part) {
    FailureOr<int64_t> chunks = getVMITypeChunksInPart(type, part);
    if (failed(chunks)) {
      return fail("requires known physical chunks per part");
    }
    bool mismatchedFactor2Chunks =
        layout.getFactor() == 2 && *chunks != *chunksPerGroup;
    if (mismatchedFactor2Chunks) {
      return fail("requires every deinterleaved part to have the same "
                  "physical chunk count");
    }
  }
  return success();
}

static FailureOr<int64_t> getContiguousMaterializationPartCountForLayout(
    Type type, VMILayoutAttr layout, int64_t arity, int64_t factor,
    std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<int64_t> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  bool isContiguousLayout =
      layout.isContiguous() && layout.getLaneStride() == 1;
  if (isContiguousLayout) {
    return arity;
  }
  bool unsupportedSplitLayout =
      !layout.isDenseSplit() ||
      (layout.getFactor() != 2 && layout.getFactor() != 4);
  if (unsupportedSplitLayout) {
    return fail("requires contiguous, deinterleaved=2/4, or "
                "block_deinterleaved=2/4 layout");
  }

  if (failed(verifyMaterializationPartCounts(type, layout, factor, reason))) {
    return failure();
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

FailureOr<int64_t> getContiguousMaterializationPartCount(Type type,
                                                         std::string *reason) {
  FailureOr<int64_t> arity = getVMIPhysicalArity(type);
  FailureOr<int64_t> factor = getVMITypeLayoutFactor(type);
  bool missingMaterializationCounts = failed(arity) || failed(factor);
  if (missingMaterializationCounts) {
    if (reason) {
      *reason = "requires computable physical arity and assigned layout";
    }
    return failure();
  }
  FailureOr<VMILayoutAttr> layout = getMaterializationLayout(type, reason);
  if (failed(layout)) {
    return failure();
  }
  return getContiguousMaterializationPartCountForLayout(type, *layout, *arity,
                                                        *factor, reason);
}

LogicalResult checkCanMaterializeToContiguous(Type type, std::string *reason) {
  return succeeded(getContiguousMaterializationPartCount(type, reason))
             ? success()
             : failure();
}

std::optional<int64_t> getConstantIndexValue(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>()) {
    return constant.value();
  }
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto integerAttr = dyn_cast<IntegerAttr>(constant.getValue())) {
      return integerAttr.getInt();
    }
  }
  return std::nullopt;
}

FailureOr<int64_t> getStaticMemRefElementCount(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  if (!memrefType || !memrefType.hasStaticShape()) {
    return failure();
  }

  int64_t elements = 1;
  for (int64_t dim : memrefType.getShape()) {
    if (llvm::MulOverflow(elements, dim, elements)) {
      return failure();
    }
  }
  return elements;
}

static Type getMemoryElementType(Type type) {
  if (auto ptrType = dyn_cast<PtrType>(type)) {
    return ptrType.getElementType();
  }
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    return memrefType.getElementType();
  }
  return {};
}

static Attribute getMemorySpace(Type type) {
  if (auto ptrType = dyn_cast<PtrType>(type)) {
    return ptrType.getMemorySpace();
  }
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    return memrefType.getMemorySpace();
  }
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

enum class VMIMemoryDirection { Read, Write };

enum class VMIMemoryCoverageKind { Dense, Prefix, Predicate };

struct VMIMemoryCoverage {
  VMIMemoryCoverageKind kind = VMIMemoryCoverageKind::Dense;
  int64_t elementCount = 0;
  Value predicate;
};

struct VMIIdentityTransfer {};
struct VMILaneExpandTransfer {
  int64_t factor = 1;
};
struct VMILowBitsCompactTransfer {
  int64_t factor = 1;
};
struct VMIGroupRepeatTransfer {
  int64_t groups = 1;
  int64_t lanesPerGroup = 1;
};
struct VMIDeinterleaveTransfer {
  int64_t factor = 2;
};
struct VMIInterleaveTransfer {
  int64_t factor = 2;
};
struct VMILaneSelectionTransfer {
  SmallVector<int64_t> lanes;
};

using VMIRegisterTransfer =
    std::variant<VMIIdentityTransfer, VMILaneExpandTransfer,
                 VMILowBitsCompactTransfer, VMIGroupRepeatTransfer,
                 VMIDeinterleaveTransfer, VMIInterleaveTransfer,
                 VMILaneSelectionTransfer>;

struct VMIPlannedAddress {
  Value base;
  Value elementOffset;
  Type elementType;
};

struct VMIMemoryLaneAddressMap {
  int64_t baseElementOffset = 0;
  int64_t elementStride = 1;
  int64_t physicalLaneFootprint = 0;

  int64_t getExclusiveEndElement() const {
    return baseElementOffset + physicalLaneFootprint * elementStride;
  }
};

struct VMIByteInterval {
  int64_t begin = 0;
  int64_t end = 0;

  bool contains(const VMIByteInterval &other) const {
    return begin <= other.begin && end >= other.end;
  }
};

struct VMIMemorySafeReadProof {
  bool proven = false;
  std::string reason;
  std::optional<int64_t> constantOffset;
  std::optional<int64_t> staticElementCount;
  std::optional<VMIMemoryLaneAddressMap> laneAddressMap;
  int64_t physicalFootprint = 0;
  std::optional<VMIByteInterval> readableEnvelope;
  std::optional<VMIByteInterval> candidateReadEnvelope;
};

struct VMIPhysicalMemorySegment {
  VMIPlannedAddress address;
  VMIMemoryCoverage coverage;
  VMIRegisterTransfer transfer = VMIIdentityTransfer{};
  VMIMemorySafeReadProof readSafety;
};

struct VMIMemoryAccessPlan {
  VMIMemoryDirection direction = VMIMemoryDirection::Read;
  SmallVector<VMIPhysicalMemorySegment, 1> segments;
  VMIVRegType valueType;
  Attribute paddingValue;
  VMISupportResult layoutSupport;

  VMIPhysicalMemorySegment &front() { return segments.front(); }
  const VMIPhysicalMemorySegment &front() const { return segments.front(); }
};

static std::optional<int64_t> getPhysicalVectorBytes(VRegType type) {
  unsigned elementBits = pto::getPTOStorageElemBitWidth(type.getElementType());
  int64_t totalBits;
  if (elementBits == 0 ||
      llvm::MulOverflow(type.getElementCount(),
                        static_cast<int64_t>(elementBits), totalBits) ||
      totalBits <= 0 || totalBits % 8 != 0) {
    return std::nullopt;
  }
  return totalBits / 8;
}

static bool isDirectMemoryDistAddressLegal(Value base, Value offset,
                                           Type addressElementType,
                                           VRegType registerType,
                                           VPTOMemoryOpFamily family,
                                           StringRef dist) {
  unsigned registerElementBits =
      pto::getPTOStorageElemBitWidth(registerType.getElementType());
  const VPTOMemoryDistContract *contract = lookupVPTOMemoryDist(
      family, dist,
      registerElementBits == 0 ? std::nullopt
                               : std::optional<unsigned>(registerElementBits));
  std::optional<int64_t> vectorBytes = getPhysicalVectorBytes(registerType);
  std::optional<int64_t> requiredAlignment =
      contract && vectorBytes
          ? contract->getRequiredAlignmentBytes(*vectorBytes)
          : std::nullopt;
  return requiredAlignment &&
         isKnownAddressAligned(base, offset, addressElementType,
                               *requiredAlignment);
}

FailureOr<VMIMemoryLaneAddressMap>
buildContiguousIdentityLaneAddressMap(int64_t constantOffset,
                                      VMIVRegType resultType,
                                      std::string *reason = nullptr) {
  auto fail = [&reason](const Twine &message) -> FailureOr<VMIMemoryLaneAddressMap> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(resultType.getElementType());
  FailureOr<int64_t> arity = getVMIPhysicalArity(resultType);
  bool unavailableAddressMapShape = failed(lanesPerPart) || failed(arity);
  if (unavailableAddressMapShape) {
    return fail("requires computable physical read footprint");
  }

  VMIMemoryLaneAddressMap map;
  map.baseElementOffset = constantOffset;
  map.physicalLaneFootprint = *arity * *lanesPerPart;
  return map;
}

VMISupportResult requireIdentityMemRefLayout(Type memoryType, StringRef role,
                                             Value memoryValue = {}) {
  auto memrefType = dyn_cast<MemRefType>(memoryType);
  if (!memrefType || memrefType.getLayout().isIdentity()) {
    return VMISupportResult::success();
  }
  std::string reason =
      (Twine(role) +
       " memref layout is non-identity; current VMI memory access plan "
       "supports only contiguous identity lane-to-address maps")
          .str();
  bool hasSubViewBase =
      memoryValue && memoryValue.getDefiningOp<memref::SubViewOp>();
  if (hasSubViewBase) {
    reason += "; memref.subview requires normalized base/offset/stride "
              "lane-to-address planning";
  }
  return VMISupportResult::failure(reason);
}

struct VMIStaticReadEnvelopes {
  VMIByteInterval readable;
  VMIByteInterval candidate;
};

static FailureOr<int64_t> getByteAddressableElementSize(
    Type elementType, std::string *reason) {
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  if (elementBits == 0 || elementBits % 8 != 0) {
    if (reason) {
      *reason = "requires byte-addressable element type";
    }
    return failure();
  }
  return static_cast<int64_t>(elementBits / 8);
}

static FailureOr<VMIStaticReadEnvelopes> buildStaticReadEnvelopes(
    int64_t constantOffset, int64_t staticElements, int64_t elementBytes,
    int64_t physicalFootprint, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<VMIStaticReadEnvelopes> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  int64_t offsetBytes;
  int64_t allocationBytes;
  int64_t footprintBytes;
  bool envelopeOverflows =
      llvm::MulOverflow(constantOffset, elementBytes, offsetBytes) ||
      llvm::MulOverflow(staticElements, elementBytes, allocationBytes) ||
      llvm::MulOverflow(physicalFootprint, elementBytes, footprintBytes);
  if (envelopeOverflows) {
    return fail("byte read envelope overflows int64");
  }
  return VMIStaticReadEnvelopes{
      VMIByteInterval{-offsetBytes, allocationBytes - offsetBytes},
      VMIByteInterval{0, footprintBytes}};
}

struct VMIStaticReadContract {
  int64_t staticElements;
  int64_t elementBytes;
  VMIMemoryLaneAddressMap addressMap;
};

static FailureOr<VMIStaticReadContract> getStaticReadContract(
    Type sourceType, int64_t constantOffset, VMIVRegType resultType,
    std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<VMIStaticReadContract> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<int64_t> staticElements = getStaticMemRefElementCount(sourceType);
  if (failed(staticElements)) {
    return fail("requires statically shaped memref source");
  }
  if (constantOffset < 0) {
    return fail("requires non-negative offset");
  }
  std::string addressMapReason;
  FailureOr<VMIMemoryLaneAddressMap> addressMap =
      buildContiguousIdentityLaneAddressMap(constantOffset, resultType,
                                            &addressMapReason);
  if (failed(addressMap)) {
    return fail(addressMapReason);
  }
  std::string elementSizeReason;
  FailureOr<int64_t> elementBytes = getByteAddressableElementSize(
      resultType.getElementType(), &elementSizeReason);
  if (failed(elementBytes)) {
    return fail(elementSizeReason);
  }
  return VMIStaticReadContract{*staticElements, *elementBytes, *addressMap};
}

VMIMemorySafeReadProof
computeSafeFullReadProof(Type sourceType, std::optional<int64_t> constantOffset,
                         VMIVRegType resultType) {
  VMIMemorySafeReadProof proof;
  proof.constantOffset = constantOffset;

  auto fail = [&proof](const Twine &message) {
    proof.proven = false;
    proof.reason = message.str();
    return proof;
  };

  if (!constantOffset) {
    return fail("requires constant index offset");
  }

  std::string contractReason;
  FailureOr<VMIStaticReadContract> contract = getStaticReadContract(
      sourceType, *constantOffset, resultType, &contractReason);
  if (failed(contract)) {
    return fail(contractReason);
  }
  proof.staticElementCount = contract->staticElements;
  proof.laneAddressMap = contract->addressMap;
  proof.physicalFootprint = contract->addressMap.physicalLaneFootprint;
  std::string envelopeReason;
  FailureOr<VMIStaticReadEnvelopes> envelopes = buildStaticReadEnvelopes(
      *constantOffset, contract->staticElements, contract->elementBytes,
      proof.physicalFootprint,
      &envelopeReason);
  if (failed(envelopes)) {
    return fail(envelopeReason);
  }
  proof.readableEnvelope = envelopes->readable;
  proof.candidateReadEnvelope = envelopes->candidate;
  if (!proof.readableEnvelope->contains(*proof.candidateReadEnvelope)) {
    return fail(Twine("full physical read footprint [") +
                Twine(contract->addressMap.baseElementOffset) + ", " +
                Twine(contract->addressMap.getExclusiveEndElement()) +
                ") exceeds static memref element count " +
                Twine(contract->staticElements));
  }

  proof.proven = true;
  return proof;
}

struct VMIStatefulOffsetRange {
  int64_t minimum;
  int64_t maximum;
};

static std::optional<int64_t> convertFiniteRangeBound(
    const APInt &bound, bool unsignedInterpretation) {
  if (unsignedInterpretation) {
    return bound.getActiveBits() > 63
               ? std::nullopt
               : std::optional<int64_t>(bound.getZExtValue());
  }
  return bound.isSignedIntN(64) ? std::optional<int64_t>(bound.getSExtValue())
                                : std::nullopt;
}

struct VMIStatefulReadEnvelopes {
  VMIByteInterval readable;
  VMIByteInterval candidate;
};

static FailureOr<VMIStatefulReadEnvelopes> buildStatefulReadEnvelopes(
    int64_t staticElements, VMIStatefulOffsetRange offsetRange,
    int64_t elementBytes, int64_t physicalFootprint, int64_t remainder,
    std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<VMIStatefulReadEnvelopes> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  int64_t minOffsetBytes;
  int64_t maxOffsetBytes;
  int64_t allocationBytes;
  int64_t footprintBytes;
  bool envelopeOverflows =
      llvm::MulOverflow(offsetRange.minimum, elementBytes, minOffsetBytes) ||
      llvm::MulOverflow(offsetRange.maximum, elementBytes, maxOffsetBytes) ||
      llvm::MulOverflow(staticElements, elementBytes, allocationBytes) ||
      llvm::MulOverflow(physicalFootprint, elementBytes, footprintBytes);
  if (envelopeOverflows) {
    return fail("stateful byte read envelope overflows int64");
  }

  constexpr int64_t blockBytes = 32;
  int64_t roundedInput;
  int64_t roundedEnd;
  bool roundedEndOverflows =
      llvm::AddOverflow(footprintBytes, remainder, roundedInput) ||
      llvm::AddOverflow(roundedInput, blockBytes - 1, roundedEnd);
  if (roundedEndOverflows) {
    return fail("stateful byte read envelope overflows int64");
  }
  roundedEnd = roundedEnd / blockBytes * blockBytes - remainder;

  int64_t physicalBegin;
  int64_t physicalEnd;
  bool physicalEnvelopeOverflows =
      llvm::SubOverflow(minOffsetBytes, remainder, physicalBegin) ||
      llvm::AddOverflow(maxOffsetBytes, roundedEnd, physicalEnd);
  if (physicalEnvelopeOverflows) {
    return fail("stateful byte read envelope overflows int64");
  }
  return VMIStatefulReadEnvelopes{
      VMIByteInterval{0, allocationBytes},
      VMIByteInterval{physicalBegin, physicalEnd}};
}

static FailureOr<int64_t> getPhysicalReadFootprintElements(
    VMIVRegType resultType, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<int64_t> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(resultType.getElementType());
  FailureOr<int64_t> arity = getVMIPhysicalArity(resultType);
  bool missingFootprint = failed(lanesPerPart) || failed(arity);
  if (missingFootprint) {
    return fail("requires computable physical read footprint");
  }
  int64_t footprintElements;
  if (llvm::MulOverflow(*arity, *lanesPerPart, footprintElements)) {
    return fail("stateful byte read envelope overflows int64");
  }
  return footprintElements;
}

static FailureOr<VMIStatefulOffsetRange>
getStatefulOffsetRange(Value source, Value offset, std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<VMIStatefulOffsetRange> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  std::optional<int64_t> minOffset = getConstantIndexValue(offset);
  std::optional<int64_t> maxOffset = minOffset;
  if (!minOffset) {
    Operation *anchor = offset.getDefiningOp();
    if (!anchor) {
      anchor = offset.getParentBlock()->getParentOp();
    }
    scf::ForOp loop = dyn_cast_or_null<scf::ForOp>(anchor);
    if (!loop && anchor) {
      loop = anchor->getParentOfType<scf::ForOp>();
    }
    func::FuncOp func =
        anchor ? anchor->getParentOfType<func::FuncOp>() : func::FuncOp();
    if (loop && func) {
      PTOValueEvolutionAnalysis valueEvolution(func);
      PTOAnalysisResult<PTOFiniteRange> range =
          valueEvolution.getRange(offset, loop);
      if (range) {
        minOffset = convertFiniteRangeBound(
            range.value->lowerInclusive, range.value->unsignedInterpretation);
        maxOffset = convertFiniteRangeBound(
            range.value->upperInclusive, range.value->unsignedInterpretation);
      }
    }
  }
  if (!minOffset || !maxOffset) {
    return fail("requires a constant offset or proven finite loop offset range");
  }
  if (*minOffset < 0 || *maxOffset < *minOffset) {
    return fail("requires a non-negative valid offset range");
  }
  return VMIStatefulOffsetRange{*minOffset, *maxOffset};
}

struct VMIStatefulReadContract {
  int64_t staticElements;
  int64_t elementBytes;
  int64_t physicalFootprint;
  int64_t remainder;
  VMIStatefulOffsetRange offsetRange;
};

static FailureOr<VMIStatefulReadContract> getStatefulReadContract(
    Value source, Value offset, VMIVRegType resultType, std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<VMIStatefulReadContract> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<int64_t> staticElements =
      getStaticMemRefElementCount(source.getType());
  if (failed(staticElements)) {
    return fail("requires statically shaped memref source");
  }
  std::string rangeReason;
  FailureOr<VMIStatefulOffsetRange> offsetRange =
      getStatefulOffsetRange(source, offset, &rangeReason);
  if (failed(offsetRange)) {
    return fail(rangeReason);
  }
  std::string elementSizeReason;
  FailureOr<int64_t> elementBytes = getByteAddressableElementSize(
      resultType.getElementType(), &elementSizeReason);
  if (failed(elementBytes)) {
    return fail(elementSizeReason);
  }
  constexpr int64_t blockBytes = 32;
  std::optional<int64_t> remainder = getKnownAddressRemainderBytes(
      source, offset, resultType.getElementType(), blockBytes);
  if (!remainder) {
    return fail("requires a proven fixed 32-byte address remainder");
  }
  std::string footprintReason;
  FailureOr<int64_t> physicalFootprint =
      getPhysicalReadFootprintElements(resultType, &footprintReason);
  if (failed(physicalFootprint)) {
    return fail(footprintReason);
  }
  return VMIStatefulReadContract{*staticElements, *elementBytes,
                                 *physicalFootprint, *remainder, *offsetRange};
}

static VMIMemorySafeReadProof
computeSafeStatefulReadProof(Value source, Value offset,
                             VMIVRegType resultType) {
  VMIMemorySafeReadProof proof;

  auto fail = [&proof](const Twine &message) {
    proof.proven = false;
    proof.reason = message.str();
    return proof;
  };

  std::string contractReason;
  FailureOr<VMIStatefulReadContract> contract =
      getStatefulReadContract(source, offset, resultType, &contractReason);
  if (failed(contract)) {
    return fail(contractReason);
  }
  std::string envelopeReason;
  FailureOr<VMIStatefulReadEnvelopes> envelopes = buildStatefulReadEnvelopes(
      contract->staticElements, contract->offsetRange, contract->elementBytes,
      contract->physicalFootprint, contract->remainder, &envelopeReason);
  if (failed(envelopes)) {
    return fail(envelopeReason);
  }
  proof.readableEnvelope = envelopes->readable;
  proof.candidateReadEnvelope = envelopes->candidate;
  proof.proven = proof.readableEnvelope->contains(*proof.candidateReadEnvelope);
  if (!proof.proven) {
    proof.reason = (Twine("stateful physical read envelope [") +
                    Twine(proof.candidateReadEnvelope->begin) + ", " +
                    Twine(proof.candidateReadEnvelope->end) +
                    ") exceeds proven allocation envelope [" +
                    Twine(proof.readableEnvelope->begin) + ", " +
                    Twine(proof.readableEnvelope->end) + ")")
                       .str();
  }
  return proof;
}

VMIMemoryAccessPlan buildReadAccessPlan(Value source, Value offset,
                                        VMIVRegType resultType,
                                        VMIMemoryCoverageKind coverageKind) {
  VMIMemoryAccessPlan plan;
  plan.direction = VMIMemoryDirection::Read;
  plan.valueType = resultType;
  VMIPhysicalMemorySegment segment;
  segment.address =
      VMIPlannedAddress{source, offset, resultType.getElementType()};
  segment.coverage =
      VMIMemoryCoverage{coverageKind, resultType.getElementCount(), {}};
  segment.transfer = VMIIdentityTransfer{};
  segment.readSafety = computeSafeFullReadProof(
      source.getType(), getConstantIndexValue(offset), resultType);
  plan.segments.push_back(std::move(segment));
  plan.layoutSupport =
      requireIdentityMemRefLayout(source.getType(), "source", source);
  return plan;
}

VMIMemoryAccessPlan buildReadAccessPlan(Value source, Type sourceType,
                                        VMIVRegType resultType,
                                        std::optional<int64_t> constantOffset,
                                        VMIMemoryCoverageKind coverageKind) {
  VMIMemoryAccessPlan plan;
  plan.direction = VMIMemoryDirection::Read;
  plan.valueType = resultType;
  VMIPhysicalMemorySegment segment;
  segment.address = VMIPlannedAddress{source, {}, resultType.getElementType()};
  segment.coverage =
      VMIMemoryCoverage{coverageKind, resultType.getElementCount(), {}};
  segment.transfer = VMIIdentityTransfer{};
  segment.readSafety =
      computeSafeFullReadProof(sourceType, constantOffset, resultType);
  plan.segments.push_back(std::move(segment));
  plan.layoutSupport =
      requireIdentityMemRefLayout(sourceType, "source", source);
  return plan;
}

VMIMemoryAccessPlan buildWriteAccessPlan(Value destination, Value offset,
                                         VMIVRegType valueType,
                                         VMIMemoryCoverageKind coverageKind) {
  VMIMemoryAccessPlan plan;
  plan.direction = VMIMemoryDirection::Write;
  plan.valueType = valueType;
  VMIPhysicalMemorySegment segment;
  segment.address =
      VMIPlannedAddress{destination, offset, valueType.getElementType()};
  segment.coverage =
      VMIMemoryCoverage{coverageKind, valueType.getElementCount(), {}};
  segment.transfer = VMIIdentityTransfer{};
  plan.segments.push_back(std::move(segment));
  plan.layoutSupport = requireIdentityMemRefLayout(destination.getType(),
                                                   "destination", destination);
  return plan;
}

VMIMemoryAccessPlan buildWriteAccessPlan(Value destination,
                                         Type destinationType,
                                         VMIVRegType valueType,
                                         VMIMemoryCoverageKind coverageKind) {
  VMIMemoryAccessPlan plan;
  plan.direction = VMIMemoryDirection::Write;
  plan.valueType = valueType;
  VMIPhysicalMemorySegment segment;
  segment.address =
      VMIPlannedAddress{destination, {}, valueType.getElementType()};
  segment.coverage =
      VMIMemoryCoverage{coverageKind, valueType.getElementCount(), {}};
  segment.transfer = VMIIdentityTransfer{};
  plan.segments.push_back(std::move(segment));
  plan.layoutSupport =
      requireIdentityMemRefLayout(destinationType, "destination", destination);
  return plan;
}

static std::string
getUnavailableReadFallbackReason(VMIMemoryCoverageKind coverageKind) {
  std::string maskedLoadReason;
  if (coverageKind == VMIMemoryCoverageKind::Predicate) {
    maskedLoadReason =
        "; target true masked/non-faulting load is unavailable because the "
        "current VPTO pto.vlds surface has no mask operand";
  }
  std::string scratchReason =
      "; scratch memory fallback resource allocation is not implemented";
  std::string guardedReason =
      "; guarded memory fallback control-flow lowering is not implemented";
  return (Twine("partial/tail read needs a scratch, guarded, or true "
                "masked/non-faulting load fallback, but no such fallback "
                "resource plan is implemented") +
          maskedLoadReason + scratchReason + guardedReason)
      .str();
}

FailureOr<int64_t> verifyFullOrSafeReadVRegChunks(Operation *op,
                                                  VMIVRegType type,
                                                  Value source, Value offset,
                                                  PatternRewriter &rewriter) {
  std::string fullChunkReason;
  FailureOr<int64_t> lanesPerPart =
      checkFullDataPhysicalChunks(type, &fullChunkReason);
  bool usesAlignedLoad =
      isKnownAddressAligned(source, offset, type.getElementType(), 32);
  bool canUseAlignedFullChunk = succeeded(lanesPerPart) && usesAlignedLoad;
  if (canUseAlignedFullChunk) {
    return *lanesPerPart;
  }

  VMIMemorySafeReadProof safeReadProof =
      usesAlignedLoad
          ? computeSafeFullReadProof(source.getType(),
                                     getConstantIndexValue(offset), type)
          : computeSafeStatefulReadProof(source, offset, type);
  if (safeReadProof.proven) {
    lanesPerPart = getDataLanesPerPart(type.getElementType());
    if (succeeded(lanesPerPart)) {
      return *lanesPerPart;
    }
  }

  std::string legalizationReason =
      succeeded(lanesPerPart)
          ? "unaligned load requires a proven stateful physical read envelope"
          : fullChunkReason;
  (void)rewriter.notifyMatchFailure(
      op, Twine("memory lowering ") + legalizationReason +
              "; safe physical-read proof failed: " + safeReadProof.reason);
  return failure();
}

LogicalResult
checkSupportedLoadShape(VMIVRegType type, Value source, Type sourceType,
                        std::optional<int64_t> constantOffset,
                        std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMIMemoryAccessPlan accessPlan = buildReadAccessPlan(
      source, sourceType, type, constantOffset, VMIMemoryCoverageKind::Dense);
  if (!accessPlan.layoutSupport.isSupported()) {
    return fail(accessPlan.layoutSupport.reason);
  }

  VMILayoutSupport supports;
  if (failed(supports.getLoadLayoutFact(type, reason))) {
    return failure();
  }

  if (getDenseLaneStrideLoadDistToken(type)) {
    return success();
  }

  if (failed(getDataLanesPerPart(type.getElementType()))) {
    return fail("requires element type with known physical lane width");
  }
  return success();
}

LogicalResult checkSupportedDeinterleaveLoadShape(
    VMIDeinterleaveLoadOp op,
    std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  auto lowType = cast<VMIVRegType>(op.getLow().getType());
  auto highType = cast<VMIVRegType>(op.getHigh().getType());
  VMILayoutSupport supports;
  if (failed(supports.getDeinterleaveLoadLayoutFactForLayouts(
          lowType, highType, reason))) {
    return failure();
  }
  if (!getX2MemoryDistToken(lowType.getElementType(), "DINTLV")) {
    return fail("requires 8/16/32-bit element type for vldsx2 DINTLV");
  }

  VMIMemoryAccessPlan accessPlan = buildReadAccessPlan(
      op.getSource(), op.getOffset(), lowType, VMIMemoryCoverageKind::Dense);
  if (!accessPlan.layoutSupport.isSupported()) {
    return fail(accessPlan.layoutSupport.reason);
  }

  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(lowType, &fullChunkReason))) {
    return fail(Twine("requires full physical chunks; ") + fullChunkReason);
  }
  return success();
}

static LogicalResult checkStorePhysicalCoverage(VMIVRegType type,
                                                std::string *reason) {
  if (getDenseLaneStrideStoreDistToken(type)) {
    return success();
  }

  std::string fullChunkReason;
  if (succeeded(checkFullDataPhysicalChunks(type, &fullChunkReason))) {
    return success();
  }

  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout) {
    return fail("requires assigned layout");
  }
  if (failed(getDataLanesPerPart(type.getElementType()))) {
    return fail("requires known physical lanes per part");
  }
  bool directContiguousStore =
      layout.isContiguous() && layout.getLaneStride() == 1;
  if (directContiguousStore) {
    return success();
  }

  std::string materializationReason;
  if (succeeded(checkCanMaterializeToContiguous(type, &materializationReason))) {
    return success();
  }
  return fail(Twine("partial/tail store requires contiguous layout or "
                    "deinterleaved layout that can materialize to contiguous; "
                    "value ") +
              fullChunkReason + ", materialization " + materializationReason);
}

LogicalResult checkSupportedStoreShape(VMIVRegType type, Value destination,
                                       Type destinationType,
                                       std::string *reason) {
  VMIMemoryAccessPlan accessPlan = buildWriteAccessPlan(
      destination, destinationType, type, VMIMemoryCoverageKind::Dense);
  if (!accessPlan.layoutSupport.isSupported()) {
    if (reason) {
      *reason = accessPlan.layoutSupport.reason;
    }
    return failure();
  }

  if (failed(checkSupportedMaskableVReg(type, reason))) {
    return failure();
  }

  VMILayoutSupport supports;
  if (failed(supports.getStoreLayoutFact(type, reason))) {
    return failure();
  }
  return checkStorePhysicalCoverage(type, reason);
}

LogicalResult checkSupportedInterleaveStoreShape(
    VMIInterleaveStoreOp op,
    std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  auto lowType = cast<VMIVRegType>(op.getLow().getType());
  auto highType = cast<VMIVRegType>(op.getHigh().getType());
  VMILayoutAttr lowLayout = lowType.getLayoutAttr();
  VMILayoutAttr highLayout = highType.getLayoutAttr();
  bool nonContiguousInputs =
      !lowLayout || !highLayout || !lowLayout.isContiguous() ||
      !highLayout.isContiguous();
  if (nonContiguousInputs) {
    return fail("requires assigned contiguous low/high input layouts");
  }
  bool mismatchedInputs = lowType.getElementCount() != highType.getElementCount() ||
                        lowType.getElementType() != highType.getElementType();
  if (mismatchedInputs) {
    return fail("requires matching low/high input shape and element type");
  }
  if (!getX2MemoryDistToken(lowType.getElementType(), "INTLV")) {
    return fail("requires 8/16/32-bit element type for vstsx2 INTLV");
  }

  VMIMemoryAccessPlan accessPlan =
      buildWriteAccessPlan(op.getDestination(), op.getOffset(), lowType,
                           VMIMemoryCoverageKind::Dense);
  if (!accessPlan.layoutSupport.isSupported()) {
    return fail(accessPlan.layoutSupport.reason);
  }
  if (failed(checkSupportedMaskableVReg(lowType, reason))) {
    return failure();
  }

  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(lowType, &fullChunkReason))) {
    return fail(Twine("requires full physical chunks; ") + fullChunkReason);
  }
  return success();
}

FailureOr<int64_t> getGroupSizeFromNumGroups(VMIVRegType type,
                                             int64_t numGroups,
                                             std::string *reason = nullptr) {
  auto fail = [&reason](const Twine &message) -> FailureOr<int64_t> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  if (numGroups <= 0) {
    return fail("requires num_groups to be positive");
  }
  bool unevenGroups = type.getElementCount() % numGroups != 0;
  if (unevenGroups) {
    return fail("requires num_groups to evenly divide logical lane count");
  }
  return type.getElementCount() / numGroups;
}

LogicalResult checkSupportedGroupChunkShape(VMIVRegType type, int64_t groupSize,
                                            std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutAttr layout = type.getLayoutAttr();
  bool nonContiguousLayout = !layout || !layout.isContiguous();
  if (nonContiguousLayout) {
    return fail("requires assigned contiguous layout");
  }
  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(type, &fullChunkReason))) {
    return fail(Twine("requires full physical chunks; ") + fullChunkReason);
  }
  FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(type.getElementType());
  if (failed(lanesPerPart)) {
    return fail("requires known physical lanes per part");
  }
  bool invalidGroupSize =
      groupSize <= 0 || type.getElementCount() % groupSize != 0;
  if (invalidGroupSize) {
    return fail("requires derived group size to evenly divide logical lane "
                "count");
  }
  if (groupSize % *lanesPerPart != 0) {
    return fail("currently requires group size to be a multiple of physical "
                "lanes per part");
  }
  return success();
}

struct Deinterleaved2GroupStoreShape {
  int64_t lanesPerPart;
  int64_t groupCount;
  int64_t chunksPerGroupPerPart;
};

static FailureOr<int64_t>
getDeinterleaved2StoreLanes(VMIVRegType type, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<int64_t> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutAttr layout = type.getLayoutAttr();
  bool unsupportedLayout =
      !layout || !layout.isDeinterleaved() || layout.getFactor() != 2 ||
      layout.getLaneStride() != 1;
  if (unsupportedLayout) {
    return fail("requires deinterleaved=2 value layout");
  }
  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(type, &fullChunkReason))) {
    return fail(Twine("requires full physical chunks; ") + fullChunkReason);
  }
  FailureOr<int64_t> lanes = getDataLanesPerPart(type.getElementType());
  if (failed(lanes)) {
    return fail("requires known physical lanes per part");
  }
  if (!getX2MemoryDistToken(type.getElementType(), "INTLV")) {
    return fail("requires 8/16/32-bit element type for vstsx2 INTLV");
  }
  return *lanes;
}

static FailureOr<Deinterleaved2GroupStoreShape>
getDeinterleaved2GroupStoreGeometry(VMIVRegType type, int64_t groupSize,
                                    int64_t lanesPerPart,
                                    std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<Deinterleaved2GroupStoreShape> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  if (groupSize <= 0) {
    return fail("requires positive derived group size");
  }
  int64_t safeGroupSize = groupSize > 0 ? groupSize : 1;
  bool unevenLogicalGroups = type.getElementCount() % safeGroupSize != 0;
  if (unevenLogicalGroups) {
    return fail("requires derived group size to evenly divide logical lane "
                "count");
  }
  if (lanesPerPart <= 0) {
    return fail("requires positive physical lane count");
  }
  int64_t pairLanes = 2 * lanesPerPart;
  if (groupSize % pairLanes != 0) {
    return fail("requires group size to be a multiple of two physical chunks");
  }

  FailureOr<int64_t> part0Chunks = getDataChunksInPart(type, /*part=*/0);
  FailureOr<int64_t> part1Chunks = getDataChunksInPart(type, /*part=*/1);
  bool mismatchedPartChunks =
      failed(part0Chunks) || failed(part1Chunks) ||
      *part0Chunks != *part1Chunks;
  if (mismatchedPartChunks) {
    return fail("requires matching deinterleaved part chunk counts");
  }

  int64_t groupCount = type.getElementCount() / groupSize;
  int64_t chunksPerGroupPerPart = groupSize / pairLanes;
  if (*part0Chunks != groupCount * chunksPerGroupPerPart) {
    return fail("requires deinterleaved chunks to align with group rows");
  }
  return Deinterleaved2GroupStoreShape{lanesPerPart, groupCount,
                                      chunksPerGroupPerPart};
}

static FailureOr<Deinterleaved2GroupStoreShape>
getDeinterleaved2GroupStoreShape(VMIVRegType type, int64_t groupSize,
                                 std::string *reason) {
  FailureOr<int64_t> lanesPerPart =
      getDeinterleaved2StoreLanes(type, reason);
  if (failed(lanesPerPart)) {
    return failure();
  }
  return getDeinterleaved2GroupStoreGeometry(type, groupSize,
                                              *lanesPerPart, reason);
}

LogicalResult checkDeinterleaved2GroupStoreChunkShape(
    VMIVRegType type, int64_t groupSize, int64_t *lanesPerPart,
    int64_t *groupCount, int64_t *chunksPerGroupPerPart,
    std::string *reason) {
  FailureOr<Deinterleaved2GroupStoreShape> shape =
      getDeinterleaved2GroupStoreShape(type, groupSize, reason);
  if (succeeded(shape)) {
    *lanesPerPart = shape->lanesPerPart;
    *groupCount = shape->groupCount;
    *chunksPerGroupPerPart = shape->chunksPerGroupPerPart;
    return success();
  }
  return failure();
}

LogicalResult checkSupportedBlockDeinterleavedGroupLoadShape(
    VMIGroupLoadOp op, VMIVRegType resultType, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutSupport supports;
  if (failed(supports.getGroupLoadLayoutFact(op, reason))) {
    return failure();
  }
  VMIMemoryAccessPlan accessPlan =
      buildReadAccessPlan(op.getSource(), op.getOffset(), resultType,
                          VMIMemoryCoverageKind::Dense);
  if (!accessPlan.layoutSupport.isSupported()) {
    return fail(accessPlan.layoutSupport.reason);
  }
  if (!isa<PtrType>(op.getSource().getType())) {
    return fail("block_deinterleaved group_load requires !pto.ptr source");
  }
  bool hasGroupMultiple = op.getNumGroupsAttr().getInt() % 8 == 0;
  if (!hasGroupMultiple) {
    return fail("block_deinterleaved group_load requires num_groups multiple of 8");
  }
  std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
  if (!rowStride || *rowStride <= 0 || *rowStride % 8 != 0) {
    return fail("block_deinterleaved group_load requires constant positive "
                "row_stride divisible by 8 f32 elements");
  }
  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(resultType, &fullChunkReason))) {
    return fail(Twine("block_deinterleaved group_load requires full physical "
                      "result chunks; ") +
                fullChunkReason);
  }
  return success();
}

LogicalResult
checkSupportedContiguousGroupLoadShape(VMIGroupLoadOp op,
                                       VMIVRegType resultType,
                                       int64_t groupSize, std::string *reason) {
  VMILayoutSupport supports;
  if (failed(supports.getGroupLoadLayoutFact(op, reason))) {
    return failure();
  }
  if (failed(checkSupportedLoadShape(resultType, op.getSource(),
                                     op.getSource().getType(), std::nullopt,
                                     reason))) {
    return failure();
  }
  std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
  bool unitGroupRowStride = rowStride && *rowStride == groupSize;
  if (unitGroupRowStride) {
    return success();
  }
  return checkSupportedGroupChunkShape(resultType, groupSize, reason);
}

LogicalResult
checkSupportedGroupLoadShape(VMIGroupLoadOp op, std::string *reason) {
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!resultLayout) {
    return emitLogicalFailure(reason, "requires assigned result layout");
  }
  FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
      resultType, op.getNumGroupsAttr().getInt(), reason);
  if (failed(groupSize)) {
    return failure();
  }

  if (resultLayout.isContiguous()) {
    return checkSupportedContiguousGroupLoadShape(op, resultType, *groupSize,
                                                  reason);
  }

  bool supportsBlockDeinterleaved =
      resultLayout.isBlockDeinterleaved() && resultType.getElementType().isF32();
  if (supportsBlockDeinterleaved) {
    return checkSupportedBlockDeinterleavedGroupLoadShape(op, resultType,
                                                          reason);
  }

  return emitLogicalFailure(
      reason, "requires contiguous or block_deinterleaved f32 result layout");
}

LogicalResult checkSupportedSlots1GroupSlotLoadShape(
    VMIGroupSlotLoadOp op, VMIVRegType resultType, std::string *reason);

LogicalResult checkSupportedSlots8GroupSlotLoadShape(
    VMIGroupSlotLoadOp op, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  std::optional<int64_t> sourceGroupStride =
      getConstantIndexValue(op.getSourceGroupStride());
  bool nonUnitSourceStride = !sourceGroupStride || *sourceGroupStride != 1;
  if (nonUnitSourceStride) {
    return fail("slots=8 group_slot_load requires constant unit "
                "source_group_stride");
  }
  return success();
}

static LogicalResult checkGroupSlotLoadMemoryContract(
    VMIGroupSlotLoadOp op, VMIVRegType resultType, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  VMIMemoryAccessPlan accessPlan = buildReadAccessPlan(
      op.getSource(), op.getOffset(), resultType, VMIMemoryCoverageKind::Dense);
  if (!accessPlan.layoutSupport.isSupported()) {
    return fail(accessPlan.layoutSupport.reason);
  }
  if (!isa<PtrType>(op.getSource().getType())) {
    return fail("group_slot_load requires !pto.ptr source");
  }
  return success();
}

LogicalResult checkSupportedGroupSlotLoadShape(
    VMIGroupSlotLoadOp op,
    std::string *reason) {
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutSupport supports;
  FailureOr<VMIGroupSlotLayoutFact> fact = supports.getGroupSlotLoadLayoutFact(
      resultType, op.getNumGroupsAttr().getInt(), reason);
  if (failed(fact)) {
    return failure();
  }

  if (failed(checkGroupSlotLoadMemoryContract(op, resultType, reason))) {
    return failure();
  }

  if (fact->slots == 8) {
    return checkSupportedSlots8GroupSlotLoadShape(op, reason);
  }

  return checkSupportedSlots1GroupSlotLoadShape(op, resultType, reason);
}

LogicalResult checkSupportedSlots1GroupSlotLoadShape(
    VMIGroupSlotLoadOp op, VMIVRegType resultType, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  bool unsupportedElementWidth = elementBits == 0 || 256 % elementBits != 0;
  if (unsupportedElementWidth) {
    return fail("slots=1 group_slot_load requires supported element width");
  }
  int64_t alignedStrideElems = 256 / elementBits;
  std::optional<int64_t> sourceGroupStride =
      getConstantIndexValue(op.getSourceGroupStride());
  if (!sourceGroupStride || *sourceGroupStride <= 0 ||
      *sourceGroupStride % alignedStrideElems != 0) {
    return fail(Twine("slots=1 group_slot_load currently lowers as one "
                      "lane-0 vsldb per group and requires constant "
                      "positive source_group_stride divisible by ") +
                Twine(alignedStrideElems) +
                " elements for 32B load alignment; packed or unaligned "
                "scalar load lowering is not implemented");
  }
  return success();
}

LogicalResult checkSupportedGroupBroadcastLoadMemory(
    VMIGroupBroadcastLoadOp op, VMIVRegType resultType, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMIMemoryAccessPlan accessPlan =
      buildReadAccessPlan(op.getSource(), op.getOffset(), resultType,
                          VMIMemoryCoverageKind::Dense);
  if (!accessPlan.layoutSupport.isSupported()) {
    return fail(accessPlan.layoutSupport.reason);
  }
  if (!isa<PtrType>(op.getSource().getType())) {
    return fail("group_broadcast_load requires !pto.ptr source");
  }
  return success();
}

LogicalResult checkSupportedGroupBroadcastLoadShape(
    VMIGroupBroadcastLoadOp op, std::string *reason) {
  VMILayoutSupport supports;
  if (failed(supports.getGroupBroadcastLoadSupport(op, reason))) {
    return failure();
  }
  return checkSupportedGroupBroadcastLoadMemory(
      op, cast<VMIVRegType>(op.getResult().getType()), reason);
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

struct OneBlockGroupStorePlan {
  int64_t groupSize = 0;
  int64_t groupsPerPart = 0;
  int64_t blockStride = 0;
};

FailureOr<OneBlockGroupStorePlan> getOneBlockGroupStorePlan(
    VMIGroupStoreOp op, VMIVRegType valueType,
    const VMIGroupStoreLayoutFact &fact, std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<OneBlockGroupStorePlan> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutAttr layout = valueType.getLayoutAttr();
  bool unsupportedBlockLayout =
      fact.blockClass != VMIGroupBlockClass::OneBlock || !layout ||
      !layout.isContiguous();
  if (unsupportedBlockLayout) {
    return fail("one-block group_store requires contiguous layout");
  }
  bool unsupportedBlockShape =
      fact.groupSize <= 0 || fact.groupSize != fact.vcgBlockElems ||
      fact.lanesPerPart <= 0 || fact.lanesPerPart % fact.groupSize != 0;
  if (unsupportedBlockShape) {
    return fail("one-block group_store requires one 32B group per VCG block");
  }
  if (!isa<PtrType>(op.getDestination().getType())) {
    return fail("one-block group_store requires !pto.ptr destination");
  }

  std::optional<int64_t> rowStride =
      getConstantIndexValue(op.getRowStride());
  bool unalignedRowStride =
      !rowStride || *rowStride <= 0 || *rowStride % fact.groupSize != 0;
  if (unalignedRowStride) {
    return fail("one-block group_store requires a constant positive row_stride "
                "aligned to 32B blocks");
  }

  int64_t blockStride = *rowStride / fact.groupSize;
  if (blockStride > 0xffff) {
    return fail("one-block group_store block_stride exceeds the 16-bit vsstb "
                "control field");
  }

  return OneBlockGroupStorePlan{
      fact.groupSize, fact.lanesPerPart / fact.groupSize, blockStride};
}

LogicalResult
checkSupportedCompactSmallGroupStoreShape(VMIGroupStoreOp op,
                                           VMIVRegType valueType,
                                           std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  if (!isa<PtrType>(op.getDestination().getType())) {
    return fail("compact small group_store requires !pto.ptr destination");
  }
  VMIMemoryAccessPlan accessPlan =
      buildWriteAccessPlan(op.getDestination(), op.getOffset(), valueType,
                           VMIMemoryCoverageKind::Dense);
  if (!accessPlan.layoutSupport.isSupported()) {
    return fail(accessPlan.layoutSupport.reason);
  }
  return success();
}

LogicalResult
checkSupportedGroupSlotsStoreShape(VMIGroupStoreOp op, VMIVRegType valueType,
                                    std::optional<int64_t> rowStride,
                                    std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  VMILayoutSupport supports;
  FailureOr<VMIGroupSlotLayoutFact> fact = supports.getGroupStoreLayoutFact(
      valueType, op.getNumGroupsAttr().getInt(), reason);
  if (failed(fact)) {
    return failure();
  }

  VMIMemoryAccessPlan accessPlan =
      buildWriteAccessPlan(op.getDestination(), op.getOffset(), valueType,
                           VMIMemoryCoverageKind::Dense);
  if (!accessPlan.layoutSupport.isSupported()) {
    return fail(accessPlan.layoutSupport.reason);
  }

  if (fact->slots == 1) {
    unsigned elementBits =
        pto::getPTOStorageElemBitWidth(valueType.getElementType());
    if (elementBits == 0 || 256 % elementBits != 0) {
      return fail("slots=1 group_store requires supported element width");
    }
    if (rowStride && *rowStride <= 0) {
      return fail("slots=1 group_store requires positive row_stride when "
                  "row_stride is constant");
    }
    if (!getPointStoreDistToken(valueType.getElementType())) {
      return fail("slots=1 group_store requires 1PT_B8/B16/B32 store "
                  "support");
    }
    return success();
  }

  if (!rowStride || *rowStride != 1) {
    return fail("slots=8 group_store currently requires constant unit "
                "row_stride");
  }
  return success();
}

LogicalResult
checkSupportedGroupStorePhysicalShape(
    VMIGroupStoreOp op, VMIVRegType valueType,
    const VMIGroupStoreLayoutFact &fact, std::string *reason) {
  if (failed(checkSupportedStoreShape(valueType,
                                      op.getDestination(),
                                      op.getDestination().getType(), reason))) {
    return failure();
  }
  bool oneBlock = fact.blockClass == VMIGroupBlockClass::OneBlock;
  if (oneBlock) {
    if (failed(getOneBlockGroupStorePlan(op, valueType, fact, reason))) {
      return failure();
    }
    return success();
  }
  bool contiguousGroupChunks = succeeded(
      checkSupportedGroupChunkShape(valueType, fact.groupSize, reason));
  if (contiguousGroupChunks) {
    return success();
  }

  int64_t lanesPerPart = 0;
  int64_t groupCount = 0;
  int64_t chunksPerGroupPerPart = 0;
  return checkDeinterleaved2GroupStoreChunkShape(
      valueType, fact.groupSize, &lanesPerPart, &groupCount,
      &chunksPerGroupPerPart, reason);
}

LogicalResult
checkSupportedGroupStoreByLayout(VMIGroupStoreOp op, VMIVRegType valueType,
                                 VMILayoutAttr layout,
                                 std::optional<int64_t> rowStride,
                                 std::string *reason) {
  bool compactSmallGroup = isCompactSmallGroupStore(
      layout, valueType, op.getNumGroupsAttr().getInt(), rowStride);
  if (compactSmallGroup) {
    return checkSupportedCompactSmallGroupStoreShape(op, valueType, reason);
  }
  if (layout && layout.isGroupSlots()) {
    return checkSupportedGroupSlotsStoreShape(op, valueType, rowStride, reason);
  }

  VMILayoutSupport supports;
  FailureOr<VMIGroupStoreLayoutFact> fact =
      supports.getGroupStoreLayoutFact(op, valueType, reason);
  if (failed(fact)) {
    return failure();
  }
  return checkSupportedGroupStorePhysicalShape(op, valueType, *fact, reason);
}

LogicalResult
checkSupportedGroupStoreShape(VMIGroupStoreOp op, std::string *reason) {
  auto valueType = cast<VMIVRegType>(op.getValue().getType());
  VMILayoutAttr layout = valueType.getLayoutAttr();
  std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
  return checkSupportedGroupStoreByLayout(op, valueType, layout, rowStride,
                                          reason);
}

LogicalResult
checkSupportedMaskedLoadShape(VMIMaskedLoadOp op, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  auto passthruType = cast<VMIVRegType>(op.getPassthru().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  VMILayoutAttr passthruLayout = passthruType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMIMemoryAccessPlan accessPlan =
      buildReadAccessPlan(op.getSource(), op.getOffset(), resultType,
                          VMIMemoryCoverageKind::Predicate);
  if (!accessPlan.layoutSupport.isSupported()) {
    return fail(accessPlan.layoutSupport.reason);
  }
  if (!resultLayout || !passthruLayout || !maskLayout) {
    return fail("requires assigned result, passthru, and mask layouts");
  }
  bool nonContiguousLayout = !resultLayout.isContiguous() ||
                             !passthruLayout.isContiguous() ||
                             !maskLayout.isContiguous();
  if (nonContiguousLayout) {
    return fail("requires contiguous result, passthru, and mask layouts");
  }

  std::string fullChunkReason;
  if (succeeded(checkFullDataPhysicalChunks(resultType, &fullChunkReason))) {
    return success();
  }

  if (accessPlan.front().readSafety.proven) {
    return success();
  }
  std::string fallbackReason =
      getUnavailableReadFallbackReason(VMIMemoryCoverageKind::Predicate);
  return fail(Twine("partial/tail masked_load requires statically safe "
                    "full-read footprint; value ") +
              fullChunkReason + ", safe-read proof " +
              accessPlan.front().readSafety.reason +
              "; fallback unavailable: " + fallbackReason);
}

LogicalResult checkSupportedGatherPhysicalShape(
    VMIVRegType resultType, VMIVRegType indicesType, VMIVRegType passthruType,
    VMIMaskType maskType, bool requiresFullChunks, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  FailureOr<int64_t> indicesArity = getVMIPhysicalArity(indicesType);
  FailureOr<int64_t> passthruArity = getVMIPhysicalArity(passthruType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  bool hasPhysicalArity = succeeded(resultArity) && succeeded(indicesArity) &&
                          succeeded(passthruArity) && succeeded(maskArity);
  if (!hasPhysicalArity) {
    return fail("requires computable physical arity");
  }
  if (*resultArity != *indicesArity || *resultArity != *passthruArity ||
      *resultArity != *maskArity) {
    return fail("requires result, indices, passthru, and mask to have the "
                "same physical arity");
  }
  if (*resultArity > mlir::pto::kValue4) {
    return fail("gather exceeds the 4 physical register limit per VMI "
                "instruction");
  }
  if (!requiresFullChunks) {
    return success();
  }
  std::string resultReason;
  std::string indicesReason;
  std::string passthruReason;
  std::string maskReason;
  if (failed(checkFullDataPhysicalChunks(resultType, &resultReason))) {
    return fail(Twine("result requires full physical chunks; ") + resultReason);
  }
  if (failed(checkFullDataPhysicalChunks(indicesType, &indicesReason))) {
    return fail(Twine("indices require full physical chunks; ") +
                indicesReason);
  }
  if (failed(checkFullDataPhysicalChunks(passthruType, &passthruReason))) {
    return fail(Twine("passthru requires full physical chunks; ") +
                passthruReason);
  }
  if (failed(checkFullVMIPhysicalChunks(maskType, &maskReason))) {
    return fail(Twine("mask requires full physical chunks; ") + maskReason);
  }
  return success();
}

static bool haveMatchingIntegerSignedness(IntegerType sourceType,
                                          IntegerType resultType) {
  return (sourceType.isUnsigned() && resultType.isUnsigned()) ||
         (!sourceType.isUnsigned() && !resultType.isUnsigned());
}

static bool isB8To16GatherContract(IntegerType sourceType,
                                   IntegerType resultType,
                                   IntegerType indexType,
                                   VMIMaskType maskType) {
  return sourceType.getWidth() == mlir::pto::kValue8 &&
         resultType.getWidth() == mlir::pto::kValue16 &&
         indexType.isUnsigned() && indexType.getWidth() == 16 &&
         maskType.getGranularity() == "b16" &&
         haveMatchingIntegerSignedness(sourceType, resultType);
}

static bool isSameWidth16IntegerGatherContract(IntegerType sourceType,
                                               IntegerType resultType,
                                               IntegerType indexType,
                                               VMIMaskType maskType) {
  return sourceType.getWidth() == mlir::pto::kValue16 &&
         resultType.getWidth() == mlir::pto::kValue16 &&
         indexType.isUnsigned() && indexType.getWidth() == 16 &&
         maskType.getGranularity() == "b16" &&
         haveMatchingIntegerSignedness(sourceType, resultType);
}

static bool isSameWidth16FloatGatherContract(Type sourceElementType,
                                             Type resultElementType,
                                             IntegerType indexType,
                                             VMIMaskType maskType) {
  return sourceElementType == resultElementType &&
         (sourceElementType.isF16() || sourceElementType.isBF16()) &&
         indexType.isUnsigned() && indexType.getWidth() == 16 &&
         maskType.getGranularity() == "b16";
}

LogicalResult
checkGatherElementContract(VMIVRegType resultType, VMIVRegType indicesType,
                            VMIMaskType maskType, Type sourceElemType,
                            std::string *reason) {
  auto fail = [&reason](const Twine &message) {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  unsigned resultBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  auto indexElementType = dyn_cast<IntegerType>(indicesType.getElementType());
  if (!indexElementType || indexElementType.isSigned()) {
    return fail("requires signless or unsigned integer indices");
  }
  auto sourceInt = dyn_cast<IntegerType>(sourceElemType);
  auto resultInt = dyn_cast<IntegerType>(resultType.getElementType());
  bool isB8To16Gather =
      resultBits == 16 && sourceInt && resultInt &&
      isB8To16GatherContract(sourceInt, resultInt, indexElementType, maskType);
  bool isSameWidth16Gather =
      resultBits == 16 &&
      ((sourceInt && resultInt && isSameWidth16IntegerGatherContract(
                                      sourceInt, resultInt, indexElementType,
                                      maskType)) ||
       isSameWidth16FloatGatherContract(sourceElemType,
                                        resultType.getElementType(),
                                        indexElementType, maskType));
  bool isB16Gather = isSameWidth16Gather || isB8To16Gather;
  bool isB32Gather = resultBits == 32 && indexElementType.getWidth() == 32 &&
                     maskType.getGranularity() == "b32";
  if (!isB16Gather && !isB32Gather) {
    return fail("requires either 32-bit results with 32-bit indices and b32 "
                "mask, or ui16/i16/f16/bf16 results with ui16 indices and "
                "b16 mask (including i8/ui8 -> i16/ui16 promotion)");
  }
  return success();
}

LogicalResult
checkGatherLayoutAndSource(VMIGatherOp op, VMIVRegType resultType,
                            VMIVRegType indicesType, VMIVRegType passthruType,
                            VMIMaskType maskType, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  VMILayoutAttr indicesLayout = indicesType.getLayoutAttr();
  VMILayoutAttr passthruLayout = passthruType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  if (!resultLayout || !indicesLayout || !passthruLayout || !maskLayout) {
    return fail("requires assigned result, indices, passthru, and mask layouts");
  }
  bool nonContiguousLayout =
      !resultLayout.isContiguous() || !indicesLayout.isContiguous() ||
      !passthruLayout.isContiguous() || !maskLayout.isContiguous();
  if (nonContiguousLayout) {
    return fail("requires contiguous result, indices, passthru, and mask layouts");
  }
  if (!isa<PtrType>(op.getSource().getType())) {
    return fail("requires !pto.ptr source because pto.vgather2_bc is pointer-only");
  }
  return success();
}

LogicalResult
checkGatherPhysicalChunkRequirement(VMIVRegType resultType,
                                    VMIVRegType indicesType,
                                    VMIVRegType passthruType,
                                    VMIMaskType maskType,
                                    bool isB16Gather, bool isB32Gather,
                                    std::string *reason) {
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (failed(resultArity)) {
    return failure();
  }
  bool requiresFullChunks = isB32Gather;
  if (isB16Gather) {
    requiresFullChunks = *resultArity != 1;
  }
  return checkSupportedGatherPhysicalShape(
      resultType, indicesType, passthruType, maskType, requiresFullChunks,
      reason);
}

LogicalResult
checkSupportedGatherShape(VMIGatherOp op, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  auto indicesType = cast<VMIVRegType>(op.getIndices().getType());
  auto passthruType = cast<VMIVRegType>(op.getPassthru().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  if (failed(checkGatherLayoutAndSource(op, resultType, indicesType,
                                        passthruType, maskType, reason))) {
    return failure();
  }

  Type sourceElemType = getMemoryElementType(op.getSource().getType());
  std::string elementReason;
  if (failed(checkGatherElementContract(resultType, indicesType, maskType,
                                        sourceElemType, &elementReason))) {
    return fail(elementReason);
  }

  unsigned resultBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  auto indexElementType = dyn_cast<IntegerType>(indicesType.getElementType());
  bool isB16Gather = resultBits == 16 && indexElementType &&
                     indexElementType.getWidth() == 16 &&
                     maskType.getGranularity() == "b16";
  bool isB32Gather = resultBits == 32 && indexElementType &&
                     indexElementType.getWidth() == 32 &&
                     maskType.getGranularity() == "b32";

  return checkGatherPhysicalChunkRequirement(
      resultType, indicesType, passthruType, maskType, isB16Gather,
      isB32Gather, reason);
}

LogicalResult checkSupportedScatterPhysicalShape(
    VMIVRegType valueType, VMIVRegType indicesType, VMIMaskType maskType,
    bool requiresFullChunks, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<int64_t> valueArity = getVMIPhysicalArity(valueType);
  FailureOr<int64_t> indicesArity = getVMIPhysicalArity(indicesType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  bool hasPhysicalArity = succeeded(valueArity) && succeeded(indicesArity) &&
                          succeeded(maskArity);
  if (!hasPhysicalArity) {
    return fail("requires computable physical arity");
  }
  if (*valueArity != *indicesArity || *valueArity != *maskArity) {
    return fail("requires value, indices, and mask to have the same physical "
                "arity");
  }
  if (!requiresFullChunks) {
    return success();
  }
  std::string valueReason;
  std::string indicesReason;
  std::string maskReason;
  if (failed(checkFullDataPhysicalChunks(valueType, &valueReason))) {
    return fail(Twine("value requires full physical chunks; ") + valueReason);
  }
  if (failed(checkFullDataPhysicalChunks(indicesType, &indicesReason))) {
    return fail(Twine("indices require full physical chunks; ") +
                indicesReason);
  }
  if (failed(checkFullVMIPhysicalChunks(maskType, &maskReason))) {
    return fail(Twine("mask requires full physical chunks; ") + maskReason);
  }
  return success();
}

LogicalResult
checkScatterLayoutAndDestination(VMIScatterOp op, VMIVRegType valueType,
                                 VMIVRegType indicesType, VMIMaskType maskType,
                                 std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  VMILayoutAttr valueLayout = valueType.getLayoutAttr();
  VMILayoutAttr indicesLayout = indicesType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  bool missingLayout = !valueLayout || !indicesLayout || !maskLayout;
  if (missingLayout) {
    return fail("requires assigned value, indices, and mask layouts");
  }
  bool nonContiguousLayout = !valueLayout.isContiguous() ||
                             !indicesLayout.isContiguous() ||
                             !maskLayout.isContiguous();
  if (nonContiguousLayout) {
    return fail("requires contiguous value, indices, and mask layouts");
  }
  if (!isa<PtrType>(op.getDestination().getType())) {
    return fail("requires !pto.ptr destination because pto.vscatter is "
                "pointer-only");
  }
  return success();
}

LogicalResult
checkScatterElementContract(VMIVRegType valueType, VMIVRegType indicesType,
                            VMIMaskType maskType,
                            std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  unsigned valueBits =
      pto::getPTOStorageElemBitWidth(valueType.getElementType());
  auto indexElementType = dyn_cast<IntegerType>(indicesType.getElementType());
  if (!indexElementType || indexElementType.isSigned()) {
    return fail("requires signless or unsigned integer indices");
  }
  bool isB8Scatter = valueBits == 8 && indexElementType.getWidth() == 16 &&
                     maskType.getGranularity() == "b16";
  bool isB16Scatter = valueBits == 16 && indexElementType.getWidth() == 16 &&
                      maskType.getGranularity() == "b16";
  bool isB32Scatter = valueBits == 32 && indexElementType.getWidth() == 32 &&
                      maskType.getGranularity() == "b32";
  bool unsupportedContract = !isB8Scatter && !isB16Scatter && !isB32Scatter;
  if (unsupportedContract) {
    return fail("requires either 32-bit values with 32-bit indices and b32 "
                "mask, 16-bit values with 16-bit indices and b16 "
                "mask, or 8-bit values with 16-bit indices and b16 "
                "mask");
  }
  return success();
}

struct ScatterShapeTypes {
  VMIVRegType value;
  VMIVRegType indices;
  VMIMaskType mask;
};

static ScatterShapeTypes getScatterShapeTypes(VMIScatterOp op) {
  return ScatterShapeTypes{
      cast<VMIVRegType>(op.getValue().getType()),
      cast<VMIVRegType>(op.getIndices().getType()),
      cast<VMIMaskType>(op.getMask().getType())};
}

LogicalResult
checkSupportedScatterShape(VMIScatterOp op, std::string *reason) {
  ScatterShapeTypes types = getScatterShapeTypes(op);
  if (failed(checkScatterLayoutAndDestination(op, types.value, types.indices,
                                              types.mask, reason))) {
    return failure();
  }
  if (failed(checkScatterElementContract(types.value, types.indices,
                                         types.mask, reason))) {
    return failure();
  }

  return checkSupportedScatterPhysicalShape(types.value, types.indices,
                                            types.mask, true, reason);
}

LogicalResult
checkSinglePhysicalStrideAccess(Type dataType, Type maskType,
                                StringRef errorMessage, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<int64_t> dataArity = getVMIPhysicalArity(dataType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  bool hasArity = succeeded(dataArity) && succeeded(maskArity);
  if (!hasArity) {
    return fail("requires computable physical arity");
  }
  if (*dataArity != 1 || *maskArity != 1) {
    return fail(errorMessage);
  }
  return success();
}

struct StrideMemoryContract {
  Type dataType;
  Type maskType;
  VMILayoutAttr dataLayout;
  VMILayoutAttr maskLayout;
  Type pointerType;
  StringRef dataName;
  StringRef pointerDiagnostic;
  StringRef arityDiagnostic;
};

static LogicalResult checkStrideMemoryContract(
    const StrideMemoryContract &contract, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  if (!contract.dataLayout || !contract.maskLayout) {
    return fail(Twine("requires assigned ") + contract.dataName +
                " and mask layouts");
  }
  bool nonContiguousLayout =
      !contract.dataLayout.isContiguous() ||
      !contract.maskLayout.isContiguous();
  if (nonContiguousLayout) {
    return fail(Twine("requires contiguous ") + contract.dataName +
                " and mask layouts");
  }
  if (!isa<PtrType>(contract.pointerType)) {
    return fail(contract.pointerDiagnostic);
  }
  return checkSinglePhysicalStrideAccess(
      contract.dataType, contract.maskType, contract.arityDiagnostic, reason);
}

LogicalResult
checkSupportedStrideStoreShape(VMIStrideStoreOp op, std::string *reason) {
  auto valueType = cast<VMIVRegType>(op.getValue().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  if (failed(checkSupportedStoreShape(valueType, op.getDestination(),
                                      op.getDestination().getType(), reason))) {
    return failure();
  }

  StrideMemoryContract contract{
      valueType,
      maskType,
      valueType.getLayoutAttr(),
      maskType.getLayoutAttr(),
      op.getDestination().getType(),
      "value",
      "requires !pto.ptr destination because pto.vsstb is pointer-only",
      "currently supports one physical value/mask chunk"};
  return checkStrideMemoryContract(contract, reason);
}

LogicalResult
checkSupportedStrideLoadShape(VMIStrideLoadOp op, std::string *reason) {
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  StrideMemoryContract contract{
      resultType,
      maskType,
      resultType.getLayoutAttr(),
      maskType.getLayoutAttr(),
      op.getSource().getType(),
      "result",
      "requires !pto.ptr source because pto.vsldb is pointer-only",
      "currently supports one physical result/mask chunk"};
  return checkStrideMemoryContract(contract, reason);
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
  auto fail = [&reason](const Twine &message) {
    if (reason) {
      *reason = message.str();
    }
    return false;
  };

  if (auto createMask = mask.getDefiningOp<VMICreateMaskOp>()) {
    auto activeConstant =
        createMask.getActiveLanes().getDefiningOp<arith::ConstantOp>();
    if (!activeConstant) {
      return fail("create_mask active_lanes is dynamic");
    }
    auto activeAttr = dyn_cast<IntegerAttr>(activeConstant.getValue());
    if (!activeAttr) {
      return fail("create_mask active_lanes is not an integer constant");
    }
    return activeAttr.getInt() >= expectedLanes
               ? true
               : fail("create_mask active_lanes is smaller than the logical "
                      "lane count");
  }

  if (auto constantMask = mask.getDefiningOp<VMIConstantMaskOp>()) {
    auto denseAttr = dyn_cast<DenseIntElementsAttr>(constantMask.getValue());
    if (!denseAttr) {
      return fail("constant_mask is not a dense integer mask");
    }
    bool invalidElementCount = denseAttr.getNumElements() != expectedLanes;
    if (invalidElementCount) {
      return fail("constant_mask element count does not match the logical "
                  "lane count");
    }
    auto values = denseAttr.getValues<bool>();
    for (bool value : values) {
      if (!value) {
        return fail("constant_mask contains an inactive lane");
      }
    }
    return true;
  }

  return fail("mask is not a static all-active create_mask or constant_mask");
}

static LogicalResult checkExpandLoadRuntimeArity(
    VMIVRegType resultType, VMIVRegType passthruType, VMIMaskType maskType,
    std::string *reason) {
  auto fail = [&reason](const Twine &message) {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  FailureOr<int64_t> passthruArity = getVMIPhysicalArity(passthruType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  bool hasComputableArity = succeeded(resultArity) &&
                            succeeded(passthruArity) && succeeded(maskArity);
  if (!hasComputableArity) {
    return fail("runtime-mask path requires computable physical arity");
  }
  bool hasSingleChunk = *resultArity == 1 && *passthruArity == 1 &&
                        *maskArity == 1;
  if (!hasSingleChunk) {
    return fail("runtime-mask path currently supports only one physical "
                "chunk because prefix indices must not reset across chunks");
  }
  return success();
}

static LogicalResult checkExpandLoadRuntimeFullChunks(
    VMIVRegType resultType, VMIVRegType passthruType, VMIMaskType maskType,
    std::string *reason) {
  auto fail = [&reason](const Twine &message) {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  std::string fullChunkReason;
  std::string passthruReason;
  std::string maskFullReason;
  if (failed(checkFullDataPhysicalChunks(resultType, &fullChunkReason))) {
    return fail(Twine("runtime-mask result requires full physical chunks; ") +
                fullChunkReason);
  }
  if (failed(checkFullDataPhysicalChunks(passthruType, &passthruReason))) {
    return fail(Twine("runtime-mask passthru requires full physical chunks; ") +
                passthruReason);
  }
  if (failed(checkFullVMIPhysicalChunks(maskType, &maskFullReason))) {
    return fail(Twine("runtime-mask mask requires full physical chunks; ") +
                maskFullReason);
  }
  return success();
}

LogicalResult
checkSupportedExpandLoadRuntimePath(
    VMIExpandLoadOp op, VMIVRegType resultType, VMIVRegType passthruType,
    VMIMaskType maskType, StringRef allActivePathReason, std::string *reason) {
  auto fail = [&reason](const Twine &message) {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  if (!isa<PtrType>(op.getSource().getType())) {
    return fail(Twine("runtime-mask path requires !pto.ptr source because "
                      "pto.vgather2_bc is pointer-only; all-active path ") +
                allActivePathReason);
  }
  bool unsupportedResultWidth =
      pto::getPTOStorageElemBitWidth(resultType.getElementType()) != 32;
  if (unsupportedResultWidth) {
    return fail("runtime-mask path currently requires 32-bit result element "
                "type so prefix indices and gather result lane counts match");
  }
  bool unsupportedMaskGranularity = maskType.getGranularity() != "b32";
  if (unsupportedMaskGranularity) {
    return fail("runtime-mask path requires b32 mask granularity");
  }
  if (failed(checkExpandLoadRuntimeArity(resultType, passthruType, maskType,
                                         reason))) {
    return failure();
  }
  return checkExpandLoadRuntimeFullChunks(resultType, passthruType, maskType,
                                          reason);
}

LogicalResult
checkSupportedExpandLoadCommonShape(VMIExpandLoadOp op,
                                     VMIVRegType resultType,
                                     VMIVRegType passthruType,
                                     VMIMaskType maskType,
                                     VMIMemoryAccessPlan &accessPlan,
                                     std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  accessPlan = buildReadAccessPlan(op.getSource(), op.getOffset(), resultType,
                                   VMIMemoryCoverageKind::Predicate);
  if (!accessPlan.layoutSupport.isSupported()) {
    return fail(accessPlan.layoutSupport.reason);
  }
  bool missingLayout = !resultType.getLayoutAttr() ||
                       !passthruType.getLayoutAttr() ||
                       !maskType.getLayoutAttr();
  if (missingLayout) {
    return fail("requires assigned result, passthru, and mask layouts");
  }
  bool nonContiguousLayout = !resultType.getLayoutAttr().isContiguous() ||
                             !passthruType.getLayoutAttr().isContiguous() ||
                             !maskType.getLayoutAttr().isContiguous();
  if (nonContiguousLayout) {
    return fail("requires contiguous result, passthru, and mask layouts");
  }
  return success();
}

static bool hasSafeExpandLoadAllActivePath(
    VMIVRegType resultType, const VMIMemoryAccessPlan &accessPlan,
    bool staticAllActive, std::string *pathReason) {
  if (!staticAllActive) {
    return false;
  }
  std::string fullChunkReason;
  bool fullChunks =
      succeeded(checkFullDataPhysicalChunks(resultType, &fullChunkReason));
  if (fullChunks || accessPlan.front().readSafety.proven) {
    return true;
  }
  std::string fallbackReason =
      getUnavailableReadFallbackReason(VMIMemoryCoverageKind::Predicate);
  *pathReason =
      (Twine("requires full physical chunks or statically safe full-read "
             "footprint; value ") +
       fullChunkReason + ", safe-read proof " +
       accessPlan.front().readSafety.reason +
       "; fallback unavailable: " + fallbackReason)
          .str();
  return false;
}

LogicalResult
checkSupportedExpandLoadShape(VMIExpandLoadOp op, std::string *reason) {
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  auto passthruType = cast<VMIVRegType>(op.getPassthru().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  VMIMemoryAccessPlan accessPlan;
  if (failed(checkSupportedExpandLoadCommonShape(
          op, resultType, passthruType, maskType, accessPlan, reason))) {
    return failure();
  }

  std::string maskReason;
  bool staticAllActive = isStaticAllActiveMask(
      op.getMask(), resultType.getElementCount(), &maskReason);

  std::string allActivePathReason;
  bool staticFullChunks = hasSafeExpandLoadAllActivePath(
      resultType, accessPlan, staticAllActive, &allActivePathReason);
  if (staticFullChunks) {
    return success();
  }

  if (!staticAllActive) {
    allActivePathReason =
        maskReason.empty() ? "requires static all-active mask" : maskReason;
  }

  return checkSupportedExpandLoadRuntimePath(
      op, resultType, passthruType, maskType, allActivePathReason, reason);
}

static LogicalResult checkMaskedStoreFullChunks(VMIVRegType valueType,
                                                VMIMaskType maskType,
                                                std::string &valueReason,
                                                std::string &maskReason) {
  return succeeded(checkFullDataPhysicalChunks(valueType, &valueReason)) &&
                 succeeded(checkFullVMIPhysicalChunks(maskType, &maskReason))
             ? success()
             : failure();
}

static LogicalResult checkMaskedStoreLayoutAndArity(
    VMIVRegType valueType, VMIMaskType maskType, std::string &valueReason,
    std::string &maskReason, std::string *reason) {
  FailureOr<AssignedValueMaskLayouts> layouts =
      getAssignedValueMaskLayouts(valueType, maskType, reason);
  if (failed(layouts)) {
    return failure();
  }
  FailureOr<int64_t> valueArity = getVMIPhysicalArity(valueType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  bool mismatchedArity = failed(valueArity) || failed(maskArity) ||
                         *valueArity != *maskArity;
  if (mismatchedArity) {
    return emitLogicalFailure(reason, "requires matching value/mask physical arity");
  }
  if (layouts->value.hasDenseLaneStride()) {
    VMILayoutSupport supports;
    if (succeeded(supports.getMaskedStoreLayoutFact(valueType, maskType,
                                                    reason))) {
      return success();
    }
  }
  std::string valueMaterializationReason;
  FailureOr<int64_t> valueParts = getContiguousMaterializationPartCount(
      valueType, &valueMaterializationReason);
  if (failed(valueParts)) {
    return emitLogicalFailure(reason,
                              Twine("value cannot materialize to contiguous; value ") +
                                  valueReason + ", materialization " +
                                  valueMaterializationReason);
  }
  std::string maskMaterializationReason;
  FailureOr<int64_t> maskParts = getContiguousMaterializationPartCount(
      maskType, &maskMaterializationReason);
  if (failed(maskParts)) {
    return emitLogicalFailure(reason,
                              Twine("mask cannot materialize to contiguous; mask ") +
                                  maskReason + ", materialization " +
                                  maskMaterializationReason);
  }
  if (*valueParts != *maskParts) {
    return emitLogicalFailure(
        reason, "requires value/mask contiguous materialization arity to match");
  }
  return success();
}

LogicalResult
checkSupportedMaskedStoreShape(VMIVRegType valueType, VMIMaskType maskType,
                               Value destination, Type destinationType,
                               std::string *reason) {
  VMIMemoryAccessPlan accessPlan =
      buildWriteAccessPlan(destination, destinationType, valueType,
                           VMIMemoryCoverageKind::Predicate);
  if (!accessPlan.layoutSupport.isSupported()) {
    if (reason) {
      *reason = accessPlan.layoutSupport.reason;
    }
    return failure();
  }

  std::string valueReason;
  std::string maskReason;
  if (succeeded(checkMaskedStoreFullChunks(valueType, maskType, valueReason,
                                            maskReason))) {
    return success();
  }

  return checkMaskedStoreLayoutAndArity(valueType, maskType, valueReason,
                                        maskReason, reason);
}

FailureOr<int64_t> getContiguousActiveDataLanes(VMIVRegType vmiType,
                                                int64_t chunk) {
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(vmiType.getElementType());
  if (failed(lanesPerPart)) {
    return failure();
  }

  int64_t remaining = vmiType.getElementCount() - chunk * *lanesPerPart;
  return std::clamp<int64_t>(remaining, 0, *lanesPerPart);
}

FailureOr<int64_t> getActiveDataLanesInPhysicalChunk(VMIVRegType vmiType,
                                                     int64_t chunk) {
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(vmiType.getElementType());
  if (failed(lanesPerPart)) {
    return failure();
  }

  int64_t active = 0;
  for (int64_t lane = 0; lane < *lanesPerPart; ++lane) {
    FailureOr<bool> padding = isPaddingLane(vmiType, /*part=*/0, chunk, lane);
    if (failed(padding)) {
      return failure();
    }
    if (!*padding) {
      ++active;
    }
  }
  return active;
}

static FailureOr<std::pair<int64_t, int64_t>> getContiguousStoreMaskLaneCounts(
    VMIVRegType vmiType, int64_t chunk) {
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(vmiType.getElementType());
  if (failed(lanesPerPart)) {
    return failure();
  }
  FailureOr<int64_t> activeLanes = getContiguousActiveDataLanes(vmiType, chunk);
  if (failed(activeLanes)) {
    return failure();
  }
  return std::make_pair(*lanesPerPart, *activeLanes);
}

FailureOr<Value> createContiguousStoreMask(Location loc, VMIVRegType vmiType,
                                           int64_t chunk, VRegType vregType,
                                           PatternRewriter &rewriter) {
  FailureOr<std::pair<int64_t, int64_t>> laneCounts =
      getContiguousStoreMaskLaneCounts(vmiType, chunk);
  if (failed(laneCounts)) {
    return failure();
  }
  if (laneCounts->second == laneCounts->first) {
    return createAllTrueMaskForVReg(loc, vregType, rewriter);
  }

  FailureOr<MaskType> maskType =
      getMaskTypeForVReg(vregType, rewriter.getContext());
  if (failed(maskType)) {
    return failure();
  }
  FailureOr<std::pair<Value, Value>> maskAndRemaining = createRuntimePrefixMask(
      loc, *maskType, createI32Constant(loc, laneCounts->second, rewriter),
      rewriter);
  if (failed(maskAndRemaining)) {
    return failure();
  }
  return maskAndRemaining->first;
}

FailureOr<Value> createMaskedStorePredicate(Location loc, VMIVRegType vmiType,
                                            int64_t chunk, Value userMask,
                                            VRegType vregType,
                                            PatternRewriter &rewriter) {
  FailureOr<std::pair<int64_t, int64_t>> laneCounts =
      getContiguousStoreMaskLaneCounts(vmiType, chunk);
  if (failed(laneCounts)) {
    return failure();
  }
  if (laneCounts->second == laneCounts->first) {
    return userMask;
  }

  auto maskType = dyn_cast<MaskType>(userMask.getType());
  if (!maskType) {
    return failure();
  }
  FailureOr<Value> tailMask =
      createContiguousStoreMask(loc, vmiType, chunk, vregType, rewriter);
  FailureOr<Value> allTrue = createAllTrueMask(loc, maskType, rewriter);
  bool failedTailMaskMaterialization = failed(tailMask) || failed(allTrue);
  if (failedTailMaskMaterialization) {
    return failure();
  }
  return rewriter.create<PandOp>(loc, maskType, userMask, *tailMask, *allTrue)
      .getResult();
}

static FailureOr<Value> compactDenseLaneStrideStorePredicate(
    Location loc, Value userMask, VMILayoutAttr layout, StringRef targetGranularity,
    PatternRewriter &rewriter) {
  auto sourceMaskType = dyn_cast<MaskType>(userMask.getType());
  if (!sourceMaskType || !layout) {
    return failure();
  }
  auto targetMaskType = MaskType::get(rewriter.getContext(), targetGranularity);
  Value compactMask = userMask;
  StringRef sourceGranularity = sourceMaskType.getGranularity();
  StringAttr lower = rewriter.getStringAttr("LOWER");
  if (sourceGranularity == targetGranularity) {
    return compactMask;
  }
  bool unpackLaneStride2 = layout.getLaneStride() == 2;
  if (unpackLaneStride2) {
    Value unpacked = rewriter
                         .create<PunpackOp>(loc, targetMaskType, compactMask,
                                            lower)
                         .getResult();
    return unpacked;
  }
  bool supportsLaneStride4 = layout.getLaneStride() == 4 &&
                             sourceGranularity == "b8" &&
                             targetGranularity == "b32";
  if (!supportsLaneStride4) {
    return failure();
  }
  auto b16MaskType = MaskType::get(rewriter.getContext(), "b16");
  compactMask = rewriter
                    .create<PunpackOp>(loc, b16MaskType, compactMask, lower)
                    .getResult();
  return rewriter
      .create<PunpackOp>(loc, targetMaskType, compactMask, lower)
      .getResult();
}

FailureOr<Value> createDenseLaneStrideStorePredicate(
    Location loc, VMIVRegType vmiType, int64_t chunk, Value userMask,
    StringRef targetGranularity, PatternRewriter &rewriter) {
  VMILayoutAttr layout = vmiType.getLayoutAttr();
  FailureOr<Value> compactMask = compactDenseLaneStrideStorePredicate(
      loc, userMask, layout, targetGranularity, rewriter);
  if (failed(compactMask)) {
    return failure();
  }

  FailureOr<int64_t> activeLanes =
      getActiveDataLanesInPhysicalChunk(vmiType, chunk);
  FailureOr<int64_t> maskLanes = getMaskLanesPerPart(targetGranularity);
  bool failedLaneCountQuery = failed(activeLanes) || failed(maskLanes);
  if (failedLaneCountQuery) {
    return failure();
  }
  if (*activeLanes == *maskLanes) {
    return *compactMask;
  }

  auto targetMaskType = MaskType::get(rewriter.getContext(), targetGranularity);
  FailureOr<Value> tailMask = createPrefixMaskForActiveLanes(
      loc, targetMaskType, *activeLanes, rewriter);
  FailureOr<Value> allTrue = createAllTrueMask(loc, targetMaskType, rewriter);
  bool failedTailMaskMaterialization = failed(tailMask) || failed(allTrue);
  if (failedTailMaskMaterialization) {
    return failure();
  }
  return rewriter
      .create<PandOp>(loc, targetMaskType, *compactMask, *tailMask, *allTrue)
      .getResult();
}

static FailureOr<std::optional<VMIPhysicalLane>> getShuffleSourcePhysicalLane(
    VMIVRegType sourceType, VMIVRegType resultType,
    ArrayRef<int64_t> indices, int64_t resultPart, int64_t resultChunk,
    int64_t lane, std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<std::optional<VMIPhysicalLane>> {
    if (reason) {
      *reason = message.str();
    }
    return std::optional<VMIPhysicalLane>();
  };
  FailureOr<bool> padding =
      isPaddingLane(resultType, resultPart, resultChunk, lane);
  if (failed(padding)) {
    return fail("failed to classify result padding lanes");
  }
  if (*padding) {
    return std::optional<VMIPhysicalLane>();
  }
  FailureOr<int64_t> logicalLane =
      mapPhysicalLaneToLogical(resultType, resultPart, resultChunk, lane);
  bool logicalLaneOutOfRange =
      failed(logicalLane) ||
      *logicalLane >= static_cast<int64_t>(indices.size());
  if (logicalLaneOutOfRange) {
    return fail("failed to map result lane");
  }
  FailureOr<VMIPhysicalLane> sourcePhysical =
      mapLogicalLaneToPhysical(sourceType, indices[*logicalLane]);
  if (failed(sourcePhysical)) {
    return fail("failed to map source lane");
  }
  if (sourcePhysical->lane != lane) {
    return fail("requires same-lane physical chunks");
  }
  return std::optional<VMIPhysicalLane>(*sourcePhysical);
}

static FailureOr<int64_t> computeShuffleForwardingSourceChunk(
    VMIVRegType sourceType, VMIVRegType resultType, ArrayRef<int64_t> indices,
    int64_t resultPart, int64_t resultChunk, int64_t lanesPerPart,
    std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<int64_t> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  std::optional<int64_t> sourcePart;
  std::optional<int64_t> sourceChunk;
  for (int64_t lane = 0; lane < lanesPerPart; ++lane) {
    FailureOr<std::optional<VMIPhysicalLane>> sourcePhysical =
        getShuffleSourcePhysicalLane(
        sourceType, resultType, indices, resultPart, resultChunk, lane,
        reason);
    if (failed(sourcePhysical)) {
      return failure();
    }
    if (!*sourcePhysical) {
      continue;
    }
    if (!sourcePart) {
      sourcePart = (*sourcePhysical)->part;
      sourceChunk = (*sourcePhysical)->chunk;
    } else if (*sourcePart != (*sourcePhysical)->part ||
               *sourceChunk != (*sourcePhysical)->chunk) {
      return fail("requires one source chunk per result chunk");
    }
  }
  if (!sourcePart || !sourceChunk) {
    return fail("requires at least one logical lane per result chunk");
  }
  FailureOr<int64_t> sourceFlatIndex =
      getDataFlatPartIndex(sourceType, *sourcePart, *sourceChunk);
  if (failed(sourceFlatIndex)) {
    return fail("source part range is out of bounds");
  }
  return *sourceFlatIndex;
}

struct ShuffleForwardingInputPlan {
  VMIVRegType sourceType;
  VMIVRegType resultType;
  ArrayRef<int64_t> indices;
  int64_t lanesPerPart;
  int64_t resultFactor;
};

static FailureOr<ShuffleForwardingInputPlan>
getShuffleForwardingInputPlan(VMIShuffleOp op, std::string *reason) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (op.getIndices().empty()) {
    if (reason) {
      *reason = "requires non-empty indices";
    }
    return failure();
  }
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(sourceType.getElementType());
  if (failed(lanesPerPart)) {
    if (reason) {
      *reason = "requires known lanes per physical part";
    }
    return failure();
  }
  FailureOr<int64_t> resultFactor = getDataLayoutFactor(resultType);
  if (failed(resultFactor)) {
    if (reason) {
      *reason = "requires assigned result layout";
    }
    return failure();
  }
  return ShuffleForwardingInputPlan{sourceType, resultType, op.getIndices(),
                                    *lanesPerPart, *resultFactor};
}

static FailureOr<SmallVector<int64_t>> materializeShuffleForwardingSourceParts(
    const ShuffleForwardingInputPlan &input, std::string *reason) {
  SmallVector<int64_t> sourceFlatIndices;
  for (int64_t resultPart = 0; resultPart < input.resultFactor; ++resultPart) {
    FailureOr<int64_t> resultChunks =
        getDataChunksInPart(input.resultType, resultPart);
    if (failed(resultChunks)) {
      if (reason) {
        *reason = "requires known result physical chunks";
      }
      return failure();
    }
    for (int64_t resultChunk = 0; resultChunk < *resultChunks; ++resultChunk) {
      FailureOr<int64_t> sourceFlatIndex =
          computeShuffleForwardingSourceChunk(
              input.sourceType, input.resultType, input.indices, resultPart,
              resultChunk, input.lanesPerPart, reason);
      if (failed(sourceFlatIndex)) {
        return failure();
      }
      sourceFlatIndices.push_back(*sourceFlatIndex);
    }
  }

  return sourceFlatIndices;
}

FailureOr<SmallVector<int64_t>>
computeShuffleForwardingSourceParts(VMIShuffleOp op, std::string *reason) {
  FailureOr<ShuffleForwardingInputPlan> input =
      getShuffleForwardingInputPlan(op, reason);
  if (failed(input)) {
    return failure();
  }
  return materializeShuffleForwardingSourceParts(*input, reason);
}

struct ShuffleVselrPlan {
  int64_t sourceFlatIndex = 0;
  int64_t baseLane = 0;
  bool descending = false;
};

static FailureOr<bool> getShuffleLaneDirection(
    int64_t baseLane, int64_t resultLane, int64_t sourceLane,
    std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<bool> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  bool ascending = sourceLane == baseLane + resultLane;
  bool descending = sourceLane == baseLane - resultLane;
  bool unsupportedDirection = !ascending && !descending;
  if (unsupportedDirection) {
    return fail("requires ASC or DESC affine source lane indices");
  }
  return descending && !ascending;
}

static FailureOr<VMIPhysicalLane> getShuffleSourceLane(
    VMIVRegType sourceType, VMIVRegType resultType,
    ArrayRef<int64_t> indices, int64_t resultPart, int64_t resultChunk,
    int64_t lane, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<VMIPhysicalLane> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<bool> padding =
      isPaddingLane(resultType, resultPart, resultChunk, lane);
  bool invalidPadding = failed(padding) || *padding;
  if (invalidPadding) {
    return fail("requires full physical result chunks");
  }
  FailureOr<int64_t> logicalLane =
      mapPhysicalLaneToLogical(resultType, resultPart, resultChunk, lane);
  bool logicalLaneOutOfRange =
      succeeded(logicalLane) &&
      *logicalLane >= static_cast<int64_t>(indices.size());
  bool invalidLogicalLane = failed(logicalLane) || logicalLaneOutOfRange;
  if (invalidLogicalLane) {
    return fail("failed to map result lane");
  }
  FailureOr<VMIPhysicalLane> sourceLane =
      mapLogicalLaneToPhysical(sourceType, indices[*logicalLane]);
  if (failed(sourceLane)) {
    return fail("failed to map source lane");
  }
  return *sourceLane;
}

struct ShuffleChunkLaneState {
  int64_t sourcePart = 0;
  int64_t sourceChunk = 0;
  int64_t baseLane = 0;
  std::optional<bool> descending;
};

static FailureOr<ShuffleChunkLaneState> updateShuffleChunkLaneState(
    VMIVRegType sourceType, VMIVRegType resultType, ArrayRef<int64_t> indices,
    int64_t resultPart, int64_t resultChunk, int64_t lane,
    std::optional<ShuffleChunkLaneState> state,
    std::string *reason) {
  FailureOr<VMIPhysicalLane> sourcePhysical = getShuffleSourceLane(
      sourceType, resultType, indices, resultPart, resultChunk, lane, reason);
  if (failed(sourcePhysical)) {
    return failure();
  }
  if (!state) {
    return ShuffleChunkLaneState{sourcePhysical->part, sourcePhysical->chunk,
                                 sourcePhysical->lane, std::nullopt};
  }
  bool sourceChunkMismatch = state->sourcePart != sourcePhysical->part ||
                             state->sourceChunk != sourcePhysical->chunk;
  if (sourceChunkMismatch) {
    if (reason) {
      *reason = "requires one source chunk per result chunk";
    }
    return failure();
  }
  FailureOr<bool> laneDescending = getShuffleLaneDirection(
      state->baseLane, lane, sourcePhysical->lane, reason);
  if (failed(laneDescending)) {
    return failure();
  }
  if (state->descending && *laneDescending != *state->descending) {
    if (reason) {
      *reason = "requires one index order per result chunk";
    }
    return failure();
  }
  state->descending = *laneDescending;
  return *state;
}

FailureOr<ShuffleVselrPlan> computeShuffleVselrPlanForChunk(
    VMIVRegType sourceType, VMIVRegType resultType, ArrayRef<int64_t> indices,
    int64_t resultPart, int64_t resultChunk, int64_t lanesPerPart,
    std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<ShuffleVselrPlan> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  std::optional<ShuffleChunkLaneState> state;
  for (int64_t lane = 0; lane < lanesPerPart; ++lane) {
    FailureOr<ShuffleChunkLaneState> nextState = updateShuffleChunkLaneState(
        sourceType, resultType, indices, resultPart, resultChunk, lane,
        state, reason);
    if (failed(nextState)) {
      return failure();
    }
    state = *nextState;
  }
  FailureOr<int64_t> sourceFlatIndex =
      getDataFlatPartIndex(sourceType, state->sourcePart, state->sourceChunk);
  if (failed(sourceFlatIndex)) {
    return fail("source part range is out of bounds");
  }
  return ShuffleVselrPlan{*sourceFlatIndex, state->baseLane,
                          state->descending.value_or(false)};
}

static FailureOr<int64_t> getShuffleResultChunkCount(
    VMIVRegType resultType, int64_t resultPart, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<int64_t> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<int64_t> resultChunks =
      getDataChunksInPart(resultType, resultPart);
  if (failed(resultChunks)) {
    return fail("requires known result physical chunks");
  }
  return *resultChunks;
}

FailureOr<int64_t> computeShuffleLane0SplatSourcePart(VMIShuffleOp op,
                                                      std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<int64_t> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  ArrayRef<int64_t> indices = op.getIndices();
  if (indices.empty()) {
    return fail("requires non-empty indices");
  }
  bool hasNonZeroIndex =
      !llvm::all_of(indices, [](int64_t index) { return index == 0; });
  if (hasNonZeroIndex) {
    return fail("requires every result lane to select source lane 0");
  }

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  FailureOr<VMIPhysicalLane> sourceLane =
      mapLogicalLaneToPhysical(sourceType, 0);
  if (failed(sourceLane)) {
    return fail("failed to map source lane 0");
  }
  FailureOr<int64_t> sourceFlatIndex =
      getDataFlatPartIndex(sourceType, sourceLane->part, sourceLane->chunk);
  if (failed(sourceFlatIndex)) {
    return fail("source lane 0 part range is out of bounds");
  }
  return *sourceFlatIndex;
}

struct ShuffleVselrInputPlan {
  VMIVRegType sourceType;
  VMIVRegType resultType;
  ArrayRef<int64_t> indices;
  int64_t lanesPerPart;
  int64_t resultFactor;
};

static FailureOr<ShuffleVselrInputPlan> buildShuffleVselrInputPlan(
    VMIShuffleOp op, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<ShuffleVselrInputPlan> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(sourceType.getElementType());
  if (failed(lanesPerPart)) {
    return fail("requires known lanes per physical part");
  }
  ArrayRef<int64_t> indices = op.getIndices();
  if (indices.empty()) {
    return fail("requires non-empty indices");
  }
  FailureOr<int64_t> resultFactor = getDataLayoutFactor(resultType);
  if (failed(resultFactor)) {
    return fail("requires assigned result layout");
  }
  return ShuffleVselrInputPlan{sourceType, resultType, indices,
                                *lanesPerPart, *resultFactor};
}

static FailureOr<SmallVector<ShuffleVselrPlan>> materializeShuffleVselrPlans(
    const ShuffleVselrInputPlan &input, std::string *reason) {
  SmallVector<ShuffleVselrPlan> plans;
  for (int64_t resultPart = 0; resultPart < input.resultFactor; ++resultPart) {
    FailureOr<int64_t> resultChunks =
        getShuffleResultChunkCount(input.resultType, resultPart, reason);
    if (failed(resultChunks)) {
      return failure();
    }
    for (int64_t resultChunk = 0; resultChunk < *resultChunks; ++resultChunk) {
      FailureOr<ShuffleVselrPlan> plan = computeShuffleVselrPlanForChunk(
          input.sourceType, input.resultType, input.indices, resultPart,
          resultChunk, input.lanesPerPart, reason);
      if (failed(plan)) {
        return failure();
      }
      plans.push_back(*plan);
    }
  }
  return plans;
}

FailureOr<SmallVector<ShuffleVselrPlan>>
computeShuffleVselrPlans(VMIShuffleOp op, std::string *reason) {
  FailureOr<ShuffleVselrInputPlan> input =
      buildShuffleVselrInputPlan(op, reason);
  if (failed(input)) {
    return failure();
  }
  return materializeShuffleVselrPlans(*input, reason);
}

struct ConstantMaskChunkMaterialization {
  SmallVector<int8_t> activeLanes;
};

template <typename LanePredicate>
FailureOr<SmallVector<ConstantMaskChunkMaterialization>>
materializeMaskChunks(VMIMaskType resultVMIType, int64_t lanesPerPart,
                      LanePredicate isActive, std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<SmallVector<ConstantMaskChunkMaterialization>> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  VMILayoutAttr layout = resultVMIType.getLayoutAttr();
  int64_t factor = layout.isDenseSplit() ? layout.getFactor() : 1;
  SmallVector<ConstantMaskChunkMaterialization> materializations;
  for (int64_t part = 0; part < factor; ++part) {
    for (int64_t chunk = 0;; ++chunk) {
      bool anyLane = false;
      ConstantMaskChunkMaterialization materialization;
      materialization.activeLanes.reserve(lanesPerPart);
      for (int64_t lane = 0; lane < lanesPerPart; ++lane) {
        FailureOr<bool> padding =
            isPaddingLane(resultVMIType, part, chunk, lane);
        if (failed(padding)) {
          return fail("failed to map physical padding lane");
        }
        if (*padding) {
          materialization.activeLanes.push_back(0);
          continue;
        }
        anyLane = true;
        FailureOr<int64_t> logicalLane =
            mapPhysicalLaneToLogical(resultVMIType, part, chunk, lane);
        if (failed(logicalLane)) {
          return fail("failed to map physical lane");
        }
        materialization.activeLanes.push_back(isActive(*logicalLane) ? 1 : 0);
      }
      if (!anyLane) {
        break;
      }
      materializations.push_back(std::move(materialization));
    }
  }
  return materializations;
}

FailureOr<SmallVector<ConstantMaskChunkMaterialization>>
computeConstantMaskMaterialization(VMIConstantMaskOp op, std::string *reason) {
  auto denseAttr = dyn_cast<DenseIntElementsAttr>(op.getValue());
  if (!denseAttr) {
    return emitFailure<SmallVector<ConstantMaskChunkMaterialization>>(
        reason, "only dense integer mask constants are supported");
  }

  auto resultVMIType = cast<VMIMaskType>(op.getResult().getType());
  VMILayoutAttr layout = resultVMIType.getLayoutAttr();
  if (!layout ||
      !VMIMaskType::isConcreteGranularity(resultVMIType.getGranularity())) {
    return emitFailure<SmallVector<ConstantMaskChunkMaterialization>>(
        reason, "requires concrete layout and granularity");
  }

  FailureOr<StringRef> physicalGranularity =
      getVMIMaskPhysicalGranularity(resultVMIType);
  FailureOr<int64_t> lanesPerPart =
      failed(physicalGranularity)
          ? FailureOr<int64_t>(failure())
          : getMaskLanesPerPart(*physicalGranularity);
  if (failed(lanesPerPart)) {
    return emitFailure<SmallVector<ConstantMaskChunkMaterialization>>(
        reason, "requires known physical mask lanes per part");
  }

  auto boolValues = denseAttr.getValues<bool>();
  return materializeMaskChunks(
      resultVMIType, *lanesPerPart,
      [&boolValues](int64_t logicalLane) { return boolValues[logicalLane]; },
      reason);
}

struct GroupMaskMaterializationPlan {
  int64_t lanesPerPart;
  int64_t groupSize;
  int64_t activeElems;
};

static FailureOr<GroupMaskMaterializationPlan>
buildGroupMaskMaterializationPlan(VMICreateGroupMaskOp op,
                                  VMIMaskType resultVMIType,
                                  std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<GroupMaskMaterializationPlan> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  auto activeConstant =
      op.getActiveElemsPerGroup().getDefiningOp<arith::ConstantOp>();
  if (!activeConstant) {
    return fail("requires constant active_elems_per_group");
  }
  auto activeAttr = dyn_cast<IntegerAttr>(activeConstant.getValue());
  if (!activeAttr) {
    return fail("active_elems_per_group must be an integer constant");
  }
  VMILayoutAttr layout = resultVMIType.getLayoutAttr();
  bool invalidMaskType =
      !layout || !VMIMaskType::isConcreteGranularity(
                     resultVMIType.getGranularity());
  if (invalidMaskType) {
    return fail("requires concrete layout and granularity");
  }
  FailureOr<StringRef> physicalGranularity =
      getVMIMaskPhysicalGranularity(resultVMIType);
  FailureOr<int64_t> lanesPerPart =
      failed(physicalGranularity)
          ? FailureOr<int64_t>(failure())
          : getMaskLanesPerPart(*physicalGranularity);
  if (failed(lanesPerPart)) {
    return fail("requires known physical mask lanes per part");
  }
  int64_t numGroups = op.getNumGroupsAttr().getInt();
  int64_t groupSize = op.getGroupSizeAttr().getInt();
  bool invalidShape =
      numGroups <= 0 || groupSize <= 0 ||
      resultVMIType.getElementCount() != numGroups * groupSize;
  if (invalidShape) {
    return fail("requires result lane count to match num_groups * group_size");
  }
  int64_t activeElems = std::clamp<int64_t>(activeAttr.getInt(), 0, groupSize);
  return GroupMaskMaterializationPlan{*lanesPerPart, groupSize, activeElems};
}

FailureOr<SmallVector<ConstantMaskChunkMaterialization>>
computeGroupMaskMaterializationForType(VMICreateGroupMaskOp op,
                                       VMIMaskType resultVMIType,
                                       std::string *reason) {
  FailureOr<GroupMaskMaterializationPlan> plan =
      buildGroupMaskMaterializationPlan(op, resultVMIType, reason);
  if (failed(plan)) {
    return failure();
  }

  return materializeMaskChunks(
      resultVMIType, plan->lanesPerPart,
      [groupSize = plan->groupSize,
       activeElems = plan->activeElems](int64_t logicalLane) {
        return logicalLane % groupSize < activeElems;
      },
      reason);
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
  if (modulus <= 0) {
    return failure();
  }

  auto vectorType = dyn_cast<VRegType>(value.getType());
  if (!vectorType) {
    return failure();
  }

  std::optional<int64_t> shift = getPowerOfTwoLog2(modulus);
  if (!shift) {
    return failure();
  }
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

static FailureOr<Value> applyGroupMaskPadding(
    VMICreateGroupMaskOp op, VMIMaskType resultVMIType, MaskType maskType,
    Value predicate, int64_t part, int64_t chunk, int64_t lanesPerPart,
    Value allMask, PatternRewriter &rewriter) {
  SmallVector<int8_t> validLanes;
  validLanes.reserve(lanesPerPart);
  bool hasPadding = false;
  for (int64_t lane = 0; lane < lanesPerPart; ++lane) {
    FailureOr<bool> padding =
        isPaddingLane(resultVMIType, part, chunk, lane);
    if (failed(padding)) {
      return rewriter.notifyMatchFailure(
          op, "failed to classify dynamic create_group_mask padding");
    }
    validLanes.push_back(*padding ? 0 : 1);
    hasPadding |= *padding;
  }
  if (!hasPadding) {
    return predicate;
  }
  FailureOr<Value> validMask = materializeConstantMaskChunk(
      op.getLoc(), maskType, validLanes, rewriter);
  if (failed(validMask)) {
    return rewriter.notifyMatchFailure(
        op, "failed to materialize dynamic create_group_mask padding mask");
  }
  return rewriter
      .create<PandOp>(op.getLoc(), maskType, predicate, *validMask, allMask)
      .getResult();
}

struct DynamicGroupMaskBlockIndex {
  Value partBlock;
  Value inBlockLane;
  std::optional<int64_t> blockShift;
};

static FailureOr<DynamicGroupMaskBlockIndex> buildDynamicGroupMaskBlockIndex(
    VMICreateGroupMaskOp op, int64_t blockElems, int64_t chunk,
    int64_t lanesPerPart, Value allMask, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  MLIRContext *ctx = rewriter.getContext();
  Type i32 = rewriter.getI32Type();
  auto indexVectorType = VRegType::get(ctx, lanesPerPart, i32);
  Value chunkBase = createI32Constant(loc, chunk * lanesPerPart, rewriter);
  Value indexInPart =
      rewriter.create<VciOp>(loc, indexVectorType, chunkBase, StringAttr{})
          .getResult();
  Value partBlock = indexInPart;
  Value inBlockLane = createI32Constant(loc, 0, rewriter);
  std::optional<int64_t> blockShiftValue = getPowerOfTwoLog2(blockElems);
  if (blockElems != 1) {
    if (!blockShiftValue) {
      return rewriter.notifyMatchFailure(
          op, "dynamic create_group_mask block size must be a power of two");
    }
    Value blockShift = createI16Constant(loc, *blockShiftValue, rewriter);
    partBlock = rewriter
                    .create<VshrsOp>(loc, indexVectorType, indexInPart,
                                     blockShift, allMask)
                    .getResult();
    Value blockBase = rewriter
                          .create<VshlsOp>(loc, indexVectorType, partBlock,
                                           blockShift, allMask)
                          .getResult();
    inBlockLane = rewriter
                      .create<VsubOp>(loc, indexVectorType, indexInPart,
                                      blockBase, allMask)
                      .getResult();
  }
  return DynamicGroupMaskBlockIndex{partBlock, inBlockLane, blockShiftValue};
}

static Value buildDynamicGroupMaskLogicalLane(
    VMICreateGroupMaskOp op, int64_t factor, int64_t blockElems, int64_t part,
    const DynamicGroupMaskBlockIndex &blockIndex, Value allMask,
    PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  auto indexVectorType = cast<VRegType>(blockIndex.partBlock.getType());
  Value factorScalar = createI32Constant(loc, factor, rewriter);
  Value logicalBlock = rewriter
                           .create<VmulsOp>(loc, indexVectorType,
                                            blockIndex.partBlock,
                                            factorScalar, allMask)
                           .getResult();
  if (part != 0) {
    Value partScalar = createI32Constant(loc, part, rewriter);
    logicalBlock = rewriter
                       .create<VaddsOp>(loc, indexVectorType, logicalBlock,
                                        partScalar, allMask)
                       .getResult();
  }
  Value logicalLane = logicalBlock;
  if (blockElems != 1) {
    Value blockShift = createI16Constant(loc, *blockIndex.blockShift, rewriter);
    Value logicalBlockBase = rewriter
                                 .create<VshlsOp>(loc, indexVectorType,
                                                  logicalBlock, blockShift,
                                                  allMask)
                                 .getResult();
    logicalLane = rewriter
                      .create<VaddOp>(loc, indexVectorType, logicalBlockBase,
                                      blockIndex.inBlockLane, allMask)
                      .getResult();
  }
  return logicalLane;
}

static FailureOr<Value> buildDynamicGroupMaskLaneIndex(
    VMICreateGroupMaskOp op, int64_t factor, int64_t blockElems, int64_t part,
    int64_t chunk, int64_t lanesPerPart, Value allMask,
    PatternRewriter &rewriter) {
  FailureOr<DynamicGroupMaskBlockIndex> blockIndex =
      buildDynamicGroupMaskBlockIndex(op, blockElems, chunk, lanesPerPart,
                                      allMask, rewriter);
  if (failed(blockIndex)) {
    return failure();
  }
  return buildDynamicGroupMaskLogicalLane(op, factor, blockElems, part,
                                          *blockIndex, allMask, rewriter);
}

FailureOr<Value> materializeDynamicGroupMaskChunk(
    VMICreateGroupMaskOp op, VMIMaskType resultVMIType, Type resultType,
    Value activeI32, int64_t factor, int64_t blockElems, int64_t part,
    int64_t chunk, int64_t lanesPerPart, PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message) -> FailureOr<Value> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  auto maskType = dyn_cast<MaskType>(resultType);
  if (!maskType || !maskType.isB32()) {
    return fail("dynamic create_group_mask result must be b32 mask");
  }
  Location loc = op.getLoc();
  FailureOr<Value> allMask = createAllTrueMask(loc, maskType, rewriter);
  if (failed(allMask)) {
    return fail("failed to create dynamic create_group_mask all mask");
  }
  FailureOr<Value> logicalLane = buildDynamicGroupMaskLaneIndex(
      op, factor, blockElems, part, chunk, lanesPerPart, *allMask, rewriter);
  if (failed(logicalLane)) {
    return fail("failed to compute dynamic create_group_mask lane index");
  }
  FailureOr<Value> laneInGroup = createPowerOfTwoRemainder(
      loc, *logicalLane, op.getGroupSizeAttr().getInt(), *allMask, rewriter);
  if (failed(laneInGroup)) {
    return fail("failed to compute dynamic create_group_mask lane index");
  }
  Value predicate =
      rewriter
          .create<VcmpsOp>(loc, maskType, *laneInGroup, activeI32, *allMask,
                           rewriter.getStringAttr("lt"))
          .getResult();
  FailureOr<Value> paddedPredicate = applyGroupMaskPadding(
      op, resultVMIType, maskType, predicate, part, chunk, lanesPerPart,
      *allMask, rewriter);
  if (failed(paddedPredicate)) {
    return failure();
  }
  return *paddedPredicate;
}

struct DynamicGroupMaskPlan {
  VMILayoutAttr layout;
  int64_t factor;
  int64_t blockElems;
  int64_t lanesPerPart;
  int64_t arity;
};

static LogicalResult checkDynamicGroupMaskLayout(
    VMICreateGroupMaskOp op, VMIMaskType resultVMIType,
    VMILayoutAttr *layout, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  *layout = resultVMIType.getLayoutAttr();
  if (!*layout) {
    return fail("dynamic create_group_mask requires assigned layout");
  }
  bool unsupportedLaneStride = layout->getLaneStride() != 1;
  if (unsupportedLaneStride) {
    return fail("dynamic create_group_mask requires lane_stride=1 layout");
  }
  bool unsupportedMaskGranularity = resultVMIType.getGranularity() != "b32";
  if (unsupportedMaskGranularity) {
    return fail("dynamic create_group_mask currently requires b32 granularity");
  }
  int64_t numGroups = op.getNumGroupsAttr().getInt();
  int64_t groupSize = op.getGroupSizeAttr().getInt();
  bool invalidLogicalShape =
      numGroups <= 0 || groupSize <= 0 ||
      resultVMIType.getElementCount() != numGroups * groupSize;
  if (invalidLogicalShape) {
    return fail("dynamic create_group_mask requires result lane count to match "
                "num_groups * group_size");
  }
  if (!getPowerOfTwoLog2(groupSize)) {
    return fail("dynamic create_group_mask currently requires power-of-two group_size");
  }
  return success();
}

static FailureOr<std::pair<int64_t, int64_t>>
getDynamicGroupMaskPhysicalShape(VMIMaskType resultVMIType,
                                 TypeRange resultTypes,
                                 std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<std::pair<int64_t, int64_t>> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<StringRef> physicalGranularity =
      getVMIMaskPhysicalGranularity(resultVMIType);
  FailureOr<int64_t> lanesPerPart =
      failed(physicalGranularity)
          ? FailureOr<int64_t>(failure())
          : getMaskLanesPerPart(*physicalGranularity);
  FailureOr<int64_t> arity = getVMIPhysicalArity(resultVMIType);
  bool missingPhysicalShape = failed(lanesPerPart) || failed(arity) || *arity < 1;
  if (missingPhysicalShape) {
    return fail("dynamic create_group_mask requires computable physical mask chunks");
  }
  bool resultArityMismatch = static_cast<int64_t>(resultTypes.size()) != *arity;
  if (resultArityMismatch) {
    return fail("dynamic create_group_mask physical result count mismatch");
  }
  return std::make_pair(*lanesPerPart, *arity);
}

static FailureOr<DynamicGroupMaskPlan> buildDynamicGroupMaskPlan(
    VMICreateGroupMaskOp op, VMIMaskType resultVMIType, TypeRange resultTypes,
    std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<DynamicGroupMaskPlan> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  VMILayoutAttr layout;
  if (failed(checkDynamicGroupMaskLayout(op, resultVMIType, &layout, reason))) {
    return failure();
  }
  FailureOr<std::pair<int64_t, int64_t>> physicalShape =
      getDynamicGroupMaskPhysicalShape(resultVMIType, resultTypes, reason);
  if (failed(physicalShape)) {
    return failure();
  }
  int64_t lanesPerPart = physicalShape->first;
  int64_t arity = physicalShape->second;
  int64_t factor = layout.isDenseSplit() ? layout.getFactor() : 1;
  FailureOr<int64_t> blockElems = getVMILayoutBlockElems(resultVMIType);
  bool invalidResultCount =
      factor <= 0 || failed(blockElems) || *blockElems <= 0 ||
      static_cast<int64_t>(resultTypes.size()) % factor != 0;
  if (invalidResultCount) {
    return fail("dynamic create_group_mask physical result count does not match layout factor");
  }
  if (!getPowerOfTwoLog2(*blockElems)) {
    return fail("dynamic create_group_mask requires a power-of-two physical block element count");
  }
  return DynamicGroupMaskPlan{layout, factor, *blockElems, lanesPerPart, arity};
}

static FailureOr<SmallVector<Value>> materializeDynamicGroupMaskChunks(
    VMICreateGroupMaskOp op, Value activeI32, VMIMaskType resultVMIType,
    TypeRange resultTypes, int64_t factor, int64_t blockElems,
    int64_t lanesPerPart, PatternRewriter &rewriter) {
  if (factor <= 0) {
    return rewriter.notifyMatchFailure(
        op, "dynamic group mask requires positive layout factor");
  }
  int64_t safeFactor = factor;
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  int64_t chunksPerPart = resultTypes.size() / safeFactor;
  for (int64_t part = 0; part < factor; ++part) {
    for (int64_t chunk = 0; chunk < chunksPerPart; ++chunk) {
      FailureOr<Value> predicate = materializeDynamicGroupMaskChunk(
          op, resultVMIType, resultTypes[part * chunksPerPart + chunk],
          activeI32, factor, blockElems, part, chunk, lanesPerPart, rewriter);
      if (failed(predicate)) {
        return failure();
      }
      results.push_back(*predicate);
    }
  }
  return results;
}

FailureOr<SmallVector<Value>> materializeDynamicGroupMaskForType(
    VMICreateGroupMaskOp op, Value activeElemsPerGroup,
    VMIMaskType resultVMIType, TypeRange resultTypes,
    PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  FailureOr<DynamicGroupMaskPlan> plan =
      buildDynamicGroupMaskPlan(op, resultVMIType, resultTypes, nullptr);
  if (failed(plan)) {
    return failure();
  }
  Value activeI32 =
      clampDynamicActiveLanes(loc, activeElemsPerGroup,
                              op.getGroupSizeAttr().getInt(), rewriter);

  return materializeDynamicGroupMaskChunks(
      op, activeI32, resultVMIType, resultTypes, plan->factor,
      plan->blockElems, plan->lanesPerPart, rewriter);
}

std::optional<int64_t> getPrefixActiveLaneCount(ArrayRef<int8_t> activeLanes) {
  bool seenInactive = false;
  int64_t activeCount = 0;
  for (int8_t active : activeLanes) {
    if (active) {
      if (seenInactive) {
        return std::nullopt;
      }
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
  if (pattern) {
    return createPatternMask(loc, maskType, *pattern, rewriter);
  }

  FailureOr<std::pair<Value, Value>> maskAndRemaining = createRuntimePrefixMask(
      loc, maskType, createI32Constant(loc, activeLanes, rewriter), rewriter);
  if (failed(maskAndRemaining)) {
    return failure();
  }
  return maskAndRemaining->first;
}

static FailureOr<int64_t> validateConstantMaskChunk(
    MaskType maskType, ArrayRef<int8_t> activeLanes) {
  FailureOr<int64_t> lanesPerPart =
      getMaskLanesPerPart(maskType.getGranularity());
  bool invalidShape = failed(lanesPerPart) ||
                      static_cast<int64_t>(activeLanes.size()) != *lanesPerPart;
  if (invalidShape) {
    return failure();
  }
  return *lanesPerPart;
}

static FailureOr<Value> materializeNonPrefixConstantMask(
    Location loc, MaskType maskType, ArrayRef<int8_t> activeLanes,
    int64_t lanesPerPart, Value allTrue, PatternRewriter &rewriter) {
  Value result;
  for (int64_t lane = 0; lane < lanesPerPart;) {
    while (lane < lanesPerPart && !activeLanes[lane]) {
      ++lane;
    }
    if (lane >= lanesPerPart) {
      break;
    }
    int64_t runBegin = lane;
    while (lane < lanesPerPart && activeLanes[lane]) {
      ++lane;
    }
    int64_t runEnd = lane;
    FailureOr<Value> prefixEnd =
        materializePrefixMask(loc, maskType, runEnd, lanesPerPart, rewriter);
    if (failed(prefixEnd)) {
      return failure();
    }
    Value runMask = *prefixEnd;
    if (runBegin != 0) {
      FailureOr<Value> prefixBegin = materializePrefixMask(
          loc, maskType, runBegin, lanesPerPart, rewriter);
      if (failed(prefixBegin)) {
        return failure();
      }
      Value notPrefixBegin =
          rewriter.create<PnotOp>(loc, maskType, *prefixBegin, allTrue)
              .getResult();
      runMask = rewriter
                    .create<PandOp>(loc, maskType, *prefixEnd, notPrefixBegin,
                                    allTrue)
                    .getResult();
    }
    if (!result) {
      result = runMask;
    } else {
      result = rewriter.create<PorOp>(loc, maskType, result, runMask, allTrue)
                   .getResult();
    }
  }
  return result;
}

FailureOr<Value> materializeConstantMaskChunk(Location loc, MaskType maskType,
                                              ArrayRef<int8_t> activeLanes,
                                              PatternRewriter &rewriter) {
  FailureOr<int64_t> lanesPerPart =
      validateConstantMaskChunk(maskType, activeLanes);
  if (failed(lanesPerPart)) {
    return failure();
  }

  if (std::optional<int64_t> prefixCount =
          getPrefixActiveLaneCount(activeLanes)) {
    return materializePrefixMask(loc, maskType, *prefixCount, *lanesPerPart,
                                 rewriter);
  }

  FailureOr<Value> allTrue = createAllTrueMask(loc, maskType, rewriter);
  if (failed(allTrue)) {
    return failure();
  }

  FailureOr<Value> result = materializeNonPrefixConstantMask(
      loc, maskType, activeLanes, *lanesPerPart, *allTrue, rewriter);
  if (failed(result)) {
    return failure();
  }

  if (*result) {
    return *result;
  }
  return materializePrefixMask(loc, maskType, 0, *lanesPerPart, rewriter);
}

FailureOr<Value> createScalarOffsetConstant(Location loc, Type type,
                                            int64_t value,
                                            PatternRewriter &rewriter);

Value createChunkOffset(Location loc, Value baseOffset, int64_t laneOffset,
                        PatternRewriter &rewriter) {
  if (laneOffset == 0) {
    return baseOffset;
  }
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
  auto fail = [&op, &rewriter](const Twine &message) {
    return rewriter.notifyMatchFailure(op, message);
  };

  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout || !layout.isContiguous()) {
    return fail("group op requires contiguous VMI layout");
  }
  if (failed(checkFullDataPhysicalChunks(type, nullptr))) {
    return fail("group op requires full physical chunks");
  }
  FailureOr<int64_t> lanes = getDataLanesPerPart(type.getElementType());
  if (failed(lanes)) {
    return fail("group op requires known physical lanes per part");
  }
  if (groupSize <= 0) {
    return fail("group op requires positive derived group size");
  }
  int64_t safeGroupSize = groupSize > 0 ? groupSize : 1;
  bool unevenLogicalGroups = type.getElementCount() % safeGroupSize != 0;
  if (unevenLogicalGroups) {
    return fail("group op requires derived group size to evenly divide lane "
                "count");
  }
  if (*lanes <= 0) {
    return fail("group op requires positive physical lanes per part");
  }
  if (groupSize % *lanes != 0) {
    return fail("group op currently requires group size to be a multiple of "
                "physical lanes per part");
  }

  *lanesPerPart = *lanes;
  *groupCount = type.getElementCount() / groupSize;
  *chunksPerGroup = groupSize / *lanes;
  return success();
}

FailureOr<Value> createZeroVector(Location loc, VRegType type,
                                  PatternRewriter &rewriter) {
  FailureOr<Value> zero =
      createScalarOffsetConstant(loc, type.getElementType(), 0, rewriter);
  FailureOr<Value> mask = createAllTrueMaskForVReg(loc, type, rewriter);
  bool failedMaterialization = failed(zero) || failed(mask);
  if (failedMaterialization) {
    return failure();
  }
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
  bool invalidRange = failed(lanesPerPart) || begin < 0 || begin > end ||
                      end > *lanesPerPart;
  if (invalidRange) {
    return failure();
  }
  SmallVector<int8_t> active(*lanesPerPart, 0);
  for (int64_t lane = begin; lane < end; ++lane) {
    active[lane] = 1;
  }
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
  bool failedSeedMaterialization =
      failed(baseScalar) || failed(maskType) || failed(allMask);
  if (failedSeedMaterialization) {
    return failure();
  }
  Value result = rewriter
                     .create<VdupOp>(loc, indexType, *baseScalar, *allMask,
                                     /*position=*/nullptr)
                     .getResult();
  if (groupSize >= lanesPerPart) {
    return result;
  }
  if (lanesPerPart % groupSize != 0) {
    return failure();
  }

  int64_t groupsPerChunk = lanesPerPart / groupSize;
  for (int64_t localGroup = 1; localGroup < groupsPerChunk; ++localGroup) {
    FailureOr<Value> groupScalar = createScalarOffsetConstant(
        loc, indexType.getElementType(),
        (baseGroupSlot + localGroup) * slotLaneStride, rewriter);
    FailureOr<Value> laneMask =
        createLaneRangeMask(loc, *maskType, localGroup * groupSize,
                            (localGroup + 1) * groupSize, rewriter);
    bool failedGroupMaterialization = failed(groupScalar) || failed(laneMask);
    if (failedGroupMaterialization) {
      return failure();
    }
    Value splat = rewriter
                      .create<VdupOp>(loc, indexType, *groupScalar, *allMask,
                                      /*position=*/nullptr)
                      .getResult();
    result = rewriter.create<VselOp>(loc, indexType, splat, result, *laneMask)
                 .getResult();
  }
  return result;
}

std::optional<std::string> getX2MemoryDistToken(Type elementType,
                                                StringRef prefix) {
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  if (elementBits != 8 && elementBits != 16 && elementBits != 32) {
    return std::nullopt;
  }
  return (Twine(prefix) + "_B" + Twine(elementBits)).str();
}

std::optional<std::string> getDenseLaneStrideLoadDistToken(VMIVRegType type) {
  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout || !layout.isContiguous()) {
    return std::nullopt;
  }
  unsigned elementBits = pto::getPTOStorageElemBitWidth(type.getElementType());
  if (layout.getLaneStride() == 2 &&
      (elementBits == 8 || elementBits == 16 || elementBits == 32)) {
    return (Twine("UNPK_B") + Twine(elementBits)).str();
  }
  bool isUnpack4 = layout.getLaneStride() == 4 && elementBits == 8;
  if (isUnpack4) {
    return std::string("UNPK4");
  }
  return std::nullopt;
}

std::optional<std::string>
getLaneStrideStoreDistToken(VMILayoutAttr layout, Type elementType) {
  if (!layout || !layout.hasLaneStride()) {
    return std::nullopt;
  }
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  bool isPackB16 = layout.getLaneStride() == 2 && elementBits == 8;
  if (isPackB16) {
    return std::string("PK_B16");
  }
  bool isPackB32 = layout.getLaneStride() == 2 && elementBits == 16;
  if (isPackB32) {
    return std::string("PK_B32");
  }
  bool isPackB64 = layout.getLaneStride() == 2 && elementBits == 32;
  if (isPackB64) {
    return std::string("PK_B64");
  }
  bool isPack4B32 = layout.getLaneStride() == 4 && elementBits == 8;
  if (isPack4B32) {
    return std::string("PK4_B32");
  }
  return std::nullopt;
}

std::optional<std::string> getDenseLaneStrideStoreDistToken(VMIVRegType type) {
  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout || !layout.isContiguous()) {
    return std::nullopt;
  }
  return getLaneStrideStoreDistToken(layout, type.getElementType());
}

std::optional<StringRef>
getLaneStrideStoreMaskGranularity(VMILayoutAttr layout, Type elementType) {
  if (!layout || !layout.hasLaneStride()) {
    return std::nullopt;
  }
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  bool isB16Mask = layout.getLaneStride() == 2 && elementBits == 8;
  if (isB16Mask) {
    return StringRef("b16");
  }
  if (layout.getLaneStride() == 2 &&
      (elementBits == 16 || elementBits == 32)) {
    return StringRef("b32");
  }
  bool isB32MaskForPack4 = layout.getLaneStride() == 4 && elementBits == 8;
  if (isB32MaskForPack4) {
    return StringRef("b32");
  }
  return std::nullopt;
}

std::optional<StringRef>
getDenseLaneStrideStoreMaskGranularity(VMIVRegType type) {
  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout || !layout.isContiguous()) {
    return std::nullopt;
  }
  return getLaneStrideStoreMaskGranularity(layout, type.getElementType());
}

std::optional<StringRef>
getDenseLaneStrideMaskedStoreMaskGranularity(VMIVRegType type) {
  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout || !layout.isContiguous()) {
    return std::nullopt;
  }
  unsigned elementBits = pto::getPTOStorageElemBitWidth(type.getElementType());
  bool isB16MaskedStore = layout.getLaneStride() == 2 && elementBits == 8;
  if (isB16MaskedStore) {
    return StringRef("b16");
  }
  bool isB32MaskedStore = layout.getLaneStride() == 2 && elementBits == 16;
  if (isB32MaskedStore) {
    return StringRef("b32");
  }
  bool isB32MaskedStorePack4 =
      layout.getLaneStride() == 4 && elementBits == 8;
  if (isB32MaskedStorePack4) {
    return StringRef("b32");
  }
  return std::nullopt;
}

std::optional<std::string> getPointStoreDistToken(Type elementType) {
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  if (elementBits != 8 && elementBits != 16 && elementBits != 32) {
    return std::nullopt;
  }
  return (Twine("1PT_B") + Twine(elementBits)).str();
}

std::optional<std::string> getScalarBroadcastLoadDistToken(Type elementType) {
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  if (elementBits != 8 && elementBits != 16 && elementBits != 32) {
    return std::nullopt;
  }
  return (Twine("BRC_B") + Twine(elementBits)).str();
}

struct VPTOCmpMode {
  StringRef mode;
  std::optional<IntegerType::SignednessSemantics> signedness;
};

std::optional<VPTOCmpMode> getVPTOCmpFMode(StringRef predicate) {
  if (predicate == "eq" || predicate == "ne" || predicate == "lt" ||
      predicate == "le" || predicate == "gt" || predicate == "ge") {
    return VPTOCmpMode{predicate, std::nullopt};
  }
  if (predicate == "oeq") {
    return VPTOCmpMode{StringRef("eq"), std::nullopt};
  }
  if (predicate == "one") {
    return VPTOCmpMode{StringRef("ne"), std::nullopt};
  }
  if (predicate == "olt") {
    return VPTOCmpMode{StringRef("lt"), std::nullopt};
  }
  if (predicate == "ole") {
    return VPTOCmpMode{StringRef("le"), std::nullopt};
  }
  if (predicate == "ogt") {
    return VPTOCmpMode{StringRef("gt"), std::nullopt};
  }
  if (predicate == "oge") {
    return VPTOCmpMode{StringRef("ge"), std::nullopt};
  }
  return std::nullopt;
}

std::optional<VPTOCmpMode> getVPTOCmpIMode(StringRef predicate) {
  if (predicate == "eq" || predicate == "ne") {
    return VPTOCmpMode{predicate, std::nullopt};
  }
  if (predicate == "ult") {
    return VPTOCmpMode{
        StringRef("lt"), IntegerType::SignednessSemantics::Unsigned};
  }
  if (predicate == "ule") {
    return VPTOCmpMode{
        StringRef("le"), IntegerType::SignednessSemantics::Unsigned};
  }
  if (predicate == "ugt") {
    return VPTOCmpMode{
        StringRef("gt"), IntegerType::SignednessSemantics::Unsigned};
  }
  if (predicate == "uge") {
    return VPTOCmpMode{
        StringRef("ge"), IntegerType::SignednessSemantics::Unsigned};
  }
  if (predicate == "slt") {
    return VPTOCmpMode{
        StringRef("lt"), IntegerType::SignednessSemantics::Signed};
  }
  if (predicate == "sle") {
    return VPTOCmpMode{
        StringRef("le"), IntegerType::SignednessSemantics::Signed};
  }
  if (predicate == "sgt") {
    return VPTOCmpMode{
        StringRef("gt"), IntegerType::SignednessSemantics::Signed};
  }
  if (predicate == "sge") {
    return VPTOCmpMode{
        StringRef("ge"), IntegerType::SignednessSemantics::Signed};
  }
  return std::nullopt;
}

template <typename SourceOp>
std::optional<VPTOCmpMode> getVPTOCmpMode(StringRef predicate) {
  if constexpr (std::is_same_v<SourceOp, VMICmpIOp>) {
    return getVPTOCmpIMode(predicate);
  } else {
    return getVPTOCmpFMode(predicate);
  }
}

template <typename SourceOp>
StringRef getSupportedComparePredicateMessage() {
  if constexpr (std::is_same_v<SourceOp, VMICmpIOp>) {
    return "eq/ne, unsigned integer forms ult/ule/ugt/uge, and signed "
           "integer forms slt/sle/sgt/sge";
  } else {
    return "eq/ne/lt/le/gt/ge and ordered FP forms oeq/one/olt/ole/ogt/oge";
  }
}

template <typename SourceOp>
LogicalResult checkSupportedComparePredicate(Operation *op,
                                             StringRef predicate) {
  if (getVPTOCmpMode<SourceOp>(predicate)) {
    return success();
  }
  return op->emitError()
         << kVMIDiagUnsupportedPrefix << "compare predicate " << predicate
         << " cannot be lowered to pto.vcmp; supported predicates are "
         << getSupportedComparePredicateMessage<SourceOp>();
}

struct OneToNVMIUnpackOpPattern : OneToNOpConversionPattern<VMIUnpackOp> {
  using OneToNOpConversionPattern<VMIUnpackOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIUnpackOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    bool arityMismatch = sourceParts.size() != op->getNumResults();
    if (arityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "converted source part count must match unpack results");
    }
    replaceOpWithFlatConvertedValues(rewriter, op, sourceParts,
                                     *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIPackOpPattern : OneToNOpConversionPattern<VMIPackOp> {
  using OneToNOpConversionPattern<VMIPackOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIPackOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<int64_t> arity = getVMIPhysicalArity(op.getResult().getType());
    SmallVector<Value> flatOperands = flattenOneToNOperands(adaptor.getOperands());
    bool arityMismatch =
        failed(arity) || static_cast<int64_t>(flatOperands.size()) != *arity;
    if (arityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "pack part count must match converted VMI result arity");
    }
    replaceOpWithFlatConvertedValues(rewriter, op, flatOperands,
                                     *this->getTypeConverter());
    return success();
  }
};

LogicalResult verifyIdentityPartForwarding(Operation *op,
                                           ValueRange sourceParts,
                                           TypeRange resultTypes,
                                           PatternRewriter &rewriter) {
  bool arityMismatch = sourceParts.size() != resultTypes.size();
  if (arityMismatch) {
    return rewriter.notifyMatchFailure(
        op, "source and result physical arity mismatch");
  }
  for (auto [part, resultType] : llvm::zip_equal(sourceParts, resultTypes)) {
    bool typeMismatch = part.getType() != resultType;
    if (typeMismatch) {
      return rewriter.notifyMatchFailure(
          op, "helper requires non-identity physical materialization");
    }
  }
  return success();
}

FailureOr<VRegType> getUnsignedCarrierVRegType(MLIRContext *ctx,
                                               unsigned elementBits) {
  if (elementBits != 8 && elementBits != 16 && elementBits != 32) {
    return failure();
  }
  auto elementType = IntegerType::get(
      ctx, elementBits, IntegerType::SignednessSemantics::Unsigned);
  return VRegType::get(ctx, 2048 / elementBits, elementType);
}

FailureOr<VRegType>
getSignednessCarrierVRegType(VRegType inputType,
                             IntegerType::SignednessSemantics signedness) {
  auto inputElementType = dyn_cast<IntegerType>(inputType.getElementType());
  if (!inputElementType) {
    return failure();
  }
  if ((signedness == IntegerType::SignednessSemantics::Signed &&
       !inputElementType.isUnsigned()) ||
      (signedness == IntegerType::SignednessSemantics::Unsigned &&
       inputElementType.isUnsigned())) {
    return inputType;
  }
  auto carrierElementType = IntegerType::get(
      inputType.getContext(), inputElementType.getWidth(), signedness);
  return VRegType::get(inputType.getContext(), inputType.getElementCount(),
                       carrierElementType);
}

FailureOr<Value> bitcastVReg(Location loc, Value value, Type resultType,
                             PatternRewriter &rewriter) {
  bool isIdentity = value.getType() == resultType;
  if (isIdentity) {
    return value;
  }
  auto inputType = dyn_cast<VRegType>(value.getType());
  auto outputType = dyn_cast<VRegType>(resultType);
  if (!inputType || !outputType) {
    return failure();
  }
  return rewriter.create<VbitcastOp>(loc, outputType, value).getResult();
}

FailureOr<VRegType> getVcaddResultType(VRegType inputType) {
  auto inputIntegerType = dyn_cast<IntegerType>(inputType.getElementType());
  bool preservesType =
      !inputIntegerType || inputIntegerType.getWidth() == 32;
  if (preservesType) {
    return inputType;
  }
  unsigned inputWidth = inputIntegerType.getWidth();
  if (inputWidth != 8 && inputWidth != 16) {
    return failure();
  }
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
  if (failed(resultType)) {
    return failure();
  }
  Value part = rewriter.create<arith::ConstantIndexOp>(loc, partIndex);
  return rewriter.create<VzunpackOp>(loc, *resultType, source, part)
      .getResult();
}

FailureOr<Value> packToPreviousCarrier(Location loc, Value source,
                                       unsigned resultBits,
                                       StringRef part,
                                       PatternRewriter &rewriter) {
  FailureOr<VRegType> resultType =
      getUnsignedCarrierVRegType(rewriter.getContext(), resultBits);
  if (failed(resultType)) {
    return failure();
  }
  return rewriter
      .create<VpackOp>(loc, *resultType, source,
                       rewriter.getStringAttr(part))
      .getResult();
}

static FailureOr<unsigned> validateDenseLaneStrideShape(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    Type elementType, int64_t laneStride, bool unpack,
    PatternRewriter &rewriter) {
  if (laneStride <= 0) {
    return rewriter.notifyMatchFailure(
        op, "dense lane_stride materialization requires positive lane stride");
  }
  int64_t safeLaneStride = laneStride;
  bool emptyParts = sourceParts.empty() || resultTypes.empty();
  bool arityMismatch = unpack
                           ? (resultTypes.size() + safeLaneStride - 1) /
                                     safeLaneStride !=
                                 sourceParts.size()
                           : (sourceParts.size() + safeLaneStride - 1) /
                                     safeLaneStride !=
                                 resultTypes.size();
  if (emptyParts || arityMismatch) {
    StringRef direction = unpack ? "unpack" : "pack";
    return rewriter.notifyMatchFailure(
        op, Twine("dense lane_stride ") + direction +
                " materialization requires one partial or complete stride "
                "group per physical part");
  }
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  bool unsupportedShape =
      (laneStride != 2 && laneStride != 4) ||
      (laneStride == 4 && elementBits != 8) ||
      (elementBits != 8 && elementBits != 16);
  if (unsupportedShape) {
    StringRef direction = unpack ? "unpack" : "pack";
    return rewriter.notifyMatchFailure(
        op, Twine("unsupported dense lane_stride ") + direction +
                " carrier shape");
  }
  return elementBits;
}

static FailureOr<Value> materializeContiguousLaneStridePart(
    Operation *op, Value source, Type resultType, unsigned elementBits,
    VRegType inputCarrier, int64_t laneStride, int64_t part,
    PatternRewriter &rewriter) {
  FailureOr<Value> current =
      bitcastVReg(op->getLoc(), source, inputCarrier, rewriter);
  if (failed(current)) {
    return failure();
  }
  FailureOr<Value> unpacked = unpackToNextCarrier(
      op->getLoc(), *current, elementBits,
      laneStride == 4 ? part / 2 : part, rewriter);
  if (failed(unpacked)) {
    return failure();
  }
  current = *unpacked;
  if (laneStride == 4) {
    unpacked = unpackToNextCarrier(op->getLoc(), *current, elementBits * 2,
                                   part % 2, rewriter);
    if (failed(unpacked)) {
      return failure();
    }
    current = *unpacked;
  }
  return bitcastVReg(op->getLoc(), *current, resultType, rewriter);
}

FailureOr<SmallVector<Value>> materializeContiguousToLaneStride(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    Type elementType, int64_t laneStride, PatternRewriter &rewriter) {
  FailureOr<unsigned> elementBits = validateDenseLaneStrideShape(
      op, sourceParts, resultTypes, elementType, laneStride, true, rewriter);
  if (failed(elementBits)) {
    return failure();
  }

  MLIRContext *ctx = rewriter.getContext();
  FailureOr<VRegType> inputCarrier =
      getUnsignedCarrierVRegType(ctx, *elementBits);
  if (failed(inputCarrier)) {
    return failure();
  }

  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (auto [resultIndex, resultType] : llvm::enumerate(resultTypes)) {
    int64_t sourceIndex = resultIndex / laneStride;
    if (sourceIndex >= static_cast<int64_t>(sourceParts.size())) {
      return failure();
    }
    int64_t part = resultIndex % laneStride;
    FailureOr<Value> result = materializeContiguousLaneStridePart(
        op, sourceParts[sourceIndex], resultType, *elementBits, *inputCarrier,
        laneStride, part, rewriter);
    if (failed(result)) {
      return failure();
    }
    results.push_back(*result);
  }
  return results;
}

static FailureOr<Value> mergeLaneStrideCarrierPair(
    Operation *op, Value lowCarrier, Value highCarrier, unsigned carrierBits,
    PatternRewriter &rewriter) {
  FailureOr<Value> low = packToPreviousCarrier(
      op->getLoc(), lowCarrier, carrierBits / 2, "LOWER", rewriter);
  if (failed(low)) {
    return failure();
  }
  FailureOr<Value> high = packToPreviousCarrier(
      op->getLoc(), highCarrier, carrierBits / 2, "HIGHER", rewriter);
  if (failed(high)) {
    return failure();
  }
  FailureOr<Value> mask = createAllTrueMaskForVReg(
      op->getLoc(), cast<VRegType>((*low).getType()), rewriter);
  if (failed(mask)) {
    return failure();
  }
  return rewriter.create<VorOp>(op->getLoc(), (*low).getType(), *low, *high,
                                *mask)
      .getResult();
}

static FailureOr<Value> materializeLaneStrideResultPart(
    Operation *op, ValueRange sourceParts, Type resultType, size_t sourceBegin,
    size_t sourceEnd, unsigned elementBits, unsigned carrierBits,
    VRegType sourceCarrier, PatternRewriter &rewriter) {
  SmallVector<Value> currentLevel;
  currentLevel.reserve(sourceEnd - sourceBegin);
  for (Value source : sourceParts.slice(sourceBegin, sourceEnd - sourceBegin)) {
    FailureOr<Value> carrier =
        bitcastVReg(op->getLoc(), source, sourceCarrier, rewriter);
    if (failed(carrier)) {
      return failure();
    }
    currentLevel.push_back(*carrier);
  }

  unsigned currentBits = carrierBits;
  while (currentBits > elementBits) {
    SmallVector<Value> nextLevel;
    nextLevel.reserve((currentLevel.size() + 1) / 2);
    for (size_t index = 0; index < currentLevel.size(); index += 2) {
      Value merged;
      if (index + 1 < currentLevel.size()) {
        FailureOr<Value> pair = mergeLaneStrideCarrierPair(
            op, currentLevel[index], currentLevel[index + 1], currentBits,
            rewriter);
        if (failed(pair)) {
          return failure();
        }
        merged = *pair;
      } else {
        FailureOr<Value> low = packToPreviousCarrier(
            op->getLoc(), currentLevel[index], currentBits / 2, "LOWER",
            rewriter);
        if (failed(low)) {
          return failure();
        }
        merged = *low;
      }
      nextLevel.push_back(merged);
    }
    currentLevel = std::move(nextLevel);
    currentBits /= 2;
  }
  bool invalidResultArity = currentLevel.size() != 1;
  if (invalidResultArity) {
    return failure();
  }
  return bitcastVReg(op->getLoc(), currentLevel.front(), resultType, rewriter);
}

static FailureOr<Value> materializeGroupSlotLaneStridePart(
    Operation *op, Value source, Type resultType, Type elementType,
    int64_t sourceStride, int64_t resultStride,
    PatternRewriter &rewriter) {
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  unsigned carrierBits = elementBits * sourceStride;
  FailureOr<VRegType> carrierType =
      getUnsignedCarrierVRegType(rewriter.getContext(), carrierBits);
  if (failed(carrierType)) {
    return failure();
  }
  FailureOr<Value> current =
      bitcastVReg(op->getLoc(), source, *carrierType, rewriter);
  if (failed(current)) {
    return failure();
  }

  int64_t currentStride = sourceStride;
  while (currentStride < resultStride) {
    FailureOr<Value> unpacked = unpackToNextCarrier(
        op->getLoc(), *current, carrierBits, /*partIndex=*/0, rewriter);
    if (failed(unpacked)) {
      return failure();
    }
    current = *unpacked;
    currentStride *= 2;
    carrierBits *= 2;
  }
  while (currentStride > resultStride) {
    FailureOr<Value> packed = packToPreviousCarrier(
        op->getLoc(), *current, carrierBits / 2, "LOWER", rewriter);
    if (failed(packed)) {
      return failure();
    }
    current = *packed;
    currentStride /= 2;
    carrierBits /= 2;
  }
  return bitcastVReg(op->getLoc(), *current, resultType, rewriter);
}

static FailureOr<SmallVector<Value>> materializeLaneStrideResultList(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    unsigned elementBits, unsigned carrierBits, VRegType sourceCarrier,
    int64_t laneStride, PatternRewriter &rewriter) {
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (auto [resultIndex, resultType] : llvm::enumerate(resultTypes)) {
    size_t sourceBegin = resultIndex * laneStride;
    size_t sourceEnd =
        std::min<size_t>(sourceBegin + laneStride, sourceParts.size());
    FailureOr<Value> result = materializeLaneStrideResultPart(
        op, sourceParts, resultType, sourceBegin, sourceEnd, elementBits,
        carrierBits, sourceCarrier, rewriter);
    if (failed(result)) {
      return failure();
    }
    results.push_back(*result);
  }
  return results;
}

FailureOr<SmallVector<Value>> materializeLaneStrideToContiguous(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    Type elementType, int64_t laneStride, PatternRewriter &rewriter) {
  FailureOr<unsigned> elementBits = validateDenseLaneStrideShape(
      op, sourceParts, resultTypes, elementType, laneStride, false, rewriter);
  if (failed(elementBits)) {
    return failure();
  }

  unsigned carrierBits =
      static_cast<unsigned>(*elementBits * static_cast<unsigned>(laneStride));
  FailureOr<VRegType> sourceCarrier =
      getUnsignedCarrierVRegType(rewriter.getContext(), carrierBits);
  if (failed(sourceCarrier)) {
    return failure();
  }

  return materializeLaneStrideResultList(
      op, sourceParts, resultTypes, *elementBits, carrierBits, *sourceCarrier,
      laneStride, rewriter);
}

static LogicalResult checkGroupSlotLaneStrideContract(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    Type elementType, int64_t sourceStride, int64_t resultStride,
    PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message) {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  bool invalidArity = sourceParts.size() != resultTypes.size() ||
                     sourceParts.empty();
  if (invalidArity) {
    return fail("group-slot lane_stride materialization requires matching "
                "non-empty source/result physical arity");
  }
  bool unsupportedStride =
      (sourceStride != 1 && sourceStride != 2 && sourceStride != 4) ||
      (resultStride != 1 && resultStride != 2 && resultStride != 4);
  if (unsupportedStride) {
    return fail("unsupported group-slot lane_stride factor");
  }
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  int64_t maxStride = std::max(sourceStride, resultStride);
  bool unsupportedCarrier =
      (elementBits != 8 && elementBits != 16) || elementBits * maxStride > 32;
  if (unsupportedCarrier) {
    return fail("unsupported group-slot lane_stride carrier shape");
  }
  return success();
}

FailureOr<SmallVector<Value>> materializeGroupSlotLaneStride(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    Type elementType, int64_t sourceStride, int64_t resultStride,
    PatternRewriter &rewriter) {
  if (failed(checkGroupSlotLaneStrideContract(
          op, sourceParts, resultTypes, elementType, sourceStride, resultStride,
          rewriter))) {
    return failure();
  }

  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (auto [source, resultType] :
       llvm::zip_equal(sourceParts, resultTypes)) {
    FailureOr<Value> result = materializeGroupSlotLaneStridePart(
        op, source, resultType, elementType, sourceStride, resultStride,
        rewriter);
    if (failed(result)) {
      return rewriter.notifyMatchFailure(
          op, "failed to bitcast group-slot result carrier");
    }
    results.push_back(*result);
  }
  return results;
}

static FailureOr<std::optional<SmallVector<Value>>> forwardIdentityLayoutParts(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    PatternRewriter &rewriter) {
  if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                          rewriter))) {
    return failure();
  }
  return std::optional<SmallVector<Value>>(
      SmallVector<Value>(sourceParts.begin(), sourceParts.end()));
}

static std::optional<SmallVector<Value>>
forwardBlockLayoutCastInputs(ValueRange sourceParts, TypeRange resultTypes) {
  bool invalidSourceArity = sourceParts.size() != 1;
  if (invalidSourceArity) {
    return std::nullopt;
  }
  auto cast = sourceParts.front().getDefiningOp<UnrealizedConversionCastOp>();
  bool invalidCast = !cast || cast.getInputs().size() != resultTypes.size();
  if (invalidCast) {
    return std::nullopt;
  }
  for (auto [input, resultType] : llvm::zip_equal(cast.getInputs(), resultTypes)) {
    bool typeMismatch = input.getType() != resultType;
    if (typeMismatch) {
      return std::nullopt;
    }
  }
  return SmallVector<Value>(cast.getInputs().begin(), cast.getInputs().end());
}

static FailureOr<std::optional<SmallVector<Value>>>
materializeGroupSlotLaneStrideLayout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    Type sourceVMIElementType, PatternRewriter &rewriter) {
  bool supported =
      sourceLayout.isGroupSlots() && resultLayout.isGroupSlots() &&
      sourceLayout.getNumGroups() == resultLayout.getNumGroups() &&
      sourceLayout.getSlots() == 8 && resultLayout.getSlots() == 8;
  if (!supported) {
    return std::optional<SmallVector<Value>>{};
  }
  FailureOr<SmallVector<Value>> result = materializeGroupSlotLaneStride(
      op, sourceParts, resultTypes, sourceVMIElementType,
      sourceLayout.getLaneStride(), resultLayout.getLaneStride(), rewriter);
  if (failed(result)) {
    return failure();
  }
  return std::optional<SmallVector<Value>>(std::move(*result));
}

static FailureOr<std::optional<SmallVector<Value>>>
materializeBlockLayoutForwarding(Operation *op, ValueRange sourceParts,
                                 TypeRange resultTypes,
                                 VMILayoutAttr sourceLayout,
                                 VMILayoutAttr resultLayout,
                                 PatternRewriter &rewriter) {
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
  if (!contiguousToBlock && !blockToContiguous) {
    return std::optional<SmallVector<Value>>{};
  }
  if (std::optional<SmallVector<Value>> castInputs =
          forwardBlockLayoutCastInputs(sourceParts, resultTypes)) {
    return std::optional<SmallVector<Value>>(std::move(*castInputs));
  }
  return forwardIdentityLayoutParts(op, sourceParts, resultTypes, rewriter);
}

static FailureOr<std::optional<SmallVector<Value>>>
materializeSimpleDataLayoutConversion(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    Type sourceVMIElementType, PatternRewriter &rewriter) {
  if (!sourceLayout || !resultLayout) {
    (void)rewriter.notifyMatchFailure(
        op, "layout materialization requires assigned source/result layouts");
    return failure();
  }

  if (sourceLayout == resultLayout) {
    return forwardIdentityLayoutParts(op, sourceParts, resultTypes, rewriter);
  }

  bool oneLaneContiguousToGroup =
      sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 &&
      resultLayout.isGroupSlots() && resultLayout.getNumGroups() == 1 &&
      resultLayout.getSlots() == 1;
  bool oneLaneGroupToContiguous =
      sourceLayout.isGroupSlots() && sourceLayout.getNumGroups() == 1 &&
      sourceLayout.getSlots() == 1 && resultLayout.isContiguous() &&
      resultLayout.getLaneStride() == 1;
  if (oneLaneContiguousToGroup || oneLaneGroupToContiguous) {
    return forwardIdentityLayoutParts(op, sourceParts, resultTypes, rewriter);
  }

  FailureOr<std::optional<SmallVector<Value>>> groupSlot =
      materializeGroupSlotLaneStrideLayout(
          op, sourceParts, resultTypes, sourceLayout, resultLayout,
          sourceVMIElementType, rewriter);
  if (failed(groupSlot)) {
    return failure();
  }
  if (groupSlot->has_value()) {
    return std::optional<SmallVector<Value>>(std::move(**groupSlot));
  }

  FailureOr<std::optional<SmallVector<Value>>> block =
      materializeBlockLayoutForwarding(op, sourceParts, resultTypes,
                                       sourceLayout, resultLayout, rewriter);
  if (failed(block)) {
    return failure();
  }
  if (block->has_value()) {
    return std::optional<SmallVector<Value>>(std::move(**block));
  }

  return std::optional<SmallVector<Value>>{};
}

static FailureOr<SmallVector<Value>> materializeDeinterleaved2ToContiguous(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    PatternRewriter &rewriter) {
  bool invalidSource = sourceParts.empty() || sourceParts.size() % 2 != 0 ||
                       resultTypes.empty();
  if (invalidSource) {
    return rewriter.notifyMatchFailure(
        op, "deinterleaved=2 to contiguous materialization requires 2*N "
            "source parts and at least one result part");
  }
  int64_t groups = sourceParts.size() / 2;
  bool resultExceedsSource =
      resultTypes.size() > static_cast<size_t>(2 * groups);
  if (resultExceedsSource) {
    return rewriter.notifyMatchFailure(
        op, "deinterleaved=2 to contiguous materialization result arity "
            "exceeds source footprint");
  }
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (int64_t i = 0; i < groups && results.size() < resultTypes.size(); ++i) {
    Value lhs = sourceParts[i];
    Value rhs = sourceParts[groups + i];
    Type lhsType = lhs.getType();
    if (lhsType != rhs.getType()) {
      return rewriter.notifyMatchFailure(
          op, "vintlv requires matching source part types");
    }
    Type lowType = resultTypes[results.size()];
    bool hasHighResult = results.size() + 1 < resultTypes.size();
    Type highType = hasHighResult ? resultTypes[results.size() + 1] : lowType;
    if (lhsType != lowType || lhsType != highType) {
      return rewriter.notifyMatchFailure(
          op, "vintlv requires operands and results to share one type");
    }
    auto materialize = rewriter.create<VintlvOp>(
        op->getLoc(), lowType, highType, lhs, rhs);
    results.push_back(materialize.getLow());
    if (hasHighResult) {
      results.push_back(materialize.getHigh());
    }
  }
  return results;
}

static LogicalResult validateContiguousToDeinterleaved2Shape(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    PatternRewriter &rewriter, int64_t &groups) {
  bool invalidResult = sourceParts.empty() || resultTypes.empty() ||
                       resultTypes.size() % 2 != 0;
  if (invalidResult) {
    return rewriter.notifyMatchFailure(
        op, "contiguous to deinterleaved=2 materialization requires at least "
            "one source part and 2*N result parts");
  }
  groups = resultTypes.size() / 2;
  bool sourceExceedsResult =
      sourceParts.size() > static_cast<size_t>(2 * groups);
  if (sourceExceedsResult) {
    return rewriter.notifyMatchFailure(
        op, "contiguous to deinterleaved=2 materialization source footprint "
            "exceeds result arity");
  }
  return success();
}

static FailureOr<std::pair<Value, Value>> materializeContiguousToDeinterleaved2Group(
    Operation *op, Value lhs, Value rhs, Type lowType, Type highType,
    PatternRewriter &rewriter) {
  bool mismatchedTypes = lhs.getType() != rhs.getType() ||
                         lhs.getType() != lowType || lhs.getType() != highType;
  if (mismatchedTypes) {
    return rewriter.notifyMatchFailure(
        op, "vdintlv requires operands and results to share one type");
  }
  auto materialize = rewriter.create<VdintlvOp>(op->getLoc(), lowType, highType,
                                                 lhs, rhs);
  return std::make_pair(materialize.getLow(), materialize.getHigh());
}

static FailureOr<SmallVector<Value>> materializeContiguousToDeinterleaved2(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    PatternRewriter &rewriter) {
  int64_t groups = 0;
  if (failed(validateContiguousToDeinterleaved2Shape(
          op, sourceParts, resultTypes, rewriter, groups))) {
    return failure();
  }
  SmallVector<Value> part0;
  SmallVector<Value> part1;
  part0.reserve(groups);
  part1.reserve(groups);
  for (int64_t i = 0; i < groups; ++i) {
    size_t lhsIndex = 2 * i;
    if (lhsIndex >= sourceParts.size()) {
      return rewriter.notifyMatchFailure(
          op, "contiguous to deinterleaved=2 materialization missing source "
              "part");
    }
    size_t rhsIndex = lhsIndex + 1 < sourceParts.size() ? lhsIndex + 1
                                                          : lhsIndex;
    Value lhs = sourceParts[lhsIndex];
    Value rhs = sourceParts[rhsIndex];
    FailureOr<std::pair<Value, Value>> materialize =
        materializeContiguousToDeinterleaved2Group(
            op, lhs, rhs, resultTypes[i], resultTypes[groups + i], rewriter);
    if (failed(materialize)) {
      return failure();
    }
    part0.push_back(materialize->first);
    part1.push_back(materialize->second);
  }
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  results.append(part0);
  results.append(part1);
  return results;
}

FailureOr<std::optional<SmallVector<Value>>> materializeDeinterleaved2Layout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    PatternRewriter &rewriter) {
  auto isElementDeinterleaved = [](VMILayoutAttr layout) {
    return layout.isDeinterleaved() && layout.getFactor() == 2 &&
           layout.getLaneStride() == 1;
  };
  bool toContiguous = sourceLayout && sourceLayout.isDeinterleaved() &&
                      isElementDeinterleaved(sourceLayout) && resultLayout &&
                      resultLayout.isContiguous() &&
                      resultLayout.getLaneStride() == 1;
  bool fromContiguous = sourceLayout && sourceLayout.isContiguous() &&
                        sourceLayout.getLaneStride() == 1 && resultLayout &&
                        resultLayout.isDeinterleaved() &&
                        isElementDeinterleaved(resultLayout);
  if (!toContiguous && !fromContiguous) {
    return std::optional<SmallVector<Value>>{};
  }

  if (toContiguous) {
    FailureOr<SmallVector<Value>> results = materializeDeinterleaved2ToContiguous(
        op, sourceParts, resultTypes, rewriter);
    if (failed(results)) {
      return failure();
    }
    return std::optional<SmallVector<Value>>(std::move(*results));
  } else {
    FailureOr<SmallVector<Value>> results =
        materializeContiguousToDeinterleaved2(op, sourceParts, resultTypes,
                                              rewriter);
    if (failed(results)) {
      return failure();
    }
    return std::optional<SmallVector<Value>>(std::move(*results));
  }
}

FailureOr<std::optional<SmallVector<Value>>> materializeDataLaneStrideConversion(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    Type sourceVMIElementType, PatternRewriter &rewriter) {
  if (!sourceLayout || !resultLayout) {
    return std::optional<SmallVector<Value>>{};
  }
  bool contiguousToLaneStride =
      sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 &&
      resultLayout.isContiguous() && resultLayout.getLaneStride() != 1;
  if (contiguousToLaneStride) {
    FailureOr<SmallVector<Value>> result = materializeContiguousToLaneStride(
        op, sourceParts, resultTypes, sourceVMIElementType,
        resultLayout.getLaneStride(), rewriter);
    if (failed(result)) {
      return failure();
    }
    return std::optional<SmallVector<Value>>(std::move(*result));
  }
  bool laneStrideToContiguous =
      sourceLayout.isContiguous() && sourceLayout.getLaneStride() != 1 &&
      resultLayout.isContiguous() && resultLayout.getLaneStride() == 1;
  if (laneStrideToContiguous) {
    FailureOr<SmallVector<Value>> result = materializeLaneStrideToContiguous(
        op, sourceParts, resultTypes, sourceVMIElementType,
        sourceLayout.getLaneStride(), rewriter);
    if (failed(result)) {
      return failure();
    }
    return std::optional<SmallVector<Value>>(std::move(*result));
  }
  return std::optional<SmallVector<Value>>{};
}

FailureOr<SmallVector<Value>> materializeDataLayoutConversion(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    Type sourceVMIElementType, PatternRewriter &rewriter);

struct DataLayoutIntermediatePlan {
  enum class Kind { Contiguous, Deinterleaved };
  Kind kind;
  size_t intermediateCount;
};

static bool isContiguousLaneStrideLayout(VMILayoutAttr layout,
                                         int64_t laneStride) {
  return layout.isContiguous() && layout.getLaneStride() == laneStride;
}

static bool isDeinterleavedUnitStrideLayout(VMILayoutAttr layout,
                                            int64_t factor) {
  return layout.isDeinterleaved() && layout.getFactor() == factor &&
         layout.getLaneStride() == 1;
}

static bool isSupportedDeinterleavedIntermediateLayout(VMILayoutAttr layout) {
  return layout.isDeinterleaved() && layout.getLaneStride() == 1 &&
         (layout.getFactor() == 2 || layout.getFactor() == 4);
}

static std::optional<DataLayoutIntermediatePlan>
getDataLayoutIntermediatePlan(VMILayoutAttr sourceLayout,
                              VMILayoutAttr resultLayout,
                              size_t sourcePartCount) {
  bool deint2ToLaneStride =
      isDeinterleavedUnitStrideLayout(sourceLayout, 2) &&
      isContiguousLaneStrideLayout(resultLayout, 2);
  bool laneStrideToDeint2 =
      isContiguousLaneStrideLayout(sourceLayout, 2) &&
      isDeinterleavedUnitStrideLayout(resultLayout, 2);
  bool laneStride2ToLaneStride4 =
      isContiguousLaneStrideLayout(sourceLayout, 2) &&
      isContiguousLaneStrideLayout(resultLayout, 4);
  bool laneStride4ToLaneStride2 =
      isContiguousLaneStrideLayout(sourceLayout, 4) &&
      isContiguousLaneStrideLayout(resultLayout, 2);
  bool useContiguousIntermediate =
      deint2ToLaneStride || laneStrideToDeint2 || laneStride2ToLaneStride4 ||
      laneStride4ToLaneStride2;
  if (useContiguousIntermediate) {
    bool needsPacking = !deint2ToLaneStride;
    size_t intermediateCount = needsPacking ? (sourcePartCount + 1) / 2
                                            : sourcePartCount;
    return DataLayoutIntermediatePlan{
        DataLayoutIntermediatePlan::Kind::Contiguous, intermediateCount};
  }

  bool useDeinterleavedIntermediate =
      isSupportedDeinterleavedIntermediateLayout(sourceLayout) &&
      isSupportedDeinterleavedIntermediateLayout(resultLayout);
  if (useDeinterleavedIntermediate) {
    return DataLayoutIntermediatePlan{
        DataLayoutIntermediatePlan::Kind::Deinterleaved, sourcePartCount};
  }
  return std::nullopt;
}

static FailureOr<SmallVector<Value>> materializeDataLayoutThroughContiguous(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    Type sourceVMIElementType, PatternRewriter &rewriter,
    const DataLayoutIntermediatePlan &plan) {
  VMILayoutAttr contiguous =
      VMILayoutAttr::getContiguous(rewriter.getContext());
  SmallVector<Type> intermediateTypes(
      plan.intermediateCount, sourceParts.front().getType());
  FailureOr<SmallVector<Value>> dense = materializeDataLayoutConversion(
      op, sourceParts, intermediateTypes, sourceLayout, contiguous,
      sourceVMIElementType, rewriter);
  if (failed(dense)) {
    return failure();
  }
  return materializeDataLayoutConversion(op, *dense, resultTypes, contiguous,
                                         resultLayout, sourceVMIElementType,
                                         rewriter);
}

static FailureOr<SmallVector<Value>> materializeDataLayoutThroughDeinterleaved(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    Type sourceVMIElementType, PatternRewriter &rewriter) {
  VMILayoutAttr contiguous =
      VMILayoutAttr::getContiguous(rewriter.getContext());
  FailureOr<SmallVector<Value>> dense = materializeDataLayoutConversion(
      op, sourceParts, resultTypes, sourceLayout, contiguous,
      sourceVMIElementType, rewriter);
  if (failed(dense)) {
    return failure();
  }
  return materializeDataLayoutConversion(op, *dense, resultTypes, contiguous,
                                         resultLayout, sourceVMIElementType,
                                         rewriter);
}

FailureOr<std::optional<SmallVector<Value>>>
materializeDataLayoutViaContiguous(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    Type sourceVMIElementType, PatternRewriter &rewriter) {
  std::optional<DataLayoutIntermediatePlan> plan =
      getDataLayoutIntermediatePlan(sourceLayout, resultLayout,
                                    sourceParts.size());
  if (!plan) {
    return std::optional<SmallVector<Value>>{};
  }
  if (sourceParts.empty()) {
    return failure();
  }

  if (plan->kind == DataLayoutIntermediatePlan::Kind::Deinterleaved) {
    FailureOr<SmallVector<Value>> results =
        materializeDataLayoutThroughDeinterleaved(
            op, sourceParts, resultTypes, sourceLayout, resultLayout,
            sourceVMIElementType, rewriter);
    if (failed(results)) {
      return failure();
    }
    return std::optional<SmallVector<Value>>(std::move(*results));
  }

  FailureOr<SmallVector<Value>> results = materializeDataLayoutThroughContiguous(
      op, sourceParts, resultTypes, sourceLayout, resultLayout,
      sourceVMIElementType, rewriter, *plan);
  if (failed(results)) {
    return failure();
  }
  return std::optional<SmallVector<Value>>(std::move(*results));
}

struct DataLayoutMaterializationContext {
  Operation *op;
  ValueRange sourceParts;
  TypeRange resultTypes;
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
  Type sourceVMIElementType;
  PatternRewriter &rewriter;
};

using DataLayoutMaterializationResult =
    FailureOr<std::optional<SmallVector<Value>>>;

static bool didHandleDataLayoutMaterialization(
    const DataLayoutMaterializationResult &result) {
  return failed(result) || result->has_value();
}

static bool isElementDeinterleavedDataLayout(VMILayoutAttr layout,
                                             int64_t factor) {
  return layout && layout.isDeinterleaved() && layout.getFactor() == factor &&
         layout.getLaneStride() == 1;
}

static FailureOr<SmallVector<Value>> materializeDeint4DataToContiguous(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    PatternRewriter &rewriter) {
  int64_t groups = sourceParts.size() / 4;
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (int64_t group = 0; group < groups; ++group) {
    Value p0 = sourceParts[group];
    Value p1 = sourceParts[groups + group];
    Value p2 = sourceParts[2 * groups + group];
    Value p3 = sourceParts[3 * groups + group];
    Type chunkType = p0.getType();
    if (p1.getType() != chunkType || p2.getType() != chunkType ||
        p3.getType() != chunkType) {
      (void)rewriter.notifyMatchFailure(
          op, "vintlv deinterleaved=4 requires matching source part types");
      return failure();
    }
    for (size_t offset = 0; offset < 4; ++offset) {
      if (resultTypes[4 * group + offset] != chunkType) {
        (void)rewriter.notifyMatchFailure(
            op, "vintlv requires operands and results to share one type");
        return failure();
      }
    }
    auto even = rewriter.create<VintlvOp>(op->getLoc(), chunkType, chunkType,
                                          p0, p2);
    auto odd = rewriter.create<VintlvOp>(op->getLoc(), chunkType, chunkType,
                                         p1, p3);
    auto low = rewriter.create<VintlvOp>(op->getLoc(), chunkType, chunkType,
                                         even.getLow(), odd.getLow());
    auto high = rewriter.create<VintlvOp>(op->getLoc(), chunkType, chunkType,
                                          even.getHigh(), odd.getHigh());
    results.append(
        {low.getLow(), low.getHigh(), high.getLow(), high.getHigh()});
  }
  return results;
}

static FailureOr<std::array<Value, 4>> materializeContiguousToDeint4DataGroup(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    int64_t groups, int64_t group, PatternRewriter &rewriter) {
  Value s0 = sourceParts[4 * group];
  Value s1 = sourceParts[4 * group + 1];
  Value s2 = sourceParts[4 * group + 2];
  Value s3 = sourceParts[4 * group + 3];
  Type chunkType = s0.getType();
  if (s0.getType() != s1.getType() || s0.getType() != s2.getType() ||
      s0.getType() != s3.getType()) {
    (void)rewriter.notifyMatchFailure(
        op, "vdintlv deinterleaved=4 requires matching source part types");
    return failure();
  }
  for (size_t offset = 0; offset < 4; ++offset) {
    if (resultTypes[group + offset * groups] != chunkType) {
      (void)rewriter.notifyMatchFailure(
          op, "vdintlv requires operands and results to share one type");
      return failure();
    }
  }
  auto low = rewriter.create<VdintlvOp>(op->getLoc(), chunkType, chunkType, s0,
                                        s1);
  auto high = rewriter.create<VdintlvOp>(op->getLoc(), chunkType, chunkType, s2,
                                         s3);
  auto even = rewriter.create<VdintlvOp>(op->getLoc(), chunkType, chunkType,
                                         low.getLow(), high.getLow());
  auto odd = rewriter.create<VdintlvOp>(op->getLoc(), chunkType, chunkType,
                                        low.getHigh(), high.getHigh());
  return std::array<Value, 4>{
      even.getLow(), odd.getLow(), even.getHigh(), odd.getHigh()};
}

static FailureOr<SmallVector<Value>> materializeContiguousToDeint4Data(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    PatternRewriter &rewriter) {
  int64_t groups = sourceParts.size() / 4;
  SmallVector<Value> part0;
  SmallVector<Value> part1;
  SmallVector<Value> part2;
  SmallVector<Value> part3;
  part0.reserve(groups);
  part1.reserve(groups);
  part2.reserve(groups);
  part3.reserve(groups);
  for (int64_t group = 0; group < groups; ++group) {
    FailureOr<std::array<Value, 4>> groupValues =
        materializeContiguousToDeint4DataGroup(op, sourceParts, resultTypes,
                                               groups, group, rewriter);
    if (failed(groupValues)) {
      return failure();
    }
    part0.push_back((*groupValues)[0]);
    part1.push_back((*groupValues)[1]);
    part2.push_back((*groupValues)[2]);
    part3.push_back((*groupValues)[3]);
  }
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  results.append(part0);
  results.append(part1);
  results.append(part2);
  results.append(part3);
  return results;
}

static DataLayoutMaterializationResult materializeDeinterleaved4DataLayout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    PatternRewriter &rewriter) {
  bool contiguousToDeint4 =
      sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 &&
      isElementDeinterleavedDataLayout(resultLayout, 4);
  bool deint4ToContiguous =
      isElementDeinterleavedDataLayout(sourceLayout, 4) &&
      resultLayout.isContiguous() && resultLayout.getLaneStride() == 1;
  if (!contiguousToDeint4 && !deint4ToContiguous) {
    return std::optional<SmallVector<Value>>{};
  }
  bool invalidArity = sourceParts.empty() ||
                      sourceParts.size() != resultTypes.size() ||
                      resultTypes.size() % 4 != 0;
  if (invalidArity) {
    (void)rewriter.notifyMatchFailure(
        op, "deinterleaved=4 data layout materialization requires 4*N parts");
    return failure();
  }
  FailureOr<SmallVector<Value>> results =
      deint4ToContiguous
          ? materializeDeint4DataToContiguous(op, sourceParts, resultTypes,
                                              rewriter)
          : materializeContiguousToDeint4Data(op, sourceParts, resultTypes,
                                              rewriter);
  if (failed(results)) {
    return failure();
  }
  return std::optional<SmallVector<Value>>(std::move(*results));
}

static DataLayoutMaterializationResult
tryDataLayoutMaterializers(const DataLayoutMaterializationContext &context) {
  DataLayoutMaterializationResult simple =
      materializeSimpleDataLayoutConversion(
          context.op, context.sourceParts, context.resultTypes,
          context.sourceLayout, context.resultLayout,
          context.sourceVMIElementType, context.rewriter);
  if (didHandleDataLayoutMaterialization(simple)) {
    return simple;
  }
  DataLayoutMaterializationResult deinterleaved2 =
      materializeDeinterleaved2Layout(
          context.op, context.sourceParts, context.resultTypes,
          context.sourceLayout, context.resultLayout, context.rewriter);
  if (didHandleDataLayoutMaterialization(deinterleaved2)) {
    return deinterleaved2;
  }
  DataLayoutMaterializationResult deinterleaved4 =
      materializeDeinterleaved4DataLayout(
          context.op, context.sourceParts, context.resultTypes,
          context.sourceLayout, context.resultLayout, context.rewriter);
  if (didHandleDataLayoutMaterialization(deinterleaved4)) {
    return deinterleaved4;
  }
  DataLayoutMaterializationResult laneStride =
      materializeDataLaneStrideConversion(
          context.op, context.sourceParts, context.resultTypes,
          context.sourceLayout, context.resultLayout,
          context.sourceVMIElementType, context.rewriter);
  if (didHandleDataLayoutMaterialization(laneStride)) {
    return laneStride;
  }
  return materializeDataLayoutViaContiguous(
      context.op, context.sourceParts, context.resultTypes,
      context.sourceLayout, context.resultLayout, context.sourceVMIElementType,
      context.rewriter);
}

FailureOr<SmallVector<Value>> materializeDataLayoutConversion(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    Type sourceVMIElementType, PatternRewriter &rewriter) {
  DataLayoutMaterializationContext context{
      op, sourceParts, resultTypes, sourceLayout, resultLayout,
      sourceVMIElementType, rewriter};
  FailureOr<std::optional<SmallVector<Value>>> results =
      tryDataLayoutMaterializers(context);
  if (failed(results)) {
    return failure();
  }
  if (results->has_value()) {
    return std::move(**results);
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
  if (failed(typeConverter.convertType(resultType, resultTypes))) {
    return failure();
  }
  return materializeDataLayoutConversion(op, sourceParts, resultTypes,
                                         sourceLayout, resultLayout,
                                         sourceType.getElementType(), rewriter);
}

enum class PredicateInterleaveKind { Interleave, Deinterleave };

static FailureOr<std::pair<Value, Value>> createPredicateInterleave(
    Location loc, Type lowType, Type highType, Value lhs, Value rhs,
    PredicateInterleaveKind kind, PatternRewriter &rewriter) {
  auto maskType = dyn_cast<MaskType>(lowType);
  if (!maskType || highType != lowType) {
    return failure();
  }
  bool deinterleave = kind == PredicateInterleaveKind::Deinterleave;
  if (maskType.isB8()) {
    if (deinterleave) {
      auto op = rewriter.create<PdintlvB8Op>(loc, lowType, highType, lhs, rhs);
      return std::make_pair(op.getLow(), op.getHigh());
    }
    auto op = rewriter.create<PintlvB8Op>(loc, lowType, highType, lhs, rhs);
    return std::make_pair(op.getLow(), op.getHigh());
  }
  if (maskType.isB16()) {
    if (deinterleave) {
      auto op = rewriter.create<PdintlvB16Op>(loc, lowType, highType, lhs, rhs);
      return std::make_pair(op.getLow(), op.getHigh());
    }
    auto op = rewriter.create<PintlvB16Op>(loc, lowType, highType, lhs, rhs);
    return std::make_pair(op.getLow(), op.getHigh());
  }
  if (maskType.isB32()) {
    if (deinterleave) {
      auto op = rewriter.create<PdintlvB32Op>(loc, lowType, highType, lhs, rhs);
      return std::make_pair(op.getLow(), op.getHigh());
    }
    auto op = rewriter.create<PintlvB32Op>(loc, lowType, highType, lhs, rhs);
    return std::make_pair(op.getLow(), op.getHigh());
  }
  return failure();
}

FailureOr<std::pair<Value, Value>>
createPredicateDintlv(Location loc, Type lowType, Type highType, Value lhs,
                      Value rhs, PatternRewriter &rewriter) {
  return createPredicateInterleave(loc, lowType, highType, lhs, rhs,
                                   PredicateInterleaveKind::Deinterleave,
                                   rewriter);
}

FailureOr<std::pair<Value, Value>>
createPredicateIntlv(Location loc, Type lowType, Type highType, Value lhs,
                     Value rhs, PatternRewriter &rewriter) {
  return createPredicateInterleave(loc, lowType, highType, lhs, rhs,
                                   PredicateInterleaveKind::Interleave,
                                   rewriter);
}

static FailureOr<SmallVector<Value>> materializeDeinterleaved2MaskToContiguous(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    PatternRewriter &rewriter) {
  int64_t groups = sourceParts.size() / 2;
  SmallVector<Value> results;
  results.reserve(sourceParts.size());
  for (int64_t i = 0; i < groups; ++i) {
    FailureOr<std::pair<Value, Value>> materialize = createPredicateIntlv(
        op->getLoc(), resultTypes[2 * i], resultTypes[2 * i + 1],
        sourceParts[i], sourceParts[groups + i], rewriter);
    if (failed(materialize)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported predicate intlv mask type");
    }
    results.append({materialize->first, materialize->second});
  }
  return results;
}

static FailureOr<SmallVector<Value>> materializeContiguousToDeinterleaved2Mask(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    PatternRewriter &rewriter) {
  int64_t groups = sourceParts.size() / 2;
  SmallVector<Value> part0;
  SmallVector<Value> part1;
  part0.reserve(groups);
  part1.reserve(groups);
  for (int64_t i = 0; i < groups; ++i) {
    FailureOr<std::pair<Value, Value>> materialize = createPredicateDintlv(
        op->getLoc(), resultTypes[i], resultTypes[groups + i],
        sourceParts[2 * i], sourceParts[2 * i + 1], rewriter);
    if (failed(materialize)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported predicate dintlv mask type");
    }
    part0.push_back(materialize->first);
    part1.push_back(materialize->second);
  }
  SmallVector<Value> results;
  results.append(part0);
  results.append(part1);
  return results;
}

enum class Deinterleaved2MaskLayoutDirection {
  Unsupported,
  ToContiguous,
  FromContiguous
};

static Deinterleaved2MaskLayoutDirection getDeinterleaved2MaskDirection(
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout) {
  bool sourceIsDeinterleaved2 =
      sourceLayout && sourceLayout.isDeinterleaved() &&
      sourceLayout.getFactor() == 2 && sourceLayout.getLaneStride() == 1;
  bool resultIsDeinterleaved2 =
      resultLayout && resultLayout.isDeinterleaved() &&
      resultLayout.getFactor() == 2 && resultLayout.getLaneStride() == 1;
  bool toContiguous = sourceIsDeinterleaved2 && resultLayout &&
                      resultLayout.isContiguous() &&
                      resultLayout.getLaneStride() == 1;
  bool fromContiguous = sourceLayout && sourceLayout.isContiguous() &&
                        sourceLayout.getLaneStride() == 1 &&
                        resultIsDeinterleaved2;
  if (toContiguous) {
    return Deinterleaved2MaskLayoutDirection::ToContiguous;
  }
  if (fromContiguous) {
    return Deinterleaved2MaskLayoutDirection::FromContiguous;
  }
  return Deinterleaved2MaskLayoutDirection::Unsupported;
}

static FailureOr<std::optional<SmallVector<Value>>>
materializeDeinterleaved2MaskLayout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    PatternRewriter &rewriter) {
  Deinterleaved2MaskLayoutDirection direction =
      getDeinterleaved2MaskDirection(sourceLayout, resultLayout);
  if (direction == Deinterleaved2MaskLayoutDirection::Unsupported) {
    return std::optional<SmallVector<Value>>{};
  }
  bool invalidArity = sourceParts.size() != resultTypes.size() ||
                      sourceParts.empty() || sourceParts.size() % 2 != 0;
  if (invalidArity) {
    (void)rewriter.notifyMatchFailure(
        op, "deinterleaved=2 mask layout materialization requires 2*N parts");
    return failure();
  }
  if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                          rewriter))) {
    return failure();
  }

  if (direction == Deinterleaved2MaskLayoutDirection::ToContiguous) {
    FailureOr<SmallVector<Value>> results =
        materializeDeinterleaved2MaskToContiguous(op, sourceParts, resultTypes,
                                                  rewriter);
    if (failed(results)) {
      return failure();
    }
    return std::optional<SmallVector<Value>>(std::move(*results));
  }
  FailureOr<SmallVector<Value>> results =
      materializeContiguousToDeinterleaved2Mask(op, sourceParts, resultTypes,
                                                rewriter);
  if (failed(results)) {
    return failure();
  }
  return std::optional<SmallVector<Value>>(std::move(*results));
}

static FailureOr<MaskType> getMaskLaneStrideResultType(
    Operation *op, Type resultType, StringRef diagnostic,
    PatternRewriter &rewriter);

static FailureOr<SmallVector<Value>> materializeMaskLaneStrideUnpack(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    int64_t laneStride, PatternRewriter &rewriter) {
  bool resultExceedsSource =
      static_cast<int64_t>(resultTypes.size()) >
      static_cast<int64_t>(sourceParts.size()) * laneStride;
  if (resultExceedsSource) {
    return rewriter.notifyMatchFailure(
        op, "dense mask lane_stride unpack materialization result arity "
            "does not fit source arity");
  }
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  StringAttr lower = rewriter.getStringAttr("LOWER");
  StringAttr higher = rewriter.getStringAttr("HIGHER");
  for (auto [resultIndex, resultType] : llvm::enumerate(resultTypes)) {
    FailureOr<MaskType> maskType = getMaskLaneStrideResultType(
        op, resultType, "dense mask lane_stride unpack requires mask result type",
        rewriter);
    if (failed(maskType)) {
      return failure();
    }
    int64_t safeLaneStride = laneStride > 0 ? laneStride : 1;
    int64_t sourceIndex = resultIndex / safeLaneStride;
    int64_t part = resultIndex % safeLaneStride;
    Value source = sourceParts[sourceIndex];
    StringAttr firstPart = laneStride == 4
                               ? (part >= 2 ? higher : lower)
                               : (part == 1 ? higher : lower);
    Value current = rewriter
                        .create<PunpackOp>(op->getLoc(), *maskType, source,
                                           firstPart)
                        .getResult();
    if (laneStride == 4) {
      current = rewriter.create<PunpackOp>(
          op->getLoc(), *maskType, current, part % 2 == 0 ? lower : higher);
    }
    results.push_back(current);
  }
  return results;
}

static FailureOr<MaskType> getMaskLaneStrideResultType(
    Operation *op, Type resultType, StringRef diagnostic,
    PatternRewriter &rewriter) {
  auto maskType = dyn_cast<MaskType>(resultType);
  if (!maskType) {
    (void)rewriter.notifyMatchFailure(op, diagnostic);
    return failure();
  }
  return maskType;
}

struct MaskLaneStridePackContext {
  Operation *op;
  PatternRewriter &rewriter;
  StringAttr lower;
  StringAttr higher;
  Value allTrue;

  FailureOr<Value> merge(Value lhs, Value rhs) {
    if (!allTrue) {
      FailureOr<Value> mask = createAllTrueMask(
          op->getLoc(), cast<MaskType>(lhs.getType()), rewriter);
      if (failed(mask)) {
        return failure();
      }
      allTrue = *mask;
    }
    return rewriter
        .create<PorOp>(op->getLoc(), lhs.getType(), lhs, rhs, allTrue)
        .getResult();
  }

  FailureOr<Value> packPair(Value lowSource, std::optional<Value> highSource,
                            MaskType maskType) {
    Value packed = rewriter.create<PpackOp>(op->getLoc(), maskType, lowSource,
                                            lower);
    if (!highSource) {
      return packed;
    }
    Value higherPacked = rewriter.create<PpackOp>(
        op->getLoc(), maskType, *highSource, higher);
    return merge(packed, higherPacked);
  }
};

static FailureOr<Value> materializeMaskLaneStridePackChunk(
    Operation *op, ValueRange sourceParts, size_t base, int64_t laneStride,
    MaskType maskType, MaskLaneStridePackContext &context,
    PatternRewriter &rewriter) {
  std::optional<Value> source1;
  if (base + 1 < sourceParts.size()) {
    source1 = sourceParts[base + 1];
  }
  FailureOr<Value> lowHalf =
      context.packPair(sourceParts[base], source1, maskType);
  if (failed(lowHalf)) {
    return failure();
  }
  Value current = *lowHalf;
  if (laneStride != 4) {
    return current;
  }
  current = rewriter.create<PpackOp>(op->getLoc(), maskType, current,
                                     context.lower);
  if (base + 2 >= sourceParts.size()) {
    return current;
  }
  std::optional<Value> source3;
  if (base + 3 < sourceParts.size()) {
    source3 = sourceParts[base + 3];
  }
  FailureOr<Value> highHalf =
      context.packPair(sourceParts[base + 2], source3, maskType);
  if (failed(highHalf)) {
    return failure();
  }
  Value higherPacked =
      rewriter.create<PpackOp>(op->getLoc(), maskType, *highHalf,
                                context.higher);
  return context.merge(current, higherPacked);
}

static FailureOr<SmallVector<Value>> materializeMaskLaneStridePack(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    int64_t laneStride, PatternRewriter &rewriter) {
  if (sourceParts.empty()) {
    return rewriter.notifyMatchFailure(
        op, "dense mask lane_stride pack materialization requires source parts");
  }
  bool sourceExceedsResult =
      static_cast<int64_t>(sourceParts.size()) >
      static_cast<int64_t>(resultTypes.size()) * laneStride;
  if (sourceExceedsResult) {
    return rewriter.notifyMatchFailure(
        op, "dense mask lane_stride pack materialization source arity does "
            "not fit result arity");
  }
  MaskLaneStridePackContext context{
      op, rewriter, rewriter.getStringAttr("LOWER"),
      rewriter.getStringAttr("HIGHER"), Value()};

  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (auto [resultIndex, resultType] : llvm::enumerate(resultTypes)) {
    FailureOr<MaskType> maskType = getMaskLaneStrideResultType(
        op, resultType, "dense mask lane_stride pack requires mask result type",
        rewriter);
    if (failed(maskType)) {
      return failure();
    }
    size_t base = resultIndex * static_cast<size_t>(laneStride);
    if (base >= sourceParts.size()) {
      break;
    }
    FailureOr<Value> current = materializeMaskLaneStridePackChunk(
        op, sourceParts, base, laneStride, *maskType, context, rewriter);
    if (failed(current)) {
      return failure();
    }
    results.push_back(*current);
  }
  bool resultArityMismatch = results.size() != resultTypes.size();
  if (resultArityMismatch) {
    return rewriter.notifyMatchFailure(
        op, "dense mask lane_stride pack materialization result arity mismatch");
  }
  return results;
}

struct MaskLaneStrideLayoutPlan {
  bool unpack;
  int64_t laneStride;
};

static std::optional<MaskLaneStrideLayoutPlan> getMaskLaneStrideLayoutPlan(
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout) {
  bool unpack = sourceLayout && sourceLayout.isContiguous() &&
                sourceLayout.getLaneStride() == 1 && resultLayout &&
                resultLayout.isContiguous() &&
                resultLayout.getLaneStride() != 1;
  bool pack = sourceLayout && sourceLayout.isContiguous() &&
              sourceLayout.getLaneStride() != 1 && resultLayout &&
              resultLayout.isContiguous() && resultLayout.getLaneStride() == 1;
  if (!unpack && !pack) {
    return std::nullopt;
  }
  return MaskLaneStrideLayoutPlan{
      unpack, unpack ? resultLayout.getLaneStride() : sourceLayout.getLaneStride()};
}

static LogicalResult checkMaskLaneStrideFactor(
    Operation *op, const MaskLaneStrideLayoutPlan &plan,
    PatternRewriter &rewriter) {
  bool supportedFactor = plan.laneStride == 2 || plan.laneStride == 4;
  if (supportedFactor) {
    return success();
  }
  return rewriter.notifyMatchFailure(
      op, plan.unpack ? "unsupported dense mask lane_stride unpack factor"
                      : "unsupported dense mask lane_stride pack factor");
}

FailureOr<std::optional<SmallVector<Value>>> materializeMaskLaneStrideLayout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    PatternRewriter &rewriter) {
  std::optional<MaskLaneStrideLayoutPlan> plan =
      getMaskLaneStrideLayoutPlan(sourceLayout, resultLayout);
  if (!plan) {
    return std::optional<SmallVector<Value>>{};
  }

  if (failed(checkMaskLaneStrideFactor(op, *plan, rewriter))) {
    return failure();
  }

  if (plan->unpack) {
    FailureOr<SmallVector<Value>> results = materializeMaskLaneStrideUnpack(
        op, sourceParts, resultTypes, plan->laneStride, rewriter);
    if (failed(results)) {
      return failure();
    }
    return std::optional<SmallVector<Value>>(std::move(*results));
  }
  FailureOr<SmallVector<Value>> results = materializeMaskLaneStridePack(
      op, sourceParts, resultTypes, plan->laneStride, rewriter);
  if (failed(results)) {
    return failure();
  }
  return std::optional<SmallVector<Value>>(std::move(*results));
}

static FailureOr<std::optional<SmallVector<Value>>> materializeIdentityMaskLayout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    PatternRewriter &rewriter) {
  if (sourceLayout != resultLayout) {
    return std::optional<SmallVector<Value>>{};
  }
  if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                          rewriter))) {
    return failure();
  }
  return std::optional<SmallVector<Value>>(
      SmallVector<Value>(sourceParts.begin(), sourceParts.end()));
}


struct MaskLayoutMaterializationContext {
  Operation *op;
  ValueRange sourceParts;
  TypeRange resultTypes;
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
  PatternRewriter &rewriter;
};

using MaskLayoutMaterializationResult =
    FailureOr<std::optional<SmallVector<Value>>>;

static bool didHandleMaskLayoutMaterialization(
    const MaskLayoutMaterializationResult &result) {
  return failed(result) || result->has_value();
}

FailureOr<SmallVector<Value>> materializeStagingDeintToContiguousMaskLayout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    int64_t factor, PatternRewriter &rewriter);
FailureOr<SmallVector<Value>> materializeStagingContiguousToDeintMaskLayout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    int64_t factor, PatternRewriter &rewriter);

static bool isElementDeinterleavedMaskLayout(VMILayoutAttr layout,
                                             int64_t factor) {
  return layout && layout.isDeinterleaved() && layout.getFactor() == factor &&
         layout.getLaneStride() == 1;
}

static MaskLayoutMaterializationResult materializeDeinterleaved4MaskLayout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    PatternRewriter &rewriter) {
  bool contiguousToDeint4 =
      sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 &&
      isElementDeinterleavedMaskLayout(resultLayout, 4);
  bool deint4ToContiguous =
      isElementDeinterleavedMaskLayout(sourceLayout, 4) &&
      resultLayout.isContiguous() && resultLayout.getLaneStride() == 1;
  if (!contiguousToDeint4 && !deint4ToContiguous) {
    return std::optional<SmallVector<Value>>{};
  }
  bool invalidArity = sourceParts.empty() ||
                      sourceParts.size() != resultTypes.size() ||
                      resultTypes.size() % 4 != 0;
  if (invalidArity) {
    (void)rewriter.notifyMatchFailure(
        op, "deinterleaved=4 mask layout materialization requires 4*N parts");
    return failure();
  }
  if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                          rewriter))) {
    return failure();
  }
  FailureOr<SmallVector<Value>> results =
      contiguousToDeint4
          ? materializeStagingContiguousToDeintMaskLayout(
                op, sourceParts, resultTypes, 4, rewriter)
          : materializeStagingDeintToContiguousMaskLayout(
                op, sourceParts, resultTypes, 4, rewriter);
  if (failed(results)) {
    return failure();
  }
  return std::optional<SmallVector<Value>>(std::move(*results));
}

static MaskLayoutMaterializationResult
tryMaskLayoutMaterializers(const MaskLayoutMaterializationContext &context) {
  MaskLayoutMaterializationResult identity = materializeIdentityMaskLayout(
      context.op, context.sourceParts, context.resultTypes,
      context.sourceLayout, context.resultLayout, context.rewriter);
  if (didHandleMaskLayoutMaterialization(identity)) {
    return identity;
  }

  MaskLayoutMaterializationResult deinterleaved2 =
      materializeDeinterleaved2MaskLayout(
          context.op, context.sourceParts, context.resultTypes,
          context.sourceLayout, context.resultLayout, context.rewriter);
  if (didHandleMaskLayoutMaterialization(deinterleaved2)) {
    return deinterleaved2;
  }

  MaskLayoutMaterializationResult deinterleaved4 =
      materializeDeinterleaved4MaskLayout(
          context.op, context.sourceParts, context.resultTypes,
          context.sourceLayout, context.resultLayout, context.rewriter);
  if (didHandleMaskLayoutMaterialization(deinterleaved4)) {
    return deinterleaved4;
  }

  return materializeMaskLaneStrideLayout(
      context.op, context.sourceParts, context.resultTypes, context.sourceLayout,
      context.resultLayout, context.rewriter);
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

  MaskLayoutMaterializationContext context{
      op, sourceParts, resultTypes, sourceLayout, resultLayout, rewriter};
  FailureOr<std::optional<SmallVector<Value>>> results =
      tryMaskLayoutMaterializers(context);
  if (failed(results)) {
    return failure();
  }
  if (results->has_value()) {
    return std::move(**results);
  }

  (void)rewriter.notifyMatchFailure(
      op, "unsupported VMI mask layout materialization");
  return failure();
}

int getMaskGranularityRank(StringRef granularity) {
  if (granularity == "b8") {
    return 0;
  }
  if (granularity == "b16") {
    return 1;
  }
  if (granularity == "b32") {
    return 2;
  }
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
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  bool laneCountMismatch =
      sourceType.getElementCount() != resultType.getElementCount();
  if (laneCountMismatch) {
    return fail("requires source and result mask lane counts to match");
  }
  bool layoutMismatch = sourceType.getLayoutAttr() != resultType.getLayoutAttr();
  if (layoutMismatch) {
    return fail("requires source and result mask layouts to match");
  }

  bool nonConcreteGranularity =
      !VMIMaskType::isConcreteGranularity(sourceType.getGranularity()) ||
      !VMIMaskType::isConcreteGranularity(resultType.getGranularity());
  if (nonConcreteGranularity) {
    return fail("requires concrete b8/b16/b32 source and result "
                "granularities");
  }

  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  bool missingArity = failed(sourceArity) || failed(resultArity);
  if (missingArity) {
    return fail("requires computable source/result physical arity");
  }
  if (*sourceArity < 1 || *resultArity < 1) {
    return fail("requires non-empty source/result physical arity");
  }

  return success();
}

FailureOr<SmallVector<Value>> materializeWideningMaskGranularityPart(
    Operation *op, MaskType resultMaskType, ValueRange sourceParts,
    int64_t sourceOffset, int64_t sourceChunks, int64_t resultChunks,
    PatternRewriter &rewriter) {
  SmallVector<Value> results;
  auto partAttr = StringAttr::get(op->getContext(), "LOWER");
  auto higherAttr = StringAttr::get(op->getContext(), "HIGHER");
  int64_t produced = 0;
  for (int64_t chunk = 0; chunk < sourceChunks && produced < resultChunks;
       ++chunk) {
    Value source = sourceParts[sourceOffset + chunk];
    results.push_back(rewriter
                          .create<PunpackOp>(op->getLoc(), resultMaskType,
                                             source, partAttr)
                          .getResult());
    ++produced;
    if (produced < resultChunks) {
      results.push_back(rewriter
                            .create<PunpackOp>(op->getLoc(), resultMaskType,
                                               source, higherAttr)
                            .getResult());
      ++produced;
    }
  }
  if (produced != resultChunks) {
    (void)rewriter.notifyMatchFailure(
        op, "widening mask granularity conversion produced the wrong number "
            "of result chunks");
    return failure();
  }
  return results;
}

static FailureOr<Value> materializeNarrowingMaskChunk(
    Operation *op, MaskType resultMaskType, ValueRange sourceParts,
    int64_t sourceOffset, int64_t sourceChunks, int64_t &consumed,
    Value &allTrue, StringAttr lowerAttr, StringAttr higherAttr,
    PatternRewriter &rewriter) {
  if (consumed >= sourceChunks) {
    (void)rewriter.notifyMatchFailure(
        op, "narrowing mask granularity conversion ran out of source chunks");
    return failure();
  }
  Value lowerSource = sourceParts[sourceOffset + consumed++];
  Value packed = rewriter
                     .create<PpackOp>(op->getLoc(), resultMaskType, lowerSource,
                                      lowerAttr)
                     .getResult();
  if (consumed >= sourceChunks) {
    return packed;
  }
  Value higherSource = sourceParts[sourceOffset + consumed++];
  Value higher = rewriter
                     .create<PpackOp>(op->getLoc(), resultMaskType, higherSource,
                                      higherAttr)
                     .getResult();
  if (!allTrue) {
    FailureOr<Value> mask =
        createAllTrueMask(op->getLoc(), resultMaskType, rewriter);
    if (failed(mask)) {
      (void)rewriter.notifyMatchFailure(
          op, "failed to create all-true mask for ppack merge");
      return failure();
    }
    allTrue = *mask;
  }
  return rewriter
      .create<PorOp>(op->getLoc(), resultMaskType, packed, higher, allTrue)
      .getResult();
}

FailureOr<SmallVector<Value>> materializeNarrowingMaskGranularityPart(
    Operation *op, MaskType resultMaskType, ValueRange sourceParts,
    int64_t sourceOffset, int64_t sourceChunks, int64_t resultChunks,
    PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message)
      -> FailureOr<SmallVector<Value>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  auto lowerAttr = StringAttr::get(op->getContext(), "LOWER");
  auto higherAttr = StringAttr::get(op->getContext(), "HIGHER");
  SmallVector<Value> results;
  Value allTrue;
  int64_t consumed = 0;
  for (int64_t chunk = 0; chunk < resultChunks; ++chunk) {
    FailureOr<Value> packed = materializeNarrowingMaskChunk(
        op, resultMaskType, sourceParts, sourceOffset, sourceChunks, consumed,
        allTrue, lowerAttr, higherAttr, rewriter);
    if (failed(packed)) {
      return failure();
    }
    results.push_back(*packed);
  }
  if (consumed != sourceChunks) {
    return fail("narrowing mask granularity conversion left unused source "
                "chunks");
  }
  return results;
}

static FailureOr<SmallVector<Value>> materializeAdjacentMaskGranularityPart(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, int sourceRank, int resultRank,
    MaskType resultMaskType, int64_t part, int64_t sourceOffset,
    PatternRewriter &rewriter) {
  FailureOr<int64_t> sourceChunks = getVMITypeChunksInPart(sourceType, part);
  FailureOr<int64_t> resultChunks = getVMITypeChunksInPart(resultType, part);
  bool invalidChunkCounts = failed(sourceChunks) || failed(resultChunks);
  if (invalidChunkCounts) {
    (void)rewriter.notifyMatchFailure(
        op, "requires computable source/result chunks per layout part");
    return failure();
  }
  bool widening = resultRank > sourceRank;
  if (widening) {
    return materializeWideningMaskGranularityPart(
        op, resultMaskType, sourceParts, sourceOffset, *sourceChunks,
        *resultChunks, rewriter);
  }
  return materializeNarrowingMaskGranularityPart(
      op, resultMaskType, sourceParts, sourceOffset, *sourceChunks,
      *resultChunks, rewriter);
}

struct MaskGranularityConversionPlan {
  int sourceRank;
  int resultRank;
  int64_t sourceArity;
  int64_t layoutFactor;
  MaskType resultMaskType;
};

static FailureOr<MaskGranularityConversionPlan>
buildMaskGranularityConversionPlan(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message)
      -> FailureOr<MaskGranularityConversionPlan> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  int sourceRank = getMaskGranularityRank(sourceType.getGranularity());
  int resultRank = getMaskGranularityRank(resultType.getGranularity());
  bool nonAdjacent = std::abs(sourceRank - resultRank) != 1;
  if (nonAdjacent) {
    return fail("mask granularity conversion must be adjacent");
  }
  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> factor = getVMITypeLayoutFactor(sourceType);
  bool sourceArityMismatch =
      failed(sourceArity) || failed(factor) ||
      static_cast<int64_t>(sourceParts.size()) != *sourceArity;
  if (sourceArityMismatch) {
    return fail("source mask part count does not match source VMI type");
  }
  return MaskGranularityConversionPlan{
      sourceRank, resultRank, *sourceArity, *factor,
      MaskType::get(op->getContext(), resultType.getGranularity())};
}

static FailureOr<SmallVector<Value>> materializeMaskGranularityParts(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, const MaskGranularityConversionPlan &plan,
    PatternRewriter &rewriter) {
  SmallVector<Value> results;
  int64_t sourceOffset = 0;
  for (int64_t part = 0; part < plan.layoutFactor; ++part) {
    FailureOr<int64_t> sourceChunks = getVMITypeChunksInPart(sourceType, part);
    if (failed(sourceChunks)) {
      (void)rewriter.notifyMatchFailure(
          op, "requires computable source chunks per layout part");
      return failure();
    }
    FailureOr<SmallVector<Value>> partResults =
        materializeAdjacentMaskGranularityPart(
            op, sourceType, resultType, sourceParts, plan.sourceRank,
            plan.resultRank, plan.resultMaskType, part, sourceOffset, rewriter);
    if (failed(partResults)) {
      return failure();
    }
    results.append(*partResults);
    sourceOffset += *sourceChunks;
  }
  return results;
}

static LogicalResult checkMaskGranularityResultArity(
    Operation *op, VMIMaskType resultType, size_t resultCount,
    PatternRewriter &rewriter) {
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  bool resultArityMismatch =
      failed(resultArity) || static_cast<int64_t>(resultCount) != *resultArity;
  if (resultArityMismatch) {
    (void)rewriter.notifyMatchFailure(
        op, "mask granularity conversion result count mismatch");
    return failure();
  }
  return success();
}

FailureOr<SmallVector<Value>> materializeAdjacentMaskGranularityConversion(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, PatternRewriter &rewriter) {
  FailureOr<MaskGranularityConversionPlan> plan =
      buildMaskGranularityConversionPlan(op, sourceType, resultType,
                                          sourceParts, rewriter);
  if (failed(plan)) {
    return failure();
  }
  FailureOr<SmallVector<Value>> results = materializeMaskGranularityParts(
      op, sourceType, resultType, sourceParts, *plan, rewriter);
  if (failed(results)) {
    return failure();
  }

  if (failed(checkMaskGranularityResultArity(op, resultType, results->size(),
                                             rewriter))) {
    return failure();
  }
  return *results;
}

static FailureOr<SmallVector<Value>> materializeMaskGranularityStep(
    Operation *op, VMIMaskType currentType, StringRef nextGranularity,
    ValueRange currentParts, PatternRewriter &rewriter) {
  VMIMaskType nextType = VMIMaskType::get(
      op->getContext(), currentType.getElementCount(), nextGranularity,
      currentType.getLayoutAttr());
  return materializeAdjacentMaskGranularityConversion(
      op, currentType, nextType, currentParts, rewriter);
}

static FailureOr<VMIMaskType> buildNextMaskGranularityType(
    VMIMaskType currentType, int rank, PatternRewriter &rewriter,
    Operation *op) {
  StringRef granularity = getMaskGranularityForRank(rank);
  if (granularity.empty()) {
    (void)rewriter.notifyMatchFailure(
        op, "invalid target mask granularity rank");
    return failure();
  }
  return VMIMaskType::get(op->getContext(), currentType.getElementCount(),
                          granularity, currentType.getLayoutAttr());
}

static FailureOr<SmallVector<Value>> materializeMaskGranularitySteps(
    Operation *op, VMIMaskType sourceType, ValueRange sourceParts,
    int sourceRank, int resultRank, PatternRewriter &rewriter) {
  VMIMaskType currentType = sourceType;
  SmallVector<Value> currentParts(sourceParts.begin(), sourceParts.end());
  int currentRank = sourceRank;
  while (currentRank != resultRank) {
    bool ascending = currentRank < resultRank;
    currentRank += ascending ? 1 : -1;
    FailureOr<VMIMaskType> nextType = buildNextMaskGranularityType(
        currentType, currentRank, rewriter, op);
    if (failed(nextType)) {
      return failure();
    }
    FailureOr<SmallVector<Value>> nextParts = materializeMaskGranularityStep(
        op, currentType, nextType->getGranularity(), currentParts, rewriter);
    if (failed(nextParts)) {
      return failure();
    }
    currentType = *nextType;
    currentParts = std::move(*nextParts);
  }
  return currentParts;
}

struct MaskGranularityRoute {
  int sourceRank;
  int resultRank;
  bool adjacent;
};

static FailureOr<MaskGranularityRoute> classifyMaskGranularityRoute(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    PatternRewriter &rewriter) {
  auto fail = [&rewriter, op](const Twine &message)
      -> FailureOr<MaskGranularityRoute> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  int sourceRank = getMaskGranularityRank(sourceType.getGranularity());
  int resultRank = getMaskGranularityRank(resultType.getGranularity());
  if (sourceRank < 0 || resultRank < 0) {
    return fail("requires concrete source and result mask granularity ranks");
  }
  return MaskGranularityRoute{sourceRank, resultRank,
                              std::abs(sourceRank - resultRank) == 1};
}

FailureOr<SmallVector<Value>> materializeMaskGranularityConversion(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType, ValueRange sourceParts,
    PatternRewriter &rewriter) {
  std::string reason;
  if (failed(checkSupportedMaskGranularityMaterialization(sourceType, resultType, &reason))) {
    (void)rewriter.notifyMatchFailure(op, reason);
    return failure();
  }

  FailureOr<MaskGranularityRoute> route = classifyMaskGranularityRoute(
      op, sourceType, resultType, rewriter);
  if (failed(route)) {
    return failure();
  }
  if (route->adjacent) {
    return materializeAdjacentMaskGranularityConversion(
        op, sourceType, resultType, sourceParts, rewriter);
  }

  return materializeMaskGranularitySteps(op, sourceType, sourceParts,
                                         route->sourceRank, route->resultRank,
                                         rewriter);
}

static SmallVector<Type> repeatMaskPartType(Type partType, int64_t arity) {
  SmallVector<Type> types;
  types.reserve(arity);
  for (int64_t i = 0; i < arity; ++i) {
    types.push_back(partType);
  }
  return types;
}

FailureOr<SmallVector<Type>> getConvertedMaskPartTypes(VMIMaskType type) {
  FailureOr<int64_t> arity = getVMIPhysicalArity(type);
  FailureOr<StringRef> physicalGranularity =
      getVMIMaskPhysicalGranularity(type);
  bool invalidMaskTypes =
      failed(arity) || failed(physicalGranularity) || *arity < 0;
  if (invalidMaskTypes) {
    return failure();
  }
  Type partType = MaskType::get(type.getContext(), *physicalGranularity);
  return repeatMaskPartType(partType, *arity);
}

static FailureOr<VMILayoutAttr>
getVMIMaskPhysicalCarrierLayout(VMIMaskType type) {
  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout) {
    return failure();
  }
  MLIRContext *ctx = type.getContext();
  if (layout.isContiguous()) {
    return VMILayoutAttr::getContiguous(ctx);
  }
  if (layout.isDeinterleaved()) {
    return VMILayoutAttr::getDeinterleaved(ctx, layout.getFactor());
  }
  if (layout.isBlockDeinterleaved()) {
    return VMILayoutAttr::getBlockDeinterleaved(ctx, layout.getFactor());
  }
  if (layout.isGroupSlots()) {
    return VMILayoutAttr::getGroupSlots(ctx, layout.getNumGroups(),
                                        layout.getSlots());
  }
  return failure();
}

static FailureOr<VMIMaskType>
getVMIMaskPhysicalCarrierType(VMIMaskType type) {
  FailureOr<StringRef> physicalGranularity =
      getVMIMaskPhysicalGranularity(type);
  FailureOr<VMILayoutAttr> physicalLayout =
      getVMIMaskPhysicalCarrierLayout(type);
  bool missingPhysicalCarrier = failed(physicalGranularity) ||
                                failed(physicalLayout);
  if (missingPhysicalCarrier) {
    return failure();
  }
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
  if (!maskType) {
    return failure();
  }
  return createPrefixMask(loc, maskType, "PAT_ALLF", rewriter);
}

FailureOr<std::array<Value, 4>> materializeFactor4DeintToContiguousGroup(
    Operation *op, ArrayRef<Value> sources, TypeRange resultTypes,
    size_t resultOffset, PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message)
      -> FailureOr<std::array<Value, 4>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  bool invalidSources = sources.size() != 4;
  bool missingResultTypes = resultTypes.empty();
  if (invalidSources || missingResultTypes) {
    return fail("factor-4 staging mask conversion requires four sources");
  }
  auto resultTypeAt = [resultTypes, resultOffset](size_t offset) -> Type {
    size_t index = resultOffset + offset;
    return index < resultTypes.size() ? resultTypes[index]
                                      : resultTypes[resultTypes.size() - 1];
  };
  FailureOr<std::pair<Value, Value>> even = createPredicateIntlv(
      op->getLoc(), resultTypeAt(0), resultTypeAt(1), sources[0], sources[2],
      rewriter);
  FailureOr<std::pair<Value, Value>> odd = createPredicateIntlv(
      op->getLoc(), resultTypeAt(0), resultTypeAt(1), sources[1], sources[3],
      rewriter);
  if (failed(even)) {
    return fail("unsupported predicate intlv staging mask type");
  }
  if (failed(odd)) {
    return fail("unsupported predicate intlv staging mask type");
  }
  FailureOr<std::pair<Value, Value>> low = createPredicateIntlv(
      op->getLoc(), resultTypeAt(0), resultTypeAt(1), even->first, odd->first,
      rewriter);
  FailureOr<std::pair<Value, Value>> high = createPredicateIntlv(
      op->getLoc(), resultTypeAt(2), resultTypeAt(3), even->second,
      odd->second, rewriter);
  if (failed(low)) {
    return fail("unsupported predicate intlv staging mask type");
  }
  if (failed(high)) {
    return fail("unsupported predicate intlv staging mask type");
  }
  return std::array<Value, 4>{low->first, low->second, high->first,
                              high->second};
}

static FailureOr<std::array<Value, 2>> materializeFactor2DeintToContiguousGroup(
    Operation *op, ArrayRef<Value> sources, TypeRange resultTypes,
    size_t resultOffset, PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message)
      -> FailureOr<std::array<Value, 2>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  bool invalidSources = sources.size() != 2 || resultTypes.empty();
  if (invalidSources) {
    return fail("factor-2 staging mask conversion requires two sources");
  }
  size_t first = std::min(resultOffset, resultTypes.size() - 1);
  size_t second = std::min(resultOffset + 1, resultTypes.size() - 1);
  FailureOr<std::pair<Value, Value>> materialized = createPredicateIntlv(
      op->getLoc(), resultTypes[first], resultTypes[second], sources[0],
      sources[1], rewriter);
  if (failed(materialized)) {
    return fail("unsupported predicate intlv staging mask type");
  }
  return std::array<Value, 2>{materialized->first, materialized->second};
}

static FailureOr<SmallVector<Value, 4>> materializeDeintToContiguousMaskGroup(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes, int64_t factor,
    int64_t groups, int64_t groupIndex, size_t resultOffset,
    PatternRewriter &rewriter) {
  SmallVector<Value, 4> sources;
  sources.reserve(factor);
  for (int64_t part = 0; part < factor; ++part) {
    sources.push_back(sourceParts[part * groups + groupIndex]);
  }
  SmallVector<Value, 4> results;
  if (factor == 2) {
    FailureOr<std::array<Value, 2>> materialized =
        materializeFactor2DeintToContiguousGroup(
            op, sources, resultTypes, resultOffset, rewriter);
    if (failed(materialized)) {
      return failure();
    }
    results.append(materialized->begin(), materialized->end());
    return results;
  }
    FailureOr<std::array<Value, 4>> materialized =
        materializeFactor4DeintToContiguousGroup(
            op, sources, resultTypes, resultOffset, rewriter);
  if (failed(materialized)) {
    return failure();
  }
  results.append(materialized->begin(), materialized->end());
  return results;
}

FailureOr<SmallVector<Value>> materializeStagingDeintToContiguousMaskLayout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    int64_t factor, PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message) -> FailureOr<SmallVector<Value>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  bool invalidGroups = (factor != 2 && factor != 4) || sourceParts.empty() ||
                       sourceParts.size() % factor != 0;
  if (invalidGroups) {
    return fail("staging deinterleaved mask layout requires grouped source "
                "parts");
  }

  int64_t groups = sourceParts.size() / factor;
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (int64_t i = 0; i < groups && results.size() < resultTypes.size(); ++i) {
    FailureOr<SmallVector<Value, 4>> materialized =
        materializeDeintToContiguousMaskGroup(
            op, sourceParts, resultTypes, factor, groups, i, results.size(),
            rewriter);
    if (failed(materialized)) {
      return failure();
    }
    for (Value value : *materialized) {
      bool resultCapacityReached = results.size() >= resultTypes.size();
      if (resultCapacityReached) {
        break;
      }
      results.push_back(value);
    }
  }
  bool resultArityMismatch = results.size() != resultTypes.size();
  if (resultArityMismatch) {
    return fail("staging deinterleaved mask layout result arity mismatch");
  }
  return results;
}

FailureOr<std::array<Value, 4>> materializeFactor4ContiguousToDeintGroup(
    Operation *op, ArrayRef<Value> sources, TypeRange resultTypes,
    int64_t groups, int64_t groupIndex, PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message)
      -> FailureOr<std::array<Value, 4>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  bool invalidSources = sources.size() != 4;
  bool insufficientResults =
      resultTypes.size() < static_cast<size_t>(4 * groups);
  if (invalidSources || insufficientResults) {
    return fail("factor-4 staging mask conversion requires four grouped results");
  }
  FailureOr<std::pair<Value, Value>> low = createPredicateDintlv(
      op->getLoc(), resultTypes[groupIndex], resultTypes[groups + groupIndex],
      sources[0], sources[1], rewriter);
  FailureOr<std::pair<Value, Value>> high = createPredicateDintlv(
      op->getLoc(), resultTypes[2 * groups + groupIndex],
      resultTypes[3 * groups + groupIndex], sources[2], sources[3], rewriter);
  if (failed(low)) {
    return fail("unsupported predicate dintlv staging mask type");
  }
  if (failed(high)) {
    return fail("unsupported predicate dintlv staging mask type");
  }
  FailureOr<std::pair<Value, Value>> even = createPredicateDintlv(
      op->getLoc(), resultTypes[groupIndex], resultTypes[2 * groups + groupIndex],
      low->first, high->first, rewriter);
  FailureOr<std::pair<Value, Value>> odd = createPredicateDintlv(
      op->getLoc(), resultTypes[groups + groupIndex],
      resultTypes[3 * groups + groupIndex], low->second, high->second, rewriter);
  if (failed(even)) {
    return fail("unsupported predicate dintlv staging mask type");
  }
  if (failed(odd)) {
    return fail("unsupported predicate dintlv staging mask type");
  }
  return std::array<Value, 4>{even->first, odd->first, even->second,
                              odd->second};
}

static FailureOr<std::array<Value, 2>> materializeFactor2ContiguousToDeintGroup(
    Operation *op, ArrayRef<Value> sources, TypeRange resultTypes,
    int64_t groups, int64_t groupIndex, PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message)
      -> FailureOr<std::array<Value, 2>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  bool invalidGroup =
      sources.size() != 2 ||
      resultTypes.size() < static_cast<size_t>(2 * groups);
  if (invalidGroup) {
    return fail("factor-2 staging mask conversion requires two grouped results");
  }
  FailureOr<std::pair<Value, Value>> materialized = createPredicateDintlv(
      op->getLoc(), resultTypes[groupIndex], resultTypes[groups + groupIndex],
      sources[0], sources[1], rewriter);
  if (failed(materialized)) {
    return fail("unsupported predicate dintlv staging mask type");
  }
  return std::array<Value, 2>{materialized->first, materialized->second};
}

static FailureOr<SmallVector<Value, 4>> materializeContiguousToDeintMaskGroup(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes, int64_t factor,
    int64_t groups, int64_t groupIndex, PatternRewriter &rewriter) {
  SmallVector<Value, 4> sources;
  size_t sourceBase = static_cast<size_t>(groupIndex * factor);
  if (sourceBase >= sourceParts.size()) {
    (void)rewriter.notifyMatchFailure(
        op, "staging contiguous mask layout ran out of source parts");
    return failure();
  }
  sources.reserve(factor);
  for (int64_t lane = 0; lane < factor; ++lane) {
    size_t index = sourceBase + lane;
    if (index < sourceParts.size()) {
      sources.push_back(sourceParts[index]);
      continue;
    }
    FailureOr<Value> zero = createAllFalseMaskLike(
        op->getLoc(), sourceParts[sourceBase], rewriter);
    if (failed(zero)) {
      (void)rewriter.notifyMatchFailure(
          op, "failed to create all-false staging mask");
      return failure();
    }
    sources.push_back(*zero);
  }
  SmallVector<Value, 4> results;
  if (factor == 2) {
    FailureOr<std::array<Value, 2>> materialized =
        materializeFactor2ContiguousToDeintGroup(
            op, sources, resultTypes, groups, groupIndex, rewriter);
    if (failed(materialized)) {
      return failure();
    }
    results.append(materialized->begin(), materialized->end());
    return results;
  }
  FailureOr<std::array<Value, 4>> materialized =
      materializeFactor4ContiguousToDeintGroup(
          op, sources, resultTypes, groups, groupIndex, rewriter);
  if (failed(materialized)) {
    return failure();
  }
  results.append(materialized->begin(), materialized->end());
  return results;
}

struct StagingMaskPartAccumulator {
  SmallVector<SmallVector<Value, 4>, 4> parts;
  int64_t factor;
  int64_t groups;

  StagingMaskPartAccumulator(int64_t factor, int64_t groups)
      : parts(factor), factor(factor), groups(groups) {
    for (SmallVector<Value, 4> &part : parts) {
      part.reserve(static_cast<size_t>(groups));
    }
  }

  LogicalResult append(Operation *op, ArrayRef<Value> values,
                       PatternRewriter &rewriter) {
    bool invalidArity = values.size() != static_cast<size_t>(factor);
    if (invalidArity) {
      (void)rewriter.notifyMatchFailure(
          op, "staging contiguous mask layout result arity mismatch");
      return failure();
    }
    for (int64_t part = 0; part < factor; ++part) {
      parts[part].push_back(values[part]);
    }
    return success();
  }

  FailureOr<SmallVector<Value>> flatten(Operation *op,
                                        TypeRange resultTypes,
                                        PatternRewriter &rewriter) {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (int64_t part = 0; part < factor; ++part) {
      bool invalidPartArity =
          parts[part].size() != static_cast<size_t>(groups);
      if (invalidPartArity) {
        (void)rewriter.notifyMatchFailure(
            op, "staging contiguous mask layout result arity mismatch");
        return failure();
      }
      results.append(parts[part]);
    }
    return results;
  }
};

FailureOr<SmallVector<Value>> materializeStagingContiguousToDeintMaskLayout(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    int64_t factor, PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message) -> FailureOr<SmallVector<Value>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  bool invalidGroupedParts =
      (factor != 2 && factor != 4) || sourceParts.empty() ||
      resultTypes.size() % factor != 0;
  if (invalidGroupedParts) {
    return fail("staging contiguous mask layout requires grouped result parts");
  }

  int64_t groups = resultTypes.size() / factor;
  bool tooManySourceParts =
      sourceParts.size() > static_cast<size_t>(groups * factor);
  if (tooManySourceParts) {
    return fail("staging contiguous mask layout has too many source parts");
  }

  StagingMaskPartAccumulator accumulator(factor, groups);

  for (int64_t i = 0; i < groups; ++i) {
    FailureOr<SmallVector<Value, 4>> materialized =
        materializeContiguousToDeintMaskGroup(
            op, sourceParts, resultTypes, factor, groups, i, rewriter);
    if (failed(materialized)) {
      return failure();
    }
    if (failed(accumulator.append(op, *materialized, rewriter))) {
      return failure();
    }
  }

  return accumulator.flatten(op, resultTypes, rewriter);
}

FailureOr<SmallVector<Value>> materializeMaskGranularityCastLayoutConversion(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, TypeRange resultTypes, PatternRewriter &rewriter);

FailureOr<SmallVector<Value>>
materializeMaskGranularityCastLayoutConversionViaContiguous(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, TypeRange resultTypes, PatternRewriter &rewriter);

FailureOr<std::optional<SmallVector<Value>>>
materializeMaskGranularityCastStagingLayout(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, TypeRange resultTypes, PatternRewriter &rewriter);

static FailureOr<SmallVector<Value>> forwardIdentityMaskParts(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    PatternRewriter &rewriter) {
  if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                          rewriter))) {
    return failure();
  }
  return SmallVector<Value>(sourceParts.begin(), sourceParts.end());
}

static bool requiresMaskDenseSplitFallback(VMILayoutAttr sourceLayout,
                                           VMILayoutAttr resultLayout) {
  return sourceLayout.isDenseSplit() || resultLayout.isDenseSplit();
}

FailureOr<std::optional<SmallVector<Value>>>
materializeMaskGranularityCastLayoutFallback(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, TypeRange resultTypes, VMILayoutAttr sourceLayout,
    VMILayoutAttr resultLayout, PatternRewriter &rewriter) {
  FailureOr<SmallVector<Value>> layoutParts = materializeMaskLayoutConversion(
      op, sourceParts, resultTypes, sourceLayout, resultLayout, rewriter);
  if (succeeded(layoutParts)) {
    return std::optional<SmallVector<Value>>(std::move(*layoutParts));
  }

  FailureOr<std::optional<SmallVector<Value>>> staging =
      materializeMaskGranularityCastStagingLayout(
          op, sourceType, resultType, sourceParts, resultTypes, rewriter);
  if (failed(staging)) {
    return failure();
  }
  if (staging->has_value()) {
    return std::move(*staging);
  }

  if (!requiresMaskDenseSplitFallback(sourceLayout, resultLayout)) {
    return std::optional<SmallVector<Value>>{};
  }
  FailureOr<SmallVector<Value>> contiguous =
      materializeMaskGranularityCastLayoutConversionViaContiguous(
          op, sourceType, resultType, sourceParts, resultTypes, rewriter);
  if (failed(contiguous)) {
    return failure();
  }
  return std::optional<SmallVector<Value>>(std::move(*contiguous));
}

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
  if (failed(contiguousTypes)) {
    return failure();
  }
  FailureOr<SmallVector<Value>> contiguousParts =
      materializeMaskGranularityCastLayoutConversion(
          op, sourceType, contiguousType, sourceParts, *contiguousTypes,
          rewriter);
  if (failed(contiguousParts)) {
    return failure();
  }
  return materializeMaskGranularityCastLayoutConversion(
      op, contiguousType, resultType, *contiguousParts, resultTypes, rewriter);
}

static std::optional<bool> getMaskStagingDirection(
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout, int64_t factor) {
  bool sourceContiguous = sourceLayout && sourceLayout.isContiguous() &&
                          sourceLayout.getLaneStride() == 1;
  bool resultContiguous = resultLayout && resultLayout.isContiguous() &&
                          resultLayout.getLaneStride() == 1;
  bool sourceDeinterleaved =
      isElementDeinterleavedLayout(sourceLayout, factor) && resultContiguous;
  bool resultDeinterleaved =
      sourceContiguous && isElementDeinterleavedLayout(resultLayout, factor);
  if (!sourceDeinterleaved && !resultDeinterleaved) {
    return std::nullopt;
  }
  return sourceDeinterleaved;
}

FailureOr<std::optional<SmallVector<Value>>>
materializeMaskGranularityCastStagingLayout(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, TypeRange resultTypes, PatternRewriter &rewriter) {
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  for (int64_t factor : {2L, 4L}) {
    std::optional<bool> sourceIsDeinterleaved =
        getMaskStagingDirection(sourceLayout, resultLayout, factor);
    if (!sourceIsDeinterleaved) {
      continue;
    }
    FailureOr<SmallVector<Value>> materialized =
        *sourceIsDeinterleaved
            ? materializeStagingDeintToContiguousMaskLayout(
                  op, sourceParts, resultTypes, factor, rewriter)
            : materializeStagingContiguousToDeintMaskLayout(
                  op, sourceParts, resultTypes, factor, rewriter);
    if (failed(materialized)) {
      return failure();
    }
    return std::optional<SmallVector<Value>>(std::move(*materialized));
  }
  return std::optional<SmallVector<Value>>{};
}

FailureOr<SmallVector<Value>> materializeMaskGranularityCastLayoutConversion(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, TypeRange resultTypes, PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message) -> FailureOr<SmallVector<Value>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };

  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  bool hasLayouts = sourceLayout && resultLayout;
  if (!hasLayouts) {
    return fail("mask granularity cast layout conversion requires layouts");
  }

  bool identityLayout = sourceLayout == resultLayout;
  if (identityLayout) {
    return forwardIdentityMaskParts(op, sourceParts, resultTypes, rewriter);
  }

  FailureOr<std::optional<SmallVector<Value>>> fallback =
      materializeMaskGranularityCastLayoutFallback(
          op, sourceType, resultType, sourceParts, resultTypes, sourceLayout,
          resultLayout, rewriter);
  if (failed(fallback)) {
    return failure();
  }
  if (fallback->has_value()) {
    return std::move(**fallback);
  }

  return fail("unsupported mask granularity cast layout conversion");
}

struct MaskGranularityCastPlan {
  VMIMaskType physicalSourceType;
  VMIMaskType physicalResultType;
};

static FailureOr<SmallVector<Value>> materializeMaskGranularityCastThroughLayout(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, TypeRange resultTypes,
    const MaskGranularityCastPlan &plan, PatternRewriter &rewriter) {
  VMIMaskType granularityType = VMIMaskType::get(
      op->getContext(), sourceType.getElementCount(),
      plan.physicalResultType.getGranularity(),
      plan.physicalSourceType.getLayoutAttr());
  FailureOr<SmallVector<Value>> granularityParts =
      materializeMaskGranularityConversion(
          op, plan.physicalSourceType, granularityType, sourceParts, rewriter);
  if (failed(granularityParts)) {
    return failure();
  }
  return materializeMaskGranularityCastLayoutConversion(
      op, granularityType, plan.physicalResultType, *granularityParts,
      resultTypes, rewriter);
}

static FailureOr<SmallVector<Value>> materializeMaskGranularityCastParts(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, TypeRange resultTypes,
    const MaskGranularityCastPlan &plan, PatternRewriter &rewriter) {
  bool samePhysicalLayout =
      plan.physicalSourceType.getLayoutAttr() ==
      plan.physicalResultType.getLayoutAttr();
  if (samePhysicalLayout) {
    return materializeMaskGranularityConversion(
        op, plan.physicalSourceType, plan.physicalResultType, sourceParts,
        rewriter);
  }

  return materializeMaskGranularityCastThroughLayout(
      op, sourceType, resultType, sourceParts, resultTypes, plan, rewriter);
}

static FailureOr<MaskGranularityCastPlan> buildMaskGranularityCastPlan(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    PatternRewriter &rewriter) {
  auto fail = [&op, &rewriter](const Twine &message)
      -> FailureOr<MaskGranularityCastPlan> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };
  bool laneCountMismatch =
      sourceType.getElementCount() != resultType.getElementCount();
  if (laneCountMismatch) {
    return fail("requires source and result mask lane counts to match");
  }
  FailureOr<VMIMaskType> physicalSourceType =
      getVMIMaskPhysicalCarrierType(sourceType);
  FailureOr<VMIMaskType> physicalResultType =
      getVMIMaskPhysicalCarrierType(resultType);
  bool missingCarrierType =
      failed(physicalSourceType) || failed(physicalResultType);
  if (missingCarrierType) {
    return fail("requires source/result mask physical carrier types");
  }
  return MaskGranularityCastPlan{*physicalSourceType, *physicalResultType};
}

FailureOr<SmallVector<Value>> materializeMaskGranularityCastConversion(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, TypeRange resultTypes, PatternRewriter &rewriter) {
  FailureOr<MaskGranularityCastPlan> plan =
      buildMaskGranularityCastPlan(op, sourceType, resultType, rewriter);
  if (failed(plan)) {
    return failure();
  }

  if (plan->physicalSourceType == plan->physicalResultType) {
    FailureOr<SmallVector<Value>> identity =
        forwardIdentityMaskParts(op, sourceParts, resultTypes, rewriter);
    if (failed(identity)) {
      return failure();
    }
    return std::move(*identity);
  }

  return materializeMaskGranularityCastParts(
      op, sourceType, resultType, sourceParts, resultTypes, *plan, rewriter);
}

struct OneToNVMIEnsureLayoutOpPattern
    : OneToNOpConversionPattern<VMIEnsureLayoutOp> {
  using OneToNOpConversionPattern<VMIEnsureLayoutOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIEnsureLayoutOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto sourceType = cast<VMIVRegType>(op.getSource().getType());
    auto resultType = cast<VMIVRegType>(op.getResult().getType());
    return lowerMaterializedResults(
        op, *this->getTypeConverter(), rewriter,
        [&]() -> FailureOr<SmallVector<Value>> {
          return materializeEnsureLayoutConversion(
              op, adaptor.getSource(), sourceType, resultType,
              *this->getTypeConverter(), rewriter);
        });
  }
};

struct OneToNVMIEnsureMaskLayoutOpPattern
    : OneToNOpConversionPattern<VMIEnsureMaskLayoutOp> {
  using OneToNOpConversionPattern<
      VMIEnsureMaskLayoutOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIEnsureMaskLayoutOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto sourceType = cast<VMIMaskType>(op.getSource().getType());
    auto resultType = cast<VMIMaskType>(op.getResult().getType());
    VMILayoutSupport supports;
    std::string supportReason;
    if (failed(supports.getEnsureMaskLayoutFact(sourceType, resultType,
                                                &supportReason))) {
      return rewriter.notifyMatchFailure(
          op, Twine("ensure_mask_layout has no registered materialization "
                    "support: ") +
                  supportReason);
    }
    bool granularityMismatch =
        sourceType.getGranularity() != resultType.getGranularity();
    if (granularityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "mask layout helper cannot also change granularity");
    }
    VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
    VMILayoutAttr resultLayout = resultType.getLayoutAttr();

    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    FailureOr<SmallVector<Value>> results = materializeMaskLayoutConversion(
        op, sourceParts, resultTypes, sourceLayout, resultLayout, rewriter);
    if (failed(results)) {
      return failure();
    }
    return replacePhysicalResults(rewriter, op, *results,
                                  *this->getTypeConverter());
  }
};

struct OneToNVMIEnsureMaskGranularityOpPattern
    : OneToNOpConversionPattern<VMIEnsureMaskGranularityOp> {
  using OneToNOpConversionPattern<
      VMIEnsureMaskGranularityOp>::OneToNOpConversionPattern;

private:
  LogicalResult replaceCheckedResults(
      VMIEnsureMaskGranularityOp op, OneToNPatternRewriter &rewriter,
      FailureOr<SmallVector<Value>> results, ArrayRef<Type> resultTypes) const {
    if (failed(results)) {
      return failure();
    }
    bool resultArityMismatch = results->size() != resultTypes.size();
    if (resultArityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "mask granularity cast result arity mismatch");
    }
    for (auto [result, type] : llvm::zip_equal(*results, resultTypes)) {
      bool resultTypeMismatch = result.getType() != type;
      if (resultTypeMismatch) {
        return rewriter.notifyMatchFailure(
            op, "mask granularity cast result type mismatch");
      }
    }
    replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                     *this->getTypeConverter());
    return success();
  }

public:

  LogicalResult
  matchAndRewrite(VMIEnsureMaskGranularityOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto sourceType = cast<VMIMaskType>(op.getSource().getType());
    auto resultType = cast<VMIMaskType>(op.getResult().getType());
    VMILayoutSupport supports;
    bool identity = sourceType.getGranularity() == resultType.getGranularity() &&
                    sourceType.getLayoutAttr() == resultType.getLayoutAttr();
    if (!identity) {
      std::string reason;
      if (failed(supports.getMaskGranularityCastLayoutFactForLayouts(
              sourceType, resultType, sourceType.getLayoutAttr(),
              resultType.getLayoutAttr(), &reason))) {
        return rewriter.notifyMatchFailure(
            op, "unsupported mask granularity cast layout relation: " + reason);
      }
    }

    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);

    FailureOr<SmallVector<Value>> results =
        materializeMaskGranularityCastConversion(
            op, sourceType, resultType, sourceParts, resultTypes, rewriter);
    return replaceCheckedResults(op, rewriter, std::move(results), resultTypes);
  }

};

struct OneToNVMIBroadcastOpPattern : OneToNOpConversionPattern<VMIBroadcastOp> {
  using OneToNOpConversionPattern<VMIBroadcastOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIBroadcastOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange inputParts = adaptor.getValue();
    bool invalidInputArity = inputParts.size() != 1;
    if (invalidInputArity) {
      return rewriter.notifyMatchFailure(
          op, "broadcast input must convert to one value");
    }
    bool inputIsVReg = isa<VMIVRegType>(op.getValue().getType());

    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto vregType = dyn_cast<VRegType>(resultType);
      if (!vregType) {
        return rewriter.notifyMatchFailure(op, "broadcast result must be vreg");
      }
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
      if (failed(mask)) {
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for broadcast mask");
      }
      StringAttr position =
          inputIsVReg ? rewriter.getStringAttr("LOWEST") : StringAttr{};
      results.push_back(rewriter
                            .create<VdupOp>(op.getLoc(), resultType,
                                            inputParts.front(), *mask, position)
                            .getResult());
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
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
  if (laneOffset == 0) {
    return base;
  }

  FailureOr<Value> offset =
      createScalarOffsetConstant(loc, base.getType(), laneOffset, rewriter);
  if (failed(offset)) {
    return failure();
  }

  if (isa<IntegerType>(base.getType())) {
    if (order == "DESC") {
      return rewriter.create<arith::SubIOp>(loc, base, *offset).getResult();
    }
    return rewriter.create<arith::AddIOp>(loc, base, *offset).getResult();
  }
  if (isa<FloatType>(base.getType())) {
    if (order == "DESC") {
      return rewriter.create<arith::SubFOp>(loc, base, *offset).getResult();
    }
    return rewriter.create<arith::AddFOp>(loc, base, *offset).getResult();
  }

  return failure();
}

struct IotaMaterializationContext {
  Location loc;
  Value base;
  StringAttr orderAttr;
  PatternRewriter &rewriter;
};

static StringRef getIotaOrder(const IotaMaterializationContext &context) {
  return context.orderAttr ? context.orderAttr.getValue() : StringRef("ASC");
}

FailureOr<Value> createIotaContiguousChunk(
    const IotaMaterializationContext &context, Type resultType,
    int64_t laneOffset) {
  // Contiguous iota is a direct VCI of the absolute chunk base (ASC
  // `vci(offset_sreg)`). Group-periodic VL128 {group=2} shares one such VL64
  // result across both physical parts (see sharedChunks below).
  StringRef order = getIotaOrder(context);
  FailureOr<Value> chunkBase =
      createIotaChunkBase(context.loc, context.base, laneOffset, order,
                          context.rewriter);
  if (failed(chunkBase)) {
    return failure();
  }
  return context.rewriter
      .create<VciOp>(context.loc, resultType, *chunkBase, context.orderAttr)
      .getResult();
}

FailureOr<std::optional<Value>> createPowerOfTwoSubVLChunk(
    Location loc, Type resultType, Value base, int64_t groupSize,
    StringRef order, Value allMask, PatternRewriter &rewriter) {
  bool unsupportedShape =
      !llvm::isPowerOf2_64(static_cast<uint64_t>(groupSize)) ||
      !isa<IntegerType>(base.getType());
  if (unsupportedShape) {
    return std::optional<Value>{};
  }

  FailureOr<Value> zeroScalar =
      createScalarOffsetConstant(loc, base.getType(), 0, rewriter);
  FailureOr<Value> maskScalar = createScalarOffsetConstant(
      loc, base.getType(), groupSize - 1, rewriter);
  bool failedScalars = failed(zeroScalar) || failed(maskScalar);
  if (failedScalars) {
    return failure();
  }

  Value laneIds =
      rewriter.create<VciOp>(loc, resultType, *zeroScalar, StringAttr{})
          .getResult();
  Value maskVec =
      rewriter
          .create<VdupOp>(loc, resultType, *maskScalar, allMask,
                          /*position=*/nullptr)
          .getResult();
  Value rem = rewriter
                  .create<VandOp>(loc, resultType, laneIds, maskVec, allMask)
                  .getResult();
  if (order == "DESC") {
    Value baseVec =
        rewriter
            .create<VdupOp>(loc, resultType, base, allMask,
                            /*position=*/nullptr)
            .getResult();
    return std::optional<Value>(
        rewriter.create<VsubOp>(loc, resultType, baseVec, rem, allMask)
            .getResult());
  }
  return std::optional<Value>(
      rewriter.create<VaddsOp>(loc, resultType, rem, base, allMask).getResult());
}

/// Pack group-periodic ramps inside one physical VL when S < physVL and
/// physVL % S == 0 (e.g. i32 L=64,group=2 → [base..base+31 | base..base+31]).
///
/// Preferred recipes (O(1), independent of G = physVL/S):
///   * S == 1 → vdup(base)
///   * S power-of-2 integer (all legal sub-VL S on this ISA) →
///       ASC:  vadds(vand(vci(0), S-1), base)
///       DESC: vsub(vdup(base), vand(vci(0), S-1))
///
/// Residual fallback (non-integer base): per-group vci(base) ∓ g*S +
/// lane-range vsel. Index iota is integer-only in practice.
///
/// When S == physVL this is just `vci(base)` (single group fills the VL).
static FailureOr<Value> materializeResidualSubVLGroup(
    Location loc, Type resultType, Value base, StringRef order, Value full,
    MaskType maskType, Value zeroScalar, Value allMask, Value previousResult,
    int64_t groupSize, int64_t localGroup, PatternRewriter &rewriter) {
  Value adjusted = full;
  if (localGroup != 0) {
    int64_t delta = localGroup * groupSize;
    FailureOr<Value> offsetScalar =
        createScalarOffsetConstant(loc, base.getType(), delta, rewriter);
    if (failed(offsetScalar)) {
      return failure();
    }
    if (order == "DESC") {
      adjusted = rewriter
                     .create<VaddsOp>(loc, resultType, full, *offsetScalar,
                                      allMask)
                     .getResult();
    } else {
      Value negOffset = isa<FloatType>(base.getType())
                            ? rewriter.create<arith::NegFOp>(
                                  loc, *offsetScalar)
                                  .getResult()
                            : rewriter
                                  .create<arith::SubIOp>(loc, zeroScalar,
                                                         *offsetScalar)
                                  .getResult();
      adjusted = rewriter
                     .create<VaddsOp>(loc, resultType, full, negOffset, allMask)
                     .getResult();
    }
  }
  FailureOr<Value> laneMask = createLaneRangeMask(
      loc, maskType, localGroup * groupSize, (localGroup + 1) * groupSize,
      rewriter);
  if (failed(laneMask)) {
    return failure();
  }
  return rewriter
      .create<VselOp>(loc, resultType, adjusted, previousResult, *laneMask)
      .getResult();
}

FailureOr<Value> createResidualSubVLGroupPeriodicChunk(
    Location loc, Type resultType, Value base, StringRef order,
    Value full, MaskType maskType, Value zeroScalar, Value allMask,
    int64_t groupSize,
    int64_t groupsPerChunk, PatternRewriter &rewriter) {
  Value result =
      rewriter
          .create<VdupOp>(loc, resultType, zeroScalar,
                          allMask,
                          /*position=*/nullptr)
          .getResult();
  for (int64_t localGroup = 0; localGroup < groupsPerChunk; ++localGroup) {
    FailureOr<Value> nextResult = materializeResidualSubVLGroup(
        loc, resultType, base, order, full, maskType, zeroScalar, allMask,
        result, groupSize, localGroup, rewriter);
    if (failed(nextResult)) {
      return failure();
    }
    result = *nextResult;
  }
  return result;
}

static FailureOr<std::optional<Value>> createSubVLPeriodicFastPath(
    const IotaMaterializationContext &context, Type resultType,
    int64_t groupSize, StringRef order, Value allMask) {
  Location loc = context.loc;
  Value base = context.base;
  PatternRewriter &rewriter = context.rewriter;
  if (groupSize == 1) {
    return std::optional<Value>(
        rewriter
            .create<VdupOp>(loc, resultType, base, allMask,
                            /*position=*/nullptr)
            .getResult());
  }

  auto vregType = dyn_cast<VRegType>(resultType);
  if (!vregType) {
    return failure();
  }
  int64_t groupsPerChunk = vregType.getElementCount() / groupSize;
  if (groupsPerChunk == 1) {
    FailureOr<Value> result =
        createIotaContiguousChunk(context, resultType, /*laneOffset=*/0);
    if (failed(result)) {
      return failure();
    }
    return std::optional<Value>(*result);
  }

  FailureOr<std::optional<Value>> powerOfTwo = createPowerOfTwoSubVLChunk(
      loc, resultType, base, groupSize, order, allMask, rewriter);
  if (failed(powerOfTwo)) {
    return failure();
  }
  return *powerOfTwo;
}

FailureOr<Value> createSubVLGroupPeriodicChunk(
    const IotaMaterializationContext &context, Type resultType,
    int64_t groupSize) {
  Location loc = context.loc;
  Value base = context.base;
  PatternRewriter &rewriter = context.rewriter;
  auto vregType = dyn_cast<VRegType>(resultType);
  if (!vregType) {
    return failure();
  }

  int64_t lanesPerPart = vregType.getElementCount();
  if (groupSize <= 0 || lanesPerPart % groupSize != 0) {
    return failure();
  }

  FailureOr<Value> allMask =
      createAllTrueMaskForVReg(loc, vregType, rewriter);
  if (failed(allMask)) {
    return failure();
  }

  StringRef order = getIotaOrder(context);
  FailureOr<std::optional<Value>> fastPath = createSubVLPeriodicFastPath(
      context, resultType, groupSize, order, *allMask);
  if (failed(fastPath)) {
    return failure();
  }
  if (fastPath->has_value()) {
    return **fastPath;
  }

  int64_t groupsPerChunk = lanesPerPart / groupSize;
  FailureOr<Value> full =
      createIotaContiguousChunk(context, resultType, /*laneOffset=*/0);
  FailureOr<MaskType> maskType =
      getMaskTypeForVReg(vregType, rewriter.getContext());
  FailureOr<Value> zeroScalar =
      createScalarOffsetConstant(loc, base.getType(), 0, rewriter);
  bool failedResidualInputs =
      failed(full) || failed(maskType) || failed(zeroScalar);
  if (failedResidualInputs) {
    return failure();
  }

  return createResidualSubVLGroupPeriodicChunk(
      loc, resultType, base, order, *full, *maskType, *zeroScalar, *allMask,
      groupSize, groupsPerChunk, rewriter);
}

FailureOr<Value> createIotaDeinterleavedChunk(
    const IotaMaterializationContext &context, Type resultType, int64_t factor,
    int64_t part, int64_t chunk, int64_t lanesPerPart) {
  Location loc = context.loc;
  Value base = context.base;
  StringAttr orderAttr = context.orderAttr;
  PatternRewriter &rewriter = context.rewriter;
  auto vregType = dyn_cast<VRegType>(resultType);
  if (!vregType) {
    return failure();
  }

  FailureOr<Value> mask = createAllTrueMaskForVReg(loc, vregType, rewriter);
  FailureOr<Value> zero =
      createScalarOffsetConstant(loc, base.getType(), 0, rewriter);
  FailureOr<Value> factorScalar =
      createScalarOffsetConstant(loc, base.getType(), factor, rewriter);
  bool failedIotaInputs = failed(mask) || failed(zero) || failed(factorScalar);
  if (failedIotaInputs) {
    return failure();
  }

  Value local =
      rewriter.create<VciOp>(loc, resultType, *zero, StringAttr{}).getResult();
  Value scaled =
      rewriter.create<VmulsOp>(loc, resultType, local, *factorScalar, *mask)
          .getResult();

  StringRef order = orderAttr ? orderAttr.getValue() : StringRef("ASC");
  int64_t partOffset = part + factor * chunk * lanesPerPart;
  FailureOr<Value> biasedBase =
      createIotaChunkBase(loc, base, partOffset, order, rewriter);
  if (failed(biasedBase)) {
    return failure();
  }

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

template <typename IotaOp>
struct OneToNVMIIotaOpPattern : OneToNOpConversionPattern<IotaOp> {
  using OneToNOpConversionPattern<IotaOp>::OneToNOpConversionPattern;
  using OpAdaptor =
      typename OneToNOpConversionPattern<IotaOp>::OpAdaptor;

private:
  struct IotaLoweringInput {
    VMIVRegType resultVMIType;
    VMILayoutAttr layout;
    Value base;
    SmallVector<Type> resultTypes;
    int64_t lanesPerPart;
  };

  FailureOr<IotaLoweringInput> getIotaLoweringInput(
      IotaOp op, OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    VMILayoutAttr layout = resultVMIType.getLayoutAttr();
    if (!layout) {
      return rewriter.notifyMatchFailure(op, "iota requires assigned layout");
    }
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(resultVMIType.getElementType());
    if (failed(lanesPerPart)) {
      return rewriter.notifyMatchFailure(
          op, "iota requires known physical lanes per part");
    }
    FailureOr<Value> base = getSingleValue(
        op, adaptor.getBase(), "iota base must convert to one value", rewriter);
    if (failed(base)) {
      return failure();
    }
    FailureOr<SmallVector<Type>> resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(resultTypes)) {
      return failure();
    }
    return IotaLoweringInput{resultVMIType, layout, *base,
                             std::move(*resultTypes), *lanesPerPart};
  }

  FailureOr<std::pair<int64_t, int64_t>> validateGroupedIotaShape(
      IotaOp op, VMIVRegType resultVMIType, VMILayoutAttr layout,
      TypeRange resultTypes, int64_t lanesPerPart,
      OneToNPatternRewriter &rewriter) const {
    if (lanesPerPart <= 0) {
      return rewriter.notifyMatchFailure(
          op, "grouped iota requires positive physical lanes per part");
    }
    int64_t numGroups = op.getGroupAttr().getInt();
    int64_t logicalLanes = resultVMIType.getElementCount();
    if (numGroups <= 0) {
      return rewriter.notifyMatchFailure(
          op, "grouped iota requires positive group count");
    }
    int64_t safeNumGroups = numGroups > 0 ? numGroups : 1;
    if (logicalLanes % safeNumGroups != 0) {
      return rewriter.notifyMatchFailure(
          op, "grouped iota requires group to divide logical lane count");
    }
    int64_t groupSize = logicalLanes / safeNumGroups;
    int64_t safeLanesPerPart = lanesPerPart > 0 ? lanesPerPart : 1;
    int64_t safeGroupSize = groupSize > 0 ? groupSize : 1;
    bool compatibleShape = groupSize % safeLanesPerPart == 0 ||
                           lanesPerPart % safeGroupSize == 0;
    if (!compatibleShape) {
      return rewriter.notifyMatchFailure(
          op, "grouped iota requires group_size to divide or be a multiple of physical lanes per part");
    }
    if (!layout.isContiguous()) {
      return rewriter.notifyMatchFailure(
          op, "grouped iota currently supports contiguous layout only; ensure_layout to contiguous before lowering");
    }
    int64_t expectedArity =
        (logicalLanes + safeLanesPerPart - 1) / safeLanesPerPart;
    bool resultArityMismatch =
        static_cast<int64_t>(resultTypes.size()) != expectedArity;
    if (resultArityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "grouped contiguous iota physical result count mismatch");
    }
    return std::make_pair(groupSize, expectedArity);
  }

  LogicalResult lowerGroupedIota(
      IotaOp op, Value base, VMIVRegType resultVMIType,
      VMILayoutAttr layout, TypeRange resultTypes, int64_t lanesPerPart,
      OneToNPatternRewriter &rewriter, SmallVectorImpl<Value> &results) const {
    FailureOr<std::pair<int64_t, int64_t>> shape = validateGroupedIotaShape(
        op, resultVMIType, layout, resultTypes, lanesPerPart, rewriter);
    if (failed(shape)) {
      return failure();
    }
    int64_t groupSize = shape->first;
    int64_t safeLanesPerPart = lanesPerPart > 0 ? lanesPerPart : 1;
    int64_t safeGroupSize = groupSize > 0 ? groupSize : 1;
    bool groupSizeMultipleOfPhys = groupSize % safeLanesPerPart == 0;
    bool physMultipleOfGroupSize = lanesPerPart % safeGroupSize == 0;

    llvm::DenseMap<std::pair<Type, int64_t>, Value> sharedChunks;
    IotaMaterializationContext context{op.getLoc(), base, op.getOrderAttr(),
                                       rewriter};
    for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
      if (!isa<VRegType>(resultType)) {
        return rewriter.notifyMatchFailure(op, "iota result must be vreg");
      }
      int64_t laneOffset = 0;
      if (groupSizeMultipleOfPhys) {
        laneOffset = (static_cast<int64_t>(index) * lanesPerPart) % groupSize;
      }
      auto key = std::make_pair(resultType, laneOffset);
      auto it = sharedChunks.find(key);
      if (it == sharedChunks.end()) {
        FailureOr<Value> result;
        if (physMultipleOfGroupSize && groupSize < lanesPerPart) {
          result = createSubVLGroupPeriodicChunk(context, resultType,
                                                 groupSize);
        } else {
          result = createIotaContiguousChunk(context, resultType, laneOffset);
        }
        if (failed(result)) {
          return rewriter.notifyMatchFailure(
              op, "failed to materialize grouped iota chunk");
        }
        it = sharedChunks.try_emplace(key, *result).first;
      }
      results.push_back(it->second);
    }
    return success();
  }

  LogicalResult lowerContiguousIota(
      IotaOp op, Value base, TypeRange resultTypes, int64_t lanesPerPart,
      OneToNPatternRewriter &rewriter, SmallVectorImpl<Value> &results) const {
    IotaMaterializationContext context{op.getLoc(), base, op.getOrderAttr(),
                                       rewriter};
    for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
      if (!isa<VRegType>(resultType)) {
        return rewriter.notifyMatchFailure(op, "iota result must be vreg");
      }
      FailureOr<Value> result = createIotaContiguousChunk(
          context, resultType, static_cast<int64_t>(index) * lanesPerPart);
      if (failed(result)) {
        return rewriter.notifyMatchFailure(
            op, "failed to materialize contiguous iota chunk");
      }
      results.push_back(*result);
    }
    return success();
  }

  LogicalResult lowerDeinterleavedIota(
      IotaOp op, Value base, VMILayoutAttr layout, TypeRange resultTypes,
      int64_t lanesPerPart, OneToNPatternRewriter &rewriter,
      SmallVectorImpl<Value> &results) const {
    int64_t factor = layout.getFactor();
    int64_t safeFactor = factor > 0 ? factor : 1;
    bool resultFactorMismatch = resultTypes.size() % safeFactor != 0;
    if (resultFactorMismatch) {
      return rewriter.notifyMatchFailure(
          op, "deinterleaved iota physical result count does not match "
              "layout factor");
    }
    int64_t chunksPerPart = resultTypes.size() / safeFactor;
    IotaMaterializationContext context{op.getLoc(), base, op.getOrderAttr(),
                                       rewriter};
    for (int64_t part = 0; part < factor; ++part) {
      for (int64_t chunk = 0; chunk < chunksPerPart; ++chunk) {
        Type resultType = resultTypes[part * chunksPerPart + chunk];
        FailureOr<Value> result = createIotaDeinterleavedChunk(
            context, resultType, factor, part, chunk, lanesPerPart);
        if (failed(result)) {
          return rewriter.notifyMatchFailure(
              op, "failed to materialize deinterleaved iota chunk");
        }
        results.push_back(*result);
      }
    }
    return success();
  }

  LogicalResult lowerAndReplaceIota(
      IotaOp op, Value base, VMIVRegType resultVMIType,
      VMILayoutAttr layout, TypeRange resultTypes, int64_t lanesPerPart,
      OneToNPatternRewriter &rewriter,
      SmallVectorImpl<Value> &results) const {
    if constexpr (std::is_same_v<IotaOp, VMIGroupIotaOp>) {
      if (failed(lowerGroupedIota(op, base, resultVMIType, layout, resultTypes,
                                  lanesPerPart, rewriter, results))) {
        return failure();
      }
    } else if (layout.isContiguous()) {
      if (failed(lowerContiguousIota(op, base, resultTypes, lanesPerPart,
                                     rewriter, results))) {
        return failure();
      }
    } else if (failed(lowerDeinterleavedIota(
                   op, base, layout, resultTypes, lanesPerPart, rewriter,
                   results))) {
      return failure();
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

public:

  LogicalResult
  matchAndRewrite(IotaOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<IotaLoweringInput> input =
        getIotaLoweringInput(op, adaptor, rewriter);
    if (failed(input)) {
      return failure();
    }
    SmallVector<Value> results;
    results.reserve(input->resultTypes.size());
    return lowerAndReplaceIota(op, input->base, input->resultVMIType,
                               input->layout, input->resultTypes,
                               input->lanesPerPart, rewriter, results);
  }
};

struct OneToNVMIConstantOpPattern : OneToNOpConversionPattern<VMIConstantOp> {
  using OneToNOpConversionPattern<VMIConstantOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerSplat(VMIConstantOp op, TypedAttr splatAttr,
                           ArrayRef<Type> resultTypes,
                           OneToNPatternRewriter &rewriter) const {
    Value scalar =
        rewriter.create<arith::ConstantOp>(op.getLoc(), splatAttr).getResult();
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto vregType = dyn_cast<VRegType>(resultType);
      if (!vregType) {
        return rewriter.notifyMatchFailure(op, "constant result must be vreg");
      }
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
      if (failed(mask)) {
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for constant mask");
      }
      results.push_back(
          rewriter
              .create<VdupOp>(op.getLoc(), resultType, scalar, *mask,
                              /*position=*/nullptr)
              .getResult());
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

public:

  LogicalResult
  matchAndRewrite(VMIConstantOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!denseAttr || !denseAttr.isSplat()) {
      return rewriter.notifyMatchFailure(
          op, "only splat dense data constants are supported");
    }
    auto splatAttr = dyn_cast<TypedAttr>(denseAttr.getSplatValue<Attribute>());
    if (!splatAttr) {
      return rewriter.notifyMatchFailure(op, "splat constant must be typed");
    }

    // arith.constant only accepts signless integer types, whereas VMI vregs may
    // carry signed/unsigned element types (e.g. ui16). Remap an unsigned/signed
    // integer splat to its signless equivalent; the downstream pto.vdup accepts
    // a signless scalar for a signed/unsigned result element.
    if (auto intAttr = dyn_cast<IntegerAttr>(splatAttr)) {
      if (auto intTy = dyn_cast<IntegerType>(intAttr.getType());
          intTy && !intTy.isSignless()) {
        splatAttr = IntegerAttr::get(rewriter.getIntegerType(intTy.getWidth()),
                                     intAttr.getValue());
      }
    }

    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    return lowerSplat(op, splatAttr, resultTypes, rewriter);
  }
};

struct OneToNVMIConstantMaskOpPattern
    : OneToNOpConversionPattern<VMIConstantMaskOp> {
  using OneToNOpConversionPattern<VMIConstantMaskOp>::OneToNOpConversionPattern;

private:
  FailureOr<SmallVector<Value>> materializePhysicalMasks(
      VMIConstantMaskOp op, ArrayRef<Type> resultTypes,
      ArrayRef<ConstantMaskChunkMaterialization> materializations,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (const ConstantMaskChunkMaterialization &materialization :
         materializations) {
      bool tooManyMasks = results.size() >= resultTypes.size();
      if (tooManyMasks) {
        return rewriter.notifyMatchFailure(
            op, "constant_mask produced too many physical masks");
      }
      auto maskType = dyn_cast<MaskType>(resultTypes[results.size()]);
      if (!maskType) {
        return rewriter.notifyMatchFailure(op,
                                           "constant_mask result must be mask");
      }
      FailureOr<Value> mask = materializeConstantMaskChunk(
          op.getLoc(), maskType, materialization.activeLanes, rewriter);
      if (failed(mask)) {
        return rewriter.notifyMatchFailure(
            op, "failed to materialize constant_mask physical chunk");
      }
      results.push_back(*mask);
    }
    bool resultArityMismatch = results.size() != resultTypes.size();
    if (resultArityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "constant_mask physical result count mismatch");
    }
    return results;
  }

public:

  LogicalResult
  matchAndRewrite(VMIConstantMaskOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    bool failedResultTypeConversion = failed(maybe_resultTypes);
    if (failedResultTypeConversion) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    std::string reason;
    FailureOr<SmallVector<ConstantMaskChunkMaterialization>> materializations =
        computeConstantMaskMaterialization(op, &reason);
    if (failed(materializations)) {
      return rewriter.notifyMatchFailure(op, Twine("constant_mask ") + reason);
    }
    FailureOr<SmallVector<Value>> results = materializePhysicalMasks(
        op, resultTypes, *materializations, rewriter);
    if (failed(results)) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                     *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMICreateMaskOpPattern
    : OneToNOpConversionPattern<VMICreateMaskOp> {
  using OneToNOpConversionPattern<VMICreateMaskOp>::OneToNOpConversionPattern;

private:
  FailureOr<SmallVector<Type>> getResultTypes(VMICreateMaskOp op) const {
    return getConvertedResultTypes(op, 0, *this->getTypeConverter());
  }

  LogicalResult lowerDynamicCreateMask(
      VMICreateMaskOp op, OpAdaptor adaptor, VMIMaskType resultVMIType,
      VMILayoutAttr layout, int64_t lanesPerPart,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<Value> active = getSingleValue(
        op, adaptor.getActiveLanes(),
        "create_mask active_lanes must convert to one value", rewriter);
    if (failed(active)) {
      return failure();
    }
    FailureOr<SmallVector<Type>> maybeResultTypes = getResultTypes(op);
    if (failed(maybeResultTypes)) {
      return failure();
    }
    SmallVector<Value> results;
    if (failed(lowerDynamicMask(op, *active, resultVMIType, layout,
                                *maybeResultTypes, lanesPerPart, rewriter,
                                results))) {
      return failure();
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  LogicalResult lowerConstantCreateMask(
      VMICreateMaskOp op, int64_t activeLanes, VMIMaskType resultVMIType,
      VMILayoutAttr layout, int64_t lanesPerPart,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<SmallVector<Type>> maybeResultTypes = getResultTypes(op);
    if (failed(maybeResultTypes)) {
      return failure();
    }
    SmallVector<Value> results;
    if (failed(lowerConstantMask(op, activeLanes, resultVMIType, layout,
                                 *maybeResultTypes, lanesPerPart, rewriter,
                                 results))) {
      return failure();
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  LogicalResult lowerDynamicMask(
      VMICreateMaskOp op, Value active, VMIMaskType resultVMIType,
      VMILayoutAttr layout, TypeRange resultTypes, int64_t lanesPerPart,
      OneToNPatternRewriter &rewriter,
      SmallVectorImpl<Value> &results) const {
    int64_t factor = layout.isDenseSplit() ? layout.getFactor() : 1;
    bool resultFactorMismatch = resultTypes.size() % factor != 0;
    if (resultFactorMismatch) {
      return rewriter.notifyMatchFailure(
          op, "dynamic create_mask physical result count does not match "
              "layout factor");
    }
    int64_t chunksPerPart = resultTypes.size() / factor;
    Value activeI32 = clampDynamicActiveLanes(
        op.getLoc(), active, resultVMIType.getElementCount(), rewriter);
    results.reserve(resultTypes.size());
    for (int64_t part = 0; part < factor; ++part) {
      Value remaining = createPartitionActiveLanes(op.getLoc(), activeI32,
                                                   factor, part, rewriter);
      for (int64_t chunk = 0; chunk < chunksPerPart; ++chunk) {
        Type resultType = resultTypes[part * chunksPerPart + chunk];
        FailureOr<std::pair<Value, Value>> maskAndRemaining =
            buildDynamicMaskChunk(op, resultType, remaining, rewriter);
        if (failed(maskAndRemaining)) {
          return failure();
        }
        results.push_back(maskAndRemaining->first);
        remaining = maskAndRemaining->second;
      }
    }
    return success();
  }

  FailureOr<std::pair<bool, int64_t>> getConstantMaskChunkActivity(
      VMICreateMaskOp op, VMIMaskType resultVMIType, int64_t part,
      int64_t chunk, int64_t activeLanes, int64_t lanesPerPart,
      OneToNPatternRewriter &rewriter) const {
    bool anyLane = false;
    int64_t activeInChunk = 0;
    for (int64_t lane = 0; lane < lanesPerPart; ++lane) {
      FailureOr<bool> padding =
          isPaddingLane(resultVMIType, part, chunk, lane);
      if (failed(padding)) {
        return rewriter.notifyMatchFailure(
            op, "failed to map create_mask physical padding lane");
      }
      if (*padding) {
        continue;
      }
      anyLane = true;
      FailureOr<int64_t> logicalLane =
          mapPhysicalLaneToLogical(resultVMIType, part, chunk, lane);
      if (failed(logicalLane)) {
        return rewriter.notifyMatchFailure(
            op, "failed to map create_mask physical lane");
      }
      if (*logicalLane < activeLanes) {
        ++activeInChunk;
      }
    }
    return std::make_pair(anyLane, activeInChunk);
  }

  FailureOr<std::pair<Value, Value>> buildDynamicMaskChunk(
      VMICreateMaskOp op, Type resultType, Value remaining,
      OneToNPatternRewriter &rewriter) const {
    auto maskType = dyn_cast<MaskType>(resultType);
    if (!maskType) {
      return rewriter.notifyMatchFailure(op, "create_mask result must be mask");
    }
    FailureOr<std::pair<Value, Value>> maskAndRemaining =
        createRuntimePrefixMask(op.getLoc(), maskType, remaining, rewriter);
    if (failed(maskAndRemaining)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported mask type for dynamic create_mask");
    }
    return *maskAndRemaining;
  }

  FailureOr<Value> materializeConstantMaskValue(
      VMICreateMaskOp op, Type resultType, int64_t activeInChunk,
      int64_t lanesPerPart, OneToNPatternRewriter &rewriter) const {
    auto maskType = dyn_cast<MaskType>(resultType);
    if (!maskType) {
      return rewriter.notifyMatchFailure(op, "create_mask result must be mask");
    }
    std::optional<std::string> pattern =
        getPrefixPattern(activeInChunk, lanesPerPart);
    if (pattern) {
      FailureOr<Value> mask =
          createPrefixMask(op.getLoc(), maskType, *pattern, rewriter);
      if (failed(mask)) {
        return rewriter.notifyMatchFailure(
            op, "unsupported mask type for create_mask");
      }
      return *mask;
    }
    FailureOr<std::pair<Value, Value>> maskAndRemaining =
        createRuntimePrefixMask(
            op.getLoc(), maskType,
            createI32Constant(op.getLoc(), activeInChunk, rewriter), rewriter);
    if (failed(maskAndRemaining)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported mask type for create_mask plt fallback");
    }
    return maskAndRemaining->first;
  }

  LogicalResult lowerConstantMask(
      VMICreateMaskOp op, int64_t activeLanes, VMIMaskType resultVMIType,
      VMILayoutAttr layout, TypeRange resultTypes, int64_t lanesPerPart,
      OneToNPatternRewriter &rewriter,
      SmallVectorImpl<Value> &results) const {
    int64_t factor = layout.isDenseSplit() ? layout.getFactor() : 1;
    results.reserve(resultTypes.size());
    for (int64_t part = 0; part < factor; ++part) {
      for (int64_t chunk = 0;; ++chunk) {
        FailureOr<std::pair<bool, int64_t>> activity =
            getConstantMaskChunkActivity(op, resultVMIType, part, chunk,
                                         activeLanes, lanesPerPart, rewriter);
        if (failed(activity)) {
          return failure();
        }
        bool anyLane = activity->first;
        int64_t activeInChunk = activity->second;
        if (!anyLane) {
          break;
        }
        bool tooManyResults = results.size() >= resultTypes.size();
        if (tooManyResults) {
          return rewriter.notifyMatchFailure(
              op, "create_mask produced too many physical masks");
        }
        FailureOr<Value> mask = materializeConstantMaskValue(
            op, resultTypes[results.size()], activeInChunk, lanesPerPart,
            rewriter);
        if (failed(mask)) {
          return failure();
        }
        results.push_back(*mask);
      }
    }
    bool resultArityMismatch = results.size() != resultTypes.size();
    if (resultArityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "create_mask physical result count mismatch");
    }
    return success();
  }

public:

  LogicalResult
  matchAndRewrite(VMICreateMaskOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto activeConstant =
        op.getActiveLanes().getDefiningOp<arith::ConstantOp>();
    auto resultVMIType = cast<VMIMaskType>(op.getResult().getType());
    VMILayoutAttr layout = resultVMIType.getLayoutAttr();
    if (!layout ||
        !VMIMaskType::isConcreteGranularity(resultVMIType.getGranularity())) {
      return rewriter.notifyMatchFailure(
          op, "create_mask requires concrete layout and granularity");
    }
    FailureOr<StringRef> physicalGranularity =
        getVMIMaskPhysicalGranularity(resultVMIType);
    FailureOr<int64_t> lanesPerPart =
        failed(physicalGranularity)
            ? FailureOr<int64_t>(failure())
            : getMaskLanesPerPart(*physicalGranularity);
    if (failed(lanesPerPart)) {
      return rewriter.notifyMatchFailure(
          op, "create_mask requires known physical mask lanes per part");
    }

    if (!activeConstant) {
      return lowerDynamicCreateMask(op, adaptor, resultVMIType, layout,
                                    *lanesPerPart, rewriter);
    }

    auto activeAttr = dyn_cast<IntegerAttr>(activeConstant.getValue());
    if (!activeAttr) {
      return rewriter.notifyMatchFailure(
          op, "create_mask active_lanes must be an integer constant");
    }

    int64_t activeLanes = activeAttr.getInt();
    if (activeLanes < 0) {
      activeLanes = 0;
    }
    if (activeLanes > resultVMIType.getElementCount()) {
      activeLanes = resultVMIType.getElementCount();
    }

    return lowerConstantCreateMask(op, activeLanes, resultVMIType, layout,
                                   *lanesPerPart, rewriter);
  }
};

struct OneToNVMICreateGroupMaskOpPattern
    : OneToNOpConversionPattern<VMICreateGroupMaskOp> {
  using OneToNOpConversionPattern<
      VMICreateGroupMaskOp>::OneToNOpConversionPattern;

private:
  FailureOr<SmallVector<Value>> materializeGroupMaskResults(
      VMICreateGroupMaskOp op,
      ArrayRef<ConstantMaskChunkMaterialization> materializations,
      ArrayRef<Type> resultTypes, StringRef overflowDiagnostic,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (const ConstantMaskChunkMaterialization &materialization :
         materializations) {
      bool tooManyMasks = results.size() >= resultTypes.size();
      if (tooManyMasks) {
        return rewriter.notifyMatchFailure(op, overflowDiagnostic);
      }
      auto maskType = dyn_cast<MaskType>(resultTypes[results.size()]);
      if (!maskType) {
        return rewriter.notifyMatchFailure(
            op, "create_group_mask result must be mask");
      }
      FailureOr<Value> mask = materializeConstantMaskChunk(
          op.getLoc(), maskType, materialization.activeLanes, rewriter);
      if (failed(mask)) {
        return rewriter.notifyMatchFailure(
            op, "failed to materialize create_group_mask physical chunk");
      }
      results.push_back(*mask);
    }
    bool resultArityMismatch = results.size() != resultTypes.size();
    if (resultArityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "create_group_mask physical result count mismatch");
    }
    return results;
  }

  LogicalResult lowerDynamicMask(
      VMICreateGroupMaskOp op, OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter, VMIMaskType resultVMIType,
      VMILayoutAttr resultLayout, ArrayRef<Type> resultTypes) const {
    FailureOr<Value> active = getSingleValue(
        op, adaptor.getActiveElemsPerGroup(),
        "create_group_mask active_elems_per_group must convert to one value",
        rewriter);
    if (failed(active)) {
      return failure();
    }

    if (resultLayout && resultLayout.isDeinterleaved()) {
      VMILayoutAttr contiguousLayout =
          VMILayoutAttr::getContiguous(op.getContext());
      auto contiguousType = VMIMaskType::get(
          op.getContext(), resultVMIType.getElementCount(),
          resultVMIType.getGranularity(), contiguousLayout);
      FailureOr<SmallVector<Value>> contiguousParts =
          materializeDynamicGroupMaskForType(op, *active, contiguousType,
                                             resultTypes, rewriter);
      if (failed(contiguousParts)) {
        return failure();
      }
      return replaceMaterializedResults(
          rewriter, op,
          materializeMaskLayoutConversion(op, *contiguousParts, resultTypes,
                                          contiguousLayout, resultLayout,
                                          rewriter),
          *this->getTypeConverter());
    }

    FailureOr<SmallVector<Value>> results = materializeDynamicGroupMaskForType(
        op, *active, resultVMIType, resultTypes, rewriter);
    if (failed(results)) {
      return failure();
    }
    return replaceMaterializedResults(rewriter, op, std::move(results),
                                      *this->getTypeConverter());
  }

  LogicalResult lowerConstantMask(
      VMICreateGroupMaskOp op, OneToNPatternRewriter &rewriter,
      ArrayRef<Type> resultTypes) const {
    std::string reason;
    FailureOr<SmallVector<ConstantMaskChunkMaterialization>> materializations =
        computeGroupMaskMaterialization(op, &reason);
    if (failed(materializations)) {
      return rewriter.notifyMatchFailure(
          op, Twine("create_group_mask ") + reason);
    }

    FailureOr<SmallVector<Value>> results = materializeGroupMaskResults(
        op, *materializations, resultTypes,
        "create_group_mask produced too many physical masks", rewriter);
    if (failed(results)) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<SmallVector<Value>> buildFactor4ContiguousParts(
      VMICreateGroupMaskOp op, OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter, VMIMaskType contiguousType,
      ArrayRef<Type> resultTypes) const {
    auto activeConstant =
        op.getActiveElemsPerGroup().getDefiningOp<arith::ConstantOp>();
    if (!activeConstant) {
      FailureOr<Value> active = getSingleValue(
          op, adaptor.getActiveElemsPerGroup(),
          "create_group_mask active_elems_per_group must convert to one value",
          rewriter);
      if (failed(active)) {
        return failure();
      }
      return materializeDynamicGroupMaskForType(
          op, *active, contiguousType, resultTypes, rewriter);
    }

    std::string contiguousReason;
    FailureOr<SmallVector<ConstantMaskChunkMaterialization>> materializations =
        computeGroupMaskMaterializationForType(op, contiguousType,
                                               &contiguousReason);
    if (failed(materializations)) {
      return rewriter.notifyMatchFailure(
          op, Twine("create_group_mask ") + contiguousReason);
    }
    return materializeGroupMaskResults(
        op, *materializations, resultTypes,
        "create_group_mask produced too many contiguous masks", rewriter);
  }

  LogicalResult lowerFactor4Block(
      VMICreateGroupMaskOp op, OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter, VMIMaskType resultVMIType,
      VMILayoutAttr resultLayout, ArrayRef<Type> resultTypes) const {
    VMILayoutAttr contiguousLayout =
        VMILayoutAttr::getContiguous(op.getContext());
    auto contiguousType =
        VMIMaskType::get(op.getContext(), resultVMIType.getElementCount(),
                         resultVMIType.getGranularity(), contiguousLayout);
    FailureOr<SmallVector<Value>> contiguousParts =
        buildFactor4ContiguousParts(op, adaptor, rewriter, contiguousType,
                                    resultTypes);
    if (failed(contiguousParts)) {
      return failure();
    }
    bool resultCountMismatch = contiguousParts->size() != resultTypes.size();
    if (resultCountMismatch) {
      return rewriter.notifyMatchFailure(
          op, "create_group_mask contiguous physical result count mismatch");
    }
    return replaceMaterializedResults(
        rewriter, op,
        materializeMaskLayoutConversion(op, *contiguousParts, resultTypes,
                                        contiguousLayout, resultLayout,
                                        rewriter),
        *this->getTypeConverter());
  }

public:

  LogicalResult
  matchAndRewrite(VMICreateGroupMaskOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    bool failedResultTypeConversion = failed(maybe_resultTypes);
    if (failedResultTypeConversion) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    auto resultVMIType = cast<VMIMaskType>(op.getResult().getType());
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    bool needsFactor4ContiguousMaterialization =
        resultLayout && resultLayout.isBlockDeinterleaved() &&
        resultLayout.getFactor() == 4;
    if (needsFactor4ContiguousMaterialization) {
      return lowerFactor4Block(op, adaptor, rewriter, resultVMIType,
                               resultLayout, resultTypes);
    }

    auto activeConstant =
        op.getActiveElemsPerGroup().getDefiningOp<arith::ConstantOp>();
    if (!activeConstant) {
      return lowerDynamicMask(op, adaptor, rewriter, resultVMIType,
                              resultLayout, resultTypes);
    }
    return lowerConstantMask(op, rewriter, resultTypes);
  }
};

struct OneToNVMILoadOpPattern : OneToNOpConversionPattern<VMILoadOp> {
  using OneToNOpConversionPattern<VMILoadOp>::OneToNOpConversionPattern;

private:
  struct LoadPhysicalPlan {
    Value source;
    Value offset;
    SmallVector<Type> resultTypes;
    SmallVector<Type> contiguousTypes;
    VMILayoutAttr resultLayout;
    int64_t lanesPerPart;
    bool noWiderThanContiguous;
  };

  FailureOr<SmallVector<Type>> getLoadResultTypes(
      VMILoadOp op) const {
    FailureOr<SmallVector<Type>> resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(resultTypes)) {
      return failure();
    }
    return std::move(*resultTypes);
  }

  FailureOr<SmallVector<Type>> getContiguousLoadTypes(
      VMILoadOp op, VMIVRegType resultVMIType,
      OneToNPatternRewriter &rewriter) const {
    VMILayoutAttr contiguousLayout =
        VMILayoutAttr::getContiguous(rewriter.getContext());
    FailureOr<SmallVector<Type>> contiguousTypes =
        getConvertedVRegTypesWithLayout(resultVMIType, contiguousLayout,
                                        *this->getTypeConverter());
    if (failed(contiguousTypes)) {
      return rewriter.notifyMatchFailure(
          op, "failed to compute contiguous load footprint");
    }
    return std::move(*contiguousTypes);
  }

  FailureOr<LoadPhysicalPlan> buildPhysicalPlan(
      VMILoadOp op, Value source, Value offset,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<SmallVector<Type>> resultTypes =
        getLoadResultTypes(op);
    if (failed(resultTypes)) {
      return failure();
    }
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    FailureOr<int64_t> lanesPerPart = verifyFullOrSafeReadVRegChunks(
        op, resultVMIType, op.getSource(), op.getOffset(), rewriter);
    if (failed(lanesPerPart)) {
      return failure();
    }
    FailureOr<SmallVector<Type>> contiguousTypes =
        getContiguousLoadTypes(op, resultVMIType, rewriter);
    if (failed(contiguousTypes)) {
      return failure();
    }
    FailureOr<bool> noWiderThanContiguous =
        hasNoWiderFootprintThanContiguous(*resultTypes, *contiguousTypes);
    if (failed(noWiderThanContiguous)) {
      return rewriter.notifyMatchFailure(
          op, "failed to compare load physical footprint");
    }
    return LoadPhysicalPlan{source,
                           offset,
                           std::move(*resultTypes),
                           std::move(*contiguousTypes),
                           resultVMIType.getLayoutAttr(),
                           *lanesPerPart,
                           *noWiderThanContiguous};
  }

  FailureOr<SmallVector<Value>> materializeLaneStrideParts(
      VMILoadOp op, OneToNPatternRewriter &rewriter, Value source, Value offset,
      VMIVRegType resultVMIType, ArrayRef<Type> resultTypes,
      StringRef dist) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    int64_t semanticOffset = 0;
    for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
      if (!isa<VRegType>(resultType)) {
        return rewriter.notifyMatchFailure(op, "load result must be vreg");
      }
      Value chunkOffset =
          createChunkOffset(op.getLoc(), offset, semanticOffset, rewriter);
      results.push_back(rewriter
                            .create<VldsOp>(op.getLoc(), resultType,
                                            /*updated_base=*/Type{}, source,
                                            chunkOffset,
                                            rewriter.getStringAttr(dist))
                            .getResult());
      FailureOr<int64_t> activeLanes =
          getActiveDataLanesInPhysicalChunk(resultVMIType, index);
      if (failed(activeLanes)) {
        return rewriter.notifyMatchFailure(
            op, "failed to compute lane_stride load active lanes");
      }
      semanticOffset += *activeLanes;
    }
    return results;
  }

  LogicalResult lowerLaneStride(
      VMILoadOp op, OneToNPatternRewriter &rewriter, Value source, Value offset,
      VMIVRegType resultVMIType, ArrayRef<Type> resultTypes,
      StringRef dist) const {
    FailureOr<SmallVector<Value>> results = materializeLaneStrideParts(
        op, rewriter, source, offset, resultVMIType, resultTypes, dist);
    if (failed(results)) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                     *this->getTypeConverter());
    return success();
  }

  LogicalResult lowerDeinterleaved2(
      VMILoadOp op, OneToNPatternRewriter &rewriter, Value source, Value offset,
      ArrayRef<Type> resultTypes, int64_t lanesPerPart, StringRef dist) const {
    bool invalidFactor2Arity = resultTypes.size() % 2 != 0;
    if (invalidFactor2Arity) {
      return rewriter.notifyMatchFailure(
          op, "vldsx2 deinterleaved=2 load requires even physical arity");
    }
    int64_t groups = resultTypes.size() / 2;
    SmallVector<Value> lows;
    SmallVector<Value> highs;
    lows.reserve(groups);
    highs.reserve(groups);
    for (int64_t group = 0; group < groups; ++group) {
      Type lowType = resultTypes[group];
      Type highType = resultTypes[groups + group];
      if (lowType != highType) {
        return rewriter.notifyMatchFailure(
            op, "vldsx2 requires matching low/high result types");
      }
      Value chunkOffset = createChunkOffset(
          op.getLoc(), offset, group * 2 * lanesPerPart, rewriter);
      auto load = rewriter.create<Vldsx2Op>(
          op.getLoc(), lowType, highType, Type{}, source, chunkOffset,
          rewriter.getStringAttr(dist));
      lows.push_back(load.getLow());
      highs.push_back(load.getHigh());
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    results.append(lows);
    results.append(highs);
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  FailureOr<std::array<Value, 4>> materializeDeinterleaved4LoadGroup(
      VMILoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, ArrayRef<Type> resultTypes, int64_t groups,
      int64_t group, int64_t lanesPerPart, StringRef dist) const {
    Type types[4] = {resultTypes[group], resultTypes[groups + group],
                     resultTypes[2 * groups + group],
                     resultTypes[3 * groups + group]};
    bool mismatchedTypes = types[0] != types[1] || types[0] != types[2] ||
                           types[0] != types[3];
    if (mismatchedTypes) {
      return rewriter.notifyMatchFailure(
          op, "vldsx2 deinterleaved=4 load requires matching part types");
    }
    Value firstOffset = createChunkOffset(
        op.getLoc(), offset, group * 4 * lanesPerPart, rewriter);
    Value secondOffset = createChunkOffset(
        op.getLoc(), offset, (group * 4 + 2) * lanesPerPart, rewriter);
    auto first = rewriter.create<Vldsx2Op>(
        op.getLoc(), types[0], types[1], Type{}, source, firstOffset,
        rewriter.getStringAttr(dist));
    auto second = rewriter.create<Vldsx2Op>(
        op.getLoc(), types[2], types[3], Type{}, source, secondOffset,
        rewriter.getStringAttr(dist));
    auto even = rewriter.create<VdintlvOp>(
        op.getLoc(), types[0], types[2], first.getLow(), second.getLow());
    auto odd = rewriter.create<VdintlvOp>(
        op.getLoc(), types[1], types[3], first.getHigh(), second.getHigh());
    return std::array<Value, 4>{even.getLow(), odd.getLow(), even.getHigh(),
                                odd.getHigh()};
  }

  LogicalResult lowerDeinterleaved4(
      VMILoadOp op, OneToNPatternRewriter &rewriter, Value source, Value offset,
      ArrayRef<Type> resultTypes, int64_t lanesPerPart, StringRef dist) const {
    bool invalidFactor4Arity = resultTypes.size() % 4 != 0;
    if (invalidFactor4Arity) {
      return rewriter.notifyMatchFailure(
          op, "vldsx2 deinterleaved=4 load requires physical arity divisible by 4");
    }
    int64_t groups = resultTypes.size() / 4;
    SmallVector<Value> parts[4];
    for (auto &part : parts) {
      part.reserve(groups);
    }
    for (int64_t group = 0; group < groups; ++group) {
      FailureOr<std::array<Value, 4>> groupValues =
          materializeDeinterleaved4LoadGroup(
              op, rewriter, source, offset, resultTypes, groups, group,
              lanesPerPart, dist);
      if (failed(groupValues)) {
        return failure();
      }
      for (size_t part = 0; part < 4; ++part) {
        parts[part].push_back((*groupValues)[part]);
      }
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto &part : parts) {
      results.append(part);
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  FailureOr<SmallVector<Value>> materializeAlignedContiguousParts(
      VMILoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, ArrayRef<Type> contiguousTypes, int64_t lanesPerPart) const {
    SmallVector<Value> parts;
    parts.reserve(contiguousTypes.size());
    for (auto [index, resultType] : llvm::enumerate(contiguousTypes)) {
      if (!isa<VRegType>(resultType)) {
        return rewriter.notifyMatchFailure(op, "load result must be vreg");
      }
      Value chunkOffset = createChunkOffset(
          op.getLoc(), offset, index * lanesPerPart, rewriter);
      parts.push_back(rewriter
                          .create<VldsOp>(op.getLoc(), resultType,
                                          /*updated_base=*/Type{}, source,
                                          chunkOffset, /*dist=*/nullptr)
                          .getResult());
    }
    return parts;
  }

  struct UnalignedLoadPart {
    Value result;
    Value base;
    Value align;
  };

  FailureOr<SmallVector<Value>> materializeUnalignedContiguousParts(
      VMILoadOp op, OneToNPatternRewriter &rewriter, Value source, Value offset,
      ArrayRef<Type> contiguousTypes, int64_t lanesPerPart) const {
    Value unalignedBase = materializeBufferPointer(
        source, getMemoryElementType(source.getType()),
        getMemorySpace(source.getType()), rewriter, op.getLoc());
    if (!unalignedBase) {
      return rewriter.notifyMatchFailure(
          op, "continuous unaligned load requires a ptr-compatible source");
    }
    unalignedBase = rewriter
                        .create<AddPtrOp>(op.getLoc(), unalignedBase.getType(),
                                          unalignedBase, offset)
                        .getResult();
    Value unalignedAlign = rewriter
                               .create<VldasOp>(
                                   op.getLoc(),
                                   AlignType::get(rewriter.getContext()),
                                   unalignedBase)
                               .getResult();
    SmallVector<Value> parts;
    parts.reserve(contiguousTypes.size());
    for (Type resultType : contiguousTypes) {
      FailureOr<UnalignedLoadPart> updatedState = emitUnalignedLoadPart(
          op, rewriter, unalignedBase, unalignedAlign, resultType,
          lanesPerPart);
      if (failed(updatedState)) {
        return failure();
      }
      parts.push_back(updatedState->result);
      unalignedBase = updatedState->base;
      unalignedAlign = updatedState->align;
    }
    return parts;
  }

  FailureOr<UnalignedLoadPart> emitUnalignedLoadPart(
      VMILoadOp op, OneToNPatternRewriter &rewriter, Value base, Value align,
      Type resultType, int64_t lanesPerPart) const {
    if (!isa<VRegType>(resultType)) {
      return rewriter.notifyMatchFailure(op, "load result must be vreg");
    }
    Value increment =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), lanesPerPart);
    auto load = rewriter.create<VldusOp>(
        op.getLoc(), resultType, align.getType(), base.getType(), base, align,
        increment);
    return UnalignedLoadPart{load.getResult(), load.getUpdatedBase(),
                             load.getUpdatedAlign()};
  }

  FailureOr<SmallVector<Value>> materializeContiguousLoadParts(
      VMILoadOp op, OneToNPatternRewriter &rewriter, Value source, Value offset,
      VMIVRegType resultVMIType, ArrayRef<Type> contiguousTypes,
      int64_t lanesPerPart) const {
    auto firstType = contiguousTypes.empty()
                         ? VRegType{}
                         : dyn_cast<VRegType>(contiguousTypes.front());
    bool useAlignedAccess =
        firstType && isDirectMemoryDistAddressLegal(
                          op.getSource(), op.getOffset(),
                          resultVMIType.getElementType(), firstType,
                          VPTOMemoryOpFamily::Load, "NORM");
    if (useAlignedAccess) {
      return materializeAlignedContiguousParts(
          op, rewriter, source, offset, contiguousTypes, lanesPerPart);
    }
    return materializeUnalignedContiguousParts(
        op, rewriter, source, offset, contiguousTypes, lanesPerPart);
  }

  LogicalResult lowerContiguous(
      VMILoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, VMIVRegType resultVMIType, ArrayRef<Type> resultTypes,
      ArrayRef<Type> contiguousTypes, int64_t lanesPerPart,
      VMILayoutAttr contiguousLayout) const {
    FailureOr<SmallVector<Value>> contiguousParts =
        materializeContiguousLoadParts(op, rewriter, source, offset,
                                       resultVMIType, contiguousTypes,
                                       lanesPerPart);
    if (failed(contiguousParts)) {
      return failure();
    }
    FailureOr<SmallVector<Value>> results = materializeDataLayoutConversion(
        op, *contiguousParts, resultTypes, contiguousLayout,
        resultVMIType.getLayoutAttr(), resultVMIType.getElementType(),
        rewriter);
    if (failed(results)) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                     *this->getTypeConverter());
    return success();
  }

  std::optional<LogicalResult> lowerDirectDeinterleaved(
      VMILoadOp op, OneToNPatternRewriter &rewriter, Value source, Value offset,
      VMIVRegType resultVMIType, VMILayoutAttr resultLayout,
      ArrayRef<Type> resultTypes, int64_t lanesPerPart,
      bool noWiderThanContiguous) const {
    bool unsupportedLayout = !resultLayout || !resultLayout.isDeinterleaved();
    if (unsupportedLayout || !noWiderThanContiguous) {
      return std::nullopt;
    }
    int64_t factor = resultLayout.getFactor();
    bool supportedFactor = factor == 2 || factor == 4;
    if (!supportedFactor) {
      return std::nullopt;
    }
    std::optional<std::string> dist =
        getX2MemoryDistToken(resultVMIType.getElementType(), "DINTLV");
    auto firstType =
        resultTypes.empty() ? VRegType{} : dyn_cast<VRegType>(resultTypes.front());
    bool canUseDist =
        dist && firstType &&
        isDirectMemoryDistAddressLegal(
            op.getSource(), op.getOffset(), resultVMIType.getElementType(),
            firstType, VPTOMemoryOpFamily::LoadX2, *dist);
    bool validArity = resultTypes.size() % static_cast<size_t>(factor) == 0;
    if (!canUseDist || !validArity) {
      return std::nullopt;
    }
    if (factor == 2) {
      return lowerDeinterleaved2(op, rewriter, source, offset, resultTypes,
                                 lanesPerPart, *dist);
    }
    return lowerDeinterleaved4(op, rewriter, source, offset, resultTypes,
                               lanesPerPart, *dist);
  }

  std::optional<std::string> getLoadLaneStrideDist(
      VMILoadOp op, VMIVRegType resultVMIType,
      ArrayRef<Type> resultTypes) const {
    std::optional<std::string> dist =
        getDenseLaneStrideLoadDistToken(resultVMIType);
    auto resultType =
        resultTypes.empty() ? VRegType{} : dyn_cast<VRegType>(resultTypes.front());
    if (!dist || !resultType ||
        !isDirectMemoryDistAddressLegal(
            op.getSource(), op.getOffset(), resultVMIType.getElementType(),
            resultType, VPTOMemoryOpFamily::Load, *dist)) {
      return std::nullopt;
    }
    return dist;
  }

public:

  LogicalResult
  matchAndRewrite(VMILoadOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    FailureOr<Value> source = getSingleValue(
        op, adaptor.getSource(), "load source must convert to one value",
        rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(), "load offset must convert to one value",
        rewriter);
    bool failedOperands = failed(source) || failed(offset);
    if (failedOperands) {
      return failure();
    }
    FailureOr<SmallVector<Type>> maybe_resultTypes = getLoadResultTypes(op);
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);

    // Contiguous lane_stride loads read packed data straight from memory with
    // a dist-bearing vlds; resolve them before the full-chunk verification so
    // the strided footprint is not mistaken for an incomplete chunk.
    std::optional<std::string> laneStrideDist =
        getLoadLaneStrideDist(op, resultVMIType, resultTypes);
    if (laneStrideDist) {
      return lowerLaneStride(op, rewriter, *source, *offset, resultVMIType,
                             resultTypes, *laneStrideDist);
    }

    FailureOr<LoadPhysicalPlan> plan =
        buildPhysicalPlan(op, *source, *offset, rewriter);
    if (failed(plan)) {
      return failure();
    }

    std::optional<LogicalResult> deinterleavedResult =
        lowerDirectDeinterleaved(op, rewriter, plan->source, plan->offset,
                                 resultVMIType, plan->resultLayout,
                                 plan->resultTypes, plan->lanesPerPart,
                                 plan->noWiderThanContiguous);
    if (deinterleavedResult) {
      return *deinterleavedResult;
    }

    return lowerContiguous(op, rewriter, plan->source, plan->offset,
                            resultVMIType, plan->resultTypes,
                            plan->contiguousTypes, plan->lanesPerPart,
                            VMILayoutAttr::getContiguous(
                                rewriter.getContext()));
  }
};

struct OneToNVMIDeinterleaveLoadOpPattern
    : OneToNOpConversionPattern<VMIDeinterleaveLoadOp> {
  using OneToNOpConversionPattern<
      VMIDeinterleaveLoadOp>::OneToNOpConversionPattern;

private:
  struct UnalignedDeinterleaveLoadPair {
    Value low;
    Value high;
    Value updatedBase;
    Value updatedAlign;
  };

  struct DeinterleaveLoadLoweringInput {
    Value source;
    Value offset;
    SmallVector<Type> lowTypes;
    SmallVector<Type> highTypes;
    VMIVRegType lowVMIType;
    int64_t lanesPerPart;
    std::string dist;
  };

  FailureOr<UnalignedDeinterleaveLoadPair> materializeUnalignedLoadPair(
      VMIDeinterleaveLoadOp op, OneToNPatternRewriter &rewriter,
      Value streamBase, Value streamAlign, Type lowType, Type highType,
      Value increment) const {
    if (lowType != highType) {
      return rewriter.notifyMatchFailure(
          op, "deinterleave_load requires matching low/high physical types");
    }
    auto first = rewriter.create<VldusOp>(
        op.getLoc(), lowType, streamAlign.getType(), streamBase.getType(),
        streamBase, streamAlign, increment);
    auto second = rewriter.create<VldusOp>(
        op.getLoc(), highType, first.getUpdatedAlign().getType(),
        first.getUpdatedBase().getType(), first.getUpdatedBase(),
        first.getUpdatedAlign(), increment);
    auto deinterleaved = rewriter.create<VdintlvOp>(
        op.getLoc(), lowType, highType, first.getResult(), second.getResult());
    return UnalignedDeinterleaveLoadPair{deinterleaved.getLow(),
                                         deinterleaved.getHigh(),
                                         second.getUpdatedBase(),
                                         second.getUpdatedAlign()};
  }

  LogicalResult lowerDirect(
      VMIDeinterleaveLoadOp op, OneToNPatternRewriter &rewriter,
      Value source, Value offset, ArrayRef<Type> lowTypes,
      ArrayRef<Type> highTypes, int64_t lanesPerPart, StringRef dist) const {
    SmallVector<Value> lows;
    SmallVector<Value> highs;
    lows.reserve(lowTypes.size());
    highs.reserve(highTypes.size());
    for (size_t index = 0; index < lowTypes.size(); ++index) {
      if (lowTypes[index] != highTypes[index]) {
        return rewriter.notifyMatchFailure(
            op, "deinterleave_load requires matching low/high physical types");
      }
      Value chunkOffset = createChunkOffset(
          op.getLoc(), offset, static_cast<int64_t>(index) * 2 * lanesPerPart,
          rewriter);
      auto load = rewriter.create<Vldsx2Op>(
          op.getLoc(), lowTypes[index], highTypes[index], Type{}, source,
          chunkOffset, rewriter.getStringAttr(dist));
      lows.push_back(load.getLow());
      highs.push_back(load.getHigh());
    }
    SmallVector<Value> results;
    results.append(lows);
    results.append(highs);
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  LogicalResult lowerUnaligned(
      VMIDeinterleaveLoadOp op, OneToNPatternRewriter &rewriter,
      Value source, Value offset, ArrayRef<Type> lowTypes,
      ArrayRef<Type> highTypes, int64_t lanesPerPart) const {
    Value streamBase = materializeBufferPointer(
        source, getMemoryElementType(source.getType()),
        getMemorySpace(source.getType()), rewriter, op.getLoc());
    if (!streamBase) {
      return rewriter.notifyMatchFailure(
          op, "unaligned deinterleave_load requires a ptr-compatible source");
    }
    streamBase = rewriter
                     .create<AddPtrOp>(op.getLoc(), streamBase.getType(),
                                       streamBase, offset)
                     .getResult();
    Value streamAlign = rewriter
                            .create<VldasOp>(
                                op.getLoc(), AlignType::get(rewriter.getContext()),
                                streamBase)
                            .getResult();
    Value increment =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), lanesPerPart);
    SmallVector<Value> lows;
    SmallVector<Value> highs;
    lows.reserve(lowTypes.size());
    highs.reserve(highTypes.size());
    for (size_t index = 0; index < lowTypes.size(); ++index) {
      FailureOr<UnalignedDeinterleaveLoadPair> pair =
          materializeUnalignedLoadPair(op, rewriter, streamBase, streamAlign,
                                       lowTypes[index], highTypes[index],
                                       increment);
      if (failed(pair)) {
        return failure();
      }
      lows.push_back(pair->low);
      highs.push_back(pair->high);
      streamBase = pair->updatedBase;
      streamAlign = pair->updatedAlign;
    }
    SmallVector<Value> results;
    results.reserve(lows.size() + highs.size());
    results.append(lows);
    results.append(highs);
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  FailureOr<std::pair<SmallVector<Type>, SmallVector<Type>>>
  getDeinterleaveLoadResultTypes(
      VMIDeinterleaveLoadOp op, OneToNPatternRewriter &rewriter) const {
    FailureOr<SmallVector<Type>> lowTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    FailureOr<SmallVector<Type>> highTypes =
        getConvertedResultTypes(op, 1, *this->getTypeConverter());
    bool failedTypeConversion = failed(lowTypes) || failed(highTypes);
    if (failedTypeConversion) {
      return failure();
    }
    bool mismatchedArity = lowTypes->size() != highTypes->size();
    if (mismatchedArity) {
      return rewriter.notifyMatchFailure(
          op, "deinterleave_load requires matching low/high physical arity");
    }
    return std::make_pair(std::move(*lowTypes), std::move(*highTypes));
  }

  FailureOr<DeinterleaveLoadLoweringInput> getLoweringInput(
      VMIDeinterleaveLoadOp op, OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const {
    auto lowVMIType = cast<VMIVRegType>(op.getLow().getType());
    FailureOr<Value> source = getSingleValue(
        op, adaptor.getSource(),
        "deinterleave_load source must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(),
        "deinterleave_load offset must convert to one value", rewriter);
    bool invalidOperands = failed(source) || failed(offset);
    if (invalidOperands) {
      return failure();
    }
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(lowVMIType.getElementType());
    if (failed(lanesPerPart)) {
      return rewriter.notifyMatchFailure(
          op, "deinterleave_load requires known physical lanes per part");
    }
    std::optional<std::string> dist =
        getX2MemoryDistToken(lowVMIType.getElementType(), "DINTLV");
    if (!dist) {
      return rewriter.notifyMatchFailure(
          op, "deinterleave_load requires vldsx2 DINTLV element support");
    }
    FailureOr<std::pair<SmallVector<Type>, SmallVector<Type>>> resultTypes =
        getDeinterleaveLoadResultTypes(op, rewriter);
    if (failed(resultTypes)) {
      return failure();
    }
    return DeinterleaveLoadLoweringInput{
        *source, *offset, std::move(resultTypes->first),
        std::move(resultTypes->second), lowVMIType, *lanesPerPart, *dist};
  }

  LogicalResult lowerByAddressPlan(
      VMIDeinterleaveLoadOp op, const DeinterleaveLoadLoweringInput &input,
      OneToNPatternRewriter &rewriter) const {
    auto firstType = input.lowTypes.empty()
                         ? VRegType{}
                         : dyn_cast<VRegType>(input.lowTypes.front());
    bool useDirectAccess =
        firstType && isDirectMemoryDistAddressLegal(
                         op.getSource(), op.getOffset(),
                         input.lowVMIType.getElementType(), firstType,
                         VPTOMemoryOpFamily::LoadX2, input.dist);
    if (!useDirectAccess) {
      return lowerUnaligned(op, rewriter, input.source, input.offset,
                            input.lowTypes, input.highTypes,
                            input.lanesPerPart);
    }
    return lowerDirect(op, rewriter, input.source, input.offset, input.lowTypes,
                       input.highTypes, input.lanesPerPart, input.dist);
  }


public:

  LogicalResult
  matchAndRewrite(VMIDeinterleaveLoadOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<DeinterleaveLoadLoweringInput> input =
        getLoweringInput(op, adaptor, rewriter);
    if (failed(input)) {
      return failure();
    }
    return lowerByAddressPlan(op, *input, rewriter);
  }
};

struct OneToNVMIGroupLoadOpPattern : OneToNOpConversionPattern<VMIGroupLoadOp> {
  using OneToNOpConversionPattern<VMIGroupLoadOp>::OneToNOpConversionPattern;

private:
  FailureOr<SmallVector<Type>> getResultTypes(
      VMIGroupLoadOp op, OneToNPatternRewriter &rewriter) const {
    FailureOr<SmallVector<Type>> resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(resultTypes)) {
      return failure();
    }
    return std::move(*resultTypes);
  }

  LogicalResult lowerContiguousPath(
      VMIGroupLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, Value rowStride, VMIVRegType resultVMIType,
      VMILayoutAttr resultLayout) const {
    FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
        resultVMIType, op.getNumGroupsAttr().getInt());
    if (failed(groupSize)) {
      return rewriter.notifyMatchFailure(
          op, "group_load requires num_groups to evenly divide lane count");
    }
    std::optional<int64_t> constantRowStride =
        getConstantIndexValue(op.getRowStride());
    FailureOr<SmallVector<Type>> resultTypes = getResultTypes(op, rewriter);
    if (failed(resultTypes)) {
      return failure();
    }
    bool unitStride = constantRowStride && *constantRowStride == *groupSize;
    if (unitStride && resultLayout && resultLayout.isContiguous()) {
      return lowerContiguousUnitStride(op, rewriter, source, offset,
                                       resultVMIType, *resultTypes);
    }
    return lowerContiguousChunks(op, rewriter, source, offset, rowStride,
                                 resultVMIType, *resultTypes);
  }

  LogicalResult lowerContiguousUnitStride(
      VMIGroupLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, VMIVRegType resultVMIType, ArrayRef<Type> resultTypes) const {
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(resultVMIType.getElementType());
    if (failed(lanesPerPart)) {
      return rewriter.notifyMatchFailure(
          op, "contiguous group_load requires known physical lanes");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
      FailureOr<Value> result = materializeContiguousUnitStrideChunk(
          op, rewriter, source, offset, resultType, index, *lanesPerPart);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  FailureOr<Value> materializeContiguousUnitStrideChunk(
      VMIGroupLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, Type resultType, size_t index, int64_t lanesPerPart) const {
    if (!isa<VRegType>(resultType)) {
      return rewriter.notifyMatchFailure(
          op, "contiguous group_load result must be vreg");
    }
    Value chunkOffset = createChunkOffset(
        op.getLoc(), offset, static_cast<int64_t>(index) * lanesPerPart,
        rewriter);
    return rewriter
        .create<VldsOp>(op.getLoc(), resultType, Type{}, source, chunkOffset,
                        nullptr)
        .getResult();
  }

  FailureOr<Value> materializeBlockDeinterleavedChunk(
      VMIGroupLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, Value rowStride, Type resultType, int64_t part,
      int64_t chunk, int64_t blockElems, int64_t constantRowStride) const {
    auto vregType = dyn_cast<VRegType>(resultType);
    if (!vregType) {
      return rewriter.notifyMatchFailure(
          op, "block_deinterleaved group_load result must be vreg");
    }
    FailureOr<Value> allMask =
        createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
    if (failed(allMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create block_deinterleaved group_load mask");
    }
    Value blockStride = rewriter.create<arith::ConstantIntOp>(
        op.getLoc(), constantRowStride / 8, 16);
    Value zeroI16 = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 16);
    Value chunkOffset = createGroupChunkOffset(
        op.getLoc(), offset, rowStride, chunk * 8, part * blockElems,
        rewriter);
    Value chunkBase = rewriter
                          .create<AddPtrOp>(op.getLoc(), source.getType(), source,
                                            chunkOffset)
                          .getResult();
    return rewriter
        .create<VsldbOp>(op.getLoc(), vregType, Type{}, chunkBase, blockStride,
                         zeroI16, *allMask)
        .getResult();
  }

  LogicalResult lowerBlockDeinterleaved(
      VMIGroupLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, Value rowStride, VMIVRegType resultVMIType,
      ArrayRef<Type> resultTypes, VMILayoutAttr resultLayout,
      int64_t factor, int64_t blockElems, int64_t chunksPerPart,
      int64_t constantRowStride) const {
    bool invalidResultArity =
        static_cast<int64_t>(resultTypes.size()) != factor * chunksPerPart;
    if (invalidResultArity) {
      return rewriter.notifyMatchFailure(
          op, "block_deinterleaved group_load arity mismatch");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (int64_t part = 0; part < factor; ++part) {
      for (int64_t chunk = 0; chunk < chunksPerPart; ++chunk) {
        int64_t flatIndex = part * chunksPerPart + chunk;
        FailureOr<Value> result = materializeBlockDeinterleavedChunk(
            op, rewriter, source, offset, rowStride, resultTypes[flatIndex],
            part, chunk, blockElems, constantRowStride);
        if (failed(result)) {
          return failure();
        }
        results.push_back(*result);
      }
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  FailureOr<Value> materializeContiguousGroupLoadChunk(
      VMIGroupLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, Value rowStride, Type resultType, int64_t group,
      int64_t chunkInGroup, int64_t lanesPerPart) const {
    if (!isa<VRegType>(resultType)) {
      return rewriter.notifyMatchFailure(op, "group_load result must be vreg");
    }
    Value chunkOffset = createGroupChunkOffset(
        op.getLoc(), offset, rowStride, group, chunkInGroup * lanesPerPart,
        rewriter);
    return rewriter
        .create<VldsOp>(op.getLoc(), resultType, Type{}, source, chunkOffset,
                        nullptr)
        .getResult();
  }

  LogicalResult lowerContiguousChunks(
      VMIGroupLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, Value rowStride, VMIVRegType resultVMIType,
      ArrayRef<Type> resultTypes) const {
    int64_t lanesPerPart = 0;
    int64_t groupCount = 0;
    int64_t chunksPerGroup = 0;
    FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
        resultVMIType, op.getNumGroupsAttr().getInt());
    if (failed(groupSize)) {
      return rewriter.notifyMatchFailure(
          op, "group_load requires num_groups to evenly divide lane count");
    }
    if (failed(checkContiguousFullGroupChunks(
            op, resultVMIType, *groupSize, &lanesPerPart, &groupCount,
            &chunksPerGroup, rewriter))) {
      return failure();
    }
    bool invalidArity = static_cast<int64_t>(resultTypes.size()) !=
                        groupCount * chunksPerGroup;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(op, "group_load arity mismatch");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
      int64_t group = index / chunksPerGroup;
      int64_t chunkInGroup = index % chunksPerGroup;
      FailureOr<Value> result = materializeContiguousGroupLoadChunk(
          op, rewriter, source, offset, rowStride, resultType, group,
          chunkInGroup, lanesPerPart);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  FailureOr<std::pair<int64_t, int64_t>> validateBlockF32Shape(
      VMIGroupLoadOp op, Value source, VMIVRegType resultVMIType,
      VMILayoutAttr resultLayout, OneToNPatternRewriter &rewriter) const {
    FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
        resultVMIType, op.getNumGroupsAttr().getInt());
    if (failed(groupSize)) {
      return rewriter.notifyMatchFailure(
          op, "group_load requires num_groups to evenly divide lane count");
    }
    bool validFactorShape =
        (*groupSize == 16 && resultLayout.getFactor() == 2) ||
        (*groupSize == 32 && resultLayout.getFactor() == 4);
    std::optional<int64_t> constantRowStride =
        getConstantIndexValue(op.getRowStride());
    bool validRowStride = constantRowStride && *constantRowStride > 0 &&
                          *constantRowStride % 8 == 0;
    bool validGroupCount = op.getNumGroupsAttr().getInt() % 8 == 0;
    if (!validFactorShape || !validGroupCount || !validRowStride ||
        !isa<PtrType>(source.getType())) {
      return rewriter.notifyMatchFailure(
          op, !validFactorShape
                  ? "block_deinterleaved group_load requires S=16/factor=2 or S=32/factor=4"
                  : !validGroupCount
                        ? "block_deinterleaved group_load requires num_groups multiple of 8"
                        : !validRowStride
                              ? "block_deinterleaved group_load requires constant positive "
                                "row_stride divisible by 8 f32 elements"
                              : "block_deinterleaved group_load requires !pto.ptr source");
    }
    return std::make_pair(*constantRowStride, resultLayout.getFactor());
  }

  LogicalResult lowerBlockF32(
      VMIGroupLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, Value rowStride, VMIVRegType resultVMIType,
      VMILayoutAttr resultLayout) const {
    FailureOr<std::pair<int64_t, int64_t>> shape =
        validateBlockF32Shape(op, source, resultVMIType, resultLayout,
                              rewriter);
    if (failed(shape)) {
      return failure();
    }
    int64_t constantRowStride = shape->first;
    int64_t factor = shape->second;
    FailureOr<SmallVector<Type>> maybeResultTypes = getResultTypes(op, rewriter);
    if (failed(maybeResultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybeResultTypes);
    FailureOr<int64_t> blockElems = getVMILayoutBlockElems(resultVMIType);
    FailureOr<int64_t> chunksPerPart =
        getDataChunksInPart(resultVMIType, 0);
    bool invalidChunks = failed(blockElems) || failed(chunksPerPart) ||
                         *chunksPerPart <= 0;
    if (invalidChunks) {
      return rewriter.notifyMatchFailure(
          op, "block_deinterleaved group_load requires known block and "
              "chunks per part");
    }
    for (int64_t part = 1; part < factor; ++part) {
      FailureOr<int64_t> currentChunks =
          getDataChunksInPart(resultVMIType, part);
      bool nonUniformChunks =
          failed(currentChunks) || *currentChunks != *chunksPerPart;
      if (nonUniformChunks) {
        return rewriter.notifyMatchFailure(
            op, "block_deinterleaved group_load requires uniform chunks per "
                "part");
      }
    }
    return lowerBlockDeinterleaved(
        op, rewriter, source, offset, rowStride, resultVMIType, resultTypes,
        resultLayout, factor, *blockElems, *chunksPerPart, constantRowStride);
  }

  LogicalResult lowerByLayout(
      VMIGroupLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, Value rowStride, VMIVRegType resultVMIType,
      VMILayoutAttr resultLayout) const {
    bool isBlockF32 = resultLayout && resultLayout.isBlockDeinterleaved() &&
                      resultVMIType.getElementType().isF32();
    if (isBlockF32) {
      return lowerBlockF32(op, rewriter, source, offset, rowStride,
                           resultVMIType, resultLayout);
    }
    return lowerContiguousPath(op, rewriter, source, offset, rowStride,
                               resultVMIType, resultLayout);
  }

public:

  LogicalResult
  matchAndRewrite(VMIGroupLoadOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
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
    bool invalidOperands =
        failed(source) || failed(offset) || failed(rowStride);
    if (invalidOperands) {
      return failure();
    }

    return lowerByLayout(op, rewriter, *source, *offset, *rowStride,
                         resultVMIType, resultVMIType.getLayoutAttr());
  }
};

struct GroupSlotLoadResultPart {
  VRegType valueType;
  MaskType maskType;
};

static FailureOr<GroupSlotLoadResultPart> getGroupSlotLoadResultPart(
    Operation *op, Type resultType, OneToNPatternRewriter &rewriter) {
  auto vregType = dyn_cast<VRegType>(resultType);
  if (!vregType) {
    (void)rewriter.notifyMatchFailure(op,
                                      "group_slot_load result must be vreg");
    return failure();
  }
  FailureOr<MaskType> maskType =
      getMaskTypeForVReg(vregType, rewriter.getContext());
  if (failed(maskType)) {
    (void)rewriter.notifyMatchFailure(
        op, "unsupported element type for group_slot_load mask");
    return failure();
  }
  return GroupSlotLoadResultPart{vregType, *maskType};
}

static LogicalResult lowerSingleGroupSlotLoad(
    Operation *op, Value source, Value offset, VMIVRegType resultVMIType,
    TypeRange resultTypes, OneToNPatternRewriter &rewriter,
    SmallVectorImpl<Value> &results) {
  std::optional<std::string> dist =
      getScalarBroadcastLoadDistToken(resultVMIType.getElementType());
  if (!dist) {
    return rewriter.notifyMatchFailure(
        op, "single-slot group_slot_load requires supported BRC load element width");
  }
  auto vregType = dyn_cast<VRegType>(resultTypes.front());
  if (!vregType) {
    return rewriter.notifyMatchFailure(
        op, "single-slot group_slot_load result must be vreg");
  }
  results.push_back(rewriter
                       .create<VldsOp>(op->getLoc(), vregType,
                                       /*updated_base=*/Type{}, source, offset,
                                       rewriter.getStringAttr(*dist))
                       .getResult());
  return success();
}

static LogicalResult emitGroupSlotLoadSlots8Chunk(
    Operation *op, Value source, Value offset, Type resultType, int64_t chunk,
    int64_t numGroups,
    OneToNPatternRewriter &rewriter, SmallVectorImpl<Value> &results) {
  FailureOr<GroupSlotLoadResultPart> resultPart =
      getGroupSlotLoadResultPart(op, resultType, rewriter);
  if (failed(resultPart)) {
    return failure();
  }
  int64_t groupBegin = chunk * 8;
  int64_t activeGroups = std::min<int64_t>(8, numGroups - groupBegin);
  if (activeGroups <= 0) {
    return rewriter.notifyMatchFailure(
        op, "slots=8 group_slot_load has no active groups for chunk");
  }
  std::string pattern = (Twine("PAT_VL") + Twine(activeGroups)).str();
  FailureOr<Value> slotMask = createPrefixMask(
      op->getLoc(), resultPart->maskType, pattern, rewriter);
  if (failed(slotMask)) {
    return rewriter.notifyMatchFailure(
        op, "failed to create slots=8 group_slot_load mask");
  }
  Value groupOffset =
      createChunkOffset(op->getLoc(), offset, groupBegin, rewriter);
  Value slotBase = rewriter
                       .create<AddPtrOp>(op->getLoc(), source.getType(), source,
                                         groupOffset)
                       .getResult();
  auto zeroI16 = rewriter.create<arith::ConstantIntOp>(op->getLoc(), 0, 16);
  results.push_back(
      rewriter
          .create<VsldbOp>(op->getLoc(), resultPart->valueType,
                          /*updated_base=*/Type{}, slotBase, zeroI16, zeroI16,
                          *slotMask)
          .getResult());
  return success();
}

static LogicalResult lowerGroupSlotLoadSlots8(
    Operation *op, Value source, Value offset, Value sourceGroupStride,
    VMIVRegType resultVMIType, TypeRange resultTypes, int64_t numGroups,
    OneToNPatternRewriter &rewriter, SmallVectorImpl<Value> &results) {
  std::optional<int64_t> stride = getConstantIndexValue(sourceGroupStride);
  if (!stride || *stride != 1) {
    return rewriter.notifyMatchFailure(
        op, "slots=8 group_slot_load requires constant unit stride");
  }
  if (numGroups == 1) {
    return lowerSingleGroupSlotLoad(op, source, offset, resultVMIType,
                                    resultTypes, rewriter, results);
  }
  for (auto [chunk, resultType] : llvm::enumerate(resultTypes)) {
    if (failed(emitGroupSlotLoadSlots8Chunk(
            op, source, offset, resultType,
            static_cast<int64_t>(chunk), numGroups, rewriter, results))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult emitGroupSlotLoadSlots1Chunk(
    Operation *op, Value source, Value offset, Value sourceGroupStride,
    Type resultType, int64_t group, OneToNPatternRewriter &rewriter,
    SmallVectorImpl<Value> &results) {
  FailureOr<GroupSlotLoadResultPart> resultPart =
      getGroupSlotLoadResultPart(op, resultType, rewriter);
  if (failed(resultPart)) {
    return failure();
  }
  FailureOr<Value> oneBlockMask = createPrefixMask(
      op->getLoc(), resultPart->maskType, "PAT_VL1", rewriter);
  if (failed(oneBlockMask)) {
    return rewriter.notifyMatchFailure(op, "failed to create group_slot_load mask");
  }
  Value groupOffset = offset;
  if (group != 0) {
    Value groupIndex =
        rewriter.create<arith::ConstantIndexOp>(op->getLoc(), group);
    Value rowOffset = rewriter
                          .create<arith::MulIOp>(op->getLoc(),
                                                 sourceGroupStride, groupIndex)
                          .getResult();
    groupOffset = rewriter
                      .create<arith::AddIOp>(op->getLoc(), groupOffset,
                                             rowOffset)
                      .getResult();
  }
  Value slotBase = rewriter
                       .create<AddPtrOp>(op->getLoc(), source.getType(), source,
                                         groupOffset)
                       .getResult();
  auto zeroI16 = rewriter.create<arith::ConstantIntOp>(op->getLoc(), 0, 16);
  results.push_back(
      rewriter
          .create<VsldbOp>(op->getLoc(), resultPart->valueType,
                          /*updated_base=*/Type{}, slotBase, zeroI16, zeroI16,
                          *oneBlockMask)
          .getResult());
  return success();
}

static LogicalResult lowerGroupSlotLoadSlots1(
    Operation *op, Value source, Value offset, Value sourceGroupStride,
    VMIVRegType resultVMIType, TypeRange resultTypes,
    OneToNPatternRewriter &rewriter, SmallVectorImpl<Value> &results) {
  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(resultVMIType.getElementType());
  if (elementBits == 0 || 256 % elementBits != 0) {
    return rewriter.notifyMatchFailure(
        op, "slots=1 group_slot_load requires supported element width");
  }
  int64_t alignedStrideElems = 256 / elementBits;
  std::optional<int64_t> constantStride =
      getConstantIndexValue(sourceGroupStride);
  if (!constantStride || *constantStride <= 0 ||
      *constantStride % alignedStrideElems != 0) {
    return rewriter.notifyMatchFailure(
        op, Twine("slots=1 group_slot_load requires constant positive "
                  "source_group_stride divisible by ") +
                Twine(alignedStrideElems) +
                " elements for 32B lane-0 vsldb alignment");
  }
  for (auto [group, resultType] : llvm::enumerate(resultTypes)) {
    if (failed(emitGroupSlotLoadSlots1Chunk(
            op, source, offset, sourceGroupStride, resultType,
            static_cast<int64_t>(group), rewriter, results))) {
      return failure();
    }
  }
  return success();
}

static FailureOr<int64_t> getGroupSlotLoadSlots(
    Operation *op, Value source, VMIVRegType resultVMIType,
    TypeRange resultTypes, int64_t numGroups,
    OneToNPatternRewriter &rewriter) {
  VMILayoutAttr layout = resultVMIType.getLayoutAttr();
  bool invalidLayout = !layout || !layout.isGroupSlots() || layout.getSlots() <= 0;
  if (invalidLayout) {
    (void)rewriter.notifyMatchFailure(
        op, "group_slot_load requires explicit group_slots layout");
    return failure();
  }
  if (!isa<PtrType>(source.getType())) {
    (void)rewriter.notifyMatchFailure(
        op, "group_slot_load requires !pto.ptr source");
    return failure();
  }
  int64_t slots = layout.getSlots();
  int64_t expectedArity = ceilDivNonNegative(numGroups, slots);
  bool arityMismatch =
      static_cast<int64_t>(resultTypes.size()) != expectedArity;
  if (arityMismatch) {
    (void)rewriter.notifyMatchFailure(op, "group_slot_load arity mismatch");
    return failure();
  }
  if (slots != 8 && slots != 1) {
    (void)rewriter.notifyMatchFailure(
        op, "group_slot_load supports only slots=8 or slots=1");
    return failure();
  }
  return slots;
}

static LogicalResult lowerGroupSlotLoadParts(
    Operation *op, Value source, Value offset, Value sourceGroupStride,
    VMIVRegType resultVMIType, TypeRange resultTypes, int64_t numGroups,
    OneToNPatternRewriter &rewriter, SmallVectorImpl<Value> &results) {
  FailureOr<int64_t> maybeSlots = getGroupSlotLoadSlots(
      op, source, resultVMIType, resultTypes, numGroups, rewriter);
  if (failed(maybeSlots)) {
    return failure();
  }
  int64_t slots = *maybeSlots;
  results.reserve(results.size() + resultTypes.size());
  if (slots == 8) {
    return lowerGroupSlotLoadSlots8(op, source, offset, sourceGroupStride,
                                    resultVMIType, resultTypes, numGroups,
                                    rewriter, results);
  }
  if (slots == 1) {
    return lowerGroupSlotLoadSlots1(op, source, offset, sourceGroupStride,
                                    resultVMIType, resultTypes, rewriter,
                                    results);
  }
  return failure();
}

static LogicalResult
validateGroupBroadcastMappingDivisors(Operation *op, int64_t groupSize,
                                      int64_t selectorPeriod,
                                      int64_t sourceSlots,
                                      int64_t lanesPerPart,
                                      bool requiresSelectorPeriod,
                                      OneToNPatternRewriter &rewriter);

static FailureOr<std::optional<int64_t>> mapSlots1GroupBroadcastLane(
    Operation *op, VMIVRegType resultVMIType, int64_t part, int64_t chunk,
    int64_t lane, int64_t firstGroup, int64_t groupSize,
    int64_t selectorPeriod, int64_t sourcePartCount,
    OneToNPatternRewriter &rewriter);

static FailureOr<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
mapSlots1GroupBroadcastSources(
    Operation *op, VMIVRegType resultVMIType,
    ValueRange sourceParts, int64_t part, int64_t chunk, int64_t firstGroup,
    int64_t groupSize, int64_t selectorPeriod, int64_t lanesPerPart,
    OneToNPatternRewriter &rewriter) {
  if (failed(validateGroupBroadcastMappingDivisors(
          op, groupSize, selectorPeriod, /*sourceSlots=*/1, lanesPerPart,
          /*requiresSelectorPeriod=*/true, rewriter))) {
    return failure();
  }
  SmallVector<int64_t> laneSourceChunks(lanesPerPart, -1);
  SmallVector<int64_t> activeSourceChunks;
  for (int64_t lane = 0; lane < lanesPerPart; ++lane) {
    FailureOr<std::optional<int64_t>> sourceChunk =
        mapSlots1GroupBroadcastLane(
            op, resultVMIType, part, chunk, lane, firstGroup, groupSize,
            selectorPeriod, static_cast<int64_t>(sourceParts.size()), rewriter);
    if (failed(sourceChunk)) {
      return failure();
    }
    if (!*sourceChunk) {
      continue;
    }
    laneSourceChunks[lane] = **sourceChunk;
    bool isNewSourceChunk =
        llvm::find(activeSourceChunks, **sourceChunk) == activeSourceChunks.end();
    if (isNewSourceChunk) {
      activeSourceChunks.push_back(**sourceChunk);
    }
  }
  if (activeSourceChunks.empty()) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast result chunk has no active lanes");
  }
  return std::make_pair(std::move(laneSourceChunks),
                        std::move(activeSourceChunks));
}

static FailureOr<Value> materializeSlots1GroupBroadcastMerge(
    Operation *op, Type resultType, ValueRange sourceParts, int64_t lanesPerPart,
    MaskType resultMaskType, ArrayRef<int64_t> laneSourceChunks,
    ArrayRef<int64_t> activeSourceChunks, OneToNPatternRewriter &rewriter,
    Value allMask);

static FailureOr<Value> materializeSlots1GroupBroadcastChunk(
    Operation *op, Type resultType, VMIVRegType resultVMIType,
    ValueRange sourceParts, int64_t part, int64_t chunk, int64_t firstGroup,
    int64_t groupSize, int64_t selectorPeriod, int64_t lanesPerPart,
    OneToNPatternRewriter &rewriter, Value allMask) {
  auto resultVRegType = dyn_cast<VRegType>(resultType);
  if (!resultVRegType) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast requires uniform physical vreg types");
  }
  FailureOr<MaskType> resultMaskType =
      getMaskTypeForVReg(resultVRegType, rewriter.getContext());
  if (failed(resultMaskType)) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast cannot derive result mask type");
  }
  FailureOr<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>> mapping =
      mapSlots1GroupBroadcastSources(op, resultVMIType, sourceParts,
                                     part, chunk, firstGroup, groupSize,
                                     selectorPeriod, lanesPerPart, rewriter);
  if (failed(mapping)) {
    return failure();
  }
  return materializeSlots1GroupBroadcastMerge(
      op, resultType, sourceParts, lanesPerPart, *resultMaskType,
      mapping->first, mapping->second, rewriter, allMask);
}

static FailureOr<Value> materializeSlots1GroupBroadcastMerge(
    Operation *op, Type resultType, ValueRange sourceParts, int64_t lanesPerPart,
    MaskType resultMaskType, ArrayRef<int64_t> laneSourceChunks,
    ArrayRef<int64_t> activeSourceChunks, OneToNPatternRewriter &rewriter,
    Value allMask) {
  auto splatSource = [&rewriter, op, resultType, sourceParts, allMask](
                         int64_t chunkIndex) {
    return rewriter
        .create<VdupOp>(op->getLoc(), resultType, sourceParts[chunkIndex],
                        allMask, rewriter.getStringAttr("LOWEST"))
        .getResult();
  };
  Value merged = splatSource(activeSourceChunks.front());
  for (int64_t chunkIndex : llvm::drop_begin(activeSourceChunks)) {
    SmallVector<int8_t> laneMaskBits(lanesPerPart, 0);
    for (auto [lane, laneSourceChunk] : llvm::enumerate(laneSourceChunks)) {
      if (laneSourceChunk == chunkIndex) {
        laneMaskBits[lane] = 1;
      }
    }
    FailureOr<Value> laneMask = materializeConstantMaskChunk(
        op->getLoc(), resultMaskType, laneMaskBits, rewriter);
    if (failed(laneMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create group_broadcast source merge mask");
    }
    Value splat = splatSource(chunkIndex);
    merged = rewriter
                 .create<VselOp>(op->getLoc(), resultType, splat, merged,
                                 *laneMask)
                 .getResult();
  }
  return merged;
}

enum class GroupBroadcastSelectorKind { Constant, LogicalRamp, VCGBlockRamp };

struct GroupBroadcastSelectorPlan {
  GroupBroadcastSelectorKind kind;
  int64_t period;
};

static LogicalResult validateGroupBroadcastMappingDivisors(
    Operation *op, int64_t groupSize, int64_t selectorPeriod,
    int64_t sourceSlots, int64_t lanesPerPart, bool requiresSelectorPeriod,
    OneToNPatternRewriter &rewriter) {
  if (groupSize <= 0 || sourceSlots <= 0 || lanesPerPart <= 0 ||
      (requiresSelectorPeriod && selectorPeriod <= 0)) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast requires positive mapping divisors");
  }
  return success();
}

static FailureOr<std::optional<int64_t>> mapSlots1GroupBroadcastLane(
    Operation *op, VMIVRegType resultVMIType, int64_t part, int64_t chunk,
    int64_t lane, int64_t firstGroup, int64_t groupSize,
    int64_t selectorPeriod, int64_t sourcePartCount,
    OneToNPatternRewriter &rewriter) {
  FailureOr<bool> padding = isPaddingLane(resultVMIType, part, chunk, lane);
  if (failed(padding)) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast failed to map result padding lanes");
  }
  if (*padding) {
    return std::optional<int64_t>();
  }
  FailureOr<int64_t> logical =
      mapPhysicalLaneToLogical(resultVMIType, part, chunk, lane);
  if (failed(logical)) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast failed to map a result lane");
  }
  int64_t safeGroupSize = groupSize > 0 ? groupSize : 1;
  int64_t safeSelectorPeriod = selectorPeriod > 0 ? selectorPeriod : 1;
  int64_t actualGroup = *logical / safeGroupSize;
  int64_t expectedGroup = firstGroup + lane / safeSelectorPeriod;
  if (actualGroup != expectedGroup) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast layout table row does not match its selector "
            "lowering plan");
  }
  if (actualGroup < 0 || actualGroup >= sourcePartCount) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast source chunk is out of range");
  }
  return std::optional<int64_t>(actualGroup);
}

static LogicalResult verifyGroupBroadcastChunkMapping(
    Operation *op, VMIVRegType resultVMIType,
    GroupBroadcastSelectorKind selectorKind, int64_t selectorPeriod,
    int64_t sourceSlots, int64_t groupSize, int64_t part, int64_t chunk,
    int64_t firstGroup, int64_t sourceChunk, int64_t lanesPerPart,
    OneToNPatternRewriter &rewriter) {
  bool requiresSelectorPeriod =
      selectorKind != GroupBroadcastSelectorKind::Constant;
  if (failed(validateGroupBroadcastMappingDivisors(
          op, groupSize, selectorPeriod, sourceSlots, lanesPerPart,
          requiresSelectorPeriod, rewriter))) {
    return failure();
  }
  int64_t safeGroupSize = groupSize;
  int64_t safeSelectorPeriod = selectorPeriod > 0 ? selectorPeriod : 1;
  int64_t safeSourceSlots = sourceSlots;
  for (int64_t lane = 0; lane < lanesPerPart; ++lane) {
    FailureOr<bool> padding =
        isPaddingLane(resultVMIType, part, chunk, lane);
    if (failed(padding)) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast failed to map result padding lanes");
    }
    if (*padding) {
      continue;
    }
    FailureOr<int64_t> logical =
        mapPhysicalLaneToLogical(resultVMIType, part, chunk, lane);
    if (failed(logical)) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast failed to map a result lane");
    }
    int64_t actualGroup = *logical / safeGroupSize;
    int64_t expectedGroup = firstGroup;
    if (selectorKind != GroupBroadcastSelectorKind::Constant) {
      expectedGroup += lane / safeSelectorPeriod;
    }
    if (actualGroup != expectedGroup ||
        actualGroup / safeSourceSlots != sourceChunk) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast layout table row does not match its selector "
              "lowering plan");
    }
  }
  return success();
}

static FailureOr<GroupBroadcastSelectorPlan> chooseGroupBroadcastSelectorPlan(
    Operation *op, const VMIGroupBroadcastLayoutFact &fact,
    VMILayoutAttr resultLayout, OneToNPatternRewriter &rewriter) {
  if (fact.blockClass == VMIGroupBlockClass::FullPartMultiple) {
    return GroupBroadcastSelectorPlan{GroupBroadcastSelectorKind::Constant, 0};
  }
  bool isUnitStrideContiguous =
      resultLayout.isContiguous() && resultLayout.getLaneStride() == 1;
  if (isUnitStrideContiguous) {
    return GroupBroadcastSelectorPlan{GroupBroadcastSelectorKind::LogicalRamp,
                                      fact.groupSize};
  }
  bool isStridedContiguous =
      resultLayout.isContiguous() && resultLayout.getLaneStride() > 1;
  if (isStridedContiguous) {
    return GroupBroadcastSelectorPlan{
        GroupBroadcastSelectorKind::VCGBlockRamp,
        fact.groupSize * resultLayout.getLaneStride()};
  }
  bool isDeinterleaved =
      resultLayout.isDeinterleaved() || resultLayout.isBlockDeinterleaved();
  if (isDeinterleaved) {
    return GroupBroadcastSelectorPlan{GroupBroadcastSelectorKind::VCGBlockRamp,
                                      fact.vcgBlockElems};
  }
  (void)rewriter.notifyMatchFailure(
      op, "group_broadcast layout table row has no selector lowering plan");
  return failure();
}

struct GroupBroadcastSelectorContext {
  Operation *op;
  GroupBroadcastSelectorKind kind;
  int64_t sourceLaneStride;
  std::optional<int64_t> selectorShift;
  std::optional<int64_t> sourceLaneStrideShift;
  Type indexScalarType;
  VRegType indexType;
  Value allMask;
  Value sharedRamp;
  llvm::DenseMap<int64_t, Value> selectorByBaseIndex;
};

struct GroupBroadcastLoweringContext {
  GroupBroadcastSelectorContext selector;
  int64_t sourceSlots;
  int64_t selectorPeriod;
};

struct GroupBroadcastSelectorMetadata {
  int64_t sourceSlots;
  int64_t sourceLaneStride;
  GroupBroadcastSelectorPlan selectorPlan;
  std::optional<int64_t> selectorShift;
  std::optional<int64_t> sourceLaneStrideShift;
};

static FailureOr<GroupBroadcastSelectorMetadata>
getGroupBroadcastSelectorMetadata(
    Operation *op, VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
    const VMIGroupBroadcastLayoutFact &fact,
    OneToNPatternRewriter &rewriter) {
  VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
  VMILayoutAttr sourceLayout = sourceVMIType.getLayoutAttr();
  int64_t sourceSlots = sourceLayout.getSlots();
  int64_t sourceLaneStride = sourceLayout.getLaneStride();
  if (sourceSlots <= 0 || sourceLaneStride <= 0) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast requires explicit positive source group slots");
  }
  FailureOr<GroupBroadcastSelectorPlan> selectorPlan =
      chooseGroupBroadcastSelectorPlan(op, fact, resultLayout, rewriter);
  if (failed(selectorPlan)) {
    return failure();
  }
  std::optional<int64_t> selectorShift;
  std::optional<int64_t> sourceLaneStrideShift;
  if (selectorPlan->kind != GroupBroadcastSelectorKind::Constant) {
    selectorShift = getPowerOfTwoLog2(selectorPlan->period);
    sourceLaneStrideShift = getPowerOfTwoLog2(sourceLaneStride);
    if (!selectorShift || !sourceLaneStrideShift) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast ramp requires power-of-two selector period "
              "and source lane stride");
    }
  }
  return GroupBroadcastSelectorMetadata{
      sourceSlots, sourceLaneStride, *selectorPlan, selectorShift,
      sourceLaneStrideShift};
}

static FailureOr<GroupBroadcastSelectorContext>
createGroupBroadcastSelectorContext(
    Operation *op, VRegType sourceType,
    const GroupBroadcastSelectorMetadata &metadata,
    OneToNPatternRewriter &rewriter) {
  unsigned indexBits = pto::getPTOStorageElemBitWidth(
      sourceType.getElementType());
  auto indexElementType = IntegerType::get(
      rewriter.getContext(), indexBits,
      IntegerType::SignednessSemantics::Unsigned);
  auto indexScalarType = IntegerType::get(rewriter.getContext(), indexBits);
  auto indexType = VRegType::get(rewriter.getContext(),
                                 sourceType.getElementCount(), indexElementType);
  FailureOr<Value> allMask =
      createAllTrueMaskForVReg(op->getLoc(), indexType, rewriter);
  if (failed(allMask)) {
    return rewriter.notifyMatchFailure(
        op, "failed to create group_broadcast all mask");
  }
  return GroupBroadcastSelectorContext{
      op,
      metadata.selectorPlan.kind,
      metadata.sourceLaneStride,
      metadata.selectorShift,
      metadata.sourceLaneStrideShift,
      indexScalarType,
      indexType,
      *allMask,
      Value(),
      llvm::DenseMap<int64_t, Value>()};
}

static FailureOr<Value> materializeConstantGroupBroadcastSelector(
    GroupBroadcastSelectorContext &context, int64_t baseIndex,
    OneToNPatternRewriter &rewriter) {
  FailureOr<Value> baseScalar = createScalarOffsetConstant(
      context.op->getLoc(), context.indexScalarType, baseIndex, rewriter);
  if (failed(baseScalar)) {
    return failure();
  }
  return rewriter
      .create<VdupOp>(context.op->getLoc(), context.indexType, *baseScalar,
                      context.allMask, /*position=*/nullptr)
      .getResult();
}

static FailureOr<Value> materializeGroupBroadcastRamp(
    GroupBroadcastSelectorContext &context, int64_t baseIndex,
    OneToNPatternRewriter &rewriter) {
  if (!context.sharedRamp) {
    FailureOr<Value> zero = createScalarOffsetConstant(
        context.op->getLoc(), context.indexScalarType, 0, rewriter);
    if (failed(zero)) {
      return failure();
    }
    context.sharedRamp =
        rewriter.create<VciOp>(context.op->getLoc(), context.indexType, *zero,
                               StringAttr{})
            .getResult();
    if (*context.selectorShift != 0) {
      Value shift = createI16Constant(context.op->getLoc(),
                                      *context.selectorShift, rewriter);
      context.sharedRamp =
          rewriter
              .create<VshrsOp>(context.op->getLoc(), context.indexType,
                               context.sharedRamp, shift, context.allMask)
              .getResult();
    }
    if (*context.sourceLaneStrideShift != 0) {
      Value shift = createI16Constant(
          context.op->getLoc(), *context.sourceLaneStrideShift, rewriter);
      context.sharedRamp =
          rewriter
              .create<VshlsOp>(context.op->getLoc(), context.indexType,
                               context.sharedRamp, shift, context.allMask)
              .getResult();
    }
  }
  Value selector = context.sharedRamp;
  if (baseIndex != 0) {
    FailureOr<Value> baseScalar = createScalarOffsetConstant(
        context.op->getLoc(), context.indexScalarType, baseIndex, rewriter);
    if (failed(baseScalar)) {
      return failure();
    }
    selector = rewriter
                   .create<VaddsOp>(context.op->getLoc(), context.indexType,
                                    selector, *baseScalar, context.allMask)
                   .getResult();
  }
  return selector;
}

static FailureOr<GroupBroadcastLoweringContext>
createGroupBroadcastLoweringContext(
    Operation *op, VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
    const VMIGroupBroadcastLayoutFact &fact, VRegType sourceType,
    OneToNPatternRewriter &rewriter) {
  FailureOr<GroupBroadcastSelectorMetadata> metadata =
      getGroupBroadcastSelectorMetadata(op, sourceVMIType, resultVMIType, fact,
                                        rewriter);
  if (failed(metadata)) {
    return failure();
  }
  FailureOr<GroupBroadcastSelectorContext> selectorContext =
      createGroupBroadcastSelectorContext(op, sourceType, *metadata, rewriter);
  if (failed(selectorContext)) {
    return failure();
  }
  return GroupBroadcastLoweringContext{std::move(*selectorContext),
                                       metadata->sourceSlots,
                                       metadata->selectorPlan.period};
}

static FailureOr<Value> getGroupBroadcastSelector(
    GroupBroadcastSelectorContext &context, int64_t baseSlot,
    OneToNPatternRewriter &rewriter) {
  int64_t baseIndex = baseSlot * context.sourceLaneStride;
  auto cached = context.selectorByBaseIndex.find(baseIndex);
  if (cached != context.selectorByBaseIndex.end()) {
    return cached->second;
  }

  if (context.kind == GroupBroadcastSelectorKind::Constant) {
    FailureOr<Value> selector = materializeConstantGroupBroadcastSelector(
        context, baseIndex, rewriter);
    if (failed(selector)) {
      return failure();
    }
    context.selectorByBaseIndex.try_emplace(baseIndex, *selector);
    return *selector;
  }
  FailureOr<Value> selector =
      materializeGroupBroadcastRamp(context, baseIndex, rewriter);
  if (failed(selector)) {
    return failure();
  }
  context.selectorByBaseIndex.try_emplace(baseIndex, *selector);
  return *selector;
}

static FailureOr<Value> materializeGroupBroadcastChunk(
    Operation *op, Type resultType, VMIVRegType resultVMIType,
    ValueRange sourceParts,
    GroupBroadcastSelectorKind selectorKind, int64_t selectorPeriod,
    int64_t sourceSlots, int64_t groupSize, int64_t part, int64_t chunk,
    int64_t firstGroup, int64_t sourceChunk, int64_t baseSlot,
    int64_t lanesPerPart, Value allMask,
    llvm::function_ref<FailureOr<Value>(int64_t)> getSelector,
    OneToNPatternRewriter &rewriter) {
  auto resultVRegType = dyn_cast<VRegType>(resultType);
  if (!resultVRegType) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast requires uniform physical vreg types");
  }
  if (failed(verifyGroupBroadcastChunkMapping(
          op, resultVMIType, selectorKind, selectorPeriod, sourceSlots,
          groupSize, part, chunk, firstGroup, sourceChunk, lanesPerPart,
          rewriter))) {
    return failure();
  }

  if (selectorKind == GroupBroadcastSelectorKind::Constant && sourceSlots == 1) {
    return rewriter
        .create<VdupOp>(op->getLoc(), resultType, sourceParts[sourceChunk],
                        allMask, rewriter.getStringAttr("LOWEST"))
        .getResult();
  }
  FailureOr<Value> selector = getSelector(baseSlot);
  if (failed(selector)) {
    return rewriter.notifyMatchFailure(
        op, "failed to create group_broadcast selector ramp");
  }
  return rewriter
      .create<VselrOp>(op->getLoc(), resultType, sourceParts[sourceChunk],
                       *selector)
      .getResult();
}

static FailureOr<Value> lowerGroupBroadcastChunk(
    Operation *op, Type resultType, VMIVRegType resultVMIType,
    ValueRange sourceParts, const VMIGroupBroadcastLayoutFact &fact,
    GroupBroadcastLoweringContext &context, int64_t part, int64_t chunk,
    OneToNPatternRewriter &rewriter) {
  FailureOr<int64_t> firstLogical =
      mapPhysicalLaneToLogical(resultVMIType, part, chunk, 0);
  if (failed(firstLogical)) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast failed to map the first result lane");
  }
  int64_t firstGroup = *firstLogical / fact.groupSize;
  int64_t sourceChunk = firstGroup / context.sourceSlots;
  int64_t baseSlot = firstGroup % context.sourceSlots;
  bool sourceChunkOutOfRange =
      sourceChunk < 0 || sourceChunk >= static_cast<int64_t>(sourceParts.size());
  if (sourceChunkOutOfRange) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast source chunk is out of range");
  }
  Value allMask = context.selector.allMask;
  if (context.sourceSlots == 1 &&
      context.selector.kind != GroupBroadcastSelectorKind::Constant) {
    return materializeSlots1GroupBroadcastChunk(
        op, resultType, resultVMIType, sourceParts, part, chunk, firstGroup,
        fact.groupSize, context.selectorPeriod, fact.lanesPerPart, rewriter,
        allMask);
  }
  if (failed(verifyGroupBroadcastChunkMapping(
          op, resultVMIType, context.selector.kind, context.selectorPeriod,
          context.sourceSlots, fact.groupSize, part, chunk, firstGroup,
          sourceChunk, fact.lanesPerPart, rewriter))) {
    return failure();
  }
  auto getSelector = [&context, &rewriter](int64_t slot) {
    return getGroupBroadcastSelector(context.selector, slot, rewriter);
  };
  return materializeGroupBroadcastChunk(
      op, resultType, resultVMIType, sourceParts, context.selector.kind,
      context.selectorPeriod, context.sourceSlots, fact.groupSize, part, chunk,
      firstGroup, sourceChunk, baseSlot, fact.lanesPerPart, allMask, getSelector,
      rewriter);
}

static FailureOr<VRegType> validateGroupBroadcastSources(
    Operation *op, ValueRange sourceParts,
    const VMIGroupBroadcastLayoutFact &fact,
    OneToNPatternRewriter &rewriter) {
  auto firstSourceType = dyn_cast<VRegType>(sourceParts.front().getType());
  if (!firstSourceType) {
    return rewriter.notifyMatchFailure(op,
                                       "group_broadcast source must be vreg");
  }
  bool hasNonUniformSourceType =
      llvm::any_of(sourceParts, [&firstSourceType](Value sourcePart) {
        return sourcePart.getType() != firstSourceType;
      });
  if (hasNonUniformSourceType) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast requires uniform physical source vreg types");
  }
  bool sourceLaneCountMismatch =
      firstSourceType.getElementCount() != fact.lanesPerPart;
  if (sourceLaneCountMismatch) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast physical source lanes do not match the supported "
            "layout row");
  }
  unsigned indexBits =
      pto::getPTOStorageElemBitWidth(firstSourceType.getElementType());
  bool unsupportedIndexBits =
      indexBits != 8 && indexBits != 16 && indexBits != 32;
  if (unsupportedIndexBits) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast requires 8/16/32-bit index elements");
  }
  return firstSourceType;
}

static FailureOr<Value> lowerGroupBroadcastResultChunk(
    Operation *op, Type resultType, VMIVRegType resultVMIType,
    ValueRange sourceParts, const VMIGroupBroadcastLayoutFact &fact,
    GroupBroadcastLoweringContext &context, VRegType expectedSourceType,
    int64_t part, int64_t chunk, OneToNPatternRewriter &rewriter);

static LogicalResult lowerGroupBroadcastResultChunks(
    Operation *op, ValueRange sourceParts, VMIVRegType resultVMIType,
    TypeRange resultTypes, const VMIGroupBroadcastLayoutFact &fact,
    GroupBroadcastLoweringContext &context, VRegType expectedSourceType,
    OneToNPatternRewriter &rewriter, SmallVectorImpl<Value> &results) {
  FailureOr<int64_t> resultLayoutFactor = getDataLayoutFactor(resultVMIType);
  if (failed(resultLayoutFactor)) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast requires a computable result layout factor");
  }
  results.clear();
  results.resize(resultTypes.size());
  int64_t flatIndex = 0;
  for (int64_t part = 0; part < *resultLayoutFactor; ++part) {
    FailureOr<int64_t> chunks =
        *resultLayoutFactor == 1
            ? FailureOr<int64_t>(resultTypes.size())
            : getDataChunksInPart(resultVMIType, part);
    if (failed(chunks)) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast failed to enumerate result chunks");
    }
    for (int64_t chunk = 0; chunk < *chunks; ++chunk, ++flatIndex) {
      if (flatIndex >= static_cast<int64_t>(resultTypes.size())) {
        return rewriter.notifyMatchFailure(
            op, "group_broadcast physical result count is too small");
      }
      FailureOr<Value> chunkResult = lowerGroupBroadcastResultChunk(
          op, resultTypes[flatIndex], resultVMIType, sourceParts, fact, context,
          expectedSourceType, part, chunk, rewriter);
      if (failed(chunkResult)) {
        return failure();
      }
      results[flatIndex] = *chunkResult;
    }
  }
  if (flatIndex != static_cast<int64_t>(resultTypes.size())) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast physical result count is too large");
  }
  return success();
}

static FailureOr<Value> lowerGroupBroadcastResultChunk(
    Operation *op, Type resultType, VMIVRegType resultVMIType,
    ValueRange sourceParts, const VMIGroupBroadcastLayoutFact &fact,
    GroupBroadcastLoweringContext &context, VRegType expectedSourceType,
    int64_t part, int64_t chunk, OneToNPatternRewriter &rewriter) {
  auto resultVRegType = dyn_cast<VRegType>(resultType);
  bool mismatchedResultType =
      !resultVRegType || resultVRegType != expectedSourceType;
  if (mismatchedResultType) {
    return rewriter.notifyMatchFailure(
        op, "group_broadcast requires uniform physical vreg types");
  }
  return lowerGroupBroadcastChunk(op, resultType, resultVMIType, sourceParts,
                                  fact, context, part, chunk, rewriter);
}

static LogicalResult lowerGroupBroadcastParts(
    Operation *op, ValueRange sourceParts, VMIVRegType sourceVMIType,
    VMIVRegType resultVMIType, TypeRange resultTypes, int64_t numGroups,
    OneToNPatternRewriter &rewriter, SmallVectorImpl<Value> &results) {
  bool emptyArity = sourceParts.empty() || resultTypes.empty();
  if (emptyArity) {
    return rewriter.notifyMatchFailure(op, "group_broadcast arity mismatch");
  }

  std::string layoutReason;
  VMILayoutSupport supports;
  FailureOr<VMIGroupBroadcastLayoutFact> fact =
      supports.getGroupBroadcastLayoutFactForLayouts(
          sourceVMIType, resultVMIType, numGroups, &layoutReason);
  if (failed(fact)) {
    return rewriter.notifyMatchFailure(
        op, Twine("group_broadcast requires a supported layout table row; ") +
                layoutReason);
  }

  FailureOr<VRegType> firstSourceType =
      validateGroupBroadcastSources(op, sourceParts, *fact, rewriter);
  if (failed(firstSourceType)) {
    return failure();
  }
  FailureOr<GroupBroadcastLoweringContext> loweringContext =
      createGroupBroadcastLoweringContext(
          op, sourceVMIType, resultVMIType, *fact, *firstSourceType, rewriter);
  if (failed(loweringContext)) {
    return failure();
  }
  GroupBroadcastLoweringContext &context = *loweringContext;
  return lowerGroupBroadcastResultChunks(
      op, sourceParts, resultVMIType, resultTypes, *fact, context,
      *firstSourceType, rewriter, results);
}

struct OneToNVMIGroupSlotLoadOpPattern
    : OneToNOpConversionPattern<VMIGroupSlotLoadOp> {
  using OneToNOpConversionPattern<
      VMIGroupSlotLoadOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIGroupSlotLoadOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    VMILayoutAttr layout = resultVMIType.getLayoutAttr();
    bool invalidLayout =
        !layout || !layout.isGroupSlots() || layout.getSlots() <= 0;
    if (invalidLayout) {
      return rewriter.notifyMatchFailure(
          op, "group_slot_load requires explicit group_slots layout");
    }

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
    bool invalidOperands =
        failed(source) || failed(offset) || failed(sourceGroupStride);
    if (invalidOperands) {
      return failure();
    }

    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }

    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    int64_t numGroups = op.getNumGroupsAttr().getInt();

    SmallVector<Value> results;
    if (failed(lowerGroupSlotLoadParts(op, *source, *offset, *sourceGroupStride,
                                       resultVMIType, resultTypes, numGroups,
                                       rewriter, results))) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIMaskedLoadOpPattern
    : OneToNOpConversionPattern<VMIMaskedLoadOp> {
  using OneToNOpConversionPattern<VMIMaskedLoadOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> materializeMaskedLoadPart(
      VMIMaskedLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, Value mask, Value passthru, Type resultType,
      int64_t index, int64_t lanesPerPart) const {
    bool invalidPartTypes = !isa<MaskType>(mask.getType()) ||
                            passthru.getType() != resultType ||
                            !isa<VRegType>(resultType);
    if (invalidPartTypes) {
      return rewriter.notifyMatchFailure(
          op, "masked_load physical part type mismatch");
    }
    Value chunkOffset = createChunkOffset(
        op.getLoc(), offset, index * lanesPerPart, rewriter);
    Value loaded = rewriter
                       .create<VldsOp>(op.getLoc(), resultType, Type{}, source,
                                       chunkOffset, nullptr)
                       .getResult();
    return rewriter
        .create<VselOp>(op.getLoc(), resultType, loaded, passthru, mask)
        .getResult();
  }

  LogicalResult lowerPhysicalParts(
      VMIMaskedLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, ValueRange maskParts, ValueRange passthruParts,
      ArrayRef<Type> resultTypes, int64_t lanesPerPart) const {
    bool arityMismatch = maskParts.size() != passthruParts.size() ||
                         passthruParts.size() != resultTypes.size();
    if (arityMismatch) {
      return rewriter.notifyMatchFailure(op,
                                         "masked_load physical arity mismatch");
    }
    return lowerPointwisePhysicalParts(
        op, resultTypes, "masked_load physical arity mismatch", rewriter,
        [&](int64_t index, Type resultType) -> FailureOr<Value> {
          return materializeMaskedLoadPart(op, rewriter, source, offset,
                                           maskParts[index],
                                           passthruParts[index], resultType,
                                           index, lanesPerPart);
        },
        *this->getTypeConverter());
  }

public:

  LogicalResult
  matchAndRewrite(VMIMaskedLoadOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    FailureOr<Value> source = getSingleValue(
        op, adaptor.getSource(), "masked_load source must convert to one value",
        rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(), "masked_load offset must convert to one value",
        rewriter);
    bool failedOperands = failed(source) || failed(offset);
    if (failedOperands) {
      return failure();
    }

    FailureOr<int64_t> lanesPerPart = verifyFullOrSafeReadVRegChunks(
        op, resultVMIType, op.getSource(), op.getOffset(), rewriter);
    if (failed(lanesPerPart)) {
      return failure();
    }

    ValueRange maskParts = adaptor.getMask();
    ValueRange passthruParts = adaptor.getPassthru();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    bool failedResultTypeConversion = failed(maybe_resultTypes);
    if (failedResultTypeConversion) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    return lowerPhysicalParts(op, rewriter, *source, *offset, maskParts,
                              passthruParts, resultTypes, *lanesPerPart);
  }
};

struct OneToNVMIGatherOpPattern : OneToNOpConversionPattern<VMIGatherOp> {
  using OneToNOpConversionPattern<VMIGatherOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> materializeGatherPart(
      VMIGatherOp op, OneToNPatternRewriter &rewriter, Value source,
      Value indices, Value mask, Value passthru, Type resultType,
      bool allActive) const {
    bool invalidPartTypes = !isa<VRegType>(indices.getType()) ||
                            !isa<MaskType>(mask.getType()) ||
                            passthru.getType() != resultType ||
                            !isa<VRegType>(resultType);
    if (invalidPartTypes) {
      return rewriter.notifyMatchFailure(
          op, "gather physical part type mismatch");
    }
    unsigned resultBits = pto::getPTOStorageElemBitWidth(
        cast<VRegType>(resultType).getElementType());
    Value gathered = resultBits == 16
                         ? rewriter.create<Vgather2Op>(
                               op.getLoc(), resultType, source, indices, mask)
                               .getResult()
                         : rewriter.create<Vgather2BcOp>(
                               op.getLoc(), resultType, source, indices, mask)
                               .getResult();
    if (allActive) {
      return gathered;
    }
    return rewriter
        .create<VselOp>(op.getLoc(), resultType, gathered, passthru, mask)
        .getResult();
  }

  LogicalResult lowerPhysicalParts(
      VMIGatherOp op, OneToNPatternRewriter &rewriter, Value source,
      ValueRange indicesParts, ValueRange maskParts, ValueRange passthruParts,
      ArrayRef<Type> resultTypes, bool allActive) const {
    bool arityMismatch = indicesParts.size() != maskParts.size() ||
                         indicesParts.size() != passthruParts.size() ||
                         indicesParts.size() != resultTypes.size();
    if (arityMismatch) {
      return rewriter.notifyMatchFailure(op, "gather physical arity mismatch");
    }
    return lowerPointwisePhysicalParts(
        op, resultTypes, "gather physical arity mismatch", rewriter,
        [&](int64_t index, Type resultType) -> FailureOr<Value> {
          return materializeGatherPart(op, rewriter, source,
                                       indicesParts[index], maskParts[index],
                                       passthruParts[index], resultType,
                                       allActive);
        },
        *this->getTypeConverter());
  }

public:

  LogicalResult
  matchAndRewrite(VMIGatherOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<Value> source =
        getSingleValue(op, adaptor.getSource(),
                       "gather source must convert to one value", rewriter);
    if (failed(source)) {
      return failure();
    }

    ValueRange indicesParts = adaptor.getIndices();
    ValueRange maskParts = adaptor.getMask();
    ValueRange passthruParts = adaptor.getPassthru();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    bool failedResultTypeConversion = failed(maybe_resultTypes);
    if (failedResultTypeConversion) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);

    // Static all-active masks select gathered[0] for every lane, so the
    // trailing vsel is a semantic no-op. Skip it and keep gathered directly.
    // Non-static masks still take the original gather + vsel path.
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    bool allActive = isStaticAllActiveMask(op.getMask(),
                                           resultVMIType.getElementCount());


    return lowerPhysicalParts(op, rewriter, *source, indicesParts, maskParts,
                              passthruParts, resultTypes, allActive);
  }
};

struct OneToNVMIExpandLoadOpPattern
    : OneToNOpConversionPattern<VMIExpandLoadOp> {
  using OneToNOpConversionPattern<VMIExpandLoadOp>::OneToNOpConversionPattern;

private:
  struct RuntimeExpandLoadPlan {
    VRegType resultType;
    Value gatherBase;
    Value mask;
    Value passthru;
  };

  FailureOr<RuntimeExpandLoadPlan> buildRuntimeExpandLoadPlan(
      VMIExpandLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, ValueRange maskParts, ValueRange passthruParts,
      ArrayRef<Type> resultTypes) const {
    bool invalidRuntimeArity = resultTypes.size() != 1 || maskParts.size() != 1 ||
                               passthruParts.size() != 1;
    if (invalidRuntimeArity) {
      return rewriter.notifyMatchFailure(
          op, "runtime expand_load supports only one physical chunk");
    }
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    bool invalidRuntimeTypes =
        !resultType || !maskType || passthruParts.front().getType() != resultType;
    if (invalidRuntimeTypes) {
      return rewriter.notifyMatchFailure(
          op, "runtime expand_load requires physical result/passthru/mask");
    }
    if (!isa<PtrType>(source.getType())) {
      return rewriter.notifyMatchFailure(op, "runtime expand_load requires ptr");
    }
    Value gatherBase = rewriter
                           .create<AddPtrOp>(op.getLoc(), source.getType(), source,
                                             offset)
                           .getResult();
    return RuntimeExpandLoadPlan{resultType, gatherBase, maskParts.front(),
                                 passthruParts.front()};
  }

  FailureOr<Value> materializeRuntimeExpandLoad(
      VMIExpandLoadOp op, OneToNPatternRewriter &rewriter,
      const RuntimeExpandLoadPlan &plan) const {
    auto indexType = VRegType::get(rewriter.getContext(),
                                   plan.resultType.getElementCount(),
                                   rewriter.getI32Type());
    FailureOr<Value> indexSeedMask =
        createAllTrueMaskForVReg(op.getLoc(), indexType, rewriter);
    if (failed(indexSeedMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create runtime expand_load index seed mask");
    }
    Value zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 32);
    Value carrier = rewriter
                        .create<VdupOp>(op.getLoc(), indexType, zero,
                                       *indexSeedMask, /*position=*/nullptr)
                        .getResult();
    Value indices = rewriter
                        .create<VusqzOp>(op.getLoc(), indexType, carrier,
                                        plan.mask)
                        .getResult();
    Value gathered = rewriter
                         .create<Vgather2BcOp>(op.getLoc(), plan.resultType,
                                               plan.gatherBase, indices,
                                               plan.mask)
                         .getResult();
    return rewriter
        .create<VselOp>(op.getLoc(), plan.resultType, gathered, plan.passthru,
                        plan.mask)
        .getResult();
  }

  FailureOr<Value> materializeStaticExpandLoadPart(
      VMIExpandLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, Type resultType, int64_t index, int64_t lanesPerPart) const {
    if (!isa<VRegType>(resultType)) {
      return rewriter.notifyMatchFailure(op, "expand_load result must be vreg");
    }
    Value chunkOffset = createChunkOffset(
        op.getLoc(), offset, index * lanesPerPart, rewriter);
    return rewriter
        .create<VldsOp>(op.getLoc(), resultType, Type{}, source, chunkOffset,
                        nullptr)
        .getResult();
  }

  LogicalResult lowerRuntimeExpandLoad(
      VMIExpandLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, ValueRange maskParts, ValueRange passthruParts,
      ArrayRef<Type> resultTypes) const {
    FailureOr<RuntimeExpandLoadPlan> plan = buildRuntimeExpandLoadPlan(
        op, rewriter, source, offset, maskParts, passthruParts, resultTypes);
    if (failed(plan)) {
      return failure();
    }
    FailureOr<Value> result = materializeRuntimeExpandLoad(op, rewriter, *plan);
    if (failed(result)) {
      return failure();
    }
    return replaceSinglePhysicalResult(rewriter, op, *result,
                                       *this->getTypeConverter());
  }

  LogicalResult lowerExpandLoadParts(
      VMIExpandLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, ValueRange maskParts, ValueRange passthruParts,
      ArrayRef<Type> resultTypes) const {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    return resultTypes.size() == 1
               ? lowerRuntimeExpandLoad(op, rewriter, source, offset, maskParts,
                                       passthruParts, resultTypes)
               : lowerStaticExpandLoad(op, rewriter, source, offset,
                                       sourceVMIType, resultTypes);
  }

  LogicalResult lowerStaticExpandLoad(
      VMIExpandLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, VMIVRegType resultVMIType, ArrayRef<Type> resultTypes) const {
    FailureOr<int64_t> lanesPerPart = verifyFullOrSafeReadVRegChunks(
        op, resultVMIType, op.getSource(), op.getOffset(), rewriter);
    if (failed(lanesPerPart)) {
      return failure();
    }
    return lowerPointwisePhysicalParts(
        op, resultTypes, "expand_load physical arity mismatch", rewriter,
        [&](int64_t index, Type resultType) -> FailureOr<Value> {
          return materializeStaticExpandLoadPart(op, rewriter, source, offset,
                                                 resultType, index,
                                                 *lanesPerPart);
        },
        *this->getTypeConverter());
  }

public:

  LogicalResult
  matchAndRewrite(VMIExpandLoadOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    FailureOr<Value> source = getSingleValue(
        op, adaptor.getSource(), "expand_load source must convert to one value",
        rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(), "expand_load offset must convert to one value",
        rewriter);
    bool operandsConverted = succeeded(source) && succeeded(offset);
    if (!operandsConverted) {
      return failure();
    }

    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }

    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    if (isStaticAllActiveMask(op.getMask(), resultVMIType.getElementCount())) {
      return lowerStaticExpandLoad(op, rewriter, *source, *offset,
                                   resultVMIType, resultTypes);
    }

    return lowerRuntimeExpandLoad(op, rewriter, *source, *offset,
                                  adaptor.getMask(), adaptor.getPassthru(),
                                  resultTypes);
  }
};

struct OneToNVMIStoreOpPattern : OneToNOpConversionPattern<VMIStoreOp> {
  using OneToNOpConversionPattern<VMIStoreOp>::OneToNOpConversionPattern;

private:
  struct StorePhysicalPlan {
    SmallVector<Type> contiguousTypes;
    VMILayoutAttr contiguousLayout;
    int64_t lanesPerPart;
    bool fullPhysicalChunks;
    bool noWiderThanContiguous;
  };

  struct StoreLoweringInput {
    Value destination;
    Value offset;
    ValueRange valueParts;
    VMIVRegType valueVMIType;
    StorePhysicalPlan plan;
  };

  FailureOr<StorePhysicalPlan> buildPhysicalPlan(
      VMIStoreOp op, ValueRange valueParts, VMIVRegType valueVMIType,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(valueVMIType.getElementType());
    if (failed(lanesPerPart)) {
      return rewriter.notifyMatchFailure(
          op, "store requires known physical lanes per part");
    }
    VMILayoutAttr contiguousLayout =
        VMILayoutAttr::getContiguous(rewriter.getContext());
    FailureOr<SmallVector<Type>> contiguousTypes =
        getContiguousStoreTypes(op, valueVMIType, rewriter);
    if (failed(contiguousTypes)) {
      return failure();
    }
    SmallVector<Type> valuePartTypes;
    valuePartTypes.reserve(valueParts.size());
    for (Value value : valueParts) {
      valuePartTypes.push_back(value.getType());
    }
    FailureOr<bool> noWiderThanContiguous =
        hasNoWiderFootprintThanContiguous(valuePartTypes, *contiguousTypes);
    if (failed(noWiderThanContiguous)) {
      return rewriter.notifyMatchFailure(
          op, "failed to compare store physical footprint");
    }
    return StorePhysicalPlan{std::move(*contiguousTypes), contiguousLayout,
                             *lanesPerPart,
                             succeeded(checkFullDataPhysicalChunks(
                                 valueVMIType, nullptr)),
                             *noWiderThanContiguous};
  }

  FailureOr<StoreLoweringInput> getLoweringInput(
      VMIStoreOp op, OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const {
    auto valueVMIType = cast<VMIVRegType>(op.getValue().getType());
    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "store destination must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(), "store offset must convert to one value",
        rewriter);
    bool invalidAddressOperands = failed(destination) || failed(offset);
    if (invalidAddressOperands) {
      return failure();
    }
    ValueRange valueParts = adaptor.getValue();
    FailureOr<StorePhysicalPlan> plan =
        buildPhysicalPlan(op, valueParts, valueVMIType, rewriter);
    if (failed(plan)) {
      return failure();
    }
    return StoreLoweringInput{*destination, *offset, valueParts, valueVMIType,
                              std::move(*plan)};
  }

  FailureOr<SmallVector<Type>> getContiguousStoreTypes(
      VMIStoreOp op, VMIVRegType valueVMIType,
      OneToNPatternRewriter &rewriter) const {
    VMILayoutAttr contiguousLayout =
        VMILayoutAttr::getContiguous(rewriter.getContext());
    FailureOr<SmallVector<Type>> contiguousTypes =
        getConvertedVRegTypesWithLayout(valueVMIType, contiguousLayout,
                                        *this->getTypeConverter());
    if (failed(contiguousTypes)) {
      return rewriter.notifyMatchFailure(
          op, "failed to compute contiguous store footprint");
    }
    return std::move(*contiguousTypes);
  }

  FailureOr<bool> tryLowerLaneStrideStore(
      VMIStoreOp op, Value destination, Value offset, ValueRange valueParts,
      VMIVRegType valueVMIType, OneToNPatternRewriter &rewriter) const {
    std::optional<std::string> dist =
        getDenseLaneStrideStoreDistToken(valueVMIType);
    auto valueType = valueParts.empty()
                         ? VRegType{}
                         : dyn_cast<VRegType>(valueParts.front().getType());
    bool canUseDist =
        dist && valueType && isDirectMemoryDistAddressLegal(
                                  op.getDestination(), op.getOffset(),
                                  valueVMIType.getElementType(), valueType,
                                  VPTOMemoryOpFamily::Store, *dist);
    if (!canUseDist) {
      return false;
    }
    std::optional<StringRef> maskGranularity =
        getDenseLaneStrideStoreMaskGranularity(valueVMIType);
    if (!maskGranularity) {
      return rewriter.notifyMatchFailure(
          op, "unsupported lane_stride store mask granularity");
    }
    if (failed(emitLaneStrideStore(op, destination, offset, valueParts,
                                   valueVMIType, *dist, *maskGranularity,
                                   rewriter))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return true;
  }

  LogicalResult emitAlignedContiguousStoreParts(
      VMIStoreOp op, Value destination, Value offset, ValueRange storeParts,
      VMIVRegType valueVMIType, int64_t lanesPerPart, bool fullPhysicalChunks,
      OneToNPatternRewriter &rewriter) const {
    for (auto [index, value] : llvm::enumerate(storeParts)) {
      auto vregType = dyn_cast<VRegType>(value.getType());
      if (!vregType) {
        return rewriter.notifyMatchFailure(op, "store value must be vreg");
      }
      if (!fullPhysicalChunks) {
        FailureOr<int64_t> activeLanes =
            getContiguousActiveDataLanes(valueVMIType, index);
        if (failed(activeLanes)) {
          return rewriter.notifyMatchFailure(
              op, "failed to compute store active lanes");
        }
        if (*activeLanes == 0) {
          continue;
        }
      }
      FailureOr<Value> mask =
          fullPhysicalChunks
              ? createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter)
              : createContiguousStoreMask(op.getLoc(), valueVMIType, index,
                                          vregType, rewriter);
      if (failed(mask)) {
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for store mask");
      }
      Value chunkOffset = createChunkOffset(op.getLoc(), offset,
                                            index * lanesPerPart, rewriter);
      rewriter.create<VstsOp>(op.getLoc(), /*updated_base=*/Type{}, value,
                              destination, chunkOffset, /*dist=*/nullptr,
                              *mask);
    }
    return success();
  }

  FailureOr<SmallVector<Value>> collectUnalignedStoreValues(
      VMIStoreOp op, ValueRange storeParts, VMIVRegType valueVMIType,
      int64_t lanesPerPart, bool fullPhysicalChunks,
      SmallVectorImpl<int64_t> &advances,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> values;
    for (auto [index, value] : llvm::enumerate(storeParts)) {
      if (!isa<VRegType>(value.getType())) {
        (void)rewriter.notifyMatchFailure(op, "store value must be vreg");
        return failure();
      }
      if (!fullPhysicalChunks) {
        FailureOr<int64_t> maybeActiveLanes =
            getContiguousActiveDataLanes(valueVMIType, index);
        if (failed(maybeActiveLanes)) {
          return rewriter.notifyMatchFailure(
              op, "failed to compute unaligned store active lanes");
        }
        if (*maybeActiveLanes == 0) {
          continue;
        }
        values.push_back(value);
        advances.push_back(*maybeActiveLanes);
        continue;
      }
      values.push_back(value);
      advances.push_back(lanesPerPart);
    }
    return values;
  }

  FailureOr<Value> materializeUnalignedStoreBase(
      VMIStoreOp op, Value destination, Value offset,
      VMIVRegType valueVMIType, OneToNPatternRewriter &rewriter) const {
    Value storeBase = materializeBufferPointer(
        destination, valueVMIType.getElementType(),
        getMemorySpace(destination.getType()), rewriter, op.getLoc());
    if (!storeBase) {
      return rewriter.notifyMatchFailure(
          op, "continuous unaligned store requires a ptr-compatible destination");
    }
    return rewriter
        .create<AddPtrOp>(op.getLoc(), storeBase.getType(), storeBase, offset)
        .getResult();
  }

  LogicalResult lowerContiguousStoreParts(
      VMIStoreOp op, Value destination, Value offset, ValueRange storeParts,
      VMIVRegType valueVMIType, int64_t lanesPerPart, bool fullPhysicalChunks,
      OneToNPatternRewriter &rewriter) const {
    auto firstStoreType =
        storeParts.empty() ? VRegType{}
                           : dyn_cast<VRegType>(storeParts.front().getType());
    bool useAlignedAccess =
        firstStoreType &&
        isDirectMemoryDistAddressLegal(
            op.getDestination(), op.getOffset(), valueVMIType.getElementType(),
            firstStoreType, VPTOMemoryOpFamily::Store, "");
    if (useAlignedAccess) {
      return emitAlignedContiguousStoreParts(
          op, destination, offset, storeParts, valueVMIType, lanesPerPart,
          fullPhysicalChunks, rewriter);
    }

    FailureOr<Value> storeBase = materializeUnalignedStoreBase(
        op, destination, offset, valueVMIType, rewriter);
    if (failed(storeBase)) {
      return failure();
    }
    SmallVector<int64_t> advances;
    FailureOr<SmallVector<Value>> values = collectUnalignedStoreValues(
        op, storeParts, valueVMIType, lanesPerPart, fullPhysicalChunks, advances,
        rewriter);
    if (failed(values)) {
      return failure();
    }
    if (failed(emitStatefulStoreStream(op, *storeBase, *values, advances,
                                       rewriter))) {
      return failure();
    }
    return success();
  }

  LogicalResult lowerByPhysicalPlan(
      VMIStoreOp op, const StoreLoweringInput &input,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<bool> laneStrideStore = tryLowerLaneStrideStore(
        op, input.destination, input.offset, input.valueParts,
        input.valueVMIType, rewriter);
    if (failed(laneStrideStore)) {
      return failure();
    }
    if (*laneStrideStore) {
      return success();
    }
    FailureOr<bool> deinterleavedStore = tryLowerDeinterleavedStore(
        op, input.destination, input.offset, input.valueParts,
        input.valueVMIType, input.plan.lanesPerPart,
        input.plan.fullPhysicalChunks, input.plan.noWiderThanContiguous,
        rewriter);
    if (failed(deinterleavedStore)) {
      return failure();
    }
    if (*deinterleavedStore) {
      return success();
    }
    FailureOr<SmallVector<Value>> storeParts = materializeDataLayoutConversion(
        op, input.valueParts, input.plan.contiguousTypes,
        input.valueVMIType.getLayoutAttr(), input.plan.contiguousLayout,
        input.valueVMIType.getElementType(), rewriter);
    if (failed(storeParts)) {
      return failure();
    }
    if (failed(lowerContiguousStoreParts(
            op, input.destination, input.offset, *storeParts,
            input.valueVMIType, input.plan.lanesPerPart,
            input.plan.fullPhysicalChunks, rewriter))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

public:

  static LogicalResult emitLaneStrideStore(
      VMIStoreOp op, Value destination, Value offset, ValueRange valueParts,
      VMIVRegType valueVMIType, StringRef laneStrideDist,
      StringRef maskGranularity, OneToNPatternRewriter &rewriter) {
    int64_t semanticOffset = 0;
    for (auto [index, value] : llvm::enumerate(valueParts)) {
      (void)index;
      auto vregType = dyn_cast<VRegType>(value.getType());
      if (!vregType) {
        return rewriter.notifyMatchFailure(op, "store value must be vreg");
      }
      FailureOr<int64_t> activeLanes =
          getActiveDataLanesInPhysicalChunk(valueVMIType, index);
      if (failed(activeLanes)) {
        return rewriter.notifyMatchFailure(
            op, "failed to compute lane_stride store active lanes");
      }
      if (*activeLanes == 0) {
        continue;
      }
      auto maskType = MaskType::get(rewriter.getContext(), maskGranularity);
      FailureOr<Value> mask = createPrefixMaskForActiveLanes(
          op.getLoc(), maskType, *activeLanes, rewriter);
      if (failed(mask)) {
        return rewriter.notifyMatchFailure(
            op, "failed to create lane_stride store mask");
      }
      Value chunkOffset =
          createChunkOffset(op.getLoc(), offset, semanticOffset, rewriter);
      rewriter.create<VstsOp>(op.getLoc(), /*updated_base=*/Type{}, value,
                              destination, chunkOffset,
                              rewriter.getStringAttr(laneStrideDist), *mask);
      semanticOffset += *activeLanes;
    }
    return success();
  }

  static LogicalResult emitDeinterleavedStore(
      VMIStoreOp op, Value destination, Value offset, ValueRange valueParts,
      int64_t lanesPerPart, StringRef dist,
      OneToNPatternRewriter &rewriter) {
    bool oddValuePartCount = valueParts.size() % 2 != 0;
    if (oddValuePartCount) {
      return failure();
    }
    int64_t groups = valueParts.size() / 2;
    for (int64_t group = 0; group < groups; ++group) {
      Value low = valueParts[group];
      Value high = valueParts[groups + group];
      bool mismatchedTypes = low.getType() != high.getType();
      if (mismatchedTypes) {
        return rewriter.notifyMatchFailure(
            op, "vstsx2 requires matching low/high value types");
      }
      auto vregType = dyn_cast<VRegType>(low.getType());
      bool invalidValueType = !vregType;
      if (invalidValueType) {
        return rewriter.notifyMatchFailure(op, "store value must be vreg");
      }
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
      bool maskFailed = failed(mask);
      if (maskFailed) {
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for store mask");
      }
      Value chunkOffset = createChunkOffset(
          op.getLoc(), offset, group * 2 * lanesPerPart, rewriter);
      rewriter.create<Vstsx2Op>(op.getLoc(), low, high, destination, chunkOffset,
                                rewriter.getStringAttr(dist), *mask);
    }
    return success();
  }

  FailureOr<bool> tryLowerDeinterleavedStore(
      VMIStoreOp op, Value destination, Value offset, ValueRange valueParts,
      VMIVRegType valueVMIType, int64_t lanesPerPart,
      bool fullPhysicalChunks, bool noWiderThanContiguous,
      OneToNPatternRewriter &rewriter) const {
    VMILayoutSupport supports;
    FailureOr<VMIStoreLayoutFact> storeFact =
        supports.getStoreLayoutFact(valueVMIType);
    bool candidate = succeeded(storeFact) &&
                     storeFact->valueLayout.isDeinterleaved() &&
                     storeFact->valueLayout.getFactor() == 2 &&
                     fullPhysicalChunks && noWiderThanContiguous;
    if (!candidate) {
      return false;
    }
    std::optional<std::string> dist =
        getX2MemoryDistToken(valueVMIType.getElementType(), "INTLV");
    auto firstType = valueParts.empty()
                         ? VRegType{}
                         : dyn_cast<VRegType>(valueParts.front().getType());
    bool canUseDist = dist && firstType &&
                      isDirectMemoryDistAddressLegal(
                          op.getDestination(), op.getOffset(),
                          valueVMIType.getElementType(), firstType,
                          VPTOMemoryOpFamily::StoreX2, *dist);
    bool evenValuePartCount = valueParts.size() % 2 == 0;
    if (!canUseDist || !evenValuePartCount) {
      return false;
    }
    if (failed(emitDeinterleavedStore(
            op, destination, offset, valueParts, lanesPerPart, *dist,
            rewriter))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return true;
  }

  LogicalResult
  matchAndRewrite(VMIStoreOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<StoreLoweringInput> input = getLoweringInput(op, adaptor, rewriter);
    if (failed(input)) {
      return failure();
    }
    return lowerByPhysicalPlan(op, *input, rewriter);
  }
};

struct OneToNVMIInterleaveStoreOpPattern
    : OneToNOpConversionPattern<VMIInterleaveStoreOp> {
  using OneToNOpConversionPattern<
      VMIInterleaveStoreOp>::OneToNOpConversionPattern;

private:
  LogicalResult emitInterleaveStoreChunk(
      VMIInterleaveStoreOp op, Value low, Value high, size_t index,
      int64_t lanesPerPart, Value destination, Value offset, StringRef dist,
      bool useDirectAccess, SmallVectorImpl<Value> &streamValues,
      SmallVectorImpl<int64_t> &streamAdvances,
      OneToNPatternRewriter &rewriter) const {
    bool mismatchedTypes = low.getType() != high.getType();
    if (mismatchedTypes) {
      return rewriter.notifyMatchFailure(
          op, "interleave_store requires matching low/high physical types");
    }
    auto vregType = dyn_cast<VRegType>(low.getType());
    if (!vregType) {
      return rewriter.notifyMatchFailure(
          op, "interleave_store value must be vreg");
    }
    if (useDirectAccess) {
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
      if (failed(mask)) {
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for interleave_store mask");
      }
      Value chunkOffset = createChunkOffset(
          op.getLoc(), offset, static_cast<int64_t>(index) * 2 * lanesPerPart,
          rewriter);
      rewriter.create<Vstsx2Op>(op.getLoc(), low, high, destination, chunkOffset,
                                rewriter.getStringAttr(dist), *mask);
      return success();
    }
    auto packets =
        rewriter.create<VintlvOp>(op.getLoc(), vregType, vregType, low, high);
    streamValues.push_back(packets.getLow());
    streamValues.push_back(packets.getHigh());
    streamAdvances.push_back(lanesPerPart);
    streamAdvances.push_back(lanesPerPart);
    return success();
  }

  FailureOr<bool> canUseDirectAccess(
      VMIInterleaveStoreOp op, ValueRange lowParts,
      VMIVRegType lowVMIType, StringRef dist) const {
    auto firstType = lowParts.empty()
                         ? VRegType{}
                         : dyn_cast<VRegType>(lowParts.front().getType());
    if (!firstType) {
      return false;
    }
    return isDirectMemoryDistAddressLegal(
        op.getDestination(), op.getOffset(), lowVMIType.getElementType(),
        firstType, VPTOMemoryOpFamily::StoreX2, dist);
  }

  FailureOr<Value> getUnalignedBase(
      VMIInterleaveStoreOp op, Value destination, Value offset,
      VMIVRegType lowVMIType, OneToNPatternRewriter &rewriter) const {
    Value streamBase = materializeBufferPointer(
        destination, lowVMIType.getElementType(),
        getMemorySpace(destination.getType()), rewriter, op.getLoc());
    if (!streamBase) {
      return rewriter.notifyMatchFailure(
          op, "unaligned interleave_store requires a ptr-compatible "
              "destination");
    }
    return rewriter
        .create<AddPtrOp>(op.getLoc(), streamBase.getType(), streamBase,
                          offset)
        .getResult();
  }

  struct InterleaveStoreAddressPlan {
    Value destination;
    Value offset;
    Value streamBase;
    bool useDirectAccess;
  };

  struct InterleaveStoreLoweringInput {
    Value destination;
    Value offset;
    ValueRange lowParts;
    ValueRange highParts;
    VMIVRegType lowVMIType;
    int64_t lanesPerPart;
    std::string dist;
  };

  FailureOr<InterleaveStoreLoweringInput> getLoweringInput(
      VMIInterleaveStoreOp op, OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const {
    auto lowVMIType = cast<VMIVRegType>(op.getLow().getType());
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(lowVMIType.getElementType());
    if (failed(lanesPerPart)) {
      return rewriter.notifyMatchFailure(
          op, "interleave_store requires known physical lanes per part");
    }
    std::optional<std::string> dist =
        getX2MemoryDistToken(lowVMIType.getElementType(), "INTLV");
    if (!dist) {
      return rewriter.notifyMatchFailure(
          op, "interleave_store requires vstsx2 INTLV element support");
    }
    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "interleave_store destination must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(),
        "interleave_store offset must convert to one value", rewriter);
    bool invalidAddressOperands = failed(destination) || failed(offset);
    if (invalidAddressOperands) {
      return failure();
    }
    ValueRange lowParts = adaptor.getLow();
    ValueRange highParts = adaptor.getHigh();
    bool arityMismatch = lowParts.size() != highParts.size();
    if (arityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "interleave_store requires matching low/high physical arity");
    }
    return InterleaveStoreLoweringInput{*destination, *offset, lowParts,
                                        highParts, lowVMIType, *lanesPerPart,
                                        *dist};
  }

  FailureOr<InterleaveStoreAddressPlan> buildAddressPlan(
      VMIInterleaveStoreOp op, Value destination, Value offset,
      ValueRange lowParts, VMIVRegType lowVMIType, StringRef dist,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<bool> directAccess =
        canUseDirectAccess(op, lowParts, lowVMIType, dist);
    if (failed(directAccess)) {
      return failure();
    }
    if (*directAccess) {
      return InterleaveStoreAddressPlan{destination, offset, Value{}, true};
    }
    FailureOr<Value> base =
        getUnalignedBase(op, destination, offset, lowVMIType, rewriter);
    if (failed(base)) {
      return failure();
    }
    return InterleaveStoreAddressPlan{destination, offset, *base, false};
  }

  LogicalResult lowerByAddressPlan(
      VMIInterleaveStoreOp op, const InterleaveStoreLoweringInput &input,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<InterleaveStoreAddressPlan> addressPlan = buildAddressPlan(
        op, input.destination, input.offset, input.lowParts, input.lowVMIType,
        input.dist, rewriter);
    if (failed(addressPlan)) {
      return failure();
    }
    SmallVector<Value> streamValues;
    SmallVector<int64_t> streamAdvances;
    for (size_t index = 0, e = input.lowParts.size(); index < e; ++index) {
      if (failed(emitInterleaveStoreChunk(
              op, input.lowParts[index], input.highParts[index], index,
              input.lanesPerPart, input.destination, input.offset, input.dist,
              addressPlan->useDirectAccess, streamValues, streamAdvances,
              rewriter))) {
        return failure();
      }
    }
    if (!addressPlan->useDirectAccess &&
        failed(emitStatefulStoreStream(op, addressPlan->streamBase,
                                       streamValues, streamAdvances, rewriter))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

public:

  LogicalResult
  matchAndRewrite(VMIInterleaveStoreOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<InterleaveStoreLoweringInput> input =
        getLoweringInput(op, adaptor, rewriter);
    if (failed(input)) {
      return failure();
    }
    return lowerByAddressPlan(op, *input, rewriter);
  }
};

struct OneToNVMIGroupStoreOpPattern
    : OneToNOpConversionPattern<VMIGroupStoreOp> {
  using OneToNOpConversionPattern<VMIGroupStoreOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerAlignedPackedByteStore(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, Value destination, Value offset,
      int64_t numGroups) const {
    MLIRContext *ctx = rewriter.getContext();
    auto ui16 = IntegerType::get(
        ctx, 16, IntegerType::SignednessSemantics::Unsigned);
    auto ui8 = IntegerType::get(
        ctx, 8, IntegerType::SignednessSemantics::Unsigned);
    auto packed16Type = VRegType::get(ctx, 128, ui16);
    auto packed8Type = VRegType::get(ctx, 256, ui8);
    Value packed16 =
        rewriter
            .create<VpackOp>(op.getLoc(), packed16Type, valueParts.front(),
                             rewriter.getStringAttr("LOWER"))
            .getResult();
    Value packed8 =
        rewriter
            .create<VpackOp>(op.getLoc(), packed8Type, packed16,
                             rewriter.getStringAttr("LOWER"))
            .getResult();
    FailureOr<MaskType> packedMaskType =
        getMaskTypeForVReg(packed8Type, ctx);
    if (failed(packedMaskType)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create packed byte group_store mask type");
    }
    FailureOr<Value> storeMask = createPrefixMaskForActiveLanes(
        op.getLoc(), *packedMaskType, numGroups, rewriter);
    if (failed(storeMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create packed byte group_store mask");
    }
    rewriter.create<VstsOp>(op.getLoc(), Type{}, packed8, destination, offset,
                            rewriter.getStringAttr("NORM_B8"), *storeMask);
    return success();
  }

  LogicalResult emitPackedByteStoreStream(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter, Value destination,
      Value offset, ArrayRef<Value> values, ArrayRef<int64_t> advances) const {
    Type destinationElementType = getMemoryElementType(destination.getType());
    Value storeBase = materializeBufferPointer(
        destination, destinationElementType,
        getMemorySpace(destination.getType()), rewriter, op.getLoc());
    if (!storeBase) {
      return rewriter.notifyMatchFailure(
          op, "unaligned packed byte group_store requires a ptr-compatible destination");
    }
    storeBase = rewriter
                    .create<AddPtrOp>(op.getLoc(), storeBase.getType(),
                                      storeBase, offset)
                    .getResult();
    return emitStatefulStoreStream(op, storeBase, values, advances, rewriter);
  }

  FailureOr<Value> materializePackedSlots1Group(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter, Value value,
      int64_t group, VRegType firstType, MaskType maskType, Value allMask,
      Value packed) const {
    auto vregType = dyn_cast<VRegType>(value.getType());
    if (!vregType || vregType != firstType) {
      return rewriter.notifyMatchFailure(
          op, "packed group_store requires uniform vreg parts");
    }
    Value splat = rewriter
                      .create<VdupOp>(op.getLoc(), firstType, value, allMask,
                                      rewriter.getStringAttr("LOWEST"))
                      .getResult();
    FailureOr<Value> laneMask =
        createLaneRangeMask(op.getLoc(), maskType, group, group + 1, rewriter);
    if (failed(laneMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create packed group_store lane mask");
    }
    return rewriter
        .create<VselOp>(op.getLoc(), firstType, splat, packed, *laneMask)
        .getResult();
  }

  FailureOr<Value> buildPackedSlots1Value(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, VMILayoutAttr layout, VRegType firstType,
      MaskType maskType, Value allMask) const {
    Value packed = rewriter
                       .create<VdupOp>(op.getLoc(), firstType,
                                       valueParts.front(), allMask,
                                       rewriter.getStringAttr("LOWEST"))
                       .getResult();
    for (int64_t group = 1; group < layout.getNumGroups(); ++group) {
      FailureOr<Value> nextPacked = materializePackedSlots1Group(
          op, rewriter, valueParts[group], group, firstType, maskType, allMask,
          packed);
      if (failed(nextPacked)) {
        return failure();
      }
      packed = *nextPacked;
    }
    return packed;
  }

  LogicalResult lowerSlots1PackedUnitStride(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, VMIVRegType valueVMIType, VMILayoutAttr layout,
      Value destination, Value offset) const {
    auto firstType = dyn_cast<VRegType>(valueParts.front().getType());
    if (!firstType) {
      return rewriter.notifyMatchFailure(op, "group_store value must be vreg");
    }
    FailureOr<MaskType> maskType =
        getMaskTypeForVReg(firstType, rewriter.getContext());
    FailureOr<Value> allMask =
        createAllTrueMaskForVReg(op.getLoc(), firstType, rewriter);
    bool unsupportedMasks = failed(maskType) || failed(allMask);
    if (unsupportedMasks) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for packed group_store mask");
    }
    FailureOr<Value> packed = buildPackedSlots1Value(
        op, rewriter, valueParts, layout, firstType, *maskType, *allMask);
    if (failed(packed)) {
      return failure();
    }
    if (isKnownAddressAligned(destination, offset,
                               valueVMIType.getElementType(), 32)) {
      FailureOr<Value> storeMask = createPrefixMaskForActiveLanes(
          op.getLoc(), *maskType, layout.getNumGroups(), rewriter);
      if (failed(storeMask)) {
        return rewriter.notifyMatchFailure(
            op, "failed to create packed group_store store mask");
      }
      rewriter.create<VstsOp>(op.getLoc(), Type{}, *packed, destination, offset,
                              nullptr, *storeMask);
    } else if (failed(emitPackedSlots1StoreStream(
                   op, rewriter, destination, offset, valueVMIType, *packed,
                   layout.getNumGroups()))) {
        return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult emitPackedSlots1StoreStream(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter, Value destination,
      Value offset, VMIVRegType valueVMIType, Value packed,
      int64_t numGroups) const {
    Value storeBase = materializeBufferPointer(
        destination, valueVMIType.getElementType(),
        getMemorySpace(destination.getType()), rewriter, op.getLoc());
    if (!storeBase) {
      return rewriter.notifyMatchFailure(
          op, "packed unaligned group_store requires a ptr-compatible destination");
    }
    storeBase = rewriter
                    .create<AddPtrOp>(op.getLoc(), storeBase.getType(), storeBase,
                                      offset)
                    .getResult();
    SmallVector<Value> streamValues{packed};
    SmallVector<int64_t> streamAdvances{numGroups};
    return emitStatefulStoreStream(op, storeBase, streamValues, streamAdvances,
                                   rewriter);
  }

  LogicalResult lowerSlots1PointStores(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, VMIVRegType valueVMIType, Value destination,
      Value offset, Value rowStride) const {
    std::optional<std::string> pointDist =
        getPointStoreDistToken(valueVMIType.getElementType());
    if (!pointDist) {
      return rewriter.notifyMatchFailure(
          op, "slots=1 group_store requires 1PT_B8/B16/B32 store support");
    }
    for (auto [group, value] : llvm::enumerate(valueParts)) {
      if (failed(emitSlots1PointStore(op, value, group, destination, offset,
                                      rowStride,
                                      *pointDist, rewriter))) {
        return failure();
      }
    }
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult emitSlots1PointStore(
      VMIGroupStoreOp op, Value value, int64_t group, Value destination,
      Value offset, Value rowStride, StringRef pointDist,
      OneToNPatternRewriter &rewriter) const {
    auto vregType = dyn_cast<VRegType>(value.getType());
    if (!vregType) {
      return rewriter.notifyMatchFailure(op, "group_store value must be vreg");
    }
    FailureOr<MaskType> maskType =
        getMaskTypeForVReg(vregType, rewriter.getContext());
    if (failed(maskType)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for group_store mask");
    }
    FailureOr<Value> mask =
        createPrefixMask(op.getLoc(), *maskType, "PAT_VL1", rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create slots=1 group_store mask");
    }
    Value groupOffset = createGroupChunkOffset(
        op.getLoc(), offset, rowStride, group, 0, rewriter);
    rewriter.create<VstsOp>(op.getLoc(), Type{}, value, destination, groupOffset,
                            rewriter.getStringAttr(pointDist), *mask);
    return success();
  }

  LogicalResult lowerSlots1(VMIGroupStoreOp op, OpAdaptor adaptor,
                            OneToNPatternRewriter &rewriter,
                            VMIVRegType valueVMIType, VMILayoutAttr layout,
                            Value destination, Value offset,
                            Value rowStride) const {
    ValueRange valueParts = adaptor.getValue();
    bool hasExpectedArity =
        static_cast<int64_t>(valueParts.size()) == layout.getNumGroups();
    if (!hasExpectedArity) {
      return rewriter.notifyMatchFailure(op,
                                         "slots=1 group_store arity mismatch");
    }
    unsigned elementBits =
        pto::getPTOStorageElemBitWidth(valueVMIType.getElementType());
    if (elementBits == 0 || 256 % elementBits != 0) {
      return rewriter.notifyMatchFailure(
          op, "slots=1 group_store requires supported element width");
    }
    std::optional<int64_t> constantRowStride =
        getConstantIndexValue(op.getRowStride());
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(valueVMIType.getElementType());
    if (constantRowStride && *constantRowStride == 1 &&
        succeeded(lanesPerPart) && layout.getNumGroups() <= *lanesPerPart) {
      return lowerSlots1PackedUnitStride(
          op, rewriter, valueParts, valueVMIType, layout, destination, offset);
    }
    if (constantRowStride && *constantRowStride <= 0) {
      return rewriter.notifyMatchFailure(
          op, "slots=1 group_store requires positive row_stride when row_stride is constant");
    }
    return lowerSlots1PointStores(op, rewriter, valueParts, valueVMIType,
                                  destination, offset, rowStride);
  }

  LogicalResult lowerScalarGroupStore(
      VMIGroupStoreOp op, OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter, VMIVRegType valueVMIType,
      Value destination, Value offset) const {
    ValueRange valueParts = adaptor.getValue();
    bool invalidValueArity = valueParts.size() != 1;
    if (invalidValueArity) {
      return rewriter.notifyMatchFailure(
          op, "scalar group_store requires one physical value part");
    }
    auto valueType = dyn_cast<VRegType>(valueParts.front().getType());
    if (!valueType) {
      return rewriter.notifyMatchFailure(
          op, "scalar group_store value must be vreg");
    }
    std::optional<std::string> pointDist =
        getPointStoreDistToken(valueVMIType.getElementType());
    if (!pointDist) {
      return rewriter.notifyMatchFailure(
          op, "scalar group_store requires point-store support");
    }
    FailureOr<MaskType> maskType =
        getMaskTypeForVReg(valueType, rewriter.getContext());
    if (failed(maskType)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for scalar group_store mask");
    }
    FailureOr<Value> mask =
        createPrefixMask(op.getLoc(), *maskType, "PAT_VL1", rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create scalar group_store mask");
    }
    rewriter.create<VstsOp>(op.getLoc(), Type{}, valueParts.front(),
                            destination, offset,
                            rewriter.getStringAttr(*pointDist), *mask);
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult lowerDeinterleaved2GroupStore(
      VMIGroupStoreOp op, OpAdaptor adaptor, OneToNPatternRewriter &rewriter,
      VMIVRegType valueVMIType, const VMIGroupStoreLayoutFact &fact,
      Value destination, Value offset, Value rowStride) const {
    int64_t lanesPerPart = 0;
    int64_t groupCount = 0;
    int64_t chunksPerGroup = 0;
    std::string reason;
    if (failed(checkDeinterleaved2GroupStoreChunkShape(
            valueVMIType, fact.groupSize, &lanesPerPart, &groupCount,
            &chunksPerGroup, &reason))) {
      return failure();
    }
    std::optional<std::string> dist =
        getX2MemoryDistToken(valueVMIType.getElementType(), "INTLV");
    if (!dist) {
      return rewriter.notifyMatchFailure(
          op, "group_store requires vstsx2 INTLV element support");
    }
    ValueRange valueParts = adaptor.getValue();
    int64_t chunksPerPart = groupCount * chunksPerGroup;
    bool hasExpectedArity =
        static_cast<int64_t>(valueParts.size()) == 2 * chunksPerPart;
    if (!hasExpectedArity) {
      return rewriter.notifyMatchFailure(
          op, "deinterleaved=2 group_store arity mismatch");
    }
    for (int64_t group = 0; group < groupCount; ++group) {
      for (int64_t chunk = 0; chunk < chunksPerGroup; ++chunk) {
        int64_t lowIndex = group * chunksPerGroup + chunk;
        int64_t highIndex = chunksPerPart + lowIndex;
        if (failed(emitDeinterleaved2GroupStorePair(
                op, valueParts[lowIndex], valueParts[highIndex], group, chunk,
                lanesPerPart, destination, offset, rowStride, *dist,
                rewriter))) {
          return failure();
        }
      }
    }
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult emitDeinterleaved2GroupStorePair(
      VMIGroupStoreOp op, Value low, Value high, int64_t group, int64_t chunk,
      int64_t lanesPerPart, Value destination, Value offset, Value rowStride,
      StringRef dist, OneToNPatternRewriter &rewriter) const {
    bool mismatchedTypes = low.getType() != high.getType();
    if (mismatchedTypes) {
      return rewriter.notifyMatchFailure(
          op, "vstsx2 group_store requires matching low/high types");
    }
    auto vregType = dyn_cast<VRegType>(low.getType());
    if (!vregType) {
      return rewriter.notifyMatchFailure(op, "group_store value must be vreg");
    }
    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for group_store mask");
    }
    Value chunkOffset = createGroupChunkOffset(
        op.getLoc(), offset, rowStride, group, chunk * 2 * lanesPerPart,
        rewriter);
    rewriter.create<Vstsx2Op>(op.getLoc(), low, high, destination, chunkOffset,
                              rewriter.getStringAttr(dist), *mask);
    return success();
  }

  LogicalResult emitContiguousGroupStorePart(
      VMIGroupStoreOp op, Value value, int64_t index, int64_t chunksPerGroup,
      int64_t lanesPerPart, Value destination, Value offset, Value rowStride,
      OneToNPatternRewriter &rewriter) const {
    if (chunksPerGroup <= 0) {
      return rewriter.notifyMatchFailure(
          op, "group_store requires positive chunks per group");
    }
    int64_t safeChunksPerGroup = chunksPerGroup;
    auto vregType = dyn_cast<VRegType>(value.getType());
    if (!vregType) {
      return rewriter.notifyMatchFailure(op,
                                         "group_store value must be vreg");
    }
    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for group_store mask");
    }
    int64_t group = index / safeChunksPerGroup;
    int64_t chunkInGroup = index % safeChunksPerGroup;
    Value chunkOffset = createGroupChunkOffset(
        op.getLoc(), offset, rowStride, group, chunkInGroup * lanesPerPart,
        rewriter);
    rewriter.create<VstsOp>(op.getLoc(), /*updated_base=*/Type{}, value,
                            destination, chunkOffset, /*dist=*/nullptr, *mask);
    return success();
  }

  LogicalResult lowerContiguousGroupStore(
      VMIGroupStoreOp op, OpAdaptor adaptor, OneToNPatternRewriter &rewriter,
      VMIVRegType valueVMIType, const VMIGroupStoreLayoutFact &fact,
      Value destination, Value offset, Value rowStride) const {
    int64_t lanesPerPart = 0;
    int64_t groupCount = 0;
    int64_t chunksPerGroup = 0;
    if (failed(checkContiguousFullGroupChunks(
            op, valueVMIType, fact.groupSize, &lanesPerPart, &groupCount,
            &chunksPerGroup, rewriter))) {
      return failure();
    }
    ValueRange valueParts = adaptor.getValue();
    bool hasExpectedArity = static_cast<int64_t>(valueParts.size()) ==
                            groupCount * chunksPerGroup;
    if (!hasExpectedArity) {
      return rewriter.notifyMatchFailure(op, "group_store arity mismatch");
    }
    for (auto [index, value] : llvm::enumerate(valueParts)) {
      if (failed(emitContiguousGroupStorePart(
              op, value, index, chunksPerGroup, lanesPerPart, destination,
              offset, rowStride, rewriter))) {
        return failure();
      }
    }
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult emitAlignedSlots8Contiguous(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, ArrayRef<Value> groupOffsets, Value destination,
      int64_t numGroups) const {
    for (auto [slotBlock, value] : llvm::enumerate(valueParts)) {
      auto vregType = dyn_cast<VRegType>(value.getType());
      if (!vregType) {
        return rewriter.notifyMatchFailure(op,
                                           "group_store value must be vreg");
      }
      FailureOr<MaskType> maskType =
          getMaskTypeForVReg(vregType, rewriter.getContext());
      if (failed(maskType)) {
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for group_store mask");
      }
      int64_t activeGroups = std::min<int64_t>(8, numGroups - slotBlock * 8);
      FailureOr<Value> mask = createPrefixMaskForActiveLanes(
          op.getLoc(), *maskType, activeGroups, rewriter);
      if (failed(mask)) {
        return rewriter.notifyMatchFailure(
            op, "failed to create slots=8 group_store mask");
      }
      rewriter.create<VstsOp>(op.getLoc(), /*updated_base=*/Type{}, value,
                              destination, groupOffsets[slotBlock],
                              /*dist=*/nullptr, *mask);
    }
    return success();
  }

  LogicalResult lowerSlots8Contiguous(
      VMIGroupStoreOp op, OpAdaptor adaptor, OneToNPatternRewriter &rewriter,
      Value destination, Value offset, Value rowStride,
      int64_t numGroups) const {
    ValueRange valueParts = adaptor.getValue();
    FailureOr<std::pair<SmallVector<Value>, bool>> offsetPlan =
        buildSlots8GroupOffsets(op, valueParts, destination, offset, rowStride,
                                "", rewriter);
    if (failed(offsetPlan)) {
      return failure();
    }
    SmallVector<Value> groupOffsets = std::move(offsetPlan->first);
    bool useDirectAccess = offsetPlan->second;

    if (!useDirectAccess) {
      SmallVector<int64_t> advances =
          buildSlots8StreamAdvances(numGroups, valueParts.size());
      if (failed(emitGroupStoreStream(op, destination, offset, valueParts,
                                      advances, rewriter))) {
        return failure();
      }
      rewriter.eraseOp(op);
      return success();
    }

    if (failed(emitAlignedSlots8Contiguous(
            op, rewriter, valueParts, groupOffsets, destination, numGroups))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult emitOneBlockGroupStorePart(
      VMIGroupStoreOp op, Value value, int64_t part,
      VMIVRegType valueVMIType, const OneBlockGroupStorePlan &plan,
      Value destination, Value offset, Value rowStride, Value blockStride,
      Value repeatStride, OneToNPatternRewriter &rewriter) const {
    auto vregType = dyn_cast<VRegType>(value.getType());
    if (!vregType) {
      return rewriter.notifyMatchFailure(
          op, "one-block group_store value must be vreg");
    }
    FailureOr<Value> mask = createContiguousStoreMask(
        op.getLoc(), valueVMIType, part, vregType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create one-block group_store mask");
    }
    Value partOffset = createGroupChunkOffset(
        op.getLoc(), offset, rowStride, part * plan.groupsPerPart,
        /*inGroupLaneOffset=*/0, rewriter);
    Value base = rewriter
                     .create<AddPtrOp>(op.getLoc(), destination.getType(),
                                       destination, partOffset)
                     .getResult();
    rewriter.create<VsstbOp>(op.getLoc(), /*updated_base=*/Type{}, value, base,
                             blockStride, repeatStride, *mask);
    return success();
  }

  LogicalResult lowerOneBlockGroupStore(
      VMIGroupStoreOp op, OpAdaptor adaptor, OneToNPatternRewriter &rewriter,
      VMIVRegType valueVMIType, const VMIGroupStoreLayoutFact &fact,
      Value destination, Value offset, Value rowStride) const {
    FailureOr<OneBlockGroupStorePlan> plan =
        getOneBlockGroupStorePlan(op, valueVMIType, fact, nullptr);
    if (failed(plan)) {
      return rewriter.notifyMatchFailure(
          op, "failed to build one-block group_store vsstb plan");
    }
    ValueRange valueParts = adaptor.getValue();
    int64_t numGroups = op.getNumGroupsAttr().getInt();
    bool hasExpectedArity =
        static_cast<int64_t>(valueParts.size()) ==
        ceilDivNonNegative(numGroups, plan->groupsPerPart);
    if (!hasExpectedArity) {
      return rewriter.notifyMatchFailure(
          op, "one-block group_store physical arity mismatch");
    }
    Value blockStride =
        rewriter.create<arith::ConstantIntOp>(op.getLoc(), plan->blockStride, 16);
    Value repeatStride = rewriter.create<arith::ConstantIntOp>(
        op.getLoc(), 0, 16);
    for (auto [part, value] : llvm::enumerate(valueParts)) {
      if (failed(emitOneBlockGroupStorePart(
              op, value, part, valueVMIType, *plan, destination, offset,
              rowStride, blockStride, repeatStride, rewriter))) {
        return failure();
      }
    }
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult emitAlignedSlots8LaneStride(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, ArrayRef<Value> groupOffsets, Value destination,
      MaskType maskType, int64_t numGroups, StringRef dist) const {
    for (auto [slotBlock, value] : llvm::enumerate(valueParts)) {
      if (!isa<VRegType>(value.getType())) {
        return rewriter.notifyMatchFailure(op,
                                           "group_store value must be vreg");
      }
      int64_t activeGroups = std::min<int64_t>(8, numGroups - slotBlock * 8);
      FailureOr<Value> mask = createPrefixMaskForActiveLanes(
          op.getLoc(), maskType, activeGroups, rewriter);
      if (failed(mask)) {
        return rewriter.notifyMatchFailure(
            op, "failed to create packed slots=8 group_store mask");
      }
      rewriter.create<VstsOp>(op.getLoc(), /*updated_base=*/Type{}, value,
                              destination, groupOffsets[slotBlock],
                              rewriter.getStringAttr(dist), *mask);
    }
    return success();
  }

  FailureOr<std::pair<SmallVector<Value>, bool>> buildSlots8GroupOffsets(
      VMIGroupStoreOp op, ValueRange valueParts, Value destination,
      Value offset, Value rowStride, StringRef dist,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> groupOffsets;
    bool useDirectAccess = true;
    for (auto [slotBlock, value] : llvm::enumerate(valueParts)) {
      auto vregType = dyn_cast<VRegType>(value.getType());
      if (!vregType) {
        (void)rewriter.notifyMatchFailure(op, "group_store value must be vreg");
        return failure();
      }
      Value groupOffset = createGroupChunkOffset(
          op.getLoc(), offset, rowStride, slotBlock * 8, 0, rewriter);
      groupOffsets.push_back(groupOffset);
      useDirectAccess &= isDirectMemoryDistAddressLegal(
          op.getDestination(), groupOffset,
          getMemoryElementType(destination.getType()), vregType,
          VPTOMemoryOpFamily::Store, dist);
    }
    return std::make_pair(std::move(groupOffsets), useDirectAccess);
  }

  FailureOr<SmallVector<Value>> materializeLaneStrideStreamValues(
      VMIGroupStoreOp op, ValueRange valueParts, VMIVRegType valueVMIType,
      VMILayoutAttr layout, OneToNPatternRewriter &rewriter) const {
    VMILayoutAttr compactLayout = VMILayoutAttr::getGroupSlots(
        rewriter.getContext(), layout.getNumGroups(), layout.getSlots());
    auto compactType = VMIVRegType::get(
        rewriter.getContext(), valueVMIType.getElementCount(),
        valueVMIType.getElementType(), compactLayout);
    FailureOr<SmallVector<Value>> compactValues = materializeEnsureLayoutConversion(
        op, valueParts, valueVMIType, compactType,
        *this->getTypeConverter(), rewriter);
    bool invalidCompactValues = failed(compactValues) ||
                                compactValues->size() != valueParts.size();
    if (invalidCompactValues) {
      return rewriter.notifyMatchFailure(
          op, "failed to compact unaligned slots=8 group_store");
    }
    return *compactValues;
  }

  SmallVector<int64_t> buildSlots8StreamAdvances(
      int64_t numGroups, size_t valueCount) const {
    SmallVector<int64_t> advances;
    advances.reserve(valueCount);
    for (size_t slotBlock = 0; slotBlock < valueCount; ++slotBlock) {
      advances.push_back(std::min<int64_t>(8, numGroups - slotBlock * 8));
    }
    return advances;
  }

  LogicalResult lowerSlots8LaneStride(
      VMIGroupStoreOp op, OpAdaptor adaptor, OneToNPatternRewriter &rewriter,
      VMIVRegType valueVMIType, VMILayoutAttr layout, Value destination,
      Value offset, Value rowStride, int64_t numGroups) const {
    std::optional<std::string> dist =
        getLaneStrideStoreDistToken(layout, valueVMIType.getElementType());
    std::optional<StringRef> maskGranularity =
        getLaneStrideStoreMaskGranularity(layout,
                                          valueVMIType.getElementType());
    if (!dist || !maskGranularity) {
      return rewriter.notifyMatchFailure(
          op, "unsupported slots=8 lane_stride group_store packing");
    }
    ValueRange valueParts = adaptor.getValue();
    auto maskType = MaskType::get(rewriter.getContext(), *maskGranularity);
    FailureOr<std::pair<SmallVector<Value>, bool>> offsetPlan =
        buildSlots8GroupOffsets(op, valueParts, destination, offset, rowStride,
                                *dist, rewriter);
    if (failed(offsetPlan)) {
      return failure();
    }
    SmallVector<Value> groupOffsets = std::move(offsetPlan->first);
    bool useDirectAccess = offsetPlan->second;
    if (!useDirectAccess) {
      FailureOr<SmallVector<Value>> compactValues =
          materializeLaneStrideStreamValues(op, valueParts, valueVMIType,
                                            layout, rewriter);
      if (failed(compactValues)) {
        return failure();
      }
      SmallVector<int64_t> advances =
          buildSlots8StreamAdvances(numGroups, compactValues->size());
      if (failed(emitGroupStoreStream(op, destination, offset, *compactValues,
                                      advances, rewriter))) {
        return failure();
      }
      rewriter.eraseOp(op);
      return success();
    }
    if (failed(emitAlignedSlots8LaneStride(
            op, rewriter, valueParts, groupOffsets, destination, maskType,
            numGroups, *dist))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

  FailureOr<Value> materializeCompactSmallGroupValue(
      VMIGroupStoreOp op, ValueRange valueParts, VMIVRegType valueVMIType,
      VMILayoutAttr layout, OneToNPatternRewriter &rewriter) const {
    Value compactValue = valueParts.front();
    bool alreadyCompact = layout.getLaneStride() == 1;
    if (alreadyCompact) {
      return compactValue;
    }
    VMILayoutAttr compactLayout = VMILayoutAttr::getGroupSlots(
        rewriter.getContext(), layout.getNumGroups(), layout.getSlots());
    auto compactVMIType = VMIVRegType::get(
        rewriter.getContext(), valueVMIType.getElementCount(),
        valueVMIType.getElementType(), compactLayout);
    FailureOr<SmallVector<Value>> packed = materializeEnsureLayoutConversion(
        op, valueParts, valueVMIType, compactVMIType,
        *this->getTypeConverter(), rewriter);
    bool invalidPacked = failed(packed) || packed->size() != 1;
    if (invalidPacked) {
      return rewriter.notifyMatchFailure(
          op, "failed to materialize compact group_store layout");
    }
    return packed->front();
  }

  LogicalResult emitAlignedCompactSmallGroupStore(
      VMIGroupStoreOp op, Value compactValue, VMIVRegType valueVMIType,
      Value destination, Value offset,
      OneToNPatternRewriter &rewriter) const {
    auto compactType = dyn_cast<VRegType>(compactValue.getType());
    std::optional<std::string> normalDist =
        getX2MemoryDistToken(valueVMIType.getElementType(), "NORM");
    if (!compactType || !normalDist) {
      return rewriter.notifyMatchFailure(
          op, "aligned compact group_store requires a supported vreg element type");
    }
    FailureOr<MaskType> maskType =
        getMaskTypeForVReg(compactType, rewriter.getContext());
    if (failed(maskType)) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive aligned compact group_store mask type");
    }
    FailureOr<Value> storeMask = createPrefixMaskForActiveLanes(
        op.getLoc(), *maskType, valueVMIType.getElementCount(), rewriter);
    if (failed(storeMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create aligned compact group_store mask");
    }
    rewriter.create<VstsOp>(op.getLoc(), /*updated_base=*/Type{}, compactValue,
                            destination, offset,
                            rewriter.getStringAttr(*normalDist), *storeMask);
    return success();
  }

  LogicalResult emitUnalignedCompactSmallGroupStore(
      VMIGroupStoreOp op, Value compactValue, VMIVRegType valueVMIType,
      Value destination, Value offset,
      OneToNPatternRewriter &rewriter) const {
    Value elementBase =
        rewriter.create<AddPtrOp>(op.getLoc(), destination.getType(),
                                  destination, offset)
            .getResult();
    SmallVector<Value> streamValues{compactValue};
    SmallVector<int64_t> streamAdvances{valueVMIType.getElementCount()};
    return emitStatefulStoreStream(op, elementBase, streamValues, streamAdvances,
                                   rewriter);
  }

  LogicalResult lowerCompactSmallGroupStore(
      VMIGroupStoreOp op, OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter, VMIVRegType valueVMIType,
      VMILayoutAttr layout, Value destination, Value offset) const {
    ValueRange valueParts = adaptor.getValue();
    bool invalidValueArity = valueParts.size() != 1;
    if (invalidValueArity) {
      return rewriter.notifyMatchFailure(
          op, "compact small group_store requires one physical value part");
    }
    auto valueType = dyn_cast<VRegType>(valueParts.front().getType());
    if (!valueType || !isa<PtrType>(destination.getType())) {
      return rewriter.notifyMatchFailure(
          op, "compact small group_store requires vreg and ptr operands");
    }

    FailureOr<Value> compactValue = materializeCompactSmallGroupValue(
        op, valueParts, valueVMIType, layout, rewriter);
    if (failed(compactValue)) {
      return failure();
    }

    if (isKnownAddressAligned(destination, offset,
                              valueVMIType.getElementType(), 32)) {
      if (failed(emitAlignedCompactSmallGroupStore(
              op, *compactValue, valueVMIType, destination, offset, rewriter))) {
        return failure();
      }
      rewriter.eraseOp(op);
      return success();
    }

    if (failed(emitUnalignedCompactSmallGroupStore(
            op, *compactValue, valueVMIType, destination, offset, rewriter))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

  FailureOr<std::tuple<Value, Value, Value>> buildPackedByteStoreBlock(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts,
      VRegType firstVRegType, MaskType maskType, Value slotIndex,
      Value destination, Value offset, Value rowStride, int64_t numGroups,
      int64_t blockStart) const {
    FailureOr<Value> zero =
        createZeroVector(op.getLoc(), firstVRegType, rewriter);
    if (failed(zero)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create packed group_store accumulator");
    }
    Value merged = *zero;
    for (int64_t localPart = 0; localPart < 4; ++localPart) {
      int64_t partIndex = blockStart / 8 + localPart;
      if (partIndex >= static_cast<int64_t>(valueParts.size())) {
        break;
      }
      int64_t activeGroups = std::min<int64_t>(8, numGroups - partIndex * 8);
      if (activeGroups <= 0) {
        break;
      }
      FailureOr<Value> nextMerged = mergePackedByteStoreBlockPart(
          op, rewriter, valueParts[partIndex], slotIndex, merged,
          firstVRegType, maskType, localPart * 8, activeGroups);
      if (failed(nextMerged)) {
        return failure();
      }
      merged = *nextMerged;
    }
    int64_t activeGroups = std::min<int64_t>(32, numGroups - blockStart);
    FailureOr<Value> storeMask = createPrefixMaskForActiveLanes(
        op.getLoc(), maskType, activeGroups, rewriter);
    if (failed(storeMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create packed group_store store mask");
    }
    Value groupOffset = createGroupChunkOffset(
        op.getLoc(), offset, rowStride, blockStart, 0, rewriter);
    return std::make_tuple(merged, *storeMask, groupOffset);
  }

  FailureOr<Value> mergePackedByteStoreBlockPart(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter, Value value,
      Value slotIndex, Value merged, VRegType valueType, MaskType maskType,
      int64_t laneStart, int64_t activeGroups) const {
    Value selected = rewriter
                         .create<VselrOp>(op.getLoc(), valueType, value, slotIndex)
                         .getResult();
    FailureOr<Value> laneMask = createLaneRangeMask(
        op.getLoc(), maskType, laneStart, laneStart + activeGroups, rewriter);
    if (failed(laneMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create packed group_store lane mask");
    }
    return rewriter
        .create<VselOp>(op.getLoc(), valueType, selected, merged, *laneMask)
        .getResult();
  }

  FailureOr<Value> buildPackedByteStatefulValue(
      VMIGroupStoreOp op, Value merged,
      OneToNPatternRewriter &rewriter) const {
    MLIRContext *ctx = rewriter.getContext();
    auto ui16 = IntegerType::get(
        ctx, 16, IntegerType::SignednessSemantics::Unsigned);
    auto ui8 = IntegerType::get(
        ctx, 8, IntegerType::SignednessSemantics::Unsigned);
    auto packed16Type = VRegType::get(ctx, 128, ui16);
    auto packed8Type = VRegType::get(ctx, 256, ui8);
    Value packed16 = rewriter
                         .create<VpackOp>(op.getLoc(), packed16Type, merged,
                                          rewriter.getStringAttr("LOWER"))
                         .getResult();
    return rewriter
        .create<VpackOp>(op.getLoc(), packed8Type, packed16,
                         rewriter.getStringAttr("LOWER"))
        .getResult();
  }

  void emitPackedByteDirectStore(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter, Value merged,
      Value destination, Value groupOffset, Value storeMask) const {
    rewriter.create<VstsOp>(op.getLoc(), /*updated_base=*/Type{}, merged,
                            destination, groupOffset,
                            rewriter.getStringAttr("PK4_B32"), storeMask);
  }

  LogicalResult emitPackedByteStoreBlocks(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, VRegType firstVRegType, MaskType maskType,
      Value slotIndex, Value destination, Value offset, Value rowStride,
      int64_t numGroups, bool useDirectPack4) const {
    SmallVector<Value> statefulValues;
    SmallVector<int64_t> statefulAdvances;
    for (int64_t blockStart = 0; blockStart < numGroups; blockStart += 32) {
      FailureOr<std::tuple<Value, Value, Value>> block =
          buildPackedByteStoreBlock(
              op, rewriter, valueParts, firstVRegType, maskType, slotIndex,
              destination, offset, rowStride, numGroups, blockStart);
      if (failed(block)) {
        return failure();
      }
      Value merged = std::get<0>(*block);
      Value storeMask = std::get<1>(*block);
      Value groupOffset = std::get<2>(*block);
      if (useDirectPack4) {
        emitPackedByteDirectStore(op, rewriter, merged, destination,
                                  groupOffset, storeMask);
        continue;
      }
      FailureOr<Value> statefulValue =
          buildPackedByteStatefulValue(op, merged, rewriter);
      if (failed(statefulValue)) {
        return failure();
      }
      statefulValues.push_back(*statefulValue);
      statefulAdvances.push_back(
          std::min<int64_t>(32, numGroups - blockStart));
    }
    if (!useDirectPack4 &&
        failed(emitPackedByteStoreStream(op, rewriter, destination, offset,
                                         statefulValues, statefulAdvances))) {
      return failure();
    }
    return success();
  }

  FailureOr<std::pair<MaskType, Value>> buildPackedByteSelectors(
      VMIGroupStoreOp op, VRegType firstVRegType,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<MaskType> maskType =
        getMaskTypeForVReg(firstVRegType, rewriter.getContext());
    FailureOr<Value> allMask =
        createAllTrueMaskForVReg(op.getLoc(), firstVRegType, rewriter);
    bool failedMasks = failed(maskType) || failed(allMask);
    if (failedMasks) {
      (void)rewriter.notifyMatchFailure(
          op, "unsupported element type for packed group_store mask");
      return failure();
    }
    auto indexElementType = IntegerType::get(
        rewriter.getContext(),
        pto::getPTOStorageElemBitWidth(firstVRegType.getElementType()));
    auto indexType = VRegType::get(rewriter.getContext(),
                                   firstVRegType.getElementCount(),
                                   indexElementType);
    FailureOr<Value> slotIndex = createGroupSlotIndexVector(
        op.getLoc(), indexType, /*groupSize=*/8, /*baseGroupSlot=*/0,
        rewriter);
    if (failed(slotIndex)) {
      (void)rewriter.notifyMatchFailure(
          op, "failed to create packed group_store lane selector");
      return failure();
    }
    return std::make_pair(*maskType, *slotIndex);
  }

  LogicalResult lowerPackedByteSlots8(
      VMIGroupStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, VMIVRegType valueVMIType, VMILayoutAttr layout,
      Value destination, Value offset, Value rowStride, int64_t numGroups,
      VRegType firstVRegType) const {
    bool laneStrided = layout.hasLaneStride();
    for (Value value : valueParts) {
      auto vregType = dyn_cast<VRegType>(value.getType());
      if (!vregType || vregType != firstVRegType) {
        return rewriter.notifyMatchFailure(
            op, "packed slots=8 group_store requires uniform vreg parts");
      }
    }
    bool alignedSinglePart =
        !laneStrided && numGroups == 8 && valueParts.size() == 1 &&
        isKnownAddressAligned(destination, offset,
                              valueVMIType.getElementType(), 32);
    if (alignedSinglePart) {
      if (failed(lowerAlignedPackedByteStore(
              op, rewriter, valueParts, destination, offset, numGroups))) {
        return failure();
      }
      rewriter.eraseOp(op);
      return success();
    }

    FailureOr<std::pair<MaskType, Value>> selectors =
        buildPackedByteSelectors(op, firstVRegType, rewriter);
    if (failed(selectors)) {
      return failure();
    }
    bool useDirectPack4 = isDirectMemoryDistAddressLegal(
        op.getDestination(), op.getOffset(),
        getMemoryElementType(op.getDestination().getType()), firstVRegType,
        VPTOMemoryOpFamily::Store, "PK4_B32");
    if (failed(emitPackedByteStoreBlocks(
            op, rewriter, valueParts, firstVRegType, selectors->first,
            selectors->second,
            destination, offset, rowStride, numGroups, useDirectPack4))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

  FailureOr<std::optional<VRegType>> getSlots8FirstVRegType(
      VMIGroupStoreOp op, OpAdaptor adaptor, VMILayoutAttr layout,
      OneToNPatternRewriter &rewriter) const {
    ValueRange valueParts = adaptor.getValue();
    bool hasExpectedArity = static_cast<int64_t>(valueParts.size()) ==
                            ceilDivNonNegative(layout.getNumGroups(), 8);
    if (!hasExpectedArity) {
      (void)rewriter.notifyMatchFailure(op,
                                        "slots=8 group_store arity mismatch");
      return failure();
    }
    if (valueParts.empty()) {
      return std::optional<VRegType>();
    }
    auto firstVRegType = dyn_cast<VRegType>(valueParts.front().getType());
    if (!firstVRegType) {
      (void)rewriter.notifyMatchFailure(op,
                                        "group_store value must be vreg");
      return failure();
    }
    return std::optional<VRegType>(firstVRegType);
  }

  LogicalResult lowerSlots8Dispatch(
      VMIGroupStoreOp op, OpAdaptor adaptor, OneToNPatternRewriter &rewriter,
      VMIVRegType valueVMIType, VMILayoutAttr layout, Value destination,
      Value offset, Value rowStride) const {
    int64_t numGroups = layout.getNumGroups();
    std::optional<int64_t> constantRowStride =
        getConstantIndexValue(op.getRowStride());
    bool hasUnitRowStride = constantRowStride && *constantRowStride == 1;
    if (!hasUnitRowStride) {
      return rewriter.notifyMatchFailure(
          op, "slots=8 group_store requires constant unit row_stride");
    }
    FailureOr<std::optional<VRegType>> firstVRegType =
        getSlots8FirstVRegType(op, adaptor, layout, rewriter);
    if (failed(firstVRegType)) {
      return failure();
    }
    if (*firstVRegType && isPackedByteGroupStore(
                               op.getDestination().getType(), **firstVRegType)) {
      return lowerPackedByteSlots8(
            op, rewriter, adaptor.getValue(), valueVMIType, layout, destination,
            offset, rowStride, numGroups, **firstVRegType);
    }
    if (layout.hasLaneStride()) {
      return lowerSlots8LaneStride(op, adaptor, rewriter, valueVMIType, layout,
                                   destination, offset, rowStride, numGroups);
    }
    return lowerSlots8Contiguous(op, adaptor, rewriter, destination, offset,
                                 rowStride, numGroups);
  }

  enum class GroupStoreLayoutKind { Scalar, Compact, Slots1, Slots8, General };

  GroupStoreLayoutKind classifyGroupStoreLayout(
      VMIGroupStoreOp op, VMIVRegType valueVMIType, VMILayoutAttr layout) const {
    int64_t numGroups = op.getNumGroupsAttr().getInt();
    bool scalar = numGroups == 1 && valueVMIType.getElementCount() == 1;
    if (scalar) {
      return GroupStoreLayoutKind::Scalar;
    }
    bool compact = isCompactSmallGroupStore(
        layout, valueVMIType, numGroups, getConstantIndexValue(op.getRowStride()));
    if (compact) {
      return GroupStoreLayoutKind::Compact;
    }
    bool slots1 = layout && layout.isGroupSlots() && layout.getSlots() == 1 &&
                  layout.getNumGroups() == numGroups;
    if (slots1) {
      return GroupStoreLayoutKind::Slots1;
    }
    bool slots8 = layout && layout.isGroupSlots() && layout.getSlots() == 8 &&
                  layout.getNumGroups() == numGroups;
    return slots8 ? GroupStoreLayoutKind::Slots8 : GroupStoreLayoutKind::General;
  }

  LogicalResult lowerGeneralGroupStore(
      VMIGroupStoreOp op, OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter, VMIVRegType valueVMIType,
      Value destination, Value offset, Value rowStride) const {
    VMILayoutSupport supports;
    FailureOr<VMIGroupStoreLayoutFact> fact =
        supports.getGroupStoreLayoutFact(op, valueVMIType);
    if (failed(fact)) {
      return rewriter.notifyMatchFailure(
          op, "group_store layout does not match the support table");
    }
    if (fact->blockClass == VMIGroupBlockClass::OneBlock) {
      return lowerOneBlockGroupStore(
          op, adaptor, rewriter, valueVMIType, *fact, destination, offset,
          rowStride);
    }
    int64_t d2LanesPerPart = 0;
    int64_t d2GroupCount = 0;
    int64_t d2ChunksPerGroupPerPart = 0;
    std::string d2Reason;
    bool hasDeinterleaved2Shape = succeeded(checkDeinterleaved2GroupStoreChunkShape(
        valueVMIType, fact->groupSize, &d2LanesPerPart, &d2GroupCount,
        &d2ChunksPerGroupPerPart, &d2Reason));
    if (hasDeinterleaved2Shape) {
      return lowerDeinterleaved2GroupStore(
          op, adaptor, rewriter, valueVMIType, *fact, destination, offset,
          rowStride);
    }
    return lowerContiguousGroupStore(
        op, adaptor, rewriter, valueVMIType, *fact, destination, offset,
        rowStride);
  }

  LogicalResult lowerByLayout(
      VMIGroupStoreOp op, OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter, VMIVRegType valueVMIType,
      VMILayoutAttr layout, Value destination, Value offset,
      Value rowStride) const {
    GroupStoreLayoutKind layoutKind =
        classifyGroupStoreLayout(op, valueVMIType, layout);
    if (layoutKind == GroupStoreLayoutKind::Scalar) {
      return lowerScalarGroupStore(op, adaptor, rewriter, valueVMIType,
                                   destination, offset);
    }
    if (layoutKind == GroupStoreLayoutKind::Compact) {
      return lowerCompactSmallGroupStore(op, adaptor, rewriter, valueVMIType,
                                         layout, destination, offset);
    }

    if (layoutKind == GroupStoreLayoutKind::Slots1) {
      return lowerSlots1(op, adaptor, rewriter, valueVMIType, layout,
                         destination, offset, rowStride);
    }
    if (layoutKind == GroupStoreLayoutKind::Slots8) {
      return lowerSlots8Dispatch(op, adaptor, rewriter, valueVMIType, layout,
                                 destination, offset, rowStride);
    }

    return lowerGeneralGroupStore(op, adaptor, rewriter, valueVMIType,
                                  destination, offset, rowStride);
  }

public:
  LogicalResult
  matchAndRewrite(VMIGroupStoreOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
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
    bool operandsConverted = succeeded(destination) && succeeded(offset) &&
                             succeeded(rowStride);
    if (!operandsConverted) {
      return failure();
    }

    // Unified scalar vstore is lowered to group_store(num_groups=1) before
    // layout assignment.  The producer may therefore carry the slots=8
    // layout selected by group_slot_load, even though the logical operation
    // still writes one scalar.  Preserve the scalar memory semantics here;
    // a masked ordinary vsts would require a 32-byte-aligned destination.
    return lowerByLayout(op, adaptor, rewriter, valueVMIType, layout,
                         *destination, *offset, *rowStride);
  }
};

struct OneToNVMIMaskedStoreOpPattern
    : OneToNOpConversionPattern<VMIMaskedStoreOp> {
  using OneToNOpConversionPattern<VMIMaskedStoreOp>::OneToNOpConversionPattern;

private:
  FailureOr<int64_t> emitLaneStrideMaskedStorePart(
      VMIMaskedStoreOp op, OneToNPatternRewriter &rewriter, Value value,
      Value mask, VMIVRegType valueVMIType, Value destination, Value offset,
      StringRef dist, StringRef maskGranularity, int64_t index,
      int64_t semanticOffset) const {
    auto vregType = dyn_cast<VRegType>(value.getType());
    bool invalidTypes = !vregType || !isa<MaskType>(mask.getType());
    if (invalidTypes) {
      return rewriter.notifyMatchFailure(
          op, "lane_stride masked_store parts must be vreg/mask");
    }
    FailureOr<int64_t> activeLanes =
        getActiveDataLanesInPhysicalChunk(valueVMIType, index);
    if (failed(activeLanes)) {
      return rewriter.notifyMatchFailure(
          op, "failed to compute lane_stride masked_store active lanes");
    }
    if (*activeLanes == 0) {
      return 0;
    }
    FailureOr<Value> storeMask = createDenseLaneStrideStorePredicate(
        op.getLoc(), valueVMIType, index, mask, maskGranularity, rewriter);
    if (failed(storeMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to compact lane_stride masked_store predicate");
    }
    Value chunkOffset =
        createChunkOffset(op.getLoc(), offset, semanticOffset, rewriter);
    bool illegalAddress = !isDirectMemoryDistAddressLegal(
        destination, chunkOffset, valueVMIType.getElementType(), vregType,
        VPTOMemoryOpFamily::Store, dist);
    if (illegalAddress) {
      return rewriter.notifyMatchFailure(
          op, "lane_stride masked_store requires a proven target alignment "
              "for every physical store chunk");
    }
    rewriter.create<VstsOp>(op.getLoc(), /*updated_base=*/Type{}, value,
                            destination, chunkOffset,
                            rewriter.getStringAttr(dist), *storeMask);
    return *activeLanes;
  }

  LogicalResult lowerLaneStride(
      VMIMaskedStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, ValueRange maskParts, VMIVRegType valueVMIType,
      VMIMaskType maskVMIType, Value destination, Value offset,
      StringRef dist, StringRef maskGranularity) const {
    VMILayoutAttr valueLayout = valueVMIType.getLayoutAttr();
    VMILayoutAttr maskLayout = maskVMIType.getLayoutAttr();
    if (!valueLayout || !maskLayout || valueLayout != maskLayout) {
      return rewriter.notifyMatchFailure(
          op, "lane_stride masked_store requires matching value/mask layouts");
    }
    int64_t semanticOffset = 0;
    for (auto [index, valueAndMask] :
         llvm::enumerate(llvm::zip_equal(valueParts, maskParts))) {
      auto [value, mask] = valueAndMask;
      FailureOr<int64_t> activeLanes = emitLaneStrideMaskedStorePart(
          op, rewriter, value, mask, valueVMIType, destination, offset, dist,
          maskGranularity, index, semanticOffset);
      if (failed(activeLanes)) {
        return failure();
      }
      semanticOffset += *activeLanes;
    }
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult emitContiguousMaskedStorePart(
      VMIMaskedStoreOp op, OneToNPatternRewriter &rewriter, Value value,
      Value mask, VMIVRegType valueVMIType, Value destination, Value offset,
      int64_t index, int64_t lanesPerPart) const {
    auto vregType = dyn_cast<VRegType>(value.getType());
    bool invalidTypes = !vregType || !isa<MaskType>(mask.getType());
    if (invalidTypes) {
      return rewriter.notifyMatchFailure(
          op, "masked_store converted parts must be vreg/mask");
    }
    FailureOr<int64_t> activeLanes =
        getContiguousActiveDataLanes(valueVMIType, index);
    if (failed(activeLanes)) {
      return rewriter.notifyMatchFailure(
          op, "failed to compute masked_store active lanes");
    }
    if (*activeLanes == 0) {
      return success();
    }
    FailureOr<Value> storeMask = createMaskedStorePredicate(
        op.getLoc(), valueVMIType, index, mask, vregType, rewriter);
    if (failed(storeMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to materialize masked_store predicate");
    }
    Value chunkOffset =
        createChunkOffset(op.getLoc(), offset, index * lanesPerPart, rewriter);
    bool illegalAddress = !isDirectMemoryDistAddressLegal(
        destination, chunkOffset, valueVMIType.getElementType(), vregType,
        VPTOMemoryOpFamily::Store, /*dist=*/{});
    if (illegalAddress) {
      return rewriter.notifyMatchFailure(
          op, "masked_store requires a proven target alignment for every "
              "physical store chunk");
    }
    rewriter.create<VstsOp>(op.getLoc(), /*updated_base=*/Type{}, value,
                            destination, chunkOffset, /*dist=*/nullptr,
                            *storeMask);
    return success();
  }

  FailureOr<std::pair<SmallVector<Value>, SmallVector<Value>>>
  materializeContiguousMaskedStoreParts(
      VMIMaskedStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, ValueRange maskParts, VMIVRegType valueVMIType,
      VMIMaskType maskVMIType) const {
    SmallVector<Type> contiguousValueTypes;
    contiguousValueTypes.reserve(valueParts.size());
    for (Value value : valueParts) {
      contiguousValueTypes.push_back(value.getType());
    }
    VMILayoutAttr contiguousLayout =
        VMILayoutAttr::getContiguous(rewriter.getContext());
    FailureOr<SmallVector<Value>> storeParts = materializeDataLayoutConversion(
        op, valueParts, contiguousValueTypes, valueVMIType.getLayoutAttr(),
        contiguousLayout, valueVMIType.getElementType(), rewriter);
    if (failed(storeParts)) {
      return failure();
    }
    SmallVector<Type> contiguousMaskTypes;
    contiguousMaskTypes.reserve(maskParts.size());
    for (Value mask : maskParts) {
      contiguousMaskTypes.push_back(mask.getType());
    }
    FailureOr<SmallVector<Value>> storeMasks = materializeMaskLayoutConversion(
        op, maskParts, contiguousMaskTypes, maskVMIType.getLayoutAttr(),
        contiguousLayout, rewriter);
    if (failed(storeMasks)) {
      return failure();
    }
    bool mismatchedArity = storeParts->size() != storeMasks->size();
    if (mismatchedArity) {
      return rewriter.notifyMatchFailure(
          op, "masked_store converted value/mask arity mismatch");
    }
    return std::make_pair(std::move(*storeParts), std::move(*storeMasks));
  }

  LogicalResult lowerContiguous(
      VMIMaskedStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, ValueRange maskParts, VMIVRegType valueVMIType,
      VMIMaskType maskVMIType, Value destination, Value offset,
      int64_t lanesPerPart) const {
    FailureOr<std::pair<SmallVector<Value>, SmallVector<Value>>> converted =
        materializeContiguousMaskedStoreParts(op, rewriter, valueParts,
                                              maskParts, valueVMIType,
                                              maskVMIType);
    if (failed(converted)) {
      return failure();
    }

    for (auto [index, valueAndMask] :
         llvm::enumerate(llvm::zip_equal(converted->first,
                                         converted->second))) {
      auto [value, mask] = valueAndMask;
      if (failed(emitContiguousMaskedStorePart(
              op, rewriter, value, mask, valueVMIType, destination, offset,
              index, lanesPerPart))) {
        return failure();
      }
    }
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult lowerByLayout(
      VMIMaskedStoreOp op, OneToNPatternRewriter &rewriter,
      ValueRange valueParts, ValueRange maskParts, VMIVRegType valueVMIType,
      VMIMaskType maskVMIType, Value destination, Value offset,
      int64_t lanesPerPart) const {
    std::optional<std::string> dist =
        getDenseLaneStrideStoreDistToken(valueVMIType);
    if (dist) {
      std::optional<StringRef> maskGranularity =
          getDenseLaneStrideMaskedStoreMaskGranularity(valueVMIType);
      if (maskGranularity) {
        return lowerLaneStride(op, rewriter, valueParts, maskParts,
                               valueVMIType, maskVMIType, destination, offset,
                               *dist, *maskGranularity);
      }
    }
    return lowerContiguous(op, rewriter, valueParts, maskParts, valueVMIType,
                           maskVMIType, destination, offset, lanesPerPart);
  }

public:

  LogicalResult
  matchAndRewrite(VMIMaskedStoreOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto valueVMIType = cast<VMIVRegType>(op.getValue().getType());
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(valueVMIType.getElementType());
    if (failed(lanesPerPart)) {
      return rewriter.notifyMatchFailure(
          op, "masked_store requires known physical lanes per part");
    }

    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "masked_store destination must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(),
        "masked_store offset must convert to one value", rewriter);
    bool invalidAddressOperands = failed(destination) || failed(offset);
    if (invalidAddressOperands) {
      return failure();
    }

    ValueRange valueParts = adaptor.getValue();
    ValueRange maskParts = adaptor.getMask();
    bool arityMismatch = valueParts.size() != maskParts.size();
    if (arityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "masked_store value/mask physical arity mismatch");
    }

    auto maskVMIType = cast<VMIMaskType>(op.getMask().getType());
    return lowerByLayout(op, rewriter, valueParts, maskParts, valueVMIType,
                         maskVMIType, *destination, *offset, *lanesPerPart);
  }
};

struct OneToNVMIGroupBroadcastLoadOpPattern
    : OneToNOpConversionPattern<VMIGroupBroadcastLoadOp> {
  using OneToNOpConversionPattern<VMIGroupBroadcastLoadOp>::OneToNOpConversionPattern;

private:
  LogicalResult validateDirectE2BBasicContract(
      VMIGroupBroadcastLoadOp op, Value source, int64_t numGroups,
      unsigned elementBits, VMILayoutAttr layout,
      OneToNPatternRewriter &rewriter) const {
    bool contiguousPacketLayout = layout && layout.isContiguous();
    bool splitPacketLayout = layout && layout.isDeinterleaved() &&
                             (layout.getFactor() == 2 ||
                              layout.getFactor() == 4) &&
                             layout.getLaneStride() == 1;
    if (!contiguousPacketLayout && !splitPacketLayout) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load E2B lowering requires contiguous result "
              "layout for direct group size or deinterleaved=2/4 result "
              "layout for split group size");
    }
    if (elementBits != 16 && elementBits != 32) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load E2B lowering requires b16 or b32 element type");
    }
    std::optional<int64_t> stride =
        getConstantIndexValue(op.getSourceGroupStride());
    if (!stride || *stride != 1) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load E2B lowering requires constant unit source_group_stride");
    }
    if (!isa<PtrType>(source.getType())) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load E2B lowering requires !pto.ptr source");
    }
    if (numGroups != 8) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load E2B lowering requires num_groups = 8");
    }
    return success();
  }

  FailureOr<Value> emitE2BPacket(
      VMIGroupBroadcastLoadOp op, OneToNPatternRewriter &rewriter,
      Value source, Value offset, Type packetType, int64_t chunk,
      StringRef e2bDist) const {
    if (!isa<VRegType>(packetType)) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load result must be vreg");
    }
    Value packetOffset =
        createChunkOffset(op.getLoc(), offset, chunk * 8, rewriter);
    return rewriter
        .create<VldsOp>(op.getLoc(), packetType, Type{}, source, packetOffset,
                        rewriter.getStringAttr(e2bDist))
        .getResult();
  }

  FailureOr<SmallVector<Value>> emitE2BPackets(
      VMIGroupBroadcastLoadOp op, OneToNPatternRewriter &rewriter,
      Value source, Value offset, ArrayRef<Type> resultTypes,
      int64_t chunksPerPart, StringRef e2bDist) const {
    SmallVector<Value> packets;
    packets.reserve(chunksPerPart);
    for (int64_t chunk = 0; chunk < chunksPerPart; ++chunk) {
      FailureOr<Value> packet = emitE2BPacket(
          op, rewriter, source, offset, resultTypes[chunk], chunk, e2bDist);
      if (failed(packet)) {
        return failure();
      }
      packets.push_back(*packet);
    }
    return packets;
  }

  FailureOr<SmallVector<Value>> buildE2BResults(
      VMIGroupBroadcastLoadOp op, ArrayRef<Value> packets,
      ArrayRef<Type> resultTypes, int64_t factor, int64_t chunksPerPart,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (int64_t part = 0; part < factor; ++part) {
      for (int64_t chunk = 0; chunk < chunksPerPart; ++chunk) {
        int64_t flatIndex = part * chunksPerPart + chunk;
        if (resultTypes[flatIndex] != resultTypes[chunk]) {
          return rewriter.notifyMatchFailure(
              op, "group_broadcast_load E2B reused packet type mismatch");
        }
        results.push_back(packets[chunk]);
      }
    }
    return results;
  }

  FailureOr<std::tuple<StringRef, int64_t, int64_t>> validateDirectE2BShape(
      VMIGroupBroadcastLoadOp op, Value source,
      VMIVRegType resultVMIType, ArrayRef<Type> resultTypes,
      int64_t numGroups, unsigned elementBits, VMILayoutAttr layout,
      OneToNPatternRewriter &rewriter) const {
    if (failed(validateDirectE2BBasicContract(op, source, numGroups,
                                              elementBits, layout, rewriter))) {
      return failure();
    }
    FailureOr<int64_t> chunksPerPart = getDataChunksInPart(resultVMIType, 0);
    bool invalidChunks = failed(chunksPerPart) || *chunksPerPart <= 0;
    if (invalidChunks) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load requires known chunks per part");
    }
    int64_t factor = layout.getFactor();
    FailureOr<int64_t> uniformChunks = validateDirectE2BChunks(
        op, resultVMIType, factor, *chunksPerPart, rewriter);
    if (failed(uniformChunks)) {
      return failure();
    }
    bool invalidArity =
        static_cast<int64_t>(resultTypes.size()) != factor * *chunksPerPart;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load physical arity mismatch");
    }
    if (*chunksPerPart != 1) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load expected one E2B packet in each part");
    }
    StringRef e2bDist = elementBits == 16 ? "E2B_B16" : "E2B_B32";
    return std::make_tuple(e2bDist, factor, *chunksPerPart);
  }

  FailureOr<int64_t> validateDirectE2BChunks(
      VMIGroupBroadcastLoadOp op, VMIVRegType resultVMIType, int64_t factor,
      int64_t chunksPerPart, OneToNPatternRewriter &rewriter) const {
    for (int64_t part = 1; part < factor; ++part) {
      FailureOr<int64_t> currentChunks =
          getDataChunksInPart(resultVMIType, part);
      bool nonUniformChunks =
          failed(currentChunks) || *currentChunks != chunksPerPart;
      if (nonUniformChunks) {
        return rewriter.notifyMatchFailure(
            op, "group_broadcast_load requires uniform chunks per part");
      }
    }
    return chunksPerPart;
  }

  LogicalResult lowerDirectE2B(
      VMIGroupBroadcastLoadOp op, OneToNPatternRewriter &rewriter,
      Value source, Value offset, VMIVRegType resultVMIType,
      ArrayRef<Type> resultTypes, int64_t numGroups, unsigned elementBits,
      VMILayoutAttr layout) const {
    FailureOr<std::tuple<StringRef, int64_t, int64_t>> shape =
        validateDirectE2BShape(op, source, resultVMIType, resultTypes,
                               numGroups, elementBits, layout, rewriter);
    if (failed(shape)) {
      return failure();
    }
    StringRef e2bDist = std::get<0>(*shape);
    int64_t factor = std::get<1>(*shape);
    int64_t chunksPerPart = std::get<2>(*shape);
    FailureOr<SmallVector<Value>> packets = emitE2BPackets(
        op, rewriter, source, offset, resultTypes, chunksPerPart, e2bDist);
    if (failed(packets)) {
      return failure();
    }
    FailureOr<SmallVector<Value>> results = buildE2BResults(
        op, *packets, resultTypes, factor, chunksPerPart, rewriter);
    if (failed(results)) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildDirectBRCResult(
      VMIGroupBroadcastLoadOp op, OneToNPatternRewriter &rewriter,
      Value source, Value offset, Value sourceGroupStride, Type resultType,
      int64_t group, StringRef brcDist) const {
    auto vregType = dyn_cast<VRegType>(resultType);
    if (!vregType) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load BRC result must be vreg");
    }
    Value groupOffset = createGroupChunkOffset(
        op.getLoc(), offset, sourceGroupStride, group, 0, rewriter);
    return rewriter
        .create<VldsOp>(op.getLoc(), resultType, Type{}, source, groupOffset,
                        rewriter.getStringAttr(brcDist))
        .getResult();
  }

  FailureOr<int64_t> validateDirectBRCShape(
      VMIGroupBroadcastLoadOp op, Value source, ArrayRef<Type> resultTypes,
      int64_t numGroups, OneToNPatternRewriter &rewriter) const {
    if (numGroups <= 0) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load BRC requires positive num_groups");
    }
    int64_t safeNumGroups = numGroups;
    bool invalidArity =
        static_cast<int64_t>(resultTypes.size()) % safeNumGroups != 0;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load BRC result arity is not divisible by num_groups");
    }
    if (!isa<PtrType>(source.getType())) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load BRC lowering requires !pto.ptr source");
    }
    int64_t chunksPerGroup =
        static_cast<int64_t>(resultTypes.size()) / safeNumGroups;
    bool invalidChunkArity =
        chunksPerGroup <= 0 ||
        static_cast<int64_t>(resultTypes.size()) !=
            safeNumGroups * chunksPerGroup;
    if (invalidChunkArity) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load BRC physical arity mismatch");
    }
    return chunksPerGroup;
  }

  FailureOr<std::pair<VMIVRegType, SmallVector<Type>>>
  buildGroupBroadcastFallbackSourcePlan(
      VMIGroupBroadcastLoadOp op, VMIVRegType resultVMIType,
      int64_t numGroups, OneToNPatternRewriter &rewriter) const {
    std::optional<int64_t> stride =
        getConstantIndexValue(op.getSourceGroupStride());
    int64_t slots = (stride && *stride == 1) ? 8 : 1;
    auto sourceVMIType = VMIVRegType::get(
        rewriter.getContext(), numGroups, resultVMIType.getElementType(),
        VMILayoutAttr::getGroupSlots(rewriter.getContext(), numGroups, slots));
    FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceVMIType);
    if (failed(sourceArity)) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load fallback cannot derive physical types");
    }
    Type sourceElementType = getVMIPhysicalDataElementType(sourceVMIType);
    FailureOr<int64_t> sourceLanesPerPart =
        getDataLanesPerPart(sourceElementType);
    if (failed(sourceLanesPerPart)) {
      return rewriter.notifyMatchFailure(
          op, "group_broadcast_load fallback cannot derive source lanes");
    }
    SmallVector<Type> sourceTypes;
    sourceTypes.reserve(*sourceArity);
    for (int64_t i = 0; i < *sourceArity; ++i) {
      sourceTypes.push_back(VRegType::get(
          rewriter.getContext(), *sourceLanesPerPart, sourceElementType));
    }
    return std::make_pair(sourceVMIType, std::move(sourceTypes));
  }

  LogicalResult lowerDirectBRC(
      VMIGroupBroadcastLoadOp op, OneToNPatternRewriter &rewriter,
      Value source, Value offset, Value sourceGroupStride,
      ArrayRef<Type> resultTypes, int64_t numGroups,
      StringRef brcDist) const {
    FailureOr<int64_t> chunksPerGroup = validateDirectBRCShape(
        op, source, resultTypes, numGroups, rewriter);
    if (failed(chunksPerGroup)) {
      return failure();
    }

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
      int64_t group = static_cast<int64_t>(index) / *chunksPerGroup;
      FailureOr<Value> result = buildDirectBRCResult(
          op, rewriter, source, offset, sourceGroupStride, resultType, group,
          brcDist);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  LogicalResult lowerGroupSlotFallback(
      VMIGroupBroadcastLoadOp op, OneToNPatternRewriter &rewriter,
      Value source, Value offset, Value sourceGroupStride,
      VMIVRegType resultVMIType, ArrayRef<Type> resultTypes,
      int64_t numGroups) const {
    FailureOr<std::pair<VMIVRegType, SmallVector<Type>>> sourcePlan =
        buildGroupBroadcastFallbackSourcePlan(op, resultVMIType, numGroups,
                                              rewriter);
    if (failed(sourcePlan)) {
      return failure();
    }
    SmallVector<Value> sourceParts;
    if (failed(lowerGroupSlotLoadParts(
            op, source, offset, sourceGroupStride, sourcePlan->first,
            sourcePlan->second, numGroups, rewriter, sourceParts))) {
      return failure();
    }
    SmallVector<Value> results;
    if (failed(lowerGroupBroadcastParts(
            op, sourceParts, sourcePlan->first, resultVMIType, resultTypes,
            numGroups, rewriter, results))) {
      return failure();
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  FailureOr<bool> tryLowerDirectBRC(
      VMIGroupBroadcastLoadOp op, OneToNPatternRewriter &rewriter,
      Value source, Value offset, Value sourceGroupStride,
      VMIVRegType resultVMIType, ArrayRef<Type> resultTypes, int64_t numGroups,
      const FailureOr<VMIGroupBroadcastLoadDirectFact> &directFact) const {
    bool candidate = succeeded(directFact) &&
                     directFact->kind == VMIGroupBroadcastLoadDirectKind::BRC &&
                     !resultTypes.empty();
    if (!candidate) {
      return false;
    }
    unsigned bits =
        pto::getPTOStorageElemBitWidth(resultVMIType.getElementType());
    std::optional<StringRef> dist;
    if (bits == 8) {
      dist = StringRef("BRC_B8");
    } else if (bits == 16) {
      dist = StringRef("BRC_B16");
    } else if (bits == 32) {
      dist = StringRef("BRC_B32");
    }
    auto firstType = dyn_cast<VRegType>(resultTypes.front());
    bool legal = dist && firstType && isDirectMemoryDistAddressLegal(
                                  op.getSource(), op.getOffset(),
                                  resultVMIType.getElementType(), firstType,
                                  VPTOMemoryOpFamily::Load, *dist);
    if (!legal) {
      return false;
    }
    if (failed(lowerDirectBRC(op, rewriter, source, offset, sourceGroupStride,
                              resultTypes, numGroups, *dist))) {
      return failure();
    }
    return true;
  }

  FailureOr<bool> tryLowerDirectE2B(
      VMIGroupBroadcastLoadOp op, OneToNPatternRewriter &rewriter,
      Value source, Value offset, VMIVRegType resultVMIType,
      ArrayRef<Type> resultTypes, int64_t numGroups,
      const FailureOr<VMIGroupBroadcastLoadDirectFact> &directFact) const {
    bool candidate = succeeded(directFact) &&
                     directFact->kind == VMIGroupBroadcastLoadDirectKind::E2B &&
                     !resultTypes.empty();
    if (!candidate) {
      return false;
    }
    unsigned bits = directFact->layout.elementBits;
    StringRef dist = bits == 16 ? StringRef("E2B_B16") : StringRef("E2B_B32");
    auto firstType = dyn_cast<VRegType>(resultTypes.front());
    bool legal = (bits == 16 || bits == 32) && firstType &&
                 isDirectMemoryDistAddressLegal(
                     op.getSource(), op.getOffset(),
                     resultVMIType.getElementType(), firstType,
                     VPTOMemoryOpFamily::Load, dist);
    if (!legal) {
      return false;
    }
    // E2B materializes one packet per physical result chunk.  A contiguous
    // result spanning multiple chunks cannot reuse one packet for every
    // chunk; route it through the group-slot fallback instead.
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    FailureOr<int64_t> chunksPerPart = getDataChunksInPart(resultVMIType, 0);
    bool requiresGroupSlotFallback =
        resultLayout && resultLayout.isContiguous() &&
        (failed(chunksPerPart) || *chunksPerPart != 1);
    if (requiresGroupSlotFallback) {
      return false;
    }
    if (failed(lowerDirectE2B(op, rewriter, source, offset, resultVMIType,
                              resultTypes, numGroups, bits,
                              resultVMIType.getLayoutAttr()))) {
      return failure();
    }
    return true;
  }

  LogicalResult lowerDirectOrFallback(
      VMIGroupBroadcastLoadOp op, OneToNPatternRewriter &rewriter,
      Value source, Value offset, Value sourceGroupStride,
      VMIVRegType resultVMIType, ArrayRef<Type> resultTypes, int64_t numGroups,
      const FailureOr<VMIGroupBroadcastLoadDirectFact> &directFact) const {
    FailureOr<bool> loweredBRC = tryLowerDirectBRC(
        op, rewriter, source, offset, sourceGroupStride, resultVMIType,
        resultTypes, numGroups, directFact);
    if (failed(loweredBRC)) {
      return failure();
    }
    if (*loweredBRC) {
      return success();
    }

    FailureOr<bool> loweredE2B = tryLowerDirectE2B(
        op, rewriter, source, offset, resultVMIType, resultTypes, numGroups,
        directFact);
    if (failed(loweredE2B)) {
      return failure();
    }
    if (*loweredE2B) {
      return success();
    }
    return lowerGroupSlotFallback(op, rewriter, source, offset,
                                  sourceGroupStride, resultVMIType, resultTypes,
                                  numGroups);
  }

public:
  LogicalResult
  matchAndRewrite(VMIGroupBroadcastLoadOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
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
    bool invalidOperands =
        failed(source) || failed(offset) || failed(sourceGroupStride);
    if (invalidOperands) {
      return failure();
    }

    VMILayoutSupport supports;
    std::string supportReason;
    FailureOr<VMIGroupBroadcastLoadLayoutFact> loadFact =
        supports.getGroupBroadcastLoadLayoutFact(op, &supportReason);
    if (failed(loadFact)) {
      return rewriter.notifyMatchFailure(
          op, Twine("group_broadcast_load has no registered support: ") +
                  supportReason);
    }

    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }

    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    FailureOr<VMIGroupBroadcastLoadDirectFact> directFact =
        supports.getGroupBroadcastLoadDirectFact(op);
    return lowerDirectOrFallback(op, rewriter, *source, *offset,
                                 *sourceGroupStride, resultVMIType, resultTypes,
                                 numGroups, directFact);
  }
};

struct OneToNVMIStrideLoadOpPattern
    : OneToNOpConversionPattern<VMIStrideLoadOp> {
  using OneToNOpConversionPattern<VMIStrideLoadOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerStrideLoad(
      VMIStrideLoadOp op, OneToNPatternRewriter &rewriter, Value source,
      Value offset, Value blockStride, Value repeatStride, ValueRange maskParts,
      ArrayRef<Type> resultTypes) const {
    bool invalidPhysicalArity = resultTypes.size() != 1 || maskParts.size() != 1;
    if (invalidPhysicalArity) {
      return rewriter.notifyMatchFailure(
          op, "stride_load supports one physical result/mask chunk");
    }
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    if (!resultType || !isa<MaskType>(maskParts.front().getType())) {
      return rewriter.notifyMatchFailure(
          op, "stride_load requires physical vreg/mask parts");
    }
    Value base = rewriter
                     .create<AddPtrOp>(op.getLoc(), source.getType(), source,
                                       offset)
                     .getResult();
    Value loaded = rewriter
                       .create<VsldbOp>(op.getLoc(), resultType,
                                        /*updated_base=*/Type{}, base,
                                        blockStride, repeatStride,
                                        maskParts.front())
                       .getResult();
    replaceOpWithFlatConvertedValues(rewriter, op, SmallVector<Value>{loaded},
                                     *this->getTypeConverter());
    return success();
  }

public:

  LogicalResult
  matchAndRewrite(VMIStrideLoadOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<Value> source = getSingleValue(
        op, adaptor.getSource(), "stride_load source must convert to one value",
        rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(), "stride_load offset must convert to one value",
        rewriter);
    FailureOr<Value> blockStride = getSingleValue(
        op, adaptor.getBlockStride(),
        "stride_load block_stride must convert to one value", rewriter);
    FailureOr<Value> repeatStride = Value(
        rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 16));
    bool invalidOperands = failed(source) || failed(offset) ||
                           failed(blockStride) || failed(repeatStride);
    if (invalidOperands) {
      return failure();
    }

    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    return lowerStrideLoad(op, rewriter, *source, *offset, *blockStride,
                           *repeatStride, maskParts, resultTypes);
  }
};

struct OneToNVMIStrideStoreOpPattern
    : OneToNOpConversionPattern<VMIStrideStoreOp> {
  using OneToNOpConversionPattern<VMIStrideStoreOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIStrideStoreOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "stride_store destination must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(),
        "stride_store offset must convert to one value", rewriter);
    FailureOr<Value> blockStride = getSingleValue(
        op, adaptor.getBlockStride(),
        "stride_store block_stride must convert to one value", rewriter);
    FailureOr<Value> repeatStride = Value(
        rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 16));
    bool failedOperands = failed(destination) || failed(offset) ||
                          failed(blockStride) || failed(repeatStride);
    if (failedOperands) {
      return failure();
    }

    ValueRange valueParts = adaptor.getValue();
    ValueRange maskParts = adaptor.getMask();
    bool invalidArity = valueParts.size() != 1 || maskParts.size() != 1;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "stride_store supports one physical value/mask chunk");
    }
    bool invalidTypes = !isa<VRegType>(valueParts.front().getType()) ||
                        !isa<MaskType>(maskParts.front().getType());
    if (invalidTypes) {
      return rewriter.notifyMatchFailure(
          op, "stride_store requires physical vreg/mask parts");
    }

    Value base = rewriter
                     .create<AddPtrOp>(op.getLoc(), (*destination).getType(),
                                       *destination, *offset)
                     .getResult();
    rewriter.create<VsstbOp>(op.getLoc(), /*updated_base=*/Type{},
                             valueParts.front(), base, *blockStride,
                             *repeatStride, maskParts.front());
    rewriter.eraseOp(op);
    return success();
  }
};

struct OneToNVMIScatterOpPattern : OneToNOpConversionPattern<VMIScatterOp> {
  using OneToNOpConversionPattern<VMIScatterOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIScatterOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "scatter destination must convert to one value", rewriter);
    if (failed(destination)) {
      return failure();
    }

    ValueRange valueParts = adaptor.getValue();
    ValueRange indicesParts = adaptor.getIndices();
    ValueRange maskParts = adaptor.getMask();
    bool invalidArity = valueParts.size() != indicesParts.size() ||
                        valueParts.size() != maskParts.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(op, "scatter physical arity mismatch");
    }

    for (auto [value, indices, mask] :
         llvm::zip_equal(valueParts, indicesParts, maskParts)) {
      bool invalidTypes = !isa<VRegType>(value.getType()) ||
                          !isa<VRegType>(indices.getType()) ||
                          !isa<MaskType>(mask.getType());
      if (invalidTypes) {
        return rewriter.notifyMatchFailure(
            op, "scatter physical part type mismatch");
      }
      rewriter.create<VscatterOp>(op.getLoc(), value, *destination, indices,
                                  mask);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

template <typename SourceOp, typename TargetOp, bool IsMaskResult = false>
struct OneToNVMIBinaryOpPattern : OneToNOpConversionPattern<SourceOp> {
  using OneToNOpConversionPattern<SourceOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> lowerBinaryPart(
      SourceOp op, Value lhs, Value rhs, Type resultType,
      OneToNPatternRewriter &rewriter) const {
    if constexpr (IsMaskResult) {
      auto maskType = dyn_cast<MaskType>(resultType);
      bool invalidTypes =
          !maskType || lhs.getType() != resultType || rhs.getType() != resultType;
      if (invalidTypes) {
        return rewriter.notifyMatchFailure(
            op, "physical mask binary part type mismatch");
      }
      FailureOr<Value> seedMask =
          createAllTrueMask(op.getLoc(), maskType, rewriter);
      if (failed(seedMask)) {
        return rewriter.notifyMatchFailure(
            op, "unsupported mask type for all-true mask binary seed");
      }
      return rewriter
          .create<TargetOp>(op.getLoc(), resultType, lhs, rhs, *seedMask)
          .getResult();
    }
    auto vregType = dyn_cast<VRegType>(resultType);
    bool invalidTypes =
        !vregType || lhs.getType() != resultType || rhs.getType() != resultType;
    if (invalidTypes) {
      return rewriter.notifyMatchFailure(
          op, "physical binary part type mismatch");
    }
    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for all-true binary mask");
    }
    return rewriter
        .create<TargetOp>(op.getLoc(), resultType, lhs, rhs, *mask)
        .getResult();
  }

  LogicalResult lowerBinaryParts(
      SourceOp op, ValueRange lhsParts, ValueRange rhsParts,
      ArrayRef<Type> resultTypes, OneToNPatternRewriter &rewriter) const {
    bool invalidArity = lhsParts.size() != rhsParts.size() ||
                        lhsParts.size() != resultTypes.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, IsMaskResult ? "physical mask binary arity mismatch"
                           : "physical binary arity mismatch");
    }
    return lowerPointwisePhysicalParts(
        op, resultTypes,
        IsMaskResult ? "physical mask binary arity mismatch"
                     : "physical binary arity mismatch",
        rewriter,
        [&](int64_t index, Type resultType) -> FailureOr<Value> {
          return lowerBinaryPart(op, lhsParts[index], rhsParts[index],
                                 resultType, rewriter);
        },
        *this->getTypeConverter());
  }

public:
  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OneToNOpConversionPattern<SourceOp>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const override {
    ValueRange lhsParts = adaptor.getLhs();
    ValueRange rhsParts = adaptor.getRhs();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    return lowerBinaryParts(op, lhsParts, rhsParts, resultTypes, rewriter);
  }
};

// VPTO vector shifts require a signed shift-count carrier regardless of the
// signedness of the value being shifted. Preserve the count bits with a
// bitcast before creating the physical shift operation.
template <typename SourceOp, typename TargetOp>
struct OneToNVMIShiftOpPattern : OneToNOpConversionPattern<SourceOp> {
  using OneToNOpConversionPattern<SourceOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> lowerShiftPart(
      SourceOp op, Value lhs, Value rhs, Type resultType,
      OneToNPatternRewriter &rewriter) const {
    auto resultVRegType = dyn_cast<VRegType>(resultType);
    auto rhsVRegType = dyn_cast<VRegType>(rhs.getType());
    auto rhsElementType =
        rhsVRegType ? dyn_cast<IntegerType>(rhsVRegType.getElementType())
                    : IntegerType();
    bool invalidPhysicalPart =
        !resultVRegType || lhs.getType() != resultType || !rhsElementType;
    if (invalidPhysicalPart) {
      return rewriter.notifyMatchFailure(op, "physical shift part type mismatch");
    }
    auto signedElementType = IntegerType::get(
        rewriter.getContext(), rhsElementType.getWidth(),
        IntegerType::SignednessSemantics::Signed);
    auto signedRhsType = VRegType::get(
        rewriter.getContext(), rhsVRegType.getElementCount(), signedElementType);
    FailureOr<Value> signedRhs =
        bitcastVReg(op.getLoc(), rhs, signedRhsType, rewriter);
    if (failed(signedRhs)) {
      return rewriter.notifyMatchFailure(
          op, "unable to normalize physical shift-count type");
    }
    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), resultVRegType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for all-true shift mask");
    }
    return rewriter
        .create<TargetOp>(op.getLoc(), resultType, lhs, *signedRhs, *mask)
        .getResult();
  }

public:

  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OneToNOpConversionPattern<SourceOp>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const override {
    ValueRange lhsParts = adaptor.getLhs();
    ValueRange rhsParts = adaptor.getRhs();
    FailureOr<SmallVector<Type>> maybeResultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybeResultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybeResultTypes);
    const bool hasMismatchedPhysicalArity =
        lhsParts.size() != rhsParts.size() ||
        lhsParts.size() != resultTypes.size();
    if (hasMismatchedPhysicalArity) {
      return rewriter.notifyMatchFailure(op, "physical shift arity mismatch");
    }

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [lhs, rhs, resultType] :
         llvm::zip_equal(lhsParts, rhsParts, resultTypes)) {
      FailureOr<Value> shifted =
          lowerShiftPart(op, lhs, rhs, resultType, rewriter);
      if (failed(shifted)) {
        return failure();
      }
      results.push_back(*shifted);
    }

    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }
};

template <typename SourceOp, typename TargetOp>
struct OneToNVMIVecScalarOpPattern : OneToNOpConversionPattern<SourceOp> {
  using OneToNOpConversionPattern<SourceOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerVectorScalarParts(
      SourceOp op, ValueRange sourceParts, Value scalar,
      ValueRange maskParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    const bool invalidArity =
        sourceParts.empty() || sourceParts.size() != maskParts.size() ||
        sourceParts.size() != resultTypes.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "physical vector-scalar arity mismatch");
    }

    return lowerPointwisePhysicalParts(
        op, resultTypes, "physical vector-scalar arity mismatch", rewriter,
        [&](int64_t index, Type resultType) -> FailureOr<Value> {
          auto vregType = dyn_cast<VRegType>(resultType);
          auto maskType = dyn_cast<MaskType>(maskParts[index].getType());
          const bool hasMismatchedPartType =
              !vregType || !maskType ||
              sourceParts[index].getType() != resultType;
          if (hasMismatchedPartType) {
            return rewriter.notifyMatchFailure(
                op, "physical vector-scalar part type mismatch");
          }
          return rewriter
              .create<TargetOp>(op.getLoc(), resultType, sourceParts[index],
                                scalar, maskParts[index])
              .getResult();
        },
        *this->getTypeConverter());
  }

public:
  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OneToNOpConversionPattern<SourceOp>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const override {
    const bool requiresPassthru =
        op.getPmode().has_value() && *op.getPmode() == "merge";
    if (requiresPassthru) {
      return rewriter.notifyMatchFailure(
          op, "merge predicate mode requires an explicit passthru lowering");
    }

    ValueRange sourceParts = adaptor.getSrc();
    FailureOr<Value> scalar =
        getSingleValue(op, adaptor.getScalar(),
                       "vector-scalar scalar must convert to one value",
                       rewriter);
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybeResultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    const bool conversionFailed = failed(scalar) || failed(maybeResultTypes);
    if (conversionFailed) {
      return failure();
    }
    Value scalarValue = *scalar;
    SmallVector<Type> resultTypes = std::move(*maybeResultTypes);
    return lowerVectorScalarParts(op, sourceParts, scalarValue, maskParts,
                                  resultTypes, rewriter);
  }
};

struct OneToNVMIVaddcOpPattern : OneToNOpConversionPattern<VMIVaddcOp> {
  using OneToNOpConversionPattern<VMIVaddcOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerParts(VMIVaddcOp op, ValueRange lhsParts,
                           ValueRange rhsParts, ValueRange maskParts,
                           ArrayRef<Type> resultTypes,
                           ArrayRef<Type> carryTypes,
                           SmallVectorImpl<Value> &results,
                           SmallVectorImpl<Value> &carries,
                           OneToNPatternRewriter &rewriter) const {
    const bool invalidArity =
        lhsParts.empty() || rhsParts.size() != lhsParts.size() ||
        maskParts.size() != lhsParts.size() ||
        resultTypes.size() != lhsParts.size() ||
        carryTypes.size() != lhsParts.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(op, "vaddc physical arity mismatch");
    }
    for (auto [lhs, rhs, mask, resultType, carryType] :
         llvm::zip_equal(lhsParts, rhsParts, maskParts, resultTypes,
                         carryTypes)) {
      auto dataType = dyn_cast<VRegType>(resultType);
      auto integerType = dataType
                             ? dyn_cast<IntegerType>(dataType.getElementType())
                             : IntegerType();
      const bool invalidPart =
          !dataType || !integerType || integerType.getWidth() != 32 ||
          !isa<MaskType>(mask.getType()) || !isa<MaskType>(carryType) ||
          !cast<MaskType>(carryType).isB32() || lhs.getType() != resultType ||
          rhs.getType() != resultType;
      if (invalidPart) {
        return rewriter.notifyMatchFailure(
            op, "vaddc requires matching 32-bit data and b32 mask parts");
      }
      auto addc = rewriter.create<VaddcOp>(op.getLoc(), resultType, carryType,
                                           lhs, rhs, mask);
      results.push_back(addc.getResult());
      carries.push_back(addc.getCarry());
    }
    return success();
  }

public:
  LogicalResult matchAndRewrite(VMIVaddcOp op, OpAdaptor adaptor,
                                OneToNPatternRewriter &rewriter) const override {
    ValueRange lhsParts = adaptor.getLhs();
    ValueRange rhsParts = adaptor.getRhs();
    ValueRange maskParts = adaptor.getMask();
    return lowerCarryResultParts(
        op, rewriter, *this->getTypeConverter(),
        [&](ArrayRef<Type> resultTypes, ArrayRef<Type> carryTypes,
            SmallVectorImpl<Value> &results,
            SmallVectorImpl<Value> &carries) {
          return lowerParts(op, lhsParts, rhsParts, maskParts, resultTypes,
                            carryTypes, results, carries, rewriter);
        });
  }
};

struct OneToNVMIVaddcsOpPattern : OneToNOpConversionPattern<VMIVaddcsOp> {
  using OneToNOpConversionPattern<VMIVaddcsOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerParts(VMIVaddcsOp op, ValueRange lhsParts,
                           ValueRange rhsParts, ValueRange carryInParts,
                           ValueRange maskParts, ArrayRef<Type> resultTypes,
                           ArrayRef<Type> carryTypes,
                           SmallVectorImpl<Value> &results,
                           SmallVectorImpl<Value> &carries,
                           OneToNPatternRewriter &rewriter) const {
    const bool invalidArity =
        lhsParts.empty() || rhsParts.size() != lhsParts.size() ||
        carryInParts.size() != lhsParts.size() ||
        maskParts.size() != lhsParts.size() ||
        resultTypes.size() != lhsParts.size() ||
        carryTypes.size() != lhsParts.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(op,
                                         "vaddcs physical arity mismatch");
    }
    for (auto [lhs, rhs, carryIn, mask, resultType, carryType] :
         llvm::zip_equal(lhsParts, rhsParts, carryInParts, maskParts,
                         resultTypes, carryTypes)) {
      auto dataType = dyn_cast<VRegType>(resultType);
      auto integerType = dataType
                             ? dyn_cast<IntegerType>(dataType.getElementType())
                             : IntegerType();
      const bool invalidPart =
          !dataType || !integerType || integerType.getWidth() != 32 ||
          !isa<MaskType>(carryIn.getType()) || !isa<MaskType>(mask.getType()) ||
          !isa<MaskType>(carryType) ||
          !cast<MaskType>(carryIn.getType()).isB32() ||
          !cast<MaskType>(mask.getType()).isB32() ||
          !cast<MaskType>(carryType).isB32() || lhs.getType() != resultType ||
          rhs.getType() != resultType;
      if (invalidPart) {
        return rewriter.notifyMatchFailure(
            op, "vaddcs requires matching 32-bit data and b32 mask parts");
      }
      auto addcs = rewriter.create<VaddcsOp>(
          op.getLoc(), resultType, carryType, lhs, rhs, carryIn, mask);
      results.push_back(addcs.getResult());
      carries.push_back(addcs.getCarry());
    }
    return success();
  }

public:
  LogicalResult matchAndRewrite(VMIVaddcsOp op, OpAdaptor adaptor,
                                OneToNPatternRewriter &rewriter) const override {
    ValueRange lhsParts = adaptor.getLhs();
    ValueRange rhsParts = adaptor.getRhs();
    ValueRange carryInParts = adaptor.getCarryIn();
    ValueRange maskParts = adaptor.getMask();
    return lowerCarryResultParts(
        op, rewriter, *this->getTypeConverter(),
        [&](ArrayRef<Type> resultTypes, ArrayRef<Type> carryTypes,
            SmallVectorImpl<Value> &results,
            SmallVectorImpl<Value> &carries) {
          return lowerParts(op, lhsParts, rhsParts, carryInParts, maskParts,
                            resultTypes, carryTypes, results, carries,
                            rewriter);
        });
  }
};

struct OneToNVMIVmullOpPattern : OneToNOpConversionPattern<VMIVmullOp> {
  using OneToNOpConversionPattern<VMIVmullOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerPart(VMIVmullOp op, Value lhs, Value rhs, Value mask,
                          Type lowType, Type highType,
                          SmallVectorImpl<Value> &lows,
                          SmallVectorImpl<Value> &highs,
                          OneToNPatternRewriter &rewriter) const {
    auto dataType = dyn_cast<VRegType>(lowType);
    auto maskType = dyn_cast<MaskType>(mask.getType());
    const bool invalidShape =
        !dataType || dataType.getElementCount() != 64 || lowType != highType ||
        lhs.getType() != lowType || rhs.getType() != lowType;
    if (invalidShape) {
      return rewriter.notifyMatchFailure(
          op, "vmull requires matching 64-lane physical data part types");
    }
    auto elementType = dyn_cast<IntegerType>(dataType.getElementType());
    const bool invalidElementType =
        !elementType || elementType.getWidth() != 32 ||
        (!elementType.isSignless() && !elementType.isUnsigned());
    if (invalidElementType) {
      return rewriter.notifyMatchFailure(
          op, "vmull requires physical i32 or ui32 data parts");
    }
    if (!maskType || !maskType.isB32()) {
      return rewriter.notifyMatchFailure(
          op, "vmull requires a corresponding b32 mask part");
    }
    auto vmull = rewriter.create<VmullOp>(op.getLoc(), lowType, highType, lhs,
                                          rhs, mask);
    lows.push_back(vmull.getLow());
    highs.push_back(vmull.getHigh());
    return success();
  }

public:
  LogicalResult
  matchAndRewrite(VMIVmullOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange aParts = adaptor.getA();
    ValueRange bParts = adaptor.getB();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybeLowTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    FailureOr<SmallVector<Type>> maybeHighTypes =
        getConvertedResultTypes(op, 1, *this->getTypeConverter());
    const bool conversionFailed =
        failed(maybeLowTypes) || failed(maybeHighTypes);
    if (conversionFailed) {
      return failure();
    }
    SmallVector<Type> lowTypes = std::move(*maybeLowTypes);
    SmallVector<Type> highTypes = std::move(*maybeHighTypes);

    size_t arity = aParts.size();
    const bool invalidArity =
        arity == 0 || bParts.size() != arity || maskParts.size() != arity ||
        lowTypes.size() != arity || highTypes.size() != arity;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "physical vmull arity mismatch across a, b, mask, low, and high");
    }

    SmallVector<Value> lows;
    SmallVector<Value> highs;
    lows.reserve(arity);
    highs.reserve(arity);
    for (size_t index = 0; index < arity; ++index) {
      if (failed(lowerPart(op, aParts[index], bParts[index], maskParts[index],
                           lowTypes[index], highTypes[index], lows, highs,
                           rewriter))) {
        return failure();
      }
    }

    SmallVector<Value> results;
    results.reserve(lows.size() + highs.size());
    results.append(lows);
    results.append(highs);
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }
};

template <typename SourceOp, typename TargetOp>
struct OneToNVMIInterleaveOpPattern : OneToNOpConversionPattern<SourceOp> {
  using OneToNOpConversionPattern<SourceOp>::OneToNOpConversionPattern;

private:
  FailureOr<std::pair<Value, Value>> materializeLaneStrideInterleavePair(
      SourceOp op, OneToNPatternRewriter &rewriter, Value lhs, Value rhs,
      Type lowType, Type highType, int64_t carrierBits) const {
    FailureOr<VRegType> carrierType = getUnsignedCarrierVRegType(
        rewriter.getContext(), static_cast<unsigned>(carrierBits));
    if (failed(carrierType)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported lane-stride interleave carrier width");
    }
    FailureOr<Value> carrierLhs =
        bitcastVReg(op.getLoc(), lhs, *carrierType, rewriter);
    FailureOr<Value> carrierRhs =
        bitcastVReg(op.getLoc(), rhs, *carrierType, rewriter);
    bool failedInputs = failed(carrierLhs) || failed(carrierRhs);
    if (failedInputs) {
      return rewriter.notifyMatchFailure(
          op, "failed to bitcast lane-stride interleave inputs");
    }
    auto interleave = rewriter.create<TargetOp>(
        op.getLoc(), *carrierType, *carrierType, *carrierLhs, *carrierRhs);
    FailureOr<Value> low =
        bitcastVReg(op.getLoc(), interleave.getLow(), lowType, rewriter);
    FailureOr<Value> high =
        bitcastVReg(op.getLoc(), interleave.getHigh(), highType, rewriter);
    bool failedResults = failed(low) || failed(high);
    if (failedResults) {
      return rewriter.notifyMatchFailure(
          op, "failed to bitcast lane-stride interleave results");
    }
    return std::make_pair(*low, *high);
  }

  FailureOr<std::pair<SmallVector<Type>, SmallVector<Type>>>
  getInterleaveResultTypes(
      SourceOp op, ValueRange lhsParts, ValueRange rhsParts,
      ValueRange maskParts, OneToNPatternRewriter &rewriter) const {
    FailureOr<SmallVector<Type>> lowTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    FailureOr<SmallVector<Type>> highTypes =
        getConvertedResultTypes(op, 1, *this->getTypeConverter());
    bool invalidArity =
        failed(lowTypes) || failed(highTypes) || lhsParts.size() != rhsParts.size() ||
        lhsParts.size() != lowTypes->size() || lhsParts.size() != highTypes->size() ||
        (!maskParts.empty() && maskParts.size() != lhsParts.size());
    if (invalidArity) {
      return rewriter.notifyMatchFailure(op, "physical interleave arity mismatch");
    }
    return std::make_pair(std::move(*lowTypes), std::move(*highTypes));
  }

  FailureOr<VMIInterleaveLayoutFact> getInterleaveLayoutFact(
      SourceOp op, VMIVRegType lhsType, VMIVRegType rhsType,
      VMIMaskType maskType, VMIVRegType lowType, VMIVRegType highType,
      OneToNPatternRewriter &rewriter) const {
    VMILayoutSupport supports;
    FailureOr<VMIInterleaveLayoutFact> fact;
    if constexpr (std::is_same_v<SourceOp, VMIVintlvOp>) {
      fact = supports.getVintlvLayoutFactForLayouts(lhsType, rhsType, maskType,
                                                    lowType, highType);
    } else {
      fact = supports.getVdintlvLayoutFactForLayouts(lhsType, rhsType, maskType,
                                                     lowType, highType);
    }
    if (failed(fact)) {
      (void)rewriter.notifyMatchFailure(op, "unsupported interleave layout relation");
    }
    return fact;
  }

  enum class InterleaveLoweringKind { LaneStride, Contiguous, ZeroCopy };
  struct InterleaveLoweringPlan {
    InterleaveLoweringKind kind;
    int64_t inputFactor = 0;
    int64_t outputFactor = 0;
    bool zeroCopyVintlv = false;
  };

  static bool hasLaneStrideInterleaveLayout(
      const VMIInterleaveLayoutFact &fact) {
    return fact.lhsLayout == fact.rhsLayout &&
           fact.lhsLayout == fact.maskLayout &&
           fact.lhsLayout == fact.lowLayout &&
           fact.lhsLayout == fact.highLayout &&
           fact.lhsLayout.isContiguous() &&
           fact.lhsLayout.getLaneStride() > 1;
  }

  static bool hasUnitStrideContiguousInterleaveLayout(
      const VMIInterleaveLayoutFact &fact) {
    auto contiguous = [](VMILayoutAttr layout) {
      return layout && layout.isContiguous() && layout.getLaneStride() == 1;
    };
    return contiguous(fact.lhsLayout) && contiguous(fact.rhsLayout) &&
           contiguous(fact.maskLayout) && contiguous(fact.lowLayout) &&
           contiguous(fact.highLayout);
  }

  std::optional<InterleaveLoweringPlan> getZeroCopyInterleavePlan(
      const VMIInterleaveLayoutFact &fact) const {
    int64_t inputFactor = getElementDeinterleaveFactor(fact.lhsLayout);
    int64_t outputFactor = getElementDeinterleaveFactor(fact.lowLayout);
    bool matchingInputs = fact.rhsLayout == fact.lhsLayout &&
                          fact.maskLayout == fact.lhsLayout;
    bool matchingOutputs = fact.highLayout == fact.lowLayout;
    bool vintlv = std::is_same_v<SourceOp, VMIVintlvOp> && inputFactor > 0 &&
                  matchingInputs && matchingOutputs &&
                  outputFactor == 2 * inputFactor;
    bool vdintlv = std::is_same_v<SourceOp, VMIVdintlvOp> && inputFactor > 0 &&
                   matchingInputs && matchingOutputs &&
                   inputFactor == 2 * outputFactor;
    if (!vintlv && !vdintlv) {
      return std::nullopt;
    }
    return InterleaveLoweringPlan{InterleaveLoweringKind::ZeroCopy,
                                  inputFactor, outputFactor, vintlv};
  }

  FailureOr<InterleaveLoweringPlan> classifyInterleaveLowering(
      SourceOp op, const VMIInterleaveLayoutFact &fact,
      OneToNPatternRewriter &rewriter) const {
    if (hasLaneStrideInterleaveLayout(fact)) {
      return InterleaveLoweringPlan{InterleaveLoweringKind::LaneStride};
    }
    if (hasUnitStrideContiguousInterleaveLayout(fact)) {
      return InterleaveLoweringPlan{InterleaveLoweringKind::Contiguous};
    }
    std::optional<InterleaveLoweringPlan> zeroCopyPlan =
        getZeroCopyInterleavePlan(fact);
    if (!zeroCopyPlan) {
      return rewriter.notifyMatchFailure(op, "unsupported interleave physical layout relation");
    }
    return *zeroCopyPlan;
  }

  LogicalResult lowerInterleaveByLayout(
      SourceOp op, OneToNPatternRewriter &rewriter, ValueRange lhsParts,
      ValueRange rhsParts, ValueRange maskParts, ArrayRef<Type> lowTypes,
      ArrayRef<Type> highTypes, Type elementType,
      const VMIInterleaveLayoutFact &fact) const {
    FailureOr<InterleaveLoweringPlan> plan =
        classifyInterleaveLowering(op, fact, rewriter);
    if (failed(plan)) {
      return failure();
    }
    if (plan->kind == InterleaveLoweringKind::LaneStride) {
      return lowerLaneStrideInterleave(op, rewriter, lhsParts, rhsParts,
                                       lowTypes, highTypes, elementType, fact);
    }
    if (plan->kind == InterleaveLoweringKind::Contiguous) {
      return lowerContiguous(op, rewriter, lhsParts, rhsParts, maskParts,
                             lowTypes, highTypes);
    }
    FailureOr<SmallVector<Value>> results = materializeZeroCopyResults(
        op, lhsParts, rhsParts, lowTypes, highTypes, plan->inputFactor,
        plan->outputFactor, plan->zeroCopyVintlv, rewriter);
    if (failed(results)) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                     *this->getTypeConverter());
    return success();
  }

  LogicalResult lowerLaneStrideInterleave(
      SourceOp op, OneToNPatternRewriter &rewriter, ValueRange lhsParts,
      ValueRange rhsParts, TypeRange lowTypes, TypeRange highTypes,
      Type elementType, const VMIInterleaveLayoutFact &fact) const {
    bool invalidArity = lhsParts.size() != 1 || rhsParts.size() != 1 ||
                        lowTypes.size() != 1 || highTypes.size() != 1;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "lane-stride interleave expects one physical carrier part");
    }
    unsigned elementBits =
        pto::getPTOStorageElemBitWidth(elementType);
    int64_t laneStride = fact.lhsLayout.getLaneStride();
    int64_t carrierBits = static_cast<int64_t>(elementBits) * laneStride;
    bool invalidCarrier = elementBits == 0 || laneStride <= 1 ||
                          carrierBits <= 0 || carrierBits > 32;
    if (invalidCarrier) {
      return rewriter.notifyMatchFailure(
          op, "invalid lane-stride interleave carrier width");
    }
    FailureOr<std::pair<Value, Value>> pair =
        materializeLaneStrideInterleavePair(
            op, rewriter, lhsParts.front(), rhsParts.front(), lowTypes.front(),
            highTypes.front(), carrierBits);
    if (failed(pair)) {
      return failure();
    }
    SmallVector<Value, 2> results = {pair->first, pair->second};
    return lowerBinaryPhysicalResults(op, results, rewriter,
                                      *this->getTypeConverter());
  }

  void appendVintlvZeroCopyResults(SmallVectorImpl<Value> &results,
                                   ValueRange lhsParts, ValueRange rhsParts,
                                   int64_t inputFactor) const {
    int64_t safeInputFactor = inputFactor > 0 ? inputFactor : 1;
    size_t groupChunks = lhsParts.size() / safeInputFactor;
    size_t halfGroupChunks = groupChunks / 2;
    for (int64_t group = 0; group < inputFactor; ++group) {
      size_t offset = group * groupChunks;
      llvm::append_range(results, lhsParts.slice(offset, halfGroupChunks));
      llvm::append_range(results, rhsParts.slice(offset, halfGroupChunks));
    }
    for (int64_t group = 0; group < inputFactor; ++group) {
      size_t offset = group * groupChunks + halfGroupChunks;
      llvm::append_range(results, lhsParts.slice(offset, halfGroupChunks));
      llvm::append_range(results, rhsParts.slice(offset, halfGroupChunks));
    }
  }

  void appendVdintlvZeroCopyResults(SmallVectorImpl<Value> &results,
                                    ValueRange lhsParts, ValueRange rhsParts,
                                    int64_t inputFactor,
                                    int64_t outputFactor) const {
    int64_t safeInputFactor = inputFactor > 0 ? inputFactor : 1;
    size_t groupChunks = lhsParts.size() / safeInputFactor;
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

  LogicalResult validateZeroCopyResultParts(
      SourceOp op, ArrayRef<Value> results, TypeRange lowTypes,
      TypeRange highTypes, OneToNPatternRewriter &rewriter) const {
    SmallVector<Type> resultTypes;
    resultTypes.reserve(lowTypes.size() + highTypes.size());
    llvm::append_range(resultTypes, lowTypes);
    llvm::append_range(resultTypes, highTypes);
    bool resultArityMismatch = results.size() != resultTypes.size();
    if (resultArityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "zero-copy interleave result arity mismatch");
    }
    for (auto [value, resultType] : llvm::zip_equal(results, resultTypes)) {
      bool resultTypeMismatch = value.getType() != resultType;
      if (resultTypeMismatch) {
        return rewriter.notifyMatchFailure(
            op, "zero-copy interleave part type mismatch");
      }
    }
    return success();
  }

  FailureOr<SmallVector<Value>> materializeZeroCopyResults(
      SourceOp op, ValueRange lhsParts, ValueRange rhsParts,
      TypeRange lowTypes, TypeRange highTypes, int64_t inputFactor,
      int64_t outputFactor, bool zeroCopyVintlv,
      OneToNPatternRewriter &rewriter) const {
    if (inputFactor <= 0) {
      return rewriter.notifyMatchFailure(
          op, "zero-copy interleave requires positive input factor");
    }
    int64_t safeInputFactor = inputFactor;
    SmallVector<Value> results;
    results.reserve(lhsParts.size() + rhsParts.size());
    if (zeroCopyVintlv) {
      bool invalidGroupCount =
          lhsParts.empty() || lhsParts.size() % (2 * safeInputFactor) != 0;
      if (invalidGroupCount) {
        return rewriter.notifyMatchFailure(
            op, "zero-copy vintlv expects input groups with even chunk count");
      }
      appendVintlvZeroCopyResults(results, lhsParts, rhsParts, inputFactor);
    } else {
      bool invalidGroupCount =
          lhsParts.empty() || lhsParts.size() % safeInputFactor != 0;
      if (invalidGroupCount) {
        return rewriter.notifyMatchFailure(
            op, "zero-copy vdintlv expects complete input layout groups");
      }
      appendVdintlvZeroCopyResults(results, lhsParts, rhsParts, inputFactor,
                                   outputFactor);
    }

    if (failed(validateZeroCopyResultParts(op, results, lowTypes, highTypes,
                                           rewriter))) {
      return failure();
    }
    return results;
  }

  LogicalResult lowerContiguous(
      SourceOp op, OneToNPatternRewriter &rewriter, ValueRange lhsParts,
      ValueRange rhsParts, ValueRange maskParts, ArrayRef<Type> lowTypes,
      ArrayRef<Type> highTypes) const {
    bool singleChunk = lhsParts.size() == 1 && rhsParts.size() == 1 &&
                       lowTypes.size() == 1 && highTypes.size() == 1;
    if (!singleChunk) {
      return rewriter.notifyMatchFailure(
          op, "single-chunk interleave expects one physical part");
    }
    bool invalidMaskPart =
        !maskParts.empty() && !isa<MaskType>(maskParts.front().getType());
    if (invalidMaskPart) {
      return rewriter.notifyMatchFailure(
          op, "single-chunk interleave mask part type mismatch");
    }
    bool invalidTypes =
        !isa<VRegType>(lowTypes.front()) || !isa<VRegType>(highTypes.front()) ||
        lhsParts.front().getType() != lowTypes.front() ||
        rhsParts.front().getType() != lowTypes.front() ||
        highTypes.front() != lowTypes.front();
    if (invalidTypes) {
      return rewriter.notifyMatchFailure(
          op, "single-chunk interleave part type mismatch");
    }
    auto interleave = rewriter.create<TargetOp>(
        op.getLoc(), lowTypes.front(), highTypes.front(), lhsParts.front(),
        rhsParts.front());
    SmallVector<Value, 2> results = {interleave.getLow(), interleave.getHigh()};
    return lowerBinaryPhysicalResults(op, results, rewriter,
                                      *this->getTypeConverter());
  }

public:

  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OneToNOpConversionPattern<SourceOp>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const override {
    ValueRange lhsParts = adaptor.getLhs();
    ValueRange rhsParts = adaptor.getRhs();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<std::pair<SmallVector<Type>, SmallVector<Type>>> resultTypes =
        getInterleaveResultTypes(op, lhsParts, rhsParts, maskParts, rewriter);
    if (failed(resultTypes)) {
      return failure();
    }
    SmallVector<Type> lowTypes = std::move(resultTypes->first);
    SmallVector<Type> highTypes = std::move(resultTypes->second);

    auto lhsType = cast<VMIVRegType>(op.getLhs().getType());
    auto rhsType = cast<VMIVRegType>(op.getRhs().getType());
    auto maskType = cast<VMIMaskType>(op.getMask().getType());
    auto lowType = cast<VMIVRegType>(op.getLow().getType());
    auto highType = cast<VMIVRegType>(op.getHigh().getType());
    FailureOr<VMIInterleaveLayoutFact> fact = getInterleaveLayoutFact(
        op, lhsType, rhsType, maskType, lowType, highType, rewriter);
    if (failed(fact)) {
      return failure();
    }

    return lowerInterleaveByLayout(
        op, rewriter, lhsParts, rhsParts, maskParts, lowTypes, highTypes,
        lhsType.getElementType(), *fact);
  }
};

struct OneToNVMIFmaOpPattern : OneToNOpConversionPattern<VMIFmaOp> {
  using OneToNOpConversionPattern<VMIFmaOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> lowerPart(VMIFmaOp op, Value lhs, Value rhs, Value acc,
                             Type resultType,
                             OneToNPatternRewriter &rewriter) const {
    auto vregType = dyn_cast<VRegType>(resultType);
    const bool invalidPart =
        !vregType || lhs.getType() != resultType || rhs.getType() != resultType ||
        acc.getType() != resultType;
    if (invalidPart) {
      (void)rewriter.notifyMatchFailure(
          op, "fma requires matching physical vreg parts");
      return failure();
    }
    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
    if (failed(mask)) {
      (void)rewriter.notifyMatchFailure(op,
                                        "unsupported element type for fma");
      return failure();
    }
    return rewriter
        .create<VmulaOp>(op.getLoc(), resultType, acc, lhs, rhs, *mask)
        .getResult();
  }

public:
  LogicalResult
  matchAndRewrite(VMIFmaOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange lhsParts = adaptor.getLhs();
    ValueRange rhsParts = adaptor.getRhs();
    ValueRange accParts = adaptor.getAcc();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    const bool invalidArity =
        lhsParts.size() != rhsParts.size() ||
        lhsParts.size() != accParts.size() ||
        lhsParts.size() != resultTypes.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(op, "fma physical arity mismatch");
    }

    return lowerPointwisePhysicalParts(
        op, resultTypes, "fma physical arity mismatch", rewriter,
        [&](int64_t index, Type resultType) -> FailureOr<Value> {
          return lowerPart(op, lhsParts[index], rhsParts[index], accParts[index],
                           resultType, rewriter);
        },
        *this->getTypeConverter());
  }
};

struct OneToNVMIVmulaOpPattern : OneToNOpConversionPattern<VMIVmulaOp> {
  using OneToNOpConversionPattern<VMIVmulaOp>::OneToNOpConversionPattern;
  LogicalResult matchAndRewrite(VMIVmulaOp op, OpAdaptor adaptor,
                                OneToNPatternRewriter &rewriter) const override {
    ValueRange acc = adaptor.getAcc();
    ValueRange lhs = adaptor.getLhs();
    ValueRange rhs = adaptor.getRhs();
    ArrayRef<ValueRange> maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> converted =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(converted) || acc.size() != lhs.size() ||
        acc.size() != rhs.size() || acc.size() != converted->size())
      return rewriter.notifyMatchFailure(op, "vmula physical arity mismatch");
    if (maskParts.size() > 1 ||
        (!maskParts.empty() && maskParts.front().size() != acc.size()))
      return rewriter.notifyMatchFailure(op, "vmula mask physical arity mismatch");
    SmallVector<Value> results;
    for (unsigned i = 0; i < acc.size(); ++i) {
      auto type = dyn_cast<VRegType>((*converted)[i]);
      if (!type || acc[i].getType() != (*converted)[i] ||
          lhs[i].getType() != (*converted)[i] ||
          rhs[i].getType() != (*converted)[i])
        return rewriter.notifyMatchFailure(op, "vmula requires matching physical vreg parts");
      Value mask;
      if (maskParts.empty()) {
        FailureOr<Value> allTrue = createAllTrueMaskForVReg(op.getLoc(), type, rewriter);
        if (failed(allTrue))
          return rewriter.notifyMatchFailure(op,
                                             "unsupported element type for vmula");
        mask = *allTrue;
      } else {
        mask = maskParts.front()[i];
      }
      Value result = rewriter
                         .create<VmulaOp>(op.getLoc(), (*converted)[i], acc[i],
                                          lhs[i], rhs[i], mask)
                         .getResult();
      // VPTO vmula preserves the accumulator on inactive lanes.  VMI's
      // explicit zero mode instead requires inactive lanes to be cleared, so
      // materialize that semantic difference after the fused operation.
      if (op.getPmode().has_value() && *op.getPmode() == "zero") {
        FailureOr<Value> zero = createZeroVector(op.getLoc(), type, rewriter);
        if (failed(zero))
          return rewriter.notifyMatchFailure(
              op, "failed to materialize vmula zero-mode value");
        result = rewriter
                     .create<VselOp>(op.getLoc(), (*converted)[i], result, *zero,
                                     mask)
                     .getResult();
      }
      results.push_back(result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIVexpdifOpPattern : OneToNOpConversionPattern<VMIVexpdifOp> {
  using OneToNOpConversionPattern<VMIVexpdifOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> lowerF32Part(VMIVexpdifOp op, Value x, Value max,
                                Value mask, Type resultType,
                                OneToNPatternRewriter &rewriter) const {
    auto vregType = dyn_cast<VRegType>(resultType);
    auto maskType = dyn_cast<MaskType>(mask.getType());
    const bool invalidPart =
        !vregType || !maskType || !vregType.getElementType().isF32() ||
        x.getType() != resultType || max.getType() != resultType ||
        maskType.getGranularity() != "b32";
    if (invalidPart) {
      (void)rewriter.notifyMatchFailure(
          op, "f32 vexpdif requires matching f32 parts and b32 masks");
      return failure();
    }
    return rewriter
        .create<VexpdifOp>(op.getLoc(), resultType, x, max, mask,
                           rewriter.getStringAttr("ODD"))
        .getResult();
  }

  FailureOr<Value> lowerF16Part(VMIVexpdifOp op, Value x, Value max,
                                Value mask, Type resultType, StringRef part,
                                OneToNPatternRewriter &rewriter) const {
    auto xType = dyn_cast<VRegType>(x.getType());
    auto resultVRegType = dyn_cast<VRegType>(resultType);
    auto maskType = dyn_cast<MaskType>(mask.getType());
    const bool invalidPart =
        !xType || !resultVRegType || !maskType ||
        !xType.getElementType().isF16() ||
        !resultVRegType.getElementType().isF32() ||
        max.getType() != x.getType() || maskType.getGranularity() != "b16";
    if (invalidPart) {
      (void)rewriter.notifyMatchFailure(
          op, "f16 vexpdif requires matching f16 parts and b16 masks");
      return failure();
    }
    return rewriter
        .create<VexpdifOp>(op.getLoc(), resultType, x, max, mask,
                           rewriter.getStringAttr(part))
        .getResult();
  }

  LogicalResult lowerF32(VMIVexpdifOp op, ValueRange xParts,
                         ValueRange maxParts, ValueRange maskParts,
                         ArrayRef<Type> resultTypes,
                         OneToNPatternRewriter &rewriter) const {
    const bool invalidArity = xParts.size() != resultTypes.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "f32 vexpdif requires one result per source part");
    }

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [x, max, mask, resultType] :
         llvm::zip_equal(xParts, maxParts, maskParts, resultTypes)) {
      FailureOr<Value> result =
          lowerF32Part(op, x, max, mask, resultType, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    return lowerBinaryPhysicalResults(op, results, rewriter,
                                      *this->getTypeConverter());
  }

  LogicalResult lowerF16(VMIVexpdifOp op, ValueRange xParts,
                         ValueRange maxParts, ValueRange maskParts,
                         ArrayRef<Type> resultTypes,
                         OneToNPatternRewriter &rewriter) const {
    const bool invalidArity = resultTypes.size() != 2 * xParts.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "f16 vexpdif requires EVEN/ODD f32 result parts");
    }

    static constexpr StringRef kParts[] = {"EVEN", "ODD"};
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [partIndex, part] : llvm::enumerate(kParts)) {
      for (auto [chunkIndex, x] : llvm::enumerate(xParts)) {
        Value max = maxParts[chunkIndex];
        Value mask = maskParts[chunkIndex];
        Type resultType = resultTypes[partIndex * xParts.size() + chunkIndex];
        FailureOr<Value> result =
            lowerF16Part(op, x, max, mask, resultType, part, rewriter);
        if (failed(result)) {
          return failure();
        }
        results.push_back(*result);
      }
    }
    return replacePhysicalResults(rewriter, op, results,
                                  *this->getTypeConverter());
  }

  LogicalResult lowerBySourceElementType(
      VMIVexpdifOp op, ValueRange xParts, ValueRange maxParts,
      ValueRange maskParts, ArrayRef<Type> resultTypes,
      Type sourceElementType, OneToNPatternRewriter &rewriter) const {
    if (sourceElementType.isF32()) {
      return lowerF32(op, xParts, maxParts, maskParts, resultTypes, rewriter);
    }
    if (!sourceElementType.isF16()) {
      return rewriter.notifyMatchFailure(
          op, "f16 vexpdif requires EVEN/ODD f32 result parts");
    }
    return lowerF16(op, xParts, maxParts, maskParts, resultTypes, rewriter);
  }

public:
  LogicalResult
  matchAndRewrite(VMIVexpdifOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    const bool requiresPassthru =
        op.getPmode().has_value() && *op.getPmode() == "merge";
    if (requiresPassthru) {
      return rewriter.notifyMatchFailure(
          op, "merge predicate mode requires an explicit passthru lowering");
    }

    ValueRange xParts = adaptor.getX();
    ValueRange maxParts = adaptor.getMax();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybeResultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybeResultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybeResultTypes);
    const bool invalidInputArity =
        xParts.size() != maxParts.size() || xParts.size() != maskParts.size();
    if (invalidInputArity) {
      return rewriter.notifyMatchFailure(op, "vexpdif physical arity mismatch");
    }

    auto sourceVMIType = cast<VMIVRegType>(op.getX().getType());
    return lowerBySourceElementType(
        op, xParts, maxParts, maskParts, resultTypes,
        sourceVMIType.getElementType(), rewriter);
  }
};

template <typename SourceOp, typename TargetOp, bool IsMaskResult = false>
struct OneToNVMIUnaryOpPattern : OneToNOpConversionPattern<SourceOp> {
  using OneToNOpConversionPattern<SourceOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> lowerPart(
      SourceOp op, Value source, Type resultType,
      OneToNPatternRewriter &rewriter) const {
    if constexpr (IsMaskResult) {
      auto maskType = dyn_cast<MaskType>(resultType);
      const bool invalidPart = !maskType || source.getType() != resultType;
      if (invalidPart) {
        return rewriter.notifyMatchFailure(
            op, "physical mask unary part type mismatch");
      }
      FailureOr<Value> seedMask =
          createAllTrueMask(op.getLoc(), maskType, rewriter);
      if (failed(seedMask)) {
        return rewriter.notifyMatchFailure(
            op, "unsupported mask type for all-true mask unary seed");
      }
      return rewriter
          .create<TargetOp>(op.getLoc(), resultType, source, *seedMask)
          .getResult();
    }
    auto vregType = dyn_cast<VRegType>(resultType);
    const bool invalidPart = !vregType || source.getType() != resultType;
    if (invalidPart) {
      return rewriter.notifyMatchFailure(
          op, "physical unary part type mismatch");
    }
    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for all-true unary mask");
    }
    return rewriter
        .create<TargetOp>(op.getLoc(), resultType, source, *mask)
        .getResult();
  }

  LogicalResult lowerParts(SourceOp op, ValueRange sourceParts,
                           ArrayRef<Type> resultTypes,
                           OneToNPatternRewriter &rewriter) const {
    const bool invalidArity = sourceParts.size() != resultTypes.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, IsMaskResult ? "physical mask unary arity mismatch"
                           : "physical unary arity mismatch");
    }
    return lowerPointwisePhysicalParts(
        op, resultTypes,
        IsMaskResult ? "physical mask unary arity mismatch"
                     : "physical unary arity mismatch",
        rewriter,
        [&](int64_t index, Type resultType) -> FailureOr<Value> {
          return lowerPart(op, sourceParts[index], resultType, rewriter);
        },
        *this->getTypeConverter());
  }

public:
  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OneToNOpConversionPattern<SourceOp>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const override {
    return lowerWithConvertedResultTypes(
        op, 0, *this->getTypeConverter(), [&](ArrayRef<Type> resultTypes) {
          return lowerParts(op, adaptor.getSource(), resultTypes, rewriter);
        });
  }
};


template <typename SourceOp>
struct OneToNVMICmpOpPattern : OneToNOpConversionPattern<SourceOp> {
  using OneToNOpConversionPattern<SourceOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> lowerPart(
      SourceOp op, Value lhs, Value rhs, Type resultType,
      const VPTOCmpMode &cmpMode,
      OneToNPatternRewriter &rewriter) const {
    auto maskType = dyn_cast<MaskType>(resultType);
    auto lhsType = dyn_cast<VRegType>(lhs.getType());
    const bool invalidPart =
        !maskType || lhs.getType() != rhs.getType() || !lhsType;
    if (invalidPart) {
      (void)rewriter.notifyMatchFailure(op,
                                        "physical cmp part type mismatch");
      return failure();
    }
    FailureOr<Value> seedMask =
        createAllTrueMask(op.getLoc(), maskType, rewriter);
    if (failed(seedMask)) {
      (void)rewriter.notifyMatchFailure(
          op, "unsupported mask type for all-true cmp seed");
      return failure();
    }
    if (cmpMode.signedness) {
      FailureOr<VRegType> carrierType =
          getSignednessCarrierVRegType(lhsType, *cmpMode.signedness);
      if (failed(carrierType)) {
        (void)rewriter.notifyMatchFailure(
            op, "unsupported integer compare signedness carrier");
        return failure();
      }
      FailureOr<Value> carrierLhs =
          bitcastVReg(op.getLoc(), lhs, *carrierType, rewriter);
      FailureOr<Value> carrierRhs =
          bitcastVReg(op.getLoc(), rhs, *carrierType, rewriter);
      const bool failedCarriers = failed(carrierLhs) || failed(carrierRhs);
      if (failedCarriers) {
        (void)rewriter.notifyMatchFailure(
            op, "failed to materialize integer compare signedness carrier");
        return failure();
      }
      lhs = *carrierLhs;
      rhs = *carrierRhs;
    }
    return rewriter
        .create<VcmpOp>(op.getLoc(), resultType, lhs, rhs, *seedMask,
                        rewriter.getStringAttr(cmpMode.mode))
        .getResult();
  }

public:
  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OneToNOpConversionPattern<SourceOp>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const override {
    std::optional<VPTOCmpMode> cmpMode =
        getVPTOCmpMode<SourceOp>(op.getPredicate());
    if (!cmpMode) {
      return op.emitOpError()
             << kVMIDiagUnsupportedPrefix << "compare predicate "
             << op.getPredicate()
             << " cannot be lowered to pto.vcmp; supported predicates are "
             << getSupportedComparePredicateMessage<SourceOp>();
    }

    ValueRange lhsParts = adaptor.getLhs();
    ValueRange rhsParts = adaptor.getRhs();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    const bool invalidArity = lhsParts.size() != rhsParts.size() ||
                              lhsParts.size() != resultTypes.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(op, "physical cmp arity mismatch");
    }

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [lhs, rhs, resultType] :
         llvm::zip_equal(lhsParts, rhsParts, resultTypes)) {
      FailureOr<Value> result =
          lowerPart(op, lhs, rhs, resultType, *cmpMode, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMISelectOpPattern : OneToNOpConversionPattern<VMISelectOp> {
  using OneToNOpConversionPattern<VMISelectOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> lowerPart(VMISelectOp op, Value mask, Value trueValue,
                             Value falseValue, Type resultType,
                             OneToNPatternRewriter &rewriter) const {
    const bool invalidPart =
        !isa<MaskType>(mask.getType()) || trueValue.getType() != resultType ||
        falseValue.getType() != resultType || !isa<VRegType>(resultType);
    if (invalidPart) {
      (void)rewriter.notifyMatchFailure(
          op, "physical select part type mismatch");
      return failure();
    }
    return rewriter
        .create<VselOp>(op.getLoc(), resultType, trueValue, falseValue, mask)
        .getResult();
  }

public:
  LogicalResult
  matchAndRewrite(VMISelectOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange maskParts = adaptor.getMask();
    ValueRange trueParts = adaptor.getTrueValue();
    ValueRange falseParts = adaptor.getFalseValue();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    const bool invalidArity =
        maskParts.size() != trueParts.size() ||
        trueParts.size() != falseParts.size() ||
        trueParts.size() != resultTypes.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(op, "physical select arity mismatch");
    }

    return lowerPointwisePhysicalParts(
        op, resultTypes, "physical select arity mismatch", rewriter,
        [&](int64_t index, Type resultType) -> FailureOr<Value> {
          return lowerPart(op, maskParts[index], trueParts[index],
                           falseParts[index], resultType, rewriter);
        },
        *this->getTypeConverter());
  }
};

struct OneToNVMIVselrOpPattern : OneToNOpConversionPattern<VMIVselrOp> {
  using OneToNOpConversionPattern<VMIVselrOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> lowerPart(VMIVselrOp op, Value source, Value index,
                             Type resultType,
                             OneToNPatternRewriter &rewriter) const {
    auto sourceType = dyn_cast<VRegType>(source.getType());
    auto indexType = dyn_cast<VRegType>(index.getType());
    auto resultVRegType = dyn_cast<VRegType>(resultType);
    const bool invalidPart =
        !sourceType || !indexType || !resultVRegType ||
        sourceType != resultVRegType ||
        sourceType.getElementCount() != indexType.getElementCount() ||
        pto::getPTOStorageElemBitWidth(sourceType.getElementType()) !=
            pto::getPTOStorageElemBitWidth(indexType.getElementType());
    if (invalidPart) {
      (void)rewriter.notifyMatchFailure(
          op, "vselr physical source/index/result type mismatch");
      return failure();
    }
    return rewriter
        .create<VselrOp>(op.getLoc(), resultVRegType, source, index)
        .getResult();
  }

public:
  LogicalResult
  matchAndRewrite(VMIVselrOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    ValueRange indexParts = adaptor.getIndex();
    FailureOr<SmallVector<Type>> maybeResultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybeResultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybeResultTypes);
    const bool invalidArity = sourceParts.size() != 1 ||
                              indexParts.size() != 1 || resultTypes.size() != 1;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "vselr supports only one physical source/index/result part");
    }
    FailureOr<Value> result = lowerPart(op, sourceParts.front(),
                                        indexParts.front(), resultTypes.front(),
                                        rewriter);
    if (failed(result)) {
      return failure();
    }
    return replaceSinglePhysicalResult(rewriter, op, *result,
                                       *this->getTypeConverter());
  }
};

struct OneToNVMIActivePrefixIndexOpPattern
    : OneToNOpConversionPattern<VMIActivePrefixIndexOp> {
  using OneToNOpConversionPattern<
      VMIActivePrefixIndexOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> lowerPart(VMIActivePrefixIndexOp op, Value mask,
                             Type resultType,
                             OneToNPatternRewriter &rewriter) const {
    auto vregType = dyn_cast<VRegType>(resultType);
    auto maskType = dyn_cast<MaskType>(mask.getType());
    const bool invalidPartTypes = !vregType || !maskType;
    if (invalidPartTypes) {
      (void)rewriter.notifyMatchFailure(
          op, "active_prefix_index requires physical vreg/mask parts");
      return failure();
    }
    auto intType = dyn_cast<IntegerType>(vregType.getElementType());
    const bool invalidElementType = !intType || !intType.isSignless();
    if (invalidElementType) {
      (void)rewriter.notifyMatchFailure(
          op, "active_prefix_index requires signless integer result part");
      return failure();
    }
    FailureOr<Value> seedMask =
        createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
    if (failed(seedMask)) {
      (void)rewriter.notifyMatchFailure(
          op, "unsupported element type for active_prefix_index seed mask");
      return failure();
    }
    Value zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0,
                                                       intType.getWidth());
    Value carrier =
        rewriter
            .create<VdupOp>(op.getLoc(), resultType, zero, *seedMask,
                            /*position=*/nullptr)
            .getResult();
    return rewriter
        .create<VusqzOp>(op.getLoc(), resultType, carrier, mask)
        .getResult();
  }

public:
  LogicalResult
  matchAndRewrite(VMIActivePrefixIndexOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybeResultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybeResultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybeResultTypes);
    const bool invalidArity = maskParts.size() != 1 || resultTypes.size() != 1;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "active_prefix_index supports only one physical part");
    }
    FailureOr<Value> result =
        lowerPart(op, maskParts.front(), resultTypes.front(), rewriter);
    if (failed(result)) {
      return failure();
    }
    return replaceSinglePhysicalResult(rewriter, op, *result,
                                       *this->getTypeConverter());
  }
};

struct OneToNVMICompressOpPattern : OneToNOpConversionPattern<VMICompressOp> {
  using OneToNOpConversionPattern<VMICompressOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> lowerPart(VMICompressOp op, Value source, Value mask,
                             Type resultType,
                             OneToNPatternRewriter &rewriter) const {
    auto resultVRegType = dyn_cast<VRegType>(resultType);
    const bool invalidPart =
        !resultVRegType || source.getType() != resultType ||
        !isa<MaskType>(mask.getType());
    if (invalidPart) {
      (void)rewriter.notifyMatchFailure(
          op, "compress requires physical source/mask/result parts");
      return failure();
    }
    return rewriter
        .create<VsqzOp>(op.getLoc(), resultVRegType, source, mask)
        .getResult();
  }

public:
  LogicalResult
  matchAndRewrite(VMICompressOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    const bool invalidArity = sourceParts.size() != 1 ||
                              maskParts.size() != 1 || resultTypes.size() != 1;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "compress supports only one physical part");
    }

    FailureOr<Value> result = lowerPart(op, sourceParts.front(),
                                        maskParts.front(), resultTypes.front(),
                                        rewriter);
    if (failed(result)) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, SmallVector<Value>{*result},
                                     *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMICompressStoreOpPattern
    : OneToNOpConversionPattern<VMICompressStoreOp> {
  using OneToNOpConversionPattern<
      VMICompressStoreOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerStore(VMICompressStoreOp op, Value destination,
                           Value offset, Value value, Value mask,
                           OneToNPatternRewriter &rewriter) const {
    auto valueType = dyn_cast<VRegType>(value.getType());
    auto destinationType = dyn_cast<PtrType>(destination.getType());
    const bool invalidTypes =
        !valueType || !isa<MaskType>(mask.getType()) || !destinationType;
    if (invalidTypes) {
      return rewriter.notifyMatchFailure(
          op, "compress_store requires physical value/mask and ptr "
              "destination");
    }
    Value storeBase =
        rewriter
            .create<AddPtrOp>(op.getLoc(), destination.getType(), destination,
                              offset)
            .getResult();
    Value squeezed =
        rewriter.create<VsqzOp>(op.getLoc(), valueType, value, mask).getResult();
    auto align = rewriter.create<InitAlignOp>(
        op.getLoc(), AlignType::get(rewriter.getContext()));
    auto store = rewriter.create<VsturOp>(
        op.getLoc(), align.getResult().getType(), align.getResult(), squeezed,
        storeBase, rewriter.getStringAttr("POST_UPDATE"));
    rewriter.create<VstarOp>(op.getLoc(), store.getAlignOut(), storeBase);
    rewriter.eraseOp(op);
    return success();
  }

public:
  LogicalResult
  matchAndRewrite(VMICompressStoreOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "compress_store destination must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(),
        "compress_store offset must convert to one value", rewriter);
    const bool failedAddress = failed(destination) || failed(offset);
    if (failedAddress) {
      return failure();
    }

    ValueRange valueParts = adaptor.getValue();
    ValueRange maskParts = adaptor.getMask();
    const bool invalidArity = valueParts.size() != 1 || maskParts.size() != 1;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "compress_store supports only one physical part");
    }

    return lowerStore(op, *destination, *offset, valueParts.front(),
                      maskParts.front(), rewriter);
  }
};

struct ReduceAddPhysicalPlan {
  VRegType resultType;
  MaskType maskType;
};

template <typename OpTy>
static FailureOr<ReduceAddPhysicalPlan> buildReduceAddPhysicalPlan(
    OpTy op, ValueRange sourceParts, ValueRange maskParts,
    TypeRange resultTypes, OneToNPatternRewriter &rewriter,
    StringRef diagnostic) {
  bool invalidArity = sourceParts.empty() || sourceParts.size() != maskParts.size() ||
                      resultTypes.size() != 1;
  if (invalidArity) {
    return rewriter.notifyMatchFailure(
        op, Twine(diagnostic) +
                " requires matching source/mask chunks and one result chunk");
  }
  auto resultType = dyn_cast<VRegType>(resultTypes.front());
  auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
  if (!resultType || !maskType) {
    return rewriter.notifyMatchFailure(
        op, Twine(diagnostic) +
                " requires matching physical source/result vregs and one mask");
  }
  for (Value sourcePart : sourceParts) {
    bool sourceTypeMismatch = sourcePart.getType() != resultType;
    if (sourceTypeMismatch) {
      return rewriter.notifyMatchFailure(
          op, Twine(diagnostic) +
                  " requires every source chunk to match result vreg type");
    }
  }
  for (Value maskPart : maskParts) {
    bool maskTypeMismatch = maskPart.getType() != maskType;
    if (maskTypeMismatch) {
      return rewriter.notifyMatchFailure(
          op, Twine(diagnostic) +
                  " requires every mask chunk to have the same predicate type");
    }
  }
  return ReduceAddPhysicalPlan{resultType, maskType};
}

template <typename ReduceOp>
static LogicalResult lowerReduceAddParts(
    ReduceOp op, ValueRange sourceParts, ValueRange maskParts,
    VRegType resultType, MaskType maskType, StringRef firstLaneDiagnostic,
    OneToNPatternRewriter &rewriter, TypeConverter &typeConverter) {
  FailureOr<Value> combined = combineEquivalentMaskedParts<VaddOp>(
      op.getLoc(), sourceParts, maskParts, resultType, rewriter);
  if (succeeded(combined)) {
    Value reduced = rewriter
                        .create<VcaddOp>(op.getLoc(), resultType, *combined,
                                         maskParts.front())
                        .getResult();
    replaceOpWithFlatConvertedValues(rewriter, op, SmallVector<Value>{reduced},
                                     typeConverter);
    return success();
  }

  Value accumulator = rewriter
                          .create<VcaddOp>(op.getLoc(), resultType,
                                           sourceParts.front(),
                                           maskParts.front())
                          .getResult();
  const bool singlePart = sourceParts.size() == 1;
  if (singlePart) {
    replaceOpWithFlatConvertedValues(rewriter, op,
                                     SmallVector<Value>{accumulator},
                                     typeConverter);
    return success();
  }
  FailureOr<Value> firstLaneMask =
      createPrefixMask(op.getLoc(), maskType, "PAT_VL1", rewriter);
  if (failed(firstLaneMask)) {
    return rewriter.notifyMatchFailure(op, firstLaneDiagnostic);
  }
  for (size_t part = 1; part < sourceParts.size(); ++part) {
    Value reduced = rewriter
                        .create<VcaddOp>(op.getLoc(), resultType,
                                         sourceParts[part], maskParts[part])
                        .getResult();
    accumulator = rewriter
                      .create<VaddOp>(op.getLoc(), resultType, reduced,
                                      accumulator, *firstLaneMask)
                      .getResult();
  }
  replaceOpWithFlatConvertedValues(rewriter, op,
                                   SmallVector<Value>{accumulator},
                                   typeConverter);
  return success();
}

struct OneToNVMIReduceAddIOpPattern
    : OneToNOpConversionPattern<VMIReduceAddIOp> {
  using OneToNOpConversionPattern<VMIReduceAddIOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIReduceAddIOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypesOrFailure(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    FailureOr<ReduceAddPhysicalPlan> plan = buildReduceAddPhysicalPlan(
        op, sourceParts, maskParts, resultTypes, rewriter, "reduce_addi");
    if (failed(plan)) {
      return failure();
    }
    return lowerReduceAddParts(
        op, sourceParts, maskParts, plan->resultType, plan->maskType,
        "failed to create reduce_addi first-lane mask", rewriter,
        *this->getTypeConverter());
  }
};

struct OneToNVMIReduceAddFOpPattern
    : OneToNOpConversionPattern<VMIReduceAddFOp> {
  using OneToNOpConversionPattern<VMIReduceAddFOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIReduceAddFOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    FailureOr<ReduceAddPhysicalPlan> plan = buildReduceAddPhysicalPlan(
        op, sourceParts, maskParts, resultTypes, rewriter, "reduce_addf");
    if (failed(plan)) {
      return failure();
    }
    return lowerReduceAddParts(
        op, sourceParts, maskParts, plan->resultType, plan->maskType,
        "failed to create reduce_addf first-lane mask", rewriter,
        *this->getTypeConverter());
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
  if (failed(fact)) {
    return failure();
  }

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
    bool deinterleavedSource =
        fact->sourceLayout && fact->sourceLayout.isDeinterleaved() &&
        fact->sourceLayout.getFactor() == 2;
    if (deinterleavedSource) {
      return GroupReduceLoweringPlan::FullDeinterleaved2VcaddRows;
    }
    return GroupReduceLoweringPlan::ContiguousVcaddRows;
  }
  llvm_unreachable("unknown group block class");
}

template <typename OpTy, typename GroupReduceOpTy, typename RowReduceOpTy,
          typename CombineOpTy>
struct OneToNVMIGroupReduceOpPattern : OneToNOpConversionPattern<OpTy> {
  using OneToNOpConversionPattern<OpTy>::OneToNOpConversionPattern;

private:
  FailureOr<Value> buildOneBlockGroupResult(
      OpTy op, Value sourcePart, Value maskPart, Type resultType,
      VRegType expectedResultType, MaskType expectedMaskType,
      OneToNPatternRewriter &rewriter) const {
    bool mismatchedTypes = sourcePart.getType() != expectedResultType ||
                           maskPart.getType() != expectedMaskType ||
                           resultType != expectedResultType;
    if (mismatchedTypes) {
      return rewriter.notifyMatchFailure(
          op, "vcg group_reduce path requires uniform physical chunk types");
    }
    return rewriter
        .create<GroupReduceOpTy>(op.getLoc(), expectedResultType, sourcePart,
                                 maskPart)
        .getResult();
  }

  LogicalResult lowerOneBlock(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, OneToNPatternRewriter &rewriter) const {
    bool invalidArity = sourceParts.size() != maskParts.size() ||
                        sourceParts.size() != resultTypes.size() ||
                        sourceParts.empty();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "vcg group_reduce path requires matching physical arity");
    }
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(
          op, "vcg group_reduce path requires physical vreg/mask");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourceIndex, sourcePart] : llvm::enumerate(sourceParts)) {
      FailureOr<Value> result = buildOneBlockGroupResult(
          op, sourcePart, maskParts[sourceIndex], resultTypes[sourceIndex],
          resultType, maskType, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildTwoBlockGroupResult(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, int64_t resultIndex, int64_t resultPartCount,
      int64_t numGroups, VRegType resultType, MaskType maskType,
      OneToNPatternRewriter &rewriter) const {
    Value loSource = sourceParts[resultIndex];
    Value hiSource = sourceParts[resultPartCount + resultIndex];
    Value loMask = maskParts[resultIndex];
    Value hiMask = maskParts[resultPartCount + resultIndex];
    bool mismatchedTypes = resultTypes[resultIndex] != resultType ||
                           loSource.getType() != resultType ||
                           hiSource.getType() != resultType ||
                           loMask.getType() != maskType ||
                           hiMask.getType() != maskType;
    if (mismatchedTypes) {
      return rewriter.notifyMatchFailure(
          op, "two-block group_reduce requires uniform physical types");
    }
    int64_t activeGroups = std::min<int64_t>(8, numGroups - resultIndex * 8);
    FailureOr<Value> combineMask = createPrefixMaskForActiveLanes(
        op.getLoc(), maskType, activeGroups, rewriter);
    if (failed(combineMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create two-block group_reduce combine mask");
    }
    Value lo = rewriter
                   .create<GroupReduceOpTy>(op.getLoc(), resultType, loSource,
                                            loMask)
                   .getResult();
    Value hi = rewriter
                   .create<GroupReduceOpTy>(op.getLoc(), resultType, hiSource,
                                            hiMask)
                   .getResult();
    return rewriter
        .create<CombineOpTy>(op.getLoc(), resultType, lo, hi, *combineMask)
        .getResult();
  }

  LogicalResult lowerTwoBlock(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, int64_t numGroups,
      OneToNPatternRewriter &rewriter) const {
    int64_t resultPartCount = resultTypes.size();
    bool invalidArity = static_cast<int64_t>(sourceParts.size()) !=
                            resultPartCount * 2 ||
                        maskParts.size() != sourceParts.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(op,
                                         "two-block group_reduce arity mismatch");
    }
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(
          op, "two-block group_reduce requires physical vreg/mask");
    }

    SmallVector<Value> results;
    results.reserve(resultPartCount);
    for (int64_t resultIndex = 0; resultIndex < resultPartCount;
         ++resultIndex) {
      FailureOr<Value> result = buildTwoBlockGroupResult(
          op, sourceParts, maskParts, resultTypes, resultIndex, resultPartCount,
          numGroups, resultType, maskType, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildFourBlockGroupResult(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, int64_t resultIndex, int64_t resultPartCount,
      int64_t numGroups, VRegType resultType, MaskType maskType,
      OneToNPatternRewriter &rewriter) const {
    int64_t activeGroups = std::min<int64_t>(8, numGroups - resultIndex * 8);
    FailureOr<Value> combineMask = createPrefixMaskForActiveLanes(
        op.getLoc(), maskType, activeGroups, rewriter);
    if (failed(combineMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create four-block group_reduce combine mask");
    }
    SmallVector<Value, 4> partials;
    partials.reserve(4);
    for (int64_t part = 0; part < 4; ++part) {
      int64_t sourceIndex = part * resultPartCount + resultIndex;
      Value source = sourceParts[sourceIndex];
      Value mask = maskParts[sourceIndex];
      bool mismatchedTypes = resultTypes[resultIndex] != resultType ||
                             source.getType() != resultType ||
                             mask.getType() != maskType;
      if (mismatchedTypes) {
        return rewriter.notifyMatchFailure(
            op, "four-block group_reduce requires uniform physical types");
      }
      partials.push_back(rewriter
                             .create<GroupReduceOpTy>(op.getLoc(), resultType,
                                                      source, mask)
                             .getResult());
    }
    Value sum01 = rewriter
                      .create<CombineOpTy>(op.getLoc(), resultType, partials[0],
                                           partials[1], *combineMask)
                      .getResult();
    Value sum23 = rewriter
                      .create<CombineOpTy>(op.getLoc(), resultType, partials[2],
                                           partials[3], *combineMask)
                      .getResult();
    return rewriter
        .create<CombineOpTy>(op.getLoc(), resultType, sum01, sum23,
                             *combineMask)
        .getResult();
  }

  LogicalResult lowerFourBlock(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, int64_t numGroups,
      OneToNPatternRewriter &rewriter) const {
    int64_t resultPartCount = resultTypes.size();
    bool invalidArity = static_cast<int64_t>(sourceParts.size()) !=
                            resultPartCount * 4 ||
                        maskParts.size() != sourceParts.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(op,
                                         "four-block group_reduce arity mismatch");
    }
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(
          op, "four-block group_reduce requires physical vreg/mask");
    }

    SmallVector<Value> results;
    results.reserve(resultPartCount);
    for (int64_t resultIndex = 0; resultIndex < resultPartCount;
         ++resultIndex) {
      FailureOr<Value> result = buildFourBlockGroupResult(
          op, sourceParts, maskParts, resultTypes, resultIndex,
          resultPartCount, numGroups, resultType, maskType, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<SmallVector<Value>> buildDeinterleaved2GroupResults(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      int64_t groupCount, int64_t chunksPerGroup, int64_t chunksPerPart,
      VRegType sourcePartType, VRegType rowResultType, MaskType maskType,
      Value firstLaneMask, OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(groupCount);
    for (int64_t group = 0; group < groupCount; ++group) {
      Value accumulator;
      for (int64_t chunk = 0; chunk < chunksPerGroup; ++chunk) {
        int64_t loIndex = group * chunksPerGroup + chunk;
        int64_t hiIndex = chunksPerPart + loIndex;
        bool mismatchedTypes =
            sourceParts[loIndex].getType() != sourcePartType ||
            sourceParts[hiIndex].getType() != sourcePartType ||
            maskParts[loIndex].getType() != maskType ||
            maskParts[hiIndex].getType() != maskType;
        if (mismatchedTypes) {
          return rewriter.notifyMatchFailure(
              op, "deinterleaved=2 group_reduce requires uniform physical "
                  "chunk types");
        }
        Value low = rewriter
                        .create<RowReduceOpTy>(op.getLoc(), rowResultType,
                                               sourceParts[loIndex],
                                               maskParts[loIndex])
                        .getResult();
        Value high = rewriter
                         .create<RowReduceOpTy>(op.getLoc(), rowResultType,
                                                sourceParts[hiIndex],
                                                maskParts[hiIndex])
                         .getResult();
        Value pair = rewriter
                         .create<CombineOpTy>(op.getLoc(), rowResultType, low,
                                              high, firstLaneMask)
                         .getResult();
        accumulator =
            accumulator
                ? rewriter
                      .create<CombineOpTy>(op.getLoc(), rowResultType, pair,
                                           accumulator, firstLaneMask)
                      .getResult()
                : pair;
      }
      results.push_back(accumulator);
    }
    return results;
  }

  FailureOr<SmallVector<Value>> restoreDeinterleaved2GroupResults(
      OpTy op, ArrayRef<Value> reducedResults, TypeRange resultTypes,
      VRegType resultType, OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (Value reducedResult : reducedResults) {
      FailureOr<Value> finalResult =
          bitcastVReg(op.getLoc(), reducedResult, resultType, rewriter);
      if (failed(finalResult)) {
        return rewriter.notifyMatchFailure(
            op, "failed to restore deinterleaved=2 group result type");
      }
      results.push_back(*finalResult);
    }
    return results;
  }

  struct Deinterleaved2GroupReduceTypes {
    VRegType resultType;
    MaskType maskType;
    VRegType sourcePartType;
    VRegType rowResultType;
    MaskType rowMaskType;
  };

  FailureOr<Deinterleaved2GroupReduceTypes>
  getDeinterleaved2GroupReduceTypes(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, OneToNPatternRewriter &rewriter) const {
    for (Type resultType : resultTypes) {
      if (!isa<VRegType>(resultType)) {
        return rewriter.notifyMatchFailure(
            op, "deinterleaved=2 group_reduce result must be vreg");
      }
    }
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(
          op, "deinterleaved=2 group_reduce requires physical vreg/mask");
    }
    auto sourcePartType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!sourcePartType) {
      return rewriter.notifyMatchFailure(
          op, "deinterleaved=2 group_reduce source must be vreg");
    }
    FailureOr<VRegType> rowResultType =
        getRowResultType(sourcePartType, resultType);
    if (failed(rowResultType)) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive deinterleaved=2 row-reduction type");
    }
    FailureOr<MaskType> rowMaskType =
        getMaskTypeForVReg(*rowResultType, rewriter.getContext());
    if (failed(rowMaskType)) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive deinterleaved=2 combine mask type");
    }
    return Deinterleaved2GroupReduceTypes{
        resultType, maskType, sourcePartType, *rowResultType, *rowMaskType};
  }

  FailureOr<std::pair<int64_t, int64_t>> validateFullDeinterleaved2Shape(
      OpTy op, VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      ValueRange sourceParts, ValueRange maskParts, TypeRange resultTypes,
      int64_t groupSize, OneToNPatternRewriter &rewriter) const {
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    bool rowLocalSlots1Result = resultLayout && resultLayout.isGroupSlots() &&
                                resultLayout.getSlots() == 1;
    if (!rowLocalSlots1Result) {
      return rewriter.notifyMatchFailure(
          op, "deinterleaved=2 full group_reduce requires slots=1 result");
    }
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(sourceVMIType.getElementType());
    if (failed(lanesPerPart)) {
      return rewriter.notifyMatchFailure(
          op, "deinterleaved=2 group_reduce requires known physical lanes");
    }
    int64_t safeLanesPerPart = *lanesPerPart > 0 ? *lanesPerPart : 1;
    int64_t safeGroupSize = groupSize > 0 ? groupSize : 1;
    bool invalidGroupSize = groupSize % (2 * safeLanesPerPart) != 0;
    if (invalidGroupSize) {
      return rewriter.notifyMatchFailure(
          op, "deinterleaved=2 group_reduce requires group size to be a "
              "multiple of two physical chunks");
    }
    int64_t groupCount = sourceVMIType.getElementCount() / safeGroupSize;
    int64_t chunksPerGroup = groupSize / (2 * safeLanesPerPart);
    int64_t chunksPerPart = groupCount * chunksPerGroup;
    bool invalidArity = sourceParts.size() != maskParts.size() ||
                        static_cast<int64_t>(sourceParts.size()) !=
                            2 * chunksPerPart ||
                        static_cast<int64_t>(resultTypes.size()) != groupCount;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "deinterleaved=2 group_reduce arity mismatch");
    }
    return std::make_pair(groupCount, chunksPerGroup);
  }

  LogicalResult lowerFullDeinterleaved2(
      OpTy op, VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      ValueRange sourceParts, ValueRange maskParts, TypeRange resultTypes,
      int64_t groupSize, OneToNPatternRewriter &rewriter) const {
    FailureOr<std::pair<int64_t, int64_t>> shape =
        validateFullDeinterleaved2Shape(
            op, sourceVMIType, resultVMIType, sourceParts, maskParts,
            resultTypes, groupSize, rewriter);
    if (failed(shape)) {
      return failure();
    }
    int64_t groupCount = shape->first;
    int64_t chunksPerGroupPerPart = shape->second;
    int64_t chunksPerPart = groupCount * chunksPerGroupPerPart;
    FailureOr<Deinterleaved2GroupReduceTypes> types =
        getDeinterleaved2GroupReduceTypes(op, sourceParts, maskParts,
                                          resultTypes, rewriter);
    if (failed(types)) {
      return failure();
    }
    FailureOr<Value> firstLaneMask =
        createPrefixMask(op.getLoc(), types->rowMaskType, "PAT_VL1", rewriter);
    if (failed(firstLaneMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create deinterleaved=2 group_reduce lane mask");
    }
    FailureOr<SmallVector<Value>> reducedResults =
        buildDeinterleaved2GroupResults(
            op, sourceParts, maskParts, groupCount, chunksPerGroupPerPart,
            chunksPerPart, types->sourcePartType, types->rowResultType,
            types->maskType,
            *firstLaneMask, rewriter);
    if (failed(reducedResults)) {
      return failure();
    }
    FailureOr<SmallVector<Value>> results = restoreDeinterleaved2GroupResults(
        op, *reducedResults, resultTypes, types->resultType, rewriter);
    if (failed(results)) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildContiguousGroupReduceResult(
      OpTy op, ValueRange sourceParts, ValueRange maskParts, int64_t group,
      int64_t chunksPerGroup, VRegType sourcePartType, VRegType rowResultType,
      MaskType maskType, Value firstLaneMask,
      OneToNPatternRewriter &rewriter) const {
    Value accumulator;
    for (int64_t chunk = 0; chunk < chunksPerGroup; ++chunk) {
      int64_t index = group * chunksPerGroup + chunk;
      bool mismatchedTypes = sourceParts[index].getType() != sourcePartType ||
                             maskParts[index].getType() != maskType;
      if (mismatchedTypes) {
        return rewriter.notifyMatchFailure(
            op, "group_reduce requires uniform physical chunk types");
      }
      Value reduced = rewriter
                          .create<RowReduceOpTy>(op.getLoc(), rowResultType,
                                                 sourceParts[index],
                                                 maskParts[index])
                          .getResult();
      accumulator = accumulator
                        ? rewriter
                              .create<CombineOpTy>(op.getLoc(), rowResultType,
                                                   reduced, accumulator,
                                                   firstLaneMask)
                              .getResult()
                        : reduced;
    }
    return accumulator;
  }

  FailureOr<SmallVector<Value>> buildContiguousGroupReduceResults(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      int64_t groupCount, int64_t chunksPerGroup, VRegType sourcePartType,
      VRegType rowResultType, MaskType maskType, Value firstLaneMask,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(groupCount);
    for (int64_t group = 0; group < groupCount; ++group) {
      FailureOr<Value> result = buildContiguousGroupReduceResult(
          op, sourceParts, maskParts, group, chunksPerGroup, sourcePartType,
          rowResultType, maskType, firstLaneMask, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    return results;
  }

  FailureOr<SmallVector<Value>> restoreContiguousGroupResults(
      OpTy op, ArrayRef<Value> reducedResults, TypeRange resultTypes,
      VRegType resultType, int64_t groupCount, int64_t chunksPerGroup,
      bool rowLocalSlots1Result,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results(resultTypes.size());
    for (int64_t group = 0; group < groupCount; ++group) {
      FailureOr<Value> finalResult =
          bitcastVReg(op.getLoc(), reducedResults[group], resultType, rewriter);
      if (failed(finalResult)) {
        return rewriter.notifyMatchFailure(
            op, "failed to restore group result type");
      }
      int64_t destChunk = rowLocalSlots1Result ? group : group * chunksPerGroup;
      if (rowLocalSlots1Result) {
        results[destChunk] = *finalResult;
      } else {
        for (int64_t chunk = 0; chunk < chunksPerGroup; ++chunk) {
          results[destChunk + chunk] = *finalResult;
        }
      }
    }
    return results;
  }

  struct ContiguousGroupReduceTypes {
    VRegType resultType;
    MaskType maskType;
    VRegType sourcePartType;
    VRegType rowResultType;
    MaskType rowMaskType;
  };

  struct ContiguousGroupReduceShape {
    int64_t lanesPerPart;
    int64_t groupCount;
    int64_t chunksPerGroup;
    bool rowLocalSlots1Result;
    int64_t expectedResultParts;
  };

  FailureOr<ContiguousGroupReduceShape> getContiguousGroupReduceShape(
      OpTy op, VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      ValueRange sourceParts, ValueRange maskParts, TypeRange resultTypes,
      int64_t groupSize, OneToNPatternRewriter &rewriter) const {
    int64_t lanesPerPart = 0;
    int64_t groupCount = 0;
    int64_t chunksPerGroup = 0;
    if (failed(checkContiguousFullGroupChunks(op, sourceVMIType, groupSize,
                                              &lanesPerPart, &groupCount,
                                              &chunksPerGroup, rewriter))) {
      return failure();
    }
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    bool rowLocalSlots1Result = resultLayout && resultLayout.isGroupSlots() &&
                                resultLayout.getNumGroups() == groupCount &&
                                resultLayout.getSlots() == 1;
    int64_t expectedResultParts =
        rowLocalSlots1Result ? groupCount : groupCount * chunksPerGroup;
    bool invalidArity =
        sourceParts.size() != maskParts.size() ||
        static_cast<int64_t>(sourceParts.size()) != groupCount * chunksPerGroup ||
        static_cast<int64_t>(resultTypes.size()) != expectedResultParts;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "group_reduce requires matching source/mask/result arity");
    }
    return ContiguousGroupReduceShape{lanesPerPart, groupCount, chunksPerGroup,
                                      rowLocalSlots1Result, expectedResultParts};
  }

  FailureOr<ContiguousGroupReduceTypes> getContiguousGroupReduceTypes(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, OneToNPatternRewriter &rewriter) const {
    for (Type resultType : resultTypes) {
      if (!isa<VRegType>(resultType)) {
        return rewriter.notifyMatchFailure(
            op, "group_reduce result must be vreg");
      }
    }
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(
          op, "group_reduce requires physical vreg result and mask");
    }
    auto sourcePartType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!sourcePartType) {
      return rewriter.notifyMatchFailure(op,
                                         "group_reduce source must be vreg");
    }
    FailureOr<VRegType> rowResultType =
        getRowResultType(sourcePartType, resultType);
    if (failed(rowResultType)) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive group row-reduction type");
    }
    FailureOr<MaskType> rowMaskType =
        getMaskTypeForVReg(*rowResultType, rewriter.getContext());
    if (failed(rowMaskType)) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive group combine mask type");
    }
    return ContiguousGroupReduceTypes{
        resultType, maskType, sourcePartType, *rowResultType, *rowMaskType};
  }

  LogicalResult lowerRowLocalContiguousGroupResults(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, const ContiguousGroupReduceShape &shape,
      const ContiguousGroupReduceTypes &types, Value firstLaneMask,
      OneToNPatternRewriter &rewriter) const {
    // One contiguous group maps to one result chunk: materialize the row
    // reduction and its final type view per group so the emitted ops keep
    // master's per-group ordering.
    SmallVector<Value> results(resultTypes.size());
    for (int64_t group = 0;
         group < shape.groupCount &&
         static_cast<size_t>(group) < results.size();
         ++group) {
      FailureOr<Value> reduced = buildContiguousGroupReduceResult(
          op, sourceParts, maskParts, group, shape.chunksPerGroup,
          types.sourcePartType, types.rowResultType, types.maskType,
          firstLaneMask, rewriter);
      if (failed(reduced)) {
        return failure();
      }
      FailureOr<Value> finalResult =
          bitcastVReg(op.getLoc(), *reduced, types.resultType, rewriter);
      if (failed(finalResult)) {
        return rewriter.notifyMatchFailure(
            op, "failed to restore group result type");
      }
      results[group] = *finalResult;
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  LogicalResult lowerContiguousRows(
      OpTy op, VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      ValueRange sourceParts, ValueRange maskParts, TypeRange resultTypes,
      int64_t groupSize, OneToNPatternRewriter &rewriter) const {
    FailureOr<ContiguousGroupReduceShape> shape =
        getContiguousGroupReduceShape(op, sourceVMIType, resultVMIType,
                                      sourceParts, maskParts, resultTypes,
                                      groupSize, rewriter);
    if (failed(shape)) {
      return failure();
    }
    FailureOr<ContiguousGroupReduceTypes> types =
        getContiguousGroupReduceTypes(op, sourceParts, maskParts, resultTypes,
                                      rewriter);
    if (failed(types)) {
      return failure();
    }
    FailureOr<Value> firstLaneMask =
        createPrefixMask(op.getLoc(), types->rowMaskType, "PAT_VL1", rewriter);
    if (failed(firstLaneMask)) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to create group_reduce masks");
    }
    if (shape->rowLocalSlots1Result) {
      return lowerRowLocalContiguousGroupResults(
          op, sourceParts, maskParts, resultTypes, *shape, *types,
          *firstLaneMask, rewriter);
    }
    return lowerBatchedContiguousGroupResults(
        op, sourceParts, maskParts, resultTypes, *shape, *types,
        *firstLaneMask, rewriter);
  }

  LogicalResult lowerBatchedContiguousGroupResults(
      OpTy op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, const ContiguousGroupReduceShape &shape,
      const ContiguousGroupReduceTypes &types, Value firstLaneMask,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<SmallVector<Value>> reducedResults =
        buildContiguousGroupReduceResults(
            op, sourceParts, maskParts, shape.groupCount, shape.chunksPerGroup,
            types.sourcePartType, types.rowResultType, types.maskType,
            firstLaneMask, rewriter);
    if (failed(reducedResults)) {
      return failure();
    }
    FailureOr<SmallVector<Value>> results = restoreContiguousGroupResults(
        op, *reducedResults, resultTypes, types.resultType, shape.groupCount,
        shape.chunksPerGroup, shape.rowLocalSlots1Result, rewriter);
    if (failed(results)) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                     *this->getTypeConverter());
    return success();
  }

public:

  LogicalResult
  matchAndRewrite(OpTy op,
                  typename OneToNOpConversionPattern<OpTy>::OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    ValueRange sourceParts = adaptor.getSource();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    bool failedResultTypeConversion = failed(maybe_resultTypes);
    if (failedResultTypeConversion) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);

    VMILayoutSupport supports;
    std::string supportReason;
    if (failed(getSupport(supports, op, &supportReason))) {
      return rewriter.notifyMatchFailure(
          op, Twine(op->getName().getStringRef()) +
                  " has no layout support: " + supportReason);
    }
    auto maskVMIType = cast<VMIMaskType>(op.getMask().getType());
    FailureOr<GroupReduceLoweringPlan> plan = classifyGroupReduceLoweringPlan(
        sourceVMIType, maskVMIType, resultVMIType,
        op.getNumGroupsAttr().getInt(), &supportReason);
    if (failed(plan)) {
      return rewriter.notifyMatchFailure(
          op, Twine(op->getName().getStringRef()) +
                  " has no lowering plan: " + supportReason);
    }

    FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
        sourceVMIType, op.getNumGroupsAttr().getInt());
    if (failed(groupSize)) {
      return rewriter.notifyMatchFailure(
          op, "group reduce requires num_groups to evenly divide lane count");
    }

    return lowerByPlan(op, *plan, *groupSize, sourceVMIType, resultVMIType,
                       sourceParts, maskParts, resultTypes, rewriter);
  }

private:
  LogicalResult lowerByPlan(
      OpTy op, GroupReduceLoweringPlan plan, int64_t groupSize,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      ValueRange sourceParts, ValueRange maskParts, TypeRange resultTypes,
      OneToNPatternRewriter &rewriter) const {
    int64_t numGroups = op.getNumGroupsAttr().getInt();
    switch (plan) {
    case GroupReduceLoweringPlan::OneBlockVcgadd:
      return lowerOneBlock(op, sourceParts, maskParts, resultTypes, rewriter);
    case GroupReduceLoweringPlan::TwoBlockDeinterleaved2VcgaddVadd:
      return lowerTwoBlock(op, sourceParts, maskParts, resultTypes, numGroups,
                           rewriter);
    case GroupReduceLoweringPlan::FourBlockDeinterleaved4VcgaddTree:
      return lowerFourBlock(op, sourceParts, maskParts, resultTypes, numGroups,
                            rewriter);
    case GroupReduceLoweringPlan::FullDeinterleaved2VcaddRows:
      return lowerFullDeinterleaved2(
          op, sourceVMIType, resultVMIType, sourceParts, maskParts, resultTypes,
          groupSize, rewriter);
    case GroupReduceLoweringPlan::ContiguousVcaddRows:
      return lowerContiguousRows(op, sourceVMIType, resultVMIType, sourceParts,
                                 maskParts, resultTypes, groupSize, rewriter);
    }
    return rewriter.notifyMatchFailure(op, "unknown group_reduce lowering plan");
  }

  FailureOr<VRegType> getRowResultType(VRegType sourceType,
                                       VRegType resultType) const {
    if constexpr (std::is_same_v<OpTy, VMIGroupReduceAddIOp>) {
      return getVcaddResultType(sourceType);
    }
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
    : OneToNOpConversionPattern<VMIGroupBroadcastOp> {
  using OneToNOpConversionPattern<VMIGroupBroadcastOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIGroupBroadcastOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<VMIPhysicalConversionInput> input =
        getVMIPhysicalConversionInput(op, adaptor, *this->getTypeConverter());
    if (failed(input)) {
      return failure();
    }
    SmallVector<Value> results;
    if (failed(lowerGroupBroadcastParts(
            op, input->sourceParts, input->sourceVMIType,
            input->resultVMIType, input->resultTypes,
            op.getNumGroupsAttr().getInt(), rewriter, results))) {
      return failure();
    }

    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// VMI vdhist / vchist → VPTO dhistv2 / chistv2 lowering (shared template)
//===----------------------------------------------------------------------===//

template <typename VMIOp, typename VPTOHistOp>
static LogicalResult lowerHistogramChunk(
    VMIOp op, Value source, Value userMask, int64_t firstLane,
    int64_t lanesPerPart, SmallVectorImpl<Value> &halves,
    ArrayRef<Value> binConsts, VRegType partType,
    OneToNPatternRewriter &rewriter) {
  auto maskType = dyn_cast<MaskType>(userMask.getType());
  if (!maskType || !maskType.isB8()) {
    return rewriter.notifyMatchFailure(op, "expected b8 source mask");
  }
  Value chunkMask = userMask;
  int64_t activeLanes = std::min<int64_t>(
      lanesPerPart,
      cast<VMIVRegType>(op.getSource().getType()).getElementCount() - firstLane);
  if (activeLanes < lanesPerPart) {
    FailureOr<Value> validMask = createPrefixMaskForActiveLanes(
        op.getLoc(), maskType, activeLanes, rewriter);
    FailureOr<Value> allMask =
        createAllTrueMask(op.getLoc(), maskType, rewriter);
    bool failedMask = failed(validMask) || failed(allMask);
    if (failedMask) {
      return rewriter.notifyMatchFailure(
          op, "failed to materialize tail-valid b8 mask");
    }
    chunkMask = rewriter
                    .create<PandOp>(op.getLoc(), maskType, chunkMask, *validMask,
                                    *allMask)
                    .getResult();
  }
  for (size_t half = 0; half < halves.size(); ++half) {
    halves[half] = rewriter
                       .create<VPTOHistOp>(op.getLoc(), partType, halves[half],
                                            source, chunkMask, binConsts[half])
                       .getResult();
  }
  return success();
}

template <typename VMIOp>
struct HistogramPhysicalPlan {
  ValueRange sourceParts;
  ValueRange maskParts;
  SmallVector<Value, 2> halves;
  SmallVector<Value, 2> binConsts;
  VRegType partType;
  int64_t lanesPerPart;
  size_t halfCount;
};

template <typename VMIOp>
static FailureOr<HistogramPhysicalPlan<VMIOp>> prepareHistogramPhysicalPlan(
    VMIOp op,
    typename OneToNOpConversionPattern<VMIOp>::OpAdaptor adaptor,
    OneToNPatternRewriter &rewriter) {
  ValueRange accParts = adaptor.getAcc();
  ValueRange sourceParts = adaptor.getSource();
  ValueRange maskParts = adaptor.getMask();
  size_t halfCount = accParts.size();
  const bool invalidHalfCount = halfCount != 1 && halfCount != 2;
  if (invalidHalfCount) {
      (void)rewriter.notifyMatchFailure(op,
                                        "expected one or two accumulator parts");
      return failure();
  }
  const bool invalidSourceMaskArity =
      sourceParts.empty() || sourceParts.size() != maskParts.size();
  if (invalidSourceMaskArity) {
      (void)rewriter.notifyMatchFailure(op,
                                        "expected matching source/mask chunks");
      return failure();
  }
  auto partType = dyn_cast<VRegType>(accParts.front().getType());
  if (!partType) {
    (void)rewriter.notifyMatchFailure(op, "expected ui16 acc parts");
    return failure();
  }
  const bool mismatchedSecondHalf =
      halfCount == 2 && accParts[1].getType() != partType;
  if (mismatchedSecondHalf) {
    (void)rewriter.notifyMatchFailure(op, "expected matching ui16 acc parts");
    return failure();
  }
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(sourceType.getElementType());
  if (failed(lanesPerPart)) {
    (void)rewriter.notifyMatchFailure(op, "failed to compute source lanes");
    return failure();
  }
  Location loc = op.getLoc();
  SmallVector<Value, 2> binConsts;
  binConsts.push_back(createI32Constant(loc, 0, rewriter));
  if (halfCount == 2) {
    binConsts.push_back(createI32Constant(loc, 1, rewriter));
  }
  return HistogramPhysicalPlan<VMIOp>{
      sourceParts, maskParts,
      SmallVector<Value, 2>(accParts.begin(), accParts.end()),
      std::move(binConsts), partType, *lanesPerPart, halfCount};
}

template <typename VMIOp, typename VPTOHistOp>
static LogicalResult
lowerVMIHistogramToVPTO(VMIOp op,
                        typename OneToNOpConversionPattern<VMIOp>::OpAdaptor
                            adaptor,
                        TypeConverter *typeConverter,
                        OneToNPatternRewriter &rewriter) {
  FailureOr<HistogramPhysicalPlan<VMIOp>> plan =
      prepareHistogramPhysicalPlan(op, adaptor, rewriter);
  if (failed(plan)) {
    return failure();
  }

  for (size_t index = 0, e = plan->sourceParts.size(); index < e; ++index) {
    if (failed(lowerHistogramChunk<VMIOp, VPTOHistOp>(
            op, plan->sourceParts[index], plan->maskParts[index],
            static_cast<int64_t>(index) * plan->lanesPerPart,
            plan->lanesPerPart, plan->halves, plan->binConsts, plan->partType,
            rewriter))) {
      return failure();
    }
  }

  replaceOpWithFlatConvertedValues(
      rewriter, op,
      SmallVector<Value>(plan->halves.begin(),
                         plan->halves.begin() + plan->halfCount),
      *typeConverter);
  return success();
}

struct OneToNVMIVdhistOpPattern : OneToNOpConversionPattern<VMIVdhistOp> {
  using OneToNOpConversionPattern<VMIVdhistOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIVdhistOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    return lowerVMIHistogramToVPTO<VMIVdhistOp, Dhistv2Op>(
        op, adaptor, this->getTypeConverter(), rewriter);
  }
};

struct OneToNVMIVchistOpPattern : OneToNOpConversionPattern<VMIVchistOp> {
  using OneToNOpConversionPattern<VMIVchistOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIVchistOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    return lowerVMIHistogramToVPTO<VMIVchistOp, Chistv2Op>(
        op, adaptor, this->getTypeConverter(), rewriter);
  }
};

template <typename SourceOp, typename ChunkReduceOp, typename CombineOp>
struct OneToNVMIReduceMinMaxOpPattern : OneToNOpConversionPattern<SourceOp> {
  using OneToNOpConversionPattern<SourceOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerSequentialReduction(
      SourceOp op, ValueRange sourceParts, ValueRange maskParts,
      VRegType resultType, MaskType maskType,
      OneToNPatternRewriter &rewriter) const {
    Value accumulator =
        rewriter
            .create<ChunkReduceOp>(op.getLoc(), resultType, sourceParts.front(),
                                   maskParts.front())
            .getResult();
    const bool singlePart = sourceParts.size() == 1;
    if (singlePart) {
      replaceOpWithFlatConvertedValues(
          rewriter, op, SmallVector<Value>{accumulator},
          *this->getTypeConverter());
      return success();
    }
    FailureOr<Value> firstLaneMask =
        createPrefixMask(op.getLoc(), maskType, "PAT_VL1", rewriter);
    if (failed(firstLaneMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create min/max reduction first-lane mask");
    }
    for (size_t part = 1; part < sourceParts.size(); ++part) {
      Value reduced = rewriter
                          .create<ChunkReduceOp>(op.getLoc(), resultType,
                                                 sourceParts[part],
                                                 maskParts[part])
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

  FailureOr<std::pair<VRegType, MaskType>> validatePhysicalParts(
      SourceOp op, ValueRange sourceParts, ValueRange maskParts,
      TypeRange resultTypes, OneToNPatternRewriter &rewriter) const {
    bool invalidArity = sourceParts.empty() || sourceParts.size() != maskParts.size() ||
                        resultTypes.size() != 1;
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "min/max reduction requires matching source/mask chunks and one result chunk");
    }
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(
          op, "min/max reduction requires matching physical source/result vregs and one mask");
    }
    for (Value sourcePart : sourceParts) {
      bool mismatch = sourcePart.getType() != resultType;
      if (mismatch) {
        return rewriter.notifyMatchFailure(
            op, "min/max reduction requires every source chunk to match result vreg type");
      }
    }
    for (Value maskPart : maskParts) {
      bool mismatch = maskPart.getType() != maskType;
      if (mismatch) {
        return rewriter.notifyMatchFailure(
            op, "min/max reduction requires every mask chunk to have the same predicate type");
      }
    }
    return std::make_pair(resultType, maskType);
  }

  LogicalResult lowerReduction(SourceOp op, ValueRange sourceParts,
                               ValueRange maskParts, VRegType resultType,
                               MaskType maskType,
                               OneToNPatternRewriter &rewriter) const {
    FailureOr<Value> combined = combineEquivalentMaskedParts<CombineOp>(
        op.getLoc(), sourceParts, maskParts, resultType, rewriter);
    if (succeeded(combined)) {
      Value reduced =
          rewriter
              .create<ChunkReduceOp>(op.getLoc(), resultType, *combined,
                                     maskParts.front())
              .getResult();
      replaceOpWithFlatConvertedValues(
          rewriter, op, SmallVector<Value>{reduced}, *this->getTypeConverter());
      return success();
    }

    return lowerSequentialReduction(op, sourceParts, maskParts, resultType,
                                    maskType, rewriter);
  }

public:

  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OneToNOpConversionPattern<SourceOp>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    ValueRange maskParts = adaptor.getMask();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    bool failedResultTypeConversion = failed(maybe_resultTypes);
    if (failedResultTypeConversion) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    FailureOr<std::pair<VRegType, MaskType>> physical =
        validatePhysicalParts(op, sourceParts, maskParts, resultTypes, rewriter);
    if (failed(physical)) {
      return failure();
    }
    return lowerReduction(op, sourceParts, maskParts, physical->first,
                          physical->second, rewriter);
  }
};

struct OneToNVMIExtFOpPattern : OneToNOpConversionPattern<VMIExtFOp> {
  using OneToNOpConversionPattern<VMIExtFOp>::OneToNOpConversionPattern;

private:
  struct ExtFPhysicalPlan {
    VRegType sourceType;
    SmallVector<VRegType> resultTypes;
  };

  FailureOr<ExtFPhysicalPlan> buildPhysicalPlan(
      VMIExtFOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    if (sourceParts.empty()) {
      return rewriter.notifyMatchFailure(
          op, "extf requires at least one physical source chunk");
    }
    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(op, "expected physical extf source");
    }
    for (Value sourcePart : sourceParts) {
      auto currentSourceType = dyn_cast<VRegType>(sourcePart.getType());
      if (!currentSourceType || currentSourceType != sourceType) {
        return rewriter.notifyMatchFailure(
            op, "extf source physical parts must have matching type");
      }
    }
    SmallVector<VRegType> resultVRegTypes;
    resultVRegTypes.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      bool invalidFirstResult =
          resultVRegTypes.empty() &&
          (!resultVRegType ||
           !(resultVRegType.getElementType().isF32() ||
             pto::isPTOBF16x2Type(resultVRegType.getElementType())));
      bool mismatchedResult =
          !resultVRegTypes.empty() && resultVRegType != resultVRegTypes.front();
      if (invalidFirstResult || mismatchedResult) {
        return rewriter.notifyMatchFailure(
            op, "unsupported physical extf result type");
      }
      resultVRegTypes.push_back(resultVRegType);
    }
    return ExtFPhysicalPlan{sourceType, std::move(resultVRegTypes)};
  }

  struct ResultViewPlan {
    bool isPackedBF16x2;
    VRegType vcvtResultType;
  };

  ResultViewPlan buildResultViewPlan(ArrayRef<VRegType> resultTypes,
                                     OneToNPatternRewriter &rewriter) const {
    bool isPackedBF16x2 =
        pto::isPTOBF16x2Type(resultTypes.front().getElementType());
    VRegType vcvtResultType = resultTypes.front();
    if (isPackedBF16x2) {
      vcvtResultType = VRegType::get(
          rewriter.getContext(), resultTypes.front().getElementCount() * 2,
          BFloat16Type::get(rewriter.getContext()));
    }
    return ResultViewPlan{isPackedBF16x2, vcvtResultType};
  }

  static Value createVcvtResult(Location loc, VRegType resultType,
                                Value sourcePart, Value mask, StringAttr rnd,
                                StringAttr sat, StringAttr part,
                                bool resultIsPackedBF16x2,
                                VRegType vcvtResultVRegType,
                                OneToNPatternRewriter &rewriter) {
    VRegType vcvtType = resultIsPackedBF16x2 ? vcvtResultVRegType : resultType;
    Value vcvt = rewriter
                     .create<VcvtOp>(loc, vcvtType, sourcePart, mask, rnd, sat,
                                     part)
                     .getResult();
    if (!resultIsPackedBF16x2) {
      return vcvt;
    }
    return rewriter.create<VbitcastOp>(loc, resultType, vcvt).getResult();
  }

  LogicalResult lowerLaneStride(
      VMIExtFOp op, OneToNPatternRewriter &rewriter, ValueRange sourceParts,
      ArrayRef<VRegType> resultTypes, Value mask, StringRef part,
      bool resultIsPackedBF16x2, VRegType vcvtResultVRegType) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      results.push_back(createVcvtResult(
          op.getLoc(), resultType, sourcePart, mask, /*rnd=*/nullptr,
          /*sat=*/nullptr, rewriter.getStringAttr(part),
          resultIsPackedBF16x2, vcvtResultVRegType, rewriter));
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  LogicalResult lowerPackedE2M1LaneStride2(
      VMIExtFOp op, ValueRange sourceParts, const ExtFPhysicalPlan &plan,
      OneToNPatternRewriter &rewriter) const {
    static constexpr StringRef kPacked2Parts[] = {"P0", "P2"};
    FailureOr<Value> mask = createSeedMask(op, plan.sourceType, rewriter);
    if (failed(mask)) {
      return failure();
    }
    ResultViewPlan viewPlan = buildResultViewPlan(plan.resultTypes, rewriter);
    return lowerFactor(op, rewriter, sourceParts, plan.resultTypes,
                       kPacked2Parts, 2, *mask, viewPlan.isPackedBF16x2,
                       viewPlan.vcvtResultType);
  }

  LogicalResult lowerFactor(
      VMIExtFOp op, OneToNPatternRewriter &rewriter, ValueRange sourceParts,
      ArrayRef<VRegType> resultTypes, ArrayRef<StringRef> parts,
      int64_t factor, Value mask, bool resultIsPackedBF16x2,
      VRegType vcvtResultVRegType) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (int64_t partIndex = 0; partIndex < factor; ++partIndex) {
      for (auto [chunkIndex, sourcePart] : llvm::enumerate(sourceParts)) {
        VRegType resultType =
            resultTypes[partIndex * sourceParts.size() + chunkIndex];
        results.push_back(createVcvtResult(
            op.getLoc(), resultType, sourcePart, mask, /*rnd=*/nullptr,
            /*sat=*/nullptr, rewriter.getStringAttr(parts[partIndex]),
            resultIsPackedBF16x2, vcvtResultVRegType, rewriter));
      }
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  struct ExtFFactorPlan {
    ArrayRef<StringRef> parts;
    int64_t factor;
  };

  FailureOr<ExtFFactorPlan> buildFactorPlan(
      VMIExtFOp op, unsigned sourceBits, size_t sourcePartCount,
      size_t resultPartCount, OneToNPatternRewriter &rewriter) const {
    if (sourceBits == 16 && resultPartCount == 2 * sourcePartCount) {
      static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
      return ExtFFactorPlan{ArrayRef<StringRef>(kEvenOddParts), 2};
    }
    if (sourceBits == 8 && resultPartCount == 4 * sourcePartCount) {
      static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
      return ExtFFactorPlan{ArrayRef<StringRef>(kPacked4Parts), 4};
    }
    return rewriter.notifyMatchFailure(
        op, "unsupported physical extf source/result width relation");
  }

  FailureOr<Value> createSeedMask(VMIExtFOp op, VRegType sourceType,
                                   OneToNPatternRewriter &rewriter) const {
    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(op, "failed to build extf seed mask");
    }
    return *mask;
  }

  LogicalResult lowerPhysicalExtF(
      VMIExtFOp op, ValueRange sourceParts, const ExtFPhysicalPlan &plan,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      OneToNPatternRewriter &rewriter) const {
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(plan.sourceType.getElementType());
    // A packed bf16x2 physical result cannot be produced directly by
    // pto.vcvt (classifyVcvtElemType has no BF16x2 branch); the widest native
    // f4 conversion result element is bf16. Build the bf16 view type (2 bf16
    // lanes per bf16x2 lane) and reinterpret each vcvt result with a
    // physical-noop VbitcastOp, mirroring the source-side reinterpret in
    // OneToNVMITruncFOpPattern (viewVcvtSource).
    ResultViewPlan viewPlan = buildResultViewPlan(plan.resultTypes, rewriter);
    bool denseLaneStrideExtension =
        sourceLayout && resultLayout && sourceLayout.isContiguous() &&
        resultLayout.isContiguous() && resultLayout.getLaneStride() == 1 &&
        ((sourceBits == 16 && sourceLayout.getLaneStride() == 2) ||
         (sourceBits == 8 && sourceLayout.getLaneStride() == 4)) &&
        plan.resultTypes.size() == sourceParts.size();
    if (denseLaneStrideExtension) {
      StringRef part = sourceBits == 16 ? StringRef("EVEN") : StringRef("P0");
      FailureOr<Value> mask = createSeedMask(op, plan.sourceType, rewriter);
      if (failed(mask)) {
        return failure();
      }
      return lowerLaneStride(op, rewriter, sourceParts, plan.resultTypes, *mask,
                             part, viewPlan.isPackedBF16x2,
                             viewPlan.vcvtResultType);
    }

    // Packed f4E2M1x2 sources stored with lane_stride = 2 (UNPK_B8) have
    // valid bytes on the even lanes; P0 (lanes 0 mod 4) plus P2 (lanes 2 mod
    // 4) cover them while P1/P3 are zero-fill gaps, so widen through the
    // {P0, P2} part pair instead of the dense factor-4 selection.
    bool packedE2M1LaneStride2 =
        sourceLayout && sourceLayout.isContiguous() &&
        sourceLayout.getLaneStride() == 2 &&
        isa<pto::F4E2M1x2Type>(plan.sourceType.getElementType()) &&
        plan.resultTypes.size() == 2 * sourceParts.size();
    if (packedE2M1LaneStride2) {
      return lowerPackedE2M1LaneStride2(op, sourceParts, plan, rewriter);
    }

    FailureOr<ExtFFactorPlan> factorPlan = buildFactorPlan(
        op, sourceBits, sourceParts.size(), plan.resultTypes.size(), rewriter);
    if (failed(factorPlan)) {
      return failure();
    }
    FailureOr<Value> mask = createSeedMask(op, plan.sourceType, rewriter);
    if (failed(mask)) {
      return failure();
    }
    return lowerFactor(op, rewriter, sourceParts, plan.resultTypes,
                       factorPlan->parts, factorPlan->factor, *mask,
                       viewPlan.isPackedBF16x2,
                       viewPlan.vcvtResultType);
  }

public:

  LogicalResult
  matchAndRewrite(VMIExtFOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<VMIPhysicalConversionInput> input =
        getVMIPhysicalConversionInput(op, adaptor, *this->getTypeConverter());
    if (failed(input)) {
      return failure();
    }
    FailureOr<ExtFPhysicalPlan> plan =
        buildPhysicalPlan(op, input->sourceParts, input->resultTypes, rewriter);
    if (failed(plan)) {
      return failure();
    }
    VMILayoutAttr sourceLayout = input->sourceVMIType.getLayoutAttr();
    VMILayoutAttr resultLayout = input->resultVMIType.getLayoutAttr();
    return lowerPhysicalExtF(op, input->sourceParts, *plan, sourceLayout,
                             resultLayout, rewriter);
  }
};

static bool hasUnsupportedPackedTruncFConversion(Type sourceElementType,
                                                 Type resultElementType) {
  bool usesPackedCarrier =
      isVMIPackedFloatCarrierType(sourceElementType) ||
      isVMIPackedFloatCarrierType(resultElementType);
  return usesPackedCarrier &&
         !lookupVMIFpToFpContract(sourceElementType, resultElementType);
}

static bool hasGroupSlotTruncFLayouts(VMILayoutAttr sourceLayout,
                                      VMILayoutAttr resultLayout) {
  return sourceLayout && resultLayout && sourceLayout.isGroupSlots() &&
         resultLayout.isGroupSlots();
}

struct OneToNVMITruncFOpPattern : OneToNOpConversionPattern<VMITruncFOp> {
  using OneToNOpConversionPattern<VMITruncFOp>::OneToNOpConversionPattern;

private:
  struct TruncFPhysicalPlan {
    VRegType sourceType;
    SmallVector<VRegType> resultTypes;
    VRegType sourceViewType;
    unsigned sourceBits;
    unsigned resultBits;
    bool sourceIsPackedBF16x2;
  };

  struct TruncFNarrowingPlan {
    ArrayRef<StringRef> parts;
    int64_t sourceFactor;
    int64_t resultLaneStride;
  };

  FailureOr<VRegType> getUniformSourceType(
      VMITruncFOp op, ValueRange sourceParts,
      OneToNPatternRewriter &rewriter) const {
    if (sourceParts.empty()) {
      return rewriter.notifyMatchFailure(op,
                                         "truncf requires source chunks");
    }
    auto firstType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!firstType) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical truncf source type");
    }
    for (Value sourcePart : sourceParts) {
      auto sourceType = dyn_cast<VRegType>(sourcePart.getType());
      if (!sourceType || sourceType != firstType) {
        return rewriter.notifyMatchFailure(
            op, "truncf source physical parts must have matching type");
      }
    }
    return firstType;
  }

  FailureOr<SmallVector<VRegType>> getUniformResultTypes(
      VMITruncFOp op, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<VRegType> resultVRegTypes;
    resultVRegTypes.reserve(resultTypes.size());
    for (Type physicalResultType : resultTypes) {
      auto resultType = dyn_cast<VRegType>(physicalResultType);
      bool invalidType = !resultType ||
                         (!resultVRegTypes.empty() &&
                          resultType != resultVRegTypes.front());
      if (invalidType) {
        return rewriter.notifyMatchFailure(
            op, "unsupported physical truncf result type");
      }
      resultVRegTypes.push_back(resultType);
    }
    return resultVRegTypes;
  }

  FailureOr<TruncFPhysicalPlan> buildPhysicalPlan(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    if (resultTypes.empty()) {
      return rewriter.notifyMatchFailure(op, "truncf requires result chunks");
    }
    FailureOr<VRegType> sourceType =
        getUniformSourceType(op, sourceParts, rewriter);
    if (failed(sourceType)) {
      return failure();
    }
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(sourceType->getElementType());
    if (sourceBits != 32 && sourceBits != 16) {
      return rewriter.notifyMatchFailure(
          op, "truncf source bit width must be 32 or 16");
    }
    FailureOr<SmallVector<VRegType>> resultVRegTypes =
        getUniformResultTypes(op, resultTypes, rewriter);
    if (failed(resultVRegTypes)) {
      return failure();
    }
    unsigned resultBits = pto::getPTOStorageElemBitWidth(
        resultVRegTypes->front().getElementType());
    if (resultBits == 0) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical truncf result type");
    }
    bool sourceIsPackedBF16x2 =
        pto::isPTOBF16x2Type(sourceType->getElementType());
    VRegType sourceViewType = *sourceType;
    if (sourceIsPackedBF16x2) {
      sourceViewType = VRegType::get(
          rewriter.getContext(), sourceType->getElementCount() * 2,
          BFloat16Type::get(rewriter.getContext()));
    }
    return TruncFPhysicalPlan{*sourceType, std::move(*resultVRegTypes),
                              sourceViewType, sourceBits, resultBits,
                              sourceIsPackedBF16x2};
  }

  LogicalResult lowerDenseLaneStride(
      VMITruncFOp op, ValueRange sourceParts,
      ArrayRef<VRegType> resultTypes, StringRef part,
      bool sourceIsPackedBF16x2, VRegType vcvtSourceVRegType,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<Value> sourceMask =
        createAllTrueMaskForVReg(op.getLoc(), vcvtSourceVRegType, rewriter);
    if (failed(sourceMask)) {
      return rewriter.notifyMatchFailure(op, "failed to build truncf masks");
    }
    StringAttr rnd = rewriter.getStringAttr(
        getTruncFRoundMode(op, resultTypes.front().getElementType()));
    StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
    StringAttr partAttr = rewriter.getStringAttr(part);
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      results.push_back(rewriter
                            .create<VcvtOp>(
                                op.getLoc(), resultType,
                                makeVcvtSourceView(
                                    op.getLoc(), sourcePart,
                                    sourceIsPackedBF16x2, vcvtSourceVRegType,
                                    rewriter),
                                *sourceMask, rnd, sat, partAttr)
                            .getResult());
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> lowerGroupSlotTruncPart(
      VMITruncFOp op, Value sourcePart, Type physicalResultType,
      Value activeSlotMask, StringAttr sat,
      OneToNPatternRewriter &rewriter) const {
    auto sourceType = dyn_cast<VRegType>(sourcePart.getType());
    auto resultType = dyn_cast<VRegType>(physicalResultType);
    const bool invalidTypes =
        !sourceType || !sourceType.getElementType().isF32() || !resultType;
    if (invalidTypes) {
      (void)rewriter.notifyMatchFailure(
          op, "unsupported group-slot truncf physical type");
      return failure();
    }
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(resultType.getElementType());
    const bool unsupportedResultBits = resultBits != 16 && resultBits != 8;
    if (unsupportedResultBits) {
      (void)rewriter.notifyMatchFailure(
          op, "unsupported group-slot truncf physical type");
      return failure();
    }
    StringAttr part =
        rewriter.getStringAttr(resultBits == 16 ? "EVEN" : "P0");
    StringAttr rnd = rewriter.getStringAttr(
        getTruncFRoundMode(op, resultType.getElementType()));
    return rewriter
        .create<VcvtOp>(op.getLoc(), resultType, sourcePart, activeSlotMask,
                        rnd, sat, part)
        .getResult();
  }

  static Value makeVcvtSourceView(Location loc, Value sourcePart,
                                  bool sourceIsPackedBF16x2,
                                  VRegType vcvtSourceVRegType,
                                  OneToNPatternRewriter &rewriter) {
    if (!sourceIsPackedBF16x2) {
      return sourcePart;
    }
    if (auto vbc = sourcePart.getDefiningOp<VbitcastOp>()) {
      if (auto srcVReg = dyn_cast<VRegType>(vbc.getInput().getType());
          srcVReg && srcVReg.getElementType().isBF16()) {
        return vbc.getInput();
      }
    }
    return rewriter.create<VbitcastOp>(loc, vcvtSourceVRegType, sourcePart)
        .getResult();
  }

  LogicalResult lowerSameWidth(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes,
      VRegType sourceViewType, StringAttr sat,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<Value> sourceMask =
        createAllTrueMaskForVReg(op.getLoc(), sourceViewType, rewriter);
    if (failed(sourceMask)) {
      return rewriter.notifyMatchFailure(op, "failed to build truncf masks");
    }
    StringAttr rnd = rewriter.getStringAttr(
        getTruncFRoundMode(op, resultTypes.front().getElementType()));
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      results.push_back(rewriter
                            .create<VcvtOp>(op.getLoc(), resultType, sourcePart,
                                            *sourceMask, rnd, sat,
                                            /*part=*/nullptr)
                            .getResult());
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildNarrowTruncResult(
      VMITruncFOp op, ValueRange sourceParts, VRegType resultType,
      ArrayRef<StringRef> allParts, int64_t chunkIndex, int64_t sourceFactor,
      int64_t resultLaneStride, VRegType sourceViewType,
      bool sourceIsPackedBF16x2, Value sourceMask, StringAttr rnd,
      StringAttr sat, OneToNPatternRewriter &rewriter) const {
    if (sourceFactor <= 0) {
      return rewriter.notifyMatchFailure(
          op, "narrow truncf requires a positive source factor");
    }
    int64_t safeSourceFactor = sourceFactor;
    FailureOr<Value> resultMask =
        createAllTrueMaskForVReg(op.getLoc(), resultType, rewriter);
    if (failed(resultMask)) {
      return failure();
    }
    SmallVector<Value> partials;
    partials.reserve(safeSourceFactor);
    for (int64_t partIndex = 0; partIndex < safeSourceFactor; ++partIndex) {
      Value sourcePart =
          sourceParts[partIndex * (sourceParts.size() / safeSourceFactor) +
                      chunkIndex];
      bool hasIndexedPart =
          partIndex * resultLaneStride < static_cast<int64_t>(allParts.size());
      StringRef part =
          hasIndexedPart ? allParts[partIndex * resultLaneStride]
                         : allParts[partIndex];
      partials.push_back(
          rewriter
              .create<VcvtOp>(
                  op.getLoc(), resultType,
                  makeVcvtSourceView(op.getLoc(), sourcePart,
                                     sourceIsPackedBF16x2, sourceViewType,
                                     rewriter),
                  sourceMask, rnd, sat, rewriter.getStringAttr(part))
              .getResult());
    }
    Value merged = partials.front();
    for (Value partial : llvm::drop_begin(partials)) {
      merged = rewriter
                   .create<VorOp>(op.getLoc(), resultType, merged, partial,
                                  *resultMask)
                   .getResult();
    }
    return merged;
  }

  LogicalResult lowerNarrow(
      VMITruncFOp op, ValueRange sourceParts,
      ArrayRef<VRegType> resultTypes, ArrayRef<StringRef> allParts,
      int64_t sourceFactor, int64_t resultLaneStride,
      VRegType sourceViewType, bool sourceIsPackedBF16x2,
      StringAttr rnd, StringAttr sat,
      OneToNPatternRewriter &rewriter) const {
    if (sourceFactor <= 0 || resultLaneStride <= 0 ||
        sourceParts.size() !=
            static_cast<size_t>(sourceFactor) * resultTypes.size()) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical truncf source/result arity relation");
    }
    FailureOr<Value> sourceMask =
        createAllTrueMaskForVReg(op.getLoc(), sourceViewType, rewriter);
    if (failed(sourceMask)) {
      return rewriter.notifyMatchFailure(op, "failed to build truncf masks");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [chunkIndex, resultType] : llvm::enumerate(resultTypes)) {
      FailureOr<Value> result = buildNarrowTruncResult(
          op, sourceParts, resultType, allParts, chunkIndex, sourceFactor,
          resultLaneStride, sourceViewType, sourceIsPackedBF16x2, *sourceMask,
          rnd, sat, rewriter);
      if (failed(result)) {
        return rewriter.notifyMatchFailure(
            op, "failed to build truncf result mask");
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<TruncFNarrowingPlan> buildNarrowingPlan(
      VMITruncFOp op, unsigned sourceBits, unsigned resultBits,
      VMILayoutAttr resultLayout, size_t sourcePartCount,
      size_t resultPartCount, OneToNPatternRewriter &rewriter) const {
    ArrayRef<StringRef> parts;
    int64_t factor = 0;
    if (resultBits * 2 == sourceBits) {
      static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
      parts = kEvenOddParts;
      factor = 2;
    } else if (resultBits * 4 == sourceBits) {
      static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
      parts = kPacked4Parts;
      factor = 4;
    } else {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical truncf source/result width relation");
    }
    int64_t resultLaneStride = resultLayout && resultLayout.isContiguous()
                                   ? resultLayout.getLaneStride()
                                   : 1;
    bool invalidResultLaneStride =
        resultLaneStride <= 0 || factor % resultLaneStride != 0;
    if (invalidResultLaneStride) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical truncf result lane stride");
    }
    int64_t sourceFactor = factor / resultLaneStride;
    bool sourceArityMismatch =
        sourcePartCount != static_cast<size_t>(sourceFactor) * resultPartCount;
    if (sourceArityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical truncf source/result arity relation");
    }
    return TruncFNarrowingPlan{parts, sourceFactor, resultLaneStride};
  }

  LogicalResult lowerGroupSlotTrunc(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      OneToNPatternRewriter &rewriter) const {
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(resultVMIType.getElementType());
    bool invalidShape =
        sourceLayout.getNumGroups() != resultLayout.getNumGroups() ||
        sourceLayout.getSlots() != resultLayout.getSlots() ||
        (sourceLayout.getSlots() != 1 && sourceLayout.getSlots() != 8) ||
        !sourceVMIType.getElementType().isF32() ||
        (resultBits != 16 && resultBits != 8) ||
        sourceParts.size() != resultTypes.size();
    if (invalidShape) {
      return rewriter.notifyMatchFailure(op, "unsupported group-slot truncf shape");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    const char *activeSlotPattern =
        sourceLayout.getSlots() == 1 ? "PAT_VL1" : "PAT_VL8";
    FailureOr<Value> activeSlotMask = createPrefixMask(
        op.getLoc(), MaskType::get(rewriter.getContext(), "b32"),
        activeSlotPattern, rewriter);
    if (failed(activeSlotMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to build group-slot truncf active slot mask");
    }
    StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
    for (auto [sourcePart, physicalResultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      FailureOr<Value> result = lowerGroupSlotTruncPart(
          op, sourcePart, physicalResultType, *activeSlotMask, sat, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<bool> tryLowerSameWidthTrunc(
      VMITruncFOp op, ValueRange sourceParts,
      const TruncFPhysicalPlan &physicalPlan, VMILayoutAttr sourceLayout,
      VMILayoutAttr resultLayout, OneToNPatternRewriter &rewriter) const {
    bool sameWidthContiguous =
        physicalPlan.sourceBits == physicalPlan.resultBits && sourceLayout &&
        resultLayout && sourceLayout.isContiguous() &&
        sourceLayout.getLaneStride() == 1 && resultLayout.isContiguous() &&
        resultLayout.getLaneStride() == 1 &&
        sourceParts.size() == physicalPlan.resultTypes.size();
    if (!sameWidthContiguous) {
      return false;
    }
    StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
    if (failed(lowerSameWidth(op, sourceParts, physicalPlan.resultTypes,
                              physicalPlan.sourceViewType, sat, rewriter))) {
      return failure();
    }
    return true;
  }

  FailureOr<bool> tryLowerDenseLaneStrideTrunc(
      VMITruncFOp op, ValueRange sourceParts,
      const TruncFPhysicalPlan &physicalPlan, VMILayoutAttr sourceLayout,
      VMILayoutAttr resultLayout, OneToNPatternRewriter &rewriter) const {
    bool denseLaneStrideNarrowing =
        sourceLayout && resultLayout && sourceLayout.isContiguous() &&
        sourceLayout.getLaneStride() == 1 && resultLayout.isContiguous() &&
        resultLayout.getLaneStride() != 1 &&
        sourceParts.size() == physicalPlan.resultTypes.size();
    if (!denseLaneStrideNarrowing) {
      return false;
    }
    bool isEven32To16 = physicalPlan.resultBits == 16 &&
                          resultLayout.getLaneStride() == 2;
    bool isPacked32To8 = physicalPlan.resultBits == 8 &&
                           resultLayout.getLaneStride() == 4;
    bool isEven16To8 = physicalPlan.resultBits == 8 &&
                         resultLayout.getLaneStride() == 2;
    if (!isEven32To16 && !isPacked32To8 && !isEven16To8) {
      return rewriter.notifyMatchFailure(
          op, "unsupported dense lane_stride truncf result layout");
    }
    StringRef part = isPacked32To8 ? "P0" : "EVEN";
    if (failed(lowerDenseLaneStride(
            op, sourceParts, physicalPlan.resultTypes, part,
            physicalPlan.sourceIsPackedBF16x2, physicalPlan.sourceViewType,
            rewriter))) {
      return failure();
    }
    return true;
  }

  LogicalResult lowerNonGroupSlotTrunc(
      VMITruncFOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<TruncFPhysicalPlan> physicalPlan =
        buildPhysicalPlan(op, sourceParts, resultTypes, rewriter);
    if (failed(physicalPlan)) {
      return failure();
    }
    bool unsupportedGroupSlotLayout = sourceLayout && sourceLayout.isGroupSlots();
    if (unsupportedGroupSlotLayout) {
      return rewriter.notifyMatchFailure(
          op, "group-slot layout for non-f32 truncf not supported");
    }
    FailureOr<bool> sameWidth = tryLowerSameWidthTrunc(
        op, sourceParts, *physicalPlan, sourceLayout, resultLayout, rewriter);
    if (failed(sameWidth)) {
      return failure();
    }
    if (*sameWidth) {
      return success();
    }
    FailureOr<bool> denseLaneStride = tryLowerDenseLaneStrideTrunc(
        op, sourceParts, *physicalPlan, sourceLayout, resultLayout, rewriter);
    if (failed(denseLaneStride)) {
      return failure();
    }
    if (*denseLaneStride) {
      return success();
    }
    FailureOr<TruncFNarrowingPlan> narrowingPlan = buildNarrowingPlan(
        op, physicalPlan->sourceBits, physicalPlan->resultBits, resultLayout,
        sourceParts.size(),
        resultTypes.size(), rewriter);
    if (failed(narrowingPlan)) {
      return failure();
    }
    StringAttr rnd = rewriter.getStringAttr(
        getTruncFRoundMode(op, physicalPlan->resultTypes.front().getElementType()));
    StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
    return lowerNarrow(op, sourceParts, physicalPlan->resultTypes,
                       narrowingPlan->parts,
                       narrowingPlan->sourceFactor,
                       narrowingPlan->resultLaneStride,
                       physicalPlan->sourceViewType,
                       physicalPlan->sourceIsPackedBF16x2, rnd, sat, rewriter);
  }

public:

  LogicalResult
  matchAndRewrite(VMITruncFOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    Type sourceElementType = sourceVMIType.getElementType();
    Type resultElementType = resultVMIType.getElementType();
    if (hasUnsupportedPackedTruncFConversion(sourceElementType,
                                             resultElementType)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported packed fp-to-fp truncf conversion");
    }
    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);

    VMILayoutAttr sourceLayout = sourceVMIType.getLayoutAttr();
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    if (hasGroupSlotTruncFLayouts(sourceLayout, resultLayout)) {
      return lowerGroupSlotTrunc(op, sourceParts, resultTypes, sourceLayout,
                                 resultLayout, sourceVMIType, resultVMIType,
                                 rewriter);
    }

    return lowerNonGroupSlotTrunc(op, sourceParts, resultTypes, sourceLayout,
                                  resultLayout, rewriter);
  }
};

template <typename OpT>
struct OneToNVMIExtIOpPattern : OneToNOpConversionPattern<OpT> {
  using OneToNOpConversionPattern<OpT>::OneToNOpConversionPattern;

private:
  FailureOr<Value> buildFactorExtensionResult(
      OpT op, Value sourcePart, VRegType resultType, Value mask,
      StringRef part, OneToNPatternRewriter &rewriter) const {
    return rewriter
        .create<VcvtOp>(op.getLoc(), resultType, sourcePart, mask,
                        /*rnd=*/nullptr, /*sat=*/nullptr,
                        rewriter.getStringAttr(part))
        .getResult();
  }

  LogicalResult emitFactorExtension(
      OpT op, ValueRange sourceParts, ArrayRef<VRegType> resultVRegTypes,
      ArrayRef<StringRef> parts, int64_t factor, Value mask,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(resultVRegTypes.size());
    for (int64_t partIndex = 0; partIndex < factor; ++partIndex) {
      for (auto [chunkIndex, sourcePart] : llvm::enumerate(sourceParts)) {
        VRegType resultType =
            resultVRegTypes[partIndex * sourceParts.size() + chunkIndex];
        FailureOr<Value> result = buildFactorExtensionResult(
            op, sourcePart, resultType, mask, parts[partIndex], rewriter);
        if (failed(result)) {
          return failure();
        }
        results.push_back(*result);
      }
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildDenseLaneExtensionResult(
      OpT op, Value sourcePart, VRegType resultType, Value mask,
      StringRef part, OneToNPatternRewriter &rewriter) const {
    return rewriter
        .create<VcvtOp>(op.getLoc(), resultType, sourcePart, mask,
                        /*rnd=*/nullptr, /*sat=*/nullptr,
                        rewriter.getStringAttr(part))
        .getResult();
  }

  LogicalResult emitDenseLaneExtension(
      OpT op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes,
      VRegType sourceType, StringRef part,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to build integer extension seed mask");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      FailureOr<Value> result = buildDenseLaneExtensionResult(
          op, sourcePart, resultType, *mask, part, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<std::pair<ArrayRef<StringRef>, int64_t>> getExtensionPartPlan(
      OpT op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      unsigned sourceBits, unsigned resultBits,
      OneToNPatternRewriter &rewriter) const {
    if (resultBits == sourceBits * 2 &&
        resultTypes.size() == 2 * sourceParts.size()) {
      static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
      return std::make_pair(ArrayRef<StringRef>(kEvenOddParts), int64_t{2});
    }
    if (resultBits == sourceBits * 4 &&
        resultTypes.size() == 4 * sourceParts.size()) {
      static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
      return std::make_pair(ArrayRef<StringRef>(kPacked4Parts), int64_t{4});
    }
    return rewriter.notifyMatchFailure(
        op, "unsupported physical integer extension source/result width relation");
  }

  FailureOr<VRegType> validateDenseGroupSlotExtensionResult(
      OpT op, Type resultType, VMIVRegType sourceVMIType,
      VMIVRegType resultVMIType, unsigned sourceBits, unsigned resultBits,
      OneToNPatternRewriter &rewriter) const {
    if (sourceBits == 0 || resultBits == 0) {
      return rewriter.notifyMatchFailure(
          op, "group-slot integer extension requires positive bit widths");
    }
    FailureOr<int64_t> sourceLanes =
        getDataLanesPerPart(sourceVMIType.getElementType());
    FailureOr<int64_t> resultLanes =
        getDataLanesPerPart(resultVMIType.getElementType());
    bool carrierShapeMismatch =
        failed(sourceLanes) || failed(resultLanes) ||
        *sourceLanes !=
            *resultLanes * static_cast<int64_t>(resultBits / sourceBits);
    auto physicalResultType = dyn_cast<VRegType>(resultType);
    bool invalidResultType =
        !physicalResultType || failed(resultLanes) ||
        physicalResultType.getElementCount() != *resultLanes ||
        pto::getPTOStorageElemBitWidth(physicalResultType.getElementType()) !=
            resultBits;
    if (carrierShapeMismatch || invalidResultType) {
      return rewriter.notifyMatchFailure(
          op, carrierShapeMismatch
                  ? "unsupported dense group-slot integer extension carrier shape"
                  : "unsupported dense group-slot integer extension result type");
    }
    return physicalResultType;
  }

  FailureOr<Value> buildDenseGroupSlotExtensionResult(
      OpT op, Value sourcePart, Type resultType, VMIVRegType sourceVMIType,
      VMIVRegType resultVMIType, IntegerType resultIntegerType,
      unsigned sourceBits, unsigned resultBits,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<VRegType> physicalResultType =
        validateDenseGroupSlotExtensionResult(
            op, resultType, sourceVMIType, resultVMIType, sourceBits,
            resultBits, rewriter);
    if (failed(physicalResultType)) {
      return failure();
    }

    Value current = sourcePart;
    unsigned currentBits = sourceBits;
    while (currentBits < resultBits) {
      FailureOr<Value> next = extendDenseGroupSlotCarrier(
          op, current, currentBits, resultIntegerType, rewriter);
      if (failed(next)) {
        return failure();
      }
      current = *next;
      currentBits *= 2;
    }
    FailureOr<Value> result =
        bitcastVReg(op.getLoc(), current, *physicalResultType, rewriter);
    if (failed(result)) {
      return rewriter.notifyMatchFailure(
          op, "failed to materialize dense group-slot unpack result");
    }
    return *result;
  }

  FailureOr<Value> extendDenseGroupSlotCarrier(
      OpT op, Value current, unsigned currentBits,
      IntegerType resultIntegerType,
      OneToNPatternRewriter &rewriter) const {
    unsigned nextBits = currentBits * 2;
    auto nextElementType = IntegerType::get(
        rewriter.getContext(), nextBits, resultIntegerType.getSignedness());
    FailureOr<int64_t> nextLanes = getDataLanesPerPart(nextElementType);
    if (failed(nextLanes)) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive dense group-slot unpack result lanes");
    }
    auto nextType =
        VRegType::get(rewriter.getContext(), *nextLanes, nextElementType);
    auto currentType = dyn_cast<VRegType>(current.getType());
    bool unpackLaneMismatch =
        !currentType || currentType.getElementCount() != *nextLanes * 2;
    if (unpackLaneMismatch) {
      return rewriter.notifyMatchFailure(
          op, "dense group-slot unpack source/result lane mismatch");
    }
    Value part = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    if constexpr (std::is_same_v<OpT, VMIExtSIOp>) {
      return rewriter.create<VsunpackOp>(op.getLoc(), nextType, current, part)
          .getResult();
    }
    return rewriter.create<VzunpackOp>(op.getLoc(), nextType, current, part)
        .getResult();
  }

  LogicalResult lowerDenseGroupSlotExtension(
      OpT op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      IntegerType resultIntegerType, unsigned sourceBits, unsigned resultBits,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      FailureOr<Value> result = buildDenseGroupSlotExtensionResult(
          op, sourcePart, resultType, sourceVMIType, resultVMIType,
          resultIntegerType, sourceBits, resultBits, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildLegacyGroupSlotExtensionResult(
      OpT op, Value sourcePart, Type resultType, VRegType conversionSourceType,
      Value slotMask, StringAttr part, unsigned resultBits,
      OneToNPatternRewriter &rewriter) const {
    auto resultVRegType = dyn_cast<VRegType>(resultType);
    bool invalidResultType =
        !resultVRegType ||
        pto::getPTOStorageElemBitWidth(resultVRegType.getElementType()) !=
            resultBits;
    if (invalidResultType) {
      return rewriter.notifyMatchFailure(
          op, "unsupported group-slot integer extension result type");
    }
    FailureOr<Value> conversionSource = bitcastVReg(
        op.getLoc(), sourcePart, conversionSourceType, rewriter);
    if (failed(conversionSource)) {
      return rewriter.notifyMatchFailure(
          op, "failed to expose group-slot extension source elements");
    }
    return rewriter
        .create<VcvtOp>(op.getLoc(), resultVRegType, *conversionSource, slotMask,
                        /*rnd=*/nullptr, /*sat=*/nullptr, part)
        .getResult();
  }

  FailureOr<std::tuple<int64_t, VRegType, Value>> prepareLegacyGroupSlotExtension(
      OpT op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMIVRegType sourceVMIType, VMILayoutAttr sourceLayout,
      VMILayoutAttr resultLayout, unsigned sourceBits, unsigned resultBits,
      OneToNPatternRewriter &rewriter) const {
    bool invalidShape =
        sourceLayout.getNumGroups() != resultLayout.getNumGroups() ||
        sourceLayout.getSlots() != resultLayout.getSlots() ||
        (sourceLayout.getSlots() != 1 && sourceLayout.getSlots() != 8) ||
        sourceBits == 0 || sourceBits >= resultBits ||
        resultBits % sourceBits != 0 ||
        (resultBits / sourceBits != 2 && resultBits / sourceBits != 4) ||
        (sourceLayout.getSlots() == 8 &&
         sourceLayout.getLaneStride() != resultBits / sourceBits) ||
        resultLayout.getLaneStride() != 1 ||
        sourceParts.size() != resultTypes.size();
    if (invalidShape) {
      return rewriter.notifyMatchFailure(
          op, "unsupported group-slot integer extension shape");
    }
    FailureOr<int64_t> sourceLanes =
        getDataLanesPerPart(sourceVMIType.getElementType());
    if (failed(sourceLanes)) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive group-slot integer extension source lanes");
    }
    auto conversionSourceType = VRegType::get(
        rewriter.getContext(), *sourceLanes, sourceVMIType.getElementType());
    FailureOr<MaskType> maskType =
        getMaskTypeForVReg(conversionSourceType, rewriter.getContext());
    if (failed(maskType)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create group-slot integer extension mask type");
    }
    FailureOr<Value> slotMask = createPrefixMaskForActiveLanes(
        op.getLoc(), *maskType,
        sourceLayout.getSlots() * sourceLayout.getLaneStride(), rewriter);
    if (failed(slotMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to build group-slot integer extension mask");
    }
    return std::make_tuple(static_cast<int64_t>(resultBits / sourceBits), conversionSourceType,
                           *slotMask);
  }

  LogicalResult lowerLegacyGroupSlotExtension(
      OpT op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      unsigned sourceBits, unsigned resultBits,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<std::tuple<int64_t, VRegType, Value>> preparation =
        prepareLegacyGroupSlotExtension(
            op, sourceParts, resultTypes, sourceVMIType, sourceLayout,
            resultLayout, sourceBits, resultBits, rewriter);
    if (failed(preparation)) {
      return failure();
    }
    int64_t widenFactor = std::get<0>(*preparation);
    VRegType conversionSourceType = std::get<1>(*preparation);
    Value slotMask = std::get<2>(*preparation);
    StringAttr part =
        rewriter.getStringAttr(widenFactor == 2 ? "EVEN" : "P0");
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      FailureOr<Value> result = buildLegacyGroupSlotExtensionResult(
          op, sourcePart, resultType, conversionSourceType, slotMask, part,
          resultBits, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  LogicalResult lowerPhysicalExtension(
      OpT op, ValueRange sourceParts, ArrayRef<VRegType> resultVRegTypes,
      ArrayRef<Type> resultTypes, VRegType sourceType, unsigned sourceBits,
      unsigned resultBits, VMILayoutAttr sourceLayout,
      VMILayoutAttr resultLayout, OneToNPatternRewriter &rewriter) const {
    bool denseLaneExtension =
        sourceLayout && resultLayout && sourceLayout.isContiguous() &&
        resultLayout.isContiguous() && resultLayout.getLaneStride() == 1 &&
        ((resultBits == sourceBits * 2 && sourceLayout.getLaneStride() == 2) ||
         (resultBits == sourceBits * 4 && sourceLayout.getLaneStride() == 4)) &&
        resultTypes.size() == sourceParts.size();
    if (denseLaneExtension) {
      StringRef part = resultBits == sourceBits * 2 ? StringRef("EVEN")
                                                    : StringRef("P0");
      return emitDenseLaneExtension(op, sourceParts, resultVRegTypes,
                                    sourceType, part, rewriter);
    }

    FailureOr<std::pair<ArrayRef<StringRef>, int64_t>> partPlan =
        getExtensionPartPlan(op, sourceParts, resultTypes, sourceBits,
                             resultBits, rewriter);
    if (failed(partPlan)) {
      return failure();
    }

    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to build integer extension seed mask");
    }

    return emitFactorExtension(op, sourceParts, resultVRegTypes, partPlan->first,
                               partPlan->second,
                               *mask, rewriter);
  }

  FailureOr<SmallVector<VRegType>> collectExtensionResultTypes(
      OpT op, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<VRegType> resultVRegTypes;
    resultVRegTypes.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      bool invalidType =
          !resultVRegType || !isa<IntegerType>(resultVRegType.getElementType()) ||
          (!resultVRegTypes.empty() &&
           resultVRegType != resultVRegTypes.front());
      if (invalidType) {
        return rewriter.notifyMatchFailure(
            op, "unsupported physical integer extension result type");
      }
      resultVRegTypes.push_back(resultVRegType);
    }
    return resultVRegTypes;
  }

  FailureOr<VRegType> getUniformExtensionSourceType(
      OpT op, ValueRange sourceParts,
      OneToNPatternRewriter &rewriter) const {
    if (sourceParts.empty()) {
      return rewriter.notifyMatchFailure(
          op, "integer extension requires at least one physical source chunk");
    }
    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(
          op, "expected physical integer extension source");
    }
    for (Value sourcePart : sourceParts) {
      auto currentSourceType = dyn_cast<VRegType>(sourcePart.getType());
      if (!currentSourceType || currentSourceType != sourceType) {
        return rewriter.notifyMatchFailure(
            op, "integer extension source physical parts must have matching "
                "type");
      }
    }
    return sourceType;
  }

  struct ExtensionLoweringInput {
    VMIVRegType sourceVMIType;
    VMIVRegType resultVMIType;
    ValueRange sourceParts;
    SmallVector<Type> resultTypes;
    VRegType sourceType;
    VMILayoutAttr sourceLayout;
    VMILayoutAttr resultLayout;
  };

  FailureOr<ExtensionLoweringInput> getLoweringInput(
      OpT op, typename OneToNOpConversionPattern<OpT>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(resultTypes)) {
      return failure();
    }
    FailureOr<VRegType> sourceType =
        getUniformExtensionSourceType(op, sourceParts, rewriter);
    if (failed(sourceType)) {
      return failure();
    }
    return ExtensionLoweringInput{
        sourceVMIType, resultVMIType, sourceParts, std::move(*resultTypes),
        *sourceType, sourceVMIType.getLayoutAttr(), resultVMIType.getLayoutAttr()};
  }

  LogicalResult lowerGroupSlotByLayout(
      OpT op, const ExtensionLoweringInput &input,
      OneToNPatternRewriter &rewriter) const {
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(input.sourceVMIType.getElementType());
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(input.resultVMIType.getElementType());
    auto sourceIntegerType =
        dyn_cast<IntegerType>(input.sourceVMIType.getElementType());
    auto resultIntegerType =
        dyn_cast<IntegerType>(input.resultVMIType.getElementType());
    int64_t slots = input.sourceLayout.getSlots();
    bool denseGroupSlotExtension =
        sourceIntegerType && resultIntegerType &&
        input.sourceLayout.getNumGroups() == input.resultLayout.getNumGroups() &&
        input.sourceLayout.getLaneStride() == 1 &&
        input.resultLayout.getLaneStride() == 1 &&
        input.sourceLayout.getSlots() == input.resultLayout.getSlots() &&
        (slots == 2 || slots == 4 || slots == 8) && sourceBits > 0 &&
        resultBits > sourceBits && resultBits % sourceBits == 0 &&
        (resultBits / sourceBits == 2 || resultBits / sourceBits == 4) &&
        input.sourceParts.size() == input.resultTypes.size();
    if (denseGroupSlotExtension) {
      return lowerDenseGroupSlotExtension(
          op, input.sourceParts, input.resultTypes, input.sourceVMIType,
          input.resultVMIType, resultIntegerType, sourceBits, resultBits,
          rewriter);
    }
    return lowerLegacyGroupSlotExtension(
        op, input.sourceParts, input.resultTypes, input.sourceVMIType,
        input.resultVMIType, input.sourceLayout, input.resultLayout,
        sourceBits, resultBits, rewriter);
  }

  LogicalResult lowerNonGroupSlotByLayout(
      OpT op, const ExtensionLoweringInput &input,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<SmallVector<VRegType>> resultVRegTypes =
        collectExtensionResultTypes(op, input.resultTypes, rewriter);
    if (failed(resultVRegTypes)) {
      return failure();
    }
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(input.sourceType.getElementType());
    unsigned resultBits = pto::getPTOStorageElemBitWidth(
        resultVRegTypes->front().getElementType());
    return lowerPhysicalExtension(
        op, input.sourceParts, *resultVRegTypes, input.resultTypes,
        input.sourceType, sourceBits, resultBits, input.sourceLayout,
        input.resultLayout, rewriter);
  }

public:
  LogicalResult
  matchAndRewrite(OpT op,
                  typename OneToNOpConversionPattern<OpT>::OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<ExtensionLoweringInput> input =
        getLoweringInput(op, adaptor, rewriter);
    if (failed(input)) {
      return failure();
    }
    bool groupSlotLayouts =
        input->sourceLayout && input->resultLayout &&
        input->sourceLayout.isGroupSlots() && input->resultLayout.isGroupSlots();
    if (groupSlotLayouts) {
      return lowerGroupSlotByLayout(op, *input, rewriter);
    }
    return lowerNonGroupSlotByLayout(op, *input, rewriter);
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
//     Lowering shape: NOSAT keeps/bitcasts the source carrier; SAT emits vcvt
//     into the logical-element vector whose live results occupy the requested
//     strided lanes.
//
// Group-slots logical layouts
//   - slots = 1 preserves the layout for 2x/4x narrowing.
//   - slots = 8 records the 2x/4x narrowing factor as the result lane_stride.
//   - 2x narrowing lowers with part = EVEN; 4x narrowing uses part = P0.
//   - 32-bit integer -> 8-bit integer, slots = 8, result lane_stride = 4
//     Lowering shape: no vcvt; keep/bitcast the 32-bit carrier and let the
//     later store consume it as PK4_B32.
struct OneToNVMITruncIOpPattern : OneToNOpConversionPattern<VMITruncIOp> {
  using OneToNOpConversionPattern<VMITruncIOp>::OneToNOpConversionPattern;

private:
  FailureOr<std::pair<VRegType, VRegType>> getUniformTruncTypes(
      VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    bool emptyPhysicalParts = sourceParts.empty() || resultTypes.empty();
    if (emptyPhysicalParts) {
      return rewriter.notifyMatchFailure(
          op, "trunci requires non-empty physical source and result parts");
    }
    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    bool invalidTypes =
        !sourceType || !isa<IntegerType>(sourceType.getElementType()) ||
        !resultType || !isa<IntegerType>(resultType.getElementType());
    if (invalidTypes) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci source/result type");
    }
    for (Value sourcePart : sourceParts) {
      auto currentType = dyn_cast<VRegType>(sourcePart.getType());
      if (!currentType || currentType != sourceType) {
        return rewriter.notifyMatchFailure(
            op, "trunci source physical parts must have matching integer type");
      }
    }
    for (Type physicalResultType : resultTypes) {
      auto currentType = dyn_cast<VRegType>(physicalResultType);
      if (!currentType || currentType != resultType) {
        return rewriter.notifyMatchFailure(
            op, "trunci result physical parts must have matching integer type");
      }
    }
    return std::make_pair(sourceType, resultType);
  }

  void finalizeResults(VMITruncIOp op, SmallVectorImpl<Value> &results,
                       bool s32ToS8Alias, ArrayRef<Type> originalResultTypes,
                       OneToNPatternRewriter &rewriter) const {
    if (s32ToS8Alias) {
      for (auto &&[index, result] : llvm::enumerate(results)) {
        result = rewriter
                     .create<VbitcastOp>(op.getLoc(), originalResultTypes[index],
                                         result)
                     .getResult();
      }
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
  }

  FailureOr<Value> lowerGroupSlotTruncPart(
      VMITruncIOp op, Value sourcePart, VRegType sourceType,
      VRegType resultType, VMIVRegType resultVMIType,
      VMILayoutAttr resultLayout, unsigned sourceLogicalBits,
      unsigned resultLogicalBits, Value activeSlotMask, StringAttr sat,
      OneToNPatternRewriter &rewriter) const {
    unsigned physicalResultBits =
        pto::getPTOStorageElemBitWidth(resultType.getElementType());
    bool directCarrier = resultLayout.hasLaneStride() &&
                         resultLayout.getLaneStride() == 4 &&
                         resultLogicalBits == 8 && physicalResultBits == 32;
    if (directCarrier) {
      return lowerGroupSlotDirectCarrier(op, sourcePart, resultType, rewriter);
    }
    bool wideCarrier = resultLayout.hasLaneStride() &&
                       resultLayout.getLaneStride() == 2 &&
                       resultLogicalBits == 16 && physicalResultBits == 32;
    if (wideCarrier) {
      return lowerGroupSlotWideCarrier(op, sourcePart, resultType,
                                       resultVMIType, activeSlotMask, sat,
                                       rewriter);
    }
    bool validNarrowResult = physicalResultBits == 16 || physicalResultBits == 8;
    if (!validNarrowResult) {
      return rewriter.notifyMatchFailure(
          op, "unsupported group-slot trunci physical type");
    }
    StringAttr part = rewriter.getStringAttr(
        sourceLogicalBits == 2 * resultLogicalBits ? "EVEN" : "P0");
    return rewriter
        .create<VcvtOp>(op.getLoc(), resultType, sourcePart, activeSlotMask,
                        nullptr, sat, part)
        .getResult();
  }

  FailureOr<Value> lowerGroupSlotDirectCarrier(
      VMITruncIOp op, Value sourcePart, VRegType resultType,
      OneToNPatternRewriter &rewriter) const {
    return sourcePart.getType() == resultType
               ? sourcePart
               : rewriter.create<VbitcastOp>(op.getLoc(), resultType, sourcePart)
                     .getResult();
  }

  FailureOr<Value> lowerGroupSlotWideCarrier(
      VMITruncIOp op, Value sourcePart, VRegType resultType,
      VMIVRegType resultVMIType, Value activeSlotMask, StringAttr sat,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<int64_t> lanes =
        getDataLanesPerPart(resultVMIType.getElementType());
    if (failed(lanes)) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive group-slot trunci conversion lanes");
    }
    auto conversionType = VRegType::get(
        rewriter.getContext(), *lanes, resultVMIType.getElementType());
    Value converted = rewriter
                          .create<VcvtOp>(op.getLoc(), conversionType,
                                          sourcePart, activeSlotMask, nullptr,
                                          sat, rewriter.getStringAttr("EVEN"))
                          .getResult();
    FailureOr<Value> carrier =
        bitcastVReg(op.getLoc(), converted, resultType, rewriter);
    if (failed(carrier)) {
      return rewriter.notifyMatchFailure(
          op, "failed to expose group-slot trunci result carrier");
    }
    return *carrier;
  }

  FailureOr<std::pair<bool, bool>> getGroupSlotTruncModes(
      VMITruncIOp op, VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      ValueRange sourceParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(sourceVMIType.getElementType());
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(resultVMIType.getElementType());
    bool supportsDirect =
        (sourceBits == 32 && (resultBits == 16 || resultBits == 8)) ||
        (sourceBits == 16 && resultBits == 8 && sourceLayout.getSlots() == 1);
    bool supportsPacked =
        sourceBits == 16 && resultBits == 8 && sourceLayout.getSlots() == 8 &&
        resultLayout.getSlots() == 8 && resultLayout.hasLaneStride() &&
        resultLayout.getLaneStride() == 2;
    bool invalidShape =
        sourceLayout.getNumGroups() != resultLayout.getNumGroups() ||
        sourceLayout.getSlots() != resultLayout.getSlots() ||
        (sourceLayout.getSlots() != 1 && sourceLayout.getSlots() != 8) ||
        (!supportsDirect && !supportsPacked) ||
        sourceParts.size() != resultTypes.size();
    if (invalidShape) {
      (void)rewriter.notifyMatchFailure(op,
                                        "unsupported group-slot trunci shape");
      return failure();
    }
    return std::make_pair(supportsDirect, supportsPacked);
  }

  FailureOr<SmallVector<Value>> lowerGroupSlotTruncParts(
      VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      VMILayoutAttr resultLayout, unsigned sourceBits, unsigned resultBits,
      bool supportsPacked, Value activeSlotMask, StringAttr sat,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, physicalResultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      auto sourceType = dyn_cast<VRegType>(sourcePart.getType());
      auto resultType = dyn_cast<VRegType>(physicalResultType);
      bool validPhysicalTypes =
          sourceType &&
          pto::getPTOStorageElemBitWidth(sourceType.getElementType()) ==
              sourceBits &&
          resultType;
      if (!validPhysicalTypes) {
        (void)rewriter.notifyMatchFailure(
            op, "unsupported group-slot trunci physical type");
        return failure();
      }
      if (supportsPacked) {
        results.push_back(rewriter
                              .create<VcvtOp>(op.getLoc(), resultType, sourcePart,
                                              activeSlotMask, nullptr, sat,
                                              rewriter.getStringAttr("EVEN"))
                              .getResult());
        continue;
      }
      FailureOr<Value> lowered = lowerGroupSlotTruncPart(
          op, sourcePart, sourceType, resultType, resultVMIType, resultLayout,
          sourceBits, resultBits, activeSlotMask, sat, rewriter);
      if (failed(lowered)) {
        return failure();
      }
      results.push_back(*lowered);
    }
    return results;
  }

  LogicalResult lowerGroupSlotTrunc(
      VMITruncIOp op, OpAdaptor adaptor, OneToNPatternRewriter &rewriter,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      ArrayRef<Type> resultTypes) const {
    ValueRange sourceParts = adaptor.getSource();
    FailureOr<std::pair<bool, bool>> modes = getGroupSlotTruncModes(
        op, sourceVMIType, resultVMIType, sourceLayout, resultLayout,
        sourceParts, resultTypes, rewriter);
    if (failed(modes)) {
      return failure();
    }
    unsigned sourceLogicalBits =
        pto::getPTOStorageElemBitWidth(sourceVMIType.getElementType());
    unsigned resultLogicalBits =
        pto::getPTOStorageElemBitWidth(resultVMIType.getElementType());
    bool supportsPacked = modes->second;

    StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
    const char *activeSlotPattern =
        sourceLayout.getSlots() == 1 ? "PAT_VL1" : "PAT_VL8";
    StringRef activeSlotGranularity = sourceLogicalBits == 16 ? "b16" : "b32";
    FailureOr<Value> activeSlotMask = createPrefixMask(
        op.getLoc(), MaskType::get(rewriter.getContext(), activeSlotGranularity),
        activeSlotPattern, rewriter);
    if (failed(activeSlotMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to build group-slot trunci active slot mask");
    }
    FailureOr<SmallVector<Value>> results = lowerGroupSlotTruncParts(
        op, sourceParts, resultTypes, sourceVMIType, resultVMIType,
        resultLayout, sourceLogicalBits, resultLogicalBits, supportsPacked,
        *activeSlotMask, sat, rewriter);
    if (failed(results)) {
      return failure();
    }
    finalizeResults(op, *results, false, resultTypes, rewriter);
    return success();
  }

  LogicalResult lowerDenseLaneStrideTrunc(
      VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      int64_t factor, StringAttr sat, bool s32ToS8Alias,
      ArrayRef<Type> originalResultTypes,
      OneToNPatternRewriter &rewriter) const {
    StringAttr part = rewriter.getStringAttr(factor == 2 ? "EVEN" : "P0");
    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported dense trunci source type");
    }
    FailureOr<Value> sourceMask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
    if (failed(sourceMask)) {
      return rewriter.notifyMatchFailure(op, "failed to build trunci masks");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      if (!resultVRegType) {
        return rewriter.notifyMatchFailure(
            op, "unsupported dense trunci result type");
      }
      results.push_back(rewriter
                            .create<VcvtOp>(op.getLoc(), resultVRegType,
                                            sourcePart, *sourceMask,
                                            /*rnd=*/nullptr, sat, part)
                            .getResult());
    }
    finalizeResults(op, results, s32ToS8Alias, originalResultTypes, rewriter);
    return success();
  }

  LogicalResult lowerNoSatDenseCarrier(
      VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      FailureOr<Value> result =
          bitcastVReg(op.getLoc(), sourcePart, resultType, rewriter);
      if (failed(result)) {
        return rewriter.notifyMatchFailure(
            op, "failed to forward NOSAT trunci carrier");
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildFactorTruncResult(
      VMITruncIOp op, ValueRange sourceParts, Type resultType,
      int64_t resultIndex, int64_t factor, ArrayRef<StringRef> parts,
      StringAttr sat, Value sourceMask, Value resultMask,
      OneToNPatternRewriter &rewriter) const {
    bool invalidFactor =
        factor <= 0 || static_cast<size_t>(factor) > parts.size();
    if (invalidFactor) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci conversion factor");
    }
    auto resultVRegType = dyn_cast<VRegType>(resultType);
    if (!resultVRegType) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci result type");
    }

    SmallVector<Value> partials;
    partials.reserve(static_cast<size_t>(factor));
    for (int64_t partIndex = 0; partIndex < factor; ++partIndex) {
      int64_t sourceIndex = resultIndex * factor + partIndex;
      if (sourceIndex < 0 ||
          sourceIndex >= static_cast<int64_t>(sourceParts.size())) {
        return rewriter.notifyMatchFailure(
            op, "trunci source part index exceeds physical arity");
      }
      partials.push_back(
          rewriter
              .create<VcvtOp>(op.getLoc(), resultVRegType,
                              sourceParts[sourceIndex], sourceMask, nullptr, sat,
                              rewriter.getStringAttr(parts[partIndex]))
              .getResult());
    }

    Value merged = partials.front();
    for (Value partial : llvm::drop_begin(partials)) {
      merged = rewriter
                   .create<VorOp>(op.getLoc(), resultVRegType, merged, partial,
                                  resultMask)
                   .getResult();
    }
    return merged;
  }

  LogicalResult lowerFactorTrunc(
      VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      ArrayRef<StringRef> parts, int64_t factor, StringAttr sat,
      bool s32ToS8Alias, ArrayRef<Type> originalResultTypes,
      OneToNPatternRewriter &rewriter) const {
    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    if (!sourceType || !resultType) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci source/result type");
    }
    FailureOr<Value> sourceMask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
    FailureOr<Value> resultMask =
        createAllTrueMaskForVReg(op.getLoc(), resultType, rewriter);
    bool failedMasks = failed(sourceMask) || failed(resultMask);
    if (failedMasks) {
      return rewriter.notifyMatchFailure(op, "failed to build trunci masks");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (int64_t resultIndex = 0;
         resultIndex < static_cast<int64_t>(resultTypes.size()); ++resultIndex) {
      FailureOr<Value> result = buildFactorTruncResult(
          op, sourceParts, resultTypes[resultIndex], resultIndex, factor, parts,
          sat, *sourceMask, *resultMask, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    finalizeResults(op, results, s32ToS8Alias, originalResultTypes, rewriter);
    return success();
  }

  struct TruncIPhysicalPlan {
    VRegType sourceType;
    VRegType resultType;
    int64_t factor;
    bool denseLaneStride;
    StringAttr saturate;
  };

  FailureOr<TruncIPhysicalPlan> buildPhysicalPlan(
      VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<std::pair<VRegType, VRegType>> uniformTypes =
        getUniformTruncTypes(op, sourceParts, resultTypes, rewriter);
    if (failed(uniformTypes)) {
      return failure();
    }
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(uniformTypes->first.getElementType());
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(uniformTypes->second.getElementType());
    bool invalidWidth = sourceBits == 0 || resultBits == 0 ||
                        sourceBits % resultBits != 0;
    if (invalidWidth) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci source/result width relation");
    }
    int64_t factor = sourceBits / resultBits;
    bool denseLaneStride =
        sourceLayout && resultLayout && sourceLayout.isContiguous() &&
        sourceLayout.getLaneStride() == 1 && resultLayout.isContiguous() &&
        resultLayout.getLaneStride() == factor &&
        sourceParts.size() == resultTypes.size();
    bool unsupportedDenseFactor = denseLaneStride && factor != 2 && factor != 4;
    if (unsupportedDenseFactor) {
      return rewriter.notifyMatchFailure(
          op, "unsupported dense lane_stride trunci result layout");
    }
    return TruncIPhysicalPlan{uniformTypes->first, uniformTypes->second, factor,
                              denseLaneStride,
                              op->getAttrOfType<StringAttr>("saturate")};
  }

  struct TruncIAliasPlan {
    SmallVector<Value> sourceParts;
    SmallVector<Type> resultTypes;
    SmallVector<Type> originalResultTypes;
    bool requiresResultBitcast = false;
  };

  TruncIAliasPlan materializeS32ToS8Alias(
      VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      const TruncIPhysicalPlan &physicalPlan,
      OneToNPatternRewriter &rewriter) const {
    TruncIAliasPlan aliasPlan;
    bool requiresAlias =
        physicalPlan.sourceType.getElementType().isSignedInteger(32) &&
        physicalPlan.resultType.getElementType().isSignedInteger(8);
    if (!requiresAlias) {
      aliasPlan.sourceParts.assign(sourceParts.begin(), sourceParts.end());
      aliasPlan.resultTypes.assign(resultTypes.begin(), resultTypes.end());
      return aliasPlan;
    }
    auto u32ElemTy = rewriter.getIntegerType(32, /*isSigned=*/false);
    auto u8ElemTy = rewriter.getIntegerType(8, /*isSigned=*/false);
    aliasPlan.originalResultTypes.assign(resultTypes.begin(), resultTypes.end());
    aliasPlan.sourceParts.reserve(sourceParts.size());
    for (Value sourcePart : sourceParts) {
      auto sourcePartType = cast<VRegType>(sourcePart.getType());
      aliasPlan.sourceParts.push_back(
          rewriter.create<VbitcastOp>(
              op.getLoc(), VRegType::get(sourcePartType.getContext(),
                                          sourcePartType.getElementCount(),
                                          u32ElemTy),
              sourcePart).getResult());
    }
    aliasPlan.resultTypes.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto resultVRegType = cast<VRegType>(resultType);
      aliasPlan.resultTypes.push_back(VRegType::get(
          resultVRegType.getContext(), resultVRegType.getElementCount(),
          u8ElemTy));
    }
    aliasPlan.requiresResultBitcast = true;
    return aliasPlan;
  }

  FailureOr<ArrayRef<StringRef>> getFactorTruncParts(
      VMITruncIOp op, int64_t factor, size_t sourcePartCount,
      size_t resultPartCount, OneToNPatternRewriter &rewriter) const {
    bool invalidFactorArity =
        (factor != 2 && factor != 4) ||
        sourcePartCount != resultPartCount * static_cast<size_t>(factor);
    if (invalidFactorArity) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci source/result arity relation");
    }
    if (factor == 2) {
      static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
      return ArrayRef<StringRef>(kEvenOddParts);
    }
    static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
    return ArrayRef<StringRef>(kPacked4Parts);
  }

  LogicalResult lowerNonGroupSlotTrunc(
      VMITruncIOp op, ValueRange sourceParts, SmallVector<Type> resultTypes,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<TruncIPhysicalPlan> physicalPlan = buildPhysicalPlan(
        op, sourceParts, resultTypes, sourceLayout, resultLayout, rewriter);
    if (failed(physicalPlan)) {
      return failure();
    }
    bool useNoSatCarrier =
        physicalPlan->denseLaneStride && physicalPlan->saturate &&
        physicalPlan->saturate.getValue() == "NOSAT";
    if (useNoSatCarrier) {
      return lowerNoSatDenseCarrier(op, sourceParts, resultTypes, rewriter);
    }
    TruncIAliasPlan aliasPlan = materializeS32ToS8Alias(
        op, sourceParts, resultTypes, *physicalPlan, rewriter);
    if (physicalPlan->denseLaneStride) {
      return lowerDenseLaneStrideTrunc(
          op, aliasPlan.sourceParts, aliasPlan.resultTypes,
          physicalPlan->factor, physicalPlan->saturate,
          aliasPlan.requiresResultBitcast, aliasPlan.originalResultTypes,
          rewriter);
    }
    FailureOr<ArrayRef<StringRef>> parts = getFactorTruncParts(
        op, physicalPlan->factor, aliasPlan.sourceParts.size(),
        aliasPlan.resultTypes.size(), rewriter);
    if (failed(parts)) {
      return failure();
    }
    return lowerFactorTrunc(
        op, aliasPlan.sourceParts, aliasPlan.resultTypes, *parts,
        physicalPlan->factor, physicalPlan->saturate,
        aliasPlan.requiresResultBitcast, aliasPlan.originalResultTypes,
        rewriter);
  }

public:

  LogicalResult
  matchAndRewrite(VMITruncIOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<VMIPhysicalConversionInput> input =
        getVMIPhysicalConversionInput(op, adaptor, *this->getTypeConverter());
    if (failed(input)) {
      return failure();
    }
    VMILayoutAttr sourceLayout = input->sourceVMIType.getLayoutAttr();
    VMILayoutAttr resultLayout = input->resultVMIType.getLayoutAttr();
    bool groupSlotLayouts =
        sourceLayout && resultLayout && sourceLayout.isGroupSlots() &&
        resultLayout.isGroupSlots();
    if (groupSlotLayouts) {
      return lowerGroupSlotTrunc(op, adaptor, rewriter, input->sourceVMIType,
                                 input->resultVMIType, sourceLayout,
                                 resultLayout, input->resultTypes);
    }

    return lowerNonGroupSlotTrunc(op, input->sourceParts,
                                  std::move(input->resultTypes), sourceLayout,
                                  resultLayout, rewriter);
  }
};

static LogicalResult lowerSameWidthFpToInt(
    Operation *op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes,
    StringAttr rnd, StringAttr sat, StringRef arityDiagnostic,
    StringRef maskDiagnostic, TypeConverter *typeConverter,
    OneToNPatternRewriter &rewriter) {
  bool invalidArity = sourceParts.size() != resultTypes.size();
  if (invalidArity) {
    return rewriter.notifyMatchFailure(op, arityDiagnostic);
  }
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (auto [sourcePart, resultType] :
       llvm::zip_equal(sourceParts, resultTypes)) {
    FailureOr<Value> mask = createAllTrueMaskForVReg(
        op->getLoc(), cast<VRegType>(sourcePart.getType()), rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(op, maskDiagnostic);
    }
    results.push_back(
        rewriter
            .create<VcvtOp>(op->getLoc(), resultType, sourcePart, *mask, rnd,
                            sat, /*part=*/nullptr)
            .getResult());
  }
  replaceOpWithFlatConvertedValues(rewriter, op, results, *typeConverter);
  return success();
}

static FailureOr<Value> buildNarrowFpToIntResult(
    Operation *op, ValueRange sourceParts, VRegType resultType,
    int64_t chunkIndex, int64_t sourceFactor, int64_t partStride,
    ArrayRef<StringRef> parts, StringAttr rnd, StringAttr sat,
    Value sourceMask, OneToNPatternRewriter &rewriter) {
  if (sourceFactor <= 0) {
    return rewriter.notifyMatchFailure(
        op, "narrow fp-to-int requires a positive source factor");
  }
  int64_t safeSourceFactor = sourceFactor;
  FailureOr<Value> resultMask =
      createAllTrueMaskForVReg(op->getLoc(), resultType, rewriter);
  if (failed(resultMask)) {
    return failure();
  }

  SmallVector<Value> partials;
  partials.reserve(safeSourceFactor);
  int64_t resultCount = sourceParts.size() / safeSourceFactor;
  for (int64_t partIndex = 0; partIndex < safeSourceFactor; ++partIndex) {
    Value sourcePart = sourceParts[partIndex * resultCount + chunkIndex];
    partials.push_back(
        rewriter
            .create<VcvtOp>(op->getLoc(), resultType, sourcePart, sourceMask,
                            rnd, sat, rewriter.getStringAttr(
                                          parts[partIndex * partStride]))
            .getResult());
  }

  Value merged = partials.front();
  for (Value partial : llvm::drop_begin(partials)) {
    merged = rewriter
                 .create<VorOp>(op->getLoc(), resultType, merged, partial,
                                *resultMask)
                 .getResult();
  }
  return merged;
}

static LogicalResult lowerNarrowFpToInt(
    Operation *op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes,
    int64_t sourceFactor, int64_t partStride, ArrayRef<StringRef> parts,
    StringAttr rnd, StringAttr sat,
    StringRef sourceMaskDiagnostic, StringRef resultMaskDiagnostic,
    TypeConverter *typeConverter, OneToNPatternRewriter &rewriter) {
  if (sourceFactor <= 0 || partStride <= 0 ||
      (sourceFactor - 1) * partStride >= static_cast<int64_t>(parts.size()) ||
      sourceParts.size() !=
                              static_cast<size_t>(sourceFactor) *
                                  resultTypes.size()) {
    return rewriter.notifyMatchFailure(
        op, "narrow fp-to-int source arity does not match conversion factor");
  }

  auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
  if (!sourceType) {
    return rewriter.notifyMatchFailure(op, "expected physical fp source type");
  }
  FailureOr<Value> sourceMask =
      createAllTrueMaskForVReg(op->getLoc(), sourceType, rewriter);
  if (failed(sourceMask)) {
    return rewriter.notifyMatchFailure(op, sourceMaskDiagnostic);
  }

  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (auto [chunkIndex, resultType] : llvm::enumerate(resultTypes)) {
    FailureOr<Value> result = buildNarrowFpToIntResult(
        op, sourceParts, resultType, chunkIndex, sourceFactor, partStride,
        parts, rnd, sat, *sourceMask, rewriter);
    if (failed(result)) {
      return rewriter.notifyMatchFailure(op, resultMaskDiagnostic);
    }
    results.push_back(*result);
  }

  replaceOpWithFlatConvertedValues(rewriter, op, results, *typeConverter);
  return success();
}

static int64_t getElementDeinterleaveFactor(VMILayoutAttr layout) {
  bool contiguous = layout && layout.isContiguous() && layout.getLaneStride() == 1;
  if (contiguous) {
    return 1;
  }
  bool deinterleaved =
      layout && layout.isDeinterleaved() && layout.getLaneStride() == 1;
  if (deinterleaved) {
    return layout.getFactor();
  }
  return 0;
}

static LogicalResult lowerWidenFpToInt(
    Operation *op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes,
    ArrayRef<StringRef> parts, StringAttr rnd, StringAttr sat,
    StringRef arityDiagnostic, StringRef maskDiagnostic,
    TypeConverter *typeConverter, OneToNPatternRewriter &rewriter) {
  bool invalidArity =
      parts.empty() || resultTypes.size() != parts.size() * sourceParts.size();
  if (invalidArity) {
    return rewriter.notifyMatchFailure(op, arityDiagnostic);
  }
  auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
  if (!sourceType) {
    return rewriter.notifyMatchFailure(op, maskDiagnostic);
  }
  FailureOr<Value> mask =
      createAllTrueMaskForVReg(op->getLoc(), sourceType, rewriter);
  if (failed(mask)) {
    return rewriter.notifyMatchFailure(op, maskDiagnostic);
  }
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (size_t partIndex = 0; partIndex < parts.size(); ++partIndex) {
    for (auto [chunkIndex, sourcePart] : llvm::enumerate(sourceParts)) {
      VRegType resultType =
          resultTypes[partIndex * sourceParts.size() + chunkIndex];
      results.push_back(
          rewriter
              .create<VcvtOp>(op->getLoc(), resultType, sourcePart, *mask, rnd,
                              sat, rewriter.getStringAttr(parts[partIndex]))
              .getResult());
    }
  }
  replaceOpWithFlatConvertedValues(rewriter, op, results, *typeConverter);
  return success();
}

template <typename OpTy>
static FailureOr<VRegType> validateFpToIntSourceParts(
    OpTy op, ValueRange sourceParts, StringRef emptyDiagnostic,
    StringRef expectedTypeDiagnostic, StringRef mismatchDiagnostic,
    OneToNPatternRewriter &rewriter) {
  if (sourceParts.empty()) {
    return rewriter.notifyMatchFailure(op, emptyDiagnostic);
  }
  auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
  if (!sourceType) {
    return rewriter.notifyMatchFailure(op, expectedTypeDiagnostic);
  }
  for (Value sourcePart : sourceParts) {
    auto currentType = dyn_cast<VRegType>(sourcePart.getType());
    bool mismatchedType = !currentType || currentType != sourceType;
    if (mismatchedType) {
      return rewriter.notifyMatchFailure(op, mismatchDiagnostic);
    }
  }
  return sourceType;
}

template <typename OpTy>
static FailureOr<SmallVector<VRegType>> validateFpToIntResultParts(
    OpTy op, TypeRange resultTypes, StringRef emptyDiagnostic,
    StringRef typeDiagnostic, OneToNPatternRewriter &rewriter) {
  if (resultTypes.empty()) {
    return rewriter.notifyMatchFailure(op, emptyDiagnostic);
  }
  SmallVector<VRegType> resultVRegTypes;
  resultVRegTypes.reserve(resultTypes.size());
  for (Type physicalResultType : resultTypes) {
    auto resultType = dyn_cast<VRegType>(physicalResultType);
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, typeDiagnostic);
    }
    resultVRegTypes.push_back(resultType);
  }
  return resultVRegTypes;
}

struct FpToIntPartValidation {
  VRegType sourceType;
  SmallVector<VRegType> resultVRegTypes;
};

template <typename OpTy>
static FailureOr<FpToIntPartValidation> validateFpToIntConversionParts(
    OpTy op, ValueRange sourceParts, TypeRange resultTypes,
    OneToNPatternRewriter &rewriter, StringRef emptySourceDiagnostic,
    StringRef sourceTypeDiagnostic, StringRef sourceMismatchDiagnostic,
    StringRef emptyResultDiagnostic, StringRef resultTypeDiagnostic) {
  FailureOr<VRegType> sourceType = validateFpToIntSourceParts(
      op, sourceParts, emptySourceDiagnostic, sourceTypeDiagnostic,
      sourceMismatchDiagnostic, rewriter);
  if (failed(sourceType)) {
    return failure();
  }
  FailureOr<SmallVector<VRegType>> resultVRegTypes = validateFpToIntResultParts(
      op, resultTypes, emptyResultDiagnostic, resultTypeDiagnostic, rewriter);
  if (failed(resultVRegTypes)) {
    return failure();
  }
  return FpToIntPartValidation{*sourceType, std::move(*resultVRegTypes)};
}

// Shared prechecks for narrow fp-to-int lowering: derives the source factor
// from the element bit widths, validates the physical result lane stride, and
// checks the source chunk arity against the expected result count.
struct NarrowFpToIntPlan {
  int64_t factor;
  int64_t resultLaneStride;
  int64_t sourceFactor;
};

template <typename OpTy>
static FailureOr<NarrowFpToIntPlan> buildNarrowFpToIntPlan(
    OpTy op, ValueRange sourceParts, TypeRange physicalResultTypes,
    VMIVRegType resultVMIType, unsigned sourceBits, unsigned resultBits,
    OneToNPatternRewriter &rewriter, StringRef positiveWidthDiagnostic,
    StringRef unsupportedLaneStrideDiagnostic,
    StringRef invalidSourceArityDiagnostic) {
  if (resultBits == 0) {
    return rewriter.notifyMatchFailure(op, positiveWidthDiagnostic);
  }
  int64_t factor = sourceBits / resultBits;
  VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
  int64_t resultLaneStride = resultLayout && resultLayout.isContiguous()
                                 ? resultLayout.getLaneStride()
                                 : 1;
  bool invalidResultLaneStride =
      resultLaneStride <= 0 || factor % resultLaneStride != 0;
  if (invalidResultLaneStride) {
    return rewriter.notifyMatchFailure(op, unsupportedLaneStrideDiagnostic);
  }
  int64_t sourceFactor = factor / resultLaneStride;
  bool invalidSourceArity =
      sourceParts.size() != sourceFactor * physicalResultTypes.size();
  if (invalidSourceArity) {
    return rewriter.notifyMatchFailure(op, invalidSourceArityDiagnostic);
  }
  return NarrowFpToIntPlan{factor, resultLaneStride, sourceFactor};
}

struct OneToNVMIFPToSIOpPattern : OneToNOpConversionPattern<VMIFPToSIOp> {
  using OneToNOpConversionPattern<VMIFPToSIOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerDenseWiden(
      VMIFPToSIOp op, ValueRange sourceParts,
      ArrayRef<VRegType> resultTypes, VRegType sourceType, unsigned sourceBits,
      StringAttr rnd, StringAttr sat, OneToNPatternRewriter &rewriter) const {
    StringRef part = sourceBits == 16 ? StringRef("EVEN") : StringRef("P0");
    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to build fptosi widen 1:1 mask");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      results.push_back(
          rewriter
              .create<VcvtOp>(op.getLoc(), resultType, sourcePart, *mask, rnd,
                              sat, rewriter.getStringAttr(part))
              .getResult());
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  LogicalResult lowerWiden(
      VMIFPToSIOp op, ValueRange sourceParts,
      ArrayRef<Type> physicalResultTypes, ArrayRef<VRegType> resultTypes,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      VRegType sourceType, unsigned sourceBits, StringAttr rnd, StringAttr sat,
      OneToNPatternRewriter &rewriter) const {
    VMILayoutAttr sourceLayout = sourceVMIType.getLayoutAttr();
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    bool denseOneToOne =
        sourceLayout && resultLayout && sourceLayout.isContiguous() &&
        resultLayout.isContiguous() && resultLayout.getLaneStride() == 1 &&
        ((sourceBits == 16 && sourceLayout.getLaneStride() == 2) ||
         (sourceBits == 8 && sourceLayout.getLaneStride() == 4)) &&
        physicalResultTypes.size() == sourceParts.size();
    if (denseOneToOne) {
      return lowerDenseWiden(op, sourceParts, resultTypes, sourceType,
                             sourceBits, rnd, sat, rewriter);
    }
    static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
    return lowerWidenFpToInt(
        op, sourceParts, resultTypes, kEvenOddParts, rnd, sat,
        "widen fptosi requires result arity = 2 × source arity",
        "failed to build fptosi widen mask", this->getTypeConverter(),
        rewriter);
  }

  LogicalResult lowerNarrow(
      VMIFPToSIOp op, ValueRange sourceParts,
      ArrayRef<Type> physicalResultTypes, ArrayRef<VRegType> resultTypes,
      VMIVRegType resultVMIType, unsigned sourceBits, unsigned resultBits,
      StringAttr rnd, StringAttr sat, OneToNPatternRewriter &rewriter) const {
    FailureOr<NarrowFpToIntPlan> plan = buildNarrowFpToIntPlan(
        op, sourceParts, physicalResultTypes, resultVMIType, sourceBits,
        resultBits, rewriter, "narrow fptosi requires positive result bit width",
        "narrow fptosi: unsupported result lane stride",
        "narrow fptosi: source arity != sourceFactor × result arity");
    if (failed(plan)) {
      return failure();
    }
    static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
    static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
    ArrayRef<StringRef> parts = plan->factor == 2
                                    ? ArrayRef<StringRef>(kEvenOddParts)
                                    : ArrayRef<StringRef>(kPacked4Parts);
    return lowerNarrowFpToInt(
        op, sourceParts, resultTypes, plan->sourceFactor,
        plan->resultLaneStride, parts, rnd, sat,
        "failed to build fptosi source mask",
        "failed to build narrow fptosi result mask", this->getTypeConverter(),
        rewriter);
  }

  LogicalResult lowerConversion(
      VMIFPToSIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      OneToNPatternRewriter &rewriter) const {
    Type sourceElementType = sourceVMIType.getElementType();
    Type resultElementType = resultVMIType.getElementType();
    auto contract =
        lookupVMIFpToSiContract(sourceElementType, resultElementType);
    if (!contract) {
      return rewriter.notifyMatchFailure(
          op, "unsupported fp-to-si conversion element type pair");
    }
    FailureOr<FpToIntPartValidation> validation =
        validateFpToIntConversionParts(
            op, sourceParts, resultTypes, rewriter,
            "fptosi requires at least one physical source chunk",
            "expected physical fptosi source type",
            "fptosi source physical parts must have matching type",
            "fptosi requires at least one physical result chunk",
            "unsupported physical fptosi result type");
    if (failed(validation)) {
      return failure();
    }

    StringAttr rnd = op->getAttrOfType<StringAttr>("rounding");
    if (!rnd) {
      rnd = rewriter.getStringAttr("R");
    }
    StringAttr sat = contract->requiresSat
                         ? op->getAttrOfType<StringAttr>("saturate")
                         : nullptr;
    if (!contract->requiresPart) {
      return lowerSameWidthFpToInt(
          op, sourceParts, validation->resultVRegTypes, rnd, sat,
          "same-width fptosi requires matching physical arity",
          "failed to build fptosi mask", this->getTypeConverter(), rewriter);
    }

    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(sourceElementType);
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(resultElementType);
    if (resultBits > sourceBits) {
      return lowerWiden(op, sourceParts, resultTypes,
                        validation->resultVRegTypes, sourceVMIType,
                        resultVMIType, validation->sourceType, sourceBits, rnd,
                        sat, rewriter);
    }
    return lowerNarrow(op, sourceParts, resultTypes,
                       validation->resultVRegTypes, resultVMIType, sourceBits,
                       resultBits, rnd, sat, rewriter);
  }

  LogicalResult lowerWithResultTypes(
      VMIFPToSIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    return lowerConversion(op, sourceParts, resultTypes, sourceVMIType,
                           resultVMIType, rewriter);
  }

public:

  LogicalResult
  matchAndRewrite(VMIFPToSIOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    return lowerWithConvertedResultTypes(
        op, 0, *this->getTypeConverter(), [&](ArrayRef<Type> resultTypes) {
          return lowerWithResultTypes(op, sourceParts, resultTypes, rewriter);
        });
  }
};

struct OneToNVMIFPToUIOpPattern
    : OneToNOpConversionPattern<VMIFPToUIOp> {
  using OneToNOpConversionPattern<VMIFPToUIOp>::OneToNOpConversionPattern;

private:
  struct FPToUILoweringInput {
    ValueRange sourceParts;
    ArrayRef<Type> resultTypes;
    VMIVRegType sourceVMIType;
    VMIVRegType resultVMIType;
    VRegType sourceType;
    SmallVector<VRegType> resultVRegTypes;
    VMIFpToUiContract contract;
    StringAttr rounding;
    StringAttr saturate;
  };

  LogicalResult lowerWiden(
      VMIFPToUIOp op, ValueRange sourceParts, ArrayRef<Type> physicalResultTypes,
      ArrayRef<VRegType> resultTypes, StringAttr rnd, StringAttr sat,
      OneToNPatternRewriter &rewriter) const {
    bool invalidWidenArity =
        physicalResultTypes.size() != 2 * sourceParts.size();
    if (invalidWidenArity) {
      return rewriter.notifyMatchFailure(
          op, "widen fptoui requires result arity = 2 × source arity");
    }
    static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
    return lowerWidenFpToInt(
        op, sourceParts, resultTypes, kEvenOddParts, rnd, sat,
        "widen fptoui requires result arity = 2 × source arity",
        "failed to build fptoui widen mask", this->getTypeConverter(),
        rewriter);
  }

  LogicalResult lowerNarrow(
      VMIFPToUIOp op, ValueRange sourceParts, ArrayRef<Type> physicalResultTypes,
      ArrayRef<VRegType> resultTypes, VMIVRegType resultVMIType,
      unsigned sourceBits, unsigned resultBits, StringAttr rnd, StringAttr sat,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<NarrowFpToIntPlan> plan = buildNarrowFpToIntPlan(
        op, sourceParts, physicalResultTypes, resultVMIType, sourceBits,
        resultBits, rewriter, "narrow fptoui requires positive result bit width",
        "narrow fptoui: unsupported result lane stride",
        "narrow fptoui: source arity != sourceFactor × result arity");
    if (failed(plan)) {
      return failure();
    }
    static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
    return lowerNarrowFpToInt(
        op, sourceParts, resultTypes, plan->sourceFactor,
        plan->resultLaneStride, kEvenOddParts, rnd, sat,
        "failed to build fptoui source mask",
        "failed to build narrow fptoui result mask", this->getTypeConverter(),
        rewriter);
  }

  FailureOr<FPToUILoweringInput> getLoweringInput(
      VMIFPToUIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      OneToNPatternRewriter &rewriter) const {
    auto contract = lookupVMIFpToUIContract(sourceVMIType.getElementType(),
                                             resultVMIType.getElementType());
    if (!contract) {
      return rewriter.notifyMatchFailure(
          op, "unsupported fp-to-ui conversion element type pair");
    }
    FailureOr<FpToIntPartValidation> validation =
        validateFpToIntConversionParts(
            op, sourceParts, resultTypes, rewriter,
            "fptoui requires at least one physical source chunk",
            "expected physical fptoui source type",
            "fptoui source physical parts must have matching type",
            "fptoui requires at least one physical result chunk",
            "unsupported physical fptoui result type");
    if (failed(validation)) {
      return failure();
    }
    StringAttr rounding = op->getAttrOfType<StringAttr>("rounding");
    if (!rounding) {
      rounding = rewriter.getStringAttr("R");
    }
    StringAttr saturate = contract->requiresSat
                              ? op->getAttrOfType<StringAttr>("saturate")
                              : nullptr;
    return FPToUILoweringInput{sourceParts, resultTypes, sourceVMIType,
                               resultVMIType, validation->sourceType,
                               std::move(validation->resultVRegTypes), *contract,
                               rounding, saturate};
  }

  LogicalResult lowerConversion(
      VMIFPToUIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<FPToUILoweringInput> input = getLoweringInput(
        op, sourceParts, resultTypes, sourceVMIType, resultVMIType, rewriter);
    if (failed(input)) {
      return failure();
    }
    if (!input->contract.requiresPart) {
      return lowerSameWidthFpToInt(
          op, input->sourceParts, input->resultVRegTypes, input->rounding,
          input->saturate,
          "same-width fptoui requires matching physical arity",
          "failed to build fptoui mask", this->getTypeConverter(), rewriter);
    }
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(input->sourceVMIType.getElementType());
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(input->resultVMIType.getElementType());
    if (resultBits > sourceBits) {
      return lowerWiden(op, input->sourceParts, input->resultTypes,
                        input->resultVRegTypes, input->rounding,
                        input->saturate, rewriter);
    }
    if (resultBits < sourceBits) {
      return lowerNarrow(op, input->sourceParts, input->resultTypes,
                         input->resultVRegTypes, input->resultVMIType,
                         sourceBits, resultBits, input->rounding,
                         input->saturate, rewriter);
    }
    return failure();
  }

  LogicalResult lowerWithResultTypes(
      VMIFPToUIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    return lowerConversion(op, sourceParts, resultTypes, sourceVMIType,
                           resultVMIType, rewriter);
  }

  LogicalResult lowerConversionWithResolvedTypes(
      VMIFPToUIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    return lowerWithResultTypes(op, sourceParts, resultTypes, rewriter);
  }

public:

  LogicalResult
  matchAndRewrite(VMIFPToUIOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    return lowerWithConvertedResultTypes(
        op, 0, *this->getTypeConverter(), [&](ArrayRef<Type> resultTypes) {
          return lowerConversionWithResolvedTypes(op, sourceParts, resultTypes,
                                                  rewriter);
        });
  }
};

struct OneToNVMISIToFPOpPattern : OneToNOpConversionPattern<VMISIToFPOp> {
  using OneToNOpConversionPattern<VMISIToFPOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerSameWidth(
      VMISIToFPOp op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes,
      Value mask, OneToNPatternRewriter &rewriter) const {
    bool invalidArity = sourceParts.size() != resultTypes.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "si32->f32 requires matching physical arity");
    }
    StringAttr rnd = rewriter.getStringAttr("R");
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      results.push_back(
          rewriter
              .create<VcvtOp>(op.getLoc(), resultType, sourcePart, mask, rnd,
                              nullptr, nullptr)
              .getResult());
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  LogicalResult lowerWiden(
      VMISIToFPOp op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes,
      Value mask, OneToNPatternRewriter &rewriter) const {
    bool invalidArity = resultTypes.size() != 2 * sourceParts.size();
    if (invalidArity) {
      return rewriter.notifyMatchFailure(
          op, "si8->f16 requires result arity = 2 x source arity");
    }
    static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (int64_t partIndex = 0; partIndex < 2; ++partIndex) {
      for (auto [chunkIndex, sourcePart] :
           llvm::enumerate(sourceParts)) {
        VRegType resultType =
            resultTypes[partIndex * sourceParts.size() + chunkIndex];
        results.push_back(
            rewriter
                .create<VcvtOp>(op.getLoc(), resultType, sourcePart, mask,
                                nullptr, nullptr,
                                rewriter.getStringAttr(kEvenOddParts[partIndex]))
                .getResult());
      }
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  LogicalResult lowerConversion(
      VMISIToFPOp op, ValueRange sourceParts,
      ArrayRef<VRegType> resultTypes, Value mask, unsigned sourceBits,
      unsigned resultBits, OneToNPatternRewriter &rewriter) const {
    if (sourceBits == 32 && resultBits == 32) {
      return lowerSameWidth(op, sourceParts, resultTypes, mask, rewriter);
    } else if (sourceBits == 8 && resultBits == 16) {
      return lowerWiden(op, sourceParts, resultTypes, mask, rewriter);
    } else {
      return rewriter.notifyMatchFailure(
          op, "unsupported sitofp source/result width relation");
    }
  }

  FailureOr<SmallVector<VRegType>> collectResultTypes(
      VMISIToFPOp op, TypeRange resultTypes,
      OneToNPatternRewriter &rewriter) const {
    if (resultTypes.empty()) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical sitofp result type");
    }
    SmallVector<VRegType> resultVRegTypes;
    resultVRegTypes.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      bool mismatchedType =
          !resultVRegType ||
          (!resultVRegTypes.empty() &&
           resultVRegType != resultVRegTypes.front());
      if (mismatchedType) {
        (void)rewriter.notifyMatchFailure(
            op, "unsupported physical sitofp result type");
        return failure();
      }
      resultVRegTypes.push_back(resultVRegType);
    }
    return resultVRegTypes;
  }

  FailureOr<VRegType> collectSourceType(
      VMISIToFPOp op, ValueRange sourceParts,
      OneToNPatternRewriter &rewriter) const {
    if (sourceParts.empty()) {
      return rewriter.notifyMatchFailure(op,
                                         "sitofp requires integer source chunks");
    }
    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    bool invalidSourceType =
        !sourceType || !isa<IntegerType>(sourceType.getElementType());
    if (invalidSourceType) {
      return rewriter.notifyMatchFailure(op,
                                         "sitofp requires integer source chunks");
    }
    for (Value sourcePart : sourceParts) {
      bool mismatchedSourceType = sourcePart.getType() != sourceType;
      if (mismatchedSourceType) {
        return rewriter.notifyMatchFailure(
            op, "sitofp requires integer source chunks");
      }
    }
    return sourceType;
  }

  LogicalResult lowerPhysicalConversion(
      VMISIToFPOp op, ValueRange sourceParts, TypeRange resultTypes,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<VRegType> sourceType =
        collectSourceType(op, sourceParts, rewriter);
    if (failed(sourceType)) {
      return failure();
    }
    FailureOr<SmallVector<VRegType>> resultVRegTypes =
        collectResultTypes(op, resultTypes, rewriter);
    if (failed(resultVRegTypes)) {
      return failure();
    }
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(sourceType->getElementType());
    unsigned resultBits = pto::getPTOStorageElemBitWidth(
        resultVRegTypes->front().getElementType());
    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), *sourceType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(op, "failed to build sitofp mask");
    }
    return lowerConversion(op, sourceParts, *resultVRegTypes, *mask,
                           sourceBits, resultBits, rewriter);
  }

public:

  LogicalResult
  matchAndRewrite(VMISIToFPOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);

    return lowerPhysicalConversion(op, sourceParts, resultTypes, rewriter);
  }
};

struct OneToNVMIBitcastOpPattern : OneToNOpConversionPattern<VMIBitcastOp> {
  using OneToNOpConversionPattern<VMIBitcastOp>::OneToNOpConversionPattern;

private:
  FailureOr<Value> buildBitcastPart(
      VMIBitcastOp op, Value sourcePart, Type resultType,
      OneToNPatternRewriter &rewriter) const {
    bool invalidPartTypes = !isa<VRegType>(sourcePart.getType()) ||
                            !isa<VRegType>(resultType);
    if (invalidPartTypes) {
      return rewriter.notifyMatchFailure(
          op, "physical bitcast part type mismatch");
    }
    return rewriter.create<VbitcastOp>(op.getLoc(), resultType, sourcePart)
        .getResult();
  }

  LogicalResult lowerParts(VMIBitcastOp op, ValueRange sourceParts,
                           ArrayRef<Type> resultTypes,
                           OneToNPatternRewriter &rewriter) const {
    bool arityMismatch = sourceParts.size() != resultTypes.size();
    if (arityMismatch) {
      return rewriter.notifyMatchFailure(op, "physical bitcast arity mismatch");
    }
    return lowerPointwisePhysicalParts(
        op, resultTypes, "physical bitcast arity mismatch", rewriter,
        [&](int64_t index, Type resultType) -> FailureOr<Value> {
          return buildBitcastPart(op, sourceParts[index], resultType, rewriter);
        },
        *this->getTypeConverter());
  }

public:

  LogicalResult
  matchAndRewrite(VMIBitcastOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    return lowerParts(op, sourceParts, resultTypes, rewriter);
  }
};

struct OneToNVMIChannelSplitOpPattern
    : OneToNOpConversionPattern<VMIChannelSplitOp> {
  using OneToNOpConversionPattern<VMIChannelSplitOp>::OneToNOpConversionPattern;

private:
  LogicalResult validateResultLayouts(
      VMIChannelSplitOp op, OneToNPatternRewriter &rewriter) const {
    return validateContiguousParts(
        op, op.getResults(), "channel_split requires contiguous result layouts",
        rewriter, isContiguousVMIVRegPart);
  }

public:

  LogicalResult
  matchAndRewrite(VMIChannelSplitOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    int64_t channels = op.getNumResults();
    bool unsupportedChannels = channels != 2 && channels != 4;
    if (unsupportedChannels) {
      return rewriter.notifyMatchFailure(
          op, "channel_split only supports 2 or 4 channels");
    }

    auto sourceType = cast<VMIVRegType>(op.getSource().getType());
    VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
    VMILayoutAttr channelLayout =
        VMILayoutAttr::getDeinterleaved(rewriter.getContext(), channels);
    bool invalidSourceLayout =
        !sourceLayout ||
        (!sourceLayout.isContiguous() && sourceLayout != channelLayout);
    if (invalidSourceLayout) {
      return rewriter.notifyMatchFailure(
          op,
          "channel_split requires contiguous or matching deinterleaved source "
          "layout");
    }
    if (failed(validateResultLayouts(op, rewriter))) {
      return failure();
    }

    FailureOr<SmallVector<Type>> maybe_resultTypes =
        getConvertedResultTypes(op, *this->getTypeConverter());
    if (failed(maybe_resultTypes)) {
      return failure();
    }
    SmallVector<Type> resultTypes = std::move(*maybe_resultTypes);
    FailureOr<SmallVector<Value>> results =
        materializeDataLayoutConversion(op, adaptor.getSource(), resultTypes,
                                        sourceLayout, channelLayout,
                                        sourceType.getElementType(), rewriter);
    if (failed(results)) {
      return failure();
    }

    replaceOpWithFlatConvertedValues(rewriter, op, *results, *this->getTypeConverter());
    return success();
  }
};

struct OneToNVMIChannelMergeOpPattern
    : OneToNOpConversionPattern<VMIChannelMergeOp> {
  using OneToNOpConversionPattern<VMIChannelMergeOp>::OneToNOpConversionPattern;

private:
  LogicalResult validateInputLayouts(
      VMIChannelMergeOp op, OneToNPatternRewriter &rewriter) const {
    return validateContiguousParts(
        op, op.getInputs(), "channel_merge requires contiguous input layouts",
        rewriter, isContiguousVMIVRegPart);
  }

  LogicalResult validateResultLayout(
      VMIChannelMergeOp op, VMILayoutAttr resultLayout,
      VMILayoutAttr channelLayout,
      OneToNPatternRewriter &rewriter) const {
    bool invalidResultLayout =
        !resultLayout ||
        (!resultLayout.isContiguous() && resultLayout != channelLayout);
    if (invalidResultLayout) {
      return rewriter.notifyMatchFailure(
          op,
          "channel_merge requires contiguous or matching deinterleaved result "
          "layout");
    }
    return success();
  }

  LogicalResult lowerChannelMerge(
      VMIChannelMergeOp op, OpAdaptor adaptor, VMILayoutAttr channelLayout,
      VMILayoutAttr resultLayout, Type resultElementType,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<SmallVector<Type>> maybeResultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(maybeResultTypes)) {
      return failure();
    }
    FailureOr<SmallVector<Value>> results = materializeDataLayoutConversion(
        op, flattenOneToNOperands(adaptor.getOperands()), *maybeResultTypes,
        channelLayout, resultLayout, resultElementType, rewriter);
    if (failed(results)) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, *results,
                                     *this->getTypeConverter());
    return success();
  }

public:

  LogicalResult
  matchAndRewrite(VMIChannelMergeOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    int64_t channels = op.getInputs().size();
    bool unsupportedChannels = channels != 2 && channels != 4;
    if (unsupportedChannels) {
      return rewriter.notifyMatchFailure(
          op, "channel_merge only supports 2 or 4 channels");
    }

    if (failed(validateInputLayouts(op, rewriter))) {
      return failure();
    }
    auto resultType = cast<VMIVRegType>(op.getResult().getType());
    VMILayoutAttr resultLayout = resultType.getLayoutAttr();
    auto channelLayout =
        VMILayoutAttr::getDeinterleaved(rewriter.getContext(), channels);
    if (failed(validateResultLayout(op, resultLayout, channelLayout, rewriter))) {
      return failure();
    }

    return lowerChannelMerge(op, adaptor, channelLayout, resultLayout,
                             resultType.getElementType(), rewriter);
  }
};

struct OneToNVMIShuffleOpPattern : OneToNOpConversionPattern<VMIShuffleOp> {
  using OneToNOpConversionPattern<VMIShuffleOp>::OneToNOpConversionPattern;

private:
  LogicalResult lowerForwarding(
      VMIShuffleOp op, OneToNPatternRewriter &rewriter, ValueRange sourceParts,
      ArrayRef<Type> resultTypes, ArrayRef<int64_t> sourceIndices) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (int64_t sourceIndex : sourceIndices) {
      bool sourceOutOfBounds =
          sourceIndex < 0 || sourceIndex >= static_cast<int64_t>(sourceParts.size());
      if (sourceOutOfBounds) {
        return rewriter.notifyMatchFailure(
            op, "shuffle forwarding source part range is out of bounds");
      }
      results.push_back(sourceParts[sourceIndex]);
    }
    if (failed(verifyIdentityPartForwarding(op, results, resultTypes, rewriter))) {
      return failure();
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  LogicalResult lowerLane0Splat(
      VMIShuffleOp op, OneToNPatternRewriter &rewriter, ValueRange sourceParts,
      ArrayRef<Type> resultTypes, int64_t sourceIndex) const {
    bool sourceOutOfBounds =
        sourceIndex < 0 || sourceIndex >= static_cast<int64_t>(sourceParts.size());
    if (sourceOutOfBounds) {
      return rewriter.notifyMatchFailure(
          op, "shuffle lane0 splat source part range is out of bounds");
    }
    Value sourcePart = sourceParts[sourceIndex];
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto sourceVRegType = dyn_cast<VRegType>(sourcePart.getType());
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      bool invalidTypes = !sourceVRegType || !resultVRegType ||
                          sourceVRegType != resultVRegType;
      if (invalidTypes) {
        return rewriter.notifyMatchFailure(
            op, "shuffle lane0 splat requires matching physical vreg type");
      }
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), resultVRegType, rewriter);
      if (failed(mask)) {
        return rewriter.notifyMatchFailure(
            op, "failed to create shuffle lane0 splat mask");
      }
      results.push_back(rewriter
                           .create<VdupOp>(op.getLoc(), resultType, sourcePart,
                                           *mask,
                                           rewriter.getStringAttr("LOWEST"))
                           .getResult());
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildShuffleVselrResult(
      VMIShuffleOp op, ValueRange sourceParts, Type resultType,
      const ShuffleVselrPlan &plan,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<VRegType> sourceVRegType = getShuffleVselrSourceType(
        op, sourceParts, resultType, plan.sourceFlatIndex, rewriter);
    if (failed(sourceVRegType)) {
      return failure();
    }
    unsigned indexBits =
        pto::getPTOStorageElemBitWidth(sourceVRegType->getElementType());
    bool unsupportedIndexBits =
        indexBits != 8 && indexBits != 16 && indexBits != 32;
    if (unsupportedIndexBits) {
      return rewriter.notifyMatchFailure(
          op, "shuffle vselr requires 8/16/32-bit index elements");
    }
    auto indexElementType = IntegerType::get(rewriter.getContext(), indexBits);
    Type indexType = VRegType::get(rewriter.getContext(),
                                   sourceVRegType->getElementCount(),
                                   indexElementType);
    FailureOr<Value> base = createScalarOffsetConstant(
        op.getLoc(), indexElementType, plan.baseLane, rewriter);
    if (failed(base)) {
      return rewriter.notifyMatchFailure(
          op, "failed to materialize shuffle vselr index base");
    }
    StringAttr orderAttr =
        plan.descending ? rewriter.getStringAttr("DESC") : StringAttr{};
    Value indexVector =
        rewriter.create<VciOp>(op.getLoc(), indexType, *base, orderAttr)
            .getResult();
    return rewriter
        .create<VselrOp>(op.getLoc(), resultType,
                         sourceParts[plan.sourceFlatIndex], indexVector)
        .getResult();
  }

  FailureOr<VRegType> getShuffleVselrSourceType(
      VMIShuffleOp op, ValueRange sourceParts, Type resultType,
      int64_t sourceIndex, OneToNPatternRewriter &rewriter) const {
    bool sourceOutOfBounds =
        sourceIndex < 0 || sourceIndex >= static_cast<int64_t>(sourceParts.size());
    if (sourceOutOfBounds) {
      return rewriter.notifyMatchFailure(
          op, "shuffle vselr source part range is out of bounds");
    }
    auto sourceVRegType =
        dyn_cast<VRegType>(sourceParts[sourceIndex].getType());
    auto resultVRegType = dyn_cast<VRegType>(resultType);
    bool invalidTypes =
        !sourceVRegType || !resultVRegType ||
        sourceVRegType.getElementCount() != resultVRegType.getElementCount() ||
        sourceVRegType.getElementType() != resultVRegType.getElementType();
    if (invalidTypes) {
      return rewriter.notifyMatchFailure(
          op, "shuffle vselr source/result type mismatch");
    }
    return sourceVRegType;
  }

  LogicalResult lowerVselr(
      VMIShuffleOp op, OneToNPatternRewriter &rewriter, ValueRange sourceParts,
      ArrayRef<Type> resultTypes, ArrayRef<ShuffleVselrPlan> plans) const {
    bool arityMismatch = plans.size() != resultTypes.size();
    if (arityMismatch) {
      return rewriter.notifyMatchFailure(op, "shuffle vselr arity mismatch");
    }
    return lowerPointwisePhysicalParts(
        op, resultTypes, "shuffle vselr arity mismatch", rewriter,
        [&](int64_t index, Type resultType) -> FailureOr<Value> {
          return buildShuffleVselrResult(op, sourceParts, resultType,
                                         plans[index], rewriter);
        },
        *this->getTypeConverter());
  }

public:

  LogicalResult
  matchAndRewrite(VMIShuffleOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    return lowerWithConvertedResultTypes(
        op, 0, *this->getTypeConverter(), [&](ArrayRef<Type> resultTypes) {
          std::string splatReason;
          FailureOr<int64_t> splatSource =
              computeShuffleLane0SplatSourcePart(op, &splatReason);
          if (succeeded(splatSource)) {
            return lowerLane0Splat(op, rewriter, sourceParts, resultTypes,
                                   *splatSource);
          }

          std::string reason;
          FailureOr<SmallVector<int64_t>> sourceFlatIndices =
              computeShuffleForwardingSourceParts(op, &reason);
          if (succeeded(sourceFlatIndices)) {
            return lowerForwarding(op, rewriter, sourceParts, resultTypes,
                                   *sourceFlatIndices);
          }

          std::string vselrReason;
          FailureOr<SmallVector<ShuffleVselrPlan>> vselrPlans =
              computeShuffleVselrPlans(op, &vselrReason);
          if (failed(vselrPlans)) {
            return rewriter.notifyMatchFailure(
                op, Twine("shuffle vselr ") + vselrReason);
          }

          return lowerVselr(op, rewriter, sourceParts, resultTypes,
                            *vselrPlans);
        });
  }
};

Block *convertBranchDestBlock(Block *block, OneToNPatternRewriter &rewriter,
                              OneToNTypeConverter &typeConverter,
                              llvm::DenseMap<Block *, Block *> &converted) {
  auto [it, inserted] = converted.try_emplace(block, nullptr);
  if (!inserted) {
    return it->second;
  }

  OneToNTypeMapping argMapping(block->getArgumentTypes());
  if (failed(typeConverter.computeTypeMapping(block->getArgumentTypes(),
                                              argMapping)) ||
      !argMapping.hasNonIdentityConversion()) {
    it->second = block;
    return block;
  }

  Block *newBlock = rewriter.applySignatureConversion(block, argMapping);
  it->second = newBlock;
  return newBlock;
}

struct OneToNCFBranchOpPattern : OneToNOpConversionPattern<cf::BranchOp> {
  using OneToNOpConversionPattern<cf::BranchOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter<OneToNTypeConverter>();
    llvm::DenseMap<Block *, Block *> convertedBlocks;
    Block *dest = convertBranchDestBlock(op.getDest(), rewriter, *converter,
                                         convertedBlocks);
    if (!adaptor.getOperandMapping().hasNonIdentityConversion() &&
        dest == op.getDest()) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, dest,
                                              adaptor.getFlatOperands());
    return success();
  }
};

struct OneToNCFCondBranchOpPattern
    : OneToNOpConversionPattern<cf::CondBranchOp> {
  using OneToNOpConversionPattern<cf::CondBranchOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter<OneToNTypeConverter>();
    llvm::DenseMap<Block *, Block *> convertedBlocks;
    Block *trueDest = convertBranchDestBlock(op.getTrueDest(), rewriter,
                                             *converter, convertedBlocks);
    Block *falseDest = convertBranchDestBlock(op.getFalseDest(), rewriter,
                                              *converter, convertedBlocks);

    if (!adaptor.getOperandMapping().hasNonIdentityConversion() &&
        trueDest == op.getTrueDest() && falseDest == op.getFalseDest()) {
      return failure();
    }

    ValueRange condition = adaptor.getCondition();
    bool conditionArityMismatch = condition.size() != 1;
    if (conditionArityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "condition converted to multiple values");
    }

    SmallVector<Value> trueOperands;
    SmallVector<Value> falseOperands;
    ValueRange flatOperands = adaptor.getFlatOperands();
    const OneToNTypeMapping &operandMapping = adaptor.getOperandMapping();
    unsigned operandIndex = 1;
    for (unsigned i = 0, e = op.getNumTrueOperands(); i < e; ++i) {
      llvm::append_range(trueOperands, operandMapping.getConvertedValues(
                                           flatOperands, operandIndex++));
    }
    for (unsigned i = 0, e = op.getNumFalseOperands(); i < e; ++i) {
      llvm::append_range(falseOperands, operandMapping.getConvertedValues(
                                            flatOperands, operandIndex++));
    }

    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(op, condition.front(),
                                                  trueDest, trueOperands,
                                                  falseDest, falseOperands);
    return success();
  }
};

struct OneToNCFSwitchOpPattern : OneToNOpConversionPattern<cf::SwitchOp> {
  using OneToNOpConversionPattern<cf::SwitchOp>::OneToNOpConversionPattern;

private:
  static void collectSwitchOperandSegments(
      ArrayRef<int32_t> segmentSizes, ValueRange flatOperands,
      const OneToNTypeMapping &operandMapping, unsigned &operandIndex,
      SmallVectorImpl<SmallVector<Value>> &storage,
      SmallVectorImpl<ValueRange> &segments) {
    storage.reserve(segmentSizes.size());
    segments.reserve(segmentSizes.size());
    for (int32_t segmentSize : segmentSizes) {
      SmallVector<Value> operands;
      for (int32_t index = 0; index < segmentSize; ++index) {
        llvm::append_range(operands, operandMapping.getConvertedValues(
                                         flatOperands, operandIndex++));
      }
      storage.push_back(std::move(operands));
    }
    for (SmallVector<Value> &operands : storage) {
      segments.push_back(operands);
    }
  }

  static Block *convertSwitchDestination(
      Block *destination, OneToNPatternRewriter &rewriter,
      OneToNTypeConverter &converter, llvm::DenseMap<Block *, Block *> &blocks) {
    return convertBranchDestBlock(destination, rewriter, converter, blocks);
  }

  static bool switchDestinationsChanged(
      cf::SwitchOp op, Block *defaultDest, ArrayRef<Block *> caseDests) {
    if (defaultDest != op.getDefaultDestination()) {
      return true;
    }
    for (auto [oldDest, newDest] :
         llvm::zip(op.getCaseDestinations(), caseDests)) {
      if (oldDest != newDest) {
        return true;
      }
    }
    return false;
  }

public:
  LogicalResult
  matchAndRewrite(cf::SwitchOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter<OneToNTypeConverter>();
    llvm::DenseMap<Block *, Block *> convertedBlocks;
    Block *defaultDest = convertSwitchDestination(
        op.getDefaultDestination(), rewriter, *converter, convertedBlocks);

    SmallVector<Block *> caseDests;
    caseDests.reserve(op.getCaseDestinations().size());
    for (Block *dest : op.getCaseDestinations()) {
      caseDests.push_back(convertSwitchDestination(dest, rewriter, *converter,
                                                   convertedBlocks));
    }

    bool changed = switchDestinationsChanged(op, defaultDest, caseDests) ||
                   adaptor.getOperandMapping().hasNonIdentityConversion();
    if (!changed) {
      return failure();
    }

    ValueRange flag = adaptor.getFlag();
    bool flagArityMismatch = flag.size() != 1;
    if (flagArityMismatch) {
      return rewriter.notifyMatchFailure(op,
                                         "flag converted to multiple values");
    }

    SmallVector<Value> defaultOperands;
    SmallVector<SmallVector<Value>> caseOperandStorage;
    SmallVector<ValueRange> caseOperands;
    ValueRange flatOperands = adaptor.getFlatOperands();
    const OneToNTypeMapping &operandMapping = adaptor.getOperandMapping();
    unsigned operandIndex = 1;
    for (unsigned i = 0, e = op.getDefaultOperands().size(); i < e; ++i) {
      llvm::append_range(defaultOperands, operandMapping.getConvertedValues(
                                              flatOperands, operandIndex++));
    }

    collectSwitchOperandSegments(op.getCaseOperandSegments(), flatOperands,
                                 operandMapping, operandIndex,
                                 caseOperandStorage, caseOperands);

    rewriter.replaceOpWithNewOp<cf::SwitchOp>(
        op, flag.front(), defaultDest, defaultOperands, op.getCaseValuesAttr(),
        caseDests, caseOperands);
    return success();
  }
};

struct OneToNSCFExecuteRegionOpPattern
    : OneToNOpConversionPattern<scf::ExecuteRegionOp> {
  using OneToNOpConversionPattern<
      scf::ExecuteRegionOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ExecuteRegionOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    const OneToNTypeMapping &resultMapping = adaptor.getResultMapping();
    for (unsigned i = 0, e = op->getNumResults(); i < e; ++i) {
      llvm::append_range(resultTypes, resultMapping.getConvertedTypes(i));
    }
    if (resultTypes == op->getResultTypes()) {
      return failure();
    }

    auto newOp =
        rewriter.create<scf::ExecuteRegionOp>(op.getLoc(), resultTypes);
    newOp->setAttrs(op->getAttrs());
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());
    rewriter.replaceOp(op, newOp->getResults(), resultMapping);
    return success();
  }
};

struct OneToNSCFIndexSwitchOpPattern
    : OneToNOpConversionPattern<scf::IndexSwitchOp> {
  using OneToNOpConversionPattern<
      scf::IndexSwitchOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IndexSwitchOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange arg = adaptor.getArg();
    bool selectorArityMismatch = arg.size() != 1;
    if (selectorArityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "index_switch selector converted to multiple values");
    }

    SmallVector<Type> resultTypes;
    const OneToNTypeMapping &resultMapping = adaptor.getResultMapping();
    for (unsigned i = 0, e = op->getNumResults(); i < e; ++i) {
      llvm::append_range(resultTypes, resultMapping.getConvertedTypes(i));
    }
    if (resultTypes == op->getResultTypes()) {
      return failure();
    }

    auto newOp = rewriter.create<scf::IndexSwitchOp>(
        op.getLoc(), resultTypes, arg.front(), op.getCases(), op.getNumCases());
    newOp->setAttrs(op->getAttrs());
    rewriter.inlineRegionBefore(op.getDefaultRegion(), newOp.getDefaultRegion(),
                                newOp.getDefaultRegion().end());
    for (auto [srcRegion, dstRegion] :
         llvm::zip(op.getCaseRegions(), newOp.getCaseRegions())) {
      rewriter.inlineRegionBefore(srcRegion, dstRegion, dstRegion.end());
    }
    rewriter.replaceOp(op, newOp->getResults(), resultMapping);
    return success();
  }
};

static void populateVMIStructuralAndMemoryPatterns(
    VMIToVPTOTypeConverter &typeConverter, RewritePatternSet &patterns) {
  populateFuncTypeConversionPatterns(typeConverter, patterns);
  scf::populateSCFStructuralOneToNTypeConversions(typeConverter, patterns);
  patterns.add<OneToNCFBranchOpPattern, OneToNCFCondBranchOpPattern,
               OneToNCFSwitchOpPattern>(typeConverter, patterns.getContext());
  patterns.add<OneToNSCFExecuteRegionOpPattern, OneToNSCFIndexSwitchOpPattern>(
      typeConverter, patterns.getContext());
  patterns.add<OneToNVMIPackOpPattern, OneToNVMIUnpackOpPattern>(
      typeConverter, patterns.getContext());
  patterns.add<
      OneToNVMIEnsureLayoutOpPattern, OneToNVMIEnsureMaskLayoutOpPattern,
      OneToNVMIBroadcastOpPattern, OneToNVMIIotaOpPattern<VMIIotaOp>,
      OneToNVMIIotaOpPattern<VMIGroupIotaOp>,
      OneToNVMIConstantOpPattern, OneToNVMIConstantMaskOpPattern,
      OneToNVMICreateMaskOpPattern, OneToNVMICreateGroupMaskOpPattern,
      OneToNVMIBinaryOpPattern<VMIMaskAndOp, PandOp, /*IsMaskResult=*/true>,
      OneToNVMIBinaryOpPattern<VMIMaskOrOp, PorOp, /*IsMaskResult=*/true>,
      OneToNVMIBinaryOpPattern<VMIMaskXOrOp, PxorOp, /*IsMaskResult=*/true>,
      OneToNVMIUnaryOpPattern<VMIMaskNotOp, PnotOp, /*IsMaskResult=*/true>,
      OneToNVMILoadOpPattern,
      OneToNVMIDeinterleaveLoadOpPattern, OneToNVMIGroupLoadOpPattern,
      OneToNVMIGroupSlotLoadOpPattern, OneToNVMIStrideLoadOpPattern,
      OneToNVMIMaskedLoadOpPattern, OneToNVMIGatherOpPattern,
      OneToNVMIExpandLoadOpPattern, OneToNVMIStoreOpPattern,
      OneToNVMIInterleaveStoreOpPattern, OneToNVMIGroupStoreOpPattern,
      OneToNVMIMaskedStoreOpPattern, OneToNVMIStrideStoreOpPattern,
      OneToNVMIScatterOpPattern>(typeConverter, patterns.getContext());
}

static void populateVMIArithmeticPatterns(
    VMIToVPTOTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<OneToNVMIBinaryOpPattern<VMIAddFOp, VaddOp>,
      OneToNVMIBinaryOpPattern<VMIAddIOp, VaddOp>,
      OneToNVMIVaddcOpPattern, OneToNVMIVaddcsOpPattern,
      OneToNVMIBinaryOpPattern<VMISubFOp, VsubOp>,
      OneToNVMIBinaryOpPattern<VMISubIOp, VsubOp>,
      OneToNVMIBinaryOpPattern<VMIMulFOp, VmulOp>,
      OneToNVMIBinaryOpPattern<VMIMulIOp, VmulOp>,
      OneToNVMIVecScalarOpPattern<VMIAddSOp, VaddsOp>,
      OneToNVMIVecScalarOpPattern<VMIMulSOp, VmulsOp>,
      OneToNVMIVecScalarOpPattern<VMIMaxSOp, VmaxsOp>,
      OneToNVMIVecScalarOpPattern<VMIMinSOp, VminsOp>,
      OneToNVMIVecScalarOpPattern<VMIShlSOp, VshlsOp>,
      OneToNVMIVecScalarOpPattern<VMIShrSOp, VshrsOp>, OneToNVMIVmullOpPattern,
      OneToNVMIVmulaOpPattern,
      OneToNVMIFmaOpPattern, OneToNVMIVexpdifOpPattern,
      OneToNVMIBinaryOpPattern<VMIDivFOp, VdivOp>,
      OneToNVMIBinaryOpPattern<VMIMinFOp, VminOp>,
      OneToNVMIBinaryOpPattern<VMIMinIOp, VminOp>,
      OneToNVMIBinaryOpPattern<VMIMaxFOp, VmaxOp>,
      OneToNVMIBinaryOpPattern<VMIMaxIOp, VmaxOp>,
      OneToNVMIUnaryOpPattern<VMINegFOp, VnegOp>,
      OneToNVMIUnaryOpPattern<VMINegIOp, VnegOp>,
      OneToNVMIUnaryOpPattern<VMIAbsFOp, VabsOp>,
      OneToNVMIUnaryOpPattern<VMIAbsIOp, VabsOp>,
      OneToNVMIUnaryOpPattern<VMISqrtOp, VsqrtOp>,
      OneToNVMIUnaryOpPattern<VMIExpOp, VexpOp>,
      OneToNVMIUnaryOpPattern<VMILnOp, VlnOp>,
      OneToNVMIUnaryOpPattern<VMIReluOp, VreluOp>,
      OneToNVMIBinaryOpPattern<VMIAndIOp, VandOp>,
      OneToNVMIBinaryOpPattern<VMIOrIOp, VorOp>,
      OneToNVMIBinaryOpPattern<VMIXOrIOp, VxorOp>,
      OneToNVMIShiftOpPattern<VMIShLIOp, VshlOp>,
      OneToNVMIShiftOpPattern<VMIShRUIOp, VshrOp>,
      OneToNVMIShiftOpPattern<VMIShRSIOp, VshrOp>,
      OneToNVMIUnaryOpPattern<VMINotOp, VnotOp>,
      OneToNVMICmpOpPattern<VMICmpFOp>, OneToNVMICmpOpPattern<VMICmpIOp>,
      OneToNVMISelectOpPattern, OneToNVMIVselrOpPattern,
      OneToNVMIActivePrefixIndexOpPattern,
      OneToNVMICompressOpPattern,
      OneToNVMICompressStoreOpPattern>(typeConverter, patterns.getContext());
}

static void populateVMIReductionAndConversionPatterns(
    VMIToVPTOTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<
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
      OneToNVMIFPToUIOpPattern,
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

void populateVMIConversionPatterns(
    VMIToVPTOTypeConverter &typeConverter, RewritePatternSet &patterns) {
  populateVMIStructuralAndMemoryPatterns(typeConverter, patterns);
  populateVMIArithmeticPatterns(typeConverter, patterns);
  populateVMIReductionAndConversionPatterns(typeConverter, patterns);
}

static WalkResult verifyNoResidualCreateMask(Operation *op) {
  if (auto createMask = dyn_cast<VMICreateMaskOp>(op)) {
    if (!createMask.getActiveLanes().getDefiningOp<arith::ConstantOp>()) {
      createMask.emitError()
          << kVMIDiagUnsupportedPrefix
          << "dynamic pto.vmi.create_mask active_lanes could not be lowered "
             "by the current runtime predicate generation plan";
      return WalkResult::interrupt();
    }
  }
  return WalkResult::advance();
}

static WalkResult verifyNoResidualConstant(Operation *op) {
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
  return WalkResult::advance();
}

LogicalResult verifyNoResidualVMIIR(ModuleOp module) {
  WalkResult result = module.walk([](Operation *op) {
    if (WalkResult result = verifyNoResidualCreateMask(op);
        result.wasInterrupted()) {
      return result;
    }
    if (WalkResult result = verifyNoResidualConstant(op);
        result.wasInterrupted()) {
      return result;
    }
    bool hasResidualVMI = isVMIOp(op) || hasVMIType(op);
    if (hasResidualVMI) {
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
  if (failed(supports.getExtFSupport(op, reason))) {
    return failure();
  }
  return success();
}

LogicalResult checkSupportedTruncFShape(VMITruncFOp op,
                                        std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (failed(supports.getTruncFSupport(op, reason))) {
    return failure();
  }
  return success();
}

LogicalResult checkSupportedExtSIShape(VMIExtSIOp op,
                                       std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (failed(supports.getExtSISupport(op, reason))) {
    return failure();
  }
  return success();
}

LogicalResult checkSupportedExtUIShape(VMIExtUIOp op,
                                       std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (failed(supports.getExtUISupport(op, reason))) {
    return failure();
  }
  return success();
}

LogicalResult checkSupportedTruncIShape(VMITruncIOp op,
                                        std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (failed(supports.getTruncISupport(op, reason))) {
    return failure();
  }
  return success();
}

static LogicalResult checkSameWidthConversionArity(
    VMIVRegType sourceType, VMIVRegType resultType, StringRef conversionName,
    std::string *reason);

template <typename ShapeOp, typename ShapeCheck>
WalkResult verifySupportedShapeOp(ShapeOp op, ShapeCheck check,
                                  StringRef diagnostic);

// Shared shape-check prologue for fp<->int and int->fp conversions: requires
// both the source and result VMI vreg layouts to be assigned.
struct AssignedSourceResultLayouts {
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
};

static FailureOr<AssignedSourceResultLayouts> getAssignedSourceResultLayouts(
    VMIVRegType sourceType, VMIVRegType resultType, std::string *reason) {
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !resultLayout) {
    return emitFailure<AssignedSourceResultLayouts>(
        reason, "requires assigned source/result layouts");
  }
  return AssignedSourceResultLayouts{sourceLayout, resultLayout};
}

template <typename OpTy, typename ContractLookup>
LogicalResult checkSupportedFPToIntShape(OpTy op, StringRef conversionName,
                                         ContractLookup lookup,
                                         std::string *reason = nullptr) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  FailureOr<AssignedSourceResultLayouts> layouts =
      getAssignedSourceResultLayouts(sourceType, resultType, reason);
  if (failed(layouts)) {
    return failure();
  }

  Type srcElem = sourceType.getElementType();
  Type dstElem = resultType.getElementType();
  auto contract = lookup(srcElem, dstElem);
  if (!contract) {
    return emitLogicalFailure(reason, Twine("unsupported ") + conversionName +
                                          " conversion element type pair");
  }

  unsigned srcBits = pto::getPTOStorageElemBitWidth(srcElem);
  unsigned dstBits = pto::getPTOStorageElemBitWidth(dstElem);

  if (srcBits == dstBits) {
    // Same-width (f32→s32, f16→s16): layout equality + arity equality.
    if (failed(checkSameWidthConversionArity(sourceType, resultType,
                                             conversionName, reason))) {
      return failure();
    }
  } else {
    // Widen or narrow: use the cast-layout framework (same as extf/truncf).
    VMILayoutSupport layoutSupport;
    FailureOr<VMICastLayoutFact> fact =
        layoutSupport.getCastLayoutFactForLayouts(
            sourceType, resultType, layouts->sourceLayout,
            layouts->resultLayout, reason);
    if (failed(fact)) {
      return failure();
    }
  }

  return success();
}

static LogicalResult checkSameWidthConversionArity(
    VMIVRegType sourceType, VMIVRegType resultType, StringRef conversionName,
    std::string *reason) {
  bool layoutMismatch = sourceType.getLayoutAttr() != resultType.getLayoutAttr();
  if (layoutMismatch) {
    if (reason) {
      *reason = (Twine("same-width ") + conversionName +
                 " requires matching layouts")
                    .str();
    }
    return failure();
  }
  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  bool arityMismatch = failed(sourceArity) || failed(resultArity) ||
                       *sourceArity != *resultArity;
  if (arityMismatch) {
    if (reason) {
      *reason = (Twine("same-width ") + conversionName +
                 " requires matching physical arity")
                    .str();
    }
    return failure();
  }
  return success();
}

LogicalResult checkSupportedFPToSIShape(VMIFPToSIOp op,
                                        std::string *reason = nullptr) {
  return checkSupportedFPToIntShape(
      op, "fp-to-si",
      [](Type source, Type result) {
        return lookupVMIFpToSiContract(source, result);
      },
      reason);
}

LogicalResult checkSupportedFPToUIShape(VMIFPToUIOp op,
                                        std::string *reason = nullptr) {
  return checkSupportedFPToIntShape(
      op, "fp-to-ui",
      [](Type source, Type result) {
        return lookupVMIFpToUIContract(source, result);
      },
      reason);
}

LogicalResult checkSupportedSIToFPShape(VMISIToFPOp op,
                                        std::string *reason = nullptr) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  FailureOr<AssignedSourceResultLayouts> layouts =
      getAssignedSourceResultLayouts(sourceType, resultType, reason);
  if (failed(layouts)) {
    return failure();
  }
  unsigned srcBits = pto::getPTOStorageElemBitWidth(sourceType.getElementType());
  unsigned dstBits = pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (srcBits == 32 && dstBits == 32) {
    if (!resultType.getElementType().isF32()) {
      return emitLogicalFailure(reason, "requires f32 result element type");
    }
    if (failed(checkSameWidthConversionArity(sourceType, resultType,
                                             "si32->f32", reason))) {
      return failure();
    }
  } else if (srcBits == 8 && dstBits == 16) {
    if (!resultType.getElementType().isF16()) {
      return emitLogicalFailure(reason, "requires f16 result element type");
    }
    VMILayoutSupport layoutSupport;
    if (failed(layoutSupport.getCastLayoutFactForLayouts(
            sourceType, resultType, layouts->sourceLayout,
            layouts->resultLayout, reason))) {
      return failure();
    }
  } else {
    return emitLogicalFailure(reason, "supports only si32 -> f32 or si8 -> f16");
  }
  return success();
}

LogicalResult checkSupportedBitcastShape(VMIBitcastOp op, std::string *reason) {
  VMILayoutSupport supports;
  if (failed(supports.getBitcastSupport(op, reason))) {
    return failure();
  }
  return success();
}



struct ChannelShapePlan {
  int64_t channels;
  VMILayoutAttr expectedLayout;
};

template <typename ValidateFn>
static LogicalResult validateContiguousParts(
    Operation *op, ValueRange parts, StringRef failureMessage,
    OneToNPatternRewriter &rewriter, ValidateFn &&validate) {
  for (Value part : parts) {
    if (!validate(part)) {
      return rewriter.notifyMatchFailure(op, failureMessage);
    }
  }
  return success();
}

template <typename OpTy>
static LogicalResult lowerPhysicalBinaryWithCarryResults(
    OpTy op, SmallVectorImpl<Value> &results, SmallVectorImpl<Value> &carries,
    OneToNPatternRewriter &rewriter, TypeConverter &typeConverter) {
  results.append(carries);
  return replacePhysicalResults(rewriter, op, results, typeConverter);
}

template <typename OpTy>
static LogicalResult lowerBinaryPhysicalResults(
    OpTy op, SmallVectorImpl<Value> &results, OneToNPatternRewriter &rewriter,
    TypeConverter &typeConverter) {
  return replacePhysicalResults(rewriter, op, results, typeConverter);
}

template <typename OpTy, typename LowerFn>
static LogicalResult lowerPointwisePhysicalParts(
    OpTy op, ArrayRef<Type> resultTypes, StringRef arityMessage,
    OneToNPatternRewriter &rewriter, LowerFn &&lowerFn,
    TypeConverter &typeConverter) {
  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
    FailureOr<Value> result = lowerFn(index, resultType);
    if (failed(result)) {
      return failure();
    }
    results.push_back(*result);
  }
  return replacePhysicalResults(rewriter, op, results, typeConverter);
}

static FailureOr<int64_t> getContiguousChannelInputArity(
    ValueRange inputs, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<int64_t> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  int64_t inputArity = 0;
  for (Value input : inputs) {
    auto inputType = dyn_cast<VMIVRegType>(input.getType());
    if (!inputType) {
      return fail("requires every input to be a VMI vreg");
    }
    VMILayoutAttr inputLayout = inputType.getLayoutAttr();
    if (!inputLayout || !inputLayout.isContiguous()) {
      return fail("requires every input layout to be contiguous");
    }
    FailureOr<int64_t> arity = getVMIPhysicalArity(inputType);
    if (failed(arity)) {
      return fail("requires computable input physical arity");
    }
    inputArity += *arity;
  }
  return inputArity;
}

static bool isContiguousVMIVRegPart(Value part) {
  auto partType = dyn_cast<VMIVRegType>(part.getType());
  VMILayoutAttr partLayout = partType.getLayoutAttr();
  return partType && partLayout && partLayout.isContiguous();
}

template <typename ChannelOp>
static FailureOr<ChannelShapePlan> buildChannelShapePlan(
    ChannelOp op, int64_t channels, StringRef operationName,
    std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<ChannelShapePlan> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  if (channels != 2 && channels != 4) {
    return fail(Twine("pto.vmi.") + operationName +
                " supports only 2 or 4 channels");
  }
  return ChannelShapePlan{
      channels, VMILayoutAttr::getDeinterleaved(op.getContext(), channels)};
}

static FailureOr<int64_t> checkChannelSplitResultShape(
    VMIChannelSplitOp op, int64_t sourceArity, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<int64_t> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  int64_t resultArity = 0;
  for (Value result : op.getResults()) {
    VMILayoutAttr resultLayout =
        cast<VMIVRegType>(result.getType()).getLayoutAttr();
    if (!resultLayout || !resultLayout.isContiguous()) {
      return fail("requires every result layout to be contiguous");
    }
    FailureOr<int64_t> arity =
        getVMIPhysicalArity(cast<VMIVRegType>(result.getType()));
    if (failed(arity)) {
      return fail("requires computable result physical arity");
    }
    resultArity += *arity;
  }
  if (sourceArity != resultArity) {
    return fail("requires source and result to have the same physical arity");
  }
  return resultArity;
}

static FailureOr<int64_t> checkChannelSplitSourceShape(
    VMIChannelSplitOp op, VMILayoutAttr expectedLayout,
    std::string *reason) {
  auto fail = [&reason](const Twine &message) -> FailureOr<int64_t> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  if (!sourceLayout) {
    return fail("requires assigned source layout");
  }
  bool invalidSourceLayout =
      !sourceLayout.isContiguous() && sourceLayout != expectedLayout;
  if (invalidSourceLayout) {
    return fail("requires source layout to be contiguous or matching deinterleaved channel layout");
  }
  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  if (failed(sourceArity)) {
    return fail("requires computable source physical arity");
  }
  return *sourceArity;
}

static LogicalResult checkChannelMergeResultShape(
    VMIChannelMergeOp op, VMILayoutAttr expectedLayout, int64_t inputArity,
    std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!resultLayout) {
    return fail("requires assigned result layout");
  }
  bool invalidResultLayout =
      !resultLayout.isContiguous() && resultLayout != expectedLayout;
  if (invalidResultLayout) {
    return fail("requires result layout to be contiguous or matching "
                "deinterleaved channel layout");
  }
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (failed(resultArity)) {
    return fail("requires computable result physical arity");
  }
  if (*resultArity != inputArity) {
    return fail("requires source and result to have the same physical arity");
  }
  return success();
}

LogicalResult checkSupportedChannelSplitShape(VMIChannelSplitOp op,
                                              std::string *reason = nullptr) {
  FailureOr<ChannelShapePlan> plan =
      buildChannelShapePlan(op, op.getNumResults(), "channel_split", reason);
  if (failed(plan)) {
    return failure();
  }
  FailureOr<int64_t> sourceArity =
      checkChannelSplitSourceShape(op, plan->expectedLayout, reason);
  if (failed(sourceArity)) {
    return failure();
  }
  if (failed(checkChannelSplitResultShape(op, *sourceArity, reason))) {
    return failure();
  }

  return success();
}

LogicalResult checkSupportedChannelMergeShape(VMIChannelMergeOp op,
                                              std::string *reason = nullptr) {
  FailureOr<ChannelShapePlan> plan = buildChannelShapePlan(
      op, op.getInputs().size(), "channel_merge", reason);
  if (failed(plan)) {
    return failure();
  }
  FailureOr<int64_t> inputArity =
      getContiguousChannelInputArity(op.getInputs(), reason);
  if (failed(inputArity)) {
    return failure();
  }

  if (failed(checkChannelMergeResultShape(op, plan->expectedLayout,
                                          *inputArity, reason))) {
    return failure();
  }
  return success();
}

struct ActivePrefixIndexShapePlan {
  VMIMaskType maskType;
  VMIVRegType resultType;
};

static LogicalResult checkActivePrefixIndexLayouts(
    VMIMaskType maskType, VMIVRegType resultType, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!maskLayout || !resultLayout) {
    return fail("requires assigned mask and result layouts");
  }
  bool nonContiguousLayout = !maskLayout.isContiguous() ||
                             !resultLayout.isContiguous();
  if (nonContiguousLayout) {
    return fail("requires contiguous mask and result layouts");
  }
  return success();
}

static LogicalResult checkActivePrefixIndexPhysicalChunks(
    VMIMaskType maskType, VMIVRegType resultType, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  std::string resultFullReason;
  if (failed(checkFullDataPhysicalChunks(resultType, &resultFullReason))) {
    return fail(Twine("requires full result physical chunks so padding mask "
                      "lanes cannot affect the observable prefix; ") +
                resultFullReason);
  }
  std::string maskFullReason;
  if (failed(checkFullVMIPhysicalChunks(maskType, &maskFullReason))) {
    return fail(Twine("requires full mask physical chunks so padding mask "
                      "lanes cannot affect the observable prefix; ") +
                maskFullReason);
  }
  return success();
}

static LogicalResult checkActivePrefixIndexSingleChunk(
    VMIMaskType maskType, VMIVRegType resultType, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  bool missingArity = failed(maskArity) || failed(resultArity);
  if (missingArity) {
    return fail("requires computable mask and result physical arity");
  }
  if (*maskArity != 1 || *resultArity != 1) {
    return fail("requires a single physical chunk; multi-chunk prefix needs "
                "cross-chunk carry");
  }
  return success();
}

static FailureOr<ActivePrefixIndexShapePlan> buildActivePrefixIndexShapePlan(
    VMIActivePrefixIndexOp op, std::string *reason) {
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (failed(checkActivePrefixIndexLayouts(maskType, resultType, reason))) {
    return failure();
  }
  if (failed(checkActivePrefixIndexPhysicalChunks(maskType, resultType,
                                                  reason))) {
    return failure();
  }
  if (failed(checkActivePrefixIndexSingleChunk(maskType, resultType, reason))) {
    return failure();
  }

  return ActivePrefixIndexShapePlan{maskType, resultType};
}

LogicalResult
checkSupportedActivePrefixIndexShape(VMIActivePrefixIndexOp op,
                                     std::string *reason = nullptr) {
  FailureOr<ActivePrefixIndexShapePlan> plan =
      buildActivePrefixIndexShapePlan(op, reason);
  if (failed(plan)) {
    return failure();
  }
  return success();
}

struct CompressPhysicalShapePlan {
  VMIVRegType valueType;
  VMIMaskType maskType;
};

static FailureOr<CompressPhysicalShapePlan> buildCompressPhysicalShapePlan(
    VMIVRegType valueType, VMIMaskType maskType, StringRef fullChunkSuffix,
    StringRef arityMessage, std::string *reason) {
  FailureOr<AssignedValueMaskLayouts> layouts =
      getAssignedValueMaskLayouts(valueType, maskType, reason);
  if (failed(layouts)) {
    return failure();
  }
  bool nonContiguousInputs =
      !layouts->value.isContiguous() || !layouts->mask.isContiguous();
  if (nonContiguousInputs) {
    return emitFailure<CompressPhysicalShapePlan>(
        reason, "requires contiguous value and mask layouts");
  }
  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(valueType, &fullChunkReason))) {
    return emitFailure<CompressPhysicalShapePlan>(
        reason, Twine("requires full physical chunks so padding mask lanes ") +
                    fullChunkSuffix + "; " + fullChunkReason);
  }
  FailureOr<int64_t> valueArity = getVMIPhysicalArity(valueType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  bool invalidArity = failed(valueArity) || failed(maskArity) ||
                      *valueArity != 1 || *maskArity != 1;
  if (invalidArity) {
    return emitFailure<CompressPhysicalShapePlan>(reason, arityMessage);
  }
  return CompressPhysicalShapePlan{valueType, maskType};
}

static LogicalResult checkSupportedCompressResultShape(
    VMIVRegType resultType, StringRef layoutMessage,
    StringRef computableArityMessage, StringRef arityMessage,
    std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!resultLayout) {
    return fail(layoutMessage);
  }
  if (!resultLayout.isContiguous()) {
    return fail("requires contiguous result layouts");
  }
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (failed(resultArity)) {
    return fail(computableArityMessage);
  }
  if (*resultArity != 1) {
    return fail(arityMessage);
  }
  return success();
}

static LogicalResult checkCompressStoreDestination(VMICompressStoreOp op,
                                                   std::string *reason) {
  if (isa<PtrType>(op.getDestination().getType())) {
    return success();
  }
  if (reason) {
    *reason = "requires !pto.ptr destination because pto.vstur is pointer-only";
  }
  return failure();
}

LogicalResult checkSupportedCompressShape(VMICompressOp op,
                                          std::string *reason = nullptr) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !maskLayout || !resultLayout) {
    return emitLogicalFailure(
        reason, "requires assigned source, mask, and result layouts");
  }
  if (!sourceLayout.isContiguous() || !maskLayout.isContiguous() ||
      !resultLayout.isContiguous()) {
    return emitLogicalFailure(
        reason, "requires contiguous source, mask, and result layouts");
  }
  FailureOr<CompressPhysicalShapePlan> plan = buildCompressPhysicalShapePlan(
      sourceType, maskType,
      "cannot be squeezed into the result",
      "requires a single physical chunk; multi-chunk compress needs cross-"
      "chunk compaction",
      reason);
  if (failed(plan)) {
    return failure();
  }
  return checkSupportedCompressResultShape(
      resultType, "requires assigned result layouts",
      "requires computable source, mask, and result physical arity",
      "requires a single physical chunk; multi-chunk compress needs "
      "cross-chunk compaction",
      reason);
}

LogicalResult checkSupportedCompressStoreShape(
    VMICompressStoreOp op,
    std::string *reason = nullptr) {
  auto valueType = cast<VMIVRegType>(op.getValue().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  FailureOr<CompressPhysicalShapePlan> plan = buildCompressPhysicalShapePlan(
      valueType, maskType, "cannot be squeezed into memory",
      "requires a single physical chunk; multi-chunk compress_store needs "
      "cross-chunk compaction and SQZN state planning",
      reason);
  if (failed(plan)) {
    return failure();
  }

  return checkCompressStoreDestination(op, reason);
}

struct ReducePhysicalShapePlan {
  VMILayoutAttr sourceLayout;
  VMILayoutAttr maskLayout;
  VMILayoutAttr resultLayout;
  int64_t sourceArity;
  int64_t resultArity;
};

template <typename OpTy>
static LogicalResult checkReduceLayouts(OpTy op, VMILayoutAttr *sourceLayout,
                                        VMILayoutAttr *maskLayout,
                                        VMILayoutAttr *resultLayout,
                                        std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  *sourceLayout = sourceType.getLayoutAttr();
  *maskLayout = maskType.getLayoutAttr();
  *resultLayout = resultType.getLayoutAttr();
  if (!*sourceLayout || !*maskLayout || !*resultLayout) {
    return fail("requires assigned source, mask, and result layouts");
  }
  bool nonContiguousLayout = !sourceLayout->isContiguous() ||
                             !maskLayout->isContiguous() ||
                             !resultLayout->isContiguous();
  if (nonContiguousLayout) {
    return fail("requires contiguous source, mask, and result layouts");
  }
  return success();
}

static LogicalResult checkReducePhysicalArity(
    VMIVRegType sourceType, VMIMaskType maskType, VMIVRegType resultType,
    int64_t *sourceArity, int64_t *resultArity, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<int64_t> sourceParts = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> maskParts = getVMIPhysicalArity(maskType);
  FailureOr<int64_t> resultParts = getVMIPhysicalArity(resultType);
  bool cannotComputeArity = failed(sourceParts) || failed(maskParts) ||
                            failed(resultParts);
  if (cannotComputeArity) {
    return fail("requires computable physical arity");
  }
  bool mismatchedInputArity = *sourceParts < 1 || *maskParts != *sourceParts;
  if (mismatchedInputArity) {
    return fail("requires source and mask physical arity to match and be "
                "non-empty");
  }
  if (*resultParts != 1) {
    return fail("requires one result physical chunk");
  }
  *sourceArity = *sourceParts;
  *resultArity = *resultParts;
  return success();
}

template <typename OpTy>
static LogicalResult checkReduceSourceChunks(OpTy op, VMIVRegType sourceType,
                                             std::string *reason) {
  std::string fullChunkReason;
  if (succeeded(checkFullDataPhysicalChunks(sourceType, &fullChunkReason))) {
    return success();
  }
  if (reason) {
    *reason = (Twine("requires full source physical chunks so padding lanes "
                    "do not participate in the reduction; ") +
               fullChunkReason)
                  .str();
  }
  return failure();
}

template <typename OpTy>
static FailureOr<ReducePhysicalShapePlan> buildReducePhysicalShapePlan(
    OpTy op, std::string *reason) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr sourceLayout;
  VMILayoutAttr maskLayout;
  VMILayoutAttr resultLayout;
  if (failed(checkReduceLayouts(op, &sourceLayout, &maskLayout, &resultLayout,
                                reason))) {
    return failure();
  }

  if (failed(checkReduceSourceChunks(op, sourceType, reason))) {
    return failure();
  }

  int64_t sourceArity;
  int64_t resultArity;
  if (failed(checkReducePhysicalArity(sourceType, maskType, resultType,
                                      &sourceArity, &resultArity, reason))) {
    return failure();
  }

  return ReducePhysicalShapePlan{sourceLayout, maskLayout, resultLayout,
                                 sourceArity, resultArity};
}

template <typename OpTy>
LogicalResult
checkSupportedReduceShape(OpTy op, bool requiresReassoc,
                          std::string *reason = nullptr) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  if (requiresReassoc && !op->hasAttr("reassoc")) {
    return fail("requires reassoc attr for pair-wise floating-point vcadd");
  }
  FailureOr<ReducePhysicalShapePlan> plan =
      buildReducePhysicalShapePlan(op, reason);
  if (failed(plan)) {
    return failure();
  }
  return success();
}

template <typename OpTy>
LogicalResult
checkSupportedGroupReduceShape(OpTy op, std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if constexpr (std::is_same_v<OpTy, VMIGroupReduceAddFOp>) {
    if (succeeded(supports.getGroupReduceAddFSupport(op, reason))) {
      return success();
    }
  } else if constexpr (std::is_same_v<OpTy, VMIGroupReduceMaxFOp>) {
    if (succeeded(supports.getGroupReduceMaxFSupport(op, reason))) {
      return success();
    }
  } else if constexpr (std::is_same_v<OpTy, VMIGroupReduceMaxIOp>) {
    if (succeeded(supports.getGroupReduceMaxISupport(op, reason))) {
      return success();
    }
  } else if constexpr (std::is_same_v<OpTy, VMIGroupReduceMinFOp>) {
    if (succeeded(supports.getGroupReduceMinFSupport(op, reason))) {
      return success();
    }
  } else if constexpr (std::is_same_v<OpTy, VMIGroupReduceMinIOp>) {
    if (succeeded(supports.getGroupReduceMinISupport(op, reason))) {
      return success();
    }
  } else {
    if (succeeded(supports.getGroupReduceAddISupport(op, reason))) {
      return success();
    }
  }
  return failure();
}

struct GroupBroadcastShapePlan {
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
  int64_t numGroups;
  int64_t lanesPerPart;
  int64_t groupSize;
  int64_t resultFactor;
};

static LogicalResult checkGroupBroadcastLogicalContract(
    VMIGroupBroadcastOp op, VMIVRegType sourceType, VMIVRegType resultType,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    int64_t numGroups, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  bool mismatchedElementTypes =
      sourceType.getElementType() != resultType.getElementType();
  if (mismatchedElementTypes) {
    return fail("requires source/result element type to match");
  }
  bool missingLayouts = !sourceLayout || !resultLayout;
  if (missingLayouts) {
    return fail("requires assigned source/result layouts");
  }
  bool invalidGroupCount = numGroups <= 0;
  if (invalidGroupCount) {
    return fail("requires positive num_groups");
  }
  int64_t safeNumGroups = numGroups;
  bool sourceLaneCountMismatch = sourceType.getElementCount() != numGroups;
  if (sourceLaneCountMismatch) {
    return fail("requires source lane count to match num_groups");
  }
  bool resultLaneCountMismatch =
      resultType.getElementCount() % safeNumGroups != 0;
  if (resultLaneCountMismatch) {
    return fail("requires num_groups to evenly divide result lane count");
  }
  bool sourceLayoutMismatch =
      !sourceLayout.isGroupSlots() || sourceLayout.getNumGroups() != numGroups;
  if (sourceLayoutMismatch) {
    return fail("requires matching num_groups source layout");
  }
  bool resultUsesGroupSlots = resultLayout.isGroupSlots();
  if (resultUsesGroupSlots) {
    return fail("requires dense result layout");
  }
  bool unsupportedSlots = sourceLayout.getSlots() > 0 &&
                          sourceLayout.getSlots() != 8 &&
                          sourceLayout.getSlots() != 1;
  if (unsupportedSlots) {
    return fail("supports only slots=8 or slots=1 group_broadcast source "
                "layouts");
  }
  return success();
}

static LogicalResult checkGroupBroadcastSupportContract(
    VMIGroupBroadcastOp op, std::string *reason) {
  VMILayoutSupport supports;
  std::string supportReason;
  if (failed(supports.getGroupBroadcastSupport(op, &supportReason))) {
    if (reason) {
      *reason = supportReason;
    }
    return failure();
  }
  return success();
}

static FailureOr<GroupBroadcastShapePlan> buildGroupBroadcastShapePlan(
    VMIGroupBroadcastOp op, std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<GroupBroadcastShapePlan> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  int64_t numGroups = op.getNumGroupsAttr().getInt();
  if (failed(checkGroupBroadcastLogicalContract(
          op, sourceType, resultType, sourceLayout, resultLayout, numGroups,
          reason))) {
    return failure();
  }
  if (failed(checkGroupBroadcastSupportContract(op, reason))) {
    return failure();
  }

  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(sourceType.getElementType());
  FailureOr<int64_t> resultLanesPerPart =
      getDataLanesPerPart(resultType.getElementType());
  if (failed(lanesPerPart) || failed(resultLanesPerPart) ||
      *lanesPerPart != *resultLanesPerPart) {
    return fail("requires matching physical lanes per part");
  }
  FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
      resultType, numGroups, reason);
  if (failed(groupSize)) {
    return failure();
  }
  if (*lanesPerPart % *groupSize != 0 && *groupSize % *lanesPerPart != 0) {
    return fail("requires derived group size to divide or be a multiple of "
                "physical lanes per part");
  }

  FailureOr<int64_t> resultFactor = getDataLayoutFactor(resultType);
  if (failed(resultFactor)) {
    return fail("requires known result layout factor");
  }
  return GroupBroadcastShapePlan{sourceLayout, resultLayout, numGroups,
                                 *lanesPerPart, *groupSize, *resultFactor};
}

static LogicalResult checkGroupBroadcastResultShape(
    VMIVRegType resultType, VMILayoutAttr resultLayout, int64_t groupSize,
    int64_t lanesPerPart, int64_t resultFactor, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  bool laneStridedDense =
      resultLayout.isDense() && resultLayout.getLaneStride() > 1;
  if (!laneStridedDense) {
    std::string fullChunkReason;
    if (failed(checkFullDataPhysicalChunks(resultType, &fullChunkReason))) {
      return fail(Twine("requires full result physical chunks; ") +
                  fullChunkReason);
    }
  }
  if (resultFactor == 1) {
    return success();
  }
  FailureOr<int64_t> resultBlockElems =
      getVMILayoutBlockElems(resultType);
  bool blockFragmentSmallGroup =
      resultLayout.isBlockDeinterleaved() && succeeded(resultBlockElems) &&
      groupSize < lanesPerPart && lanesPerPart % *resultBlockElems == 0;
  bool deinterleavedSmallGroup =
      resultLayout.isDeinterleaved() &&
      groupSize < lanesPerPart && groupSize >= resultFactor &&
      groupSize % resultFactor == 0 &&
      lanesPerPart % (groupSize / resultFactor) == 0;
  if (blockFragmentSmallGroup || deinterleavedSmallGroup) {
    return success();
  }
  int64_t logicalSpanPerResultChunk = lanesPerPart * resultFactor;
  bool groupSpansMultipleChunks =
      groupSize < lanesPerPart || groupSize % logicalSpanPerResultChunk != 0;
  if (groupSpansMultipleChunks) {
    return fail("deinterleaved result requires every physical result chunk to "
                "stay within one logical group");
  }
  return success();
}

LogicalResult checkSupportedGroupBroadcastShape(
    VMIGroupBroadcastOp op,
    std::string *reason = nullptr) {
  FailureOr<GroupBroadcastShapePlan> plan =
      buildGroupBroadcastShapePlan(op, reason);
  if (failed(plan)) {
    return failure();
  }
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  return checkGroupBroadcastResultShape(
      resultType, plan->resultLayout, plan->groupSize, plan->lanesPerPart,
      plan->resultFactor, reason);
}

LogicalResult checkSupportedVdhistShape(VMIVdhistOp op,
                                       std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (succeeded(supports.getVdhistSupport(op, reason))) {
    return success();
  }
  return failure();
}

LogicalResult checkSupportedVchistShape(VMIVchistOp op,
                                       std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (succeeded(supports.getVchistSupport(op, reason))) {
    return success();
  }
  return failure();
}

struct VmullShapePlan {
  VMIVRegType dataType;
  VMILayoutAttr layout;
  int64_t arity;
};

static LogicalResult validateVmullDataLayout(
    VMIVRegType aType, VMIVRegType bType, VMIVRegType lowType,
    VMIVRegType highType, VMIMaskType maskType, std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  if (aType != bType || aType != lowType || aType != highType) {
    return fail("requires identical a, b, low, and high VMI vreg types");
  }
  VMILayoutAttr layout = aType.getLayoutAttr();
  if (!layout) {
    return fail("requires an assigned data layout");
  }
  bool supportedLayout =
      layout.getLaneStride() == 1 &&
      (layout.isContiguous() ||
       (layout.isDeinterleaved() &&
        (layout.getFactor() == 2 || layout.getFactor() == 4)));
  if (!supportedLayout) {
    return fail("requires contiguous or deinterleaved factor 2/4 layout with "
                "lane_stride=1");
  }
  bool maskLayoutMismatch = maskType.getLayoutAttr() != layout;
  if (maskLayoutMismatch) {
    return fail("requires the mask and all four data values to share one layout");
  }
  bool unsupportedMaskGranularity = maskType.getGranularity() != "b32";
  if (unsupportedMaskGranularity) {
    return fail("requires b32 mask granularity");
  }
  return success();
}

static FailureOr<VmullShapePlan> validateVmullLogicalShape(
    VMIVmullOp op, VMIVRegType aType, VMIVRegType bType,
    VMIVRegType lowType, VMIVRegType highType, VMIMaskType maskType,
    std::string *reason) {
  auto fail = [&reason](const Twine &message)
      -> FailureOr<VmullShapePlan> {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  auto elementType = dyn_cast<IntegerType>(aType.getElementType());
  bool unsupportedElementType =
      !elementType || elementType.getWidth() != 32 ||
      (!elementType.isSignless() && !elementType.isUnsigned());
  if (unsupportedElementType) {
    return fail("requires element type to be exactly i32 or ui32");
  }
  int64_t lanes = aType.getElementCount();
  if (lanes != 64 && lanes != 128 && lanes != 256) {
    return fail("requires logical lane count 64, 128, or 256");
  }
  if (failed(validateVmullDataLayout(aType, bType, lowType, highType, maskType,
                                     reason))) {
    return failure();
  }
  return VmullShapePlan{aType, aType.getLayoutAttr(), 0};
}

static FailureOr<VmullShapePlan> buildVmullShapePlan(
    VMIVmullOp op, std::string *reason) {
  auto aType = cast<VMIVRegType>(op.getA().getType());
  auto bType = cast<VMIVRegType>(op.getB().getType());
  auto lowType = cast<VMIVRegType>(op.getLow().getType());
  auto highType = cast<VMIVRegType>(op.getHigh().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());

  FailureOr<VmullShapePlan> logical = validateVmullLogicalShape(
      op, aType, bType, lowType, highType, maskType, reason);
  if (failed(logical)) {
    return failure();
  }

  FailureOr<int64_t> aArity = getVMIPhysicalArity(aType);
  FailureOr<int64_t> bArity = getVMIPhysicalArity(bType);
  FailureOr<int64_t> lowArity = getVMIPhysicalArity(lowType);
  FailureOr<int64_t> highArity = getVMIPhysicalArity(highType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  bool missingArity =
      failed(aArity) || failed(bArity) || failed(lowArity) ||
      failed(highArity) || failed(maskArity) || *aArity < 1;
  if (missingArity) {
    if (reason) {
      *reason = "requires computable non-empty physical arity on every port";
    }
    return failure();
  }
  bool arityMismatch =
      *aArity != *bArity || *aArity != *lowArity || *aArity != *highArity ||
      *aArity != *maskArity;
  if (arityMismatch) {
    if (reason) {
      *reason = "requires matching physical arity on a, b, mask, low, and high";
    }
    return failure();
  }
  return VmullShapePlan{aType, logical->layout, *aArity};
}

static LogicalResult checkVmullPhysicalShape(VMIVRegType dataType,
                                             VMIMaskType maskType,
                                             std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(dataType.getElementType());
  Type physicalElementType = getVMIPhysicalDataElementType(dataType);
  FailureOr<StringRef> physicalMaskGranularity =
      getVMIMaskPhysicalGranularity(maskType);
  bool invalidPhysicalShape =
      failed(lanesPerPart) || *lanesPerPart != 64 ||
      physicalElementType != dataType.getElementType() ||
      failed(physicalMaskGranularity) || *physicalMaskGranularity != "b32";
  if (invalidPhysicalShape) {
    return fail("requires 64xi32/ui32 data parts with corresponding b32 mask "
                "parts");
  }
  return success();
}

LogicalResult checkSupportedVmullShape(VMIVmullOp op,
                                       std::string *reason = nullptr) {
  FailureOr<VmullShapePlan> plan = buildVmullShapePlan(op, reason);
  if (failed(plan)) {
    return failure();
  }
  VMIVRegType aType = plan->dataType;
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  return checkVmullPhysicalShape(aType, maskType, reason);
}

static LogicalResult checkAddCarryMaskPort(VMIMaskType maskType,
                                           VMILayoutAttr dataLayout,
                                           int64_t dataArity,
                                           std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  bool layoutMismatch = maskType.getLayoutAttr() != dataLayout;
  if (layoutMismatch) {
    return fail("requires all data and mask ports to share one layout");
  }
  bool unsupportedGranularity = maskType.getGranularity() != "b32";
  if (unsupportedGranularity) {
    return fail("requires b32 mask granularity");
  }
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  bool hasMatchingArity = succeeded(maskArity) && *maskArity == dataArity;
  if (!hasMatchingArity) {
    return fail("requires matching physical arity on data and mask ports");
  }
  FailureOr<StringRef> physicalGranularity =
      getVMIMaskPhysicalGranularity(maskType);
  bool unsupportedPhysicalGranularity = failed(physicalGranularity) ||
                                        *physicalGranularity != "b32";
  if (unsupportedPhysicalGranularity) {
    return fail("requires physical b32 mask parts");
  }
  return success();
}

static LogicalResult
checkSupportedVMIAddCarryPorts(VMIVRegType lhsType, VMIVRegType rhsType,
                               VMIVRegType resultType,
                               ArrayRef<VMIMaskType> maskTypes,
                               std::string *reason = nullptr) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  auto integerType = dyn_cast<IntegerType>(lhsType.getElementType());
  bool unsupportedIntegerType = !integerType || integerType.getWidth() != 32;
  if (unsupportedIntegerType) {
    return fail("requires 32-bit integer data elements");
  }
  bool mismatchedTypes = lhsType != rhsType || lhsType != resultType;
  if (mismatchedTypes) {
    return fail("requires matching lhs, rhs, and result VMI types");
  }
  if (!lhsType.getLayoutAttr()) {
    return fail("requires assigned data layout");
  }
  if (failed(checkSupportedMaskableVReg(lhsType))) {
    return fail("requires computable physical data parts");
  }

  FailureOr<int64_t> dataArity = getVMIPhysicalArity(lhsType);
  bool invalidDataArity = failed(dataArity) || *dataArity < 1;
  if (invalidDataArity) {
    return fail("requires non-empty physical data parts");
  }
  for (VMIMaskType maskType : maskTypes) {
    if (failed(checkAddCarryMaskPort(maskType, lhsType.getLayoutAttr(),
                                     *dataArity, reason))) {
      return failure();
    }
  }
  FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(lhsType.getElementType());
  bool hasExpectedLanes = succeeded(lanesPerPart) && *lanesPerPart == 64;
  if (!hasExpectedLanes) {
    return fail("requires 64-lane 32-bit data parts");
  }
  return success();
}

LogicalResult checkSupportedVMIAddcShape(VMIVaddcOp op,
                                         std::string *reason = nullptr) {
  return checkSupportedVMIAddCarryPorts(
      cast<VMIVRegType>(op.getLhs().getType()),
      cast<VMIVRegType>(op.getRhs().getType()),
      cast<VMIVRegType>(op.getResult().getType()),
      {cast<VMIMaskType>(op.getMask().getType()),
       cast<VMIMaskType>(op.getCarry().getType())},
      reason);
}

LogicalResult checkSupportedVMIAddcsShape(VMIVaddcsOp op,
                                          std::string *reason = nullptr) {
  return checkSupportedVMIAddCarryPorts(
      cast<VMIVRegType>(op.getLhs().getType()),
      cast<VMIVRegType>(op.getRhs().getType()),
      cast<VMIVRegType>(op.getResult().getType()),
      {cast<VMIMaskType>(op.getCarryIn().getType()),
       cast<VMIMaskType>(op.getMask().getType()),
       cast<VMIMaskType>(op.getCarry().getType())},
      reason);
}

LogicalResult
checkSupportedFmaShape(VMIFmaOp op, std::string *reason = nullptr) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  auto lhsType = cast<VMIVRegType>(op.getLhs().getType());
  FailureOr<int64_t> arity = getVMIPhysicalArity(lhsType);
  bool hasNonEmptyArity = succeeded(arity) && *arity >= 1;
  if (!hasNonEmptyArity) {
    return fail("requires computable non-empty physical arity");
  }

  return success();
}

LogicalResult
checkSupportedReluShape(VMIReluOp op, std::string *reason = nullptr) {
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (failed(checkSupportedMaskableVReg(resultType, reason))) {
    return failure();
  }

  return success();
}

LogicalResult
checkSupportedVselrShape(VMIVselrOp op, std::string *reason = nullptr) {
  VMILayoutSupport supports;
  return supports.getVselrSupport(op, reason);
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

WalkResult emitMemoryUnsupported(Operation *memoryOp, StringRef opName,
                                 VMIVRegType type, Value source,
                                 std::optional<int64_t> constantOffset) {
    std::string reason;
    if (succeeded(checkSupportedLoadShape(type, source, source.getType(),
                                          constantOffset, &reason))) {
      return WalkResult::advance();
    }

    memoryOp->emitError()
        << kVMIDiagUnsupportedPrefix << opName
        << " direct lowering requires a supported memory source (" << reason
        << ")";
    return WalkResult::interrupt();
}

static std::optional<WalkResult> verifySupportedVMIMaskedLoadOp(
    Operation *op, bool enableStableGatherMaskedLoad) {
  if (auto load = dyn_cast<VMIMaskedLoadOp>(op)) {
    if (enableStableGatherMaskedLoad) {
      load.emitError() << kVMIDiagUnsupportedPrefix
                       << "pto.vmi.masked_load stable VGATHER-based lowering "
                          "is reserved for strict masked/tail loads but is "
                          "not implemented yet";
      return WalkResult::interrupt();
    }
    return verifySupportedShapeOp(
        load, checkSupportedMaskedLoadShape,
        "pto.vmi.masked_load direct lowering requires a supported memory source, "
        "contiguous result/passthru/mask layouts, and either full physical "
        "chunks or a statically safe full-read footprint (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIGatherOp(Operation *op) {
  if (auto gather = dyn_cast<VMIGatherOp>(op)) {
    return verifySupportedShapeOp(
        gather, checkSupportedGatherShape,
        "pto.vmi.gather lowers through pto.vgather2/pto.vgather2_bc + pto.vsel "
        "only for UB pointer sources, contiguous full physical chunks, "
        "ui16/i16/f16/bf16 results with ui16 indices and b16 masks, or "
        "32-bit results with i32 indices and b32 masks (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIExpandLoadOp(Operation *op) {
  if (auto load = dyn_cast<VMIExpandLoadOp>(op)) {
    return verifySupportedShapeOp(
        load, checkSupportedExpandLoadShape,
        "pto.vmi.expand_load direct lowering is currently supported for either "
        "a static all-active mask lowered as pto.vlds, or a one-full-chunk "
        "32-bit UB runtime mask lowered through pto.vusqz + pto.vgather2_bc + "
        "pto.vsel (");
  }
  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIMemoryAdvancedLoadOp(
    Operation *op, bool enableStableGatherMaskedLoad) {
  if (auto result = verifySupportedVMIMaskedLoadOp(
          op, enableStableGatherMaskedLoad);
      result.has_value()) {
    return result;
  }
  if (auto result = verifySupportedVMIGatherOp(op); result.has_value()) {
    return result;
  }
  return verifySupportedVMIExpandLoadOp(op);
}

static std::optional<WalkResult> verifySupportedVMIStructuredMaskedStoreOp(
    Operation *op) {
  if (auto store = dyn_cast<VMIMaskedStoreOp>(op)) {
    std::string reason;
    if (succeeded(checkSupportedMaskedStoreShape(
            cast<VMIVRegType>(store.getValue().getType()),
            cast<VMIMaskType>(store.getMask().getType()),
            store.getDestination(), store.getDestination().getType(),
            &reason))) {
      return WalkResult::advance();
    }
    store.emitError()
        << kVMIDiagUnsupportedPrefix
        << "pto.vmi.masked_store requires either full physical chunks or "
           "contiguous tail-store value/mask layout, with UB-backed "
           "destination ("
        << reason << ")";
    return WalkResult::interrupt();
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIInterleaveStoreOp(
    Operation *op) {
  if (auto store = dyn_cast<VMIInterleaveStoreOp>(op)) {
    return verifySupportedShapeOp(
        store, checkSupportedInterleaveStoreShape,
        "pto.vmi.interleave_store lowers through pto.vstsx2 only for matching "
        "contiguous full low/high input chunks with a supported UB destination "
        "and 8/16/32-bit element type (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIGroupStoreOp(Operation *op) {
  if (auto store = dyn_cast<VMIGroupStoreOp>(op)) {
    return verifySupportedShapeOp(
        store, checkSupportedGroupStoreShape,
        "pto.vmi.group_store requires a supported UB destination and a table-"
        "supported value layout lowering through one-block vsstb, full-chunk "
        "vsts, or deinterleaved vstsx2 (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIStrideStoreOp(Operation *op) {
  if (auto store = dyn_cast<VMIStrideStoreOp>(op)) {
    return verifySupportedShapeOp(
        store, checkSupportedStrideStoreShape,
        "pto.vmi.stride_store lowers through pto.vsstb only for one contiguous "
        "physical value/mask chunk and a supported UB destination (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIScatterOp(Operation *op) {
  if (auto scatter = dyn_cast<VMIScatterOp>(op)) {
    return verifySupportedShapeOp(
        scatter, checkSupportedScatterShape,
        "pto.vmi.scatter lowers through pto.vscatter only with a UB pointer "
        "destination, contiguous full physical chunks, 32-bit value elements, "
        "i32 indices, and b32 masks (");
  }
  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIStructuredStoreOp(Operation *op) {
  if (auto maskedStore = verifySupportedVMIStructuredMaskedStoreOp(op);
      maskedStore.has_value()) {
    return maskedStore;
  }
  if (auto result = verifySupportedVMIInterleaveStoreOp(op);
      result.has_value()) {
    return result;
  }
  if (auto result = verifySupportedVMIGroupStoreOp(op); result.has_value()) {
    return result;
  }
  if (auto result = verifySupportedVMIStrideStoreOp(op); result.has_value()) {
    return result;
  }
  return verifySupportedVMIScatterOp(op);
}

static LogicalResult checkSupportedVMIStoreShape(VMIStoreOp op,
                                                 std::string *reason) {
  return checkSupportedStoreShape(
      cast<VMIVRegType>(op.getValue().getType()), op.getDestination(),
      op.getDestination().getType(), reason);
}

std::optional<WalkResult> verifySupportedVMIMemoryStoreOp(Operation *op) {
  if (auto store = dyn_cast<VMIStoreOp>(op)) {
    return verifySupportedShapeOp(
        store, checkSupportedVMIStoreShape,
        "pto.vmi.store requires an 8/16/32-bit predicate-maskable element "
        "type and either full physical chunks or contiguous tail-store "
        "layout, with UB-backed destination (");
  }
  if (auto structuredResult = verifySupportedVMIStructuredStoreOp(op);
      structuredResult.has_value()) {
    return *structuredResult;
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIDeinterleaveLoadOp(
    Operation *op) {
  if (auto load = dyn_cast<VMIDeinterleaveLoadOp>(op)) {
    return verifySupportedShapeOp(
        load, checkSupportedDeinterleaveLoadShape,
        "pto.vmi.deinterleave_load lowers through pto.vldsx2 only for "
        "matching contiguous full low/high result chunks with a supported "
        "UB source and 8/16/32-bit element type (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIStrideLoadOp(Operation *op) {
  if (auto load = dyn_cast<VMIStrideLoadOp>(op)) {
    return verifySupportedShapeOp(
        load, checkSupportedStrideLoadShape,
        "pto.vmi.stride_load lowers through pto.vsldb only for one "
        "contiguous physical result/mask chunk and a supported UB source (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIGroupLoadOp(Operation *op) {
  if (auto load = dyn_cast<VMIGroupLoadOp>(op)) {
    return verifySupportedShapeOp(
        load, checkSupportedGroupLoadShape,
        "pto.vmi.group_load requires contiguous full result chunks, a "
        "supported UB source, and num_groups deriving a group size aligned "
        "to physical chunks (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIGroupSlotLoadOp(
    Operation *op) {
  if (auto load = dyn_cast<VMIGroupSlotLoadOp>(op)) {
    return verifySupportedShapeOp(
        load, checkSupportedGroupSlotLoadShape,
        "pto.vmi.group_slot_load requires explicit group_slots result layout "
        "matching num_groups, a supported UB pointer source, and either "
        "slots=8 with constant unit source_group_stride or slots=1 row-local "
        "lowering (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIGroupBroadcastLoadOp(
    Operation *op) {
  if (auto load = dyn_cast<VMIGroupBroadcastLoadOp>(op)) {
    return verifySupportedShapeOp(
        load, checkSupportedGroupBroadcastLoadShape,
        "pto.vmi.group_broadcast_load requires either the BRC full-group "
        "chunk form, the E2B packet form for b16/b32 direct or split group "
        "size, or the generic group-slot-load then group-broadcast fallback "
        "with supported UB pointer source and source_group_stride (");
  }
  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIStructuredLoadOp(Operation *op) {
  if (auto result = verifySupportedVMIDeinterleaveLoadOp(op);
      result.has_value()) {
    return result;
  }
  if (auto result = verifySupportedVMIStrideLoadOp(op); result.has_value()) {
    return result;
  }
  if (auto result = verifySupportedVMIGroupLoadOp(op); result.has_value()) {
    return result;
  }
  if (auto result = verifySupportedVMIGroupSlotLoadOp(op);
      result.has_value()) {
    return result;
  }
  return verifySupportedVMIGroupBroadcastLoadOp(op);
}

std::optional<WalkResult> verifySupportedVMIMemoryLoadOp(
    Operation *op, bool enableStableGatherMaskedLoad) {
  if (auto load = dyn_cast<VMILoadOp>(op)) {
    return emitMemoryUnsupported(
        op, "pto.vmi.load", cast<VMIVRegType>(load.getResult().getType()),
        load.getSource(), getConstantIndexValue(load.getOffset()));
  }
  if (auto structuredResult = verifySupportedVMIStructuredLoadOp(op);
      structuredResult.has_value()) {
    return *structuredResult;
  }
  if (auto advancedResult = verifySupportedVMIMemoryAdvancedLoadOp(
          op, enableStableGatherMaskedLoad);
      advancedResult.has_value()) {
    return *advancedResult;
  }
  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIMemoryOp(
    Operation *op, bool enableStableGatherMaskedLoad) {
  if (auto loadResult = verifySupportedVMIMemoryLoadOp(
          op, enableStableGatherMaskedLoad);
      loadResult.has_value()) {
    return *loadResult;
  }
  return verifySupportedVMIMemoryStoreOp(op);
}

std::optional<WalkResult> verifySupportedVMIEnsureLayoutOp(Operation *op) {
  if (auto ensure = dyn_cast<VMIEnsureLayoutOp>(op)) {
    auto sourceType = cast<VMIVRegType>(ensure.getSource().getType());
    auto resultType = cast<VMIVRegType>(ensure.getResult().getType());
    std::string reason;
    VMILayoutSupport supports;
    if (succeeded(supports.getEnsureLayoutFact(sourceType, resultType,
                                               &reason))) {
      return WalkResult::advance();
    }

    emitEnsureLayoutMaterializationError(ensure, sourceType, resultType,
                                         reason);
    return WalkResult::interrupt();
  }

  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIMaskLayoutOp(Operation *op) {
  if (auto ensure = dyn_cast<VMIEnsureMaskLayoutOp>(op)) {
    auto sourceType = cast<VMIMaskType>(ensure.getSource().getType());
    auto resultType = cast<VMIMaskType>(ensure.getResult().getType());
    std::string reason;
    VMILayoutSupport supports;
    if (succeeded(supports.getEnsureMaskLayoutFact(sourceType, resultType,
                                                   &reason))) {
      return WalkResult::advance();
    }

    ensure.emitError()
        << kVMIDiagUnsupportedPrefix
        << "pto.vmi.ensure_mask_layout cannot materialize the requested mask "
           "layout conversion ("
        << reason
        << "); partial/tail predicate layout materialization requires an "
           "explicit packing plan";
    return WalkResult::interrupt();
  }

  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIMaskGranularityOp(Operation *op) {
  if (auto ensure = dyn_cast<VMIEnsureMaskGranularityOp>(op)) {
    auto sourceType = cast<VMIMaskType>(ensure.getSource().getType());
    auto resultType = cast<VMIMaskType>(ensure.getResult().getType());
    bool identity = sourceType.getGranularity() == resultType.getGranularity() &&
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

  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMILayoutOp(Operation *op) {
  if (auto result = verifySupportedVMIEnsureLayoutOp(op);
      result.has_value()) {
    return *result;
  }
  if (auto result = verifySupportedVMIMaskLayoutOp(op); result.has_value()) {
    return *result;
  }
  return verifySupportedVMIMaskGranularityOp(op);
}

template <typename MaskCheck>
WalkResult verifySupportedCompareValue(Operation *op, StringRef opName,
                                       VMIVRegType lhsType, MaskCheck checkMaskable,
                                       LogicalResult predicateCheck) {
  WalkResult physical = checkMaskable(op, opName, lhsType);
  if (physical.wasInterrupted()) {
    return physical;
  }
  if (succeeded(predicateCheck)) {
    return WalkResult::advance();
  }
  return WalkResult::interrupt();
}

template <typename MaskCheck>
std::optional<WalkResult> verifySupportedVMICompareOp(Operation *op,
                                                      MaskCheck checkMaskable) {
  if (auto cmpf = dyn_cast<VMICmpFOp>(op)) {
    return verifySupportedCompareValue(
        op, "pto.vmi.cmpf", cast<VMIVRegType>(cmpf.getLhs().getType()),
        checkMaskable,
        checkSupportedComparePredicate<VMICmpFOp>(op, cmpf.getPredicate()));
  }

  if (auto cmpi = dyn_cast<VMICmpIOp>(op)) {
    return verifySupportedCompareValue(
        op, "pto.vmi.cmpi", cast<VMIVRegType>(cmpi.getLhs().getType()),
        checkMaskable,
        checkSupportedComparePredicate<VMICmpIOp>(op, cmpi.getPredicate()));
  }

  return std::nullopt;
}

template <typename VecScalarOp, typename MaskableCheck>
WalkResult verifySupportedVecScalarOp(VecScalarOp op, StringRef opName,
                                      MaskableCheck checkMaskable) {
  bool requiresPassthru =
      op.getPmode().has_value() && *op.getPmode() == "merge";
  if (requiresPassthru) {
    op.emitError() << kVMIDiagUnsupportedPrefix << opName
                   << " with pmode=merge requires an explicit passthru lowering";
    return WalkResult::interrupt();
  }
  return checkMaskable(op, opName,
                       cast<VMIVRegType>(op.getResult().getType()));
}

template <typename MaskableOp, typename MaskableCheck>
WalkResult verifySupportedMaskableOp(MaskableOp op, StringRef opName,
                                     MaskableCheck checkMaskable) {
  return checkMaskable(op.getOperation(), opName,
                       cast<VMIVRegType>(op.getResult().getType()));
}

WalkResult emitMaskableUnsupported(Operation *op, StringRef opName,
                                   VMIVRegType type) {
  std::string reason;
  if (succeeded(checkSupportedMaskableVReg(type, &reason))) {
    return WalkResult::advance();
  }
  op->emitError()
      << kVMIDiagUnsupportedPrefix << opName
      << " direct lowering requires physical vreg parts with b8/b16/b32 "
         "predicate masks ("
      << reason << ")";
  return WalkResult::interrupt();
}

template <typename MaskableCheck>
static std::optional<WalkResult> verifySupportedVMIUnaryBinaryArithmeticOp(
    Operation *op, MaskableCheck check) {
#define PTO_VERIFY_MASKABLE(Op, Name)                                      \
  if (auto value = dyn_cast<Op>(op)) {                                    \
    return verifySupportedMaskableOp(value, Name, check);                  \
  }
  PTO_VERIFY_MASKABLE(VMIAddFOp, "pto.vmi.addf");
  PTO_VERIFY_MASKABLE(VMIAddIOp, "pto.vmi.addi");
  PTO_VERIFY_MASKABLE(VMISubFOp, "pto.vmi.subf");
  PTO_VERIFY_MASKABLE(VMISubIOp, "pto.vmi.subi");
  PTO_VERIFY_MASKABLE(VMIMulFOp, "pto.vmi.mulf");
  PTO_VERIFY_MASKABLE(VMIMulIOp, "pto.vmi.muli");
  PTO_VERIFY_MASKABLE(VMIDivFOp, "pto.vmi.divf");
  PTO_VERIFY_MASKABLE(VMIMinFOp, "pto.vmi.minf");
  PTO_VERIFY_MASKABLE(VMIMinIOp, "pto.vmi.mini");
  PTO_VERIFY_MASKABLE(VMIMaxFOp, "pto.vmi.maxf");
  PTO_VERIFY_MASKABLE(VMIMaxIOp, "pto.vmi.maxi");
  PTO_VERIFY_MASKABLE(VMINegFOp, "pto.vmi.negf");
  PTO_VERIFY_MASKABLE(VMINegIOp, "pto.vmi.negi");
  PTO_VERIFY_MASKABLE(VMIAbsFOp, "pto.vmi.absf");
  PTO_VERIFY_MASKABLE(VMIAbsIOp, "pto.vmi.absi");
  PTO_VERIFY_MASKABLE(VMISqrtOp, "pto.vmi.sqrt");
  PTO_VERIFY_MASKABLE(VMIExpOp, "pto.vmi.exp");
  PTO_VERIFY_MASKABLE(VMILnOp, "pto.vmi.ln");
  PTO_VERIFY_MASKABLE(VMIAndIOp, "pto.vmi.andi");
  PTO_VERIFY_MASKABLE(VMIOrIOp, "pto.vmi.ori");
  PTO_VERIFY_MASKABLE(VMIXOrIOp, "pto.vmi.xori");
  PTO_VERIFY_MASKABLE(VMIShLIOp, "pto.vmi.shli");
  PTO_VERIFY_MASKABLE(VMIShRUIOp, "pto.vmi.shrui");
  PTO_VERIFY_MASKABLE(VMIShRSIOp, "pto.vmi.shrsi");
  PTO_VERIFY_MASKABLE(VMINotOp, "pto.vmi.not");
  PTO_VERIFY_MASKABLE(VMISelectOp, "pto.vmi.select");
#undef PTO_VERIFY_MASKABLE
  return std::nullopt;
}

template <typename MaskableCheck>
static std::optional<WalkResult> verifySupportedVMIVecScalarArithmeticOp(
    Operation *op, MaskableCheck check) {
  if (auto value = dyn_cast<VMIAddSOp>(op)) {
    return verifySupportedVecScalarOp(value, "pto.vmi.vadds", check);
  }
  if (auto value = dyn_cast<VMIMulSOp>(op)) {
    return verifySupportedVecScalarOp(value, "pto.vmi.vmuls", check);
  }
  if (auto value = dyn_cast<VMIMaxSOp>(op)) {
    return verifySupportedVecScalarOp(value, "pto.vmi.vmaxs", check);
  }
  if (auto value = dyn_cast<VMIMinSOp>(op)) {
    return verifySupportedVecScalarOp(value, "pto.vmi.vmins", check);
  }
  if (auto value = dyn_cast<VMIShlSOp>(op)) {
    return verifySupportedVecScalarOp(value, "pto.vmi.vshls", check);
  }
  if (auto value = dyn_cast<VMIShrSOp>(op)) {
    return verifySupportedVecScalarOp(value, "pto.vmi.vshrs", check);
  }
  return std::nullopt;
}

template <typename MaskableCheck>
std::optional<WalkResult> verifySupportedVMIArithmeticOp(Operation *op,
                                                         MaskableCheck check) {
  if (auto result = verifySupportedVMIUnaryBinaryArithmeticOp(op, check);
      result.has_value()) {
    return result;
  }
  return verifySupportedVMIVecScalarArithmeticOp(op, check);
}

template <typename ReduceOp>
WalkResult verifySupportedReduceOp(ReduceOp op, bool requiresReassoc,
                                   StringRef diagnostic) {
  std::string reason;
  if (succeeded(checkSupportedReduceShape(op, requiresReassoc, &reason))) {
    return WalkResult::advance();
  }
  op.emitError() << kVMIDiagUnsupportedPrefix << diagnostic << reason << ")";
  return WalkResult::interrupt();
}

template <typename GroupReduceOp>
WalkResult verifySupportedGroupReduceOp(GroupReduceOp op, StringRef diagnostic) {
  std::string reason;
  if (succeeded(checkSupportedGroupReduceShape(op, &reason))) {
    return WalkResult::advance();
  }
  op.emitError() << kVMIDiagUnsupportedPrefix << diagnostic << reason << ")";
  return WalkResult::interrupt();
}

template <typename ShapeOp, typename ShapeCheck>
WalkResult verifySupportedShapeOp(ShapeOp op, ShapeCheck check,
                                  StringRef diagnostic) {
  std::string reason;
  if (succeeded(check(op, &reason))) {
    return WalkResult::advance();
  }
  op.emitError() << kVMIDiagUnsupportedPrefix << diagnostic << reason << ")";
  return WalkResult::interrupt();
}

WalkResult verifySupportedConstantMaskOp(VMIConstantMaskOp op) {
  std::string reason;
  if (succeeded(computeConstantMaskMaterialization(op, &reason))) {
    return WalkResult::advance();
  }
  op.emitError()
      << kVMIDiagUnsupportedPrefix
      << "pto.vmi.constant_mask requires a dense bool constant with concrete "
         "layout and b8/b16/b32 granularity ("
      << reason << ")";
  return WalkResult::interrupt();
}

template <typename ChannelOp, typename ShapeCheck>
WalkResult verifySupportedChannelOp(ChannelOp op, int64_t channels,
                                     ShapeCheck check, StringRef supportedText,
                                     StringRef shapeText) {
  std::string reason;
  if (succeeded(check(op, &reason))) {
    return WalkResult::advance();
  }
  if (channels != 2 && channels != 4) {
    op.emitError() << kVMIDiagUnsupportedPrefix << supportedText;
  } else {
    op.emitError() << kVMIDiagUnsupportedPrefix << shapeText << reason << ")";
  }
  return WalkResult::interrupt();
}

static std::optional<WalkResult> verifySupportedVMIFloatConversionOp(
    Operation *op) {
  if (auto fptosi = dyn_cast<VMIFPToSIOp>(op)) {
    return verifySupportedShapeOp(
        fptosi, checkSupportedFPToSIShape,
        "pto.vmi.fptosi supports fp-to-signed-int conversion pairs listed in "
        "the VPTO vcvt contract; check lookupVMIFpToSiContract (");
  }
  if (auto fptoui = dyn_cast<VMIFPToUIOp>(op)) {
    return verifySupportedShapeOp(
        fptoui, checkSupportedFPToUIShape,
        "pto.vmi.fptoui supports fp-to-unsigned-int conversion pairs listed "
        "in the VPTO vcvt contract (e.g. f16 → u8); "
        "check lookupVMIFpToUIContract (");
  }
  if (auto sitofp = dyn_cast<VMISIToFPOp>(op)) {
    return verifySupportedShapeOp(
        sitofp, checkSupportedSIToFPShape,
        "pto.vmi.sitofp supports si32->f32 or si8->f16 conversion shapes (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIIntegerConversionOp(
    Operation *op) {
  if (auto extsi = dyn_cast<VMIExtSIOp>(op)) {
    return verifySupportedShapeOp(
        extsi, checkSupportedExtSIShape,
        "pto.vmi.extsi supports contiguous signed/signless 8-bit or 16-bit "
        "integer physical source chunks to 2x/4x wider integer "
        "deinterleaved results, or matching group_slots(num_groups=G, "
        "slots=1) layouts and natural group_slots(num_groups=G, slots=8, "
        "lane_stride=2/4) to group_slots(num_groups=G, slots=8) widening "
        "layouts (");
  }
  if (auto extui = dyn_cast<VMIExtUIOp>(op)) {
    return verifySupportedShapeOp(
        extui, checkSupportedExtUIShape,
        "pto.vmi.extui supports contiguous unsigned 8-bit or 16-bit integer "
        "physical source chunks to 2x/4x wider unsigned integer "
        "deinterleaved results, or matching group_slots(num_groups=G, "
        "slots=1) layouts and natural group_slots(num_groups=G, slots=8, "
        "lane_stride=2/4) to group_slots(num_groups=G, slots=8) widening "
        "layouts (");
  }
  if (auto trunci = dyn_cast<VMITruncIOp>(op)) {
    return verifySupportedShapeOp(
        trunci, checkSupportedTruncIShape,
        "pto.vmi.trunci supports integer deinterleaved source layouts whose "
        "factor is the 2x/4x narrowing multiple of the contiguous or "
        "deinterleaved result layout factor, or matching group_slots "
        "layouts and natural slots=8 narrowing layouts (");
  }
  if (auto bitcast = dyn_cast<VMIBitcastOp>(op)) {
    return verifySupportedShapeOp(
        bitcast, checkSupportedBitcastShape,
        "pto.vmi.bitcast requires matching source/result layouts with "
        "width-changing forms restricted to supported layout table rows (");
  }
  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIConversionOp(Operation *op) {
  if (auto result = verifySupportedVMIFloatConversionOp(op);
      result.has_value()) {
    return result;
  }
  return verifySupportedVMIIntegerConversionOp(op);
}

template <typename CarryOp, typename ShapeCheck>
std::optional<WalkResult> verifyAddCarryShape(CarryOp op, ShapeCheck check,
                                               StringRef diagnostic) {
  std::string reason;
  if (succeeded(check(op, &reason))) {
    return WalkResult::advance();
  }
  op.emitError() << kVMIDiagUnsupportedPrefix << diagnostic << reason << ")";
  return WalkResult::interrupt();
}

std::optional<WalkResult> verifySupportedVMIAddCarryOp(Operation *op) {
  if (auto addc = dyn_cast<VMIVaddcOp>(op)) {
    return verifyAddCarryShape(
        addc, checkSupportedVMIAddcShape,
        "pto.vmi.vaddc requires matching 32-bit data and b32 mask parts (");
  }
  if (auto addcs = dyn_cast<VMIVaddcsOp>(op)) {
    return verifyAddCarryShape(
        addcs, checkSupportedVMIAddcsShape,
        "pto.vmi.vaddcs requires matching 32-bit data and b32 mask parts (");
  }
  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIMultiplyLongOp(Operation *op) {
  if (auto vmull = dyn_cast<VMIVmullOp>(op)) {
    std::string reason;
    if (succeeded(checkSupportedVmullShape(vmull, &reason))) {
      return WalkResult::advance();
    }
    vmull.emitError()
        << kVMIDiagUnsupportedPrefix
        << "pto.vmi.vmull requires equal 64/128/256-lane i32/ui32 data "
           "ports, a matching b32 mask, and contiguous or deinterleaved "
           "factor-2/factor-4 lane_stride=1 layout ("
        << reason << ")";
    return WalkResult::interrupt();
  }
  return std::nullopt;
}

template <typename SpecialOp, typename ShapeCheck>
std::optional<WalkResult> verifySpecialUnaryShape(
    SpecialOp op, ShapeCheck check, StringRef diagnostic) {
  std::string reason;
  if (succeeded(check(op, &reason))) {
    return WalkResult::advance();
  }
  op.emitError() << kVMIDiagUnsupportedPrefix << diagnostic << reason << ")";
  return WalkResult::interrupt();
}

std::optional<WalkResult> verifySupportedVMISpecialUnaryOp(Operation *op) {
  if (auto relu = dyn_cast<VMIReluOp>(op)) {
    return verifySpecialUnaryShape(
        relu, checkSupportedReluShape,
        "pto.vmi.relu direct lowering requires physical vreg parts with b32 "
        "predicates for si32 or matching b16/b32 predicates for f16/f32 (");
  }
  if (auto vselr = dyn_cast<VMIVselrOp>(op)) {
    return verifySpecialUnaryShape(
        vselr, checkSupportedVselrShape,
        "pto.vmi.vselr supports only contiguous lane_stride=1 layouts with "
        "N=64, 128, or 256 for 8-bit, N=64 or 128 for 16-bit, or N=64 for "
        "32-bit elements (");
  }
  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMISpecialOp(Operation *op) {
  if (auto result = verifySupportedVMIAddCarryOp(op); result.has_value()) {
    return *result;
  }
  if (auto result = verifySupportedVMIMultiplyLongOp(op); result.has_value()) {
    return *result;
  }
  return verifySupportedVMISpecialUnaryOp(op);
}

static std::optional<WalkResult> verifySupportedVMINormalFloatReductionOp(
    Operation *op) {
  if (auto reduce = dyn_cast<VMIReduceAddFOp>(op)) {
    return verifySupportedReduceOp(
        reduce, true,
        "pto.vmi.reduce_addf lowers through pto.vcadd only with reassoc, "
        "f32 contiguous full source chunks, matching mask chunks, and one "
        "init/result chunk (");
  }
  if (auto reduce = dyn_cast<VMIReduceMaxFOp>(op)) {
    return verifySupportedReduceOp(
        reduce, false,
        "pto.vmi.reduce_maxf lowers through pto.vcmax only for f16/f32 "
        "contiguous full source chunks with matching mask chunks and one "
        "init/result chunk (");
  }
  if (auto reduce = dyn_cast<VMIReduceMinFOp>(op)) {
    return verifySupportedReduceOp(
        reduce, false,
        "pto.vmi.reduce_minf lowers through pto.vcmin only for f16/f32 "
        "contiguous full source chunks with matching mask chunks and one "
        "init/result chunk (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMINormalIntegerReductionOp(
    Operation *op) {
  if (auto reduce = dyn_cast<VMIReduceAddIOp>(op)) {
    return verifySupportedReduceOp(
        reduce, false,
        "pto.vmi.reduce_addi lowers through pto.vcadd only for contiguous "
        "full 32-bit integer source chunks with matching mask chunks and one "
        "init/result chunk (");
  }
  if (auto reduce = dyn_cast<VMIReduceMaxIOp>(op)) {
    return verifySupportedReduceOp(
        reduce, false,
        "pto.vmi.reduce_maxi lowers through pto.vcmax only for contiguous "
        "full integer source chunks with matching mask chunks and one "
        "init/result chunk (");
  }
  if (auto reduce = dyn_cast<VMIReduceMinIOp>(op)) {
    return verifySupportedReduceOp(
        reduce, false,
        "pto.vmi.reduce_mini lowers through pto.vcmin only for contiguous "
        "full integer source chunks with matching mask chunks and one "
        "init/result chunk (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMINormalReductionOp(
    Operation *op) {
  if (auto result = verifySupportedVMINormalFloatReductionOp(op);
      result.has_value()) {
    return result;
  }
  return verifySupportedVMINormalIntegerReductionOp(op);
}

static std::optional<WalkResult> verifySupportedVMIGroupFloatReductionOp(
    Operation *op) {
  if (auto reduce = dyn_cast<VMIGroupReduceAddFOp>(op)) {
    return verifySupportedGroupReduceOp(
        reduce,
        "pto.vmi.group_reduce_addf lowers through pto.vcgadd for 32B blocks "
        "or through pto.vcadd for contiguous full source/mask chunks, "
        "#pto.vmi.layout<num_groups = G, slots = K> result chunks, and "
        "num_groups deriving a group size aligned to physical chunks (");
  }
  if (auto reduce = dyn_cast<VMIGroupReduceMaxFOp>(op)) {
    return verifySupportedGroupReduceOp(
        reduce,
        "pto.vmi.group_reduce_maxf lowers through pto.vcgmax/vmax for 32B "
        "blocks or through pto.vcmax for contiguous full chunks, matching "
        "source/mask chunks, #pto.vmi.layout<num_groups = G, slots = K> "
        "result chunks, and num_groups deriving a group size aligned to "
        "physical chunks (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIGroupIntegerReductionOp(
    Operation *op) {
  if (auto reduce = dyn_cast<VMIGroupReduceAddIOp>(op)) {
    return verifySupportedGroupReduceOp(
        reduce,
        "pto.vmi.group_reduce_addi lowers through pto.vcgadd/vadd for "
        "supported 32B block classes or through an internal widening "
        "pto.vcadd path for aligned full chunks (");
  }
  if (auto reduce = dyn_cast<VMIGroupReduceMaxIOp>(op)) {
    return verifySupportedGroupReduceOp(
        reduce,
        "pto.vmi.group_reduce_maxi lowers through pto.vcgmax/vmax for "
        "supported 32B block classes or through pto.vcmax for aligned full "
        "chunks (");
  }
  if (auto reduce = dyn_cast<VMIGroupReduceMinIOp>(op)) {
    return verifySupportedGroupReduceOp(
        reduce,
        "pto.vmi.group_reduce_mini lowers through pto.vcgmin/vmin for "
        "supported 32B block classes or through pto.vcmin for aligned full "
        "chunks (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIGroupReductionOp(
    Operation *op) {
  if (auto result = verifySupportedVMIGroupFloatReductionOp(op);
      result.has_value()) {
    return result;
  }
  return verifySupportedVMIGroupIntegerReductionOp(op);
}

std::optional<WalkResult> verifySupportedVMIReductionOp(Operation *op) {
  if (auto normalResult = verifySupportedVMINormalReductionOp(op);
      normalResult.has_value()) {
    return *normalResult;
  }
  return verifySupportedVMIGroupReductionOp(op);
}

std::optional<WalkResult> verifySupportedVMIFloatOp(Operation *op) {
  if (auto fma = dyn_cast<VMIFmaOp>(op)) {
    return verifySupportedShapeOp(
        fma, checkSupportedFmaShape,
        "pto.vmi.fma lowers through pto.vmula only for f16/bf16/f32 element "
        "types (");
  }
  if (auto extf = dyn_cast<VMIExtFOp>(op)) {
    return verifySupportedShapeOp(
        extf, checkSupportedExtFShape,
        "pto.vmi.extf supports contiguous 16-bit float-like or fp8-like "
        "physical source chunks to f32 deinterleaved=2/4 results; "
        "partial/tail is allowed only when source padding maps to result "
        "padding (");
  }
  if (auto truncf = dyn_cast<VMITruncFOp>(op)) {
    return verifySupportedShapeOp(
        truncf, checkSupportedTruncFShape,
        "pto.vmi.truncf supports f32/f16/bf16 source narrowing (dense "
        "EvenOdd, Packed4, or f32 group_slots(num_groups=G, slots=1) to "
        "f16 group_slots(num_groups=G, slots=1)); non-f32 sources currently "
        "require dense contiguous layouts (");
  }
  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIBasicMiscOp(Operation *op) {
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
  if (auto broadcast = dyn_cast<VMIBroadcastOp>(op)) {
    return emitMaskableUnsupported(
        op, "pto.vmi.broadcast",
        cast<VMIVRegType>(broadcast.getResult().getType()));
  }
  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIHistogramOp(Operation *op) {
  if (auto broadcast = dyn_cast<VMIGroupBroadcastOp>(op)) {
    return verifySupportedShapeOp(
        broadcast, checkSupportedGroupBroadcastShape,
        "pto.vmi.group_broadcast requires #pto.vmi.layout<num_groups = G, "
        "slots = K> source, a dense full result layout, and num_groups "
        "deriving a group size that divides or is a multiple of physical "
        "chunk lanes (");
  }
  if (auto hist = dyn_cast<VMIVdhistOp>(op)) {
    return verifySupportedShapeOp(
        hist, checkSupportedVdhistShape,
        "pto.vmi.vdhist requires contiguous Nx{ui8|i8} source, contiguous "
        "b8 mask, and contiguous 256x{ui16|i16} acc/result (");
  }
  if (auto hist = dyn_cast<VMIVchistOp>(op)) {
    return verifySupportedShapeOp(
        hist, checkSupportedVchistShape,
        "pto.vmi.vchist requires contiguous Nx{ui8|i8} source, contiguous "
        "b8 mask, and contiguous 256x{ui16|i16} acc/result (");
  }
  return std::nullopt;
}

template <typename CompressionOp, typename ShapeCheck>
std::optional<WalkResult> verifyCompressionShape(
    CompressionOp op, ShapeCheck check, StringRef diagnostic) {
  std::string reason;
  if (succeeded(check(op, &reason))) {
    return WalkResult::advance();
  }
  op.emitError() << kVMIDiagUnsupportedPrefix << diagnostic << reason << ")";
  return WalkResult::interrupt();
}

std::optional<WalkResult> verifySupportedVMICompressionOp(Operation *op) {
  if (auto activePrefix = dyn_cast<VMIActivePrefixIndexOp>(op)) {
    return verifyCompressionShape(
        activePrefix, checkSupportedActivePrefixIndexShape,
        "pto.vmi.active_prefix_index lowers through pto.vusqz only for one "
        "contiguous physical chunk (");
  }
  if (auto compress = dyn_cast<VMICompressOp>(op)) {
    return verifyCompressionShape(
        compress, checkSupportedCompressShape,
        "pto.vmi.compress lowers through pto.vsqz only for one contiguous "
        "full physical chunk (");
  }
  if (auto compressStore = dyn_cast<VMICompressStoreOp>(op)) {
    return verifyCompressionShape(
        compressStore, checkSupportedCompressStoreShape,
        "pto.vmi.compress_store lowers through pto.vsqz + pto.vstur only for "
        "one contiguous full physical chunk with a UB pointer destination (");
  }
  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIMiscOp(Operation *op) {
  if (auto result = verifySupportedVMIBasicMiscOp(op); result.has_value()) {
    return *result;
  }
  if (auto result = verifySupportedVMIHistogramOp(op); result.has_value()) {
    return *result;
  }
  return verifySupportedVMICompressionOp(op);
}

std::optional<WalkResult>
verifySupportedVMIChannelShuffleOp(Operation *op) {
  if (auto split = dyn_cast<VMIChannelSplitOp>(op)) {
    return verifySupportedChannelOp(
        split, split.getNumResults(), checkSupportedChannelSplitShape,
        "pto.vmi.channel_split supports only 2 or 4 channels",
        "pto.vmi.channel_split requires source layout to be contiguous or "
        "matching deinterleaved channel layout, every result layout to be "
        "contiguous, and complete physical channel groups (");
  }
  if (auto merge = dyn_cast<VMIChannelMergeOp>(op)) {
    return verifySupportedChannelOp(
        merge, merge.getInputs().size(), checkSupportedChannelMergeShape,
        "pto.vmi.channel_merge supports only 2 or 4 channels",
        "pto.vmi.channel_merge requires every input layout to be contiguous "
        "and result layout to be contiguous or matching deinterleaved "
        "channel layout, with complete physical channel groups (");
  }
  if (auto shuffle = dyn_cast<VMIShuffleOp>(op)) {
    std::string reason;
    if (succeeded(computeShuffleForwardingSourceParts(shuffle, &reason))) {
      return WalkResult::advance();
    }
    std::string splatReason;
    if (succeeded(computeShuffleLane0SplatSourcePart(shuffle, &splatReason))) {
      return WalkResult::advance();
    }
    std::string vselrReason;
    if (succeeded(computeShuffleVselrPlans(shuffle, &vselrReason))) {
      return WalkResult::advance();
    }

    shuffle.emitError()
        << kVMIDiagUnsupportedPrefix
        << "pto.vmi.shuffle requires physical chunk forwarding or "
           "lane0 splat or vci-materializable vselr indices (forwarding: "
        << reason << "; lane0 splat: " << splatReason
        << "; vselr: " << vselrReason << ")";
    return WalkResult::interrupt();
  }
  if (auto constantMask = dyn_cast<VMIConstantMaskOp>(op)) {
    return verifySupportedConstantMaskOp(constantMask);
  }
  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIStandardOp(
    Operation *op, bool enableStableGatherMaskedLoad) {
  if (auto memoryResult = verifySupportedVMIMemoryOp(
          op, enableStableGatherMaskedLoad);
      memoryResult.has_value()) {
    return *memoryResult;
  }
  if (auto layoutResult = verifySupportedVMILayoutOp(op);
      layoutResult.has_value()) {
    return *layoutResult;
  }
  auto compareResult = verifySupportedVMICompareOp(
      op, emitMaskableUnsupported);
  if (compareResult.has_value()) {
    return *compareResult;
  }
  if (auto miscResult = verifySupportedVMIMiscOp(op);
      miscResult.has_value()) {
    return *miscResult;
  }
  if (auto arithmeticResult = verifySupportedVMIArithmeticOp(
          op, emitMaskableUnsupported);
      arithmeticResult.has_value()) {
    return *arithmeticResult;
  }
  if (auto specialResult = verifySupportedVMISpecialOp(op);
      specialResult.has_value()) {
    return *specialResult;
  }
  if (auto reductionResult = verifySupportedVMIReductionOp(op);
      reductionResult.has_value()) {
    return *reductionResult;
  }
  if (auto floatResult = verifySupportedVMIFloatOp(op);
      floatResult.has_value()) {
    return *floatResult;
  }
  return verifySupportedVMIConversionOp(op);
}

static WalkResult verifySupportedVMIToVPTOOp(
    Operation *op, bool enableStableGatherMaskedLoad) {
  if (auto standardResult = verifySupportedVMIStandardOp(
          op, enableStableGatherMaskedLoad);
      standardResult.has_value()) {
    return *standardResult;
  }
  if (auto channelShuffleResult = verifySupportedVMIChannelShuffleOp(op);
      channelShuffleResult.has_value()) {
    return *channelShuffleResult;
  }
  return WalkResult::advance();
}

LogicalResult
verifySupportedVMIToVPTOOps(ModuleOp module,
                            bool enableStableGatherMaskedLoad) {
  WalkResult result = module.walk(
      [&enableStableGatherMaskedLoad](Operation *op) {
        return verifySupportedVMIToVPTOOp(op, enableStableGatherMaskedLoad);
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
    if (failed(applyPartialOneToNConversion(module, typeConverter,
                                            std::move(patterns)))) {
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
