// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
//===- VMIToVPTOConversionInternals.inc - VMIToVPTO internals -*- C++ -*-===//
//===----------------------------------------------------------------------===//


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

  VMILayoutAttr layout = type.getLayoutAttr();
  int64_t laneStride = layout && layout.hasLaneStride() ? layout.getLaneStride()
                                                        : 1;
  int64_t physicalBits = bits * laneStride;
  StringRef physicalGranularity = getMaskGranularityForBits(physicalBits);
  if (physicalGranularity.empty()) {
    return failure();
  }
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
  if (failed(converted)) {
    op->emitError() << "expected converted result types";
    return;
  }
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
  if (rhs <= 0) {
    return 0;
  }
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


