// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// the CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. This software is provided on an "AS IS" BASIS.

#include "VPTOLLVMEmitterInternal.h"

#include "PTO/IR/VPTOMemoryDist.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::pto {

Value getI64Constant(OpBuilder &builder, Location loc, uint64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(value));
}

Value getI32Constant(OpBuilder &builder, Location loc, uint64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(value));
}

Value packShiftedI64Fields(OpBuilder &builder, Location loc, Value config,
                            ArrayRef<std::pair<Value, uint64_t>> fields) {
  for (auto [value, amount] : fields) {
    Value shift = getI64Constant(builder, loc, amount);
    Value shifted = builder.create<arith::ShLIOp>(loc, value, shift);
    config = builder.create<arith::OrIOp>(loc, config, shifted);
  }
  return config;
}

Value packMaskedI64Fields(OpBuilder &builder, Location loc, Value config,
                           ArrayRef<std::pair<Value, uint64_t>> fields,
                           uint64_t mask) {
  Value maskValue = getI64Constant(builder, loc, mask);
  for (auto [value, amount] : fields) {
    Value masked = builder.create<arith::AndIOp>(loc, value, maskValue);
    Value shift = getI64Constant(builder, loc, amount);
    Value shifted = builder.create<arith::ShLIOp>(loc, masked, shift);
    config = builder.create<arith::OrIOp>(loc, config, shifted);
  }
  return config;
}

Value castIntegerLikeTo(Operation *anchor, Value value, Type targetType) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  if (value.getType() == targetType) {
    return value;
  }
  auto targetInt = dyn_cast<IntegerType>(targetType);
  if (value.getType().isIndex() && targetInt) {
    return builder.create<arith::IndexCastOp>(anchor->getLoc(), targetType, value);
  }
  if (auto sourceInt = dyn_cast<IntegerType>(value.getType())) {
    if (targetInt) {
      if (sourceInt.getWidth() < targetInt.getWidth()) {
        return builder.create<arith::ExtUIOp>(anchor->getLoc(), targetType, value);
      }
      if (sourceInt.getWidth() > targetInt.getWidth()) {
        return builder.create<arith::TruncIOp>(anchor->getLoc(), targetType, value);
      }
      return value;
    }
    if (targetType.isIndex()) {
      return builder.create<arith::IndexCastOp>(anchor->getLoc(), targetType, value);
    }
  }
  return {};
}

FailureOr<SmallVector<Value, 7>>
castIntegerLikeOperands(Operation *anchor, ValueRange operands,
                        ArrayRef<unsigned> indices, Type targetType) {
  SmallVector<Value, 7> converted;
  converted.reserve(indices.size());
  for (unsigned index : indices) {
    if (index >= operands.size()) {
      return failure();
    }
    Value value = castIntegerLikeTo(anchor, operands[index], targetType);
    if (!value) {
      return failure();
    }
    converted.push_back(value);
  }
  return converted;
}

FailureOr<Value> reinterpretPointerToAddrSpace(Operation *anchor, Value value,
                                                unsigned targetAddressSpace) {
  auto sourcePtrType = dyn_cast<LLVM::LLVMPointerType>(value.getType());
  if (!sourcePtrType) {
    return failure();
  }
  if (sourcePtrType.getAddressSpace() == targetAddressSpace) {
    return value;
  }
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();
  Value asInt = builder.create<LLVM::PtrToIntOp>(loc, builder.getI64Type(), value);
  Type targetPtrType =
      LLVM::LLVMPointerType::get(anchor->getContext(), targetAddressSpace);
  return builder.create<LLVM::IntToPtrOp>(loc, targetPtrType, asInt).getResult();
}

FailureOr<SmallVector<Value, 2>> reinterpretPointerOperands(
    Operation *anchor, ArrayRef<Value> values, ArrayRef<unsigned> addressSpaces) {
  if (values.size() != addressSpaces.size()) {
    return failure();
  }
  SmallVector<Value, 2> converted;
  converted.reserve(values.size());
  for (auto [value, addressSpace] : llvm::zip(values, addressSpaces)) {
    FailureOr<Value> result =
        reinterpretPointerToAddrSpace(anchor, value, addressSpace);
    if (failed(result)) {
      return failure();
    }
    converted.push_back(*result);
  }
  return converted;
}

FailureOr<Value> packLoopPair(Operation *anchor, Value low, Value high) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Value lowI64 = castIntegerLikeTo(anchor, low, builder.getI64Type());
  Value highI64 = castIntegerLikeTo(anchor, high, builder.getI64Type());
  if (!lowI64 || !highI64) {
    return failure();
  }
  Value highShifted = builder.create<arith::ShLIOp>(
      anchor->getLoc(), highI64, getI64Constant(builder, anchor->getLoc(), 40));
  return builder.create<arith::OrIOp>(anchor->getLoc(), highShifted, lowI64)
      .getResult();
}

FailureOr<Value> packLoopSize(Operation *anchor, Value loop2, Value loop1) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Value loop2I64 = castIntegerLikeTo(anchor, loop2, builder.getI64Type());
  Value loop1I64 = castIntegerLikeTo(anchor, loop1, builder.getI64Type());
  if (!loop2I64 || !loop1I64) {
    return failure();
  }
  Value loop2Shifted = builder.create<arith::ShLIOp>(
      anchor->getLoc(), loop2I64, getI64Constant(builder, anchor->getLoc(), 21));
  return builder.create<arith::OrIOp>(anchor->getLoc(), loop2Shifted, loop1I64)
      .getResult();
}

std::string getLowPrecisionElementFragment(Type type) {
  if (pto::isPTOHiFloat8x2Type(type)) {
    return "hif8x2";
  }
  if (pto::isPTOHiFloat8Type(type)) {
    return "hif8";
  }
  if (isa<pto::F4E1M2x2Type>(type)) {
    return "f4e1m2x2";
  }
  if (isa<pto::F4E2M1x2Type>(type)) {
    return "f4e2m1x2";
  }
  if (pto::isPTOBF16x2Type(type)) {
    return "bf16x2";
  }
  if (pto::isPTOFloat8E4M3LikeType(type)) {
    return "f8e4m3";
  }
  if (pto::isPTOFloat8E5M2LikeType(type)) {
    return "f8e5m2";
  }
  return {};
}

std::string getElementTypeFragment(Type type) {
  if (type.isF16()) {
    return "f16";
  }
  if (type.isBF16()) {
    return "bf16";
  }
  if (type.isF32()) {
    return "f32";
  }
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return (intType.isUnsigned() ? "u" : "s") +
           std::to_string(intType.getWidth());
  }
  return {};
}

std::string getMemoryElementTypeFragment(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return "i" + std::to_string(intType.getWidth());
  }
  if (std::string elem = getElementTypeFragment(type); !elem.empty()) {
    return elem;
  }
  return getLowPrecisionElementFragment(type);
}

std::string getCopyElementFragment(Type elementType) {
  if (!elementType) {
    return {};
  }
  if (elementType.isF16()) {
    return "f16";
  }
  if (elementType.isBF16()) {
    return "bf16";
  }
  if (elementType.isF32()) {
    return "f32";
  }
  std::string typeText;
  llvm::raw_string_ostream os(typeText);
  elementType.print(os);
  os.flush();
  std::string lower = StringRef(typeText).lower();
  if (StringRef(lower).contains("e4m3")) {
    return "e4m3";
  }
  if (StringRef(lower).contains("e5m2")) {
    return "e5m2";
  }
  if (StringRef(lower).contains("e8m0")) {
    return "e8m0";
  }
  if (StringRef(lower).contains("hif8")) {
    return "hif8";
  }
  if (StringRef(lower).contains("e1m2x2") ||
      StringRef(lower).contains("e2m1x2")) {
    return "u8";
  }
  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    switch (intType.getWidth()) {
    case 8:
      return intType.isUnsigned() ? "u8" : "s8";
    case 16:
      return intType.isUnsigned() ? "u16" : "s16";
    case 32:
      return intType.isUnsigned() ? "u32" : "s32";
    default:
      return {};
    }
  }
  return {};
}

Type getElementTypeFromVectorLike(Type type) {
  if (auto vecType = dyn_cast<pto::VRegType>(type)) {
    return vecType.getElementType();
  }
  if (auto vecType = dyn_cast<VectorType>(type)) {
    return vecType.getElementType();
  }
  if (auto vecType = dyn_cast<LLVM::LLVMFixedVectorType>(type)) {
    return vecType.getElementType();
  }
  return {};
}

std::optional<int64_t> getElementCountFromVectorLike(Type type) {
  if (auto vecType = dyn_cast<pto::VRegType>(type)) {
    return vecType.getElementCount();
  }
  if (auto vecType = dyn_cast<VectorType>(type)) {
    if (vecType.getRank() != 1) {
      return std::nullopt;
    }
    return vecType.getShape().front();
  }
  if (auto vecType = dyn_cast<LLVM::LLVMFixedVectorType>(type)) {
    return vecType.getNumElements();
  }
  return std::nullopt;
}

bool isOnePointStoreDist(StringRef dist) {
  const auto *contract =
      lookupVPTOMemoryDist(VPTOMemoryOpFamily::Store, dist);
  return contract && contract->isOnePointStore();
}

VPTOTypeConverter::VPTOTypeConverter(MLIRContext *context) {
  addConversion([](Type type) { return type; });
  addConversion([](Type type) -> Type {
    Builder builder(type.getContext());
    return convertVPTOType(type, builder);
  });
  addSourceMaterialization(materializeVPTOCast);
  addTargetMaterialization(materializeVPTOCast);
}

Type getLowPrecisionLLVMType(Type type, MLIRContext *context) {
  if (pto::isPTOHiFloat8Type(type))
  {
    return LLVM::LLVMHiFloat8Type::get(context);
  }
  if (isa<pto::F4E1M2x2Type>(type))
  {
    return LLVM::LLVMFloat4E1M2x2Type::get(context);
  }
  if (isa<pto::F4E2M1x2Type>(type))
  {
    return LLVM::LLVMFloat4E2M1x2Type::get(context);
  }
  if (pto::isPTOFloat8E4M3LikeType(type))
  {
    return LLVM::LLVMFloat8E4M3Type::get(context);
  }
  if (pto::isPTOFloat8E5M2LikeType(type))
  {
    return LLVM::LLVMFloat8E5M2Type::get(context);
  }
  return {};
}

bool isLLVMExtensionVectorElementType(Type type) {
  return isa<LLVM::LLVMHiFloat8Type, LLVM::LLVMFloat8E4M3Type,
             LLVM::LLVMFloat8E5M2Type, LLVM::LLVMFloat4E1M2x2Type,
             LLVM::LLVMFloat4E2M1x2Type>(type);
}

Type getLLVMCompatibleVectorType(ArrayRef<int64_t> shape,
                                        Type elementType,
                                        ArrayRef<bool> scalableDims = {}) {
  if (shape.size() == 1 && isLLVMExtensionVectorElementType(elementType)) {
    return LLVM::LLVMFixedVectorType::get(elementType, shape.front());
  }
  return VectorType::get(shape, elementType, scalableDims);
}

Type normalizePayloadTypeForLLVMLowering(Type type, Builder &builder) {
  if (pto::isPTOHiFloat8x2Type(type)) {
    return getLLVMCompatibleVectorType(
        {2}, LLVM::LLVMHiFloat8Type::get(builder.getContext()));
}
  // bf16x2 is a 4-byte packed pair; lower it as an opaque i32 so vregs whose
  // element type is bf16x2 get a valid LLVM type.
  if (pto::isPTOBF16x2Type(type)) {
    return builder.getI32Type();
}
  if (Type lowpType = getLowPrecisionLLVMType(type, builder.getContext()))
  {
    return lowpType;
  }

  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (!intType.isSignless())
    {
      return builder.getIntegerType(intType.getWidth());
    }
    return type;
  }

  if (auto vecType = dyn_cast<VectorType>(type)) {
    Type normalizedElement =
        normalizePayloadTypeForLLVMLowering(vecType.getElementType(), builder);
    if (normalizedElement == vecType.getElementType())
    {
      return type;
    }
    return getLLVMCompatibleVectorType(vecType.getShape(), normalizedElement,
                                       vecType.getScalableDims());
  }

  return type;
}

Type normalizeGEPElementTypeForLLVMLowering(Type type,
                                                   Builder &builder) {
  if (pto::isPTOHiFloat8x2Type(type))
  {
    return builder.getI16Type();
  }
  // bf16x2 is 4 bytes, not an 8-bit low-precision type.
  if (pto::isPTOBF16x2Type(type)) {
    return builder.getI32Type();
  }
  if (pto::isPTOLowPrecisionType(type))
  {
    return builder.getI8Type();
  }
  if (isa<LLVM::LLVMHiFloat8Type, LLVM::LLVMFloat8E4M3Type,
          LLVM::LLVMFloat8E5M2Type, LLVM::LLVMFloat4E1M2x2Type,
          LLVM::LLVMFloat4E2M1x2Type>(type)) {
    return builder.getI8Type();
  }

  if (auto vecType = dyn_cast<VectorType>(type)) {
    Type normalizedElement =
        normalizeGEPElementTypeForLLVMLowering(vecType.getElementType(),
                                               builder);
    if (normalizedElement == vecType.getElementType())
    {
      return normalizePayloadTypeForLLVMLowering(type, builder);
    }
    return getLLVMCompatibleVectorType(vecType.getShape(), normalizedElement,
                                       vecType.getScalableDims());
  }

  if (auto vecType = dyn_cast<LLVM::LLVMFixedVectorType>(type)) {
    Type normalizedElement =
        normalizeGEPElementTypeForLLVMLowering(vecType.getElementType(),
                                               builder);
    if (normalizedElement == vecType.getElementType()) {
      return normalizePayloadTypeForLLVMLowering(type, builder);
    }
    return getLLVMCompatibleVectorType({vecType.getNumElements()},
                                       normalizedElement);
  }

  return normalizePayloadTypeForLLVMLowering(type, builder);
}

Type convertVPTOType(Type type, Builder &builder) {
  if (auto vecType = dyn_cast<pto::VRegType>(type)) {
    Type elementType =
        normalizePayloadTypeForLLVMLowering(vecType.getElementType(), builder);
    return getLLVMCompatibleVectorType({vecType.getElementCount()},
                                       elementType);
  }
  if (isa<pto::MaskType>(type)) {
    return VectorType::get({256}, builder.getI1Type());
  }
  if (isa<pto::AlignType>(type)) {
    return VectorType::get({32}, builder.getI8Type());
  }
  if (isa<pto::StructType>(type))
  {
    return LLVM::LLVMPointerType::get(builder.getContext());
  }
  if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
    return LLVM::LLVMPointerType::get(
        builder.getContext(),
        static_cast<unsigned>(ptrType.getMemorySpace().getAddressSpace()));
  }
  return normalizePayloadTypeForLLVMLowering(type, builder);
}

unsigned getNaturalByteAlignment(Type type) {
  if (auto vecType = dyn_cast<VectorType>(type)) {
    unsigned elemAlign = getNaturalByteAlignment(vecType.getElementType());
    if (elemAlign == 0)
    {
      return 0;
    }
    int64_t elems = 1;
    for (int64_t dim : vecType.getShape())
    {
      elems *= dim;
    }
    return elemAlign * static_cast<unsigned>(elems);
  }
  if (auto vecType = dyn_cast<LLVM::LLVMFixedVectorType>(type)) {
    unsigned elemAlign = getNaturalByteAlignment(vecType.getElementType());
    if (elemAlign == 0) {
      return 0;
    }
    return elemAlign * vecType.getNumElements();
  }
  if (auto intType = dyn_cast<IntegerType>(type))
  {
    return llvm::divideCeil(unsigned(intType.getWidth()), 8u);
  }
  if (pto::isPTOHiFloat8x2Type(type))
  {
    return 2;
  }
  if (pto::isPTOBF16x2Type(type)) {
    return 4;
  }
  if (pto::isPTOLowPrecisionType(type))
  {
    return 1;
  }
  if (type.isF16() || type.isBF16())
  {
    return 2;
  }
  if (type.isF32())
  {
    return 4;
  }
  if (type.isF64())
  {
    return 8;
  }
  return 0;
}

bool hasVPTOConvertibleType(Type type) {
  if (!type)
  {
    return false;
  }
  if (isa<pto::VRegType, pto::MaskType, pto::AlignType, pto::PtrType,
          pto::StructType>(type) ||
      pto::isPTOLowPrecisionType(type)) {
    return true;
  }
  if (auto vecType = dyn_cast<VectorType>(type))
  {
    return hasVPTOConvertibleType(vecType.getElementType());
  }
  return false;
}

bool hasVPTOConvertibleType(TypeRange types) {
  return llvm::any_of(types, [](Type type) { return hasVPTOConvertibleType(type); });
}

Value materializeVPTOCast(OpBuilder &builder, Type resultType,
                                 ValueRange inputs, Location loc) {
  if (inputs.size() != 1) {
    return {};
  }
  return builder
      .create<UnrealizedConversionCastOp>(loc, TypeRange{resultType}, inputs)
      .getResult(0);
}


// Struct values carry the address of stack-local storage. Keep the pointee
// type local to struct access lowering so the public type conversion remains
// an opaque LLVM pointer, consistent with other pointer-like PTO handles.
LLVM::LLVMStructType getVPTOStructStorageType(pto::StructType structType,
                                                      Builder &builder) {
  struct Frame {
    pto::StructType type;
    bool materialize;
  };

  // PTO structs form an acyclic type tree. Build literal LLVM struct types in
  // explicit post-order so deeply nested legal structs do not consume the C++
  // call stack during lowering.
  SmallVector<Frame> worklist{{structType, false}};
  llvm::DenseMap<pto::StructType, LLVM::LLVMStructType> storageTypes;
  while (!worklist.empty()) {
    Frame frame = worklist.pop_back_val();
    if (!frame.materialize) {
      worklist.push_back({frame.type, true});
      for (Type fieldType : frame.type.getFieldTypes()) {
        if (auto nestedStruct = dyn_cast<pto::StructType>(fieldType)) {
          worklist.push_back({nestedStruct, false});
        }
      }
      continue;
    }

    SmallVector<Type> fieldTypes;
    fieldTypes.reserve(frame.type.getNumFields());
    for (Type fieldType : frame.type.getFieldTypes()) {
      if (auto nestedStruct = dyn_cast<pto::StructType>(fieldType)) {
        fieldTypes.push_back(storageTypes.find(nestedStruct)->second);
        continue;
      }
      fieldTypes.push_back(convertVPTOType(fieldType, builder));
    }
    storageTypes[frame.type] =
        LLVM::LLVMStructType::getLiteral(builder.getContext(), fieldTypes);
  }
  return storageTypes.find(structType)->second;
}

FailureOr<Value>
getVPTOStructFieldAddress(ConversionPatternRewriter &rewriter, Location loc,
                          Value root, pto::StructType rootType,
                          ArrayRef<int64_t> path) {
  auto pointerType = LLVM::LLVMPointerType::get(rewriter.getContext());
  Value address = root;
  pto::StructType currentType = rootType;
  for (auto [depth, index] : llvm::enumerate(path)) {
    if (index < 0 || index >= static_cast<int64_t>(currentType.getNumFields()))
    {
      return failure();
    }
    Type storageType = getVPTOStructStorageType(currentType, rewriter);
    address = rewriter.create<LLVM::GEPOp>(
        loc, pointerType, storageType, address,
        ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(index)});
    Type fieldType = currentType.getFieldType(static_cast<unsigned>(index));
    if (depth + 1 == path.size())
    {
      continue;
    }
    auto nestedStruct = dyn_cast<pto::StructType>(fieldType);
    if (!nestedStruct)
    {
      return failure();
    }
    currentType = nestedStruct;
  }
  return address;
}

} // namespace mlir::pto
