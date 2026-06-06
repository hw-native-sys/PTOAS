// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// https://discourse.llvm.org/t/matchandrewrite-hiding-virtual-functions/84933/8
#pragma GCC diagnostic ignored "-Woverloaded-virtual"

#include "PTO/Transforms/VPTOLLVMEmitter.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/PTOSyncUtils.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

namespace mlir::pto {

void materializeVecScopeCarrierLoops(ModuleOp module);
LogicalResult applyQueriedTargetAttrs(ModuleOp module,
                                      const VPTOEmissionOptions &options,
                                      llvm::raw_ostream &diagOS);
LogicalResult attachAIVectorScopeMetadata(llvm::Module &llvmModule,
                                          llvm::raw_ostream &diagOS);
void attachHIVMKernelAnnotations(llvm::Module &llvmModule,
                                 ModuleOp sourceModule);

namespace {

constexpr llvm::StringLiteral kVectorSuffix = "_mix_aiv";
constexpr llvm::StringLiteral kCubeSuffix = "_mix_aic";

static std::string getElementTypeFragment(Type type);
static Type getElementTypeFromVectorLike(Type type);
static std::optional<int64_t> getElementCountFromVectorLike(Type type);

static Type getLowPrecisionLLVMType(Type type, MLIRContext *context) {
  if (pto::isPTOHiFloat8Type(type))
    return LLVM::LLVMHiFloat8Type::get(context);
  if (isa<pto::F4E1M2x2Type>(type))
    return LLVM::LLVMFloat4E1M2x2Type::get(context);
  if (isa<pto::F4E2M1x2Type>(type))
    return LLVM::LLVMFloat4E2M1x2Type::get(context);
  if (type.isFloat8E4M3() || type.isFloat8E4M3FN() ||
      type.isFloat8E4M3FNUZ() || type.isFloat8E4M3B11FNUZ())
    return LLVM::LLVMFloat8E4M3Type::get(context);
  if (type.isFloat8E5M2() || type.isFloat8E5M2FNUZ())
    return LLVM::LLVMFloat8E5M2Type::get(context);
  return {};
}

static Type getLLVMCompatibleVectorType(ArrayRef<int64_t> shape,
                                        Type elementType,
                                        ArrayRef<bool> scalableDims = {}) {
  if (shape.size() == 1 && !elementType.isIntOrIndexOrFloat())
    return LLVM::LLVMFixedVectorType::get(elementType, shape.front());
  return VectorType::get(shape, elementType, scalableDims);
}

static Type normalizePayloadTypeForLLVMLowering(Type type, Builder &builder) {
  if (pto::isPTOHiFloat8x2Type(type))
    return LLVM::LLVMFixedVectorType::get(
        LLVM::LLVMHiFloat8Type::get(builder.getContext()), 2);
  if (Type lowpType = getLowPrecisionLLVMType(type, builder.getContext()))
    return lowpType;

  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (!intType.isSignless())
      return builder.getIntegerType(intType.getWidth());
    return type;
  }

  if (auto vecType = dyn_cast<VectorType>(type)) {
    Type normalizedElement =
        normalizePayloadTypeForLLVMLowering(vecType.getElementType(), builder);
    if (normalizedElement == vecType.getElementType())
      return type;
    return getLLVMCompatibleVectorType(vecType.getShape(), normalizedElement,
                                       vecType.getScalableDims());
  }

  return type;
}

static Type convertVPTOType(Type type, Builder &builder) {
  if (auto vecType = dyn_cast<pto::VRegType>(type)) {
    Type elementType =
        normalizePayloadTypeForLLVMLowering(vecType.getElementType(), builder);
    return getLLVMCompatibleVectorType({vecType.getElementCount()},
                                       elementType);
  }
  if (isa<pto::MaskType>(type))
    return VectorType::get({256}, builder.getI1Type());
  if (isa<pto::AlignType>(type))
    return VectorType::get({32}, builder.getI8Type());
  if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
    return LLVM::LLVMPointerType::get(
        builder.getContext(),
        static_cast<unsigned>(ptrType.getMemorySpace().getAddressSpace()));
  }
  return normalizePayloadTypeForLLVMLowering(type, builder);
}

static unsigned getNaturalByteAlignment(Type type) {
  if (auto vecType = dyn_cast<VectorType>(type)) {
    unsigned elemAlign = getNaturalByteAlignment(vecType.getElementType());
    if (!elemAlign)
      return 0;
    int64_t elems = 1;
    for (int64_t dim : vecType.getShape())
      elems *= dim;
    return elemAlign * static_cast<unsigned>(elems);
  }
  if (auto vecType = dyn_cast<LLVM::LLVMFixedVectorType>(type)) {
    unsigned elemAlign = getNaturalByteAlignment(vecType.getElementType());
    if (!elemAlign)
      return 0;
    return elemAlign * vecType.getNumElements();
  }
  if (auto intType = dyn_cast<IntegerType>(type))
    return llvm::divideCeil(unsigned(intType.getWidth()), 8u);
  if (pto::isPTOHiFloat8x2Type(type))
    return 2;
  if (pto::isPTOLowPrecisionType(type))
    return 1;
  if (type.isF16() || type.isBF16())
    return 2;
  if (type.isF32())
    return 4;
  if (type.isF64())
    return 8;
  return 0;
}

static bool hasVPTOConvertibleType(Type type) {
  return isa<pto::VRegType, pto::MaskType, pto::AlignType, pto::PtrType>(type);
}

static bool hasVPTOConvertibleType(TypeRange types) {
  return llvm::any_of(types, [](Type type) { return hasVPTOConvertibleType(type); });
}

static Value materializeVPTOCast(OpBuilder &builder, Type resultType,
                                 ValueRange inputs, Location loc) {
  if (inputs.size() != 1)
    return {};
  return builder
      .create<UnrealizedConversionCastOp>(loc, TypeRange{resultType}, inputs)
      .getResult(0);
}

class VPTOTypeConverter final : public TypeConverter {
public:
  explicit VPTOTypeConverter(MLIRContext *context) {
    addConversion([](Type type) { return type; });
    addConversion([](Type type) -> Type {
      // The conversion callback outlives this constructor, so build on demand
      // from the current type context instead of capturing a local Builder.
      Builder builder(type.getContext());
      return convertVPTOType(type, builder);
    });
    addSourceMaterialization(materializeVPTOCast);
    addTargetMaterialization(materializeVPTOCast);
    addArgumentMaterialization(materializeVPTOCast);
  }
};

struct PlannedDecl {
  std::string name;
  FunctionType type;
};

struct LoweringState {
  SmallVector<PlannedDecl> plannedDecls;
};

enum class VcvtElemKind {
  Invalid,
  F16,
  BF16,
  F32,
  S8,
  U8,
  S16,
  U16,
  S32,
  U32,
  S64,
};

struct VcvtContract {
  const char *intrinsic;
  bool requiresRnd;
  bool requiresSat;
  bool requiresPart;
  unsigned maskBitWidth;
  bool satBeforeRnd = false;
};

static Value getI64Constant(OpBuilder &builder, Location loc, uint64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(value))
      .getResult();
}

static Value getI32Constant(OpBuilder &builder, Location loc, uint64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(value))
      .getResult();
}

[[maybe_unused]] static Value getI1Constant(OpBuilder &builder, Location loc,
                                            bool value) {
  return builder
      .create<arith::ConstantOp>(
          loc, builder.getIntegerAttr(builder.getI1Type(), value ? 1 : 0))
      .getResult();
}

static bool isMxElementType(Type ty) {
  if (auto floatType = dyn_cast<FloatType>(ty))
    return floatType.getWidth() == 8;
  std::string typeText;
  llvm::raw_string_ostream os(typeText);
  ty.print(os);
  os.flush();
  return StringRef(typeText).starts_with("f8");
}

static std::string getMadMxElementFragment(Type type) {
  if (type.isF16())
    return "f16";
  if (type.isBF16())
    return "bf16";

  std::string typeText;
  llvm::raw_string_ostream os(typeText);
  type.print(os);
  os.flush();

  std::string lower = StringRef(typeText).lower();
  if (StringRef(lower).contains("e4m3"))
    return "e4m3";
  if (StringRef(lower).contains("e5m2"))
    return "e5m2";
  if (StringRef(lower).contains("hif4"))
    return "hif4";
  if (StringRef(lower).contains("e2m1x2"))
    return "e2m1x2";
  if (StringRef(lower).contains("e1m2x2"))
    return "e1m2x2";
  return {};
}

static FailureOr<StringRef> buildMadMxCalleeName(MLIRContext *context,
                                                 Type lhsElem, Type rhsElem) {
  std::string lhs = getMadMxElementFragment(lhsElem);
  std::string rhs = getMadMxElementFragment(rhsElem);
  if (lhs.empty() || rhs.empty())
    return failure();
  return StringAttr::get(context, "llvm.hivm.MMAD.MX." + lhs + rhs).getValue();
}

static bool isSignedOrSignlessInteger(IntegerType intType, unsigned width) {
  return intType && intType.getWidth() == width &&
         (intType.isSigned() || intType.isSignless());
}

static std::string getMadRhsFragment(Type type) {
  if (type.isF16())
    return "f16";
  if (type.isBF16())
    return "bf16";
  if (type.isF32())
    return "f32";
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (isSignedOrSignlessInteger(intType, 4))
      return "s4";
    if (isSignedOrSignlessInteger(intType, 8))
      return "s8";
    if (intType.isUnsigned() && intType.getWidth() == 2)
      return "u2";
  }

  std::string typeText;
  llvm::raw_string_ostream os(typeText);
  type.print(os);
  os.flush();
  std::string lower = StringRef(typeText).lower();
  if (StringRef(lower).contains("e8m0"))
    return "e8m0";
  return {};
}

static bool isMadE4M3ElementType(Type type) {
  return type.isFloat8E4M3() || type.isFloat8E4M3FN() ||
         type.isFloat8E4M3FNUZ() || type.isFloat8E4M3B11FNUZ();
}

static std::string getMadDstFragment(Type type) {
  if (type.isF16())
    return "f16";
  if (type.isF32())
    return "f32";
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (isSignedOrSignlessInteger(intType, 32))
      return "s32";
  }
  return {};
}

static FailureOr<StringRef> buildMadTypedCalleeName(MLIRContext *context,
                                                    Type lhsElem, Type rhsElem,
                                                    Type dstElem) {
  std::string rhs = getMadRhsFragment(rhsElem);
  std::string dst = getMadDstFragment(dstElem);
  if (lhsElem.isF16() && rhs == "f16" && dst == "f32")
    return StringAttr::get(context, "llvm.hivm.MAD.f162f32.c310").getValue();
  if (lhsElem.isF16() && rhs == "f16" && dst == "f16")
    return StringAttr::get(context, "llvm.hivm.MAD.f162f16").getValue();
  if (lhsElem.isF16() && rhs == "f16" && dst == "s32")
    return StringAttr::get(context, "llvm.hivm.MAD.f162s32.1952").getValue();
  if (lhsElem.isBF16() && rhs == "bf16" && dst == "f32")
    return StringAttr::get(context, "llvm.hivm.MAD.bf162f32.c310").getValue();
  if (lhsElem.isF32() && rhs == "f32" && dst == "f32")
    return StringAttr::get(context, "llvm.hivm.MAD.f322f32.c310").getValue();
  if (isSignedOrSignlessInteger(dyn_cast<IntegerType>(lhsElem), 8) &&
      rhs == "s8" && dst == "s32")
    return StringAttr::get(context, "llvm.hivm.MAD.s8.c310").getValue();
  if (isMadE4M3ElementType(lhsElem) && isMadE4M3ElementType(rhsElem) &&
      dst == "f32")
    return StringAttr::get(context, "llvm.hivm.MAD.e4m3e4m3.c310").getValue();
  if (pto::isPTOHiFloat8Type(lhsElem) && pto::isPTOHiFloat8Type(rhsElem) &&
      dst == "f32")
    return StringAttr::get(context, "llvm.hivm.MAD.e4m3e4m3.c310").getValue();
  if (lhsElem.isF16() && rhs == "s4")
    return StringAttr::get(context, "llvm.hivm.MAD.f16s4.c310").getValue();
  if (lhsElem.isF16() && rhs == "s8")
    return StringAttr::get(context, "llvm.hivm.MAD.f16s8.c310").getValue();
  if (lhsElem.isF16() && rhs == "u2")
    return StringAttr::get(context, "llvm.hivm.MAD.f16u2").getValue();
  if (lhsElem.isF16() && rhs == "e8m0")
    return StringAttr::get(context, "llvm.hivm.MAD.f16e8m0.c310").getValue();
  return failure();
}

static FailureOr<StringRef> buildLaneTypedCallee(MLIRContext *context,
                                                 Type resultType,
                                                 StringRef stem,
                                                 StringRef suffix) {
  std::string vec =
      getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
    return failure();

  return StringAttr::get(context, "llvm.hivm." + stem.str() + ".v" +
                                      std::to_string(*lanes) + vec +
                                      suffix.str())
      .getValue();
}

static FailureOr<StringRef> buildLaneTypedCalleeFromInput(MLIRContext *context,
                                                          Type inputType,
                                                          StringRef stem,
                                                          StringRef suffix) {
  std::string vec =
      getElementTypeFragment(getElementTypeFromVectorLike(inputType));
  auto lanes = getElementCountFromVectorLike(inputType);
  if (vec.empty() || !lanes)
    return failure();

  return StringAttr::get(context, "llvm.hivm." + stem.str() + ".v" +
                                      std::to_string(*lanes) + vec +
                                      suffix.str())
      .getValue();
}

static std::string getElementTypeFragment(Type type) {
  if (type.isF16())
    return "f16";
  if (type.isBF16())
    return "bf16";
  if (type.isF32())
    return "f32";
  if (auto intType = dyn_cast<IntegerType>(type))
    return (intType.isUnsigned() ? "u" : "s") + std::to_string(intType.getWidth());
  return {};
}

static std::string getLowPrecisionElementFragment(Type type) {
  if (pto::isPTOHiFloat8x2Type(type))
    return "hif8x2";
  if (pto::isPTOHiFloat8Type(type))
    return "hif8";
  if (isa<pto::F4E1M2x2Type>(type))
    return "f4e1m2x2";
  if (isa<pto::F4E2M1x2Type>(type))
    return "f4e2m1x2";
  if (type.isFloat8E4M3() || type.isFloat8E4M3FN() ||
      type.isFloat8E4M3FNUZ() || type.isFloat8E4M3B11FNUZ())
    return "f8e4m3";
  if (type.isFloat8E5M2() || type.isFloat8E5M2FNUZ())
    return "f8e5m2";
  return {};
}

static std::string getAtomicElementTypeFragment(Type type,
                                                Attribute signednessAttr) {
  if (auto vecType = dyn_cast<VectorType>(type)) {
    if (vecType.getRank() != 1 || vecType.getDimSize(0) != 2)
      return {};
    if (vecType.getElementType().isF16())
      return "f16x2";
    if (vecType.getElementType().isBF16())
      return "bf16x2";
    return {};
  }
  if (type.isF16())
    return "fp16";
  if (type.isBF16())
    return "bf16";
  if (type.isF32())
    return "fp32";
  auto intType = dyn_cast<IntegerType>(type);
  if (!intType)
    return {};
  if (intType.getWidth() != 32 && intType.getWidth() != 64)
    return {};
  if (signednessAttr) {
    auto signedness = cast<pto::SignednessAttr>(signednessAttr).getValue();
    return std::string(signedness == pto::Signedness::Unsigned ? "u" : "s") +
           std::to_string(intType.getWidth());
  }
  return std::string(intType.isUnsigned() ? "u" : "s") +
         std::to_string(intType.getWidth());
}

static std::string getL0LoadElementFragment(Type type) {
  std::string elem = getElementTypeFragment(type);
  if (!elem.empty())
    return elem;

  std::string typeText;
  llvm::raw_string_ostream os(typeText);
  type.print(os);
  os.flush();
  std::string lower = StringRef(typeText).lower();
  if (StringRef(lower).contains("e4m3") ||
      StringRef(lower).contains("e5m2") ||
      StringRef(lower).contains("e8m0") ||
      StringRef(lower).contains("hif8"))
    return "s8";
  return {};
}

static std::string getVbrScalarFragment(Type type) {
  if (type.isF16())
    return "f16";
  if (type.isBF16())
    return "bf16";
  if (type.isF32())
    return "f32";
  if (auto intType = dyn_cast<IntegerType>(type))
    return (intType.isUnsigned() ? "u" : "s") + std::to_string(intType.getWidth());
  return {};
}

static std::string getShuffleIntrinsicTypeFragment(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    switch (intType.getWidth()) {
    case 32:
      return "i32";
    case 64:
      return "i64";
    default:
      return {};
    }
  }
  if (type.isF16())
    return "f16";
  if (type.isF32())
    return "f32";
  if (auto vecType = dyn_cast<VectorType>(type)) {
    if (vecType.getRank() == 1 && vecType.getDimSize(0) == 2 &&
        vecType.getElementType().isF16())
      return "v2f16";
  }
  return {};
}

static std::string getReduxIntrinsicTypeFragment(Type type,
                                                 Attribute signednessAttr) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (intType.getWidth() != 32)
      return {};
    bool isUnsigned = false;
    if (signednessAttr) {
      isUnsigned = cast<pto::SignednessAttr>(signednessAttr).getValue() ==
                   pto::Signedness::Unsigned;
    }
    return isUnsigned ? "u32" : "s32";
  }
  if (type.isF16())
    return "f16";
  if (type.isF32())
    return "f32";
  return {};
}

static Type getElementTypeFromVectorLike(Type type) {
  if (auto vecType = dyn_cast<pto::VRegType>(type))
    return vecType.getElementType();
  if (auto vecType = dyn_cast<VectorType>(type))
    return vecType.getElementType();
  if (auto vecType = dyn_cast<LLVM::LLVMFixedVectorType>(type))
    return vecType.getElementType();
  return {};
}

static std::optional<int64_t> getElementCountFromVectorLike(Type type) {
  if (auto vecType = dyn_cast<pto::VRegType>(type))
    return vecType.getElementCount();
  if (auto vecType = dyn_cast<VectorType>(type)) {
    if (vecType.getRank() != 1)
      return std::nullopt;
    return vecType.getShape().front();
  }
  if (auto vecType = dyn_cast<LLVM::LLVMFixedVectorType>(type))
    return vecType.getNumElements();
  return std::nullopt;
}

static Value castIntegerLikeTo(Operation *anchor, Value value, Type targetType) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);

  if (value.getType() == targetType)
    return value;

  auto targetInt = dyn_cast<IntegerType>(targetType);
  if (value.getType().isIndex() && targetInt)
    return builder.create<arith::IndexCastOp>(anchor->getLoc(), targetType, value);
  if (auto sourceInt = dyn_cast<IntegerType>(value.getType())) {
    if (targetInt) {
      if (sourceInt.getWidth() < targetInt.getWidth())
        return builder.create<arith::ExtUIOp>(anchor->getLoc(), targetType, value);
      if (sourceInt.getWidth() > targetInt.getWidth())
        return builder.create<arith::TruncIOp>(anchor->getLoc(), targetType, value);
      return value;
    }
    if (targetType.isIndex())
      return builder.create<arith::IndexCastOp>(anchor->getLoc(), targetType, value);
  }

  return {};
}

static FailureOr<Value> reinterpretPointerToAddrSpace(Operation *anchor,
                                                      Value value,
                                                      unsigned targetAddressSpace) {
  auto sourcePtrType = dyn_cast<LLVM::LLVMPointerType>(value.getType());
  if (!sourcePtrType)
    return failure();
  if (sourcePtrType.getAddressSpace() == targetAddressSpace)
    return value;

  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();
  Value asInt = builder.create<LLVM::PtrToIntOp>(loc, builder.getI64Type(), value);
  Type targetPtrType =
      LLVM::LLVMPointerType::get(anchor->getContext(), targetAddressSpace);
  return builder.create<LLVM::IntToPtrOp>(loc, targetPtrType, asInt).getResult();
}

static FailureOr<Value> normalizeVdupScalarOperand(OpBuilder &builder, Location loc,
                                                   Value input,
                                                   Type resultType) {
  auto intType = dyn_cast<IntegerType>(input.getType());
  if (!intType || intType.getWidth() != 8)
    return input;

  Type resultElemType = getElementTypeFromVectorLike(resultType);
  std::string resultElemFragment = getElementTypeFragment(resultElemType);
  if (resultElemFragment != "s8" && resultElemFragment != "u8")
    return input;

  if (intType.isSignless())
    return input;

  Type signlessType = builder.getIntegerType(intType.getWidth());
  return builder
      .create<UnrealizedConversionCastOp>(loc, TypeRange{signlessType}, input)
      .getResult(0);
}

static Value normalizeByteScalarOperandForHivmCall(OpBuilder &builder, Location loc,
                                                   Value input,
                                                   Type semanticElementType) {
  auto intType = dyn_cast<IntegerType>(input.getType());
  if (!intType || intType.getWidth() != 8)
    return input;

  Type i16Type = builder.getIntegerType(16);
  auto semanticIntType = dyn_cast<IntegerType>(semanticElementType);
  if (semanticIntType && semanticIntType.isUnsigned())
    return builder.create<arith::ExtUIOp>(loc, i16Type, input).getResult();
  return builder.create<arith::ExtSIOp>(loc, i16Type, input).getResult();
}

static bool isCompatibleScalarForSemanticType(Type semanticType,
                                              Type scalarType) {
  if (semanticType == scalarType)
    return true;

  auto semanticInt = dyn_cast<IntegerType>(semanticType);
  auto scalarInt = dyn_cast<IntegerType>(scalarType);
  if (!semanticInt || !scalarInt || semanticInt.getWidth() != scalarInt.getWidth())
    return false;

  if (semanticInt.isSigned())
    return scalarInt.isSigned() || scalarInt.isSignless();
  if (semanticInt.isUnsigned())
    return scalarInt.isUnsigned() || scalarInt.isSignless();
  return scalarInt.isSignless();
}

static std::string getCopyElementFragment(Type elementType) {
  if (!elementType)
    return {};
  if (elementType.isF16())
    return "f16";
  if (elementType.isBF16())
    return "bf16";
  if (elementType.isF32())
    return "f32";
  // Handle FP8 family (e4m3/e5m2/e8m0/hif8) used by cube-matmul/mad_mx.
  std::string typeText;
  llvm::raw_string_ostream os(typeText);
  elementType.print(os);
  os.flush();
  std::string lower = StringRef(typeText).lower();
  if (StringRef(lower).contains("e4m3"))
    return "e4m3";
  if (StringRef(lower).contains("e5m2"))
    return "e5m2";
  if (StringRef(lower).contains("e8m0"))
    return "e8m0";
  if (StringRef(lower).contains("hif8"))
    return "hif8";
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

static std::string getNd2NzCopyElementFragment(Type elementType) {
  if (!elementType)
    return {};
  std::string typeText;
  llvm::raw_string_ostream os(typeText);
  elementType.print(os);
  os.flush();
  std::string lower = StringRef(typeText).lower();
  if (StringRef(lower).contains("e4m3") || StringRef(lower).contains("e5m2") ||
      StringRef(lower).contains("e8m0") || StringRef(lower).contains("hif8"))
    return "U8";

  if (elementType.isF16() || elementType.isBF16())
    return "U16";
  if (elementType.isF32())
    return "U32";
  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    switch (intType.getWidth()) {
    case 8:
      return "U8";
    case 16:
      return "U16";
    case 32:
      return "U32";
    default:
      return {};
    }
  }
  return {};
}

static std::optional<uint64_t> parsePredicatePatternImmediate(StringRef pattern) {
  if (pattern == "PAT_ALL")
    return 0;
  if (pattern == "PAT_VL1")
    return 1;
  if (pattern == "PAT_VL2")
    return 2;
  if (pattern == "PAT_VL3")
    return 3;
  if (pattern == "PAT_VL4")
    return 4;
  if (pattern == "PAT_VL8")
    return 5;
  if (pattern == "PAT_VL16")
    return 6;
  if (pattern == "PAT_VL32")
    return 7;
  if (pattern == "PAT_VL64")
    return 8;
  if (pattern == "PAT_VL128")
    return 9;
  if (pattern == "PAT_M3")
    return 10;
  if (pattern == "PAT_M4")
    return 11;
  if (pattern == "PAT_H")
    return 12;
  if (pattern == "PAT_Q")
    return 13;
  if (pattern == "PAT_ALLF")
    return 15;
  return std::nullopt;
}

static std::optional<uint64_t> parseHiLoPartImmediate(StringRef part) {
  if (part == "LOWER")
    return 0;
  if (part == "HIGHER")
    return 1;
  return std::nullopt;
}

static std::optional<uint64_t> parseRoundModeImmediate(StringRef roundMode) {
  if (roundMode == "R" || roundMode == "ROUND_R")
    return 0;
  if (roundMode == "A" || roundMode == "ROUND_A")
    return 1;
  if (roundMode == "F" || roundMode == "ROUND_F")
    return 2;
  if (roundMode == "C" || roundMode == "ROUND_C")
    return 3;
  if (roundMode == "Z" || roundMode == "ROUND_Z")
    return 4;
  if (roundMode == "O" || roundMode == "ROUND_O")
    return 5;
  return std::nullopt;
}

static std::optional<uint64_t> parseSaturationImmediate(StringRef sat) {
  if (sat == "SAT")
    return 1;
  if (sat == "NOSAT")
    return 0;
  return std::nullopt;
}

static std::optional<uint64_t> parsePartImmediate(StringRef part) {
  if (part == "EVEN" || part == "PART_EVEN")
    return 0;
  if (part == "ODD" || part == "PART_ODD")
    return 1;
  return std::nullopt;
}

static std::optional<uint64_t> parseVcvtPartImmediate(StringRef part) {
  if (part == "EVEN" || part == "PART_EVEN" || part == "P0" ||
      part == "PART_P0")
    return 0;
  if (part == "ODD" || part == "PART_ODD" || part == "P1" ||
      part == "PART_P1")
    return 1;
  if (part == "P2" || part == "PART_P2")
    return 2;
  if (part == "P3" || part == "PART_P3")
    return 3;
  return std::nullopt;
}

static std::optional<uint64_t> parsePredicateStoreDistImmediate(StringRef dist) {
  if (dist == "NORM")
    return 0;
  if (dist == "PK")
    return 1;
  return std::nullopt;
}

static std::optional<uint64_t> parsePredicateLoadDistImmediate(StringRef dist) {
  if (dist.empty() || dist == "NORM")
    return 0;
  if (dist == "US")
    return 1;
  if (dist == "DS")
    return 2;
  return std::nullopt;
}

static std::optional<int32_t> parsePostModeImmediate(StringRef mode) {
  if (mode == "NO_POST_UPDATE")
    return 0;
  if (mode == "POST_UPDATE")
    return 1;
  return std::nullopt;
}

static std::optional<uint64_t> parsePipeImmediate(StringRef pipe) {
  if (pipe == "PIPE_S")
    return 0;
  if (pipe == "PIPE_V")
    return 1;
  if (pipe == "PIPE_M")
    return 2;
  if (pipe == "PIPE_MTE1")
    return 3;
  if (pipe == "PIPE_MTE2")
    return 4;
  if (pipe == "PIPE_MTE3")
    return 5;
  if (pipe == "PIPE_ALL")
    return 6;
  if (pipe == "PIPE_MTE4")
    return 7;
  if (pipe == "PIPE_MTE5")
    return 8;
  if (pipe == "PIPE_V2")
    return 9;
  if (pipe == "PIPE_FIX")
    return 10;
  if (pipe == "VIRTUAL_PIPE_MTE2_L1A")
    return 11;
  if (pipe == "VIRTUAL_PIPE_MTE2_L1B")
    return 12;
  return std::nullopt;
}

static std::optional<uint64_t> parseEventImmediate(StringRef event) {
  if (!event.consume_front("EVENT_ID"))
    return std::nullopt;
  uint64_t value = 0;
  if (event.getAsInteger(10, value))
    return std::nullopt;
  return value;
}

static std::optional<uint64_t> parseSprImmediate(StringRef spr) {
  if (spr == "AR")
    return 74;
  return std::nullopt;
}

static std::optional<unsigned> getDistElementWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();
  if (type.isF16() || type.isBF16())
    return 16;
  if (type.isF32())
    return 32;
  if (type.isF64())
    return 64;
  return std::nullopt;
}

static VcvtElemKind classifyVcvtElemType(Type type) {
  if (type.isF16())
    return VcvtElemKind::F16;
  if (type.isBF16())
    return VcvtElemKind::BF16;
  if (type.isF32())
    return VcvtElemKind::F32;
  if (auto intType = dyn_cast<IntegerType>(type)) {
    switch (intType.getWidth()) {
    case 8:
      return intType.isUnsigned() ? VcvtElemKind::U8 : VcvtElemKind::S8;
    case 16:
      return intType.isUnsigned() ? VcvtElemKind::U16 : VcvtElemKind::S16;
    case 32:
      return intType.isUnsigned() ? VcvtElemKind::U32 : VcvtElemKind::S32;
    case 64:
      return intType.isUnsigned() ? VcvtElemKind::Invalid : VcvtElemKind::S64;
    default:
      return VcvtElemKind::Invalid;
    }
  }
  return VcvtElemKind::Invalid;
}

static std::optional<VcvtContract> lookupVcvtContract(VcvtElemKind src,
                                                      VcvtElemKind dst) {
  switch (src) {
  case VcvtElemKind::F32:
    switch (dst) {
    case VcvtElemKind::F16:
      return VcvtContract{"llvm.hivm.vcvtff.f322f16.x", true, true, true, 32};
    case VcvtElemKind::BF16:
      return VcvtContract{"llvm.hivm.vcvtff.f322bf16.x", true, true, true, 32};
    case VcvtElemKind::S16:
      return VcvtContract{"llvm.hivm.vcvtfi.f322s16.x", true, true, true, 32};
    case VcvtElemKind::S32:
      return VcvtContract{"llvm.hivm.vcvtfi.f322s32.x", true, true, false, 32};
    case VcvtElemKind::S64:
      return VcvtContract{"llvm.hivm.vcvtfi.f322s64.x", true, true, true, 32};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::F16:
    switch (dst) {
    case VcvtElemKind::F32:
      return VcvtContract{"llvm.hivm.vcvtff.f162f32.x", false, false, true, 16};
    case VcvtElemKind::S32:
      return VcvtContract{"llvm.hivm.vcvtfi.f162s32.x", true, false, true, 16};
    case VcvtElemKind::S16:
      return VcvtContract{"llvm.hivm.vcvtfi.f162s16.x", true, true, false, 16};
    case VcvtElemKind::S8:
      return VcvtContract{"llvm.hivm.vcvtfi.f162s8.x", true, true, true, 16};
    case VcvtElemKind::U8:
      return VcvtContract{"llvm.hivm.vcvtfi.f162u8.x", true, true, true, 16};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::BF16:
    switch (dst) {
    case VcvtElemKind::F16:
      return VcvtContract{"llvm.hivm.vcvtff.bf162f16.x", true, true, false, 16,
                          true};
    case VcvtElemKind::F32:
      return VcvtContract{"llvm.hivm.vcvtff.bf162f32.x", false, false, true, 16};
    case VcvtElemKind::S32:
      return VcvtContract{"llvm.hivm.vcvtfi.bf162s32.x", true, true, true, 16};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::U8:
    switch (dst) {
    case VcvtElemKind::F16:
      return VcvtContract{"llvm.hivm.vcvtif.u82f16.x", false, false, true, 8};
    case VcvtElemKind::U16:
      return VcvtContract{"llvm.hivm.vcvtii.u82u16.x", false, false, true, 8};
    case VcvtElemKind::U32:
      return VcvtContract{"llvm.hivm.vcvtii.u82u32.x", false, false, true, 8};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::S8:
    switch (dst) {
    case VcvtElemKind::F16:
      return VcvtContract{"llvm.hivm.vcvtif.s82f16.x", false, false, true, 8};
    case VcvtElemKind::S16:
      return VcvtContract{"llvm.hivm.vcvtii.s82s16.x", false, false, true, 8};
    case VcvtElemKind::S32:
      return VcvtContract{"llvm.hivm.vcvtii.s82s32.x", false, false, true, 8};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::U16:
    switch (dst) {
    case VcvtElemKind::U8:
      return VcvtContract{"llvm.hivm.vcvtii.u162u8.x", false, true, true, 16};
    case VcvtElemKind::U32:
      return VcvtContract{"llvm.hivm.vcvtii.u162u32.x", false, false, true, 16};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::S16:
    switch (dst) {
    case VcvtElemKind::F16:
      return VcvtContract{"llvm.hivm.vcvtif.s162f16.x", true, false, false, 16};
    case VcvtElemKind::F32:
      return VcvtContract{"llvm.hivm.vcvtif.s162f32.x", false, false, true, 16};
    case VcvtElemKind::U8:
      return VcvtContract{"llvm.hivm.vcvtii.s162u8.x", false, true, true, 16};
    case VcvtElemKind::U32:
      return VcvtContract{"llvm.hivm.vcvtii.s162u32.x", false, false, true, 16};
    case VcvtElemKind::S32:
      return VcvtContract{"llvm.hivm.vcvtii.s162s32.x", false, false, true, 16};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::U32:
    switch (dst) {
    case VcvtElemKind::U8:
      return VcvtContract{"llvm.hivm.vcvtii.u322u8.x", false, true, true, 32};
    case VcvtElemKind::U16:
      return VcvtContract{"llvm.hivm.vcvtii.u322u16.x", false, true, true, 32};
    case VcvtElemKind::S16:
      return VcvtContract{"llvm.hivm.vcvtii.u322s16.x", false, true, true, 32};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::S32:
    switch (dst) {
    case VcvtElemKind::F32:
      return VcvtContract{"llvm.hivm.vcvtif.s322f32.x", true, false, false, 32};
    case VcvtElemKind::U8:
      return VcvtContract{"llvm.hivm.vcvtii.s322u8.x", false, true, true, 32};
    case VcvtElemKind::U16:
      return VcvtContract{"llvm.hivm.vcvtii.s322u16.x", false, true, true, 32};
    case VcvtElemKind::S16:
      return VcvtContract{"llvm.hivm.vcvtii.s322s16.x", false, true, true, 32};
    case VcvtElemKind::S64:
      return VcvtContract{"llvm.hivm.vcvtii.s322s64.x", false, false, true, 32};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::S64:
    switch (dst) {
    case VcvtElemKind::F32:
      return VcvtContract{"llvm.hivm.vcvtif.s642f32.x", true, false, true, 32};
    case VcvtElemKind::S32:
      return VcvtContract{"llvm.hivm.vcvtii.s642s32.x", false, true, true, 32};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::Invalid:
    return std::nullopt;
  }
  return std::nullopt;
}

// VSQZ #st hint must only be set when the compacted vector feeds VSTUR.
// Emitting #st=1 without a matching VSTUR consumer can deadlock hardware queues.
static uint64_t determineVsqzStoreHint(pto::VsqzOp vsqz) {
  Value result = vsqz.getResult();
  for (Operation *user : result.getUsers()) {
    auto vstur = dyn_cast<pto::VsturOp>(user);
    if (!vstur)
      continue;
    if (vstur.getValue() == result)
      return 1;
  }
  return 0;
}

static std::optional<uint64_t> parseLoadDistImmediate(StringRef dist,
                                                      Type elementType) {
  auto width = getDistElementWidth(elementType);
  if (dist.empty() || dist == "NORM")
    return 0;
  if (!width)
    return std::nullopt;
  if (dist == "BRC_B8")
    return std::optional<uint64_t>(1);
  if (dist == "BRC_B16")
    return std::optional<uint64_t>(2);
  if (dist == "BRC_B32")
    return std::optional<uint64_t>(3);
  if (dist == "US_B8")
    return std::optional<uint64_t>(6);
  if (dist == "US_B16")
    return std::optional<uint64_t>(7);
  if (dist == "DS_B8")
    return std::optional<uint64_t>(8);
  if (dist == "DS_B16")
    return std::optional<uint64_t>(9);
  if (dist == "UNPK_B8")
    return std::optional<uint64_t>(13);
  if (dist == "UNPK_B16")
    return std::optional<uint64_t>(14);
  if (dist == "UNPK_B32")
    return std::optional<uint64_t>(18);
  if (dist == "BRC_BLK")
    return 15;
  if (dist == "E2B_B16")
    return std::optional<uint64_t>(16);
  if (dist == "E2B_B32")
    return std::optional<uint64_t>(17);
  if (dist == "UNPK4")
    return *width == 8 ? std::optional<uint64_t>(20) : std::nullopt;
  if (dist == "SPLT4CHN")
    return *width == 8 ? std::optional<uint64_t>(21) : std::nullopt;
  if (dist == "SPLT2CHN_B8")
    return std::optional<uint64_t>(22);
  if (dist == "SPLT2CHN_B16")
    return std::optional<uint64_t>(23);
  return std::nullopt;
}

static std::optional<uint64_t> parseLoadX2DistImmediate(StringRef dist,
                                                        Type elementType) {
  auto width = getDistElementWidth(elementType);
  if (dist == "BDINTLV")
    return 10;
  if (!width)
    return std::nullopt;
  if (dist == "DINTLV_B8")
    return std::optional<uint64_t>(11);
  if (dist == "DINTLV_B16")
    return std::optional<uint64_t>(12);
  if (dist == "DINTLV_B32")
    return std::optional<uint64_t>(19);
  return std::nullopt;
}

static std::optional<uint64_t> parseStoreDistImmediate(StringRef dist,
                                                       Type elementType) {
  auto width = getDistElementWidth(elementType);
  if (dist.empty()) {
    if (!width)
      return std::nullopt;
    if (*width == 8)
      return 0;
    if (*width == 16)
      return 1;
    if (*width == 32)
      return 2;
    return std::nullopt;
  }
  if (dist == "NORM_B8")
    return std::optional<uint64_t>(0);
  if (dist == "NORM_B16")
    return std::optional<uint64_t>(1);
  if (dist == "NORM_B32")
    return std::optional<uint64_t>(2);
  if (dist == "1PT_B8")
    return std::optional<uint64_t>(3);
  if (dist == "1PT_B16")
    return std::optional<uint64_t>(4);
  if (dist == "1PT_B32")
    return std::optional<uint64_t>(5);
  if (dist == "PK_B16")
    return std::optional<uint64_t>(6);
  if (dist == "PK_B32")
    return std::optional<uint64_t>(7);
  if (dist == "PK_B64")
    return std::optional<uint64_t>(10);
  if (dist == "PK4_B32")
    return std::optional<uint64_t>(12);
  if (dist == "MRG4CHN_B8")
    return std::optional<uint64_t>(13);
  if (dist == "MRG2CHN_B8")
    return std::optional<uint64_t>(14);
  if (dist == "MRG2CHN_B16")
    return std::optional<uint64_t>(15);
  return std::nullopt;
}

static std::optional<uint64_t> parseStoreX2DistImmediate(StringRef dist,
                                                         Type elementType) {
  auto width = getDistElementWidth(elementType);
  if (!width)
    return std::nullopt;
  if (dist == "INTLV_B8")
    return std::optional<uint64_t>(8);
  if (dist == "INTLV_B16")
    return std::optional<uint64_t>(9);
  if (dist == "INTLV_B32")
    return std::optional<uint64_t>(11);
  return std::nullopt;
}

static Value packBlockRepeatStride(Operation *anchor, Value blockStride,
                                   Value repeatStride) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);

  Value blockI32 = castIntegerLikeTo(anchor, blockStride, builder.getI32Type());
  Value repeatI32 =
      castIntegerLikeTo(anchor, repeatStride, builder.getI32Type());
  if (!blockI32 || !repeatI32)
    return {};

  auto c16 = builder.create<arith::ConstantIntOp>(anchor->getLoc(), 16, 32);
  auto blockShifted =
      builder.create<arith::ShLIOp>(anchor->getLoc(), blockI32, c16);
  return builder
      .create<arith::OrIOp>(anchor->getLoc(), blockShifted, repeatI32)
      .getResult();
}

static std::optional<uint64_t> parseOrderImmediate(StringRef order) {
  if (order.empty() || order == "ASC")
    return 0;
  if (order == "DESC")
    return 1;
  return std::nullopt;
}

static FailureOr<Value> packLoopPair(Operation *anchor, Value low, Value high) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);

  Value lowI64 = castIntegerLikeTo(anchor, low, builder.getI64Type());
  Value highI64 = castIntegerLikeTo(anchor, high, builder.getI64Type());
  if (!lowI64 || !highI64)
    return failure();

  Value shift = getI64Constant(builder, anchor->getLoc(), 40);
  Value highShifted =
      builder.create<arith::ShLIOp>(anchor->getLoc(), highI64, shift).getResult();
  return builder.create<arith::OrIOp>(anchor->getLoc(), highShifted, lowI64)
      .getResult();
}

static FailureOr<Value> packLoopSize(Operation *anchor, Value loop2, Value loop1) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);

  Value loop2I64 = castIntegerLikeTo(anchor, loop2, builder.getI64Type());
  Value loop1I64 = castIntegerLikeTo(anchor, loop1, builder.getI64Type());
  if (!loop2I64 || !loop1I64)
    return failure();

  Value shift = getI64Constant(builder, anchor->getLoc(), 21);
  Value loop2Shifted =
      builder.create<arith::ShLIOp>(anchor->getLoc(), loop2I64, shift).getResult();
  return builder.create<arith::OrIOp>(anchor->getLoc(), loop2Shifted, loop1I64)
      .getResult();
}

static FailureOr<Value>
packCopyGmToUbConfig0(Operation *anchor, ValueRange operands) {
  if (operands.size() != 11)
    return failure();

  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  auto getI64Operand = [&](unsigned idx) -> Value {
    return castIntegerLikeTo(anchor, operands[idx], builder.getI64Type());
  };

  Value sid = getI64Operand(2);
  Value nBurst = getI64Operand(3);
  Value lenBurst = getI64Operand(4);
  Value leftPadding = getI64Operand(5);
  Value rightPadding = getI64Operand(6);
  Value dataSelect = castIntegerLikeTo(anchor, operands[7], builder.getI64Type());
  Value cacheCtl = getI64Operand(8);
  if (!sid || !nBurst || !lenBurst || !leftPadding || !rightPadding ||
      !dataSelect || !cacheCtl)
    return failure();

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value,
                                         getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value {
    return builder.create<arith::OrIOp>(loc, lhs, rhs);
  };

  Value config = sid;
  config = bitOr(config, shl(nBurst, 4));
  config = bitOr(config, shl(lenBurst, 25));
  config = bitOr(config, shl(leftPadding, 46));
  config = bitOr(config, shl(rightPadding, 52));
  config = bitOr(config, shl(dataSelect, 58));
  config = bitOr(config, shl(cacheCtl, 60));
  return config;
}

static FailureOr<Value>
packCopyGmToUbConfig1(Operation *anchor, ValueRange operands) {
  if (operands.size() != 11)
    return failure();
  return packLoopPair(anchor, operands[9], operands[10]);
}

[[maybe_unused]] static FailureOr<Value>
packCopyGmToUbConfig0(Operation *anchor, Value sid, Value nBurst,
                      Value lenBurst, Value leftPadding, Value rightPadding,
                      Value dataSelect, Value cacheCtl) {
  SmallVector<Value, 11> operands(11);
  operands[2] = sid;
  operands[3] = nBurst;
  operands[4] = lenBurst;
  operands[5] = leftPadding;
  operands[6] = rightPadding;
  operands[7] = dataSelect;
  operands[8] = cacheCtl;
  return packCopyGmToUbConfig0(anchor, operands);
}

static FailureOr<Value>
packCopyUbToGmConfig0(Operation *anchor, ValueRange operands) {
  if (operands.size() != 8)
    return failure();

  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  auto getI64Operand = [&](unsigned idx) -> Value {
    return castIntegerLikeTo(anchor, operands[idx], builder.getI64Type());
  };

  Value sid = getI64Operand(2);
  Value nBurst = getI64Operand(3);
  Value lenBurst = getI64Operand(4);
  Value reserved = getI64Operand(5);
  if (!sid || !nBurst || !lenBurst || !reserved)
    return failure();

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value,
                                         getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value {
    return builder.create<arith::OrIOp>(loc, lhs, rhs);
  };

  Value config = sid;
  config = bitOr(config, shl(nBurst, 4));
  config = bitOr(config, shl(lenBurst, 25));
  config = bitOr(config, shl(reserved, 60));
  return config;
}

static FailureOr<Value>
packCopyUbToGmConfig1(Operation *anchor, ValueRange operands) {
  if (operands.size() != 8)
    return failure();
  return packLoopPair(anchor, operands[6], operands[7]);
}

[[maybe_unused]] static FailureOr<Value>
packCopyUbToGmConfig0(Operation *anchor, Value sid, Value nBurst,
                      Value lenBurst, Value reserved) {
  SmallVector<Value, 8> operands(8);
  operands[2] = sid;
  operands[3] = nBurst;
  operands[4] = lenBurst;
  operands[5] = reserved;
  return packCopyUbToGmConfig0(anchor, operands);
}

static FailureOr<Value>
packCopyUbToUbConfig(Operation *anchor, ValueRange operands) {
  if (operands.size() != 7)
    return failure();
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  auto getI64Operand = [&](unsigned idx) -> Value {
    return castIntegerLikeTo(anchor, operands[idx], builder.getI64Type());
  };

  Value nBurst = getI64Operand(3);
  Value lenBurst = getI64Operand(4);
  Value srcStride = getI64Operand(5);
  Value dstStride = getI64Operand(6);
  if (!nBurst || !lenBurst || !srcStride || !dstStride)
    return failure();

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value,
                                         getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value {
    return builder.create<arith::OrIOp>(loc, lhs, rhs);
  };

  Value config = nBurst;
  config = bitOr(config, shl(lenBurst, 16));
  config = bitOr(config, shl(srcStride, 32));
  config = bitOr(config, shl(dstStride, 48));
  return config;
}

static FailureOr<Value>
packCopyCbufToUbConfig(Operation *anchor, ValueRange operands) {
  if (operands.size() != 7)
    return failure();
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  auto getI64Operand = [&](unsigned idx) -> Value {
    return castIntegerLikeTo(anchor, operands[idx], builder.getI64Type());
  };

  Value sid = getI64Operand(2);
  Value nBurst = getI64Operand(3);
  Value lenBurst = getI64Operand(4);
  Value srcStride = getI64Operand(5);
  Value dstStride = getI64Operand(6);
  if (!sid || !nBurst || !lenBurst || !srcStride || !dstStride)
    return failure();

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value,
                                         getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value {
    return builder.create<arith::OrIOp>(loc, lhs, rhs);
  };

  Value config = sid;
  config = bitOr(config, shl(nBurst, 4));
  config = bitOr(config, shl(lenBurst, 16));
  config = bitOr(config, shl(srcStride, 32));
  config = bitOr(config, shl(dstStride, 48));
  return config;
}

static FailureOr<Value>
packCopyUbToCbufConfig(Operation *anchor, ValueRange operands) {
  if (operands.size() != 7)
    return failure();
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  auto getI64Operand = [&](unsigned idx) -> Value {
    return castIntegerLikeTo(anchor, operands[idx], builder.getI64Type());
  };

  Value sid = getI64Operand(2);
  Value nBurst = getI64Operand(3);
  Value lenBurst = getI64Operand(4);
  Value srcStride = getI64Operand(5);
  Value dstStride = getI64Operand(6);
  if (!sid || !nBurst || !lenBurst || !srcStride || !dstStride)
    return failure();

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value,
                                         getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value {
    return builder.create<arith::OrIOp>(loc, lhs, rhs);
  };

  Value config = sid;
  config = bitOr(config, shl(nBurst, 4));
  config = bitOr(config, shl(lenBurst, 16));
  config = bitOr(config, shl(srcStride, 32));
  config = bitOr(config, shl(dstStride, 48));
  return config;
}

static FailureOr<Value>
packCopyGmToCbufConfig0(Operation *anchor, Value nBurst, Value lenBurst) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value nBurstI64 = castIntegerLikeTo(anchor, nBurst, builder.getI64Type());
  Value lenBurstI64 = castIntegerLikeTo(anchor, lenBurst, builder.getI64Type());
  if (!nBurstI64 || !lenBurstI64)
    return failure();

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value,
                                         getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value {
    return builder.create<arith::OrIOp>(loc, lhs, rhs);
  };

  Value config0 = getI64Constant(builder, loc, 0); // sid
  config0 = bitOr(config0, shl(nBurstI64, 4));     // burst_num[24:4]
  config0 = bitOr(config0, shl(lenBurstI64, 25));  // burst_len[45:25]
  return config0;
}

static FailureOr<Value>
packCopyGmToCbufConfig1(Operation *anchor, Value srcStride,
                               Value dstStride) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value srcStrideI64 = castIntegerLikeTo(anchor, srcStride, builder.getI64Type());
  Value dstStrideI64 = castIntegerLikeTo(anchor, dstStride, builder.getI64Type());
  if (!srcStrideI64 || !dstStrideI64)
    return failure();

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value,
                                         getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value {
    return builder.create<arith::OrIOp>(loc, lhs, rhs);
  };

  // config1 packs burst_src_stride[39:0] and burst_dst_stride[60:40].
  return bitOr(srcStrideI64, shl(dstStrideI64, 40));
}

static FailureOr<Value>
packCopyGmToCbufMultiConfig0(Operation *anchor, Value sid,
                             Value loop1SrcStride, Value l2CacheCtl,
                             Value nValue) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value sidI64 = castIntegerLikeTo(anchor, sid, builder.getI64Type());
  Value loop1SrcStrideI64 =
      castIntegerLikeTo(anchor, loop1SrcStride, builder.getI64Type());
  Value l2CacheCtlI64 = castIntegerLikeTo(anchor, l2CacheCtl, builder.getI64Type());
  Value nValueI64 = castIntegerLikeTo(anchor, nValue, builder.getI64Type());
  if (!sidI64 || !loop1SrcStrideI64 || !l2CacheCtlI64 || !nValueI64)
    return failure();

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value,
                                         getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value {
    return builder.create<arith::OrIOp>(loc, lhs, rhs);
  };

  Value config0 = sidI64;
  config0 = bitOr(config0, shl(loop1SrcStrideI64, 4));
  config0 = bitOr(config0, shl(l2CacheCtlI64, 44));
  config0 = bitOr(config0, shl(nValueI64, 48));
  return config0;
}

static FailureOr<Value>
packCopyGmToCbufMultiConfig1(Operation *anchor, Value dValue,
                             Value loop4SrcStride, Value smallC0En) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value dValueI64 = castIntegerLikeTo(anchor, dValue, builder.getI64Type());
  Value loop4SrcStrideI64 =
      castIntegerLikeTo(anchor, loop4SrcStride, builder.getI64Type());
  Value smallC0EnI64 = castIntegerLikeTo(anchor, smallC0En, builder.getI64Type());
  if (!dValueI64 || !loop4SrcStrideI64 || !smallC0EnI64)
    return failure();

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value,
                                         getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value {
    return builder.create<arith::OrIOp>(loc, lhs, rhs);
  };

  Value config1 = dValueI64;
  config1 = bitOr(config1, shl(loop4SrcStrideI64, 21));
  config1 = bitOr(config1, shl(smallC0EnI64, 61));
  return config1;
}

static FailureOr<Value> packCopyCbufToBtConfig(Operation *anchor,
                                               Value convControl,
                                               Value nBurst, Value lenBurst,
                                               Value sourceGap,
                                               Value dstGap) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value convControlI64 =
      castIntegerLikeTo(anchor, convControl, builder.getI64Type());
  Value nBurstI64 = castIntegerLikeTo(anchor, nBurst, builder.getI64Type());
  Value lenBurstI64 = castIntegerLikeTo(anchor, lenBurst, builder.getI64Type());
  Value sourceGapI64 = castIntegerLikeTo(anchor, sourceGap, builder.getI64Type());
  Value dstGapI64 = castIntegerLikeTo(anchor, dstGap, builder.getI64Type());
  if (!convControlI64 || !nBurstI64 || !lenBurstI64 || !sourceGapI64 ||
      !dstGapI64)
    return failure();

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value,
                                         getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value {
    return builder.create<arith::OrIOp>(loc, lhs, rhs);
  };

  Value config = shl(convControlI64, 3);
  config = bitOr(config, shl(nBurstI64, 4));
  config = bitOr(config, shl(lenBurstI64, 16));
  config = bitOr(config, shl(sourceGapI64, 32));
  config = bitOr(config, shl(dstGapI64, 48));
  return config;
}

static FailureOr<Value> packCopyCbufToFbufConfig(Operation *anchor, Value nBurst,
                                                 Value lenBurst,
                                                 Value sourceGap,
                                                 Value dstGap) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value nBurstI64 = castIntegerLikeTo(anchor, nBurst, builder.getI64Type());
  Value lenBurstI64 = castIntegerLikeTo(anchor, lenBurst, builder.getI64Type());
  Value sourceGapI64 = castIntegerLikeTo(anchor, sourceGap, builder.getI64Type());
  Value dstGapI64 = castIntegerLikeTo(anchor, dstGap, builder.getI64Type());
  if (!nBurstI64 || !lenBurstI64 || !sourceGapI64 || !dstGapI64)
    return failure();

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value,
                                         getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value {
    return builder.create<arith::OrIOp>(loc, lhs, rhs);
  };

  Value config = shl(nBurstI64, 4);
  config = bitOr(config, shl(lenBurstI64, 16));
  config = bitOr(config, shl(sourceGapI64, 32));
  config = bitOr(config, shl(dstGapI64, 48));
  return config;
}

static FailureOr<Value>
packLoadCbufToS4Config0(Operation *anchor, Value mStart, Value kStart,
                        Value mStep, Value kStep) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value mStartI64 = castIntegerLikeTo(anchor, mStart, builder.getI64Type());
  Value kStartI64 = castIntegerLikeTo(anchor, kStart, builder.getI64Type());
  Value mStepI64 = castIntegerLikeTo(anchor, mStep, builder.getI64Type());
  Value kStepI64 = castIntegerLikeTo(anchor, kStep, builder.getI64Type());
  if (!mStartI64 || !kStartI64 || !mStepI64 || !kStepI64)
    return failure();

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value,
                                         getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value {
    return builder.create<arith::OrIOp>(loc, lhs, rhs);
  };

  Value config0 = mStartI64;
  config0 = bitOr(config0, shl(kStartI64, 16));
  config0 = bitOr(config0, shl(mStepI64, 32));
  config0 = bitOr(config0, shl(kStepI64, 40));
  return config0;
}

static FailureOr<Value>
packLoadCbufToS4Config1(Operation *anchor, Value srcStride, Value dstStride) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value srcStrideI64 = castIntegerLikeTo(anchor, srcStride, builder.getI64Type());
  Value dstStrideI64 = castIntegerLikeTo(anchor, dstStride, builder.getI64Type());
  if (!srcStrideI64 || !dstStrideI64)
    return failure();

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value,
                                         getI64Constant(builder, loc, amount));
  };
  return builder.create<arith::OrIOp>(loc, srcStrideI64, shl(dstStrideI64, 16))
      .getResult();
}

static FailureOr<Value>
packLoadCbufToCaConfig0(Operation *anchor, Value mStart, Value kStart,
                        Value mStep, Value kStep) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value mStartI64 = castIntegerLikeTo(anchor, mStart, builder.getI64Type());
  Value kStartI64 = castIntegerLikeTo(anchor, kStart, builder.getI64Type());
  Value mStepI64 = castIntegerLikeTo(anchor, mStep, builder.getI64Type());
  Value kStepI64 = castIntegerLikeTo(anchor, kStep, builder.getI64Type());
  if (!mStartI64 || !kStartI64 || !mStepI64 || !kStepI64)
    return failure();

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value,
                                         getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value {
    return builder.create<arith::OrIOp>(loc, lhs, rhs);
  };

  Value config0 = mStartI64;
  config0 = bitOr(config0, shl(kStartI64, 16));
  config0 = bitOr(config0, shl(mStepI64, 32));
  config0 = bitOr(config0, shl(kStepI64, 40));
  return config0;
}

static FailureOr<Value>
packLoadCbufToCaConfig1(Operation *anchor, Value srcStride, Value dstStride) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value srcStrideI64 =
      castIntegerLikeTo(anchor, srcStride, builder.getI64Type());
  Value dstStrideI64 =
      castIntegerLikeTo(anchor, dstStride, builder.getI64Type());
  if (!srcStrideI64 || !dstStrideI64)
    return failure();

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value,
                                         getI64Constant(builder, loc, amount));
  };
  return builder.create<arith::OrIOp>(loc, srcStrideI64, shl(dstStrideI64, 16))
      .getResult();
}

static FailureOr<Value>
packLoadCbufToCbConfig0(Operation *anchor, Value mStart, Value kStart,
                        Value mStep, Value kStep) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value mStartI64 = castIntegerLikeTo(anchor, mStart, builder.getI64Type());
  Value kStartI64 = castIntegerLikeTo(anchor, kStart, builder.getI64Type());
  Value mStepI64 = castIntegerLikeTo(anchor, mStep, builder.getI64Type());
  Value kStepI64 = castIntegerLikeTo(anchor, kStep, builder.getI64Type());
  if (!mStartI64 || !kStartI64 || !mStepI64 || !kStepI64)
    return failure();

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value,
                                         getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value {
    return builder.create<arith::OrIOp>(loc, lhs, rhs);
  };

  Value config0 = mStartI64;
  config0 = bitOr(config0, shl(kStartI64, 16));
  config0 = bitOr(config0, shl(mStepI64, 32));
  config0 = bitOr(config0, shl(kStepI64, 40));
  return config0;
}

static FailureOr<Value>
packLoadCbufToCbConfig1(Operation *anchor, Value srcStride, Value dstStride) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value srcStrideI64 =
      castIntegerLikeTo(anchor, srcStride, builder.getI64Type());
  Value dstStrideI64 =
      castIntegerLikeTo(anchor, dstStride, builder.getI64Type());
  if (!srcStrideI64 || !dstStrideI64)
    return failure();

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value,
                                         getI64Constant(builder, loc, amount));
  };
  return builder.create<arith::OrIOp>(loc, srcStrideI64, shl(dstStrideI64, 16))
      .getResult();
}

static Value buildMadBiasDestination(Operation *anchor,
                                     ConversionPatternRewriter &rewriter,
                                     Value dst, Value bias) {
  Type i64Ty = rewriter.getI64Type();
  Value dstAddr = rewriter.create<LLVM::PtrToIntOp>(anchor->getLoc(), i64Ty, dst);
  Value biasAddr =
      rewriter.create<LLVM::PtrToIntOp>(anchor->getLoc(), i64Ty, bias);
  Value lowMask = getI64Constant(rewriter, anchor->getLoc(), 0xffffffffULL);
  Value dstLow = rewriter.create<arith::AndIOp>(anchor->getLoc(), dstAddr, lowMask);
  Value biasLow = rewriter.create<arith::AndIOp>(anchor->getLoc(), biasAddr, lowMask);
  Value biasHigh = rewriter.create<arith::ShLIOp>(
      anchor->getLoc(), biasLow, getI64Constant(rewriter, anchor->getLoc(), 32));
  Value packed = rewriter.create<arith::OrIOp>(anchor->getLoc(), dstLow, biasHigh);
  return rewriter.create<LLVM::IntToPtrOp>(anchor->getLoc(), dst.getType(), packed);
}

static FailureOr<Value> packVbitsortConfig(Operation *anchor, Value repeatTimes) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value repeatI64 = castIntegerLikeTo(anchor, repeatTimes, builder.getI64Type());
  if (!repeatI64)
    return failure();
  return builder
      .create<arith::ShLIOp>(loc, repeatI64, getI64Constant(builder, loc, 56))
      .getResult();
}

static FailureOr<Value> convertElementOffsetToBytes(Operation *anchor, Value offset,
                                                    Type elementType) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);

  Value offsetI32 = castIntegerLikeTo(anchor, offset, builder.getI32Type());
  if (!offsetI32)
    return failure();

  unsigned bitWidth = 0;
  if (auto intType = dyn_cast<IntegerType>(elementType))
    bitWidth = intType.getWidth();
  else if (auto floatType = dyn_cast<FloatType>(elementType))
    bitWidth = floatType.getWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return failure();

  Value scale = builder.create<arith::ConstantOp>(
      anchor->getLoc(), builder.getI32IntegerAttr(bitWidth / 8));
  return builder.create<arith::MulIOp>(anchor->getLoc(), offsetI32, scale)
      .getResult();
}

[[maybe_unused]] static FailureOr<Value>
materializeDynamicPltMask(ConversionPatternRewriter &rewriter,
                          LoweringState &state, Location loc, Value laneCount,
                          Type vectorElemType) {
  Type i32Type = rewriter.getI32Type();
  Value laneCountI32 = laneCount;
  if (laneCountI32.getType() != i32Type) {
    laneCountI32 = castIntegerLikeTo(rewriter.getInsertionBlock()->getParentOp(),
                                     laneCountI32, i32Type);
    if (!laneCountI32)
      return failure();
  }

  StringRef calleeName;
  if (vectorElemType.isF32()) {
    calleeName = StringRef("llvm.hivm.plt.b32.v300");
  } else if (vectorElemType.isF16() || vectorElemType.isBF16()) {
    calleeName = StringRef("llvm.hivm.plt.b16.v300");
  } else if (auto intType = dyn_cast<IntegerType>(vectorElemType)) {
    if (intType.getWidth() == 32)
      calleeName = StringRef("llvm.hivm.plt.b32.v300");
    else if (intType.getWidth() == 16)
      calleeName = StringRef("llvm.hivm.plt.b16.v300");
    else if (intType.getWidth() == 8)
      calleeName = StringRef("llvm.hivm.plt.b8.v300");
  }
  if (calleeName.empty())
    return failure();

  Type maskType = VectorType::get({256}, rewriter.getI1Type());
  auto funcType =
      rewriter.getFunctionType(TypeRange{i32Type}, TypeRange{maskType, i32Type});
  auto call = rewriter.create<func::CallOp>(loc, calleeName, funcType.getResults(),
                                            ValueRange{laneCountI32});
  state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
  return call.getResult(0);
}

static FailureOr<StringRef> buildCarryBinaryCallee(MLIRContext *context,
                                                   Type resultType,
                                                   StringRef stem) {
  std::string vec =
      getElementTypeFragment(cast<pto::VRegType>(resultType).getElementType());
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
    return failure();
  return StringAttr::get(context, "llvm.hivm." + stem.str() + ".v" +
                                      std::to_string(*lanes) + vec)
      .getValue();
}

template <typename UnaryOp>
static StringRef getUnaryMaskedStem() {
  if constexpr (std::is_same_v<UnaryOp, pto::VabsOp>)
    return "vabs";
  if constexpr (std::is_same_v<UnaryOp, pto::VexpOp>)
    return "vexp";
  if constexpr (std::is_same_v<UnaryOp, pto::VlnOp>)
    return "vln";
  if constexpr (std::is_same_v<UnaryOp, pto::VnegOp>)
    return "vneg";
  if constexpr (std::is_same_v<UnaryOp, pto::VsqrtOp>)
    return "vsqrt";
  if constexpr (std::is_same_v<UnaryOp, pto::VreluOp>)
    return "vrelu";
  if constexpr (std::is_same_v<UnaryOp, pto::VnotOp>)
    return "vnot";
  return {};
}

template <typename BinaryOp>
static StringRef getBinaryMaskedStem() {
  if constexpr (std::is_same_v<BinaryOp, pto::VaddOp>)
    return "vadd";
  if constexpr (std::is_same_v<BinaryOp, pto::VsubOp>)
    return "vsub";
  if constexpr (std::is_same_v<BinaryOp, pto::VmulOp>)
    return "vmul";
  if constexpr (std::is_same_v<BinaryOp, pto::VdivOp>)
    return "vdiv";
  if constexpr (std::is_same_v<BinaryOp, pto::VmaxOp>)
    return "vmax";
  if constexpr (std::is_same_v<BinaryOp, pto::VminOp>)
    return "vmin";
  if constexpr (std::is_same_v<BinaryOp, pto::VandOp>)
    return "vand";
  if constexpr (std::is_same_v<BinaryOp, pto::VorOp>)
    return "vor";
  if constexpr (std::is_same_v<BinaryOp, pto::VxorOp>)
    return "vxor";
  if constexpr (std::is_same_v<BinaryOp, pto::VshlOp>)
    return "vshl";
  if constexpr (std::is_same_v<BinaryOp, pto::VshrOp>)
    return "vshr";
  if constexpr (std::is_same_v<BinaryOp, pto::VpreluOp>)
    return "vprelu";
  return {};
}

template <typename CarryOp>
static StringRef getCarryBinaryStem() {
  if constexpr (std::is_same_v<CarryOp, pto::VaddcOp>)
    return "vaddc";
  if constexpr (std::is_same_v<CarryOp, pto::VsubcOp>)
    return "vsubc";
  if constexpr (std::is_same_v<CarryOp, pto::VaddcsOp>)
    return "vaddcs";
  if constexpr (std::is_same_v<CarryOp, pto::VsubcsOp>)
    return "vsubcs";
  return {};
}

template <typename CarryOp>
static constexpr bool hasCarryInput() {
  return std::is_same_v<CarryOp, pto::VaddcsOp> ||
         std::is_same_v<CarryOp, pto::VsubcsOp>;
}

static FailureOr<StringRef> buildVselCallee(MLIRContext *context,
                                            Type resultType) {
  std::string vec =
      getElementTypeFragment(cast<pto::VRegType>(resultType).getElementType());
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
    return failure();
  return StringAttr::get(context, "llvm.hivm.vsel.v" + std::to_string(*lanes) +
                                      vec)
      .getValue();
}

static FailureOr<StringRef> buildVselrCallee(MLIRContext *context,
                                             Type resultType) {
  Type elemType = getElementTypeFromVectorLike(resultType);
  auto lanes = getElementCountFromVectorLike(resultType);
  if (!elemType || !lanes)
    return failure();

  std::string vec = getElementTypeFragment(elemType);
  if (auto floatType = dyn_cast<FloatType>(elemType);
      floatType && floatType.isF32())
    vec = "u32";
  if (vec.empty())
    return failure();

  return StringAttr::get(context, "llvm.hivm.vselr.v" + std::to_string(*lanes) +
                                      vec)
      .getValue();
}

static FailureOr<StringRef> buildVdupCallee(MLIRContext *context, pto::VdupOp op) {
  Type inputType = op.getInput().getType();
  Type resultType = op.getResult().getType();
  std::string vec = getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
    return failure();

  if (isa<VectorType, pto::VRegType>(inputType)) {
    StringRef position = op.getPosition().value_or("LOWEST");
    StringRef family = position == "HIGHEST" ? "vdupm" : "vdup";
    return StringAttr::get(context, "llvm.hivm." + family.str() + ".v" +
                                        std::to_string(*lanes) + vec + ".z")
        .getValue();
  }

  return StringAttr::get(context, "llvm.hivm.vdups.v" + std::to_string(*lanes) +
                                      vec + ".z")
      .getValue();
}

static FailureOr<StringRef> buildVbrCallee(MLIRContext *context,
                                          Type semanticElementType) {
  std::string scalar = getVbrScalarFragment(semanticElementType);
  if (scalar.empty())
    return failure();
  return StringAttr::get(context, "llvm.hivm.vbr." + scalar + ".v300").getValue();
}

static FailureOr<StringRef> buildPstuCallee(MLIRContext *context, pto::PstuOp op) {
  if (auto maskType = dyn_cast<pto::MaskType>(op.getValue().getType())) {
    if (maskType.isB16())
      return StringAttr::get(context, "llvm.hivm.pstu.b16").getValue();
    if (maskType.isB32())
      return StringAttr::get(context, "llvm.hivm.pstu.b32").getValue();
  }
  return failure();
}

static StringRef buildVstusCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vstus").getValue();
}

static StringRef buildVsturCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vstur").getValue();
}

static StringRef buildInitAlignCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.init.vector.align.data").getValue();
}

template <typename QueryOp>
static StringRef buildRuntimeQueryCallee(MLIRContext *context);

template <>
StringRef buildRuntimeQueryCallee<pto::GetCtrlOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.CTRL").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetVms4SrOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.VMS4.SR").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetTidXOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.TID.X").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetTidYOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.TID.Y").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetTidZOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.TID.Z").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetBlockDimXOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.BLOCK.DIM.X").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetBlockDimYOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.BLOCK.DIM.Y").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetBlockDimZOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.BLOCK.DIM.Z").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetGridDimXOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.GRID.DIM.X").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetGridDimYOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.GRID.DIM.Y").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetGridDimZOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.GRID.DIM.Z").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetBlockIdxXOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.BLOCK.IDX.X").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetBlockIdxYOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.BLOCK.IDX.Y").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetBlockIdxZOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.BLOCK.IDX.Z").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetVecCoreIdOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.tpe.get.VECCOREID").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetLaneIdOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.laneID").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetClock32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.CLOCK32").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetClock64Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.CLOCK64").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetLaneMaskEqOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.LANEMASK.EQ").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetLaneMaskLeOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.LANEMASK.LE").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetLaneMaskLtOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.LANEMASK.LT").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetLaneMaskGeOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.LANEMASK.GE").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetLaneMaskGtOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.get.LANEMASK.GT").getValue();
}

static StringRef buildSprclrCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.sprclr").getValue();
}

template <typename ConfigOp>
static StringRef buildUnaryConfigCallee(MLIRContext *context);

template <>
StringRef buildUnaryConfigCallee<pto::SetCtrlOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.CTRL").getValue();
}

static StringRef buildStoreVfSimtInfoCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.store.vfsimt.info").getValue();
}

static StringRef buildSyncthreadsCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.sync.workitems").getValue();
}

static StringRef buildThreadfenceCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.fence.workitems").getValue();
}

static StringRef buildThreadfenceBlockCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.fenceblock.workitems").getValue();
}

static StringRef buildVstarCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vstar").getValue();
}

static StringRef buildVstasCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vstas").getValue();
}

template <typename VoteOp>
static StringRef buildVoteCallee(MLIRContext *context);

template <>
StringRef buildVoteCallee<pto::VoteAllOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vote.all").getValue();
}

template <>
StringRef buildVoteCallee<pto::VoteAnyOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vote.any").getValue();
}

template <>
StringRef buildVoteCallee<pto::VoteUniOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vote.uni").getValue();
}

template <>
StringRef buildVoteCallee<pto::VoteBallotOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vote.ballot").getValue();
}

template <typename BinaryOp>
static StringRef buildBinaryI64PureCallee(MLIRContext *context);

template <>
StringRef buildBinaryI64PureCallee<pto::Sbitset0Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SBITSET0").getValue();
}

template <>
StringRef buildBinaryI64PureCallee<pto::Sbitset1Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SBITSET1").getValue();
}

template <typename ShuffleOp>
static FailureOr<StringRef> buildShuffleCallee(MLIRContext *context,
                                               Type valueType);

template <>
FailureOr<StringRef> buildShuffleCallee<pto::ShuffleIdxOp>(MLIRContext *context,
                                                           Type valueType) {
  std::string elem = getShuffleIntrinsicTypeFragment(valueType);
  if (elem.empty())
    return failure();
  return StringAttr::get(context, "llvm.hivm.shfl.idx." + elem).getValue();
}

template <>
FailureOr<StringRef> buildShuffleCallee<pto::ShuffleUpOp>(MLIRContext *context,
                                                          Type valueType) {
  std::string elem = getShuffleIntrinsicTypeFragment(valueType);
  if (elem.empty())
    return failure();
  return StringAttr::get(context, "llvm.hivm.shfl.up." + elem).getValue();
}

template <>
FailureOr<StringRef> buildShuffleCallee<pto::ShuffleDownOp>(MLIRContext *context,
                                                            Type valueType) {
  std::string elem = getShuffleIntrinsicTypeFragment(valueType);
  if (elem.empty())
    return failure();
  return StringAttr::get(context, "llvm.hivm.shfl.down." + elem).getValue();
}

template <>
FailureOr<StringRef> buildShuffleCallee<pto::ShuffleBflyOp>(MLIRContext *context,
                                                            Type valueType) {
  std::string elem = getShuffleIntrinsicTypeFragment(valueType);
  if (elem.empty())
    return failure();
  return StringAttr::get(context, "llvm.hivm.shfl.bfly." + elem).getValue();
}

static Value buildShuffleControlValue(OpBuilder &builder, Location loc,
                                      Value controlValue, int64_t widthValue,
                                      unsigned controlMask) {
  Value lowBits = builder.create<arith::AndIOp>(
      loc, controlValue, getI32Constant(builder, loc, 0x1f));
  Value encodedWidth =
      getI32Constant(builder, loc, static_cast<uint32_t>(32 - widthValue) << 16);
  Value encodedMask =
      getI32Constant(builder, loc, static_cast<uint32_t>(controlMask) << 8);
  Value highBits = builder.create<arith::OrIOp>(loc, encodedWidth, encodedMask);
  return builder.create<arith::OrIOp>(loc, highBits, lowBits);
}

template <typename ReduxOp>
static FailureOr<StringRef> buildReduxCallee(MLIRContext *context,
                                             Type valueType,
                                             Attribute signednessAttr);

template <>
FailureOr<StringRef> buildReduxCallee<pto::ReduxAddOp>(MLIRContext *context,
                                                      Type valueType,
                                                      Attribute signednessAttr) {
  std::string elem = getReduxIntrinsicTypeFragment(valueType, signednessAttr);
  if (elem.empty())
    return failure();
  return StringAttr::get(context, "llvm.hivm.redux.add." + elem).getValue();
}

template <>
FailureOr<StringRef> buildReduxCallee<pto::ReduxMaxOp>(MLIRContext *context,
                                                      Type valueType,
                                                      Attribute signednessAttr) {
  std::string elem = getReduxIntrinsicTypeFragment(valueType, signednessAttr);
  if (elem.empty())
    return failure();
  return StringAttr::get(context, "llvm.hivm.redux.max." + elem).getValue();
}

template <>
FailureOr<StringRef> buildReduxCallee<pto::ReduxMinOp>(MLIRContext *context,
                                                      Type valueType,
                                                      Attribute signednessAttr) {
  std::string elem = getReduxIntrinsicTypeFragment(valueType, signednessAttr);
  if (elem.empty())
    return failure();
  return StringAttr::get(context, "llvm.hivm.redux.min." + elem).getValue();
}

template <typename AtomicOp>
static FailureOr<StringRef> buildAtomicCallee(MLIRContext *context,
                                              Type ptrType, Type valueType,
                                              Attribute signednessAttr);

static FailureOr<StringRef> buildAtomicCalleeName(MLIRContext *context,
                                                  Type ptrType, Type valueType,
                                                  Attribute signednessAttr,
                                                  StringRef opName) {
  std::string elem = getAtomicElementTypeFragment(valueType, signednessAttr);
  if (elem.empty())
    return failure();
  auto ptrTy = dyn_cast<pto::PtrType>(ptrType);
  if (!ptrTy)
    return failure();

  StringRef space;
  switch (ptrTy.getMemorySpace().getAddressSpace()) {
  case pto::AddressSpace::GM:
    space = "G";
    break;
  case pto::AddressSpace::VEC:
    if (valueType.isInteger(64))
      return failure();
    space = "S";
    break;
  default:
    return failure();
  }

  return StringAttr::get(context, "llvm.hivm.atom." + opName.str() + "." +
                                      space.str() + "." + elem)
      .getValue();
}

#define PTO_BUILD_ATOMIC_CALLEE(OP, NAME)                                      \
  template <>                                                                  \
  [[maybe_unused]] FailureOr<StringRef> buildAtomicCallee<pto::OP>(            \
      MLIRContext *context, Type ptrType, Type valueType,                      \
      Attribute signednessAttr) {                                              \
    return buildAtomicCalleeName(context, ptrType, valueType, signednessAttr,  \
                                 NAME);                                        \
  }

PTO_BUILD_ATOMIC_CALLEE(AtomicCasOp, "CAS")
PTO_BUILD_ATOMIC_CALLEE(AtomicExchOp, "EXCH")
PTO_BUILD_ATOMIC_CALLEE(AtomicAddOp, "ADD")
PTO_BUILD_ATOMIC_CALLEE(AtomicSubOp, "SUB")
PTO_BUILD_ATOMIC_CALLEE(AtomicMinOp, "MIN")
PTO_BUILD_ATOMIC_CALLEE(AtomicMaxOp, "MAX")
PTO_BUILD_ATOMIC_CALLEE(AtomicAndOp, "AND")
PTO_BUILD_ATOMIC_CALLEE(AtomicOrOp, "OR")
PTO_BUILD_ATOMIC_CALLEE(AtomicXorOp, "XOR")

#undef PTO_BUILD_ATOMIC_CALLEE

static FailureOr<StringRef> buildL1CacheLoadCallee(MLIRContext *context,
                                                   Type resultType,
                                                   pto::L1Cache l1cache) {
  std::string elem;
  if (auto intType = dyn_cast<IntegerType>(resultType)) {
    if (intType.getWidth() == 8)
      elem = "s8";
    else if (intType.getWidth() == 16)
      elem = "s16";
    else if (intType.getWidth() == 32)
      elem = "s32";
    else if (intType.getWidth() == 64)
      elem = "s64";
  } else if (resultType.isF16() || resultType.isBF16()) {
    elem = "s16";
  } else if (resultType.isF32()) {
    elem = "s32";
  } else if (resultType.isF64()) {
    elem = "s64";
  } else if (pto::isPTOFloat8Type(resultType) ||
             pto::isPTOHiFloat8Type(resultType)) {
    elem = "s8";
  }
  if (elem.empty())
    return failure();
  StringRef l1cacheName =
      l1cache == pto::L1Cache::Cache ? "cache" : "uncache";
  return StringAttr::get(context,
                         "llvm.hivm.ldg." + l1cacheName.str() + "." + elem)
      .getValue();
}

static FailureOr<StringRef> buildL1CacheStoreCallee(MLIRContext *context,
                                                    Type valueType,
                                                    pto::L1Cache l1cache) {
  std::string elem;
  if (auto intType = dyn_cast<IntegerType>(valueType)) {
    if (intType.getWidth() == 8)
      elem = "b8";
    else if (intType.getWidth() == 16)
      elem = "b16";
    else if (intType.getWidth() == 32)
      elem = "b32";
    else if (intType.getWidth() == 64)
      elem = "b64";
  } else if (valueType.isF16() || valueType.isBF16()) {
    elem = "b16";
  } else if (valueType.isF32()) {
    elem = "b32";
  } else if (valueType.isF64()) {
    elem = "b64";
  } else if (pto::isPTOFloat8Type(valueType) ||
             pto::isPTOHiFloat8Type(valueType)) {
    elem = "b8";
  }
  if (elem.empty())
    return failure();
  StringRef l1cacheName =
      l1cache == pto::L1Cache::Cache ? "cache" : "uncache";
  return StringAttr::get(context,
                         "llvm.hivm.stg." + l1cacheName.str() + "." + elem)
      .getValue();
}

template <typename ScalarOp>
static StringRef buildScalarIntrinsicCallee(MLIRContext *context);

template <>
StringRef buildScalarIntrinsicCallee<pto::PrmtOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.prmt").getValue();
}

static FailureOr<StringRef>
buildMulhiCallee(MLIRContext *context, Type resultType,
                 pto::Signedness signedness) {
  if (resultType.isInteger(32)) {
    return StringAttr::get(
               context, signedness == pto::Signedness::Unsigned
                            ? "llvm.hivm.mulhi.ui"
                            : "llvm.hivm.mulhi.i")
        .getValue();
  }
  if (resultType.isInteger(64) && signedness == pto::Signedness::Unsigned)
    return StringAttr::get(context, "llvm.hivm.mul64hi.ui").getValue();
  return failure();
}

static FailureOr<StringRef>
buildMulI32ToI64Callee(MLIRContext *context, pto::Signedness signedness) {
  return StringAttr::get(
             context, signedness == pto::Signedness::Unsigned
                          ? "llvm.hivm.mul.i32toi64.ui"
                          : "llvm.hivm.mul.i32toi64.i")
      .getValue();
}

static std::string getScalarFloatBuiltinFragment(Type type) {
  if (type.isF32())
    return "f32";
  if (type.isF16())
    return "f16";
  if (type.isBF16())
    return "bf16";
  return {};
}

static std::string getLLVMFloatBuiltinFragment(Type type) {
  std::string scalar = getScalarFloatBuiltinFragment(type);
  if (!scalar.empty())
    return scalar;

  auto vecType = dyn_cast<VectorType>(type);
  if (!vecType || vecType.getRank() != 1 || vecType.getDimSize(0) != 2)
    return {};
  Type elementType = vecType.getElementType();
  if (elementType.isF16())
    return "v2f16";
  if (elementType.isBF16())
    return "v2bf16";
  return {};
}

static std::string getHIVMFloatBuiltinFragment(Type type) {
  std::string scalar = getScalarFloatBuiltinFragment(type);
  if (!scalar.empty())
    return scalar;

  auto vecType = dyn_cast<VectorType>(type);
  if (!vecType || vecType.getRank() != 1 || vecType.getDimSize(0) != 2)
    return {};
  Type elementType = vecType.getElementType();
  if (elementType.isF16())
    return "f16x2";
  if (elementType.isBF16())
    return "bf16x2";
  return {};
}

static FailureOr<StringRef> buildSqrtCallee(MLIRContext *context, Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f32" && elem != "f16" && elem != "v2f16")
    return failure();
  return StringAttr::get(context, "llvm.sqrt." + elem).getValue();
}

static std::string getScalarHIVMFloatShortFragment(Type type) {
  if (type.isF32())
    return "f";
  if (type.isF16())
    return "h";
  if (type.isBF16())
    return "y";
  return {};
}

template <typename UnaryOp>
static FailureOr<StringRef> buildUnaryScalarMathCallee(MLIRContext *context,
                                                       Type valueType);

template <>
FailureOr<StringRef> buildUnaryScalarMathCallee<pto::AbsFOp>(MLIRContext *context,
                                                             Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f32" && elem != "v2f16" && elem != "v2bf16")
    return failure();
  return StringAttr::get(context, "llvm.fabs." + elem).getValue();
}

template <>
FailureOr<StringRef> buildUnaryScalarMathCallee<pto::ExpOp>(MLIRContext *context,
                                                            Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f32" && elem != "f16" && elem != "v2f16")
    return failure();
  return StringAttr::get(context, "llvm.exp." + elem).getValue();
}

template <>
FailureOr<StringRef> buildUnaryScalarMathCallee<pto::LogOp>(MLIRContext *context,
                                                            Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f32" && elem != "f16" && elem != "v2f16")
    return failure();
  return StringAttr::get(context, "llvm.log." + elem).getValue();
}

template <>
FailureOr<StringRef> buildUnaryScalarMathCallee<pto::CeilOp>(MLIRContext *context,
                                                             Type valueType) {
  std::string elem = getScalarHIVMFloatShortFragment(valueType);
  if (elem.empty())
    return failure();
  return StringAttr::get(context, "llvm.hivm.ceil." + elem).getValue();
}

template <>
FailureOr<StringRef> buildUnaryScalarMathCallee<pto::FloorOp>(MLIRContext *context,
                                                              Type valueType) {
  std::string elem = getScalarHIVMFloatShortFragment(valueType);
  if (elem.empty())
    return failure();
  return StringAttr::get(context, "llvm.hivm.floor." + elem).getValue();
}

template <>
FailureOr<StringRef> buildUnaryScalarMathCallee<pto::RintOp>(MLIRContext *context,
                                                             Type valueType) {
  std::string elem = getScalarHIVMFloatShortFragment(valueType);
  if (elem.empty())
    return failure();
  return StringAttr::get(context, "llvm.hivm.rint." + elem).getValue();
}

template <>
FailureOr<StringRef> buildUnaryScalarMathCallee<pto::RoundOp>(MLIRContext *context,
                                                              Type valueType) {
  std::string elem = getScalarHIVMFloatShortFragment(valueType);
  if (elem.empty())
    return failure();
  return StringAttr::get(context, "llvm.hivm.round." + elem).getValue();
}

template <typename BinaryOp>
static FailureOr<StringRef> buildBinaryScalarMathCallee(MLIRContext *context,
                                                        Type valueType);

template <>
FailureOr<StringRef> buildBinaryScalarMathCallee<pto::FMinOp>(MLIRContext *context,
                                                              Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f32" && elem != "bf16" && elem != "v2f16" &&
      elem != "v2bf16")
    return failure();
  return StringAttr::get(context, "llvm.minnum." + elem).getValue();
}

template <>
FailureOr<StringRef> buildBinaryScalarMathCallee<pto::FMaxOp>(MLIRContext *context,
                                                              Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f32" && elem != "bf16" && elem != "v2f16" &&
      elem != "v2bf16")
    return failure();
  return StringAttr::get(context, "llvm.maxnum." + elem).getValue();
}

template <>
FailureOr<StringRef> buildBinaryScalarMathCallee<pto::PowOp>(MLIRContext *context,
                                                             Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f32" && elem != "f16" && elem != "v2f16")
    return failure();
  return StringAttr::get(context, "llvm.pow." + elem).getValue();
}

static FailureOr<StringRef> buildFmaCallee(MLIRContext *context, Type valueType) {
  std::string elem = getHIVMFloatBuiltinFragment(valueType);
  if (elem.empty())
    return failure();
  return StringAttr::get(context, "llvm.hivm.ffma." + elem + ".rrr").getValue();
}

static std::string getConvertScalarFragment(Type type,
                                            Attribute signednessAttr) {
  if (auto vecType = dyn_cast<VectorType>(type)) {
    if (vecType.getRank() != 1 || vecType.getDimSize(0) != 2)
      return {};
    Type elementType = vecType.getElementType();
    if (std::string elem = getLowPrecisionElementFragment(elementType);
        !elem.empty() && !pto::isPTOFloat4PackedType(elementType))
      return elem + "x2";
    if (elementType.isF32())
      return "f32x2";
    if (elementType.isF16())
      return "f16x2";
    if (elementType.isBF16())
      return "bf16x2";
    return {};
  }
  if (type.isF32())
    return "fp32";
  if (type.isF16())
    return "fp16";
  if (type.isBF16())
    return "bf16";
  if (std::string elem = getLowPrecisionElementFragment(type); !elem.empty())
    return elem;
  auto intType = dyn_cast<IntegerType>(type);
  if (!intType || (intType.getWidth() != 32 && intType.getWidth() != 64) ||
      !signednessAttr)
    return {};
  auto signedness = cast<pto::SignednessAttr>(signednessAttr).getValue();
  return std::string(signedness == pto::Signedness::Unsigned ? "u" : "s") +
         std::to_string(intType.getWidth());
}

static FailureOr<StringRef> buildConvertCallee(MLIRContext *context,
                                               Type srcType, Type dstType,
                                               Attribute signednessAttr) {
  std::string src = getConvertScalarFragment(srcType, signednessAttr);
  std::string dst = getConvertScalarFragment(dstType, signednessAttr);
  if (src.empty() || dst.empty())
    return failure();
  return StringAttr::get(context,
                         "llvm.hivm." + src + ".to." + dst)
      .getValue();
}

static FailureOr<StringRef> buildVldsPostCallee(MLIRContext *context,
                                                Type resultType) {
  std::string vec = getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
    return failure();
  return StringAttr::get(context, "llvm.hivm.vldsx1.post.v" +
                                      std::to_string(*lanes) + vec)
      .getValue();
}

static FailureOr<StringRef> buildVstsPostCallee(MLIRContext *context,
                                                Type valueType) {
  std::string vec = getElementTypeFragment(getElementTypeFromVectorLike(valueType));
  auto lanes = getElementCountFromVectorLike(valueType);
  if (vec.empty() || !lanes)
    return failure();
  return StringAttr::get(context, "llvm.hivm.vstsx1.post.v" +
                                      std::to_string(*lanes) + vec)
      .getValue();
}

static StringRef buildVldasCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vldas").getValue();
}

static FailureOr<StringRef> buildVldusCallee(MLIRContext *context,
                                             Type resultType) {
  std::string vec = getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
    return failure();
  return StringAttr::get(context, "llvm.hivm.vldus.v" +
                                      std::to_string(*lanes) + vec)
      .getValue();
}

static FailureOr<StringRef> buildVcmpCallee(MLIRContext *context, Type inputType,
                                            StringRef cmpMode,
                                            bool isScalarCompare) {
  std::string elem = getElementTypeFragment(getElementTypeFromVectorLike(inputType));
  if (elem.empty())
    return failure();
  StringRef stem = isScalarCompare ? "vcmps" : "vcmp";
  return StringAttr::get(context, "llvm.hivm." + stem.str() + "." +
                                      cmpMode.str() + "." + elem + ".z")
      .getValue();
}

template <typename VecScalarOp>
static StringRef getVecScalarMaskedStem() {
  if constexpr (std::is_same_v<VecScalarOp, pto::VmulsOp>)
    return "vmuls";
  if constexpr (std::is_same_v<VecScalarOp, pto::VaddsOp>)
    return "vadds";
  if constexpr (std::is_same_v<VecScalarOp, pto::VmaxsOp>)
    return "vmaxs";
  if constexpr (std::is_same_v<VecScalarOp, pto::VminsOp>)
    return "vmins";
  if constexpr (std::is_same_v<VecScalarOp, pto::VlreluOp>)
    return "vlrelu";
  if constexpr (std::is_same_v<VecScalarOp, pto::VshlsOp>)
    return "vshls";
  if constexpr (std::is_same_v<VecScalarOp, pto::VshrsOp>)
    return "vshrs";
  return {};
}

template <typename ReductionOp>
static StringRef getReductionUnaryStem() {
  if constexpr (std::is_same_v<ReductionOp, pto::VcaddOp>)
    return "vcadd";
  if constexpr (std::is_same_v<ReductionOp, pto::VcmaxOp>)
    return "vcmax";
  if constexpr (std::is_same_v<ReductionOp, pto::VcminOp>)
    return "vcmin";
  if constexpr (std::is_same_v<ReductionOp, pto::VcgaddOp>)
    return "vcgadd";
  if constexpr (std::is_same_v<ReductionOp, pto::VcgmaxOp>)
    return "vcgmax";
  if constexpr (std::is_same_v<ReductionOp, pto::VcgminOp>)
    return "vcgmin";
  if constexpr (std::is_same_v<ReductionOp, pto::VcpaddOp>)
    return "vcpadd";
  return {};
}

static FailureOr<StringRef> buildCopyGmToUbCallee(MLIRContext *context,
                                                  Type sourceType) {
  auto ptrType = dyn_cast<pto::PtrType>(sourceType);
  if (!ptrType)
    return failure();
  Type elementType = ptrType.getElementType();
  if ((isa<IntegerType>(elementType) &&
       cast<IntegerType>(elementType).getWidth() == 64) ||
      elementType.isF64()) {
    return StringAttr::get(context, "llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.s32.DV")
        .getValue();
  }
  std::string elem = getCopyElementFragment(elementType);
  if (elem.empty())
    return failure();
  return StringAttr::get(context, "llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2." + elem +
                                      ".DV")
      .getValue();
}

static StringRef buildCopyUbToGmCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.DV")
      .getValue();
}

static StringRef buildCopyUbToUbCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.MOV.UB.TO.UB.v310").getValue();
}

static StringRef buildCopyCbufToUbCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.MOV.L1.TO.UB.v310").getValue();
}

static StringRef buildCopyUbToCbufCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.MOV.UB.TO.L1.v310").getValue();
}

static FailureOr<StringRef> buildOrdinaryMadCallee(MLIRContext *context,
                                                   pto::MadRawOpInterface op) {
  auto lhsType = dyn_cast<pto::PtrType>(op.getLhs().getType());
  auto rhsType = dyn_cast<pto::PtrType>(op.getRhs().getType());
  auto dstType = dyn_cast<pto::PtrType>(op.getDst().getType());
  if (!lhsType || !rhsType || !dstType)
    return failure();

  return buildMadTypedCalleeName(context, lhsType.getElementType(),
                                 rhsType.getElementType(),
                                 dstType.getElementType());
}

static FailureOr<StringRef> buildMxMadCallee(MLIRContext *context,
                                             pto::MadRawOpInterface op) {
  auto lhsType = dyn_cast<pto::PtrType>(op.getLhs().getType());
  auto rhsType = dyn_cast<pto::PtrType>(op.getRhs().getType());
  if (!lhsType || !rhsType)
    return failure();
  if (isMxElementType(lhsType.getElementType()) &&
      isMxElementType(rhsType.getElementType())) {
    return buildMadMxCalleeName(context, lhsType.getElementType(),
                                rhsType.getElementType());
  }
  return failure();
}

static FailureOr<StringRef> buildCopyGmToCbufCallee(MLIRContext *context,
                                                    Type sourceType) {
  auto ptrType = dyn_cast<pto::PtrType>(sourceType);
  if (!ptrType)
    return failure();
  std::string elem = getCopyElementFragment(ptrType.getElementType());
  if (elem.empty())
    return failure();
  return StringAttr::get(context, "llvm.hivm.MOV.OUT.TO.L1.ALIGN.V2." + elem +
                                      ".DV")
      .getValue();
}

static FailureOr<StringRef>
buildCopyGmToCbufMultiNd2NzCallee(MLIRContext *context, Type sourceType) {
  auto ptrType = dyn_cast<pto::PtrType>(sourceType);
  if (!ptrType)
    return failure();
  std::string elem = getNd2NzCopyElementFragment(ptrType.getElementType());
  if (elem.empty())
    return failure();
  return StringAttr::get(context, "llvm.hivm.MOV.OUT.TO.L1.MULTI.ND2NZ." +
                                      elem + ".V310")
      .getValue();
}

static std::string getDn2NzCopyElementFragment(Type type) {
  auto ptrType = dyn_cast<pto::PtrType>(type);
  if (!ptrType)
    return {};

  Type elementType = ptrType.getElementType();
  std::string typeText;
  llvm::raw_string_ostream os(typeText);
  elementType.print(os);
  os.flush();
  std::string lower = StringRef(typeText).lower();
  if (StringRef(lower).contains("e4m3") || StringRef(lower).contains("e5m2") ||
      StringRef(lower).contains("e8m0") || StringRef(lower).contains("hif8"))
    return "u8";

  if (elementType.isF16() || elementType.isBF16())
    return "u16";
  if (elementType.isF32())
    return "u32";

  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    switch (intType.getWidth()) {
    case 8:
      return "u8";
    case 16:
      return "u16";
    case 32:
      return "u32";
    default:
      return {};
    }
  }
  return {};
}

static FailureOr<StringRef>
buildCopyGmToCbufMultiDn2NzCallee(MLIRContext *context, Type sourceType) {
  auto ptrType = dyn_cast<pto::PtrType>(sourceType);
  if (!ptrType)
    return failure();
  std::string elem = getDn2NzCopyElementFragment(sourceType);
  if (elem.empty())
    return failure();
  return StringAttr::get(context,
                         "llvm.hivm.MOV.OUT.TO.L1.MULTI.DN2NZ." + elem)
      .getValue();
}

static FailureOr<StringRef> buildLoadCbufToCaCallee(MLIRContext *context,
                                                     Type sourceType) {
  auto ptrType = dyn_cast<pto::PtrType>(sourceType);
  if (!ptrType)
    return failure();
  std::string elem = getL0LoadElementFragment(ptrType.getElementType());
  if (elem.empty())
    return failure();
  return StringAttr::get(context, "llvm.hivm.LOAD.L1.TO.L0A.2Dv2." + elem)
      .getValue();
}

static StringRef buildLoadCbufToCaS4Callee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.LOAD.L1.TO.L0A.2Dv2.s4")
      .getValue();
}

static FailureOr<StringRef> buildLoadCbufToCbCallee(MLIRContext *context,
                                                     Type sourceType) {
  auto ptrType = dyn_cast<pto::PtrType>(sourceType);
  if (!ptrType)
    return failure();
  std::string elem = getL0LoadElementFragment(ptrType.getElementType());
  if (elem.empty())
    return failure();
  return StringAttr::get(context, "llvm.hivm.LOAD.L1.TO.L0B.2Dv2." + elem)
      .getValue();
}

static StringRef buildLoadCbufToCbS4Callee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.LOAD.L1.TO.L0B.2Dv2.s4")
      .getValue();
}

static StringRef buildLoadCbufToCaMxCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.LOAD.L1.TO.L0A.MX.2Dv2.v")
      .getValue();
}

[[maybe_unused]] static StringRef buildLoadCbufToCbMxCallee(
    MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.LOAD.L1.TO.L0B.MX.2Dv2.v")
      .getValue();
}

static StringRef buildCopyMatrixCcToGmCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.FIX.L0C.TO.OUT.f32.EXT")
      .getValue();
}

static StringRef buildCopyMatrixCcToCbufCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.FIX.L0C.TO.L1.f32.EXT")
      .getValue();
}

static FailureOr<StringRef> buildCopyMatrixCcToUbCallee(MLIRContext *context,
                                                         Type destinationType) {
  auto ptrType = dyn_cast<pto::PtrType>(destinationType);
  if (!ptrType)
    return failure();
  Type dstElem = ptrType.getElementType();
  if (dstElem.isF16())
    return StringAttr::get(context, "llvm.hivm.FIX.L0C.TO.UB.f322f16.EXT")
        .getValue();
  if (dstElem.isF32())
    return StringAttr::get(context, "llvm.hivm.FIX.L0C.TO.UB.f32.EXT")
        .getValue();
  return failure();
}

static FailureOr<StringRef> buildCopyCbufToBtCallee(pto::CopyCbufToBtOp op) {
  auto ptrType = dyn_cast<pto::PtrType>(op.getSource().getType());
  if (!ptrType)
    return failure();
  Type srcElem = ptrType.getElementType();
  if (srcElem.isF16())
    return StringAttr::get(op.getContext(), "llvm.hivm.MOV.L1.TO.BT.f16")
        .getValue();
  if (srcElem.isBF16())
    return StringAttr::get(op.getContext(), "llvm.hivm.MOV.L1.TO.BT.bf16")
        .getValue();
  if (srcElem.isF32())
    return StringAttr::get(op.getContext(), "llvm.hivm.MOV.L1.TO.BT.f32")
        .getValue();
  if (auto intType = dyn_cast<IntegerType>(srcElem);
      intType && intType.getWidth() == 32) {
    return StringAttr::get(op.getContext(), "llvm.hivm.MOV.L1.TO.BT.s32")
        .getValue();
  }
  return failure();
}

static StringRef buildCopyCbufToFbufCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.MOV.L1.TO.FB.v220").getValue();
}

static StringRef buildPstiCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.psti.b8").getValue();
}

static StringRef buildPstsCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.psts.b8").getValue();
}

static StringRef buildPldiCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pldi.b8").getValue();
}

static StringRef buildPldsCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.plds.b8").getValue();
}

static StringRef buildPnotCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pnot.z").getValue();
}

static StringRef buildPselCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.psel").getValue();
}

static StringRef buildPandCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pand.z").getValue();
}

static StringRef buildPorCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.por.z").getValue();
}

static StringRef buildPxorCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pxor.z").getValue();
}

static StringRef buildPpackCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.ppack.z").getValue();
}

static StringRef buildPunpackCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.punpack").getValue();
}

template <typename Op>
static StringRef buildPredicatePairReorderCallee(MLIRContext *context);

template <>
StringRef buildPredicatePairReorderCallee<pto::PdintlvB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pdintlv.b8").getValue();
}

template <>
StringRef buildPredicatePairReorderCallee<pto::PdintlvB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pdintlv.b16").getValue();
}

template <>
StringRef buildPredicatePairReorderCallee<pto::PdintlvB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pdintlv.b32").getValue();
}

template <>
StringRef buildPredicatePairReorderCallee<pto::PintlvB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pintlv.b8").getValue();
}

template <>
StringRef buildPredicatePairReorderCallee<pto::PintlvB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pintlv.b16").getValue();
}

template <>
StringRef buildPredicatePairReorderCallee<pto::PintlvB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pintlv.b32").getValue();
}

static FailureOr<StringRef> buildInterleaveCallee(MLIRContext *context,
                                                  Type resultType,
                                                  StringRef stem) {
  return buildLaneTypedCallee(context, resultType, stem, "");
}

static FailureOr<StringRef> buildUnpackCallee(MLIRContext *context,
                                              Type inputType,
                                              Type resultType,
                                              StringRef stem) {
  std::string input =
      getElementTypeFragment(getElementTypeFromVectorLike(inputType));
  std::string result =
      getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  if (input.empty() || result.empty())
    return failure();
  return StringAttr::get(context,
                         "llvm.hivm." + stem.str() + "." + input + "2" + result)
      .getValue();
}

static FailureOr<StringRef> buildVpackCallee(MLIRContext *context, Type inputType,
                                             Type resultType) {
  std::string input =
      getElementTypeFragment(getElementTypeFromVectorLike(inputType));
  std::string result =
      getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  if (input.empty() || result.empty())
    return failure();

  return StringAttr::get(context, "llvm.hivm.vpack." + input + "2" + result + ".x")
      .getValue();
}

static FailureOr<StringRef> buildVsqzCallee(MLIRContext *context,
                                            Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vsqz", ".x.v300");
}

static FailureOr<StringRef> buildVusqzCallee(MLIRContext *context,
                                             Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vusqz", ".m");
}

static FailureOr<StringRef> buildVmulaCallee(MLIRContext *context,
                                             Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vmula", ".m");
}

static FailureOr<StringRef> buildVmullCallee(MLIRContext *context,
                                             Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vmull", "");
}

template <typename StoreOp>
static StringRef getPredicateStoreCallee(MLIRContext *context);

template <>
StringRef getPredicateStoreCallee<pto::PstiOp>(MLIRContext *context) {
  return buildPstiCallee(context);
}

template <>
StringRef getPredicateStoreCallee<pto::PstsOp>(MLIRContext *context) {
  return buildPstsCallee(context);
}

template <typename LoadOp>
static StringRef getPredicateLoadCallee(MLIRContext *context);

template <>
StringRef getPredicateLoadCallee<pto::PldiOp>(MLIRContext *context) {
  return buildPldiCallee(context);
}

template <>
StringRef getPredicateLoadCallee<pto::PldsOp>(MLIRContext *context) {
  return buildPldsCallee(context);
}

template <typename PredicateMaskOp>
static StringRef getPredicateMaskCallee(MLIRContext *context);

template <>
StringRef getPredicateMaskCallee<pto::PnotOp>(MLIRContext *context) {
  return buildPnotCallee(context);
}

template <>
StringRef getPredicateMaskCallee<pto::PselOp>(MLIRContext *context) {
  return buildPselCallee(context);
}

template <>
StringRef getPredicateMaskCallee<pto::PandOp>(MLIRContext *context) {
  return buildPandCallee(context);
}

template <>
StringRef getPredicateMaskCallee<pto::PorOp>(MLIRContext *context) {
  return buildPorCallee(context);
}

template <>
StringRef getPredicateMaskCallee<pto::PxorOp>(MLIRContext *context) {
  return buildPxorCallee(context);
}

template <typename PackOp>
static StringRef getPredicatePackCallee(MLIRContext *context);

template <>
StringRef getPredicatePackCallee<pto::PpackOp>(MLIRContext *context) {
  return buildPpackCallee(context);
}

template <>
StringRef getPredicatePackCallee<pto::PunpackOp>(MLIRContext *context) {
  return buildPunpackCallee(context);
}

template <typename PltOp>
static StringRef buildPltCallee(MLIRContext *context);

template <>
StringRef buildPltCallee<pto::PltB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.plt.b8.v300").getValue();
}

template <>
StringRef buildPltCallee<pto::PltB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.plt.b16.v300").getValue();
}

template <>
StringRef buildPltCallee<pto::PltB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.plt.b32.v300").getValue();
}

template <typename PsetOp>
static StringRef buildPsetCallee(MLIRContext *context);

template <>
StringRef buildPsetCallee<pto::PsetB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pset.b8").getValue();
}

template <>
StringRef buildPsetCallee<pto::PsetB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pset.b16").getValue();
}

template <>
StringRef buildPsetCallee<pto::PsetB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pset.b32").getValue();
}

template <typename PgeOp>
static StringRef buildPgeCallee(MLIRContext *context);

template <>
StringRef buildPgeCallee<pto::PgeB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pge.b8").getValue();
}

template <>
StringRef buildPgeCallee<pto::PgeB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pge.b16").getValue();
}

template <>
StringRef buildPgeCallee<pto::PgeB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pge.b32").getValue();
}

static FailureOr<StringRef> buildVldsCallee(MLIRContext *context, Type resultType) {
  std::string vec = getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
    return failure();
  return StringAttr::get(context, "llvm.hivm.vldsx1.v" + std::to_string(*lanes) +
                                      vec)
      .getValue();
}

static FailureOr<StringRef> buildVldsx2Callee(MLIRContext *context,
                                              Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vldsx2", "");
}

static StringRef buildVsldbCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vsldb").getValue();
}

static FailureOr<StringRef> buildVstsCallee(MLIRContext *context, Type valueType) {
  std::string vec = getElementTypeFragment(getElementTypeFromVectorLike(valueType));
  auto lanes = getElementCountFromVectorLike(valueType);
  if (vec.empty() || !lanes)
    return failure();
  return StringAttr::get(context, "llvm.hivm.vstsx1.v" + std::to_string(*lanes) +
                                      vec)
      .getValue();
}

static FailureOr<StringRef> buildVstsx2Callee(MLIRContext *context, Type valueType) {
  return buildLaneTypedCallee(context, valueType, "vstsx2", "");
}

static StringRef buildVsstbCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vsstb").getValue();
}

static StringRef buildVsstbPostCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vsstb.post").getValue();
}

static FailureOr<StringRef> buildVgather2Callee(MLIRContext *context,
                                                Type resultType) {
  std::string vec =
      getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
    return failure();
  return StringAttr::get(context, "llvm.hivm.vgather2.v300.v" +
                                      std::to_string(*lanes) + vec)
      .getValue();
}

static FailureOr<StringRef> buildVgather2BcCallee(MLIRContext *context,
                                                  Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vgather2.bc", "");
}

static FailureOr<StringRef> buildVgatherbCallee(MLIRContext *context,
                                                Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vgatherb.v310", "");
}

static FailureOr<StringRef> buildVscatterCallee(MLIRContext *context,
                                                Type valueType) {
  return buildLaneTypedCallee(context, valueType, "vscatter", ".v300");
}

static FailureOr<StringRef> buildVaxpyCallee(MLIRContext *context,
                                             Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vaxpy", ".m");
}

static FailureOr<StringRef> buildVciCallee(MLIRContext *context, Type resultType) {
  std::string vec =
      getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
    return failure();
  if (vec == "f16" || vec == "f32")
    return StringAttr::get(context, "llvm.hivm.vci.v" + std::to_string(*lanes) +
                                        vec + "." + vec)
        .getValue();
  return StringAttr::get(context,
                         "llvm.hivm.vci.v" + std::to_string(*lanes) + vec)
      .getValue();
}

static FailureOr<StringRef> buildVtrcCallee(MLIRContext *context, Type resultType) {
  std::string vec =
      getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
    return failure();
  return StringAttr::get(context, "llvm.hivm.vtrc." + vec + ".x").getValue();
}

static FailureOr<StringRef> buildVexpdifCallee(MLIRContext *context,
                                               Type inputType,
                                               Type resultType) {
  std::string srcVec =
      getElementTypeFragment(getElementTypeFromVectorLike(inputType));
  auto srcLanes = getElementCountFromVectorLike(inputType);
  std::string dstElem =
      getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  if (srcVec.empty() || dstElem.empty() || !srcLanes)
    return failure();
  return StringAttr::get(context, "llvm.hivm.vexpdif.v" +
                                      std::to_string(*srcLanes) + srcVec +
                                      dstElem)
      .getValue();
}

static FailureOr<StringRef> buildVbitsortCallee(MLIRContext *context,
                                                pto::VbitsortOp op) {
  Type sourceElemType = cast<pto::PtrType>(op.getSource().getType()).getElementType();
  if (sourceElemType.isF16())
    return StringAttr::get(context, "llvm.hivm.VBS32.V300.f16").getValue();
  if (sourceElemType.isF32())
    return StringAttr::get(context, "llvm.hivm.VBS32.V300.f32").getValue();
  return failure();
}

static FailureOr<StringRef> buildVmrgsort4Callee(MLIRContext *context,
                                                 pto::Vmrgsort4Op op) {
  Type elemType =
      cast<pto::PtrType>(op.getDestination().getType()).getElementType();
  if (elemType.isF16())
    return StringAttr::get(context, "llvm.hivm.VMRGSORT.f16.V300").getValue();
  if (elemType.isF32())
    return StringAttr::get(context, "llvm.hivm.VMRGSORT.f32.V300").getValue();
  return failure();
}

static FailureOr<Value> packVmrgsort4SourceAddr(Operation *anchor, Value source0,
                                                Value source1, Value source2,
                                                Value source3, Type elemType) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();
  unsigned addrShift = 0;
  if (elemType.isF16())
    addrShift = 3;
  else if (elemType.isF32())
    addrShift = 3;
  else
    return failure();

  auto packOne = [&](Value source, uint64_t laneShift) -> FailureOr<Value> {
    FailureOr<Value> ubPtr = reinterpretPointerToAddrSpace(anchor, source, 6);
    if (failed(ubPtr))
      return failure();
    Value asInt =
        builder.create<LLVM::PtrToIntOp>(loc, builder.getI64Type(), *ubPtr);
    Value shifted = builder.create<arith::ShRUIOp>(
        loc, asInt, getI64Constant(builder, loc, addrShift));
    Value masked = builder.create<arith::AndIOp>(
        loc, shifted, getI64Constant(builder, loc, 0xFFFFULL));
    if (laneShift == 0)
      return masked;
    return builder
        .create<arith::ShLIOp>(loc, masked,
                               getI64Constant(builder, loc, laneShift))
        .getResult();
  };

  FailureOr<Value> low0 = packOne(source0, 0);
  FailureOr<Value> low1 = packOne(source1, 16);
  FailureOr<Value> low2 = packOne(source2, 32);
  FailureOr<Value> low3 = packOne(source3, 48);
  if (failed(low0) || failed(low1) || failed(low2) || failed(low3))
    return failure();

  Value packed01 = builder.create<arith::OrIOp>(loc, *low0, *low1);
  Value packed23 = builder.create<arith::OrIOp>(loc, *low2, *low3);
  Value packed = builder.create<arith::OrIOp>(loc, packed01, packed23);
  Type ubPtrTy = LLVM::LLVMPointerType::get(anchor->getContext(), 6);
  return builder.create<LLVM::IntToPtrOp>(loc, ubPtrTy, packed).getResult();
}

static FailureOr<VcvtContract> buildVcvtContract(pto::VcvtOp op) {
  Type inputElemType = getElementTypeFromVectorLike(op.getInput().getType());
  Type resultElemType = getElementTypeFromVectorLike(op.getResult().getType());
  if (!inputElemType || !resultElemType)
    return failure();
  auto contract = lookupVcvtContract(classifyVcvtElemType(inputElemType),
                                     classifyVcvtElemType(resultElemType));
  if (!contract)
    return failure();
  return *contract;
}

static bool needsV300CtrlModeForVPTOFunc(func::FuncOp funcOp) {
  if (!pto::isPTOEntryFunction(funcOp) || funcOp.getBlocks().empty())
    return false;

  bool needsCtrlSetup = false;
  funcOp.walk([&](pto::VcvtOp vcvtOp) {
    FailureOr<VcvtContract> contract = buildVcvtContract(vcvtOp);
    if (succeeded(contract) && (*contract).requiresSat) {
      needsCtrlSetup = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return needsCtrlSetup;
}

template <typename LoopOp>
static StringRef buildSetLoopCallee(MLIRContext *context);

template <typename ConfigOp>
static StringRef buildUnaryConfigCallee(MLIRContext *context);

template <typename ConfigOp>
static StringRef buildNullaryConfigCallee(MLIRContext *context);

template <>
StringRef buildSetLoopCallee<pto::SetLoop2StrideOutToUbOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP2.STRIDE.OUTTOUB")
      .getValue();
}

template <>
StringRef buildSetLoopCallee<pto::SetLoop1StrideOutToUbOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP1.STRIDE.OUTTOUB")
      .getValue();
}

template <>
StringRef buildSetLoopCallee<pto::SetLoopSizeOutToUbOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP.SIZE.OUTTOUB")
      .getValue();
}

template <>
StringRef buildSetLoopCallee<pto::SetLoop2StrideUbToOutOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP2.STRIDE.UBTOOUT")
      .getValue();
}

template <>
StringRef buildSetLoopCallee<pto::SetLoop1StrideUbToOutOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP1.STRIDE.UBTOOUT")
      .getValue();
}

template <>
StringRef buildSetLoopCallee<pto::SetLoopSizeUbToOutOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP.SIZE.UBTOOUT")
      .getValue();
}

template <>
StringRef buildSetLoopCallee<pto::SetLoop3ParaOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP3.PARA").getValue();
}

template <>
StringRef buildSetLoopCallee<pto::SetChannelParaOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.CHANNEL.PARA").getValue();
}

template <>
StringRef buildUnaryConfigCallee<pto::SetMovPadValOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.MOV.PAD.VAL").getValue();
}

template <>
StringRef buildUnaryConfigCallee<pto::SetQuantPreOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.QUANT.PRE.v300").getValue();
}

template <>
StringRef buildUnaryConfigCallee<pto::SetReluAlphaOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.RELU.ALPHA").getValue();
}

template <>
StringRef buildUnaryConfigCallee<pto::SetFixClipReluOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.FIX.CLIP.RELU").getValue();
}

template <>
StringRef buildUnaryConfigCallee<pto::SetLoop2StrideOutToL1Op>(
    MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP2.STRIDE.OUTTOL1")
      .getValue();
}

template <>
StringRef buildUnaryConfigCallee<pto::SetLoop1StrideOutToL1Op>(
    MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP1.STRIDE.OUTTOL1")
      .getValue();
}

template <>
StringRef buildUnaryConfigCallee<pto::SetLoopSizeOutToL1Op>(
    MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP.SIZE.OUTTOL1")
      .getValue();
}

template <>
StringRef buildUnaryConfigCallee<pto::SetMte2NzParaOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.MTE2.NZ.PARA").getValue();
}

template <>
StringRef buildUnaryConfigCallee<pto::SetPadValOutToL1Op>(
    MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.PAD.VAL.OUTTOL1")
      .getValue();
}

template <>
StringRef buildUnaryConfigCallee<pto::SetFpcOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.FPC").getValue();
}

template <>
StringRef buildNullaryConfigCallee<pto::SetAtomicS32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.ATOMIC.S32").getValue();
}

template <>
StringRef buildNullaryConfigCallee<pto::SetAtomicS8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.ATOMIC.S8").getValue();
}

static FailureOr<Value> encodeMovPadValue(Location loc, Value value,
                                          ConversionPatternRewriter &rewriter) {
  Type type = value.getType();
  Value payload = value;
  unsigned bitWidth = 0;

  if (auto intType = dyn_cast<IntegerType>(type)) {
    bitWidth = intType.getWidth();
  } else if (auto floatType = dyn_cast<FloatType>(type)) {
    bitWidth = floatType.getWidth();
    auto intType = rewriter.getIntegerType(bitWidth);
    payload = rewriter.create<arith::BitcastOp>(loc, intType, value);
  } else {
    return failure();
  }

  if (bitWidth != 8 && bitWidth != 16 && bitWidth != 32)
    return failure();

  return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), payload)
      .getResult();
}

template <typename SyncOp>
static StringRef buildSyncCallee(MLIRContext *context);

template <>
StringRef buildSyncCallee<pto::SetFlagOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.FLAG.IMM").getValue();
}

template <>
StringRef buildSyncCallee<pto::WaitFlagOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.WAIT.FLAG.IMM").getValue();
}

template <>
StringRef buildSyncCallee<pto::SetFlagDynOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.FLAG.REG").getValue();
}

template <>
StringRef buildSyncCallee<pto::WaitFlagDynOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.WAIT.FLAG.REG").getValue();
}

template <>
StringRef buildSyncCallee<pto::BarrierOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.BARRIER").getValue();
}

template <>
StringRef buildSyncCallee<pto::SyncSetOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.INTRA.BLOCK.mode").getValue();
}

template <>
StringRef buildSyncCallee<pto::SyncWaitOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.WAIT.INTRA.BLOCK.mode").getValue();
}

static StringRef buildMemBarCallee(MemBarKind kind, MLIRContext *context) {
  switch (kind) {
  case MemBarKind::VV_ALL:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vv.all").getValue();
  case MemBarKind::VST_VLD:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vst.vld").getValue();
  case MemBarKind::VLD_VST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vld.vst").getValue();
  case MemBarKind::VST_VST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vst.vst").getValue();
  case MemBarKind::VS_ALL:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vs.all").getValue();
  case MemBarKind::VST_LD:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vst.ld").getValue();
  case MemBarKind::VLD_ST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vld.st").getValue();
  case MemBarKind::VST_ST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.vst.st").getValue();
  case MemBarKind::SV_ALL:
    return StringAttr::get(context, "llvm.hivm.mem.bar.sv.all").getValue();
  case MemBarKind::ST_VLD:
    return StringAttr::get(context, "llvm.hivm.mem.bar.st.vld").getValue();
  case MemBarKind::LD_VST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.ld.vst").getValue();
  case MemBarKind::ST_VST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.st.vst").getValue();
  case MemBarKind::SS_ALL:
    return StringAttr::get(context, "llvm.hivm.mem.bar.ss.all").getValue();
  case MemBarKind::ST_LD:
    return StringAttr::get(context, "llvm.hivm.mem.bar.st.ld").getValue();
  case MemBarKind::LD_ST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.ld.st").getValue();
  case MemBarKind::ST_ST:
    return StringAttr::get(context, "llvm.hivm.mem.bar.st.st").getValue();
  }
  llvm_unreachable("unexpected membar kind");
}

template <>
StringRef buildSyncCallee<pto::GetBufOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.BUFI.mode").getValue();
}

template <>
StringRef buildSyncCallee<pto::RlsBufOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.RLS.BUFI.mode").getValue();
}

template <typename QueryOp>
static StringRef buildRuntimeQueryCallee(MLIRContext *context);

template <>
StringRef buildRuntimeQueryCallee<pto::GetBlockIdxOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.BLOCK.IDX").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetSubBlockIdxOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.SUBBLOCKID").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetBlockNumOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.BLOCK.NUM").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetSubBlockNumOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.SUBBLOCKDIM").getValue();
}

static LogicalResult
materializeDecls(ModuleOp module, ArrayRef<PlannedDecl> plannedDecls,
                 llvm::raw_ostream &diagOS) {
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(&module.getBodyRegion().front());
  for (const PlannedDecl &decl : plannedDecls) {
    if (func::FuncOp existing = module.lookupSymbol<func::FuncOp>(decl.name)) {
      if (existing.getFunctionType() != decl.type) {
        diagOS << "VPTO LLVM emission failed: conflicting declaration for "
               << decl.name << "\n";
        return failure();
      }
      continue;
    }
    auto func =
        builder.create<func::FuncOp>(module.getLoc(), decl.name, decl.type);
    func.setPrivate();
  }
  return success();
}

template <typename UnaryOp>
class LowerUnaryMaskedOpPattern final : public OpConversionPattern<UnaryOp> {
public:
  explicit LowerUnaryMaskedOpPattern(TypeConverter &typeConverter,
                                     MLIRContext *context,
                                     LoweringState &state)
      : OpConversionPattern<UnaryOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(UnaryOp op, typename UnaryOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef stem = getUnaryMaskedStem<UnaryOp>();
    FailureOr<StringRef> calleeName =
        buildLaneTypedCallee(op.getContext(), op.getResult().getType(), stem, ".x");
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported unary VPTO signature");

    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert unary result type");

    Value input = adaptor.getOperands()[0];
    Value mask = adaptor.getOperands()[1];
    Type expectedMaskType =
        this->getTypeConverter()->convertType(op->getOperand(1).getType());
    if (!input || !mask || input.getType() != resultType ||
        mask.getType() != expectedMaskType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted unary VPTO operand types");
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{resultType},
                                              ValueRange{input, mask});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVsqzOpPattern final : public OpConversionPattern<pto::VsqzOp> {
public:
  explicit LowerVsqzOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::VsqzOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VsqzOp op, pto::VsqzOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName =
        buildVsqzCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vsqz VPTO signature");

    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType)
      return rewriter.notifyMatchFailure(op, "failed to convert vsqz types");

    Value input = adaptor.getInput();
    Value mask = adaptor.getMask();
    if (!input || !mask || input.getType() != resultType ||
        mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vsqz operand types");
    }

    Value storeHint =
        getI32Constant(rewriter, op.getLoc(), determineVsqzStoreHint(op));
    auto funcType = rewriter.getFunctionType(
        TypeRange{resultType, maskType, storeHint.getType()}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{input, mask, storeHint});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVusqzOpPattern final : public OpConversionPattern<pto::VusqzOp> {
public:
  explicit LowerVusqzOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VusqzOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VusqzOp op, pto::VusqzOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName =
        buildVusqzCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vusqz VPTO signature");

    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType)
      return rewriter.notifyMatchFailure(op, "failed to convert vusqz types");

    Value src = adaptor.getSrc();
    Value mask = adaptor.getMask();
    if (!src || !mask || src.getType() != resultType || mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vusqz operand types");
    }

    auto funcType =
        rewriter.getFunctionType(TypeRange{resultType, maskType}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType}, ValueRange{src, mask});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVmulaOpPattern final : public OpConversionPattern<pto::VmulaOp> {
public:
  explicit LowerVmulaOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                               LoweringState &state)
      : OpConversionPattern<pto::VmulaOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VmulaOp op, pto::VmulaOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName =
        buildVmulaCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vmula VPTO signature");

    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType)
      return rewriter.notifyMatchFailure(op, "failed to convert vmula types");

    Value acc = adaptor.getAcc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value mask = adaptor.getMask();
    if (!acc || !lhs || !rhs || !mask || acc.getType() != resultType ||
        lhs.getType() != resultType || rhs.getType() != resultType ||
        mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vmula operand types");
    }

    auto funcType = rewriter.getFunctionType(
        TypeRange{resultType, resultType, resultType, maskType},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{acc, lhs, rhs, mask});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVmullOpPattern final : public OpConversionPattern<pto::VmullOp> {
public:
  explicit LowerVmullOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                               LoweringState &state)
      : OpConversionPattern<pto::VmullOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VmullOp op, pto::VmullOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName =
        buildVmullCallee(op.getContext(), op.getLow().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vmull VPTO signature");

    Type inputType = this->getTypeConverter()->convertType(op.getLhs().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    SmallVector<Type> resultTypes;
    if (!inputType || !maskType ||
        failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert vmull types");
    }
    if (resultTypes.size() != 2 || resultTypes[0] != resultTypes[1])
      return rewriter.notifyMatchFailure(op, "unexpected converted vmull results");

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value mask = adaptor.getMask();
    if (!lhs || !rhs || !mask || lhs.getType() != inputType ||
        rhs.getType() != inputType || mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vmull operand types");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{inputType, inputType, maskType},
                                             resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, resultTypes,
                                              ValueRange{lhs, rhs, mask});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename BinaryOp>
class LowerBinaryMaskedOpPattern final : public OpConversionPattern<BinaryOp> {
public:
  explicit LowerBinaryMaskedOpPattern(TypeConverter &typeConverter,
                                      MLIRContext *context,
                                      LoweringState &state)
      : OpConversionPattern<BinaryOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(BinaryOp op, typename BinaryOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef stem = getBinaryMaskedStem<BinaryOp>();
    FailureOr<StringRef> calleeName =
        buildLaneTypedCallee(op.getContext(), op.getResult().getType(), stem, ".x");
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported binary VPTO signature");

    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert binary result type");

    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];
    Value mask = adaptor.getOperands()[2];
    Type expectedMaskType =
        this->getTypeConverter()->convertType(op->getOperand(2).getType());
    if (!lhs || !rhs || !mask || lhs.getType() != resultType ||
        rhs.getType() != resultType || mask.getType() != expectedMaskType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted binary VPTO operand types");
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{resultType},
                                              ValueRange{lhs, rhs, mask});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename CarryOp>
class LowerCarryBinaryOpPattern final : public OpConversionPattern<CarryOp> {
public:
  explicit LowerCarryBinaryOpPattern(TypeConverter &typeConverter,
                                     MLIRContext *context, LoweringState &state)
      : OpConversionPattern<CarryOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(CarryOp op, typename CarryOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef stem = getCarryBinaryStem<CarryOp>();
    FailureOr<StringRef> calleeName =
        buildCarryBinaryCallee(op.getContext(), op.getResult().getType(), stem);
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported carry VPTO signature");

    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    Type carryType =
        this->getTypeConverter()->convertType(op->getResult(1).getType());
    if (!resultType || !carryType)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert carry result types");

    SmallVector<Value> callArgs;
    callArgs.append(adaptor.getOperands().begin(), adaptor.getOperands().end());
    const size_t expectedArgCount = hasCarryInput<CarryOp>() ? 4 : 3;
    if (callArgs.size() != expectedArgCount || callArgs[0].getType() != resultType ||
        callArgs[1].getType() != resultType || callArgs.back().getType() != carryType)
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted carry operand types");
    if constexpr (hasCarryInput<CarryOp>()) {
      if (callArgs[2].getType() != carryType)
        return rewriter.notifyMatchFailure(
            op, "unexpected converted carry input operand type");
    }

    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType, carryType}, callArgs);
    state.plannedDecls.push_back(
        PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename CopyOp>
class LowerCopyOpPattern final : public OpConversionPattern<CopyOp> {
public:
  explicit LowerCopyOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<CopyOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(CopyOp op, typename CopyOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName = failure();
    if constexpr (std::is_same_v<CopyOp, pto::CopyGmToUbufOp>)
      calleeName = buildCopyGmToUbCallee(op.getContext(), op.getSource().getType());
    else
      calleeName = buildCopyUbToGmCallee(op.getContext());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported copy VPTO signature");

    auto llvmSourceType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getOperands()[0].getType());
    auto llvmDestType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getOperands()[1].getType());
    if (!llvmSourceType || !llvmDestType)
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer copy operands");

    FailureOr<Value> config0 = failure();
    FailureOr<Value> config1 = failure();
    if constexpr (std::is_same_v<CopyOp, pto::CopyGmToUbufOp>) {
      config0 = packCopyGmToUbConfig0(op, adaptor.getOperands());
      config1 = packCopyGmToUbConfig1(op, adaptor.getOperands());
    } else {
      config0 = packCopyUbToGmConfig0(op, adaptor.getOperands());
      config1 = packCopyUbToGmConfig1(op, adaptor.getOperands());
    }
    if (failed(config0) || failed(config1))
      return rewriter.notifyMatchFailure(op, "failed to materialize copy config");

    SmallVector<Value> args{adaptor.getOperands()[1], adaptor.getOperands()[0],
                            *config0, *config1};
    auto funcType = rewriter.getFunctionType(
        TypeRange{llvmDestType, llvmSourceType, rewriter.getI64Type(),
                  rewriter.getI64Type()},
        TypeRange{});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    (void)call;
    return success();
  }

private:
  LoweringState &state;
};

class LowerCopyUbufToUbufOpPattern final
    : public OpConversionPattern<pto::CopyUbufToUbufOp> {
public:
  explicit LowerCopyUbufToUbufOpPattern(TypeConverter &typeConverter,
                                        MLIRContext *context,
                                        LoweringState &state)
      : OpConversionPattern<pto::CopyUbufToUbufOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::CopyUbufToUbufOp op,
                  pto::CopyUbufToUbufOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto llvmSourceType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getOperands()[0].getType());
    auto llvmDestType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getOperands()[1].getType());
    if (!llvmSourceType || !llvmDestType)
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer copy operands");

    FailureOr<Value> config = packCopyUbToUbConfig(op, adaptor.getOperands());
    if (failed(config))
      return rewriter.notifyMatchFailure(op, "failed to materialize copy config");

    StringRef calleeName = buildCopyUbToUbCallee(op.getContext());
    SmallVector<Value> args{adaptor.getOperands()[1], adaptor.getOperands()[0],
                            *config};
    auto funcType = rewriter.getFunctionType(
        TypeRange{llvmDestType, llvmSourceType, rewriter.getI64Type()},
        TypeRange{});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              TypeRange{}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    (void)call;
    return success();
  }

private:
  LoweringState &state;
};

class LowerCopyCbufToUbufOpPattern final
    : public OpConversionPattern<pto::CopyCbufToUbufOp> {
public:
  explicit LowerCopyCbufToUbufOpPattern(TypeConverter &typeConverter,
                                        MLIRContext *context,
                                        LoweringState &state)
      : OpConversionPattern<pto::CopyCbufToUbufOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::CopyCbufToUbufOp op,
                  pto::CopyCbufToUbufOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    if (!sourceRaw || !destinationRaw)
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) ||
        !isa<LLVM::LLVMPointerType>(destinationRaw.getType()))
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");

    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    constexpr unsigned ubufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::VEC);
    FailureOr<Value> source =
        reinterpretPointerToAddrSpace(op, sourceRaw, cbufAddressSpace);
    FailureOr<Value> destination =
        reinterpretPointerToAddrSpace(op, destinationRaw, ubufAddressSpace);
    if (failed(source) || failed(destination))
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/ubuf pointer spaces");

    FailureOr<Value> config = packCopyCbufToUbConfig(op, adaptor.getOperands());
    if (failed(config))
      return rewriter.notifyMatchFailure(op, "failed to materialize copy config");

    StringRef calleeName = buildCopyCbufToUbCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(
        TypeRange{destination->getType(), source->getType(),
                  rewriter.getI64Type()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{*destination, *source, *config});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerCopyUbufToCbufOpPattern final
    : public OpConversionPattern<pto::CopyUbufToCbufOp> {
public:
  explicit LowerCopyUbufToCbufOpPattern(TypeConverter &typeConverter,
                                        MLIRContext *context,
                                        LoweringState &state)
      : OpConversionPattern<pto::CopyUbufToCbufOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::CopyUbufToCbufOp op,
                  pto::CopyUbufToCbufOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    if (!sourceRaw || !destinationRaw)
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) ||
        !isa<LLVM::LLVMPointerType>(destinationRaw.getType()))
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");

    constexpr unsigned ubufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::VEC);
    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    FailureOr<Value> source =
        reinterpretPointerToAddrSpace(op, sourceRaw, ubufAddressSpace);
    FailureOr<Value> destination =
        reinterpretPointerToAddrSpace(op, destinationRaw, cbufAddressSpace);
    if (failed(source) || failed(destination))
      return rewriter.notifyMatchFailure(op, "failed to map ubuf/cbuf pointer spaces");

    FailureOr<Value> config = packCopyUbToCbufConfig(op, adaptor.getOperands());
    if (failed(config))
      return rewriter.notifyMatchFailure(op, "failed to materialize copy config");

    StringRef calleeName = buildCopyUbToCbufCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(
        TypeRange{destination->getType(), source->getType(),
                  rewriter.getI64Type()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{*destination, *source, *config});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

static LogicalResult lowerMadRawOp(pto::MadRawOpInterface op,
                                   ValueRange convertedOperands,
                                   ConversionPatternRewriter &rewriter,
                                   LoweringState &state) {
  Value lhsRaw = convertedOperands[0];
  Value rhsRaw = convertedOperands[1];
  Value dstRaw = convertedOperands[2];
  Value biasRaw = op.hasBiasOperand() ? convertedOperands[3] : Value();
  Value xt = convertedOperands[op.hasBiasOperand() ? 4 : 3];
  if (!lhsRaw || !rhsRaw || !dstRaw || !xt ||
      (op.hasBiasOperand() && !biasRaw))
    return rewriter.notifyMatchFailure(op, "expected converted mad raw operands");

  if (!isa<LLVM::LLVMPointerType>(lhsRaw.getType()) ||
      !isa<LLVM::LLVMPointerType>(rhsRaw.getType()) ||
      !isa<LLVM::LLVMPointerType>(dstRaw.getType()) ||
      (biasRaw && !isa<LLVM::LLVMPointerType>(biasRaw.getType()))) {
    return rewriter.notifyMatchFailure(
        op, "expected LLVM pointer lhs/rhs/dst/bias operands");
  }

  Type i64Ty = rewriter.getI64Type();
  constexpr unsigned caAddressSpace =
      static_cast<unsigned>(pto::AddressSpace::LEFT);
  constexpr unsigned cbAddressSpace =
      static_cast<unsigned>(pto::AddressSpace::RIGHT);
  constexpr unsigned ccAddressSpace =
      static_cast<unsigned>(pto::AddressSpace::ACC);
  constexpr unsigned btAddressSpace =
      static_cast<unsigned>(pto::AddressSpace::BIAS);
  FailureOr<Value> lhs =
      reinterpretPointerToAddrSpace(op, lhsRaw, caAddressSpace);
  FailureOr<Value> rhs =
      reinterpretPointerToAddrSpace(op, rhsRaw, cbAddressSpace);
  FailureOr<Value> dst =
      reinterpretPointerToAddrSpace(op, dstRaw, ccAddressSpace);
  FailureOr<Value> bias;
  if (biasRaw)
    bias = reinterpretPointerToAddrSpace(op, biasRaw, btAddressSpace);
  if (failed(lhs) || failed(rhs) || failed(dst) ||
      (biasRaw && failed(bias))) {
    return rewriter.notifyMatchFailure(op, "failed to map cube pointer spaces");
  }

  FailureOr<StringRef> calleeName =
      op.isMadMxFamily() ? buildMxMadCallee(op.getContext(), op)
                         : buildOrdinaryMadCallee(op.getContext(), op);
  if (failed(calleeName))
    return rewriter.notifyMatchFailure(
        op, "unsupported mad element types for raw dispatch");

  Value callDst = *dst;
  if (biasRaw)
    callDst = buildMadBiasDestination(op, rewriter, *dst, *bias);
  auto funcType = rewriter.getFunctionType(
      TypeRange{dst->getType(), lhs->getType(), rhs->getType(), i64Ty},
      TypeRange{});
  auto call = rewriter.create<func::CallOp>(
      op->getLoc(), *calleeName, TypeRange{},
      ValueRange{callDst, *lhs, *rhs, xt});
  state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
  rewriter.replaceOp(op, call.getResults());
  return success();
}

template <typename RawOp>
class LowerMadRawPattern final : public OpConversionPattern<RawOp> {
public:
  explicit LowerMadRawPattern(TypeConverter &typeConverter,
                              MLIRContext *context, LoweringState &state)
      : OpConversionPattern<RawOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(RawOp op, typename RawOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto raw = dyn_cast<pto::MadRawOpInterface>(op.getOperation());
    if (!raw)
      return failure();
    return lowerMadRawOp(raw, adaptor.getOperands(), rewriter, state);
  }

private:
  LoweringState &state;
};

class LowerCopyGmToCbufOpPattern final
    : public OpConversionPattern<pto::CopyGmToCbufOp> {
public:
  explicit LowerCopyGmToCbufOpPattern(TypeConverter &typeConverter,
                                             MLIRContext *context,
                                             LoweringState &state)
      : OpConversionPattern<pto::CopyGmToCbufOp>(typeConverter, context),
        state(state) {}

  LogicalResult matchAndRewrite(
      pto::CopyGmToCbufOp op,
      pto::CopyGmToCbufOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    Value nBurst = adaptor.getNBurst();
    Value lenBurst = adaptor.getLenBurst();
    Value srcStride = adaptor.getSrcStride();
    Value dstStride = adaptor.getDstStride();
    if (!sourceRaw || !destinationRaw || !nBurst || !lenBurst || !srcStride ||
        !dstStride) {
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    }

    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) ||
        !isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    Type i64Ty = rewriter.getI64Type();
    if (nBurst.getType() != i64Ty || lenBurst.getType() != i64Ty ||
        srcStride.getType() != i64Ty || dstStride.getType() != i64Ty) {
      return rewriter.notifyMatchFailure(op, "expected i64 config operands");
    }

    constexpr unsigned gmAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::GM);
    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    FailureOr<Value> source = reinterpretPointerToAddrSpace(op, sourceRaw, gmAddressSpace);
    FailureOr<Value> destination =
        reinterpretPointerToAddrSpace(op, destinationRaw, cbufAddressSpace);
    if (failed(source) || failed(destination))
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/gm pointer spaces");

    FailureOr<StringRef> calleeName =
        buildCopyGmToCbufCallee(op.getContext(), op.getSource().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op,
                                         "unsupported copy_gm_to_cbuf element type");
    FailureOr<Value> config0 =
        packCopyGmToCbufConfig0(op, nBurst, lenBurst);
    FailureOr<Value> config1 =
        packCopyGmToCbufConfig1(op, srcStride, dstStride);
    if (failed(config0) || failed(config1))
      return rewriter.notifyMatchFailure(op,
                                         "failed to pack copy_gm_to_cbuf config");

    auto funcType = rewriter.getFunctionType(
        TypeRange{destination->getType(), source->getType(), i64Ty, i64Ty},
        TypeRange{});
    rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{},
        ValueRange{*destination, *source, *config0, *config1});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename CopyOp>
class LowerCopyGmToCbufMultiOpPattern final
    : public OpConversionPattern<CopyOp> {
public:
  explicit LowerCopyGmToCbufMultiOpPattern(TypeConverter &typeConverter,
                                           MLIRContext *context,
                                           LoweringState &state)
      : OpConversionPattern<CopyOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(CopyOp op, typename CopyOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    if (!sourceRaw || !destinationRaw)
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) ||
        !isa<LLVM::LLVMPointerType>(destinationRaw.getType()))
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");

    constexpr unsigned gmAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::GM);
    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    FailureOr<Value> source =
        reinterpretPointerToAddrSpace(op, sourceRaw, gmAddressSpace);
    FailureOr<Value> destination =
        reinterpretPointerToAddrSpace(op, destinationRaw, cbufAddressSpace);
    if (failed(source) || failed(destination))
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/gm pointer spaces");

    FailureOr<Value> config0 = packCopyGmToCbufMultiConfig0(
        op, adaptor.getSid(), adaptor.getLoop1SrcStride(),
        adaptor.getL2CacheCtrl(), adaptor.getNValue());
    FailureOr<Value> config1 =
        packCopyGmToCbufMultiConfig1(op, adaptor.getDValue(),
                                     adaptor.getLoop4SrcStride(),
                                     adaptor.getSmallc0En());
    if (failed(config0) || failed(config1))
      return rewriter.notifyMatchFailure(op, "failed to pack multi copy config");

    FailureOr<StringRef> calleeName = [&] (MLIRContext *ctx, Type sourceType)
        -> FailureOr<StringRef> {
      if constexpr (std::is_same_v<CopyOp, pto::CopyGmToCbufMultiNd2NzOp>)
        return buildCopyGmToCbufMultiNd2NzCallee(ctx, op.getSource().getType());
      return buildCopyGmToCbufMultiDn2NzCallee(ctx, sourceType);
    }(op.getContext(), op.getSource().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(
          op, "unsupported copy_gm_to_cbuf_multi element type");

    Type i64Ty = rewriter.getI64Type();
    auto funcType = rewriter.getFunctionType(
        TypeRange{destination->getType(), source->getType(), i64Ty, i64Ty},
        TypeRange{});
    rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{},
        ValueRange{*destination, *source, *config0, *config1});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerCopyCbufToBtOpPattern final
    : public OpConversionPattern<pto::CopyCbufToBtOp> {
public:
  explicit LowerCopyCbufToBtOpPattern(TypeConverter &typeConverter,
                                      MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::CopyCbufToBtOp>(typeConverter, context),
        state(state) {}

  LogicalResult matchAndRewrite(pto::CopyCbufToBtOp op,
                                pto::CopyCbufToBtOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    if (!sourceRaw || !destinationRaw)
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) ||
        !isa<LLVM::LLVMPointerType>(destinationRaw.getType()))
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");

    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    constexpr unsigned btAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::BIAS);
    FailureOr<Value> source =
        reinterpretPointerToAddrSpace(op, sourceRaw, cbufAddressSpace);
    FailureOr<Value> destinationPtr =
        reinterpretPointerToAddrSpace(op, destinationRaw, btAddressSpace);
    if (failed(source) || failed(destinationPtr))
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/bt pointer spaces");

    FailureOr<Value> config = packCopyCbufToBtConfig(
        op, adaptor.getConvControl(), adaptor.getNBurst(), adaptor.getLenBurst(),
        adaptor.getSourceGap(), adaptor.getDstGap());
    if (failed(config))
      return rewriter.notifyMatchFailure(op, "failed to pack copy_cbuf_to_bt config");

    Type i64Ty = rewriter.getI64Type();
    Value destination =
        rewriter.create<LLVM::PtrToIntOp>(op.getLoc(), i64Ty, *destinationPtr);
    FailureOr<StringRef> calleeName = buildCopyCbufToBtCallee(op);
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported copy_cbuf_to_bt source element type");
    auto funcType = rewriter.getFunctionType(
        TypeRange{i64Ty, source->getType(), i64Ty}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{},
                                  ValueRange{destination, *source, *config});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerCopyCbufToFbufOpPattern final
    : public OpConversionPattern<pto::CopyCbufToFbufOp> {
public:
  explicit LowerCopyCbufToFbufOpPattern(TypeConverter &typeConverter,
                                        MLIRContext *context,
                                        LoweringState &state)
      : OpConversionPattern<pto::CopyCbufToFbufOp>(typeConverter, context),
        state(state) {}

  LogicalResult matchAndRewrite(pto::CopyCbufToFbufOp op,
                                pto::CopyCbufToFbufOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    if (!sourceRaw || !destinationRaw)
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) ||
        !isa<LLVM::LLVMPointerType>(destinationRaw.getType()))
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");

    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    constexpr unsigned fbufAddressSpace = 7;
    FailureOr<Value> source =
        reinterpretPointerToAddrSpace(op, sourceRaw, cbufAddressSpace);
    FailureOr<Value> destination =
        reinterpretPointerToAddrSpace(op, destinationRaw, fbufAddressSpace);
    if (failed(source) || failed(destination))
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/fbuf pointer spaces");

    FailureOr<Value> config = packCopyCbufToFbufConfig(
        op, adaptor.getNBurst(), adaptor.getLenBurst(), adaptor.getSourceGap(),
        adaptor.getDstGap());
    if (failed(config))
      return rewriter.notifyMatchFailure(op, "failed to pack copy_cbuf_to_fbuf config");

    Type i64Ty = rewriter.getI64Type();
    StringRef calleeName = buildCopyCbufToFbufCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(
        TypeRange{destination->getType(), source->getType(), i64Ty}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{*destination, *source, *config});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerLoadCbufToCaOpPattern final
    : public OpConversionPattern<pto::LoadCbufToCaOp> {
public:
  explicit LowerLoadCbufToCaOpPattern(TypeConverter &typeConverter,
                                      MLIRContext *context,
                                      LoweringState &state)
      : OpConversionPattern<pto::LoadCbufToCaOp>(typeConverter, context),
        state(state) {}

  LogicalResult matchAndRewrite(pto::LoadCbufToCaOp op,
                                pto::LoadCbufToCaOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    Value mStart = adaptor.getMStart();
    Value kStart = adaptor.getKStart();
    Value mStep = adaptor.getMStep();
    Value kStep = adaptor.getKStep();
    Value srcStride = adaptor.getSrcStride();
    Value dstStride = adaptor.getDstStride();
    if (!sourceRaw || !destinationRaw || !mStart || !kStart || !mStep ||
        !kStep || !srcStride || !dstStride)
      return rewriter.notifyMatchFailure(op, "expected converted operands");

    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) ||
        !isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    Type i64Ty = rewriter.getI64Type();

    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    constexpr unsigned caAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::LEFT);
    FailureOr<Value> source = reinterpretPointerToAddrSpace(op, sourceRaw, cbufAddressSpace);
    FailureOr<Value> destination =
        reinterpretPointerToAddrSpace(op, destinationRaw, caAddressSpace);
    if (failed(source) || failed(destination))
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/ca pointer spaces");

    FailureOr<Value> config0 =
        packLoadCbufToCaConfig0(op, mStart, kStart, mStep, kStep);
    FailureOr<Value> config1 =
        packLoadCbufToCaConfig1(op, srcStride, dstStride);
    if (failed(config0) || failed(config1))
      return rewriter.notifyMatchFailure(op, "failed to pack load_cbuf_to_ca config");
    Value transpose =
        getI64Constant(rewriter, op.getLoc(), op.getTranspose() ? 1 : 0);

    FailureOr<StringRef> calleeName =
        buildLoadCbufToCaCallee(op.getContext(), op.getSource().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op,
                                         "unsupported load_cbuf_to_ca element type");
    auto funcType = rewriter.getFunctionType(
        TypeRange{destination->getType(), source->getType(), i64Ty, i64Ty,
                  i64Ty},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{},
                                  ValueRange{*destination, *source, *config0,
                                             *config1, transpose});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename LoadOp>
class LowerLoadCbufToS4OpPattern final : public OpConversionPattern<LoadOp> {
public:
  explicit LowerLoadCbufToS4OpPattern(TypeConverter &typeConverter,
                                      MLIRContext *context,
                                      LoweringState &state)
      : OpConversionPattern<LoadOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(LoadOp op, typename LoadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    if (!sourceRaw || !destinationRaw)
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) ||
        !isa<LLVM::LLVMPointerType>(destinationRaw.getType()))
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");

    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    constexpr unsigned targetAddressSpace =
        std::is_same_v<LoadOp, pto::LoadCbufToCaS4Op>
            ? static_cast<unsigned>(pto::AddressSpace::LEFT)
            : static_cast<unsigned>(pto::AddressSpace::RIGHT);
    FailureOr<Value> source =
        reinterpretPointerToAddrSpace(op, sourceRaw, cbufAddressSpace);
    FailureOr<Value> destination =
        reinterpretPointerToAddrSpace(op, destinationRaw, targetAddressSpace);
    if (failed(source) || failed(destination))
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/cube pointer spaces");

    FailureOr<Value> config0 = packLoadCbufToS4Config0(
        op, adaptor.getMStart(), adaptor.getKStart(), adaptor.getMStep(),
        adaptor.getKStep());
    FailureOr<Value> config1 =
        packLoadCbufToS4Config1(op, adaptor.getSrcStride(),
                                adaptor.getDstStride());
    if (failed(config0) || failed(config1))
      return rewriter.notifyMatchFailure(op, "failed to pack load_cbuf_to_*_s4 config");

    Value transpose =
        castIntegerLikeTo(op, adaptor.getTranspose(), rewriter.getI64Type());
    if (!transpose)
      return rewriter.notifyMatchFailure(op, "failed to cast transpose to i64");

    StringRef calleeName = std::is_same_v<LoadOp, pto::LoadCbufToCaS4Op>
                               ? buildLoadCbufToCaS4Callee(op.getContext())
                               : buildLoadCbufToCbS4Callee(op.getContext());
    Type i64Ty = rewriter.getI64Type();
    auto funcType = rewriter.getFunctionType(
        TypeRange{destination->getType(), source->getType(), i64Ty, i64Ty,
                  i64Ty},
        TypeRange{});
    rewriter.create<func::CallOp>(
        op.getLoc(), calleeName, TypeRange{},
        ValueRange{*destination, *source, *config0, *config1, transpose});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerLoadCbufToCbOpPattern final
    : public OpConversionPattern<pto::LoadCbufToCbOp> {
public:
  explicit LowerLoadCbufToCbOpPattern(TypeConverter &typeConverter,
                                      MLIRContext *context,
                                      LoweringState &state)
      : OpConversionPattern<pto::LoadCbufToCbOp>(typeConverter, context),
        state(state) {}

  LogicalResult matchAndRewrite(pto::LoadCbufToCbOp op,
                                pto::LoadCbufToCbOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    Value mStart = adaptor.getMStart();
    Value kStart = adaptor.getKStart();
    Value mStep = adaptor.getMStep();
    Value kStep = adaptor.getKStep();
    Value srcStride = adaptor.getSrcStride();
    Value dstStride = adaptor.getDstStride();
    if (!sourceRaw || !destinationRaw || !mStart || !kStart || !mStep ||
        !kStep || !srcStride || !dstStride)
      return rewriter.notifyMatchFailure(op, "expected converted operands");

    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) ||
        !isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    Type i64Ty = rewriter.getI64Type();

    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    constexpr unsigned cbAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::RIGHT);
    FailureOr<Value> source = reinterpretPointerToAddrSpace(op, sourceRaw, cbufAddressSpace);
    FailureOr<Value> destination =
        reinterpretPointerToAddrSpace(op, destinationRaw, cbAddressSpace);
    if (failed(source) || failed(destination))
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/cb pointer spaces");

    bool transpose = op.getTranspose();
    FailureOr<Value> config0 =
        packLoadCbufToCbConfig0(op, mStart, kStart, mStep, kStep);
    FailureOr<Value> config1 =
        packLoadCbufToCbConfig1(op, srcStride, dstStride);
    if (failed(config0) || failed(config1))
      return rewriter.notifyMatchFailure(op, "failed to pack load_cbuf_to_cb config");
    Value transposeValue =
        getI64Constant(rewriter, op.getLoc(), transpose ? 1 : 0);

    FailureOr<StringRef> calleeName =
        buildLoadCbufToCbCallee(op.getContext(), op.getSource().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op,
                                         "unsupported load_cbuf_to_cb element type");
    auto funcType = rewriter.getFunctionType(
        TypeRange{destination->getType(), source->getType(), i64Ty, i64Ty,
                  i64Ty},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{},
                                  ValueRange{*destination, *source, *config0,
                                             *config1, transposeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerLoadCbufToCaMxOpPattern final
    : public OpConversionPattern<pto::LoadCbufToCaMxOp> {
public:
  explicit LowerLoadCbufToCaMxOpPattern(TypeConverter &typeConverter,
                                        MLIRContext *context,
                                        LoweringState &state)
      : OpConversionPattern<pto::LoadCbufToCaMxOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::LoadCbufToCaMxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value srcRaw = adaptor.getSource();
    Value dstRaw = adaptor.getDestination();
    if (!srcRaw || !dstRaw || !adaptor.getM() || !adaptor.getK())
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    if (!isa<LLVM::LLVMPointerType>(srcRaw.getType()) ||
        !isa<LLVM::LLVMPointerType>(dstRaw.getType()))
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");

    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    constexpr unsigned caAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::LEFT);
    FailureOr<Value> src = reinterpretPointerToAddrSpace(op, srcRaw, cbufAddressSpace);
    FailureOr<Value> dst = reinterpretPointerToAddrSpace(op, dstRaw, caAddressSpace);
    if (failed(src) || failed(dst))
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/ca pointer spaces");

    Type sourceElemType = cast<pto::PtrType>(op.getSource().getType()).getElementType();
    unsigned elemBitWidth = sourceElemType.getIntOrFloatBitWidth();
    if (elemBitWidth == 0 || (elemBitWidth % 8) != 0)
      return rewriter.notifyMatchFailure(op,
                                         "unsupported load_cbuf_to_ca_mx element type");
    uint64_t elemBytes = elemBitWidth / 8;
    Location loc = op.getLoc();
    auto constant = [&](uint64_t value) -> Value {
      return rewriter.create<arith::ConstantIntOp>(loc, value, 64);
    };
    auto ceilDivConst = [&](Value value, uint64_t divisor) -> Value {
      Value bias = constant(divisor - 1);
      Value sum = rewriter.create<arith::AddIOp>(loc, value, bias);
      return rewriter.create<arith::DivUIOp>(loc, sum, constant(divisor));
    };
    Value zero = constant(0);
    Value mStep = ceilDivConst(adaptor.getM(), 16);
    Value kBytes =
        rewriter.create<arith::MulIOp>(loc, adaptor.getK(), constant(elemBytes));
    Value kStep = ceilDivConst(kBytes, 32);
    Value stride = ceilDivConst(adaptor.getM(), 16);
    FailureOr<Value> config0 =
        packLoadCbufToCaConfig0(op, zero, zero, mStep, kStep);
    FailureOr<Value> config1 =
        packLoadCbufToCaConfig1(op, stride, stride);
    if (failed(config0) || failed(config1))
      return rewriter.notifyMatchFailure(op,
                                         "failed to pack load_cbuf_to_ca_mx config");
    auto i64Ty = rewriter.getI64Type();
    Value dstAddr = rewriter.create<LLVM::PtrToIntOp>(op.getLoc(), i64Ty, *dst);

    StringRef calleeName = buildLoadCbufToCaMxCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(
        TypeRange{i64Ty, src->getType(), i64Ty, i64Ty},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{dstAddr, *src, *config0, *config1});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerLoadCbufToCbMxOpPattern final
    : public OpConversionPattern<pto::LoadCbufToCbMxOp> {
public:
  explicit LowerLoadCbufToCbMxOpPattern(TypeConverter &typeConverter,
                                        MLIRContext *context,
                                        LoweringState &state)
      : OpConversionPattern<pto::LoadCbufToCbMxOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::LoadCbufToCbMxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value srcRaw = adaptor.getSource();
    Value dstRaw = adaptor.getDestination();
    if (!srcRaw || !dstRaw || !adaptor.getXStartPosition() ||
        !adaptor.getYStartPosition() || !adaptor.getXStep() ||
        !adaptor.getYStep() || !adaptor.getSrcStride() ||
        !adaptor.getDstStride())
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    if (!isa<LLVM::LLVMPointerType>(srcRaw.getType()) ||
        !isa<LLVM::LLVMPointerType>(dstRaw.getType()))
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");

    constexpr unsigned cbufAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::MAT);
    constexpr unsigned cbAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::RIGHT);
    FailureOr<Value> src = reinterpretPointerToAddrSpace(op, srcRaw, cbufAddressSpace);
    FailureOr<Value> dst = reinterpretPointerToAddrSpace(op, dstRaw, cbAddressSpace);
    if (failed(src) || failed(dst))
      return rewriter.notifyMatchFailure(op, "failed to map cbuf/cb pointer spaces");

    Type sourceElemType = cast<pto::PtrType>(op.getSource().getType()).getElementType();
    unsigned elemBitWidth = sourceElemType.getIntOrFloatBitWidth();
    if (elemBitWidth == 0 || (elemBitWidth % 8) != 0)
      return rewriter.notifyMatchFailure(op,
                                         "unsupported load_cbuf_to_cb_mx element type");
    FailureOr<Value> config0 =
        packLoadCbufToCaConfig0(op, adaptor.getXStartPosition(),
                                adaptor.getYStartPosition(), adaptor.getXStep(),
                                adaptor.getYStep());
    FailureOr<Value> config1 =
        packLoadCbufToCaConfig1(op, adaptor.getSrcStride(),
                                adaptor.getDstStride());
    if (failed(config0) || failed(config1))
      return rewriter.notifyMatchFailure(op,
                                         "failed to pack load_cbuf_to_cb_mx config");
    auto i64Ty = rewriter.getI64Type();
    Value dstAddr = rewriter.create<LLVM::PtrToIntOp>(op.getLoc(), i64Ty, *dst);

    StringRef calleeName = buildLoadCbufToCbMxCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(
        TypeRange{i64Ty, src->getType(), i64Ty, i64Ty},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{dstAddr, *src, *config0, *config1});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerCopyMatrixCcToGmOpPattern final
    : public OpConversionPattern<pto::CopyMatrixCcToGmOp> {
public:
  explicit LowerCopyMatrixCcToGmOpPattern(TypeConverter &typeConverter,
                                          MLIRContext *context,
                                          LoweringState &state)
      : OpConversionPattern<pto::CopyMatrixCcToGmOp>(typeConverter, context),
        state(state) {}

  LogicalResult matchAndRewrite(
      pto::CopyMatrixCcToGmOp op, pto::CopyMatrixCcToGmOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    Value xm = adaptor.getXm();
    Value xt = adaptor.getXt();
    if (!sourceRaw || !destinationRaw || !xm || !xt)
      return rewriter.notifyMatchFailure(op, "expected converted operands");

    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) ||
        !isa<LLVM::LLVMPointerType>(destinationRaw.getType())) {
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");
    }

    Type i64Ty = rewriter.getI64Type();
    if (xm.getType() != i64Ty || xt.getType() != i64Ty)
      return rewriter.notifyMatchFailure(op, "expected i64 xm/xt operands");

    constexpr unsigned gmAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::GM);
    constexpr unsigned ccAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::ACC);
    FailureOr<Value> source = reinterpretPointerToAddrSpace(op, sourceRaw, ccAddressSpace);
    FailureOr<Value> destination =
        reinterpretPointerToAddrSpace(op, destinationRaw, gmAddressSpace);
    if (failed(source) || failed(destination))
      return rewriter.notifyMatchFailure(op, "failed to map cc/gm pointer spaces");

    StringRef calleeName = buildCopyMatrixCcToGmCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(
        TypeRange{destination->getType(), source->getType(), i64Ty, i64Ty},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{*destination, *source, xm, xt});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename CopyOp>
class LowerCopyMatrixCcToBufOpPattern final
    : public OpConversionPattern<CopyOp> {
public:
  explicit LowerCopyMatrixCcToBufOpPattern(TypeConverter &typeConverter,
                                           MLIRContext *context,
                                           LoweringState &state)
      : OpConversionPattern<CopyOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(CopyOp op, typename CopyOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value sourceRaw = adaptor.getSource();
    Value destinationRaw = adaptor.getDestination();
    if (!sourceRaw || !destinationRaw)
      return rewriter.notifyMatchFailure(op, "expected converted operands");
    if (!isa<LLVM::LLVMPointerType>(sourceRaw.getType()) ||
        !isa<LLVM::LLVMPointerType>(destinationRaw.getType()))
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer src/dst");

    constexpr unsigned ccAddressSpace =
        static_cast<unsigned>(pto::AddressSpace::ACC);
    constexpr unsigned targetAddressSpace =
        std::is_same_v<CopyOp, pto::CopyMatrixCcToCbufOp>
            ? static_cast<unsigned>(pto::AddressSpace::MAT)
            : static_cast<unsigned>(pto::AddressSpace::VEC);
    FailureOr<Value> source =
        reinterpretPointerToAddrSpace(op, sourceRaw, ccAddressSpace);
    FailureOr<Value> destination =
        reinterpretPointerToAddrSpace(op, destinationRaw, targetAddressSpace);
    if (failed(source) || failed(destination))
      return rewriter.notifyMatchFailure(op, "failed to map cc->buf pointer spaces");

    Type i64Ty = rewriter.getI64Type();
    Value config0 = castIntegerLikeTo(op, adaptor.getConfig0(), i64Ty);
    Value config1 = castIntegerLikeTo(op, adaptor.getConfig1(), i64Ty);
    if (!config0 || !config1)
      return rewriter.notifyMatchFailure(op, "failed to cast config operands to i64");

    FailureOr<StringRef> calleeName =
        std::is_same_v<CopyOp, pto::CopyMatrixCcToCbufOp>
            ? FailureOr<StringRef>(buildCopyMatrixCcToCbufCallee(op.getContext()))
            : buildCopyMatrixCcToUbCallee(op.getContext(),
                                          op.getDestination().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(
          op, "unsupported copy_matrix_cc_to_{cbuf,ub} element type");
    auto funcType = rewriter.getFunctionType(
        TypeRange{destination->getType(), source->getType(), i64Ty, i64Ty},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{},
                                  ValueRange{*destination, *source, config0,
                                             config1});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename VecScalarOp>
class LowerVecScalarMaskedOpPattern final
    : public OpConversionPattern<VecScalarOp> {
public:
  explicit LowerVecScalarMaskedOpPattern(TypeConverter &typeConverter,
                                         MLIRContext *context,
                                         LoweringState &state)
      : OpConversionPattern<VecScalarOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(VecScalarOp op, typename VecScalarOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef stem = getVecScalarMaskedStem<VecScalarOp>();
    FailureOr<StringRef> calleeName =
        buildLaneTypedCallee(op.getContext(), op.getResult().getType(), stem, ".x");
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op,
                                         "unsupported vec-scalar VPTO signature");

    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert vec-scalar result type");

    Value input = adaptor.getOperands()[0];
    Value scalar = adaptor.getOperands()[1];
    Value mask = adaptor.getOperands()[2];
    Type expectedMaskType =
        this->getTypeConverter()->convertType(op->getOperand(2).getType());
    if (!input || !scalar || !mask || input.getType() != resultType ||
        mask.getType() != expectedMaskType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted vec-scalar VPTO operand types");
    }

    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{input, scalar, mask});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename ReductionOp>
class LowerReductionUnaryOpPattern final
    : public OpConversionPattern<ReductionOp> {
public:
  explicit LowerReductionUnaryOpPattern(TypeConverter &typeConverter,
                                        MLIRContext *context,
                                        LoweringState &state)
      : OpConversionPattern<ReductionOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(ReductionOp op, typename ReductionOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef stem = getReductionUnaryStem<ReductionOp>();
    FailureOr<StringRef> calleeName =
        buildLaneTypedCallee(op.getContext(), op.getResult().getType(), stem, ".x");
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op,
                                         "unsupported reduction VPTO signature");

    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert reduction result type");
    }

    Value input = adaptor.getInput();
    Value mask = adaptor.getMask();
    if (!input || !mask || input.getType() != resultType ||
        mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted reduction operand types");
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{resultType},
                                              ValueRange{input, mask});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename ReductionOp>
class LowerWideningReductionUnaryOpPattern final
    : public OpConversionPattern<ReductionOp> {
public:
  explicit LowerWideningReductionUnaryOpPattern(TypeConverter &typeConverter,
                                                MLIRContext *context,
                                                LoweringState &state)
      : OpConversionPattern<ReductionOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(ReductionOp op, typename ReductionOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName = buildLaneTypedCalleeFromInput(
        op.getContext(), op.getInput().getType(),
        getReductionUnaryStem<ReductionOp>(), ".x");
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op,
                                         "unsupported widening reduction VPTO signature");

    Type inputType =
        this->getTypeConverter()->convertType(op.getInput().getType());
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!inputType || !resultType || !maskType)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert widening reduction types");

    Value input = adaptor.getInput();
    Value mask = adaptor.getMask();
    if (!input || !mask || input.getType() != inputType ||
        mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted widening reduction operand types");
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{resultType},
                                              ValueRange{input, mask});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVselOpPattern final : public OpConversionPattern<pto::VselOp> {
public:
  explicit LowerVselOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::VselOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VselOp op, pto::VselOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName =
        buildVselCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vsel VPTO signature");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType)
      return rewriter.notifyMatchFailure(op, "failed to convert vsel result type");

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value mask = adaptor.getMask();
    if (!src0 || !src1 || !mask || src0.getType() != resultType ||
        src1.getType() != resultType || mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vsel operand types");
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{resultType},
                                              ValueRange{src0, src1, mask});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVdupOpPattern final : public OpConversionPattern<pto::VdupOp> {
public:
  explicit LowerVdupOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::VdupOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VdupOp op, pto::VdupOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName = buildVdupCallee(op.getContext(), op);
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vdup VPTO signature");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType)
      return rewriter.notifyMatchFailure(op, "failed to convert vdup result type");

    Value mask = adaptor.getMask();
    if (!mask || mask.getType() != maskType)
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vdup mask type");

    SmallVector<Value> callArgs;
    bool vectorInput = isa<VectorType, pto::VRegType>(op.getInput().getType());
    if (vectorInput) {
      Value input = adaptor.getInput();
      if (!input || input.getType() != resultType) {
        return rewriter.notifyMatchFailure(
            op, "vector-input vdup requires matching result type");
      }
      callArgs.push_back(input);
    } else {
      Type scalarType = getElementTypeFromVectorLike(op.getResult().getType());
      if (!scalarType ||
          (op.getInput().getType() != scalarType &&
           !isCompatibleScalarForSemanticType(scalarType,
                                              op.getInput().getType()))) {
        return rewriter.notifyMatchFailure(op,
                                           "unexpected scalar-input vdup type");
      }
      FailureOr<Value> normalizedScalar =
          normalizeVdupScalarOperand(rewriter, op.getLoc(), adaptor.getInput(),
                                     op.getResult().getType());
      if (failed(normalizedScalar))
        return rewriter.notifyMatchFailure(op,
                                           "failed to normalize scalar vdup input");
      Value scalarForCall = normalizeByteScalarOperandForHivmCall(
          rewriter, op.getLoc(), *normalizedScalar, scalarType);
      callArgs.push_back(scalarForCall);
    }

    callArgs.push_back(mask);
    callArgs.push_back(getI32Constant(rewriter, op.getLoc(), 1));

    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType}, callArgs);
    state.plannedDecls.push_back(
        PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVbrOpPattern final : public OpConversionPattern<pto::VbrOp> {
public:
  explicit LowerVbrOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                             LoweringState &state)
      : OpConversionPattern<pto::VbrOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VbrOp op, pto::VbrOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName =
        buildVbrCallee(op.getContext(),
                       cast<pto::VRegType>(op.getResult().getType()).getElementType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vbr VPTO signature");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert vbr result type");

    Value scalar = adaptor.getValue();
    Type expectedScalarType =
        this->getTypeConverter()->convertType(op.getValue().getType());
    if (!scalar || !expectedScalarType || scalar.getType() != expectedScalarType)
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vbr operand type");

    scalar = normalizeByteScalarOperandForHivmCall(
        rewriter, op.getLoc(), scalar,
        cast<pto::VRegType>(op.getResult().getType()).getElementType());

    auto funcType = rewriter.getFunctionType(TypeRange{scalar.getType()},
                                             TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{resultType},
                                              ValueRange{scalar});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVselrOpPattern final : public OpConversionPattern<pto::VselrOp> {
public:
  explicit LowerVselrOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                               LoweringState &state)
      : OpConversionPattern<pto::VselrOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VselrOp op, pto::VselrOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName =
        buildVselrCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vselr VPTO signature");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    auto resultVectorType = dyn_cast<VectorType>(resultType);
    if (!resultVectorType)
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vselr result type");

    Type intrinsicResultType = resultType;
    if (auto floatType = dyn_cast<FloatType>(resultVectorType.getElementType());
        floatType && floatType.isF32()) {
      intrinsicResultType = VectorType::get(
          resultVectorType.getShape(), rewriter.getI32Type(),
          resultVectorType.getScalableDims());
    }

    Type indexType = this->getTypeConverter()->convertType(op.getSrc1().getType());
    if (!indexType)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert vselr index type");

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    if (!src0 || !src1 || src1.getType() != indexType)
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vselr operand types");

    if (src0.getType() != intrinsicResultType) {
      if (src0.getType() != resultType)
        return rewriter.notifyMatchFailure(op,
                                           "unexpected converted vselr source type");
      src0 = rewriter.create<LLVM::BitcastOp>(op.getLoc(), intrinsicResultType, src0);
    }

    auto funcType = rewriter.getFunctionType(
        TypeRange{intrinsicResultType, indexType}, TypeRange{intrinsicResultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{intrinsicResultType},
        ValueRange{src0, src1});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});

    Value result = call.getResult(0);
    if (intrinsicResultType != resultType)
      result = rewriter.create<LLVM::BitcastOp>(op.getLoc(), resultType, result);
    rewriter.replaceOp(op, ValueRange{result});
    return success();
  }

private:
  LoweringState &state;
};

class LowerPnotOpPattern final : public OpConversionPattern<pto::PnotOp> {
public:
  explicit LowerPnotOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::PnotOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::PnotOp op, pto::PnotOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert pnot result type");

    Value input = adaptor.getInput();
    Value mask = adaptor.getMask();
    if (!input || !mask || input.getType() != resultType ||
        mask.getType() != resultType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted pnot operand types");
    }

    StringRef calleeName = getPredicateMaskCallee<pto::PnotOp>(op.getContext());
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              TypeRange{resultType},
                                              ValueRange{input, mask});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName.str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename InterleaveOp>
class LowerInterleaveOpPattern final
    : public OpConversionPattern<InterleaveOp> {
public:
  explicit LowerInterleaveOpPattern(TypeConverter &typeConverter,
                                    MLIRContext *context, LoweringState &state)
      : OpConversionPattern<InterleaveOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(InterleaveOp op, typename InterleaveOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef stem = std::is_same_v<InterleaveOp, pto::VintlvOp> ? "vintlv" : "vdintlv";
    FailureOr<StringRef> calleeName =
        buildInterleaveCallee(op.getContext(), op.getLow().getType(), stem);
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op,
                                         "unsupported interleave VPTO signature");

    Type lowType = this->getTypeConverter()->convertType(op.getLow().getType());
    Type highType = this->getTypeConverter()->convertType(op.getHigh().getType());
    if (!lowType || !highType || lowType != highType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert interleave result types");
    }

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (!lhs || !rhs || lhs.getType() != lowType || rhs.getType() != lowType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted interleave operand types");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{lowType, lowType},
                                             TypeRange{lowType, highType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{lowType, highType}, ValueRange{lhs, rhs});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename PackOp>
class LowerPredicatePackOpPattern final : public OpConversionPattern<PackOp> {
public:
  explicit LowerPredicatePackOpPattern(TypeConverter &typeConverter,
                                       MLIRContext *context,
                                       LoweringState &state)
      : OpConversionPattern<PackOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(PackOp op, typename PackOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert predicate-pack result type");

    auto part = parseHiLoPartImmediate(op.getPart());
    if (!part)
      return rewriter.notifyMatchFailure(
          op, "unsupported predicate-pack part immediate");

    Value input = adaptor.getInput();
    if (!input || input.getType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "unexpected converted predicate-pack operand type");

    Value partValue = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(*part));
    StringRef calleeName = getPredicatePackCallee<PackOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(
        TypeRange{resultType, rewriter.getI32Type()}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), calleeName, TypeRange{resultType}, ValueRange{input, partValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename UnpackOp>
class LowerUnpackOpPattern final : public OpConversionPattern<UnpackOp> {
public:
  explicit LowerUnpackOpPattern(TypeConverter &typeConverter,
                                MLIRContext *context, LoweringState &state)
      : OpConversionPattern<UnpackOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(UnpackOp op, typename UnpackOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef stem = std::is_same_v<UnpackOp, pto::VsunpackOp> ? "vsunpack"
                                                               : "vzunpack";
    FailureOr<StringRef> calleeName = buildUnpackCallee(
        op.getContext(), op.getSrc().getType(), op.getResult().getType(), stem);
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op,
                                         "unsupported unpack VPTO signature");

    Type srcType = this->getTypeConverter()->convertType(op.getSrc().getType());
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!srcType || !resultType)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert unpack types");

    Value src = adaptor.getSrc();
    if (!src || src.getType() != srcType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted unpack source type");
    }

    Value part = castIntegerLikeTo(op, adaptor.getPart(), rewriter.getI32Type());
    if (!part)
      return rewriter.notifyMatchFailure(op, "failed to materialize unpack part");

    auto funcType = rewriter.getFunctionType(TypeRange{srcType, part.getType()},
                                             TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType}, ValueRange{src, part});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVpackOpPattern final : public OpConversionPattern<pto::VpackOp> {
public:
  explicit LowerVpackOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VpackOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VpackOp op, pto::VpackOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName =
        buildVpackCallee(op.getContext(), op.getSrc().getType(),
                         op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vpack VPTO signature");

    Type srcType = this->getTypeConverter()->convertType(op.getSrc().getType());
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!srcType || !resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert vpack types");

    auto partImm = parseHiLoPartImmediate(op.getPart());
    if (!partImm)
      return rewriter.notifyMatchFailure(op, "unsupported vpack part immediate");

    Value src = adaptor.getSrc();
    if (!src || src.getType() != srcType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted vpack source type");
    }

    Value part = getI32Constant(rewriter, op.getLoc(), *partImm);
    auto funcType = rewriter.getFunctionType(TypeRange{srcType, part.getType()},
                                             TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType}, ValueRange{src, part});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename PredicateMaskOp>
class LowerPredicateMaskBinaryOpPattern final
    : public OpConversionPattern<PredicateMaskOp> {
public:
  explicit LowerPredicateMaskBinaryOpPattern(TypeConverter &typeConverter,
                                             MLIRContext *context,
                                             LoweringState &state)
      : OpConversionPattern<PredicateMaskOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(PredicateMaskOp op, typename PredicateMaskOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert predicate-mask result type");

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value mask = adaptor.getMask();
    if (!src0 || !src1 || !mask || src0.getType() != resultType ||
        src1.getType() != resultType || mask.getType() != resultType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted predicate-mask operand types");
    }

    StringRef calleeName = getPredicateMaskCallee<PredicateMaskOp>(op.getContext());
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              TypeRange{resultType},
                                              ValueRange{src0, src1, mask});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName.str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename ReorderOp>
class LowerPredicatePairReorderOpPattern final
    : public OpConversionPattern<ReorderOp> {
public:
  explicit LowerPredicatePairReorderOpPattern(TypeConverter &typeConverter,
                                              MLIRContext *context,
                                              LoweringState &state)
      : OpConversionPattern<ReorderOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(ReorderOp op, typename ReorderOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(
          op, "failed to convert predicate-pair-reorder result types");
    if (resultTypes.size() != 2 || resultTypes[0] != resultTypes[1])
      return rewriter.notifyMatchFailure(
          op, "unexpected predicate-pair-reorder converted result types");

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (!lhs || !rhs || lhs.getType() != resultTypes[0] ||
        rhs.getType() != resultTypes[0]) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted predicate-pair-reorder operand types");
    }

    StringRef calleeName =
        buildPredicatePairReorderCallee<ReorderOp>(op.getContext());
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName, resultTypes,
                                              ValueRange{lhs, rhs});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName.str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename CmpOp>
class LowerCmpOpPattern final : public OpConversionPattern<CmpOp> {
public:
  explicit LowerCmpOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                             LoweringState &state)
      : OpConversionPattern<CmpOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(CmpOp op, typename CmpOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    constexpr bool isScalarCompare = std::is_same_v<CmpOp, pto::VcmpsOp>;
    Type inputType = Type();
    if constexpr (isScalarCompare)
      inputType = op.getSrc().getType();
    else
      inputType = op.getSrc0().getType();
    FailureOr<StringRef> calleeName =
        buildVcmpCallee(op.getContext(), inputType, op.getCmpMode(),
                        isScalarCompare);
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op,
                                         "unsupported compare VPTO signature");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType =
        this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert compare result type");
    if (resultType != maskType)
      return rewriter.notifyMatchFailure(op,
                                         "unexpected compare mask conversion");

    SmallVector<Value> callArgs;
    callArgs.append(adaptor.getOperands().begin(), adaptor.getOperands().end());
    if constexpr (isScalarCompare) {
      if (callArgs.size() != 3 || !callArgs[0] || !callArgs[1] || !callArgs[2] ||
          callArgs[2].getType() != maskType) {
        return rewriter.notifyMatchFailure(
            op, "unexpected converted scalar-compare operand types");
      }
      callArgs[1] = normalizeByteScalarOperandForHivmCall(
          rewriter, op.getLoc(), callArgs[1],
          cast<pto::VRegType>(op.getSrc().getType()).getElementType());
    } else {
      if (callArgs.size() != 3 || !callArgs[0] || !callArgs[1] || !callArgs[2] ||
          callArgs[0].getType() != callArgs[1].getType() ||
          callArgs[2].getType() != maskType) {
        return rewriter.notifyMatchFailure(
            op, "unexpected converted compare operand types");
      }
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{resultType}, callArgs);
    state.plannedDecls.push_back(
        PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename PltOp>
class LowerPltOpPattern final : public OpConversionPattern<PltOp> {
public:
  explicit LowerPltOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                             LoweringState &state)
      : OpConversionPattern<PltOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(PltOp op, typename PltOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value laneCount = castIntegerLikeTo(op, adaptor.getScalar(), rewriter.getI32Type());
    if (!laneCount)
      return rewriter.notifyMatchFailure(op, "failed to materialize plt lane count");

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "failed to convert plt result types");

    StringRef calleeName = buildPltCallee<PltOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{rewriter.getI32Type()},
                                             resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              resultTypes, ValueRange{laneCount});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename PsetOp>
class LowerPsetOpPattern final : public OpConversionPattern<PsetOp> {
public:
  explicit LowerPsetOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<PsetOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(PsetOp op, typename PsetOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto pattern = parsePredicatePatternImmediate(op.getPattern());
    if (!pattern)
      return rewriter.notifyMatchFailure(op, "unsupported pset pattern");

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "failed to convert pset result types");

    StringRef calleeName = buildPsetCallee<PsetOp>(op.getContext());
    Value patternValue = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(*pattern));
    auto funcType = rewriter.getFunctionType(TypeRange{rewriter.getI32Type()},
                                             resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              resultTypes, ValueRange{patternValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename PgeOp>
class LowerPgeOpPattern final : public OpConversionPattern<PgeOp> {
public:
  explicit LowerPgeOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                             LoweringState &state)
      : OpConversionPattern<PgeOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(PgeOp op, typename PgeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto pattern = parsePredicatePatternImmediate(op.getPattern());
    if (!pattern)
      return rewriter.notifyMatchFailure(op, "unsupported pge pattern");

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "failed to convert pge result types");

    StringRef calleeName = buildPgeCallee<PgeOp>(op.getContext());
    Value patternValue = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(*pattern));
    Value zero = rewriter.create<arith::ConstantOp>(op.getLoc(),
                                                    rewriter.getI32IntegerAttr(0));
    auto funcType = rewriter.getFunctionType(
        TypeRange{rewriter.getI32Type(), rewriter.getI32Type()}, resultTypes);
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), calleeName, resultTypes,
                                      ValueRange{patternValue, zero});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVldsOpPattern final : public OpConversionPattern<pto::VldsOp> {
public:
  explicit LowerVldsOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::VldsOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VldsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type ptoResultType = op.getResult().getType();
    Type elementType = getElementTypeFromVectorLike(ptoResultType);
    if (!elementType)
      return rewriter.notifyMatchFailure(op, "unsupported vlds element type");
    auto offsetBytes = convertElementOffsetToBytes(op, adaptor.getOffset(), elementType);
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    auto dist =
        parseLoadDistImmediate(op.getDist().value_or("NORM"), elementType);
    if (failed(offsetBytes) || !basePtr || !dist)
      return rewriter.notifyMatchFailure(op, "failed to materialize vlds operands");

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "failed to convert vlds result types");

    bool usePostIntrinsic = static_cast<bool>(op.getUpdatedBase());
    if (usePostIntrinsic) {
      if (resultTypes.size() != 2 || resultTypes[1] != adaptor.getSource().getType())
        return rewriter.notifyMatchFailure(op,
                                           "unsupported vlds post-update results");
    } else if (resultTypes.size() != 1) {
      return rewriter.notifyMatchFailure(op, "unsupported vlds result count");
    }

    FailureOr<StringRef> calleeName =
        usePostIntrinsic ? buildVldsPostCallee(op.getContext(), ptoResultType)
                         : buildVldsCallee(op.getContext(), ptoResultType);
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vlds signature");

    Value distValue = getI32Constant(rewriter, op.getLoc(), *dist);
    Value postValue = getI32Constant(rewriter, op.getLoc(), usePostIntrinsic ? 1 : 0);
    SmallVector<Value> args{adaptor.getSource(), *offsetBytes, distValue, postValue};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSource().getType(), (*offsetBytes).getType(),
                  distValue.getType(), postValue.getType()},
        resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              resultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVldsx2OpPattern final : public OpConversionPattern<pto::Vldsx2Op> {
public:
  explicit LowerVldsx2OpPattern(TypeConverter &typeConverter,
                                MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::Vldsx2Op>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::Vldsx2Op op, pto::Vldsx2Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elementType = getElementTypeFromVectorLike(op.getLow().getType());
    if (!elementType)
      return rewriter.notifyMatchFailure(op, "unsupported vldsx2 element type");

    auto offsetBytes =
        convertElementOffsetToBytes(op, adaptor.getOffset(), elementType);
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    auto dist = parseLoadX2DistImmediate(op.getDist(), elementType);
    if (failed(offsetBytes) || !basePtr || !dist) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize vldsx2 operands");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes)) ||
        resultTypes.size() != 2) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert vldsx2 result types");
    }

    FailureOr<StringRef> calleeName =
        buildVldsx2Callee(op.getContext(), op.getLow().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vldsx2 signature");

    Value distValue = getI32Constant(rewriter, op.getLoc(), *dist);
    Value zeroValue = getI32Constant(rewriter, op.getLoc(), 0);
    SmallVector<Value> args{adaptor.getSource(), *offsetBytes, distValue,
                            zeroValue};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSource().getType(), (*offsetBytes).getType(),
                  distValue.getType(), zeroValue.getType()},
        resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              resultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVsldbOpPattern final : public OpConversionPattern<pto::VsldbOp> {
public:
  explicit LowerVsldbOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VsldbOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VsldbOp op, pto::VsldbOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    Value packedStride =
        packBlockRepeatStride(op, adaptor.getBlockStride(), adaptor.getRepeatStride());
    if (!basePtr || !packedStride)
      return rewriter.notifyMatchFailure(op, "failed to materialize vsldb operands");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert vsldb result type");

    StringRef calleeName = buildVsldbCallee(op.getContext());
    Value zeroValue = getI32Constant(rewriter, op.getLoc(), 0);
    SmallVector<Value> args{adaptor.getSource(), packedStride, zeroValue,
                            adaptor.getMask()};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSource().getType(), packedStride.getType(),
                  zeroValue.getType(), adaptor.getMask().getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              TypeRange{resultType}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerInitAlignOpPattern final
    : public OpConversionPattern<pto::InitAlignOp> {
public:
  explicit LowerInitAlignOpPattern(TypeConverter &typeConverter,
                                   MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::InitAlignOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::InitAlignOp op, pto::InitAlignOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert init_align result type");

    StringRef calleeName = buildInitAlignCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{}, TypeRange{resultType});
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{resultType});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVldasOpPattern final : public OpConversionPattern<pto::VldasOp> {
public:
  explicit LowerVldasOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VldasOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VldasOp op, pto::VldasOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!sourceType || !resultType)
      return rewriter.notifyMatchFailure(op,
                                         "expected converted vldas operand/result types");

    StringRef calleeName = buildVldasCallee(op.getContext());
    auto funcType =
        rewriter.getFunctionType(TypeRange{adaptor.getSource().getType()},
                                 TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              TypeRange{resultType},
                                              ValueRange{adaptor.getSource()});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVldusOpPattern final : public OpConversionPattern<pto::VldusOp> {
public:
  explicit LowerVldusOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VldusOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VldusOp op, pto::VldusOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    SmallVector<Type> resultTypes;
    if (!sourceType ||
        failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)) ||
        resultTypes.size() != 2 || adaptor.getAlign().getType() != resultTypes[1]) {
      return rewriter.notifyMatchFailure(op,
                                         "expected converted vldus operand/result types");
    }

    FailureOr<StringRef> calleeName =
        buildVldusCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vldus signature");

    SmallVector<Type> intrinsicResultTypes(resultTypes.begin(), resultTypes.end());
    // The installed no-post A5 vldus intrinsic returns an extra hidden base ptr.
    intrinsicResultTypes.push_back(adaptor.getSource().getType());

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSource().getType(), adaptor.getAlign().getType()},
        intrinsicResultTypes);
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, intrinsicResultTypes,
        ValueRange{adaptor.getSource(), adaptor.getAlign()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults().take_front(resultTypes.size()));
    return success();
  }

private:
  LoweringState &state;
};

class LowerSprclrOpPattern final : public OpConversionPattern<pto::SprclrOp> {
public:
  explicit LowerSprclrOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                                LoweringState &state)
      : OpConversionPattern<pto::SprclrOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::SprclrOp op, pto::SprclrOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto spr = parseSprImmediate(op.getSpr());
    if (!spr)
      return rewriter.notifyMatchFailure(op, "unsupported sprclr target");

    StringRef calleeName = buildSprclrCallee(op.getContext());
    Value sprValue = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI16IntegerAttr(*spr));
    auto funcType = rewriter.getFunctionType(TypeRange{sprValue.getType()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{sprValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerVstsOpPattern final : public OpConversionPattern<pto::VstsOp> {
public:
  explicit LowerVstsOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::VstsOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VstsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elementType = getElementTypeFromVectorLike(op.getValue().getType());
    if (!elementType)
      return rewriter.notifyMatchFailure(op, "unsupported vsts element type");
    auto offsetBytes =
        convertElementOffsetToBytes(op, adaptor.getOffset(), elementType);
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    auto dist =
        parseStoreDistImmediate(op.getDist().value_or(""), elementType);
    if (failed(offsetBytes) || !basePtr || !dist)
      return rewriter.notifyMatchFailure(op, "failed to materialize vsts operands");

    FailureOr<StringRef> calleeName =
        op.getUpdatedBase()
            ? buildVstsPostCallee(op.getContext(), op.getValue().getType())
            : buildVstsCallee(op.getContext(), op.getValue().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vsts signature");

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes)))
      return rewriter.notifyMatchFailure(op, "failed to convert vsts result types");
    bool usePostIntrinsic = static_cast<bool>(op.getUpdatedBase());
    if (usePostIntrinsic) {
      if (resultTypes.size() != 1 ||
          resultTypes[0] != adaptor.getDestination().getType())
        return rewriter.notifyMatchFailure(op,
                                           "unsupported vsts post-update result");
    } else if (!resultTypes.empty()) {
      return rewriter.notifyMatchFailure(op, "unsupported vsts result count");
    }

    Value distValue = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(*dist));
    Value zero = rewriter.create<arith::ConstantOp>(op.getLoc(),
                                                    rewriter.getI32IntegerAttr(
                                                        usePostIntrinsic ? 1 : 0));
    SmallVector<Value> args{adaptor.getValue(), adaptor.getDestination(),
                            *offsetBytes, distValue, zero, adaptor.getMask()};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getValue().getType(), adaptor.getDestination().getType(),
                  rewriter.getI32Type(), rewriter.getI32Type(),
                  rewriter.getI32Type(), adaptor.getMask().getType()},
        resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              resultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    if (usePostIntrinsic)
      rewriter.replaceOp(op, call.getResults());
    else
      rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerVsstbOpPattern final : public OpConversionPattern<pto::VsstbOp> {
public:
  explicit LowerVsstbOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VsstbOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VsstbOp op, pto::VsstbOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto basePtr =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    Value packedStride =
        packBlockRepeatStride(op, adaptor.getBlockStride(), adaptor.getRepeatStride());
    if (!basePtr || !packedStride)
      return rewriter.notifyMatchFailure(op, "failed to materialize vsstb operands");

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes)))
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert vsstb result types");
    bool usePostIntrinsic = static_cast<bool>(op.getUpdatedBase());
    if (usePostIntrinsic) {
      if (resultTypes.size() != 1 ||
          resultTypes[0] != adaptor.getDestination().getType())
        return rewriter.notifyMatchFailure(
            op, "unsupported vsstb post-update result");
    } else if (!resultTypes.empty()) {
      return rewriter.notifyMatchFailure(op, "unsupported vsstb result count");
    }

    StringRef calleeName = usePostIntrinsic
                               ? buildVsstbPostCallee(op.getContext())
                               : buildVsstbCallee(op.getContext());
    Value zeroValue = getI32Constant(rewriter, op.getLoc(), usePostIntrinsic ? 1 : 0);
    SmallVector<Value> args{adaptor.getValue(), adaptor.getDestination(),
                            packedStride, zeroValue, adaptor.getMask()};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getValue().getType(), adaptor.getDestination().getType(),
                  packedStride.getType(), zeroValue.getType(),
                  adaptor.getMask().getType()},
        resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              resultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    if (usePostIntrinsic)
      rewriter.replaceOp(op, call.getResults());
    else
      rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerVstsx2OpPattern final : public OpConversionPattern<pto::Vstsx2Op> {
public:
  explicit LowerVstsx2OpPattern(TypeConverter &typeConverter,
                                MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::Vstsx2Op>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::Vstsx2Op op, pto::Vstsx2Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elementType = getElementTypeFromVectorLike(op.getLow().getType());
    if (!elementType)
      return rewriter.notifyMatchFailure(op, "unsupported vstsx2 element type");

    auto offsetBytes =
        convertElementOffsetToBytes(op, adaptor.getOffset(), elementType);
    auto basePtr =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    auto dist = parseStoreX2DistImmediate(op.getDist(), elementType);
    if (failed(offsetBytes) || !basePtr || !dist) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize vstsx2 operands");
    }

    FailureOr<StringRef> calleeName =
        buildVstsx2Callee(op.getContext(), op.getLow().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vstsx2 signature");

    Value distValue = getI32Constant(rewriter, op.getLoc(), *dist);
    Value zeroValue = getI32Constant(rewriter, op.getLoc(), 0);
    SmallVector<Value> args{adaptor.getLow(), adaptor.getHigh(),
                            adaptor.getDestination(), *offsetBytes, distValue,
                            zeroValue, adaptor.getMask()};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getLow().getType(), adaptor.getHigh().getType(),
                  adaptor.getDestination().getType(), (*offsetBytes).getType(),
                  distValue.getType(), zeroValue.getType(),
                  adaptor.getMask().getType()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerPstuOpPattern final : public OpConversionPattern<pto::PstuOp> {
public:
  explicit LowerPstuOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::PstuOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::PstuOp op, pto::PstuOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName = buildPstuCallee(op.getContext(), op);
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported pstu signature");

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "failed to convert pstu result types");
    if (resultTypes.size() != 2)
      return rewriter.notifyMatchFailure(op, "unexpected converted pstu result arity");

    auto baseType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getBase().getType());
    if (!baseType || adaptor.getAlignIn().getType() != resultTypes[0] ||
        adaptor.getBase().getType() != resultTypes[1]) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted pstu operand/result types");
    }

    SmallVector<Value> args{adaptor.getValue(), adaptor.getBase(), adaptor.getAlignIn()};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getValue().getType(), adaptor.getBase().getType(),
                  adaptor.getAlignIn().getType()},
        resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, resultTypes,
                                              args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVstusOpPattern final : public OpConversionPattern<pto::VstusOp> {
public:
  explicit LowerVstusOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                               LoweringState &state)
      : OpConversionPattern<pto::VstusOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VstusOp op, pto::VstusOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elementType = getElementTypeFromVectorLike(op.getValue().getType());
    if (!elementType)
      return rewriter.notifyMatchFailure(op, "unsupported vstus element type");

    auto offsetBytes = convertElementOffsetToBytes(op, adaptor.getOffset(), elementType);
    if (failed(offsetBytes))
      return rewriter.notifyMatchFailure(op, "failed to convert vstus offset");

    Type resultType = this->getTypeConverter()->convertType(op.getAlignOut().getType());
    auto baseType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getBase().getType());
    if (!resultType || !baseType || adaptor.getAlignIn().getType() != resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vstus operand/result types");
    }

    StringRef calleeName = buildVstusCallee(op.getContext());
    SmallVector<Value> args{adaptor.getValue(), adaptor.getBase(), *offsetBytes,
                            adaptor.getAlignIn()};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getValue().getType(), adaptor.getBase().getType(),
                  (*offsetBytes).getType(), adaptor.getAlignIn().getType()},
        TypeRange{resultType});
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{resultType}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVsturOpPattern final : public OpConversionPattern<pto::VsturOp> {
public:
  explicit LowerVsturOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                               LoweringState &state)
      : OpConversionPattern<pto::VsturOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VsturOp op, pto::VsturOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto postMode = parsePostModeImmediate(op.getMode());
    if (!postMode)
      return rewriter.notifyMatchFailure(op, "unsupported vstur mode immediate");

    Type resultType = this->getTypeConverter()->convertType(op.getAlignOut().getType());
    auto baseType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getBase().getType());
    if (!resultType || !baseType || adaptor.getAlignIn().getType() != resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vstur operand/result types");
    }

    StringRef calleeName = buildVsturCallee(op.getContext());
    Value modeValue = getI32Constant(rewriter, op.getLoc(), *postMode);
    Value zeroValue = getI32Constant(rewriter, op.getLoc(), 0);
    SmallVector<Value> args{adaptor.getValue(), adaptor.getBase(), adaptor.getAlignIn(),
                            modeValue, zeroValue};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getValue().getType(), adaptor.getBase().getType(),
                  adaptor.getAlignIn().getType(), modeValue.getType(),
                  zeroValue.getType()},
        TypeRange{resultType});
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{resultType}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVstarOpPattern final : public OpConversionPattern<pto::VstarOp> {
public:
  explicit LowerVstarOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                               LoweringState &state)
      : OpConversionPattern<pto::VstarOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VstarOp op, pto::VstarOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto baseType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    Type alignType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!baseType || !alignType || adaptor.getValue().getType() != alignType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vstar operand types");
    }

    StringRef calleeName = buildVstarCallee(op.getContext());
    Value zeroValue = getI32Constant(rewriter, op.getLoc(), 0);
    SmallVector<Value> args{adaptor.getValue(), adaptor.getDestination(), zeroValue};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getValue().getType(), adaptor.getDestination().getType(),
                  zeroValue.getType()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerVstasOpPattern final : public OpConversionPattern<pto::VstasOp> {
public:
  explicit LowerVstasOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                               LoweringState &state)
      : OpConversionPattern<pto::VstasOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VstasOp op, pto::VstasOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto baseType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    Type alignType = this->getTypeConverter()->convertType(op.getValue().getType());
    auto dstType = dyn_cast<pto::PtrType>(op.getDestination().getType());
    if (!baseType || !alignType || adaptor.getValue().getType() != alignType || !dstType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vstas operand types");
    }

    auto offsetBytes =
        convertElementOffsetToBytes(op, adaptor.getOffset(), dstType.getElementType());
    if (failed(offsetBytes))
      return rewriter.notifyMatchFailure(op, "failed to convert vstas offset");

    StringRef calleeName = buildVstasCallee(op.getContext());
    Value zeroValue = getI32Constant(rewriter, op.getLoc(), 0);
    SmallVector<Value> args{adaptor.getValue(), adaptor.getDestination(), *offsetBytes,
                            zeroValue};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getValue().getType(), adaptor.getDestination().getType(),
                  (*offsetBytes).getType(), zeroValue.getType()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerVgather2OpPattern final
    : public OpConversionPattern<pto::Vgather2Op> {
public:
  explicit LowerVgather2OpPattern(TypeConverter &typeConverter,
                                  MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::Vgather2Op>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::Vgather2Op op, pto::Vgather2Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elemType = getElementTypeFromVectorLike(op.getResult().getType());
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    if (!elemType || !basePtr)
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vgather2 operand types");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert vgather2 result type");

    FailureOr<StringRef> calleeName =
        buildVgather2Callee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vgather2 signature");

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSource().getType(), adaptor.getOffsets().getType(),
                  adaptor.getMask().getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getSource(), adaptor.getOffsets(), adaptor.getMask()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVgather2BcOpPattern final
    : public OpConversionPattern<pto::Vgather2BcOp> {
public:
  explicit LowerVgather2BcOpPattern(TypeConverter &typeConverter,
                                    MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::Vgather2BcOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::Vgather2BcOp op, pto::Vgather2BcOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!basePtr || !resultType)
      return rewriter.notifyMatchFailure(op,
          "unexpected converted vgather2_bc operand/result types");

    FailureOr<StringRef> calleeName =
        buildVgather2BcCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vgather2_bc signature");

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSource().getType(), adaptor.getOffsets().getType(),
                  adaptor.getMask().getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getSource(), adaptor.getOffsets(), adaptor.getMask()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVgatherbOpPattern final
    : public OpConversionPattern<pto::VgatherbOp> {
public:
  explicit LowerVgatherbOpPattern(TypeConverter &typeConverter,
                                  MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VgatherbOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::VgatherbOp op, pto::VgatherbOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!basePtr || !resultType)
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vgatherb operand/result types");

    FailureOr<StringRef> calleeName =
        buildVgatherbCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vgatherb signature");

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSource().getType(), adaptor.getOffsets().getType(),
                  adaptor.getMask().getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getSource(), adaptor.getOffsets(), adaptor.getMask()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVscatterOpPattern final
    : public OpConversionPattern<pto::VscatterOp> {
public:
  explicit LowerVscatterOpPattern(TypeConverter &typeConverter,
                                  MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VscatterOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::VscatterOp op, pto::VscatterOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elemType = getElementTypeFromVectorLike(op.getValue().getType());
    auto basePtr =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    if (!elemType || !basePtr)
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vscatter operand types");

    FailureOr<StringRef> calleeName =
        buildVscatterCallee(op.getContext(), op.getValue().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vscatter signature");

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getValue().getType(), adaptor.getDestination().getType(),
                  adaptor.getOffsets().getType(), adaptor.getMask().getType()},
        TypeRange{});
    rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{},
        ValueRange{adaptor.getValue(), adaptor.getDestination(),
                   adaptor.getOffsets(), adaptor.getMask()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerVaxpyOpPattern final : public OpConversionPattern<pto::VaxpyOp> {
public:
  explicit LowerVaxpyOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VaxpyOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VaxpyOp op, pto::VaxpyOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elemType = getElementTypeFromVectorLike(op.getResult().getType());
    if (!elemType)
      return rewriter.notifyMatchFailure(op, "unsupported vaxpy signature");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert vaxpy result type");

    FailureOr<StringRef> calleeName =
        buildVaxpyCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vaxpy callee");

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSrc1().getType(), adaptor.getSrc0().getType(),
                  adaptor.getAlpha().getType(), adaptor.getMask().getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getSrc1(), adaptor.getSrc0(), adaptor.getAlpha(),
                   adaptor.getMask()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVciOpPattern final : public OpConversionPattern<pto::VciOp> {
public:
  explicit LowerVciOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                             LoweringState &state)
      : OpConversionPattern<pto::VciOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VciOp op, pto::VciOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto order = parseOrderImmediate(op.getOrder().value_or("ASC"));
    if (!order)
      return rewriter.notifyMatchFailure(op, "unsupported vci order");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert vci result type");

    FailureOr<StringRef> calleeName =
        buildVciCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vci callee");

    Value indexValue = adaptor.getIndex();
    Type resultElemType =
        cast<pto::VRegType>(op.getResult().getType()).getElementType();
    if (auto intType = dyn_cast<IntegerType>(resultElemType)) {
      if (intType.getWidth() == 8) {
        Type loweredIndexType = rewriter.getI16Type();
        if (intType.isUnsigned())
          indexValue = rewriter.create<arith::ExtUIOp>(op.getLoc(),
                                                       loweredIndexType,
                                                       indexValue);
        else
          indexValue = rewriter.create<arith::ExtSIOp>(op.getLoc(),
                                                       loweredIndexType,
                                                       indexValue);
      }
    }

    Value orderValue = getI32Constant(rewriter, op.getLoc(), *order);
    auto funcType = rewriter.getFunctionType(
        TypeRange{indexValue.getType(), orderValue.getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{indexValue, orderValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVexpdifOpPattern final
    : public OpConversionPattern<pto::VexpdifOp> {
public:
  explicit LowerVexpdifOpPattern(TypeConverter &typeConverter,
                                 MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VexpdifOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::VexpdifOp op, pto::VexpdifOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto part = parsePartImmediate(op.getPart());
    if (!part)
      return rewriter.notifyMatchFailure(op, "unsupported vexpdif signature");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert vexpdif result type");

    FailureOr<StringRef> calleeName =
        buildVexpdifCallee(op.getContext(), op.getInput().getType(),
                           op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vexpdif callee");

    Value partValue = getI32Constant(rewriter, op.getLoc(), *part);
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getInput().getType(), adaptor.getMax().getType(),
                  adaptor.getMask().getType(), partValue.getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getInput(), adaptor.getMax(), adaptor.getMask(),
                   partValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVbitsortOpPattern final
    : public OpConversionPattern<pto::VbitsortOp> {
public:
  explicit LowerVbitsortOpPattern(TypeConverter &typeConverter,
                                  MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VbitsortOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::VbitsortOp op, pto::VbitsortOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    auto srcType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    auto idxType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getIndices().getType());
    if (!dstType || !srcType || !idxType)
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vbitsort operand types");

    FailureOr<Value> config = packVbitsortConfig(op, adaptor.getRepeatTimes());
    if (failed(config))
      return rewriter.notifyMatchFailure(op, "failed to pack vbitsort config");

    FailureOr<StringRef> calleeName = buildVbitsortCallee(op.getContext(), op);
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vbitsort signature");

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getDestination().getType(), adaptor.getSource().getType(),
                  adaptor.getIndices().getType(), (*config).getType()},
        TypeRange{});
    rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{},
        ValueRange{adaptor.getDestination(), adaptor.getSource(),
                   adaptor.getIndices(), *config});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerVmrgsort4OpPattern final
    : public OpConversionPattern<pto::Vmrgsort4Op> {
public:
  explicit LowerVmrgsort4OpPattern(TypeConverter &typeConverter,
                                   MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::Vmrgsort4Op>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::Vmrgsort4Op op, pto::Vmrgsort4Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    auto src0Type =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource0().getType());
    auto src1Type =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource1().getType());
    auto src2Type =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource2().getType());
    auto src3Type =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource3().getType());
    if (!dstType || !src0Type || !src1Type || !src2Type || !src3Type)
      return rewriter.notifyMatchFailure(
          op, "unexpected converted vmrgsort4 operand types");

    Type elemType =
        cast<pto::PtrType>(op.getDestination().getType()).getElementType();
    FailureOr<Value> packedSrc = packVmrgsort4SourceAddr(
        op, adaptor.getSource0(), adaptor.getSource1(), adaptor.getSource2(),
        adaptor.getSource3(), elemType);
    if (failed(packedSrc))
      return rewriter.notifyMatchFailure(
          op, "failed to pack vmrgsort4 source addresses");

    FailureOr<Value> dst = reinterpretPointerToAddrSpace(op, adaptor.getDestination(), 6);
    if (failed(dst))
      return rewriter.notifyMatchFailure(op, "failed to normalize vmrgsort4 destination");

    FailureOr<StringRef> calleeName = buildVmrgsort4Callee(op.getContext(), op);
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vmrgsort4 signature");

    auto funcType = rewriter.getFunctionType(
        TypeRange{(*dst).getType(), (*packedSrc).getType(),
                  adaptor.getCount().getType(), adaptor.getConfig().getType()},
        TypeRange{});
    rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{},
        ValueRange{*dst, *packedSrc, adaptor.getCount(), adaptor.getConfig()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerVcvtOpPattern final : public OpConversionPattern<pto::VcvtOp> {
public:
  explicit LowerVcvtOpPattern(TypeConverter &typeConverter,
                              MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VcvtOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VcvtOp op, pto::VcvtOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<VcvtContract> contract = buildVcvtContract(op);
    if (failed(contract))
      return rewriter.notifyMatchFailure(op, "unsupported vcvt type pair");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert vcvt result type");

    SmallVector<Value> callArgs;
    SmallVector<Type> argTypes;
    callArgs.push_back(adaptor.getInput());
    argTypes.push_back(adaptor.getInput().getType());
    callArgs.push_back(adaptor.getMask());
    argTypes.push_back(adaptor.getMask().getType());

    auto appendRndArg = [&]() -> LogicalResult {
      auto roundMode =
          op.getRndAttr() ? parseRoundModeImmediate(*op.getRnd()) : std::nullopt;
      if (!roundMode)
        return rewriter.notifyMatchFailure(op, "vcvt requires valid rnd attr");
      Value roundValue = getI32Constant(rewriter, op.getLoc(), *roundMode);
      callArgs.push_back(roundValue);
      argTypes.push_back(roundValue.getType());
      return success();
    };

    auto appendSatArg = [&]() -> LogicalResult {
      auto saturation =
          op.getSatAttr() ? parseSaturationImmediate(*op.getSat()) : std::nullopt;
      if (!saturation)
        return rewriter.notifyMatchFailure(op, "vcvt requires valid sat attr");
      Value satValue = getI32Constant(rewriter, op.getLoc(), *saturation);
      callArgs.push_back(satValue);
      argTypes.push_back(satValue.getType());
      return success();
    };

    if ((*contract).satBeforeRnd) {
      if ((*contract).requiresSat && failed(appendSatArg()))
        return failure();
      if ((*contract).requiresRnd && failed(appendRndArg()))
        return failure();
    } else {
      if ((*contract).requiresRnd && failed(appendRndArg()))
        return failure();
      if ((*contract).requiresSat && failed(appendSatArg()))
        return failure();
    }

    if ((*contract).requiresPart) {
      auto part =
          op.getPartAttr() ? parseVcvtPartImmediate(*op.getPart()) : std::nullopt;
      if (!part)
        return rewriter.notifyMatchFailure(op, "vcvt requires valid part attr");
      Value partValue = getI32Constant(rewriter, op.getLoc(), *part);
      callArgs.push_back(partValue);
      argTypes.push_back(partValue.getType());
    }

    auto funcType = rewriter.getFunctionType(argTypes, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), StringRef((*contract).intrinsic), TypeRange{resultType}, callArgs);
    state.plannedDecls.push_back(
        PlannedDecl{std::string((*contract).intrinsic), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVbitcastOpPattern final
    : public OpConversionPattern<pto::VbitcastOp> {
public:
  explicit LowerVbitcastOpPattern(TypeConverter &typeConverter,
                                  MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VbitcastOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(pto::VbitcastOp op, pto::VbitcastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert vbitcast result type");
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, resultType,
                                                 adaptor.getInput());
    return success();
  }
};

class LowerPbitcastOpPattern final
    : public OpConversionPattern<pto::PbitcastOp> {
public:
  explicit LowerPbitcastOpPattern(TypeConverter &typeConverter,
                                  MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::PbitcastOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(pto::PbitcastOp op, pto::PbitcastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert pbitcast result type");
    if (adaptor.getInput().getType() != resultType) {
      return rewriter.notifyMatchFailure(
          op, "pbitcast expects identical lowered input/result types");
    }
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

class LowerVtrcOpPattern final : public OpConversionPattern<pto::VtrcOp> {
public:
  explicit LowerVtrcOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::VtrcOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VtrcOp op, pto::VtrcOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto roundMode = parseRoundModeImmediate(op.getRoundMode());
    if (!roundMode)
      return rewriter.notifyMatchFailure(op, "unsupported vtrc signature");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert vtrc result type");

    FailureOr<StringRef> calleeName =
        buildVtrcCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vtrc callee");

    Value roundValue = getI32Constant(rewriter, op.getLoc(), *roundMode);
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getInput().getType(), roundValue.getType(),
                  adaptor.getMask().getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getInput(), roundValue, adaptor.getMask()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename StoreOp>
class LowerPredicateStoreOpPattern final : public OpConversionPattern<StoreOp> {
public:
  explicit LowerPredicateStoreOpPattern(TypeConverter &typeConverter,
                                        MLIRContext *context,
                                        LoweringState &state)
      : OpConversionPattern<StoreOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(StoreOp op, typename StoreOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto llvmDestType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    Type valueType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!llvmDestType || !valueType)
      return rewriter.notifyMatchFailure(
          op, "expected converted predicate-store operand types");

    auto dist = parsePredicateStoreDistImmediate(op.getDist());
    if (!dist)
      return rewriter.notifyMatchFailure(
          op, "unsupported predicate-store dist immediate");

    Value offset = castIntegerLikeTo(op, adaptor.getOffset(), rewriter.getI32Type());
    if (!offset)
      return rewriter.notifyMatchFailure(
          op, "failed to convert predicate-store offset to i32");

    StringRef calleeName = getPredicateStoreCallee<StoreOp>(op.getContext());
    SmallVector<Value> args;
    args.push_back(adaptor.getValue());
    args.push_back(adaptor.getDestination());
    args.push_back(offset);
    args.push_back(rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(*dist)));
    args.push_back(rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(0)));
    auto funcType = rewriter.getFunctionType(
        TypeRange{valueType, llvmDestType, rewriter.getI32Type(),
                  rewriter.getI32Type(), rewriter.getI32Type()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename LoadOp>
class LowerPredicateLoadOpPattern final : public OpConversionPattern<LoadOp> {
public:
  explicit LowerPredicateLoadOpPattern(TypeConverter &typeConverter,
                                       MLIRContext *context,
                                       LoweringState &state)
      : OpConversionPattern<LoadOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(LoadOp op, typename LoadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto llvmSourceType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!llvmSourceType || !resultType)
      return rewriter.notifyMatchFailure(
          op, "expected converted predicate-load operand/result types");

    auto dist = parsePredicateLoadDistImmediate(op.getDist());
    if (!dist)
      return rewriter.notifyMatchFailure(
          op, "unsupported predicate-load dist immediate");

    Value offset = castIntegerLikeTo(op, adaptor.getOffset(), rewriter.getI32Type());
    if (!offset)
      return rewriter.notifyMatchFailure(
          op, "failed to convert predicate-load offset to i32");

    StringRef calleeName = getPredicateLoadCallee<LoadOp>(op.getContext());
    SmallVector<Value> args;
    args.push_back(adaptor.getSource());
    args.push_back(offset);
    args.push_back(rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(*dist)));
    args.push_back(rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(0)));
    auto funcType = rewriter.getFunctionType(
        TypeRange{llvmSourceType, rewriter.getI32Type(), rewriter.getI32Type(),
                  rewriter.getI32Type()},
        TypeRange{resultType});
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), calleeName, resultType, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename LoopOp>
class LowerSetLoopConfigOpPattern final : public OpConversionPattern<LoopOp> {
public:
  explicit LowerSetLoopConfigOpPattern(TypeConverter &typeConverter,
                                       MLIRContext *context,
                                       LoweringState &state)
      : OpConversionPattern<LoopOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(LoopOp op, typename LoopOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> packed = failure();
    if constexpr (std::is_same_v<LoopOp, pto::SetLoopSizeOutToUbOp> ||
                  std::is_same_v<LoopOp, pto::SetLoopSizeUbToOutOp>) {
      packed = packLoopSize(op, adaptor.getFirst(), adaptor.getSecond());
    } else {
      packed = packLoopPair(op, adaptor.getFirst(), adaptor.getSecond());
    }
    if (failed(packed))
      return rewriter.notifyMatchFailure(op,
                                         "failed to pack loop configuration");

    StringRef calleeName = buildSetLoopCallee<LoopOp>(op.getContext());
    auto funcType =
        rewriter.getFunctionType(TypeRange{rewriter.getI64Type()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{*packed});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename ConfigOp>
class LowerUnaryConfigOpPattern final : public OpConversionPattern<ConfigOp> {
public:
  explicit LowerUnaryConfigOpPattern(TypeConverter &typeConverter,
                                     MLIRContext *context,
                                     LoweringState &state)
      : OpConversionPattern<ConfigOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(ConfigOp op, typename ConfigOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> encoded =
        encodeMovPadValue(op.getLoc(), adaptor.getValue(), rewriter);
    if (failed(encoded))
      return rewriter.notifyMatchFailure(
          op, "expected 8/16/32-bit integer or float mov-pad payload");

    StringRef calleeName = buildUnaryConfigCallee<ConfigOp>(op.getContext());
    auto funcType =
        rewriter.getFunctionType(TypeRange{rewriter.getI64Type()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{*encoded});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename ConfigOp>
class LowerUnaryI64ConfigOpPattern final : public OpConversionPattern<ConfigOp> {
public:
  explicit LowerUnaryI64ConfigOpPattern(TypeConverter &typeConverter,
                                        MLIRContext *context,
                                        LoweringState &state)
      : OpConversionPattern<ConfigOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(ConfigOp op, typename ConfigOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef calleeName = buildUnaryConfigCallee<ConfigOp>(op.getContext());
    auto funcType =
        rewriter.getFunctionType(TypeRange{adaptor.getValue().getType()},
                                 TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{adaptor.getValue()});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerStoreVfSimtInfoOpPattern final
    : public OpConversionPattern<pto::StoreVfSimtInfoOp> {
public:
  explicit LowerStoreVfSimtInfoOpPattern(TypeConverter &typeConverter,
                                         MLIRContext *context,
                                         LoweringState &state)
      : OpConversionPattern<pto::StoreVfSimtInfoOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::StoreVfSimtInfoOp op,
                  pto::StoreVfSimtInfoOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value dimZ = adaptor.getDimZ();
    Value dimY = adaptor.getDimY();
    Value dimX = adaptor.getDimX();
    if (!dimZ || !dimY || !dimX)
      return rewriter.notifyMatchFailure(op, "missing converted SIMT dims");

    auto i64Type = rewriter.getI64Type();
    auto castToI64 = [&](Value value) -> Value {
      if (value.getType().isInteger(64))
        return value;
      return rewriter.create<arith::ExtUIOp>(loc, i64Type, value).getResult();
    };

    Value dimZI64 = castToI64(dimZ);
    Value dimYI64 = castToI64(dimY);
    Value dimXI64 = castToI64(dimX);
    Value dimYShift = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(16));
    Value dimZShift = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(32));
    Value packedDimY =
        rewriter.create<arith::ShLIOp>(loc, dimYI64, dimYShift).getResult();
    Value packedDimZ =
        rewriter.create<arith::ShLIOp>(loc, dimZI64, dimZShift).getResult();
    Value payload =
        rewriter.create<arith::OrIOp>(loc, dimXI64, packedDimY).getResult();
    payload =
        rewriter.create<arith::OrIOp>(loc, payload, packedDimZ).getResult();

    StringRef calleeName = buildStoreVfSimtInfoCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{i64Type}, TypeRange{});
    rewriter.create<func::CallOp>(loc, calleeName, TypeRange{},
                                  ValueRange{payload});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename FenceOp>
static StringRef buildSimtFenceCallee(MLIRContext *context);

template <>
StringRef buildSimtFenceCallee<pto::SyncthreadsOp>(MLIRContext *context) {
  return buildSyncthreadsCallee(context);
}

template <>
StringRef buildSimtFenceCallee<pto::ThreadfenceOp>(MLIRContext *context) {
  return buildThreadfenceCallee(context);
}

template <>
StringRef buildSimtFenceCallee<pto::ThreadfenceBlockOp>(MLIRContext *context) {
  return buildThreadfenceBlockCallee(context);
}

template <typename FenceOp>
class LowerSimtFenceOpPattern final : public OpConversionPattern<FenceOp> {
public:
  explicit LowerSimtFenceOpPattern(TypeConverter &typeConverter,
                                   MLIRContext *context,
                                   LoweringState &state)
      : OpConversionPattern<FenceOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(FenceOp op, typename FenceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    FunctionType funcType = rewriter.getFunctionType({}, {});
    StringRef calleeName = buildSimtFenceCallee<FenceOp>(op.getContext());
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

static std::string buildRepeatedInlineAsmConstraints(StringRef constraint,
                                                     size_t count) {
  std::string result;
  llvm::raw_string_ostream os(result);
  for (size_t i = 0; i < count; ++i) {
    if (i != 0)
      os << ",";
    os << constraint;
  }
  return os.str();
}

static std::string appendSimtKeepResumeClobbers(std::string constraints,
                                                ArrayRef<std::pair<int64_t, unsigned>> physicalRegs) {
  llvm::raw_string_ostream os(constraints);
  // The asm body names fixed SIMT R registers directly. Model those registers
  // as clobbers so LLVM does not allocate a different operand/result into a
  // slot that another keep/resume line reads or overwrites in the same asm.
  for (auto [reg, registerCount] : physicalRegs) {
    for (unsigned offset = 0; offset < registerCount; ++offset)
      os << ",~{R" << (reg + offset) << "}";
  }
  return os.str();
}

template <typename OpT>
static SmallVector<OpT, 4> collectConsecutiveOps(OpT first) {
  SmallVector<OpT, 4> ops;
  for (Operation *cur = first.getOperation(); cur; cur = cur->getNextNode()) {
    auto typed = dyn_cast<OpT>(cur);
    if (!typed)
      break;
    ops.push_back(typed);
  }
  return ops;
}

static bool hasPreviousSameOp(Operation *op) {
  Operation *prev = op->getPrevNode();
  return prev && prev->getName() == op->getName();
}

static std::optional<unsigned> getSimtKeepResumeBitWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (intType.getWidth() <= 64)
      return intType.getWidth();
    return std::nullopt;
  }
  if (type.isF16() || type.isBF16())
    return 16;
  if (type.isF32())
    return 32;
  return std::nullopt;
}

static Value packSimtKeepResumePayload(Location loc, Value value,
                                       ConversionPatternRewriter &rewriter) {
  Type type = value.getType();
  std::optional<unsigned> width = getSimtKeepResumeBitWidth(type);
  if (!width)
    return {};

  Type intType = rewriter.getIntegerType(*width);
  Value bits = value;
  if (!isa<IntegerType>(type))
    bits = rewriter.create<LLVM::BitcastOp>(loc, intType, value);
  else if (bits.getType() != intType)
    bits = rewriter.create<LLVM::BitcastOp>(loc, intType, bits);
  if (*width < 32)
    return rewriter.create<LLVM::ZExtOp>(loc, rewriter.getI32Type(), bits);
  if (*width == 32 && bits.getType() != rewriter.getI32Type())
    return rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), bits);
  return bits;
}

static Value unpackSimtKeepResumePayload(Location loc, Value value,
                                         Type resultType,
                                         ConversionPatternRewriter &rewriter) {
  std::optional<unsigned> width = getSimtKeepResumeBitWidth(resultType);
  if (!width)
    return {};

  Type intType = rewriter.getIntegerType(*width);
  Value bits = value;
  if (*width < 32)
    bits = rewriter.create<LLVM::TruncOp>(loc, intType, bits);
  else if (bits.getType() != intType)
    bits = rewriter.create<LLVM::BitcastOp>(loc, intType, bits);

  if (isa<IntegerType>(resultType)) {
    if (bits.getType() == resultType)
      return bits;
    return rewriter.create<LLVM::BitcastOp>(loc, resultType, bits);
  }
  return rewriter.create<LLVM::BitcastOp>(loc, resultType, bits);
}

static unsigned getSimtKeepResumeRegisterCount(Type type) {
  std::optional<unsigned> width = getSimtKeepResumeBitWidth(type);
  return width && *width > 32 ? 2 : 1;
}

static FailureOr<SmallVector<int64_t, 4>> computeSimtKeepResumePhysicalRegs(
    ArrayRef<std::pair<int64_t, unsigned>> logicalSlots) {
  SmallVector<int64_t, 4> physicalRegs(logicalSlots.size(), -1);
  for (auto [index, logicalSlot] : llvm::enumerate(logicalSlots)) {
    auto [slot, registerCount] = logicalSlot;
    if (slot < 0 || slot >= 123)
      return failure();
    if (registerCount == 2 && ((slot % 2) != 0 || slot + 1 >= 123))
      return failure();
    // Slots are user-assigned storage words, not dense ordinals in the current
    // keep/resume group. This keeps a consumer that resumes only a subset of
    // slots from changing where the remaining slots are read from.
    int64_t reg = 4 + slot;
    if (reg + static_cast<int64_t>(registerCount) - 1 > 126)
      return failure();
    physicalRegs[index] = reg;
  }
  return physicalRegs;
}

static bool isValidSimtKeepResumeSlot(int64_t slot, unsigned registerCount) {
  if (slot < 0 || slot >= 123)
    return false;
  if (registerCount == 2 && ((slot % 2) != 0 || slot + 1 >= 123))
    return false;
  return true;
}

class LowerKeepOpPattern final : public OpConversionPattern<pto::KeepOp> {
public:
  explicit LowerKeepOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::KeepOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::KeepOp op, pto::KeepOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    if (hasPreviousSameOp(op.getOperation()))
      return rewriter.notifyMatchFailure(
          op, "only the first keep in a contiguous group is lowered");

    SmallVector<pto::KeepOp, 4> keepOps = collectConsecutiveOps(op);
    SmallVector<Value, 4> payloads;
    SmallVector<std::pair<int64_t, unsigned>, 4> logicalSlots;
    std::string asmString;
    llvm::raw_string_ostream asmOS(asmString);
    for (pto::KeepOp keep : keepOps) {
      Value payload = rewriter.getRemappedValue(keep.getPayload());
      if (!payload)
        return rewriter.notifyMatchFailure(keep, "payload is not remapped");
      payload = packSimtKeepResumePayload(keep.getLoc(), payload, rewriter);
      if (!payload)
        return rewriter.notifyMatchFailure(
            keep, "expected integer scalar up to 64 bits or f16/bf16/f32");
      int64_t slot = keep.getSlot();
      unsigned registerCount = getSimtKeepResumeRegisterCount(payload.getType());
      if (!isValidSimtKeepResumeSlot(slot, registerCount))
        return rewriter.notifyMatchFailure(keep,
                                           "slot must be in range [0, 122] and 64-bit slots must be even");
      logicalSlots.push_back({slot, registerCount});
      payloads.push_back(payload);
    }
    FailureOr<SmallVector<int64_t, 4>> physicalRegs =
        computeSimtKeepResumePhysicalRegs(logicalSlots);
    if (failed(physicalRegs))
      return rewriter.notifyMatchFailure(
          op, "keep slots must map to valid non-overlapping SIMT registers");

    SmallVector<std::pair<int64_t, unsigned>, 4> clobbers;
    for (auto [index, keep] : llvm::enumerate(keepOps)) {
      (void)keep;
      if (index != 0)
        asmOS << "\n";
      if (logicalSlots[index].second == 2)
        asmOS << "IMAD.WIDE.u32 R" << (*physicalRegs)[index]
              << ", RZ, RZ, $" << index << " wait:0b0000000 stall:1";
      else
        asmOS << "MOV R" << (*physicalRegs)[index] << ", $" << index
              << " wait:0b0000000 stall:1";
      clobbers.push_back({(*physicalRegs)[index], logicalSlots[index].second});
    }
    asmOS.flush();

    rewriter.setInsertionPoint(op);
    rewriter.create<LLVM::InlineAsmOp>(
        op.getLoc(), TypeRange{}, payloads, asmString,
        appendSimtKeepResumeClobbers(
            buildRepeatedInlineAsmConstraints("R", payloads.size()), clobbers),
        true, false,
        LLVM::AsmDialectAttr::get(op.getContext(), LLVM::AsmDialect::AD_ATT),
        ArrayAttr{});
    for (pto::KeepOp keep : llvm::reverse(keepOps))
      rewriter.eraseOp(keep);
    return success();
  }

private:
  LoweringState &state;
};

class LowerResumeOpPattern final : public OpConversionPattern<pto::ResumeOp> {
public:
  explicit LowerResumeOpPattern(TypeConverter &typeConverter,
                                MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::ResumeOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::ResumeOp op, pto::ResumeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    if (hasPreviousSameOp(op.getOperation()))
      return rewriter.notifyMatchFailure(
          op, "only the first resume in a contiguous group is lowered");

    SmallVector<pto::ResumeOp, 4> resumeOps = collectConsecutiveOps(op);
    SmallVector<std::pair<int64_t, unsigned>, 4> logicalSlots;
    SmallVector<Type, 4> asmResultTypes;
    std::string asmString;
    llvm::raw_string_ostream asmOS(asmString);
    for (pto::ResumeOp resume : resumeOps) {
      Type resultType = getTypeConverter()->convertType(resume.getType());
      if (!resultType || !getSimtKeepResumeBitWidth(resultType))
        return rewriter.notifyMatchFailure(
            resume, "expected integer scalar up to 64 bits or f16/bf16/f32");
      int64_t slot = resume.getSlot();
      unsigned registerCount = getSimtKeepResumeRegisterCount(resultType);
      if (!isValidSimtKeepResumeSlot(slot, registerCount))
        return rewriter.notifyMatchFailure(resume,
                                           "slot must be in range [0, 122] and 64-bit slots must be even");
      logicalSlots.push_back({slot, registerCount});
      asmResultTypes.push_back(rewriter.getIntegerType(
          *getSimtKeepResumeBitWidth(resultType) > 32 ? 64 : 32));
    }
    FailureOr<SmallVector<int64_t, 4>> physicalRegs =
        computeSimtKeepResumePhysicalRegs(logicalSlots);
    if (failed(physicalRegs))
      return rewriter.notifyMatchFailure(
          op, "resume slots must map to valid non-overlapping SIMT registers");

    SmallVector<std::pair<int64_t, unsigned>, 4> clobbers;
    for (auto [index, resume] : llvm::enumerate(resumeOps)) {
      (void)resume;
      if (index != 0)
        asmOS << "\n";
      if (logicalSlots[index].second == 2)
        asmOS << "IMAD.WIDE.u32 $" << index << ", RZ, RZ, R"
              << (*physicalRegs)[index] << " wait:0b0000000 stall:1";
      else
        asmOS << "MOV $" << index << ", R" << (*physicalRegs)[index]
              << " wait:0b0000000 stall:1";
      clobbers.push_back({(*physicalRegs)[index], logicalSlots[index].second});
    }
    asmOS.flush();

    Type asmResultType = asmResultTypes.front();
    if (asmResultTypes.size() > 1) {
      asmResultType =
          LLVM::LLVMStructType::getLiteral(op.getContext(), asmResultTypes);
    }
    rewriter.setInsertionPoint(op);
    auto asmOp = rewriter.create<LLVM::InlineAsmOp>(
        op.getLoc(), TypeRange{asmResultType}, ValueRange{}, asmString,
        appendSimtKeepResumeClobbers(
            buildRepeatedInlineAsmConstraints("=R", resumeOps.size()), clobbers),
        true, false,
        LLVM::AsmDialectAttr::get(op.getContext(), LLVM::AsmDialect::AD_ATT),
        ArrayAttr{});

    if (resumeOps.size() == 1) {
      Type resultType = getTypeConverter()->convertType(op.getType());
      Value result = unpackSimtKeepResumePayload(op.getLoc(), asmOp.getRes(),
                                                 resultType, rewriter);
      if (!result)
        return rewriter.notifyMatchFailure(op, "failed to unpack result");
      rewriter.replaceOp(op, result);
      return success();
    }

    rewriter.setInsertionPointAfter(asmOp);
    SmallVector<Value, 4> results;
    for (auto [index, resume] : llvm::enumerate(resumeOps)) {
      auto extract = rewriter.create<LLVM::ExtractValueOp>(
          resume.getLoc(), asmOp.getRes(),
          ArrayRef<int64_t>{static_cast<int64_t>(index)});
      Type resultType = getTypeConverter()->convertType(resume.getType());
      Value result = unpackSimtKeepResumePayload(resume.getLoc(),
                                                 extract.getRes(), resultType,
                                                 rewriter);
      if (!result)
        return rewriter.notifyMatchFailure(resume, "failed to unpack result");
      results.push_back(result);
    }
    for (auto [resume, result] : llvm::zip(resumeOps, results))
      rewriter.replaceOp(resume, result);
    return success();
  }

private:
  LoweringState &state;
};

template <typename ConfigOp>
class LowerNullaryConfigOpPattern final : public OpConversionPattern<ConfigOp> {
public:
  explicit LowerNullaryConfigOpPattern(TypeConverter &typeConverter,
                                       MLIRContext *context,
                                       LoweringState &state)
      : OpConversionPattern<ConfigOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(ConfigOp op, typename ConfigOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    StringRef calleeName = buildNullaryConfigCallee<ConfigOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename SyncOp>
class LowerPipeEventSyncOpPattern final : public OpConversionPattern<SyncOp> {
public:
  explicit LowerPipeEventSyncOpPattern(TypeConverter &typeConverter,
                                       MLIRContext *context,
                                       LoweringState &state)
      : OpConversionPattern<SyncOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(SyncOp op, typename SyncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto src = parsePipeImmediate(stringifyPIPE(op.getSrcPipe().getPipe()));
    auto dst = parsePipeImmediate(stringifyPIPE(op.getDstPipe().getPipe()));
    auto event = parseEventImmediate(stringifyEVENT(op.getEventId().getEvent()));
    if (!src || !dst || !event)
      return rewriter.notifyMatchFailure(op, "unsupported sync immediate");

    StringRef calleeName = buildSyncCallee<SyncOp>(op.getContext());
    Value srcValue = getI64Constant(rewriter, op.getLoc(), *src);
    Value dstValue = getI64Constant(rewriter, op.getLoc(), *dst);
    Value eventValue = getI64Constant(rewriter, op.getLoc(), *event);
    auto funcType = rewriter.getFunctionType(
        TypeRange{rewriter.getI64Type(), rewriter.getI64Type(),
                  rewriter.getI64Type()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{srcValue, dstValue, eventValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename SyncOp>
class LowerPipeEventDynSyncOpPattern final : public OpConversionPattern<SyncOp> {
public:
  explicit LowerPipeEventDynSyncOpPattern(TypeConverter &typeConverter,
                                          MLIRContext *context,
                                          LoweringState &state)
      : OpConversionPattern<SyncOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(SyncOp op, typename SyncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto src = parsePipeImmediate(stringifyPIPE(op.getSrcPipe().getPipe()));
    auto dst = parsePipeImmediate(stringifyPIPE(op.getDstPipe().getPipe()));
    if (!src || !dst)
      return rewriter.notifyMatchFailure(op, "unsupported sync pipe");

    StringRef calleeName = buildSyncCallee<SyncOp>(op.getContext());
    Value srcValue = getI64Constant(rewriter, op.getLoc(), *src);
    Value dstValue = getI64Constant(rewriter, op.getLoc(), *dst);
    
    Value eventIdValue = adaptor.getEventId();
    if (!eventIdValue)
      return rewriter.notifyMatchFailure(op, "missing event_id operand");
    
    Value eventValue = eventIdValue;
    
    while (eventValue.getDefiningOp()) {
      auto unrealizedCast = dyn_cast<UnrealizedConversionCastOp>(eventValue.getDefiningOp());
      if (!unrealizedCast || unrealizedCast.getInputs().size() != 1)
        break;
      eventValue = unrealizedCast.getInputs()[0];
    }
    
    if (eventValue.getType().isIndex()) {
      eventValue = rewriter.create<arith::IndexCastOp>(op.getLoc(), 
                                                        rewriter.getI64Type(), 
                                                        eventValue);
    } else if (auto intType = dyn_cast<IntegerType>(eventValue.getType())) {
      if (intType.getWidth() < 64) {
        eventValue = rewriter.create<LLVM::ZExtOp>(op.getLoc(), 
                                                    rewriter.getI64Type(), 
                                                    eventValue);
      }
    } else {
      return rewriter.notifyMatchFailure(op, "unexpected event_id type");
    }
    
    auto funcType = rewriter.getFunctionType(
        TypeRange{rewriter.getI64Type(), rewriter.getI64Type(),
                  rewriter.getI64Type()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{srcValue, dstValue, eventValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename SyncOp>
class LowerInterCoreSyncOpPattern final : public OpConversionPattern<SyncOp> {
public:
  explicit LowerInterCoreSyncOpPattern(TypeConverter &typeConverter,
                                       MLIRContext *context,
                                       LoweringState &state)
      : OpConversionPattern<SyncOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(SyncOp op, typename SyncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto pipe = parsePipeImmediate(stringifyPIPE(op.getPipe().getPipe()));
    if (!pipe)
      return rewriter.notifyMatchFailure(op, "unsupported inter-core sync pipe");

    Value pipeValue = getI64Constant(rewriter, op.getLoc(), *pipe);
    Value eventValue;
    if (IntegerAttr eventIdAttr = op.getEventIdAttr()) {
      eventValue = getI64Constant(rewriter, op.getLoc(), eventIdAttr.getInt());
    } else {
      Value eventIdDyn = adaptor.getEventIdDyn();
      if (!eventIdDyn)
        return rewriter.notifyMatchFailure(
            op, "expected static or dynamic event-id operand");

      eventValue = castIntegerLikeTo(op, eventIdDyn, rewriter.getI64Type());
      if (!eventValue) {
        return rewriter.notifyMatchFailure(
            op, "failed to cast dynamic event-id to i64");
      }
    }

    StringRef calleeName = buildSyncCallee<SyncOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(
        TypeRange{rewriter.getI64Type(), rewriter.getI64Type()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{pipeValue, eventValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerBarrierOpPattern final : public OpConversionPattern<pto::BarrierOp> {
public:
  explicit LowerBarrierOpPattern(TypeConverter &typeConverter,
                                 MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::BarrierOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto pipe = parsePipeImmediate(stringifyPIPE(op.getPipe().getPipe()));
    if (!pipe)
      return rewriter.notifyMatchFailure(op, "unsupported barrier pipe");

    StringRef calleeName = buildSyncCallee<pto::BarrierOp>(op.getContext());
    Value pipeValue = getI64Constant(rewriter, op.getLoc(), *pipe);
    auto funcType =
        rewriter.getFunctionType(TypeRange{rewriter.getI64Type()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{pipeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerMemBarOpPattern final : public OpConversionPattern<pto::MemBarOp> {
public:
  explicit LowerMemBarOpPattern(TypeConverter &typeConverter,
                                MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::MemBarOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::MemBarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    StringRef calleeName = buildMemBarCallee(op.getKind().getKind(), op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename BufSyncOp>
class LowerBufSyncOpPattern final : public OpConversionPattern<BufSyncOp> {
public:
  explicit LowerBufSyncOpPattern(TypeConverter &typeConverter,
                                 MLIRContext *context, LoweringState &state)
      : OpConversionPattern<BufSyncOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(BufSyncOp op, typename BufSyncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    PIPE pipe = PIPE::PIPE_UNASSIGNED;
    if (auto pipeAttr = dyn_cast<PipeAttr>(op.getOpTypeAttr())) {
      pipe = pipeAttr.getPipe();
    } else {
      auto opTypeOr = parseSyncOpTypeLikeAttr(op.getOpTypeAttr());
      if (failed(opTypeOr))
        return rewriter.notifyMatchFailure(
            op, "buffer sync expects pipe/sync_op_type/pipe_event_type attr");
      pipe = mapSyncOpTypeToPipe(*opTypeOr);
    }
    if (!isConcreteSyncPipe(pipe))
      return rewriter.notifyMatchFailure(op,
                                         "buffer sync op_type cannot map to concrete pipe");

    auto pipeImm = parsePipeImmediate(stringifyPIPE(pipe));
    if (!pipeImm)
      return rewriter.notifyMatchFailure(op, "unsupported buffer sync pipe");

    StringRef calleeName = buildSyncCallee<BufSyncOp>(op.getContext());
    Value pipeValue = getI64Constant(rewriter, op.getLoc(), *pipeImm);
    Value bufIdValue =
        getI64Constant(rewriter, op.getLoc(), op.getBufIdAttr().getInt());
    Value modeValue =
        getI64Constant(rewriter, op.getLoc(), op.getModeAttr().getInt());
    auto funcType = rewriter.getFunctionType(
        TypeRange{rewriter.getI64Type(), rewriter.getI64Type(),
                  rewriter.getI64Type()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{pipeValue, bufIdValue, modeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename QueryOp>
class LowerRuntimeQueryOpPattern final : public OpConversionPattern<QueryOp> {
public:
  explicit LowerRuntimeQueryOpPattern(TypeConverter &typeConverter,
                                      MLIRContext *context,
                                      LoweringState &state)
      : OpConversionPattern<QueryOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(QueryOp op, typename QueryOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert runtime-query result type");

    StringRef calleeName = buildRuntimeQueryCallee<QueryOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              TypeRange{resultType}, ValueRange{});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename VoteOp>
class LowerVoteOpPattern final : public OpConversionPattern<VoteOp> {
public:
  explicit LowerVoteOpPattern(TypeConverter &typeConverter,
                              MLIRContext *context, LoweringState &state)
      : OpConversionPattern<VoteOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(VoteOp op, typename VoteOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert vote result type");

    Type predType = this->getTypeConverter()->convertType(op.getPred().getType());
    if (!predType || predType != rewriter.getI1Type())
      return rewriter.notifyMatchFailure(op, "failed to convert vote predicate type");

    StringRef calleeName = buildVoteCallee<VoteOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{predType}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              TypeRange{resultType},
                                              ValueRange{adaptor.getPred()});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename ShuffleOp>
class LowerShuffleOpPattern final : public OpConversionPattern<ShuffleOp> {
public:
  explicit LowerShuffleOpPattern(TypeConverter &typeConverter,
                                 MLIRContext *context, LoweringState &state)
      : OpConversionPattern<ShuffleOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(ShuffleOp op, typename ShuffleOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert shuffle result type");

    Type valueType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!valueType || valueType != resultType)
      return rewriter.notifyMatchFailure(op, "unexpected converted shuffle operand type");

    FailureOr<StringRef> calleeName =
        buildShuffleCallee<ShuffleOp>(op.getContext(), op.getValue().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported shuffle VPTO signature");

    IntegerAttr widthAttr = op.getWidthAttr();
    Value controlValue;
    unsigned controlMask = 0;
    if constexpr (std::is_same_v<ShuffleOp, pto::ShuffleIdxOp>) {
      controlValue = adaptor.getIndex();
      controlMask = 0x1f;
    } else if constexpr (std::is_same_v<ShuffleOp, pto::ShuffleUpOp>) {
      controlValue = adaptor.getOffset();
      controlMask = 0;
    } else if constexpr (std::is_same_v<ShuffleOp, pto::ShuffleDownOp>) {
      controlValue = adaptor.getOffset();
      controlMask = 0x1f;
    } else if constexpr (std::is_same_v<ShuffleOp, pto::ShuffleBflyOp>) {
      controlValue = adaptor.getMask();
      controlMask = 0x1f;
    }
    if (!controlValue)
      return rewriter.notifyMatchFailure(op, "missing shuffle control operand");

    Value control = buildShuffleControlValue(
        rewriter, op.getLoc(), controlValue, widthAttr.getInt(), controlMask);

    Type i32Type = rewriter.getI32Type();
    auto funcType = rewriter.getFunctionType(TypeRange{resultType, i32Type},
                                             TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getValue(), control});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename ReduxOp>
class LowerReduxOpPattern final : public OpConversionPattern<ReduxOp> {
public:
  explicit LowerReduxOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<ReduxOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(ReduxOp op, typename ReduxOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert redux result type");

    Type valueType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!valueType || valueType != resultType)
      return rewriter.notifyMatchFailure(op, "unexpected converted redux operand type");

    FailureOr<StringRef> calleeName = buildReduxCallee<ReduxOp>(
        op.getContext(), op.getValue().getType(), op.getSignednessAttr());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported redux VPTO signature");

    auto funcType = rewriter.getFunctionType(TypeRange{resultType},
                                             TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{resultType},
                                              ValueRange{adaptor.getValue()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename AtomicOp>
class LowerAtomicBinaryOpPattern final : public OpConversionPattern<AtomicOp> {
public:
  explicit LowerAtomicBinaryOpPattern(TypeConverter &typeConverter,
                                      MLIRContext *context,
                                      LoweringState &state)
      : OpConversionPattern<AtomicOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(AtomicOp op, typename AtomicOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getOld().getType());
    Type valueType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!resultType || !valueType || resultType != valueType)
      return rewriter.notifyMatchFailure(op,
                                         "unexpected atomic operand/result type");

    Type ptrType = this->getTypeConverter()->convertType(op.getPtr().getType());
    if (!ptrType)
      return rewriter.notifyMatchFailure(op, "failed to convert atomic pointer type");

    FailureOr<StringRef> calleeName = buildAtomicCallee<AtomicOp>(
        op.getContext(), op.getPtr().getType(), op.getValue().getType(),
        op.getSignednessAttr());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported atomic VPTO signature");

    auto funcType = rewriter.getFunctionType(
        TypeRange{ptrType, valueType, rewriter.getI32Type()},
        TypeRange{resultType});
    Value modeValue = getI32Constant(
        rewriter, op.getLoc(),
        static_cast<uint64_t>(op.getL2cacheAttr()
                                  ? op.getL2cacheAttr().getValue()
                                  : pto::StL2Cache::NMFV));
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getPtr(), adaptor.getValue(), modeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerAtomicCasOpPattern final
    : public OpConversionPattern<pto::AtomicCasOp> {
public:
  explicit LowerAtomicCasOpPattern(TypeConverter &typeConverter,
                                   MLIRContext *context,
                                   LoweringState &state)
      : OpConversionPattern<pto::AtomicCasOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::AtomicCasOp op, pto::AtomicCasOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getOld().getType());
    Type compareType =
        this->getTypeConverter()->convertType(op.getCompare().getType());
    Type valueType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!resultType || !compareType || !valueType || resultType != compareType ||
        resultType != valueType)
      return rewriter.notifyMatchFailure(op, "unexpected atomic CAS type");

    Type ptrType = this->getTypeConverter()->convertType(op.getPtr().getType());
    if (!ptrType)
      return rewriter.notifyMatchFailure(op, "failed to convert atomic pointer type");

    FailureOr<StringRef> calleeName = buildAtomicCallee<pto::AtomicCasOp>(
        op.getContext(), op.getPtr().getType(), op.getValue().getType(),
        op.getSignednessAttr());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported atomic CAS signature");

    auto funcType = rewriter.getFunctionType(
        TypeRange{ptrType, compareType, valueType, rewriter.getI32Type()},
        TypeRange{resultType});
    Value modeValue = getI32Constant(
        rewriter, op.getLoc(),
        static_cast<uint64_t>(op.getL2cacheAttr()
                                  ? op.getL2cacheAttr().getValue()
                                  : pto::StL2Cache::NMFV));
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getPtr(), adaptor.getCompare(), adaptor.getValue(),
                   modeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename ScalarOp>
class LowerScalarIntrinsicOpPattern final : public OpConversionPattern<ScalarOp> {
public:
  explicit LowerScalarIntrinsicOpPattern(TypeConverter &typeConverter,
                                         MLIRContext *context,
                                         LoweringState &state)
      : OpConversionPattern<ScalarOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(ScalarOp op, typename ScalarOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes)))
      return rewriter.notifyMatchFailure(op, "failed to convert scalar result types");

    SmallVector<Type> operandTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getOperandTypes(),
                                                      operandTypes)))
      return rewriter.notifyMatchFailure(op, "failed to convert scalar operand types");

    StringRef calleeName = buildScalarIntrinsicCallee<ScalarOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(operandTypes, resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              resultTypes, adaptor.getOperands());
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerMulhiOpPattern final : public OpConversionPattern<pto::MulhiOp> {
public:
  explicit LowerMulhiOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::MulhiOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::MulhiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    Type lhsType = getTypeConverter()->convertType(op.getLhs().getType());
    Type rhsType = getTypeConverter()->convertType(op.getRhs().getType());
    if (!resultType || !lhsType || !rhsType || lhsType != resultType ||
        rhsType != resultType)
      return rewriter.notifyMatchFailure(op, "unexpected mulhi type");

    pto::Signedness signedness = op.getSignednessAttr().getValue();
    FailureOr<StringRef> calleeName =
        buildMulhiCallee(op.getContext(), op.getResult().getType(), signedness);
    if (succeeded(calleeName)) {
      auto funcType =
          rewriter.getFunctionType(TypeRange{lhsType, rhsType}, TypeRange{resultType});
      auto call = rewriter.create<func::CallOp>(
          op.getLoc(), *calleeName, TypeRange{resultType},
          ValueRange{adaptor.getLhs(), adaptor.getRhs()});
      state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
      rewriter.replaceOp(op, call.getResults());
      return success();
    }

    if (!op.getResult().getType().isInteger(64) ||
        signedness != pto::Signedness::Signed)
      return rewriter.notifyMatchFailure(op, "unsupported mulhi signature");

    FailureOr<StringRef> unsignedCalleeName =
        buildMulhiCallee(op.getContext(), op.getResult().getType(),
                         pto::Signedness::Unsigned);
    if (failed(unsignedCalleeName))
      return rewriter.notifyMatchFailure(op, "unsupported mul64hi signature");

    auto funcType =
        rewriter.getFunctionType(TypeRange{lhsType, rhsType}, TypeRange{resultType});
    auto unsignedCall = rewriter.create<func::CallOp>(
        op.getLoc(), *unsignedCalleeName, TypeRange{resultType},
        ValueRange{adaptor.getLhs(), adaptor.getRhs()});
    state.plannedDecls.push_back(PlannedDecl{unsignedCalleeName->str(), funcType});

    Value zero = getI64Constant(rewriter, op.getLoc(), 0);
    Value lhsNeg = rewriter.create<LLVM::ICmpOp>(
        op.getLoc(), LLVM::ICmpPredicate::slt, adaptor.getLhs(), zero);
    Value rhsNeg = rewriter.create<LLVM::ICmpOp>(
        op.getLoc(), LLVM::ICmpPredicate::slt, adaptor.getRhs(), zero);
    Value subRhs = rewriter.create<LLVM::SubOp>(
        op.getLoc(), unsignedCall.getResult(0), adaptor.getRhs());
    Value correctedLhs = rewriter.create<LLVM::SelectOp>(
        op.getLoc(), resultType, lhsNeg, subRhs, unsignedCall.getResult(0));
    Value subLhs = rewriter.create<LLVM::SubOp>(
        op.getLoc(), correctedLhs, adaptor.getLhs());
    Value corrected = rewriter.create<LLVM::SelectOp>(
        op.getLoc(), resultType, rhsNeg, subLhs, correctedLhs);
    rewriter.replaceOp(op, corrected);
    return success();
  }

private:
  LoweringState &state;
};

class LowerMulI32ToI64OpPattern final
    : public OpConversionPattern<pto::MulI32ToI64Op> {
public:
  explicit LowerMulI32ToI64OpPattern(TypeConverter &typeConverter,
                                     MLIRContext *context,
                                     LoweringState &state)
      : OpConversionPattern<pto::MulI32ToI64Op>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::MulI32ToI64Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    Type lhsType = getTypeConverter()->convertType(op.getLhs().getType());
    Type rhsType = getTypeConverter()->convertType(op.getRhs().getType());
    if (!resultType || !lhsType || !rhsType)
      return rewriter.notifyMatchFailure(op, "unexpected mul_i32toi64 type");

    FailureOr<StringRef> calleeName =
        buildMulI32ToI64Callee(op.getContext(),
                               op.getSignednessAttr().getValue());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op,
                                         "unsupported mul_i32toi64 signature");

    auto funcType =
        rewriter.getFunctionType(TypeRange{lhsType, rhsType}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getLhs(), adaptor.getRhs()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerSqrtOpPattern final : public OpConversionPattern<pto::SqrtOp> {
public:
  explicit LowerSqrtOpPattern(TypeConverter &typeConverter,
                              MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::SqrtOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::SqrtOp op, pto::SqrtOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type valueType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!resultType || !valueType || valueType != resultType)
      return rewriter.notifyMatchFailure(op, "unexpected sqrt operand/result type");

    FailureOr<StringRef> calleeName =
        buildSqrtCallee(op.getContext(), op.getValue().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported sqrt VPTO signature");

    auto funcType = rewriter.getFunctionType(TypeRange{valueType},
                                             TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{resultType},
                                              ValueRange{adaptor.getValue()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename UnaryOp>
class LowerUnaryScalarMathOpPattern final : public OpConversionPattern<UnaryOp> {
public:
  explicit LowerUnaryScalarMathOpPattern(TypeConverter &typeConverter,
                                         MLIRContext *context,
                                         LoweringState &state)
      : OpConversionPattern<UnaryOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(UnaryOp op, typename UnaryOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type valueType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!resultType || !valueType || valueType != resultType)
      return rewriter.notifyMatchFailure(op, "unexpected unary scalar math type");

    FailureOr<StringRef> calleeName =
        buildUnaryScalarMathCallee<UnaryOp>(op.getContext(), op.getValue().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported unary scalar math signature");

    auto funcType = rewriter.getFunctionType(TypeRange{valueType},
                                             TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{resultType},
                                              ValueRange{adaptor.getValue()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename BinaryOp>
class LowerBinaryScalarMathOpPattern final : public OpConversionPattern<BinaryOp> {
public:
  explicit LowerBinaryScalarMathOpPattern(TypeConverter &typeConverter,
                                          MLIRContext *context,
                                          LoweringState &state)
      : OpConversionPattern<BinaryOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(BinaryOp op, typename BinaryOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type lhsType = this->getTypeConverter()->convertType(op.getLhs().getType());
    Type rhsType = this->getTypeConverter()->convertType(op.getRhs().getType());
    if (!resultType || !lhsType || !rhsType ||
        lhsType != rhsType || lhsType != resultType)
      return rewriter.notifyMatchFailure(op, "unexpected binary scalar math type");

    FailureOr<StringRef> calleeName =
        buildBinaryScalarMathCallee<BinaryOp>(op.getContext(), op.getLhs().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported binary scalar math signature");

    auto funcType = rewriter.getFunctionType(TypeRange{lhsType, rhsType},
                                             TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getLhs(), adaptor.getRhs()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerFmaOpPattern final : public OpConversionPattern<pto::FmaOp> {
public:
  explicit LowerFmaOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                             LoweringState &state)
      : OpConversionPattern<pto::FmaOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::FmaOp op, pto::FmaOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type lhsType = this->getTypeConverter()->convertType(op.getLhs().getType());
    Type rhsType = this->getTypeConverter()->convertType(op.getRhs().getType());
    Type accType = this->getTypeConverter()->convertType(op.getAcc().getType());
    if (!resultType || !lhsType || !rhsType || !accType ||
        lhsType != rhsType || lhsType != accType || lhsType != resultType)
      return rewriter.notifyMatchFailure(op, "unexpected fma scalar math type");

    FailureOr<StringRef> calleeName = buildFmaCallee(op.getContext(),
                                                     op.getLhs().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported fma scalar signature");

    auto funcType = rewriter.getFunctionType(TypeRange{lhsType, rhsType, accType},
                                             TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getLhs(), adaptor.getRhs(), adaptor.getAcc()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerConvertOpPattern final : public OpConversionPattern<pto::ConvertOp> {
public:
  explicit LowerConvertOpPattern(TypeConverter &typeConverter,
                                 MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::ConvertOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::ConvertOp op, pto::ConvertOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getDst().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert result type");

    FailureOr<StringRef> calleeName =
        buildConvertCallee(op.getContext(), op.getSrc().getType(),
                           op.getDst().getType(), op.getSignednessAttr());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported convert signature");

    Value rounding = getI32Constant(
        rewriter, op.getLoc(), static_cast<uint64_t>(op.getRounding()));
    Value saturation = getI32Constant(
        rewriter, op.getLoc(), static_cast<uint64_t>(op.getSaturation()));

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSrc().getType(), rewriter.getI32Type(),
                  rewriter.getI32Type()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getSrc(), rounding, saturation});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerGetVms4SrOpPattern final
    : public OpConversionPattern<pto::GetVms4SrOp> {
public:
  explicit LowerGetVms4SrOpPattern(TypeConverter &typeConverter,
                                   MLIRContext *context,
                                   LoweringState &state)
      : OpConversionPattern<pto::GetVms4SrOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::GetVms4SrOp op, pto::GetVms4SrOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes)) ||
        resultTypes.size() != 4)
      return rewriter.notifyMatchFailure(
          op, "failed to convert get_vms4_sr result types");

    StringRef calleeName = buildRuntimeQueryCallee<pto::GetVms4SrOp>(
        op.getContext());
    auto funcType =
        rewriter.getFunctionType(TypeRange{}, TypeRange{rewriter.getI64Type()});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), calleeName, TypeRange{rewriter.getI64Type()},
        ValueRange{});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});

    SmallVector<Value> counts;
    counts.reserve(4);
    Value raw = call.getResult(0);
    for (unsigned i = 0; i < 4; ++i) {
      Value shifted = raw;
      if (i != 0)
        shifted = rewriter.create<arith::ShRUIOp>(
            op.getLoc(), raw, getI64Constant(rewriter, op.getLoc(), i * 16));
      counts.push_back(rewriter.create<arith::TruncIOp>(
          op.getLoc(), resultTypes[i], shifted));
    }
    rewriter.replaceOp(op, counts);
    return success();
  }

private:
  LoweringState &state;
};

template <typename BinaryOp>
class LowerBinaryI64PureOpPattern final : public OpConversionPattern<BinaryOp> {
public:
  explicit LowerBinaryI64PureOpPattern(TypeConverter &typeConverter,
                                       MLIRContext *context,
                                       LoweringState &state)
      : OpConversionPattern<BinaryOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(BinaryOp op, typename BinaryOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    StringRef calleeName = buildBinaryI64PureCallee<BinaryOp>(op.getContext());
    auto funcType =
        rewriter.getFunctionType(TypeRange{adaptor.getFirst().getType(),
                                           adaptor.getSecond().getType()},
                                 TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), calleeName, TypeRange{resultType},
        ValueRange{adaptor.getFirst(), adaptor.getSecond()});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class ConvertVPTOUnrealizedCastOp final
    : public OpConversionPattern<UnrealizedConversionCastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();
    if (!hasVPTOConvertibleType(op->getOperandTypes()) &&
        !hasVPTOConvertibleType(op->getResultTypes()))
      return failure();

    Type convertedResultType =
        getTypeConverter()->convertType(op.getResult(0).getType());
    if (!convertedResultType)
      return failure();

    Value input = adaptor.getOperands().front();
    if (input.getType() != convertedResultType)
      return failure();

    rewriter.replaceOp(op, input);
    return success();
  }
};

class ConvertPtoAddPtrOp final : public OpConversionPattern<pto::AddPtrOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedResultType = getTypeConverter()->convertType(op.getResult().getType());
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(convertedResultType);
    if (!llvmPtrType)
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer result type");

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex())
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(),
                                                     rewriter.getI64Type(), offset);

    auto gep = rewriter.create<LLVM::GEPOp>(
        op.getLoc(), llvmPtrType, cast<pto::PtrType>(op.getPtr().getType()).getElementType(),
        adaptor.getPtr(), ValueRange{offset});
    rewriter.replaceOp(op, gep.getResult());
    return success();
  }
};

class ConvertPtoCastPtrOp final : public OpConversionPattern<pto::CastPtrOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::CastPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedResultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedResultType)
      return rewriter.notifyMatchFailure(op,
                                         "could not convert castptr result type");

    Value input = adaptor.getInput();
    Type inputType = input.getType();
    if (inputType == convertedResultType) {
      rewriter.replaceOp(op, input);
      return success();
    }

    if (auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(convertedResultType)) {
      if (isa<IntegerType>(inputType)) {
        rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, llvmPtrType, input);
        return success();
      }
      auto sourcePtrType = dyn_cast<LLVM::LLVMPointerType>(inputType);
      if (!sourcePtrType)
        return rewriter.notifyMatchFailure(op,
                                           "expected integer or LLVM pointer input");
      if (sourcePtrType.getAddressSpace() == llvmPtrType.getAddressSpace()) {
        rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, llvmPtrType, input);
        return success();
      }
      return rewriter.notifyMatchFailure(
          op, "cross-address-space ptr casts are unsupported");
    }

    if (auto resultIntType = dyn_cast<IntegerType>(convertedResultType)) {
      if (isa<LLVM::LLVMPointerType>(inputType)) {
        rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, resultIntType, input);
        return success();
      }
    }

    return rewriter.notifyMatchFailure(op, "unsupported castptr conversion");
  }
};

class ConvertPtoLoadScalarOp final
    : public OpConversionPattern<pto::LoadScalarOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::LoadScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!llvmPtrType)
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");

    Type convertedValueType =
        getTypeConverter()->convertType(op.getValue().getType());
    if (!convertedValueType)
      return rewriter.notifyMatchFailure(op,
                                         "could not convert load_scalar result type");

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex())
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(),
                                                     rewriter.getI64Type(), offset);

    Value elemPtr = adaptor.getPtr();
    if (!matchPattern(offset, m_Zero())) {
      elemPtr = rewriter.create<LLVM::GEPOp>(op.getLoc(), llvmPtrType,
                                             convertedValueType, adaptor.getPtr(),
                                             ValueRange{offset});
    }

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        op, convertedValueType, elemPtr,
        getNaturalByteAlignment(convertedValueType));
    return success();
  }
};

class ConvertPtoStoreScalarOp final
    : public OpConversionPattern<pto::StoreScalarOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::StoreScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!llvmPtrType)
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex())
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(),
                                                     rewriter.getI64Type(), offset);

    Value elemPtr = adaptor.getPtr();
    if (!matchPattern(offset, m_Zero())) {
      elemPtr = rewriter.create<LLVM::GEPOp>(op.getLoc(), llvmPtrType,
                                             adaptor.getValue().getType(),
                                             adaptor.getPtr(), ValueRange{offset});
    }

    rewriter.create<LLVM::StoreOp>(op.getLoc(), adaptor.getValue(), elemPtr,
                                   getNaturalByteAlignment(adaptor.getValue().getType()));
    rewriter.eraseOp(op);
    return success();
  }
};

class ConvertPtoLoadOp final : public OpConversionPattern<pto::PTOLoadOp> {
public:
  ConvertPtoLoadOp(TypeConverter &typeConverter, MLIRContext *context,
                   LoweringState &state)
      : OpConversionPattern<pto::PTOLoadOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::PTOLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!llvmPtrType)
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");

    Type convertedValueType =
        getTypeConverter()->convertType(op.getValue().getType());
    if (!convertedValueType)
      return rewriter.notifyMatchFailure(op, "could not convert load result type");

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex())
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(),
                                                     rewriter.getI64Type(), offset);

    Value elemPtr = adaptor.getPtr();
    if (!matchPattern(offset, m_Zero())) {
      elemPtr = rewriter.create<LLVM::GEPOp>(op.getLoc(), llvmPtrType,
                                             convertedValueType, adaptor.getPtr(),
                                             ValueRange{offset});
    }

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        op, convertedValueType, elemPtr,
        getNaturalByteAlignment(convertedValueType));
    return success();
  }

private:
  LoweringState &state;
};

static Type getLdgCallResultType(Type valueType, Type convertedValueType,
                                 ConversionPatternRewriter &rewriter) {
  if (auto intType = dyn_cast<IntegerType>(valueType)) {
    unsigned width = intType.getWidth();
    if (width == 8 || width == 16)
      return rewriter.getI32Type();
    return convertedValueType;
  }
  if (valueType.isF16() || valueType.isBF16() || valueType.isF32())
    return rewriter.getI32Type();
  if (valueType.isF64())
    return rewriter.getI64Type();
  if (pto::isPTOFloat8Type(valueType) || pto::isPTOHiFloat8Type(valueType))
    return rewriter.getI32Type();
  return convertedValueType;
}

static Value convertLdgCallResult(Location loc, Type valueType,
                                  Type convertedValueType, Value callResult,
                                  ConversionPatternRewriter &rewriter) {
  if (auto intType = dyn_cast<IntegerType>(valueType)) {
    unsigned width = intType.getWidth();
    if (width == 8 || width == 16)
      return rewriter.create<arith::TruncIOp>(
          loc, rewriter.getIntegerType(width), callResult);
    return callResult;
  }

  if (valueType.isF16() || valueType.isBF16()) {
    Value payload =
        rewriter.create<arith::TruncIOp>(loc, rewriter.getI16Type(), callResult);
    return rewriter.create<LLVM::BitcastOp>(loc, convertedValueType, payload);
  }
  if (valueType.isF32() || valueType.isF64())
    return rewriter.create<LLVM::BitcastOp>(loc, convertedValueType,
                                            callResult);
  if (pto::isPTOFloat8Type(valueType) || pto::isPTOHiFloat8Type(valueType))
    return rewriter.create<arith::TruncIOp>(loc, rewriter.getI8Type(),
                                            callResult);
  return callResult;
}

class ConvertPtoLdgOp final : public OpConversionPattern<pto::PTOLdgOp> {
public:
  ConvertPtoLdgOp(TypeConverter &typeConverter, MLIRContext *context,
                  LoweringState &state)
      : OpConversionPattern<pto::PTOLdgOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::PTOLdgOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!llvmPtrType)
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");

    Type convertedValueType =
        getTypeConverter()->convertType(op.getValue().getType());
    if (!convertedValueType)
      return rewriter.notifyMatchFailure(op, "could not convert ldg result type");

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex())
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(),
                                                     rewriter.getI64Type(), offset);

    Value elemPtr = adaptor.getPtr();
    if (!matchPattern(offset, m_Zero())) {
      elemPtr = rewriter.create<LLVM::GEPOp>(op.getLoc(), llvmPtrType,
                                             convertedValueType, adaptor.getPtr(),
                                             ValueRange{offset});
    }

    auto ptrTy = cast<pto::PtrType>(op.getPtr().getType());
    FailureOr<Value> ptr = reinterpretPointerToAddrSpace(
        op, elemPtr,
        static_cast<unsigned>(ptrTy.getMemorySpace().getAddressSpace()));
    if (failed(ptr))
      return rewriter.notifyMatchFailure(op, "failed to map ldg pointer");

    pto::L1Cache l1cache = op.getL1cacheAttr()
                               ? op.getL1cacheAttr().getValue()
                               : pto::L1Cache::Cache;
    FailureOr<StringRef> calleeName = buildL1CacheLoadCallee(
        op.getContext(), op.getValue().getType(), l1cache);
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported ldg signature");

    pto::LdL2Cache mode = op.getL2cacheAttr()
                               ? op.getL2cacheAttr().getValue()
                               : pto::LdL2Cache::NMFV;
    Value modeValue =
        getI32Constant(rewriter, op.getLoc(), static_cast<uint64_t>(mode));
    Type callResultType = getLdgCallResultType(op.getValue().getType(),
                                               convertedValueType, rewriter);
    auto funcType =
        rewriter.getFunctionType(TypeRange{ptr->getType(), rewriter.getI32Type()},
                                 TypeRange{callResultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{callResultType},
        ValueRange{*ptr, modeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    Value result = convertLdgCallResult(op.getLoc(), op.getValue().getType(),
                                        convertedValueType, call.getResult(0),
                                        rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  LoweringState &state;
};

class ConvertPtoStoreOp final : public OpConversionPattern<pto::PTOStoreOp> {
public:
  ConvertPtoStoreOp(TypeConverter &typeConverter, MLIRContext *context,
                    LoweringState &state)
      : OpConversionPattern<pto::PTOStoreOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::PTOStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!llvmPtrType)
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex())
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(),
                                                     rewriter.getI64Type(), offset);

    Value elemPtr = adaptor.getPtr();
    if (!matchPattern(offset, m_Zero())) {
      elemPtr = rewriter.create<LLVM::GEPOp>(op.getLoc(), llvmPtrType,
                                             adaptor.getValue().getType(),
                                             adaptor.getPtr(), ValueRange{offset});
    }

    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(
        op, adaptor.getValue(), elemPtr,
        getNaturalByteAlignment(adaptor.getValue().getType()));
    return success();
  }

private:
  LoweringState &state;
};

static Value convertStgValue(Location loc, Type valueType, Value value,
                             ConversionPatternRewriter &rewriter) {
  if (auto intType = dyn_cast<IntegerType>(valueType)) {
    unsigned width = intType.getWidth();
    if (width == 8)
      return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), value);
    if (width == 16)
      return rewriter.create<LLVM::BitcastOp>(loc, rewriter.getF16Type(), value);
    return value;
  }

  if (pto::isPTOFloat8Type(valueType) || pto::isPTOHiFloat8Type(valueType))
    return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), value);
  if (valueType.isBF16())
    return rewriter.create<LLVM::BitcastOp>(loc, rewriter.getF16Type(), value);
  if (valueType.isF32())
    return rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
  if (valueType.isF64())
    return rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
  return value;
}

class ConvertPtoStgOp final : public OpConversionPattern<pto::PTOStgOp> {
public:
  ConvertPtoStgOp(TypeConverter &typeConverter, MLIRContext *context,
                  LoweringState &state)
      : OpConversionPattern<pto::PTOStgOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::PTOStgOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!llvmPtrType)
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex())
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(),
                                                     rewriter.getI64Type(), offset);

    Value elemPtr = adaptor.getPtr();
    if (!matchPattern(offset, m_Zero())) {
      elemPtr = rewriter.create<LLVM::GEPOp>(op.getLoc(), llvmPtrType,
                                             adaptor.getValue().getType(),
                                             adaptor.getPtr(), ValueRange{offset});
    }

    auto ptrTy = cast<pto::PtrType>(op.getPtr().getType());
    FailureOr<Value> ptr = reinterpretPointerToAddrSpace(
        op, elemPtr,
        static_cast<unsigned>(ptrTy.getMemorySpace().getAddressSpace()));
    if (failed(ptr))
      return rewriter.notifyMatchFailure(op, "failed to map stg pointer");

    pto::L1Cache l1cache = op.getL1cacheAttr()
                               ? op.getL1cacheAttr().getValue()
                               : pto::L1Cache::Cache;
    FailureOr<StringRef> calleeName = buildL1CacheStoreCallee(
        op.getContext(), op.getValue().getType(), l1cache);
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported stg signature");

    pto::StL2Cache mode = op.getL2cacheAttr()
                               ? op.getL2cacheAttr().getValue()
                               : pto::StL2Cache::NMFV;
    Value modeValue =
        getI32Constant(rewriter, op.getLoc(), static_cast<uint64_t>(mode));
    Value storedValue = convertStgValue(op.getLoc(), op.getValue().getType(),
                                        adaptor.getValue(), rewriter);
    auto funcType =
        rewriter.getFunctionType(TypeRange{ptr->getType(), storedValue.getType(),
                                           rewriter.getI32Type()},
                                 TypeRange{});
    rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{},
        ValueRange{*ptr, storedValue, modeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class ConvertVPTOTypedCarrierOp final : public ConversionPattern {
public:
  ConvertVPTOTypedCarrierOp(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<pto::CastPtrOp>(op))
      return failure();
    if (!hasVPTOConvertibleType(op->getOperandTypes()) &&
        !hasVPTOConvertibleType(op->getResultTypes()))
      return failure();
    if (op->getNumRegions() != 0)
      return rewriter.notifyMatchFailure(
          op, "region ops with VPTO types are handled structurally");

    FailureOr<Operation *> converted =
        convertOpResultTypes(op, operands, *typeConverter, rewriter);
    if (failed(converted))
      return failure();
    return success();
  }
};

static void populateVPTOOpLoweringPatterns(VPTOTypeConverter &typeConverter,
                                           RewritePatternSet &patterns,
                                           LoweringState &state) {
  patterns.add<LowerUnaryMaskedOpPattern<pto::VabsOp>,
               LowerUnaryMaskedOpPattern<pto::VexpOp>,
               LowerUnaryMaskedOpPattern<pto::VlnOp>,
               LowerUnaryMaskedOpPattern<pto::VnegOp>,
               LowerUnaryMaskedOpPattern<pto::VsqrtOp>,
               LowerUnaryMaskedOpPattern<pto::VreluOp>,
               LowerUnaryMaskedOpPattern<pto::VnotOp>,
               LowerVsqzOpPattern, LowerVusqzOpPattern,
               LowerVmulaOpPattern, LowerVmullOpPattern,
               LowerBinaryMaskedOpPattern<pto::VaddOp>,
               LowerBinaryMaskedOpPattern<pto::VsubOp>,
               LowerBinaryMaskedOpPattern<pto::VmulOp>,
               LowerBinaryMaskedOpPattern<pto::VdivOp>,
               LowerBinaryMaskedOpPattern<pto::VmaxOp>,
               LowerBinaryMaskedOpPattern<pto::VminOp>,
               LowerBinaryMaskedOpPattern<pto::VandOp>,
               LowerBinaryMaskedOpPattern<pto::VorOp>,
               LowerBinaryMaskedOpPattern<pto::VxorOp>,
               LowerBinaryMaskedOpPattern<pto::VpreluOp>,
               LowerCarryBinaryOpPattern<pto::VaddcOp>,
               LowerCarryBinaryOpPattern<pto::VsubcOp>,
               LowerCarryBinaryOpPattern<pto::VaddcsOp>,
               LowerCarryBinaryOpPattern<pto::VsubcsOp>,
               LowerBinaryMaskedOpPattern<pto::VshlOp>,
               LowerBinaryMaskedOpPattern<pto::VshrOp>,
               LowerVecScalarMaskedOpPattern<pto::VmulsOp>,
               LowerVecScalarMaskedOpPattern<pto::VaddsOp>,
               LowerVecScalarMaskedOpPattern<pto::VmaxsOp>,
               LowerVecScalarMaskedOpPattern<pto::VminsOp>,
               LowerVecScalarMaskedOpPattern<pto::VlreluOp>,
               LowerVecScalarMaskedOpPattern<pto::VshlsOp>,
               LowerVecScalarMaskedOpPattern<pto::VshrsOp>,
               LowerWideningReductionUnaryOpPattern<pto::VcaddOp>,
               LowerReductionUnaryOpPattern<pto::VcmaxOp>,
               LowerReductionUnaryOpPattern<pto::VcminOp>,
               LowerReductionUnaryOpPattern<pto::VcgaddOp>,
               LowerReductionUnaryOpPattern<pto::VcgmaxOp>,
               LowerReductionUnaryOpPattern<pto::VcgminOp>,
               LowerReductionUnaryOpPattern<pto::VcpaddOp>,
               LowerVdupOpPattern,
               LowerVbrOpPattern,
               LowerPredicatePackOpPattern<pto::PpackOp>,
               LowerPredicatePackOpPattern<pto::PunpackOp>,
               LowerVselOpPattern, LowerVselrOpPattern, LowerPnotOpPattern,
               LowerPredicateMaskBinaryOpPattern<pto::PselOp>,
               LowerPredicateMaskBinaryOpPattern<pto::PandOp>,
               LowerPredicateMaskBinaryOpPattern<pto::PorOp>,
               LowerPredicateMaskBinaryOpPattern<pto::PxorOp>,
               LowerPredicatePairReorderOpPattern<pto::PdintlvB8Op>,
               LowerPredicatePairReorderOpPattern<pto::PdintlvB16Op>,
               LowerPredicatePairReorderOpPattern<pto::PdintlvB32Op>,
               LowerPredicatePairReorderOpPattern<pto::PintlvB8Op>,
               LowerPredicatePairReorderOpPattern<pto::PintlvB16Op>,
               LowerPredicatePairReorderOpPattern<pto::PintlvB32Op>,
               LowerUnpackOpPattern<pto::VsunpackOp>,
               LowerUnpackOpPattern<pto::VzunpackOp>,
               LowerVpackOpPattern,
               LowerInterleaveOpPattern<pto::VintlvOp>,
               LowerInterleaveOpPattern<pto::VdintlvOp>,
               LowerCmpOpPattern<pto::VcmpOp>,
               LowerCmpOpPattern<pto::VcmpsOp>,
               LowerPltOpPattern<pto::PltB8Op>,
               LowerPltOpPattern<pto::PltB16Op>,
               LowerPltOpPattern<pto::PltB32Op>,
               LowerPsetOpPattern<pto::PsetB8Op>,
               LowerPsetOpPattern<pto::PsetB16Op>,
               LowerPsetOpPattern<pto::PsetB32Op>,
               LowerPgeOpPattern<pto::PgeB8Op>,
               LowerPgeOpPattern<pto::PgeB16Op>,
               LowerPgeOpPattern<pto::PgeB32Op>,
               LowerRuntimeQueryOpPattern<pto::GetCtrlOp>,
               LowerGetVms4SrOpPattern,
               LowerRuntimeQueryOpPattern<pto::GetTidXOp>,
               LowerRuntimeQueryOpPattern<pto::GetTidYOp>,
               LowerRuntimeQueryOpPattern<pto::GetTidZOp>,
               LowerRuntimeQueryOpPattern<pto::GetBlockDimXOp>,
               LowerRuntimeQueryOpPattern<pto::GetBlockDimYOp>,
               LowerRuntimeQueryOpPattern<pto::GetBlockDimZOp>,
               LowerRuntimeQueryOpPattern<pto::GetGridDimXOp>,
               LowerRuntimeQueryOpPattern<pto::GetGridDimYOp>,
               LowerRuntimeQueryOpPattern<pto::GetGridDimZOp>,
               LowerRuntimeQueryOpPattern<pto::GetBlockIdxXOp>,
               LowerRuntimeQueryOpPattern<pto::GetBlockIdxYOp>,
               LowerRuntimeQueryOpPattern<pto::GetBlockIdxZOp>,
               LowerRuntimeQueryOpPattern<pto::GetVecCoreIdOp>,
               LowerRuntimeQueryOpPattern<pto::GetLaneIdOp>,
               LowerRuntimeQueryOpPattern<pto::GetClock32Op>,
               LowerRuntimeQueryOpPattern<pto::GetClock64Op>,
               LowerRuntimeQueryOpPattern<pto::GetLaneMaskEqOp>,
               LowerRuntimeQueryOpPattern<pto::GetLaneMaskLeOp>,
               LowerRuntimeQueryOpPattern<pto::GetLaneMaskLtOp>,
               LowerRuntimeQueryOpPattern<pto::GetLaneMaskGeOp>,
               LowerRuntimeQueryOpPattern<pto::GetLaneMaskGtOp>,
               LowerVoteOpPattern<pto::VoteAllOp>,
               LowerVoteOpPattern<pto::VoteAnyOp>,
               LowerVoteOpPattern<pto::VoteUniOp>,
               LowerVoteOpPattern<pto::VoteBallotOp>,
               LowerShuffleOpPattern<pto::ShuffleIdxOp>,
               LowerShuffleOpPattern<pto::ShuffleUpOp>,
               LowerShuffleOpPattern<pto::ShuffleDownOp>,
               LowerShuffleOpPattern<pto::ShuffleBflyOp>,
               LowerReduxOpPattern<pto::ReduxAddOp>,
               LowerReduxOpPattern<pto::ReduxMaxOp>,
               LowerReduxOpPattern<pto::ReduxMinOp>,
               LowerAtomicCasOpPattern,
               LowerAtomicBinaryOpPattern<pto::AtomicExchOp>,
               LowerAtomicBinaryOpPattern<pto::AtomicAddOp>,
               LowerAtomicBinaryOpPattern<pto::AtomicSubOp>,
               LowerAtomicBinaryOpPattern<pto::AtomicMinOp>,
               LowerAtomicBinaryOpPattern<pto::AtomicMaxOp>,
               LowerAtomicBinaryOpPattern<pto::AtomicAndOp>,
               LowerAtomicBinaryOpPattern<pto::AtomicOrOp>,
               LowerAtomicBinaryOpPattern<pto::AtomicXorOp>,
               LowerScalarIntrinsicOpPattern<pto::PrmtOp>,
               LowerMulhiOpPattern,
               LowerMulI32ToI64OpPattern,
               LowerSqrtOpPattern,
               LowerUnaryScalarMathOpPattern<pto::AbsFOp>,
               LowerUnaryScalarMathOpPattern<pto::ExpOp>,
               LowerUnaryScalarMathOpPattern<pto::LogOp>,
               LowerUnaryScalarMathOpPattern<pto::CeilOp>,
               LowerUnaryScalarMathOpPattern<pto::FloorOp>,
               LowerUnaryScalarMathOpPattern<pto::RintOp>,
               LowerUnaryScalarMathOpPattern<pto::RoundOp>,
               LowerBinaryScalarMathOpPattern<pto::FMinOp>,
               LowerBinaryScalarMathOpPattern<pto::FMaxOp>,
               LowerBinaryScalarMathOpPattern<pto::PowOp>,
               LowerFmaOpPattern,
               LowerConvertOpPattern,
               LowerSimtFenceOpPattern<pto::SyncthreadsOp>,
               LowerSimtFenceOpPattern<pto::ThreadfenceOp>,
               LowerSimtFenceOpPattern<pto::ThreadfenceBlockOp>,
               LowerKeepOpPattern,
               LowerResumeOpPattern,
               LowerBinaryI64PureOpPattern<pto::Sbitset0Op>,
               LowerBinaryI64PureOpPattern<pto::Sbitset1Op>,
               LowerSetLoopConfigOpPattern<pto::SetLoop2StrideOutToUbOp>,
               LowerSetLoopConfigOpPattern<pto::SetLoop1StrideOutToUbOp>,
               LowerSetLoopConfigOpPattern<pto::SetLoopSizeOutToUbOp>,
               LowerSetLoopConfigOpPattern<pto::SetLoop2StrideUbToOutOp>,
               LowerSetLoopConfigOpPattern<pto::SetLoop1StrideUbToOutOp>,
               LowerSetLoopConfigOpPattern<pto::SetLoopSizeUbToOutOp>,
               LowerSetLoopConfigOpPattern<pto::SetLoop3ParaOp>,
               LowerSetLoopConfigOpPattern<pto::SetChannelParaOp>,
               LowerUnaryI64ConfigOpPattern<pto::SetCtrlOp>,
               LowerStoreVfSimtInfoOpPattern,
               LowerUnaryConfigOpPattern<pto::SetMovPadValOp>,
               LowerUnaryI64ConfigOpPattern<pto::SetQuantPreOp>,
               LowerUnaryI64ConfigOpPattern<pto::SetReluAlphaOp>,
               LowerUnaryI64ConfigOpPattern<pto::SetFixClipReluOp>,
               LowerUnaryI64ConfigOpPattern<pto::SetLoop2StrideOutToL1Op>,
               LowerUnaryI64ConfigOpPattern<pto::SetLoop1StrideOutToL1Op>,
               LowerUnaryI64ConfigOpPattern<pto::SetLoopSizeOutToL1Op>,
               LowerUnaryI64ConfigOpPattern<pto::SetMte2NzParaOp>,
               LowerUnaryI64ConfigOpPattern<pto::SetPadValOutToL1Op>,
               LowerUnaryI64ConfigOpPattern<pto::SetFpcOp>,
               LowerNullaryConfigOpPattern<pto::SetAtomicS32Op>,
               LowerNullaryConfigOpPattern<pto::SetAtomicS8Op>,
               LowerPipeEventSyncOpPattern<pto::SetFlagOp>,
               LowerPipeEventSyncOpPattern<pto::WaitFlagOp>,
               LowerPipeEventDynSyncOpPattern<pto::SetFlagDynOp>,
               LowerPipeEventDynSyncOpPattern<pto::WaitFlagDynOp>,
               LowerBarrierOpPattern, LowerMemBarOpPattern,
               LowerBufSyncOpPattern<pto::GetBufOp>,
               LowerBufSyncOpPattern<pto::RlsBufOp>,
               LowerRuntimeQueryOpPattern<pto::GetBlockIdxOp>,
               LowerRuntimeQueryOpPattern<pto::GetSubBlockIdxOp>,
               LowerRuntimeQueryOpPattern<pto::GetBlockNumOp>,
               LowerRuntimeQueryOpPattern<pto::GetSubBlockNumOp>,
               LowerVldsOpPattern, LowerVldsx2OpPattern, LowerVsldbOpPattern,
               LowerVldasOpPattern, LowerInitAlignOpPattern,
               LowerVldusOpPattern, LowerSprclrOpPattern,
               LowerVstsOpPattern, LowerVsstbOpPattern,
               LowerVstsx2OpPattern,
               LowerVstarOpPattern, LowerVstasOpPattern,
               LowerVgather2OpPattern, LowerVgather2BcOpPattern,
               LowerVgatherbOpPattern, LowerVscatterOpPattern,
               LowerVaxpyOpPattern,
               LowerVciOpPattern, LowerVexpdifOpPattern,
               LowerVbitsortOpPattern, LowerVmrgsort4OpPattern,
               LowerVtrcOpPattern, LowerVcvtOpPattern,
               LowerVbitcastOpPattern, LowerPbitcastOpPattern,
               LowerPredicateLoadOpPattern<pto::PldiOp>,
               LowerPredicateLoadOpPattern<pto::PldsOp>,
               LowerPredicateStoreOpPattern<pto::PstiOp>,
               LowerPredicateStoreOpPattern<pto::PstsOp>,
               LowerPstuOpPattern, LowerVstusOpPattern, LowerVsturOpPattern,
               LowerInterCoreSyncOpPattern<pto::SyncSetOp>,
               LowerInterCoreSyncOpPattern<pto::SyncWaitOp>,
               LowerCopyGmToCbufOpPattern, LowerLoadCbufToCaOpPattern,
               LowerLoadCbufToCbOpPattern,
               LowerLoadCbufToS4OpPattern<pto::LoadCbufToCaS4Op>,
               LowerLoadCbufToS4OpPattern<pto::LoadCbufToCbS4Op>,
               LowerLoadCbufToCaMxOpPattern,
               LowerLoadCbufToCbMxOpPattern, LowerCopyMatrixCcToGmOpPattern,
               LowerCopyMatrixCcToBufOpPattern<pto::CopyMatrixCcToCbufOp>,
               LowerCopyMatrixCcToBufOpPattern<pto::CopyMatrixCcToUbOp>,
               LowerCopyCbufToBtOpPattern, LowerCopyCbufToFbufOpPattern,
               LowerCopyGmToCbufMultiOpPattern<pto::CopyGmToCbufMultiNd2NzOp>,
               LowerCopyGmToCbufMultiOpPattern<pto::CopyGmToCbufMultiDn2NzOp>,
               LowerMadRawPattern<pto::MadRawOp>,
               LowerMadRawPattern<pto::MadBiasRawOp>,
               LowerMadRawPattern<pto::MadMxRawOp>,
               LowerMadRawPattern<pto::MadMxBiasRawOp>,
               LowerCopyOpPattern<pto::CopyGmToUbufOp>,
               LowerCopyOpPattern<pto::CopyUbufToGmOp>,
               LowerCopyUbufToUbufOpPattern,
               LowerCopyCbufToUbufOpPattern,
               LowerCopyUbufToCbufOpPattern>(
      typeConverter, patterns.getContext(), state);
}

static void configureVPTOOpLoweringTarget(ConversionTarget &target,
                                          VPTOTypeConverter &typeConverter) {
  (void)typeConverter;
  target.addLegalOp<ModuleOp>();
  target.addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect,
                         LLVM::LLVMDialect,
                         func::FuncDialect, scf::SCFDialect>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addIllegalOp<pto::SetFlagOp, pto::WaitFlagOp, pto::SetFlagDynOp, pto::WaitFlagDynOp, pto::SyncSetOp,
                      pto::SyncWaitOp, pto::BarrierOp, pto::MemBarOp,
                      pto::GetBufOp, pto::RlsBufOp>();
  target.addIllegalOp<pto::GetBlockIdxOp, pto::GetSubBlockIdxOp,
                      pto::GetBlockNumOp, pto::GetSubBlockNumOp,
                      pto::GetCtrlOp, pto::GetVms4SrOp, pto::GetTidXOp,
                      pto::GetTidYOp, pto::GetTidZOp,
                      pto::GetBlockDimXOp, pto::GetBlockDimYOp,
                      pto::GetBlockDimZOp, pto::GetGridDimXOp,
                      pto::GetGridDimYOp, pto::GetGridDimZOp,
                      pto::GetBlockIdxXOp, pto::GetBlockIdxYOp,
                      pto::GetBlockIdxZOp, pto::GetVecCoreIdOp,
                      pto::GetLaneIdOp, pto::GetClock32Op, pto::GetClock64Op,
                      pto::GetLaneMaskEqOp, pto::GetLaneMaskLeOp,
                      pto::GetLaneMaskLtOp, pto::GetLaneMaskGeOp,
                      pto::GetLaneMaskGtOp, pto::VoteAllOp, pto::VoteAnyOp,
                      pto::VoteUniOp, pto::VoteBallotOp, pto::ShuffleIdxOp,
                      pto::ShuffleUpOp, pto::ShuffleDownOp,
                      pto::ShuffleBflyOp, pto::ReduxAddOp, pto::ReduxMaxOp,
                      pto::ReduxMinOp, pto::AtomicCasOp, pto::AtomicExchOp,
                      pto::AtomicAddOp, pto::AtomicSubOp,
                      pto::AtomicMinOp, pto::AtomicMaxOp,
                      pto::AtomicAndOp, pto::AtomicOrOp,
                      pto::AtomicXorOp, pto::PrmtOp,
                      pto::MulhiOp, pto::MulI32ToI64Op, pto::SqrtOp,
                      pto::AbsFOp, pto::ExpOp, pto::LogOp, pto::CeilOp,
                      pto::FloorOp, pto::RintOp, pto::RoundOp, pto::FMinOp,
                      pto::FMaxOp, pto::PowOp, pto::FmaOp, pto::ConvertOp,
                      pto::SyncthreadsOp, pto::ThreadfenceOp,
                      pto::ThreadfenceBlockOp, pto::KeepOp, pto::ResumeOp>();
  target.addIllegalOp<pto::SetLoop2StrideOutToUbOp, pto::SetLoop1StrideOutToUbOp,
                      pto::SetLoopSizeOutToUbOp, pto::SetLoop2StrideUbToOutOp,
                      pto::SetLoop1StrideUbToOutOp, pto::SetLoopSizeUbToOutOp,
                      pto::SetLoop3ParaOp, pto::SetChannelParaOp,
                      pto::SetLoop2StrideOutToL1Op, pto::SetLoop1StrideOutToL1Op,
                      pto::SetLoopSizeOutToL1Op, pto::SetMte2NzParaOp,
                      pto::SetPadValOutToL1Op, pto::SetReluAlphaOp,
                      pto::SetFixClipReluOp, pto::SetFpcOp,
                      pto::SetAtomicS32Op, pto::SetAtomicS8Op, pto::SetCtrlOp,
                      pto::StoreVfSimtInfoOp,
                      pto::SetMovPadValOp, pto::SetQuantPreOp>();
  target.addIllegalOp<pto::Sbitset0Op, pto::Sbitset1Op>();
  target.addIllegalOp<pto::VldsOp, pto::Vldsx2Op, pto::VsldbOp,
                      pto::VldasOp, pto::InitAlignOp, pto::VldusOp,
                      pto::SprclrOp, pto::VstsOp, pto::VsstbOp, pto::Vstsx2Op,
                      pto::VstarOp, pto::VstasOp, pto::Vgather2Op,
                      pto::Vgather2BcOp, pto::VgatherbOp, pto::VscatterOp,
                      pto::PldiOp, pto::PldsOp, pto::PstiOp, pto::PstsOp,
                      pto::PstuOp, pto::VstusOp, pto::VsturOp>();
  target.addIllegalOp<pto::PltB8Op, pto::PltB16Op, pto::PltB32Op,
                      pto::PsetB8Op, pto::PsetB16Op, pto::PsetB32Op,
                      pto::PgeB8Op, pto::PgeB16Op, pto::PgeB32Op>();
  target.addIllegalOp<pto::VabsOp, pto::VexpOp, pto::VlnOp, pto::VnegOp,
                      pto::VsqrtOp, pto::VreluOp, pto::VnotOp, pto::VsqzOp,
                      pto::VusqzOp, pto::VmulaOp, pto::VmullOp, pto::VaddOp,
                      pto::VsubOp, pto::VmulOp,
                      pto::VdivOp, pto::VmaxOp, pto::VminOp, pto::VandOp,
                      pto::VorOp, pto::VxorOp, pto::VaddcOp, pto::VsubcOp,
                      pto::VaddcsOp, pto::VsubcsOp, pto::VshlOp, pto::VshrOp,
                      pto::VmulsOp, pto::VaddsOp, pto::VmaxsOp,
                      pto::VminsOp, pto::VlreluOp, pto::VshlsOp, pto::VshrsOp,
                      pto::VcaddOp, pto::VcmaxOp, pto::VcminOp,
                      pto::VcgaddOp, pto::VcgmaxOp, pto::VcgminOp, pto::VcpaddOp,
                      pto::VdupOp, pto::VbrOp,
                      pto::PpackOp, pto::PunpackOp, pto::PbitcastOp,
                      pto::VselOp, pto::VselrOp,
                      pto::PnotOp, pto::PselOp, pto::PandOp, pto::PorOp, pto::PxorOp,
                      pto::PdintlvB8Op, pto::PdintlvB16Op, pto::PdintlvB32Op,
                      pto::PintlvB8Op, pto::PintlvB16Op, pto::PintlvB32Op,
                      pto::VsunpackOp, pto::VzunpackOp, pto::VpackOp,
                      pto::VintlvOp, pto::VdintlvOp, pto::VpreluOp,
                      pto::VaxpyOp, pto::VciOp, pto::VexpdifOp,
                      pto::VbitsortOp, pto::Vmrgsort4Op, pto::VtrcOp,
                      pto::VcvtOp,
                      pto::VbitcastOp,
                      pto::VcmpOp, pto::VcmpsOp,
                      pto::CopyGmToUbufOp, pto::CopyUbufToGmOp,
                      pto::CopyUbufToUbufOp, pto::CopyCbufToUbufOp,
                      pto::CopyUbufToCbufOp,
                      pto::CopyGmToCbufOp, pto::LoadCbufToCaOp,
                      pto::LoadCbufToCbOp, pto::LoadCbufToCaS4Op,
                      pto::LoadCbufToCbS4Op, pto::LoadCbufToCaMxOp,
                      pto::LoadCbufToCbMxOp, pto::CopyMatrixCcToGmOp,
                      pto::CopyMatrixCcToCbufOp, pto::CopyMatrixCcToUbOp,
                      pto::CopyCbufToBtOp, pto::CopyCbufToFbufOp,
                      pto::CopyGmToCbufMultiNd2NzOp,
                      pto::CopyGmToCbufMultiDn2NzOp,
                      pto::MadOp, pto::MadAccOp, pto::MadBiasOp, pto::MadMxOp,
                      pto::MadMxAccOp, pto::MadMxBiasOp,
                      pto::MadRawOp, pto::MadBiasRawOp, pto::MadMxRawOp,
                      pto::MadMxBiasRawOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
}

static void populateVPTOStructuralTypePatterns(
    VPTOTypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns,
                                                       target);
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
}

static void foldVPTOTypeCasts(ModuleOp module, TypeConverter &typeConverter) {
  SmallVector<UnrealizedConversionCastOp> castsToFold;
  module.walk([&](UnrealizedConversionCastOp castOp) {
    if (castOp->getNumOperands() != 1 || castOp->getNumResults() != 1)
      return;
    if (!hasVPTOConvertibleType(castOp->getOperandTypes()) &&
        !hasVPTOConvertibleType(castOp->getResultTypes()))
      return;
    Type convertedResultType =
        typeConverter.convertType(castOp.getResult(0).getType());
    if (convertedResultType &&
        convertedResultType == castOp.getOperand(0).getType())
      castsToFold.push_back(castOp);
  });
  for (UnrealizedConversionCastOp castOp : castsToFold) {
    castOp.getResult(0).replaceAllUsesWith(castOp.getOperand(0));
    castOp.erase();
  }
}

static LogicalResult lowerVPTOOps(ModuleOp module, llvm::raw_ostream &diagOS) {
  MLIRContext *context = module.getContext();
  VPTOTypeConverter typeConverter(context);
  ConversionTarget target(*context);
  RewritePatternSet patterns(context);
  LoweringState state;

  configureVPTOOpLoweringTarget(target, typeConverter);
  populateVPTOOpLoweringPatterns(typeConverter, patterns, state);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    diagOS << "VPTO LLVM emission failed: VPTO op lowering failed\n";
    return failure();
  }
  if (failed(materializeDecls(module, state.plannedDecls, diagOS)))
    return failure();
  return success();
}

static LogicalResult lowerVPTOTypes(ModuleOp module, llvm::raw_ostream &diagOS) {
  MLIRContext *context = module.getContext();
  VPTOTypeConverter typeConverter(context);
  ConversionTarget target(*context);
  RewritePatternSet patterns(context);
  LoweringState state;

  target.addLegalOp<ModuleOp>();
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType()) &&
           typeConverter.isLegal(&op.getBody());
  });
  target.addDynamicallyLegalOp<func::CallOp>(
      [&](func::CallOp op) { return typeConverter.isLegal(op); });
  target.addDynamicallyLegalOp<func::ReturnOp>(
      [&](func::ReturnOp op) { return typeConverter.isLegal(op); });
  target.addDynamicallyLegalOp<cf::BranchOp, cf::CondBranchOp>(
      [&](Operation *op) {
        return isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                                typeConverter);
      });
  target.addIllegalOp<pto::AddPtrOp, pto::CastPtrOp, pto::LoadScalarOp,
                      pto::StoreScalarOp, pto::PTOLoadOp, pto::PTOStoreOp,
                      pto::PTOLdgOp, pto::PTOStgOp>();
  target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
      [&](UnrealizedConversionCastOp op) {
        return !hasVPTOConvertibleType(op->getOperandTypes()) &&
               !hasVPTOConvertibleType(op->getResultTypes());
      });
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return typeConverter.isLegal(op->getOperandTypes()) &&
           typeConverter.isLegal(op->getResultTypes());
  });

  populateVPTOStructuralTypePatterns(typeConverter, patterns, target);
  patterns.add<ConvertPtoAddPtrOp, ConvertPtoCastPtrOp, ConvertPtoLoadScalarOp,
               ConvertPtoStoreScalarOp>(typeConverter, context);
  patterns.add<ConvertPtoLoadOp, ConvertPtoStoreOp, ConvertPtoLdgOp,
               ConvertPtoStgOp>(
      typeConverter, context, state);
  patterns.add<ConvertVPTOUnrealizedCastOp>(typeConverter, context);
  patterns.add<ConvertVPTOTypedCarrierOp>(typeConverter, context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    diagOS << "VPTO LLVM emission failed: VPTO type lowering failed\n";
    return failure();
  }
  if (failed(materializeDecls(module, state.plannedDecls, diagOS)))
    return failure();
  foldVPTOTypeCasts(module, typeConverter);
  return success();
}

static Type normalizeTypeForOfficialLLVMLowering(Type type, Builder &builder) {
  type = convertVPTOType(type, builder);
  return type;
}

static void normalizeFuncSignaturesForOfficialLLVMLowering(ModuleOp module) {
  Builder builder(module.getContext());

  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    FunctionType oldType = funcOp.getFunctionType();
    SmallVector<Type> newInputs;
    SmallVector<Type> newResults;
    bool changed = false;

    for (Type input : oldType.getInputs()) {
      Type normalized = normalizeTypeForOfficialLLVMLowering(input, builder);
      changed |= (normalized != input);
      newInputs.push_back(normalized);
    }
    for (Type result : oldType.getResults()) {
      Type normalized = normalizeTypeForOfficialLLVMLowering(result, builder);
      changed |= (normalized != result);
      newResults.push_back(normalized);
    }

    if (!changed)
      continue;

    auto newType = builder.getFunctionType(newInputs, newResults);
    funcOp.setFunctionTypeAttr(TypeAttr::get(newType));

    if (funcOp.isExternal())
      continue;
    Block &entry = funcOp.getBody().front();
    for (auto [arg, newType] : llvm::zip(entry.getArguments(), newInputs))
      if (arg.getType() != newType)
        arg.setType(newType);
  }
}

static void forceV300CtrlModeForVPTOFuncs(ModuleOp module) {
  OpBuilder builder(module.getContext());

  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    if (!needsV300CtrlModeForVPTOFunc(funcOp))
      continue;

    Block &entry = funcOp.getBody().front();
    builder.setInsertionPointToStart(&entry);
    auto i64Type = builder.getI64Type();
    auto bit60 = builder.create<arith::ConstantOp>(
        funcOp.getLoc(), i64Type, builder.getI64IntegerAttr(60));
    Value ctrl =
        builder.create<pto::GetCtrlOp>(funcOp.getLoc(), i64Type).getResult();
    Value ctrlV300 = builder
                         .create<pto::Sbitset0Op>(funcOp.getLoc(), i64Type,
                                                  ctrl, bit60.getResult())
                         .getResult();
    builder.create<pto::SetCtrlOp>(funcOp.getLoc(), ctrlV300);
  }
}

static std::optional<FunctionKernelKind> getKernelKind(ModuleOp module) {
  auto kernelKind = module->getAttrOfType<FunctionKernelKindAttr>(
      FunctionKernelKindAttr::name);
  if (!kernelKind)
    return std::nullopt;
  return kernelKind.getKernelKind();
}

static VPTOEmissionOptions
makeDeviceEmissionOptions(const VPTOEmissionOptions &baseOptions,
                          FunctionKernelKind kind) {
  VPTOEmissionOptions options = baseOptions;
  constexpr llvm::StringLiteral kVecTargetFeatures =
      "+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,"
      "+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,"
      "+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec";
  constexpr llvm::StringLiteral kCubeTargetFeatures =
      "+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,"
      "+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,"
      "+MOVX8,+SPR7bits,+SyncV,+dav-c310-cube";
  if (kind == FunctionKernelKind::Vector) {
    options.march = "dav-c310-vec";
    options.aicoreArch = "dav-c310-vec";
    options.defaultTargetCPU = "dav-c310-vec";
    options.defaultTargetFeatures = kVecTargetFeatures.str();
  } else if (kind == FunctionKernelKind::Cube) {
    options.march = "dav-c310-cube";
    options.aicoreArch = "dav-c310-cube";
    options.defaultTargetCPU = "dav-c310-cube";
    options.defaultTargetFeatures = kCubeTargetFeatures.str();
  }
  return options;
}

static FailureOr<ModuleOp>
getUniqueDeviceModuleByKernelKind(ModuleOp module, FunctionKernelKind kind,
                                  llvm::raw_ostream &diagOS) {
  ModuleOp matched;
  for (ModuleOp child : module.getOps<ModuleOp>()) {
    auto kernelKind = getKernelKind(child);
    if (!kernelKind)
      continue;
    if (*kernelKind != kind)
      continue;
    if (matched) {
      diagOS << "VPTO LLVM emission failed: duplicate device module with "
             << FunctionKernelKindAttr::name << "\n";
      return failure();
    }
    matched = child;
  }
  return matched;
}

static LogicalResult renameKernelFunctionsForKernelKind(ModuleOp module,
                                                        llvm::raw_ostream &diagOS) {
  auto kernelKind = getKernelKind(module);
  if (!kernelKind) {
    diagOS << "VPTO LLVM emission failed: device module missing "
           << FunctionKernelKindAttr::name << "\n";
    return failure();
  }

  StringRef suffix;
  if (*kernelKind == FunctionKernelKind::Vector)
    suffix = kVectorSuffix;
  else if (*kernelKind == FunctionKernelKind::Cube)
    suffix = kCubeSuffix;
  else {
    diagOS << "VPTO LLVM emission failed: unsupported "
           << FunctionKernelKindAttr::name << "\n";
    return failure();
  }

  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    if (!pto::hasPTOKernelAttr(funcOp.getOperation()))
      continue;
    if (funcOp.getSymName().ends_with(suffix))
      continue;
    funcOp.setSymName((funcOp.getSymName() + suffix).str());
  }
  return success();
}

struct LowerVPTOOpsPass final
    : public PassWrapper<LowerVPTOOpsPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    materializeVecScopeCarrierLoops(getOperation());
    // Remove dead pto.alloc_tile ops before lowering. These can appear when
    // the original kernel's tile_buf intrinsics have already been folded away
    // by FoldTileBufIntrinsics, but a subsequent pass (e.g. AIC-scope cloning)
    // re-introduces alloc_tile copies whose results have no users. The lowering
    // patterns do not cover AllocTileOp, so leaving them in the IR causes
    // translateModuleToLLVMIR to fail.
    {
      SmallVector<pto::AllocTileOp> deadAllocs;
      getOperation().walk([&](pto::AllocTileOp alloc) {
        if (alloc.use_empty())
          deadAllocs.push_back(alloc);
      });
      for (pto::AllocTileOp alloc : llvm::reverse(deadAllocs))
        alloc.erase();
    }
    if (failed(lowerVPTOOps(getOperation(), llvm::errs())))
      signalPassFailure();
  }
};

struct LowerVPTOTypesPass final
    : public PassWrapper<LowerVPTOTypesPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    if (failed(lowerVPTOTypes(getOperation(), llvm::errs())))
      signalPassFailure();
  }
};

struct NormalizeFuncSignaturesForLLVMLoweringPass final
    : public PassWrapper<NormalizeFuncSignaturesForLLVMLoweringPass,
                         OperationPass<ModuleOp>> {
  void runOnOperation() override {
    normalizeFuncSignaturesForOfficialLLVMLowering(getOperation());
  }
};

struct PrepareVPTOLLVMLoweringPass final
    : public PassWrapper<PrepareVPTOLLVMLoweringPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    pto::annotatePTOEntryFunctions(module);
    forceV300CtrlModeForVPTOFuncs(module);
    if (failed(renameKernelFunctionsForKernelKind(module, llvm::errs())))
      signalPassFailure();
  }
};

llvm::StringSet<llvm::BumpPtrAllocator>
collectSimtEntryFunctionNames(ModuleOp module) {
  llvm::StringSet<llvm::BumpPtrAllocator> simtEntries;
  module.walk([&](func::FuncOp funcOp) {
    if (funcOp->hasAttr(pto::kPTOSimtEntryAttrName))
      simtEntries.insert(funcOp.getSymName());
  });
  return simtEntries;
}

static void applySimtEntryCallingConvention(
    llvm::Module &llvmModule,
    const llvm::StringSet<llvm::BumpPtrAllocator> &simtEntryNames) {
  for (llvm::Function &function : llvmModule) {
    if (simtEntryNames.contains(function.getName())) {
      function.setCallingConv(llvm::CallingConv::SimtEntry);
      function.addFnAttr(llvm::Attribute::NoInline);
      // Match Bisheng's C++ frontend shape for SIMT outlined bodies. The
      // exported wrapper owns the real kernel metadata, while the SIMT body is
      // an ODR helper called with the SIMT calling convention. In CANN beta.1,
      // leaving the SIMT body as a strong GLOBAL FUNC makes the runtime count it
      // as an extra kernel without matching .ascend.meta, which can corrupt the
      // selected kernel metadata. linkonce_odr lowers to a weak helper symbol
      // and avoids that beta.1 metadata mismatch.
      function.setLinkage(llvm::GlobalValue::LinkOnceODRLinkage);
    }
  }

  for (llvm::Function &function : llvmModule) {
    for (llvm::BasicBlock &block : function) {
      for (llvm::Instruction &inst : block) {
        auto *call = llvm::dyn_cast<llvm::CallBase>(&inst);
        if (!call)
          continue;
        auto *callee = call->getCalledFunction();
        if (!callee || !simtEntryNames.contains(callee->getName()))
          continue;
        call->setCallingConv(llvm::CallingConv::SimtEntry);
      }
    }
  }
}

static FailureOr<EmittedLLVMModule>
emitDeviceLLVMModule(ModuleOp deviceModule, StringRef kernelKind,
                     const VPTOEmissionOptions &options,
                     const llvm::StringSet<llvm::BumpPtrAllocator> &simtEntryNames,
                     llvm::raw_ostream &diagOS) {
  if (!deviceModule)
    return EmittedLLVMModule{};
  if (failed(applyQueriedTargetAttrs(deviceModule, options, diagOS)))
    return failure();

  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  registerBuiltinDialectTranslation(*deviceModule.getContext());
  registerLLVMDialectTranslation(*deviceModule.getContext());
  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(deviceModule.getOperation(), *llvmContext);
  if (!llvmModule) {
    diagOS << "VPTO LLVM emission failed: LLVM IR export failed for "
           << kernelKind << " module\n";
    return failure();
  }

  applySimtEntryCallingConvention(*llvmModule, simtEntryNames);
  if (failed(attachAIVectorScopeMetadata(*llvmModule, diagOS)))
    return failure();
  attachHIVMKernelAnnotations(*llvmModule, deviceModule);
  llvmModule->setModuleIdentifier(("ptoas.hivm.official." + kernelKind).str());
  llvmModule->setSourceFileName(("ptoas.hivm.official." + kernelKind).str());
  return EmittedLLVMModule{std::move(llvmContext), std::move(llvmModule)};
}

template <typename EmitFn>
static LogicalResult runPipeline(ModuleOp module, llvm::raw_ostream &diagOS,
                                 EmitFn &&emit) {
  OwningOpRef<Operation *> clonedOp(module->clone());
  ModuleOp clonedModule = cast<ModuleOp>(*clonedOp);

  if (failed(validateVPTOAuthoringIR(clonedModule, &diagOS))) {
    diagOS << "VPTO LLVM emission failed: authoring-stage VPTO legality "
              "validation failed\n";
    return failure();
  }

  PassManager pm(clonedModule.getContext());
  pm.enableVerifier();
  auto &kernelModulePM = pm.nest<ModuleOp>();
  kernelModulePM.addPass(std::make_unique<PrepareVPTOLLVMLoweringPass>());
  kernelModulePM.addPass(std::make_unique<LowerVPTOOpsPass>());
  kernelModulePM.addPass(std::make_unique<LowerVPTOTypesPass>());
  kernelModulePM.addPass(
      std::make_unique<NormalizeFuncSignaturesForLLVMLoweringPass>());
  kernelModulePM.addPass(arith::createArithExpandOpsPass());
  kernelModulePM.addPass(createConvertSCFToCFPass());
  kernelModulePM.addPass(createArithToLLVMConversionPass());
  kernelModulePM.addPass(createConvertIndexToLLVMPass());
  kernelModulePM.addPass(createFinalizeMemRefToLLVMConversionPass());
  kernelModulePM.addPass(createConvertFuncToLLVMPass());
  kernelModulePM.addPass(createConvertControlFlowToLLVMPass());
  kernelModulePM.addPass(createReconcileUnrealizedCastsPass());
  if (failed(mlir::applyPassManagerCLOptions(pm))) {
    diagOS << "VPTO LLVM emission failed: unable to apply MLIR pass manager "
              "command-line options\n";
    return failure();
  }
  if (failed(pm.run(clonedModule))) {
    diagOS << "VPTO LLVM emission failed: official lowering pipeline failed\n";
    return failure();
  }
  return emit(clonedModule);
}

} // namespace

LogicalResult lowerVPTOModuleToLLVMModules(
    ModuleOp module, const VPTOEmissionOptions &options,
    EmittedLLVMModule &cubeModule, EmittedLLVMModule &vectorModule,
    llvm::raw_ostream &diagOS) {
  llvm::StringSet<llvm::BumpPtrAllocator> simtEntryNames =
      collectSimtEntryFunctionNames(module);
  cubeModule.context.reset();
  cubeModule.module.reset();
  vectorModule.context.reset();
  vectorModule.module.reset();
  return runPipeline(module, diagOS,
                     [&](ModuleOp loweredModule) {
    auto vectorDeviceModule =
        getUniqueDeviceModuleByKernelKind(
            loweredModule, FunctionKernelKind::Vector, diagOS);
    if (failed(vectorDeviceModule))
      return failure();
    auto cubeDeviceModule =
        getUniqueDeviceModuleByKernelKind(
            loweredModule, FunctionKernelKind::Cube, diagOS);
    if (failed(cubeDeviceModule))
      return failure();

    if (*vectorDeviceModule) {
      auto vectorOptions =
          makeDeviceEmissionOptions(options, FunctionKernelKind::Vector);
      auto emitted =
          emitDeviceLLVMModule(*vectorDeviceModule, "vector", vectorOptions,
                               simtEntryNames, diagOS);
      if (failed(emitted))
        return failure();
      vectorModule.context = std::move(emitted->context);
      vectorModule.module = std::move(emitted->module);
    }
    if (*cubeDeviceModule) {
      auto cubeOptions =
          makeDeviceEmissionOptions(options, FunctionKernelKind::Cube);
      auto emitted =
          emitDeviceLLVMModule(*cubeDeviceModule, "cube", cubeOptions,
                               simtEntryNames, diagOS);
      if (failed(emitted))
        return failure();
      cubeModule.context = std::move(emitted->context);
      cubeModule.module = std::move(emitted->module);
    }
    return success();
                     });
}

} // namespace mlir::pto
