// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCCommon.cpp - shared helper definitions ---------===//
//===----------------------------------------------------------------------===//

#include "PTOToEmitCEmitters.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"
namespace mlir {
namespace pto {

void PTOToEmitCTypeConverter::registerBasicConversions(MLIRContext *Ctx) {
  registerFloatConversions(Ctx);
  registerIntegerConversions(Ctx);
}

// Floating-point and low-precision element types.
void PTOToEmitCTypeConverter::registerFloatConversions(MLIRContext *Ctx) {
// 1. 基本类型 (f32, i32, index)
// ---------------------------------------------------------
addConversion([Ctx](FloatType type) -> Type {
  if (pto::isPTOFloat8E4M3LikeType(type)) {
    return emitc::OpaqueType::get(Ctx, "float8_e4m3_t");
  }
  if (pto::isPTOFloat8E5M2LikeType(type)) {
    return emitc::OpaqueType::get(Ctx, "float8_e5m2_t");
  }
  if (type.isF32()) {
    return emitc::OpaqueType::get(Ctx, "float");
  }
  if (type.isF16()) {
    return emitc::OpaqueType::get(Ctx, "half");
  }
  if (type.isBF16()) {
    return emitc::OpaqueType::get(Ctx, "bfloat16_t");
  }
  if (type.isF64()) {
    return emitc::OpaqueType::get(Ctx, "double");
  }
  llvm::errs() << "[Debug] Unsupported FloatType: " << type << "\n";
  return Type{};
});

addConversion([Ctx](pto::HiF8Type) -> Type {
  return emitc::OpaqueType::get(Ctx, "hifloat8_t");
});
addConversion([Ctx](Type type) -> std::optional<Type> {
  if (isF8E8M0ElemType(type))
    return emitc::OpaqueType::get(Ctx, "float8_e8m0_t");
  return std::nullopt;
});
addConversion([Ctx](pto::F4E1M2x2Type) -> Type {
  return emitc::OpaqueType::get(Ctx, "float4_e1m2x2_t");
});
addConversion([Ctx](pto::F4E2M1x2Type) -> Type {
  return emitc::OpaqueType::get(Ctx, "float4_e2m1x2_t");
});
}

// Integer/index types, including the 64-bit index lowering.
void PTOToEmitCTypeConverter::registerIntegerConversions(MLIRContext *Ctx) {addConversion([Ctx](IntegerType type) -> Type {
  if (type.getWidth() == 1)
    return type;

  // Prefer fixed-width C types. Preserve signedness if the MLIR integer is
  // explicitly signed/unsigned; treat signless as signed by default.
  const bool isUnsigned = type.isUnsignedInteger();
  switch (type.getWidth()) {
  case 8:
    return emitc::OpaqueType::get(Ctx, isUnsigned ? "uint8_t" : "int8_t");
  case 16:
    return emitc::OpaqueType::get(Ctx,
                                  isUnsigned ? "uint16_t" : "int16_t");
  case 32:
    return emitc::OpaqueType::get(Ctx,
                                  isUnsigned ? "uint32_t" : "int32_t");
  case 64:
    return emitc::OpaqueType::get(Ctx,
                                  isUnsigned ? "uint64_t" : "int64_t");
  default:
    llvm::errs() << "[Debug] Unsupported IntegerType width: "
                 << type.getWidth() << "\n";
    return emitc::OpaqueType::get(Ctx, "int32_t"); // Fallback
  }
});

addConversion([Ctx](IndexType type) -> Type {
  return emitc::OpaqueType::get(Ctx, "int64_t");
});

// vector<4xi16> (e.g. TMRGSORT executedNumList) -> pto::MrgSortExecutedNumList
addConversion([Ctx](VectorType type) -> Type {
  if (type.getRank() == 1 && type.getNumElements() == 4 &&
      type.getElementType().isInteger(16))
    return emitc::OpaqueType::get(Ctx, "pto::MrgSortExecutedNumList");
  return Type{};
});

// ---------------------------------------------------------
}

void PTOToEmitCTypeConverter::registerPTOValueConversions(MLIRContext *Ctx) {
  registerPointerAndStructConversions(Ctx);
  registerRuntimeValueConversions(Ctx);
}

// Pipe/event/array/struct/view value types.
void PTOToEmitCTypeConverter::registerPointerAndStructConversions(MLIRContext *Ctx) {
// 2. PTO 特殊类型 (透传或转换)
// ---------------------------------------------------------
addConversion([](emitc::OpaqueType type) { return type; });
addConversion([](emitc::PointerType type) { return type; });

// ---------------------------------------------------------
// 2.5 PtrType 转换 (指针类型)
// ---------------------------------------------------------
addConversion([this, Ctx](pto::PtrType type) -> std::optional<Type> {
  Type elemType = type.getElementType();
  Type newElemType = convertType(elemType);
  if (!newElemType)
    return std::nullopt;

  std::string elemTypeStr;
  if (auto opq = dyn_cast<emitc::OpaqueType>(newElemType)) {
    elemTypeStr = opq.getValue().str();
  } else {
    llvm::errs() << "  [Error] PtrType elem type is not OpaqueType: "
                 << newElemType << "\n";
    return std::nullopt;
  }

  std::string qualifier =
      addrSpaceQualifier(getAddressSpaceOrGM(type.getMemorySpace()));

  return getEmitCPointerType(Ctx, qualifier, elemTypeStr);
});
}

// Runtime handle types (async sessions, views, tile buffers, pipes).
void PTOToEmitCTypeConverter::registerRuntimeValueConversions(MLIRContext *Ctx) {
  registerViewAndHandleConversions(Ctx);
  registerTileConversions(Ctx);
}

// Views, async handles, pipes, events, arrays, structs.
void PTOToEmitCTypeConverter::registerViewAndHandleConversions(MLIRContext *Ctx) {addConversion([Ctx](pto::PipeType type) -> Type {
  (void)type;
  return emitc::OpaqueType::get(Ctx, "auto");
});

addConversion([Ctx](pto::EventIdArrayType type) -> Type {
  std::string tok = "PTOAS_EventIdArray<" + std::to_string(type.getSize()) + ">";
  return emitc::OpaqueType::get(Ctx, tok);
});

// !pto.local_array<D1 x D2 x ... x T> -> !emitc.array<D1 x D2 x ... x T>.
// Variables of this type render as `T a[D1][D2]...;` in the emitted C++.
addConversion([this](pto::LocalArrayType type) -> std::optional<Type> {
  Type convertedElem = convertType(type.getElementType());
  if (!convertedElem)
    return std::nullopt;
  return emitc::ArrayType::get(type.getShape(), convertedElem);
});

// !pto.struct<...> -> !emitc.opaque<"PtoStruct_...">. The matching C++
// `struct PtoStruct_... { ... };` definition is emitted at file scope by
// the pass (see runOnOperation), keyed on the same content-derived name.
// A struct is carried as a pointer to its storage. It cannot be carried by
// value (emitc.member needs an lvalue, so every field write would land in a
// copy), and it cannot be carried as an lvalue either: emitc.func rejects
// an lvalue argument outright, and the C++ emitter refuses one on func.func
// too, which would make a struct impossible to pass to a helper function.
// A pointer is legal in a signature and still names the caller's storage.
addConversion([Ctx](pto::StructType type) -> Type {
  return emitc::PointerType::get(
      emitc::OpaqueType::get(Ctx, getStructTypeName(type)));
});

addConversion([Ctx](pto::AsyncSessionType type) -> Type {
  (void)type;
  return emitc::OpaqueType::get(Ctx, "pto::comm::AsyncSession");
});

addConversion([Ctx](pto::AsyncEventType type) -> Type {
  (void)type;
  return emitc::OpaqueType::get(Ctx, "pto::comm::AsyncEvent");
});

addConversion([Ctx](pto::PrefetchAsyncContextType type) -> Type {
  (void)type;
  return emitc::OpaqueType::get(Ctx, "pto::PrefetchAsyncContext");
});

addConversion([Ctx](pto::TensorViewType type) -> Type {
  std::string layout = type.getLayoutAttr()
                           ? layoutToEmitCString(
                                 type.getLayoutAttr().getLayout())
                           : "pto::Layout::ND";
  return getRuntimeGlobalTensorOpaqueType(Ctx, type.getElementType(),
                                          type.getShape(), layout);
});

addConversion([Ctx](pto::PartitionTensorViewType type) -> Type {
  std::string layout = type.getLayoutAttr()
                           ? layoutToEmitCString(
                                 type.getLayoutAttr().getLayout())
                           : "pto::Layout::ND";
  return getRuntimeGlobalTensorOpaqueType(Ctx, type.getElementType(),
                                          type.getShape(), layout);
});
}

// Tile buffer types (tile_buf/alloc_tile handles).
void PTOToEmitCTypeConverter::registerTileConversions(MLIRContext *Ctx) {addConversion([Ctx](pto::TileBufType type) -> std::optional<Type> {
  auto typeString = getEmitCTileTypeString(type);
  if (!typeString)
    return std::nullopt;
  return emitc::OpaqueType::get(Ctx, *typeString);
});

// ---------------------------------------------------------
}

void PTOToEmitCTypeConverter::registerMemRefConversions(MLIRContext *Ctx) {
// 3. MemRef 转换 (Debug 重点)
// ---------------------------------------------------------
addConversion([this, Ctx](MemRefType type) -> std::optional<Type> {
  LLVM_DEBUG(llvm::dbgs() << "Converting MemRef: " << type << "\n");

  // A. 转换元素类型
  Type elemType = type.getElementType();
  Type newElemType = convertType(elemType); 
  if (!newElemType) {
    llvm::errs() << "  [Error] Failed to convert element type: " << elemType << "\n";
    return std::nullopt;
  }
  
  // 获取元素类型的字符串
  std::string elemTypeStr;
  if (auto opq = dyn_cast<emitc::OpaqueType>(newElemType)) {
    elemTypeStr = opq.getValue().str();
  } else {
     llvm::errs() << "  [Error] Converted element type is not OpaqueType: " << newElemType << "\n";
     return std::nullopt;
  }

  // B. 处理 Memory Space
  std::string qualifier = "";
  Attribute memorySpace = type.getMemorySpace();
  
  if (!memorySpace) {
     qualifier = "__gm__";
  } else if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(memorySpace)) {
     qualifier = addrSpaceQualifier(ptoAttr.getAddressSpace());
  } else {
     llvm::errs() << "  [Warning] Unknown MemorySpace Attribute type: " << memorySpace << "\n";
     qualifier = "__gm__"; // Fallback
  }

  std::string finalTypeStr = qualifier + " " + elemTypeStr;
  LLVM_DEBUG(llvm::dbgs() << "  [Success] -> " << finalTypeStr << "*\n");
  
  return getEmitCPointerType(Ctx, finalTypeStr);
});

// ---------------------------------------------------------

}

void PTOToEmitCTypeConverter::registerFunctionAndMaterializations() {
// 4. Function & Materialization
// ---------------------------------------------------------
addConversion([this](FunctionType type) -> Type {
  SmallVector<Type> inputs;
  if (failed(convertTypes(type.getInputs(), inputs))) {
    return Type{};
  }
  SmallVector<Type> results;
  if (failed(convertTypes(type.getResults(), results))) {
    return Type{};
  }
  return FunctionType::get(type.getContext(), inputs, results);
});

auto materializeCast = [](OpBuilder &Builder, Type ResultType,
                          ValueRange Inputs, Location Loc) -> Value {
  if (Inputs.size() != 1) {
    return Value();
  }
  return Builder.create<UnrealizedConversionCastOp>(Loc, ResultType, Inputs[0]).getResult(0);
};

  addSourceMaterialization(materializeCast);
  addTargetMaterialization(materializeCast);
}

} // namespace pto
} // namespace mlir
