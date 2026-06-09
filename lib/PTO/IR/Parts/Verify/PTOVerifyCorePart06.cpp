// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOVerifyCore.cpp; kept as a fragment included by PTOVerifyCore.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

LogicalResult mlir::pto::PartitionViewOp::verify() {
  auto srcTy = dyn_cast<mlir::pto::TensorViewType>(getSource().getType());
  auto resTy = dyn_cast<mlir::pto::PartitionTensorViewType>(getResult().getType());
  if (!srcTy || !resTy)
    return emitOpError("expects tensor_view source and partition_tensor_view result");
  if (srcTy.getElementType() != resTy.getElementType()) {
    return emitOpError() << "element type mismatch between source and result: src="
                         << srcTy.getElementType() << " result="
                         << resTy.getElementType();
  }

  int64_t srcRank = srcTy.getRank();
  if (static_cast<int64_t>(getOffsets().size()) != srcRank)
    return emitOpError() << "offset count (" << getOffsets().size()
                         << ") must match source rank (" << srcRank << ")";
  if (static_cast<int64_t>(getSizes().size()) != srcRank)
    return emitOpError() << "size count (" << getSizes().size()
                         << ") must match source rank (" << srcRank << ")";

  ArrayRef<int64_t> srcShape = srcTy.getShape();
  ArrayRef<int64_t> resShape = resTy.getShape();
  bool sameRank = resTy.getRank() == srcRank;
  for (int64_t i = 0; i < srcRank; ++i) {
    if (failed(
            verifyPartitionViewDimension(*this, i, srcShape, resShape, sameRank)))
      return failure();
  }
  return success();
}

LogicalResult mlir::pto::AddPtrOp::verify() {
  Value ptr = getOperation()->getOperand(0);
  Value result = getOperation()->getResult(0);

  auto ptrTy = dyn_cast<mlir::pto::PtrType>(ptr.getType());
  if (!ptrTy)
    return emitOpError("ptr operand must be !pto.ptr<...>");

  auto resTy = dyn_cast<mlir::pto::PtrType>(result.getType());
  if (!resTy)
    return emitOpError("result must be !pto.ptr<...>");

  if (ptrTy != resTy)
    return emitOpError("result type must match ptr operand type");

  return success();
}

static LogicalResult verifyPtrLikeForAddressCast(Operation *op, Type type,
                                                 StringRef name) {
  if (isa<mlir::pto::PtrType>(type))
    return success();

  auto memTy = dyn_cast<MemRefType>(type);
  if (!memTy)
    return op->emitOpError()
           << "expects " << name << " to be !pto.ptr<...> or a GM memref";

  if (memTy.getRank() != 1)
    return op->emitOpError()
           << "expects lowered memref " << name << " to be rank-1";

  if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
    return op->emitOpError()
           << "expects lowered memref " << name << " to use GM address space";

  return success();
}

static Type getPointerLikeElementType(Type type) {
  if (auto ptrTy = dyn_cast<mlir::pto::PtrType>(type))
    return ptrTy.getElementType();
  if (auto memTy = dyn_cast<MemRefType>(type))
    return memTy.getElementType();
  return Type();
}

static bool isEmitCSupportedScalarType(Type type) {
  if (!type)
    return false;
  if (type.isF16() || type.isBF16() || type.isF32() || type.isF64())
    return true;
  if (auto intTy = dyn_cast<IntegerType>(type))
    return intTy.getWidth() == kPTOI8BitWidth || intTy.getWidth() == kPTOI16BitWidth ||
           intTy.getWidth() == kPTOI32BitWidth || intTy.getWidth() == kPTOI64BitWidth;
  if (mlir::pto::isPTOFloat8Type(type))
    return true;
  if (isa<mlir::pto::HiF8Type, mlir::pto::F4E1M2x2Type,
          mlir::pto::F4E2M1x2Type>(type))
    return true;
  return false;
}

LogicalResult mlir::pto::PtrToIntOp::verify() {
  Type resultTy = getResult().getType();
  auto intTy = dyn_cast<IntegerType>(resultTy);
  if (!intTy || intTy.getWidth() != kPTOI64BitWidth)
    return emitOpError("result must be i64");

  return verifyPtrLikeForAddressCast(getOperation(), getPtr().getType(),
                                     "ptr operand");
}

LogicalResult mlir::pto::IntToPtrOp::verify() {
  auto addrTy = dyn_cast<IntegerType>(getAddr().getType());
  if (!addrTy || addrTy.getWidth() != kPTOI64BitWidth)
    return emitOpError("address operand must be i64");

  if (failed(verifyPtrLikeForAddressCast(getOperation(), getResult().getType(),
                                         "result")))
    return failure();

  Type dstElem = getPointerLikeElementType(getResult().getType());
  if (!isEmitCSupportedScalarType(dstElem))
    return emitOpError("result element type is not supported by EmitC: ")
           << dstElem;

  return success();
}

LogicalResult mlir::pto::LocalArrayGetOp::verify() {
  auto arrayTy = getArray().getType();
  int64_t rank = arrayTy.getRank();
  int64_t numIdx = static_cast<int64_t>(getIndices().size());
  if (numIdx != rank)
    return emitOpError() << "expects " << rank
                         << " indices for !pto.local_array of rank " << rank
                         << ", got " << numIdx;
  if (getResult().getType() != arrayTy.getElementType())
    return emitOpError()
           << "result type " << getResult().getType()
           << " does not match array element type "
           << arrayTy.getElementType();
  return success();
}

LogicalResult mlir::pto::LocalArraySetOp::verify() {
  auto arrayTy = getArray().getType();
  int64_t rank = arrayTy.getRank();
  int64_t numIdx = static_cast<int64_t>(getIndices().size());
  if (numIdx != rank)
    return emitOpError() << "expects " << rank
                         << " indices for !pto.local_array of rank " << rank
                         << ", got " << numIdx;
  if (getValue().getType() != arrayTy.getElementType())
    return emitOpError() << "value type " << getValue().getType()
                         << " does not match array element type "
                         << arrayTy.getElementType();
  return success();
}
AddressSpaceAttr mlir::pto::getPTOAddressSpaceAttr(Type type) {
  auto memRefType = dyn_cast<BaseMemRefType>(type);
  if (!memRefType)
    return {};
  auto scopeAttr = dyn_cast<AddressSpaceAttr>(memRefType.getMemorySpace());
  if (!scopeAttr)
    return {};
  return scopeAttr;
}

bool mlir::pto::isScalarPtrOrMemRef(Type type) {
  if (auto pty = dyn_cast<mlir::pto::PtrType>(type))
    return true;
  if (auto memTy = dyn_cast<MemRefType>(type))
    return isGmAddressSpaceAttr(memTy.getMemorySpace());
  return false;
}

bool mlir::pto::hasExplicitPTOEntryAttr(func::FuncOp func) {
  return func && (func->hasAttrOfType<UnitAttr>(kPTOEntryAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kLegacyHACCEntryAttrName));
}

static constexpr StringLiteral kEffectivePTOEntryAttrName =
    "pto.internal.entry";

static SmallVector<func::FuncOp> getPTOFunctionDefinitions(ModuleOp module) {
  SmallVector<func::FuncOp> defs;
  if (!module)
    return defs;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (!func.isDeclaration())
      defs.push_back(func);
  }
  return defs;
}

bool mlir::pto::isPTOEntryFunction(func::FuncOp func) {
  if (!func || func.isDeclaration())
    return false;
  if (auto attr = func->getAttrOfType<BoolAttr>(kEffectivePTOEntryAttrName))
    return attr.getValue();
  if (hasExplicitPTOEntryAttr(func))
    return true;

  ModuleOp module = func->getParentOfType<ModuleOp>();
  if (!module)
    return false;
  SmallVector<func::FuncOp> defs = getPTOFunctionDefinitions(module);
  return defs.size() == 1 && defs.front() == func;
}

LogicalResult mlir::pto::validatePTOEntryFunctions(ModuleOp module) {
  if (!module)
    return success();

  for (auto func : module.getOps<func::FuncOp>()) {
    if (!hasExplicitPTOEntryAttr(func))
      continue;
    if (func.isDeclaration()) {
      return func.emitOpError()
             << "`" << kPTOEntryAttrName
             << "` is only valid on function definitions";
    }
  }

  for (auto func : module.getOps<func::FuncOp>()) {
    if (!isPTOEntryFunction(func))
      continue;
    if (func.getFunctionType().getNumResults() != 0) {
      return func.emitOpError()
             << "PTO entry functions must return void";
    }
  }
  return success();
}
