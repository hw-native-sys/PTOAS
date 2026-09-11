// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

LogicalResult mlir::pto::AddPtrOp::verify()
{
    Value ptr = getOperation()->getOperand(0);
    Value result = getOperation()->getResult(0);

    auto ptrTy = dyn_cast<mlir::pto::PtrType>(ptr.getType());
    if (!ptrTy) {
        return emitOpError("ptr operand must be !pto.ptr<...>");
    }

    auto resTy = dyn_cast<mlir::pto::PtrType>(result.getType());
    if (!resTy) {
        return emitOpError("result must be !pto.ptr<...>");
    }

    if (ptrTy != resTy) {
        return emitOpError("result type must match ptr operand type");
    }

    return success();
}

static Type getPointerLikeElementType(Type type)
{
    if (auto ptrTy = dyn_cast<mlir::pto::PtrType>(type)) {
        return ptrTy.getElementType();
    }
    return Type();
}

static bool isEmitCSupportedScalarType(Type type)
{
    if (!type) {
        return false;
    }
    if (type.isF16() || type.isBF16() || type.isF32() || type.isF64()) {
        return true;
    }
    if (auto intTy = dyn_cast<IntegerType>(type)) {
        return intTy.getWidth() == 8 || intTy.getWidth() == 16 || intTy.getWidth() == 32 || intTy.getWidth() == 64;
    }
    if (mlir::pto::isPTOFloat8Type(type)) {
        return true;
    }
    if (isa<mlir::pto::HiF8Type, mlir::pto::F4E1M2x2Type, mlir::pto::F4E2M1x2Type>(type)) {
        return true;
    }
    return false;
}

LogicalResult mlir::pto::PtrToIntOp::verify()
{
    Type resultTy = getResult().getType();
    auto intTy = dyn_cast<IntegerType>(resultTy);
    if (!intTy || intTy.getWidth() != 64) {
        return emitOpError("result must be i64");
    }

    if (!isa<mlir::pto::PtrType>(getPtr().getType())) {
        return emitOpError("ptr operand must be !pto.ptr<...>");
    }
    return success();
}

LogicalResult mlir::pto::IntToPtrOp::verify()
{
    auto addrTy = dyn_cast<IntegerType>(getAddr().getType());
    if (!addrTy || addrTy.getWidth() != 64) {
        return emitOpError("address operand must be i64");
    }

    if (!isa<mlir::pto::PtrType>(getResult().getType())) {
        return emitOpError("result must be !pto.ptr<...>");
    }

    Type dstElem = getPointerLikeElementType(getResult().getType());
    if (!isEmitCSupportedScalarType(dstElem)) {
        return emitOpError("result element type is not supported by EmitC: ") << dstElem;
    }

    return success();
}

LogicalResult mlir::pto::LocalArrayGetOp::verify()
{
    auto arrayTy = getArray().getType();
    int64_t rank = arrayTy.getRank();
    int64_t numIdx = static_cast<int64_t>(getIndices().size());
    if (numIdx != rank) {
        return emitOpError() << "expects " << rank << " indices for !pto.local_array of rank " << rank << ", got "
                             << numIdx;
    }
    if (getResult().getType() != arrayTy.getElementType()) {
        return emitOpError() << "result type " << getResult().getType() << " does not match array element type "
                             << arrayTy.getElementType();
    }
    return success();
}

LogicalResult mlir::pto::LocalArraySetOp::verify()
{
    auto arrayTy = getArray().getType();
    int64_t rank = arrayTy.getRank();
    int64_t numIdx = static_cast<int64_t>(getIndices().size());
    if (numIdx != rank) {
        return emitOpError() << "expects " << rank << " indices for !pto.local_array of rank " << rank << ", got "
                             << numIdx;
    }
    if (getValue().getType() != arrayTy.getElementType()) {
        return emitOpError() << "value type " << getValue().getType() << " does not match array element type "
                             << arrayTy.getElementType();
    }
    return success();
}

// Resolve the field type reached by following a constant `path` of field
// indices from `root`, descending through nested structs. Emits an actionable
// op error and returns failure on an empty path, an out-of-range index, or a
// descent into a non-struct field. On success writes the terminal field type to
// `fieldTyOut`.
static LogicalResult walkStructPath(
    Operation* op, mlir::pto::StructType root, llvm::ArrayRef<int64_t> path, Type& fieldTyOut)
{
    if (path.empty()) {
        return op->emitOpError() << "struct path must have at least one index";
    }
    Type cur = root;
    for (auto [depth, idx] : llvm::enumerate(path)) {
        auto st = dyn_cast<mlir::pto::StructType>(cur);
        if (!st) {
            return op->emitOpError() << "struct path index " << depth << " descends into non-struct field of type "
                                     << cur;
        }
        if (idx < 0 || idx >= static_cast<int64_t>(st.getNumFields())) {
            return op->emitOpError() << "struct path index " << depth << " (" << idx << ") is out of range for " << st
                                     << " with " << st.getNumFields() << " field(s)";
        }
        cur = st.getFieldType(static_cast<unsigned>(idx));
    }
    fieldTyOut = cur;
    return success();
}

// The declared struct is stack storage owned by the enclosing scope, and the
// value lowers to a pointer to that storage. Letting it reach a terminator
// would publish that address outside the owning scope: `return %s` hands the
// caller a pointer into a dead frame, and `scf.yield %s` carries it out of the
// region that owns it. Both are rejected here rather than emitted as C++ that
// looks fine and is undefined at run time.
LogicalResult mlir::pto::DeclareStructOp::verify()
{
    for (Operation* user : getResult().getUsers()) {
        if (!user->hasTrait<mlir::OpTrait::IsTerminator>()) {
            continue;
        }
        return emitOpError() << "stack-local struct must not escape the scope that declares it, "
                                "but its value is passed to '"
                             << user->getName()
                             << "', which would expose the address of storage that is about to "
                                "die; declare the struct in the outer scope and mutate it from "
                                "the nested region instead (pto.struct_set mutates in place, "
                                "so a struct never needs to be returned or yielded)";
    }
    return success();
}

// Both accessors bottom out at a scalar. A path ending on a nested !pto.struct
// is rejected: the member chain lowers to `emitc.member`, which yields an
// lvalue, and handing a whole aggregate back as an SSA value would mean copying
// it out of the struct — so reaching into a nested struct is spelled as a longer
// path instead.
static LogicalResult verifyStructLeafIsScalar(Operation* op, Type fieldTy)
{
    if (!fieldTy.isIntOrFloat()) {
        return op->emitOpError() << "struct path must end at a scalar field, but ends at " << fieldTy
                                 << "; extend the path to reach a scalar inside it";
    }
    return success();
}

template <typename PathRange>
static LogicalResult verifyStructAccess(Operation *op, PathRange path,
                                        Type valueType, StringRef valueLabel) {
  Type fieldTy;
  if (failed(walkStructPath(
          op, cast<mlir::pto::StructType>(op->getOperand(0).getType()), path,
          fieldTy)) ||
      failed(verifyStructLeafIsScalar(op, fieldTy)))
    return failure();
  if (valueType != fieldTy)
    return op->emitOpError()
           << valueLabel << " type " << valueType << " does not match field type "
           << fieldTy << " at the given path";
  return success();
}

LogicalResult mlir::pto::StructGetOp::verify()
{
    return verifyStructAccess(getOperation(), getPath(), getValue().getType(),
                              "result");
}

LogicalResult mlir::pto::StructSetOp::verify()
{
    return verifyStructAccess(getOperation(), getPath(), getValue().getType(),
                              "value");
}

LogicalResult mlir::pto::CastPtrOp::verify()
{
    Type inputType = getInput().getType();
    Type resultType = getResult().getType();

    auto inputPtrType = dyn_cast<mlir::pto::PtrType>(inputType);
    auto resultPtrType = dyn_cast<mlir::pto::PtrType>(resultType);
    auto inputMemRefType = dyn_cast<BaseMemRefType>(inputType);
    bool inputIsInteger = isa<IntegerType>(inputType);
    bool resultIsInteger = isa<IntegerType>(resultType);

    if (!inputPtrType && !inputMemRefType && !inputIsInteger) {
        return emitOpError("input must be an integer, memref, or !pto.ptr<...>");
    }
    if (!resultPtrType && !resultIsInteger) {
        return emitOpError("result must be an integer or !pto.ptr<...>");
    }

    if (inputIsInteger && resultIsInteger) {
        return emitOpError("integer-to-integer cast is not a ptr cast");
    }

    if (inputMemRefType && resultIsInteger) {
        return emitOpError("memref-to-integer cast is unsupported");
    }

    if (inputMemRefType && resultPtrType) {
        auto memrefSpace = dyn_cast_or_null<mlir::pto::AddressSpaceAttr>(inputMemRefType.getMemorySpace());
        auto resultSpace = resultPtrType.getMemorySpace();
        if (memrefSpace && memrefSpace != resultSpace) {
            return emitOpError("memref-to-ptr cast must stay within the same PTO memory space");
        }
    }

    if (inputPtrType && resultPtrType && inputPtrType.getMemorySpace() != resultPtrType.getMemorySpace()) {
        return emitOpError("ptr-to-ptr cast must stay within the same PTO memory space");
    }

    return success();
}

void PTODialect::initialize()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "PTO/IR/PTOTypeDefs.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "PTO/IR/PTOOps.cpp.inc"
        >();

    addAttributes<
#define GET_ATTRDEF_LIST
#include "PTO/IR/PTOAttrs.cpp.inc"
        >();

    addInterfaces<PTOInlinerInterface>();
}
