//===- A5VM.cpp - A5VM dialect -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/A5VM.h"

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::pto::a5vm;

#define GET_TYPEDEF_CLASSES
#include "PTO/IR/A5VMTypes.cpp.inc"

#include "PTO/IR/A5VMDialect.cpp.inc"

static std::string formatVecType(int64_t elementCount, Type elementType) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << "!a5vm.vec<" << elementCount << "x" << elementType << ">";
  return storage;
}

static LogicalResult verifyVecTypeLike(Operation *op, Type type,
                                       StringRef roleDescription) {
  auto vecType = dyn_cast<VecType>(type);
  if (!vecType)
    return op->emitOpError() << roleDescription << " must be !a5vm.vec<...>";

  return VecType::verify(
      [&]() { return op->emitOpError() << roleDescription << " "; },
      vecType.getElementCount(), vecType.getElementType());
}

Type VecType::parse(AsmParser &parser) {
  SmallVector<int64_t, 1> shape;
  Type elementType;
  SMLoc loc = parser.getCurrentLocation();

  if (failed(parser.parseLess()) ||
      failed(parser.parseDimensionList(shape, /*allowDynamic=*/false,
                                       /*withTrailingX=*/true)) ||
      shape.size() != 1 || failed(parser.parseType(elementType)) ||
      failed(parser.parseGreater()))
    return {};

  return parser.getChecked<VecType>(loc, parser.getContext(), shape.front(),
                                    elementType);
}

void VecType::print(AsmPrinter &printer) const {
  printer << "<" << getElementCount() << "x";
  printer.printType(getElementType());
  printer << ">";
}

LogicalResult VecType::verify(function_ref<InFlightDiagnostic()> emitError,
                              int64_t elementCount, Type elementType) {
  if (elementCount <= 0)
    return emitError() << "'" << formatVecType(elementCount, elementType)
                       << "' expected a positive element count";

  auto intOrFloat = mlir::dyn_cast<IntegerType>(elementType);
  unsigned elementBitWidth = 0;
  if (intOrFloat) {
    elementBitWidth = intOrFloat.getWidth();
  } else if (auto floatType = mlir::dyn_cast<FloatType>(elementType)) {
    elementBitWidth = floatType.getWidth();
  } else {
    return emitError() << "'" << formatVecType(elementCount, elementType)
                       << "' expected an integer or floating-point element type";
  }

  if (elementCount * static_cast<int64_t>(elementBitWidth) != 2048)
    return emitError() << "'" << formatVecType(elementCount, elementType)
                       << "' expected exactly 256 bytes";

  return success();
}

void A5VMDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "PTO/IR/A5VMTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "PTO/IR/A5VMOps.cpp.inc"
      >();
}

Attribute A5VMDialect::parseAttribute(DialectAsmParser &parser,
                                      Type type) const {
  parser.emitError(parser.getCurrentLocation(),
                   "A5VM dialect defines no custom attributes");
  return {};
}

void A5VMDialect::printAttribute(Attribute attr,
                                 DialectAsmPrinter &printer) const {
  llvm_unreachable("A5VM dialect defines no custom attributes");
}

void LoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getBaseMutable());
}

LogicalResult LoadOp::verify() {
  if (failed(verifyVecTypeLike(*this, getResult().getType(), "result type")))
    return failure();

  if (!getLayoutAttr() || !getDomainAttr())
    return emitOpError("requires layout and domain string attributes");
  if (!getValidRowsAttr() || !getValidColsAttr())
    return emitOpError("requires valid_rows and valid_cols integer attributes");

  return success();
}

void StoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getBaseMutable());
}

LogicalResult AbsOp::verify() {
  if (failed(verifyVecTypeLike(*this, getInput().getType(), "operand type")))
    return failure();
  if (failed(verifyVecTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (getInput().getType() != getResult().getType())
    return emitOpError("mismatched vector types");
  return success();
}

LogicalResult StoreOp::verify() {
  if (failed(verifyVecTypeLike(*this, getValue().getType(), "value type")))
    return failure();
  if (!getLayoutAttr() || !getDomainAttr())
    return emitOpError("requires layout and domain string attributes");
  return success();
}

#define GET_OP_CLASSES
#include "PTO/IR/A5VMOps.cpp.inc"
