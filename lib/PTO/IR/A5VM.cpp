//===- A5VM.cpp - A5VM dialect -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/A5VM.h"
#include "PTO/IR/PTO.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
using namespace mlir::a5vm;

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

enum class MemoryRole {
  Unknown,
  GM,
  UB,
  Other,
};

static MemoryRole classifyMemoryRole(Type type) {
  auto memrefType = dyn_cast<BaseMemRefType>(type);
  if (!memrefType) {
    if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(type)) {
      switch (ptrType.getAddressSpace()) {
      case static_cast<unsigned>(pto::AddressSpace::GM):
      case static_cast<unsigned>(pto::AddressSpace::Zero):
        return MemoryRole::GM;
      case static_cast<unsigned>(pto::AddressSpace::VEC):
        return MemoryRole::UB;
      default:
        return MemoryRole::Other;
      }
    }
    return MemoryRole::Other;
  }

  Attribute memorySpace = memrefType.getMemorySpace();
  if (!memorySpace)
    return MemoryRole::Unknown;

  if (auto addrSpace = dyn_cast<pto::AddressSpaceAttr>(memorySpace)) {
    switch (addrSpace.getAddressSpace()) {
    case pto::AddressSpace::GM:
    case pto::AddressSpace::Zero:
      return MemoryRole::GM;
    case pto::AddressSpace::VEC:
      return MemoryRole::UB;
    default:
      return MemoryRole::Other;
    }
  }

  if (auto intAttr = dyn_cast<IntegerAttr>(memorySpace)) {
    switch (intAttr.getInt()) {
    case static_cast<int64_t>(pto::AddressSpace::GM):
    case static_cast<int64_t>(pto::AddressSpace::Zero):
      return MemoryRole::GM;
    case static_cast<int64_t>(pto::AddressSpace::VEC):
      return MemoryRole::UB;
    default:
      return MemoryRole::Other;
    }
  }

  return MemoryRole::Other;
}

static bool isBufferLike(Type type) {
  return isa<BaseMemRefType, LLVM::LLVMPointerType>(type);
}

static LogicalResult verifySyncToken(Operation *op, StringAttr token,
                                     StringRef role) {
  if (!token || token.getValue().empty())
    return op->emitOpError() << "requires non-empty " << role;
  return success();
}

template <typename CopyOp>
static LogicalResult verifyCopyGmToUbufOp(CopyOp op, bool expectSourceGM) {
  if (!isBufferLike(op.getSource().getType()) ||
      !isBufferLike(op.getDestination().getType()))
    return op.emitOpError("requires pointer-like source and destination");

  bool hasAllMetadata =
      op.getLayoutAttr() && op.getDataSelectBitAttr() && op.getUbPadAttr();

  MemoryRole sourceRole = classifyMemoryRole(op.getSource().getType());
  MemoryRole destinationRole = classifyMemoryRole(op.getDestination().getType());
  bool directionMatches = true;
  if (expectSourceGM) {
    directionMatches &= sourceRole != MemoryRole::UB;
    directionMatches &= destinationRole != MemoryRole::GM;
  } else {
    directionMatches &= sourceRole != MemoryRole::GM;
    directionMatches &= destinationRole != MemoryRole::UB;
  }

  if (!hasAllMetadata || !directionMatches) {
    return op.emitOpError()
           << "requires "
           << (expectSourceGM ? "GM source, UB destination"
                              : "UB source, GM destination")
           << ", and complete transfer metadata";
  }

  return success();
}

template <typename CopyOp>
static LogicalResult verifyCopyUbufToGmOp(CopyOp op, bool expectSourceGM) {
  if (!isBufferLike(op.getSource().getType()) ||
      !isBufferLike(op.getDestination().getType()))
    return op.emitOpError("requires pointer-like source and destination");

  bool hasAllMetadata = op.getLayoutAttr();

  MemoryRole sourceRole = classifyMemoryRole(op.getSource().getType());
  MemoryRole destinationRole = classifyMemoryRole(op.getDestination().getType());
  bool directionMatches = true;
  if (expectSourceGM) {
    directionMatches &= sourceRole != MemoryRole::UB;
    directionMatches &= destinationRole != MemoryRole::GM;
  } else {
    directionMatches &= sourceRole != MemoryRole::GM;
    directionMatches &= destinationRole != MemoryRole::UB;
  }

  if (!hasAllMetadata || !directionMatches) {
    return op.emitOpError()
           << "requires "
           << (expectSourceGM ? "GM source, UB destination"
                              : "UB source, GM destination")
           << ", and complete transfer metadata";
  }

  return success();
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

void CopyGmToUbufOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult CopyGmToUbufOp::verify() {
  return verifyCopyGmToUbufOp(*this, true);
}

LogicalResult SetFlagOp::verify() {
  if (failed(verifySyncToken(*this, getSrcPipeAttr(), "src_pipe")) ||
      failed(verifySyncToken(*this, getDstPipeAttr(), "dst_pipe")) ||
      failed(verifySyncToken(*this, getEventIdAttr(), "event_id")))
    return failure();
  return success();
}

LogicalResult WaitFlagOp::verify() {
  if (failed(verifySyncToken(*this, getSrcPipeAttr(), "src_pipe")) ||
      failed(verifySyncToken(*this, getDstPipeAttr(), "dst_pipe")) ||
      failed(verifySyncToken(*this, getEventIdAttr(), "event_id")))
    return failure();
  return success();
}

LogicalResult PipeBarrierOp::verify() {
  return verifySyncToken(*this, getPipeAttr(), "pipe");
}

void VldsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VldsOp::verify() {
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a pointer-like source");

  if (failed(verifyVecTypeLike(*this, getResult().getType(), "result type")))
    return failure();

  MemoryRole sourceRole = classifyMemoryRole(getSource().getType());
  if (sourceRole == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");

  return success();
}

LogicalResult VabsOp::verify() {
  if (failed(verifyVecTypeLike(*this, getInput().getType(), "operand type")))
    return failure();
  if (failed(verifyVecTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (getInput().getType() != getResult().getType())
    return emitOpError("requires matching register vector shape");
  return success();
}

template <typename UnaryOp>
static LogicalResult verifyUnaryVecOp(UnaryOp op) {
  if (failed(verifyVecTypeLike(op, op.getInput().getType(), "operand type")))
    return failure();
  if (failed(verifyVecTypeLike(op, op.getResult().getType(), "result type")))
    return failure();
  if (op.getInput().getType() != op.getResult().getType())
    return op.emitOpError("requires matching register vector shape");
  return success();
}

LogicalResult VexpOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VlnOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VsqrtOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VrecOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VreluOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VnotOp::verify() { return verifyUnaryVecOp(*this); }

template <typename BinaryOp>
static LogicalResult verifyBinaryVecOp(BinaryOp op) {
  if (failed(verifyVecTypeLike(op, op.getLhs().getType(), "lhs type")))
    return failure();
  if (failed(verifyVecTypeLike(op, op.getRhs().getType(), "rhs type")))
    return failure();
  if (failed(verifyVecTypeLike(op, op.getResult().getType(), "result type")))
    return failure();
  if (op.getLhs().getType() != op.getRhs().getType() ||
      op.getLhs().getType() != op.getResult().getType())
    return op.emitOpError("requires matching register vector shapes");
  return success();
}

LogicalResult VaddOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VsubOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VmulOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VdivOp::verify() { return verifyBinaryVecOp(*this); }

void VstsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VstsOp::verify() {
  if (failed(verifyVecTypeLike(*this, getValue().getType(), "value type")))
    return failure();

  if (!isBufferLike(getDestination().getType()))
    return emitOpError("requires a pointer-like destination");

  MemoryRole destinationRole = classifyMemoryRole(getDestination().getType());
  if (destinationRole == MemoryRole::GM)
    return emitOpError("requires a UB-backed destination");

  return success();
}

void CopyUbufToGmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult CopyUbufToGmOp::verify() {
  return verifyCopyUbufToGmOp(*this, false);
}

#define GET_OP_CLASSES
#include "PTO/IR/A5VMOps.cpp.inc"
