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
#include "llvm/ADT/APFloat.h"
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

static LogicalResult verifyMaskTypeLike(Operation *op, Type type,
                                        StringRef roleDescription) {
  if (!isa<MaskType>(type))
    return op->emitOpError() << roleDescription << " must be !a5vm.mask";
  return success();
}

static LogicalResult verifyAlignTypeLike(Operation *op, Type type,
                                         StringRef roleDescription) {
  if (!isa<AlignType>(type))
    return op->emitOpError() << roleDescription << " must be !a5vm.align";
  return success();
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

LogicalResult VbrOp::verify() {
  if (failed(verifyVecTypeLike(*this, getResult().getType(), "result")))
    return failure();

  auto resultVecType = cast<VecType>(getResult().getType());
  Type elementType = getValue().getType();
  if (isa<ShapedType, VectorType>(elementType))
    return emitOpError("value must be a scalar matching the result element type");
  if (elementType != resultVecType.getElementType())
    return emitOpError("value type must match result element type");
  return success();
}

LogicalResult VcaddOp::verify() {
  if (failed(verifyVecTypeLike(*this, getInput().getType(), "input")) ||
      failed(verifyVecTypeLike(*this, getResult().getType(), "result")))
    return failure();
  if (getInput().getType() != getResult().getType())
    return emitOpError("input and result must have the same vector type");
  return success();
}

LogicalResult VcmaxOp::verify() {
  if (failed(verifyVecTypeLike(*this, getInput().getType(), "input")) ||
      failed(verifyVecTypeLike(*this, getResult().getType(), "result")))
    return failure();
  if (getInput().getType() != getResult().getType())
    return emitOpError("input and result must have the same vector type");
  return success();
}

LogicalResult VcminOp::verify() {
  if (failed(verifyVecTypeLike(*this, getInput().getType(), "input")) ||
      failed(verifyVecTypeLike(*this, getResult().getType(), "result")))
    return failure();
  if (getInput().getType() != getResult().getType())
    return emitOpError("input and result must have the same vector type");
  return success();
}

LogicalResult VciOp::verify() {
  auto resultType = dyn_cast<VecType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be !a5vm.vec<...>");
  if (!isa<IntegerType>(resultType.getElementType()))
    return emitOpError("result element type must be integer");
  auto indexType = dyn_cast<IntegerType>(getIndex().getType());
  if (!indexType)
    return emitOpError("index must be an integer scalar");
  if (indexType != resultType.getElementType())
    return emitOpError("index type must match result element type");
  return success();
}

void Vgather2Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult Vgather2Op::verify() {
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a pointer-like source");
  MemoryRole sourceRole = classifyMemoryRole(getSource().getType());
  if (sourceRole == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");

  auto offsetsType = dyn_cast<VecType>(getOffsets().getType());
  auto resultType = dyn_cast<VecType>(getResult().getType());
  if (!offsetsType || !resultType)
    return emitOpError("offsets and result must be !a5vm.vec<...>");
  if (!isa<IntegerType>(offsetsType.getElementType()))
    return emitOpError("offset vector must use integer element type");
  if (offsetsType.getElementCount() != resultType.getElementCount())
    return emitOpError("offset and result vectors must have the same element count");
  if (!getActiveLanes().getType().isIndex())
    return emitOpError("active_lanes must be index");
  return success();
}

LogicalResult CopyUbufToUbufOp::verify() {
  if (!isBufferLike(getSource().getType()) || !isBufferLike(getDestination().getType()))
    return emitOpError("requires pointer-like source and destination");
  if (classifyMemoryRole(getSource().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getDestination().getType()) != MemoryRole::UB)
    return emitOpError("requires UB-backed source and destination");
  return success();
}

void VgatherbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VgatherbOp::verify() {
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a pointer-like source");
  MemoryRole sourceRole = classifyMemoryRole(getSource().getType());
  if (sourceRole == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");

  auto offsetsType = dyn_cast<VecType>(getOffsets().getType());
  auto resultType = dyn_cast<VecType>(getResult().getType());
  if (!offsetsType || !resultType)
    return emitOpError("offsets and result must be !a5vm.vec<...>");
  auto offsetsElemType = dyn_cast<IntegerType>(offsetsType.getElementType());
  if (!offsetsElemType)
    return emitOpError("offset vector must use integer element type");
  if (offsetsElemType.getWidth() != 32)
    return emitOpError("currently requires 32-bit offset vector elements");
  if (offsetsType.getElementCount() != resultType.getElementCount())
    return emitOpError("offset and result vectors must have the same element count");
  if (!getActiveLanes().getType().isIndex())
    return emitOpError("active_lanes must be index");
  return success();
}

LogicalResult VbitsortOp::verify() {
  if (!isBufferLike(getDestination().getType()) || !isBufferLike(getSource().getType()) ||
      !isBufferLike(getIndices().getType()))
    return emitOpError("requires pointer-like destination/source/indices");
  if (classifyMemoryRole(getDestination().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSource().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getIndices().getType()) != MemoryRole::UB)
    return emitOpError("requires UB-backed destination/source/indices");
  if (!getRepeatTimes().getType().isIndex())
    return emitOpError("repeat_times must be index");
  return success();
}

LogicalResult Vmrgsort4Op::verify() {
  if (!isBufferLike(getDestination().getType()) || !isBufferLike(getSource0().getType()) ||
      !isBufferLike(getSource1().getType()) || !isBufferLike(getSource2().getType()) ||
      !isBufferLike(getSource3().getType()))
    return emitOpError("requires pointer-like destination and sources");
  if (classifyMemoryRole(getDestination().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSource0().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSource1().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSource2().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSource3().getType()) != MemoryRole::UB)
    return emitOpError("requires UB-backed destination and sources");
  return success();
}

LogicalResult VmaxOp::verify() {
  if (failed(verifyVecTypeLike(*this, getLhs().getType(), "lhs")) ||
      failed(verifyVecTypeLike(*this, getRhs().getType(), "rhs")) ||
      failed(verifyVecTypeLike(*this, getResult().getType(), "result")))
    return failure();
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getResult().getType())
    return emitOpError("lhs, rhs, and result must have the same vector type");
  return success();
}

LogicalResult VminOp::verify() {
  if (failed(verifyVecTypeLike(*this, getLhs().getType(), "lhs")) ||
      failed(verifyVecTypeLike(*this, getRhs().getType(), "rhs")) ||
      failed(verifyVecTypeLike(*this, getResult().getType(), "result")))
    return failure();
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getResult().getType())
    return emitOpError("lhs, rhs, and result must have the same vector type");
  return success();
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

template <typename BufOp>
static LogicalResult verifyBufTokenOp(BufOp op) {
  if (failed(verifySyncToken(op, op.getPipeAttr(), "pipe")))
    return failure();
  if (!isa<IntegerType>(op.getBufId().getType()) || !isa<IntegerType>(op.getMode().getType()))
    return op.emitOpError("requires integer buf_id and mode operands");
  return success();
}

LogicalResult GetBufOp::verify() { return verifyBufTokenOp(*this); }
LogicalResult RlsBufOp::verify() { return verifyBufTokenOp(*this); }

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

  if (getDistAttr()) {
    StringRef dist = *getDist();
    if (dist != "NORM" && dist != "BLK" && dist != "DINTLV_B32" &&
        dist != "UNPK_B16")
      return emitOpError(
          "supports only NORM, BLK, DINTLV_B32, and UNPK_B16 distributions");
  }

  return success();
}

void VldasOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VldasOp::verify() {
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a pointer-like source");
  if (failed(verifyAlignTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");
  return success();
}

void VldusOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VldusOp::verify() {
  if (failed(verifyAlignTypeLike(*this, getAlign().getType(), "align type")) ||
      failed(verifyVecTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a pointer-like source");
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");
  return success();
}

LogicalResult VdupOp::verify() {
  auto resultType = dyn_cast<VecType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be !a5vm.vec<...>");

  Type inputType = getInput().getType();
  if (auto inputVecType = dyn_cast<VecType>(inputType)) {
    if (inputVecType != resultType)
      return emitOpError("vector input must match result vector type");
    return success();
  }

  if (inputType != resultType.getElementType())
    return emitOpError("scalar input must match result element type");

  return success();
}

LogicalResult PsetB8Op::verify() {
  if (failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();

  StringRef pattern = getPattern();
  if (pattern != "PAT_ALL" && pattern != "PAT_ALLF")
    return emitOpError("supports only PAT_ALL and PAT_ALLF patterns");
  return success();
}

LogicalResult PsetB16Op::verify() {
  if (failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();

  StringRef pattern = getPattern();
  if (pattern != "PAT_ALL" && pattern != "PAT_ALLF")
    return emitOpError("supports only PAT_ALL and PAT_ALLF patterns");
  return success();
}

LogicalResult PpackOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getInput().getType(), "input type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (getPart() != "LOWER")
    return emitOpError("currently supports only LOWER part");
  return success();
}

LogicalResult PunpackOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getInput().getType(), "input type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (getPart() != "LOWER")
    return emitOpError("currently supports only LOWER part");
  return success();
}

void PldsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult PldsOp::verify() {
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a pointer-like source");
  if (failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  MemoryRole sourceRole = classifyMemoryRole(getSource().getType());
  if (sourceRole == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");
  return success();
}

template <typename OpTy>
static LogicalResult verifyVecScalarOpLike(OpTy op) {
  auto inputType = dyn_cast<VecType>(op.getInput().getType());
  auto resultType = dyn_cast<VecType>(op.getResult().getType());
  if (!inputType || !resultType)
    return op.emitOpError("input and result must be !a5vm.vec<...>");
  if (inputType != resultType)
    return op.emitOpError("input and result vector types must match");
  if (op.getScalar().getType() != inputType.getElementType())
    return op.emitOpError("scalar type must match vector element type");
  return success();
}

LogicalResult VmulsOp::verify() { return verifyVecScalarOpLike(*this); }
LogicalResult VaddsOp::verify() { return verifyVecScalarOpLike(*this); }
LogicalResult VmaxsOp::verify() { return verifyVecScalarOpLike(*this); }
LogicalResult VminsOp::verify() { return verifyVecScalarOpLike(*this); }
LogicalResult VlreluOp::verify() { return verifyVecScalarOpLike(*this); }

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
LogicalResult VandOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VorOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VxorOp::verify() { return verifyBinaryVecOp(*this); }

LogicalResult VselOp::verify() {
  if (failed(verifyVecTypeLike(*this, getSrc0().getType(), "src0 type")) ||
      failed(verifyVecTypeLike(*this, getSrc1().getType(), "src1 type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVecTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (getSrc0().getType() != getSrc1().getType() ||
      getSrc0().getType() != getResult().getType())
    return emitOpError("requires src0, src1, and result to have identical vector types");
  return success();
}

static bool isSupportedCmpMode(StringRef mode) {
  return mode == "eq" || mode == "ne" || mode == "lt" || mode == "le" ||
         mode == "gt" || mode == "ge";
}

LogicalResult VcmpOp::verify() {
  if (failed(verifyVecTypeLike(*this, getSrc0().getType(), "src0 type")) ||
      failed(verifyVecTypeLike(*this, getSrc1().getType(), "src1 type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (getSrc0().getType() != getSrc1().getType())
    return emitOpError("requires src0 and src1 to have identical vector types");
  if (!isSupportedCmpMode(getCmpMode()))
    return emitOpError("requires cmp_mode to be one of eq/ne/lt/le/gt/ge");
  return success();
}

LogicalResult VcmpsOp::verify() {
  if (failed(verifyVecTypeLike(*this, getSrc().getType(), "src type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  auto srcType = cast<VecType>(getSrc().getType());
  if (getScalar().getType() != srcType.getElementType())
    return emitOpError("requires scalar type to match source element type");
  if (!isSupportedCmpMode(getCmpMode()))
    return emitOpError("requires cmp_mode to be one of eq/ne/lt/le/gt/ge");
  return success();
}

LogicalResult VcvtOp::verify() {
  auto inputType = dyn_cast<VecType>(getInput().getType());
  auto resultType = dyn_cast<VecType>(getResult().getType());
  if (!inputType || !resultType)
    return emitOpError("input and result must be !a5vm.vec<...>");

  auto inputElemType = inputType.getElementType();
  auto resultElemType = resultType.getElementType();
  auto isSupportedElemType = [](Type type) {
    return type.isF16() || type.isBF16() || type.isF32();
  };
  if (!isSupportedElemType(inputElemType) || !isSupportedElemType(resultElemType))
    return emitOpError("currently supports only f16/bf16/f32 vector element types");

  if (getRoundModeAttr()) {
    StringRef roundMode = *getRoundMode();
    if (roundMode != "ROUND_R" && roundMode != "ROUND_A" &&
        roundMode != "ROUND_F" && roundMode != "ROUND_C" &&
        roundMode != "ROUND_Z" && roundMode != "ROUND_O")
      return emitOpError("round_mode must be one of ROUND_R/ROUND_A/ROUND_F/ROUND_C/ROUND_Z/ROUND_O");
  }

  if (getSatAttr()) {
    StringRef sat = *getSat();
    if (sat != "RS_ENABLE" && sat != "RS_DISABLE")
      return emitOpError("sat must be RS_ENABLE or RS_DISABLE");
  }

  if (getPartAttr()) {
    StringRef part = *getPart();
    if (part != "PART_EVEN" && part != "PART_ODD")
      return emitOpError("part must be PART_EVEN or PART_ODD");
  }

  return success();
}

LogicalResult PdintlvB8Op::verify() {
  if (failed(verifyMaskTypeLike(*this, getLhs().getType(), "lhs type")) ||
      failed(verifyMaskTypeLike(*this, getRhs().getType(), "rhs type")) ||
      failed(verifyMaskTypeLike(*this, getLow().getType(), "low type")) ||
      failed(verifyMaskTypeLike(*this, getHigh().getType(), "high type")))
    return failure();
  return success();
}

LogicalResult PintlvB16Op::verify() {
  if (failed(verifyMaskTypeLike(*this, getLhs().getType(), "lhs type")) ||
      failed(verifyMaskTypeLike(*this, getRhs().getType(), "rhs type")) ||
      failed(verifyMaskTypeLike(*this, getLow().getType(), "low type")) ||
      failed(verifyMaskTypeLike(*this, getHigh().getType(), "high type")))
    return failure();
  return success();
}

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

void VscatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VscatterOp::verify() {
  if (failed(verifyVecTypeLike(*this, getValue().getType(), "value type")))
    return failure();
  if (!isBufferLike(getDestination().getType()))
    return emitOpError("requires a pointer-like destination");
  auto offsetsType = dyn_cast<VecType>(getOffsets().getType());
  auto valueType = dyn_cast<VecType>(getValue().getType());
  if (!offsetsType || !valueType)
    return emitOpError("value and offsets must be !a5vm.vec<...>");
  auto offsetsElemType = dyn_cast<IntegerType>(offsetsType.getElementType());
  if (!offsetsElemType)
    return emitOpError("offset vector must use integer element type");
  if (offsetsElemType.getWidth() != 32)
    return emitOpError("currently requires 32-bit offset vector elements");
  if (offsetsType.getElementCount() != valueType.getElementCount())
    return emitOpError("offset and value vectors must have the same element count");
  MemoryRole destinationRole = classifyMemoryRole(getDestination().getType());
  if (destinationRole == MemoryRole::GM)
    return emitOpError("requires a UB-backed destination");
  if (!getActiveLanes().getType().isIndex())
    return emitOpError("active_lanes must be index");
  return success();
}

void VstsPredOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VstsPredOp::verify() {
  if (failed(verifyVecTypeLike(*this, getValue().getType(), "value type")))
    return failure();
  if (!isBufferLike(getDestination().getType()))
    return emitOpError("requires a pointer-like destination");
  if (!getActiveLanes().getType().isIndex())
    return emitOpError("requires index active_lanes");
  MemoryRole destinationRole = classifyMemoryRole(getDestination().getType());
  if (destinationRole == MemoryRole::GM)
    return emitOpError("requires a UB-backed destination");
  return success();
}

void PstsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult PstsOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getValue().getType(), "value type")))
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
