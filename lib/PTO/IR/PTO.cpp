//===- PTO.cpp - PTO Dialect ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOSyncUtils.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include <algorithm>
#include <numeric>
#include <optional>

using namespace mlir;
using namespace mlir::pto;

// Forward declarations for custom shape/type printers used by tensor_view and
// partition_tensor_view.
namespace mlir {
namespace pto {
static LogicalResult parseShapeAndElem(AsmParser &parser,
                                       SmallVectorImpl<int64_t> &shape,
                                       Type &elementType,
                                       bool allowDynamic = true);
static void printShapeAndElem(AsmPrinter &printer,
                              ArrayRef<int64_t> shape,
                              Type elementType);
} // namespace pto
} // namespace mlir

// =============================================================================
// TileBufType 的自定义 Shape 解析与打印函数
// =============================================================================

// 解析逻辑：解析形如 "32x32" 的维度列表
static ParseResult parseShape(AsmParser &parser, SmallVectorImpl<int64_t> &shape) {
  // parseDimensionList 会解析 "dim x dim x ...", 遇到无法解析为维度的字符停止
  // 参数 allowDynamic=true (允许 ?), withTrailingX=false (不吞掉末尾的 x)
  if (parser.parseDimensionList(shape, /*allowDynamic=*/true, /*withTrailingX=*/false))
    return failure();
  return success();
}

// 打印逻辑：打印形如 "32x32" 的维度列表
static void printShape(AsmPrinter &printer, ArrayRef<int64_t> shape) {
  for (auto it = shape.begin(); it != shape.end(); ++it) {
    if (it != shape.begin()) printer << "x"; // 维度间的分隔符
    if (*it == ShapedType::kDynamic)
      printer << "?";
    else
      printer << *it;
  }
  // 注意：我们不在这里打印末尾的 'x'，因为 assemblyFormat 中已经写了 `x` $elementType
}

static std::optional<pto::AddressSpace> getPTOMemorySpaceEnum(Type ty);
enum class VerifierTargetArch {
  A2A3,
  A5,
};
static VerifierTargetArch getVerifierTargetArch(Operation *op);
static std::optional<StringRef> getVerifierArchName(Operation *op);
static bool isSupportedVecElemType(Type ty, bool allowBf16 = true,
                                   bool allowInt8 = true);
static bool isSupportedLoadStoreElemTypeA2A3(Type ty);
static bool isSupportedGatherElemTypeA2A3(Type ty);
static bool isSupportedGatherElemTypeA5(Type ty);
static SmallVector<int64_t, 4> getShapeVec(Type ty);
static SmallVector<int64_t, 4> getValidShapeVec(Type ty);
static SmallVector<int64_t, 4> getValidShapeVec(Value value);
static LogicalResult verifyTileBufCommon(Operation *op, Type ty, StringRef name);
static LogicalResult verifyTileBufSameShapeAndElem(Operation *op, Type lhs, Type rhs,
                                                   StringRef lhsName,
                                                   StringRef rhsName);
static LogicalResult verifyTileBufSameValidShape(Operation *op, Type lhs, Type rhs,
                                                 StringRef lhsName, StringRef rhsName);
static LogicalResult verifyVecTileCommon(Operation *op, Type ty, StringRef name);
static LogicalResult verifyVecTileCommonA2A3(Operation *op, Type ty,
                                             StringRef name);
static LogicalResult verifyVecTileCommonA5(Operation *op, Type ty,
                                           StringRef name);
static LogicalResult verifyVecTileUnaryOp(Operation *op, Type srcTy, Type dstTy,
                                          StringRef srcName = "src",
                                          StringRef dstName = "dst",
                                          bool allowBf16 = true,
                                          bool allowInt8 = true);
static LogicalResult verifyVecTileBinaryOp(Operation *op, Type src0Ty, Type src1Ty,
                                           Type dstTy,
                                           bool allowBf16 = true,
                                           bool allowInt8 = true);
static LogicalResult verifyAccTileCommon(Operation *op, Type ty, StringRef name);
static LogicalResult verifyAccTileCommonA2A3(Operation *op, Type ty,
                                             StringRef name);
static LogicalResult verifyAccTileCommonA5(Operation *op, Type ty,
                                           StringRef name);
static LogicalResult verifyMatTileOperands(Operation *op, Type lhsTy, Type rhsTy,
                                           Type dstTy);
static LogicalResult verifyMatTileOperandsA2A3(Operation *op, Type lhsTy,
                                               Type rhsTy, Type dstTy);
static LogicalResult verifyMatTileOperandsA5(Operation *op, Type lhsTy,
                                             Type rhsTy, Type dstTy);
static LogicalResult verifyGemvTileOperands(Operation *op, Type lhsTy, Type rhsTy,
                                            Type dstTy);
static LogicalResult verifyGemvTileOperandsA2A3(Operation *op, Type lhsTy,
                                                Type rhsTy, Type dstTy);
static LogicalResult verifyGemvTileOperandsA5(Operation *op, Type lhsTy,
                                              Type rhsTy, Type dstTy);
static LogicalResult verifyMatBiasTile(Operation *op, Type biasTy, Type dstTy,
                                       bool requireFloatBias = false);
static LogicalResult verifyMatBiasTileA2A3(Operation *op, Type biasTy, Type dstTy,
                                           bool requireFloatBias = false);
static LogicalResult verifyMatBiasTileA5(Operation *op, Type biasTy, Type dstTy,
                                         bool requireFloatBias = false);
static LogicalResult verifyMatmulTypeTriple(Operation *op, Type lhsElemTy,
                                            Type rhsElemTy, Type dstElemTy);
static std::optional<pto::Layout> getLogicalViewLayout(Value value);
static std::optional<pto::Layout> getTileBufLogicalLayout(pto::TileBufType type);
static std::optional<int64_t> getConstantIntegerValue(Value value);
static LogicalResult verifyPartialValidPattern(Operation *op, Type src0Ty,
                                               Type src1Ty, Type dstTy);
static bool isRowMajorTileBuf(Type ty);

#define GET_ENUM_CLASSES
#include "PTO/IR/PTOEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "PTO/IR/PTOTypeDefs.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "PTO/IR/PTOAttrs.cpp.inc"

#include "PTO/IR/PTODialect.cpp.inc"

static LogicalResult parseShapeAndElemStable(mlir::AsmParser &parser,
                                             llvm::SmallVectorImpl<int64_t> &shape,
                                             mlir::Type &elementType) {
  if (failed(parser.parseLess()))
    return failure();

  if (failed(parser.parseDimensionList(shape, /*allowDynamic=*/true)))
    return failure();

  if (failed(parser.parseType(elementType)))
    return failure();

  if (failed(parser.parseGreater()))
    return failure();

  return success();
}

static int64_t getPTOTypeRank(Type type) {
  // 1. 处理标准的 MLIR 类型 (MemRef, Tensor, Vector)
  if (auto shapedTy = dyn_cast<ShapedType>(type)) {
    if (shapedTy.hasRank())
      return shapedTy.getRank();
    return -1; // Unranked type
  }
  
  // 2. 处理 PTO 自定义类型
  if (auto tvTy = dyn_cast<pto::TensorViewType>(type))
    return tvTy.getRank();

  if (auto tileTy = dyn_cast<pto::TileType>(type))
    return tileTy.getRank();
    
  if (auto tileViewTy = dyn_cast<pto::PartitionTensorViewType>(type))
    return tileViewTy.getRank();

  if (auto tileBufTy = dyn_cast<pto::TileBufType>(type))
    return tileBufTy.getRank();

  // 3. 不支持的类型
  return -1;
}

static bool isGmAddressSpaceAttr(Attribute memorySpace) {
  if (!memorySpace)
    return true;
  if (auto addr = mlir::dyn_cast<pto::AddressSpaceAttr>(memorySpace))
    return addr.getAddressSpace() == pto::AddressSpace::GM;
  if (auto intAttr = mlir::dyn_cast<IntegerAttr>(memorySpace))
    return intAttr.getInt() == 0;
  return false;
}

static VerifierTargetArch getVerifierTargetArch(Operation *op) {
  auto module = op ? op->getParentOfType<ModuleOp>() : ModuleOp();
  if (!module)
    return VerifierTargetArch::A2A3;
  auto arch = module->getAttrOfType<StringAttr>("pto.target_arch");
  if (arch && arch.getValue().equals_insensitive("a5"))
    return VerifierTargetArch::A5;
  return VerifierTargetArch::A2A3;
}

static std::optional<StringRef> getVerifierArchName(Operation *op) {
  auto module = op ? op->getParentOfType<ModuleOp>() : ModuleOp();
  if (!module)
    return std::nullopt;
  if (auto arch = module->getAttrOfType<StringAttr>("pto.target_arch"))
    return arch.getValue();
  return std::nullopt;
}

static bool shouldBypassDecodedMemrefVerifier(Operation *op) {
  if (!op)
    return false;
  for (Value operand : op->getOperands()) {
    if (isa<MemRefType>(operand.getType()))
      return true;
    if (operand.getDefiningOp<pto::BindTileOp>())
      return true;
  }
  return false;
}

static SmallVector<int64_t, 4> canonicalizeTileBufValidShape(ArrayRef<int64_t> validShape) {
  SmallVector<int64_t, 4> canonical;
  canonical.reserve(validShape.size());
  for (int64_t dim : validShape)
    canonical.push_back(dim < 0 ? ShapedType::kDynamic : dim);
  return canonical;
}

template <typename FnA2A3, typename FnA5>
static LogicalResult dispatchVerifierByArch(Operation *op, FnA2A3 &&verifyA2A3,
                                            FnA5 &&verifyA5) {
  if (shouldBypassDecodedMemrefVerifier(op))
    return success();
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyA2A3();
  case VerifierTargetArch::A5:
    return verifyA5();
  }
}

static mlir::Type parsePTOTypeAllowNoBang(mlir::OpAsmParser &parser) {
  mlir::Type ty;

  mlir::OptionalParseResult opt = parser.parseOptionalType(ty);

  if (opt.has_value()) {         
    if (failed(*opt))
      return mlir::Type();       
    return ty;                    
  }


  llvm::StringRef head;
  if (failed(parser.parseKeyword(&head)))
    return mlir::Type();

  mlir::MLIRContext *ctx = parser.getContext();

  auto parseShapeElemForOpParser =
      [&](llvm::SmallVectorImpl<int64_t> &shape, mlir::Type &elem) -> mlir::LogicalResult {
        if (failed(parser.parseLess()))
          return failure();
        if (failed(parser.parseDimensionList(shape, /*allowDynamic=*/true)))
          return failure();
        if (failed(parser.parseType(elem)))
          return failure();
        if (failed(parser.parseGreater()))
          return failure();
        return success();
      };

  if (head == "pto.tile_view") {
    llvm::SmallVector<int64_t, 4> shape;
    mlir::Type elem;
    if (failed(parseShapeElemForOpParser(shape, elem)))
      return mlir::Type();
    return mlir::pto::PartitionTensorViewType::get(ctx, shape, elem);
  }

  if (head == "pto.tile") {
    llvm::SmallVector<int64_t, 4> shape;
    mlir::Type elem;
    if (failed(parseShapeElemForOpParser(shape, elem)))
      return mlir::Type();
    return mlir::pto::TileType::get(ctx, shape, elem);
  }

  if (head == "pto.ptr") {
    if (failed(parser.parseLess()))
      return mlir::Type();
    mlir::Type elem;
    if (failed(parser.parseType(elem)))
      return mlir::Type();
    if (succeeded(parser.parseOptionalComma())) {
      // ptr no longer accepts an address space; consume the attr for recovery.
      mlir::Attribute memorySpace;
      (void)parser.parseAttribute(memorySpace);
      parser.emitError(parser.getCurrentLocation(),
                       "!pto.ptr no longer accepts address space; use !pto.ptr<elem>");
      return mlir::Type();
    }
    if (failed(parser.parseGreater()))
      return mlir::Type();
    return mlir::pto::PtrType::get(ctx, elem);
  }

  if (head == "pto.tensor_view") {
    llvm::SmallVector<int64_t, 4> shape;
    mlir::Type elem;
    if (failed(parseShapeElemForOpParser(shape, elem)))
      return mlir::Type();
    return mlir::pto::TensorViewType::get(ctx, shape, elem);
  }

  return mlir::Type();
}

mlir::Type TensorViewType::parse(::mlir::AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elementType;
  if (failed(parseShapeAndElem(parser, shape, elementType, /*allowDynamic=*/true)))
    return Type();
  return TensorViewType::get(parser.getContext(), shape, elementType);
}

void TensorViewType::print(::mlir::AsmPrinter &printer) const {
  printShapeAndElem(printer, getShape(), getElementType());
}

//===----------------------------------------------------------------------===//
// pto.tdivs custom asm to support both:
//   pto.tdivs ins(%src, %scalar : !pto.tile_buf<...>, f32) outs(%dst : !pto.tile_buf<...>)
//   pto.tdivs ins(%scalar, %src : f32, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
// The operand order in the op remains (src, scalar, dst); order is determined
// by the type of the first operand in the textual format.
//===----------------------------------------------------------------------===//

ParseResult mlir::pto::TDivSOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand op0, op1, dst;
  Type ty0, ty1, dstTy;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(op0) || parser.parseComma() ||
      parser.parseOperand(op1) || parser.parseColonType(ty0) ||
      parser.parseComma() || parser.parseType(ty1) || parser.parseRParen())
    return failure();

  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen())
    return failure();

  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs))
    return failure();

  auto tile0 = dyn_cast<mlir::pto::TileBufType>(ty0);
  auto tile1 = dyn_cast<mlir::pto::TileBufType>(ty1);
  if ((tile0 && tile1) || (!tile0 && !tile1))
    return parser.emitError(parser.getCurrentLocation(),
                            "expected exactly one tile_buf operand and one scalar operand");

  if (!dyn_cast<mlir::pto::TileBufType>(dstTy))
    return parser.emitError(parser.getCurrentLocation(),
                            "expected outs type to be !pto.tile_buf<...>");

  // Determine order based on types: if first operand is tile_buf, order is (tile, scalar)
  // Otherwise, order is (scalar, tile)
  const bool scalarFirst = (tile1 != nullptr);

  if (!scalarFirst) {
    // ins(%src, %scalar : tile_buf, scalar_ty)
    // Operands in op: (src, scalar, dst)
    if (parser.resolveOperand(op0, ty0, result.operands) ||
        parser.resolveOperand(op1, ty1, result.operands))
      return failure();
  } else {
    // ins(%scalar, %src : scalar_ty, tile_buf)
    // Operands in op: (src, scalar, dst) - need to swap
    if (parser.resolveOperand(op1, ty1, result.operands) ||
        parser.resolveOperand(op0, ty0, result.operands))
      return failure();
  }

  if (parser.resolveOperand(dst, dstTy, result.operands))
    return failure();

  result.addAttributes(attrs);
  return success();
}

void mlir::pto::TDivSOp::print(OpAsmPrinter &p) {
  // Determine order based on operand types
  // If src is tile_buf and scalar is not, print (src, scalar)
  // If src is scalar and scalar is tile_buf, print (scalar, src)
  auto srcType = getSrc().getType();
  auto scalarType = getScalar().getType();
  
  bool srcIsTile = isa<mlir::pto::TileBufType>(srcType);
  bool scalarIsTile = isa<mlir::pto::TileBufType>(scalarType);
  
  p << " ins(";
  if (srcIsTile && !scalarIsTile) {
    // Print: (tile, scalar) - operands are already in correct order
    p << getSrc() << ", " << getScalar() << " : "
      << getSrc().getType() << ", " << getScalar().getType();
  } else if (!srcIsTile && scalarIsTile) {
    // Print: (scalar, tile) - need to swap operands in output
    p << getScalar() << ", " << getSrc() << " : "
      << getScalar().getType() << ", " << getSrc().getType();
  } else {
    // Default: assume src is tile (should not happen if types are correct)
    p << getSrc() << ", " << getScalar() << " : "
      << getSrc().getType() << ", " << getScalar().getType();
  }
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";

  p.printOptionalAttrDict((*this)->getAttrs());
}


//===----------------------------------------------------------------------===//
// pto.tgather custom asm to support both:
//   pto.tgather ins(%src, %indices : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
//   pto.tgather ins(%src, {maskPattern = #pto.mask_pattern<P0101>} : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
//
// Legacy syntax support (for existing test artifacts):
//   pto.tgather ins(%src : !pto.tile_buf<...>, %indices : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>) [maskPattern = ...]
//===----------------------------------------------------------------------===//

ParseResult mlir::pto::TGatherOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, dst, indices;
  Type srcTy, dstTy, indicesTy;
  bool hasIndices = false;

  // ins(...)
  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(src))
    return failure();

  // New-style: ins(%src, <indices-or-{maskPattern=...}> : ...)
  if (succeeded(parser.parseOptionalComma())) {
    // Mask form: ins(%src, {maskPattern = #pto.mask_pattern<Pxxxx>} : srcTy)
    if (succeeded(parser.parseOptionalLBrace())) {
      if (parser.parseKeyword("maskPattern") || parser.parseEqual())
        return failure();

      Attribute rawMaskAttr;
      if (parser.parseAttribute(rawMaskAttr) || parser.parseRBrace())
        return failure();

      auto mp = llvm::dyn_cast<mlir::pto::MaskPatternAttr>(rawMaskAttr);
      if (!mp)
        return parser.emitError(parser.getCurrentLocation(),
                                "expected #pto.mask_pattern<Pxxxx> for maskPattern");

      result.addAttribute("maskPattern", mp);

      if (parser.parseColonType(srcTy) || parser.parseRParen())
        return failure();
    } else {
      // Index form: ins(%src, %indices : srcTy, indicesTy)
      if (parser.parseOperand(indices) || parser.parseColonType(srcTy) ||
          parser.parseComma() || parser.parseType(indicesTy) || parser.parseRParen())
        return failure();
      hasIndices = true;
    }
  } else if (succeeded(parser.parseOptionalColon())) {
    // Legacy typed-operand form:
    //   ins(%src : srcTy, %indices : indicesTy)
    if (parser.parseType(srcTy) || parser.parseComma() ||
        parser.parseOperand(indices) || parser.parseColonType(indicesTy) ||
        parser.parseRParen())
      return failure();
    hasIndices = true;
  } else {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected ',' or ':' after src operand in ins(...)");
  }

  // outs(...)
  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) || parser.parseRParen())
    return failure();

  // Optional legacy: maskPattern = #pto.mask_pattern<Pxxxx>
  if (succeeded(parser.parseOptionalKeyword("maskPattern"))) {
    if (parser.parseEqual())
      return failure();
    Attribute rawMaskAttr;
    if (parser.parseAttribute(rawMaskAttr))
      return failure();
    auto mp = llvm::dyn_cast<mlir::pto::MaskPatternAttr>(rawMaskAttr);
    if (!mp)
      return parser.emitError(parser.getCurrentLocation(),
                              "expected #pto.mask_pattern<Pxxxx> for maskPattern");
    result.addAttribute("maskPattern", mp);
  }

  // attr-dict
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Resolve operands in op order: (src, dst, optional indices).
  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands))
    return failure();
  if (hasIndices && parser.resolveOperand(indices, indicesTy, result.operands))
    return failure();

  return success();
}

void mlir::pto::TGatherOp::print(OpAsmPrinter &p) {
  auto indices = getIndices();
  auto mp = getMaskPatternAttr();

  p << " ins(" << getSrc() << ", ";
  if (indices) {
    p << indices << " : " << getSrc().getType() << ", " << indices.getType();
  } else {
    p << "{maskPattern = " << mp << "} : " << getSrc().getType();
  }
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";

  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"maskPattern"});
}

ParseResult mlir::pto::MakeTensorViewOp::parse(OpAsmParser &parser,
                                               OperationState &result) {
  OpAsmParser::UnresolvedOperand ptr;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> shapeOps;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> strideOps;

  Type resultTy;

  // %ptr
  if (parser.parseOperand(ptr))
    return failure();

  // , shape = [ ... ]
  if (parser.parseComma() || parser.parseKeyword("shape") || parser.parseEqual() ||
      parser.parseLSquare() ||
      parser.parseOperandList(shapeOps) ||
      parser.parseRSquare())
    return failure();

  // strides = [ ... ]
  if (parser.parseComma() || parser.parseKeyword("strides") || parser.parseEqual() ||
      parser.parseLSquare() ||
      parser.parseOperandList(strideOps) ||
      parser.parseRSquare())
    return failure();

  // attr-dict
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // : result-type
  if (parser.parseColonType(resultTy))
    return failure();
  result.addTypes(resultTy);

  auto tvTy = llvm::dyn_cast<mlir::pto::TensorViewType>(resultTy);
  if (!tvTy)
    return parser.emitError(parser.getCurrentLocation(),
                            "expected result type pto.tensor_view<...>");

  Type elemTy = tvTy.getElementType();

  Type ptrTy = mlir::pto::PtrType::get(parser.getContext(), elemTy);

  // resolve %ptr
  if (parser.resolveOperand(ptr, ptrTy, result.operands))
    return failure();

  // resolve shape/strides 为 index
  Type indexTy = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(shapeOps, indexTy, result.operands))
    return failure();
  if (parser.resolveOperands(strideOps, indexTy, result.operands))
    return failure();

  auto segAttr = parser.getBuilder().getDenseI32ArrayAttr(
      {1, (int32_t)shapeOps.size(), (int32_t)strideOps.size()});
  result.addAttribute("operandSegmentSizes", segAttr);

  return success();
}

void mlir::pto::MakeTensorViewOp::print(OpAsmPrinter &p) {
  p << " " << getPtr();

  p << ", shape = [";
  p.printOperands(getShape());
  p << "]";

  p << ", strides = [";
  p.printOperands(getStrides());
  p << "]";

  p.printOptionalAttrDict((*this)->getAttrs(),
                        /*elidedAttrs=*/{"operandSegmentSizes"});

  p << " : " << getResult().getType();
}

// Layout inference helpers for make_tensor_view
static std::optional<int64_t> getConstIndexValue(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
    return c.value();
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt();
  }
  return std::nullopt;
}

static unsigned getElemByteSize(Type ty) {
  if (auto f = dyn_cast<FloatType>(ty))
    return f.getWidth() / 8;
  if (auto i = dyn_cast<IntegerType>(ty))
    return i.getWidth() / 8;
  return 0;
}

static bool isSupportedLoadStoreElemTypeA2A3(Type ty) {
  if (ty.isF16() || ty.isBF16() || ty.isF32())
    return true;
  if (auto it = dyn_cast<IntegerType>(ty)) {
    unsigned width = it.getWidth();
    return width == 8 || width == 16 || width == 32 || width == 64;
  }
  return false;
}

static bool isSupportedGatherElemTypeA2A3(Type ty) {
  if (ty.isF16() || ty.isF32())
    return true;
  if (auto it = dyn_cast<IntegerType>(ty)) {
    unsigned width = it.getWidth();
    return width == 16 || width == 32;
  }
  return false;
}

static bool isSupportedGatherElemTypeA5(Type ty) {
  if (isSupportedGatherElemTypeA2A3(ty) || ty.isBF16())
    return true;
  if (auto ft = dyn_cast<FloatType>(ty)) {
    unsigned width = ft.getWidth();
    return width == 8;
  }
  if (auto it = dyn_cast<IntegerType>(ty))
    return it.getWidth() == 8 || it.getWidth() == 16 || it.getWidth() == 32;
  return false;
}

static std::optional<mlir::pto::Layout>
inferLayout(ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
            unsigned elemBytes) {
  if (shape.size() != strides.size() || elemBytes == 0)
    return std::nullopt;

  // NZ / fractal: rank>=5, check middle dims (sh3/sh4/sh5 per spec)
  if (shape.size() >= 5) {
    int64_t sh3 = shape[2], sh4 = shape[3], sh5 = shape[4];
    int64_t st4 = strides[3], st5 = strides[4];
    bool alignMatch = (sh3 == 16) && (sh3 * sh4 * elemBytes == 512);
    bool strideMatch = (st5 == 1) && (st4 == sh5);
    if (alignMatch && strideMatch)
      return mlir::pto::Layout::NZ;
  }

  // ND: row-major contiguous
  bool isRowMajor = true;
  for (int i = 0, e = (int)shape.size() - 1; i < e; ++i) {
    if (strides[i] != strides[i + 1] * shape[i + 1]) {
      isRowMajor = false;
      break;
    }
  }
  if (isRowMajor && strides.back() == 1)
    return mlir::pto::Layout::ND;

  // DN: col-major
  bool isColMajor = true;
  for (int i = 0, e = (int)shape.size() - 1; i < e; ++i) {
    if (strides[i + 1] != strides[i] * shape[i]) {
      isColMajor = false;
      break;
    }
  }
  if (isColMajor && strides.front() == 1)
    return mlir::pto::Layout::DN;

  return mlir::pto::Layout::ND; // fallback
}

static std::optional<pto::Layout> getLogicalViewLayout(Value value) {
  if (!value)
    return std::nullopt;
  if (auto part = value.getDefiningOp<pto::PartitionViewOp>())
    return getLogicalViewLayout(part.getSource());
  if (auto make = value.getDefiningOp<pto::MakeTensorViewOp>()) {
    auto tvTy = dyn_cast<pto::TensorViewType>(make.getResult().getType());
    if (!tvTy)
      return std::nullopt;
    SmallVector<int64_t> shape(tvTy.getShape().begin(), tvTy.getShape().end());
    SmallVector<int64_t> strides;
    strides.reserve(make.getStrides().size());
    for (Value stride : make.getStrides()) {
      auto cst = getConstIndexValue(stride);
      if (!cst)
        return std::nullopt;
      strides.push_back(*cst);
    }
    return inferLayout(shape, strides, getElemByteSize(tvTy.getElementType()));
  }
  return std::nullopt;
}

static std::optional<pto::Layout> getTileBufLogicalLayout(pto::TileBufType type) {
  if (!type)
    return std::nullopt;
  int32_t sl = type.getSLayoutValueI32();
  int32_t bl = type.getBLayoutValueI32();
  if (sl != static_cast<int32_t>(pto::SLayout::NoneBox))
    return pto::Layout::NZ;
  if (bl == static_cast<int32_t>(pto::BLayout::RowMajor))
    return pto::Layout::ND;
  if (bl == static_cast<int32_t>(pto::BLayout::ColMajor))
    return pto::Layout::DN;
  return std::nullopt;
}

static bool isRowMajorTileBuf(Type ty) {
  auto tb = mlir::dyn_cast<pto::TileBufType>(ty);
  return tb && tb.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
}

static LogicalResult verifyRowReductionSrcLayout(Operation *op, Type ty,
                                                 StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name)))
    return failure();
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor))
      return op->emitOpError() << "expects " << name << " to use the row_major blayout";
  }
  if (auto mr = dyn_cast<MemRefType>(ty))
    (void)mr;
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
      return op->emitOpError() << "expects " << name
                               << " to use the none_box slayout";
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    auto layout = getTileBufLogicalLayout(tb);
    if (layout && *layout != pto::Layout::ND)
      return op->emitOpError() << "expects " << name
                               << " to use an ND-style tile layout";
  }
  return success();
}

static LogicalResult verifyRowReductionDstLayout(Operation *op, Type ty,
                                                 StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name)))
    return failure();
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
      return op->emitOpError() << "expects " << name
                               << " to use the none_box slayout";
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) &&
        tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor))
      return op->emitOpError() << "expects " << name
                               << " to use the row_major or col_major blayout";
  }
  if (auto mr = dyn_cast<MemRefType>(ty))
    (void)mr;
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    auto layout = getTileBufLogicalLayout(tb);
    if (layout && *layout == pto::Layout::DN) {
      auto shape = getShapeVec(ty);
      if (shape.size() == 2 && shape[1] != ShapedType::kDynamic && shape[1] != 1)
        return op->emitOpError() << "expects DN-style " << name
                                 << " to have shape[1] == 1";
      return success();
    }
    if (layout && *layout == pto::Layout::ND)
      return success();
    if (layout)
      return op->emitOpError() << "expects " << name
                               << " to use a DN-style column vector tile or legacy ND-style tile";
  }
  return success();
  auto valid = getValidShapeVec(ty);
  if (valid.size() != 2)
    return op->emitOpError() << "expects " << name << " to have rank-2 valid_shape";
  if (valid[1] != ShapedType::kDynamic && valid[1] != 1)
    return op->emitOpError() << "expects " << name << " valid_shape[1] to be 1";
  return success();
}

static LogicalResult verifyRowReductionValidRegion(Operation *op, Type srcTy,
                                                   Type dstTy) {
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != 2 || dstValid.size() != 2)
    return op->emitOpError("expects src and dst to have rank-2 valid_shape");
  if (srcValid[0] != ShapedType::kDynamic && srcValid[0] == 0)
    return op->emitOpError("expects src valid_shape[0] to be non-zero");
  if (srcValid[1] != ShapedType::kDynamic && srcValid[1] == 0)
    return op->emitOpError("expects src valid_shape[1] to be non-zero");
  if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
      srcValid[0] != dstValid[0])
    return op->emitOpError("expects src and dst to have the same valid_shape[0]");
  if (dstValid[1] != ShapedType::kDynamic && dstValid[1] != 1)
    return op->emitOpError("expects dst valid_shape[1] to be 1");
  return success();
}

static LogicalResult verifyNDStyleVecTile(Operation *op, Type ty, StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name)))
    return failure();
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor))
      return op->emitOpError() << "expects " << name << " to use the row_major blayout";
    if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
      return op->emitOpError() << "expects " << name << " to use the none_box slayout";
  }
  return success();
}

static LogicalResult verifyColReductionValidRegion(Operation *op, Type srcTy,
                                                   Type dstTy,
                                                   bool requireNonZeroSrc) {
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != 2 || dstValid.size() != 2)
    return op->emitOpError("expects src and dst to have rank-2 valid_shape");
  if (requireNonZeroSrc) {
    if (srcValid[0] != ShapedType::kDynamic && srcValid[0] == 0)
      return op->emitOpError("expects src valid_shape[0] to be non-zero");
    if (srcValid[1] != ShapedType::kDynamic && srcValid[1] == 0)
      return op->emitOpError("expects src valid_shape[1] to be non-zero");
  }
  if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1])
    return op->emitOpError("expects src and dst to have the same valid_shape[1]");
  return success();
}

static std::optional<int64_t> getConstantIntegerValue(Value value) {
  if (!value)
    return std::nullopt;
  if (auto arithCst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(arithCst.getValue()))
      return intAttr.getInt();
  }
  return std::nullopt;
}

LogicalResult mlir::pto::MakeTensorViewOp::verify() {
  auto tvTy = dyn_cast<mlir::pto::TensorViewType>(getResult().getType());
  if (!tvTy)
    return emitOpError("result must be pto.tensor_view<...>");

  auto pty = dyn_cast<mlir::pto::PtrType>(getPtr().getType());
  if (!pty)
    return emitOpError("ptr operand must be !pto.ptr<...>");

  if (pty.getElementType() != tvTy.getElementType())
    return emitOpError() << "ptr element type must match tensor_view element type, but got ptr="
                         << pty.getElementType() << " view=" << tvTy.getElementType();

  int64_t rank = tvTy.getRank();

  if ((int64_t)getShape().size() != rank || (int64_t)getStrides().size() != rank)
    return emitOpError() << "shape/strides operand counts must match tensor_view rank="
                         << rank;

  // Detect dynamic shape/stride.
  bool hasDynamicShape = llvm::any_of(tvTy.getShape(), [](int64_t v) {
    return v == ShapedType::kDynamic;
  });
  bool hasDynamicStride = llvm::any_of(getStrides(), [](Value s) {
    return !getConstIndexValue(s).has_value();
  });

  auto layoutAttr = getLayoutAttr();

  // 1) Dynamic shape/stride without explicit layout: warn and keep going.
  if ((hasDynamicShape || hasDynamicStride) && !layoutAttr) {
    return success();
  }

  // 2) Static shape/stride with explicit layout: verify correctness.
  bool allStaticStride = true;
  SmallVector<int64_t> strideInts;
  strideInts.reserve(getStrides().size());
  for (Value s : getStrides()) {
    auto val = getConstIndexValue(s);
    if (!val) {
      allStaticStride = false;
      break;
    }
    strideInts.push_back(*val);
  }

  bool allStaticShape =
      llvm::none_of(tvTy.getShape(), [](int64_t v) { return v == ShapedType::kDynamic; });

  if (layoutAttr && allStaticShape && allStaticStride) {
    SmallVector<int64_t> shapeInts(tvTy.getShape().begin(), tvTy.getShape().end());
    if (auto inferred = inferLayout(shapeInts, strideInts,
                                    getElemByteSize(tvTy.getElementType()))) {
      (void)inferred;
    }
  }

  return success();
}

LogicalResult mlir::pto::PartitionViewOp::verify() {
  auto sourceTy = dyn_cast<mlir::pto::TensorViewType>(getSource().getType());
  if (!sourceTy)
    return emitOpError("source must be !pto.tensor_view<...>");
  auto resultTy = dyn_cast<mlir::pto::PartitionTensorViewType>(getResult().getType());
  if (!resultTy)
    return emitOpError("result must be !pto.partition_tensor_view<...>");

  int64_t rank = sourceTy.getRank();
  if (static_cast<int64_t>(getOffsets().size()) != rank ||
      static_cast<int64_t>(getSizes().size()) != rank)
    return emitOpError() << "offsets and sizes must match source rank " << rank;

  if (resultTy.getElementType() != sourceTy.getElementType())
    return emitOpError("result element type must match source element type");

  return success();
}

LogicalResult mlir::pto::AddPtrOp::verify() {
  auto ptrTy = dyn_cast<mlir::pto::PtrType>(getPtr().getType());
  if (!ptrTy)
    return emitOpError("ptr operand must be !pto.ptr<...>");

  auto resTy = dyn_cast<mlir::pto::PtrType>(getResult().getType());
  if (!resTy)
    return emitOpError("result must be !pto.ptr<...>");

  if (ptrTy != resTy)
    return emitOpError("result type must match ptr operand type");

  return success();
}




void PTODialect::initialize() {
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
}


AddressSpaceAttr mlir::pto::getPTOAddressSpaceAttr(Type type) {
  auto memRefType = dyn_cast<BaseMemRefType>(type);
  assert(memRefType && "input type must be a memref type");
  auto scopeAttr = dyn_cast<AddressSpaceAttr>(memRefType.getMemorySpace());
  assert(scopeAttr && "memory scope should be a pto address scope");
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

void mlir::pto::annotatePTOEntryFunctions(ModuleOp module) {
  if (!module)
    return;

  SmallVector<func::FuncOp> defs = getPTOFunctionDefinitions(module);
  for (auto func : module.getOps<func::FuncOp>())
    func->removeAttr(kEffectivePTOEntryAttrName);

  if (defs.empty())
    return;
  if (defs.size() == 1) {
    defs.front()->setAttr(kEffectivePTOEntryAttrName,
                          BoolAttr::get(module.getContext(), true));
    return;
  }

  for (auto func : defs) {
    func->setAttr(kEffectivePTOEntryAttrName,
                  BoolAttr::get(module.getContext(),
                                hasExplicitPTOEntryAttr(func)));
  }
}

//===----------------------------------------------------------------------===//
// PTO Load/Store/Addf (non-DPS polymorphic) verification + inference.
//  - If operands are memref/tensor: verify strictly.
//  - Otherwise (tile_view/tile etc): accept (so old IR can still parse).
//===----------------------------------------------------------------------===//

static LogicalResult verifyMemrefToTensorLoad(Operation *op, Value src, Value res) {
  auto mr = dyn_cast<MemRefType>(src.getType());
  auto rt = dyn_cast<RankedTensorType>(res.getType());
  if (!mr)
    return success(); // non-memref case: don't block old IR
  if (!rt)
    return op->emitOpError("when src is memref, result must be ranked tensor");

  if (mr.getElementType() != rt.getElementType())
    return op->emitOpError() << "memref/tensor element type mismatch: memref="
                             << mr.getElementType() << " tensor=" << rt.getElementType();

  if (mr.getRank() != rt.getRank())
    return op->emitOpError() << "rank mismatch: memref rank=" << mr.getRank()
                             << " tensor rank=" << rt.getRank();

  if (mr.hasStaticShape()) {
    if (!rt.hasStaticShape())
      return op->emitOpError("memref has static shape but result tensor is not static");
    if (mr.getShape() != rt.getShape())
      return op->emitOpError() << "shape mismatch: memref=" << mr << " tensor=" << rt;
  } else {
    // For dynamic memref dims: if tensor dim is static, allow it; if it's dynamic too, also fine.
    // We only reject when a memref static dim conflicts with tensor static dim.
    for (int64_t i = 0; i < mr.getRank(); ++i) {
      int64_t md = mr.getDimSize(i);
      int64_t td = rt.getDimSize(i);
      if (md != ShapedType::kDynamic && td != ShapedType::kDynamic && md != td)
        return op->emitOpError() << "dim mismatch at " << i << ": memref=" << md << " tensor=" << td;
    }
  }
  return success();
}

static LogicalResult verifyMemrefTensorStore(Operation *op, Value dst, Value src) {
  auto mr = dyn_cast<MemRefType>(dst.getType());
  if (!mr)
    return success(); // non-memref case: old tile IR allowed
  auto rt = dyn_cast<RankedTensorType>(src.getType());
  if (!rt)
    return op->emitOpError("when dst is memref, src must be ranked tensor");

  if (mr.getElementType() != rt.getElementType())
    return op->emitOpError() << "memref/tensor element type mismatch: memref="
                             << mr.getElementType() << " tensor=" << rt.getElementType();

  if (mr.getRank() != rt.getRank())
    return op->emitOpError() << "rank mismatch: memref rank=" << mr.getRank()
                             << " tensor rank=" << rt.getRank();

  for (int64_t i = 0; i < mr.getRank(); ++i) {
    int64_t md = mr.getDimSize(i);
    int64_t td = rt.getDimSize(i);
    if (md != ShapedType::kDynamic && td != ShapedType::kDynamic && md != td)
      return op->emitOpError() << "dim mismatch at " << i << ": memref=" << md << " tensor=" << td;
  }
  return success();
}

LogicalResult AllocTileOp::verify() {
  auto ty = getResult().getType(); // TileBufType

  // op 上有没有传 operands
  bool hasVR = getValidRow() != nullptr;
  bool hasVC = getValidCol() != nullptr;

  // type 上的 validShape
  auto vs = ty.getValidShape();
  if (vs.size() != 2)
    return emitOpError("result tile_buf must have rank-2 validShape");

  // TileBuf valid dims use a negative sentinel (e.g. '?' / -1). Be robust to
  // any negative value (some code may materialize MLIR dynamic sentinels).
  bool needVR = (vs[0] < 0);
  bool needVC = (vs[1] < 0);

  // 你要求的：v_row=?, v_col=? 时必须同时给两个
  // （这条规则由下面两句自然实现）
  if (hasVR != needVR)
    return emitOpError() << "valid_row operand "
                         << (needVR ? "is required" : "must be absent")
                         << " because result type v_row is "
                         << (needVR ? "?" : std::to_string(vs[0]));

  if (hasVC != needVC)
    return emitOpError() << "valid_col operand "
                         << (needVC ? "is required" : "must be absent")
                         << " because result type v_col is "
                         << (needVC ? "?" : std::to_string(vs[1]));

  return success();
}

LogicalResult TLoadOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<std::pair<pto::PartitionTensorViewType, pto::TileBufType>> {
    auto srcPart = dyn_cast<pto::PartitionTensorViewType>(getSrc().getType());
    auto dstTile = dyn_cast<pto::TileBufType>(getDst().getType());
    if (!srcPart || !dstTile) {
      emitOpError("expects src to be !pto.partition_tensor_view and dst to be !pto.tile_buf");
      return failure();
    }
    if (failed(verifyTileBufCommon(*this, dstTile, "dst")))
      return failure();

    auto srcShape = srcPart.getShape();
    for (unsigned i = 0; i < srcShape.size(); ++i) {
      if (srcShape[i] != ShapedType::kDynamic && srcShape[i] <= 0) {
        emitOpError() << "expects src shape[" << i << "] to be positive";
        return failure();
      }
    }
    auto dstValid = dstTile.getValidShape();
    for (unsigned i = 0; i < dstValid.size(); ++i) {
      if (dstValid[i] != ShapedType::kDynamic && dstValid[i] <= 0) {
        emitOpError() << "expects dst valid_shape[" << i << "] to be positive";
        return failure();
      }
    }
    return std::make_pair(srcPart, dstTile);
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    auto common = verifyCommon();
    if (failed(common))
      return failure();
    auto [srcPart, dstTile] = *common;

    Type srcElem = srcPart.getElementType();
    Type dstElem = dstTile.getElementType();
    if (!(dstElem.isInteger(8) || dstElem.isInteger(16) || dstElem.isInteger(32) ||
          dstElem.isInteger(64) || dstElem.isF16() || dstElem.isBF16() || dstElem.isF32()))
      return emitOpError("expects A2/A3 tload dst element type to be i8/u8/i16/u16/i32/u32/i64/u64/f16/bf16/f32");

    auto dstSpace = getPTOMemorySpaceEnum(dstTile);
    if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                      *dstSpace != pto::AddressSpace::MAT))
      return emitOpError("expects A2/A3 tload dst to use loc=vec or loc=mat");

    if (getElemByteSize(srcElem) != getElemByteSize(dstElem))
      return emitOpError("expects src and dst element types to have the same bitwidth");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    auto common = verifyCommon();
    if (failed(common))
      return failure();
    auto [srcPart, dstTile] = *common;

    Type srcElem = srcPart.getElementType();
    Type dstElem = dstTile.getElementType();
    unsigned srcBytes = getElemByteSize(srcElem);
    unsigned dstBytes = getElemByteSize(dstElem);
    if (srcBytes != dstBytes)
      return emitOpError("expects src and dst element types to have the same element size");
    if (!(dstBytes == 1 || dstBytes == 2 || dstBytes == 4 || dstBytes == 8))
      return emitOpError("expects A5 tload dst element size to be 1, 2, 4, or 8 bytes");

    if (dstElem.isInteger(64)) {
      auto pad = dstTile.getPadValueI32();
      if (pad != static_cast<int32_t>(pto::PadValue::Null) &&
          pad != static_cast<int32_t>(pto::PadValue::Zero))
        return emitOpError("expects A5 i64/u64 tload dst pad to be null or zero");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::SetFFTsOp::verify() {
  auto mr = llvm::dyn_cast<mlir::MemRefType>(getFfts().getType());
  if (!mr)
    return emitOpError("expects a memref operand");

  if (!mr.getElementType().isInteger(64) && !mr.getElementType().isInteger(8))
    return emitOpError("expects element type i64 (or i8)");

  return mlir::success();
}

LogicalResult TStoreOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<std::pair<pto::TileBufType, pto::PartitionTensorViewType>> {
    auto srcTile = dyn_cast<pto::TileBufType>(getSrc().getType());
    auto dstPart = dyn_cast<pto::PartitionTensorViewType>(getDst().getType());
    if (!srcTile || !dstPart) {
      emitOpError("expects src to be !pto.tile_buf and dst to be !pto.partition_tensor_view");
      return failure();
    }
    if (failed(verifyTileBufCommon(*this, srcTile, "src")))
      return failure();
    for (auto [idx, dim] : llvm::enumerate(dstPart.getShape())) {
      if (dim != ShapedType::kDynamic && dim <= 0) {
        emitOpError() << "expects dst shape[" << idx << "] to be positive";
        return failure();
      }
    }
    auto srcValid = srcTile.getValidShape();
    for (auto [idx, dim] : llvm::enumerate(srcValid)) {
      if (dim != ShapedType::kDynamic && dim <= 0) {
        emitOpError() << "expects src valid_shape[" << idx << "] to be positive";
        return failure();
      }
    }
    return std::make_pair(srcTile, dstPart);
  };

  auto isLoadStoreElemType = [&](Type ty) -> bool {
    return ty.isInteger(8) || ty.isInteger(16) || ty.isInteger(32) ||
           ty.isInteger(64) || ty.isF16() || ty.isBF16() || ty.isF32();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    auto common = verifyCommon();
    if (failed(common))
      return failure();
    auto [srcTile, dstPart] = *common;
    auto srcSpace = getPTOMemorySpaceEnum(srcTile);
    if (!srcSpace || (*srcSpace != pto::AddressSpace::VEC &&
                      *srcSpace != pto::AddressSpace::MAT &&
                      *srcSpace != pto::AddressSpace::ACC))
      return emitOpError("expects A2/A3 tstore src to use loc=vec, loc=mat, or loc=acc");

    Type srcElem = srcTile.getElementType();
    Type dstElem = dstPart.getElementType();
    if (*srcSpace == pto::AddressSpace::VEC || *srcSpace == pto::AddressSpace::MAT) {
      if (!isLoadStoreElemType(srcElem))
        return emitOpError("expects A2/A3 vec/mat tstore src element type to be i8/u8/i16/u16/i32/u32/i64/u64/f16/bf16/f32");
      if (getElemByteSize(srcElem) != getElemByteSize(dstElem))
        return emitOpError("expects A2/A3 vec/mat tstore src and dst element types to have the same bitwidth");
      return success();
    }

    if (!(srcElem.isSignlessInteger(32) || srcElem.isF32()))
      return emitOpError("expects A2/A3 acc tstore src element type to be i32 or f32");
    if (!(dstElem.isSignlessInteger(32) || dstElem.isF32() || dstElem.isF16() ||
          dstElem.isBF16()))
      return emitOpError("expects A2/A3 acc tstore dst element type to be i32/f32/f16/bf16");

    auto srcShape = srcTile.getShape();
    if (srcShape[1] != ShapedType::kDynamic &&
        (srcShape[1] < 1 || srcShape[1] > 4095))
      return emitOpError("expects A2/A3 acc tstore src cols to be in [1, 4095]");
    auto srcValid = srcTile.getValidShape();
    if (srcValid[1] != ShapedType::kDynamic &&
        (srcValid[1] < 1 || srcValid[1] > 4095))
      return emitOpError("expects A2/A3 acc tstore src valid_shape[1] to be in [1, 4095]");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    auto common = verifyCommon();
    if (failed(common))
      return failure();
    auto [srcTile, dstPart] = *common;
    auto srcSpace = getPTOMemorySpaceEnum(srcTile);
    if (!srcSpace || (*srcSpace != pto::AddressSpace::VEC &&
                      *srcSpace != pto::AddressSpace::ACC))
      return emitOpError("expects A5 tstore src to use loc=vec or loc=acc");

    Type srcElem = srcTile.getElementType();
    Type dstElem = dstPart.getElementType();
    if (*srcSpace == pto::AddressSpace::VEC) {
      if (!isLoadStoreElemType(srcElem))
        return emitOpError("expects A5 vec tstore src element type to be i8/u8/i16/u16/i32/u32/i64/u64/f16/bf16/f32");
      if (getElemByteSize(srcElem) != getElemByteSize(dstElem))
        return emitOpError("expects A5 vec tstore src and dst element types to have the same bitwidth");
      return success();
    }

    if (!(srcElem.isSignlessInteger(32) || srcElem.isF32()))
      return emitOpError("expects A5 acc tstore src element type to be i32 or f32");
    if (!(dstElem.isSignlessInteger(32) || dstElem.isF32() || dstElem.isF16() ||
          dstElem.isBF16()))
      return emitOpError("expects A5 acc tstore dst element type to be i32/f32/f16/bf16");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAbsOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
      failed(verifyVecTileCommon(*this, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameShapeAndElem(*this, srcTy, dstTy, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
    return failure();

  Type elemTy;
  if (auto tb = dyn_cast<pto::TileBufType>(srcTy))
    elemTy = tb.getElementType();
  else if (auto mr = dyn_cast<MemRefType>(srcTy))
    elemTy = mr.getElementType();
  if (!(elemTy.isF16() || elemTy.isF32()))
    return emitOpError() << "expects element type to be f16 or f32";

  return success();
}
// PTO.cpp

static bool isPTOShapedLike(Type ty) {
  return ty.isa<MemRefType, RankedTensorType,
                pto::TileBufType, pto::PartitionTensorViewType>();
}

static bool isTileLikeType(Type ty) {
  return isa<pto::TileBufType, MemRefType>(ty);
}

static Type getElemTy(Type ty) {
  if (auto mr = ty.dyn_cast<MemRefType>()) return mr.getElementType();
  if (auto tt = ty.dyn_cast<RankedTensorType>()) return tt.getElementType();
  if (auto tb = ty.dyn_cast<pto::TileBufType>()) return tb.getElementType();
  if (auto tv = ty.dyn_cast<pto::PartitionTensorViewType>()) return tv.getElementType();
  return Type();
}

static SmallVector<int64_t, 4> getShapeVec(Type ty) {
  SmallVector<int64_t, 4> s;
  if (auto mr = ty.dyn_cast<MemRefType>())
    return SmallVector<int64_t,4>(mr.getShape().begin(), mr.getShape().end());
  if (auto tt = ty.dyn_cast<RankedTensorType>())
    return SmallVector<int64_t,4>(tt.getShape().begin(), tt.getShape().end());
  if (auto tb = ty.dyn_cast<pto::TileBufType>())
    return SmallVector<int64_t,4>(tb.getShape().begin(), tb.getShape().end());
  if (auto tv = ty.dyn_cast<pto::PartitionTensorViewType>())
    return SmallVector<int64_t,4>(tv.getShape().begin(), tv.getShape().end());
  return {};
}

static SmallVector<int64_t, 4> getValidShapeVec(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty))
    return SmallVector<int64_t, 4>(tb.getValidShape().begin(), tb.getValidShape().end());
  return getShapeVec(ty);
}

static int64_t getConstantIndexOrDynamic(Value value) {
  if (!value)
    return ShapedType::kDynamic;
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  return ShapedType::kDynamic;
}

static SmallVector<int64_t, 4> getValidShapeVec(Value value) {
  if (!value)
    return {};
  auto valid = getValidShapeVec(value.getType());
  if (auto bind = value.getDefiningOp<pto::BindTileOp>()) {
    if (valid.size() >= 1 && bind.getValidRow())
      valid[0] = getConstantIndexOrDynamic(bind.getValidRow());
    if (valid.size() >= 2 && bind.getValidCol())
      valid[1] = getConstantIndexOrDynamic(bind.getValidCol());
  }
  return valid;
}

static std::optional<pto::AddressSpace> getPTOMemorySpaceEnum(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (auto as = dyn_cast_or_null<pto::AddressSpaceAttr>(tb.getMemorySpace()))
      return as.getAddressSpace();
    return std::nullopt;
  }
  if (auto mr = dyn_cast<MemRefType>(ty)) {
    if (auto as = dyn_cast_or_null<pto::AddressSpaceAttr>(mr.getMemorySpace()))
      return as.getAddressSpace();
    if (!mr.getMemorySpace())
      return pto::AddressSpace::GM;
  }
  return std::nullopt;
}

static bool isRank2TileBuf(Type ty) {
  auto tb = dyn_cast<pto::TileBufType>(ty);
  return tb && tb.getRank() == 2 && tb.getValidShape().size() == 2;
}

static bool isSupportedVecElemType(Type ty, bool allowBf16,
                                   bool allowInt8) {
  if (ty.isF16() || ty.isF32())
    return true;
  if (allowBf16 && ty.isBF16())
    return true;
  if (auto it = dyn_cast<IntegerType>(ty)) {
    if (!it.isSignless() && !it.isUnsigned())
      return false;
    switch (it.getWidth()) {
    case 32:
    case 16:
      return true;
    case 8:
      return allowInt8;
    default:
      return false;
    }
  }
  return false;
}

static LogicalResult verifyTileBufCommon(Operation *op, Type ty, StringRef name) {
  auto tb = dyn_cast<pto::TileBufType>(ty);
  if (tb) {
    if (tb.getRank() != 2)
      return op->emitOpError() << "expects " << name << " to be a rank-2 tile_buf";
  } else if (auto mr = dyn_cast<MemRefType>(ty)) {
    if (mr.getRank() != 2)
      return op->emitOpError() << "expects " << name << " to be a rank-2 memref";
  } else {
    return op->emitOpError() << "expects " << name << " to be a !pto.tile_buf or rank-2 memref";
  }

  auto validShape = getValidShapeVec(ty);
  if (validShape.size() != 2)
    return op->emitOpError() << "expects " << name << " to have a rank-2 valid_shape";
  auto shape = getShapeVec(ty);
  for (unsigned i = 0; i < 2; ++i) {
    if (shape[i] != ShapedType::kDynamic && validShape[i] != ShapedType::kDynamic &&
        validShape[i] > shape[i])
      return op->emitOpError() << "expects " << name << " to satisfy valid_shape[" << i
                               << "] <= shape[" << i << "]";
  }
  return success();
}

static LogicalResult verifyTileBufSameShapeAndElem(Operation *op, Type lhs, Type rhs,
                                                   StringRef lhsName,
                                                   StringRef rhsName) {
  if (!isTileLikeType(lhs) || !isTileLikeType(rhs))
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to be !pto.tile_buf or memref";
  if (getElemTy(lhs) != getElemTy(rhs))
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to have the same element type";
  if (getShapeVec(lhs) != getShapeVec(rhs))
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to have the same shape";
  return success();
}

static LogicalResult verifyTileBufSameValidShape(Operation *op, Type lhs, Type rhs,
                                                 StringRef lhsName, StringRef rhsName) {
  if (!isTileLikeType(lhs) || !isTileLikeType(rhs))
    return success();
  auto lhsValid = getValidShapeVec(lhs);
  auto rhsValid = getValidShapeVec(rhs);
  for (size_t i = 0; i < lhsValid.size() && i < rhsValid.size(); ++i) {
    if (lhsValid[i] != ShapedType::kDynamic && rhsValid[i] != ShapedType::kDynamic &&
        lhsValid[i] != rhsValid[i])
      return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                               << " to have the same valid_shape";
  }
  if (lhsValid.size() != rhsValid.size())
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to have the same valid_shape";
  return success();
}

static LogicalResult verifyPartialValidPattern(Operation *op, Type src0Ty,
                                               Type src1Ty, Type dstTy) {
  auto src0Valid = getValidShapeVec(src0Ty);
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src0Valid.size() != 2 || src1Valid.size() != 2 || dstValid.size() != 2)
    return op->emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");

  auto lessEqualKnown = [](int64_t lhs, int64_t rhs) {
    return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic || lhs <= rhs;
  };
  auto equalsKnown = [](ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
    for (auto [a, b] : llvm::zip(lhs, rhs)) {
      if (a != ShapedType::kDynamic && b != ShapedType::kDynamic && a != b)
        return false;
    }
    return true;
  };

  for (unsigned i = 0; i < 2; ++i) {
    if (!lessEqualKnown(src0Valid[i], dstValid[i]) ||
        !lessEqualKnown(src1Valid[i], dstValid[i]))
      return op->emitOpError(
          "expects src0/src1 valid_shape to be less than or equal to dst valid_shape");
  }
  if (!equalsKnown(src0Valid, dstValid) && !equalsKnown(src1Valid, dstValid))
    return op->emitOpError(
        "expects at least one of src0/src1 valid_shape to match dst valid_shape");
  return success();
}

static bool hasKnownZeroValidRegion(Type ty) {
  auto valid = getValidShapeVec(ty);
  if (valid.size() != 2)
    return false;
  return valid[0] == 0 || valid[1] == 0;
}

static LogicalResult verifyScalarTileOp(Operation *op, Type srcTy, Type dstTy,
                                        StringRef srcName, StringRef dstName,
                                        bool requireValidRowsEqual,
                                        bool requireValidColsEqual) {
  if (failed(verifyTileBufCommon(op, srcTy, srcName)) ||
      failed(verifyTileBufCommon(op, dstTy, dstName)))
    return failure();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!srcSpace || *srcSpace != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << srcName
                             << " to be in the vec address space";
  if (!dstSpace || *dstSpace != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << dstName
                             << " to be in the vec address space";
  if (failed(verifyTileBufSameShapeAndElem(op, srcTy, dstTy, srcName, dstName)))
    return failure();

  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != 2 || dstValid.size() != 2)
    return op->emitOpError()
           << "expects " << srcName << " and " << dstName
           << " to have rank-2 valid_shape";
  if (requireValidRowsEqual &&
      srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
      srcValid[0] != dstValid[0])
    return op->emitOpError()
           << "expects " << srcName << " and " << dstName
           << " to have the same valid_shape[0]";
  if (requireValidColsEqual &&
      srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1])
    return op->emitOpError()
           << "expects " << srcName << " and " << dstName
           << " to have the same valid_shape[1]";
  return success();
}

static bool hasCompatibleKnownExtent(int64_t lhs, int64_t rhs) {
  return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic || lhs == rhs;
}

static bool isKnownUnitExtent(int64_t value) {
  return value == ShapedType::kDynamic || value == 1;
}

static LogicalResult verifyVecTileStorage(Operation *op, Type ty, StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name)))
    return failure();
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  return success();
}

static LogicalResult verifyVecTileCommonA2A3(Operation *op, Type ty,
                                             StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name)))
    return failure();
  auto tb = dyn_cast<pto::TileBufType>(ty);
  auto as = getPTOMemorySpaceEnum(ty);
  if (as && *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  if (tb && tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor))
    return op->emitOpError() << "expects " << name << " to use the row_major blayout";
  return success();
}

static LogicalResult verifyVecTileCommonA5(Operation *op, Type ty,
                                           StringRef name) {
  return verifyVecTileCommonA2A3(op, ty, name);
}

static LogicalResult verifyVecTileCommon(Operation *op, Type ty, StringRef name) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyVecTileCommonA2A3(op, ty, name);
  case VerifierTargetArch::A5:
    return verifyVecTileCommonA5(op, ty, name);
  }
}

static LogicalResult verifyVecTileUnaryOp(Operation *op, Type srcTy, Type dstTy,
                                          StringRef srcName,
                                          StringRef dstName,
                                          bool allowBf16,
                                          bool allowInt8) {
  if (failed(verifyVecTileCommon(op, srcTy, srcName)) ||
      failed(verifyVecTileCommon(op, dstTy, dstName)))
    return failure();
  if (failed(verifyTileBufSameShapeAndElem(op, srcTy, dstTy, srcName, dstName)))
    return failure();
  if (!isSupportedVecElemType(getElemTy(srcTy), allowBf16, allowInt8))
    return op->emitOpError() << "expects vec tile element types to be supported";
  return success();
}

static LogicalResult verifyVecTileBinaryOp(Operation *op, Type src0Ty, Type src1Ty,
                                           Type dstTy,
                                           bool allowBf16,
                                           bool allowInt8) {
  if (failed(verifyVecTileCommon(op, src0Ty, "src0")) ||
      failed(verifyVecTileCommon(op, src1Ty, "src1")) ||
      failed(verifyVecTileCommon(op, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameShapeAndElem(op, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameShapeAndElem(op, src0Ty, dstTy, "src0", "dst")))
    return failure();
  if (!isSupportedVecElemType(getElemTy(src0Ty), allowBf16, allowInt8))
    return op->emitOpError() << "expects vec tile element types to be supported";
  return success();
}

static LogicalResult verifyAccTileCommonA2A3(Operation *op, Type ty,
                                             StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name)))
    return failure();
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::ACC)
    return op->emitOpError() << "expects " << name << " to be in the acc address space";
  return success();
}

static LogicalResult verifyAccTileCommonA5(Operation *op, Type ty,
                                           StringRef name) {
  return verifyAccTileCommonA2A3(op, ty, name);
}

static LogicalResult verifyAccTileCommon(Operation *op, Type ty, StringRef name) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyAccTileCommonA2A3(op, ty, name);
  case VerifierTargetArch::A5:
    return verifyAccTileCommonA5(op, ty, name);
  }
}

static LogicalResult verifyMatTileOperandsA2A3(Operation *op, Type lhsTy,
                                               Type rhsTy, Type dstTy) {
  if (failed(verifyTileBufCommon(op, lhsTy, "lhs")) ||
      failed(verifyTileBufCommon(op, rhsTy, "rhs")) ||
      failed(verifyAccTileCommon(op, dstTy, "dst")))
    return failure();
  auto lhsSpace = getPTOMemorySpaceEnum(lhsTy);
  auto rhsSpace = getPTOMemorySpaceEnum(rhsTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!lhsSpace || !rhsSpace || !dstSpace)
    return op->emitOpError("expects lhs, rhs, and dst to have explicit address spaces");
  if (*lhsSpace != pto::AddressSpace::LEFT || *rhsSpace != pto::AddressSpace::RIGHT ||
      *dstSpace != pto::AddressSpace::ACC)
    return op->emitOpError(
        "expects lhs, rhs, and dst to use the left, right, and acc address spaces");
  auto lhsShape = getShapeVec(lhsTy);
  auto rhsShape = getShapeVec(rhsTy);
  auto dstShape = getShapeVec(dstTy);
  if ((lhsShape[0] != dstShape[0] || rhsShape[1] != dstShape[1] || lhsShape[1] != rhsShape[0]))
    return op->emitOpError(
        "expects static matmul tile shapes lhs[M,K], rhs[K,N], and dst[M,N]");
  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  if (lhsValid.size() == 2 && rhsValid.size() == 2) {
    int64_t m = lhsValid[0];
    int64_t k = lhsValid[1];
    int64_t n = rhsValid[1];
    if ((m != ShapedType::kDynamic && (m < 1 || m > 4095)) ||
        (k != ShapedType::kDynamic && (k < 1 || k > 4095)) ||
        (n != ShapedType::kDynamic && (n < 1 || n > 4095)))
      return op->emitOpError("expects m, k, and n valid sizes to be in [1, 4095]");
  }
  return success();
}

static LogicalResult verifyMatTileOperandsA5(Operation *op, Type lhsTy,
                                             Type rhsTy, Type dstTy) {
  if (failed(verifyMatTileOperandsA2A3(op, lhsTy, rhsTy, dstTy)))
    return failure();

  auto lhsTb = mlir::dyn_cast<pto::TileBufType>(lhsTy);
  auto rhsTb = mlir::dyn_cast<pto::TileBufType>(rhsTy);
  auto dstTb = mlir::dyn_cast<pto::TileBufType>(dstTy);
  if (!lhsTb || !rhsTb || !dstTb)
    return success();

  if (lhsTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor))
    return op->emitOpError("expects lhs to use the col_major blayout on A5");
  if (rhsTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor))
    return op->emitOpError("expects rhs to use the row_major blayout on A5");
  if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor))
    return op->emitOpError("expects dst to use the col_major blayout on A5");

  if (lhsTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))
    return op->emitOpError("expects lhs to use the row_major slayout on A5");
  if (rhsTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor))
    return op->emitOpError("expects rhs to use the col_major slayout on A5");
  if (dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))
    return op->emitOpError("expects dst to use the row_major slayout on A5");
  return success();
}

static LogicalResult verifyMatTileOperands(Operation *op, Type lhsTy, Type rhsTy,
                                           Type dstTy) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyMatTileOperandsA2A3(op, lhsTy, rhsTy, dstTy);
  case VerifierTargetArch::A5:
    return verifyMatTileOperandsA5(op, lhsTy, rhsTy, dstTy);
  }
}

static LogicalResult verifyGemvTileOperandsA2A3(Operation *op, Type lhsTy,
                                                Type rhsTy, Type dstTy) {
  if (failed(verifyTileBufCommon(op, lhsTy, "lhs")) ||
      failed(verifyTileBufCommon(op, rhsTy, "rhs")) ||
      failed(verifyAccTileCommon(op, dstTy, "dst")))
    return failure();

  auto lhsSpace = getPTOMemorySpaceEnum(lhsTy);
  auto rhsSpace = getPTOMemorySpaceEnum(rhsTy);
  if (!lhsSpace || !rhsSpace)
    return op->emitOpError("expects lhs and rhs to have explicit address spaces");
  if (*lhsSpace != pto::AddressSpace::LEFT || *rhsSpace != pto::AddressSpace::RIGHT)
    return op->emitOpError(
        "expects lhs and rhs to use the left and right address spaces");

  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (lhsValid[0] != ShapedType::kDynamic && lhsValid[0] != 1)
    return op->emitOpError("expects lhs valid_shape[0] to be 1 for tgemv");
  if (isa<pto::TileBufType>(dstTy) && dstValid[0] != ShapedType::kDynamic &&
      dstValid[0] != 1)
    return op->emitOpError("expects dst valid_shape[0] to be 1 for tgemv");
  if (lhsValid[1] != ShapedType::kDynamic && rhsValid[0] != ShapedType::kDynamic &&
      lhsValid[1] != rhsValid[0])
    return op->emitOpError()
           << "expects lhs valid_shape[1] to equal rhs valid_shape[0], but got "
           << lhsValid[1] << " vs " << rhsValid[0];
  if (rhsValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      rhsValid[1] != dstValid[1])
    return op->emitOpError()
           << "expects rhs valid_shape[1] to equal dst valid_shape[1], but got "
           << rhsValid[1] << " vs " << dstValid[1];
  return success();
}

static LogicalResult verifyGemvTileOperandsA5(Operation *op, Type lhsTy,
                                              Type rhsTy, Type dstTy) {
  if (failed(verifyGemvTileOperandsA2A3(op, lhsTy, rhsTy, dstTy)))
    return failure();
  return verifyMatTileOperandsA5(op, lhsTy, rhsTy, dstTy);
}

static LogicalResult verifyGemvTileOperands(Operation *op, Type lhsTy, Type rhsTy,
                                            Type dstTy) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyGemvTileOperandsA2A3(op, lhsTy, rhsTy, dstTy);
  case VerifierTargetArch::A5:
    return verifyGemvTileOperandsA5(op, lhsTy, rhsTy, dstTy);
  }
}

static LogicalResult verifyMatBiasTileA2A3(Operation *op, Type biasTy, Type dstTy,
                                           bool requireFloatBias) {
  if (failed(verifyTileBufCommon(op, biasTy, "bias")))
    return failure();
  auto biasSpace = getPTOMemorySpaceEnum(biasTy);
  if (!biasSpace || *biasSpace != pto::AddressSpace::BIAS)
    return op->emitOpError("expects bias to be in the bias address space");
  auto biasShape = getShapeVec(biasTy);
  if (biasShape[0] != ShapedType::kDynamic && biasShape[0] != 1)
    return op->emitOpError("expects bias to have 1 row");
  if (requireFloatBias) {
    if (!getElemTy(biasTy).isF32())
      return op->emitOpError("expects bias to have element type f32");
  } else if (getElemTy(biasTy) != getElemTy(dstTy)) {
    return op->emitOpError("expects bias and dst to have the same element type");
  }
  return success();
}

static LogicalResult verifyMatBiasTileA5(Operation *op, Type biasTy, Type dstTy,
                                         bool requireFloatBias) {
  if (failed(verifyMatBiasTileA2A3(op, biasTy, dstTy, requireFloatBias)))
    return failure();
  if (auto biasTb = dyn_cast<pto::TileBufType>(biasTy)) {
    if (biasTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor))
      return op->emitOpError("expects bias to use the row_major blayout on A5");
  }
  return success();
}

static LogicalResult verifyMatBiasTile(Operation *op, Type biasTy, Type dstTy,
                                       bool requireFloatBias) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyMatBiasTileA2A3(op, biasTy, dstTy, requireFloatBias);
  case VerifierTargetArch::A5:
    return verifyMatBiasTileA5(op, biasTy, dstTy, requireFloatBias);
  }
}

static LogicalResult verifyMatmulTypeTriple(Operation *op, Type lhsElemTy,
                                            Type rhsElemTy, Type dstElemTy) {
  bool isA5 = getVerifierTargetArch(op) == VerifierTargetArch::A5;
  auto isInt8 = [](Type ty) {
    return ty.isInteger(8);
  };
  if (dstElemTy.isInteger(32) && isInt8(lhsElemTy) && isInt8(rhsElemTy))
    return success();

  auto isSupportedFpInput = [](Type ty) {
    return ty.isF16() || ty.isBF16() || ty.isF32();
  };
  if (dstElemTy.isF32() && lhsElemTy == rhsElemTy && isSupportedFpInput(lhsElemTy))
    return success();

  if (isA5 && dstElemTy.isF32() && lhsElemTy == rhsElemTy) {
    if (auto ft = mlir::dyn_cast<FloatType>(lhsElemTy)) {
      unsigned width = ft.getWidth();
      if (width == 8 || width == 16 || width == 32)
        return success();
    }
  }

  return op->emitOpError()
         << "expects (dst, lhs, rhs) element types to match one of "
            "(i32, i8, i8), (f32, f16, f16), (f32, bf16, bf16), (f32, f32, f32)"
            << (isA5 ? ", or an A5-supported fp8 pair" : "");
}

LogicalResult pto::TAddOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (failed(verifyTileBufCommon(*this, t0, "src0")) ||
        failed(verifyTileBufCommon(*this, t1, "src1")) ||
        failed(verifyTileBufCommon(*this, td, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, t0, t1, "src0", "src1")) ||
        failed(verifyTileBufSameShapeAndElem(*this, t0, td, "src0", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, t0, t1, "src0", "src1")) ||
        failed(verifyTileBufSameValidShape(*this, t0, td, "src0", "dst")))
      return failure();
    if (!isRowMajorTileBuf(t0) || !isRowMajorTileBuf(t1) || !isRowMajorTileBuf(td))
      return emitOpError("expects src0, src1, and dst to use row-major layout");
    Type elem = getElemTy(t0);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A2/A3 tadd element type to be i32/i16/f16/f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (failed(verifyTileBufCommon(*this, t0, "src0")) ||
        failed(verifyTileBufCommon(*this, t1, "src1")) ||
        failed(verifyTileBufCommon(*this, td, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, t0, t1, "src0", "src1")) ||
        failed(verifyTileBufSameShapeAndElem(*this, t0, td, "src0", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, t0, t1, "src0", "src1")) ||
        failed(verifyTileBufSameValidShape(*this, t0, td, "src0", "dst")))
      return failure();
    if (!isRowMajorTileBuf(t0) || !isRowMajorTileBuf(t1) || !isRowMajorTileBuf(td))
      return emitOpError("expects src0, src1, and dst to use row-major layout");
    Type elem = getElemTy(getSrc0().getType());
    if (elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
        elem.isF16() || elem.isBF16() || elem.isF32())
      return success();
    return emitOpError("expects A5 tadd element type to be i32/u32/i16/u16/i8/u8/f16/bf16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAddCOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type t0 = getSrc0().getType();
  Type t1 = getSrc1().getType();
  Type t2 = getSrc2().getType();
  Type td = getDst().getType();

  if (!isPTOShapedLike(t0) || !isPTOShapedLike(t1) ||
      !isPTOShapedLike(t2) || !isPTOShapedLike(td))
    return emitOpError("expects src0/src1/src2/dst to be memref/tile_buf types");

  auto s0 = getShapeVec(t0);
  auto s1 = getShapeVec(t1);
  auto s2 = getShapeVec(t2);
  auto sd = getShapeVec(td);
  if (s0 != s1 || s0 != s2 || s0 != sd)
    return emitOpError("expects src0/src1/src2/dst to have the same shape");
  return success();
}
LogicalResult pto::TAddSOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type ts = getSrc().getType();
    Type td = getDst().getType();
    if (failed(verifyScalarTileOp(*this, ts, td, "src", "dst",
                                  /*requireValidRowsEqual=*/true,
                                  /*requireValidColsEqual=*/true)))
      return failure();
    Type elem = getElemTy(ts);
    Type scalarTy = getScalar().getType();
    if (!scalarTy.isa<IntegerType, FloatType>())
      return emitOpError("scalar must be a scalar type (integer/float)");
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A2/A3 tadds element type to be i32/i16/f16/f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type ts = getSrc().getType();
    Type td = getDst().getType();
    if (failed(verifyScalarTileOp(*this, ts, td, "src", "dst",
                                  /*requireValidRowsEqual=*/false,
                                  /*requireValidColsEqual=*/true)))
      return failure();
    Type elem = getElemTy(ts);
    Type scalarTy = getScalar().getType();
    if (!scalarTy.isa< IntegerType, FloatType>())
      return emitOpError("scalar must be a scalar type (integer/float)");
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isBF16() || elem.isF32()))
      return emitOpError("expects A5 tadds element type to be i32/u32/i16/u16/i8/u8/f16/bf16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
LogicalResult pto::TAddSCOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type ts0 = getSrc0().getType();
  Type ts1 = getSrc1().getType();
  Type td = getDst().getType();
  if (!isPTOShapedLike(ts0) || !isPTOShapedLike(ts1) || !isPTOShapedLike(td))
    return emitOpError("expects src0/src1/dst to be PTO shaped-like types");

  auto s0 = getShapeVec(ts0);
  auto s1 = getShapeVec(ts1);
  auto sd = getShapeVec(td);
  if (s0 != s1 || s0 != sd)
    return emitOpError("expects src0/src1/dst to have the same shape");
  return success();
}

LogicalResult pto::TAndOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (failed(verifyTileBufCommon(*this, t0, "src0")) ||
        failed(verifyTileBufCommon(*this, t1, "src1")) ||
        failed(verifyTileBufCommon(*this, td, "dst")))
      return failure();

    Type e0 = getElemTy(t0);
    Type e1 = getElemTy(t1);
    Type ed = getElemTy(td);
    if (!e0 || !e1 || !ed) {
      emitOpError("failed to get element type for operands");
      return failure();
    }
    if (e0 != e1 || e0 != ed) {
      emitOpError("expects src0, src1, and dst to have the same element type");
      return failure();
    }
    if (!isRowMajorTileBuf(t0) || !isRowMajorTileBuf(t1) || !isRowMajorTileBuf(td)) {
      emitOpError("expects src0, src1, and dst to use row-major layout");
      return failure();
    }
    if (failed(verifyTileBufSameValidShape(*this, t0, td, "src0", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, t1, td, "src1", "dst")))
      return failure();
    return e0;
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16))
      return emitOpError(
          "expects A2/A3 tand src0, src1, and dst element type to be i8/u8/i16/u16");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32))
      return emitOpError(
          "expects A5 tand src0, src1, and dst element type to be i8/u8/i16/u16/i32/u32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAndSOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    if (getSrc() == getDst()) {
      emitOpError("expects src and dst to use different storage");
      return failure();
    }
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();

    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !dstElem) {
      emitOpError("failed to get element type for src/dst");
      return failure();
    }
    if (srcElem != dstElem) {
      emitOpError("expects src and dst to have the same element type");
      return failure();
    }
    if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
      emitOpError("expects src and dst to use row-major layout");
      return failure();
    }
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    return srcElem;
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16))
      return emitOpError(
          "expects A2/A3 tands src, scalar, and dst element type to be i8/u8/i16/u16");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32))
      return emitOpError(
          "expects A5 tands src, scalar, and dst element type to be i8/u8/i16/u16/i32/u32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TCIOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, dstTy, "dst")))
    return failure();

  auto elemTy = getElemTy(dstTy).dyn_cast<IntegerType>();
  if (!elemTy)
    return emitOpError("expects dst element type to be integer");

  unsigned bw = elemTy.getWidth();
  if (bw != 16 && bw != 32)
    return emitOpError("expects dst element type to be i16/u16/i32/u32");

  auto sTy = getS().getType().dyn_cast<IntegerType>();
  if (!sTy)
    return emitOpError("expects S to be integer");

  if (sTy != elemTy)
    return emitOpError("expects S and dst element type to be exactly the same type");
  auto shape = getShapeVec(dstTy);
  if (shape.size() != 2)
    return emitOpError("expects dst to be rank-2");
  if (shape[1] != ShapedType::kDynamic && shape[1] == 1)
    return emitOpError("expects dst cols to be different from 1");

  return success();
}
LogicalResult pto::TCmpOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (failed(verifyVecTileStorage(*this, t0, "src0")) ||
        failed(verifyVecTileStorage(*this, t1, "src1")) ||
        failed(verifyVecTileStorage(*this, td, "dst")))
      return failure();

    Type e0 = getElemTy(t0);
    Type e1 = getElemTy(t1);
    Type ed = getElemTy(td);
    if (!e0 || !e1 || !ed)
      return emitOpError("failed to get element type for src0/src1/dst");
    if (e0 != e1)
      return emitOpError("expects src0 and src1 to have the same element type");
    if (!(e0.isInteger(32) || e0.isF16() || e0.isF32()))
      return emitOpError("expects A2/A3 tcmp input element type to be i32/f16/f32");
    if (auto it = dyn_cast<IntegerType>(ed)) {
      if (it.getWidth() != 8)
        return emitOpError("expects dst element type to be i8/u8");
    } else {
      return emitOpError("expects dst element type to be i8/u8");
    }

    if (getShapeVec(t0) != getShapeVec(t1) || getShapeVec(t0) != getShapeVec(td))
      return emitOpError("expects src0, src1, and dst to have the same shape");
    if (failed(verifyTileBufSameValidShape(*this, t0, td, "src0", "dst")))
      return failure();
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (failed(verifyTileBufCommon(*this, t0, "src0")) ||
        failed(verifyTileBufCommon(*this, t1, "src1")) ||
        failed(verifyTileBufCommon(*this, td, "dst")))
      return failure();

    Type e0 = getElemTy(t0);
    Type e1 = getElemTy(t1);
    Type ed = getElemTy(td);
    if (!e0 || !e1 || !ed)
      return emitOpError("failed to get element type for src0/src1/dst");
    if (e0 != e1)
      return emitOpError("expects src0 and src1 to have the same element type");
    bool inputOk = e0.isF16() || e0.isF32() || e0.isBF16() ||
                   e0.isInteger(8) || e0.isInteger(16) || e0.isInteger(32);
    if (!inputOk)
      return emitOpError("expects A5 tcmp input element type to be i8/u8/i16/u16/i32/u32/f16/bf16/f32");
    if (auto it = dyn_cast<IntegerType>(ed)) {
      if (it.getWidth() != 8)
        return emitOpError("expects dst element type to be i8/u8");
    } else {
      return emitOpError("expects dst element type to be i8/u8");
    }

    if (getShapeVec(t0) != getShapeVec(t1) || getShapeVec(t0) != getShapeVec(td))
      return emitOpError("expects src0, src1, and dst to have the same shape");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

// ---- TCMPS verify ----
LogicalResult pto::TCmpSOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst")))
      return failure();

    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(16) || elemTy.isInteger(32) ||
          elemTy.isF16() || elemTy.isF32()))
      return emitOpError("expects A2/A3 tcmps input element type to be i16/u16/i32/u32/f16/f32");

    auto scalarTy = getScalar().getType();
    if (!(scalarTy.isIntOrIndexOrFloat()))
      return emitOpError("expects scalar to be integer, index, or float");

    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() != 2 || dstValid.size() != 2)
      return emitOpError("expects src and dst to have rank-2 valid_shape");
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        srcValid[0] != dstValid[0])
      return emitOpError("expects src and dst to have the same valid_shape[0]");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst")))
      return failure();

    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(8) || elemTy.isInteger(16) || elemTy.isInteger(32) ||
          elemTy.isF16() || elemTy.isF32()))
      return emitOpError("expects A5 tcmps input element type to be i8/u8/i16/u16/i32/u32/f16/f32");

    auto scalarTy = getScalar().getType();
    if (!(scalarTy.isIntOrIndexOrFloat()))
      return emitOpError("expects scalar to be integer, index, or float");

    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() != 2 || dstValid.size() != 2)
      return emitOpError("expects src and dst to have rank-2 valid_shape");
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        srcValid[0] != dstValid[0])
      return emitOpError("expects src and dst to have the same valid_shape[0]");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
LogicalResult pto::TColExpandOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
      failed(verifyNDStyleVecTile(*this, dstTy, "dst")))
    return failure();
  if (getElemTy(srcTy) != getElemTy(dstTy))
    return emitOpError("expects src and dst to have the same element type");
  if (!isSupportedVecElemType(getElemTy(srcTy), /*allowBf16=*/true,
                              /*allowInt8=*/true))
    return emitOpError("expects tcolexpand element type to be supported");
  auto srcValid = getValidShapeVec(getSrc());
  auto dstValid = getValidShapeVec(getDst());
  if (srcValid.size() != 2 || dstValid.size() != 2)
    return emitOpError("expects src and dst to have rank-2 valid_shape");
  if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1])
    return emitOpError("expects src and dst to have the same valid_shape[1]");
  return success();
}
static LogicalResult verifyTColExpandBinaryLikeOp(Operation *op, Type t0, Type t1,
                                                  Type td) {
  if (!isPTOShapedLike(t0) || !isPTOShapedLike(t1) || !isPTOShapedLike(td))
    return op->emitOpError("expects src0/src1/dst to be PTO shaped-like types");

  Type e0 = getElemTy(t0);
  Type e1 = getElemTy(t1);
  Type ed = getElemTy(td);
  if (!e0 || !e1 || !ed)
    return op->emitOpError("failed to get element type for src0/src1/dst");

  auto isSupportedFloat = [](Type elemTy) {
    auto ft = elemTy.dyn_cast<FloatType>();
    return ft && (ft.isF16() || ft.isF32());
  };
  if (!isSupportedFloat(e0) || !isSupportedFloat(e1) || !isSupportedFloat(ed))
    return op->emitOpError("expects src0/src1/dst element type to be f16 or f32");

  if (getShapeVec(t0) != getShapeVec(td))
    return op->emitOpError("expects src0/dst to have same shape");

  if (auto src1TileTy = dyn_cast<TileBufType>(t1)) {
    if (src1TileTy.getBLayoutValueI32() != 0)
      return op->emitOpError("expects src1 to use row-major layout");
  }
  if (auto dstTileTy = dyn_cast<TileBufType>(td)) {
    if (dstTileTy.getBLayoutValueI32() != 0)
      return op->emitOpError("expects dst to use row-major layout");
  }

  return success();
}
LogicalResult pto::TColExpandMulOp::verify() {
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType());
}
LogicalResult pto::TColExpandMaxOp::verify() {
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType());
}
LogicalResult pto::TColExpandMinOp::verify() {
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType());
}
LogicalResult pto::TColMaxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
        failed(verifyNDStyleVecTile(*this, dstTy, "dst")))
      return failure();
    if (getElemTy(srcTy) != getElemTy(dstTy))
      return emitOpError("expects src and dst to have the same element type");
    Type elem = getElemTy(srcTy);
    if (!(elem.isF16() || elem.isF32() || elem.isInteger(16) || elem.isInteger(32)))
      return emitOpError("expects A2/A3 tcolmax element type to be f16/f32/i16/i32");
    if (failed(verifyColReductionValidRegion(*this, srcTy, dstTy,
                                             /*requireNonZeroSrc=*/false)))
      return failure();
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
        failed(verifyNDStyleVecTile(*this, dstTy, "dst")))
      return failure();
    if (getElemTy(srcTy) != getElemTy(dstTy))
      return emitOpError("expects src and dst to have the same element type");
    Type elem = getElemTy(srcTy);
    if (!(elem.isF16() || elem.isF32() || elem.isBF16() ||
          elem.isInteger(8) || elem.isInteger(16) || elem.isInteger(32)))
      return emitOpError("expects A5 tcolmax element type to be i8/u8/i16/u16/i32/u32/f16/bf16/f32");
    if (failed(verifyColReductionValidRegion(*this, srcTy, dstTy,
                                             /*requireNonZeroSrc=*/true)))
      return failure();
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
LogicalResult pto::TColMinOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
        failed(verifyNDStyleVecTile(*this, dstTy, "dst")))
      return failure();
    if (getElemTy(srcTy) != getElemTy(dstTy))
      return emitOpError("expects src and dst to have the same element type");
    Type elem = getElemTy(srcTy);
    if (!(elem.isF16() || elem.isF32() || elem.isInteger(16) || elem.isInteger(32)))
      return emitOpError("expects A2/A3 tcolmin element type to be f16/f32/i16/i32");
    if (failed(verifyColReductionValidRegion(*this, srcTy, dstTy,
                                             /*requireNonZeroSrc=*/false)))
      return failure();
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
        failed(verifyNDStyleVecTile(*this, dstTy, "dst")))
      return failure();
    if (getElemTy(srcTy) != getElemTy(dstTy))
      return emitOpError("expects src and dst to have the same element type");
    Type elem = getElemTy(srcTy);
    if (!(elem.isF16() || elem.isF32() || elem.isBF16() ||
          elem.isInteger(8) || elem.isInteger(16) || elem.isInteger(32)))
      return emitOpError("expects A5 tcolmin element type to be i8/u8/i16/u16/i32/u32/f16/bf16/f32");
    if (failed(verifyColReductionValidRegion(*this, srcTy, dstTy,
                                             /*requireNonZeroSrc=*/true)))
      return failure();
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}



ParseResult mlir::pto::TColSumOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  OpAsmParser::UnresolvedOperand tmp;
  OpAsmParser::UnresolvedOperand dst;
  Type srcTy, tmpTy, dstTy;
  bool hasTmp = false;

  // Parse: ins(%src : type) or ins(%src, %tmp {isBinary = ...}: type, type)
  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(src))
    return failure();

  // Check for optional tmp operand (format 2)
  if (succeeded(parser.parseOptionalComma())) {
    // Format 2: ins(%src, %tmp {isBinary = ...}: type, type)
    if (parser.parseOperand(tmp))
      return failure();
    hasTmp = true;

    // Parse attributes (isBinary)
    if (parser.parseOptionalAttrDict(result.attributes))
      return failure();

    // Parse types: : type, type
    if (parser.parseColonType(srcTy) || parser.parseComma() || parser.parseType(tmpTy))
      return failure();
  } else {
    // Format 1: ins(%src : type)
    if (parser.parseColonType(srcTy))
      return failure();
  }

  if (parser.parseRParen())
    return failure();

  // Parse: outs(%dst : type)
  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen())
    return failure();

  // Parse any remaining attributes (for format 1)
  if (!hasTmp) {
    if (parser.parseOptionalAttrDict(result.attributes))
      return failure();
  }

  // Resolve operands
  if (parser.resolveOperand(src, srcTy, result.operands))
    return failure();

  int32_t tmpSize = hasTmp ? 1 : 0;

  if (hasTmp) {
    if (parser.resolveOperand(tmp, tmpTy, result.operands))
      return failure();
  }

  if (parser.resolveOperand(dst, dstTy, result.operands))
    return failure();

  return success();
}

void mlir::pto::TColSumOp::print(OpAsmPrinter &p) {
  if (getTmp()) {
    // Format 2: ins(%src, %tmp {isBinary = ...}: type, type) outs(%dst : type)
    p << " ins(" << getSrc() << ", " << getTmp();
    // Print isBinary attribute if present
    SmallVector<StringRef, 1> elidedAttrs;
    if (!getIsBinaryAttr() || getIsBinaryAttr().getValue() == false) {
      elidedAttrs.push_back("isBinary");
    }
    p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
    p << " : " << getSrc().getType() << ", " << getTmp().getType() << ")";
  } else {
    // Format 1: ins(%src : type) outs(%dst : type)
    p << " ins(" << getSrc() << " : " << getSrc().getType() << ")";
  }

  p << " outs(" << getDst() << " : " << getDst().getType() << ")";

  // Print remaining attributes for format 1 (excluding isBinary)
  if (!getTmp()) {
    SmallVector<StringRef, 1> elidedAttrs = {"isBinary"};
    p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  }
}

LogicalResult pto::TColSumOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
        failed(verifyNDStyleVecTile(*this, dstTy, "dst")))
      return failure();
    bool hasTmp = (bool)getTmp();
    bool hasIsBinary = (bool)getIsBinaryAttr();
    if (hasTmp != hasIsBinary) {
      if (hasTmp)
        return emitOpError("tmp operand requires isBinary attribute");
      return emitOpError("isBinary attribute requires tmp operand");
    }
    if (getTmp()) {
      Type tmpTy = getTmp().getType();
      if (failed(verifyNDStyleVecTile(*this, tmpTy, "tmp")))
        return failure();
      if (getElemTy(srcTy) != getElemTy(dstTy) || getElemTy(srcTy) != getElemTy(tmpTy))
        return emitOpError("expects src/tmp/dst element types to match");
    }
    if (getElemTy(srcTy) != getElemTy(dstTy))
      return emitOpError("expects src/dst element types to match");
    if (failed(verifyColReductionValidRegion(*this, srcTy, dstTy,
                                             /*requireNonZeroSrc=*/false)))
      return failure();
    Type elem = getElemTy(srcTy);
    if (!(elem.isF16() || elem.isF32() || elem.isInteger(16) || elem.isInteger(32)))
      return emitOpError("expects A2/A3 tcolsum element type to be f16/f32/i16/i32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
        failed(verifyNDStyleVecTile(*this, dstTy, "dst")))
      return failure();
    bool hasTmp = (bool)getTmp();
    bool hasIsBinary = (bool)getIsBinaryAttr();
    if (hasTmp != hasIsBinary) {
      if (hasTmp)
        return emitOpError("tmp operand requires isBinary attribute");
      return emitOpError("isBinary attribute requires tmp operand");
    }
    if (getTmp()) {
      Type tmpTy = getTmp().getType();
      if (failed(verifyNDStyleVecTile(*this, tmpTy, "tmp")))
        return failure();
      if (getElemTy(srcTy) != getElemTy(dstTy) || getElemTy(srcTy) != getElemTy(tmpTy))
        return emitOpError("expects src/tmp/dst element types to match");
    }
    if (getElemTy(srcTy) != getElemTy(dstTy))
      return emitOpError("expects src/dst element types to match");
    if (failed(verifyColReductionValidRegion(*this, srcTy, dstTy,
                                             /*requireNonZeroSrc=*/true)))
      return failure();
    Type elem = getElemTy(srcTy);
    if (!(elem.isF16() || elem.isF32() || elem.isBF16() || elem.isInteger(8) ||
          elem.isInteger(16) || elem.isInteger(32)))
      return emitOpError("expects A5 tcolsum element type to be i8/u8/i16/u16/i32/u32/f16/bf16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

llvm::LogicalResult mlir::pto::TCvtOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst")))
    return failure();
  if (getShapeVec(srcTy) != getShapeVec(dstTy))
    return emitOpError("expects src and dst to have compatible shapes");
  if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
    return failure();

  return mlir::success();
}

LogicalResult mlir::pto::TDivOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, src0Ty, "src0")) ||
        failed(verifyTileBufCommon(*this, src1Ty, "src1")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameShapeAndElem(*this, src0Ty, dstTy, "src0", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, dstTy, "src0", "dst")))
      return failure();
    if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
        !isRowMajorTileBuf(dstTy))
      return emitOpError("expects src0, src1, and dst to use row-major layout");
    auto elem0 = getElemTy(src0Ty);
    if (!(elem0.isF16() || elem0.isF32()))
      return emitOpError("expects A2/A3 tdiv element type to be f16 or f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, src0Ty, "src0")) ||
        failed(verifyTileBufCommon(*this, src1Ty, "src1")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameShapeAndElem(*this, src0Ty, dstTy, "src0", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, dstTy, "src0", "dst")))
      return failure();
    if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
        !isRowMajorTileBuf(dstTy))
      return emitOpError("expects src0, src1, and dst to use row-major layout");
    auto elem0 = getElemTy(src0Ty);
    if (!(elem0.isF16() || elem0.isF32() || elem0.isInteger(16) || elem0.isInteger(32)))
      return emitOpError("expects A5 tdiv element type to be i32/u32/i16/u16/f16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TDivSOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyScalarTileOp(*this, srcTy, dstTy, "src", "dst",
                                  /*requireValidRowsEqual=*/true,
                                  /*requireValidColsEqual=*/true)))
      return failure();
    Type scalarTy = getScalar().getType();
    if (!scalarTy.isa<IntegerType, FloatType>())
      return emitOpError("scalar must be a scalar type (integer/float)");
    Type elem = getElemTy(srcTy);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A2/A3 tdivs element type to be i32/i16/f16/f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyScalarTileOp(*this, srcTy, dstTy, "src", "dst",
                                  /*requireValidRowsEqual=*/true,
                                  /*requireValidColsEqual=*/true)))
      return failure();
    Type scalarTy = getScalar().getType();
    if (!scalarTy.isa<IntegerType, FloatType>())
      return emitOpError("scalar must be a scalar type (integer/float)");
    Type elem = getElemTy(srcTy);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isF32()))
      return emitOpError("expects A5 tdivs element type to be i32/u32/i16/u16/i8/u8/f16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TExpOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                    /*allowBf16=*/false, /*allowInt8=*/false)))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    Type srcElem = getElemTy(srcTy);
    if (!srcElem.isF16() && !srcElem.isF32())
      return emitOpError("expects element type to be f16 or f32");
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TExpandsOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                      *dstSpace != pto::AddressSpace::MAT))
      return emitOpError("expects dst to be in the vec or mat address space");
    Type dstElem = getElemTy(dstTy);
    Type scalarTy = getScalar().getType();
    if (scalarTy != dstElem)
      return emitOpError("expects scalar type == dst element type");
    if (*dstSpace == pto::AddressSpace::VEC && !isRowMajorTileBuf(dstTy))
      return emitOpError("expects vec dst to use row-major layout on A2/A3");
    if (dstElem.isF16() || dstElem.isF32())
      return mlir::success();
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(dstElem)) {
      unsigned w = it.getWidth();
      if (w == 16 || w == 32)
        return mlir::success();
    }
    return emitOpError("expects A2/A3 texpands dst element type to be i16/i32/f16/f32");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                      *dstSpace != pto::AddressSpace::MAT))
      return emitOpError("expects dst to be in the vec or mat address space");
    Type dstElem = getElemTy(dstTy);
    Type scalarTy = getScalar().getType();
    if (scalarTy != dstElem)
      return emitOpError("expects scalar type == dst element type");
    if (dstElem.isF16() || dstElem.isF32())
      return mlir::success();
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(dstElem)) {
      unsigned w = it.getWidth();
      if (w == 8 || w == 16 || w == 32)
        return mlir::success();
    }
    return emitOpError("expects A5 texpands dst element type to be i8/u8/i16/u16/i32/u32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TExtractOp::verify() {
  auto getConstIndex = [&](Value v) -> std::optional<int64_t> {
    auto cst = v.getDefiningOp<mlir::arith::ConstantOp>();
    if (!cst)
      return std::nullopt;
    if (auto attr = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
      return attr.getInt();
    return std::nullopt;
  };
  auto verifyIndexOperands = [&]() -> LogicalResult {
    if (!getIndexRow().getType().isIndex() || !getIndexCol().getType().isIndex())
      return emitOpError("expects indexRow and indexCol to be index type");
    auto row = getConstIndex(getIndexRow());
    auto col = getConstIndex(getIndexCol());
    if (row && *row < 0)
      return emitOpError("expects indexRow to be non-negative");
    if (col && *col < 0)
      return emitOpError("expects indexCol to be non-negative");
    return success();
  };
  auto verifyStaticBounds = [&](Type srcTy, Type dstTy) -> LogicalResult {
    auto row = getConstIndex(getIndexRow());
    auto col = getConstIndex(getIndexCol());
    auto srcShape = getShapeVec(srcTy);
    auto dstShape = getShapeVec(dstTy);
    if (srcShape.size() != 2 || dstShape.size() != 2)
      return emitOpError("expects src and dst to be rank-2 tile_buf");
    if (row && srcShape[0] != ShapedType::kDynamic &&
        dstShape[0] != ShapedType::kDynamic &&
        *row + dstShape[0] > srcShape[0])
      return emitOpError("expects indexRow + dst.rows <= src.rows");
    if (col && srcShape[1] != ShapedType::kDynamic &&
        dstShape[1] != ShapedType::kDynamic &&
        *col + dstShape[1] > srcShape[1])
      return emitOpError("expects indexCol + dst.cols <= src.cols");
    return success();
  };
  auto hasMatExtractSourceLayoutA2A3 = [&](pto::TileBufType srcTy) -> bool {
    int32_t bl = srcTy.getBLayoutValueI32();
    int32_t sl = srcTy.getSLayoutValueI32();
    return bl == static_cast<int32_t>(pto::BLayout::RowMajor) ||
           (bl != static_cast<int32_t>(pto::BLayout::RowMajor) &&
            sl == static_cast<int32_t>(pto::SLayout::RowMajor));
  };
  auto hasMatExtractSourceLayoutA5 = [&](pto::TileBufType srcTy,
                                         pto::AddressSpace dstSpace) -> bool {
    int32_t bl = srcTy.getBLayoutValueI32();
    int32_t sl = srcTy.getSLayoutValueI32();
    if (dstSpace == pto::AddressSpace::LEFT) {
      return (bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
              sl == static_cast<int32_t>(pto::SLayout::ColMajor)) ||
             (bl != static_cast<int32_t>(pto::BLayout::RowMajor) &&
              sl == static_cast<int32_t>(pto::SLayout::RowMajor)) ||
             bl == static_cast<int32_t>(pto::BLayout::RowMajor);
    }
    return (bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
            sl == static_cast<int32_t>(pto::SLayout::ColMajor)) ||
           (bl != static_cast<int32_t>(pto::BLayout::RowMajor) &&
            sl == static_cast<int32_t>(pto::SLayout::RowMajor));
  };
  auto isA2A3ExtractElemType = [&](Type ty) -> bool {
    return ty.isInteger(8) || ty.isF16() || ty.isBF16() || ty.isF32();
  };
  auto isA5ExtractElemType = [&](Type ty) -> bool {
    if (auto it = dyn_cast<IntegerType>(ty))
      return it.getWidth() == 8;
    if (auto ft = dyn_cast<FloatType>(ty))
      return ft.getWidth() == 8 || ft.isF16() || ft.isBF16() || ft.isF32();
    return false;
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
    auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
    if (!srcTb || !dstTb)
      return emitOpError("expects src and dst to be !pto.tile_buf");
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")) ||
        failed(verifyIndexOperands()) ||
        failed(verifyStaticBounds(srcTy, dstTy)))
      return failure();
    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !dstElem || srcElem != dstElem)
      return emitOpError("expects src and dst to have the same element type");
    if (!isA2A3ExtractElemType(dstElem))
      return emitOpError("expects A2/A3 textract element type to be i8/f16/bf16/f32");
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!srcSpace || *srcSpace != pto::AddressSpace::MAT)
      return emitOpError("expects A2/A3 textract src to use loc=mat");
    if (!dstSpace || (*dstSpace != pto::AddressSpace::LEFT &&
                      *dstSpace != pto::AddressSpace::RIGHT))
      return emitOpError("expects A2/A3 textract dst to use loc=left or loc=right");
    if (!hasMatExtractSourceLayoutA2A3(srcTb))
      return emitOpError("expects A2/A3 textract src to use a supported mat blayout/slayout combination");
    if (*dstSpace == pto::AddressSpace::LEFT) {
      if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
          dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))
        return emitOpError("expects A2/A3 left dst to use row_major blayout and row_major slayout");
    } else {
      if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
          dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor))
        return emitOpError("expects A2/A3 right dst to use row_major blayout and col_major slayout");
    }
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
    auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
    if (!srcTb || !dstTb)
      return emitOpError("expects src and dst to be !pto.tile_buf");
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")) ||
        failed(verifyIndexOperands()) ||
        failed(verifyStaticBounds(srcTy, dstTy)))
      return failure();
    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !dstElem || srcElem != dstElem)
      return emitOpError("expects src and dst to have the same element type");
    if (!isA5ExtractElemType(dstElem))
      return emitOpError("expects A5 textract element type to be an fp8/f16/bf16/f32 or int8 family type");
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!srcSpace || !dstSpace)
      return emitOpError("expects src and dst to have explicit loc");
    bool okPair =
        (*srcSpace == pto::AddressSpace::MAT &&
         (*dstSpace == pto::AddressSpace::LEFT ||
          *dstSpace == pto::AddressSpace::RIGHT ||
          *dstSpace == pto::AddressSpace::SCALING)) ||
        (*srcSpace == pto::AddressSpace::VEC &&
         *dstSpace == pto::AddressSpace::MAT);
    if (!okPair)
      return emitOpError("expects A5 textract to use a supported src/dst loc pair");
    if (*srcSpace == pto::AddressSpace::MAT) {
      if (!hasMatExtractSourceLayoutA5(srcTb, *dstSpace))
        return emitOpError("expects A5 textract src to use a supported mat blayout/slayout combination");
      if (*dstSpace == pto::AddressSpace::LEFT) {
        if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
            dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))
          return emitOpError("expects A5 left dst to use row_major blayout and row_major slayout");
      } else if (*dstSpace == pto::AddressSpace::RIGHT) {
        if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
            dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor))
          return emitOpError("expects A5 right dst to use row_major blayout and col_major slayout");
      }
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
mlir::LogicalResult mlir::pto::TInsertOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (!isPTOShapedLike(srcTy) || !isPTOShapedLike(dstTy))
    return emitOpError("expects src/dst to be PTO shaped-like types");

  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  if (srcShape.size() != 2 || dstShape.size() != 2)
    return emitOpError("expects rank-2 shaped types for src/dst");

  Type srcElemTy = getElemTy(srcTy);
  Type dstElemTy = getElemTy(dstTy);
  bool sameElemTy = srcElemTy == dstElemTy;
  bool castElemTy =
      srcElemTy.isF32() && (dstElemTy.isF16() || dstElemTy.isBF16());
  if (!sameElemTy && !castElemTy)
    return emitOpError(
        "expects src/dst element types to match, or src=f32 with dst=f16/bf16");

  if (!getIndexRow().getType().isIndex() || !getIndexCol().getType().isIndex())
    return emitOpError("expects indexRow/indexCol to be index type");

  auto readConstIndex = [&](Value v, int64_t &out) -> bool {
    if (auto cOp = v.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
      out = cOp.value();
      return true;
    }
    if (auto cInt = v.getDefiningOp<mlir::arith::ConstantIntOp>()) {
      out = cInt.value();
      return true;
    }
    if (auto cOp = v.getDefiningOp<mlir::arith::ConstantOp>()) {
      if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(cOp.getValue())) {
        out = ia.getInt();
        return true;
      }
    }
    return false;
  };

  int64_t r0 = 0;
  int64_t c0 = 0;
  bool rowConst = readConstIndex(getIndexRow(), r0);
  bool colConst = readConstIndex(getIndexCol(), c0);
  if (rowConst && r0 < 0)
    return emitOpError("indexRow must be non-negative");
  if (colConst && c0 < 0)
    return emitOpError("indexCol must be non-negative");

  int64_t srcRows = srcShape[0];
  int64_t srcCols = srcShape[1];
  int64_t dstRows = dstShape[0];
  int64_t dstCols = dstShape[1];
  if (rowConst && srcRows != mlir::ShapedType::kDynamic &&
      dstRows != mlir::ShapedType::kDynamic &&
      r0 + srcRows > dstRows)
    return emitOpError("indexRow + src rows exceeds dst rows");
  if (colConst && srcCols != mlir::ShapedType::kDynamic &&
      dstCols != mlir::ShapedType::kDynamic &&
      c0 + srcCols > dstCols)
    return emitOpError("indexCol + src cols exceeds dst cols");

  return mlir::success();
}

static mlir::LogicalResult verifyTFillPadLike(Operation *op, Type srcTy, Type dstTy,
                                              bool allowDstExpand,
                                              llvm::StringRef opName) {
  if (!isPTOShapedLike(srcTy) || !isPTOShapedLike(dstTy))
    return op->emitError("expects src/dst to be PTO shaped-like types");

  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  if (srcShape.size() != 2 || dstShape.size() != 2)
    return op->emitError("expects rank-2 shaped types for src/dst");

  auto srcElem = getElemTy(srcTy);
  auto dstElem = getElemTy(dstTy);

  auto getElemBytes = [](mlir::Type t) -> int64_t {
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(t))
      return it.getWidth() / 8;
    if (auto ft = mlir::dyn_cast<mlir::FloatType>(t))
      return ft.getWidth() / 8;
    return -1;
  };

  int64_t srcB = getElemBytes(srcElem);
  int64_t dstB = getElemBytes(dstElem);
  if (srcB < 0 || dstB < 0)
    return op->emitError("unsupported element type (expects int/float element types)");
  if (srcB != dstB)
    return op->emitError("expects sizeof(src element) == sizeof(dst element)");
  if (!(srcB == 1 || srcB == 2 || srcB == 4))
    return op->emitError("expects element size to be 1, 2, or 4 bytes");

  if (auto dstTileTy = mlir::dyn_cast<mlir::pto::TileBufType>(dstTy)) {
    auto padAttr = mlir::dyn_cast<mlir::pto::PadValueAttr>(dstTileTy.getPadValueAttr());
    if (!padAttr || padAttr.getValue() == mlir::pto::PadValue::Null)
      return op->emitError() << "expects dst PadVal != Null for " << opName;
  }

  if (!allowDstExpand) {
    if (srcShape != dstShape)
      return op->emitError()
             << "expects src and dst to have the same static shape for " << opName;
    return mlir::success();
  }

  if (srcShape[0] > dstShape[0] || srcShape[1] > dstShape[1]) {
    return op->emitError()
           << "expects dst static shape to be >= src static shape for " << opName;
  }

  return mlir::success();
}

mlir::LogicalResult mlir::pto::TFillPadOp::verify() {
  return verifyTFillPadLike(getOperation(), getSrc().getType(), getDst().getType(),
                            /*allowDstExpand=*/false, "tfillpad");
}

mlir::LogicalResult mlir::pto::TFillPadExpandOp::verify() {
  return verifyTFillPadLike(getOperation(), getSrc().getType(), getDst().getType(),
                            /*allowDstExpand=*/true, "tfillpad_expand");
}


// TGather: must provide exactly one of {indices operand, maskPattern attr}.
llvm::LogicalResult mlir::pto::TGatherOp::verify() {
  auto verifyCommon = [&](bool allow16BitIndices,
                          bool allowA5MaskTypes) -> LogicalResult {
    Value indices = getIndices();
    auto maskAttr = getMaskPatternAttr();
    const bool hasIdx = static_cast<bool>(indices);
    const bool hasMask = static_cast<bool>(maskAttr);
    if (hasIdx == hasMask)
      return emitOpError()
             << "expects exactly one of: 'indices' operand OR 'maskPattern' attribute";

    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();

    auto dstValid = getValidShapeVec(dstTy);
    auto dstShape = getShapeVec(dstTy);
    if (dstValid.size() == 2 && dstShape.size() == 2 &&
        dstValid[1] != ShapedType::kDynamic && dstShape[1] != ShapedType::kDynamic &&
        dstValid[1] != dstShape[1])
      return emitOpError("expects dst valid_shape[1] to equal dst cols");

    if (hasIdx) {
      Type idxTy = indices.getType();
      if (failed(verifyTileBufCommon(*this, idxTy, "indices")))
        return failure();
      Type srcElem = getElemTy(srcTy);
      Type dstElem = getElemTy(dstTy);
      if (!srcElem || !dstElem)
        return emitOpError("failed to get element type for src/dst");
      if (srcElem != dstElem)
        return emitOpError("expects src and dst to have the same element type");
      if (!isSupportedGatherElemTypeA2A3(srcElem) ||
          !isSupportedGatherElemTypeA2A3(dstElem))
        return emitOpError(
            "expects gather src/dst element type to be i16/u16/i32/u32/f16/f32");
      auto idxElem = dyn_cast<IntegerType>(getElemTy(idxTy));
      if (!idxElem)
        return emitOpError() << "indices element type must be integer";
      unsigned width = idxElem.getWidth();
      if (!(width == 32 || (allow16BitIndices && width == 16)))
        return emitOpError() << "expects indices element type to be i32/u32"
                             << (allow16BitIndices ? " or i16/u16" : "");
      auto idxValid = getValidShapeVec(idxTy);
      if (idxValid.size() == 2 && getShapeVec(idxTy).size() == 2 &&
          idxValid[1] != ShapedType::kDynamic &&
          getShapeVec(idxTy)[1] != ShapedType::kDynamic &&
          idxValid[1] != getShapeVec(idxTy)[1])
        return emitOpError("expects indices valid_shape[1] to equal indices cols");
      return success();
    }

    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !dstElem)
      return emitOpError("failed to get element type for src/dst");
    if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy))
      return emitOpError("expects src and dst to use row-major layout");
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::VEC ||
        *dstSpace != pto::AddressSpace::VEC)
      return emitOpError("expects src and dst to be in the vec address space");
    unsigned srcElemBytes = srcElem.getIntOrFloatBitWidth() / 8;
    unsigned dstElemBytes = dstElem.getIntOrFloatBitWidth() / 8;
    if (srcElemBytes != dstElemBytes)
      return emitOpError("expects src and dst element sizes to match");
    if (allowA5MaskTypes) {
      if (!(srcElemBytes == 1 || srcElemBytes == 2 || srcElemBytes == 4))
        return emitOpError("expects A5 mask-pattern gather element size to be 1, 2, or 4 bytes");
      if (!isSupportedGatherElemTypeA5(srcElem) || !isSupportedGatherElemTypeA5(dstElem))
        return emitOpError(
            "expects A5 mask-pattern gather src/dst element type to be i8/u8/i16/u16/i32/u32/f16/bf16/f32/fp8-like");
    } else {
      if (!(srcElemBytes == 2 || srcElemBytes == 4))
        return emitOpError("expects A2/A3 mask-pattern gather element size to be 2 or 4 bytes");
      if (!(isSupportedGatherElemTypeA2A3(srcElem) || srcElem.isBF16()) ||
          !(isSupportedGatherElemTypeA2A3(dstElem) || dstElem.isBF16()))
        return emitOpError(
            "expects A2/A3 mask-pattern gather src/dst element type to be i16/u16/i32/u32/f16/bf16/f32");
    }
    return success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyCommon(/*allow16BitIndices=*/false, /*allowA5MaskTypes=*/false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyCommon(/*allow16BitIndices=*/true, /*allowA5MaskTypes=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
mlir::LogicalResult mlir::pto::TGatherBOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<std::pair<Type, Type>> {
    Type srcTy = getSrc().getType();
    Type offTy = getOffsets().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, offTy, "offsets")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    auto offElem = getElemTy(offTy).dyn_cast<IntegerType>();
    if (!offElem || offElem.getWidth() != 32 || !offElem.isUnsigned())
      return emitOpError() << "expects offsets element type to be ui32";
    auto srcElemTy = getElemTy(srcTy);
    auto dstElemTy = getElemTy(dstTy);
    if (!srcElemTy || !dstElemTy)
      return emitOpError() << "failed to get element type for src/dst";
    return std::make_pair(srcElemTy, dstElemTy);
  };

  auto isSupportedGatherBElemType = [](Type ty) -> bool {
    if (ty.isBF16() || ty.isF16() || ty.isF32())
      return true;
    if (auto it = mlir::dyn_cast<IntegerType>(ty)) {
      unsigned width = it.getWidth();
      return width == 8 || width == 16 || width == 32;
    }
    return false;
  };

  auto getElemBytes = [](Type ty) -> std::optional<unsigned> {
    if (ty.isBF16())
      return 2;
    if (auto it = mlir::dyn_cast<IntegerType>(ty))
      return it.getWidth() / 8;
    if (auto ft = mlir::dyn_cast<FloatType>(ty))
      return ft.getWidth() / 8;
    return std::nullopt;
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<std::pair<Type, Type>> elems = verifyCommon();
    if (failed(elems))
      return failure();
    Type dstTy = getDst().getType();
    Type srcElemTy = elems->first;
    Type dstElemTy = elems->second;
    if (!isRowMajorTileBuf(dstTy))
      return emitOpError() << "expects dst to use row-major layout";
    auto dstBytes = getElemBytes(dstElemTy);
    if (!dstBytes || (*dstBytes != 1 && *dstBytes != 2 && *dstBytes != 4))
      return emitOpError() << "expects dst element size to be 1, 2, or 4 bytes";
    if (!isSupportedGatherBElemType(srcElemTy) || !isSupportedGatherBElemType(dstElemTy))
      return emitOpError() << "expects src/dst element type to be i8/u8/i16/u16/i32/u32/f16/bf16/f32";
    return mlir::success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<std::pair<Type, Type>> elems = verifyCommon();
    if (failed(elems))
      return failure();
    Type srcElemTy = elems->first;
    Type dstElemTy = elems->second;
    auto dstBytes = getElemBytes(dstElemTy);
    if (!dstBytes || (*dstBytes != 1 && *dstBytes != 2 && *dstBytes != 4))
      return emitOpError() << "expects dst element size to be 1, 2, or 4 bytes";
    if (!isSupportedGatherBElemType(srcElemTy) || !isSupportedGatherBElemType(dstElemTy))
      return emitOpError() << "expects src/dst element type to be i8/u8/i16/u16/i32/u32/f16/bf16/f32";
    return mlir::success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TLogOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false)))
    return failure();
  if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
    return failure();
  auto elemTy = getElemTy(srcTy);
  if (!(elemTy.isF16() || elemTy.isF32()))
    return emitOpError() << "expects element type to be f16 or f32";
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TLReluOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    auto valid = getValidShapeVec(srcTy);
    if (valid.size() != 2)
      return emitOpError("expects src to have rank-2 valid_shape");
    if (valid[0] != ShapedType::kDynamic && valid[0] <= 0)
      return emitOpError("expects src valid_shape[0] to be positive");
    if (valid[1] != ShapedType::kDynamic && valid[1] <= 0)
      return emitOpError("expects src valid_shape[1] to be positive");
    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isF16() || elemTy.isF32()))
      return emitOpError() << "expects A2/A3 tlrelu element type to be f16 or f32";
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isF16() || elemTy.isF32()))
      return emitOpError() << "expects A5 tlrelu element type to be f16 or f32";
    if (!getSlope().getType().isF32())
      return emitOpError() << "expects slope to have type f32";
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TMaxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, src0Ty, "src0")) ||
        failed(verifyTileBufCommon(*this, src1Ty, "src1")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameShapeAndElem(*this, src0Ty, dstTy, "src0", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, dstTy, "src0", "dst")))
      return failure();
    if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
        !isRowMajorTileBuf(dstTy))
      return emitOpError("expects src0, src1, and dst to use row-major layout");
    Type elem = getElemTy(src0Ty);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A2/A3 tmax element type to be i32/i16/f16/f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, src0Ty, "src0")) ||
        failed(verifyTileBufCommon(*this, src1Ty, "src1")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameShapeAndElem(*this, src0Ty, dstTy, "src0", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, dstTy, "src0", "dst")))
      return failure();
    if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
        !isRowMajorTileBuf(dstTy))
      return emitOpError("expects src0, src1, and dst to use row-major layout");
    Type elem = getElemTy(src0Ty);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isF32()))
      return emitOpError("expects A5 tmax element type to be i32/u32/i16/u16/i8/u8/f16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TMaxSOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyScalarTileOp(*this, srcTy, dstTy, "src", "dst",
                                  /*requireValidRowsEqual=*/true,
                                  /*requireValidColsEqual=*/true)))
      return failure();
    Type scalarTy = getScalar().getType();
    if (!scalarTy.isa<IntegerType, FloatType>())
      return emitOpError("scalar must be a scalar type (integer/float)");
    Type elem = getElemTy(srcTy);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A2/A3 tmaxs element type to be i32/i16/f16/f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyScalarTileOp(*this, srcTy, dstTy, "src", "dst",
                                  /*requireValidRowsEqual=*/true,
                                  /*requireValidColsEqual=*/true)))
      return failure();
    Type scalarTy = getScalar().getType();
    if (!scalarTy.isa<IntegerType, FloatType>())
      return emitOpError("scalar must be a scalar type (integer/float)");
    Type elem = getElemTy(srcTy);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isF32()))
      return emitOpError("expects A5 tmaxs element type to be i32/u32/i16/u16/i8/u8/f16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TMinOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, src0Ty, "src0")) ||
        failed(verifyTileBufCommon(*this, src1Ty, "src1")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameShapeAndElem(*this, src0Ty, dstTy, "src0", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, dstTy, "src0", "dst")))
      return failure();
    if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
        !isRowMajorTileBuf(dstTy))
      return emitOpError("expects src0, src1, and dst to use row-major layout");
    Type elem = getElemTy(src0Ty);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A2/A3 tmin element type to be i32/i16/f16/f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, src0Ty, "src0")) ||
        failed(verifyTileBufCommon(*this, src1Ty, "src1")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameShapeAndElem(*this, src0Ty, dstTy, "src0", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, dstTy, "src0", "dst")))
      return failure();
    if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
        !isRowMajorTileBuf(dstTy))
      return emitOpError("expects src0, src1, and dst to use row-major layout");
    Type elem = getElemTy(src0Ty);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isF32()))
      return emitOpError("expects A5 tmin element type to be i32/u32/i16/u16/i8/u8/f16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TMinSOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyScalarTileOp(*this, getSrc().getType(), getDst().getType(),
                                  "src", "dst",
                                  /*requireValidRowsEqual=*/true,
                                  /*requireValidColsEqual=*/true)))
      return failure();
    Type elem = getElemTy(getSrc().getType());
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A2/A3 tmins element type to be i32/i16/f16/f32");
    Type scalarTy = getScalar().getType();
    if (!scalarTy.isa<IntegerType, FloatType>())
      return emitOpError("scalar must be a scalar type (integer/float)");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyScalarTileOp(*this, getSrc().getType(), getDst().getType(),
                                  "src", "dst",
                                  /*requireValidRowsEqual=*/false,
                                  /*requireValidColsEqual=*/true)))
      return failure();
    Type elem = getElemTy(getSrc().getType());
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isBF16() || elem.isF32()))
      return emitOpError("expects A5 tmins element type to be i32/u32/i16/u16/i8/u8/f16/bf16/f32");
    Type scalarTy = getScalar().getType();
    if (!scalarTy.isa<IntegerType, FloatType>())
      return emitOpError("scalar must be a scalar type (integer/float)");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TMovOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    auto srcShape = getShapeVec(srcTy);
    auto dstShape = getShapeVec(dstTy);
    if (srcShape != dstShape)
      return emitOpError() << "expects src and dst to have the same shape";
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!srcSpace || !dstSpace)
      return emitOpError() << "expects src and dst to have explicit address spaces";
    bool okPair =
        (*srcSpace == pto::AddressSpace::MAT &&
         (*dstSpace == pto::AddressSpace::LEFT ||
          *dstSpace == pto::AddressSpace::RIGHT ||
          *dstSpace == pto::AddressSpace::BIAS ||
          *dstSpace == pto::AddressSpace::SCALING)) ||
        (*srcSpace == pto::AddressSpace::VEC &&
         *dstSpace == pto::AddressSpace::VEC) ||
        (*srcSpace == pto::AddressSpace::ACC &&
         *dstSpace == pto::AddressSpace::MAT);
    if (!okPair)
      return emitOpError() << "expects an A2/A3-supported tmov address-space pair";
    auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
    auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
    if (srcTb && dstTb && *srcSpace == pto::AddressSpace::ACC &&
        *dstSpace == pto::AddressSpace::MAT &&
        dstTb.getSFractalSizeI32() != 512)
      return emitOpError() << "expects A2/A3 acc-to-mat tmov destination fractal to be 512";
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!srcSpace || !dstSpace)
      return emitOpError() << "expects src and dst to have explicit address spaces";
    auto srcShape = getShapeVec(srcTy);
    auto dstShape = getShapeVec(dstTy);
    if (*srcSpace == pto::AddressSpace::MAT && srcShape != dstShape)
      return emitOpError() << "expects A5 mat-source tmov to use matching src/dst shapes";
    bool okPair =
        (*srcSpace == pto::AddressSpace::MAT &&
         (*dstSpace == pto::AddressSpace::LEFT ||
          *dstSpace == pto::AddressSpace::RIGHT ||
          *dstSpace == pto::AddressSpace::BIAS ||
          *dstSpace == pto::AddressSpace::SCALING)) ||
        (*srcSpace == pto::AddressSpace::VEC &&
         (*dstSpace == pto::AddressSpace::VEC ||
          *dstSpace == pto::AddressSpace::MAT)) ||
        (*srcSpace == pto::AddressSpace::ACC &&
         (*dstSpace == pto::AddressSpace::VEC ||
          *dstSpace == pto::AddressSpace::MAT));
    if (!okPair)
      return emitOpError() << "expects an A5-supported tmov address-space pair";
    return mlir::success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TMovFPOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type fpTy  = getFp().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, fpTy, "fp")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    auto srcElemTy = getElemTy(srcTy);
    auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
    if (!(srcElemTy.isF32() ||
          (srcIntTy && srcIntTy.getWidth() == 32 &&
           (srcIntTy.isSignless() || srcIntTy.isUnsigned()))))
      return emitOpError()
             << "expects src to have element type f32, i32, or u32";
    auto fpSpace = getPTOMemorySpaceEnum(fpTy);
    if (!fpSpace || *fpSpace != mlir::pto::AddressSpace::SCALING)
      return emitOpError() << "expects fp to be in the scaling address space";
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    if (!srcSpace || *srcSpace != mlir::pto::AddressSpace::ACC)
      return emitOpError() << "expects src to be in the acc address space";
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!dstSpace || *dstSpace != mlir::pto::AddressSpace::MAT)
      return emitOpError() << "expects dst to be in the mat address space";
    auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
    auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
    if (srcTb &&
        (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
         srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)))
      return emitOpError()
             << "expects src to use blayout=col_major and slayout=row_major";
    if (dstTb &&
        (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
         dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)))
      return emitOpError()
             << "expects dst to use blayout=col_major and slayout=row_major";
    if (dstTb && dstTb.getSFractalSizeI32() != 512)
      return emitOpError() << "expects dst to use fractal size 512";
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type fpTy  = getFp().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, fpTy, "fp")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    auto srcElemTy = getElemTy(srcTy);
    auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
    if (!(srcElemTy.isF32() ||
          (srcIntTy && srcIntTy.getWidth() == 32 &&
           (srcIntTy.isSignless() || srcIntTy.isUnsigned()))))
      return emitOpError()
             << "expects src to have element type f32, i32, or u32";
    auto fpSpace = getPTOMemorySpaceEnum(fpTy);
    if (!fpSpace || *fpSpace != mlir::pto::AddressSpace::SCALING)
      return emitOpError() << "expects fp to be in the scaling address space";
    auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
    if (srcTb &&
        (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
         srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)))
      return emitOpError()
             << "expects src to use blayout=col_major and slayout=row_major";
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
// 辅助函数：获取 Rank，支持 ShapedType 和 PTO TileTypes
static int64_t getRankHelper(Type t) {
  if (auto s = dyn_cast<ShapedType>(t)) return s.getRank();
  if (auto tile = dyn_cast<pto::TileBufType>(t)) return tile.getRank();
  if (auto view = dyn_cast<pto::PartitionTensorViewType>(t)) return view.getRank();
  return -1;
}

static LogicalResult verifyMatmulLike(Operation *op, Type aTy, Type bTy, Type dstTy, bool checkRank = true) {
  // 1. 检查类型 (ShapedType 或 Tile 类型)
  bool aValid = isa<ShapedType, pto::TileBufType, pto::PartitionTensorViewType>(aTy);
  bool bValid = isa<ShapedType, pto::TileBufType, pto::PartitionTensorViewType>(bTy);
  bool dValid = isa<ShapedType, pto::TileBufType, pto::PartitionTensorViewType>(dstTy);

  if (!aValid || !bValid || !dValid)
    return op->emitOpError("expects inputs/outputs to be shaped types or PTO tile types");

  if (checkRank) {
    int64_t aRank = getRankHelper(aTy);
    int64_t bRank = getRankHelper(bTy);
    int64_t dRank = getRankHelper(dstTy);

    // 检查 Rank 一致性
    if (aRank != -1 && dRank != -1 && aRank != dRank)
      return op->emitOpError("expects a and dst to have the same rank");
    if (bRank != -1 && dRank != -1 && bRank != dRank)
      return op->emitOpError("expects b and dst to have the same rank");
  }

  return success();
}

// ---- LoadScalarOp ----
LogicalResult LoadScalarOp::verify() {
  Type ptrTy = getPtr().getType();
  Type elemTy;
  if (auto pty = dyn_cast<mlir::pto::PtrType>(ptrTy)) {
    elemTy = pty.getElementType();
  } else if (auto memTy = dyn_cast<MemRefType>(ptrTy)) {
    elemTy = memTy.getElementType();
    if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
      return emitOpError() << "scalar load only supports GM address space pointers";
  } else {
    return emitOpError("expects ptr to be !pto.ptr or memref type");
  }

  if (getValue().getType() != elemTy)
    return emitOpError("expects result type to match ptr element type");

  return success();
}
// ---- StoreScalarOp ----
LogicalResult StoreScalarOp::verify() {
  Type ptrTy = getPtr().getType();
  Type elemTy;
  if (auto pty = dyn_cast<mlir::pto::PtrType>(ptrTy)) {
    elemTy = pty.getElementType();
  } else if (auto memTy = dyn_cast<MemRefType>(ptrTy)) {
    elemTy = memTy.getElementType();
    if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
      return emitOpError() << "scalar store only supports GM address space pointers";
  } else {
    return emitOpError("expects ptr to be !pto.ptr or memref type");
  }

  if (getValue().getType() != elemTy)
    return emitOpError("expects value type to match ptr element type");

  return success();
}

// ---- GetBufOp / RlsBufOp ----
static LogicalResult verifyBufSyncOp(Operation *op, Attribute opTypeAttr,
                                     IntegerAttr bufIdAttr, IntegerAttr modeAttr) {
  if (!opTypeAttr)
    return op->emitOpError("expects 'op_type' attribute");

  auto opTypeOr = parseSyncOpTypeLikeAttr(opTypeAttr);
  if (failed(opTypeOr)) {
    auto diag =
        op->emitOpError("expects 'op_type' to be pipe_event_type/sync_op_type, got ");
    diag << opTypeAttr;
    return failure();
  }
  pto::PIPE pipe = mapSyncOpTypeToPipe(*opTypeOr);
  if (!isConcreteSyncPipe(pipe))
    return op->emitOpError("expects 'op_type' to map to a concrete pipe, not PIPE_ALL/PIPE_UNASSIGNED");

  if (!bufIdAttr)
    return op->emitOpError("expects 'buf_id' attribute");
  int64_t bufId = bufIdAttr.getInt();
  if (bufId < 0 || bufId > 31)
    return op->emitOpError("expects 'buf_id' in range [0, 31]");

  if (modeAttr) {
    int64_t mode = modeAttr.getInt();
    if (mode < 0)
      return op->emitOpError("expects 'mode' to be non-negative");
  }

  return success();
}

LogicalResult GetBufOp::verify() {
  return verifyBufSyncOp(getOperation(), getOpTypeAttr(), getBufIdAttr(),
                         getModeAttr());
}

LogicalResult RlsBufOp::verify() {
  return verifyBufSyncOp(getOperation(), getOpTypeAttr(), getBufIdAttr(),
                         getModeAttr());
}
// ---- TOp ----
LogicalResult TGemvBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyGemvTileOperands(*this, getA().getType(), getB().getType(),
                                      getDst().getType())) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType())))
      return failure();
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getA().getType()),
                                      getElemTy(getB().getType()),
                                      getElemTy(getDst().getType()))))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TMatmulBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyMatTileOperands(*this, getA().getType(), getB().getType(),
                                         getDst().getType())) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType())))
      return failure();
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getA().getType()),
                                      getElemTy(getB().getType()),
                                      getElemTy(getDst().getType()))))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TMatmulMxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyTileBufCommon(*this, getAScale().getType(), "a_scale")) ||
        failed(verifyTileBufCommon(*this, getBScale().getType(), "b_scale")))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TMatmulMxAccOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyAccTileCommon(*this, getCIn().getType(), "c_in")) ||
        failed(verifyTileBufCommon(*this, getAScale().getType(), "a_scale")) ||
        failed(verifyTileBufCommon(*this, getBScale().getType(), "b_scale")))
      return failure();
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
LogicalResult TMatmulMxBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyTileBufCommon(*this, getAScale().getType(), "a_scale")) ||
        failed(verifyTileBufCommon(*this, getBScale().getType(), "b_scale")) ||
        failed(verifyMatTileOperands(*this, getA().getType(), getB().getType(),
                                         getDst().getType())) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType(),
                              /*requireFloatBias=*/true)))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
// ---- TSetValOp ----
LogicalResult TSetValOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  // dst can be tile/tensor/tilebuf (PTODpsType). Keep checks minimal.
  if (auto shaped = dyn_cast<ShapedType>(getDst().getType())) {
    if (shaped.getElementType() != getVal().getType())
      return emitOpError("expects val type to match dst element type");
  }
  return success();
}
// ---- TGetValOp ----
LogicalResult TGetValOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  if (!srcTy.isa<pto::TileBufType, MemRefType>())
    return emitOpError("expects src to be tile_buf or memref type");

  // Memory space must be vec (Ascend does not support getval from MAT etc.).
  Attribute memSpace =
      isa<pto::TileBufType>(srcTy)
          ? cast<pto::TileBufType>(srcTy).getMemorySpace()
          : cast<MemRefType>(srcTy).getMemorySpace();
  auto addrSpaceAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memSpace);
  if (!addrSpaceAttr ||
      addrSpaceAttr.getAddressSpace() != pto::AddressSpace::VEC) {
    if (addrSpaceAttr &&
        addrSpaceAttr.getAddressSpace() == pto::AddressSpace::MAT)
      return emitOpError(
          "Ascend hardware does not support reading from Mat tile_buf to Scalar unit");
    return emitOpError("expects src memory space to be vec");
  }

  if (getElemTy(srcTy) != getDst().getType())
    return emitOpError("expects dst type to match src element type");
  return success();
}

// ---- MScatterOp ----
LogicalResult MScatterOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  int64_t srcrank = getPTOTypeRank(getSrc().getType());
  int64_t memrank = getPTOTypeRank(getMem().getType());
  int64_t idxrank = getPTOTypeRank(getIdx().getType());
  
  if (memrank == -1 || idxrank == -1 || srcrank == -1) {
    return emitOpError("src, idx, mem does not support PTO type");
  }
  return success();
}

// ---- MGatherOp ----
LogicalResult MGatherOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  int64_t memrank = getPTOTypeRank(getMem().getType());
  int64_t idxrank = getPTOTypeRank(getIdx().getType());
  int64_t dstrank = getPTOTypeRank(getDst().getType());

  if (memrank == -1 || idxrank == -1 || memrank == -1) {
    return emitOpError("mem, idx and dst does not support PTO type");
  }

  return success();
}

void mlir::pto::TMrgSortOp::print(OpAsmPrinter &p) {
  if (isFormat1()) {
    p << " ins(" << getSrc() << ", " << getBlockLen() << " : " << getSrc().getType()
      << ", " << getBlockLen().getType() << ") outs(" << getDst() << " : "
      << getDst().getType() << ")";
  } else {
    assert(isFormat2());
    p << " ins(" << getSrcs()[0] << ", " << getSrcs()[1] << ", " << getSrcs()[2]
      << ", " << getSrcs()[3] << " {exhausted = " << (getExhausted() ? "true" : "false")
      << "} : " << getSrcs()[0].getType() << ", " << getSrcs()[1].getType() << ", "
      << getSrcs()[2].getType() << ", " << getSrcs()[3].getType() << ") outs("
      << getDst() << ", " << getTmp() << ", " << getExcuted() << " : " << getDst().getType() << ", "
      << getTmp().getType() << ", " << getExcuted().getType() << ")";
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes", "exhausted"});
}

ParseResult mlir::pto::TMrgSortOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseKeyword("ins") || parser.parseLParen())
    return failure();
  OpAsmParser::UnresolvedOperand first, second;
  if (parser.parseOperand(first) || parser.parseComma() || parser.parseOperand(second))
    return failure();

  if (parser.parseOptionalColon().succeeded()) {
    Type srcTy, blockLenTy, dstTy;
    if (parser.parseType(srcTy) || parser.parseComma() || parser.parseType(blockLenTy) ||
        parser.parseRParen() || parser.parseKeyword("outs") || parser.parseLParen())
      return failure();
    OpAsmParser::UnresolvedOperand dstOp;
    if (parser.parseOperand(dstOp) || parser.parseColon() || parser.parseType(dstTy) ||
        parser.parseRParen())
      return failure();
    result.addAttribute("operandSegmentSizes",
                        parser.getBuilder().getDenseI32ArrayAttr({1, 1, 1, 0}));
    if (parser.resolveOperand(first, srcTy, result.operands) ||
        parser.resolveOperand(second, blockLenTy, result.operands) ||
        parser.resolveOperand(dstOp, dstTy, result.operands))
      return failure();
    if (parser.parseOptionalAttrDict(result.attributes))
      return failure();
    if (!result.attributes.get("exhausted"))
      result.addAttribute("exhausted", parser.getBuilder().getBoolAttr(false));
    return success();
  }

  SmallVector<OpAsmParser::UnresolvedOperand, 4> srcs = {first, second};
  OpAsmParser::UnresolvedOperand third, fourth;
  if (parser.parseComma() || parser.parseOperand(third) || parser.parseComma() ||
      parser.parseOperand(fourth))
    return failure();
  srcs.push_back(third);
  srcs.push_back(fourth);
  bool exhaustedVal = false;
  if (parser.parseOptionalLBrace().succeeded()) {
    if (parser.parseKeyword("exhausted") || parser.parseEqual())
      return failure();
    StringRef kw;
    if (parser.parseKeyword(&kw) || parser.parseRBrace())
      return failure();
    exhaustedVal = (kw == "true");
  }
  SmallVector<Type, 4> srcTypes(4);
  if (parser.parseColon() || parser.parseType(srcTypes[0]) || parser.parseComma() ||
      parser.parseType(srcTypes[1]) || parser.parseComma() || parser.parseType(srcTypes[2]) ||
      parser.parseComma() || parser.parseType(srcTypes[3]) || parser.parseRParen() ||
      parser.parseKeyword("outs") || parser.parseLParen())
    return failure();
  OpAsmParser::UnresolvedOperand dstOp, tmpOp, excutedOp;
  Type dstTy, tmpTy, excutedTy;
  if (parser.parseOperand(dstOp) || parser.parseComma() || parser.parseOperand(tmpOp) ||
      parser.parseComma() || parser.parseOperand(excutedOp) || parser.parseColon() ||
      parser.parseType(dstTy) || parser.parseComma() || parser.parseType(tmpTy) ||
      parser.parseComma() || parser.parseType(excutedTy) || parser.parseRParen())
    return failure();
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr({4, 0, 2, 1}));
  if (parser.resolveOperands(srcs, srcTypes, parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperand(dstOp, dstTy, result.operands) ||
      parser.resolveOperand(tmpOp, tmpTy, result.operands) ||
      parser.resolveOperand(excutedOp, excutedTy, result.operands))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (!result.attributes.get("exhausted"))
    result.addAttribute("exhausted", parser.getBuilder().getBoolAttr(exhaustedVal));
  return success();
}

mlir::LogicalResult mlir::pto::TMrgSortOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (isFormat1()) {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (!isPTOShapedLike(srcTy) || !isPTOShapedLike(dstTy))
      return emitOpError() << "format1 expects PTO shaped-like types for src/dst";
    if (getElemTy(srcTy) != getElemTy(dstTy))
      return emitOpError() << "expects src/dst to have the same element type";
    if (!getElemTy(srcTy).isF16() && !getElemTy(srcTy).isF32())
      return emitOpError() << "expects element type to be f16 or f32";
    auto ss = getShapeVec(srcTy);
    auto ds = getShapeVec(dstTy);
    if (ss.size() != 2 || ds.size() != 2)
      return emitOpError() << "expects src/dst to be rank-2 tile-shaped";
    if (ss[0] != mlir::ShapedType::kDynamic && ss[0] != 1)
      return emitOpError() << "expects src rows == 1";
    if (ds[0] != mlir::ShapedType::kDynamic && ds[0] != 1)
      return emitOpError() << "expects dst rows == 1";
    if (ss[1] != mlir::ShapedType::kDynamic && ds[1] != mlir::ShapedType::kDynamic && ss[1] != ds[1])
      return emitOpError() << "expects src/dst cols to match";
    if (getBlockLen()) {
      if (auto cstOp = getBlockLen().getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(cstOp.getValue())) {
          int64_t v = intAttr.getValue().getSExtValue();
          if (v <= 0 || (v % 64) != 0)
            return emitOpError() << "expects blockLen > 0 and multiple of 64";
        }
      }
    }
    return mlir::success();
  }
  if (isFormat2()) {
    for (Value v : getSrcs())
      if (!isPTOShapedLike(v.getType()))
        return emitOpError() << "format2 expects PTO shaped-like type for each src";
    if (getDsts().size() != 2u || !getExcuted())
      return emitOpError() << "format2 expects outs(dst, tmp) and excuted=vector";
    Type dstTy = getDst().getType();
    Type tmpTy = getTmp().getType();
    if (!isPTOShapedLike(dstTy) || !isPTOShapedLike(tmpTy))
      return emitOpError() << "format2 outs must be PTO shaped-like (dst/tmp)";
    auto excutedTy = mlir::dyn_cast<mlir::VectorType>(getExcuted().getType());
    if (!excutedTy || excutedTy.getRank() != 1 || excutedTy.getNumElements() != 4 ||
        !excutedTy.getElementType().isInteger(16))
      return emitOpError() << "format2 excuted must be vector<4xi16>";
    if (getElemTy(dstTy) != getElemTy(tmpTy))
      return emitOpError() << "format2 expects dst/tmp element types to match";
    return mlir::success();
  }
  return emitOpError() << "tmrgsort expects format1 (1 src + blockLen + 1 dst) or format2 (4 srcs, outs dst, excuted)";
}

mlir::LogicalResult mlir::pto::TMulOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, src0Ty, "src0")) ||
        failed(verifyTileBufCommon(*this, src1Ty, "src1")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameShapeAndElem(*this, src0Ty, dstTy, "src0", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, dstTy, "src0", "dst")))
      return failure();
    if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
        !isRowMajorTileBuf(dstTy))
      return emitOpError("expects src0, src1, and dst to use row-major layout");
    Type elem = getElemTy(src0Ty);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A2/A3 tmul element type to be i32/i16/f16/f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, src0Ty, "src0")) ||
        failed(verifyTileBufCommon(*this, src1Ty, "src1")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameShapeAndElem(*this, src0Ty, dstTy, "src0", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, dstTy, "src0", "dst")))
      return failure();
    if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
        !isRowMajorTileBuf(dstTy))
      return emitOpError("expects src0, src1, and dst to use row-major layout");
    Type elem = getElemTy(src0Ty);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A5 tmul element type to be i32/u32/i16/u16/f16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TMulSOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc0().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyScalarTileOp(*this, srcTy, dstTy, "src", "dst",
                                  /*requireValidRowsEqual=*/true,
                                  /*requireValidColsEqual=*/true)))
      return failure();
    Type scalarTy = getScalar().getType();
    if (!scalarTy.isa<IntegerType, FloatType>())
      return emitOpError("scalar must be a scalar type (integer/float)");
    Type elem = getElemTy(srcTy);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A2/A3 tmuls element type to be i32/i16/f16/f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc0().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyScalarTileOp(*this, srcTy, dstTy, "src", "dst",
                                  /*requireValidRowsEqual=*/false,
                                  /*requireValidColsEqual=*/true)))
      return failure();
    Type scalarTy = getScalar().getType();
    if (!scalarTy.isa<IntegerType, FloatType>())
      return emitOpError("scalar must be a scalar type (integer/float)");
    Type elem = getElemTy(srcTy);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isBF16() || elem.isF32()))
      return emitOpError("expects A5 tmuls element type to be i32/u32/i16/u16/i8/u8/f16/bf16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TShlSOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst")))
    return failure();

  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem)
    return emitOpError() << "failed to get element type for src/dst";
  if (srcElem != dstElem)
    return emitOpError() << "expects src and dst to have the same element type";
  if (!mlir::isa<IntegerType>(srcElem))
    return emitOpError() << "expects integral element types";
  if (auto scalarValue = getConstantIntegerValue(getScalar()); scalarValue && *scalarValue < 0)
    return emitOpError("expects tshls scalar to be non-negative");
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TShrSOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();

    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !dstElem) {
      emitOpError("failed to get element type for src/dst");
      return failure();
    }
    if (srcElem != dstElem) {
      emitOpError("expects src and dst to have the same element type");
      return failure();
    }
    return srcElem;
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 16 && it.getWidth() != 32))
      return emitOpError(
          "expects A2/A3 tshrs src and dst element type to be i16/u16/i32/u32");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32))
      return emitOpError(
          "expects A5 tshrs src and dst element type to be i8/u8/i16/u16/i32/u32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TNegOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();

    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(16) || elemTy.isInteger(32) || elemTy.isF16() ||
          elemTy.isF32()))
      return emitOpError()
             << "expects A2/A3 tneg element type to be i16/i32/f16/f32";
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, srcTy, dstTy, "src", "dst")))
      return failure();

    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() != 2 || dstValid.size() != 2)
      return emitOpError() << "expects src and dst to have rank-2 valid_shape";
    if (srcValid[1] != ShapedType::kDynamic &&
        dstValid[1] != ShapedType::kDynamic &&
        srcValid[1] != dstValid[1])
      return emitOpError()
             << "expects src and dst to have the same valid_shape[1]";

    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(8) || elemTy.isInteger(16) || elemTy.isInteger(32) ||
          elemTy.isF16() || elemTy.isF32() || elemTy.isBF16()))
      return emitOpError()
             << "expects A5 tneg element type to be i8/u8/i16/u16/i32/u32/f16/f32/bf16";
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TNotOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    auto elemTy = getElemTy(srcTy);
    if (elemTy != getElemTy(dstTy))
      return emitOpError() << "expects src and dst to have the same element type";
    if (!elemTy.isInteger(16))
      return emitOpError() << "expects A2/A3 tnot element type to be i16/u16";
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    auto elemTy = getElemTy(srcTy);
    if (elemTy != getElemTy(dstTy))
      return emitOpError() << "expects src and dst to have the same element type";
    if (!(elemTy.isInteger(8) || elemTy.isInteger(16) || elemTy.isInteger(32)))
      return emitOpError() << "expects A5 tnot element type to be i8/u8/i16/u16/i32/u32";
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TOrOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, src0Ty, "src0")) ||
        failed(verifyTileBufCommon(*this, src1Ty, "src1")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();

    Type e0 = getElemTy(src0Ty);
    Type e1 = getElemTy(src1Ty);
    Type ed = getElemTy(dstTy);
    if (!e0 || !e1 || !ed) {
      emitOpError("failed to get element type for operands");
      return failure();
    }
    if (e0 != e1 || e0 != ed) {
      emitOpError("expects src0, src1, and dst to have the same element type");
      return failure();
    }
    if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
        !isRowMajorTileBuf(dstTy)) {
      emitOpError("expects src0, src1, and dst to use row-major layout");
      return failure();
    }
    if (failed(verifyTileBufSameValidShape(*this, src0Ty, dstTy, "src0", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, src1Ty, dstTy, "src1", "dst")))
      return failure();
    return e0;
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16))
      return emitOpError(
          "expects A2/A3 tor src0, src1, and dst element type to be i8/u8/i16/u16");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32))
      return emitOpError(
          "expects A5 tor src0, src1, and dst element type to be i8/u8/i16/u16/i32/u32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TOrSOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    if (getSrc() == getDst()) {
      emitOpError("expects src and dst to use different storage");
      return failure();
    }
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();

    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !dstElem) {
      emitOpError("failed to get element type for src/dst");
      return failure();
    }
    if (srcElem != dstElem) {
      emitOpError("expects src and dst to have the same element type");
      return failure();
    }
    if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
      emitOpError("expects src and dst to use row-major layout");
      return failure();
    }
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    return srcElem;
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16))
      return emitOpError(
          "expects A2/A3 tors src and dst element type to be i8/u8/i16/u16");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32))
      return emitOpError(
          "expects A5 tors src and dst element type to be i8/u8/i16/u16/i32/u32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TPartAddOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
        !isPTOShapedLike(dstTy))
      return emitOpError() << "expects PTO shaped-like src0/src1/dst";
    if (getElemTy(src0Ty) != getElemTy(src1Ty) ||
        getElemTy(src0Ty) != getElemTy(dstTy))
      return emitOpError() << "expects src0/src1/dst to have the same element type";
    auto s0 = getShapeVec(src0Ty);
    auto s1 = getShapeVec(src1Ty);
    auto d = getShapeVec(dstTy);
    if (s0.size() != 2 || s1.size() != 2 || d.size() != 2)
      return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped)";
    if (failed(verifyPartialValidPattern(*this, src0Ty, src1Ty, dstTy)))
      return failure();
    Type elem = getElemTy(src0Ty);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A2/A3 tpartadd element type to be i32/i16/f16/f32");
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
        !isPTOShapedLike(dstTy))
      return emitOpError() << "expects PTO shaped-like src0/src1/dst";
    if (getElemTy(src0Ty) != getElemTy(src1Ty) ||
        getElemTy(src0Ty) != getElemTy(dstTy))
      return emitOpError() << "expects src0/src1/dst to have the same element type";
    Type elem = getElemTy(src0Ty);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isBF16() || elem.isF32()))
      return emitOpError("expects A5 tpartadd element type to be i32/u32/i16/u16/i8/u8/f16/bf16/f32");
    auto s0 = getShapeVec(src0Ty);
    auto s1 = getShapeVec(src1Ty);
    auto d = getShapeVec(dstTy);
    if (s0.size() != 2 || s1.size() != 2 || d.size() != 2)
      return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped)";
    return mlir::success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TPartMaxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (!isPTOShapedLike(t0) || !isPTOShapedLike(t1) || !isPTOShapedLike(td))
      return emitOpError("expects src0/src1/dst to be memref/tensor/tile_buf/tile_view types");
    Type e0 = getElemTy(t0), e1 = getElemTy(t1), ed = getElemTy(td);
    if (!e0 || !e1 || !ed)
      return emitOpError("failed to get element type for operands");
    if (e0 != e1 || e0 != ed)
      return emitOpError("expects src0/src1/dst to have the same element type");
    auto s0 = getShapeVec(t0), s1 = getShapeVec(t1), sd = getShapeVec(td);
    if (s0 != s1 || s0 != sd)
      return emitOpError("expects src0/src1/dst to have the same shape");
    if (failed(verifyPartialValidPattern(*this, t0, t1, td)))
      return failure();
    if (!(e0.isInteger(32) || e0.isInteger(16) || e0.isF16() || e0.isF32()))
      return emitOpError("expects A2/A3 tpartmax element type to be i32/i16/f16/f32");
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (!isPTOShapedLike(t0) || !isPTOShapedLike(t1) || !isPTOShapedLike(td))
      return emitOpError("expects src0/src1/dst to be memref/tensor/tile_buf/tile_view types");
    Type e0 = getElemTy(t0), e1 = getElemTy(t1), ed = getElemTy(td);
    if (!e0 || !e1 || !ed)
      return emitOpError("failed to get element type for operands");
    if (e0 != e1 || e0 != ed)
      return emitOpError("expects src0/src1/dst to have the same element type");
    if (!(e0.isInteger(32) || e0.isInteger(16) || e0.isInteger(8) ||
          e0.isF16() || e0.isBF16() || e0.isF32()))
      return emitOpError("expects A5 tpartmax element type to be i32/u32/i16/u16/i8/u8/f16/bf16/f32");
    auto s0 = getShapeVec(t0), s1 = getShapeVec(t1), sd = getShapeVec(td);
    if (s0 != s1 || s0 != sd)
      return emitOpError("expects src0/src1/dst to have the same shape");
    return mlir::success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TPartMinOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (!isPTOShapedLike(t0) || !isPTOShapedLike(t1) || !isPTOShapedLike(td))
      return emitOpError("expects src0/src1/dst to be memref/tensor/tile_buf/tile_view types");
    Type e0 = getElemTy(t0), e1 = getElemTy(t1), ed = getElemTy(td);
    if (!e0 || !e1 || !ed)
      return emitOpError("failed to get element type for operands");
    if (e0 != e1 || e0 != ed)
      return emitOpError("expects src0/src1/dst to have the same element type");
    auto s0 = getShapeVec(t0), s1 = getShapeVec(t1), sd = getShapeVec(td);
    if (s0 != s1 || s0 != sd)
      return emitOpError("expects src0/src1/dst to have the same shape");
    if (failed(verifyPartialValidPattern(*this, t0, t1, td)))
      return failure();
    if (!(e0.isInteger(32) || e0.isInteger(16) || e0.isF16() || e0.isF32()))
      return emitOpError("expects A2/A3 tpartmin element type to be i32/i16/f16/f32");
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (!isPTOShapedLike(t0) || !isPTOShapedLike(t1) || !isPTOShapedLike(td))
      return emitOpError("expects src0/src1/dst to be memref/tensor/tile_buf/tile_view types");
    Type e0 = getElemTy(t0), e1 = getElemTy(t1), ed = getElemTy(td);
    if (!e0 || !e1 || !ed)
      return emitOpError("failed to get element type for operands");
    if (e0 != e1 || e0 != ed)
      return emitOpError("expects src0/src1/dst to have the same element type");
    if (!(e0.isInteger(32) || e0.isInteger(16) || e0.isInteger(8) ||
          e0.isF16() || e0.isBF16() || e0.isF32()))
      return emitOpError("expects A5 tpartmin element type to be i32/u32/i16/u16/i8/u8/f16/bf16/f32");
    auto s0 = getShapeVec(t0), s1 = getShapeVec(t1), sd = getShapeVec(td);
    if (s0 != s1 || s0 != sd)
      return emitOpError("expects src0/src1/dst to have the same shape");
    return mlir::success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TPReluOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type t0 = getSrc0().getType();
  Type t1 = getSrc1().getType();
  Type tt = getTmp().getType();
  Type td = getDst().getType();
  if (!isPTOShapedLike(t0) || !isPTOShapedLike(t1) || !isPTOShapedLike(tt) ||
      !isPTOShapedLike(td))
    return emitOpError(
        "expects src0/src1/tmp/dst to be memref/tensor/tile_buf/tile_view types");
  Type e0 = getElemTy(t0), e1 = getElemTy(t1), et = getElemTy(tt), ed = getElemTy(td);
  if (!e0 || !e1 || !et || !ed)
    return emitOpError("failed to get element type for operands");
  // TPRELU C++ API (TPreluCheck): dst/src0/src1 same type (half or float); tmp must be uint8_t.
  if (e0 != e1 || e0 != ed)
    return emitOpError("expects src0/src1/dst to have the same element type (f16 or f32)");
  if (!e0.isa<FloatType>() || (!e0.isF16() && !e0.isF32()))
    return emitOpError("expects src0/src1/dst element type to be f16 or f32");
  auto intTy = et.dyn_cast<IntegerType>();
  if (!intTy || intTy.getWidth() != 8 || !intTy.isUnsigned())
    return emitOpError("expects tmp to have element type uint8 (unsigned 8-bit integer)");
  auto s0 = getShapeVec(t0), s1 = getShapeVec(t1), st = getShapeVec(tt),
       sd = getShapeVec(td);
  if (s0 != s1 || s0 != st || s0 != sd)
    return emitOpError("expects src0/src1/tmp/dst to have the same shape");
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TRecipOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type ts = getSrc().getType();
  Type td = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, ts, td, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false)))
    return failure();
  if (failed(verifyTileBufSameValidShape(*this, ts, td, "src", "dst")))
    return failure();
  Type elemTy = getElemTy(ts);
  if (!(elemTy.isF16() || elemTy.isF32()))
    return emitOpError() << "expects element type to be f16 or f32";
  if (auto arch = getVerifierArchName(getOperation());
      arch && arch->equals_insensitive("a3") && getSrc() == getDst())
    return emitOpError("expects A3 trecip src and dst to use different storage");
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TReluOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type ts = getSrc().getType();
    Type td = getDst().getType();
    if (failed(verifyVecTileCommon(*this, ts, "src")) ||
        failed(verifyVecTileCommon(*this, td, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, ts, td, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, ts, td, "src", "dst")))
      return failure();
    Type elemTy = getElemTy(ts);
    if (!(elemTy.isInteger(32) || elemTy.isF16() || elemTy.isF32()))
      return emitOpError() << "expects A2/A3 trelu element type to be i32/f16/f32";
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type ts = getSrc().getType();
    Type td = getDst().getType();
    if (failed(verifyVecTileCommon(*this, ts, "src")) ||
        failed(verifyVecTileCommon(*this, td, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, ts, td, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, ts, td, "src", "dst")))
      return failure();
    Type elemTy = getElemTy(ts);
    if (!(elemTy.isInteger(32) || elemTy.isF16() || elemTy.isF32()))
      return emitOpError() << "expects A5 trelu element type to be i32/f16/f32";
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRemOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, src0Ty, "src0")) ||
        failed(verifyTileBufCommon(*this, src1Ty, "src1")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameShapeAndElem(*this, src0Ty, dstTy, "src0", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, dstTy, "src0", "dst")))
      return failure();
    if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
        !isRowMajorTileBuf(dstTy))
      return emitOpError("expects src0, src1, and dst to use row-major layout");
    Type elem = getElemTy(src0Ty);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A2/A3 trem element type to be i32/i16/f16/f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, src0Ty, "src0")) ||
        failed(verifyTileBufCommon(*this, src1Ty, "src1")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameShapeAndElem(*this, src0Ty, dstTy, "src0", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, dstTy, "src0", "dst")))
      return failure();
    if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
        !isRowMajorTileBuf(dstTy))
      return emitOpError("expects src0, src1, and dst to use row-major layout");
    Type elem = getElemTy(src0Ty);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A5 trem element type to be i32/u32/i16/u16/f16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRemSOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type ts = getSrc().getType();
  Type td = getDst().getType();
  if (failed(verifyTileBufCommon(*this, ts, "src")) ||
      failed(verifyTileBufCommon(*this, td, "dst")))
    return failure();
  if (failed(verifyTileBufSameShapeAndElem(*this, ts, td, "src", "dst")))
    return failure();
  if (!mlir::isa<mlir::FloatType>(getScalar().getType()))
    return emitOpError("expects scalar to be a float type");
  return mlir::success();
}


static std::optional<int64_t> getStaticNumElements(ArrayRef<int64_t> shape) {
  int64_t numel = 1;
  for (int64_t d : shape) {
    if (d == ShapedType::kDynamic)
      return std::nullopt;
    if (d < 0)
      return std::nullopt;
    numel *= d;
  }
  return numel;
}

static std::optional<int64_t> getElemBytes(Type elemTy) {
  if (!elemTy)
    return std::nullopt;
  if (auto ft = dyn_cast<FloatType>(elemTy)) {
    if (ft.isF16() || ft.isBF16())
      return 2;
    if (ft.isF32())
      return 4;
    if (ft.isF64())
      return 8;
    return std::nullopt;
  }
  if (auto it = dyn_cast<IntegerType>(elemTy)) {
    int64_t bits = it.getWidth();
    if (bits <= 0)
      return std::nullopt;
    return std::max<int64_t>(1, bits / 8);
  }
  return std::nullopt;
}

static bool isTileBufOrMemref(Type ty) {
  return ty.isa<MemRefType, pto::TileBufType>();
}

mlir::LogicalResult mlir::pto::TReshapeOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type ts = getSrc().getType();
  Type tr = getResult().getType();
  auto srcTb = dyn_cast<pto::TileBufType>(ts);
  auto dstTb = dyn_cast<pto::TileBufType>(tr);
  if (!srcTb || !dstTb)
    return emitOpError("expects src/result to be !pto.tile_buf types");

  if (failed(verifyTileBufCommon(*this, ts, "src")) ||
      failed(verifyTileBufCommon(*this, tr, "dst")))
    return failure();

  if (srcTb.getMemorySpace() != dstTb.getMemorySpace())
    return emitOpError("expects src and dst to use the same loc");

  Type srcElem = srcTb.getElementType();
  Type dstElem = dstTb.getElementType();
  auto srcElemBytes = getElemBytes(srcElem);
  auto dstElemBytes = getElemBytes(dstElem);
  if (!srcElem || !dstElem || !srcElemBytes.has_value() || !dstElemBytes.has_value())
    return emitOpError("failed to get element byte width for src/dst");

  auto srcNumel = getStaticNumElements(getShapeVec(ts));
  auto dstNumel = getStaticNumElements(getShapeVec(tr));
  if (!srcNumel.has_value() || !dstNumel.has_value())
    return emitOpError("expects static shapes for treshape");

  if (srcElemBytes.value() * srcNumel.value() !=
      dstElemBytes.value() * dstNumel.value())
    return emitOpError("expects src and dst to have the same total byte size");

  bool srcBoxed =
      srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox);
  bool dstBoxed =
      dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox);
  if (srcBoxed != dstBoxed)
    return emitOpError("cannot reshape between boxed and non-boxed tile layouts");

  return success();
}

mlir::LogicalResult mlir::pto::BitcastOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  auto srcTy = llvm::dyn_cast<TileBufType>(getSrc().getType());
  auto dstTy = llvm::dyn_cast<TileBufType>(getResult().getType());
  if (!srcTy || !dstTy)
    return emitOpError("expects tile_buf src and tile_buf result");

  if (srcTy.getMemorySpace() != dstTy.getMemorySpace())
    return emitOpError("expects src/result to have the same memorySpace");

  if (srcTy.getElementType() == dstTy.getElementType())
    return emitOpError(
        "expects src/result to have different element types; use "
        "pto.treshape for shape/config changes");

  if (srcTy.getShape() != dstTy.getShape())
    return emitOpError("expects src/result to have the same shape; use pto.treshape for shape changes");

  if (srcTy.getValidShape() != dstTy.getValidShape())
    return emitOpError("expects src/result to have the same validShape");

  auto srcCfg = srcTy.getConfigAttr();
  auto dstCfg = dstTy.getConfigAttr();
  if (srcCfg != dstCfg)
    return emitOpError("expects src/result to have the same tile config");

  auto numel = getStaticNumElements(srcTy.getShape());
  if (!numel.has_value())
    return emitOpError("expects static shapes for bitcast");

  auto srcBytes = getElemBytes(srcTy.getElementType());
  auto dstBytes = getElemBytes(dstTy.getElementType());
  if (!srcBytes.has_value() || !dstBytes.has_value())
    return emitOpError("unsupported element type for bitcast");

  int64_t srcTotalBytes = numel.value() * srcBytes.value();
  int64_t dstTotalBytes = numel.value() * dstBytes.value();
  if (dstTotalBytes > srcTotalBytes)
    return emitOpError("bitcast result requires more bytes than source storage");

  return success();
}


mlir::LogicalResult mlir::pto::TRowExpandOp::verify() {
  auto verifyCommon = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyNDStyleVecTile(*this, dstTy, "dst")))
      return failure();
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    if (!srcSpace || *srcSpace != pto::AddressSpace::VEC)
      return emitOpError("expects src to be in the vec address space");
    if (auto srcTb = dyn_cast<pto::TileBufType>(srcTy)) {
      if (srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
        return emitOpError("expects src to use the none_box slayout");
    }
    if (getElemTy(srcTy) != getElemTy(dstTy))
      return emitOpError("expects src and dst to have the same element type");
    if (!isSupportedVecElemType(getElemTy(srcTy), /*allowBf16=*/true,
                                /*allowInt8=*/true))
      return emitOpError("expects trowexpand element type to be supported");
    auto srcValid = getValidShapeVec(getSrc());
    auto dstValid = getValidShapeVec(getDst());
    if (srcValid.size() != 2 || dstValid.size() != 2)
      return emitOpError("expects src and dst to have rank-2 valid_shape");
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        srcValid[0] != dstValid[0])
      return emitOpError("expects src and dst to have the same valid_shape[0]");
    if (srcValid[0] != ShapedType::kDynamic && srcValid[0] == 0)
      return emitOpError("expects src valid_shape[0] to be non-zero");
    if (srcValid[1] != ShapedType::kDynamic && srcValid[1] == 0)
      return emitOpError("expects src valid_shape[1] to be non-zero");
    if (dstValid[0] != ShapedType::kDynamic && dstValid[0] == 0)
      return emitOpError("expects dst valid_shape[0] to be non-zero");
    if (dstValid[1] != ShapedType::kDynamic && dstValid[1] == 0)
      return emitOpError("expects dst valid_shape[1] to be non-zero");
    return success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyCommon();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyCommon();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowExpandDivOp::verify() {
  auto verifyCommon = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, src0Ty, "src0")) ||
        failed(verifyTileBufCommon(*this, src1Ty, "src1")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, src0Ty, dstTy, "src0", "dst")))
      return failure();
    if (getElemTy(src0Ty) != getElemTy(src1Ty))
      return emitOpError("expects src0 and src1 to have the same element type");
    if (!isRowMajorTileBuf(dstTy))
      return emitOpError("expects dst to use row-major layout");
    auto elemTy = getElemTy(src0Ty).dyn_cast<mlir::FloatType>();
    if (!elemTy || (!elemTy.isF16() && !elemTy.isF32()))
      return emitOpError("expects element type to be f16 or f32");
    return mlir::success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyCommon();
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyCommon(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowExpandMulOp::verify() {
  auto verifyCommon = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, src0Ty, "src0")) ||
        failed(verifyTileBufCommon(*this, src1Ty, "src1")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, src0Ty, dstTy, "src0", "dst")))
      return failure();
    if (getElemTy(src0Ty) != getElemTy(src1Ty))
      return emitOpError("expects src0 and src1 to have the same element type");
    if (!isRowMajorTileBuf(dstTy))
      return emitOpError("expects dst to use row-major layout");
    auto ft = getElemTy(src0Ty).dyn_cast<mlir::FloatType>();
    if (!ft || (!ft.isF16() && !ft.isF32()))
      return emitOpError("expects element type to be f16 or f32");
    return mlir::success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyCommon();
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyCommon(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowExpandSubOp::verify() {
  auto verifyCommon = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, src0Ty, "src0")) ||
        failed(verifyTileBufCommon(*this, src1Ty, "src1")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, src0Ty, dstTy, "src0", "dst")))
      return failure();
    if (getElemTy(src0Ty) != getElemTy(src1Ty))
      return emitOpError("expects src0 and src1 to have the same element type");
    if (!isRowMajorTileBuf(dstTy))
      return emitOpError("expects dst to use row-major layout");
    auto ft = getElemTy(src0Ty).dyn_cast<mlir::FloatType>();
    if (!ft || (!ft.isF16() && !ft.isF32()))
      return emitOpError("expects element type to be f16 or f32");
    return mlir::success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyCommon();
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyCommon(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowMaxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type ts = getSrc().getType();
    Type td = getDst().getType();
    if (failed(verifyRowReductionSrcLayout(*this, ts, "src")) ||
        failed(verifyRowReductionDstLayout(*this, td, "dst")))
      return failure();
    if (getElemTy(ts) != getElemTy(td))
      return emitOpError("expects src and dst to have the same element type");
    if (failed(verifyRowReductionValidRegion(*this, ts, td)))
      return failure();
    auto ft = getElemTy(ts).dyn_cast<mlir::FloatType>();
    if (!ft || (!ft.isF16() && !ft.isF32()))
      return emitOpError("expects element type to be f16 or f32");
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type ts = getSrc().getType();
    Type td = getDst().getType();
    if (failed(verifyRowReductionSrcLayout(*this, ts, "src")) ||
        failed(verifyRowReductionDstLayout(*this, td, "dst")))
      return failure();
    if (getElemTy(ts) != getElemTy(td))
      return emitOpError("expects src and dst to have the same element type");
    if (failed(verifyRowReductionValidRegion(*this, ts, td)))
      return failure();
    auto ft = getElemTy(ts).dyn_cast<mlir::FloatType>();
    if (!ft || (!ft.isF16() && !ft.isF32()))
      return emitOpError("expects element type to be f16 or f32");
    return mlir::success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowMinOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type ts = getSrc().getType();
    Type tt = getTmp().getType();
    Type td = getDst().getType();
    if (failed(verifyRowReductionSrcLayout(*this, ts, "src")) ||
        failed(verifyVecTileCommon(*this, tt, "tmp")) ||
        failed(verifyRowReductionDstLayout(*this, td, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, ts, tt, "src", "tmp")) ||
        failed(verifyTileBufSameValidShape(*this, ts, tt, "src", "tmp")))
      return failure();
    if (getElemTy(ts) != getElemTy(td))
      return emitOpError("expects src and dst to have the same element type");
    if (failed(verifyRowReductionValidRegion(*this, ts, td)))
      return failure();
    auto ft = getElemTy(ts).dyn_cast<mlir::FloatType>();
    if (!ft || (!ft.isF16() && !ft.isF32()))
      return emitOpError("expects element type to be f16 or f32");
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type ts = getSrc().getType();
    Type tt = getTmp().getType();
    Type td = getDst().getType();
    if (failed(verifyRowReductionSrcLayout(*this, ts, "src")) ||
        failed(verifyVecTileCommon(*this, tt, "tmp")) ||
        failed(verifyRowReductionDstLayout(*this, td, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, ts, tt, "src", "tmp")) ||
        failed(verifyTileBufSameValidShape(*this, ts, tt, "src", "tmp")))
      return failure();
    if (getElemTy(ts) != getElemTy(td))
      return emitOpError("expects src and dst to have the same element type");
    if (failed(verifyRowReductionValidRegion(*this, ts, td)))
      return failure();
    auto ft = getElemTy(ts).dyn_cast<mlir::FloatType>();
    if (!ft || (!ft.isF16() && !ft.isF32()))
      return emitOpError("expects element type to be f16 or f32");
    return mlir::success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowSumOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type ts = getSrc().getType();
    Type td = getDst().getType();
    if (failed(verifyRowReductionSrcLayout(*this, ts, "src")) ||
        failed(verifyRowReductionDstLayout(*this, td, "dst")))
      return failure();
    if (getElemTy(ts) != getElemTy(td))
      return emitOpError("expects src and dst to have the same element type");
    if (failed(verifyRowReductionValidRegion(*this, ts, td)))
      return failure();
    auto ft = getElemTy(ts).dyn_cast<mlir::FloatType>();
    if (!ft || (!ft.isF16() && !ft.isF32()))
      return emitOpError("expects element type to be f16 or f32");
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type ts = getSrc().getType();
    Type td = getDst().getType();
    if (failed(verifyRowReductionSrcLayout(*this, ts, "src")) ||
        failed(verifyRowReductionDstLayout(*this, td, "dst")))
      return failure();
    if (getElemTy(ts) != getElemTy(td))
      return emitOpError("expects src and dst to have the same element type");
    if (failed(verifyRowReductionValidRegion(*this, ts, td)))
      return failure();
    auto ft = getElemTy(ts).dyn_cast<mlir::FloatType>();
    if (!ft || (!ft.isF16() && !ft.isF32()))
      return emitOpError("expects element type to be f16 or f32");
    return mlir::success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRsqrtOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type ts = getSrc().getType();
  Type td = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, ts, td, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false)))
    return failure();
  if (failed(verifyTileBufSameValidShape(*this, ts, td, "src", "dst")))
    return failure();
  auto ft = getElemTy(ts).dyn_cast<mlir::FloatType>();
  if (!ft || (!ft.isF16() && !ft.isF32()))
    return emitOpError("expects element type to be f16 or f32");
  return mlir::success();
}


mlir::LogicalResult mlir::pto::TScatterOp::verify() {
  auto verifyCommon = [&]() -> LogicalResult {
    Type ts = getSrc().getType();
    Type ti = getIndexes().getType();
    Type td = getDst().getType();
    if (failed(verifyVecTileStorage(*this, ts, "src")) ||
        failed(verifyVecTileStorage(*this, ti, "indexes")) ||
        failed(verifyVecTileStorage(*this, td, "dst")))
      return failure();

    Type srcElem = getElemTy(ts), dstElem = getElemTy(td), idxElem = getElemTy(ti);
    if (!srcElem || !dstElem || !idxElem)
      return emitOpError("failed to get element type for operands");
    if (srcElem != dstElem)
      return emitOpError("expects src/dst to have the same element type");

    auto isAllowedDataElem = [&](mlir::Type t) -> bool {
      if (t.isF16() || t.isF32() || t.isBF16()) return true;
      if (auto it = mlir::dyn_cast<mlir::IntegerType>(t))
        return (it.getWidth() == 8 || it.getWidth() == 16 || it.getWidth() == 32);
      return false;
    };
    auto isAllowedIndexElem = [&](mlir::Type t) -> bool {
      if (auto it = mlir::dyn_cast<mlir::IntegerType>(t))
        return (it.getWidth() == 16 || it.getWidth() == 32);
      return false;
    };
    if (!isAllowedDataElem(srcElem))
      return emitOpError("expects src/dst element type to be i8/u8/i16/u16/i32/u32/f16/bf16/f32");
    if (!isAllowedIndexElem(idxElem))
      return emitOpError("expects indexes element type to be i16/u16/i32/u32");

    auto bwData = srcElem.getIntOrFloatBitWidth();
    auto bwIdx  = idxElem.getIntOrFloatBitWidth();
    if (bwData != 8 && bwData != 16 && bwData != 32)
      return emitOpError("unexpected src/dst element bitwidth");

    unsigned dataBytes = bwData / 8;
    unsigned idxBytes  = bwIdx / 8;
    unsigned expectedIdxBytes = (dataBytes == 1) ? 2 : dataBytes;
    if (idxBytes != expectedIdxBytes)
      return emitOpError("expects indexes element size to match the documented scatter rule");
    return mlir::success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyCommon(); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyCommon(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TSelOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (failed(verifyTileBufCommon(*this, t0, "src0")) ||
        failed(verifyTileBufCommon(*this, t1, "src1")) ||
        failed(verifyTileBufCommon(*this, td, "dst")))
      return failure();

    Type srcElem = getElemTy(t0);
    Type src1Elem = getElemTy(t1);
    Type dstElem = getElemTy(td);
    if (!srcElem || !src1Elem || !dstElem) {
      emitOpError("failed to get element type for operands");
      return failure();
    }
    if (srcElem != src1Elem || srcElem != dstElem) {
      emitOpError("expects src0, src1, and dst to have the same element type");
      return failure();
    }

    if (!isRowMajorTileBuf(t0) || !isRowMajorTileBuf(t1) ||
        !isRowMajorTileBuf(td)) {
      emitOpError(
          "expects src0, src1, and dst to use row-major layout");
      return failure();
    }
    return srcElem;
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> srcElem = verifyCommon();
    if (failed(srcElem))
      return failure();
    Type elem = *srcElem;
    bool ok = elem.isF16() || elem.isF32();
    if (auto it = dyn_cast<IntegerType>(elem))
      ok = it.getWidth() == 16 || it.getWidth() == 32;
    if (!ok)
      return emitOpError(
          "expects A2/A3 tsel src0, src1, and dst element type to be i16/u16/i32/u32/f16/f32");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> srcElem = verifyCommon();
    if (failed(srcElem))
      return failure();
    Type elem = *srcElem;
    bool ok = elem.isF16() || elem.isF32();
    if (auto it = dyn_cast<IntegerType>(elem))
      ok = it.getWidth() == 8 || it.getWidth() == 16 || it.getWidth() == 32;
    if (!ok)
      return emitOpError(
          "expects A5 tsel src0, src1, and dst element type to be i8/u8/i16/u16/i32/u32/f16/f32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TSelSOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (failed(verifyTileBufCommon(*this, t0, "src0")) ||
        failed(verifyTileBufCommon(*this, t1, "src1")) ||
        failed(verifyTileBufCommon(*this, td, "dst")))
      return failure();
    Type e0 = getElemTy(t0), e1 = getElemTy(t1), ed = getElemTy(td);
    if (!e0 || !e1 || !ed) {
      emitOpError("failed to get element type for operands");
      return failure();
    }
    if (e0 != e1 || e0 != ed) {
      emitOpError("expects src0, src1, and dst to have the same element type");
      return failure();
    }
    if (failed(verifyTileBufSameValidShape(*this, t0, td, "src0", "dst")))
      return failure();
    return e0;
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    Type t0 = getSrc0().getType();
    Type td = getDst().getType();
    if (!isRowMajorTileBuf(t0) || !isRowMajorTileBuf(td))
      return emitOpError("expects src0 and dst to use row-major layout");
    Type elem = *elemOr;
    bool ok = elem.isF16() || elem.isF32();
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem))
      ok = it.getWidth() == 16 || it.getWidth() == 32;
    if (!ok)
      return emitOpError(
          "expects A2/A3 tsels src0, src1, and dst element type to be i16/u16/i32/u32/f16/f32");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (!isRowMajorTileBuf(t0) || !isRowMajorTileBuf(t1) ||
        !isRowMajorTileBuf(td))
      return emitOpError("expects src0, src1, and dst to use row-major layout");
    Type elem = *elemOr;
    bool ok = elem.isF16() || elem.isF32();
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem))
      ok = it.getWidth() == 8 || it.getWidth() == 16 || it.getWidth() == 32;
    if (!ok)
      return emitOpError(
          "expects A5 tsels src0, src1, and dst element type to be i8/u8/i16/u16/i32/u32/f16/f32");
    return success();
  };

  if (!isa<IntegerType>(getSelectMode().getType()))
    return emitOpError("expects selectMode to be an integer scalar");
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TShlOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (failed(verifyTileBufCommon(*this, t0, "src0")) ||
        failed(verifyTileBufCommon(*this, t1, "src1")) ||
        failed(verifyTileBufCommon(*this, td, "dst")))
      return failure();
    Type e0 = getElemTy(t0);
    Type e1 = getElemTy(t1);
    if (!e0 || !e1) {
      emitOpError("failed to get element type for operands");
      return failure();
    }
    if (e0 != e1) {
      emitOpError("expects src0 and src1 to have the same element type");
      return failure();
    }
    if (!isRowMajorTileBuf(t0) || !isRowMajorTileBuf(t1) ||
        !isRowMajorTileBuf(td)) {
      emitOpError("expects src0, src1, and dst to use row-major layout");
      return failure();
    }
    if (failed(verifyTileBufSameValidShape(*this, t0, td, "src0", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, t1, td, "src1", "dst")))
      return failure();
    return e0;
  };

  auto verifyByArch = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32))
      return emitOpError(
          "expects tshl src0 and src1 element type to be i8/u8/i16/u16/i32/u32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}


mlir::LogicalResult mlir::pto::TShrOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, src0Ty, "src0")) ||
        failed(verifyTileBufCommon(*this, src1Ty, "src1")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    Type e0 = getElemTy(src0Ty);
    Type e1 = getElemTy(src1Ty);
    if (!e0 || !e1) {
      emitOpError("failed to get element type for operands");
      return failure();
    }
    if (e0 != e1) {
      emitOpError("expects src0 and src1 to have the same element type");
      return failure();
    }
    if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
        !isRowMajorTileBuf(dstTy)) {
      emitOpError("expects src0, src1, and dst to use row-major layout");
      return failure();
    }
    if (failed(verifyTileBufSameValidShape(*this, src0Ty, dstTy, "src0", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, src1Ty, dstTy, "src1", "dst")))
      return failure();
    return e0;
  };

  auto verifyByArch = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32))
      return emitOpError(
          "expects tshr src0 and src1 element type to be i8/u8/i16/u16/i32/u32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}


mlir::LogicalResult mlir::pto::TSort32Op::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  Type idxTy = getIdx().getType();
  if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
      failed(verifyVecTileCommon(*this, dstTy, "dst")) ||
      failed(verifyVecTileCommon(*this, idxTy, "idx")))
    return failure();

  auto srcElem = getElemTy(srcTy);
  auto dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem || srcElem != dstElem)
    return emitOpError() << "expects src and dst to have the same element type";
  if (!(srcElem.isF16() || srcElem.isF32()))
    return emitOpError() << "expects src and dst element type to be f16 or f32";

  auto idxElem = getElemTy(idxTy);
  auto idxInt = dyn_cast<IntegerType>(idxElem);
  if (!idxInt || idxInt.getWidth() != 32 || !idxInt.isUnsigned())
    return emitOpError() << "expects idx element type to be u32";
  return mlir::success();
}


mlir::LogicalResult mlir::pto::TSqrtOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false)))
    return failure();
  if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
    return failure();

  auto srcElem = getElemTy(srcTy);
  if (!(mlir::isa<mlir::FloatType>(srcElem) || mlir::isa<mlir::Float16Type>(srcElem)))
    return emitOpError() << "expects src and dst element type to be float or half";

  return mlir::success();
}



mlir::LogicalResult mlir::pto::TStoreFPOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type fpTy = getFp().getType();
    Type dstTy = getDst().getType();
    if (!isa<pto::TileBufType>(srcTy))
      return emitOpError() << "expects src to be a !pto.tile_buf";
    if (!isa<pto::TileBufType>(fpTy))
      return emitOpError() << "expects fp to be a !pto.tile_buf";
    if (!isa<MemRefType>(dstTy))
      return emitOpError() << "expects dst to be a memref";
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, fpTy, "fp")))
      return failure();
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    if (!srcSpace || *srcSpace != pto::AddressSpace::ACC)
      return emitOpError() << "expects src to be in the acc address space";
    auto srcElemTy = getElemTy(srcTy);
    auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
    if (!(srcElemTy.isF32() ||
          (srcIntTy && srcIntTy.getWidth() == 32 &&
           (srcIntTy.isSignless() || srcIntTy.isUnsigned()))))
      return emitOpError()
             << "expects src to have element type f32, i32, or u32";
    auto srcShape = getShapeVec(srcTy);
    if (srcShape.size() != 2)
      return emitOpError() << "expects src to have rank 2";
    if (srcShape[1] != ShapedType::kDynamic &&
        (srcShape[1] < 1 || srcShape[1] > 4095))
      return emitOpError() << "expects src.cols to be in the range [1, 4095]";
    auto srcValid = getValidShapeVec(srcTy);
    if (srcValid.size() != 2)
      return emitOpError() << "expects src to have a rank-2 valid_shape";
    if (srcValid[1] != ShapedType::kDynamic &&
        (srcValid[1] < 1 || srcValid[1] > 4095))
      return emitOpError()
             << "expects src.valid_shape[1] to be in the range [1, 4095]";
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type fpTy = getFp().getType();
    Type dstTy = getDst().getType();
    if (!isa<pto::TileBufType>(srcTy))
      return emitOpError() << "expects src to be a !pto.tile_buf";
    if (!isa<pto::TileBufType>(fpTy))
      return emitOpError() << "expects fp to be a !pto.tile_buf";
    if (!isa<MemRefType>(dstTy))
      return emitOpError() << "expects dst to be a memref";
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, fpTy, "fp")))
      return failure();
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    if (!srcSpace || *srcSpace != pto::AddressSpace::ACC)
      return emitOpError() << "expects src to be in the acc address space";
    return mlir::success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TSubOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (failed(verifyTileBufCommon(*this, t0, "src0")) ||
        failed(verifyTileBufCommon(*this, t1, "src1")) ||
        failed(verifyTileBufCommon(*this, td, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, t0, t1, "src0", "src1")) ||
        failed(verifyTileBufSameShapeAndElem(*this, t0, td, "src0", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, t0, t1, "src0", "src1")) ||
        failed(verifyTileBufSameValidShape(*this, t0, td, "src0", "dst")))
      return failure();
    if (!isRowMajorTileBuf(t0) || !isRowMajorTileBuf(t1) || !isRowMajorTileBuf(td))
      return emitOpError("expects src0, src1, and dst to use row-major layout");
    Type elem = getElemTy(t0);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A2/A3 tsub element type to be i32/i16/f16/f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (failed(verifyTileBufCommon(*this, t0, "src0")) ||
        failed(verifyTileBufCommon(*this, t1, "src1")) ||
        failed(verifyTileBufCommon(*this, td, "dst")))
      return failure();
    if (failed(verifyTileBufSameShapeAndElem(*this, t0, t1, "src0", "src1")) ||
        failed(verifyTileBufSameShapeAndElem(*this, t0, td, "src0", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, t0, t1, "src0", "src1")) ||
        failed(verifyTileBufSameValidShape(*this, t0, td, "src0", "dst")))
      return failure();
    if (!isRowMajorTileBuf(t0) || !isRowMajorTileBuf(t1) || !isRowMajorTileBuf(td))
      return emitOpError("expects src0, src1, and dst to use row-major layout");
    Type elem = getElemTy(t0);
    if (elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
        elem.isF16() || elem.isF32())
      return success();
    return emitOpError("expects A5 tsub element type to be i32/u32/i16/u16/i8/u8/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TSubCOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type src2Ty = getSrc2().getType();
  Type dstTy = getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) || !isPTOShapedLike(src2Ty) || !isPTOShapedLike(dstTy))
    return emitOpError() << "expects PTO shaped-like src0, src1, src2, and dst";

  auto d = getShapeVec(dstTy);
  if (getShapeVec(src0Ty).size() != d.size() || getShapeVec(src1Ty).size() != d.size() || getShapeVec(src2Ty).size() != d.size())
    return emitOpError() << "expects all tensors to have the same rank";
  return mlir::success();
}


mlir::LogicalResult mlir::pto::TSubSOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyScalarTileOp(*this, srcTy, dstTy, "src", "dst",
                                  /*requireValidRowsEqual=*/true,
                                  /*requireValidColsEqual=*/true)))
      return failure();
    Type scalarTy = getScalar().getType();
    if (!scalarTy.isa<IntegerType, FloatType>())
      return emitOpError("scalar must be a scalar type (integer/float)");
    Type elem = getElemTy(srcTy);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A2/A3 tsubs element type to be i32/i16/f16/f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyScalarTileOp(*this, srcTy, dstTy, "src", "dst",
                                  /*requireValidRowsEqual=*/false,
                                  /*requireValidColsEqual=*/true)))
      return failure();
    Type scalarTy = getScalar().getType();
    if (!scalarTy.isa<IntegerType, FloatType>())
      return emitOpError("scalar must be a scalar type (integer/float)");
    Type elem = getElemTy(srcTy);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isBF16() || elem.isF32()))
      return emitOpError("expects A5 tsubs element type to be i32/u32/i16/u16/i8/u8/f16/bf16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TSubSCOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type dstTy = getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) || !isPTOShapedLike(dstTy))
    return emitOpError() << "expects PTO shaped-like src0, src1, and dst";

  auto d = getShapeVec(dstTy);
  if (getShapeVec(src0Ty).size() != d.size() || getShapeVec(src1Ty).size() != d.size())
    return emitOpError() << "expects src0, src1, and dst to have the same rank";
  return mlir::success();
}
mlir::LogicalResult mlir::pto::TTransOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type tmpTy = getTmp().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, tmpTy, "tmp")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    Type srcElem = getElemTy(srcTy);
    Type tmpElem = getElemTy(tmpTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !tmpElem || !dstElem || srcElem != dstElem || srcElem != tmpElem)
      return emitOpError() << "expects src and dst to have the same element type";
    if (auto srcTb = dyn_cast<pto::TileBufType>(srcTy)) {
      if (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor))
        return emitOpError() << "expects A2/A3 transpose src to use the row_major blayout";
    }
    unsigned elemBytes = srcElem.getIntOrFloatBitWidth() / 8;
    if (elemBytes != 1 && elemBytes != 2 && elemBytes != 4)
      return emitOpError() << "expects transpose element size to be 1, 2, or 4 bytes";
    auto isAllowedWidthType = [&](Type ty) {
      if (elemBytes == 4)
        return ty.isInteger(32) || ty.isF32();
      if (elemBytes == 2)
        return ty.isInteger(16) || ty.isF16() || ty.isBF16();
      return ty.isInteger(8);
    };
    if (!isAllowedWidthType(srcElem))
      return emitOpError() << "expects transpose element type to match the supported set for its width";
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type tmpTy = getTmp().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, tmpTy, "tmp")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    Type srcElem = getElemTy(srcTy);
    Type tmpElem = getElemTy(tmpTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !tmpElem || !dstElem || srcElem != dstElem || srcElem != tmpElem)
      return emitOpError() << "expects src, tmp, and dst to have the same element type";
    unsigned elemBytes = srcElem.getIntOrFloatBitWidth() / 8;
    if (elemBytes != 1 && elemBytes != 2 && elemBytes != 4)
      return emitOpError() << "expects transpose element size to be 1, 2, or 4 bytes";
    auto isAllowedWidthType = [&](Type ty) {
      if (elemBytes == 4)
        return ty.isInteger(32) || ty.isF32();
      if (elemBytes == 2)
        return ty.isInteger(16) || ty.isF16() || ty.isBF16();
      return ty.isInteger(8);
    };
    if (!isAllowedWidthType(srcElem))
      return emitOpError() << "expects transpose element type to match the supported set for its width";
    auto checkAlignedMajor = [&](Type ty, StringRef name) -> LogicalResult {
      auto tb = mlir::dyn_cast<pto::TileBufType>(ty);
      if (!tb)
        return success();
      auto shape = getShapeVec(ty);
      if (shape.size() != 2)
        return success();
      bool rowMajor = tb.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
      int64_t major = rowMajor ? shape[1] : shape[0];
      if (major != ShapedType::kDynamic && (major * static_cast<int64_t>(elemBytes)) % 32 != 0)
        return emitOpError() << "expects " << name << " major dimension times element size to be 32-byte aligned on A5";
      return success();
    };
    if (failed(checkAlignedMajor(srcTy, "src")) || failed(checkAlignedMajor(dstTy, "dst")))
      return failure();
    return mlir::success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TXorOp::verify() {
  auto verifyBase = [&]() -> FailureOr<Type> {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, src0Ty, "src0")) ||
        failed(verifyTileBufCommon(*this, src1Ty, "src1")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    Type e0 = getElemTy(src0Ty);
    Type e1 = getElemTy(src1Ty);
    Type ed = getElemTy(dstTy);
    if (!e0 || !e1 || !ed) {
      emitOpError("failed to get element type for operands");
      return failure();
    }
    if (e0 != e1 || e0 != ed) {
      emitOpError("expects src0, src1, and dst to have the same element type");
      return failure();
    }
    if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
        !isRowMajorTileBuf(dstTy)) {
      emitOpError("expects src0, src1, and dst to use row-major layout");
      return failure();
    }
    if (failed(verifyTileBufSameValidShape(*this, src0Ty, dstTy, "src0", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, src1Ty, dstTy, "src1", "dst")))
      return failure();
    return e0;
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyBase();
    if (failed(elemOr))
      return failure();
    Type tmpTy = getTmp().getType();
    if (failed(verifyTileBufCommon(*this, tmpTy, "tmp")))
      return failure();
    Type elem = *elemOr;
    if (getElemTy(tmpTy) != elem)
      return emitOpError("expects tmp to have the same element type as src0, src1, and dst");
    if (!isRowMajorTileBuf(tmpTy))
      return emitOpError("expects tmp to use row-major layout");
    if (failed(verifyTileBufSameValidShape(*this, tmpTy, getDst().getType(), "tmp", "dst")))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(elem);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16))
      return emitOpError(
          "expects A2/A3 txor src0, src1, tmp, and dst element type to be i8/u8/i16/u16");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyBase();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32))
      return emitOpError(
          "expects A5 txor src0, src1, and dst element type to be i8/u8/i16/u16/i32/u32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TXorSOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    if (getSrc() == getDst()) {
      emitOpError("expects src and dst to use different storage");
      return failure();
    }
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();

    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !dstElem) {
      emitOpError("failed to get element type for src/dst");
      return failure();
    }
    if (srcElem != dstElem) {
      emitOpError("expects src and dst to have the same element type");
      return failure();
    }
    if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
      emitOpError("expects src and dst to use row-major layout");
      return failure();
    }
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    return srcElem;
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16))
      return emitOpError(
          "expects A2/A3 txors src and dst element type to be i8/u8/i16/u16");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32))
      return emitOpError(
          "expects A5 txors src and dst element type to be i8/u8/i16/u16/i32/u32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
mlir::LogicalResult mlir::pto::TPrintOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  auto srcType = getSrc().getType();
  if (auto tb = mlir::dyn_cast<mlir::pto::TileBufType>(srcType)) {
    auto elem = tb.getElementType();
    if (!(elem.isF16() || elem.isF32() ||
          elem.isInteger(8) || elem.isInteger(16) || elem.isInteger(32)))
      return emitOpError() << "expects printable tile element type";
    auto space = getPTOMemorySpaceEnum(srcType);
    if (!space || *space != pto::AddressSpace::VEC)
      return emitOpError() << "expects printable tile_buf to be in vec address space";
    return success();
  }
  if (mlir::dyn_cast<MemRefType>(srcType) ||
      mlir::dyn_cast<mlir::pto::PartitionTensorViewType>(srcType))
    return mlir::success();
  return emitOpError() << "expects tile_buf, memref, or partition_tensor_view for src";
}



static LogicalResult verifyMatmulCommon(Operation *op, Value lhs, Value rhs,
                                       Value biasOpt, Type maybeDstElemTy,
                                       Type maybeResultElemTy) {
  // ---- case A: tensor/memref (ShapedType) ----
  if (auto lhsTy = dyn_cast<ShapedType>(lhs.getType())) {
    auto rhsTy = dyn_cast<ShapedType>(rhs.getType());
    if (!rhsTy || !lhsTy.hasRank() || !rhsTy.hasRank())
      return op->emitOpError("expects lhs and rhs to be ranked tensors or memrefs");

    if (lhsTy.getElementType() != rhsTy.getElementType())
      return op->emitOpError()
             << "expects lhs and rhs to have the same element type, but got lhs="
             << lhsTy.getElementType() << " rhs=" << rhsTy.getElementType();

    if (biasOpt) {
      auto biasTy = dyn_cast<ShapedType>(biasOpt.getType());
      if (!biasTy || !biasTy.hasRank())
        return op->emitOpError("expects bias to be a ranked tensor or memref");
      if (biasTy.getElementType() != lhsTy.getElementType())
        return op->emitOpError()
               << "expects bias to have the same element type as lhs and rhs, but got bias="
               << biasTy.getElementType() << " vs " << lhsTy.getElementType();
    }

    if (maybeDstElemTy && maybeDstElemTy != lhsTy.getElementType())
      return op->emitOpError()
             << "expects dst to have the same element type as lhs and rhs, but got dst="
             << maybeDstElemTy << " vs " << lhsTy.getElementType();

    if (maybeResultElemTy && maybeResultElemTy != lhsTy.getElementType())
      return op->emitOpError()
             << "expects result to have the same element type as lhs and rhs, but got result="
             << maybeResultElemTy << " vs " << lhsTy.getElementType();

    return success();
  }

  // ---- case B: tile ----
  auto lhsTile = dyn_cast<mlir::pto::TileType>(lhs.getType());
  auto rhsTile = dyn_cast<mlir::pto::TileType>(rhs.getType());
  if (!lhsTile || !rhsTile)
    return op->emitOpError("expects lhs and rhs to be ranked tensors, memrefs, or !pto.tile");

  if (lhsTile.getElementType() != rhsTile.getElementType())
    return op->emitOpError() << "expects lhs and rhs tiles to have the same element type, but got lhs="
                             << lhsTile.getElementType() << " rhs=" << rhsTile.getElementType();

  if ((int64_t)lhsTile.getShape().size() != 2 || (int64_t)rhsTile.getShape().size() != 2)
    return op->emitOpError("expects lhs and rhs tiles to be 2D");

  if (lhsTile.getShape()[1] != rhsTile.getShape()[0])
    return op->emitOpError() << "expects lhs dim1 to equal rhs dim0, but got "
                             << lhsTile.getShape()[1] << " vs " << rhsTile.getShape()[0];

  if (biasOpt) {
    auto biasTile = dyn_cast<mlir::pto::TileType>(biasOpt.getType());
    if (!biasTile)
      return op->emitOpError("expects bias to be !pto.tile when lhs and rhs are !pto.tile");
    if (biasTile.getElementType() != lhsTile.getElementType())
      return op->emitOpError("expects bias to have the same element type as lhs and rhs");
  }

  if (maybeDstElemTy && maybeDstElemTy != lhsTile.getElementType())
    return op->emitOpError() << "expects dst to have the same element type as lhs and rhs";

  if (maybeResultElemTy && maybeResultElemTy != lhsTile.getElementType())
    return op->emitOpError() << "expects result to have the same element type as lhs and rhs";

  return success();
}

LogicalResult mlir::pto::TMatmulOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyMatTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                     getDst().getType())))
      return failure();
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getLhs().getType()),
                                      getElemTy(getRhs().getType()),
                                      getElemTy(getDst().getType()))))
      return failure();
    return verifyMatmulLike(*this, getLhs().getType(), getRhs().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TGemvOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyGemvTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                      getDst().getType())))
      return failure();
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getLhs().getType()),
                                      getElemTy(getRhs().getType()),
                                      getElemTy(getDst().getType()))))
      return failure();
    return verifyMatmulLike(*this, getLhs().getType(), getRhs().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TMatmulAccOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyAccTileCommon(*this, getAccIn().getType(), "acc_in")) ||
      failed(verifyMatTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                   getDst().getType())))
    return failure();
  return success();
}

LogicalResult mlir::pto::TGemvAccOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyAccTileCommon(*this, getAccIn().getType(), "acc_in")) ||
      failed(verifyGemvTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                    getDst().getType())))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// inferReturnTypes() for matmul ops (keep your existing code)
//===----------------------------------------------------------------------===
static mlir::Type inferMatmulTileResult2DFromAB(MLIRContext *context, ValueRange operands) {
  if (operands.size() < 2)
    return mlir::Type();

  auto lhsTile = dyn_cast<mlir::pto::TileType>(operands[0].getType());
  auto rhsTile = dyn_cast<mlir::pto::TileType>(operands[1].getType());
  if (!lhsTile || !rhsTile)
    return mlir::Type();

  Type elemTy = lhsTile.getElementType();

  if (operands.size() >= 3) {
    if (auto biasTile = dyn_cast<mlir::pto::TileType>(operands[2].getType())) {
      return mlir::pto::TileType::get(context, biasTile.getShape(), elemTy);
    }
  }

  auto lhsShape = lhsTile.getShape();
  auto rhsShape = rhsTile.getShape();
  if (lhsShape.size() >= 2 && rhsShape.size() >= 2) {
    int64_t M = lhsShape[0];
    int64_t N = rhsShape[1];
    llvm::SmallVector<int64_t, 2> outShape = {M, N};
    return mlir::pto::TileType::get(context, outShape, elemTy);
  }

  return mlir::Type();
}

static RankedTensorType inferMatmulResult2DFromAB(ValueRange operands) {
  if (operands.size() < 2)
    return RankedTensorType();

  auto lhsTy = dyn_cast<ShapedType>(operands[0].getType());
  auto rhsTy = dyn_cast<ShapedType>(operands[1].getType());
  if (!lhsTy || !rhsTy || !lhsTy.hasRank() || !rhsTy.hasRank())
    return RankedTensorType();

  Type elemTy = lhsTy.getElementType();

  if (operands.size() >= 3) {
    if (auto biasRT = dyn_cast<RankedTensorType>(operands[2].getType()))
      return RankedTensorType::get(biasRT.getShape(), elemTy);
    if (auto biasMR = dyn_cast<MemRefType>(operands[2].getType())) {
      if (biasMR.hasStaticShape())
        return RankedTensorType::get(biasMR.getShape(), elemTy);
    }
  }

  if (lhsTy.getRank() >= 2 && rhsTy.getRank() >= 2) {
    int64_t M = lhsTy.getDimSize(0);
    int64_t N = rhsTy.getDimSize(1);
    return RankedTensorType::get({M, N}, elemTy);
  }

  return RankedTensorType();
}

static RankedTensorType inferAccReturnFromAccIn(ValueRange operands) {
  if (operands.empty())
    return RankedTensorType();
  if (auto accRT = dyn_cast<RankedTensorType>(operands[0].getType()))
    return accRT;
  return RankedTensorType();
}

namespace mlir {
namespace pto {

static LogicalResult parseShapeAndElem(AsmParser &parser,
                                       SmallVectorImpl<int64_t> &shape,
                                       Type &elementType,
                                       bool allowDynamic) {
  if (parser.parseLess())
    return failure();

  if (parser.parseDimensionList(shape, allowDynamic))
    return failure();

  if (parser.parseType(elementType))
    return failure();

  if (parser.parseGreater())
    return failure();

  return success();
}

static void printShapeAndElem(AsmPrinter &printer,
                              ArrayRef<int64_t> shape,
                              Type elementType) {
  printer << "<";
  for (auto d : shape) {
    if (d == ShapedType::kDynamic)
      printer << "?";
    else
      printer << d;
    printer << "x";
  }
  printer.printType(elementType);
  printer << ">";
}

// =============================================================================
// PartitionTensorViewType Implementation
// =============================================================================

Type PartitionTensorViewType::parse(AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elemTy;
  if (failed(parseShapeAndElem(parser, shape, elemTy, /*allowDynamic=*/true)))
    return Type();
  
  return PartitionTensorViewType::get(parser.getContext(), shape, elemTy);
}

void PartitionTensorViewType::print(AsmPrinter &printer) const {
  printShapeAndElem(printer, getShape(), getElementType());
}

// ---- TileType ----
Type TileType::parse(AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elemTy;
  if (failed(parseShapeAndElem(parser, shape, elemTy, /*allowDynamic=*/true)))
    return Type();
  return TileType::get(parser.getContext(), shape, elemTy);
}

void TileType::print(AsmPrinter &printer) const {
  printShapeAndElem(printer, getShape(), getElementType());
}

// =============================================================================
// Decompose Helper (Reverse Engineering AffineMap -> Strides)
// =============================================================================

// Helper: 递归地将 Add 表达式拆解为单独的项列表
static void flattenAddExpr(AffineExpr expr, SmallVectorImpl<AffineExpr> &terms) {
  if (auto add = expr.dyn_cast<AffineBinaryOpExpr>()) {
    if (add.getKind() == AffineExprKind::Add) {
      flattenAddExpr(add.getLHS(), terms);
      flattenAddExpr(add.getRHS(), terms);
      return;
    }
  }
  terms.push_back(expr);
}

// Helper: 从 AffineMap 中提取 Strides
static void decomposeStridedLayout(AffineMap map, SmallVectorImpl<int64_t> &strides) {
  // 1. 初始化
  strides.assign(map.getNumDims(), 0);
  
  if (map.getNumResults() != 1) return;
  
  // 2. 摊平表达式
  SmallVector<AffineExpr, 4> terms;
  flattenAddExpr(map.getResult(0), terms);

  // 3. 分析每一项
  for (auto term : terms) {
    // 情况 A: dN * Const 或 Const * dN
    if (auto mul = term.dyn_cast<AffineBinaryOpExpr>()) {
      if (mul.getKind() == AffineExprKind::Mul) {
        AffineExpr lhs = mul.getLHS();
        AffineExpr rhs = mul.getRHS();

        // 尝试匹配 LHS=Dim, RHS=Const
        if (auto dim = lhs.dyn_cast<AffineDimExpr>()) {
          if (auto cst = rhs.dyn_cast<AffineConstantExpr>()) {
            strides[dim.getPosition()] = cst.getValue();
            continue;
          }
        }
        
        // 尝试匹配 LHS=Const, RHS=Dim (乘法交换律)
        if (auto dim = rhs.dyn_cast<AffineDimExpr>()) {
          if (auto cst = lhs.dyn_cast<AffineConstantExpr>()) {
            strides[dim.getPosition()] = cst.getValue();
            continue;
          }
        }
      }
    }
    // 情况 B: 单独的 dN (隐含 Stride = 1)
    else if (auto dim = term.dyn_cast<AffineDimExpr>()) {
      strides[dim.getPosition()] = 1;
    }
  }
}

// =============================================================================
// [Critical] Strict Alignment Protocol Helper
// =============================================================================
// This function is the SINGLE source of truth for building the AffineMap.
// Both the Parser and the Op Inference MUST use this exact function.
// It ensures that the order of AffineExpr addition is:
//   0 + (d0*str0 + d1*str1...) + (s0*str0 + s1*str1...)
// This guarantees bitwise-identical AffineMaps for verification.
static AffineMap buildStrictBitwiseAffineMap(MLIRContext *ctx, 
                                             ArrayRef<int64_t> strides, 
                                             bool isMultiDimSymbol) {
  unsigned rank = strides.size();
  
  // Step 1: Initialize with Constant(0)
  AffineExpr totalExpr = getAffineConstantExpr(0, ctx);

  // Step 2: Add Dimensions (d0*str0 + d1*str1...)
  // Strictly in order: 0, 1, 2...
  for (unsigned i = 0; i < rank; ++i) {
    auto dim = getAffineDimExpr(i, ctx);
    auto str = getAffineConstantExpr(strides[i], ctx);
    totalExpr = totalExpr + (dim * str);
  }

  // Step 3: Add Symbols (s0*str0 + s1*str1...)
  // Strictly in order: 0, 1, 2...
  if (isMultiDimSymbol) {
    for (unsigned i = 0; i < rank; ++i) {
      auto sym = getAffineSymbolExpr(i, ctx);
      auto str = getAffineConstantExpr(strides[i], ctx);
      totalExpr = totalExpr + (sym * str);
    }
  } 
  // (Optional: handle single dynamic offset case if needed, omitted for clarity)

  // numSymbols is rank if multi-dim (for offsets), else 0
  unsigned numSymbols = isMultiDimSymbol ? rank : 0;
  return AffineMap::get(rank, numSymbols, totalExpr);
}


// =============================================================================
// Parser Implementation
// =============================================================================

// Helper for parsing [64, 1]
static ParseResult parseStrideList(AsmParser &parser, SmallVectorImpl<int64_t> &strides) {
  if (parser.parseLSquare()) return failure();
  do {
    int64_t stride;
    if (parser.parseInteger(stride)) return failure();
    strides.push_back(stride);
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseRSquare()) return failure();
  return success();
}

// The custom attribute parser for: strided<[64, 1], offset: [?, ?]>
static ParseResult parseStridedLayout(AsmParser &parser, Attribute &layout) {
  if (parser.parseLess()) return failure();
  
  // 1. Parse Strides
  SmallVector<int64_t> strides;
  if (parseStrideList(parser, strides)) return failure();
  
  bool isMultiDim = false;
  unsigned numSymbols = 0;

  // 2. Parse Offset
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseKeyword("offset") || parser.parseColon()) return failure();
    
    // Check for multi-dim syntax: [?, ?]
    if (succeeded(parser.parseOptionalLSquare())) {
      isMultiDim = true;
      do {
        if (parser.parseQuestion()) return failure();
        numSymbols++;
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRSquare()) return failure();
    } else {
      // Fallback for old scalar syntax '?'
      if (parser.parseOptionalQuestion()) { /* handle single scalar */ }
    }
  }
  
  if (parser.parseGreater()) return failure();

  // 3. Validation
  if (isMultiDim && numSymbols != strides.size()) {
    return parser.emitError(parser.getCurrentLocation(), 
                            "Number of offset symbols must match rank");
  }

  // 4. [CALL SHARED BUILDER]
  // Delegate to the strict builder
  MLIRContext *ctx = parser.getContext();
  AffineMap map = buildStrictBitwiseAffineMap(ctx, strides, isMultiDim);
  
  layout = AffineMapAttr::get(map);
  return success();
}

// =============================================================================
// Printer Implementation
// =============================================================================

static void printLayout(AsmPrinter &printer, Attribute layoutAttr) {
  if (!layoutAttr) return;
  auto mapAttr = llvm::dyn_cast<AffineMapAttr>(layoutAttr);
  if (!mapAttr) { printer << ", " << layoutAttr; return; }

  AffineMap map = mapAttr.getValue();
  if (map.isIdentity()) return; 

  // 1. [核心修改] 反解 Strides
  SmallVector<int64_t> strides;
  decomposeStridedLayout(map, strides);

  printer << ", strided<[";
  // 2. 打印真实的 strides
  llvm::interleaveComma(strides, printer); 
  printer << "]";

  // Print Offset: [?, ?]
  unsigned numSyms = map.getNumSymbols();
  if (numSyms > 0) {
    printer << ", offset: [";
    for (unsigned i = 0; i < numSyms; ++i) {
      printer << "?";
      if (i < numSyms - 1) printer << ", ";
    }
    printer << "]";
  }
  printer << ">";
}

// ---- TileBuf ---


// Tile subset 相关实现

// =============================================================================
// Op Interface Implementation: SubsetOp
// =============================================================================

LogicalResult SubsetOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {

  // 1. 获取 Source Type
  if (operands.empty()) return failure();
  auto sourceType = llvm::dyn_cast<TileBufType>(operands[0].getType());
  if (!sourceType) return failure();

  // 2. 获取 Result Shape (Sizes)
  ArrayAttr sizeAttr;
  if (properties) {
    const auto *prop = properties.as<SubsetOp::Properties *>();
    if (prop) sizeAttr = prop->sizes;
  }
  if (!sizeAttr && attributes) {
    sizeAttr = attributes.getAs<ArrayAttr>("sizes");
  }
  if (!sizeAttr) return failure();

  SmallVector<int64_t> resultShape;
  for (auto attr : sizeAttr) {
    int64_t dim = llvm::cast<IntegerAttr>(attr).getInt();
    resultShape.push_back(dim);
  }

  // Derive valid shape from parent valid dims when possible.
  SmallVector<int64_t> validShape;
  constexpr int64_t kDynamicValidDim = -1;
  ArrayRef<int64_t> parentValid = sourceType.getValidShape();
  for (size_t i = 0, e = resultShape.size(); i < e; ++i) {
    int64_t sizeDim = resultShape[i];
    int64_t vdim = sizeDim;

    if (parentValid.size() == resultShape.size()) {
      int64_t pv = parentValid[i];
      if (pv < 0) {
        vdim = kDynamicValidDim;
      } else {
        int64_t off = 0;
        // operands: [source, offsets...]
        if (operands.size() > 1 + i) {
          auto offOpt = getConstIndexValue(operands[1 + i]);
          if (!offOpt) {
            vdim = kDynamicValidDim;
            validShape.push_back(vdim);
            continue;
          }
          off = *offOpt;
          // Interpret parent valid dims as a per-tile "period" when the parent
          // buffer is wider than the valid region (e.g. ping/pong workspace).
          // This avoids inferring a zero valid dim when taking a view at an
          // offset equal to the parent valid dim.
          //
          // Example:
          //   parent: shape 32x64, valid 32x32
          //   subset: offset [0,32], sizes [32,32]
          // should infer v_col=32 (not 0).
          int64_t diff = 0;
          if (pv > 0) {
            int64_t offMod = off % pv;
            if (offMod < 0)
              offMod += pv;
            diff = pv - offMod; // in [1, pv] when pv>0
          }
          if (diff < 0)
            diff = 0;
          vdim = std::min<int64_t>(sizeDim, diff);
        } else {
          vdim = kDynamicValidDim;
        }
      }
    }

    validShape.push_back(vdim);
  }

  // 3. 继承 Config (若为空使用默认)
  auto cfg = sourceType.getConfigAttr();
  if (!cfg) cfg = TileBufConfigAttr::getDefault(context);

  // 4. 构建 Result Type
  auto canonicalValidShape = canonicalizeTileBufValidShape(validShape);
  auto resultType = TileBufType::get(
      context, resultShape, sourceType.getElementType(),
      sourceType.getMemorySpace(), canonicalValidShape, cfg);

  inferredReturnTypes.push_back(resultType);
  return success();
}

// =============================================================================
// SubsetOp verifier
// =============================================================================
static bool getConstIndex(Value v, int64_t &out) {
  if (auto cOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
    out = cOp.value();
    return true;
  }
  if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>()) {
    out = cInt.value();
    return true;
  }
  if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(cOp.getValue())) {
      out = ia.getInt();
      return true;
    }
  }
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>())
    return getConstIndex(castOp.getIn(), out);
  if (auto extOp = v.getDefiningOp<arith::ExtSIOp>())
    return getConstIndex(extOp.getIn(), out);
  if (auto extOp = v.getDefiningOp<arith::ExtUIOp>())
    return getConstIndex(extOp.getIn(), out);
  if (auto truncOp = v.getDefiningOp<arith::TruncIOp>())
    return getConstIndex(truncOp.getIn(), out);
  return false;
}

static LogicalResult computeInnerShape(TileBufConfigAttr cfg, Type elemTy,
                                       int64_t &innerRows, int64_t &innerCols,
                                       bool &boxed, int32_t &bl, int32_t &sl) {
  auto readBLayoutI32 = [](Attribute attr, int32_t &out) -> bool {
    if (auto a = dyn_cast<BLayoutAttr>(attr)) {
      out = (int32_t)a.getValue();
      return true;
    }
    if (auto a = dyn_cast<IntegerAttr>(attr)) {
      out = (int32_t)a.getInt();
      return true;
    }
    return false;
  };
  auto readSLayoutI32 = [](Attribute attr, int32_t &out) -> bool {
    if (auto a = dyn_cast<SLayoutAttr>(attr)) {
      out = (int32_t)a.getValue();
      return true;
    }
    if (auto a = dyn_cast<IntegerAttr>(attr)) {
      out = (int32_t)a.getInt();
      return true;
    }
    return false;
  };
  bl = 0;
  sl = 0;
  int32_t fr = 512;
  (void)readBLayoutI32(cfg.getBLayout(), bl);
  (void)readSLayoutI32(cfg.getSLayout(), sl);
  if (auto attr = dyn_cast<IntegerAttr>(cfg.getSFractalSize())) fr = (int32_t)attr.getInt();

  boxed = (sl != 0);
  if (!boxed) {
    innerRows = 1;
    innerCols = 1;
    return success();
  }

  int64_t elemBytes = -1;
  if (auto ft = elemTy.dyn_cast<FloatType>()) {
    if (ft.isF16() || ft.isBF16()) elemBytes = 2;
    else if (ft.isF32()) elemBytes = 4;
    else if (ft.isF64()) elemBytes = 8;
  } else if (auto it = elemTy.dyn_cast<IntegerType>()) {
    int64_t bytes = it.getWidth() / 8;
    elemBytes = bytes > 0 ? bytes : 1;
  }
  if (elemBytes <= 0) return failure();

  if (fr == 1024) {
    innerRows = 16;
    innerCols = 16;
    return success();
  }
  if (fr == 32) {
    innerRows = 16;
    innerCols = 2;
    return success();
  }
  if (fr == 512) {
    if (sl == 1) {
      innerRows = 16;
      innerCols = 32 / elemBytes;
      return success();
    }
    if (sl == 2) {
      innerRows = 32 / elemBytes;
      innerCols = 16;
      return success();
    }
  }
  return failure();
}

mlir::LogicalResult mlir::pto::SubsetOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  auto srcTy = llvm::dyn_cast<TileBufType>(getSource().getType());
  auto dstTy = llvm::dyn_cast<TileBufType>(getResult().getType());
  if (!srcTy || !dstTy)
    return emitOpError("expects tile_buf src and tile_buf result");
  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError("expects rank-2 tilebuf for src/dst");

  auto cfg = srcTy.getConfigAttr();
  if (!cfg) cfg = TileBufConfigAttr::getDefault(getContext());

  int64_t innerRows = 1, innerCols = 1;
  bool boxed = false;
  int32_t bl = 0, sl = 0;
  if (failed(computeInnerShape(cfg, srcTy.getElementType(), innerRows, innerCols,
                               boxed, bl, sl)))
    return emitOpError("unsupported tile layout for subset");

  if (!boxed)
    return success();

  // Boxed layout: require static 2D sizes with inner alignment. Offsets may be
  // dynamic, but static offsets must be aligned.
  auto sizesAttr = getSizes();
  if (!sizesAttr || sizesAttr.size() != 2)
    return emitOpError("boxed layout subset expects 2D sizes");

  int64_t sizeR = cast<IntegerAttr>(sizesAttr[0]).getInt();
  int64_t sizeC = cast<IntegerAttr>(sizesAttr[1]).getInt();
  if (sizeR <= 0 || sizeC <= 0)
    return emitOpError("subset sizes must be positive");

  if (sizeR % innerRows != 0 || sizeC % innerCols != 0)
    return emitOpError("boxed layout subset sizes must be multiples of inner shape");

  if (getOffsets().size() != 2)
    return emitOpError("boxed layout subset expects 2D offsets");

  int64_t offR = 0, offC = 0;
  bool offRConst = getConstIndex(getOffsets()[0], offR);
  bool offCConst = getConstIndex(getOffsets()[1], offC);

  if (offRConst) {
    if (offR < 0)
      return emitOpError("subset offsets must be non-negative");
    if (offR % innerRows != 0)
      return emitOpError("boxed layout subset offsets must be multiples of inner shape");
  }
  if (offCConst) {
    if (offC < 0)
      return emitOpError("subset offsets must be non-negative");
    if (offC % innerCols != 0)
      return emitOpError("boxed layout subset offsets must be multiples of inner shape");
  }

  auto srcShape = srcTy.getShape();
  if (srcShape.size() == 2 &&
      srcShape[0] != ShapedType::kDynamic &&
      srcShape[1] != ShapedType::kDynamic) {
    if (bl == 0) {
      if (sizeC != srcShape[1])
        return emitOpError("boxed RowMajor subset must keep full cols");
      if (!offCConst || offC != 0)
        return emitOpError("boxed RowMajor subset requires static col offset = 0");
    } else if (bl == 1) {
      if (sizeR != srcShape[0])
        return emitOpError("boxed ColMajor subset must keep full rows");
      if (!offRConst || offR != 0)
        return emitOpError("boxed ColMajor subset requires static row offset = 0");
    }
  } else {
    return emitOpError("boxed layout subset requires static source shape");
  }

  return success();
}

} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;
 
// =============================================================================
// Helper Functions
// =============================================================================
 
static AddressSpace getAddressSpace(Value val) {
  auto type = llvm::dyn_cast<MemRefType>(val.getType());
  if (!type) return AddressSpace::Zero; // Default
 
  // 假设你的 AddressSpaceAttr 存储在 MemRef 的 memorySpace 中
  // 需要根据你的 getPTOAddressSpaceAttr 实现来调整
  auto attr = llvm::dyn_cast_or_null<AddressSpaceAttr>(type.getMemorySpace());
  if (attr) return attr.getAddressSpace();
  return AddressSpace::Zero;
}
 
// =============================================================================
// Side Effects Implementation
// =============================================================================
 
// [Fix] 辅助函数：重载以支持 OpOperand* 和 OpResult，避免直接传 Value
 
// 针对操作数 (Operand) 的重载
static void addEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects,
    OpOperand *operand, MemoryEffects::Effect *effect) {
  if (operand)
    effects.emplace_back(effect, operand, SideEffects::DefaultResource::get());
}
 
// 针对结果 (Result) 的重载
static void addEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects,
    OpResult result, MemoryEffects::Effect *effect) {
  if (result)
    effects.emplace_back(effect, result, SideEffects::DefaultResource::get());
}

// === TLoadOp ===
// Read: src, Write: dst
// 针对 OpOperand* 的重载
void TLoadOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // [Fix] 单个操作数，直接取地址
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TAbsOp ===
// Read: src, Write: dst
void TAbsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TStoreOp ===
// Read: src, Write: dst (GM)
void TStoreOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMovOp ===
// Read: src, Write: dst
void TMovOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

#define PTO_ADD_READ(operand) addEffect(effects, &(operand), MemoryEffects::Read::get())
#define PTO_ADD_WRITE(operand) addEffect(effects, &(operand), MemoryEffects::Write::get())

#define PTO_DEFINE_UNARY_EFFECTS(OpClass, srcOperand, dstOperand)                    \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(srcOperand);                                                       \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

#define PTO_DEFINE_BINARY_EFFECTS(OpClass, lhsOperand, rhsOperand, dstOperand)       \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(lhsOperand);                                                       \
    PTO_ADD_READ(rhsOperand);                                                       \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

#define PTO_DEFINE_TERNARY_EFFECTS(OpClass, op0, op1, op2, dstOperand)               \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(op0);                                                              \
    PTO_ADD_READ(op1);                                                              \
    PTO_ADD_READ(op2);                                                              \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

void LoadScalarOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getPtrMutable());
}

void StoreScalarOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getPtrMutable());
}

// === Tile/Device ops added for InsertSync ===

// MGATHER: Read(mem, idx) -> Write(dst)
void MGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getMemMutable());
  PTO_ADD_READ(getIdxMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// MSCATTER: Read(src, idx) -> Write(mem)
void MScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getIdxMutable());
  PTO_ADD_WRITE(getMemMutable());
}

// TGETVAL: Read(src) -> scalar result
void TGetValOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
}

// TSETVAL: Write(dst) (single element update)
void TSetValOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

// Elementwise + reductions: mostly PIPE_V tilebuf ops
PTO_DEFINE_BINARY_EFFECTS(TAddOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_TERNARY_EFFECTS(TAddCOp, getSrc0Mutable(), getSrc1Mutable(), getSrc2Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TAddSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TAddSCOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())

PTO_DEFINE_BINARY_EFFECTS(TAndOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TAndSOp, getSrcMutable(), getDstMutable())

// TCI: Write(dst) (generates sequence)
void TCIOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TCmpOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TCmpSOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_UNARY_EFFECTS(TColExpandOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandMulOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandMaxOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandMinOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TColMaxOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TColMinOp, getSrcMutable(), getDstMutable())

void TColSumOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty()) {
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TCvtOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TDivOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())

// TDIVS has custom assembly format; conservatively treat first 2 operands as reads.
void TDivSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getScalarMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TExpOp, getSrcMutable(), getDstMutable())

// TEXPANDS: Write(dst) (broadcast scalar)
void TExpandsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

// TEXTRACT: Read(src) -> Write(dst)
void TExtractOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TINSERT: Read(src) -> Write(dst)
void TInsertOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TFillPadOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TFillPadExpandOp, getSrcMutable(), getDstMutable())

void TGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto indices = getIndicesMutable();
  if (!indices.empty()) {
    PTO_ADD_READ(indices[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TGatherBOp, getSrcMutable(), getOffsetsMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TLogOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TLReluOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_BINARY_EFFECTS(TMaxOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMaxSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TMinOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMinSOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_BINARY_EFFECTS(TMovFPOp, getSrcMutable(), getFpMutable(), getDstMutable())

void TMrgSortOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  for (auto &opnd : getSrcsMutable()) {
    PTO_ADD_READ(opnd);
  }
  for (auto &opnd : getDstsMutable()) {
    PTO_ADD_WRITE(opnd);
  }
  auto executed = getExcutedMutable();
  if (!executed.empty()) {
    PTO_ADD_WRITE(executed[0]);
  }
}

PTO_DEFINE_BINARY_EFFECTS(TMulOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMulSOp, getSrc0Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TNegOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TNotOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TOrOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TOrSOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_BINARY_EFFECTS(TPartAddOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TPartMaxOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TPartMinOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
// TPRELU: Read(src0, src1) -> Write(tmp, dst)
void TPReluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TRecipOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TReluOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TRemOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TRemSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TRowExpandOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TRowExpandDivOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TRowExpandMulOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TRowExpandSubOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())

// Row reductions use tmp scratch tile.
void TRowMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TRowMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TRowSumOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TRsqrtOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TScatterOp, getSrcMutable(), getIndexesMutable(), getDstMutable())

// Select: Read(mask, src0, src1) -> Write(dst)
void TSelOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getMaskMutable());
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TSelSOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())

PTO_DEFINE_BINARY_EFFECTS(TShlOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TShrOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TShlSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TShrSOp, getSrcMutable(), getDstMutable())

// TSORT32: Read(src) -> Write(dst, idx)
void TSort32Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getDstMutable());
  PTO_ADD_WRITE(getIdxMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TSqrtOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TSubOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_TERNARY_EFFECTS(TSubCOp, getSrc0Mutable(), getSrc1Mutable(), getSrc2Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TSubSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TSubSCOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())

PTO_DEFINE_UNARY_EFFECTS(TXorSOp, getSrcMutable(), getDstMutable())

// TXOR: Read(src0, src1) -> Write(tmp?, dst)
void TXorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TTRANS: Read(src) -> Write(tmp, dst)
void TTransOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TPrintOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getSrcMutable());
}

#undef PTO_DEFINE_TERNARY_EFFECTS
#undef PTO_DEFINE_BINARY_EFFECTS
#undef PTO_DEFINE_UNARY_EFFECTS
#undef PTO_ADD_WRITE
#undef PTO_ADD_READ

// === TMatmulOp ===
// Read: lhs, rhs, (bias), Write: dst
void TMatmulOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Singleton -> 直接取地址
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulAccOp ===
// Read: acc_in, lhs, rhs, Write: dst
void TMatmulAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAccInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulBiasOp ===
// Read: a, b, bias, Write: dst
void TMatmulBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  // 这里的 bias 是必选的 AnyType:$bias，所以是 Singleton
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvOp ===
// Read: lhs, rhs, Write: dst
void TGemvOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvAccOp ===
// Read: acc_in, lhs, rhs, Write: dst
void TGemvAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAccInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvBiasOp ===
// Read: a, b, bias, Write: dst
void TGemvBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulOp ===
void TMatmulMxOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulAccMxOp ===
// Read: acc_in, lhs, rhs, Write: dst
void TMatmulMxAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getCInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulBiasMxOp ===
// Read: a, b, bias, Write: dst
void TMatmulMxBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  // 这里的 bias 是必选的 AnyType:$bias，所以是 Singleton
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// [Include 必须放在最后]
#include "PTO/IR/PTOInterfaces.cpp.inc"
#define GET_OP_CLASSES
#include "PTO/IR/PTOOps.cpp.inc"
