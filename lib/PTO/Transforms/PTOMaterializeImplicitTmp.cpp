// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOMaterializeImplicitTmp.cpp --------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/Passes.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTODialect.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cassert>

using namespace mlir;

namespace {

constexpr int64_t kRowMajorNoneBoxSFractalSize = 512;

static pto::TileBufConfigAttr makeRowMajorNoneBoxConfig(MLIRContext *ctx) {
  OpBuilder builder(ctx);
  return pto::TileBufConfigAttr::get(
      ctx, pto::BLayoutAttr::get(ctx, pto::BLayout::RowMajor),
      pto::SLayoutAttr::get(ctx, pto::SLayout::NoneBox),
      builder.getI32IntegerAttr(kRowMajorNoneBoxSFractalSize),
      pto::PadValueAttr::get(ctx, pto::PadValue::Null),
      pto::CompactModeAttr::get(ctx, pto::CompactMode::Null));
}

static unsigned getTCIDstBitWidth(pto::TCIOp op) {
  auto tileTy = dyn_cast<pto::TileBufType>(op.getDst().getType());
  if (!tileTy) {
    return 0;
  }
  auto elemTy = dyn_cast<IntegerType>(tileTy.getElementType());
  if (!elemTy) {
    return 0;
  }
  return elemTy.getWidth();
}

static pto::TileBufType makeTCITmpType(MLIRContext *ctx, unsigned dstBitWidth) {
  // PTO-ISA TCI A2/A3 vector path needs 768B for b32 dst and 1792B for
  // b16 dst. Use an f32 1xN tmp with the exact minimum capacity.
  int64_t cols = dstBitWidth == 16 ? 448 : 192;
  return pto::TileBufType::get(
      ctx, {1, cols}, Float32Type::get(ctx),
      pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::VEC), {1, cols},
      makeRowMajorNoneBoxConfig(ctx));
}

static std::optional<int64_t> getElemBytes(Type elemTy) {
  unsigned bits = pto::getPTOStorageElemBitWidth(elemTy);
  if (bits == 0 || bits % mlir::pto::kValue8 != 0) {
    return std::nullopt;
  }
  return bits / mlir::pto::kValue8;
}

static SmallVector<int64_t, mlir::pto::kValue4> getValidShapeVec(Type ty) {
  if (auto tileTy = dyn_cast<pto::TileBufType>(ty)) {
    return SmallVector<int64_t, mlir::pto::kValue4>(tileTy.getValidShape().begin(),
                                   tileTy.getValidShape().end());
  }
  return {};
}

static SmallVector<int64_t, mlir::pto::kValue4> getShapeVec(Type ty) {
  if (auto tileTy = dyn_cast<pto::TileBufType>(ty)) {
    return SmallVector<int64_t, mlir::pto::kValue4>(tileTy.getShape().begin(),
                                   tileTy.getShape().end());
  }
  return {};
}

static int64_t ceilDiv(int64_t lhs, int64_t rhs) {
  return (lhs + rhs - 1) / rhs;
}

static bool hasDynamicDim(ArrayRef<int64_t> dims) {
  return llvm::any_of(dims, [](int64_t dim) {
    return dim == ShapedType::kDynamic;
  });
}

static pto::TileBufType makeVecTmpType(MLIRContext *ctx,
                                       ArrayRef<int64_t> shape,
                                       Type elementType,
                                       ArrayRef<int64_t> validShape) {
  return pto::TileBufType::get(
      ctx, shape, elementType,
      pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::VEC), validShape,
      makeRowMajorNoneBoxConfig(ctx));
}

static FailureOr<pto::TileBufType> makeSameShapeTmpType(MLIRContext *ctx,
                                                        Value like,
                                                        Type elementType = {}) {
  auto likeTy = dyn_cast<pto::TileBufType>(like.getType());
  if (!likeTy) {
    return failure();
  }
  if (!elementType) {
    elementType = likeTy.getElementType();
  }
  auto shape = getShapeVec(like.getType());
  auto validShape = getValidShapeVec(like.getType());
  if (shape.empty() || validShape.empty() || hasDynamicDim(shape) ||
      hasDynamicDim(validShape)) {
    return failure();
  }
  return makeVecTmpType(ctx, shape, elementType, validShape);
}

static FailureOr<Value> createAllocTmp(OpBuilder &builder, Location loc,
                                       pto::TileBufType tmpType) {
  return builder
      .create<pto::AllocTileOp>(loc, tmpType, Value(), Value(), Value())
      .getResult();
}

static void copyAttrsExceptOperandSegments(Operation *from, OperationState &to) {
  for (NamedAttribute attr : from->getAttrs()) {
    if (attr.getName() == "operandSegmentSizes") {
      continue;
    }
    to.addAttribute(attr.getName(), attr.getValue());
  }
}

static void rebuildWithOperands(Operation *op, ArrayRef<Value> operands,
                                std::optional<ArrayRef<int32_t>> segments) {
  OpBuilder builder(op);
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(operands);
  if (op->hasTrait<OpTrait::AttrSizedOperandSegments>()) {
    assert(segments && "AttrSizedOperandSegments op must supply segments");
    state.addAttribute("operandSegmentSizes",
                       builder.getDenseI32ArrayAttr(*segments));
  }
  copyAttrsExceptOperandSegments(op, state);
  builder.create(state);
  op->erase();
}

static bool validShapesCompatible(ArrayRef<int64_t> lhs,
                                  ArrayRef<int64_t> rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (auto [l, r] : llvm::zip(lhs, rhs)) {
    if (l != ShapedType::kDynamic && r != ShapedType::kDynamic && l != r) {
      return false;
    }
  }
  return true;
}

static bool isRowMajorTile(Value value) {
  auto tileTy = dyn_cast<pto::TileBufType>(value.getType());
  return tileTy && tileTy.getBLayoutValueI32() ==
                       static_cast<int32_t>(pto::BLayout::RowMajor);
}

static bool isColMajorTile(Value value) {
  auto tileTy = dyn_cast<pto::TileBufType>(value.getType());
  return tileTy && tileTy.getBLayoutValueI32() ==
                       static_cast<int32_t>(pto::BLayout::ColMajor);
}

enum class RowExpandMode {
  Unknown,
  Mode1ColMajorScalar,
  Mode2RowMajorBlock,
};

static RowExpandMode classifyTRowExpandBinaryMode(Value src0, Value src1,
                                                  Value dst) {
  auto dstValid = getValidShapeVec(dst.getType());
  auto src0Valid = getValidShapeVec(src0.getType());
  auto src1Valid = getValidShapeVec(src1.getType());
  if (dstValid.size() != mlir::pto::kValue2 || src0Valid.size() != mlir::pto::kValue2 || src1Valid.size() != mlir::pto::kValue2) {
    return RowExpandMode::Unknown;
  }

  Value expanded;
  ArrayRef<int64_t> expandedValid;
  if (validShapesCompatible(src0Valid, dstValid)) {
    expanded = src1;
    expandedValid = src1Valid;
  } else if (validShapesCompatible(src1Valid, dstValid)) {
    expanded = src0;
    expandedValid = src0Valid;
  } else {
    return RowExpandMode::Unknown;
  }

  int64_t expandedCols = expandedValid[1];
  if (isColMajorTile(expanded) &&
      (expandedCols == ShapedType::kDynamic || expandedCols == 1)) {
    return RowExpandMode::Mode1ColMajorScalar;
  }

  auto dstTileTy = dyn_cast<pto::TileBufType>(dst.getType());
  if (!dstTileTy) {
    return RowExpandMode::Unknown;
  }
  auto elemBytes = getElemBytes(dstTileTy.getElementType());
  if (!elemBytes || *elemBytes == 0) {
    return RowExpandMode::Unknown;
  }
  int64_t expectedMode2Cols = 32 / *elemBytes;
  if (isRowMajorTile(expanded) &&
      (expandedCols == ShapedType::kDynamic ||
       expandedCols == expectedMode2Cols)) {
    return RowExpandMode::Mode2RowMajorBlock;
  }

  return RowExpandMode::Unknown;
}

static pto::TileBufType makeTRowExpandTmpType(MLIRContext *ctx,
                                              pto::TileBufType dstTy) {
  constexpr int64_t kTmpBytes = 8192;
  std::optional<int64_t> elemBytes = getElemBytes(dstTy.getElementType());
  int64_t cols = elemBytes && *elemBytes > 0 ? kTmpBytes / *elemBytes : 2048;
  return pto::TileBufType::get(
      ctx, {1, cols}, dstTy.getElementType(),
      pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::VEC), {1, cols},
      makeRowMajorNoneBoxConfig(ctx));
}

static FailureOr<pto::TileBufType> makeA5PlaceholderTmpType(
    MLIRContext *ctx, Value like, Type elementType = {}) {
  auto likeTy = dyn_cast<pto::TileBufType>(like.getType());
  if (!likeTy) {
    return failure();
  }
  if (!elementType) {
    elementType = likeTy.getElementType();
  }
  auto elemBytes = getElemBytes(elementType);
  if (!elemBytes || *elemBytes <= 0) {
    return failure();
  }
  int64_t cols = std::max<int64_t>(1, 32 / *elemBytes);
  return makeVecTmpType(ctx, {1, cols}, elementType, {1, cols});
}

static void replaceTRowExpandBinaryOpWithTmp(Operation *op, Value src0,
                                             Value src1, Value tmp, Value dst) {
  rebuildWithOperands(op, {src0, src1, tmp, dst}, ArrayRef<int32_t>{1, 1, 1, 1});
}

template <typename OpTy>
static LogicalResult materializeTRowExpandTmp(OpTy op, bool requireExplicitTmp,
                                              MLIRContext *ctx) {
  if (op.getTmp()) {
    return success();
  }
  if (pto::getTargetArch(op.getOperation()) == pto::PTOArch::A5) {
    return success();
  }

  RowExpandMode mode =
      classifyTRowExpandBinaryMode(op.getSrc0(), op.getSrc1(), op.getDst());
  if (mode != RowExpandMode::Mode1ColMajorScalar) {
    return success();
  }

  if (requireExplicitTmp) {
    // Row-expand is the one A2/A3 tmp-aware family whose no-tmp overload is
    // still a valid backend contract: mode 1 falls back to pto-isa's internal
    // 8KB TMP_UB_OFFSET scratch area, while mode 2 does not need tmp.  Level3
    // inputs are already memory-planned by the frontend, so preserve their
    // no-tmp form instead of creating an unaddressed alloc_tile.
    return success();
  }

  auto dstTy = dyn_cast<pto::TileBufType>(op.getDst().getType());
  if (!dstTy) {
    return op.emitOpError("expects tile_buf dst when materializing implicit tmp");
  }

  OpBuilder builder(op);
  Value tmp =
      builder
          .create<pto::AllocTileOp>(op.getLoc(),
                                    makeTRowExpandTmpType(ctx, dstTy), Value(),
                                    Value(), Value())
          .getResult();
  replaceTRowExpandBinaryOpWithTmp(op.getOperation(), op.getSrc0(), op.getSrc1(),
                                   tmp, op.getDst());
  return success();
}

static LogicalResult replaceTColSumWithTmp(pto::TColSumOp op,
                                           bool requireExplicitTmp,
                                           MLIRContext *ctx) {
  if (op.getTmp() || !op.getIsBinary()) {
    return success();
  }
  if (requireExplicitTmp) {
    return op.emitOpError(
        "requires explicit tmp for binary tcolsum when PlanMemory is skipped");
  }

  auto srcTy = dyn_cast<pto::TileBufType>(op.getSrc().getType());
  if (!srcTy) {
    return op.emitOpError("expects tile_buf src when materializing implicit tmp");
  }
  auto valid = getValidShapeVec(op.getSrc().getType());
  if (valid.size() != mlir::pto::kValue2 || hasDynamicDim(valid)) {
    return op.emitOpError(
        "requires static src valid_shape to materialize binary tcolsum tmp");
  }

  SmallVector<int64_t, mlir::pto::kValue2> tmpShape{ceilDiv(valid[0], mlir::pto::kValue2), valid[1]};
  auto tmpType = makeVecTmpType(ctx, tmpShape, srcTy.getElementType(), tmpShape);
  OpBuilder builder(op);
  FailureOr<Value> tmp = createAllocTmp(builder, op.getLoc(), tmpType);
  if (failed(tmp)) {
    return failure();
  }

  rebuildWithOperands(op.getOperation(), {op.getSrc(), *tmp, op.getDst()},
                      ArrayRef<int32_t>{1, 1, 1});
  return success();
}

static LogicalResult replaceTQuantWithTmp(pto::TQuantOp op,
                                          bool requireExplicitTmp,
                                          MLIRContext *ctx) {
  if (op.getTmp() || pto::getTargetArch(op.getOperation()) == pto::PTOArch::A5) {
    return success();
  }
  if (requireExplicitTmp) {
    return op.emitOpError("requires explicit tmp when PlanMemory is skipped");
  }

  FailureOr<pto::TileBufType> tmpType = makeSameShapeTmpType(
      ctx, op.getSrc(), Float32Type::get(ctx));
  if (failed(tmpType)) {
    return op.emitOpError(
        "requires static tile_buf src to materialize implicit tquant tmp");
  }
  OpBuilder builder(op);
  FailureOr<Value> tmp = createAllocTmp(builder, op.getLoc(), *tmpType);
  if (failed(tmp)) {
    return failure();
  }

  SmallVector<Value> operands{op.getSrc(), op.getFp()};
  if (op.getOffset()) {
    operands.push_back(op.getOffset());
  }
  operands.push_back(*tmp);
  operands.push_back(op.getDst());
  rebuildWithOperands(op.getOperation(), operands,
                      ArrayRef<int32_t>{1, 1, op.getOffset() ? 1 : 0, 1, 1});
  return success();
}

static bool isFloatingPointTile(Value value) {
  auto tileTy = dyn_cast<pto::TileBufType>(value.getType());
  return tileTy && isa<FloatType>(tileTy.getElementType());
}

static LogicalResult replaceTPowWithTmp(pto::TPowOp op,
                                        bool requireExplicitTmp,
                                        MLIRContext *ctx) {
  if (op.getTmp() || !isFloatingPointTile(op.getDst())) {
    return success();
  }
  if (requireExplicitTmp) {
    return op.emitOpError("requires explicit tmp when PlanMemory is skipped");
  }

  FailureOr<pto::TileBufType> tmpType = makeSameShapeTmpType(ctx, op.getDst());
  if (failed(tmpType)) {
    return op.emitOpError(
        "requires static tile_buf dst to materialize implicit tpow tmp");
  }
  OpBuilder builder(op);
  FailureOr<Value> tmp = createAllocTmp(builder, op.getLoc(), *tmpType);
  if (failed(tmp)) {
    return failure();
  }

  rebuildWithOperands(op.getOperation(),
                      {op.getBase(), op.getExp(), op.getDst(), *tmp},
                      ArrayRef<int32_t>{1, 1, 1, 1});
  return success();
}

static LogicalResult replaceTPowSWithTmp(pto::TPowSOp op,
                                         bool requireExplicitTmp,
                                         MLIRContext *ctx) {
  if (op.getTmp() || !isFloatingPointTile(op.getDst())) {
    return success();
  }
  if (requireExplicitTmp) {
    return op.emitOpError("requires explicit tmp when PlanMemory is skipped");
  }

  FailureOr<pto::TileBufType> tmpType = makeSameShapeTmpType(ctx, op.getDst());
  if (failed(tmpType)) {
    return op.emitOpError(
        "requires static tile_buf dst to materialize implicit tpows tmp");
  }
  OpBuilder builder(op);
  FailureOr<Value> tmp = createAllocTmp(builder, op.getLoc(), *tmpType);
  if (failed(tmp)) {
    return failure();
  }

  rebuildWithOperands(op.getOperation(),
                      {op.getSrc(), op.getScalar(), op.getDst(), *tmp},
                      ArrayRef<int32_t>{1, 1, 1, 1});
  return success();
}

static LogicalResult replaceTSort32WithTmp(pto::TSort32Op op,
                                           bool requireExplicitTmp,
                                           MLIRContext *ctx) {
  if (op.getTmp()) {
    return success();
  }
  auto valid = getValidShapeVec(op.getSrc().getType());
  if (valid.size() != mlir::pto::kValue2) {
    return success();
  }
  // Only a statically-known 32-aligned width provably skips the tail path and
  // needs no tmp. A dynamic width may be non-aligned at runtime, so it must be
  // treated conservatively rather than assumed aligned.
  bool dynamicWidth = valid[1] == ShapedType::kDynamic;
  if (!dynamicWidth && valid[1] % mlir::pto::kValue32 == 0) {
    return success();
  }
  if (requireExplicitTmp) {
    return op.emitOpError(
        "requires explicit tmp for tsort32 with dynamic or non-32-aligned width "
        "when PlanMemory is skipped");
  }

  FailureOr<pto::TileBufType> tmpType = failure();
  if (dynamicWidth) {
    // The runtime width may be non-32-aligned, so size the scratch buffer by
    // the full physical width to guarantee room for the tail path.
    auto srcTy = dyn_cast<pto::TileBufType>(op.getSrc().getType());
    auto shape = getShapeVec(op.getSrc().getType());
    if (srcTy && shape.size() == mlir::pto::kValue2 && !hasDynamicDim(shape)) {
      tmpType = makeVecTmpType(ctx, shape, srcTy.getElementType(), shape);
    }
  } else {
    tmpType = makeSameShapeTmpType(ctx, op.getSrc());
  }
  if (failed(tmpType)) {
    return op.emitOpError(
        "requires static tile_buf src to materialize implicit tsort32 tmp");
  }
  OpBuilder builder(op);
  FailureOr<Value> tmp = createAllocTmp(builder, op.getLoc(), *tmpType);
  if (failed(tmp)) {
    return failure();
  }

  rebuildWithOperands(op.getOperation(), {op.getSrc(), op.getIdx(), *tmp, op.getDst()},
                      ArrayRef<int32_t>{1, 1, 1, 1});
  return success();
}

template <typename OpTy>
static LogicalResult replaceRowReductionWithTmp(OpTy op,
                                                 bool requireExplicitTmp,
                                                 MLIRContext *ctx) {
  if (op.getTmp()) {
    return success();
  }

  bool isA5 = pto::getTargetArch(op.getOperation()) == pto::PTOArch::A5;
  if (requireExplicitTmp && !isA5) {
    return op.emitOpError("requires explicit tmp when PlanMemory is skipped");
  }

  FailureOr<pto::TileBufType> tmpType =
      isA5 ? makeA5PlaceholderTmpType(ctx, op.getSrc())
           : makeSameShapeTmpType(ctx, op.getSrc());
  if (failed(tmpType)) {
    return op.emitOpError(
        "requires static tile_buf src to materialize implicit row-reduction tmp");
  }
  OpBuilder builder(op);
  FailureOr<Value> tmp = createAllocTmp(builder, op.getLoc(), *tmpType);
  if (failed(tmp)) {
    return failure();
  }

  rebuildWithOperands(op.getOperation(), {op.getSrc(), *tmp, op.getDst()},
                      ArrayRef<int32_t>{1, 1, 1});
  return success();
}

static LogicalResult replaceTXorWithTmp(pto::TXorOp op,
                                        bool requireExplicitTmp,
                                        MLIRContext *ctx) {
  if (op.getTmp()) {
    return success();
  }
  bool isA5 = pto::getTargetArch(op.getOperation()) == pto::PTOArch::A5;
  if (requireExplicitTmp && !isA5) {
    return op.emitOpError("requires explicit tmp when PlanMemory is skipped");
  }
  FailureOr<pto::TileBufType> tmpType =
      isA5 ? makeA5PlaceholderTmpType(ctx, op.getDst())
           : makeSameShapeTmpType(ctx, op.getDst());
  if (failed(tmpType)) {
    return op.emitOpError(
        "requires static tile_buf dst to materialize implicit txor tmp");
  }
  OpBuilder builder(op);
  FailureOr<Value> tmp = createAllocTmp(builder, op.getLoc(), *tmpType);
  if (failed(tmp)) {
    return failure();
  }
  rebuildWithOperands(op.getOperation(),
                      {op.getSrc0(), op.getSrc1(), *tmp, op.getDst()},
                      ArrayRef<int32_t>{1, 1, 1, 1});
  return success();
}

static LogicalResult replaceTXorSWithTmp(pto::TXorSOp op,
                                         bool requireExplicitTmp,
                                         MLIRContext *ctx) {
  if (op.getTmp()) {
    return success();
  }
  bool isA5 = pto::getTargetArch(op.getOperation()) == pto::PTOArch::A5;
  if (requireExplicitTmp && !isA5) {
    return op.emitOpError("requires explicit tmp when PlanMemory is skipped");
  }
  FailureOr<pto::TileBufType> tmpType =
      isA5 ? makeA5PlaceholderTmpType(ctx, op.getDst())
           : makeSameShapeTmpType(ctx, op.getDst());
  if (failed(tmpType)) {
    return op.emitOpError(
        "requires static tile_buf dst to materialize implicit txors tmp");
  }
  OpBuilder builder(op);
  FailureOr<Value> tmp = createAllocTmp(builder, op.getLoc(), *tmpType);
  if (failed(tmp)) {
    return failure();
  }
  rebuildWithOperands(op.getOperation(),
                      {op.getSrc(), op.getScalar(), *tmp, op.getDst()},
                      ArrayRef<int32_t>{1, 1, 1, 1});
  return success();
}

static LogicalResult replaceFixedDpsOpWithTmp(
    Operation *op, ArrayRef<Value> operands, pto::TileBufType tmpType,
    ArrayRef<int32_t> operandSegments, bool requireExplicitTmp,
    StringRef opName) {
  if (requireExplicitTmp) {
    return op->emitOpError(
        "requires explicit tmp when PlanMemory is skipped");
  }
  OpBuilder builder(op);
  FailureOr<Value> tmp = createAllocTmp(builder, op->getLoc(), tmpType);
  if (failed(tmp)) {
    return failure();
  }
  SmallVector<Value> finalOperands;
  finalOperands.reserve(operands.size() + 1);
  for (Value operand : operands) {
    if (operand) {
      finalOperands.push_back(operand);
    } else {
      finalOperands.push_back(*tmp);
}
  }
  rebuildWithOperands(op, finalOperands, operandSegments);
  (void)opName;
  return success();
}

static FailureOr<pto::TileBufType> makeTPReluTmpType(MLIRContext *ctx,
                                                      Value dst) {
  auto dstTy = dyn_cast<pto::TileBufType>(dst.getType());
  auto shape = getShapeVec(dst.getType());
  auto valid = getValidShapeVec(dst.getType());
  if (!dstTy || shape.size() != mlir::pto::kValue2 || valid.size() != mlir::pto::kValue2 ||
      hasDynamicDim(shape) || hasDynamicDim(valid)) {
    return failure();
  }
  int64_t validCols = ceilDiv(valid[1], 8);
  int64_t cols = std::max<int64_t>(32, ceilDiv(validCols, 32) * 32);
  return makeVecTmpType(ctx, {valid[0] + 1, cols}, IntegerType::get(ctx, mlir::pto::kValue8),
                        {valid[0], validCols});
}

static FailureOr<pto::TileBufType> makeRowsTmpType(MLIRContext *ctx,
                                                    Value dst, int64_t rows) {
  auto dstTy = dyn_cast<pto::TileBufType>(dst.getType());
  auto shape = getShapeVec(dst.getType());
  auto valid = getValidShapeVec(dst.getType());
  if (!dstTy || shape.size() != mlir::pto::kValue2 || valid.size() != mlir::pto::kValue2 ||
      hasDynamicDim(shape) || hasDynamicDim(valid)) {
    return failure();
  }
  return makeVecTmpType(ctx, {rows, shape[1]}, dstTy.getElementType(),
                        {rows, valid[1]});
}

static LogicalResult materializeFixedMandatoryTmp(Operation *op,
                                                   bool requireExplicitTmp,
                                                   MLIRContext *ctx) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<pto::TPReluOp>([&](auto typedOp) -> LogicalResult {
        if (typedOp.getTmp()) {
          return success();
        }
        bool isA5 =
            pto::getTargetArch(op) == pto::PTOArch::A5;
        auto type = isA5 ? makeA5PlaceholderTmpType(
                               ctx, typedOp.getDst(),
                               IntegerType::get(ctx, 8))
                         : makeTPReluTmpType(ctx, typedOp.getDst());
        if (failed(type)) {
          return typedOp.emitOpError(
              "requires static tile_buf dst to materialize implicit tprelu tmp");
        }
        return replaceFixedDpsOpWithTmp(
            op, {typedOp.getSrc0(), typedOp.getSrc1(), Value(),
                 typedOp.getDst()},
            *type, {1, 1, 1, 1}, isA5 ? false : requireExplicitTmp,
            "tprelu");
      })
      .Case<pto::TRemOp>([&](auto typedOp) -> LogicalResult {
        if (typedOp.getTmp()) {
          return success();
        }
        bool isA5 =
            pto::getTargetArch(op) == pto::PTOArch::A5;
        auto type = isA5 ? makeA5PlaceholderTmpType(ctx, typedOp.getDst())
                         : makeRowsTmpType(ctx, typedOp.getDst(), 2);
        if (failed(type)) {
          return typedOp.emitOpError(
              "requires static tile_buf dst to materialize implicit trem tmp");
        }
        return replaceFixedDpsOpWithTmp(
            op, {typedOp.getSrc0(), typedOp.getSrc1(), Value(),
                 typedOp.getDst()},
            *type, {1, 1, 1, 1}, isA5 ? false : requireExplicitTmp, "trem");
      })
      .Case<pto::TRemSOp>([&](auto typedOp) -> LogicalResult {
        if (typedOp.getTmp()) {
          return success();
        }
        bool isA5 =
            pto::getTargetArch(op) == pto::PTOArch::A5;
        auto type = isA5 ? makeA5PlaceholderTmpType(ctx, typedOp.getDst())
                         : makeRowsTmpType(ctx, typedOp.getDst(), 1);
        if (failed(type)) {
          return typedOp.emitOpError(
              "requires static tile_buf dst to materialize implicit trems tmp");
        }
        return replaceFixedDpsOpWithTmp(
            op, {typedOp.getSrc(), typedOp.getScalar(), Value(),
                 typedOp.getDst()},
            *type, {1, 1, 1, 1}, isA5 ? false : requireExplicitTmp, "trems");
      })
      .Case<pto::TSelOp>([&](auto typedOp) -> LogicalResult {
        if (typedOp.getTmp()) {
          return success();
        }
        bool isA5 =
            pto::getTargetArch(op) == pto::PTOArch::A5;
        auto type = isA5 ? makeA5PlaceholderTmpType(
                               ctx, typedOp.getDst(),
                               IntegerType::get(ctx, 32))
                         : makeVecTmpType(ctx, {1, 16},
                                          IntegerType::get(ctx, 32), {1, 16});
        if (failed(type)) {
          return typedOp.emitOpError(
              "requires static tile_buf dst to materialize implicit tsel tmp");
        }
        return replaceFixedDpsOpWithTmp(
            op, {typedOp.getMask(), typedOp.getSrc0(), typedOp.getSrc1(),
                 Value(), typedOp.getDst()},
            *type, {1, 1, 1, 1, 1}, isA5 ? false : requireExplicitTmp, "tsel");
      })
      .Case<pto::TSelSOp>([&](auto typedOp) -> LogicalResult {
        if (typedOp.getTmp()) {
          return success();
        }
        bool isA5 =
            pto::getTargetArch(op) == pto::PTOArch::A5;
        auto type = isA5 ? makeA5PlaceholderTmpType(ctx, typedOp.getSrc())
                         : makeRowsTmpType(ctx, typedOp.getSrc(), 1);
        if (failed(type)) {
          return typedOp.emitOpError(
              "requires static tile_buf src to materialize implicit tsels tmp");
        }
        return replaceFixedDpsOpWithTmp(
            op, {typedOp.getMask(), typedOp.getSrc(), Value(),
                 typedOp.getScalar(), typedOp.getDst()},
            *type, {1, 1, 1, 1, 1}, isA5 ? false : requireExplicitTmp, "tsels");
      })
      .Case<pto::TTransOp>([&](auto typedOp) -> LogicalResult {
        if (typedOp.getTmp()) {
          return success();
        }
        bool isA5 =
            pto::getTargetArch(op) == pto::PTOArch::A5;
        auto srcTy = dyn_cast<pto::TileBufType>(typedOp.getSrc().getType());
        auto dstTy = dyn_cast<pto::TileBufType>(typedOp.getDst().getType());
        auto srcShape = getShapeVec(typedOp.getSrc().getType());
        auto dstShape = getShapeVec(typedOp.getDst().getType());
        if (!srcTy || !dstTy || srcShape.size() != mlir::pto::kValue2 || dstShape.size() != mlir::pto::kValue2 ||
            hasDynamicDim(srcShape) || hasDynamicDim(dstShape)) {
          return typedOp.emitOpError(
              "requires static tile_buf src to materialize implicit ttrans tmp");
        }
        auto elemBytes = getElemBytes(srcTy.getElementType());
        if (!elemBytes) {
          return typedOp.emitOpError("failed to infer ttrans element size");
        }
        int64_t rowStride = *elemBytes == 1 ? 32 : 16;
        int64_t elemPerBlock = 32 / *elemBytes;
        bool usesTmp = dstShape[1] % rowStride == 0 &&
                       srcShape[1] % elemPerBlock == 0 &&
                       srcShape[1] / elemPerBlock <= 255;
        FailureOr<pto::TileBufType> type =
            isA5 ? makeA5PlaceholderTmpType(ctx, typedOp.getSrc())
                 : makeSameShapeTmpType(ctx, typedOp.getSrc());
        if (!isA5 && !usesTmp) {
          type = makeVecTmpType(ctx, {1, elemPerBlock},
                                srcTy.getElementType(), {1, elemPerBlock});
        }
        if (failed(type)) {
          return typedOp.emitOpError("failed to build implicit ttrans tmp");
        }
        return replaceFixedDpsOpWithTmp(
            op, {typedOp.getSrc(), Value(), typedOp.getDst()}, *type,
            {1, 1, 1}, isA5 ? false : requireExplicitTmp, "ttrans");
      })
      .Default([](Operation *) { return success(); });
}

static bool tcvtNeedsTmp(pto::TCvtOp op) {
  if (pto::getTargetArch(op.getOperation()) == pto::PTOArch::A5 ||
      op.getSatMode() != pto::SaturationMode::OFF) {
    return false;
  }
  auto srcTy = dyn_cast<pto::TileBufType>(op.getSrc().getType());
  auto dstTy = dyn_cast<pto::TileBufType>(op.getDst().getType());
  if (!srcTy || !dstTy) {
    return false;
  }
  Type srcElem = srcTy.getElementType();
  Type dstElem = dstTy.getElementType();
  return (srcElem.isF32() && dstElem.isInteger(mlir::pto::kValue16)) ||
         (srcElem.isF16() &&
          (dstElem.isInteger(mlir::pto::kValue16) || dstElem.isInteger(8)));
}

static FailureOr<pto::TileBufType> makeTCvtTmpType(MLIRContext *ctx,
                                                   pto::TCvtOp op) {
  auto srcShape = getShapeVec(op.getSrc().getType());
  auto dstValid = getValidShapeVec(op.getDst().getType());
  auto srcTy = dyn_cast<pto::TileBufType>(op.getSrc().getType());
  auto dstTy = dyn_cast<pto::TileBufType>(op.getDst().getType());
  if (!srcTy || !dstTy || srcShape.size() != mlir::pto::kValue2 || dstValid.size() != mlir::pto::kValue2 ||
      hasDynamicDim(srcShape) || hasDynamicDim(dstValid)) {
    return failure();
  }
  int64_t rows = dstValid[0], cols = dstValid[1];
  int64_t bytes = 0;
  if (rows > 0 && cols > 0 && srcTy.getElementType().isF32()) {
    int64_t head = 4 * 64 * std::min<int64_t>(cols / 64, 255);
    int64_t remainder = cols % 64;
    int64_t tail = remainder == 0
                       ? 0
                       : 32 * ((std::min<int64_t>(rows, 255) - 1) *
                                   (srcShape[1] / 8) +
                               ceilDiv(remainder, 8));
    bytes = std::max(head, tail);
  } else if (cols > 0 && srcTy.getElementType().isF16()) {
    int64_t width = std::min<int64_t>(cols, 64);
    int64_t halfToI16 = 32 * ceilDiv(width, 8);
    int64_t halfToI8 = std::max(halfToI16, 128 + 32 * ceilDiv(width, 16));
    bytes = dstTy.getElementType().isInteger(mlir::pto::kValue8) ? halfToI8 : halfToI16;
  }
  int64_t allocatedBytes = std::max<int64_t>(32, ceilDiv(bytes, 32) * 32);
  return makeVecTmpType(ctx, {1, allocatedBytes}, IntegerType::get(ctx, mlir::pto::kValue8),
                        {1, allocatedBytes});
}

static LogicalResult materializeTCvtTmp(pto::TCvtOp op,
                                        bool requireExplicitTmp,
                                        MLIRContext *ctx) {
  if (op.getTmp() || !tcvtNeedsTmp(op)) {
    return success();
  }
  if (requireExplicitTmp) {
    return op.emitOpError(
        "requires explicit tmp for non-saturating narrowing tcvt when PlanMemory is skipped");
  }
  auto type = makeTCvtTmpType(ctx, op);
  if (failed(type)) {
    return op.emitOpError(
        "requires static tile_buf shapes to materialize implicit tcvt tmp");
  }
  return replaceFixedDpsOpWithTmp(op.getOperation(),
                                  {op.getSrc(), Value(), op.getDst()}, *type,
                                  {1, 1, 1}, requireExplicitTmp, "tcvt");
}

static LogicalResult materializeTMrgSortTmp(pto::TMrgSortOp op,
                                            bool requireExplicitTmp,
                                            MLIRContext *ctx) {
  if (!op.isFormat2WithoutTmp()) {
    return success();
  }
  if (requireExplicitTmp) {
    return op.emitOpError(
        "requires explicit tmp for tmrgsort format2 when PlanMemory is skipped");
  }
  int64_t totalCols = 0;
  Type elementType;
  SmallVector<Value> operands;
  for (Value src : op.getSrcs()) {
    auto srcTy = dyn_cast<pto::TileBufType>(src.getType());
    auto shape = getShapeVec(src.getType());
    if (!srcTy || shape.size() != mlir::pto::kValue2 || hasDynamicDim(shape)) {
      return op.emitOpError(
          "requires static rank-2 tile_buf srcs to materialize tmrgsort tmp");
    }
    if (!elementType) {
      elementType = srcTy.getElementType();
    }
    totalCols += shape[1];
    operands.push_back(src);
  }
  if (!elementType || totalCols <= 0) {
    return op.emitOpError("failed to infer tmrgsort format2 tmp type");
  }
  // The verifier requires tmp.cols >= dst.cols as well as tmp.cols >=
  // sum(src.cols), so size the scratch by the wider of the two.
  int64_t dstCols = 0;
  for (Value dst : op.getDsts()) {
    auto dstShape = getShapeVec(dst.getType());
    if (dstShape.size() == mlir::pto::kValue2 && dstShape[1] != ShapedType::kDynamic) {
      dstCols = std::max(dstCols, dstShape[1]);
    }
  }
  int64_t tmpCols = std::max(totalCols, dstCols);
  pto::TileBufType tmpType =
      makeVecTmpType(ctx, {1, tmpCols}, elementType, {1, tmpCols});
  OpBuilder builder(op);
  FailureOr<Value> tmp = createAllocTmp(builder, op.getLoc(), tmpType);
  if (failed(tmp)) {
    return failure();
  }
  SmallVector<Value> finalOperands;
  finalOperands.append(operands.begin(), operands.end());
  finalOperands.append(op.getDsts().begin(), op.getDsts().end());
  finalOperands.push_back(*tmp);
  finalOperands.push_back(op.getExcuted());
  rebuildWithOperands(op.getOperation(), finalOperands,
                      ArrayRef<int32_t>{static_cast<int32_t>(op.getSrcs().size()),
                                        0, 1, 1, 1});
  return success();
}

struct PTOMaterializeImplicitTmpPass
    : public PassWrapper<PTOMaterializeImplicitTmpPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOMaterializeImplicitTmpPass)

  PTOMaterializeImplicitTmpPass() = default;
  explicit PTOMaterializeImplicitTmpPass(bool requireExplicitTmp)
      : requireExplicitTmp(requireExplicitTmp) {}

  StringRef getArgument() const final { return "pto-materialize-implicit-tmp"; }
  StringRef getDescription() const final {
    return "Materialize implicit tmp tiles for PTO ops before memplan";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();
    bool failed = false;

    SmallVector<pto::TCIOp> tciOps;
    func.walk([&](pto::TCIOp op) {
      if (!op.getTmp()) {
        tciOps.push_back(op);
      }
    });

    for (pto::TCIOp op : tciOps) {
      if (pto::getTargetArch(op.getOperation()) == pto::PTOArch::A5) {
        continue;
      }

      if (requireExplicitTmp) {
        op.emitOpError("requires explicit tmp when PlanMemory is skipped");
        failed = true;
        continue;
      }

      OpBuilder builder(op);
      Location loc = op.getLoc();
      auto tmpType = makeTCITmpType(ctx, getTCIDstBitWidth(op));
      Value tmp =
          builder.create<pto::AllocTileOp>(loc, tmpType, Value(), Value(),
                                           Value())
              .getResult();

      auto newOp = builder.create<pto::TCIOp>(
          loc, TypeRange{}, op.getS(), tmp, op.getDst(),
          op.getDescendingAttr());
      for (NamedAttribute attr : op->getAttrs()) {
        if (attr.getName() == "operandSegmentSizes") {
          continue;
        }
        newOp->setAttr(attr.getName(), attr.getValue());
      }
      op.erase();
    }

    SmallVector<Operation *> rowExpandOps;
    func.walk([&](Operation *op) {
      if (isa<pto::TRowExpandAddOp, pto::TRowExpandSubOp,
              pto::TRowExpandMulOp, pto::TRowExpandDivOp,
              pto::TRowExpandMaxOp, pto::TRowExpandMinOp>(op)) {
        rowExpandOps.push_back(op);
      }
    });

    for (Operation *op : rowExpandOps) {
      LogicalResult result =
          llvm::TypeSwitch<Operation *, LogicalResult>(op)
              .Case<pto::TRowExpandAddOp, pto::TRowExpandSubOp,
                    pto::TRowExpandMulOp, pto::TRowExpandDivOp,
                    pto::TRowExpandMaxOp, pto::TRowExpandMinOp>(
                  [&](auto typedOp) {
                    return materializeTRowExpandTmp(typedOp, requireExplicitTmp,
                                                    ctx);
                  })
              .Default([](Operation *) { return success(); });
      if (mlir::failed(result)) {
        failed = true;
      }
    }

    SmallVector<Operation *> optionalTmpOps;
    func.walk([&](Operation *op) {
      if (isa<pto::TColSumOp, pto::TQuantOp, pto::TPowOp,
              pto::TPowSOp, pto::TSort32Op, pto::TXorOp,
              pto::TXorSOp, pto::TCvtOp, pto::TMrgSortOp>(op)) {
        optionalTmpOps.push_back(op);
      }
    });

    for (Operation *op : optionalTmpOps) {
      LogicalResult result =
          llvm::TypeSwitch<Operation *, LogicalResult>(op)
              .Case<pto::TColSumOp>([&](auto typedOp) {
                return replaceTColSumWithTmp(typedOp, requireExplicitTmp, ctx);
              })
              .Case<pto::TQuantOp>([&](auto typedOp) {
                return replaceTQuantWithTmp(typedOp, requireExplicitTmp, ctx);
              })
              .Case<pto::TPowOp>([&](auto typedOp) {
                return replaceTPowWithTmp(typedOp, requireExplicitTmp, ctx);
              })
              .Case<pto::TPowSOp>([&](auto typedOp) {
                return replaceTPowSWithTmp(typedOp, requireExplicitTmp, ctx);
              })
              .Case<pto::TSort32Op>([&](auto typedOp) {
                return replaceTSort32WithTmp(typedOp, requireExplicitTmp, ctx);
              })
              .Case<pto::TXorOp>([&](auto typedOp) {
                return replaceTXorWithTmp(typedOp, requireExplicitTmp, ctx);
              })
              .Case<pto::TXorSOp>([&](auto typedOp) {
                return replaceTXorSWithTmp(typedOp, requireExplicitTmp, ctx);
              })
              .Case<pto::TCvtOp>([&](auto typedOp) {
                return materializeTCvtTmp(typedOp, requireExplicitTmp, ctx);
              })
              .Case<pto::TMrgSortOp>([&](auto typedOp) {
                return materializeTMrgSortTmp(typedOp, requireExplicitTmp, ctx);
              })
              .Default([](Operation *) { return success(); });
      if (mlir::failed(result)) {
        failed = true;
      }
    }

    SmallVector<Operation *> rowReductionOps;
    func.walk([&](Operation *op) {
      if (isa<pto::TRowMaxOp, pto::TRowMinOp, pto::TRowSumOp,
              pto::TRowProdOp, pto::TColArgMaxOp, pto::TColArgMinOp,
              pto::TRowArgMaxOp, pto::TRowArgMinOp>(op)) {
        rowReductionOps.push_back(op);
      }
    });

    for (Operation *op : rowReductionOps) {
      LogicalResult result =
          llvm::TypeSwitch<Operation *, LogicalResult>(op)
              .Case<pto::TRowMaxOp, pto::TRowMinOp, pto::TRowSumOp,
                    pto::TRowProdOp, pto::TColArgMaxOp, pto::TColArgMinOp,
                    pto::TRowArgMaxOp, pto::TRowArgMinOp>([&](auto typedOp) {
                return replaceRowReductionWithTmp(typedOp, requireExplicitTmp,
                                                  ctx);
              })
              .Default([](Operation *) { return success(); });
      if (mlir::failed(result)) {
        failed = true;
      }
    }

    SmallVector<Operation *> mandatoryTmpOps;
    func.walk([&](Operation *op) {
      if (isa<pto::TPReluOp, pto::TRemOp, pto::TRemSOp, pto::TSelOp,
              pto::TSelSOp, pto::TTransOp>(op)) {
        mandatoryTmpOps.push_back(op);
      }
    });
    for (Operation *op : mandatoryTmpOps) {
      if (mlir::failed(
              materializeFixedMandatoryTmp(op, requireExplicitTmp, ctx))) {
        failed = true;
      }
    }

    if (failed) {
      signalPassFailure();
    }
  }

private:
  bool requireExplicitTmp = false;
};

} // namespace

std::unique_ptr<Pass>
mlir::pto::createPTOMaterializeImplicitTmpPass(bool requireExplicitTmp) {
  return std::make_unique<PTOMaterializeImplicitTmpPass>(requireExplicitTmp);
}
