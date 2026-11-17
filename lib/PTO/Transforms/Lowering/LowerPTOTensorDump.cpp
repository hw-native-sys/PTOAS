// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- LowerPTOTensorDump.cpp - lower pto.tdump ---------------------------===//
//===----------------------------------------------------------------------===//
//
// Lowers pto.tdump (VPTO-only persistent tensor dump) to a 64-byte
// little-endian metadata header written via pto.stg, followed by a
// pto.mte_ub_gm copy of the raw tile data. Runs on every architecture
// on the VPTO backend before architecture-specific tile lowering.
//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/DirectDmaUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_LOWERPTOTENSORDUMP
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

// Magic "PTD0": stored little-endian, so the on-disk byte order is
// 50 54 44 30.
constexpr uint32_t kDumpMagic = 0x30445450;
constexpr int64_t kDumpHeaderBytes = 64;
constexpr int64_t kDumpHeaderWords = 16;

using DumpViewInfo = pto::dma::DmaViewInfo;

static unsigned dumpElemSize(Type elem)
{
    const bool is2Byte = elem.isF16() || elem.isBF16() || elem.isInteger(16);
    if (is2Byte) {
        return mlir::pto::kValue2;
    }
    const bool is4Byte = elem.isF32() || elem.isInteger(32);
    if (is4Byte) {
        return mlir::pto::kValue4;
    }
    if (elem.isInteger(mlir::pto::kValue8)) {
        return 1;
    }
    if (elem.isInteger(mlir::pto::kValue64)) {
        return mlir::pto::kValue8;
    }
    return 0;
}

static uint32_t dumpElemTypeCode(Type elem)
{
    if (elem.isF16()) {
        return 1;
    }
    if (elem.isBF16()) {
        return mlir::pto::kValue2;
    }
    if (elem.isF32()) {
        return mlir::pto::kValue3;
    }
    if (elem.isInteger(mlir::pto::kValue8)) {
        return mlir::pto::kValue4;
    }
    if (elem.isInteger(mlir::pto::kValue16)) {
        return mlir::pto::kValue5;
    }
    if (elem.isInteger(mlir::pto::kValue32)) {
        return mlir::pto::kValue6;
    }
    if (elem.isInteger(mlir::pto::kValue64)) {
        return mlir::pto::kValue7;
    }
    return 0;
}

static Value indexConst(int64_t val, Location loc, OpBuilder &b) {
  return b.create<arith::ConstantOp>(loc, b.getIndexAttr(val));
}

static FailureOr<DumpViewInfo> extractDumpViewInfo(pto::TensorDumpOp op) {
  return pto::dma::extractDirectDmaViewInfo(
      op.getDst(), op.getOperation(),
      "expects the innermost tensor_view stride to be the constant 1; "
      "a strided or transposed destination is not supported",
      "expects every partition_view in the chain to have "
      "the rank of the one feeding pto.tdump",
      "expects dst to be a pto.partition_view (a chain of "
      "nested partition_view ops is allowed)",
      "expects dst to trace back to a pto.make_tensor_view");
}

// Element offset of the destination view when every offset and stride is a
// known constant; nullopt as soon as one of them is dynamic.
static FailureOr<std::optional<int64_t>>
staticElemOffset(pto::TensorDumpOp op, const DumpViewInfo &info) {
  int64_t total = 0;
  for (size_t i = 0; i < info.offsets.size() && i < info.strides.size(); ++i) {
    APInt offset;
    APInt stride;
    const bool offsetIsConst =
        matchPattern(info.offsets[i], m_ConstantInt(&offset));
    const bool strideIsConst =
        matchPattern(info.strides[i], m_ConstantInt(&stride));
    if (!offsetIsConst || !strideIsConst) {
      return std::optional<int64_t>{};
    }
    const bool fitsI64 = offset.isSignedIntN(64) && stride.isSignedIntN(64);
    if (!fitsI64) {
      return op.emitError("expects static destination offsets and strides "
                          "to fit in signed 64 bits");
    }
    int64_t term = 0;
    const bool productOverflows =
        llvm::MulOverflow(offset.getSExtValue(), stride.getSExtValue(), term);
    if (productOverflows) {
      return op.emitError("static destination element offset overflows "
                          "signed 64 bits");
    }
    const bool sumOverflows = llvm::AddOverflow(total, term, total);
    if (sumOverflows) {
      return op.emitError("static destination element offset overflows "
                          "signed 64 bits");
    }
  }
  return std::optional<int64_t>{total};
}

// The 64-byte header is written with 32-bit pto.stg stores, so the destination
// address must be 4-byte aligned. Alignment of the base pointer is the caller's
// contract; view offsets must be statically aligned or provably preserve it.
static LogicalResult verifyKnownDumpAlignment(pto::TensorDumpOp op, int64_t elemOffset, unsigned elemSize)
{
    int64_t byteOffset = 0;
    if (llvm::MulOverflow(elemOffset, static_cast<int64_t>(elemSize), byteOffset) != 0) {
        return op.emitError("static destination byte offset overflows signed 64 bits");
    }
    if (byteOffset % mlir::pto::kValue4 != 0) {
        return op.emitError(
            "expects a 4-byte aligned destination address: the dump header is "
            "written with 32-bit stores");
    }
    return success();
}

static LogicalResult verifyDynamicDumpAlignment(pto::TensorDumpOp op, const DumpViewInfo& info, unsigned elemSize)
{
    if (elemSize % mlir::pto::kValue4 == 0) {
        return success();
    }
    int64_t byteOffsetModulo4 = 0;
    for (size_t index = 0; index < info.offsets.size() && index < info.strides.size(); ++index) {
        APInt offset;
        APInt stride;
        const bool offsetIsConst = matchPattern(info.offsets[index], m_ConstantInt(&offset));
        const bool strideIsConst = matchPattern(info.strides[index], m_ConstantInt(&stride));
        const bool zeroTerm = (offsetIsConst && offset.isZero()) || (strideIsConst && stride.isZero());
        if (zeroTerm) {
            continue;
        }
        if (!strideIsConst || !stride.isSignedIntN(mlir::pto::kValue64)) {
            return op.emitError("expects destination strides to prove 4-byte header alignment");
        }
        if (!offsetIsConst) {
            const int64_t strideModulo4 = stride.getSExtValue() % 4;
            const bool alignedStride = (strideModulo4 * static_cast<int64_t>(elemSize)) % 4 == 0;
            if (!alignedStride) {
                return op.emitError(
                    "expects dynamic destination offsets to preserve 4-byte header "
                    "alignment");
            }
            continue;
        }
        if (!offset.isSignedIntN(mlir::pto::kValue64)) {
            return op.emitError(
                "expects static destination offsets and strides to fit in signed "
                "64 bits");
        }
        const int64_t offsetModulo4 = offset.getSExtValue() % 4;
        const int64_t strideModulo4 = stride.getSExtValue() % 4;
        byteOffsetModulo4 = (byteOffsetModulo4 + offsetModulo4 * strideModulo4 * elemSize) % mlir::pto::kValue4;
    }
    if (byteOffsetModulo4 != 0) {
        return op.emitError() << "expects a 4-byte aligned destination address: "
                              << "the dump header is written with 32-bit stores "
                              << "the destination view offset is not statically "
                              << "known to preserve 4-byte alignment";
    }
    return success();
}

static LogicalResult verifyDumpAlignment(pto::TensorDumpOp op,
                                         const DumpViewInfo &info,
                                         unsigned elemSize) {
  auto elemOffset = staticElemOffset(op, info);
  if (failed(elemOffset)) {
    return failure();
  }
  if (*elemOffset) {
    return verifyKnownDumpAlignment(op, **elemOffset, elemSize);
  }
  return verifyDynamicDumpAlignment(op, info, elemSize);
}

static LogicalResult verifyDumpDestinationPitch(pto::TensorDumpOp op,
                                               const DumpViewInfo &info) {
  const bool rankTooSmall = info.sizes.size() < 2 || info.strides.size() < 2;
  if (rankTooSmall) {
    return op.emitError("expects a rank-2 destination view");
  }
  const Value widthValue = info.sizes.back();
  const Value rowStrideValue = info.strides[info.strides.size() - 2];
  APInt width;
  APInt rowStride;
  const bool widthIsConst = matchPattern(widthValue, m_ConstantInt(&width));
  const bool rowStrideIsConst =
      matchPattern(rowStrideValue, m_ConstantInt(&rowStride));
  const bool equalStaticValues =
      widthIsConst && rowStrideIsConst && width.isSignedIntN(64) &&
      rowStride.isSignedIntN(64) &&
      width.getSExtValue() == rowStride.getSExtValue();
  const bool sameDynamicValue =
      !widthIsConst && !rowStrideIsConst && widthValue == rowStrideValue;
  if (!equalStaticValues && !sameDynamicValue) {
    return op.emitError(
        "expects destination row stride to equal its row width; "
        "padded or unproven dynamic row pitches are unsupported");
  }
  return success();
}

static LogicalResult verifyDumpDestinationSizes(pto::TensorDumpOp op,
                                               const DumpViewInfo &info) {
  const auto validShape =
      cast<pto::TileBufType>(op.getSrc().getType()).getValidShape();
  const size_t leading = info.sizes.size() - validShape.size();
  for (auto [index, sizeValue] : llvm::enumerate(info.sizes)) {
    const int64_t validDim = index < leading ? 1 : validShape[index - leading];
    if (validDim == ShapedType::kDynamic) {
      continue;
    }
    APInt size;
    const bool sizeIsConst =
        matchPattern(sizeValue, m_ConstantInt(&size));
    const bool matchesValidDim = sizeIsConst && size.isSignedIntN(64) &&
                                 size.getSExtValue() == validDim;
    if (!matchesValidDim) {
      if (index < leading) {
        return op.emitError(
            "expects leading destination size operands to be the constant 1");
      }
      return op.emitError(
          "expects destination size operands to match static src valid_shape "
          "dimensions");
    }
  }
  return success();
}

static Value computeByteOffset(Location loc, OpBuilder &b,
                               const DumpViewInfo &info, unsigned elemSize) {
  Value total = indexConst(0, loc, b);
  for (size_t i = 0; i < info.offsets.size() && i < info.strides.size(); ++i) {
    APInt constOff;
    const bool offsetIsZero = matchPattern(info.offsets[i],
                                           m_ConstantInt(&constOff)) &&
                              constOff.isZero();
    if (offsetIsZero) {
      continue;
    }
    Value dimOff = b.create<arith::MulIOp>(loc, info.offsets[i],
                                           info.strides[i]).getResult();
    total = b.create<arith::AddIOp>(loc, total, dimOff).getResult();
  }
  if (elemSize > 1) {
    total = b.create<arith::MulIOp>(loc, total, indexConst(elemSize, loc, b))
                .getResult();
  }
  return total;
}

static Value offsetPtrByBytes(Location loc, OpBuilder &b, Value ptr,
                              Value byteOff) {
  APInt constOff;
  const bool offsetIsZero = matchPattern(byteOff, m_ConstantInt(&constOff)) &&
                            constOff.isZero();
  if (offsetIsZero) {
    return ptr;
  }
  auto origTy = cast<pto::PtrType>(ptr.getType());
  auto byteTy =
      pto::PtrType::get(b.getContext(), b.getI8Type(), origTy.getMemorySpace());
  Value bytePtr = b.create<pto::CastPtrOp>(loc, byteTy, ptr);
  Value off = byteOff;
  if (!off.getType().isIndex()) {
    off = b.create<arith::IndexCastOp>(loc, b.getIndexType(), off).getResult();
  }
  return b.create<pto::CastPtrOp>(
      loc, origTy, b.create<pto::AddPtrOp>(loc, byteTy, bytePtr, off));
}

static void emitHeaderWord(Location loc, OpBuilder &b, Value i32Ptr,
                           int64_t wordIdx, uint32_t val) {
  auto attr = b.getIntegerAttr(b.getI32Type(), APInt(32, val));
  Value v = b.create<arith::ConstantOp>(loc, b.getI32Type(), attr);
  b.create<pto::PTOStgOp>(loc, i32Ptr, indexConst(wordIdx, loc, b), v,
                          /*l1cache=*/nullptr, /*l2cache=*/nullptr);
}

static void emitDumpHeader(
    Location loc, OpBuilder& b, Value i32Ptr, Type elem, unsigned elemSize, ArrayRef<int64_t> shape,
    ArrayRef<int64_t> valid)
{
    auto section =
        b.create<pto::SectionSimtOp>(loc, b.getI32IntegerAttr(1), b.getI32IntegerAttr(1), b.getI32IntegerAttr(1));
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&section.getBody().emplaceBlock());
    emitHeaderWord(loc, b, i32Ptr, 0, kDumpMagic);
    emitHeaderWord(loc, b, i32Ptr, 1, 1);
    emitHeaderWord(loc, b, i32Ptr, mlir::pto::kValue2, elemSize);
    emitHeaderWord(loc, b, i32Ptr, mlir::pto::kValue3, dumpElemTypeCode(elem));
    emitHeaderWord(loc, b, i32Ptr, mlir::pto::kValue4, static_cast<uint32_t>(shape.size()));
    emitHeaderWord(loc, b, i32Ptr, mlir::pto::kValue5, static_cast<uint32_t>(shape[0]));
    emitHeaderWord(loc, b, i32Ptr, mlir::pto::kValue6, static_cast<uint32_t>(shape[1]));
    auto validWord = [](int64_t dim) -> uint32_t {
        return dim == ShapedType::kDynamic ? UINT32_MAX : static_cast<uint32_t>(dim);
    };
    emitHeaderWord(loc, b, i32Ptr, mlir::pto::kValue7, valid.size() > 0 ? validWord(valid[0]) : 0);
    emitHeaderWord(loc, b, i32Ptr, mlir::pto::kValue8, valid.size() > 1 ? validWord(valid[1]) : 0);
    emitHeaderWord(loc, b, i32Ptr, mlir::pto::kValue9, kDumpHeaderBytes);
    for (int i = 10; i < kDumpHeaderWords; ++i) {
        emitHeaderWord(loc, b, i32Ptr, i, 0);
    }
}

static Value resolveUbPtr(Location loc, OpBuilder &b, pto::TensorDumpOp op,
                          Type elem, bool isA2A3) {
  auto vecPtrTy = [&b, elem]() {
    return pto::PtrType::get(b.getContext(), elem,
                             pto::AddressSpaceAttr::get(
                                 b.getContext(), pto::AddressSpace::VEC));
  };
  if (isA2A3) {
    if (auto alloc = op.getSrc().getDefiningOp<pto::AllocTileOp>()) {
      if (alloc.getAddr()) {
        return b.create<pto::CastPtrOp>(loc, vecPtrTy(), alloc.getAddr());
      }
    }
  }
  return b.create<pto::TileBufAddrOp>(loc, vecPtrTy(), op.getSrc()).getDst();
}

static LogicalResult emitDataCopy(pto::TensorDumpOp op, OpBuilder &b,
                                  Value ubPtr, Value gmPtr,
                                  const DumpViewInfo &info, Type elem,
                                  ArrayRef<int64_t> tileShape) {
  if (failed(pto::dma::emitMteUbGm(op.getLoc(), b, ubPtr, gmPtr, info,
                                  dumpElemSize(elem), tileShape))) {
    return op.emitError(
        "expects the destination partition view to be at least rank 2");
  }
  return success();
}

static LogicalResult lowerTensorDump(pto::TensorDumpOp op, OpBuilder &b,
                                     bool isA2A3) {
  Location loc = op.getLoc();
  auto tileTy = cast<pto::TileBufType>(op.getSrc().getType());
  Type elem = tileTy.getElementType();
  unsigned elemSize = dumpElemSize(elem);
  const bool dumpable =
      elemSize != 0 && tileTy.getShape().size() == mlir::pto::kValue2;
  if (!dumpable) {
    return op.emitError("expects a rank-2 tile with a dumpable element type "
                        "(f16/bf16/f32/i8/i16/i32/i64)");
  }

  auto viewInfo = extractDumpViewInfo(op);
  if (failed(viewInfo)) {
    // extractDumpViewInfo reported the reason already.
    return failure();
  }

  if (failed(verifyDumpAlignment(op, *viewInfo, elemSize))) {
    return failure();
  }
  if (failed(verifyDumpDestinationPitch(op, *viewInfo))) {
    return failure();
  }
  if (failed(verifyDumpDestinationSizes(op, *viewInfo))) {
    return failure();
  }

  Value byteOff = computeByteOffset(loc, b, *viewInfo, elemSize);
  Value gmPtr = offsetPtrByBytes(loc, b, viewInfo->gmPtr, byteOff);

  auto gmSpace =
      pto::AddressSpaceAttr::get(b.getContext(), pto::AddressSpace::GM);
  auto i32PtrTy = pto::PtrType::get(b.getContext(), b.getI32Type(), gmSpace);
  Value i32Ptr = gmPtr;
  const bool needsHeaderCast = gmPtr.getType() != i32PtrTy;
  if (needsHeaderCast) {
    i32Ptr = b.create<pto::CastPtrOp>(loc, i32PtrTy, gmPtr);
  }
  emitDumpHeader(loc, b, i32Ptr, elem, elemSize, tileTy.getShape(),
                 tileTy.getValidShape());

  Value ubPtr = resolveUbPtr(loc, b, op, elem, isA2A3);
  Value dataPtr = offsetPtrByBytes(loc, b, gmPtr, indexConst(kDumpHeaderBytes, loc, b));
  if (failed(emitDataCopy(op, b, ubPtr, dataPtr, *viewInfo, elem,
                          tileTy.getShape()))) {
    return failure();
  }
  op.erase();
  return success();
}

struct LowerPTOTensorDumpPass
    : public pto::impl::LowerPTOTensorDumpBase<LowerPTOTensorDumpPass> {
  using LowerPTOTensorDumpBase::LowerPTOTensorDumpBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal()) {
      return;
    }
    ModuleOp mod = func->getParentOfType<ModuleOp>();
    if (!mod) {
      return;
    }
    auto archAttr = mod->getAttrOfType<StringAttr>("pto.target_arch");
    const bool isA2A3 =
        archAttr && (archAttr.getValue() == "a2" || archAttr.getValue() == "a3");
    OpBuilder builder(&getContext());
    SmallVector<pto::TensorDumpOp> dumps;
    func.walk([&dumps](pto::TensorDumpOp op) { dumps.push_back(op); });
    for (auto op : dumps) {
      builder.setInsertionPoint(op);
      if (failed(lowerTensorDump(op, builder, isA2A3))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

namespace mlir::pto {

std::unique_ptr<Pass> createLowerPTOTensorDumpPass() {
  return std::make_unique<LowerPTOTensorDumpPass>();
}

} // namespace mlir::pto
