// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "Utils.h"

#include <limits>

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "pto-utils"

namespace mlir {
namespace pto {

static constexpr llvm::StringLiteral kFrontendPipeIdAttrName =
    "__pto.frontend_id";

FailureOr<bool> hasTFillPadExpandedPhysicalShape(TFillPadOp op) {
  auto srcType = dyn_cast<TileBufType>(op.getSrc().getType());
  auto dstType = dyn_cast<TileBufType>(op.getDst().getType());
  if (!srcType || !dstType || srcType.getRank() != dstType.getRank()) {
    return failure();
  }

  bool expanded = false;
  for (auto [srcDim, dstDim] :
       llvm::zip_equal(srcType.getShape(), dstType.getShape())) {
    if (srcDim == dstDim) {
      continue;
    }
    if (ShapedType::isDynamic(srcDim) || ShapedType::isDynamic(dstDim) ||
        dstDim < srcDim) {
      return failure();
    }
    expanded = true;
  }
  return expanded;
}

static Value peelTFillPadStorageAlias(Value value) {
  constexpr unsigned kMaxDepth = 32;
  for (unsigned depth = 0; value && depth < kMaxDepth; ++depth) {
    Operation *def = value.getDefiningOp();
    if (!def) {
      break;
    }
    if (auto cast = dyn_cast<UnrealizedConversionCastOp>(def)) {
      if (cast.getNumOperands() != 1 || cast.getNumResults() != 1) {
        break;
      }
      value = cast.getOperand(0);
      continue;
    }
    if (auto bitcast = dyn_cast<BitcastOp>(def)) {
      value = bitcast.getSrc();
      continue;
    }
    if (auto reshape = dyn_cast<TReshapeOp>(def)) {
      value = reshape.getSrc();
      continue;
    }
    break;
  }
  return value;
}

static bool haveSameKnownTFillPadStartAddress(Value src, Value dst) {
  src = peelTFillPadStorageAlias(src);
  dst = peelTFillPadStorageAlias(dst);
  if (src == dst) {
    return true;
  }

  auto srcAlloc = src.getDefiningOp<AllocTileOp>();
  auto dstAlloc = dst.getDefiningOp<AllocTileOp>();
  if (!srcAlloc || !dstAlloc || !srcAlloc.getAddr() || !dstAlloc.getAddr()) {
    return false;
  }

  Value srcAddr = srcAlloc.getAddr();
  Value dstAddr = dstAlloc.getAddr();
  if (srcAddr == dstAddr) {
    return true;
  }

  IntegerAttr srcConst;
  IntegerAttr dstConst;
  return matchPattern(srcAddr, m_Constant(&srcConst)) &&
         matchPattern(dstAddr, m_Constant(&dstConst)) &&
         srcConst.getValue() == dstConst.getValue();
}

FailureOr<TFillPadLoweringKind>
inferTFillPadLoweringKindAfterMemoryPlanning(TFillPadOp op) {
  FailureOr<bool> expanded = hasTFillPadExpandedPhysicalShape(op);
  if (failed(expanded)) {
    return failure();
  }

  auto srcSpace = GetBufferSpaceAttr(op.getSrc());
  auto dstSpace = GetBufferSpaceAttr(op.getDst());
  bool isVec = srcSpace && dstSpace &&
               srcSpace->getAddressSpace() == AddressSpace::VEC &&
               dstSpace->getAddressSpace() == AddressSpace::VEC;

  if (*expanded) {
    if (!isVec) {
      return failure();
    }
    return TFillPadLoweringKind::Expand;
  }
  if (isVec &&
      haveSameKnownTFillPadStartAddress(op.getSrc(), op.getDst())) {
    return TFillPadLoweringKind::InPlace;
  }
  return TFillPadLoweringKind::Normal;
}

std::optional<PhysicalSectionKind>
inferPhysicalSectionKindFromPipe(Operation *op) {
  auto pipeOp = dyn_cast_or_null<OpPipeInterface>(op);
  if (!pipeOp) {
    return std::nullopt;
  }

  switch (pipeOp.getPipe()) {
  case PIPE::PIPE_M:
  case PIPE::PIPE_MTE1:
    return PhysicalSectionKind::Cube;
  case PIPE::PIPE_V:
  case PIPE::PIPE_V2:
  case PIPE::PIPE_S:
    return PhysicalSectionKind::Vector;
  default:
    return std::nullopt;
  }
}

func::ReturnOp getAssumedUniqueReturnOp(func::FuncOp funcOp) {
  func::ReturnOp returnOp;
  for (Block &b : funcOp.getBody()) {
    if (auto candidateOp = dyn_cast<func::ReturnOp>(b.getTerminator())) {
      if (returnOp) {
        return nullptr;
      }
      returnOp = candidateOp;
    }
  }
  return returnOp;
}

Value peelUnrealized(Value value) {
  if (auto castOp = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    return castOp.getOperand(0);
  }
  return value;
}

bool isScalarFixpipeQuant(FixpipeQuant quant) {
  switch (quant) {
  case FixpipeQuant::DEQF16Scalar:
  case FixpipeQuant::REQ8Scalar:
  case FixpipeQuant::QF322B8PreScalar:
  case FixpipeQuant::QF322F16PreScalar:
  case FixpipeQuant::QF322BF16PreScalar:
  case FixpipeQuant::QS322BF16PreScalar:
  case FixpipeQuant::QF322HIF8PreScalar:
  case FixpipeQuant::QF322FP8PreScalar:
    return true;
  default:
    return false;
  }
}

bool isVectorFixpipeQuant(FixpipeQuant quant) {
  switch (quant) {
  case FixpipeQuant::DEQF16Vec:
  case FixpipeQuant::REQ8Vec:
  case FixpipeQuant::QF322B8PreVec:
  case FixpipeQuant::QS322BF16PreVec:
    return true;
  default:
    return false;
  }
}

Operation *getPipeInitDef(Value pipeHandle) {
  pipeHandle = peelUnrealized(pipeHandle);
  return pipeHandle ? pipeHandle.getDefiningOp() : nullptr;
}

AccPushEpilogueAttr getPipeInitAccPushEpilogue(Operation *initOp) {
  if (auto init = dyn_cast_or_null<InitializeL2LPipeOp>(initOp)) {
    return init.getAccPushEpilogueAttr();
  }
  if (auto init = dyn_cast_or_null<InitializeL2G2LPipeOp>(initOp)) {
    return init.getAccPushEpilogueAttr();
  }
  return {};
}

std::optional<int32_t> getFrontendPipeIdFromInit(Operation *initOp) {
  if (!initOp) {
    return std::nullopt;
  }
  if (auto attr = initOp->getAttrOfType<IntegerAttr>(kFrontendPipeIdAttrName)) {
    return static_cast<int32_t>(attr.getInt());
  }
  return std::nullopt;
}

std::optional<int32_t> getFrontendPipeIdFromHandle(Value pipeHandle) {
  return getFrontendPipeIdFromInit(getPipeInitDef(pipeHandle));
}

// New helper function to get the updated BaseMemRefType
BaseMemRefType getBaseMemRefTypeWithNewScope(BaseMemRefType type,
                                             AddressSpaceAttr targetMemScope) {
  if (auto memRefType = dyn_cast<MemRefType>(type)) {
    return MemRefType::Builder(memRefType).setMemorySpace(targetMemScope);
  } else if (auto unrankedMemRefType = dyn_cast<UnrankedMemRefType>(type)) {
    return UnrankedMemRefType::get(unrankedMemRefType.getElementType(),
                                   targetMemScope);
  }
  llvm_unreachable("Unexpected BaseMemRefType");
  return type;
}

void setBaseMemRefTypeScope(Value val, AddressSpaceAttr targetMemScope) {
  Type type = val.getType();
  if (!isa<BaseMemRefType>(type)) {
    return;
  }

  if (auto curMemScope = dyn_cast_if_present<AddressSpaceAttr>(
          dyn_cast<BaseMemRefType>(type).getMemorySpace())) {
    if (curMemScope != targetMemScope) {
      llvm::report_fatal_error("memref scope mismatch while propagating PTO address space");
    }
    return;
  }

  auto memRefType = cast<BaseMemRefType>(type);
  auto newMemRefType =
      getBaseMemRefTypeWithNewScope(memRefType, targetMemScope);
  val.setType(newMemRefType);
}


std::optional<AddressSpaceAttr> GetBufferSpaceAttr(Value operand) {
  if (auto tileTy = dyn_cast<pto::TileBufType>(operand.getType())) {
    if (auto memorySpaceAttr =
            dyn_cast_or_null<AddressSpaceAttr>(tileTy.getMemorySpace())) {
      return memorySpaceAttr;
    }
    return std::nullopt;
  }
  if (auto multiTy = dyn_cast<pto::MultiTileBufType>(operand.getType())) {
    if (auto memorySpaceAttr = dyn_cast_or_null<AddressSpaceAttr>(
            multiTy.getSlotType().getMemorySpace())) {
      return memorySpaceAttr;
    }
    return std::nullopt;
  }

  if (!llvm::isa<MemRefType>(operand.getType())) {
    return std::nullopt;
  }
  auto memRefType = cast<MemRefType>(operand.getType());
  auto memorySpace = memRefType.getMemorySpace();
  if (!memorySpace) {
    return std::nullopt;
  }
  auto memorySpaceAttr = dyn_cast<AddressSpaceAttr>(memorySpace);
  if (!memorySpaceAttr) {
    return std::nullopt;
  }
  return memorySpaceAttr;
}

static std::optional<std::pair<Value, Value>>
getPTOOperationAliasInfo(Operation *op) {
  if (auto makeViewOp = dyn_cast<pto::MakeTensorViewOp>(op)) {
    return std::make_pair(makeViewOp.getResult(), makeViewOp.getPtr());
  }
  if (auto partViewOp = dyn_cast<pto::PartitionViewOp>(op)) {
    return std::make_pair(partViewOp.getResult(), partViewOp.getSource());
  }
  if (auto addPtrOp = dyn_cast<pto::AddPtrOp>(op)) {
    return std::make_pair(addPtrOp.getResult(), addPtrOp.getPtr());
  }
  if (auto ptrToIntOp = dyn_cast<pto::PtrToIntOp>(op)) {
    return std::make_pair(ptrToIntOp.getResult(), ptrToIntOp.getPtr());
  }
  if (auto intToPtrOp = dyn_cast<pto::IntToPtrOp>(op)) {
    return std::make_pair(intToPtrOp.getResult(), intToPtrOp.getAddr());
  }
  if (auto castPtrOp = dyn_cast<pto::CastPtrOp>(op)) {
    return std::make_pair(castPtrOp.getResult(), castPtrOp.getInput());
  }
  if (auto subViewOp = dyn_cast<pto::SubViewOp>(op)) {
    return std::make_pair(subViewOp.getResult(), subViewOp.getSource());
  }
  if (auto bitcastOp = dyn_cast<pto::BitcastOp>(op)) {
    return std::make_pair(bitcastOp.getResult(), bitcastOp.getSrc());
  }
  if (auto reshapeOp = dyn_cast<pto::TReshapeOp>(op)) {
    return std::make_pair(reshapeOp.getResult(), reshapeOp.getSrc());
  }
  if (auto multiGetOp = dyn_cast<pto::MultiTileGetOp>(op)) {
    return std::make_pair(multiGetOp.getResult(), multiGetOp.getSource());
  }
  return std::nullopt;
}

static std::optional<std::pair<Value, Value>>
getMemRefOperationAliasInfo(Operation *op) {
  if (auto subViewOp = dyn_cast<memref::SubViewOp>(op)) {
    return std::make_pair(subViewOp.getResult(), subViewOp.getViewSource());
  }
  if (auto collapseShapeOp = dyn_cast<memref::CollapseShapeOp>(op)) {
    return std::make_pair(collapseShapeOp.getResult(),
                          collapseShapeOp.getViewSource());
  }
  if (auto expandShapeOp = dyn_cast<memref::ExpandShapeOp>(op)) {
    return std::make_pair(expandShapeOp.getResult(),
                          expandShapeOp.getViewSource());
  }
  if (auto viewOp = dyn_cast<memref::ViewOp>(op)) {
    return std::make_pair(viewOp.getResult(), viewOp.getViewSource());
  }
  if (auto reinterpretCastOp = dyn_cast<memref::ReinterpretCastOp>(op)) {
    return std::make_pair(reinterpretCastOp.getResult(),
                          reinterpretCastOp.getViewSource());
  }
  if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(op)) {
    return std::make_pair(reshapeOp.getResult(), reshapeOp.getViewSource());
  }
  if (auto castOp = dyn_cast<memref::CastOp>(op)) {
    return std::make_pair(castOp.getResult(), castOp.getViewSource());
  }
  if (auto extractOp = dyn_cast<memref::ExtractStridedMetadataOp>(op)) {
    return std::make_pair(extractOp.getBaseBuffer(), extractOp.getViewSource());
  }
  return std::nullopt;
}

std::optional<std::pair<Value, Value>> getOperationAliasInfo(Operation *op) {
  if (auto alias = getPTOOperationAliasInfo(op)) {
    return alias;
  }
  if (auto alias = getMemRefOperationAliasInfo(op)) {
    return alias;
  }
  if (auto extSliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
    return std::make_pair(extSliceOp.getResult(), extSliceOp.getSource());
  }
  if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
    if (castOp.getNumOperands() == 1 && castOp.getNumResults() == 1) {
      return std::make_pair(castOp.getResult(0), castOp.getOperand(0));
    }
  }
  if (auto toMemrefOp = dyn_cast<bufferization::ToMemrefOp>(op)) {
    return std::make_pair(toMemrefOp.getResult(), toMemrefOp.getOperand());
  }
  if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(op)) {
    return std::make_pair(toTensorOp.getResult(), toTensorOp.getOperand());
  }
  return std::nullopt;
}

SmallVector<std::pair<Value, Value>, kValue15>
getSemanticNoAliasPairs(Operation *op) {
  SmallVector<std::pair<Value, Value>, kValue15> pairs;
  if (auto tmov = dyn_cast<TMovOp>(op)) {
    if (classifyTMovForm(tmov.getFp()) == TMovForm::XToZz) {
      pairs.emplace_back(tmov.getSrc(), tmov.getDst());
      pairs.emplace_back(tmov.getSrc(), tmov.getFp());
      pairs.emplace_back(tmov.getFp(), tmov.getDst());
    }
    return pairs;
  }

  if (auto tquant = dyn_cast<TQuantMxOp>(op)) {
    SmallVector<Value, kValue6> tiles{tquant.getSrc(), tquant.getDst(),
                                      tquant.getExp(), tquant.getMax(),
                                      tquant.getScaling()};
    if (Value expZz = tquant.getExpZz()) {
      tiles.push_back(expZz);
    }
    for (unsigned lhs = 0; lhs < tiles.size(); ++lhs) {
      for (unsigned rhs = lhs + 1; rhs < tiles.size(); ++rhs) {
        pairs.emplace_back(tiles[lhs], tiles[rhs]);
      }
    }
  }
  return pairs;
}

namespace {

struct SemanticRange {
  Value root;
  uint64_t relativeBegin = 0;
  uint64_t bytes = 0;
  std::optional<uint64_t> absoluteBegin;
  std::optional<AddressSpace> addressSpace;
  std::optional<uint64_t> rowStrideBytes;
  std::optional<uint64_t> colStrideBytes;
  uint64_t elemBytes = 0;
};

struct StaticTileStrides {
  uint64_t rowBytes;
  uint64_t colBytes;
  uint64_t elemBytes;
};

static std::optional<uint64_t>
getRowPlusOneElementCount(ArrayRef<int64_t> shape, bool rowMajor) {
  bool hasInvalidShape = shape.size() != kValue2 ||
                         llvm::is_contained(shape, ShapedType::kDynamic);
  if (hasInvalidShape) {
    return std::nullopt;
  }
  uint64_t major = static_cast<uint64_t>(rowMajor ? shape[0] : shape[1]);
  uint64_t minor = static_cast<uint64_t>(rowMajor ? shape[1] : shape[0]);
  if (major == 0 || minor == 0) {
    return uint64_t{0};
  }
  bool sizeOverflows =
      minor == std::numeric_limits<uint64_t>::max() ||
      major - 1 > std::numeric_limits<uint64_t>::max() / (minor + 1);
  if (sizeOverflows) {
    return std::nullopt;
  }
  uint64_t elements = (major - 1) * (minor + 1);
  bool additionOverflows =
      minor > std::numeric_limits<uint64_t>::max() - elements;
  if (additionOverflows) {
    return std::nullopt;
  }
  return elements + minor;
}

static std::optional<uint64_t>
getDenseElementCount(ArrayRef<int64_t> shape) {
  uint64_t elements = 1;
  for (int64_t dim : shape) {
    if (dim < 0) {
      return std::nullopt;
    }
    if (dim == 0) {
      return uint64_t{0};
    }
    bool productOverflows =
        elements > std::numeric_limits<uint64_t>::max() /
                       static_cast<uint64_t>(dim);
    if (productOverflows) {
      return std::nullopt;
    }
    elements *= static_cast<uint64_t>(dim);
  }
  return elements;
}

static std::optional<uint64_t> getStaticTileBytes(TileBufType type) {
  unsigned elemBytes = getPTOStorageElemByteSize(type.getElementType());
  if (elemBytes == 0) {
    return std::nullopt;
  }
  ArrayRef<int64_t> shape = type.getShape();
  bool rowPlusOne = type.getCompactModeI32() ==
                    static_cast<int32_t>(CompactMode::RowPlusOne);
  bool rowMajor = type.getBLayoutValueI32() ==
                  static_cast<int32_t>(BLayout::RowMajor);
  std::optional<uint64_t> elements =
      rowPlusOne ? getRowPlusOneElementCount(shape, rowMajor)
                 : getDenseElementCount(shape);
  if (!elements ||
      *elements > std::numeric_limits<uint64_t>::max() / elemBytes) {
    return std::nullopt;
  }
  return *elements * elemBytes;
}

static std::optional<uint64_t> getConstantAddress(Value value) {
  IntegerAttr attr;
  bool isInvalid =
      !value || !matchPattern(value, m_Constant(&attr)) || attr.getInt() < 0;
  if (isInvalid) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(attr.getInt());
}

static std::optional<StaticTileStrides>
getStaticTileStrides(TileBufType type) {
  ArrayRef<int64_t> shape = type.getShape();
  unsigned elemBytes = getPTOStorageElemByteSize(type.getElementType());
  bool hasInvalidShape =
      shape.size() != kValue2 || elemBytes == 0 ||
      llvm::is_contained(shape, ShapedType::kDynamic) || shape[0] < 0 ||
      shape[1] < 0;
  if (hasInvalidShape) {
    return std::nullopt;
  }

  // Boxed layouts are not affine rank-2 row/column views. Callers preserve the
  // complete parent range for them instead of guessing an offset envelope.
  bool isBoxedLayout = type.getSLayoutValueI32() !=
                       static_cast<int32_t>(SLayout::NoneBox);
  if (isBoxedLayout) {
    return std::nullopt;
  }

  bool rowMajor = type.getBLayoutValueI32() ==
                  static_cast<int32_t>(BLayout::RowMajor);
  uint64_t rows = static_cast<uint64_t>(shape[0]);
  uint64_t cols = static_cast<uint64_t>(shape[1]);
  uint64_t rowElems = rowMajor ? cols : 1;
  uint64_t colElems = rowMajor ? 1 : rows;
  if (type.getCompactModeI32() ==
      static_cast<int32_t>(CompactMode::RowPlusOne)) {
    if (rowMajor) {
      if (cols == std::numeric_limits<uint64_t>::max()) {
        return std::nullopt;
      }
      rowElems = cols + 1;
    } else {
      if (rows == std::numeric_limits<uint64_t>::max()) {
        return std::nullopt;
      }
      colElems = rows + 1;
    }
  }
  bool strideOverflows =
      rowElems > std::numeric_limits<uint64_t>::max() / elemBytes ||
      colElems > std::numeric_limits<uint64_t>::max() / elemBytes;
  if (strideOverflows) {
    return std::nullopt;
  }
  return StaticTileStrides{rowElems * elemBytes, colElems * elemBytes,
                           elemBytes};
}

static std::optional<AddressSpace> getTileAddressSpace(TileBufType type) {
  auto attr = dyn_cast_or_null<AddressSpaceAttr>(type.getMemorySpace());
  if (!attr) {
    return std::nullopt;
  }
  return attr.getAddressSpace();
}

static std::optional<uint64_t>
getSubviewByteOffset(SubViewOp op, const SemanticRange &source) {
  bool hasInvalidRank = op.getOffsets().size() != kValue2;
  if (hasInvalidRank) {
    return std::nullopt;
  }
  IntegerAttr rowAttr;
  IntegerAttr colAttr;
  bool hasInvalidOffset =
      !matchPattern(op.getOffsets()[0], m_Constant(&rowAttr)) ||
      !matchPattern(op.getOffsets()[1], m_Constant(&colAttr)) ||
      rowAttr.getInt() < 0 || colAttr.getInt() < 0;
  if (hasInvalidOffset) {
    return std::nullopt;
  }
  if (!source.rowStrideBytes || !source.colStrideBytes) {
    return std::nullopt;
  }
  uint64_t row = static_cast<uint64_t>(rowAttr.getInt());
  uint64_t col = static_cast<uint64_t>(colAttr.getInt());
  if (row > std::numeric_limits<uint64_t>::max() /
                *source.rowStrideBytes) {
    return std::nullopt;
  }
  uint64_t bytes = row * *source.rowStrideBytes;
  if (col > std::numeric_limits<uint64_t>::max() /
                *source.colStrideBytes) {
    return std::nullopt;
  }
  uint64_t colBytes = col * *source.colStrideBytes;
  bool offsetOverflows =
      colBytes > std::numeric_limits<uint64_t>::max() - bytes;
  if (offsetOverflows) {
    return std::nullopt;
  }
  return bytes + colBytes;
}

static std::optional<uint64_t>
getSubviewByteSpan(SubViewOp op, const SemanticRange &source) {
  bool hasInvalidStride = !source.rowStrideBytes || !source.colStrideBytes ||
                          source.elemBytes == 0;
  if (hasInvalidStride) {
    return std::nullopt;
  }
  ArrayAttr sizes = op.getSizes();
  bool hasInvalidSizes = !sizes || sizes.size() != kValue2;
  if (hasInvalidSizes) {
    return std::nullopt;
  }
  int64_t rowsValue = cast<IntegerAttr>(sizes[0]).getInt();
  int64_t colsValue = cast<IntegerAttr>(sizes[1]).getInt();
  if (rowsValue < 0 || colsValue < 0) {
    return std::nullopt;
  }
  uint64_t rows = static_cast<uint64_t>(rowsValue);
  uint64_t cols = static_cast<uint64_t>(colsValue);
  if (rows == 0 || cols == 0) {
    return uint64_t{0};
  }
  bool spanOverflows =
      rows - 1 > std::numeric_limits<uint64_t>::max() /
                     *source.rowStrideBytes ||
      cols - 1 > std::numeric_limits<uint64_t>::max() /
                     *source.colStrideBytes;
  if (spanOverflows) {
    return std::nullopt;
  }
  uint64_t span = (rows - 1) * *source.rowStrideBytes;
  uint64_t colSpan = (cols - 1) * *source.colStrideBytes;
  bool columnSpanOverflows =
      colSpan > std::numeric_limits<uint64_t>::max() - span;
  if (columnSpanOverflows) {
    return std::nullopt;
  }
  span += colSpan;
  bool elementSpanOverflows =
      source.elemBytes > std::numeric_limits<uint64_t>::max() - span;
  if (elementSpanOverflows) {
    return std::nullopt;
  }
  return span + source.elemBytes;
}

static std::optional<SemanticRange> resolveSemanticRange(Value value);

static std::optional<SemanticRange> makeTileRange(Value root, TileBufType type,
                                                  uint64_t bytes,
                                                  std::optional<uint64_t> base) {
  std::optional<StaticTileStrides> strides = getStaticTileStrides(type);
  return SemanticRange{
      root, 0, bytes, base, getTileAddressSpace(type),
      strides ? std::optional<uint64_t>(strides->rowBytes) : std::nullopt,
      strides ? std::optional<uint64_t>(strides->colBytes) : std::nullopt,
      strides ? strides->elemBytes : uint64_t{0}};
}

static std::optional<SemanticRange> resolveAllocTileRange(AllocTileOp alloc) {
  auto tileType = dyn_cast<TileBufType>(alloc.getResult().getType());
  std::optional<uint64_t> bytes =
      tileType ? getStaticTileBytes(tileType) : std::nullopt;
  if (!tileType || !bytes) {
    return std::nullopt;
  }
  return makeTileRange(alloc.getResult(), tileType, *bytes,
                       getConstantAddress(alloc.getAddr()));
}

static FailureOr<std::optional<uint64_t>> getMultiTileSlotBase(
    AllocMultiTileOp alloc, IntegerAttr slotAttr, uint64_t slotBytes) {
  std::optional<uint64_t> base = getConstantAddress(alloc.getAddr());
  if (!base) {
    auto addresses =
        alloc->getAttrOfType<DenseI64ArrayAttr>(kPtoMultiBufferAddrsAttrName);
    if (!addresses) {
      return base;
    }
    bool hasInvalidAddress =
        slotAttr.getInt() >= static_cast<int64_t>(addresses.size()) ||
        addresses[slotAttr.getInt()] < 0;
    if (hasInvalidAddress) {
      return failure();
    }
    return std::optional<uint64_t>(
        static_cast<uint64_t>(addresses[slotAttr.getInt()]));
  }
  uint64_t slot = static_cast<uint64_t>(slotAttr.getInt());
  bool addressOverflows =
      slot > std::numeric_limits<uint64_t>::max() / slotBytes ||
      *base > std::numeric_limits<uint64_t>::max() - slot * slotBytes;
  if (addressOverflows) {
    return failure();
  }
  return std::optional<uint64_t>(*base + slot * slotBytes);
}

static std::optional<SemanticRange>
resolveMultiTileRange(MultiTileGetOp multiGet) {
  auto alloc = multiGet.getSource().getDefiningOp<AllocMultiTileOp>();
  auto slotType = dyn_cast<TileBufType>(multiGet.getResult().getType());
  IntegerAttr slotAttr;
  if (!alloc || !slotType ||
      !matchPattern(multiGet.getSlot(), m_Constant(&slotAttr)) ||
      slotAttr.getInt() < 0) {
    return std::nullopt;
  }
  std::optional<uint64_t> slotBytes = getStaticTileBytes(slotType);
  if (!slotBytes) {
    return std::nullopt;
  }
  FailureOr<std::optional<uint64_t>> base =
      getMultiTileSlotBase(alloc, slotAttr, *slotBytes);
  if (failed(base)) {
    return std::nullopt;
  }
  return makeTileRange(alloc.getResult(), slotType, *slotBytes, *base);
}

static std::optional<SemanticRange> resolveSubviewRange(SubViewOp subview) {
  std::optional<SemanticRange> source =
      resolveSemanticRange(subview.getSource());
  if (!source) {
    return std::nullopt;
  }
  std::optional<uint64_t> offset = getSubviewByteOffset(subview, *source);
  std::optional<uint64_t> bytes = getSubviewByteSpan(subview, *source);
  if (!offset || !bytes) {
    return source;
  }
  if (*offset > source->bytes || *bytes > source->bytes - *offset ||
      *offset >
          std::numeric_limits<uint64_t>::max() - source->relativeBegin) {
    return std::nullopt;
  }
  source->relativeBegin += *offset;
  source->bytes = *bytes;
  if (source->absoluteBegin) {
    if (*offset >
        std::numeric_limits<uint64_t>::max() - *source->absoluteBegin) {
      return std::nullopt;
    }
    *source->absoluteBegin += *offset;
  }
  return source;
}

static std::optional<SemanticRange> resolveViewRange(Value sourceValue,
                                                     Value resultValue) {
  std::optional<SemanticRange> source = resolveSemanticRange(sourceValue);
  auto viewType = dyn_cast<TileBufType>(resultValue.getType());
  if (!source || !viewType) {
    return std::nullopt;
  }
  if (auto strides = getStaticTileStrides(viewType)) {
    source->rowStrideBytes = strides->rowBytes;
    source->colStrideBytes = strides->colBytes;
    source->elemBytes = strides->elemBytes;
  } else {
    source->rowStrideBytes.reset();
    source->colStrideBytes.reset();
    source->elemBytes = 0;
  }
  return source;
}

static std::optional<SemanticRange> resolveSemanticRange(Value value) {
  if (!value) {
    return std::nullopt;
  }
  if (auto alloc = value.getDefiningOp<AllocTileOp>()) {
    return resolveAllocTileRange(alloc);
  }
  if (auto multiGet = value.getDefiningOp<MultiTileGetOp>()) {
    return resolveMultiTileRange(multiGet);
  }
  if (auto subview = value.getDefiningOp<SubViewOp>()) {
    return resolveSubviewRange(subview);
  }
  if (auto bitcast = value.getDefiningOp<BitcastOp>()) {
    return resolveViewRange(bitcast.getSrc(), bitcast.getResult());
  }
  if (auto reshape = value.getDefiningOp<TReshapeOp>()) {
    return resolveViewRange(reshape.getSrc(), reshape.getResult());
  }
  if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    bool hasSingleOperand = cast.getNumOperands() == 1;
    if (hasSingleOperand) {
      return resolveSemanticRange(cast.getOperand(0));
    }
  }
  return std::nullopt;
}

static bool rangesOverlap(const SemanticRange &lhs, const SemanticRange &rhs) {
  auto halfOpenRangesOverlap = [](uint64_t lhsBegin, uint64_t lhsBytes,
                                  uint64_t rhsBegin, uint64_t rhsBytes) {
    if (lhsBytes == 0 || rhsBytes == 0) {
      return false;
    }
    if (lhsBegin <= rhsBegin) {
      return rhsBegin - lhsBegin < lhsBytes;
    }
    return lhsBegin - rhsBegin < rhsBytes;
  };
  if (lhs.root == rhs.root) {
    return halfOpenRangesOverlap(lhs.relativeBegin, lhs.bytes,
                                 rhs.relativeBegin, rhs.bytes);
  }
  if (!lhs.absoluteBegin || !rhs.absoluteBegin ||
      lhs.addressSpace != rhs.addressSpace) {
    return false;
  }
  return halfOpenRangesOverlap(*lhs.absoluteBegin, lhs.bytes,
                               *rhs.absoluteBegin, rhs.bytes);
}

} // namespace

LogicalResult verifySemanticNoAliasRanges(func::FuncOp func) {
  LogicalResult result = success();
  func.walk([&](Operation *op) {
    if (failed(result))
      return;
    for (auto [lhs, rhs] : getSemanticNoAliasPairs(op)) {
      auto lhsRange = resolveSemanticRange(lhs);
      auto rhsRange = resolveSemanticRange(rhs);
      if (!lhsRange || !rhsRange || !rangesOverlap(*lhsRange, *rhsRange))
        continue;
      op->emitError("PlanMemory semantic no-alias violation: operand byte ranges overlap");
      result = failure();
      return;
    }
  });
  return result;
}

static Value tracebackBlockArgument(BlockArgument arg) {
  Operation *parentOp = arg.getParentRegion()->getParentOp();
  if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
    bool hasMatchingInit = arg.getArgNumber() > 0 &&
                           forOp.getInitArgs().size() > arg.getArgNumber() - 1;
    if (hasMatchingInit) {
      return forOp.getInitArgs()[arg.getArgNumber() - 1];
    }
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
    bool isBeforeArgument =
        arg.getParentRegion() == &whileOp.getBefore() &&
        arg.getArgNumber() < whileOp.getInits().size();
    if (isBeforeArgument) {
      return whileOp.getInits()[arg.getArgNumber()];
    }
    bool isAfterArgument = arg.getParentRegion() == &whileOp.getAfter();
    if (isAfterArgument) {
      auto conditionArgs = whileOp.getConditionOp().getArgs();
      bool hasMatchingCondition = arg.getArgNumber() < conditionArgs.size();
      if (hasMatchingCondition) {
        return conditionArgs[arg.getArgNumber()];
      }
    }
  }
  return {};
}

static Value tracebackCastLike(Value value, Operation *def) {
  Value result;
  if (auto op = dyn_cast<memref::CastOp>(def)) {
    result = op.getSource();
  } else if (auto op = dyn_cast<memref::CollapseShapeOp>(def)) {
    result = op.getSrc();
  } else if (auto op = dyn_cast<memref::ExpandShapeOp>(def)) {
    result = op.getSrc();
  } else if (auto op = dyn_cast<memref::MemorySpaceCastOp>(def)) {
    result = op.getSource();
  } else if (auto op = dyn_cast<memref::ReinterpretCastOp>(def)) {
    result = op.getSource();
  } else if (auto op = dyn_cast<memref::ReshapeOp>(def)) {
    result = op.getSource();
  } else if (auto op = dyn_cast<memref::TransposeOp>(def)) {
    result = op.getIn();
  } else if (auto op = dyn_cast<UnrealizedConversionCastOp>(def)) {
    result = op.getOperand(cast<OpResult>(value).getResultNumber());
  } else if (auto op = dyn_cast<scf::ForOp>(def)) {
    result = op.getInitArgs()[cast<OpResult>(value).getResultNumber()];
  } else if (auto op = dyn_cast<scf::WhileOp>(def)) {
    unsigned resultNo = cast<OpResult>(value).getResultNumber();
    if (resultNo < op.getInits().size()) {
      result = op.getInits()[resultNo];
    }
  }
  return result;
}

static Value tracebackViewLike(Operation *def) {
  if (auto op = dyn_cast<memref::ViewOp>(def)) {
    return op.getViewSource();
  }
  if (auto op = dyn_cast<memref::SubViewOp>(def)) {
    return op.getViewSource();
  }
  return {};
}

static Value tracebackImpl(Value memrefVal) {
  if (auto arg = dyn_cast<BlockArgument>(memrefVal)) {
    return tracebackBlockArgument(arg);
  }
  Operation *def = memrefVal.getDefiningOp();
  if (!def) {
    return {};
  }
  if (Value result = tracebackCastLike(memrefVal, def)) {
    return result;
  }
  return tracebackViewLike(def);
}

static bool isAllocLikeOp(Operation *op) {
  if (!op) {
    return false;
  }
  return isa<memref::AllocOp>(op) || isa<memref::AllocaOp>(op);
}

static bool isAllocLikeOp(Value val) {
  return isAllocLikeOp(val.getDefiningOp());
}

std::optional<int64_t> getStaticTotalSize(const ArrayRef<int64_t> &shapes) {
  int64_t totalSize = 1;
  for (const auto &shape : shapes) {
    if (ShapedType::isDynamic(shape)) {
      return std::nullopt;
    }
    totalSize = totalSize * shape;
  }
  return totalSize;
}

uint64_t AlignUp(uint64_t lhs, uint64_t rhs) {
  if (rhs == 0) {
    return lhs;
  }
  if (lhs % rhs != 0) {
    lhs += rhs - (lhs % rhs);
  }
  return lhs;
}

Value tracebackMemRef(Value memrefVal) {
  int loopBound = 256;
  while (memrefVal && !isAllocLikeOp(memrefVal)) {
    auto upward = tracebackImpl(memrefVal);
    if (!upward) {
      break;
    }

    memrefVal = upward;

    // avoid infinite loop
    if (loopBound-- < 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "tracebackMemRef exceeds loopBound(" << loopBound << ")!");
      break;
    }
  }

  return memrefVal;
}

std::optional<memref::AllocOp> tracebackMemRefToAlloc(Value memrefVal) {
  auto tracedValue = tracebackMemRef(memrefVal);
  return isAllocLikeOp(tracedValue)
             ? tracedValue.getDefiningOp<memref::AllocOp>()
             : std::optional<memref::AllocOp>();
}

/// trace value and judge if it is function argument
bool isFromFunctionArg(mlir::Value v) {
  return tracebackMemRef(v).getDefiningOp() == nullptr;
}

bool isLocalBuffer(std::optional<AddressSpaceAttr> memorySpaceAttr) {
  if (!memorySpaceAttr.has_value()) {
    return false;
  }

  if (memorySpaceAttr.value().getAddressSpace() == pto::AddressSpace::GM) {
    return false;
  }
  if (LocalBufferSpace.count(memorySpaceAttr.value().getAddressSpace())) {
    return true;
  }
  llvm_unreachable("Currently only support (UB | L1 | L0C) allocation");
}

static SmallVector<Value> getOpTouchBuffer(Operation *op) {
  SmallVector<Value> touchBuffer;
  touchBuffer.insert(touchBuffer.end(), op->getResults().begin(),
                     op->getResults().end());
  for (OpOperand &operand : op->getOpOperands()) {
    touchBuffer.push_back(operand.get());
  }
  return touchBuffer;
}

bool isOpTouchLocalBuffer(Operation *op) {
  auto touchBuffer = getOpTouchBuffer(op);
  for (Value buffer : touchBuffer) {
    auto bufferSpace = GetBufferSpaceAttr(buffer);
    if (isLocalBuffer(bufferSpace)) {
      return true;
    }
  }
  return false;
}

ModuleOp getTopLevelModuleOp(Operation *op) {
  ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
  while (moduleOp && moduleOp->getParentOp()) {
    moduleOp = moduleOp->getParentOfType<ModuleOp>();
  }
  return moduleOp;
}

/// Index of yielded value where is alias of targetVal.
static std::optional<int> getYieldValueIdx(Value targetVal, ValueRange yieldedValues) {
  auto it = std::find(yieldedValues.begin(), yieldedValues.end(), targetVal);
  if (it != yieldedValues.end()) {
    return it - yieldedValues.begin();
  }

  return std::nullopt;
}

LoopLikeOpInterface getParentLoop(Value val) {
  if (!val.getDefiningOp()) {
    return nullptr;
  }

  // Firstly, get parent loop
  LoopLikeOpInterface parentLoop =
      val.getDefiningOp()->getParentOfType<LoopLikeOpInterface>();
  if (!parentLoop) {
    return nullptr;
  }

  // Need to determine whether val is yielded by the loop.
  auto yieldedValues = parentLoop.getYieldedValues();
  if (yieldedValues.empty()) {
    return parentLoop;
  }

  auto idxLoopRes = getYieldValueIdx(val, yieldedValues);
  if (idxLoopRes.has_value()) {
    // The val is yielded by loop, so need to find parent of parent loop.
    auto res = parentLoop.getLoopResults().value()[*idxLoopRes];
    return getParentLoop(res);
  }

  // Need to determine whether val is yielded by if/else.
  auto parentIf = val.getDefiningOp()->getParentOfType<scf::IfOp>();
  if (!parentIf || parentIf.getResults().empty()) {
    return parentLoop;
  }

  auto thenYieldOp = parentIf.thenYield();
  auto thenYieldOpers = thenYieldOp.getOperands();

  auto idxThenYielded = getYieldValueIdx(val, thenYieldOpers);
  if (idxThenYielded.has_value()) {
    // The val is yielded by ifOp, need to find parent loop of ifOp's result
    auto res = parentIf.getResults()[*idxThenYielded];
    return getParentLoop(res);
  }

  auto elseYieldOp = parentIf.elseYield();
  auto elseYieldOpers = elseYieldOp.getOperands();
  auto idxElseYielded = getYieldValueIdx(val, elseYieldOpers);
  if (idxElseYielded.has_value()) {
    auto res = parentIf.getResults()[*idxElseYielded];
    return getParentLoop(res);
  }

  return parentLoop;
}

}
}
