// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- LoadStoreSupport.cpp - LoadStore lowering helpers --------------------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "../PTOToEmitCEmitters.h"
#include "LoadStoreInternal.h"

#include <limits>

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

// The A5 UB transfer paths read the pitch out of a different axis per layout:
// TLoadVecND2ND/TStoreVecND burst along the second-to-last dimension and take
// gStride3, while TLoadVecDN2DN/TStoreVecDN burst along the last one and take
// gStride4. The axis that stays contiguous is never read, so it keeps its
// element-unit stride.
static bool isPackedFp4TransferDim(int64_t dim, int64_t rank, bool isColMajor) {
  return isColMajor || dim + 1 < rank;
}

// The GlobalTensor shape must describe the packed extent visible to the A5
// transfer implementation as well. ND packs the last logical dimension into
// carriers, while DN packs the right-aligned matrix-row dimension. Leading
// dimensions are batch/group dimensions and must remain unchanged.
static bool isPackedFp4ShapeDim(int64_t dim, int64_t rank, bool isColMajor) {
  if (isColMajor) {
      return dim == (rank > 1 ? rank - mlir::pto::kValue2 : 0);
  }
  return dim + 1 == rank;
}

// The view dimensions whose stride carries the transfer pitch for this layout.
static SmallVector<int64_t, mlir::pto::kValue5>
getPackedFp4TransferDims(int64_t rank, bool isColMajor) {
  SmallVector<int64_t, mlir::pto::kValue5> packDims;
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (isPackedFp4TransferDim(dim, rank, isColMajor)) {
      packDims.push_back(dim);
    }
  }
  return packDims;
}

// The NZ path mixes the nibble contract with C0-block byte math, so it is
// rejected instead of being scaled on a guess.
static LogicalResult checkPackedFp4TransferLayout(Operation *anchor,
                                                  pto::TileBufType tileType) {
  const bool isNoneBox =
      getTileBufSLayoutValue(tileType.getConfigAttr()) == pto::SLayout::NoneBox;
  if (isNoneBox) {
    return success();
  }
  return anchor->emitError()
         << "packed FP4 " << anchor->getName()
         << " supports the ND and DN UB layouts only: the NZ transfer path "
            "does not follow the FP4 nibble stride contract";
}

static std::optional<pto::AddressSpace>
getTileAddressSpace(pto::TileBufType tileType) {
  auto spaceAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(
      tileType.getMemorySpace());
  if (!spaceAttr) {
    return std::nullopt;
  }
  return spaceAttr.getAddressSpace();
}

// Static descriptors rebuild the transfer view from the partition shape and
// scale the dimensions required by the selected A5 FP4 layout.
static FailureOr<Value> buildStaticPackedFp4TransferGlobalTensor(
    ConversionPatternRewriter &rewriter, Location loc, Value data,
    pto::PartitionTensorViewType partitionType, ArrayRef<int64_t> packDims,
    bool isColMajor, bool scaleShape, SmallVectorImpl<int64_t> &shape,
    SmallVectorImpl<int64_t> &strides, StringRef layoutString) {
  if (scaleShape) {
    for (int64_t dim = 0; dim < static_cast<int64_t>(shape.size()); ++dim) {
      if (isPackedFp4ShapeDim(dim, shape.size(), isColMajor)) {
        const bool shapeOverflow =
            shape[dim] > std::numeric_limits<int64_t>::max() /
                          mlir::pto::kFp4PackFactor;
        if (shapeOverflow) {
          return failure();
        }
        shape[dim] *= mlir::pto::kFp4PackFactor;
      }
    }
  }
  for (int64_t dim : packDims) {
    const bool strideOverflow =
        strides[dim] > std::numeric_limits<int64_t>::max() /
                      mlir::pto::kFp4PackFactor;
    if (strideOverflow) {
      return failure();
    }
    strides[dim] *= mlir::pto::kFp4PackFactor;
  }
  return buildGlobalTensorViewFromPointer(
      rewriter, loc, data, partitionType.getElementType(),
      shape, strides, std::nullopt, layoutString);
}

// Dynamic descriptors carry their shape and strides at runtime, so the carrier
// values are read back out of the descriptor and scaled on the way into the
// replacement one.
static FailureOr<Value> buildRuntimePackedFp4TransferGlobalTensor(
    ConversionPatternRewriter &rewriter, Location loc, Value gmTensor,
    Value data, pto::PartitionTensorViewType partitionType,
    [[maybe_unused]] ArrayRef<int64_t> packDims, bool isColMajor, bool scaleShape,
    StringRef layoutString) {
  int64_t rank = partitionType.getRank();
  SmallVector<int64_t, mlir::pto::kValue5> transferShape(
      partitionType.getShape().begin(), partitionType.getShape().end());
  for (int64_t dim = 0; dim < rank; ++dim) {
    const bool shouldScaleShape =
        scaleShape && isPackedFp4ShapeDim(dim, rank, isColMajor) &&
        transferShape[dim] != ShapedType::kDynamic;
    if (shouldScaleShape) {
      const bool shapeOverflow =
          transferShape[dim] > std::numeric_limits<int64_t>::max() /
                              mlir::pto::kFp4PackFactor;
      if (shapeOverflow) {
        return failure();
      }
      transferShape[dim] *= mlir::pto::kFp4PackFactor;
    }
  }
  SmallVector<Value, mlir::pto::kValue5> shape;
  SmallVector<Value, mlir::pto::kValue5> strides;
  for (int64_t dim = 0; dim < rank; ++dim) {
    Value logicalDim = makeViewIndexConstant(rewriter, loc, dim);
    Value shapeValue = getRuntimeGlobalTensorMetadata(
        rewriter, loc, gmTensor, logicalDim, rank, /*isStride=*/false);
    if (scaleShape && isPackedFp4ShapeDim(dim, rank, isColMajor)) {
      Value packFactor =
          makeViewIndexConstant(rewriter, loc, mlir::pto::kFp4PackFactor);
      shapeValue = rewriter
                       .create<emitc::MulOp>(loc, shapeValue.getType(),
                                             shapeValue, packFactor)
                       .getResult();
    }
    shape.push_back(shapeValue);
    Value stride = getRuntimeGlobalTensorMetadata(rewriter, loc, gmTensor,
                                                  logicalDim, rank,
                                                  /*isStride=*/true);
    if (isPackedFp4TransferDim(dim, rank, isColMajor)) {
      Value packFactor =
          makeViewIndexConstant(rewriter, loc, mlir::pto::kFp4PackFactor);
      stride = rewriter
                   .create<emitc::MulOp>(loc, stride.getType(), stride,
                                         packFactor)
                   .getResult();
    }
    strides.push_back(stride);
  }
  return buildRuntimeGlobalTensor(rewriter, loc, data,
                                  partitionType.getElementType(),
                                  transferShape, shape, strides,
                                  layoutString);
}

static FailureOr<Value> buildPackedFp4TransferDescriptor(
    ConversionPatternRewriter &rewriter, Operation *anchor, Value gmTensor,
    Value gmSource, pto::PartitionTensorViewType partitionType,
    pto::AddressSpace space) {
  auto sourceLayout = resolveLayoutForGlobalTensor(anchor, gmSource);
  const bool scaleShape =
      space == pto::AddressSpace::VEC ||
      (sourceLayout && *sourceLayout == pto::Layout::NZ);
  const bool isColMajor = sourceLayout && *sourceLayout == pto::Layout::DN;
  SmallVector<int64_t, mlir::pto::kValue5> packDims = getPackedFp4TransferDims(
      partitionType.getRank(), isColMajor);
  const bool noTransferDims = packDims.empty();
  const bool cubeWithoutTransferDims =
      noTransferDims && space != pto::AddressSpace::VEC;
  if (cubeWithoutTransferDims) {
    return gmTensor;
  }

  Location loc = anchor->getLoc();
  std::string layoutString = sourceLayout
                                 ? layoutToEmitCString(*sourceLayout)
                                 : "pto::Layout::ND";
  Value data = materializeGlobalTensorDataPointer(rewriter, loc, gmTensor,
                                                 gmSource.getType());

  SmallVector<int64_t, mlir::pto::kValue5> staticStrides;
  bool hasStaticStrides = succeeded(getStaticTensorViewStrides(
      gmSource, gmTensor, partitionType.getRank(), staticStrides));
  if (hasStaticStrides) {
    // A dynamic descriptor emits `pto::Stride<-1, ...>` and only carries its
    // strides at runtime, so those values have to come back through the
    // metadata reads below. Real strides are never negative.
    hasStaticStrides =
        llvm::all_of(staticStrides, [](int64_t s) { return s >= 0; });
  }
  if (hasStaticStrides &&
      !llvm::is_contained(partitionType.getShape(), ShapedType::kDynamic)) {
    SmallVector<int64_t, mlir::pto::kValue5> staticShape(
        partitionType.getShape().begin(), partitionType.getShape().end());
    return buildStaticPackedFp4TransferGlobalTensor(
        rewriter, loc, data, partitionType, packDims, isColMajor,
        scaleShape,
        staticShape, staticStrides, layoutString);
  }
  return buildRuntimePackedFp4TransferGlobalTensor(
      rewriter, loc, gmTensor, data, partitionType, packDims, isColMajor,
      scaleShape, layoutString);
}

FailureOr<Value> buildPackedFp4TransferGlobalTensor(
    ConversionPatternRewriter &rewriter, Operation *anchor, Value gmTensor,
    Value gmSource, pto::PartitionTensorViewType partitionType,
    pto::TileBufType tileType) {
  if (!mlir::pto::isPTOFloat4PackedType(tileType.getElementType())) {
    return gmTensor;
  }
  auto space = getTileAddressSpace(tileType);
  if (!space || (*space != pto::AddressSpace::VEC &&
                 *space != pto::AddressSpace::MAT)) {
    return gmTensor;
  }
  // A5's one-row MAT vector path uses sizeof(L1Type), not GetByteSize<FP4>.
  // Keep its carrier descriptor unchanged; ordinary MAT cube transfers below
  // still need the packed conversion.
  if (*space == pto::AddressSpace::MAT && tileType.getShape()[0] == 1 &&
      getTileBufBLayoutValue(tileType.getConfigAttr()) ==
          pto::BLayout::RowMajor &&
      getTileBufSLayoutValue(tileType.getConfigAttr()) ==
          pto::SLayout::RowMajor) {
    return gmTensor;
  }
  if (*space == pto::AddressSpace::VEC &&
      failed(checkPackedFp4TransferLayout(anchor, tileType))) {
    return failure();
  }
  return buildPackedFp4TransferDescriptor(
      rewriter, anchor, gmTensor, gmSource, partitionType, *space);
}

// CANN Open Software License Agreement Version 2.0 (the "License").
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

//===- PTOToEmitCLoadStore.cpp - tload/tstore/matmul lowering ---------===//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// pto.matmul_dps lowering (Simplified: No internal copy/sync)
//===----------------------------------------------------------------------===//
// Render `pto.tmatmul` as one of three forms depending on the optional
// `acc_phase` attribute, as follows.
//   * absent / Unspecified  -> `TMATMUL(dst, lhs, rhs)`
//   * Partial               -> `TMATMUL<pto::AccPhase::Partial>(dst, lhs, rhs)`
//   * Final                 -> `TMATMUL<pto::AccPhase::Final>(dst, lhs, rhs)`
// The Unspecified default keeps backward compatibility with all upstream IR
// that does not yet emit an explicit phase attribute.

// Emit an opaque call for a DPS tile op and forward (or erase) the op: when
// the op has a result, it is replaced by its dst operand.
void emitTileCallAndReplace(Operation *op, ConversionPatternRewriter &rewriter,
                                   StringRef callee, ArrayAttr templateArgs,
                                   ValueRange operands, Value dst) {
  rewriter.create<emitc::CallOpaqueOp>(op->getLoc(), TypeRange{}, callee,
                                       /*args=*/ArrayAttr{},
                                       /*templateArgs=*/templateArgs, operands);
  if (op->getNumResults() == 1) {
    rewriter.replaceOp(op, dst);
  } else {
    rewriter.eraseOp(op);
  }
}

//===----------------------------------------------------------------------===//
// pto.tgemv lowering
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// pto.tgemv.acc lowering
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// pto.matmul_acc_dps lowering (Simplified: No internal copy/sync)
//===----------------------------------------------------------------------===//

ArrayAttr buildAccPhaseTemplateArgs(ConversionPatternRewriter &rewriter,
                                           pto::AccPhase phase) {
  StringRef tmpl;
  switch (phase) {
  case pto::AccPhase::Unspecified:
    return ArrayAttr{};
  case pto::AccPhase::Partial:
    tmpl = "pto::AccPhase::Partial";
    break;
  case pto::AccPhase::Final:
    tmpl = "pto::AccPhase::Final";
    break;
  }
  if (tmpl.empty())
    return ArrayAttr{};
  return rewriter.getArrayAttr(
      {emitc::OpaqueAttr::get(rewriter.getContext(), tmpl)});
}


} // namespace pto
} // namespace mlir
