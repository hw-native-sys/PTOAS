// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. Do not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

//===----------------------------------------------------------------------===//
// ConvTile / IMG2COL config op verification
//===----------------------------------------------------------------------===//

static LogicalResult verifyConvTileCommon(Operation *op, Type ty, StringRef name,
                                          bool allowLowPrecision) {
  auto ct = dyn_cast<pto::ConvTileBufType>(ty);
  if (!ct) {
    return op->emitOpError() << "expects " << name << " to be a !pto.conv_tile";
  }

  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::MAT) {
    return op->emitOpError() << "expects " << name
                             << " to be in the mat address space";
  }

  if (!allowLowPrecision && isPTOLowPrecisionType(ct.getElementType())) {
    return op->emitOpError() << name << ": dtype " << ct.getElementType()
                             << " is not supported by this op yet";
  }

  auto shape = getShapeVec(ty);
  if (shape.empty() || shape.size() > 6) {
    return op->emitOpError() << "expects " << name
                             << " to have a rank in [1, 6]";
  }
  for (unsigned i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 0) {
      return op->emitOpError() << "expects " << name << " shape[" << i
                               << "] to be positive";
    }
  }

  auto bufferSize = dyn_cast<IntegerAttr>(ct.getBufferSize());
  if (!bufferSize) {
    return op->emitOpError() << "expects " << name
                             << " to have a signless integer buffer_size";
  }
  if (bufferSize.getInt() <= 0) {
    return op->emitOpError() << "expects " << name
                             << " buffer_size to be positive";
  }

  int64_t logicalSize = 1;
  for (int64_t dim : shape) {
    if (logicalSize > std::numeric_limits<int64_t>::max() / dim) {
      return op->emitOpError() << "cannot compute logical size of " << name
                               << " without overflow";
    }
    logicalSize *= dim;
  }
  if (bufferSize.getInt() < logicalSize) {
    return op->emitOpError() << "expects " << name
                             << " buffer_size to cover at least "
                             << logicalSize << " elements";
  }

  return success();
}

static LogicalResult verifyTileLikeCommon(Operation *op, Type ty, StringRef name,
                                          bool allowLowPrecision) {
  if (isa<pto::TileBufType>(ty)) {
    return verifyTileBufCommon(op, ty, name, allowLowPrecision);
  }
  if (isa<pto::ConvTileBufType>(ty)) {
    return verifyConvTileCommon(op, ty, name, allowLowPrecision);
  }
  return op->emitOpError() << "expects " << name
                           << " to be a !pto.tile_buf or !pto.conv_tile";
}

// Shared verifier for the fmatrix-programming ops on a conv_tile source
// (set_fmatrix, set_img2col_rpt, set_img2col_padding): all require a valid
// mat conv_tile src and a manual fmatrix_mode selector.
template <typename OpT>
static LogicalResult verifySetFmatrixFamilyOp(OpT op) {
  if (failed(verifyConvTileCommon(op, op.getSrc().getType(), "src",
                                  /*allowLowPrecision=*/true))) {
    return failure();
  }
  auto mode = op.getFmatrixMode();
  if (mode != pto::FmatrixMode::FMATRIX_A_MANUAL &&
      mode != pto::FmatrixMode::FMATRIX_B_MANUAL) {
    return op.emitOpError("expects fmatrix_mode to be a_manual or b_manual");
  }
  return success();
}

LogicalResult mlir::pto::SetFmatrixOp::verify() {
  return verifySetFmatrixFamilyOp(*this);
}

LogicalResult mlir::pto::SetImg2colRptOp::verify() {
  return verifySetFmatrixFamilyOp(*this);
}

LogicalResult mlir::pto::SetImg2colPaddingOp::verify() {
  return verifySetFmatrixFamilyOp(*this);
}

static LogicalResult verifyTImg2colSrcConvLayout(TImg2colOp op) {
  auto srcCT = dyn_cast<pto::ConvTileBufType>(op.getSrc().getType());
  if (!srcCT) {
    return op.emitOpError("expects src to be a !pto.conv_tile");
  }
  auto srcLayout = dyn_cast_or_null<pto::ConvLayoutAttr>(srcCT.getLayout());
  if (!srcLayout ||
      (srcLayout.getValue() != pto::ConvLayout::NC1HWC0 &&
       srcLayout.getValue() != pto::ConvLayout::NDC1HWC0)) {
    return op.emitOpError("expects src layout to be NC1HWC0 or NDC1HWC0");
  }
  return success();
}

static LogicalResult verifyTImg2colDstLayout(TImg2colOp op) {
  auto dstTile = dyn_cast<pto::TileBufType>(op.getDst().getType());
  if (!dstTile) {
    return op.emitOpError("expects dst to be a !pto.tile_buf");
  }
  if (failed(verifyTileBufLayoutConstraints(op, dstTile, "dst"))) {
    return failure();
  }
  // The runtime generations disagree on the dst BLayout: A5's TIMG2COL
  // wants col_major (a5/TImg2col.hpp), A2/A3's wants row_major
  // (a2a3/TImg2col.hpp TIMG2COL_IMPL).
  const bool isA5 = getVerifierTargetArch(op.getOperation()) ==
                    VerifierTargetArch::A5;
  const auto expectedB = static_cast<int32_t>(isA5 ? pto::BLayout::ColMajor
                                                   : pto::BLayout::RowMajor);
  if (dstTile.getBLayoutValueI32() != expectedB ||
      dstTile.getSLayoutValueI32() !=
          static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op.emitOpError()
           << "expects dst layout to be BLayout="
           << (isA5 ? "col_major" : "row_major")
           << " and SLayout=row_major on this arch";
  }
  auto dstSpace = getPTOMemorySpaceEnum(dstTile);
  if (!dstSpace || *dstSpace != pto::AddressSpace::LEFT) {
    return op.emitOpError("expects dst to use loc=left");
  }
  return success();
}

static LogicalResult verifyTImg2colFmatrixMode(TImg2colOp op) {
  auto mode = op.getFmatrixMode();
  if (mode != pto::FmatrixMode::FMATRIX_A_AUTO &&
      mode != pto::FmatrixMode::FMATRIX_B_AUTO &&
      mode != pto::FmatrixMode::FMATRIX_A_MANUAL &&
      mode != pto::FmatrixMode::FMATRIX_B_MANUAL) {
    return op.emitOpError(
        "expects fmatrix_mode to be one of a_auto, b_auto, a_manual, or b_manual");
  }
  return success();
}

// Mirrors Timg2colConvTileCheck in a5/TImg2col.hpp: the tile dtypes are
// limited to int8/uint8/int16/uint16/int32/uint32/half/bfloat16/float. PTO
// integer types are signless, so the uint twins share one case each.
static LogicalResult verifyTImg2colElemType(TImg2colOp op) {
  // Mirrors Timg2colConvTileCheck: A5 accepts i8/i16/i32/f16/bf16/f32
  // (a5/TImg2col.hpp), while A2/A3's TIMG2COL_IMPL rejects i16/i32
  // (a2a3/TImg2col.hpp "Invalid data type").
  const bool isA5 = getVerifierTargetArch(op.getOperation()) ==
                    VerifierTargetArch::A5;
  Type elem = getElemTy(op.getDst().getType());
  const bool legal =
      elem.isInteger(8) || elem.isF16() || elem.isBF16() || elem.isF32() ||
      (isA5 && (elem.isInteger(16) || elem.isInteger(32)));
  if (!legal) {
    return op.emitOpError() << "expects timg2col element type to be "
                               "i8/f16/bf16/f32"
                            << (isA5 ? "/i16/i32" : "")
                            << " on this arch";
  }
  return success();
}

// A5: posK selects the first K position of the im2col expansion, and the
// call covers FH*FW*C0 consecutive K positions starting there. A window
// that starts past the fmap channel extent makes TIMG2COL walk the L1
// conv tile out of range and fault the AICORE (observed as EZ9999 error
// 143 on Ascend950PR with fmap C_in=16 and posK=8), so a constant posK
// is range-checked against the fmap K extent. posM is left unchecked:
// it offsets the M window over padded fmap rows, which the hardware
// clamps. New code follows the repo .clang-format (Google, 4-space).
static LogicalResult verifyTImg2colPosKWindow(TImg2colOp op) {
    const bool isA5 =
        getVerifierTargetArch(op.getOperation()) == VerifierTargetArch::A5;
    if (!isA5 || !op.getPosK()) {
        return success();
    }
    auto srcCT = dyn_cast<pto::ConvTileBufType>(op.getSrc().getType());
    auto dstTB = dyn_cast<pto::TileBufType>(op.getDst().getType());
    auto posK = mlir::getConstantIntValue(op.getPosK());
    if (!srcCT || !dstTB || !posK || *posK <= 0 || srcCT.getShape().empty() ||
        llvm::is_contained(srcCT.getShape(), ShapedType::kDynamic)) {
        return success();
    }
    auto *defOp = op.getSrc().getDefiningOp();
    auto allocOp = defOp ? dyn_cast<pto::AllocTileOp>(defOp) : nullptr;
    auto cfg = allocOp ? allocOp.getConfig() : pto::ConvTileConfigAttr();
    if (!cfg) {
        return success();
    }
    // fmap channel extent: K_total = C1 * C0 * FH * FW (im2col K dim).
    ArrayRef<int64_t> ctShape = srcCT.getShape();
    const int64_t kTotal = ctShape[1] * ctShape[4] *
                               cfg->getFilterH().getInt() *
                               cfg->getFilterW().getInt();
    // The dst left tile k-window spanned by this call.
    auto dstValid = dstTB.getValidShape();
    if (dstValid.size() == 2 && dstValid[1] >= 0 &&
        *posK + dstValid[1] > kTotal) {
        return op.emitOpError()
               << "posK " << *posK << " + dst k-width " << dstValid[1]
               << " exceeds the fmap K extent " << kTotal
               << " (C1*C0*FH*FW); TIMG2COL would walk the L1 conv tile "
                  "out of range and fault the AICORE";
    }
    return success();
}

LogicalResult mlir::pto::TImg2colOp::verify() {
  if (failed(verifyTileBufCommon(*this, getDst().getType(), "dst",
                                 /*allowLowPrecision=*/true))) {
    return failure();
  }
  if (failed(verifyConvTileCommon(*this, getSrc().getType(), "src",
                                  /*allowLowPrecision=*/true))) {
    return failure();
  }
  if (getElemTy(getDst().getType()) != getElemTy(getSrc().getType())) {
    return emitOpError() << "expects src and dst to have the same element type";
  }
  if (failed(verifyTImg2colElemType(*this)) ||
      failed(verifyTImg2colSrcConvLayout(*this)) ||
      failed(verifyTImg2colDstLayout(*this)) ||
      failed(verifyTImg2colFmatrixMode(*this))) {
    return failure();
  }

  // posM/posK are runtime offsets; only constant ones can be statically
  // range-checked against the uint16 the runtime takes.
  auto checkPos = [&](Value pos, StringRef name) -> LogicalResult {
    std::optional<int64_t> constant = mlir::getConstantIntValue(pos);
    if (constant && (*constant < 0 ||
                     *constant > std::numeric_limits<uint16_t>::max())) {
      return emitOpError() << "expects " << name
                           << " to fit in an unsigned 16-bit integer";
    }
    return success();
  };
  if (getPosM() && failed(checkPos(getPosM(), "posM"))) {
    return failure();
  }
  if (getPosK() && failed(checkPos(getPosK(), "posK"))) {
    return failure();
  }

  if (failed(verifyTImg2colPosKWindow(*this))) {
    return failure();
  }

  return success();
}