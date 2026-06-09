// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOVerifyCore.cpp; kept as a fragment included by PTOVerifyCore.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

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

[[maybe_unused]] static LogicalResult verifyMemrefToTensorLoad(Operation *op, Value src, Value res) {
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

[[maybe_unused]] static LogicalResult verifyMemrefTensorStore(Operation *op, Value dst, Value src) {
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
  if (failed(verifyTileBufLayoutConstraints(*this, ty, "result")))
    return failure();

  // op 上有没有传 operands
  bool hasVR = getValidRow() != nullptr;
  bool hasVC = getValidCol() != nullptr;

  // type 上的 validShape
  auto vs = ty.getValidShape();
  if (vs.size() != kPTORowColRank)
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

LogicalResult MaterializeTileOp::verify() {
  auto sourceTy = cast<MemRefType>(getSource().getType());
  auto resultTy = cast<TileBufType>(getResult().getType());

  if (sourceTy.getRank() != kPTORowColRank)
    return emitOpError("source memref must be rank-2 to materialize a tile handle");
  if (resultTy.getRank() != kPTORowColRank)
    return emitOpError("result tile_buf must be rank-2");
  if (failed(verifyTileBufLayoutConstraints(*this, resultTy, "result")))
    return failure();

  auto viewSemantics = (*this)->getAttrOfType<StringAttr>("pto.view_semantics");
  bool isSubview = viewSemantics && viewSemantics.getValue() == "subview";
  if (!isSubview && sourceTy.getShape() != resultTy.getShape())
    return emitOpError() << "source/result shape mismatch: source="
                         << sourceTy << " result=" << resultTy;

  if (sourceTy.getElementType() != resultTy.getElementType())
    return emitOpError() << "source/result element type mismatch: source="
                         << sourceTy.getElementType()
                         << " result=" << resultTy.getElementType();

  if (sourceTy.getMemorySpace() != resultTy.getMemorySpace())
    return emitOpError() << "source/result memory space mismatch";

  if (getConfig() != resultTy.getConfigAttr())
    return emitOpError("config attribute must match the result tile_buf config");

  auto shape = resultTy.getShape();
  auto validShape = resultTy.getValidShape();
  if (validShape.size() != kPTORowColRank)
    return emitOpError("result tile_buf must have rank-2 validShape");
  for (unsigned i = 0; i < kPTORowColRank; ++i) {
    if (shape[i] != ShapedType::kDynamic &&
        validShape[i] != ShapedType::kDynamic && validShape[i] > shape[i]) {
      return emitOpError() << "valid_shape[" << i << "] must be <= shape["
                           << i << "]";
    }
  }

  return success();
}

LogicalResult TAssignOp::verify() {
  if (getTile().getType() != getResult().getType()) {
    return emitOpError("result type must match tile operand type");
  }
  return success();
}

LogicalResult TLoadOp::verify() {
  auto verifyA2A3 = [this]() -> LogicalResult {
    auto common =
        verifyTLoadCommon(*this, getSrc(), getDst(), /*allowLowPrecision=*/false);
    if (failed(common))
      return failure();
    return verifyTLoadA2A3(*this, *common);
  };

  auto verifyA5 = [this]() -> LogicalResult {
    auto common =
        verifyTLoadCommon(*this, getSrc(), getDst(), /*allowLowPrecision=*/true);
    if (failed(common))
      return failure();
    return verifyTLoadA5(*this, *common);
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TPrefetchOp::verify() {
  auto verifyByArch = [this](bool allowLowPrecision) -> LogicalResult {
    auto srcElem = verifyTPrefetchSrcElemType(*this, getSrc().getType());
    if (failed(srcElem))
      return failure();
    auto dstElem =
        verifyTPrefetchDstElemType(*this, getDst().getType(), allowLowPrecision);
    if (failed(dstElem))
      return failure();
    return verifyTPrefetchElemTypes(*this, *srcElem, *dstElem,
                                    allowLowPrecision);
  };
  return dispatchVerifierByArch(
      getOperation(),
      [&verifyByArch]() {
        return verifyByArch(/*allowLowPrecision=*/false);
      },
      [&verifyByArch]() {
        return verifyByArch(/*allowLowPrecision=*/true);
      });
}

LogicalResult MakePrefetchAsyncContextOp::verify() {
  Type workspaceTy = getWorkspace().getType();
  Type elemTy = nullptr;
  if (auto ptrTy = dyn_cast<pto::PtrType>(workspaceTy)) {
    elemTy = ptrTy.getElementType();
  } else if (auto memTy = dyn_cast<MemRefType>(workspaceTy)) {
    if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
      return emitOpError("expects workspace memref to be in GM address space");
    elemTy = memTy.getElementType();
  } else {
    return emitOpError("expects workspace to be !pto.ptr<i8> or GM memref<i8>");
  }
  if (!isByteIntegerType(elemTy))
    return emitOpError("expects workspace element type to be an 8-bit integer");
  return success();
}
