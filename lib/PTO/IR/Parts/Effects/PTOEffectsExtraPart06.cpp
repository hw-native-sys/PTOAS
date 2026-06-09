// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTOEffectsExtra.cpp; kept as a fragment included by PTOEffectsExtra.cpp.
// Intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

LogicalResult TPutOp::verify() {
  return verifyCommTransferWithStaging(getOperation(), getDst(), getSrc(),
                                       getPing(), getPong());
}

LogicalResult TGetOp::verify() {
  return verifyCommTransferWithStaging(getOperation(), getDst(), getSrc(),
                                       getPing(), getPong());
}

LogicalResult TNotifyOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommSignalLike(*this, getSignal(), "signal")))
    return failure();
  auto valueTy = dyn_cast<IntegerType>(getValue().getType());
  if (!valueTy || valueTy.getWidth() != kPTOI32BitWidth)
    return emitOpError("expects value to be i32");
  return success();
}

LogicalResult TWaitOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommSignalLike(*this, getSignal(), "signal")))
    return failure();
  auto cmpTy = dyn_cast<IntegerType>(getCmpValue().getType());
  if (!cmpTy || cmpTy.getWidth() != kPTOI32BitWidth)
    return emitOpError("expects cmp_value to be i32");
  return success();
}

LogicalResult TTestOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommSignalLike(*this, getSignal(), "signal")))
    return failure();
  auto cmpTy = dyn_cast<IntegerType>(getCmpValue().getType());
  if (!cmpTy || cmpTy.getWidth() != kPTOI32BitWidth)
    return emitOpError("expects cmp_value to be i32");
  return success();
}

static LogicalResult verifySyncAllGmWorkspace(Operation *op, Value workspace,
                                              StringRef name) {
  Type ty = workspace.getType();
  if (!isa<MemRefType, pto::TensorViewType, pto::PartitionTensorViewType>(ty))
    return op->emitOpError() << "expects " << name
                             << " to be a GM memref/tensor_view/partition_view";
  if (auto memTy = dyn_cast<MemRefType>(ty)) {
    if (!memTy.hasRank())
      return op->emitOpError() << "expects " << name << " to be ranked";
    if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
      return op->emitOpError() << "expects " << name
                               << " to be in GM address space";
  }

  auto elemTy = dyn_cast<IntegerType>(getElemTy(ty));
  if (!elemTy || elemTy.getWidth() != kPTOI32BitWidth)
    return op->emitOpError() << "expects " << name
                             << " element type to be i32";

  SmallVec4<int64_t> shape = getShapeVec(ty);
  if (shape.empty())
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  for (int64_t dim : shape) {
    if (dim != ShapedType::kDynamic && dim <= 0)
      return op->emitOpError() << "expects " << name
                               << " shape to be positive";
  }
  return success();
}

static LogicalResult verifySyncAllTileWorkspace(Operation *op, Value workspace,
                                                StringRef name,
                                                pto::AddressSpace expectedSpace) {
  Type ty = workspace.getType();
  if (!isa<pto::TileBufType, MemRefType>(ty))
    return op->emitOpError() << "expects " << name
                             << " to be tile_buf or memref type";
  if (isa<pto::TileBufType>(ty) && failed(verifyTileBufCommon(op, ty, name)))
    return failure();

  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != expectedSpace)
    return op->emitOpError() << "expects " << name << " to be in "
                             << (expectedSpace == pto::AddressSpace::VEC
                                     ? "vec"
                                     : "mat")
                             << " address space";

  Type elemTy = getElemTy(ty);
  auto intTy = dyn_cast_or_null<IntegerType>(elemTy);
  if (!intTy || intTy.getWidth() != kPTOI32BitWidth)
    return op->emitOpError() << "expects " << name
                             << " element type to be i32";

  auto shape = getShapeVec(ty);
  if (shape.empty() || shape.size() > kPTORowColRank)
    return op->emitOpError() << "expects " << name
                             << " to be rank-1 or rank-2";
  for (int64_t dim : shape) {
    if (dim != ShapedType::kDynamic && dim <= 0)
      return op->emitOpError() << "expects " << name
                               << " shape to be positive";
  }
  return success();
}

static LogicalResult verifySyncAllHardMode(SyncAllOp op, bool hasGm, bool hasUb,
                                           bool hasL1) {
  if (hasGm || hasUb || hasL1 || op.getUsedCores()) {
    return op.emitOpError(
        "expects hard syncall to have no workspace operands or used_cores");
  }
  return success();
}

static LogicalResult verifySyncAllUsedCores(SyncAllOp op) {
  if (auto used = op.getUsedCores()) {
    auto intTy = dyn_cast<IntegerType>(used.getType());
    if (!intTy || intTy.getWidth() != kPTOI32BitWidth)
      return op.emitOpError("expects used_cores to be i32");
  }
  return success();
}

static LogicalResult verifySyncAllSoftWorkspaces(SyncAllOp op, bool hasUb,
                                                 bool hasL1) {
  switch (op.getCoreType().getValue()) {
  case pto::SyncCoreType::AIVOnly:
    if (!hasUb || hasL1) {
      return op.emitOpError(
          "expects soft AIV-only syncall to use gm_workspace + ub_workspace only");
    }
    return verifySyncAllTileWorkspace(op.getOperation(), op.getUbWorkspace(),
                                      "ub_workspace", pto::AddressSpace::VEC);
  case pto::SyncCoreType::AICOnly:
    if (hasUb || !hasL1) {
      return op.emitOpError(
          "expects soft AIC-only syncall to use gm_workspace + l1_workspace only");
    }
    return verifySyncAllTileWorkspace(op.getOperation(), op.getL1Workspace(),
                                      "l1_workspace", pto::AddressSpace::MAT);
  case pto::SyncCoreType::Mix:
    if (!hasUb || !hasL1) {
      return op.emitOpError(
          "expects soft mixed syncall to use gm_workspace + ub_workspace + l1_workspace");
    }
    if (failed(verifySyncAllTileWorkspace(op.getOperation(), op.getUbWorkspace(),
                                          "ub_workspace",
                                          pto::AddressSpace::VEC))) {
      return failure();
    }
    return verifySyncAllTileWorkspace(op.getOperation(), op.getL1Workspace(),
                                      "l1_workspace", pto::AddressSpace::MAT);
  }
  llvm_unreachable("unhandled SyncCoreType");
}

LogicalResult SyncAllOp::verify() {
  bool hasGm = static_cast<bool>(getGmWorkspace());
  bool hasUb = static_cast<bool>(getUbWorkspace());
  bool hasL1 = static_cast<bool>(getL1Workspace());
  if (getMode().getValue() == pto::SyncAllMode::Hard)
    return verifySyncAllHardMode(*this, hasGm, hasUb, hasL1);
  if (!hasGm)
    return emitOpError("expects soft syncall to provide gm_workspace");
  if (failed(verifySyncAllGmWorkspace(getOperation(), getGmWorkspace(),
                                      "gm_workspace")) ||
      failed(verifySyncAllUsedCores(*this)) ||
      failed(verifySyncAllSoftWorkspaces(*this, hasUb, hasL1)))
    return failure();
  return success();
}

LogicalResult TBroadcastOp::verify() {
  if (failed(verifyRootedCommTileTransfer(getOperation(), getSrc(), getGroup(),
                                          getRoot(), getPing(), getPong())))
    return failure();
  if (getSrc().getType() != getGroup().front().getType())
    return emitOpError("expects src type to match group member type");
  return success();
}

LogicalResult CommTGatherOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommGlobalLike(*this, getDst(), "dst")) ||
      failed(verifyCommStagingTileLike(*this, getPing(), "ping")) ||
      failed(verifyCommPingPongSameType(*this, getPing(), getPong(), "ping",
                                        "pong")) ||
      failed(verifyCommGlobalGroup(*this, getGroup(), "group")))
    return failure();
  if (getRoot() >= static_cast<uint32_t>(getGroup().size()))
    return emitOpError("expects root to index into group operands");
  if (getElemTy(getDst().getType()) != getElemTy(getGroup().front().getType()))
    return emitOpError("expects dst element type to match group member type");
  if (getElemTy(getPing().getType()) != getElemTy(getDst().getType()))
    return emitOpError("expects staging tile element type to match dst");
  return success();
}
