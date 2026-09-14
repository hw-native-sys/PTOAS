// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

static bool isA5OnlyFrontendFixpipeQuant(pto::FixpipeQuant quant) {
  return quant == pto::FixpipeQuant::QS322BF16PreScalar ||
         quant == pto::FixpipeQuant::QS322BF16PreVec ||
         quant == pto::FixpipeQuant::QF322HIF8PreScalar ||
         quant == pto::FixpipeQuant::QF322FP8PreScalar;
}

template <typename InitOpT>
static LogicalResult verifyFrontendInitFixpipe(InitOpT op, PTOArch arch) {
  auto epilogue = op.getAccPushEpilogueAttr();
  if (!epilogue)
    return success();
  if (op.getDirMask() != 1)
    return op.emitOpError(
        "expects fixpipe pipe (with 'acc_push_epilogue') to have dir_mask = 1 (C2V only)");
  if (!op.getNosplit())
    return op.emitOpError(
        "expects fixpipe pipe (with 'acc_push_epilogue') to have nosplit = true");
  if (op.getC2vConsumerBuf()) {
    Operation *def = op.getC2vConsumerBuf().getDefiningOp();
    if (!def || !isa<ReserveBufferOp, ImportReservedBufferOp>(def))
      return op.emitOpError(
          "expects fixpipe pipe 'c2v_consumer_buf' to trace to reserve_buffer or "
          "import_reserved_buffer for peer contract verification");
  }
  auto relu = epilogue.getRelu();
  if (relu != pto::FixpipeRelu::NoRelu &&
      relu != pto::FixpipeRelu::NormalRelu)
    return op.emitOpError(
        "expects 'acc_push_epilogue.relu' to be 'no_relu' or 'normal_relu' in v1");
  auto quant = epilogue.getQuant();
  if (!isAllowedFrontendFixpipeQuant(quant))
    return op.emitOpError(
        "expects 'acc_push_epilogue.quant' to be one of the v1 allowed quantization modes");
  if (arch != PTOArch::A5 && isA5OnlyFrontendFixpipeQuant(quant))
    return op.emitOpError(
        quant == pto::FixpipeQuant::QS322BF16PreScalar ||
                quant == pto::FixpipeQuant::QS322BF16PreVec
            ? "expects 'qs322bf16_pre_*' quantization modes to be used only on A5 target"
            : "expects 'qf322hif8_pre_scalar'/'qf322fp8_pre_scalar' to be used only on A5 target");
  return success();
}
template <typename InitOpT>
static LogicalResult verifyFrontendInitCommon(InitOpT op,
                                              FunctionKernelKind expected,
                                              StringRef kernelName) {
  auto slotNumResult = verifyFrontendInitIdentity(op, expected, kernelName);
  if (failed(slotNumResult))
    return failure();
  int32_t slotNum = *slotNumResult;
  PTOArch arch = getTargetArch(op.getOperation());
  auto globalOnly = verifyFrontendInitGlobalBacking(op, arch);
  if (failed(globalOnly))
    return failure();
  if (*globalOnly)
    return success();
  if (failed(verifyFrontendInitLocalBuffers(op)) ||
      failed(verifyFrontendInitLocalSlots(op, arch, slotNum)))
    return failure();

  return verifyFrontendInitFixpipe(op, arch);
}

ParseResult AicInitializePipeOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return parseFrontendInitializePipeOp(parser, result);
}

void AicInitializePipeOp::print(OpAsmPrinter &p) {
  printFrontendInitializePipeOp(*this, p);
}

ParseResult AivInitializePipeOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return parseFrontendInitializePipeOp(parser, result);
}

void AivInitializePipeOp::print(OpAsmPrinter &p) {
  printFrontendInitializePipeOp(*this, p);
}

ReserveBufferOp mlir::pto::findReserveBufferByName(func::FuncOp funcOp,
                                                   StringRef name) {
  ReserveBufferOp found;
  funcOp.walk([&](ReserveBufferOp reserveOp) {
    if (reserveOp.getName() != name) {
      return WalkResult::advance();
    }
    found = reserveOp;
    return WalkResult::interrupt();
  });
  return found;
}

LogicalResult ReserveBufferOp::verify() {
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return emitOpError("must be nested under a func.func");
  }

  if (getSize() <= 0) {
    return emitOpError("expects 'size' to be greater than 0");
  }

  auto location = getLocation().getAddressSpace();
  if (location != AddressSpace::VEC && location != AddressSpace::MAT) {
    return emitOpError("expects 'location' to be #pto.address_space<vec> or #pto.address_space<mat>");
  }

  if (!getAutoAlloc() && !getBaseAttr()) {
    return emitOpError("expects 'base' when 'auto' is false");
  }

  if (auto baseAttr = getBaseAttr(); baseAttr && baseAttr.getInt() < 0) {
    return emitOpError("expects 'base' to be non-negative when present");
  }

  unsigned sameNameCount = 0;
  funcOp.walk([&](ReserveBufferOp reserveOp) {
    if (reserveOp.getName() == getName()) {
      ++sameNameCount;
    }
  });
  if (sameNameCount > 1) {
    return emitOpError("requires 'name' to be unique within the function");
  }

  return success();
}

LogicalResult ImportReservedBufferOp::verify() {
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return emitOpError("must be nested under a func.func");
  }

  auto peerFunc = lookupPeerFuncAcrossContainer(getOperation(), getPeerFuncAttr());
  if (!peerFunc) {
    return emitOpError("expects 'peer_func' to reference an existing func.func");
  }

  unsigned sameImportCount = 0;
  funcOp.walk([&](ImportReservedBufferOp importOp) {
    if (importOp.getName() == getName() &&
        importOp.getPeerFuncAttr() == getPeerFuncAttr()) {
      ++sameImportCount;
    }
  });
  if (sameImportCount > 1) {
    return emitOpError(
        "requires (name, peer_func) to be unique within the function");
  }

  if (!findReserveBufferByName(peerFunc, getName())) {
    return emitOpError("expects matching peer reserve_buffer to exist");
  }

  return success();
}

constexpr llvm::StringLiteral kFrontendPipeIdAttrName = "__pto.frontend_id";
constexpr llvm::StringLiteral kPipePeerOwnerFuncAttrName =
    "__pto.peer_owner_func";
constexpr llvm::StringLiteral kPipePeerReserveNameAttrName =
    "__pto.peer_reserve_name";
constexpr llvm::StringLiteral kPipePeerDirMaskAttrName = "__pto.peer_dir_mask";

struct FixpipeQuantStateResource
    : public SideEffects::Resource::Base<FixpipeQuantStateResource> {
  StringRef getName() final { return "PTOFixpipeQuantState"; }
};

static IntegerAttr getFixpipeQuantStateIdAttr(Operation *op, int32_t id) {
    return IntegerAttr::get(IntegerType::get(op->getContext(), mlir::pto::kValue32), id);
}

static FailureOr<Operation *> lookupFrontendInitOpById(Operation *op,
                                                       func::FuncOp funcOp,
                                                       int32_t id) {
  Operation *matchedInit = nullptr;
  unsigned matchedInitCount = 0;
  funcOp.walk([&](Operation *candidate) {
    if (auto aic = dyn_cast<AicInitializePipeOp>(candidate)) {
      if (aic.getId() == static_cast<uint32_t>(id)) {
        matchedInit = candidate;
        ++matchedInitCount;
      }
      return WalkResult::advance();
    }
    if (auto aiv = dyn_cast<AivInitializePipeOp>(candidate)) {
      if (aiv.getId() == static_cast<uint32_t>(id)) {
        matchedInit = candidate;
        ++matchedInitCount;
      }
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });

  if (matchedInitCount == 0) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match a frontend initialize_pipe op in the same function";
    return failure();
  }
  if (matchedInitCount > 1) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match exactly one frontend initialize_pipe op in the same function";
    return failure();
  }
  return matchedInit;
}

static std::optional<int32_t> getFrontendPipeIdFromHandle(Value pipeHandle) {
  if (!pipeHandle) {
    return std::nullopt;
  }
  Operation *defOp = pipeHandle.getDefiningOp();
  if (!defOp) {
    return std::nullopt;
  }
  auto frontendIdAttr = defOp->getAttrOfType<IntegerAttr>(kFrontendPipeIdAttrName);
  if (!frontendIdAttr) {
    return std::nullopt;
  }
  return static_cast<int32_t>(frontendIdAttr.getInt());
}

static pto::AccPushEpilogueAttr getAccPushEpilogueFromInitOp(Operation *initOp) {
  if (!initOp) {
    return {};
  }
  if (auto aicInit = dyn_cast<AicInitializePipeOp>(initOp)) {
    return aicInit.getAccPushEpilogueAttr();
  }
  if (auto aivInit = dyn_cast<AivInitializePipeOp>(initOp)) {
    return aivInit.getAccPushEpilogueAttr();
  }
  if (auto l2lInit = dyn_cast<InitializeL2LPipeOp>(initOp)) {
    return l2lInit.getAccPushEpilogueAttr();
  }
  if (auto l2g2lInit = dyn_cast<InitializeL2G2LPipeOp>(initOp)) {
    return l2g2lInit.getAccPushEpilogueAttr();
  }
  return {};
}

static bool matchesLoweredFixpipePeerContract(Operation *initOp,
                                              func::FuncOp expectedOwnerFunc,
                                              StringRef expectedReserveName) {
  if (!initOp || !isa<InitializeL2LPipeOp, InitializeL2G2LPipeOp>(initOp)) {
    return false;
  }

  auto ownerAttr =
      initOp->getAttrOfType<FlatSymbolRefAttr>(kPipePeerOwnerFuncAttrName);
  auto reserveAttr =
      initOp->getAttrOfType<StringAttr>(kPipePeerReserveNameAttrName);
  auto dirMaskAttr =
      initOp->getAttrOfType<IntegerAttr>(kPipePeerDirMaskAttrName);
  if (!ownerAttr || !reserveAttr || !dirMaskAttr) {
    return false;
  }

  if (ownerAttr.getValue() != expectedOwnerFunc.getSymName() ||
      reserveAttr.getValue() != expectedReserveName ||
      dirMaskAttr.getInt() != 1) {
    return false;
  }

  return static_cast<bool>(getAccPushEpilogueFromInitOp(initOp));
}

static FailureOr<Operation *> lookupFrontendOrLoweredInitOpById(
    Operation *op, func::FuncOp funcOp, int32_t id);
