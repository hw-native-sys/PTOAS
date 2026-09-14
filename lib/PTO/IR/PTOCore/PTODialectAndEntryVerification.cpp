// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

AddressSpaceAttr mlir::pto::getPTOAddressSpaceAttr(Type type)
{
    if (auto ptrType = dyn_cast<PtrType>(type)) {
        return ptrType.getMemorySpace();
    }
    return {};
}

bool mlir::pto::hasExplicitPTOEntryAttr(func::FuncOp func)
{
    return func && (func->hasAttrOfType<UnitAttr>(kPTOEntryAttrName) ||
                    func->hasAttrOfType<UnitAttr>(kLegacyHACCEntryAttrName) ||
                    func->hasAttrOfType<UnitAttr>(kPTOKernelAttrName) ||
                    func->hasAttrOfType<UnitAttr>(kLegacyPTOAICoreAttrName));
}

bool mlir::pto::hasExplicitPTOEntryAttr(LLVM::LLVMFuncOp func)
{
    return func && (func->hasAttrOfType<UnitAttr>(kPTOEntryAttrName) ||
                    func->hasAttrOfType<UnitAttr>(kLegacyHACCEntryAttrName) ||
                    func->hasAttrOfType<UnitAttr>(kPTOKernelAttrName) ||
                    func->hasAttrOfType<UnitAttr>(kLegacyPTOAICoreAttrName));
}

bool mlir::pto::isPTOEntryFunction(func::FuncOp func)
{
    if (!func || func.isDeclaration()) {
        return false;
    }
    return hasExplicitPTOEntryAttr(func);
}

bool mlir::pto::isPTOEntryFunction(LLVM::LLVMFuncOp func)
{
    if (!func || func.isDeclaration()) {
        return false;
    }
    return hasExplicitPTOEntryAttr(func);
}

bool mlir::pto::hasExternalArtifactVisibility(func::FuncOp func)
{
    if (!func || func.isDeclaration()) {
        return false;
    }
    if (isPTOEntryFunction(func)) {
        return true;
    }
    auto attr = func->getAttrOfType<StringAttr>(kPTOVisibilityAttrName);
    if (!attr) {
        return false;
    }
    return attr.getValue() == kPTOVisibilityExternalValue;
}

void mlir::pto::setExternalArtifactVisibility(func::FuncOp func, bool isExternal)
{
    if (!func) {
        return;
    }
    if (isExternal) {
        func->setAttr(kPTOVisibilityAttrName, StringAttr::get(func.getContext(), kPTOVisibilityExternalValue));
        return;
    }
    func->removeAttr(kPTOVisibilityAttrName);
}

LogicalResult mlir::pto::validatePTOEntryFunctions(ModuleOp module)
{
    if (!module) {
        return success();
    }

    for (auto func : module.getOps<func::FuncOp>()) {
        if (!hasExplicitPTOEntryAttr(func)) {
            continue;
        }
        if (func.isDeclaration()) {
            return func.emitOpError() << "`" << kPTOEntryAttrName << "` is only valid on function definitions";
        }
    }

    for (auto func : module.getOps<func::FuncOp>()) {
        if (!isPTOEntryFunction(func)) {
            continue;
        }
        if (func.getFunctionType().getNumResults() != 0) {
            return func.emitOpError() << "PTO entry functions must return void";
        }
    }
    return success();
}

// A !pto.struct is represented as a pointer to stack storage. Its provenance
// must therefore remain explicit: the value comes directly from
// pto.declare_struct in the owning function. Function arguments/results and
// operations such as arith.select and scf.if must not manufacture or relay a
// struct-typed SSA value, because that alias hides the declaration from
// DeclareStructOp's direct-use escape check. CFG block arguments cannot make a
// declaration safe to forward either: the branch is a terminator and is
// rejected by DeclareStructOp::verify.
LogicalResult mlir::pto::validateStructProvenance(ModuleOp module)
{
    if (!module) {
        return success();
    }

    WalkResult result = module.walk([&](Operation* op) -> WalkResult {
        if (auto func = dyn_cast<func::FuncOp>(op)) {
            for (auto [i, inputTy] : llvm::enumerate(func.getFunctionType().getInputs())) {
                if (!isa<StructType>(inputTy)) {
                    continue;
                }
                func.emitOpError() << "argument " << i << " has type " << inputTy
                                   << ", but a stack-local struct must not be a function argument; "
                                      "structs must originate from 'pto.declare_struct' in the same "
                                      "function";
                return WalkResult::interrupt();
            }
            for (auto [i, resultTy] : llvm::enumerate(func.getFunctionType().getResults())) {
                if (!isa<StructType>(resultTy)) {
                    continue;
                }
                func.emitOpError() << "result " << i << " has type " << resultTy
                                   << ", but a stack-local struct must not be returned: the value is "
                                      "a pointer into the callee's frame, and returning it (even "
                                      "when it merely passes an argument back through) launders its "
                                      "provenance; keep the struct in its declaring function "
                                      "(pto.struct_set mutates in place, so a result is never needed)";
                return WalkResult::interrupt();
            }
        }

        if (!isa<DeclareStructOp>(op)) {
            for (auto [i, opResult] : llvm::enumerate(op->getResults())) {
                if (!isa<StructType>(opResult.getType())) {
                    continue;
                }
                op->emitOpError() << "result " << i << " has type " << opResult.getType()
                                  << ", but only 'pto.declare_struct' may produce a !pto.struct "
                                     "result; derived results hide the stack-storage lifetime and "
                                     "can escape their declaring scope";
                return WalkResult::interrupt();
            }
        }

        return WalkResult::advance();
    });
    return result.wasInterrupted() ? failure() : success();
}

void mlir::pto::annotatePTOEntryFunctions(ModuleOp module) { (void)module; }

//===----------------------------------------------------------------------===//
// PTO Load/Store/Addf (non-DPS polymorphic) verification + inference.
//===----------------------------------------------------------------------===//

static std::optional<uint64_t>
getLocalAddressAlignmentBytes(Attribute memorySpace) {
  auto addrSpace = dyn_cast_or_null<AddressSpaceAttr>(memorySpace);
  if (!addrSpace) {
    return std::nullopt;
  }

  // Keep this verifier as a conservative front-line guard for explicit local
  // tile addresses. PTO-ISA's buffer_limits.hpp defines the baseline
  // TASSIGN<Addr> alignment as 32 bytes for local tile memories. For L0 tile
  // bases, PTOAS level3/manual IR historically uses a 4096-bit (512-byte)
  // granularity; fuller per-arch/per-layout bounds checks belong in PTO-ISA.
  switch (addrSpace.getAddressSpace()) {
  case AddressSpace::VEC:
  case AddressSpace::MAT:
  case AddressSpace::BIAS:
  case AddressSpace::SCALING:
    return 32;
  case AddressSpace::LEFT:
  case AddressSpace::RIGHT:
  case AddressSpace::ACC:
    return 512;
  case AddressSpace::GM:
  case AddressSpace::Zero:
    return std::nullopt;
  }
  return std::nullopt;
}

static LogicalResult verifyConstantLocalAddress(Operation *op, Value addr,
                                                Attribute memorySpace,
                                                int addrIndex = -1) {
  std::optional<uint64_t> alignment =
      getLocalAddressAlignmentBytes(memorySpace);
  if (!alignment || *alignment == 0) {
    return success();
  }

  std::optional<int64_t> constantAddr = mlir::getConstantIntValue(addr);
  if (!constantAddr) {
    return success();
  }

  auto emitAddrError = [&]() {
    InFlightDiagnostic diag = op->emitOpError();
    if (addrIndex >= 0) {
      diag << "addr[" << addrIndex << "]";
    } else {
      diag << "addr";
}
    return diag;
  };

  if (*constantAddr < 0) {
    return emitAddrError() << " must be non-negative, got " << *constantAddr;
  }

  uint64_t unsignedAddr = static_cast<uint64_t>(*constantAddr);
  if ((unsignedAddr % *alignment) != 0) {
    return emitAddrError()
           << " must be aligned to " << *alignment
           << " bytes for local tile memory space, got " << unsignedAddr;
  }

  return success();
}

LogicalResult AllocTileOp::verify() {
  auto ty = getResult().getType(); // TileBufType

  if (failed(verifyTileBufLayoutConstraints(*this, ty, "result"))) {
    return failure();
  }

  if (failed(verifyConstantLocalAddress(getOperation(), getAddr(),
                                        ty.getMemorySpace()))) {
    return failure();
  }

  // op 上有没有传 operands
  bool hasVR = getValidRow() != nullptr;
  bool hasVC = getValidCol() != nullptr;

  // type 上的 validShape
  auto vs = ty.getValidShape();
  if (vs.size() != 2) {
    return emitOpError("result tile_buf must have rank-2 validShape");
  }

  // TileBuf valid dims use a negative sentinel (e.g. '?' / -1). Be robust to
  // any negative value (some code may materialize MLIR dynamic sentinels).
  bool needVR = (vs[0] < 0);
  bool needVC = (vs[1] < 0);

  // 你要求的：v_row=?, v_col=? 时必须同时给两个
  // （这条规则由下面两句自然实现）
  if (hasVR != needVR) {
    return emitOpError() << "valid_row operand "
                         << (needVR ? "is required" : "must be absent")
                         << " because result type v_row is "
                         << (needVR ? "?" : std::to_string(vs[0]));
  }

  if (hasVC != needVC) {
    return emitOpError() << "valid_col operand "
                         << (needVC ? "is required" : "must be absent")
                         << " because result type v_col is "
                         << (needVC ? "?" : std::to_string(vs[1]));
  }

  return success();
}
