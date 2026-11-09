// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

#include "llvm/ADT/APInt.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/Matchers.h"

// An alignment is a claim about an exact pointer value, in bytes. It is
// deliberately limited to signed 64-bit positive values for the analysis API.
static LogicalResult verifyPointerAlignmentValue(Operation *op, Attribute value)
{
    auto alignment = dyn_cast<IntegerAttr>(value);
    if (!alignment || !alignment.getValue().isStrictlyPositive() ||
        alignment.getValue().getActiveBits() > 63 || !alignment.getValue().isPowerOf2()) {
        return op->emitError("pto.alignment must be a positive power of two fitting in 63 bits");
    }
    return success();
}

LogicalResult PTODialect::verifyOperationAttribute(Operation *op, NamedAttribute attr)
{
    if (attr.getName() != "pto.alignment") {
        return success();
    }
    auto ptrCast = dyn_cast<CastPtrOp>(op);
    if (!ptrCast || !isa<IntegerType>(ptrCast.getInput().getType()) ||
        !isa<PtrType>(ptrCast.getResult().getType())) {
        return op->emitError("pto.alignment requires an integer-to-pointer pto.castptr");
    }
    if (failed(verifyPointerAlignmentValue(op, attr.getValue()))) {
        return failure();
    }
    auto alignment = cast<IntegerAttr>(attr.getValue()).getValue().getZExtValue();
    auto inputType = cast<IntegerType>(ptrCast.getInput().getType());
    if (llvm::Log2_64(alignment) >= inputType.getWidth()) {
        return op->emitError("pto.alignment does not fit the address width");
    }
    IntegerAttr constant;
    if (matchPattern(ptrCast.getInput(), m_Constant(&constant)) &&
        constant.getValue().urem(alignment) != 0) {
        return op->emitError("pto.alignment contradicts the constant address");
    }
    return success();
}

LogicalResult PTODialect::verifyRegionArgAttribute(Operation *op, unsigned regionIndex,
                                                    unsigned argIndex, NamedAttribute attr)
{
    if (attr.getName() != "pto.alignment") {
        return success();
    }
    auto func = dyn_cast<func::FuncOp>(op);
    if (!func || regionIndex != 0 || argIndex >= func.getNumArguments() ||
        !isa<PtrType>(func.getArgumentTypes()[argIndex])) {
        return op->emitError("pto.alignment requires a pointer function argument");
    }
    return verifyPointerAlignmentValue(op, attr.getValue());
}

LogicalResult PTODialect::verifyRegionResultAttribute(Operation *op, unsigned,
                                                       unsigned, NamedAttribute attr)
{
    if (attr.getName() == "pto.alignment") {
        return op->emitError("pto.alignment is not supported on function results");
    }
    return success();
}

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
    // LLVM::LLVMFuncOp keeps the default Symbol::isDeclaration(), which always
    // reports a definition, so inspect the body instead. Otherwise an external
    // declaration carrying pto.entry would be mistaken for an entry definition.
    if (!func || func.isExternal()) {
        return false;
    }
    return hasExplicitPTOEntryAttr(func);
}

static bool hasExternalArtifactVisibilityImpl(mlir::Operation* func, bool isEntry)
{
    if (isEntry) {
        return true;
    }
    auto attr = func->getAttrOfType<mlir::StringAttr>(mlir::pto::kPTOVisibilityAttrName);
    return attr && attr.getValue() == mlir::pto::kPTOVisibilityExternalValue;
}

bool mlir::pto::hasExternalArtifactVisibility(func::FuncOp func)
{
    if (!func || func.isDeclaration()) {
        return false;
    }
    return hasExternalArtifactVisibilityImpl(func.getOperation(), isPTOEntryFunction(func));
}

bool mlir::pto::hasExternalArtifactVisibility(LLVM::LLVMFuncOp func)
{
    // Same isDeclaration() caveat as isPTOEntryFunction(LLVM::LLVMFuncOp):
    // inspect the body instead.
    if (!func || func.isExternal()) {
        return false;
    }
    return hasExternalArtifactVisibilityImpl(func.getOperation(), isPTOEntryFunction(func));
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

    WalkResult result = module.walk([](Operation* op) -> WalkResult {
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
  constexpr uint64_t kLocalTileAlignBytes = mlir::pto::kValue32;
  constexpr uint64_t kL0TileBaseAlignBytes = mlir::pto::kValue512;
  switch (addrSpace.getAddressSpace()) {
  case AddressSpace::VEC:
  case AddressSpace::MAT:
  case AddressSpace::BIAS:
  case AddressSpace::SCALING:
    return kLocalTileAlignBytes;
  case AddressSpace::LEFT:
  case AddressSpace::RIGHT:
  case AddressSpace::ACC:
    return kL0TileBaseAlignBytes;
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

  auto emitAddrError = [op, addrIndex]() {
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

// pad_value flows to SetPadValue(DType) through an implicit C++
// conversion. A float value feeding an integer tile truncates toward
// zero and an out-of-range integer wraps -- both silently. Reject the
// combinations that lose data; an integer pad on a float tile converts
// exactly (up to 2^24) and stays legal.
static LogicalResult verifyConvTilePadValue(pto::AllocTileOp op,
                                            pto::ConvTileBufType convTy,
                                            pto::ConvTileConfigAttr cfg) {
  Type tileElem = convTy.getElementType();
  Attribute padValue = cfg.getPadValue();
  if (auto floatAttr = dyn_cast<mlir::FloatAttr>(padValue)) {
    if (tileElem.isa<mlir::IntegerType>()) {
      return op.emitOpError()
             << "config pad_value " << floatAttr.getValueAsDouble()
             << " is a float but the tile element type is "
             << tileElem << "; the runtime would silently truncate it";
    }
    return success();
  }
  if (auto intAttr = dyn_cast<mlir::IntegerAttr>(padValue)) {
    if (auto intTile = dyn_cast<mlir::IntegerType>(tileElem)) {
      int64_t value = intAttr.getInt();
      int64_t minValue = llvm::APInt::getSignedMinValue(intTile.getWidth())
                             .getSExtValue();
      int64_t maxValue = llvm::APInt::getSignedMaxValue(intTile.getWidth())
                             .getSExtValue();
      if (value < minValue || value > maxValue) {
        return op.emitOpError()
               << "config pad_value " << value
               << " does not fit the tile element type " << tileElem
               << "; the runtime would silently wrap it";
      }
    }
    return success();
  }
  return success();
}

static LogicalResult verifyConvTileAlloc(pto::AllocTileOp op) {
  if (op.getValidRow() || op.getValidCol()) {
    return op.emitOpError(
        "valid_row and valid_col operands require result to be "
        "`!pto.conv_tile_buf`");
  }
  auto convTy = cast<pto::ConvTileBufType>(op.getResult().getType());
  if (auto cfg = op.getConfig()) {
    // a2a3's ConvTile has no dst_stride/dst_mposition fields and EmitC
    // drops them; A5 packs them into the SET_IMG2COL_RPT config.
    if (getVerifierTargetArch(op.getOperation()) != VerifierTargetArch::A5 &&
        (cfg->getDstStride().getInt() != 0 ||
         cfg->getDstMposition().getInt() != 0)) {
      return op.emitOpError(
          "config dst_stride/dst_mposition require --pto-arch=a5");
    }
    if (failed(verifyConvTilePadValue(op, convTy, *cfg))) {
      return failure();
    }
  }
  return verifyConstantLocalAddress(op.getOperation(), op.getAddr(),
                                    convTy.getMemorySpace());
}

LogicalResult AllocTileOp::verify() {
  auto resultType = getResult().getType();
  if (isa<pto::ConvTileBufType>(resultType)) {
    return verifyConvTileAlloc(*this);
  }
  if (getConfig()) {
    return emitOpError("config attribute requires result to be `!pto.conv_tile_buf`");
  }

  auto ty = dyn_cast<pto::TileBufType>(resultType);
  if (!ty) {
    return emitOpError("result must be `!pto.tile_buf` or `!pto.conv_tile`");
  }

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
