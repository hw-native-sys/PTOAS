//===- VPTO.cpp - VPTO dialect -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

using namespace mlir;
using namespace mlir::pto;

static llvm::cl::opt<bool> disableVPTOAlignChainVerification(
    "vpto-disable-align-chain-verification",
    llvm::cl::desc("Disable !pto.align linear-chain verifier checks"),
    llvm::cl::init(false), llvm::cl::Hidden);

static std::string formatVRegType(int64_t elementCount, Type elementType) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << "!pto.vreg<" << elementCount << "x" << elementType << ">";
  return storage;
}

static std::string formatMaskType(StringRef granularity) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << "!pto.mask<" << granularity << ">";
  return storage;
}

static LogicalResult verifyVRegTypeLike(Operation *op, Type type,
                                       StringRef roleDescription) {
  auto vecType = dyn_cast<VRegType>(type);
  if (!vecType)
    return op->emitOpError() << roleDescription << " must be !pto.vreg<...>";

  return VRegType::verify(
      [&]() { return op->emitOpError() << roleDescription << " "; },
      vecType.getElementCount(), vecType.getElementType());
}

static LogicalResult verifyMaskTypeLike(Operation *op, Type type,
                                        StringRef roleDescription) {
  if (!isa<MaskType>(type))
    return op->emitOpError() << roleDescription << " must be !pto.mask<...>";
  return success();
}

static LogicalResult verifyMaskTypeWithGranularityLike(Operation *op, Type type,
                                                       StringRef roleDescription,
                                                       StringRef granularity) {
  auto maskType = dyn_cast<MaskType>(type);
  if (!maskType)
    return op->emitOpError() << roleDescription << " must be !pto.mask<...>";
  if (maskType.getGranularity() != granularity) {
    return op->emitOpError()
           << roleDescription << " must be " << formatMaskType(granularity);
  }
  return success();
}

static LogicalResult verifyEnclosingLoopLike(Operation *op,
                                             StringRef opNameForDiag) {
  if (!op->getParentOfType<LoopLikeOpInterface>()) {
    return op->emitOpError()
           << "requires enclosing loop structure for " << opNameForDiag
           << " lowering";
  }
  return success();
}

static LogicalResult verifyNotNestedInVecScope(Operation *op,
                                               StringRef opNameForDiag) {
  if (op->getParentOfType<VecScopeOp>() ||
      op->getParentOfType<StrictVecScopeOp>()) {
    return op->emitOpError()
           << "must not be nested under pto.vecscope/pto.strict_vecscope; "
           << opNameForDiag << " is a UB helper op rather than a vecscope op";
  }
  return success();
}

static LogicalResult verifyNestedInVecScope(Operation *op,
                                            StringRef opNameForDiag) {
  if (op->getParentOfType<VecScopeOp>() || op->getParentOfType<StrictVecScopeOp>())
    return success();
  return op->emitOpError()
         << "must be nested under pto.vecscope/pto.strict_vecscope; "
         << opNameForDiag << " is part of the vecscope control sequence";
}

static LogicalResult verifyAlignTypeLike(Operation *op, Type type,
                                         StringRef roleDescription) {
  if (!isa<AlignType>(type))
    return op->emitOpError() << roleDescription << " must be !pto.align";
  return success();
}

static bool isSupportedVdupPosition(std::optional<StringRef> position) {
  return !position || *position == "LOWEST" || *position == "HIGHEST";
}

static std::optional<StringRef> getVdupMaskGranularity(Type elementType) {
  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    switch (intType.getWidth()) {
    case 8:
      return StringRef("b8");
    case 16:
      return StringRef("b16");
    case 32:
      return StringRef("b32");
    default:
      return std::nullopt;
    }
  }
  if (elementType.isF16() || elementType.isBF16())
    return StringRef("b16");
  if (elementType.isF32())
    return StringRef("b32");
  return std::nullopt;
}

static bool isStoreAlignProducer(Operation *op) {
  return isa<InitAlignOp, PstuOp, VstusOp, VsturOp>(op);
}

static bool isStoreAlignSink(Operation *op) {
  return isa<VstasOp, VstarOp>(op);
}

static bool isLoadAlignProducer(Operation *op) {
  return isa<VldasOp, VldusOp>(op);
}

static bool isValueOwnedByRegion(Value value, Region *region) {
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return blockArg.getParentRegion() == region;
  if (Operation *def = value.getDefiningOp())
    return def->getParentRegion() == region;
  return false;
}

static FailureOr<Value> resolveStoreAlignRoot(Value value, Operation *user) {
  llvm::SmallPtrSet<void *, 8> visited;
  Value current = value;

  while (true) {
    if (!visited.insert(current.getAsOpaquePointer()).second) {
      return failure();
    }

    if (auto blockArg = dyn_cast<BlockArgument>(current)) {
      auto *owner = blockArg.getOwner();
      auto forOp = dyn_cast<scf::ForOp>(owner->getParentOp());
      if (!forOp)
        return failure();
      unsigned argNumber = blockArg.getArgNumber();
      unsigned ivCount = forOp.getNumInductionVars();
      if (argNumber < ivCount)
        return failure();
      unsigned iterIdx = argNumber - ivCount;
      if (iterIdx >= forOp.getInitArgs().size())
        return failure();
      current = forOp.getInitArgs()[iterIdx];
      continue;
    }

    if (Operation *def = current.getDefiningOp()) {
      if (isStoreAlignProducer(def))
        return current;
      if (auto forOp = dyn_cast<scf::ForOp>(def)) {
        auto result = dyn_cast<OpResult>(current);
        if (!result)
          return failure();
        unsigned resultIdx = result.getResultNumber();
        if (resultIdx >= forOp.getYieldedValues().size())
          return failure();
        current = forOp.getYieldedValues()[resultIdx];
        continue;
      }
    }

    return failure();
  }
}

static LogicalResult verifyStoreAlignLoopThreading(Value align, Operation *user,
                                                   StringRef roleDescription) {
  Operation *cursor = user;
  while (auto forOp = cursor->getParentOfType<scf::ForOp>()) {
    Region *body = &forOp.getRegion();
    if (!isValueOwnedByRegion(align, body)) {
      return user->emitOpError()
             << roleDescription
             << " must be threaded through scf.for iter_args when used inside a "
                "loop";
    }
    cursor = forOp;
  }
  return success();
}

static LogicalResult verifyStoreAlignLinearUses(Value value, Operation *user) {
  llvm::SmallPtrSet<void *, 16> visited;
  Value current = value;

  while (visited.insert(current.getAsOpaquePointer()).second) {
    SmallVector<Value> nextValues;
    SmallVector<Operation *> terminalUsers;

    for (OpOperand &use : current.getUses()) {
      Operation *owner = use.getOwner();
      if (isStoreAlignSink(owner)) {
        terminalUsers.push_back(owner);
        continue;
      }
      if (auto stateOp = dyn_cast<PstuOp>(owner)) {
        nextValues.push_back(stateOp.getAlignOut());
        continue;
      }
      if (auto stateOp = dyn_cast<VstusOp>(owner)) {
        nextValues.push_back(stateOp.getAlignOut());
        continue;
      }
      if (auto stateOp = dyn_cast<VsturOp>(owner)) {
        nextValues.push_back(stateOp.getAlignOut());
        continue;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
        unsigned firstInitArg = forOp.getNumControlOperands();
        if (use.getOperandNumber() < firstInitArg)
          return user->emitOpError()
                 << "found unexpected scf.for control operand use for !pto.align";
        unsigned iterIdx = use.getOperandNumber() - firstInitArg;
        if (iterIdx >= forOp.getRegionIterArgs().size())
          return user->emitOpError()
                 << "found invalid scf.for iter_args use for !pto.align";
        nextValues.push_back(forOp.getRegionIterArgs()[iterIdx]);
        continue;
      }
      if (auto yieldOp = dyn_cast<scf::YieldOp>(owner)) {
        auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
        if (!forOp)
          return user->emitOpError()
                 << "found !pto.align yielded from non-scf.for loop";
        unsigned resultIdx = use.getOperandNumber();
        if (resultIdx >= forOp.getNumResults())
          return user->emitOpError()
                 << "found invalid scf.yield result mapping for !pto.align";
        nextValues.push_back(forOp.getResult(resultIdx));
        continue;
      }
      return user->emitOpError()
             << "found unsupported !pto.align consumer " << owner->getName();
    }

    if (nextValues.size() + terminalUsers.size() > 1) {
      return user->emitOpError()
             << "!pto.align value must form a single linear store-state chain";
    }
    if (nextValues.empty())
      return success();
    current = nextValues.front();
  }

  return success();
}

static LogicalResult verifyStoreAlignChain(Value align, Operation *user,
                                           StringRef roleDescription) {
  if (disableVPTOAlignChainVerification)
    return success();

  if (failed(verifyAlignTypeLike(user, align.getType(), roleDescription)))
    return failure();

  if (failed(verifyStoreAlignLoopThreading(align, user, roleDescription)))
    return failure();

  FailureOr<Value> root = resolveStoreAlignRoot(align, user);
  if (failed(root)) {
    if (Operation *def = align.getDefiningOp()) {
      if (!isa<scf::ForOp>(def)) {
        return user->emitOpError()
               << roleDescription
               << " must be produced by pto.init_align or a prior store-state op, got "
               << def->getName();
      }
    }
    return user->emitOpError()
           << roleDescription
           << " must be produced by pto.init_align or a prior store-state op";
  }

  Operation *def = (*root).getDefiningOp();
  if (!isStoreAlignProducer(def)) {
    return user->emitOpError()
           << roleDescription
           << " must be produced by pto.init_align or a prior store-state op, got "
           << def->getName();
  }

  return verifyStoreAlignLinearUses(*root, user);
}

static FailureOr<Value> resolveLoadAlignRoot(Value value, Operation *user) {
  llvm::SmallPtrSet<void *, 8> visited;
  Value current = value;

  while (true) {
    if (!visited.insert(current.getAsOpaquePointer()).second)
      return failure();

    if (auto blockArg = dyn_cast<BlockArgument>(current)) {
      auto *owner = blockArg.getOwner();
      auto forOp = dyn_cast<scf::ForOp>(owner->getParentOp());
      if (!forOp)
        return failure();
      unsigned argNumber = blockArg.getArgNumber();
      unsigned ivCount = forOp.getNumInductionVars();
      if (argNumber < ivCount)
        return failure();
      unsigned iterIdx = argNumber - ivCount;
      if (iterIdx >= forOp.getInitArgs().size())
        return failure();
      current = forOp.getInitArgs()[iterIdx];
      continue;
    }

    if (Operation *def = current.getDefiningOp()) {
      if (isLoadAlignProducer(def))
        return current;
      if (auto forOp = dyn_cast<scf::ForOp>(def)) {
        auto result = dyn_cast<OpResult>(current);
        if (!result)
          return failure();
        unsigned resultIdx = result.getResultNumber();
        if (resultIdx >= forOp.getYieldedValues().size())
          return failure();
        current = forOp.getYieldedValues()[resultIdx];
        continue;
      }
    }

    return failure();
  }
}

static LogicalResult verifyLoadAlignLoopThreading(Value align, Operation *user,
                                                  StringRef roleDescription) {
  Operation *cursor = user;
  while (auto forOp = cursor->getParentOfType<scf::ForOp>()) {
    Region *body = &forOp.getRegion();
    if (!isValueOwnedByRegion(align, body)) {
      return user->emitOpError()
             << roleDescription
             << " must be threaded through scf.for iter_args when used inside a "
                "loop";
    }
    cursor = forOp;
  }
  return success();
}

static LogicalResult verifyLoadAlignLinearUses(Value value, Operation *user) {
  llvm::SmallPtrSet<void *, 16> visited;
  Value current = value;

  while (visited.insert(current.getAsOpaquePointer()).second) {
    SmallVector<Value> nextValues;

    for (OpOperand &use : current.getUses()) {
      Operation *owner = use.getOwner();
      if (auto stateOp = dyn_cast<VldusOp>(owner)) {
        nextValues.push_back(stateOp.getUpdatedAlign());
        continue;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
        unsigned firstInitArg = forOp.getNumControlOperands();
        if (use.getOperandNumber() < firstInitArg) {
          return user->emitOpError()
                 << "found unexpected scf.for control operand use for !pto.align";
        }
        unsigned iterIdx = use.getOperandNumber() - firstInitArg;
        if (iterIdx >= forOp.getRegionIterArgs().size()) {
          return user->emitOpError()
                 << "found invalid scf.for iter_args use for !pto.align";
        }
        nextValues.push_back(forOp.getRegionIterArgs()[iterIdx]);
        continue;
      }
      if (auto yieldOp = dyn_cast<scf::YieldOp>(owner)) {
        auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
        if (!forOp) {
          return user->emitOpError()
                 << "found !pto.align yielded from non-scf.for loop";
        }
        unsigned resultIdx = use.getOperandNumber();
        if (resultIdx >= forOp.getNumResults()) {
          return user->emitOpError()
                 << "found invalid scf.yield result mapping for !pto.align";
        }
        nextValues.push_back(forOp.getResult(resultIdx));
        continue;
      }
      return user->emitOpError()
             << "found unsupported !pto.align consumer " << owner->getName();
    }

    if (nextValues.size() > 1) {
      return user->emitOpError()
             << "!pto.align value must form a single linear load-state chain";
    }
    if (nextValues.empty())
      return success();
    current = nextValues.front();
  }

  return success();
}

static LogicalResult verifyLoadAlignChain(Value align, Operation *user,
                                          StringRef roleDescription) {
  if (disableVPTOAlignChainVerification)
    return success();

  if (failed(verifyAlignTypeLike(user, align.getType(), roleDescription)))
    return failure();

  if (failed(verifyLoadAlignLoopThreading(align, user, roleDescription)))
    return failure();

  FailureOr<Value> root = resolveLoadAlignRoot(align, user);
  if (failed(root)) {
    if (Operation *def = align.getDefiningOp()) {
      if (!isa<scf::ForOp>(def)) {
        return user->emitOpError()
               << roleDescription
               << " must be produced by pto.vldas or a prior load-state op, got "
               << def->getName();
      }
    }
    return user->emitOpError()
           << roleDescription
           << " must be produced by pto.vldas or a prior load-state op";
  }

  Operation *def = (*root).getDefiningOp();
  if (!isLoadAlignProducer(def)) {
    return user->emitOpError()
           << roleDescription
           << " must be produced by pto.vldas or a prior load-state op, got "
           << def->getName();
  }

  return verifyLoadAlignLinearUses(*root, user);
}

static bool isSupportedPredicatePattern(StringRef pattern) {
  return pattern == "PAT_ALL" || pattern == "PAT_VL1" || pattern == "PAT_VL2" ||
         pattern == "PAT_VL3" || pattern == "PAT_VL4" || pattern == "PAT_VL8" ||
         pattern == "PAT_VL16" || pattern == "PAT_VL32" ||
         pattern == "PAT_VL64" || pattern == "PAT_VL128" ||
         pattern == "PAT_M3" || pattern == "PAT_M4" || pattern == "PAT_H" ||
         pattern == "PAT_Q" || pattern == "PAT_ALLF";
}

static bool isSupportedPredicateLoadDist(StringRef dist) {
  return dist == "NORM" || dist == "US" || dist == "DS";
}

static bool isSupportedPredicateStoreDist(StringRef dist) {
  return dist == "NORM" || dist == "PK";
}

static bool isSupportedStrideToken(StringRef stride) {
  return stride == "STRIDE_S3_B16" || stride == "STRIDE_S4_B64" ||
         stride == "STRIDE_S8_B32" || stride == "STRIDE_S2_B64" ||
         stride == "STRIDE_VSST_S8_B16";
}

static bool isSupportedPartToken(StringRef part) {
  return part == "LOWER" || part == "HIGHER";
}

static bool isSupportedSprToken(StringRef spr) { return spr == "AR"; }

static std::optional<StringRef> normalizeRoundModeToken(StringRef token) {
  if (token == "R" || token == "ROUND_R")
    return StringRef("R");
  if (token == "A" || token == "ROUND_A")
    return StringRef("A");
  if (token == "F" || token == "ROUND_F")
    return StringRef("F");
  if (token == "C" || token == "ROUND_C")
    return StringRef("C");
  if (token == "Z" || token == "ROUND_Z")
    return StringRef("Z");
  if (token == "O" || token == "ROUND_O")
    return StringRef("O");
  return std::nullopt;
}

static std::optional<StringRef> normalizeSaturationToken(StringRef token) {
  if (token == "SAT" || token == "RS_ENABLE")
    return StringRef("SAT");
  if (token == "NOSAT" || token == "RS_DISABLE")
    return StringRef("NOSAT");
  return std::nullopt;
}

static std::optional<StringRef> normalizeEvenOddPartToken(StringRef token) {
  if (token == "EVEN" || token == "PART_EVEN")
    return StringRef("EVEN");
  if (token == "ODD" || token == "PART_ODD")
    return StringRef("ODD");
  return std::nullopt;
}

namespace {

enum class VcvtElemKind {
  Invalid,
  F16,
  BF16,
  F32,
  S8,
  U8,
  S16,
  U16,
  S32,
  U32,
  S64,
};

struct VcvtContract {
  bool requiresRnd;
  bool requiresSat;
  bool requiresPart;
};

static VcvtElemKind classifyVcvtElemType(Type type) {
  if (type.isF16())
    return VcvtElemKind::F16;
  if (type.isBF16())
    return VcvtElemKind::BF16;
  if (type.isF32())
    return VcvtElemKind::F32;
  if (auto intType = dyn_cast<IntegerType>(type)) {
    switch (intType.getWidth()) {
    case 8:
      return intType.isUnsigned() ? VcvtElemKind::U8 : VcvtElemKind::S8;
    case 16:
      return intType.isUnsigned() ? VcvtElemKind::U16 : VcvtElemKind::S16;
    case 32:
      return intType.isUnsigned() ? VcvtElemKind::U32 : VcvtElemKind::S32;
    case 64:
      return intType.isUnsigned() ? VcvtElemKind::Invalid : VcvtElemKind::S64;
    default:
      return VcvtElemKind::Invalid;
    }
  }
  return VcvtElemKind::Invalid;
}

static std::optional<unsigned> getVcvtElemBitWidth(VcvtElemKind kind) {
  switch (kind) {
  case VcvtElemKind::F16:
  case VcvtElemKind::BF16:
  case VcvtElemKind::S16:
  case VcvtElemKind::U16:
    return 16;
  case VcvtElemKind::F32:
  case VcvtElemKind::S32:
  case VcvtElemKind::U32:
    return 32;
  case VcvtElemKind::S8:
  case VcvtElemKind::U8:
    return 8;
  case VcvtElemKind::S64:
    return 64;
  case VcvtElemKind::Invalid:
    return std::nullopt;
  }
  return std::nullopt;
}

static std::optional<VcvtContract> lookupVcvtContract(VcvtElemKind src,
                                                      VcvtElemKind dst) {
  switch (src) {
  case VcvtElemKind::F32:
    switch (dst) {
    case VcvtElemKind::F16:
    case VcvtElemKind::BF16:
    case VcvtElemKind::S16:
    case VcvtElemKind::S64:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/true,
                          /*requiresPart=*/true};
    case VcvtElemKind::S32:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/true,
                          /*requiresPart=*/false};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::F16:
    switch (dst) {
    case VcvtElemKind::F32:
    case VcvtElemKind::S32:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/false,
                          /*requiresPart=*/true};
    case VcvtElemKind::S16:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/true,
                          /*requiresPart=*/false};
    case VcvtElemKind::S8:
    case VcvtElemKind::U8:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/true,
                          /*requiresPart=*/true};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::BF16:
    switch (dst) {
    case VcvtElemKind::F32:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/false,
                          /*requiresPart=*/true};
    case VcvtElemKind::S32:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/true,
                          /*requiresPart=*/true};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::U8:
    switch (dst) {
    case VcvtElemKind::F16:
    case VcvtElemKind::U16:
    case VcvtElemKind::U32:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/false,
                          /*requiresPart=*/true};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::S8:
    switch (dst) {
    case VcvtElemKind::F16:
    case VcvtElemKind::S16:
    case VcvtElemKind::S32:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/false,
                          /*requiresPart=*/true};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::U16:
    switch (dst) {
    case VcvtElemKind::U8:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/true,
                          /*requiresPart=*/true};
    case VcvtElemKind::U32:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/false,
                          /*requiresPart=*/true};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::S16:
    switch (dst) {
    case VcvtElemKind::F16:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/false,
                          /*requiresPart=*/false};
    case VcvtElemKind::F32:
    case VcvtElemKind::U32:
    case VcvtElemKind::S32:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/false,
                          /*requiresPart=*/true};
    case VcvtElemKind::U8:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/true,
                          /*requiresPart=*/true};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::U32:
    switch (dst) {
    case VcvtElemKind::U8:
    case VcvtElemKind::U16:
    case VcvtElemKind::S16:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/true,
                          /*requiresPart=*/true};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::S32:
    switch (dst) {
    case VcvtElemKind::F32:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/false,
                          /*requiresPart=*/false};
    case VcvtElemKind::U8:
    case VcvtElemKind::U16:
    case VcvtElemKind::S16:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/true,
                          /*requiresPart=*/true};
    case VcvtElemKind::S64:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/false,
                          /*requiresPart=*/true};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::S64:
    switch (dst) {
    case VcvtElemKind::F32:
      return VcvtContract{/*requiresRnd=*/true, /*requiresSat=*/false,
                          /*requiresPart=*/true};
    case VcvtElemKind::S32:
      return VcvtContract{/*requiresRnd=*/false, /*requiresSat=*/true,
                          /*requiresPart=*/true};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::Invalid:
    return std::nullopt;
  }
  return std::nullopt;
}

} // namespace

static std::optional<unsigned> getDistElementWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();
  if (type.isF16() || type.isBF16())
    return 16;
  if (type.isF32())
    return 32;
  if (type.isF64())
    return 64;
  return std::nullopt;
}

static bool matchesWidthFamily(StringRef dist, unsigned width,
                               ArrayRef<unsigned> allowedWidths) {
  return llvm::is_contained(allowedWidths, width);
}

static bool isSupportedVldx2DistToken(StringRef dist) {
  return dist == "BDINTLV" || dist == "DINTLV";
}

static bool isSupportedVldsDistToken(StringRef dist) {
  return dist == "NORM" || dist == "BRC" || dist == "US" || dist == "DS" ||
         dist == "UNPK" || dist == "BRC_BLK" || dist == "E2B" ||
         dist == "UNPK4" || dist == "SPLT4CHN" || dist == "SPLT2CHN";
}

static bool isSupportedVstsDistToken(StringRef dist) {
  return dist == "NORM" || dist == "1PT" || dist == "PK" ||
         dist == "PK4" || dist == "MRG4CHN" || dist == "MRG2CHN";
}

static bool isSupportedVstsx2DistToken(StringRef dist) {
  return dist == "INTLV";
}

static LogicalResult verifyVldsDistWidth(Operation *op, StringRef dist,
                                         Type elementType) {
  auto width = getDistElementWidth(elementType);
  if (!width)
    return op->emitOpError("requires load element type with a concrete bit width");

  if (dist == "NORM" || dist == "BRC_BLK")
    return success();
  if (dist == "BRC")
    return matchesWidthFamily(dist, *width, {8, 16, 32})
               ? success()
               : op->emitOpError("dist BRC only supports 8/16/32-bit elements");
  if (dist == "US")
    return matchesWidthFamily(dist, *width, {8, 16})
               ? success()
               : op->emitOpError("dist US only supports 8/16-bit elements");
  if (dist == "DS")
    return matchesWidthFamily(dist, *width, {8, 16})
               ? success()
               : op->emitOpError("dist DS only supports 8/16-bit elements");
  if (dist == "UNPK")
    return matchesWidthFamily(dist, *width, {8, 16, 32})
               ? success()
               : op->emitOpError("dist UNPK only supports 8/16/32-bit elements");
  if (dist == "E2B")
    return matchesWidthFamily(dist, *width, {16, 32})
               ? success()
               : op->emitOpError("dist E2B only supports 16/32-bit elements");
  if (dist == "UNPK4")
    return *width == 8
               ? success()
               : op->emitOpError("dist UNPK4 only supports 8-bit elements");
  if (dist == "SPLT4CHN")
    return *width == 8
               ? success()
               : op->emitOpError("dist SPLT4CHN only supports 8-bit elements");
  if (dist == "SPLT2CHN")
    return matchesWidthFamily(dist, *width, {8, 16})
               ? success()
               : op->emitOpError("dist SPLT2CHN only supports 8/16-bit elements");

  return op->emitOpError("requires a supported load distribution token");
}

static LogicalResult verifyVldsx2DistWidth(Operation *op, StringRef dist,
                                           Type elementType) {
  auto width = getDistElementWidth(elementType);
  if (!width)
    return op->emitOpError(
        "requires x2 load element type with a concrete bit width");
  if (dist == "BDINTLV")
    return success();
  if (dist == "DINTLV")
    return matchesWidthFamily(dist, *width, {8, 16, 32})
               ? success()
               : op->emitOpError("dist DINTLV only supports 8/16/32-bit elements");
  return op->emitOpError("requires a supported x2 load distribution token");
}

static LogicalResult verifyVstsDistWidth(Operation *op, StringRef dist,
                                         Type elementType) {
  auto width = getDistElementWidth(elementType);
  if (!width)
    return op->emitOpError(
        "requires store element type with a concrete bit width");

  if (dist == "NORM")
    return matchesWidthFamily(dist, *width, {8, 16, 32})
               ? success()
               : op->emitOpError("dist NORM only supports 8/16/32-bit elements");
  if (dist == "1PT")
    return matchesWidthFamily(dist, *width, {8, 16, 32})
               ? success()
               : op->emitOpError("dist 1PT only supports 8/16/32-bit elements");
  if (dist == "PK")
    return matchesWidthFamily(dist, *width, {16, 32, 64})
               ? success()
               : op->emitOpError("dist PK only supports 16/32/64-bit elements");
  if (dist == "PK4")
    return *width == 32
               ? success()
               : op->emitOpError("dist PK4 only supports 32-bit elements");
  if (dist == "MRG4CHN")
    return *width == 8
               ? success()
               : op->emitOpError("dist MRG4CHN only supports 8-bit elements");
  if (dist == "MRG2CHN")
    return matchesWidthFamily(dist, *width, {8, 16})
               ? success()
               : op->emitOpError("dist MRG2CHN only supports 8/16-bit elements");

  return op->emitOpError("requires a supported store distribution token");
}

static LogicalResult verifyVstsx2DistWidth(Operation *op, StringRef dist,
                                           Type elementType) {
  auto width = getDistElementWidth(elementType);
  if (!width)
    return op->emitOpError(
        "requires x2 store element type with a concrete bit width");
  if (dist == "INTLV")
    return matchesWidthFamily(dist, *width, {8, 16, 32})
               ? success()
               : op->emitOpError("dist INTLV only supports 8/16/32-bit elements");
  return op->emitOpError("requires a supported x2 store distribution token");
}

static bool isSupportedPostMode(StringRef mode) {
  return mode == "NO_POST_UPDATE" || mode == "POST_UPDATE";
}

static std::optional<StringRef> getOptionalPostModeAttr(Operation *op) {
  if (auto mode = op->getAttrOfType<StringAttr>("mode"))
    return mode.getValue();
  return std::nullopt;
}

static unsigned getIntOrFloatBitWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();
  if (auto floatType = dyn_cast<FloatType>(type))
    return floatType.getWidth();
  return 0;
}

static bool isIntegerOrFloatLike(Type type) {
  return isa<IntegerType>(type) || isa<FloatType>(type);
}

static std::optional<int64_t> getVRegStorageBitWidth(Type type) {
  auto vecType = dyn_cast<VRegType>(type);
  if (!vecType)
    return std::nullopt;
  unsigned elemWidth = getIntOrFloatBitWidth(vecType.getElementType());
  if (!elemWidth)
    return std::nullopt;
  return vecType.getElementCount() * static_cast<int64_t>(elemWidth);
}

static LogicalResult verifyIntegerVRegTypeLike(Operation *op, Type type,
                                              StringRef roleDescription) {
  if (failed(verifyVRegTypeLike(op, type, roleDescription)))
    return failure();
  auto vecType = cast<VRegType>(type);
  if (!isa<IntegerType>(vecType.getElementType()))
    return op->emitOpError()
           << roleDescription << " must use integer vector element type";
  return success();
}

enum class MemoryRole {
  Unknown,
  GM,
  UB,
  Other,
};

static MemoryRole classifyMemoryRole(Type type) {
  auto memrefType = dyn_cast<BaseMemRefType>(type);
  if (!memrefType) {
    if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
      switch (ptrType.getMemorySpace().getAddressSpace()) {
      case pto::AddressSpace::GM:
      case pto::AddressSpace::Zero:
        return MemoryRole::GM;
      case pto::AddressSpace::VEC:
        return MemoryRole::UB;
      default:
        return MemoryRole::Other;
      }
    }
    return MemoryRole::Other;
  }

  Attribute memorySpace = memrefType.getMemorySpace();
  if (!memorySpace)
    return MemoryRole::Unknown;

  if (auto addrSpace = dyn_cast<pto::AddressSpaceAttr>(memorySpace)) {
    switch (addrSpace.getAddressSpace()) {
    case pto::AddressSpace::GM:
    case pto::AddressSpace::Zero:
      return MemoryRole::GM;
    case pto::AddressSpace::VEC:
      return MemoryRole::UB;
    default:
      return MemoryRole::Other;
    }
  }

  if (auto intAttr = dyn_cast<IntegerAttr>(memorySpace)) {
    switch (intAttr.getInt()) {
    case static_cast<int64_t>(pto::AddressSpace::GM):
    case static_cast<int64_t>(pto::AddressSpace::Zero):
      return MemoryRole::GM;
    case static_cast<int64_t>(pto::AddressSpace::VEC):
      return MemoryRole::UB;
    default:
      return MemoryRole::Other;
    }
  }

  return MemoryRole::Other;
}

static bool isBufferLike(Type type) {
  return isa<BaseMemRefType, pto::PtrType>(type);
}

static int64_t getPtrElementByteSize(Type type) {
  auto ptrType = dyn_cast<pto::PtrType>(type);
  if (!ptrType)
    return 0;

  Type elementType = ptrType.getElementType();
  if (auto floatType = dyn_cast<FloatType>(elementType))
    return (floatType.getWidth() + 7) / 8;
  if (auto intType = dyn_cast<IntegerType>(elementType))
    return (intType.getWidth() + 7) / 8;
  return 0;
}

template <typename CopyOp>
static LogicalResult verifyCopyGmToUbufOp(CopyOp op, bool expectSourceGM) {
  auto sourceType = dyn_cast<pto::PtrType>(op.getSource().getType());
  auto destinationType = dyn_cast<pto::PtrType>(op.getDestination().getType());
  if (!sourceType || !destinationType)
    return op.emitOpError("requires typed !pto.ptr source and destination");

  MemoryRole sourceRole = classifyMemoryRole(op.getSource().getType());
  MemoryRole destinationRole = classifyMemoryRole(op.getDestination().getType());
  bool directionMatches = true;
  if (expectSourceGM) {
    directionMatches &= sourceRole != MemoryRole::UB;
    directionMatches &= destinationRole != MemoryRole::GM;
  } else {
    directionMatches &= sourceRole != MemoryRole::GM;
    directionMatches &= destinationRole != MemoryRole::UB;
  }

  if (!directionMatches) {
    return op.emitOpError()
           << "requires "
           << (expectSourceGM ? "GM source and UB destination"
                              : "UB source and GM destination");
  }

  int64_t sourceElemBytes = getPtrElementByteSize(sourceType);
  int64_t destinationElemBytes = getPtrElementByteSize(destinationType);
  if (sourceElemBytes <= 0 || destinationElemBytes <= 0)
    return op.emitOpError("requires copy source and destination element types with known byte width");
  if (sourceElemBytes != destinationElemBytes)
    return op.emitOpError("requires source and destination element byte widths to match");

  return success();
}

template <typename CopyOp>
static LogicalResult verifyCopyUbufToGmOp(CopyOp op, bool expectSourceGM) {
  auto sourceType = dyn_cast<pto::PtrType>(op.getSource().getType());
  auto destinationType = dyn_cast<pto::PtrType>(op.getDestination().getType());
  if (!sourceType || !destinationType)
    return op.emitOpError("requires typed !pto.ptr source and destination");

  MemoryRole sourceRole = classifyMemoryRole(op.getSource().getType());
  MemoryRole destinationRole = classifyMemoryRole(op.getDestination().getType());
  bool directionMatches = true;
  if (expectSourceGM) {
    directionMatches &= sourceRole != MemoryRole::UB;
    directionMatches &= destinationRole != MemoryRole::GM;
  } else {
    directionMatches &= sourceRole != MemoryRole::GM;
    directionMatches &= destinationRole != MemoryRole::UB;
  }

  if (!directionMatches) {
    return op.emitOpError()
           << "requires "
           << (expectSourceGM ? "GM source and UB destination"
                              : "UB source and GM destination");
  }

  int64_t sourceElemBytes = getPtrElementByteSize(sourceType);
  int64_t destinationElemBytes = getPtrElementByteSize(destinationType);
  if (sourceElemBytes <= 0 || destinationElemBytes <= 0)
    return op.emitOpError("requires copy source and destination element types with known byte width");
  if (sourceElemBytes != destinationElemBytes)
    return op.emitOpError("requires source and destination element byte widths to match");

  return success();
}

Type VRegType::parse(AsmParser &parser) {
  SmallVector<int64_t, 1> shape;
  Type elementType;
  SMLoc loc = parser.getCurrentLocation();

  if (failed(parser.parseLess()) ||
      failed(parser.parseDimensionList(shape, /*allowDynamic=*/false,
                                       /*withTrailingX=*/true)) ||
      shape.size() != 1 || failed(parser.parseType(elementType)) ||
      failed(parser.parseGreater()))
    return {};

  return parser.getChecked<VRegType>(loc, parser.getContext(), shape.front(),
                                    elementType);
}

void VRegType::print(AsmPrinter &printer) const {
  printer << "<" << getElementCount() << "x";
  printer.printType(getElementType());
  printer << ">";
}

LogicalResult VRegType::verify(function_ref<InFlightDiagnostic()> emitError,
                              int64_t elementCount, Type elementType) {
  if (elementCount <= 0)
    return emitError() << "'" << formatVRegType(elementCount, elementType)
                       << "' expected a positive element count";

  auto intOrFloat = mlir::dyn_cast<IntegerType>(elementType);
  unsigned elementBitWidth = 0;
  if (intOrFloat) {
    elementBitWidth = intOrFloat.getWidth();
  } else if (auto floatType = mlir::dyn_cast<FloatType>(elementType)) {
    elementBitWidth = floatType.getWidth();
  } else {
    return emitError() << "'" << formatVRegType(elementCount, elementType)
                       << "' expected an integer or floating-point element type";
  }

  if (elementCount * static_cast<int64_t>(elementBitWidth) != 2048)
    return emitError() << "'" << formatVRegType(elementCount, elementType)
                       << "' expected exactly 256 bytes";

  return success();
}

LogicalResult VecScopeOp::verify() {
  Region &bodyRegion = getBody();
  if (bodyRegion.empty())
    return emitOpError("expects a non-empty body region");

  Block &body = bodyRegion.front();
  if (body.getNumArguments() != 0)
    return emitOpError() << "expects body block to have no arguments, got "
                         << body.getNumArguments();

  return success();
}

LogicalResult StrictVecScopeOp::verify() {
  Region &bodyRegion = getBody();
  if (bodyRegion.empty())
    return emitOpError("expects a non-empty body region");

  Block &body = bodyRegion.front();
  if (body.getNumArguments() != getCaptures().size())
    return emitOpError() << "expects body block to have "
                         << getCaptures().size()
                         << " arguments to match explicit captures, got "
                         << body.getNumArguments();

  for (auto [idx, pair] :
       llvm::enumerate(llvm::zip(body.getArguments(), getCaptures()))) {
    BlockArgument blockArg = std::get<0>(pair);
    Value capture = std::get<1>(pair);
    if (blockArg.getType() != capture.getType())
      return emitOpError() << "expects body block argument #" << idx
                           << " to have type " << capture.getType()
                           << ", got " << blockArg.getType();
  }
  return success();
}

bool MaskType::isSupportedGranularity(StringRef granularity) {
  return granularity == "b8" || granularity == "b16" ||
         granularity == "b32";
}

Type MaskType::parse(AsmParser &parser) {
  auto loc = parser.getCurrentLocation();
  StringRef granularity;
  if (failed(parser.parseLess()) || failed(parser.parseKeyword(&granularity)) ||
      failed(parser.parseGreater()))
    return {};

  return parser.getChecked<MaskType>(loc, parser.getContext(), granularity);
}

void MaskType::print(AsmPrinter &printer) const {
  printer << "<" << getGranularity() << ">";
}

LogicalResult
MaskType::verify(function_ref<InFlightDiagnostic()> emitError,
                 StringRef granularity) {
  if (!isSupportedGranularity(granularity))
    return emitError() << "'" << formatMaskType(granularity)
                       << "' expected granularity to be one of b8, b16, b32";
  return success();
}

void CopyGmToUbufOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult CopyGmToUbufOp::verify() {
  return verifyCopyGmToUbufOp(*this, true);
}

LogicalResult VbrOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getResult().getType(), "result")))
    return failure();

  auto resultVecType = cast<VRegType>(getResult().getType());
  Type elementType = getValue().getType();
  if (isa<ShapedType, VectorType>(elementType))
    return emitOpError("value must be a scalar matching the result element type");
  if (elementType != resultVecType.getElementType())
    return emitOpError("value type must match result element type");
  return success();
}

LogicalResult VcaddOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getInput().getType(), "input")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result")))
    return failure();
  if (getInput().getType() != getResult().getType())
    return emitOpError("input and result must have the same vector type");
  return success();
}

LogicalResult VcmaxOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getInput().getType(), "input")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result")))
    return failure();
  if (getInput().getType() != getResult().getType())
    return emitOpError("input and result must have the same vector type");
  return success();
}

LogicalResult VcminOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getInput().getType(), "input")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result")))
    return failure();
  if (getInput().getType() != getResult().getType())
    return emitOpError("input and result must have the same vector type");
  return success();
}

LogicalResult VciOp::verify() {
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be !pto.vreg<...>");
  if (!isa<IntegerType>(resultType.getElementType()))
    return emitOpError("result element type must be integer");
  auto indexType = dyn_cast<IntegerType>(getIndex().getType());
  if (!indexType)
    return emitOpError("index must be an integer scalar");
  if (indexType != resultType.getElementType())
    return emitOpError("index type must match result element type");
  return success();
}

void Vgather2Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult Vgather2Op::verify() {
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a pointer-like source");
  MemoryRole sourceRole = classifyMemoryRole(getSource().getType());
  if (sourceRole == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");

  auto offsetsType = dyn_cast<VRegType>(getOffsets().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!offsetsType || !resultType)
    return emitOpError("offsets and result must be !pto.vreg<...>");
  if (!isa<IntegerType>(offsetsType.getElementType()))
    return emitOpError("offset vector must use integer element type");
  if (offsetsType.getElementCount() != resultType.getElementCount())
    return emitOpError("offset and result vectors must have the same element count");
  if (!getActiveLanes().getType().isIndex())
    return emitOpError("active_lanes must be index");
  return success();
}

LogicalResult CopyUbufToUbufOp::verify() {
  if (!isBufferLike(getSource().getType()) || !isBufferLike(getDestination().getType()))
    return emitOpError("requires pointer-like source and destination");
  if (classifyMemoryRole(getSource().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getDestination().getType()) != MemoryRole::UB)
    return emitOpError("requires UB-backed source and destination");
  return success();
}

void VgatherbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VgatherbOp::verify() {
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a pointer-like source");
  MemoryRole sourceRole = classifyMemoryRole(getSource().getType());
  if (sourceRole == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");

  if (failed(verifyMaskTypeWithGranularityLike(getOperation(), getMask().getType(),
                                               "mask type", "b32")))
    return failure();

  auto offsetsType = dyn_cast<VRegType>(getOffsets().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!offsetsType || !resultType)
    return emitOpError("offsets and result must be !pto.vreg<...>");
  auto offsetsElemType = dyn_cast<IntegerType>(offsetsType.getElementType());
  if (!offsetsElemType)
    return emitOpError("offset vector must use integer element type");
  if (offsetsElemType.getWidth() != 32)
    return emitOpError("currently requires 32-bit offset vector elements");
  if (offsetsType.getElementCount() != resultType.getElementCount())
    return emitOpError("offset and result vectors must have the same element count");
  return success();
}

void Vgather2BcOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult Vgather2BcOp::verify() {
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a pointer-like source");
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");
  if (failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")))
    return failure();

  auto offsetsType = dyn_cast<VRegType>(getOffsets().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!offsetsType || !resultType)
    return emitOpError("offsets and result must be !pto.vreg<...>");
  auto offsetsElemType = dyn_cast<IntegerType>(offsetsType.getElementType());
  if (!offsetsElemType)
    return emitOpError("offset vector must use integer element type");
  if (offsetsElemType.getWidth() != 32)
    return emitOpError("currently requires 32-bit offset vector elements");
  if (offsetsType.getElementCount() != resultType.getElementCount())
    return emitOpError("offset and result vectors must have the same element count");
  return success();
}

LogicalResult VbitsortOp::verify() {
  if (!isBufferLike(getDestination().getType()) || !isBufferLike(getSource().getType()) ||
      !isBufferLike(getIndices().getType()))
    return emitOpError("requires pointer-like destination/source/indices");
  if (classifyMemoryRole(getDestination().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSource().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getIndices().getType()) != MemoryRole::UB)
    return emitOpError("requires UB-backed destination/source/indices");
  if (!getRepeatTimes().getType().isIndex())
    return emitOpError("repeat_times must be index");
  if (failed(verifyNotNestedInVecScope(*this, "pto.vbitsort")))
    return failure();
  return success();
}

void VbitsortOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getIndicesMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult Vmrgsort4Op::verify() {
  if (!isBufferLike(getDestination().getType()) || !isBufferLike(getSource0().getType()) ||
      !isBufferLike(getSource1().getType()) || !isBufferLike(getSource2().getType()) ||
      !isBufferLike(getSource3().getType()))
    return emitOpError("requires pointer-like destination and sources");
  if (classifyMemoryRole(getDestination().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSource0().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSource1().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSource2().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSource3().getType()) != MemoryRole::UB)
    return emitOpError("requires UB-backed destination and sources");
  return success();
}

LogicalResult VmaxOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getLhs().getType(), "lhs")) ||
      failed(verifyVRegTypeLike(*this, getRhs().getType(), "rhs")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result")))
    return failure();
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getResult().getType())
    return emitOpError("lhs, rhs, and result must have the same vector type");
  return success();
}

LogicalResult VminOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getLhs().getType(), "lhs")) ||
      failed(verifyVRegTypeLike(*this, getRhs().getType(), "rhs")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result")))
    return failure();
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getResult().getType())
    return emitOpError("lhs, rhs, and result must have the same vector type");
  return success();
}

void VldsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

template <typename LoadOp>
static LogicalResult verifyVldsCommon(LoadOp op) {
  if (!isBufferLike(op.getSource().getType()))
    return op.emitOpError("requires a pointer-like source");

  if (failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type")))
    return failure();

  MemoryRole sourceRole = classifyMemoryRole(op.getSource().getType());
  if (sourceRole == MemoryRole::GM)
    return op.emitOpError("requires a UB-backed source");

  if (op.getDistAttr()) {
    StringRef dist = *op.getDist();
    if (!isSupportedVldsDistToken(dist))
      return op.emitOpError(
          "supports only NORM, BRC, US, DS, UNPK, BRC_BLK, E2B, UNPK4, "
          "and SPLT2CHN/SPLT4CHN load distributions");
    if (failed(verifyVldsDistWidth(
            op.getOperation(), dist,
            cast<VRegType>(op.getResult().getType()).getElementType())))
      return failure();
  }

  return success();
}

LogicalResult VldsOp::verify() {
  if (failed(verifyVldsCommon(*this)))
    return failure();
  if (std::optional<StringRef> mode = getOptionalPostModeAttr(getOperation());
      mode && !isSupportedPostMode(*mode))
    return emitOpError("requires mode to be POST_UPDATE or NO_POST_UPDATE");
  return success();
}
void VldsPostOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VldsPostOp::verify() {
  if (failed(verifyVldsCommon(*this)))
    return failure();
  if (getUpdatedSource().getType() != getSource().getType())
    return emitOpError("requires updated source result to match source type");
  return success();
}

void VldasOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VldasOp::verify() {
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a pointer-like source");
  if (failed(verifyAlignTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");
  return success();
}

LogicalResult InitAlignOp::verify() {
  return verifyAlignTypeLike(*this, getResult().getType(), "result type");
}

LogicalResult SprclrOp::verify() {
  if (!isSupportedSprToken(getSpr()))
    return emitOpError("requires spr to be \"AR\"");
  if (failed(verifyNestedInVecScope(*this, "pto.sprclr")))
    return failure();
  return success();
}

void VldusOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VldusOp::verify() {
  if (failed(verifyLoadAlignChain(getAlign(), *this, "align type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type")) ||
      failed(verifyAlignTypeLike(*this, getUpdatedAlign().getType(),
                                 "updated align type")))
    return failure();
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a pointer-like source");
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");
  return success();
}

void UvldOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult UvldOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a buffer-like source");
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");

  auto sourceMemRef = dyn_cast<BaseMemRefType>(getSource().getType());
  if (!sourceMemRef)
    return success();

  Type sourceElementType = sourceMemRef.getElementType();
  Type vectorElementType = cast<VRegType>(getResult().getType()).getElementType();
  if (sourceElementType != vectorElementType)
    return emitOpError(
        "requires source element type to match vector element type");
  return success();
}

LogicalResult VdupOp::verify() {
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be !pto.vreg<...>");

  std::optional<StringRef> granularity =
      getVdupMaskGranularity(resultType.getElementType());
  if (!granularity)
    return emitOpError("result element type must use b8, b16, or b32 mask granularity");
  if (failed(verifyMaskTypeWithGranularityLike(
          getOperation(), getMask().getType(), "mask type", *granularity)))
    return failure();

  if (!isSupportedVdupPosition(getPosition()))
    return emitOpError("position must be LOWEST or HIGHEST");

  Type inputType = getInput().getType();
  if (auto inputVecType = dyn_cast<VRegType>(inputType)) {
    if (inputVecType != resultType)
      return emitOpError("vector input must match result vector type");
    return success();
  }

  if (getPosition())
    return emitOpError("position is only supported for vector input");

  if (inputType != resultType.getElementType())
    return emitOpError("scalar input must match result element type");

  return success();
}

LogicalResult PsetB8Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b8")))
    return failure();

  if (!isSupportedPredicatePattern(getPattern()))
    return emitOpError("requires a supported PAT_* predicate pattern");
  return success();
}

LogicalResult PsetB16Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b16")))
    return failure();

  if (!isSupportedPredicatePattern(getPattern()))
    return emitOpError("requires a supported PAT_* predicate pattern");
  return success();
}

LogicalResult PsetB32Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b32")))
    return failure();
  if (!isSupportedPredicatePattern(getPattern()))
    return emitOpError("requires a supported PAT_* predicate pattern");
  return success();
}

LogicalResult PgeB8Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b8")))
    return failure();
  if (!isSupportedPredicatePattern(getPattern()))
    return emitOpError("requires a supported PAT_* predicate pattern");
  return success();
}

LogicalResult PgeB16Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b16")))
    return failure();
  if (!isSupportedPredicatePattern(getPattern()))
    return emitOpError("requires a supported PAT_* predicate pattern");
  return success();
}

LogicalResult PgeB32Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b32")))
    return failure();
  if (!isSupportedPredicatePattern(getPattern()))
    return emitOpError("requires a supported PAT_* predicate pattern");
  return success();
}

template <typename PltOp>
static LogicalResult verifyPredicateLaneCountOp(PltOp op,
                                                StringRef granularity) {
  if (failed(verifyMaskTypeWithGranularityLike(op, op.getMask().getType(),
                                               "mask type", granularity)))
    return failure();
  Type scalarType = op.getScalar().getType();
  auto scalarIntType = dyn_cast<IntegerType>(scalarType);
  if (!scalarIntType || scalarIntType.getWidth() != 32)
    return op.emitOpError("requires scalar to be i32");
  if (op.getScalarOut().getType() != scalarType)
    return op.emitOpError("requires scalar_out to match scalar type");
  return success();
}

LogicalResult PltB8Op::verify() { return verifyPredicateLaneCountOp(*this, "b8"); }
LogicalResult PltB16Op::verify() {
  return verifyPredicateLaneCountOp(*this, "b16");
}
LogicalResult PltB32Op::verify() {
  return verifyPredicateLaneCountOp(*this, "b32");
}

LogicalResult PpackOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getInput().getType(), "input type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (getPart() != "LOWER")
    return emitOpError("currently supports only LOWER part");
  return success();
}

LogicalResult PunpackOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getInput().getType(), "input type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (getPart() != "LOWER")
    return emitOpError("currently supports only LOWER part");
  return success();
}

LogicalResult PnotOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getInput().getType(), "input type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  return success();
}

LogicalResult PselOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getSrc0().getType(), "src0 type")) ||
      failed(verifyMaskTypeLike(*this, getSrc1().getType(), "src1 type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  return success();
}

template <typename BinaryMaskOp>
static LogicalResult verifyBinaryMaskOp(BinaryMaskOp op) {
  if (failed(verifyMaskTypeLike(op, op.getSrc0().getType(), "src0 type")) ||
      failed(verifyMaskTypeLike(op, op.getSrc1().getType(), "src1 type")) ||
      failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")) ||
      failed(verifyMaskTypeLike(op, op.getResult().getType(), "result type")))
    return failure();
  return success();
}

LogicalResult PandOp::verify() { return verifyBinaryMaskOp(*this); }
LogicalResult PorOp::verify() { return verifyBinaryMaskOp(*this); }
LogicalResult PxorOp::verify() { return verifyBinaryMaskOp(*this); }

void PldsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult PldsOp::verify() {
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a pointer-like source");
  if (failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  MemoryRole sourceRole = classifyMemoryRole(getSource().getType());
  if (sourceRole == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");
  if (!getOffset().getType().isIndex())
    return emitOpError("requires index offset");
  if (!isSupportedPredicateLoadDist(getDist()))
    return emitOpError("requires predicate load dist to be NORM, US, or DS");
  if (failed(verifyEnclosingLoopLike(*this, "pto.plds")))
    return failure();
  return success();
}

void PldiOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult PldiOp::verify() {
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a pointer-like source");
  if (failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");
  if (!matchPattern(getOffset(), m_Constant()))
    return emitOpError("requires offset to be a constant index immediate");
  if (!isSupportedPredicateLoadDist(getDist()))
    return emitOpError("requires predicate load dist to be NORM, US, or DS");
  if (failed(verifyEnclosingLoopLike(*this, "pto.pldi")))
    return failure();
  return success();
}

template <typename OpTy>
static LogicalResult verifyElementwiseVecScalarOpLike(OpTy op) {
  auto inputType = dyn_cast<VRegType>(op.getInput().getType());
  auto resultType = dyn_cast<VRegType>(op.getResult().getType());
  if (!inputType || !resultType)
    return op.emitOpError("input and result must be !pto.vreg<...>");
  if (inputType != resultType)
    return op.emitOpError("input and result vector types must match");

  Type elemType = inputType.getElementType();
  Type scalarType = op.getScalar().getType();
  if (scalarType == elemType)
    return success();

  auto elemInt = dyn_cast<IntegerType>(elemType);
  auto scalarInt = dyn_cast<IntegerType>(scalarType);
  if (!elemInt || !scalarInt || elemInt.getWidth() != scalarInt.getWidth())
    return op.emitOpError("scalar type must match vector element type");

  if (elemInt.isSigned() && (scalarInt.isSigned() || scalarInt.isSignless()))
    return success();
  if (elemInt.isUnsigned() &&
      (scalarInt.isUnsigned() || scalarInt.isSignless()))
    return success();
  if (elemInt.isSignless() && scalarInt.isSignless())
    return success();

  return op.emitOpError(
      "integer scalar type must match vector element width and use matching signedness or signless i<width>");
}

template <typename OpTy>
static LogicalResult verifyVecScalarOpLike(OpTy op) {
  if (failed(verifyElementwiseVecScalarOpLike(op)))
    return failure();
  return success();
}

template <typename OpTy>
static LogicalResult verifySignedSaturatingVecScalarOpLike(OpTy op) {
  if (failed(verifyElementwiseVecScalarOpLike(op)))
    return failure();
  auto inputType = cast<VRegType>(op.getInput().getType());
  auto elemType = dyn_cast<IntegerType>(inputType.getElementType());
  if (!elemType || elemType.isUnsigned() || elemType.getWidth() != 16)
    return op.emitOpError("requires s16 vector element type");
  return success();
}

template <typename OpTy>
static LogicalResult verifyVecScalarMaskedOpLike(OpTy op) {
  if (failed(verifyElementwiseVecScalarOpLike(op)))
    return failure();
  if (failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")))
    return failure();
  return success();
}

template <typename CarryOp>
static LogicalResult verifyCarryVecOp(CarryOp op) {
  if (failed(verifyIntegerVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyIntegerVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")) ||
      failed(verifyIntegerVRegTypeLike(op, op.getResult().getType(),
                                      "result type")) ||
      failed(verifyMaskTypeLike(op, op.getCarry().getType(), "carry type")))
    return failure();

  auto lhsType = cast<VRegType>(op.getLhs().getType());
  auto rhsType = cast<VRegType>(op.getRhs().getType());
  auto resultType = cast<VRegType>(op.getResult().getType());
  auto lhsElemType = cast<IntegerType>(lhsType.getElementType());
  if (lhsType != rhsType || lhsType != resultType)
    return op.emitOpError("requires lhs, rhs, and result to have matching vector types");
  if (lhsElemType.getWidth() != 32)
    return op.emitOpError("currently requires 32-bit integer vector elements");
  return success();
}

template <typename CarryWithInputOp>
static LogicalResult verifyCarryVecOpWithInput(CarryWithInputOp op) {
  if (failed(verifyCarryVecOp(op)) ||
      failed(verifyMaskTypeLike(op, op.getCarryIn().getType(),
                                "carry_in type")))
    return failure();
  return success();
}

LogicalResult VmulsOp::verify() { return verifyVecScalarMaskedOpLike(*this); }
LogicalResult VaddsOp::verify() { return verifyVecScalarMaskedOpLike(*this); }
LogicalResult VsaddsOp::verify() {
  if (failed(verifySignedSaturatingVecScalarOpLike(*this)))
    return failure();
  return verifyMaskTypeLike(*this, getMask().getType(), "mask type");
}
LogicalResult VmaxsOp::verify() { return verifyVecScalarMaskedOpLike(*this); }
LogicalResult VminsOp::verify() { return verifyVecScalarMaskedOpLike(*this); }
LogicalResult VlreluOp::verify() { return verifyVecScalarMaskedOpLike(*this); }
LogicalResult VshlsOp::verify() {
  auto inputType = dyn_cast<VRegType>(getInput().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!inputType || !resultType)
    return emitOpError("input and result must be !pto.vreg<...>");
  if (failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")))
    return failure();
  if (inputType != resultType)
    return emitOpError("input and result vector types must match");
  if (!isa<IntegerType>(inputType.getElementType()))
    return emitOpError("requires integer vector and integer scalar");
  auto scalarType = dyn_cast<IntegerType>(getScalar().getType());
  if (!scalarType || !scalarType.isSignlessInteger(16))
    return emitOpError("requires signless i16 scalar");
  return success();
}
LogicalResult VshrsOp::verify() {
  auto inputType = dyn_cast<VRegType>(getInput().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!inputType || !resultType)
    return emitOpError("input and result must be !pto.vreg<...>");
  if (failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")))
    return failure();
  if (inputType != resultType)
    return emitOpError("input and result vector types must match");
  if (!isa<IntegerType>(inputType.getElementType()))
    return emitOpError("requires integer vector and integer scalar");
  auto scalarType = dyn_cast<IntegerType>(getScalar().getType());
  if (!scalarType || !scalarType.isSignlessInteger(16))
    return emitOpError("requires signless i16 scalar");
  return success();
}

LogicalResult VabsOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getInput().getType(), "operand type")))
    return failure();
  if (failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")))
    return failure();
  if (failed(verifyVRegTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (getInput().getType() != getResult().getType())
    return emitOpError("requires matching register vector shape");
  return success();
}

template <typename UnaryOp>
static LogicalResult verifyUnaryVecOp(UnaryOp op) {
  if (failed(verifyVRegTypeLike(op, op.getInput().getType(), "operand type")))
    return failure();
  if (failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")))
    return failure();
  if (failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type")))
    return failure();
  if (op.getInput().getType() != op.getResult().getType())
    return op.emitOpError("requires matching register vector shape");
  return success();
}

LogicalResult VexpOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VlnOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VsqrtOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VnegOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VrsqrtOp::verify() {
  if (failed(verifyUnaryVecOp(*this)))
    return failure();
  auto inputType = cast<VRegType>(getInput().getType());
  Type elemType = inputType.getElementType();
  if (!elemType.isF16() && !elemType.isF32())
    return emitOpError("requires f16 or f32 vector element type");
  return success();
}
LogicalResult VrecOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VreluOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VnotOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VbcntOp::verify() {
  if (failed(verifyUnaryVecOp(*this)))
    return failure();
  auto inputType = cast<VRegType>(getInput().getType());
  if (!isa<IntegerType>(inputType.getElementType()))
    return emitOpError("requires integer vector element type");
  return success();
}
LogicalResult VclsOp::verify() {
  if (failed(verifyUnaryVecOp(*this)))
    return failure();
  auto inputType = cast<VRegType>(getInput().getType());
  if (!isa<IntegerType>(inputType.getElementType()))
    return emitOpError("requires integer vector element type");
  return success();
}

template <typename BinaryOp>
static LogicalResult verifyBinaryVecOp(BinaryOp op) {
  if (failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")))
    return failure();
  if (failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")))
    return failure();
  if (failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")))
    return failure();
  if (failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type")))
    return failure();
  if (op.getLhs().getType() != op.getRhs().getType() ||
      op.getLhs().getType() != op.getResult().getType())
    return op.emitOpError("requires matching register vector shapes");
  return success();
}

template <typename BinaryOp>
static LogicalResult verifySignedSaturatingBinaryVecOp(BinaryOp op) {
  if (failed(verifyBinaryVecOp(op)))
    return failure();
  auto lhsType = cast<VRegType>(op.getLhs().getType());
  auto elemType = dyn_cast<IntegerType>(lhsType.getElementType());
  if (!elemType || elemType.isUnsigned() || elemType.getWidth() != 16)
    return op.emitOpError("requires s16 vector element type");
  return success();
}

LogicalResult VaddOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VsubOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VsaddOp::verify() {
  return verifySignedSaturatingBinaryVecOp(*this);
}
LogicalResult VssubOp::verify() {
  return verifySignedSaturatingBinaryVecOp(*this);
}
LogicalResult VmulOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VdivOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VandOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VorOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VxorOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VshlOp::verify() {
  if (failed(verifyBinaryVecOp(*this)))
    return failure();
  auto lhsType = cast<VRegType>(getLhs().getType());
  if (!isa<IntegerType>(lhsType.getElementType()))
    return emitOpError("requires integer vector element type");
  return success();
}
LogicalResult VshrOp::verify() {
  if (failed(verifyBinaryVecOp(*this)))
    return failure();
  auto lhsType = cast<VRegType>(getLhs().getType());
  if (!isa<IntegerType>(lhsType.getElementType()))
    return emitOpError("requires integer vector element type");
  return success();
}
LogicalResult VaddcOp::verify() { return verifyCarryVecOp(*this); }
LogicalResult VsubcOp::verify() { return verifyCarryVecOp(*this); }
LogicalResult VaddcsOp::verify() { return verifyCarryVecOpWithInput(*this); }
LogicalResult VsubcsOp::verify() { return verifyCarryVecOpWithInput(*this); }

template <typename ReductionOp>
static LogicalResult verifyReductionVecOp(ReductionOp op) {
  return verifyUnaryVecOp(op);
}

template <typename ReductionOp>
static LogicalResult verifyGroupReductionVecOp(ReductionOp op) {
  if (failed(verifyReductionVecOp(op)))
    return failure();
  auto inputType = cast<VRegType>(op.getInput().getType());
  Type elemType = inputType.getElementType();
  if (auto intType = dyn_cast<IntegerType>(elemType)) {
    if (intType.getWidth() < 16 || intType.getWidth() > 32)
      return op.emitOpError(
          "requires 16-bit or 32-bit integer vector element type");
    return success();
  }
  if (!elemType.isF16() && !elemType.isF32())
    return op.emitOpError("requires i16/i32/f16/f32 vector element type");
  return success();
}

LogicalResult VcgaddOp::verify() { return verifyGroupReductionVecOp(*this); }
LogicalResult VcgmaxOp::verify() { return verifyGroupReductionVecOp(*this); }
LogicalResult VcgminOp::verify() { return verifyGroupReductionVecOp(*this); }
LogicalResult VcpaddOp::verify() {
  if (failed(verifyReductionVecOp(*this)))
    return failure();
  auto inputType = cast<VRegType>(getInput().getType());
  Type elemType = inputType.getElementType();
  if (!elemType.isF16() && !elemType.isF32())
    return emitOpError("requires f16 or f32 vector element type");
  return success();
}

template <typename SelectOp>
static LogicalResult verifyLaneSelectOp(SelectOp op) {
  if (failed(verifyVRegTypeLike(op, op.getSrc0().getType(), "src0 type")) ||
      failed(verifyVRegTypeLike(op, op.getSrc1().getType(), "src1 type")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type")))
    return failure();

  auto src0Type = cast<VRegType>(op.getSrc0().getType());
  auto src1Type = cast<VRegType>(op.getSrc1().getType());
  auto resultType = cast<VRegType>(op.getResult().getType());
  if (src0Type != resultType)
    return op.emitOpError("requires src0 and result to have identical vector types");
  if (src1Type.getElementCount() != src0Type.getElementCount())
    return op.emitOpError("requires src0/src1 to have identical element counts");
  auto src1ElemType = dyn_cast<IntegerType>(src1Type.getElementType());
  if (!src1ElemType)
    return op.emitOpError("requires src1 to use integer vector elements");
  if (src1ElemType.getWidth() != getIntOrFloatBitWidth(src0Type.getElementType()))
    return op.emitOpError("requires src1 integer element width to match src0 element width");
  return success();
}

template <typename PairOp>
static LogicalResult verifyPairVecResults(PairOp op) {
  if (failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyVRegTypeLike(op, op.getLow().getType(), "low result type")) ||
      failed(verifyVRegTypeLike(op, op.getHigh().getType(), "high result type")))
    return failure();
  if (op.getLhs().getType() != op.getRhs().getType() ||
      op.getLhs().getType() != op.getLow().getType() ||
      op.getLhs().getType() != op.getHigh().getType())
    return op.emitOpError("requires operands and results to share one vector type");
  return success();
}

template <typename PartOp>
static LogicalResult verifyPartVecOp(PartOp op) {
  if (failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type")))
    return failure();
  if (op.getLhs().getType() != op.getRhs().getType() ||
      op.getLhs().getType() != op.getResult().getType())
    return op.emitOpError("requires operands and result to share one vector type");
  if (!isSupportedPartToken(op.getPart()))
    return op.emitOpError("requires part to be LOWER or HIGHER");
  return success();
}

LogicalResult VselOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc0().getType(), "src0 type")) ||
      failed(verifyVRegTypeLike(*this, getSrc1().getType(), "src1 type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (getSrc0().getType() != getSrc1().getType() ||
      getSrc0().getType() != getResult().getType())
    return emitOpError("requires src0, src1, and result to have identical vector types");
  return success();
}

LogicalResult VselrOp::verify() { return verifyLaneSelectOp(*this); }
LogicalResult Vselrv2Op::verify() { return verifyLaneSelectOp(*this); }

LogicalResult VslideOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc0().getType(), "src0 type")) ||
      failed(verifyVRegTypeLike(*this, getSrc1().getType(), "src1 type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (getSrc0().getType() != getSrc1().getType() ||
      getSrc0().getType() != getResult().getType())
    return emitOpError("requires src0, src1, and result to share one vector type");
  return success();
}

LogicalResult VsqzOp::verify() { return verifyUnaryVecOp(*this); }

LogicalResult VusqzOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc().getType(), "src type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (getSrc().getType() != getResult().getType())
    return emitOpError("requires src and result to share one vector type");
  auto srcType = cast<VRegType>(getSrc().getType());
  auto elemType = dyn_cast<IntegerType>(srcType.getElementType());
  if (!elemType)
    return emitOpError("requires signed integer vector element type");
  if (elemType.isUnsigned())
    return emitOpError("requires signed integer vector element type");
  unsigned width = elemType.getWidth();
  if (width != 8 && width != 16 && width != 32)
    return emitOpError("requires s8/s16/s32 vector element type");
  return success();
}

LogicalResult VpackOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc().getType(), "src type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (!isSupportedPartToken(getPart()))
    return emitOpError("requires part to be LOWER or HIGHER");
  auto srcType = cast<VRegType>(getSrc().getType());
  auto resultType = cast<VRegType>(getResult().getType());
  Type srcElemType = srcType.getElementType();
  Type resultElemType = resultType.getElementType();
  if (!isa<IntegerType>(srcElemType) || !isa<IntegerType>(resultElemType))
    return emitOpError("currently requires integer source and result element types");
  if (resultType.getElementCount() != srcType.getElementCount() * 2)
    return emitOpError(
        "requires result element count to be twice the source element count");
  unsigned srcWidth = getIntOrFloatBitWidth(srcElemType);
  unsigned resultWidth = getIntOrFloatBitWidth(resultElemType);
  if (!srcWidth || resultWidth * 2 != srcWidth)
    return emitOpError(
        "requires result element width to be half the source element width");
  auto srcIntType = cast<IntegerType>(srcElemType);
  auto resultIntType = cast<IntegerType>(resultElemType);
  if (!resultIntType.isUnsigned())
    return emitOpError("requires unsigned result element type");
  if (!((srcIntType.getWidth() == 32 && resultIntType.getWidth() == 16) ||
        (srcIntType.getWidth() == 16 && resultIntType.getWidth() == 8)))
    return emitOpError(
        "currently supports only s32/u32 -> u16 and s16/u16 -> u8");
  return success();
}

template <typename UnpackOp>
static LogicalResult verifyUnpackVecOp(UnpackOp op) {
  if (failed(verifyVRegTypeLike(op, op.getSrc().getType(), "src type")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type")))
    return failure();
  auto srcType = cast<VRegType>(op.getSrc().getType());
  auto resultType = cast<VRegType>(op.getResult().getType());
  Type srcElemType = srcType.getElementType();
  Type resultElemType = resultType.getElementType();
  if (!isa<IntegerType>(srcElemType) || !isa<IntegerType>(resultElemType))
    return op.emitOpError(
        "currently requires integer source and result element types");
  if (srcType.getElementCount() != resultType.getElementCount() * 2)
    return op.emitOpError(
        "requires source element count to be twice the result element count");
  unsigned srcWidth = getIntOrFloatBitWidth(srcElemType);
  unsigned resultWidth = getIntOrFloatBitWidth(resultElemType);
  if (!srcWidth || srcWidth * 2 != resultWidth)
    return op.emitOpError(
        "requires result element width to be twice the source element width");
  return success();
}

LogicalResult VsunpackOp::verify() { return verifyUnpackVecOp(*this); }
LogicalResult VzunpackOp::verify() { return verifyUnpackVecOp(*this); }

static bool isSupportedCmpMode(StringRef mode) {
  return mode == "eq" || mode == "ne" || mode == "lt" || mode == "le" ||
         mode == "gt" || mode == "ge";
}

LogicalResult VcmpOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc0().getType(), "src0 type")) ||
      failed(verifyVRegTypeLike(*this, getSrc1().getType(), "src1 type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (getSrc0().getType() != getSrc1().getType())
    return emitOpError("requires src0 and src1 to have identical vector types");
  if (!isSupportedCmpMode(getCmpMode()))
    return emitOpError("requires cmp_mode to be one of eq/ne/lt/le/gt/ge");
  return success();
}

LogicalResult VcmpsOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc().getType(), "src type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  auto srcType = cast<VRegType>(getSrc().getType());
  if (getScalar().getType() != srcType.getElementType())
    return emitOpError("requires scalar type to match source element type");
  if (!isSupportedCmpMode(getCmpMode()))
    return emitOpError("requires cmp_mode to be one of eq/ne/lt/le/gt/ge");
  return success();
}

ParseResult VtrcOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  std::string roundModeToken;
  NamedAttrList attrs;
  Type inputType, resultType;

  if (parser.parseOperand(input) || parser.parseComma() ||
      parser.parseKeywordOrString(&roundModeToken) ||
      parser.parseOptionalAttrDict(attrs) ||
      parser.parseColonType(inputType) || parser.parseArrow() ||
      parser.parseType(resultType))
    return failure();

  auto normalized = normalizeRoundModeToken(roundModeToken);
  if (!normalized)
    return parser.emitError(parser.getCurrentLocation())
           << "round mode must be one of R/A/F/C/Z/O or "
              "ROUND_R/ROUND_A/ROUND_F/ROUND_C/ROUND_Z/ROUND_O";

  attrs.set("round_mode", parser.getBuilder().getStringAttr(*normalized));
  result.addAttributes(attrs);
  if (parser.resolveOperand(input, inputType, result.operands))
    return failure();
  result.addTypes(resultType);
  return success();
}

void VtrcOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getInput() << ", ";
  Builder builder(getContext());
  auto normalized = normalizeRoundModeToken(getRoundMode());
  printer.printAttributeWithoutType(
      builder.getStringAttr(normalized.value_or(getRoundMode())));
  printer.printOptionalAttrDict((*this)->getAttrs(), {"round_mode"});
  printer << " : " << getInput().getType() << " -> " << getResult().getType();
}

LogicalResult VtrcOp::verify() {
  auto inputType = dyn_cast<VRegType>(getInput().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!inputType || !resultType)
    return emitOpError("input and result must be !pto.vreg<...>");
  if (inputType != resultType)
    return emitOpError("requires input and result to have identical vreg type");
  if (!normalizeRoundModeToken(getRoundMode()))
    return emitOpError("round mode must be one of R/A/F/C/Z/O");
  return success();
}

ParseResult VcvtOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  NamedAttrList attrs;
  Type inputType, resultType;

  if (parser.parseOperand(input) || parser.parseOptionalAttrDict(attrs) ||
      parser.parseColonType(inputType) || parser.parseArrow() ||
      parser.parseType(resultType))
    return failure();

  Attribute legacyRndAttr = attrs.get("round_mode");
  Attribute rndAttr = attrs.get("rnd");
  if (legacyRndAttr && rndAttr)
    return parser.emitError(parser.getCurrentLocation())
           << "rnd and round_mode cannot be specified together";

  auto normalizeNamedStringAttr =
      [&](StringRef sourceName, StringRef canonicalName,
          auto normalizeFn) -> ParseResult {
    Attribute rawAttr = attrs.get(sourceName);
    if (!rawAttr)
      return success();
    auto strAttr = dyn_cast<StringAttr>(rawAttr);
    if (!strAttr)
      return parser.emitError(parser.getCurrentLocation())
             << sourceName << " must be a string literal";
    auto normalized = normalizeFn(strAttr.getValue());
    if (!normalized)
      return parser.emitError(parser.getCurrentLocation())
             << sourceName << " has unsupported value '" << strAttr.getValue()
             << "'";
    attrs.erase(sourceName);
    attrs.set(canonicalName, parser.getBuilder().getStringAttr(*normalized));
    return success();
  };

  if (failed(normalizeNamedStringAttr("round_mode", "rnd",
                                      normalizeRoundModeToken)) ||
      failed(normalizeNamedStringAttr("rnd", "rnd", normalizeRoundModeToken)) ||
      failed(normalizeNamedStringAttr("sat", "sat", normalizeSaturationToken)) ||
      failed(
          normalizeNamedStringAttr("part", "part", normalizeEvenOddPartToken)))
    return failure();

  result.addAttributes(attrs);
  if (parser.resolveOperand(input, inputType, result.operands))
    return failure();
  result.addTypes(resultType);
  return success();
}

void VcvtOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getInput();
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getInput().getType() << " -> " << getResult().getType();
}

LogicalResult VcvtOp::verify() {
  auto inputType = dyn_cast<VRegType>(getInput().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!inputType || !resultType)
    return emitOpError("input and result must be !pto.vreg<...>");

  VcvtElemKind inputElemKind = classifyVcvtElemType(inputType.getElementType());
  VcvtElemKind resultElemKind = classifyVcvtElemType(resultType.getElementType());
  auto contract = lookupVcvtContract(inputElemKind, resultElemKind);
  if (!contract)
    return emitOpError("unsupported vcvt source/result element type pair");

  auto inputElemBits = getVcvtElemBitWidth(inputElemKind);
  auto resultElemBits = getVcvtElemBitWidth(resultElemKind);
  if (!inputElemBits || !resultElemBits)
    return emitOpError("could not determine vcvt element bit width");
  if (inputType.getElementCount() * static_cast<int64_t>(*inputElemBits) !=
      resultType.getElementCount() * static_cast<int64_t>(*resultElemBits)) {
    return emitOpError("requires source and result vectors to carry the same "
                       "total number of bits");
  }

  if (getRndAttr()) {
    StringRef roundMode = *getRnd();
    if (!normalizeRoundModeToken(roundMode))
      return emitOpError("rnd must be one of R/A/F/C/Z/O");
  }
  if (static_cast<bool>(getRndAttr()) != contract->requiresRnd) {
    return contract->requiresRnd ? emitOpError("requires rnd attr for this vcvt type pair")
                                 : emitOpError("rnd attr is not valid for this vcvt type pair");
  }

  if (getSatAttr()) {
    StringRef sat = *getSat();
    if (!normalizeSaturationToken(sat))
      return emitOpError("sat must be SAT or NOSAT");
  }
  if (static_cast<bool>(getSatAttr()) != contract->requiresSat) {
    return contract->requiresSat ? emitOpError("requires sat attr for this vcvt type pair")
                                 : emitOpError("sat attr is not valid for this vcvt type pair");
  }

  if (getPartAttr()) {
    StringRef part = *getPart();
    if (!normalizeEvenOddPartToken(part))
      return emitOpError("part must be EVEN or ODD");
  }
  if (static_cast<bool>(getPartAttr()) != contract->requiresPart) {
    return contract->requiresPart ? emitOpError("requires part attr for this vcvt type pair")
                                  : emitOpError("part attr is not valid for this vcvt type pair");
  }

  return success();
}

LogicalResult PdintlvB8Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getLhs().getType(),
                                               "lhs type", "b8")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getRhs().getType(),
                                               "rhs type", "b8")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getLow().getType(),
                                               "low type", "b8")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getHigh().getType(),
                                               "high type", "b8")))
    return failure();
  return success();
}

LogicalResult PdintlvB16Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getLhs().getType(),
                                               "lhs type", "b16")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getRhs().getType(),
                                               "rhs type", "b16")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getLow().getType(),
                                               "low type", "b16")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getHigh().getType(),
                                               "high type", "b16")))
    return failure();
  return success();
}

LogicalResult PdintlvB32Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getLhs().getType(),
                                               "lhs type", "b32")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getRhs().getType(),
                                               "rhs type", "b32")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getLow().getType(),
                                               "low type", "b32")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getHigh().getType(),
                                               "high type", "b32")))
    return failure();
  return success();
}

LogicalResult PintlvB8Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getLhs().getType(),
                                               "lhs type", "b8")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getRhs().getType(),
                                               "rhs type", "b8")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getLow().getType(),
                                               "low type", "b8")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getHigh().getType(),
                                               "high type", "b8")))
    return failure();
  return success();
}

LogicalResult PintlvB16Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getLhs().getType(),
                                               "lhs type", "b16")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getRhs().getType(),
                                               "rhs type", "b16")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getLow().getType(),
                                               "low type", "b16")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getHigh().getType(),
                                               "high type", "b16")))
    return failure();
  return success();
}

LogicalResult PintlvB32Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getLhs().getType(),
                                               "lhs type", "b32")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getRhs().getType(),
                                               "rhs type", "b32")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getLow().getType(),
                                               "low type", "b32")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getHigh().getType(),
                                               "high type", "b32")))
    return failure();
  return success();
}

LogicalResult VintlvOp::verify() { return verifyPairVecResults(*this); }
LogicalResult VdintlvOp::verify() { return verifyPairVecResults(*this); }
LogicalResult Vintlvv2Op::verify() { return verifyPartVecOp(*this); }
LogicalResult Vdintlvv2Op::verify() { return verifyPartVecOp(*this); }

LogicalResult VmullOp::verify() {
  if (failed(verifyPairVecResults(*this)) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")))
    return failure();
  auto lhsType = cast<VRegType>(getLhs().getType());
  auto lhsElemType = dyn_cast<IntegerType>(lhsType.getElementType());
  if (!lhsElemType)
    return emitOpError("requires integer vector element type");
  if (lhsElemType.getWidth() != 32)
    return emitOpError("currently requires 32-bit integer vector elements");
  return success();
}

LogicalResult VmulaOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getAcc().getType(), "acc type")) ||
      failed(verifyVRegTypeLike(*this, getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(*this, getRhs().getType(), "rhs type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (getAcc().getType() != getLhs().getType() ||
      getAcc().getType() != getRhs().getType() ||
      getAcc().getType() != getResult().getType())
    return emitOpError("requires acc, lhs, rhs, and result to share one vector type");
  return success();
}

template <typename BinaryVecNoMaskOp>
static LogicalResult verifyBinaryVecNoMaskOp(BinaryVecNoMaskOp op) {
  if (failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type")))
    return failure();
  if (op.getLhs().getType() != op.getRhs().getType() ||
      op.getLhs().getType() != op.getResult().getType())
    return op.emitOpError("requires lhs, rhs, and result to share one vector type");
  return success();
}

template <typename BinaryVecNoMaskOp>
static LogicalResult verifyFloatBinaryVecNoMaskOp(BinaryVecNoMaskOp op) {
  if (failed(verifyBinaryVecNoMaskOp(op)))
    return failure();
  auto lhsType = cast<VRegType>(op.getLhs().getType());
  Type elemType = lhsType.getElementType();
  if (!elemType.isF16() && !elemType.isF32())
    return op.emitOpError("requires f16 or f32 vector element type");
  return success();
}

LogicalResult VpreluOp::verify() { return verifyFloatBinaryVecNoMaskOp(*this); }
LogicalResult VexpdiffOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getInput().getType(), "input type")) ||
      failed(verifyVRegTypeLike(*this, getMax().getType(), "max type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type")))
    return failure();

  auto inputType = cast<VRegType>(getInput().getType());
  auto maxType = cast<VRegType>(getMax().getType());
  auto resultType = cast<VRegType>(getResult().getType());
  if (inputType != maxType)
    return emitOpError("requires input and max to share one vector type");

  Type inputElemType = inputType.getElementType();
  if (!inputElemType.isF16() && !inputElemType.isF32())
    return emitOpError("requires f16 or f32 input vector element type");
  if (!resultType.getElementType().isF32())
    return emitOpError("requires f32 result vector element type");

  auto inputBits = getVRegStorageBitWidth(inputType);
  auto resultBits = getVRegStorageBitWidth(resultType);
  if (!inputBits || !resultBits || *inputBits != *resultBits)
    return emitOpError(
        "requires source and result to preserve total vector storage width");

  StringRef part = getPart();
  if (part != "EVEN" && part != "ODD")
    return emitOpError("part must be EVEN or ODD");
  return success();
}

LogicalResult VaxpyOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc0().getType(), "src0 type")) ||
      failed(verifyVRegTypeLike(*this, getSrc1().getType(), "src1 type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  auto src0Type = cast<VRegType>(getSrc0().getType());
  auto src1Type = cast<VRegType>(getSrc1().getType());
  auto resultType = cast<VRegType>(getResult().getType());
  if (src0Type != src1Type || src0Type != resultType)
    return emitOpError("requires src0, src1, and result to share one vector type");
  Type elemType = src0Type.getElementType();
  if (!elemType.isF16() && !elemType.isF32())
    return emitOpError("requires f16 or f32 vector element type");
  if (getAlpha().getType() != elemType)
    return emitOpError("requires alpha type to match vector element type");
  return success();
}

template <typename ConvOp>
static LogicalResult verifyFusedConvVecOp(ConvOp op) {
  if (failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type")))
    return failure();
  auto lhsType = cast<VRegType>(op.getLhs().getType());
  auto rhsType = cast<VRegType>(op.getRhs().getType());
  auto resultType = cast<VRegType>(op.getResult().getType());
  if (lhsType != rhsType)
    return op.emitOpError("requires lhs and rhs to share one vector type");
  if (!isIntegerOrFloatLike(lhsType.getElementType()) ||
      !isIntegerOrFloatLike(resultType.getElementType()))
    return op.emitOpError(
        "requires integer or floating-point vector element types");
  auto lhsBits = getVRegStorageBitWidth(lhsType);
  auto resultBits = getVRegStorageBitWidth(resultType);
  if (!lhsBits || !resultBits || *lhsBits != *resultBits)
    return op.emitOpError(
        "requires source and result to preserve total vector storage width");
  return success();
}

LogicalResult VaddreluconvOp::verify() {
  return verifyFusedConvVecOp(*this);
}
LogicalResult VmulconvOp::verify() { return verifyFusedConvVecOp(*this); }

void Vldsx2Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult Vldsx2Op::verify() {
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a pointer-like source");
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");
  if (!getOffset().getType().isIndex())
    return emitOpError("requires index offset");
  if (failed(verifyVRegTypeLike(*this, getLow().getType(), "low result type")) ||
      failed(verifyVRegTypeLike(*this, getHigh().getType(), "high result type")))
    return failure();
  if (getLow().getType() != getHigh().getType())
    return emitOpError("requires low/high results to share one vector type");
  if (!isSupportedVldx2DistToken(getDist()))
    return emitOpError("requires a supported x2 load distribution token");
  if (failed(verifyVldsx2DistWidth(
          getOperation(), getDist(),
          cast<VRegType>(getLow().getType()).getElementType())))
    return failure();
  return success();
}

void VstsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

template <typename StoreOp>
static LogicalResult verifyVstsCommon(StoreOp op) {
  if (failed(verifyVRegTypeLike(op, op.getValue().getType(), "value type")))
    return failure();
  if (failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")))
    return failure();

  if (!isBufferLike(op.getDestination().getType()))
    return op.emitOpError("requires a pointer-like destination");

  MemoryRole destinationRole = classifyMemoryRole(op.getDestination().getType());
  if (destinationRole == MemoryRole::GM)
    return op.emitOpError("requires a UB-backed destination");

  if (std::optional<StringRef> dist = op.getDist();
      dist && !isSupportedVstsDistToken(*dist)) {
    return op.emitOpError("requires a supported store distribution token");
  }
  if (std::optional<StringRef> dist = op.getDist();
      dist &&
      failed(verifyVstsDistWidth(
          op.getOperation(), *dist,
          cast<VRegType>(op.getValue().getType()).getElementType())))
    return failure();

  return success();
}

LogicalResult VstsOp::verify() {
  if (failed(verifyVstsCommon(*this)))
    return failure();
  if (std::optional<StringRef> mode = getOptionalPostModeAttr(getOperation());
      mode && !isSupportedPostMode(*mode))
    return emitOpError("requires mode to be POST_UPDATE or NO_POST_UPDATE");
  return success();
}
void VstsPostOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VstsPostOp::verify() {
  if (failed(verifyVstsCommon(*this)))
    return failure();
  if (getUpdatedDestination().getType() != getDestination().getType())
    return emitOpError(
        "requires updated destination result to match destination type");
  return success();
}

void Vstsx2Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getLowMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getHighMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult Vstsx2Op::verify() {
  if (failed(verifyVRegTypeLike(*this, getLow().getType(), "low value type")) ||
      failed(verifyVRegTypeLike(*this, getHigh().getType(), "high value type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")))
    return failure();
  if (getLow().getType() != getHigh().getType())
    return emitOpError("requires low/high values to share one vector type");
  if (!isBufferLike(getDestination().getType()))
    return emitOpError("requires a pointer-like destination");
  if (classifyMemoryRole(getDestination().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed destination");
  if (!getOffset().getType().isIndex())
    return emitOpError("requires index offset");
  if (!isSupportedVstsx2DistToken(getDist()))
    return emitOpError("requires a supported x2 store distribution token");
  if (failed(verifyVstsx2DistWidth(
          getOperation(), getDist(),
          cast<VRegType>(getLow().getType()).getElementType())))
    return failure();
  return success();
}

void VscatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VscatterOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getValue().getType(), "value type")))
    return failure();
  if (!isBufferLike(getDestination().getType()))
    return emitOpError("requires a pointer-like destination");
  auto offsetsType = dyn_cast<VRegType>(getOffsets().getType());
  auto valueType = dyn_cast<VRegType>(getValue().getType());
  if (!offsetsType || !valueType)
    return emitOpError("value and offsets must be !pto.vreg<...>");
  auto offsetsElemType = dyn_cast<IntegerType>(offsetsType.getElementType());
  if (!offsetsElemType)
    return emitOpError("offset vector must use integer element type");
  if (offsetsElemType.getWidth() != 32)
    return emitOpError("currently requires 32-bit offset vector elements");
  if (offsetsType.getElementCount() != valueType.getElementCount())
    return emitOpError("offset and value vectors must have the same element count");
  MemoryRole destinationRole = classifyMemoryRole(getDestination().getType());
  if (destinationRole == MemoryRole::GM)
    return emitOpError("requires a UB-backed destination");
  if (!getActiveLanes().getType().isIndex())
    return emitOpError("active_lanes must be index");
  return success();
}

void VsldbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VsldbOp::verify() {
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a pointer-like source");
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");
  if (failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (!getBlockStride().getType().isSignlessInteger(16))
    return emitOpError("requires block_stride to be i16");
  if (!getRepeatStride().getType().isSignlessInteger(16))
    return emitOpError("requires repeat_stride to be i16");
  return success();
}

void PstsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

void PstiOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult PstiOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getValue().getType(), "value type")))
    return failure();
  if (!isBufferLike(getDestination().getType()))
    return emitOpError("requires a pointer-like destination");
  if (classifyMemoryRole(getDestination().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed destination");
  if (!matchPattern(getOffset(), m_Constant()))
    return emitOpError("requires offset to be a constant index immediate");
  if (!isSupportedPredicateStoreDist(getDist()))
    return emitOpError("requires predicate store dist to be NORM or PK");
  return success();
}

LogicalResult PstsOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getValue().getType(), "value type")))
    return failure();
  if (!isBufferLike(getDestination().getType()))
    return emitOpError("requires a pointer-like destination");
  MemoryRole destinationRole = classifyMemoryRole(getDestination().getType());
  if (destinationRole == MemoryRole::GM)
    return emitOpError("requires a UB-backed destination");
  if (!getOffset().getType().isIndex())
    return emitOpError("requires index offset");
  if (!isSupportedPredicateStoreDist(getDist()))
    return emitOpError("requires predicate store dist to be NORM or PK");
  return success();
}

void VsstbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VsstbOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getValue().getType(), "value type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")))
    return failure();
  if (!isBufferLike(getDestination().getType()))
    return emitOpError("requires a pointer-like destination");
  if (classifyMemoryRole(getDestination().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed destination");
  if (!getBlockStride().getType().isSignlessInteger(16))
    return emitOpError("requires block_stride to be i16");
  if (!getRepeatStride().getType().isSignlessInteger(16))
    return emitOpError("requires repeat_stride to be i16");
  return success();
}

void VstasOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VstasOp::verify() {
  if (failed(verifyStoreAlignChain(getValue(), *this, "value type")))
    return failure();
  if (!isBufferLike(getDestination().getType()))
    return emitOpError("requires a pointer-like destination");
  if (classifyMemoryRole(getDestination().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed destination");
  return success();
}

void VstarOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VstarOp::verify() {
  if (failed(verifyStoreAlignChain(getValue(), *this, "value type")))
    return failure();
  if (!isBufferLike(getDestination().getType()))
    return emitOpError("requires a pointer-like destination");
  if (classifyMemoryRole(getDestination().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed destination");
  return success();
}

void PstuOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getAlignInMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getBaseMutable());
}

LogicalResult PstuOp::verify() {
  if (failed(verifyStoreAlignChain(getAlignIn(), *this, "align_in type")) ||
      failed(verifyMaskTypeLike(*this, getValue().getType(), "value type")) ||
      failed(verifyAlignTypeLike(*this, getAlignOut().getType(), "align_out type")))
    return failure();
  if (!isBufferLike(getBase().getType()) || !isBufferLike(getBaseOut().getType()))
    return emitOpError("requires pointer-like base and base_out");
  if (getBase().getType() != getBaseOut().getType())
    return emitOpError("requires base and base_out to have identical types");
  if (classifyMemoryRole(getBase().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed base");
  auto baseType = cast<pto::PtrType>(getBase().getType());
  auto maskType = cast<pto::MaskType>(getValue().getType());
  auto elemType = dyn_cast<IntegerType>(baseType.getElementType());
  if (!elemType || elemType.isSigned() || (elemType.getWidth() != 16 && elemType.getWidth() != 32))
    return emitOpError("requires ui16/ui32 UB base type");
  if (maskType.isB16() && elemType.getWidth() != 16)
    return emitOpError("requires !pto.mask<b16> to pair with !pto.ptr<ui16, ub>");
  if (maskType.isB32() && elemType.getWidth() != 32)
    return emitOpError("requires !pto.mask<b32> to pair with !pto.ptr<ui32, ub>");
  return success();
}

void VstusOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getAlignInMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getBaseMutable());
}

LogicalResult VstusOp::verify() {
  if (failed(verifyStoreAlignChain(getAlignIn(), *this, "align_in type")) ||
      failed(verifyVRegTypeLike(*this, getValue().getType(), "value type")) ||
      failed(verifyAlignTypeLike(*this, getAlignOut().getType(), "align_out type")))
    return failure();
  if (!isBufferLike(getBase().getType()))
    return emitOpError("requires a pointer-like base");
  if (classifyMemoryRole(getBase().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed base");
  return success();
}

void VsturOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getAlignInMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getBaseMutable());
}

LogicalResult VsturOp::verify() {
  if (failed(verifyStoreAlignChain(getAlignIn(), *this, "align_in type")) ||
      failed(verifyVRegTypeLike(*this, getValue().getType(), "value type")) ||
      failed(verifyAlignTypeLike(*this, getAlignOut().getType(), "align_out type")))
    return failure();
  if (!isBufferLike(getBase().getType()))
    return emitOpError("requires a pointer-like base");
  if (classifyMemoryRole(getBase().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed base");
  if (!isSupportedPostMode(getMode()))
    return emitOpError("requires mode to be POST_UPDATE or NO_POST_UPDATE");
  return success();
}

void CopyUbufToGmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult CopyUbufToGmOp::verify() {
  return verifyCopyUbufToGmOp(*this, false);
}
