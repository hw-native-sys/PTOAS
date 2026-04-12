#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VPTOLowering.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOVPTOPTRBOUNDARY
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static Type convertVPTOBoundaryMemRefType(Type type) {
  auto memrefType = dyn_cast<BaseMemRefType>(type);
  if (!memrefType)
    return type;
  auto memorySpace =
      dyn_cast_or_null<pto::AddressSpaceAttr>(memrefType.getMemorySpace());
  if (!memorySpace)
    return {};
  return pto::PtrType::get(type.getContext(), memrefType.getElementType(),
                           memorySpace);
}

static bool isTrivialVPTOBoundaryCastPtr(pto::CastPtrOp castOp) {
  return castOp.getInput().getType() == castOp.getResult().getType();
}

static LogicalResult eraseDeadVPTOMemRefScaffold(ModuleOp module) {
  bool erasedAny = true;
  while (erasedAny) {
    erasedAny = false;
    SmallVector<pto::CastPtrOp> trivialCasts;
    SmallVector<Operation *> deadOps;
    module.walk([&](Operation *op) {
      if (auto castOp = dyn_cast<pto::CastPtrOp>(op)) {
        if (isTrivialVPTOBoundaryCastPtr(castOp)) {
          trivialCasts.push_back(castOp);
          return;
        }
        if (castOp->use_empty())
          deadOps.push_back(op);
        return;
      }

      if (!op->use_empty())
        return;
      if (isa<pto::PointerCastOp, pto::BindTileOp, memref::ReinterpretCastOp,
              memref::SubViewOp, memref::MemorySpaceCastOp>(op))
        deadOps.push_back(op);
    });

    for (pto::CastPtrOp castOp : trivialCasts) {
      if (!castOp->getBlock())
        continue;
      castOp.getResult().replaceAllUsesWith(castOp.getInput());
      castOp.erase();
      erasedAny = true;
    }

    for (Operation *op : deadOps) {
      if (!op->getBlock())
        continue;
      op->erase();
      erasedAny = true;
    }
  }
  return success();
}

static Type getVPTOBufferElementType(Value value) {
  Type type = value.getType();
  if (auto tileType = dyn_cast<pto::TileBufType>(type))
    return tileType.getElementType();
  if (auto memrefType = dyn_cast<BaseMemRefType>(type))
    return memrefType.getElementType();
  if (auto ptrType = dyn_cast<pto::PtrType>(type))
    return ptrType.getElementType();
  return {};
}

static Attribute getVPTOBufferMemorySpace(Value value) {
  Type type = value.getType();
  if (auto tileType = dyn_cast<pto::TileBufType>(type))
    return tileType.getMemorySpace();
  if (auto memrefType = dyn_cast<BaseMemRefType>(type))
    return memrefType.getMemorySpace();
  if (auto ptrType = dyn_cast<pto::PtrType>(type))
    return ptrType.getMemorySpace();
  return {};
}

static bool needsPtrCanonicalization(Value value) {
  return isa<BaseMemRefType, pto::TileBufType>(value.getType());
}

static bool isSupportedVPTOBufferLikeBoundaryOp(Operation *op) {
  return isa<pto::VldsOp, pto::UvldOp, pto::PldsOp, pto::PldiOp, pto::VstsOp,
             pto::PstsOp, pto::PstiOp, pto::Vldsx2Op, pto::Vstsx2Op,
             pto::VsldbOp, pto::VsstbOp, pto::VstasOp, pto::VstarOp,
             pto::LoadScalarOp, pto::StoreScalarOp>(op);
}

static LogicalResult canonicalizeBoundaryCastPtrOps(ModuleOp module,
                                                    llvm::raw_ostream *diagOS) {
  SmallVector<pto::CastPtrOp> castsToRewrite;
  module.walk([&](pto::CastPtrOp castOp) {
    if (!isa<BaseMemRefType, pto::TileBufType>(castOp.getInput().getType()))
      return;
    if (!isa<pto::PtrType>(castOp.getResult().getType()))
      return;
    castsToRewrite.push_back(castOp);
  });

  PatternRewriter rewriter(module.getContext());
  for (pto::CastPtrOp castOp : castsToRewrite) {
    if (!castOp->getBlock())
      continue;

    auto resultType = dyn_cast<pto::PtrType>(castOp.getResult().getType());
    if (!resultType)
      continue;

    rewriter.setInsertionPoint(castOp);
    Value ptrValue = pto::materializeBufferPointer(
        castOp.getInput(), resultType.getElementType(),
        resultType.getMemorySpace(), rewriter, castOp.getLoc());
    if (!ptrValue) {
      if (diagOS) {
        *diagOS << "VPTO emission-boundary ptr rewrite failed: could not "
                   "canonicalize pto.castptr input for ";
        castOp->print(*diagOS);
        *diagOS << "\n";
      }
      return failure();
    }

    castOp.getResult().replaceAllUsesWith(ptrValue);
    rewriter.eraseOp(castOp);
  }

  return success();
}

static LogicalResult canonicalizeSupportedVPTOBufferLikeOps(
    ModuleOp module, llvm::raw_ostream *diagOS) {
  SmallVector<Operation *> opsToRewrite;
  module.walk([&](Operation *op) {
    if (isSupportedVPTOBufferLikeBoundaryOp(op))
      opsToRewrite.push_back(op);
  });

  PatternRewriter rewriter(module.getContext());
  for (Operation *op : opsToRewrite) {
    rewriter.setInsertionPoint(op);

    SmallVector<Value> newOperands;
    newOperands.reserve(op->getNumOperands());
    bool changed = false;

    for (Value operand : op->getOperands()) {
      if (!needsPtrCanonicalization(operand)) {
        newOperands.push_back(operand);
        continue;
      }

      Type elementType = getVPTOBufferElementType(operand);
      Attribute memorySpace = getVPTOBufferMemorySpace(operand);
      if (!elementType || !memorySpace) {
        if (diagOS) {
          *diagOS << "VPTO emission-boundary ptr rewrite failed: could not "
                     "derive element type or memory space for operand of ";
          op->print(*diagOS);
          *diagOS << "\n";
        }
        return failure();
      }

      Value ptrValue = pto::materializeBufferPointer(operand, elementType,
                                                     memorySpace, rewriter,
                                                     op->getLoc());
      if (!ptrValue) {
        if (diagOS) {
          *diagOS << "VPTO emission-boundary ptr rewrite failed: could not "
                     "materialize pointer operand for ";
          op->print(*diagOS);
          *diagOS << "\n";
        }
        return failure();
      }

      changed = changed || (ptrValue != operand);
      newOperands.push_back(ptrValue);
    }

    if (!changed)
      continue;

    OperationState state(op->getLoc(), op->getName().getStringRef());
    state.addOperands(newOperands);
    state.addTypes(op->getResultTypes());
    state.addAttributes(op->getAttrs());

    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
  }

  return success();
}

struct PTOVPTOPtrBoundaryPass
    : public pto::impl::PTOVPTOPtrBoundaryBase<PTOVPTOPtrBoundaryPass> {
  using pto::impl::PTOVPTOPtrBoundaryBase<
      PTOVPTOPtrBoundaryPass>::PTOVPTOPtrBoundaryBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(pto::convertVPTOEmissionBoundaryToPtr(module, &llvm::errs())))
      signalPassFailure();
  }
};

} // namespace

LogicalResult mlir::pto::convertVPTOEmissionBoundaryToPtr(
    ModuleOp module, llvm::raw_ostream *diagOS) {
  // VPTO kernels use ptr-only entry semantics at the emission boundary: the
  // function ABI keeps only the same-space base pointer, while shape/stride
  // state remains in SSA. Body-level op canonicalization is added on top of
  // this entry rewrite in follow-up tasks.
  if (failed(eraseDeadVPTOMemRefScaffold(module)))
    return failure();

  bool sawFailure = false;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (func.isExternal())
      continue;

    FunctionType functionType = func.getFunctionType();
    SmallVector<Type> newInputs(functionType.getInputs().begin(),
                                functionType.getInputs().end());
    bool changed = false;

    for (auto [idx, inputType] : llvm::enumerate(functionType.getInputs())) {
      auto memrefType = dyn_cast<BaseMemRefType>(inputType);
      if (!memrefType)
        continue;

      Type newType = convertVPTOBoundaryMemRefType(inputType);
      if (!newType) {
        if (diagOS)
          *diagOS << "VPTO emission-boundary ptr rewrite failed: unsupported "
                     "memref argument type in "
                  << func.getName() << ": " << inputType << "\n";
        sawFailure = true;
        continue;
      }

      BlockArgument arg = func.getArgument(idx);
      SmallVector<Operation *> users(arg.getUsers().begin(), arg.getUsers().end());
      arg.setType(newType);
      newInputs[idx] = newType;
      changed = true;

      for (Operation *user : users) {
        if (auto cast = dyn_cast<CastPtrOp>(user)) {
          if (cast.getInput() != arg)
            continue;
          if (cast.getResult().getType() == newType) {
            cast.getResult().replaceAllUsesWith(arg);
            cast.erase();
          }
          continue;
        }

        if (isa<memref::ReinterpretCastOp, memref::SubViewOp,
                memref::MemorySpaceCastOp>(user) &&
            user->use_empty()) {
          user->erase();
          continue;
        }

        if (isSupportedVPTOBufferLikeBoundaryOp(user))
          continue;

        if (diagOS) {
          *diagOS << "VPTO emission-boundary ptr rewrite failed: argument "
                  << idx << " of " << func.getName()
                  << " still feeds a memref-dependent user after ptr rewrite:\n";
          user->print(*diagOS);
          *diagOS << "\n";
        }
        sawFailure = true;
      }
    }

    for (Type resultType : functionType.getResults()) {
      if (!isa<BaseMemRefType>(resultType))
        continue;
      if (diagOS)
        *diagOS << "VPTO emission-boundary ptr rewrite failed: memref result "
                   "is unsupported for "
                << func.getName() << ": " << resultType << "\n";
      sawFailure = true;
    }

    if (changed) {
      func.setFunctionType(
          FunctionType::get(module.getContext(), newInputs, functionType.getResults()));
    }
  }

  if (sawFailure)
    return failure();

  if (failed(canonicalizeBoundaryCastPtrOps(module, diagOS)))
    return failure();

  if (failed(canonicalizeSupportedVPTOBufferLikeOps(module, diagOS)))
    return failure();

  return eraseDeadVPTOMemRefScaffold(module);
}

std::unique_ptr<Pass> mlir::pto::createPTOVPTOPtrBoundaryPass() {
  return std::make_unique<PTOVPTOPtrBoundaryPass>();
}
