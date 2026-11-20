// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIMaskGranularityAssignment.cpp - Assign VMI mask granularity -----===//
//===----------------------------------------------------------------------===//
//
// This pass assigns one concrete b8/b16/b32 granularity to every VMI mask value
// before layout assignment.  It deliberately does not choose layouts: mask
// layout is assigned later by vmi-layout-assignment.
//
// The pass is a three phase solver built around one invariant: granularity is a
// property of a mask value (of its equivalence class), never of a use.  A use
// only files a request; it must never rewrite the granularity of the value it
// consumes.  Yields, iter_args, call boundaries and ensure_mask_layout are
// identity relations and therefore do not create a decision either.
//
//   1. Collect.  Equivalence classes are built exclusively from identity
//      relations.  Producer-side requirements are the only hard constraints: a
//      compare pins its result to the granularity of the compared element width
//      and a carry result is b32.  Every other mask use files a per-use request.
//      The mask form of pto.vmi.mask_and/or/xor/not and
//      pto.vmi.vand/vor/vxor/vnot only records that its operands and result must
//      share one granularity; it does not decide that granularity.
//   2. Solve.  A class with an inherent requirement uses it, and two disagreeing
//      requirements are a VMI-LAYOUT-CONTRACT error.  An unconstrained mask
//      logic result may inherit the granularity its operands already agree on,
//      an unconstrained histogram/scatter predicate keeps its consumer
//      granularity, and every other class falls back to b32.  A mask logic group
//      runs at the granularity of its result's class, so a shared root mask
//      stays logical and each use adapts instead of forcing one width on it.
//   3. Materialize.  Rewrite the mask types of every class, then convert every
//      use whose request does not match the value: cheap mask producers are
//      rematerialized at the use, everything else goes through
//      pto.vmi.ensure_mask_granularity.  Mask logic operands are converted to
//      the granularity of their result's class, which makes the internal
//      consistency of those operations hold by construction.  A narrow producer
//      that crosses into a pto.vecscope is rebuilt inside that scope, because
//      VPTO lowering can only reconstruct it where it is produced.
//
// Compare results are never recomputed: mask rematerialization only recreates
// pure mask producers, so a predicate that captured a comparison keeps its
// condition snapshot.

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VMIControlFlowSupport.h"
#include "VMI/VMIMaskRematerialization.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMIMASKGRANULARITYASSIGNMENT
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr unsigned kMaskGranularity8Bit = 8;
constexpr unsigned kMaskGranularity16Bit = 16;
constexpr unsigned kMaskGranularity32Bit = 32;
constexpr unsigned kInvalidMaskId = ~0U;

/// Operand indexes of the mask values combined by one mask logic operation.
constexpr unsigned kMaskLogicUnaryOperands[] = {0};
constexpr unsigned kMaskLogicBinaryOperands[] = {0, 1};

struct MaskNode {
    Value value;
    VMIMaskType type;
    unsigned parent = 0;
    /// Granularity the value itself requires: either declared by the input IR or
    /// imposed by its producer (a compare result follows the compared element
    /// width).  Uses never write this field.
    std::string requiredGranularity;
};

/// One mask operand together with the granularity its consumer needs.  A request
/// never changes the granularity of the value it consumes: it drives the per-use
/// materialization and, for histogram/scatter consumers, the fallback
/// granularity of an otherwise unconstrained value.
struct MaskUseRequest {
    OpOperand *operand;
    std::string granularity;
    /// Set when the mask is produced outside the pto.vecscope that consumes it and
    /// therefore has to be rebuilt at the use even when the granularity already
    /// matches.
    bool rebuildInsideScope = false;
};

/// Mask values combined by one mask logic operation (pto.vmi.mask_and/or/xor,
/// pto.vmi.mask_not, or the mask form of pto.vmi.vand/vor/vxor/vnot).  They all
/// have to run at one physical granularity, but the operation does not get to
/// choose it: the solver reads the granularity of the result's class and the
/// materializer converts every operand at this use.  Keeping the decision with
/// the result is what allows one logical root mask to feed consumers of
/// different element widths.
struct MaskLogicGroup {
    Value result;
    SmallVector<OpOperand *, mlir::pto::kValue2> operands;
};

static unsigned getElementBitWidth(Type type) {
  if (isa<IndexType>(type)) {
    return mlir::pto::kValue64;
  }
  return pto::getPTOStorageElemBitWidth(type);
}

static StringRef getMaskGranularityForElement(Type elementType) {
  switch (getElementBitWidth(elementType)) {
  case kMaskGranularity8Bit:
    return "b8";
  case kMaskGranularity16Bit:
    return "b16";
  case kMaskGranularity32Bit:
    return "b32";
  default:
    return "";
  }
}

static bool containsVMIType(Type type) {
  if (isa<VMIVRegType, VMIMaskType>(type)) {
    return true;
  }
  if (auto functionType = dyn_cast<FunctionType>(type)) {
    return llvm::any_of(functionType.getInputs(), containsVMIType) ||
           llvm::any_of(functionType.getResults(), containsVMIType);
  }
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return containsVMIType(shapedType.getElementType());
  }
  return false;
}

/// Innermost enclosing pto.vecscope/pto.strict_vecscope of \p op, if any.
static Operation *enclosingVecScope(Operation *op)
{
    for (Operation *current = op; current; current = current->getParentOp()) {
        if (isa<VecScopeOp, StrictVecScopeOp>(current)) {
            return current;
        }
    }
    return nullptr;
}

/// Whether \p value is a pure mask producer that can be recreated cheaply.
static bool isRematerializableMaskProducer(Value value)
{
    Operation *definingOp = value.getDefiningOp();
    return isVMIUnspecifiedValue(definingOp) ||
        isa_and_nonnull<VMICreateMaskOp, VMICreateGroupMaskOp,
            VMIConstantMaskOp>(definingOp);
}

/// A mask produced outside a vector scope and consumed inside it has to be
/// rebuilt inside that scope: VPTO lowering materializes a narrow mask from a
/// predicate at the producing site, and a mask value that crosses the scope
/// boundary cannot carry that reconstruction.  Rebuilding the producer at the
/// use keeps the class granularity (a value property) untouched and only adapts
/// the placement.
static bool needsInScopeRebuild(Value value, Operation *user)
{
    Operation *userScope = enclosingVecScope(user);
    if (!userScope) {
        return false;
    }
    Operation *definingOp = value.getDefiningOp();
    if (!definingOp) {
        return false;
    }
    Operation *definingScope = enclosingVecScope(definingOp);
    if (definingScope == userScope) {
        return false;
    }
    return isRematerializableMaskProducer(value);
}

struct MaskGranularitySolver {
  explicit MaskGranularitySolver(ModuleOp module)
      : module(module), ctx(module.getContext()) {}

    unsigned addMaskValue(Value value)
    {
        auto type = dyn_cast<VMIMaskType>(value.getType());
        if (!type) {
            return kInvalidMaskId;
        }
        auto [it, inserted] = maskIds.try_emplace(value, maskNodes.size());
        if (inserted) {
            std::string requiredGranularity;
            if (VMIMaskType::isConcreteGranularity(type.getGranularity())) {
                requiredGranularity = type.getGranularity().str();
            }
            maskNodes.push_back(
                MaskNode{value, type, it->second, requiredGranularity});
        }
        return it->second;
    }

  unsigned findMask(unsigned id) {
    if (maskNodes[id].parent == id) {
      return id;
    }
    maskNodes[id].parent = findMask(maskNodes[id].parent);
    return maskNodes[id].parent;
  }

    /// Equivalence class of \p value, or kInvalidMaskId when it is not a mask.
    unsigned maskClass(Value value)
    {
        unsigned id = addMaskValue(value);
        return id == kInvalidMaskId ? kInvalidMaskId : findMask(id);
    }

    LogicalResult uniteMask(Value lhs, Value rhs, Operation *op)
    {
        unsigned lhsId = addMaskValue(lhs);
        unsigned rhsId = addMaskValue(rhs);
        if (lhsId == kInvalidMaskId || rhsId == kInvalidMaskId) {
            return success();
        }
        unsigned lhsRoot = findMask(lhsId);
        unsigned rhsRoot = findMask(rhsId);
        if (lhsRoot == rhsRoot) {
            return success();
        }

        MaskNode &lhsNode = maskNodes[lhsRoot];
        MaskNode &rhsNode = maskNodes[rhsRoot];
        StringRef lhsRequired = lhsNode.requiredGranularity;
        StringRef rhsRequired = rhsNode.requiredGranularity;
        bool bothRequired = !lhsRequired.empty() && !rhsRequired.empty();
        if (bothRequired && lhsRequired != rhsRequired) {
            return op->emitError() << kVMIDiagLayoutContractPrefix
                << "conflicting mask granularities " << lhsRequired
                << " and " << rhsRequired;
        }

        rhsNode.parent = lhsRoot;
        if (lhsNode.requiredGranularity.empty()) {
            lhsNode.requiredGranularity = rhsNode.requiredGranularity;
        }
        return success();
    }

    /// Records a granularity the value itself requires.  This is the only writer
    /// of a class requirement; consumers file a request with requestMaskUse()
    /// instead of narrowing the value they consume.
    LogicalResult requireMaskGranularity(Value mask, StringRef granularity, Operation *op)
    {
        unsigned id = addMaskValue(mask);
        if (id == kInvalidMaskId) {
            return success();
        }
        if (granularity.empty()) {
            return op->emitError() << kVMIDiagLayoutContractPrefix
                << "cannot infer concrete mask granularity";
        }
        MaskNode &node = maskNodes[findMask(id)];
        bool alreadyRequired = !node.requiredGranularity.empty();
        if (alreadyRequired && node.requiredGranularity != granularity) {
            return op->emitError()
                << kVMIDiagLayoutContractPrefix
                << "conflicting mask granularities " << node.requiredGranularity
                << " and " << granularity;
        }
        node.requiredGranularity = granularity.str();
        return success();
    }

    /// Records that the mask values of one mask logic operation have to share one
    /// granularity.  The granularity itself stays open: the solver takes it from
    /// the result's class.
    LogicalResult addMaskLogicGroup(Value result, ArrayRef<unsigned> indexes, Operation *op)
    {
        if (!isa<VMIMaskType>(result.getType())) {
            return success();
        }
        MaskLogicGroup group;
        group.result = result;
        for (unsigned index : indexes) {
            if (index >= op->getNumOperands()) {
                continue;
            }
            OpOperand &operand = op->getOpOperand(index);
            if (!isa<VMIMaskType>(operand.get().getType())) {
                continue;
            }
            group.operands.push_back(&operand);
        }
        logicGroups.push_back(std::move(group));
        return success();
    }

  LogicalResult requestMaskUse(OpOperand &operand, StringRef granularity,
                               Operation *op) {
    if (!isa<VMIMaskType>(operand.get().getType())) {
      return success();
    }
    if (granularity.empty()) {
      return op->emitError() << kVMIDiagLayoutContractPrefix
                             << "cannot infer concrete mask use granularity";
    }
    maskUseRequests.push_back(MaskUseRequest{
        &operand, granularity.str(), needsInScopeRebuild(operand.get(), op)});
    return success();
  }

  LogicalResult collect() {
    module.walk([this](Operation *op) {
      for (Value result : op->getResults()) {
        addMaskValue(result);
      }
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (BlockArgument arg : block.getArguments()) {
            addMaskValue(arg);
          }
        }
      }
    });
    return success();
  }

  LogicalResult constrainDirectMaskedElementwise(
      Operation *op, unsigned sourceOperandIndex,
      std::optional<unsigned> maskOperandIndex) {
    if (!maskOperandIndex || *maskOperandIndex >= op->getNumOperands()) {
      return success();
    }
    auto sourceType =
        dyn_cast<VMIVRegType>(op->getOperand(sourceOperandIndex).getType());
    if (!sourceType) {
      return success();
    }
    return requestMaskUse(
        op->getOpOperand(*maskOperandIndex),
        getMaskGranularityForElement(sourceType.getElementType()), op);
  }

  static std::optional<WalkResult> constraintResult(LogicalResult result) {
    return failed(result) ? WalkResult::interrupt() : WalkResult::advance();
  }

    std::optional<WalkResult> addMaskValueConstraint(Operation *op)
    {
        return llvm::TypeSwitch<Operation *, std::optional<WalkResult>>(op)
            .Case<VMIMaskAndOp, VMIMaskOrOp, VMIMaskXOrOp>([this, op](auto maskOp) {
                return constraintResult(
                    addMaskLogicGroup(maskOp.getResult(), kMaskLogicBinaryOperands, op));
            })
            .Case<VMIMaskNotOp>([this, op](auto maskOp) {
                return constraintResult(
                    addMaskLogicGroup(maskOp.getResult(), kMaskLogicUnaryOperands, op));
            })
            .Case<VMIVandOp, VMIVorOp, VMIVxorOp>([this, op](auto) {
                // Only the mask-valued form is a logic operation; the elementwise
                // vreg form keeps its own predicate use constraint.
                if (!isa<VMIMaskType>(op->getResult(0).getType())) {
                    return std::optional<WalkResult>{};
                }
                return constraintResult(
                    addMaskLogicGroup(op->getResult(0), kMaskLogicBinaryOperands, op));
            })
            .Case<VMIVnotOp>([this, op](auto) {
                if (!isa<VMIMaskType>(op->getResult(0).getType())) {
                    return std::optional<WalkResult>{};
                }
                return constraintResult(
                    addMaskLogicGroup(op->getResult(0), kMaskLogicUnaryOperands, op));
            })
            .Case<VMIEnsureMaskLayoutOp>([this, op](auto maskOp) {
                // Reshaping the physical layout does not change the value.
                return constraintResult(
                    uniteMask(maskOp.getSource(), maskOp.getResult(), op));
            })
            .Case<VMICmpFOp, VMICmpIOp>([this, op](auto compareOp) {
                auto lhsType = cast<VMIVRegType>(compareOp.getLhs().getType());
                return constraintResult(requireMaskGranularity(
                    compareOp.getResult(),
                    getMaskGranularityForElement(lhsType.getElementType()), op));
            })
            .Case<VMIVcmpOp, VMIVcmpsOp>([this, op](auto) {
                auto sourceType = cast<VMIVRegType>(op->getOperand(0).getType());
                return constraintResult(requireMaskGranularity(
                    op->getResult(0),
                    getMaskGranularityForElement(sourceType.getElementType()), op));
            })
            .Default([](Operation *) { return std::nullopt; });
    }

  std::optional<WalkResult> addResultMaskUseConstraint(Operation *op) {
    return llvm::TypeSwitch<Operation *, std::optional<WalkResult>>(op)
        .Case<VMISelectOp, VMIvSelOp, VMIActivePrefixIndexOp, VMICompressOp,
              VMIStrideLoadOp, VMIMaskedLoadOp, VMIGatherOp, VMIExpandLoadOp>(
            [this, op](auto maskOp) {
              auto resultType = cast<VMIVRegType>(maskOp.getResult().getType());
              return constraintResult(requestMaskUse(
                  maskOp.getMaskMutable(),
                  getMaskGranularityForElement(resultType.getElementType()),
                  op));
            })
        .Default([](Operation *) { return std::nullopt; });
  }

  /// Constrains the optional variadic predicate of a unified elementwise op whose
  /// physical predicate, when present, is the operand at `maskIndex`.
  std::optional<WalkResult> constrainOptionalElementwiseMask(Operation *op,
                                                             unsigned maskIndex) {
    if (!isa<VMIVRegType>(op->getResult(0).getType())) {
      return constraintResult(success());
    }
    std::optional<unsigned> index = op->getNumOperands() > maskIndex
                                        ? std::optional<unsigned>(maskIndex)
                                        : std::nullopt;
    return constraintResult(constrainDirectMaskedElementwise(op, 0, index));
  }

  /// Constrains the mandatory predicate operand of a unified op.
  std::optional<WalkResult> constrainMandatoryMask(Operation *op,
                                                   unsigned maskIndex) {
    return constraintResult(
        constrainDirectMaskedElementwise(op, 0, maskIndex));
  }

  /// Constrains the optional conversion predicate of a VMI cast.  The predicate
  /// is expressed in source-element granularity (issue #1671); the operand is
  /// optional, so a plain cast has no constraint at all.
  std::optional<WalkResult> constrainConversionMask(Operation* op)
  {
      return llvm::TypeSwitch<Operation*, std::optional<WalkResult>>(op)
          .Case<
              VMIExtFOp, VMITruncFOp, VMIFPToSIOp, VMIFPToUIOp, VMISIToFPOp, VMIExtSIOp, VMIExtUIOp, VMITruncIOp,
              VMICvtOp>([this, op](auto maskOp) {
              MutableOperandRange masks = maskOp.getMaskMutable();
              if (masks.empty()) {
                  return constraintResult(success());
              }
              return constraintResult(requestMaskUseForSource(*masks.begin(), maskOp.getSource(), op));
          })
          .Default([](Operation*) { return std::nullopt; });
  }

  std::optional<WalkResult> addUnifiedSourceMaskUseConstraint(Operation *op) {
      // A conversion predicate is a source-mask use in the unified cast family too.
      if (std::optional<WalkResult> conversion = constrainConversionMask(op)) {
          return conversion;
      }
    return llvm::TypeSwitch<Operation *, std::optional<WalkResult>>(op)
        .Case<VMIVabsOp>([this, op](auto unaryOp) {
          MutableOperandRange maskOperands = unaryOp.getMaskMutable();
          if (maskOperands.empty()) {
            return constraintResult(success());
          }
          return constraintResult(requestMaskUseForSource(
              *maskOperands.begin(), unaryOp.getSource(), op));
        })
        .Case<VMIVaddOp, VMIVsubOp, VMIVmulOp, VMIVdivOp, VMIVminOp,
              VMIVmaxOp, VMIVandOp, VMIVorOp, VMIVxorOp, VMIVshlOp,
              VMIVshrOp, VMIAndIOp, VMIOrIOp, VMIXOrIOp>([this, op](auto) {
          return constrainOptionalElementwiseMask(op, mlir::pto::kValue2);
        })
        .Case<VMIVnegOp, VMIVsqrtOp, VMIVexpOp, VMIVlnOp, VMIVreluOp,
              VMIVnotOp, VMINotOp>([this, op](auto) {
          return constrainOptionalElementwiseMask(op, 1);
        })
        .Case<VMIVmulaOp>([this, op](auto) {
          return constrainOptionalElementwiseMask(op, mlir::pto::kValue3);
        })
        .Case<VMIVaxpyOp>([this, op](auto) {
          return constrainMandatoryMask(op, mlir::pto::kValue3);
        })
        .Case<VMIVcmpOp, VMIVcmpsOp>([this, op](auto) {
          return constrainMandatoryMask(op, mlir::pto::kValue2);
        })
        .Case<VMIVlreluOp, VMIVpreluOp>([this, op](auto) {
          return constrainMandatoryMask(op, mlir::pto::kValue2);
        })
        .Default([](Operation *) { return std::nullopt; });
  }

  std::optional<WalkResult> addLegacySourceMaskUseConstraint(Operation *op) {
      // A conversion predicate is a source-mask use in the legacy cast family too.
      if (std::optional<WalkResult> conversion = constrainConversionMask(op)) {
          return conversion;
      }
    return llvm::TypeSwitch<Operation *, std::optional<WalkResult>>(op)
        .Case<VMIAddSOp, VMIMulSOp, VMIMaxSOp, VMIMinSOp, VMIShlSOp,
              VMIShrSOp>([this, op](auto maskOp) {
          return constraintResult(requestMaskUseForSource(
              maskOp.getMaskMutable(), maskOp.getSrc(), op));
        })
        .Case<VMIReduceAddIOp, VMIReduceAddFOp, VMIReduceMaxFOp,
              VMIReduceMinFOp, VMIReduceMaxIOp, VMIReduceMinIOp,
              VMIGroupReduceAddFOp, VMIGroupReduceMaxFOp,
              VMIGroupReduceMinFOp, VMIGroupReduceAddIOp,
              VMIGroupReduceMaxIOp, VMIGroupReduceMinIOp>(
            [this, op](auto maskOp) {
              return constraintResult(requestMaskUseForSource(
                  maskOp.getMaskMutable(), maskOp.getSource(), op));
            })
        .Case<VMIVexpdifOp>([this, op](auto maskOp) {
          return constraintResult(requestMaskUseForSource(
              maskOp.getMaskMutable(), maskOp.getX(), op));
        })
        .Case<VMIVintlvOp, VMIVdintlvOp>([this, op](auto maskOp) {
          return constraintResult(requestMaskUseForSource(
              maskOp.getMaskMutable(), maskOp.getLhs(), op));
        })
        .Case<VMIvStoreOp>([this, op](auto maskOp) {
            MutableOperandRange masks = maskOp.getMaskMutable();
            if (masks.empty()) {
                return constraintResult(success());
            }
            return constraintResult(
                requestMaskUseForSource(*masks.begin(), maskOp.getValue(), op));
        })
        .Case<VMIMaskedStoreOp, VMIStrideStoreOp, VMIScatterOp,
              VMICompressStoreOp>([this, op](auto maskOp) {
          return constraintResult(requestMaskUseForSource(
              maskOp.getMaskMutable(), maskOp.getValue(), op));
        })
        .Default([](Operation *) { return std::nullopt; });
  }

  std::optional<WalkResult> addSourceMaskUseConstraint(Operation *op) {
    if (std::optional<WalkResult> unified =
            addUnifiedSourceMaskUseConstraint(op)) {
      return unified;
    }
    return addLegacySourceMaskUseConstraint(op);
  }

  std::optional<WalkResult> addSpecialMaskConstraint(Operation *op) {
    return llvm::TypeSwitch<Operation *, std::optional<WalkResult>>(op)
        .Case<VMIVmullOp>([this, op](VMIVmullOp vmull) {
          return constraintResult(requestMaskUse(vmull.getMaskMutable(),
                                                 "b32", op));
        })
        .Case<VMIVaddcOp, VMIVsubcOp>([this, op](auto carryOp) {
          bool failedToConstrain =
            failed(requestMaskUse(carryOp.getMaskMutable(), "b32", op)) ||
            failed(requireMaskGranularity(carryOp.getCarry(), "b32", op));
          return constraintResult(failure(failedToConstrain));
        })
        .Case<VMIVaddcsOp, VMIVsubcsOp>([this, op](auto carryOp) {
          bool failedToConstrain =
            failed(requestMaskUse(carryOp.getCarryInMutable(), "b32", op)) ||
            failed(requestMaskUse(carryOp.getMaskMutable(), "b32", op)) ||
            failed(requireMaskGranularity(carryOp.getCarry(), "b32", op));
          return constraintResult(failure(failedToConstrain));
        })
        .Case<VMIVdhistOp, VMIVchistOp>([this, op](auto histogram) {
          return constraintResult(
              requestMaskUse(histogram.getMaskMutable(), "b8", op));
        })
        .Default([](Operation *) { return std::nullopt; });
  }

  LogicalResult addSwitchConstraints(cf::SwitchOp switchOp) {
    if (failed(addBranchConstraints(switchOp.getDefaultDestination(),
                                    switchOp.getDefaultOperands(), switchOp))) {
      return failure();
    }
    for (auto [destination, operands] :
         llvm::zip(switchOp.getCaseDestinations(), switchOp.getCaseOperands())) {
      if (failed(addBranchConstraints(destination, operands, switchOp))) {
        return failure();
      }
    }
    return success();
  }

  std::optional<WalkResult> addControlFlowConstraint(Operation *op) {
    return llvm::TypeSwitch<Operation *, std::optional<WalkResult>>(op)
        .Case<scf::IfOp>([this](auto controlOp) {
          return constraintResult(addIfConstraints(controlOp));
        })
        .Case<scf::ExecuteRegionOp>([this](auto controlOp) {
          return constraintResult(addExecuteRegionConstraints(controlOp));
        })
        .Case<scf::IndexSwitchOp>([this](auto controlOp) {
          return constraintResult(addIndexSwitchConstraints(controlOp));
        })
        .Case<scf::WhileOp>([this](auto controlOp) {
          return constraintResult(addWhileConstraints(controlOp));
        })
        .Case<scf::ForOp>([this](auto controlOp) {
          return constraintResult(addForConstraints(controlOp));
        })
        .Case<cf::BranchOp>([this, op](cf::BranchOp branch) {
          return constraintResult(addBranchConstraints(
              branch.getDest(), branch.getDestOperands(), op));
        })
        .Case<cf::CondBranchOp>([this, op](cf::CondBranchOp branch) {
          bool failedToConstrain =
              failed(addBranchConstraints(branch.getTrueDest(),
                                          branch.getTrueDestOperands(), op)) ||
              failed(addBranchConstraints(branch.getFalseDest(),
                                          branch.getFalseOperands(), op));
          return constraintResult(failure(failedToConstrain));
        })
        .Case<cf::SwitchOp>([this](auto switchOp) {
          return constraintResult(addSwitchConstraints(switchOp));
        })
        .Case<func::ReturnOp>([this](auto returnOp) {
          return constraintResult(addReturnConstraints(returnOp));
        })
        .Case<func::CallOp>([this](auto callOp) {
          return constraintResult(addCallConstraints(callOp));
        })
        .Default([](Operation *) { return std::nullopt; });
  }

  WalkResult addConstraint(Operation *op) {
    for (auto constraint : {addMaskValueConstraint(op),
                            addResultMaskUseConstraint(op),
                            addSourceMaskUseConstraint(op),
                            addSpecialMaskConstraint(op),
                            addControlFlowConstraint(op)}) {
      if (constraint) {
        return *constraint;
      }
    }
    bool invalidIndirectCall =
        op->getName().getStringRef() == "func.call_indirect" &&
        hasVMIValueTypes(op);
    if (invalidIndirectCall) {
      op->emitError() << kVMIDiagLayoutContractPrefix
                      << "VMI typed call requires a direct internal callee "
                         "with a body";
      return WalkResult::interrupt();
    }
    if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
      bool invalidDeclaration = funcOp.empty() && hasVMIFunctionType(funcOp);
      if (invalidDeclaration) {
        funcOp.emitError()
            << kVMIDiagLayoutContractPrefix
            << "VMI typed function declaration requires an explicit "
               "external ABI materialization plan";
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  }

  LogicalResult addConstraints() {
    WalkResult result =
        module.walk([this](Operation *op) { return addConstraint(op); });
    return failure(result.wasInterrupted());
  }

  LogicalResult requestMaskUseForSource(OpOperand &mask, Value source,
                                        Operation *op) {
    auto sourceType = dyn_cast<VMIVRegType>(source.getType());
    if (!sourceType) {
      return success();
    }
    return requestMaskUse(mask,
                          getMaskGranularityForElement(
                              sourceType.getElementType()),
                          op);
  }

  LogicalResult uniteEquivalentValues(Value lhs, Value rhs, Operation *op) {
    return uniteMask(lhs, rhs, op);
  }

  LogicalResult addIfConstraints(scf::IfOp ifOp) {
    for (OpResult result : ifOp->getResults()) {
      unsigned resultNo = result.getResultNumber();
      for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
        if (region->empty()) {
          continue;
        }
        auto yieldOp = dyn_cast<scf::YieldOp>(region->front().getTerminator());
        if (!yieldOp || resultNo >= yieldOp.getNumOperands()) {
          continue;
        }
        if (failed(uniteEquivalentValues(result, yieldOp.getOperand(resultNo),
                                         ifOp))) {
          return failure();
        }
      }
    }
    return success();
  }

  LogicalResult addYieldConstraints(ResultRange results, scf::YieldOp yieldOp,
                                    Operation *op) {
    for (auto [index, result] : llvm::enumerate(results)) {
      if (index >= yieldOp.getNumOperands()) {
        break;
      }
      if (failed(uniteEquivalentValues(result, yieldOp.getOperand(index), op))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult addExecuteRegionConstraints(scf::ExecuteRegionOp executeOp) {
    WalkResult result = executeOp.getRegion().walk(
        [this, executeOp](scf::YieldOp yieldOp) mutable {
          const bool belongsToExecuteRegion =
              yieldOp->getParentOp() == executeOp.getOperation();
          if (!belongsToExecuteRegion) {
            return WalkResult::advance();
          }
          if (failed(addYieldConstraints(executeOp->getResults(), yieldOp,
                                         executeOp))) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    return failure(result.wasInterrupted());
  }

  LogicalResult addIndexSwitchConstraints(scf::IndexSwitchOp indexSwitchOp) {
    auto addBlockTerminator =
        [this, indexSwitchOp](Block &block) mutable -> LogicalResult {
      auto yieldOp = dyn_cast<scf::YieldOp>(block.getTerminator());
      if (!yieldOp) {
        return success();
      }
      return addYieldConstraints(indexSwitchOp->getResults(), yieldOp,
                                 indexSwitchOp);
    };
    if (failed(addBlockTerminator(indexSwitchOp.getDefaultBlock()))) {
      return failure();
    }
    for (unsigned idx = 0, e = indexSwitchOp.getNumCases(); idx < e; ++idx) {
      if (failed(addBlockTerminator(indexSwitchOp.getCaseBlock(idx)))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult addWhileConstraints(scf::WhileOp whileOp) {
    return VMIControlFlowSupport::addWhileConstraints(
        whileOp, [this](Value lhs, Value rhs, Operation *op) {
          return uniteEquivalentValues(lhs, rhs, op);
        });
  }

  LogicalResult addForConstraints(scf::ForOp forOp) {
    return VMIControlFlowSupport::addForConstraints(
        forOp, [this](Value lhs, Value rhs, Operation *op) {
          return uniteEquivalentValues(lhs, rhs, op);
        });
  }

  LogicalResult addBranchConstraints(Block *dest, OperandRange operands,
                                     Operation *op) {
    if (!dest) {
      return success();
    }
    for (auto [index, operand] : llvm::enumerate(operands)) {
      if (index >= dest->getNumArguments()) {
        break;
      }
      if (failed(uniteEquivalentValues(operand, dest->getArgument(index), op))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult addReturnConstraints(func::ReturnOp returnOp) {
    auto func = returnOp->getParentOfType<func::FuncOp>();
    if (!func) {
      return success();
    }

    auto it = firstReturnOperandsByFunc.find(func);
    if (it == firstReturnOperandsByFunc.end()) {
      SmallVector<Value> operands(returnOp.getOperands());
      firstReturnOperandsByFunc.try_emplace(func, std::move(operands));
      return success();
    }

    ArrayRef<Value> firstOperands = it->second;
    for (auto [index, operand] : llvm::enumerate(returnOp.getOperands())) {
      if (index >= firstOperands.size()) {
        break;
      }
      if (failed(
              uniteEquivalentValues(firstOperands[index], operand, returnOp))) {
        return failure();
      }
    }
    return success();
  }

  bool hasVMIValueTypes(Operation *op) const {
    return llvm::any_of(op->getOperandTypes(), containsVMIType) ||
           llvm::any_of(op->getResultTypes(), containsVMIType);
  }

  bool hasVMIFunctionType(func::FuncOp func) const {
    FunctionType type = func.getFunctionType();
    return llvm::any_of(type.getInputs(), containsVMIType) ||
           llvm::any_of(type.getResults(), containsVMIType);
  }

  LogicalResult addCallConstraints(func::CallOp callOp) {
    if (!hasVMIValueTypes(callOp)) {
      return success();
    }

    auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        callOp, callOp.getCalleeAttr());
    if (!callee || callee.empty()) {
      return callOp.emitError()
             << kVMIDiagLayoutContractPrefix
             << "VMI typed call requires a direct internal callee with a body";
    }

    for (auto [operand, argument] :
         llvm::zip(callOp.getOperands(), callee.getArguments())) {
      if (failed(uniteEquivalentValues(operand, argument, callOp))) {
        return failure();
      }
    }

    SmallVector<func::ReturnOp> returns;
    callee.walk(
        [&returns](func::ReturnOp returnOp) { returns.push_back(returnOp); });
    for (func::ReturnOp returnOp : returns) {
      for (auto [index, result] : llvm::enumerate(callOp.getResults())) {
        if (index >= returnOp.getNumOperands()) {
          break;
        }
        if (failed(uniteEquivalentValues(result, returnOp.getOperand(index),
                                         callOp))) {
          return failure();
        }
      }
    }
    return success();
  }

    /// Collects the union-find roots once and seeds the classes whose granularity
    /// is inherent.  Unions only happen while collecting constraints, so the class
    /// list is stable during solving.
    void collectMaskClasses()
    {
        for (unsigned id = 0, e = maskNodes.size(); id != e; ++id) {
            unsigned root = findMask(id);
            if (root != id) {
                continue;
            }
            maskClasses.push_back(id);
            if (!maskNodes[id].requiredGranularity.empty()) {
                solvedGranularities[id] = maskNodes[id].requiredGranularity;
            }
        }
    }

    /// Granularity the solver already decided for one class, or an empty string
    /// while the class is undecided.  Only the b32 fallback runs after this.
    StringRef solvedClassGranularity(unsigned root) const
    {
        auto it = solvedGranularities.find(root);
        return it == solvedGranularities.end() ? StringRef() : StringRef(it->second);
    }

    StringRef solvedGranularityOfValue(Value value)
    {
        unsigned root = maskClass(value);
        return root == kInvalidMaskId ? StringRef() : solvedClassGranularity(root);
    }

    bool assignClassGranularity(unsigned root, StringRef granularity)
    {
        return solvedGranularities.try_emplace(root, granularity.str()).second;
    }

    /// Granularity the mask values of one logic group already agree on.  Values
    /// that are undecided do not participate; a group whose decided values
    /// disagree has no common physical granularity and stays undecided.
    StringRef agreedGroupGranularity(const MaskLogicGroup &group)
    {
        StringRef granularity;
        for (OpOperand *operand : group.operands) {
            StringRef operandGranularity = solvedGranularityOfValue(operand->get());
            if (operandGranularity.empty()) {
                continue;
            }
            if (granularity.empty()) {
                granularity = operandGranularity;
            } else if (granularity != operandGranularity) {
                return StringRef();
            }
        }
        return granularity;
    }

    /// Lets a mask logic result inherit the granularity its operands already agree
    /// on.  That keeps a plain mask_not/vnot of a compare at the compare
    /// granularity instead of forcing a conversion, which is the behaviour a
    /// single fused class used to provide.  The decision is a property of the
    /// result value only: the operands are never narrowed or widened here.
    /// The rounds use a snapshot, so the outcome does not depend on the order in
    /// which the operations were walked, and the budget only guards against a
    /// pathological cycle.
    void solveMaskLogicGroups()
    {
        unsigned maxRounds = 4 * (logicGroups.size() + 1) + 16;
        for (unsigned round = 0; round != maxRounds; ++round) {
            // Owning copies: assigning one class may rehash the solved map.
            DenseMap<unsigned, std::string> pending;
            for (const MaskLogicGroup &group : logicGroups) {
                unsigned root = maskClass(group.result);
                if (root == kInvalidMaskId ||
                    !maskNodes[root].requiredGranularity.empty() ||
                    !solvedClassGranularity(root).empty()) {
                    continue;
                }
                StringRef granularity = agreedGroupGranularity(group);
                if (granularity.empty()) {
                    continue;
                }
                auto [it, inserted] = pending.try_emplace(root, granularity.str());
                if (!inserted && it->second != granularity) {
                    // Two groups define one class through an identity relation and
                    // disagree: leave the class undecided instead of letting the walk
                    // order pick a winner.
                    it->second.clear();
                }
            }
            bool changed = false;
            for (const auto &entry : pending) {
                if (entry.second.empty()) {
                    continue;
                }
                changed |= assignClassGranularity(entry.first, entry.second);
            }
            if (!changed) {
                return;
            }
        }
    }

    /// Keeps the long-standing histogram/scatter fallback: an otherwise
    /// unconstrained predicate used by a histogram or scatter consumer starts in
    /// that consumer's granularity.  Falling back to b32 there would add a
    /// b32-to-b8 conversion whose deinterleaved carrier cannot represent short
    /// vectors.  The decision now belongs to the solver instead of rewriting the
    /// classes after the fact.
    void solveHistogramScatterFallback()
    {
        struct Preference {
            StringRef granularity;
            bool conflict = false;
            bool histogramOrScatter = false;
        };
        DenseMap<unsigned, Preference> preferences;
        for (const MaskUseRequest &request : maskUseRequests) {
            unsigned root = maskClass(request.operand->get());
            if (root == kInvalidMaskId || !solvedClassGranularity(root).empty()) {
                continue;
            }
            Preference &preference = preferences[root];
            if (preference.granularity.empty()) {
                preference.granularity = request.granularity;
            } else if (preference.granularity != request.granularity) {
                preference.conflict = true;
            }
            preference.histogramOrScatter =
                preference.histogramOrScatter ||
                isa<VMIScatterOp, VMIVdhistOp, VMIVchistOp>(
                    request.operand->getOwner());
        }
        for (const auto &entry : preferences) {
            const Preference &preference = entry.second;
            if (preference.histogramOrScatter && !preference.conflict) {
                assignClassGranularity(entry.first, preference.granularity);
            }
        }
    }

    /// Applies the documented fallback for classes that neither carry a
    /// requirement, nor inherited a granularity from a mask logic group, nor took
    /// the histogram/scatter fallback.
    void defaultUnconstrainedMaskClasses()
    {
        for (unsigned root : maskClasses) {
            if (solvedClassGranularity(root).empty()) {
                assignClassGranularity(root, "b32");
            }
        }
    }

    /// Solves one granularity per equivalence class.  An inherent requirement wins
    /// and disagreeing requirements are diagnosed while the classes are united; a
    /// mask logic result may inherit the granularity its operands agree on; the
    /// histogram/scatter fallback and the b32 default close the rest.
    void solveMaskGranularities()
    {
        collectMaskClasses();
        solveMaskLogicGroups();
        solveHistogramScatterFallback();
        defaultUnconstrainedMaskClasses();
    }

    void rewriteMaskTypes()
    {
        for (unsigned id = 0, e = maskNodes.size(); id != e; ++id) {
            MaskNode &node = maskNodes[id];
            StringRef granularity = solvedClassGranularity(findMask(id));
            if (granularity.empty()) {
                granularity = "b32";
            }
            node.value.setType(VMIMaskType::get(ctx, node.type.getElementCount(),
                granularity,
                node.type.getLayoutAttr()));
        }
    }

  void rewriteFunctionType() {
    module.walk([this](func::FuncOp func) {
      if (func.empty()) {
        return;
      }

      SmallVector<Type> inputs = getFunctionInputTypes(func);
      SmallVector<Type> results;
      SmallVector<Type> callResultTypes =
          getConsistentCallResultTypes(module, func);
      auto it = firstReturnOperandsByFunc.find(func);
      if (!callResultTypes.empty()) {
        for (Type type : callResultTypes) {
          results.push_back(type ? type : Type{});
        }
      } else if (it != firstReturnOperandsByFunc.end()) {
        for (Value operand : it->second) {
          results.push_back(operand.getType());
        }
      } else {
        FunctionType functionType = func.getFunctionType();
        for (Type type : functionType.getResults()) {
          if (auto maskType = dyn_cast<VMIMaskType>(type)) {
            StringRef granularity =
                VMIMaskType::isConcreteGranularity(maskType.getGranularity())
                    ? maskType.getGranularity()
                    : StringRef("b32");
            results.push_back(VMIMaskType::get(
                ctx, maskType.getElementCount(), granularity,
                maskType.getLayoutAttr()));
          } else {
            results.push_back(type);
          }
        }
      }

      for (auto [index, type] : llvm::enumerate(results)) {
        if (!type) {
          results[index] = func.getFunctionType().getResult(index);
        }
      }

      func.setFunctionType(FunctionType::get(ctx, inputs, results));
    });
  }

    /// Converts every mask use that does not already match its consumer.  A cheap
    /// mask producer is rematerialized at the use, everything else goes through
    /// pto.vmi.ensure_mask_granularity, so compare results are never recomputed.
    LogicalResult insertMaskUseMaterializations(ArrayRef<MaskUseRequest> requests)
    {
        OpBuilder builder(ctx);
        for (const MaskUseRequest &request : requests) {
            Value value = request.operand->get();
            auto sourceType = dyn_cast<VMIMaskType>(value.getType());
            if (!sourceType) {
                continue;
            }
            if (!request.rebuildInsideScope &&
                sourceType.getGranularity() == request.granularity) {
                continue;
            }

            builder.setInsertionPoint(request.operand->getOwner());
            auto resultType = VMIMaskType::get(ctx, sourceType.getElementCount(),
                request.granularity,
                sourceType.getLayoutAttr());
            std::optional<Value> rematerialized =
                mask_rematerialization::rematerializeMaskProducer(
                    value, resultType, request.operand->getOwner()->getLoc(),
                    builder);
            Value current = rematerialized.value_or(Value{});
            if (!current) {
                current = builder.create<VMIEnsureMaskGranularityOp>(
                    request.operand->getOwner()->getLoc(), resultType, value);
            }
            request.operand->set(current);
        }
        return success();
    }

    /// Materializes every mask use at the granularity it requested.  Mask logic
    /// operands are converted to the granularity of their result's class, which is
    /// what makes those operations internally consistent without any post-rewrite
    /// alignment fixup.
    LogicalResult materializeMaskUses()
    {
        SmallVector<MaskUseRequest> requests(maskUseRequests.begin(),
            maskUseRequests.end());
        for (const MaskLogicGroup &group : logicGroups) {
            StringRef granularity = solvedGranularityOfValue(group.result);
            if (granularity.empty()) {
                continue;
            }
            for (OpOperand *operand : group.operands) {
                if (!isa<VMIMaskType>(operand->get().getType())) {
                    continue;
                }
                requests.push_back(MaskUseRequest{
                    operand, granularity.str(),
                    needsInScopeRebuild(operand->get(), operand->getOwner())});
            }
        }
        return insertMaskUseMaterializations(requests);
    }

  LogicalResult run() {
    if (failed(collect())) {
      return failure();
    }
    if (failed(addConstraints())) {
      return failure();
    }
    solveMaskGranularities();
    rewriteMaskTypes();
    rewriteFunctionType();
    return materializeMaskUses();
  }

    ModuleOp module;
    MLIRContext *ctx;
    DenseMap<Value, unsigned> maskIds;
    DenseMap<func::FuncOp, SmallVector<Value>> firstReturnOperandsByFunc;
    DenseMap<unsigned, std::string> solvedGranularities;
    SmallVector<MaskNode> maskNodes;
    SmallVector<unsigned> maskClasses;
    SmallVector<MaskLogicGroup> logicGroups;
    /// Per-use requests of ordinary mask consumers.  Mask logic operands are kept
    /// in logicGroups because their granularity is only known after solving.
    SmallVector<MaskUseRequest> maskUseRequests;
};

struct VMIMaskGranularityAssignmentPass
    : public mlir::pto::impl::VMIMaskGranularityAssignmentBase<
          VMIMaskGranularityAssignmentPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMIMaskGranularityAssignmentPass)

  void runOnOperation() override {
    if (failed(MaskGranularitySolver(getOperation()).run())) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMIMaskGranularityAssignmentPass() {
  return std::make_unique<VMIMaskGranularityAssignmentPass>();
}
