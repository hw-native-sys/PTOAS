// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/VPTOBridgeRegistry.h"

using namespace mlir;
using namespace mlir::pto;

namespace {
bool matchesBridgePointer(BridgePointerSpace space, Type type) {
  if (auto llvmPointer = dyn_cast<LLVM::LLVMPointerType>(type)) {
    const unsigned expected = space == BridgePointerSpace::Global ? 1 : 0;
    return llvmPointer.getAddressSpace() == expected;
  }
  auto pointer = dyn_cast<PtrType>(type);
  return space == BridgePointerSpace::Global && pointer &&
         pointer.getMemorySpace().getAddressSpace() == AddressSpace::GM;
}

bool matchesBridgeValue(BridgeValueType expected, Type type) {
  switch (expected.kind) {
  case BridgeValueKind::F32:
    return type.isF32();
  case BridgeValueKind::I32:
    return type.isInteger(mlir::pto::kValue32);
  case BridgeValueKind::I64:
  case BridgeValueKind::U64:
    return type.isInteger(mlir::pto::kValue64);
  case BridgeValueKind::Pointer:
    return matchesBridgePointer(expected.space, type);
  case BridgeValueKind::ObjectHandle:
    return isa<PipeType>(type);
  }
  return false;
}

LogicalResult verifyBridgeTypes(Operation *op, TypeRange types,
                               ArrayRef<BridgeValueType> expected,
                               StringRef role) {
  const bool wrongCount = types.size() != expected.size();
  if (wrongCount) {
    return op->emitOpError() << "expects " << expected.size() << " " << role
                             << " values, got " << types.size();
  }
  for (auto indexed : llvm::enumerate(types)) {
    const bool matches = matchesBridgeValue(expected[indexed.index()], indexed.value());
    if (!matches) {
      return op->emitOpError() << role << " #" << indexed.index()
                               << " has incompatible ABI type " << indexed.value();
    }
  }
  return success();
}
// The spec attribute kind is owned by the entry's family.
LogicalResult verifyBridgeSpecKind(Operation *op, const BridgeFunctionDesc &desc,
                                   Attribute spec) {
  if (!spec) {
    return success();
  }
  const bool matches =
      (desc.family == BridgeFamily::Pipe && isa<BridgePipeSpecAttr>(spec)) ||
      (desc.family == BridgeFamily::Cube && isa<BridgeCubeSpecAttr>(spec));
  if (!matches) {
    return op->emitOpError() << "entry '" << stringifyBridgeEntryId(desc.id)
                             << "' does not accept specialization " << spec;
  }
  return success();
}
} // namespace

LogicalResult BridgeCallOp::verify() {
  const auto *desc = findBridgeFunction(getEntry());
  if (!desc || desc->createsObject) {
    return emitOpError("requires a registered non-object entry");
  }
  const bool invalidSpecOrArgs =
      failed(verifyBridgeSpecKind(*this, *desc, getSpecAttr())) ||
      failed(verifyBridgeTypes(*this, getArgs().getTypes(), desc->arguments, "argument"));
  if (invalidSpecOrArgs) {
    return failure();
  }
  return verifyBridgeTypes(*this, getResults().getTypes(), desc->results, "result");
}

LogicalResult BridgeObjectCreateOp::verify() {
  const auto *desc = findBridgeFunction(getEntry());
  const bool validContract = desc && desc->createsObject &&
      !desc->arguments.empty() && desc->arguments.front().kind == BridgeValueKind::Pointer &&
      desc->arguments.front().space == BridgePointerSpace::Private &&
      desc->results.empty() && desc->storageSize != 0 && desc->storageAlign != 0;
  if (!validContract) {
    return emitOpError("requires a registered void init entry with storage as its first argument");
  }
  const bool validHandle = isa<PipeType>(getResult().getType());
  if (!validHandle) {
    return emitOpError("requires a logical !pto.pipe storage handle result");
  }
  if (failed(verifyBridgeSpecKind(*this, *desc, getSpecAttr()))) {
    return failure();
  }
  return verifyBridgeTypes(*this, getArgs().getTypes(), desc->arguments.drop_front(), "argument");
}

LogicalResult BridgeInstanceOp::verify() {
  const auto *desc = findBridgeFunction(getEntry());
  if (!desc) {
    return emitOpError("requires a registered bridge entry");
  }
  if (!isa_and_nonnull<ModuleOp>((*this)->getParentOp())) {
    return emitOpError("must be a module-level instance table entry");
  }
  const bool coreMatches =
      desc->core == BridgeCoreKind::Any || desc->core == getCore();
  if (!coreMatches) {
    return emitOpError() << "entry '" << stringifyBridgeEntryId(desc->id)
                         << "' requires core '"
                         << stringifyBridgeCoreKind(desc->core) << "', got '"
                         << stringifyBridgeCoreKind(getCore()) << "'";
  }
  const bool missingSpec = !getSpecAttr() && desc->family != BridgeFamily::Print;
  if (missingSpec) {
    return emitOpError("requires a structured specialization");
  }
  return verifyBridgeSpecKind(*this, *desc, getSpecAttr());
}

namespace {
template <typename BridgeOpT>
LogicalResult verifyBridgeInstanceUse(BridgeOpT op,
                                      SymbolTableCollection &symbolTable) {
  FlatSymbolRefAttr reference = op.getInstanceAttr();
  if (!reference) {
    return success();
  }
  auto instance = symbolTable.lookupNearestSymbolFrom<BridgeInstanceOp>(
      op.getOperation(), reference);
  if (!instance) {
    return op.emitOpError() << "references unknown bridge instance "
                            << reference;
  }
  const bool entryDiffers = instance.getEntry() != op.getEntry();
  if (entryDiffers) {
    return op.emitOpError() << "entry differs from bridge instance "
                            << reference;
  }
  const bool specDiffers =
      op.getSpecAttr() && op.getSpecAttr() != instance.getSpecAttr();
  if (specDiffers) {
    return op.emitOpError() << "specialization differs from bridge instance "
                            << reference;
  }
  return success();
}
} // namespace

LogicalResult
BridgeCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyBridgeInstanceUse(*this, symbolTable);
}

LogicalResult
BridgeObjectCreateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyBridgeInstanceUse(*this, symbolTable);
}

//===----------------------------------------------------------------------===//
// Bridge specializations
//===----------------------------------------------------------------------===//

LogicalResult BridgeTileSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError, TileBufType tile)
{
    if (!tile) {
        return emitError() << "bridge tile spec requires a tile_buf type";
    }
    ArrayRef<int64_t> shape = tile.getShape();
    ArrayRef<int64_t> valid = tile.getValidShape();
    const bool wrongRank = shape.size() != 2 || valid.size() != 2;
    if (wrongRank) {
        return emitError() << "bridge tile spec requires a rank-2 tile, got " << tile;
    }
    for (unsigned dim = 0; dim < mlir::pto::kValue2; ++dim) {
        const bool unusableShape = ShapedType::isDynamic(shape[dim]) || shape[dim] <= 0;
        if (unusableShape) {
            return emitError() << "bridge tile spec requires a static positive shape, got " << tile;
        }
        if (ShapedType::isDynamic(valid[dim])) {
            return emitError() << "bridge tile spec requires static valid dimensions";
        }
        const bool validFits = valid[dim] > 0 && valid[dim] <= shape[dim];
        if (!validFits) {
            return emitError() << "bridge tile spec valid shape exceeds its shape in " << tile;
        }
    }
    auto memory = llvm::dyn_cast_or_null<AddressSpaceAttr>(tile.getMemorySpace());
    if (!memory) {
        return emitError() << "bridge tile spec requires a tile memory space, got " << tile;
    }
    switch (memory.getAddressSpace()) {
        case AddressSpace::LEFT:
        case AddressSpace::RIGHT:
        case AddressSpace::ACC:
        case AddressSpace::VEC:
        case AddressSpace::MAT:
            break;
        default:
            return emitError() << "bridge tile spec has unsupported tile kind in " << tile;
    }
    Type element = tile.getElementType();
    const bool knownElement = element.isF16() || element.isBF16() || element.isF32() || element.isInteger(8) ||
                              element.isInteger(16) || element.isInteger(32);
    if (!knownElement) {
        return emitError() << "bridge tile spec has unsupported element type " << element;
    }
    return success();
}

LogicalResult BridgePipeSpecAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, int32_t dirMask, int32_t slotSize, int32_t slotNum,
    int32_t localSlotNum, int32_t flagBase, BoolAttr nosplit, std::optional<int32_t> split, BridgeTileSpecAttr producer,
    BridgeTileSpecAttr consumer)
{
    if (dirMask != 1 && dirMask != mlir::pto::kValue2) {
        return emitError() << "bridge pipe spec supports dir_mask 1 (C2V) or 2 (V2C), got " << dirMask;
    }
    if (slotSize <= 0 || slotNum <= 0 || localSlotNum <= 0 || flagBase < 0) {
        return emitError() << "bridge pipe spec requires positive slot sizes and "
                              "counts and a non-negative flag base";
    }
    if (!nosplit) {
        return emitError() << "bridge pipe spec requires a nosplit flag";
    }
    if (split && (*split < 0 || *split > mlir::pto::kValue4)) {
        return emitError() << "bridge pipe spec has unsupported split " << *split;
    }
    // C2V produces on cube (Acc) and consumes on vector (Vec); V2C the reverse.
    auto hasSpace = [](BridgeTileSpecAttr spec, AddressSpace space) {
        auto memory = llvm::cast<AddressSpaceAttr>(spec.getTile().getMemorySpace());
        return memory.getAddressSpace() == space;
    };
    const AddressSpace producerSpace = dirMask == 1 ? AddressSpace::ACC : AddressSpace::VEC;
    const AddressSpace consumerSpace = dirMask == 1 ? AddressSpace::VEC : AddressSpace::MAT;
    if (producer && !hasSpace(producer, producerSpace)) {
        return emitError() << "bridge pipe spec producer tile does not match dir_mask " << dirMask;
    }
    if (consumer && !hasSpace(consumer, consumerSpace)) {
        return emitError() << "bridge pipe spec consumer tile does not match dir_mask " << dirMask;
    }
    return success();
}

LogicalResult
BridgeCubeSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                           BridgeTileSpecAttr dst, BridgeTileSpecAttr lhs,
                           BridgeTileSpecAttr rhs, AccPhaseAttr accPhase,
                           BridgeTileSpecAttr accIn) {
  if (!dst || !lhs || !rhs || !accPhase) {
    return emitError() << "bridge cube spec requires dst, lhs, rhs and acc_phase";
  }
  auto space = [](BridgeTileSpecAttr spec) {
    return llvm::cast<AddressSpaceAttr>(spec.getTile().getMemorySpace()).getAddressSpace();
  };
  const bool wrongKinds = space(dst) != AddressSpace::ACC ||
                          space(lhs) != AddressSpace::LEFT ||
                          space(rhs) != AddressSpace::RIGHT;
  if (wrongKinds) {
    return emitError() << "bridge cube spec requires acc dst, left lhs and right rhs";
  }
  ArrayRef<int64_t> m = dst.getTile().getShape();
  ArrayRef<int64_t> a = lhs.getTile().getShape();
  ArrayRef<int64_t> b = rhs.getTile().getShape();
  if (a[0] != m[0] || b[1] != m[1] || a[1] != b[0]) {
    return emitError() << "bridge cube spec shapes do not form MxK * KxN = MxN";
  }
  if (accIn) {
    const bool usableAccumulator = space(accIn) == AddressSpace::ACC &&
                                   accIn.getTile().getShape() == m;
    if (!usableAccumulator) {
      return emitError() << "bridge cube spec accumulator must be an acc tile "
                            "shaped like the destination";
    }
  }
  return success();
}
