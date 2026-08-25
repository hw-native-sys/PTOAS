// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/VecScopeMemBar/VecScopeMemoryFootprint.h"

#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"

#include <algorithm>
#include <cstdint>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecscopemembar;

namespace {

static std::optional<AddressSpace> getTypeAddressSpace(Type type) {
  if (auto ptrType = dyn_cast<pto::PtrType>(type))
    return ptrType.getMemorySpace().getAddressSpace();
  if (auto memrefType = dyn_cast<BaseMemRefType>(type)) {
    if (auto space = dyn_cast_or_null<pto::AddressSpaceAttr>(
            memrefType.getMemorySpace()))
      return space.getAddressSpace();
    if (auto intSpace =
            dyn_cast_or_null<IntegerAttr>(memrefType.getMemorySpace()))
      return static_cast<AddressSpace>(intSpace.getInt());
  }
  return std::nullopt;
}

static Type getPointeeElementType(Type type) {
  if (auto ptrType = dyn_cast<pto::PtrType>(type))
    return ptrType.getElementType();
  if (auto memrefType = dyn_cast<BaseMemRefType>(type))
    return memrefType.getElementType();
  return Type();
}

static std::optional<uint64_t> elementByteSize(Type type) {
  Type elem = getPointeeElementType(type);
  if (!elem)
    return std::nullopt;
  unsigned bytes = pto::getPTOStorageElemByteSize(elem);
  if (bytes == 0)
    return std::nullopt;
  return uint64_t(bytes);
}

static std::optional<uint64_t> vregElementCount(Type type) {
  if (auto vreg = dyn_cast<pto::VRegType>(type))
    return uint64_t(vreg.getElementCount());
  return std::nullopt;
}

static AffineByteExpr extractAffineElement(Value value, ArrayRef<Value> ivs);

static AffineByteExpr extractAffineExpr(mlir::AffineExpr expr, ValueRange dims,
                                        ValueRange symbols,
                                        ArrayRef<Value> ivs) {
  AffineByteExpr result;
  if (auto constant = dyn_cast<mlir::AffineConstantExpr>(expr)) {
    result.constant = constant.getValue();
    return result;
  }
  if (auto dim = dyn_cast<mlir::AffineDimExpr>(expr)) {
    if (dim.getPosition() >= dims.size()) {
      result.exact = false;
      return result;
    }
    return extractAffineElement(dims[dim.getPosition()], ivs);
  }
  if (auto symbol = dyn_cast<mlir::AffineSymbolExpr>(expr)) {
    if (symbol.getPosition() >= symbols.size()) {
      result.exact = false;
      return result;
    }
    return extractAffineElement(symbols[symbol.getPosition()], ivs);
  }
  auto binary = dyn_cast<mlir::AffineBinaryOpExpr>(expr);
  if (!binary) {
    result.exact = false;
    return result;
  }
  auto lhs = extractAffineExpr(binary.getLHS(), dims, symbols, ivs);
  auto rhs = extractAffineExpr(binary.getRHS(), dims, symbols, ivs);
  if (!lhs.exact || !rhs.exact) {
    result.exact = false;
    return result;
  }
  switch (binary.getKind()) {
  case mlir::AffineExprKind::Add:
    return lhs.combine(rhs);
  case mlir::AffineExprKind::Mul:
    if (rhs.isConstant()) {
      lhs.scale(rhs.constant);
      return lhs;
    }
    if (lhs.isConstant()) {
      rhs.scale(lhs.constant);
      return rhs;
    }
    result.exact = false;
    return result;
  default:
    // FloorDiv/CeilDiv/Mod need a bounded integer-domain model.  Keeping them
    // inexact is required for correctness: the caller will conservatively
    // retain a hazard instead of proving a false NoAlias.
    result.exact = false;
    return result;
  }
}

static AffineByteExpr extractAffineElement(Value value, ArrayRef<Value> ivs) {
  AffineByteExpr expr;

  APInt c;
  if (matchPattern(value, m_ConstantInt(&c))) {
    expr.constant = c.getSExtValue();
    return expr;
  }

  for (Value iv : ivs) {
    if (value == iv) {
      expr.coefficients.push_back({value, 1});
      return expr;
    }
  }

  if (auto addIOp = value.getDefiningOp<arith::AddIOp>()) {
    auto lhs = extractAffineElement(addIOp.getLhs(), ivs);
    auto rhs = extractAffineElement(addIOp.getRhs(), ivs);
    if (!lhs.exact || !rhs.exact) {
      expr.exact = false;
      return expr;
    }
    return lhs.combine(rhs);
  }
  if (auto subIOp = value.getDefiningOp<arith::SubIOp>()) {
    auto lhs = extractAffineElement(subIOp.getLhs(), ivs);
    auto rhs = extractAffineElement(subIOp.getRhs(), ivs);
    if (!lhs.exact || !rhs.exact) {
      expr.exact = false;
      return expr;
    }
    rhs.scale(-1);
    return lhs.combine(rhs);
  }
  if (auto mulIOp = value.getDefiningOp<arith::MulIOp>()) {
    APInt mc;
    if (matchPattern(mulIOp.getLhs(), m_ConstantInt(&mc))) {
      auto rhs = extractAffineElement(mulIOp.getRhs(), ivs);
      if (!rhs.exact)
        return rhs;
      rhs.scale(mc.getSExtValue());
      return rhs;
    }
    if (matchPattern(mulIOp.getRhs(), m_ConstantInt(&mc))) {
      auto lhs = extractAffineElement(mulIOp.getLhs(), ivs);
      if (!lhs.exact)
        return lhs;
      lhs.scale(mc.getSExtValue());
      return lhs;
    }
    expr.exact = false;
    return expr;
  }
  if (auto castOp = value.getDefiningOp<arith::IndexCastOp>())
    return extractAffineElement(castOp.getIn(), ivs);
  if (auto castOp = value.getDefiningOp<arith::IndexCastUIOp>())
    return extractAffineElement(castOp.getIn(), ivs);
  if (auto extOp = value.getDefiningOp<arith::ExtSIOp>())
    return extractAffineElement(extOp.getIn(), ivs);
  if (auto extOp = value.getDefiningOp<arith::ExtUIOp>())
    return extractAffineElement(extOp.getIn(), ivs);
  if (auto truncOp = value.getDefiningOp<arith::TruncIOp>())
    return extractAffineElement(truncOp.getIn(), ivs);

  if (auto apply = value.getDefiningOp<affine::AffineApplyOp>()) {
    AffineMap map = apply.getAffineMap();
    if (map.getNumResults() != 1) {
      AffineByteExpr result;
      result.exact = false;
      return result;
    }
    ValueRange operands = apply.getMapOperands();
    unsigned numDims = map.getNumDims();
    if (operands.size() != numDims + map.getNumSymbols()) {
      AffineByteExpr result;
      result.exact = false;
      return result;
    }
    return extractAffineExpr(map.getResult(0), operands.take_front(numDims),
                             operands.drop_front(numDims), ivs);
  }

  expr.coefficients.push_back({value, 1});
  expr.exact = false;
  return expr;
}

static AffineByteExpr extractSubviewElementOffset(memref::SubViewOp subview,
                                                  ArrayRef<Value> ivs) {
  AffineByteExpr result;
  auto sourceType = dyn_cast<MemRefType>(subview.getSource().getType());
  if (!sourceType) {
    result.exact = false;
    return result;
  }

  SmallVector<int64_t> sourceStrides;
  int64_t ignoredSourceOffset = ShapedType::kDynamic;
  if (failed(mlir::pto::getPTOMemRefStridesAndOffset(sourceType, sourceStrides,
                                                     ignoredSourceOffset))) {
    result.exact = false;
    return result;
  }

  auto mixedOffsets = subview.getMixedOffsets();
  if (mixedOffsets.size() > sourceStrides.size()) {
    result.exact = false;
    return result;
  }

  for (auto [mixedOffset, stride] :
       llvm::zip(mixedOffsets, ArrayRef<int64_t>(sourceStrides))) {
    if (stride == ShapedType::kDynamic) {
      result.exact = false;
      return result;
    }

    AffineByteExpr offset;
    if (auto attr = mixedOffset.dyn_cast<Attribute>()) {
      auto integer = dyn_cast<IntegerAttr>(attr);
      if (!integer) {
        result.exact = false;
        return result;
      }
      offset.constant = integer.getInt();
    } else if (auto value = mixedOffset.dyn_cast<Value>()) {
      offset = extractAffineElement(value, ivs);
    } else {
      result.exact = false;
      return result;
    }

    if (!offset.exact) {
      result.exact = false;
      return result;
    }
    offset.scale(stride);
    result.combine(offset);
  }
  return result;
}

static MemoryRootKind resolveRoot(Value ptr, AffineByteExpr &byteOffset,
                                  Value &rootOut,
                                  std::optional<uint64_t> &absoluteBase,
                                  ArrayRef<Value> ivs,
                                  bool allowIterArgInit = false) {
  if (auto castOp = ptr.getDefiningOp<pto::CastPtrOp>()) {
    Value input = castOp.getInput();
    if (isa<IntegerType>(input.getType()) || input.getType().isIndex()) {
      APInt v;
      if (matchPattern(input, m_ConstantInt(&v))) {
        absoluteBase = uint64_t(v.getZExtValue());
        rootOut = ptr;
        return MemoryRootKind::Absolute;
      }
      rootOut = ptr;
      return MemoryRootKind::Unknown;
    }
    return resolveRoot(input, byteOffset, rootOut, absoluteBase, ivs,
                       allowIterArgInit);
  }

  if (auto addOp = ptr.getDefiningOp<pto::AddPtrOp>()) {
    auto elemOffset = extractAffineElement(addOp.getOffset(), ivs);
    std::optional<uint64_t> bytes = elementByteSize(addOp.getType());
    if (!bytes || !elemOffset.exact) {
      rootOut = ptr;
      return MemoryRootKind::Unknown;
    }
    elemOffset.scale(int64_t(*bytes));
    byteOffset.combine(elemOffset);
    return resolveRoot(addOp.getPtr(), byteOffset, rootOut, absoluteBase, ivs,
                       allowIterArgInit);
  }

  if (auto subview = ptr.getDefiningOp<memref::SubViewOp>()) {
    auto subviewOffset = extractSubviewElementOffset(subview, ivs);
    auto bytes = elementByteSize(subview.getSource().getType());
    if (!subviewOffset.exact || !bytes)
      byteOffset.exact = false;
    else {
      subviewOffset.scale(int64_t(*bytes));
      byteOffset.combine(subviewOffset);
    }
    return resolveRoot(subview.getSource(), byteOffset, rootOut, absoluteBase,
                       ivs, allowIterArgInit);
  }
  if (auto cast = ptr.getDefiningOp<memref::CastOp>())
    return resolveRoot(cast.getSource(), byteOffset, rootOut, absoluteBase,
                       ivs, allowIterArgInit);
  if (auto tileAddr = ptr.getDefiningOp<pto::TileBufAddrOp>())
    return resolveRoot(tileAddr.getSrc(), byteOffset, rootOut, absoluteBase,
                       ivs, allowIterArgInit);
  if (auto rc = ptr.getDefiningOp<memref::ReinterpretCastOp>()) {
    byteOffset.exact = false;
    return resolveRoot(rc.getSource(), byteOffset, rootOut, absoluteBase, ivs,
                       allowIterArgInit);
  }
  if (auto msc = ptr.getDefiningOp<memref::MemorySpaceCastOp>())
    return resolveRoot(msc.getSource(), byteOffset, rootOut, absoluteBase, ivs,
                       allowIterArgInit);

  // Alloc/allocation-like memrefs are unique storage objects.  Preserve that
  // fact separately from generic symbolic roots so distinct allocations can be
  // proven NoAlias without inventing physical addresses.
  if (isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(ptr.getDefiningOp())) {
    rootOut = ptr;
    return MemoryRootKind::ProvenAllocation;
  }

  // scf.for region iter-argument: when allowed (post-update stores whose
  // destination base is carried through the loop), resolve the base from the
  // loop's init operand. The updated_base result advances the pointer only
  // forward across iterations, so the init value is the lower bound of the
  // write range; the caller marks forcesMayAlias and leaves byteSize unknown
  // to encode the open upper bound.
  if (allowIterArgInit) {
    if (auto blockArg = dyn_cast<BlockArgument>(ptr)) {
      if (auto forOp = blockArg.getOwner()
                           ? dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())
                           : scf::ForOp()) {
        Block::BlockArgListType iterArgs = forOp.getRegionIterArgs();
        for (auto [idx, arg] : llvm::enumerate(iterArgs)) {
          if (arg == ptr) {
            Value init = forOp.getInits()[idx];
            return resolveRoot(init, byteOffset, rootOut, absoluteBase, ivs,
                               /*allowIterArgInit=*/true);
          }
        }
      }
    }
  }

  // Block argument, memref.alloc, or pto.alloc-style allocation: symbolic
  // root. Distinct SSA roots that are not proven to be the same allocation
  // compare MayAlias.
  rootOut = ptr;
  return MemoryRootKind::Symbolic;
}

} // namespace

int64_t AffineByteExpr::getCoeff(Value v) const {
  for (auto [val, coeff] : coefficients)
    if (val == v)
      return coeff;
  return 0;
}

AffineByteExpr &AffineByteExpr::scale(int64_t factor) {
  constant *= factor;
  for (auto &[val, coeff] : coefficients)
    coeff *= factor;
  return *this;
}

AffineByteExpr &AffineByteExpr::combine(const AffineByteExpr &other) {
  constant += other.constant;
  for (auto [val, coeff] : other.coefficients) {
    bool found = false;
    for (auto &[v, c] : coefficients) {
      if (v == val) {
        c += coeff;
        found = true;
        break;
      }
    }
    if (!found)
      coefficients.push_back({val, coeff});
  }
  return *this;
}

namespace {

// Fill the descriptor's offset/size fields for a contiguous access. `bytes`
// is the per-element byte size; `elemCount` the number of elements accessed.
static void fillContiguous(VecMemoryAccessDescriptor &desc, Value base,
                           Value offset, std::optional<uint64_t> bytes,
                           std::optional<uint64_t> elemCount,
                           ArrayRef<Value> ivs) {
  desc.base = base;
  desc.addressSpace = getTypeAddressSpace(base.getType());
  desc.byteOffset = extractAffineElement(offset, ivs);
  if (bytes && desc.byteOffset.exact)
    desc.byteOffset.scale(int64_t(*bytes));
  else if (!bytes)
    desc.byteOffset.exact = false;
  if (bytes && elemCount)
    desc.conservativeByteSize = (*bytes) * (*elemCount);
}

static std::optional<uint64_t> parseMaskPatternLaneCount(StringRef pattern,
                                                          uint64_t fullCount) {
  if (pattern == "PAT_ALL") {
    return fullCount;
  }
  if (!pattern.starts_with("PAT_VL")) {
    return std::nullopt;
  }

  uint64_t activeCount = 0;
  if (pattern.drop_front(6).getAsInteger(10, activeCount)) {
    return std::nullopt;
  }
  return std::min(activeCount, fullCount);
}

static std::optional<uint64_t> getMaskPatternLaneCount(Value mask,
                                                        uint64_t fullCount) {
  StringRef pattern;
  if (auto op = mask.getDefiningOp<pto::PsetB8Op>()) {
    pattern = op.getPattern();
  } else if (auto op = mask.getDefiningOp<pto::PsetB16Op>()) {
    pattern = op.getPattern();
  } else if (auto op = mask.getDefiningOp<pto::PsetB32Op>()) {
    pattern = op.getPattern();
  } else if (auto op = mask.getDefiningOp<pto::PgeB8Op>()) {
    pattern = op.getPattern();
  } else if (auto op = mask.getDefiningOp<pto::PgeB16Op>()) {
    pattern = op.getPattern();
  } else if (auto op = mask.getDefiningOp<pto::PgeB32Op>()) {
    pattern = op.getPattern();
  } else if (auto op = mask.getDefiningOp<pto::PltB8Op>()) {
    APInt value;
    bool isConstant = matchPattern(op.getScalar(), m_ConstantInt(&value));
    if (!isConstant || value.isNegative()) {
      return std::nullopt;
    }
    return std::min(value.getZExtValue(), fullCount);
  } else if (auto op = mask.getDefiningOp<pto::PltB16Op>()) {
    APInt value;
    bool isConstant = matchPattern(op.getScalar(), m_ConstantInt(&value));
    if (!isConstant || value.isNegative()) {
      return std::nullopt;
    }
    return std::min(value.getZExtValue(), fullCount);
  } else if (auto op = mask.getDefiningOp<pto::PltB32Op>()) {
    APInt value;
    bool isConstant = matchPattern(op.getScalar(), m_ConstantInt(&value));
    if (!isConstant || value.isNegative()) {
      return std::nullopt;
    }
    return std::min(value.getZExtValue(), fullCount);
  }

  if (pattern.empty()) {
    return std::nullopt;
  }
  return parseMaskPatternLaneCount(pattern, fullCount);
}

static std::optional<uint64_t>
maskedStoredElementCount(Value mask, std::optional<uint64_t> unmaskedCount) {
  if (!unmaskedCount) {
    return std::nullopt;
  }
  auto activeCount = getMaskPatternLaneCount(mask, *unmaskedCount);
  if (!activeCount) {
    return unmaskedCount;
  }
  return activeCount;
}

// Return the number of contiguous destination elements written by `vsts`.
// Pack distributions write only the packed payload, not the full source-vreg
// storage width. Round up so unusual lane counts remain conservative.
static std::optional<uint64_t> vstsStoredElementCount(pto::VstsOp op) {
  auto count = vregElementCount(op.getValue().getType());
  if (!count)
    return std::nullopt;
  auto dist = op.getDist();
  if (!dist)
    return count;
  if (*dist == "PK_B16" || *dist == "PK_B32" || *dist == "PK_B64")
    return (*count + 1) / 2;
  if (*dist == "PK4_B32")
    return (*count + 3) / 4;
  return count;
}

} // namespace

FailureOr<VecMemoryAccessDescriptor>
vecscopemembar::buildAccessDescriptor(Operation *op, ArrayRef<Value> ivs) {
  VecMemoryAccessDescriptor desc;

  if (auto uvld = dyn_cast<pto::UvldOp>(op)) {
    desc.kind = VecScopeAccessKind::Load;
    fillContiguous(desc, uvld.getSource(), uvld.getOffset(),
                   elementByteSize(uvld.getSource().getType()),
                   vregElementCount(uvld.getResult().getType()), ivs);
    return desc;
  }
  if (auto vlds = dyn_cast<pto::VldsOp>(op)) {
    desc.kind = VecScopeAccessKind::Load;
    fillContiguous(desc, vlds.getSource(), vlds.getOffset(),
                   elementByteSize(vlds.getSource().getType()),
                   vregElementCount(vlds.getResult().getType()), ivs);
    return desc;
  }
  if (auto vsts = dyn_cast<pto::VstsOp>(op)) {
    desc.kind = VecScopeAccessKind::Store;
    auto elemCount =
        maskedStoredElementCount(vsts.getMask(), vstsStoredElementCount(vsts));
    fillContiguous(desc, vsts.getDestination(), vsts.getOffset(),
                   elementByteSize(vsts.getDestination().getType()),
                   elemCount, ivs);
    return desc;
  }
  if (auto vldsx2 = dyn_cast<pto::Vldsx2Op>(op)) {
    desc.kind = VecScopeAccessKind::Load;
    Value src = vldsx2.getSource();
    auto lo = vregElementCount(vldsx2.getLow().getType());
    auto hi = vregElementCount(vldsx2.getHigh().getType());
    std::optional<uint64_t> total;
    if (lo && hi)
      total = *lo + *hi;
    fillContiguous(desc, src, vldsx2.getOffset(),
                   elementByteSize(src.getType()), total, ivs);
    return desc;
  }
  if (auto vstsx2 = dyn_cast<pto::Vstsx2Op>(op)) {
    desc.kind = VecScopeAccessKind::Store;
    Value dst = vstsx2.getDestination();
    auto lo = vregElementCount(vstsx2.getLow().getType());
    auto hi = vregElementCount(vstsx2.getHigh().getType());
    std::optional<uint64_t> total;
    if (lo && hi)
      total = *lo + *hi;
    total = maskedStoredElementCount(vstsx2.getMask(), total);
    fillContiguous(desc, dst, vstsx2.getOffset(),
                   elementByteSize(dst.getType()), total, ivs);
    return desc;
  }
  if (auto vsstb = dyn_cast<pto::VsstbOp>(op)) {
    desc.kind = VecScopeAccessKind::Store;
    desc.base = vsstb.getDestination();
    desc.addressSpace = getTypeAddressSpace(desc.base.getType());
    // Strided/post-update stores can cover a non-contiguous footprint and may
    // carry their next base through scf.for.  Model them conservatively until
    // block/repeat stride ranges are represented explicitly.
    desc.forcesMayAlias = true;
    return desc;
  }

  // Gather: load, force MayAlias.
  if (isa<pto::Vgather2Op, pto::VgatherbOp, pto::Vgather2BcOp>(op)) {
    desc.kind = VecScopeAccessKind::Load;
    desc.forcesMayAlias = true;
    Value src;
    if (auto g = dyn_cast<pto::Vgather2Op>(op))
      src = g.getSource();
    else if (auto g = dyn_cast<pto::VgatherbOp>(op))
      src = g.getSource();
    else if (auto g = dyn_cast<pto::Vgather2BcOp>(op))
      src = g.getSource();
    if (src) {
      desc.base = src;
      desc.addressSpace = getTypeAddressSpace(src.getType());
    }
    return desc;
  }
  // Scatter: store, force MayAlias.
  if (auto vscatter = dyn_cast<pto::VscatterOp>(op)) {
    desc.kind = VecScopeAccessKind::Store;
    desc.forcesMayAlias = true;
    Value dst = vscatter.getDestination();
    desc.base = dst;
    desc.addressSpace = getTypeAddressSpace(dst.getType());
    return desc;
  }

  // Keep newly added or stateful UB vector-memory operations analyzable even
  // before they gain an exact footprint model. The analysis discovers these
  // operations through MemoryEffectOpInterface, so rejecting an operation
  // here would turn a conservative optimization limitation into a compiler
  // failure for otherwise valid VPTO.
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
  cast<MemoryEffectOpInterface>(op).getEffects(effects);
  for (const auto &effect : effects) {
    Value value = effect.getValue();
    if (!value || getTypeAddressSpace(value.getType()) != AddressSpace::VEC)
      continue;
    desc.base = value;
    desc.addressSpace = AddressSpace::VEC;
    desc.kind = isa<MemoryEffects::Write>(effect.getEffect())
                    ? VecScopeAccessKind::Store
                    : VecScopeAccessKind::Load;
    break;
  }

  // These stateful stores currently describe their UB base as a read effect
  // because the pointer itself participates in the state update. Classify the
  // actual UB access by operation semantics until their effects are split into
  // pointer-state reads and memory writes.
  if (isa<pto::PstuOp, pto::VstusOp, pto::VsturOp>(op))
    desc.kind = VecScopeAccessKind::Store;

  desc.forcesMayAlias = true;
  return desc;
}

// Reconstruct the full footprint (with resolved root) for an access. The
// descriptor stores the base + byte offset + size; here we resolve the root
// provenance from `base` so two accesses can be compared at root equality.
VecScopeMemoryFootprint
vecscopemembar::footprintFromDescriptor(const VecMemoryAccessDescriptor &desc,
                                        ArrayRef<Value> ivs) {
  VecScopeMemoryFootprint fp;
  fp.addressSpace = desc.addressSpace;
  fp.byteOffset = desc.byteOffset;
  fp.byteSize = desc.conservativeByteSize;
  fp.forcesMayAlias = desc.forcesMayAlias;
  if (desc.forcesMayAlias) {
    // Non-contiguous footprint (vsstb/gather/scatter) cannot be precisely
    // sized, but its base provenance is still resolvable. Resolve the root
    // (allowing scf.for iter-arg init backtracking for post-update stores)
    // so the access can be proven NoAlias against disjoint absolute or
    // distinct-allocation buffers. forcesMayAlias + byteSize=nullopt encodes
    // "lower bound known, open upper bound" and keeps same-root pairs
    // conservatively MayAlias.
    fp.rootKind =
        resolveRoot(desc.base, fp.byteOffset, fp.root, fp.absoluteBase, ivs,
                    /*allowIterArgInit=*/true);
    return fp;
  }
  fp.rootKind =
      resolveRoot(desc.base, fp.byteOffset, fp.root, fp.absoluteBase, ivs);
  return fp;
}

using WideInt = __int128_t;

static bool intervalsOverlap(WideInt aLo, uint64_t aSize, WideInt bLo,
                             uint64_t bSize) {
  // Use a signed type wider than every input so adding an offset or a size
  // cannot wrap at 64 bits.
  WideInt aEnd = aLo + static_cast<WideInt>(aSize);
  WideInt bEnd = bLo + static_cast<WideInt>(bSize);
  return aLo < bEnd && bLo < aEnd;
}

VecScopeAliasResult
vecscopemembar::aliasSameIteration(const VecScopeMemoryFootprint &producer,
                                   const VecScopeMemoryFootprint &consumer) {
  // Rule 1: different memory spaces -> NoAlias.
  if (producer.addressSpace && consumer.addressSpace &&
      *producer.addressSpace != *consumer.addressSpace)
    return VecScopeAliasResult::NoAlias;

  // Rule 2: gather/scatter forces MayAlias within same space, but only after
  // trying to prove the (possibly open-ended) footprints disjoint at the root
  // level. This lets a post-update vsstb whose destination base resolves to
  // an absolute address be proven NoAlias against other absolute buffers that
  // live entirely below its write-range lower bound, and against distinct
  // proven allocations.
  if (producer.forcesMayAlias || consumer.forcesMayAlias) {
    // Both absolute with closed ranges: compare intervals.
    if (producer.rootKind == MemoryRootKind::Absolute &&
        consumer.rootKind == MemoryRootKind::Absolute) {
      if (producer.absoluteBase && consumer.absoluteBase &&
          producer.byteSize && consumer.byteSize &&
          producer.byteOffset.exact && consumer.byteOffset.exact &&
          producer.byteOffset.isConstant() && consumer.byteOffset.isConstant()) {
        WideInt producerStart =
            static_cast<WideInt>(*producer.absoluteBase) +
            static_cast<WideInt>(producer.byteOffset.constant);
        WideInt consumerStart =
            static_cast<WideInt>(*consumer.absoluteBase) +
            static_cast<WideInt>(consumer.byteOffset.constant);
        if (producerStart >= 0 && consumerStart >= 0 &&
            !intervalsOverlap(producerStart, *producer.byteSize, consumerStart,
                              *consumer.byteSize))
          return VecScopeAliasResult::NoAlias;
      }
      // One side open-ended (byteSize unknown, e.g. vsstb post-update): the
      // open side's base B is the lower bound of its write range. The other
      // side's closed interval [a, a+s) is disjoint iff it ends at or before
      // B (open side is consumer) or starts at/after B+s (open side is
      // producer). Only the "ends before B" case is provable since the open
      // upper bound is unknown.
      if (producer.absoluteBase && consumer.absoluteBase &&
          producer.byteOffset.exact && consumer.byteOffset.exact &&
          producer.byteOffset.isConstant() && consumer.byteOffset.isConstant()) {
        WideInt prodStart =
            static_cast<WideInt>(*producer.absoluteBase) +
            static_cast<WideInt>(producer.byteOffset.constant);
        WideInt consStart =
            static_cast<WideInt>(*consumer.absoluteBase) +
            static_cast<WideInt>(consumer.byteOffset.constant);
        if (prodStart >= 0 && consStart >= 0) {
          // Producer open-ended (e.g. vsstb): consumer [consStart, consStart+s)
          // ends at or before prodStart.
          if (producer.forcesMayAlias && !producer.byteSize && consumer.byteSize &&
              consStart + static_cast<WideInt>(*consumer.byteSize) <= prodStart)
            return VecScopeAliasResult::NoAlias;
          // Consumer open-ended: producer [prodStart, prodStart+s) ends at or
          // before consStart.
          if (consumer.forcesMayAlias && !consumer.byteSize && producer.byteSize &&
              prodStart + static_cast<WideInt>(*producer.byteSize) <= consStart)
            return VecScopeAliasResult::NoAlias;
        }
      }
    }
    // Distinct proven allocations cannot alias even with forcesMayAlias.
    if (producer.rootKind == MemoryRootKind::ProvenAllocation &&
        consumer.rootKind == MemoryRootKind::ProvenAllocation &&
        producer.root != consumer.root)
      return VecScopeAliasResult::NoAlias;
    return VecScopeAliasResult::MayAlias;
  }

  // Rule 3: both absolute -> compare absolute intervals.
  if (producer.rootKind == MemoryRootKind::Absolute &&
      consumer.rootKind == MemoryRootKind::Absolute) {
    if (!producer.absoluteBase || !consumer.absoluteBase ||
        !producer.byteSize || !consumer.byteSize ||
        !producer.byteOffset.exact || !consumer.byteOffset.exact ||
        !producer.byteOffset.isConstant() || !consumer.byteOffset.isConstant())
      return VecScopeAliasResult::MayAlias;
    WideInt producerStart = static_cast<WideInt>(*producer.absoluteBase) +
                            static_cast<WideInt>(producer.byteOffset.constant);
    WideInt consumerStart = static_cast<WideInt>(*consumer.absoluteBase) +
                            static_cast<WideInt>(consumer.byteOffset.constant);
    if (producerStart < 0 || consumerStart < 0)
      return VecScopeAliasResult::MayAlias;
    if (intervalsOverlap(producerStart, *producer.byteSize, consumerStart,
                         *consumer.byteSize))
      return VecScopeAliasResult::MustOrPartialAlias;
    return VecScopeAliasResult::NoAlias;
  }

  // Absolute vs symbolic/unknown: cannot prove disjoint -> MayAlias.
  if (producer.rootKind == MemoryRootKind::Absolute ||
      consumer.rootKind == MemoryRootKind::Absolute)
    return VecScopeAliasResult::MayAlias;

  // Rule 7: unknown provenance -> MayAlias.
  if (producer.rootKind == MemoryRootKind::Unknown ||
      consumer.rootKind == MemoryRootKind::Unknown)
    return VecScopeAliasResult::MayAlias;

  // Rule 6: different symbolic roots -> MayAlias.
  if (producer.rootKind == MemoryRootKind::ProvenAllocation &&
      consumer.rootKind == MemoryRootKind::ProvenAllocation &&
      producer.root != consumer.root)
    return VecScopeAliasResult::NoAlias;

  // Generic symbolic roots (for example function arguments) may alias even
  // when represented by different SSA values.
  if (producer.root != consumer.root)
    return VecScopeAliasResult::MayAlias;

  // Rule 5: same root -> compare relative intervals.
  if (!producer.byteSize || !consumer.byteSize || !producer.byteOffset.exact ||
      !consumer.byteOffset.exact)
    return VecScopeAliasResult::MayAlias;

  if (producer.byteOffset.isConstant() && consumer.byteOffset.isConstant()) {
    WideInt pLo = static_cast<WideInt>(producer.byteOffset.constant);
    WideInt cLo = static_cast<WideInt>(consumer.byteOffset.constant);
    if (intervalsOverlap(pLo, *producer.byteSize, cLo, *consumer.byteSize))
      return VecScopeAliasResult::MustOrPartialAlias;
    return VecScopeAliasResult::NoAlias;
  }

  // IV/symbolic offset terms: first version cannot prove disjointness across
  // the iteration domain -> MayAlias. Loop-carried Presburger reasoning lives
  // in the analysis module.
  return VecScopeAliasResult::MayAlias;
}
