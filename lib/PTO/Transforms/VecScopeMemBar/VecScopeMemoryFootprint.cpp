// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/VecScopeMemBar/VecScopeMemoryFootprint.h"

#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VPTOAddressSemantics.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

static std::optional<AddressSpace> getTypeAddressSpace(Type type)
{
    if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
        return ptrType.getMemorySpace().getAddressSpace();
    }
    if (auto memrefType = dyn_cast<BaseMemRefType>(type)) {
        if (auto space = dyn_cast_or_null<pto::AddressSpaceAttr>(memrefType.getMemorySpace())) {
            return space.getAddressSpace();
        }
        if (auto intSpace = dyn_cast_or_null<IntegerAttr>(memrefType.getMemorySpace())) {
            return static_cast<AddressSpace>(intSpace.getInt());
        }
    }
    return std::nullopt;
}

static Type getPointeeElementType(Type type)
{
    if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
        return ptrType.getElementType();
    }
    if (auto memrefType = dyn_cast<BaseMemRefType>(type)) {
        return memrefType.getElementType();
    }
    return Type();
}

static std::optional<uint64_t> elementByteSize(Type type)
{
    Type elem = getPointeeElementType(type);
    if (!elem) {
        return std::nullopt;
    }
    unsigned bytes = pto::getPTOStorageElemByteSize(elem);
    if (bytes == 0) {
        return std::nullopt;
    }
    return uint64_t(bytes);
}

static std::optional<uint64_t> vregElementCount(Type type)
{
    if (auto vreg = dyn_cast<pto::VRegType>(type)) {
        return uint64_t(vreg.getElementCount());
    }
    return std::nullopt;
}

// An empty expression marked inexact: the caller must conservatively retain
// the hazard instead of proving a false NoAlias.
static AffineByteExpr makeInexactExpr()
{
    AffineByteExpr expr;
    expr.exact = false;
    return expr;
}

static AffineByteExpr extractAffineElement(Value value, ArrayRef<Value> ivs);

static AffineByteExpr extractAffineExpr(mlir::AffineExpr expr, ValueRange dims, ValueRange symbols, ArrayRef<Value> ivs)
{
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

// Recognize the arithmetic op shapes with an affine form: add/sub and mul by
// a constant. Returns nullopt when `value` is not defined by one of them.
static std::optional<AffineByteExpr> extractArithAffineElement(Value value, ArrayRef<Value> ivs)
{
    if (auto addIOp = value.getDefiningOp<arith::AddIOp>()) {
        auto lhs = extractAffineElement(addIOp.getLhs(), ivs);
        auto rhs = extractAffineElement(addIOp.getRhs(), ivs);
        if (!lhs.exact || !rhs.exact) {
            return makeInexactExpr();
        }
        return lhs.combine(rhs);
    }
    if (auto subIOp = value.getDefiningOp<arith::SubIOp>()) {
        auto lhs = extractAffineElement(subIOp.getLhs(), ivs);
        auto rhs = extractAffineElement(subIOp.getRhs(), ivs);
        if (!lhs.exact || !rhs.exact) {
            return makeInexactExpr();
        }
        rhs.scale(-1);
        return lhs.combine(rhs);
    }
    if (auto mulIOp = value.getDefiningOp<arith::MulIOp>()) {
        APInt mc;
        if (matchPattern(mulIOp.getLhs(), m_ConstantInt(&mc))) {
            auto rhs = extractAffineElement(mulIOp.getRhs(), ivs);
            if (!rhs.exact) {
                return rhs;
            }
            rhs.scale(mc.getSExtValue());
            return rhs;
        }
        if (matchPattern(mulIOp.getRhs(), m_ConstantInt(&mc))) {
            auto lhs = extractAffineElement(mulIOp.getLhs(), ivs);
            if (!lhs.exact) {
                return lhs;
            }
            lhs.scale(mc.getSExtValue());
            return lhs;
        }
        return makeInexactExpr();
    }
    return std::nullopt;
}

// The width-changing no-ops with an affine form: index casts, sign/zero
// extensions, and truncation all forward to their input. Returns nullopt
// when `value` is not defined by one of them.
static std::optional<AffineByteExpr> extractCastAffineElement(Value value, ArrayRef<Value> ivs)
{
    if (auto castOp = value.getDefiningOp<arith::IndexCastOp>()) {
        return extractAffineElement(castOp.getIn(), ivs);
    }
    if (auto castOp = value.getDefiningOp<arith::IndexCastUIOp>()) {
        return extractAffineElement(castOp.getIn(), ivs);
    }
    if (auto extOp = value.getDefiningOp<arith::ExtSIOp>()) {
        return extractAffineElement(extOp.getIn(), ivs);
    }
    if (auto extOp = value.getDefiningOp<arith::ExtUIOp>()) {
        return extractAffineElement(extOp.getIn(), ivs);
    }
    if (auto truncOp = value.getDefiningOp<arith::TruncIOp>()) {
        return extractAffineElement(truncOp.getIn(), ivs);
    }
    return std::nullopt;
}

// Recognize affine.apply: a single-result map whose operands decompose
// affinely. Returns nullopt when `value` is not an affine.apply.
static std::optional<AffineByteExpr> extractAffineApplyElement(Value value, ArrayRef<Value> ivs)
{
    auto apply = value.getDefiningOp<affine::AffineApplyOp>();
    if (!apply) {
        return std::nullopt;
    }
    AffineMap map = apply.getAffineMap();
    if (map.getNumResults() != 1) {
        return makeInexactExpr();
    }
    ValueRange operands = apply.getMapOperands();
    unsigned numDims = map.getNumDims();
    if (operands.size() != numDims + map.getNumSymbols()) {
        return makeInexactExpr();
    }
    return extractAffineExpr(map.getResult(0), operands.take_front(numDims), operands.drop_front(numDims), ivs);
}

static AffineByteExpr extractAffineElement(Value value, ArrayRef<Value> ivs)
{
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

    if (auto arith = extractArithAffineElement(value, ivs)) {
        return *arith;
    }
    if (auto cast = extractCastAffineElement(value, ivs)) {
        // A width-changing cast of an unbounded symbol can wrap. Preserve
        // only the existing bounded-IV model through these casts.
        for (auto [term, coefficient] : cast->coefficients) {
            if (coefficient != 0 && !llvm::is_contained(ivs, term)) {
                return makeInexactExpr();
            }
        }
        return *cast;
    }
    if (auto applied = extractAffineApplyElement(value, ivs)) {
        return *applied;
    }

    // Opaque index values can cancel within one execution. Dependence queries
    // must reject remaining non-IV symbols rather than silently dropping them.
    expr.coefficients.push_back({value, 1});
    auto arg = dyn_cast<BlockArgument>(value);
    expr.exact = value.getType().isIndex() && arg && isa<func::FuncOp>(arg.getOwner()->getParentOp());
    return expr;
}

static AffineByteExpr extractSubviewElementOffset(memref::SubViewOp subview, ArrayRef<Value> ivs)
{
    AffineByteExpr result;
    auto sourceType = dyn_cast<MemRefType>(subview.getSource().getType());
    if (!sourceType) {
        result.exact = false;
        return result;
    }

    SmallVector<int64_t> sourceStrides;
    int64_t ignoredSourceOffset = ShapedType::kDynamic;
    if (failed(mlir::pto::getPTOMemRefStridesAndOffset(sourceType, sourceStrides, ignoredSourceOffset))) {
        result.exact = false;
        return result;
    }

    auto mixedOffsets = subview.getMixedOffsets();
    if (mixedOffsets.size() > sourceStrides.size()) {
        result.exact = false;
        return result;
    }

    for (auto [mixedOffset, stride] : llvm::zip(mixedOffsets, ArrayRef<int64_t>(sourceStrides))) {
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

static MemoryRootKind resolveRoot(
    Value ptr, AffineByteExpr& byteOffset, Value& rootOut, std::optional<uint64_t>& absoluteBase, ArrayRef<Value> ivs,
    bool allowIterArgInit = false);

// pto.cast_ptr from an integer/index value: an absolute address when the
// integer is constant, else an unknown root.
static std::optional<MemoryRootKind> tryResolveCastPtr(
    Value ptr, AffineByteExpr& byteOffset, Value& rootOut, std::optional<uint64_t>& absoluteBase, ArrayRef<Value> ivs,
    bool allowIterArgInit)
{
    auto castOp = ptr.getDefiningOp<pto::CastPtrOp>();
    if (!castOp) {
        return std::nullopt;
    }
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
    return resolveRoot(input, byteOffset, rootOut, absoluteBase, ivs, allowIterArgInit);
}

// pto.add_ptr: fold the affine element offset scaled by the element byte
// size into the byte offset, then keep resolving the base pointer.
static std::optional<MemoryRootKind> tryResolveAddPtr(
    Value ptr, AffineByteExpr& byteOffset, Value& rootOut, std::optional<uint64_t>& absoluteBase, ArrayRef<Value> ivs,
    bool allowIterArgInit)
{
    auto addOp = ptr.getDefiningOp<pto::AddPtrOp>();
    if (!addOp) {
        return std::nullopt;
    }
    auto elemOffset = extractAffineElement(addOp.getOffset(), ivs);
    std::optional<uint64_t> bytes = elementByteSize(addOp.getType());
    if (!bytes || !elemOffset.exact) {
        rootOut = ptr;
        return MemoryRootKind::Unknown;
    }
    elemOffset.scale(int64_t(*bytes));
    byteOffset.combine(elemOffset);
    return resolveRoot(addOp.getPtr(), byteOffset, rootOut, absoluteBase, ivs, allowIterArgInit);
}

// memref.subview: fold the subview element offset scaled by the source
// element byte size into the byte offset. An inexact offset only taints the
// byte offset; the root keeps resolving.
static std::optional<MemoryRootKind> tryResolveSubview(
    Value ptr, AffineByteExpr& byteOffset, Value& rootOut, std::optional<uint64_t>& absoluteBase, ArrayRef<Value> ivs,
    bool allowIterArgInit)
{
    auto subview = ptr.getDefiningOp<memref::SubViewOp>();
    if (!subview) {
        return std::nullopt;
    }
    auto subviewOffset = extractSubviewElementOffset(subview, ivs);
    auto bytes = elementByteSize(subview.getSource().getType());
    if (!subviewOffset.exact || !bytes) {
        byteOffset.exact = false;
    } else {
        subviewOffset.scale(int64_t(*bytes));
        byteOffset.combine(subviewOffset);
    }
    return resolveRoot(subview.getSource(), byteOffset, rootOut, absoluteBase, ivs, allowIterArgInit);
}

// Pointer-preserving memref casts: resolve through to the source. A
// reinterpret_cast also taints the byte offset (its own offset is not
// modelled).
static std::optional<MemoryRootKind> tryResolveMemrefCasts(
    Value ptr, AffineByteExpr& byteOffset, Value& rootOut, std::optional<uint64_t>& absoluteBase, ArrayRef<Value> ivs,
    bool allowIterArgInit)
{
    if (auto cast = ptr.getDefiningOp<memref::CastOp>()) {
        return resolveRoot(cast.getSource(), byteOffset, rootOut, absoluteBase, ivs, allowIterArgInit);
    }
    if (auto tileAddr = ptr.getDefiningOp<pto::TileBufAddrOp>()) {
        return resolveRoot(tileAddr.getSrc(), byteOffset, rootOut, absoluteBase, ivs, allowIterArgInit);
    }
    if (auto rc = ptr.getDefiningOp<memref::ReinterpretCastOp>()) {
        byteOffset.exact = false;
        return resolveRoot(rc.getSource(), byteOffset, rootOut, absoluteBase, ivs, allowIterArgInit);
    }
    if (auto msc = ptr.getDefiningOp<memref::MemorySpaceCastOp>()) {
        return resolveRoot(msc.getSource(), byteOffset, rootOut, absoluteBase, ivs, allowIterArgInit);
    }
    return std::nullopt;
}

// Predicate address units and post-update offsets follow the shared IR contract.
static bool isPredicateMemory(Operation* op)
{
    return isa<pto::PldsOp, pto::PldiOp, pto::PstsOp, pto::PstiOp>(op);
}

static std::optional<std::pair<Value, int64_t>> predicatePointerAdvance(Value value)
{
    Operation* op = value.getDefiningOp();
    if (!op || !isPredicateMemory(op)) {
        return std::nullopt;
    }
    auto semantics = getDefaultVPTOAddressSemantics(op);
    auto update = semantics.postUpdate;
    APInt amount;
    if (!update || update->updatedBase != value || !update->advanceOperand ||
        !matchPattern(update->advanceOperand->get(), m_ConstantInt(&amount)) || amount.getBitWidth() > 64) {
        return std::nullopt;
    }
    auto unit = getVPTOAddressUnitBytes(op, update->advanceUnit, update->elementTypeSource);
    if (!unit) {
        return std::nullopt;
    }
    __int128 bytes = static_cast<__int128>(amount.getSExtValue()) * *unit;
    if (bytes < INT64_MIN || bytes > INT64_MAX) {
        return std::nullopt;
    }
    return std::make_pair(update->baseOperand->get(), static_cast<int64_t>(bytes));
}

// Trace a straight-line pointer update back to its own iter argument. Do not
// cross a region/conditional or another carried value. Bound the walk so even
// malformed or exceptionally long chains retain conservative synchronization.
static std::optional<int64_t> fixedPointerAdvance(Value value, Value initial)
{
    __int128 delta = 0;
    for (unsigned depth = 0; depth < 32; ++depth) {
        if (value == initial) {
            return static_cast<int64_t>(delta);
        }
        APInt amount;
        if (auto store = value.getDefiningOp<pto::VsstbOp>()) {
            if (value != store.getUpdatedBase() ||
                !matchPattern(store.getRepeatStride(), m_ConstantInt(&amount))) {
                return std::nullopt;
            }
            delta += static_cast<__int128>(amount.getZExtValue()) * 32;
            value = store.getDestination();
        } else if (auto update = predicatePointerAdvance(value)) {
            delta += update->second;
            value = update->first;
        } else if (auto add = value.getDefiningOp<pto::AddPtrOp>()) {
            auto bytes = elementByteSize(add.getType());
            if (!bytes || !matchPattern(add.getOffset(), m_ConstantInt(&amount)) ||
                amount.getBitWidth() > 64) {
                return std::nullopt;
            }
            delta += static_cast<__int128>(amount.getSExtValue()) * *bytes;
            value = add.getPtr();
        } else {
            return std::nullopt;
        }
        if (delta < INT64_MIN || delta > INT64_MAX) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

// Express a fixed recurrence in the surrounding IV coordinate system. Requiring
// divisibility avoids silently rounding (iv-lower)/step; other strides fall back.
static std::optional<MemoryRootKind> resolveFixedIterArgRoot(
    Value ptr, AffineByteExpr& byteOffset, Value& rootOut, std::optional<uint64_t>& absoluteBase,
    ArrayRef<Value> ivs)
{
    auto arg = dyn_cast<BlockArgument>(ptr);
    auto loop = arg ? dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp()) : scf::ForOp();
    if (!loop || arg.getArgNumber() == 0 || !llvm::is_contained(ivs, loop.getInductionVar())) {
        return std::nullopt;
    }
    unsigned index = arg.getArgNumber() - 1;
    auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
    if (!yield || index >= yield.getNumOperands()) {
        return std::nullopt;
    }
    auto delta = fixedPointerAdvance(yield.getOperand(index), ptr);
    APInt lower;
    APInt upper;
    APInt step;
    if (!delta || !matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
        !matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) ||
        !matchPattern(loop.getStep(), m_ConstantInt(&step)) ||
        lower.getBitWidth() > 64 || upper.getBitWidth() > 64 || step.getBitWidth() > 64) {
        return std::nullopt;
    }
    int64_t lb = lower.getSExtValue();
    int64_t ub = upper.getSExtValue();
    int64_t stride = step.getSExtValue();
    if (stride <= 0 || ub <= lb || *delta % stride != 0) {
        return std::nullopt;
    }
    __int128 count = (static_cast<__int128>(ub) - lb + stride - 1) / stride;
    __int128 extent = count * *delta;
    int64_t coefficient = *delta / stride;
    __int128 constant = -static_cast<__int128>(lb) * coefficient + byteOffset.constant;
    if (extent < INT64_MIN || extent > INT64_MAX || constant < INT64_MIN || constant > INT64_MAX ||
        byteOffset.getCoeff(loop.getInductionVar()) != 0) {
        return std::nullopt;
    }
    // Resolve the init separately so adding its constant cannot overflow the
    // recurrence adjustment. Nested/offset initializers retain their IV terms.
    AffineByteExpr initOffset;
    auto kind = resolveRoot(loop.getInits()[index], initOffset, rootOut, absoluteBase, ivs, false);
    constant += initOffset.constant;
    if (!initOffset.exact || constant < INT64_MIN || constant > INT64_MAX) {
        return MemoryRootKind::Unknown;
    }
    // combine() can retain zero coefficients after cancellation. Merge by
    // variable rather than appending duplicates that getCoeff() would hide.
    AffineByteExpr recurrence;
    recurrence.coefficients.push_back({loop.getInductionVar(), coefficient});
    byteOffset.constant = static_cast<int64_t>(constant);
    byteOffset.combine(recurrence);
    for (auto [iv, coeff] : initOffset.coefficients) {
        if (byteOffset.getCoeff(iv) != 0) {
            return MemoryRootKind::Unknown;
        }
    }
    // The initializer constant was already included in the checked sum above.
    initOffset.constant = 0;
    byteOffset.combine(initOffset);
    return byteOffset.exact ? kind : MemoryRootKind::Unknown;
}

// scf.for region iter-argument: when allowed (post-update stores whose
// destination base is carried through the loop), resolve the base from the
// loop's init operand. The updated_base result advances the pointer only
// forward across iterations, so the init value is the lower bound of the
// write range; the caller marks forcesMayAlias and leaves byteSize unknown
// to encode the open upper bound. Returns nullopt when `ptr` is not a
// carried iter argument.
static std::optional<MemoryRootKind> resolveIterArgInitRoot(
    Value ptr, AffineByteExpr& byteOffset, Value& rootOut, std::optional<uint64_t>& absoluteBase, ArrayRef<Value> ivs)
{
    auto blockArg = dyn_cast<BlockArgument>(ptr);
    if (!blockArg) {
        return std::nullopt;
    }
    auto forOp = blockArg.getOwner() ? dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp()) : scf::ForOp();
    if (!forOp) {
        return std::nullopt;
    }
    Block::BlockArgListType iterArgs = forOp.getRegionIterArgs();
    for (auto [idx, arg] : llvm::enumerate(iterArgs)) {
        if (arg == ptr) {
            Value init = forOp.getInits()[idx];
            return resolveRoot(
                init, byteOffset, rootOut, absoluteBase, ivs,
                true);
        }
    }
    return std::nullopt;
}

static MemoryRootKind resolveRoot(
    Value ptr, AffineByteExpr& byteOffset, Value& rootOut, std::optional<uint64_t>& absoluteBase, ArrayRef<Value> ivs,
    bool allowIterArgInit)
{
    if (auto store = ptr.getDefiningOp<pto::VsstbOp>()) {
        APInt repeat;
        if (ptr == store.getUpdatedBase() && matchPattern(store.getRepeatStride(), m_ConstantInt(&repeat))) {
            AffineByteExpr advance;
            advance.constant = static_cast<int64_t>(repeat.getZExtValue() * 32);
            byteOffset.combine(advance);
            return resolveRoot(store.getDestination(), byteOffset, rootOut, absoluteBase, ivs, allowIterArgInit);
        }
    }
    if (auto update = predicatePointerAdvance(ptr)) {
        AffineByteExpr advance;
        advance.constant = update->second;
        byteOffset.combine(advance);
        return resolveRoot(update->first, byteOffset, rootOut, absoluteBase, ivs, allowIterArgInit);
    }
    if (auto castPtr = tryResolveCastPtr(ptr, byteOffset, rootOut, absoluteBase, ivs, allowIterArgInit)) {
        return *castPtr;
    }
    if (auto addPtr = tryResolveAddPtr(ptr, byteOffset, rootOut, absoluteBase, ivs, allowIterArgInit)) {
        return *addPtr;
    }
    if (auto subview = tryResolveSubview(ptr, byteOffset, rootOut, absoluteBase, ivs, allowIterArgInit)) {
        return *subview;
    }
    if (auto cast = tryResolveMemrefCasts(ptr, byteOffset, rootOut, absoluteBase, ivs, allowIterArgInit)) {
        return *cast;
    }

    // Alloc/allocation-like memrefs are unique storage objects.  Preserve that
    // fact separately from generic symbolic roots so distinct allocations can be
    // proven NoAlias without inventing physical addresses.
    if (isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(ptr.getDefiningOp())) {
        rootOut = ptr;
        return MemoryRootKind::ProvenAllocation;
    }

    if (auto recurrence = resolveFixedIterArgRoot(ptr, byteOffset, rootOut, absoluteBase, ivs)) {
        return *recurrence;
    }

    if (allowIterArgInit) {
        if (auto viaInit = resolveIterArgInitRoot(ptr, byteOffset, rootOut, absoluteBase, ivs)) {
            return *viaInit;
        }
    }

    // Block argument, memref.alloc, or pto.alloc-style allocation: symbolic
    // root. Distinct SSA roots that are not proven to be the same allocation
    // compare MayAlias.
    rootOut = ptr;
    return MemoryRootKind::Symbolic;
}

} // namespace

int64_t AffineByteExpr::getCoeff(Value v) const
{
    for (auto [val, coeff] : coefficients) {
        if (val == v) {
            return coeff;
        }
    }
    return 0;
}

AffineByteExpr& AffineByteExpr::scale(int64_t factor)
{
    __int128 scaled = static_cast<__int128>(constant) * factor;
    if (scaled < INT64_MIN || scaled > INT64_MAX) {
        exact = false;
        return *this;
    }
    constant = static_cast<int64_t>(scaled);
    for (auto& [val, coeff] : coefficients) {
        scaled = static_cast<__int128>(coeff) * factor;
        if (scaled < INT64_MIN || scaled > INT64_MAX) {
            exact = false;
            return *this;
        }
        coeff = static_cast<int64_t>(scaled);
    }
    return *this;
}

AffineByteExpr& AffineByteExpr::combine(const AffineByteExpr& other)
{
    exact &= other.exact;
    __int128 sum = static_cast<__int128>(constant) + other.constant;
    if (sum < INT64_MIN || sum > INT64_MAX) {
        exact = false;
        return *this;
    }
    constant = static_cast<int64_t>(sum);
    for (auto [val, coeff] : other.coefficients) {
        bool found = false;
        for (auto& [v, c] : coefficients) {
            if (v == val) {
                sum = static_cast<__int128>(c) + coeff;
                if (sum < INT64_MIN || sum > INT64_MAX) {
                    exact = false;
                    return *this;
                }
                c = static_cast<int64_t>(sum);
                found = true;
                break;
            }
        }
        if (!found) {
            coefficients.push_back({val, coeff});
        }
    }
    return *this;
}

namespace {

// Fill the descriptor's offset/size fields for a contiguous access. `bytes`
// is the per-element byte size; `elemCount` the number of elements accessed.
static void fillContiguous(
    VecMemoryAccessDescriptor& desc, Value base, Value offset, std::optional<uint64_t> bytes,
    std::optional<uint64_t> elemCount, ArrayRef<Value> ivs)
{
    desc.base = base;
    desc.addressSpace = getTypeAddressSpace(base.getType());
    desc.byteOffset = extractAffineElement(offset, ivs);
    if (bytes && desc.byteOffset.exact) {
        desc.byteOffset.scale(int64_t(*bytes));
    } else if (!bytes) {
        desc.byteOffset.exact = false;
    }
    if (bytes && elemCount) {
        desc.conservativeByteSize = (*bytes) * (*elemCount);
    }
}

static std::optional<uint64_t> parseMaskPatternLaneCount(StringRef pattern, uint64_t fullCount)
{
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

static std::optional<uint64_t> getMaskPatternLaneCount(Value mask, uint64_t fullCount)
{
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

static std::optional<uint64_t> maskedStoredElementCount(Value mask, std::optional<uint64_t> unmaskedCount)
{
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
static std::optional<uint64_t> vstsStoredElementCount(pto::VstsOp op)
{
    auto count = vregElementCount(op.getValue().getType());
    if (!count) {
        return std::nullopt;
    }
    auto dist = op.getDist();
    if (!dist) {
        return count;
    }
    if (*dist == "PK_B16" || *dist == "PK_B32" || *dist == "PK_B64") {
        return (*count + 1) / 2;
    }
    if (*dist == "PK4_B32") {
        return (*count + 3) / 4;
    }
    return count;
}

// Return the number of contiguous bytes read from memory by a `vlds`
// distribution, when the access footprint can be modelled more precisely than
// the full result-vreg width. A scalar broadcast distribution (BRC_B8/B16/B32)
// reads a single hardware element whose width is fixed by the `_Bn` suffix
// (1/2/4 bytes), independent of the source element type, so the verifier
// accepts width-mismatched forms such as `BRC_B32` on `ptr<f16>` (4 bytes,
// i.e. two f16 elements). Reporting the footprint as `n/8` bytes keeps the
// model sound under those mismatches; reporting one source element would
// under-count (e.g. 2 bytes for `BRC_B32` on f16) and could drop a required
// loop-carried barrier. BRC_BLK reads an entire block and keeps the
// conservative vreg width; the unpack/upcast/shift distributions (US_*, DS_*,
// UNPK_*, E2B_*) read a different element type than they produce, so their
// read width is left conservatively at the full vreg width until the
// element-type pairing is modelled.
static std::optional<uint64_t> vldsBroadcastByteSize(pto::VldsOp op)
{
    auto dist = op.getDist();
    if (!dist) {
        return std::nullopt;
    }
    if (*dist == "BRC_B8") {
        return uint64_t(1);
    }
    if (*dist == "BRC_B16") {
        return uint64_t(2);
    }
    if (*dist == "BRC_B32") {
        return uint64_t(4);
    }
    return std::nullopt;
}

// Describe the single-vreg contiguous accesses: uvld/vlds loads and the vsts
// store. Returns nullopt when `op` is none of them.
static std::optional<VecMemoryAccessDescriptor> describeContiguousAccess(Operation* op, ArrayRef<Value> ivs)
{
    VecMemoryAccessDescriptor desc;

    // SIMT scalar load/store has an element-offset UB footprint when used
    // inside a vecscope.
    if (auto load = dyn_cast<pto::PTOLoadOp>(op)) {
        desc.kind = VecScopeAccessKind::Load;
        desc.unit = VecScopeAccessUnit::Scalar;
        fillContiguous(
            desc, load.getPtr(), load.getOffset(), elementByteSize(load.getPtr().getType()), uint64_t(1), ivs);
        return desc;
    }
    if (auto store = dyn_cast<pto::PTOStoreOp>(op)) {
        desc.kind = VecScopeAccessKind::Store;
        desc.unit = VecScopeAccessUnit::Scalar;
        fillContiguous(
            desc, store.getPtr(), store.getOffset(), elementByteSize(store.getPtr().getType()), uint64_t(1), ivs);
        return desc;
    }
    if (auto uvld = dyn_cast<pto::UvldOp>(op)) {
        desc.kind = VecScopeAccessKind::Load;
        fillContiguous(
            desc, uvld.getSource(), uvld.getOffset(), elementByteSize(uvld.getSource().getType()),
            vregElementCount(uvld.getResult().getType()), ivs);
        return desc;
    }
    if (auto vlds = dyn_cast<pto::VldsOp>(op)) {
        desc.kind = VecScopeAccessKind::Load;
        fillContiguous(
            desc, vlds.getSource(), vlds.getOffset(), elementByteSize(vlds.getSource().getType()),
            vregElementCount(vlds.getResult().getType()), ivs);
        // A scalar broadcast distribution reads a fixed-width hardware element
        // (BRC_B8/B16/B32 -> 1/2/4 bytes) regardless of source element type.
        // Override the full-vreg size with the precise broadcast footprint so the
        // analysis neither over-approximates (fabricating a cross-iteration RAW
        // against an adjacent buffer) nor under-reports (dropping a real RAW on a
        // width-mismatched form such as BRC_B32 on ptr<f16>).
        if (auto broadcastBytes = vldsBroadcastByteSize(vlds)) {
            desc.conservativeByteSize = *broadcastBytes;
        }
        return desc;
    }
    if (auto vsts = dyn_cast<pto::VstsOp>(op)) {
        desc.kind = VecScopeAccessKind::Store;
        auto elemCount = maskedStoredElementCount(vsts.getMask(), vstsStoredElementCount(vsts));
        fillContiguous(
            desc, vsts.getDestination(), vsts.getOffset(), elementByteSize(vsts.getDestination().getType()), elemCount,
            ivs);
        return desc;
    }
    return std::nullopt;
}

// A5 predicate payloads occupy 32 bytes (NORM), 16 bytes (US/PK),
// or 64 bytes (DS). Alignment is not the payload size for DS.
static std::optional<VecMemoryAccessDescriptor> describePredicateAccess(Operation* op, ArrayRef<Value> ivs)
{
    if (!isPredicateMemory(op) || getTargetArch(op) != PTOArch::A5) {
        return std::nullopt;
    }
    VecMemoryAccessDescriptor desc;
    desc.kind = isa<pto::PldsOp, pto::PldiOp>(op) ? VecScopeAccessKind::Load : VecScopeAccessKind::Store;
    auto semantics = getDefaultVPTOAddressSemantics(op);
    const auto& access = semantics.currentAccesses.front();
    desc.base = access.baseOperand->get();
    desc.addressSpace = getTypeAddressSpace(desc.base.getType());
    if (access.offset) {
        auto unit = getVPTOAddressUnitBytes(op, access.offset->unit, access.offset->elementTypeSource);
        desc.byteOffset = extractAffineElement(access.offset->operand->get(), ivs);
        if (unit) {
            desc.byteOffset.scale(*unit);
        } else {
            desc.byteOffset.exact = false;
        }
    }
    auto dist = op->getAttrOfType<StringAttr>("dist");
    if (dist && dist.getValue() == "NORM") {
        desc.conservativeByteSize = 32;
    } else if (dist && (dist.getValue() == "US" || dist.getValue() == "PK")) {
        desc.conservativeByteSize = 16;
    } else if (dist && dist.getValue() == "DS") {
        desc.conservativeByteSize = 64;
    }
    return desc;
}

// Describe the paired-vreg contiguous accesses: the vldsx2 load and the
// vstsx2 store. Returns nullopt when `op` is neither.
static std::optional<VecMemoryAccessDescriptor> describePairedContiguousAccess(Operation* op, ArrayRef<Value> ivs)
{
    VecMemoryAccessDescriptor desc;
    if (auto vldsx2 = dyn_cast<pto::Vldsx2Op>(op)) {
        desc.kind = VecScopeAccessKind::Load;
        Value src = vldsx2.getSource();
        auto lo = vregElementCount(vldsx2.getLow().getType());
        auto hi = vregElementCount(vldsx2.getHigh().getType());
        std::optional<uint64_t> total;
        if (lo && hi) {
            total = *lo + *hi;
        }
        fillContiguous(desc, src, vldsx2.getOffset(), elementByteSize(src.getType()), total, ivs);
        return desc;
    }
    if (auto vstsx2 = dyn_cast<pto::Vstsx2Op>(op)) {
        desc.kind = VecScopeAccessKind::Store;
        Value dst = vstsx2.getDestination();
        auto lo = vregElementCount(vstsx2.getLow().getType());
        auto hi = vregElementCount(vstsx2.getHigh().getType());
        std::optional<uint64_t> total;
        if (lo && hi) {
            total = *lo + *hi;
        }
        total = maskedStoredElementCount(vstsx2.getMask(), total);
        fillContiguous(desc, dst, vstsx2.getOffset(), elementByteSize(dst.getType()), total, ivs);
        return desc;
    }
    return std::nullopt;
}

// Constant-stride vsstb writes a union of 32-byte blocks, not the enclosing
// interval: gaps must remain available to independent accesses. Unknown masks
// retain all blocks. Partial predicates deliberately do not shrink the range
// until their block-selection semantics are proven.
static std::optional<VecMemoryAccessDescriptor> describeBlockStore(Operation* op, ArrayRef<Value> ivs)
{
    auto store = dyn_cast<pto::VsstbOp>(op);
    if (!store) {
        return std::nullopt;
    }
    APInt stride;
    if (!matchPattern(store.getBlockStride(), m_ConstantInt(&stride))) {
        return std::nullopt;
    }
    VecMemoryAccessDescriptor desc;
    desc.kind = VecScopeAccessKind::Store;
    desc.base = store.getDestination();
    desc.addressSpace = getTypeAddressSpace(desc.base.getType());
    // In post-update form, repeat_stride advances updated_base after this
    // access; this write's blocks start at destination. resolveRoot models
    // the returned pointer separately. Only the non-post-update form adds
    // repeat_stride to this access's byte offset.
    //
    // Handle both forms independently of pipeline ordering: post-update IR
    // may already be present when this pass runs, including direct pass tests.
    if (!store.getUpdatedBase()) {
        desc.byteOffset = extractAffineElement(store.getRepeatStride(), ivs);
        if (!desc.byteOffset.exact) {
            return std::nullopt;
        }
        // The control field is unsigned, including i16 constants above 32767.
        APInt repeat;
        if (matchPattern(store.getRepeatStride(), m_ConstantInt(&repeat))) {
            desc.byteOffset.constant = repeat.getZExtValue();
        } else {
            // Without an unsigned range proof, integer wrapping is not affine.
            return std::nullopt;
        }
        desc.byteOffset.scale(32);
    }
    desc.conservativeByteSize = 32;
    for (unsigned block = 0; block < 8; ++block) {
        desc.blockByteOffsets.push_back(static_cast<int64_t>(block * stride.getZExtValue() * 32));
    }
    return desc;
}

// Describe the accesses whose footprint cannot be modelled exactly and must
// stay conservatively MayAlias: strided/post-update stores (vsstb), gathers,
// and scatters. Returns nullopt when `op` is none of them.
static std::optional<VecMemoryAccessDescriptor> describeForcedMayAliasAccess(Operation* op)
{
    VecMemoryAccessDescriptor desc;
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
        if (auto g = dyn_cast<pto::Vgather2Op>(op)) {
            src = g.getSource();
        } else if (auto g = dyn_cast<pto::VgatherbOp>(op)) {
            src = g.getSource();
        } else if (auto g = dyn_cast<pto::Vgather2BcOp>(op)) {
            src = g.getSource();
        }
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
    return std::nullopt;
}

// Keep newly added or stateful UB vector-memory operations analyzable even
// before they gain an exact footprint model. The analysis discovers these
// operations through MemoryEffectOpInterface, so rejecting an operation
// here would turn a conservative optimization limitation into a compiler
// failure for otherwise valid VPTO.
static void describeByMemoryEffects(Operation* op, VecMemoryAccessDescriptor& desc)
{
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
    cast<MemoryEffectOpInterface>(op).getEffects(effects);
    for (const auto& effect : effects) {
        Value value = effect.getValue();
        if (!value || getTypeAddressSpace(value.getType()) != AddressSpace::VEC) {
            continue;
        }
        desc.base = value;
        desc.addressSpace = AddressSpace::VEC;
        desc.kind =
            isa<MemoryEffects::Write>(effect.getEffect()) ? VecScopeAccessKind::Store : VecScopeAccessKind::Load;
        break;
    }

    // These stateful stores currently describe their UB base as a read effect
    // because the pointer itself participates in the state update. Classify the
    // actual UB access by operation semantics until their effects are split into
    // pointer-state reads and memory writes.
    if (isa<pto::PstuOp, pto::VstusOp, pto::VsturOp>(op)) {
        desc.kind = VecScopeAccessKind::Store;
    }

    desc.forcesMayAlias = true;
}

} // namespace

FailureOr<VecMemoryAccessDescriptor> vecscopemembar::buildAccessDescriptor(Operation* op, ArrayRef<Value> ivs)
{
    if (auto predicate = describePredicateAccess(op, ivs)) {
        return *predicate;
    }
    if (auto blocks = describeBlockStore(op, ivs)) {
        return *blocks;
    }
    if (auto contiguous = describeContiguousAccess(op, ivs)) {
        return *contiguous;
    }
    if (auto paired = describePairedContiguousAccess(op, ivs)) {
        return *paired;
    }
    if (auto forced = describeForcedMayAliasAccess(op)) {
        return *forced;
    }

    VecMemoryAccessDescriptor desc;
    describeByMemoryEffects(op, desc);
    return desc;
}

// Reconstruct the full footprint (with resolved root) for an access. The
// descriptor stores the base + byte offset + size; here we resolve the root
// provenance from `base` so two accesses can be compared at root equality.
VecScopeMemoryFootprint vecscopemembar::footprintFromDescriptor(
    const VecMemoryAccessDescriptor& desc, ArrayRef<Value> ivs)
{
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
        fp.rootKind = resolveRoot(
            desc.base, fp.byteOffset, fp.root, fp.absoluteBase, ivs,
            true);
        return fp;
    }
    fp.rootKind = resolveRoot(desc.base, fp.byteOffset, fp.root, fp.absoluteBase, ivs);
    return fp;
}

using WideInt = __int128_t;

static bool intervalsOverlap(WideInt aLo, uint64_t aSize, WideInt bLo, uint64_t bSize)
{
    // Use a signed type wider than every input so adding an offset or a size
    // cannot wrap at 64 bits.
    WideInt aEnd = aLo + static_cast<WideInt>(aSize);
    WideInt bEnd = bLo + static_cast<WideInt>(bSize);
    return aLo < bEnd && bLo < aEnd;
}

namespace {

// The absolute start address of a footprint with a resolved absolute base and
// constant byte offset.
static WideInt absoluteStart(const VecScopeMemoryFootprint& fp)
{
    return static_cast<WideInt>(*fp.absoluteBase) + static_cast<WideInt>(fp.byteOffset.constant);
}

// Both absolute with closed ranges: compare intervals directly.
static std::optional<VecScopeAliasResult> aliasAbsoluteClosed(
    const VecScopeMemoryFootprint& producer, const VecScopeMemoryFootprint& consumer)
{
    if (producer.absoluteBase && consumer.absoluteBase && producer.byteSize && consumer.byteSize &&
        producer.byteOffset.exact && consumer.byteOffset.exact && producer.byteOffset.isConstant() &&
        consumer.byteOffset.isConstant()) {
        WideInt producerStart = absoluteStart(producer);
        WideInt consumerStart = absoluteStart(consumer);
        if (producerStart >= 0 && consumerStart >= 0 &&
            !intervalsOverlap(producerStart, *producer.byteSize, consumerStart, *consumer.byteSize)) {
            return VecScopeAliasResult::NoAlias;
        }
    }
    return std::nullopt;
}

// One side open-ended (byteSize unknown, e.g. vsstb post-update): the open
// side's base B is the lower bound of its write range. The other side's
// closed interval [a, a+s) is disjoint iff it ends at or before B (open side
// is consumer) or starts at/after B+s (open side is producer). Only the
// "ends before B" case is provable since the open upper bound is unknown.
static std::optional<VecScopeAliasResult> aliasAbsoluteOpenEnded(
    const VecScopeMemoryFootprint& producer, const VecScopeMemoryFootprint& consumer)
{
    if (!producer.absoluteBase || !consumer.absoluteBase || !producer.byteOffset.exact || !consumer.byteOffset.exact ||
        !producer.byteOffset.isConstant() || !consumer.byteOffset.isConstant()) {
        return std::nullopt;
    }
    WideInt prodStart = absoluteStart(producer);
    WideInt consStart = absoluteStart(consumer);
    if (prodStart < 0 || consStart < 0) {
        return std::nullopt;
    }
    // Producer open-ended (e.g. vsstb): consumer [consStart, consStart+s)
    // ends at or before prodStart.
    if (producer.forcesMayAlias && !producer.byteSize && consumer.byteSize &&
        consStart + static_cast<WideInt>(*consumer.byteSize) <= prodStart) {
        return VecScopeAliasResult::NoAlias;
    }
    // Consumer open-ended: producer [prodStart, prodStart+s) ends at or
    // before consStart.
    if (consumer.forcesMayAlias && !consumer.byteSize && producer.byteSize &&
        prodStart + static_cast<WideInt>(*producer.byteSize) <= consStart) {
        return VecScopeAliasResult::NoAlias;
    }
    return std::nullopt;
}

// Rule 2: gather/scatter forces MayAlias within same space, but only after
// trying to prove the (possibly open-ended) footprints disjoint at the root
// level. Disjoint absolute ranges and distinct proven allocations remain
// provable NoAlias; everything else stays MayAlias.
static VecScopeAliasResult aliasForcedMayAlias(
    const VecScopeMemoryFootprint& producer, const VecScopeMemoryFootprint& consumer)
{
    if (producer.rootKind == MemoryRootKind::Absolute && consumer.rootKind == MemoryRootKind::Absolute) {
        if (auto closed = aliasAbsoluteClosed(producer, consumer)) {
            return *closed;
        }
        if (auto open = aliasAbsoluteOpenEnded(producer, consumer)) {
            return *open;
        }
    }
    // Distinct proven allocations cannot alias even with forcesMayAlias.
    if (producer.rootKind == MemoryRootKind::ProvenAllocation &&
        consumer.rootKind == MemoryRootKind::ProvenAllocation && producer.root != consumer.root) {
        return VecScopeAliasResult::NoAlias;
    }
    return VecScopeAliasResult::MayAlias;
}

// Rule 3: both absolute -> compare absolute intervals.
static VecScopeAliasResult aliasBothAbsolute(
    const VecScopeMemoryFootprint& producer, const VecScopeMemoryFootprint& consumer)
{
    if (!producer.absoluteBase || !consumer.absoluteBase || !producer.byteSize || !consumer.byteSize ||
        !producer.byteOffset.exact || !consumer.byteOffset.exact || !producer.byteOffset.isConstant() ||
        !consumer.byteOffset.isConstant()) {
        return VecScopeAliasResult::MayAlias;
    }
    WideInt producerStart = absoluteStart(producer);
    WideInt consumerStart = absoluteStart(consumer);
    if (producerStart < 0 || consumerStart < 0) {
        return VecScopeAliasResult::MayAlias;
    }
    if (intervalsOverlap(producerStart, *producer.byteSize, consumerStart, *consumer.byteSize)) {
        return VecScopeAliasResult::MustOrPartialAlias;
    }
    return VecScopeAliasResult::NoAlias;
}

// Rule 5, same root -> compare the root-relative intervals.
static VecScopeAliasResult aliasSameRootRelative(
    const VecScopeMemoryFootprint& producer, const VecScopeMemoryFootprint& consumer)
{
    if (!producer.byteSize || !consumer.byteSize || !producer.byteOffset.exact || !consumer.byteOffset.exact) {
        return VecScopeAliasResult::MayAlias;
    }

    // Equal symbolic terms cancel for same-iteration comparisons only. Use
    // wide subtraction so extreme constant offsets cannot wrap into NoAlias.
    bool equalTerms = true;
    for (auto [value, coefficient] : producer.byteOffset.coefficients) {
        // This helper is also used for sibling-loop comparisons. Their IVs
        // describe independent executions and must stay in the relation model.
        auto arg = dyn_cast<BlockArgument>(value);
        bool loopArgument = arg && isa<scf::ForOp>(arg.getOwner()->getParentOp());
        equalTerms &= !loopArgument && consumer.byteOffset.getCoeff(value) == coefficient;
    }
    for (auto [value, coefficient] : consumer.byteOffset.coefficients) {
        equalTerms &= producer.byteOffset.getCoeff(value) == coefficient;
    }
    if (equalTerms) {
        WideInt pLo = static_cast<WideInt>(producer.byteOffset.constant);
        WideInt cLo = static_cast<WideInt>(consumer.byteOffset.constant);
        if (intervalsOverlap(pLo, *producer.byteSize, cLo, *consumer.byteSize)) {
            return VecScopeAliasResult::MustOrPartialAlias;
        }
        return VecScopeAliasResult::NoAlias;
    }

    // IV/symbolic offset terms: first version cannot prove disjointness across
    // the iteration domain -> MayAlias. Loop-carried Presburger reasoning lives
    // in the analysis module.
    return VecScopeAliasResult::MayAlias;
}

} // namespace

VecScopeAliasResult vecscopemembar::aliasSameIteration(
    const VecScopeMemoryFootprint& producer, const VecScopeMemoryFootprint& consumer)
{
    // Rule 1: different memory spaces -> NoAlias.
    if (producer.addressSpace && consumer.addressSpace && *producer.addressSpace != *consumer.addressSpace) {
        return VecScopeAliasResult::NoAlias;
    }

    // Rule 2: gather/scatter forces MayAlias within same space, but only after
    // trying to prove the (possibly open-ended) footprints disjoint at the root
    // level. This lets a post-update vsstb whose destination base resolves to
    // an absolute address be proven NoAlias against other absolute buffers that
    // live entirely below its write-range lower bound, and against distinct
    // proven allocations.
    if (producer.forcesMayAlias || consumer.forcesMayAlias) {
        return aliasForcedMayAlias(producer, consumer);
    }

    // Rule 3: both absolute -> compare absolute intervals.
    if (producer.rootKind == MemoryRootKind::Absolute && consumer.rootKind == MemoryRootKind::Absolute) {
        return aliasBothAbsolute(producer, consumer);
    }

    // Absolute vs symbolic/unknown: cannot prove disjoint -> MayAlias.
    if (producer.rootKind == MemoryRootKind::Absolute || consumer.rootKind == MemoryRootKind::Absolute) {
        return VecScopeAliasResult::MayAlias;
    }

    // Rule 7: unknown provenance -> MayAlias.
    if (producer.rootKind == MemoryRootKind::Unknown || consumer.rootKind == MemoryRootKind::Unknown) {
        return VecScopeAliasResult::MayAlias;
    }

    // Rule 6: different symbolic roots -> MayAlias.
    if (producer.rootKind == MemoryRootKind::ProvenAllocation &&
        consumer.rootKind == MemoryRootKind::ProvenAllocation && producer.root != consumer.root) {
        return VecScopeAliasResult::NoAlias;
    }

    // Generic symbolic roots (for example function arguments) may alias even
    // when represented by different SSA values.
    if (producer.root != consumer.root) {
        return VecScopeAliasResult::MayAlias;
    }

    // Rule 5: same root -> compare relative intervals.
    return aliasSameRootRelative(producer, consumer);
}
