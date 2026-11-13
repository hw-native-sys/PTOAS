// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Analysis/PTOMemoryAccessAnalysis.h"
#include "PTO/Analysis/PTOIntegerRange.h"
#include "PTOMemoryIndexRange.h"

#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VPTOAddressSemantics.h"
#include "PTO/IR/VPTOMemoryDist.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"

#include <algorithm>
#include <cstdint>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::integer_range;

using mlir::pto::integer_range::allIntegerLanes;
using mlir::pto::integer_range::constantIndexLanes;

namespace {
struct ConstantOperand {
    Value value;
    int64_t constant;
};

static std::optional<ConstantOperand> matchConstantOperand(Operation* binary)
{
    for (unsigned side = 0; side < 2; ++side) {
        auto constant = integer_range::getConstantInt(binary->getOperand(side));
        if (constant) {
            return ConstantOperand{binary->getOperand(1 - side), *constant};
        }
    }
    return std::nullopt;
}

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

// An empty expression marked inexact and lossy: information was lost, so the
// caller must conservatively retain the hazard instead of proving a false
// NoAlias, and symbolic-term proofs must not treat it as a bounded offset.
static PTOAffineByteExpr makeInexactExpr()
{
    PTOAffineByteExpr expr;
    expr.exact = false;
    expr.lossy = true;
    return expr;
}

// Scale `expr` by `factor` while keeping the constant and every coefficient representable as int64.
static bool scaleChecked(PTOAffineByteExpr& expr, int64_t factor)
{
    if (factor == 1) {
        return true;
    }
    __int128 constant = static_cast<__int128>(expr.constant) * factor;
    if (constant < INT64_MIN || constant > INT64_MAX) {
        return false;
    }
    for (auto& [value, coefficient] : expr.coefficients) {
        __int128 scaled = static_cast<__int128>(coefficient) * factor;
        if (scaled < INT64_MIN || scaled > INT64_MAX) {
            return false;
        }
        coefficient = static_cast<int64_t>(scaled);
    }
    expr.constant = static_cast<int64_t>(constant);
    return true;
}

static PTOAffineByteExpr extractAffineElement(Value value, ArrayRef<Value> ivs);
static PTOAffineByteExpr extractAffineExpr(
    mlir::AffineExpr expr, ValueRange dims, ValueRange symbols, ArrayRef<Value> ivs);

static PTOAffineByteExpr combineAffineOperands(mlir::AffineExprKind kind, PTOAffineByteExpr lhs, PTOAffineByteExpr rhs)
{
    switch (kind) {
        case mlir::AffineExprKind::Add:
            return lhs.combine(rhs);
        case mlir::AffineExprKind::Mul:
            if (rhs.isConstant()) {
                if (!scaleChecked(lhs, rhs.constant)) {
                    return makeInexactExpr();
                }
                return lhs;
            }
            if (lhs.isConstant()) {
                if (!scaleChecked(rhs, lhs.constant)) {
                    return makeInexactExpr();
                }
                return rhs;
            }
            return makeInexactExpr();
        default:
            // FloorDiv/CeilDiv/Mod need a bounded integer-domain model.  Keeping them
            // lossy is required for correctness: the caller will conservatively
            // retain a hazard instead of proving a false NoAlias.
            return makeInexactExpr();
    }
}

static PTOAffineByteExpr extractAffineBinaryExpr(
    mlir::AffineBinaryOpExpr binary, ValueRange dims, ValueRange symbols, ArrayRef<Value> ivs)
{
    auto lhs = extractAffineExpr(binary.getLHS(), dims, symbols, ivs);
    auto rhs = extractAffineExpr(binary.getRHS(), dims, symbols, ivs);
    if (lhs.lossy || rhs.lossy) {
        return makeInexactExpr();
    }
    return combineAffineOperands(binary.getKind(), lhs, rhs);
}

static PTOAffineByteExpr extractAffineExpr(
    mlir::AffineExpr expr, ValueRange dims, ValueRange symbols, ArrayRef<Value> ivs)
{
    PTOAffineByteExpr result;
    if (auto constant = dyn_cast<mlir::AffineConstantExpr>(expr)) {
        result.constant = constant.getValue();
        return result;
    }
    if (auto dim = dyn_cast<mlir::AffineDimExpr>(expr)) {
        bool invalidDimension = dim.getPosition() >= dims.size();
        if (invalidDimension) {
            result.exact = false;
            result.lossy = true;
            return result;
        }
        return extractAffineElement(dims[dim.getPosition()], ivs);
    }
    if (auto symbol = dyn_cast<mlir::AffineSymbolExpr>(expr)) {
        bool invalidSymbol = symbol.getPosition() >= symbols.size();
        if (invalidSymbol) {
            result.exact = false;
            result.lossy = true;
            return result;
        }
        return extractAffineElement(symbols[symbol.getPosition()], ivs);
    }
    auto binary = dyn_cast<mlir::AffineBinaryOpExpr>(expr);
    return binary ? extractAffineBinaryExpr(binary, dims, symbols, ivs) : makeInexactExpr();
}

// Recognize the arithmetic op shapes with an affine form: add/sub and mul by
// a constant.
static std::optional<PTOAffineByteExpr> extractArithAffineElement(Value value, ArrayRef<Value> ivs)
{
    if (auto addIOp = value.getDefiningOp<arith::AddIOp>()) {
        // The sign-corrected remainder sum is one correlated leaf: splitting
        // it into independent symbolic terms loses the correlation between
        // the remainder and its sign correction.
        if (matchSignCorrectedRemSum(value)) {
            PTOAffineByteExpr expr;
            expr.coefficients.push_back({value, 1});
            expr.exact = false;
            return expr;
        }
        auto lhs = extractAffineElement(addIOp.getLhs(), ivs);
        auto rhs = extractAffineElement(addIOp.getRhs(), ivs);
        if (lhs.lossy || rhs.lossy) {
            return makeInexactExpr();
        }
        return lhs.combine(rhs);
    }
    if (auto subIOp = value.getDefiningOp<arith::SubIOp>()) {
        auto lhs = extractAffineElement(subIOp.getLhs(), ivs);
        auto rhs = extractAffineElement(subIOp.getRhs(), ivs);
        if (lhs.lossy || rhs.lossy) {
            return makeInexactExpr();
        }
        if (!scaleChecked(rhs, -1)) {
            return makeInexactExpr();
        }
        return lhs.combine(rhs);
    }
    if (auto mulIOp = value.getDefiningOp<arith::MulIOp>()) {
        auto scaled = matchConstantOperand(mulIOp);
        if (!scaled) {
            return makeInexactExpr();
        }
        auto expr = extractAffineElement(scaled->value, ivs);
        if (expr.lossy || !scaleChecked(expr, scaled->constant)) {
            return makeInexactExpr();
        }
        return expr;
    }
    return std::nullopt;
}

// Recognize cast candidates. The caller forwards their affine input only
// after certifying that the cast preserves values over the loop envelopes.
static std::optional<PTOAffineByteExpr> extractCastAffineElement(Value value, ArrayRef<Value> ivs)
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
static std::optional<PTOAffineByteExpr> extractAffineApplyElement(Value value, ArrayRef<Value> ivs)
{
    auto apply = value.getDefiningOp<affine::AffineApplyOp>();
    if (!apply) {
        return std::nullopt;
    }
    AffineMap map = apply.getAffineMap();
    bool multipleResults = map.getNumResults() != 1;
    if (multipleResults) {
        return makeInexactExpr();
    }
    ValueRange operands = apply.getMapOperands();
    unsigned numDims = map.getNumDims();
    bool invalidOperands = operands.size() != numDims + map.getNumSymbols();
    if (invalidOperands) {
        return makeInexactExpr();
    }
    return extractAffineExpr(map.getResult(0), operands.take_front(numDims), operands.drop_front(numDims), ivs);
}

static bool castPreservesValue(Value value, ArrayRef<Value> ivs)
{
    Operation* cast = value.getDefiningOp();
    auto function = cast->getParentOfType<func::FuncOp>();
    if (!function) {
        return false;
    }
    PTOValueEvolutionAnalysis evolution(function);
    auto preserved = evolution.preservesValueAt(cast, cast);
    if (!preserved || !*preserved.value) {
        return false;
    }
    // Cast preservation alone does not authorize flattening its input chain.
    SmallVector<integer_range::PTOLoopDomain> bounds;
    for (Value iv : ivs) {
        auto argument = dyn_cast<BlockArgument>(iv);
        auto loop = argument ? dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp()) : scf::ForOp();
        bounds.push_back(loop ? integer_range::getLoopEnvelope(loop) : integer_range::PTOLoopDomain{});
    }
    return integer_range::getAddressRange(value, ivs, bounds).has_value();
}

static PTOAffineByteExpr extractAffineElement(Value value, ArrayRef<Value> ivs)
{
    PTOAffineByteExpr expr;

    APInt c;
    if (matchPattern(value, m_ConstantInt(&c))) {
        if (!c.isSignedIntN(64)) {
            return makeInexactExpr();
        }
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
        if (!castPreservesValue(value, ivs)) {
            PTOAffineByteExpr opaque;
            opaque.exact = false;
            opaque.coefficients.push_back({value, 1});
            return opaque;
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

static PTOAffineByteExpr extractSubviewElementOffset(memref::SubViewOp subview, ArrayRef<Value> ivs)
{
    PTOAffineByteExpr result;
    auto sourceType = dyn_cast<MemRefType>(subview.getSource().getType());
    if (!sourceType) {
        return makeInexactExpr();
    }

    SmallVector<int64_t> sourceStrides;
    int64_t ignoredSourceOffset = ShapedType::kDynamic;
    if (failed(mlir::pto::getPTOMemRefStridesAndOffset(sourceType, sourceStrides, ignoredSourceOffset))) {
        return makeInexactExpr();
    }

    auto mixedOffsets = subview.getMixedOffsets();
    bool invalidStrides = mixedOffsets.size() > sourceStrides.size();
    if (invalidStrides) {
        return makeInexactExpr();
    }

    for (auto [mixedOffset, stride] : llvm::zip(mixedOffsets, ArrayRef<int64_t>(sourceStrides))) {
        if (stride == ShapedType::kDynamic) {
            return makeInexactExpr();
        }

        PTOAffineByteExpr offset;
        if (auto attr = mixedOffset.dyn_cast<Attribute>()) {
            auto integer = dyn_cast<IntegerAttr>(attr);
            if (!integer || !integer.getValue().isSignedIntN(64)) {
                return makeInexactExpr();
            }
            offset.constant = integer.getInt();
        } else if (auto value = mixedOffset.dyn_cast<Value>()) {
            offset = extractAffineElement(value, ivs);
        } else {
            return makeInexactExpr();
        }

        if (!offset.exact) {
            return makeInexactExpr();
        }
        offset.scale(stride);
        result.combine(offset);
    }
    return result;
}

static PTOMemoryRootKind resolveRoot(
    Value ptr, PTOAffineByteExpr& byteOffset, Value& rootOut, std::optional<uint64_t>& absoluteBase,
    ArrayRef<Value> ivs, bool allowIterArgInit = false);

// pto.cast_ptr from an integer/index value: an absolute address when the
// integer is constant, else an unknown root.
static std::optional<PTOMemoryRootKind> tryResolveCastPtr(
    Value ptr, PTOAffineByteExpr& byteOffset, Value& rootOut, std::optional<uint64_t>& absoluteBase,
    ArrayRef<Value> ivs, bool allowIterArgInit)
{
    auto castOp = ptr.getDefiningOp<pto::CastPtrOp>();
    if (!castOp) {
        return std::nullopt;
    }
    Value input = castOp.getInput();
    bool integerInput = isa<IntegerType>(input.getType()) || input.getType().isIndex();
    if (integerInput) {
        APInt v;
        if (matchPattern(input, m_ConstantInt(&v))) {
            bool oversizedAddress = v.getActiveBits() > 64;
            if (oversizedAddress) {
                rootOut = ptr;
                return PTOMemoryRootKind::Unknown;
            }
            absoluteBase = uint64_t(v.getZExtValue());
            rootOut = ptr;
            return PTOMemoryRootKind::Absolute;
        }
        rootOut = ptr;
        return PTOMemoryRootKind::Unknown;
    }
    return resolveRoot(input, byteOffset, rootOut, absoluteBase, ivs, allowIterArgInit);
}

// pto.add_ptr: fold the affine element offset scaled by the element byte
// size into the byte offset, then keep resolving the base pointer.
static std::optional<PTOMemoryRootKind> tryResolveAddPtr(
    Value ptr, PTOAffineByteExpr& byteOffset, Value& rootOut, std::optional<uint64_t>& absoluteBase,
    ArrayRef<Value> ivs, bool allowIterArgInit)
{
    auto addOp = ptr.getDefiningOp<pto::AddPtrOp>();
    if (!addOp) {
        return std::nullopt;
    }
    auto elemOffset = extractAffineElement(addOp.getOffset(), ivs);
    std::optional<uint64_t> bytes = elementByteSize(addOp.getType());
    if (!bytes || !scaleChecked(elemOffset, int64_t(*bytes))) {
        rootOut = ptr;
        return PTOMemoryRootKind::Unknown;
    }
    // An inexact-but-faithful offset (symbolic terms) only taints the byte
    // offset while the root keeps resolving; a lossy offset is dropped the
    // same way, combine marks the byte offset lossy.
    byteOffset.combine(elemOffset);
    return resolveRoot(addOp.getPtr(), byteOffset, rootOut, absoluteBase, ivs, allowIterArgInit);
}

// memref.subview: fold the subview element offset scaled by the source
// element byte size into the byte offset.
static std::optional<PTOMemoryRootKind> tryResolveSubview(
    Value ptr, PTOAffineByteExpr& byteOffset, Value& rootOut, std::optional<uint64_t>& absoluteBase,
    ArrayRef<Value> ivs, bool allowIterArgInit)
{
    auto subview = ptr.getDefiningOp<memref::SubViewOp>();
    if (!subview) {
        return std::nullopt;
    }
    auto subviewOffset = extractSubviewElementOffset(subview, ivs);
    auto bytes = elementByteSize(subview.getSource().getType());
    if (!subviewOffset.exact || subviewOffset.lossy || !bytes) {
        byteOffset.exact = false;
        byteOffset.lossy = true;
    } else {
        if (!scaleChecked(subviewOffset, int64_t(*bytes))) {
            byteOffset.exact = false;
            byteOffset.lossy = true;
        } else {
            byteOffset.combine(subviewOffset);
        }
    }
    return resolveRoot(subview.getSource(), byteOffset, rootOut, absoluteBase, ivs, allowIterArgInit);
}

// Pointer-preserving memref casts: resolve through to the source. A
// reinterpret_cast also taints the byte offset (its own offset is not
// modelled).
static std::optional<PTOMemoryRootKind> tryResolveMemrefCasts(
    Value ptr, PTOAffineByteExpr& byteOffset, Value& rootOut, std::optional<uint64_t>& absoluteBase,
    ArrayRef<Value> ivs, bool allowIterArgInit)
{
    if (auto cast = ptr.getDefiningOp<memref::CastOp>()) {
        return resolveRoot(cast.getSource(), byteOffset, rootOut, absoluteBase, ivs, allowIterArgInit);
    }
    if (auto tileAddr = ptr.getDefiningOp<pto::TileBufAddrOp>()) {
        return resolveRoot(tileAddr.getSrc(), byteOffset, rootOut, absoluteBase, ivs, allowIterArgInit);
    }
    if (auto rc = ptr.getDefiningOp<memref::ReinterpretCastOp>()) {
        byteOffset.exact = false;
        byteOffset.lossy = true;
        return resolveRoot(rc.getSource(), byteOffset, rootOut, absoluteBase, ivs, allowIterArgInit);
    }
    if (auto msc = ptr.getDefiningOp<memref::MemorySpaceCastOp>()) {
        return resolveRoot(msc.getSource(), byteOffset, rootOut, absoluteBase, ivs, allowIterArgInit);
    }
    return std::nullopt;
}

// Predicate address units and post-update offsets follow the shared IR contract.
static bool isPredicateMemory(Operation* op) { return isa<pto::PldsOp, pto::PldiOp, pto::PstsOp, pto::PstiOp>(op); }

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
            bool unsupportedUpdate =
                value != store.getUpdatedBase() || !matchPattern(store.getRepeatStride(), m_ConstantInt(&amount));
            if (unsupportedUpdate) {
                return std::nullopt;
            }
            delta += static_cast<__int128>(amount.getZExtValue()) * 32;
            value = store.getDestination();
        } else if (auto update = predicatePointerAdvance(value)) {
            delta += update->second;
            value = update->first;
        } else if (auto add = value.getDefiningOp<pto::AddPtrOp>()) {
            auto bytes = elementByteSize(add.getType());
            bool invalidOffset =
                !bytes || !matchPattern(add.getOffset(), m_ConstantInt(&amount)) || amount.getBitWidth() > 64;
            if (invalidOffset) {
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
static std::optional<PTOMemoryRootKind> resolveFixedIterArgRoot(
    Value ptr, PTOAffineByteExpr& byteOffset, Value& rootOut, std::optional<uint64_t>& absoluteBase,
    ArrayRef<Value> ivs)
{
    auto arg = dyn_cast<BlockArgument>(ptr);
    auto loop = arg ? dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp()) : scf::ForOp();
    bool notEnclosingIV = !loop || arg.getArgNumber() == 0 || !llvm::is_contained(ivs, loop.getInductionVar());
    if (notEnclosingIV) {
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
    bool nonConstantRecurrence = !delta || !matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
                                 !matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) ||
                                 !matchPattern(loop.getStep(), m_ConstantInt(&step)) || lower.getBitWidth() > 64 ||
                                 upper.getBitWidth() > 64 || step.getBitWidth() > 64;
    if (nonConstantRecurrence) {
        return std::nullopt;
    }
    int64_t lb = lower.getSExtValue();
    int64_t ub = upper.getSExtValue();
    APInt stride = step.sext(128);
    if (!stride.isStrictlyPositive()) {
        return std::nullopt;
    }
    APInt advance(128, *delta, true);
    if (ub <= lb || !advance.srem(stride).isZero()) {
        return std::nullopt;
    }
    APInt count = (upper.sext(128) - lower.sext(128) + stride - 1).sdiv(stride);
    APInt extent = count * advance;
    APInt quotient = advance.sdiv(stride);
    if (!quotient.isSignedIntN(64)) {
        return std::nullopt;
    }
    int64_t coefficient = quotient.getSExtValue();
    __int128 constant = -static_cast<__int128>(lb) * coefficient + byteOffset.constant;
    bool unsupportedRecurrence = !extent.isSignedIntN(64) || constant < INT64_MIN || constant > INT64_MAX ||
                                 byteOffset.getCoeff(loop.getInductionVar()) != 0;
    if (unsupportedRecurrence) {
        return std::nullopt;
    }
    // Resolve the init separately so adding its constant cannot overflow the
    // recurrence adjustment. Nested/offset initializers retain their IV terms.
    PTOAffineByteExpr initOffset;
    auto kind = resolveRoot(loop.getInits()[index], initOffset, rootOut, absoluteBase, ivs, false);
    constant += initOffset.constant;
    if (!initOffset.exact || constant < INT64_MIN || constant > INT64_MAX) {
        return PTOMemoryRootKind::Unknown;
    }
    // combine() can retain zero coefficients after cancellation. Merge by
    // variable rather than appending duplicates that getCoeff() would hide.
    PTOAffineByteExpr recurrence;
    recurrence.coefficients.push_back({loop.getInductionVar(), coefficient});
    byteOffset.constant = static_cast<int64_t>(constant);
    byteOffset.combine(recurrence);
    for (auto [iv, coeff] : initOffset.coefficients) {
        bool duplicateIV = byteOffset.getCoeff(iv) != 0;
        if (duplicateIV) {
            return PTOMemoryRootKind::Unknown;
        }
    }
    // The initializer constant was already included in the checked sum above.
    initOffset.constant = 0;
    byteOffset.combine(initOffset);
    return byteOffset.exact ? kind : PTOMemoryRootKind::Unknown;
}

// Only an unchanged carried pointer can use its initializer as an unbounded
// recurrence's address lower bound. A positive step alone does not exclude
// pointer wraparound; bounded recurrences use resolveFixedIterArgRoot instead.
static std::optional<PTOMemoryRootKind> resolveIterArgInitRoot(
    Value ptr, PTOAffineByteExpr& byteOffset, Value& rootOut, std::optional<uint64_t>& absoluteBase,
    ArrayRef<Value> ivs)
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
            auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
            bool unchanged = yield && idx < yield.getNumOperands() && yield.getOperand(idx) == ptr;
            if (!unchanged) {
                return PTOMemoryRootKind::Unknown;
            }
            Value init = forOp.getInits()[idx];
            return resolveRoot(init, byteOffset, rootOut, absoluteBase, ivs, true);
        }
    }
    return std::nullopt;
}

static PTOMemoryRootKind resolveRoot(
    Value ptr, PTOAffineByteExpr& byteOffset, Value& rootOut, std::optional<uint64_t>& absoluteBase,
    ArrayRef<Value> ivs, bool allowIterArgInit)
{
    if (auto store = ptr.getDefiningOp<pto::VsstbOp>()) {
        APInt repeat;
        bool fixedUpdate =
            ptr == store.getUpdatedBase() && matchPattern(store.getRepeatStride(), m_ConstantInt(&repeat)) &&
            repeat.getActiveBits() <= 58;
        if (fixedUpdate) {
            PTOAffineByteExpr advance;
            advance.constant = static_cast<int64_t>(repeat.getZExtValue() * 32);
            byteOffset.combine(advance);
            return resolveRoot(store.getDestination(), byteOffset, rootOut, absoluteBase, ivs, allowIterArgInit);
        }
    }
    if (auto update = predicatePointerAdvance(ptr)) {
        PTOAffineByteExpr advance;
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
        return PTOMemoryRootKind::ProvenAllocation;
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
    return PTOMemoryRootKind::Symbolic;
}

} // namespace

int64_t PTOAffineByteExpr::getCoeff(Value v) const
{
    for (auto [val, coeff] : coefficients) {
        if (val == v) {
            return coeff;
        }
    }
    return 0;
}

PTOAffineByteExpr& PTOAffineByteExpr::scale(int64_t factor)
{
    __int128 scaled = static_cast<__int128>(constant) * factor;
    if (scaled < INT64_MIN || scaled > INT64_MAX) {
        exact = false;
        lossy = true;
        return *this;
    }
    constant = static_cast<int64_t>(scaled);
    for (auto& [val, coeff] : coefficients) {
        scaled = static_cast<__int128>(coeff) * factor;
        if (scaled < INT64_MIN || scaled > INT64_MAX) {
            exact = false;
            lossy = true;
            return *this;
        }
        coeff = static_cast<int64_t>(scaled);
    }
    return *this;
}

PTOAffineByteExpr& PTOAffineByteExpr::combine(const PTOAffineByteExpr& other)
{
    exact &= other.exact;
    lossy = lossy || other.lossy;
    __int128 sum = static_cast<__int128>(constant) + other.constant;
    if (sum < INT64_MIN || sum > INT64_MAX) {
        exact = false;
        lossy = true;
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
                    lossy = true;
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
// The offset operand is recorded so the analysis can verify the whole address
// computation (base pointer chain + offset) for wrapping.
static void fillContiguous(
    PTOMemoryAccessDescriptor& desc, Value base, Value offset, std::optional<uint64_t> bytes,
    std::optional<uint64_t> elemCount, ArrayRef<Value> ivs)
{
    desc.base = base;
    desc.accessOffset = offset;
    desc.addressSpace = getTypeAddressSpace(base.getType());
    desc.byteOffset = extractAffineElement(offset, ivs);
    // Byte units for exact and inexact offsets alike: the inexact form keeps its symbolic terms, and
    // consumers that reject inexact expressions do so through the flag, never through a unit mismatch.
    if (bytes) {
        if (!scaleChecked(desc.byteOffset, int64_t(*bytes))) {
            desc.byteOffset = makeInexactExpr();
        }
    } else {
        desc.byteOffset.exact = false;
        desc.byteOffset.lossy = true;
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

static Value stripIntegerSignCast(Value value)
{
    auto cast = value.getDefiningOp<UnrealizedConversionCastOp>();
    auto type = dyn_cast<IntegerType>(value.getType());
    bool oneIntegerInput = cast && cast.getInputs().size() == 1 && type;
    if (oneIntegerInput) {
        auto source = dyn_cast<IntegerType>(cast.getInputs()[0].getType());
        bool sameWidth = source && source.getWidth() == type.getWidth();
        if (sameWidth) {
            return cast.getInputs()[0];
        }
    }
    return value;
}

} // namespace

// A constant arithmetic result is useful only when every requested lane is
// defined. Keep this certificate to explicit prefix predicates.

bool mlir::pto::integer_range::allIntegerLanes(Value mask, uint64_t lanes, unsigned width)
{
    StringRef pattern;
    if (auto op = mask.getDefiningOp<pto::PsetB16Op>()) {
        pattern = width == 0 || width == 16 ? op.getPattern() : StringRef();
    } else if (auto op = mask.getDefiningOp<pto::PsetB32Op>()) {
        pattern = width == 0 || width == 32 ? op.getPattern() : StringRef();
    }
    auto count = parseMaskPatternLaneCount(pattern, lanes);
    return lanes && count && *count == lanes;
}

namespace {

// Evaluate only fully defined, constant integer lanes. UB loads require a
// dominating full-table initializer and a function-wide no-clobber proof.
static std::optional<SmallVector<uint64_t, 128>> constantTableLanes(pto::VldsOp load, unsigned& budget);

static std::optional<uint64_t> constantBinaryIndex(Operation* op, uint64_t lhs, uint64_t rhs, IntegerType elem)
{
    if (isa<pto::VandOp>(op)) {
        return lhs & rhs;
    }
    if (isa<pto::VshrOp>(op)) {
        uint64_t signedMax = APInt::getSignedMaxValue(elem.getWidth()).getZExtValue();
        bool unsafeShift = rhs >= elem.getWidth() || (!elem.isUnsigned() && lhs > signedMax);
        if (unsafeShift) {
            return std::nullopt;
        }
        return lhs >> rhs;
    }
    uint64_t result = isa<pto::VmulOp>(op) ? lhs * rhs : lhs + rhs;
    bool overflow = result > APInt::getMaxValue(elem.getWidth()).getZExtValue();
    if (overflow) {
        return std::nullopt;
    }
    return result;
}

} // namespace

std::optional<SmallVector<uint64_t, 128>> mlir::pto::integer_range::constantIndexLanes(
    Value value, unsigned& budget, bool allowTable)
{
    auto type = dyn_cast<pto::VRegType>(value.getType());
    auto elem = type ? dyn_cast<IntegerType>(type.getElementType()) : IntegerType();
    bool unsupportedInteger = !budget || !elem || (elem.getWidth() != 16 && elem.getWidth() != 32);
    if (unsupportedInteger) {
        return std::nullopt;
    }
    --budget;
    uint64_t lanes = type.getElementCount();
    uint64_t limit = APInt::getMaxValue(elem.getWidth()).getZExtValue();
    auto scalar = [&](Value input) -> std::optional<uint64_t> {
        APInt constant;
        input = stripIntegerSignCast(input);
        bool invalidScalar =
            !matchPattern(input, m_ConstantInt(&constant)) || constant.getBitWidth() != elem.getWidth();
        if (invalidScalar) {
            return std::nullopt;
        }
        return constant.getZExtValue();
    };
    if (auto load = value.getDefiningOp<pto::VldsOp>()) {
        return allowTable ? constantTableLanes(load, budget) : std::nullopt;
    }
    if (auto cast = value.getDefiningOp<pto::VbitcastOp>()) {
        auto source = dyn_cast<pto::VRegType>(cast.getInput().getType());
        auto sourceElem = source ? dyn_cast<IntegerType>(source.getElementType()) : IntegerType();
        bool sameGeometry = sourceElem && sourceElem.getWidth() == elem.getWidth() &&
                            static_cast<uint64_t>(source.getElementCount()) == lanes;
        if (!sameGeometry) {
            return std::nullopt;
        }
        return constantIndexLanes(cast.getInput(), budget, allowTable);
    }
    if (auto vci = value.getDefiningOp<pto::VciOp>()) {
        auto start = scalar(vci.getIndex());
        bool invalidRamp = !start || !lanes || lanes > 256 || (vci.getOrder() && *vci.getOrder() != "ASC") ||
                           *start > limit - (lanes - 1);
        if (invalidRamp) {
            return std::nullopt;
        }
        SmallVector<uint64_t, 128> result;
        for (uint64_t lane = 0; lane < lanes; ++lane) {
            result.push_back(*start + lane);
        }
        return result;
    }
    if (auto dup = value.getDefiningOp<pto::VdupOp>()) {
        auto input = scalar(dup.getInput());
        bool undefinedDuplicate = !input || dup.getPosition() || !lanes || lanes > 256 ||
                                  !allIntegerLanes(dup.getMask(), lanes, elem.getWidth());
        if (undefinedDuplicate) {
            return std::nullopt;
        }
        return SmallVector<uint64_t, 128>(lanes, *input);
    }
    Operation* op = value.getDefiningOp();
    bool supportedOp = op && isa<pto::VandOp, pto::VaddOp, pto::VmulOp, pto::VshrOp, pto::VmulsOp, pto::VshrsOp>(op);
    bool unsupportedArithmetic = !supportedOp || !allIntegerLanes(op->getOperand(2), lanes, elem.getWidth());
    if (unsupportedArithmetic) {
        return std::nullopt;
    }
    auto lhs = constantIndexLanes(op->getOperand(0), budget, allowTable);
    bool invalidLhs = !lhs || lhs->size() != lanes;
    if (invalidLhs) {
        return std::nullopt;
    }
    if (isa<pto::VandOp, pto::VaddOp, pto::VmulOp, pto::VshrOp>(op)) {
        auto rhs = constantIndexLanes(op->getOperand(1), budget, allowTable);
        bool invalidRhs = !rhs || rhs->size() != lanes;
        if (invalidRhs) {
            return std::nullopt;
        }
        for (uint64_t lane = 0; lane < lanes; ++lane) {
            auto next = constantBinaryIndex(op, (*lhs)[lane], (*rhs)[lane], elem);
            if (!next) {
                return std::nullopt;
            }
            (*lhs)[lane] = *next;
        }
        return lhs;
    }
    if (auto shift = dyn_cast<pto::VshrsOp>(op)) {
        if (!elem.isUnsigned()) {
            return std::nullopt;
        }
        APInt amount;
        bool invalidShift = !matchPattern(stripIntegerSignCast(shift.getScalar()), m_ConstantInt(&amount)) ||
                            amount.getBitWidth() != 16 || amount.getZExtValue() >= elem.getWidth();
        if (invalidShift) {
            return std::nullopt;
        }
        for (uint64_t& lane : *lhs) {
            lane >>= amount.getZExtValue();
        }
        return lhs;
    }
    auto factor = scalar(op->getOperand(1));
    if (!factor) {
        return std::nullopt;
    }
    for (uint64_t& lane : *lhs) {
        uint64_t product = lane * *factor;
        if (product > limit) {
            return std::nullopt;
        }
        lane = product;
    }
    return lhs;
}

namespace {

namespace rangeProof = mlir::pto::integer_range;

// Pointer ranges here certify every addptr in its actual integer type before
// using an interval. Unknown roots and wrapping pointer chains stay opaque.
static std::optional<std::pair<uint64_t, uint64_t>> tablePointerRange(
    Value pointer, ArrayRef<Value> ivs, ArrayRef<rangeProof::PTOLoopDomain> bounds, unsigned depth = 0)
{
    if (depth == 32) {
        return std::nullopt;
    }
    if (auto cast = pointer.getDefiningOp<pto::CastPtrOp>()) {
        APInt address;
        if (matchPattern(cast.getInput(), m_ConstantInt(&address))) {
            bool wideAddress = address.getBitWidth() > 64;
            if (wideAddress) {
                return std::nullopt;
            }
            return std::make_pair(address.getZExtValue(), address.getZExtValue());
        }
        return tablePointerRange(cast.getInput(), ivs, bounds, depth + 1);
    }
    auto add = pointer.getDefiningOp<pto::AddPtrOp>();
    if (!add) {
        return std::nullopt;
    }
    auto base = tablePointerRange(add.getPtr(), ivs, bounds, depth + 1);
    auto offset = rangeProof::getAddressRange(add.getOffset(), ivs, bounds);
    auto bytes = elementByteSize(pointer.getType());
    if (!base || !offset || !bytes) {
        return std::nullopt;
    }
    __int128 lo = static_cast<__int128>(base->first) + static_cast<__int128>(offset->lo) * *bytes;
    __int128 hi = static_cast<__int128>(base->second) + static_cast<__int128>(offset->hi) * *bytes;
    if (lo < 0 || hi > UINT64_MAX) {
        return std::nullopt;
    }
    return std::make_pair(static_cast<uint64_t>(lo), static_cast<uint64_t>(hi));
}

static void tableLoopBounds(
    Operation* op, SmallVectorImpl<Value>& ivs, SmallVectorImpl<rangeProof::PTOLoopDomain>& bounds)
{
    for (Operation* parent = op->getParentOp(); parent; parent = parent->getParentOp()) {
        auto loop = dyn_cast<scf::ForOp>(parent);
        if (loop) {
            ivs.push_back(loop.getInductionVar());
            bounds.push_back(rangeProof::getLoopEnvelope(loop));
        }
    }
}

// An inclusive interval for every byte touched by a descriptor. This is used
// only for no-clobber certification, never as a replacement access operation.
static std::optional<std::pair<uint64_t, uint64_t>> tableAccessRange(
    const PTOMemoryAccessDescriptor& desc, Operation* op)
{
    if (desc.forcesMayAlias || desc.byteOffset.lossy || !desc.conservativeByteSize || !*desc.conservativeByteSize) {
        return std::nullopt;
    }
    SmallVector<Value> ivs;
    SmallVector<rangeProof::PTOLoopDomain> bounds;
    tableLoopBounds(op, ivs, bounds);
    auto base = tablePointerRange(desc.base, ivs, bounds);
    if (!base) {
        return std::nullopt;
    }
    // For contiguous operations certify the original element offset, rather
    // than trusting an affine decomposition through possibly wrapping adds.
    if (desc.accessOffset) {
        auto offset = rangeProof::getAddressRange(desc.accessOffset, ivs, bounds);
        auto bytes = elementByteSize(desc.base.getType());
        if (!offset || !bytes) {
            return std::nullopt;
        }
        __int128 lo = static_cast<__int128>(base->first) + static_cast<__int128>(offset->lo) * *bytes;
        __int128 hi = static_cast<__int128>(base->second) + static_cast<__int128>(offset->hi) * *bytes +
                      *desc.conservativeByteSize - 1;
        if (lo < 0 || hi > UINT64_MAX || !desc.blockByteOffsets.empty()) {
            return std::nullopt;
        }
        return std::make_pair(static_cast<uint64_t>(lo), static_cast<uint64_t>(hi));
    }
    __int128 lo = static_cast<__int128>(base->first) + desc.byteOffset.constant;
    __int128 hi = static_cast<__int128>(base->second) + desc.byteOffset.constant;
    // A non-contiguous descriptor is accepted only when its closed offset
    // was certified directly (constant scatter lanes or the DMA contract).
    bool unmodelledOffset = !desc.byteOffset.isConstant() || !desc.blockByteOffsets.empty();
    if (unmodelledOffset) {
        return std::nullopt;
    }
    hi += *desc.conservativeByteSize - 1;
    if (lo < 0 || hi > UINT64_MAX) {
        return std::nullopt;
    }
    return std::make_pair(static_cast<uint64_t>(lo), static_cast<uint64_t>(hi));
}

// Legacy GM->UB copies use byte strides and whole 32-byte tail blocks. For
// now accept only padding-free copies with both outer DMA loops set to one.
static std::optional<PTOMemoryAccessDescriptor> tableDmaWrite(pto::CopyGmToUbufOp copy, func::FuncOp function)
{
    bool configured = false;
    bool valid = true;
    DominanceInfo dominance(function);
    function.walk([&](pto::SetLoopSizeOutToUbOp config) {
        bool ones =
            llvm::all_of(config->getOperands(), [](Value operand) { return rangeProof::getConstantInt(operand) == 1; });
        valid &= ones;
        configured |= ones && dominance.properlyDominates(config, copy);
    });
    bool unsupportedDma = !valid || !configured || rangeProof::getConstantInt(copy.getLeftPaddingCount()) != 0 ||
                          rangeProof::getConstantInt(copy.getRightPaddingCount()) != 0;
    if (unsupportedDma) {
        return std::nullopt;
    }
    SmallVector<Value> ivs;
    SmallVector<rangeProof::PTOLoopDomain> bounds;
    tableLoopBounds(copy, ivs, bounds);
    auto count = rangeProof::getAddressRange(copy.getNBurst(), ivs, bounds);
    auto length = rangeProof::getAddressRange(copy.getLenBurst(), ivs, bounds);
    auto stride = rangeProof::getAddressRange(copy.getUbStride(), ivs, bounds);
    if (!count || !length || !stride || count->lo < 0 || count->hi > 65535 || count->hi < 1 || length->lo < 0 ||
        length->hi > 65535 || stride->lo < 0 || stride->hi > 2097151) {
        return std::nullopt;
    }
    uint64_t rounded = (static_cast<uint64_t>(length->hi) + 31) / 32 * 32;
    PTOMemoryAccessDescriptor desc;
    desc.kind = PTOMemoryAccessKind::Store;
    desc.base = copy.getDestination();
    desc.addressSpace = AddressSpace::VEC;
    desc.conservativeByteSize = static_cast<uint64_t>(count->hi - 1) * stride->hi + rounded;
    return desc;
}

// Evaluate scatter lanes separately: its accessOffset is a ramp origin,
// not the final byte offset after vector scaling.
static std::optional<PTOMemoryAccessDescriptor> tableScatterWrite(pto::VscatterOp store)
{
    Value offsets = store.getOffsets();
    auto type = cast<pto::VRegType>(offsets.getType());
    auto elem = dyn_cast<IntegerType>(type.getElementType());
    auto bytes = elementByteSize(store.getDestination().getType());
    bool supported = elem && bytes && (elem.getWidth() == 16 || elem.getWidth() == 32);
    if (!supported) {
        return std::nullopt;
    }
    uint64_t scale = 1;
    if (auto multiply = offsets.getDefiningOp<pto::VmulsOp>()) {
        APInt factor;
        bool validFactor = matchPattern(stripIntegerSignCast(multiply.getScalar()), m_ConstantInt(&factor));
        bool validScale = validFactor && elem.isUnsigned() && factor.getBitWidth() == elem.getWidth() &&
                          allIntegerLanes(multiply.getMask(), type.getElementCount(), elem.getWidth());
        if (!validScale) {
            return std::nullopt;
        }
        scale = factor.getZExtValue();
        offsets = multiply.getInput();
    }
    while (auto cast = offsets.getDefiningOp<pto::VbitcastOp>()) {
        auto source = mlir::cast<pto::VRegType>(cast.getInput().getType());
        auto sourceElement = dyn_cast<IntegerType>(source.getElementType());
        bool sameGeometry = sourceElement && source.getElementCount() == type.getElementCount() &&
                            sourceElement.getWidth() == elem.getWidth();
        if (!sameGeometry) {
            return std::nullopt;
        }
        offsets = cast.getInput();
    }
    auto vci = offsets.getDefiningOp<pto::VciOp>();
    bool ascending = vci && (!vci.getOrder() || *vci.getOrder() == "ASC");
    if (!ascending) {
        return std::nullopt;
    }
    SmallVector<Value> ivs;
    SmallVector<rangeProof::PTOLoopDomain> bounds;
    tableLoopBounds(store, ivs, bounds);
    auto start = rangeProof::getAddressRange(vci.getIndex(), ivs, bounds);
    if (!start || start->lo < 0) {
        return std::nullopt;
    }
    // Only explicit PSET prefixes at the matching granularity may shrink
    // the final ramp. PGE and dynamic masks retain every possible lane.
    StringRef pattern;
    if (auto mask = store.getMask().getDefiningOp<pto::PsetB16Op>()) {
        pattern = elem.getWidth() == 16 ? mask.getPattern() : StringRef();
    } else if (auto mask = store.getMask().getDefiningOp<pto::PsetB32Op>()) {
        pattern = elem.getWidth() == 32 ? mask.getPattern() : StringRef();
    }
    uint64_t active = parseMaskPatternLaneCount(pattern, type.getElementCount()).value_or(type.getElementCount());
    if (!active) {
        return std::nullopt;
    }
    __int128 lo = static_cast<__int128>(start->lo) * scale;
    __int128 hi = (static_cast<__int128>(start->hi) + active - 1) * scale;
    if (hi > APInt::getMaxValue(elem.getWidth()).getZExtValue()) {
        return std::nullopt;
    }
    PTOMemoryAccessDescriptor desc;
    desc.kind = PTOMemoryAccessKind::Store;
    desc.base = store.getDestination();
    desc.addressSpace = AddressSpace::VEC;
    desc.byteOffset.constant = static_cast<int64_t>(lo) * *bytes;
    desc.conservativeByteSize = static_cast<uint64_t>(hi - lo + 1) * *bytes;
    return desc;
}

// Ignore the predicate when checking possible clobbers, but retain the ISA
// memory distribution: packed and one-point stores do not write a full vreg.
static std::optional<uint64_t> tableFullStoreBytes(Operation* op)
{
    Value value;
    StringRef dist;
    VPTOMemoryOpFamily family = VPTOMemoryOpFamily::Store;
    if (auto store = dyn_cast<pto::VstsOp>(op)) {
        if (store.getUpdatedBase()) {
            return std::nullopt;
        }
        value = store.getValue();
        dist = store.getDist().value_or("");
    } else if (auto store = dyn_cast<pto::Vstsx2Op>(op)) {
        value = store.getLow();
        dist = store.getDist();
        family = VPTOMemoryOpFamily::StoreX2;
    } else if (auto store = dyn_cast<pto::PTOStoreOp>(op)) {
        return elementByteSize(store.getPtr().getType());
    } else {
        return std::nullopt;
    }
    auto type = cast<pto::VRegType>(value.getType());
    unsigned elementBytes = pto::getPTOStorageElemByteSize(type.getElementType());
    __int128 storage = static_cast<__int128>(type.getElementCount()) * elementBytes;
    if (storage <= 0 || storage > INT64_MAX) {
        return std::nullopt;
    }
    uint64_t bytes = static_cast<uint64_t>(storage);
    auto contract = lookupVPTOMemoryDist(family, dist, elementBytes * 8);
    if (!contract) {
        return std::nullopt;
    }
    auto footprint = contract->getFullActiveFootprintBytes(static_cast<int64_t>(bytes));
    if (!footprint || *footprint <= 0) {
        return std::nullopt;
    }
    return static_cast<uint64_t>(*footprint);
}

static std::optional<PTOMemoryAccessDescriptor> tableStandardWrite(Operation* op)
{
    SmallVector<Value> ivs;
    SmallVector<rangeProof::PTOLoopDomain> bounds;
    tableLoopBounds(op, ivs, bounds);
    auto desc = buildMemoryAccessDescriptor(op, ivs);
    if (failed(desc)) {
        return std::nullopt;
    }
    auto bytes = tableFullStoreBytes(op);
    if (!bytes) {
        return std::nullopt;
    }
    desc->conservativeByteSize = *bytes;
    return *desc;
}

static bool tableOtherOperationSafe(Operation* op)
{
    if (auto effects = dyn_cast<MemoryEffectOpInterface>(op)) {
        SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> instances;
        effects.getEffects(instances);
        for (const auto& effect : instances) {
            Value value = effect.getValue();
            bool unknownWrite = isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()) &&
                                (!value || isUBBackedType(value.getType()));
            if (unknownWrite) {
                return false;
            }
        }
        return true;
    }
    return isMemoryEffectFree(op) || isa<scf::YieldOp, func::ReturnOp>(op) ||
           isa<pto::SetLoopSizeOutToUbOp, pto::SetLoopSizeUbToOutOp, pto::SetFlagOp, pto::WaitFlagOp, pto::SetFlagDynOp,
               pto::WaitFlagDynOp, pto::BarrierOp, pto::GetBufOp, pto::RlsBufOp, pto::GetCtrlOp, pto::MemBarOp>(op);
}

static bool tableOperationNoClobber(
    Operation* op, func::FuncOp function, pto::VstsOp initializer, std::pair<uint64_t, uint64_t> table)
{
    bool skipOperation = op == function.getOperation() || op == initializer.getOperation();
    if (skipOperation || isa<pto::VecScopeOp, scf::ForOp, scf::IfOp>(op)) {
        return true; // Every contained operation is checked separately, including all iterations.
    }
    std::optional<PTOMemoryAccessDescriptor> write;
    if (auto copy = dyn_cast<pto::CopyGmToUbufOp>(op)) {
        write = tableDmaWrite(copy, function);
    } else if (auto scatter = dyn_cast<pto::VscatterOp>(op)) {
        write = tableScatterWrite(scatter);
    } else if (isa<pto::SetCtrlOp, pto::SetStoreAtomicCfgOp>(op)) {
        // Initial setup is allowed, but intervening mode changes invalidate propagation.
        auto scope = initializer->getParentOfType<pto::VecScopeOp>();
        return op->getBlock() == scope->getBlock() && op->isBeforeInBlock(scope);
    } else if (isa<pto::PstuOp, pto::VstusOp, pto::VsturOp>(op)) {
        return false; // These stores currently advertise only read effects.
    } else if (isPredicateMemory(op)) {
        return false; // Predicate address units require a separate certificate.
    } else if (isa<pto::VstsOp, pto::Vstsx2Op, pto::PTOStoreOp>(op)) {
        write = tableStandardWrite(op);
    } else {
        return tableOtherOperationSafe(op);
    }
    if (!write) {
        return false;
    }
    auto range = tableAccessRange(*write, op);
    return range && (range->second < table.first || range->first > table.second);
}

static bool tableNoClobber(func::FuncOp function, pto::VstsOp initializer, std::pair<uint64_t, uint64_t> table)
{
    bool safe = true;
    function.walk([&](Operation* op) {
        if (safe) {
            safe = tableOperationNoClobber(op, function, initializer, table);
        }
    });
    return safe;
}

static std::optional<SmallVector<uint64_t, 128>> constantTableLanes(pto::VldsOp load, unsigned& budget)
{
    auto function = load->getParentOfType<func::FuncOp>();
    auto loadType = cast<pto::VRegType>(load.getResult().getType());
    bool invalidLoad = !function || getTargetArch(load) != PTOArch::A5 || load.getDist() || load.getUpdatedBase() ||
                       getPointeeElementType(load.getSource().getType()) != loadType.getElementType();
    if (invalidLoad) {
        return std::nullopt;
    }
    SmallVector<Value> ivs;
    SmallVector<rangeProof::PTOLoopDomain> bounds;
    tableLoopBounds(load, ivs, bounds);
    auto desc = buildMemoryAccessDescriptor(load, ivs);
    auto table = succeeded(desc) ? tableAccessRange(*desc, load) : std::nullopt;
    if (!table || table->second - table->first != *desc->conservativeByteSize - 1) {
        return std::nullopt;
    }
    DominanceInfo dominance(function);
    std::optional<SmallVector<uint64_t, 128>> result;
    function.walk([&](pto::VstsOp store) {
        auto scope = store->getParentOfType<pto::VecScopeOp>();
        bool entryInitializer =
            scope && scope->getBlock() == &function.front() && store->getBlock() == &scope.getBody().front();
        // Across VFs, the unconditional entry-block store completes with its scope.
        // Inside that scope, its own position must precede the load.
        Operation* initializerAnchor = scope && scope->isProperAncestor(load) ? store.getOperation() : scope;
        bool invalidInitializer = result || !entryInitializer || store.getDist() || store.getUpdatedBase() ||
                                  !dominance.properlyDominates(initializerAnchor, load) ||
                                  store.getValue().getType() != load.getResult().getType();
        if (invalidInitializer) {
            return;
        }
        auto type = cast<pto::VRegType>(store.getValue().getType());
        bool sameElement = getPointeeElementType(store.getDestination().getType()) == type.getElementType();
        if (!sameElement) {
            return;
        }
        if (!allIntegerLanes(store.getMask(), type.getElementCount(), type.getElementType().getIntOrFloatBitWidth())) {
            return;
        }
        auto storeDesc = buildMemoryAccessDescriptor(store, {});
        auto stored = succeeded(storeDesc) ? tableAccessRange(*storeDesc, store) : std::nullopt;
        if (!stored || *stored != *table) {
            return;
        }
        auto lanes = constantIndexLanes(store.getValue(), budget);
        if (lanes && tableNoClobber(function, store, *table)) {
            result = std::move(lanes);
        }
    });
    return result;
}

// A bounded ascending integer VCI and a prefix mask give a closed scatter envelope.
static std::optional<PTOMemoryAccessDescriptor> describeContiguousScatter(pto::VscatterOp op)
{
    Value offsets = op.getOffsets();
    auto offsetsType = dyn_cast<pto::VRegType>(offsets.getType());
    auto offsetElem = offsetsType ? dyn_cast<IntegerType>(offsetsType.getElementType()) : IntegerType();
    bool unsupportedIndexType = !offsetElem || offsetElem.getWidth() > 32;
    if (unsupportedIndexType) {
        return std::nullopt;
    }
    uint64_t scale = 1;
    if (auto multiply = offsets.getDefiningOp<pto::VmulsOp>()) {
        Value scalar = stripIntegerSignCast(multiply.getScalar());
        auto factor = mlir::pto::integer_range::getConstantInt(scalar);
        if (!factor || *factor < 0 || !allIntegerLanes(multiply.getMask(), offsetsType.getElementCount())) {
            return std::nullopt;
        }
        scale = static_cast<uint64_t>(*factor);
        offsets = multiply.getInput();
    }
    while (auto bitcast = offsets.getDefiningOp<pto::VbitcastOp>()) {
        auto sourceType = dyn_cast<pto::VRegType>(bitcast.getInput().getType());
        auto sourceElem = sourceType ? dyn_cast<IntegerType>(sourceType.getElementType()) : IntegerType();
        bool incompatibleGeometry = !sourceElem || sourceElem.getWidth() != offsetElem.getWidth() ||
                                    sourceType.getElementCount() != offsetsType.getElementCount();
        if (incompatibleGeometry) {
            return std::nullopt;
        }
        offsets = bitcast.getInput();
    }
    auto vci = offsets.getDefiningOp<pto::VciOp>();
    bool ascending = vci && (!vci.getOrder() || *vci.getOrder() == "ASC");
    if (!ascending) {
        return std::nullopt;
    }
    auto base = mlir::pto::integer_range::inferredLeafRange(vci.getIndex());
    if (!base || base->lo < 0) {
        return std::nullopt;
    }
    // PGE patterns need not be prefixes. Only accept PSET's explicit prefix
    // patterns here rather than borrowing the generic store-size helper.
    StringRef pattern;
    if (auto mask = op.getMask().getDefiningOp<pto::PsetB16Op>()) {
        pattern = mask.getPattern();
    } else if (auto mask = op.getMask().getDefiningOp<pto::PsetB32Op>()) {
        pattern = mask.getPattern();
    }
    uint64_t requests = offsetsType.getElementCount();
    auto active = parseMaskPatternLaneCount(pattern, requests);
    auto elemBytes = elementByteSize(op.getDestination().getType());
    if (!active || !*active || !elemBytes) {
        return std::nullopt;
    }
    __int128 last = (static_cast<__int128>(base->hi) + (scale == 1 ? *active : requests) - 1) * scale;
    if (last > APInt::getMaxValue(offsetElem.getWidth()).getZExtValue()) {
        return std::nullopt;
    }
    PTOMemoryAccessDescriptor desc;
    desc.kind = PTOMemoryAccessKind::Store;
    desc.base = op.getDestination();
    desc.addressSpace = getTypeAddressSpace(desc.base.getType());
    desc.byteOffset.constant = base->lo * scale * *elemBytes;
    desc.conservativeByteSize = ((static_cast<uint64_t>(base->hi - base->lo) + *active - 1) * scale + 1) * *elemBytes;
    desc.accessOffset = vci.getIndex();
    return desc;
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

// Reuse the distribution contract for memory width, not result-vreg width.
// Refine only verified shrinking forms; broadcasts retain their fixed ISA width.
static std::optional<uint64_t> vldsMemoryFootprintBytes(pto::VldsOp op)
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
    auto bytes = elementByteSize(op.getSource().getType());
    auto count = vregElementCount(op.getResult().getType());
    if (!bytes || !count || *count == 0) {
        return std::nullopt;
    }
    __int128 vectorBytes = static_cast<__int128>(*bytes) * static_cast<__int128>(*count);
    if (vectorBytes <= 0 || vectorBytes > INT64_MAX) {
        return std::nullopt;
    }
    const VPTOMemoryDistContract* contract =
        pto::lookupVPTOMemoryDist(pto::VPTOMemoryOpFamily::Load, *dist, std::nullopt);
    if (!contract) {
        return std::nullopt;
    }
    if (contract->operandElementBits != 0 &&
        static_cast<int64_t>(contract->operandElementBits) != static_cast<int64_t>(*bytes) * 8) {
        return std::nullopt;
    }
    switch (contract->transfer) {
        case VPTOMemoryTransferKind::Upsample2:
        case VPTOMemoryTransferKind::LaneExpand2:
        case VPTOMemoryTransferKind::LaneExpand4:
        case VPTOMemoryTransferKind::ElementToBlock:
            break;
        default:
            return std::nullopt;
    }
    auto footprint = contract->getFullActiveFootprintBytes(static_cast<int64_t>(vectorBytes));
    bool footprintInvalid = !footprint || (footprint && *footprint < 0);
    bool footprintTooLarge = footprint && static_cast<__int128>(*footprint) > vectorBytes;
    if (footprintInvalid || footprintTooLarge) {
        return std::nullopt;
    }
    return uint64_t(*footprint);
}

// Describe the single-vreg contiguous accesses: uvld/vlds loads and the vsts
// store. Returns nullopt when `op` is none of them.
static std::optional<PTOMemoryAccessDescriptor> describeContiguousAccess(Operation* op, ArrayRef<Value> ivs)
{
    PTOMemoryAccessDescriptor desc;

    // SIMT scalar load/store has an element-offset UB footprint when used
    // inside a vecscope.
    if (auto load = dyn_cast<pto::PTOLoadOp>(op)) {
        desc.kind = PTOMemoryAccessKind::Load;
        desc.unit = PTOMemoryAccessUnit::Scalar;
        fillContiguous(
            desc, load.getPtr(), load.getOffset(), elementByteSize(load.getPtr().getType()), uint64_t(1), ivs);
        return desc;
    }
    if (auto store = dyn_cast<pto::PTOStoreOp>(op)) {
        desc.kind = PTOMemoryAccessKind::Store;
        desc.unit = PTOMemoryAccessUnit::Scalar;
        fillContiguous(
            desc, store.getPtr(), store.getOffset(), elementByteSize(store.getPtr().getType()), uint64_t(1), ivs);
        return desc;
    }
    if (auto uvld = dyn_cast<pto::UvldOp>(op)) {
        desc.kind = PTOMemoryAccessKind::Load;
        fillContiguous(
            desc, uvld.getSource(), uvld.getOffset(), elementByteSize(uvld.getSource().getType()),
            vregElementCount(uvld.getResult().getType()), ivs);
        return desc;
    }
    if (auto vlds = dyn_cast<pto::VldsOp>(op)) {
        desc.kind = PTOMemoryAccessKind::Load;
        fillContiguous(
            desc, vlds.getSource(), vlds.getOffset(), elementByteSize(vlds.getSource().getType()),
            vregElementCount(vlds.getResult().getType()), ivs);
        // A distribution reads a different memory-side width than the full result-vreg storage (a scalar
        // broadcast reads a fixed-width element; an unpack reads half the register).
        if (auto memoryBytes = vldsMemoryFootprintBytes(vlds)) {
            desc.conservativeByteSize = *memoryBytes;
        }
        return desc;
    }
    if (auto vsts = dyn_cast<pto::VstsOp>(op)) {
        desc.kind = PTOMemoryAccessKind::Store;
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
static std::optional<PTOMemoryAccessDescriptor> describePredicateAccess(Operation* op, ArrayRef<Value> ivs)
{
    bool unsupportedPredicate = !isPredicateMemory(op) || getTargetArch(op) != PTOArch::A5;
    if (unsupportedPredicate) {
        return std::nullopt;
    }
    PTOMemoryAccessDescriptor desc;
    desc.kind = isa<pto::PldsOp, pto::PldiOp>(op) ? PTOMemoryAccessKind::Load : PTOMemoryAccessKind::Store;
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
            desc.byteOffset.lossy = true;
        }
    }
    auto dist = op->getAttrOfType<StringAttr>("dist");
    bool normalDistribution = dist && dist.getValue() == "NORM";
    if (normalDistribution) {
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
static std::optional<PTOMemoryAccessDescriptor> describePairedContiguousAccess(Operation* op, ArrayRef<Value> ivs)
{
    PTOMemoryAccessDescriptor desc;
    if (auto vldsx2 = dyn_cast<pto::Vldsx2Op>(op)) {
        desc.kind = PTOMemoryAccessKind::Load;
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
        desc.kind = PTOMemoryAccessKind::Store;
        Value dst = vstsx2.getDestination();
        auto lo = vregElementCount(vstsx2.getLow().getType());
        auto hi = vregElementCount(vstsx2.getHigh().getType());
        std::optional<uint64_t> total;
        if (lo && hi) {
            total = *lo + *hi;
        }
        // INTLV ignores the predicate and writes both complete registers.
        fillContiguous(desc, dst, vstsx2.getOffset(), elementByteSize(dst.getType()), total, ivs);
        return desc;
    }
    return std::nullopt;
}

// Constant-stride vsstb writes a union of 32-byte blocks, not the enclosing
// interval: gaps must remain available to independent accesses. Unknown masks
// retain all blocks. Partial predicates deliberately do not shrink the range
// until their block-selection semantics are proven.
static std::optional<PTOMemoryAccessDescriptor> describeBlockStore(Operation* op, ArrayRef<Value> ivs)
{
    auto store = dyn_cast<pto::VsstbOp>(op);
    if (!store) {
        return std::nullopt;
    }
    APInt stride;
    if (!matchPattern(store.getBlockStride(), m_ConstantInt(&stride))) {
        return std::nullopt;
    }
    PTOMemoryAccessDescriptor desc;
    desc.kind = PTOMemoryAccessKind::Store;
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

static std::optional<PTOMemoryAccessDescriptor> describeBoundedIndexedAccess(
    Value base, Value offsets, Value mask, PTOMemoryAccessKind kind)
{
    auto bytes = elementByteSize(base.getType());
    auto range = bytes ? rangeProof::boundedIndexByteRange(offsets, mask, *bytes) : std::nullopt;
    if (!range) {
        return std::nullopt;
    }
    PTOMemoryAccessDescriptor desc;
    desc.kind = kind;
    desc.base = base;
    desc.addressSpace = getTypeAddressSpace(base.getType());
    desc.byteOffset.constant = range->first;
    desc.conservativeByteSize = range->second;
    desc.offsetDomainCovered = true;
    return desc;
}

// Describe the accesses whose footprint cannot be modelled exactly and must
// stay conservatively MayAlias: strided/post-update stores (vsstb), gathers,
// and scatters. Returns nullopt when `op` is none of them.
static std::optional<PTOMemoryAccessDescriptor> describeForcedMayAliasAccess(Operation* op)
{
    PTOMemoryAccessDescriptor desc;
    if (auto vsstb = dyn_cast<pto::VsstbOp>(op)) {
        desc.kind = PTOMemoryAccessKind::Store;
        desc.base = vsstb.getDestination();
        desc.addressSpace = getTypeAddressSpace(desc.base.getType());
        // Strided/post-update stores can cover a non-contiguous footprint and may
        // carry their next base through scf.for.  Model them conservatively until
        // block/repeat stride ranges are represented explicitly.
        desc.forcesMayAlias = true;
        return desc;
    }

    // Predicate-certified indices give a closed envelope; otherwise cover the
    // complete unsigned hardware domain, independent of vector arithmetic.
    if (isa<pto::Vgather2Op, pto::Vgather2BcOp>(op)) {
        if (auto bounded = describeBoundedIndexedAccess(
                op->getOperand(0), op->getOperand(1), op->getOperand(2), PTOMemoryAccessKind::Load)) {
            return bounded;
        }
        Value offsets = op->getOperand(1);
        Value source = op->getOperand(0);
        auto offsetsType = dyn_cast<pto::VRegType>(offsets.getType());
        auto elem = offsetsType ? dyn_cast<IntegerType>(offsetsType.getElementType()) : IntegerType();
        auto bytes = elementByteSize(source.getType());
        bool supportedWidth = elem && (elem.getWidth() == 16 || elem.getWidth() == 32);
        if (supportedWidth && bytes) {
            desc.kind = PTOMemoryAccessKind::Load;
            desc.base = source;
            desc.addressSpace = getTypeAddressSpace(desc.base.getType());
            desc.conservativeByteSize = (uint64_t(1) << elem.getWidth()) * *bytes;
            unsigned budget = 64;
            if (auto indices = constantIndexLanes(offsets, budget, true)) {
                auto bounds = std::minmax_element(indices->begin(), indices->end());
                desc.byteOffset.constant = *bounds.first * *bytes;
                desc.conservativeByteSize = (*bounds.second - *bounds.first + 1) * *bytes;
            }
            desc.offsetDomainCovered = true;
            return desc;
        }
    }
    // Other gather forms retain their original conservative contract.
    if (isa<pto::Vgather2Op, pto::VgatherbOp, pto::Vgather2BcOp>(op)) {
        desc.kind = PTOMemoryAccessKind::Load;
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
    // Preserve the tighter prefix/ramp model before trying a general envelope.
    if (auto vscatter = dyn_cast<pto::VscatterOp>(op)) {
        if (auto contiguous = describeContiguousScatter(vscatter)) {
            return *contiguous;
        }
        if (auto bounded = describeBoundedIndexedAccess(
                vscatter.getDestination(), vscatter.getOffsets(), vscatter.getMask(), PTOMemoryAccessKind::Store)) {
            return bounded;
        }
        desc.kind = PTOMemoryAccessKind::Store;
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
static void describeByMemoryEffects(Operation* op, PTOMemoryAccessDescriptor& desc)
{
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
    cast<MemoryEffectOpInterface>(op).getEffects(effects);
    for (const auto& effect : effects) {
        Value value = effect.getValue();
        bool outsideUB = !value || getTypeAddressSpace(value.getType()) != AddressSpace::VEC;
        if (outsideUB) {
            continue;
        }
        desc.base = value;
        desc.addressSpace = AddressSpace::VEC;
        desc.kind =
            isa<MemoryEffects::Write>(effect.getEffect()) ? PTOMemoryAccessKind::Store : PTOMemoryAccessKind::Load;
        break;
    }

    // The stateful unaligned-store family also reports a Read on the same UB
    // base because the pointer participates in the state update, so the
    // first-VEC-effect scan above can pick that Read. Keep the explicit Store
    // classification as the semantic source of truth even though the Write
    // effect is now present; it also guards the analysis if the effect order
    // changes.
    if (isa<pto::PstuOp, pto::VstusOp, pto::VsturOp>(op)) {
        desc.kind = PTOMemoryAccessKind::Store;
    }

    desc.forcesMayAlias = true;
}

} // namespace

FailureOr<PTOMemoryAccessDescriptor> pto::buildMemoryAccessDescriptor(Operation* op, ArrayRef<Value> ivs)
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

    PTOMemoryAccessDescriptor desc;
    describeByMemoryEffects(op, desc);
    return desc;
}

// Reconstruct the full footprint (with resolved root) for an access. The
// descriptor stores the base + byte offset + size; here we resolve the root
// provenance from `base` so two accesses can be compared at root equality.
PTOMemoryFootprint pto::memoryFootprintFromDescriptor(const PTOMemoryAccessDescriptor& desc, ArrayRef<Value> ivs)
{
    PTOMemoryFootprint fp;
    fp.addressSpace = desc.addressSpace;
    fp.byteOffset = desc.byteOffset;
    fp.byteSize = desc.conservativeByteSize;
    fp.forcesMayAlias = desc.forcesMayAlias;
    fp.accessBase = desc.base;
    fp.accessOffset = desc.accessOffset;
    fp.offsetDomainCovered = desc.offsetDomainCovered;
    if (desc.forcesMayAlias) {
        // Non-contiguous footprint (vsstb/gather/scatter) cannot be precisely
        // sized, but its base provenance is still resolvable. Resolve the root
        // (allowing scf.for iter-arg init backtracking for post-update stores)
        // so the access can be proven NoAlias against disjoint absolute or
        // distinct-allocation buffers. forcesMayAlias + byteSize=nullopt encodes
        // "lower bound known, open upper bound" and keeps same-root pairs
        // conservatively MayAlias.
        fp.rootKind = resolveRoot(desc.base, fp.byteOffset, fp.root, fp.absoluteBase, ivs, true);
        return fp;
    }
    fp.rootKind = resolveRoot(desc.base, fp.byteOffset, fp.root, fp.absoluteBase, ivs);
    return fp;
}

SmallVector<PTOMemoryFootprint> pto::getMemoryFootprints(Operation* op, ArrayRef<Value> ivs)
{
    auto descriptor = buildMemoryAccessDescriptor(op, ivs);
    if (failed(descriptor)) {
        return {};
    }
    return getMemoryFootprints(*descriptor, ivs);
}

SmallVector<PTOMemoryFootprint> pto::getMemoryFootprints(
    const PTOMemoryAccessDescriptor& descriptor, ArrayRef<Value> ivs)
{
    if (descriptor.blockByteOffsets.empty()) {
        return {memoryFootprintFromDescriptor(descriptor, ivs)};
    }
    SmallVector<PTOMemoryFootprint> result;
    for (int64_t offset : descriptor.blockByteOffsets) {
        auto piece = descriptor;
        PTOAffineByteExpr displacement;
        displacement.constant = offset;
        piece.byteOffset.combine(displacement);
        result.push_back(memoryFootprintFromDescriptor(piece, ivs));
    }
    return result;
}
