// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Analysis/PTOIntegerRange.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::integer_range;

namespace {

struct ConstantOperand {
    Value value;
    int64_t constant = 0;
};

static std::optional<ConstantOperand> matchConstantOperand(Operation* binary)
{
    APInt constant;
    for (unsigned side = 0; side < 2; ++side) {
        bool isConstant = matchPattern(binary->getOperand(side), m_ConstantInt(&constant));
        bool fitsInt64 = isConstant && constant.isSignedIntN(64);
        if (isConstant && fitsInt64) {
            return ConstantOperand{binary->getOperand(1 - side), constant.getSExtValue()};
        }
    }
    return std::nullopt;
}

static std::optional<ConstantOperand> matchScaledValue(Value value)
{
    auto multiply = value.getDefiningOp<arith::MulIOp>();
    // Canonicalization removes multiplication by one in the new i32 stage
    // form. Keep that form identical to the explicitly scaled idiom.
    return multiply ? matchConstantOperand(multiply) : std::optional<ConstantOperand>(ConstantOperand{value, 1});
}

static bool isRemainderSign(Value value, Value remainder, int64_t divisor, IntegerType narrow, IntegerType wide)
{
    auto mask = value.getDefiningOp<arith::AndIOp>();
    auto masked = mask ? matchConstantOperand(mask) : std::nullopt;
    if (!masked || masked->constant != divisor) {
        return false;
    }
    auto shift = masked->value.getDefiningOp<arith::ShRSIOp>();
    APInt amount;
    bool validShift = shift && matchPattern(shift.getRhs(), m_ConstantInt(&amount));
    bool shiftFits = validShift && amount.isSignedIntN(64);
    bool expectedShift = shiftFits && amount.getSExtValue() == int64_t(narrow.getWidth()) - 1;
    if (!validShift || !shiftFits || !expectedShift) {
        return false;
    }
    if (narrow == wide) {
        return shift.getLhs() == remainder;
    }
    auto trunc = shift.getLhs().getDefiningOp<arith::TruncIOp>();
    return trunc && trunc.getIn() == remainder;
}

// Check the correction product and all representation bounds before
// recognizing K*(r + (r<0 ? c : 0)) as one non-negative remainder leaf.
static std::optional<int64_t> matchRemainderCorrection(
    Value corrected, Value remainder, int64_t divisor, int64_t scale, IntegerType wide)
{
    Value product = corrected;
    if (auto ext = corrected.getDefiningOp<arith::ExtSIOp>()) {
        product = ext.getIn();
    } else if (auto ext = corrected.getDefiningOp<arith::ExtUIOp>()) {
        product = ext.getIn();
    }
    auto narrow = dyn_cast<IntegerType>(product.getType());
    auto scaled = matchScaledValue(product);
    if (!narrow || !scaled || scaled->constant != scale ||
        !isRemainderSign(scaled->value, remainder, divisor, narrow, wide)) {
        return std::nullopt;
    }
    __int128 narrowLimit = static_cast<__int128>(1) << (narrow.getWidth() - 1);
    __int128 wideLimit = static_cast<__int128>(1) << (wide.getWidth() - 1);
    bool divisorFits = static_cast<__int128>(divisor) <= narrowLimit;
    bool scaleFitsNarrow = static_cast<__int128>(divisor) * scale <= narrowLimit - 1;
    bool scaleFitsWide = static_cast<__int128>(divisor - 1) * scale <= wideLimit - 1;
    if (!divisorFits || !scaleFitsNarrow || !scaleFitsWide) {
        return std::nullopt;
    }
    return static_cast<int64_t>(static_cast<__int128>(scale) * (divisor - 1));
}

static std::optional<int64_t> matchRemSumOrdered(Value corrected, Value remainder)
{
    // This proof uses bounded host-width arithmetic; reject wider chains
    // before extracting APInt constants or shifting representation limits.
    auto wide = dyn_cast<IntegerType>(remainder.getType());
    bool unsupportedWidth = wide && wide.getWidth() > 64;
    if (!wide || unsupportedWidth) {
        return std::nullopt;
    }
    auto scaled = matchScaledValue(remainder);
    if (!scaled || scaled->constant < 0) {
        return std::nullopt;
    }
    auto rem = scaled->value.getDefiningOp<arith::RemSIOp>();
    APInt divisor;
    bool validDivisor = rem && matchPattern(rem.getRhs(), m_ConstantInt(&divisor));
    bool divisorFits = validDivisor && divisor.isSignedIntN(64);
    bool divisorPositive = divisorFits && divisor.getSExtValue() >= 1;
    if (!validDivisor || !divisorFits || !divisorPositive) {
        return std::nullopt;
    }
    return matchRemainderCorrection(corrected, scaled->value, divisor.getSExtValue(), scaled->constant, wide);
}

} // namespace

std::optional<int64_t> integer_range::matchSignCorrectedRemSum(Value sum)
{
    auto add = sum.getDefiningOp<arith::AddIOp>();
    if (!add) {
        return std::nullopt;
    }
    return matchSignCorrectedRemPair(add.getLhs(), add.getRhs());
}

std::optional<int64_t> integer_range::matchSignCorrectedRemPair(Value lhs, Value rhs)
{
    if (auto bound = matchRemSumOrdered(lhs, rhs)) {
        return bound;
    }
    return matchRemSumOrdered(rhs, lhs);
}
