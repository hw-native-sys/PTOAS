// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#pragma once

// Short-vector interleave result repacking helpers.
// Included by VMIToVPTOPatternInternals4.cpp after its shared constants.

template <typename SourceOp>
struct ShortInterleaveRepacker {
    LogicalResult repackShortInterleave(
        SourceOp op, SmallVectorImpl<Value>& results, VRegType indexType, Value ramp, Value all, MaskType maskType,
        int64_t lanes, OneToNPatternRewriter& rewriter) const
    {
        int64_t capacity = indexType.getElementCount();
        auto constant = [&](int64_t value) {
            return createScalarOffsetConstant(op.getLoc(), indexType.getElementType(), value, rewriter);
        };
        auto offset = constant(lanes);
        auto wrap = constant(capacity - 1);
        auto prefix = createLaneRangeMask(op.getLoc(), maskType, 0, capacity - lanes, rewriter);
        bool invalidSelector = failed(offset) || failed(wrap) || failed(prefix);
        if (invalidSelector) {
            return failure();
        }
        Value index = rewriter.create<VaddsOp>(op.getLoc(), indexType, ramp, *offset, all);
        Value limit = rewriter.create<VdupOp>(op.getLoc(), indexType, *wrap, all, StringAttr{});
        index = rewriter.create<VandOp>(op.getLoc(), indexType, index, limit, all);
        Value low = rewriter.create<VselrOp>(op.getLoc(), indexType, results[0], index);
        Value high = rewriter.create<VselrOp>(op.getLoc(), indexType, results[1], index);
        results[1] = rewriter.create<VselOp>(op.getLoc(), indexType, low, high, *prefix);
        return success();
    }

    LogicalResult repackShortDeinterleave(
        SourceOp op, SmallVectorImpl<Value>& results, VRegType indexType, Value ramp, Value all, MaskType maskType,
        int64_t lanes, OneToNPatternRewriter& rewriter) const
    {
        int64_t capacity = indexType.getElementCount();
        auto constant = [&](int64_t value) {
            return createScalarOffsetConstant(op.getLoc(), indexType.getElementType(), value, rewriter);
        };
        auto offset = constant((capacity - lanes) / kInterleaveFactor);
        auto prefix = createLaneRangeMask(op.getLoc(), maskType, 0, lanes / kInterleaveFactor, rewriter);
        bool invalidSelector = failed(offset) || failed(prefix);
        if (invalidSelector) {
            return failure();
        }
        Value shifted = rewriter.create<VaddsOp>(op.getLoc(), indexType, ramp, *offset, all);
        Value index = rewriter.create<VselOp>(op.getLoc(), indexType, ramp, shifted, *prefix);
        for (Value& result : results) {
            result = rewriter.create<VselrOp>(op.getLoc(), indexType, result, index);
        }
        return success();
    }

    LogicalResult bitcastInterleaveResults(
        SourceOp op, SmallVectorImpl<Value>& results, VRegType type, OneToNPatternRewriter& rewriter) const
    {
        for (Value& result : results) {
            auto converted = bitcastVReg(op.getLoc(), result, type, rewriter);
            if (failed(converted)) {
                return failure();
            }
            result = *converted;
        }
        return success();
    }

    // Native pair ops split at the carrier midpoint, whereas VMI splits at VL/2.
    // Repack a short logical vector before applying its output predicate.
    LogicalResult repackShortResults(
        SourceOp op, SmallVectorImpl<Value>& results, OneToNPatternRewriter& rewriter) const
    {
        auto type = cast<VRegType>(results.front().getType());
        int64_t lanes = cast<VMIVRegType>(op.getLhs().getType()).getElementCount();
        int64_t capacity = type.getElementCount();
        if (lanes >= capacity) {
            return success();
        }
        auto bits = pto::getPTOStorageElemBitWidth(type.getElementType());
        auto scalarType = IntegerType::get(rewriter.getContext(), bits);
        auto indexType = VRegType::get(rewriter.getContext(), capacity, scalarType);
        VRegType originalType = type;
        if (failed(bitcastInterleaveResults(op, results, indexType, rewriter))) {
            return failure();
        }
        type = indexType;
        auto maskType = getMaskTypeForVReg(type, rewriter.getContext());
        auto all = createAllTrueMaskForVReg(op.getLoc(), indexType, rewriter);
        bool invalidSeed = failed(maskType) || failed(all);
        if (invalidSeed) {
            return failure();
        }
        auto zero = createScalarOffsetConstant(op.getLoc(), scalarType, 0, rewriter);
        if (failed(zero)) {
            return failure();
        }
        Value ramp = rewriter.create<VciOp>(op.getLoc(), indexType, *zero, StringAttr{});
        LogicalResult repacked = success();
        if constexpr (std::is_same_v<SourceOp, VMIVintlvOp>) {
            repacked = repackShortInterleave(op, results, indexType, ramp, *all, *maskType, lanes, rewriter);
        } else {
            repacked = repackShortDeinterleave(op, results, indexType, ramp, *all, *maskType, lanes, rewriter);
        }
        if (failed(repacked)) {
            return failure();
        }
        return bitcastInterleaveResults(op, results, originalType, rewriter);
    }
};
