// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Support/CodeConstants.h"
#include "VPTOExpandAsyncComm.h"

#include "PTO/IR/PTO.h"
#include "PTO/Support/AsyncSessionABI.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {

using mlir::pto::expand_async::DmaArch;

static Value getI64Constant(Location loc, RewriterBase& rewriter, uint64_t value)
{
    return rewriter.create<arith::ConstantIntOp>(loc, value, mlir::pto::kValue64);
}

static Value getI32Constant(Location loc, RewriterBase& rewriter, int64_t value)
{
    return rewriter.create<arith::ConstantIntOp>(loc, value, mlir::pto::kValue32);
}

//===----------------------------------------------------------------------===//
// Async SDMA post
//===----------------------------------------------------------------------===//

namespace comm_abi = mlir::pto::comm;

// Materialize a GM pointer to `elementType` at `baseAddr + byteOffset`.
// Descriptor and SQE fields sit at fixed byte offsets but have mixed widths, so
// each access folds its offset into the address and then loads or stores at
// element index zero. That keeps the element-offset operand of ld_dev/st_dev out
// of the picture, where a stale element size would silently move the access.
static Value gmFieldPointer(Location loc, RewriterBase& rewriter, Value baseAddr, int64_t byteOffset, Type elementType)
{
    Value addr = baseAddr;
    if (byteOffset != 0) {
        Value offset = getI64Constant(loc, rewriter, byteOffset);
        addr = rewriter.create<arith::AddIOp>(loc, addr, offset);
    }
    auto ptrType = pto::PtrType::get(
        rewriter.getContext(), elementType, pto::AddressSpaceAttr::get(rewriter.getContext(), pto::AddressSpace::GM));
    return rewriter.create<pto::CastPtrOp>(loc, ptrType, addr);
}

static Value loadDevField(Location loc, RewriterBase& rewriter, Value baseAddr, int64_t byteOffset, Type elementType)
{
    Value ptr = gmFieldPointer(loc, rewriter, baseAddr, byteOffset, elementType);
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    return rewriter.create<pto::PTOLdDevOp>(loc, elementType, ptr, zero);
}

static void storeDevField(Location loc, RewriterBase& rewriter, Value baseAddr, int64_t byteOffset, Value value)
{
    Value ptr = gmFieldPointer(loc, rewriter, baseAddr, byteOffset, value.getType());
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    rewriter.create<pto::PTOStDevOp>(loc, value, ptr, zero);
}

// Descriptor and SQE writes go through an ordinary store, not st_dev.
// On 910B1 only the first handful of st_dev stores to HBM take effect and the
// rest are dropped, with barriers making no difference; an ordinary store lands
// every time. That is enough to disqualify st_dev here, since one post writes
// seven words per SQE. st_dev is still what rings the doorbell, which is a real
// device register rather than memory.
static void storeGmField(Location loc, RewriterBase& rewriter, Value baseAddr, int64_t byteOffset, Value value)
{
    Value ptr = gmFieldPointer(loc, rewriter, baseAddr, byteOffset, value.getType());
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    rewriter.create<pto::PTOStoreOp>(loc, ptr, zero, value);
}

// A2/A3 rings the doorbell through UB instead of storing to it.
// sq_reg_base names a register rather than memory, and on this generation it
// only takes a value that arrives by MTE: st_dev has no effect there, and a
// scalar store to it faults the vector unit hard enough to leave the card in
// an unrecoverable RAS state. Staging the tail in UB and moving four bytes out
// is what the reference SDMA implementation does. The staging slot is the
// session's tmp_buf, which exists for exactly this.
static void ringDoorbellViaUb(
    Location loc, RewriterBase& rewriter, Value tmpBufAddr, Value syncId32, Value doorbellAddr, int64_t byteOffset,
    Value tail32)
{
    MLIRContext* ctx = rewriter.getContext();
    Type i32Type = rewriter.getI32Type();

    auto ubPtrType = pto::PtrType::get(ctx, i32Type, pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::VEC));
    Value ubPtr = rewriter.create<pto::CastPtrOp>(loc, ubPtrType, tmpBufAddr);
    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    rewriter.create<pto::PTOStoreOp>(loc, ubPtr, zeroIdx, tail32);

    // The scalar unit has to be done with the slot before MTE3 picks it up. The
    // event id comes from the session so a caller who is already using MTE3
    // events elsewhere can keep this staging off them; a fixed id would collide
    // silently.
    auto pipe = [&ctx](pto::PIPE p) { return pto::PipeAttr::get(ctx, p); };
    Value eventId = rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getIndexType(), syncId32);
    rewriter.create<pto::SetFlagDynOp>(loc, pipe(pto::PIPE::PIPE_S), pipe(pto::PIPE::PIPE_MTE3), eventId);
    rewriter.create<pto::WaitFlagDynOp>(loc, pipe(pto::PIPE::PIPE_S), pipe(pto::PIPE::PIPE_MTE3), eventId);

    Value dbPtr = gmFieldPointer(loc, rewriter, doorbellAddr, byteOffset, i32Type);
    Value four = getI64Constant(loc, rewriter, 4);
    Value one = getI64Constant(loc, rewriter, 1);
    Value zero = getI64Constant(loc, rewriter, 0);

    // A doorbell takes one 32-bit write. The default c220 store carries its
    // length in 32-byte blocks and would round these four bytes down to a
    // transfer of nothing, leaving the engine waiting on a ring it was never
    // told about, so this asks for the byte-granular path.
    auto doorbellStore = rewriter.create<pto::CopyUbufToGmOp>(loc, ubPtr, dbPtr, zero, one, four, zero, zero, zero);
    doorbellStore->setAttr("vpto.byte_granular", rewriter.getUnitAttr());
}

// Those stores land in the data cache, and the engine reads memory, so the
// cache has to be pushed out before the doorbell is rung. One flush of the
// whole data cache covers every SQE of the post plus the descriptor, which is
// also what the reference SDMA implementation does.
static void writebackDataCache(Location loc, RewriterBase& rewriter, Value addr)
{
    Value ptr = gmFieldPointer(loc, rewriter, addr, 0, rewriter.getI8Type());
    rewriter.create<pto::DcciOp>(
        loc, ptr, pto::DcciCacheLineAttr::get(rewriter.getContext(), pto::DcciCacheLine::ENTIRE_DATA_CACHE),
        pto::DcciDstAttr{});
}

// Read one session config field and widen it to i64 for address arithmetic.
static Value readSessionFieldI64(
    Location loc, RewriterBase& rewriter, Value session, comm_abi::SessionField field, unsigned width)
{
    Type fieldType = rewriter.getIntegerType(width);
    Value raw = rewriter.create<pto::StructGetOp>(
        loc, fieldType, session, rewriter.getDenseI64ArrayAttr({comm_abi::sessionFieldIndex(field)}));
    if (width == mlir::pto::kValue64) {
        return raw;
    }
    return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), raw);
}

// For fields that stay 32-bit all the way into an SQE word or an event id,
// where widening to i64 would only have to be undone.
static Value readSessionFieldI32(Location loc, RewriterBase& rewriter, Value session, comm_abi::SessionField field)
{
    return rewriter.create<pto::StructGetOp>(
        loc, rewriter.getI32Type(), session, rewriter.getDenseI64ArrayAttr({comm_abi::sessionFieldIndex(field)}));
}

// The struct that physically backs a session, built from the ABI's width table
// so this pass keeps no second copy of the layout.
static pto::StructType getSessionStructType(MLIRContext* ctx)
{
    SmallVector<Type> fields;
    fields.reserve(comm_abi::kSessionNumFields);
    for (unsigned width : comm_abi::kSessionFieldWidths) {
        fields.push_back(IntegerType::get(ctx, width));
    }
    return pto::StructType::get(ctx, fields);
}

// Materialize the session a pto.session_init produced.
// A session is opaque in the IR a user writes, so the struct behind it is
// declared here, at the point the session was built. That is also the only
// place it can be declared: stack storage may originate from
// pto.declare_struct and nowhere else.
static Value materializeSession(pto::SessionInitOp op, RewriterBase& rewriter)
{
    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);

    pto::StructType structType = getSessionStructType(op.getContext());
    Value session = rewriter.create<pto::DeclareStructOp>(loc, structType);

    Value templateAddr = rewriter.create<pto::CastPtrOp>(loc, rewriter.getI64Type(), op.getTemplateGm());

    for (unsigned i = 0; i < comm_abi::kSessionNumFields; ++i) {
        Type fieldType = structType.getFieldTypes()[i];
        const int64_t offset = static_cast<int64_t>(i * comm_abi::session_tmpl::kSlotBytes);
        Value value = loadDevField(loc, rewriter, templateAddr, offset, fieldType);
        rewriter.create<pto::StructSetOp>(
            loc, session, rewriter.getDenseI64ArrayAttr({static_cast<int64_t>(i)}), value);
    }
    return session;
}

// Expand one asynchronous SDMA post into descriptor reads, SQE writes, a
// release barrier, and a doorbell write.
// The transfer is split into at most `block_bytes` per SQE. Splitting runs as an
// scf.for whose trip count is only known at runtime, mirroring how the DMA
// wrapper ops expand their software loops.
// This version drives a single channel of the group. Spreading a post across the
// group is a scheduling policy on top of the same sequence and is left to a
// follow-up.
// Not a rewrite pattern: a session changes type across this expansion, from the
// opaque !pto.dma_session a caller holds to the struct that backs it, and the
// greedy rewriter cannot replace a value with one of another type. The ordered
// walk in lowerAsyncSessions drives this instead, which also fixes the order
// the two ops are handled in rather than leaving it to pattern selection.
struct SdmaGmGmExpansion {
    static LogicalResult run(pto::SdmaGmGmOp op, Value session, RewriterBase& rewriter, DmaArch dmaArch)
    {
        Location loc = op.getLoc();
        rewriter.setInsertionPoint(op);

        // A5's engine will not PUT to a peer. The same kick becomes a synchronous
        // GM→UB→GM copy, which is how the reference stack writes remotely there.
        // Other generations ignore the attr and keep posting SQEs.
        if (dmaArch == DmaArch::A5 && op.getSoftPutAttr())
            return expandA5SoftPut(op, session, rewriter);

        EnginePostState state = loadEnginePostState(loc, rewriter, session, op);
        Value finalTail32 = emitSqePostLoop(loc, rewriter, op, state, dmaArch);
        publishTailAndRing(loc, rewriter, session, state, finalTail32, dmaArch);
        rewriter.replaceOp(op, completionRecord(loc, rewriter, op, state.recordAddr));
        return success();
    }

private:
    // Queue geometry and endpoints needed to post one engine transfer.
    struct EnginePostState {
        Value recordAddr;
        Value qos32;
        Value blockBytes;
        Value sqBase;
        Value doorbellAddr;
        Value streamId32;
        Value tailAddr;
        Value sqHead;
        Value slotMask;
        Value initialTail;
        Value srcAddr;
        Value dstAddr;
    };

    static EnginePostState loadEnginePostState(Location loc, RewriterBase& rewriter, Value session, pto::SdmaGmGmOp op)
    {
        EnginePostState state;
        state.recordAddr = channelRecordAddr(loc, rewriter, session, op);
        state.qos32 = readSessionFieldI32(loc, rewriter, session, comm_abi::SessionField::Qos);
        if (auto attr = op.getBlockBytes())
            state.blockBytes = getI64Constant(loc, rewriter, *attr);
        else
            state.blockBytes =
                readSessionFieldI64(loc, rewriter, session, comm_abi::SessionField::BlockBytes, mlir::pto::kValue64);

        Type i32Type = rewriter.getI32Type();
        Type i64Type = rewriter.getI64Type();
        state.sqBase = loadDevField(loc, rewriter, state.recordAddr, comm_abi::channel::kSqBaseOffset, i64Type);
        state.doorbellAddr = loadDevField(loc, rewriter, state.recordAddr, comm_abi::channel::kDoorbellOffset, i64Type);
        Value slotMask32 = loadDevField(loc, rewriter, state.recordAddr, comm_abi::channel::kSlotMaskOffset, i32Type);
        state.streamId32 = loadDevField(loc, rewriter, state.recordAddr, comm_abi::channel::kStreamIdOffset, i32Type);
        state.tailAddr = loadDevField(loc, rewriter, state.recordAddr, comm_abi::channel::kTailAddrOffset, i64Type);
        Value headAddr = loadDevField(loc, rewriter, state.recordAddr, comm_abi::channel::kHeadAddrOffset, i64Type);
        Value sqTail32 = loadDevField(loc, rewriter, state.tailAddr, 0, i32Type);
        Value sqHead32 = loadDevField(loc, rewriter, headAddr, 0, i32Type);
        state.sqHead = rewriter.create<arith::ExtUIOp>(loc, i64Type, sqHead32);
        state.slotMask = rewriter.create<arith::ExtUIOp>(loc, i64Type, slotMask32);
        state.initialTail = rewriter.create<arith::ExtUIOp>(loc, i64Type, sqTail32);
        state.srcAddr = rewriter.create<pto::CastPtrOp>(loc, i64Type, op.getSource());
        state.dstAddr = rewriter.create<pto::CastPtrOp>(loc, i64Type, op.getDestination());
        return state;
    }

    // Split the transfer into at most block_bytes per SQE. Returns the truncated
    // final tail that must be published and rung.
    static Value emitSqePostLoop(
        Location loc, RewriterBase& rewriter, pto::SdmaGmGmOp op, EnginePostState state, DmaArch dmaArch)
    {
        Type i32Type = rewriter.getI32Type();
        Value nbytes = op.getNbytes();
        Value oneBlock = getI64Constant(loc, rewriter, 1);
        Value blockBytes = rewriter.create<arith::MaxUIOp>(loc, state.blockBytes, oneBlock);
        Value iterations = rewriter.create<arith::CeilDivUIOp>(loc, nbytes, blockBytes);
        Value iterationsIdx = rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getIndexType(), iterations);
        Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        auto forOp = rewriter.create<scf::ForOp>(loc, zeroIdx, iterationsIdx, oneIdx, ValueRange{state.initialTail});
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(forOp.getBody());
            Value nextTail = emitOneSqeIteration(loc, rewriter, forOp, state, blockBytes, nbytes, dmaArch);
            rewriter.create<scf::YieldOp>(loc, ValueRange{nextTail});
        }
        Value finalTail = forOp.getResult(0);
        return rewriter.create<arith::TruncIOp>(loc, i32Type, finalTail);
    }

    static Value emitOneSqeIteration(
        Location loc, RewriterBase& rewriter, scf::ForOp forOp, EnginePostState state, Value blockBytes, Value nbytes,
        DmaArch dmaArch)
    {
        Type i32Type = rewriter.getI32Type();
        Type i64Type = rewriter.getI64Type();
        Value iv = rewriter.create<arith::IndexCastUIOp>(loc, i64Type, forOp.getInductionVar());
        Value tail = forOp.getRegionIterArg(0);
        Value chunkOffset = rewriter.create<arith::MulIOp>(loc, iv, blockBytes);
        Value remaining = rewriter.create<arith::SubIOp>(loc, nbytes, chunkOffset);
        Value chunkBytes = rewriter.create<arith::MinUIOp>(loc, blockBytes, remaining);
        Value chunkSrc = rewriter.create<arith::AddIOp>(loc, state.srcAddr, chunkOffset);
        Value chunkDst = rewriter.create<arith::AddIOp>(loc, state.dstAddr, chunkOffset);
        Value slot = rewriter.create<arith::AndIOp>(loc, tail, state.slotMask);
        Value sqeBytes = getI64Constant(loc, rewriter, static_cast<int64_t>(comm_abi::sqe::kBytes));
        Value slotOffset = rewriter.create<arith::MulIOp>(loc, slot, sqeBytes);
        Value sqeAddr = rewriter.create<arith::AddIOp>(loc, state.sqBase, slotOffset);
        Value taskId = rewriter.create<arith::SubIOp>(loc, tail, state.sqHead);
        Value taskId32 = rewriter.create<arith::TruncIOp>(loc, i32Type, taskId);
        writeMemcpySqe(
            loc, rewriter, sqeAddr, chunkSrc, chunkDst, chunkBytes, state.streamId32, taskId32, state.qos32, dmaArch);
        Value one = getI64Constant(loc, rewriter, 1);
        Value nextTail = rewriter.create<arith::AddIOp>(loc, tail, one);
        return rewriter.create<arith::AndIOp>(loc, nextTail, state.slotMask);
    }

    static void publishTailAndRing(
        Location loc, RewriterBase& rewriter, Value session, EnginePostState state, Value finalTail32, DmaArch dmaArch)
    {
        // Publish only the tail. The head belongs to the engine.
        storeGmField(loc, rewriter, state.tailAddr, 0, finalTail32);
        writebackDataCache(loc, rewriter, state.sqBase);
        rewriter.create<pto::DsbOp>(loc, pto::DsbMemAttr::get(rewriter.getContext(), pto::DsbMem::DDR));

        const int64_t doorbellOffset =
            dmaArch == DmaArch::A5 ? comm_abi::sqe::kDoorbellOffsetA5 : comm_abi::sqe::kDoorbellOffsetA2A3;
        if (dmaArch == DmaArch::A5) {
            storeDevField(loc, rewriter, state.doorbellAddr, doorbellOffset, finalTail32);
            return;
        }
        Value tmpBufAddr = readSessionFieldI64(loc, rewriter, session, comm_abi::SessionField::TmpBufAddr, 64);
        Value syncId32 = readSessionFieldI32(loc, rewriter, session, comm_abi::SessionField::SyncId);
        ringDoorbellViaUb(loc, rewriter, tmpBufAddr, syncId32, state.doorbellAddr, doorbellOffset, finalTail32);
    }

    static Value channelRecordAddr(Location loc, RewriterBase& rewriter, Value session, pto::SdmaGmGmOp op)
    {
        Value contextGm = readSessionFieldI64(loc, rewriter, session, comm_abi::SessionField::ContextGm, 64);
        Value channelNum = readSessionFieldI64(loc, rewriter, session, comm_abi::SessionField::ChannelNum, 32);
        Value channelIdx;
        if (auto attr = op.getChannelIdx())
            channelIdx = getI64Constant(loc, rewriter, *attr);
        else
            channelIdx =
                readSessionFieldI64(loc, rewriter, session, comm_abi::SessionField::ChannelIdx, mlir::pto::kValue32);
        Value recordIndex = rewriter.create<arith::MulIOp>(loc, channelIdx, channelNum);
        Value recordBytes = getI64Constant(loc, rewriter, static_cast<int64_t>(comm_abi::channel::kRecordBytes));
        Value recordOffset = rewriter.create<arith::MulIOp>(loc, recordIndex, recordBytes);
        return rewriter.create<arith::AddIOp>(loc, contextGm, recordOffset);
    }

    static Value completionRecord(Location loc, RewriterBase& rewriter, pto::SdmaGmGmOp op, Value recordAddr)
    {
        return rewriter.create<pto::CastPtrOp>(loc, op.getCqRecord().getType(), recordAddr);
    }

    static LogicalResult expandA5SoftPut(pto::SdmaGmGmOp op, Value session, RewriterBase& rewriter)
    {
        Location loc = op.getLoc();
        MLIRContext* ctx = rewriter.getContext();
        Value syncId32 = readSessionFieldI32(loc, rewriter, session, comm_abi::SessionField::SyncId);
        Value tmpBufAddr = readSessionFieldI64(loc, rewriter, session, comm_abi::SessionField::TmpBufAddr, 64);
        Type i64Type = rewriter.getI64Type();
        Value srcAddr = rewriter.create<pto::CastPtrOp>(loc, i64Type, op.getSource());
        Value dstAddr = rewriter.create<pto::CastPtrOp>(loc, i64Type, op.getDestination());
        auto i8Type = rewriter.getI8Type();
        auto gmI8Type = pto::PtrType::get(ctx, i8Type, pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::GM));
        auto ubType = pto::PtrType::get(ctx, i8Type, pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::VEC));
        Value ub = rewriter.create<pto::CastPtrOp>(loc, ubType, tmpBufAddr);
        emitSoftPutChunkLoop(loc, rewriter, ctx, op.getNbytes(), srcAddr, dstAddr, ub, gmI8Type, syncId32);
        rewriter.create<pto::DsbOp>(loc, pto::DsbMemAttr::get(ctx, pto::DsbMem::DDR));
        Value nullRecord = getI64Constant(loc, rewriter, 0);
        rewriter.replaceOp(op, completionRecord(loc, rewriter, op, nullRecord));
        return success();
    }

    static void emitSoftPutChunkLoop(
        Location loc, RewriterBase& rewriter, MLIRContext* ctx, Value nbytes, Value srcAddr, Value dstAddr, Value ub,
        pto::PtrType gmI8Type, Value syncId32)
    {
        Value chunkBytes = getI64Constant(loc, rewriter, 32768);
        Value one = getI64Constant(loc, rewriter, 1);
        chunkBytes = rewriter.create<arith::MaxUIOp>(loc, chunkBytes, one);
        Value iterations = rewriter.create<arith::CeilDivUIOp>(loc, nbytes, chunkBytes);
        Value iterationsIdx = rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getIndexType(), iterations);
        Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        Value eventId = rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getIndexType(), syncId32);

        auto forOp = rewriter.create<scf::ForOp>(loc, zeroIdx, iterationsIdx, oneIdx);
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(forOp.getBody());
        emitSoftPutChunk(loc, rewriter, ctx, forOp, nbytes, chunkBytes, one, srcAddr, dstAddr, ub, gmI8Type, eventId);
    }

    static void emitSoftPutChunk(
        Location loc, RewriterBase& rewriter, MLIRContext* ctx, scf::ForOp forOp, Value nbytes, Value chunkBytes,
        Value one, Value srcAddr, Value dstAddr, Value ub, pto::PtrType gmI8Type, Value eventId)
    {
        Type i64Type = rewriter.getI64Type();
        Value zero64 = getI64Constant(loc, rewriter, 0);
        Value falseBit = rewriter.create<arith::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getBoolAttr(false));
        auto pipe = [&ctx](pto::PIPE p) { return pto::PipeAttr::get(ctx, p); };
        Value iv = rewriter.create<arith::IndexCastUIOp>(loc, i64Type, forOp.getInductionVar());
        Value chunkOffset = rewriter.create<arith::MulIOp>(loc, iv, chunkBytes);
        Value remaining = rewriter.create<arith::SubIOp>(loc, nbytes, chunkOffset);
        Value thisBytes = rewriter.create<arith::MinUIOp>(loc, chunkBytes, remaining);
        Value chunkSrc =
            rewriter.create<pto::CastPtrOp>(loc, gmI8Type, rewriter.create<arith::AddIOp>(loc, srcAddr, chunkOffset));
        Value chunkDst =
            rewriter.create<pto::CastPtrOp>(loc, gmI8Type, rewriter.create<arith::AddIOp>(loc, dstAddr, chunkOffset));
        rewriter.create<pto::CopyGmToUbufOp>(
            loc, chunkSrc, ub, zero64, one, thisBytes, zero64, zero64, falseBit, zero64, zero64, zero64);
        rewriter.create<pto::SetFlagDynOp>(loc, pipe(pto::PIPE::PIPE_MTE2), pipe(pto::PIPE::PIPE_MTE3), eventId);
        rewriter.create<pto::WaitFlagDynOp>(loc, pipe(pto::PIPE::PIPE_MTE2), pipe(pto::PIPE::PIPE_MTE3), eventId);
        rewriter.create<pto::CopyUbufToGmOp>(loc, ub, chunkDst, zero64, one, thisBytes, zero64, zero64, zero64);
        rewriter.create<pto::SetFlagDynOp>(loc, pipe(pto::PIPE::PIPE_MTE3), pipe(pto::PIPE::PIPE_MTE2), eventId);
        rewriter.create<pto::WaitFlagDynOp>(loc, pipe(pto::PIPE::PIPE_MTE3), pipe(pto::PIPE::PIPE_MTE2), eventId);
    }

    static void writeMemcpySqe(
        Location loc, RewriterBase& rewriter, Value sqeAddr, Value src, Value dst, Value bytes, Value streamId32,
        Value taskId32, Value qos32, DmaArch dmaArch)
    {
        const bool isA5 = dmaArch == DmaArch::A5;
        Value qos = rewriter.create<arith::AndIOp>(loc, qos32, getI32Constant(loc, rewriter, comm_abi::sqe::kQosMask));
        writeMemcpySqeControlWords(loc, rewriter, sqeAddr, streamId32, taskId32, qos, isA5);
        writeMemcpySqePayload(loc, rewriter, sqeAddr, src, dst, bytes, isA5);
    }

    static void writeMemcpySqeControlWords(
        Location loc, RewriterBase& rewriter, Value sqeAddr, Value streamId32, Value taskId32, Value qos, bool isA5)
    {
        storeGmField(
            loc, rewriter, sqeAddr, comm_abi::sqe::kWord0Offset,
            getI32Constant(loc, rewriter, isA5 ? comm_abi::sqe::a5::kWord0Memcpy : comm_abi::sqe::a2a3::kWord0Memcpy));
        Value halfMask = getI32Constant(loc, rewriter, 0xFFFF);
        Value rtStreamId = rewriter.create<arith::AndIOp>(loc, streamId32, halfMask);
        Value taskId = rewriter.create<arith::AndIOp>(loc, taskId32, halfMask);
        Value taskIdShift = getI32Constant(loc, rewriter, comm_abi::sqe::kTaskIdShift);
        Value taskIdField = rewriter.create<arith::ShLIOp>(loc, taskId, taskIdShift);
        Value word1 = rewriter.create<arith::OrIOp>(loc, rtStreamId, taskIdField);
        storeGmField(loc, rewriter, sqeAddr, comm_abi::sqe::kWord1Offset, word1);
        storeGmField(
            loc, rewriter, sqeAddr, comm_abi::sqe::kWord3Offset,
            getI32Constant(loc, rewriter, isA5 ? comm_abi::sqe::a5::kWord3Memcpy : comm_abi::sqe::a2a3::kWord3Memcpy));
        Value word4 =
            getI32Constant(loc, rewriter, isA5 ? comm_abi::sqe::a5::kWord4Memcpy : comm_abi::sqe::a2a3::kWord4Memcpy);
        if (!isA5) {
            Value qosField =
                rewriter.create<arith::ShLIOp>(loc, qos, getI32Constant(loc, rewriter, comm_abi::sqe::a2a3::kQosShift));
            word4 = rewriter.create<arith::OrIOp>(loc, word4, qosField);
        }
        storeGmField(loc, rewriter, sqeAddr, comm_abi::sqe::kWord4Offset, word4);
        if (isA5) {
            Value word5 =
                rewriter.create<arith::ShLIOp>(loc, qos, getI32Constant(loc, rewriter, comm_abi::sqe::a5::kQosShift));
            storeGmField(loc, rewriter, sqeAddr, comm_abi::sqe::a5::kWord5Offset, word5);
        }
    }

    static void writeMemcpySqePayload(
        Location loc, RewriterBase& rewriter, Value sqeAddr, Value src, Value dst, Value bytes, bool isA5)
    {
        storeGmField(loc, rewriter, sqeAddr, comm_abi::sqe::kSrcAddrOffset, src);
        storeGmField(loc, rewriter, sqeAddr, comm_abi::sqe::kDstAddrOffset, dst);
        Value bytes32 = rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(), bytes);
        storeGmField(
            loc, rewriter, sqeAddr, isA5 ? comm_abi::sqe::a5::kLengthOffset : comm_abi::sqe::a2a3::kLengthOffset,
            bytes32);
        if (!isA5)
            storeGmField(
                loc, rewriter, sqeAddr, comm_abi::sqe::a2a3::kLinkTypeOffset,
                getI32Constant(loc, rewriter, comm_abi::sqe::a2a3::kLinkTypeNone));
    }
};

// Lower every async session in one ordered walk: build each session, hand it to
// the posts that read it, then drop the pto.session_init ops.
// The order is the reason this is not left to the greedy rewriter. A post needs
// the struct its session was materialized into, so every session_init has to be
// expanded before any post reading it, and a session_init can only be erased
// once no post still refers to its result.

} // namespace

LogicalResult mlir::pto::expand_async::lowerAsyncSessions(func::FuncOp func, mlir::pto::expand_async::DmaArch dmaArch)
{
    SmallVector<pto::SessionInitOp> inits;
    func.walk([&](pto::SessionInitOp op) { inits.push_back(op); });
    SmallVector<pto::SdmaGmGmOp> posts;
    func.walk([&](pto::SdmaGmGmOp op) { posts.push_back(op); });
    if (inits.empty() && posts.empty()) {
        return success();
    }

    IRRewriter rewriter(func.getContext());
    DenseMap<Value, Value> sessions;
    for (pto::SessionInitOp init : inits) {
        sessions[init.getSession()] = materializeSession(init, rewriter);
    }

    for (pto::SdmaGmGmOp post : posts) {
        Value session = sessions.lookup(post.getSession());
        if (!session) {
            // Well-typed but not traceable to a session this pass materialized, so
            // there is no struct to read. The verifier cannot catch this: the type is
            // right, the value just came from somewhere else.
            return post.emitOpError() << "session must come from a pto.session_init in the same function";
        }
        if (failed(SdmaGmGmExpansion::run(post, session, rewriter, dmaArch))) {
            return failure();
        }
    }

    // Safe now: only a post can consume a session, and every post is gone.
    for (pto::SessionInitOp init : inits) {
        rewriter.eraseOp(init);
    }
    return success();
}
