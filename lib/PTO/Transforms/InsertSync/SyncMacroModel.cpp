// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "../Support/TileSubviewAddress.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr size_t kTensorRank = 2;

SmallVector<unsigned> getSequentialEventIds(unsigned count) {
  SmallVector<unsigned> eventIds;
  eventIds.reserve(count);
  for (unsigned eventId = 0; eventId < count; ++eventId) {
    eventIds.push_back(eventId);
  }
  return eventIds;
}

SyncMacroPhase& addPhase(SyncMacroModel& model, PipelineType pipe, ValueRange defValues, ValueRange useValues)
{
    // 逐字段赋值而非聚合初始化：SyncMacroPhase 的字段会随模型演进增减，
    // 聚合初始化在 -Wmissing-field-initializers（CI 开 -Werror）下会因缺
    // 尾部字段初始化器而报错。
    SyncMacroPhase phase;
    phase.phaseId = static_cast<unsigned>(model.phases.size());
    phase.pipe = pipe;
    phase.defValues.append(defValues.begin(), defValues.end());
    phase.useValues.append(useValues.begin(), useValues.end());
    model.phases.push_back(std::move(phase));
    return model.phases.back();
}

void addHiddenEvent(SyncMacroModel &model, PipelineType srcPipe,
                    PipelineType dstPipe, ArrayRef<unsigned> eventIds) {
  model.hiddenEvents.push_back(
      SyncMacroHiddenEvent{srcPipe, dstPipe,
                           SmallVector<unsigned>(eventIds.begin(),
                                                 eventIds.end())});
}

void addBidirectionalHiddenEvent(SyncMacroModel &model, PipelineType firstPipe,
                                 PipelineType secondPipe,
                                 ArrayRef<unsigned> eventIds) {
  addHiddenEvent(model, firstPipe, secondPipe, eventIds);
  addHiddenEvent(model, secondPipe, firstPipe, eventIds);
}

SmallVector<Value> getPingPongValues(Value ping, Value pong = {}) {
  SmallVector<Value> values;
  values.push_back(ping);
  if (pong) {
    values.push_back(pong);
  }
  return values;
}

std::optional<SyncMacroModel> getP2PCommSyncMacroModel(Operation *op) {
  Value dst;
  Value src;
  Value ping;
  Value pong;
  unsigned laneCount = 1;
  if (auto tput = dyn_cast<pto::TPutOp>(op)) {
    dst = tput.getDst();
    src = tput.getSrc();
    ping = tput.getPing();
    pong = tput.getPong();
    laneCount = tput.getPong() ? 2U : 1U;
  } else if (auto tget = dyn_cast<pto::TGetOp>(op)) {
    dst = tget.getDst();
    src = tget.getSrc();
    ping = tget.getPing();
    pong = tget.getPong();
    laneCount = tget.getPong() ? 2U : 1U;
  } else {
    return std::nullopt;
  }

  SyncMacroModel model;
  SmallVector<Value> staging = getPingPongValues(ping, pong);
  // P2P comm library calls first read the source GM through MTE2, then write
  // the destination GM through MTE3, using ping/pong as staging tiles.
  addPhase(model, PipelineType::PIPE_MTE2, staging, ValueRange{src});
  addPhase(model, PipelineType::PIPE_MTE3, ValueRange{dst}, staging);
  addBidirectionalHiddenEvent(model, PipelineType::PIPE_MTE2,
                              PipelineType::PIPE_MTE3,
                              getSequentialEventIds(laneCount));
  return model;
}

// Models one collective communication op that stages through MTE2 (read) and
// MTE3 (write). Returns the event lane count the model needs, or std::nullopt
// when `op` is not a modeled collective op.
static std::optional<unsigned> buildCollectiveCommPhases(Operation *op,
                                                         SyncMacroModel &model) {
  if (auto tgather = dyn_cast<pto::CommTGatherOp>(op)) {
    // TGATHER_IMPL reads each group source through MTE2 and writes the gathered
    // result into dst through MTE3, using ping/pong as staging tiles.
    SmallVector<Value> staging =
        getPingPongValues(tgather.getPing(), tgather.getPong());
    addPhase(model, PipelineType::PIPE_MTE2, staging,
             tgather.getGroup());
    addPhase(model, PipelineType::PIPE_MTE3, ValueRange{tgather.getDst()},
             staging);
    return tgather.getPong() ? 2U : 1U;
  }
  if (auto tscatter = dyn_cast<pto::CommTScatterOp>(op)) {
    // TSCATTER_IMPL reads the source through MTE2 and writes every group
    // destination through MTE3, using ping/pong as staging tiles.
    SmallVector<Value> staging =
        getPingPongValues(tscatter.getPing(), tscatter.getPong());
    addPhase(model, PipelineType::PIPE_MTE2, staging,
             ValueRange{tscatter.getSrc()});
    addPhase(model, PipelineType::PIPE_MTE3, tscatter.getGroup(),
             staging);
    return tscatter.getPong() ? 2U : 1U;
  }
  if (auto tbroadcast = dyn_cast<pto::TBroadcastOp>(op)) {
    // TBROADCAST_IMPL reads the source through MTE2 and writes every group
    // destination through MTE3, using ping/pong as staging tiles.
    SmallVector<Value> staging =
        getPingPongValues(tbroadcast.getPing(), tbroadcast.getPong());
    addPhase(model, PipelineType::PIPE_MTE2, staging,
             ValueRange{tbroadcast.getSrc()});
    addPhase(model, PipelineType::PIPE_MTE3, tbroadcast.getGroup(),
             staging);
    return tbroadcast.getPong() ? 2U : 1U;
  }
  if (auto treduce = dyn_cast<pto::TReduceOp>(op)) {
    // TREDUCE_IMPL reads group sources through MTE2, reduces into acc on the
    // vector pipe, and stores the final result into dst through MTE3, using
    // recvPing/recvPong as receive staging tiles.
    SmallVector<Value> recvStaging =
        getPingPongValues(treduce.getRecvPing(), treduce.getRecvPong());
    SmallVector<Value> reduceUses{treduce.getAcc()};
    reduceUses.append(recvStaging.begin(), recvStaging.end());
    addPhase(model, PipelineType::PIPE_MTE2, recvStaging, treduce.getGroup());
    addPhase(model, PipelineType::PIPE_V, ValueRange{treduce.getAcc()},
             reduceUses);
    addPhase(model, PipelineType::PIPE_MTE3, ValueRange{treduce.getDst()},
             ValueRange{treduce.getAcc()});
    return treduce.getRecvPong() ? 3U : 2U;
  }
  return std::nullopt;
}

std::optional<SyncMacroModel> getCollectiveCommSyncMacroModel(Operation *op) {
  SyncMacroModel model;
  auto laneCount = buildCollectiveCommPhases(op, model);
  if (!laneCount) {
    return std::nullopt;
  }

  SmallVector<unsigned> eventIds = getSequentialEventIds(*laneCount);
  addBidirectionalHiddenEvent(model, PipelineType::PIPE_MTE2,
                              PipelineType::PIPE_MTE3, eventIds);
  if (isa<pto::TReduceOp>(op)) {
    addHiddenEvent(model, PipelineType::PIPE_MTE2, PipelineType::PIPE_V,
                   eventIds);
    addHiddenEvent(model, PipelineType::PIPE_V, PipelineType::PIPE_MTE2,
                   eventIds);
    addHiddenEvent(model, PipelineType::PIPE_V, PipelineType::PIPE_MTE3,
                   eventIds);
  }

  return model;
}

std::optional<SyncMacroModel> getTScatterSyncMacroModel(pto::TScatterOp op) {
  if (!op.hasIndexForm() || getTargetArch(op.getOperation()) == PTOArch::A5) {
    return std::nullopt;
  }

  SyncMacroModel model;
  // A2/A3 indexed TSCATTER first initializes dst with vector_dup, then runs a
  // scalar UB scatter loop over src/indexes.
  addPhase(model, PipelineType::PIPE_V, ValueRange{op.getDst()}, ValueRange{});
  addPhase(model, PipelineType::PIPE_S, ValueRange{op.getDst()},
           ValueRange{op.getSrc(), op.getIndexes()});
  addHiddenEvent(model, PipelineType::PIPE_V, PipelineType::PIPE_S,
                 ArrayRef<unsigned>{0});
  return model;
}

// Byte layout of the A2/A3 compare-form TGATHER scratch as the pinned pto-isa
// implementation derives it from the tmp tile pointer (TGather_cmp in
// include/pto/npu/a2a3/TGather.hpp). Every Rows/Cols in those formulas is the
// EMITTED tile shape (see getEmittedPhysicalType): an 8x128 subview of an
// 8x256 backing emits Rows=8/Cols=256, so the index table spans 8KiB, while a
// single-row 1x32 window compacts to 1x32 and the index starts at tmp+32:
//   bitmap : [tmpBase, + tmpPhysRows*tmpPhysCols) -- one byte per element
//   index  : [bitmapEnd, + srcPhysRows*srcPhysCols*sizeof(dstElem)) --
//            filled linearly on S, read row-wise by the V reduce
//   cvt    : [indexEnd, + srcPhysRows*sizeof(srcElem)) -- typed-K thresholds
//            copied per row on S; PTOAS only emits typed K tiles (K dtype ==
//            src dtype), so the vconv legacy paths are unreachable.
// An emitted shape that cannot be proven (dynamic shapes, non-row-major,
// other producers such as tassign/select/multi-buffer slots) degrades the
// affected regions to unknown-extent instead of guessing a finite extent.
struct TGatherCmpScratchLayout {
    uint64_t bitmapBytes{0};
    bool bitmapKnown{false};
    uint64_t indexOffset{0};
    uint64_t indexBytes{0};
    bool indexKnown{false};
    uint64_t cvtOffset{0};
    uint64_t cvtBytes{0};
    bool cvtKnown{false};
};

static bool hasStaticRank2Shape(ArrayRef<int64_t> shape)
{
    return shape.size() == kTensorRank &&
           !llvm::any_of(shape, [](int64_t dim) { return dim <= 0 || dim == ShapedType::kDynamic; });
}

static bool isStaticRowMajorTile(Type ty)
{
    auto tile = dyn_cast<pto::TileBufType>(ty);
    return tile && hasStaticRank2Shape(tile.getShape()) &&
           tile.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
}

// The EMITTED (physical) tile type of a tgather operand, mirroring the
// lowering's resolveTileHandle(Address) derivation so the sync model counts
// exactly the rows/cols the emitted C++ Tile carries:
//   - alloc_tile emits its own type;
//   - a subview keeps its compact child shape only when the child's
//     addressing matches the inherited strides (getTileSubviewPhysicalType;
//     e.g. a single-row window compacts, an 8x128 window out of an 8x256
//     backing stays 8x256 with the slice in valid);
//   - a treshape repacks and emits its own result type over the source's
//     address.
// Returns a null type when the chain cannot be proven (other producers such
// as tassign/select/multi-buffer slots, dynamic or non-row-major shapes):
// callers must treat the affected regions as unknown-extent.
static pto::TileBufType getEmittedPhysicalType(Value tile)
{
    Operation* def = tile.getDefiningOp();
    if (!def || !isa<pto::TileBufType>(tile.getType())) {
        return pto::TileBufType();
    }
    if (isa<pto::AllocTileOp, pto::TReshapeOp>(def)) {
        return cast<pto::TileBufType>(tile.getType());
    }
    if (auto sub = dyn_cast<pto::SubViewOp>(def)) {
        pto::TileBufType parent = getEmittedPhysicalType(sub.getSource());
        if (!parent) {
            return pto::TileBufType();
        }
        return pto::getTileSubviewPhysicalType(sub, parent);
    }
    return pto::TileBufType();
}

static bool checkedMul(uint64_t a, uint64_t b, uint64_t& result) { return __builtin_mul_overflow(a, b, &result); }

static bool checkedAdd(uint64_t a, uint64_t b, uint64_t& result) { return __builtin_add_overflow(a, b, &result); }

static TGatherCmpScratchLayout computeTGatherCmpScratchLayout(pto::TGatherOp op)
{
    TGatherCmpScratchLayout layout;

    auto dstType = dyn_cast<pto::TileBufType>(op.getDst().getType());
    pto::TileBufType tmpPhys = getEmittedPhysicalType(op.getTmp());
    pto::TileBufType srcPhys = getEmittedPhysicalType(op.getSrc());
    if (!dstType || !tmpPhys || !srcPhys || !isStaticRowMajorTile(tmpPhys) || !isStaticRowMajorTile(srcPhys)) {
        return layout;
    }
    uint64_t srcElem = getPTOStorageElemByteSize(srcPhys.getElementType());
    uint64_t dstElem = getPTOStorageElemByteSize(dstType.getElementType());
    if (srcElem == 0 || dstElem == 0) {
        return layout;
    }

    // Offsets and spans use the EMITTED tile shapes (see getEmittedPhysicalType):
    // every Rows/Cols the pto-isa formulas see -- including the linear
    // indexTmpAddr/cvtTmpAddr offsets -- is the emitted physical shape.
    ArrayRef<int64_t> tmpShape = tmpPhys.getShape();
    ArrayRef<int64_t> srcShape = srcPhys.getShape();
    const uint64_t tmpRows = static_cast<uint64_t>(tmpShape[0]);
    const uint64_t tmpCols = static_cast<uint64_t>(tmpShape[1]);
    const uint64_t srcRows = static_cast<uint64_t>(srcShape[0]);
    const uint64_t srcCols = static_cast<uint64_t>(srcShape[1]);

    uint64_t bitmapBytes = 0;
    uint64_t indexBytes = 0;
    if (!checkedMul(tmpRows, tmpCols, bitmapBytes) && !checkedMul(srcRows, srcCols, indexBytes) &&
        !checkedMul(indexBytes, dstElem, indexBytes) && !checkedMul(srcRows, srcElem, layout.cvtBytes) &&
        !checkedAdd(bitmapBytes, indexBytes, layout.cvtOffset)) {
        layout.bitmapBytes = bitmapBytes;
        layout.bitmapKnown = true;
        layout.indexOffset = bitmapBytes;
        layout.indexBytes = indexBytes;
        layout.indexKnown = true;
        layout.cvtKnown = true;
    }
    return layout;
}

std::optional<SyncMacroModel> getTGatherSyncMacroModel(pto::TGatherOp op)
{
    if (!op.hasCompareForm() || getTargetArch(op.getOperation()) == PTOArch::A5) {
        return std::nullopt;
    }

    Value tmp = op.getTmp();
    Value cdst = op.getCdst();
    if (!tmp || !cdst) {
        return std::nullopt;
    }

    TGatherCmpScratchLayout scratch = computeTGatherCmpScratchLayout(op);
    // A region whose extent cannot be derived degrades to {tmp, 0, 0}: an
    // unknown-size range anchored at the tmp view base, which MemAlias treats
    // as may-alias with everything in the scope instead of guessing a finite
    // interval.
    auto anchored = [&tmp](bool known, uint64_t offset, uint64_t size) {
        return known ? SyncMacroRange{tmp, offset, size} : SyncMacroRange{tmp, 0, 0};
    };

    SyncMacroModel model;
    // A2/A3 compare TGATHER (pto-isa TGather_cmp) executes, in order:
    //   S: the index-fill loop writes the trailing index table, and the typed-K
    //      thresholds are copied per row into the trailing conversion area;
    //      RunTcmps later reads those copied thresholds back on the scalar pipe.
    //   V: TCMPS writes the bitmap (the first tmpRows*tmpCols bytes of tmp)
    //      comparing src against the per-row threshold.
    //   V: vreducev2 reads the bitmap and the index table, writing dst.
    //   S: the reserved-count writeback stores into cdst.
    // The trailing regions have no IR operands, so their accesses ride on
    // anchored byte ranges instead of operand values.
    SyncMacroPhase& prep = addPhase(model, PipelineType::PIPE_S, ValueRange{}, ValueRange{});
    prep.defRanges.push_back(anchored(scratch.indexKnown, scratch.indexOffset, scratch.indexBytes));
    prep.defRanges.push_back(anchored(scratch.cvtKnown, scratch.cvtOffset, scratch.cvtBytes));
    prep.useRanges.push_back(anchored(scratch.cvtKnown, scratch.cvtOffset, scratch.cvtBytes));
    if (op.getKValue() && isa<pto::TileBufType>(op.getKValue().getType())) {
        prep.useValues.push_back(op.getKValue());
    }

    SyncMacroPhase& cmps = addPhase(model, PipelineType::PIPE_V, ValueRange{}, ValueRange{op.getSrc()});
    cmps.defRanges.push_back(anchored(scratch.bitmapKnown, 0, scratch.bitmapBytes));

    SyncMacroPhase& reduce = addPhase(model, PipelineType::PIPE_V, ValueRange{op.getDst()}, ValueRange{});
    reduce.useRanges.push_back(anchored(scratch.bitmapKnown, 0, scratch.bitmapBytes));
    reduce.useRanges.push_back(anchored(scratch.indexKnown, scratch.indexOffset, scratch.indexBytes));

    addPhase(model, PipelineType::PIPE_S, ValueRange{cdst}, ValueRange{});

    // pto-isa pairs the pipes bidirectionally through the fixed EVENT_ID0
    // (PtoSetWaitFlag<PIPE_S, PIPE_V> in FillIndexAndConvertK and
    // PtoSetWaitFlag<PIPE_V, PIPE_S> in RunTcmps), so reserve both directions.
    addHiddenEvent(model, PipelineType::PIPE_S, PipelineType::PIPE_V, ArrayRef<unsigned>{0});
    addHiddenEvent(model, PipelineType::PIPE_V, PipelineType::PIPE_S, ArrayRef<unsigned>{0});
    return model;
}

// Resolve the PTO address space of an MGatherOp tile operand type.
static std::optional<pto::AddressSpace>
getMGatherOperandAddressSpace(::mlir::Type ty) {
  if (auto tb = ::mlir::dyn_cast<::mlir::pto::TileBufType>(ty)) {
    if (auto as = ::mlir::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
            tb.getMemorySpace())) {
      return as.getAddressSpace();
    }
    return std::nullopt;
  }
  return std::nullopt;
}

static std::optional<SmallVector<int64_t>>
getMGatherOperandShapeFromType(::mlir::Type ty, bool useValidShape = true) {
  if (auto tb = ::mlir::dyn_cast<::mlir::pto::TileBufType>(ty)) {
    ArrayRef<int64_t> shape =
        useValidShape ? tb.getValidShape() : tb.getShape();
    return SmallVector<int64_t>(shape.begin(), shape.end());
  }
  return std::nullopt;
}

static std::optional<int64_t> getConstantIndex(Value value) {
  if (!value) {
    return std::nullopt;
  }

  APInt intValue;
  if (!matchPattern(value, m_ConstantInt(&intValue))) {
    return std::nullopt;
  }
  return intValue.getSExtValue();
}

static std::optional<SmallVector<Value, mlir::pto::kValue2>> lookupMGatherValidDims(Value value) {
  if (auto regionResult = dyn_cast<OpResult>(value)) {
    if (auto fusionRegion = dyn_cast<pto::FusionRegionOp>(regionResult.getOwner())) {
      auto yieldOp =
          dyn_cast<pto::YieldOp>(fusionRegion.getBody().front().getTerminator());
      if (!yieldOp) {
        return std::nullopt;
      }
      unsigned resultIndex = regionResult.getResultNumber();
      if (resultIndex >= yieldOp.getNumOperands()) {
        return std::nullopt;
      }
      return lookupMGatherValidDims(yieldOp.getOperand(resultIndex));
    }
  }

  return std::nullopt;
}

static std::optional<SmallVector<int64_t>>
getMGatherOperandShape(Value value, bool useValidShape = true) {
  auto shape = getMGatherOperandShapeFromType(value.getType(), useValidShape);
  if (!shape || !useValidShape) {
    return shape;
  }

  auto validDims = lookupMGatherValidDims(value);
  if (!validDims || validDims->size() != kTensorRank) {
    return shape;
  }

  if (shape->size() >= 1) {
    if (auto validRow = getConstantIndex((*validDims)[0])) {
      (*shape)[0] = *validRow;
    } else if ((*validDims)[0]) {
      (*shape)[0] = ShapedType::kDynamic;
    }
  }
  if (shape->size() >= mlir::pto::kValue2) {
    if (auto validCol = getConstantIndex((*validDims)[1])) {
      (*shape)[1] = *validCol;
    } else if ((*validDims)[1]) {
      (*shape)[1] = ShapedType::kDynamic;
    }
  }
  return shape;
}

// MGather is a macro-like library call whose internal pipe usage depends on the
// destination address space, the coalesce mode, and the target arch. Model each
// reachable lowering path as SyncMacroPhases so InsertSyncAnalysis can derive the
// real cross-pipe sync (notably the MTE2->S wait for the scalar index read that
// MGatherOp::getPipe() hides by reporting a single pipe).
// Also reserve pto-isa's fixed internal event ids through hiddenEvents so
// compiler-generated sync around the macro does not reuse EVENT_ID0 on pipe
// pairs already consumed inside the library implementation.
// P1/P2: GM -> L1 (Row + Elem). S reads GM idx; MTE2 DMAs GM -> L1 into dst.
// Elem additionally writes the GM scratch (owned by S in the template) before
// the MTE2 bulk copy reads it, so model scratch as a S def / MTE2 use.
static SyncMacroModel buildMGatherGm2L1Model(pto::MGatherOp op,
                                             pto::Coalesce coalesce) {
  SyncMacroModel model;
  SmallVector<Value> sDefs;
  if (coalesce == pto::Coalesce::Elem && op.getScratch()) {
    sDefs.push_back(op.getScratch());
  }
  addPhase(model, PipelineType::PIPE_S, ValueRange(sDefs),
           ValueRange{op.getIdx()});
  if (coalesce == pto::Coalesce::Elem && op.getScratch()) {
    SmallVector<Value> mte2Uses;
    mte2Uses.push_back(op.getMem());
    mte2Uses.push_back(op.getScratch());
    addPhase(model, PipelineType::PIPE_MTE2,
             ValueRange{op.getDst()}, ValueRange(mte2Uses));
  } else {
    addPhase(model, PipelineType::PIPE_MTE2,
             ValueRange{op.getDst()}, ValueRange{op.getMem()});
  }
  addHiddenEvent(model, PipelineType::PIPE_S, PipelineType::PIPE_MTE2,
                 ArrayRef<unsigned>{0});
  return model;
}

// P3/P4/P5: A5 lowers GM -> UB through a SIMT kernel on the vector pipe (no
// scalar index loop). The 1x1 scalar overload (P5) reads idx on V then does
// a scalar GM load + dst write on S.
static SyncMacroModel buildMGatherA5Gm2UbModel(pto::MGatherOp op,
                                               pto::Coalesce coalesce) {
  SyncMacroModel model;
  auto dstShape = getMGatherOperandShape(op.getDst());
  const bool isElem1x1 =
      coalesce == pto::Coalesce::Elem && dstShape && dstShape->size() == 2 &&
      (*dstShape)[0] == 1 && (*dstShape)[1] == 1;
  if (isElem1x1) {
    addPhase(model, PipelineType::PIPE_V, ValueRange{},
             ValueRange{op.getIdx()});
    addPhase(model, PipelineType::PIPE_S, ValueRange{op.getDst()},
             ValueRange{op.getMem()});
    addBidirectionalHiddenEvent(model, PipelineType::PIPE_V,
                                PipelineType::PIPE_S,
                                ArrayRef<unsigned>{0});
  } else {
    addPhase(model, PipelineType::PIPE_V, ValueRange{op.getDst()},
             ValueRange{op.getMem(), op.getIdx()});
  }
  return model;
}

// A2/A3 GM -> UB. Row (P6): S reads the UB index tile to compute GM addresses,
// then MTE2 DMAs each gathered row GM -> UB. Elem (P7): the whole gather runs
// on the scalar pipe (scalar GM loads + scalar UB dst writes), single phase.
static SyncMacroModel buildMGatherA2A3Gm2UbModel(pto::MGatherOp op,
                                                 pto::Coalesce coalesce) {
  SyncMacroModel model;
  if (coalesce == pto::Coalesce::Row) {
    addPhase(model, PipelineType::PIPE_S, ValueRange{},
             ValueRange{op.getIdx()});
    addPhase(model, PipelineType::PIPE_MTE2, ValueRange{op.getDst()},
             ValueRange{op.getMem()});
    // Leading index-readiness waits (V/MTE3 -> S) and the internal S -> MTE2
    // handshake stay inside the macro in both sync modes.
    addHiddenEvent(model, PipelineType::PIPE_V, PipelineType::PIPE_S,
                   ArrayRef<unsigned>{0});
    addHiddenEvent(model, PipelineType::PIPE_MTE3, PipelineType::PIPE_S,
                   ArrayRef<unsigned>{0});
    addHiddenEvent(model, PipelineType::PIPE_S, PipelineType::PIPE_MTE2,
                   ArrayRef<unsigned>{0});
    // The default macro additionally drains the whole row-DMA loop on the
    // Vector and MTE3 queues (MTE2/S -> V/MTE3). The async variant
    // (GatherSync::Async, issue #1674 / pto-isa #380) drops those completion
    // fences: dst completion is emitted by InsertSync at the actual consumers
    // (the MTE2: def dst phase above), matching the pto-isa expansion, which
    // only fences S -> MTE2 in async mode. Gated on the IR mark the sync
    // passes set, so skipped functions (hand-written sync) model the Default
    // macro and stay in sync with what EmitC actually emits.
    if (!hasAsyncA2A3RowMGatherMark(op.getOperation())) {
      addHiddenEvent(model, PipelineType::PIPE_MTE2, PipelineType::PIPE_V,
                     ArrayRef<unsigned>{0});
      addHiddenEvent(model, PipelineType::PIPE_MTE2, PipelineType::PIPE_MTE3,
                     ArrayRef<unsigned>{0});
      addHiddenEvent(model, PipelineType::PIPE_S, PipelineType::PIPE_V,
                     ArrayRef<unsigned>{0});
      addHiddenEvent(model, PipelineType::PIPE_S, PipelineType::PIPE_MTE3,
                     ArrayRef<unsigned>{0});
    }
  } else {
    SmallVector<Value> sUses;
    sUses.push_back(op.getIdx());
    sUses.push_back(op.getMem());
    addPhase(model, PipelineType::PIPE_S, ValueRange{op.getDst()},
             ValueRange(sUses));
    addHiddenEvent(model, PipelineType::PIPE_V, PipelineType::PIPE_S,
                   ArrayRef<unsigned>{0});
    addHiddenEvent(model, PipelineType::PIPE_MTE3, PipelineType::PIPE_S,
                   ArrayRef<unsigned>{0});
    addHiddenEvent(model, PipelineType::PIPE_MTE2, PipelineType::PIPE_S,
                   ArrayRef<unsigned>{0});
    addHiddenEvent(model, PipelineType::PIPE_S, PipelineType::PIPE_V,
                   ArrayRef<unsigned>{0});
    addHiddenEvent(model, PipelineType::PIPE_S, PipelineType::PIPE_MTE2,
                   ArrayRef<unsigned>{0});
    addHiddenEvent(model, PipelineType::PIPE_S, PipelineType::PIPE_MTE3,
                   ArrayRef<unsigned>{0});
  }
  return model;
}

std::optional<SyncMacroModel> getMGatherSyncMacroModel(pto::MGatherOp op) {
  // GM -> L1 (dst is an L1 / cube MAT tile). A5/A2A3 share the same data flow:
  // the scalar pipe reads the GM index to compute src rows, then MTE2 issues the
  // GM -> L1 nd2nz DMA. pto-isa currently keeps an internal fixed EVENT_ID0 on
  // S -> MTE2, so reserve that pair here.
  auto dstSpace = getMGatherOperandAddressSpace(op.getDst().getType());
  const bool isGm2L1 = dstSpace && *dstSpace == pto::AddressSpace::MAT;

  auto coalesceAttr = op.getCoalesceAttr();
  if (!coalesceAttr) {
    return std::nullopt;
  }

  const PTOArch arch = getTargetArch(op.getOperation());
  const pto::Coalesce coalesce = coalesceAttr.getValue();

  if (isGm2L1) {
    return buildMGatherGm2L1Model(op, coalesce);
  }
  if (arch == PTOArch::A5) {
    return buildMGatherA5Gm2UbModel(op, coalesce);
  }
  return buildMGatherA2A3Gm2UbModel(op, coalesce);
}

} // namespace

bool mlir::pto::useAsyncA2A3RowMGather(pto::MGatherOp op) {
  // The pto-isa async row-gather overload (GatherSync::Async) only exists for
  // the A2/A3 GM -> UB ND path with the valid-index contract: Coalesce::Row,
  // GatherOOB::Undefined (or the elided default), and a non-MAT destination.
  // Its static_asserts additionally reject NZ tables and every other mode.
  // Pure op-level eligibility: actual use additionally requires the
  // kMGatherAsyncRowAttrName mark set by a sync pass that ran on the
  // enclosing function (see markAsyncA2A3RowMGathers).
  //
  // Note the three-parameter async overload is currently NPU-arch guarded in
  // pto-isa, so CPU-simulation builds of async output need the pto-isa CPU
  // forwarding support first.
  auto coalesceAttr = op.getCoalesceAttr();
  if (!coalesceAttr || coalesceAttr.getValue() != pto::Coalesce::Row) {
    return false;
  }
  auto gatherOobAttr =
      dyn_cast_or_null<pto::GatherOOBAttr>(op.getProperties().gatherOob);
  if (gatherOobAttr && gatherOobAttr.getValue() != pto::GatherOOB::Undefined) {
    return false;
  }
  if (getTargetArch(op.getOperation()) != PTOArch::A3) {
    return false;
  }
  auto dstSpace = getMGatherOperandAddressSpace(op.getDst().getType());
  const bool isGm2L1 = dstSpace && *dstSpace == pto::AddressSpace::MAT;
  return !isGm2L1;
}

std::optional<SyncMacroModel> mlir::pto::getSyncMacroModel(Operation *op) {
  if (auto model = getP2PCommSyncMacroModel(op)) {
    return model;
  }
  if (auto model = getCollectiveCommSyncMacroModel(op)) {
    return model;
  }
  if (auto tscatter = dyn_cast<pto::TScatterOp>(op)) {
    return getTScatterSyncMacroModel(tscatter);
  }
  if (auto tgather = dyn_cast<pto::TGatherOp>(op)) {
    return getTGatherSyncMacroModel(tgather);
  }
  if (auto mgather = dyn_cast<pto::MGatherOp>(op)) {
    return getMGatherSyncMacroModel(mgather);
  }
  return std::nullopt;
}

std::optional<std::pair<unsigned, unsigned>> mlir::pto::getSyncMacroSpan(const SyncIRs &syncIR,
                                                           std::size_t firstIndex) {
  if (firstIndex >= syncIR.size()) {
    return std::nullopt;
  }
  auto *first = dyn_cast<CompoundInstanceElement>(syncIR[firstIndex].get());
  if (!first || !first->elementOp) {
    return std::nullopt;
  }
  unsigned end = first->GetIndex() + 1;
  for (size_t j = firstIndex + 1; j < syncIR.size(); ++j) {
    auto *other = dyn_cast<pto::CompoundInstanceElement>(syncIR[j].get());
    if (!other || other->elementOp != first->elementOp) {
      continue;
    }
    end = other->GetIndex();
  }
  unsigned begin = first->GetIndex();
  if (begin > 0) {
    --begin;
  }
  if (end + 1 < syncIR.size()) {
    ++end;
  }
  if (begin >= end) {
    return std::nullopt;
  }
  return std::make_pair(begin, end);
}

void mlir::pto::markAsyncA2A3RowMGathers(func::FuncOp func) {
  func.walk([](pto::MGatherOp op) {
    if (!useAsyncA2A3RowMGather(op)) {
      return;
    }
    // The function-level dependency analysis only models calls into
    // sync-annotated helpers (UpdateHelperCallInfo); a gather whose
    // destination escapes into an ordinary func.call has consumers the
    // solver cannot see, so it must keep the Default macro whose internal
    // fences order those consumers (review P1).
    for (Operation *user : op.getDst().getUsers()) {
      if (isa<func::CallOp>(user)) {
        return;
      }
    }
    op->setAttr(kMGatherAsyncRowAttrName, UnitAttr::get(op.getContext()));
  });
}

bool mlir::pto::hasAsyncA2A3RowMGatherMark(Operation *op) {
  // Re-verify eligibility instead of trusting the attribute alone: a forged
  // or stale mark on a clamp/A5/Elem/GM->L1 gather must not lower it to the
  // Undefined/Async overload.
  auto mgather = dyn_cast<pto::MGatherOp>(op);
  return mgather && mgather->hasAttr(kMGatherAsyncRowAttrName) &&
         useAsyncA2A3RowMGather(mgather);
}
