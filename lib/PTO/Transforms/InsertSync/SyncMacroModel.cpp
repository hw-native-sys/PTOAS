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

void addPhase(SyncMacroModel &model, PipelineType pipe, ValueRange defValues,
              ValueRange useValues) {
  model.phases.push_back(SyncMacroPhase{
      static_cast<unsigned>(model.phases.size()), pipe,
      SmallVector<Value>(defValues.begin(), defValues.end()),
      SmallVector<Value>(useValues.begin(), useValues.end())});
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

std::optional<SyncMacroModel> getCollectiveCommSyncMacroModel(Operation *op) {
  SyncMacroModel model;
  unsigned laneCount = 1;

  if (auto tgather = dyn_cast<pto::CommTGatherOp>(op)) {
    laneCount = tgather.getPong() ? 2U : 1U;
    // TGATHER_IMPL reads each group source through MTE2 and writes the gathered
    // result into dst through MTE3, using ping/pong as staging tiles.
    SmallVector<Value> staging =
        getPingPongValues(tgather.getPing(), tgather.getPong());
    addPhase(model, PipelineType::PIPE_MTE2, staging,
             tgather.getGroup());
    addPhase(model, PipelineType::PIPE_MTE3, ValueRange{tgather.getDst()},
             staging);
  } else if (auto tscatter = dyn_cast<pto::CommTScatterOp>(op)) {
    laneCount = tscatter.getPong() ? 2U : 1U;
    // TSCATTER_IMPL reads the source through MTE2 and writes every group
    // destination through MTE3, using ping/pong as staging tiles.
    SmallVector<Value> staging =
        getPingPongValues(tscatter.getPing(), tscatter.getPong());
    addPhase(model, PipelineType::PIPE_MTE2, staging,
             ValueRange{tscatter.getSrc()});
    addPhase(model, PipelineType::PIPE_MTE3, tscatter.getGroup(),
             staging);
  } else if (auto tbroadcast = dyn_cast<pto::TBroadcastOp>(op)) {
    laneCount = tbroadcast.getPong() ? 2U : 1U;
    // TBROADCAST_IMPL reads the source through MTE2 and writes every group
    // destination through MTE3, using ping/pong as staging tiles.
    SmallVector<Value> staging =
        getPingPongValues(tbroadcast.getPing(), tbroadcast.getPong());
    addPhase(model, PipelineType::PIPE_MTE2, staging,
             ValueRange{tbroadcast.getSrc()});
    addPhase(model, PipelineType::PIPE_MTE3, tbroadcast.getGroup(),
             staging);
  } else if (auto treduce = dyn_cast<pto::TReduceOp>(op)) {
    laneCount = treduce.getRecvPong() ? 3U : 2U;
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
  } else {
    return std::nullopt;
  }

  SmallVector<unsigned> eventIds = getSequentialEventIds(laneCount);
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

std::optional<SyncMacroModel> getTGatherSyncMacroModel(pto::TGatherOp op) {
  if (!op.hasCompareForm() || getTargetArch(op.getOperation()) == PTOArch::A5) {
    return std::nullopt;
  }

  Value tmp = op.getTmp();
  Value cdst = op.getCdst();
  if (!tmp || !cdst) {
    return std::nullopt;
  }

  SyncMacroModel model;
  // A2/A3 compare TGATHER lowers to TCMPS on PIPE_V, an index generation loop
  // on PIPE_S, a V reduce over the generated indexes, and a scalar writeback of
  // the reserved count into cdst.
  addPhase(model, PipelineType::PIPE_V, ValueRange{tmp},
           ValueRange{op.getSrc()});
  addPhase(model, PipelineType::PIPE_S, ValueRange{tmp}, ValueRange{});
  addPhase(model, PipelineType::PIPE_V, ValueRange{op.getDst()},
           ValueRange{tmp});
  addPhase(model, PipelineType::PIPE_S, ValueRange{cdst}, ValueRange{});
  addHiddenEvent(model, PipelineType::PIPE_S, PipelineType::PIPE_V,
                 ArrayRef<unsigned>{0});
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

  SyncMacroModel model;

  if (isGm2L1) {
    // P1/P2: GM -> L1 (Row + Elem). S reads GM idx; MTE2 DMAs GM -> L1 into dst.
    // Elem additionally writes the GM scratch (owned by S in the template) before
    // the MTE2 bulk copy reads it, so model scratch as a S def / MTE2 use.
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

  // GM -> UB (dst is a VEC tile).
  if (arch == PTOArch::A5) {
    // P3/P4: A5 lowers GM -> UB through a SIMT kernel on the vector pipe (no
    // scalar index loop). The 1x1 scalar overload (P5) reads idx on V then does
    // a scalar GM load + dst write on S.
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
  if (coalesce == pto::Coalesce::Row) {
    addPhase(model, PipelineType::PIPE_S, ValueRange{},
             ValueRange{op.getIdx()});
    addPhase(model, PipelineType::PIPE_MTE2, ValueRange{op.getDst()},
             ValueRange{op.getMem()});
    addHiddenEvent(model, PipelineType::PIPE_V, PipelineType::PIPE_S,
                   ArrayRef<unsigned>{0});
    addHiddenEvent(model, PipelineType::PIPE_MTE3, PipelineType::PIPE_S,
                   ArrayRef<unsigned>{0});
    addHiddenEvent(model, PipelineType::PIPE_S, PipelineType::PIPE_MTE2,
                   ArrayRef<unsigned>{0});
    addHiddenEvent(model, PipelineType::PIPE_MTE2, PipelineType::PIPE_V,
                   ArrayRef<unsigned>{0});
    addHiddenEvent(model, PipelineType::PIPE_MTE2, PipelineType::PIPE_MTE3,
                   ArrayRef<unsigned>{0});
    addHiddenEvent(model, PipelineType::PIPE_S, PipelineType::PIPE_V,
                   ArrayRef<unsigned>{0});
    addHiddenEvent(model, PipelineType::PIPE_S, PipelineType::PIPE_MTE3,
                   ArrayRef<unsigned>{0});
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

} // namespace

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
