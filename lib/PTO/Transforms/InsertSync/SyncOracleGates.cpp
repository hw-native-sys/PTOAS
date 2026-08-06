// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- SyncOracleGates.cpp - Oracle correctness gates ---------------------===//
//
// Implements G3 (device-id legality), G2 (non-interference), the happens-before
// edge model, and G1-self (dependency coverage), plus the three read-only passes
// that run them on a compiled function.
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/InsertSync/SyncOracleGates.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOSyncUtils.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "llvm/ADT/DenseMap.h"
#include <tuple>
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Format.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <map>
#include <set>

namespace mlir {
namespace pto {
namespace func = ::mlir::func;

#define GEN_PASS_DEF_PTOCHECKSYNCIDS
#define GEN_PASS_DEF_PTOCHECKSYNCSELFCOVERAGE
#define GEN_PASS_DEF_PTOCHECKSYNCINTERFERENCE
#define GEN_PASS_DEF_PTODUMPSYNCCOVERAGE
#define GEN_PASS_DEF_PTOCHECKSYNCCOVERAGE
#define GEN_PASS_DEF_PTODUMPSYNCCOUNTS
#define GEN_PASS_DEF_PTOCHECKSYNCCOUNTFLOOR
#include "PTO/Transforms/Passes.h.inc"

} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::oracle;

llvm::StringRef mlir::pto::oracle::idViolationKindName(IdViolationKind kind) {
  switch (kind) {
  case IdViolationKind::EventIdOutOfRange:
    return "event-id-out-of-range";
  case IdViolationKind::EventIdReservedForDirection:
    return "event-id-reserved-for-direction";
  case IdViolationKind::EventIdReservedForBlockAll:
    return "event-id-reserved-for-block-all";
  case IdViolationKind::BlockAllEventIdWrong:
    return "block-all-event-id-wrong";
  case IdViolationKind::BarrierCarriesEventId:
    return "barrier-carries-event-id";
  case IdViolationKind::BufIdOutOfRange:
    return "buf-id-out-of-range";
  case IdViolationKind::BufIdExceedsArchCapacity:
    return "buf-id-exceeds-arch-capacity";
  case IdViolationKind::EventIdCollidesWithMacroHidden:
    return "event-id-collides-with-macro-hidden";
  case IdViolationKind::EventSetWaitUnbalanced:
    return "event-set-wait-unbalanced";
  }
  return "?";
}

namespace {

std::string pipeDir(PipelineType src, PipelineType dst) {
  return (oracle::pipelineTypeName(src) + "->" + oracle::pipelineTypeName(dst))
      .str();
}

/// R1 + R2: an intra-core set/wait event id.
void checkIntraCoreEventId(int64_t id, PipelineType src, PipelineType dst,
                           unsigned poolSize, unsigned order,
                           llvm::StringRef opName,
                           llvm::SmallVectorImpl<IdViolation> &out) {
  if (id < 0 || id >= static_cast<int64_t>(poolSize)) {
    out.push_back({IdViolationKind::EventIdOutOfRange, order, opName.str(), id,
                   ("intra-core event pool for " + pipeDir(src, dst) + " is [0, " +
                    std::to_string(poolSize) + ")")});
    return;
  }
  uint64_t reserved = SyncEventIdAllocation::GetReservedEventIdNum(src, dst);
  if (reserved > 0 && id >= static_cast<int64_t>(poolSize - reserved)) {
    out.push_back(
        {IdViolationKind::EventIdReservedForDirection, order, opName.str(), id,
         (pipeDir(src, dst) + " reserves its top " + std::to_string(reserved) +
          " id(s); available is [0, " + std::to_string(poolSize - reserved) +
          ")")});
  }
}

/// R4 + R5: a buffer id.
void checkBufId(int64_t id, unsigned order, llvm::StringRef opName,
                const DeviceIdLimits &limits,
                llvm::SmallVectorImpl<IdViolation> &out) {
  if (id < 0 || id >= static_cast<int64_t>(limits.bufIdHwMax)) {
    out.push_back({IdViolationKind::BufIdOutOfRange, order, opName.str(), id,
                   ("hardware buf_id field is [0, " +
                    std::to_string(limits.bufIdHwMax) + ")")});
    return;
  }
  if (id >= static_cast<int64_t>(limits.bufIdCapacity)) {
    out.push_back({IdViolationKind::BufIdExceedsArchCapacity, order,
                   opName.str(), id,
                   ("this compilation exposes K=" +
                    std::to_string(limits.bufIdCapacity) +
                    " buffer id(s); buffer-id sync needs K > 0 (A5)")});
  }
}

} // namespace

llvm::SmallVector<IdViolation>
mlir::pto::oracle::checkDeviceIdLegality(llvm::ArrayRef<IRSyncRecord> records,
                                         const DeviceIdLimits &limits) {
  llvm::SmallVector<IdViolation> violations;

  for (const IRSyncRecord &rec : records) {
    if (rec.opName == "pto.set_flag" || rec.opName == "pto.wait_flag") {
      checkIntraCoreEventId(rec.eventId, rec.srcPipeType, rec.dstPipeType,
                            limits.intraCoreEventIdNum, rec.order, rec.opName,
                            violations);
    } else if (rec.opName == "pto.get_buf" || rec.opName == "pto.rls_buf") {
      checkBufId(rec.bufId, rec.order, rec.opName, limits, violations);
    } else if (rec.opName == "pto.barrier" && rec.eventId >= 0) {
      violations.push_back({IdViolationKind::BarrierCarriesEventId, rec.order,
                            rec.opName, rec.eventId,
                            "a pipe barrier takes no event id"});
    }
  }

  // Every (direction, id) must carry as many sets as waits. A shortfall either way
  // is a hang: a wait with no set never retires, and a set with no wait leaves the
  // flag raised for whoever takes the id next.
  //
  // Rotating pairs are excluded, not overlooked: `set_flag_dyn`/`wait_flag_dyn`
    // choose their id at runtime and it is recorded as -1, so a rotating set cannot be
    // paired with its wait. Nothing here checks a rotating id.
  llvm::DenseMap<std::tuple<int, int, int64_t>, std::pair<unsigned, unsigned>> tally;
  llvm::DenseMap<std::tuple<int, int, int64_t>, unsigned> firstOrder;
  for (const IRSyncRecord &rec : records) {
    bool isSet = rec.opName == "pto.set_flag";
    bool isWait = rec.opName == "pto.wait_flag";
    if ((!isSet && !isWait) || rec.eventId < 0)
      continue;
    auto key = std::make_tuple(static_cast<int>(rec.srcPipeType),
                               static_cast<int>(rec.dstPipeType), rec.eventId);
    auto &slot = tally[key];
    (isSet ? slot.first : slot.second)++;
    if (!firstOrder.count(key))
      firstOrder[key] = rec.order;
  }
  llvm::SmallVector<std::tuple<int, int, int64_t>> keys;
  for (const auto &kv : tally)
    keys.push_back(kv.first);
  llvm::sort(keys);
  for (const auto &key : keys) {
    auto [sets, waits] = tally[key];
    if (sets == waits)
      continue;
    violations.push_back(
        {IdViolationKind::EventSetWaitUnbalanced, firstOrder[key], "pto.set_flag",
         static_cast<int>(std::get<2>(key)),
         ("direction " + std::to_string(std::get<0>(key)) + "->" +
          std::to_string(std::get<1>(key)) + " carries " + std::to_string(sets) +
          " set_flag and " + std::to_string(waits) + " wait_flag")});
  }

  return violations;
}

llvm::SmallVector<IdViolation>
mlir::pto::oracle::checkDeviceIdLegality(llvm::ArrayRef<SyncOpRecord> records,
                                         const DeviceIdLimits &limits) {
  using TYPE = SyncOperation::TYPE;
  llvm::SmallVector<IdViolation> violations;

  // The block-sync pool only shrinks when a SYNC_BLOCK_ALL is actually present;
  // mirrors SyncEventIdAllocation::reserveBlockAllEventIds().
  bool blockAllPresent = llvm::any_of(records, [](const SyncOpRecord &rec) {
    return rec.typeCode == static_cast<int>(TYPE::SYNC_BLOCK_ALL);
  });
  const unsigned blockPool =
      limits.blockSyncEventIdNum -
      (blockAllPresent ? kReservedBlockSyncEventIdNum : 0u);

  for (const SyncOpRecord &rec : records) {
    auto src = static_cast<PipelineType>(rec.srcPipe);
    auto dst = static_cast<PipelineType>(rec.dstPipe);

    switch (static_cast<TYPE>(rec.typeCode)) {
    case TYPE::SET_EVENT:
    case TYPE::WAIT_EVENT:
        // EVERY id a hazard holds is checked, not just the first. A rotating hazard
        // carries several, and its emitted form (`set_flag_dyn`/`wait_flag_dyn`)
        // resolves the id at runtime, so once emitted the id cannot be checked at
        // all -- these pre-codegen records are the only place the full set is visible.
      // Weakening or narrowing this loop --
      // e.g. back to `eventIds[0]` -- would not fail any test: it would make that
      // blind spot LIVE SILENTLY, because no other view can catch an illegal id
      // on a rotating pair.
      //
        // Iterating the FULL set is the point: nothing else validates these ids, so
        // narrowing it to `eventIds[0]` would silently stop checking the rest.
      for (int id : rec.eventIds)
        checkIntraCoreEventId(id, src, dst, limits.intraCoreEventIdNum,
                              rec.syncIndex, rec.type, violations);
      break;

    case TYPE::SYNC_BLOCK_SET:
    case TYPE::SYNC_BLOCK_WAIT:
      for (int id : rec.eventIds) {
        if (blockAllPresent && (id == static_cast<int>(kBlockSyncAllCubeEventId) ||
                                id == static_cast<int>(kBlockSyncAllVectorEventId))) {
          violations.push_back(
              {IdViolationKind::EventIdReservedForBlockAll, rec.syncIndex,
               rec.type, id,
               ("ids " + std::to_string(kBlockSyncAllCubeEventId) + "/" +
                std::to_string(kBlockSyncAllVectorEventId) +
                " belong to sync_block_all while one is live")});
          continue;
        }
        if (id < 0 || id >= static_cast<int64_t>(blockPool))
          violations.push_back({IdViolationKind::EventIdOutOfRange,
                                rec.syncIndex, rec.type, id,
                                ("block-sync pool is [0, " +
                                 std::to_string(blockPool) + ")")});
      }
      break;

    case TYPE::SYNC_BLOCK_ALL:
      for (int id : rec.eventIds)
        if (id != static_cast<int>(kBlockSyncAllCubeEventId) &&
            id != static_cast<int>(kBlockSyncAllVectorEventId))
          violations.push_back(
              {IdViolationKind::BlockAllEventIdWrong, rec.syncIndex, rec.type,
               id,
               ("sync_block_all must use " +
                std::to_string(kBlockSyncAllCubeEventId) + " (cube) or " +
                std::to_string(kBlockSyncAllVectorEventId) + " (vector)")});
      break;

    case TYPE::PIPE_BARRIER:
    case TYPE::PIPE_BARRIER_CUBE:
    case TYPE::PIPE_BARRIER_VECTOR:
      for (int id : rec.eventIds)
        violations.push_back({IdViolationKind::BarrierCarriesEventId,
                              rec.syncIndex, rec.type, id,
                              "a pipe barrier takes no event id"});
      break;
    }
  }

  return violations;
}

llvm::SmallVector<IdViolation>
mlir::pto::oracle::checkMacroHiddenEventCollisions(
    const SyncIRs &syncIR, llvm::ArrayRef<SyncOpRecord> records) {
  llvm::SmallVector<IdViolation> violations;

  // One reservation per (direction, id) a macro's library implementation uses,
  // over the macro's own span. Built to the same rule the incumbent reserves by,
  // so that a green result here means what a green result there means.
  struct Reservation {
    int srcPipe, dstPipe, id;
    unsigned begin, end;
    std::string macroName;
  };
  llvm::SmallVector<Reservation> reserved;

  for (size_t i = 0; i < syncIR.size(); ++i) {
    auto *first = dyn_cast<pto::CompoundInstanceElement>(syncIR[i].get());
    if (!first || first->macroOpInstanceId != 0 || !first->elementOp)
      continue;
    auto model = pto::getSyncMacroModel(first->elementOp);
    if (!model || model->hiddenEvents.empty())
      continue;

    // The span is first phase -> last phase of the SAME op, then padded one index
    // each side. Without the pad, a hazard ending exactly where the call begins
    // reads as disjoint from it and the collision is missed.
    unsigned end = first->GetIndex() + 1;
    for (size_t j = i + 1; j < syncIR.size(); ++j) {
      auto *other = dyn_cast<pto::CompoundInstanceElement>(syncIR[j].get());
      if (!other || other->elementOp != first->elementOp)
        continue;
      end = other->GetIndex();
    }
    unsigned begin = first->GetIndex();
    if (begin > 0)
      --begin;
    if (end + 1 < syncIR.size())
      ++end;
    if (begin >= end)
      continue;

    for (const auto &hidden : model->hiddenEvents)
      for (unsigned id : hidden.eventIds)
        reserved.push_back({int(hidden.srcPipe), int(hidden.dstPipe), int(id),
                            begin, end,
                            first->elementOp->getName().getStringRef().str()});
  }
  if (reserved.empty())
    return violations;

  // An allocated hazard occupies the interval spanned by its own records: the set
  // and the wait share a `syncIndex`, so the group's min/max irIndex is its live
  // range. Barriers carry no id and cannot collide.
  struct Live {
    int srcPipe = -1, dstPipe = -1;
    unsigned lo = ~0u, hi = 0;
    llvm::SmallVector<int, 2> ids;
  };
  llvm::MapVector<unsigned, Live> byHazard;
  for (const SyncOpRecord &r : records) {
    if (r.eventIds.empty())
      continue;
    Live &L = byHazard[r.syncIndex];
    L.srcPipe = r.srcPipe;
    L.dstPipe = r.dstPipe;
    L.lo = std::min(L.lo, r.irIndex);
    L.hi = std::max(L.hi, r.irIndex);
    for (int id : r.eventIds)
      if (!llvm::is_contained(L.ids, id))
        L.ids.push_back(id);
  }

  for (const auto &entry : byHazard) {
    const Live &L = entry.second;
    for (const Reservation &R : reserved) {
      if (L.srcPipe != R.srcPipe || L.dstPipe != R.dstPipe)
        continue;
      if (L.hi < R.begin || R.end < L.lo)
        continue; // disjoint spans
      for (int id : L.ids) {
        if (id != R.id)
          continue;
        violations.push_back(
            {IdViolationKind::EventIdCollidesWithMacroHidden, entry.first,
             "set/wait", id,
             ("event id " + std::to_string(id) + " on direction " +
              std::to_string(R.srcPipe) + "->" + std::to_string(R.dstPipe) +
              " is reserved by `" + R.macroName + "` over SyncIR [" +
              std::to_string(R.begin) + ", " + std::to_string(R.end) +
              "]; this hazard holds it over [" + std::to_string(L.lo) + ", " +
              std::to_string(L.hi) +
              "], so the library's own wait can be satisfied by this producer")});
      }
    }
  }
  return violations;
}


void mlir::pto::oracle::printIdViolations(llvm::raw_ostream &os,
                                          llvm::StringRef funcName,
                                          llvm::StringRef view,
                                          size_t recordCount,
                                          llvm::ArrayRef<IdViolation> violations) {
  os << "[sync-gate:g3] func=" << funcName << " view=" << view
     << " records=" << recordCount << " violations=" << violations.size()
     << (violations.empty() ? " OK" : "") << "\n";
  for (const IdViolation &v : violations) {
    os << "  !! #" << v.order << " " << v.opName
       << " kind=" << idViolationKindName(v.kind) << " id=" << v.id << " : "
       << v.detail << "\n";
  }
}

//===----------------------------------------------------------------------===//
// G4: sync-count floor
//===----------------------------------------------------------------------===//

llvm::StringRef mlir::pto::oracle::countViolationKindName(CountViolationKind k) {
  switch (k) {
  case CountViolationKind::TotalIncreased:
    return "total-sync-count-increased";
  case CountViolationKind::BarrierIncreased:
    return "barrier-count-increased";
  case CountViolationKind::BarrierAllIncreased:
    return "barrier-all-count-increased";
  case CountViolationKind::MissingReference:
    return "missing-reference-profile";
  }
  return "?";
}

SyncCountProfile
mlir::pto::oracle::computeSyncCounts(llvm::ArrayRef<IRSyncRecord> records) {
  SyncCountProfile p;
  for (const IRSyncRecord &rec : records) {
    // Rotation's dyn ops count toward the SAME classes as their static
    // counterparts -- they are set/wait pairs whose id is chosen at runtime. They
    // were previously counted as NOTHING, so G4 undercounted every rotating
    // kernel and an allocator that replaced static pairs with dyn pairs would
    // have looked like it REDUCED the op count.
    if (rec.opName == "pto.set_flag" || rec.opName == "pto.set_flag_dyn")
      ++p.setFlag;
    else if (rec.opName == "pto.wait_flag" || rec.opName == "pto.wait_flag_dyn")
      ++p.waitFlag;
    else if (rec.opName == "pto.get_buf")
      ++p.getBuf;
    else if (rec.opName == "pto.rls_buf")
      ++p.rlsBuf;
    else if (rec.opName == "pto.barrier") {
      ++p.barrier;
      if (rec.srcPipeType == PipelineType::PIPE_ALL)
        ++p.barrierAll;
    }
  }
  return p;
}

llvm::SmallVector<CountViolation>
mlir::pto::oracle::checkSyncCountFloor(const SyncCountProfile &cand,
                                       const SyncCountProfile &ref) {
  llvm::SmallVector<CountViolation> violations;

  if (cand.total() > ref.total())
    violations.push_back({CountViolationKind::TotalIncreased, cand.total(),
                          ref.total(),
                          "the new mode emits more sync ops than InsertSync"});

  // Load-bearing: one barrier can replace a set/wait pair, lowering the total
  // while strictly worsening the schedule. On vadd the degenerate ties on total.
  if (cand.barrier > ref.barrier)
    violations.push_back(
        {CountViolationKind::BarrierIncreased, cand.barrier, ref.barrier,
         "a barrier orders every pipe; a set/wait pair orders one direction"});

  if (cand.barrierAll > ref.barrierAll)
    violations.push_back({CountViolationKind::BarrierAllIncreased,
                          cand.barrierAll, ref.barrierAll,
                          "PIPE_ALL barriers fully serialize the core"});

  return violations;
}

void mlir::pto::oracle::printSyncCounts(llvm::raw_ostream &os,
                                        llvm::StringRef funcName,
                                        const SyncCountProfile &p) {
  os << "[sync-count] func=" << funcName << " set=" << p.setFlag
     << " wait=" << p.waitFlag << " get=" << p.getBuf << " rls=" << p.rlsBuf
     << " barrier=" << p.barrier << " barrier_all=" << p.barrierAll
     << " total=" << p.total() << "\n";
}

llvm::StringMap<SyncCountProfile>
mlir::pto::oracle::parseSyncCounts(llvm::StringRef text) {
  llvm::StringMap<SyncCountProfile> profiles;

  llvm::SmallVector<llvm::StringRef> lines;
  text.split(lines, '\n');
  for (llvm::StringRef line : lines) {
    line = line.trim();
    if (!line.consume_front("[sync-count]"))
      continue;

    llvm::StringRef name;
    SyncCountProfile p;
    llvm::SmallVector<llvm::StringRef> fields;
    line.split(fields, ' ', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (llvm::StringRef field : fields) {
      auto [key, value] = field.split('=');
      if (key == "func") {
        name = value;
        continue;
      }
      unsigned parsed = 0;
      if (value.getAsInteger(10, parsed))
        continue;
      if (key == "set")
        p.setFlag = parsed;
      else if (key == "wait")
        p.waitFlag = parsed;
      else if (key == "get")
        p.getBuf = parsed;
      else if (key == "rls")
        p.rlsBuf = parsed;
      else if (key == "barrier")
        p.barrier = parsed;
      else if (key == "barrier_all")
        p.barrierAll = parsed;
      // `total` is derived; ignore it on read so a hand-edited file cannot lie.
    }
    if (!name.empty())
      profiles[name] = p;
  }

  return profiles;
}

void mlir::pto::oracle::printCountViolations(
    llvm::raw_ostream &os, llvm::StringRef funcName,
    const SyncCountProfile &cand, const SyncCountProfile &ref,
    llvm::ArrayRef<CountViolation> violations) {
  os << "[sync-gate:g4] func=" << funcName << " candidate_total=" << cand.total()
     << " reference_total=" << ref.total()
     << " violations=" << violations.size() << (violations.empty() ? " OK" : "")
     << "\n";
  os << "  candidate: ";
  printSyncCounts(os, funcName, cand);
  os << "  reference: ";
  printSyncCounts(os, funcName, ref);
  for (const CountViolation &v : violations) {
    os << "  !! kind=" << countViolationKindName(v.kind)
       << " candidate=" << v.candidate << " reference=" << v.reference << " : "
       << v.detail << "\n";
  }
}

//===----------------------------------------------------------------------===//
// G2: non-interference
//===----------------------------------------------------------------------===//

llvm::StringRef mlir::pto::oracle::interferenceKindName(InterferenceKind k) {
  switch (k) {
  case InterferenceKind::EventIdNested:
    return "event-id-nested";
  case InterferenceKind::EventIdWaitWithoutSet:
    return "event-id-wait-without-set";
  case InterferenceKind::EventIdUnclosed:
    return "event-id-unclosed";
  case InterferenceKind::EventIdPairGuardMismatch:
    return "event-id-pair-guard-mismatch";
  case InterferenceKind::BufIdNested:
    return "buf-id-nested";
  case InterferenceKind::BufIdReleaseWithoutAcquire:
    return "buf-id-release-without-acquire";
  case InterferenceKind::BufIdUnclosed:
    return "buf-id-unclosed";
  case InterferenceKind::BufIdPipeMismatch:
    return "buf-id-pipe-mismatch";
  case InterferenceKind::BufIdSinglePipe:
    return "buf-id-single-pipe";
  case InterferenceKind::BufIdTooManyPipes:
    return "buf-id-spans-more-than-2-pipes";
  case InterferenceKind::BufIdPairSplitAcrossBlocks:
    return "buf-id-pair-split-across-blocks";
  case InterferenceKind::BufIdDirectionalModeUnsupported:
    return "buf-id-directional-mode-unsupported";
  }
  return "?";
}

llvm::StringRef mlir::pto::oracle::severityName(Severity s) {
  return s == Severity::Warning ? "warning" : "error";
}

namespace {

/// One live get_buf, remembered so its release can be checked against it: the
/// release must be on the same pipe (B4) and in the same block (B7).
struct BufOpen {
  unsigned order;
  PipelineType pipe;
  mlir::Block *block;
};

/// One live interval stack per id key. `depth > 1` means two hazards hold the
/// same id at once -- the bug G2 exists to find.
/// An open interval owned by nothing in particular. The IR view has no notion of an
/// owning hazard (a rotating pair's ops carry eventId = -1 there), so it passes this
/// and every overlap conflicts, exactly as before.
constexpr int kNoOwner = -1;

/// The `scf.if`/`scf.index_switch` REGIONS enclosing `op`, outermost first.
///
/// Loops are skipped on purpose. Two ops in different blocks of the same loop nest
/// are still reached together whenever the loop body runs at all, and requiring the
/// two halves of an event pair to share a block would reject the set-outside /
/// wait-inside priming idiom, which is what the corpus overwhelmingly emits.
llvm::SmallVector<mlir::Region *, 4> conditionalGuards(Operation *op) {
  llvm::SmallVector<mlir::Region *, 4> guards;
  for (Operation *p = op; p; p = p->getParentOp()) {
    Operation *parent = p->getParentOp();
    if (parent && isa<scf::IfOp, scf::IndexSwitchOp>(parent))
      guards.push_back(p->getParentRegion());
  }
  std::reverse(guards.begin(), guards.end());
  return guards;
}

/// One open interval. `op` is null in the pre-codegen record view, which carries no
/// ops; a null disables only the checks that need a position in the IR.
struct Open {
  unsigned order;
  int owner;
  Operation *op;
};

struct LiveIntervals {
  llvm::SmallVector<Open> opens;

  /// Returns the order of an already-live interval that CONFLICTS with this open, or
  /// -1 if this open is legal.
  ///
  /// Two opens with the same non-sentinel owner do NOT conflict: they are two halves
  /// of ONE hazard -- the loop-carried compensation pair and the in-body pair it
  /// primes, which share the rotating ids by construction (one stamp covers set,
  /// wait, head and tail alike). Anything else still conflicts, INCLUDING a
  /// compensation op against a different hazard, which is the whole point of keying on
  /// the owner rather than just skipping compensation records.
  int64_t open(unsigned order, int owner = kNoOwner, Operation *op = nullptr) {
    int64_t live = -1;
    for (const auto &o : opens) {
      if (owner != kNoOwner && o.owner == owner)
        continue; // same hazard: its own prime/drain, not a competing interval
      live = int64_t(o.order);
      break;
    }
    opens.push_back({order, owner, op});
    return live;
  }
  /// The interval this close pops, or nullopt when there is none to pop.
  std::optional<Open> close() {
    if (opens.empty())
      return std::nullopt;
    Open top = opens.back();
    opens.pop_back();
    return top;
  }
};

std::string eventKeyName(PipelineType src, PipelineType dst, int64_t id) {
  return (oracle::pipelineTypeName(src) + "->" + oracle::pipelineTypeName(dst) +
          " id=" + llvm::Twine(id))
      .str();
}

/// The shared scan. `Key` is whatever makes two intervals conflict.
template <typename KeyT>
void scanOpen(llvm::DenseMap<KeyT, LiveIntervals> &live, const KeyT &key,
              unsigned order, llvm::StringRef opName, llvm::StringRef keyName,
              InterferenceKind nestedKind,
              llvm::SmallVectorImpl<InterferenceViolation> &out,
              int owner = kNoOwner, Operation *op = nullptr) {
  int64_t alreadyLive = live[key].open(order, owner, op);
  if (alreadyLive >= 0)
    out.push_back({nestedKind, Severity::Error, order, unsigned(alreadyLive),
                   opName.str(), keyName.str(),
                   "this id is still live from the interval opened earlier; two "
                   "overlapping hazards share it"});
}

template <typename KeyT>
void scanClose(llvm::DenseMap<KeyT, LiveIntervals> &live, const KeyT &key,
               unsigned order, llvm::StringRef opName, llvm::StringRef keyName,
               InterferenceKind orphanKind,
               llvm::SmallVectorImpl<InterferenceViolation> &out,
               Operation *op = nullptr) {
  std::optional<Open> opened = live[key].close();
  if (!opened) {
    out.push_back({orphanKind, Severity::Error, order, 0, opName.str(),
                   keyName.str(), "closes an id that no earlier op opened"});
    return;
  }
  // The two halves must be equally guarded, or a path runs one without the other.
  // A flattened scan cannot see this: the counts balance and the stack pairs them.
  if (op && opened->op &&
      conditionalGuards(opened->op) != conditionalGuards(op))
    out.push_back({InterferenceKind::EventIdPairGuardMismatch, Severity::Error,
                   order, opened->order, opName.str(), keyName.str(),
                   "this op and the one that opened the id sit under different "
                   "conditionals, so some path executes one without the other: a "
                   "wait with no set hangs, a set with no wait leaks the id"});
}

} // namespace

llvm::SmallVector<InterferenceViolation>
mlir::pto::oracle::checkNonInterference(llvm::ArrayRef<IRSyncRecord> records) {
  llvm::SmallVector<InterferenceViolation> violations;

  // Event ids are per-(src,dst) disjoint hardware; buffer ids are a global pool.
  llvm::DenseMap<std::tuple<int, int, int64_t>, LiveIntervals> eventLive;
  // Buffer ids need more than a depth counter: the release must match the
  // acquire's pipe (B4) and block (B7), so remember both per open interval.
  llvm::DenseMap<int64_t, llvm::SmallVector<BufOpen>> bufLive;
  std::map<int64_t, std::set<int>> bufPipes; // ordered: deterministic reporting

  for (const IRSyncRecord &rec : records) {
      // set_flag_dyn / wait_flag_dyn carry a runtime id, recorded as -1, so they
      // never reach this check and a rotating id is not validated here.

    if (rec.opName == "pto.set_flag" || rec.opName == "pto.wait_flag") {
      auto key = std::make_tuple(int(rec.srcPipeType), int(rec.dstPipeType),
                                 rec.eventId);
      std::string name = eventKeyName(rec.srcPipeType, rec.dstPipeType, rec.eventId);
      if (rec.opName == "pto.set_flag")
        scanOpen(eventLive, key, rec.order, rec.opName, name,
                 InterferenceKind::EventIdNested, violations, kNoOwner, rec.op);
      else
        scanClose(eventLive, key, rec.order, rec.opName, name,
                  InterferenceKind::EventIdWaitWithoutSet, violations, rec.op);
    } else if (rec.opName == "pto.get_buf" || rec.opName == "pto.rls_buf") {
      // Buffer-ID legality. See SyncOracleGates.h for the rules, and
      // docs/bufid_sync_a5_design.md "Correctness Invariants" for the protocol.
      std::string key = ("buf_id=" + llvm::Twine(rec.bufId)).str();

      // Track pipes-per-id (B5/B6) for the bidirectional token only: the
      // directional mode has a different pipe discipline and is deferred.
      if (rec.mode == 0)
        bufPipes[rec.bufId].insert(int(rec.srcPipeType));

      // The strict pairing discipline is defined for the bidirectional token
      // (mode 0). The directional mode is explicitly freed from strict pairing
      // by the spec and is a deferred mechanism -- report it, do not pair it.
      if (rec.mode != 0) {
        violations.push_back(
            {InterferenceKind::BufIdDirectionalModeUnsupported, Severity::Error,
             rec.order, 0, rec.opName, key,
             "mode " + std::to_string(rec.mode) +
                 " is the directional token (set_flagV2), a deferred mechanism "
                 "this allocator does not emit"});
        continue;
      }

      if (rec.opName == "pto.get_buf") {
        // B1: re-acquiring a live id. Covers same-pipe nesting and
        // cross-pipe interleave alike -- both are "opened twice".
        auto &live = bufLive[rec.bufId];
        if (!live.empty()) {
          violations.push_back(
              {InterferenceKind::BufIdNested, Severity::Error, rec.order,
               live.front().order, rec.opName, key,
               "this id is still live from the interval opened earlier; a "
               "second acquire double-increments the counter and the get may "
               "never pop"});
        }
        live.push_back({rec.order, rec.srcPipeType, rec.block});
      } else {
        auto &live = bufLive[rec.bufId];
        if (live.empty()) {
          // B2
          violations.push_back(
              {InterferenceKind::BufIdReleaseWithoutAcquire, Severity::Error,
               rec.order, 0, rec.opName, key,
               "closes an id that no earlier get_buf opened"});
        } else {
          BufOpen open = live.back();
          live.pop_back();
          // B4: the release must be on the same pipe as its acquire.
          if (open.pipe != rec.srcPipeType) {
            violations.push_back(
                {InterferenceKind::BufIdPipeMismatch, Severity::Error,
                 rec.order, open.order, rec.opName, key,
                 "released on " +
                     oracle::pipelineTypeName(rec.srcPipeType).str() +
                     " but acquired on " +
                     oracle::pipelineTypeName(open.pipe).str() +
                     "; the token brackets one op on one pipe"});
          }
          // B7: the pair must sit in one block, or only one arm may execute.
          if (open.block && rec.block && open.block != rec.block) {
            violations.push_back(
                {InterferenceKind::BufIdPairSplitAcrossBlocks, Severity::Error,
                 rec.order, open.order, rec.opName, key,
                 "get_buf and rls_buf are in different blocks; if they are in "
                 "different control-flow arms only one executes and the id's "
                 "counters desync"});
          }
        }
      }
    }
  }

  // B5 / B6: how many distinct pipes does each id span?
  llvm::SmallVector<int64_t> ids;
  for (const auto &entry : bufPipes)
    ids.push_back(entry.first);
  llvm::sort(ids);
  for (int64_t id : ids) {
    const auto &pipes = bufPipes[id];
    std::string key = ("buf_id=" + llvm::Twine(id)).str();
    if (pipes.size() == 1) {
      violations.push_back(
          {InterferenceKind::BufIdSinglePipe, Severity::Error, 0, 0,
           "pto.get_buf", key,
           "spans only 1 pipe (" +
               oracle::pipelineTypeName(PipelineType(*pipes.begin())).str() +
               "); buffer-ID is only for sync between TWO different pipes -- "
               "same-pipe ordering needs bar.pipe"});
    } else if (pipes.size() > 2) {
      // Soft rule, not an illegality: this gate must accept an id that spans
      // more than two pipes, so the case is reported rather than failed.
      violations.push_back(
          {InterferenceKind::BufIdTooManyPipes, Severity::Warning, 0, 0,
           "pto.get_buf", key,
           "spans " + std::to_string(pipes.size()) +
               " pipes (>2 pipe-pairs on one id); the spec advises against "
               "this -- it is the coalescing bound the allocator must decide"});
    }
  }

  // A leaked id: opened, never closed. Deterministic order for diffability.
  llvm::SmallVector<InterferenceViolation> leaks;
  for (const auto &entry : eventLive)
    for (const auto &open : entry.second.opens) {
      unsigned order = open.order;
      leaks.push_back({InterferenceKind::EventIdUnclosed, Severity::Error,
                       order, order, "pto.set_flag",
                       eventKeyName(PipelineType(std::get<0>(entry.first)),
                                    PipelineType(std::get<1>(entry.first)),
                                    std::get<2>(entry.first)),
                       "opened but never waited: the id is leaked"});
    }
  for (const auto &entry : bufLive)
    for (const BufOpen &open : entry.second)
      leaks.push_back({InterferenceKind::BufIdUnclosed, Severity::Error,
                       open.order, open.order, "pto.get_buf",
                       ("buf_id=" + llvm::Twine(entry.first)).str(),
                       "acquired but never released: the id's counters desync"});
  llvm::sort(leaks, [](const InterferenceViolation &a,
                       const InterferenceViolation &b) {
    return a.order < b.order;
  });
  violations.append(leaks.begin(), leaks.end());

  return violations;
}
llvm::SmallVector<InterferenceViolation>
mlir::pto::oracle::checkNonInterference(llvm::ArrayRef<SyncOpRecord> records) {
  using TYPE = SyncOperation::TYPE;
  llvm::SmallVector<InterferenceViolation> violations;
  llvm::DenseMap<std::tuple<int, int, int64_t>, LiveIntervals> eventLive;

  // THE SCAN REQUIRES PROGRAM ORDER, AND THE CALLER'S ORDER IS NOT IT.
  //
  // `extractFromSyncOps` emits records in STORAGE order: group by group, and
  // within a group the set before the wait -- regardless of where they actually
  // sit in the program. For a BACKWARD-matched pair (a loop-carried WAR, whose
  // wait is satisfied by the previous iteration's set and therefore appears
  // EARLIER in the program than its set) that hands the scan [set, wait], which
  // reads as perfectly well-formed. The gate then reports OK on a kernel whose
  // event realization deadlocks on iteration 1 -- 188 such pairs exist across 49
  // of 98 corpus kernels.
  //
  // Sorting by `irIndex` restores true program order for the set/wait relation.
  // Safe to do here, and verified across the corpus: NO pair has
  // set.irIndex == wait.irIndex (644 forward, 188 backward, 0 ties), so the
  // relation is never ambiguous.
  //
  // KNOWN LIMIT, deliberately not papered over: `irIndex` is SyncIR-ELEMENT
  // granular, not op granular -- two syncs on the same element (one in
  // `pipeBefore`, one in `pipeAfter`) share an index, and `SyncOpRecord` does not
  // record which side. A stable sort leaves those in storage order. That cannot
  // affect the set/wait relation (0 ties, above), but it means the relative order
  // of two DIFFERENT hazards colliding on one element is not resolved here. It
  // only matters once ids are reused across hazards, and G2 on the IR view
  // -- true program order, post-codegen -- remains the authority.
  llvm::SmallVector<SyncOpRecord> ordered(records.begin(), records.end());
  llvm::stable_sort(ordered, [](const SyncOpRecord &a, const SyncOpRecord &b) {
    return a.irIndex < b.irIndex;
  });

  for (const SyncOpRecord &rec : ordered) {
    TYPE type = static_cast<TYPE>(rec.typeCode);
    if (type != TYPE::SET_EVENT && type != TYPE::WAIT_EVENT)
      continue;
    for (int id : rec.eventIds) {
      auto key = std::make_tuple(rec.srcPipe, rec.dstPipe, int64_t(id));
      std::string name = eventKeyName(PipelineType(rec.srcPipe),
                                      PipelineType(rec.dstPipe), id);
      // The OWNING hazard: a compensation pair reports the in-body hazard it primes,
      // everything else reports itself. So a compensation op and the pair it primes
      // share an owner and do not nest, while a collision with any other hazard --
      // compensation or not -- is still reported. Scoping this to `isCompensation`
      // alone would have excused the latter too, and syncops is the ONLY view that
      // checks rotating ids at all (the IR view's dyn ops carry eventId = -1).
      int owner = rec.isCompensation && rec.compensationOf >= 0
                      ? rec.compensationOf
                      : int(rec.syncIndex);
      if (type == TYPE::SET_EVENT)
        scanOpen(eventLive, key, rec.syncIndex, rec.type, name,
                 InterferenceKind::EventIdNested, violations, owner);
      else
        scanClose(eventLive, key, rec.syncIndex, rec.type, name,
                  InterferenceKind::EventIdWaitWithoutSet, violations);
    }
  }

  llvm::SmallVector<InterferenceViolation> leaks;
  for (const auto &entry : eventLive)
    for (const auto &open : entry.second.opens) {
      unsigned order = open.order;
      leaks.push_back({InterferenceKind::EventIdUnclosed, Severity::Error,
                       order, order, "set_event",
                       eventKeyName(PipelineType(std::get<0>(entry.first)),
                                    PipelineType(std::get<1>(entry.first)),
                                    std::get<2>(entry.first)),
                       "opened but never waited: the id is leaked"});
    }
  llvm::sort(leaks, [](const InterferenceViolation &a,
                       const InterferenceViolation &b) {
    return a.order < b.order;
  });
  violations.append(leaks.begin(), leaks.end());

  return violations;
}


unsigned mlir::pto::oracle::countErrors(
    llvm::ArrayRef<InterferenceViolation> violations) {
  return llvm::count_if(violations, [](const InterferenceViolation &v) {
    return v.severity == Severity::Error;
  });
}

void mlir::pto::oracle::printInterferenceViolations(
    llvm::raw_ostream &os, llvm::StringRef funcName, llvm::StringRef view,
    size_t recordCount, llvm::ArrayRef<InterferenceViolation> violations) {
  unsigned errors = countErrors(violations);
  unsigned warnings = violations.size() - errors;
  os << "[sync-gate:g2] func=" << funcName << " view=" << view
     << " records=" << recordCount << " errors=" << errors
     << " warnings=" << warnings << (errors == 0 ? " OK" : "") << "\n";
  for (const InterferenceViolation &v : violations) {
    os << (v.severity == Severity::Warning ? "  ~~ " : "  !! ");
    os << "#" << v.order << " " << v.opName
       << " kind=" << interferenceKindName(v.kind) << " " << v.key;
    if (v.kind == InterferenceKind::EventIdNested ||
        v.kind == InterferenceKind::BufIdNested ||
        v.kind == InterferenceKind::BufIdPipeMismatch ||
        v.kind == InterferenceKind::BufIdPairSplitAcrossBlocks)
      os << " live_since=#" << v.openedAt;
    os << " : " << v.detail << "\n";
  }
}

//===----------------------------------------------------------------------===//
// The happens-before edge model, consumed by G1-self
//===----------------------------------------------------------------------===//

namespace {

/// An op that needs ordering: carries a pipe and touches memory. This is the
/// same test InjectBarrierAllSync uses to decide where a barrier is required.
bool isAnchor(Operation *op) {
  if (isa<pto::SetFlagOp, pto::WaitFlagOp, pto::GetBufOp, pto::RlsBufOp,
          pto::BarrierOp, pto::SetFlagDynOp, pto::WaitFlagDynOp>(op))
    return false;
  auto memEffect = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffect || !(memEffect.hasEffect<MemoryEffects::Read>() ||
                      memEffect.hasEffect<MemoryEffects::Write>()))
    return false;
  return isa<pto::OpPipeInterface>(op);
}

/// A sync op, positioned by how many anchors precede it.
struct SyncEvent {
  enum Kind {
    BarrierAll, BarrierPipe, SetFlag, WaitFlag, GetBuf, RlsBuf,
    SetFlagDyn, WaitFlagDyn
  } kind;
  unsigned slot;
  pto::PIPE srcPipe;
  pto::PIPE dstPipe;
  int64_t id; // event id, or buf id
  // The iteration dimension. `order` is the walk position, used only to say
  // "before"/"after" in program order; `loop` is the NEAREST enclosing loop, or
  // null at function level.
  Operation *op = nullptr;
  unsigned order = 0;
  Operation *loop = nullptr;
  /// Rotation depth N for a Dyn op: the ordering it establishes is at DISTANCE N,
  /// not 1. 1 for every static op.
  unsigned rotation = 1;
};

/// What one loop contributes to the iteration dimension: the anchor range of
/// its body (inclusive) and the walk position of the earliest op inside it.
///
/// `firstOrder` is taken from the loop's CONTENTS, not from the loop op itself:
/// `walk` is post-order, so the loop op is visited *after* its body, and using
/// its own position would place the whole body "before" the loop.
struct LoopSpan {
  unsigned firstAnchor = std::numeric_limits<unsigned>::max();
  unsigned lastAnchor = 0;
  unsigned firstOrder = std::numeric_limits<unsigned>::max();
  bool hasAnchors() const { return firstAnchor <= lastAnchor; }
};

/// Recover the rotation depth N from a `set_flag_dyn` / `wait_flag_dyn` id operand.
///
/// Codegen builds the id as `eventIds[slot % N]`: an `arith.remui <slot>, N`
/// feeding a chain of `arith.select`s over N constants
/// (`CreateSetWaitOpForMultiBuffer`). N is the modulus, so walking back from the
/// operand to the nearest `RemUIOp` with a constant RHS recovers it exactly.
///
/// Returns 0 when N cannot be recovered. Callers treat that as "unknown depth"
/// and credit NOTHING -- crediting an ordering whose distance cannot be named
/// would be a false pass, and this model has no way to express "some distance".
unsigned rotationDepthOf(Value eventId) {
  llvm::SmallVector<Value, 8> work{eventId};
  llvm::SmallPtrSet<Operation *, 8> seen;
  while (!work.empty()) {
    Value v = work.pop_back_val();
    Operation *def = v.getDefiningOp();
    if (!def || !seen.insert(def).second)
      continue;
    if (auto rem = dyn_cast<arith::RemUIOp>(def)) {
      llvm::APInt n;
      if (matchPattern(rem.getRhs(), m_ConstantInt(&n)) && n.getZExtValue() >= 2)
        return unsigned(n.getZExtValue());
    }
    for (Value operand : def->getOperands())
      work.push_back(operand);
  }
  return 0;
}

/// Is `op` nested anywhere inside `loop`?
bool insideLoop(Operation *op, Operation *loop) {
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
    if (p == loop)
      return true;
  return false;
}

/// The blocks on `loop`'s ancestor spine: the block holding `loop` itself, plus
/// the block holding each of its ancestor ops, up to the function.
///
/// An op in one of these blocks and earlier in program order executes on EVERY
/// path that reaches `loop` (structured control flow, single-block regions), so
/// the spine is what "unconditionally before the loop" means concretely. An op
/// in any other block sits inside a region the loop is not inside -- a
/// conditional arm, or a sibling loop that may run zero times -- and may not
/// execute at all. This is strictly stronger than "not inside the loop", and it
/// has to be: see the P1' note in `isPrimed`.
llvm::SmallPtrSet<Block *, 8> spineBlocks(Operation *loop) {
  llvm::SmallPtrSet<Block *, 8> blocks;
  for (Operation *p = loop; p; p = p->getParentOp())
    if (Block *block = p->getBlock())
      blocks.insert(block);
  return blocks;
}

/// Does `op` execute exactly once per iteration of `loop`?
///
/// Only if it is a DIRECT child of the loop's body: an op one level deeper sits
/// inside some other region -- an `scf.if` arm, or a nested loop that may run
/// zero times -- and may be skipped on an iteration. A carried pair whose
/// producer can be skipped establishes nothing on the iteration that skips it.
bool executesEachIteration(Operation *op, Operation *loop) {
  return op->getParentOp() == loop;
}

} // namespace

CoverageProfile mlir::pto::oracle::computeCoverage(
    func::FuncOp func, llvm::DenseMap<Operation *, unsigned> *anchorIndex) {
  CoverageProfile profile;

  llvm::SmallVector<pto::PIPE> anchorPipes;
  /// Parallel to `anchorPipes`, same index space, filled by the same walk. The edge
  /// filter needs the op itself to ask whether a sync reaches it; building this list
  /// in a second walk is what would let the two drift out of step.
  llvm::SmallVector<Operation *> anchorOps;
  llvm::SmallVector<SyncEvent> events;
  std::string signatureText;
  // Deterministic iteration order: the carried-edge scan walks these.
  llvm::MapVector<Operation *, LoopSpan> loops;
  unsigned walkOrder = 0;

  // Enclosing loops of `op`, innermost first, stopping at the function.
  auto enclosingLoops = [&](Operation *op) {
    llvm::SmallVector<Operation *, 4> chain;
    for (Operation *p = op->getParentOp(); p && p != func.getOperation();
         p = p->getParentOp())
      if (isa<LoopLikeOpInterface>(p))
        chain.push_back(p);
    return chain;
  };

  func.walk([&](Operation *op) {
    unsigned order = walkOrder++;
    llvm::SmallVector<Operation *, 4> chain = enclosingLoops(op);
    for (Operation *loop : chain) {
      LoopSpan &span = loops[loop];
      span.firstOrder = std::min(span.firstOrder, order);
    }

    if (isAnchor(op)) {
      pto::PIPE pipe = cast<pto::OpPipeInterface>(op).getPipe();
      unsigned index = anchorPipes.size();
      anchorPipes.push_back(pipe);
      anchorOps.push_back(op);
      if (anchorIndex)
        (*anchorIndex)[op] = index;
      for (Operation *loop : chain) {
        LoopSpan &span = loops[loop];
        span.firstAnchor = std::min(span.firstAnchor, index);
        span.lastAnchor = std::max(span.lastAnchor, index);
      }
      signatureText += op->getName().getStringRef().str();
      signatureText += ':';
      signatureText += stringifyPIPE(pipe).str();
      // The carried-edge comparison presumes both runs agree on loop
      // structure. The signature is where presumptions get checked rather than
      // assumed, so the nest depth goes in it.
      signatureText += "#";
      signatureText += std::to_string(chain.size());
      signatureText += ';';
      return;
    }
    unsigned slot = anchorPipes.size();
    Operation *nearest = chain.empty() ? nullptr : chain.front();
    if (auto setFlag = dyn_cast<pto::SetFlagOp>(op)) {
      events.push_back({SyncEvent::SetFlag, slot,
                        setFlag.getSrcPipeAttr().getPipe(),
                        setFlag.getDstPipeAttr().getPipe(),
                        int64_t(setFlag.getEventIdAttr().getEvent()), op, order,
                        nearest});
    } else if (auto waitFlag = dyn_cast<pto::WaitFlagOp>(op)) {
      events.push_back({SyncEvent::WaitFlag, slot,
                        waitFlag.getSrcPipeAttr().getPipe(),
                        waitFlag.getDstPipeAttr().getPipe(),
                        int64_t(waitFlag.getEventIdAttr().getEvent()), op, order,
                        nearest});
    } else if (auto getBuf = dyn_cast<pto::GetBufOp>(op)) {
      // A spelling this does not decode drops the event, and a dropped event is
      // indistinguishable from an absent one: the ordering the token establishes
      // then reads as uncovered. `isConcreteSyncPipe` mirrors `verifyBufSyncOp`,
      // which rejects PIPE_ALL and PIPE_UNASSIGNED on these ops.
      pto::PIPE pipe = oracle::bufSyncPipe(getBuf.getOpTypeAttr());
      if (pto::isConcreteSyncPipe(pipe))
        events.push_back({SyncEvent::GetBuf, slot, pipe, pipe,
                          int64_t(getBuf.getBufId()), op, order, nearest});
    } else if (auto rlsBuf = dyn_cast<pto::RlsBufOp>(op)) {
      pto::PIPE pipe = oracle::bufSyncPipe(rlsBuf.getOpTypeAttr());
      if (pto::isConcreteSyncPipe(pipe))
        events.push_back({SyncEvent::RlsBuf, slot, pipe, pipe,
                          int64_t(rlsBuf.getBufId()), op, order, nearest});
    } else if (auto setDyn = dyn_cast<pto::SetFlagDynOp>(op)) {
      // The rotating pair is what multi-buffering emits, and it was
      // previously invisible: the type chain below had no Dyn case, so the op
      // fell off the end and contributed NO edge. `id` is -1 because there is no
      // static id -- the id is chosen at runtime -- so it cannot collide with a
      // static key.
      events.push_back({SyncEvent::SetFlagDyn, slot,
                        setDyn.getSrcPipeAttr().getPipe(),
                        setDyn.getDstPipeAttr().getPipe(), -1, op, order,
                        nearest, rotationDepthOf(setDyn.getEventId())});
    } else if (auto waitDyn = dyn_cast<pto::WaitFlagDynOp>(op)) {
      events.push_back({SyncEvent::WaitFlagDyn, slot,
                        waitDyn.getSrcPipeAttr().getPipe(),
                        waitDyn.getDstPipeAttr().getPipe(), -1, op, order,
                        nearest, rotationDepthOf(waitDyn.getEventId())});
    } else if (auto barrier = dyn_cast<pto::BarrierOp>(op)) {
      pto::PIPE pipe = barrier.getPipeAttr().getPipe();
      if (pipe == pto::PIPE::PIPE_ALL) {
        ++profile.barrierAll;
      }
      events.push_back({pipe == pto::PIPE::PIPE_ALL ? SyncEvent::BarrierAll
                                                    : SyncEvent::BarrierPipe,
                        slot, pipe, pipe, -1, op, order, nearest});
    }
  });

  profile.anchorCount = anchorPipes.size();
  profile.anchorSignature =
      static_cast<uint64_t>(llvm::hash_value(llvm::StringRef(signatureText)));

  llvm::DenseSet<std::pair<unsigned, unsigned>> edges;
  /// `reachedBy` is the op whose execution the edge depends on -- the CONSUMING end of
  /// the mechanism: a barrier itself, the `wait_flag` of a flag pair, the `get_buf` of
  /// a token pair. An edge is credited only when that op is reached on every path that
  /// reaches the sink, because a sync the sink can skip establishes nothing.
  ///
  /// "Reached on every path" is tested as: the conditionals enclosing the sync must be
  /// a PREFIX of those enclosing the sink. Enclosing LOOPS are not compared.
  ///
  /// NOT `DominanceInfo`, and the difference is not cosmetic. Structural dominance
  /// says an op in a loop body never dominates an op after the loop, which withdraws
  /// credit from a barrier inside a constant-trip-count loop -- it always executes, so
  /// the ordering does hold. Measured: that rejection alone turns
  /// `control_flow_nested_vec` from clean to three uncovered same-pipe dependencies.
  /// A loop whose trip count can be zero is a real hole, and it is NOT covered here;
  /// the carried rules test the loop spine for that case instead.
  ///
  /// Sink side only, deliberately. A sync inside an `scf.if` arm legitimately orders a
  /// dependence whose sink is in that same arm -- every path reaching the sink passes
  /// through it -- so also requiring the sync to be reached whenever the SOURCE is
  /// would reject the shape the emitter produces most.
  auto addEdges = [&](unsigned beforeSlot, pto::PIPE fromPipe, unsigned afterSlot,
                      pto::PIPE toPipe, bool anyPipe, Operation *reachedBy) {
    llvm::SmallVector<mlir::Region *, 4> syncGuards;
    if (reachedBy)
      syncGuards = conditionalGuards(reachedBy);
    for (unsigned i = 0; i < beforeSlot; ++i) {
      if (!anyPipe && anchorPipes[i] != fromPipe)
        continue;
      for (unsigned j = afterSlot; j < anchorPipes.size(); ++j) {
        if (!anyPipe && anchorPipes[j] != toPipe)
          continue;
        if (reachedBy) {
          llvm::SmallVector<mlir::Region *, 4> sinkGuards =
              conditionalGuards(anchorOps[j]);
          if (syncGuards.size() > sinkGuards.size() ||
              !std::equal(syncGuards.begin(), syncGuards.end(),
                          sinkGuards.begin()))
            continue;
        }
        if (i < j)
          edges.insert({i, j});
      }
    }
  };

  for (size_t e = 0; e < events.size(); ++e) {
    const SyncEvent &ev = events[e];
    switch (ev.kind) {
    case SyncEvent::BarrierAll:
      addEdges(ev.slot, ev.srcPipe, ev.slot, ev.dstPipe, /*anyPipe=*/true, ev.op);
      break;
    case SyncEvent::BarrierPipe:
      addEdges(ev.slot, ev.srcPipe, ev.slot, ev.dstPipe, /*anyPipe=*/false, ev.op);
      break;
    case SyncEvent::SetFlag:
      // Pair with the next matching wait. G2 guarantees the interval discipline
      // that makes "next" the right answer.
      for (size_t w = e + 1; w < events.size(); ++w) {
        const SyncEvent &wait = events[w];
        if (wait.kind == SyncEvent::WaitFlag && wait.srcPipe == ev.srcPipe &&
            wait.dstPipe == ev.dstPipe && wait.id == ev.id) {
          // The WAIT is the consuming end: the sink is ordered only if the wait ran.
          addEdges(ev.slot, ev.srcPipe, wait.slot, wait.dstPipe, false, wait.op);
          break;
        }
      }
      break;
    case SyncEvent::RlsBuf:
      // Buffer-token happens-before: get_buf acquires what the previous holder
      // released, so rls precedes get for the same buffer id.
      for (size_t g = e + 1; g < events.size(); ++g) {
        const SyncEvent &get = events[g];
        if (get.kind == SyncEvent::GetBuf && get.id == ev.id)
          // The acquire is the consuming end, as the wait is for a flag pair.
          addEdges(ev.slot, ev.srcPipe, get.slot, get.dstPipe, false, get.op);
      }
      break;
    case SyncEvent::WaitFlag:
    case SyncEvent::GetBuf:
      break;
    case SyncEvent::SetFlagDyn:
    case SyncEvent::WaitFlagDyn:
      // Deliberately NO FORWARD EDGE. A rotating pair orders iteration j-N
      // against iteration j; within ONE iteration it orders nothing, because the
      // set and the wait in the same iteration touch DIFFERENT event ids
      // (eventIds[j%N] is set at the bottom, and the wait at the top consumed
      // whatever iteration j-N set). Crediting a same-iteration edge here would
      // invent an ordering the hardware does not establish. Their contribution is
      // the CARRIED edge at distance N, added by the carried scan below.
      break;
    }
  }

  profile.edges.assign(edges.begin(), edges.end());
  llvm::sort(profile.edges);

  //===--------------------------------------------------------------------===//
  // The iteration dimension. Rules 1-3 (see SyncOracleGates.h).
  //
  // Deliberately a SECOND, INDEPENDENT scan. The forward pass above already
  // consumes the head/tail compensation into two forward pairs, so bending it to
  // pair backwards would change nothing -- which is precisely the measurement
  // that motivated this. Nothing above this line is touched, so no existing
  // forward edge can move.
  //
  // BARRIER-REALISED CARRIED ORDERINGS are handled here: a barrier
  // are credited by the covering rule below. It was going to be deferred
  // as untestable, but adversarial review showed it FIRES TODAY -- see the
  // derivation at the covering loop.
  //===--------------------------------------------------------------------===//

  // (from, to, distance) -> the loop that carries it. Second insert from a
  // DIFFERENT loop means one anchor pair is carried at two nest levels, which
  // this flat key cannot tell apart. The counter is a tripwire: nonzero means the
  // key has become ambiguous and the closure below cannot be trusted.
  std::map<std::tuple<unsigned, unsigned, unsigned>, Operation *> carriedOwner;
  unsigned loopKeyCollisions = 0;

  auto addCarried = [&](const SyncEvent &producer, const SyncEvent &consumer,
                        Operation *loop, unsigned distance) {
    const LoopSpan &span = loops[loop];
    if (!span.hasAnchors())
      return;
    // Rule 2. Both endpoints restricted to the carrying loop's body: an anchor
    // outside it has no iteration index, so a d=1 edge touching one would be
    // meaningless.
    for (unsigned x = span.firstAnchor;
         x <= span.lastAnchor && x < producer.slot; ++x) {
      if (anchorPipes[x] != producer.srcPipe)
        continue;
      for (unsigned y = std::max(span.firstAnchor, consumer.slot);
           y <= span.lastAnchor; ++y) {
        if (anchorPipes[y] != consumer.dstPipe)
          continue;
        auto key = std::make_tuple(x, y, distance);
        auto it = carriedOwner.find(key);
        if (it == carriedOwner.end())
          carriedOwner[key] = loop;
        else if (it->second != loop)
          ++loopKeyCollisions;
      }
    }
  };

  // Rule 3: is this loop-carried EVENT pair primed by a set outside the loop?
  //
  // P1' IS A REACHABILITY TEST, NOT A NESTING TEST -- and that distinction was
  // a REAL FALSE PASS, found by adversarial review and reproduced. The first
  // version asked only `!insideLoop(head.op, loop)`, which credits a head that
  // may never execute. Four deadlocking shapes were blessed `violations=0 OK`:
  //   (a) `scf.if %c { set_flag }` before the loop  -- hangs when %c is false;
  //   (b) `scf.if %c { set_flag } else { <the loop> }` -- prime and loop on
  //       MUTUALLY EXCLUSIVE arms, so the prime provably never ran when the loop
  //       runs: hangs for EVERY value of %c;
  //   (c) the same with a guarded drain too, which balances G2's flat scan --
  //       every gate reported OK on a certain hang;
  //   (d) `scf.for %j = 0 to %n { set_flag }` before the loop -- a zero-trip
  //       prime, on the very kernel named `test_dynamic_valid_shape`.
  // `scf.if` is not a LoopLikeOpInterface, so nesting-only reasoning cannot see
  // any of them. Blessing a hang is worse than having no iteration dimension at
  // all, so P1' requires the head to be ON THE LOOP'S SPINE.
  auto isPrimed = [&](const SyncEvent &producer, Operation *loop) {
    unsigned entry = loops[loop].firstOrder;
    llvm::SmallPtrSet<Block *, 8> spine = spineBlocks(loop);
    for (const SyncEvent &head : events) {
      if (head.kind != SyncEvent::SetFlag || head.id != producer.id ||
          head.srcPipe != producer.srcPipe || head.dstPipe != producer.dstPipe)
        continue;
      // P1' -- unconditionally reached. Subsumes the old P1: a block inside the
      // loop body is not on the spine either.
      if (!spine.contains(head.op->getBlock()))
        continue;
      if (head.order >= entry) // P2
        continue;
      bool consumed = false; // P3: still live at loop entry?
      for (const SyncEvent &wait : events) {
        if (wait.kind != SyncEvent::WaitFlag || wait.id != producer.id ||
            wait.srcPipe != producer.srcPipe || wait.dstPipe != producer.dstPipe)
          continue;
        if (insideLoop(wait.op, loop))
          continue;
        if (wait.order > head.order && wait.order < entry) {
          consumed = true;
          break;
        }
      }
      if (!consumed)
        return true;
    }
    return false;
  };

  for (const auto &entry : loops) {
    Operation *loop = entry.first;

    // Rule 1: match forward within THIS body only; the leftovers wrap.
    // Key: (src, dst, id) for flags; the buffer rule keys on the buffer id
    // alone, matching the forward rule above.
    std::map<std::tuple<int, int, int64_t>, llvm::SmallVector<unsigned>>
        openProducers, strandedConsumers;
    auto flagKey = [](const SyncEvent &e) {
      return std::make_tuple(int(e.srcPipe), int(e.dstPipe), e.id);
    };
    auto bufKey = [](const SyncEvent &e) {
      return std::make_tuple(-1, -1, e.id);
    };

    for (unsigned i = 0; i < events.size(); ++i) {
      const SyncEvent &ev = events[i];
      if (ev.loop != loop)
        continue;
      // The mirror of P1' on the IN-BODY side, and a reproduced false pass in
      // its own right: with an unconditional prime and drain but the in-body
      // producer under `scf.if`, iterations 2..N have nothing to wait on and the
      // kernel hangs -- yet G1 and G2 both reported OK. `ev.loop` is the NEAREST
      // enclosing LOOP, so an op inside an `scf.if` inside the body still lands
      // here; only a direct child of the body runs every iteration.
      if (!executesEachIteration(ev.op, loop))
        continue;
      switch (ev.kind) {
      case SyncEvent::SetFlag:
      case SyncEvent::SetFlagDyn:
        openProducers[flagKey(ev)].push_back(i);
        break;
      case SyncEvent::RlsBuf:
        openProducers[bufKey(ev)].push_back(i);
        break;
      case SyncEvent::WaitFlag:
      case SyncEvent::WaitFlagDyn:
      case SyncEvent::GetBuf: {
        auto key =
            ev.kind == SyncEvent::GetBuf ? bufKey(ev) : flagKey(ev);
        auto &open = openProducers[key];
        if (!open.empty())
          open.erase(open.begin()); // forward pair, consumed in this iteration
        else
          strandedConsumers[key].push_back(i);
        break;
      }
      case SyncEvent::BarrierAll:
      case SyncEvent::BarrierPipe:
        break;
      }
    }

    // Barrier-realised carried orderings fire on real kernels. A barrier in the
    // body establishes carried orderings without any flag pair, so a candidate
    // that serializes instead of pairing has nothing for Rule 1 to detect and
    // was REJECTED for an ordering it does establish. That is a false failure,
    // and it is reachable with an in-tree flag on an in-tree sample:
    //   --enable-inject-barrier-all-sync on samples/Sync/test_dynamic_valid_shape.pto
    // reported `!! missing-backward-ordering anchor#1 -> anchor#0 (d=1)` against
    // the InsertSync reference. (The earlier "no corpus kernel emits that shape"
    // reading was scoped to PINNED tests, which is not the same as reachable.)
    //
    // THE COVERING RULE. Unroll two iterations around a body barrier at slot s:
    //     iter k:   a_0 .. a_{s-1}  BARRIER  a_s .. a_{n-1}
    //     iter k+1: a_0 .. a_{s-1}  BARRIER  a_s .. a_{n-1}
    // For the carried edge (from=x, to=y) -- "x in iter k before y in iter k+1":
    //   x < s   => iter k's OWN barrier follows x and precedes all of iter k+1;
    //   y >= s  => iter k+1's barrier precedes y and follows all of iter k.
    // So it is covered iff (x < s || y >= s), i.e. UNCOVERED only in the wrap
    // gap y < s <= x. A PIPE_ALL barrier covers any pipes; a per-pipe barrier
    // covers only anchors on its own pipe, mirroring the forward rule.
    //
    // A barrier needs no priming -- it is unconditional hardware serialization,
    // not a flag -- but it must still RUN every iteration, hence the
    // `executesEachIteration` filter above applies to it as well.
    for (const SyncEvent &ev : events) {
      if (ev.loop != loop || !executesEachIteration(ev.op, loop))
        continue;
      if (ev.kind != SyncEvent::BarrierAll && ev.kind != SyncEvent::BarrierPipe)
        continue;
      const LoopSpan &span = loops[loop];
      if (!span.hasAnchors())
        continue;
      bool anyPipe = ev.kind == SyncEvent::BarrierAll;
      for (unsigned x = span.firstAnchor; x <= span.lastAnchor; ++x) {
        if (!anyPipe && anchorPipes[x] != ev.srcPipe)
          continue;
        for (unsigned y = span.firstAnchor; y <= span.lastAnchor; ++y) {
          if (!anyPipe && anchorPipes[y] != ev.dstPipe)
            continue;
          if (!(x < ev.slot || y >= ev.slot))
            continue;
          // A barrier drains the pipe, so it orders EVERY pair of adjacent
          // iterations -- distance 1, and by transitive closure every larger
          // distance too. It is the strongest carried ordering, so it is
          // recorded at d=1 regardless of any rotation in the loop.
          auto key = std::make_tuple(x, y, 1u);
          if (!carriedOwner.count(key))
            carriedOwner[key] = loop;
        }
      }
    }

    for (auto &open : openProducers) {
      auto stranded = strandedConsumers.find(open.first);
      if (stranded == strandedConsumers.end())
        continue;
      unsigned pairs = std::min(open.second.size(), stranded->second.size());
      for (unsigned k = 0; k < pairs; ++k) {
        const SyncEvent &producer = events[open.second[k]];
        const SyncEvent &consumer = events[stranded->second[k]];
        if (consumer.order >= producer.order)
          continue; // not backward: nothing wraps
        // Rule 3 applies to events only. A buffer token's ticket lock starts
        // free, so iteration 1's get_buf needs no priming op -- there is no
        // unprimed state to discriminate against.
        // A DYN pair rotates: `wait_flag_dyn` in iteration j waits on
        // eventIds[j % N] and `set_flag_dyn` in iteration k sets eventIds[k % N],
        // so the wait at j is satisfied by the latest k < j with k%N == j%N --
        // i.e. k = j - N. The ordering it establishes is at DISTANCE N, not 1.
        // Derived from the emitted selector's modulus, not assumed.
        bool isDyn = producer.kind == SyncEvent::SetFlagDyn ||
                     consumer.kind == SyncEvent::WaitFlagDyn;
        unsigned distance = 1;
        if (isDyn) {
          unsigned n = std::max(producer.rotation, consumer.rotation);
          if (n < 2)
            continue; // depth unrecoverable: credit nothing, never guess
          distance = n;
        }
        // Rule 3 (priming) applies to the static event pair only. A rotating pair
        // is primed by N static set_flags before the loop -- emitted through the
        // compensation fan-out -- which the static scan below already sees as
        // ordinary primes on the same (src,dst).
        if (producer.kind == SyncEvent::SetFlag && !isPrimed(producer, loop))
          continue;
        addCarried(producer, consumer, loop, distance);
      }
    }
  }

  for (const auto &owned : carriedOwner)
    profile.carriedEdges.push_back({std::get<0>(owned.first),
                                    std::get<1>(owned.first),
                                    std::get<2>(owned.first)});
  llvm::sort(profile.carriedEdges);

  // Through emitReport like every other oracle print: nested func passes run in
  // PARALLEL and ptoas does not disable threading, so an unsynchronised write here
  // can interleave with another function's report.
  if (loopKeyCollisions)
    oracle::emitReport([&](llvm::raw_ostream &os) {
      os << "[sync-cov] !! carried-edge-loop-collision func=" << func.getName()
         << " count=" << loopKeyCollisions
         << " : one anchor pair is carried at two nest levels; the "
            "(from,to,distance) key cannot tell them apart\n";
    });

  return profile;
}

//===----------------------------------------------------------------------===//
// G1-self: absolute coverage against the dependency analysis
//===----------------------------------------------------------------------===//
// G1: coverage superset (differential)
//===----------------------------------------------------------------------===//

llvm::StringRef
mlir::pto::oracle::coverageViolationKindName(CoverageViolationKind k) {
  switch (k) {
  case CoverageViolationKind::MissingOrdering:
    return "missing-ordering";
  case CoverageViolationKind::MissingBackwardOrdering:
    return "missing-backward-ordering";
  case CoverageViolationKind::AnchorMismatch:
    return "anchor-signature-mismatch";
  case CoverageViolationKind::MissingReference:
    return "missing-reference-profile";
  }
  return "?";
}

llvm::SmallVector<CoverageViolation>
mlir::pto::oracle::checkCoverageSuperset(const CoverageProfile &cand,
                                         const CoverageProfile &ref) {
  llvm::SmallVector<CoverageViolation> violations;

  if (cand.anchorCount != ref.anchorCount ||
      cand.anchorSignature != ref.anchorSignature) {
    violations.push_back(
        {CoverageViolationKind::AnchorMismatch, 0, 0,
         "the two runs did not see the same ops: candidate has " +
             std::to_string(cand.anchorCount) + " anchor(s), reference has " +
             std::to_string(ref.anchorCount) +
             " -- the coverage comparison would be meaningless"});
    return violations;
  }

  llvm::DenseSet<std::pair<unsigned, unsigned>> have;
  for (const auto &edge : cand.edges)
    have.insert(edge);
  for (const auto &edge : ref.edges)
    if (!have.count(edge))
      violations.push_back(
          {CoverageViolationKind::MissingOrdering, edge.first, edge.second,
           "the reference orders this pair; the candidate does not"});

  // Carried edges are a SEPARATE superset, not merged into the forward set.
  // A carried edge (x, y, d=1) with x < y is possible and would key identically
  // to the forward edge (x, y) -- yet it is a strictly WEAKER claim, so merging
  // would let a candidate that established only the cross-iteration ordering
  // satisfy a reference demanding the same-iteration one.
  std::set<std::tuple<unsigned, unsigned, unsigned>> haveCarried;
  for (const CarriedEdge &edge : cand.carriedEdges)
    haveCarried.insert({edge.from, edge.to, edge.distance});
  for (const CarriedEdge &edge : ref.carriedEdges)
    if (!haveCarried.count({edge.from, edge.to, edge.distance}))
      violations.push_back(
          {CoverageViolationKind::MissingBackwardOrdering, edge.from, edge.to,
           "the reference orders this pair across iterations; the candidate "
           "does not -- an unprimed loop-carried wait blocks forever",
           edge.distance});

  return violations;
}

void mlir::pto::oracle::printCoverage(llvm::raw_ostream &os,
                                      llvm::StringRef funcName,
                                      const CoverageProfile &p) {
  os << "[sync-cov] func=" << funcName << " anchors=" << p.anchorCount
     << " sig=" << llvm::format_hex(p.anchorSignature, 18)
     << " edge_count=" << p.edges.size() << " edges=";
  for (size_t i = 0; i < p.edges.size(); ++i) {
    if (i)
      os << ",";
    os << p.edges[i].first << "-" << p.edges[i].second;
  }
  // Carried-edge fields are APPENDED, never interleaved: every existing pin ends at the
  // forward `edges=` list and FileCheck matches substrings, so a line that grows
  // a suffix still matches. Reordering would break all of them for no reason.
  os << " bedge_count=" << p.carriedEdges.size() << " bedges=";
  for (size_t i = 0; i < p.carriedEdges.size(); ++i) {
    if (i)
      os << ",";
    os << p.carriedEdges[i].from << "-" << p.carriedEdges[i].to << "@"
       << p.carriedEdges[i].distance;
  }
  os << " barrier_all=" << p.barrierAll << "\n";
}

llvm::StringMap<CoverageProfile>
mlir::pto::oracle::parseCoverage(llvm::StringRef text) {
  llvm::StringMap<CoverageProfile> profiles;

  llvm::SmallVector<llvm::StringRef> lines;
  text.split(lines, '\n');
  for (llvm::StringRef line : lines) {
    line = line.trim();
    if (!line.consume_front("[sync-cov]"))
      continue;

    llvm::StringRef name;
    CoverageProfile p;
    llvm::SmallVector<llvm::StringRef> fields;
    line.split(fields, ' ', -1, /*KeepEmpty=*/false);
    for (llvm::StringRef field : fields) {
      auto [key, value] = field.split('=');
      if (key == "func") {
        name = value;
      } else if (key == "anchors") {
        if (value.getAsInteger(10, p.anchorCount))
          p.anchorCount = 0;
      } else if (key == "sig") {
        if (value.getAsInteger(0, p.anchorSignature))
          p.anchorSignature = 0;
      } else if (key == "edges" && !value.empty()) {
        llvm::SmallVector<llvm::StringRef> pairs;
        value.split(pairs, ',', -1, /*KeepEmpty=*/false);
        for (llvm::StringRef pair : pairs) {
          auto [a, b] = pair.split('-');
          unsigned from = 0, to = 0;
          if (!a.getAsInteger(10, from) && !b.getAsInteger(10, to))
            p.edges.push_back({from, to});
        }
      } else if (key == "barrier_all") {
        // A profile written before this field existed parses to 0, which reads as
        // "no barrier delta" and leaves the verdict untouched -- the field is
        // reported, never differenced into a pass.
        if (value.getAsInteger(10, p.barrierAll))
          p.barrierAll = 0;
      } else if (key == "bedges" && !value.empty()) {
        // `j-i@d`. A profile written before carried edges existed has no `bedges=` field and
        // parses to an empty carried set -- which is the right reading for the
        // hand-written deliberate-mismatch fixtures. No version tag: no .cov
        // file is checked in, so producer and consumer always move together.
        llvm::SmallVector<llvm::StringRef> triples;
        value.split(triples, ',', -1, /*KeepEmpty=*/false);
        for (llvm::StringRef triple : triples) {
          auto [pair, distText] = triple.split('@');
          auto [a, b] = pair.split('-');
          unsigned from = 0, to = 0, distance = 0;
          if (!a.getAsInteger(10, from) && !b.getAsInteger(10, to) &&
              !distText.getAsInteger(10, distance))
            p.carriedEdges.push_back({from, to, distance});
        }
      }
    }
    if (!name.empty()) {
      llvm::sort(p.edges);
      llvm::sort(p.carriedEdges);
      profiles[name] = p;
    }
  }

  return profiles;
}

void mlir::pto::oracle::printCoverageViolations(
    llvm::raw_ostream &os, llvm::StringRef funcName,
    const CoverageProfile &cand, const CoverageProfile &ref,
    llvm::ArrayRef<CoverageViolation> violations) {
  os << "[sync-gate:g1] func=" << funcName
     << " candidate_edges=" << cand.edges.size()
     << " reference_edges=" << ref.edges.size()
     << " violations=" << violations.size() << (violations.empty() ? " OK" : "")
     << " candidate_bedges=" << cand.carriedEdges.size()
     << " reference_bedges=" << ref.carriedEdges.size()
     << " candidate_barrier_all=" << cand.barrierAll
     << " reference_barrier_all=" << ref.barrierAll << "\n";
  // The two barrier counts are reported and nothing is concluded from them. An
  // earlier version added a line reading the difference as coarseness rather than
  // lost coverage, and the deliberate-fault fixtures refused it: an empty allocation
  // and a hand-written missing-carried-ordering both show the reference ahead on
  // barriers, because the reference carries the auto-sync tail barrier and they do
  // not. The same delta therefore appears on the exact fault class this gate exists
  // to catch, and a line explaining it away would have excused them. The reading
  // belongs where it can be qualified; see BARRIER COARSENESS in the header.
  unsigned shown = 0;
  for (const CoverageViolation &v : violations) {
    if ((v.kind == CoverageViolationKind::MissingOrdering ||
         v.kind == CoverageViolationKind::MissingBackwardOrdering) &&
        shown >= 8) {
      os << "  ... and " << (violations.size() - shown) << " more\n";
      break;
    }
    ++shown;
    os << "  !! kind=" << coverageViolationKindName(v.kind);
    if (v.kind == CoverageViolationKind::MissingOrdering)
      os << " anchor#" << v.from << " -> anchor#" << v.to;
    if (v.kind == CoverageViolationKind::MissingBackwardOrdering)
      os << " anchor#" << v.from << " -> anchor#" << v.to << " (d=" << v.distance
         << ")";
    os << " : " << v.detail << "\n";
  }
}

//===----------------------------------------------------------------------===//

namespace {

/// The anchor pipe of a compound's op. Taken from `OpPipeInterface` -- the same
/// source `computeCoverage` keys its edge filtering on.
std::optional<pto::PIPE> anchorPipe(Operation *op) {
  if (auto pipeOp = dyn_cast<pto::OpPipeInterface>(op))
    return pipeOp.getPipe();
  return std::nullopt;
}

std::string anchorPipeName(Operation *op) {
  if (auto pipe = anchorPipe(op))
    return stringifyPIPE(*pipe).str();
  return "?";
}

/// Can `a` and `b` both execute in one pass through the function?
///
/// FALSE only when they sit in DIFFERENT REGIONS of a common `scf.if`: the arms
/// are mutually exclusive, so no ordering between them can ever be required and
/// no sync can ever establish one. This is the same reasoning the carried rules apply to
/// priming -- a set on the `then` arm cannot prime a loop on the `else` arm.
///
/// DELIBERATELY AS NARROW AS THE CORPUS ALLOWS. A too-broad exclusivity
/// predicate silently EXCUSES real dependencies, and that failure is invisible in
/// any "the number went down" check -- so this recognises `scf.if` and nothing
/// else -- not `scf.index_switch`, `cf.cond_br` or `cf.switch`. If another
/// mutually-exclusive-region op appears in the post-sync IR, this must be revisited
/// explicitly rather than generalised on a hunch.
///
/// A one-armed `scf.if` is handled correctly by construction: if `b` is outside
/// the `if` entirely, the `if` never appears as a common ancestor, so the pair is
/// NOT excluded -- `b` always runs while `a` runs conditionally, and the ordering
/// is still required.
bool canBothExecute(Operation *a, Operation *b) {
  llvm::SmallDenseMap<Operation *, Region *, 8> aRegionOf;
  for (Operation *p = a; p && p->getParentOp(); p = p->getParentOp())
    aRegionOf[p->getParentOp()] = p->getParentRegion();
  for (Operation *p = b; p && p->getParentOp(); p = p->getParentOp()) {
    auto it = aRegionOf.find(p->getParentOp());
    if (it == aRegionOf.end() || it->second == p->getParentRegion())
      continue;
    if (isa<scf::IfOp>(p->getParentOp()))
      return false;
  }
  return true;
}

/// The innermost loop enclosing `op`, or null.
Operation *innermostLoop(Operation *op) {
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
    if (isa<LoopLikeOpInterface>(p))
      return p;
  return nullptr;
}

} // namespace

mlir::pto::oracle::SelfCoverageReport
mlir::pto::oracle::computeSelfCoverage(func::FuncOp func) {
  SelfCoverageReport report;

  // --- The CANDIDATE side: what the emitted sync actually orders -------------
  llvm::DenseMap<Operation *, unsigned> anchorIndex;
  CoverageProfile profile = computeCoverage(func, &anchorIndex);
  unsigned n = profile.anchorCount;

  // Transitive closure with distances. reach[i][j] = 0 (same iteration), any
  // distance up to the cap (loop-carried), or unreached. Happens-before is
  // transitive and the raw edge set is not closed -- pipe filtering breaks chains
  // across three pipes -- so closing it is required to avoid false "uncovered".
  // Distances compose additively; only a sum above the cap is dropped.
  // The closure is CAPPED, and the cap direction is load-bearing.
  //
  // `reach[i][j]` holds the SMALLEST distance at which emitted sync orders i
  // before j. It used to be {0, 1, unreached}; rotation needs general N. The cap
  // bounds the table, and ANYTHING ABOVE IT FALLS TO UNREACHED -- i.e. to
  // UNCOVERED, never to covered. A cap that silently credited large N would be a
  // false pass hiding in an implementation detail, which is the one direction
  // that must not happen.
  //
  // 32 matches K, the buffer-id pool, and exceeds every event pool (8 per
  // direction), so no representable rotation depth can reach it.
  constexpr uint8_t kMaxDistance = 32;
  constexpr uint8_t kUnreached = kMaxDistance + 1;
  std::vector<uint8_t> reach(size_t(n) * n, kUnreached);
  auto at = [&](unsigned i, unsigned j) -> uint8_t & {
    return reach[size_t(i) * n + j];
  };
  for (const auto &e : profile.edges)
    if (e.first < n && e.second < n)
      at(e.first, e.second) = 0;
  for (const CarriedEdge &e : profile.carriedEdges)
    if (e.from < n && e.to < n && e.distance >= 1 &&
        e.distance <= kMaxDistance)
      at(e.from, e.to) =
          std::min<uint8_t>(at(e.from, e.to), uint8_t(e.distance));
  for (unsigned k = 0; k < n; ++k)
    for (unsigned i = 0; i < n; ++i) {
      uint8_t ik = at(i, k);
      if (ik == kUnreached)
        continue;
      for (unsigned j = 0; j < n; ++j) {
        uint8_t kj = at(k, j);
        if (kj == kUnreached)
          continue;
        // Distances compose additively: i -before- k at d1 and k -before- j at
        // d2 gives i -before- j at d1+d2. Above the cap the composition is
        // DROPPED, which leaves the pair unreached -> uncovered. Conservative by
        // construction.
        unsigned sum = unsigned(ik) + unsigned(kj);
        if (sum <= kMaxDistance)
          at(i, j) = std::min<uint8_t>(at(i, j), uint8_t(sum));
      }
    }

  // --- The EXPECTED side: the front-end dependency analysis, used DIRECTLY ---
  // Nothing below reads InsertSync's decisions. `IsMemInfoHasDependency` is
  // deliberately NOT called: it layers InsertSync's policy (the tload->tload WAW
  // exemption, the ACC read/read case) on top of DepBetween, and policy is the
  // thing under audit.
  // The A5 PIPE_V intra-pipe guarantee. Read through the SAME predicate the
  // codegen uses (`isTargetArchA5`, PTO/IR/PTO.h) rather than re-deriving it from
  // the module attribute, so the oracle and the emitter cannot drift apart.
  const bool isA5 = pto::isTargetArchA5(func.getOperation());

  MemoryDependentAnalyzer memAnalyzer;
  SyncIRs syncIR;
  Buffer2MemInfoMap buffer2MemInfoMap;
  PTOIRTranslator translator(syncIR, memAnalyzer, buffer2MemInfoMap, func,
                             SyncAnalysisMode::NORMALSYNC);
  translator.Build();

  llvm::SmallVector<pto::CompoundInstanceElement *> compounds;
  for (auto &element : syncIR)
    if (auto *c = dyn_cast<pto::CompoundInstanceElement>(element.get()))
      if (c->elementOp)
        compounds.push_back(c);
  report.compounds = compounds.size();

  auto anchorOf = [&](pto::CompoundInstanceElement *c) -> int {
    auto it = anchorIndex.find(c->elementOp);
    return it == anchorIndex.end() ? -1 : int(it->second);
  };

  // One requirement per (pair, hazard class, distance).
  auto record = [&](pto::CompoundInstanceElement *src,
                    pto::CompoundInstanceElement *dst, llvm::StringRef hazard,
                    unsigned distance) {
    int as = anchorOf(src), ad = anchorOf(dst);
    if (as < 0 || ad < 0) {
      // NOT dropped -- counted. An endpoint with no anchor cannot be expressed
      // in the coverage coordinate system at all.
      ++report.unmappable;
      return;
    }
    ++report.dependencies;
    // Any distance >= 1 is loop-carried; 0 is same-iteration. Counting only
    // d == 1 would hide rotating deps from the report once they carry
    // their true distance.
    if (distance >= 1)
      ++report.carriedDeps;
    bool covered = at(unsigned(as), unsigned(ad)) != kUnreached &&
                   at(unsigned(as), unsigned(ad)) <= distance;
    if (covered) {
      ++report.covered;
      if (distance >= 1)
        ++report.carriedCovered;
      return;
    }
    // (a2) Mutually exclusive arms. Counted, NOT silently dropped: `armExcluded`
    // exists so the exclusion is auditable, and so that
    // `uncovered + armExcluded` reproduces the pre-predicate uncovered count
    // exactly. A predicate that quietly swallowed real dependencies would look
    // identical to a correct one in any total-only check.
    if (!canBothExecute(src->elementOp, dst->elementOp)) {
      ++report.armExcluded;
      return;
    }
    // A5 PIPE_V intra-pipe ordering is treated as a TARGET GUARANTEE. No sync is
    // emitted for such a pair on A5 (`SyncCodegen`, `LoweringSyncToPipe`), so
    // demanding one would make this gate report every such pair on every A5 kernel.
    //
    // NARROW ON PURPOSE -- this is NOT "A5 same-pipe is fine". The uncovered A5
    // population is overwhelmingly V->V, but a residue is `MTE3->MTE3 waw/carried`
    // and cross-pipe `MTE3->V war/carried`, which this test does not excuse.
    // Widening it to "same pipe" would silence them -- re-blinding the gate in the
    // same edit that fixes it.
    //
    // SCOPE. This test asks only whether both endpoints sit on PIPE_V; it does not
    // distinguish a same-allocation pair from a cross-allocation one. Aliasing views
    // are expressed as distinct allocations overlapping in memory once addresses are
    // assigned, so the exemption is broader than a per-allocation test would be.
    //
    // It is deliberately not narrowed: the sync pass treats those cross-allocation
    // pairs identically, excusing all of them and covering none, so narrowing here
    // would reclassify them from excused to uncovered without changing what is
    // emitted.
    if (isA5) {
      auto srcPipe = anchorPipe(src->elementOp);
      auto dstPipe = anchorPipe(dst->elementOp);
      if (srcPipe && dstPipe && *srcPipe == pto::PIPE::PIPE_V &&
          *dstPipe == pto::PIPE::PIPE_V) {
        ++report.archGuaranteed;
        return;
      }
    }
    // (a3) PIPE_S->PIPE_S, on every arch.
    //
    // THIS IS INFERENCE FROM COMPILER BEHAVIOUR, NOT AN ARCHITECTURAL FACT, and it is
    // counted separately from `archGuaranteed` for that reason. No ISA or design
    // document in this tree states the ordering the scalar pipe gives its own ops, and
    // `pipe_barrier(PIPE_S)` is expressible and does reach the emitted code, so the
    // omission below cannot be argued from the barrier being unavailable.
    //
    // What it does rest on is the same assumption encoded independently TWICE:
    // `IsNoNeedToInsertSync` returns early for a same-pipe PIPE_S pair before any
    // dependence analysis runs, deliberately -- the comment beside it declines to
    // short-circuit same-pipe pairs in general -- and GraphSyncSolver's same-pipe
    // barrier path (`SyncSolver.cpp`, guarded by `corePipeSrc == corePipeDst`) returns
    // for PIPE_S unconditionally while gating PIPE_V and PIPE_M on `isRegBasedArch`.
    // Two separate allocators decline to order this pair on every arch, so demanding
    // it here would make the gate unsatisfiable rather than strict.
    //
    // Arch-independent, unlike (a1), because that is how both allocators key it.
    {
      auto srcPipe = anchorPipe(src->elementOp);
      auto dstPipe = anchorPipe(dst->elementOp);
      if (srcPipe && dstPipe && *srcPipe == pto::PIPE::PIPE_S &&
          *dstPipe == pto::PIPE::PIPE_S) {
        ++report.pipeSelfOrdered;
        return;
      }
    }
    if (anchorPipeName(src->elementOp) == anchorPipeName(dst->elementOp))
      ++report.uncoveredSamePipe;
    else
      ++report.uncoveredCrossPipe;
    report.violations.push_back({unsigned(as), unsigned(ad), distance,
                                 hazard.str(),
                                 src->opName.getStringRef().str(),
                                 dst->opName.getStringRef().str(),
                                 anchorPipeName(src->elementOp),
                                 anchorPipeName(dst->elementOp)});
  };

  DepBaseMemInfoPairVec scratch;
  // Accumulates the aliasing pairs across ALL hazard classes for one compound
  // pair, because `scratch` is reset per call and the slot count below must see
  // every pair, not just the last class tested.
  DepBaseMemInfoPairVec allPairs;
  auto dep = [&](pto::CompoundInstanceElement *a,
                 pto::CompoundInstanceElement *b, bool aDef, bool bDef) {
    scratch.clear();
    bool found = memAnalyzer.DepBetween(aDef ? a->defVec : a->useVec,
                                        bDef ? b->defVec : b->useVec, scratch);
    if (found)
      allPairs.append(scratch.begin(), scratch.end());
    return found;
  };

  for (size_t i = 0; i < compounds.size(); ++i) {
    for (size_t j = i + 1; j < compounds.size(); ++j) {
      pto::CompoundInstanceElement *front = compounds[i];
      pto::CompoundInstanceElement *now = compounds[j];

      // Three hazard classes, front -> now, in the SAME iteration.
      allPairs.clear();
      bool raw = dep(front, now, /*aDef=*/true, /*bDef=*/false);  // W then R
      bool war = dep(front, now, /*aDef=*/false, /*bDef=*/true);  // R then W
      bool waw = dep(front, now, /*aDef=*/true, /*bDef=*/true);   // W then W
      if (raw)
        record(front, now, "raw", 0);
      if (war)
        record(front, now, "war", 0);
      if (waw)
        record(front, now, "waw", 0);

      // The SAME alias relation also creates a LOOP-CARRIED requirement when
      // both sit in one loop body: `now` in iteration k conflicts with `front` in
      // iteration k+1, across the back edge. A forward RAW (front writes, now
      // reads) is a carried WAR (now reads, then front writes next iteration),
      // and so on -- which is why the sync pass emits backward pairs at
      // all. This is the requirement only carried edges can satisfy.
      if (!(raw || war || waw))
        continue;
      Operation *loop = innermostLoop(front->elementOp);
      if (!loop || loop != innermostLoop(now->elementOp))
        continue;

      // THE CARRIED DISTANCE IS THE SLOT COUNT, NOT ALWAYS 1.
      //
      // With N multi-buffer slots, iteration k touches slot k%N, so the next
      // access that can CONFLICT is the one that reuses that slot -- iteration
      // k+N. At N=1 this is 1 and nothing changes, which is the whole corpus.
      //
      // WIDEN, NEVER DROP. The forward precedent
      // (`isForwardDepDroppableBySlotAffine`) DROPS a slot-disjoint dep, because
      // forward it needs no sync at all. Mirroring that here would EXCUSE the
      // dependency, and a kernel with NO ROTATION would then pass -- the
      // false-pass direction. InsertSync's own comment guards exactly this:
      // "Back-edge deps stay untouched -- they still need per-slot syncing
      // through the dyn-event-id pipeline." The dep is real; only its DISTANCE
      // moves. There is deliberately no skip/continue on this path.
      //
      // INDEPENDENCE: `baseAddresses` is front-end data, populated by the
      // translator from the program's own `pto.multi_tile_get` slot structure. It
      // is a fact about the PROGRAM, not about anything InsertSync decided.
      unsigned slots = 1;
      for (const auto &pair : allPairs) {
        if (pair.first)
          slots = std::max(slots, unsigned(pair.first->baseAddresses.size()));
        if (pair.second)
          slots = std::max(slots, unsigned(pair.second->baseAddresses.size()));
      }
      if (raw)
        record(now, front, "war/carried", slots);
      if (war)
        record(now, front, "raw/carried", slots);
      if (waw)
        record(now, front, "waw/carried", slots);
    }
  }

  // SELF-PAIRS: one compound against ITSELF across the back edge.
  //
  // The pair loop above starts at `j = i + 1`, so `x` in iteration k is never
  // asked about against `x` in iteration k+1. That requirement is real -- an op
  // that touches the same location on consecutive iterations must be ordered
  // against its own previous execution -- and until it is enumerated here it sits
  // in NO bucket: not `dependencies`, so not `covered`, `uncovered`, `armExcluded`
  // or `archGuaranteed`. A gate that cannot see a requirement reports GREEN when
  // the op realising it is deleted, which is indistinguishable from the
  // requirement having never existed.
  //
  // The coverage side already answers these without change. A same-pipe body
  // barrier records carried edges over every (x, y) anchor pair in its loop span
  // subject to `x < slot || y >= slot`, which is trivially true when x == y, so
  // the barrier contributes a carried SELF-edge at distance 1. Nothing seeds the
  // closure diagonal otherwise -- the reach table starts every cell at
  // `kUnreached` -- so a self-pair is covered exactly when some emitted op really
  // does order the op against its own next execution.
  //
  // Only ops inside a loop are considered: without a back edge there is no second
  // execution to conflict with.
  for (size_t i = 0; i < compounds.size(); ++i) {
    pto::CompoundInstanceElement *x = compounds[i];
    if (!innermostLoop(x->elementOp))
      continue;

    allPairs.clear();
    // Read against write in both orders: across the back edge "k writes then
    // k+1 reads" and "k reads then k+1 writes" are two distinct orderings, and
    // both have to hold. They are recorded separately for that reason, not by
    // oversight.
    bool waw = dep(x, x, /*aDef=*/true, /*bDef=*/true);
    bool raw = dep(x, x, /*aDef=*/true, /*bDef=*/false);
    bool war = dep(x, x, /*aDef=*/false, /*bDef=*/true);
    if (!(waw || raw || war))
      continue;

    unsigned slots = 1;
    for (const auto &pair : allPairs) {
      if (pair.first)
        slots = std::max(slots, unsigned(pair.first->baseAddresses.size()));
      if (pair.second)
        slots = std::max(slots, unsigned(pair.second->baseAddresses.size()));
    }

    if (waw)
      record(x, x, "waw/carried-self", slots);
    if (raw)
      record(x, x, "raw/carried-self", slots);
    if (war)
      record(x, x, "war/carried-self", slots);
  }

  return report;
}

void mlir::pto::oracle::printSelfCoverage(llvm::raw_ostream &os,
                                          llvm::StringRef funcName,
                                          const SelfCoverageReport &report,
                                          unsigned cap) {
  unsigned uncovered = report.violations.size();
  os << "[sync-gate:g1-self] func=" << funcName
     << " compounds=" << report.compounds
     << " deps=" << report.dependencies << " covered=" << report.covered
     << " uncovered=" << uncovered << " (samepipe=" << report.uncoveredSamePipe
     << " crosspipe=" << report.uncoveredCrossPipe << ")"
     << " carried=" << report.carriedDeps << "/"
     << report.carriedCovered << " arm_excluded=" << report.armExcluded
     << " arch_guaranteed=" << report.archGuaranteed
     << " pipe_self_ordered=" << report.pipeSelfOrdered
     << " unmappable=" << report.unmappable
     << (uncovered == 0 ? " OK" : "") << "\n";
  unsigned shown = 0;
  for (const SelfCoverageViolation &v : report.violations) {
    if (cap != 0 && shown >= cap) {
      os << "  ... and " << (uncovered - shown) << " more\n";
      break;
    }
    ++shown;
    os << "  !! kind=uncovered-dependency " << v.hazard << " anchor#"
       << v.srcAnchor << " (" << v.srcName << ") -> anchor#" << v.dstAnchor
       << " (" << v.dstName << ")"
       << " " << v.srcPipe << "->" << v.dstPipe;
    if (v.distance == 1)
      os << " (d=1)";
    os << " : the dependency analysis reports this ordering; no emitted sync "
          "establishes it\n";
  }
}

namespace {

/// Test-only synthetic faults for the classes the IR cannot spell.
///
/// The injection happens *after* extraction, into the gate's own record buffer.
/// It never touches the IR and therefore cannot reach codegen. Its only purpose
/// is to prove the gate rejects what it claims to reject.
/// False when `kind` names no fault, which the caller must report: a silent no-op
/// would leave the gate finding nothing and passing, so a misspelled selector would
/// vacuously satisfy the very test that exists to prove the gate is not vacuous.
bool injectFaultRecords(llvm::StringRef kind,
                        llvm::SmallVectorImpl<IRSyncRecord> &irRecords,
                        llvm::SmallVectorImpl<SyncOpRecord> &syncOpRecords) {
  using TYPE = SyncOperation::TYPE;

  if (kind == "event-oor") {
    // An event id past the end of the intra-core pool. Unrepresentable in IR
    // (`pto.set_flag`'s event_id enum stops at EVENT_ID7) but writable into
    // SyncOperation::eventIds by any allocator.
    SyncOpRecord rec;
    rec.type = "set_event";
    rec.typeCode = static_cast<int>(TYPE::SET_EVENT);
    rec.srcPipe = static_cast<int>(PipelineType::PIPE_V);
    rec.dstPipe = static_cast<int>(PipelineType::PIPE_MTE3);
    rec.eventIds.push_back(9);
    syncOpRecords.push_back(std::move(rec));
    return true;
  }

  if (kind == "block-all-stomp") {
    // A live sync_block_all owns id 14; a block set that also takes 14 stomps it.
    SyncOpRecord all;
    all.type = "sync_block_all";
    all.typeCode = static_cast<int>(TYPE::SYNC_BLOCK_ALL);
    all.srcPipe = static_cast<int>(PipelineType::PIPE_ALL);
    all.dstPipe = static_cast<int>(PipelineType::PIPE_ALL);
    all.eventIds.push_back(static_cast<int>(kBlockSyncAllCubeEventId));
    syncOpRecords.push_back(std::move(all));

    SyncOpRecord set;
    set.type = "sync_block_set";
    set.typeCode = static_cast<int>(TYPE::SYNC_BLOCK_SET);
    set.srcPipe = static_cast<int>(PipelineType::PIPE_MTE3);
    set.dstPipe = static_cast<int>(PipelineType::PIPE_V);
    set.eventIds.push_back(static_cast<int>(kBlockSyncAllCubeEventId));
    set.syncIndex = 1;
    syncOpRecords.push_back(std::move(set));
    return true;
  }

  if (kind == "bufid-oor") {
    // Past the hardware buf_id field, which real IR cannot spell, so the fault is
    // injected into the gate's record buffer instead.
    IRSyncRecord rec;
    rec.opName = "pto.get_buf";
    rec.srcPipe = "PIPE_MTE2";
    rec.dstPipe = "PIPE_MTE2";
    rec.srcPipeType = PipelineType::PIPE_MTE2;
    rec.dstPipeType = PipelineType::PIPE_MTE2;
    rec.bufId = 64;
    rec.order = irRecords.size();
    irRecords.push_back(std::move(rec));
    return true;
  }

  return false;
}

/// Appends a synthetic G2 shape to a record list. Returns false when `kind` names
/// none, which the caller must report: injecting nothing would leave the gate with
/// nothing to find, so a misspelled selector would report clean and prove nothing.
///
/// Both shapes concern the pre-codegen view, and neither can be written as input IR.
/// A compensation record's link to the hazard it primes exists only before codegen,
/// and a rotating hazard's ids are resolved at runtime once emitted.
bool injectInterferenceRecords(llvm::StringRef kind,
                               llvm::SmallVectorImpl<SyncOpRecord> &records) {
  using TYPE = SyncOperation::TYPE;

  if (kind == "comp-conflict") {
    // A compensation record shares its ids with the hazard it primes BY DESIGN, so
    // that pairing must not read as a collision. A collision with any OTHER hazard
    // still must: exempting on `isCompensation` alone would excuse both, which is why
    // the scan keys on the owning hazard rather than on the flag.
    SyncOpRecord comp;
    comp.type = "set_event";
    comp.typeCode = static_cast<int>(TYPE::SET_EVENT);
    comp.srcPipe = static_cast<int>(PipelineType::PIPE_MTE2);
    comp.dstPipe = static_cast<int>(PipelineType::PIPE_V);
    comp.eventIds.push_back(3);
    comp.syncIndex = 900;
    comp.irIndex = 900;
    comp.isCompensation = true;
    comp.compensationOf = 901;
    SyncOpRecord other = comp;
    other.isCompensation = false;
    other.compensationOf = -1;
    other.syncIndex = 902;
    other.irIndex = 901;
    records.push_back(std::move(comp));
    records.push_back(std::move(other));
    return true;
  }

  if (kind == "multi-id-conflict") {
    // The scan must check EVERY id a hazard holds, not just the first. These two
    // records differ on their first id and collide on their second, so a
    // first-id-only scan finds nothing while the full scan reports the collision.
    SyncOpRecord rot;
    rot.type = "set_event";
    rot.typeCode = static_cast<int>(TYPE::SET_EVENT);
    rot.srcPipe = static_cast<int>(PipelineType::PIPE_MTE2);
    rot.dstPipe = static_cast<int>(PipelineType::PIPE_V);
    rot.eventIds.push_back(0);
    rot.eventIds.push_back(1);
    rot.syncIndex = 910;
    rot.irIndex = 910;
    rot.isCompensation = true;
    rot.compensationOf = 911;
    SyncOpRecord clash = rot;
    clash.eventIds.clear();
    clash.eventIds.push_back(7);
    clash.eventIds.push_back(1);
    clash.isCompensation = false;
    clash.compensationOf = -1;
    clash.syncIndex = 912;
    clash.irIndex = 911;
    records.push_back(std::move(rot));
    records.push_back(std::move(clash));
    return true;
  }

  return false;
}

/// Runs G3 on a compiled function. Read-only apart from diagnostics.
struct PTOCheckSyncIdsPass
    : public mlir::pto::impl::PTOCheckSyncIdsBase<PTOCheckSyncIdsPass> {
  unsigned bufidCapacity = 0;
  std::string injectFault;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isDeclaration())
      return;

    DeviceIdLimits limits;
    limits.bufIdCapacity = bufidCapacity;

    llvm::SmallVector<IRSyncRecord> irRecords = oracle::extractFromIR(func);
    llvm::SmallVector<SyncOpRecord> syncOpRecords;
    if (!injectFault.empty() &&
        !injectFaultRecords(injectFault, irRecords, syncOpRecords)) {
      func.emitError() << "unknown --check-sync-ids-inject-fault selector '"
                       << injectFault
                       << "'; expected event-oor, block-all-stomp or bufid-oor";
      signalPassFailure();
      return;
    }

    auto irViolations = oracle::checkDeviceIdLegality(irRecords, limits);
    llvm::SmallVector<IdViolation> syncOpViolations;
    if (!syncOpRecords.empty())
      syncOpViolations = oracle::checkDeviceIdLegality(syncOpRecords, limits);

    oracle::emitReport([&](llvm::raw_ostream &os) {
      oracle::printIdViolations(os, func.getSymName(), "ir", irRecords.size(),
                                irViolations);
      if (!syncOpRecords.empty())
        oracle::printIdViolations(os, func.getSymName(), "syncops",
                                  syncOpRecords.size(), syncOpViolations);
    });

    size_t total = irViolations.size() + syncOpViolations.size();
    if (total > 0) {
      func.emitError() << "G3 device-id legality: " << total
                       << " illegal sync id(s)";
      signalPassFailure();
    }
  }
};

/// G2: runs the non-interference check on a compiled function.
struct PTOCheckSyncInterferencePass
    : public mlir::pto::impl::PTOCheckSyncInterferenceBase<
          PTOCheckSyncInterferencePass> {
  std::string injectFault;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isDeclaration())
      return;

    auto records = oracle::extractFromIR(func);
    auto violations = oracle::checkNonInterference(records);
    oracle::emitReport([&](llvm::raw_ostream &os) {
      oracle::printInterferenceViolations(os, func.getSymName(), "ir",
                                          records.size(), violations);
    });

    // The two synthetic shapes concern the pre-codegen view, so they are judged on
    // their own record list rather than mixed into the emitted-op scan above.
    if (!injectFault.empty()) {
      llvm::SmallVector<SyncOpRecord> synthetic;
      if (!injectInterferenceRecords(injectFault, synthetic)) {
        func.emitError() << "unknown --check-sync-interference-inject-fault "
                            "selector '"
                         << injectFault
                         << "'; expected comp-conflict or multi-id-conflict";
        signalPassFailure();
        return;
      }
      auto synthViolations = oracle::checkNonInterference(synthetic);
      oracle::emitReport([&](llvm::raw_ostream &os) {
        oracle::printInterferenceViolations(os, func.getSymName(), "syncops",
                                            synthetic.size(), synthViolations);
      });
      if (unsigned errors = oracle::countErrors(synthViolations)) {
        func.emitError() << "G2 non-interference: " << errors
                         << " illegal sync id use(s)";
        signalPassFailure();
        return;
      }
    }
    // Warnings (a soft rule, such as an id spanning more than two pipes) are
    // reported but do not fail the gate; only errors do.
    if (unsigned errors = oracle::countErrors(violations)) {
      func.emitError() << "G2 non-interference: " << errors
                       << " illegal sync id use(s)";
      signalPassFailure();
    }
  }
};

/// G1-self: ABSOLUTE, not differential. Asserts that every dependency the
/// front-end analysis reports is ordered by emitted sync. No reference file --
/// the expected set comes from `DepBetween`, so there is nothing to compare
/// against and nothing inherited from InsertSync's choices.
struct PTOCheckSyncSelfCoveragePass
    : public mlir::pto::impl::PTOCheckSyncSelfCoverageBase<
          PTOCheckSyncSelfCoveragePass> {
  /// Violations printed per function; 0 = all. Threaded in from the driver rather
  /// than read from a global, so nothing that links this library inherits a flag.
  unsigned maxViolations = 8;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isDeclaration())
      return;
    auto report = oracle::computeSelfCoverage(func);
    oracle::emitReport([&](llvm::raw_ostream &os) {
      oracle::printSelfCoverage(os, func.getSymName(), report, maxViolations);
    });
    if (!report.violations.empty()) {
      func.emitError() << "G1-self coverage: " << report.violations.size()
                       << " dependency ordering(s) are not established by any "
                          "emitted sync";
      signalPassFailure();
    }
  }
};


/// G1 reference side: writes the happens-before edges this run's sync induces.
struct PTODumpSyncCoveragePass
    : public mlir::pto::impl::PTODumpSyncCoverageBase<PTODumpSyncCoveragePass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isDeclaration())
      return;
    auto profile = oracle::computeCoverage(func);
    oracle::emitReport([&](llvm::raw_ostream &os) {
      oracle::printCoverage(os, func.getSymName(), profile);
    });
  }
};

/// G1: every ordering the reference establishes must also be established here.
struct PTOCheckSyncCoveragePass
    : public mlir::pto::impl::PTOCheckSyncCoverageBase<PTOCheckSyncCoveragePass> {
  std::string referenceFile;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isDeclaration())
      return;

    auto bufferOr = llvm::MemoryBuffer::getFile(referenceFile);
    if (!bufferOr) {
      func.emitError() << "G1 coverage: cannot read reference profile '"
                       << referenceFile << "': " << bufferOr.getError().message();
      return signalPassFailure();
    }

    llvm::StringMap<CoverageProfile> reference =
        oracle::parseCoverage((*bufferOr)->getBuffer());
    llvm::StringRef name = func.getSymName();
    CoverageProfile candidate = oracle::computeCoverage(func);

    auto it = reference.find(name);
    if (it == reference.end()) {
      oracle::emitReport([&](llvm::raw_ostream &os) {
        os << "[sync-gate:g1] func=" << name << " violations=1\n  !! kind="
           << oracle::coverageViolationKindName(
                  CoverageViolationKind::MissingReference)
           << " : no reference profile for this function in '" << referenceFile
           << "'\n";
      });
      func.emitError() << "G1 coverage: no reference profile for '" << name
                       << "'";
      return signalPassFailure();
    }

    const CoverageProfile &ref = it->second;
    auto violations = oracle::checkCoverageSuperset(candidate, ref);
    oracle::emitReport([&](llvm::raw_ostream &os) {
      oracle::printCoverageViolations(os, name, candidate, ref, violations);
    });

    if (!violations.empty()) {
      // An anchor mismatch is not "missing orderings" -- nothing is missing, the
      // two runs simply did not compile the same program. Say so.
      if (violations.front().kind == CoverageViolationKind::AnchorMismatch)
        func.emitError() << "G1 coverage: candidate and reference did not see "
                            "the same program (anchor signature mismatch)";
      else
        func.emitError() << "G1 coverage: " << violations.size()
                         << " ordering(s) the reference establishes are missing";
      signalPassFailure();
    }
  }
};

/// Writes this run's kernel-body sync-op counts. Read-only.
struct PTODumpSyncCountsPass
    : public mlir::pto::impl::PTODumpSyncCountsBase<PTODumpSyncCountsPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isDeclaration())
      return;
    auto profile = oracle::computeSyncCounts(oracle::extractFromIR(func));
    oracle::emitReport([&](llvm::raw_ostream &os) {
      oracle::printSyncCounts(os, func.getSymName(), profile);
    });
  }
};

/// G4: compares this run's counts against a reference run's profile.
struct PTOCheckSyncCountFloorPass
    : public mlir::pto::impl::PTOCheckSyncCountFloorBase<
          PTOCheckSyncCountFloorPass> {
  std::string referenceFile;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isDeclaration())
      return;

    auto bufferOr = llvm::MemoryBuffer::getFile(referenceFile);
    if (!bufferOr) {
      func.emitError() << "G4 sync-count floor: cannot read reference profile '"
                       << referenceFile << "': " << bufferOr.getError().message();
      return signalPassFailure();
    }

    llvm::StringMap<SyncCountProfile> reference =
        oracle::parseSyncCounts((*bufferOr)->getBuffer());

    llvm::StringRef name = func.getSymName();
    SyncCountProfile candidate =
        oracle::computeSyncCounts(oracle::extractFromIR(func));

    auto it = reference.find(name);
    if (it == reference.end()) {
      // Not a pass: a gate that skips what it cannot find is not a gate.
      oracle::emitReport([&](llvm::raw_ostream &os) {
        os << "[sync-gate:g4] func=" << name << " violations=1\n  !! kind="
           << oracle::countViolationKindName(CountViolationKind::MissingReference)
           << " : no reference profile for this function in '" << referenceFile
           << "'\n";
      });
      func.emitError() << "G4 sync-count floor: no reference profile for '"
                       << name << "'";
      return signalPassFailure();
    }

    const SyncCountProfile &ref = it->second;
    auto violations = oracle::checkSyncCountFloor(candidate, ref);
    oracle::emitReport([&](llvm::raw_ostream &os) {
      oracle::printCountViolations(os, name, candidate, ref, violations);
    });

    if (!violations.empty()) {
      func.emitError() << "G4 sync-count floor: " << violations.size()
                       << " regression(s) against the reference profile";
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass>
mlir::pto::createPTOCheckSyncIdsPass(unsigned bufidCapacity,
                                     llvm::StringRef injectFault) {
  auto pass = std::make_unique<PTOCheckSyncIdsPass>();
  pass->bufidCapacity = bufidCapacity;
  pass->injectFault = injectFault.str();
  return pass;
}

std::unique_ptr<Pass>
mlir::pto::createPTOCheckSyncInterferencePass(llvm::StringRef injectFault) {
  auto pass = std::make_unique<PTOCheckSyncInterferencePass>();
  pass->injectFault = injectFault.str();
  return pass;
}

std::unique_ptr<Pass>
mlir::pto::createPTOCheckSyncSelfCoveragePass(unsigned maxViolations) {
  auto pass = std::make_unique<PTOCheckSyncSelfCoveragePass>();
  pass->maxViolations = maxViolations;
  return pass;
}

std::unique_ptr<Pass> mlir::pto::createPTODumpSyncCoveragePass() {
  return std::make_unique<PTODumpSyncCoveragePass>();
}

std::unique_ptr<Pass>
mlir::pto::createPTOCheckSyncCoveragePass(llvm::StringRef referenceFile) {
  auto pass = std::make_unique<PTOCheckSyncCoveragePass>();
  pass->referenceFile = referenceFile.str();
  return pass;
}

std::unique_ptr<Pass> mlir::pto::createPTODumpSyncCountsPass() {
  return std::make_unique<PTODumpSyncCountsPass>();
}

std::unique_ptr<Pass>
mlir::pto::createPTOCheckSyncCountFloorPass(llvm::StringRef referenceFile) {
  auto pass = std::make_unique<PTOCheckSyncCountFloorPass>();
  pass->referenceFile = referenceFile.str();
  return pass;
}
