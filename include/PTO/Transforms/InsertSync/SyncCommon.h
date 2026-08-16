// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_SYNC_COMMON_H
#define MLIR_DIALECT_PTO_TRANSFORMS_SYNC_COMMON_H
 
#include <utility>
#include <deque>
#include <string>
#include <map>
#include <algorithm>
#include <memory>
#include <optional>
#include <vector>
 
#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
 
// 引入 PTO Dialect 定义 (获取 AddressSpace 等)
#include "PTO/IR/PTO.h"
 
#define MAX_MULTI_BUFFER_NUM 16
 
namespace mlir {
namespace pto {
 
enum class SyncAnalysisMode {
  NORMALSYNC, // 核内同步 (Intra-Core): 解决流水线冒险
  BLOCKSYNC   // 核间同步 (Inter-Core): 解决 CV 分离后的通讯
};
 
/// Pipeline 类型定义 (对应硬件流水线)
/// 必须与 PTOIRTranslator 中的 getOpPipeline 保持一致
enum class PipelineType : uint32_t {
  PIPE_S = 0,
  PIPE_V = 1,
  PIPE_M = 2,
  PIPE_MTE1 = 3,   // L1 -> L0
  PIPE_MTE2 = 4,   // GM -> L1 (Load)
  PIPE_MTE3 = 5,   // L1 -> GM (Store)
  PIPE_ALL = 6,    // Barrier
  
  // 预留/扩展
  PIPE_MTE4 = 7,
  PIPE_MTE5 = 8,
  PIPE_V2 = 9,
  PIPE_FIX = 10,
  
  // 虚拟 Pipe (如果有)
  VIRTUAL_PIPE_MTE2_L1A = 11,
  VIRTUAL_PIPE_MTE2_L1B = 12,
  
  PIPE_NUM = 13,
  PIPE_UNASSIGNED = 99,
  PIPE_LAST = PIPE_UNASSIGNED
};
 
// 辅助函数：获取 Pipeline 数量
static constexpr unsigned getPipeNum() {
  return static_cast<unsigned>(PipelineType::PIPE_LAST);
}
 
/// 核心类型定义：对应计算单元类型
enum class TCoreType {
  VECTOR,
  CUBE,
  CUBE_OR_VECTOR,
  CUBE_AND_VECTOR
};
 
/// Meminfo of the target buffer
/// 用于追踪 Buffer 的别名和根节点
struct BaseMemInfo {
  BaseMemInfo(
      Value baseBuffer, Value rootBuffer, pto::AddressSpace scope,
      SmallVector<uint64_t> baseAddresses, uint64_t allocateSize,
      bool hasKnownPhysicalAddresses = false,
      bool aliasesUnknownRange = false)
      : baseBuffer(baseBuffer), rootBuffer(rootBuffer), scope(scope),
        baseAddresses(std::move(baseAddresses)), allocateSize(allocateSize),
        hasKnownPhysicalAddresses(hasKnownPhysicalAddresses),
        aliasesUnknownRange(aliasesUnknownRange) {}
 
  /// baseBuffer: 当前操作直接使用的 Buffer (可能是 View 或 Alias)
  Value baseBuffer;
  /// rootBuffer: 最底层的分配源头 (alloc_tile 或 Kernel Argument)
  Value rootBuffer;
  
  pto::AddressSpace scope;
  SmallVector<uint64_t> baseAddresses; // 用于 Offset 分析
  uint64_t allocateSize;
  // PlanMemory materializes static local allocations as physical addresses.
  // This distinguishes their physical addresses from root-relative offsets.
  bool hasKnownPhysicalAddresses;
  // Address-preserving provenance can be lost when a pointer is round-tripped
  // through integer IR. When set, this memory object aliases any range in the
  // same address space.
  bool aliasesUnknownRange;
 
  bool areVectorEqual(const SmallVector<uint64_t>& vec1,
                      const SmallVector<uint64_t>& vec2) const {
    if (vec1.size() != vec2.size()) return false;
    for (size_t i = 0; i < vec1.size(); ++i) {
      if (vec1[i] != vec2[i]) return false;
    }
    return true;
  }
 
  bool operator==(const BaseMemInfo &other) const {
    if (!areVectorEqual(baseAddresses, other.baseAddresses)) return false;
    if (rootBuffer != other.rootBuffer) return false;
    if (scope != other.scope) return false;
    if (hasKnownPhysicalAddresses != other.hasKnownPhysicalAddresses)
      return false;
    if (aliasesUnknownRange != other.aliasesUnknownRange)
      return false;
    // allocateSize 和 baseBuffer 的严格相等性在某些别名分析中可能太强了，
    // 但为了保持原有逻辑，先保留。重点是 rootBuffer 必须一致。
    if (allocateSize != other.allocateSize) return false;
    if (baseBuffer != other.baseBuffer) return false;
    return true;
  }
 
  std::unique_ptr<BaseMemInfo> clone() const {
    return std::make_unique<BaseMemInfo>(
        baseBuffer, rootBuffer, scope, baseAddresses, allocateSize,
        hasKnownPhysicalAddresses, aliasesUnknownRange);
  }

  std::unique_ptr<BaseMemInfo> clone(Value cloneBaseBuffer) const {
    return std::make_unique<BaseMemInfo>(
        cloneBaseBuffer, rootBuffer, scope, baseAddresses, allocateSize,
        hasKnownPhysicalAddresses, aliasesUnknownRange);
  }
};
 
using DepBaseMemInfoPairVec =
    SmallVector<std::pair<const BaseMemInfo *, const BaseMemInfo *>>;
 
// 表示一个具体的同步指令 (Set, Wait, Barrier)
class SyncOperation {
public:
  enum class TYPE {
    SET_EVENT,
    WAIT_EVENT,
    PIPE_BARRIER,
    PIPE_BARRIER_CUBE,
    PIPE_BARRIER_VECTOR,
    SYNC_BLOCK_SET,
    SYNC_BLOCK_WAIT,
    SYNC_BLOCK_ALL,
  };

  /// Which hardware resource backs this sync -- and therefore, per the buf-id
  /// design revision, WHERE it may be placed.
  ///
  /// This is deliberately a DIFFERENT axis from TYPE. `TYPE` is the op *shape*
  /// (a set, a wait, a barrier) and is decided by the analysis. `MECHANISM` is
  /// the resource *class* that backs it, and is decided by the allocator:
  ///
  ///   EVENT   Per-(src,dst) private pool of 8 flags. Carried at a SyncIR
  ///           position, so `MoveSyncState` may hoist it out of a branch or
  ///           loop. This is what every in-tree sync is today.
  ///   BUFID   Shared rotating pool of K buffer-ID tokens (K=0 on A3, 32 on A5).
  ///           A token's get/rel counters are BIDIRECTIONAL: one token discharges
  ///           the forward RAW *and* the reverse WAR. Two consequences, and they
  ///           are the reason this enum exists rather than a bool:
  ///             - no backward-matched pair is needed (`UpdateBackwardMatchSync`
  ///               lives in SyncEventIdAllocation, which the unified allocator
  ///               never runs -- so there is nothing to delete);
  ///             - it MUST be emitted OP-ANCHORED, bracketing the access, not at
  ///               a hoisted SyncIR position.
  ///           Set by the allocator when it routes a hazard off events.
  ///   BARRIER Spill. Carries no id.
  ///   BLOCK   Cross-core block sync (reserved ids 14/15). Not routed by this
  ///           allocator; present so a block sync can never be mistaken for an
  ///           event.
  ///
  /// Every value EXCEPT BUFID is derivable from TYPE, and the constructor does
  /// derive it (see `DefaultMechanismFor`), so no construction site has to
  /// remember to set it and a barrier can never silently claim to be an event.
  /// BUFID is the one value that is *not* derivable -- buffer-ID sync does not
  /// go through `SyncOperation` at all today -- and it is precisely the routing
  /// decision the unified allocator exists to make.
  enum class MECHANISM { EVENT, BUFID, BARRIER, BLOCK };

  bool isCompensation = false;
  /// For a compensation op (head-set / tail-wait), the `kSyncIndex_` of the in-body
  /// hazard it primes and drains; -1 otherwise. Needed because a compensation pair is
  /// created as its OWN sync group, so `GetSyncIndex()` does not identify its owner --
  /// and G2 must be able to tell "these two records are halves of one hazard" from
  /// "two different hazards collide on an id". Without it the exemption would have to
  /// excuse every compensation record, which would drop real conflicts.
  int compensationOf = -1;

  static const int kNullEventId{-1};
 
public:
  SmallVector<int> eventIds;
  // Root buffers participating in the dependency that created this sync pair.
  // These are kept for allocation/widening heuristics and debug printing; set/
  // wait redundancy pruning is based on the pipe pair semantics.
  //
  // `rootBuffer` is the ROOT ALLOCATION, which is much COARSER than a buffer: many
  // distinct tiles share one root. It is therefore NOT a buffer identity, and must
  // not be used as one -- see `depMemInfos` below.
  SmallVector<Value> depRootBuffers;
  // The BaseMemInfo objects behind this dependency, kept so a consumer can
  // recover the TILE identity `(scope, baseAddr, size)` rather than the coarse
  // root above. Owned by the translator's `Buffer2MemInfoMap`; these are raw
  // pointers into that arena, which every analysis in the pipeline shares.
  //
  // This exists because `InsertSyncAnalysis` HAS these pointers when it builds a
  // sync pair and used to discard them, keeping only the roots -- which left the
  // unified allocator unable to tell which buffer a hazard was actually on.
  SmallVector<const BaseMemInfo *> depMemInfos;
  bool uselessSync{false};
  int eventIdNum{1};
  // For multi-buffer dyn-event sync: the slot SSA expression at this access
  // site. set_flag_dyn / wait_flag_dyn use `slotSSAExpr % slotCount` as the
  // hardware event-id index. Empty when this sync is single-buffer.
  Value slotSSAExpr;
  uint32_t slotCount{1};
  Value lowestCommonAncestorBuffer{nullptr};
  int reuseCntForWiden{0};
  bool reallocatedLoopHeadTailSync{false};
  bool autoSyncTailBarrier{false};
  TCoreType syncCoreType{TCoreType::CUBE_OR_VECTOR};
  Value block_sync_event_value{nullptr};
 
public:
  SyncOperation(TYPE type, pto::PipelineType srcPipe, pto::PipelineType dstPipe,
                unsigned kSyncIndex, unsigned syncIRIndex,
                std::optional<int> forEndIndex, bool isComp = false)
      : isCompensation(isComp), eventIds({}), type_(type),
        mechanism_(DefaultMechanismFor(type)), srcPipe_(srcPipe),
        dstPipe_(dstPipe), kSyncIndex_(kSyncIndex), syncIRIndex_(syncIRIndex),
        forEndIndex_(forEndIndex) {};

  ~SyncOperation() = default;

  std::unique_ptr<SyncOperation> GetMatchSync(unsigned index) const;

  TYPE GetType() const { return type_; }

  /// The resource class backing this sync. Defaults from TYPE; only the
  /// allocator moves a sync off its default (today: onto BUFID).
  MECHANISM GetMechanism() const { return mechanism_; }
  void SetMechanism(MECHANISM m) { mechanism_ = m; }

  /// The mechanism a given TYPE implies. Kept next to the enum so the two can
  /// never drift; `SetPipeAll` and the constructor are its only callers.
  static MECHANISM DefaultMechanismFor(TYPE t);
  pto::PipelineType GetSrcPipe() const { return srcPipe_; }
  pto::PipelineType GetDstPipe() const { return dstPipe_; }
  
  // PTO 暂时不需要 Virtual Pipe 映射，直接返回原值
  pto::PipelineType GetActualSrcPipe() const { return srcPipe_; }
  pto::PipelineType GetActualDstPipe() const { return dstPipe_; }
 
  SmallVector<int> GetEventIDs() const { return eventIds; }
  unsigned GetSyncIndex() const { return kSyncIndex_; }
  unsigned GetSyncIRIndex() const { return syncIRIndex_; }
  void SetSyncIRIndex(unsigned index) { syncIRIndex_ = index; }
  std::optional<int> GetForEndIndex() const { return forEndIndex_; }
  void SetDepSyncIRIndex(unsigned index) { depSyncIRIndex_ = index; }
  unsigned GetDepSyncIRIndex() const { return depSyncIRIndex_; }
  void SetType(TYPE syncType) { type_ = syncType; }
  bool operator==(const SyncOperation &other) const;
 
  bool isSyncSetType() const;
  bool isSyncWaitType() const;
  bool isBarrierType() const;
 
  static std::string TypeName(TYPE t);
  static std::string MechanismName(MECHANISM m);
  std::string GetCoreTypeName(TCoreType t) const;
 
  using SyncOperations =
      SmallVector<SmallVector<std::unique_ptr<SyncOperation>>>;
 
  // 设置为 PipeAll (用于资源耗尽时的降级)
  void SetPipeAll();
  bool IsAutoSyncTailBarrier() const { return autoSyncTailBarrier; }
  void MarkAutoSyncTailBarrier() { autoSyncTailBarrier = true; }
 
private:
  TYPE type_;
  // Private on purpose: it must stay consistent with type_ (see SetPipeAll,
  // which downgrades both together). Write it through SetMechanism only.
  MECHANISM mechanism_;
  pto::PipelineType srcPipe_;
  pto::PipelineType dstPipe_;
  const unsigned kSyncIndex_;
  unsigned syncIRIndex_;
  std::optional<int> forEndIndex_{};
  unsigned depSyncIRIndex_{0};
};

SmallVector<const void *>
canonicalizeSyncDepRoots(const SmallVector<Value> &roots);

bool hasSameSyncDepRoots(const SyncOperation *lhs, const SyncOperation *rhs);

/// The two clauses of `MoveForSync`'s wait rule, factored out so there is ONE
/// source of truth for "would the loop motion hoist this wait out of the loop".
///
/// `MoveSyncState::PlanMoveOutWaitSync` produces that decision; the unified
/// allocator consumes it when deciding whether a hazard may take a buffer-id
/// token, because a wait hoisted out of a loop runs once where a token pays per
/// iteration.
///
/// Duplicating the rule in the allocator would let the two drift apart
/// silently: a change to the loop motion would stop being reflected in the
/// routing predicate, and the allocator would start routing hazards whose
/// event form was strictly cheaper. Both callers therefore go through these.
///
/// Clause (1): a wait carried by THIS loop is pinned to the body.
inline bool moveForSyncWaitPinnedByCarry(const SyncOperation *wait,
                                         unsigned loopEnd) {
  return wait->GetForEndIndex().has_value() &&
         static_cast<unsigned>(wait->GetForEndIndex().value()) == loopEnd;
}

/// Clause (2): the paired set lies outside the loop, so the dependency does not
/// come from within it and the wait may be lifted to the loop head.
inline bool moveForSyncSetOutsideLoop(const SyncOperation *pairedSet,
                                      unsigned loopBegin, unsigned loopEnd) {
  return pairedSet->GetSyncIRIndex() > loopEnd ||
         pairedSet->GetSyncIRIndex() < loopBegin;
}

/// The full decision. Only for callers that already hold both ops; the mover
/// itself evaluates the clauses separately because clause (1) must short
/// circuit before it looks the paired set up.
inline bool moveForSyncHoistsWait(const SyncOperation *wait,
                                  const SyncOperation *pairedSet,
                                  unsigned loopBegin, unsigned loopEnd) {
  return !moveForSyncWaitPinnedByCarry(wait, loopEnd) &&
         moveForSyncSetOutsideLoop(pairedSet, loopBegin, loopEnd);
}

/// True when `sync` lowers to `pto.set_flag_dyn` / `pto.wait_flag_dyn`, i.e.
/// its rendezvous is keyed on the runtime slot index rather than on a fixed
/// event id.
///
/// Such a sync orders ONLY the accesses that resolve to its own event lane
/// (`slotSSAExpr % slotCount`); it does not serialize its (src, dst) pipe pair
/// the way a static flag does. Every place that reasons about one sync
/// "covering" another must therefore exclude it, or a second access through a
/// different slot is left unguarded (issue #1118).
///
/// `slotSSAExpr` is the discriminator, NOT `GetForEndIndex()`. Slot keying is
/// only ever established for a back-edge dependency (InsertSyncOperation), so
/// it may be tempting to test for one -- but the synthetic prologue-prime and
/// epilogue-drain pairs that SyncEventIdAllocation::UpdateBackwardMatchSync
/// derives from such a pair inherit both `forEndIndex` and `eventIdNum` while
/// deliberately carrying no slot, and they lower to STATIC flags. Keying off
/// the back edge would misclassify exactly those.
inline bool isSlotKeyedSync(const SyncOperation *sync) {
  return sync && sync->eventIdNum > 1 && sync->slotSSAExpr;
}
 
using SyncOps = std::deque<SyncOperation *>;
 
// SyncIR 的基类节点
class InstanceElement {
public:
  Operation *elementOp = nullptr;
  SyncOps pipeBefore;
  SyncOps pipeAfter;
  enum class KindTy { COMPOUND, LOOP, BRANCH, PLACE_HOLDER };
 
public:
  virtual ~InstanceElement() = default;
  unsigned GetIndex() const { return kIndex; }
  KindTy GetKind() const { return kKindTy; }
  static bool RemoveSync(SyncOps &syncVector, const SyncOperation *sync);
 
protected:
  const unsigned kIndex;
  InstanceElement(KindTy kind, unsigned index)
      : kIndex(index), kKindTy(kind) {};
 
private:
  const KindTy kKindTy;
};
 
class PlaceHolderInstanceElement : public InstanceElement {
public:
  unsigned parentScopeId;
  PlaceHolderInstanceElement(unsigned index, unsigned parentScopeId)
      : InstanceElement(KindTy::PLACE_HOLDER, index),
        parentScopeId(parentScopeId) {};
 
  std::unique_ptr<PlaceHolderInstanceElement> Clone() const;
  static bool classof(const InstanceElement *e);
 
  // [新增] 标记这是一个为了同步而虚拟出来的 Else 块
  bool isVirtualElse = false; 
  // [新增] 保存父 Op (scf.if)，以便后续操作
  Operation* parentIfOp = nullptr;
};
 
enum class KindOfLoop { LOOP_BEGIN, LOOP_END };
 
class LoopInstanceElement : public InstanceElement {
public:
  unsigned beginId;
  unsigned endId;
 
public:
  LoopInstanceElement(unsigned index, unsigned beginId, unsigned endId,
                      KindOfLoop loopKind = KindOfLoop::LOOP_BEGIN)
      : InstanceElement(KindTy::LOOP, index), beginId(beginId), endId(endId),
        kLoopKind(loopKind) {}
 
  ~LoopInstanceElement() override = default;
  std::unique_ptr<InstanceElement> CloneFor(KindOfLoop loopKind) const;
  KindOfLoop getLoopKind() const { return kLoopKind; }
  static bool classof(const InstanceElement *e);
  bool ignore_block_sync_move_out{false};
 
private:
  const KindOfLoop kLoopKind;
};
 
enum class KindOfBranch { IF_BEGIN, ELSE_BEGIN, IF_END };
 
class BranchInstanceElement : public InstanceElement {
public:
  unsigned beginId;
  unsigned branchId;
  unsigned endId{0};
 
public:
  BranchInstanceElement(unsigned index, unsigned beginId,
                        KindOfBranch branchKind = KindOfBranch::IF_BEGIN)
      : InstanceElement(KindTy::BRANCH, index), beginId(beginId),
        branchId(beginId), kBranchKind(branchKind) {}
 
  ~BranchInstanceElement() override = default;
  std::unique_ptr<BranchInstanceElement>
  CloneBranch(KindOfBranch branchKind) const;
 
  KindOfBranch getBranchKind() const { return kBranchKind; }
  static bool classof(const InstanceElement *e);
 
  BranchInstanceElement(unsigned index, unsigned beginId, unsigned branchId,
                        unsigned endId,
                        KindOfBranch branchKind = KindOfBranch::IF_BEGIN)
      : InstanceElement(KindTy::BRANCH, index), beginId(beginId),
        branchId(branchId), endId(endId), kBranchKind(branchKind) {}
 
private:
  const KindOfBranch kBranchKind;
};
 
// Unit Flag 状态 (用于指令级同步优化)
enum class UNIT_FLAG {
  DISABLED,
  ENABLED_WITH_UPDATE,
  ENABLED_ONLY_FIRST_ITER,
  ENABLED_ONLY_LAST_ITER,
  ENABLED_ONLY_FIRST_AND_LAST_ITERS
};
 
// 核心节点：代表一个计算或搬运指令
class CompoundInstanceElement : public InstanceElement {
public:
  // Def/Use 列表 (指向 BaseMemInfo)
  SmallVector<const BaseMemInfo *> defVec;
  SmallVector<const BaseMemInfo *> useVec;
 
  // 该指令归属的 Pipeline
  pto::PipelineType kPipeValue;
 
  OperationName opName;
  TCoreType compoundCoreType{TCoreType::CUBE_OR_VECTOR};
  SyncOperation *BwdPipeMPipeMTE1SyncPtr{nullptr};
 
  UNIT_FLAG unitFlagModeAsSet{UNIT_FLAG::DISABLED};
  UNIT_FLAG unitFlagModeAsWait{UNIT_FLAG::DISABLED};
  CompoundInstanceElement *linkedUnitFlagCompAsSet{nullptr};
  CompoundInstanceElement *linkedUnitFlagCompAsWait{nullptr};
  // Macro-like operations may be represented by multiple SyncIR compounds that
  // point at the same source operation. The phase id distinguishes those
  // synthetic compounds without changing the source IR.
  int macroOpInstanceId{-1};
 
public:
  CompoundInstanceElement(unsigned index,
                          SmallVector<const BaseMemInfo *> defVec,
                          SmallVector<const BaseMemInfo *> useVec,
                          const pto::PipelineType PipeValue, OperationName opName)
      : InstanceElement(KindTy::COMPOUND, index), defVec(std::move(defVec)),
        useVec(std::move(useVec)), kPipeValue(PipeValue), opName(opName) {}
 
  ~CompoundInstanceElement() override = default;
 
  static bool classof(const InstanceElement *e);
  UNIT_FLAG getUnitFlagMode() const;
  
  // PTO 暂时简化，去掉复杂的 UnitFlag 条件生成逻辑，或者稍后在 CPP 中适配
  std::optional<mlir::Value> getUnitFlagCond(Location loc, OpBuilder &rewriter);
};
 
using SyncIRs = SmallVector<std::unique_ptr<InstanceElement>>;
using SyncOperations = SmallVector<SmallVector<std::unique_ptr<SyncOperation>>>;
using Buffer2MemInfoMap =
    llvm::DenseMap<Value, llvm::SmallVector<std::unique_ptr<BaseMemInfo>>>;
 
// Utilities
bool checkAllParentLoopsAreForLoops(Operation *op);
void checkSyncIRIndex(const SyncIRs &syncIR, int index);
void checkCondition(bool condition, const std::string &message);
 
} // namespace pto
} // namespace mlir
 
#endif // MLIR_DIALECT_PTO_TRANSFORMS_SYNC_COMMON_H
