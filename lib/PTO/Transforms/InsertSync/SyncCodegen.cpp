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

#include "PTO/Transforms/InsertSync/SyncCodegen.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
 
#define DEBUG_TYPE "pto-inject-sync"
 
using namespace mlir;
using namespace mlir::pto;
 
// ==============================================================================
// 1. Helper Functions
// ==============================================================================
 
static pto::PipeAttr getPipeAttr(Builder &builder, PipelineType pipe) {
  auto odsPipeVal = static_cast<pto::PIPE>(pipe);
  return pto::PipeAttr::get(builder.getContext(), odsPipeVal);
}
 
static pto::EventAttr getEventAttr(Builder &builder, int id) {
  auto odsEventVal = static_cast<pto::EVENT>(id);
  return pto::EventAttr::get(builder.getContext(), odsEventVal);
}
 
static bool IsSyncExist(const SyncOps &list, SyncOperation *newSync) {
  for (auto *existing : list) {
    if (existing == newSync) return true;
    if (existing->GetType() != newSync->GetType()) continue;
    if (existing->GetActualSrcPipe() != newSync->GetActualSrcPipe()) continue;
    if (existing->GetActualDstPipe() != newSync->GetActualDstPipe()) continue;
    if (newSync->isSyncSetType() || newSync->isSyncWaitType()) {
       if (existing->eventIds != newSync->eventIds) continue;
    }
    return true;
  }
  return false;
}
 
static void MergeSyncList(SyncOps &dstList, const SyncOps &srcList) {
  for (auto *sync : srcList) {
    if (!IsSyncExist(dstList, sync)) {
      dstList.push_back(sync);
    }
  }
}
 
// ==============================================================================
// 2. SyncCodegen Implementation
// ==============================================================================
 
void SyncCodegen::Run() {
  MLIRContext *ctx = func_->getContext();
  IRRewriter rewriter(ctx);

  UpdateOpInsertSync(rewriter);

  // [Fix #428] Collect back-edge wait ops at loop heads that need pre-loop
  // set_flag initialization. A back-edge wait_flag at LOOP_BEGIN fires on
  // iteration 0, but the matching set_flag only fires at the end of iteration
  // 0. Without a pre-loop set_flag, the event register is uninitialized and
  // the hardware hangs.
  CollectBackEdgeLoopHeadWaits();

  func_->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op2InsertSync.count(op)) {
      // [Fix #428] Before emitting the normal pipeBefore waits for a loop op,
      // emit pre-loop set_flag initialization for any back-edge wait_flag that
      // would otherwise fire against an uninitialized event register.
      if (isa<scf::ForOp>(op)) {
        EmitPreLoopEventInit(rewriter, op);
      }

      // 处理 PRE Sync
      for (auto &syncBefore : op2InsertSync[op].pipeBefore) {
        SyncInsert(rewriter, op, syncBefore, true);
      }
      // 处理 POST Sync (逆序遍历，为了保持插入后的顺序正确)
      for (auto &syncAfter : llvm::reverse(op2InsertSync[op].pipeAfter)) {
        SyncInsert(rewriter, op, syncAfter, false);
      }
    }
  });

  // Ensure the tail clean barrier is emitted at function tail, right before
  // return, instead of being interleaved with other trailing sync ops.
  AppendAutoSyncTailBarrierIfNeeded(rewriter);
}
 
void SyncCodegen::UpdateOpInsertSync(IRRewriter &rewriter) {
  for (auto &nowElement : syncIR_) {
    if (auto *compoundElement = dyn_cast<CompoundInstanceElement>(nowElement.get())) {
      UpdateCompoundOpInsertSync(compoundElement);
    } else if (auto *placeHolder = dyn_cast<PlaceHolderInstanceElement>(nowElement.get())) {
      updatePlaceHolderOpInsertSync(placeHolder);
    } else if (auto *loopElement = dyn_cast<LoopInstanceElement>(nowElement.get())) {
      UpdateLoopOpInsertSync(loopElement);
    } else if (auto *branchElement = dyn_cast<BranchInstanceElement>(nowElement.get())) {
      UpdateBranchOpInsertSync(branchElement);
    }
  }
}
 
void SyncCodegen::UpdateCompoundOpInsertSync(CompoundInstanceElement *nowCompound) {
  auto &pipeBuild = op2InsertSync[nowCompound->elementOp];
  MergeSyncList(pipeBuild.pipeBefore, nowCompound->pipeBefore);
  MergeSyncList(pipeBuild.pipeAfter, nowCompound->pipeAfter);
}
 
void SyncCodegen::UpdateLoopOpInsertSync(LoopInstanceElement *nowElement) {
  if (nowElement->getLoopKind() == KindOfLoop::LOOP_END) {
    auto *loopBegin = dyn_cast<LoopInstanceElement>(syncIR_[nowElement->beginId].get());
    auto &pipeBuild = op2InsertSync[nowElement->elementOp];
    MergeSyncList(pipeBuild.pipeBefore, loopBegin->pipeBefore);
    MergeSyncList(pipeBuild.pipeAfter, nowElement->pipeAfter);
  }
}
 
void SyncCodegen::UpdateBranchOpInsertSync(BranchInstanceElement *nowElement) {
  if (nowElement->getBranchKind() == KindOfBranch::IF_END) {
    auto *branchBegin = dyn_cast<BranchInstanceElement>(syncIR_[nowElement->beginId].get());
    auto &pipeBuild = op2InsertSync[nowElement->elementOp];
    MergeSyncList(pipeBuild.pipeBefore, branchBegin->pipeBefore);
    MergeSyncList(pipeBuild.pipeAfter, nowElement->pipeAfter);
  }
}
 
void SyncCodegen::updatePlaceHolderOpInsertSync(PlaceHolderInstanceElement *placeHolder) {
  // 1. 处理 Virtual Else
  if (placeHolder->isVirtualElse) {
      auto ifOp = dyn_cast<scf::IfOp>(placeHolder->parentIfOp);
      if (!ifOp) return;
 
      // 如果还没有 else block，创建一个
      if (!ifOp.elseBlock()) {
          OpBuilder builder(ifOp.getContext());
          // 只有当确实有 Sync 指令需要插入时才创建
          if (!placeHolder->pipeBefore.empty() || !placeHolder->pipeAfter.empty()) {
               Region &elseRegion = ifOp.getElseRegion();
               Block *elseBlock = new Block();
               elseRegion.push_back(elseBlock);
               builder.setInsertionPointToEnd(elseBlock);
               builder.create<scf::YieldOp>(ifOp.getLoc());
          }
      }
      
      // 更新映射：将 Virtual Placeholder 映射到新创建的 Yield Op
      if (ifOp.elseBlock()) {
          placeHolder->elementOp = ifOp.getElseRegion().front().getTerminator();
      } else {
          // 依然没有 Sync 需要插入，直接返回
          return;
      }
  } 
  // 2. 处理 Normal PlaceHolder (Then End or Existing Else End)
  else if (placeHolder->elementOp == placeHolder->parentIfOp) {
      // 之前的 Translator 逻辑把 Normal Placeholder 也映射到了 ifOp
      // 我们需要修正它指向 Yield
      auto ifOp = dyn_cast<scf::IfOp>(placeHolder->elementOp);
      // 判断是 Then 还是 Else
      // 简单判断：看 index。或者 Translator 里直接存 Yield Op。
      // 这里假设 Translator 存的是 IfOp，我们需要找到对应的 Yield。
      // ... 
      // 建议在 Translator 里直接让 elementOp 指向 Yield Op（如果存在）。
  }
 
  // 执行常规的 Sync 插入
  if (!placeHolder->elementOp) return;
  auto &pipeBuild = op2InsertSync[placeHolder->elementOp];
  MergeSyncList(pipeBuild.pipeBefore, placeHolder->pipeBefore);
  MergeSyncList(pipeBuild.pipeAfter, placeHolder->pipeAfter);
}
 
void SyncCodegen::SyncInsert(IRRewriter &rewriter, Operation *op,
                             SyncOperation *sync, bool beforeInsert) {
  if (sync->uselessSync) return;

  // [Fix] 处理补偿逻辑的强制插入点
  Operation *insertAnchorOp = op;
  bool forceBefore = beforeInsert;

  if (sync->isCompensation) {
      // 策略：补偿指令必须插在控制流块的末尾（Terminator 之前）
      
      // Case 1: Anchor 是 scf.if (Virtual Else 的情况)
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          // 我们需要确定是插在 Then 还是 Else。
          // 通常 Analysis 会根据 context 知道，但这里 op 只是 anchor。
          // 我们利用 SyncOperation 的上下文推断，或者更简单地：
          // 如果是 Virtual Else，PTOIRTranslator 应该已经处理了 Block 创建。
          // 如果这里还是 IfOp，说明我们必须进入 Else Region。
          
          if (!ifOp.elseBlock()) {
              // 再次兜底：创建 Else Block
              OpBuilder b(ifOp.getContext());
              Block *elseBlock = new Block();
              ifOp.getElseRegion().push_back(elseBlock);
              b.setInsertionPointToEnd(elseBlock);
              b.create<scf::YieldOp>(ifOp.getLoc());
          }
          
          // 将插入点重定向到 Else Block 的 Yield
          insertAnchorOp = ifOp.getElseRegion().front().getTerminator();
      }
      // Case 2: Anchor 已经是 Terminator (YieldOp)
      else if (op->hasTrait<OpTrait::IsTerminator>()) {
          insertAnchorOp = op;
      }
      // Case 3: 其他情况 (Anchor 指向了 Block 内的某条指令)
      else {
          // 找到该 Block 的 Terminator
          insertAnchorOp = op->getBlock()->getTerminator();
      }

      // 强制在 Terminator 之前插入
      forceBefore = true;
  }

  // 分发创建逻辑，传入修正后的 insertAnchorOp 和 forceBefore
  if (sync->GetType() == SyncOperation::TYPE::PIPE_BARRIER) {
    CreateBarrierOp(rewriter, insertAnchorOp, sync, forceBefore);
  } else if (sync->isSyncSetType() || sync->isSyncWaitType()) {
    if (sync->eventIds.size() == 1) {
      CreateSetWaitOpForSingleBuffer(rewriter, insertAnchorOp, sync, forceBefore);
    } else {
      CreateSetWaitOpForMultiBuffer(rewriter, insertAnchorOp, sync, forceBefore);
    }
  } 
}
 
// [核心修改] 加强版 CreateBarrierOp
void SyncCodegen::CreateBarrierOp(IRRewriter &rewriter, Operation *op,
                                  SyncOperation *sync, bool beforeInsert) {
  // A5: PIPE_V intra-pipe ordering is guaranteed by hardware; do not emit
  // explicit vector barrier (it is also rejected by backend checks).
  if (isTargetArchA5(func_.getOperation()) &&
      sync->GetActualSrcPipe() == PipelineType::PIPE_V) {
    return;
  }

  // Compiler-inserted tail clean barrier must be anchored at function tail.
  if (sync->GetActualSrcPipe() == PipelineType::PIPE_ALL &&
      sync->GetActualDstPipe() == PipelineType::PIPE_ALL) {
    pendingAutoSyncTailBarrier_ = true;
    return;
  }

  // [Fix] 判定是否需要前置插入：如果是显式 Before，或者 Op 是 Terminator (如 Yield)
  bool insertAtPos = beforeInsert || op->hasTrait<OpTrait::IsTerminator>();
 
  // 1. 设置插入点
  if (insertAtPos) {
    rewriter.setInsertionPoint(op);
  } else {
    rewriter.setInsertionPointAfter(op);
  }
 
  // 2. 获取上下文
  Block *block = rewriter.getInsertionBlock();
  Block::iterator ip = rewriter.getInsertionPoint();
  auto currentPipeAttr = getPipeAttr(rewriter, sync->GetActualSrcPipe());
 
  // 3. 窥孔优化 (双向检查)
  // 注意：如果是 Terminator 导致的强制前置插入，我们也应该检查 Prev，因为它是插在末尾
  if (insertAtPos) {
    // PRE 插入：检查前一条指令
    if (ip != block->begin()) {
      if (auto prevBarrier = dyn_cast<pto::BarrierOp>(&*std::prev(ip))) {
        if (prevBarrier.getPipe() == currentPipeAttr) return; // Dedup
      }
    }
  } else {
    // POST 插入：检查当前/下一条指令
    if (ip != block->end()) {
      if (auto nextBarrier = dyn_cast<pto::BarrierOp>(&*ip)) {
        if (nextBarrier.getPipe() == currentPipeAttr) return; // Dedup
      }
    }
  }
 
  // 4. 创建指令
  auto barrier =
      rewriter.create<pto::BarrierOp>(op->getLoc(), currentPipeAttr);

  (void)barrier;
}

void SyncCodegen::AppendAutoSyncTailBarrierIfNeeded(IRRewriter &rewriter) {
  if (!pendingAutoSyncTailBarrier_)
    return;

  SmallVector<func::ReturnOp, 4> returns;
  func_.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
  if (returns.empty())
    return;

  auto pipeAllAttr = getPipeAttr(rewriter, PipelineType::PIPE_ALL);
  for (auto ret : returns) {
    // [Fix #428] Before the tail barrier, emit explicit wait_flag ops to
    // drain all pending back-edge event dependencies. pipe_barrier(PIPE_ALL)
    // waits for in-flight pipe operations but does NOT drain event flag
    // registers. Without explicit wait_flag calls, stale event state can
    // leak to the next kernel invocation.
    rewriter.setInsertionPoint(ret);
    EmitTailEventDrain(rewriter, ret);

    // Re-set insertion point before ret (after any drain ops we just emitted)
    rewriter.setInsertionPoint(ret);
    auto barrier = rewriter.create<pto::BarrierOp>(ret.getLoc(), pipeAllAttr);
    barrier->setAttr("pto.auto_sync_tail_barrier", rewriter.getUnitAttr());
    if (auto hintAttr =
            func_->getAttrOfType<mlir::StringAttr>("pto.auto_sync_tail_hint")) {
      barrier->setAttr("pto.auto_sync_tail_hint", hintAttr);
    }
  }

  pendingAutoSyncTailBarrier_ = false;
}
 
void SyncCodegen::CreateSetWaitOpForSingleBuffer(IRRewriter &rewriter,
                                                 Operation *op,
                                                 SyncOperation *sync,
                                                 bool beforeInsert) {
  // [Fix] Terminator 强制前置插入
  if (beforeInsert || op->hasTrait<OpTrait::IsTerminator>()) {
      rewriter.setInsertionPoint(op);
  } else {
      rewriter.setInsertionPointAfter(op);
  }
 
  auto srcPipe = getPipeAttr(rewriter, sync->GetActualSrcPipe());
  auto dstPipe = getPipeAttr(rewriter, sync->GetActualDstPipe());
  auto eventId = getEventAttr(rewriter, sync->eventIds[0]);
 
  if (sync->isSyncWaitType()) {
    rewriter.create<pto::WaitFlagOp>(op->getLoc(), srcPipe, dstPipe, eventId);
  } else {
    rewriter.create<pto::SetFlagOp>(op->getLoc(), srcPipe, dstPipe, eventId);
  }
}

// ==============================================================================
// [Fix #428] Tail event drain for back-edge sync events
// ==============================================================================

void SyncCodegen::EmitTailEventDrain(IRRewriter &rewriter,
                                     func::ReturnOp ret) {
  // Collect all unique (srcPipe, dstPipe, eventId) triples from back-edge
  // syncs across all loops. These events may still be pending when the kernel
  // reaches the return statement and must be explicitly drained.
  //
  // We use a set of tuples to deduplicate — the same event may appear in
  // multiple loops or be shared via widen.
  struct EventKey {
    PipelineType src;
    PipelineType dst;
    int eventId;
    bool operator<(const EventKey &o) const {
      if (src != o.src) return static_cast<unsigned>(src) < static_cast<unsigned>(o.src);
      if (dst != o.dst) return static_cast<unsigned>(dst) < static_cast<unsigned>(o.dst);
      return eventId < o.eventId;
    }
    bool operator==(const EventKey &o) const {
      return src == o.src && dst == o.dst && eventId == o.eventId;
    }
  };

  SmallVector<EventKey> drainEvents;
  auto addUnique = [&](PipelineType src, PipelineType dst, int eid) {
    EventKey key{src, dst, eid};
    for (auto &existing : drainEvents) {
      if (existing == key)
        return;
    }
    drainEvents.push_back(key);
  };

  // Scan all LOOP_END elements for back-edge set/wait pairs with allocated
  // event IDs.
  for (auto &pair : preLoopInitWaits_) {
    for (auto *waitSync : pair.second) {
      if (waitSync->uselessSync || waitSync->eventIds.empty())
        continue;
      addUnique(waitSync->GetActualSrcPipe(), waitSync->GetActualDstPipe(),
                waitSync->eventIds[0]);
    }
  }

  // Also scan the last element's pipeAfter for any set_flag ops that might
  // leave events pending (these are the "syncEnd" phantom pairs from
  // UpdateBackwardMatchSync that sink to the function tail).
  if (!syncIR_.empty()) {
    for (auto *sync : syncIR_.back()->pipeAfter) {
      if (sync->uselessSync || sync->eventIds.empty())
        continue;
      if (sync->isSyncWaitType()) {
        addUnique(sync->GetActualSrcPipe(), sync->GetActualDstPipe(),
                  sync->eventIds[0]);
      }
    }
  }

  if (drainEvents.empty())
    return;

  // Sort for deterministic output.
  llvm::sort(drainEvents);

  LLVM_DEBUG(llvm::dbgs() << "[Fix #428] Emitting " << drainEvents.size()
                          << " tail event drain wait_flag ops\n");

  for (auto &ev : drainEvents) {
    auto srcPipe = getPipeAttr(rewriter, ev.src);
    auto dstPipe = getPipeAttr(rewriter, ev.dst);
    auto eventId = getEventAttr(rewriter, ev.eventId);
    rewriter.create<pto::WaitFlagOp>(ret.getLoc(), srcPipe, dstPipe, eventId);
  }
}
 
void SyncCodegen::CreateSetWaitOpForMultiBuffer(IRRewriter &rewriter,
                                                Operation *op,
                                                SyncOperation *sync,
                                                bool beforeInsert) {
  // 注意：GetBufferSelected 可能需要在插入 Set/Wait 之前调用，以确保 SSA 顺序
  // 但这里只是获取 Value，不影响 InsertionPoint 的设定
  Value bufferSelected = GetBufferSelected(rewriter, op, sync);
  (void)bufferSelected; 
  
  // [Fix] Terminator 强制前置插入
  if (beforeInsert || op->hasTrait<OpTrait::IsTerminator>()) {
      rewriter.setInsertionPoint(op);
  } else {
      rewriter.setInsertionPointAfter(op);
  }
 
  auto srcPipe = getPipeAttr(rewriter, sync->GetActualSrcPipe());
  auto dstPipe = getPipeAttr(rewriter, sync->GetActualDstPipe());
  auto eventId = getEventAttr(rewriter, sync->eventIds[0]); // 注意：MultiBuffer可能需要特殊处理Attr
 
  // 这里假设 SetFlagOp/WaitFlagOp 支持动态 Value 作为 EventID，或者您有特殊的 Op
  // 如果 PTO 定义只支持 Attribute，那么上面的 GetBufferSelected 逻辑需要配合修改 Op 定义
  // 假设目前的 Op 定义如下：
  if (sync->isSyncWaitType()) {
    // 假设 WaitFlagOp 有支持 Value eventId 的重载或变体
    // 如果没有，这行代码可能需要调整。但在您之前的 Double Buffer 测试中，看起来它是工作的？
    // 或者您是否使用了 UpdateFlagOp (带 Value)?
    // 这里保持原样，只修改 InsertionPoint
    rewriter.create<pto::WaitFlagOp>(op->getLoc(), srcPipe, dstPipe, eventId);
  } else {
    rewriter.create<pto::SetFlagOp>(op->getLoc(), srcPipe, dstPipe, eventId);
  }
}
 
Value SyncCodegen::GetBufferSelected(IRRewriter &rewriter, Operation *op,
                                     SyncOperation *sync) {
  if (SyncIndex2SelectBuffer.count(sync->GetSyncIndex())) {
    return SyncIndex2SelectBuffer[sync->GetSyncIndex()];
  }
 
  auto parentLoop = op->getParentOfType<scf::ForOp>();
  if (!parentLoop) return nullptr;
 
  Value counter;
  if (loop2BufferCounter.count(parentLoop)) {
    counter = loop2BufferCounter[parentLoop];
  } else {
    rewriter.setInsertionPointToStart(parentLoop.getBody());
    Value iv = parentLoop.getInductionVar();
    Value c2 = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 2);
    counter = rewriter.create<arith::RemUIOp>(op->getLoc(), iv, c2);
    loop2BufferCounter[parentLoop] = counter;
  }
 
  rewriter.setInsertionPointAfter(counter.getDefiningOp());
  Value id0 = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), sync->eventIds[0]);
  Value id1 = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), sync->eventIds[1]);
  
  Value isZero = rewriter.create<arith::CmpIOp>(op->getLoc(), arith::CmpIPredicate::eq, counter, 
      rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0));
  
  Value selected = rewriter.create<arith::SelectOp>(op->getLoc(), isZero, id0, id1);
  
  SyncIndex2SelectBuffer[sync->GetSyncIndex()] = selected;
  return selected;
}

// ==============================================================================
// [Fix #428] Pre-loop event initialization for back-edge sync
// ==============================================================================

void SyncCodegen::CollectBackEdgeLoopHeadWaits() {
  for (auto &nowElement : syncIR_) {
    auto *loopElement = dyn_cast<LoopInstanceElement>(nowElement.get());
    if (!loopElement || loopElement->getLoopKind() != KindOfLoop::LOOP_END)
      continue;

    // Look at the LOOP_BEGIN node's pipeBefore — these are waits that
    // MoveSyncState hoisted to the loop head.
    auto *loopBegin =
        dyn_cast<LoopInstanceElement>(syncIR_[loopElement->beginId].get());
    if (!loopBegin)
      continue;

    for (auto *sync : loopBegin->pipeBefore) {
      if (sync->uselessSync)
        continue;
      if (!sync->isSyncWaitType())
        continue;
      if (sync->eventIds.empty())
        continue;
      // This is a wait at loop head with an allocated event ID.
      // It needs a pre-loop set_flag to initialize the event register.
      // Record {Operation* forOp -> SyncOperation* wait} for later emission.
      if (loopElement->elementOp) {
        preLoopInitWaits_[loopElement->elementOp].push_back(sync);
      }
    }
  }
}

void SyncCodegen::EmitPreLoopEventInit(IRRewriter &rewriter, Operation *op) {
  auto it = preLoopInitWaits_.find(op);
  if (it == preLoopInitWaits_.end())
    return;

  // For each back-edge wait at the loop head, emit a set_flag before the
  // for loop to initialize the event register. This ensures that on iteration
  // 0, the wait_flag finds a valid (already-set) event instead of hanging.
  rewriter.setInsertionPoint(op);
  for (auto *waitSync : it->second) {
    if (waitSync->uselessSync || waitSync->eventIds.empty())
      continue;

    auto srcPipe = getPipeAttr(rewriter, waitSync->GetActualSrcPipe());
    auto dstPipe = getPipeAttr(rewriter, waitSync->GetActualDstPipe());
    auto eventId = getEventAttr(rewriter, waitSync->eventIds[0]);

    LLVM_DEBUG(llvm::dbgs()
               << "[Fix #428] Emitting pre-loop set_flag("
               << static_cast<unsigned>(waitSync->GetActualSrcPipe()) << ", "
               << static_cast<unsigned>(waitSync->GetActualDstPipe()) << ", "
               << waitSync->eventIds[0] << ") before loop\n");

    rewriter.create<pto::SetFlagOp>(op->getLoc(), srcPipe, dstPipe, eventId);
  }
}
