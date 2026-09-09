// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <memory>
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"


using namespace mlir;
using namespace mlir::pto;

namespace {

// ==========================================================
// 更严格的活跃性分析
// ==========================================================
 
// 辅助：判断是否是实质性的资源操作 (Resource Op)
// Wait 和 Set 不算作实质性操作。
// 只有真正消耗计算或带宽的指令才算"活跃"。
bool isResourceOp(Operation *op, Attribute targetPipe) {
    if (auto loadOp = dyn_cast<pto::TLoadOp>(op)) {
        return pto::PipeAttr::get(op->getContext(), pto::PIPE::PIPE_MTE2) == targetPipe;
    }
    if (auto storeOp = dyn_cast<pto::TStoreOp>(op)) {
        return pto::PipeAttr::get(op->getContext(), pto::PIPE::PIPE_MTE3) == targetPipe;
    }
    if (auto addfOp = dyn_cast<pto::TAddOp>(op)) {
        return pto::PipeAttr::get(op->getContext(), pto::PIPE::PIPE_V) == targetPipe;
    }
    return false;
}
 
// 递归检查 Region 内是否有实质性操作
// 用于深入 scf.if / scf.for 内部查找
bool isPipeUsedInRegion(Region &region, Attribute targetPipe) {
    for (Block &block : region) {
        for (Operation &op : block) {
            // 1. 如果是实质性操作，返回 True
            if (isResourceOp(&op, targetPipe)) {
              return true;
            }

            // 2. 递归检查嵌套 (if/for)
            for (Region &nestedRegion : op.getRegions()) {
                if (isPipeUsedInRegion(nestedRegion, targetPipe)) {
                  return true;
                }
            }
        }
    }
    return false;
}
 
// 向后扫描：检查 targetPipe 在当前 Block 后续是否"真正"活跃
// WaitOp 不再被视为活跃标志。
// 如果一个 Pipe 后面只剩 Wait，说明它已经完成了工作，发给它的信号是多余的。
static bool hasPipelineActivityAfterOp(Operation *parentOp, Attribute targetPipe) {
    Block *parentBlock = parentOp ? parentOp->getBlock() : nullptr;
    if (!parentBlock) {
      return false;
    }
    for (auto it = std::next(parentOp->getIterator()); it != parentBlock->end(); ++it) {
        if (isResourceOp(&*it, targetPipe)) {
          return true;
        }
        if (it->getNumRegions() > 0) {
          return true;
        }
        if (isa<func::ReturnOp>(&*it)) {
          return false;
        }
    }
    return false;
}

bool isPipelineActiveFuture(Block *block, Block::iterator startIt, Attribute targetPipe) {
    for (auto it = startIt; it != block->end(); ++it) {
        Operation *op = &*it;

// 1. 遇到实质性操作 -> 活跃
        if (isResourceOp(op, targetPipe)) {
          return true;
        }

        // [注意] 这里故意跳过了 WaitOp 的检查。
        // WaitOp 只是同步原语，不代表该 Pipeline 在"干活"。

        // 2. 递归检查嵌套区域 (scf.if, scf.for)
        for (Region &region : op->getRegions()) {
            if (isPipeUsedInRegion(region, targetPipe)) {
              return true;
            }
        }

        // 3. 处理 Terminator (跨 Block 检查)
        if (op->hasTrait<OpTrait::IsTerminator>()) {
            // 如果是 Return，肯定死了
            if (isa<func::ReturnOp>(op)) {
              return false;
            }
            return hasPipelineActivityAfterOp(block->getParentOp(), targetPipe);
        }
    }
    return false;
}


// ==========================================================
// Op classification helpers
// ==========================================================
namespace {

// Maps resource ops to the pipe they occupy (MTE2 load / MTE3 store / V add).
class OpPipeTable {
public:
  explicit OpPipeTable(MLIRContext *ctx)
      : mte2(pto::PipeAttr::get(ctx, pto::PIPE::PIPE_MTE2)),
        mte3(pto::PipeAttr::get(ctx, pto::PIPE::PIPE_MTE3)),
        vec(pto::PipeAttr::get(ctx, pto::PIPE::PIPE_V)) {}

  Attribute pipeFor(Operation *op) const {
    if (isa<pto::TLoadOp>(op)) {
      return mte2;
    }
    if (isa<pto::TStoreOp>(op)) {
      return mte3;
    }
    if (isa<pto::TAddOp>(op)) {
      return vec;
    }
    return {};
  }

private:
  Attribute mte2;
  Attribute mte3;
  Attribute vec;
};

static bool getSetSyncPipes(Operation *op, Attribute &src, Attribute &dst) {
  if (auto setOp = dyn_cast<pto::SetFlagOp>(op)) {
    src = setOp.getSrcPipe();
    dst = setOp.getDstPipe();
    return true;
  }
  StringRef opName = op->getName().getStringRef();
  if (opName == "pto.set_flag_dyn" || opName == "pto.set_flag_d") {
    auto srcAttr = op->getAttrOfType<pto::PipeAttr>("src_pipe");
    auto dstAttr = op->getAttrOfType<pto::PipeAttr>("dst_pipe");
    if (!srcAttr || !dstAttr) {
      return false;
    }
    src = srcAttr;
    dst = dstAttr;
    return true;
  }
  return false;
}

static bool getWaitSyncDst(Operation *op, Attribute &dst) {
  if (auto waitOp = dyn_cast<pto::WaitFlagOp>(op)) {
    dst = waitOp.getDstPipe();
    return true;
  }
  StringRef opName = op->getName().getStringRef();
  if (opName == "pto.wait_flag_dyn" || opName == "pto.wait_flag_d") {
    auto dstAttr = op->getAttrOfType<pto::PipeAttr>("dst_pipe");
    if (!dstAttr) {
      return false;
    }
    dst = dstAttr;
    return true;
  }
  return false;
}

// Rule A: dead pipeline - nothing active for this pipe after the barrier.
// Rule B: clean pipeline - the pipe has no dirty state to protect.
// Rule C: subsumed by a following set on the same pipe (set implies barrier).
bool isRedundantBarrier(pto::BarrierOp barrierOp, Block *block,
                        Block::iterator it,
                        const llvm::DenseSet<Attribute> &intraPipeDirtySet) {
  Attribute bPipe = barrierOp.getPipe();
  if (!isPipelineActiveFuture(block, std::next(it), bPipe)) {
    return true;
  }
  if (!intraPipeDirtySet.count(bPipe)) {
    return true;
  }
  auto nextIt = std::next(it);
  if (nextIt != block->end()) {
    Attribute nextSrc;
    Attribute nextDst;
    if (getSetSyncPipes(&*nextIt, nextSrc, nextDst) && nextSrc == bPipe) {
      return true;
    }
  }
  return false;
}

} // namespace

// ==========================================================
// Pass 实现
// ==========================================================
struct PTORemoveRedundantBarrierPass : public PassWrapper<PTORemoveRedundantBarrierPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTORemoveRedundantBarrierPass)

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpPipeTable pipes(&getContext());
    llvm::SmallVector<Operation *> opsToErase;

    func.walk([&](Block *block) {
      // 记录 Block 内脏状态 (Intra-Block Dirty State)
      // 用于判断是否需要发广播
      llvm::DenseSet<Attribute> intraPipeDirtySet;

      for (auto it = block->begin(); it != block->end(); ++it) {
        Operation *op = &*it;

        // === 1. 状态更新 ===
        Attribute pipe = pipes.pipeFor(op);
        if (pipe) {
          intraPipeDirtySet.insert(pipe);
          continue;
        }

        // === 2. Barrier 消除 ===
        if (auto barrierOp = dyn_cast<pto::BarrierOp>(op)) {
          if (isRedundantBarrier(barrierOp, block, it, intraPipeDirtySet)) {
            opsToErase.push_back(op);
            continue;
          }
          // 如果 Barrier 留下了，管线变干净
          intraPipeDirtySet.erase(barrierOp.getPipe());
        }

        // === 3. Wait 消除 (幽灵 Wait 消除) ===
        // Dead Consumer: dst 后面没有 Resource Op，等待毫无意义。
        Attribute waitDst;
        if (getWaitSyncDst(op, waitDst) &&
            !isPipelineActiveFuture(block, std::next(it), waitDst)) {
          opsToErase.push_back(op);
          continue;
        }

        // === 4. Set 消除 (死信 & 陈旧广播消除) ===
        Attribute setSrc;
        Attribute setDst;
        if (getSetSyncPipes(op, setSrc, setDst)) {
          // Dead Receiver: dst 后面没有 Resource Op，发信号没人用。
          // isPipelineActiveFuture 忽略 WaitOp，因此仅剩 Wait <Src,Dst>
          // 时这里与上面的 Wait 消除形成闭环。
          if (!isPipelineActiveFuture(block, std::next(it), setDst)) {
            opsToErase.push_back(op);
            continue;
          }
          // Stale Broadcast: Src 未脏过 (scf.if 中 MTE2->MTE3 冗余广播)。
          if (!intraPipeDirtySet.count(setSrc)) {
            opsToErase.push_back(op);
          }
        }
      }
    });

    for (Operation *op : opsToErase) {
      op->erase();
    }
  }
};
 
} // namespace
 
namespace mlir {
namespace pto {
std::unique_ptr<Pass> createPTORemoveRedundantBarrierPass() {
  return std::make_unique<PTORemoveRedundantBarrierPass>();
}
} // namespace pto
} // namespace mlir
