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
// 同步原语识别 (Sync Primitive Recognition)
// ==========================================================

// 资源操作所属的管线 (MTE2/MTE3/V)；非资源操作返回空 Attribute。
static Attribute getOpPipe(Operation *op) {
  if (isa<pto::TLoadOp>(op)) {
    return pto::PipeAttr::get(op->getContext(), pto::PIPE::PIPE_MTE2);
  }
  if (isa<pto::TStoreOp>(op)) {
    return pto::PipeAttr::get(op->getContext(), pto::PIPE::PIPE_MTE3);
  }
  if (isa<pto::TAddOp>(op)) {
    return pto::PipeAttr::get(op->getContext(), pto::PIPE::PIPE_V);
  }
  return {};
}

// 识别 Set 类同步原语 (set_flag 及其动态形式)，提取 src/dst 管线。
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

// 识别 Wait 类同步原语 (wait_flag 及其动态形式)，提取 dst 管线。
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

// ==========================================================
// 消除规则 (Elimination Rules)
// ==========================================================

// Barrier 消除规则 A/B/C。返回 true 表示该 Barrier 冗余可删：
//   A: Dead Pipeline —— 后面没活干了，保护空气没有意义
//   B: Clean Pipeline —— 管线本来就是干净的
//   C: Subsumed by Set —— 紧跟 Set，Set 隐含 Barrier
static bool isRedundantBarrier(pto::BarrierOp barrierOp, Block *block,
                               Block::iterator it,
                               llvm::DenseSet<Attribute> &intraPipeDirtySet) {
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
  // 如果 Barrier 留下了，管线变干净
  intraPipeDirtySet.erase(bPipe);
  return false;
}

// Wait 消除 (幽灵 Wait 消除)：Dead Consumer 规则。
// 如果 dst 后面没有 Resource Op，这个 Wait 是毫无意义的阻塞。
// 即使逻辑上需要等，但如果等完不干活，等它干嘛？
static bool isRedundantWait(Operation *op, Block *block, Block::iterator it,
                            Attribute waitDst) {
  return !isPipelineActiveFuture(block, std::next(it), waitDst);
}

// Set 消除规则 A/B。返回 true 表示该 Set 冗余可删：
//   A: Dead Receiver (死信) —— dst 后面没有 Resource Op，发信号也没人用。
//      注意：因为 isPipelineActiveFuture 忽略了 WaitOp，
//      所以如果后面只有 Wait <Src, Dst> 而没有 Dst 的实质操作，这里也会判定为 Dead，
//      从而删除 Set。Wait 消除逻辑会删除那个 Wait。完美闭环。
//   B: Stale Broadcast (陈旧广播) —— Src 在当前 Block 没脏过 (没干活)，就不发广播。
//      这精准删除了 scf.if 中 MTE2->MTE3 的冗余广播，因为 MTE2 在分支里通常是不动的。
static bool isRedundantSet(Operation *op, Block *block, Block::iterator it,
                           Attribute setSrc, Attribute setDst,
                           llvm::DenseSet<Attribute> &intraPipeDirtySet) {
  if (!isPipelineActiveFuture(block, std::next(it), setDst)) {
    return true;
  }
  if (!intraPipeDirtySet.count(setSrc)) {
    return true;
  }
  return false;
}

// ==========================================================
// Pass 实现
// ==========================================================

// 对单个 op 依次应用 Barrier/Wait/Set 三条冗余同步消除规则，
// 命中任一规则则返回 true（调用方负责收集后统一 erase）。
static bool tryRemoveRedundantSync(Operation *op, Block *block,
                                   Block::iterator it,
                                   llvm::DenseSet<Attribute> &intraPipeDirtySet) {
  // === 1. 状态更新 ===
  Attribute pipe = getOpPipe(op);
  if (pipe) {
    intraPipeDirtySet.insert(pipe);
    return false;
  }

  // === 2. Barrier 消除 ===
  if (auto barrierOp = dyn_cast<pto::BarrierOp>(op)) {
    return isRedundantBarrier(barrierOp, block, it, intraPipeDirtySet);
  }

  // === 3. Wait 消除 (幽灵 Wait 消除) ===
  Attribute waitDst;
  if (getWaitSyncDst(op, waitDst)) {
    return isRedundantWait(op, block, it, waitDst);
  }

  // === 4. Set 消除 (死信 & 陈旧广播消除) ===
  Attribute setSrc;
  Attribute setDst;
  if (getSetSyncPipes(op, setSrc, setDst)) {
    return isRedundantSet(op, block, it, setSrc, setDst, intraPipeDirtySet);
  }
  return false;
}

struct PTORemoveRedundantBarrierPass : public PassWrapper<PTORemoveRedundantBarrierPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTORemoveRedundantBarrierPass)

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    llvm::SmallVector<Operation*> opsToErase;

    func.walk([&](Block *block) {
      // 记录 Block 内脏状态 (Intra-Block Dirty State)
      // 用于判断是否需要发广播
      llvm::DenseSet<Attribute> intraPipeDirtySet;

      for (auto it = block->begin(); it != block->end(); ++it) {
        Operation *op = &*it;
        if (tryRemoveRedundantSync(op, block, it, intraPipeDirtySet)) {
          opsToErase.push_back(op);
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
