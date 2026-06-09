// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h" 
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
 
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
    if (auto loadOp = dyn_cast<pto::TLoadOp>(op)) 
        return pto::PipeAttr::get(op->getContext(), pto::PIPE::PIPE_MTE2) == targetPipe;
    if (auto storeOp = dyn_cast<pto::TStoreOp>(op)) 
        return pto::PipeAttr::get(op->getContext(), pto::PIPE::PIPE_MTE3) == targetPipe;
    if (auto addfOp = dyn_cast<pto::TAddOp>(op)) 
        return pto::PipeAttr::get(op->getContext(), pto::PIPE::PIPE_V) == targetPipe;
    return false;
}
 
// 递归检查 Region 内是否有实质性操作
// 用于深入 scf.if / scf.for 内部查找
bool isPipeUsedInRegion(Region &region, Attribute targetPipe) {
    for (Block &block : region) {
        for (Operation &op : block) {
            // 1. 如果是实质性操作，返回 True
            if (isResourceOp(&op, targetPipe)) return true;
            
            // 2. 递归检查嵌套 (if/for)
            for (Region &nestedRegion : op.getRegions()) {
                if (isPipeUsedInRegion(nestedRegion, targetPipe)) return true;
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
    if (!parentBlock)
      return false;
    for (auto it = std::next(parentOp->getIterator()); it != parentBlock->end(); ++it) {
        if (isResourceOp(&*it, targetPipe))
          return true;
        if (it->getNumRegions() > 0)
          return true;
        if (isa<func::ReturnOp>(&*it))
          return false;
    }
    return false;
}

bool isPipelineActiveFuture(Block *block, Block::iterator startIt, Attribute targetPipe) {
    for (auto it = startIt; it != block->end(); ++it) {
        Operation *op = &*it;
        
        // 1. 遇到实质性操作 -> 活跃
        if (isResourceOp(op, targetPipe)) return true;
 
        // [注意] 这里故意跳过了 WaitOp 的检查。
        // WaitOp 只是同步原语，不代表该 Pipeline 在"干活"。
 
        // 2. 递归检查嵌套区域 (scf.if, scf.for)
        for (Region &region : op->getRegions()) {
            if (isPipeUsedInRegion(region, targetPipe)) return true;
        }
 
        // 3. 处理 Terminator (跨 Block 检查)
        if (op->hasTrait<OpTrait::IsTerminator>()) {
            // 如果是 Return，肯定死了
            if (isa<func::ReturnOp>(op)) return false;
            return hasPipelineActivityAfterOp(block->getParentOp(), targetPipe);
        }
    }
    return false;
}

static Attribute getTrackedPipe(Operation *op, Attribute attrMTE2,
                                Attribute attrMTE3, Attribute attrVec) {
  if (isa<pto::TLoadOp>(op))
    return attrMTE2;
  if (isa<pto::TStoreOp>(op))
    return attrMTE3;
  if (isa<pto::TAddOp>(op))
    return attrVec;
  return {};
}

static bool getSetSyncPipes(Operation *op, Attribute &src, Attribute &dst) {
  if (auto setOp = dyn_cast<pto::SetFlagOp>(op)) {
    src = setOp.getSrcPipe();
    dst = setOp.getDstPipe();
    return true;
  }
  StringRef opName = op->getName().getStringRef();
  if (opName != "pto.set_flag_dyn" && opName != "pto.set_flag_d")
    return false;
  auto srcAttr = op->getAttrOfType<pto::PipeAttr>("src_pipe");
  auto dstAttr = op->getAttrOfType<pto::PipeAttr>("dst_pipe");
  if (!srcAttr || !dstAttr)
    return false;
  src = srcAttr;
  dst = dstAttr;
  return true;
}

static bool getWaitSyncDst(Operation *op, Attribute &dst) {
  if (auto waitOp = dyn_cast<pto::WaitFlagOp>(op)) {
    dst = waitOp.getDstPipe();
    return true;
  }
  StringRef opName = op->getName().getStringRef();
  if (opName != "pto.wait_flag_dyn" && opName != "pto.wait_flag_d")
    return false;
  auto dstAttr = op->getAttrOfType<pto::PipeAttr>("dst_pipe");
  if (!dstAttr)
    return false;
  dst = dstAttr;
  return true;
}

static bool shouldEraseBarrierOp(Block *block, Block::iterator it,
                                 llvm::DenseSet<Attribute> &intraPipeDirtySet) {
  auto barrierOp = dyn_cast<pto::BarrierOp>(&*it);
  if (!barrierOp)
    return false;
  Attribute bPipe = barrierOp.getPipe();
  if (!isPipelineActiveFuture(block, std::next(it), bPipe))
    return true;
  if (intraPipeDirtySet.count(bPipe) == 0)
    return true;
  auto nextIt = std::next(it);
  if (nextIt == block->end())
    return false;
  Attribute nextSrc;
  Attribute nextDst;
  return getSetSyncPipes(&*nextIt, nextSrc, nextDst) && nextSrc == bPipe;
}

static bool shouldEraseWaitOp(Block *block, Block::iterator it) {
  Attribute waitDst;
  return getWaitSyncDst(&*it, waitDst) &&
         !isPipelineActiveFuture(block, std::next(it), waitDst);
}

static bool shouldEraseSetOp(Block *block, Block::iterator it,
                             llvm::DenseSet<Attribute> &intraPipeDirtySet) {
  Attribute setSrc;
  Attribute setDst;
  if (!getSetSyncPipes(&*it, setSrc, setDst))
    return false;
  if (!isPipelineActiveFuture(block, std::next(it), setDst))
    return true;
  return intraPipeDirtySet.count(setSrc) == 0;
}

static void collectRedundantBarriersInBlock(
    Block *block, Attribute attrMTE2, Attribute attrMTE3, Attribute attrVec,
    llvm::SmallVector<Operation *> &opsToErase) {
  llvm::DenseSet<Attribute> intraPipeDirtySet;
  for (auto it = block->begin(); it != block->end(); ++it) {
    Operation *op = &*it;
    Attribute pipe = getTrackedPipe(op, attrMTE2, attrMTE3, attrVec);
    if (pipe) {
      intraPipeDirtySet.insert(pipe);
      continue;
    }
    if (shouldEraseBarrierOp(block, it, intraPipeDirtySet)) {
      opsToErase.push_back(op);
      continue;
    }
    if (auto barrierOp = dyn_cast<pto::BarrierOp>(op)) {
      intraPipeDirtySet.erase(barrierOp.getPipe());
      continue;
    }
    if (shouldEraseWaitOp(block, it) ||
        shouldEraseSetOp(block, it, intraPipeDirtySet)) {
      opsToErase.push_back(op);
    }
  }
}
 
// ==========================================================
// Pass 实现
// ==========================================================
struct PTORemoveRedundantBarrierPass : public PassWrapper<PTORemoveRedundantBarrierPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTORemoveRedundantBarrierPass)
 
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();
    
    Attribute attrMTE2 = pto::PipeAttr::get(ctx, pto::PIPE::PIPE_MTE2);
    Attribute attrMTE3 = pto::PipeAttr::get(ctx, pto::PIPE::PIPE_MTE3);
    Attribute attrVec  = pto::PipeAttr::get(ctx, pto::PIPE::PIPE_V);
 
    llvm::SmallVector<Operation*> opsToErase;
 
    func.walk([&](Block *block) {
      collectRedundantBarriersInBlock(block, attrMTE2, attrMTE3, attrVec,
                                      opsToErase);
    });
 
    for (Operation *op : opsToErase) op->erase();
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
