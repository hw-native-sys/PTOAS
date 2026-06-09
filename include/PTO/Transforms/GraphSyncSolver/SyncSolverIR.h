// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===------------- SyncSolverIR.h ---- Graph Sync Solver ------------------===//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERIR_H
#define MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERIR_H

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/GraphSyncSolver/MemInfo.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <memory>
#include <utility>

namespace mlir::pto::syncsolver {
#ifndef ASSERT
#ifndef NDEBUG
#define ASSERT(condition) assert(condition)
#else
#define ASSERT(condition) ((void)0)
#endif
#endif

// PTO does not currently expose HIVM's unit-flag interface. Keep the upstream
// fields and method names, but make the feature inert unless matching PTO ops
// are added later.
class UnitFlagInfoBase {
public:
  virtual ~UnitFlagInfoBase() = default;
  virtual void reset() {}
  void merge(const UnitFlagInfoBase &, bool = true, bool = true) const {}
  bool disabledAsSet() const { return true; }
  bool disabledAsWait() const { return true; }
  llvm::SmallVector<int64_t> getUnitFlagModesAsSet(bool = true) const {
    return {};
  }
  llvm::SmallVector<int64_t> getUnitFlagModesAsWait(bool = true) const {
    return {};
  }
};

class OperationBase;
class Scope;
class Loop;
class Condition;
class RWOperation;
class MmadL0Operation;
using Body = std::vector<std::unique_ptr<OperationBase>>;

// Currently gss-code-gen will handle offsetting induction variables for
// multibuffer-enabled sync pairs, which can be done by create-preload.
// A later pipeline cleanup can move create-preload after gss in the PTO
// compilation pipeline and let it handle preload-offset values.
struct EventIdInfo {
  int64_t eventIdNum{0};
  int64_t eventIdRepeatNum{1};
  int64_t preloadOffset1{0};
  int64_t preloadOffset2{0};
  LoopLikeOpInterface multibufferLoop{nullptr};
  LoopLikeOpInterface multibufferUnrollLoop1{nullptr};
  LoopLikeOpInterface multibufferUnrollLoop2{nullptr};
  EventIdInfo() {};
  explicit EventIdInfo(int64_t eventIdNum) : eventIdNum(eventIdNum) {};
};

enum struct OpType {
  OPERATION,
  PLACE_HOLDER,
  SCOPE,
  FUNCTION,
  FUNCTION_BLOCK,
  LOOP,
  LOOP_END,
  MMAD_SCOPE,
  CONDITION,
  SCOPE_END,
  SYNC_OP,
  BARRIER_OP,
  SW_FLAG_OP,
  SET_FLAG_OP,
  WAIT_FLAG_OP,
  SW_FLAG_OP_END,
  SYNC_OP_END,
  RW_OPERATION,
  MMAD_OPERATION,
  MMAD_LOAD_L0A_OPERATION,
  MMAD_LOAD_L0B_OPERATION,
  MMAD_LOAD_BIAS_OPERATION,
  RW_OPERATION_END
};

std::string getOpTypeStr(OpType opType);

class OperationBase {
public:
  int id{-1};
  const OpType opType;
  mlir::Operation *op{nullptr};
  OperationBase *parentOp{nullptr};

private:
  // Monotonic id allocator used to assign stable ids to in-memory ops.
  static int globalIndex;

public:
  OperationBase() = delete;
  OperationBase(const OpType &opType, Operation *op, OperationBase *parentOp)
      : opType(opType), op(op), parentOp(parentOp) {
    id = globalIndex++;
  }

public:
  virtual ~OperationBase() = default;

  // Return true when op1 and op2 share the same immediate parent operation.
  static bool sameScope(const OperationBase *op1, const OperationBase *op2);

  // Compute the depth (levels up to root) of the provided operation.
  int getDepth() const;

  // Return the ancestor `dist` levels above this operation.
  static OperationBase *getNthParent(OperationBase *op, int dist);

  // Given two operations, return the pair of operations directly below their
  // LCA.
  static std::pair<OperationBase *, OperationBase *>
  getLCAPair(OperationBase *op1, OperationBase *op2);

  template <typename TyOp> TyOp *getParentOfType() const {
    OperationBase *cur = this->parentOp;
    while (cur != nullptr && !isa<TyOp>(cur)) {
      cur = cur->parentOp;
    }
    return llvm::dyn_cast_if_present<TyOp>(cur);
  }

  // Find nearest parent operation that is a loop-like construct, or nullptr.
  static OperationBase *getParentloop(OperationBase *op);

  // Find the nearest parent condition operation, or nullptr.
  static OperationBase *getParentCondition(OperationBase *op);

  // Return true if this operation is a strict ancestor of `op`.
  bool isProperAncestor(const OperationBase *op) const;

  // Collect and return all parent operations (walking upwards).
  llvm::SmallVector<OperationBase *> getAllParents() const;

  // Human-readable string representation (override in derived classes).
  virtual std::string str(int indent = 0, bool recursive = false) const = 0;

  static OperationBase *getUnlikelyParentCondition(OperationBase *op);
};

class PlaceHolder : public OperationBase {
public:
  mlir::Block *block{nullptr};
  OperationBase *beforeOp{nullptr};
  OperationBase *afterOp{nullptr};
  Scope *scopeBegin{nullptr};
  Scope *scopeEnd{nullptr};

public:
  PlaceHolder(Operation *op, OperationBase *parentOp)
      : OperationBase(OpType::PLACE_HOLDER, op, parentOp) {}

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::PLACE_HOLDER;
  }

  std::string str(int indent, bool recursive) const override;
};

class Scope : public OperationBase {
public:
  Body body;
  std::optional<int64_t> preloadNum;
  std::optional<int64_t> maxPreloadNum;

public:
  Scope(const OpType &opType = OpType::SCOPE, Operation *op = nullptr,
        OperationBase *parentOp = nullptr)
      : OperationBase(opType, op, parentOp) {}

  static bool classof(const OperationBase *e) {
    return e->opType >= OpType::SCOPE && e->opType < OpType::SCOPE_END;
  }

  std::string str(int indent, bool recursive) const override;
};

class FunctionBlock : public Scope {
public:
  FunctionBlock() : Scope(OpType::FUNCTION_BLOCK) {}

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::FUNCTION_BLOCK;
  }
};

class Function : public Scope {
public:
  Function(Operation *op) : Scope(OpType::FUNCTION, op, nullptr) {}

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::FUNCTION;
  }
};

class Loop : public Scope {
private:
public:
  bool isParallel{false};
  std::optional<int64_t> multibufferUnrollNum;
  Loop(Operation *op, OperationBase *parentOp)
      : Scope(OpType::LOOP, op, parentOp) {}

  static bool classof(const OperationBase *e) {
    return e->opType >= OpType::LOOP && e->opType < OpType::LOOP_END;
  }

  std::string str(int indent, bool recursive) const override;
};

class MmadL1LoopOp : public Scope {
private:
public:
  MmadL0Operation *mmadL0Op{nullptr};

  MmadL1LoopOp(Operation *op, OperationBase *parentOp)
      : Scope(OpType::MMAD_SCOPE, op, parentOp) {};

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::MMAD_SCOPE;
  }
};

class Condition : public Scope {
private:
public:
  Scope *trueScope{nullptr};
  Scope *falseScope{nullptr};
  bool isUnlikely{false};
  Condition(Operation *op, OperationBase *parentOp,
            std::unique_ptr<Scope> trueScope, std::unique_ptr<Scope> falseScope)
      : Scope(OpType::CONDITION, op, parentOp) {
    if (trueScope != nullptr) {
      this->setTrueScope(std::move(trueScope));
    }
    if (falseScope != nullptr) {
      ASSERT(this->trueScope != nullptr);
      this->setFalseScope(std::move(falseScope));
    }
  };

  Scope *getTrueScope() const {
    ASSERT(this->trueScope != nullptr);
    return this->trueScope;
  }

  Scope *getFalseScope() const {
    ASSERT(this->falseScope != nullptr);
    return this->falseScope;
  }

  void setFalseScope(std::unique_ptr<Scope> falseScope) {
    ASSERT(falseScope != nullptr);
    falseScope->parentOp = this;
    this->falseScope = falseScope.get();
    this->body.push_back(std::move(falseScope));
  }

  void setTrueScope(std::unique_ptr<Scope> trueScope) {
    ASSERT(trueScope != nullptr);
    trueScope->parentOp = this;
    this->trueScope = trueScope.get();
    this->body.push_back(std::move(trueScope));
  }

  bool hasFalseScope() const { return this->falseScope != nullptr; }

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::CONDITION;
  }

  std::string str(int indent, bool recursive) const override;
};

class RWOperation : public OperationBase {
public:
  pto::TCoreType coreType{pto::TCoreType::CUBE_OR_VECTOR};
  pto::PIPE pipeRead{pto::PIPE::PIPE_UNASSIGNED};
  pto::PIPE pipeWrite{pto::PIPE::PIPE_UNASSIGNED};
  llvm::SmallVector<MemInfo> readMemInfo;
  llvm::SmallVector<MemInfo> writeMemInfo;
  bool hasUnitFlagFeat{false};
  UnitFlagInfoBase mergedUnitFlagInfo;

  const llvm::SmallVector<Value> readMemVals;
  const llvm::SmallVector<Value> writeMemVals;
  const llvm::SmallVector<llvm::SmallVector<int64_t>> testReadMemVals;
  const llvm::SmallVector<llvm::SmallVector<int64_t>> testWriteMemVals;

public:
  RWOperation(Operation *op, OperationBase *parentOp, pto::TCoreType coreType,
              pto::PIPE pipeRead, pto::PIPE pipeWrite,
              const llvm::SmallVector<Value> &readMemVals,
              const llvm::SmallVector<Value> &writeMemVals,
              OpType opType = OpType::RW_OPERATION)
      : OperationBase(opType, op, parentOp), coreType(coreType),
        pipeRead(pipeRead), pipeWrite(pipeWrite), readMemVals(readMemVals),
        writeMemVals(writeMemVals) {
    for (auto &val : readMemVals) {
      readMemInfo.push_back(getMemInfo(val));
    }
    for (auto &val : writeMemVals) {
      writeMemInfo.push_back(getMemInfo(val));
    }
  };
  RWOperation(
      Operation *op, OperationBase *parentOp, pto::TCoreType coreType,
      pto::PIPE pipeRead, pto::PIPE pipeWrite,
      const llvm::SmallVector<llvm::SmallVector<int64_t>> &testReadMemVals,
      const llvm::SmallVector<llvm::SmallVector<int64_t>> &testWriteMemVals,
      OpType opType = OpType::RW_OPERATION)
      : OperationBase(opType, op, parentOp), coreType(coreType),
        pipeRead(pipeRead), pipeWrite(pipeWrite),
        testReadMemVals(testReadMemVals), testWriteMemVals(testWriteMemVals) {
    for (auto &val : testReadMemVals) {
      readMemInfo.push_back(getMemInfo(val));
    }
    for (auto &val : testWriteMemVals) {
      writeMemInfo.push_back(getMemInfo(val));
    }
  };

  std::string str(int indent, bool recursive) const override;

  static bool classof(const OperationBase *e) {
    return e->opType >= OpType::RW_OPERATION &&
           e->opType < OpType::RW_OPERATION_END;
  }
};

class LoadL0AOp : public RWOperation {
private:
public:
  LoadL0AOp(Operation *op, OperationBase *parentOp, pto::TCoreType coreType,
            pto::PIPE pipeRead, pto::PIPE pipeWrite,
            const llvm::SmallVector<Value> &readMemVals,
            const llvm::SmallVector<Value> &writeMemVals)
      : RWOperation(op, parentOp, coreType, pipeRead, pipeWrite, readMemVals,
                    writeMemVals, OpType::MMAD_LOAD_L0A_OPERATION) {}

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::MMAD_LOAD_L0A_OPERATION;
  }
};

class LoadL0BOp : public RWOperation {
private:
public:
  LoadL0BOp(Operation *op, OperationBase *parentOp, pto::TCoreType coreType,
            pto::PIPE pipeRead, pto::PIPE pipeWrite,
            const llvm::SmallVector<Value> &readMemVals,
            const llvm::SmallVector<Value> &writeMemVals)
      : RWOperation(op, parentOp, coreType, pipeRead, pipeWrite, readMemVals,
                    writeMemVals, OpType::MMAD_LOAD_L0B_OPERATION) {}

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::MMAD_LOAD_L0B_OPERATION;
  }
};

class LoadBiasOp : public RWOperation {
private:
public:
  LoadBiasOp(Operation *op, OperationBase *parentOp, pto::TCoreType coreType,
             pto::PIPE pipeRead, pto::PIPE pipeWrite,
             const llvm::SmallVector<Value> &readMemVals,
             const llvm::SmallVector<Value> &writeMemVals)
      : RWOperation(op, parentOp, coreType, pipeRead, pipeWrite, readMemVals,
                    writeMemVals, OpType::MMAD_LOAD_BIAS_OPERATION) {}

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::MMAD_LOAD_BIAS_OPERATION;
  }
};

class MmadL0Operation : public RWOperation {
private:
public:
  MmadL0Operation(Operation *op, OperationBase *parentOp,
                  pto::TCoreType coreType, pto::PIPE pipeRead,
                  pto::PIPE pipeWrite,
                  const llvm::SmallVector<Value> &readMemVals,
                  const llvm::SmallVector<Value> &writeMemVals)
      : RWOperation(op, parentOp, coreType, pipeRead, pipeWrite, readMemVals,
                    writeMemVals, OpType::MMAD_OPERATION) {}

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::MMAD_OPERATION;
  }
};

class SyncOp : public OperationBase {
public:
  std::optional<int> debugId;
  SyncOp(const OpType &opType, Operation *op, OperationBase *parentOp)
      : OperationBase(opType, op, parentOp) {}

  static bool classof(const OperationBase *e) {
    return e->opType >= OpType::SYNC_OP && e->opType < OpType::SYNC_OP_END;
  }
};

class SetWaitOp : public SyncOp {
public:
  llvm::SmallVector<int64_t> eventIds;
  pto::TCoreType coreType{pto::TCoreType::CUBE_OR_VECTOR};
  pto::PIPE pipeSrc{pto::PIPE::PIPE_UNASSIGNED};
  pto::PIPE pipeDst{pto::PIPE::PIPE_UNASSIGNED};
  EventIdInfo eventIdInfo;
  bool allAtOnce{false};
  bool checkFirstIter{false};
  bool checkLastIter{false};

  SetWaitOp(const OpType &opType, Operation *op, OperationBase *parentOp,
            const llvm::SmallVector<int64_t> &eventIds, pto::PIPE pipeSrc,
            pto::PIPE pipeDst)
      : SyncOp(opType, op, parentOp), eventIds(eventIds), pipeSrc(pipeSrc),
        pipeDst(pipeDst) {}

  static bool classof(const OperationBase *e) {
    return e->opType >= OpType::SW_FLAG_OP &&
           e->opType < OpType::SW_FLAG_OP_END;
  }
};

class SetFlagOp : public SetWaitOp {
private:
public:
  SetFlagOp(Operation *op, OperationBase *parentOp,
            const llvm::SmallVector<int64_t> &eventIds, pto::PIPE pipeSrc,
            pto::PIPE pipeDst)
      : SetWaitOp(OpType::SET_FLAG_OP, op, parentOp, eventIds, pipeSrc,
                  pipeDst) {}

  std::unique_ptr<SetFlagOp> clone() {
    return std::make_unique<SetFlagOp>(op, parentOp, eventIds, pipeSrc,
                                       pipeDst);
  }

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::SET_FLAG_OP;
  }

  std::string str(int indent, bool recursive) const override;
};

class WaitFlagOp : public SetWaitOp {
private:
public:
  WaitFlagOp(Operation *op, OperationBase *parentOp,
             const llvm::SmallVector<int64_t> &eventIds, pto::PIPE pipeSrc,
             pto::PIPE pipeDst)
      : SetWaitOp(OpType::WAIT_FLAG_OP, op, parentOp, eventIds, pipeSrc,
                  pipeDst) {}

  std::unique_ptr<WaitFlagOp> clone() {
    return std::make_unique<WaitFlagOp>(op, parentOp, eventIds, pipeSrc,
                                        pipeDst);
  }

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::WAIT_FLAG_OP;
  }

  std::string str(int indent, bool recursive) const override;
};

class BarrierOp : public SyncOp {
public:
  pto::PIPE pipe{pto::PIPE::PIPE_UNASSIGNED};

private:
public:
  BarrierOp(Operation *op, OperationBase *parentOp, pto::PIPE pipe)
      : SyncOp(OpType::BARRIER_OP, op, parentOp), pipe(pipe) {}

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::BARRIER_OP;
  }

  std::string str(int indent, bool recursive) const override;
};

// Bool comparator for sync ops ordering (used for containers).
bool operator<(const SyncOp &op1, const SyncOp &op2);
} // namespace mlir::pto::syncsolver

#endif // MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERIR_H
