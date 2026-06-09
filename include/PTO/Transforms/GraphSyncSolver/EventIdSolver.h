// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===------------- EventIdSolver.h ---- Graph Sync Solver
//-------------------===//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_EVENTIDSOLVER_H
#define MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_EVENTIDSOLVER_H

#include "PTO/Transforms/GraphSyncSolver/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include <memory>
#include <stack>
#include <vector>

namespace mlir::pto::syncsolver {
enum class ACTION_TYPE {
  NONE,
  ADD_NODE,
  ADD_EDGE,
  INSERT_CONFLICT_PAIR,
  ASSIGN_EVENT_IDS,
  ASSIGN_NEED_RECALC,
};

class Action {
public:
  const ACTION_TYPE actionType;
  Action() = delete;
  Action(ACTION_TYPE actionType) : actionType(actionType) {};
  virtual ~Action() = default;
  virtual std::string str() const = 0;
};

class ActionNone : public Action {
public:
  ActionNone() : Action(ACTION_TYPE::NONE) {}
  static bool classof(const Action *e) {
    return e->actionType == ACTION_TYPE::NONE;
  }
  std::string str() const override { return "NONE()"; }
};

class ActionAddNode : public Action {
public:
  EventIdNode *const node;
  ActionAddNode(EventIdNode *node) : Action(ACTION_TYPE::ADD_NODE), node(node) {
    ASSERT(node != nullptr);
  }
  static bool classof(const Action *e) {
    return e->actionType == ACTION_TYPE::ADD_NODE;
  }
  std::string str() const override {
    return "ADD_NODE(" + std::to_string(node->id) + ")";
  }
};

class ActionAddEdge : public Action {
public:
  EventIdNode *const node1;
  EventIdNode *const node2;
  ActionAddEdge(EventIdNode *node1, EventIdNode *node2)
      : Action(ACTION_TYPE::ADD_EDGE), node1(node1), node2(node2) {
    ASSERT(node1 != nullptr);
    ASSERT(node2 != nullptr);
  }
  static bool classof(const Action *e) {
    return e->actionType == ACTION_TYPE::ADD_EDGE;
  }
  std::string str() const override {
    return "ADD_EDGE(" + std::to_string(node1->id) + ", " +
           std::to_string(node2->id) + ")";
  }
};

class ActionInsertConflictPair : public Action {
public:
  EventIdNode *const node;
  ConflictPair *const conflictPair;
  ActionInsertConflictPair(EventIdNode *node, ConflictPair *conflictPair)
      : Action(ACTION_TYPE::INSERT_CONFLICT_PAIR), node(node),
        conflictPair(conflictPair) {
    ASSERT(node != nullptr);
    ASSERT(conflictPair != nullptr);
  }
  static bool classof(const Action *e) {
    return e->actionType == ACTION_TYPE::INSERT_CONFLICT_PAIR;
  }
  std::string str() const override {
    return "INSERT_CONFLICT_PAIR(" + std::to_string(node->id) + ", " +
           std::to_string(conflictPair->id) + ")";
  }
};

class ActionAssignEventIds : public Action {
public:
  EventIdNode *const node;
  const llvm::SmallVector<int64_t> oldEventIds;
  const llvm::SmallVector<int64_t> newEventIds;
  ActionAssignEventIds(EventIdNode *node,
                       const llvm::SmallVector<int64_t> &oldEventIds,
                       const llvm::SmallVector<int64_t> &newEventIds)
      : Action(ACTION_TYPE::ASSIGN_EVENT_IDS), node(node),
        oldEventIds(oldEventIds), newEventIds(newEventIds) {
    ASSERT(node != nullptr);
  }
  static bool classof(const Action *e) {
    return e->actionType == ACTION_TYPE::ASSIGN_EVENT_IDS;
  }
  std::string str() const override {
    std::string ret;
    ret += "ASSIGN_EVENT_IDS(" + std::to_string(node->id) + ", ";
    {
      std::string tmp;
      llvm::raw_string_ostream rso(tmp);
      llvm::interleaveComma(oldEventIds, rso);
      ret += "[" + rso.str() + "]";
    }
    ret += ", ";
    {
      std::string tmp;
      llvm::raw_string_ostream rso(tmp);
      llvm::interleaveComma(newEventIds, rso);
      ret += "[" + rso.str() + "]";
    }
    ret += ")";
    return ret;
  }
};

class ActionAssignNeedRecalc : public Action {
public:
  const bool oldValue;
  const bool newValue;
  ActionAssignNeedRecalc(bool oldValue, bool newValue)
      : Action(ACTION_TYPE::ASSIGN_NEED_RECALC), oldValue(oldValue),
        newValue(newValue) {}
  static bool classof(const Action *e) {
    return e->actionType == ACTION_TYPE::ASSIGN_NEED_RECALC;
  }
  std::string str() const override {
    return "ASSIGN_NEED_RECALC(" + std::to_string(oldValue) + ", " +
           std::to_string(newValue) + ")";
  }
};

class EventIdSolver {
private:
  int64_t eventIdsNumMax{-1};
  bool needRecalculateEventIds{false};
  llvm::SmallVector<std::unique_ptr<EventIdNode>> nodes;
  llvm::DenseMap<EventIdNode *, llvm::DenseMap<EventIdNode *, int64_t>> adjList;
  llvm::DenseMap<EventIdNode *, int64_t> sumAdjListSizes;
  llvm::DenseMap<ConflictPair *, EventIdNode *> conflictPair2Node;
  std::stack<std::unique_ptr<Action>> actionsStack;

public:
  EventIdSolver(int64_t eventIdNumMax) : eventIdsNumMax(eventIdNumMax) {}
  ~EventIdSolver() = default;

  bool isColorable();

  llvm::LogicalResult shrinkEventIdMaxToEventIdNum();

  EventIdNode *getNode(ConflictPair *conflictPair);

  EventIdNode *createNode(ConflictPair *conflictPair, int64_t eventIdNum = 1,
                          bool reversePriority = false);

  void addConflicts(ConflictPair *conflictPairSrc,
                    const std::vector<ConflictPair *> &conflictPairsDst);

  void calcEventIds();

  void pushActionNone() { actionsStack.push(std::make_unique<ActionNone>()); }

  void clearActionStack() {
    while (!actionsStack.empty()) {
      actionsStack.pop();
    }
  }

  void undoActions();

  void debugPrint();

private:
  int64_t getEventIdsNum(bool dontCalcEventIds = false);

  std::unique_ptr<EventIdSolver> clone();

public:
  // do
  void insertConflictPair(EventIdNode *node, ConflictPair *conflictPair);

private:
  // do
  void addNode(std::unique_ptr<EventIdNode> node);

  void addEdge(EventIdNode *node1, EventIdNode *node2);

  // undo
  void removeNode(EventIdNode *node);

  void eraseConflictPair(EventIdNode *node, ConflictPair *conflictPair);

  void removeEdge(EventIdNode *node1, EventIdNode *node2);

  // do-undo
  void assignEventIds(EventIdNode *node,
                      const llvm::SmallVector<int64_t> &eventIds,
                      bool pushAction = true);

  void assignNeedRecalc(bool newValue, bool pushAction = true);

  // calc
  llvm::SmallVector<int64_t> getAdjNodesUsedEventIds(EventIdNode *node);

  llvm::SmallVector<int64_t> getChosenEventIds(EventIdNode *node,
                                               int64_t eventIdMax);
};
} // namespace mlir::pto::syncsolver

#endif // MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_EVENTIDSOLVER_H
