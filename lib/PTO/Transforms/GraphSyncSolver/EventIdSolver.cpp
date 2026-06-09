// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===----------- EventIdSolver.cpp ---- Graph Sync Solver -----------------===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/GraphSyncSolver/EventIdSolver.h"
#include "PTO/Transforms/GraphSyncSolver/Utility.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <numeric>
#include <utility>

#define DEBUG_TYPE "PTO-gss-eventidsolver"

using namespace mlir;
using namespace pto::syncsolver;

int64_t EventIdSolver::getEventIdsNum(bool dontCalcEventIds) {
  if (!dontCalcEventIds) {
    calcEventIds();
  }
  ASSERT(!needRecalculateEventIds);
  llvm::SmallDenseSet<int64_t> usedEventIds;
  for (auto &node : nodes) {
    auto &eventIds = node->getEventIds();
    ASSERT(!eventIds.empty());
    usedEventIds.insert(eventIds.begin(), eventIds.end());
  }
  return usedEventIds.size();
}

llvm::LogicalResult EventIdSolver::shrinkEventIdMaxToEventIdNum() {
  if (needRecalculateEventIds || !actionsStack.empty() || !isColorable()) {
    return llvm::failure();
  }
  this->eventIdsNumMax = getEventIdsNum();
  calcEventIds();
  clearActionStack();
  return llvm::success();
}

bool EventIdSolver::isColorable() {
  if (needRecalculateEventIds) {
    calcEventIds();
  }
  return getEventIdsNum(/*dontCalcEventIds=*/true) <= eventIdsNumMax;
}

void EventIdSolver::addNode(std::unique_ptr<EventIdNode> node) {
  actionsStack.push(std::make_unique<ActionAddNode>(node.get()));
  nodes.push_back(std::move(node));
}

void EventIdSolver::removeNode(EventIdNode *node) {
  ASSERT(adjList[node].empty());
  ASSERT(sumAdjListSizes[node] == 0);
  adjList.erase(node);
  sumAdjListSizes.erase(node);
  ASSERT(nodes.back().get() == node);
  nodes.pop_back();
}

void EventIdSolver::insertConflictPair(EventIdNode *node,
                                       ConflictPair *conflictPair) {
  ASSERT(node != nullptr && conflictPair != nullptr);
  actionsStack.push(
      std::make_unique<ActionInsertConflictPair>(node, conflictPair));
  conflictPair2Node[conflictPair] = node;
  node->insertConflictPair(conflictPair);
}

void EventIdSolver::eraseConflictPair(EventIdNode *node,
                                      ConflictPair *conflictPair) {
  ASSERT(node != nullptr && conflictPair != nullptr);
  conflictPair2Node.erase(conflictPair);
  node->eraseConflictPair(conflictPair);
}

void EventIdSolver::addEdge(EventIdNode *node1, EventIdNode *node2) {
  ASSERT(node1 != nullptr && node2 != nullptr);
  LLVM_DEBUG({
    llvm::dbgs() << "add-edge: " << node1->str(false) << ' '
                 << node2->str(false) << '\n';
  });
  if (node1 == node2) {
    return;
  }
  actionsStack.push(std::make_unique<ActionAddEdge>(node1, node2));
  int64_t &node1ToNode2Count = adjList[node1][node2];
  if (node1ToNode2Count == 0) {
    sumAdjListSizes[node1] += node2->eventIdNum;
  }
  node1ToNode2Count += 1;
  int64_t &node2ToNode1Count = adjList[node2][node1];
  if (node2ToNode1Count == 0) {
    sumAdjListSizes[node2] += node1->eventIdNum;
  }
  node2ToNode1Count += 1;
  if (!needRecalculateEventIds) {
    if ((sumAdjListSizes[node1] + node1->eventIdNum > eventIdsNumMax) ||
        (sumAdjListSizes[node2] + node2->eventIdNum > eventIdsNumMax)) {
      assignNeedRecalc(true);
    }
  }
}

void EventIdSolver::removeEdge(EventIdNode *node1, EventIdNode *node2) {
  ASSERT(node1 != nullptr && node2 != nullptr);
  int64_t &node1ToNode2Count = adjList[node1][node2];
  node1ToNode2Count -= 1;
  if (node1ToNode2Count == 0) {
    adjList[node1].erase(node2);
    sumAdjListSizes[node1] -= node2->eventIdNum;
  }
  int64_t &node2ToNode1Count = adjList[node2][node1];
  node2ToNode1Count -= 1;
  if (node2ToNode1Count == 0) {
    adjList[node2].erase(node1);
    sumAdjListSizes[node2] -= node1->eventIdNum;
  }
}

void EventIdSolver::assignEventIds(EventIdNode *node,
                                   const llvm::SmallVector<int64_t> &eventIds,
                                   bool pushAction) {
  ASSERT(node != nullptr);
  if (node->getEventIds() == eventIds) {
    return;
  }
  if (pushAction) {
    actionsStack.push(std::make_unique<ActionAssignEventIds>(
        node, node->getEventIds(), eventIds));
  }
  node->setEventIds(eventIds);
}

void EventIdSolver::assignNeedRecalc(bool newValue, bool pushAction) {
  if (newValue == needRecalculateEventIds) {
    return;
  }
  if (pushAction) {
    actionsStack.push(std::make_unique<ActionAssignNeedRecalc>(
        needRecalculateEventIds, newValue));
  }
  needRecalculateEventIds = newValue;
}

EventIdNode *EventIdSolver::getNode(ConflictPair *conflictPair) {
  ASSERT(conflictPair != nullptr);
  auto it = conflictPair2Node.find(conflictPair);
  ASSERT(it != conflictPair2Node.end());
  return it->second;
}

std::unique_ptr<EventIdSolver> EventIdSolver::clone() {
  auto clonedEventIdSolver = std::make_unique<EventIdSolver>(eventIdsNumMax);
  llvm::DenseMap<EventIdNode *, EventIdNode *> mp;
  for (auto &node : nodes) {
    auto clonedNode = node->clone();
    mp[node.get()] = clonedNode.get();
    clonedEventIdSolver->nodes.push_back(std::move(clonedNode));
  }
  for (auto [nodeSrc, nodesDstFrq] : adjList) {
    auto *clonedNodeSrc = mp.at(nodeSrc);
    for (auto [nodeDst, frq] : nodesDstFrq) {
      ASSERT(frq > 0);
      auto *clonedNodeDst = mp.at(nodeDst);
      clonedEventIdSolver->adjList[clonedNodeSrc][clonedNodeDst] += frq;
    }
  }
  for (auto [node, val] : sumAdjListSizes) {
    auto *clonedNode = mp.at(node);
    clonedEventIdSolver->sumAdjListSizes[clonedNode] = val;
  }
  for (auto [conflictPair, eventIdNode] : conflictPair2Node) {
    auto *clonedNode = mp.at(eventIdNode);
    clonedEventIdSolver->conflictPair2Node[conflictPair] = clonedNode;
  }
  return clonedEventIdSolver;
}

EventIdNode *EventIdSolver::createNode(ConflictPair *conflictPair,
                                       int64_t eventIdNum,
                                       bool reversePriority) {
  ASSERT(conflictPair != nullptr);
  ASSERT(eventIdNum > 0);
  auto node =
      std::make_unique<EventIdNode>(conflictPair, eventIdNum, reversePriority);
  auto *nodePtr = node.get();
  addNode(std::move(node));
  insertConflictPair(nodePtr, conflictPair);
  llvm::SmallVector<int64_t> eventIds(eventIdNum, 0);
  std::iota(eventIds.begin(), eventIds.end(), 0);
  assignEventIds(nodePtr, eventIds);
  return nodePtr;
}

void EventIdSolver::addConflicts(
    ConflictPair *conflictPairSrc,
    const std::vector<ConflictPair *> &conflictPairsDst) {
  ASSERT(conflictPairSrc != nullptr);
  EventIdNode *node1 = getNode(conflictPairSrc);
  for (auto *conflictPairDst : conflictPairsDst) {
    LLVM_DEBUG({
      llvm::dbgs() << "add-conflict: " << conflictPairDst->str() << '\n';
    });
    EventIdNode *node2 = getNode(conflictPairDst);
    addEdge(node1, node2);
  }
}

llvm::SmallVector<int64_t>
EventIdSolver::getAdjNodesUsedEventIds(EventIdNode *node) {
  llvm::SmallDenseSet<int64_t> usedEventIds;
  auto adjIt = adjList.find(node);
  ASSERT(adjIt != adjList.end());
  for (auto [otherNode, frq] : adjIt->second) {
    ASSERT(frq > 0);
    auto &otherEventIds = otherNode->getEventIds();
    usedEventIds.insert(otherEventIds.begin(), otherEventIds.end());
  }
  LLVM_DEBUG({
    llvm::dbgs() << "used-event-ids: ";
    for (auto e : usedEventIds)
      llvm::dbgs() << e << ' ';
    llvm::dbgs() << "\n";
  });
  llvm::SmallVector<int64_t> usedEventIdsVec(usedEventIds.begin(),
                                             usedEventIds.end());
  llvm::sort(usedEventIdsVec);
  return usedEventIdsVec;
}

static llvm::SmallVector<int64_t>
chooseForwardEventIds(ArrayRef<int64_t> usedEventIds, int64_t eventIdNum) {
  llvm::SmallVector<int64_t> chosenEventIds;
  int64_t curEventId = 0;
  auto *it = usedEventIds.begin();
  while (static_cast<int64_t>(chosenEventIds.size()) < eventIdNum) {
    while ((it != usedEventIds.end()) && ((*it) < curEventId))
      ++it;
    if ((it != usedEventIds.end()) && ((*it) == curEventId)) {
      ++it;
    } else {
      chosenEventIds.push_back(curEventId);
    }
    ++curEventId;
  }
  return chosenEventIds;
}

static llvm::SmallVector<int64_t>
chooseReverseEventIds(ArrayRef<int64_t> usedEventIds, int64_t eventIdNum,
                      int64_t eventIdMax, int64_t eventIdsNumMax) {
  llvm::SmallVector<int64_t> chosenEventIds;
  int64_t curEventId = std::max(eventIdMax, eventIdsNumMax - 1);
  auto it = usedEventIds.rbegin();
  while ((curEventId >= 0) &&
         (static_cast<int64_t>(chosenEventIds.size()) < eventIdNum)) {
    while ((it != usedEventIds.rend()) && ((*it) > curEventId))
      ++it;
    if ((it != usedEventIds.rend()) && ((*it) == curEventId)) {
      ++it;
    } else {
      chosenEventIds.push_back(curEventId);
    }
    --curEventId;
  }
  std::reverse(chosenEventIds.begin(), chosenEventIds.end());
  if (int64_t rem = eventIdNum - static_cast<int64_t>(chosenEventIds.size());
      rem > 0) {
    for (int64_t i = 0; i < rem; ++i)
      chosenEventIds.push_back(eventIdMax + i + 1);
  }
  return chosenEventIds;
}

llvm::SmallVector<int64_t>
EventIdSolver::getChosenEventIds(EventIdNode *node, int64_t eventIdMax) {
  llvm::SmallVector<int64_t> usedEventIds = getAdjNodesUsedEventIds(node);
  auto chosenEventIds =
      node->reversePriority
          ? chooseReverseEventIds(usedEventIds, node->eventIdNum, eventIdMax,
                                  this->eventIdsNumMax)
          : chooseForwardEventIds(usedEventIds, node->eventIdNum);
  LLVM_DEBUG({
    llvm::dbgs() << "chosen-event-ids: ";
    for (auto e : chosenEventIds)
      llvm::dbgs() << e << ' ';
    llvm::dbgs() << '\n';
  });
  ASSERT(node->eventIdNum == static_cast<int64_t>(chosenEventIds.size()));
  ASSERT(llvm::is_sorted(chosenEventIds));
  return chosenEventIds;
}

void EventIdSolver::calcEventIds() {
  auto cmp = [](const std::pair<int64_t, EventIdNode *> &a,
                const std::pair<int64_t, EventIdNode *> &b) {
    if (a.first != b.first) {
      return a.first < b.first;
    }
    if (a.second->reversePriority != b.second->reversePriority) {
      return !a.second->reversePriority && b.second->reversePriority;
    }
    return a.second->id < b.second->id;
  };
  std::set<std::pair<int64_t, EventIdNode *>, decltype(cmp)> st(cmp);

  llvm::DenseMap<EventIdNode *, int64_t> nodeValue;
  for (auto &node : nodes) {
    assignEventIds(node.get(), {});
    int64_t curNodeValue = sumAdjListSizes[node.get()] + node->eventIdNum;
    nodeValue[node.get()] = curNodeValue;
    st.emplace(curNodeValue, node.get());
  }

  llvm::SmallVector<EventIdNode *> orderedNodes;
  while (!st.empty()) {
    auto *node = st.begin()->second;
    st.erase(st.begin());
    nodeValue.erase(node);
    orderedNodes.emplace_back(node);
    for (auto [adjNode, frq] : adjList[node]) {
      if (nodeValue.contains(adjNode)) {
        ASSERT(st.count({nodeValue[adjNode], adjNode}));
        st.erase({nodeValue[adjNode], adjNode});
        nodeValue[adjNode] -= node->eventIdNum;
        st.insert({nodeValue[adjNode], adjNode});
      }
    }
  }

  int64_t eventIdMax = 0;
  for (auto *node : llvm::reverse(orderedNodes)) {
    auto chosenEventIds = getChosenEventIds(node, eventIdMax);
    ASSERT(!chosenEventIds.empty());
    assignEventIds(node, chosenEventIds);
    eventIdMax = std::max(eventIdMax, chosenEventIds.back());
    LLVM_DEBUG({ llvm::dbgs() << node->str(false) << '\n'; });
  }

  assignNeedRecalc(false);
}

void EventIdSolver::debugPrint() {
  llvm::dbgs() << "EventIdSolver:\n";
  for (auto &node : nodes) {
    llvm::dbgs() << node->str() << '\n';
    llvm::dbgs() << "adj:";
    for (auto [otherNode, frq] : adjList[node.get()]) {
      ASSERT(frq > 0);
      llvm::dbgs() << otherNode->id << ", ";
    }
    llvm::dbgs() << "\n";
  }
}

void EventIdSolver::undoActions() {
  while (!actionsStack.empty() &&
         (actionsStack.top()->actionType != ACTION_TYPE::NONE)) {
    auto action = std::move(actionsStack.top());
    LLVM_DEBUG(llvm::dbgs() << "undo: " << action->str() << "\n";);
    actionsStack.pop();
    llvm::TypeSwitch<Action *, void>(action.get())
        .Case([this](ActionAddNode *action) { removeNode(action->node); })
        .Case([this](ActionAddEdge *action) {
          removeEdge(action->node1, action->node2);
        })
        .Case([this](ActionInsertConflictPair *action) {
          eraseConflictPair(action->node, action->conflictPair);
        })
        .Case([this](ActionAssignEventIds *action) {
          assignEventIds(action->node, action->oldEventIds,
                         /*pushAction=*/false);
        })
        .Case([this](ActionAssignNeedRecalc *action) {
          assignNeedRecalc(action->oldValue, /*pushAction=*/false);
        })
        .Default([](Action *) {
          llvm_unreachable("EventIdSolver: unhandled action_type.");
        });
  }
  if (!actionsStack.empty()) {
    ASSERT(actionsStack.top()->actionType == ACTION_TYPE::NONE);
    actionsStack.pop();
  }
}
