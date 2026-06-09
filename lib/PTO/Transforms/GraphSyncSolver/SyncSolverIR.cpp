// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===---------- SyncSolverIR.cpp ---- Graph Sync Solver -------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/GraphSyncSolver/MemInfo.h"
#include "PTO/Transforms/GraphSyncSolver/Utility.h"
#include "llvm/ADT/StringExtras.h"
#include <string>

using namespace mlir;
using namespace pto::syncsolver;

namespace mlir::pto::syncsolver {

static constexpr int kStep = 2;

int OperationBase::globalIndex = 0;

static llvm::StringRef stringifyTCoreType(pto::TCoreType coreType) {
  switch (coreType) {
  case pto::TCoreType::VECTOR:
    return "VECTOR";
  case pto::TCoreType::CUBE:
    return "CUBE";
  case pto::TCoreType::CUBE_OR_VECTOR:
    return "CUBE_OR_VECTOR";
  case pto::TCoreType::CUBE_AND_VECTOR:
    return "CUBE_AND_VECTOR";
  }
  return "UNKNOWN";
}

// Map OpType enum to human-readable strings for debugging output.
std::string getOpTypeStr(OpType opType) {
  const llvm::DenseMap<OpType, std::string> conv = {
      {OpType::OPERATION, "OperationBase"},
      {OpType::PLACE_HOLDER, "PlaceHolder"},
      {OpType::SCOPE, "Scope"},
      {OpType::FUNCTION, "Function"},
      {OpType::FUNCTION_BLOCK, "FunctionBlock"},
      {OpType::LOOP, "Loop"},
      {OpType::MMAD_SCOPE, "MmadLoop"},
      {OpType::CONDITION, "Condition"},
      {OpType::BARRIER_OP, "BarrierOp"},
      {OpType::SET_FLAG_OP, "SetFlagOp"},
      {OpType::WAIT_FLAG_OP, "WaitFlagOp"},
      {OpType::RW_OPERATION, "RWOperation"},
      {OpType::MMAD_OPERATION, "MmadOp"},
      {OpType::MMAD_LOAD_L0A_OPERATION, "LoadMmadL0AOp"},
      {OpType::MMAD_LOAD_L0B_OPERATION, "LoadMmadL0BOp"},
      {OpType::MMAD_LOAD_BIAS_OPERATION, "LoadMmadBias"},
      {OpType::RW_OPERATION_END, "RW_OPERATION_END"},
  };
  return conv.at(opType);
}

bool operator<(const SyncOp &op1, const SyncOp &op2) {
  return isa<WaitFlagOp>(&op1) && isa<SetFlagOp>(&op2);
}

struct Comma {
  bool comma = false;
  std::string get() {
    std::string ret = comma ? ", " : "";
    comma = true;
    return ret;
  }
};

std::string PointerLikeInfo::str() const {
  std::string ret = "PointerLikeInfo(";
  Comma comma;
  if (addressSpace.has_value()) {
    ret += comma.get();
    ret += stringifyEnum(addressSpace.value());
  }
  ret += comma.get();
  {
    Comma comma;
    ret += "[";
    for (auto addr : addresses) {
      ret += comma.get();
      ret += std::to_string(addr);
    }
    ret += "]";
  }
  if (allocateSize.has_value()) {
    ret += comma.get();
    ret += std::to_string(allocateSize.value());
  }
  ret += ")";
  return ret;
}

std::string MemInfo::str() const {
  std::string ret = "MemInfo(";
  Comma comma;
  if (this->value) {
    ret += comma.get();
    ret += op2str(value);
  }
  if (this->pointerLikeInfo) {
    ret += comma.get();
    ret += this->pointerLikeInfo->str();
  }
  ret += ")";
  return ret;
}

// Provide readable string representations for IR nodes used in logs and dumps.
// Each specialized .str implementation documents what it prints.
std::string PlaceHolder::str(int indent, bool recursive) const {
  std::string opStr =
      (op != nullptr
           ? op2str(op)
           : llvm::convertToCamelFromSnakeCase(getOpTypeStr(this->opType))) +
      std::to_string(this->id);
  return std::string(indent, ' ') + opStr;
}

std::string Scope::str(int indent, bool recursive) const {
  std::string ret =
      std::string(indent, ' ') +
      llvm::convertToCamelFromSnakeCase(getOpTypeStr(this->opType)) +
      std::to_string(this->id);
  if (maxPreloadNum.has_value()) {
    ret += " max-preload-num=" + std::to_string(maxPreloadNum.value());
  }
  if (preloadNum.has_value()) {
    ret += " preload-num=" + std::to_string(preloadNum.value());
  }
  if (recursive) {
    ret += " {\n";
    for (auto &op : body) {
      ret += op->str(indent + kStep, true) + "\n";
    }
    ret += std::string(indent, ' ') + "}";
  }
  return ret;
}

std::string Loop::str(int indent, bool recursive) const {
  std::string ret =
      std::string(indent, ' ') +
      llvm::convertToCamelFromSnakeCase(getOpTypeStr(this->opType)) +
      std::to_string(this->id);
  if (isParallel) {
    ret += " parallel-loop";
  }
  if (multibufferUnrollNum.has_value()) {
    ret += " multibuffer-unroll-num=" +
           std::to_string(multibufferUnrollNum.value());
  }
  if (recursive) {
    ret += " {\n";
    for (auto &op : body) {
      ret += op->str(indent + kStep, true) + "\n";
    }
    ret += std::string(indent, ' ') + "}";
  }
  return ret;
}

std::string Condition::str(int indent, bool recursive) const {
  std::string ret =
      std::string(indent, ' ') +
      llvm::convertToCamelFromSnakeCase(getOpTypeStr(this->opType)) +
      std::to_string(this->id);
  if (isUnlikely) {
    ret += " unlikely-cond";
  }
  if (recursive) {
    ret += " {\n";
    for (auto &op : body) {
      if (op.get() == getTrueScope()) {
        ret += std::string(indent + kStep, ' ') +
               "(trueScope)\n";
      } else if (op.get() == getFalseScope()) {
        ret += std::string(indent + kStep, ' ') +
               "(falseScope)\n";
      }
      ret += op->str(indent + kStep, true) + "\n";
    }
    ret += std::string(indent, ' ') + "}";
  }
  return ret;
}

std::string RWOperation::str(int indent, bool recursive) const {
  std::string ret;
  std::string opStr =
      (op != nullptr
           ? op2str(op)
           : llvm::convertToCamelFromSnakeCase(getOpTypeStr(this->opType))) +
      std::to_string(this->id);
  std::string coreTypeStr;
  if (coreType != pto::TCoreType::CUBE_OR_VECTOR) {
    coreTypeStr = "[<" + stringifyTCoreType(coreType).str() + ">]";
  }
  std::string pipesStr;
  if (this->pipeRead != this->pipeWrite) {
    pipesStr = "[<" + stringifyPIPE(this->pipeRead).str() + ">, <" +
               stringifyPIPE(this->pipeRead).str() + ">]";
  } else {
    pipesStr = "[<" + stringifyPIPE(this->pipeRead).str() + ">]";
  }
  std::string unitFlag;
  if (!mergedUnitFlagInfo.disabledAsSet()) {
    std::string iteratorsStr;
    llvm::raw_string_ostream ss(iteratorsStr);
    ss << "unitFlagAsSet(";
    llvm::interleaveComma(
        mergedUnitFlagInfo.getUnitFlagModesAsSet(/*compress=*/true), ss);
    ss << ")";
    unitFlag += ss.str();
  }
  if (!mergedUnitFlagInfo.disabledAsWait()) {
    std::string iteratorsStr;
    llvm::raw_string_ostream ss(iteratorsStr);
    ss << "unitFlagAsWait(";
    llvm::interleaveComma(
        mergedUnitFlagInfo.getUnitFlagModesAsWait(/*compress=*/true), ss);
    ss << ")";
    unitFlag += (!unitFlag.empty() ? " " : "") + ss.str();
  }
  ret += std::string(indent, ' ') + opStr + " " + coreTypeStr + " " + pipesStr +
         " " + unitFlag + "\n";
  if (indent != 0) {
    for (auto memInfo : this->readMemInfo) {
      ret += std::string(indent + kStep, ' ') + "read: " + memInfo.str() + "\n";
    }
    for (auto memInfo : this->writeMemInfo) {
      ret += std::string(indent + kStep, ' ') + "write: " + memInfo.str() + "\n";
    }
  }
  ret.pop_back();
  return ret;
}

std::string SetFlagOp::str(int indent, bool recursive) const {
  std::string ret;
  ret += std::string(indent, ' ') +
         llvm::convertToCamelFromSnakeCase(getOpTypeStr(this->opType)) +
         std::to_string(this->id);
  if (this->debugId.has_value()) {
    ret += " [" + std::to_string(this->debugId.value()) + "]";
  }
  ret += " [<";
  if (this->coreType != pto::TCoreType::CUBE_OR_VECTOR) {
    ret += stringifyTCoreType(this->coreType).str() + ">, <";
  }
  ret += stringifyPIPE(this->pipeSrc).str() + ">, <" +
         stringifyPIPE(this->pipeDst).str() + ">, (";
  Comma comma;
  for (auto eventId : this->eventIds) {
    ret += comma.get() + "EVENT_ID" + llvm::itostr(eventId);
  }
  ret += ")]";
  if (allAtOnce) {
    ret += " all-at-once";
  }
  if (checkFirstIter) {
    ret += " check-first-iter";
  }
  if (checkLastIter) {
    ret += " check-last-iter";
  }
  return ret;
}

std::string WaitFlagOp::str(int indent, bool recursive) const {
  std::string ret;
  ret += std::string(indent, ' ') +
         llvm::convertToCamelFromSnakeCase(getOpTypeStr(this->opType)) +
         std::to_string(this->id);
  if (this->debugId.has_value()) {
    ret += " [" + std::to_string(this->debugId.value()) + "]";
  }
  ret += " [<";
  if (this->coreType != pto::TCoreType::CUBE_OR_VECTOR) {
    ret += stringifyTCoreType(this->coreType).str() + ">, <";
  }
  ret += stringifyPIPE(this->pipeSrc).str() + ">, <" +
         stringifyPIPE(this->pipeDst).str() + ">, (";
  Comma comma;
  for (auto eventId : this->eventIds) {
    ret += comma.get() + "EVENT_ID" + llvm::itostr(eventId);
  }
  ret += ")]";
  if (allAtOnce) {
    ret += " all-at-once";
  }
  if (checkFirstIter) {
    ret += " check-first-iter";
  }
  if (checkLastIter) {
    ret += " check-last-iter";
  }
  return ret;
}

std::string BarrierOp::str(int indent, bool recursive) const {
  std::string ret;
  ret += std::string(indent, ' ') +
         llvm::convertToCamelFromSnakeCase(getOpTypeStr(this->opType)) +
         std::to_string(this->id);
  if (this->debugId.has_value()) {
    ret += " [" + std::to_string(this->debugId.value()) + "]";
  }
  ret += " [<" + stringifyPIPE(this->pipe).str() + ">]";
  return ret;
}

std::string ConflictPair::str() const {
  std::string ret;
  ret += "ConflictPair" + std::to_string(this->id);
  ret += " (" + std::to_string(this->startIndex) + ", " +
         std::to_string(this->endIndex) + ")";
  if (this->isBarrier()) {
    ret += " [<" + stringifyPIPE(setCorePipeInfo.pipe).str() + ">]";
  } else {
    if (setCorePipeInfo.coreType != pto::TCoreType::CUBE_OR_VECTOR ||
        waitCorePipeInfo.coreType != pto::TCoreType::CUBE_OR_VECTOR) {
      ret += "[<" + stringifyTCoreType(setCorePipeInfo.coreType).str() +
             ">, <" + stringifyTCoreType(waitCorePipeInfo.coreType).str() +
             ">]";
    }
    ret += " [<" + stringifyPIPE(setCorePipeInfo.pipe).str() + ">, <" +
           stringifyPIPE(waitCorePipeInfo.pipe).str() + ">, (";
    Comma comma;
    if (this->eventIdNode != nullptr) {
      for (auto eventId : this->eventIdNode->getEventIds()) {
        ret += comma.get() + "EVENT_ID" + llvm::itostr(eventId);
      }
    }
    ret += ")]";
  }

  Comma comma;
  ret += " {";
  ret += (isBarrier() ? (comma.get() + "is-barrier") : "");
  ret += (isInnerBackward ? (comma.get() + "is-backward") : "");
  ret += (isUseless ? (comma.get() + "is-useless") : "");
  ret +=
      (replacedWithUnitFlag ? (comma.get() + "replaced-with-unit-flag") : "");
  ret += "}";

  ret += "\n";
  if (this->op1 != nullptr) {
    ret += this->op1->str(kStep, false) + '\n';
  }
  if (this->op2 != nullptr) {
    ret += this->op2->str(kStep, false) + '\n';
  }
  // ret += this->opSet->str(0, false) + '\n';
  // ret += this->opWait->str(0, false) + '\n';
  ret.pop_back();
  return ret;
}
} // namespace mlir::pto::syncsolver
