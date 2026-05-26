// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <cctype>
#include <cstring>
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/FileSystem.h" // [Fix] Required for OF_None
#include "ptobc/ptobc_decode.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include <string>

using namespace mlir;
using namespace pto;

#ifndef PTOAS_RELEASE_VERSION
#define PTOAS_RELEASE_VERSION "unknown"
#endif

namespace {

constexpr unsigned kSeenCalleeInlineCapacity = 8;
constexpr int kDefaultGraphSyncSolverEventIdMax = 8;
constexpr unsigned kStringRefInlineCapacity = 4;
constexpr unsigned kEmptyExpressionInlineCapacity = 8;
constexpr unsigned kBranchInlineCapacity = 16;
constexpr size_t kMarkerCallReserveExtra = 16;
constexpr size_t kRewriteOutputReserveExtra = 64;
constexpr size_t kMarkerRewriteMinArgCount = 2;
constexpr size_t kMarkerRewriteTernaryArgCount = 3;

using StringRefVector =
    llvm::SmallVector<llvm::StringRef, kStringRefInlineCapacity>;

} // namespace

static void printPTOASVersion(llvm::raw_ostream &os) {
  os << "ptoas " << PTOAS_RELEASE_VERSION << "\n";
}

static LogicalResult reorderEmitCFunctions(ModuleOp module) {
  SmallVector<emitc::FuncOp> declarations;
  SmallVector<emitc::FuncOp> definitions;
  llvm::DenseMap<StringAttr, emitc::FuncOp> definitionsByName;

  for (auto func : module.getOps<emitc::FuncOp>()) {
    if (func.isDeclaration()) {
      declarations.push_back(func);
      continue;
    }
    definitions.push_back(func);
    definitionsByName[func.getSymNameAttr()] = func;
  }

  llvm::DenseMap<Operation *, unsigned> indegree;
  llvm::DenseMap<Operation *, SmallVector<Operation *>> outgoing;
  for (auto func : definitions)
    indegree[func.getOperation()] = 0;

  for (auto caller : definitions) {
    Operation *callerOp = caller.getOperation();
    llvm::SmallPtrSet<Operation *, kSeenCalleeInlineCapacity> seenCallees;
    bool hasCycle = false;
    caller.walk([&](emitc::CallOp call) {
      auto calleeAttr = call.getCalleeAttr();
      if (!calleeAttr)
        return;
      auto it = definitionsByName.find(calleeAttr.getLeafReference());
      if (it == definitionsByName.end())
        return;
      Operation *calleeOp = it->second.getOperation();
      if (calleeOp == callerOp) {
        hasCycle = true;
        return;
      }
      if (!seenCallees.insert(calleeOp).second)
        return;
      outgoing[calleeOp].push_back(callerOp);
      ++indegree[callerOp];
    });
    if (hasCycle) {
      return caller.emitOpError()
             << "recursive function calls are not supported for EmitC C++ "
                "emission";
    }
  }

  SmallVector<Operation *> ready;
  for (auto func : definitions) {
    if (indegree[func.getOperation()] == 0)
      ready.push_back(func.getOperation());
  }

  SmallVector<emitc::FuncOp> sortedDefinitions;
  while (!ready.empty()) {
    Operation *next = ready.front();
    ready.erase(ready.begin());
    auto nextFunc = cast<emitc::FuncOp>(next);
    sortedDefinitions.push_back(nextFunc);

    for (Operation *user : outgoing[next]) {
      unsigned &userIndegree = indegree[user];
      if (--userIndegree == 0)
        ready.push_back(user);
    }
  }

  if (sortedDefinitions.size() != definitions.size()) {
    return module.emitError()
           << "cyclic function call graph is not supported for EmitC C++ emission";
  }

  if (declarations.empty() && definitions.size() <= 1)
    return success();

  SmallVector<emitc::FuncOp> desiredOrder;
  desiredOrder.append(declarations.begin(), declarations.end());
  desiredOrder.append(sortedDefinitions.begin(), sortedDefinitions.end());

  Block &body = module.getBodyRegion().front();
  Operation *anchor = nullptr;
  for (Operation &op : body.getOperations()) {
    if (isa<emitc::FuncOp>(op)) {
      anchor = &op;
      break;
    }
  }
  if (!anchor)
    return success();

  auto advanceAnchor = [&]() {
    while (anchor) {
      anchor = anchor->getNextNode();
      if (!anchor || isa<emitc::FuncOp>(anchor))
        return;
    }
  };

  for (auto func : desiredOrder) {
    if (func.getOperation() == anchor) {
      advanceAnchor();
      continue;
    }
    if (anchor)
      func->moveBefore(anchor);
    else
      func->moveBefore(&body, body.end());
  }

  return success();
}

// --------------------------------------------------------------------------
// Command Line Options
// --------------------------------------------------------------------------
static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string> outputFilename("o",
                                                 llvm::cl::desc("Output filename"),
                                                 llvm::cl::value_desc("filename"),
                                                 llvm::cl::init("-"));

static llvm::cl::opt<bool> enableInsertSync("enable-insert-sync",
                                            llvm::cl::desc("Enable automatic synchronization insertion pass"),
                                            llvm::cl::init(false));

static llvm::cl::opt<bool> enableInjectBarrierAllSync(
    "enable-inject-barrier-all-sync",
    llvm::cl::desc("Enable conservative synchronization by inserting "
                   "pto.barrier PIPE_ALL before memory-effecting PTO pipe ops"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> enableGraphSyncSolver(
    "enable-graph-sync-solver",
    llvm::cl::desc("Enable the graph-based intra-core sync solver "
                   "(experimental). Mutually exclusive with "
                   "--enable-insert-sync and "
                   "--enable-inject-barrier-all-sync."),
    llvm::cl::init(false));

static llvm::cl::opt<int> graphSyncSolverEventIdMax(
    "graph-sync-solver-event-id-max",
    llvm::cl::desc(
        "Maximum EVENT_ID slots for the graph sync solver (default 8). "
        "Lower values exercise the PIPE_ALL coloring fallback sooner."),
    llvm::cl::init(kDefaultGraphSyncSolverEventIdMax));

static llvm::cl::opt<bool> disableInferLayout(
    "disable-infer-layout",
    llvm::cl::desc("Disable PTO layout inference pass (static-only)"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> emitAddPtrTrace(
    "emit-addptr-trace",
    llvm::cl::desc("Emit addptr trace comments in generated C++ output"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> emitMlirIR(
    "emit-pto-ir",
    llvm::cl::desc("Emit PTO IR after lowering instead of C++"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> ptoTargetArch(
    "pto-arch",
    llvm::cl::desc("Target Ascend architecture for codegen: a3 or a5 (default: a3)"),
    llvm::cl::value_desc("a3|a5"),
    llvm::cl::init("a3"));

static llvm::cl::opt<std::string> ptoBuildLevel(
    "pto-level",
    llvm::cl::desc("Build level for pass pipeline: level1, level2, or level3 (default: level2)"),
    llvm::cl::value_desc("level1|level2|level3"),
    llvm::cl::init("level2"));

enum class PTOBuildLevel {
  Level1,
  Level2,
  Level3,
};

static PTOBuildLevel defaultBuildLevel() {
  return PTOBuildLevel::Level2;
}

static bool parseBuildLevel(llvm::StringRef levelStr, PTOBuildLevel &out) {
  std::string s = levelStr.str();
  for (char &c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "level1") {
    out = PTOBuildLevel::Level1;
    return true;
  }
  if (s == "level2") {
    out = PTOBuildLevel::Level2;
    return true;
  }
  if (s == "level3") {
    out = PTOBuildLevel::Level3;
    return true;
  }
  return false;
}

static constexpr llvm::StringLiteral kAutoSyncTailPolicyBarrierAll =
    "barrier_all";
static constexpr llvm::StringLiteral kAutoSyncTailPolicyMte3ToSEvent0 =
    "setwait_mte3_to_s_event0";

static bool parseAutoSyncTailHint(llvm::StringRef hintStr, std::string &normalized) {
  std::string s = hintStr.str();
  for (char &c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "barrier-all" || s == "barrier_all" || s == "default") {
    normalized = kAutoSyncTailPolicyBarrierAll.str();
    return true;
  }
  if (s == "mte3-to-s-event0" || s == "mte3_to_s_event0" ||
      s == "setwait-mte3-to-s-event0" ||
      s == "setwait_mte3_to_s_event0") {
    normalized = kAutoSyncTailPolicyMte3ToSEvent0.str();
    return true;
  }
  return false;
}

// --------------------------------------------------------------------------
// Post-process C++ output: rewrite marker calls into Tile member calls.
// We emit marker calls in EmitC IR because EmitC currently does not provide a
// first-class op for member-function invocation. After translation, we rewrite:
//   PTOAS__TILE_SET_VALUE(dst, offset, val) -> dst.SetValue(offset, val)
//   PTOAS__TILE_GET_VALUE(src, offset)      -> src.GetValue(offset)
//   PTOAS__TILE_DATA(obj)                   -> obj.data()
//   PTOAS__TILE_SET_VALIDSHAPE(obj, r, c)   -> obj.SetValidShape(r, c)
//   PTOAS__TILE_GET_VALID_ROW(obj)          -> obj.GetValidRow()
//   PTOAS__TILE_GET_VALID_COL(obj)          -> obj.GetValidCol()
//   PTOAS__PTR_LOAD(ptr, offset)            -> ptr[offset]
//   PTOAS__PTR_STORE(ptr, offset, val)      -> ptr[offset] = val
//   PTOAS__EVENTID_ARRAY_LOAD(arr, idx)     -> arr[idx]
//   PTOAS__EVENTID_ARRAY_STORE(arr, idx, v) -> arr[idx] = v
// --------------------------------------------------------------------------
struct ParsedMarkerCall {
  size_t markerPos = std::string::npos;
  size_t rparenPos = std::string::npos;
  StringRefVector args;
};

struct MarkerRewriteSpec {
  llvm::StringRef marker;
  llvm::StringRef memberName;
  unsigned expectedNumArgs = 0;
};

struct MarkerSubscriptRewriteSpec {
  llvm::StringRef marker;
  unsigned expectedNumArgs = 0;
  bool isStore = false;
};

static bool parseMarkerArgs(llvm::StringRef argsRef,
                            llvm::SmallVectorImpl<llvm::StringRef> &args) {
  size_t partBegin = 0;
  int parenDepth = 0;
  for (size_t i = 0; i < argsRef.size(); ++i) {
    char c = argsRef[i];
    if (c == '(') {
      ++parenDepth;
      continue;
    }
    if (c == ')') {
      if (parenDepth > 0)
        --parenDepth;
      continue;
    }
    if (c == ',' && parenDepth == 0) {
      args.push_back(argsRef.slice(partBegin, i).trim());
      partBegin = i + 1;
    }
  }
  if (partBegin > argsRef.size())
    return false;
  args.push_back(argsRef.drop_front(partBegin).trim());
  return true;
}

static std::optional<ParsedMarkerCall>
findNextMarkerCall(const std::string &cpp, llvm::StringRef marker,
                   size_t searchPos) {
  ParsedMarkerCall call;
  call.markerPos = cpp.find(marker.str(), searchPos);
  if (call.markerPos == std::string::npos)
    return std::nullopt;

  size_t lparenPos = call.markerPos + marker.size();
  if (lparenPos >= cpp.size() || cpp[lparenPos] != '(')
    return ParsedMarkerCall{call.markerPos, std::string::npos, {}};

  size_t argsBegin = lparenPos + 1;
  int parenDepth = 0;
  for (size_t i = argsBegin; i < cpp.size(); ++i) {
    char c = cpp[i];
    if (c == '(') {
      ++parenDepth;
      continue;
    }
    if (c != ')')
      continue;
    if (parenDepth == 0) {
      call.rparenPos = i;
      break;
    }
    --parenDepth;
  }
  if (call.rparenPos == std::string::npos)
    return call;

  llvm::StringRef argsRef(cpp.data() + argsBegin, call.rparenPos - argsBegin);
  if (!parseMarkerArgs(argsRef, call.args))
    call.args.clear();
  return call;
}

template <typename BuildReplacementFn>
static bool rewriteMarkerCalls(std::string &cpp, llvm::StringRef marker,
                               BuildReplacementFn buildReplacement) {
  size_t searchPos = 0;
  bool changed = false;
  for (auto call = findNextMarkerCall(cpp, marker, searchPos); call;
       call = findNextMarkerCall(cpp, marker, searchPos)) {
    if (call->rparenPos == std::string::npos) {
      searchPos = call->markerPos + marker.size();
      continue;
    }

    std::optional<std::string> replacement = buildReplacement(*call);
    if (!replacement) {
      searchPos = call->rparenPos + 1;
      continue;
    }

    cpp.replace(call->markerPos, (call->rparenPos - call->markerPos) + 1,
                *replacement);
    changed = true;
    searchPos = call->markerPos + replacement->size();
  }
  return changed;
}

static bool rewriteMarkerCallToMember(std::string &cpp, llvm::StringRef marker,
                                      llvm::StringRef memberName,
                                      unsigned expectedNumArgs) {
  return rewriteMarkerCalls(
      cpp, marker, [&](const ParsedMarkerCall &call) -> std::optional<std::string> {
        if (call.args.size() != expectedNumArgs)
          return std::nullopt;

        std::string replacement;
        replacement.reserve(marker.size() + kMarkerCallReserveExtra);
        replacement.append(call.args[0].str());
        replacement.push_back('.');
        replacement.append(memberName.str());
        replacement.push_back('(');
        if (expectedNumArgs >= kMarkerRewriteMinArgCount)
          replacement.append(call.args[1].str());
        if (expectedNumArgs == kMarkerRewriteTernaryArgCount) {
          replacement.append(", ");
          replacement.append(call.args[2].str());
        }
        replacement.push_back(')');
        return replacement;
      });
}

static void rewriteMarkerCallsToMembers(
    std::string &cpp, llvm::ArrayRef<MarkerRewriteSpec> rewrites) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (const MarkerRewriteSpec &rewrite : rewrites) {
      changed |= rewriteMarkerCallToMember(cpp, rewrite.marker,
                                           rewrite.memberName,
                                           rewrite.expectedNumArgs);
    }
  }
}

static bool rewriteMarkerCallToField(std::string &cpp, llvm::StringRef marker,
                                     llvm::StringRef fieldName,
                                     size_t expectedNumArgs) {
  return rewriteMarkerCalls(
      cpp, marker, [&](const ParsedMarkerCall &call) -> std::optional<std::string> {
        if (call.args.size() != expectedNumArgs)
          return std::nullopt;
        if (call.args.empty())
          return std::nullopt;
        std::string replacement;
        replacement.reserve(call.args.front().size() + fieldName.size() + 1);
        replacement.append(call.args.front().str());
        replacement.push_back('.');
        replacement.append(fieldName.str());
        return replacement;
      });
}

static void rewriteTileGetSetValueMarkers(std::string &cpp) {
  static const MarkerRewriteSpec kTileMarkerRewrites[] = {
      {"PTOAS__TILE_SET_VALUE", "SetValue", 3},
      {"PTOAS__TILE_GET_VALUE", "GetValue", 2},
      {"PTOAS__TILE_DATA", "data", 1},
      {"PTOAS__TILE_SET_VALIDSHAPE", "SetValidShape", 3},
      {"PTOAS__TILE_GET_VALID_ROW", "GetValidRow", 1},
      {"PTOAS__TILE_GET_VALID_COL", "GetValidCol", 1},
  };
  rewriteMarkerCallsToMembers(cpp, kTileMarkerRewrites);
}

static void rewriteAsyncEventMarkers(std::string &cpp) {
  static const MarkerRewriteSpec kAsyncEventMarkerRewrites[] = {
      {"PTOAS__ASYNC_EVENT_WAIT", "Wait", 2},
      {"PTOAS__ASYNC_EVENT_TEST", "Test", 2},
  };
  rewriteMarkerCallsToMembers(cpp, kAsyncEventMarkerRewrites);
  (void)rewriteMarkerCallToField(cpp, "PTOAS__PREFETCH_CTX_SESSION",
                                 "session", 1);
}

// --------------------------------------------------------------------------
// EmitC cleanup: drop trivial emitc.expression ops.
// After FormExpressions + CSE, EmitC expressions can become invalid in two
// ways:
//   1. the root op is CSE'd away, leaving an empty expression region
//   2. the region degenerates to `emitc.yield %outer_value`, i.e. the yielded
//      value is defined outside the expression body
// Both cases crash mlir::emitc::translateToCpp because ExpressionOp expects a
// root op defined within the region.
// --------------------------------------------------------------------------
static void dropEmptyEmitCExpressions(Operation *rootOp) {
  llvm::SmallVector<emitc::ExpressionOp, kEmptyExpressionInlineCapacity>
      toErase;
  rootOp->walk([&](emitc::ExpressionOp expr) {
    Block *body = expr.getBody();
    if (!body)
      return;
    auto yield = dyn_cast<emitc::YieldOp>(body->getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return;
    Value yielded = yield.getOperand(0);
    Operation *defOp = yielded.getDefiningOp();
    bool yieldedFromOutside = !defOp || defOp->getBlock() != body;
    if (!yieldedFromOutside && expr.getRootOp())
      return;
    expr.getResult().replaceAllUsesWith(yielded);
    toErase.push_back(expr);
  });
  for (emitc::ExpressionOp expr : llvm::reverse(toErase))
    expr.erase();
}

static Attribute getDefaultEmitCVariableInitAttr(OpBuilder &builder, Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type))
    return builder.getIntegerAttr(intTy, 0);
  if (isa<IndexType>(type))
    return builder.getIndexAttr(0);
  if (auto floatTy = dyn_cast<FloatType>(type))
    return builder.getFloatAttr(floatTy, 0.0);
  if (isa<emitc::OpaqueType, emitc::PointerType>(type))
    return emitc::OpaqueAttr::get(builder.getContext(), "");
  return Attribute{};
}

// FormExpressions may inline conditions into emitc.expression, but the C++
// emitter prints cf.br/cf.cond_br operands by variable name rather than by
// recursively emitting an expression. Materialize such operands so CFG-based
// lowering (e.g. scf.while -> cf.*) stays valid.
static void materializeControlFlowOperands(Operation *rootOp) {
  llvm::SmallVector<Operation *, kBranchInlineCapacity> branches;
  rootOp->walk([&](Operation *op) {
    if (isa<cf::BranchOp, cf::CondBranchOp>(op))
      branches.push_back(op);
  });

  OpBuilder builder(rootOp->getContext());
  for (Operation *op : branches) {
    builder.setInsertionPoint(op);
    for (OpOperand &operand : op->getOpOperands()) {
      Value value = operand.get();
      auto expr = dyn_cast_or_null<emitc::ExpressionOp>(value.getDefiningOp());
      if (!expr)
        continue;

      Attribute initAttr =
          getDefaultEmitCVariableInitAttr(builder, value.getType());
      if (!initAttr)
        continue;

      Value tmp =
          builder.create<emitc::VariableOp>(op->getLoc(), value.getType(),
                                            initAttr)
              .getResult();
      builder.create<emitc::AssignOp>(op->getLoc(), tmp, value);
      operand.set(tmp);
    }
  }
}

static bool rewriteMarkerCallToSubscript(std::string &cpp, llvm::StringRef marker,
                                         unsigned expectedNumArgs,
                                         bool isStore) {
  return rewriteMarkerCalls(
      cpp, marker, [&](const ParsedMarkerCall &call) -> std::optional<std::string> {
        if (call.args.size() != expectedNumArgs)
          return std::nullopt;
        if (isStore) {
          return (call.args[0] + "[" + call.args[1] + "] = " + call.args[2])
              .str();
        }
        return (call.args[0] + "[" + call.args[1] + "]").str();
      });
}

static void rewriteMarkerCallsToSubscripts(
    std::string &cpp, llvm::ArrayRef<MarkerSubscriptRewriteSpec> rewrites) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (const MarkerSubscriptRewriteSpec &rewrite : rewrites) {
      changed |= rewriteMarkerCallToSubscript(cpp, rewrite.marker,
                                              rewrite.expectedNumArgs,
                                              rewrite.isStore);
    }
  }
}

static void rewritePtrScalarMarkers(std::string &cpp) {
  static const MarkerSubscriptRewriteSpec kPtrMarkerRewrites[] = {
      {"PTOAS__PTR_LOAD", 2, false},
      {"PTOAS__PTR_STORE", 3, true},
  };
  rewriteMarkerCallsToSubscripts(cpp, kPtrMarkerRewrites);
}

static std::string getLineIndent(llvm::StringRef line) {
  size_t firstNonSpace = line.find_first_not_of(" \t");
  if (firstNonSpace == llvm::StringRef::npos)
    return line.str();
  return line.take_front(firstNonSpace).str();
}

static bool isAICOREFunctionStart(llvm::StringRef trimmed) {
  if (trimmed.empty() || trimmed.starts_with("#") || trimmed.starts_with("//"))
    return false;
  if (!trimmed.contains("AICORE"))
    return false;
  return trimmed.contains("(");
}

static int countBraceDelta(llvm::StringRef line) {
  int delta = 0;
  for (char c : line) {
    if (c == '{')
      ++delta;
    else if (c == '}')
      --delta;
  }
  return delta;
}

static void appendScalarGMFlush(std::string &out, llvm::StringRef indent) {
  out.append(indent.str());
  out.append("pipe_barrier(PIPE_ALL);\n");
  out.append(indent.str());
  out.append("dcci((__gm__ void*)0, ENTIRE_DATA_CACHE, CACHELINE_OUT);\n");
  out.append(indent.str());
  out.append("dsb((mem_dsb_t)0);\n");
}

static bool stripScalarGMFlushMarkersFromLine(std::string &line) {
  static constexpr llvm::StringLiteral kMarker =
      "PTOAS__SCALAR_GM_STORE_FLUSH";

  bool changed = false;
  size_t searchPos = 0;
  while (true) {
    auto call = findNextMarkerCall(line, kMarker, searchPos);
    if (!call)
      break;
    if (call->rparenPos == std::string::npos) {
      searchPos = call->markerPos + kMarker.size();
      continue;
    }

    size_t eraseBegin = call->markerPos;
    while (eraseBegin > 0 &&
           (line[eraseBegin - 1] == ' ' || line[eraseBegin - 1] == '\t'))
      --eraseBegin;

    size_t eraseEnd = call->rparenPos + 1;
    while (eraseEnd < line.size() &&
           (line[eraseEnd] == ' ' || line[eraseEnd] == '\t'))
      ++eraseEnd;
    if (eraseEnd < line.size() && line[eraseEnd] == ';')
      ++eraseEnd;
    while (eraseEnd < line.size() &&
           (line[eraseEnd] == ' ' || line[eraseEnd] == '\t'))
      ++eraseEnd;

    line.erase(eraseBegin, eraseEnd - eraseBegin);
    changed = true;
    searchPos = eraseBegin;
  }
  return changed;
}

static bool previousSignificantLineIsTailFlushPoint(
    llvm::ArrayRef<std::string> lines, size_t index) {
  for (size_t i = index; i > 0; --i) {
    llvm::StringRef prev = llvm::StringRef(lines[i - 1]).trim();
    if (prev.empty())
      continue;
    return prev.starts_with("#endif // __DAV_") ||
           prev.starts_with("ptoas_auto_sync_tail(");
  }
  return false;
}

static bool previousSignificantLineIsExitOrTailFlushPoint(
    llvm::ArrayRef<std::string> lines, size_t index) {
  for (size_t i = index; i > 0; --i) {
    llvm::StringRef prev = llvm::StringRef(lines[i - 1]).trim();
    if (prev.empty())
      continue;
    return prev.starts_with("return") ||
           prev.starts_with("#endif // __DAV_") ||
           prev.starts_with("ptoas_auto_sync_tail(");
  }
  return false;
}

static std::string rewriteScalarGMStoreFlushMarkersInFunction(
    llvm::ArrayRef<std::string> functionLines, bool hasTrailingNewline) {
  bool needsScalarGMFlush = false;
  llvm::SmallVector<std::string, 32> lines;
  lines.reserve(functionLines.size());

  for (const std::string &rawLine : functionLines) {
    std::string line = rawLine;
    bool hadMarker = stripScalarGMFlushMarkersFromLine(line);
    needsScalarGMFlush |= hadMarker;
    if (hadMarker && llvm::StringRef(line).trim().empty()) {
      continue;
    }
    lines.push_back(std::move(line));
  }

  if (!needsScalarGMFlush) {
    std::string unchanged;
    unchanged.reserve(kRewriteOutputReserveExtra);
    for (size_t i = 0; i < lines.size(); ++i) {
      unchanged.append(lines[i]);
      if (i + 1 < lines.size() || hasTrailingNewline)
        unchanged.push_back('\n');
    }
    return unchanged;
  }

  std::string out;
  out.reserve(kRewriteOutputReserveExtra);
  bool inserted = false;
  size_t fallbackIndex = lines.size();
  for (size_t i = lines.size(); i > 0; --i) {
    llvm::StringRef trimmed = llvm::StringRef(lines[i - 1]).trim();
    if (trimmed.empty())
      continue;
    if (trimmed.starts_with("}"))
      fallbackIndex = i - 1;
    break;
  }

  for (size_t i = 0; i < lines.size(); ++i) {
    llvm::StringRef lineRef(lines[i]);
    llvm::StringRef trimmed = lineRef.trim();
    bool insertHere = false;
    if (trimmed.starts_with("return")) {
      insertHere = !previousSignificantLineIsTailFlushPoint(lines, i);
    } else {
      insertHere = trimmed.starts_with("#endif // __DAV_") ||
                   trimmed.starts_with("ptoas_auto_sync_tail(");
    }
    if (i == fallbackIndex &&
        !previousSignificantLineIsExitOrTailFlushPoint(lines, i))
      insertHere = true;
    if (insertHere) {
      appendScalarGMFlush(out, getLineIndent(lineRef));
      inserted = true;
    }
    out.append(lines[i]);
    if (i + 1 < lines.size() || hasTrailingNewline)
      out.push_back('\n');
  }

  if (!inserted)
    appendScalarGMFlush(out, "  ");
  return out;
}

static void rewriteScalarGMStoreFlushMarkers(std::string &cpp) {
  std::string out;
  out.reserve(cpp.size() + kRewriteOutputReserveExtra);

  llvm::SmallVector<std::string, 32> functionLines;
  bool inFunction = false;
  bool sawFunctionBrace = false;
  int braceDepth = 0;

  auto flushFunction = [&](bool hasTrailingNewline) {
    out.append(rewriteScalarGMStoreFlushMarkersInFunction(functionLines,
                                                         hasTrailingNewline));
    functionLines.clear();
    inFunction = false;
    sawFunctionBrace = false;
    braceDepth = 0;
  };

  llvm::StringRef ref(cpp);
  while (!ref.empty()) {
    auto split = ref.split('\n');
    std::string line = split.first.str();
    bool hadNewline = !split.second.empty();
    ref = split.second;

    llvm::StringRef trimmed = llvm::StringRef(line).trim();
    if (!inFunction && isAICOREFunctionStart(trimmed))
      inFunction = true;

    if (!inFunction) {
      out.append(line);
      if (hadNewline)
        out.push_back('\n');
      continue;
    }

    functionLines.push_back(std::move(line));
    int delta = countBraceDelta(functionLines.back());
    if (delta != 0)
      sawFunctionBrace = true;
    braceDepth += delta;
    if (sawFunctionBrace && braceDepth == 0)
      flushFunction(hadNewline);
  }

  if (!functionLines.empty())
    flushFunction(false);
  cpp.swap(out);
}

static void rewriteEventIdArrayMarkers(std::string &cpp) {
  static const MarkerSubscriptRewriteSpec kEventIdMarkerRewrites[] = {
      {"PTOAS__EVENTID_ARRAY_LOAD", 2, false},
      {"PTOAS__EVENTID_ARRAY_STORE", 3, true},
  };
  rewriteMarkerCallsToSubscripts(cpp, kEventIdMarkerRewrites);
}

static bool rewriteAddPtrTraceMarkers(std::string &cpp, bool showTrace) {
  size_t searchPos = 0;
  bool changed = false;
  for (auto call = findNextMarkerCall(cpp, "PTOAS__ADDPTR_TRACE", searchPos);
       call; call = findNextMarkerCall(cpp, "PTOAS__ADDPTR_TRACE", searchPos)) {
    if (call->rparenPos == std::string::npos) {
      searchPos = call->markerPos + 1;
      continue;
    }
    if (call->args.size() != kMarkerRewriteTernaryArgCount) {
      searchPos = call->rparenPos + 1;
      continue;
    }

    std::string replacement;
    if (showTrace) {
      replacement.reserve(kRewriteOutputReserveExtra);
      replacement.append("/* ADDPTR_TRACE: ");
      replacement.append(call->args[0].str());
      replacement.append(" = ");
      replacement.append(call->args[1].str());
      replacement.append(" + ");
      replacement.append(call->args[2].str());
      replacement.append(" */");
    }

    size_t replaceEnd = call->rparenPos;
    if (!showTrace) {
      size_t i = call->rparenPos + 1;
      while (i < cpp.size() && std::isspace(static_cast<unsigned char>(cpp[i])))
        ++i;
      if (i < cpp.size() && cpp[i] == ';')
        replaceEnd = i;
    }

    cpp.replace(call->markerPos, (replaceEnd - call->markerPos) + 1,
                replacement);
    changed = true;
    searchPos = call->markerPos + replacement.size();
  }
  return changed;
}

static bool isGeneratedGlobalTensorDecl(llvm::StringRef trimmed,
                                        llvm::StringRef &decl,
                                        llvm::StringRef &varName) {
  if (!trimmed.starts_with("GlobalTensor<") || !trimmed.ends_with(";") ||
      trimmed.contains('=') || trimmed.contains('(')) {
    return false;
  }

  decl = trimmed.drop_back().rtrim();
  size_t lastWs = decl.find_last_of(" \t");
  if (lastWs == llvm::StringRef::npos)
    return false;
  varName = decl.drop_front(lastWs + 1);
  if (!varName.starts_with("v") || varName.size() <= 1)
    return false;
  return llvm::all_of(varName.drop_front(1),
                      [](char c) { return std::isdigit(c); });
}

static void rewriteHoistedGlobalTensorDecls(std::string &cpp) {
  // When `declareVariablesAtTop` is enabled, the C++ emitter hoists SSA value
  // declarations to the top of the function and emits assignments later. This
  // requires the C++ type to be default-constructible.
  //
  // `GlobalTensor<...>` from pto-isa does NOT have a default constructor, so
  // hoisted declarations of that type must be rewritten with a null-pointer
  // initializer before the later assignment remains in place.
  // We keep the assignment later; the null-initialized value is never used.
  std::string out;
  out.reserve(cpp.size() + kRewriteOutputReserveExtra);

  llvm::StringRef ref(cpp);
  while (!ref.empty()) {
    auto split = ref.split('\n');
    llvm::StringRef line = split.first;
    llvm::StringRef rest = split.second;

    llvm::StringRef trimmed = line.trim();
    bool rewritten = false;
    llvm::StringRef decl;
    llvm::StringRef varName;
    if (isGeneratedGlobalTensorDecl(trimmed, decl, varName)) {
      size_t indentLen = line.find_first_not_of(" \t");
      if (indentLen == std::string::npos)
        indentLen = 0;
      llvm::StringRef indent = line.take_front(indentLen);

      out.append(indent.str());
      out.append(decl.str());
      out.append("(nullptr);");
      rewritten = true;
    }

    if (!rewritten)
      out.append(line.str());
    if (!rest.empty())
      out.push_back('\n');
    ref = rest;
  }

  cpp.swap(out);
}

namespace {
struct ConstantDeclCandidate {
  size_t declLine = 0;
  std::string indent;
  std::string type;
  bool hasInitializer = false;
  std::string initializer;
  size_t assignmentCount = 0;
  size_t assignmentLine = 0;
  std::string assignmentRhs;
};
} // namespace

static bool isGeneratedValueName(llvm::StringRef name) {
  if (!name.consume_front("v") || name.empty())
    return false;
  return llvm::all_of(name, [](char c) { return std::isdigit(c); });
}

static bool isConstFoldableScalarType(llvm::StringRef type) {
  type = type.trim();
  if (type.starts_with("const ") || type.starts_with("constexpr "))
    return false;
  return llvm::StringSwitch<bool>(type)
      .Cases("bool", "float", "double", "half", "bfloat16_t", true)
      .Cases("int8_t", "uint8_t", "int16_t", "uint16_t", true)
      .Cases("int32_t", "uint32_t", "int64_t", "uint64_t", true)
      .Default(false);
}

static bool isLiteralInitializer(llvm::StringRef rhs) {
  rhs = rhs.trim();
  if (rhs.empty())
    return false;
  if (rhs == "true" || rhs == "false" || rhs == "nullptr")
    return true;

  static const llvm::Regex kIntLiteral(
      R"(^[+-]?(0[xX][0-9A-Fa-f]+|[0-9]+)[uUlL]*$)");
  static const llvm::Regex kFloatLiteral(
      R"(^[+-]?(([0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)([eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+)[fF]?$)");
  static const llvm::Regex kHexFloatLiteral(
      R"(^[+-]?0[xX]([0-9A-Fa-f]+\.[0-9A-Fa-f]*|[0-9A-Fa-f]+|\.[0-9A-Fa-f]+)[pP][+-]?[0-9]+[fF]?$)");
  static const llvm::Regex kSpecialFloatLiteral(
      R"(^[+-]?(nan|inf)[fF]?$)");

  return kIntLiteral.match(rhs) || kFloatLiteral.match(rhs) ||
         kHexFloatLiteral.match(rhs) || kSpecialFloatLiteral.match(rhs);
}

static std::string normalizeConstInitializer(llvm::StringRef type,
                                             llvm::StringRef rhs) {
  type = type.trim();
  rhs = rhs.trim();
  if (type == "bool") {
    if (rhs == "0" || rhs == "false")
      return "false";
    if (rhs == "1" || rhs == "-1" || rhs == "true")
      return "true";
  }
  return rhs.str();
}

static bool parseConstantDeclarationLine(llvm::StringRef line,
                                         ConstantDeclCandidate &candidate,
                                         std::string &valueName) {
  llvm::StringRef trimmed = line.trim();
  if (trimmed.empty() || trimmed.starts_with("#") || trimmed.starts_with("//") ||
      !trimmed.ends_with(";"))
    return false;

  llvm::StringRef body = trimmed.drop_back().rtrim();
  if (body.starts_with("return") || body.starts_with("goto ") ||
      body.starts_with("if ") || body.starts_with("if(") ||
      body.starts_with("switch ") || body.starts_with("switch(") ||
      body.starts_with("for ") || body.starts_with("for(") ||
      body.starts_with("while ") || body.starts_with("while(") ||
      body.starts_with("case ") || body == "default")
    return false;

  llvm::StringRef lhs = body;
  llvm::StringRef rhs;
  if (size_t eqPos = body.find('='); eqPos != llvm::StringRef::npos) {
    lhs = body.take_front(eqPos).rtrim();
    rhs = body.drop_front(eqPos + 1).trim();
  }

  size_t lastWs = lhs.find_last_of(" \t");
  if (lastWs == llvm::StringRef::npos)
    return false;

  llvm::StringRef type = lhs.take_front(lastWs).rtrim();
  llvm::StringRef name = lhs.drop_front(lastWs + 1).trim();
  if (!isGeneratedValueName(name) || !isConstFoldableScalarType(type))
    return false;

  size_t indentLen = line.find_first_not_of(" \t");
  if (indentLen == llvm::StringRef::npos)
    indentLen = 0;
  candidate.indent = line.take_front(indentLen).str();
  candidate.type = type.str();
  valueName = name.str();

  if (!rhs.empty()) {
    if (!isLiteralInitializer(rhs))
      return false;
    candidate.hasInitializer = true;
    candidate.initializer = normalizeConstInitializer(type, rhs);
  }

  return true;
}

static bool parseGeneratedValueAssignment(llvm::StringRef line,
                                          llvm::StringRef &valueName,
                                          llvm::StringRef &rhs) {
  llvm::StringRef trimmed = line.trim();
  if (trimmed.empty() || trimmed.starts_with("#") || trimmed.starts_with("//") ||
      !trimmed.ends_with(";"))
    return false;

  llvm::StringRef body = trimmed.drop_back().rtrim();
  size_t eqPos = body.find('=');
  if (eqPos == llvm::StringRef::npos)
    return false;

  llvm::StringRef lhs = body.take_front(eqPos).rtrim();
  rhs = body.drop_front(eqPos + 1).trim();
  if (!isGeneratedValueName(lhs))
    return false;
  valueName = lhs;
  return true;
}

static void rewriteScalarConstantDecls(std::string &cpp) {
  llvm::SmallVector<std::string, 0> lines;
  for (llvm::StringRef ref(cpp); !ref.empty(); ref = ref.split('\n').second) {
    auto split = ref.split('\n');
    lines.push_back(split.first.str());
  }

  llvm::SmallVector<bool, 0> eraseLine(lines.size(), false);
  auto rewriteSegment = [&](size_t beginLine, size_t endLine) {
    llvm::StringMap<ConstantDeclCandidate> candidates;

    for (size_t i = beginLine; i <= endLine; ++i) {
      ConstantDeclCandidate candidate;
      std::string valueName;
      if (parseConstantDeclarationLine(lines[i], candidate, valueName)) {
        candidate.declLine = i;
        candidates[valueName] = std::move(candidate);
        continue;
      }

      llvm::StringRef assignedName;
      llvm::StringRef rhs;
      if (!parseGeneratedValueAssignment(lines[i], assignedName, rhs))
        continue;

      auto it = candidates.find(assignedName);
      if (it == candidates.end())
        continue;

      ConstantDeclCandidate &info = it->second;
      ++info.assignmentCount;
      info.assignmentLine = i;
      info.assignmentRhs = rhs.str();
    }

    for (auto &entry : candidates) {
      llvm::StringRef valueName = entry.getKey();
      ConstantDeclCandidate &info = entry.getValue();

      std::string initializer;
      if (info.hasInitializer) {
        if (info.assignmentCount != 0)
          continue;
        initializer = info.initializer;
      } else {
        if (info.assignmentCount != 1)
          continue;
        if (!isLiteralInitializer(info.assignmentRhs))
          continue;
        initializer = normalizeConstInitializer(
            info.type, llvm::StringRef(info.assignmentRhs));
        eraseLine[info.assignmentLine] = true;
      }

      lines[info.declLine] = (info.indent + "const " + info.type + " " +
                              valueName.str() + " = " + initializer + ";");
    }
  };

  int braceDepth = 0;
  size_t segmentStart = 0;
  for (size_t i = 0; i < lines.size(); ++i) {
    int depthBefore = braceDepth;
    for (char c : lines[i]) {
      if (c == '{')
        ++braceDepth;
      else if (c == '}')
        --braceDepth;
    }

    if (depthBefore == 0 && braceDepth > 0)
      segmentStart = i;
    if (depthBefore > 0 && braceDepth == 0)
      rewriteSegment(segmentStart, i);
  }

  std::string out;
  out.reserve(cpp.size());
  for (size_t i = 0; i < lines.size(); ++i) {
    if (eraseLine[i])
      continue;
    out.append(lines[i]);
    if (i + 1 != lines.size())
      out.push_back('\n');
  }
  cpp.swap(out);
}

static bool shouldDeclareVariablesAtTop(ModuleOp module) {
  auto hasMultiBlockFunc = [](auto func) { return func.getBlocks().size() > 1; };
  return llvm::any_of(module.getOps<func::FuncOp>(), hasMultiBlockFunc) ||
         llvm::any_of(module.getOps<emitc::FuncOp>(), hasMultiBlockFunc);
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::scf::SCFDialect>();

  registry.insert<mlir::pto::PTODialect>();
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  pto::registerBufferizableOpInterfaceExternalModels(registry);

  registry.insert<emitc::EmitCDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();

  llvm::cl::SetVersionPrinter(printPTOASVersion);

  bool cliArchSpecified = false;
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    if (arg == "--pto-arch" || arg.starts_with("--pto-arch=")) {
      cliArchSpecified = true;
      break;
    }
  }

  // Register all passes so that --mlir-print-ir-after/before can resolve
  // pass names like 'cse' at option-parse time.
  mlir::registerAllPasses();
  registerPTOPasses();

  // Parse command line options
  mlir::registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "PTO Assembler (ptoas)\n");

  // Read whole input first (so we can auto-detect .ptobc by magic).
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!fileOrErr) {
    llvm::errs() << "Error: Could not open input file: "
                 << fileOrErr.getError().message() << "\n";
    return 1;
  }

  MLIRContext context(registry);
  // Be tolerant: ptobc decode may materialize ops from dialects that aren't
  // explicitly registered/loaded in this tool yet.
  context.allowUnregisteredDialects(true);

  context.getOrLoadDialect<emitc::EmitCDialect>();
  context.getOrLoadDialect<mlir::pto::PTODialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<affine::AffineDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  OwningOpRef<ModuleOp> module;
  llvm::StringRef buf = (*fileOrErr)->getBuffer();
  const bool isPTOBC = (buf.size() >= 6 && std::memcmp(buf.data(), "PTOBC\0", 6) == 0);
  auto normalizeArch = [](llvm::StringRef archValue) {
    std::string normalized = archValue.str();
    for (char &c : normalized)
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return normalized;
  };
  auto detectTextualModuleArch = [&](llvm::StringRef text) -> std::optional<std::string> {
    StringRefVector matches;
    llvm::Regex archRegex(
        R"ptoarch("?(pto\.target_arch)"?[[:space:]]*=[[:space:]]*"([[:alpha:][:digit:]_]+)")ptoarch");
    if (!archRegex.match(text, &matches) || matches.size() < 3)
      return std::nullopt;
    return normalizeArch(matches[2]);
  };

  std::string arch = normalizeArch(ptoTargetArch);
  if (cliArchSpecified) {
    if (arch != "a3" && arch != "a5" && arch != "kirin9030") {
      llvm::errs() << "Error: invalid --pto-arch='" << ptoTargetArch
                   << "'. Expected 'a3', 'a5', or 'kirin9030'.\n";
      return 1;
    }
  } else if (!isPTOBC) {
    if (auto detectedArch = detectTextualModuleArch(buf))
      arch = *detectedArch;
  }
  if (arch != "a3" && arch != "a5" && arch != "kirin9030")
    arch = "a3";

  if (isPTOBC) {
    // Decode PTO bytecode directly into an MLIR module.
    llvm::ArrayRef<uint8_t> bytes(reinterpret_cast<const uint8_t *>(buf.data()), buf.size());
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
    try {
      module = ptobc::decodePTOBCToModule(bytes, context);
    } catch (...) {
      llvm::errs() << "Error: Failed to decode PTOBC.\n";
      return 1;
    }
#else
    module = ptobc::decodePTOBCToModule(bytes, context);
#endif
    if (!module) {
      llvm::errs() << "Error: Failed to decode PTOBC.\n";
      return 1;
    }
  } else {
    // Parse textual MLIR (.pto).
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    pto::ScopedPTOParserTargetArch scopedParserArch(
        &context, arch == "a5" ? pto::PTOParserTargetArch::A5
               : arch == "kirin9030" ? pto::PTOParserTargetArch::Kirin9030
                                     : pto::PTOParserTargetArch::A3);
    module = parseSourceFile<ModuleOp>(sourceMgr, &context);
    if (!module) {
      llvm::errs() << "Error: Failed to parse MLIR.\n";
      return 1;
    }
  }

  // If the CLI explicitly requested an arch, it overrides the input module.
  // Otherwise, preserve the textual module's arch when present and only fall
  // back to the effective default.
  if (cliArchSpecified || !module->getOperation()->hasAttr("pto.target_arch")) {
    module->getOperation()->setAttr("pto.target_arch",
                                    mlir::StringAttr::get(&context, arch));
  }

  PTOBuildLevel effectiveLevel = defaultBuildLevel();
  if (!parseBuildLevel(ptoBuildLevel, effectiveLevel)) {
    llvm::errs() << "Error: invalid --pto-level='" << ptoBuildLevel
                 << "'. Expected 'level1', 'level2', or 'level3'.\n";
    return 1;
  }

  bool invalidAutoSyncTailHint = false;
  module->walk([&](mlir::func::FuncOp func) {
    auto hintAttr =
        func->getAttrOfType<mlir::StringAttr>("pto.auto_sync_tail_hint");
    if (!hintAttr)
      return;

    std::string normalizedHint;
    if (!parseAutoSyncTailHint(hintAttr.getValue(), normalizedHint)) {
      func.emitError("invalid pto.auto_sync_tail_hint '")
          << hintAttr.getValue()
          << "'. Expected 'barrier-all' (or 'default') or "
             "'mte3-to-s-event0'.";
      invalidAutoSyncTailHint = true;
      return;
    }
    func->setAttr("pto.auto_sync_tail_hint",
                  mlir::StringAttr::get(&context, normalizedHint));
  });
  if (invalidAutoSyncTailHint)
    return 1;

  bool hasTAssign = false;
  module->walk([&](pto::TAssignOp) { hasTAssign = true; });

  if (hasTAssign && effectiveLevel != PTOBuildLevel::Level3) {
    llvm::errs() << "Error: pto.tassign is only supported when "
                    "--pto-level=level3.\n";
    return 1;
  }

  if (hasTAssign && enableInsertSync) {
    llvm::errs() << "Error: pto.tassign requires --enable-insert-sync to be "
                    "disabled.\n";
    return 1;
  }

  int enabledAutoSyncModes =
      (enableInsertSync ? 1 : 0) + (enableInjectBarrierAllSync ? 1 : 0) +
      (enableGraphSyncSolver ? 1 : 0);
  if (enabledAutoSyncModes > 1) {
    llvm::errs() << "Error: --enable-insert-sync, "
                    "--enable-inject-barrier-all-sync, and "
                    "--enable-graph-sync-solver are mutually exclusive.\n";
    return 1;
  }
  if (hasTAssign && enableInjectBarrierAllSync) {
    llvm::errs() << "Error: pto.tassign requires "
                    "--enable-inject-barrier-all-sync to be disabled.\n";
    return 1;
  }
  if (hasTAssign && enableGraphSyncSolver) {
    llvm::errs() << "Error: pto.tassign requires --enable-graph-sync-solver "
                    "to be disabled.\n";
    return 1;
  }

  if (effectiveLevel == PTOBuildLevel::Level3) {
    bool missing = false;
    module->walk([&](pto::AllocTileOp op) {
      if (!op.getAddr()) {
        op.emitError("requires 'addr' operand when --pto-level=level3");
        missing = true;
      }
    });
    if (missing)
      return 1;
  } else {
    bool hasAddr = false;
    module->walk([&](pto::AllocTileOp op) {
      if (op.getAddr()) {
        op.emitError(
            "unexpected 'addr' operand: only supported when --pto-level=level3");
        hasAddr = true;
      }
    });
    if (hasAddr)
      return 1;
  }

  // Main PassManager
  PassManager pm(&context);

  if (failed(applyPassManagerCLOptions(pm)))
    return 1;

  pm.addNestedPass<mlir::func::FuncOp>(
      pto::createPTOAssignDefaultFrontendPipeIdPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      pto::createPTOLowerFrontendPipeOpsPass());
  //pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOVerifyTFreePass());
  pm.addPass(pto::createPTOInferValidatePipeInitPass());
  pm.addNestedPass<mlir::func::FuncOp>(pto::createLoweringSyncToPipePass());
  if (!disableInferLayout)
    pm.addNestedPass<mlir::func::FuncOp>(pto::createInferPTOLayoutPass());
  pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOA5NormalizeTMovPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      pto::createPTOValidateIntToPtrUsesPass());
  pm.addPass(pto::createPTOViewToMemrefPass());

  if (effectiveLevel != PTOBuildLevel::Level3) {
    PlanMemoryOptions planMemoryOption;
    planMemoryOption.memMode = MemPlanMode::LOCAL_MEM_PLAN;
    planMemoryOption.enableGlobalReuse = false;
    planMemoryOption.enablePrintMemoryAllocatedSize = false;
    pm.addPass(pto::createPlanMemoryPass(planMemoryOption));
  }
  pm.addPass(pto::createPTOResolveReservedBuffersPass());

  // Conditionally add one automatic synchronization mode. Barrier-all is a
  // conservative standalone pass; InsertSync and GraphSyncSolver are set/wait
  // solvers.
  if (enableInsertSync)
    pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOInsertSyncPass());
  else if (enableInjectBarrierAllSync)
    pm.addNestedPass<mlir::func::FuncOp>(
        pto::createPTOInjectBarrierAllSyncPass());
  else if (enableGraphSyncSolver) {
    PTOGraphSyncSolverOptions graphSyncOpts;
    graphSyncOpts.eventIdNumMax = graphSyncSolverEventIdMax;
    pm.addNestedPass<mlir::func::FuncOp>(
        pto::createPTOGraphSyncSolverPass(graphSyncOpts));
  }



  std::unique_ptr<llvm::ToolOutputFile> outputFile;
  llvm::raw_ostream *outputOS = &llvm::outs();
  if (outputFilename != "-") {
    std::error_code ec;
    outputFile = std::make_unique<llvm::ToolOutputFile>(
        outputFilename, ec, llvm::sys::fs::OF_None);
    if (ec) {
      llvm::errs() << ec.message() << "\n";
      return 1;
    }
    outputOS = &outputFile->os();
  }

  if (emitMlirIR) {
    if (failed(pm.run(*module))) {
      llvm::errs() << "Error: Pass execution failed.\n";
      return 1;
    }
    module->print(*outputOS);
    if (outputFile)
      outputFile->keep();
    return 0;
  }

  pm.addPass(pto::createPTOMaterializeTileHandlesPass());
  pm.addPass(createCSEPass());
  if (arch == "a3") {
    pm.addPass(pto::createEmitPTOManualPass(pto::PTOArch::A3));
  } else if (arch == "a5") {
    pm.addPass(pto::createEmitPTOManualPass(pto::PTOArch::A5));
  } else if (arch == "kirin9030") {
    pm.addPass(pto::createEmitPTOManualPass(pto::PTOArch::Kirin9030));
  }
  pm.addPass(emitc::createFormExpressionsPass());
  pm.addPass(mlir::createCSEPass());

  if (failed(pm.run(*module))) {
    llvm::errs() << "Error: Pass execution failed.\n";
    return 1;
  }

  dropEmptyEmitCExpressions(module.get());
  materializeControlFlowOperands(module.get());
  if (failed(reorderEmitCFunctions(module.get()))) {
    llvm::errs() << "Error: Failed to order emitted functions for C++ emission.\n";
    return 1;
  }

  // Emit C++ to string, then post-process, then write to output file.
  std::string cppOutput;
  llvm::raw_string_ostream cppOS(cppOutput);
  // CFG-style lowering (e.g. scf.while -> cf.br/cf.cond_br) may introduce
  // multiple blocks, requiring variables to be declared at the top for valid
  // C++ emission.
  bool declareVariablesAtTop = shouldDeclareVariablesAtTop(*module);
  if (failed(emitc::translateToCpp(*module, cppOS,
                                  /*declareVariablesAtTop=*/declareVariablesAtTop))) {
    llvm::errs() << "Error: Failed to emit C++.\n";
    return 1;
  }
  cppOS.flush();
  rewriteTileGetSetValueMarkers(cppOutput);
  rewriteAsyncEventMarkers(cppOutput);
  rewritePtrScalarMarkers(cppOutput);
  rewriteScalarGMStoreFlushMarkers(cppOutput);
  rewriteEventIdArrayMarkers(cppOutput);
  rewriteAddPtrTraceMarkers(cppOutput, emitAddPtrTrace);
  rewriteScalarConstantDecls(cppOutput);
  rewriteHoistedGlobalTensorDecls(cppOutput);

  *outputOS << cppOutput;
  outputOS->flush();

  if (outputFile)
    outputFile->keep(); // Success, keep the file

  return 0;
}
