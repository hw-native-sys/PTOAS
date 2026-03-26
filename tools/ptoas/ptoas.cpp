//===- ptoas.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/A5VM.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/A5VMTextEmitter.h"
#include "PTO/Transforms/A5VMLLVMEmitter.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Diagnostics.h"
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
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include <string>

using namespace mlir;
using namespace pto;

#ifndef PTOAS_RELEASE_VERSION
#define PTOAS_RELEASE_VERSION "unknown"
#endif

static void printPTOASVersion(llvm::raw_ostream &os) {
  os << "ptoas " << PTOAS_RELEASE_VERSION << "\n";
}

// #define ADD_CANONICALIZER_PASS \
//    CanonicalizerOptions options; \
//    options.enableExtendedPattern = true; \
//    std::vector<std::string> disabledPatterns{}; \
//    options.disabledPatterns = disabledPatterns; \
//    pm.addPass(createCanonicalizerPass(options))

// #define ADD_CANONICALIZER_PASS_WITHOUT_OPTION_DEFS \
//    pm.nest<func::FuncOp>().addPass(createCanonicalizerPass(options))

// static void canonicalizationPipeline(OpPassManager &pm) {
//    pm.addPass(createArithToAffineConversionPass());
//    ADD_CANONICALIZER_PASS;
//    pm.addPass(createSCFForLoopCanonicalizationPass());
//    pm.addPass(createCSEPass());
//    ADD_CANONICALIZER_PASS_WITHOUT_OPTION_DEFS;
//    //pm.nest<func::FuncOp>().addPass(createHIVMOptSinglePointPass());
//    ADD_CANONICALIZER_PASS_WITHOUT_OPTION_DEFS;
//    pm.nest<func::FuncOp>().addPass(memref::createDeadStoreEliminationPass());
// }

static void bufferizationPipeline(OpPassManager &pm) {
  bufferization::OneShotBufferizationOptions oneShotOptions;
  oneShotOptions.bufferizeFunctionBoundaries = true;
  oneShotOptions.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);
  oneShotOptions.allowReturnAllocsFromLoops = true;
  oneShotOptions.allowUnknownOps = true;
  pm.addPass(bufferization::createOneShotBufferizePass(oneShotOptions));
  // pm.addPass(bufferization::createOneShotBufferizePass());

  // if (hivmPipelineOptions.enableVfMerge) {
  //    pm.addPass(hfusion::createMergeVecScopePass());
  // }
  // canonicalizationPipeline(pm);
  // pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
  // canonicalizationPipeline(pm);
  // pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
  pm.addPass(createConvertToPTOOpPass());
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

static llvm::cl::opt<bool> disableInferLayout(
    "disable-infer-layout",
    llvm::cl::desc("Disable PTO layout inference pass (static-only)"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> emitAddPtrTrace(
    "emit-addptr-trace",
    llvm::cl::desc("Emit addptr trace comments in generated C++ output"),
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

static llvm::cl::opt<std::string> ptoBackend(
    "pto-backend",
    llvm::cl::desc("Final PTOAS backend: emitc or a5vm (default: emitc)"),
    llvm::cl::value_desc("emitc|a5vm"),
    llvm::cl::init("emitc"));

static llvm::cl::opt<bool> a5vmPrintIR(
    "a5vm-print-ir",
    llvm::cl::desc("Print post-pass A5VM backend IR to stderr"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> a5vmLoweringStrategy(
    "a5vm-lowering-strategy",
    llvm::cl::desc("A5VM vector lowering strategy: post-update or no-post-update"),
    llvm::cl::value_desc("post-update|no-post-update"),
    llvm::cl::init("post-update"));

static llvm::cl::opt<bool> dumpA5VMIR(
    "dump-a5vm-ir",
    llvm::cl::desc("Print post-pass A5VM backend IR to stderr"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> ptoPrintSeamIR(
    "pto-print-seam-ir",
    llvm::cl::desc("Print shared pre-backend seam IR to stderr"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> ptoSeamIRFile(
    "pto-seam-ir-file",
    llvm::cl::desc("Write shared pre-backend seam IR to a file"),
    llvm::cl::value_desc("path"),
    llvm::cl::init(""));

static llvm::cl::opt<bool> a5vmPrintIntrinsics(
    "a5vm-print-intrinsics",
    llvm::cl::desc("Print A5VM intrinsic selection decisions to stderr"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> a5vmEmitHIVMText(
    "a5vm-emit-hivm-text",
    llvm::cl::desc("After lowering to A5VM IR, emit textual LLVM/HIVM instead of raw A5VM IR"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> a5vmEmitHIVMOfficialLLVM(
    "a5vm-emit-hivm-llvm",
    llvm::cl::desc("After lowering to A5VM IR, emit textual LLVM/HIVM via the official LLVM dialect export path"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> a5vmEmitHIVMOfficialBitcode(
    "a5vm-emit-hivm-bc",
    llvm::cl::desc("After lowering to A5VM IR, emit LLVM bitcode via the official LLVM dialect export path"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> a5vmAllowUnresolved(
    "a5vm-allow-unresolved",
    llvm::cl::desc("Emit explicit unresolved A5VM comments instead of failing"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> a5vmUnresolvedReport(
    "a5vm-unresolved-report",
    llvm::cl::desc("Write unresolved A5VM mappings to a sidecar report"),
    llvm::cl::value_desc("path"),
    llvm::cl::init(""));

static llvm::cl::opt<std::string> hivmUnresolvedReport(
    "hivm-unresolved-report",
    llvm::cl::desc("Write unresolved HIVM mappings to a sidecar report"),
    llvm::cl::value_desc("path"),
    llvm::cl::init(""));

enum class PTOBuildLevel {
  Level1,
  Level2,
  Level3,
};

enum class PTOBackend {
  EmitC,
  A5VM,
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

static bool parseBackend(llvm::StringRef backendStr, PTOBackend &out) {
  std::string s = backendStr.str();
  for (char &c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "emitc") {
    out = PTOBackend::EmitC;
    return true;
  }
  if (s == "a5vm") {
    out = PTOBackend::A5VM;
    return true;
  }
  return false;
}

static void addSharedPreBackendPasses(OpPassManager &pm,
                                      PTOBuildLevel effectiveLevel) {
  pm.addNestedPass<mlir::func::FuncOp>(pto::createLoweringSyncToPipePass());

  if (!disableInferLayout)
    pm.addNestedPass<mlir::func::FuncOp>(pto::createInferPTOLayoutPass());
  pm.addPass(pto::createPTOViewToMemrefPass());

  if (effectiveLevel != PTOBuildLevel::Level3) {
    PlanMemoryOptions planMemoryOption;
    planMemoryOption.memMode = MemPlanMode::LOCAL_MEM_PLAN;
    planMemoryOption.enableGlobalReuse = false;
    planMemoryOption.enablePrintMemoryAllocatedSize = false;
    pm.addPass(pto::createPlanMemoryPass(planMemoryOption));
  }

  if (enableInsertSync) {
    if (effectiveLevel == PTOBuildLevel::Level3) {
      llvm::errs() << "Warning: --enable-insert-sync is ignored because "
                      "--pto-level=level3.\n";
    } else {
      pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOInsertSyncPass());
    }
  }

  pm.addPass(createCSEPass());
}

static void addBackendLoweringPasses(OpPassManager &pm,
                                     PTOBackend effectiveBackend) {
  if (effectiveBackend == PTOBackend::A5VM) {
    pm.addPass(pto::createLowerPTOToA5VMPass(a5vmLoweringStrategy));
    pm.addPass(mlir::createCSEPass());
    return;
  }

  std::string arch = ptoTargetArch;
  for (char &c : arch)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (arch == "a3") {
    pm.addPass(pto::createEmitPTOManualPass(pto::PTOArch::A3));
  } else {
    pm.addPass(pto::createEmitPTOManualPass(pto::PTOArch::A5));
  }
  pm.addPass(emitc::createFormExpressionsPass());
  pm.addPass(mlir::createCSEPass());
}

static void printA5VMIROpSummary(ModuleOp module, llvm::raw_ostream &os) {
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    for (Operation &op : func.getBody().front().getOperations()) {
      os << "A5VM IR op: " << op.getName().getStringRef() << "\n";
      for (Region &region : op.getRegions()) {
        for (Block &block : region) {
          for (Operation &nested : block.getOperations())
            os << "A5VM IR op: " << nested.getName().getStringRef() << "\n";
        }
      }
    }
  }
}

static LogicalResult emitSharedPreBackendSeamIR(ModuleOp module,
                                                llvm::StringRef outputPath) {
  if (outputPath.empty())
    return success();

  if (outputPath == "-") {
    module->print(llvm::outs());
    llvm::outs() << "\n";
    llvm::outs().flush();
    return success();
  }

  std::error_code ec;
  llvm::ToolOutputFile outputFile(outputPath, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "Error: failed to open seam IR file '" << outputPath
                 << "': " << ec.message() << "\n";
    return failure();
  }

  module->print(outputFile.os());
  outputFile.os() << "\n";
  outputFile.keep();
  return success();
}

// --------------------------------------------------------------------------
// Post-process C++ output: rewrite marker calls into Tile member calls.
//
// We emit marker calls in EmitC IR because EmitC currently does not provide a
// first-class op for member-function invocation. After translation, we rewrite:
//   PTOAS__TILE_SET_VALUE(dst, offset, val) -> dst.SetValue(offset, val)
//   PTOAS__TILE_GET_VALUE(src, offset)      -> src.GetValue(offset)
//   PTOAS__TILE_DATA(obj)                  -> obj.data()
//   PTOAS__PTR_LOAD(ptr, offset)           -> ptr[offset]
//   PTOAS__PTR_STORE(ptr, offset, val)     -> ptr[offset] = val
// --------------------------------------------------------------------------
static bool rewriteMarkerCallToMember(std::string &cpp, llvm::StringRef marker,
                                      llvm::StringRef memberName,
                                      unsigned expectedNumArgs) {
  size_t searchPos = 0;
  bool changed = false;
  while (true) {
    size_t markerPos = cpp.find(marker.str(), searchPos);
    if (markerPos == std::string::npos)
      break;

    size_t lparenPos = markerPos + marker.size();
    if (lparenPos >= cpp.size() || cpp[lparenPos] != '(') {
      searchPos = markerPos + marker.size();
      continue;
    }

    // Find the matching ')' for this call, tracking nested parentheses.
    size_t argsBegin = lparenPos + 1;
    int parenDepth = 0;
    size_t rparenPos = std::string::npos;
    for (size_t i = argsBegin; i < cpp.size(); ++i) {
      char c = cpp[i];
      if (c == '(') {
        ++parenDepth;
      } else if (c == ')') {
        if (parenDepth == 0) {
          rparenPos = i;
          break;
        }
        --parenDepth;
      }
    }
    if (rparenPos == std::string::npos) {
      // Unbalanced parentheses; stop trying to rewrite.
      break;
    }

    llvm::StringRef argsRef(cpp.data() + argsBegin, rparenPos - argsBegin);
    llvm::SmallVector<llvm::StringRef, 4> args;
    size_t partBegin = 0;
    parenDepth = 0;
    for (size_t i = 0; i < argsRef.size(); ++i) {
      char c = argsRef[i];
      if (c == '(') {
        ++parenDepth;
      } else if (c == ')') {
        if (parenDepth > 0)
          --parenDepth;
      } else if (c == ',' && parenDepth == 0) {
        args.push_back(argsRef.slice(partBegin, i).trim());
        partBegin = i + 1;
      }
    }
    if (partBegin <= argsRef.size())
      args.push_back(argsRef.drop_front(partBegin).trim());

    if (args.size() != expectedNumArgs) {
      searchPos = rparenPos + 1;
      continue;
    }

    std::string replacement;
    replacement.reserve(marker.size() + argsRef.size() + 16);
    replacement.append(args[0].str());
    replacement.push_back('.');
    replacement.append(memberName.str());
    replacement.push_back('(');
    if (expectedNumArgs == 1) {
      // no args
    } else if (expectedNumArgs == 2) {
      replacement.append(args[1].str());
    } else if (expectedNumArgs == 3) {
      replacement.append(args[1].str());
      replacement.append(", ");
      replacement.append(args[2].str());
    }
    replacement.push_back(')');

    cpp.replace(markerPos, (rparenPos - markerPos) + 1, replacement);
    changed = true;
    searchPos = markerPos + replacement.size();
  }
  return changed;
}

static void rewriteTileGetSetValueMarkers(std::string &cpp) {
  // Keep applying until fixed-point in case rewrites shift subsequent matches.
  bool changed = true;
  while (changed) {
    changed = false;
    changed |= rewriteMarkerCallToMember(
        cpp, "PTOAS__TILE_SET_VALUE", "SetValue", /*expectedNumArgs=*/3);
    changed |= rewriteMarkerCallToMember(
        cpp, "PTOAS__TILE_GET_VALUE", "GetValue", /*expectedNumArgs=*/2);
    changed |= rewriteMarkerCallToMember(
        cpp, "PTOAS__TILE_DATA", "data", /*expectedNumArgs=*/1);
  }
}

// --------------------------------------------------------------------------
// EmitC cleanup: drop empty emitc.expression ops.
//
// After FormExpressions + CSE, EmitC expressions can become empty when their
// root op is CSE'd with an equivalent dominating value outside the expression
// region. Such expressions crash mlir::emitc::translateToCpp because
// ExpressionOp::getRootOp() returns nullptr.
// --------------------------------------------------------------------------
static void dropEmptyEmitCExpressions(Operation *rootOp) {
  llvm::SmallVector<emitc::ExpressionOp, 8> toErase;
  rootOp->walk([&](emitc::ExpressionOp expr) {
    if (expr.getRootOp())
      return;
    Block *body = expr.getBody();
    if (!body)
      return;
    auto yield = dyn_cast<emitc::YieldOp>(body->getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return;
    Value yielded = yield.getOperand(0);
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
  llvm::SmallVector<Operation *, 16> branches;
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
  size_t searchPos = 0;
  bool changed = false;
  while (true) {
    size_t markerPos = cpp.find(marker.str(), searchPos);
    if (markerPos == std::string::npos)
      break;

    size_t lparenPos = markerPos + marker.size();
    if (lparenPos >= cpp.size() || cpp[lparenPos] != '(') {
      searchPos = markerPos + marker.size();
      continue;
    }

    size_t argsBegin = lparenPos + 1;
    int parenDepth = 0;
    size_t rparenPos = std::string::npos;
    for (size_t i = argsBegin; i < cpp.size(); ++i) {
      char c = cpp[i];
      if (c == '(') {
        ++parenDepth;
      } else if (c == ')') {
        if (parenDepth == 0) {
          rparenPos = i;
          break;
        }
        --parenDepth;
      }
    }
    if (rparenPos == std::string::npos) {
      break;
    }

    llvm::StringRef argsRef(cpp.data() + argsBegin, rparenPos - argsBegin);
    llvm::SmallVector<llvm::StringRef, 4> args;
    size_t partBegin = 0;
    parenDepth = 0;
    for (size_t i = 0; i < argsRef.size(); ++i) {
      char c = argsRef[i];
      if (c == '(') {
        ++parenDepth;
      } else if (c == ')') {
        if (parenDepth > 0)
          --parenDepth;
      } else if (c == ',' && parenDepth == 0) {
        args.push_back(argsRef.slice(partBegin, i).trim());
        partBegin = i + 1;
      }
    }
    if (partBegin <= argsRef.size())
      args.push_back(argsRef.drop_front(partBegin).trim());

    if (args.size() != expectedNumArgs) {
      searchPos = rparenPos + 1;
      continue;
    }

    std::string replacement;
    if (isStore) {
      replacement = (args[0] + "[" + args[1] + "] = " + args[2]).str();
    } else {
      replacement = (args[0] + "[" + args[1] + "]").str();
    }

    cpp.replace(markerPos, (rparenPos - markerPos) + 1, replacement);
    changed = true;
    searchPos = markerPos + replacement.size();
  }
  return changed;
}

static void rewritePtrScalarMarkers(std::string &cpp) {
  bool changed = true;
  while (changed) {
    changed = false;
    changed |= rewriteMarkerCallToSubscript(
        cpp, "PTOAS__PTR_LOAD", /*expectedNumArgs=*/2, /*isStore=*/false);
    changed |= rewriteMarkerCallToSubscript(
        cpp, "PTOAS__PTR_STORE", /*expectedNumArgs=*/3, /*isStore=*/true);
  }
}

static bool rewriteAddPtrTraceMarkers(std::string &cpp, bool showTrace) {
  size_t searchPos = 0;
  bool changed = false;
  while (true) {
    size_t markerPos = cpp.find("PTOAS__ADDPTR_TRACE", searchPos);
    if (markerPos == std::string::npos)
      break;

    size_t lparenPos = markerPos + (sizeof("PTOAS__ADDPTR_TRACE") - 1);
    if (lparenPos >= cpp.size() || cpp[lparenPos] != '(') {
      searchPos = markerPos + 1;
      continue;
    }

    size_t argsBegin = lparenPos + 1;
    int parenDepth = 0;
    size_t rparenPos = std::string::npos;
    for (size_t i = argsBegin; i < cpp.size(); ++i) {
      char c = cpp[i];
      if (c == '(') {
        ++parenDepth;
      } else if (c == ')') {
        if (parenDepth == 0) {
          rparenPos = i;
          break;
        }
        --parenDepth;
      }
    }
    if (rparenPos == std::string::npos) {
      break;
    }

    llvm::StringRef argsRef(cpp.data() + argsBegin, rparenPos - argsBegin);
    llvm::SmallVector<llvm::StringRef, 4> args;
    size_t partBegin = 0;
    parenDepth = 0;
    for (size_t i = 0; i < argsRef.size(); ++i) {
      char c = argsRef[i];
      if (c == '(') {
        ++parenDepth;
      } else if (c == ')') {
        if (parenDepth > 0)
          --parenDepth;
      } else if (c == ',' && parenDepth == 0) {
        args.push_back(argsRef.slice(partBegin, i).trim());
        partBegin = i + 1;
      }
    }
    if (partBegin <= argsRef.size())
      args.push_back(argsRef.drop_front(partBegin).trim());

    if (args.size() != 3) {
      searchPos = rparenPos + 1;
      continue;
    }

    std::string replacement;
    if (showTrace) {
      replacement.reserve(64 + argsRef.size());
      replacement.append("/* ADDPTR_TRACE: ");
      replacement.append(args[0].str());
      replacement.append(" = ");
      replacement.append(args[1].str());
      replacement.append(" + ");
      replacement.append(args[2].str());
      replacement.append(" */");
    }

    size_t replaceEnd = rparenPos;
    if (!showTrace) {
      size_t i = rparenPos + 1;
      while (i < cpp.size() && std::isspace(static_cast<unsigned char>(cpp[i])))
        ++i;
      if (i < cpp.size() && cpp[i] == ';')
        replaceEnd = i;
    }

    cpp.replace(markerPos, (replaceEnd - markerPos) + 1, replacement);
    changed = true;
    searchPos = markerPos + replacement.size();
  }
  return changed;
}

static void rewriteHoistedGlobalTensorDecls(std::string &cpp) {
  // When `declareVariablesAtTop` is enabled, the C++ emitter hoists SSA value
  // declarations to the top of the function and emits assignments later. This
  // requires the C++ type to be default-constructible.
  //
  // `GlobalTensor<...>` from pto-isa does NOT have a default constructor, so a
  // hoisted declaration like:
  //   GlobalTensor<...> v42;
  // fails to compile. Initialize those hoisted temporaries with a null pointer
  // so they are constructible:
  //   GlobalTensor<...> v42(nullptr);
  //
  // We keep the assignment later; the null-initialized value is never used.
  std::string out;
  out.reserve(cpp.size() + 64);

  llvm::StringRef ref(cpp);
  while (!ref.empty()) {
    auto split = ref.split('\n');
    llvm::StringRef line = split.first;
    llvm::StringRef rest = split.second;

    llvm::StringRef trimmed = line.trim();
    bool rewritten = false;
    if (trimmed.starts_with("GlobalTensor<") && trimmed.ends_with(";") &&
        !trimmed.contains('=') && !trimmed.contains('(')) {
      llvm::StringRef decl = trimmed.drop_back().rtrim();
      size_t lastWs = decl.find_last_of(" \t");
      if (lastWs != llvm::StringRef::npos) {
        llvm::StringRef varName = decl.drop_front(lastWs + 1);
        if (varName.starts_with("v") && varName.size() > 1) {
          bool allDigits = true;
          for (char c : varName.drop_front(1)) {
            if (c < '0' || c > '9') {
              allDigits = false;
              break;
            }
          }
          if (allDigits) {
            size_t indentLen = line.find_first_not_of(" \t");
            if (indentLen == std::string::npos)
              indentLen = 0;
            llvm::StringRef indent = line.take_front(indentLen);

            out.append(indent.str());
            out.append(decl.str());
            out.append("(nullptr);");
            rewritten = true;
          }
        }
      }
    }

    if (!rewritten)
      out.append(line.str());
    if (!rest.empty())
      out.push_back('\n');
    ref = rest;
  }

  cpp.swap(out);
}

static bool containsA5VMIR(llvm::StringRef input) {
  llvm::StringRef rest = input;
  while (!rest.empty()) {
    auto split = rest.split('\n');
    llvm::StringRef line = split.first.trim();
    if (!line.starts_with("//") &&
        (line.contains("!a5vm.vec<") || line.contains("a5vm.")))
      return true;
    rest = split.second;
  }
  return false;
}

static llvm::SmallVector<llvm::StringRef> splitTopLevelModules(llvm::StringRef input) {
  llvm::SmallVector<llvm::StringRef> modules;
  size_t searchFrom = 0;
  while (true) {
    size_t modulePos = input.find("module", searchFrom);
    if (modulePos == llvm::StringRef::npos)
      break;

    size_t bracePos = input.find('{', modulePos);
    if (bracePos == llvm::StringRef::npos)
      break;

    int depth = 0;
    size_t endPos = bracePos;
    for (; endPos < input.size(); ++endPos) {
      if (input[endPos] == '{')
        ++depth;
      else if (input[endPos] == '}') {
        --depth;
        if (depth == 0) {
          ++endPos;
          break;
        }
      }
    }
    if (depth != 0)
      break;

    modules.push_back(input.slice(modulePos, endPos));
    searchFrom = endPos;
  }
  return modules;
}

static bool emitA5VMParseBundle(llvm::StringRef inputFilename, llvm::StringRef input,
                                MLIRContext &context, llvm::raw_ostream &os) {
  llvm::SmallVector<llvm::StringRef> modules = splitTopLevelModules(input);
  if (modules.empty())
    modules.push_back(input);

  bool emittedAny = false;
  for (llvm::StringRef moduleText : modules) {
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBufferCopy(moduleText, inputFilename),
        llvm::SMLoc());

    std::string diagStorage;
    llvm::raw_string_ostream diagStream(diagStorage);
    SourceMgrDiagnosticHandler diagHandler(sourceMgr, &context, diagStream);
    OwningOpRef<ModuleOp> parsedModule = parseSourceFile<ModuleOp>(sourceMgr, &context);
    diagStream.flush();

    if (!parsedModule) {
      os << diagStorage;
      if (!diagStorage.empty() && diagStorage.back() != '\n')
        os << "\n";
      emittedAny = true;
      continue;
    }

    parsedModule->print(os);
    os << "\n";
    emittedAny = true;
  }

  return emittedAny;
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

  registry.insert<mlir::a5vm::A5VMDialect>();
  registry.insert<mlir::pto::PTODialect>();
  //mlir::registerAllDialects(registry);
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  //func::registerBufferizableOpInterfaceExternalModels(registry);
  pto::registerBufferizableOpInterfaceExternalModels(registry);

  registry.insert<emitc::EmitCDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();

  llvm::cl::SetVersionPrinter(printPTOASVersion);

  // Parse command line options
  llvm::cl::ParseCommandLineOptions(argc, argv, "PTO Assembler (ptoas)\n");

  PTOBackend effectiveBackend = PTOBackend::EmitC;
  if (!parseBackend(ptoBackend, effectiveBackend)) {
    llvm::errs() << "Error: invalid --pto-backend='" << ptoBackend
                 << "'. Expected 'emitc' or 'a5vm'.\n";
    return 1;
  }

  if (a5vmEmitHIVMOfficialLLVM && a5vmEmitHIVMOfficialBitcode) {
    llvm::errs() << "Error: --a5vm-emit-hivm-llvm and --a5vm-emit-hivm-bc "
                    "cannot be used together.\n";
    return 1;
  }

  if (effectiveBackend != PTOBackend::A5VM &&
      (a5vmEmitHIVMText || a5vmEmitHIVMOfficialLLVM ||
       a5vmEmitHIVMOfficialBitcode || a5vmPrintIR || dumpA5VMIR ||
       a5vmPrintIntrinsics || a5vmAllowUnresolved ||
       !a5vmUnresolvedReport.empty() || !hivmUnresolvedReport.empty())) {
    llvm::errs() << "Error: A5VM-specific flags require "
                    "--pto-backend=a5vm.\n";
    return 1;
  }

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
  context.getOrLoadDialect<mlir::a5vm::A5VMDialect>();
  context.getOrLoadDialect<mlir::pto::PTODialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<affine::AffineDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  OwningOpRef<ModuleOp> module;
  llvm::StringRef buf = (*fileOrErr)->getBuffer();
  const bool isPTOBC = (buf.size() >= 6 && std::memcmp(buf.data(), "PTOBC\0", 6) == 0);
  const bool inputIsA5VMIR = containsA5VMIR(buf);

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
    if (effectiveBackend == PTOBackend::EmitC && inputIsA5VMIR) {
      std::error_code ec;
      llvm::ToolOutputFile outputFile(outputFilename, ec, llvm::sys::fs::OF_None);
      if (ec) {
        llvm::errs() << ec.message() << "\n";
        return 1;
      }
      if (!emitA5VMParseBundle(inputFilename, buf, context, outputFile.os())) {
        llvm::errs() << "Error: Failed to parse MLIR.\n";
        return 1;
      }
      outputFile.keep();
      return 0;
    }

    // Parse textual MLIR (.pto).
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    module = parseSourceFile<ModuleOp>(sourceMgr, &context);
    if (!module) {
      llvm::errs() << "Error: Failed to parse MLIR.\n";
      return 1;
    }
  }

  PTOBuildLevel effectiveLevel = defaultBuildLevel();
  if (!parseBuildLevel(ptoBuildLevel, effectiveLevel)) {
    llvm::errs() << "Error: invalid --pto-level='" << ptoBuildLevel
                 << "'. Expected 'level1', 'level2', or 'level3'.\n";
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

  // [Fix] ToolOutputFile Usage
  std::error_code ec;
  llvm::ToolOutputFile outputFile(outputFilename, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << ec.message() << "\n";
    return 1;
  }

  // Main PassManager
  PassManager pm(&context);

  // pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOInsertCVMovPass());
  // pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOConvertToDPSPass());
  // pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOInsertLoadStoreForMixCVPass());
  const bool skipPreBackendPasses =
      effectiveBackend == PTOBackend::A5VM && inputIsA5VMIR;
  if (!skipPreBackendPasses) {
    // bufferizationPipeline(pm);
    //pm.addPass(createInferPTOMemScopePass());
    // pm.addNestedPass<mlir::func::FuncOp>(pto::createPTORemoveRedundantBarrierPass());
    // pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOHighDimLoweringPass());
    // pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOVFloopGatherPass());
    addSharedPreBackendPasses(pm, effectiveLevel);
  }
  std::string arch = ptoTargetArch;
  for (char &c : arch)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (arch != "a3" && arch != "a5") {
    llvm::errs() << "Error: invalid --pto-arch='" << ptoTargetArch
                 << "'. Expected 'a3' or 'a5'.\n";
    return 1;
  }
  module->getOperation()->setAttr("pto.target_arch",
                                  mlir::StringAttr::get(&context, arch));
  if (!skipPreBackendPasses) {
    if (failed(pm.run(*module))) {
      llvm::errs() << "Error: shared pre-backend pass execution failed.\n";
      return 1;
    }
  }

  if (ptoPrintSeamIR || !ptoSeamIRFile.empty()) {
    if (skipPreBackendPasses) {
      llvm::errs() << "Error: shared pre-backend seam IR is unavailable when "
                      "the input is already A5VM IR.\n";
      return 1;
    }
    if (ptoPrintSeamIR) {
      module->print(llvm::errs());
      llvm::errs() << "\n";
    }
    if (failed(emitSharedPreBackendSeamIR(*module, ptoSeamIRFile)))
      return 1;
  }

  if (!skipPreBackendPasses) {
    PassManager backendPM(&context);
    addBackendLoweringPasses(backendPM, effectiveBackend);
    if (failed(backendPM.run(*module))) {
      llvm::errs() << "Error: backend lowering pass execution failed.\n";
      return 1;
    }
  }

  if (effectiveBackend == PTOBackend::A5VM) {
    if (a5vmPrintIR || dumpA5VMIR) {
      printA5VMIROpSummary(*module, llvm::errs());
      module->print(llvm::errs());
      llvm::errs() << "\n";
    }

    if (!a5vmEmitHIVMText && !a5vmEmitHIVMOfficialLLVM &&
        !a5vmEmitHIVMOfficialBitcode) {
      module->print(outputFile.os());
      outputFile.os() << "\n";
      outputFile.keep();
      return 0;
    }

    pto::A5VMEmissionOptions options;
    options.dumpA5VMIR = a5vmPrintIR || dumpA5VMIR;
    options.printIntrinsicSelections = a5vmPrintIntrinsics;
    options.allowUnresolved = a5vmAllowUnresolved;
    options.unresolvedReportPath =
        !hivmUnresolvedReport.empty() ? hivmUnresolvedReport : a5vmUnresolvedReport;
    if (arch == "a5") {
      options.targetTriple = "hiipu64-hisilicon-cce";
      options.march = "dav-c310-vec";
      options.aicoreArch = "dav-c310-vec";
      options.defaultTargetCPU = "dav-c310-vec";
      options.defaultTargetFeatures =
          "+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,"
          "+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,"
          "+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec";
    }

    LogicalResult emissionStatus =
        a5vmEmitHIVMOfficialBitcode
            ? pto::translateA5VMModuleToLLVMBitcode(*module, outputFile.os(),
                                                    options, llvm::errs())
            : a5vmEmitHIVMOfficialLLVM
            ? pto::translateA5VMModuleToLLVMText(*module, outputFile.os(),
                                                 options, llvm::errs())
            : pto::translateA5VMModuleToText(*module, outputFile.os(), options,
                                             llvm::errs());
    if (failed(emissionStatus)) {
      llvm::errs() << "Error: Failed to emit A5VM text.\n";
      return 1;
    }
    outputFile.keep();
    return 0;
  }

  dropEmptyEmitCExpressions(module.get());
  materializeControlFlowOperands(module.get());

  // Emit C++ to string, then post-process, then write to output file.
  std::string cppOutput;
  llvm::raw_string_ostream cppOS(cppOutput);
  // CFG-style lowering (e.g. scf.while -> cf.br/cf.cond_br) may introduce
  // multiple blocks, requiring variables to be declared at the top for valid
  // C++ emission.
  bool declareVariablesAtTop = false;
  for (auto func : module->getOps<func::FuncOp>()) {
    if (func.getBlocks().size() > 1) {
      declareVariablesAtTop = true;
      break;
    }
  }
  if (!declareVariablesAtTop) {
    for (auto func : module->getOps<emitc::FuncOp>()) {
      if (func.getBlocks().size() > 1) {
        declareVariablesAtTop = true;
        break;
      }
    }
  }
  if (failed(emitc::translateToCpp(*module, cppOS,
                                  /*declareVariablesAtTop=*/declareVariablesAtTop))) {
    llvm::errs() << "Error: Failed to emit C++.\n";
    return 1;
  }
  cppOS.flush();
  rewriteTileGetSetValueMarkers(cppOutput);
  rewritePtrScalarMarkers(cppOutput);
  rewriteAddPtrTraceMarkers(cppOutput, emitAddPtrTrace);
  rewriteHoistedGlobalTensorDecls(cppOutput);
  outputFile.os() << cppOutput;

  outputFile.keep(); // Success, keep the file

  return 0;
}
