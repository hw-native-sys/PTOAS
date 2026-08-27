// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===--- ptoas_cpp_rewrite.cpp ---------------------------------------------------------===//
// C++ text post-processing: generated call-site markers, EmitC integer
// attribute normalization, scalar-GM flush insertion, malformed
// verbatim repair, and scalar-constant declaration hoisting.
//===----------------------------------------------------------------------===//

#include "ptoas_internal.h"

#include "ptoas.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Transforms/BufferizableOpInterfaceImpl.h"
#include "PTO/Transforms/CppPostprocess.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VPTOLLVMEmitter.h"
#include "VPTOHostStubEmission.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "ptobc/ptobc_decode.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <thread>

#include <sys/types.h>
#include <unistd.h>


using namespace mlir;
using namespace pto;

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
      if (parenDepth > 0) {
        --parenDepth;
      }
      continue;
    }
    if (c == ',' && parenDepth == 0) {
      args.push_back(argsRef.slice(partBegin, i).trim());
      partBegin = i + 1;
    }
  }
  if (partBegin > argsRef.size()) {
    return false;
  }
  args.push_back(argsRef.drop_front(partBegin).trim());
  return true;
}

static std::optional<ParsedMarkerCall>
findNextMarkerCall(const std::string &cpp, llvm::StringRef marker,
                   size_t searchPos) {
  ParsedMarkerCall call;
  call.markerPos = cpp.find(marker.str(), searchPos);
  if (call.markerPos == std::string::npos) {
    return std::nullopt;
  }

  size_t lparenPos = call.markerPos + marker.size();
  const bool missingOpeningParen =
      lparenPos >= cpp.size() || cpp[lparenPos] != '(';
  if (missingOpeningParen) {
    return ParsedMarkerCall{call.markerPos, std::string::npos, {}};
  }

  size_t argsBegin = lparenPos + 1;
  int parenDepth = 0;
  for (size_t i = argsBegin; i < cpp.size(); ++i) {
    char c = cpp[i];
    if (c == '(') {
      ++parenDepth;
      continue;
    }
    if (c != ')') {
      continue;
    }
    if (parenDepth == 0) {
      call.rparenPos = i;
      break;
    }
    --parenDepth;
  }
  if (call.rparenPos == std::string::npos) {
    return call;
  }

  llvm::StringRef argsRef(cpp.data() + argsBegin, call.rparenPos - argsBegin);
  if (!parseMarkerArgs(argsRef, call.args)) {
    call.args.clear();
  }
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
        if (call.args.size() != expectedNumArgs) {
          return std::nullopt;
        }

        std::string replacement;
        replacement.reserve(marker.size() + kMarkerCallReserveExtra);
        replacement.append(call.args[0].str());
        replacement.push_back('.');
        replacement.append(memberName.str());
        replacement.push_back('(');
        if (expectedNumArgs >= kMarkerRewriteMinArgCount) {
          replacement.append(call.args[1].str());
        }
        if (expectedNumArgs == kMarkerRewriteTernaryArgCount) {
          replacement.append(", ");
          replacement.append(call.args[kThirdMarkerArgumentIndex].str());
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
        if (call.args.size() != expectedNumArgs) {
          return std::nullopt;
        }
        if (call.args.empty()) {
          return std::nullopt;
        }
        std::string replacement;
        replacement.reserve(call.args.front().size() + fieldName.size() + 1);
        replacement.append(call.args.front().str());
        replacement.push_back('.');
        replacement.append(fieldName.str());
        return replacement;
      });
}

void rewriteTileGetSetValueMarkers(std::string &cpp) {
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

void rewriteAsyncEventMarkers(std::string &cpp) {
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
void dropEmptyEmitCExpressions(Operation *rootOp) {
  llvm::SmallVector<emitc::ExpressionOp, kEmptyExpressionInlineCapacity>
      toErase;
  rootOp->walk([&](emitc::ExpressionOp expr) {
    Block *body = expr.getBody();
    if (!body) {
      return;
    }
    auto yield = dyn_cast<emitc::YieldOp>(body->getTerminator());
    if (!yield || yield.getNumOperands() != 1) {
      return;
    }
    Value yielded = yield.getOperand(0);
    Operation *defOp = yielded.getDefiningOp();
    bool yieldedFromOutside = !defOp || defOp->getBlock() != body;
    if (!yieldedFromOutside && expr.getRootOp()) {
      return;
    }
    expr.getResult().replaceAllUsesWith(yielded);
    toErase.push_back(expr);
  });
  for (emitc::ExpressionOp expr : llvm::reverse(toErase)) {
    expr.erase();
  }
}

static void appendEmitCIntegerAttrLiteral(std::string &storage,
                                          const APInt &value, bool isUnsigned) {
  if (value.getBitWidth() == 0) {
    storage.append("0");
    return;
  }
  if (value.getBitWidth() == 1) {
    storage.append(value.getBoolValue() ? "true" : "false");
    return;
  }

  SmallString<kRawStringInlineCapacity> strValue;
  value.toString(strValue, kEmitCIntegerRadix, !isUnsigned, false);
  storage.append(strValue.data(), strValue.size());
}

static bool shouldPrintEmitCIntegerAttrAsUnsigned(IntegerAttr attr) {
  auto intTy = dyn_cast<IntegerType>(attr.getType());
  return intTy && intTy.getSignedness() == IntegerType::Unsigned;
}

static std::string getEmitCIntegerAttrLiteral(IntegerAttr attr) {
  std::string literal;
  appendEmitCIntegerAttrLiteral(literal, attr.getValue(),
                                shouldPrintEmitCIntegerAttrAsUnsigned(attr));
  return literal;
}

static std::optional<std::string>
getEmitCDenseIntElementsAttrLiteral(DenseIntElementsAttr attr) {
  auto tensorTy = dyn_cast<TensorType>(attr.getType());
  if (!tensorTy) {
    return std::nullopt;
  }

  Type elementType = tensorTy.getElementType();
  bool isUnsigned = false;
  if (auto intTy = dyn_cast<IntegerType>(elementType)) {
    isUnsigned = intTy.getSignedness() == IntegerType::Unsigned;
  } else if (!isa<IndexType>(elementType)) {
    return std::nullopt;
  }

  std::string literal;
  literal.push_back('{');
  bool first = true;
  for (const APInt &value : attr) {
    if (!first) {
      literal.append(", ");
    }
    first = false;
    appendEmitCIntegerAttrLiteral(literal, value, isUnsigned);
  }
  literal.push_back('}');
  return literal;
}

static Attribute normalizeEmitCPrintedAttrForCppEmission(MLIRContext *ctx,
                                                         Attribute attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    return emitc::OpaqueAttr::get(ctx, getEmitCIntegerAttrLiteral(intAttr));
  }

  if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(attr)) {
    if (std::optional<std::string> literal =
            getEmitCDenseIntElementsAttrLiteral(denseAttr))
      return emitc::OpaqueAttr::get(ctx, *literal);
  }

  if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
    SmallVector<Attribute> normalized;
    normalized.reserve(arrayAttr.size());
    bool changed = false;
    for (Attribute element : arrayAttr) {
      Attribute normalizedElement =
          normalizeEmitCPrintedAttrForCppEmission(ctx, element);
      changed |= normalizedElement != element;
      normalized.push_back(normalizedElement);
    }
    if (changed) {
      return ArrayAttr::get(ctx, normalized);
    }
  }

  return attr;
}

static IntegerAttr normalizeEmitCIndexPlaceholderAttr(MLIRContext *ctx,
                                                      IntegerAttr attr) {
  const APInt &value = attr.getValue();
  int64_t index = value.getBitWidth() == 0 ? 0 : value.getSExtValue();
  return IntegerAttr::get(IndexType::get(ctx),
                          APInt(kIndexBitWidth, index));
}

static ArrayAttr normalizeEmitCCallArgsForCppEmission(MLIRContext *ctx,
                                                      ArrayAttr args) {
  SmallVector<Attribute> normalized;
  normalized.reserve(args.size());
  bool changed = false;

  for (Attribute attr : args) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      if (isa<IndexType>(intAttr.getType())) {
        Attribute normalizedAttr =
            normalizeEmitCIndexPlaceholderAttr(ctx, intAttr);
        changed |= normalizedAttr != attr;
        normalized.push_back(normalizedAttr);
        continue;
      }

      Attribute normalizedAttr =
          normalizeEmitCPrintedAttrForCppEmission(ctx, attr);
      changed |= normalizedAttr != attr;
      normalized.push_back(normalizedAttr);
      continue;
    }

    Attribute normalizedAttr =
        normalizeEmitCPrintedAttrForCppEmission(ctx, attr);
    changed |= normalizedAttr != attr;
    normalized.push_back(normalizedAttr);
  }

  return changed ? ArrayAttr::get(ctx, normalized) : args;
}

static ArrayAttr normalizeEmitCTemplateArgsForCppEmission(MLIRContext *ctx,
                                                          ArrayAttr args) {
  SmallVector<Attribute> normalized;
  normalized.reserve(args.size());
  bool changed = false;

  for (Attribute attr : args) {
    Attribute normalizedAttr =
        normalizeEmitCPrintedAttrForCppEmission(ctx, attr);
    changed |= normalizedAttr != attr;
    normalized.push_back(normalizedAttr);
  }

  return changed ? ArrayAttr::get(ctx, normalized) : args;
}

void normalizeEmitCIntegerAttrsForCppEmission(Operation *rootOp) {
  MLIRContext *ctx = rootOp->getContext();
  rootOp->walk([&](Operation *op) {
    if (auto constant = dyn_cast<emitc::ConstantOp>(op)) {
      Attribute value = constant.getValue();
      Attribute normalized =
          normalizeEmitCPrintedAttrForCppEmission(ctx, value);
      if (normalized != value) {
        constant.getProperties().setValue(normalized);
      }
      return;
    }

    if (auto variable = dyn_cast<emitc::VariableOp>(op)) {
      Attribute value = variable.getValue();
      Attribute normalized =
          normalizeEmitCPrintedAttrForCppEmission(ctx, value);
      if (normalized != value) {
        variable.getProperties().setValue(normalized);
      }
      return;
    }

    if (auto global = dyn_cast<emitc::GlobalOp>(op)) {
      std::optional<Attribute> initialValue = global.getInitialValue();
      if (!initialValue) {
        return;
      }
      Attribute normalized =
          normalizeEmitCPrintedAttrForCppEmission(ctx, *initialValue);
      if (normalized != *initialValue) {
        global.getProperties().setInitialValue(normalized);
      }
      return;
    }

    if (auto call = dyn_cast<emitc::CallOpaqueOp>(op)) {
      if (std::optional<ArrayAttr> args = call.getArgs()) {
        ArrayAttr normalized = normalizeEmitCCallArgsForCppEmission(ctx, *args);
        if (normalized != *args) {
          call.getProperties().setArgs(normalized);
        }
      }
      if (std::optional<ArrayAttr> templateArgs = call.getTemplateArgs()) {
        ArrayAttr normalized =
            normalizeEmitCTemplateArgsForCppEmission(ctx, *templateArgs);
        if (normalized != *templateArgs) {
          call.getProperties().setTemplateArgs(normalized);
        }
      }
      return;
    }
  });
}

static Attribute getDefaultEmitCVariableInitAttr(OpBuilder &builder, Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type)) {
    if (intTy.getWidth() == 0) {
      return emitc::OpaqueAttr::get(builder.getContext(), "0");
    }
    return builder.getIntegerAttr(intTy, 0);
  }
  if (isa<IndexType>(type)) {
    return builder.getIndexAttr(0);
  }
  if (auto floatTy = dyn_cast<FloatType>(type)) {
    return builder.getFloatAttr(floatTy, 0.0);
  }
  if (isa<emitc::OpaqueType, emitc::PointerType>(type)) {
    return emitc::OpaqueAttr::get(builder.getContext(), "");
  }
  return Attribute{};
}

static Type getEmitCVariableStorageType(Type valueType) {
  return valueType;
}

// FormExpressions may inline conditions into emitc.expression, but the C++
// emitter prints cf.br/cf.cond_br operands by variable name rather than by
// recursively emitting an expression. Materialize such operands so CFG-based
// lowering (e.g. scf.while -> cf.*) stays valid.
void materializeControlFlowOperands(Operation *rootOp) {
  llvm::SmallVector<Operation *, kBranchInlineCapacity> branches;
  rootOp->walk([&](Operation *op) {
    if (isa<cf::BranchOp, cf::CondBranchOp>(op)) {
      branches.push_back(op);
    }
  });

  OpBuilder builder(rootOp->getContext());
  for (Operation *op : branches) {
    builder.setInsertionPoint(op);
    for (OpOperand &operand : op->getOpOperands()) {
      Value value = operand.get();
      auto expr = dyn_cast_or_null<emitc::ExpressionOp>(value.getDefiningOp());
      if (!expr) {
        continue;
      }

      Attribute initAttr =
          getDefaultEmitCVariableInitAttr(builder, value.getType());
      if (!initAttr) {
        continue;
      }

      Value tmp = builder
                      .create<emitc::VariableOp>(
                          op->getLoc(), getEmitCVariableStorageType(value.getType()),
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
        if (call.args.size() != expectedNumArgs) {
          return std::nullopt;
        }
        std::string replacement;
        replacement.reserve(call.args[0].size() + call.args[1].size() +
                            kMarkerReplacementReserveExtra +
                            (isStore
                                 ? call.args[kThirdMarkerArgumentIndex].size()
                                 : 0));
        replacement.push_back('(');
        replacement.append(call.args[0].str());
        replacement.push_back(')');
        replacement.push_back('[');
        replacement.append(call.args[1].str());
        replacement.push_back(']');
        if (isStore) {
          replacement.append(" = ");
          replacement.append(call.args[kThirdMarkerArgumentIndex].str());
        }
        return replacement;
      });
}

void rewriteGlobalTensorMetadataMarkers(std::string &cpp) {
  auto rewrite = [&](llvm::StringRef marker, llvm::StringRef method) {
    (void)rewriteMarkerCalls(
        cpp, marker,
        [&](const ParsedMarkerCall &call) -> std::optional<std::string> {
          const bool hasExpectedArgs =
              call.args.size() == kMarkerRewriteMinArgCount;
          if (!hasExpectedArgs) {
            return std::nullopt;
          }
          return ("(" + call.args[0] + ")." + method +
                  "(static_cast<int>(" + call.args[1] + "))")
              .str();
        });
  };
  rewrite("PTOAS__GLOBAL_TENSOR_GET_SHAPE", "GetShape");
  rewrite("PTOAS__GLOBAL_TENSOR_GET_STRIDE", "GetStride");
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

void rewritePtrScalarMarkers(std::string &cpp) {
  static const MarkerSubscriptRewriteSpec kPtrMarkerRewrites[] = {
      {"PTOAS__PTR_LOAD", 2, false},
      {"PTOAS__PTR_STORE", 3, true},
  };
  rewriteMarkerCallsToSubscripts(cpp, kPtrMarkerRewrites);
}

static std::string getLineIndent(llvm::StringRef line) {
  size_t firstNonSpace = line.find_first_not_of(" \t");
  if (firstNonSpace == llvm::StringRef::npos) {
    return line.str();
  }
  return line.take_front(firstNonSpace).str();
}

static bool isAICOREFunctionStart(llvm::StringRef trimmed) {
  const bool isCommentOrEmpty =
      trimmed.empty() || trimmed.starts_with("#") ||
      trimmed.starts_with("//");
  if (isCommentOrEmpty) {
    return false;
  }
  if (!trimmed.contains("AICORE")) {
    return false;
  }
  return trimmed.contains("(");
}

static int countBraceDelta(llvm::StringRef line) {
  int delta = 0;
  for (char c : line) {
    if (c == '{') {
      ++delta;
    } else if (c == '}') {
      --delta;
    }
  }
  return delta;
}

static void appendScalarGMFlush(std::string &out, llvm::StringRef indent) {
  out.append(indent.str());
  out.append("pipe_barrier(PIPE_ALL);\n");
  out.append(indent.str());
  out.append("dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);\n");
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
    if (!call) {
      break;
    }
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
           (line[eraseEnd] == ' ' || line[eraseEnd] == '\t')) {
      ++eraseEnd;
    }
    if (eraseEnd < line.size() && line[eraseEnd] == ';') {
      ++eraseEnd;
    }
    while (eraseEnd < line.size() &&
           (line[eraseEnd] == ' ' || line[eraseEnd] == '\t')) {
      ++eraseEnd;
    }

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
    if (prev.empty()) {
      continue;
    }
    return prev.starts_with("#endif // __DAV_") ||
           prev.starts_with("ptoas_auto_sync_tail(");
  }
  return false;
}

static bool previousSignificantLineIsExitOrTailFlushPoint(
    llvm::ArrayRef<std::string> lines, size_t index) {
  for (size_t i = index; i > 0; --i) {
    llvm::StringRef prev = llvm::StringRef(lines[i - 1]).trim();
    if (prev.empty()) {
      continue;
    }
    return prev.starts_with("return") ||
           prev.starts_with("#endif // __DAV_") ||
           prev.starts_with("ptoas_auto_sync_tail(");
  }
  return false;
}

static llvm::SmallVector<std::string, kFunctionLineInlineCapacity>
stripScalarGMFlushMarkers(llvm::ArrayRef<std::string> functionLines,
                          bool &needsScalarGMFlush) {
  llvm::SmallVector<std::string, kFunctionLineInlineCapacity> lines;
  lines.reserve(functionLines.size());
  for (const std::string &rawLine : functionLines) {
    std::string line = rawLine;
    const bool hadMarker = stripScalarGMFlushMarkersFromLine(line);
    needsScalarGMFlush |= hadMarker;
    if (hadMarker && llvm::StringRef(line).trim().empty()) {
      continue;
    }
    lines.push_back(std::move(line));
  }
  return lines;
}

static std::string joinScalarGMFlushLines(llvm::ArrayRef<std::string> lines,
                                          bool hasTrailingNewline) {
  std::string output;
  output.reserve(kRewriteOutputReserveExtra);
  for (size_t i = 0; i < lines.size(); ++i) {
    output.append(lines[i]);
    const bool needsNewline = i + 1 < lines.size() || hasTrailingNewline;
    if (needsNewline) {
      output.push_back('\n');
    }
  }
  return output;
}

static size_t findScalarGMFlushFallbackIndex(
    llvm::ArrayRef<std::string> lines) {
  for (size_t i = lines.size(); i > 0; --i) {
    llvm::StringRef trimmed = llvm::StringRef(lines[i - 1]).trim();
    if (trimmed.empty()) {
      continue;
    }
    return trimmed.starts_with("}") ? i - 1 : lines.size();
  }
  return lines.size();
}

static bool shouldInsertScalarGMFlush(llvm::ArrayRef<std::string> lines,
                                       size_t index, size_t fallbackIndex) {
  llvm::StringRef trimmed = llvm::StringRef(lines[index]).trim();
  bool insertHere = false;
  if (trimmed.starts_with("return")) {
    insertHere = !previousSignificantLineIsTailFlushPoint(lines, index);
  } else {
    insertHere = trimmed.starts_with("#endif // __DAV_") ||
                 trimmed.starts_with("ptoas_auto_sync_tail(");
  }
  if (index == fallbackIndex &&
      !previousSignificantLineIsExitOrTailFlushPoint(lines, index)) {
    insertHere = true;
  }
  return insertHere;
}

static std::string rewriteScalarGMStoreFlushMarkersInFunction(
    llvm::ArrayRef<std::string> functionLines, bool hasTrailingNewline) {
  bool needsScalarGMFlush = false;
  llvm::SmallVector<std::string, kFunctionLineInlineCapacity> lines =
      stripScalarGMFlushMarkers(functionLines, needsScalarGMFlush);

  if (!needsScalarGMFlush) {
    return joinScalarGMFlushLines(lines, hasTrailingNewline);
  }

  std::string out;
  out.reserve(kRewriteOutputReserveExtra);
  bool inserted = false;
  const size_t fallbackIndex = findScalarGMFlushFallbackIndex(lines);

  for (size_t i = 0; i < lines.size(); ++i) {
    llvm::StringRef lineRef(lines[i]);
    if (shouldInsertScalarGMFlush(lines, i, fallbackIndex)) {
      appendScalarGMFlush(out, getLineIndent(lineRef));
      inserted = true;
    }
    out.append(lines[i]);
    if (i + 1 < lines.size() || hasTrailingNewline) {
      out.push_back('\n');
    }
  }

  if (!inserted) {
    appendScalarGMFlush(out, "  ");
  }
  return out;
}

void rewriteScalarGMStoreFlushMarkers(std::string &cpp) {
  std::string out;
  out.reserve(cpp.size() + kRewriteOutputReserveExtra);

  llvm::SmallVector<std::string, kFunctionLineInlineCapacity> functionLines;
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
    if (!inFunction && isAICOREFunctionStart(trimmed)) {
      inFunction = true;
    }

    if (!inFunction) {
      out.append(line);
      if (hadNewline) {
        out.push_back('\n');
      }
      continue;
    }

    functionLines.push_back(std::move(line));
    int delta = countBraceDelta(functionLines.back());
    if (delta != 0) {
      sawFunctionBrace = true;
    }
    braceDepth += delta;
    if (sawFunctionBrace && braceDepth == 0) {
      flushFunction(hadNewline);
    }
  }

  if (!functionLines.empty()) {
    flushFunction(false);
  }
  cpp.swap(out);
}

void rewriteEventIdArrayMarkers(std::string &cpp) {
  static const MarkerSubscriptRewriteSpec kEventIdMarkerRewrites[] = {
      {"PTOAS__EVENTID_ARRAY_LOAD", 2, false},
      {"PTOAS__EVENTID_ARRAY_STORE", 3, true},
  };
  rewriteMarkerCallsToSubscripts(cpp, kEventIdMarkerRewrites);
}

static bool isPreprocessorDirectiveLine(llvm::StringRef trimmedLine) {
  return trimmedLine.starts_with("#");
}

static bool normalizeMalformedVerbatimLine(llvm::StringRef line,
                                           bool prevWasPreprocessorDirective,
                                           std::string &current) {
  current = line.str();
  llvm::StringRef trimmed = llvm::StringRef(current).trim();
  if (trimmed == ";" && prevWasPreprocessorDirective) {
    return false;
  }

  const bool hasTrailingDirectiveSemicolon =
      isPreprocessorDirectiveLine(trimmed) && trimmed.ends_with(";");
  const bool hasDuplicateStatementSemicolon =
      !trimmed.empty() && !trimmed.starts_with("//") &&
      !trimmed.starts_with("/*") && trimmed.ends_with(";;");
  if (hasTrailingDirectiveSemicolon || hasDuplicateStatementSemicolon) {
    size_t semicolonPos = current.find_last_of(';');
    if (semicolonPos != std::string::npos) {
      current.erase(semicolonPos, 1);
    }
  }
  return true;
}

// Nested emitc.verbatim ops inside emitc.for / emitc.if regions currently
// pick up an extra trailing semicolon from EmitC C++ emission, which produces
// invalid lines such as `#if defined(__DAV_VEC__);` and `set_mask_norm();;`.
// Trim only those malformed suffixes here so bisheng can compile the emitted
// source until the upstream printer behavior is fixed.
void rewriteMalformedVerbatimSemicolons(std::string &cpp) {
  if (cpp.empty()) {
    return;
  }

  llvm::StringRef input(cpp);
  std::string rewritten;
  rewritten.reserve(cpp.size());

  bool prevWasPreprocessorDirective = false;
  size_t offset = 0;
  while (offset < input.size()) {
    size_t newlinePos = input.find('\n', offset);
    bool hasNewline = newlinePos != llvm::StringRef::npos;
    llvm::StringRef line =
        hasNewline ? input.slice(offset, newlinePos) : input.drop_front(offset);
    std::string current;
    if (!normalizeMalformedVerbatimLine(line, prevWasPreprocessorDirective,
                                        current)) {
      // `#endif ...` in nested verbatim blocks currently materializes as the
      // directive line followed by a standalone `;` on the next line.
      prevWasPreprocessorDirective = false;
    } else {
      rewritten.append(current);
      if (hasNewline) {
        rewritten.push_back('\n');
      }
      prevWasPreprocessorDirective =
          isPreprocessorDirectiveLine(llvm::StringRef(current).trim());
    }

    if (!hasNewline) {
      break;
    }
    offset = newlinePos + 1;
  }

  cpp.swap(rewritten);
}

bool rewriteAddPtrTraceMarkers(std::string &cpp, bool showTrace) {
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
      replacement.append(call->args[kThirdMarkerArgumentIndex].str());
      replacement.append(" */");
    }

    size_t replaceEnd = call->rparenPos;
    if (!showTrace) {
      size_t i = call->rparenPos + 1;
      while (i < cpp.size() && std::isspace(static_cast<unsigned char>(cpp[i]))) {
        ++i;
      }
      if (i < cpp.size() && cpp[i] == ';') {
        replaceEnd = i;
      }
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
  if (lastWs == llvm::StringRef::npos) {
    return false;
  }
  varName = decl.drop_front(lastWs + 1);
  if (!varName.starts_with("v") || varName.size() <= 1) {
    return false;
  }
  return llvm::all_of(varName.drop_front(1),
                      [](char c) { return std::isdigit(c); });
}

void rewriteHoistedGlobalTensorDecls(std::string &cpp) {
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
      if (indentLen == std::string::npos) {
        indentLen = 0;
      }
      llvm::StringRef indent = line.take_front(indentLen);

      out.append(indent.str());
      out.append(decl.str());
      out.append("(nullptr);");
      rewritten = true;
    }

    if (!rewritten) {
      out.append(line.str());
    }
    if (!rest.empty()) {
      out.push_back('\n');
    }
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

bool isGeneratedValueName(llvm::StringRef name) {
  if (!name.consume_front("v") || name.empty()) {
    return false;
  }
  return llvm::all_of(name, [](char c) { return std::isdigit(c); });
}

static bool isConstFoldableScalarType(llvm::StringRef type) {
  type = type.trim();
  if (type.starts_with("const ") || type.starts_with("constexpr ")) {
    return false;
  }
  return llvm::StringSwitch<bool>(type)
      .Cases("bool", "float", "double", "half", "bfloat16_t", true)
      .Cases("int8_t", "uint8_t", "int16_t", "uint16_t", true)
      .Cases("int32_t", "uint32_t", "int64_t", "uint64_t", true)
      .Default(false);
}

static bool isLiteralInitializer(llvm::StringRef rhs) {
  rhs = rhs.trim();
  if (rhs.empty()) {
    return false;
  }
  if (rhs == "true" || rhs == "false" || rhs == "nullptr") {
    return true;
  }

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
    if (rhs == "0" || rhs == "false") {
      return "false";
    }
    if (rhs == "1" || rhs == "-1" || rhs == "true") {
      return "true";
    }
  }
  return rhs.str();
}

static bool startsWithAny(llvm::StringRef value,
                          llvm::ArrayRef<llvm::StringRef> prefixes) {
  for (llvm::StringRef prefix : prefixes) {
    if (value.starts_with(prefix)) {
      return true;
    }
  }
  return false;
}

static bool isNonDeclarationStatement(llvm::StringRef body) {
  static constexpr llvm::StringRef kStatementPrefixes[] = {
      "return", "go" "to ", "if ",    "if(",     "switch ", "switch(",
      "for ",   "for(",    "while ", "while(",  "case "};
  return body == "default" || startsWithAny(body, kStatementPrefixes);
}

static bool splitConstantDeclaration(llvm::StringRef body,
                                     llvm::StringRef &type,
                                     llvm::StringRef &name,
                                     llvm::StringRef &rhs) {
  llvm::StringRef lhs = body;
  rhs = llvm::StringRef();
  if (size_t eqPos = body.find('='); eqPos != llvm::StringRef::npos) {
    lhs = body.take_front(eqPos).rtrim();
    rhs = body.drop_front(eqPos + 1).trim();
  }

  size_t lastWs = lhs.find_last_of(" \t");
  if (lastWs == llvm::StringRef::npos) {
    return false;
  }
  type = lhs.take_front(lastWs).rtrim();
  name = lhs.drop_front(lastWs + 1).trim();
  return true;
}

static bool parseConstantInitializer(llvm::StringRef type,
                                     llvm::StringRef rhs,
                                     ConstantDeclCandidate &candidate) {
  if (rhs.empty()) {
    return true;
  }
  if (!isLiteralInitializer(rhs)) {
    return false;
  }
  candidate.hasInitializer = true;
  candidate.initializer = normalizeConstInitializer(type, rhs);
  return true;
}

static bool parseConstantDeclarationLine(llvm::StringRef line,
                                         ConstantDeclCandidate &candidate,
                                         std::string &valueName) {
  llvm::StringRef trimmed = line.trim();
  if (trimmed.empty() || trimmed.starts_with("#") || trimmed.starts_with("//") ||
      !trimmed.ends_with(";")) {
    return false;
  }

  llvm::StringRef body = trimmed.drop_back().rtrim();
  if (isNonDeclarationStatement(body)) {
    return false;
  }

  llvm::StringRef type;
  llvm::StringRef name;
  llvm::StringRef rhs;
  if (!splitConstantDeclaration(body, type, name, rhs)) {
    return false;
  }
  if (!isGeneratedValueName(name) || !isConstFoldableScalarType(type)) {
    return false;
  }

  size_t indentLen = line.find_first_not_of(" \t");
  if (indentLen == llvm::StringRef::npos) {
    indentLen = 0;
  }
  candidate.indent = line.take_front(indentLen).str();
  candidate.type = type.str();
  valueName = name.str();
  return parseConstantInitializer(type, rhs, candidate);
}

static bool parseGeneratedValueAssignment(llvm::StringRef line,
                                          llvm::StringRef &valueName,
                                          llvm::StringRef &rhs) {
  llvm::StringRef trimmed = line.trim();
  if (trimmed.empty() || trimmed.starts_with("#") || trimmed.starts_with("//") ||
      !trimmed.ends_with(";")) {
    return false;
  }

  llvm::StringRef body = trimmed.drop_back().rtrim();
  size_t eqPos = body.find('=');
  if (eqPos == llvm::StringRef::npos) {
    return false;
  }

  llvm::StringRef lhs = body.take_front(eqPos).rtrim();
  rhs = body.drop_front(eqPos + 1).trim();
  if (!isGeneratedValueName(lhs)) {
    return false;
  }
  valueName = lhs;
  return true;
}

static void collectScalarConstantCandidates(
    llvm::ArrayRef<std::string> lines, size_t beginLine, size_t endLine,
    llvm::StringMap<ConstantDeclCandidate> &candidates) {
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
    if (!parseGeneratedValueAssignment(lines[i], assignedName, rhs)) {
      continue;
    }

    auto it = candidates.find(assignedName);
    if (it == candidates.end()) {
      continue;
    }

    ConstantDeclCandidate &info = it->second;
    ++info.assignmentCount;
    info.assignmentLine = i;
    info.assignmentRhs = rhs.str();
  }
}

static void applyScalarConstantRewrites(
    llvm::SmallVectorImpl<std::string> &lines,
    llvm::SmallVectorImpl<bool> &eraseLine,
    llvm::StringMap<ConstantDeclCandidate> &candidates) {
  for (auto &entry : candidates) {
    llvm::StringRef valueName = entry.getKey();
    ConstantDeclCandidate &info = entry.getValue();

    std::string initializer;
    if (info.hasInitializer) {
      if (info.assignmentCount != 0) {
        continue;
      }
      initializer = info.initializer;
    } else {
      if (info.assignmentCount != 1) {
        continue;
      }
      if (!isLiteralInitializer(info.assignmentRhs)) {
        continue;
      }
      initializer = normalizeConstInitializer(
          info.type, llvm::StringRef(info.assignmentRhs));
      eraseLine[info.assignmentLine] = true;
    }

    lines[info.declLine] = (info.indent + "const " + info.type + " " +
                            valueName.str() + " = " + initializer + ";");
  }
}

void rewriteScalarConstantDecls(std::string &cpp) {
  llvm::SmallVector<std::string, 0> lines;
  for (llvm::StringRef ref(cpp); !ref.empty(); ref = ref.split('\n').second) {
    auto split = ref.split('\n');
    lines.push_back(split.first.str());
  }

  llvm::SmallVector<bool, 0> eraseLine(lines.size(), false);

  int braceDepth = 0;
  size_t segmentStart = 0;
  for (size_t i = 0; i < lines.size(); ++i) {
    int depthBefore = braceDepth;
    for (char c : lines[i]) {
      if (c == '{') {
        ++braceDepth;
      } else if (c == '}') {
        --braceDepth;
      }
    }

    if (depthBefore == 0 && braceDepth > 0) {
      segmentStart = i;
    }
    if (depthBefore > 0 && braceDepth == 0) {
      llvm::StringMap<ConstantDeclCandidate> candidates;
      collectScalarConstantCandidates(lines, segmentStart, i, candidates);
      applyScalarConstantRewrites(lines, eraseLine, candidates);
    }
  }

  std::string out;
  out.reserve(cpp.size());
  for (size_t i = 0; i < lines.size(); ++i) {
    if (eraseLine[i]) {
      continue;
    }
    out.append(lines[i]);
    if (i + 1 != lines.size()) {
      out.push_back('\n');
    }
  }
  cpp.swap(out);
}
