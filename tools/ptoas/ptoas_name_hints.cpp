// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===--- ptoas_name_hints.cpp ---------------------------------------------------------===//
// Name-hint and provenance machinery: textual SSA name recovery,
// location hint attach/strip, provenance annotation, and hint-marker
// decoding for C++ emission.
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

using FunctionBlockArgHintMap = llvm::StringMap<
    llvm::SmallVector<llvm::SmallVector<std::string, kNameHintInlineCapacity>,
                      kNameHintInlineCapacity>>;

static SmallVector<std::string, kNameHintInlineCapacity>
getValueNameHints(Value value);

static bool isCppIdentifierStart(char c) {
  return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}

static bool isCppIdentifierChar(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

static std::optional<std::string> getTextualNameFromSMRange(llvm::SMRange range) {
  if (!range.Start.isValid() || !range.End.isValid()) {
    return std::nullopt;
  }
  const char *begin = range.Start.getPointer();
  const char *end = range.End.getPointer();
  if (!begin || !end || end < begin) {
    return std::nullopt;
  }
  llvm::StringRef name(begin, static_cast<size_t>(end - begin));
  if (name.empty()) {
    return std::nullopt;
  }
  name = name.trim();
  if (name.consume_front("%") && name.empty()) {
    return std::nullopt;
  }
  return name.str();
}

static SmallVector<std::string, kNameHintInlineCapacity>
expandTextualResultGroupHints(const AsmParserState::OperationDefinition &opDef,
                              unsigned groupIndex) {
  SmallVector<std::string, kNameHintInlineCapacity> hints;
  if (groupIndex >= opDef.resultGroups.size()) {
    return hints;
  }
  const auto &group = opDef.resultGroups[groupIndex];
  std::optional<std::string> baseName =
      getTextualNameFromSMRange(group.definition.loc);
  if (!baseName) {
    return hints;
  }

  unsigned resultStart = group.startIndex;
  unsigned resultEnd = groupIndex + 1 == opDef.resultGroups.size()
                           ? opDef.op->getNumResults()
                           : opDef.resultGroups[groupIndex + 1].startIndex;
  if (resultStart >= resultEnd) {
    return hints;
  }
  if (resultEnd - resultStart == 1) {
    hints.push_back(*baseName);
    return hints;
  }
  for (unsigned idx = resultStart; idx < resultEnd; ++idx) {
    hints.push_back(*baseName + "#" + std::to_string(idx - resultStart));
  }
  return hints;
}

static std::string sanitizeCppIdentifier(llvm::StringRef name) {
  std::string sanitized;
  sanitized.reserve(name.size() + kCppIdentifierReserveExtra);

  auto appendUnderscore = [&sanitized]() {
    if (sanitized.empty() || sanitized.back() != '_') {
      sanitized.push_back('_');
    }
  };

  for (char c : name) {
    if (isCppIdentifierChar(c)) {
      sanitized.push_back(c);
    }
    else {
      appendUnderscore();
    }
  }

  while (!sanitized.empty() && sanitized.back() == '_') {
    sanitized.pop_back();
  }

  if (sanitized.empty()) {
    return {};
  }
  if (!isCppIdentifierStart(sanitized.front())) {
    sanitized.insert(sanitized.begin(), '_');
  }
  return sanitized;
}

static void appendSanitizedLocationMetadata(
    Attribute metadata, SmallVectorImpl<std::string> &hints) {
  if (auto strAttr = dyn_cast<StringAttr>(metadata)) {
    std::string sanitized = sanitizeCppIdentifier(strAttr.getValue());
    if (!sanitized.empty()) {
      hints.push_back(std::move(sanitized));
    }
    return;
  }
  auto arrayAttr = dyn_cast<ArrayAttr>(metadata);
  if (!arrayAttr) {
    return;
  }
  for (Attribute attr : arrayAttr) {
    auto strAttr = dyn_cast<StringAttr>(attr);
    if (!strAttr) {
      continue;
    }
    std::string sanitized = sanitizeCppIdentifier(strAttr.getValue());
    if (!sanitized.empty()) {
      hints.push_back(std::move(sanitized));
    }
  }
}

static void appendLocationNameHints(Location loc,
                                    SmallVectorImpl<std::string> &hints) {
  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    appendSanitizedLocationMetadata(nameLoc.getName(), hints);
    return;
  }
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    if (Attribute metadata = fusedLoc.getMetadata()) {
      appendSanitizedLocationMetadata(metadata, hints);
    }
    // Only metadata explicitly attached by PTOAS name-hint recovery carries an
    // ordered result-name list. Ordinary fused child locations are debug
    // provenance, not result-indexed name hints.
    return;
  }
  if (auto callSiteLoc = dyn_cast<CallSiteLoc>(loc)) {
    appendLocationNameHints(callSiteLoc.getCallee(), hints);
    if (hints.empty()) {
      appendLocationNameHints(callSiteLoc.getCaller(), hints);
    }
  }
}

static bool hasLocationNameHints(Location loc) {
  SmallVector<std::string, kNameHintInlineCapacity> hints;
  appendLocationNameHints(loc, hints);
  return !hints.empty();
}

// Read the *raw* (unsanitized) source SSA name hints carried in the Location
// metadata. Unlike appendLocationNameHints, this preserves the original textual
// form (e.g. "0", "24", "query_tile") so that issue #337's "pto: %N" provenance
// comments can map a generated C++ variable back to its input .pto SSA name,
// even for pure-digit names that would otherwise be sanitized to "_0".
static void appendRawLocationMetadata(Attribute metadata,
                                      SmallVectorImpl<std::string> &hints) {
  if (auto strAttr = dyn_cast<StringAttr>(metadata)) {
    if (!strAttr.getValue().empty()) {
      hints.push_back(strAttr.getValue().str());
    }
    return;
  }
  auto arrayAttr = dyn_cast<ArrayAttr>(metadata);
  if (!arrayAttr) {
    return;
  }
  for (Attribute attr : arrayAttr) {
    auto strAttr = dyn_cast<StringAttr>(attr);
    if (strAttr && !strAttr.getValue().empty()) {
      hints.push_back(strAttr.getValue().str());
    }
  }
}

static void appendRawLocationProvenance(Location loc,
                                        SmallVectorImpl<std::string> &hints) {
  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    appendRawLocationMetadata(nameLoc.getName(), hints);
    return;
  }
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    if (Attribute metadata = fusedLoc.getMetadata()) {
      appendRawLocationMetadata(metadata, hints);
    }
    // Only metadata explicitly attached by PTOAS name-hint recovery carries an
    // ordered result-name list. Ordinary fused child locations are debug
    // provenance, not result-indexed name hints.
    return;
  }
  if (auto callSiteLoc = dyn_cast<CallSiteLoc>(loc)) {
    appendRawLocationProvenance(callSiteLoc.getCallee(), hints);
    if (hints.empty()) {
      appendRawLocationProvenance(callSiteLoc.getCaller(), hints);
    }
  }
}

// Recover the raw provenance (input .pto SSA name) for an op's results.
// Returns one raw name per result when available, mirroring getResultNameHints
// but without sanitization.
static SmallVector<std::string, kProvenanceInlineCapacity>
getRawResultProvenance(Operation *op) {
  SmallVector<std::string, kProvenanceInlineCapacity> hints;
  if (!op || op->getNumResults() == 0) {
    return hints;
  }
  appendRawLocationProvenance(op->getLoc(), hints);
  if (hints.empty()) {
    return hints;
  }
  hints.erase(std::remove_if(hints.begin(), hints.end(),
                              [](const std::string &name) {
                                return name.empty();
                              }),
              hints.end());
  if (hints.empty()) {
    return hints;
  }
  if (op->getNumResults() == 1) {
    if (hints.size() > 1) {
      hints.resize(1);
    }
    return hints;
  }
  if (hints.size() > op->getNumResults()) {
    hints.resize(op->getNumResults());
  }
  return hints;
}

static SmallVector<std::string, kProvenanceInlineCapacity>
getRawLocationProvenance(Location loc) {
  SmallVector<std::string, kProvenanceInlineCapacity> hints;
  appendRawLocationProvenance(loc, hints);
  hints.erase(std::remove_if(hints.begin(), hints.end(),
                             [](const std::string &hint) {
                               return hint.empty();
                             }),
              hints.end());
  return hints;
}

static Location getIndexedRawProvenanceLoc(Location fallbackLoc, unsigned index) {
  SmallVector<std::string, kProvenanceInlineCapacity> hints =
      getRawLocationProvenance(fallbackLoc);
  if (index >= hints.size()) {
    return fallbackLoc;
  }
  return NameLoc::get(StringAttr::get(fallbackLoc.getContext(), hints[index]),
                      fallbackLoc);
}

static Location attachLocationNameHints(Location baseLoc,
                                        llvm::ArrayRef<std::string> hints,
                                        MLIRContext *context) {
  SmallVector<Attribute, kNameHintInlineCapacity> attrs;
  attrs.reserve(hints.size());
  for (llvm::StringRef hint : hints) {
    if (!hint.empty()) {
      attrs.push_back(StringAttr::get(context, hint));
    }
  }
  if (attrs.empty()) {
    return baseLoc;
  }
  if (attrs.size() == 1) {
    return NameLoc::get(cast<StringAttr>(attrs.front()), baseLoc);
  }
  return FusedLoc::get(ArrayRef<Location>{baseLoc}, ArrayAttr::get(context, attrs),
                       context);
}

static void applyValueNameHints(Value value, llvm::ArrayRef<std::string> hints) {
  if (!value || hints.empty() || hasLocationNameHints(value.getLoc())) {
    return;
  }
  value.setLoc(attachLocationNameHints(value.getLoc(), hints, value.getContext()));
}

static void applyOperationResultNameHints(Operation *op,
                                          llvm::ArrayRef<std::string> hints) {
  if (!op || op->getNumResults() == 0 || hints.empty() ||
      hasLocationNameHints(op->getLoc())) {
    return;
  }

  SmallVector<std::string, kNameHintInlineCapacity> limitedHints;
  limitedHints.reserve(std::min<size_t>(op->getNumResults(), hints.size()));
  for (size_t i = 0, e = std::min<size_t>(op->getNumResults(), hints.size());
       i < e; ++i)
    limitedHints.push_back(hints[i]);
  if (limitedHints.empty()) {
    return;
  }

  op->setLoc(attachLocationNameHints(op->getLoc(), limitedHints, op->getContext()));
}

static void splitDerivedSingleResultProvenanceLocsInRegion(Region &region);

static void splitDerivedSingleResultProvenanceLocsInBlock(Block &block) {
  SmallVector<Operation *, kBranchInlineCapacity> ops;
  ops.reserve(block.getOperations().size());
  for (Operation &op : block) {
    ops.push_back(&op);
  }

  for (size_t i = 0; i < ops.size();) {
    Operation *op = ops[i];
    if (op->getNumResults() != 1) {
      ++i;
      continue;
    }

    SmallVector<std::string, kProvenanceInlineCapacity> hints =
        getRawLocationProvenance(op->getLoc());
    if (hints.size() <= 1) {
      ++i;
      continue;
    }

    size_t runEnd = i + 1;
    while (runEnd < ops.size() && ops[runEnd]->getNumResults() == 1 &&
           ops[runEnd]->getLoc() == op->getLoc()) {
      ++runEnd;
    }

    size_t runSize = runEnd - i;
    if (runSize == hints.size()) {
      Location sharedLoc = op->getLoc();
      for (size_t j = 0; j < runSize; ++j) {
        ops[i + j]->setLoc(getIndexedRawProvenanceLoc(sharedLoc, j));
      }
    }

    i = runEnd;
  }

  for (Operation &op : block) {
    for (Region &region : op.getRegions()) {
      splitDerivedSingleResultProvenanceLocsInRegion(region);
    }
  }
}

static void splitDerivedSingleResultProvenanceLocsInRegion(Region &region) {
  for (Block &block : region) {
    splitDerivedSingleResultProvenanceLocsInBlock(block);
  }
}

void splitDerivedSingleResultProvenanceLocs(Operation *root) {
  if (!root) {
    return;
  }
  for (Region &region : root->getRegions()) {
    splitDerivedSingleResultProvenanceLocsInRegion(region);
  }
}

void narrowUnusedMultiResultProvenanceLocs(Operation *root) {
  if (!root) {
    return;
  }

  root->walk([&](Operation *op) {
    if (op->getNumResults() <= 1) {
      return;
    }

    SmallVector<std::string, kProvenanceInlineCapacity> hints =
        getRawLocationProvenance(op->getLoc());
    if (hints.size() != op->getNumResults()) {
      return;
    }

    SmallVector<std::string, kProvenanceInlineCapacity> liveHints;
    liveHints.reserve(hints.size());
    for (auto [index, result] : llvm::enumerate(op->getResults())) {
      if (!result.use_empty()) {
        liveHints.push_back(hints[index]);
      }
    }

    if (liveHints.empty() || liveHints.size() == hints.size()) {
      return;
    }

    op->setLoc(attachLocationNameHints(op->getLoc(), liveHints,
                                       op->getContext()));
  });
}

static void collectNonEntryBlocksInSourceOrder(
    Operation *op, SmallVectorImpl<Block *> &blocks) {
  for (Region &region : op->getRegions()) {
    bool isEntryBlock = true;
    for (Block &block : region) {
      if (!isEntryBlock && block.getNumArguments() != 0) {
        blocks.push_back(&block);
      }
      isEntryBlock = false;
      for (Operation &nestedOp : block) {
        collectNonEntryBlocksInSourceOrder(&nestedOp, blocks);
      }
    }
  }
}

void mlir::pto::applyTextualNameHintsToModule(ModuleOp module,
                                              const AsmParserState &parserState) {
  if (!module) {
    return;
  }

  for (const AsmParserState::BlockDefinition &blockDef : parserState.getBlockDefs()) {
    if (!blockDef.block) {
      continue;
    }
    for (auto [argIndex, argDef] : llvm::enumerate(blockDef.arguments)) {
      if (argIndex >= blockDef.block->getNumArguments()) {
        break;
      }
      std::optional<std::string> hint = getTextualNameFromSMRange(argDef.loc);
      if (!hint) {
        continue;
      }
      applyValueNameHints(blockDef.block->getArgument(argIndex),
                          llvm::ArrayRef<std::string>{*hint});
    }
  }

  for (const AsmParserState::OperationDefinition &opDef : parserState.getOpDefs()) {
    if (!opDef.op || opDef.op->getNumResults() == 0) {
      continue;
    }

    SmallVector<std::string, kNameHintInlineCapacity> hints;
    hints.reserve(opDef.op->getNumResults());
    for (unsigned groupIndex = 0, e = opDef.resultGroups.size(); groupIndex < e;
         ++groupIndex) {
      SmallVector<std::string, kNameHintInlineCapacity> groupHints =
          expandTextualResultGroupHints(opDef, groupIndex);
      hints.append(groupHints.begin(), groupHints.end());
    }
    if (hints.empty()) {
      continue;
    }
    applyOperationResultNameHints(opDef.op, hints);
  }
}

FunctionBlockArgHintMap collectFunctionBlockArgNameHints(ModuleOp module) {
  FunctionBlockArgHintMap hintsByFunction;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    SmallVector<Block *, kBlockInlineCapacity> nonEntryBlocks;
    collectNonEntryBlocksInSourceOrder(func.getOperation(), nonEntryBlocks);
    if (nonEntryBlocks.empty()) {
      continue;
    }

    SmallVector<SmallVector<std::string, kNameHintInlineCapacity>,
                kNameHintInlineCapacity>
        blockHints;
    blockHints.reserve(nonEntryBlocks.size());
    for (Block *block : nonEntryBlocks) {
      SmallVector<std::string, kNameHintInlineCapacity> argHints;
      bool hasAllHints = block->getNumArguments() != 0;
      for (BlockArgument arg : block->getArguments()) {
        SmallVector<std::string, kNameHintInlineCapacity> hints =
            getValueNameHints(arg);
        if (hints.empty()) {
          hasAllHints = false;
          break;
        }
        argHints.push_back(std::move(hints.front()));
      }
      if (hasAllHints) {
        blockHints.push_back(std::move(argHints));
      }
    }

    if (!blockHints.empty()) {
      hintsByFunction[func.getSymNameAttr()] = std::move(blockHints);
    }
  }
  return hintsByFunction;
}

void applyFunctionBlockArgNameHintsToEmitC(
    ModuleOp module, const FunctionBlockArgHintMap &blockArgHints) {
  for (emitc::FuncOp func : module.getOps<emitc::FuncOp>()) {
    auto it = blockArgHints.find(func.getSymNameAttr());
    if (it == blockArgHints.end() || it->second.empty()) {
      continue;
    }

    SmallVector<Block *, kBlockInlineCapacity> nonEntryBlocks;
    collectNonEntryBlocksInSourceOrder(func.getOperation(), nonEntryBlocks);
    if (nonEntryBlocks.size() != it->second.size()) {
      continue;
    }

    bool shapeMatches = true;
    for (auto [blockIndex, block] : llvm::enumerate(nonEntryBlocks)) {
      if (block->getNumArguments() != it->second[blockIndex].size()) {
        shapeMatches = false;
        break;
      }
    }
    if (!shapeMatches) {
      continue;
    }

    for (auto [blockIndex, block] : llvm::enumerate(nonEntryBlocks)) {
      const auto &argHints = it->second[blockIndex];
      for (auto [argIndex, arg] : llvm::enumerate(block->getArguments())) {
        applyValueNameHints(arg, llvm::ArrayRef<std::string>{argHints[argIndex]});
      }
    }
  }
}

static SmallVector<std::string, kNameHintInlineCapacity>
getValueNameHints(Value value) {
  SmallVector<std::string, kNameHintInlineCapacity> hints;
  if (!value) {
    return hints;
  }
  appendLocationNameHints(value.getLoc(), hints);
  if (hints.size() > 1) {
    hints.resize(1);
  }
  return hints;
}

static std::string buildHintMarker(llvm::StringRef prefix,
                                   llvm::ArrayRef<std::string> hints) {
  auto encodeHintMarkerToken = [](llvm::StringRef token) {
    auto hexDigit = [](unsigned value) -> char {
      return value < 10 ? static_cast<char>('0' + value)
                        : static_cast<char>('A' + (value - 10));
    };

    auto isSafeMarkerChar = [](unsigned char c) {
      return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
             (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-';
    };

    std::string encoded;
    encoded.reserve(token.size());
    for (unsigned char c : token.bytes()) {
      if (isSafeMarkerChar(c)) {
        encoded.push_back(static_cast<char>(c));
        continue;
      }
      encoded.push_back('%');
      encoded.push_back(
          hexDigit((c >> kHexNibbleBitWidth) & kHexNibbleMask));
      encoded.push_back(hexDigit(c & kHexNibbleMask));
    }
    return encoded;
  };

  std::string marker = ("/* " + prefix + ":").str();
  for (size_t i = 0; i < hints.size(); ++i) {
    if (i != 0) {
      marker.push_back(',');
    }
    marker.append(encodeHintMarkerToken(hints[i]));
  }
  marker.append(" */\n");
  return marker;
}

static SmallVector<std::string, kProvenanceInlineCapacity>
collectExpressionProvenance(emitc::ExpressionOp expr) {
  SmallVector<std::string, kProvenanceInlineCapacity> provenance;
  auto appendUnique = [&](llvm::ArrayRef<std::string> names) {
    for (const std::string &name : names) {
      if (name.empty()) {
        continue;
      }
      if (std::find(provenance.begin(), provenance.end(), name) !=
          provenance.end()) {
        continue;
      }
      provenance.push_back(name);
    }
  };

  expr.walk<WalkOrder::PreOrder>([&](Operation *nested) {
    if (nested == expr.getOperation()) {
      return WalkResult::advance();
    }
    if (nested->getNumResults() == 0 || isa<emitc::VerbatimOp>(nested)) {
      return WalkResult::advance();
    }
    appendUnique(getRawResultProvenance(nested));
    return WalkResult::advance();
  });
  appendUnique(getRawResultProvenance(expr.getOperation()));
  return provenance;
}

void annotateEmitCProvenanceHints(ModuleOp module) {
  struct ProvenanceMarker {
    Operation *op = nullptr;
    SmallVector<std::string, kProvenanceInlineCapacity> names;
  };

  llvm::SmallVector<ProvenanceMarker, kProvenanceMarkerInlineCapacity>
      opsToAnnotate;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op->getNumResults() == 0 || isa<emitc::VerbatimOp>(op)) {
      return WalkResult::advance();
    }

    if (auto expr = dyn_cast<emitc::ExpressionOp>(op)) {
      SmallVector<std::string, kProvenanceInlineCapacity> provenance =
          collectExpressionProvenance(expr);
      if (provenance.empty()) {
        return WalkResult::skip();
      }
      opsToAnnotate.push_back(
          ProvenanceMarker{
              op, SmallVector<std::string, kProvenanceInlineCapacity>(provenance)});
      return WalkResult::skip();
    }

    if (op->getParentOfType<emitc::ExpressionOp>()) {
      return WalkResult::advance();
    }
    // Only carry raw provenance into the C++ post-pass. Semantic renaming is
    // intentionally deferred until naming can happen inside the emitter's own
    // symbol table instead of via post-hoc C++ text rewriting.
    SmallVector<std::string, kProvenanceInlineCapacity> provenance =
        getRawResultProvenance(op);
    if (provenance.empty()) {
      return WalkResult::advance();
    }
    opsToAnnotate.push_back(ProvenanceMarker{
        op, SmallVector<std::string, kProvenanceInlineCapacity>(
                provenance.begin(), provenance.end())});
    return WalkResult::advance();
  });

  OpBuilder builder(module.getContext());
  for (const ProvenanceMarker &marker : opsToAnnotate) {
    // Emit a provenance marker carrying the raw input SSA name. This is
    // consumed by the C++ post-processor to emit `// pto: %N` comments so a
    // reader can map a generated variable back to its .pto source (issue #337
    // point 1: locatability without strict number alignment).
    if (!marker.names.empty()) {
      builder.setInsertionPoint(marker.op);
      builder.create<emitc::VerbatimOp>(
          marker.op->getLoc(),
          builder.getStringAttr(
              buildHintMarker("PTOAS_PROVENANCE", marker.names)));
    }
  }
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

static int decodeNameHintHexDigit(char c) {
  if (c >= '0' && c <= '9') {
    return c - '0';
  }
  if (c >= 'a' && c <= 'f') {
    return c - 'a' + 10;
  }
  if (c >= 'A' && c <= 'F') {
    return c - 'A' + 10;
  }
  return -1;
}

static std::string decodeNameHintMarkerToken(llvm::StringRef token) {
  std::string decoded;
  decoded.reserve(token.size());
  for (size_t i = 0; i < token.size();) {
    if (token[i] == '%' && i + 2 < token.size()) {
      int hi = decodeNameHintHexDigit(token[i + 1]);
      int lo = decodeNameHintHexDigit(token[i + 2]);
      if (hi >= 0 && lo >= 0) {
        decoded.push_back(static_cast<char>(
            (static_cast<unsigned>(hi) << kHexNibbleBitWidth) | lo));
        i += 3;
        continue;
      }
    }
    decoded.push_back(token[i]);
    ++i;
  }
  return decoded;
}

static std::optional<llvm::SmallVector<std::string, kNameHintInlineCapacity>>
parseNameHintMarker(llvm::StringRef markerBody) {

  llvm::SmallVector<std::string, kNameHintInlineCapacity> hints;
  markerBody = markerBody.trim();
  if (markerBody.empty()) {
    return std::nullopt;
  }

  size_t start = 0;
  while (start <= markerBody.size()) {
    size_t comma = markerBody.find(',', start);
    llvm::StringRef token = markerBody.slice(
        start, comma == llvm::StringRef::npos ? markerBody.size() : comma);
    token = token.trim();
    if (!token.empty()) {
      hints.push_back(decodeNameHintMarkerToken(token));
    }
    if (comma == llvm::StringRef::npos) {
      break;
    }
    start = comma + 1;
  }

  if (hints.empty()) {
    return std::nullopt;
  }
  return hints;
}

static void stripHintMarkersWithPrefix(std::string &cpp,
                                       llvm::StringRef markerPrefix) {
  std::string out;
  out.reserve(cpp.size());
  size_t searchPos = 0;
  while (searchPos < cpp.size()) {
    size_t markerPos = cpp.find(markerPrefix.str(), searchPos);
    if (markerPos == std::string::npos) {
      out.append(cpp, searchPos, std::string::npos);
      break;
    }

    out.append(cpp, searchPos, markerPos - searchPos);
    size_t markerEnd = cpp.find("*/", markerPos + markerPrefix.size());
    if (markerEnd == std::string::npos) {
      out.append(cpp, markerPos, std::string::npos);
      break;
    }
    markerEnd += kCommentTerminatorLength;
    while (markerEnd < cpp.size() &&
           (cpp[markerEnd] == '\r' || cpp[markerEnd] == '\n')) {
      ++markerEnd;
    }
    searchPos = markerEnd;
  }
  cpp.swap(out);
}

static void stripAllHintMarkers(std::string &cpp) {
  stripHintMarkersWithPrefix(cpp, "/* PTOAS_PROVENANCE:");
}

static std::string sanitizeCommentText(llvm::StringRef text) {
  auto hexDigit = [](unsigned value) -> char {
    return value < 10 ? static_cast<char>('0' + value)
                      : static_cast<char>('A' + (value - 10));
  };

  std::string sanitized;
  sanitized.reserve(text.size());
  for (unsigned char c : text.bytes()) {
    switch (c) {
    case '\n':
      sanitized.append("\\n");
      break;
    case '\r':
      sanitized.append("\\r");
      break;
    case '\t':
      sanitized.append("\\t");
      break;
    default:
      if (std::iscntrl(c)) {
        sanitized.push_back('\\');
        sanitized.push_back('x');
        sanitized.push_back(
            hexDigit((c >> kHexNibbleBitWidth) & kHexNibbleMask));
        sanitized.push_back(hexDigit(c & kHexNibbleMask));
      } else {
        sanitized.push_back(static_cast<char>(c));
      }
      break;
    }
  }
  return sanitized;
}

// Convert `/* PTOAS_PROVENANCE:rawname,... */` markers into standalone
// `// pto: %rawname` comment lines in-place. This avoids guessing which later
// generated declaration a marker should attach to after EmitC/Cpp emission,
// hoisting, or inlining. The marker is consumed (removed) here.
static constexpr llvm::StringLiteral kProvenanceMarkerPrefix =
    "/* PTOAS_PROVENANCE:";

static size_t appendProvenanceMarkerComment(llvm::StringRef segment,
                                            size_t markerPos, size_t markerEnd,
                                            std::string &out) {
  auto names = parseNameHintMarker(
      segment.slice(markerPos + kProvenanceMarkerPrefix.size(), markerEnd));
  if (names && !names->empty()) {
    out.append("// pto: ");
    for (size_t idx = 0; idx < names->size(); ++idx) {
      if (idx != 0) {
        out.append(", ");
      }
      out.push_back('%');
      out.append(sanitizeCommentText((*names)[idx]));
    }
    out.push_back('\n');
  }

  size_t next = markerEnd + kCommentTerminatorLength;
  while (next < segment.size()) {
    const char current = segment[next];
    if (current != '\r' && current != '\n') {
      break;
    }
    ++next;
  }
  return next;
}

static void emitProvenanceComments(std::string &segment) {
  std::string out;
  out.reserve(segment.size() + kRawStringInlineCapacity);
  size_t i = 0;
  while (i < segment.size()) {
    size_t mp = segment.find(kProvenanceMarkerPrefix.str(), i);
    if (mp == std::string::npos) {
      out.append(segment, i, std::string::npos);
      break;
    }
    out.append(segment, i, mp - i);
    size_t me = segment.find("*/", mp + kProvenanceMarkerPrefix.size());
    if (me == std::string::npos) {
      out.append(segment, i, std::string::npos);
      break;
    }
    i = appendProvenanceMarkerComment(segment, mp, me, out);
  }
  segment.swap(out);
}

void rewriteNameHintMarkers(std::string &cpp) {
  emitProvenanceComments(cpp);
  stripAllHintMarkers(cpp);
}
