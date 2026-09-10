// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/CppPostprocess.h"

#include "PTO/Support/CodeConstants.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"

#include <cctype>
#include <optional>
#include <string>

namespace mlir {
namespace pto {

namespace {

constexpr size_t kLastUseReplacementReserve = 32;

struct ParsedMarkerCall {
  size_t markerPos;
  size_t rparenPos;
  llvm::SmallVector<llvm::StringRef, mlir::pto::kValue8> args;
};

static bool parseMarkerArgs(llvm::StringRef argsRef,
                            llvm::SmallVectorImpl<llvm::StringRef> &args) {
  args.clear();
  if (argsRef.empty()) {
    return true;
  }

  int parenDepth = 0;
  size_t partBegin = 0;
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

static bool parseLastUseMarkerName(llvm::StringRef markerName,
                                   std::string &callee,
                                   std::string &lastUseArgs) {
  static constexpr llvm::StringLiteral kPrefix = "PTOAS__LAST_USE__";
  if (!markerName.starts_with(kPrefix)) {
    return false;
  }

  llvm::StringRef payload = markerName.drop_front(kPrefix.size());
  size_t split = payload.find("__");
  if (split == llvm::StringRef::npos) {
    return false;
  }

  callee = payload.take_front(split).str();
  llvm::StringRef encoded = payload.drop_front(split + 2);
  if (callee.empty() || encoded.empty()) {
    return false;
  }

  lastUseArgs.clear();
  size_t pos = 0;
  while (pos < encoded.size()) {
    size_t next = encoded.find("__", pos);
    llvm::StringRef token =
        next == llvm::StringRef::npos ? encoded.drop_front(pos)
                                      : encoded.slice(pos, next);
    if (token.empty()) {
      return false;
    }
    if (!llvm::all_of(token, [](char c) { return std::isdigit(c); })) {
      return false;
    }
    if (!lastUseArgs.empty()) {
      lastUseArgs.append(", ");
    }
    lastUseArgs.append(token.str());
    if (next == llvm::StringRef::npos) {
      break;
    }
    pos = next + mlir::pto::kValue2;
  }
  return !lastUseArgs.empty();
}

static size_t findMarkerLparen(const std::string &cpp, size_t searchFrom) {
  size_t lparenPos = searchFrom;
  while (lparenPos < cpp.size() && cpp[lparenPos] != '(') {
    ++lparenPos;
  }
  return lparenPos;
}

static size_t findMatchingRparen(const std::string &cpp, size_t argsBegin) {
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
      return i;
    }
    --parenDepth;
  }
  return std::string::npos;
}

static std::string
buildLastUseReplacement(const std::string &callee, const std::string &lastUseArgs,
                        const llvm::SmallVectorImpl<llvm::StringRef> &args,
                        size_t argsRefSize) {
  std::string replacement;
  replacement.reserve(callee.size() + lastUseArgs.size() + argsRefSize +
                      kLastUseReplacementReserve);
  replacement.append("[[pto::last_use(");
  replacement.append(lastUseArgs);
  replacement.append(")]] ");
  replacement.append(callee);
  replacement.push_back('(');
  for (size_t i = 0; i < args.size(); ++i) {
    if (i != 0) {
      replacement.append(", ");
    }
    replacement.append(args[i].str());
  }
  replacement.push_back(')');
  return replacement;
}

} // namespace

bool rewriteLastUseMarkersInCpp(std::string &cpp) {
  size_t searchPos = 0;
  bool changed = false;
  static constexpr llvm::StringLiteral kPrefix = "PTOAS__LAST_USE__";
  while (true) {
    size_t markerPos = cpp.find(kPrefix.str(), searchPos);
    if (markerPos == std::string::npos) {
      break;
    }

    size_t lparenPos = findMarkerLparen(cpp, markerPos + kPrefix.size());
    if (lparenPos >= cpp.size()) {
      searchPos = markerPos + 1;
      continue;
    }

    size_t argsBegin = lparenPos + 1;
    size_t rparenPos = findMatchingRparen(cpp, argsBegin);
    if (rparenPos == std::string::npos) {
      searchPos = markerPos + 1;
      continue;
    }

    llvm::StringRef argsRef(cpp.data() + argsBegin, rparenPos - argsBegin);
    ParsedMarkerCall call{markerPos, rparenPos, {}};
    if (!parseMarkerArgs(argsRef, call.args)) {
      searchPos = rparenPos + 1;
      continue;
    }

    llvm::StringRef markerName(cpp.data() + markerPos, lparenPos - markerPos);
    std::string callee;
    std::string lastUseArgs;
    if (!parseLastUseMarkerName(markerName, callee, lastUseArgs)) {
      searchPos = rparenPos + 1;
      continue;
    }

    std::string replacement = buildLastUseReplacement(callee, lastUseArgs,
                                                      call.args, argsRef.size());

    cpp.replace(markerPos, (rparenPos - markerPos) + 1, replacement);
    changed = true;
    searchPos = markerPos + replacement.size();
  }
  return changed;
}

} // namespace pto
} // namespace mlir
