// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "ObjectEmission.h"
#include "PTO/IR/PTO.h"
#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/Passes.h"
#include "VPTOHostStubEmission.h"
#include "ptoas.h"
#include "ptobc/ptobc_decode.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"


using namespace mlir;
using mlir::pto::PTOASContext;

#ifndef PTOAS_RELEASE_VERSION
#define PTOAS_RELEASE_VERSION "unknown"
#endif

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static void printPTOASVersion(llvm::raw_ostream &os) {
  os << "ptoas " << PTOAS_RELEASE_VERSION << "\n";
}

static bool parseRequestedOutputCANNVersion(
    llvm::StringRef versionText, std::optional<mlir::pto::CANNVersion> &version,
    llvm::raw_ostream &diagOS) {
  version.reset();
  if (versionText.empty()) {
    return true;
  }
  std::optional<mlir::pto::CANNVersion> parsed =
      mlir::pto::parseCANNVersion(versionText);
  if (!parsed) {
    diagOS << "Error: invalid --cann-output-version='" << versionText
           << "'. Expected forms like '9.0.0' or '9.0.0-beta.1'.\n";
    return false;
  }
  version = *parsed;
  return true;
}

static mlir::pto::CANNVersion selectEffectiveOutputCANNVersion(
    const mlir::pto::CANNVersion &detectedVersion,
    const std::optional<mlir::pto::CANNVersion> &overrideVersion) {
  return overrideVersion.value_or(detectedVersion);
}

static bool hasCLIOption(int argc, char **argv, llvm::StringRef option) {
  const std::string optionWithValue = (option + "=").str();
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    if (arg == option || arg.starts_with(optionWithValue)) {
      return true;
    }
  }
  return false;
}

static std::string normalizePTOASArch(llvm::StringRef archValue) {
  std::string normalized = archValue.str();
  for (char &c : normalized) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return normalized;
}

static bool isSupportedPTOASArch(llvm::StringRef archValue) {
  return archValue == "a2" || archValue == "a3" || archValue == "a5";
}

constexpr size_t kArchRegexCaptureGroupCount = 3;
constexpr size_t kPTOBCMagicSize = 6;

static std::optional<std::string>
detectPTOASTextualModuleArch(llvm::StringRef text) {
  llvm::SmallVector<llvm::StringRef, mlir::pto::kValue4> matches;
  llvm::Regex archRegex(
      R"ptoarch("?(pto\.target_arch)"?[[:space:]]*=[[:space:]]*"([[:alpha:][:digit:]_]+)")ptoarch");
  if (!archRegex.match(text, &matches) ||
      matches.size() < kArchRegexCaptureGroupCount) {
    return std::nullopt;
  }
  return normalizePTOASArch(matches[mlir::pto::kValue2]);
}

static bool isPTOBCBuffer(llvm::StringRef buffer) {
  return buffer.size() >= kPTOBCMagicSize &&
         std::memcmp(buffer.data(), "PTOBC\0", kPTOBCMagicSize) == 0;
}

static std::unique_ptr<llvm::MemoryBuffer> readInputBuffer() {
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!fileOrErr) {
    llvm::errs() << "Error: Could not open input file: "
                 << fileOrErr.getError().message() << "\n";
    return nullptr;
  }
  return std::move(*fileOrErr);
}

static bool resolveTextInputArch(llvm::StringRef buffer, bool cliArchSpecified,
                                 std::string &arch) {
  arch = normalizePTOASArch(mlir::pto::ptoTargetArch);
  if (cliArchSpecified) {
    if (!isSupportedPTOASArch(arch)) {
      llvm::errs() << "Error: invalid --pto-arch='" << mlir::pto::ptoTargetArch
                   << "'. Expected 'a2', 'a3', or 'a5'.\n";
      return false;
    }
    return true;
  }

  if (auto detectedArch = detectPTOASTextualModuleArch(buffer)) {
    arch = *detectedArch;
  }
  if (!isSupportedPTOASArch(arch)) {
    arch = "a3";
  }
  return true;
}

static OwningOpRef<ModuleOp> decodePTOBCModule(llvm::StringRef buffer,
                                               MLIRContext &context) {
  llvm::ArrayRef<uint8_t> bytes(reinterpret_cast<const uint8_t *>(buffer.data()),
                                buffer.size());
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
  try {
    return ptobc::decodePTOBCToModule(bytes, context);
  } catch (...) {
    llvm::errs() << "Error: Failed to decode PTOBC.\n";
    return {};
  }
#else
  OwningOpRef<ModuleOp> module = ptobc::decodePTOBCToModule(bytes, context);
  if (!module) {
    llvm::errs() << "Error: Failed to decode PTOBC.\n";
  }
  return module;
#endif
}

static OwningOpRef<ModuleOp>
parseTextualModule(std::unique_ptr<llvm::MemoryBuffer> inputBuffer,
                   MLIRContext &context, llvm::StringRef arch) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(inputBuffer), llvm::SMLoc());
  mlir::pto::ScopedPTOParserTargetArch scopedParserArch(
      &context, arch == "a5" ? mlir::pto::PTOParserTargetArch::A5
                             : mlir::pto::PTOParserTargetArch::A3);
  ParserConfig parserConfig(&context);
  Block parsedBlock;
  LocationAttr sourceFileLoc = UnknownLoc::get(&context);
  if (const auto *sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())) {
    sourceFileLoc = FileLineColLoc::get(&context, sourceBuf->getBufferIdentifier(),
                                        /*line=*/0, /*column=*/0);
  }
  AsmParserState parserState;
  if (failed(parseAsmSourceFile(sourceMgr, &parsedBlock, parserConfig,
                                &parserState))) {
    llvm::errs() << "Error: Failed to parse MLIR.\n";
    return OwningOpRef<ModuleOp>();
  }
  // `parseSourceFile<ModuleOp>` internally uses the same helper to wrap the
  // parsed top-level block. We spell it out here because the public wrapper
  // does not expose `AsmParserState`, which we need for textual SSA-name
  // recovery.
  OwningOpRef<ModuleOp> module =
      mlir::detail::constructContainerOpForParserIfNecessary<ModuleOp>(
          &parsedBlock, &context, sourceFileLoc);
  if (!module) {
    llvm::errs() << "Error: Failed to build parsed module.\n";
    return module;
  }
  mlir::pto::applyTextualNameHintsToModule(*module, parserState);
  return module;
}

static OwningOpRef<ModuleOp>
loadInputModule(std::unique_ptr<llvm::MemoryBuffer> inputBuffer,
                MLIRContext &context, bool cliArchSpecified,
                std::string &arch) {
  llvm::StringRef buffer = inputBuffer->getBuffer();

  OwningOpRef<ModuleOp> module;
  if (isPTOBCBuffer(buffer)) {
    arch = normalizePTOASArch(mlir::pto::ptoTargetArch);
    if (cliArchSpecified && !isSupportedPTOASArch(arch)) {
      llvm::errs() << "Error: invalid --pto-arch='" << mlir::pto::ptoTargetArch
                   << "'. Expected 'a2', 'a3', or 'a5'.\n";
      return {};
    }
    module = decodePTOBCModule(buffer, context);
  } else {
    if (!resolveTextInputArch(buffer, cliArchSpecified, arch)) {
      return {};
    }
    module = parseTextualModule(std::move(inputBuffer), context, arch);
  }
  if (!module) {
    return {};
  }

  Operation *moduleOp = module.get().getOperation();
  if (cliArchSpecified) {
    moduleOp->setAttr("pto.target_arch",
                      mlir::StringAttr::get(moduleOp->getContext(), arch));
  } else if (auto archAttr = moduleOp->getAttrOfType<StringAttr>("pto.target_arch")) {
    std::string moduleArch = normalizePTOASArch(archAttr.getValue());
    if (isSupportedPTOASArch(moduleArch)) {
      arch = std::move(moduleArch);
    } else {
      if (!isSupportedPTOASArch(arch)) {
        arch = "a3";
      }
      moduleOp->setAttr("pto.target_arch",
                        mlir::StringAttr::get(moduleOp->getContext(), arch));
    }
  } else {
    if (!isSupportedPTOASArch(arch)) {
      arch = "a3";
    }
    moduleOp->setAttr("pto.target_arch",
                      mlir::StringAttr::get(moduleOp->getContext(), arch));
  }

  if (failed(mlir::verify(*module))) {
    llvm::errs() << "Error: input module verification failed.\n";
    return {};
  }
  return module;
}

static bool parseDriverBackend(llvm::StringRef backendStr,
                               mlir::pto::PTOBackend &out) {
  std::string s = backendStr.str();
  for (char &c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  if (s == "emitc") {
    out = mlir::pto::PTOBackend::EmitC;
    return true;
  }
  if (s == "vpto") {
    out = mlir::pto::PTOBackend::VPTO;
    return true;
  }
  return false;
}

static LogicalResult
parseDriverBackendAttr(Operation *op,
                       std::optional<mlir::pto::PTOBackend> &backend) {
  backend = std::nullopt;
  Attribute rawBackendAttr = op->getAttr("pto.backend");
  if (!rawBackendAttr) {
    return success();
  }

  auto backendAttr = dyn_cast<StringAttr>(rawBackendAttr);
  if (!backendAttr) {
    return op->emitError("invalid pto.backend attribute. Expected string "
                         "value 'emitc' or 'vpto'.");
  }

  mlir::pto::PTOBackend attrBackend = mlir::pto::PTOBackend::EmitC;
  if (!parseDriverBackend(backendAttr.getValue(), attrBackend)) {
    return op->emitError("invalid pto.backend '")
           << backendAttr.getValue() << "'. Expected 'emitc' or 'vpto'.";
  }

  backend = attrBackend;
  return success();
}

static bool isBackendPartitionedContainer(ModuleOp module) {
  Block *body = module.getBody();
  if (!body) {
    return false;
  }
  return llvm::all_of(body->getOperations(),
                      [](Operation &op) { return isa<ModuleOp>(op); });
}

static bool isUserVisibleIROutputRequested() {
  return mlir::pto::emitMlirIR || mlir::pto::emitVPTO ||
         mlir::pto::emitVPTOLLVMDialect || mlir::pto::ptoPrintSeamIR ||
         !mlir::pto::ptoSeamIRFile.empty();
}

static SmallVector<StringRef> collectImportedPeerNames(ModuleOp module) {
  SmallVector<StringRef> names;
  module.walk([&](pto::ImportReservedBufferOp importOp) {
    names.push_back(importOp.getPeerFuncAttr().getValue());
  });
  llvm::sort(names);
  names.erase(std::unique(names.begin(), names.end()), names.end());
  return names;
}

static SmallVector<StringRef> collectDirectCalleeNames(ModuleOp module) {
  SmallVector<StringRef> names;
  module.walk([&](func::CallOp callOp) {
    names.push_back(callOp.getCalleeAttr().getLeafReference());
  });
  llvm::sort(names);
  names.erase(std::unique(names.begin(), names.end()), names.end());
  return names;
}

static SmallVector<StringRef> collectDirectCalleeNames(func::FuncOp funcOp) {
  SmallVector<StringRef> names;
  if (!funcOp || funcOp.isDeclaration()) {
    return names;
  }
  funcOp.walk([&](func::CallOp callOp) {
    if (callOp->getParentOfType<func::FuncOp>() != funcOp) {
      return;
    }
    names.push_back(callOp.getCalleeAttr().getLeafReference());
  });
  llvm::sort(names);
  names.erase(std::unique(names.begin(), names.end()), names.end());
  return names;
}

static void copyModuleAttrsToJobModule(ModuleOp source, ModuleOp jobModule) {
  for (NamedAttribute attr : source->getAttrs()) {
    StringRef attrName = attr.getName().getValue();
    if (attrName == SymbolTable::getSymbolAttrName() ||
        attrName == "pto.backend") {
      continue;
    }
    jobModule->setAttr(attr.getName(), attr.getValue());
  }
}

static func::FuncOp findFunctionByLogicalName(ModuleOp module,
                                              StringRef logicalName) {
  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    if (funcOp.getSymName() == logicalName) {
      return funcOp;
    }
  }
  return {};
}

static func::FuncOp findFunctionBySymbolName(ModuleOp module,
                                             StringRef symbolName) {
  return dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupSymbolIn(module, symbolName));
}

static func::FuncOp findFunctionForPeerReference(ModuleOp module,
                                                 StringRef peerRef) {
  if (func::FuncOp exact = findFunctionBySymbolName(module, peerRef)) {
    return exact;
  }

  func::FuncOp privateMatch;
  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    if (mlir::pto::getPTODSLLogicalNameOrSymbolName(funcOp) != peerRef) {
      continue;
    }

    auto visibility = funcOp->getAttrOfType<StringAttr>("sym_visibility");
    if (!visibility || visibility.getValue() != "private") {
      return funcOp;
    }
    if (!privateMatch) {
      privateMatch = funcOp;
    }
  }
  return privateMatch;
}

static FailureOr<func::FuncOp>
findSiblingSourceFunction(ModuleOp outer, ModuleOp targetChild,
                          StringRef symbolName, bool allowLogicalNameMatch,
                          StringRef referenceKind) {
  SmallVector<func::FuncOp> exactMatches;
  SmallVector<func::FuncOp> logicalMatches;
  for (ModuleOp child : outer.getOps<ModuleOp>()) {
    if (child == targetChild) {
      continue;
    }
    for (func::FuncOp funcOp : child.getOps<func::FuncOp>()) {
      auto visibility = funcOp->getAttrOfType<StringAttr>("sym_visibility");
      if (visibility && visibility.getValue() == "private") {
        continue;
      }
      if (funcOp.getSymName() == symbolName) {
        exactMatches.push_back(funcOp);
        continue;
      }
      if (allowLogicalNameMatch &&
          mlir::pto::getPTODSLLogicalNameOrSymbolName(funcOp) == symbolName) {
        logicalMatches.push_back(funcOp);
      }
    }
  }

  if (exactMatches.size() > 1) {
    targetChild.emitError("mixed-backend child assembly does not yet support ambiguous cross-child ")
        << referenceKind << " '@" << symbolName
        << "'; found multiple sibling public func.func definitions";
    return failure();
  }
  if (!exactMatches.empty()) {
    return exactMatches.front();
  }

  if (logicalMatches.size() > 1) {
    targetChild.emitError("mixed-backend child assembly does not yet support ambiguous cross-child logical ")
        << referenceKind << " '@" << symbolName
        << "'; found multiple sibling public func.func definitions";
    return failure();
  }
  if (!logicalMatches.empty()) {
    return logicalMatches.front();
  }
  return func::FuncOp();
}

static LogicalResult verifyImportedPeerCloneContract(func::FuncOp peerSource,
                                                     StringRef logicalName) {
  SmallVector<StringRef> directCalleeNames = collectDirectCalleeNames(peerSource);
  if (directCalleeNames.empty()) {
    return success();
  }

  peerSource.emitError(
      "mixed-backend child assembly does not yet support transitive cross-child function closure for imported peer '")
      << logicalName << "'; cloned peer body @" << peerSource.getSymName()
      << " directly calls @" << directCalleeNames.front()
      << ". Keep peer_func targets leaf-only for now.";
  return failure();
}

static func::FuncOp cloneFunctionIntoModule(ModuleOp jobModule,
                                            func::FuncOp sourceFunc,
                                            StringRef newName,
                                            StringRef visibility) {
  OpBuilder builder(jobModule.getContext());
  builder.setInsertionPointToEnd(&jobModule.getBodyRegion().front());
  auto cloned = cast<func::FuncOp>(sourceFunc->clone());
  cloned.setSymName(newName);
  cloned->setAttr("sym_visibility", StringAttr::get(jobModule.getContext(),
                                                     visibility));
  mlir::pto::setExternalArtifactVisibility(cloned, visibility != "private" &&
                                                       mlir::pto::hasExternalArtifactVisibility(sourceFunc));
  builder.insert(cloned);
  return cloned;
}

static func::FuncOp cloneFunctionDeclarationIntoModule(ModuleOp jobModule,
                                                       func::FuncOp sourceFunc,
                                                       StringRef newName,
                                                       StringRef visibility) {
  func::FuncOp cloned =
      cloneFunctionIntoModule(jobModule, sourceFunc, newName, visibility);
  while (!cloned.getBody().empty()) {
    cloned.getBody().front().erase();
  }
  return cloned;
}

static void rewriteExportedFunctionToLogicalWrapper(func::FuncOp exportedFunc,
                                                    StringRef logicalName) {
  if (logicalName == exportedFunc.getSymName()) {
    return;
  }

  while (!exportedFunc.getBody().empty()) {
    exportedFunc.getBody().front().erase();
  }
  Block *entry = exportedFunc.addEntryBlock();
  OpBuilder builder(entry, entry->begin());

  auto call = builder.create<func::CallOp>(
      exportedFunc.getLoc(), logicalName, exportedFunc.getResultTypes(),
      entry->getArguments());
  builder.create<func::ReturnOp>(exportedFunc.getLoc(), call.getResults());
}

static LogicalResult
verifyInChildLogicalWrapperAmbiguity(ModuleOp targetChild,
                                     ArrayRef<func::FuncOp> exportedFuncs) {
  llvm::SmallDenseMap<StringRef, SmallVector<func::FuncOp, mlir::pto::kValue2>> grouped;
  for (func::FuncOp exportedFunc : exportedFuncs) {
    auto kernelKindAttr =
        exportedFunc->getAttrOfType<mlir::pto::FunctionKernelKindAttr>(
            mlir::pto::FunctionKernelKindAttr::name);
    if (kernelKindAttr) {
      continue;
    }
    StringRef logicalName =
        mlir::pto::getPTODSLLogicalNameOrSymbolName(exportedFunc);
    grouped[logicalName].push_back(exportedFunc);
  }

  for (const auto &entry : grouped) {
    if (entry.second.size() <= 1) {
      continue;
    }
    targetChild.emitError(
        "mixed-backend child assembly does not yet support ambiguous in-child logical reference '@")
        << entry.first
        << "'; found multiple ABI-specialized public func.func definitions";
    return failure();
  }
  return success();
}

static FailureOr<OwningOpRef<ModuleOp>>
buildBackendChildCompileUnit(ModuleOp outer, ModuleOp targetChild) {
  ModuleOp jobModule = ModuleOp::create(outer.getLoc());
  copyModuleAttrsToJobModule(outer, jobModule);
  copyModuleAttrsToJobModule(targetChild, jobModule);

  for (Operation &op : targetChild.getBodyRegion().front().getOperations()) {
    jobModule.push_back(op.clone());
  }

  SmallVector<StringRef> directCalleeNames = collectDirectCalleeNames(targetChild);
  for (StringRef calleeName : directCalleeNames) {
    if (findFunctionByLogicalName(jobModule, calleeName)) {
      continue;
    }
    FailureOr<func::FuncOp> siblingSourceOr =
        findSiblingSourceFunction(outer, targetChild, calleeName,
                                  /*allowLogicalNameMatch=*/false,
                                  /*referenceKind=*/"function reference");
    if (failed(siblingSourceOr)) {
      return failure();
    }
    func::FuncOp siblingSource = *siblingSourceOr;
    if (!siblingSource) {
      targetChild.emitError(
          "mixed-backend child assembly does not yet support unresolved cross-child function reference '@")
          << calleeName
          << "'; each cross-child func.call must resolve to one sibling public func.func";
      return failure();
    }
    cloneFunctionDeclarationIntoModule(jobModule, siblingSource, calleeName,
                                       "private");
  }

  SmallVector<StringRef> importedPeerNames = collectImportedPeerNames(targetChild);
  SmallVector<func::FuncOp> exportedFuncs;
  for (func::FuncOp funcOp : jobModule.getOps<func::FuncOp>()) {
    auto visibility = funcOp->getAttrOfType<StringAttr>("sym_visibility");
    if (visibility && visibility.getValue() == "private") {
      continue;
    }
    if (funcOp.isExternal()) {
      continue;
    }
    exportedFuncs.push_back(funcOp);
  }
  if (failed(verifyInChildLogicalWrapperAmbiguity(targetChild, exportedFuncs))) {
    return failure();
  }
  for (func::FuncOp exportedFunc : exportedFuncs) {
    StringRef logicalName =
        mlir::pto::getPTODSLLogicalNameOrSymbolName(exportedFunc);
    auto kernelKindAttr =
        exportedFunc->getAttrOfType<mlir::pto::FunctionKernelKindAttr>(
            mlir::pto::FunctionKernelKindAttr::name);
    if (kernelKindAttr) {
      continue;
    }
    if (!findFunctionByLogicalName(jobModule, logicalName)) {
      cloneFunctionIntoModule(jobModule, exportedFunc, logicalName, "private");
    }
    rewriteExportedFunctionToLogicalWrapper(exportedFunc, logicalName);
  }

  for (StringRef logicalName : importedPeerNames) {
    FailureOr<func::FuncOp> peerSourceOr =
        findSiblingSourceFunction(outer, targetChild, logicalName,
                                  /*allowLogicalNameMatch=*/true,
                                  /*referenceKind=*/"peer_func reference");
    if (failed(peerSourceOr)) {
      return failure();
    }
    func::FuncOp peerSource = *peerSourceOr;
    if (!peerSource) {
      targetChild.emitError(
          "mixed-backend child assembly does not yet support unresolved cross-child peer_func reference '@")
          << logicalName
          << "'; each import_reserved_buffer peer_func must resolve to one sibling public func.func";
      return failure();
    }
    if (failed(verifyImportedPeerCloneContract(peerSource, logicalName))) {
      return failure();
    }

    StringRef peerSymbolName = peerSource.getSymName();
    if (!findFunctionBySymbolName(jobModule, peerSymbolName)) {
      cloneFunctionIntoModule(jobModule, peerSource, peerSymbolName, "private");
    }

    jobModule.walk([&](pto::ImportReservedBufferOp importOp) {
      if (importOp.getPeerFuncAttr().getValue() != logicalName) {
        return;
      }
      importOp.setPeerFuncAttr(
          FlatSymbolRefAttr::get(jobModule.getContext(), peerSymbolName));
    });
  }

  jobModule.walk([&](pto::ImportReservedBufferOp importOp) {
    StringRef peerRef = importOp.getPeerFuncAttr().getValue();
    func::FuncOp localPeer = findFunctionForPeerReference(jobModule, peerRef);
    if (!localPeer) {
      return;
    }
    if (localPeer.getSymName() == peerRef) {
      return;
    }
    importOp.setPeerFuncAttr(
        FlatSymbolRefAttr::get(jobModule.getContext(), localPeer.getSymName()));
  });

  return OwningOpRef<ModuleOp>(jobModule);
}

static constexpr llvm::StringLiteral kEmptyHostStubSource =
    "#ifndef __global__\n#define __global__\n#endif\n\n"
    "#ifndef __gm__\n#define __gm__\n#endif\n\n";

static std::string summarizeMixedChildModule(ModuleOp module) {
  std::string summary;
  llvm::raw_string_ostream os(summary);

  if (auto backendAttr = module->getAttrOfType<StringAttr>("pto.backend")) {
    os << "backend=" << backendAttr.getValue() << " ";
  }
  if (auto kindAttr = module->getAttrOfType<StringAttr>("pto.kernel_kind")) {
    os << "kernel_kind=" << kindAttr.getValue() << " ";
  }

  SmallVector<std::string, mlir::pto::kValue4> exportedNames;
  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    auto visibility = funcOp->getAttrOfType<StringAttr>("sym_visibility");
    if (visibility && visibility.getValue() == "private") {
      continue;
    }
    if (funcOp.isExternal()) {
      continue;
    }
    exportedNames.push_back(funcOp.getSymName().str());
  }

  if (!exportedNames.empty()) {
    os << "exports=[";
    for (size_t i = 0; i < exportedNames.size(); ++i) {
      if (i) {
        os << ", ";
      }
      os << exportedNames[i];
    }
    os << "]";
  } else {
    os << "exports=[]";
  }

  os.flush();
  return summary;
}

static void dumpFailedMixedChildCompileUnit(llvm::StringRef backendName,
                                            llvm::StringRef summary,
                                            ModuleOp module) {
  llvm::errs() << "Error: mixed-backend child module compilation failed"
               << " [" << backendName << "]";
  if (!summary.empty()) {
    llvm::errs() << " {" << summary << "}";
  }
  llvm::errs() << "\n";
  llvm::errs() << "// ----- failed mixed-backend child compile unit ----- //\n";
  module.print(llvm::errs());
  llvm::errs() << "\n";
}

static LogicalResult emitVPTOLLVMFatobj(
    const mlir::pto::PTOASCompileResult &jobResult,
    mlir::pto::PTOASContext &context, llvm::StringRef moduleId,
    llvm::StringRef outputPath);

mlir::pto::PTOASContext::PTOASContext(DialectRegistry &registry,
                                      llvm::StringRef outputPath, int argc,
                                      char **argv)
    : ownedMlirContext(std::make_unique<MLIRContext>(registry)),
      mlirContext(ownedMlirContext.get()), outputPath(outputPath.str()),
      argc(argc), argv(argv) {}

mlir::pto::PTOASContext::PTOASContext(
    MLIRContext &borrowedContext, llvm::StringRef outputPath, int argc,
    char **argv)
    : mlirContext(&borrowedContext), outputPath(outputPath.str()), argc(argc),
      argv(argv) {}

mlir::pto::PTOASContext::~PTOASContext() = default;

LogicalResult
mlir::pto::PTOASContext::initializeEnvironment(bool requiresToolchain,
                                               llvm::raw_ostream &diagOS) {
  if (requiresToolchain) {
    return initializeToolchain(diagOS);
  }
  return success();
}

void mlir::pto::PTOASContext::initializeMLIRContext() {
  // Be tolerant: ptobc decode may materialize ops from dialects that aren't
  // explicitly registered/loaded in this tool yet.
  mlirContext->allowUnregisteredDialects(true);
  mlir::pto::loadPTOASDialects(*mlirContext);
}

MLIRContext &mlir::pto::PTOASContext::getMLIRContext() { return *mlirContext; }

void mlir::pto::PTOASContext::setArch(std::string value) {
  arch = std::move(value);
}

llvm::StringRef mlir::pto::PTOASContext::getArch() const { return arch; }

void mlir::pto::PTOASContext::setBackendInfo(BackendInfo value) {
  backendInfo = std::move(value);
}

const mlir::pto::BackendInfo &mlir::pto::PTOASContext::getBackendInfo() const {
  return backendInfo;
}

void mlir::pto::PTOASContext::setVFSIMTSizeFixMode(VFSIMTSizeFixMode value) {
  vfsimtSizeFixMode = value;
}

mlir::pto::VFSIMTSizeFixMode
mlir::pto::PTOASContext::getVFSIMTSizeFixMode() const {
  return vfsimtSizeFixMode;
}

void mlir::pto::PTOASContext::setOutputCANNVersionOverride(
    std::optional<CANNVersion> value) {
  outputCANNVersionOverride = std::move(value);
}

int mlir::pto::PTOASContext::getArgc() const { return argc; }

char **mlir::pto::PTOASContext::getArgv() const { return argv; }

llvm::StringRef mlir::pto::PTOASContext::getOutputPath() const {
  return outputPath;
}

std::string mlir::pto::PTOASContext::allocModuleId() {
  static size_t nextModuleId = 0;
  return "ptoas_module_" + std::to_string(nextModuleId++);
}

LogicalResult
mlir::pto::PTOASContext::initializeToolchain(llvm::raw_ostream &diagOS) {
  if (toolchain) {
    return success();
  }
  std::optional<mlir::pto::CANNToolchain> discovered =
      mlir::pto::CANNToolchain::create(diagOS);
  if (!discovered) {
    return failure();
  }
  std::optional<CANNVersion> parsedVersion =
      parseCANNVersion(discovered->cannVersionString);
  if (!parsedVersion) {
    diagOS << "Warning: unable to parse CANN version: "
           << discovered->cannVersionString
           << "; defaulting detected version to 9.0.0-beta.1.\n";
    parsedVersion = kDefaultCANNVersion;
  }
  CANNVersion effectiveVersion =
      selectEffectiveOutputCANNVersion(*parsedVersion,
                                       outputCANNVersionOverride);
  discovered->cannVersion = effectiveVersion;
  cannVersion = effectiveVersion;
  toolchain = std::move(*discovered);
  return success();
}

const mlir::pto::CANNToolchain *
mlir::pto::PTOASContext::getToolchain(llvm::raw_ostream &diagOS) const {
  if (!toolchain) {
    diagOS << "Error: CANN toolchain is required but was not initialized.\n";
    return nullptr;
  }
  return &*toolchain;
}

mlir::pto::CANNVersion
mlir::pto::PTOASContext::getCANNVersionOrDefault() const {
  return selectEffectiveOutputCANNVersion(cannVersion,
                                          outputCANNVersionOverride);
}

mlir::pto::TempFileRegistry &mlir::pto::PTOASContext::getTempFiles() {
  return tempFiles;
}

LogicalResult
mlir::pto::PTOASContext::createTempPath(llvm::StringRef prefix,
                                        llvm::StringRef suffix,
                                        std::string &path) {
  return tempFiles.create(prefix, suffix, path, llvm::errs());
}

static bool hasPTOEntry(ModuleOp module) {
  bool found = false;
  module.walk([&](func::FuncOp func) {
    if (mlir::pto::isPTOEntryFunction(func)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

class EmitCBackendJob {
public:
  EmitCBackendJob(OwningOpRef<ModuleOp> &module,
                  mlir::pto::PTOASCompileResult &result)
      : module(module), result(result) {}

  LogicalResult run(PTOASContext &context);

private:
  OwningOpRef<ModuleOp> &module;
  mlir::pto::PTOASCompileResult &result;
};

class VPTOBackendJob {
public:
  VPTOBackendJob(OwningOpRef<ModuleOp> &module,
                 mlir::pto::PTOASCompileResult &result)
      : module(module), result(result) {}

  LogicalResult run(PTOASContext &context);

private:
  OwningOpRef<ModuleOp> &module;
  mlir::pto::PTOASCompileResult &result;
};

class BackendChildJob {
public:
  virtual ~BackendChildJob() = default;
  virtual LogicalResult run(PTOASContext &context) = 0;
};

class EmitCBackendChildJob final : public BackendChildJob {
public:
  EmitCBackendChildJob(OwningOpRef<ModuleOp> &&module,
                       std::string summary,
                       SmallVectorImpl<std::string> &fatobjPaths)
      : module(std::move(module)), summary(std::move(summary)),
        fatobjPaths(fatobjPaths) {}

  LogicalResult run(PTOASContext &context) override {
    ModuleOp op = module.get();
    op->setAttr("pto.backend", StringAttr::get(op.getContext(), "emitc"));

    mlir::pto::PTOASCompileResult jobResult;
    if (mlir::pto::compilePTOASModule(module, context,
                                      mlir::pto::PTOBackend::EmitC, jobResult,
                                      /*emitVPTOHostStub=*/false) != 0) {
      dumpFailedMixedChildCompileUnit("emitc", summary, op);
      return failure();
    }
    if (jobResult.kind != mlir::pto::PTOASCompileResultKind::Text) {
      llvm::errs() << "Error: EmitC backend child job produced non-text "
                      "output.\n";
      dumpFailedMixedChildCompileUnit("emitc", summary, op);
      return failure();
    }

    std::string fatobjPath;
    if (failed(context.createTempPath("ptoas-emitc-fatobj", ".o", fatobjPath))) {
      return failure();
    }
    const mlir::pto::CANNToolchain *toolchain =
        context.getToolchain(llvm::errs());
    if (!toolchain) {
      return failure();
    }
    if (failed(mlir::pto::emitFatobjCCE(
            jobResult.textOutput, fatobjPath, *toolchain,
            context.getTempFiles(), llvm::errs()))) {
      dumpFailedMixedChildCompileUnit("emitc", summary, op);
      return failure();
    }

    fatobjPaths.push_back(std::move(fatobjPath));
    return success();
  }

private:
  OwningOpRef<ModuleOp> module;
  std::string summary;
  SmallVectorImpl<std::string> &fatobjPaths;
};

class VPTOBackendChildJob final : public BackendChildJob {
public:
  VPTOBackendChildJob(OwningOpRef<ModuleOp> &&module, std::string summary,
                      std::string moduleId,
                      SmallVectorImpl<std::string> &fatobjPaths)
      : module(std::move(module)), summary(std::move(summary)),
        moduleId(std::move(moduleId)),
        fatobjPaths(fatobjPaths) {}

  LogicalResult run(PTOASContext &context) override {
    ModuleOp op = module.get();
    op->setAttr("pto.backend", StringAttr::get(op.getContext(), "vpto"));

    bool emitHostStub = hasPTOEntry(op);
    mlir::pto::PTOASCompileResult jobResult;
    if (mlir::pto::compilePTOASModule(
            module, context, mlir::pto::PTOBackend::VPTO, jobResult,
            emitHostStub) != 0) {
      dumpFailedMixedChildCompileUnit("vpto", summary, op);
      return failure();
    }
    if (jobResult.kind != mlir::pto::PTOASCompileResultKind::VPTOObject) {
      llvm::errs() << "Error: VPTO backend child job produced non-object "
                      "output.\n";
      dumpFailedMixedChildCompileUnit("vpto", summary, op);
      return failure();
    }

    std::string fatobjPath;
    if (failed(context.createTempPath("ptoas-vpto-fatobj", ".o", fatobjPath))) {
      return failure();
    }

    if (failed(emitVPTOLLVMFatobj(jobResult, context, moduleId, fatobjPath))) {
      dumpFailedMixedChildCompileUnit("vpto", summary, op);
      return failure();
    }

    fatobjPaths.push_back(std::move(fatobjPath));
    return success();
  }

private:
  OwningOpRef<ModuleOp> module;
  std::string summary;
  std::string moduleId;
  SmallVectorImpl<std::string> &fatobjPaths;
};

class FatobjLinkJob {
public:
  explicit FatobjLinkJob(ArrayRef<std::string> fatobjPaths)
      : fatobjPaths(fatobjPaths) {}

  LogicalResult run(PTOASContext &context) {
    if (fatobjPaths.size() < mlir::pto::kValue2) {
      llvm::errs()
          << "Error: mixed backend link requires at least two fatobjs.\n";
      return failure();
    }

    std::string stderrPath;
    if (failed(context.createTempPath("ptoas-fatobj", ".log", stderrPath))) {
      return failure();
    }
    const mlir::pto::CANNToolchain *toolchain =
        context.getToolchain(llvm::errs());
    if (!toolchain) {
      return failure();
    }
    return mlir::pto::linkFatobjs(fatobjPaths, context.getOutputPath(),
                                  *toolchain, stderrPath, llvm::errs());
  }

private:
  ArrayRef<std::string> fatobjPaths;
};

LogicalResult EmitCBackendJob::run(PTOASContext &context) {
  OwningOpRef<ModuleOp> singleChildJobModule;
  OwningOpRef<ModuleOp> *compileUnit = &module;
  ModuleOp op = module.get();
  op->setAttr("pto.backend", StringAttr::get(op.getContext(), "emitc"));

  SmallVector<ModuleOp, mlir::pto::kValue4> children(op.getOps<ModuleOp>());
  if (!isUserVisibleIROutputRequested() && children.size() == 1 &&
      isBackendPartitionedContainer(op)) {
    FailureOr<OwningOpRef<ModuleOp>> jobModuleOr =
        buildBackendChildCompileUnit(op, children.front());
    if (failed(jobModuleOr)) {
      return failure();
    }
    singleChildJobModule = std::move(*jobModuleOr);
    singleChildJobModule.get()->setAttr(
        "pto.backend",
        StringAttr::get(singleChildJobModule.get()->getContext(), "emitc"));
    compileUnit = &singleChildJobModule;
  }

  if (mlir::pto::compilePTOASModule(*compileUnit, context,
                                    mlir::pto::PTOBackend::EmitC, result,
                                    /*emitVPTOHostStub=*/false) != 0) {
    return failure();
  }
  if (result.kind != mlir::pto::PTOASCompileResultKind::Text) {
    llvm::errs() << "Error: EmitC backend job produced non-text output.\n";
    return failure();
  }
  return success();
}

LogicalResult VPTOBackendJob::run(PTOASContext &context) {
  OwningOpRef<ModuleOp> singleChildJobModule;
  OwningOpRef<ModuleOp> *compileUnit = &module;
  ModuleOp op = module.get();
  op->setAttr("pto.backend", StringAttr::get(op.getContext(), "vpto"));

  SmallVector<ModuleOp, mlir::pto::kValue4> children(op.getOps<ModuleOp>());
  // PTODSL emits a backend-partitioned outer container even when there is only
  // one child module.  For object compilation, the actual VPTO compile unit is
  // the normalized child job module with outer attributes/imports folded in.
  // Keep user-visible IR output modes on the original container so dump flags
  // still reflect the input module structure the user asked to inspect.
  if (!isUserVisibleIROutputRequested() && children.size() == 1 &&
      isBackendPartitionedContainer(op)) {
    FailureOr<OwningOpRef<ModuleOp>> jobModuleOr =
        buildBackendChildCompileUnit(op, children.front());
    if (failed(jobModuleOr)) {
      return failure();
    }
    singleChildJobModule = std::move(*jobModuleOr);
    singleChildJobModule.get()->setAttr(
        "pto.backend",
        StringAttr::get(singleChildJobModule.get()->getContext(), "vpto"));
    compileUnit = &singleChildJobModule;
    op = singleChildJobModule.get();
  }

  bool emitHostStub = hasPTOEntry(op);
  if (mlir::pto::compilePTOASModule(
          *compileUnit, context, mlir::pto::PTOBackend::VPTO, result,
          emitHostStub) != 0) {
    return failure();
  }
  if (result.kind == mlir::pto::PTOASCompileResultKind::Text) {
    return success();
  }
  if (result.kind != mlir::pto::PTOASCompileResultKind::VPTOObject) {
    llvm::errs() << "Error: VPTO backend job produced non-VPTO output.\n";
    return failure();
  }

  if (context.getOutputPath().empty() || context.getOutputPath() == "-") {
    llvm::errs() << "Error: object output requires an explicit file path "
                    "passed with -o.\n";
    return failure();
  }

  std::string moduleId = context.allocModuleId();
  if (failed(emitVPTOLLVMFatobj(result, context, moduleId,
                                context.getOutputPath()))) {
    return failure();
  }

  result.reset();
  result.kind = mlir::pto::PTOASCompileResultKind::MixedObject;
  return success();
}

static LogicalResult emitVPTOLLVMFatobj(
    const mlir::pto::PTOASCompileResult &jobResult, PTOASContext &context,
    llvm::StringRef moduleId, llvm::StringRef outputPath) {
  llvm::StringRef stubSource = kEmptyHostStubSource;
  if (!jobResult.vptoStubSource.empty()) {
    stubSource = jobResult.vptoStubSource;
  }

  const mlir::pto::CANNToolchain *toolchain =
      context.getToolchain(llvm::errs());
  if (!toolchain) {
    return failure();
  }
  if (failed(mlir::pto::emitFatobjLLVM(
          jobResult.vptoCubeModule.module.get(),
          jobResult.vptoVectorModule.module.get(), stubSource,
          jobResult.vptoBridgeWrapperSource,
          outputPath, moduleId, *toolchain, context.getTempFiles(),
          context.getVFSIMTSizeFixMode(), llvm::errs()))) {
    return failure();
  }
  return success();
}

static LogicalResult collectChildJobs(
    ModuleOp module, mlir::pto::PTOBackend defaultBackend,
    bool cliBackendOverride,
    PTOASContext &context, SmallVectorImpl<std::string> &fatobjPaths,
    SmallVectorImpl<std::unique_ptr<BackendChildJob>> &backendJobs) {
  SmallVector<ModuleOp, mlir::pto::kValue4> children(module.getOps<ModuleOp>());
  for (ModuleOp child : children) {
    std::optional<mlir::pto::PTOBackend> childBackend;
    if (failed(parseDriverBackendAttr(child.getOperation(), childBackend))) {
      return failure();
    }

    FailureOr<OwningOpRef<ModuleOp>> jobModuleOr =
        buildBackendChildCompileUnit(module, child);
    if (failed(jobModuleOr)) {
      return failure();
    }
    OwningOpRef<ModuleOp> jobModule = std::move(*jobModuleOr);
    if (llvm::sys::Process::GetEnv("PTOAS_DEBUG_CHILD_UNIT")) {
      llvm::errs() << "// ----- child compile unit ----- //\n";
      jobModule->print(llvm::errs());
      llvm::errs() << "\n";
    }
    std::string summary = summarizeMixedChildModule(jobModule.get());
    mlir::pto::PTOBackend effectiveBackend =
        cliBackendOverride ? defaultBackend
                           : childBackend.value_or(defaultBackend);
    if (effectiveBackend == mlir::pto::PTOBackend::VPTO) {
      backendJobs.push_back(std::make_unique<VPTOBackendChildJob>(
          std::move(jobModule), std::move(summary), context.allocModuleId(),
          fatobjPaths));
    } else {
      backendJobs.push_back(std::make_unique<EmitCBackendChildJob>(
          std::move(jobModule), std::move(summary), fatobjPaths));
}
  }
  return success();
}

static LogicalResult resolveSingleBackend(
    bool cliBackendSpecified,
    std::optional<mlir::pto::PTOBackend> moduleBackend,
    mlir::pto::PTOBackend defaultBackend, ModuleOp module,
    std::optional<mlir::pto::PTOBackend> &singleBackend) {
  singleBackend = std::nullopt;
  if (cliBackendSpecified) {
    SmallVector<ModuleOp, mlir::pto::kValue4> children(module.getOps<ModuleOp>());
    if (!isUserVisibleIROutputRequested() && children.size() > 1 &&
        isBackendPartitionedContainer(module)) {
      return success();
    }
    singleBackend = defaultBackend;
    return success();
  }
  if (moduleBackend) {
    singleBackend = *moduleBackend;
    return success();
  }

  SmallVector<ModuleOp, mlir::pto::kValue4> children(module.getOps<ModuleOp>());
  if (!isUserVisibleIROutputRequested() && children.size() > 1) {
    if (!isBackendPartitionedContainer(module)) {
      llvm::errs() << "Error: mixed pto.backend fatobj mode expects either a "
                      "single module or an outer module containing only child "
                      "modules; found non-module top-level ops alongside child "
                      "modules.\n";
      return failure();
    }
    return success();
  }

  std::optional<mlir::pto::PTOBackend> firstChildBackend;
  for (ModuleOp child : children) {
    std::optional<mlir::pto::PTOBackend> childBackend;
    if (failed(parseDriverBackendAttr(child.getOperation(), childBackend))) {
      return failure();
    }

    mlir::pto::PTOBackend effectiveChildBackend =
        childBackend.value_or(defaultBackend);
    if (!firstChildBackend) {
      firstChildBackend = effectiveChildBackend;
      continue;
    }
    if (*firstChildBackend != effectiveChildBackend) {
      return success();
    }
  }

  if (firstChildBackend) {
    singleBackend = *firstChildBackend;
  }
  else {
    singleBackend = defaultBackend;
  }
  return success();
}

static LogicalResult buildBackendInfo(ModuleOp module, bool cliBackendSpecified,
                                      mlir::pto::BackendInfo &backendInfo) {
  backendInfo = mlir::pto::BackendInfo();
  backendInfo.cliBackendOverride = cliBackendSpecified;
  if (!parseDriverBackend(mlir::pto::ptoBackend,
                          backendInfo.defaultBackend)) {
    llvm::errs() << "Error: invalid --pto-backend='" << mlir::pto::ptoBackend
                 << "'. Expected 'emitc' or 'vpto'.\n";
    return failure();
  }

  std::optional<mlir::pto::PTOBackend> moduleBackend;
  if (!cliBackendSpecified) {
    if (failed(parseDriverBackendAttr(module.getOperation(), moduleBackend))) {
      return failure();
    }
  }

  if (failed(resolveSingleBackend(cliBackendSpecified, moduleBackend,
                                  backendInfo.defaultBackend, module,
                                  backendInfo.singleBackend))) {
    return failure();
  }

  if (backendInfo.singleBackend) {
    backendInfo.requiresToolchain =
        *backendInfo.singleBackend == mlir::pto::PTOBackend::VPTO &&
        !mlir::pto::emitMlirIR && !mlir::pto::emitVPTO &&
        !mlir::pto::emitVPTOLLVMDialect;
    return success();
  }

  if (mlir::pto::emitMlirIR || mlir::pto::emitVPTO ||
      mlir::pto::emitVPTOLLVMDialect ||
      mlir::pto::ptoPrintSeamIR || !mlir::pto::ptoSeamIRFile.empty()) {
    llvm::errs() << "Error: mixed pto.backend fatobj mode does not support "
                    "debug IR output flags.\n";
    return failure();
  }
  if (outputFilename.empty() || outputFilename == "-") {
    llvm::errs() << "Error: mixed pto.backend fatobj mode requires an "
                    "explicit file path passed with -o.\n";
    return failure();
  }

  backendInfo.requiresToolchain = true;
  return success();
}

static LogicalResult runPTOASJobs(OwningOpRef<ModuleOp> &module,
                                  PTOASContext &context,
                                  mlir::pto::PTOASCompileResult &result) {
  const mlir::pto::BackendInfo &backendInfo = context.getBackendInfo();
  if (backendInfo.singleBackend) {
    if (*backendInfo.singleBackend == mlir::pto::PTOBackend::EmitC) {
      EmitCBackendJob singleJob(module, result);
      return singleJob.run(context);
    }
    VPTOBackendJob singleJob(module, result);
    return singleJob.run(context);
  }

  SmallVector<std::unique_ptr<BackendChildJob>, mlir::pto::kValue4> backendJobs;
  SmallVector<std::string, mlir::pto::kValue4> fatobjPaths;
  if (failed(collectChildJobs(module.get(), backendInfo.defaultBackend,
                              backendInfo.cliBackendOverride,
                              context, fatobjPaths, backendJobs))) {
    return failure();
  }

  result.reset();
  result.kind = mlir::pto::PTOASCompileResultKind::MixedObject;

  for (size_t i = 0, e = backendJobs.size(); i < e; ++i) {
    if (failed(backendJobs[i]->run(context))) {
      return failure();
    }
  }

  FatobjLinkJob linkJob(fatobjPaths);
  if (failed(linkJob.run(context))) {
    return failure();
  }

  return success();
}

static LogicalResult writeTextOutput(llvm::StringRef output,
                                     llvm::StringRef outputPath) {
  std::error_code ec;
  llvm::ToolOutputFile outputFile(outputPath, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << ec.message() << "\n";
    return failure();
  }
  outputFile.os() << output;
  outputFile.os().flush();
  outputFile.keep();
  return success();
}

// PTOAS driver jobs:
// +----------------------------------------------------------+
// |                        .pto                              |
// +----------------------------------------------------------+
// +-------------+ +------------+ +------------+ +------------+
// | EmitC job   | | VPTO job   | | EmitC      | | VPTO       |
// |             | |            | | child job  | | child job  |
// |             | |            | +------------+ +------------+
// |             | |            | +---------------------------+
// |             | |            | | Fatobj link job           |
// +-------------+ +------------+ +---------------------------+
// +-------------+ +------------------------------------------+
// | C++ source  | |                fatobj                    |
// +-------------+ +------------------------------------------+
static int runPTOASDriver(int argc, char **argv,
                          MLIRContext *borrowedContext = nullptr) {
  DialectRegistry registry;
  mlir::pto::registerPTOASDialects(registry);
  if (borrowedContext) {
    borrowedContext->appendDialectRegistry(registry);
  }
  mlir::pto::registerPTOASPassesAndCLOptions();
  llvm::cl::SetVersionPrinter(printPTOASVersion);

  // The Python entry point may invoke the driver repeatedly in one process.
  // Restore every registered LLVM option to its declared default before
  // parsing the next invocation.
  llvm::cl::ResetAllOptionOccurrences();

  const bool cliArchSpecified = hasCLIOption(argc, argv, "--pto-arch");
  const bool cliBackendSpecified = hasCLIOption(argc, argv, "--pto-backend");

  llvm::cl::ParseCommandLineOptions(argc, argv, "PTO Assembler (ptoas)\n");

  std::optional<mlir::pto::CANNVersion> outputCANNVersionOverride;
  if (!parseRequestedOutputCANNVersion(mlir::pto::cannOutputVersion,
                                       outputCANNVersionOverride,
                                       llvm::errs())) {
    return 1;
  }

  std::unique_ptr<PTOASContext> context;
  if (borrowedContext) {
    context = std::make_unique<PTOASContext>(*borrowedContext, outputFilename,
                                             argc, argv);
  } else {
    context =
        std::make_unique<PTOASContext>(registry, outputFilename, argc, argv);
  }
  context->setOutputCANNVersionOverride(outputCANNVersionOverride);
  context->setVFSIMTSizeFixMode(mlir::pto::vptoFixVFSIMTSize);
  context->initializeMLIRContext();

  std::unique_ptr<llvm::MemoryBuffer> inputBuffer = readInputBuffer();
  if (!inputBuffer) {
    return 1;
  }

  std::string arch;
  OwningOpRef<ModuleOp> module = loadInputModule(
      std::move(inputBuffer), context->getMLIRContext(), cliArchSpecified, arch);
  if (!module) {
    return 1;
  }
  context->setArch(std::move(arch));

  mlir::pto::BackendInfo backendInfo;
  if (failed(buildBackendInfo(module.get(), cliBackendSpecified, backendInfo))) {
    return 1;
  }
  context->setBackendInfo(std::move(backendInfo));
  (void)context->initializeEnvironment(
      context->getBackendInfo().requiresToolchain, llvm::errs());

  mlir::pto::PTOASCompileResult result;
  if (failed(runPTOASJobs(module, *context, result))) {
    return 1;
  }

  if (result.kind == mlir::pto::PTOASCompileResultKind::Text) {
    return failed(writeTextOutput(result.textOutput, context->getOutputPath()));
  }
  if (result.kind == mlir::pto::PTOASCompileResultKind::MixedObject) {
    return 0;
  }

  llvm::errs() << "Error: unsupported ptoas compile result.\n";
  return 1;
}

int mlir::pto::runPTOAS(int argc, char **argv) {
  return runPTOASDriver(argc, argv);
}

int mlir::pto::runPTOAS(int argc, char **argv,
                        MLIRContext &borrowedContext) {
  return runPTOASDriver(argc, argv, &borrowedContext);
}
