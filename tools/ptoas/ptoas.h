// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTOAS_H
#define PTOAS_H

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "ObjectEmission.h"
#include "PTO/Compiler/CompilerApi.h"
#include "PTO/Transforms/VPTOLLVMEmitter.h"
#include "VFSIMTSizePatcher.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"


namespace mlir {
class AsmParserState;
class DialectRegistry;
} // namespace mlir

namespace mlir::pto {

extern llvm::cl::opt<bool> emitMlirIR;
extern llvm::cl::opt<std::string> ptoTargetArch;
extern llvm::cl::opt<std::string> ptoBackend;
extern llvm::cl::opt<bool> emitVPTO;
extern llvm::cl::opt<bool> emitVPTOLLVMDialect;
extern llvm::cl::opt<bool> ptoPrintSeamIR;
extern llvm::cl::opt<std::string> ptoSeamIRFile;
extern llvm::cl::opt<VFSIMTSizeFixMode> vptoFixVFSIMTSize;

enum class PTOBackend {
  EmitC,
  VPTO,
};

struct BackendInfo {
  PTOBackend defaultBackend = PTOBackend::EmitC;
  std::optional<PTOBackend> singleBackend;
  bool cliBackendOverride = false;
  bool requiresToolchain = false;
};

enum class PTOASCompileResultKind {
  Text,
  VPTOObject,
  MixedObject,
};

class PTOASContext {
public:
  PTOASContext(DialectRegistry &registry, llvm::StringRef outputPath);
  // `borrowedContext` is a non-owning, mutable borrow: PTOAS dialect loading
  // mutates the caller's context, hence a pointer rather than a const ref.
  PTOASContext(MLIRContext *borrowedContext, llvm::StringRef outputPath);
  ~PTOASContext();

  LogicalResult initializeEnvironment(bool requiresToolchain,
                                      llvm::raw_ostream &diagOS);
  // Loads the PTOAS dialects into the borrowed context (shallow-const: the
  // pointed-to context is intentionally mutated).
  void initializeMLIRContext() const;

  // Returns the borrowed context; the reference stays mutable by design so
  // driver callers can parse and lower into it.
  MLIRContext &getMLIRContext() const;

  void setArch(std::string value);
  llvm::StringRef getArch() const;

  void setBackendInfo(BackendInfo value);
  const BackendInfo &getBackendInfo() const;

  void setVFSIMTSizeFixMode(VFSIMTSizeFixMode value);
  VFSIMTSizeFixMode getVFSIMTSizeFixMode() const;

  llvm::StringRef getOutputPath() const;
  std::string allocModuleId() const;

  const CANNToolchain *getToolchain(llvm::raw_ostream &diagOS) const;
  CANNVersion getCANNVersionOrDefault() const;

  TempFileRegistry &getTempFiles();
  LogicalResult createTempPath(llvm::StringRef prefix, llvm::StringRef suffix,
                               std::string &path);

private:
  std::unique_ptr<MLIRContext> ownedMlirContext;
  MLIRContext *mlirContext = nullptr;
  std::string outputPath;
  std::string arch;
  BackendInfo backendInfo;
  VFSIMTSizeFixMode vfsimtSizeFixMode = VFSIMTSizeFixMode::Auto;
  CANNVersion cannVersion = kDefaultCANNVersion;
  std::optional<CANNToolchain> toolchain;
  TempFileRegistry tempFiles;

  LogicalResult initializeToolchain(llvm::raw_ostream &diagOS);
};

struct PTOASCompileResult {
  void reset() {
    textOutput.clear();
    vptoStubSource.clear();
    vptoCubeModule.reset();
    vptoVectorModule.reset();
    kind = PTOASCompileResultKind::Text;
  }

  PTOASCompileResultKind kind = PTOASCompileResultKind::Text;
  std::string textOutput;
  std::string vptoStubSource;
  EmittedLLVMModule vptoCubeModule;
  EmittedLLVMModule vptoVectorModule;
};

int compilePTOASModule(OwningOpRef<ModuleOp> &module,
                       PTOASContext &context, PTOBackend effectiveBackend,
                       PTOASCompileResult &result,
                       bool emitVPTOHostStub = true);
void registerPTOASDialects(DialectRegistry &registry);
void registerPTOASPassesAndCLOptions();
void loadPTOASDialects(MLIRContext &context);

// Reusable driver entry shared by the Python extension and standalone CLI.
PTOAS_COMPILER_EXPORT int runPTOAS(int argc, char **argv);
PTOAS_COMPILER_EXPORT int
runPTOAS(int argc, char **argv, MLIRContext &borrowedContext);
// String-vector variant used by the Python binding; the char* bridging to
// the LLVM command-line parser stays internal to the driver.
PTOAS_COMPILER_EXPORT int
runPTOAS(const std::vector<std::string> &args, MLIRContext &borrowedContext);

// Attach textual-.pto SSA name hints (function args, block args, op results)
// to the parsed module's Locations as debug metadata. Called by the driver
// right after parsing a textual .pto input so the names survive lowering.
// No-op for non-textual (PTOBC) inputs or modules without recoverable names.
void applyTextualNameHintsToModule(ModuleOp module,
                                   const AsmParserState &parserState);

} // namespace mlir::pto

#endif
