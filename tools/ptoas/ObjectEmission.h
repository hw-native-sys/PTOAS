// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTOAS_OBJECT_EMISSION_H
#define PTOAS_OBJECT_EMISSION_H

#include "BishengScheduler.h"
#include "PTO/Support/CodeConstants.h"
#include "PTO/Support/CANNVersion.h"
#include "VFSIMTSizePatcher.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Support/LogicalResult.h"

#include <optional>
#include <string>
#include <vector>

namespace llvm {
class ToolOutputFile;
class Module;
class raw_ostream;
}

namespace mlir::pto {

enum class ObjectEmissionDeviceTarget {
  Vector,
  Cube,
};

class CANNToolchain {
public:
  static std::optional<CANNToolchain> create(llvm::raw_ostream &diagOS);

  llvm::StringRef vptoPublicABISuffix(ObjectEmissionDeviceTarget target) const;
  LogicalResult validate(llvm::raw_ostream &diagOS) const;

  std::string ascendHomePath;
  std::string bishengPath;
  std::string bishengCc1Path;
  std::string cceLdPath;
  std::string ldLldPath;
  std::string resourceDirPath;
  std::string resourceIncludeDirPath;
  std::string cceStubDirPath;
  std::string bishengCompilerBinDirPath;
  std::string ptoIsaPath;
  std::string cannVersionString;
  CANNVersion cannVersion = kDefaultCANNVersion;
  std::vector<std::string> cppIncludeDirs;
};

class TempFileRegistry {
public:
  TempFileRegistry() = default;
  TempFileRegistry(const TempFileRegistry &) = delete;
  TempFileRegistry &operator=(const TempFileRegistry &) = delete;
  TempFileRegistry(TempFileRegistry &&) = delete;
  TempFileRegistry &operator=(TempFileRegistry &&) = delete;

  ~TempFileRegistry();

  void cleanup();
  LogicalResult create(llvm::StringRef prefix, llvm::StringRef suffix,
                       std::string &path, llvm::raw_ostream &diagOS);

private:
  llvm::SmallVector<std::string, mlir::pto::kValue8> paths;
};

LogicalResult writeLLVMModule(llvm::Module &module, llvm::StringRef path,
                              llvm::raw_ostream &diagOS);

LogicalResult writeCppSource(llvm::StringRef cppSource, llvm::StringRef path,
                             llvm::raw_ostream &diagOS);

LogicalResult writeHostStubSource(llvm::StringRef stubSource,
                                  llvm::StringRef path,
                                  llvm::raw_ostream &diagOS);

/// Install the merged, canonical pass-through list for this invocation. The
/// driver renders every source into this list and writes the same tokens back
/// as `pto.bisheng_args`, so object emission has one representation. The tokens
/// are forwarded verbatim: PTOAS does not validate or reorder them, it only
/// places them after its own arguments and before the per-variant tail.
void setBishengArgs(llvm::ArrayRef<std::string> args);

LogicalResult compileCppToDeviceObject(
    llvm::StringRef cppPath, llvm::StringRef outObjPath,
    ObjectEmissionDeviceTarget target, const CANNToolchain &toolchain,
    llvm::StringRef stderrPath, llvm::raw_ostream &diagOS);

LogicalResult compileLLVMToDeviceObject(
    llvm::StringRef llPath, llvm::StringRef outObjPath, ObjectEmissionDeviceTarget target,
    const CANNToolchain& toolchain, llvm::StringRef stderrPath, llvm::raw_ostream& diagOS,
    BishengSchedulerMode schedulerMode = BishengSchedulerMode::Auto);

LogicalResult emitCppVectorDeviceObject(
    llvm::StringRef cppSource, llvm::StringRef cppPath,
    llvm::StringRef outObjPath, const CANNToolchain &toolchain,
    llvm::StringRef stderrPath, llvm::raw_ostream &diagOS);

LogicalResult emitCppCubeDeviceObject(
    llvm::StringRef cppSource, llvm::StringRef cppPath,
    llvm::StringRef outObjPath, const CANNToolchain &toolchain,
    llvm::StringRef stderrPath, llvm::raw_ostream &diagOS);

LogicalResult emitCppFatobj(llvm::StringRef cppSource, llvm::StringRef cppPath,
                            llvm::StringRef outObjPath,
                            const CANNToolchain &toolchain,
                            llvm::StringRef stderrPath,
                            llvm::raw_ostream &diagOS);

LogicalResult emitFatobjCCE(llvm::StringRef cppSource,
                            llvm::StringRef outputPath,
                            const CANNToolchain &toolchain,
                            TempFileRegistry &tempFiles,
                            llvm::raw_ostream &diagOS);

/// One VPTO bridge wrapper translation unit, as produced by WrapperGen. An
/// absent core means the unit is core-independent and is compiled for every
/// device module present.
struct BridgeWrapperTU {
  std::string source;
  std::optional<ObjectEmissionDeviceTarget> core;
  /// A print unit needs the CCE print support enabled in its device compile.
  bool needsPrintSupport = false;
};

/// How a printing kernel receives its DebugTunnel pointer. None means the
/// module does not print, so no print support is enabled anywhere.
enum class PrintDTDataMode { None, Implicit, Explicit };

/// Bridge wrapper bitcode to link into one device module before the Bisheng
/// object compile. `linkedLLPath` is where the linked IR is written.
struct BridgeLinkInputs {
  llvm::ArrayRef<std::string> bitcodePaths;
  llvm::StringRef linkedLLPath;
};

LogicalResult emitVPTOVectorDeviceObject(
    llvm::Module& module, llvm::StringRef llPath, llvm::StringRef outObjPath, const CANNToolchain& toolchain,
    llvm::StringRef stderrPath, llvm::raw_ostream& diagOS,
    BishengSchedulerMode schedulerMode = BishengSchedulerMode::Auto,
    BridgeLinkInputs bridge = {},
    PrintDTDataMode printMode = PrintDTDataMode::None);

LogicalResult emitVPTOCubeDeviceObject(
    llvm::Module &module, llvm::StringRef llPath, llvm::StringRef outObjPath,
    const CANNToolchain &toolchain, llvm::StringRef stderrPath,
    llvm::raw_ostream &diagOS,
    BishengSchedulerMode schedulerMode = BishengSchedulerMode::Auto,
    BridgeLinkInputs bridge = {},
    PrintDTDataMode printMode = PrintDTDataMode::None);

LogicalResult emitFatobjLLVM(
    llvm::Module* cubeModule, llvm::Module* vectorModule, llvm::StringRef stubSource, llvm::StringRef outputPath,
    llvm::StringRef moduleId, const CANNToolchain& toolchain, TempFileRegistry& tempFiles,
    VFSIMTSizeFixMode vfsimtSizeFixMode, llvm::raw_ostream& diagOS,
    BishengSchedulerMode schedulerMode = BishengSchedulerMode::Auto,
    llvm::ArrayRef<BridgeWrapperTU> bridgeWrappers = {},
    PrintDTDataMode printMode = PrintDTDataMode::None);

// Device-object output mode: runs the same device pipeline as
// emitFatobjLLVM (bridge wrapper linking, per-target device compilation,
// VFSIMT size repair, and the ld.lld -r merge) but stops before host-stub
// compilation and fatobj packaging, writing the merged relocatable AI Core
// ELF to outputPath. Requires at least one device module.
LogicalResult emitVPTODeviceObjectsLLVM(
    llvm::Module* cubeModule, llvm::Module* vectorModule,
    llvm::StringRef outputPath, const CANNToolchain& toolchain,
    TempFileRegistry& tempFiles, VFSIMTSizeFixMode vfsimtSizeFixMode,
    llvm::raw_ostream& diagOS,
    BishengSchedulerMode schedulerMode = BishengSchedulerMode::Auto,
    llvm::ArrayRef<BridgeWrapperTU> bridgeWrappers = {},
    PrintDTDataMode printMode = PrintDTDataMode::None);

LogicalResult mergeDeviceObjects(llvm::ArrayRef<std::string> deviceObjPaths,
                                 llvm::StringRef outObjPath,
                                 const CANNToolchain &toolchain,
                                 llvm::StringRef stderrPath,
                                 llvm::raw_ostream &diagOS);

LogicalResult compileStubToFatobj(
    llvm::StringRef stubPath, llvm::StringRef deviceObjPath,
    llvm::StringRef outputPath, llvm::StringRef moduleId,
    const CANNToolchain &toolchain, llvm::StringRef stderrPath,
    llvm::raw_ostream &diagOS);

LogicalResult linkFatobjs(llvm::ArrayRef<std::string> fatobjPaths,
                          llvm::StringRef outputPath,
                          const CANNToolchain &toolchain,
                          llvm::StringRef stderrPath,
                          llvm::raw_ostream &diagOS);

LogicalResult emitFatobjLLVMWithRuntime(
    llvm::Module* cubeModule, llvm::Module* vectorModule, llvm::StringRef stubSource, llvm::ToolOutputFile& outputFile,
    VFSIMTSizeFixMode vfsimtSizeFixMode, llvm::raw_ostream& diagOS,
    BishengSchedulerMode schedulerMode = BishengSchedulerMode::Auto,
    llvm::ArrayRef<BridgeWrapperTU> bridgeWrappers = {},
    PrintDTDataMode printMode = PrintDTDataMode::None);

} // namespace mlir::pto

#endif
