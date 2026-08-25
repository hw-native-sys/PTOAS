// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTOAS_OBJECT_EMISSION_H
#define PTOAS_OBJECT_EMISSION_H

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

LogicalResult compileCppToDeviceObject(
    llvm::StringRef cppPath, llvm::StringRef outObjPath,
    ObjectEmissionDeviceTarget target, const CANNToolchain &toolchain,
    llvm::StringRef stderrPath, llvm::raw_ostream &diagOS);

LogicalResult compileLLVMToDeviceObject(
    llvm::StringRef llPath, llvm::StringRef outObjPath,
    ObjectEmissionDeviceTarget target, const CANNToolchain &toolchain,
    llvm::StringRef stderrPath, llvm::raw_ostream &diagOS);

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

LogicalResult emitVPTOVectorDeviceObject(
    llvm::Module &module, llvm::StringRef llPath, llvm::StringRef outObjPath,
    const CANNToolchain &toolchain, llvm::StringRef bridgeBitcodePath,
    llvm::StringRef stderrPath, llvm::raw_ostream &diagOS);

LogicalResult emitVPTOCubeDeviceObject(
    llvm::Module &module, llvm::StringRef llPath, llvm::StringRef outObjPath,
    const CANNToolchain &toolchain, llvm::StringRef bridgeBitcodePath,
    llvm::StringRef stderrPath, llvm::raw_ostream &diagOS);

LogicalResult emitFatobjLLVM(
    llvm::Module *cubeModule, llvm::Module *vectorModule,
    llvm::StringRef stubSource, llvm::StringRef bridgeWrapperSource,
    llvm::StringRef outputPath,
    llvm::StringRef moduleId, const CANNToolchain &toolchain,
    TempFileRegistry &tempFiles, VFSIMTSizeFixMode vfsimtSizeFixMode,
    llvm::raw_ostream &diagOS);

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

LogicalResult emitFatobjLLVMWithRuntime(llvm::Module *cubeModule,
                                        llvm::Module *vectorModule,
                                        llvm::StringRef stubSource,
                                        llvm::StringRef bridgeWrapperSource,
                                        llvm::ToolOutputFile &outputFile,
                                        VFSIMTSizeFixMode vfsimtSizeFixMode,
                                        llvm::raw_ostream &diagOS);

} // namespace mlir::pto

#endif
