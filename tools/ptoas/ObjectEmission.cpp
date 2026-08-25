// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Support/CodeConstants.h"
#include "ObjectEmission.h"

#include "PTO/Transforms/VPTOLLVMEmitter.h"
#include "VFSIMTSizePatcher.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <cctype>
#include <optional>
#include <string>

namespace {

using llvm::StringRef;

enum class BishengVFAutoSyncMode {
  Unspecified,
  Off,
  Fused,
  Global,
};

static llvm::cl::opt<bool> enableBishengVecMISched(
    "enable-bisheng-vec-misched",
    llvm::cl::desc("Use Bisheng's default vector MI scheduler behavior for "
                   "VPTO device compilation instead of explicitly disabling "
                   "the scheduler"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> enableSimtFastMath(
    "simt-fastmath",
    llvm::cl::desc("Enable Bisheng SIMT floating-point contraction and fast "
                   "math combining for VPTO device compilation"),
    llvm::cl::init(false));

static llvm::cl::opt<BishengVFAutoSyncMode> bishengVFAutoSyncMode(
    "bisheng-vf-auto-sync",
    llvm::cl::desc("Explicit Bisheng VF auto-sync mode for VPTO device "
                   "compilation; omit to use the toolchain default"),
    llvm::cl::value_desc("off|fused|global"),
    llvm::cl::values(clEnumValN(BishengVFAutoSyncMode::Off, "off",
                                "Disable Bisheng VF auto-sync"),
                     clEnumValN(BishengVFAutoSyncMode::Fused, "fused",
                                "Use Bisheng fused VF auto-sync"),
                     clEnumValN(BishengVFAutoSyncMode::Global, "global",
                                "Use Bisheng global VF auto-sync")),
    llvm::cl::init(BishengVFAutoSyncMode::Unspecified));

static bool runCommandWithStderr(llvm::StringRef program,
                                 llvm::ArrayRef<std::string> ownedArgs,
                                 llvm::StringRef stderrPath,
                                 llvm::raw_ostream &diagOS,
                                 llvm::StringRef what,
                                 std::optional<llvm::StringRef> stdinPath =
                                     std::nullopt);

static bool writeTextFile(StringRef path, StringRef content,
                          llvm::raw_ostream &diagOS) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    diagOS << "Error: failed to open " << path << " for write: "
           << ec.message() << "\n";
    return false;
  }
  os << content;
  os.flush();
  if (os.has_error()) {
    diagOS << "Error: failed to write to " << path << "\n";
    os.clear_error();
    return false;
  }
  return true;
}

static void stripUnsupportedBishengAttrs(llvm::Module &module) {
  for (llvm::Function &function : module) {
    // LLVM 19 prints memory effect attributes in textual form like
    // `memory(none)`. beta.1 Bisheng cannot parse that syntax, so remove only
    // the unsupported memory-effect attribute before serializing the module.
    function.setAttributes(
        function.getAttributes().removeFnAttribute(module.getContext(),
                                                   llvm::Attribute::Memory));
  }
}

static bool writeLLVMModuleFile(llvm::Module &module, StringRef path,
                                llvm::raw_ostream &diagOS) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    diagOS << "Error: failed to open " << path << " for write: "
           << ec.message() << "\n";
    return false;
  }
  stripUnsupportedBishengAttrs(module);
  module.print(os, nullptr);
  os.flush();
  if (os.has_error()) {
    diagOS << "Error: failed to write LLVM module to " << path << "\n";
    os.clear_error();
    return false;
  }
  return true;
}

static std::string sanitizeModuleId(llvm::StringRef raw) {
  std::string out;
  out.reserve(raw.size());
  for (char c : raw) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
      out.push_back(c);
    }
    else {
      out.push_back('_');
    }
  }
  if (out.empty()) {
    out = "ptoas_fatobj";
  }
  return out;
}

static std::optional<std::string> getAscendHomePath() {
  const char *env = std::getenv("ASCEND_HOME_PATH");
  if (!env || !*env) {
    return std::nullopt;
  }
  return std::string(env);
}

static std::optional<std::string> getEnvPath(llvm::StringRef name) {
  const char *env = std::getenv(name.str().c_str());
  if (!env || !*env) {
    return std::nullopt;
  }
  return std::string(env);
}

static std::string joinPath(llvm::StringRef lhs, llvm::StringRef rhs) {
  llvm::SmallString<mlir::pto::kValue256> joined(lhs);
  llvm::sys::path::append(joined, rhs);
  return std::string(joined.str());
}

static std::optional<std::string> parseCANNVersionInfo(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    return std::nullopt;
  }
  llvm::StringRef content = buffer.get()->getBuffer();
  llvm::SmallVector<llvm::StringRef, mlir::pto::kValue16> lines;
  content.split(lines, '\n');
  for (llvm::StringRef line : lines) {
    line = line.trim();
    llvm::StringRef keys[] = {"Version=", "version="};
    for (llvm::StringRef key : keys) {
      if (!line.starts_with(key)) {
        continue;
      }
      llvm::StringRef value = line.drop_front(key.size()).trim();
      if (value.consume_front("\"")) {
        value.consume_back("\"");
      }
      if (!value.empty()) {
        return value.str();
      }
    }
  }
  return std::nullopt;
}

static std::optional<std::string>
discoverCANNVersion(llvm::StringRef ascendHome) {
  for (llvm::StringRef relPath :
       {"compiler/version.info",
        "x86_64-linux/ascend_toolkit_install.info",
        "x86_64-linux/ascend_all_cann_install.info",
        "aarch64-linux/ascend_toolkit_install.info",
        "aarch64-linux/ascend_all_cann_install.info",
        "ascend_toolkit_install.info", "ascend_all_cann_install.info",
        "opp/version.info"}) {
    if (auto version = parseCANNVersionInfo(joinPath(ascendHome, relPath))) {
      return version;
    }
  }
  return std::nullopt;
}

static std::optional<std::string> locateProgram(llvm::StringRef envPath,
                                                llvm::StringRef fallbackName) {
  if (!envPath.empty() && llvm::sys::fs::exists(envPath)) {
    return envPath.str();
  }
  if (auto found = llvm::sys::findProgramByName(fallbackName)) {
    return *found;
  }
  return std::nullopt;
}

static bool hasPTOISAHeader(llvm::StringRef includeDir) {
  return llvm::sys::fs::exists(joinPath(includeDir, "pto/pto-inst.hpp"));
}

static void addExistingIncludeDir(llvm::SmallVectorImpl<std::string> &dirs,
                                  llvm::StringRef path) {
  if (path.empty() || !llvm::sys::fs::is_directory(path)) {
    return;
  }
  if (llvm::is_contained(dirs, path)) {
    return;
  }
  dirs.push_back(path.str());
}

static void addPTOISAIncludeDirs(llvm::SmallVectorImpl<std::string> &dirs,
                                 llvm::StringRef ptoIsaPath) {
  if (ptoIsaPath.empty() || !llvm::sys::fs::is_directory(ptoIsaPath)) {
    return;
  }
  std::string includeDir = joinPath(ptoIsaPath, "include");
  if (hasPTOISAHeader(includeDir)) {
    addExistingIncludeDir(dirs, includeDir);
  }
  std::string commonDir = joinPath(ptoIsaPath, "tests/common");
  addExistingIncludeDir(dirs, commonDir);
  if (hasPTOISAHeader(ptoIsaPath)) {
    addExistingIncludeDir(dirs, ptoIsaPath);
  }
}

static llvm::SmallVector<std::string, mlir::pto::kValue8>
discoverCppIncludeDirs(llvm::StringRef ascendHome,
                       llvm::raw_ostream &diagOS,
                       std::string &ptoIsaPath) {
  llvm::SmallVector<std::string, mlir::pto::kValue8> includeDirs;
  if (auto env = getEnvPath("PTO_ISA_PATH")) {
    ptoIsaPath = *env;
  } else if (auto env = getEnvPath("PTO_ISA_ROOT")) {
    ptoIsaPath = *env;
  }

  if (ptoIsaPath.empty()) {
    // Probe common development checkouts so the bridge wrapper and C++
    // device emission find pto/pto-inst.hpp without extra environment setup.
    if (auto home = llvm::sys::Process::GetEnv("HOME")) {
      std::string candidates[] = {
          joinPath(*home, "pto-isa"),
          joinPath(joinPath(*home, "llvm-workspace"), "pto-isa"),
      };
      for (const std::string &candidate : candidates) {
        if (llvm::sys::fs::is_directory(candidate)) {
          ptoIsaPath = candidate;
          break;
        }
      }
    }
  }

  addPTOISAIncludeDirs(includeDirs, ptoIsaPath);
  addExistingIncludeDir(includeDirs, joinPath(ascendHome, "include"));
  std::string driverPath =
      getEnvPath("ASCEND_DRIVER_PATH").value_or("/usr/local/Ascend/driver");
  addExistingIncludeDir(includeDirs, joinPath(driverPath, "kernel/inc"));

  if (ptoIsaPath.empty()) {
    diagOS << "Warning: PTO_ISA_PATH/PTO_ISA_ROOT is not set; "
              "C++ device object emission may fail to include pto/pto-inst.hpp.\n";
  } else if (includeDirs.empty()) {
    diagOS << "Warning: no PTO-ISA include directory containing "
              "pto/pto-inst.hpp was found under "
           << ptoIsaPath << ".\n";
  }
  return includeDirs;
}

// Merges externally compiled bitcode (e.g. PTO-ISA template instantiation
// bridge wrappers, the cce::printf print wrapper) into the VPTO device LLVM
// IR before Bisheng compiles the device object. All inputs are bitcode/IR of
// the same LLVM version, linked with the llvm-link shipped with the Bisheng
// toolchain.
static bool linkDeviceLLVMBitcode(llvm::StringRef llPath,
                                  llvm::ArrayRef<std::string> extraBitcodePaths,
                                  llvm::StringRef linkedPath,
                                  const mlir::pto::CANNToolchain &toolchain,
                                  llvm::StringRef stderrPath,
                                  llvm::raw_ostream &diagOS) {
  if (extraBitcodePaths.empty()) {
    return true;
  }
  std::string llvmLinkPath =
      joinPath(toolchain.bishengCompilerBinDirPath, "llvm-link");
  llvm::SmallVector<std::string, 8> args = {llvmLinkPath, llPath.str()};
  for (const std::string &path : extraBitcodePaths) {
    args.push_back(path);
  }
  args.push_back("-o");
  args.push_back(linkedPath.str());
  return runCommandWithStderr(llvmLinkPath, args, stderrPath, diagOS,
                              "device LLVM bridge link");
}

static bool compileDeviceLLVMToObject(llvm::StringRef llPath,
                                      llvm::StringRef outObjPath,
                                      llvm::StringRef targetCPU,
                                      llvm::StringRef bishengPath,
                                      llvm::StringRef stderrPath,
                                      llvm::raw_ostream &diagOS);
static bool compileHostStubToObject(llvm::StringRef stubPath,
                                    llvm::StringRef outObjPath,
                                    llvm::StringRef moduleId,
                                    llvm::StringRef targetCPU,
                                    const mlir::pto::CANNToolchain &toolchain,
                                    llvm::StringRef deviceObjPath,
                                    bool usesPrint,
                                    llvm::StringRef stderrPath,
                                    llvm::raw_ostream &diagOS);
static bool compileHostStubToObjectDriverMode(
    llvm::StringRef stubPath, llvm::StringRef outObjPath,
    llvm::StringRef moduleId, llvm::StringRef targetCPU,
    const mlir::pto::CANNToolchain &toolchain,
    llvm::StringRef deviceObjPath, llvm::StringRef stderrPath,
    llvm::raw_ostream &diagOS);
static bool mergeDeviceObjects(llvm::ArrayRef<std::string> deviceObjPaths,
                               llvm::StringRef outObjPath,
                               llvm::StringRef ldLldPath,
                               llvm::StringRef stderrPath,
                               llvm::raw_ostream &diagOS);

static llvm::StringRef
getTargetCPU(mlir::pto::ObjectEmissionDeviceTarget target) {
  switch (target) {
  case mlir::pto::ObjectEmissionDeviceTarget::Vector:
    return "dav-c310-vec";
  case mlir::pto::ObjectEmissionDeviceTarget::Cube:
    return "dav-c310-cube";
  }
  llvm_unreachable("unknown object emission device target");
}

static std::string resolveTargetCPU(llvm::Module &module,
                                    mlir::pto::ObjectEmissionDeviceTarget fallback) {
  for (llvm::Function &f : module) {
    if (f.hasFnAttribute("target-cpu")) {
      std::string cpu = f.getFnAttribute("target-cpu").getValueAsString().str();
      if (!cpu.empty()) {
        return cpu;
      }
    }
  }
  return getTargetCPU(fallback).str();
}

// Compiles the generated bridge wrapper C++ source to device bitcode for
// one core kind. The wrapper separates the core entries with __DAV_CUBE__ /
// __DAV_VEC__ guards, so compiling the same source once per target yields
// exactly the entries of that core. The command mirrors the former
// hand-written wrapper build (bisheng -O2 -std=c++17 -c -emit-llvm -xcce
// --cce-aicore-only -DREGISTER_BASE).
static bool compileBridgeWrapperToBitcode(
    llvm::StringRef wrapperSource,
    mlir::pto::ObjectEmissionDeviceTarget target,
    const mlir::pto::CANNToolchain &toolchain,
    mlir::pto::TempFileRegistry &tempFiles, llvm::StringRef stderrPath,
    std::string &outBitcodePath, llvm::raw_ostream &diagOS) {
  std::string sourcePath;
  if (failed(tempFiles.create("ptoas-vpto-bridge-wrapper", ".cpp", sourcePath,
                              diagOS)))
    return false;
  if (!writeTextFile(sourcePath, wrapperSource, diagOS))
    return false;
  if (failed(tempFiles.create("ptoas-vpto-bridge-wrapper", ".bc",
                              outBitcodePath, diagOS)))
    return false;
  llvm::SmallVector<std::string, 32> args = {
      toolchain.bishengPath,
      "-O2",
      "-std=c++17",
      "-c",
      "-emit-llvm",
      "-xcce",
      "--cce-aicore-only",
      std::string("--cce-aicore-arch=") + getTargetCPU(target).str(),
      "-DREGISTER_BASE",
  };
  for (const std::string &includeDir : toolchain.cppIncludeDirs) {
    args.push_back("-I" + includeDir);
  }
  args.push_back(sourcePath);
  args.push_back("-o");
  args.push_back(outBitcodePath);
  return runCommandWithStderr(toolchain.bishengPath, args, stderrPath, diagOS,
                              "bridge wrapper bitcode compilation");
}

// pto.print / pto.tprint lower to calls of external shims (pto_print_f32,
// pto_print_i64, ...) defined in pt_print.cpp.  The shims wrap cce::printf,
// which only bisheng can instantiate (template + DebugTunnel internals), so
// pt_print.cpp is compiled to a device bitcode and llvm-linked into the
// kernel modules before object emission.  Driver mode with -emit-llvm is
// used (not CC1): it resolves the CCE/CANN include chains and libstdc++
// paths the same way the host-stub driver mode does, and it must NOT pass
// -cce-enable-mix (which splits symbols into .vector/.cube variants).
static bool compilePrintWrapperToBitcode(
    const mlir::pto::CANNToolchain &toolchain, llvm::StringRef targetCPU,
    llvm::StringRef wrapperSourcePath, llvm::StringRef outBcPath,
    llvm::StringRef stderrPath, llvm::raw_ostream &diagOS) {
  llvm::SmallVector<std::string, 16> args = {
      toolchain.bishengPath,
      "-xcce",
      "--cce-aicore-only",
      std::string("--cce-aicore-arch=") + targetCPU.str(),
      "-D__CCE_ENABLE_PRINT_FOUND_CANN__",
      "--cce-enable-print",
      "-std=c++17",
      "-O2",
      "-c",
      "-emit-llvm",
      "-x",
      "cce",
      wrapperSourcePath.str(),
      "-o",
      outBcPath.str(),
  };
  return runCommandWithStderr(toolchain.bishengPath, args, stderrPath, diagOS,
                              "print wrapper bitcode compilation");
}

// Locate the shipped pt_print.cpp wrapper source.  It lives in a directory
// without a CMakeLists.txt (so the LLVM source-list check does not require
// it to belong to a target) and its path is baked in at configure time.
static std::string getPrintWrapperSourcePath(llvm::raw_ostream &diagOS) {
#ifdef PTOAS_DEFAULT_PRINT_WRAPPER_PATH
  std::string path = joinPath(PTOAS_DEFAULT_PRINT_WRAPPER_PATH, "pt_print.cpp");
  if (llvm::sys::fs::exists(path))
    return path;
  diagOS << "Error: print wrapper source not found at " << path << ".\n";
#else
  diagOS << "Error: PTOAS_DEFAULT_PRINT_WRAPPER_PATH is not defined; cannot "
            "locate the pt_print.cpp wrapper source.\n";
#endif
  return std::string();
}

static std::string getPrintHostWrapperSourcePath(llvm::raw_ostream &diagOS) {
#ifdef PTOAS_DEFAULT_PRINT_WRAPPER_PATH
  std::string path =
      joinPath(PTOAS_DEFAULT_PRINT_WRAPPER_PATH, "pt_print_host.cpp");
  if (llvm::sys::fs::exists(path))
    return path;
  diagOS << "Error: Print host wrapper source not found at " << path << ".\n";
#else
  diagOS << "Error: PTOAS_DEFAULT_PRINT_WRAPPER_PATH is not defined; cannot "
            "locate pt_print_host.cpp.\n";
#endif
  return std::string();
}

class VPTOFatobjArtifacts {
public:
  explicit VPTOFatobjArtifacts(mlir::pto::TempFileRegistry &tempFiles)
      : tempFiles(tempFiles) {}

  bool emitStubSource(StringRef stubSource, llvm::raw_ostream &diagOS) {
    if (failed(tempFiles.create("ptoas-host-stub", ".cpp", stubPath, diagOS))) {
      return false;
    }
    if (!writeTextFile(stubPath, stubSource, diagOS)) {
      return false;
    }
    return true;
  }

  bool initCommandLogs(llvm::raw_ostream &diagOS) {
    if (failed(tempFiles.create("ptoas-stderr", ".log", stderrPath, diagOS))) {
      return false;
    }
    return true;
  }

  bool emitCubeObject(llvm::Module *module,
                      const mlir::pto::CANNToolchain &toolchain,
                      llvm::StringRef bridgeBitcodePath, bool usesPrint,
                      llvm::raw_ostream &diagOS) {
    if (!module) {
      return true;
    }
    if (failed(tempFiles.create("ptoas-device", ".ll", cubeLLPath, diagOS))) {
      return false;
    }
    if (failed(tempFiles.create("ptoas-device", ".o", cubeObjPath, diagOS))) {
      return false;
    }
    return succeeded(mlir::pto::emitVPTOCubeDeviceObject(
        *module, cubeLLPath, cubeObjPath, toolchain, bridgeBitcodePath,
        usesPrint, stderrPath, diagOS));
  }

  bool emitVectorObject(llvm::Module *module,
                        const mlir::pto::CANNToolchain &toolchain,
                        llvm::StringRef bridgeBitcodePath,
                        mlir::pto::VFSIMTSizeFixMode vfsimtSizeFixMode,
                        bool usesPrint, llvm::raw_ostream &diagOS) {
    if (!module) {
      return true;
    }
    if (failed(tempFiles.create("ptoas-device", ".ll", vectorLLPath, diagOS))) {
      return false;
    }
    std::string rawVectorObjPath;
    if (failed(tempFiles.create("ptoas-device-vector-raw", ".o",
                                rawVectorObjPath, diagOS))) {
      return false;
    }
    if (failed(mlir::pto::emitVPTOVectorDeviceObject(
            *module, vectorLLPath, rawVectorObjPath, toolchain,
            bridgeBitcodePath, usesPrint, stderrPath, diagOS))) {
      return false;
    }
    if (vfsimtSizeFixMode == mlir::pto::VFSIMTSizeFixMode::Off) {
      vectorObjPath = std::move(rawVectorObjPath);
      return true;
    }

    std::string patchedVectorObjPath;
    if (failed(tempFiles.create("ptoas-device-vector-patched", ".o",
                                patchedVectorObjPath, diagOS))) {
      return false;
    }
    mlir::FailureOr<mlir::pto::VFSIMTSizePatchResult> result =
        mlir::pto::verifyAndPatchVFSIMTSize(
            *module, rawVectorObjPath, patchedVectorObjPath,
            vfsimtSizeFixMode, diagOS);
    if (failed(result)) {
      return false;
    }
    vectorObjPath = std::move(result->objectPath);
    return true;
  }

  bool compileBridgeWrapper(llvm::StringRef wrapperSource,
                            mlir::pto::ObjectEmissionDeviceTarget target,
                            const mlir::pto::CANNToolchain &toolchain,
                            std::string &outBitcodePath,
                            llvm::raw_ostream &diagOS) {
    if (wrapperSource.empty()) {
      return true;
    }
    return compileBridgeWrapperToBitcode(wrapperSource, target, toolchain,
                                         tempFiles, stderrPath,
                                         outBitcodePath, diagOS);
  }

  bool mergeDeviceObjects(const mlir::pto::CANNToolchain &toolchain,
                          llvm::raw_ostream &diagOS) {
    llvm::SmallVector<std::string, mlir::pto::kValue2> deviceObjPaths;
    if (!cubeObjPath.empty()) {
      deviceObjPaths.push_back(cubeObjPath);
    }
    if (!vectorObjPath.empty()) {
      deviceObjPaths.push_back(vectorObjPath);
    }
    if (deviceObjPaths.empty()) {
      diagOS << "Error: VPTO fatobj emission requires at least one device module.\n";
      return false;
    }
    if (failed(tempFiles.create("ptoas-device-merged", ".o",
                                mergedDeviceObjPath, diagOS))) {
      return false;
    }
    return ::mergeDeviceObjects(deviceObjPaths, mergedDeviceObjPath,
                                toolchain.ldLldPath, stderrPath, diagOS);
  }

  bool compileHostStub(const mlir::pto::CANNToolchain &toolchain,
                       llvm::StringRef moduleId,
                       llvm::StringRef targetCPU, bool usesPrint,
                       llvm::raw_ostream &diagOS) {
    if (failed(tempFiles.create("ptoas-host-stub", ".o", hostStubObjPath,
                                diagOS))) {
      return false;
    }
    return compileHostStubToObject(stubPath, hostStubObjPath, moduleId,
                                   targetCPU, toolchain, mergedDeviceObjPath,
                                   usesPrint, stderrPath, diagOS);
  }

  bool compileHostStubToFatobj(const mlir::pto::CANNToolchain &toolchain,
                               llvm::StringRef moduleId,
                               llvm::StringRef targetCPU,
                               llvm::StringRef outputPath, bool usesPrint,
                               llvm::raw_ostream &diagOS) {
    if (!usesPrint) {
      return compileHostStubToObject(stubPath, outputPath, moduleId, targetCPU,
                                     toolchain, mergedDeviceObjPath, usesPrint,
                                     stderrPath, diagOS);
    }

    // Print path: compile the stub in driver mode (DebugTunnel support),
    // then fold in pt_print_host.cpp which overrides __DebugTunnel_Close so
    // the print buffer is flushed after aclrtSynchronizeStream.
    std::string baseFatobjPath;
    std::string closeOverrideObjPath;
    if (failed(tempFiles.create("ptoas-print-base", ".o", baseFatobjPath,
                                diagOS)) ||
        failed(tempFiles.create("ptoas-print-host", ".o",
                                closeOverrideObjPath, diagOS)))
      return false;
    if (!compileHostStubToObject(stubPath, baseFatobjPath, moduleId, targetCPU,
                                 toolchain, mergedDeviceObjPath,
                                 /*usesPrint=*/true, stderrPath, diagOS))
      return false;

    std::string sourcePath = getPrintHostWrapperSourcePath(diagOS);
    if (sourcePath.empty())
      return false;
    llvm::SmallVector<std::string, 16> compileArgs = {
        toolchain.bishengPath, "-xc++", "-std=c++17", "-fPIC", "-I",
        toolchain.ascendHomePath + "/include", "-c", sourcePath, "-o",
        closeOverrideObjPath};
    if (!runCommandWithStderr(toolchain.bishengPath, compileArgs, stderrPath,
                              diagOS, "print host close override compilation"))
      return false;

    llvm::SmallVector<std::string, 8> linkArgs = {
        toolchain.ldLldPath, "-r", baseFatobjPath, closeOverrideObjPath,
        "-o", outputPath.str()};
    return runCommandWithStderr(toolchain.ldLldPath, linkArgs, stderrPath,
                                diagOS, "print fatobj host override link");
  }

  bool repackFatObj(const mlir::pto::CANNToolchain &toolchain,
                    llvm::StringRef moduleId, llvm::StringRef targetCPU,
                    llvm::StringRef outPath, llvm::raw_ostream &diagOS) {
    llvm::SmallVector<std::string, mlir::pto::kValue16> args = {
        toolchain.cceLdPath,
        toolchain.ldLldPath,
        "-x",
        "-cce-lite-bin-module-id",
        moduleId.str(),
        std::string("-cce-aicore-arch=") + targetCPU.str(),
        // Keep device externals relocatable until the final
        // --cce-fatobj-link link step.
        "-dc",
        "-r",
        "-o",
        outPath.str(),
        "-cce-stub-dir",
        toolchain.cceStubDirPath,
        "-cce-install-dir",
        toolchain.bishengCompilerBinDirPath,
        "-cce-inputs-number",
        "1",
        hostStubObjPath,
    };
    return runCommandWithStderr(toolchain.cceLdPath, args, stderrPath, diagOS,
                                "fatobj repack");
  }

private:
  mlir::pto::TempFileRegistry &tempFiles;
  std::string cubeLLPath;
  std::string cubeObjPath;
  std::string vectorLLPath;
  std::string vectorObjPath;
  std::string mergedDeviceObjPath;
  std::string stderrPath;
  std::string stubPath;
  std::string hostStubObjPath;
};

static bool runCommandWithStderr(llvm::StringRef program,
                                 llvm::ArrayRef<std::string> ownedArgs,
                                 llvm::StringRef stderrPath,
                                 llvm::raw_ostream &diagOS,
                                 llvm::StringRef what,
                                 std::optional<llvm::StringRef> stdinPath) {
  llvm::SmallVector<llvm::StringRef, mlir::pto::kValue16> args;
  args.reserve(ownedArgs.size());
  for (const std::string &arg : ownedArgs) {
    args.push_back(arg);
  }
  llvm::SmallVector<std::optional<llvm::StringRef>, mlir::pto::kValue3> redirects = {
      stdinPath, stderrPath, stderrPath};

  std::string execErr;
  bool execFailed = false;
  int rc = llvm::sys::ExecuteAndWait(program, args, std::nullopt, redirects, 0,
                                     0, &execErr, &execFailed);
  if (!execFailed && rc == 0) {
    return true;
  }

  diagOS << "Error: " << what << " failed\n";
  diagOS << "Command:";
  for (llvm::StringRef arg : args) {
    diagOS << " " << arg;
  }
  diagOS << "\n";
  if (!execErr.empty()) {
    diagOS << execErr << "\n";
  }
  if (auto buffer = llvm::MemoryBuffer::getFile(stderrPath)) {
    diagOS << buffer.get()->getBuffer() << "\n";
  }
  return false;
}

static bool compileDeviceLLVMToObject(llvm::StringRef llPath,
                                      llvm::StringRef outObjPath,
                                      llvm::StringRef targetCPU,
                                      llvm::StringRef bishengPath,
                                      llvm::StringRef stderrPath,
                                      llvm::raw_ostream &diagOS) {
  llvm::SmallVector<std::string, mlir::pto::kValue24> args = {
      bishengPath.str(),
      std::string("--cce-aicore-arch=") + targetCPU.str(),
      "--cce-aicore-only",
      "-O2",
      "--cce-generic-addrspace=off",
      "-cce-bitcode-is-aicore",
      "-Wno-override-module",
      "-dc",
      "--cce-long-scbz=true",
      "-mllvm",
      "-cce-dyn-kernel-stack-size=true",
  };
  switch (bishengVFAutoSyncMode) {
  case BishengVFAutoSyncMode::Unspecified:
    break;
  case BishengVFAutoSyncMode::Off:
    args.push_back("-mllvm");
    args.push_back("-cce-vf-auto-sync=off");
    break;
  case BishengVFAutoSyncMode::Fused:
    args.push_back("-mllvm");
    args.push_back("-cce-vf-auto-sync=fused");
    break;
  case BishengVFAutoSyncMode::Global:
    args.push_back("-mllvm");
    args.push_back("-cce-vf-auto-sync=global");
    break;
  }
  // Enabling vector MI scheduling deliberately omits this argument instead of
  // passing `=1`, so Bisheng retains the default behavior of the selected
  // toolchain version.
  if (!enableBishengVecMISched) {
    args.push_back("-mllvm");
    args.push_back("--cce-aicore-vec-misched=0");
  }
  args.push_back("-mllvm");
  args.push_back(std::string("--cce-simt-fpmath-combine=") +
                 (enableSimtFastMath ? "true" : "false"));
  args.push_back("-c");
  args.push_back("-x");
  args.push_back("ir");
  args.push_back("-");
  args.push_back("-o");
  args.push_back(outObjPath.str());
  return runCommandWithStderr(bishengPath, args, stderrPath, diagOS,
                              "device LLVM compilation", llPath);
}

static bool compileCppDeviceSourceToObject(
    llvm::StringRef cppPath, llvm::StringRef outObjPath,
    llvm::StringRef targetCPU, const mlir::pto::CANNToolchain &toolchain,
    llvm::StringRef stderrPath, llvm::raw_ostream &diagOS) {
  llvm::SmallVector<std::string, mlir::pto::kValue32> args = {
      toolchain.bishengPath,
      "-xcce",
      "-fenable-matrix",
      "--cce-aicore-enable-tl",
      "--cce-aicore-only",
      "-fPIC",
      "-Xhost-start",
      "-Xhost-end",
      "-mllvm",
      "-cce-aicore-stack-size=0x8000",
      "-mllvm",
      "-cce-aicore-function-stack-size=0x8000",
      "-mllvm",
      "-cce-aicore-record-overflow=true",
      "-mllvm",
      "-cce-aicore-addr-transform",
      "-mllvm",
      "-cce-aicore-dcci-insert-for-scalar=false",
      std::string("--cce-aicore-arch=") + targetCPU.str(),
      "-DREGISTER_BASE",
      "-std=c++17",
      "-dc",
  };
  for (const std::string &includeDir : toolchain.cppIncludeDirs) {
    args.push_back("-I" + includeDir);
  }
  args.push_back("-c");
  args.push_back(cppPath.str());
  args.push_back("-o");
  args.push_back(outObjPath.str());

  return runCommandWithStderr(toolchain.bishengPath, args, stderrPath, diagOS,
                              "C++ device compilation");
}

static bool compileCppDeviceSourceToFatobj(
    llvm::StringRef cppPath, llvm::StringRef outObjPath,
    const mlir::pto::CANNToolchain &toolchain,
    llvm::StringRef stderrPath, llvm::raw_ostream &diagOS) {
  llvm::SmallVector<std::string, mlir::pto::kValue32> args = {
      toolchain.bishengPath,
      "-xcce",
      "-fenable-matrix",
      "--cce-aicore-enable-tl",
      "-fPIC",
      "-Xhost-start",
      "-Xhost-end",
      "-mllvm",
      "-cce-aicore-stack-size=0x8000",
      "-mllvm",
      "-cce-aicore-function-stack-size=0x8000",
      "-mllvm",
      "-cce-aicore-record-overflow=true",
      "-mllvm",
      "-cce-aicore-addr-transform",
      "-mllvm",
      "-cce-aicore-dcci-insert-for-scalar=false",
      "--cce-aicore-arch=dav-c310",
      "-DREGISTER_BASE",
      "-std=c++17",
      "-O2",
      "-dc",
      "-c",
  };
  for (const std::string &includeDir : toolchain.cppIncludeDirs) {
    args.push_back("-I" + includeDir);
  }
  args.push_back(cppPath.str());
  args.push_back("-o");
  args.push_back(outObjPath.str());

  return runCommandWithStderr(toolchain.bishengPath, args, stderrPath, diagOS,
                              "C++ fatobj compilation");
}

static std::string resolveHostTargetCPU() {
  if (const char *envCPU = std::getenv("PTOAS_HOST_TARGET_CPU")) {
    if (envCPU[0] != '\0') {
      return std::string(envCPU);
    }
  }
  std::string hostCPU = llvm::sys::getHostCPUName().str();
  if (hostCPU == "cortex-x925") {
    return "tsv200m";
  }
  if (hostCPU == "znver4" || hostCPU == "znver5") {
    return "znver3";
  }
  return hostCPU;
}

static bool compileHostStubToObject(llvm::StringRef stubPath,
                                    llvm::StringRef outObjPath,
                                    llvm::StringRef moduleId,
                                    llvm::StringRef targetCPU,
                                    const mlir::pto::CANNToolchain &toolchain,
                                    llvm::StringRef deviceObjPath,
                                    bool usesPrint,
                                    llvm::StringRef stderrPath,
                                    llvm::raw_ostream &diagOS) {
  // Print support needs the DebugTunnel infrastructure, whose headers pull
  // in libstdc++ pieces the CC1 invocation below cannot resolve; route to
  // the bisheng driver instead.
  if (usesPrint) {
    return compileHostStubToObjectDriverMode(stubPath, outObjPath, moduleId,
                                             targetCPU, toolchain,
                                             deviceObjPath, stderrPath,
                                             diagOS);
  }
  std::string coverageDir = ".";
  std::string debugDir = ".";
  std::string hostTriple = llvm::sys::getProcessTriple();
  std::string hostTargetCPU = resolveHostTargetCPU();

  llvm::SmallVector<std::string, mlir::pto::kValue32> args = {
      toolchain.bishengCc1Path,
      "-cc1",
      "-triple",
      hostTriple,
      "-target-cpu",
      hostTargetCPU,
      "-fcce-aicpu-legacy-launch",
      "-fcce-is-host",
      "-cce-enable-mix",
      "-mllvm",
      "-enable-mix=true",
      "-cce-launch-with-flagv2-impl",
      "-fcce-aicore-arch",
      targetCPU.str(),
      "-fcce-fatobj-compile",
      "-emit-obj",
      "--mrelax-relocations",
      "-disable-free",
      "-clear-ast-before-backend",
      "-disable-llvm-verifier",
      "-discard-value-names",
      "-main-file-name",
      "stub.cpp",
      "-mrelocation-model",
      "pic",
      "-pic-level",
      "2",
      "-fhalf-no-semantic-interposition",
      "-mframe-pointer=none",
      "-fmath-errno",
      "-ffp-contract=on",
      "-fno-rounding-math",
      "-mconstructor-aliases",
      "-funwind-tables=2",
      "-fallow-half-arguments-and-returns",
      "-mllvm",
      "-treat-scalable-fixed-error-as-warning",
      std::string("-fcoverage-compilation-dir=") + coverageDir,
      "-resource-dir",
      toolchain.resourceDirPath,
      "-internal-isystem",
      toolchain.resourceIncludeDirPath,
      "-include",
      "__clang_cce_runtime_wrapper.h",
      "-D",
      "_FORTIFY_SOURCE=2",
      "-D",
      "REGISTER_BASE",
      "-O2",
      "-Wno-macro-redefined",
      "-Wno-ignored-attributes",
      "-std=c++17",
      "-fdeprecated-macro",
      std::string("-fdebug-compilation-dir=") + debugDir,
      "-ferror-limit",
      "19",
      "-stack-protector",
      "2",
      "-fno-signed-char",
      "-fgnuc-version=4.2.1",
      "-fcxx-exceptions",
      "-fexceptions",
      "-vectorize-loops",
      "-vectorize-slp",
      "-mllvm",
      "-cce-aicore-stack-size=0x8000",
      "-mllvm",
      "-cce-aicore-function-stack-size=0x8000",
      "-mllvm",
      "-cce-aicore-record-overflow=true",
      "-mllvm",
      "-cce-aicore-addr-transform",
      "-mllvm",
      "-cce-aicore-dcci-insert-for-scalar=false",
      "-fcce-include-aibinary",
      deviceObjPath.str(),
      "-fcce-device-module-id",
      moduleId.str(),
      "-faddrsig",
      "-D__GCC_HAVE_DWARF2_CFI_ASM=1",
      "-o",
      outObjPath.str(),
      "-x",
      "cce",
      stubPath.str(),
  };
  return runCommandWithStderr(toolchain.bishengCc1Path, args, stderrPath, diagOS,
                              "host stub compilation");
}

// Compile a host stub in bisheng driver mode (-xcce).  This is used when the
// module contains pto.print / pto.tprint ops and the host stub needs the
// DebugTunnel print infrastructure, which pulls in C++ standard library
// headers that CC1 mode cannot handle.
static bool compileHostStubToObjectDriverMode(
    llvm::StringRef stubPath, llvm::StringRef outObjPath,
    llvm::StringRef moduleId, llvm::StringRef targetCPU,
    const mlir::pto::CANNToolchain &toolchain,
    llvm::StringRef deviceObjPath, llvm::StringRef stderrPath,
    llvm::raw_ostream &diagOS) {
  llvm::SmallVector<std::string, 32> args = {
      toolchain.bishengPath,
      "-xcce",
      "--cce-enable-print",
      "-cce-enable-mix",
      "-cce-launch-with-flagv2-impl",
      std::string("--cce-aicore-arch=") + targetCPU.str(),
      "-DREGISTER_BASE",
      "-D__CCE_ENABLE_PRINT_FOUND_CANN__",
      "-std=c++17",
      "-fPIC",
      "-O2",
      "-Wno-macro-redefined",
      "-Wno-ignored-attributes",
  };

  // CANN runtime include paths needed by the DebugTunnel print
  // infrastructure (host-side headers for print buffer management).
  std::string ascendHome;
  if (const char *env = ::getenv("ASCEND_HOME_PATH"))
    ascendHome = env;
  else if (const char *env2 = ::getenv("ASCEND_HOME"))
    ascendHome = env2;
  if (!ascendHome.empty()) {
    args.push_back("-I");
    args.push_back(ascendHome + "/include");
    args.push_back("-I");
    args.push_back(ascendHome + "/include/experiment");
    args.push_back("-I");
    args.push_back(ascendHome + "/pkg_inc/");
    args.push_back("-I");
    args.push_back(ascendHome + "/pkg_inc/runtime/runtime/");
    args.push_back("-I");
    args.push_back(ascendHome + "/pkg_inc/profiling/");
  }

  // Route device-object embedding and module-id through -Xclang so the
  // driver forwards them to the CC1 frontend.
  args.push_back("-Xclang");
  args.push_back("-fcce-include-aibinary");
  args.push_back("-Xclang");
  args.push_back(deviceObjPath.str());
  args.push_back("-Xclang");
  args.push_back("-fcce-device-module-id");
  args.push_back("-Xclang");
  args.push_back(moduleId.str());

  args.push_back("-c");
  args.push_back(stubPath.str());
  args.push_back("-o");
  args.push_back(outObjPath.str());

  return runCommandWithStderr(toolchain.bishengPath, args, stderrPath, diagOS,
                              "host stub compilation (driver mode)");
}

static bool mergeDeviceObjects(llvm::ArrayRef<std::string> deviceObjPaths,
                               llvm::StringRef outObjPath,
                               llvm::StringRef ldLldPath,
                               llvm::StringRef stderrPath,
                               llvm::raw_ostream &diagOS) {
  if (deviceObjPaths.empty()) {
    return false;
  }

  llvm::SmallVector<std::string, mlir::pto::kValue16> args = {
      ldLldPath.str(),
      "-m",
      "aicorelinux",
      "-Ttext",
      "0",
  };
  for (const std::string &path : deviceObjPaths) {
    args.push_back(path);
  }
  args.push_back("-o");
  args.push_back(outObjPath.str());
  args.push_back("-r");
  args.push_back("--allow-multiple-definition");
  return runCommandWithStderr(ldLldPath, args, stderrPath, diagOS,
                              "device object merge");
}

static bool linkFatobjFiles(llvm::ArrayRef<std::string> fatobjPaths,
                            llvm::StringRef outObjPath,
                            const mlir::pto::CANNToolchain &toolchain,
                            llvm::StringRef stderrPath,
                            llvm::raw_ostream &diagOS) {
  if (fatobjPaths.empty()) {
    return false;
  }

  llvm::SmallVector<std::string, mlir::pto::kValue32> args = {
      toolchain.bishengPath,
      "--cce-fatobj-link",
      "--cce-aicore-arch=dav-c310",
      "-r",
      "-o",
      outObjPath.str(),
  };
  for (const std::string &path : fatobjPaths) {
    args.push_back(path);
  }

  return runCommandWithStderr(toolchain.bishengPath, args, stderrPath, diagOS,
                              "fatobj link");
}

} // namespace

mlir::pto::TempFileRegistry::~TempFileRegistry() { cleanup(); }

void mlir::pto::TempFileRegistry::cleanup() {
  for (const std::string &path : paths) {
    llvm::sys::fs::remove(path);
  }
  paths.clear();
}

mlir::LogicalResult
mlir::pto::TempFileRegistry::create(llvm::StringRef prefix,
                                    llvm::StringRef suffix, std::string &path,
                                    llvm::raw_ostream &diagOS) {
  llvm::SmallString<mlir::pto::kValue128> tempPath;
  int fd = -1;
  std::error_code ec =
      llvm::sys::fs::createTemporaryFile(prefix, suffix, fd, tempPath);
  if (ec) {
    diagOS << "Error: failed to create temporary file for " << prefix << suffix
           << ": " << ec.message() << "\n";
    return failure();
  }
  llvm::sys::Process::SafelyCloseFileDescriptor(fd);
  path = tempPath.str().str();
  paths.push_back(path);
  return success();
}

std::optional<mlir::pto::CANNToolchain>
mlir::pto::CANNToolchain::create(llvm::raw_ostream &diagOS) {
  std::optional<std::string> ascendHome = getAscendHomePath();
  if (!ascendHome) {
    diagOS << "Error: ASCEND_HOME_PATH is required for VPTO fatobj emission.\n";
    return std::nullopt;
  }

  CANNToolchain toolchain;
  toolchain.ascendHomePath = *ascendHome;
  toolchain.bishengPath = joinPath(toolchain.ascendHomePath, "bin/bisheng");
  toolchain.bishengCc1Path =
      joinPath(toolchain.ascendHomePath, "tools/bisheng_compiler/bin/bisheng");
  toolchain.cceLdPath = joinPath(toolchain.ascendHomePath, "bin/cce-ld");
  toolchain.ldLldPath =
      locateProgram(joinPath(toolchain.ascendHomePath, "bin/ld.lld"), "ld.lld")
          .value_or(std::string());
  toolchain.resourceDirPath = joinPath(
      toolchain.ascendHomePath, "tools/bisheng_compiler/lib/clang/15.0.5");
  toolchain.resourceIncludeDirPath =
      joinPath(toolchain.resourceDirPath, "include");
  toolchain.cceStubDirPath =
      joinPath(toolchain.resourceIncludeDirPath, "cce_stub");
  toolchain.bishengCompilerBinDirPath =
      joinPath(toolchain.ascendHomePath, "tools/bisheng_compiler/bin");
  toolchain.cannVersionString =
      discoverCANNVersion(toolchain.ascendHomePath).value_or("9.0.0-beta.1");
  llvm::SmallVector<std::string, mlir::pto::kValue8> cppIncludeDirs = discoverCppIncludeDirs(
      toolchain.ascendHomePath, diagOS, toolchain.ptoIsaPath);
  toolchain.cppIncludeDirs.assign(cppIncludeDirs.begin(),
                                  cppIncludeDirs.end());
  if (failed(toolchain.validate(diagOS))) {
    return std::nullopt;
  }
  return toolchain;
}

mlir::LogicalResult
mlir::pto::CANNToolchain::validate(llvm::raw_ostream &diagOS) const {
  if (!llvm::sys::fs::exists(bishengPath)) {
    diagOS << "Error: unable to locate bisheng: " << bishengPath << "\n";
    return failure();
  }
  if (!llvm::sys::fs::exists(bishengCc1Path)) {
    diagOS << "Error: unable to locate bisheng cc1 frontend: "
           << bishengCc1Path << "\n";
    return failure();
  }
  if (!llvm::sys::fs::exists(cceLdPath)) {
    diagOS << "Error: unable to locate cce-ld: " << cceLdPath << "\n";
    return failure();
  }
  if (ldLldPath.empty() || !llvm::sys::fs::exists(ldLldPath)) {
    diagOS << "Error: unable to locate ld.lld.\n";
    return failure();
  }
  return success();
}

llvm::StringRef mlir::pto::CANNToolchain::vptoPublicABISuffix(
    ObjectEmissionDeviceTarget target) const {
  const bool usesNewABI = cannVersion >= kCANN900Beta2Version;
  switch (target) {
  case ObjectEmissionDeviceTarget::Vector:
    return usesNewABI ? llvm::StringRef(".vector") : llvm::StringRef("_mix_aiv");
  case ObjectEmissionDeviceTarget::Cube:
    return usesNewABI ? llvm::StringRef(".cube") : llvm::StringRef("_mix_aic");
  }
  llvm_unreachable("unknown object emission device target");
}

mlir::LogicalResult mlir::pto::writeLLVMModule(llvm::Module &module,
                                               llvm::StringRef path,
                                               llvm::raw_ostream &diagOS) {
  return writeLLVMModuleFile(module, path, diagOS) ? success() : failure();
}

mlir::LogicalResult mlir::pto::writeCppSource(llvm::StringRef cppSource,
                                              llvm::StringRef path,
                                              llvm::raw_ostream &diagOS) {
  return writeTextFile(path, cppSource, diagOS) ? success() : failure();
}

mlir::LogicalResult mlir::pto::writeHostStubSource(
    llvm::StringRef stubSource, llvm::StringRef path,
    llvm::raw_ostream &diagOS) {
  return writeTextFile(path, stubSource, diagOS) ? success() : failure();
}

mlir::LogicalResult mlir::pto::compileCppToDeviceObject(
    llvm::StringRef cppPath, llvm::StringRef outObjPath,
    ObjectEmissionDeviceTarget target, const CANNToolchain &toolchain,
    llvm::StringRef stderrPath, llvm::raw_ostream &diagOS) {
  return compileCppDeviceSourceToObject(cppPath, outObjPath,
                                        getTargetCPU(target), toolchain,
                                        stderrPath, diagOS)
             ? success()
             : failure();
}

mlir::LogicalResult mlir::pto::compileLLVMToDeviceObject(
    llvm::StringRef llPath, llvm::StringRef outObjPath,
    ObjectEmissionDeviceTarget target, const CANNToolchain &toolchain,
    llvm::StringRef stderrPath, llvm::raw_ostream &diagOS) {
  return compileDeviceLLVMToObject(llPath, outObjPath, getTargetCPU(target),
                                   toolchain.bishengPath, stderrPath, diagOS)
             ? success()
             : failure();
}

mlir::LogicalResult mlir::pto::emitCppVectorDeviceObject(
    llvm::StringRef cppSource, llvm::StringRef cppPath,
    llvm::StringRef outObjPath, const CANNToolchain &toolchain,
    llvm::StringRef stderrPath, llvm::raw_ostream &diagOS) {
  if (failed(writeCppSource(cppSource, cppPath, diagOS))) {
    return failure();
  }
  return compileCppToDeviceObject(cppPath, outObjPath,
                                  ObjectEmissionDeviceTarget::Vector,
                                  toolchain, stderrPath, diagOS);
}

mlir::LogicalResult mlir::pto::emitCppCubeDeviceObject(
    llvm::StringRef cppSource, llvm::StringRef cppPath,
    llvm::StringRef outObjPath, const CANNToolchain &toolchain,
    llvm::StringRef stderrPath, llvm::raw_ostream &diagOS) {
  if (failed(writeCppSource(cppSource, cppPath, diagOS))) {
    return failure();
  }
  return compileCppToDeviceObject(cppPath, outObjPath,
                                  ObjectEmissionDeviceTarget::Cube,
                                  toolchain, stderrPath, diagOS);
}

mlir::LogicalResult mlir::pto::emitCppFatobj(
    llvm::StringRef cppSource, llvm::StringRef cppPath,
    llvm::StringRef outObjPath, const CANNToolchain &toolchain,
    llvm::StringRef stderrPath, llvm::raw_ostream &diagOS) {
  if (failed(writeCppSource(cppSource, cppPath, diagOS))) {
    return failure();
  }
  return compileCppDeviceSourceToFatobj(cppPath, outObjPath, toolchain,
                                        stderrPath, diagOS)
             ? success()
             : failure();
}

mlir::LogicalResult mlir::pto::emitFatobjCCE(
    llvm::StringRef cppSource, llvm::StringRef outputPath,
    const CANNToolchain &toolchain, TempFileRegistry &tempFiles,
    llvm::raw_ostream &diagOS) {
  std::string cppPath;
  std::string stderrPath;
  if (failed(tempFiles.create("ptoas-emitc", ".cpp", cppPath, diagOS)) ||
      failed(tempFiles.create("ptoas-emitc-fatobj", ".log", stderrPath,
                              diagOS))) {
    return failure();
  }
  return emitCppFatobj(cppSource, cppPath, outputPath, toolchain, stderrPath,
                       diagOS);
}

static bool isVPTOKernelABISymbol(llvm::StringRef name) {
  return name.ends_with("_mix_aiv") || name.ends_with("_mix_aic");
}

static bool isLegacyVPTOPublicABISymbol(llvm::StringRef name) {
  return name.ends_with(".vector") || name.ends_with(".cube");
}

static mlir::LogicalResult renameLLVMFunction(llvm::Module &module,
                                              llvm::StringRef sourceName,
                                              llvm::StringRef abiName,
                                              llvm::raw_ostream &diagOS) {
  if (sourceName == abiName) {
    return mlir::success();
  }
  llvm::Function *function = module.getFunction(sourceName);
  if (!function) {
    return mlir::success();
  }
  if (llvm::Function *existing = module.getFunction(abiName);
      existing && existing != function) {
    diagOS << "Error: cannot rename LLVM symbol '" << sourceName << "' to '"
           << abiName << "': target symbol already exists.\n";
    return mlir::failure();
  }
  function->setName(abiName);
  return mlir::success();
}

static mlir::LogicalResult applyVPTOLLVMABINames(llvm::Module &module,
                                                 llvm::StringRef suffix,
                                                 llvm::raw_ostream &diagOS) {
  for (llvm::Function &function : module) {
    if (function.isDeclaration() || !function.hasExternalLinkage()) {
      continue;
    }
    llvm::StringRef name = function.getName();
    if (name.empty() || isVPTOKernelABISymbol(name) ||
        isLegacyVPTOPublicABISymbol(name)) {
      continue;
    }
    if (failed(renameLLVMFunction(module, name, (name + suffix).str(), diagOS))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::emitVPTOVectorDeviceObject(
    llvm::Module &module, llvm::StringRef llPath, llvm::StringRef outObjPath,
    const CANNToolchain &toolchain, llvm::StringRef bridgeBitcodePath,
    bool usesPrint, llvm::StringRef stderrPath, llvm::raw_ostream &diagOS) {
  if (failed(applyVPTOLLVMABINames(
          module,
          toolchain.vptoPublicABISuffix(ObjectEmissionDeviceTarget::Vector),
          diagOS))) {
    return failure();
  }
  if (failed(writeLLVMModule(module, llPath, diagOS))) {
    return failure();
  }
  std::string compileInput = llPath.str();
  llvm::SmallVector<std::string, mlir::pto::kValue2> extraBitcodePaths;
  if (!bridgeBitcodePath.empty()) {
    extraBitcodePaths.push_back(bridgeBitcodePath.str());
  }
  if (usesPrint) {
    std::string wrapperSourcePath = getPrintWrapperSourcePath(diagOS);
    if (wrapperSourcePath.empty())
      return failure();
    std::string printBitcodePath = (outObjPath + ".print.bc").str();
    if (!compilePrintWrapperToBitcode(
            toolchain,
            resolveTargetCPU(module, ObjectEmissionDeviceTarget::Vector),
            wrapperSourcePath, printBitcodePath, stderrPath, diagOS))
      return failure();
    extraBitcodePaths.push_back(printBitcodePath);
  }
  if (!extraBitcodePaths.empty()) {
    std::string linkedPath = (outObjPath + ".linked.bc").str();
    if (!linkDeviceLLVMBitcode(llPath, extraBitcodePaths, linkedPath,
                               toolchain, stderrPath, diagOS))
      return failure();
    compileInput = linkedPath;
  }
  return compileDeviceLLVMToObject(
             compileInput, outObjPath,
             resolveTargetCPU(module, ObjectEmissionDeviceTarget::Vector),
             toolchain.bishengPath, stderrPath, diagOS)
             ? success()
             : failure();
}

mlir::LogicalResult mlir::pto::emitVPTOCubeDeviceObject(
    llvm::Module &module, llvm::StringRef llPath, llvm::StringRef outObjPath,
    const CANNToolchain &toolchain, llvm::StringRef bridgeBitcodePath,
    bool usesPrint, llvm::StringRef stderrPath, llvm::raw_ostream &diagOS) {
  if (failed(applyVPTOLLVMABINames(
          module,
          toolchain.vptoPublicABISuffix(ObjectEmissionDeviceTarget::Cube),
          diagOS))) {
    return failure();
  }
  if (failed(writeLLVMModule(module, llPath, diagOS))) {
    return failure();
  }
  std::string compileInput = llPath.str();
  llvm::SmallVector<std::string, mlir::pto::kValue2> extraBitcodePaths;
  if (!bridgeBitcodePath.empty()) {
    extraBitcodePaths.push_back(bridgeBitcodePath.str());
  }
  if (usesPrint) {
    std::string wrapperSourcePath = getPrintWrapperSourcePath(diagOS);
    if (wrapperSourcePath.empty())
      return failure();
    std::string printBitcodePath = (outObjPath + ".print.bc").str();
    if (!compilePrintWrapperToBitcode(
            toolchain,
            resolveTargetCPU(module, ObjectEmissionDeviceTarget::Cube),
            wrapperSourcePath, printBitcodePath, stderrPath, diagOS))
      return failure();
    extraBitcodePaths.push_back(printBitcodePath);
  }
  if (!extraBitcodePaths.empty()) {
    std::string linkedPath = (outObjPath + ".linked.bc").str();
    if (!linkDeviceLLVMBitcode(llPath, extraBitcodePaths, linkedPath,
                               toolchain, stderrPath, diagOS))
      return failure();
    compileInput = linkedPath;
  }
  return compileDeviceLLVMToObject(
             compileInput, outObjPath,
             resolveTargetCPU(module, ObjectEmissionDeviceTarget::Cube),
             toolchain.bishengPath, stderrPath, diagOS)
             ? success()
             : failure();
}

mlir::LogicalResult mlir::pto::emitFatobjLLVM(
    llvm::Module *cubeModule, llvm::Module *vectorModule,
    llvm::StringRef stubSource, llvm::StringRef bridgeWrapperSource,
    llvm::StringRef outputPath,
    llvm::StringRef moduleId, const CANNToolchain &toolchain,
    TempFileRegistry &tempFiles, VFSIMTSizeFixMode vfsimtSizeFixMode,
    bool usesPrint, llvm::raw_ostream &diagOS) {
  if (!cubeModule && !vectorModule) {
    diagOS << "Error: VPTO fatobj emission requires at least one LLVM module.\n";
    return failure();
  }

  VPTOFatobjArtifacts artifacts(tempFiles);
  if (!artifacts.emitStubSource(stubSource, diagOS)) {
    return failure();
  }
  if (!artifacts.initCommandLogs(diagOS)) {
    return failure();
  }
  std::string cubeBridgeBitcodePath;
  if (cubeModule &&
      !artifacts.compileBridgeWrapper(bridgeWrapperSource,
                                      ObjectEmissionDeviceTarget::Cube,
                                      toolchain, cubeBridgeBitcodePath,
                                      diagOS))
    return failure();
  std::string vectorBridgeBitcodePath;
  if (vectorModule &&
      !artifacts.compileBridgeWrapper(bridgeWrapperSource,
                                      ObjectEmissionDeviceTarget::Vector,
                                      toolchain, vectorBridgeBitcodePath,
                                      diagOS))
    return failure();
  if (!artifacts.emitCubeObject(cubeModule, toolchain, cubeBridgeBitcodePath,
                                usesPrint, diagOS)) {
    return failure();
  }
  if (!artifacts.emitVectorObject(vectorModule, toolchain,
                                  vectorBridgeBitcodePath, vfsimtSizeFixMode,
                                  usesPrint, diagOS)) {
    return failure();
  }
  if (!artifacts.mergeDeviceObjects(toolchain, diagOS)) {
    return failure();
  }

  constexpr llvm::StringLiteral targetCPU = "dav-c310";
  if (!artifacts.compileHostStubToFatobj(toolchain, moduleId, targetCPU,
                                         outputPath, usesPrint, diagOS)) {
    return failure();
  }
  return success();
}

mlir::LogicalResult mlir::pto::mergeDeviceObjects(
    llvm::ArrayRef<std::string> deviceObjPaths, llvm::StringRef outObjPath,
    const CANNToolchain &toolchain, llvm::StringRef stderrPath,
    llvm::raw_ostream &diagOS) {
  return ::mergeDeviceObjects(deviceObjPaths, outObjPath, toolchain.ldLldPath,
                              stderrPath, diagOS)
             ? success()
             : failure();
}

mlir::LogicalResult mlir::pto::compileStubToFatobj(
    llvm::StringRef stubPath, llvm::StringRef deviceObjPath,
    llvm::StringRef outputPath, llvm::StringRef moduleId,
    const CANNToolchain &toolchain, llvm::StringRef stderrPath,
    llvm::raw_ostream &diagOS) {
  constexpr llvm::StringLiteral targetCPU = "dav-c310";
  return compileHostStubToObject(stubPath, outputPath, moduleId, targetCPU,
                                 toolchain, deviceObjPath,
                                 /*usesPrint=*/false, stderrPath, diagOS)
             ? success()
             : failure();
}

mlir::LogicalResult mlir::pto::linkFatobjs(
    llvm::ArrayRef<std::string> fatobjPaths, llvm::StringRef outputPath,
    const CANNToolchain &toolchain, llvm::StringRef stderrPath,
    llvm::raw_ostream &diagOS) {
  return linkFatobjFiles(fatobjPaths, outputPath, toolchain, stderrPath, diagOS)
             ? success()
             : failure();
}

mlir::LogicalResult mlir::pto::emitFatobjLLVMWithRuntime(
    llvm::Module *cubeModule, llvm::Module *vectorModule,
    llvm::StringRef stubSource, llvm::StringRef bridgeWrapperSource,
    llvm::ToolOutputFile &outputFile,
    VFSIMTSizeFixMode vfsimtSizeFixMode,
    llvm::raw_ostream &diagOS) {
  if (!cubeModule && !vectorModule) {
    diagOS << "Error: VPTO fatobj emission requires at least one LLVM module.\n";
    return failure();
  }

  std::optional<CANNToolchain> toolchain = CANNToolchain::create(diagOS);
  if (!toolchain) {
    return failure();
  }

  TempFileRegistry tempFiles;
  VPTOFatobjArtifacts artifacts(tempFiles);
  if (!artifacts.emitStubSource(stubSource, diagOS)) {
    return failure();
  }
  if (!artifacts.initCommandLogs(diagOS)) {
    return failure();
  }

  std::string cubeBridgeBitcodePath;
  if (cubeModule &&
      !artifacts.compileBridgeWrapper(bridgeWrapperSource,
                                      ObjectEmissionDeviceTarget::Cube,
                                      *toolchain, cubeBridgeBitcodePath,
                                      diagOS))
    return failure();
  std::string vectorBridgeBitcodePath;
  if (vectorModule &&
      !artifacts.compileBridgeWrapper(bridgeWrapperSource,
                                      ObjectEmissionDeviceTarget::Vector,
                                      *toolchain, vectorBridgeBitcodePath,
                                      diagOS))
    return failure();
  if (!artifacts.emitCubeObject(cubeModule, *toolchain,
                                cubeBridgeBitcodePath, /*usesPrint=*/false,
                                diagOS)) {
    return failure();
  }
  if (!artifacts.emitVectorObject(vectorModule, *toolchain,
                                  vectorBridgeBitcodePath, vfsimtSizeFixMode,
                                  /*usesPrint=*/false, diagOS)) {
    return failure();
  }

  if (!artifacts.mergeDeviceObjects(*toolchain, diagOS)) {
    return failure();
  }

  std::string moduleId = sanitizeModuleId(outputFile.getFilename());
  constexpr llvm::StringLiteral hostTargetCPU = "dav-c310";
  if (!artifacts.compileHostStub(*toolchain, moduleId, hostTargetCPU,
                                 /*usesPrint=*/false, diagOS)) {
    return failure();
  }

  if (!artifacts.repackFatObj(*toolchain, moduleId, hostTargetCPU,
                              outputFile.getFilename(), diagOS)) {
    return failure();
  }
  outputFile.keep();
  return success();
}
