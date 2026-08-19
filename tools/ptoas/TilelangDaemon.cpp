// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Support/PythonExecutable.h"
#include "TilelangDaemon.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include <chrono>
#include <cstdlib>
#include <signal.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern char **environ;

namespace ptoas {

std::optional<std::pair<int, std::string>> DaemonManager::processInfo;

std::string DaemonManager::generateSocketPath() {
  return "/tmp/tilelib_daemon_" + std::to_string(::getpid()) + ".sock";
}

bool DaemonManager::start(const std::string &socketPath,
                          const std::string &daemonModule,
                          const std::string &pythonExe,
                          const std::string &pkgPath,
                          const std::string &templateDir) {
  auto pythonPath =
      mlir::pto::resolvePythonExecutable(pythonExe.empty() ? "python3"
                                                           : pythonExe);
  if (!pythonPath) {
    llvm::errs() << "Error: Cannot find Python executable '"
                 << (pythonExe.empty() ? "python3" : pythonExe)
                 << "' for daemon\n";
    return false;
  }

  llvm::SmallVector<llvm::StringRef, 8> args = {*pythonPath};
  if (daemonModule == "ptodsl.tilelib.serving.daemon") {
    args.push_back("-S");
  }
  args.append({"-m", daemonModule, "--socket", socketPath});
  if (!templateDir.empty()) {
    args.push_back("--template-dir");
    args.push_back(templateDir);
  }

  llvm::SmallVector<llvm::StringRef> envp;
  std::string pythonPathEnv;
  std::vector<std::string> envStorage;

  if (!pkgPath.empty()) {
    const char *existingPath = ::getenv("PYTHONPATH");
    pythonPathEnv = "PYTHONPATH=" + pkgPath;
    if (existingPath && existingPath[0] != '\0') {
      pythonPathEnv += ":";
      pythonPathEnv += existingPath;
    }
    for (char **e = environ; *e; ++e) {
      llvm::StringRef entry(*e);
      bool skipEntry = entry.starts_with("PYTHONPATH=") || entry.starts_with("SKBUILD_EDITABLE_SKIP=");
      if (skipEntry) {
        continue;
      }
      envStorage.push_back(std::string(entry));
    }
    envStorage.push_back(pythonPathEnv);
    // The configured build-tree package must win over an editable wheel's
    // meta-path redirect.  Otherwise the daemon can load stale MLIR bindings
    // from site-packages even though PYTHONPATH names this checkout first.
    envStorage.push_back("SKBUILD_EDITABLE_SKIP=1");
    for (auto &s : envStorage)
      envp.push_back(s);
  }

  std::string errMsg;
  bool executionFailed = false;

  llvm::sys::ProcessInfo procInfo = llvm::sys::ExecuteNoWait(
      *pythonPath, args,
      !pkgPath.empty()
          ? std::optional<llvm::ArrayRef<llvm::StringRef>>(envp)
          : std::nullopt,
      {}, 0, &errMsg, &executionFailed, nullptr, true);

  if (executionFailed || procInfo.Pid == llvm::sys::ProcessInfo::InvalidPid) {
    llvm::errs() << "Error: Failed to start TileLib daemon module '"
                 << daemonModule << "': " << errMsg << "\n";
    return false;
  }

  processInfo = std::make_pair(procInfo.Pid, socketPath);

  // Python startup time depends on the selected TileLib frontend and its
  // imports. Poll instead of relying on one fixed sleep.
  bool socketReady = false;
  for (int attempt = 0; attempt < 200; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (llvm::sys::fs::exists(socketPath)) {
      socketReady = true;
      break;
    }
  }

  if (!socketReady) {
    llvm::errs() << "Error: Daemon socket not created at " << socketPath << "\n";
    llvm::errs() << "Note: Daemon process started (pid=" << procInfo.Pid
                 << ") but socket not found. Check daemon logs.\n";
    kill(procInfo.Pid, SIGTERM);
    processInfo = std::nullopt;
    return false;
  }

  llvm::errs() << "TileLib daemon '" << daemonModule << "' started (pid="
               << procInfo.Pid
               << ", socket=" << socketPath << ")\n";
  return true;
}

void DaemonManager::stop() {
  if (!processInfo)
    return;

  int pid = processInfo->first;
  std::string socketPath = processInfo->second;

  kill(pid, SIGTERM);

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  if (llvm::sys::fs::exists(socketPath)) {
    llvm::sys::fs::remove(socketPath);
  }

  llvm::errs() << "TileLib daemon stopped (pid=" << pid << ")\n";
  processInfo = std::nullopt;
}

bool DaemonManager::isRunning() {
  return processInfo.has_value();
}

static void daemonCleanupHandler() {
  DaemonManager::stop();
}

void registerDaemonCleanup() {
  std::atexit(daemonCleanupHandler);
}

} // namespace ptoas
