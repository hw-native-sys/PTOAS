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
#include <cstring>
#include <errno.h>
#include <signal.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern char **environ;

namespace ptoas {

std::optional<std::pair<int, std::string>> DaemonManager::processInfo;

std::string DaemonManager::generateSocketPath() {
  return "/tmp/tilelib_daemon_" + std::to_string(::getpid()) + ".sock";
}

// Outcome of a daemon terminate-and-reap attempt.
enum class ReapResult {
  ExitedGracefully, // reaped after SIGTERM
  ForceKilled,      // reaped after SIGKILL
  AlreadyGone,      // process did not exist when we first signalled it
  AlreadyReaped,    // waitpid returned ECHILD — already reaped elsewhere
  SyscallError,     // unexpected kill/waitpid failure
};

// Forward declaration: defined after stop(), used by the startup-timeout
// path in start() so a daemon that never opens its socket is still
// precisely terminated and reaped rather than orphaned.
static ReapResult terminateAndReap(int pid, int *outStatus);

bool DaemonManager::start(const std::string &socketPath,
                          const std::string &daemonModule,
                          const std::string &pythonExe,
                          const std::string &pkgPath,
                          const std::string &templateDir) {
  // Stop any previously-started daemon before launching a new one.  The
  // static processInfo singleton holds only one PID; without this call a
  // second start() (e.g. test_ptoas_runtime invokes _core.main() twice)
  // would orphan the first daemon, which keeps an inherited copy of the
  // parent's stdout pipe open and blocks ctest until its 1500s timeout.
  // If stop() fails with a syscall error, the old daemon may still be alive;
  // do NOT overwrite the PID — abort start() so the caller can retry.
  if (processInfo && !stop()) {
    llvm::errs() << "Error: cannot start a new daemon while the previous "
                    "daemon (pid="
                 << processInfo->first
                 << ") is still active or its status is unknown\n";
    return false;
  }

  auto pythonPath =
      mlir::pto::resolvePythonExecutable(pythonExe.empty() ? "python3"
                                                           : pythonExe);
  if (!pythonPath) {
    llvm::errs() << "Error: Cannot find Python executable '"
                 << (pythonExe.empty() ? "python3" : pythonExe)
                 << "' for daemon\n";
    return false;
  }

  // Run the daemon with full site initialization rather than `-S`. The
  // editable (scikit-build redirect) install relies on a meta-path finder
  // registered by the site-package `.pth` file; `-S` skips site.py and never
  // installs it. Without site initialization the source-tree `ptoas` package
  // (a regular package with `__init__.py`) shadows the build-tree
  // `ptoas.mlir` namespace package on PYTHONPATH, so the daemon fails to
  // import `ptoas.mlir.dialects.pto` and never opens its socket.
  llvm::SmallVector<llvm::StringRef, 8> args = {*pythonPath};
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

  // Redirect the daemon's stdin/stdout/stderr to /dev/null so the detached
  // process does not inherit the parent's pipe descriptors.  When an
  // ExecuteNoWait child keeps a copy of the parent's stdout (ctest captures
  // it via a pipe), ctest blocks on the pipe until every write end is closed.
  // A leaked daemon that outlives the parent (e.g. a second start() orphaning
  // the first) would hold that write end open and stall ctest until timeout.
  std::optional<llvm::StringRef> redirects[] = {
      llvm::StringRef("/dev/null"), llvm::StringRef("/dev/null"),
      llvm::StringRef("/dev/null")};

  llvm::sys::ProcessInfo procInfo = llvm::sys::ExecuteNoWait(
      *pythonPath, args,
      !pkgPath.empty()
          ? std::optional<llvm::ArrayRef<llvm::StringRef>>(envp)
          : std::nullopt,
      redirects, 0, &errMsg, &executionFailed, nullptr, true);

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
    // Reuse the same precise terminate-and-reap path as stop(); otherwise the
    // PID is dropped here and atexit cleanup can no longer find it, leaving an
    // orphan if the daemon ignores or delays handling SIGTERM.
    int dummyStatus = 0;
    ReapResult reapResult = terminateAndReap(procInfo.Pid, &dummyStatus);
    if (reapResult == ReapResult::SyscallError) {
      // Unexpected syscall failure: daemon may still be alive — keep the
      // PID so atexit can retry.  Do NOT overwrite processInfo in start().
      processInfo = std::make_pair(procInfo.Pid, socketPath);
      llvm::errs() << "Error: Could not confirm daemon (pid=" << procInfo.Pid
                   << ") exit; PID retained for atexit retry\n";
    } else {
      // Confirmed dead (graceful, force-killed, already gone, or already
      // reaped elsewhere).  Safe to clear.
      if (llvm::sys::fs::exists(socketPath)) {
        llvm::sys::fs::remove(socketPath);
      }
      processInfo = std::nullopt;
    }
    return false;
  }

  llvm::errs() << "TileLib daemon '" << daemonModule << "' started (pid="
               << procInfo.Pid
               << ", socket=" << socketPath << ")\n";
  return true;
}

// Terminate and reap exactly one daemon PID.  Shared by stop() and the
// startup-timeout path so neither can drop the PID before the process is
// actually gone (which would leave an orphan that atexit cleanup can no
// longer find).  Uses waitpid(pid, ...) rather than waitpid(-1, ...) so it
// never steals the exit status of an unrelated child that PTOAS may have
// spawned (compiler invocations, inline helpers).
//
// Returns a ReapResult so callers can distinguish a graceful SIGTERM exit
// from a forced SIGKILL, and diagnose unexpected syscall failures.  Every
// kill/waitpid return value is explicitly checked.
static ReapResult terminateAndReap(int pid, int *outStatus) {
  int status = 0;

  // ---- Step 1: graceful SIGTERM ----
  int termRet = kill(pid, SIGTERM);
  if (termRet == -1) {
    if (errno == ESRCH) {
      // Process already gone; try a non-blocking reap to avoid a zombie.
      // (Loop with explicit break so the control statement does not nest a
      // function call inside its condition — avoids codecheck false-positive.)
      for (;;) {
        pid_t wr = waitpid(pid, &status, WNOHANG);
        if (wr != -1 || errno != EINTR) {
          break;
        }
      }
      if (outStatus) {
        *outStatus = status;
      }
      return ReapResult::AlreadyGone;
    }
    // EPERM or other unexpected error.
    llvm::errs() << "Warning: kill(SIGTERM) for daemon pid=" << pid
                 << " failed: " << std::strerror(errno) << "\n";
    if (outStatus) {
      *outStatus = 0;
    }
    return ReapResult::SyscallError;
  }

  // ---- Step 2: wait up to 2s for graceful exit ----
  // Python's serve_forever(poll_interval=0.05) + shutdown() completes in
  // ~60-80ms, but allow generous headroom for GC/import teardown.  Use
  // waitpid(WNOHANG) rather than kill(pid, 0) to detect exit so a PID reused
  // by another process is not mistaken for a still-running daemon.
  bool reaped = false;
  bool alreadyReaped = false;
  for (int attempt = 0; attempt < 200; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    pid_t w = waitpid(pid, &status, WNOHANG);
    if (w == pid) {
      reaped = true;
      break;
    }
    if (w == -1) {
      if (errno == ECHILD) {
        // Not our child (e.g. already reaped elsewhere).  We cannot confirm
        // the daemon actually exited; stop here but flag the uncertainty.
        alreadyReaped = true;
        break;
      }
      if (errno != EINTR) {
        llvm::errs() << "Warning: waitpid(WNOHANG) for daemon pid=" << pid
                     << " failed: " << std::strerror(errno) << "\n";
        if (outStatus) {
          *outStatus = 0;
        }
        return ReapResult::SyscallError;
      }
      // EINTR: retry the loop.
    }
    // w == 0: still running; continue polling.
  }

  if (alreadyReaped) {
    if (outStatus) {
      *outStatus = 0;
    }
    return ReapResult::AlreadyReaped;
  }

  // ---- Step 3: force-kill if still alive ----
  if (!reaped) {
    int killRet = kill(pid, SIGKILL);
    if (killRet == -1) {
      if (errno == ESRCH) {
        // Exited between the check and the SIGKILL; try one more reap.
        for (;;) {
          pid_t wr = waitpid(pid, &status, WNOHANG);
          if (wr != -1 || errno != EINTR) {
            break;
          }
        }
        if (outStatus) {
          *outStatus = status;
        }
        return ReapResult::ExitedGracefully;
      }
      llvm::errs() << "Warning: kill(SIGKILL) for daemon pid=" << pid
                   << " failed: " << std::strerror(errno) << "\n";
      if (outStatus) {
        *outStatus = 0;
      }
      return ReapResult::SyscallError;
    }
    // Block until it is actually gone so the PID is not reused by another
    // process before we finish cleanup.  Retry on EINTR; surface other
    // errors rather than silently dropping the PID.
    while (true) {
      pid_t w = waitpid(pid, &status, 0);
      if (w == pid) {
        break;
      }
      if (w == -1 && errno == EINTR) {
        continue;
      }
      llvm::errs() << "Warning: blocking waitpid for daemon pid=" << pid
                   << " failed: " << std::strerror(errno) << "\n";
      if (outStatus) {
        *outStatus = 0;
      }
      return ReapResult::SyscallError;
    }
    if (outStatus) {
      *outStatus = status;
    }
    return ReapResult::ForceKilled;
  }

  if (outStatus) {
    *outStatus = status;
  }
  return ReapResult::ExitedGracefully;
}

bool DaemonManager::stop() {
  if (!processInfo) {
    return true;
  }

  int pid = processInfo->first;
  std::string socketPath = processInfo->second;

  // Precisely terminate and reap *only* this daemon PID. The daemon is an
  // ExecuteNoWait child, so it is still our child for reap purposes; using
  // waitpid(-1) here would steal the exit status of other children (compiler
  // invocations, helpers) that PTOAS may have spawned, causing their callers
  // to see ECHILD.
  int status = 0;
  ReapResult result = terminateAndReap(pid, &status);
  switch (result) {
  case ReapResult::ForceKilled:
    llvm::errs() << "Warning: TileLib daemon (pid=" << pid
                 << ") did not exit on SIGTERM and was force-killed\n";
    break;
  case ReapResult::SyscallError:
    // Do NOT clear processInfo: the daemon may still be alive, and clearing
    // the PID here would prevent a later atexit retry.  Report the failure
    // and leave the socket in place so a subsequent stop() can retry.
    llvm::errs() << "Error: TileLib daemon (pid=" << pid
                 << ") termination failed with a syscall error; "
                 << "PID not cleared for safety\n";
    return false;
  case ReapResult::AlreadyReaped:
    // ECHILD under the current direct-child model means the process was
    // already reaped elsewhere and is no longer our child.  The PID may be
    // reused by an unrelated process, so we must NOT keep it for a retry.
    // Treat as a terminal outcome: clean up state, log the uncertainty.
    llvm::errs() << "Warning: TileLib daemon (pid=" << pid
                 << ") was already reaped (ECHILD); exit status unknown\n";
    break;
  case ReapResult::AlreadyGone:
  case ReapResult::ExitedGracefully:
    break;
  }

  if (llvm::sys::fs::exists(socketPath)) {
    llvm::sys::fs::remove(socketPath);
  }

  llvm::errs() << "TileLib daemon stopped (pid=" << pid << ")\n";
  processInfo = std::nullopt;
  return true;
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
