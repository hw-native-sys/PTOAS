// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTOAS_TILELANG_DAEMON_H
#define PTOAS_TILELANG_DAEMON_H

#include <optional>
#include <string>
#include <utility>

namespace llvm::sys {
using procid_t = int;
}

namespace ptoas {

class DaemonManager {
public:
  static std::string generateSocketPath();

  static bool start(const std::string &socketPath,
                    const std::string &daemonModule,
                    const std::string &pythonExe,
                    const std::string &pkgPath,
                    const std::string &templateDir = "");

  static void stop();

  static bool isRunning();

private:
  static std::optional<std::pair<int, std::string>> processInfo;
};

void registerDaemonCleanup();

} // namespace ptoas

#endif // PTOAS_TILELANG_DAEMON_H
