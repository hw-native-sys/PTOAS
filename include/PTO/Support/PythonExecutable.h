// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_SUPPORT_PYTHONEXECUTABLE_H
#define PTO_SUPPORT_PYTHONEXECUTABLE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

#include <optional>
#include <string>

namespace mlir::pto {

inline std::optional<std::string>
resolvePythonExecutable(llvm::StringRef pythonExe) {
  if (pythonExe.empty()) {
    return std::nullopt;
  }

  if (llvm::sys::path::is_absolute(pythonExe)) {
    if (llvm::sys::fs::can_execute(pythonExe)) {
      return pythonExe.str();
    }
    return std::nullopt;
  }

  auto found = llvm::sys::findProgramByName(pythonExe);
  if (!found) {
    return std::nullopt;
  }
  return *found;
}

} // namespace mlir::pto

#endif // PTO_SUPPORT_PYTHONEXECUTABLE_H
