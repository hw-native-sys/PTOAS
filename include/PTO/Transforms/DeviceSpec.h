//===- DeviceSpec.h - PTO device/arch helpers ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_DEVICE_SPEC_H
#define MLIR_DIALECT_PTO_TRANSFORMS_DEVICE_SPEC_H

#include "PTO/Transforms/Passes.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir {
namespace pto {

inline std::optional<PTOArch> parsePTOArch(llvm::StringRef s) {
  if (s.empty())
    return std::nullopt;
  if (s.equals_insensitive("a3"))
    return PTOArch::A3;
  if (s.equals_insensitive("a5"))
    return PTOArch::A5;
  return std::nullopt;
}

inline llvm::StringRef toString(PTOArch arch) {
  switch (arch) {
  case PTOArch::A3:
    return "a3";
  case PTOArch::A5:
    return "a5";
  }
  return "a3";
}

// Map known device-spec strings to PTOArch.
//
// Policy:
// - Ascend910B / Ascend91093 -> A3
// - Ascend910_950 / Ascend910_95xx / Ascend950 -> A5
inline std::optional<PTOArch> inferPTOArchFromDeviceSpec(llvm::StringRef spec) {
  if (spec.empty())
    return std::nullopt;

  if (spec.starts_with("Ascend910B"))
    return PTOArch::A3;
  if (spec.starts_with("Ascend910_93") || spec.starts_with("Ascend91093"))
    return PTOArch::A3;

  // A5 instruction set devices.
  if (spec.starts_with("Ascend910_95") || spec.starts_with("Ascend91095") ||
      spec.starts_with("Ascend950"))
    return PTOArch::A5;

  return std::nullopt;
}

inline bool deviceSpecMatchesArch(llvm::StringRef spec, PTOArch arch) {
  auto inferred = inferPTOArchFromDeviceSpec(spec);
  return !inferred.has_value() || inferred.value() == arch;
}

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_DEVICE_SPEC_H

