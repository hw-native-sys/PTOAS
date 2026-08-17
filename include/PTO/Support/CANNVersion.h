// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_SUPPORT_CANNVERSION_H
#define PTO_SUPPORT_CANNVERSION_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <limits>
#include <optional>

namespace mlir::pto {

inline constexpr unsigned kCANNVersionComponentCount = 3;
inline constexpr unsigned kDecimalRadix = 10;

struct CANNVersion {
  static constexpr unsigned RELEASE = std::numeric_limits<unsigned>::max();

  constexpr CANNVersion(unsigned major, unsigned minor, unsigned patch,
                        unsigned beta)
      : major(major), minor(minor), patch(patch), beta(beta) {}

  static CANNVersion release(unsigned major, unsigned minor, unsigned patch) {
    return CANNVersion{major, minor, patch, RELEASE};
  }

  unsigned major;
  unsigned minor;
  unsigned patch;
  unsigned beta;

  bool operator<(const CANNVersion &rhs) const {
    if (major != rhs.major) {
      return major < rhs.major;
    }
    if (minor != rhs.minor) {
      return minor < rhs.minor;
    }
    if (patch != rhs.patch) {
      return patch < rhs.patch;
    }
    return beta < rhs.beta;
  }

  bool operator==(const CANNVersion &rhs) const {
    return major == rhs.major && minor == rhs.minor && patch == rhs.patch &&
           beta == rhs.beta;
  }

  bool operator>=(const CANNVersion &rhs) const { return !(*this < rhs); }
};

inline constexpr CANNVersion kDefaultCANNVersion{9, 0, 0, 1};
inline constexpr CANNVersion kCANN900Beta2Version{9, 0, 0, 2};

inline std::optional<unsigned> parseCANNVersionComponent(
    llvm::StringRef value) {
  unsigned result = 0;
  if (value.empty() || value.getAsInteger(kDecimalRadix, result)) {
    return std::nullopt;
  }
  return result;
}

inline std::optional<unsigned> parseCANNBetaVersion(
    llvm::StringRef prerelease) {
  if (!prerelease.consume_front("beta.")) {
    return std::nullopt;
  }
  return parseCANNVersionComponent(prerelease);
}

inline std::optional<CANNVersion> parseCANNVersion(llvm::StringRef version) {
  auto [core, prerelease] = version.split('-');
  llvm::SmallVector<llvm::StringRef, kCANNVersionComponentCount> components;
  core.split(components, '.');
  if (components.size() != kCANNVersionComponentCount) {
    return std::nullopt;
  }

  std::optional<unsigned> major = parseCANNVersionComponent(components[0]);
  std::optional<unsigned> minor = parseCANNVersionComponent(components[1]);
  std::optional<unsigned> patch = parseCANNVersionComponent(components[2]);
  if (!major || !minor || !patch) {
    return std::nullopt;
  }

  unsigned beta = CANNVersion::RELEASE;
  if (!prerelease.empty()) {
    std::optional<unsigned> parsedBeta = parseCANNBetaVersion(prerelease);
    if (!parsedBeta) {
      return std::nullopt;
    }
    beta = *parsedBeta;
  }

  return CANNVersion{*major, *minor, *patch, beta};
}

} // namespace mlir::pto

#endif // PTO_SUPPORT_CANNVERSION_H
