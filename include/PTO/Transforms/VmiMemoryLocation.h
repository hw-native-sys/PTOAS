// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_VMIMEMORYLOCATION_H
#define PTO_TRANSFORMS_VMIMEMORYLOCATION_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include <cstdint>
#include <optional>

namespace mlir {
namespace pto {

/// A compile-time storage root for a pointer_cast-created UB view.  The
/// address is the numeric address from the IR, not the SSA identity of the
/// arith.constant operation.  `storageBytes` is the statically-known extent of
/// the view when available; it is used only to prove two roots disjoint.
struct VmiStorageRoot {
  int64_t address = 0;
  std::optional<int64_t> storageBytes;
  Type viewType;

  bool operator==(const VmiStorageRoot &other) const {
    return address == other.address && viewType == other.viewType;
  }
};

std::optional<VmiStorageRoot> resolveVmiStorageRoot(Value base);

/// Returns true when the two statically-known views may overlap.  Unknown
/// extents are treated conservatively.  Equal numeric addresses always alias,
/// including views with different element types.
bool mayAliasVmiStorageRoot(const VmiStorageRoot &lhs,
                            const VmiStorageRoot &rhs);

struct VmiAccessLocation {
  VmiStorageRoot root;
  Value elementOffset;
  int64_t accessBytes = 0;
};

/// Compare two accesses in the same element-index domain.  Unknown offsets,
/// element widths, or access sizes conservatively return true when their
/// storage roots may alias.
bool mayAliasVmiAccess(const VmiAccessLocation &lhs,
                       const VmiAccessLocation &rhs);

} // namespace pto
} // namespace mlir

#endif
