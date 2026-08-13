// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOPOSTUPDATEUTILS_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOPOSTUPDATEUTILS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringMap.h"
#include <cstdint>
#include <optional>

namespace mlir {
namespace pto {

enum class PostUpdateAddressUnit {
  Element,
  Block,
  Alignment,
  Byte,
};

enum class PostUpdateStrideConstraint {
  Dynamic,
  Constant,
  SignedI8,
};

// Mathematical interpretation of an address operand when it is widened or
// converted into a post-update pointer delta. Block stride fields are unsigned
// in the VPTO ABI; the remaining registered address operands are signed.
enum class PostUpdateAddressDomain {
  Signed,
  Unsigned,
};

// Selects the type whose element width defines an Element address unit. Most
// stores address in base-pointer elements, while load distributions and
// unaligned streams may address in payload elements instead.
enum class PostUpdateElementTypeSource {
  Base,
  Operand,
  Result,
};

// Shared semantic description for every operation considered by VPTO soft
// post-update. `strideParticipatesInCurrentAddress` distinguishes ordinary
// base+offset operations from stateful operations such as vstus, whose offset
// advances the returned base but is not added to the current access address.
struct PostUpdateOpInfo {
  unsigned baseOperandIdx;
  std::optional<unsigned> strideOperandIdx;
  bool strideParticipatesInCurrentAddress;
  PostUpdateAddressUnit addressUnit;
  unsigned minResultsForPost;
  PostUpdateAddressDomain strideDomain = PostUpdateAddressDomain::Signed;
  PostUpdateStrideConstraint strideConstraint =
      PostUpdateStrideConstraint::Dynamic;
  PostUpdateElementTypeSource elementTypeSource =
      PostUpdateElementTypeSource::Base;
  unsigned elementTypeIndex = 0;
};

using PostUpdateOpTable = llvm::StringMap<PostUpdateOpInfo>;

const PostUpdateOpTable &getPostUpdateOpTable();
const PostUpdateOpInfo *getPostUpdateOpInfo(Operation *op);

// Match the canonical VPTO address recurrence. The value must be an
// i16 scf.for iter_arg whose backedge is a constant-step arith.addi/subi with
// the no-wrap flag corresponding to `domain`. The overflow flag records the
// proof established by address normalization; consumers must not infer the
// same fact again from the recurrence shape alone.
std::optional<int64_t>
getCanonicalAddressRecurrenceStep(Value value, scf::ForOp forOp,
                                  PostUpdateAddressDomain domain);

// Remove loop-carried values and pure def chains that are no longer reachable
// from side-effecting operations or externally used loop results.
scf::ForOp pruneDeadLoopCarriedValues(scf::ForOp forOp, OpBuilder &builder);

std::optional<int64_t> getPostUpdateBaseUnitBytes(Value base);
std::optional<int64_t> getPostUpdateAddressUnitBytes(Operation *op,
                                                     const PostUpdateOpInfo &info,
                                                     int64_t baseElementBytes);

bool satisfiesPostUpdateStrideConstraint(PostUpdateStrideConstraint constraint,
                                         std::optional<int64_t> constantStride);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOPOSTUPDATEUTILS_H
