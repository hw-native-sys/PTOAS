// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOMemoryDist.cpp - VPTO vector memory dist contracts ------------===//

#include "PTO/IR/VPTOMemoryDist.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>

using namespace mlir;
using namespace mlir::pto;

namespace {

using Family = VPTOMemoryOpFamily;
using Dist = VPTOMemoryDist;
using Predicate = VPTOPredicatePolicy;
using Transfer = VPTOMemoryTransferKind;
using Size = VPTOMemorySizeRule;

static const VPTOMemoryDistContract contracts[] = {
    {Family::Load,
     Dist::LoadNorm,
     "NORM",
     0,
     0,
     1,
     Predicate::NotPresent,
     Transfer::Identity,
     Size::Block,
     Size::Vector,
     {}},
    {Family::Load,
     Dist::BrcB8,
     "BRC_B8",
     8,
     1,
     1,
     Predicate::NotPresent,
     Transfer::ScalarBroadcast,
     Size::Element,
     Size::Element,
     {}},
    {Family::Load,
     Dist::BrcB16,
     "BRC_B16",
     16,
     2,
     1,
     Predicate::NotPresent,
     Transfer::ScalarBroadcast,
     Size::Element,
     Size::Element,
     {}},
    {Family::Load,
     Dist::BrcB32,
     "BRC_B32",
     32,
     3,
     1,
     Predicate::NotPresent,
     Transfer::ScalarBroadcast,
     Size::Element,
     Size::Element,
     {}},
    {Family::Load,
     Dist::UsB8,
     "US_B8",
     8,
     6,
     1,
     Predicate::NotPresent,
     Transfer::Upsample2,
     Size::VectorDiv2,
     Size::VectorDiv2,
     {}},
    {Family::Load,
     Dist::UsB16,
     "US_B16",
     16,
     7,
     1,
     Predicate::NotPresent,
     Transfer::Upsample2,
     Size::VectorDiv2,
     Size::VectorDiv2,
     {}},
    {Family::Load,
     Dist::DsB8,
     "DS_B8",
     8,
     8,
     1,
     Predicate::NotPresent,
     Transfer::Downsample2,
     Size::Block,
     Size::VectorTimes2,
     {}},
    {Family::Load,
     Dist::DsB16,
     "DS_B16",
     16,
     9,
     1,
     Predicate::NotPresent,
     Transfer::Downsample2,
     Size::Block,
     Size::VectorTimes2,
     {}},
    {Family::LoadX2,
     Dist::BlockDeinterleave,
     "BDINTLV",
     0,
     10,
     2,
     Predicate::NotPresent,
     Transfer::BlockDeinterleave2,
     Size::Block,
     Size::VectorTimes2,
     {}},
    {Family::LoadX2,
     Dist::DeinterleaveB8,
     "DINTLV_B8",
     8,
     11,
     2,
     Predicate::NotPresent,
     Transfer::ElementDeinterleave2,
     Size::Block,
     Size::VectorTimes2,
     {}},
    {Family::LoadX2,
     Dist::DeinterleaveB16,
     "DINTLV_B16",
     16,
     12,
     2,
     Predicate::NotPresent,
     Transfer::ElementDeinterleave2,
     Size::Block,
     Size::VectorTimes2,
     {}},
    {Family::Load,
     Dist::UnpackB8,
     "UNPK_B8",
     8,
     13,
     1,
     Predicate::NotPresent,
     Transfer::LaneExpand2,
     Size::VectorDiv2,
     Size::VectorDiv2,
     {}},
    {Family::Load,
     Dist::UnpackB16,
     "UNPK_B16",
     16,
     14,
     1,
     Predicate::NotPresent,
     Transfer::LaneExpand2,
     Size::VectorDiv2,
     Size::VectorDiv2,
     {}},
    {Family::Load,
     Dist::BroadcastBlock,
     "BRC_BLK",
     0,
     15,
     1,
     Predicate::NotPresent,
     Transfer::BlockBroadcast,
     Size::Block,
     Size::Block,
     {}},
    {Family::Load,
     Dist::ElementToBlockB16,
     "E2B_B16",
     16,
     16,
     1,
     Predicate::NotPresent,
     Transfer::ElementToBlock,
     Size::VectorDiv16,
     Size::VectorDiv16,
     {}},
    {Family::Load,
     Dist::ElementToBlockB32,
     "E2B_B32",
     32,
     17,
     1,
     Predicate::NotPresent,
     Transfer::ElementToBlock,
     Size::VectorDiv8,
     Size::VectorDiv8,
     {}},
    {Family::Load,
     Dist::UnpackB32,
     "UNPK_B32",
     32,
     18,
     1,
     Predicate::NotPresent,
     Transfer::LaneExpand2,
     Size::VectorDiv2,
     Size::VectorDiv2,
     {}},
    {Family::LoadX2,
     Dist::DeinterleaveB32,
     "DINTLV_B32",
     32,
     19,
     2,
     Predicate::NotPresent,
     Transfer::ElementDeinterleave2,
     Size::Block,
     Size::VectorTimes2,
     {}},
    {Family::Load,
     Dist::Unpack4B8,
     "UNPK4",
     8,
     20,
     1,
     Predicate::NotPresent,
     Transfer::LaneExpand4,
     Size::VectorDiv4,
     Size::VectorDiv4,
     {}},
    {Family::Load,
     Dist::Split4ChannelB8,
     "SPLT4CHN",
     8,
     21,
     1,
     Predicate::NotPresent,
     Transfer::SplitChannel,
     Size::Block,
     Size::Vector,
     {}},
    {Family::Load,
     Dist::Split2ChannelB8,
     "SPLT2CHN_B8",
     8,
     22,
     1,
     Predicate::NotPresent,
     Transfer::SplitChannel,
     Size::Block,
     Size::Vector,
     {}},
    {Family::Load,
     Dist::Split2ChannelB16,
     "SPLT2CHN_B16",
     16,
     23,
     1,
     Predicate::NotPresent,
     Transfer::SplitChannel,
     Size::Block,
     Size::Vector,
     {}},
    {Family::Store,
     Dist::StoreNormB8,
     "NORM_B8",
     8,
     0,
     1,
     Predicate::Applied,
     Transfer::Identity,
     Size::Block,
     Size::Vector,
     {}},
    {Family::Store,
     Dist::StoreNormB16,
     "NORM_B16",
     16,
     1,
     1,
     Predicate::Applied,
     Transfer::Identity,
     Size::Block,
     Size::Vector,
     {}},
    {Family::Store,
     Dist::StoreNormB32,
     "NORM_B32",
     32,
     2,
     1,
     Predicate::Applied,
     Transfer::Identity,
     Size::Block,
     Size::Vector,
     {}},
    {Family::Store,
     Dist::OnePointB8,
     "1PT_B8",
     8,
     3,
     1,
     Predicate::Ignored,
     Transfer::Point,
     Size::Element,
     Size::Element,
     {}},
    {Family::Store,
     Dist::OnePointB16,
     "1PT_B16",
     16,
     4,
     1,
     Predicate::Ignored,
     Transfer::Point,
     Size::Element,
     Size::Element,
     {}},
    {Family::Store,
     Dist::OnePointB32,
     "1PT_B32",
     32,
     5,
     1,
     Predicate::Ignored,
     Transfer::Point,
     Size::Element,
     Size::Element,
     {}},
    {Family::Store, Dist::PackB16, "PK_B16", 8, 6, 1, Predicate::Applied,
     Transfer::LaneCompact2, Size::VectorDiv2, Size::VectorDiv2, "b16"},
    {Family::Store, Dist::PackB32, "PK_B32", 16, 7, 1, Predicate::Applied,
     Transfer::LaneCompact2, Size::VectorDiv2, Size::VectorDiv2, "b32"},
    {Family::StoreX2,
     Dist::InterleaveB8,
     "INTLV_B8",
     8,
     8,
     2,
     Predicate::Ignored,
     Transfer::ElementInterleave2,
     Size::Block,
     Size::VectorTimes2,
     {}},
    {Family::StoreX2,
     Dist::InterleaveB16,
     "INTLV_B16",
     16,
     9,
     2,
     Predicate::Ignored,
     Transfer::ElementInterleave2,
     Size::Block,
     Size::VectorTimes2,
     {}},
    {Family::Store, Dist::PackB64, "PK_B64", 32, 10, 1, Predicate::Applied,
     Transfer::LaneCompact2, Size::VectorDiv2, Size::VectorDiv2, "b32"},
    {Family::StoreX2,
     Dist::InterleaveB32,
     "INTLV_B32",
     32,
     11,
     2,
     Predicate::Ignored,
     Transfer::ElementInterleave2,
     Size::Block,
     Size::VectorTimes2,
     {}},
    // PK4_B32 is defined over the raw 256-byte register payload. Existing
    // legal producers use both byte-typed values and b32 carrier values after
    // explicit lane selection, so the register element type is not fixed.
    {Family::Store, Dist::Pack4B32, "PK4_B32", 0, 12, 1, Predicate::Applied,
     Transfer::LaneCompact4, Size::VectorDiv4, Size::VectorDiv4, "b32"},
    {Family::Store, Dist::Merge4ChannelB8, "MRG4CHN_B8", 8, 13, 1,
     Predicate::Applied, Transfer::MergeChannel, Size::Block, Size::Vector,
     "b32"},
    {Family::Store, Dist::Merge2ChannelB8, "MRG2CHN_B8", 8, 14, 1,
     Predicate::Applied, Transfer::MergeChannel, Size::Block, Size::Vector,
     "b16"},
    {Family::Store, Dist::Merge2ChannelB16, "MRG2CHN_B16", 16, 15, 1,
     Predicate::Applied, Transfer::MergeChannel, Size::Block, Size::Vector,
     "b32"},
};

static std::optional<int64_t> evaluateSizeRule(VPTOMemorySizeRule rule,
                                               int64_t vectorBytes,
                                               unsigned elementBits) {
  if (vectorBytes <= 0) {
    return std::nullopt;
  }

  switch (rule) {
  case Size::Element:
    if (elementBits == 0 || elementBits % 8 != 0) {
      return std::nullopt;
    }
    return static_cast<int64_t>(elementBits / 8);
  case Size::Block:
    return 32;
  case Size::Vector:
    return vectorBytes;
  case Size::VectorTimes2:
    if (
        vectorBytes > std::numeric_limits<int64_t>::max() / 2) {
      return std::nullopt;
    }
    return vectorBytes * 2;
  case Size::VectorDiv2:
    return vectorBytes % 2 == 0 ? std::optional<int64_t>(vectorBytes / 2)
                                : std::nullopt;
  case Size::VectorDiv4:
    return vectorBytes % 4 == 0 ? std::optional<int64_t>(vectorBytes / 4)
                                : std::nullopt;
  case Size::VectorDiv8:
    return vectorBytes % 8 == 0 ? std::optional<int64_t>(vectorBytes / 8)
                                : std::nullopt;
  case Size::VectorDiv16:
    return vectorBytes % 16 == 0 ? std::optional<int64_t>(vectorBytes / 16)
                                 : std::nullopt;
  }
  return std::nullopt;
}

static llvm::StringRef getDefaultToken(VPTOMemoryOpFamily family,
                                       std::optional<unsigned> elementBits) {
  if (family == Family::Load) {
    return "NORM";
  }
  if (family != Family::Store || !elementBits) {
    return {};
  }
  switch (*elementBits) {
  case 8:
    return "NORM_B8";
  case 16:
    return "NORM_B16";
  case 32:
    return "NORM_B32";
  default:
    return {};
  }
}

} // namespace

std::optional<int64_t>
VPTOMemoryDistContract::getRequiredAlignmentBytes(int64_t vectorBytes) const {
  std::optional<int64_t> alignment =
      evaluateSizeRule(alignmentRule, vectorBytes, operandElementBits);
  if (!alignment) {
    return std::nullopt;
  }
  if (alignmentRule == Size::VectorDiv2 || alignmentRule == Size::VectorDiv4) {
    return std::min<int64_t>(32, *alignment);
  }
  return alignment;
}

std::optional<int64_t>
VPTOMemoryDistContract::getFullActiveFootprintBytes(int64_t vectorBytes) const {
  return evaluateSizeRule(footprintRule, vectorBytes, operandElementBits);
}

int64_t VPTOMemoryDistContract::getDependencyGranularityBytes(
    int64_t vectorBytes) const {
  if (transfer == Transfer::ScalarBroadcast) {
    return 32;
  }
  return getFullActiveFootprintBytes(vectorBytes).value_or(0);
}

bool VPTOMemoryDistContract::isOnePointStore() const {
  return dist == Dist::OnePointB8 || dist == Dist::OnePointB16 ||
         dist == Dist::OnePointB32;
}

const VPTOMemoryDistContract *
mlir::pto::lookupVPTOMemoryDist(VPTOMemoryOpFamily family,
                                llvm::StringRef token,
                                std::optional<unsigned> defaultElementBits) {
  llvm::StringRef normalizedToken = token;
  if (normalizedToken.empty()) {
    normalizedToken = getDefaultToken(family, defaultElementBits);
  }
  if (normalizedToken.empty()) {
    return nullptr;
  }

  for (const VPTOMemoryDistContract &contract : contracts) {
    if (contract.family != family || contract.token != normalizedToken) {
      continue;
    }
    return &contract;
  }
  return nullptr;
}

const VPTOMemoryDistContract *
mlir::pto::getVPTOMemoryDistContract(VPTOMemoryDist dist) {
  for (const VPTOMemoryDistContract &contract : contracts) {
    if (contract.dist == dist) {
      return &contract;
    }
  }
  return nullptr;
}
