// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_IR_VPTOMEMORYDIST_H
#define PTO_IR_VPTOMEMORYDIST_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir::pto {

enum class VPTOMemoryOpFamily {
  Load,
  LoadX2,
  Store,
  StoreX2,
};

enum class VPTOMemoryDist {
  LoadNorm,
  BrcB8,
  BrcB16,
  BrcB32,
  UsB8,
  UsB16,
  DsB8,
  DsB16,
  BlockDeinterleave,
  DeinterleaveB8,
  DeinterleaveB16,
  DeinterleaveB32,
  UnpackB8,
  UnpackB16,
  UnpackB32,
  BroadcastBlock,
  ElementToBlockB16,
  ElementToBlockB32,
  Unpack4B8,
  Split4ChannelB8,
  Split2ChannelB8,
  Split2ChannelB16,
  StoreNormB8,
  StoreNormB16,
  StoreNormB32,
  OnePointB8,
  OnePointB16,
  OnePointB32,
  PackB16,
  PackB32,
  PackB64,
  InterleaveB8,
  InterleaveB16,
  InterleaveB32,
  Pack4B32,
  Merge4ChannelB8,
  Merge2ChannelB8,
  Merge2ChannelB16,
};

enum class VPTOPredicatePolicy {
  NotPresent,
  Applied,
  Ignored,
};

enum class VPTOMemoryTransferKind {
  Identity,
  ScalarBroadcast,
  Upsample2,
  Downsample2,
  BlockDeinterleave2,
  ElementDeinterleave2,
  LaneExpand2,
  BlockBroadcast,
  ElementToBlock,
  LaneExpand4,
  SplitChannel,
  Point,
  LaneCompact2,
  ElementInterleave2,
  LaneCompact4,
  MergeChannel,
};

enum class VPTOMemorySizeRule {
  Element,
  Block,
  Vector,
  VectorTimes2,
  VectorDiv2,
  VectorDiv4,
  VectorDiv8,
  VectorDiv16,
};

struct VPTOMemoryDistContract {
  VPTOMemoryOpFamily family;
  VPTOMemoryDist dist;
  llvm::StringRef token;
  unsigned operandElementBits;
  uint64_t a5Immediate;
  unsigned registerArity;
  VPTOPredicatePolicy predicatePolicy;
  VPTOMemoryTransferKind transfer;
  VPTOMemorySizeRule alignmentRule;
  VPTOMemorySizeRule footprintRule;
  llvm::StringRef maskGranularity;

  std::optional<int64_t> getRequiredAlignmentBytes(int64_t vectorBytes) const;

  std::optional<int64_t> getFullActiveFootprintBytes(int64_t vectorBytes) const;

  int64_t getDependencyGranularityBytes(int64_t vectorBytes) const;

  bool isOnePointStore() const;
};

/// Look up a dist token for an operation family. Empty load/store tokens denote
/// the corresponding NORM form, selected using `defaultElementBits` when the
/// family has width-specific defaults. An explicit token carries its own
/// physical element width; it is independent of the vreg carrier element type.
const VPTOMemoryDistContract *
lookupVPTOMemoryDist(VPTOMemoryOpFamily family, llvm::StringRef token,
                     std::optional<unsigned> defaultElementBits = std::nullopt);

const VPTOMemoryDistContract *getVPTOMemoryDistContract(VPTOMemoryDist dist);

} // namespace mlir::pto

#endif // PTO_IR_VPTOMEMORYDIST_H
