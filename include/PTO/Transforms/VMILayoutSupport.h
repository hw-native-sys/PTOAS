// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMILayoutSupport.h - VMI layout support queries ------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_VMILAYOUTSUPPORT_H
#define PTO_TRANSFORMS_VMILAYOUTSUPPORT_H

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::pto {

struct VMILoadLayoutFact {
  VMILayoutAttr resultLayout;
};

enum class VMIDeinterleaveLoadLayoutPort {
  Low,
  High,
};

struct VMIDeinterleaveLoadLayoutFact {
  VMILayoutAttr lowLayout;
  VMILayoutAttr highLayout;
};

struct VMIStoreLayoutFact {
  VMILayoutAttr valueLayout;
};

struct VMIMaskedStoreLayoutFact {
  VMILayoutAttr valueLayout;
  VMILayoutAttr maskLayout;
};

struct VMIMaskedLoadLayoutFact {
  VMILayoutAttr resultLayout;
  VMILayoutAttr maskLayout;
  VMILayoutAttr passthruLayout;
};

struct VMIEnsureLayoutFact {
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
};

struct VMIEnsureMaskLayoutFact {
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
};

enum class VMICastLayoutPort {
  Source,
  Result,
};

enum class VMICastLayoutPriority {
  Normal,
  High,
  LaneStrideNarrowing,
};

enum class VMIInterleaveLayoutPort {
  Lhs,
  Rhs,
  Mask,
  Low,
  High,
};

struct VMICastLayoutFact {
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
  int64_t sourceBits = 0;
  int64_t resultBits = 0;
  VMICastLayoutPriority priority = VMICastLayoutPriority::Normal;
};

struct VMIMaskGranularityCastLayoutFact {
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
  int64_t sourceGranularityBits = 0;
  int64_t resultGranularityBits = 0;
};

struct VMIInterleaveLayoutFact {
  VMILayoutAttr lhsLayout;
  VMILayoutAttr rhsLayout;
  VMILayoutAttr maskLayout;
  VMILayoutAttr lowLayout;
  VMILayoutAttr highLayout;
  int64_t elementCount = 0;
  int64_t lanesPerPart = 0;
};

struct VMIBitcastLayoutFact {
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
};

enum class VMIGroupBlockClass {
  QuarterBlock,
  HalfBlock,
  OneBlock,
  TwoBlock,
  FourBlock,
  FullPartMultiple,
};

struct VMIGroupStoreLayoutFact {
  VMILayoutAttr valueLayout;
  VMIGroupBlockClass blockClass = VMIGroupBlockClass::OneBlock;
  int64_t groupSize = 0;
  int64_t lanesPerPart = 0;
  int64_t vcgBlockElems = 0;
};

struct VMIGroupReduceLayoutFact {
  VMIGroupBlockClass blockClass = VMIGroupBlockClass::OneBlock;
  VMILayoutAttr sourceLayout;
  VMILayoutAttr maskLayout;
  VMILayoutAttr resultLayout;
  int64_t groupSize = 0;
  int64_t lanesPerPart = 0;
  int64_t vcgBlockElems = 0;
};

struct VMIGroupBroadcastLayoutFact {
  VMIGroupBlockClass blockClass = VMIGroupBlockClass::OneBlock;
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
  int64_t groupSize = 0;
  int64_t lanesPerPart = 0;
  int64_t vcgBlockElems = 0;
};

enum class VMIGroupBroadcastLoadDirectKind {
  E2B,
  BRC,
};

struct VMIGroupBroadcastLoadLayoutFact {
  VMIGroupBlockClass blockClass = VMIGroupBlockClass::OneBlock;
  VMILayoutAttr resultLayout;
  int64_t groupSize = 0;
  int64_t lanesPerPart = 0;
  int64_t vcgBlockElems = 0;
  int64_t elementBits = 0;
};

struct VMIGroupBroadcastLoadDirectFact {
  VMIGroupBroadcastLoadDirectKind kind = VMIGroupBroadcastLoadDirectKind::E2B;
  VMIGroupBroadcastLoadLayoutFact layout;
};

struct VMIGroupLoadLayoutFact {
  VMIGroupBlockClass blockClass = VMIGroupBlockClass::TwoBlock;
  VMILayoutAttr resultLayout;
  int64_t groupSize = 0;
};

struct VMIGroupSlotLayoutFact {
  VMILayoutAttr layout;
  int64_t numGroups = 0;
  int64_t slots = 0;
};

enum class VMIGroupReduceLayoutPort {
  Source,
  Mask,
  Result,
};

enum class VMIGroupBroadcastLayoutPort {
  Source,
  Result,
};

struct VMIHistogramLayoutFact {
  VMILayoutAttr accLayout;
  VMILayoutAttr sourceLayout;
  VMILayoutAttr maskLayout;
  VMILayoutAttr resultLayout;
};

struct VMIVselrLayoutFact {
  VMILayoutAttr sourceLayout;
  VMILayoutAttr indexLayout;
  VMILayoutAttr resultLayout;
};

class VMILayoutSupport {
public:
  FailureOr<VMILoadLayoutFact>
  getLoadLayoutFact(VMIVRegType resultType,
                    std::string *reason = nullptr) const;

  FailureOr<VMIDeinterleaveLoadLayoutFact>
  getPreferredDeinterleaveLoadLayoutFact(
      VMIVRegType valueType, std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMIDeinterleaveLoadLayoutFact, mlir::pto::kValue4>>
  getDeinterleaveLoadLayoutFactsForLayout(
      VMIVRegType valueType, VMIDeinterleaveLoadLayoutPort port,
      VMILayoutAttr layout, std::string *reason = nullptr) const;

  FailureOr<VMIDeinterleaveLoadLayoutFact>
  getDeinterleaveLoadLayoutFactForLayouts(
      VMIVRegType lowType, VMIVRegType highType,
      std::string *reason = nullptr) const;

  FailureOr<VMIStoreLayoutFact>
  getStoreLayoutFact(VMIVRegType valueType,
                     std::string *reason = nullptr) const;

  FailureOr<VMIStoreLayoutFact>
  getPreferredStoreLayoutFact(VMIVRegType valueType,
                              std::string *reason = nullptr) const;

  FailureOr<VMIMaskedStoreLayoutFact>
  getMaskedStoreLayoutFact(VMIVRegType valueType, VMIMaskType maskType,
                           std::string *reason = nullptr) const;

  FailureOr<VMIMaskedStoreLayoutFact>
  getPreferredMaskedStoreLayoutFact(VMIVRegType valueType,
                                    VMIMaskType maskType,
                                    std::string *reason = nullptr) const;

  FailureOr<VMIMaskedLoadLayoutFact>
  getMaskedLoadLayoutFact(VMIVRegType resultType, VMIMaskType maskType,
                          VMIVRegType passthruType,
                          std::string *reason = nullptr) const;

  FailureOr<VMIEnsureLayoutFact>
  getEnsureLayoutFact(VMIVRegType sourceType, VMIVRegType resultType,
                      std::string *reason = nullptr) const;

  FailureOr<VMIEnsureMaskLayoutFact>
  getEnsureMaskLayoutFact(VMIMaskType sourceType, VMIMaskType resultType,
                          std::string *reason = nullptr) const;

  FailureOr<VMICastLayoutFact>
  getPreferredCastLayoutFact(VMIVRegType sourceType, VMIVRegType resultType,
                             std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMICastLayoutFact, mlir::pto::kValue4>>
  getCastLayoutFactsForLayout(VMIVRegType sourceType, VMIVRegType resultType,
                              VMICastLayoutPort port, VMILayoutAttr layout,
                              std::string *reason = nullptr) const;

  FailureOr<VMICastLayoutFact> getCastLayoutFactForSourceLayout(
      VMIVRegType sourceType, VMIVRegType resultType,
      VMILayoutAttr sourceLayout, std::string *reason = nullptr) const;

  FailureOr<VMICastLayoutFact> getCastLayoutFactForResultLayout(
      VMIVRegType sourceType, VMIVRegType resultType,
      VMILayoutAttr resultLayout, std::string *reason = nullptr) const;

  FailureOr<VMICastLayoutFact> getCastLayoutFactForLayouts(
      VMIVRegType sourceType, VMIVRegType resultType, VMILayoutAttr sourceLayout,
      VMILayoutAttr resultLayout, std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMIMaskGranularityCastLayoutFact, mlir::pto::kValue4>>
  getMaskGranularityCastLayoutFactsForLayout(
      VMIMaskType sourceType, VMIMaskType resultType, VMICastLayoutPort port,
      VMILayoutAttr layout, std::string *reason = nullptr) const;

  FailureOr<VMIMaskGranularityCastLayoutFact>
  getMaskGranularityCastLayoutFactForLayouts(
      VMIMaskType sourceType, VMIMaskType resultType,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      std::string *reason = nullptr) const;

  FailureOr<VMILayoutAttr> getWidenSourceLayoutForResultLayout(
      VMIVRegType sourceType, VMIVRegType resultType,
      VMILayoutAttr requestedResultLayout, std::string *reason = nullptr) const;

  FailureOr<VMIInterleaveLayoutFact>
  getPreferredVintlvLayoutFact(VMIVRegType valueType,
                               std::string *reason = nullptr) const;

  FailureOr<VMIInterleaveLayoutFact>
  getPreferredVdintlvLayoutFact(VMIVRegType valueType,
                                std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMIInterleaveLayoutFact, mlir::pto::kValue4>>
  getVintlvLayoutFactsForLayout(VMIVRegType valueType,
                                VMIInterleaveLayoutPort port,
                                VMILayoutAttr layout,
                                std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMIInterleaveLayoutFact, mlir::pto::kValue4>>
  getVdintlvLayoutFactsForLayout(VMIVRegType valueType,
                                 VMIInterleaveLayoutPort port,
                                 VMILayoutAttr layout,
                                 std::string *reason = nullptr) const;

  FailureOr<VMIInterleaveLayoutFact> getVintlvLayoutFactForLayouts(
      VMIVRegType lhsType, VMIVRegType rhsType, VMIMaskType maskType,
      VMIVRegType lowType, VMIVRegType highType,
      std::string *reason = nullptr) const;

  FailureOr<VMIInterleaveLayoutFact> getVdintlvLayoutFactForLayouts(
      VMIVRegType lhsType, VMIVRegType rhsType, VMIMaskType maskType,
      VMIVRegType lowType, VMIVRegType highType,
      std::string *reason = nullptr) const;

  FailureOr<VMIGroupSlotLayoutFact>
  getGroupSlotLoadLayoutFact(VMIVRegType resultType, int64_t numGroups,
                             std::string *reason = nullptr) const;

  FailureOr<VMIGroupLoadLayoutFact>
  getGroupLoadLayoutFact(VMIGroupLoadOp op,
                         std::string *reason = nullptr) const;
  FailureOr<VMIGroupLoadLayoutFact>
  getGroupLoadLayoutFact(VMIVRegType resultType, Value rowStride,
                         int64_t numGroups,
                         std::string *reason = nullptr) const;

  FailureOr<VMIGroupSlotLayoutFact>
  getGroupStoreLayoutFact(VMIVRegType valueType, int64_t numGroups,
                          std::string *reason = nullptr) const;

  FailureOr<VMIGroupStoreLayoutFact>
  getGroupStoreLayoutFact(VMIGroupStoreOp op, VMIVRegType valueType,
                          std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMIGroupStoreLayoutFact, mlir::pto::kValue4>>
  getGroupStoreLayoutFactsForLayout(VMIGroupStoreOp op,
                                    VMIVRegType valueType,
                                    VMILayoutAttr layout,
                                    std::string *reason = nullptr) const;

  FailureOr<VMIGroupStoreLayoutFact>
  getPreferredGroupStoreLayoutFact(VMIGroupStoreOp op, VMIVRegType valueType,
                                   std::string *reason = nullptr) const;

  FailureOr<VMIGroupStoreLayoutFact>
  getHighPriorityGroupStoreLayoutFact(VMIGroupStoreOp op,
                                      VMIVRegType valueType,
                                      std::string *reason = nullptr) const;

  FailureOr<VMIGroupReduceLayoutFact>
  getPreferredGroupReduceLayoutFact(VMIVRegType sourceType, int64_t numGroups,
                                    std::string *reason = nullptr) const;

  FailureOr<VMIGroupReduceLayoutFact> getGroupReduceLayoutFactForLayouts(
      VMIVRegType sourceType, VMIMaskType maskType, VMIVRegType resultType,
      int64_t numGroups, std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMIGroupReduceLayoutFact, mlir::pto::kValue4>>
  getGroupReduceLayoutFactsForLayout(VMIVRegType sourceType,
                                     int64_t numGroups,
                                     VMIGroupReduceLayoutPort port,
                                     VMILayoutAttr layout,
                                     std::string *reason = nullptr) const;

  FailureOr<VMIGroupBroadcastLayoutFact>
  getGroupBroadcastLayoutFactForLayouts(VMIVRegType sourceType,
                                        VMIVRegType resultType,
                                        int64_t numGroups,
                                        std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMIGroupBroadcastLayoutFact, mlir::pto::kValue4>>
  getGroupBroadcastLayoutFactsForLayout(VMIVRegType sourceType,
                                        VMIVRegType resultType,
                                        int64_t numGroups,
                                        VMIGroupBroadcastLayoutPort port,
                                        VMILayoutAttr layout,
                                        std::string *reason = nullptr) const;

  FailureOr<VMIGroupBroadcastLoadLayoutFact>
  getGroupBroadcastLoadLayoutFact(VMIGroupBroadcastLoadOp op,
                                  std::string *reason = nullptr) const;
  FailureOr<VMIGroupBroadcastLoadLayoutFact>
  getGroupBroadcastLoadLayoutFact(VMIVRegType resultType,
                                  Value sourceGroupStride, int64_t numGroups,
                                  std::string *reason = nullptr) const;
  FailureOr<VMIGroupBroadcastLoadDirectFact> getGroupBroadcastLoadDirectFact(
      VMIGroupBroadcastLoadOp op, std::string *reason = nullptr) const;
  FailureOr<VMIGroupBroadcastLoadDirectFact> getGroupBroadcastLoadDirectFact(
      VMIVRegType resultType, Type sourceType, Value sourceGroupStride,
      int64_t numGroups, std::string *reason = nullptr) const;

  FailureOr<VMIHistogramLayoutFact>
  getVdhistLayoutFact(VMIVdhistOp op, std::string *reason = nullptr) const;

  FailureOr<VMIHistogramLayoutFact>
  getVchistLayoutFact(VMIVchistOp op, std::string *reason = nullptr) const;

  FailureOr<VMIVselrLayoutFact>
  getPreferredVselrLayoutFact(VMIVselrOp op,
                              std::string *reason = nullptr) const;

  FailureOr<VMIVselrLayoutFact>
  getVselrLayoutFact(VMIVselrOp op,
                     std::string *reason = nullptr) const;

  LogicalResult getVselrSupport(VMIVselrOp op,
                                std::string *reason = nullptr) const;

  LogicalResult getGroupReduceAddFSupport(VMIGroupReduceAddFOp op,
                                          std::string *reason = nullptr) const;

  LogicalResult getGroupReduceMaxFSupport(VMIGroupReduceMaxFOp op,
                                          std::string *reason = nullptr) const;

  LogicalResult getGroupReduceMinFSupport(VMIGroupReduceMinFOp op,
                                          std::string *reason = nullptr) const;

  LogicalResult getGroupReduceAddISupport(VMIGroupReduceAddIOp op,
                                          std::string *reason = nullptr) const;

  LogicalResult getGroupReduceMaxISupport(VMIGroupReduceMaxIOp op,
                                          std::string *reason = nullptr) const;

  LogicalResult getGroupReduceMinISupport(VMIGroupReduceMinIOp op,
                                          std::string *reason = nullptr) const;

  LogicalResult getGroupBroadcastSupport(VMIGroupBroadcastOp op,
                                         std::string *reason = nullptr) const;

  LogicalResult getGroupBroadcastSupport(VMIVRegType sourceType,
                                         VMIVRegType resultType,
                                         int64_t numGroups,
                                         std::string *reason = nullptr) const;

  LogicalResult getGroupBroadcastLoadSupport(
      VMIGroupBroadcastLoadOp op, std::string *reason = nullptr) const;

  LogicalResult getTruncFSupport(VMITruncFOp op,
                                 std::string *reason = nullptr) const;

  LogicalResult getExtFSupport(VMIExtFOp op,
                               std::string *reason = nullptr) const;

  LogicalResult getExtSISupport(VMIExtSIOp op,
                                std::string *reason = nullptr) const;

  LogicalResult getExtUISupport(VMIExtUIOp op,
                                std::string *reason = nullptr) const;

  LogicalResult getTruncISupport(VMITruncIOp op,
                                 std::string *reason = nullptr) const;

  FailureOr<VMIBitcastLayoutFact>
  getBitcastLayoutFact(VMIBitcastOp op,
                       std::string *reason = nullptr) const;

  FailureOr<SmallVector<VMIBitcastLayoutFact, mlir::pto::kValue4>>
  getBitcastLayoutFactsForLayout(VMIVRegType sourceType,
                                 VMIVRegType resultType,
                                 VMICastLayoutPort port,
                                 VMILayoutAttr layout,
                                 std::string *reason = nullptr) const;

  LogicalResult getBitcastSupport(VMIBitcastOp op,
                                  std::string *reason = nullptr) const;

  LogicalResult getVdhistSupport(VMIVdhistOp op,
                                std::string *reason = nullptr) const;

  LogicalResult getVchistSupport(VMIVchistOp op,
                                std::string *reason = nullptr) const;
};

} // namespace mlir::pto

#endif // PTO_TRANSFORMS_VMILAYOUTSUPPORT_H
