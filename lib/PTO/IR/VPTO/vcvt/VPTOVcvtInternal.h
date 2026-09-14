// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVcvtInternal.h - shared VPTOVcvt helpers -----------------------===//
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Shared helpers. They live in a detail namespace so the original
// unqualified MLIR/LLVM names keep resolving.
// Internal to lib/PTO/IR/VPTO/vcvt; not installed.
//===----------------------------------------------------------------------===//

#ifndef PTO_IR_VPTO_VPTOVCVT_INTERNAL_H
#define PTO_IR_VPTO_VPTOVCVT_INTERNAL_H

#include "VPTOInternal.h"

namespace mlir::pto::vcvt_detail {

using namespace mlir;
using namespace mlir::pto;

  [[maybe_unused]] static std::optional<StringRef> normalizePacked4PartToken(StringRef token) {
    if (token == "P0" || token == "PART_P0") {
      return StringRef("P0");
    }
    if (token == "P1" || token == "PART_P1") {
      return StringRef("P1");
    }
    if (token == "P2" || token == "PART_P2") {
      return StringRef("P2");
    }
    if (token == "P3" || token == "PART_P3") {
      return StringRef("P3");
    }
    return std::nullopt;
  }

  [[maybe_unused]] static std::optional<StringRef> normalizeVcvtPartToken(StringRef token) {
    if (auto normalized = normalizeEvenOddPartToken(token)) {
      return normalized;
    }
    return normalizePacked4PartToken(token);
  }

  namespace {

  enum class VcvtElemKind {
    Invalid,
    F16,
    BF16,
    F32,
    F8E4M3,
    F8E5M2,
    HiF8,
    F4E1M2x2,
    F4E2M1x2,
    S8,
    U8,
    S16,
    U16,
    S32,
    U32,
    S64,
  };

  enum class VcvtPartFamily {
    EvenOdd,
    Packed4,
  };

  struct VcvtContract {
    bool requiresRnd;
    bool requiresSat;
    bool requiresPart;
    std::optional<VcvtPartFamily> partFamily = std::nullopt;
    const char *allowedRndModes = nullptr;
  };

  [[maybe_unused]] static VcvtElemKind classifyVcvtElemType(Type type) {
    if (type.isF16()) {
      return VcvtElemKind::F16;
    }
    if (type.isBF16()) {
      return VcvtElemKind::BF16;
    }
    if (type.isF32()) {
      return VcvtElemKind::F32;
    }
    if (pto::isPTOFloat8E4M3LikeType(type)) {
      return VcvtElemKind::F8E4M3;
    }
    if (pto::isPTOFloat8E5M2LikeType(type)) {
      return VcvtElemKind::F8E5M2;
    }
    if (pto::isPTOHiFloat8Type(type)) {
      return VcvtElemKind::HiF8;
    }
    if (isa<pto::F4E1M2x2Type>(type)) {
      return VcvtElemKind::F4E1M2x2;
    }
    if (isa<pto::F4E2M1x2Type>(type)) {
      return VcvtElemKind::F4E2M1x2;
    }
    if (auto intType = dyn_cast<IntegerType>(type)) {
      switch (intType.getWidth()) {
      case mlir::pto::kValue8:
        return intType.isUnsigned() ? VcvtElemKind::U8 : VcvtElemKind::S8;
      case mlir::pto::kValue16:
        return intType.isUnsigned() ? VcvtElemKind::U16 : VcvtElemKind::S16;
      case mlir::pto::kValue32:
        return intType.isUnsigned() ? VcvtElemKind::U32 : VcvtElemKind::S32;
      case mlir::pto::kValue64:
        return intType.isUnsigned() ? VcvtElemKind::Invalid : VcvtElemKind::S64;
      default:
        return VcvtElemKind::Invalid;
      }
    }
    return VcvtElemKind::Invalid;
  }

  [[maybe_unused]] static std::optional<unsigned> getVcvtElemBitWidth(VcvtElemKind kind) {
    switch (kind) {
    case VcvtElemKind::F16:
    case VcvtElemKind::BF16:
    case VcvtElemKind::S16:
    case VcvtElemKind::U16:
      return mlir::pto::kValue16;
    case VcvtElemKind::F32:
    case VcvtElemKind::S32:
    case VcvtElemKind::U32:
      return mlir::pto::kValue32;
    case VcvtElemKind::F8E4M3:
    case VcvtElemKind::F8E5M2:
    case VcvtElemKind::HiF8:
    case VcvtElemKind::F4E1M2x2:
    case VcvtElemKind::F4E2M1x2:
    case VcvtElemKind::S8:
    case VcvtElemKind::U8:
      return mlir::pto::kValue8;
    case VcvtElemKind::S64:
      return mlir::pto::kValue64;
    case VcvtElemKind::Invalid:
      return std::nullopt;
    }
    return std::nullopt;
  }

  [[maybe_unused]] static std::optional<VcvtPartFamily> classifyVcvtPartFamily(unsigned srcBits,
                                                              unsigned dstBits) {
    unsigned largerBits = std::max(srcBits, dstBits);
    unsigned smallerBits = std::min(srcBits, dstBits);
    if (largerBits == smallerBits * mlir::pto::kValue2) {
      return VcvtPartFamily::EvenOdd;
    }
    if (largerBits == smallerBits * mlir::pto::kValue4) {
      return VcvtPartFamily::Packed4;
    }
    return std::nullopt;
  }

  [[maybe_unused]] static bool isValidVcvtPartForFamily(StringRef part, VcvtPartFamily family) {
    switch (family) {
    case VcvtPartFamily::EvenOdd:
      return part == "EVEN" || part == "ODD";
    case VcvtPartFamily::Packed4:
      return part == "P0" || part == "P1" || part == "P2" || part == "P3";
    }
    return false;
  }

  [[maybe_unused]] static bool isValidVcvtRoundModeForContract(StringRef roundMode,
                                              const VcvtContract &contract) {
    if (!contract.allowedRndModes) {
      return true;
    }
    return StringRef(contract.allowedRndModes).contains(roundMode);
  }

  // Vcvt contract lookup helpers — one per source element kind.
  [[maybe_unused]] static std::optional<VcvtContract> lookupF32Contract(VcvtElemKind dst) {
    switch (dst) {
    case VcvtElemKind::F8E4M3:
    case VcvtElemKind::F8E5M2:
      return VcvtContract{true, true, true, VcvtPartFamily::Packed4, "RAHZ"};
    case VcvtElemKind::HiF8:
      return VcvtContract{true, true, true, VcvtPartFamily::Packed4, "AH"};
    case VcvtElemKind::F16:
    case VcvtElemKind::BF16:
    case VcvtElemKind::S16:
    case VcvtElemKind::S64:
      return VcvtContract{true, true, true};
    case VcvtElemKind::S32:
      return VcvtContract{true, true, false};
    default: return std::nullopt;
    }
  }

  [[maybe_unused]] static std::optional<VcvtContract> lookupF16Contract(VcvtElemKind dst) {
    switch (dst) {
    case VcvtElemKind::F8E4M3:
    case VcvtElemKind::F8E5M2:
      return VcvtContract{true, true, true, std::nullopt, "RAFZC"};
    case VcvtElemKind::HiF8:
      return VcvtContract{true, true, true, std::nullopt, "AH"};
    case VcvtElemKind::F32:
      return VcvtContract{false, false, true};
    case VcvtElemKind::S32:
      return VcvtContract{true, false, true};
    case VcvtElemKind::S16:
      return VcvtContract{true, true, false};
    case VcvtElemKind::S8:
    case VcvtElemKind::U8:
      return VcvtContract{true, true, true};
    case VcvtElemKind::BF16:
      return VcvtContract{true, false, false};
    default: return std::nullopt;
    }
  }

  [[maybe_unused]] static std::optional<VcvtContract> lookupBF16Contract(VcvtElemKind dst) {
    switch (dst) {
    case VcvtElemKind::F8E4M3:
    case VcvtElemKind::F8E5M2:
      return VcvtContract{true, true, true, std::nullopt, "RAFZC"};
    case VcvtElemKind::F4E1M2x2:
    case VcvtElemKind::F4E2M1x2:
      return VcvtContract{true, false, true, VcvtPartFamily::Packed4, "RAFZC"};
    case VcvtElemKind::F16:
      return VcvtContract{true, true, false};
    case VcvtElemKind::F32:
      return VcvtContract{false, false, true};
    case VcvtElemKind::S32:
      return VcvtContract{true, true, true};
    default: return std::nullopt;
    }
  }

  [[maybe_unused]] static std::optional<VcvtContract> lookupU8Contract(VcvtElemKind dst) {
    if (dst == VcvtElemKind::F16 || dst == VcvtElemKind::U16 ||
        dst == VcvtElemKind::U32) {
      return VcvtContract{false, false, true};
    }
    return std::nullopt;
  }

  [[maybe_unused]] static std::optional<VcvtContract> lookupS8Contract(VcvtElemKind dst) {
    if (dst == VcvtElemKind::F16 || dst == VcvtElemKind::S16 ||
        dst == VcvtElemKind::S32) {
      return VcvtContract{false, false, true};
    }
    return std::nullopt;
  }

  [[maybe_unused]] static std::optional<VcvtContract> lookupU16Contract(VcvtElemKind dst) {
    if (dst == VcvtElemKind::U8) {
      return VcvtContract{false, true, true};
    }
    if (dst == VcvtElemKind::U32) {
      return VcvtContract{false, false, true};
    }
    return std::nullopt;
  }

  [[maybe_unused]] static std::optional<VcvtContract> lookupS16Contract(VcvtElemKind dst) {
    if (dst == VcvtElemKind::F16) {
      return VcvtContract{true, false, false};
    }
    if (dst == VcvtElemKind::U8) {
      return VcvtContract{false, true, true};
    }
    if (dst == VcvtElemKind::F32 || dst == VcvtElemKind::U32 ||
        dst == VcvtElemKind::S32) {
      return VcvtContract{false, false, true};
    }
    return std::nullopt;
  }

  [[maybe_unused]] static std::optional<VcvtContract> lookupU32Contract(VcvtElemKind dst) {
    if (dst == VcvtElemKind::U8 || dst == VcvtElemKind::U16 ||
        dst == VcvtElemKind::S16) {
      return VcvtContract{false, true, true};
    }
    return std::nullopt;
  }

  [[maybe_unused]] static std::optional<VcvtContract> lookupS32Contract(VcvtElemKind dst) {
    if (dst == VcvtElemKind::F32) {
      return VcvtContract{true, false, false};
    }
    if (dst == VcvtElemKind::S64) {
      return VcvtContract{false, false, true};
    }
    if (dst == VcvtElemKind::U8 || dst == VcvtElemKind::U16 ||
        dst == VcvtElemKind::S16) {
      return VcvtContract{false, true, true};
    }
    return std::nullopt;
  }

  [[maybe_unused]] static std::optional<VcvtContract> lookupS64Contract(VcvtElemKind dst) {
    if (dst == VcvtElemKind::F32) {
      return VcvtContract{true, false, true};
    }
    if (dst == VcvtElemKind::S32) {
      return VcvtContract{false, true, true};
    }
    return std::nullopt;
  }

  [[maybe_unused]] static std::optional<VcvtContract> lookupVcvtContract(VcvtElemKind src,
                                                        VcvtElemKind dst) {
    switch (src) {
    case VcvtElemKind::F32: return lookupF32Contract(dst);
    case VcvtElemKind::F16: return lookupF16Contract(dst);
    case VcvtElemKind::BF16: return lookupBF16Contract(dst);
    case VcvtElemKind::U8: return lookupU8Contract(dst);
    case VcvtElemKind::S8: return lookupS8Contract(dst);
    case VcvtElemKind::U16: return lookupU16Contract(dst);
    case VcvtElemKind::S16: return lookupS16Contract(dst);
    case VcvtElemKind::U32: return lookupU32Contract(dst);
    case VcvtElemKind::S32: return lookupS32Contract(dst);
    case VcvtElemKind::S64: return lookupS64Contract(dst);
    case VcvtElemKind::F8E4M3:
    case VcvtElemKind::F8E5M2:
    case VcvtElemKind::HiF8:
      if (dst == VcvtElemKind::F32) { return VcvtContract{false, false, true, VcvtPartFamily::Packed4}; }
      return std::nullopt;
    case VcvtElemKind::F4E1M2x2:
    case VcvtElemKind::F4E2M1x2:
      if (dst == VcvtElemKind::BF16) { return VcvtContract{false, false, true, VcvtPartFamily::Packed4}; }
      return std::nullopt;
    default: return std::nullopt;
    }
  }

  } // namespace

  [[maybe_unused]] static StringRef getVcvtMaskGranularityByWidth(unsigned elemBits) {
    unsigned maskBitWidth = std::min(elemBits, 32u);
    if (maskBitWidth == mlir::pto::kValue8) {
      return "b8";
    }
    if (maskBitWidth == mlir::pto::kValue16) {
      return "b16";
    }
    if (maskBitWidth == mlir::pto::kValue32) {
      return "b32";
    }
    return "";
  }

  [[maybe_unused]] static LogicalResult verifyVcvtMaskGranularity(VcvtOp op, Type maskType,
                                                 VcvtElemKind inputElemKind,
                                                 VcvtElemKind resultElemKind) {
    auto inputElemBits = getVcvtElemBitWidth(inputElemKind);
    auto resultElemBits = getVcvtElemBitWidth(resultElemKind);
    if (!inputElemBits || !resultElemBits) {
      return op.emitOpError("could not determine vcvt element bit width");
    }
    StringRef expectedMaskGranularity = getVcvtMaskGranularityByWidth(*inputElemBits);
    if (expectedMaskGranularity.empty()) {
      return op.emitOpError("could not determine vcvt mask granularity");
    }
    return verifyMaskTypeWithGranularityLike(op, maskType, "mask type",
                                             expectedMaskGranularity);
  }

  [[maybe_unused]] static LogicalResult verifyVcvtTotalElementBits(VcvtOp op, Type inputType,
                                                  Type resultType,
                                                  VcvtElemKind inputElemKind,
                                                  VcvtElemKind resultElemKind) {
    auto inputVTy = cast<VRegType>(inputType);
    auto resultVTy = cast<VRegType>(resultType);
    auto inputElemBits = getVcvtElemBitWidth(inputElemKind);
    auto resultElemBits = getVcvtElemBitWidth(resultElemKind);
    if (!inputElemBits || !resultElemBits) {
      return op.emitOpError("could not determine vcvt element bit width");
    }
    if (inputVTy.getElementCount() * static_cast<int64_t>(*inputElemBits) !=
        resultVTy.getElementCount() * static_cast<int64_t>(*resultElemBits)) {
      return op.emitOpError("requires source and result vectors to carry the same "
                            "total number of bits");
    }
    return success();
  }

  [[maybe_unused]] static LogicalResult verifyVcvtRndAttr(VcvtOp op, const VcvtContract &contract) {
    if (op.getRndAttr()) {
      StringRef roundMode = *op.getRnd();
      auto normalizedRoundMode = normalizeRoundModeToken(roundMode);
      if (!normalizedRoundMode) {
        return op.emitOpError("rnd must be one of R/A/F/C/Z/O/H");
      }
      if (!isValidVcvtRoundModeForContract(*normalizedRoundMode, contract)) {
        return op.emitOpError("rnd attr is not valid for this vcvt type pair");
      }
    }
    if (static_cast<bool>(op.getRndAttr()) != contract.requiresRnd) {
      if (contract.requiresRnd) {
        return op.emitOpError("requires rnd attr for this vcvt type pair");
      }
      return op.emitOpError("rnd attr is not valid for this vcvt type pair");
    }
    return success();
  }

  [[maybe_unused]] static LogicalResult verifyVcvtSatAttr(VcvtOp op, const VcvtContract &contract) {
    if (op.getSatAttr()) {
      StringRef sat = *op.getSat();
      if (!normalizeSaturationToken(sat)) {
        return op.emitOpError("sat must be SAT or NOSAT");
      }
    }
    if (static_cast<bool>(op.getSatAttr()) != contract.requiresSat) {
      if (contract.requiresSat) {
        return op.emitOpError("requires sat attr for this vcvt type pair");
      }
      return op.emitOpError("sat attr is not valid for this vcvt type pair");
    }
    return success();
  }

  [[maybe_unused]] static LogicalResult verifyVcvtPartAttr(VcvtOp op, const VcvtContract &contract,
                                          VcvtElemKind inputElemKind,
                                          VcvtElemKind resultElemKind) {
    if (op.getPartAttr()) {
      StringRef part = *op.getPart();
      auto normalizedPart = normalizeVcvtPartToken(part);
      if (!normalizedPart) {
        return op.emitOpError("part must be one of EVEN/ODD/P0/P1/P2/P3");
      }
      std::optional<VcvtPartFamily> partFamily = contract.partFamily;
      if (!partFamily) {
        auto inputElemBits = getVcvtElemBitWidth(inputElemKind);
        auto resultElemBits = getVcvtElemBitWidth(resultElemKind);
        if (inputElemBits && resultElemBits) {
          partFamily = classifyVcvtPartFamily(*inputElemBits, *resultElemBits);
        }
      }
      if (!partFamily) {
        return op.emitOpError("part attr is not supported for this vcvt width relation");
      }
      if (!isValidVcvtPartForFamily(*normalizedPart, *partFamily)) {
        if (*partFamily == VcvtPartFamily::EvenOdd) {
          return op.emitOpError("part must be EVEN or ODD for 8/16 and 16/32 vcvt forms");
        }
        return op.emitOpError("part must be P0, P1, P2, or P3 for packed vcvt forms");
      }
    }
    if (static_cast<bool>(op.getPartAttr()) != contract.requiresPart) {
      if (contract.requiresPart) {
        return op.emitOpError("requires part attr for this vcvt type pair");
      }
      return op.emitOpError("part attr is not valid for this vcvt type pair");
    }
    return success();
  }


} // namespace mlir::pto::vcvt_detail

#endif // PTO_IR_VPTO_VPTOVCVT_INTERNAL_H
