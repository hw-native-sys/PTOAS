// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "ptobc_opcodes_v0.h"

namespace ptobc::v0 {
namespace {

struct FullNameEntry {
  const char *name;
  OpcodeAndVariant value;
};

struct VariantOperandEntry {
  uint16_t opcode;
  uint8_t variant;
  int operandCount;
};

inline constexpr FullNameEntry kVariantNames[] = {
    {"pto.section.cube", {0x0006, 1, 0}},
    {"pto.section.vector", {0x0006, 1, 1}},
    {"pto.tgemv", {0x102A, 1, 0}},
    {"pto.tgemv.acc", {0x102A, 1, 1}},
    {"pto.tgemv.bias", {0x102A, 1, 2}},
    {"pto.tgemv.mx", {0x102A, 1, 3}},
    {"pto.tmatmul", {0x1032, 1, 0}},
    {"pto.tmatmul.acc", {0x1032, 1, 1}},
    {"pto.tmatmul.bias", {0x1032, 1, 2}},
    {"pto.tmatmul.mx", {0x1033, 1, 0}},
    {"pto.tmatmul.mx.acc", {0x1033, 1, 1}},
    {"pto.tmatmul.mx.bias", {0x1033, 1, 2}},
};

inline constexpr VariantOperandEntry kVariantOperands[] = {
    {0x102A, 0, 3}, {0x102A, 1, 4}, {0x102A, 2, 4}, {0x102A, 3, 5},
    {0x1032, 0, 3}, {0x1032, 1, 4}, {0x1032, 2, 4},
    {0x1033, 0, 5}, {0x1033, 1, 6}, {0x1033, 2, 6},
};

} // namespace

const OpInfo *lookupByOpcode(uint16_t opcode) {
  size_t lo = 0;
  size_t hi = sizeof(kOpTable) / sizeof(kOpTable[0]);
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    uint16_t value = kOpTable[mid].opcode;
    if (value == opcode)
      return &kOpTable[mid];
    if (value < opcode)
      lo = mid + 1;
    else
      hi = mid;
  }
  return nullptr;
}

std::optional<uint16_t> lookupOpcodeByName(llvm::StringRef name) {
  for (const OpInfo &info : kOpTable) {
    if (name == info.name)
      return info.opcode;
  }
  return std::nullopt;
}

const OpInfo *lookupByName(llvm::StringRef name) {
  auto opcode = lookupOpcodeByName(name);
  if (!opcode)
    return nullptr;
  return lookupByOpcode(*opcode);
}

std::optional<OpcodeAndVariant>
lookupOpcodeAndVariantByFullName(llvm::StringRef fullName) {
  for (const FullNameEntry &entry : kVariantNames) {
    if (fullName == entry.name)
      return entry.value;
  }

  auto opcode = lookupOpcodeByName(fullName);
  if (!opcode)
    return std::nullopt;
  return OpcodeAndVariant{*opcode, 0, 0};
}

const char *fullNameFromOpcodeVariant(uint16_t opcode, uint8_t variant) {
  const OpInfo *info = lookupByOpcode(opcode);
  if (!info)
    return nullptr;
  if (!info->has_variant_u8)
    return info->name;

  for (const FullNameEntry &entry : kVariantNames) {
    if (entry.value.opcode == opcode && entry.value.variant == variant)
      return entry.name;
  }
  return info->name;
}

std::optional<int> lookupOperandsByVariant(uint16_t opcode, uint8_t variant) {
  for (const VariantOperandEntry &entry : kVariantOperands) {
    if (entry.opcode == opcode && entry.variant == variant)
      return entry.operandCount;
  }
  return std::nullopt;
}

} // namespace ptobc::v0
