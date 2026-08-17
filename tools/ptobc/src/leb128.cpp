// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "ptobc/leb128.h"

#include <stdexcept>

namespace ptobc {

namespace {
constexpr unsigned kLeb128PayloadBits = 7;
constexpr uint8_t kLeb128PayloadMask = 0x7FU;
constexpr uint8_t kLeb128ContinuationBit = 0x80U;
constexpr uint8_t kLeb128SignBit = 0x40U;
constexpr unsigned kInt64MaxShift = 63;
constexpr unsigned kInt64BitWidth = 64;
} // namespace

void writeULEB128(uint64_t value, std::vector<uint8_t>& out) {
  do {
    uint8_t byte = static_cast<uint8_t>(value & kLeb128PayloadMask);
    value >>= kLeb128PayloadBits;
    if (value != 0) {
      byte |= kLeb128ContinuationBit;
    }
    out.push_back(byte);
  } while (value != 0);
}

void writeSLEB128(int64_t value, std::vector<uint8_t>& out) {
  bool more = true;
  while (more) {
    uint8_t byte = static_cast<uint8_t>(value & kLeb128PayloadMask);
    int64_t sign = byte & kLeb128SignBit;
    value >>= kLeb128PayloadBits;
    if ((value == 0 && sign == 0) || (value == -1 && sign != 0)) {
      more = false;
    } else {
      byte |= kLeb128ContinuationBit;
    }
    out.push_back(byte);
  }
}

size_t readULEB128(const uint8_t* data, size_t size, uint64_t& value) {
  value = 0;
  unsigned shift = 0;
  for (size_t i = 0; i < size; ++i) {
    uint8_t byte = data[i];
    value |= (uint64_t(byte & kLeb128PayloadMask) << shift);
    if ((byte & kLeb128ContinuationBit) == 0) {
      return i + 1;
    }
    shift += kLeb128PayloadBits;
    if (shift > kInt64MaxShift) {
      throw std::runtime_error("ULEB128 too large");
    }
  }
  throw std::runtime_error("Unexpected EOF in ULEB128");
}

size_t readSLEB128(const uint8_t* data, size_t size, int64_t& value) {
  value = 0;
  unsigned shift = 0;
  uint8_t byte = 0;
  size_t i = 0;
  for (; i < size; ++i) {
    byte = data[i];
    value |= (int64_t(byte & kLeb128PayloadMask) << shift);
    shift += kLeb128PayloadBits;
    if ((byte & kLeb128ContinuationBit) == 0) {
      break;
    }
    if (shift > kInt64MaxShift) {
      throw std::runtime_error("SLEB128 too large");
    }
  }
  if (i == size) {
    throw std::runtime_error("Unexpected EOF in SLEB128");
  }

  // sign extend
  if ((shift < kInt64BitWidth) && (byte & kLeb128SignBit)) {
    value |= static_cast<int64_t>(~uint64_t(0) << shift);
  }
  return i + 1;
}

} // namespace ptobc
