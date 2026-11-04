// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

// A host-only model of the manual-address Tile descriptor API. This checks
// generated C++ value/alias behavior, not NPU instruction implementation.
#define __DAV_VEC__
#define AICORE
#define __gm__
#define __ubuf__

namespace pto {
inline void check(bool condition) {
  if (!condition) {
    throw std::runtime_error("Tile descriptor model check failed");
  }
}
enum class TileType { Vec };
enum class BLayout { RowMajor, ColMajor };
enum class SLayout { NoneBox };
enum class PadValue { Null, Zero };
enum class CompactMode { Null };
enum class TFillPadMode { InPlace };
enum class cache_line_t { SINGLE_CACHE_LINE };
enum { PIPE_ALL, PIPE_MTE3, PIPE_S, EVENT_ID0 };
inline void set_flag(int, int, int) {}
inline void wait_flag(int, int, int) {}
inline void pipe_barrier(int) {}
inline void set_mask_norm() {}
inline void set_vector_mask(int64_t, int64_t) {}
inline void dcci(void *, cache_line_t) {}

template <TileType Loc, typename T, int Rows, int Cols, BLayout Layout,
          int ValidRows, int ValidCols, SLayout SubLayout, int Fractal,
          PadValue Pad, CompactMode Compact>
struct Tile {
  using DType = T;
  static constexpr int rows = Rows;
  static constexpr int cols = Cols;
  T *pointer = nullptr;
  unsigned validRow = ValidRows < 0 ? 0 : ValidRows;
  unsigned validCol = ValidCols < 0 ? 0 : ValidCols;
  Tile() = default;
  Tile(unsigned row, unsigned col) { SetValidShape(row, col); }
  T *&data() { return pointer; }
  unsigned GetValidRow() const { return validRow; }
  unsigned GetValidCol() const { return validCol; }
  void SetValidShape(unsigned row, unsigned col) {
    pto::check(row <= Rows && col <= Cols);
    validRow = row;
    validCol = col;
  }
};

template <typename T, typename Address>
void TASSIGN(T &tile, Address address) {
  tile.data() = reinterpret_cast<typename T::DType *>(address);
}

template <typename Dst, typename Src>
void TRESHAPE(Dst &dst, Src &src) {
  // Like the manual-address PTO API, reshape binds storage; it does not
  // initialize the destination's dynamic valid extents.
  dst.data() = reinterpret_cast<typename Dst::DType *>(src.data());
}

struct FillPadObservation {
  void *data;
  unsigned rows;
  unsigned cols;
};

template <typename Dst, typename Lhs, typename Rhs>
void TADD(Dst &dst, Lhs &lhs, Rhs &rhs) {
  for (unsigned row = 0; row < dst.GetValidRow(); ++row) {
    for (unsigned col = 0; col < dst.GetValidCol(); ++col) {
      dst.data()[row * Dst::cols + col] =
          lhs.data()[row * Lhs::cols + col] + rhs.data()[row * Rhs::cols + col];
    }
  }
}
inline std::array<FillPadObservation, 8> fillpadCalls{};
inline std::size_t fillpadCount = 0;

template <TFillPadMode Mode, typename Dst, typename Src>
void TFILLPAD(Dst &dst, Src &src) {
  static_assert(Mode == TFillPadMode::InPlace);
  pto::check(dst.data() == src.data());
  pto::check(fillpadCount < fillpadCalls.size());
  // Observe the descriptor supplied to this consumer. NPU padding arithmetic
  // is deliberately outside this host descriptor model.
  fillpadCalls[fillpadCount++] =
      {src.data(), src.GetValidRow(), src.GetValidCol()};
}
} // namespace pto
