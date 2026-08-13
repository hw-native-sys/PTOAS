// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Minimal A5 C2V local-tile bridge for TPush/TPop/TFREE experiments.
// This TU is intentionally compiled by Bisheng to device bitcode and linked
// with VPTO LLVM IR; it is not part of the host PTOAS build.
#include <pto/pto-inst.hpp>
#include <pto/npu/a5/TFree.hpp>
#include <pto/npu/a5/TPop.hpp>
#include <pto/npu/a5/TPush.hpp>
#include <stddef.h>

[aicore] inline void *operator new(size_t, void *ptr) noexcept { return ptr; }
#include <stdint.h>

using Pipe = pto::TPipe<0, pto::Direction::DIR_C2V, 1024, 8, 2, false>;
using AccTile = pto::Tile<pto::TileType::Acc, float, 16, 16,
                          pto::BLayout::ColMajor, 16, 16,
                          pto::SLayout::RowMajor, 1024>;
using VecTile = pto::Tile<pto::TileType::Vec, float, 8, 16,
                          pto::BLayout::RowMajor, 8, 16>;

extern "C" [aicore] void pto_vpto_pipe_init(
    void *storage, uint32_t c2vConsumerBuffer) {
  new (storage) Pipe(nullptr, c2vConsumerBuffer, 0);
}

extern "C" [aicore] void pto_vpto_pipe_finish(void *storage) {
  reinterpret_cast<Pipe *>(storage)->~Pipe();
}

#ifdef __DAV_CUBE__
extern "C" [aicore] void pto_vpto_pipe_push(void *storage, uint64_t accAddress) {
  auto &pipe = *reinterpret_cast<Pipe *>(storage);
  AccTile tile;
  pto::TASSIGN_IMPL(tile, accAddress);
  pto::TPUSH<Pipe, AccTile, pto::TileSplitAxis::TILE_UP_DOWN>(pipe, tile);
}

#endif

#ifdef __DAV_VEC__
extern "C" [aicore] uint64_t pto_vpto_pipe_pop(void *storage) {
  auto &pipe = *reinterpret_cast<Pipe *>(storage);
  VecTile tile;
  // Keep the C2V ready dependency visible at the bridge boundary. TPOP also
  // performs this wait internally, but the explicit intrinsic prevents the
  // standalone wrapper bitcode from being scheduled ahead of the producer's
  // fixpipe completion after llvm-link.
#ifdef __DAV_VEC__
  wait_intra_block(PIPE_V, 0);
  pipe.cons.setWaitStatus(false);
#endif
  pto::TPOP<Pipe, VecTile, pto::TileSplitAxis::TILE_UP_DOWN>(pipe, tile);
  pipe_barrier(PIPE_ALL);
  return reinterpret_cast<uint64_t>(tile.data());
}

extern "C" [aicore] void pto_vpto_pipe_free(void *storage) {
  auto &pipe = *reinterpret_cast<Pipe *>(storage);
  pto::TFREE<Pipe, pto::TileSplitAxis::TILE_UP_DOWN>(pipe);
}
#endif
