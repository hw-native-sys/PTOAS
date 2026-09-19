// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Declaration-only consumer for codegen compilation. No private compatibility
// entry points or synchronization implementations are available to generated C++.
#ifndef PTOAS_TEST_PUBLIC_SYNC_HPP
#define PTOAS_TEST_PUBLIC_SYNC_HPP

#include <cstdint>

#define AICORE
#define __global__
#define __gm__

enum pipe_t { PIPE_S, PIPE_V, PIPE_MTE1, PIPE_MTE2, PIPE_MTE3, PIPE_FIX, PIPE_ALL };
enum event_t { EVENT_ID0 };
enum class cache_line_t { SINGLE_CACHE_LINE };

namespace pto {
#if !defined(PTOAS_TEST_A5)
constexpr uint16_t FFTS_MODE_VAL = 2;
#endif
uint16_t getFFTSMsg(uint16_t mode, uint16_t event, uint16_t baseCount = 1);
} // namespace pto

void set_ffts_base_addr(uint64_t base);
void ffts_cross_core_sync(pipe_t pipe, uint16_t message);
#if defined(PTOAS_TEST_A5)
void wait_flag_dev(pipe_t pipe, int64_t event);
#else
void wait_flag_dev(int32_t event);
#endif
void set_intra_block(pipe_t pipe, uint64_t event);
void wait_intra_block(pipe_t pipe, uint64_t event);
void set_flag(pipe_t producer, pipe_t consumer, event_t event);
void wait_flag(pipe_t producer, pipe_t consumer, event_t event);
void pipe_barrier(pipe_t pipe);
void dcci(void *address, cache_line_t mode);
void set_mask_norm();
void set_vector_mask(uint64_t high, uint64_t low);

#endif
