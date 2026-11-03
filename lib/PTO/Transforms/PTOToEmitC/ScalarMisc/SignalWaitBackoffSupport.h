// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_SCALAR_MISC_SIGNAL_WAIT_BACKOFF_SUPPORT_H
#define PTO_SCALAR_MISC_SIGNAL_WAIT_BACKOFF_SUPPORT_H

namespace mlir::pto {

inline constexpr const char* kSignalWaitBackoffSupport = R"cpp(
    template <uint64_t BackoffTicks, typename Signal>
    static AICORE inline void PTOAS__WAIT_GE_BACKOFF(Signal& signal, int32_t expected)
    {
        auto ptr = reinterpret_cast<volatile __gm__ int32_t*>(signal.data());
        uint32_t spin = 0;
        while (true) {
            __asm__ __volatile__("" ::: "memory");
            dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);
            __asm__ __volatile__("" ::: "memory");
            if (*ptr >= expected)
                break;
            const uint64_t wait_start = get_sys_cnt();
            while ((uint64_t)(get_sys_cnt() - wait_start) < BackoffTicks) {
                __asm__ __volatile__("" ::: "memory");
            }
            PTO_ASSERT(spin < 100000000, "local wait spin limit");
            if ((++spin & 63) == 0) {
                pipe_barrier(PIPE_ALL);
            }
        }
    }
)cpp";

} // namespace mlir::pto
#endif
