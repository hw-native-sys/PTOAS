// Copyright (c) 2026 Huawei Technologies Co., Ltd.

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

__global__ AICORE void flash_attention_qk_block(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3);

void LaunchMatmul_block(float *a, float *b, float *out, void *stream) {
    flash_attention_qk_block<<<1, nullptr, stream>>>(a, b, out);
}
