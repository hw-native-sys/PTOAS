// Copyright (c) 2026 Huawei Technologies Co., Ltd.

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

__global__ AICORE void gqa_softmax_block(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3);

void LaunchSoftmax_block(float *scores, float *groupScale, float *softmax, void *stream) {
    gqa_softmax_block<<<1, nullptr, stream>>>(scores, groupScale, softmax);
}
