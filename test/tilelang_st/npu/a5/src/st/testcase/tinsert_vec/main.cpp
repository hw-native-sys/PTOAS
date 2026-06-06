// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace PtoTestCommon;

void LaunchVec2VecND_f16_16x16_into_32x32_idx00(
    uint16_t *src, uint16_t *dst, uint16_t *out, void *stream);
void LaunchVec2VecND_f16_16x16_into_32x32_idx816(
    uint16_t *src, uint16_t *dst, uint16_t *out, void *stream);
void LaunchVec2VecND_f32_16x16_into_32x32_idx00(
    float *src, float *dst, float *out, void *stream);

using LaunchFn = void (*)(void *, void *, void *, void *);

struct TestCase {
    const char *name;
    LaunchFn    launch;
    size_t      src_rows, src_cols;
    size_t      dst_rows, dst_cols;
    size_t      idx_row, idx_col;
    bool        has_output;
    size_t      elem_bytes;
};

static const TestCase kCases[] = {
    {"vec2vec_nd_f16_16x16_into_32x32_idx00",
     reinterpret_cast<LaunchFn>(LaunchVec2VecND_f16_16x16_into_32x32_idx00),
     16, 16, 32, 32, 0, 0, true, 2},
    {"vec2vec_nd_f16_16x16_into_32x32_idx816",
     reinterpret_cast<LaunchFn>(LaunchVec2VecND_f16_16x16_into_32x32_idx816),
     16, 16, 32, 32, 8, 16, true, 2},
    {"vec2vec_nd_f32_16x16_into_32x32_idx00",
     reinterpret_cast<LaunchFn>(LaunchVec2VecND_f32_16x16_into_32x32_idx00),
     16, 16, 32, 32, 0, 0, true, 4},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    (void)deviceId;
    int rc = 0;
    const size_t srcElems = tc.src_rows * tc.src_cols;
    const size_t dstElems = tc.dst_rows * tc.dst_cols;
    const size_t srcBytes = srcElems * tc.elem_bytes;
    const size_t dstBytes = dstElems * tc.elem_bytes;
    size_t srcFileSize = srcBytes;
    size_t dstFileSize = dstBytes;

    std::printf("[INFO] === case: %s src=%zux%zu dst=%zux%zu idx=(%zu,%zu) ===\n",
                tc.name, tc.src_rows, tc.src_cols, tc.dst_rows, tc.dst_cols,
                tc.idx_row, tc.idx_col);

    std::string caseDir = std::string("./") + tc.name;

    void *srcHost = nullptr, *dstHost = nullptr, *outHost = nullptr;
    void *srcDev = nullptr, *dstDev = nullptr, *outDev = nullptr;

    aclrtMallocHost(&srcHost, srcBytes);
    aclrtMallocHost(&dstHost, dstBytes);
    aclrtMallocHost(&outHost, dstBytes);
    aclrtMalloc(&srcDev, srcBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dstDev, dstBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&outDev, dstBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input1.bin").c_str(), srcFileSize, srcHost, srcBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
        rc = 1;
    }
    if (rc == 0 && !ReadFile((caseDir + "/input2.bin").c_str(), dstFileSize, dstHost, dstBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input2.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(srcDev, srcBytes, srcHost, srcBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(dstDev, dstBytes, dstHost, dstBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemset(outDev, dstBytes, 0, dstBytes);

        tc.launch(srcDev, dstDev, outDev, stream);

        aclrtSynchronizeStream(stream);

        if (tc.has_output) {
            aclrtMemcpy(outHost, dstBytes, outDev, dstBytes, ACL_MEMCPY_DEVICE_TO_HOST);
            if (!WriteFile((caseDir + "/output.bin").c_str(), outHost, dstBytes)) {
                std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
                rc = 1;
            }
        }
    }

    if (srcDev) aclrtFree(srcDev);
    if (dstDev) aclrtFree(dstDev);
    if (outDev) aclrtFree(outDev);
    if (srcHost) aclrtFreeHost(srcHost);
    if (dstHost) aclrtFreeHost(dstHost);
    if (outHost) aclrtFreeHost(outHost);

    if (rc == 0)
        std::printf("[INFO] case %s done\n", tc.name);
    return rc;
}

int main(int argc, char *argv[]) {
    const char *caseFilter = (argc > 1) ? argv[1] : nullptr;

    int rc = 0;
    int deviceId = 0;
    aclrtStream stream = nullptr;

    aclInit(nullptr);
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) {
        deviceId = std::atoi(envDevice);
    }
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    for (size_t i = 0; i < kNumCases; ++i) {
        if (caseFilter != nullptr && std::strcmp(kCases[i].name, caseFilter) != 0) {
            continue;
        }
        int ret = RunCase(kCases[i], deviceId, stream);
        if (ret != 0) {
            std::fprintf(stderr, "[ERROR] case %s failed\n", kCases[i].name);
            rc = 1;
            break;
        }
    }

    if (stream != nullptr)
        aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return rc;
}