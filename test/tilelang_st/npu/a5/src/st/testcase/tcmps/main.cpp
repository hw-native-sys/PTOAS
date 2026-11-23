// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang tcmps ST — case-table driven.
// tcmps: dst = packed mask of (src < scalar).
// Numerical comparison is done externally by compare.py.

#include "acl/acl.h"
#include "test_common.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

using namespace PtoTestCommon;

// Kernel launch wrappers (defined in launch.cpp)
void LaunchTCMP_f32_1x64(float *src, uint8_t *dst, void *stream);
void LaunchTCMP_f32_4x64(float *src, uint8_t *dst, void *stream);
void LaunchTCMP_f32_8x64(float *src, uint8_t *dst, void *stream);
void LaunchTCMP_f32_32x64(float *src, uint8_t *dst, void *stream);
void LaunchTCMP_f32_128x128(float *src, uint8_t *dst, void *stream);
void LaunchTCMP_i32_16x32(int32_t *src, uint8_t *dst, void *stream);
void LaunchTCMP_i32_32x32(int32_t *src, uint8_t *dst, void *stream);
void LaunchTCMP_i32_32x64_valid32x64(int32_t *src, uint8_t *dst, void *stream);
void LaunchTCMP_f32_7x448(float *src, uint8_t *dst, void *stream);
void LaunchTCMP_f32_256x16(float *src, uint8_t *dst, void *stream);
void LaunchTCMP_i32_31x128(int32_t *src, uint8_t *dst, void *stream);
void LaunchTCMP_f16_32x128(uint16_t *src, uint8_t *dst, void *stream);
void LaunchTCMP_i16_32x128(int16_t *src, uint8_t *dst, void *stream);

struct TestCase {
    const char *name;
    void (*launch)(void *, void *, void *);  // src, dst, stream
    size_t      rows;       // allocated tile rows
    size_t      cols;       // allocated tile cols
    size_t      validRows;  // effective computation rows  (<= rows)
    size_t      validCols;  // effective computation cols  (<= cols)
    size_t      srcElemSize; // bytes per source element
    size_t      dstElemSize; // bytes per destination element
};

static size_t GetDstElemCount(const TestCase &tc) {
    // Match the GM destination views in tcmps.pto, including 256x16 -> 256x32.
    constexpr size_t kMinDstCols = 32;
    return tc.rows * std::max(tc.cols, kMinDstCols);
}

static const TestCase kCases[] = {
    {"f32_1x64",              (void (*)(void*,void*,void*))LaunchTCMP_f32_1x64,              1,   64,   1,   64,  sizeof(float),    sizeof(uint8_t)},
    {"f32_4x64",              (void (*)(void*,void*,void*))LaunchTCMP_f32_4x64,              4,   64,   4,   64,  sizeof(float),    sizeof(uint8_t)},
    {"f32_8x64",              (void (*)(void*,void*,void*))LaunchTCMP_f32_8x64,              8,   64,   8,   64,  sizeof(float),    sizeof(uint8_t)},
    {"f32_32x64",             (void (*)(void*,void*,void*))LaunchTCMP_f32_32x64,             32,  64,   32,  64,  sizeof(float),    sizeof(uint8_t)},
    {"f32_128x128",           (void (*)(void*,void*,void*))LaunchTCMP_f32_128x128,           128, 128,  128, 128, sizeof(float),    sizeof(uint8_t)},
    {"i32_16x32",             (void (*)(void*,void*,void*))LaunchTCMP_i32_16x32,             16,  32,   16,  32,  sizeof(int32_t),  sizeof(uint8_t)},
    {"i32_32x32",             (void (*)(void*,void*,void*))LaunchTCMP_i32_32x32,             32,  32,   32,  32,  sizeof(int32_t),  sizeof(uint8_t)},
    {"i32_32x64_valid32x64",  (void (*)(void*,void*,void*))LaunchTCMP_i32_32x64_valid32x64,  64,  64,   32,  64,  sizeof(int32_t),  sizeof(uint8_t)},
    {"f32_7x448",             (void (*)(void*,void*,void*))LaunchTCMP_f32_7x448,             7,   448,  7,   448, sizeof(float),    sizeof(uint8_t)},
    {"f32_256x16",            (void (*)(void*,void*,void*))LaunchTCMP_f32_256x16,            256, 16,   256, 16,  sizeof(float),    sizeof(uint8_t)},
    {"i32_31x128",            (void (*)(void*,void*,void*))LaunchTCMP_i32_31x128,            31,  128,  31,  128, sizeof(int32_t),  sizeof(uint8_t)},
    {"f16_32x128",            (void (*)(void*,void*,void*))LaunchTCMP_f16_32x128,            32,  128,  32,  128, sizeof(uint16_t), sizeof(uint8_t)},
    {"i16_32x128",            (void (*)(void*,void*,void*))LaunchTCMP_i16_32x128,            32,  128,  32,  128, sizeof(int16_t),  sizeof(uint8_t)},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    const size_t srcElemCount = tc.rows * tc.cols;
    const size_t dstElemCount = GetDstElemCount(tc);
    const size_t srcFileSize  = srcElemCount * tc.srcElemSize;
    const size_t dstFileSize  = dstElemCount * tc.dstElemSize;

    std::printf("[INFO] === case: %s (shape=%zux%zu, valid=%zux%zu) ===\n",
                tc.name, tc.rows, tc.cols, tc.validRows, tc.validCols);

    // Per-case data directory
    std::string caseDir = std::string("./") + tc.name;
    size_t inputFileSize = srcFileSize;

    void *srcHost = nullptr, *dstHost = nullptr;
    void *srcDevice = nullptr, *dstDevice = nullptr;

    aclrtMallocHost(&srcHost, srcFileSize);
    aclrtMallocHost(&dstHost, dstFileSize);

    aclrtMalloc(&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input1.bin").c_str(), inputFileSize, srcHost, srcFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch(srcDevice, dstDevice, stream);

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, dstFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (srcDevice != nullptr)
        aclrtFree(srcDevice);
    if (dstDevice != nullptr)
        aclrtFree(dstDevice);
    if (srcHost != nullptr)
        aclrtFreeHost(srcHost);
    if (dstHost != nullptr)
        aclrtFreeHost(dstHost);

    if (rc == 0)
        std::printf("[INFO] case %s done\n", tc.name);
    return rc;
}

int main(int argc, char *argv[]) {
    // Optional case filter: ./tcmps [case_name]
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
