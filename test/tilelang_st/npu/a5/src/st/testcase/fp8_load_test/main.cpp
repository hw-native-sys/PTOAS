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

void LaunchFP8_LOAD_16x64(uint8_t *src, uint8_t *dst, void *stream);
void LaunchFP8_LOAD_32x64(uint8_t *src, uint8_t *dst, void *stream);
void LaunchFP8_LOAD_128x64(uint8_t *src, uint8_t *dst, void *stream);

using LaunchFn = void (*)(uint8_t *, uint8_t *, void *);

struct TestCase {
    const char *name;
    size_t rows;
    size_t cols;
    LaunchFn launch;
};

static const TestCase kCases[] = {
    {"fp8_load_16x64", 16, 64, LaunchFP8_LOAD_16x64},
    {"fp8_load_32x64", 32, 64, LaunchFP8_LOAD_32x64},
    {"fp8_load_128x64", 128, 64, LaunchFP8_LOAD_128x64},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, aclrtStream stream) {
    int rc = 0;
    const size_t bytes = tc.rows * tc.cols * sizeof(uint8_t);

    std::printf("[INFO] === case: %s (%zux%zu) ===\n", tc.name, tc.rows, tc.cols);

    std::string caseDir = std::string("./") + tc.name;
    size_t inputFileSize = bytes;

    void *inputHost = nullptr;
    void *outputHost = nullptr;
    void *inputDevice = nullptr;
    void *outputDevice = nullptr;

    aclrtMallocHost(&inputHost, bytes);
    aclrtMallocHost(&outputHost, bytes);
    aclrtMalloc(&inputDevice, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&outputDevice, bytes, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input.bin").c_str(), inputFileSize, inputHost, bytes)) {
        std::fprintf(stderr, "[ERROR] read input failed\n");
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(inputDevice, bytes, inputHost, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemset(outputDevice, bytes, 0, bytes);

        tc.launch(static_cast<uint8_t *>(inputDevice), static_cast<uint8_t *>(outputDevice), stream);
        aclrtSynchronizeStream(stream);

        aclrtMemcpy(outputHost, bytes, outputDevice, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), outputHost, bytes)) {
        std::fprintf(stderr, "[ERROR] write output failed\n");
        rc = 1;
    }

    if (inputDevice != nullptr)
        aclrtFree(inputDevice);
    if (outputDevice != nullptr)
        aclrtFree(outputDevice);
    if (inputHost != nullptr)
        aclrtFreeHost(inputHost);
    if (outputHost != nullptr)
        aclrtFreeHost(outputHost);

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
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
        deviceId = std::atoi(envDevice);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    for (size_t i = 0; i < kNumCases; ++i) {
        if (caseFilter != nullptr && std::strcmp(kCases[i].name, caseFilter) != 0)
            continue;
        int ret = RunCase(kCases[i], stream);
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
