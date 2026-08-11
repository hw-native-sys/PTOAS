// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang tfillpad_inplace ST.
// Matches C++ reference test case: Case 5

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

using namespace PtoTestCommon;

// Kernel launch wrapper (defined in launch.cpp)
// Inplace kernel takes single buffer pointer
void LaunchTFILLPAD_INPLACE_f32_260x16_noexpand(float *buf, float *dummy, void *stream);

enum class DataType { F32 };

struct TestCase {
    const char *name;
    DataType    dtype;
    size_t      rows;
    size_t      cols;
    size_t      validRows;
    size_t      validCols;
    size_t      elemSize;
};

static const TestCase kCases[] = {
    // Case: float, fill cols 7..15 in place in a single buffer.
    {"f32_260x16_noexpand", DataType::F32,
     260, 16, 260, 7, sizeof(float)},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    size_t elemCount = tc.rows * tc.cols;
    size_t fileSize  = elemCount * tc.elemSize;

    std::printf("[INFO] === case: %s (%zux%zu, inplace) ===\n",
                tc.name, tc.validRows, tc.validCols);

    std::string caseDir = std::string("./") + tc.name;

    // Single buffer for inplace operation
    void *bufHost = nullptr;
    void *bufDevice = nullptr;

    aclrtMallocHost(&bufHost, fileSize);
    aclrtMalloc(&bufDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // Load input data into the single buffer
    if (!ReadFile((caseDir + "/input.bin").c_str(), fileSize, bufHost, fileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        // Copy input to device buffer
        aclrtMemcpy(bufDevice, fileSize, bufHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        // Run inplace kernel (src == dst = bufDevice)
        // Note: launch wrapper takes two args but inplace kernel uses same physical address
        LaunchTFILLPAD_INPLACE_f32_260x16_noexpand((float *)bufDevice, (float *)bufDevice, stream);

        aclrtSynchronizeStream(stream);
        // Copy result back (same buffer contains output)
        aclrtMemcpy(bufHost, fileSize, bufDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), bufHost, fileSize)) {
        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (bufDevice != nullptr)
        aclrtFree(bufDevice);
    if (bufHost != nullptr)
        aclrtFreeHost(bufHost);

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
