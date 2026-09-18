// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>

using namespace PtoTestCommon;

#define ACL_CHECK(expr)                                                                                    \
    do {                                                                                                   \
        const aclError _ret = (expr);                                                                      \
        if (_ret != ACL_SUCCESS) {                                                                         \
            std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr, (int)_ret, __FILE__, __LINE__); \
            return cleanup(1);                                                                             \
        }                                                                                                  \
    } while (0)

#define FILE_CHECK(expr)                                                                    \
    do {                                                                                    \
        if (!(expr)) {                                                                      \
            std::fprintf(stderr, "[ERROR] %s failed (%s:%d)\n", #expr, __FILE__, __LINE__); \
            return cleanup(1);                                                              \
        }                                                                                   \
    } while (0)

void LaunchVmi_vmula_bf16_vl8_unaligned_kernel(int16_t* acc, int16_t* lhs, int16_t* rhs, int16_t* dst, void* stream);

int main()
{
    constexpr size_t kElems = 16; // 32-byte staging image per operand (kernel copies 32 B)
    constexpr size_t kStagingBytes = kElems * sizeof(int16_t);
    constexpr size_t kOutputBytes = 8 * sizeof(int16_t);
    size_t fileSize = 0;
    int16_t* accHost = nullptr;
    int16_t* lhsHost = nullptr;
    int16_t* rhsHost = nullptr;
    int16_t* dstHost = nullptr;
    int16_t* accDevice = nullptr;
    int16_t* lhsDevice = nullptr;
    int16_t* rhsDevice = nullptr;
    int16_t* dstDevice = nullptr;
    bool aclInited = false;
    bool deviceSet = false;
    int deviceId = 0;
    aclrtStream stream = nullptr;

    const auto cleanup = [&](int result) {
        aclrtFree(accDevice);
        aclrtFree(lhsDevice);
        aclrtFree(rhsDevice);
        aclrtFree(dstDevice);
        aclrtFreeHost(accHost);
        aclrtFreeHost(lhsHost);
        aclrtFreeHost(rhsHost);
        aclrtFreeHost(dstHost);
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            aclrtResetDevice(deviceId);
        }
        if (aclInited) {
            aclFinalize();
        }
        return result;
    };

    const char* envDevice = std::getenv("ACL_DEVICE_ID");
    if (envDevice != nullptr) {
        const std::string_view deviceText(envDevice);
        const auto parseResult = std::from_chars(deviceText.data(), deviceText.data() + deviceText.size(), deviceId);
        const bool invalidDevice =
            parseResult.ec != std::errc() || parseResult.ptr != deviceText.data() + deviceText.size();
        if (invalidDevice) {
            std::fprintf(stderr, "[ERROR] invalid ACL_DEVICE_ID: %s\n", envDevice);
            return cleanup(1);
        }
    }
    ACL_CHECK(aclInit(nullptr));
    aclInited = true;
    ACL_CHECK(aclrtSetDevice(deviceId));
    deviceSet = true;
    ACL_CHECK(aclrtCreateStream(&stream));
    ACL_CHECK(aclrtMallocHost((void**)(&accHost), kStagingBytes));
    ACL_CHECK(aclrtMallocHost((void**)(&lhsHost), kStagingBytes));
    ACL_CHECK(aclrtMallocHost((void**)(&rhsHost), kStagingBytes));
    ACL_CHECK(aclrtMallocHost((void**)(&dstHost), kOutputBytes));
    ACL_CHECK(aclrtMalloc((void**)&accDevice, kStagingBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void**)&lhsDevice, kStagingBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void**)&rhsDevice, kStagingBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void**)&dstDevice, kOutputBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    fileSize = 0;
    FILE_CHECK(ReadFile("./v1.bin", fileSize, accHost, kStagingBytes) && fileSize == kStagingBytes);
    fileSize = 0;
    FILE_CHECK(ReadFile("./v2.bin", fileSize, lhsHost, kStagingBytes) && fileSize == kStagingBytes);
    fileSize = 0;
    FILE_CHECK(ReadFile("./v3.bin", fileSize, rhsHost, kStagingBytes) && fileSize == kStagingBytes);
    fileSize = 0;
    FILE_CHECK(ReadFile("./v4.bin", fileSize, dstHost, kOutputBytes) && fileSize == kOutputBytes);
    ACL_CHECK(aclrtMemcpy(accDevice, kStagingBytes, accHost, kStagingBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(lhsDevice, kStagingBytes, lhsHost, kStagingBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(rhsDevice, kStagingBytes, rhsHost, kStagingBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(dstDevice, kOutputBytes, dstHost, kOutputBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    LaunchVmi_vmula_bf16_vl8_unaligned_kernel(accDevice, lhsDevice, rhsDevice, dstDevice, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(dstHost, kOutputBytes, dstDevice, kOutputBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    FILE_CHECK(WriteFile("./v4.bin", dstHost, kOutputBytes));

    return cleanup(0);
}
