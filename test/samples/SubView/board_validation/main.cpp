// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "test_common.h"
#include "acl/acl.h"

#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

#define ACL_CHECK(expr)                                                                          \
    do {                                                                                         \
        const aclError _ret = (expr);                                                            \
        if (_ret != ACL_SUCCESS) {                                                               \
            std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr, (int)_ret, __FILE__, __LINE__); \
            const char *_recent = aclGetRecentErrMsg();                                          \
            if (_recent != nullptr && _recent[0] != '\0') {                                      \
                std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent);                     \
            }                                                                                     \
            rc = 1;                                                                               \
            goto cleanup;                                                                         \
        }                                                                                         \
    } while (0)

void LaunchSubViewSplit4_kernel(float *src, float *out0, float *out1, float *out2,
                                float *out3, void *stream);

int main() {
    size_t elem_src = 16 * 16;
    size_t elem_out = 8 * 8;
    size_t bytes_src = elem_src * sizeof(float);
    size_t bytes_out = elem_out * sizeof(float);

    float *srcHost = nullptr;
    float *out0Host = nullptr;
    float *out1Host = nullptr;
    float *out2Host = nullptr;
    float *out3Host = nullptr;

    float *srcDevice = nullptr;
    float *out0Device = nullptr;
    float *out1Device = nullptr;
    float *out2Device = nullptr;
    float *out3Device = nullptr;

    int rc = 0;
    bool aclInited = false;
    bool deviceSet = false;
    int deviceId = 0;
    aclrtStream stream = nullptr;

    ACL_CHECK(aclInit(nullptr));
    aclInited = true;
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) {
        deviceId = std::atoi(envDevice);
    }
    ACL_CHECK(aclrtSetDevice(deviceId));
    deviceSet = true;
    ACL_CHECK(aclrtCreateStream(&stream));

    ACL_CHECK(aclrtMallocHost((void **)(&srcHost), bytes_src));
    ACL_CHECK(aclrtMallocHost((void **)(&out0Host), bytes_out));
    ACL_CHECK(aclrtMallocHost((void **)(&out1Host), bytes_out));
    ACL_CHECK(aclrtMallocHost((void **)(&out2Host), bytes_out));
    ACL_CHECK(aclrtMallocHost((void **)(&out3Host), bytes_out));

    ACL_CHECK(aclrtMalloc((void **)&srcDevice, bytes_src, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&out0Device, bytes_out, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&out1Device, bytes_out, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&out2Device, bytes_out, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&out3Device, bytes_out, ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("./src.bin", bytes_src, srcHost, bytes_src);
    ReadFile("./out0.bin", bytes_out, out0Host, bytes_out);
    ReadFile("./out1.bin", bytes_out, out1Host, bytes_out);
    ReadFile("./out2.bin", bytes_out, out2Host, bytes_out);
    ReadFile("./out3.bin", bytes_out, out3Host, bytes_out);

    ACL_CHECK(aclrtMemcpy(srcDevice, bytes_src, srcHost, bytes_src,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(out0Device, bytes_out, out0Host, bytes_out,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(out1Device, bytes_out, out1Host, bytes_out,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(out2Device, bytes_out, out2Host, bytes_out,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(out3Device, bytes_out, out3Host, bytes_out,
                          ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchSubViewSplit4_kernel(srcDevice, out0Device, out1Device, out2Device,
                               out3Device, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtMemcpy(out0Host, bytes_out, out0Device, bytes_out,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(out1Host, bytes_out, out1Device, bytes_out,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(out2Host, bytes_out, out2Device, bytes_out,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(out3Host, bytes_out, out3Device, bytes_out,
                          ACL_MEMCPY_DEVICE_TO_HOST));

    WriteFile("./out0.bin", out0Host, bytes_out);
    WriteFile("./out1.bin", out1Host, bytes_out);
    WriteFile("./out2.bin", out2Host, bytes_out);
    WriteFile("./out3.bin", out3Host, bytes_out);

cleanup:
    aclrtFree(srcDevice);
    aclrtFree(out0Device);
    aclrtFree(out1Device);
    aclrtFree(out2Device);
    aclrtFree(out3Device);

    aclrtFreeHost(srcHost);
    aclrtFreeHost(out0Host);
    aclrtFreeHost(out1Host);
    aclrtFreeHost(out2Host);
    aclrtFreeHost(out3Host);

    if (stream != nullptr) {
        (void)aclrtDestroyStream(stream);
    }
    if (deviceSet) {
        (void)aclrtResetDevice(deviceId);
    }
    if (aclInited) {
        (void)aclFinalize();
    }

    return rc;
}
