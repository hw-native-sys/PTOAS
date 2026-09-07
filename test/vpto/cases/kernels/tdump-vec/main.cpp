// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: kernels/tdump-vec
// family: kernels
// target_ops: pto.mte_gm_ub, pto.tdump
// scenarios: tdump-vec-f32-4x16, header-plus-row-major-payload
// -----------------------------------------------------------------------------
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
            if (_recent != nullptr && _recent[0] != '\0')                                        \
                std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent);                     \
            rc = 1;                                                                              \
            goto cleanup;                                                                        \
        }                                                                                        \
    } while (0)

void LaunchTdump_vec_kernel(float *v1, float *v2, void *stream);

int main() {
    size_t elemCountIn = 4 * 16;
    size_t fileSizeIn = elemCountIn * sizeof(float);
    // 64-byte header + 4x16 f32 payload.
    size_t fileSizeOut = 64 + elemCountIn * sizeof(float);
    float *v1Host = nullptr;
    float *v2Host = nullptr;
    float *v1Device = nullptr;
    float *v2Device = nullptr;
    aclrtStream stream = nullptr;

    int rc = 0;
    bool aclInited = false;
    bool deviceSet = false;
    int deviceId = 0;

    ACL_CHECK(aclInit(nullptr));
    aclInited = true;
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
        deviceId = std::atoi(envDevice);
    ACL_CHECK(aclrtSetDevice(deviceId));
    deviceSet = true;
    ACL_CHECK(aclrtCreateStream(&stream));

    ACL_CHECK(aclrtMallocHost((void **)(&v1Host), fileSizeIn));
    ACL_CHECK(aclrtMallocHost((void **)(&v2Host), fileSizeOut));
    ACL_CHECK(aclrtMalloc((void **)&v1Device, fileSizeIn, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v2Device, fileSizeOut, ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("./v1.bin", fileSizeIn, v1Host, fileSizeIn);
    ACL_CHECK(aclrtMemcpy(v1Device, fileSizeIn, v1Host, fileSizeIn, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchTdump_vec_kernel(v1Device, v2Device, stream);

    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(v2Host, fileSizeOut, v2Device, fileSizeOut, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./v2.bin", v2Host, fileSizeOut);

cleanup:
    aclrtFree(v1Device); aclrtFree(v2Device);
    aclrtFreeHost(v1Host); aclrtFreeHost(v2Host);
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    if (deviceSet) {
        aclrtResetDevice(deviceId);
    }
    if (aclInited) {
        aclFinalize();
    }
    return rc;
}
