// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang tload_mat ST — case-table driven.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

using namespace PtoTestCommon;

// Kernel launch wrappers (defined in launch.cpp)
// All cases: dst(f32) + x1 + x2 (3 GM pointers, dst always float32 for ACC output)
void LaunchTLOAD_MAT_f16_nd2nz(float *dst, float *x1, float *x2, void *stream);
void LaunchTLOAD_MAT_bf16_nd2nz(float *dst, void *x1, void *x2, void *stream);
void LaunchTLOAD_MAT_f32_nd2nz(float *dst, float *x1, float *x2, void *stream);
void LaunchTLOAD_MAT_f16_dn2nz(float *dst, float *x1, float *x2, void *stream);
void LaunchTLOAD_MAT_bf16_dn2nz(float *dst, void *x1, void *x2, void *stream);
void LaunchTLOAD_MAT_f32_dn2nz(float *dst, float *x1, float *x2, void *stream);

using LaunchFn3Float = void (*)(float *, float *, float *, void *);
using LaunchFn3Void  = void (*)(void *, void *, void *, void *);

struct TestCase {
    const char *name;
    void *launch;
    size_t M, N, K;
    size_t x1ElemSize;   // sizeof src dtype
    size_t x2ElemSize;   // sizeof src dtype
    size_t dstElemSize;  // sizeof dst dtype (always 4 = f32)
    bool dstIsFp32;      // true if dst uses float* launch signature
};

static const TestCase kCases[] = {
    {"f16_nd2nz",  (void*)LaunchTLOAD_MAT_f16_nd2nz,  16, 32, 16, 2, 2, 4, true},
    {"bf16_nd2nz", (void*)LaunchTLOAD_MAT_bf16_nd2nz, 16, 32, 16, 2, 2, 4, true},
    {"f32_nd2nz",  (void*)LaunchTLOAD_MAT_f32_nd2nz,  16, 32, 16, 4, 4, 4, true},
    {"f16_dn2nz",  (void*)LaunchTLOAD_MAT_f16_dn2nz,  16, 32, 16, 2, 2, 4, true},
    {"bf16_dn2nz", (void*)LaunchTLOAD_MAT_bf16_dn2nz, 16, 32, 16, 2, 2, 4, true},
    {"f32_dn2nz",  (void*)LaunchTLOAD_MAT_f32_dn2nz,  16, 32, 16, 4, 4, 4, true},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    const size_t x1Count = tc.M * tc.K;
    const size_t x2Count = tc.K * tc.N;
    const size_t dstCount = tc.M * tc.N;
    const size_t x1Size = x1Count * tc.x1ElemSize;
    const size_t x2Size = x2Count * tc.x2ElemSize;
    const size_t dstSize = dstCount * tc.dstElemSize;

    std::printf("[INFO] === case: %s (M=%zu,N=%zu,K=%zu) ===\n",
                tc.name, tc.M, tc.N, tc.K);

    std::string caseDir = std::string("./") + tc.name;

    void *x1Host = nullptr, *x2Host = nullptr, *dstHost = nullptr;
    void *x1Dev = nullptr, *x2Dev = nullptr, *dstDev = nullptr;

    aclrtMallocHost(&x1Host, x1Size);
    aclrtMallocHost(&x2Host, x2Size);
    aclrtMallocHost(&dstHost, dstSize);

    aclrtMalloc(&x1Dev, x1Size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&x2Dev, x2Size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dstDev, dstSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t x1SizeVar = x1Size, x2SizeVar = x2Size;

    if (!ReadFile((caseDir + "/x1_gm.bin").c_str(), x1SizeVar, x1Host, x1Size)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/x1_gm.bin\n", caseDir.c_str());
        rc = 1;
    }
    if (rc == 0 && !ReadFile((caseDir + "/x2_gm.bin").c_str(), x2SizeVar, x2Host, x2Size)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/x2_gm.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(x1Dev, x1Size, x1Host, x1Size, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(x2Dev, x2Size, x2Host, x2Size, ACL_MEMCPY_HOST_TO_DEVICE);

        // All cases have float dst (ACC output)
        if (tc.dstIsFp32) {
            LaunchFn3Float launch = (LaunchFn3Float)tc.launch;
            launch((float*)dstDev, (float*)x1Dev, (float*)x2Dev, stream);
        } else {
            LaunchFn3Void launch = (LaunchFn3Void)tc.launch;
            launch(dstDev, x1Dev, x2Dev, stream);
        }

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(dstHost, dstSize, dstDev, dstSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, dstSize)) {
        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (x1Dev) aclrtFree(x1Dev);
    if (x2Dev) aclrtFree(x2Dev);
    if (dstDev) aclrtFree(dstDev);
    if (x1Host) aclrtFreeHost(x1Host);
    if (x2Host) aclrtFreeHost(x2Host);
    if (dstHost) aclrtFreeHost(dstHost);

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

    if (stream) aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return rc;
}
