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

using LaunchFn = void (*)(void *, void *, void *, void *, void *, void *, void *);

void LaunchTQUANT_MX_DN_fp8_ocp_64x64(void *, void *, void *, void *, void *, void *, void *);
void LaunchTQUANT_MX_DN_fp8_ocp_128x32(void *, void *, void *, void *, void *, void *, void *);
void LaunchTQUANT_MX_DN_fp8_nv_64x64(void *, void *, void *, void *, void *, void *, void *);
void LaunchTQUANT_MX_DN_fp4_bf16_ocp_64x64(void *, void *, void *, void *, void *, void *, void *);
void LaunchTQUANT_MX_DN_fp8_ocp_64x96(void *, void *, void *, void *, void *, void *, void *);
void LaunchTQUANT_MX_ND_fp8_ocp_16x128(void *, void *, void *, void *, void *, void *, void *);
void LaunchTQUANT_MX_ND_fp8_ocp_20x128(void *, void *, void *, void *, void *, void *, void *);

struct TestCase {
    const char *name;
    size_t m;
    size_t n;
    bool bf16;
    bool fp4;
    LaunchFn launch;
};

static const TestCase kCases[] = {
    {"dn_fp8_ocp_64x64", 64, 64, false, false, LaunchTQUANT_MX_DN_fp8_ocp_64x64},
    {"dn_fp8_ocp_128x32", 128, 32, false, false, LaunchTQUANT_MX_DN_fp8_ocp_128x32},
    {"dn_fp8_nv_64x64", 64, 64, false, false, LaunchTQUANT_MX_DN_fp8_nv_64x64},
    {"dn_fp4_bf16_ocp_64x64", 64, 64, true, true, LaunchTQUANT_MX_DN_fp4_bf16_ocp_64x64},
    {"dn_fp8_ocp_64x96", 64, 96, false, false, LaunchTQUANT_MX_DN_fp8_ocp_64x96},
    {"nd_fp8_f32_ocp_16x128", 16, 128, false, false, LaunchTQUANT_MX_ND_fp8_ocp_16x128},
    {"nd_fp8_f32_ocp_20x128", 20, 128, false, false, LaunchTQUANT_MX_ND_fp8_ocp_20x128},
};

static std::string getCaseRoot() {
    if (const char *root = std::getenv("TILELANG_ST_CASE_ROOT"))
        return root;
    return ".";
}

static bool readInput(const std::string &path, void *buffer, size_t bytes) {
    size_t fileSize = bytes;
    return ReadFile(path.c_str(), fileSize, buffer, bytes);
}

static int runCase(const TestCase &tc, aclrtStream stream) {
    const size_t srcBytes = tc.m * tc.n * (tc.bf16 ? 2 : 4);
    const size_t dstBytes = tc.m * tc.n / (tc.fp4 ? 2 : 1);
    const size_t groupCount = tc.m * tc.n / 32;
    const size_t expBytes = groupCount;
    const size_t auxBytes = groupCount * (tc.bf16 ? 2 : 4);
    const size_t expZzBytes = std::strncmp(tc.name, "nd_", 3) == 0
                                  ? ((tc.m + 15) / 16 * 16) * (tc.n / 32)
                                  : groupCount;
    const size_t sizes[] = {srcBytes, dstBytes, expBytes, auxBytes, auxBytes, expZzBytes};
    void *host[6] = {};
    void *device[6] = {};
    int rc = 0;

    std::printf("[INFO] === case: %s (%zux%zu, %s, %s) ===\n", tc.name, tc.m, tc.n,
                tc.bf16 ? "bf16" : "f32", tc.fp4 ? "MXFP4" : "MXFP8");

    for (size_t i = 0; i < 6; ++i) {
        if (aclrtMallocHost(&host[i], sizes[i]) != ACL_SUCCESS ||
            aclrtMalloc(&device[i], sizes[i], ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            std::fprintf(stderr, "[ERROR] allocation failed for %s buffer %zu\n", tc.name, i);
            rc = 1;
            goto cleanup;
        }
        std::memset(host[i], 0, sizes[i]);
    }

    {
        const std::string caseDir = getCaseRoot() + "/" + tc.name;
        if (!readInput(caseDir + "/input.bin", host[0], srcBytes)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input.bin\n", caseDir.c_str());
            rc = 1;
            goto cleanup;
        }
        if (aclrtMemcpy(device[0], srcBytes, host[0], srcBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            rc = 1;
            goto cleanup;
        }
        for (size_t i = 1; i < 6; ++i) {
            if (aclrtMemcpy(device[i], sizes[i], host[i], sizes[i],
                            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                rc = 1;
                goto cleanup;
            }
        }

        tc.launch(device[0], device[1], device[2], device[3], device[4], device[5], stream);
        if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            rc = 1;
            goto cleanup;
        }
        for (size_t i = 1; i < 6; ++i) {
            if (aclrtMemcpy(host[i], sizes[i], device[i], sizes[i],
                            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                rc = 1;
                goto cleanup;
            }
        }

        const char *names[] = {"", "dst", "exp", "max", "scaling", "exp_zz"};
        for (size_t i = 1; i < 6; ++i) {
            const std::string output = caseDir + "/output_" + names[i] + ".bin";
            if (!WriteFile(output.c_str(), host[i], sizes[i])) {
                std::fprintf(stderr, "[ERROR] failed to write %s\n", output.c_str());
                rc = 1;
                goto cleanup;
            }
        }
    }

cleanup:
    for (size_t i = 0; i < 6; ++i) {
        if (device[i] != nullptr)
            aclrtFree(device[i]);
        if (host[i] != nullptr)
            aclrtFreeHost(host[i]);
    }
    return rc;
}

int main(int argc, char **argv) {
    const char *filter = argc > 1 ? argv[1] : nullptr;
    int deviceId = 0;
    if (const char *value = std::getenv("ACL_DEVICE_ID"))
        deviceId = std::atoi(value);
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(deviceId) != ACL_SUCCESS)
        return 1;

    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS)
        return 1;
    int rc = 0;
    bool matched = filter == nullptr;
    for (const TestCase &tc : kCases) {
        if (filter != nullptr && std::strcmp(filter, tc.name) != 0)
            continue;
        matched = true;
        if (runCase(tc, stream) != 0) {
            rc = 1;
            break;
        }
    }
    if (!matched) {
        std::fprintf(stderr, "[ERROR] unknown case: %s\n", filter);
        rc = 1;
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return rc;
}
