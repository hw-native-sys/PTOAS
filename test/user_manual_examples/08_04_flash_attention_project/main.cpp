// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>

namespace {
void ReadBinaryFile(const char *path, void *buffer, size_t size) {
    std::ifstream ifs(path, std::ios::binary);
    ifs.read(static_cast<char *>(buffer), static_cast<std::streamsize>(size));
}

void WriteBinaryFile(const char *path, const void *buffer, size_t size) {
    std::ofstream ofs(path, std::ios::binary);
    ofs.write(static_cast<const char *>(buffer), static_cast<std::streamsize>(size));
}
}  // namespace

void LaunchFlash_attention_qk_block(float *v1, float *v2, float *v3, void *stream);
void LaunchFlash_attention_softmax_block(float *v1, float *v2, void *stream);
void LaunchFlash_attention_sv_block(float *v1, float *v2, float *v3, void *stream);

namespace {
bool CheckAcl(const char *name, aclError status) {
    if (status != ACL_ERROR_NONE) {
        std::fprintf(stderr, "%s failed: %d\n", name, static_cast<int>(status));
        return false;
    }
    return true;
}
}

int main() {
        size_t fileSize = 32 * 32 * sizeof(float);
    float *dstHost0 = nullptr;
    float *dstDevice0 = nullptr;
    float *srcHost0 = nullptr;
    float *srcHost1 = nullptr;
    float *srcHost2 = nullptr;
    float *srcHost3 = nullptr;
    float *srcHost4 = nullptr;
    float *srcDevice0 = nullptr;
    float *srcDevice1 = nullptr;
    float *srcDevice2 = nullptr;
    float *srcDevice3 = nullptr;
    float *srcDevice4 = nullptr;

    if (!CheckAcl("aclInit", aclInit(nullptr))) return 1;
    if (!CheckAcl("aclrtSetDevice", aclrtSetDevice(0))) {
        aclFinalize();
        return 1;
    }
    aclrtStream stream;
    if (!CheckAcl("aclrtCreateStream", aclrtCreateStream(&stream))) {
        aclrtResetDevice(0);
        aclFinalize();
        return 1;
    }

        aclrtMallocHost((void **)(&dstHost0), fileSize);
    aclrtMallocHost((void **)(&srcHost0), fileSize);
    aclrtMallocHost((void **)(&srcHost1), fileSize);
    aclrtMallocHost((void **)(&srcHost2), fileSize);
    aclrtMallocHost((void **)(&srcHost3), fileSize);
    aclrtMallocHost((void **)(&srcHost4), fileSize);
        aclrtMalloc((void **)&dstDevice0, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice0, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice1, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice2, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice3, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice4, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

        ReadBinaryFile("./v1.bin", srcHost0, fileSize);
    ReadBinaryFile("./v2.bin", srcHost1, fileSize);
    ReadBinaryFile("./v3.bin", srcHost2, fileSize);
    ReadBinaryFile("./v4.bin", srcHost3, fileSize);
    ReadBinaryFile("./v5.bin", srcHost4, fileSize);
        aclrtMemcpy(srcDevice0, fileSize, srcHost0, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice1, fileSize, srcHost1, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice2, fileSize, srcHost2, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice3, fileSize, srcHost3, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice4, fileSize, srcHost4, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        LaunchFlash_attention_qk_block(srcDevice0, srcDevice1, srcDevice3, stream);
    LaunchFlash_attention_softmax_block(srcDevice3, srcDevice4, stream);
    LaunchFlash_attention_sv_block(srcDevice4, srcDevice2, dstDevice0, stream);

    aclrtSynchronizeStream(stream);
        aclrtMemcpy(dstHost0, fileSize, dstDevice0, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

        WriteBinaryFile("./v6.bin", dstHost0, fileSize);

        aclrtFree(srcDevice0);
    aclrtFree(srcDevice1);
    aclrtFree(srcDevice2);
    aclrtFree(srcDevice3);
    aclrtFree(srcDevice4);
    aclrtFree(dstDevice0);
        aclrtFreeHost(srcHost0);
    aclrtFreeHost(srcHost1);
    aclrtFreeHost(srcHost2);
    aclrtFreeHost(srcHost3);
    aclrtFreeHost(srcHost4);
    aclrtFreeHost(dstHost0);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
