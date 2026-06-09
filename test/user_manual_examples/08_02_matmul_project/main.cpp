// Copyright (c) 2026 Huawei Technologies Co., Ltd.

#include "acl/acl.h"
#include <cstdio>
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

void LaunchMatmul_block(float *a, float *b, float *out, void *stream);

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
    constexpr size_t kM = 32;
    constexpr size_t kK = 32;
    constexpr size_t kN = 32;
    const size_t aSize = kM * kK * sizeof(float);
    const size_t bSize = kK * kN * sizeof(float);
    const size_t outSize = kM * kN * sizeof(float);

    float *hostOut = nullptr;
    float *hostA = nullptr;
    float *hostB = nullptr;
    float *devOut = nullptr;
    float *devA = nullptr;
    float *devB = nullptr;

    if (!CheckAcl("aclInit", aclInit(nullptr))) return 1;
    if (!CheckAcl("aclrtSetDevice", aclrtSetDevice(0))) {
        aclFinalize();
        return 1;
    }
    aclrtStream stream = nullptr;
    if (!CheckAcl("aclrtCreateStream", aclrtCreateStream(&stream))) {
        aclrtResetDevice(0);
        aclFinalize();
        return 1;
    }

    aclrtMallocHost((void **)(&hostOut), outSize);
    aclrtMallocHost((void **)(&hostA), aSize);
    aclrtMallocHost((void **)(&hostB), bSize);
    aclrtMalloc((void **)&devOut, outSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&devA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&devB, bSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadBinaryFile("./v1.bin", hostA, aSize);
    ReadBinaryFile("./v2.bin", hostB, bSize);

    aclrtMemcpy(devA, aSize, hostA, aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(devB, bSize, hostB, bSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchMatmul_block(devA, devB, devOut, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(hostOut, outSize, devOut, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteBinaryFile("./v4.bin", hostOut, outSize);

    aclrtFree(devA);
    aclrtFree(devB);
    aclrtFree(devOut);
    aclrtFreeHost(hostA);
    aclrtFreeHost(hostB);
    aclrtFreeHost(hostOut);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
