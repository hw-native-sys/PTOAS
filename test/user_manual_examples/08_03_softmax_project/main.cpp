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

void LaunchSoftmax_block(float *scores, float *groupScale, float *softmax, void *stream);

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
    const size_t fileSize = 32 * 32 * sizeof(float);

    float *hostOut = nullptr;
    float *hostScores = nullptr;
    float *hostGroupScale = nullptr;
    float *devOut = nullptr;
    float *devScores = nullptr;
    float *devGroupScale = nullptr;

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

    aclrtMallocHost((void **)(&hostOut), fileSize);
    aclrtMallocHost((void **)(&hostScores), fileSize);
    aclrtMallocHost((void **)(&hostGroupScale), fileSize);
    aclrtMalloc((void **)&devOut, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&devScores, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&devGroupScale, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadBinaryFile("./v1.bin", hostScores, fileSize);
    ReadBinaryFile("./v2.bin", hostGroupScale, fileSize);
    aclrtMemcpy(devScores, fileSize, hostScores, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(devGroupScale, fileSize, hostGroupScale, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchSoftmax_block(devScores, devGroupScale, devOut, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(hostOut, fileSize, devOut, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteBinaryFile("./v3.bin", hostOut, fileSize);

    aclrtFree(devScores);
    aclrtFree(devGroupScale);
    aclrtFree(devOut);
    aclrtFreeHost(hostScores);
    aclrtFreeHost(hostGroupScale);
    aclrtFreeHost(hostOut);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
