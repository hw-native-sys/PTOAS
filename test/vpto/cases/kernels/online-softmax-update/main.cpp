// -----------------------------------------------------------------------------
// case: kernels/online-softmax-update
// family: kernels
// target_ops: pto.get_block_idx, pto.copy_gm_to_ubuf, pto.copy_ubuf_to_gm, pto.vlds, pto.vcmax, pto.vdup, pto.vmax, pto.vexpdiff, pto.vcadd, pto.vadd, pto.vmul, pto.vdiv, pto.vsts
// scenarios: online-softmax-update, dynamic-rows-and-seq, max-seq-128, block-rows-8, oldmax-oldsum-qk-to-newmax-newsum-expmax-out
// -----------------------------------------------------------------------------
#include "test_common.h"
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

#ifndef TMRGSORT_HPP
namespace pto {
struct MrgSortExecutedNumList {
    uint16_t mrgSortList0;
    uint16_t mrgSortList1;
    uint16_t mrgSortList2;
    uint16_t mrgSortList3;
};
} // namespace pto
#endif

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

void LaunchOnline_softmax_update_kernel_2d(float *v1, float *v2, float *v3,
                                           float *v4, float *v5, float *v6,
                                           float *v7, int32_t v8, int32_t v9,
                                           void *stream);

int main() {
    constexpr size_t elemCountSeq = 1;
    constexpr size_t elemCountRows = 1;
    size_t fileSizeSeq = elemCountSeq * sizeof(int32_t);
    size_t fileSizeRows = elemCountRows * sizeof(int32_t);
    size_t elemCountState = 0;
    size_t elemCountOut = 0;
    size_t fileSizeState = 0;
    size_t fileSizeOut = 0;
    float *v1Host = nullptr, *v2Host = nullptr, *v3Host = nullptr;
    float *v4Host = nullptr, *v5Host = nullptr, *v6Host = nullptr;
    float *v7Host = nullptr;
    float *v1Device = nullptr, *v2Device = nullptr, *v3Device = nullptr;
    float *v4Device = nullptr, *v5Device = nullptr, *v6Device = nullptr;
    float *v7Device = nullptr;
    int32_t v8Host = 0, v9Host = 0;

    int rc = 0;
    bool aclInited = false;
    bool deviceSet = false;
    int deviceId = 0;
    aclrtStream stream = nullptr;

    ACL_CHECK(aclInit(nullptr));
    aclInited = true;
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
        deviceId = std::atoi(envDevice);
    ACL_CHECK(aclrtSetDevice(deviceId));
    deviceSet = true;
    ACL_CHECK(aclrtCreateStream(&stream));

    ReadFile("./v8.bin", fileSizeSeq, &v8Host, fileSizeSeq);
    ReadFile("./v9.bin", fileSizeRows, &v9Host, fileSizeRows);

    elemCountState = static_cast<size_t>(v9Host);
    elemCountOut = static_cast<size_t>(v9Host) * 128;
    fileSizeState = elemCountState * sizeof(float);
    fileSizeOut = elemCountOut * sizeof(float);

    ACL_CHECK(aclrtMallocHost((void **)(&v1Host), fileSizeState));
    ACL_CHECK(aclrtMallocHost((void **)(&v2Host), fileSizeState));
    ACL_CHECK(aclrtMallocHost((void **)(&v3Host), fileSizeOut));
    ACL_CHECK(aclrtMallocHost((void **)(&v4Host), fileSizeState));
    ACL_CHECK(aclrtMallocHost((void **)(&v5Host), fileSizeState));
    ACL_CHECK(aclrtMallocHost((void **)(&v6Host), fileSizeState));
    ACL_CHECK(aclrtMallocHost((void **)(&v7Host), fileSizeOut));

    ACL_CHECK(aclrtMalloc((void **)&v1Device, fileSizeState, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v2Device, fileSizeState, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v3Device, fileSizeOut, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v4Device, fileSizeState, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v5Device, fileSizeState, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v6Device, fileSizeState, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v7Device, fileSizeOut, ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("./v1.bin", fileSizeState, v1Host, fileSizeState);
    ReadFile("./v2.bin", fileSizeState, v2Host, fileSizeState);
    ReadFile("./v3.bin", fileSizeOut, v3Host, fileSizeOut);
    ReadFile("./v4.bin", fileSizeState, v4Host, fileSizeState);
    ReadFile("./v5.bin", fileSizeState, v5Host, fileSizeState);
    ReadFile("./v6.bin", fileSizeState, v6Host, fileSizeState);
    ReadFile("./v7.bin", fileSizeOut, v7Host, fileSizeOut);

    ACL_CHECK(aclrtMemcpy(v1Device, fileSizeState, v1Host, fileSizeState, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v2Device, fileSizeState, v2Host, fileSizeState, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v3Device, fileSizeOut, v3Host, fileSizeOut, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v4Device, fileSizeState, v4Host, fileSizeState, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v5Device, fileSizeState, v5Host, fileSizeState, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v6Device, fileSizeState, v6Host, fileSizeState, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v7Device, fileSizeOut, v7Host, fileSizeOut, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchOnline_softmax_update_kernel_2d(v1Device, v2Device, v3Device,
                                          v4Device, v5Device, v6Device,
                                          v7Device, v8Host, v9Host, stream);

    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(v4Host, fileSizeState, v4Device, fileSizeState, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(v5Host, fileSizeState, v5Device, fileSizeState, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(v6Host, fileSizeState, v6Device, fileSizeState, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(v7Host, fileSizeOut, v7Device, fileSizeOut, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./v4.bin", v4Host, fileSizeState);
    WriteFile("./v5.bin", v5Host, fileSizeState);
    WriteFile("./v6.bin", v6Host, fileSizeState);
    WriteFile("./v7.bin", v7Host, fileSizeOut);

cleanup:
    aclrtFree(v1Device); aclrtFree(v2Device); aclrtFree(v3Device);
    aclrtFree(v4Device); aclrtFree(v5Device); aclrtFree(v6Device); aclrtFree(v7Device);
    aclrtFreeHost(v1Host); aclrtFreeHost(v2Host); aclrtFreeHost(v3Host);
    aclrtFreeHost(v4Host); aclrtFreeHost(v5Host); aclrtFreeHost(v6Host); aclrtFreeHost(v7Host);
    if (stream != nullptr) {
        const aclError _ret = aclrtDestroyStream(stream);
        if (_ret != ACL_SUCCESS)
            std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n",
                         "aclrtDestroyStream(stream)", (int)_ret, __FILE__, __LINE__);
    }
    if (deviceSet) {
        const aclError _ret = aclrtResetDevice(deviceId);
        if (_ret != ACL_SUCCESS)
            std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n",
                         "aclrtResetDevice(deviceId)", (int)_ret, __FILE__, __LINE__);
    }
    if (aclInited) {
        const aclError _ret = aclFinalize();
        if (_ret != ACL_SUCCESS)
            std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n",
                         "aclFinalize()", (int)_ret, __FILE__, __LINE__);
    }

    return rc;
}
