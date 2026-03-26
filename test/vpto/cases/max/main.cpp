/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
*/

#include "test_common.h"
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

#define ACL_CHECK(expr)                                                          \
  do {                                                                           \
    const aclError _ret = (expr);                                                \
    if (_ret != ACL_SUCCESS) {                                                   \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,             \
                   (int)_ret, __FILE__, __LINE__);                               \
      rc = 1;                                                                    \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

void LaunchMax_kernel_2d(float *v1, float *v2, float *v3, void *stream);

int main() {
  size_t elemCount_v1 = 1024;
  size_t fileSize_v1 = elemCount_v1 * sizeof(float);
  size_t elemCount_v2 = 1024;
  size_t fileSize_v2 = elemCount_v2 * sizeof(float);
  size_t elemCount_v3 = 1024;
  size_t fileSize_v3 = elemCount_v3 * sizeof(float);
  float *v1Host = nullptr;
  float *v1Device = nullptr;
  float *v2Host = nullptr;
  float *v2Device = nullptr;
  float *v3Host = nullptr;
  float *v3Device = nullptr;
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
  ACL_CHECK(aclrtMallocHost((void **)(&v1Host), fileSize_v1));
  ACL_CHECK(aclrtMallocHost((void **)(&v2Host), fileSize_v2));
  ACL_CHECK(aclrtMallocHost((void **)(&v3Host), fileSize_v3));
  ACL_CHECK(aclrtMalloc((void **)&v1Device, fileSize_v1, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&v2Device, fileSize_v2, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&v3Device, fileSize_v3, ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./v1.bin", fileSize_v1, v1Host, fileSize_v1);
  ReadFile("./v2.bin", fileSize_v2, v2Host, fileSize_v2);
  ReadFile("./v3.bin", fileSize_v3, v3Host, fileSize_v3);
  ACL_CHECK(aclrtMemcpy(v1Device, fileSize_v1, v1Host, fileSize_v1,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(v2Device, fileSize_v2, v2Host, fileSize_v2,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(v3Device, fileSize_v3, v3Host, fileSize_v3,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  LaunchMax_kernel_2d(v1Device, v2Device, v3Device, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(v3Host, fileSize_v3, v3Device, fileSize_v3,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v3.bin", v3Host, fileSize_v3);

cleanup:
  aclrtFree(v1Device);
  aclrtFree(v2Device);
  aclrtFree(v3Device);
  aclrtFreeHost(v1Host);
  aclrtFreeHost(v2Host);
  aclrtFreeHost(v3Host);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
