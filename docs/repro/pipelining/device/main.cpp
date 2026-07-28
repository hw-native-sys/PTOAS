// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Minimal host: fill f32 input, launch CCE stages=2 scale, check vs host ref.
#include "acl/acl.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

void LaunchScaleStages2(void* x, void* scale, void* out, void* stream);

int main() {
  constexpr int N = 2048;
  const size_t nbytes = N * sizeof(float);
  const float scale_v = 1.5f;

  aclInit(nullptr);
  aclrtSetDevice(0);
  aclrtStream stream;
  aclrtCreateStream(&stream);

  float *x_h = nullptr, *y_h = nullptr, *scale_h = nullptr;
  void *x_d = nullptr, *y_d = nullptr, *scale_d = nullptr;
  aclrtMallocHost((void**)&x_h, nbytes);
  aclrtMallocHost((void**)&y_h, nbytes);
  aclrtMallocHost((void**)&scale_h, sizeof(float));
  aclrtMalloc(&x_d, nbytes, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc(&y_d, nbytes, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc(&scale_d, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);

  scale_h[0] = scale_v;
  for (int i = 0; i < N; ++i) {
    x_h[i] = 0.25f * static_cast<float>(i % 17);
    y_h[i] = 0.f;
  }

  aclrtMemcpy(x_d, nbytes, x_h, nbytes, ACL_MEMCPY_HOST_TO_DEVICE);
  aclrtMemcpy(scale_d, sizeof(float), scale_h, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
  aclrtMemcpy(y_d, nbytes, y_h, nbytes, ACL_MEMCPY_HOST_TO_DEVICE);

  LaunchScaleStages2(x_d, scale_d, y_d, stream);
  aclrtSynchronizeStream(stream);
  aclrtMemcpy(y_h, nbytes, y_d, nbytes, ACL_MEMCPY_DEVICE_TO_HOST);

  int mism = 0;
  float max_err = 0.f;
  for (int i = 0; i < N; ++i) {
    float ref = x_h[i] * scale_v;
    float err = std::fabs(ref - y_h[i]);
    if (err > max_err) max_err = err;
    if (err > 1e-5f) ++mism;
  }
  std::printf("scale_stages2: N=%d mism=%d max_err=%.4g %s\n", N, mism, max_err,
              mism == 0 ? "OK" : "FAIL");

  aclrtFree(x_d);
  aclrtFree(y_d);
  aclrtFree(scale_d);
  aclrtFreeHost(x_h);
  aclrtFreeHost(y_h);
  aclrtFreeHost(scale_h);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  return mism == 0 ? 0 : 1;
}
