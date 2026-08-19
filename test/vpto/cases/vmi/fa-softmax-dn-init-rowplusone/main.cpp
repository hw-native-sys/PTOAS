// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "acl/acl.h"
#include "test_common.h"
#include <cstdio>
#include <cstdlib>
using namespace PtoTestCommon;
#define ACL_CHECK(expr) do { const aclError _ret = (expr); if (_ret != ACL_SUCCESS) { std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr, (int)_ret, __FILE__, __LINE__); rc = 1; goto cleanup; } } while (0)
void LaunchFa_dn_softmax_128x64_rowplusone(float *scores, __bf16 *x_exp, float *global_max, float *global_sum, __bf16 *nz_out, void *stream);
int main() {
  constexpr size_t kScoresElems=128*64, kXExpElems=128*64, kReduceElems=1*64, kNzElems=128*64;
  size_t scoresBytes=kScoresElems*sizeof(float), xExpBytes=kXExpElems*sizeof(__bf16), reduceBytes=kReduceElems*sizeof(float), nzBytes=kNzElems*sizeof(__bf16);
  float *scoresHost=nullptr,*scoresDevice=nullptr; __bf16 *xExpHost=nullptr,*xExpDevice=nullptr;
  float *gmaxHost=nullptr,*gmaxDevice=nullptr,*gsumHost=nullptr,*gsumDevice=nullptr;
  __bf16 *nzHost=nullptr,*nzDevice=nullptr;
  int rc=0; bool aclInited=false,deviceSet=false; int deviceId=0; aclrtStream stream=nullptr;
  ACL_CHECK(aclInit(nullptr)); aclInited=true;
  if (const char *envDevice=std::getenv("ACL_DEVICE_ID")) deviceId=std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId)); deviceSet=true; ACL_CHECK(aclrtCreateStream(&stream));
  ACL_CHECK(aclrtMallocHost((void**)(&scoresHost),scoresBytes)); ACL_CHECK(aclrtMallocHost((void**)(&xExpHost),xExpBytes));
  ACL_CHECK(aclrtMallocHost((void**)(&gmaxHost),reduceBytes)); ACL_CHECK(aclrtMallocHost((void**)(&gsumHost),reduceBytes));
  ACL_CHECK(aclrtMallocHost((void**)(&nzHost),nzBytes));
  ACL_CHECK(aclrtMalloc((void**)&scoresDevice,scoresBytes,ACL_MEM_MALLOC_HUGE_FIRST)); ACL_CHECK(aclrtMalloc((void**)&xExpDevice,xExpBytes,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&gmaxDevice,reduceBytes,ACL_MEM_MALLOC_HUGE_FIRST)); ACL_CHECK(aclrtMalloc((void**)&gsumDevice,reduceBytes,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&nzDevice,nzBytes,ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./v1.bin",scoresBytes,scoresHost,scoresBytes);
  ACL_CHECK(aclrtMemcpy(scoresDevice,scoresBytes,scoresHost,scoresBytes,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemset(xExpDevice,xExpBytes,0,xExpBytes)); ACL_CHECK(aclrtMemset(gmaxDevice,reduceBytes,0,reduceBytes)); ACL_CHECK(aclrtMemset(gsumDevice,reduceBytes,0,reduceBytes));
  ACL_CHECK(aclrtMemset(nzDevice,nzBytes,0,nzBytes));
  LaunchFa_dn_softmax_128x64_rowplusone(scoresDevice,xExpDevice,gmaxDevice,gsumDevice,nzDevice,stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(xExpHost,xExpBytes,xExpDevice,xExpBytes,ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(gmaxHost,reduceBytes,gmaxDevice,reduceBytes,ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(gsumHost,reduceBytes,gsumDevice,reduceBytes,ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(nzHost,nzBytes,nzDevice,nzBytes,ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v2.bin",xExpHost,xExpBytes); WriteFile("./v3.bin",gmaxHost,reduceBytes); WriteFile("./v4.bin",gsumHost,reduceBytes); WriteFile("./v5.bin",nzHost,nzBytes);
cleanup:
  aclrtFree(scoresDevice);aclrtFree(xExpDevice);aclrtFree(gmaxDevice);aclrtFree(gsumDevice);aclrtFree(nzDevice);
  aclrtFreeHost(scoresHost);aclrtFreeHost(xExpHost);aclrtFreeHost(gmaxHost);aclrtFreeHost(gsumHost);aclrtFreeHost(nzHost);
  if(stream)aclrtDestroyStream(stream); if(deviceSet)aclrtResetDevice(deviceId); if(aclInited)aclFinalize();
  return rc;
}
