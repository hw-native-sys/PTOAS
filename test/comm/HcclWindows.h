// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TEST_COMM_HCCLWINDOWS_H
#define PTO_TEST_COMM_HCCLWINDOWS_H

// Test scaffolding: obtains the peer window table a remote transfer needs.
//
// A cross-device transfer needs an address on the far rank that the local DMA
// engine can resolve. Peer access is not enough -- measured on 910B1, an
// ordinary device pointer from another device is not reachable, and even a
// runtime device-to-device copy over such a pair hangs. The addresses that do
// work come from the communication domain, which is what sets up the
// translation, and HCCL hands them over as a window table.
//
// The route here is the one the reference stack uses:
//
//   HcclGetCommName          name the group behind an existing comm
//   HcomGetL0TopoTypeEx      reports the L0 topo; A5 often returns CUSTOM(5)
//                            rather than 1D mesh. The V2 tiling path still
//                            fills a CommDeviceContext-shaped window table.
//   HcomGetCommHandleByGroup the handle the allocator wants
//   HcclAllocComResourceByTiling
//                            on mesh, returns a device CommDeviceContext
//
// None of those four are declared in a public CANN header, and the tiling
// struct is an internal layout transcribed here, so this is a test aid and not
// something to ship. The supported call for this, HcclSymWinGetPeerPointer, is
// declared in hccl_sym_win.h but exported by no library in CANN 9.0.0; once it
// works, all of this collapses into one call.

#if defined(__CCE_KT_TEST__) || defined(__CCE__)
#error "HcclWindows.h is a host-only header and cannot be included in device code."
#endif

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include <dlfcn.h>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

namespace mlir::pto::comm {

//===----------------------------------------------------------------------===//
// HCCL entry points used here but not declared in any public header
//===----------------------------------------------------------------------===//
//
// These live behind libhccl on CANN 9.0.0, but 9.2.0 moved them into a
// dependency that lld will not pull in through -lhccl. Resolve them the same
// way AsyncWorkspace does: look in the already-loaded HCCL stack first, then
// in the libraries that have carried them across releases.

using HcclAllocComResourceByTilingFn = HcclResult (*)(HcclComm, void *, void *,
                                                      void **);
using HcomGetCommHandleByGroupFn = HcclResult (*)(const char *, HcclComm *);
using HcomGetL0TopoTypeExFn = HcclResult (*)(const char *, uint32_t *,
                                             uint32_t);

namespace hccl_win {

constexpr uint32_t kTopoMesh = 0b1u;
constexpr uint32_t kIsNotSetDevice = 0;
constexpr uint32_t kMaxRankNum = 64;
constexpr uint32_t kGroupNameSize = 128;
constexpr uint32_t kAlgConfigSize = 128;
constexpr uint32_t kMaxCcTilingNum = 8;

// Internal HCCL tiling layout. Field order and the constants below are what the
// allocator expects; they come from the reference implementation and are not
// derivable from anything public.
struct Mc2InitTilingInner {
  uint32_t version;
  uint32_t mc2HcommCnt;
  uint32_t offset[kMaxCcTilingNum];
  uint8_t debugMode;
  uint8_t preparePosition;
  uint16_t queueNum;
  uint16_t commBlockNum;
  uint8_t devType;
  char reserved[17];
};

struct Mc2cCTilingInner {
  uint8_t skipLocalRankCopy;
  uint8_t skipBufferWindowCopy;
  uint8_t stepSize;
  uint8_t version;
  char reserved[9];
  uint8_t commEngine;
  uint8_t srcDataType;
  uint8_t dstDataType;
  char groupName[kGroupNameSize];
  char algConfig[kAlgConfigSize];
  uint32_t opType;
  uint32_t reduceType;
};

struct Mc2CommConfigV2 {
  Mc2InitTilingInner init;
  Mc2cCTilingInner inner;
};

inline void fillDefaultTiling(Mc2CommConfigV2 &tiling, const char *groupName) {
  std::memset(&tiling, 0, sizeof(tiling));
  tiling.init.version = 100U;
  tiling.init.mc2HcommCnt = 1U;
  tiling.init.commBlockNum = 48U;
  tiling.init.devType = 4U;
  tiling.init.offset[0] =
      static_cast<uint32_t>(offsetof(Mc2CommConfigV2, inner));
  tiling.inner.opType = 18U;
  tiling.inner.commEngine = 3U;
  tiling.inner.version = 1U;
  if (groupName != nullptr) {
    std::strncpy(tiling.inner.groupName, groupName, kGroupNameSize - 1);
    tiling.inner.groupName[kGroupNameSize - 1] = '\0';
  }
  std::strncpy(tiling.inner.algConfig, "BatchWrite=level0:fullmesh",
               kAlgConfigSize - 1);
  tiling.inner.algConfig[kAlgConfigSize - 1] = '\0';
}

} // namespace hccl_win

//===----------------------------------------------------------------------===//
// Window table
//===----------------------------------------------------------------------===//

// What the allocator leaves in device memory on a mesh topology. windowsIn[r]
// is this rank's address for rank r's inbound window, which is exactly what a
// remote transfer names as its destination.
struct CommDeviceContext {
  uint64_t workSpace;
  uint64_t workSpaceSize;
  uint32_t rankId;
  uint32_t rankNum;
  uint64_t winSize;
  uint64_t windowsIn[hccl_win::kMaxRankNum];
  uint64_t windowsOut[hccl_win::kMaxRankNum];
};

class HcclWindows {
public:
  // Collective across the ranks of `comm`, so every rank has to call it, and in
  // a single-process run that means one thread per rank.
  bool init(HcclComm comm, void *stream) {
    if (!loadSymbols())
      return false;

    char group[hccl_win::kGroupNameSize] = {};
    if (HcclGetCommName(comm, group) != HCCL_SUCCESS)
      return fail("HcclGetCommName failed");

    if (hcomGetL0TopoTypeEx_(group, &topo_, hccl_win::kIsNotSetDevice) !=
        HCCL_SUCCESS)
      return fail("HcomGetL0TopoTypeEx failed");

    HcclComm handle = nullptr;
    if (hcomGetCommHandleByGroup_(group, &handle) != HCCL_SUCCESS ||
        handle == nullptr)
      return fail("HcomGetCommHandleByGroup failed");

    hccl_win::Mc2CommConfigV2 tiling{};
    hccl_win::fillDefaultTiling(tiling, group);

    void *devCtx = nullptr;
    const HcclResult hret =
        hcclAllocComResourceByTiling_(handle, stream, &tiling, &devCtx);
    if (hret != HCCL_SUCCESS || devCtx == nullptr)
      return fail("HcclAllocComResourceByTiling failed: " +
                  std::to_string(static_cast<int>(hret)));

    if (aclrtMemcpy(&context_, sizeof(context_), devCtx, sizeof(context_),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
      return fail("reading the window table back failed");

    if (context_.rankNum == 0 || context_.rankNum > hccl_win::kMaxRankNum)
      return fail("window table reports rankNum " +
                  std::to_string(context_.rankNum) + " (topo " +
                  std::to_string(topo_) + ")");
    if (context_.rankId >= context_.rankNum)
      return fail("window table reports rankId " +
                  std::to_string(context_.rankId) + " of " +
                  std::to_string(context_.rankNum) + " (topo " +
                  std::to_string(topo_) + ")");
    for (uint32_t r = 0; r < context_.rankNum; ++r) {
      if (context_.windowsIn[r] == 0)
        return fail("windowsIn[" + std::to_string(r) + "] is empty (topo " +
                    std::to_string(topo_) + ")");
    }

    deviceContext_ = devCtx;
    valid_ = true;
    return true;
  }

  const CommDeviceContext &context() const { return context_; }

  // Address of rank `rank`'s inbound window, as this rank must name it.
  uint64_t window(uint32_t rank) const {
    return rank < context_.rankNum ? context_.windowsIn[rank] : 0;
  }

  uint64_t windowBytes() const { return context_.winSize; }
  uint32_t rankId() const { return context_.rankId; }
  uint32_t rankNum() const { return context_.rankNum; }
  uint32_t topo() const { return topo_; }

  // Owned by HCCL; nothing to free here.
  void *deviceContext() const { return deviceContext_; }

  bool valid() const { return valid_; }
  const std::string &error() const { return error_; }

private:
  bool fail(std::string message) {
    error_ = std::move(message);
    return false;
  }

  template <typename Fn>
  bool bind(Fn &slot, const char *name) {
    if (slot)
      return true;
    slot = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, name));
    if (slot)
      return true;
    static const char *kLibs[] = {"libhccl.so", "libhcom.so", "libhccl_plf.so",
                                  "libhccl_heterog.so"};
    for (const char *lib : kLibs) {
      void *handle = dlopen(lib, RTLD_NOW | RTLD_GLOBAL);
      if (!handle)
        continue;
      slot = reinterpret_cast<Fn>(dlsym(handle, name));
      if (slot)
        return true;
    }
    return fail(std::string("dlsym ") + name + " failed");
  }

  bool loadSymbols() {
    return bind(hcomGetL0TopoTypeEx_, "HcomGetL0TopoTypeEx") &&
           bind(hcomGetCommHandleByGroup_, "HcomGetCommHandleByGroup") &&
           bind(hcclAllocComResourceByTiling_, "HcclAllocComResourceByTiling");
  }

  CommDeviceContext context_{};
  void *deviceContext_ = nullptr;
  bool valid_ = false;
  uint32_t topo_{0};
  std::string error_;
  HcomGetL0TopoTypeExFn hcomGetL0TopoTypeEx_{nullptr};
  HcomGetCommHandleByGroupFn hcomGetCommHandleByGroup_{nullptr};
  HcclAllocComResourceByTilingFn hcclAllocComResourceByTiling_{nullptr};
};

} // namespace mlir::pto::comm

#endif // PTO_TEST_COMM_HCCLWINDOWS_H
