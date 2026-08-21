// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: async-comm/sdma-gm-gm-peer-get
// target_ops: pto.sdma_gm_gm
// scenarios: two-device, peer GM->local GM, one channel, block splitting
// -----------------------------------------------------------------------------
//
// Rank 0 posts a GET from rank 1's window into its own. A5's engine can read
// a peer, so this kick posts SQEs: the tail must advance, and the destination
// is polled because returning from the kernel does not mean the engine has
// drained. sdma-gm-gm-peer-put uses `{soft_put}` and leaves the tail at zero.
//
// The same transfer is repeated at several block sizes, matching
// sdma-gm-gm-local: data alone would not show that splitting happened.
//
// Both endpoints sit in the HCCL window, because that is the memory a remote
// transfer can name. Rank 1 writes its own window, rank 0 names that range
// through its peer view, and rank 0 reads its own destination, so a pass
// cannot be explained by the bytes already sitting on the near device.

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl.h"

#include "AsyncWorkspace.h"
#include "HcclWindows.h"

using mlir::pto::comm::AsyncWorkspace;
using mlir::pto::comm::HcclWindows;
namespace workspace = mlir::pto::comm::workspace;

void LaunchSdmaGmGmPeerGet(int8_t *dst, int8_t *src, int8_t *sessionGm,
                           int64_t nbytes, void *stream);

namespace {

constexpr size_t kTransferBytes = 4096;
constexpr unsigned kChannels = 1;
constexpr unsigned kRanks = 2;

// Every size is a multiple of 64, the smallest transfer the engine accepts.
const size_t kBlockSizes[] = {4096, 1024, 1536, 128};

// The engine can queue behind other work, and this machine is shared, so the
// window is wide enough that a slow completion is not read as a failure.
constexpr int kPollTimeoutMsDefault = 30000;

#define ACL_CHECK(expr)                                                        \
  do {                                                                         \
    const aclError _ret = (expr);                                              \
    if (_ret != ACL_SUCCESS) {                                                 \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,           \
                   (int)_ret, __FILE__, __LINE__);                             \
      const char *_recent = aclGetRecentErrMsg();                              \
      if (_recent != nullptr && _recent[0] != '\0') {                          \
        std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent);           \
      }                                                                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define HCCL_CHECK(expr)                                                       \
  do {                                                                         \
    const HcclResult _ret = (expr);                                            \
    if (_ret != HCCL_SUCCESS) {                                                \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,           \
                   (int)_ret, __FILE__, __LINE__);                             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

uint32_t readSqTail(AsyncWorkspace &ws) {
  uint8_t raw[workspace::kChannelDescBytes];
  if (!ws.readChannelDescriptor(0, kChannels, 0, raw)) {
    return 0xFFFFFFFFU;
  }
  uint32_t tail;
  std::memcpy(&tail, raw + workspace::kChannelSqTailOffset, sizeof(tail));
  return tail;
}

struct Rank {
  int32_t device = -1;
  aclrtStream stream = nullptr;
  HcclWindows windows;
  bool ok = false;
};

// One per rank, on its own thread: ACL binds a device per thread, and the
// domain build has to be entered by both ranks at once.
void setupRank(Rank *rank, HcclComm comm) {
  if (aclrtSetDevice(rank->device) != ACL_SUCCESS) {
    return;
  }
  if (aclrtCreateStream(&rank->stream) != ACL_SUCCESS) {
    return;
  }
  if (!rank->windows.init(comm, rank->stream)) {
    std::fprintf(stderr, "[ERROR] device %d window setup failed: %s\n",
                 rank->device, rank->windows.error().c_str());
    return;
  }
  rank->ok = true;
}

} // namespace

int main(int argc, char **argv) {
  int32_t devices[kRanks] = {0, 1};
  if (argc > 2) {
    devices[0] = std::atoi(argv[1]);
    devices[1] = std::atoi(argv[2]);
  }
  const int pollTimeoutMs = std::getenv("SDMA_POLL_TIMEOUT_MS")
                                ? std::atoi(std::getenv("SDMA_POLL_TIMEOUT_MS"))
                                : kPollTimeoutMsDefault;

  std::printf("case: async-comm/sdma-gm-gm-peer-get  direction=GET\n");
  std::printf("ranks: 0 -> device %d, 1 -> device %d\n", devices[0],
              devices[1]);

  ACL_CHECK(aclInit(nullptr));
  // HcclCommInitAll reads the calling thread's current device even though it
  // opens all of them, and fails outright if none is set.
  ACL_CHECK(aclrtSetDevice(devices[0]));

  HcclComm comms[kRanks] = {nullptr, nullptr};
  HCCL_CHECK(HcclCommInitAll(kRanks, devices, comms));

  Rank ranks[kRanks];
  for (unsigned r = 0; r < kRanks; ++r) {
    ranks[r].device = devices[r];
  }
  {
    std::thread t0(setupRank, &ranks[0], comms[0]);
    std::thread t1(setupRank, &ranks[1], comms[1]);
    t0.join();
    t1.join();
  }
  for (unsigned r = 0; r < kRanks; ++r) {
    if (!ranks[r].ok) {
      std::fprintf(stderr, "[ERROR] rank %u setup failed\n", r);
      return 1;
    }
  }

  const uint64_t windowBytes = ranks[0].windows.windowBytes();
  std::printf("window %" PRIu64 " bytes, rank0 sees: own %#" PRIx64
              ", peer %#" PRIx64 "\n",
              windowBytes, ranks[0].windows.window(0),
              ranks[0].windows.window(1));
  std::printf("rank1 sees: own %#" PRIx64 ", peer %#" PRIx64 "\n",
              ranks[1].windows.window(1), ranks[1].windows.window(0));
  if (windowBytes < kTransferBytes) {
    std::fprintf(stderr, "[ERROR] window smaller than the transfer\n");
    return 1;
  }

  // Payload lives on rank 1. Rank 0 names it through its peer view and writes
  // into its own window.
  ACL_CHECK(aclrtSetDevice(devices[1]));
  void *farSrc = reinterpret_cast<void *>(ranks[1].windows.window(1));
  std::vector<uint8_t> pattern(kTransferBytes);
  for (size_t i = 0; i < kTransferBytes; ++i) {
    pattern[i] = static_cast<uint8_t>((i * 31 + 7) & 0xFF);
  }
  ACL_CHECK(aclrtMemcpy(farSrc, kTransferBytes, pattern.data(), kTransferBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  ACL_CHECK(aclrtSetDevice(devices[0]));
  void *src = reinterpret_cast<void *>(ranks[0].windows.window(1));
  void *dst = reinterpret_cast<void *>(ranks[0].windows.window(0));

  AsyncWorkspace ws;
  if (!ws.init(kChannels)) {
    std::fprintf(stderr, "[ERROR] AsyncWorkspace::init failed: %s\n",
                 ws.error().c_str());
    return 1;
  }
  std::printf("workspace %p, sq_tail before post = %u\n", ws.contextGm(),
              readSqTail(ws));
  std::printf("posting %zu bytes: src=%p (dev %d) dst=%p (dev %d)\n",
              kTransferBytes, src, devices[1], dst, devices[0]);

  namespace comm = mlir::pto::comm;
  void *sessionDevice = nullptr;
  ACL_CHECK(aclrtMalloc(&sessionDevice, comm::session_tmpl::kBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  int rc = 0;
  std::vector<uint8_t> readback(kTransferBytes);
  const size_t nBlocks = sizeof(kBlockSizes) / sizeof(kBlockSizes[0]);
  for (size_t i = 0; i < nBlocks; ++i) {
    const size_t block = kBlockSizes[i];
    const uint32_t wantEntries =
        static_cast<uint32_t>((kTransferBytes + block - 1) / block);

    comm::session_tmpl::Builder tmpl;
    tmpl.set(comm::SessionField::ContextGm,
             reinterpret_cast<uint64_t>(ws.contextGm()))
        .set(comm::SessionField::ChannelIdx, 0)
        .set(comm::SessionField::ChannelNum, 1)
        .set(comm::SessionField::BlockBytes, block)
        .set(comm::SessionField::Engine,
             static_cast<uint64_t>(comm::SessionEngine::Sdma))
        .set(comm::SessionField::Flags, comm::kSessionFlagValid)
        .set(comm::SessionField::Qos, comm::sqe::kQosDefault);
    ACL_CHECK(aclrtMemcpy(sessionDevice, comm::session_tmpl::kBytes,
                          tmpl.data(), comm::session_tmpl::kBytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtMemset(dst, kTransferBytes, 0, kTransferBytes));
    const uint32_t tailBefore = readSqTail(ws);

    LaunchSdmaGmGmPeerGet(static_cast<int8_t *>(dst),
                          static_cast<int8_t *>(src),
                          static_cast<int8_t *>(sessionDevice),
                          static_cast<int64_t>(kTransferBytes),
                          ranks[0].stream);
    ACL_CHECK(aclrtSynchronizeStream(ranks[0].stream));

    bool matched = false;
    int elapsedMs = 0;
    const auto start = std::chrono::steady_clock::now();
    while (elapsedMs < pollTimeoutMs) {
      ACL_CHECK(aclrtMemcpy(readback.data(), kTransferBytes, dst,
                            kTransferBytes, ACL_MEMCPY_DEVICE_TO_HOST));
      if (std::memcmp(readback.data(), pattern.data(), kTransferBytes) == 0) {
        matched = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      elapsedMs = static_cast<int>(
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start)
              .count());
    }

    const uint32_t tailAfter = readSqTail(ws);
    const uint32_t posted = tailAfter - tailBefore;
    const bool ok = matched && posted == wantEntries;
    std::printf("block %5zu -> %u entries (want %u), data %s  %s\n", block,
                posted, wantEntries, matched ? "match" : "MISMATCH",
                ok ? "OK" : "FAIL");
    std::fflush(stdout);

    if (!ok) {
      if (!matched) {
        size_t changed = 0;
        for (size_t b = 0; b < kTransferBytes; ++b) {
          if (readback[b] != 0) {
            ++changed;
          }
        }
        std::fprintf(stderr,
                     "  destination did not match within %d ms "
                     "(%zu of %zu bytes non-zero)\n",
                     pollTimeoutMs, changed, kTransferBytes);
      }
      rc = 1;
      break;
    }
  }

  if (rc == 0) {
    std::printf("PASS: every split posted the entries it implies\n");
  }

  ws.finalize();
  aclrtFree(sessionDevice);
  for (unsigned r = 0; r < kRanks; ++r) {
    if (ranks[r].stream != nullptr) {
      aclrtSetDevice(ranks[r].device);
      aclrtDestroyStream(ranks[r].stream);
    }
    HcclCommDestroy(comms[r]);
  }
  aclrtResetDevice(devices[0]);
  aclrtResetDevice(devices[1]);
  aclFinalize();
  return rc;
}
