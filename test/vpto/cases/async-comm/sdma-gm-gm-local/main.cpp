// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: async-comm/sdma-gm-gm-local
// target_ops: pto.sdma_gm_gm
// scenarios: single-device, local GM->GM, one channel, block splitting
// -----------------------------------------------------------------------------
//
// The kernel returning only means the doorbell was written; the engine drains
// the queue on its own schedule. So the destination is polled rather than read
// once, and the queue tail is reported either way: a tail that advanced with a
// destination that never changed points at the engine, while a tail that
// stayed at zero points at the post itself.
//
// The same transfer is repeated at several block sizes. Correct data alone
// would not show that splitting happened, since one oversized entry moves the
// same bytes, so each round also checks that the tail advanced by exactly the
// number of entries the split implies. The sizes cover the divisible case, a
// case where the last entry is short, and one small enough to fill a good part
// of the ring.

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "acl/acl.h"

#include "AsyncWorkspace.h"

using mlir::pto::comm::AsyncWorkspace;
namespace workspace = mlir::pto::comm::workspace;

void LaunchSdmaGmGmLocal(int8_t *dst, int8_t *src, int8_t *sessionGm,
                         int64_t nbytes, void *stream);

namespace {

constexpr size_t kTransferBytes = 4096;
constexpr int kPollTimeoutMs = 2000;
constexpr unsigned kChannels = 1;

// Every size is a multiple of 64, the smallest transfer the engine accepts.
const size_t kBlockSizes[] = {4096, 1024, 1536, 128};

#define ACL_CHECK(expr)                                                        \
  do {                                                                         \
    const aclError _ret = (expr);                                              \
    if (_ret != ACL_SUCCESS) {                                                 \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,           \
                   (int)_ret, __FILE__, __LINE__);                             \
      const char *_recent = aclGetRecentErrMsg();                              \
      if (_recent != nullptr && _recent[0] != '\0')                            \
        std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent);           \
      return 1;                                                                \
    }                                                                          \
  } while (0)

uint32_t readSqTail(AsyncWorkspace &ws) {
  uint8_t raw[workspace::kChannelDescBytes];
  if (!ws.readChannelDescriptor(0, kChannels, 0, raw))
    return 0xFFFFFFFFU;
  uint32_t tail;
  std::memcpy(&tail, raw + workspace::kChannelSqTailOffset, sizeof(tail));
  return tail;
}

} // namespace

int main(int argc, char **argv) {
  const int deviceId =
      argc > 1 ? std::atoi(argv[1])
               : (std::getenv("ACL_DEVICE_ID")
                      ? std::atoi(std::getenv("ACL_DEVICE_ID"))
                      : 0);

  ACL_CHECK(aclInit(nullptr));
  ACL_CHECK(aclrtSetDevice(deviceId));

  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  AsyncWorkspace ws;
  if (!ws.init(kChannels)) {
    std::fprintf(stderr, "[ERROR] AsyncWorkspace::init failed: %s\n",
                 ws.error().c_str());
    return 1;
  }
  std::printf("workspace %p, sq_tail before post = %u\n", ws.contextGm(),
              readSqTail(ws));

  // What the kernel will read for channel 0. Printing it here separates a bad
  // table from a bad read of a good one, which are the two ways a post can go
  // wrong before the engine is ever involved.
  {
    uint8_t rec[mlir::pto::comm::channel::kRecordBytes];
    if (ws.readChannelRecord(0, rec)) {
      namespace ch = mlir::pto::comm::channel;
      uint64_t sqBase, doorbell, tailAddr, headAddr;
      uint32_t slotMask, streamId;
      std::memcpy(&sqBase, rec + ch::kSqBaseOffset, sizeof(sqBase));
      std::memcpy(&doorbell, rec + ch::kDoorbellOffset, sizeof(doorbell));
      std::memcpy(&tailAddr, rec + ch::kTailAddrOffset, sizeof(tailAddr));
      std::memcpy(&headAddr, rec + ch::kHeadAddrOffset, sizeof(headAddr));
      std::memcpy(&slotMask, rec + ch::kSlotMaskOffset, sizeof(slotMask));
      std::memcpy(&streamId, rec + ch::kStreamIdOffset, sizeof(streamId));
      std::printf("record 0: sq_base=%#" PRIx64 " doorbell=%#" PRIx64
                  " tail@%#" PRIx64 " head@%#" PRIx64 " slot_mask=%u"
                  " stream=%u\n",
                  sqBase, doorbell, tailAddr, headAddr, slotMask, streamId);
    }
  }

  std::vector<uint8_t> pattern(kTransferBytes);
  for (size_t i = 0; i < kTransferBytes; ++i)
    pattern[i] = static_cast<uint8_t>((i * 31 + 7) & 0xFF);

  void *srcDevice = nullptr;
  void *dstDevice = nullptr;
  ACL_CHECK(aclrtMalloc(&srcDevice, kTransferBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(&dstDevice, kTransferBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMemcpy(srcDevice, kTransferBytes, pattern.data(),
                        kTransferBytes, ACL_MEMCPY_HOST_TO_DEVICE));

  // The session the kernel will copy in. Everything the kernel knows about the
  // transfer beyond its size comes from here, so a different split is a
  // different template rather than a different kernel.
  namespace comm = mlir::pto::comm;
  void *sessionDevice = nullptr;
  ACL_CHECK(aclrtMalloc(&sessionDevice, comm::session_tmpl::kBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  std::printf("posting %zu bytes: src=%p dst=%p session=%p\n", kTransferBytes,
              srcDevice, dstDevice, sessionDevice);

  int rc = 0;
  std::vector<uint8_t> readback(kTransferBytes);

  for (size_t i = 0; i < sizeof(kBlockSizes) / sizeof(kBlockSizes[0]); ++i) {
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

    ACL_CHECK(aclrtMemset(dstDevice, kTransferBytes, 0, kTransferBytes));
    const uint32_t tailBefore = readSqTail(ws);

    LaunchSdmaGmGmLocal(static_cast<int8_t *>(dstDevice),
                        static_cast<int8_t *>(srcDevice),
                        static_cast<int8_t *>(sessionDevice),
                        static_cast<int64_t>(kTransferBytes), stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    bool matched = false;
    int elapsedMs = 0;
    const auto start = std::chrono::steady_clock::now();
    while (elapsedMs < kPollTimeoutMs) {
      ACL_CHECK(aclrtMemcpy(readback.data(), kTransferBytes, dstDevice,
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

    // The tail is what separates "moved the bytes" from "moved them the way it
    // was asked to": the same data arrives either way, but only a real split
    // leaves this many entries behind.
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
        for (size_t b = 0; b < kTransferBytes; ++b)
          if (readback[b] != 0)
            ++changed;
        std::fprintf(stderr,
                     "  destination did not match within %d ms "
                     "(%zu of %zu bytes non-zero)\n",
                     kPollTimeoutMs, changed, kTransferBytes);
      }
      rc = 1;
      break;
    }
  }

  if (rc == 0)
    std::printf("PASS: every split posted the entries it implies\n");

  aclrtFree(srcDevice);
  aclrtFree(dstDevice);
  aclrtFree(sessionDevice);
  ws.finalize();
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return rc;
}
