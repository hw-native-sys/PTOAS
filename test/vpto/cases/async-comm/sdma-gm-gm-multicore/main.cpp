// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: async-comm/sdma-gm-gm-multicore
// target_ops: pto.sdma_gm_gm
// scenarios: single-device, local GM->GM, one channel group per core
// -----------------------------------------------------------------------------
//
// Correct data would not by itself show that the cores stayed on separate
// queues: one core doing all the work, or several sharing a ring and happening
// not to collide, both produce the same bytes. So the check is per channel --
// every group must have advanced by exactly one entry. A group that moved by
// more took someone else's post, and a group that did not move at all was
// never used.
//
// Each core's slice carries a distinct byte pattern, so a mismatch also says
// which core's transfer went wrong rather than only that something did.

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

void LaunchSdmaGmGmMulticore(int8_t *dst, int8_t *src, int8_t *sessionGm,
                             int64_t chunkBytes, uint32_t blocks, void *stream);

namespace {

constexpr unsigned kCores = 4;
constexpr size_t kChunkBytes = 1024;
constexpr size_t kTotalBytes = kCores * kChunkBytes;
constexpr int kPollTimeoutMs = 2000;

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

uint32_t readSqTail(AsyncWorkspace &ws, unsigned channel) {
  uint8_t raw[workspace::kChannelDescBytes];
  if (!ws.readChannelDescriptor(0, kCores, channel, raw))
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
  if (!ws.init(kCores)) {
    std::fprintf(stderr, "[ERROR] AsyncWorkspace::init failed: %s\n",
                 ws.error().c_str());
    return 1;
  }

  std::vector<uint32_t> tailBefore(kCores);
  for (unsigned c = 0; c < kCores; ++c)
    tailBefore[c] = readSqTail(ws, c);

  std::printf("workspace %p, %u channel groups\n", ws.contextGm(), kCores);
  for (unsigned c = 0; c < kCores; ++c) {
    uint8_t rec[mlir::pto::comm::channel::kRecordBytes];
    if (!ws.readChannelRecord(c, rec))
      continue;
    namespace ch = mlir::pto::comm::channel;
    uint64_t sqBase, doorbell;
    uint32_t streamId;
    std::memcpy(&sqBase, rec + ch::kSqBaseOffset, sizeof(sqBase));
    std::memcpy(&doorbell, rec + ch::kDoorbellOffset, sizeof(doorbell));
    std::memcpy(&streamId, rec + ch::kStreamIdOffset, sizeof(streamId));
    std::printf("  group %u: sq_base=%#" PRIx64 " doorbell=%#" PRIx64
                " stream=%u tail=%u\n",
                c, sqBase, doorbell, streamId, tailBefore[c]);
  }

  // A distinct pattern per slice, so a wrong slice is visible as the wrong
  // core's data rather than as generic corruption.
  std::vector<uint8_t> pattern(kTotalBytes);
  for (unsigned c = 0; c < kCores; ++c)
    for (size_t i = 0; i < kChunkBytes; ++i)
      pattern[c * kChunkBytes + i] =
          static_cast<uint8_t>(((i * 31 + 7) ^ (c * 0x5B)) & 0xFF);

  void *srcDevice = nullptr;
  void *dstDevice = nullptr;
  ACL_CHECK(aclrtMalloc(&srcDevice, kTotalBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(&dstDevice, kTotalBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMemcpy(srcDevice, kTotalBytes, pattern.data(), kTotalBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemset(dstDevice, kTotalBytes, 0, kTotalBytes));

  // One template for every core. The channel group is the only thing that
  // differs, and the kernel writes that itself from its core id.
  namespace comm = mlir::pto::comm;
  void *sessionDevice = nullptr;
  ACL_CHECK(aclrtMalloc(&sessionDevice, comm::session_tmpl::kBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  {
    comm::session_tmpl::Builder tmpl;
    tmpl.set(comm::SessionField::ContextGm,
             reinterpret_cast<uint64_t>(ws.contextGm()))
        .set(comm::SessionField::ChannelNum, 1)
        .set(comm::SessionField::BlockBytes, kChunkBytes)
        .set(comm::SessionField::Engine,
             static_cast<uint64_t>(comm::SessionEngine::Sdma))
        .set(comm::SessionField::Flags, comm::kSessionFlagValid)
        .set(comm::SessionField::Qos, comm::sqe::kQosDefault);
    ACL_CHECK(aclrtMemcpy(sessionDevice, comm::session_tmpl::kBytes,
                          tmpl.data(), comm::session_tmpl::kBytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
  }

  std::printf("posting %u slices of %zu bytes: src=%p dst=%p session=%p\n",
              kCores, kChunkBytes, srcDevice, dstDevice, sessionDevice);

  LaunchSdmaGmGmMulticore(static_cast<int8_t *>(dstDevice),
                          static_cast<int8_t *>(srcDevice),
                          static_cast<int8_t *>(sessionDevice),
                          static_cast<int64_t>(kChunkBytes), kCores, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  std::vector<uint8_t> readback(kTotalBytes);
  bool matched = false;
  int elapsedMs = 0;
  const auto start = std::chrono::steady_clock::now();
  while (elapsedMs < kPollTimeoutMs) {
    ACL_CHECK(aclrtMemcpy(readback.data(), kTotalBytes, dstDevice, kTotalBytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    if (std::memcmp(readback.data(), pattern.data(), kTotalBytes) == 0) {
      matched = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    elapsedMs = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
  }

  int rc = 0;
  for (unsigned c = 0; c < kCores; ++c) {
    const uint32_t after = readSqTail(ws, c);
    const uint32_t posted = after - tailBefore[c];

    size_t badBytes = 0;
    for (size_t i = 0; i < kChunkBytes; ++i)
      if (readback[c * kChunkBytes + i] != pattern[c * kChunkBytes + i])
        ++badBytes;

    const bool ok = posted == 1 && badBytes == 0;
    std::printf("group %u: %u entry (want 1), %zu bad bytes  %s\n", c, posted,
                badBytes, ok ? "OK" : "FAIL");
    if (!ok)
      rc = 1;
  }

  if (rc == 0 && matched)
    std::printf("PASS: every core posted through its own group\n");
  else if (rc == 0)
    std::printf("PASS (late): data settled after %d ms\n", elapsedMs);

  aclrtFree(srcDevice);
  aclrtFree(dstDevice);
  aclrtFree(sessionDevice);
  ws.finalize();
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return rc;
}
