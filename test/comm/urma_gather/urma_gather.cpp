// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <pto/pto-inst.hpp>

#include "common.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>

#if defined(__CCE_AICORE__)
#include "generated_urma_gather.hpp"

#if defined(PTO_URMA_GATHER_FORCE_SIMT) && !defined(PTO_URMA_GATHER_SIMT_AVAILABLE)
#error "This test requires PTO-ISA SIMT Gather support"
#endif
#endif

namespace {
constexpr size_t kBufferBytes = 2 * 1024 * 1024;
constexpr size_t kDestinationsOffset = 64;
constexpr size_t kSegmentsOffset = 256;
constexpr size_t kPayloadOffset = 4096;
constexpr size_t kReceiveOffset = 1024 * 1024;
constexpr size_t kReceiveStride = 256 * 1024;
constexpr size_t kPrefixReceiveIndex = 0;
constexpr size_t kDeferredReceiveIndex = 1;
constexpr size_t kImmediateReceiveIndex = 2;
constexpr size_t kReceiveRegionCount = kImmediateReceiveIndex + 1;
constexpr size_t kSegmentStride = 16 * 1024;
constexpr size_t kMaxSegments = 25;
constexpr uint32_t kMinSegmentBytes = 4 * 1024;
constexpr uint32_t kMaxSegmentBytes = 2 * kMinSegmentBytes;
constexpr size_t kCanaryBytes = 64;
constexpr int kMinRanks = 2;
constexpr size_t kMaxRanks = 3;
constexpr uint8_t kSentinel = 0xff;
constexpr std::array kSegmentCounts{2U, 12U, 13U, 25U};

static_assert(kPayloadOffset + (kMaxRanks - 1) * kMaxSegments * kSegmentStride <= kReceiveOffset);
static_assert(kMaxSegments * kMaxSegmentBytes + kCanaryBytes <= kReceiveStride);
static_assert(kReceiveOffset + kReceiveRegionCount * kReceiveStride <= kBufferBytes);

constexpr uint32_t segmentLength(uint32_t segment) {
  return (1U + segment % 2U) * kMinSegmentBytes;
}

#if defined(__CCE_AICORE__)
AICORE constexpr uint32_t deviceSegmentLength(uint32_t segment) {
  return (1U + segment % 2U) * kMinSegmentBytes;
}
#endif

__global__ AICORE void launchGather(__gm__ uint8_t *buffer,
                                    __gm__ uint8_t *workspace,
                                    uint32_t destinationCount,
                                    uint32_t segmentsPerPeer, bool batch) {
#if defined(__CCE_AICORE__)
  static_assert(kDestinationsOffset % alignof(pto::comm::GatherDestination) == 0);
  static_assert(kSegmentsOffset % alignof(pto::comm::GatherSegment) == 0);
  static_assert(kDestinationsOffset + kReceiveRegionCount * (kMaxRanks - 1) *
                                          sizeof(pto::comm::GatherDestination) <= kSegmentsOffset);
  static_assert(kSegmentsOffset + (kMaxRanks - 1) * kMaxSegments * sizeof(pto::comm::GatherSegment) <= kPayloadOffset);
  auto *destinations = reinterpret_cast<__gm__ pto::comm::GatherDestination *>(buffer + kDestinationsOffset);
  auto *segments = reinterpret_cast<__gm__ pto::comm::GatherSegment *>(buffer + kSegmentsOffset);
  auto *status = reinterpret_cast<__gm__ int32_t *>(buffer);
  for (uint32_t target = 0; target < destinationCount; ++target) {
    const uint32_t peer = target + 1;
    const uint64_t remote = pto::comm::urma::UrmaPeerMrBaseAddr(workspace, peer);
    // Each policy writes its own receive region so later writes cannot mask a
    // missing transfer from an earlier policy.
    for (size_t region = 0; region < kReceiveRegionCount; ++region) {
      const size_t index = region * destinationCount + target;
      destinations[index].addr =
          reinterpret_cast<__gm__ uint8_t *>(remote + kReceiveOffset + region * kReceiveStride);
      destinations[index].segCount = segmentsPerPeer;
      destinations[index].peer = peer;
    }
    for (uint32_t segment = 0; segment < segmentsPerPeer; ++segment) {
      const size_t i = target * segmentsPerPeer + segment;
      segments[i].addr = buffer + kPayloadOffset + i * kSegmentStride;
      segments[i].length = deviceSegmentLength(segment);
    }
  }
  const uint32_t segmentCount = destinationCount * segmentsPerPeer;
  pto::comm::urma::DcciCachelines(buffer + kDestinationsOffset,
                                 kReceiveRegionCount * destinationCount * sizeof(*destinations));
  pto::comm::urma::DcciCachelines(buffer + kSegmentsOffset, segmentCount * sizeof(*segments));
  dsb(DSB_DDR);
  auto *workspaceBytes = reinterpret_cast<__gm__ int8_t *>(workspace);
  auto *segmentBytes = reinterpret_cast<__gm__ int8_t *>(buffer + kSegmentsOffset);
  auto *immediateBytes =
      reinterpret_cast<__gm__ int8_t *>(destinations + kImmediateReceiveIndex * destinationCount);
  if (batch) {
    auto *prefixBytes =
        reinterpret_cast<__gm__ int8_t *>(destinations + kPrefixReceiveIndex * destinationCount);
    auto *deferredBytes =
        reinterpret_cast<__gm__ int8_t *>(destinations + kDeferredReceiveIndex * destinationCount);
    *status = urma_batch(workspaceBytes, prefixBytes, deferredBytes, immediateBytes, segmentBytes,
                         destinationCount, segmentCount, 0U);
  } else {
    *status = urma_immediate(workspaceBytes, immediateBytes, segmentBytes, destinationCount, segmentCount);
  }
  pto::comm::urma::DcciCachelines(buffer, sizeof(*status));
  dsb(DSB_DDR);
  pipe_barrier(PIPE_ALL);
#endif
}

bool allRanksPassed(bool ok, int ranks) {
  const uint8_t local = ok;
  std::array<uint8_t, kMaxRanks> results{};
  return CommMpiAllgather(&local, sizeof(local), results.data(), sizeof(local)) == 0 &&
         std::all_of(results.begin(), results.begin() + ranks, [](uint8_t value) { return value == 1; });
}

bool setupCommunicator(UrmaTestContext &context) {
  using MpiAbortFunc = int (*)(MPI_Comm, int);
  auto broadcast = comm_mpi::GetFunc<MpiBcastFunc>("MPI_Bcast");
  auto abortJob = comm_mpi::GetFunc<MpiAbortFunc>("MPI_Abort");
  HcclRootInfo rootInfo{};
  const bool rootInfoReady = context.rankId != 0 ||
      (rtSetDevice(context.deviceId) == 0 && HcclGetRootInfo(&rootInfo) == HCCL_SUCCESS);
  if (!allRanksPassed(broadcast && abortJob && rootInfoReady, context.nRanks)) {
    return false;
  }
  const int broadcastStatus =
      broadcast(&rootInfo, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0, COMM_MPI_COMM_WORLD);
  if (!allRanksPassed(broadcastStatus == 0, context.nRanks)) {
    return false;
  }
  const HcclResult status = HcclCommInitRootInfo(context.nRanks, &rootInfo, context.rankId, &context.comm);
  if (status != HCCL_SUCCESS) {
    std::cerr << "Rank " << context.rankId << ": HcclCommInitRootInfo failed: " << static_cast<int>(status) << '\n';
    // Other ranks may still be inside the HCCL collective.
    abortJob(COMM_MPI_COMM_WORLD, EXIT_FAILURE);
  }
  return allRanksPassed(status == HCCL_SUCCESS, context.nRanks);
}

bool setup(UrmaTestContext &context, int rank, int ranks) {
  context.deviceId = rank;
  context.rankId = rank;
  context.nRanks = ranks;
  context.allocSize = kBufferBytes;
  const bool local = aclrtSetDevice(rank) == ACL_SUCCESS &&
                     rtStreamCreate(&context.stream, RT_STREAM_PRIORITY_DEFAULT) == 0 &&
                     aclrtMalloc(&context.devBuf, context.allocSize, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS;
  if (!allRanksPassed(local, ranks)) {
    return false;
  }
  if (!setupCommunicator(context)) {
    return false;
  }
  constexpr uint32_t kAivCount = 1;
  constexpr uint32_t kJetties = 1;
  constexpr auto kLayout = UrmaLayout::PER_PEER;
  UrmaWorkspaceManager::GetWorkspaceSize(ranks, &context.workspaceSize, kLayout, kAivCount, kJetties);
  const bool allocated = context.workspaceSize != 0 &&
                         aclrtMalloc(&context.workspaceAddr, context.workspaceSize,
                                     ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
  return allRanksPassed(allocated, ranks) &&
         context.urmaMgr.Init(context.comm, rank, ranks, context.devBuf, context.allocSize,
                              context.workspaceAddr, context.workspaceSize, kLayout, kAivCount, kJetties);
}

uint8_t payloadByte(size_t offset) {
  constexpr size_t kPayloadSeed = 17;
  constexpr size_t kPayloadPeriod = 251;
  return static_cast<uint8_t>((offset + kPayloadSeed) % kPayloadPeriod);
}

bool runCase(const UrmaTestContext &context, uint32_t segmentsPerPeer, bool batch) {
  const int rank = context.rankId;
  const int ranks = context.nRanks;
  auto *buffer = static_cast<uint8_t *>(context.devBuf);
  bool ready = aclrtMemset(buffer, kBufferBytes, kSentinel, kBufferBytes) == ACL_SUCCESS;
  if (rank == 0 && ready) {
    std::vector<uint8_t> payload((ranks - 1) * segmentsPerPeer * kSegmentStride);
    for (size_t i = 0; i < payload.size(); ++i) {
      payload[i] = payloadByte(i);
    }
    ready = aclrtMemcpy(buffer + kPayloadOffset, kReceiveOffset - kPayloadOffset,
                        payload.data(), payload.size(), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
  }
  if (!allRanksPassed(ready, ranks)) {
    return false;
  }

  int sync = 0;
  if (rank == 0) {
    launchGather<<<1, nullptr, context.stream>>>(
        buffer, static_cast<uint8_t *>(context.workspaceAddr), ranks - 1, segmentsPerPeer, batch);
    sync = aclrtSynchronizeStream(context.stream);
  }
  CommMpiBarrier();

  if (rank == 0) {
    int32_t status = 0;
    return sync == 0 &&
           aclrtMemcpy(&status, sizeof(status), buffer, sizeof(status), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS &&
           status == 1;
  }
  std::vector<uint8_t> expected;
  for (uint32_t segment = 0; segment < segmentsPerPeer; ++segment) {
    const size_t source = ((rank - 1) * segmentsPerPeer + segment) * kSegmentStride;
    for (uint32_t byte = 0; byte < segmentLength(segment); ++byte) {
      expected.push_back(payloadByte(source + byte));
    }
  }
  expected.resize(expected.size() + kCanaryBytes, kSentinel);
  std::vector<uint8_t> received(expected.size());
  const size_t firstRegion = batch ? kPrefixReceiveIndex : kImmediateReceiveIndex;
  for (size_t region = firstRegion; region < kReceiveRegionCount; ++region) {
    auto *source = buffer + kReceiveOffset + region * kReceiveStride;
    if (aclrtMemcpy(received.data(), received.size(), source,
                   received.size(), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        received != expected) {
      std::cerr << "Rank " << rank << ": mismatch in receive region " << region << '\n';
      return false;
    }
  }
  return true;
}

bool runCases(const UrmaTestContext &context) {
#if !defined(PTO_URMA_GATHER_FORCE_SIMT)
  constexpr std::array modes{false, true};
#else
  constexpr std::array modes{true};
#endif
  bool pass = true;
  for (bool batch : modes) {
    for (uint32_t segments : kSegmentCounts) {
      const bool ok = runCase(context, segments, batch);
      if (!ok) {
        std::cerr << "Rank " << context.rankId << ": Gather failed, segments="
                  << segments << ", batch=" << batch << '\n';
      }
      const bool allOk = allRanksPassed(ok, context.nRanks);
      pass = pass && allOk;
    }
  }
  return pass;
}
} // namespace

int main(int argc, char **argv) {
  if (!CommMpiInit(&argc, &argv)) {
    return 1;
  }
  const int rank = CommMpiRank();
  const int ranks = CommMpiSize();
  if (ranks < kMinRanks || ranks > static_cast<int>(kMaxRanks)) {
    std::cerr << "Run with 2 or 3 MPI ranks\n";
    CommMpiFinalize();
    return 1;
  }

  const bool aclReady = aclInit(nullptr) == ACL_SUCCESS;
  UrmaTestContext context;
  bool pass = allRanksPassed(aclReady, ranks);
  if (pass) {
    const bool ready = setup(context, rank, ranks);
    pass = allRanksPassed(ready, ranks);
    if (!pass) {
      std::cerr << "Rank " << rank << ": A5 URMA workspace setup failed\n";
    }
    if (pass) {
      pass = runCases(context);
    }
    context.Cleanup();
  }
  if (aclReady) {
    aclFinalize();
  }
  CommMpiFinalize();
  return pass ? 0 : 1;
}
