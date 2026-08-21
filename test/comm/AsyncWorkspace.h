// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TEST_COMM_ASYNCWORKSPACE_H
#define PTO_TEST_COMM_ASYNCWORKSPACE_H

// Test scaffolding: builds the async workspace that pto.sdma_gm_gm reads.
//
// Not a shipped interface, and deliberately not under include/. PTOAS owns the
// layout contract in PTO/Support/AsyncSessionABI.h, which is what code
// generation emits against; producing the workspace belongs to the
// communication library. The supported way to get one is HCCL, which acquires
// channels per engine (HcclChannelAcquire with COMM_ENGINE_AIV) and hands back
// an engine context. This file exists so the device path could be exercised on
// hardware before that binding was written.
//
// What it does, since none of it can be computed on the host -- the queue ring
// base and the doorbell address are STARS facts only an AICPU query reports:
//
//   1. create one device-only stream per channel and collect its ids
//   2. allocate and zero the workspace in device memory
//   3. stage the stream table where the operator can read it
//   4. run aclnnShmemSdmaStarsQuery, which fills the channel descriptors
//   5. check those descriptors and repack them into channel records
//   6. hand the record table to the kernel as the session context
//
// Step 5 is what keeps the descriptor layout from reaching generated code. It
// is a CANN-release fact rather than a hardware one -- the pto headers have
// already moved it once -- so it is resolved here, once, where a change fails
// the check instead of producing a wrong address inside a kernel.
//
// Everything is resolved with dlopen, so a program that never constructs an
// AsyncWorkspace links and runs unchanged on a machine without a toolkit.
//
// Two caveats that make this unsuitable for shipping. The stream-table and
// operator-descriptor structs below are transcribed rather than included, so a
// layout change on the CANN side corrupts memory silently instead of failing to
// build; and rtStreamGetSqid, rtStreamGetCqid and rtGetDeviceInfo are runtime
// internals rather than public ACL. pto-isa carries a near-identical copy of
// all this in sdma_workspace_manager.hpp; both are readers of a format neither
// owns, which is why agreement between them proves nothing and the check in
// step 5 compares against values fed to the query instead.
//
// Requires CANN 9.0.0 or newer at run time, for aclnnShmemSdmaStarsQuery. That
// is stricter than the device code needs: the generated kernel itself compiles
// and runs under CANN 8.x.

#if defined(__CCE_KT_TEST__) || defined(__CCE__)
#error "AsyncWorkspace.h is a host-only header and cannot be included in device code."
#endif

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <dlfcn.h>

#include "PTO/Support/AsyncSessionABI.h"

namespace mlir::pto::comm {

namespace detail {

// ACL values this header depends on, spelled out rather than included so the
// header stays free of CANN at build time. Taken from acl_rt.h / acl_base.h.
constexpr uint32_t kAclStreamFastLaunch = 0x00000001U;
constexpr uint32_t kAclStreamFastSync = 0x00000002U;
constexpr uint32_t kAclStreamDeviceUseOnly = 0x00000020U;
constexpr int32_t kAclStreamAttrFailureMode = 1;
constexpr int32_t kAclMemMallocHugeFirst = 0;
constexpr int32_t kAclMemcpyHostToDevice = 1;
constexpr int32_t kAclMemcpyDeviceToHost = 2;
constexpr int32_t kAclUint64 = 10;
constexpr int32_t kAclFormatND = 2;

// rtGetDeviceInfo selector for the physical die id.
constexpr int32_t kInfoTypePhyDieId = 19;

// One entry of the stream table the STARS query consumes. The operator matches
// on the exact size, so the padding is load-bearing.
struct HostStreamInfo {
  uint64_t stream;
  uint64_t ctx;
  int32_t streamId;
  uint32_t sqId;
  uint32_t cqId;
  uint32_t logicCqId;
  uint64_t cqeAddr;
  int32_t devId;
  uint8_t reserved[20];
};
static_assert(sizeof(HostStreamInfo) == 64, "HostStreamInfo must be 64 bytes");

// Descriptor the operator reads to find the stream table and the workspace.
struct OpResInfo {
  uint64_t size;
  uint64_t streamsAddr;
  uint64_t workspaceAddr;
  uint8_t reserved[40];
};
static_assert(sizeof(OpResInfo) == 64, "OpResInfo must be 64 bytes");

} // namespace detail

class AsyncWorkspace {
public:
  AsyncWorkspace() = default;
  ~AsyncWorkspace() { finalize(); }

  AsyncWorkspace(const AsyncWorkspace &) = delete;
  AsyncWorkspace &operator=(const AsyncWorkspace &) = delete;

  // Builds the workspace for `channels` channels on the current device, which
  // must already be selected with aclrtSetDevice. Returns false and leaves
  // `error()` set on failure.
  bool init(unsigned channels = workspace::kMaxChannels) {
    if (inited_)
      return true;
    if (channels == 0 || channels > workspace::kMaxChannels)
      return fail("channel count must be in [1, " +
                  std::to_string(workspace::kMaxChannels) + "]");

    if (!loadSymbols() || !createStreams(channels) || !allocWorkspace() ||
        !stageStreamTable() || !runStarsQuery() || !validateAndRepack()) {
      finalize();
      return false;
    }
    inited_ = true;
    return true;
  }

  void finalize() {
    if (recordsDevice_) {
      aclrtFree_(recordsDevice_);
      recordsDevice_ = nullptr;
    }
    if (streamsDevice_) {
      aclrtFree_(streamsDevice_);
      streamsDevice_ = nullptr;
    }
    if (opResDevice_) {
      aclrtFree_(opResDevice_);
      opResDevice_ = nullptr;
    }
    if (workspaceDevice_) {
      aclrtFree_(workspaceDevice_);
      workspaceDevice_ = nullptr;
    }
    for (auto &s : streams_) {
      if (s.stream)
        aclrtDestroyStream_(reinterpret_cast<void *>(s.stream));
    }
    streams_.clear();
    closeLibs();
    inited_ = false;
  }

  // Pass this to the kernel as SessionField::ContextGm. It addresses the
  // channel records this class builds, not the workspace the query wrote.
  void *contextGm() const { return recordsDevice_; }

  // The workspace itself, which stays allocated because the live head and tail
  // remain in it and the records point at them.
  void *workspaceGm() const { return workspaceDevice_; }

  unsigned channelCount() const {
    return static_cast<unsigned>(streams_.size());
  }

  // Channels the query reported populating, or 0 if it reported nothing. Only
  // meaningful after a successful init.
  uint32_t reportedQueueNum() const { return reportedQueueNum_; }

  const std::string &error() const { return error_; }

  // Reads one channel descriptor back to the host. Only useful for checking
  // that the STARS query actually populated the table.
  bool readChannelDescriptor(unsigned channelIdx, unsigned channelNum,
                             unsigned channelInGroup,
                             uint8_t (&out)[workspace::kChannelDescBytes]) {
    if (!inited_)
      return fail("workspace is not initialized");
    size_t offset =
        workspace::channelDescOffset(channelIdx, channelNum, channelInGroup);
    if (offset + workspace::kChannelDescBytes > workspace::kContextBytes)
      return fail("channel descriptor lies outside the context region");
    void *src = static_cast<uint8_t *>(workspaceDevice_) + offset;
    if (aclrtMemcpy_(out, sizeof(out), src, sizeof(out),
                     detail::kAclMemcpyDeviceToHost) != 0)
      return fail("aclrtMemcpy of channel descriptor failed");
    return true;
  }

  // Reads one channel record back, which is what a kernel sees for that
  // channel. Only useful for checking that the table reached the device.
  bool readChannelRecord(unsigned index,
                         uint8_t (&out)[comm::channel::kRecordBytes]) {
    if (!inited_ || index >= streams_.size())
      return fail("no such channel record");
    void *src = static_cast<uint8_t *>(recordsDevice_) +
                comm::channel::recordOffset(index, 1, 0);
    if (aclrtMemcpy_(out, sizeof(out), src, sizeof(out),
                     detail::kAclMemcpyDeviceToHost) != 0)
      return fail("aclrtMemcpy of channel record failed");
    return true;
  }

private:
  // Checks what the query wrote and repacks it into the channel records.
  //
  // The check is exact rather than a plausibility test, because the query was
  // handed the stream, sq, cq and device ids and echoes them back: a descriptor
  // whose ids sit where they are expected is at the layout this build assumes,
  // and one byte of drift breaks the match. Only the fields the query invents
  // -- the ring base, the doorbell and the depth -- fall back to a range check.
  bool validateAndRepack() {
    const unsigned channels = static_cast<unsigned>(streams_.size());

    uint8_t header[workspace::kFlagInfoBytes];
    if (!readWorkspace(0, header, sizeof(header)))
      return fail("reading the workspace flag-info header failed");
    std::memcpy(&reportedQueueNum_,
                header + workspace::kFlagInfoTotalQueueNumOffset,
                sizeof(reportedQueueNum_));

    // Trust the reported count when it is sane; a query that reports fewer
    // channels than asked has populated fewer descriptors, and indexing past
    // them would read the zeroed allocation.
    if (reportedQueueNum_ != 0 && reportedQueueNum_ < channels)
      return fail("the query populated " + std::to_string(reportedQueueNum_) +
                  " channels but " + std::to_string(channels) +
                  " were requested");

    std::vector<uint8_t> records(static_cast<size_t>(channels) *
                                     comm::channel::kRecordBytes,
                                 0);

    for (unsigned i = 0; i < channels; ++i) {
      const size_t descOffset =
          workspace::kFlagInfoBytes +
          static_cast<size_t>(i) * workspace::kChannelDescBytes;
      uint8_t desc[workspace::kChannelDescBytes];
      if (!readWorkspace(descOffset, desc, sizeof(desc)))
        return fail("reading channel descriptor " + std::to_string(i) +
                    " failed");

      const detail::HostStreamInfo &want = streams_[i];
      if (!expectU32(desc, workspace::kChannelStreamIdOffset,
                     static_cast<uint32_t>(want.streamId), "stream_id", i) ||
          !expectU32(desc, workspace::kChannelSqIdOffset, want.sqId, "sq_id",
                     i) ||
          !expectU32(desc, workspace::kChannelCqIdOffset, want.cqId, "cq_id",
                     i) ||
          !expectU32(desc, workspace::kChannelLogicCqIdOffset, want.logicCqId,
                     "logic_cq_id", i) ||
          !expectU32(desc, workspace::kChannelDevIdOffset,
                     static_cast<uint32_t>(want.devId), "dev_id", i))
        return false;

      uint64_t sqBase = 0, sqRegBase = 0;
      uint32_t sqDepth = 0, sqHead = 0, sqTail = 0;
      std::memcpy(&sqBase, desc + workspace::kChannelSqBaseOffset,
                  sizeof(sqBase));
      std::memcpy(&sqRegBase, desc + workspace::kChannelSqRegBaseOffset,
                  sizeof(sqRegBase));
      std::memcpy(&sqDepth, desc + workspace::kChannelSqDepthOffset,
                  sizeof(sqDepth));
      std::memcpy(&sqHead, desc + workspace::kChannelSqHeadOffset,
                  sizeof(sqHead));
      std::memcpy(&sqTail, desc + workspace::kChannelSqTailOffset,
                  sizeof(sqTail));

      if (sqBase == 0 || (sqBase & (workspace::kChannelDescBytes - 1)) != 0)
        return fail("channel " + std::to_string(i) + " has sq_base " +
                    hex(sqBase) + ", which is not a usable ring base");
      if (sqRegBase == 0)
        return fail("channel " + std::to_string(i) + " has no doorbell address");
      // The ring the engine actually uses is a power of two; generated code
      // wraps with a mask. A5's STARS query on CANN 9.2.0 reports 2049 for a
      // 2048-slot ring, so a count that is one past a power of two is taken as
      // that ring rather than rejected. Any other non-power-of-two stays a
      // hard error: masking it would land posts in the wrong slot.
      uint32_t ringDepth = sqDepth;
      if (ringDepth > 1 && (ringDepth & (ringDepth - 1)) != 0 &&
          ((ringDepth - 1) & (ringDepth - 2)) == 0)
        ringDepth = sqDepth - 1;
      if (ringDepth == 0 || (ringDepth & (ringDepth - 1)) != 0)
        return fail("channel " + std::to_string(i) + " reports depth " +
                    std::to_string(sqDepth) +
                    ", which is not a power of two and cannot be masked");
      if (sqHead >= ringDepth || sqTail >= ringDepth)
        return fail("channel " + std::to_string(i) + " starts with head " +
                    std::to_string(sqHead) + " tail " + std::to_string(sqTail) +
                    " outside a queue of " + std::to_string(ringDepth));

      const uint64_t descGm =
          reinterpret_cast<uint64_t>(workspaceDevice_) + descOffset;
      uint8_t *rec = records.data() + comm::channel::recordOffset(i, 1, 0);
      const uint64_t tailAddr = descGm + workspace::kChannelSqTailOffset;
      const uint64_t headAddr = descGm + workspace::kChannelSqHeadOffset;
      const uint32_t streamId = static_cast<uint32_t>(want.streamId);
      const uint32_t slotMask = ringDepth - 1;
      std::memcpy(rec + comm::channel::kSqBaseOffset, &sqBase, sizeof(sqBase));
      std::memcpy(rec + comm::channel::kDoorbellOffset, &sqRegBase,
                  sizeof(sqRegBase));
      std::memcpy(rec + comm::channel::kTailAddrOffset, &tailAddr,
                  sizeof(tailAddr));
      std::memcpy(rec + comm::channel::kHeadAddrOffset, &headAddr,
                  sizeof(headAddr));
      std::memcpy(rec + comm::channel::kSlotMaskOffset, &slotMask,
                  sizeof(slotMask));
      std::memcpy(rec + comm::channel::kStreamIdOffset, &streamId,
                  sizeof(streamId));
    }

    if (aclrtMalloc_(&recordsDevice_, records.size(),
                     detail::kAclMemMallocHugeFirst) != 0)
      return fail("aclrtMalloc of the channel record table failed");
    if (aclrtMemcpy_(recordsDevice_, records.size(), records.data(),
                     records.size(), detail::kAclMemcpyHostToDevice) != 0)
      return fail("aclrtMemcpy of the channel record table failed");
    return true;
  }

  bool readWorkspace(size_t offset, void *out, size_t bytes) {
    if (offset + bytes > workspace::kContextBytes)
      return false;
    void *src = static_cast<uint8_t *>(workspaceDevice_) + offset;
    return aclrtMemcpy_(out, bytes, src, bytes,
                        detail::kAclMemcpyDeviceToHost) == 0;
  }

  // Compares one echoed id, and on a mismatch says where the value did land, so
  // a layout that shifted reports the shift instead of just a bad number.
  bool expectU32(const uint8_t *desc, size_t offset, uint32_t want,
                 const char *what, unsigned channel) {
    uint32_t got = 0;
    std::memcpy(&got, desc + offset, sizeof(got));
    if (got == want)
      return true;

    std::string where;
    for (size_t probe = 0; probe + sizeof(uint32_t) <= workspace::kChannelDescBytes;
         probe += sizeof(uint32_t)) {
      uint32_t at = 0;
      std::memcpy(&at, desc + probe, sizeof(at));
      if (at == want)
        where += (where.empty() ? "" : ", ") + std::to_string(probe);
    }
    return fail(
        "channel " + std::to_string(channel) + ": expected " + what + " " +
        std::to_string(want) + " at descriptor offset " +
        std::to_string(offset) + " but found " + std::to_string(got) +
        (where.empty()
             ? "; it appears nowhere in the descriptor, so the query did not "
               "run as expected"
             : "; it appears at offset " + where +
                   ", so the descriptor layout has moved"));
  }

  static std::string hex(uint64_t v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%#llx",
                  static_cast<unsigned long long>(v));
    return buf;
  }

  using AclrtCreateStreamWithConfigFn = int32_t (*)(void **, uint32_t, uint32_t);
  using AclrtDestroyStreamFn = int32_t (*)(void *);
  using AclrtStreamGetIdFn = int32_t (*)(void *, int32_t *);
  using AclrtGetCurrentContextFn = int32_t (*)(void **);
  using AclrtGetDeviceFn = int32_t (*)(int32_t *);
  using AclrtMallocFn = int32_t (*)(void **, size_t, int32_t);
  using AclrtFreeFn = int32_t (*)(void *);
  using AclrtMemsetFn = int32_t (*)(void *, size_t, int32_t, size_t);
  using AclrtMemcpyFn = int32_t (*)(void *, size_t, const void *, size_t,
                                    int32_t);
  using AclrtSynchronizeStreamFn = int32_t (*)(void *);
  using AclrtSetStreamAttributeFn = int32_t (*)(void *, int32_t, const void *);

  using RtStreamGetSqidFn = int32_t (*)(const void *, uint32_t *);
  using RtStreamGetCqidFn = int32_t (*)(const void *, uint32_t *, uint32_t *);
  using RtGetDeviceInfoFn = int32_t (*)(uint32_t, int32_t, int32_t, int64_t *);

  using AclCreateTensorFn = void *(*)(const int64_t *, uint64_t, int32_t,
                                      const int64_t *, int64_t, int32_t,
                                      const int64_t *, uint64_t, void *);
  using AclDestroyTensorFn = int32_t (*)(void *);
  using StarsQueryGetWsSizeFn = int32_t (*)(const void *, void *, uint64_t *,
                                            void **);
  using StarsQueryExecFn = int32_t (*)(void *, uint64_t, void *, void *);

  bool fail(const std::string &what) {
    if (error_.empty())
      error_ = what;
    return false;
  }

  template <typename Fn>
  bool bind(Fn &slot, void *handle, const char *name) {
    slot = reinterpret_cast<Fn>(dlsym(handle, name));
    if (!slot)
      return fail(std::string("dlsym ") + name + " failed");
    return true;
  }

  bool loadSymbols() {
    aclHandle_ = dlopen("libascendcl.so", RTLD_NOW | RTLD_GLOBAL);
    if (!aclHandle_)
      return fail(std::string("dlopen libascendcl.so failed: ") + dlerror());
    rtHandle_ = dlopen("libruntime.so", RTLD_NOW);
    if (!rtHandle_)
      return fail(std::string("dlopen libruntime.so failed: ") + dlerror());
    opapiHandle_ = dlopen("libopapi.so", RTLD_NOW);
    if (!opapiHandle_)
      return fail(std::string("dlopen libopapi.so failed: ") + dlerror());
    nnopHandle_ = dlopen("libnnopbase.so", RTLD_NOW | RTLD_GLOBAL);
    if (!nnopHandle_)
      return fail(std::string("dlopen libnnopbase.so failed: ") + dlerror());

    return bind(aclrtCreateStreamWithConfig_, aclHandle_,
                "aclrtCreateStreamWithConfig") &&
           bind(aclrtDestroyStream_, aclHandle_, "aclrtDestroyStream") &&
           bind(aclrtStreamGetId_, aclHandle_, "aclrtStreamGetId") &&
           bind(aclrtGetCurrentContext_, aclHandle_, "aclrtGetCurrentContext") &&
           bind(aclrtGetDevice_, aclHandle_, "aclrtGetDevice") &&
           bind(aclrtMalloc_, aclHandle_, "aclrtMalloc") &&
           bind(aclrtFree_, aclHandle_, "aclrtFree") &&
           bind(aclrtMemset_, aclHandle_, "aclrtMemset") &&
           bind(aclrtMemcpy_, aclHandle_, "aclrtMemcpy") &&
           bind(aclrtSynchronizeStream_, aclHandle_,
                "aclrtSynchronizeStream") &&
           bind(aclrtSetStreamAttribute_, aclHandle_,
                "aclrtSetStreamAttribute") &&
           bind(rtStreamGetSqid_, rtHandle_, "rtStreamGetSqid") &&
           bind(rtStreamGetCqid_, rtHandle_, "rtStreamGetCqid") &&
           bind(rtGetDeviceInfo_, rtHandle_, "rtGetDeviceInfo") &&
           bind(aclCreateTensor_, nnopHandle_, "aclCreateTensor") &&
           bind(aclDestroyTensor_, nnopHandle_, "aclDestroyTensor") &&
           bind(starsQueryGetWsSize_, opapiHandle_,
                "aclnnShmemSdmaStarsQueryGetWorkspaceSize") &&
           bind(starsQueryExec_, opapiHandle_, "aclnnShmemSdmaStarsQuery");
  }

  void closeLibs() {
    for (void **h : {&opapiHandle_, &nnopHandle_, &rtHandle_, &aclHandle_}) {
      if (*h) {
        dlclose(*h);
        *h = nullptr;
      }
    }
  }

  bool createStreams(unsigned channels) {
    int32_t deviceId = -1;
    if (aclrtGetDevice_(&deviceId) != 0)
      return fail("aclrtGetDevice failed; call aclrtSetDevice first");

    int64_t dieId = -1;
    if (rtGetDeviceInfo_(static_cast<uint32_t>(deviceId), 0,
                         detail::kInfoTypePhyDieId, &dieId) != 0)
      return fail("rtGetDeviceInfo(die id) failed");

    streams_.resize(channels);
    for (unsigned i = 0; i < channels; ++i) {
      detail::HostStreamInfo &info = streams_[i];
      std::memset(&info, 0, sizeof(info));

      void *stream = nullptr;
      if (aclrtCreateStreamWithConfig_(&stream, 0,
                                       detail::kAclStreamDeviceUseOnly) != 0)
        return fail("aclrtCreateStreamWithConfig failed on channel " +
                    std::to_string(i));
      info.stream = reinterpret_cast<uint64_t>(stream);

      int32_t streamId = 0;
      if (aclrtStreamGetId_(stream, &streamId) != 0)
        return fail("aclrtStreamGetId failed on channel " + std::to_string(i));

      uint32_t sqId = 0;
      if (rtStreamGetSqid_(stream, &sqId) != 0)
        return fail("rtStreamGetSqid failed on channel " + std::to_string(i));

      uint32_t cqId = 0;
      uint32_t logicCqId = 0;
      if (rtStreamGetCqid_(stream, &cqId, &logicCqId) != 0)
        return fail("rtStreamGetCqid failed on channel " + std::to_string(i));

      void *ctx = nullptr;
      if (aclrtGetCurrentContext_(&ctx) != 0)
        return fail("aclrtGetCurrentContext failed on channel " +
                    std::to_string(i));

      info.ctx = reinterpret_cast<uint64_t>(ctx);
      info.streamId = streamId;
      info.sqId = sqId;
      info.cqId = cqId;
      info.logicCqId = logicCqId;
      info.devId = static_cast<int32_t>(dieId);
    }
    return true;
  }

  bool allocWorkspace() {
    if (aclrtMalloc_(&workspaceDevice_, workspace::kTotalBytes,
                     detail::kAclMemMallocHugeFirst) != 0)
      return fail("aclrtMalloc of the async workspace failed");
    // The kernel reads descriptors the query does not touch, so start from a
    // known state rather than whatever the allocator returned.
    if (aclrtMemset_(workspaceDevice_, workspace::kTotalBytes, 0,
                     workspace::kTotalBytes) != 0)
      return fail("aclrtMemset of the async workspace failed");
    return true;
  }

  bool stageStreamTable() {
    size_t streamsBytes = streams_.size() * sizeof(detail::HostStreamInfo);
    if (aclrtMalloc_(&streamsDevice_, streamsBytes,
                     detail::kAclMemMallocHugeFirst) != 0)
      return fail("aclrtMalloc of the stream table failed");
    if (aclrtMemcpy_(streamsDevice_, streamsBytes, streams_.data(),
                     streamsBytes, detail::kAclMemcpyHostToDevice) != 0)
      return fail("aclrtMemcpy of the stream table failed");

    detail::OpResInfo opRes{};
    opRes.size = streams_.size();
    opRes.streamsAddr = reinterpret_cast<uint64_t>(streamsDevice_);
    opRes.workspaceAddr = reinterpret_cast<uint64_t>(workspaceDevice_);

    if (aclrtMalloc_(&opResDevice_, sizeof(opRes),
                     detail::kAclMemMallocHugeFirst) != 0)
      return fail("aclrtMalloc of the operator descriptor failed");
    if (aclrtMemcpy_(opResDevice_, sizeof(opRes), &opRes, sizeof(opRes),
                     detail::kAclMemcpyHostToDevice) != 0)
      return fail("aclrtMemcpy of the operator descriptor failed");
    return true;
  }

  // A device buffer plus the tensor that wraps it, released together.
  struct TensorHandle {
    void *data{nullptr};
    void *tensor{nullptr};
  };

  bool makeScalarTensor(const std::vector<uint64_t> &values,
                        TensorHandle &out) {
    const int64_t shape = static_cast<int64_t>(values.size());
    const int64_t stride = 1;
    size_t bytes = values.size() * sizeof(uint64_t);

    if (aclrtMalloc_(&out.data, bytes, detail::kAclMemMallocHugeFirst) != 0)
      return fail("aclrtMalloc of a query tensor failed");
    if (aclrtMemcpy_(out.data, bytes, values.data(), bytes,
                     detail::kAclMemcpyHostToDevice) != 0)
      return fail("aclrtMemcpy of a query tensor failed");

    out.tensor = aclCreateTensor_(&shape, 1, detail::kAclUint64, &stride, 0,
                                  detail::kAclFormatND, &shape, 1, out.data);
    if (!out.tensor)
      return fail("aclCreateTensor failed");
    return true;
  }

  void releaseTensor(TensorHandle &handle) {
    if (handle.tensor) {
      aclDestroyTensor_(handle.tensor);
      handle.tensor = nullptr;
    }
    if (handle.data) {
      aclrtFree_(handle.data);
      handle.data = nullptr;
    }
  }

  bool runStarsQuery() {
    void *stream = nullptr;
    if (aclrtCreateStreamWithConfig_(&stream, 0,
                                     detail::kAclStreamFastLaunch |
                                         detail::kAclStreamFastSync) != 0)
      return fail("aclrtCreateStreamWithConfig for the query stream failed");

    // Ask the runtime to report a failed query rather than tearing the process
    // down, so a missing STARS backend surfaces as a return code.
    uint32_t failureMode = 1;
    aclrtSetStreamAttribute_(stream, detail::kAclStreamAttrFailureMode,
                             &failureMode);

    TensorHandle input;
    TensorHandle output;
    void *aclnnWs = nullptr;
    bool ok = false;

    do {
      if (!makeScalarTensor({reinterpret_cast<uint64_t>(opResDevice_),
                             reinterpret_cast<uint64_t>(workspaceDevice_)},
                            input))
        break;
      if (!makeScalarTensor({0}, output))
        break;

      uint64_t wsSize = 0;
      void *executor = nullptr;
      if (starsQueryGetWsSize_(input.tensor, output.tensor, &wsSize,
                               &executor) != 0) {
        fail("aclnnShmemSdmaStarsQueryGetWorkspaceSize failed");
        break;
      }
      if (wsSize > 0 && aclrtMalloc_(&aclnnWs, wsSize,
                                     detail::kAclMemMallocHugeFirst) != 0) {
        fail("aclrtMalloc of the operator workspace failed");
        break;
      }
      if (starsQueryExec_(aclnnWs, wsSize, executor, stream) != 0) {
        fail("aclnnShmemSdmaStarsQuery failed");
        break;
      }
      if (aclrtSynchronizeStream_(stream) != 0) {
        fail("aclrtSynchronizeStream after the query failed");
        break;
      }
      ok = true;
    } while (false);

    if (aclnnWs)
      aclrtFree_(aclnnWs);
    releaseTensor(input);
    releaseTensor(output);
    aclrtDestroyStream_(stream);
    return ok;
  }

  bool inited_{false};
  std::string error_;
  uint32_t reportedQueueNum_{0};
  std::vector<detail::HostStreamInfo> streams_;
  void *recordsDevice_{nullptr};
  void *workspaceDevice_{nullptr};
  void *streamsDevice_{nullptr};
  void *opResDevice_{nullptr};

  void *aclHandle_{nullptr};
  void *rtHandle_{nullptr};
  void *opapiHandle_{nullptr};
  void *nnopHandle_{nullptr};

  AclrtCreateStreamWithConfigFn aclrtCreateStreamWithConfig_{nullptr};
  AclrtDestroyStreamFn aclrtDestroyStream_{nullptr};
  AclrtStreamGetIdFn aclrtStreamGetId_{nullptr};
  AclrtGetCurrentContextFn aclrtGetCurrentContext_{nullptr};
  AclrtGetDeviceFn aclrtGetDevice_{nullptr};
  AclrtMallocFn aclrtMalloc_{nullptr};
  AclrtFreeFn aclrtFree_{nullptr};
  AclrtMemsetFn aclrtMemset_{nullptr};
  AclrtMemcpyFn aclrtMemcpy_{nullptr};
  AclrtSynchronizeStreamFn aclrtSynchronizeStream_{nullptr};
  AclrtSetStreamAttributeFn aclrtSetStreamAttribute_{nullptr};

  RtStreamGetSqidFn rtStreamGetSqid_{nullptr};
  RtStreamGetCqidFn rtStreamGetCqid_{nullptr};
  RtGetDeviceInfoFn rtGetDeviceInfo_{nullptr};

  AclCreateTensorFn aclCreateTensor_{nullptr};
  AclDestroyTensorFn aclDestroyTensor_{nullptr};
  StarsQueryGetWsSizeFn starsQueryGetWsSize_{nullptr};
  StarsQueryExecFn starsQueryExec_{nullptr};
};

} // namespace mlir::pto::comm

#endif // PTO_TEST_COMM_ASYNCWORKSPACE_H
