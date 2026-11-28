// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string_view>

#include <acl/acl.h>

#if defined(__CCE_AICORE__)
#include "a3_comm_p2p-pto.cpp"
#endif

namespace {

// Match the PTO transfer width. Each allocation holds a source, a destination,
// and a guard region.
constexpr int32_t kCount = 128;
constexpr int32_t kGuard = -7;
using Buffer = std::array<int32_t, 3 * kCount>;
constexpr size_t kBytes = sizeof(Buffer);

} // namespace

__global__ __aicore__ void LaunchA3CommP2P(__gm__ int32_t *local, __gm__ int32_t *peer)
{
#if defined(__CCE_AICORE__)
    a3_comm_p2p(local, local + kCount, peer, peer + kCount);
#endif
}

namespace {

bool CheckAcl(aclError code, const char *operation)
{
    if (code == ACL_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "%s failed with ACL error %d\n", operation, code);
    return false;
}

bool ParseDeviceId(std::string_view field, int32_t &id)
{
    const char *end = field.data() + field.size();
    const auto parsed = std::from_chars(field.data(), end, id);
    return parsed.ec == std::errc{} && parsed.ptr == end && id >= 0;
}

bool GetReservedDevices(std::array<int32_t, 2> &devices)
{
    const char *reservation = std::getenv("TASK_DEVICE");
    const char *scheduled = std::getenv("TASKQUEUE_INSIDE");
    if (scheduled == nullptr || *scheduled == '\0' || reservation == nullptr) {
        std::fprintf(stderr, "A3 P2P requires a task-submit reservation\n");
        return false;
    }
    const std::string_view ids(reservation);
    const size_t comma = ids.find(',');
    const bool parsed = comma != std::string_view::npos && ParseDeviceId(ids.substr(0, comma), devices[0]) &&
                        ParseDeviceId(ids.substr(comma + 1), devices[1]);
    if (!parsed || devices[0] == devices[1]) {
        std::fprintf(stderr, "A3 P2P requires two distinct devices assigned by task-submit\n");
        return false;
    }
    return true;
}

std::string_view GetEnv(const char *name)
{
    const char *value = std::getenv(name);
    return value == nullptr ? std::string_view{} : std::string_view(value);
}

bool IsExclusiveNightlyJob()
{
    const auto repository = GetEnv("ATOMGIT_REPOSITORY");
    const auto job = GetEnv("ATOMGIT_JOB");
    return GetEnv("TASKQUEUE_INSIDE").empty() && GetEnv("ATOMGIT_ACTIONS") == "true" &&
           (repository == "cann/pto-as" || repository == "CANN/pto-as") &&
           GetEnv("ATOMGIT_WORKFLOW") == "A3 Nightly Board" &&
           (job == "a3-board" || job == "A3 nightly board validation");
}

bool FindJobDevicePair(int32_t &localId, int32_t &peerId)
{
    uint32_t count = 0;
    if (!CheckAcl(aclrtGetDeviceCount(&count), "aclrtGetDeviceCount")) {
        return false;
    }
    if (count > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
        std::fprintf(stderr, "A3 P2P received an invalid ACL device count: %u\n", count);
        return false;
    }
    // ACL enumerates the job's devices with logical IDs even when the runner
    // assigns non-contiguous physical IDs.
    const int32_t deviceCount = static_cast<int32_t>(count);
    for (int32_t local = 0; local < deviceCount; ++local) {
        for (int32_t peer = 0; peer < deviceCount; ++peer) {
            if (local == peer) {
                continue;
            }
            int32_t canAccess = 0;
            if (!CheckAcl(aclrtDeviceCanAccessPeer(&canAccess, local, peer), "aclrtDeviceCanAccessPeer")) {
                return false;
            }
            if (canAccess != 0) {
                localId = local;
                peerId = peer;
                return true;
            }
        }
    }
    std::fprintf(stderr, "A3 P2P found no compatible pair among %u job devices\n", count);
    return false;
}

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    ~DeviceBuffer()
    {
        if (context == nullptr) {
            return;
        }
        if (MakeCurrent()) {
            if (stream != nullptr) {
                CheckAcl(aclrtDestroyStream(stream), "aclrtDestroyStream");
            }
            if (peerId >= 0) {
                CheckAcl(aclrtDeviceDisablePeerAccess(peerId), "aclrtDeviceDisablePeerAccess");
            }
            if (data != nullptr) {
                CheckAcl(aclrtFree(data), "aclrtFree");
            }
        }
        CheckAcl(aclrtDestroyContext(context), "aclrtDestroyContext");
    }

    bool Init(int32_t id, const Buffer &values, aclrtMemMallocPolicy policy)
    {
        return CheckAcl(aclrtCreateContext(&context, id), "aclrtCreateContext") &&
               CheckAcl(aclrtMalloc(&data, kBytes, policy), "aclrtMalloc") &&
               CheckAcl(aclrtMemcpy(data, kBytes, values.data(), kBytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D");
    }

    bool MakeCurrent() const
    {
        return CheckAcl(aclrtSetCurrentContext(context), "aclrtSetCurrentContext");
    }

    bool EnablePeer(int32_t id)
    {
        if (!CheckAcl(aclrtDeviceEnablePeerAccess(id, 0), "aclrtDeviceEnablePeerAccess")) {
            return false;
        }
        peerId = id;
        return true;
    }

    bool Read(Buffer &values) const
    {
        return MakeCurrent() &&
               CheckAcl(aclrtMemcpy(values.data(), kBytes, data, kBytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H");
    }

    void *data = nullptr;
    aclrtStream stream = nullptr;

private:
    aclrtContext context = nullptr;
    int32_t peerId = -1;
};

Buffer MakeInput(int32_t base)
{
    Buffer values{};
    values.fill(kGuard);
    for (int32_t i = 0; i < kCount; ++i) {
        values[i] = base + i;
    }
    return values;
}

bool Verify(const Buffer &actual, const Buffer &source, const Buffer &received, const char *operation)
{
    for (int32_t i = 0; i < kCount; ++i) {
        if (actual[i] != source[i] || actual[kCount + i] != received[i] || actual[2 * kCount + i] != kGuard) {
            std::fprintf(stderr, "%s mismatch at %d: src=%d (expected %d), dst=%d (expected %d), guard=%d\n",
                         operation, i, actual[i], source[i], actual[kCount + i], received[i], actual[2 * kCount + i]);
            return false;
        }
    }
    return true;
}

bool Run(int32_t localId, int32_t peerId)
{
    const Buffer localInput = MakeInput(1000);
    const Buffer peerInput = MakeInput(-2000);
    // Destroy the local context and release peer access before freeing the remote allocation.
    DeviceBuffer peer;
    DeviceBuffer local;
    const bool initialized = peer.Init(peerId, peerInput, ACL_MEM_MALLOC_HUGE_FIRST_P2P) &&
                             local.Init(localId, localInput, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!initialized) {
        return false;
    }
    int32_t canAccess = 0;
    if (!CheckAcl(aclrtDeviceCanAccessPeer(&canAccess, localId, peerId), "aclrtDeviceCanAccessPeer")) {
        return false;
    }
    if (!canAccess) {
        std::fprintf(stderr, "Devices %d and %d do not support P2P access\n", localId, peerId);
        return false;
    }
    const bool ready = local.EnablePeer(peerId) && peer.MakeCurrent() &&
                       CheckAcl(aclrtMemP2PMap(peer.data, kBytes, localId, 0), "aclrtMemP2PMap") &&
                       local.MakeCurrent() && CheckAcl(aclrtCreateStream(&local.stream), "aclrtCreateStream");
    if (!ready) {
        return false;
    }

    std::printf("Launching A3 P2P on device %d with peer device %d\n", localId, peerId);
    LaunchA3CommP2P<<<1, nullptr, local.stream>>>(static_cast<int32_t *>(local.data),
                                                 static_cast<int32_t *>(peer.data));
    if (!CheckAcl(aclrtSynchronizeStream(local.stream), "aclrtSynchronizeStream")) {
        return false;
    }

    Buffer localResult{};
    Buffer peerResult{};
    return local.Read(localResult) && peer.Read(peerResult) && Verify(peerResult, peerInput, localInput, "TPUT") &&
           Verify(localResult, localInput, peerInput, "TGET");
}

} // namespace

int main(int argc, char **argv)
{
    const bool useJobDevices = argc == 2 && std::string_view(argv[1]) == "--job-devices";
    if (argc != 1 && (!useJobDevices || !IsExclusiveNightlyJob())) {
        std::fprintf(stderr, "A3 P2P job-device mode requires the GitCode A3 Nightly job\n");
        return EXIT_FAILURE;
    }
    std::array<int32_t, 2> reserved{};
    const bool initialized = (useJobDevices || GetReservedDevices(reserved)) && CheckAcl(aclInit(nullptr), "aclInit");
    if (!initialized) {
        return EXIT_FAILURE;
    }
    // Use ACL's translation so runtime visibility masks cannot redirect the test to an unreserved device.
    int32_t localId = -1;
    int32_t peerId = -1;
    const bool selected = useJobDevices
        ? FindJobDevicePair(localId, peerId)
        : CheckAcl(aclrtGetLogicDevIdByPhyDevId(reserved[0], &localId), "aclrtGetLogicDevIdByPhyDevId local") &&
              CheckAcl(aclrtGetLogicDevIdByPhyDevId(reserved[1], &peerId), "aclrtGetLogicDevIdByPhyDevId peer");
    const bool passed = selected && localId >= 0 && peerId >= 0 && localId != peerId && Run(localId, peerId);
    const bool finalized = CheckAcl(aclFinalize(), "aclFinalize");
    if (!passed || !finalized) {
        return EXIT_FAILURE;
    }
    std::puts("A3 P2P TPUT/TGET verification passed");
    return EXIT_SUCCESS;
}
