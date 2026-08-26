// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// A C entry surface over AsyncWorkspace, so a Python test can drive the engine
// path without a second copy of the session ABI.
//
// The point of this file is what it does not expose. Every field position, slot
// width and default in AsyncSessionABI.h stays on this side: a caller asks for a
// session by naming values, never offsets, and gets back an opaque byte image.
// Nothing here hands out a struct layout for a caller to mirror, because a
// mirrored layout is exactly the second list of fields that header exists to
// prevent.
//
// It also stays free of CANN headers. AsyncWorkspace resolves everything by
// dlopen, so this translation unit compiles with a plain host C++ compiler and
// needs only -ldl at link time; the toolkit is required to run, not to build.
//
// One workspace per process, addressed as a singleton. A test process needs one,
// the queue state it wraps is per-device, and a handle would add a parameter to
// every call while buying nothing. Cleanup rides on the static's destructor, so
// a caller that exits without calling finalize still releases the device memory
// and streams.

#include <cstdint>
#include <cstring>
#include <string>

#include "AsyncWorkspace.h"

namespace {

using mlir::pto::comm::AsyncWorkspace;
namespace comm = mlir::pto::comm;

// Function-local rather than a namespace-scope object, so construction order
// across translation units never matters.
AsyncWorkspace &workspace() {
  static AsyncWorkspace ws;
  return ws;
}

// Separate from AsyncWorkspace::error() so a shim-level rejection, such as a
// short output buffer, reads back through the same call as a workspace failure.
std::string &lastError() {
  static std::string error;
  return error;
}

int fail(const char *what) {
  lastError() = what;
  return -1;
}

} // namespace

extern "C" {

// --- lifecycle ---------------------------------------------------------------

// Build the workspace for `channels` channels on the currently selected device.
// Returns 0 on success, -1 with pto_async_ws_error() set otherwise.
//
// The device must already be selected. Under the PTODSL ST harness that is
// torch_npu's doing, which is the arrangement this shim exists to test: the
// workspace has to attach to a device and context someone else set up.
int pto_async_ws_init(unsigned channels) {
  lastError().clear();
  if (!workspace().init(channels)) {
    lastError() = workspace().error();
    return -1;
  }
  return 0;
}

void pto_async_ws_finalize() {
  lastError().clear();
  workspace().finalize();
}

// Valid until the next call into this shim, which may overwrite the buffer.
// Callers are expected to copy it, as a ctypes c_char_p restype does.
const char *pto_async_ws_error() {
  if (!lastError().empty()) {
    return lastError().c_str();
  }
  return workspace().error().c_str();
}

// --- what the kernel needs ---------------------------------------------------

// SessionField::ContextGm, or 0 before a successful init.
//
// A device address has to cross this boundary as an integer, which is the same
// reason AsyncSessionABI.h carries pointers as i64: the session struct rejects
// pointer fields. It stays opaque either way -- it is produced here, consumed by
// pto_async_session_pack, and never interpreted by the caller.
uint64_t pto_async_ws_context_gm() {
  return reinterpret_cast<uint64_t>(workspace().contextGm());
}

unsigned pto_async_ws_channel_count() { return workspace().channelCount(); }

// Channels the STARS query reported populating, which bounds what a session may
// index however many channels were asked for.
uint32_t pto_async_ws_reported_queue_num() {
  return workspace().reportedQueueNum();
}

// --- observation -------------------------------------------------------------

// The live queue tail for one channel, or -1 on failure so a caller can tell a
// failed read from a tail that legitimately reads zero.
//
// This is the only way to see that a post happened the way it was asked to.
// Correct data at the destination says the bytes moved; one oversized entry
// moves the same bytes as a correct split, and only the tail separates them.
long long pto_async_ws_sq_tail(unsigned channel_idx, unsigned channel_num) {
  static_assert(comm::workspace::kChannelSqTailOffset + sizeof(uint32_t) <=
                    comm::workspace::kChannelDescBytes,
                "the queue tail must lie inside one channel descriptor");

  lastError().clear();
  uint8_t desc[comm::workspace::kChannelDescBytes];
  if (!workspace().readChannelDescriptor(channel_idx, channel_num, 0, desc)) {
    lastError() = workspace().error();
    return -1;
  }
  uint32_t tail = 0;
  std::memcpy(&tail, desc + comm::workspace::kChannelSqTailOffset,
              sizeof(tail));
  return static_cast<long long>(tail);
}

// --- session template --------------------------------------------------------

// How large a buffer pto_async_session_pack needs. Asked for rather than
// assumed, so the slot width stays a fact of the C++ header.
uint32_t pto_async_session_bytes() { return comm::session_tmpl::kBytes; }

// Values and limits a caller would otherwise hard-code.
uint32_t pto_async_session_qos_default() { return comm::sqe::kQosDefault; }

uint32_t pto_async_ws_min_transfer_bytes() {
  return comm::workspace::kMinTransferBytes;
}

uint32_t pto_async_ws_max_channels() { return comm::workspace::kMaxChannels; }

// Build one session image into `out`, which must hold at least
// pto_async_session_bytes() bytes. Returns 0 on success, -1 otherwise.
//
// Arguments are named values rather than a struct on purpose: a struct would
// have a layout the caller has to agree with, which is the duplication this shim
// avoids. ContextGm comes from the live workspace, and Engine and Flags have
// only one sensible value here, so none of the three is a parameter.
//
// tmp_buf_addr and sync_id matter on A2/A3, where the doorbell is reachable only
// by MTE and the tail therefore goes out through UB. A5 writes it with st_dev
// and reads neither.
int pto_async_session_pack(void *out, uint32_t out_bytes, uint32_t channel_idx,
                           uint32_t channel_num, uint64_t block_bytes,
                           uint64_t tmp_buf_addr, uint32_t tmp_buf_size,
                           uint32_t sync_id, uint64_t comm_block_offset,
                           uint32_t qos, uint32_t dest_rank_id) {
  // The destination bound is checked below; this is the source side.
  static_assert(comm::session_tmpl::Builder::bytes() ==
                    comm::session_tmpl::kBytes,
                "the builder must hold exactly one session template");

  lastError().clear();
  if (out == nullptr) {
    return fail("session output buffer is null");
  }
  if (out_bytes < comm::session_tmpl::kBytes) {
    return fail("session output buffer is shorter than the session template");
  }

  // A zero block size would make the engine expansion's trip count degenerate.
  // It clamps the value at run time rather than dividing by it, so this is a
  // caller error to report here instead of a wrong transfer to discover later.
  if (block_bytes == 0) {
    return fail("session block_bytes must be non-zero");
  }
  if (channel_num == 0) {
    return fail("session channel_num must be non-zero");
  }

  comm::session_tmpl::Builder tmpl;
  tmpl.set(comm::SessionField::ContextGm, pto_async_ws_context_gm())
      .set(comm::SessionField::TmpBufAddr, tmp_buf_addr)
      .set(comm::SessionField::TmpBufSize, tmp_buf_size)
      .set(comm::SessionField::SyncId, sync_id)
      .set(comm::SessionField::ChannelIdx, channel_idx)
      .set(comm::SessionField::ChannelNum, channel_num)
      .set(comm::SessionField::BlockBytes, block_bytes)
      .set(comm::SessionField::CommBlockOffset, comm_block_offset)
      .set(comm::SessionField::Engine,
           static_cast<uint64_t>(comm::SessionEngine::Sdma))
      .set(comm::SessionField::DestRankId, dest_rank_id)
      .set(comm::SessionField::QpIdx, 0)
      .set(comm::SessionField::Flags, comm::kSessionFlagValid)
      .set(comm::SessionField::Qos, qos);

  std::memcpy(out, tmpl.data(), comm::session_tmpl::kBytes);
  return 0;
}

} // extern "C"
