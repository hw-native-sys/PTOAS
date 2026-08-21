// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_SUPPORT_ASYNCSESSIONABI_H
#define PTO_SUPPORT_ASYNCSESSIONABI_H

#include <cstddef>
#include <cstdint>

// Layout contract shared by the host that populates an async workspace, the
// device code that reads it, and the compiler passes that generate that device
// code. Every hard-coded field position must come from here.
//
// The constants below fall into three groups that differ in who owns them, how
// often they change, and what can be done to catch a change. Keeping the groups
// apart is the point of this file's structure, because a check that suits one
// group is worthless for another.
//
//   1. Ours. The session config field order and the channel record layout are
//      invented here; the host writes them, generated code reads them, and no
//      outside party has to agree. Nothing can drift, so nothing is checked.
//
//   2. Foreign, produced elsewhere. The async workspace and its channel
//      descriptors are written by the AICPU STARS query. We only read them, so
//      the layout is not ours to fix and can move between CANN releases. It is
//      therefore confined to the host, which reads it once at init, checks it
//      against values it fed the query, and repacks into group 1. Generated
//      code never sees it.
//
//   3. Hardware, per generation. The SQE is the DMA engine's queue entry
//      format. It is fixed for a generation and changes only across them, which
//      is what the a5/a2a3 split expresses. No check is possible at build or
//      run time -- a wrong bit is simply wrong behaviour -- so the only
//      assurance is running it on each generation's silicon.
//
// An async session splits in two, because one aggregate cannot serve both
// halves:
//
//   - Config: immutable scalars, carried in a stack-local `!pto.struct` filled
//     once per core at kernel entry so later reads cost no memory traffic.
//   - Runtime: the queue tail/head pair, which tracks a hardware queue position
//     and therefore has to survive across kernel launches; a stack copy would
//     silently drop it. It stays where the engine keeps it, and the channel
//     record names its address.
//
// `!pto.struct` also rejects array fields, which independently keeps the
// per-channel state out of the config struct.

namespace mlir::pto::comm {

//===----------------------------------------------------------------------===//
// Group 1: ours
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Session config struct
//===----------------------------------------------------------------------===//

// Field positions in the session config struct, spelled in IR as
//   !pto.struct<i64, i64, i32, i32, i32, i32, i64, i64, i32, i32, i32, i32, i32>
// and addressed by `pto.struct_get` / `pto.struct_set`.
//
// Pointers are carried as `i64` because `!pto.struct` rejects pointer fields;
// consumers cast back with `pto.castptr`.
enum class SessionField : int64_t {
  // Base of the channel record table in GM, which the host builds at init. It
  // is not the async workspace itself: the workspace holds the descriptors the
  // STARS query wrote, in a layout owned elsewhere, and this points at the
  // repacked view of them.
  ContextGm = 0,
  // Base of the UB scratch used for staging. Required wherever the doorbell is
  // only reachable by MTE, since the tail has to pass through UB to get there.
  TmpBufAddr = 1,
  TmpBufSize = 2,
  // Pipe event id for that staging. It is a session field rather than a fixed
  // id so a kernel already using MTE3 events can keep the two apart.
  SyncId = 3,
  // Channel group owned by this core. Defaults to the block index, which is
  // what makes the queue state per-core rather than shared.
  ChannelIdx = 4,
  // Channels in the group. Bounds the per-channel descriptor indexing.
  ChannelNum = 5,
  // Bytes per engine post; drives how one transfer splits into SQEs.
  BlockBytes = 6,
  CommBlockOffset = 7,
  Engine = 8,
  DestRankId = 9,
  QpIdx = 10,
  Flags = 11,
  // Memory-system service class for the transfers this session posts, in the
  // MPAM sense: it shapes priority and bandwidth share, not correctness. The
  // host sets it when it builds the session, matching how the surrounding stack
  // treats QoS as a property of the communication domain rather than of one
  // transfer. It is deliberately independent of any QoS the domain itself was
  // configured with, because these posts bypass that path and drive the queue
  // directly.
  Qos = 12,
  NumFields = 13,
};

constexpr int64_t sessionFieldIndex(SessionField field) {
  return static_cast<int64_t>(field);
}

constexpr unsigned kSessionNumFields =
    static_cast<unsigned>(SessionField::NumFields);

enum class SessionEngine : uint32_t {
  Sdma = 0,
  Urma = 1,
  Rdma = 2,
};

// Bits in SessionField::Flags.
constexpr uint32_t kSessionFlagValid = 1u << 0;

//===----------------------------------------------------------------------===//
// Session template (GM)
//===----------------------------------------------------------------------===//
//
// What the host writes so a kernel does not have to spell its session out in
// constants. The kernel copies it into its own stack-local struct at entry, so
// this is a read-only initial value, not shared state: each core gets its own
// copy and may then diverge.
//
// The layout is one 8-byte slot per field, so an offset is the field index
// scaled and neither side needs a table. That wastes a few bytes on the 32-bit
// fields, which is irrelevant for a per-launch template and buys the property
// that matters: host and generated code address this through the same
// SessionField enum, so there is no second list of fields to keep in step.

namespace session_tmpl {

constexpr size_t kSlotBytes = 8;
constexpr size_t kBytes = kSessionNumFields * kSlotBytes;

constexpr size_t slotOffset(SessionField field) {
  return static_cast<size_t>(sessionFieldIndex(field)) * kSlotBytes;
}

// Fills the template by field rather than by position, so adding a field
// cannot silently shift the ones after it. Narrow fields are written into the
// low half of their slot, which is where a 32-bit load looks on a
// little-endian target.
class Builder {
public:
  Builder &set(SessionField field, uint64_t value) {
    slots_[sessionFieldIndex(field)] = value;
    return *this;
  }

  uint64_t get(SessionField field) const {
    return slots_[sessionFieldIndex(field)];
  }

  const void *data() const { return slots_; }
  static constexpr size_t bytes() { return kBytes; }

private:
  uint64_t slots_[kSessionNumFields] = {};
};

} // namespace session_tmpl

//===----------------------------------------------------------------------===//
// Channel record (64 bytes)
//===----------------------------------------------------------------------===//
//
// Everything a post needs about one channel, in a layout this project defines.
// The host fills the table at init from the descriptors the STARS query wrote,
// after checking them; generated code reads only this.
//
// The head and tail are held as addresses rather than copies because the engine
// reads and advances them in place. Resolving them here is what keeps the
// foreign descriptor layout out of generated code: a release that moves those
// fields changes two host-side address computations and nothing else.

namespace channel {

constexpr size_t kRecordBytes = 64;

constexpr size_t kSqBaseOffset = 0;    // uint64, SQE ring base
constexpr size_t kDoorbellOffset = 8;  // uint64, doorbell register base
constexpr size_t kTailAddrOffset = 16; // uint64, address of the live tail
constexpr size_t kHeadAddrOffset = 24; // uint64, address of the live head
// Queue depth less one, so wrapping is a mask rather than a division. The
// depth is a power of two and the host rejects it otherwise, which is what
// makes the two equivalent. Dividing here would be worse than slow: the
// divisor is data read from memory, and a zero one stops the core with the
// stream never completing, which takes the card down with it. A mask of zero
// merely aims every post at slot zero.
constexpr size_t kSlotMaskOffset = 32; // uint32
constexpr size_t kStreamIdOffset = 36; // uint32
// 40..63 reserved.

// Record for one channel of one group, relative to SessionField::ContextGm.
constexpr size_t recordOffset(unsigned channelIdx, unsigned channelNum,
                              unsigned channelInGroup) {
  return (static_cast<size_t>(channelIdx) * channelNum + channelInGroup) *
         kRecordBytes;
}

} // namespace channel

//===----------------------------------------------------------------------===//
// Group 2: foreign, host-only
//===----------------------------------------------------------------------===//
//
// The async workspace as the AICPU STARS query leaves it. Read once at init,
// checked, and repacked into the channel records above.
//
// Nothing here may be used by a compiler pass or reach generated code. These
// positions are a CANN-release fact, not a hardware one, and they have already
// moved once; baking them into an instruction stream turns a future move into a
// wrong address at run time instead of a failed check at init.

namespace workspace {

// Most channels the query is asked for. Descriptors past what it actually
// populated are left as allocated, so a session must not index beyond the count
// the header below reports.
constexpr unsigned kMaxChannels = 40;
constexpr unsigned kSqDepth = 2048;
// Shortest transfer the engine accepts.
constexpr unsigned kMinTransferBytes = 64;

// Flag-info header that precedes the channel table. It carries no version or
// magic, so it says nothing about the layout, but totalQueueNum does report how
// far the table was populated -- the one thing here that need not be assumed.
constexpr size_t kFlagInfoBytes = 64;
constexpr size_t kFlagInfoFlagOffset = 0;           // uint32
constexpr size_t kFlagInfoTotalQueueNumOffset = 4;  // uint32

// Region the AICPU STARS query fills in: the flag-info header followed by the
// full channel table, padded out. The host must allocate at least this much,
// because the query writes the whole table regardless of how many channels a
// session ends up using.
constexpr size_t kContextBytes = 16 * 1024;

// Per-group payload area that follows the context region. Nothing here is used
// by a plain transfer; it backs the flag/signal variants. It is still part of
// the allocation so the layout matches what the rest of the stack expects.
constexpr size_t kFlagPayloadBytesPerGroup = 512;

constexpr size_t kTotalBytes =
    kContextBytes + kMaxChannels * kFlagPayloadBytesPerGroup;

//===----------------------------------------------------------------------===//
// Channel descriptor (64 bytes). Holds both the engine-facing queue geometry
// and the mutable head/tail pair. Head and tail are adjacent and are persisted
// together as one 64-bit store, tail in the high half.
//===----------------------------------------------------------------------===//

constexpr size_t kChannelDescBytes = 64;

constexpr size_t kChannelSqHeadOffset = 0;   // uint32
constexpr size_t kChannelSqTailOffset = 4;   // uint32
constexpr size_t kChannelSqBaseOffset = 8;   // uint64, SQE ring base
constexpr size_t kChannelSqRegBaseOffset = 16; // uint64, doorbell address
constexpr size_t kChannelSqDepthOffset = 24; // uint32
constexpr size_t kChannelSqIdOffset = 28;    // uint32
constexpr size_t kChannelCqIdOffset = 32;    // uint32
constexpr size_t kChannelLogicCqIdOffset = 36; // uint32
constexpr size_t kChannelCqeAddrOffset = 40; // uint64
constexpr size_t kChannelReportCqeNumOffset = 48; // uint32
constexpr size_t kChannelStreamIdOffset = 52; // uint32
constexpr size_t kChannelDevIdOffset = 56;   // uint32

// Descriptor for one channel of one group, relative to SessionField::ContextGm.
constexpr size_t channelDescOffset(unsigned channelIdx, unsigned channelNum,
                                   unsigned channelInGroup) {
  return kFlagInfoBytes +
         (static_cast<size_t>(channelIdx) * channelNum + channelInGroup) *
             kChannelDescBytes;
}

} // namespace workspace

//===----------------------------------------------------------------------===//
// Group 3: hardware, per generation
//===----------------------------------------------------------------------===//
//
// SDMA SQE (64 bytes).
//
// A post writes only the fields a memcpy needs; the rest of the slot is left as
// the host initialized it.
//
// The two generations share the slot size, the SQE type, the stream/task word,
// the credit position, and the address pair, but differ everywhere else that
// matters: A5 moved the transfer length to offset 48, where A2/A3 keeps a link
// type, and A2/A3 puts an `ie2` bit ahead of sssv in word 4, shifting the four
// address-attribute bits up by one.
//
// This is the one group with no automatic protection. A wrong bit produces
// wrong engine behaviour with nothing to catch it, so each generation's
// constants are only as good as a run on that generation's hardware. A2/A3 has
// had one; A5 has not.

namespace sqe {

constexpr size_t kBytes = 64;

// Word 1: rtStreamId:16 | taskId:16
constexpr size_t kWord1Offset = 4;
constexpr unsigned kRtStreamIdShift = 0;
constexpr unsigned kTaskIdShift = 16;

// Word 3 holds the credit at the same position on both generations.
constexpr size_t kWord3Offset = 12;
constexpr unsigned kKernelCreditShift = 16;

constexpr size_t kWord0Offset = 0;
constexpr size_t kWord4Offset = 16;

// Address pair. Low and high halves are adjacent, so each address is one
// 64-bit store.
constexpr size_t kSrcAddrOffset = 32;
constexpr size_t kDstAddrOffset = 40;

constexpr uint32_t kTypeSdma = 11;
constexpr unsigned kTypeShift = 0;
constexpr unsigned kOpcodeShift = 0;

// Four bits on both generations, so the same session value is valid either way
// even though the field lives in a different word. The default matches what the
// reference SDMA implementation posts.
constexpr uint32_t kQosMask = 0xF;
constexpr uint32_t kQosDefault = 6;

namespace a5 {

// Word 0: type:6 | lock:1 | unlock:1 | ie:1 | preP:1 | postP:1 | wrCqe:1 |
//         ptrMode:1 | rttMode:1 | headUpdate:1 | reserved0:1 | numBlocks:16
constexpr unsigned kWrCqeShift = 11;
constexpr unsigned kNumBlocksShift = 16;

// Word 4: opcode:8 | sssv:1 | dssv:1 | sns:1 | dns:1 | sro:1 | dro:1 |
//         stride:2 | ie2:1 | compEn:1 | res4:14
constexpr unsigned kSssvShift = 8;
constexpr unsigned kDssvShift = 9;
constexpr unsigned kSnsShift = 10;
constexpr unsigned kDnsShift = 11;

constexpr size_t kLengthOffset = 48;

// Word 5: sqeId:16 | mpamPartId:8 | mpamns:1 | pmg:2 | qos:4 | d2dOffsetFlag:1
//
// QoS sits in a different word here than on A2/A3, where it shares word 4 with
// the address attributes. Nothing else in word 5 is written, so the whole word
// is the QoS field shifted into place.
constexpr size_t kWord5Offset = 20;
constexpr unsigned kQosShift = 27;

constexpr uint32_t kKernelCreditDefault = 254;

// Source and destination are both "secure, non-shareable" virtual addresses.
constexpr uint32_t kWord4Memcpy =
    (1u << kSssvShift) | (1u << kDssvShift) | (1u << kSnsShift) | (1u << kDnsShift);

// SDMA type, request a CQE, single block.
constexpr uint32_t kWord0Memcpy =
    (kTypeSdma << kTypeShift) | (1u << kWrCqeShift);

constexpr uint32_t kWord3Memcpy = kKernelCreditDefault << kKernelCreditShift;

} // namespace a5

namespace a2a3 {

// Word 0: type:6 | res1:10 | blockDim:16. There is no wrCqe bit here; the
// A2/A3 SQE reports completion without being asked.
constexpr unsigned kBlockDimShift = 16;

// Word 4: opcode:8 | ie2:1 | sssv:1 | dssv:1 | sns:1 | dns:1 | qos:4 |
//         sro:1 | dro:1 | partid:8 | mpam:1 | res6:4
constexpr unsigned kIe2Shift = 8;
constexpr unsigned kSssvShift = 9;
constexpr unsigned kDssvShift = 10;
constexpr unsigned kSnsShift = 11;
constexpr unsigned kDnsShift = 12;
constexpr unsigned kQosShift = 13;

constexpr size_t kLengthOffset = 28;

// Byte 48 starts linkType:8 followed by three reserved bytes, so the whole
// word is written at once.
constexpr size_t kLinkTypeOffset = 48;
constexpr uint32_t kLinkTypeNone = 255;

constexpr uint32_t kKernelCreditDefault = 240;

// QoS is not folded in here: it comes from the session, so the expansion ors it
// into this word at run time.
constexpr uint32_t kWord4Memcpy =
    (1u << kSssvShift) | (1u << kDssvShift) | (1u << kSnsShift) |
    (1u << kDnsShift);

constexpr uint32_t kWord0Memcpy = kTypeSdma << kTypeShift;

constexpr uint32_t kWord3Memcpy = kKernelCreditDefault << kKernelCreditShift;

} // namespace a2a3

// Doorbell address relative to the channel's sq_reg_base.
constexpr size_t kDoorbellOffsetA5 = 0;
constexpr size_t kDoorbellOffsetA2A3 = 8;

} // namespace sqe

} // namespace mlir::pto::comm

#endif // PTO_SUPPORT_ASYNCSESSIONABI_H
