// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef VMILAYOUTSUPPORTGROUPBROADCASTTABLES_INC
#define VMILAYOUTSUPPORTGROUPBROADCASTTABLES_INC

// Grouped broadcast layout facts, consumed by the shared layout queries.

struct GroupBroadcastLayoutPattern {
  GroupBlockPattern block;
  LayoutPattern sourceLayout;
  LayoutPattern resultLayout;
  ElementBitsPattern elementBits = bits<8, 16, 32, 64>();
  bool integerOnly = false;
};

static constexpr GroupBroadcastLayoutPattern kGroupBroadcastLayoutPatterns[] = {
    {gbCompact(), gs(8), c()},
    // Explicit integer truncation preserves each group slot's low byte.
    {gbCompact(), gs(8, 2), c(), bits<8>(), true},
    {gbCompact(), gs(8, 4), c(), bits<8>(), true},
    {gbCompact(), gs(1), c()},
    {gb(1, 4), gs(8), ls(4)},
    {gb(1, 2), gs(8), ls(2)},
    {gb(1), gs(8), c()},
    {gb(1), gs(8), ls(2)},
    {gb(1), gs(8, 2), c()},
    {gb(1), gs(8, 4), c()},
    {gb(2), gs(8), c()},
    {gb(2), gs(8), d(2)},
    {gb(2), gs(8), bd(2)},
    // bf16 G=4 on L=128: groupSize=32 = two 32B blocks; slots=1 source.
    {gb(2), gs(1), c()},
    {gb(4), gs(8), c()},
    // Group-4 broadcasts whose logical result occupies half of a physical part
    // use lane-stride two so the partial tail is represented without a padded
    // contiguous chunk.  The fork states this row too (fork
    // VMILayoutSupport.cpp:857-860); it was the one row of that block the port
    // did not carry over, which is why pto.vmi.group_broadcast with a
    // num_groups = 4 source and a lane-strided result exposed no relation at all
    // and aborted the conformance run on its case at line 122.
    {gb(4), gs(8), ls(2)},
    {gb(4), gs(8), d(4)},
    {gb(4), gs(8), bd(4)},
    {gb(4), gs(1), c()},
    {gbFull(), gs(8), c()},
    {gbFull(), gs(1), c()},
    {gbFull(2), gs(1), d(2)},
    {gbFull(4), gs(1), d(4)},
    // The selector materializer scales both the ramp and its base slot by
    // source lane stride. Register the integer sum cases without changing
    // existing floating-point broadcast layout choices.
    {gbCompact(), gs(8, 2), c(), bits<16>(), true},
    {gb(1, 4), gs(8, 2), ls(4), bits<16>(), true},
    {gb(1, 2), gs(8, 2), ls(2), bits<16>(), true},
    {gb(1), gs(8, 2), ls(2), bits<16>(), true},
    {gb(2), gs(8, 2), c(), bits<16>(), true},
    {gb(2), gs(8, 2), d(2), bits<16>(), true},
    {gb(2), gs(8, 2), bd(2), bits<16>(), true},
    {gb(4), gs(8, 2), c(), bits<16>(), true},
    {gb(4), gs(8, 2), d(4), bits<16>(), true},
    {gb(4), gs(8, 2), bd(4), bits<16>(), true},
    {gbFull(), gs(8, 2), c(), bits<16>(), true},
};

#endif // VMILAYOUTSUPPORTGROUPBROADCASTTABLES_INC
