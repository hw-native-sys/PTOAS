# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise real resource exhaustion after exact protocol deduplication."""
from pathlib import Path
import sys

from check_slot_sync_dedup_pipeline import phase_families
from check_slot_sync_protocol import check


text = Path(sys.argv[1]).read_text()
before = phase_families(text, "After Remove Redundant Sync")
after = phase_families(text, "After Slot Event Deduplication")
assert len(before) == 20, before
assert len(after) == 5, after
for index, pair in after.items():
    assert pair == before[index], (index, before[index], pair)
    assert set(pair) == {"set_flag", "wait_flag"}
    assert pair["set_flag"][2] == pair["wait_flag"][2] == 2
check(text, 3, expected_ids=8, allow_loop_barrier=True,
      writes_per_trip=5, roots=5, reads_per_trip=10,
      require_loop_barrier=True)
print("PASS: 20 -> 5 families; existing lifetimes; eight-ID pool and safe fallback")
