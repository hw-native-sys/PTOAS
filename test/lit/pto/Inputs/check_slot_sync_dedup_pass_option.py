# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check the standalone pass setting without the ptoas CLI option registry."""
from pathlib import Path
import sys

from check_slot_sync_dedup_pipeline import check_dedup, phase_families

disabled = Path(sys.argv[1]).read_text()
enabled = Path(sys.argv[2]).read_text()
before = phase_families(disabled, "After Remove Redundant Sync")
after = phase_families(disabled, "After Slot Event Deduplication")
assert len(before) == 8 and before == after
assert check_dedup(enabled, 32) == 2
print("PASS: standalone pass option independently enables/disables slot dedup")
