# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Keep legacy fallback when an unrelated slot rotation has no boundary proof."""
import argparse
from pathlib import Path

from check_slot_sync_dedup_pipeline import compile_fixture, phase_families
from check_slot_sync_protocol import check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    original = args.input.read_text()
    for trips in (2, 3, 8):
        source = original.replace("%trips = arith.constant 2", f"%trips = arith.constant {trips}")
        source = source.replace("%total = arith.constant 576", f"%total = arith.constant {(trips + 1) * 192}")
        disabled_cpp, disabled_log = compile_fixture(args, source, f"n{trips}-off", False)
        cpp, log = compile_fixture(args, source, f"n{trips}-on", True)
        default_cpp, default_log = compile_fixture(args, source, f"n{trips}-default", None)
        assert cpp == default_cpp == disabled_cpp, "guard must preserve disabled codegen"
        for text in (log, default_log, disabled_log):
            before = phase_families(text, "After Remove Redundant Sync")
            after = phase_families(text, "After Slot Event Deduplication")
            assert len(before) == 5 and after == before, "whole-function guard changed a live family"
            check(text, trips, allow_loop_barrier=True, writes_per_trip=2,
                  extra_writes=1, roots=2, reads_per_trip=3, require_loop_barrier=True)
        assert cpp.count("pipe_barrier(PIPE_ALL)") == 2, "missing loop fallback or completion barrier"
    print("PASS: shifted prefetch: whole-function guard, unchanged codegen and safe fallback")


if __name__ == "__main__":
    main()
