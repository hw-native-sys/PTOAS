# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check exact slot-channel dedup and actual allocated Mat event protocols.

The independent protocol replay is derived from MR403's fixture oracle. Only
Mat read recognition is extended to include full-size TMOV operations. These
are static compilation checks, not numerical/device or performance tests.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess

from check_slot_sync_protocol import check as check_protocol


def phase_families(log, label):
    marker = f"[PTOInsertSync Debug] {label} === //"
    if marker not in log:
        raise ValueError(f"missing synchronization phase {label}")
    section = log.split(marker, 1)[1].split("// ========================================= //", 1)[0]
    families = defaultdict(dict)
    node = None
    for line in section.splitlines():
        header = re.match(r"\s*\[\s*(\d+)\]\s+(?:COMPOUND|LOOP|BRANCH|PLACE_HOLDER)\b", line)
        if header:
            node = int(header[1])
        event = re.search(
            r"(PRE |POST): (set_flag|wait_flag) <PIPE_MTE1 -> PIPE_MTE2> idx=(\d+)(.*)", line
        )
        if not event or "useless" in event[4]:
            continue
        index = int(event[3])
        kind = event[2]
        if kind in families[index]:
            raise ValueError(f"duplicate live {kind} endpoint in family {index}")
        lanes = re.search(r"eventIdNum=(\d+)", event[4])
        loop = re.search(r"forEnd=(\d+)", event[4])
        families[index][kind] = (node, event[1].strip(), int(lanes[1]) if lanes else 1,
                                 int(loop[1]) if loop else None)
    return dict(families)


def check_dedup(log, trips):
    before = phase_families(log, "After Remove Redundant Sync")
    after = phase_families(log, "After Slot Event Deduplication")
    if trips == 0 and not before:
        assert not after
        return 0
    assert len(before) == 8, before
    assert len(after) == 2, after
    for index, pair in after.items():
        assert pair == before[index], ("surviving lifetime changed", index, before[index], pair)
        assert set(pair) == {"set_flag", "wait_flag"}, pair
        assert pair["set_flag"][1] == "POST" and pair["wait_flag"][1] == "PRE", pair
        assert pair["set_flag"][2] == pair["wait_flag"][2] == 2, pair
        assert pair["set_flag"][3] == pair["wait_flag"][3], pair
    assert len(set(before) - set(after)) == 6
    return len(after)


def compile_fixture(args, source, tag, enabled):
    inp = args.work_dir / f"{tag}.pto"
    cpp = args.work_dir / f"{tag}.cpp"
    log = args.work_dir / f"{tag}.log"
    inp.write_text(source)
    command = [args.compiler, str(inp), "--pto-arch=a3", "--enable-insert-sync", "--pto-level=level2",
               "--pto-insert-sync-debug=3", "--mlir-print-ir-after=pto-insert-sync",
               "--check-sync-ids", "--check-sync-interference", "--check-sync-self-coverage", "-o", str(cpp)]
    if enabled is not None:
        command.append(f"--pto-deduplicate-slot-events={'true' if enabled else 'false'}")
    with log.open("w") as output:
        process = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
    assert process.returncode == 0, log.read_text()[-6000:]
    return cpp.read_text(), log.read_text()


def check_cpp(cpp, kind):
    assert cpp.count("pipe_barrier(PIPE_ALL)") == 1
    assert "ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll)" in cpp
    kernel = cpp.split("AICORE void outer_pipe_matmul", 1)[1]
    assert "pipe_barrier(PIPE_ALL)" not in kernel
    assert ("TEXTRACT(" if kind == "textract" else "TMOV(") in kernel
    if kind == "tmov":
        assert "TEXTRACT(" not in kernel


def compute_sequence(cpp):
    return re.findall(r"\b(TLOAD|TEXTRACT|TMOV|TMATMUL_ACC|TMATMUL|TSTORE)\(", cpp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=("textract", "tmov"), required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    original = args.input.read_text()
    chunk = 192 if args.kind == "textract" else 96
    results = []
    for trips in (0, 1, 2, 3, 8, 31, 32, 33):
        source = original.replace("arith.constant 32 : index", f"arith.constant {trips} : index")
        # Keep a valid input tensor shape in the zero-trip synchronization test.
        source = source.replace(f"arith.constant {32 * chunk} : index",
                                f"arith.constant {max(trips, 1) * chunk} : index")
        cpp, log = compile_fixture(args, source, f"n{trips}", True)
        if trips == 32:
            original_compute_sequence = compute_sequence(cpp)
        families = check_dedup(log, trips)
        if trips:
            check_cpp(cpp, args.kind)
        else:
            assert cpp.count("pipe_barrier(PIPE_ALL)") == 1
        expected_ids = 4 if families else None
        check_protocol(log, trips, expected_ids=expected_ids)
        results.append({"trips": trips, "families": families,
                        "backward_ids": expected_ids, "loop_pipe_all": 0,
                        "preserved_lifetimes": True, "protocol_replay": "PASS"})
    # Explicitly disabled optimization retains the event-exhaustion reproducer.
    disabled_cpp, disabled_log = compile_fixture(args, original, "disabled", False)
    assert disabled_cpp.count("pipe_barrier(PIPE_ALL)") == 2
    assert compute_sequence(disabled_cpp) == original_compute_sequence
    assert phase_families(disabled_log, "After Remove Redundant Sync") == phase_families(
        disabled_log, "After Slot Event Deduplication")
    check_protocol(disabled_log, 32, allow_loop_barrier=True, require_loop_barrier=True)
    # The default invocation must also select the new dedup implementation.
    default_cpp, default_log = compile_fixture(args, original, "default", None)
    check_cpp(default_cpp, args.kind)
    check_dedup(default_log, 32)
    check_protocol(default_log, 32, expected_ids=4)
    (args.work_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"PASS: {args.kind}: eight trip counts, preserved event lifetimes, 4 IDs, no loop PIPE_ALL")


if __name__ == "__main__":
    main()
