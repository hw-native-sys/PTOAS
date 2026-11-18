# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check the emitted two-slot Mat protocol, independently of SyncIR grouping.

Interpret the arithmetic, loop and branch operations in the post-insert-sync IR.
Build a command graph from pipe order and matched event signals, then require
happens-before for load-to-read and read-to-overwrite hazards. This also detects
an early FREE signal which balances perfectly but does not protect the last
reader. The checker does not execute matrix arithmetic or claim numerical
correctness. Short loops enumerate all branch choices; longer loops sample
actual, uniform and alternating branch choices.
Memory keys are logical allocation roots and slots; unknown physical aliases
are checked by proof-rejection fixtures, not by this memory model. FREE lanes
also require happens-before from the previous wait to the next set; this is
not an arbitrary asynchronous event-capacity state exploration.
"""

import argparse
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
import re


VALUE = r"%[-\w.$]+"
FREE = ("PIPE_MTE1", "PIPE_MTE2")


def fail(message):
    raise ValueError(message)


def extract_function(text):
    start = re.search(r"^[ \t]*func\.func @", text, re.MULTILINE)
    if not start:
        fail("missing post-insert-sync function")
    lines = text[start.start():].splitlines()
    # A compiler dump can include later diagnostics: stop at the function end.
    statements = []
    stack = [statements]
    for raw in lines[1:]:
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("scf.for "):
            item = ["for", line, []]
            stack[-1].append(item)
            stack.append(item[2])
        elif line.startswith("scf.if "):
            item = ["if", line, [], []]
            stack[-1].append(item)
            stack.append(item[2])
        elif line.startswith("} else {"):
            if len(stack) < 2:
                fail("unmatched else")
            stack.pop()
            item = stack[-1][-1]
            if item[0] != "if":
                fail("else does not belong to an if")
            stack.append(item[3])
        elif line.startswith("}"):
            if len(stack) == 1:
                break
            stack.pop()
        else:
            stack[-1].append(["op", line])
    if len(stack) != 1:
        fail("unclosed loop or branch")
    return statements


def find_kind(block, kind):
    result = []
    for item in block:
        if item[0] == kind:
            result.append(item)
        if item[0] in ("for", "if"):
            result.extend(find_kind(item[2], kind))
            if item[0] == "if":
                result.extend(find_kind(item[3], kind))
    return result


class Replay:
    """A command-level ordering oracle, with no compiler dependency formulas."""

    def __init__(self, decisions=None, allow_loop_barrier=False):
        self.env = {}
        self.roots = set()
        self.pending = {}
        self.free_ids = set()
        self.free_dynamic = Counter()
        self.free_boundary = Counter()
        self.last_free_wait = {}
        self.free_reuses = []
        self.graph = defaultdict(set)
        self.last_pipe = {}
        self.labels = []
        self.accesses = defaultdict(list)
        self.decisions = iter(decisions) if decisions is not None else None
        self.in_loop = False
        self.loop_count = None
        self.tail_barriers = 0
        self.loop_barriers = 0
        self.read_count = 0
        self.write_count = 0
        self.allow_loop_barrier = allow_loop_barrier

    def value(self, text):
        if text.startswith("%"):
            if text not in self.env:
                fail(f"undefined SSA value {text}")
            return self.env[text]
        return int(text)

    def command(self, pipe, label):
        node = len(self.labels)
        self.labels.append(label)
        if pipe in self.last_pipe:
            self.graph[self.last_pipe[pipe]].add(node)
        self.last_pipe[pipe] = node
        return node

    def flag(self, line):
        match = re.search(
            r"pto\.(set_flag|wait_flag)(_dyn)?\[<(PIPE_\w+)>, <(PIPE_\w+)>, "
            r"(" + VALUE + r"|<EVENT_ID(\d+)>)\]", line
        )
        if not match:
            return False
        kind, dynamic, source, target, operand, static_id = match.groups()
        event_id = self.value(operand) if dynamic else int(static_id)
        key = source, target, event_id
        if (source, target) == FREE:
            if not 0 <= event_id < 8:
                fail(f"Mat reuse event ID outside the eight-ID pool: {event_id}")
            self.free_ids.add(event_id)
            if dynamic:
                self.free_dynamic[kind] += 1
            elif not self.in_loop:
                self.free_boundary[(kind, event_id)] += 1
        pipe = source if kind == "set_flag" else target
        node = self.command(pipe, f"{kind} {key}")
        if kind == "set_flag":
            if key in self.pending:
                fail(f"duplicate set without wait for {key}")
            self.pending[key] = node
            if (source, target) == FREE and key in self.last_free_wait:
                self.free_reuses.append((self.last_free_wait[key], node, key))
        else:
            if key not in self.pending:
                fail(f"wait without a preceding set for {key}")
            self.graph[self.pending.pop(key)].add(node)
            if (source, target) == FREE:
                self.last_free_wait[key] = node
        return True

    def scalar(self, line):
        match = re.match(r"(" + VALUE + r") = arith\.constant (-?\d+)\b", line)
        if match:
            self.env[match[1]] = int(match[2])
            return True
        match = re.match(
            r"(" + VALUE + r") = arith\.(addi|subi|muli|maxsi|remsi|remui) "
            r"(" + VALUE + r"), (" + VALUE + r")", line
        )
        if match:
            name, op, lhs, rhs = match.groups()
            a, b = self.value(lhs), self.value(rhs)
            if op in ("remsi", "remui") and (a < 0 or b <= 0):
                fail("this fixture requires nonnegative rotation operands")
            self.env[name] = {
                "addi": lambda: a + b, "subi": lambda: a - b,
                "muli": lambda: a * b, "maxsi": lambda: max(a, b),
                "remsi": lambda: a % b, "remui": lambda: a % b,
            }[op]()
            return True
        match = re.match(
            r"(" + VALUE + r") = arith\.cmpi (eq|ne|slt|sle|sgt|sge|ult|ule|ugt|uge), "
            r"(" + VALUE + r"), (" + VALUE + r")", line
        )
        if match:
            name, predicate, lhs, rhs = match.groups()
            a, b = self.value(lhs), self.value(rhs)
            self.env[name] = {
                "eq": a == b, "ne": a != b,
                "slt": a < b, "ult": a < b,
                "sle": a <= b, "ule": a <= b,
                "sgt": a > b, "ugt": a > b,
                "sge": a >= b, "uge": a >= b,
            }[predicate]
            return True
        match = re.match(
            r"(" + VALUE + r") = arith\.select (" + VALUE + r"), "
            r"(" + VALUE + r"), (" + VALUE + r")", line
        )
        if match:
            name, condition, yes, no = match.groups()
            self.env[name] = self.value(yes if self.value(condition) else no)
            return True
        return False

    def operation(self, line):
        if self.flag(line) or self.scalar(line):
            return
        match = re.match(r"(" + VALUE + r") = pto\.alloc_multi_tile\b", line)
        if match and re.search(r"tile_buf<(?:loc=)?mat\b", line):
            self.roots.add(match[1])
            self.env[match[1]] = (match[1], None)
            return
        match = re.match(
            r"(" + VALUE + r") = pto\.multi_tile_get (" + VALUE + r")\[(" + VALUE + r")\]", line
        )
        if match:
            slot = self.value(match[3])
            if match[2] in self.roots and slot not in (0, 1):
                fail(f"out-of-range Mat slot {slot}")
            self.env[match[1]] = (match[2], slot)
            return
        if "pto.barrier <PIPE_ALL>" in line:
            if self.in_loop and not self.allow_loop_barrier:
                fail("PIPE_ALL inside the loop")
            if not self.in_loop and "pto.auto_sync_tail_barrier" not in line:
                fail("unexpected non-tail PIPE_ALL")
            node = len(self.labels)
            self.labels.append("tail barrier")
            for previous in set(self.last_pipe.values()):
                self.graph[previous].add(node)
            for pipe in ("PIPE_MTE1", "PIPE_MTE2", "PIPE_M", "PIPE_FIX", "PIPE_V", "PIPE_MTE3"):
                self.last_pipe[pipe] = node
            if not self.in_loop:
                self.tail_barriers += 1
            else:
                self.loop_barriers += 1
            return
        match = re.search(r"pto\.barrier <(PIPE_\w+)>", line)
        if match:
            self.command(match[1], line)
            return
        if "pto.tload " in line:
            node = self.command("PIPE_MTE2", line)
            output = re.search(r"outs\((" + VALUE + r")\s*:", line)
            if not output:
                fail("unrecognized tload output")
            tile = self.env.get(output[1])
            if isinstance(tile, tuple) and tile[0] in self.roots:
                self.accesses[tile].append((node, "write"))
                self.write_count += 1
            return
        if "pto.textract " in line or "pto.tmov " in line:
            source = re.search(r"ins\((" + VALUE + r")(?:,|\s*:)", line)
            if not source:
                fail("unrecognized Mat read source")
            tile = self.env.get(source[1])
            node = self.command("PIPE_MTE1", line)
            if isinstance(tile, tuple) and tile[0] in self.roots:
                self.accesses[tile].append((node, "read"))
                self.read_count += 1
            return
        if "pto.tmatmul" in line:
            self.command("PIPE_M", line)
            return
        if "pto.tstore " in line:
            self.command("PIPE_FIX", line)
            return
        # Fail closed for every unmodeled operation, including an opaque call
        # with no tile operand: operand spelling does not bound its effects.
        if re.search(r"= pto\.(alloc_tile|make_tensor_view|partition_view)\b", line):
            return
        if line in ("return", "func.return", "scf.yield"):
            return
        fail(f"unhandled operation in the protocol fixture: {line}")

    def block(self, statements):
        for item in statements:
            kind, line = item[:2]
            if kind == "op":
                self.operation(line)
            elif kind == "for":
                match = re.match(
                    r"scf\.for (" + VALUE + r") = (" + VALUE + r") to "
                    r"(" + VALUE + r") step (" + VALUE + r")", line
                )
                if not match or self.in_loop:
                    fail("expected one non-nested counted loop")
                iv, lower, upper, step = match.groups()
                # A dynamic-bound rejection fixture is checked at this supplied
                # runtime bound, without claiming compile-time proof of it.
                if upper not in self.env:
                    self.env[upper] = self.runtime_upper
                iterations = range(self.value(lower), self.value(upper), self.value(step))
                self.loop_count = len(iterations)
                self.in_loop = True
                for value in iterations:
                    self.env[iv] = value
                    self.block(item[2])
                self.in_loop = False
            else:
                match = re.match(r"scf\.if (" + VALUE + r")", line)
                if not match:
                    fail("unrecognized branch condition")
                decision = self.value(match[1]) if self.decisions is None else next(self.decisions)
                self.block(item[2] if decision else item[3])

    def reachable(self, source, target):
        work, visited = [source], {source}
        while work:
            node = work.pop()
            if node == target:
                return True
            for child in self.graph[node]:
                if child not in visited:
                    visited.add(child)
                    work.append(child)
        return False

    def check(self, trips, expected_ids, writes_per_trip, extra_writes,
              roots, reads_per_trip, require_loop_barrier):
        if self.pending:
            fail(f"undrained event tokens: {sorted(self.pending)}")
        if self.loop_count != trips and not (trips == 0 and self.loop_count is None):
            fail(f"expected {trips} loop iterations, found {self.loop_count}")
        if len(self.roots) != roots and not (trips == 0 and not self.roots):
            fail(f"expected {roots} distinct Mat roots, found {len(self.roots)}")
        if expected_ids is not None and len(self.free_ids) != expected_ids:
            fail(f"expected {expected_ids} MTE1->MTE2 IDs, found {sorted(self.free_ids)}")
        if expected_ids is not None:
            if expected_ids <= 0 or expected_ids % 2:
                fail("expected a positive even number of allocated FREE lanes")
            families = expected_ids // 2
            expected_dynamic = Counter({"set_flag": families * trips, "wait_flag": families * trips})
            if self.free_dynamic != expected_dynamic:
                fail(f"expected one dynamic FREE pair per root/iteration: {self.free_dynamic}")
            expected_boundary = Counter((kind, event_id)
                                        for kind in ("set_flag", "wait_flag")
                                        for event_id in self.free_ids)
            if self.free_boundary != expected_boundary:
                fail("each FREE event lane needs exactly one prime and one drain")
        if self.tail_barriers != 1:
            fail(f"expected one completion barrier, found {self.tail_barriers}")
        if require_loop_barrier and not self.loop_barriers:
            fail("expected safe PIPE_ALL fallback for event-pool pressure")
        for previous_wait, next_set, key in self.free_reuses:
            if not self.reachable(previous_wait, next_set):
                fail(f"FREE lane reused before its preceding wait completes: {key}")
        if self.write_count != writes_per_trip * trips + extra_writes or self.read_count != reads_per_trip * trips:
            fail(f"missing Mat accesses: {self.write_count} writes, {self.read_count} reads")
        for tile, accesses in self.accesses.items():
            writer, readers = None, []
            for node, mode in accesses:
                if mode == "read":
                    if writer is None or not self.reachable(writer, node):
                        fail(f"load-ready dependency missing for {tile}: {self.labels[node]}")
                    readers.append(node)
                else:
                    for reader in readers:
                        if not self.reachable(reader, node):
                            fail(f"buffer reuse before last read completes for {tile}")
                    writer, readers = node, []


def check(text, trips, expected_ids=None, allow_loop_barrier=False,
          writes_per_trip=2, extra_writes=0, roots=2, reads_per_trip=4,
          require_loop_barrier=False):
    statements = extract_function(text)
    loops = find_kind(statements, "for")
    empty = trips == 0 and not loops
    if not empty and (len(loops) != 1 or len(find_kind(loops[0][2], "if")) != 1):
        fail("expected the fixture's one loop and one init/acc branch")
    decisions = [None]
    if trips <= 3:
        decisions.extend(product((False, True), repeat=trips))
    else:
        decisions.extend([
            [False] * trips, [True] * trips,
            [bool(i % 2) for i in range(trips)],
            [not bool(i % 2) for i in range(trips)],
        ])
    for path in decisions:
        replay = Replay(path, allow_loop_barrier)
        replay.runtime_upper = trips
        replay.block(statements)
        replay.check(trips, expected_ids, writes_per_trip, extra_writes,
                     roots, reads_per_trip, require_loop_barrier)
    print(f"Mat slot protocol verified: trips={trips}, branch replays={len(decisions)}")


def self_test():
    """The graph must reject early releases even when flag counts balance."""
    header = """
func.func @protocol() {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c3 = arith.constant 3 : index
%a = pto.alloc_multi_tile : !pto.multi_tile_buf<!pto.tile_buf<mat, 1x2xbf16>, count=2>
%b = pto.alloc_multi_tile : !pto.multi_tile_buf<!pto.tile_buf<mat, 2x1xbf16>, count=2>
"""
    prime = "\n".join(f"pto.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID{i}>]" for i in range(4))
    loop = """
scf.for %i = %c0 to %c3 step %c1 {
%s = arith.remsi %i, %c2 : index
%as = pto.multi_tile_get %a[%s] : !pto.multi_tile_buf<!pto.tile_buf<mat, 1x2xbf16>, count=2> -> !pto.tile_buf<mat, 1x2xbf16>
%bs = pto.multi_tile_get %b[%s] : !pto.multi_tile_buf<!pto.tile_buf<mat, 2x1xbf16>, count=2> -> !pto.tile_buf<mat, 2x1xbf16>
%odd = arith.cmpi eq, %s, %c1 : index
%fa = arith.select %odd, %c1, %c0 : index
%fb = arith.select %odd, %c3, %c2 : index
pto.wait_flag_dyn[<PIPE_MTE1>, <PIPE_MTE2>, %fa]
pto.tload ins(%x : !view) outs(%as : !tile)
pto.wait_flag_dyn[<PIPE_MTE1>, <PIPE_MTE2>, %fb]
pto.tload ins(%y : !view) outs(%bs : !tile)
pto.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
pto.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
%init = arith.cmpi eq, %i, %c0 : index
scf.if %init {
pto.textract ins(%as, %c0, %c0 : !tile, index, index) outs(%l0 : !left)
pto.textract ins(%bs, %c0, %c0 : !tile, index, index) outs(%r0 : !right)
pto.textract ins(%as, %c0, %c1 : !tile, index, index) outs(%l1 : !left)
pto.textract ins(%bs, %c1, %c0 : !tile, index, index) outs(%r1 : !right)
} else {
pto.textract ins(%as, %c0, %c0 : !tile, index, index) outs(%l0 : !left)
pto.textract ins(%bs, %c0, %c0 : !tile, index, index) outs(%r0 : !right)
pto.textract ins(%as, %c0, %c1 : !tile, index, index) outs(%l1 : !left)
pto.textract ins(%bs, %c1, %c0 : !tile, index, index) outs(%r1 : !right)
}
"""
    release = "pto.set_flag_dyn[<PIPE_MTE1>, <PIPE_MTE2>, %fa]\npto.set_flag_dyn[<PIPE_MTE1>, <PIPE_MTE2>, %fb]\n"
    drain = "\n".join(f"pto.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID{i}>]" for i in range(4))
    valid = (
        header + prime + loop + release + "}\n" + drain
        + "\npto.barrier <PIPE_ALL> {pto.auto_sync_tail_barrier}\n}\n"
    )
    check(valid, 3, 4)
    mutations = {
        "duplicate prime": valid.replace(prime, prime + "\n" + prime.splitlines()[0], 1),
        "missing prime": valid.replace(prime.splitlines()[0], "", 1),
        "missing drain": valid.replace(drain.splitlines()[0], "", 1),
        "early release": valid.replace(release, "", 1).replace("scf.if %init {", release + "scf.if %init {", 1),
        "missing load-ready": valid.replace("pto.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]", "", 1)
            .replace("pto.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]", "", 1),
        "opaque call": valid.replace("scf.if %init {", "func.call @unknown() : () -> ()\nscf.if %init {", 1),
    }
    for label, text in mutations.items():
        try:
            check(text, 3, 4)
        except ValueError:
            continue
        fail(f"oracle failed to reject mutation: {label}")
    short = valid.replace("%i = %c0 to %c3", "%i = %c0 to %c1", 1)
    check(short, 1, 4)
    try:
        check(short.replace(drain.splitlines()[1], "", 1), 1, 4)
    except ValueError:
        pass
    else:
        fail("oracle missed an undrained initial token of an unused slot")
    unordered = short.replace(release, "", 1).replace(
        "pto.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]",
        release + "pto.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]", 1,
    )
    try:
        check(unordered, 1, 4)
    except ValueError as error:
        if "FREE lane reused before its preceding wait completes" not in str(error):
            raise
    else:
        fail("oracle missed unordered wait-to-next-set reuse in a one-trip loop")
    family = (
        "\n[1] COMPOUND pto.tload [PIPE_MTE2]\n"
        "PRE : wait_flag <PIPE_MTE1 -> PIPE_MTE2> idx=10 eventIdNum=2\n"
        "[2] BRANCH IF_END\n"
        "POST: set_flag <PIPE_MTE1 -> PIPE_MTE2> idx=10 eventIdNum=2\n"
    )
    phases = "After Remove Redundant Sync" + family + "After Mat Slot Reuse Coalescing" + family
    unchanged_groups(phases)
    changed = phases.rsplit("eventIdNum=2", 1)
    try:
        unchanged_groups("eventIdNum=1".join(changed))
    except ValueError:
        pass
    else:
        fail("oracle missed a changed unproved event family")
    print("Mat slot protocol oracle rejected all malformed mutations")


def phase_groups(text, label):
    if label not in text:
        fail(f"missing synchronization phase {label}")
    section = text.split(label, 1)[1].split("After ", 1)[0]
    result, node = Counter(), None
    for line in section.splitlines():
        header = re.match(r"\s*\[\s*(\d+)\]\s+(COMPOUND|LOOP|BRANCH)\b", line)
        if header:
            node = int(header[1])
        event = re.search(
            r"(PRE |POST): (set_flag|wait_flag) <PIPE_MTE1 -> PIPE_MTE2> idx=(\d+)(.*)", line
        )
        if event and "useless" not in event[4]:
            count = re.search(r"eventIdNum=(\d+)", event[4])
            result[(node, event[1].strip(), event[2], int(event[3]),
                    int(count[1]) if count else 1)] += 1
    if not result:
        fail(f"expected live Mat reuse families in {label}")
    return result


def unchanged_groups(text):
    """A rejected whole-function proof must leave live WAR families intact."""
    before = phase_groups(text, "After Remove Redundant Sync")
    after = phase_groups(text, "After Mat Slot Reuse Coalescing")
    if before != after:
        fail(f"unsupported proof changed WAR families: removed={before-after}, added={after-before}")
    print("Unproved Mat reuse families remain unchanged")


def coalesced_families(text, expected):
    """Check pressure is exerted on new two-lane groups before allocation."""
    groups = phase_groups(text, "After Mat Slot Reuse Coalescing")
    families = defaultdict(Counter)
    for (_, _, kind, index, lanes), count in groups.items():
        if lanes != 2:
            fail(f"coalesced family {index} lost its two-lane shape")
        families[index][kind] += count
    if len(families) != expected:
        fail(f"expected {expected} coalesced families before allocation, found {len(families)}")
    for index, pair in families.items():
        if pair != Counter({"set_flag": 1, "wait_flag": 1}):
            fail(f"unpaired coalesced family {index}: {pair}")
    print(f"Coalesced two-lane families verified: {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ir", nargs="?", type=Path)
    parser.add_argument("--trips", type=int, default=32)
    parser.add_argument("--expected-ids", type=int)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--allow-loop-barrier", action="store_true")
    parser.add_argument("--writes-per-trip", type=int, default=2)
    parser.add_argument("--extra-writes", type=int, default=0)
    parser.add_argument("--unchanged-groups", action="store_true")
    parser.add_argument("--roots", type=int, default=2)
    parser.add_argument("--reads-per-trip", type=int, default=4)
    parser.add_argument("--require-loop-barrier", action="store_true")
    parser.add_argument("--coalesced-families", type=int)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.ir is None or not 0 <= args.trips <= 4096:
            parser.error("provide an IR dump and 0 <= trips <= 4096")
        try:
            text = args.ir.read_text(encoding="utf-8")
            if args.unchanged_groups:
                unchanged_groups(text)
            else:
                if args.coalesced_families is not None:
                    coalesced_families(text, args.coalesced_families)
                check(text, args.trips, args.expected_ids,
                      args.allow_loop_barrier, args.writes_per_trip, args.extra_writes,
                      args.roots, args.reads_per_trip, args.require_loop_barrier)
        except (ValueError, KeyError, StopIteration) as error:
            parser.exit(1, f"Mat slot protocol error: {error}\n")


if __name__ == "__main__":
    main()
