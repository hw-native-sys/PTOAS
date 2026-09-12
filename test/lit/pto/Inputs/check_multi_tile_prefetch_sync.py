# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check emitted flag balance and cross-pipe happens-before edges for #1519.

This evaluates scalar IR and builds vector clocks for the issued pipe commands.
It checks binary event tokens and tile RAW/WAR/WAW ordering, without relying on
the CPU simulator's no-op flag stubs or on a particular event-id assignment.
"""

import operator
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from ptoas.mlir.dialects import pto
from ptoas.mlir.ir import Context, IntegerAttr, Module


PIPES = ("PIPE_MTE2", "PIPE_V", "PIPE_MTE3")
BINARY = {
    "arith.addi": operator.add,
    "arith.subi": operator.sub,
    "arith.muli": operator.mul,
    "arith.andi": operator.and_,
    "arith.remui": operator.mod,
    "arith.remsi": operator.mod,
}
COMPARE = (operator.eq, operator.ne, operator.lt, operator.le, operator.gt,
           operator.ge, operator.lt, operator.le, operator.gt, operator.ge)


class Protocol:
    def __init__(self):
        self.events = {}
        self.clocks = {pipe: [0] * len(PIPES) for pipe in PIPES}
        self.readers = {}
        self.writers = {}

    def flag(self, op, values):
        src = re.search(r"PIPE_\w+", str(op.attributes["src_pipe"])).group()
        dst = re.search(r"PIPE_\w+", str(op.attributes["dst_pipe"])).group()
        event = values[0] if values else int(
            re.search(r"EVENT_ID(\d+)", str(op.attributes["event_id"])).group(1))
        key = (src, dst, event)
        if "set_flag" in op.name:
            if key in self.events:
                raise ValueError(f"duplicate set without wait: {key}")
            self.events[key] = tuple(self.clocks[src])
        else:
            if key not in self.events:
                raise ValueError(f"wait without set: {key}")
            released = self.events.pop(key)
            self.clocks[dst] = list(map(max, self.clocks[dst], released))

    def require_order(self, earlier, pipe, address, hazard):
        if any(a > b for a, b in zip(earlier, self.clocks[pipe])):
            raise ValueError(f"unordered {hazard} at tile address {address}, pipe {pipe}")

    def access(self, pipe, reads, writes):
        self.clocks[pipe][PIPES.index(pipe)] += 1
        clock = tuple(self.clocks[pipe])
        for address in reads:
            if address in self.writers:
                self.require_order(self.writers[address], pipe, address, "RAW")
            self.readers.setdefault(address, {})[pipe] = clock
        for address in writes:
            if address in self.writers:
                self.require_order(self.writers[address], pipe, address, "WAW")
            for reader in self.readers.get(address, {}).values():
                self.require_order(reader, pipe, address, "WAR")
            self.readers[address] = {}
            self.writers[address] = clock

    def barrier(self, op):
        if "PIPE_ALL" in str(op):
            joined = list(map(max, *self.clocks.values()))
            self.clocks = {pipe: list(joined) for pipe in PIPES}


def scalar(op, values):
    if op.name == "arith.constant":
        return IntegerAttr(op.attributes["value"]).value
    if op.name in BINARY:
        return BINARY[op.name](*values)
    if op.name == "arith.cmpi":
        return COMPARE[IntegerAttr(op.attributes["predicate"]).value](*values)
    if op.name == "arith.select":
        return values[1] if values[0] else values[2]
    if op.name in ("arith.index_cast", "arith.index_castui"):
        return values[0]
    raise ValueError(f"unsupported scalar operation: {op.name}")


def run_loop(op, values, env, protocol):
    lower, upper, step, *carried = values
    body = op.regions[0].blocks[0]
    for induction in range(lower, upper, step):
        env.update(zip(body.arguments, (induction, *carried)))
        carried = run_block(body, env, protocol)
    env.update(zip(op.results, carried))


def run_block(block, env, protocol):
    for view in block.operations:
        op = view.operation
        values = [env[value] for value in op.operands]
        if op.name.startswith("arith."):
            env[op.results[0]] = scalar(op, values)
        elif op.name == "scf.for":
            run_loop(op, values, env, protocol)
        elif op.name == "scf.if":
            region = op.regions[0 if values[0] else 1]
            if region.blocks:
                results = run_block(region.blocks[0], env, protocol)
                env.update(zip(op.results, results))
        elif op.name in ("scf.yield", "func.return"):
            return values
        elif op.name == "pto.alloc_tile":
            env[op.results[0]] = values[0]
        elif "flag" in op.name:
            protocol.flag(op, values)
        elif op.name == "pto.tload":
            protocol.access("PIPE_MTE2", [], [values[-1]])
        elif op.name == "pto.tadd":
            protocol.access("PIPE_V", values[:2], [values[2]])
        elif op.name == "pto.barrier":
            protocol.barrier(op)
        else:
            raise ValueError(f"unsupported test operation: {op.name}")
    return []


def check_module(text, label):
    with Context() as context:
        pto.register_dialect(context, load=True)
        module = Module.parse(text)
        for func in module.body.operations:
            body = func.regions[0].blocks[0]
            for upper in range(-1, 13):
                protocol = Protocol()
                env = dict(zip(body.arguments, (None, upper, 1)))
                try:
                    run_block(body, env, protocol)
                    if protocol.events:
                        raise ValueError(f"undrained events: {tuple(protocol.events)}")
                except ValueError as error:
                    raise ValueError(f"{label}, upper={upper}: {error}") from error


def variants(source):
    yield "remui", source
    yield "remsi", source.replace("arith.remui", "arith.remsi")
    yield "andi", source.replace("arith.remui", "arith.andi").replace(
        "%i,    %c2", "%i,    %c1").replace("%next, %c2", "%next, %c1")
    yield "lower_one", source.replace("%i = %c0 to", "%i = %c1 to").replace(
        "%mb[%c0]", "%mb[%c1]")
    yield "step_two", source.replace("step %c1", "step %c2")
    yield "dynamic_lower", source.replace(
        "%n : index)", "%n : index, %lower : index)").replace(
            "%i = %c0 to", "%i = %lower to").replace("%mb[%c0]", "%mb[%c1]")
    for count in (3, 4, 8, 16):
        yield f"count_{count}", source.replace(
            "count=2", f"count={count}").replace(
                "%c2 = arith.constant 2", f"%c2 = arith.constant {count}")
    multiplied = source.replace("%next     = arith.addi  %i,", (
        "%c3 = arith.constant 3 : index\n"
        "      %scaled = arith.muli %i, %c3 : index\n"
        "      %next     = arith.addi  %scaled,"))
    yield "odd_multiplier", multiplied.replace("%i,    %c2", "%scaled,    %c2")
    carried = source.replace(
        "scf.for %i = %c0 to %n step %c1 {",
        "%last_slot = scf.for %i = %c0 to %n step %c1 "
        "iter_args(%slot = %c0) -> (index) {")
    carried = carried.replace(
        "%cur_idx  = arith.remui %i,    %c2 : index",
        "%cur_idx = arith.addi %slot, %c0 : index").replace(
        "%next_idx = arith.remui %next, %c2 : index",
        "%next_idx = arith.subi %c1, %slot : index").replace(
        "    }\n    return", "      scf.yield %next_idx : index\n    }\n    return")
    yield "loop_carried", carried
    epilogue = """    %last_idx = arith.remui %n, %c2 : index
    %last = pto.multi_tile_get %mb[%last_idx] : !pto.multi_tile_buf<vec, 16x16xf16, count=2>
      -> !pto.tile_buf<vec, 16x16xf16>
    pto.tadd ins(%last, %last : !pto.tile_buf<vec, 16x16xf16>, !pto.tile_buf<vec, 16x16xf16>)
      outs(%acc : !pto.tile_buf<vec, 16x16xf16>)
"""
    yield "epilogue", source.replace("    return", epilogue + "    return")
    yield "epilogue_remsi", source.replace(
        "    return", epilogue + "    return").replace("arith.remui", "arith.remsi")
    yield "nested", source.replace(
        "    // preload iter", "    scf.for %outer = %c0 to %c2 step %c1 {\n    // preload iter").replace(
            "    return", "    }\n    return")


def main():
    compiler = shutil.which(sys.argv[1])
    if compiler is None:
        raise FileNotFoundError(sys.argv[1])
    source = Path(sys.argv[2]).resolve().read_text(encoding="utf-8")
    check_module(Path(sys.argv[3]).resolve().read_text(encoding="utf-8"), "repro")
    with tempfile.TemporaryDirectory(prefix="pto-prefetch-sync-") as directory:
        root = Path(directory)
        for label, text in variants(source):
            for level in ("level2", "level3"):
                if level == "level3":
                    text = text.replace(
                        "%mb = pto.alloc_multi_tile :", (
                            "%base = arith.constant 0 : i64\n"
                            "    %acc_base = arith.constant 8192 : i64\n"
                            "    %mb = pto.alloc_multi_tile addr = %base :")).replace(
                                "%acc = pto.alloc_tile :", "%acc = pto.alloc_tile addr = %acc_base :")
                input_path = root / f"{label}-{level}.pto"
                output_path = root / f"{label}-{level}-sync.pto"
                input_path.write_text(text, encoding="utf-8")
                result = subprocess.run(
                    [compiler, str(input_path), "--enable-insert-sync", "--pto-arch=a3",
                     f"--pto-level={level}", "--emit-pto-ir", "-o", str(output_path)],
                    check=False, capture_output=True, text=True, timeout=60)
                if result.returncode != 0:
                    raise ValueError(f"{label}-{level}: compiler failed\n{result.stderr}")
                check_module(output_path.read_text(encoding="utf-8"), f"{label}-{level}")
    print("prefetch sync: balanced events and ordered tile accesses")


if __name__ == "__main__":
    main()
