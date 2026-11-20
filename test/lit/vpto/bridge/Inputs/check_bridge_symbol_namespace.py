#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check generated names and symbol binding with configured LLVM tools.

Simplified LLVM bodies use the actual generated names. Real vendor C++/ABI
compatibility and runtime numerical results belong to the CANN/SIM suites.
"""

from pathlib import Path
import re
import sys


SYMBOL = re.compile(r"pto\.bridge\.instance @([A-Za-z0-9_]+) <pipe_init>")


def resolve(directory, variant, creates=2):
    resolved = (directory / (variant + ".resolved")).read_text(encoding="utf-8")
    symbols = SYMBOL.findall(resolved)
    if len(symbols) != 1:
        raise RuntimeError(f"equal specializations must share one instance: {resolved}")
    symbol = symbols[0]
    if not re.fullmatch(r"pto_vpto_pipe_init__[0-9a-f]{64}_(vector|cube)_0", symbol):
        raise RuntimeError(f"invalid wrapper namespace: {symbol}")
    if resolved.count(f"instance = @{symbol}") != creates:
        raise RuntimeError("every create must reference the resolved instance")
    rendered = (directory / (variant + ".wrapper")).read_text(encoding="utf-8")
    lowered = (directory / (variant + ".calls")).read_text(encoding="utf-8")
    if f"void {symbol}(" not in rendered or f"call @{symbol}(" not in lowered:
        raise RuntimeError("C++ definition and device IR declaration must use the instance symbol")
    return symbol


def write_link_inputs(directory, symbols):
    for index, symbol in enumerate(symbols):
        device = directory / f"device{index}.ll"
        wrapper = directory / f"wrapper{index}.ll"
        device.write_text(
            f"declare i32 @{symbol}()\n"
            f"define i32 @unit{index}() {{ %v = call i32 @{symbol}() ret i32 %v }}\n",
            encoding="utf-8",
        )
        wrapper.write_text(f"define i32 @{symbol}() {{ ret i32 {index + 1} }}\n", encoding="utf-8")
    main = directory / "main.ll"
    main.write_text(
        "declare i32 @unit0()\ndeclare i32 @unit1()\n"
        "define i32 @main() { %a = call i32 @unit0() %b = call i32 @unit1() "
        "%sum = add i32 %a, %b %ok = icmp eq i32 %sum, 3 "
        "%rc = select i1 %ok, i32 0, i32 1 ret i32 %rc }\n",
        encoding="utf-8",
    )


def check_linked(directory, variants, creates):
    """Verify each independent caller still resolves to its own wrapper."""
    text = (directory / "linked.ll").read_text(encoding="utf-8")
    names = [resolve(directory, variant, creates) for variant in variants]
    for index, symbol in enumerate(names):
        caller = re.search(rf"define i32 @unit{index}\(\)\s*\{{(.*?)\n\}}", text, re.DOTALL)
        wrapper = re.search(rf"define i32 @{re.escape(symbol)}\(\)\s*\{{(.*?)\n\}}", text, re.DOTALL)
        if caller is None or f"call i32 @{symbol}()" not in caller.group(1):
            raise RuntimeError(f"linked caller {index} no longer references its wrapper")
        if wrapper is None or f"ret i32 {index + 1}" not in wrapper.group(1):
            raise RuntimeError(f"linked wrapper {index} lost its independent implementation")


def split_ptodsl(source, directory):
    """Extract independent VPTO children after normal peer FIFO resolution."""
    from ptoas.mlir import ir
    from ptodsl._context import make_context

    with make_context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        kinds = []
        for child in module.body.operations:
            attrs = child.operation.attributes
            if str(attrs["pto.backend"]) != '"vpto"':
                continue
            kind = str(attrs["pto.kernel_kind"])
            if kind not in ("#pto.kernel_kind<vector>", "#pto.kernel_kind<cube>"):
                raise RuntimeError(f"unexpected child core: {kind}")
            core = "vector" if "<vector>" in kind else "cube"
            kinds.append(core)
            (directory / (core + ".pto")).write_text(str(child), encoding="utf-8")
        if sorted(kinds) != ["cube", "vector"]:
            raise RuntimeError("PTODSL must produce one independent child for each core")


def ir_variants(source):
    return {
        "original": source,
        "other_unit": source.replace("@unit_a", "@unit_b"),
        "other_spec": source.replace("slot_size = 1024", "slot_size = 2048"),
        "other_core": source.replace("#pto.kernel_kind<vector>", "#pto.kernel_kind<cube>"),
        "other_payload": source.replace("dense<[1, 2]>", "dense<[3, 4]>"),
        "location": source.replace("return\n", 'return loc("a/different/path.py":123:4)\n'),
        "argument_location": source.replace("%local: i32)", '%local: i32 loc("another/path.py":7:8))'),
    }


def check_ir(directory):
    names = {variant: resolve(directory, variant) for variant in ir_variants("")}
    repeat = (directory / "repeat.resolved").read_text(encoding="utf-8")
    original = (directory / "original.resolved").read_text(encoding="utf-8")
    same_identity = [names[key] for key in ("original", "location", "argument_location")]
    if repeat != original or len(set(same_identity)) != 1:
        raise RuntimeError("wrapper names must be deterministic and independent of debug locations")
    independent = [names[key] for key in ("original", "other_unit", "other_spec", "other_core", "other_payload")]
    if len(set(independent)) != len(independent):
        raise RuntimeError(f"independent units, cores and specializations must not collide: {names}")
    write_link_inputs(directory, [names["original"], names["other_unit"]])


def main():
    action, fixture, output = sys.argv[1:]
    directory = Path(output + ".namespace")
    if action.startswith("check-linked-"):
        is_ptodsl = action == "check-linked-ptodsl"
        variants = ("vector", "cube") if is_ptodsl else ("original", "other_unit")
        check_linked(directory, variants, creates=1 if is_ptodsl else 2)
    elif action == "check-ptodsl":
        names = [resolve(directory, core, creates=1) for core in ("vector", "cube")]
        if names[0] == names[1]:
            raise RuntimeError("independent PTODSL children export the same wrapper symbol")
        write_link_inputs(directory, names)
    else:
        source = Path(fixture).read_text(encoding="utf-8")
        if action == "split-ptodsl":
            split_ptodsl(source, directory)
        elif action == "prepare":
            directory.mkdir(parents=True, exist_ok=True)
            for variant, text in ir_variants(source).items():
                (directory / (variant + ".pto")).write_text(text, encoding="utf-8")
        elif action == "check-ir":
            check_ir(directory)
        else:
            raise ValueError(f"unsupported action: {action}")


if __name__ == "__main__":
    main()
