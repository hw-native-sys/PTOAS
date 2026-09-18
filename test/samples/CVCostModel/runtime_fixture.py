# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Iteration-distinct C/V fixtures and independent exact-integer golden."""
from pathlib import Path
import re

from pto_costmodel.wire import integer, require


def source_text(count, crossing=False):
    integer(count, "iterations")
    require(count <= 8, "RANGE", "runtime fixture accepts at most eight iterations")
    source = Path(__file__).with_name("four_stage_serial.pto").read_text(encoding="utf-8")
    # A5 ND->NZ rounds its row stride to 16. A split 16x16 matrix
    # would produce an 8-row NZ tile whose declared storage is too small.
    source = source.replace("16", "32").replace("8", "16")
    capacity = max(4, count)
    source = source.replace("%c2 = arith.constant 2 : index", f"%c2 = arith.constant {count} : index")
    source = source.replace("%c32 = arith.constant 32 : index",
                            f"%c32 = arith.constant 32 : index\n    %rows = arith.constant {max(32, count*32)} : index")
    source = source.replace("shape=[%c32,%c32]", "shape=[%rows,%c32]")
    source = source.replace("size=4096", f"size={4096*capacity}")
    source = source.replace("slot_num=4", f"slot_num={capacity}")
    source = source.replace("slot_size=1024", "slot_size=4096")
    lines, cube_views, vector_views = [], [], []
    for line in source.splitlines():
        if "pto.partition_view" in line:
            target = vector_views if "%ot =" in line else cube_views
            target.append(line.replace("offsets=[%c0,%c0]", "offsets=[%iterrow,%c0]")
                          .replace("offsets=[%row,%c0]", "offsets=[%outrow,%c0]"))
        else:
            lines.append(line)
    result, loop = [], 0
    for line in lines:
        result.append(line)
        if "scf.for %i" in line:
            result.append("      %iterrow = arith.muli %i, %c32 : index")
            if loop:
                result.append("      %outrow = arith.addi %iterrow, %row : index")
            result.extend(vector_views if loop else cube_views)
            loop += 1
    require(loop == 2, "FIXTURE", "expected the fixed C/V loop pair")
    text = "\n".join(result) + "\n"
    return text.replace("pto.tadd ins(%pv,%pv", "pto.tadd ins(%pv,%a") if crossing else text


def multibuffer_stress_source_text(repetitions):
    """Return the fixed crossing/N=8 fixture with an odd alternating tneg chain.

    The last operation always writes ``%a``, so the observable calculation stays
    P=-QK while the vector prefix is long enough to expose C/V overlap.
    """
    integer(repetitions, "repetitions")
    require(repetitions in (129, 257, 513) and repetitions % 2 == 1,
            "FIXTURE", "stress repetitions must be one of the frozen odd choices")
    text = source_text(8, crossing=True)
    vector = text.index("func.func @vector")
    allocation = re.search(r"(?m)^(\s*)%a = pto\.alloc_tile([^\n]+)$", text[vector:])
    require(allocation is not None, "FIXTURE", "vector P allocation not found")
    end = vector + allocation.end()
    indent, suffix = allocation.group(1), allocation.group(2)
    text = text[:end] + "\n" + indent + "%d = pto.alloc_tile" + suffix + text[end:]

    vector = text.index("func.func @vector")
    operation = re.search(
        r"(?m)^(\s*)pto\.tneg ins\(%qk : ([^)]+)\) outs\(%a : ([^)]+)\)$", text[vector:])
    require(operation is not None and operation.group(2) == operation.group(3),
            "FIXTURE", "vector P tneg not found")
    source_type = operation.group(2)
    lines = []
    for index in range(repetitions):
        source = "%qk" if index == 0 else ("%a" if index % 2 else "%d")
        target = "%a" if index % 2 == 0 else "%d"
        lines.append(f"{operation.group(1)}pto.tneg ins({source} : {source_type}) "
                     f"outs({target} : {source_type})")
    start = vector + operation.start()
    end = vector + operation.end()
    return text[:start] + "\n".join(lines) + text[end:]


def input_arrays(count, seed, crossing=False):
    import numpy as np
    require(count in (1, 2, 4, 8) and seed in (0, 1, 2), "FIXTURE", "unsupported runtime input")
    rng = np.random.default_rng(seed)
    q, k, v = [rng.integers(-2, 3, size=(count, 32, 32)).astype(np.float32) for _ in range(3)]
    prefix = -(q.astype(np.int64) @ k.astype(np.int64))
    pv = prefix @ v.astype(np.int64)
    golden = (pv + prefix if crossing else 2 * pv).astype(np.float32)
    require(np.all(np.abs(golden) < 2**24), "FIXTURE", "golden exceeds exact FP32 integer range")
    require(all(not np.array_equal(golden[i], golden[j]) for i in range(count) for j in range(i)),
            "FIXTURE", "iteration outputs must differ")
    return dict(q=q, k=k, v=v, golden=golden)
