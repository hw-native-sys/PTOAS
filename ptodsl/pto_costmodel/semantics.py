# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Bounded scalar expressions and GM access validation for the static subset."""
from pto_costmodel.wire import ContractError, require


class StaticValues:
    def __init__(self, program, iteration=0, lane=0):
        self.program = program
        self.definitions = {v: op for op in program["operations"] for v in op["results"]}
        self.known = {loop["induction_value_id"]: loop["lower"] + iteration * loop["step"]
                      for loop in program["loops"]}
        self.lane = lane

    def scalar(self, value, depth=0):
        require(depth < 64, "UNSUPPORTED", "scalar expression exceeds depth limit")
        if value in self.known:
            return self.known[value]
        require(value in self.definitions, "UNSUPPORTED", f"unbound scalar: {value}")
        op = self.definitions[value]
        name = op["name"]
        if name == "arith.constant":
            result = op["attributes"]["value"]
        elif name == "pto.get_subblock_idx":
            result = self.lane
        else:
            args = [self.scalar(v, depth + 1) for v in op["operands"]]
            result = self._arithmetic(name, args)
        self.known[value] = result
        return result

    @staticmethod
    def _arithmetic(name, args):
        if name == "arith.index_cast":
            return args[0]
        if name == "arith.addi":
            return args[0] + args[1]
        if name == "arith.subi":
            return args[0] - args[1]
        if name == "arith.muli":
            return args[0] * args[1]
        if name == "arith.remui":
            require(args[0] >= 0 and args[1] > 0, "UNSUPPORTED", "unsupported unsigned remainder")
            return args[0] % args[1]
        raise ContractError("UNSUPPORTED", f"unmodeled scalar operation: {name}")

    def region(self, value):
        op = self.definitions[value]
        require(op["name"] == "pto.partition_view", "UNSUPPORTED", "expected partition view")
        view = self.definitions[op["operands"][0]]
        require(view["name"] == "pto.make_tensor_view", "UNSUPPORTED", "nested tensor views unsupported")
        base = view["operands"][0]
        offsets = [self.scalar(v) for v in op["operands"][1:3]]
        sizes = [self.scalar(v) for v in op["operands"][3:5]]
        shape = [self.scalar(v) for v in view["operands"][1:3]]
        strides = [self.scalar(v) for v in view["operands"][3:5]]
        return dict(argument_id=base, offsets=offsets, sizes=sizes, shape=shape, strides=strides)


def verify_global_accesses(package):
    program = package["program"]
    task_ops = {op: task for task in program["tasks"] for op in task["operation_ids"]}
    require(all(a["op_id"] in task_ops for a in program["memory_accesses"]), "UNSUPPORTED",
            "tile accesses outside the stage loops require extended liveness")
    operations = [op for op in program["operations"] if op["name"] in ("pto.tload", "pto.tstore")]
    reads, writes, regions = set(), set(), []
    count = program["loops"][0]["trip_count"]
    require(len(operations) * count * 2 <= 4096, "RANGE", "GM access verification exceeds 4096 instances")
    for iteration in range(count):
        for op in operations:
            require(op["id"] in task_ops, "UNSUPPORTED", "GM accesses outside stage loops")
            lanes = [0] if task_ops[op["id"]]["core"] == "cube" else [0, 1]
            for lane in lanes:
                value = op["operands"][0] if op["name"] == "pto.tload" else op["operands"][1]
                region = StaticValues(program, iteration, lane).region(value)
                _check_region(region, package["bindings"])
                arg = region["argument_id"]
                (reads if op["name"] == "pto.tload" else writes).add(arg)
                if op["name"] == "pto.tstore":
                    _check_writer_overlap(regions, region, task_ops[op["id"]]["core"], lane)
                    regions.append((region, task_ops[op["id"]]["core"], lane))
    require(not reads.intersection(writes), "UNSUPPORTED", "GM read/write recurrence requires dependency analysis")
    return dict(status="checked", argument_aliasing=package["bindings"]["alias_contract"],
                bounds="static_checked", cross_core_writes="disjoint")


def _check_region(region, bindings):
    actual = bindings["arguments"][region["argument_id"]]
    require(region["shape"] == actual["shape"] and region["strides"] == actual["strides"],
            "BINDINGS", "tensor view and argument shape/strides differ")
    require(region["strides"][1] == 1 and region["strides"][0] >= region["shape"][1],
            "UNSUPPORTED", "only non-overlapping row-strided GM views supported")
    for offset, size, dimension in zip(region["offsets"], region["sizes"], region["shape"]):
        require(0 <= offset and 0 < size and offset + size <= dimension, "BOUNDS", "GM partition out of bounds")


def _check_writer_overlap(previous, region, core, lane):
    for other, other_core, other_lane in previous:
        if other["argument_id"] != region["argument_id"] or (core, lane) == (other_core, other_lane):
            continue
        separated = any(a + n <= b or b + m <= a for a, n, b, m in
                        zip(region["offsets"], region["sizes"], other["offsets"], other["sizes"]))
        require(separated, "ALIAS", "cross-core GM stores may overlap")
