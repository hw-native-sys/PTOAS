# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Conservative four-stage, static-loop graph and buffer ownership extraction."""
from __future__ import annotations

from ptoas.mlir import ir
from ptoas._cv_common import SCHEMA, fingerprint, integer, require
from ptoas._cv_ir import ALIASES, IRIndex, attr, constant, owner_operation, space_name, tile_info, walk


def function_kind(func):
    kind = attr(func, "pto.kernel_kind")
    require(kind in ("#pto.kernel_kind<cube>", "#pto.kernel_kind<vector>"),
            "UNSUPPORTED", "exactly one cube and one vector function are required")
    return "cube" if kind.endswith("<cube>") else "vector"


def loop_record(loop, scope, index):
    require(len(loop.operands) == 3 and len(loop.results) == 0,
            "UNSUPPORTED", "loop-carried SSA values are outside the v1 subset")
    lower, upper, step = [constant(value) for value in loop.operands]
    integer(step, "loop step", 1)
    count = max(0, (upper - lower + step - 1) // step)
    require(len(loop.regions[0].blocks) == 1, "UNSUPPORTED", "expected single-block loop")
    return dict(id=f"{scope}.loop", op_id=index.ids[loop], axis="iteration",
                induction_value_id=index.values[loop.regions[0].blocks[0].arguments[0]],
                lower=lower, upper=upper, step=step, trip_count=count)


class Graph:
    """One explicitly delimited serial C/V pair, with no silent op fallback."""

    def __init__(self, module, profile, bindings):
        self.index = IRIndex(module)
        self.profile = profile
        self.signature = self.index.signature()
        require(bindings == {}, "UNSUPPORTED", "v1 has no runtime scalar bindings")
        self.buffers = []
        self.roots = {}
        self.buffer_ops = {}
        self.loop_ops = []
        self.loops = []
        self.pipes = {}
        self.handles = {}
        self.functions = {}
        self.transactions = []
        self.tasks = []
        self._functions()
        self._storage()
        self._pipes()
        self._transactions()
        self._tasks()
        self._verify_borrowed_lifetimes()
        self.manifest = dict(schema_version=SCHEMA, pipeline_id="cv0", arch="a5",
                             input_fingerprint=fingerprint(self.signature),
                             target_profile_fingerprint=fingerprint(profile),
                             bindings_fingerprint=fingerprint(bindings),
                             loops=self.loops, buffers=self.buffers,
                             pipes=list(self.pipes.values()), tasks=self.tasks,
                             transactions=self.transactions, operations=self.signature,
                             memory_accesses=self._accesses(),
                             schedule_kind="prefix_suffix_v1", preload_semantics="iteration_distance",
                             local_schedule="off", validation_scope="annotation_only", gm_aliasing="not_proven")

    def _functions(self):
        funcs = [op for op in self.index.operations if op.name == "func.func"]
        require(len(funcs) == 2, "UNSUPPORTED", "v1 accepts exactly two kernel functions")
        for func in funcs:
            kind = function_kind(func)
            require(kind not in self.functions, "UNSUPPORTED", "duplicate kernel kind")
            scope = attr(func, "sym_name")
            require(len(func.regions) == 1 and len(func.regions[0].blocks) == 1,
                    "UNSUPPORTED", "expected defined single-block function")
            args = func.regions[0].blocks[0].arguments
            require(all(str(v.type).startswith("!pto.ptr<") for v in args),
                    "UNSUPPORTED", "v1 kernel arguments must be GM pointers")
            loops = [op for op in walk(func) if op.name == "scf.for"]
            require(len(loops) == 1, "UNSUPPORTED", "one static loop per kernel is required")
            require(loops[0].parent == func, "UNSUPPORTED", "nested loops are unsupported")
            self.loops.append(loop_record(loops[0], scope, self.index))
            self.loop_ops.append(loops[0])
            self.functions[kind] = (func, loops[0], scope)
        require(all(self.loops[0][key] == self.loops[1][key]
                    for key in ("lower", "upper", "step", "trip_count")),
                "LEGALITY", "C/V loop domains differ")

    def _add_tile(self, op, scope):
        info = tile_info(op.results[0].type, self.profile)
        owner = "local" if op.name == "pto.alloc_tile" else "borrowed_entry"
        require(not op.operands, "UNSUPPORTED", "v1 allocations/declarations must have no dynamic operands")
        key = f"{scope}.{self.index.ids[op]}"
        self.buffers.append(dict(id=key, op_id=self.index.ids[op], source_scope=scope,
                                 debug_value=self.index.values[op.results[0]], owner=owner,
                                 multi_buffer_eligible=owner == "local", **info))
        self.roots[op.results[0]] = key
        self.buffer_ops[key] = op

    def _storage(self):
        for func, _, scope in self.functions.values():
            for op in walk(func):
                if op.name in ("pto.alloc_tile", "pto.declare_tile"):
                    self._add_tile(op, scope)
                elif op.name in ALIASES:
                    require(op.operands[0] in self.roots, "UNSUPPORTED", "unresolved tile alias")
                    tile_info(op.results[0].type, self.profile)
                    for offset in op.operands[1:]:
                        constant(offset)
                    self.roots[op.results[0]] = self.roots[op.operands[0]]
                elif op.name == "pto.reserve_buffer":
                    size = integer(attr(op, "size"), "reserved bytes", 1)
                    key = f"{scope}.reserve.{attr(op, 'name')}"
                    require(key not in self.buffer_ops, "LEGALITY", "duplicate reserved buffer")
                    self.buffer_ops[key] = op
                    self.buffers.append(dict(id=key, op_id=self.index.ids[op], source_scope=scope,
                                             owner="pipe_backing", allocation_bytes=size,
                                             memory_space=space_name(op.attributes["location"]),
                                             multi_buffer_eligible=False))
        for value, key in self.roots.items():
            if owner_operation(value).name in ALIASES:
                row = next(b for b in self.buffers if b["id"] == key)
                row.setdefault("aliases", []).append(self.index.values[value])

    def _backing(self, value, scope):
        owner = owner_operation(value)
        require(isinstance(owner, ir.Operation), "UNSUPPORTED", "pipe backing must be reserved")
        if owner.name == "pto.import_reserved_buffer":
            peer = ir.FlatSymbolRefAttr(owner.attributes["peer_func"]).value
            scope = peer
        else:
            require(owner.name == "pto.reserve_buffer", "UNSUPPORTED", "unsupported pipe backing")
        key = f"{scope}.reserve.{attr(owner, 'name')}"
        require(key in self.buffer_ops, "LEGALITY", "unknown peer reserved buffer")
        return key

    def _pipes(self):
        for kind, (func, _, scope) in self.functions.items():
            for op in walk(func):
                if op.name != "pto.initialize_l2l_pipe":
                    continue
                require(len(op.operands) == 1, "UNSUPPORTED", "v1 requires unidirectional L2L pipes")
                backing = self._backing(op.operands[0], scope)
                direction = attr(op, "dir_mask")
                require(direction in (1, 2), "UNSUPPORTED", "v1 uses separate directional pipes")
                size = integer(attr(op, "slot_size"), "slot_size", 1)
                count = integer(attr(op, "slot_num"), "slot_num", 1)
                row = dict(id=backing + ".pipe", backing_buffer_id=backing,
                           direction="c2v" if direction == 1 else "v2c", entry_kind="tile",
                           slot_size_bytes=size, effective_slot_num=count, endpoints=[])
                if backing in self.pipes:
                    old = self.pipes[backing]
                    require(all(old[k] == row[k] for k in row if k != "endpoints"),
                            "LEGALITY", "pipe endpoint contracts differ")
                else:
                    self.pipes[backing] = row
                self.pipes[backing]["endpoints"].append(dict(core=kind, op_id=self.index.ids[op]))
                self.handles[op.results[0]] = self.pipes[backing]
        require(len(self.pipes) == 3, "UNSUPPORTED", "four stages require three logical pipes")
        for pipe in self.pipes.values():
            require({e["core"] for e in pipe["endpoints"]} == {"cube", "vector"}
                    and len(pipe["endpoints"]) == 2, "LEGALITY", "pipe must connect both kernels")

    def _transactions(self):
        for kind, (func, loop, _) in self.functions.items():
            for op in walk(func):
                if op.name not in ("pto.tpush", "pto.tpop", "pto.tfree"):
                    continue
                require(op.parent == loop, "UNSUPPORTED", "transactions must be in the serial loop")
                handle = op.operands[0] if op.name == "pto.tfree" else op.operands[1]
                require(handle in self.handles, "LEGALITY", "unresolved pipe handle")
                pipe = self.handles[handle]
                split = attr(op, "split")
                require(split in (0, 1), "UNSUPPORTED", "v1 supports no-split and row-split")
                producer = "cube" if pipe["direction"] == "c2v" else "vector"
                expected = producer if op.name == "pto.tpush" else ("vector" if producer == "cube" else "cube")
                require(kind == expected, "LEGALITY", "transaction on wrong core")
                row = dict(op_id=self.index.ids[op], pipe_id=pipe["id"], core=kind,
                           action=op.name[4:], split=split)
                if op.name != "pto.tfree":
                    require(op.operands[0] in self.roots, "UNSUPPORTED", "unknown transaction tile")
                    row["buffer_id"] = self.roots[op.operands[0]]
                    row["transfer_bytes_per_core"] = tile_info(op.operands[0].type, self.profile)["allocation_bytes"]
                self.transactions.append(row)
        for pipe in self.pipes.values():
            tx = [t for t in self.transactions if t["pipe_id"] == pipe["id"]]
            require(sorted(t["action"] for t in tx) == ["tfree", "tpop", "tpush"],
                    "LEGALITY", "one push/pop/free per pipe per iteration is required")
            require(len({t["split"] for t in tx}) == 1, "LEGALITY", "transaction split mismatch")
            pipe["split"] = tx[0]["split"]
            pipe["core_expanded_edges"] = 2 if tx[0]["split"] == 1 else 1
            pop = next(t for t in tx if t["action"] == "tpop")
            row = next(b for b in self.buffers if b["id"] == pop["buffer_id"])
            require(row["owner"] == "borrowed_entry", "LEGALITY", "pop must borrow a declared tile")
            row["backing_buffer_id"] = pipe["backing_buffer_id"]
            backing = next(b for b in self.buffers if b["id"] == pipe["backing_buffer_id"])
            require(backing["allocation_bytes"] >= pipe["slot_size_bytes"] * pipe["effective_slot_num"],
                    "CAPACITY", "reserved backing is smaller than the pipe contract")
            for event in tx:
                if event["action"] == "tfree":
                    continue
                factor = 2 if event["core"] == "vector" and event["split"] == 1 else 1
                require(event["transfer_bytes_per_core"] * factor == pipe["slot_size_bytes"],
                        "LEGALITY", "tile transfer size differs from pipe slot size")

    def _tasks(self):
        sequences = {}
        for kind, (_, loop, scope) in self.functions.items():
            body = [op.operation for op in loop.regions[0].blocks[0].operations]
            tx = [t for op in body for t in self.transactions if t["op_id"] == self.index.ids[op]]
            patterns = {"cube": ["tpush", "tpop", "tfree", "tpush"],
                        "vector": ["tpop", "tfree", "tpush", "tpop", "tfree"]}
            require([t["action"] for t in tx] == patterns[kind],
                    "UNSUPPORTED", "expected serial four-stage transaction order")
            sequences[kind] = tx
            cut = next(i for i, op in enumerate(body) if op.name == "pto.tpush") + 1
            for stage, ops in zip(("C_QK", "C_PV") if kind == "cube" else ("V_P", "V_O"),
                                  (body[:cut], body[cut:])):
                self.tasks.append(dict(id=stage, core=kind, loop_id=f"{scope}.loop",
                                       operation_ids=[self.index.ids[op] for op in ops if op.name != "scf.yield"]))
        for pipe in self.pipes.values():
            tx = [t for t in self.transactions if t["pipe_id"] == pipe["id"]]
            for action, field in (("tpush", "producer_task_id"), ("tpop", "consumer_task_id")):
                event = next(t for t in tx if t["action"] == action)
                pipe[field] = next(t["id"] for t in self.tasks if event["op_id"] in t["operation_ids"])
            pipe["iteration_distance"] = 0
        cube, vector = sequences["cube"], sequences["vector"]
        require(cube[0]["pipe_id"] == vector[0]["pipe_id"]
                and cube[1]["pipe_id"] == vector[2]["pipe_id"]
                and cube[3]["pipe_id"] == vector[3]["pipe_id"],
                "LEGALITY", "pipe edges do not form QK -> P -> PV -> O")

    def _verify_borrowed_lifetimes(self):
        positions = {self.index.ids[op]: i for i, op in enumerate(self.index.operations)}
        accesses = self._accesses()
        for buffer in self.buffers:
            if buffer["owner"] != "borrowed_entry":
                continue
            pops = [t for t in self.transactions if t.get("buffer_id") == buffer["id"] and t["action"] == "tpop"]
            require(len(pops) == 1, "LEGALITY", "borrowed entry must have exactly one pop")
            pop = pops[0]
            free = next(t for t in self.transactions if t["pipe_id"] == pop["pipe_id"] and t["action"] == "tfree")
            start, end = positions[pop["op_id"]], positions[free["op_id"]]
            require(start < end, "LEGALITY", "free precedes pop")
            for use in accesses:
                if use["buffer_id"] == buffer["id"]:
                    require(start <= positions[use["op_id"]] < end, "LEGALITY", "borrowed alias used outside pop/free")

    def _accesses(self):
        accesses = []
        for op in self.index.operations:
            tiles = [(i, self.roots[v]) for i, v in enumerate(op.operands) if v in self.roots]
            if not tiles or op.name in ALIASES:
                continue
            for pos, root in tiles:
                mode = "read"
                if op.name == "pto.tpop" or (op.name not in ("pto.tpush", "pto.tstore")
                                             and pos == len(op.operands) - 1):
                    mode = "read_write" if op.name == "pto.tmatmul.acc" else "write"
                accesses.append(dict(op_id=self.index.ids[op], buffer_id=root, operand=pos, access=mode))
        return accesses
