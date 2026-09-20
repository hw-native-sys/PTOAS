# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Bounded completion-order proof for the static A5 tile-entry subset.

L2L C2V pop/free execute on V; V2C pop/free execute on MTE1. Producer
publish executes on FIX/MTE3 respectively (A5 TPipe::Consumer/Producer).
Scalar source order alone never supplies a completion edge.
"""
from collections import Counter, defaultdict, deque
import hashlib

from ptoas._cv_ir import attr
from ptoas.mlir.dialects import pto
from pto_costmodel.wire import require

ENGINES = ("S", "V", "M", "MTE1", "MTE2", "MTE3", "FIX")


class CompletionGraph:
    def __init__(self):
        self.edges = defaultdict(set)
        self.evidence = []

    def edge(self, source, target, reason):
        if source is not None and source != target:
            self.edges[source].add(target)
            self.evidence.append(dict(source=source, target=target, reason=reason))
        self.edges[target]

    def reaches(self, source, target):
        pending, seen = [source], set()
        while pending:
            node = pending.pop()
            if node == target:
                return True
            if node not in seen:
                seen.add(node)
                pending.extend(self.edges[node])
        return False

    def acyclic(self):
        counts = {n: 0 for n in self.edges}
        for children in self.edges.values():
            for child in children:
                counts[child] += 1
        ready = deque(n for n, degree in counts.items() if degree == 0)
        seen = 0
        while ready:
            node = ready.popleft()
            seen += 1
            for child in self.edges[node]:
                counts[child] -= 1
                if counts[child] == 0:
                    ready.append(child)
        return seen == len(counts)


def _pipe(token):
    text = str(token)
    return next((p for p in ("ALL", *ENGINES) if f"PIPE_{p}>" in text), None)


def _engine(op, core):
    fixed = {"pto.tload": "MTE2", "pto.tstore": "MTE3", "pto.tmatmul": "M",
             "pto.tmatmul.acc": "M", "pto.tneg": "V", "pto.tadd": "V"}
    if op.name in fixed:
        return fixed[op.name]
    if op.name == "pto.tpush":
        return "FIX" if core == "cube" else "MTE3"
    if op.name in ("pto.tpop", "pto.tfree"):
        return "MTE1" if core == "cube" else "V"
    if op.name == "pto.tmov":
        if core == "vector":
            return "V"
        if "tile_buf<mat," in str(op.operands[0].type):
            return "MTE1"
    return None


def _events(module, graph, trip_count):
    found, errors = {}, []
    for func in module.body.operations:
        if func.operation.name != "func.func":
            continue
        core = "cube" if "cube" in str(func.attributes["pto.kernel_kind"]) else "vector"
        last, flags = {}, defaultdict(deque)
        for number, (op, iteration) in enumerate(_expanded(func.operation)):
            key = attr(op, "pto.costmodel.event_id")
            if key is not None and key.endswith("@loop"):
                if iteration is None and trip_count == 1:
                    iteration = 0
                key = key[:-4] + str(iteration)
            node = key or f"{core}.native{number}"
            if op.name in ("pto.set_flag", "pto.wait_flag"):
                _flag(op, node, last, flags, graph, errors)
            elif op.name == "pto.barrier":
                engine = _pipe(op.attributes["pipe"])
                if engine is None:
                    errors.append(dict(code="UNKNOWN_BARRIER", event=node))
                    continue
                engines = ENGINES if engine == "ALL" else (engine,)
                for p in engines:
                    graph.edge(last.get(p), node, "native_barrier")
                    last[p] = node
            elif key is not None:
                engine = _engine(op, core)
                if engine is not None:
                    if key in found:
                        errors.append(dict(code="DUPLICATE_EVENT", event=key))
                    found[key] = op
                    graph.edge(last.get(engine), key, "in_order_" + engine)
                    last[engine] = key
            elif op.name.startswith("pto.t"):
                errors.append(dict(code="UNMAPPED_EFFECT", operation=op.name))
            elif op.name in ("pto.set_flag_dyn", "pto.wait_flag_dyn", "scf.for", "scf.while", "scf.if"):
                errors.append(dict(code="UNSUPPORTED_CONTROL", operation=op.name))
    return found, errors


def _expanded(op, iteration=None):
    if op.name == "scf.for":
        require(iteration is None and len(op.operands) == 3, "UNSUPPORTED", "nested/carried final loop")
        bounds = [attr(v.owner, "value") if v.owner.name == "arith.constant" else None for v in op.operands]
        require(all(isinstance(v, int) for v in bounds) and bounds[2] > 0,
                "UNSUPPORTED", "nonconstant final loop domain")
        values = range(*bounds)
        require(len(values) <= 256, "RANGE", "final loop exceeds expansion bound")
        for index in values:
            for child in op.regions[0].blocks[0].operations:
                yield from _expanded(child.operation, (index - bounds[0]) // bounds[2])
        return
    yield op, iteration
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _expanded(child.operation, iteration)


def _flag(op, node, last, flags, graph, errors):
    # Native names are checked explicitly; attribute iteration order is not semantic.
    src, dst = _pipe(op.attributes["src_pipe"]), _pipe(op.attributes["dst_pipe"])
    if src not in ENGINES or dst not in ENGINES:
        errors.append(dict(code="UNKNOWN_FLAG_PIPE", event=node))
        return
    flag = str(op.attributes["event_id"])
    engine = src if op.name == "pto.set_flag" else dst
    graph.edge(last.get(engine), node, "native_flag_order")
    last[engine] = node
    pair = (src, dst, flag)
    if op.name == "pto.set_flag":
        flags[pair].append(node)
    elif flags[pair]:
        graph.edge(flags[pair].popleft(), node, "native_set_wait")
    else:
        errors.append(dict(code="UNMATCHED_WAIT", event=node))


def verify_completion(module, package, candidate, memory):
    graph = CompletionGraph()
    found, errors = _events(module, graph, package["program"]["loops"][0]["trip_count"])
    _layout_preflight(found, errors)
    _backing_preflight(memory, errors)
    if any(package["program"]["loops"][0]["trip_count"] > p["effective_slot_num"]
           for p in package["program"]["pipes"]):
        errors.append(dict(code="UNKNOWN_BATCHED_FIFO_REUSE"))
    events = {e["id"]: e for e in candidate["schedule"]["events"] if e["reads"] or e["writes"]}
    require(len(events) <= 2048, "RANGE", "completion feedback exceeds 2048 memory events")
    for key in events.keys() - found.keys():
        errors.append(dict(code="MISSING_MEMORY_EVENT", event=key))
    for edge in candidate["schedule"]["dependencies"]:
        if edge["kind"] in ("pipe", "fifo_capacity"):
            graph.edge(edge["source"], edge["target"], "A5_L2L_" + edge["kind"])
    _operand_mapping(package, events, found, errors)
    _transaction_mapping(package, found, errors)
    proofs = []
    for version in candidate["schedule"]["buffer_versions"]:
        for reader in version["readers"]:
            _obligation(graph, version["writer"], reader, "RAW", proofs, errors)
    _release_obligations(package, candidate, graph, proofs, errors)
    _storage_hazards(events, memory, found, graph, proofs, errors)
    if not graph.acyclic():
        errors.append(dict(code="COMPLETION_CYCLE"))
    invalid = {"VERSION_MAPPING", "TRANSACTION_MAPPING", "LIVE_STORAGE_OVERLAP",
               "COMPLETION_CYCLE", "NATIVE_LAYOUT_OVERFLOW"}
    status = "fail" if any(e["code"] in invalid for e in errors) else "unknown" if errors else "pass"
    compact, proof_summary = _compact_proofs(proofs)
    return dict(status=status, semantic_profile="a5_l2l_tile_entry_v1",
                obligations=compact, obligation_summary=proof_summary,
                edges=graph.evidence, unresolved=errors,
                limitations=["only mapped FP32 static tile operations; target headers must match the declared profile"])


def _compact_proofs(proofs, limit=4096):
    """Bound diagnostic output without weakening the checks that produced it."""
    digest = hashlib.sha256()
    for proof in proofs:
        digest.update(repr((proof["source"], proof["target"], proof["kind"])).encode("utf-8"))
        digest.update(b"\n")
    summary = dict(total=len(proofs), retained=min(len(proofs), limit),
                   by_kind=dict(sorted(Counter(p["kind"] for p in proofs).items())),
                   sha256=digest.hexdigest(), truncated=len(proofs) > limit)
    return proofs[:limit], summary


def _layout_preflight(found, errors):
    for key, op in found.items():
        tiles = [pto.TileBufType(v.type) for v in op.operands if pto.TileBufType.isinstance(v.type)]
        if any(str(t.element_type) != "f32" for t in tiles):
            errors.append(dict(code="UNKNOWN_NATIVE_DTYPE", event=key))
        if op.name != "pto.tmov" or len(tiles) != 2:
            continue
        src, dst = tiles
        if ("vec" in str(src.memory_space) and "vec" in str(dst.memory_space)
                and src.slayout_value == 0 and dst.slayout_value == 1):
            # Installed A5 TMovToVecNd2Nz uses alignRow=ceil(validRow/16)*16.
            rows = ((dst.valid_shape[0] + 15) // 16) * 16
            if dst.shape[0] < rows:
                errors.append(dict(code="NATIVE_LAYOUT_OVERFLOW", event=key,
                                   declared_rows=dst.shape[0], required_rows=rows))


def _backing_preflight(memory, errors):
    rows = [r for r in memory["allocations"] if r["offset_bytes"] is not None]
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if a["owner"] != "pipe_backing" and b["owner"] != "pipe_backing":
                continue
            same_core = bool(set(a["physical_instances"]) & set(b["physical_instances"]))
            if a["memory_space"] != b["memory_space"] or not same_core:
                continue
            if max(a["offset_bytes"], b["offset_bytes"]) < min(a["offset_bytes"] + a["reserved_bytes"],
                                                             b["offset_bytes"] + b["reserved_bytes"]):
                errors.append(dict(code="UNKNOWN_BACKING_REUSE",
                                   allocations=[a["allocation_id"], b["allocation_id"]]))


def _obligation(graph, source, target, kind, proofs, errors):
    result = dict(source=source, target=target, kind=kind)
    if graph.reaches(source, target):
        proofs.append(result)
    else:
        errors.append(dict(code="UNPROVEN_COMPLETION", **result))


def _transaction_mapping(package, found, errors):
    transactions = {t["op_id"]: t for t in package["program"]["transactions"]}
    for key, op in found.items():
        transaction = transactions.get(key.split("@")[0])
        if transaction is None:
            continue
        owner = op.operands[-1].owner
        if (op.name != "pto." + transaction["action"] or attr(op, "split") != transaction["split"]
                or attr(owner, "pto.costmodel.pipe_id") != transaction["pipe_id"]):
            errors.append(dict(code="TRANSACTION_MAPPING", event=key))


def _release_obligations(package, candidate, graph, proofs, errors):
    transactions = package["program"]["transactions"]
    for pipe in package["program"]["pipes"]:
        pop = next(t for t in transactions if t["pipe_id"] == pipe["id"] and t["action"] == "tpop")
        free = next(t for t in transactions if t["pipe_id"] == pipe["id"] and t["action"] == "tfree")
        for version in candidate["schedule"]["buffer_versions"]:
            if version["buffer_id"] != pop["buffer_id"]:
                continue
            iteration = version["writer"].split("@")[-1]
            for reader in version["readers"]:
                _obligation(graph, reader, f"{free['op_id']}@{iteration}", "borrowed_release", proofs, errors)


def _storage_hazards(events, memory, found, graph, proofs, errors):
    rows = {(r["source_buffer_id"], r["slot"]): r for r in memory["allocations"]}
    uses = defaultdict(list)
    for key, event in events.items():
        for mode, entries in (("read", event["reads"]), ("write", event["writes"])):
            for entry in entries:
                row = rows.get((entry["buffer_id"], entry["slot"]))
                if row is not None and row["offset_bytes"] is not None:
                    for core in row["physical_instances"]:
                        uses[(core, row["memory_space"])].append((key, mode, row, entry["version"]))
    positions = {key: n for n, key in enumerate(found)}
    for accesses in uses.values():
        _live_overlap(accesses, positions, errors)
        for index, (left, mode, a, _) in enumerate(accesses):
            for right, other_mode, b, _ in accesses[index + 1:]:
                overlap = max(a["offset_bytes"], b["offset_bytes"]) < min(
                    a["offset_bytes"] + a["payload_bytes"], b["offset_bytes"] + b["payload_bytes"])
                if left == right or not overlap or mode == other_mode == "read":
                    continue
                if left not in positions or right not in positions:
                    continue
                source, target = sorted((left, right), key=positions.get)
                _obligation(graph, source, target, "physical_overlap", proofs, errors)


def _live_overlap(accesses, positions, errors):
    writers = {version: key for key, mode, _, version in accesses if mode == "write"}
    for reader, mode, a, version in accesses:
        if mode != "read" or reader not in positions or writers.get(version) not in positions:
            continue
        start, stop = positions[writers[version]], positions[reader]
        for writer, other_mode, b, other_version in accesses:
            if other_mode != "write" or other_version == version or writer not in positions:
                continue
            overlaps = max(a["offset_bytes"], b["offset_bytes"]) < min(
                a["offset_bytes"] + a["payload_bytes"], b["offset_bytes"] + b["payload_bytes"])
            if overlaps and start < positions[writer] < stop:
                errors.append(dict(code="LIVE_STORAGE_OVERLAP", version=version, overwrite=writer, reader=reader))


def _operand_mapping(package, events, found, errors):
    accesses = defaultdict(list)
    for access in package["program"]["memory_accesses"]:
        accesses[access["op_id"]].append(access)
    borrowed = {}
    for key, op in found.items():
        if op.name == "pto.tpop" and key in events:
            borrowed[(op.operands[0], events[key]["iteration"])] = events[key]["writes"][0]["version"]
    for key, event in events.items():
        if key not in found:
            continue
        op = found[key]
        for access in accesses[event["op_id"]]:
            root = access["buffer_id"]
            mode = "reads" if access["access"] == "read" else "writes"
            expected = next(e for e in event[mode] if e["buffer_id"] == root)
            operand = op.operands[access["operand"]]
            if expected["slot"] is None:
                matches = borrowed.get((operand, event["iteration"])) == expected["version"]
            else:
                owner = operand.owner
                matches = (attr(owner, "pto.costmodel.buffer_id") == root
                           and attr(owner, "pto.costmodel.slot", 0) == expected["slot"])
            if not matches:
                errors.append(dict(code="VERSION_MAPPING", event=key, buffer_id=root))
