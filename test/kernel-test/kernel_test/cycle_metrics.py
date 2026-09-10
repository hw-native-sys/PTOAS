# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared cannsim cycle-metric parsing helpers for kernel-test."""

from __future__ import annotations

import glob
import csv
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

INSTR_LOG_RE = re.compile(
    r"start:\s*(\d+),\s*tick:\s*(\d+).*?blkDim:\s*(\d+)",
    re.DOTALL,
)
SOC_CYCLE_RE = re.compile(
    r"\[Hardware\]\s+parallel simulation finish\.\s+sim time:\s*"
    r"(?:SoC sub \d+ )?([\d.]+)s,\s*cycle:\s*(\d+)",
)


@dataclass(frozen=True)
class LaunchRecord:
    start: int
    tick: int
    blk_dim: int

    @property
    def duration(self) -> int:
        return self.tick - self.start


@dataclass(frozen=True)
class InstrLogMetrics:
    launches: tuple[LaunchRecord, ...]
    span: int
    max_dur: int
    max_blk_dim: int

    @classmethod
    def empty(cls) -> InstrLogMetrics:
        return cls(launches=(), span=0, max_dur=0, max_blk_dim=0)


@dataclass(frozen=True)
class TraceMetrics:
    core_vf_span: int | None = None
    rvec_span: int | None = None
    pushq_vf_dur: int | None = None
    mte2_span: int | None = None
    mte3_span: int | None = None
    vector_span: int | None = None
    arith_sum_dur: int | None = None
    rvec_op_counts: dict[str, int] = field(default_factory=dict)
    rvec_event_count: int = 0
    trace_path: str | None = None

    @property
    def vf_cycles(self) -> int | None:
        if self.core_vf_span is not None and self.core_vf_span > 0:
            return self.core_vf_span
        if self.rvec_span is not None and self.rvec_span > 0:
            return self.rvec_span
        if self.pushq_vf_dur is not None and self.pushq_vf_dur > 0:
            return self.pushq_vf_dur
        return None


@dataclass(frozen=True)
class MsprofMetrics:
    opprof_dir: str | None = None
    instr_csv_path: str | None = None
    core_vf_cycles: int | None = None
    arith_cycles: int | None = None
    pipe_cycles: dict[str, int] = field(default_factory=dict)
    instr_cycles: dict[str, int] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> MsprofMetrics:
        return cls()


@dataclass(frozen=True)
class SocCycleRecord:
    sim_wall_s: float
    soc_cycles: int


@dataclass(frozen=True)
class RunMetrics:
    out_dir: str
    cannsim_run_dir: str | None
    msprof: MsprofMetrics
    instr: InstrLogMetrics
    trace: TraceMetrics
    soc_cycles: tuple[SocCycleRecord, ...]
    steady_soc_cycles: int | None
    measured_kernel_cycles: int | None

    @property
    def primary_vf_cycles(self) -> int | None:
        if self.msprof.core_vf_cycles is not None and self.msprof.core_vf_cycles > 0:
            return self.msprof.core_vf_cycles
        if self.trace.vf_cycles is not None:
            return self.trace.vf_cycles
        if self.instr.max_dur > 0:
            return self.instr.max_dur
        return self.measured_kernel_cycles


def find_cannsim_run_dir(out_dir: str) -> str:
    pattern = os.path.join(out_dir, "cannsim_*")
    candidates = [path for path in glob.glob(pattern) if os.path.isdir(path)]
    if not candidates:
        raise FileNotFoundError(f"No cannsim_* directory under {out_dir}")
    return max(candidates, key=os.path.getmtime)


def maybe_find_cannsim_run_dir(out_dir: str) -> str | None:
    pattern = os.path.join(out_dir, "cannsim_*")
    candidates = [path for path in glob.glob(pattern) if os.path.isdir(path)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _parse_instr_log_file(path: str) -> list[LaunchRecord]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    records: list[LaunchRecord] = []
    for match in INSTR_LOG_RE.finditer(text):
        start, tick, blk_dim = (int(match.group(i)) for i in range(1, 4))
        records.append(LaunchRecord(start=start, tick=tick, blk_dim=blk_dim))
    return records


def _group_launches(
    launches: list[LaunchRecord],
    gap_threshold: int = 3000,
) -> list[list[LaunchRecord]]:
    if not launches:
        return []
    sorted_launches = sorted(launches, key=lambda record: record.start)
    groups: list[list[LaunchRecord]] = [[sorted_launches[0]]]
    for record in sorted_launches[1:]:
        prev_end = max(item.tick for item in groups[-1])
        if record.start - prev_end > gap_threshold:
            groups.append([record])
        else:
            groups[-1].append(record)
    return groups


def metrics_from_launches(launches: Iterable[LaunchRecord]) -> InstrLogMetrics:
    items = list(launches)
    if not items:
        return InstrLogMetrics.empty()
    starts = [record.start for record in items]
    ticks = [record.tick for record in items]
    durations = [record.duration for record in items]
    blk_dims = [record.blk_dim for record in items]
    return InstrLogMetrics(
        launches=tuple(items),
        span=max(ticks) - min(starts),
        max_dur=max(durations),
        max_blk_dim=max(blk_dims),
    )


def parse_instr_log_dir(log_ca_dir: str, measured_only: bool = True) -> InstrLogMetrics:
    pattern = os.path.join(log_ca_dir, "core*.veccore*.instr_log.dump")
    paths = sorted(glob.glob(pattern))
    if not paths:
        return InstrLogMetrics.empty()

    all_launches: list[LaunchRecord] = []
    for path in paths:
        all_launches.extend(_parse_instr_log_file(path))

    if not all_launches:
        return InstrLogMetrics.empty()

    if measured_only:
        max_blk = max(record.blk_dim for record in all_launches)
        kernel_launches = [record for record in all_launches if record.blk_dim == max_blk]
        if kernel_launches:
            groups = _group_launches(kernel_launches, gap_threshold=2000)
            all_launches = groups[-1] if groups else kernel_launches

    return metrics_from_launches(all_launches)


def parse_soc_cycles(cannsim_log: str) -> list[SocCycleRecord]:
    with open(cannsim_log, encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    records: list[SocCycleRecord] = []
    for match in SOC_CYCLE_RE.finditer(text):
        sim_wall_s = float(match.group(1))
        soc_cycles = int(match.group(2))
        records.append(SocCycleRecord(sim_wall_s=sim_wall_s, soc_cycles=soc_cycles))
    return records


def steady_state_soc_cycles(records: Iterable[SocCycleRecord]) -> int | None:
    items = list(records)
    if not items:
        return None
    if len(items) == 1:
        return items[0].soc_cycles
    return items[-1].soc_cycles


def find_trace_json(run_dir: str) -> str | None:
    patterns = [
        os.path.join(run_dir, "report", "trace_core0.json"),
        os.path.join(run_dir, "**", "trace_core0.json"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return max(matches, key=os.path.getmtime)
    return None


def find_msprof_run_dir(out_dir: str) -> str | None:
    patterns = [
        os.path.join(out_dir, "msprof", "OPPROF_*"),
        os.path.join(out_dir, "OPPROF_*"),
    ]
    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(path for path in glob.glob(pattern) if os.path.isdir(path))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def find_msprof_instr_csv(opprof_dir: str) -> str | None:
    patterns = [
        os.path.join(opprof_dir, "simulator", "core0.veccore0", "core0.veccore0_instr_exe.csv"),
        os.path.join(opprof_dir, "**", "core*.veccore*.instr_exe.csv"),
        os.path.join(opprof_dir, "**", "*instr_exe*.csv"),
    ]
    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(path for path in glob.glob(pattern, recursive=True) if os.path.isfile(path))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def parse_msprof_metrics(out_dir: str) -> MsprofMetrics:
    opprof_dir = find_msprof_run_dir(out_dir)
    if not opprof_dir:
        return MsprofMetrics.empty()

    instr_csv_path = find_msprof_instr_csv(opprof_dir)
    if not instr_csv_path:
        return MsprofMetrics(opprof_dir=opprof_dir)

    pipe_cycles: dict[str, int] = {}
    instr_cycles: dict[str, int] = {}
    with open(instr_csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            normalized = {
                str(key).strip().lower(): (value or "").strip()
                for key, value in row.items()
                if key is not None
            }
            instr = (
                normalized.get("instr")
                or normalized.get("instruction")
                or normalized.get("instr_name")
                or ""
            )
            pipe = normalized.get("pipe") or normalized.get("pipeline") or ""
            cycle_value = (
                normalized.get("cycles")
                or normalized.get("cycle")
                or normalized.get("duration")
                or "0"
            )
            try:
                cycles = int(float(cycle_value))
            except ValueError:
                continue
            if pipe:
                pipe_cycles[pipe] = pipe_cycles.get(pipe, 0) + cycles
            if instr:
                instr_cycles[instr] = instr_cycles.get(instr, 0) + cycles

    arith_instrs = {
        "RV_VMUL",
        "RV_VADD",
        "RV_VSUB",
        "RV_VDIV",
        "RV_VMAX",
        "RV_VMIN",
        "RV_VMAC",
        "RV_VMADD",
        "RV_VMLS",
    }
    arith_cycles = sum(cycles for instr, cycles in instr_cycles.items() if instr in arith_instrs) or None
    core_vf_cycles = pipe_cycles.get("RVECEX") or None

    return MsprofMetrics(
        opprof_dir=opprof_dir,
        instr_csv_path=instr_csv_path,
        core_vf_cycles=core_vf_cycles,
        arith_cycles=arith_cycles,
        pipe_cycles=pipe_cycles,
        instr_cycles=instr_cycles,
    )


def _load_trace_events(trace_path: str) -> list[dict]:
    with open(trace_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, list):
        return payload
    return payload.get("traceEvents", [])


def _pipe_span(events: list[dict], processes: dict[int, str], *needles: str) -> int | None:
    pids = {pid for pid, name in processes.items() if any(needle in name.upper() for needle in needles)}
    exec_ev = [event for event in events if event.get("ph") == "X" and "dur" in event and event.get("pid") in pids]
    if not exec_ev:
        return None
    start = min(event["ts"] for event in exec_ev)
    end = max(event["ts"] + event["dur"] for event in exec_ev)
    span = int(end - start)
    return span if span > 0 else None


def parse_trace_metrics(trace_path: str) -> TraceMetrics:
    if not os.path.isfile(trace_path):
        return TraceMetrics()

    events = _load_trace_events(trace_path)
    processes = {
        event["pid"]: event["args"]["name"]
        for event in events
        if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    exec_ev = [event for event in events if event.get("ph") == "X" and "dur" in event]

    pushq_pids = {pid for pid, name in processes.items() if "PUSHQ" in name.upper()}
    vf_dispatch = [
        event for event in exec_ev if event.get("pid") in pushq_pids and "VF" in event.get("name", "")
    ]
    pushq_vf_dur = max((event.get("dur", 0) for event in vf_dispatch), default=0) or None

    rvec_pids = {pid for pid, name in processes.items() if "RVEC" in name.upper()}
    rvec_events = [event for event in exec_ev if event.get("pid") in rvec_pids]
    rvecex_pids = {pid for pid, name in processes.items() if "RVECEX" in name.upper()}
    rvecex_events = [event for event in exec_ev if event.get("pid") in rvecex_pids]
    arith_events = [
        event
        for event in rvecex_events
        if event.get("name") in {"RV_VMUL", "RV_VADD", "RV_VSUB", "RV_VDIV", "RV_VMAX", "RV_VMIN"}
    ]
    core_vf_span = None
    arith_sum_dur = None
    if rvecex_events:
        start = min(event["ts"] for event in rvecex_events)
        end = max(event["ts"] + event["dur"] for event in rvecex_events)
        core_vf_span = int(end - start)
    if arith_events:
        arith_sum_dur = int(sum(event.get("dur", 0) for event in arith_events))

    rvec_span = None
    rvec_op_counts: dict[str, int] = {}
    rvec_event_count = 0
    if rvec_events:
        start = min(event["ts"] for event in rvec_events)
        end = max(event["ts"] + event["dur"] for event in rvec_events)
        rvec_span = int(end - start)
        rvec_op_counts = dict(Counter(event.get("name", "?") for event in rvec_events))
        rvec_event_count = len(rvec_events)

    return TraceMetrics(
        core_vf_span=core_vf_span,
        rvec_span=rvec_span,
        pushq_vf_dur=pushq_vf_dur,
        mte2_span=_pipe_span(events, processes, "MTE2"),
        mte3_span=_pipe_span(events, processes, "MTE3"),
        vector_span=_pipe_span(events, processes, "RVEC", "VECTOR", "VEC"),
        arith_sum_dur=arith_sum_dur,
        rvec_op_counts=rvec_op_counts,
        rvec_event_count=rvec_event_count,
        trace_path=trace_path,
    )


def parse_marker_soc_cycles(cannsim_log: str) -> tuple[list[SocCycleRecord], list[SocCycleRecord]]:
    with open(cannsim_log, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    in_window = False
    all_records: list[SocCycleRecord] = []
    window_records: list[SocCycleRecord] = []

    for line in lines:
        if "CYCLE_MARKER" in line:
            in_window = True
            window_records.clear()
            continue
        if "CYCLE_DONE" in line:
            in_window = False
            continue
        match = SOC_CYCLE_RE.search(line)
        if not match:
            continue
        record = SocCycleRecord(sim_wall_s=float(match.group(1)), soc_cycles=int(match.group(2)))
        all_records.append(record)
        if in_window:
            window_records.append(record)

    return all_records, window_records


def measured_kernel_soc_cycles(cannsim_log: str) -> int | None:
    _, window = parse_marker_soc_cycles(cannsim_log)
    if not window:
        return None
    return window[-1].soc_cycles


def parse_run_metrics(out_dir: str) -> RunMetrics:
    msprof = parse_msprof_metrics(out_dir)
    run_dir = maybe_find_cannsim_run_dir(out_dir)

    if run_dir is None and msprof.opprof_dir is None:
        raise FileNotFoundError(
            f"No cycle artifacts found under {out_dir}; expected msprof/OPPROF_* or cannsim_* outputs"
        )

    log_ca = os.path.join(run_dir, "log_ca") if run_dir else ""
    cannsim_log = os.path.join(run_dir, "cannsim.log") if run_dir else ""

    instr = parse_instr_log_dir(log_ca) if log_ca and os.path.isdir(log_ca) else InstrLogMetrics.empty()
    soc = parse_soc_cycles(cannsim_log) if cannsim_log and os.path.isfile(cannsim_log) else []
    measured = (
        measured_kernel_soc_cycles(cannsim_log)
        if cannsim_log and os.path.isfile(cannsim_log)
        else None
    )

    trace_path = find_trace_json(run_dir) if run_dir else None
    trace = parse_trace_metrics(trace_path) if trace_path else TraceMetrics()

    return RunMetrics(
        out_dir=out_dir,
        cannsim_run_dir=run_dir,
        msprof=msprof,
        instr=instr,
        trace=trace,
        soc_cycles=tuple(soc),
        steady_soc_cycles=steady_state_soc_cycles(soc),
        measured_kernel_cycles=measured,
    )


def format_run_summary(metrics: RunMetrics, label: str | None = None) -> str:
    title = label or os.path.basename(metrics.out_dir.rstrip("/"))
    lines = [f"=== {title} ({metrics.out_dir}) ==="]
    if metrics.msprof.opprof_dir:
        lines.append(f"msprof:                    {metrics.msprof.opprof_dir}")
    if metrics.cannsim_run_dir:
        lines.append(f"cannsim run dir:           {metrics.cannsim_run_dir}")

    primary = metrics.primary_vf_cycles
    if primary is not None:
        if metrics.msprof.core_vf_cycles and metrics.msprof.core_vf_cycles > 0:
            source = "msprof RVECEX cycles"
        elif metrics.trace.core_vf_span and metrics.trace.core_vf_span > 0:
            source = "RVECEX span"
        elif metrics.trace.rvec_span and metrics.trace.rvec_span > 0:
            source = "RVEC span"
        elif metrics.trace.pushq_vf_dur and metrics.trace.pushq_vf_dur > 0:
            source = "PUSHQ VF dur"
        elif metrics.instr.max_dur > 0:
            source = "instr MaxDur"
        else:
            source = "SoC (fallback)"
        lines.append(f"primary VF cycles:         {primary}  ({source})")

    if metrics.msprof.core_vf_cycles:
        lines.append(f"msprof core VF cycles:     {metrics.msprof.core_vf_cycles}")
    if metrics.msprof.arith_cycles:
        lines.append(f"msprof arith cycles:       {metrics.msprof.arith_cycles}")
    if metrics.msprof.pipe_cycles:
        ordered_pipes = ["RVECEX", "RVECLD", "RVECST", "RVECSU", "MTE2", "MTE3", "VECTOR"]
        rendered = ", ".join(
            f"{pipe}={metrics.msprof.pipe_cycles[pipe]}"
            for pipe in ordered_pipes
            if pipe in metrics.msprof.pipe_cycles
        )
        if rendered:
            lines.append(f"msprof pipe cycles:        {rendered}")
    if metrics.trace.core_vf_span:
        lines.append(f"core VF span (RVECEX):     {metrics.trace.core_vf_span}")
    if metrics.trace.rvec_span:
        lines.append(f"RVEC span:                 {metrics.trace.rvec_span}")
    if metrics.trace.pushq_vf_dur:
        lines.append(f"PUSHQ VF dur:              {metrics.trace.pushq_vf_dur}")
    if metrics.trace.arith_sum_dur:
        lines.append(f"arith sum dur:             {metrics.trace.arith_sum_dur}")
    if metrics.trace.mte2_span:
        lines.append(f"MTE2 span:                 {metrics.trace.mte2_span}")
    if metrics.trace.mte3_span:
        lines.append(f"MTE3 span:                 {metrics.trace.mte3_span}")
    if metrics.trace.rvec_op_counts:
        top_ops = sorted(metrics.trace.rvec_op_counts.items(), key=lambda item: -item[1])[:6]
        lines.append("top RVEC ops: " + ", ".join(f"{name}={count}" for name, count in top_ops))

    if metrics.instr.max_dur:
        lines.append(
            f"instr MaxDur / span:       {metrics.instr.max_dur} / {metrics.instr.span} "
            f"(blkDim={metrics.instr.max_blk_dim})"
        )
    elif not metrics.cannsim_run_dir or not os.path.isdir(os.path.join(metrics.cannsim_run_dir, "log_ca")):
        lines.append("instr MaxDur / span:       (log_ca unavailable)")

    if metrics.measured_kernel_cycles is not None:
        lines.append(
            f"SoC cycles (measured):     {metrics.measured_kernel_cycles}  "
            "(coarse; ~420 is normal)"
        )
    elif metrics.steady_soc_cycles is not None:
        lines.append(f"SoC cycles (steady):       {metrics.steady_soc_cycles}  (coarse)")

    if metrics.trace.trace_path:
        lines.append(f"trace:                     {metrics.trace.trace_path}")
    elif metrics.cannsim_run_dir and not find_trace_json(metrics.cannsim_run_dir):
        lines.append("trace:                     (none — need instr.bin + cannsim report)")

    return "\n".join(lines)


def format_table(rows: list[tuple[str, RunMetrics]]) -> str:
    header = (
        f"{'case':<22} {'primary':>8} {'RVECEX':>8} {'RVEC':>8} "
        f"{'MaxDur':>8} {'Span':>8} {'SoC':>6}"
    )
    lines = [header, "-" * len(header)]
    for label, metrics in rows:
        primary = metrics.primary_vf_cycles
        rvecex = metrics.msprof.core_vf_cycles or metrics.trace.core_vf_span
        lines.append(
            f"{label:<22} "
            f"{primary if primary is not None else '-':>8} "
            f"{rvecex if rvecex else '-':>8} "
            f"{metrics.trace.rvec_span if metrics.trace.rvec_span else '-':>8} "
            f"{metrics.instr.max_dur if metrics.instr.max_dur else '-':>8} "
            f"{metrics.instr.span if metrics.instr.span else '-':>8} "
            f"{metrics.measured_kernel_cycles if metrics.measured_kernel_cycles else '-':>6}"
        )
    return "\n".join(lines)
