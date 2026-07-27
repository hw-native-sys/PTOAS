"""
Parse cannsim performance artifacts for per_block_cast VF sim.

Primary VF metrics (per ops-simulator guidance):
- RVEC span from trace_core*.json (vector-pipe VF body cycles)
- PUSHQ VF dispatch duration (fallback)
- MaxDur / span from log_ca/*.instr_log.dump (AI-core kernel window)

SoC cycles from cannsim.log are kept for reference but are too coarse (~420)
to compare VF variants or tile sizes.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
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
    rvec_span: int | None = None
    pushq_vf_dur: int | None = None
    mte2_span: int | None = None
    mte3_span: int | None = None
    vector_span: int | None = None
    rvec_op_counts: dict[str, int] = field(default_factory=dict)
    rvec_event_count: int = 0
    trace_path: str | None = None

    @property
    def vf_cycles(self) -> int | None:
        """Best available VF latency estimate."""
        if self.rvec_span is not None and self.rvec_span > 0:
            return self.rvec_span
        if self.pushq_vf_dur is not None and self.pushq_vf_dur > 0:
            return self.pushq_vf_dur
        return None


@dataclass(frozen=True)
class SocCycleRecord:
    sim_wall_s: float
    soc_cycles: int


@dataclass(frozen=True)
class RunMetrics:
    out_dir: str
    cannsim_run_dir: str
    instr: InstrLogMetrics
    trace: TraceMetrics
    soc_cycles: tuple[SocCycleRecord, ...]
    steady_soc_cycles: int | None
    measured_kernel_cycles: int | None

    @property
    def primary_vf_cycles(self) -> int | None:
        if self.trace.vf_cycles is not None:
            return self.trace.vf_cycles
        if self.instr.max_dur > 0:
            return self.instr.max_dur
        return self.measured_kernel_cycles


def find_cannsim_run_dir(out_dir: str) -> str:
    pattern = os.path.join(out_dir, "cannsim_*")
    candidates = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if not candidates:
        raise FileNotFoundError(f"No cannsim_* directory under {out_dir}")
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
    launches: list[LaunchRecord], gap_threshold: int = 3000
) -> list[list[LaunchRecord]]:
    if not launches:
        return []
    sorted_launches = sorted(launches, key=lambda r: r.start)
    groups: list[list[LaunchRecord]] = [[sorted_launches[0]]]
    for record in sorted_launches[1:]:
        prev_end = max(r.tick for r in groups[-1])
        if record.start - prev_end > gap_threshold:
            groups.append([record])
        else:
            groups[-1].append(record)
    return groups


def metrics_from_launches(launches: Iterable[LaunchRecord]) -> InstrLogMetrics:
    items = list(launches)
    if not items:
        return InstrLogMetrics.empty()
    starts = [r.start for r in items]
    ticks = [r.tick for r in items]
    durations = [r.duration for r in items]
    blk_dims = [r.blk_dim for r in items]
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
        max_blk = max(r.blk_dim for r in all_launches)
        kernel_launches = [r for r in all_launches if r.blk_dim == max_blk]
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


def _load_trace_events(trace_path: str) -> list[dict]:
    with open(trace_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, list):
        return payload
    return payload.get("traceEvents", [])


def _pipe_span(events: list[dict], processes: dict[int, str], *needles: str) -> int | None:
    pids = {pid for pid, name in processes.items() if any(n in name.upper() for n in needles)}
    exec_ev = [e for e in events if e.get("ph") == "X" and "dur" in e and e.get("pid") in pids]
    if not exec_ev:
        return None
    start = min(e["ts"] for e in exec_ev)
    end = max(e["ts"] + e["dur"] for e in exec_ev)
    span = int(end - start)
    return span if span > 0 else None


def parse_trace_metrics(trace_path: str) -> TraceMetrics:
    if not os.path.isfile(trace_path):
        return TraceMetrics()

    events = _load_trace_events(trace_path)
    processes = {
        e["pid"]: e["args"]["name"]
        for e in events
        if e.get("ph") == "M" and e.get("name") == "process_name"
    }
    exec_ev = [e for e in events if e.get("ph") == "X" and "dur" in e]

    pushq_pids = {pid for pid, name in processes.items() if "PUSHQ" in name.upper()}
    vf_dispatch = [
        e for e in exec_ev if e.get("pid") in pushq_pids and "VF" in e.get("name", "")
    ]
    pushq_vf_dur = max((e.get("dur", 0) for e in vf_dispatch), default=0) or None

    rvec_pids = {pid for pid, name in processes.items() if "RVEC" in name.upper()}
    rvec_events = [e for e in exec_ev if e.get("pid") in rvec_pids]
    rvec_span = None
    rvec_op_counts: dict[str, int] = {}
    rvec_event_count = 0
    if rvec_events:
        start = min(e["ts"] for e in rvec_events)
        end = max(e["ts"] + e["dur"] for e in rvec_events)
        rvec_span = int(end - start)
        rvec_op_counts = dict(Counter(e.get("name", "?") for e in rvec_events))
        rvec_event_count = len(rvec_events)

    return TraceMetrics(
        rvec_span=rvec_span,
        pushq_vf_dur=pushq_vf_dur,
        mte2_span=_pipe_span(events, processes, "MTE2"),
        mte3_span=_pipe_span(events, processes, "MTE3"),
        vector_span=_pipe_span(events, processes, "RVEC", "VECTOR", "VEC"),
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
        rec = SocCycleRecord(sim_wall_s=float(match.group(1)), soc_cycles=int(match.group(2)))
        all_records.append(rec)
        if in_window:
            window_records.append(rec)

    return all_records, window_records


def measured_kernel_soc_cycles(cannsim_log: str, warmup_cycle: int = 248) -> int | None:
    _, window = parse_marker_soc_cycles(cannsim_log)
    measured = [r.soc_cycles for r in window if r.soc_cycles != warmup_cycle]
    if not measured:
        return None
    return measured[-1]


def parse_run_metrics(out_dir: str) -> RunMetrics:
    run_dir = find_cannsim_run_dir(out_dir)
    log_ca = os.path.join(run_dir, "log_ca")
    cannsim_log = os.path.join(run_dir, "cannsim.log")

    instr = parse_instr_log_dir(log_ca) if os.path.isdir(log_ca) else InstrLogMetrics.empty()
    soc = parse_soc_cycles(cannsim_log) if os.path.isfile(cannsim_log) else []
    measured = measured_kernel_soc_cycles(cannsim_log) if os.path.isfile(cannsim_log) else None

    trace_path = find_trace_json(run_dir)
    trace = parse_trace_metrics(trace_path) if trace_path else TraceMetrics()

    return RunMetrics(
        out_dir=out_dir,
        cannsim_run_dir=run_dir,
        instr=instr,
        trace=trace,
        soc_cycles=tuple(soc),
        steady_soc_cycles=steady_state_soc_cycles(soc),
        measured_kernel_cycles=measured,
    )


def format_run_summary(m: RunMetrics, label: str | None = None) -> str:
    """Human-readable one-run summary."""
    title = label or os.path.basename(m.out_dir.rstrip("/"))
    lines = [f"=== {title} ({m.out_dir}) ===", f"cannsim run dir: {m.cannsim_run_dir}"]

    primary = m.primary_vf_cycles
    if primary is not None:
        if m.trace.rvec_span and m.trace.rvec_span > 0:
            source = "RVEC span"
        elif m.trace.pushq_vf_dur and m.trace.pushq_vf_dur > 0:
            source = "PUSHQ VF dur"
        elif m.instr.max_dur > 0:
            source = "instr MaxDur"
        else:
            source = "SoC (fallback)"
        lines.append(f"primary VF cycles:         {primary}  ({source})")

    if m.trace.rvec_span:
        lines.append(f"RVEC span:                 {m.trace.rvec_span}")
    if m.trace.pushq_vf_dur:
        lines.append(f"PUSHQ VF dur:              {m.trace.pushq_vf_dur}")
    if m.trace.mte2_span:
        lines.append(f"MTE2 span:                 {m.trace.mte2_span}")
    if m.trace.mte3_span:
        lines.append(f"MTE3 span:                 {m.trace.mte3_span}")
    if m.trace.rvec_op_counts:
        top_ops = sorted(m.trace.rvec_op_counts.items(), key=lambda kv: -kv[1])[:6]
        lines.append("top RVEC ops: " + ", ".join(f"{k}={v}" for k, v in top_ops))

    if m.instr.max_dur:
        lines.append(
            f"instr MaxDur / span:       {m.instr.max_dur} / {m.instr.span} "
            f"(blkDim={m.instr.max_blk_dim})"
        )
    elif not os.path.isdir(os.path.join(m.cannsim_run_dir, "log_ca")):
        lines.append("instr MaxDur / span:       (log_ca unavailable)")

    if m.measured_kernel_cycles is not None:
        lines.append(
            f"SoC cycles (measured):     {m.measured_kernel_cycles}  "
            "(coarse; ~420 is normal)"
        )
    elif m.steady_soc_cycles is not None:
        lines.append(f"SoC cycles (steady):       {m.steady_soc_cycles}  (coarse)")

    if m.trace.trace_path:
        lines.append(f"trace:                     {m.trace.trace_path}")
    elif not find_trace_json(m.cannsim_run_dir):
        lines.append("trace:                     (none — need instr.bin + cannsim report)")

    return "\n".join(lines)


def format_table(rows: list[tuple[str, RunMetrics]]) -> str:
    header = (
        f"{'case':<22} {'primary':>8} {'RVEC':>8} {'MaxDur':>8} "
        f"{'Span':>8} {'SoC':>6}"
    )
    lines = [header, "-" * len(header)]
    for label, m in rows:
        primary = m.primary_vf_cycles
        lines.append(
            f"{label:<22} "
            f"{primary if primary is not None else '-':>8} "
            f"{m.trace.rvec_span if m.trace.rvec_span else '-':>8} "
            f"{m.instr.max_dur if m.instr.max_dur else '-':>8} "
            f"{m.instr.span if m.instr.span else '-':>8} "
            f"{m.measured_kernel_cycles if m.measured_kernel_cycles else '-':>6}"
        )
    return "\n".join(lines)


def default_cycle_out_dirs(sim_root: str | None = None) -> list[str]:
    """Return sim_outputs/cycle_{dtype}_{mode} dirs in canonical variant order."""
    root = sim_root or os.path.join(os.path.dirname(__file__), "..", "sim_outputs")
    try:
        from ref.tile_config import DTYPES, MODES

        ordered = [
            os.path.join(root, f"cycle_{dtype}_{mode}")
            for dtype in DTYPES
            for mode in MODES
            if os.path.isdir(os.path.join(root, f"cycle_{dtype}_{mode}"))
        ]
        if ordered:
            return ordered
    except ImportError:
        pass
    return sorted(
        p for p in glob.glob(os.path.join(root, "cycle_*")) if os.path.isdir(p)
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Parse rope VF cannsim cycle metrics")
    parser.add_argument(
        "out_dirs",
        nargs="*",
        help="sim_outputs subdirs (default: all cycle_{dtype}_{mode})",
    )
    parser.add_argument("--table", action="store_true", help="Print compact table")
    args = parser.parse_args(argv)

    dirs = args.out_dirs if args.out_dirs else default_cycle_out_dirs()
    if not dirs:
        print("No sim_outputs/cycle_* dirs found. Run scripts/run_cycle_bench.sh first.", file=sys.stderr)
        return 1

    rows: list[tuple[str, RunMetrics]] = []
    for path in dirs:
        try:
            m = parse_run_metrics(path)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            continue
        label = os.path.basename(path.rstrip("/"))
        rows.append((label, m))
        if not args.table:
            print(format_run_summary(m, label))
            print()

    if args.table and rows:
        print(format_table(rows))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
