#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""On-device AscendC vs VMI FP32 block-quant: correctness + µs.

Dependencies: CANN, torch/torch_npu, PTOAS ptodsl.
AscendC: ctypes launch of libfp32_block_quant.so (bisheng --shared).
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import os
import struct
import subprocess
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable

import torch
import torch_npu  # noqa: F401  # registers torch.npu

# This script lives under backlog/scripts/; package root is two levels up.
REPRO = Path(__file__).resolve().parents[2]
BACKLOG = REPRO / "backlog"
FIXTURES = BACKLOG / "fixtures"
ASC_SO = FIXTURES / "fp32_block_quant_artifact" / "libfp32_block_quant.so"
BUILD_SH = BACKLOG / "scripts" / "build_fp32_block_quant_asc.sh"
N_CORES = int(os.environ.get("RG_N_CORES", "72"))

_ASC_LIB: ctypes.CDLL | None = None
_VMI_CACHE: dict[tuple[int, int], object] = {}
# Reuse LaunchHandle: kn[grid, stream] builds a new handle each time and reloads
# the native .so on first call (~10–20 ms), which destroys Event timing.
_VMI_LAUNCH: dict[tuple[int, int], object] = {}


def _device() -> str:
    return f"npu:{int(os.environ.get('NPU_DEVICE', os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '0').split(',')[0]))}"


def _sync() -> None:
    torch.npu.synchronize()


def _stream_ptr() -> int:
    return torch.npu.current_stream().npu_stream


def ensure_asc_lib() -> Path:
    if ASC_SO.is_file() and os.environ.get("FP32_BQ_REBUILD", "") != "1":
        return ASC_SO
    if not BUILD_SH.is_file():
        raise FileNotFoundError(BUILD_SH)
    subprocess.run(["bash", str(BUILD_SH)], check=True)
    if not ASC_SO.is_file():
        raise FileNotFoundError(ASC_SO)
    return ASC_SO


def load_asc() -> ctypes.CDLL:
    global _ASC_LIB
    if _ASC_LIB is not None:
        return _ASC_LIB
    so = ensure_asc_lib()
    lib = ctypes.CDLL(str(so))
    lib.call_fp32_block_quant.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.call_fp32_block_quant.restype = None
    _ASC_LIB = lib
    return lib


def launch_asc(x: torch.Tensor, out: torch.Tensor, sf: torch.Tensor, *, do_sync: bool = True) -> None:
    m, n = x.shape
    load_asc().call_fp32_block_quant(
        ctypes.c_void_p(_stream_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(sf.data_ptr()),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int32(m),
        ctypes.c_int32(n // 32),
    )
    if do_sync:
        _sync()


def _load_vmi_module(m: int, n: int):
    name = f"current_vmi_fp32_block_quant_{m}x{n}.ptodsl.py"
    path = FIXTURES / name
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(f"vmi_fp32_bq_{m}_{n}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def compile_vmi(m: int, n: int):
    key = (m, n)
    if key not in _VMI_CACHE:
        mod = _load_vmi_module(m, n)
        kn_fn = getattr(mod, f"fp32_block_quant_{m}x{n}")
        _VMI_CACHE[key] = kn_fn.compile()
    return _VMI_CACHE[key]


def _vmi_launch_handle(m: int, n: int):
    key = (m, n)
    h = _VMI_LAUNCH.get(key)
    if h is None:
        # stream=None → current torch_npu stream inside ptodsl launch.
        h = compile_vmi(m, n)[N_CORES, None]
        _VMI_LAUNCH[key] = h
    return h


def launch_vmi(x: torch.Tensor, out: torch.Tensor, sf: torch.Tensor, *, do_sync: bool = True) -> None:
    m, n = x.shape
    _vmi_launch_handle(m, n)(out.data_ptr(), sf.data_ptr(), x.data_ptr())
    if do_sync:
        _sync()


def sf_shape(m: int, n: int) -> tuple[int, int]:
    return m // 32, n // 32


def io_gb(m: int, n: int) -> float:
    return (m * n * 4 + m * n + (m // 32) * (n // 32) * 4) / 1e9


@contextmanager
def _suppress_io():
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        yield


def _parse_ffts_total_ns(prof_path: Path) -> float:
    by_name: dict[str, list[float]] = defaultdict(list)
    for f in prof_path.rglob("ffts_profile*"):
        if f.name.endswith(".done") or f.stat().st_size == 0:
            continue
        data = f.read_bytes()
        for i in range(len(data) // 128):
            vals = struct.unpack_from("<16q", data, i * 128)
            if vals[1] == 0:
                continue
            is_aiv = vals[2] >= (1 << 31)
            dur_ns = float(vals[15] - vals[14])
            by_name["aiv" if is_aiv else "aic"].append(dur_ns)
    durs = by_name.get("aiv") or by_name.get("aic") or []
    return float(sum(durs))


def _bench_msprof_us(fn: Callable[[], None], *, rep: int) -> float:
    import torch_npu.profiler

    tmp = tempfile.mkdtemp(prefix="fq_msprof_")
    old = os.environ.get("PROFILING_WORK_PATH")
    os.environ["PROFILING_WORK_PATH"] = tmp
    try:
        with _suppress_io():
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(
                    wait=0, warmup=0, active=1, repeat=1, skip_first=0
                ),
                on_trace_ready=lambda _p: None,
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                    l2_cache=False,
                    data_simplification=False,
                ),
            ) as prof:
                for _ in range(rep):
                    fn()
                _sync()
                prof.step()
            total_ns = _parse_ffts_total_ns(Path(prof.prof_if.prof_path))
        if total_ns <= 0:
            raise RuntimeError("msprof ffts_profile empty")
        return total_ns / rep / 1e3
    finally:
        if old is None:
            os.environ.pop("PROFILING_WORK_PATH", None)
        else:
            os.environ["PROFILING_WORK_PATH"] = old
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


def _bench_event_us(fn: Callable[[], None], *, rep: int) -> float:
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    _sync()
    return float(start.elapsed_time(end) * 1000.0 / rep)


def bench_us(
    fn: Callable[[], None],
    *,
    warmup: int = 5,
    rep: int = 30,
    timer: str = "event",
) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    if timer == "msprof":
        return _bench_msprof_us(fn, rep=rep)
    if timer == "auto":
        try:
            return _bench_msprof_us(fn, rep=rep)
        except Exception:
            pass
    return _bench_event_us(fn, rep=rep)


def run_shape(m: int, n: int, *, side: str, do_bench: bool, timer: str) -> dict:
    torch.manual_seed(0)
    dev = _device()
    x = torch.randn(m, n, dtype=torch.float32, device=dev).clamp(-10, 10)
    out_a = torch.empty(m, n, dtype=torch.float8_e4m3fn, device=dev)
    sf_a = torch.empty(*sf_shape(m, n), dtype=torch.float32, device=dev)
    out_v = torch.empty_like(out_a)
    sf_v = torch.empty_like(sf_a)

    result: dict = {"m": m, "n": n}
    if side in ("asc", "both"):
        launch_asc(x, out_a, sf_a)
        result["asc_ok"] = True
    if side in ("vmi", "both"):
        launch_vmi(x, out_v, sf_v)
        result["vmi_ok"] = True
    if side == "both":
        sf_miss = int((sf_v != sf_a).sum().item())
        out_miss = int((out_v.view(torch.uint8) != out_a.view(torch.uint8)).sum().item())
        result["sf_mismatch"] = sf_miss
        result["out_mismatch"] = out_miss
        print(
            f"{m}x{n}: sf_mismatch={sf_miss} out_lane_mismatch={out_miss}",
            flush=True,
        )
        if sf_miss != 0 or out_miss != 0:
            raise AssertionError(f"VMI vs AscendC mismatch sf={sf_miss} out={out_miss}")

    if do_bench:
        gb = io_gb(m, n)
        if side in ("asc", "both"):

            def ra():
                launch_asc(x, out_a, sf_a, do_sync=False)

            a_us = bench_us(ra, timer=timer)
            result["asc_us"] = a_us
            result["asc_gbs"] = gb / a_us * 1e6
            print(f"AscendC {a_us:.1f} us  {result['asc_gbs']:.1f} GB/s", flush=True)
        if side in ("vmi", "both"):

            def rv():
                launch_vmi(x, out_v, sf_v, do_sync=False)

            v_us = bench_us(rv, timer=timer)
            result["vmi_us"] = v_us
            result["vmi_gbs"] = gb / v_us * 1e6
            print(f"VMI     {v_us:.1f} us  {result['vmi_gbs']:.1f} GB/s", flush=True)
        if side == "both":
            ratio = result["asc_us"] / result["vmi_us"]
            result["asc_over_vmi"] = ratio
            print(f"AscendC_us/VMI_us = {ratio:.3f}  (≥1 ⇒ VMI faster)", flush=True)
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--side", choices=("asc", "vmi", "both"), default="both")
    p.add_argument("--shapes", default="512x2048,8192x2048")
    p.add_argument("--no-bench", action="store_true")
    p.add_argument(
        "--timer",
        choices=("event", "msprof", "auto"),
        default=os.environ.get("FP32_BQ_TIMER", "event"),
        help="Host µs source (default: npu Event; msprof ffts can mis-sum)",
    )
    args = p.parse_args()

    ptoas = os.environ.get("PTOAS_ROOT", "")
    if ptoas:
        sys.path.insert(0, str(Path(ptoas) / "ptodsl"))

    torch.npu.set_device(int(_device().split(":")[1]))
    if args.side in ("asc", "both"):
        # Rebuild at most once per process when FP32_BQ_REBUILD=1.
        ensure_asc_lib()
        os.environ.pop("FP32_BQ_REBUILD", None)

    rows = []
    for token in args.shapes.split(","):
        m_s, n_s = token.lower().split("x")
        m, n = int(m_s), int(n_s)
        print(f"\n=== fp32_block_quant {m}x{n} side={args.side} timer={args.timer} ===", flush=True)
        rows.append(
            run_shape(
                m, n, side=args.side, do_bench=not args.no_bench, timer=args.timer
            )
        )

    print("\n# summary", flush=True)
    for r in rows:
        if "asc_over_vmi" in r:
            print(
                f"{r['m']}x{r['n']}: AscendC {r['asc_us']:.1f} us  VMI {r['vmi_us']:.1f} us  "
                f"ratio {r['asc_over_vmi']:.3f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
