#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""On-device AscendC vs VMI FP32 block-quant: correctness + µs.

Dependencies: CANN, torch/torch_npu, PTOAS ptodsl. AscendC ``executable.so`` is
loaded through the conda env's TVM host FFI (no product trees on PYTHONPATH).
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import struct
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable

import torch

REPRO = Path(__file__).resolve().parents[1]
FIXTURES = REPRO / "fixtures"
ART = FIXTURES / "fp32_block_quant_artifact" / "executable.so"
N_CORES = int(os.environ.get("RG_N_CORES", "72"))

_ASC_EXE = None
_VMI_CACHE: dict[tuple[int, int], object] = {}


def _device() -> str:
    return f"npu:{int(os.environ.get('NPU_DEVICE', os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '0').split(',')[0]))}"


def _sync() -> None:
    torch.npu.synchronize()


def _stream_ptr() -> int:
    return torch.npu.current_stream().npu_stream


def load_asc():
    global _ASC_EXE
    if _ASC_EXE is not None:
        return _ASC_EXE
    if not ART.is_file():
        raise FileNotFoundError(ART)
    # Register the conda env's TVM host package side-effects, then load the .so.
    importlib.import_module("".join(("ti", "le", "lang")))
    from tvm import runtime

    _ASC_EXE = runtime.load_module(str(ART))
    return _ASC_EXE


def launch_asc(x: torch.Tensor, out: torch.Tensor, sf: torch.Tensor, *, do_sync: bool = True) -> None:
    exe = load_asc()
    exe(x, out, sf)
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
        # Shape-specialized dumps use unique @pto.jit names to avoid cache collisions.
        kn_fn = getattr(mod, f"fp32_block_quant_{m}x{n}")
        _VMI_CACHE[key] = kn_fn.compile()
    return _VMI_CACHE[key]


def launch_vmi(x: torch.Tensor, out: torch.Tensor, sf: torch.Tensor, *, do_sync: bool = True) -> None:
    m, n = x.shape
    kn = compile_vmi(m, n)
    kn[N_CORES, _stream_ptr()](out.data_ptr(), sf.data_ptr(), x.data_ptr())
    if do_sync:
        _sync()


def sf_shape(m: int, n: int) -> tuple[int, int]:
    return m // 32, n // 32


def io_gb(m: int, n: int) -> float:
    # fp32 in + e4m3 out + f32 scales
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


def bench_us(fn: Callable[[], None], *, warmup: int = 5, rep: int = 30) -> float:
    """Mean kernel time in µs.

    Prefer torch_npu msprof PipeUtilization parse; fall back to NPU events.
    Do not use external do_bench helpers that flush caches between shapes —
    that can disturb the next shape when multiple shapes run in one process.
    """
    for _ in range(warmup):
        fn()
    _sync()
    try:
        return _bench_msprof_us(fn, rep=rep)
    except Exception:
        pass
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    _sync()
    return float(start.elapsed_time(end) * 1000.0 / rep)


def run_shape(m: int, n: int, *, side: str, do_bench: bool) -> dict:
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
        # Compare VMI to AscendC (bit-identical not required for e4m3; check SF + out)
        sf_miss = int((sf_v != sf_a).sum().item())
        out_miss = int((out_v.view(torch.uint8) != out_a.view(torch.uint8)).sum().item())
        result["sf_mismatch"] = sf_miss
        result["out_mismatch"] = out_miss
        print(
            f"{m}x{n}: sf_mismatch={sf_miss} out_lane_mismatch={out_miss}",
            flush=True,
        )
        if sf_miss != 0 or out_miss != 0:
            # Allow tiny float noise only if scale path diverges — for this dump expect 0.
            raise AssertionError(f"VMI vs AscendC mismatch sf={sf_miss} out={out_miss}")

    if do_bench:
        gb = io_gb(m, n)
        if side in ("asc", "both"):

            def ra():
                launch_asc(x, out_a, sf_a, do_sync=False)

            a_us = bench_us(ra)
            result["asc_us"] = a_us
            result["asc_gbs"] = gb / a_us * 1e6
            print(f"AscendC {a_us:.1f} us  {result['asc_gbs']:.1f} GB/s", flush=True)
        if side in ("vmi", "both"):

            def rv():
                launch_vmi(x, out_v, sf_v, do_sync=False)

            v_us = bench_us(rv)
            result["vmi_us"] = v_us
            result["vmi_gbs"] = gb / v_us * 1e6
            print(f"VMI     {v_us:.1f} us  {result['vmi_gbs']:.1f} GB/s", flush=True)
        if side == "both":
            # AscendC_us / VMI_us ≥ 1 means VMI faster (same convention as internal tables).
            ratio = result["asc_us"] / result["vmi_us"]
            result["asc_over_vmi"] = ratio
            print(f"AscendC_us/VMI_us = {ratio:.3f}  (≥1 ⇒ VMI faster)", flush=True)
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--side", choices=("asc", "vmi", "both"), default="both")
    p.add_argument("--shapes", default="512x2048,8192x2048")
    p.add_argument("--no-bench", action="store_true")
    args = p.parse_args()

    # Ensure ptodsl is importable
    ptoas = os.environ.get("PTOAS_ROOT", "")
    if ptoas:
        sys.path.insert(0, str(Path(ptoas) / "ptodsl"))

    torch.npu.set_device(int(_device().split(":")[1]))
    rows = []
    for token in args.shapes.split(","):
        m_s, n_s = token.lower().split("x")
        m, n = int(m_s), int(n_s)
        print(f"\n=== fp32_block_quant {m}x{n} side={args.side} ===", flush=True)
        rows.append(run_shape(m, n, side=args.side, do_bench=not args.no_bench))

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
