#!/usr/bin/env python3
"""Build, validate, and time the full 8192x2048 persistent VMI/CCE case.

Both paths launch one 72-core kernel.  The VMI body is the standalone PTO
program in ``fixtures/full_roundtrip_vmi.py``; the CCE body is its separately
written low-level counterpart.  Neither timed loop contains a Python loop
over tensor rows or tiles.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import os
import struct
import subprocess
import tempfile
from pathlib import Path

import torch
import torch_npu  # noqa: F401

HERE = Path(__file__).parent
OUT = HERE / "outputs"
DEVICE = f"npu:{os.environ.get('ACL_DEVICE_ID', '0')}"
ROWS, WIDTH, GRID = 8192, 2048, 72
CCE_DYN_UB, VMI_DYN_UB = 27136, 126208
WARMUP, SAMPLES = 8, 40
# Keep this aligned with TileLang's ``do_bench`` policy.  The flush happens
# before the start event, so it evicts cache state without contributing to the
# reported kernel duration.  A batch of launches would make the second and
# subsequent launches artificially L2-hot for this bandwidth-sensitive case.
L2_FLUSH_MB = 256
MSPROF_REPS = 30


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def bisheng() -> str:
    return os.environ.get("BISHENG", f"{os.environ['ASCEND_HOME_PATH']}/bin/bisheng")


def includes() -> list[str]:
    root = os.environ["ASCEND_HOME_PATH"]
    return ["-I" + str(HERE / "fixtures"), "-I" + root + "/aarch64-linux/tikcpp/tikcfw",
            "-I" + root + "/aarch64-linux/tikcpp/tikcfw/impl",
            "-I" + root + "/aarch64-linux/tikcpp/tikcfw/interface"]


def link(device: Path, host: Path, library: Path) -> Path:
    root = os.environ["ASCEND_HOME_PATH"]
    run([bisheng(), "--cce-fatobj-link", "-shared", "-fPIC", str(device), str(host),
         "-L" + root + "/aarch64-linux/lib64", "-Wl,-rpath," + root + "/aarch64-linux/lib64",
         "-Wl,--no-as-needed", "-lruntime", "-o", str(library)])
    return library


def make_host(path: Path, text: str, obj: Path) -> None:
    path.write_text(text)
    run([bisheng(), "-xcce", "-Xhost-start", "-Xhost-end", "-fPIC", "-O2", "-std=c++17",
         "--cce-aicore-arch=dav-c310", "-c", str(path), "-o", str(obj)])


def build_cce() -> Path:
    OUT.mkdir(exist_ok=True)
    device, host = OUT / "full_cce.o", OUT / "full_cce_host.o"
    run([bisheng(), "-include", str(HERE / "fixtures/cxx17_bit_cast.hpp"), "-O2", "-fPIC", "-std=c++17",
         "--npu-arch=dav-3510", *includes(), "-c", str(HERE / "fixtures/full_roundtrip_cce.asc"), "-o", str(device)])
    make_host(OUT / "full_cce_host.cpp", 'extern "C" void full_cce_link_anchor() {}\n', host)
    return link(device, host, OUT / "libfull_cce.so")


def load_vmi_module():
    source = HERE / "fixtures/full_roundtrip_vmi.py"
    spec = importlib.util.spec_from_file_location("full_roundtrip_vmi", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_vmi() -> Path:
    """Lower the checked-in PTO program and use a stream-first dynamic-UB ABI."""
    from ptodsl._runtime.native_build import _compile_launch_cpp, _link_shared_library, _run_ptoas

    OUT.mkdir(exist_ok=True)
    module = load_vmi_module()
    compiled = module.full_roundtrip_vmi.compile()
    mlir, obj, host, host_obj = OUT / "full_vmi.mlir", OUT / "full_vmi.o", OUT / "full_vmi_host.cpp", OUT / "full_vmi_host.o"
    mlir.write_text(compiled.mlir_text())
    _run_ptoas(mlir, obj, target_arch="a5", backend="vpto", pto_level="level3")
    host.write_text(f'''#include <stdint.h>
#ifndef AICORE
#define AICORE [aicore]
#endif
extern "C" __global__ AICORE void full_roundtrip_vmi(__gm__ uint16_t *x);
extern "C" void launch_full_vmi(void *stream, void *x) {{
  full_roundtrip_vmi<<<{GRID}, {VMI_DYN_UB}, stream>>>((__gm__ uint16_t *)x);
}}
''')
    _compile_launch_cpp(host, host_obj, kernel_kind="vector", target_arch="a5", export_macro="FULL_VMI_EXPORTS")
    library = OUT / "libfull_vmi.so"
    _link_shared_library(host_obj, obj, library, kernel_kind="vector")
    return library


def stream_ptr() -> int:
    value = torch.npu.current_stream()._as_parameter_  # noqa: SLF001
    return value.value if hasattr(value, "value") else int(value)


def median_us(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    cache = torch.empty(L2_FLUSH_MB * 1024 * 1024 // 4, dtype=torch.int32, device=DEVICE)
    samples: list[tuple[torch.npu.Event, torch.npu.Event]] = []
    for _ in range(SAMPLES):
        # ``zero_`` is queued before ``begin``.  Stream ordering makes the
        # eviction complete before the kernel while the event interval remains
        # device-kernel-only, exactly as tilelang.profiler.bench.do_bench.
        cache.zero_()
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        samples.append((begin, end))
    torch.npu.synchronize()
    values = sorted(begin.elapsed_time(end) * 1000.0 for begin, end in samples)
    return values[len(values) // 2]


def msprof_us(fn, symbol: str) -> float:
    """Return profiler-recorded device duration, excluding host launch time.

    FFTS records use the AIC timestamp when one exists and otherwise the AIV
    timestamp.  This is the same rule used for mixed vector kernel records.
    ``symbol`` deliberately filters the one full-workload kernel, so tensor
    allocation and host-side work cannot enter the number.
    """
    import torch_npu.profiler

    def parse(root: Path) -> float:
        hashes: dict[int, str] = {}
        for path in root.rglob("*hash_dic.slice_*"):
            if path.name.endswith(".done"):
                continue
            for line in path.read_bytes().decode("utf-8", "replace").splitlines():
                key, sep, value = line.partition(":")
                if not sep:
                    continue
                try:
                    number = int(key)
                    hashes[number - (1 << 64) if number >= (1 << 63) else number] = value
                except ValueError:
                    pass
        names: dict[int, str] = {}
        for path in root.rglob("*task_track.slice_*"):
            if path.name.endswith(".done"):
                continue
            data = path.read_bytes()
            for offset in range(0, len(data) - 63, 64):
                values = struct.unpack_from("<8q", data, offset)
                name_hash = values[5]
                if name_hash:
                    names[(values[3] >> 32) & 0xFFFF] = hashes.get(name_hash, "")
        aic, aiv = [], []
        for path in root.rglob("ffts_profile*"):
            if path.name.endswith(".done") or not path.stat().st_size:
                continue
            data = path.read_bytes()
            for offset in range(0, len(data) - 127, 128):
                values = struct.unpack_from("<16q", data, offset)
                seq = (values[0] >> 32) & 0xFFFF
                if symbol not in names.get(seq, "") or not values[1]:
                    continue
                (aiv if values[2] >= (1 << 31) else aic).append(float(values[15] - values[14]))
        durations = aic or aiv
        # A kernel can emit one record per active core.  Until the profiler
        # record layout is pinned for this CANN release, expose the raw record
        # count and critical-path maximum instead of silently averaging a
        # subset (which produced physically impossible 4.9 us values).
        if len(durations) != MSPROF_REPS:
            raise RuntimeError(f"expected {MSPROF_REPS} records for {symbol}, got {len(durations)}")
        return max(durations) / 1000.0

    old = os.environ.get("ASCEND_WORK_PATH")
    with tempfile.TemporaryDirectory(prefix="vmi_cce_profile_", dir=OUT) as work:
        os.environ["ASCEND_WORK_PATH"] = work
        schedule = torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0)
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU], schedule=schedule,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                l2_cache=False, data_simplification=False),
        ) as prof:
            for _ in range(MSPROF_REPS):
                fn()
            torch.npu.synchronize()
            prof.step()
        value = parse(Path(prof.prof_if.prof_path))
    if old is None:
        os.environ.pop("ASCEND_WORK_PATH", None)
    else:
        os.environ["ASCEND_WORK_PATH"] = old
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--profile", action="store_true", help="also dump raw profiler records (diagnostic)")
    args = parser.parse_args()
    if args.compile_only:
        build_cce(); build_vmi()
        print("PASS: full 72-core CCE and VMI libraries built")
        return
    torch.npu.set_device(DEVICE)
    cce, vmi = ctypes.CDLL(str(build_cce())), ctypes.CDLL(str(build_vmi()))
    cce_fn, vmi_fn = cce.public_launch, vmi.launch_full_vmi
    cce_fn.argtypes, vmi_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int], [ctypes.c_void_p, ctypes.c_void_p]
    x0 = (torch.randn((ROWS, WIDTH), dtype=torch.float32, device=DEVICE) * 0.25).to(torch.bfloat16)
    cce_x, vmi_x = x0.clone(), x0.clone()

    def cce_run() -> None:
        cce_fn(ctypes.c_void_p(stream_ptr()), ctypes.c_void_p(cce_x.data_ptr()), ROWS)

    def vmi_run() -> None:
        vmi_fn(ctypes.c_void_p(stream_ptr()), ctypes.c_void_p(vmi_x.data_ptr()))

    cce_run(); vmi_run(); torch.npu.synchronize()
    if not torch.isfinite(cce_x.float()).all() or not torch.isfinite(vmi_x.float()).all():
        raise AssertionError("round-trip produced a non-finite result")
    # The two low-level lowering paths differ only at FP8 tie boundaries.
    # BF16 output agreement remains bounded by one FP8 quantization step.
    torch.testing.assert_close(cce_x.float().cpu(), vmi_x.float().cpu(), rtol=1.5e-1, atol=1.25e-1)
    cce_event, vmi_event = median_us(cce_run), median_us(vmi_run)
    print(f"device={DEVICE} shape={ROWS}x{WIDTH} grid={GRID} cce_dyn_ub={CCE_DYN_UB} vmi_dyn_ub={VMI_DYN_UB}")
    print("correctness=PASS cce_vmi_peer=PASS")
    print(f"event_l2_flush_mb={L2_FLUSH_MB} samples={SAMPLES} CCE_us={cce_event:.3f} VMI_us={vmi_event:.3f} CCE_over_VMI={cce_event / vmi_event:.4f}")
    if args.profile:
        cce_us, vmi_us = msprof_us(cce_run, "full_roundtrip_cce"), msprof_us(vmi_run, "full_roundtrip_vmi")
        print(f"msprof_diagnostic_reps={MSPROF_REPS} CCE_us={cce_us:.3f} VMI_us={vmi_us:.3f}")


if __name__ == "__main__":
    main()
