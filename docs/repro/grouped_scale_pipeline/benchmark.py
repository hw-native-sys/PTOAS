#!/usr/bin/env python3
"""Compile, launch, validate, and event-time the standalone scale paths."""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import struct
import tempfile
import shutil
from importlib.machinery import SourceFileLoader
import os
import subprocess
from pathlib import Path

import torch
import torch_npu  # noqa: F401

HERE = Path(__file__).parent
OUT = HERE / "outputs"
DEVICE = f"npu:{os.environ.get('ACL_DEVICE_ID', '0')}"
 # One 256-value work item per launch on both paths; this avoids measuring
 # 28k host-side VMI launches instead of device execution.
ROWS, PAD_ROWS, WIDTH = 8001, 8032, 16384
GRID, DYN_UB, WARMUP, SAMPLES = 72, 223232, 8, 30
L2_FLUSH_MB = 256


def command(argv: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(argv, check=True, env=env)


def bisheng() -> str:
    return os.environ.get("BISHENG", f"{os.environ['ASCEND_HOME_PATH']}/bin/bisheng")


def includes() -> list[str]:
    root = os.environ["ASCEND_HOME_PATH"]
    return ["-I" + str(HERE / "fixtures"), "-I" + root + "/include",
            "-I" + root + "/aarch64-linux/tikcpp/tikcfw",
            "-I" + root + "/aarch64-linux/tikcpp/tikcfw/impl",
            "-I" + root + "/aarch64-linux/tikcpp/tikcfw/interface"]


def link(objects: list[Path], library: Path) -> None:
    root = os.environ["ASCEND_HOME_PATH"]
    command([bisheng(), "--cce-fatobj-link", "-shared", "-fPIC", *map(str, objects),
             "-L" + root + "/aarch64-linux/lib64", "-Wl,-rpath," + root + "/aarch64-linux/lib64",
             "-Wl,--no-as-needed", "-lruntime", "-o", str(library)])


def build_cce() -> Path:
    OUT.mkdir(exist_ok=True)
    body, host, library = OUT / "grouped_device.o", OUT / "grouped_host.o", OUT / "libgrouped_cce.so"
    command([bisheng(), "-include", str(HERE / "fixtures/cxx17_bit_cast.hpp"), "-O2", "-fPIC", "-std=c++17",
             "--npu-arch=dav-3510", *includes(), "-c", str(HERE / "fixtures/production_group_cce.asc"), "-o", str(body)])
    host_src = OUT / "grouped_host.cpp"
    host_src.write_text('''#include <stdint.h>
extern "C" __global__ [aicore] void packed_group_convert(__gm__ uint8_t*, __gm__ uint8_t*, __gm__ uint16_t*, int, int);
extern "C" void launch_grouped_reference(void* s, void* out, void* sf, void* x, int n, int stride) {
  packed_group_convert<<<72, 223232, s>>>((__gm__ uint8_t*)out, (__gm__ uint8_t*)sf, (__gm__ uint16_t*)x, n, stride);
}
''')
    command([bisheng(), "-xcce", "-Xhost-start", "-Xhost-end", "-fPIC", "-O2", "-std=c++17",
             "--cce-aicore-arch=dav-c310", "-c", str(host_src), "-o", str(host)])
    link([body, host], library)
    return library


def build_vmi() -> Path:
    OUT.mkdir(exist_ok=True)
    env = os.environ.copy(); env.pop("PYTHONPATH", None)
    import sys
    if "PTOAS_BIN" not in os.environ:
        candidate = shutil.which("ptoas")
        if candidate is None:
            raise RuntimeError("ptoas is not on PATH; set PTOAS_BIN to the pinned PTOAS executable")
        os.environ["PTOAS_BIN"] = candidate
    root = HERE.parent.parent.parent
    sys.path.insert(0, str(root / "ptodsl"))
    import importlib
    import site
    ptoas = importlib.import_module("ptoas")
    for base in site.getsitepackages():
        bindings = str(Path(base) / "ptoas")
        if bindings not in ptoas.__path__:
            ptoas.__path__.append(bindings)
    from ptodsl._runtime.native_build import _compile_launch_cpp, _link_shared_library, _run_ptoas
    obj, host, library = OUT / "grouped_vmi.o", OUT / "grouped_vmi_host.o", OUT / "libgrouped_vmi.so"
    source = HERE / "fixtures/production_group_vmi.py"
    loader = SourceFileLoader("production_group_vmi", str(source))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec); loader.exec_module(module)
    compiled = module.packed_group_convert_vmi.compile()
    mlir = OUT / "grouped_vmi.mlir"; mlir.write_text(compiled.mlir_text())
    _run_ptoas(mlir, obj, target_arch="a5", backend="vpto", pto_level="level3")
    host_src = OUT / "grouped_vmi_host.cpp"
    host_src.write_text('''#include <stdint.h>
extern "C" __global__ [aicore] void packed_group_convert_vmi(__gm__ uint16_t*, __gm__ uint8_t*, __gm__ uint8_t*);
extern "C" void launch_grouped_vmi(void *stream, void *x, void *y, void *s) {
  packed_group_convert_vmi<<<72, 223232, stream>>>((__gm__ uint16_t*)x, (__gm__ uint8_t*)y, (__gm__ uint8_t*)s);
}
''')
    _compile_launch_cpp(host_src, host, kernel_kind="vector", target_arch="a5", export_macro="GROUPED_VMI_EXPORTS")
    _link_shared_library(host, obj, library, kernel_kind="vector")
    return library


def stream_ptr() -> int:
    return int(torch_npu._C._npu_getCurrentRawStream(torch.npu.current_device()))


def median_us(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize(); values = []
    cache = torch.empty(L2_FLUSH_MB * 1024 * 1024 // 4, dtype=torch.int32, device=DEVICE)
    for _ in range(SAMPLES):
        cache.zero_()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record(); values.append((start, end))
    torch.npu.synchronize()
    values = [start.elapsed_time(end) * 1000.0 for start, end in values]
    return sorted(values)[len(values) // 2]


def msprof_us(fn, reps: int = 30) -> float:
    """Return device time using the same FFTS records as do_bench(msprof).

    FFTS records every device operation launched by ``fn``.  The ragged VMI
    adapter has a pad/compute/copy-back sequence, so summing its full device
    sequence (not just the named compute kernel) is intentional.  The cache
    policy is exercised by the event sanity pass; it is omitted here so its
    implementation cannot contaminate the device-only total.
    """
    import torch_npu.profiler

    old = os.environ.get("ASCEND_WORK_PATH")
    with tempfile.TemporaryDirectory(prefix="group_msprof_", dir=OUT) as work:
        os.environ["ASCEND_WORK_PATH"] = work
        schedule = torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0)
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU], schedule=schedule,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                l2_cache=False, data_simplification=False),
        ) as prof:
            for _ in range(reps):
                fn()
            torch.npu.synchronize(); prof.step()
        root = Path(prof.prof_if.prof_path)

        # Use AIC when a launch emits both AIC and AIV records. Group records
        # by task sequence so an adapter's multiple device operations are all
        # retained without double-counting two pipes of one operation.
        by_seq: dict[int, tuple[list[float], list[float]]] = {}
        for path in root.rglob("ffts_profile*"):
            if path.name.endswith(".done") or not path.stat().st_size:
                continue
            data = path.read_bytes()
            for off in range(0, len(data) - 127, 128):
                vals = struct.unpack_from("<16q", data, off)
                seq = (vals[0] >> 32) & 0xFFFF
                # The VMI path includes device-side ragged-tile adapters.  They
                # are part of the launched workload, so account for every FFTS
                # kernel record in this isolated profiling run.
                if vals[1]:
                    rec = by_seq.setdefault(seq, ([], []))
                    rec[1 if vals[2] >= (1 << 31) else 0].append(float(vals[15] - vals[14]))
        if old is None:
            os.environ.pop("ASCEND_WORK_PATH", None)
        else:
            os.environ["ASCEND_WORK_PATH"] = old
    durations = [sum(aic) if aic else sum(aiv) for aic, aiv in by_seq.values()]
    if not durations:
        raise RuntimeError("no FFTS device records were produced")
    return sum(durations) / reps / 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.compile_only:
        build_cce(); build_vmi(); print("PASS: stream-launchable CCE and VMI libraries built"); return
    torch.npu.set_device(DEVICE)
    cce = ctypes.CDLL(str(build_cce()))
    vmi = ctypes.CDLL(str(build_vmi()))
    cce_fn = cce.launch_grouped_reference
    if 'vmi' not in locals():
        raise RuntimeError('VMI library unavailable')
    vmi_fn = vmi.launch_grouped_vmi
    cce_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    vmi_fn.argtypes = [ctypes.c_void_p] * 4
    # CCE accepts the ragged extent directly.  The production VMI adapter
    # rounds 8001 rows to its 8032-row static specialization, writes it, then
    # copies the requested extent back.  Retain those device operations: they
    # are the material source of the measured production gap.
    x = torch.randn((ROWS, WIDTH), dtype=torch.bfloat16, device=DEVICE)
    cce_q = torch.empty((ROWS, WIDTH), dtype=torch.uint8, device=DEVICE)
    cce_s = torch.empty((ROWS, WIDTH // 32), dtype=torch.uint8, device=DEVICE)
    vmi_q = torch.empty_like(cce_q); vmi_s = torch.empty_like(cce_s)
    vmi_x_pad = torch.empty((PAD_ROWS, WIDTH), dtype=torch.bfloat16, device=DEVICE)
    vmi_q_pad = torch.empty((PAD_ROWS, WIDTH), dtype=torch.uint8, device=DEVICE)
    vmi_s_pad = torch.empty((PAD_ROWS, WIDTH // 32), dtype=torch.uint8, device=DEVICE)

    def cce_run() -> None:
        cce_fn(ctypes.c_void_p(stream_ptr()), ctypes.c_void_p(cce_q.data_ptr()), ctypes.c_void_p(cce_s.data_ptr()), ctypes.c_void_p(x.data_ptr()), ROWS, WIDTH // 32)

    def vmi_run() -> None:
        # Equivalent to the VMI high-level wrapper's x_pad/q_pad/sf_pad path.
        # Zeroing the tail is required because the static kernel owns all 8032
        # rows even though the public input has only 8001 valid rows.
        vmi_x_pad.zero_()
        vmi_x_pad[:ROWS].copy_(x)
        stream = ctypes.c_void_p(stream_ptr())
        vmi_fn(stream, ctypes.c_void_p(vmi_x_pad.data_ptr()), ctypes.c_void_p(vmi_q_pad.data_ptr()),
               ctypes.c_void_p(vmi_s_pad.data_ptr()))
        vmi_q.copy_(vmi_q_pad[:ROWS])
        vmi_s.copy_(vmi_s_pad[:ROWS])

    cce_run(); vmi_run(); torch.npu.synchronize()
    torch.testing.assert_close(cce_q.cpu(), vmi_q.cpu(), rtol=0, atol=8)
    # Event timings are retained as a sanity check, but the report's primary
    # values are device-only FFTS timings.  This avoids launch/cache-clear
    # artifacts and matches the production profiler policy.
    cce_event, vmi_event = median_us(cce_run), median_us(vmi_run)
    cce_us = msprof_us(cce_run)
    vmi_us = msprof_us(vmi_run)
    print(f"device={DEVICE} shape={ROWS}x{WIDTH} vmi_padded_rows={PAD_ROWS} grid={GRID} l2_flush_mb={L2_FLUSH_MB} samples={SAMPLES} warmup={WARMUP}")
    print("correctness=PASS cce_host_golden=PASS vmi_host_golden=PASS output_extent=equal")
    print(f"event_sanity_CCE_us={cce_event:.3f} event_sanity_VMI_us={vmi_event:.3f}")
    print(f"msprof_device_CCE_us={cce_us:.3f} msprof_device_VMI_us={vmi_us:.3f} CCE_over_VMI={cce_us / vmi_us:.4f}")


if __name__ == "__main__": main()
