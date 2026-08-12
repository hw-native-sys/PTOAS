#!/usr/bin/env python3
"""Build, validate, and time the full 8192x2048 persistent VMI/CCE case.

The CCE path is one 72-core persistent kernel.  The no-round VMI path is the
equivalent two-kernel operation (quantize, then dequantize), because the
current VMI surface cannot fuse floating-point scales.  Both timed callables
use preallocated device buffers and contain no Python loop over rows or tiles.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import os
import struct
import subprocess
import tempfile
import shutil
from pathlib import Path

import torch
import torch_npu  # noqa: F401

HERE = Path(__file__).parent
OUT = HERE / "outputs"
DEVICE = f"npu:{os.environ.get('ACL_DEVICE_ID', '0')}"
ROWS, WIDTH, GRID = 8192, 2048, 72
# CCE is a 9-row x 2048-element, three-slot persistent schedule. VMI retains
# the production-shaped 32x1024 quantize and 64x256 dequantize tiles.  These
# dynamic-UB sizes are the allocator-selected launch contracts for exactly
# these two VMI bodies; using the hardware maximum changes the launch ABI and
# does not represent the production schedule.
CCE_DYN_UB, VMI_QUANT_UB, VMI_DEQUANT_UB = 231552, 221184, 204800
WARMUP, SAMPLES = 8, 40
# Keep this aligned with the device-profiler cache policy.  The flush happens
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


def link_vmi(objects: list[Path], host: Path, library: Path) -> Path:
    """Link separately lowered VMI stages into one ctypes-loadable library."""
    root = os.environ["ASCEND_HOME_PATH"]
    run([bisheng(), "--cce-fatobj-link", "-shared", "-fPIC", str(host),
         *map(str, objects), "-L" + root + "/aarch64-linux/lib64",
         "-Wl,-rpath," + root + "/aarch64-linux/lib64", "-Wl,--no-as-needed",
         "-lruntime", "-o", str(library)])
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


def build_vmi() -> tuple[Path, Path]:
    """Lower the checked-in PTO program and build its dynamic-UB launch ABI."""
    # Keep the reproducer self-contained: the pinned PTODSL source shipped in
    # this repository provides the native-build helpers.  A site-installed
    # ``ptodsl`` may be an older namespace package without ``_runtime``.
    import sys
    if "PTOAS_BIN" not in os.environ:
        candidates = [Path(sys.executable).parent / "ptoas"]
        if os.environ.get("CONDA_PREFIX"):
            candidates.append(Path(os.environ["CONDA_PREFIX"]) / "bin/ptoas")
        candidate = next((str(path) for path in candidates if path.is_file()), None)
        if candidate is None:
            candidate = shutil.which("ptoas")
        if candidate is None:
            raise RuntimeError("ptoas is not on PATH; set PTOAS_BIN to the pinned PTOAS executable")
        os.environ["PTOAS_BIN"] = candidate
    root = HERE.parent.parent.parent
    sys.path.insert(0, str(root / "ptodsl"))
    # The installed extension supplies the MLIR Python bindings; the checked
    # in ``ptoas`` namespace only supplies lightweight compatibility modules.
    import importlib
    import site
    ptoas = importlib.import_module("ptoas")
    for base in site.getsitepackages():
        bindings = str(Path(base) / "ptoas")
        if bindings not in ptoas.__path__:
            ptoas.__path__.append(bindings)
    from ptodsl._runtime.native_build import _compile_launch_cpp, _link_shared_library, _run_ptoas

    OUT.mkdir(exist_ok=True)
    module = load_vmi_module()
    quant, dequant = module.float_scale_quantize.compile(), module.float_scale_dequantize.compile()
    # PTOAS accepts one top-level module per invocation.  Keep the two
    # production stages as independent objects, then link them into one
    # convenience library.  Concatenating their modules happens to compile but
    # changes the entry/module ABI and can leave the first launch unusable.
    quant_mlir, dequant_mlir = OUT / "quantize.mlir", OUT / "dequantize.mlir"
    quant_obj, dequant_obj = OUT / "quantize.o", OUT / "dequantize.o"
    quant_mlir.write_text(quant.mlir_text())
    dequant_mlir.write_text(dequant.mlir_text())
    _run_ptoas(quant_mlir, quant_obj, target_arch="a5", backend="vpto", pto_level="level3")
    _run_ptoas(dequant_mlir, dequant_obj, target_arch="a5", backend="vpto", pto_level="level3")
    quant_host, dequant_host = OUT / "quantize_host.cpp", OUT / "dequantize_host.cpp"
    quant_host_obj, dequant_host_obj = OUT / "quantize_host.o", OUT / "dequantize_host.o"
    quant_host.write_text(f'''#include <stdint.h>
#ifndef AICORE
#define AICORE [aicore]
#endif
extern "C" __global__ AICORE void float_scale_quantize(__gm__ uint16_t *, __gm__ uint8_t *, __gm__ float *);
extern "C" void launch_quantize_vmi(void *x, void *q, void *sf, void *stream) {{
  float_scale_quantize<<<{GRID}, {VMI_QUANT_UB}, stream>>>((__gm__ uint16_t *)x, (__gm__ uint8_t *)q, (__gm__ float *)sf);
}}
''')
    dequant_host.write_text(f'''#include <stdint.h>
#ifndef AICORE
#define AICORE [aicore]
#endif
extern "C" __global__ AICORE void float_scale_dequantize(__gm__ uint16_t *, __gm__ uint8_t *, __gm__ float *);
extern "C" void launch_dequantize_vmi(void *x, void *q, void *sf, void *stream) {{
  float_scale_dequantize<<<{GRID}, {VMI_DEQUANT_UB}, stream>>>((__gm__ uint16_t *)x, (__gm__ uint8_t *)q, (__gm__ float *)sf);
}}
''')
    _compile_launch_cpp(quant_host, quant_host_obj, kernel_kind="vector", target_arch="a5", export_macro="QUANT_VMI_EXPORTS")
    _compile_launch_cpp(dequant_host, dequant_host_obj, kernel_kind="vector", target_arch="a5", export_macro="DEQUANT_VMI_EXPORTS")
    quant_library, dequant_library = OUT / "libquantize_vmi.so", OUT / "libdequantize_vmi.so"
    link_vmi([quant_obj], quant_host_obj, quant_library)
    link_vmi([dequant_obj], dequant_host_obj, dequant_library)
    return quant_library, dequant_library


def stream_ptr() -> int:
    return int(torch_npu._C._npu_getCurrentRawStream(torch.npu.current_device()))


def median_us(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    cache = torch.empty(L2_FLUSH_MB * 1024 * 1024 // 4, dtype=torch.int32, device=DEVICE)
    samples: list[tuple[torch.npu.Event, torch.npu.Event]] = []
    for _ in range(SAMPLES):
        # ``zero_`` is queued before ``begin``.  Stream ordering makes the
        # eviction complete before the kernel while the event interval remains
        # device-kernel-only with the same stream ordering as the profiler.
        cache.zero_()
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        samples.append((begin, end))
    torch.npu.synchronize()
    values = sorted(begin.elapsed_time(end) * 1000.0 for begin, end in samples)
    return values[len(values) // 2]


def msprof_us(fn, symbols: tuple[str, ...]) -> float:
    """Return profiler-recorded device duration, excluding host launch time.

    FFTS records use the AIC timestamp when one exists and otherwise the AIV
    timestamp.  This is the same rule used for mixed vector kernel records.
    ``symbols`` filter the workload kernels, so tensor allocation and host-side
    work cannot enter the number. The VMI operation has two required kernels;
    their records are summed per repetition before computing the mean.
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
                if not any(symbol in names.get(seq, "") for symbol in symbols) or not values[1]:
                    continue
                (aiv if values[2] >= (1 << 31) else aic).append(float(values[15] - values[14]))
        durations = aic or aiv
        # The production msprof benchmark sums all records for the named
        # kernel and divides by the number of repetitions.  This report has
        # one vector kernel per launch, so this is its mean device duration;
        # using the maximum sample inflated CCE and erased the real gap.
        expected = MSPROF_REPS * len(symbols)
        if len(durations) != expected:
            raise RuntimeError(f"expected {expected} records for {symbols}, got {len(durations)}")
        return sum(durations) / MSPROF_REPS / 1000.0

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
    parser.add_argument("--cce-only", action="store_true")
    parser.add_argument("--vmi-only", action="store_true")
    parser.add_argument("--quantize-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dequantize-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.quantize_only and args.dequantize_only:
        parser.error("choose at most one VMI stage diagnostic")
    if args.compile_only:
        build_cce(); build_vmi()
        print("PASS: full 72-core CCE and VMI libraries built")
        return
    torch.npu.set_device(DEVICE)
    cce_fn = None
    if not args.vmi_only:
        cce = ctypes.CDLL(str(build_cce()))
        cce_fn = cce.public_launch
        cce_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    if not args.cce_only:
        quant_library, dequant_library = build_vmi()
        quant_vmi = ctypes.CDLL(str(quant_library))
        dequant_vmi = ctypes.CDLL(str(dequant_library))
        quantize_fn, dequantize_fn = quant_vmi.launch_quantize_vmi, dequant_vmi.launch_dequantize_vmi
        quantize_fn.argtypes = dequantize_fn.argtypes = [ctypes.c_void_p] * 4
    x0 = (torch.randn((ROWS, WIDTH), dtype=torch.float32, device=DEVICE) * 0.25).to(torch.bfloat16)
    cce_x, vmi_x = x0.clone(), x0.clone()
    vmi_q = vmi_sf = None
    if not args.cce_only:
        vmi_q = torch.empty((ROWS, WIDTH), dtype=torch.float8_e4m3fn, device=DEVICE)
        vmi_sf = torch.empty((ROWS, WIDTH // 32), dtype=torch.float32, device=DEVICE)

    def cce_run() -> None:
        assert cce_fn is not None
        cce_fn(ctypes.c_void_p(stream_ptr()), ctypes.c_void_p(cce_x.data_ptr()), ROWS)

    def vmi_run() -> None:
        quantize_run()
        dequantize_run()

    def quantize_run() -> None:
        assert not args.cce_only and vmi_q is not None and vmi_sf is not None
        quantize_fn(ctypes.c_void_p(vmi_x.data_ptr()), ctypes.c_void_p(vmi_q.data_ptr()),
                    ctypes.c_void_p(vmi_sf.data_ptr()), ctypes.c_void_p(stream_ptr()))

    def dequantize_run() -> None:
        assert not args.cce_only and vmi_q is not None and vmi_sf is not None
        dequantize_fn(ctypes.c_void_p(vmi_x.data_ptr()), ctypes.c_void_p(vmi_q.data_ptr()),
                      ctypes.c_void_p(vmi_sf.data_ptr()), ctypes.c_void_p(stream_ptr()))

    if args.quantize_only:
        quantize_run(); torch.npu.synchronize()
        print("PASS: VMI quantize launch completed")
        return
    if args.dequantize_only:
        dequantize_run(); torch.npu.synchronize()
        print("PASS: VMI dequantize launch completed")
        return

    if not args.vmi_only:
        cce_run(); torch.npu.synchronize()
    if not args.cce_only:
        vmi_run(); torch.npu.synchronize()
    if args.cce_only:
        print("PASS: CCE launch completed")
        return
    if args.vmi_only:
        print("PASS: VMI launches completed")
        return
    if not torch.isfinite(cce_x.float()).all() or not torch.isfinite(vmi_x.float()).all():
        raise AssertionError("round-trip produced a non-finite result")
    # The two low-level lowering paths differ only at FP8 tie boundaries.
    # BF16 output agreement remains bounded by one FP8 quantization step.
    torch.testing.assert_close(cce_x.float().cpu(), vmi_x.float().cpu(), rtol=1.5e-1, atol=1.25e-1)
    cce_event, vmi_event = median_us(cce_run), median_us(vmi_run)
    print(
        f"device={DEVICE} shape={ROWS}x{WIDTH} grid={GRID} cce_dyn_ub={CCE_DYN_UB} "
        f"vmi_quant_ub={VMI_QUANT_UB} vmi_dequant_ub={VMI_DEQUANT_UB}"
    )
    print("correctness=PASS cce_vmi_peer=PASS")
    print(f"event_l2_flush_mb={L2_FLUSH_MB} samples={SAMPLES} CCE_us={cce_event:.3f} VMI_us={vmi_event:.3f} CCE_over_VMI={cce_event / vmi_event:.4f}")
    if args.profile:
        cce_us = msprof_us(cce_run, ("full_roundtrip_cce_kernel",))
        vmi_us = msprof_us(vmi_run, ("float_scale_quantize", "float_scale_dequantize"))
        quant_us = msprof_us(vmi_run, ("float_scale_quantize",))
        dequant_us = msprof_us(vmi_run, ("float_scale_dequantize",))
        print(f"msprof_device_reps={MSPROF_REPS} CCE_us={cce_us:.3f} VMI_us={vmi_us:.3f} CCE_over_VMI={cce_us / vmi_us:.4f}")
        print(f"msprof_vmi_components quantize_us={quant_us:.3f} dequantize_us={dequant_us:.3f}")


if __name__ == "__main__":
    main()
