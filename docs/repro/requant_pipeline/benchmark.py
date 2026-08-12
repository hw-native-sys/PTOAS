#!/usr/bin/env python3
"""Standalone production-shaped FP8 rescale benchmark for A5.

The VMI implementation is intentionally a two-stage composition.  Its first
stage widens packed FP8 input in eight 1024-row strips; its second stage
computes group maxima and requantizes the complete tensor.  The CCE control
implements the same transformation in one 72-core kernel.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import os
import struct
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
import torch_npu  # noqa: F401

HERE = Path(__file__).parent
OUT = HERE / "outputs"
DEVICE = f"npu:{os.environ.get('ACL_DEVICE_ID', '0')}"
ROWS, WIDTH, STRIP_ROWS, GRID = 8064, 7168, 1024, 72
INPUT_SF_COLS, OUTPUT_SF_ROWS = WIDTH // 32, ROWS // 32
CCE_DYN_UB, UNPACK_DYN_UB, REQUANT_DYN_UB = 158144, 101376, 204800
WARMUP, SAMPLES, MSPROF_REPS = 8, 30, 30
L2_FLUSH_MB = 256


def run(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


def bisheng() -> str:
    return os.environ.get("BISHENG", f"{os.environ['ASCEND_HOME_PATH']}/bin/bisheng")


def includes() -> list[str]:
    root = os.environ["ASCEND_HOME_PATH"]
    return [
        "-I" + str(HERE / "fixtures"),
        "-I" + root + "/include",
        "-I" + root + "/aarch64-linux/tikcpp/tikcfw",
        "-I" + root + "/aarch64-linux/tikcpp/tikcfw/impl",
        "-I" + root + "/aarch64-linux/tikcpp/tikcfw/interface",
    ]


def link(objects: list[Path], library: Path) -> Path:
    root = os.environ["ASCEND_HOME_PATH"]
    run([
        bisheng(), "--cce-fatobj-link", "-shared", "-fPIC", *map(str, objects),
        "-L" + root + "/aarch64-linux/lib64", "-Wl,-rpath," + root + "/aarch64-linux/lib64",
        "-Wl,--no-as-needed", "-lruntime", "-o", str(library),
    ])
    return library


def make_host(path: Path, text: str, obj: Path) -> None:
    path.write_text(text)
    run([
        bisheng(), "-xcce", "-Xhost-start", "-Xhost-end", "-fPIC", "-O2", "-std=c++17",
        "--cce-aicore-arch=dav-c310", "-c", str(path), "-o", str(obj),
    ])


def build_cce() -> Path:
    OUT.mkdir(exist_ok=True)
    body, host = OUT / "rescale_cce.o", OUT / "rescale_cce_host.o"
    run([
        bisheng(), "-include", str(HERE / "fixtures/cxx17_bit_cast.hpp"), "-O2", "-fPIC", "-std=c++17",
        "--npu-arch=dav-3510", *includes(), "-c", str(HERE / "fixtures/production_reference.asc"), "-o", str(body),
    ])
    # The device source exports the stream-first launch function.  A tiny host
    # object supplies a stable shared-library link unit without a framework ABI.
    make_host(host.with_suffix(".cpp"), 'extern "C" void rescale_cce_link_anchor() {}\n', host)
    return link([body, host], OUT / "librescale_cce.so")


def load_fixture(source_name: str, kernel_name: str):
    source = HERE / "fixtures" / f"{source_name}.py"
    spec = importlib.util.spec_from_file_location(kernel_name, source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pto_native_build():
    """Use the pinned PTODSL native builder without a framework launcher."""
    import site
    import sys

    root = HERE.parent.parent.parent
    sys.path.insert(0, str(root / "ptodsl"))
    import importlib

    ptoas_module = importlib.import_module("ptoas")
    for base in site.getsitepackages():
        bindings = str(Path(base) / "ptoas")
        if bindings not in ptoas_module.__path__:
            ptoas_module.__path__.append(bindings)
    from ptodsl._runtime.native_build import _compile_launch_cpp, _link_shared_library, _run_ptoas

    return _compile_launch_cpp, _link_shared_library, _run_ptoas


def build_vmi_stage(source_name: str, name: str, signature: str, call_args: str, dyn_ub: int) -> Path:
    OUT.mkdir(exist_ok=True)
    compile_host, link_shared, run_ptoas = pto_native_build()
    module = load_fixture(source_name, name)
    kernel = getattr(module, name)
    mlir, obj = OUT / f"{name}.mlir", OUT / f"{name}.o"
    mlir.write_text(kernel.compile().mlir_text())
    run_ptoas(mlir, obj, target_arch="a5", backend="vpto", pto_level="level3")
    host_src, host_obj = OUT / f"{name}_host.cpp", OUT / f"{name}_host.o"
    host_src.write_text(f'''#include <stdint.h>
#ifndef AICORE
#define AICORE [aicore]
#endif
extern "C" __global__ AICORE void {name}({signature});
extern "C" void launch_{name}(void *stream, void *a, void *b, void *c) {{
  {name}<<<{GRID}, {dyn_ub}, stream>>>({call_args});
}}
''')
    compile_host(host_src, host_obj, kernel_kind="vector", target_arch="a5", export_macro=f"{name.upper()}_EXPORTS")
    library = OUT / f"lib{name}.so"
    link_shared(host_obj, obj, library, kernel_kind="vector")
    return library


def build_vmi() -> tuple[Path, Path]:
    unpack = build_vmi_stage(
        "unpack_stage_vmi",
        "unpack_stage",
        "__gm__ uint16_t *, __gm__ uint8_t *, __gm__ uint8_t *",
        "(__gm__ uint16_t *)a, (__gm__ uint8_t *)b, (__gm__ uint8_t *)c",
        UNPACK_DYN_UB,
    )
    requant = build_vmi_stage(
        "requant_stage_vmi",
        "requant_stage",
        "__gm__ uint8_t *, __gm__ float *, __gm__ uint16_t *",
        "(__gm__ uint8_t *)a, (__gm__ float *)b, (__gm__ uint16_t *)c",
        REQUANT_DYN_UB,
    )
    return unpack, requant


def stream_ptr() -> int:
    handle = torch.npu.current_stream()._as_parameter_  # noqa: SLF001
    return handle.value if hasattr(handle, "value") else int(handle)


def median_event_us(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    cache = torch.empty(L2_FLUSH_MB * 1024 * 1024 // 4, dtype=torch.int32, device=DEVICE)
    intervals: list[tuple[torch.npu.Event, torch.npu.Event]] = []
    for _ in range(SAMPLES):
        # Queue eviction before the start event.  It cannot enter the timed
        # interval but establishes the same cold-L2 condition on both paths.
        cache.zero_()
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        intervals.append((begin, end))
    torch.npu.synchronize()
    values = sorted(begin.elapsed_time(end) * 1000.0 for begin, end in intervals)
    return values[len(values) // 2]


def msprof_device_us(fn, expected_names: tuple[str, ...]) -> float:
    """Sum device durations using the same FFTS rule as ``do_bench(msprof)``.

    FFTS records both AIC and AIV for a mixed operation.  The authoritative
    value uses the AIC records when present, otherwise AIV, averages across
    launches, then sums all expected kernels in the composed operation.
    """
    import torch_npu.profiler

    def parse(root: Path) -> tuple[float, dict[str, float]]:
        hashes: dict[int, str] = {}
        for path in root.rglob("*hash_dic.slice_*"):
            if path.name.endswith(".done"):
                continue
            for line in path.read_bytes().decode("utf-8", "replace").splitlines():
                key, separator, value = line.partition(":")
                if not separator:
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
            for offset in range(0, len(data) // 64 * 64, 64):
                values = struct.unpack_from("<8q", data, offset)
                name_hash = values[5]
                if name_hash:
                    names[(values[3] >> 32) & 0xFFFF] = hashes.get(name_hash, "")
        records: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
        for path in root.rglob("ffts_profile*"):
            if path.name.endswith(".done") or not path.stat().st_size:
                continue
            data = path.read_bytes()
            for offset in range(0, len(data) // 128 * 128, 128):
                values = struct.unpack_from("<16q", data, offset)
                if not values[1]:
                    continue
                name = names.get((values[0] >> 32) & 0xFFFF, "<unknown>")
                if not any(token in name for token in expected_names):
                    continue
                records[name][1 if values[2] >= (1 << 31) else 0].append(float(values[15] - values[14]))
        by_kernel = {name: sum(aic or aiv) / MSPROF_REPS / 1000.0 for name, (aic, aiv) in records.items()}
        missing = [token for token in expected_names if not any(token in name for name in by_kernel)]
        if missing:
            raise RuntimeError(f"missing FFTS records for {missing}; saw {sorted(by_kernel)}")
        return sum(by_kernel.values()), by_kernel

    old = os.environ.get("ASCEND_WORK_PATH")
    with tempfile.TemporaryDirectory(prefix="rescale_msprof_", dir=OUT) as work:
        os.environ["ASCEND_WORK_PATH"] = work
        schedule = torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0)
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU], schedule=schedule,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                l2_cache=False, data_simplification=False,
            ),
        ) as profiler:
            for _ in range(MSPROF_REPS):
                fn()
            torch.npu.synchronize()
            profiler.step()
        total, by_kernel = parse(Path(profiler.prof_if.prof_path))
    if old is None:
        os.environ.pop("ASCEND_WORK_PATH", None)
    else:
        os.environ["ASCEND_WORK_PATH"] = old
    print("msprof_kernel_us=" + ",".join(f"{name}:{value:.3f}" for name, value in sorted(by_kernel.items())))
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.compile_only:
        build_cce()
        build_vmi()
        print("PASS: full CCE and composed VMI libraries built")
        return

    torch.npu.set_device(DEVICE)
    cce = ctypes.CDLL(str(build_cce()))
    unpack_lib, requant_lib = map(ctypes.CDLL, map(str, build_vmi()))
    cce_fn = cce.launch_rescale_reference
    unpack_fn, requant_fn = unpack_lib.launch_unpack_stage, requant_lib.launch_requant_stage
    cce_fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int, ctypes.c_int]
    unpack_fn.argtypes = requant_fn.argtypes = [ctypes.c_void_p] * 4

    # Packed input scales contain UE8M0 exponents.  127 represents unit scale
    # and keeps the correctness check away from special-value encoding paths.
    source = (torch.randn((ROWS, WIDTH), dtype=torch.float32, device=DEVICE) * 0.25).to(torch.float8_e4m3fn)
    input_sf = torch.full((ROWS, INPUT_SF_COLS), 127, dtype=torch.uint8, device=DEVICE)
    cce_out = torch.empty_like(source)
    cce_sf = torch.empty((OUTPUT_SF_ROWS, WIDTH), dtype=torch.float32, device=DEVICE)
    vmi_tmp = torch.empty((ROWS, WIDTH), dtype=torch.bfloat16, device=DEVICE)
    vmi_out = torch.empty_like(source)
    vmi_sf = torch.empty_like(cce_sf)
    # The final 896-row strip uses the same static 1024-row VMI kernel as the
    # production composition.  Retain its device pad/copy behavior so the
    # standalone schedule remains byte-for-byte equivalent in launch geometry.
    tail_rows = ROWS % STRIP_ROWS
    full_rows = ROWS - tail_rows
    tail_source = torch.empty((STRIP_ROWS, WIDTH), dtype=source.dtype, device=DEVICE)
    tail_input_sf = torch.empty((STRIP_ROWS, INPUT_SF_COLS), dtype=input_sf.dtype, device=DEVICE)
    tail_tmp = torch.empty((STRIP_ROWS, WIDTH), dtype=vmi_tmp.dtype, device=DEVICE)

    def cce_run() -> None:
        cce_fn(
            ctypes.c_void_p(stream_ptr()), ctypes.c_void_p(cce_out.data_ptr()), ctypes.c_void_p(cce_sf.data_ptr()),
            ctypes.c_void_p(source.data_ptr()), ctypes.c_void_p(input_sf.data_ptr()), ROWS, WIDTH,
        )

    def vmi_run() -> None:
        stream = ctypes.c_void_p(stream_ptr())
        for row0 in range(0, full_rows, STRIP_ROWS):
            unpack_fn(
                stream, ctypes.c_void_p(vmi_tmp[row0 : row0 + STRIP_ROWS].data_ptr()),
                ctypes.c_void_p(source[row0 : row0 + STRIP_ROWS].data_ptr()),
                ctypes.c_void_p(input_sf[row0 : row0 + STRIP_ROWS].data_ptr()),
            )
        tail_source.zero_()
        tail_input_sf.zero_()
        tail_source[:tail_rows].copy_(source[full_rows:])
        tail_input_sf[:tail_rows].copy_(input_sf[full_rows:])
        unpack_fn(stream, ctypes.c_void_p(tail_tmp.data_ptr()), ctypes.c_void_p(tail_source.data_ptr()), ctypes.c_void_p(tail_input_sf.data_ptr()))
        vmi_tmp[full_rows:].copy_(tail_tmp[:tail_rows])
        requant_fn(stream, ctypes.c_void_p(vmi_out.data_ptr()), ctypes.c_void_p(vmi_sf.data_ptr()), ctypes.c_void_p(vmi_tmp.data_ptr()))

    cce_run()
    vmi_run()
    torch.npu.synchronize()
    out_match = torch.isclose(cce_out.float(), vmi_out.float(), rtol=0.15, atol=0.125)
    sf_match = torch.isclose(cce_sf, vmi_sf, rtol=0.05, atol=0.0)
    out_bad, sf_bad = int((~out_match).sum().item()), int((~sf_match).sum().item())
    out_head_bad = int((~out_match[:full_rows]).sum().item())
    sf_head_rows = full_rows // 32
    sf_head_bad = int((~sf_match[:sf_head_rows]).sum().item())
    # The retained padded final strip currently diverges on its valid rows.
    # Do not discard the production schedule to conceal this: report the
    # mismatch as a VMI functional blocker and still measure the full path.
    correctness = "PASS" if not out_bad and not sf_bad else "VMI_TAIL_MISMATCH"

    cce_event, vmi_event = median_event_us(cce_run), median_event_us(vmi_run)
    cce_us = msprof_device_us(cce_run, ("rescale_cce_kernel",))
    vmi_us = msprof_device_us(vmi_run, ("unpack_stage", "requant_stage"))
    print(f"device={DEVICE} shape={ROWS}x{WIDTH} grid={GRID} unpack_launches={(ROWS + STRIP_ROWS - 1) // STRIP_ROWS} strip_rows={STRIP_ROWS}")
    print(f"cce_dyn_ub={CCE_DYN_UB} unpack_dyn_ub={UNPACK_DYN_UB} requant_dyn_ub={REQUANT_DYN_UB}")
    print(f"correctness={correctness} output_mismatched={out_bad} output_head_mismatched={out_head_bad} scale_mismatched={sf_bad} scale_head_mismatched={sf_head_bad} output_extent=equal")
    print(f"event_l2_flush_mb={L2_FLUSH_MB} samples={SAMPLES} CCE_us={cce_event:.3f} VMI_us={vmi_event:.3f} CCE_over_VMI={cce_event / vmi_event:.4f}")
    print(f"msprof_device_reps={MSPROF_REPS} CCE_us={cce_us:.3f} VMI_us={vmi_us:.3f} CCE_over_VMI={cce_us / vmi_us:.4f}")


if __name__ == "__main__":
    main()
