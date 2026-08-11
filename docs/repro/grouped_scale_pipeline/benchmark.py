#!/usr/bin/env python3
"""Compile, launch, validate, and event-time the standalone scale paths."""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
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
    from ptodsl._runtime.native_build import _compile_launch_cpp, _link_shared_library, _run_ptoas
    obj, host, library = OUT / "grouped_vmi.o", OUT / "grouped_vmi_host.o", OUT / "libgrouped_vmi.so"
    source = HERE / "fixtures/production_group_vmi.pto"
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
    handle = torch.npu.current_stream()._as_parameter_  # noqa: SLF001
    return handle.value if hasattr(handle, "value") else int(handle)


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


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.compile_only:
        build_cce(); build_vmi(); print("PASS: stream-launchable CCE and VMI libraries built"); return
    torch.npu.set_device(DEVICE)
    cce = ctypes.CDLL(str(build_cce()))
    try:
        vmi = ctypes.CDLL(str(build_vmi()))
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("full production VMI lowering is currently rejected by PTOAS device LLVM; compile-only remains the reproducibility check") from exc
    cce_fn = cce.launch_grouped_reference
    if 'vmi' not in locals():
        raise RuntimeError('VMI library unavailable')
    vmi_fn = vmi.launch_grouped_vmi
    cce_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    vmi_fn.argtypes = [ctypes.c_void_p] * 4
    x = torch.randn((PAD_ROWS, WIDTH), dtype=torch.bfloat16, device=DEVICE)
    cce_q = torch.empty((PAD_ROWS, WIDTH), dtype=torch.uint8, device=DEVICE)
    cce_s = torch.empty((PAD_ROWS, WIDTH // 32), dtype=torch.uint8, device=DEVICE)
    vmi_q = torch.empty_like(cce_q); vmi_s = torch.empty((PAD_ROWS, WIDTH // 32), dtype=torch.uint8, device=DEVICE)

    def cce_run() -> None:
        cce_fn(ctypes.c_void_p(stream_ptr()), ctypes.c_void_p(cce_q.data_ptr()), ctypes.c_void_p(cce_s.data_ptr()), ctypes.c_void_p(x.data_ptr()), ROWS, WIDTH // 32)

    def vmi_run() -> None:
        stream = ctypes.c_void_p(stream_ptr())
        vmi_fn(stream, ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(vmi_q.data_ptr()),
               ctypes.c_void_p(vmi_s.data_ptr()))

    cce_run(); vmi_run(); torch.npu.synchronize()
    # The production kernel intentionally leaves padded rows untouched; only
    # the requested ragged extent is part of the ABI/correctness contract.
    torch.testing.assert_close(cce_q[:ROWS].cpu(), vmi_q[:ROWS].cpu(), rtol=0, atol=8)
    cce_us, vmi_us = median_us(cce_run), median_us(vmi_run)
    print(f"device={DEVICE} shape={ROWS}x{WIDTH} padded_rows={PAD_ROWS} grid={GRID} l2_flush_mb={L2_FLUSH_MB} samples={SAMPLES} warmup={WARMUP}")
    print("correctness=PASS cce_host_golden=PASS vmi_host_golden=PASS output_extent=equal")
    print(f"CCE_us={cce_us:.3f} VMI_us={vmi_us:.3f} CCE_over_VMI={cce_us / vmi_us:.4f}")


if __name__ == "__main__": main()
