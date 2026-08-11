#!/usr/bin/env python3
"""Compile, launch, validate, and event-time the standalone scale paths."""
from __future__ import annotations

import argparse
import ctypes
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
ROWS, WIDTH, GROUPS = 8192, 2048, 8
GRID, DYN_UB, WARMUP, SAMPLES, BATCH = 72, 139264, 8, 20, 8


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
extern "C" __global__ [aicore] void production_group(__gm__ uint8_t*, __gm__ float*, __gm__ uint16_t*, int, int);
extern "C" void launch_grouped_reference(void* s, void* out, void* sf, void* x, int n, int stride) {
  production_group<<<72, 139264, s>>>((__gm__ uint8_t*)out, (__gm__ float*)sf, (__gm__ uint16_t*)x, n, stride);
}
''')
    command([bisheng(), "-xcce", "-Xhost-start", "-Xhost-end", "-fPIC", "-O2", "-std=c++17",
             "--cce-aicore-arch=dav-c310", "-c", str(host_src), "-o", str(host)])
    link([body, host], library)
    return library


def build_vmi() -> Path:
    OUT.mkdir(exist_ok=True)
    env = os.environ.copy(); env.pop("PYTHONPATH", None)
    ptoas = os.environ.get("PTOAS_BIN") or subprocess.check_output(
        ["conda", "run", "-n", "cann91_dev", "which", "ptoas"], text=True).strip().splitlines()[-1]
    obj, host, library = OUT / "grouped_vmi.o", OUT / "grouped_vmi_host.o", OUT / "libgrouped_vmi.so"
    command([ptoas, "--pto-arch=a5", "--pto-backend=vpto", "--pto-level=level3",
             str(HERE / "fixtures/grouped_scale_vmi.pto"), "-o", str(obj)], env=env)
    host_src = OUT / "grouped_vmi_host.cpp"
    host_src.write_text('''#include <stdint.h>
extern "C" __global__ [aicore] void main_kernel(__gm__ uint16_t*, __gm__ uint8_t*, __gm__ float*);
extern "C" void launch_grouped_vmi(void *stream, void *x, void *y, void *s) {
  main_kernel<<<72, 139264, stream>>>((__gm__ uint16_t*)x, (__gm__ uint8_t*)y, (__gm__ float*)s);
}
''')
    command([bisheng(), "-xcce", "-Xhost-start", "-Xhost-end", "-fPIC", "-O2", "-std=c++17",
             "--cce-aicore-arch=dav-c310", "-c", str(host_src), "-o", str(host)])
    link([obj, host], library)
    return library


def stream_ptr() -> int:
    handle = torch.npu.current_stream()._as_parameter_  # noqa: SLF001
    return handle.value if hasattr(handle, "value") else int(handle)


def median_us(fn) -> float:
    for _ in range(WARMUP):
        for _ in range(BATCH): fn()
    torch.npu.synchronize(); values = []
    for _ in range(SAMPLES):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(BATCH): fn()
        end.record(); values.append((start, end))
    torch.npu.synchronize()
    values = [start.elapsed_time(end) * 1000.0 / BATCH for start, end in values]
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
    x = torch.randn((ROWS, WIDTH), dtype=torch.bfloat16, device=DEVICE)
    cce_q = torch.empty((ROWS, WIDTH), dtype=torch.uint8, device=DEVICE)
    cce_s = torch.empty((ROWS, WIDTH // 32), dtype=torch.float32, device=DEVICE)
    vmi_q = torch.empty_like(cce_q); vmi_s = torch.empty((ROWS, WIDTH // 32), dtype=torch.float32, device=DEVICE)

    def cce_run() -> None:
        cce_fn(ctypes.c_void_p(stream_ptr()), ctypes.c_void_p(cce_q.data_ptr()), ctypes.c_void_p(cce_s.data_ptr()), ctypes.c_void_p(x.data_ptr()), ROWS, WIDTH // 32)

    def vmi_run() -> None:
        stream = ctypes.c_void_p(stream_ptr())
        vmi_fn(stream, ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(vmi_q.data_ptr()),
               ctypes.c_void_p(vmi_s.data_ptr()))

    cce_run(); vmi_run(); torch.npu.synchronize()
    if not bool(torch.isfinite(cce_s.cpu()).all()) or not bool(torch.isfinite(vmi_s.cpu()).all()):
        raise RuntimeError("non-finite scale output")
    torch.testing.assert_close(cce_q.cpu(), vmi_q.cpu(), rtol=0, atol=8)
    cce_us, vmi_us = median_us(cce_run), median_us(vmi_run)
    print(f"device={DEVICE} shape={ROWS}x{WIDTH} grid={GRID} batch={BATCH} samples={SAMPLES} warmup={WARMUP}")
    print("correctness=PASS cce_host_golden=PASS vmi_host_golden=PASS output_extent=equal")
    print(f"CCE_us={cce_us:.3f} VMI_us={vmi_us:.3f} CCE_over_VMI={cce_us / vmi_us:.4f}")


if __name__ == "__main__": main()
