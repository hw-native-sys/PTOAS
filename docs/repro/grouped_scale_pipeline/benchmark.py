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
ROWS, WIDTH, GROUPS = 128, 7168, 8
GRID, DYN_UB, WARMUP, SAMPLES = 72, 204800, 20, 30


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
             "--npu-arch=dav-3510", *includes(), "-c", str(HERE / "fixtures/reference_device.asc"), "-o", str(body)])
    command([bisheng(), "-xcce", "-Xhost-start", "-Xhost-end", "-fPIC", "-O2", "-std=c++17",
             "--cce-aicore-arch=dav-c310", "-c", str(HERE / "fixtures/reference_cce.cpp"), "-o", str(host)])
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
extern "C" __global__ [aicore] void grouped_scale_body(__gm__ uint16_t*, __gm__ uint8_t*, __gm__ uint16_t*);
extern "C" void launch_grouped_vmi(void *stream, void *x, void *y, void *s) {
  grouped_scale_body<<<1, nullptr, stream>>>((__gm__ uint16_t*)x, (__gm__ uint8_t*)y, (__gm__ uint16_t*)s);
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
    for _ in range(WARMUP): fn()
    torch.npu.synchronize(); values = []
    for _ in range(SAMPLES):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record(); fn(); end.record(); end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0)
    return sorted(values)[len(values) // 2]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.compile_only:
        build_cce(); build_vmi(); print("PASS: stream-launchable CCE and VMI libraries built"); return
    torch.npu.set_device(DEVICE)
    cce, vmi = ctypes.CDLL(str(build_cce())), ctypes.CDLL(str(build_vmi()))
    cce_fn, vmi_fn = cce.launch_grouped_reference, vmi.launch_grouped_vmi
    cce_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    vmi_fn.argtypes = [ctypes.c_void_p] * 4
    x = torch.ones((ROWS, WIDTH), dtype=torch.bfloat16, device=DEVICE)
    cce_q = torch.empty((ROWS, WIDTH), dtype=torch.uint8, device=DEVICE)
    cce_s = torch.empty((4, WIDTH), dtype=torch.float32, device=DEVICE)
    vmi_q = torch.empty_like(cce_q); vmi_s = torch.empty((ROWS, GROUPS), dtype=torch.bfloat16, device=DEVICE)

    def cce_run() -> None:
        cce_fn(ctypes.c_void_p(stream_ptr()), ctypes.c_void_p(cce_q.data_ptr()), ctypes.c_void_p(cce_s.data_ptr()),
               ctypes.c_void_p(x.data_ptr()), ROWS, WIDTH)

    # The public VMI source deliberately retains one 256-value work item.  Tile
    # it over the exact CCE input/output extent so timing includes the current
    # launch-level blocker rather than silently comparing different amounts of data.
    def vmi_run() -> None:
        stream = ctypes.c_void_p(stream_ptr())
        for offset in range(0, ROWS * WIDTH, 256):
            row, group = divmod(offset // 256, WIDTH // 256)
            vmi_fn(stream, ctypes.c_void_p(x.data_ptr() + offset * 2), ctypes.c_void_p(vmi_q.data_ptr() + offset),
                   ctypes.c_void_p(vmi_s.data_ptr() + (row * GROUPS + (group % GROUPS)) * 2))

    cce_run(); vmi_run(); torch.npu.synchronize()
    torch.testing.assert_close(cce_q.cpu(), torch.full((ROWS, WIDTH), 126, dtype=torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(vmi_q.cpu(), torch.full((ROWS, WIDTH), 0x38, dtype=torch.uint8), rtol=0, atol=0)
    if not bool(torch.isfinite(cce_s.cpu()).all()) or not bool(torch.all(vmi_s.cpu() == 1)):
        raise RuntimeError("non-finite CCE scale or unexpected VMI scale")
    cce_us, vmi_us = median_us(cce_run), median_us(vmi_run)
    print(f"device={DEVICE} shape={ROWS}x{WIDTH} grid={GRID} dynamic_ub={DYN_UB} samples={SAMPLES} warmup={WARMUP}")
    print("correctness=PASS cce_host_golden=PASS vmi_host_golden=PASS output_extent=equal")
    print(f"CCE_us={cce_us:.3f} VMI_us={vmi_us:.3f} CCE_over_VMI={cce_us / vmi_us:.4f}")


if __name__ == "__main__": main()
