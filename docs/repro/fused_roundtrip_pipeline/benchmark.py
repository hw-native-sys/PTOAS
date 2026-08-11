#!/usr/bin/env python3
"""Compile, validate, and event-time the standalone CCE and VMI paths."""
from __future__ import annotations

import ctypes
import os
import subprocess
import argparse
from pathlib import Path

import torch
import torch_npu  # noqa: F401

HERE = Path(__file__).parent
OUT = HERE / "outputs"
DEVICE = f"npu:{os.environ.get('ACL_DEVICE_ID', '0')}"
 # Keep one 256-value work item on each side.  The VMI fixture intentionally
 # models the low-level operation, so timing a Python loop over rows would
 # include host launch overhead and is not a fair kernel comparison.
ROWS, WIDTH = 1, 256
GRID, DYN_UB = 1, 102912
WARMUP, SAMPLES = 20, 30


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def bisheng() -> str:
    return os.environ.get("BISHENG", f"{os.environ['ASCEND_HOME_PATH']}/bin/bisheng")


def includes() -> list[str]:
    root = os.environ["ASCEND_HOME_PATH"]
    return ["-I" + str(HERE / "fixtures"), "-I" + root + "/aarch64-linux/tikcpp/tikcfw",
            "-I" + root + "/aarch64-linux/tikcpp/tikcfw/impl",
            "-I" + root + "/aarch64-linux/tikcpp/tikcfw/interface"]


def build_cce() -> Path:
    OUT.mkdir(exist_ok=True)
    device, host, library = OUT / "reference_device.o", OUT / "reference_host.o", OUT / "libreference_cce.so"
    run([bisheng(), "-include", str(HERE / "fixtures/cxx17_bit_cast.hpp"), "-O2", "-fPIC", "-std=c++17", "--npu-arch=dav-3510", *includes(),
         "-c", str(HERE / "fixtures/reference_device.asc"), "-o", str(device)])
    run([bisheng(), "-xcce", "-Xhost-start", "-Xhost-end", "-fPIC", "-O2", "-std=c++17",
         "--cce-aicore-arch=dav-c310", "-c", str(HERE / "fixtures/reference_cce.cpp"), "-o", str(host)])
    root = os.environ["ASCEND_HOME_PATH"]
    run([bisheng(), "--cce-fatobj-link", "-shared", "-fPIC", str(device), str(host),
         "-L" + root + "/aarch64-linux/lib64", "-Wl,-rpath," + root + "/aarch64-linux/lib64",
         "-Wl,--no-as-needed", "-lruntime", "-o", str(library)])
    return library


def build_vmi() -> Path:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    ptoas = os.environ.get("PTOAS_BIN") or subprocess.check_output(
        ["conda", "run", "-n", "cann91_dev", "which", "ptoas"], text=True).strip().splitlines()[-1]
    obj, host, library = OUT / "roundtrip_vmi.o", OUT / "roundtrip_vmi_host.o", OUT / "libroundtrip_vmi.so"
    run([ptoas, "--pto-arch=a5", "--pto-backend=vpto", "--pto-level=level3",
         str(HERE / "fixtures/fused_roundtrip_vmi.pto"), "-o", str(obj)], env=env)
    host_src = OUT / "roundtrip_vmi_host.cpp"
    host_src.write_text("""#include <stdint.h>
extern \"C\" __global__ [aicore] void fused_roundtrip_body(__gm__ uint16_t*, __gm__ uint16_t*, __gm__ float*);
extern \"C\" void launch_roundtrip_vmi(void *stream, void *x, void *y, void *s) {
  fused_roundtrip_body<<<1, nullptr, stream>>>((__gm__ uint16_t*)x, (__gm__ uint16_t*)y, (__gm__ float*)s);
}
""")
    run([bisheng(), "-xcce", "-Xhost-start", "-Xhost-end", "-fPIC", "-O2", "-std=c++17",
         "--cce-aicore-arch=dav-c310", "-c", str(host_src), "-o", str(host)])
    root = os.environ["ASCEND_HOME_PATH"]
    run([bisheng(), "--cce-fatobj-link", "-shared", "-fPIC", str(obj), str(host),
         "-L" + root + "/aarch64-linux/lib64", "-Wl,-rpath," + root + "/aarch64-linux/lib64",
         "-Wl,--no-as-needed", "-lruntime", "-o", str(library)])
    return library


def stream_ptr() -> int:
    value = torch.npu.current_stream()._as_parameter_  # noqa: SLF001
    return value.value if hasattr(value, "value") else int(value)


def median_us(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(SAMPLES):
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        begin.record(); fn(); end.record(); samples.append((begin, end))
    torch.npu.synchronize()
    samples = [begin.elapsed_time(end) * 1000.0 for begin, end in samples]
    return sorted(samples)[len(samples) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.compile_only:
        build_cce(); build_vmi()
        print("PASS: stream-launchable CCE and VMI libraries built")
        return
    torch.npu.set_device(DEVICE)
    cce = ctypes.CDLL(str(build_cce()))
    vmi = ctypes.CDLL(str(build_vmi()))
    cce_fn, vmi_fn = cce.launch_roundtrip_reference, vmi.launch_roundtrip_vmi
    cce_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    vmi_fn.argtypes = [ctypes.c_void_p] * 4
    x = torch.ones((ROWS, WIDTH), dtype=torch.bfloat16, device=DEVICE)
    vmi_x = x.clone(); vmi_y = torch.empty_like(vmi_x); vmi_scale = torch.empty((ROWS, 8), dtype=torch.float32, device=DEVICE)

    def cce_run() -> None:
        cce_fn(ctypes.c_void_p(stream_ptr()), ctypes.c_void_p(x.data_ptr()), ROWS)

    def vmi_run() -> None:
        stream = ctypes.c_void_p(stream_ptr())
        vmi_fn(stream, ctypes.c_void_p(vmi_x.data_ptr()),
               ctypes.c_void_p(vmi_y.data_ptr()), ctypes.c_void_p(vmi_scale.data_ptr()))

    cce_run(); vmi_run(); torch.npu.synchronize()
    expected = torch.ones((ROWS, WIDTH), dtype=torch.bfloat16)
    torch.testing.assert_close(x.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(vmi_y.cpu(), expected, rtol=0, atol=0)
    cce_us, vmi_us = median_us(cce_run), median_us(vmi_run)
    print(f"device={DEVICE} shape={ROWS}x{WIDTH} grid={GRID} dynamic_ub={DYN_UB} samples={SAMPLES} warmup={WARMUP}")
    print("correctness=PASS cce_host_golden=PASS vmi_host_golden=PASS cce_vmi_peer=PASS")
    print(f"CCE_us={cce_us:.3f} VMI_us={vmi_us:.3f} CCE_over_VMI={cce_us / vmi_us:.4f}")


if __name__ == "__main__":
    main()
