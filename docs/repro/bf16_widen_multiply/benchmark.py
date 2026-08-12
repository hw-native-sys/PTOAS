#!/usr/bin/env python3
"""Build, launch, validate, and time direct CCE and PTODSL VMI kernels."""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import os
import shutil
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import torch
import torch_npu  # noqa: F401

HERE = Path(__file__).parent
OUT = HERE / "outputs"
ELEMENTS = 131072
ELEMENTS_PER_CORE = 2048
GRID = ELEMENTS // ELEMENTS_PER_CORE
UB_BYTES = ELEMENTS_PER_CORE * 8
DEVICE = f"npu:{os.environ.get('ACL_DEVICE_ID', '0')}"
WARMUP, SAMPLES = 8, 30


def command(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


def bisheng() -> str:
    return os.environ.get("BISHENG", f"{os.environ['ASCEND_HOME_PATH']}/bin/bisheng")


def includes() -> list[str]:
    root = os.environ["ASCEND_HOME_PATH"]
    return ["-I" + str(HERE / "fixtures"), "-I" + root + "/include", "-I" + root + "/aarch64-linux/tikcpp/tikcfw", "-I" + root + "/aarch64-linux/tikcpp/tikcfw/impl", "-I" + root + "/aarch64-linux/tikcpp/tikcfw/interface"]


def link(objects: list[Path], library: Path) -> Path:
    root = os.environ["ASCEND_HOME_PATH"]
    command([bisheng(), "--cce-fatobj-link", "-shared", "-fPIC", *map(str, objects), "-L" + root + "/aarch64-linux/lib64", "-Wl,-rpath," + root + "/aarch64-linux/lib64", "-Wl,--no-as-needed", "-lruntime", "-o", str(library)])
    return library


def build_cce() -> Path:
    OUT.mkdir(exist_ok=True)
    body, host = OUT / "cce.o", OUT / "cce_host.o"
    command([bisheng(), "-O2", "-fPIC", "-std=c++17", "--npu-arch=dav-3510", *includes(), "-c", str(HERE / "fixtures/bf16_widen_multiply_cce.asc"), "-o", str(body)])
    source = OUT / "cce_host.cpp"
    source.write_text(f'''#include <stdint.h>
extern "C" __global__ [aicore] void bf16_widen_multiply_cce(__gm__ uint16_t*, __gm__ uint16_t*, __gm__ float*);
extern "C" void launch_cce(void* stream, void* a, void* b, void* out) {{
  bf16_widen_multiply_cce<<<{GRID}, {UB_BYTES}, stream>>>((__gm__ uint16_t*)a, (__gm__ uint16_t*)b, (__gm__ float*)out);
}}
''')
    command([bisheng(), "-xcce", "-Xhost-start", "-Xhost-end", "-fPIC", "-O2", "-std=c++17", "--cce-aicore-arch=dav-c310", "-c", str(source), "-o", str(host)])
    return link([body, host], OUT / "libbf16_cce.so")


def build_vmi() -> Path:
    OUT.mkdir(exist_ok=True)
    if "PTOAS_BIN" not in os.environ:
        candidates = [Path(sys.executable).parent / "ptoas", Path(os.environ.get("CONDA_PREFIX", "")) / "bin/ptoas"]
        candidate = next((path for path in candidates if path.is_file()), None)
        os.environ["PTOAS_BIN"] = str(candidate) if candidate else (shutil.which("ptoas") or "")
    if not os.environ["PTOAS_BIN"]:
        raise RuntimeError("ptoas is not on PATH; set PTOAS_BIN to the pinned PTOAS executable")
    sys.path.insert(0, str(HERE.parents[2] / "ptodsl"))
    import ptoas
    import site
    for base in site.getsitepackages():
        bindings = str(Path(base) / "ptoas")
        if bindings not in ptoas.__path__: ptoas.__path__.append(bindings)
    from ptodsl._runtime.native_build import _compile_launch_cpp, _link_shared_library, _run_ptoas
    source = HERE / "fixtures/bf16_widen_multiply_vmi.py"
    loader = SourceFileLoader("bf16_widen_multiply_vmi", str(source))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec); loader.exec_module(module)
    mlir = OUT / "vmi.mlir"; mlir.write_text(module.bf16_widen_multiply_vmi.compile().mlir_text())
    body, host = OUT / "vmi.o", OUT / "vmi_host.o"
    _run_ptoas(mlir, body, target_arch="a5", backend="vpto", pto_level="level3")
    host_source = OUT / "vmi_host.cpp"
    host_source.write_text(f'''#include <stdint.h>
extern "C" __global__ [aicore] void bf16_widen_multiply_vmi(__gm__ uint16_t*, __gm__ uint16_t*, __gm__ float*);
extern "C" void launch_vmi(void* stream, void* a, void* b, void* out) {{
  bf16_widen_multiply_vmi<<<{GRID}, {UB_BYTES}, stream>>>((__gm__ uint16_t*)a, (__gm__ uint16_t*)b, (__gm__ float*)out);
}}
''')
    _compile_launch_cpp(host_source, host, kernel_kind="vector", target_arch="a5", export_macro="BF16_VMI_EXPORTS")
    _link_shared_library(host, body, OUT / "libbf16_vmi.so", kernel_kind="vector")
    return OUT / "libbf16_vmi.so"


def stream_ptr() -> int:
    return int(torch_npu._C._npu_getCurrentRawStream(torch.npu.current_device()))


def median_us(fn) -> float:
    for _ in range(WARMUP): fn()
    torch.npu.synchronize(); samples = []
    for _ in range(SAMPLES):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record(); fn(); end.record(); samples.append((start, end))
    torch.npu.synchronize()
    return sorted(start.elapsed_time(end) * 1000.0 for start, end in samples)[len(samples) // 2]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.compile_only:
        build_cce(); build_vmi(); print("PASS: direct CCE and standalone PTODSL VMI libraries built"); return
    torch.npu.set_device(DEVICE)
    cce, vmi = ctypes.CDLL(str(build_cce())), ctypes.CDLL(str(build_vmi()))
    for fn in (cce.launch_cce, vmi.launch_vmi): fn.argtypes = [ctypes.c_void_p] * 4
    a = torch.randn(ELEMENTS, dtype=torch.bfloat16, device=DEVICE); b = torch.randn_like(a)
    cce_out = torch.empty(ELEMENTS, dtype=torch.float32, device=DEVICE); vmi_out = torch.empty_like(cce_out)
    def cce_run(): cce.launch_cce(ctypes.c_void_p(stream_ptr()), ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(b.data_ptr()), ctypes.c_void_p(cce_out.data_ptr()))
    def vmi_run(): vmi.launch_vmi(ctypes.c_void_p(stream_ptr()), ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(b.data_ptr()), ctypes.c_void_p(vmi_out.data_ptr()))
    cce_run(); vmi_run(); torch.npu.synchronize()
    reference = a.float() * b.float()
    torch.testing.assert_close(cce_out, reference, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(vmi_out, reference, rtol=1e-3, atol=1e-3)
    cce_us, vmi_us = median_us(cce_run), median_us(vmi_run)
    print(f"correctness=PASS elements={ELEMENTS} grid={GRID} CCE_us={cce_us:.3f} VMI_us={vmi_us:.3f} CCE_over_VMI={cce_us / vmi_us:.4f}")


if __name__ == "__main__": main()
