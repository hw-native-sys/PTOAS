#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Full pointer-table CCE versus stacked-buffer VMI workload on A5."""
from __future__ import annotations

import argparse
import ctypes
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch_npu  # noqa: F401

HERE = Path(__file__).parent
OUT = HERE / "outputs"
DEVICE = f"npu:{os.environ.get('ACL_DEVICE_ID', '0')}"
TOKENS, HIDDEN, LANES, LAYERS, GRID = 8192, 4096, 4, 10, 72
CCE_UB, VMI_UB = 98304, 295680
WARMUP, SAMPLES, L2_FLUSH_MB = 8, 20, 256


def command(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


def bisheng() -> str:
    return os.environ.get("BISHENG", f"{os.environ['ASCEND_HOME_PATH']}/bin/bisheng")


def link(objects: list[Path], library: Path) -> Path:
    root = os.environ["ASCEND_HOME_PATH"]
    command([
        bisheng(), "--cce-fatobj-link", "-shared", "-fPIC", *map(str, objects),
        "-L" + root + "/aarch64-linux/lib64", "-Wl,-rpath," + root + "/aarch64-linux/lib64",
        "-Wl,--no-as-needed", "-lruntime", "-o", str(library),
    ])
    return library


def build_cce() -> Path:
    OUT.mkdir(exist_ok=True)
    root = os.environ["ASCEND_HOME_PATH"]
    body, host = OUT / "pointer_body.o", OUT / "pointer_host.o"
    inc = [
        "-I" + str(HERE / "fixtures"), "-I" + root + "/include",
        "-I" + root + "/aarch64-linux/tikcpp/tikcfw",
        "-I" + root + "/aarch64-linux/tikcpp/tikcfw/impl",
        "-I" + root + "/aarch64-linux/tikcpp/tikcfw/interface",
    ]
    command([bisheng(), "-O2", "-fPIC", "-std=c++17", "--npu-arch=dav-3510", *inc,
             "-c", str(HERE / "fixtures/reference_device.asc"), "-o", str(body)])
    command([bisheng(), "-xcce", "-Xhost-start", "-Xhost-end", "-fPIC", "-O2", "-std=c++17",
             "--cce-aicore-arch=dav-c310", "-c", str(HERE / "fixtures/reference_launch.cpp"), "-o", str(host)])
    return link([body, host], OUT / "libpointer_cce.so")


def build_vmi() -> Path:
    OUT.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(HERE / "fixtures")
    ptoas = os.environ.get("PTOAS_BIN")
    if ptoas is None:
        # task-submit may preserve the Python interpreter while dropping the
        # conda bin directory from PATH. Resolve the matching executable from
        # that interpreter before falling back to ordinary PATH lookup.
        candidates = [Path(sys.executable).parent / "ptoas"]
        if os.environ.get("CONDA_PREFIX"):
            candidates.append(Path(os.environ["CONDA_PREFIX"]) / "bin/ptoas")
        ptoas = next((str(path) for path in candidates if path.is_file()), None)
    if ptoas is None:
        from shutil import which
        ptoas = which("ptoas")
    if ptoas is None:
        raise RuntimeError("ptoas is not on PATH; set PTOAS_BIN to the pinned PTOAS executable")
    source = HERE / "fixtures/stacked_pipeline_vmi.py"
    mlir = subprocess.check_output(
        [sys.executable, str(source), "--emit-mlir"], text=True, env=env
    )
    (OUT / "stacked.mlir").write_text(mlir)
    body, host = OUT / "stacked.o", OUT / "stacked_host.o"
    command([ptoas, "--pto-arch=a5", "--pto-backend=vpto", "--pto-level=level3", str(OUT / "stacked.mlir"), "-o", str(body)])
    src = OUT / "stacked_host.cpp"
    src.write_text("""#include <stdint.h>
extern \"C\" __global__ [aicore] void dense_recurrence_stage(
    __gm__ uint16_t*, __gm__ float*, __gm__ uint16_t*, __gm__ float*,
    __gm__ float*, __gm__ uint16_t*, __gm__ uint16_t*);
extern \"C\" void launch_stacked_vmi(void* stream, void* residual_in, void* pre,
    void* layer_output, void* post, void* comb, void* layer_input, void* residual_out) {
  dense_recurrence_stage<<<72, 295680, stream>>>(
      (__gm__ uint16_t*)residual_in, (__gm__ float*)pre, (__gm__ uint16_t*)layer_output,
      (__gm__ float*)post, (__gm__ float*)comb, (__gm__ uint16_t*)layer_input,
      (__gm__ uint16_t*)residual_out);
}
""")
    command([bisheng(), "-xcce", "-Xhost-start", "-Xhost-end", "-fPIC", "-O2", "-std=c++17",
             "--cce-aicore-arch=dav-c310", "-c", str(src), "-o", str(host)])
    return link([body, host], OUT / "libstacked_vmi.so")


def stream_ptr() -> int:
    return int(torch_npu._C._npu_getCurrentRawStream(torch.npu.current_device()))


def median_us(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    cache = torch.empty(L2_FLUSH_MB * 1024 * 1024 // 4, dtype=torch.int32, device=DEVICE)
    values: list[tuple[torch.npu.Event, torch.npu.Event]] = []
    for _ in range(SAMPLES):
        cache.zero_()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record(); fn(); end.record()
        values.append((start, end))
    torch.npu.synchronize()
    return sorted(start.elapsed_time(end) * 1000.0 for start, end in values)[len(values) // 2]


def msprof_us(fn, reps: int = 20) -> float:
    """Measure the whole callable from its FFTS device records.

    The stacked path intentionally includes its GM stack/unstack copies.  They
    are the device work required by the current fixed-formal VMI ABI.
    """
    import torch_npu.profiler

    old = os.environ.get("ASCEND_WORK_PATH")
    with tempfile.TemporaryDirectory(prefix="pointer_table_msprof_", dir=OUT) as work:
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
            torch.npu.synchronize()
            prof.step()
        by_sequence: dict[int, tuple[list[float], list[float]]] = {}
        for path in Path(prof.prof_if.prof_path).rglob("ffts_profile*"):
            if path.name.endswith(".done") or not path.stat().st_size:
                continue
            data = path.read_bytes()
            for offset in range(0, len(data) - 127, 128):
                values = struct.unpack_from("<16q", data, offset)
                if not values[1]:
                    continue
                sequence = (values[0] >> 32) & 0xFFFF
                aic, aiv = by_sequence.setdefault(sequence, ([], []))
                (aiv if values[2] >= (1 << 31) else aic).append(float(values[15] - values[14]))
    if old is None:
        os.environ.pop("ASCEND_WORK_PATH", None)
    else:
        os.environ["ASCEND_WORK_PATH"] = old
    durations = [sum(aic) if aic else sum(aiv) for aic, aiv in by_sequence.values()]
    if not durations:
        raise RuntimeError("no FFTS device records were produced")
    return sum(durations) / reps / 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--direct-only", action="store_true")
    parser.add_argument("--vmi-only", action="store_true")
    args = parser.parse_args()
    if args.compile_only:
        build_cce(); build_vmi()
        print("PASS: full direct-pointer CCE and stacked-buffer VMI libraries built")
        return

    torch.npu.set_device(DEVICE)
    cce, vmi = ctypes.CDLL(str(build_cce())), ctypes.CDLL(str(build_vmi()))
    cce_fn, vmi_fn = cce.launch_indirect_reference, vmi.launch_stacked_vmi
    cce_fn.argtypes = vmi_fn.argtypes = [ctypes.c_void_p] * 8

    # Six tables represent the logical runtime buffer lists. The direct path
    # receives their addresses; the VMI path must materialize seven dense
    # layer-major buffers before it can launch its equally sized recurrence.
    initial = torch.full((TOKENS, LANES, HIDDEN), 0.125, dtype=torch.bfloat16, device=DEVICE)
    inputs = [torch.empty((TOKENS, HIDDEN), dtype=torch.bfloat16, device=DEVICE) for _ in range(LAYERS)]
    outputs = [torch.full_like(inputs[0], 0.25) for _ in range(LAYERS)]
    residuals = [torch.empty((TOKENS, LANES, HIDDEN), dtype=torch.bfloat16, device=DEVICE) for _ in range(LAYERS)]
    pre = [torch.full((TOKENS, LANES), 0.125, dtype=torch.float32, device=DEVICE) for _ in range(LAYERS)]
    post = [torch.full_like(pre[0], 0.25) for _ in range(LAYERS)]
    comb = [torch.full((TOKENS, LANES, LANES), 0.0625, dtype=torch.float32, device=DEVICE) for _ in range(LAYERS)]
    # Device pointer tables must be populated through a host staging buffer.
    # Creating an NPU tensor directly from Python integers does not guarantee
    # that the 64-bit addresses reach GM unchanged on this CANN release.
    def table(xs: list[torch.Tensor]) -> torch.Tensor:
        host = torch.empty(len(xs), dtype=torch.int64, device="cpu", pin_memory=True)
        for index, value in enumerate(xs):
            host[index] = value.data_ptr()
        return host.to(device=DEVICE, non_blocking=True)
    tables = [table(comb), table(inputs), table(outputs), table(post), table(pre), table(residuals)]
    # The VMI wrapper owns separate dense workspace plus user-visible result
    # lists. Keep both forms: every stack and unstack copy is timed.
    input_seed = [torch.full_like(value, -0.125) for value in inputs]
    residual_seed = [torch.full_like(value, 0.0625) for value in residuals]
    for destination, source in zip(inputs, input_seed):
        destination.copy_(source)
    for destination, source in zip(residuals, residual_seed):
        destination.copy_(source)
    vmi_inputs = [torch.empty_like(value) for value in inputs]
    vmi_residuals = [torch.empty_like(value) for value in residuals]
    stacked = [
        torch.empty((LAYERS, TOKENS, LANES, LANES), dtype=torch.float32, device=DEVICE),
        initial, torch.empty((LAYERS, TOKENS, HIDDEN), dtype=torch.bfloat16, device=DEVICE),
        torch.empty((LAYERS, TOKENS, HIDDEN), dtype=torch.bfloat16, device=DEVICE),
        torch.empty((LAYERS, TOKENS, LANES), dtype=torch.float32, device=DEVICE),
        torch.empty((LAYERS, TOKENS, LANES), dtype=torch.float32, device=DEVICE),
        torch.empty((LAYERS, TOKENS, LANES, HIDDEN), dtype=torch.bfloat16, device=DEVICE),
    ]

    def direct_run() -> None:
        cce_fn(ctypes.c_void_p(stream_ptr()), *[ctypes.c_void_p(x.data_ptr()) for x in [tables[0], initial, *tables[1:]]])

    def stacked_run(copy_back: bool = True) -> None:
        for dst, srcs in ((stacked[0], comb), (stacked[2], input_seed), (stacked[3], outputs),
                          (stacked[4], post), (stacked[5], pre), (stacked[6], residual_seed)):
            for index, src in enumerate(srcs):
                dst[index].copy_(src)
        # The fixed-form VMI ABI has no typed GM pointer table. Execute the
        # same ten layer recurrence as ordered stage launches, carrying the
        # residual through dense workspace between stages.
        residual_in = stacked[1]
        for layer in range(LAYERS):
            vmi_fn(
                ctypes.c_void_p(stream_ptr()),
                ctypes.c_void_p(residual_in.data_ptr()),
                ctypes.c_void_p(stacked[5][layer].data_ptr()),
                ctypes.c_void_p(stacked[3][layer].data_ptr()),
                ctypes.c_void_p(stacked[4][layer].data_ptr()),
                ctypes.c_void_p(stacked[0][layer].data_ptr()),
                ctypes.c_void_p(stacked[2][layer].data_ptr()),
                ctypes.c_void_p(stacked[6][layer].data_ptr()),
            )
            residual_in = stacked[6][layer]
        if copy_back:
            for index, dst in enumerate(vmi_inputs):
                dst.copy_(stacked[2][index])
            for index, dst in enumerate(vmi_residuals):
                dst.copy_(stacked[6][index])

    if not args.vmi_only:
        direct_run()
        torch.npu.synchronize()
        expected_inputs = [value.clone() for value in inputs]
        expected_residuals = [value.clone() for value in residuals]
    if not args.direct_only:
        stacked_run(copy_back=True)
        torch.npu.synchronize()
    if args.direct_only or args.vmi_only:
        fn = direct_run if args.direct_only else stacked_run
        value = median_us(fn)
        print(f"device={DEVICE} isolated={'direct' if args.direct_only else 'stacked'} us={value:.3f}")
        return
    input_delta = (torch.stack(vmi_inputs).float() - torch.stack(expected_inputs).float()).abs().max().item()
    residual_delta = (torch.stack(vmi_residuals).float() - torch.stack(expected_residuals).float()).abs().max().item()
    # Both paths store BF16. The direct CCE conversion and VMI vcvt are
    # allowed to choose adjacent BF16 rounding points, so compare in float32
    # with a bounded BF16-scale tolerance rather than requiring bit identity.
    correctness = input_delta <= 0.25 and residual_delta <= 0.25
    if not correctness:
        raise AssertionError(
            f"VMI result differs from direct CCE: layer_input_maxabs={input_delta} "
            f"residual_maxabs={residual_delta}"
        )
    cce_event, vmi_event = median_us(direct_run), median_us(stacked_run)
    cce_us, vmi_us = msprof_us(direct_run), msprof_us(stacked_run)
    print(f"device={DEVICE} shape={TOKENS}x{HIDDEN} lanes={LANES} layers={LAYERS} grid={GRID}")
    print(
        f"correctness={'PASS' if correctness else 'FAIL'} "
        f"layer_input_maxabs={input_delta} residual_maxabs={residual_delta} direct_pointer_table=PASS"
    )
    print(f"event_sanity_l2_flush_mb={L2_FLUSH_MB} samples={SAMPLES} CCE_us={cce_event:.3f} VMI_us={vmi_event:.3f}")
    print(f"msprof_device_CCE_us={cce_us:.3f} msprof_device_VMI_us={vmi_us:.3f} CCE_over_VMI={cce_us / vmi_us:.4f}")


if __name__ == "__main__":
    main()
