# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate and time the serialized and barrier-free FP32 accumulator chains."""

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import random
import statistics

import numpy as np


class Device:
    def __init__(self, device, library):
        self.device = device
        self.acl = C.CDLL("libascendcl.so", mode=C.RTLD_GLOBAL)
        self.acl.aclGetRecentErrMsg.restype = C.c_char_p
        self.acl.aclrtGetSocName.restype = C.c_char_p
        signatures = {
            "aclInit": [C.c_char_p],
            "aclFinalize": [],
            "aclrtSetDevice": [C.c_int],
            "aclrtResetDevice": [C.c_int],
            "aclrtCreateStream": [C.POINTER(C.c_void_p)],
            "aclrtDestroyStream": [C.c_void_p],
            "aclrtSynchronizeStream": [C.c_void_p],
            "aclrtMalloc": [C.POINTER(C.c_void_p), C.c_size_t, C.c_int],
            "aclrtFree": [C.c_void_p],
            "aclrtMemcpy": [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t, C.c_int],
            "aclrtCreateEventWithFlag": [C.POINTER(C.c_void_p), C.c_uint32],
            "aclrtRecordEvent": [C.c_void_p, C.c_void_p],
            "aclrtDestroyEvent": [C.c_void_p],
            "aclrtEventElapsedTime": [C.POINTER(C.c_float), C.c_void_p, C.c_void_p],
        }
        for name, arguments in signatures.items():
            getattr(self.acl, name).argtypes = arguments
            getattr(self.acl, name).restype = C.c_int
        self.check("aclInit", None)
        self.check("aclrtSetDevice", device)
        self.soc = self.acl.aclrtGetSocName().decode()
        if "950" not in self.soc:
            raise RuntimeError(
                f"This benchmark requires Ascend950, received {self.soc}"
            )
        self.stream = C.c_void_p()
        self.check("aclrtCreateStream", C.byref(self.stream))
        self.events = [C.c_void_p(), C.c_void_p()]
        for event in self.events:
            self.check("aclrtCreateEventWithFlag", C.byref(event), 8)
        self.allocations = []
        self.lib = C.CDLL(str(library), mode=C.RTLD_GLOBAL)
        self.launch = self.lib.LaunchBarrierBenchmark
        self.launch.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_uint32,
            C.c_void_p,
        ]
        self.launch.restype = None

    def check(self, name, *arguments):
        status = getattr(self.acl, name)(*arguments)
        if status:
            detail = self.acl.aclGetRecentErrMsg()
            raise RuntimeError(f"{name}: {status}: {detail.decode() if detail else ''}")

    def upload(self, array):
        pointer = C.c_void_p()
        self.check("aclrtMalloc", C.byref(pointer), array.nbytes, 0)
        self.allocations.append(pointer)
        self.overwrite(pointer, array)
        return pointer

    def overwrite(self, pointer, array):
        self.check(
            "aclrtMemcpy",
            pointer,
            array.nbytes,
            C.c_void_p(array.ctypes.data),
            array.nbytes,
            1,
        )

    def download(self, pointer, shape):
        array = np.empty(shape, dtype=np.float32)
        self.check(
            "aclrtMemcpy",
            C.c_void_p(array.ctypes.data),
            array.nbytes,
            pointer,
            array.nbytes,
            2,
        )
        return array

    def measure(self, a, w, out, rows, padded_rows, serialize):
        self.check("aclrtRecordEvent", self.events[0], self.stream)
        self.launch(
            a,
            w,
            out,
            rows,
            padded_rows,
            serialize,
            (padded_rows // 16) * 4,
            self.stream,
        )
        self.check("aclrtRecordEvent", self.events[1], self.stream)
        self.check("aclrtSynchronizeStream", self.stream)
        elapsed = C.c_float()
        self.check("aclrtEventElapsedTime", C.byref(elapsed), *self.events)
        if not np.isfinite(elapsed.value) or elapsed.value <= 0:
            raise RuntimeError(f"Invalid device time: {elapsed.value}")
        return elapsed.value * 1000

    def close(self):
        for pointer in self.allocations:
            self.check("aclrtFree", pointer)
        for event in self.events:
            self.check("aclrtDestroyEvent", event)
        self.check("aclrtDestroyStream", self.stream)
        self.check("aclrtResetDevice", self.device)
        self.check("aclFinalize")


def partials(device, output, rows, padded_rows):
    values = device.download(output, (4 * padded_rows, 32))
    return np.stack(
        [
            values[split * padded_rows : split * padded_rows + rows, :24]
            for split in range(4)
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library",
        type=Path,
        default=Path(__file__).parent / "build/libbarrier_benchmark.so",
    )
    parser.add_argument("--device", type=int)
    parser.add_argument("--rows", type=int, default=72)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[20261106, 20261107, 20261108]
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "build/results.json"
    )
    args = parser.parse_args()
    allocated = os.environ.get("TASK_DEVICE", "")
    if args.device is None:
        if not allocated:
            parser.error("Pass --device, or use a device queue that sets TASK_DEVICE")
        args.device = int(allocated.split(",")[0])
    if allocated and args.device not in [int(value) for value in allocated.split(",")]:
        parser.error("--device must belong to the task allocation")
    if min(args.rows, args.rounds, args.batches) < 1 or args.warmup < 0:
        parser.error(
            "rows, rounds and batches must be positive; warmup must be nonnegative"
        )
    args.library = args.library.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    padded_rows = ((args.rows + 15) // 16) * 16
    device = Device(args.device, args.library)
    result = {
        "metadata": {
            "soc": device.soc,
            "device": args.device,
            "rows": args.rows,
            "rounds": args.rounds,
            "warmup": args.warmup,
            "batches": args.batches,
            "seeds": args.seeds,
            "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
            "timing": "ACL device events; native kernel including prologue/epilogue",
            "cache": "warm input reuse; no forced L2 flush",
            "protocol": "paired random order; poison before and check after each timed call",
        },
        "validation": [],
        "batches": [],
    }
    try:
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            a = rng.normal(0, 0.125, (args.rows, 20480)).astype(np.float32)
            w = rng.normal(0, 0.125, (24, 20480)).astype(np.float32)
            template = np.full((4 * padded_rows, 32), np.nan, dtype=np.float32)
            da, dw = device.upload(a), device.upload(w)
            outputs = [device.upload(template), device.upload(template)]
            reference = np.stack(
                [
                    a[:, split * 5120 : (split + 1) * 5120].astype(np.float64)
                    @ w[:, split * 5120 : (split + 1) * 5120].astype(np.float64).T
                    for split in range(4)
                ]
            )
            checked = []
            for serialize in [0, 1]:
                device.overwrite(outputs[serialize], template)
                device.measure(
                    da, dw, outputs[serialize], args.rows, padded_rows, serialize
                )
                got = partials(device, outputs[serialize], args.rows, padded_rows)
                np.testing.assert_allclose(got, reference, rtol=1e-4, atol=1e-4)
                if not np.all(np.isfinite(got)):
                    raise AssertionError("Unwritten or non-finite output")
                checked.append(got)
                result["validation"].append(
                    {
                        "seed": seed,
                        "serialize": serialize,
                        "max_abs_error": float(np.max(np.abs(got - reference))),
                    }
                )
            np.testing.assert_array_equal(checked[0], checked[1])
            if seed == args.seeds[0]:
                fixture = da, dw, outputs, template, checked[0]
        da, dw, outputs, template, expected = fixture
        for batch in range(args.batches):
            for _ in range(args.warmup):
                for serialize in [0, 1]:
                    device.measure(
                        da, dw, outputs[serialize], args.rows, padded_rows, serialize
                    )
            samples = [[], []]
            pairs = []
            rng = random.Random(args.seeds[0] + batch)
            for _ in range(args.rounds):
                order = [0, 1]
                rng.shuffle(order)
                pair = {}
                for serialize in order:
                    device.overwrite(outputs[serialize], template)
                    elapsed = device.measure(
                        da, dw, outputs[serialize], args.rows, padded_rows, serialize
                    )
                    np.testing.assert_array_equal(
                        partials(device, outputs[serialize], args.rows, padded_rows),
                        expected,
                    )
                    samples[serialize].append(elapsed)
                    pair[serialize] = elapsed
                pairs.append(
                    {
                        "serialized_us": pair[1],
                        "barrier_free_us": pair[0],
                        "order": order,
                    }
                )
            record = {
                "batch": batch,
                "serialized_median_us": statistics.median(samples[1]),
                "barrier_free_median_us": statistics.median(samples[0]),
                "median_paired_delta_percent": statistics.median(
                    (pair["barrier_free_us"] / pair["serialized_us"] - 1) * 100
                    for pair in pairs
                ),
                "pairs": pairs,
            }
            result["batches"].append(record)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(
                json.dumps(
                    {key: value for key, value in record.items() if key != "pairs"}
                ),
                flush=True,
            )
        serialized = statistics.median(
            batch["serialized_median_us"] for batch in result["batches"]
        )
        barrier_free = statistics.median(
            batch["barrier_free_median_us"] for batch in result["batches"]
        )
        result["summary"] = {
            "serialized_us": serialized,
            "barrier_free_us": barrier_free,
            "delta_percent": (barrier_free / serialized - 1) * 100,
            "median_paired_delta_percent": statistics.median(
                batch["median_paired_delta_percent"] for batch in result["batches"]
            ),
        }
        result["metadata"]["status"] = "PASS"
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["summary"]), flush=True)
    finally:
        device.close()


if __name__ == "__main__":
    main()
