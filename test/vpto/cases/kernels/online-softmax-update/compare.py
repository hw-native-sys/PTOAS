#!/usr/bin/env python3
# case: kernels/online-softmax-update
# family: kernels
# target_ops: pto.copy_gm_to_ubuf, pto.copy_ubuf_to_gm, pto.vlds, pto.vcmax, pto.vdup, pto.vmax, pto.vexpdiff, pto.vcadd, pto.vadd, pto.vmul, pto.vdiv, pto.vsts
# scenarios: online-softmax-update, 16x128-f32, oldmax-oldsum-qk-to-newmax-newsum-expmax-out

import os
import sys

import numpy as np


def compare_bin(golden_path, output_path, dtype, eps):
    if not os.path.exists(output_path):
        print(f"[ERROR] Output missing: {output_path}")
        return False
    if not os.path.exists(golden_path):
        print(f"[ERROR] Golden missing: {golden_path}")
        return False
    dtype_np = np.dtype(dtype)
    golden = np.fromfile(golden_path, dtype=dtype_np)
    output = np.fromfile(output_path, dtype=dtype_np)
    if golden.shape != output.shape:
        print(f"[ERROR] Shape mismatch: {golden.shape} vs {output.shape}")
        return False
    if not np.allclose(golden, output, atol=eps, rtol=eps, equal_nan=True):
        abs_diff = np.abs(golden.astype(np.float64) - output.astype(np.float64))
        idx = int(np.argmax(abs_diff))
        print(
            f"[ERROR] Mismatch: max diff={float(abs_diff[idx])} at idx={idx} "
            f"(golden={float(golden[idx])}, out={float(output[idx])}, dtype={dtype_np})"
        )
        return False
    return True


def main():
    ok = True
    ok = compare_bin("golden_v4.bin", "v4.bin", np.float32, 1e-4) and ok
    ok = compare_bin("golden_v5.bin", "v5.bin", np.float32, 1e-4) and ok
    ok = compare_bin("golden_v6.bin", "v6.bin", np.float32, 1e-4) and ok
    ok = compare_bin("golden_v7.bin", "v7.bin", np.float32, 1e-4) and ok
    if not ok:
        print("[ERROR] compare failed")
        sys.exit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
