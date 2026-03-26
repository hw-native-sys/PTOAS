#!/usr/bin/python3

import os
import sys
import numpy as np


def compare_bin(golden_path, output_path, dtype, eps):
    if not os.path.exists(golden_path) or not os.path.exists(output_path):
        return False
    golden = np.fromfile(golden_path, dtype=dtype)
    output = np.fromfile(output_path, dtype=dtype)
    return golden.shape == output.shape and np.allclose(golden, output, atol=eps, rtol=eps, equal_nan=True)


def main():
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    ok = compare_bin("golden_v3.bin", "v3.bin", np.float32, 1e-4)
    if not ok:
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
