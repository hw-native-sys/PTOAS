#!/usr/bin/python3
# coding=utf-8

import os
import sys
import numpy as np


def compare_one(name: str, eps: float = 1e-6) -> bool:
    g = f"golden_{name}.bin"
    o = f"{name}.bin"
    if not os.path.exists(g):
        print(f"[ERROR] Golden missing: {g}")
        return False
    if not os.path.exists(o):
        print(f"[ERROR] Output missing: {o}")
        return False

    gv = np.fromfile(g, dtype=np.float32)
    ov = np.fromfile(o, dtype=np.float32)
    if gv.shape != ov.shape:
        print(f"[ERROR] Shape mismatch for {name}: golden={gv.shape}, out={ov.shape}")
        return False

    if not np.allclose(gv, ov, atol=eps, rtol=eps, equal_nan=True):
        diff = np.abs(gv.astype(np.float64) - ov.astype(np.float64))
        idx = int(np.argmax(diff))
        print(
            f"[ERROR] {name} mismatch: max_diff={float(diff[idx])} idx={idx} "
            f"golden={float(gv[idx])} out={float(ov[idx])}"
        )
        return False

    print(f"[INFO] {name} compare passed")
    return True


def main():
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    ok = True
    for n in ["out0", "out1", "out2", "out3"]:
        ok = compare_one(n) and ok

    if ok:
        print("[INFO] compare passed")
        return

    if strict:
        print("[ERROR] compare failed")
        sys.exit(2)
    print("[WARN] compare failed (non-gating)")


if __name__ == "__main__":
    main()
