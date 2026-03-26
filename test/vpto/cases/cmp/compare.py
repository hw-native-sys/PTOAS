#!/usr/bin/python3

import os
import sys
import numpy as np

REPEAT_BYTES = 256


def _ceil_div(x, y):
    return (x + y - 1) // y


def _packed_pred_storage_bytes(logical_elems, src_elem_bytes):
    repeat_elems = REPEAT_BYTES // src_elem_bytes
    if src_elem_bytes == 4:
        repeat_times = _ceil_div(logical_elems, repeat_elems) + 1
        return (repeat_times // 2) * 16
    return _ceil_div(logical_elems, repeat_elems) * (repeat_elems // 8)


def compare_packed_pred_mask(golden_path, output_path, logical_elems, src_elem_bytes):
    if not os.path.exists(golden_path) or not os.path.exists(output_path):
        return False
    golden = np.fromfile(golden_path, dtype=np.uint8)
    output = np.fromfile(output_path, dtype=np.uint8)
    prefix = _packed_pred_storage_bytes(logical_elems, src_elem_bytes)
    if golden.size < prefix or output.size < prefix:
        return False
    if not np.array_equal(golden[:prefix], output[:prefix]):
        diff = np.nonzero(golden[:prefix] != output[:prefix])[0]
        idx = int(diff[0]) if diff.size else 0
        print(f"[ERROR] Mismatch (packed mask): idx={idx} golden={int(golden[idx])} out={int(output[idx])}")
        return False
    return True


def main():
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    ok = compare_packed_pred_mask("golden_v3.bin", "v3.bin", 32 * 32, 4)
    if not ok:
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
