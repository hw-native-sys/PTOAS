#!/usr/bin/python3
# coding=utf-8

import os
import sys
import numpy as np


REPEAT_BYTES = 256


def _ceil_div(x, y):
    return (x + y - 1) // y


def _packed_pred_storage_bytes(logical_elems, src_elem_bytes):
    logical_elems = int(logical_elems)
    src_elem_bytes = int(src_elem_bytes)
    if logical_elems <= 0:
        raise ValueError(f"logical_elems must be > 0, got {logical_elems}")
    if src_elem_bytes not in (1, 2, 4):
        raise ValueError(f"unsupported packed predicate source size: {src_elem_bytes}")

    repeat_elems = REPEAT_BYTES // src_elem_bytes
    if src_elem_bytes == 4:
        repeat_times = _ceil_div(logical_elems, repeat_elems) + 1
        loop_count = repeat_times // 2
        return loop_count * 16

    repeat_times = _ceil_div(logical_elems, repeat_elems)
    return repeat_times * (repeat_elems // 8)


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
        print(f"[ERROR] Shape mismatch: {golden_path} {golden.shape} vs {output_path} {output.shape}")
        return False
    if not np.allclose(golden, output, atol=eps, rtol=eps, equal_nan=True):
        if golden.size:
            if np.issubdtype(dtype_np, np.floating):
                g = golden.astype(np.float64, copy=False)
                o = output.astype(np.float64, copy=False)
            elif np.issubdtype(dtype_np, np.integer) or np.issubdtype(dtype_np, np.unsignedinteger):
                g = golden.astype(np.int64, copy=False)
                o = output.astype(np.int64, copy=False)
            else:
                g = golden.astype(np.float64, copy=False)
                o = output.astype(np.float64, copy=False)
            abs_diff = np.abs(g - o)
            idx = int(np.argmax(abs_diff))
            diff = float(abs_diff[idx])
            print(
                f"[ERROR] Mismatch: {golden_path} vs {output_path}, max diff={diff} at idx={idx} "
                f"(golden={g[idx]}, out={o[idx]}, dtype={dtype_np})"
            )
        else:
            print(f"[ERROR] Mismatch: {golden_path} vs {output_path}, empty buffers, dtype={dtype_np}")
        return False
    return True


def compare_bin_prefix(golden_path, output_path, dtype, eps, count):
    if not os.path.exists(output_path):
        print(f"[ERROR] Output missing: {output_path}")
        return False
    if not os.path.exists(golden_path):
        print(f"[ERROR] Golden missing: {golden_path}")
        return False
    try:
        count = int(count)
    except Exception:
        print(f"[ERROR] Invalid prefix count: {count}")
        return False
    if count <= 0:
        print(f"[ERROR] Invalid prefix count: {count}")
        return False

    dtype_np = np.dtype(dtype)
    golden = np.fromfile(golden_path, dtype=dtype_np, count=count)
    output = np.fromfile(output_path, dtype=dtype_np, count=count)

    if golden.size != count or output.size != count:
        print(
            f"[ERROR] Prefix read too small: need={count} elems, "
            f"golden={golden.size}, out={output.size}"
        )
        return False

    if not np.allclose(golden, output, atol=eps, rtol=eps, equal_nan=True):
        if golden.size:
            if np.issubdtype(dtype_np, np.floating):
                g = golden.astype(np.float64, copy=False)
                o = output.astype(np.float64, copy=False)
            elif np.issubdtype(dtype_np, np.integer) or np.issubdtype(dtype_np, np.unsignedinteger):
                g = golden.astype(np.int64, copy=False)
                o = output.astype(np.int64, copy=False)
            else:
                g = golden.astype(np.float64, copy=False)
                o = output.astype(np.float64, copy=False)
            abs_diff = np.abs(g - o)
            idx = int(np.argmax(abs_diff))
            diff = float(abs_diff[idx])
            print(
                f"[ERROR] Mismatch (prefix): {golden_path} vs {output_path}, max diff={diff} at idx={idx} "
                f"(golden={g[idx]}, out={o[idx]}, dtype={dtype_np}, count={count})"
            )
        else:
            print(f"[ERROR] Mismatch (prefix): {golden_path} vs {output_path}, empty buffers, dtype={dtype_np}")
        return False
    return True


def compare_packed_pred_mask(golden_path, output_path, logical_elems, src_elem_bytes):
    """
    Compare outputs of pto.tcmp / pto.tcmps.

    PTO-ISA stores packed predicate results as a linear PK byte stream via
    `psts`, with the exact written prefix length determined by the typed
    TCMP/TCMPS repeat schedule. Compare only that semantic prefix.
    """
    if not os.path.exists(output_path):
        print(f"[ERROR] Output missing: {output_path}")
        return False
    if not os.path.exists(golden_path):
        print(f"[ERROR] Golden missing: {golden_path}")
        return False
    try:
        logical_elems = int(logical_elems)
        src_elem_bytes = int(src_elem_bytes)
    except Exception:
        print(
            "[ERROR] Invalid packed mask compare arguments: "
            f"logical_elems={logical_elems} src_elem_bytes={src_elem_bytes}"
        )
        return False
    if logical_elems <= 0 or src_elem_bytes <= 0:
        print(
            "[ERROR] Invalid packed mask compare arguments: "
            f"logical_elems={logical_elems} src_elem_bytes={src_elem_bytes}"
        )
        return False

    golden = np.fromfile(golden_path, dtype=np.uint8)
    output = np.fromfile(output_path, dtype=np.uint8)
    try:
        prefix_bytes = _packed_pred_storage_bytes(logical_elems, src_elem_bytes)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return False

    if golden.size < prefix_bytes or output.size < prefix_bytes:
        print(
            f"[ERROR] Packed mask buffer too small: need={prefix_bytes} bytes, "
            f"golden={golden.size}, out={output.size}"
        )
        return False

    golden_sel = golden[:prefix_bytes]
    output_sel = output[:prefix_bytes]

    if not np.array_equal(golden_sel, output_sel):
        diff = np.nonzero(golden_sel != output_sel)[0]
        idx = int(diff[0]) if diff.size else 0
        print(
            f"[ERROR] Mismatch (packed mask): {golden_path} vs {output_path}, first diff at idx={idx} "
            f"(golden={int(golden_sel[idx])}, out={int(output_sel[idx])}, "
            f"logical_elems={logical_elems}, src_elem_bytes={src_elem_bytes}, prefix_bytes={prefix_bytes})"
        )
        return False
    return True


def main():
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    ok = True
    ok = compare_bin("golden_v2.bin", "v2.bin", np.float32, 0.0001) and ok
    if not ok:
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
