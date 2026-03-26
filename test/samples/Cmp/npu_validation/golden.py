#!/usr/bin/env python3
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ROWS = 32
COLS = 32
SEED = 19
SRC_ELEM_BYTES = 4
REPEAT_BYTES = 256


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _packed_pred_storage_bytes(logical_elems: int, src_elem_bytes: int) -> int:
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


def _pack_predicate_mask(mask: np.ndarray, src_elem_bytes: int) -> np.ndarray:
    if mask.dtype != np.bool_:
        raise TypeError(f"expected bool mask, got {mask.dtype}")
    if mask.shape != (ROWS, COLS):
        raise ValueError(f"expected mask shape {(ROWS, COLS)}, got {mask.shape}")

    logical_elems = ROWS * COLS
    stored_bytes = _packed_pred_storage_bytes(logical_elems, src_elem_bytes)
    packed_bits = np.packbits(mask.reshape(-1).astype(np.uint8, copy=False), bitorder="little")
    out = np.zeros((logical_elems,), dtype=np.uint8)
    out[:stored_bytes] = 0
    out[: packed_bits.size] = packed_bits
    return out


def generate(output_dir: Path, seed: int, src_elem_bytes: int) -> None:
    rng = np.random.default_rng(seed)

    v1 = rng.random((ROWS, COLS), dtype=np.float32)
    v2 = rng.random((ROWS, COLS), dtype=np.float32)
    mask = np.less(v1, v2)

    packed_mask = _pack_predicate_mask(mask, src_elem_bytes)
    output_init = np.zeros((ROWS * COLS,), dtype=np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.reshape(-1).tofile(output_dir / "v1.bin")
    v2.reshape(-1).tofile(output_dir / "v2.bin")
    output_init.tofile(output_dir / "v3.bin")
    packed_mask.tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numpy-based inputs/golden for test/samples/Cmp npu_validation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where v1.bin/v2.bin/v3.bin/golden_v3.bin are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Numpy random seed.",
    )
    parser.add_argument(
        "--src-elem-bytes",
        type=int,
        default=SRC_ELEM_BYTES,
        help="Source element byte width used by TCMP/TCMPS semantics.",
    )
    args = parser.parse_args()

    generate(args.output_dir, args.seed, args.src_elem_bytes)


if __name__ == "__main__":
    main()
