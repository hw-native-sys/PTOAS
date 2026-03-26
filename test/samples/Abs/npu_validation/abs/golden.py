#!/usr/bin/env python3
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ROWS = 32
COLS = 32
SEED = 19


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)

    v1 = rng.random((ROWS, COLS), dtype=np.float32)
    v2 = np.zeros((ROWS, COLS), dtype=np.float32)
    golden_v2 = np.abs(v1).astype(np.float32, copy=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.reshape(-1).tofile(output_dir / "v1.bin")
    v2.reshape(-1).tofile(output_dir / "v2.bin")
    golden_v2.reshape(-1).tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numpy-based inputs/golden for test/samples/Abs npu_validation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where v1.bin/v2.bin/golden_v2.bin are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Numpy random seed.",
    )
    args = parser.parse_args()

    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
