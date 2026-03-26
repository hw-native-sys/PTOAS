#!/usr/bin/env python3
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ROWS = 32
COLS = 32
SEED = 19
SCALAR = np.float32(3.14)


def generate(output_dir: Path, seed: int) -> None:
    del seed
    v1 = np.zeros((ROWS, COLS), dtype=np.float32)
    golden_v1 = np.full((ROWS, COLS), SCALAR, dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.reshape(-1).tofile(output_dir / "v1.bin")
    golden_v1.reshape(-1).tofile(output_dir / "golden_v1.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
