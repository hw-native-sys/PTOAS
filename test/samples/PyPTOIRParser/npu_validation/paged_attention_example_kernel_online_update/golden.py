#!/usr/bin/env python3
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ROWS = 16
COLS = 128
SEED = 19


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)

    v1 = rng.uniform(-2.0, 2.0, size=(ROWS, 1)).astype(np.float32)
    v2 = rng.uniform(0.5, 2.5, size=(ROWS, 1)).astype(np.float32)
    v3 = rng.uniform(-3.0, 3.0, size=(ROWS, COLS)).astype(np.float32)
    v4 = np.zeros((ROWS, 1), dtype=np.float32)
    v5 = np.zeros((ROWS, 1), dtype=np.float32)
    v6 = np.zeros((ROWS, COLS), dtype=np.float32)
    v7 = np.zeros((ROWS, COLS), dtype=np.float32)
    golden_v7 = (v3 / v2).astype(np.float32, copy=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in {
        "v1": v1,
        "v2": v2,
        "v3": v3,
        "v4": v4,
        "v5": v5,
        "v6": v6,
        "v7": v7,
        "golden_v7": golden_v7,
    }.items():
        value.reshape(-1).tofile(output_dir / f"{name}.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
