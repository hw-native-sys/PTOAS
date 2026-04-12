#!/usr/bin/env python3
# case: kernels/online-softmax-update
# family: kernels
# target_ops: pto.get_block_idx, pto.copy_gm_to_ubuf, pto.copy_ubuf_to_gm, pto.vlds, pto.vcmax, pto.vdup, pto.vmax, pto.vexpdiff, pto.vcadd, pto.vadd, pto.vmul, pto.vdiv, pto.vsts
# scenarios: online-softmax-update, dynamic-rows-and-seq, max-seq-128, block-rows-8, oldmax-oldsum-qk-to-newmax-newsum-expmax-out

import argparse
from pathlib import Path

import numpy as np


ROWS = 24
COLS = 128
SEED = 19
SEQ = 73


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    seq = SEQ
    oldmax = rng.uniform(-3.0, 1.5, size=(ROWS,)).astype(np.float32)
    oldsum = rng.uniform(0.5, 4.0, size=(ROWS,)).astype(np.float32)
    qk = rng.normal(loc=0.0, scale=1.5, size=(ROWS, COLS)).astype(np.float32)

    qk_active = qk[:, :seq]
    qk_rowmax = np.max(qk_active, axis=1)
    newmax = np.maximum(qk_rowmax, oldmax)
    tmp_active = np.exp(qk_active - newmax[:, None], dtype=np.float32)
    cursum = np.sum(tmp_active, axis=1, dtype=np.float32)
    raw_expmax = np.exp(oldmax - newmax, dtype=np.float32)
    newsum = raw_expmax * oldsum + cursum
    expmax = (raw_expmax * oldsum) / newsum
    out = np.zeros((ROWS, COLS), dtype=np.float32)
    out[:, :seq] = tmp_active / newsum[:, None]

    zeros_state = np.zeros((ROWS,), dtype=np.float32)
    zeros_out = np.zeros((ROWS, COLS), dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    oldmax.tofile(output_dir / "v1.bin")
    oldsum.tofile(output_dir / "v2.bin")
    qk.reshape(-1).tofile(output_dir / "v3.bin")
    zeros_state.tofile(output_dir / "v4.bin")
    zeros_state.tofile(output_dir / "v5.bin")
    zeros_state.tofile(output_dir / "v6.bin")
    zeros_out.reshape(-1).tofile(output_dir / "v7.bin")
    np.array([seq], dtype=np.int32).tofile(output_dir / "v8.bin")
    np.array([ROWS], dtype=np.int32).tofile(output_dir / "v9.bin")
    newmax.tofile(output_dir / "golden_v4.bin")
    newsum.tofile(output_dir / "golden_v5.bin")
    expmax.tofile(output_dir / "golden_v6.bin")
    out.astype(np.float32, copy=False).reshape(-1).tofile(output_dir / "golden_v7.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
