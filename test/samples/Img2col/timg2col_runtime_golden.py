#!/usr/bin/env python3
"""Golden generator for the timg2col runtime sample.

Geometry (mirrors pto-isa ST timg2col float32 cases):
  v1 fmap   f32 NC1HWC0 [1, 2, 4, 6, 8]
  v2 weight f32 5D FRACTAL_Z [C1=2, FH=2, FW=2, N=16, C0=8]
  v3 out    f32 [M=24, N=16]

Reference model: the kernel computes tmatmul(left, right) where
left = timg2col(fmap) with k ordered [C1, FH, FW, C0] (c-major) and
right = extract(weight) with right[k, n] = weight[c1, fh, fw, n, c0].
The reference therefore convolves the NHWC fmap: out[m, n] equals the
sliding-window dot product with the same k ordering.
"""
from pathlib import Path
import sys

import numpy as np

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / "validation_runtime.py").is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import load_case_meta, rng, write_buffers, write_golden

FMAP_N, FMAP_C1, FMAP_H, FMAP_W, FMAP_C0 = 1, 2, 4, 6, 8
WG_C1, WG_FH, WG_FW, WG_N, WG_C0 = 2, 2, 2, 16, 8
STRIDE_H, STRIDE_W = 1, 1
PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT = 1, 0, 1, 0
C_IN = FMAP_C1 * FMAP_C0
H_OUT = (FMAP_H + PAD_TOP + PAD_BOTTOM - (WG_FH - 1) - 1) // STRIDE_H + 1
W_OUT = (FMAP_W + PAD_LEFT + PAD_RIGHT - (WG_FW - 1) - 1) // STRIDE_W + 1
M = FMAP_N * H_OUT * W_OUT
N = WG_N


def k_index(c1, fh, fw, c0):
    """Flat K index with the c-major ordering TIMG2COL produces."""
    return c1 * (WG_FH * WG_FW * WG_C0) + fh * (WG_FW * WG_C0) + fw * WG_C0 + c0


def main():
    meta = load_case_meta()
    generator = rng()

    in_names = list(meta.inputs)
    out_names = list(meta.outputs)
    src_name = in_names[0] if in_names else "v1"
    wgt_name = in_names[1] if len(in_names) > 1 else "v2"
    dst_name = out_names[0] if out_names else "v3"

    fmap = generator.uniform(-3, 3, size=FMAP_N * FMAP_C1 * FMAP_H * FMAP_W * FMAP_C0).astype(np.float32)
    fmap = fmap.reshape(FMAP_N, FMAP_C1, FMAP_H, FMAP_W, FMAP_C0)
    weight = generator.uniform(-3, 3, size=WG_C1 * WG_FH * WG_FW * WG_N * WG_C0).astype(np.float32)
    weight = weight.reshape(WG_C1, WG_FH, WG_FW, WG_N, WG_C0)

    fmap_nhwc = fmap.transpose(0, 2, 3, 1, 4).reshape(FMAP_N, FMAP_H, FMAP_W, C_IN)
    padded = np.pad(
        fmap_nhwc,
        pad_width=((0, 0), (PAD_TOP, PAD_BOTTOM), (PAD_LEFT, PAD_RIGHT), (0, 0)),
        mode="constant",
    )

    # weight row for output channel n, in the same k ordering as left columns.
    w_rows = np.zeros((WG_N, VALID_K := WG_C1 * WG_FH * WG_FW * WG_C0), dtype=np.float64)
    for no in range(WG_N):
        for c1 in range(WG_C1):
            for fh in range(WG_FH):
                for fw in range(WG_FW):
                    for c0 in range(WG_C0):
                        w_rows[no, k_index(c1, fh, fw, c0)] = weight[c1, fh, fw, no, c0]

    out = np.zeros((M, N), dtype=np.float32)
    for n_idx in range(FMAP_N):
        for ho in range(H_OUT):
            for wo in range(W_OUT):
                m = n_idx * H_OUT * W_OUT + ho * W_OUT + wo
                patch = padded[n_idx, ho * STRIDE_H:ho * STRIDE_H + WG_FH,
                               wo * STRIDE_W:wo * STRIDE_W + WG_FW, :]  # [FH, FW, C]
                # Build left row in the k ordering: k = c1*64 + fh*16 + fw*8 + c0.
                col = np.zeros(VALID_K, dtype=np.float64)
                for c1 in range(WG_C1):
                    for fh in range(WG_FH):
                        for fw in range(WG_FW):
                            for c0 in range(WG_C0):
                                col[k_index(c1, fh, fw, c0)] = patch[fh, fw, c1 * WG_C0 + c0]
                for no in range(WG_N):
                    out[m, no] = np.float32(np.sum(col * w_rows[no]))

    buffers = {src_name: fmap.reshape(-1), wgt_name: weight.reshape(-1),
               dst_name: np.zeros(M * N, dtype=np.float32)}
    write_buffers(meta, buffers)
    write_golden(meta, {dst_name: out.reshape(-1)})


if __name__ == "__main__":
    main()
