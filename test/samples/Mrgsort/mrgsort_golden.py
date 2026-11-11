# PyPTO tile-op baseline: golden for pto.tmrgsort format1 (blockLen=64).
#
# Device semantics decoded on-board (238 / Ascend950PR, pto-isa pin b1d460e):
# - The 1x1024 f32 tile holds 512 interleaved (score: f32, index: u32-bits)
#   struct pairs of 8 bytes (STRUCT_BYTES=8 per pto-isa TMrgSort.hpp).
# - INPUT contract (else device errcode 343 "input data of the sorting
#   instruction is not correctly sorted"): the list must be laid out as
#   runs of blockLen=64 structs, each run sorted DESCENDING by score,
#   with the index field ascending 0..511 across the whole list.
# - OUTPUT: pairs preserved exactly (set-equal); each adjacent pair of
#   64-runs is merged into an independent 128-struct block sorted
#   DESCENDING by score (merge breakpoints observed at struct 127/255/383).
#   It is a 2-way merge accelerator per block, not a full-list sort.
import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import default_buffers, float_values, load_case_meta, rng, single_output, write_buffers, write_golden

PAIRS = 512      # 1024 f32 elements = 512 struct pairs
RUN = 64         # blockLen: structs per input run
BLOCK = 128      # output granularity: merged 2-run blocks (256 floats)


def main():
    meta = load_case_meta()
    [src_name] = meta.inputs
    generator = rng()
    n = meta.elem_counts[src_name]

    # Input: 8 runs x 64 structs, each run descending by score; idx 0..511.
    runs = []
    for _ in range(PAIRS // RUN):
        sc = np.clip(np.abs(float_values(generator, RUN, style='signed')), 0.05, 3.0)
        sc = np.sort(sc)[::-1]          # abs FIRST, then sort descending
        runs.append(sc)
    scores = np.concatenate(runs).astype(np.float32)
    idxs = np.arange(PAIRS, dtype=np.uint32)
    src = np.empty(2 * PAIRS, dtype=np.float32)
    src[0::2] = scores
    src[1::2] = idxs.view(np.float32)

    buffers = default_buffers(meta)
    buffers[src_name] = src
    write_buffers(meta, buffers)

    # Expected: per-128-struct block, merge of its two 64-runs, descending.
    out = src.copy()
    for b in range(0, PAIRS, BLOCK):
        blk_sc = out[2 * b:2 * (b + BLOCK):2].copy()
        blk_ix = out[2 * b + 1:2 * (b + BLOCK):2].copy().view(np.uint32)
        order = np.argsort(-blk_sc, kind='stable')
        out[2 * b:2 * (b + BLOCK):2] = blk_sc[order]
        out[2 * b + 1:2 * (b + BLOCK):2] = blk_ix[order].view(np.float32)

    if n > 2 * PAIRS:
        out = np.concatenate([out, np.zeros(n - 2 * PAIRS, dtype=np.float32)])
    write_golden(meta, {single_output(meta): np.asarray(out, dtype=np.float32)})


if __name__ == '__main__':
    main()
