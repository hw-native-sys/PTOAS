# PyPTO tile-op baseline: golden for pto.thistogram (ui16 source, both byte variants).
#
# Semantics (mirrors pto-isa include/pto/cpu/THistogram.hpp, the authoritative
# reference implementation):
#   byte=1 (MSB): per row, count occurrences of each high-byte value (0..255).
#   byte=0 (LSB): per row, count low-byte values, but only for elements whose
#                 high byte equals idx[row] (the radix-filter pass).
#   The stored dst is the CUMULATIVE histogram (prefix sum over bins):
#   dst[row, bin] = sum(counts[0..bin]).
#
# The builder runs both variants on the same src/idx tiles, so this golden
# writes both outputs (dst0 = byte1, dst1 = byte0).

import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import default_buffers, load_case_meta, rng, write_buffers, write_golden

ROWS, COLS, BINS = 32, 32, 256


def main():
    meta = load_case_meta()
    src_name, idx_name = meta.inputs[:2]
    out_names = [n for n in meta.outputs]
    generator = rng()
    src = generator.integers(0, 2**16, size=ROWS * COLS).astype(np.uint16)
    idx = generator.integers(0, 256, size=ROWS).astype(np.uint8)
    buffers = default_buffers(meta)
    buffers[src_name] = src
    buffers[idx_name] = idx
    write_buffers(meta, buffers)

    src2 = src.reshape(ROWS, COLS)
    msb = (src2 >> 8).astype(np.uint16)
    lsb = (src2 & 0xFF).astype(np.uint16)

    hist_msb = np.zeros((ROWS, BINS), dtype=np.uint32)
    hist_lsb = np.zeros((ROWS, BINS), dtype=np.uint32)
    for r in range(ROWS):
        # byte=1: histogram the high byte of every element.
        counts = np.bincount(msb[r], minlength=BINS)
        hist_msb[r] = counts.astype(np.uint32)
        # byte=0: histogram the low byte, filtered by msb == idx[r].
        sel = msb[r] == idx[r]
        counts = np.bincount(lsb[r][sel], minlength=BINS)
        hist_lsb[r] = counts.astype(np.uint32)
    # dst holds the cumulative histogram (prefix sum over bins).
    hist_msb = np.cumsum(hist_msb, axis=1).astype(np.uint32)
    hist_lsb = np.cumsum(hist_lsb, axis=1).astype(np.uint32)

    goldens = {}
    for name, hist in zip(out_names, (hist_msb, hist_lsb)):
        goldens[name] = np.asarray(hist.reshape(-1), dtype=np.uint32)
    write_golden(meta, goldens)


if __name__ == '__main__':
    main()
