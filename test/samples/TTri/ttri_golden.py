# PyPTO tile-op baseline: golden for pto.ttri (triangular 0/1 mask).
# Builder stores two 32x32 tiles into one 64x32 GM buffer:
#   rows   0..31: TTRI(diagonal=0, lower)   -> tril with d=0
#   rows  32..63: TTRI(diagonal=16, upper)  -> triu with d=16
# Semantics per pto-isa include/pto/cpu/TTRI.hpp:
#   TTril(d): dst[r,c] = 1 if c < d + r + 1 else 0
#   TTriu(d): dst[r,c] = 1 if c >= d + r  else 0

import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import load_case_meta, single_output, write_golden

ROWS, COLS = 32, 32


def _tril(d):
    r = np.arange(ROWS)[:, None]
    c = np.arange(COLS)[None, :]
    return (c < d + r + 1).astype(np.int32)


def _triu(d):
    r = np.arange(ROWS)[:, None]
    c = np.arange(COLS)[None, :]
    return (c >= d + r).astype(np.int32)


def main():
    meta = load_case_meta()
    out = np.vstack([_tril(0), _triu(16)]).reshape(-1)
    write_golden(meta, {single_output(meta): np.asarray(out, dtype=np.int32)})


if __name__ == '__main__':
    main()
