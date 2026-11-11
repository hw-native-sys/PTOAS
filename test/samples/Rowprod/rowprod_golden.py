# PyPTO tile-op baseline: golden for pto.trowprod (row-wise product across columns).
# Factors are exact dyadic rationals (k/16, k in [1,15]) so the 32-element row
# product accumulates no rounding error in f32; the golden is exact.

import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import default_buffers, load_case_meta, rng, single_output, write_buffers, write_golden


def main():
    meta = load_case_meta()
    [src_name] = meta.inputs
    generator = rng()
    ks = generator.integers(1, 16, size=meta.elem_counts[src_name]).astype(np.int32)
    src = (ks.astype(np.float32)) / 16.0   # exact in f32
    buffers = default_buffers(meta)
    buffers[src_name] = src
    write_buffers(meta, buffers)
    rows = src.size // 32
    out = src.reshape(rows, 32).astype(np.float64).prod(axis=1).astype(np.float32)
    write_golden(meta, {single_output(meta): np.asarray(out, dtype=np.float32)})


if __name__ == '__main__':
    main()
