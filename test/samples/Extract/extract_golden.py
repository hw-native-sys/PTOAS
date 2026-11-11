# PyPTO tile-op baseline: golden for pto.textract (leading row of A) + gemv, golden = A@B.
import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import default_buffers, float_values, load_case_meta, rng, single_output, write_buffers, write_golden


def main():
    meta = load_case_meta()
    [a_name, b_name] = meta.inputs
    generator = rng()
    a = float_values(generator, meta.elem_counts[a_name], style='signed').astype(np.float32)
    b = float_values(generator, meta.elem_counts[b_name], style='signed').astype(np.float32)
    buffers = default_buffers(meta)
    buffers[a_name] = a; buffers[b_name] = b
    write_buffers(meta, buffers)
    # extract(0,0) pulls the leading M x K block of A (identity for M=1 full row),
    # then gemv: out = A_extracted @ B  (1x256 @ 256x32 -> 1x32)
    av = a.reshape(-1)[:256].astype(np.float64)
    bv = b.reshape(-1)[:256*32].reshape(256, 32).astype(np.float64)
    out = (av @ bv).astype(np.float32)
    write_golden(meta, {single_output(meta): np.asarray(out, dtype=np.float32)})

if __name__ == '__main__':
    main()
