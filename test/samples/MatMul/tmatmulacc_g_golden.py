# PyPTO tile-op baseline: golden for pto.tmatmul + tmatmul.acc double accumulate (golden = 2*A@B).
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
    [a_name, b_name] = meta.inputs[:2]
    generator = rng()
    a = float_values(generator, meta.elem_counts[a_name], style='signed')
    b = float_values(generator, meta.elem_counts[b_name], style='signed')
    buffers = default_buffers(meta)
    buffers[a_name] = a; buffers[b_name] = b
    write_buffers(meta, buffers)
    A = a.reshape(32, 32).astype(np.float64)
    B = b.reshape(32, 32).astype(np.float64)
    out = (2.0 * (A @ B)).astype(np.float32)
    write_golden(meta, {single_output(meta): np.asarray(out, dtype=np.float32)})

if __name__ == '__main__':
    main()
