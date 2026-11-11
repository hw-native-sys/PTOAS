# PyPTO tile-op baseline: golden for pto.tconcat (32x16 + 32x16 -> 32x32 column concat).
import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import default_buffers, float_values, int_values, load_case_meta, rng, single_output, write_buffers, write_golden


def main():
    meta = load_case_meta()
    [s0, s1] = meta.inputs
    generator = rng()
    a = int_values(generator, meta.elem_counts[s0], dtype=np.int16, style='bitwise')
    b = int_values(generator, meta.elem_counts[s1], dtype=np.int16, style='bitwise')
    buffers = default_buffers(meta)
    buffers[s0] = a; buffers[s1] = b
    write_buffers(meta, buffers)
    out = np.concatenate([a.reshape(32, 16), b.reshape(32, 16)], axis=1).reshape(-1)
    write_golden(meta, {single_output(meta): np.asarray(out, dtype=np.int16)})

if __name__ == '__main__':
    main()
