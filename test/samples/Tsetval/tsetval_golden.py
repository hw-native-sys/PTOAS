# PyPTO tile-op baseline: 
# golden for tsetval: input with three elements replaced per dtype.
import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import default_buffers, float_values, int_values, load_case_meta, rng, write_buffers, write_golden

OFFSETS = [0, 8 * 32 // 2 + 7, 8 * 32 - 1]
F32_VALS = [3.5, -1.25, 7.0]
I32_VALS = [3, -5, 7]


def main():
    meta = load_case_meta()
    generator = rng()
    buffers = default_buffers(meta)
    goldens = {}
    for in_name, out_name in zip(meta.inputs, meta.outputs):
        if meta.np_types[in_name] == np.float32:
            src = float_values(generator, meta.elem_counts[in_name], style='signed')
            vals = F32_VALS
        else:
            src = int_values(generator, meta.elem_counts[in_name], dtype=np.int32, style='bitwise')
            vals = I32_VALS
        out = src.copy()
        for off, v in zip(OFFSETS, vals):
            out.reshape(-1)[off] = meta.np_types[out_name].type(v)
        buffers[in_name] = src
        goldens[out_name] = np.asarray(out, dtype=meta.np_types[out_name])
    write_buffers(meta, buffers)
    write_golden(meta, goldens)


if __name__ == '__main__':
    main()
