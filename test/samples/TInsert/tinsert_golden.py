# PyPTO tile-op baseline: golden for pto.tinsert (A@B acc block -> f16 mat, then @I).
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
    [a_name, b_name, i_name] = meta.inputs
    generator = rng()
    a = float_values(generator, meta.elem_counts[a_name], style='signed')
    b = float_values(generator, meta.elem_counts[b_name], style='signed')
    i16 = float_values(generator, meta.elem_counts[i_name], style='signed').astype(np.float16)
    buffers = default_buffers(meta)
    buffers[a_name] = a; buffers[b_name] = b; buffers[i_name] = i16
    write_buffers(meta, buffers)
    # acc = A @ B (f32); insert casts acc block into the f16 mat (f16 rounding);
    # out = dst_block @ I in f16 math, result widened to f32.
    A = a.reshape(32, 32).astype(np.float64)
    B = b.reshape(32, 32).astype(np.float64)
    acc = A @ B
    dst16 = acc.astype(np.float16).astype(np.float64)
    I = i16.reshape(32, 32).astype(np.float64)
    out = (dst16 @ I).astype(np.float32)
    write_golden(meta, {single_output(meta): np.asarray(out, dtype=np.float32)})

if __name__ == '__main__':
    main()
