# PyPTO tile-op baseline: golden for pto.mscatter coalesce=row (GM[idx[i],:] = src[i,:]).
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
    [src_name, idx_name] = meta.inputs[:2]
    out_name = meta.outputs[0]
    generator = rng()
    src = int_values(generator, meta.elem_counts[src_name], dtype=np.int32, style='bitwise')
    idx = generator.integers(0, 32, size=meta.elem_counts[idx_name]).astype(np.int32)  # in-range row indices
    buffers = default_buffers(meta)
    buffers[src_name] = src; buffers[idx_name] = idx
    write_buffers(meta, buffers)
    gm = np.zeros(32 * 32, dtype=np.int32)
    s = src.reshape(32, 32)
    rows = idx.reshape(-1)[:32] % 32
    for i in range(32):
        gm.reshape(32,32)[rows[i], :] = s[i, :]
    write_golden(meta, {out_name: np.asarray(gm, dtype=np.int32)})

if __name__ == '__main__':
    main()
