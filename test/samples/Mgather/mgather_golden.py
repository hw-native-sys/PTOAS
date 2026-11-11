# PyPTO tile-op baseline: golden for pto.mgather coalesce=row (dst[i,:] = src[idx[i], :]).
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
    [gm_name, idx_name] = meta.inputs[:2]
    generator = rng()
    gm = int_values(generator, meta.elem_counts[gm_name], dtype=np.int32, style='bitwise')
    idx = generator.integers(0, 32, size=meta.elem_counts[idx_name]).astype(np.int32)  # in-range row indices
    buffers = default_buffers(meta)
    buffers[gm_name] = gm; buffers[idx_name] = idx
    write_buffers(meta, buffers)
    src = gm.reshape(32, 32)
    rows = idx.reshape(-1)[:32] % 32  # in-range row indices
    out = src[rows, :].reshape(-1)
    write_golden(meta, {meta.outputs[0]: np.asarray(out, dtype=np.int32)})

if __name__ == '__main__':
    main()
