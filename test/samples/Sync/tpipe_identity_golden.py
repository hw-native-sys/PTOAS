# PyPTO tile-op baseline: golden for tpipe_identity (C2V pushed A@I).
# inputs: a (16x16 f32, random), i (16x16 f32, identity). Golden = A @ I = A.
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
    a_name, i_name = meta.inputs[:2]
    generator = rng()
    a = float_values(generator, meta.elem_counts[a_name], style='signed').astype(np.float32)
    buffers = default_buffers(meta)
    buffers[a_name] = a
    eye = np.eye(16, dtype=np.float32).reshape(-1)
    buffers[i_name] = np.concatenate([eye, np.zeros(max(meta.elem_counts[i_name] - eye.size, 0), dtype=np.float32)])
    write_buffers(meta, buffers)
    out = a.reshape(-1)[:16 * 16]
    out_name = single_output(meta)
    padded = np.zeros(meta.elem_counts[out_name], dtype=np.float32)
    padded[:out.size] = out
    write_golden(meta, {out_name: padded})


if __name__ == '__main__':
    main()
