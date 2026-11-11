# PyPTO tile-op baseline: golden for tpipe_v2c (V2C pipe matmul, A[16,32] @ B[32,16]).
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
    a = float_values(generator, meta.elem_counts[a_name], style='signed').astype(np.float32)
    b = float_values(generator, meta.elem_counts[b_name], style='signed').astype(np.float32)
    buffers = default_buffers(meta)
    buffers[a_name] = a
    buffers[b_name] = b
    write_buffers(meta, buffers)
    A = a.reshape(-1)[:16 * 16].reshape(16, 16)
    B = b.reshape(-1)[:16 * 16].reshape(16, 16)
    out = (A @ B).astype(np.float32).reshape(-1)
    out_name = single_output(meta)
    padded = np.zeros(meta.elem_counts[out_name], dtype=np.float32)
    padded[:out.size] = out
    write_golden(meta, {out_name: padded})


if __name__ == '__main__':
    main()
