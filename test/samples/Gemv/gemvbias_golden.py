# PyPTO tile-op baseline: golden for pto.tgemv.bias (A@B + bias, f16 in f32 out).
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
    [a_name, b_name, bias_name] = meta.inputs
    generator = rng()
    a = float_values(generator, meta.elem_counts[a_name], style='signed').astype(np.float16)
    b = float_values(generator, meta.elem_counts[b_name], style='signed').astype(np.float16)
    bias = float_values(generator, meta.elem_counts[bias_name], style='signed').astype(np.float16)
    buffers = default_buffers(meta)
    buffers[a_name] = a; buffers[b_name] = b; buffers[bias_name] = bias
    write_buffers(meta, buffers)
    av = a.reshape(-1)[:256].astype(np.float32)
    bv = b.reshape(-1)[:256*32].reshape(256, 32).astype(np.float32)
    biasv = bias.reshape(-1)[:32].astype(np.float32)
    out = (av @ bv + biasv[None, :]).astype(np.float32)
    write_golden(meta, {single_output(meta): np.asarray(out, dtype=np.float32)})

if __name__ == '__main__':
    main()
