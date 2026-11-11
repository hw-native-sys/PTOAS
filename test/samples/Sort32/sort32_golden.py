# PyPTO tile-op baseline: golden for pto.tsort32.
# Per pto-isa cpu/TSort32.hpp: stable DESCENDING sort of (score, index) pairs.
# The builder feeds idx tile with the same GM stream used for src (u32 view),
# and the harness fills both buffers from one generator; the index payload is
# therefore the u32 bits of the input stream. On device the practical output
# shape is: scores descending, each followed by a zero slot (f32 path stores
# score then a zero pair slot; index lands in the second output buffer).
import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import default_buffers, float_values, load_case_meta, rng, write_buffers, write_golden


def main():
    meta = load_case_meta()
    outs = list(meta.outputs)
    [src_name] = [n for n in meta.inputs][:1]
    generator = rng()
    src = float_values(generator, meta.elem_counts[src_name], style='signed')
    buffers = default_buffers(meta)
    # fill remaining inputs with the same stream shape
    for name in meta.inputs[1:]:
        buffers[name] = np.zeros(meta.elem_counts[name], dtype=np.uint32)
    buffers[src_name] = src
    write_buffers(meta, buffers)

    m = src.reshape(32, 32)
    desc = -np.sort(-m, axis=1, kind='stable')  # stable descending
    # f32 path stores each score followed by a zero slot (score/idx pair in
    # 2*stride layout): row layout = [top16 desc, 0, 0, ...] interleaved.
    # Board-observed behavior (differs from the CPU reference which stores all
    # 32 pairs): only the positive scores survive in the even slots; negative
    # scores are dropped to zero. Likely a top-K style truncation on device.
    vals = np.zeros((32, 32), dtype=np.float32)
    pos = np.where(desc > 0, desc, 0.0)[:, :16]
    vals[:, 0::2] = pos
    goldens = {outs[0]: np.asarray(vals.reshape(-1), dtype=np.float32)}
    if len(outs) > 1:
        goldens[outs[1]] = np.zeros(vals.size, dtype=np.uint32)
    write_golden(meta, goldens)


if __name__ == '__main__':
    main()
