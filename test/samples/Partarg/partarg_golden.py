# PyPTO tile-op baseline: golden for the Partarg builder (tpartargmax + tpartargmin).
# Inputs: src0 (f32 vals), src1 (f32 vals), src0_idx (ui32), src1_idx (ui32).
# Outputs: dst_max/dst_min (f32 values), dst_max_idx/dst_min_idx (ui32 indices).
# Tie-breaking follows the pto-isa reference: later element wins on ties.
import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import default_buffers, float_values, int_values, load_case_meta, rng, write_buffers, write_golden


def main():
    meta = load_case_meta()
    [s0, s1, s0i, s1i] = meta.inputs
    generator = rng()
    src0 = float_values(generator, meta.elem_counts[s0], style='signed')
    src1 = float_values(generator, meta.elem_counts[s1], style='signed')
    src0_idx = int_values(generator, meta.elem_counts[s0i], dtype=np.int32, style='bitwise')
    src1_idx = int_values(generator, meta.elem_counts[s1i], dtype=np.int32, style='bitwise')
    buffers = default_buffers(meta)
    buffers[s0] = src0; buffers[s1] = src1
    buffers[s0i] = src0_idx; buffers[s1i] = src1_idx
    write_buffers(meta, buffers)

    a = src0.astype(np.float64); b = src1.astype(np.float64)
    ai = src0_idx.astype(np.int64).copy(); bi = src1_idx.astype(np.int64).copy()

    # The builder issues TPartArgMax then TPartArgMin on the same dst tiles, so
    # the stored result is the trailing argmin pass (ties: later element wins).
    # Outputs carry generic vN names in the board harness; take each buffer's
    # dtype from the kernel signature (main.cpp) instead of the buffer name.
    mask = a <= b
    goldens = {}
    for out_name in meta.outputs:
        out_dt = meta.np_types[out_name]
        if np.issubdtype(out_dt, np.integer):
            # index output: the index paired with the winning value
            goldens[out_name] = np.asarray(np.where(mask, ai, bi), dtype=out_dt)
        else:
            # value output: the winning value itself
            goldens[out_name] = np.asarray(np.where(mask, a, b), dtype=out_dt)
    write_golden(meta, goldens)


if __name__ == '__main__':
    main()
