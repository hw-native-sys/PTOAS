# PyPTO tile-op baseline: golden for pto.trowexpandadd (a+b).
# dst[r,c] = src0[r,c] a+b src1 broadcast; b = row broadcast (32,1)

import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import default_buffers, float_values, load_case_meta, matrix32, rng, single_output, write_buffers, write_golden


def main():
    meta = load_case_meta()
    src0_name, src1_name = meta.inputs
    generator = rng()
    src0 = float_values(generator, meta.elem_counts[src0_name], style='signed')
    src1 = float_values(generator, meta.elem_counts[src1_name], style='signed')
    src0_m = matrix32(src0)
    # The harness sizes src1's GM buffer by the valid column count (32), so
    # each row r gets exactly one scalar: standard row broadcast.
    phys = None
    buffers = default_buffers(meta)
    buffers[src0_name] = src0
    buffers[src1_name] = src1
    write_buffers(meta, buffers)
    out = src0_m + src1.astype(np.float32).reshape(-1)[:32][:, None]
    write_golden(meta, {single_output(meta): np.asarray(out, dtype=np.float32)})


if __name__ == '__main__':
    main()
