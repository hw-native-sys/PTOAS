# PyPTO tile-op baseline: golden for pto.tcolexpandexpdif (expdif).
# dst[r,c] = src0[r,c] expdif src1 broadcast; b = col broadcast (1,32)

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
    src0 = np.clip(float_values(generator, meta.elem_counts[src0_name], style='signed'), -3.0, 3.0)
    src1 = np.clip(float_values(generator, meta.elem_counts[src1_name], style='signed'), -3.0, 3.0)
    src0_m = matrix32(src0)
    scal = src1.astype(np.float32).reshape(-1)[:32]
    b = scal[None, :]   # col broadcast (1,32)
    buffers = default_buffers(meta)
    buffers[src0_name] = src0
    buffers[src1_name] = src1
    write_buffers(meta, buffers)
    out = np.exp(np.clip(src0_m - b, -20.0, 20.0))
    write_golden(meta, {single_output(meta): np.asarray(out, dtype=np.float32)})


if __name__ == '__main__':
    main()
