# PyPTO tile-op baseline: golden for tpow: base**exp, base kept positive to avoid NaN
# Raw-op builder + auto pipeline (samples convention; PyPTO lowers straight to pto.* IR).

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
    base_name, exp_name = meta.inputs
    generator = rng()
    base = np.abs(float_values(generator, meta.elem_counts[base_name], style='signed')) + 0.5
    exp = float_values(generator, meta.elem_counts[exp_name], style='signed')
    buffers = default_buffers(meta)
    buffers[base_name] = base
    buffers[exp_name] = exp
    write_buffers(meta, buffers)
    out = base ** exp
    write_golden(meta, {single_output(meta): np.asarray(out, dtype=np.float32)})


if __name__ == '__main__':
    main()
