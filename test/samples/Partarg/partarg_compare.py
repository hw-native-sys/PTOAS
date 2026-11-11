# PyPTO tile-op baseline: compare for the Partarg builder (tpartargmax/tpartargmin).
# Per-output dtype-aware: value outputs f32, index outputs int.

import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import compare_file, finalize_compare, load_case_meta

if __name__ == '__main__':
    meta = load_case_meta()
    ok = True
    for name in meta.outputs:
        # Outputs carry generic vN names in the board harness; take the dtype
        # from the kernel signature (main.cpp) instead of the buffer name.
        out_dt = meta.np_types[name]
        atol = 0.0 if np.issubdtype(out_dt, np.integer) else 1e-5
        ok = compare_file('golden_%s.bin' % name, '%s.bin' % name, out_dt, atol) and ok
    finalize_compare(ok)
