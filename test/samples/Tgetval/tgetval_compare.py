# PyPTO tile-op baseline: 
# compare for tgetval: per-output dtype-aware, exact (identity roundtrip).
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
        ok = compare_file('golden_%s.bin' % name, '%s.bin' % name,
                          meta.np_types[name], 0.0) and ok
    finalize_compare(ok)
