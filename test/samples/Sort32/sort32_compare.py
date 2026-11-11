# PyPTO tile-op baseline: compare for pto.tsort32 (f32 values + u32 indices).
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
        dt = np.uint32 if name != meta.outputs[0] and 'idx' not in name and name.endswith('2') else None
    # 简化：第一个输出为 f32 值，第二个为 u32 索引
    outs = list(meta.outputs)
    for name, dt in zip(outs, [np.float32, np.uint32][:len(outs)]):
        ok = compare_file('golden_%s.bin' % name, '%s.bin' % name, dt, 0) and ok
    finalize_compare(ok)
