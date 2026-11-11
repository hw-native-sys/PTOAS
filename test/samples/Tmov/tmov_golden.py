# PyPTO tile-op baseline: 
# golden for tmov multi-dtype kernel: both outputs are identity copies.
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
    generator = rng()
    buffers = default_buffers(meta)
    goldens = {}
    # 输入与输出按 kernel 参数顺序一一对应（a_f32->b_f32, a_i32->b_i32）
    for in_name, out_name in zip(meta.inputs, meta.outputs):
        if meta.np_types[in_name] == np.float32:
            src = float_values(generator, meta.elem_counts[in_name], style='signed')
        else:
            src = int_values(generator, meta.elem_counts[in_name], dtype=np.int32, style='bitwise')
        buffers[in_name] = src
        goldens[out_name] = np.asarray(src, dtype=meta.np_types[out_name])
    write_buffers(meta, buffers)
    write_golden(meta, goldens)


if __name__ == '__main__':
    main()
