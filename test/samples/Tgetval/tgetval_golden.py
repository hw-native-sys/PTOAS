# PyPTO tile-op baseline: 
# golden for tgetval: 1x3 outputs holding first/middle/last source elements.
import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import default_buffers, float_values, int_values, load_case_meta, rng, write_buffers, write_golden

OFFSETS = [0, 8 * 32 // 2 + 7, 8 * 32 - 1]


def main():
    meta = load_case_meta()
    generator = rng()
    buffers = default_buffers(meta)
    goldens = {}
    for in_name, out_name in zip(meta.inputs, meta.outputs):
        if meta.np_types[in_name] == np.float32:
            src = float_values(generator, meta.elem_counts[in_name], style='signed')
        else:
            src = int_values(generator, meta.elem_counts[in_name], dtype=np.int32, style='bitwise')
        flat = src.reshape(-1)
        buffers[in_name] = src
        # 输出 GM 缓冲按 harness 尺寸分配（与输入同尺寸），getval 只写前 3 个
        # 元素，其余保持零初始化——golden 按完整尺寸零填充。
        out = np.zeros(meta.elem_counts[out_name], dtype=meta.np_types[out_name])
        out.reshape(-1)[:len(OFFSETS)] = [flat[o] for o in OFFSETS]
        goldens[out_name] = out
    write_buffers(meta, buffers)
    write_golden(meta, goldens)


if __name__ == '__main__':
    main()
