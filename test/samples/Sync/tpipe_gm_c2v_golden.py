# PyPTO tile-op baseline: golden for tpipe_gm_c2v (pre-filled GM-slot identity).
# inputs[0] = slot content X (16x16 f32, random), rest zeroed.
# Golden = identity on X.
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
    data_name = meta.inputs[0]
    generator = rng()
    data = float_values(generator, meta.elem_counts[data_name], style='signed').astype(np.float32)
    buffers = default_buffers(meta)
    buffers[data_name] = data
    for other in meta.inputs[1:]:
        buffers[other] = np.zeros(meta.elem_counts[other], dtype=np.float32)
    write_buffers(meta, buffers)
    out = data.reshape(-1)[:16 * 16]
    out_name = single_output(meta)
    padded = np.zeros(meta.elem_counts[out_name], dtype=np.float32)
    padded[:out.size] = out
    write_golden(meta, {out_name: padded})


if __name__ == '__main__':
    main()
