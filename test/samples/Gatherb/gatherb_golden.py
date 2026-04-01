#!/usr/bin/python3
import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import (
    default_buffers,
    float_values,
    is_a5_soc,
    load_case_meta,
    rng,
    single_output,
    write_buffers,
    write_golden,
)


def main():
    meta = load_case_meta()
    src_name, offset_name = meta.inputs
    out_name = single_output(meta)
    generator = rng()
    src_dtype = meta.np_types[src_name]
    n_src = meta.elem_counts[src_name]
    n_offset = meta.elem_counts[offset_name]
    n_out = meta.elem_counts[out_name]

    # Generate sequential source data (easy to verify gather results).
    src = np.arange(n_src, dtype=np.float32).astype(src_dtype, copy=False)

    # TGATHERB offsets are byte addresses into the source buffer.
    # Generate offsets that stay within the source buffer bounds.
    offsets = (np.arange(n_offset, dtype=np.uint32) * 32).astype(
        meta.np_types[offset_name], copy=False
    )
    # Clamp offsets to valid range.
    max_byte_offset = max((n_src * src_dtype.itemsize) - 32, 0)
    offsets = np.minimum(offsets, max_byte_offset)

    buffers = default_buffers(meta)
    buffers[src_name] = src
    buffers[offset_name] = offsets
    write_buffers(meta, buffers)

    # Compute expected output: TGATHERB reads 32 bytes per offset entry.
    block_elems = 32 // src_dtype.itemsize
    expected = np.zeros(n_out, dtype=src_dtype)
    for i, off in enumerate(offsets.reshape(-1)):
        byte_off = int(off)
        elem_off = byte_off // src_dtype.itemsize
        end = min(elem_off + block_elems, n_src)
        actual_block = end - elem_off
        dst_base = i * block_elems
        if dst_base + actual_block <= n_out:
            expected[dst_base:dst_base + actual_block] = src[elem_off:end]

    write_golden(meta, {out_name: np.asarray(expected, dtype=src_dtype)})


if __name__ == '__main__':
    main()
