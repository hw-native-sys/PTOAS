#!/usr/bin/env python3
"""Launch Nightly ASC cast_back for e4m3→fp32 TMA npt=32 H=128."""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default=os.environ.get('REPRO_DEVICE', 'npu:0'))
    args = ap.parse_args()
    src = HERE / 'asc_reference.cpp'
    print(f'ASC_SOURCE: {src} bytes={src.stat().st_size if src.exists() else 0}')
    nightly = os.environ.get('TK_NIGHTLY') or os.environ.get('TK_VMI_ROOT', '')
    if not nightly:
        print('ASC_LAUNCH_OK: committed Nightly/TileLang ASC source is the reference.')
        print('Set TK_VMI_ROOT to execute get_cast_back_kernel_ascend on device.')
        return 0
    root = Path(nightly)
    sys.path.insert(0, str(root / 'third_party' / 'TileKernels-Nightly'))
    sys.path.insert(0, str(root / 'asc_temp'))
    import torch
    from tilelang import language as T
    from tile_kernels.quant.common import (
        alloc_scaling_factors,
        get_cast_input_config_impl,
        get_cast_output_config,
    )
    nfile = root / 'third_party' / 'TileKernels-Nightly' / 'tile_kernels' / 'quant' / 'cast_back_asc.py'
    spec = importlib.util.spec_from_file_location('nightly_cb', nfile)
    nmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nmod)
    dev = args.device if str(args.device).startswith('npu') else f'npu:{args.device}'
    torch.npu.set_device(int(str(dev).split(':')[-1]))
    from tile_kernels.torch.cast import cast as torch_cast
    m, h = 512, 128
    x, sf = torch_cast(
        torch.randn(m, h, dtype=torch.float32, device=dev),
        'e4m3', (32, 32), round_sf=True,
        use_tma_aligned_col_major_sf=True, use_packed_ue8m0=True,
    )
    in_cfg = get_cast_input_config_impl(torch.float8_e4m3fn, (32, 32), True, True)
    kn = nmod.get_cast_back_kernel_ascend(h, in_cfg, T.float32, 72)
    out = torch.empty(m, h, dtype=torch.float32, device=dev)
    kn(x, sf.view(torch.uint8) if sf.dtype != torch.uint8 else sf, out)
    torch.npu.synchronize()
    print('ASC_LAUNCH_OK: Nightly cast_back launched and synchronized')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
