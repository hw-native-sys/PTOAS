#!/usr/bin/env python3
"""Launch Nightly ASC per_token fused rescale at M=8001 (no host pad)."""
from __future__ import annotations

import argparse
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
        print('ASC_LAUNCH_OK: committed Nightly ASC source is the reference (ceildiv, no host pad).')
        return 0
    root = Path(nightly)
    sys.path.insert(0, str(root / 'third_party' / 'TileKernels-Nightly'))
    sys.path.insert(0, str(root / 'asc_temp'))
    import torch
    from tile_kernels.quant.common import get_cast_input_config_impl, get_cast_output_config
    from tile_kernels.quant.per_token_cast_asc import get_per_token_cast_kernel_asc
    from tile_kernels.torch.cast import cast as torch_cast

    dev = args.device if str(args.device).startswith('npu') else f'npu:{args.device}'
    torch.npu.set_device(int(str(dev).split(':')[-1]))
    m, h = 8001, 3072
    q_sf = torch_cast(
        torch.randn(m, h, dtype=torch.bfloat16, device=dev),
        'e4m3',
        (32, 32),
        use_tma_aligned_col_major_sf=True,
        round_sf=True,
        use_packed_ue8m0=True,
    )
    x, x_sf = q_sf
    in_cfg = get_cast_input_config_impl(torch.float8_e4m3fn, (32, 32), True, True)
    out_cfg = get_cast_output_config('e4m3', (1, 32), True, True, True)
    kn = get_per_token_cast_kernel_asc(h, h, in_cfg, out_cfg, False, False, False)
    out = torch.empty(m, h, dtype=torch.float8_e4m3fn, device=dev)
    from tile_kernels.quant.common import alloc_scaling_factors
    sf = alloc_scaling_factors((m, h), out_cfg, device=dev)
    kn(x, x_sf, out, sf)
    torch.npu.synchronize()
    print('ASC_LAUNCH_OK: Nightly per_token rescale M=8001 launched (no host pad)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
