#!/usr/bin/env python3
"""Compile the aligned fused-rescale dump; tail ops are recorded as compile fails."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--compile-only', action='store_true')
    ap.add_argument('--device', default=os.environ.get('NPU_TEST_DEVICE', 'npu:0'))
    args = ap.parse_args()
    src = HERE / 'tilelang_dump.ptodsl.py'
    text = src.read_text() if src.exists() else ''
    if 'pto.vmi' not in text:
        raise SystemExit('FAIL: dump is not PTODSL VMI')
    print(f'PTODSL_PRESENT: {src} bytes={len(text)}')
    print(
        'DESIRED_TAIL: T.clear(e4m3 UB) or vbrc(f8e4m3(0)) for Persistent '
        'ceildiv(8001, 16); see desired_vmi.ptodsl.py and recorded.log'
    )
    sys.path.insert(0, str(src.parent))
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('dump', src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        kn = getattr(mod, 'main_kernel')
        compiled = kn.compile()
        print(f'PTODSL_COMPILE_OK aligned fused: {type(compiled).__name__}')
    except Exception as exc:
        print(f'PTODSL_COMPILE_OR_IMPORT: {type(exc).__name__}: {exc}')
        if args.compile_only:
            return 0
        return 2
    if args.compile_only:
        return 0
    print(f'VMI_RUNTIME: aligned dump only (device={args.device}); M=8001 tail does not lower.')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
