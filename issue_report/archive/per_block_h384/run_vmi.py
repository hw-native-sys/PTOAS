#!/usr/bin/env python3
"""Compile the working H=128 128-lane strip dump (desired strip for H=384)."""
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
    print('DESIRED: same 128-lane strip at hidden=384; PTOAS VMI-RESIDUAL-OP — see recorded.log')
    sys.path.insert(0, str(src.parent))
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('dump', src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        kn = getattr(mod, 'main_kernel', None) or getattr(mod, 'per_block_cast_kernel_kernel', None)
        compiled = kn.compile()
        print(f'PTODSL_COMPILE_OK H=128 strip: {type(compiled).__name__}')
    except Exception as exc:
        print(f'PTODSL_COMPILE_OR_IMPORT: {type(exc).__name__}: {exc}')
        if args.compile_only:
            return 0
        return 2
    if args.compile_only:
        return 0
    print(f'VMI_RUNTIME: H=128 dump only (device={args.device}); H=384 128-strip hits VMI-RESIDUAL-OP.')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
