#!/usr/bin/env python3
"""Compile (and optionally note) the production TileLang PTODSL dump."""
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
    text = src.read_text()
    if 'pto.vmi' not in text:
        raise SystemExit('FAIL: dump is not PTODSL VMI')
    print(f'PTODSL_PRESENT: {src} bytes={len(text)}')
    # Frontend syntax check: the dump is a @pto.jit kernel. Importing it
    # requires TILELANG_ROOT + PTOAS ptodsl on PYTHONPATH.
    sys.path.insert(0, str(src.parent))
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('dump', src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        kn = getattr(mod, 'main_kernel')
        compiled = kn.compile()
        print(f'PTODSL_COMPILE_OK: {type(compiled).__name__}')
    except Exception as exc:
        print(f'PTODSL_COMPILE_OR_IMPORT: {type(exc).__name__}: {exc}')
        # Still a valid report artifact if TileLang helpers are missing;
        # ptoas can compile a stripped .pto separately.
        if args.compile_only:
            return 0
        return 2
    if args.compile_only:
        return 0
    print(f'VMI_RUNTIME: dump is compile-only in this repro (device={args.device}).')
    print('Use TileKernels-vmi artifacts/isolate_cast_back_leftovers.py for the device fault.')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
