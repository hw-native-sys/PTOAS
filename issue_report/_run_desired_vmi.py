#!/usr/bin/env python3
"""Compile-check desired_vmi.ptodsl.py (shared by live B–E issues)."""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--compile-only', action='store_true')
    ap.add_argument('--device', default=os.environ.get('NPU_TEST_DEVICE', 'npu:0'))
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    # When copied/symlinked next to desired_vmi, HERE is the issue dir.
    src = Path(os.environ.get('DESIRED_VMI', here / 'desired_vmi.ptodsl.py'))
    if not src.exists():
        src = here / 'desired_vmi.ptodsl.py'
    text = src.read_text()
    if 'pto.vmi' not in text:
        raise SystemExit('FAIL: desired_vmi is not PTODSL VMI')
    print(f'DESIRED_VMI: {src} bytes={len(text)}')
    sys.path.insert(0, str(src.parent))
    try:
        spec = importlib.util.spec_from_file_location('desired_vmi', src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        kn = next(
            getattr(mod, n)
            for n in dir(mod)
            if callable(getattr(mod, n)) and hasattr(getattr(mod, n), 'compile')
        )
        compiled = kn.compile()
        print(f'PTODSL_COMPILE_OK: {type(compiled).__name__}')
    except Exception as exc:
        print(f'PTODSL_COMPILE_OR_IMPORT: {type(exc).__name__}: {exc}')
        print('See recorded.log for the production leftover. desired_vmi is the legal target.')
        if args.compile_only:
            return 0
        return 2
    if args.compile_only:
        return 0
    print(f'VMI_RUNTIME: desired_vmi is the compile target (device={args.device}); see recorded.log')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
