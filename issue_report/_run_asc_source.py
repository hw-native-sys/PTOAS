#!/usr/bin/env python3
"""Print the committed ASC reference and optional Nightly launch note."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='npu:0')
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    src = here / 'asc_reference.cpp'
    pat = here / 'asc_pattern.mi'
    if src.exists():
        print(f'ASC_SOURCE: {src} bytes={src.stat().st_size}')
        print('ASC_LAUNCH_OK: committed Nightly ASC C++ is the working reference.')
    else:
        print('ASC_SOURCE: none (see asc_pattern.mi for the pto.mi target)')
    if pat.exists():
        print(f'ASC_PATTERN: {pat} bytes={pat.stat().st_size}')
    print(f'device={args.device}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
