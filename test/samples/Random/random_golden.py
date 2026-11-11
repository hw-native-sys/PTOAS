# PyPTO tile-op baseline: golden for pto.trandom (Philox-style generator).
# numpy reimplementation ported from test/tilelib-st/a5/trandom/case.py; the
# key/counter constants must stay in sync with random.py (KEY0/KEY1/COUNTER).

import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import load_case_meta, single_output, write_golden

ROWS, COLS = 4, 256
ROUNDS = 10
KEY0, KEY1 = 123456789, 987654321
COUNTER = (1, 2, 3, 4)

_M0 = np.uint64(0xD2511F53)
_M1 = np.uint64(0xCD9E8D57)
_K0 = np.uint32(0x9E3779B9)
_K1 = np.uint32(0xBB67AE85)


def _philox_round(ctr0, ctr1, ctr2, ctr3, key0, key1):
    product0 = ctr0.astype(np.uint64) * _M0
    product1 = ctr2.astype(np.uint64) * _M1
    low0 = product0.astype(np.uint32)
    high0 = (product0 >> 32).astype(np.uint32)
    low1 = product1.astype(np.uint32)
    high1 = (product1 >> 32).astype(np.uint32)
    return ((high1 ^ ctr1) ^ key0, low1, (high0 ^ ctr3) ^ key1, low0,
            (key0 + _K0).astype(np.uint32), (key1 + _K1).astype(np.uint32))


def _interleave(ctr0, ctr1, ctr2, ctr3):
    half = len(ctr0) // 2
    low0 = np.empty_like(ctr0)
    high0 = np.empty_like(ctr0)
    low1 = np.empty_like(ctr0)
    high1 = np.empty_like(ctr0)
    for i in range(half):
        low0[2 * i:2 * i + 2] = (ctr0[i], ctr2[i])
        high0[2 * i:2 * i + 2] = (ctr0[i + half], ctr2[i + half])
        low1[2 * i:2 * i + 2] = (ctr1[i], ctr3[i])
        high1[2 * i:2 * i + 2] = (ctr1[i + half], ctr3[i + half])
    result0 = np.empty_like(ctr0)
    result1 = np.empty_like(ctr0)
    result2 = np.empty_like(ctr0)
    result3 = np.empty_like(ctr0)
    for i in range(half):
        result0[2 * i:2 * i + 2] = (low0[i], low1[i])
        result1[2 * i:2 * i + 2] = (low0[i + half], low1[i + half])
        result2[2 * i:2 * i + 2] = (high0[i], high1[i])
        result3[2 * i:2 * i + 2] = (high0[i + half], high1[i + half])
    return result0, result1, result2, result3


def _add_with_128bits(ctr0, ctr1, ctr2, ctr3, value):
    """Mirror the legacy generator's lane-wise 128-bit counter addition."""
    ctr0_old = ctr0.copy()
    ctr0 = (ctr0.astype(np.uint64) + value.astype(np.uint64)).astype(np.uint32)
    carry = (ctr0.astype(np.uint64) < ctr0_old.astype(np.uint64)).astype(np.uint32)

    ctr1_old = ctr1.copy()
    ctr1 = (ctr1.astype(np.uint64) + carry.astype(np.uint64)).astype(np.uint32)
    carry = (ctr1.astype(np.uint64) < ctr1_old.astype(np.uint64)).astype(np.uint32)

    ctr2_old = ctr2.copy()
    ctr2 = (ctr2.astype(np.uint64) + carry.astype(np.uint64)).astype(np.uint32)
    carry = (ctr2.astype(np.uint64) < ctr2_old.astype(np.uint64)).astype(np.uint32)

    ctr3 = (ctr3.astype(np.uint64) + carry.astype(np.uint64)).astype(np.uint32)
    return ctr0, ctr1, ctr2, ctr3


def _golden():
    key0 = np.uint32(KEY0)
    key1 = np.uint32(KEY1)
    ctr0 = np.full(64, np.uint32(COUNTER[0]), dtype=np.uint32)
    ctr1 = np.full(64, np.uint32(COUNTER[1]), dtype=np.uint32)
    ctr2 = np.full(64, np.uint32(COUNTER[2]), dtype=np.uint32)
    ctr3 = np.full(64, np.uint32(COUNTER[3]), dtype=np.uint32)
    ctr0, ctr1, ctr2, ctr3 = _add_with_128bits(
        ctr0, ctr1, ctr2, ctr3, np.arange(64, dtype=np.uint32)
    )
    out = np.empty((ROWS, COLS), dtype=np.uint32)
    for row in range(ROWS):
        tmp0, tmp1, tmp2, tmp3 = ctr0.copy(), ctr1.copy(), ctr2.copy(), ctr3.copy()
        tmp_key0 = np.full(64, key0, dtype=np.uint32)
        tmp_key1 = np.full(64, key1, dtype=np.uint32)
        for _ in range(ROUNDS):
            tmp0, tmp1, tmp2, tmp3, tmp_key0, tmp_key1 = _philox_round(
                tmp0, tmp1, tmp2, tmp3, tmp_key0, tmp_key1
            )
        out[row] = np.concatenate(_interleave(tmp0, tmp1, tmp2, tmp3))
        ctr0, ctr1, ctr2, ctr3 = _add_with_128bits(
            ctr0, ctr1, ctr2, ctr3, np.full(64, 64, dtype=np.uint32)
        )
    return out


def main():
    meta = load_case_meta()
    out = _golden().view(np.int32)
    write_golden(meta, {single_output(meta): np.asarray(out, dtype=np.int32)})


if __name__ == '__main__':
    main()
