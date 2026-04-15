Qwen3 decode PTO kernels for A3, generated from `pypto-lib/examples/models/qwen3/qwen3_32b_decode.py`.

Scope:
- compile-regression inputs for `ptoas`
- board-validation inputs with per-case custom golden

Notes:
- This directory vendors the 17 emitted `qwen3_decode_incore_*.pto` fragments for the A3 lowering.
- `runop.sh` defaults these cases to `--pto-level=level3`.
- `runop.sh` skips this directory on A5 / Ascend950 targets.
- Each case has a sibling `<case>_golden.py`; shared reference logic lives in `qwen3_decode_golden_lib.py`.
