Qwen3-32B decode PTO kernels for A3, generated from `pypto-lib/models/qwen3_32b/decode.py` at `008a709dc9a57d11930390c12b1921e371e42ebb`, using `hw-native-sys/pypto` commit `93d789bf1727343efc48949581642bee47a683f9`.

Scope:
- compile-regression inputs for `ptoas`
- board-validation inputs with per-case custom golden

Notes:
- Export command: `pypto.ir.compile(build_qwen3_decode_program(), output_dir=..., skip_ptoas=True, dump_passes=False, platform="a2a3")`.
- `kernels/` is copied directly from the PyPTO output. The `aic/`, `aiv/`, filenames, and PTO contents are not flattened, renamed, or rewritten.
- The export contains 14 raw `.pto` files and all compile with `--pto-level=level3 --pto-arch=a3`.
- `runop.sh` defaults these cases to `--pto-level=level3`.
- `runop.sh` skips this directory on A5 / Ascend950 targets.
- Each raw kernel basename has a sibling `<case>_golden.py`; shared reference logic lives in `qwen3_decode_golden_lib.py`.
