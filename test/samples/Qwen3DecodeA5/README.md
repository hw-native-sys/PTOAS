Qwen3-32B decode PTO kernels for A5, generated from
`pypto-lib/models/qwen3_32b/decode.py` at
`fcc141b7ac547259630e241a679d26fe31311eee`, using `hw-native-sys/pypto`
commit `508caf4e4f27190ce471e796028ea0eae3635843`.

Scope:
- compile-regression inputs for `ptoas`
- board-validation inputs with per-case custom golden

Notes:
- Export command: `python3 models/qwen3_32b/decode.py -p a5sim -d 0` with
  `PYPTO_PROG_BUILD_DIR` set to an isolated output directory.
- `kernels/` contains the raw PyPTO `ptoas/` output without rewriting the PTO
  contents or kernel names.
- The export contains 14 raw `.pto` files and all compile with `--pto-level=level3 --pto-arch=a5`.
- `runop.sh` defaults these cases to `--pto-arch a5 --pto-level=level3`.
- `runop.sh` skips this directory on non-A5 / non-Ascend950 targets.
- Each raw kernel basename has a sibling `<case>_golden.py`; shared reference logic lives in `qwen3_decode_golden_lib.py`.
