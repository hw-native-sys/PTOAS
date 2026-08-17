DeepSeek V4 Flash MTP decode PTO kernels for A5, generated from
`hw-native-sys/pypto-lib` `models/deepseek_v4_flash_mtp/decode_fwd.py` at
commit `fcc141b7ac547259630e241a679d26fe31311eee`, using `hw-native-sys/pypto`
commit `508caf4e4f27190ce471e796028ea0eae3635843`.

Scope:
- compile-regression inputs for `ptoas`
- board-validation inputs with custom golden wrappers where available

Notes:
- Export command: `python3 models/deepseek_v4_flash_mtp/decode_fwd.py --compile-only -p a5 --ep 2 --tp 2 -d 0,1` with `PYPTO_PROG_BUILD_DIR` set to an isolated output directory.
- This directory was regenerated from the source revisions above; it is not retained from an earlier DeepSeek export.
- At the pinned `pypto-lib` revision, the export writes 355 raw `.pto`
  fragments, including 96 unsuffixed representative kernel families.
- This directory vendors the 93 representative families that compile with
  `--pto-level=level3 --pto-arch=a5`; the other 3 are explicitly accounted for
  below.
- `kernels/` preserves the PyPTO output hierarchy and byte-for-byte PTO files; no section-prefix flattening or kernel renaming is applied.
- Existing siblings named `<case>_golden.py` provide custom validation; newly
  accepted helper families use the generic deterministic validator. Shared
  reference logic lives in `deepseek_v4_decode_golden_lib.py`.
- The shared helper provides independent CPU oracles for `kv_hadamard` and
  `rms_norm`. Cases without an independent oracle are reported as `OK` with
  `validation=determinism-only`; the runner summary still counts them
  separately from independent-golden correctness results.
- `runop.sh` defaults these cases to `--pto-level=level3` and skips the A5 directory on non-A5 targets.

Families excluded from the 96-family source snapshot (3 total):
- `aiv/topk`, `aiv/route_sort`, and `aiv/lm_head_greedy_sample` do not parse in
  the current PTOAS input pipeline.
