DeepSeek V4 Flash MTP decode PTO kernels for A3, generated from `hw-native-sys/pypto-lib` `models/deepseek_v4_flash_mtp/decode_fwd.py` at commit `008a709dc9a57d11930390c12b1921e371e42ebb`, using `hw-native-sys/pypto` commit `93d789bf1727343efc48949581642bee47a683f9`.

Scope:
- compile-regression inputs for `ptoas`
- board-validation inputs with per-case custom golden wrappers

Notes:
- Export command: `python3 models/deepseek_v4_flash_mtp/decode_fwd.py --compile-only -p a2a3 --ep 2 --tp 2 -d 0,1` with `PYPTO_PROG_BUILD_DIR` set to an isolated output directory.
- This directory was regenerated from the source revisions above; it is not retained from an earlier DeepSeek export.
- At the pinned `pypto-lib` revision, the export writes 355 raw `.pto` fragments under `_jit_l3_decode_fwd_*/next_levels/decode_fwd/kernels/`, including 96 unsuffixed representative kernel families.
- This directory vendors the 87 representative families that compile with `--pto-level=level3 --pto-arch=a3`; the other 9 are explicitly accounted for below.
- `kernels/` preserves the PyPTO output hierarchy and byte-for-byte PTO files; no section-prefix flattening or kernel renaming is applied.
- Each raw kernel basename has a sibling `<case>_golden.py`; shared reference logic lives in `deepseek_v4_decode_golden_lib.py`.
- The shared helper provides independent CPU oracles for `kv_hadamard` and
  `rms_norm`. Cases without an independent oracle are reported as `OK` with
  `validation=determinism-only`; the runner summary still counts them
  separately from independent-golden correctness results.
- `runop.sh` defaults these cases to `--pto-level=level3` and skips the A3 directory on non-A3 targets.

Families excluded from the 96-family source snapshot (9 total):
- 5 families, `aiv/combine`, `aiv/dispatch_meta`, `aiv/dispatch_push`, `aiv/lm_head_combine_push`, and `aiv/lm_head_dispatch_push`, fail PTOAS memory-consistency validation because `CommRemoteOffset_*` helpers have not been inlined.
- 1 family, `aiv/csa_slots_build_valid_qk_plan`, did not finish PTOAS compilation within 60 seconds and is excluded to keep sample CI bounded.
- 3 families, `aiv/topk`, `aiv/route_sort`, and `aiv/lm_head_greedy_sample`, use mask-pattern `pto.tgather` without the required `axis` attribute in the direct PyPTO export.
