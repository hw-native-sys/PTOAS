# DeepSeek V4 Flash MTP prefill (A5)

Generated from `models/deepseek_v4_flash_mtp/prefill_fwd.py` at pypto-lib
commit `fcc141b7ac547259630e241a679d26fe31311eee`, using PyPTO commit
`508caf4e4f27190ce471e796028ea0eae3635843` with `--compile-only -p a5
--ep 2 --tp 2`.

The full export has 104 representative families. Fifty-nine are already
covered by `DeepseekV4DecodeA5`; this directory adds 44 of the 45 prefill-only
families that compile with PTOAS level 3. `prefill_idx_topk` is explicitly
excluded in `../PYPTO_MODEL_COVERAGE.md`.
