# Qwen3-32B 4-D decode (A5)

Generated from `models/qwen3_32b/decode_4d.py` at pypto-lib commit
`fcc141b7ac547259630e241a679d26fe31311eee`, using PyPTO commit
`508caf4e4f27190ce471e796028ea0eae3635843` and platform `a5`.

All 17 emitted PTO kernels compile with PTOAS level 3 for A5. This is kept
separate from `Qwen3DecodeA5` because the two programs exercise different
tensor layouts even when some kernel names overlap.
