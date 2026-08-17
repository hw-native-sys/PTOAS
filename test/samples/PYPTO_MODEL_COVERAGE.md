# PyPTO model PTO coverage

This inventory tracks the non-draft model directories in
`hw-native-sys/pypto-lib/models` at commit
`fcc141b7ac547259630e241a679d26fe31311eee`. The exports use PyPTO commit
`508caf4e4f27190ce471e796028ea0eae3635843`.

The sample set contains one PTOAS-compilable representative per generated
kernel family. Repeated per-layer/per-rank files such as `foo_0.pto` are omitted
when the export also contains the unsuffixed `foo.pto`. Draft files are omitted,
matching pypto-lib CI.

| pypto-lib model | PTOAS sample coverage | Arch | Included families |
| --- | --- | --- | ---: |
| `qwen3_14b` | `Qwen3_14BDecodeA3`, `Qwen3_14BPrefillA3` | A3 | 19 + 21 |
| `qwen3_32b` | `Qwen3DecodeA5`, `Qwen3_32BDecode4DA5` | A5 | 14 + 17 |
| `deepseek_v3_2` | `DeepseekV3_2DecodeFrontA5`, `DeepseekV3_2DecodeBackA5`, `DeepseekV3_2PrefillBackA5` | A5 | 37 + 8 + 10 |
| `deepseek_v4_flash_dspark` | `DeepseekV4FlashDsparkA5` | A5 | 148 |
| `deepseek_v4_flash_mtp` | `DeepseekV4DecodeA5`, `DeepseekV4FlashMtpPrefillA5` | A5 | 93 + 44 |
| `deepseek_v4_pro` | `DeepseekV4ProDecodeA5`, `DeepseekV4ProPrefillA5` | A5 | 79 + 38 |

The A5 `deepseek`/`qwen` group therefore discovers 488 checked-in PTO cases.
Qwen3-14B decode is A3-only in the current upstream CLI, so its 40
representative cases are correctly selected by the A3 group rather than being
mislabeled as A5.

## Explicitly excluded source families

The following generated PTO files do not parse with the PTOAS revision used by
this branch and are not presented as runnable cases:

- `deepseek_v3_2`: `s3_topk_extract`
- `deepseek_v4_flash_dspark`: `topk`, `route_sort`,
  `lm_head_greedy_sample`, `prefill_idx_topk`
- `deepseek_v4_flash_mtp`: `topk`, `route_sort`,
  `lm_head_greedy_sample`, `prefill_idx_topk`
- `deepseek_v4_pro`: `topk`, `route_sort`, `prefill_idx_topk`

Board results use `SampleName/testcase` identifiers so identically named
kernels from different model versions remain distinguishable. Passing cases
without an independent CPU oracle are still reported as `OK`, with
`validation=determinism-only` in the result metadata; the runner summary keeps
separate independent-golden and determinism-only counts.
