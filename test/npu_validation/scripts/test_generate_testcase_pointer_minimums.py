#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import re
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "generate_testcase",
    SCRIPT_DIR / "generate_testcase.py",
)
GENERATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GENERATOR)


A5_BOARD_FAILURE_MINIMUMS = {
    ("Qwen3DecodeA5", "sv_matmul"): (262_144, 524_288, 67_108_864),
    ("Qwen3_32BDecode4DA5", "down_proj"): (409_600, 209_715_200, 131_072),
    ("Qwen3_32BDecode4DA5", "k_proj"): (131_072, 8_388_608, 16_384),
    ("Qwen3_32BDecode4DA5", "out_proj"): (131_072, 67_108_864, 131_072),
    ("Qwen3_32BDecode4DA5", "v_proj"): (131_072, 8_388_608, 16_384),
    ("DeepseekV4ProPrefillA5", "rope"): (8_388_608, 1_048_576, 8_192, 8_192),
    ("DeepseekV4ProPrefillA5", "prefill_idx_qr_proj"): (8_192, 1_048_576, 196_608, 12_582_912, 128),
    ("DeepseekV4ProDecodeA5", "hca_cache_topk"): (8, 4, 1_024),
    ("DeepseekV4FlashMtpPrefillA5", "merge_rope_pack"): (
        1_024, 1_024, 8_192, 8_192, 4_194_304, 4_194_304, 1, 1, 512, 64,
    ),
    ("DeepseekV4FlashMtpPrefillA5", "prefill_idx_c4_rmsnorm_rope"): (
        32, 32, 1_048_576, 1_048_576, 4_096, 128, 4_096,
    ),
    ("DeepseekV4FlashMtpPrefillA5", "prefill_idx_qr_proj"): (
        131_072, 8_388_608, 8_192, 1_048_576, 128,
    ),
    ("DeepseekV4FlashDsparkA5", "gather_ori_kv"): (25_165_824, 16_384, 512),
    ("DeepseekV4FlashDsparkA5", "lm_head_combine_gather"): (16_547_840, 16_547_840),
    ("DeepseekV4FlashDsparkA5", "merge_rope_pack"): (
        1_024, 1_024, 8_192, 8_192, 4_194_304, 4_194_304, 1, 1, 512, 64,
    ),
    ("DeepseekV4FlashDsparkA5", "prefill_idx_qr_proj"): (
        131_072, 8_388_608, 8_192, 1_048_576, 128,
    ),
    ("DeepseekV4DecodeA5", "hca_rope"): (128, 128, 512, 512, 8, 1_048_576, 1_048_576),
    ("DeepseekV3_2PrefillBackA5", "deepseek_v3_2_prefill_back_layer_incore_1"): (
        1_048_576, 117_440_512, 458_752,
    ),
    ("DeepseekV3_2PrefillBackA5", "deepseek_v3_2_prefill_back_layer_incore_5"): (
        458_752, 132_120_576, 8_192,
    ),
    ("DeepseekV3_2PrefillBackA5", "deepseek_v3_2_prefill_back_layer_incore_6"): (
        458_752, 132_120_576, 8_192,
    ),
    ("DeepseekV3_2PrefillBackA5", "deepseek_v3_2_prefill_back_layer_incore_8"): (
        1_179_648, 132_120_576, 8_192,
    ),
    ("DeepseekV3_2DecodeFrontA5", "decode_cache_write"): (
        16, 262_144, 262_144, 8_192, 9_216, 33_554_432, 4_194_304,
    ),
    ("DeepseekV3_2DecodeFrontA5", "kv_a_proj"): (114_688, 4_128_768, 1_024),
    ("DeepseekV3_2DecodeFrontA5", "q_head_proj"): (24_576, 37_748_736, 1_024),
    ("DeepseekV3_2DecodeFrontA5", "q_lora_proj"): (114_688, 11_010_048, 24_576),
    ("DeepseekV3_2DecodeFrontA5", "s2_k_idx_proj"): (114_688, 917_504, 1_024),
    ("DeepseekV3_2DecodeFrontA5", "s2_q_idx_proj"): (24_576, 12_582_912, 2_048),
    ("DeepseekV3_2DecodeFrontA5", "s4_dispatch"): (1, 33_554_432, 262_144),
    ("DeepseekV3_2DecodeFrontA5", "s4_softmax"): (
        32_768, 33_554_432, 4_194_304, 8_192, 1_024, 8_192,
    ),
    ("DeepseekV3_2DecodeBackA5", "deepseek_v3_2_decode_back_layer_incore_0"): (
        262_144, 117_440_512, 1_024,
    ),
    ("DeepseekV3_2DecodeBackA5", "deepseek_v3_2_decode_back_layer_incore_3"): (
        114_688, 132_120_576, 4_096,
    ),
    ("DeepseekV3_2DecodeBackA5", "deepseek_v3_2_decode_back_layer_incore_4"): (
        114_688, 132_120_576, 4_096,
    ),
    ("DeepseekV3_2DecodeBackA5", "deepseek_v3_2_decode_back_layer_incore_6"): (
        294_912, 132_120_576, 2_048,
    ),
}

A3_BOARD_FAILURE_MINIMUMS = {
    ("DeepseekV4DecodeA3", "gate"): (64, 65_536, 262_144, 16, 4_096, 2_048),
    ("DeepseekV4DecodeA3", "qk_pv"): (
        1_024, 524_288, 512, 512, 262_144, 262_144, 32_768,
    ),
    ("DeepseekV4DecodeA3", "score"): (
        4, 8, 65_536, 512, 1_024, 32_768, 256, 16_384, 128, 16_384,
    ),
    ("Qwen3_14BPrefillA3", "lm_head"): (152_064, 81_920, 778_567_680),
    ("Qwen3_14BPrefillA3", "qk_pv_online_phase"): (
        8_192, 8_192, 1_048_576, 1, 131_072, 131_072, 1_048_576, 16_384,
    ),
    ("Qwen3_14BPrefillA3", "qk_pv_skew_probe"): (
        8_192, 1_048_576, 1, 131_072, 131_072, 1_048_576, 327_680, 32_768,
    ),
}

A3_MIXED_SCALAR_DEFAULTS = {
    ("DeepseekV4DecodeA3", "gate"): {"v7": 0, "v8": 1},
    ("DeepseekV4DecodeA3", "qk_pv"): {"v8": 0, "v9": 1},
    ("DeepseekV4DecodeA3", "score"): {"v11": 1, "v12": 0, "v13": 1},
    ("Qwen3_14BPrefillA3", "qk_pv_online_phase"): {
        "v9": 1,
        "v10": 0,
        "v11": 0,
        "v12": 1,
        "v13": 0,
        "v14": 0,
        "v15": 0,
        "v16": 0,
        "v17": 1,
        "v18": 128,
        "v19": 0,
        "v20": 1,
    },
    ("Qwen3_14BPrefillA3", "qk_pv_skew_probe"): {
        "v9": 0,
        "v10": 0,
        "v11": 1,
        "v12": 0,
        "v13": 0,
        "v14": 0,
        "v15": 1,
        "v16": 128,
        "v17": 0,
        "v18": 1,
    },
}


def _generate_counts(root: Path, sample: str, testcase: str, expected_count: int) -> tuple[int, ...]:
    sample_dir = root / sample
    sample_dir.mkdir(parents=True, exist_ok=True)
    params = ", ".join(f"__gm__ float* v{index}" for index in range(1, expected_count + 1))
    kernel = sample_dir / f"{testcase}-pto.cpp"
    kernel.write_text(
        f'extern "C" __global__ AICORE void {testcase}({params}) {{}}\n',
        encoding="utf-8",
    )
    output_root = root / "generated"
    GENERATOR.generate_testcase(kernel, output_root, testcase, "npu", "Ascend950")
    main_cpp = (output_root / sample / testcase / "main.cpp").read_text(encoding="utf-8")
    return tuple(
        int(re.search(rf"elemCount_v{index} = (\d+);", main_cpp).group(1))
        for index in range(1, expected_count + 1)
    )


def main() -> None:
    assert len(A5_BOARD_FAILURE_MINIMUMS) == 32
    with tempfile.TemporaryDirectory(prefix="ptoas-npu-validation-test-") as temp_dir:
        root = Path(temp_dir)
        for (sample, testcase), expected in A5_BOARD_FAILURE_MINIMUMS.items():
            actual = _generate_counts(root, sample, testcase, len(expected))
            assert actual == expected, f"{sample}/{testcase}: expected {expected}, got {actual}"

        for (sample, testcase), expected in A3_BOARD_FAILURE_MINIMUMS.items():
            actual = _generate_counts(root, sample, testcase, len(expected))
            assert actual == expected, f"{sample}/{testcase}: expected {expected}, got {actual}"

        for (sample, testcase), expected in A3_MIXED_SCALAR_DEFAULTS.items():
            actual = {
                name: GENERATOR._integer_scalar_default_value(
                    testcase,
                    name,
                    "int64_t" if name in {"v11", "v13", "v17", "v18"} else "int32_t",
                    sample,
                )
                for name in expected
            }
            assert actual == expected, f"{sample}/{testcase}: expected {expected}, got {actual}"

        isolated = _generate_counts(root, "UnrelatedSample", "sv_matmul", 3)
        assert isolated == (1_024, 1_024, 1_024), isolated


if __name__ == "__main__":
    main()
