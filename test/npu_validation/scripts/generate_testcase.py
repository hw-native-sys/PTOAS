#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import argparse
import ast
import os
import re
import shutil
from pathlib import Path
from typing import Optional

INCLUDE_REPLACEMENT = (
    "// ---------------------------------------------------------------------------\n"
    "// PTOAS compatibility layer\n"
    "//\n"
    "// The upstream pto-isa headers reference some FP8/FP4 types and the\n"
    "// __VEC_SCOPE__ marker that are not available on every AICore arch/toolchain\n"
    "// combination (e.g. __NPU_ARCH__==2201).\n"
    "//\n"
    "// For our PTOAS-generated kernels we don't rely on these types today, but the\n"
    "// headers still mention them in templates/static_asserts. Provide minimal\n"
    "// fallbacks to keep compilation working on dav-c220.\n"
    "// ---------------------------------------------------------------------------\n"
    "#ifndef __VEC_SCOPE__\n"
    "#define __VEC_SCOPE__\n"
    "#endif\n"
    "\n"
    "// Only provide low-precision fallbacks when bisheng does not already\n"
    "// expose them through its built-in __clang_cce_types.h header.\n"
    "#if defined(__CCE_AICORE__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201) && !__has_include(<__clang_cce_types.h>)\n"
    "typedef struct { unsigned char v; } hifloat8_t;\n"
    "typedef struct { unsigned char v; } float8_e4m3_t;\n"
    "typedef struct { unsigned char v; } float8_e5m2_t;\n"
    "typedef struct { unsigned char v; } float8_e8m0_t;\n"
    "typedef struct { unsigned char v; } float4_e1m2x2_t;\n"
    "typedef struct { unsigned char v; } float4_e2m1x2_t;\n"
    "#endif\n"
    "#include <stdint.h>\n"
    "\n"
    "// AICore printf support is gated behind `--cce-enable-print` on some\n"
    "// toolchains. When enabled, include the CCE print header so `cce::printf`\n"
    "// resolves in device compilation.\n"
    "#if defined(__CCE_AICORE__) && defined(PTOAS_ENABLE_CCE_PRINT)\n"
    "#include <ccelib/print/print.h>\n"
    "#endif\n"
    "#include <pto/pto-inst.hpp>\n"
    "#include <pto/common/constants.hpp>\n"
    "\n"
    "// Some PTO-ISA types are only available in the __CCE_AICORE__ compilation\n"
    "// path, but `bisheng -xcce` still performs a host-side parse pass.\n"
    "// Provide minimal fallbacks only when the corresponding header wasn't\n"
    "// pulled in by the selected arch implementation.\n"
    "// CANN 9.2.0+ defines MrgSortExecutedNumList in pto/common/type.hpp\n"
    "// without defining TMRGSORT_HPP, so detect the providing header directly.\n"
    "#if !defined(__CCE_AICORE__) && !__has_include(<pto/common/type.hpp>)\n"
    "namespace pto {\n"
    "struct MrgSortExecutedNumList {\n"
    "    uint16_t mrgSortList0;\n"
    "    uint16_t mrgSortList1;\n"
    "    uint16_t mrgSortList2;\n"
    "    uint16_t mrgSortList3;\n"
    "};\n"
    "} // namespace pto\n"
    "#endif\n"
    "#ifndef __CPU_SIM\n"
    "#include \"acl/acl.h\"\n"
    "#endif\n"
)

UNSTABLE_A3_CUSTOM_GOLDEN_CASES = frozenset({
    "abs",
    "partmin",
    "prelu",
    "rowexpanddiv",
    "rowexpandmul",
    "rowexpandsub",
    "scatter",
    "sel",
    "sels",
    "sub",
    "xor",
})

DEEPSEEK_V4_DIRECT_CASES = frozenset({
    "attention_csa_test_refresh_incore_81",
    "attention_hca_test_incore_54",
    "attention_swa_test_incore_40",
    "decode_csa_test_incore_81",
    "decode_hca_test_incore_54",
    "decode_swa_test_incore_40",
    "sparse_attn_test_incore_7",
})

CASE_INT_SCALAR_DEFAULTS = {
    testcase: {
        "v4": 0,
        "v5": 32,
    }
    for testcase in DEEPSEEK_V4_DIRECT_CASES
}

CASE_INT_SCALAR_DEFAULTS["hc_head_reduce"] = {"v9": 0, "v10": 1}
CASE_INT_SCALAR_DEFAULTS["hc_post"] = {
    "v6": 8,
    "v7": 8,
    "v8": 0,
    "v9": 1,
}
CASE_INT_SCALAR_DEFAULTS["qr_hadamard_quant"] = {"v4": 0, "v5": 1}
CASE_INT_SCALAR_DEFAULTS["rms_norm"] = {"v4": 0, "v5": 1}
CASE_INT_SCALAR_DEFAULTS["swa_gather_kv"] = {"v4": 1, "v5": 0, "v6": 1}

# Qwen3 decode exports use compact per-block inputs.  The generic fallback of
# `1` for every dynamic integer selects the second batch/pair and makes the
# golden builders address beyond those compact buffers.  Keep one context
# block for the attention reductions, but select the first batch/block/pair.
CASE_INT_SCALAR_DEFAULTS.update({
    "down_proj_residual": {"v6": 0, "v7": 0},
    "gate_proj": {"v4": 0, "v5": 0, "v6": 0},
    "kv_proj": {"v6": 0, "v7": 0},
    "online_softmax": {"v6": 0},
    "out_proj_residual": {"v6": 0, "v7": 0},
    "q_proj": {"v4": 0, "v5": 0},
    "qk_matmul": {"v4": 0, "v6": 0},
    "rope_kv_cache": {"v11": 0, "v12": 1},
    "silu": {"v4": 0, "v5": 0, "v6": 0},
    "softmax": {"v7": 0, "v8": 0},
    "sv_matmul": {"v5": 0, "v6": 0},
    "up_proj": {"v4": 0, "v5": 0, "v6": 0},
})

# PyPTO model exports pass the SPMD block index and block count as explicit
# trailing scalar arguments.  The generated validation harness launches one
# worker, so the representative invocation is always block 0 of 1.  Keep the
# overrides sample-scoped because identically named kernels in other model
# revisions can have different scalar signatures.
SAMPLE_CASE_INT_SCALAR_DEFAULTS = {
    "qwen3_14bdecodea3": {
        # The exported projection kernels take tensor-view offsets followed by
        # the PyPTO SPMD worker id/count.  Exercise the first valid view with a
        # single worker; the generic value `1` selects a shifted weight block.
        "k_proj": {"v4": 0, "v5": 0, "v6": 1},
        "v_proj": {"v4": 0, "v5": 0, "v6": 1},
        "q_proj": {"v4": 0, "v5": 0, "v6": 1},
        "down_proj": {"v4": 0, "v5": 0, "v6": 0},
        "out_proj": {"v4": 0, "v5": 0, "v6": 0},
    },
    "qwen3_14bprefilla3": {
        # Dynamic tensor dimensions must describe the backing views, while
        # offset and SPMD-index arguments select the first validation tile.
        "gate_up_proj": {
            "v6": 0,
            "v7": 0,
            "v8": 0,
            "v9": 128,
            "v10": 5_120,
            "v11": 0,
            "v12": 1,
        },
        "kv_proj": {"v6": 0, "v7": 5_120, "v8": 0, "v9": 1},
        "out_proj_aic": {"v4": 0, "v5": 5_120, "v6": 0, "v7": 1},
        # The AIV projection uses one 64-row output tile.  Keep validation on
        # the first row/block; the generic scalar fallback of 1 selects a
        # shifted window and leaves the runtime output nondeterministic.
        "out_proj_aiv": {"v4": 0, "v5": 64, "v6": 0, "v7": 1},
        "q_proj": {"v4": 0, "v5": 5_120, "v6": 0, "v7": 1},
        "lm_head": {"v4": 1, "v5": 1, "v6": 0, "v7": 1},
        "qk_pv_skew_probe": {
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
        "qk_pv_online_phase": {
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
    },
    "deepseekv4decodea3": {
        "qk_pv": {"v8": 0, "v9": 1},
        "score": {"v11": 1, "v12": 0, "v13": 1},
        "gate": {"v7": 0, "v8": 1},
    },
    "deepseekv4decodea5": {
        "idx_qr_proj_matmul": {"v4": 0, "v5": 1},
        # Exercise one complete 16-row tile instead of truncating arg5 / 16
        # to zero, and select the first SPMD worker.
        "kv_proj_matmul": {"v6": 16, "v7": 16, "v8": 0, "v9": 1},
    },
    "deepseekv4prodecodea5": {
        "idx_qr_proj_matmul": {"v4": 0, "v5": 1},
        "kv_score_proj": {"v8": 0, "v9": 1},
    },
    "deepseekv4flashdsparka5": {
        "kv_proj_matmul": {"v8": 0, "v9": 1},
        "qproj_matmul": {"v6": 0, "v7": 1},
        "qr_proj_matmul": {"v8": 0, "v9": 1},
        "lm_head_combine_gather": {"v4": 0, "v5": 1},
        "rmsnorm_rope": {"v7": 0, "v8": 1},
    },
}

# The generated EmitC for these atomic-add matmuls aliases the destination
# tensor through intermediate GlobalTensor objects.  The generic writer scan
# can lose that alias and fall back to the final pointer (the weight tensor),
# which makes determinism validation compare an unchanged input.  Pin the
# actual PTO destination parameter for the affected sample revisions.
SAMPLE_CASE_OUTPUT_PARAMS = {
    "tpipe": {
        "a3_pipe_roundtrip": ["v2"],
    },
    "deepseekv4decodea5": {
        "kv_proj_matmul": ["v1"],
    },
    "deepseekv4flashdsparka5": {
        "kv_proj_matmul": ["v1"],
        "qr_proj_matmul": ["v1"],
    },
}

CASE_BOOL_SCALAR_DEFAULTS = {}

CASE_POINTER_COUNT_MINIMUMS = {
    # These PyPTO kernels use the explicit SPMD block index to select GM
    # subviews.  The EmitC footprint analysis sees one tile at a time but does
    # not prove all loop iterations or the enclosing block offset, which left
    # the generated A5 harness buffers smaller than the accessed tensor views
    # and caused exit=139 during board execution.  Use the complete PTO tensor
    # view sizes for the single-worker validation defaults above.
    "hc_post": {
        "v1": 8 * 16_384,
        "v2": 8 * 4,
        "v3": 8 * 4_096,
        "v4": 8 * 16,
        "v5": 8 * 16_384,
    },
    "kv_hadamard": {"v1": 2_048, "v2": 2_048, "v3": 16_384},
    "merge_norm": {
        # The packed sparse output is written through a large tiled GM address
        # space even though each individual store only covers 448 bf16 values.
        # Keep the backing buffer large enough for the maximum `v7=1` testcase
        # offset emitted by the vendored DeepSeek V4 A3 sample.
        "v2": 4_198_400,
    },
    "down_proj_residual": {
        "v1": 123648,
        "v2": 123648,
    },
    "out_proj_residual": {
        "v1": 123648,
        "v2": 123648,
    },
    # These model kernels select GM subviews with dynamic base offsets.  The
    # best-effort EmitC analysis sees the tile footprint but cannot always
    # prove the enclosing loop/offset maxima, so use the complete PTO tensor
    # view sizes as the validation backing-buffer lower bounds.
    "exp_gate_mm": {"v1": 32_768, "v2": 2_097_152, "v3": 268_435_456},
    "exp_up_mm": {"v1": 32_768, "v2": 2_097_152, "v3": 268_435_456},
    "exp_w2_mm": {"v1": 65_536, "v2": 32_768, "v3": 268_435_456},
    "gate": {"v1": 64, "v2": 65_536, "v3": 262_144, "v4": 16, "v5": 4_096, "v6": 16},
    "hc_head_reduce": {
        "v1": 1,
        "v2": 8,
        "v3": 128,
        "v4": 128,
        "v5": 32_768,
        "v6": 131_072,
    },
    "idx_qr_proj_matmul": {"v1": 131_072, "v2": 8_192, "v3": 8_388_608},
    "lm_head_matmul": {"v1": 1_034_240, "v2": 65_536, "v3": 264_765_440},
    "proj_a_mm": {"v1": 262_144, "v2": 33_554_432, "v3": 131_072},
    "proj_b_mm": {"v1": 524_288, "v2": 131_072, "v3": 33_554_432},
    "qr_hadamard_quant": {"v1": 512 * 128, "v2": 512, "v3": 512 * 128},
    "rms_norm": {"v1": 32_768, "v2": 32_768, "v3": 4_096},
    "sh_gate_mm": {"v1": 32_768, "v2": 8_388_608, "v3": 32_768},
    "sh_up_mm": {"v1": 32_768, "v2": 8_388_608, "v3": 32_768},
    "swa_gather_kv": {"v1": 1_024 * 512, "v2": 8 * 128, "v3": 128 * 512},
    "tquant_mx": {
        # The generated MX auxiliary tiles use 1x32 Vec storage even though
        # the logical tensor views are 1x16. Keep the GM backing buffers large
        # enough for the lowered TSTORE footprint on A5.
        "v3": 32,
        "v4": 32,
        "v5": 32,
    },
    **{
        testcase: {
            "v1": 1024 * 4096,
            "v2": 8192 * 64,
            "v3": 8192 * 64,
        }
        for testcase in DEEPSEEK_V4_DIRECT_CASES
    },
}

# The generic footprint analysis follows the currently selected tile, but
# these model kernels address strided weight tensors whose backing storage is
# larger than that tile.  Use the full tensor-view extents for the affected
# source revision.  Scope by sample so older DeepSeek variants with the same
# testcase names retain their already validated layouts.
SAMPLE_CASE_POINTER_COUNT_MINIMUMS = {
    "tpipe": {
        "a3_pipe_roundtrip": {"v3": 2_048},
    },
    "deepseekv4decodea3": {
        # `rope_cs` fills two 8x32 output partitions at column offsets 0 and
        # 32, and loads two 8x16 input partitions at offsets 0 and 16.  The
        # best-effort footprint scan only sees the first loop iteration and
        # used to allocate 480/464 elements for these 8x64 tensor views.  Keep
        # the complete backing views so the second partition stays in bounds.
        "rope_cs": {
            "v2": 8 * 64,
            "v3": 8 * 64,
            "v4": 8 * 64,
            "v5": 8 * 64,
        },
        # Mixed AIC/AIV kernels need their full exported tensor views plus the
        # backing GM FIFO described by initialize_pipe. Keep these A3-export
        # layouts sample-scoped because the A5 snapshot uses different shapes.
        "qk_pv": {
            "v1": 8 * 128,
            "v2": 1_024 * 512,
            "v3": 512,
            "v4": 512,
            "v5": 512 * 512,
            "v6": 512 * 512,
            "v7": 32_768,
        },
        "score": {
            "v1": 4,
            "v2": 4 * 2,
            "v3": 512 * 128,
            "v4": 512,
            "v5": 16 * 64,
            "v6": 8 * 4_096,
            "v7": 256,
            "v8": 128 * 128,
            "v9": 128,
            "v10": 16_384,
        },
        "gate": {
            "v1": 64,
            "v2": 16 * 4_096,
            "v3": 64 * 4_096,
            "v4": 16,
            "v5": 16 * 256,
            "v6": 2_048,
        },
    },
    "qwen3_14bdecodea3": {
        # The generic EmitC footprint analysis follows individual partition
        # views and can miss later tiles in these large strided tensors.  Size
        # the backing allocations from the complete PTO tensor-view shapes.
        "k_proj": {
            "v1": 16 * 1_024,
            "v2": 16 * 5_120,
            "v3": 5_120 * 1_024,
        },
        "v_proj": {
            "v1": 16 * 1_024,
            "v2": 16 * 5_120,
            "v3": 5_120 * 1_024,
        },
        "q_proj": {
            "v1": 16 * 5_120,
            "v2": 16 * 5_120,
            "v3": 5_120 * 5_120,
        },
        "down_proj": {
            "v1": 16 * 17_408,
            "v2": 17_408 * 5_120,
            "v3": 16 * 5_120,
        },
        "out_proj": {
            "v1": 16 * 5_120,
            "v2": 5_120 * 5_120,
            "v3": 16 * 5_120,
        },
    },
    "qwen3_14bprefilla3": {
        "gate_up_proj": {
            "v1": 128 * 8_704,
            "v2": 128 * 5_120,
            "v3": 5_120 * 17_408,
            "v4": 128 * 8_704,
            "v5": 5_120 * 17_408,
        },
        "kv_proj": {
            "v1": 64 * 1_024,
            "v2": 64 * 1_024,
            "v3": 64 * 5_120,
            "v4": 5_120 * 1_024,
            "v5": 5_120 * 1_024,
        },
        "out_proj_aic": {
            "v1": 64 * 5_120,
            "v2": 64 * 5_120,
            "v3": 5_120 * 5_120,
        },
        "q_proj": {
            "v1": 64 * 5_120,
            "v2": 64 * 5_120,
            "v3": 5_120 * 5_120,
        },
        # The exported SPMD worker uses a grid-stride loop over all 792
        # vocabulary chunks (worker 0 visits chunks 0, 24, 48, ...).  Keep the
        # full tensor-view backing storage so those non-contiguous reads stay
        # within the allocation.
        "lm_head": {
            "v1": 152_064,
            "v2": 16 * 5_120,
            "v3": 152_064 * 5_120,
        },
        # The standalone mixed-kernel harness runs worker 0. Keep the complete
        # tensor views reached by that worker and one 8-slot bidirectional GM
        # FIFO (8 * 16 KiB == 32768 float elements), matching the upstream
        # orchestration allocation per worker.
        "qk_pv_skew_probe": {
            "v1": 8_192,
            "v2": 8_192 * 128,
            "v3": 1,
            "v4": 128 * 1_024,
            "v5": 128 * 1_024,
            "v6": 8_192 * 128,
            "v7": 64 * 5_120,
            "v8": 32_768,
        },
        # Match the first valid upstream orchestration call: one final token,
        # cache block zero, worker zero, and one 8-slot bidirectional GM FIFO
        # (8 * 8 KiB == 16384 float elements) for this mixed AIC/AIV kernel.
        "qk_pv_online_phase": {
            "v1": 8_192,
            "v2": 8_192,
            "v3": 8_192 * 128,
            "v4": 1,
            "v5": 128 * 1_024,
            "v6": 128 * 1_024,
            "v7": 8_192 * 128,
            "v8": 16_384,
        },
    },
    "qwen3decodea5": {
        "sv_matmul": {
            "v1": 2_048 * 128,
            "v2": 2_048 * 256,
            "v3": 524_288 * 128,
        },
    },
    "qwen3_32bdecode4da5": {
        "down_proj": {
            "v1": 1_600 * 256,
            "v2": 409_600 * 512,
            "v3": 16 * 16 * 512,
        },
        "k_proj": {
            "v1": 1_024 * 128,
            "v2": 32_768 * 256,
            "v3": 4 * 16 * 256,
        },
        "out_proj": {
            "v1": 1_024 * 128,
            "v2": 131_072 * 512,
            "v3": 16 * 16 * 512,
        },
        "v_proj": {
            "v1": 1_024 * 128,
            "v2": 32_768 * 256,
            "v3": 4 * 16 * 256,
        },
    },
    "deepseekv4proprefilla5": {
        "rope": {
            "v1": 2_048 * 4_096,
            "v2": 128 * 128 * 64,
            "v3": 128 * 64,
            "v4": 128 * 64,
        },
        "prefill_idx_qr_proj": {
            "v1": 8_192,
            "v2": 128 * 8_192,
            "v3": 128 * 1_536,
            "v4": 1_536 * 8_192,
            "v5": 128,
        },
    },
    "deepseekv4prodecodea5": {
        "hca_cache_topk": {
            "v1": 8,
            "v2": 4,
            "v3": 1_024,
        },
        "idx_qr_proj_matmul": {
            "v1": 16 * 8_192,
            "v2": 8 * 1_536,
            "v3": 1_536 * 8_192,
        },
        "kv_score_proj": {
            "v1": 7_168,
            "v2": 512 * 7_168,
            "v3": 512 * 7_168,
            "v4": 16 * 512,
            "v5": 16 * 512,
        },
    },
    "deepseekv4flashmtpprefilla5": {
        "merge_rope_pack": {
            "v1": 128 * 8,
            "v2": 16 * 64,
            "v3": 128 * 64,
            "v4": 128 * 64,
            "v5": 8_192 * 512,
            "v6": 1_024 * 4_096,
            "v7": 1,
            "v8": 1,
            "v9": 512,
            "v10": 64,
        },
        "prefill_idx_c4_rmsnorm_rope": {
            "v1": 32,
            "v2": 32,
            "v3": 16_384 * 64,
            "v4": 16_384 * 64,
            "v5": 32 * 128,
            "v6": 128,
            "v7": 32 * 128,
        },
        "prefill_idx_qr_proj": {
            "v1": 128 * 1_024,
            "v2": 1_024 * 8_192,
            "v3": 8_192,
            "v4": 128 * 8_192,
            "v5": 128,
        },
    },
    "deepseekv4flashdsparka5": {
        "kv_proj_matmul": {
            "v1": 512,
            "v2": 4_096,
            "v3": 4_096 * 512,
        },
        "qproj_matmul": {
            "v1": 32_768,
            "v2": 1_024,
            "v3": 1_024 * 32_768,
        },
        "qr_proj_matmul": {
            "v1": 1_024,
            "v2": 4_096,
            "v3": 4_096 * 1_024,
        },
        "rmsnorm_rope": {
            "v1": 16 * 64,
            "v2": 16 * 64,
            "v3": 16 * 128,
            "v4": 16 * 128,
            "v5": 128,
        },
        "gather_ori_kv": {
            "v1": 49_152 * 512,
            "v2": 128 * 128,
            "v3": 512,
        },
        "lm_head_combine_gather": {
            "v1": 128 * 129_280,
            "v2": 128 * 129_280,
        },
        "merge_rope_pack": {
            "v1": 128 * 8,
            "v2": 16 * 64,
            "v3": 128 * 64,
            "v4": 128 * 64,
            "v5": 8_192 * 512,
            "v6": 1_024 * 4_096,
            "v7": 1,
            "v8": 1,
            "v9": 512,
            "v10": 64,
        },
        "prefill_idx_qr_proj": {
            "v1": 128 * 1_024,
            "v2": 1_024 * 8_192,
            "v3": 8_192,
            "v4": 128 * 8_192,
            "v5": 128,
        },
    },
    "deepseekv4decodea5": {
        "hca_rope": {
            "v1": 4 * 32,
            "v2": 4 * 32,
            "v3": 8 * 64,
            "v4": 8 * 64,
            "v5": 8,
            "v6": 16_384 * 64,
            "v7": 16_384 * 64,
        },
        "kv_proj_matmul": {
            "v1": 16 * 512,
            "v2": 16 * 4_096,
            "v3": 4_096 * 512,
        },
    },
    "deepseekv3_2prefillbacka5": {
        "deepseek_v3_2_prefill_back_layer_incore_1": {
            "v1": 64 * 16_384,
            "v2": 16_384 * 7_168,
            "v3": 64 * 7_168,
        },
        "deepseek_v3_2_prefill_back_layer_incore_5": {
            "v1": 64 * 7_168,
            "v2": 7_168 * 18_432,
            "v3": 64 * 128,
        },
        "deepseek_v3_2_prefill_back_layer_incore_6": {
            "v1": 64 * 7_168,
            "v2": 7_168 * 18_432,
            "v3": 64 * 128,
        },
        "deepseek_v3_2_prefill_back_layer_incore_8": {
            "v1": 64 * 18_432,
            "v2": 18_432 * 7_168,
            "v3": 64 * 128,
        },
    },
    "deepseekv3_2decodefronta5": {
        "decode_cache_write": {
            "v1": 16,
            "v2": 256 * 1_024,
            "v3": 256 * 1_024,
            "v4": 8 * 1_024,
            "v5": 9 * 1_024,
            "v6": 32_768 * 1_024,
            "v7": 4_096 * 1_024,
        },
        "kv_a_proj": {
            "v1": 16 * 7_168,
            "v2": 7_168 * 576,
            "v3": 16 * 64,
        },
        "q_head_proj": {
            "v1": 16 * 1_536,
            "v2": 1_536 * 24_576,
            "v3": 16 * 64,
        },
        "q_lora_proj": {
            "v1": 16 * 7_168,
            "v2": 7_168 * 1_536,
            "v3": 16 * 1_536,
        },
        "s2_k_idx_proj": {
            "v1": 16 * 7_168,
            "v2": 7_168 * 128,
            "v3": 16 * 64,
        },
        "s2_q_idx_proj": {
            "v1": 16 * 1_536,
            "v2": 1_536 * 8_192,
            "v3": 16 * 128,
        },
        "s4_dispatch": {
            "v1": 1,
            "v2": 32_768 * 1_024,
            "v3": 256 * 1_024,
        },
        "s4_softmax": {
            "v1": 256 * 128,
            "v2": 32_768 * 1_024,
            "v3": 4_096 * 1_024,
            "v4": 8 * 1_024,
            "v5": 1_024,
            "v6": 8 * 1_024,
        },
    },
    "deepseekv3_2decodebacka5": {
        "deepseek_v3_2_decode_back_layer_incore_0": {
            "v1": 16 * 16_384,
            "v2": 16_384 * 7_168,
            "v3": 16 * 64,
        },
        "deepseek_v3_2_decode_back_layer_incore_3": {
            "v1": 16 * 7_168,
            "v2": 7_168 * 18_432,
            "v3": 16 * 256,
        },
        "deepseek_v3_2_decode_back_layer_incore_4": {
            "v1": 16 * 7_168,
            "v2": 7_168 * 18_432,
            "v3": 16 * 256,
        },
        "deepseek_v3_2_decode_back_layer_incore_6": {
            "v1": 16 * 18_432,
            "v2": 18_432 * 7_168,
            "v3": 16 * 128,
        },
    },
}

CASE_COMPARE_PREFIX_COUNTS = {
    # The dynamic output has shape [arg5, 32].  With the validation default
    # arg5=1, only the first row is defined; the remaining physical ACC tile
    # rows must not participate in the determinism comparison.
    "hc_pre_linear": {"v3": 32},
}

QWEN3_DECODE_POINTER_COUNT_MINIMUMS = {
    "down_proj_residual": {
        "v1": 131072,
        "v2": 131072,
        "v3": 409600,
        "v4": 209715200,
        "v5": 8192,
    },
    "gate_proj": {
        "v1": 131072,
        "v2": 209715200,
        "v3": 409600,
    },
    "kv_proj": {
        "v1": 131072,
        "v2": 8388608,
        "v3": 8388608,
        "v4": 16384,
        "v5": 16384,
    },
    "online_softmax": {
        "v1": 131072,
        "v2": 1024,
        "v3": 1024,
        "v4": 8192,
    },
    "out_proj_residual": {
        "v1": 131072,
        "v2": 131072,
        "v3": 131072,
        "v4": 67108864,
        "v5": 8192,
    },
    "post_rmsnorm": {
        "v1": 131072,
        "v2": 131072,
        "v3": 8192,
    },
    "q_proj": {
        "v1": 131072,
        "v2": 67108864,
        "v3": 131072,
    },
    "qk_matmul": {
        "v1": 16384,
        "v2": 524288,
        "v3": 4194304,
    },
    "rmsnorm": {
        "v1": 131072,
        "v2": 131072,
        "v3": 8192,
    },
    "rope_kv_cache": {
        "v1": 16384,
        "v2": 4194304,
        "v3": 4194304,
        "v4": 16384,
        "v5": 64,
        "v6": 64,
        "v7": 64,
        "v8": 64,
        "v9": 16384,
        "v10": 131072,
    },
    "silu": {
        "v1": 409600,
        "v2": 409600,
        "v3": 409600,
    },
    "softmax": {
        "v1": 1024,
        "v2": 1024,
        "v3": 524288,
        "v4": 524288,
    },
    "sv_matmul": {
        "v1": 65536,
        "v2": 524288,
        "v3": 4194304,
    },
    "up_proj": {
        "v1": 131072,
        "v2": 209715200,
        "v3": 409600,
    },
}


def _pointer_count_minimums_for_case(sample_name: str, testcase: str) -> dict[str, int]:
    minimums = dict(CASE_POINTER_COUNT_MINIMUMS.get(testcase, {}))
    sample_name_lc = sample_name.lower()
    if sample_name_lc.startswith("qwen") and "decode" in sample_name_lc:
        minimums.update(QWEN3_DECODE_POINTER_COUNT_MINIMUMS.get(testcase, {}))
    minimums.update(
        SAMPLE_CASE_POINTER_COUNT_MINIMUMS
        .get(sample_name_lc, {})
        .get(testcase, {})
    )
    return minimums


def _parse_shape(text: str):
    match = re.search(r"Shape<(\d+)\s*,\s*(\d+)>", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"Shape<\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*(\d+)\s*,\s*(\d+)>", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 32, 32


def _split_params_blob(params_blob: str):
    params_blob = params_blob.strip()
    if not params_blob:
        return []
    params = []
    depth = 0
    start = 0
    for idx, ch in enumerate(params_blob):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(depth - 1, 0)
        elif ch == "," and depth == 0:
            params.append(params_blob[start:idx].strip())
            start = idx + 1
    last = params_blob[start:].strip()
    if last:
        params.append(last)
    return params


def _find_matching_brace(text: str, open_brace_index: int) -> Optional[int]:
    depth = 0
    for idx in range(open_brace_index, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx
    return None


def _find_matching_paren(text: str, open_paren_index: int) -> Optional[int]:
    depth = 0
    for idx in range(open_paren_index, len(text)):
        ch = text[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return idx
    return None


def _split_top_level(text: str, sep: str) -> list[str]:
    parts = []
    start = 0
    paren_depth = 0
    brace_depth = 0
    bracket_depth = 0
    for idx, ch in enumerate(text):
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth = max(paren_depth - 1, 0)
        elif ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth = max(brace_depth - 1, 0)
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth = max(bracket_depth - 1, 0)
        elif (
            ch == sep
            and paren_depth == 0
            and brace_depth == 0
            and bracket_depth == 0
        ):
            parts.append(text[start:idx].strip())
            start = idx + 1
    parts.append(text[start:].strip())
    return parts


def _extract_function_body(function_text: str) -> str:
    brace_index = function_text.find("{")
    if brace_index < 0:
        return ""
    end_index = _find_matching_brace(function_text, brace_index)
    if end_index is None:
        return ""
    body = function_text[brace_index + 1:end_index].strip()
    body = re.sub(r"\breturn\s*;\s*$", "", body, flags=re.S).rstrip()
    return body


def _strip_ptoas_auto_sync_tail(body: str) -> tuple[str, bool]:
    pattern = re.compile(
        r"\n?\s*ptoas_auto_sync_tail\s*\([^;]*\)\s*;\s*$",
        re.S,
    )
    updated = pattern.sub("", body.rstrip())
    return updated.rstrip(), updated != body.rstrip()


def _indent_block(text: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    return "\n".join((prefix + line) if line else "" for line in text.splitlines())


def _split_cpp_args(text: str):
    text = text.strip()
    if not text:
        return []
    parts = []
    depth_angle = 0
    depth_paren = 0
    depth_brace = 0
    depth_bracket = 0
    start = 0
    for idx, ch in enumerate(text):
        if ch == "<":
            depth_angle += 1
        elif ch == ">":
            depth_angle = max(depth_angle - 1, 0)
        elif ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(depth_paren - 1, 0)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(depth_brace - 1, 0)
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket = max(depth_bracket - 1, 0)
        elif ch == "," and depth_angle == 0 and depth_paren == 0 and depth_brace == 0 and depth_bracket == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _extract_aicore_functions(text: str):
    pattern = re.compile(
        r"(?P<extern_c>extern\s+\"C\"\s+)?(?P<global>__global__\s+)?AICORE\s+void\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)\s*\{",
        re.S,
    )
    functions = []
    for match in pattern.finditer(text):
        brace_index = text.find("{", match.end("params"))
        if brace_index < 0:
            continue
        end_index = _find_matching_brace(text, brace_index)
        if end_index is None:
            continue
        params_blob = match.group("params").strip()
        functions.append(
            {
                "name": match.group("name"),
                "params_blob": params_blob,
                "raw_params": _split_params_blob(params_blob),
                "is_global": bool(match.group("global")),
                "is_extern_c": bool(match.group("extern_c")),
                "text": text[match.start():end_index + 1],
            }
        )
    return functions


def _describe_kernel_source(text: str):
    functions = _extract_aicore_functions(text)
    for func in functions:
        if func["is_global"]:
            return {
                "kind": "global",
                "kernel_name": func["name"],
                "raw_params": func["raw_params"],
                "analysis_texts": [func["text"]],
                "writer_texts": [func["text"]],
                "call_text": func["text"],
                "needs_global_wrapper": False,
            }

    if len(functions) == 1:
        func = functions[0]
        return {
            "kind": "global",
            "kernel_name": func["name"],
            "raw_params": func["raw_params"],
            "analysis_texts": [func["text"]],
            "writer_texts": [func["text"]],
            "call_text": func["text"],
            "needs_global_wrapper": not func["is_global"],
        }

    mixed_groups = {}
    for func in functions:
        name = func["name"]
        for suffix in ("_aic", "_aiv"):
            if not name.endswith(suffix):
                continue
            base = name[: -len(suffix)]
            group = mixed_groups.setdefault(base, {})
            group[suffix[1:]] = func
            break

    for base, group in mixed_groups.items():
        if "aic" in group and "aiv" in group:
            aic_params = group["aic"]["raw_params"]
            aiv_params = group["aiv"]["raw_params"]
            aiv_local_param = None
            if (
                len(aiv_params) == len(aic_params) + 1
                and aiv_params[:len(aic_params)] == aic_params
                and _extract_cpp_type(aiv_params[-1]) in {"int32_t", "int"}
                and "%subblock_idx" in group["aiv"]["text"]
            ):
                # PyPTO mixed A3 exports append the lane-local subblock id only
                # to the AIV function. It is supplied by the device runtime,
                # not by the host launch ABI.
                params = aic_params
                aiv_local_param = _extract_cpp_name(aiv_params[-1])
            else:
                params = aiv_params or aic_params
            return {
                "kind": "mixed",
                "kernel_name": base,
                "raw_params": params,
                "analysis_texts": [group["aic"]["text"], group["aiv"]["text"]],
                "writer_texts": [group["aiv"]["text"]],
                "aic_text": group["aic"]["text"],
                "aiv_text": group["aiv"]["text"],
                "aiv_local_param": aiv_local_param,
                "call_text": group["aiv"]["text"],
                "needs_global_wrapper": False,
            }

    return {
        "kind": "fallback",
        "kernel_name": "kernel",
        "raw_params": [],
        "analysis_texts": [text],
        "writer_texts": [text],
        "call_text": text,
        "needs_global_wrapper": False,
    }


def _append_single_kernel_global_wrapper(
    kernel_text: str,
    kernel_name: str,
    raw_params: list[str],
) -> str:
    impl_name = f"__ptoas_{kernel_name}_impl"
    pattern = re.compile(
        rf"(?P<prefix>extern\s+\"C\"\s+)?(?P<global>__global__\s+)?(?P<static>static\s+)?"
        rf"AICORE\s+(?P<inline>inline\s+)?void\s+{re.escape(kernel_name)}\s*\((?P<params>[^)]*)\)\s*\{{",
        re.S,
    )

    def _replace_entry(match):
        params = match.group("params").strip()
        return f"static AICORE inline void {impl_name}({params}) {{"

    rewritten, count = pattern.subn(_replace_entry, kernel_text, count=1)
    if count == 0:
        return kernel_text

    call_args = ", ".join(_extract_cpp_name(param) for param in raw_params)
    wrapper = (
        "\n\n"
        f"extern \"C\" __global__ AICORE void {kernel_name}({', '.join(raw_params)}) {{\n"
        f"  {impl_name}({call_args});\n"
        "}\n"
    )
    return rewritten.rstrip() + wrapper


def _append_mixed_kernel_wrapper(
    kernel_text: str,
    kernel_name: str,
    raw_params: list[str],
    aic_text: str,
    aiv_text: str,
    aiv_local_param: Optional[str] = None,
) -> str:
    pipe_decl_pattern = re.compile(
        r"^(?P<indent>\s*)auto\s+(?P<name>\w+)\s*=\s*(?P<type>TPipe<[^;=]+>)\s*\((?P<args>[^;]*)\)\s*;\s*$",
        re.M,
    )
    param_names = {_extract_cpp_name(param) for param in raw_params}
    safe_identifiers = {"nullptr", "NULL", "true", "false"}

    def _find_decl_init(prefix: str, name: str):
        pattern = re.compile(
            rf"^\s*(?P<type>[^=\n;]+?)\s+{re.escape(name)}\s*=\s*(?P<init>[^;]+);\s*$",
            re.M,
        )
        match = None
        for current in pattern.finditer(prefix):
            match = current
        if match is None:
            return None, None, None
        return match.group("type").strip(), match.group("init").strip(), match.start()

    def _render_pointer_init(type_text: str, init_text: str) -> str:
        expr = init_text.strip()
        if "*" not in type_text:
            return expr
        if expr.startswith("(") or expr.startswith("reinterpret_cast") or expr.startswith("static_cast"):
            return expr
        return f"({type_text}){expr}"

    def _resolve_ctor_arg(arg_text: str, prefix: str, depth: int = 0):
        arg_text = arg_text.strip()
        if not arg_text:
            return None
        if depth > 8:
            return None
        if not re.fullmatch(r"[A-Za-z_]\w*", arg_text):
            return arg_text
        if arg_text in safe_identifiers:
            return arg_text
        if arg_text in param_names:
            return arg_text
        type_text, init_text, decl_start = _find_decl_init(prefix, arg_text)
        if type_text is None or init_text is None:
            return None
        resolved_init = init_text
        if (
            re.fullmatch(r"[A-Za-z_]\w*", init_text)
            and init_text not in param_names
            and init_text not in safe_identifiers
        ):
            resolved_init = _resolve_ctor_arg(init_text, prefix[:decl_start], depth + 1)
            if resolved_init is None:
                return None
        return _render_pointer_init(type_text, resolved_init)

    def _extract_pipe_decls(body: str):
        decls = []
        for match in pipe_decl_pattern.finditer(body):
            ctor_args = _split_cpp_args(match.group("args"))
            prefix = body[:match.start()]
            resolved_args = []
            for arg in ctor_args:
                resolved = _resolve_ctor_arg(arg, prefix)
                if resolved is None:
                    break
                resolved_args.append(resolved)
            else:
                decls.append(
                    {
                        "name": match.group("name"),
                        "type_text": match.group("type").strip(),
                        "ctor_args": tuple(resolved_args),
                        "span": match.span(),
                    }
                )
        return decls

    def _rewrite_body(body: str, replacements):
        rewritten = body
        for replacement in sorted(replacements, key=lambda item: item["span"][0], reverse=True):
            start, end = replacement["span"]
            rewritten = rewritten[:start] + rewritten[end:]
        for replacement in replacements:
            rewritten = re.sub(
                rf"\b{re.escape(replacement['old_name'])}\b",
                replacement["new_name"],
                rewritten,
            )
        return rewritten.strip()

    def _next_shared_name(seed: int, texts: list[str]) -> str:
        index = seed
        while True:
            name = f"__ptoas_shared_pipe{index}"
            if all(name not in text for text in texts):
                return name
            index += 1

    aic_body = _extract_function_body(aic_text)
    aiv_body = _extract_function_body(aiv_text)
    if aiv_local_param:
        aiv_body = re.sub(
            rf"\b{re.escape(aiv_local_param)}\b",
            "get_subblockid()",
            aiv_body,
        )
    aic_body, aic_has_tail = _strip_ptoas_auto_sync_tail(aic_body)
    aiv_body, aiv_has_tail = _strip_ptoas_auto_sync_tail(aiv_body)
    aic_decls = _extract_pipe_decls(aic_body)
    aiv_decls = _extract_pipe_decls(aiv_body)

    shared_pairs = []
    aiv_by_key = {}
    for decl in aiv_decls:
        key = (decl["type_text"], decl["ctor_args"])
        aiv_by_key.setdefault(key, []).append(decl)
    for decl in aic_decls:
        key = (decl["type_text"], decl["ctor_args"])
        bucket = aiv_by_key.get(key)
        if not bucket:
            continue
        shared_pairs.append((decl, bucket.pop(0)))

    shared_decls = []
    aic_replacements = []
    aiv_replacements = []
    shared_seed = 0
    texts_for_name_check = [kernel_text, aic_body, aiv_body]
    for aic_decl, aiv_decl in shared_pairs:
        shared_name = _next_shared_name(shared_seed, texts_for_name_check)
        shared_seed += 1
        texts_for_name_check.append(shared_name)
        shared_decls.append(
            f"  auto {shared_name} = {aic_decl['type_text']}({', '.join(aic_decl['ctor_args'])});"
        )
        aic_replacements.append(
            {
                "old_name": aic_decl["name"],
                "new_name": shared_name,
                "span": aic_decl["span"],
            }
        )
        aiv_replacements.append(
            {
                "old_name": aiv_decl["name"],
                "new_name": shared_name,
                "span": aiv_decl["span"],
            }
        )

    wrapper_blocks = []
    for body in (_rewrite_body(aic_body, aic_replacements), _rewrite_body(aiv_body, aiv_replacements)):
        if not body:
            continue
        wrapper_blocks.append("  {\n" + _indent_block(body) + "\n  }")

    if not wrapper_blocks:
        return kernel_text

    wrapper = (
        "\n\n"
        f"extern \"C\" __global__ AICORE void {kernel_name}({', '.join(raw_params)}) {{\n"
        + ("\n".join(shared_decls) + ("\n\n" if shared_decls else ""))
        + "\n".join(wrapper_blocks)
        + ("\n  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);" if (aic_has_tail or aiv_has_tail) else "")
        + "\n"
        "}\n"
    )
    return kernel_text.rstrip() + wrapper


def _is_gm_pointer_param(param: str) -> bool:
    return "__gm__" in param and "*" in param


def _extract_cpp_type(param: str) -> str:
    match = re.search(r"__gm__\s+([A-Za-z_]\w*)", param)
    if match:
        return match.group(1)

    tokens = param.replace("*", " ").strip().split()
    if not tokens:
        return "float"
    if len(tokens) == 1:
        return tokens[0]
    qualifiers = {"const", "volatile", "restrict", "__restrict", "__restrict__"}
    type_tokens = [t for t in tokens[:-1] if t not in qualifiers]
    return " ".join(type_tokens) if type_tokens else tokens[0]


def _extract_cpp_name(param: str) -> str:
    parts = param.strip().split()
    if not parts:
        return "arg"
    name = parts[-1].replace("*", "").strip()
    if name.startswith("__"):
        return "arg"
    return name


def _strip_param_name(raw: str, name: str) -> str:
    """
    Return the type part of a parameter declaration, keeping qualifiers and the
    pointer '*' but removing the trailing variable name.
    Example: "__gm__ float* v1" -> "__gm__ float*"
    """
    pattern = rf"\b{re.escape(name)}\b\s*$"
    stripped = re.sub(pattern, "", raw.strip())
    return stripped.strip()


def _strip_enclosing_parens(expr: str) -> str:
    expr = expr.strip()
    while expr.startswith("(") and expr.endswith(")"):
        depth = 0
        ok = True
        for idx, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and idx != len(expr) - 1:
                    ok = False
                    break
        if ok and depth == 0:
            expr = expr[1:-1].strip()
        else:
            break
    return expr


def _strip_simple_casts(expr: str) -> str:
    cur = expr.strip()
    for _ in range(8):
        prev = cur
        cur = _strip_enclosing_parens(cur)
        match = re.match(r"^(?:reinterpret_cast|static_cast|const_cast|dynamic_cast)\s*<[^>]+>\s*\((.*)\)$", cur, re.S)
        if match:
            cur = match.group(1).strip()
            continue
        match = re.match(r"^\(\s*[^()]+\s*\)\s*(.+)$", cur, re.S)
        if match:
            cur = match.group(1).strip()
            continue
        if cur == prev:
            break
    return cur


def _infer_void_gm_pointee_type(text: str, param_name: str) -> Optional[str]:
    # Common patterns in PTOAS-generated kernels:
    #   __gm__ int16_t* v16 = (__gm__ int16_t*) v1;
    #   __gm__ half*   v16 = (__gm__ half*) v1;
    name = re.escape(param_name)
    patterns = [
        # Direct assignment after implicit conversion (some kernels keep the
        # ABI as `void*` and only materialize the real type for arithmetic).
        rf"__gm__\s+([A-Za-z_]\w*)\s*\*\s*\w+\s*=\s*{name}\b",
        rf"\(__gm__\s+([A-Za-z_]\w*)\s*\*\)\s*{name}\b",
        rf"reinterpret_cast<__gm__\s+([A-Za-z_]\w*)\s*\*\s*>\(\s*{name}\s*\)",
        rf"static_cast<__gm__\s+([A-Za-z_]\w*)\s*\*\s*>\(\s*{name}\s*\)",
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            ty = match.group(1)
            if ty and ty != "void":
                return ty
    return None


def _ordered_unique(items):
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _resolve_pointer_param_from_expr(expr: str, pointer_param_names, ptr_to_param, ptr_to_base) -> Optional[str]:
    if not expr:
        return None
    cur = _strip_simple_casts(expr)
    match = re.match(r"^(\w+)\s*\+", cur)
    if match:
        cur = match.group(1)
    elif re.fullmatch(r"[A-Za-z_]\w*", cur):
        cur = cur
    else:
        return None

    pointer_params = set(pointer_param_names)
    seen = set()
    for _ in range(12):
        if cur in seen:
            break
        seen.add(cur)
        if cur in pointer_params:
            return cur
        mapped = ptr_to_param.get(cur)
        if mapped:
            cur = mapped
            continue
        mapped = ptr_to_base.get(cur)
        if mapped:
            cur = mapped
            continue
        break
    return None


def _detect_output_pointer_params(text: str, pointer_param_names):
    if not pointer_param_names:
        return []

    tstore_gts = re.findall(r"\bTSTORE\s*\(\s*(\w+)\s*,", text)
    if not tstore_gts:
        return []

    gt_to_expr = {}
    for match in re.finditer(
        r"\bGlobalTensor<[^;\n]*>\s+(\w+)\s*=\s*GlobalTensor<[^;\n]*>\(([^,]+?)\s*,",
        text,
    ):
        gt_to_expr.setdefault(match.group(1), match.group(2).strip())
    for match in re.finditer(
        r"\b(?:pto::)?GlobalTensor<[^;\n]*>\s+(\w+)\s*=\s*(?:pto::)?GlobalTensor<[^;\n]*>\s*\(([^;]*)\);",
        text,
    ):
        args = _split_cpp_args(match.group(2))
        if args:
            gt_to_expr.setdefault(match.group(1), args[0].strip())
    for match in re.finditer(
        r"\b(\w+)\s*=\s*(?:pto::)?GlobalTensor<[^;\n]*>\s*\(([^;]*)\);",
        text,
    ):
        args = _split_cpp_args(match.group(2))
        if args:
            gt_to_expr.setdefault(match.group(1), args[0].strip())
    for match in re.finditer(r"\b(\w+)\s+(\w+)\s*=\s*\1\s*\(([^,]+?)\s*,", text):
        gt_to_expr.setdefault(match.group(2), match.group(3).strip())

    ptr_to_base = {}
    for match in re.finditer(r"__gm__\s+[\w:<>]+\s*\*\s*(\w+)\s*=\s*(\w+)\s*\+", text):
        ptr_to_base[match.group(1)] = match.group(2)
    for match in re.finditer(r"\b(\w+)\s*=\s*(\w+)\s*\+\s*[^;]+;", text):
        ptr_to_base.setdefault(match.group(1), match.group(2))

    ptr_to_param = {}
    for match in re.finditer(
        r"__gm__\s+[\w:<>]+\s*\*\s*(\w+)\s*=\s*\(__gm__\s+[\w:<>]+\s*\*\)\s*(\w+)\b",
        text,
    ):
        ptr_to_param[match.group(1)] = match.group(2)
    for match in re.finditer(r"\b(\w+)\s*=\s*\(__gm__\s+[\w:<>]+\s*\*\)\s*(\w+)\b", text):
        ptr_to_param.setdefault(match.group(1), match.group(2))

    for match in re.finditer(
        r"__gm__\s+[\w:<>]+\s*\*\s*(\w+)\s*=\s*PTOAS__GLOBAL_TENSOR_DATA\(\s*(\w+)\s*\);",
        text,
    ):
        alias, tensor = match.group(1), match.group(2)
        expr = gt_to_expr.get(tensor)
        param = _resolve_pointer_param_from_expr(expr, pointer_param_names, ptr_to_param, ptr_to_base)
        if param:
            ptr_to_param[alias] = param
    for match in re.finditer(
        r"\b(\w+)\s*=\s*PTOAS__GLOBAL_TENSOR_DATA\(\s*(\w+)\s*\);",
        text,
    ):
        alias, tensor = match.group(1), match.group(2)
        expr = gt_to_expr.get(tensor)
        param = _resolve_pointer_param_from_expr(expr, pointer_param_names, ptr_to_param, ptr_to_base)
        if param:
            ptr_to_param[alias] = param

    outputs = []
    for gt in tstore_gts:
        expr = gt_to_expr.get(gt)
        param = _resolve_pointer_param_from_expr(expr, pointer_param_names, ptr_to_param, ptr_to_base)
        if param:
            outputs.append(param)
    return _ordered_unique(outputs)


def _detect_set_ffts_pointer_params(text: str, pointer_param_names):
    if not pointer_param_names:
        return set()

    def _is_fully_wrapped_by_parentheses(expr: str) -> bool:
        if not (expr.startswith("(") and expr.endswith(")")):
            return False
        depth = 0
        for i, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(expr) - 1:
                    return False
        return depth == 0

    def _extract_identifier(expr: str) -> Optional[str]:
        cur = expr.strip()
        for _ in range(8):
            prev = cur
            while _is_fully_wrapped_by_parentheses(cur):
                cur = cur[1:-1].strip()

            m = re.match(r"^(?:reinterpret_cast|static_cast|const_cast|dynamic_cast)\s*<[^>]+>\s*\((.*)\)$", cur, re.S)
            if m:
                cur = m.group(1).strip()
                continue

            # C-style cast: (uint64_t) v1 / (__gm__ int64_t*) v1
            m = re.match(r"^\(\s*[^()]+\s*\)\s*(.+)$", cur, re.S)
            if m:
                cur = m.group(1).strip()
                continue

            if cur == prev:
                break

        return cur if re.fullmatch(r"[A-Za-z_]\w*", cur) else None

    pointer_set = set(pointer_param_names)
    alias = {}
    # Track simple alias chains introduced by casted assignments, e.g.:
    #   uint64_t v6 = (uint64_t)v1;
    #   auto v7 = reinterpret_cast<uint64_t>(v6);
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*=\s*([^;]+);", text):
        lhs = m.group(1)
        rhs = m.group(2).strip()
        src = _extract_identifier(rhs)
        if src:
            alias[lhs] = src

    def _resolve_pointer_param(name: str) -> Optional[str]:
        cur = name
        seen = set()
        for _ in range(12):
            if cur in seen:
                break
            seen.add(cur)
            if cur in pointer_set:
                return cur
            nxt = alias.get(cur)
            if not nxt:
                return None
            cur = nxt
        return None

    hits = set()
    for m in re.finditer(r"\bset_ffts_base_addr\s*\(([^)]*)\)", text, re.S):
        raw_arg = m.group(1).strip()
        arg_name = _extract_identifier(raw_arg)
        if not arg_name:
            continue
        resolved = _resolve_pointer_param(arg_name)
        if resolved:
            hits.add(resolved)

    # Compatibility fallback for unusual formatting.
    if not hits:
        for name in pointer_param_names:
            pat = rf"\bset_ffts_base_addr\b[^\n;]*\b{re.escape(name)}\b"
            if re.search(pat, text):
                hits.add(name)
    return hits


def _detect_prefetch_workspace_pointer_params(text: str, pointer_param_names):
    if not pointer_param_names:
        return set()
    def _is_fully_wrapped_by_parentheses(expr: str) -> bool:
        if not (expr.startswith("(") and expr.endswith(")")):
            return False
        depth = 0
        for i, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(expr) - 1:
                    return False
        return depth == 0

    def _extract_identifier(expr: str) -> Optional[str]:
        cur = expr.strip()
        for _ in range(8):
            prev = cur
            while _is_fully_wrapped_by_parentheses(cur):
                cur = cur[1:-1].strip()

            m = re.match(r"^(?:reinterpret_cast|static_cast|const_cast|dynamic_cast)\s*<[^>]+>\s*\((.*)\)$", cur, re.S)
            if m:
                cur = m.group(1).strip()
                continue

            m = re.match(r"^\(\s*[^()]+\s*\)\s*(.+)$", cur, re.S)
            if m:
                cur = m.group(1).strip()
                continue

            if cur == prev:
                break

        return cur if re.fullmatch(r"[A-Za-z_]\w*", cur) else None

    pointer_set = set(pointer_param_names)
    alias = {}
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*=\s*([^;]+);", text):
        lhs = m.group(1)
        rhs = m.group(2).strip()
        src = _extract_identifier(rhs)
        if src:
            alias[lhs] = src

    def _resolve_pointer_param(name: str) -> Optional[str]:
        cur = name
        seen = set()
        for _ in range(12):
            if cur in seen:
                break
            seen.add(cur)
            if cur in pointer_set:
                return cur
            nxt = alias.get(cur)
            if not nxt:
                return None
            cur = nxt
        return None

    hits = set()
    for m in re.finditer(r"\bPrefetchAsyncContext\s+\w+\s*=\s*[^;]*\(([^)]*)\)\s*;", text, re.S):
        raw_arg = m.group(1).strip()
        arg_name = _extract_identifier(raw_arg)
        if not arg_name:
            continue
        resolved = _resolve_pointer_param(arg_name)
        if resolved:
            hits.add(resolved)

    if not hits:
        for name in pointer_param_names:
            pat = rf"\bPrefetchAsyncContext\b[^\n;]*\b{re.escape(name)}\b"
            if re.search(pat, text):
                hits.add(name)
    return hits


def _parse_kernel_params(text: str):
    match = re.search(r"__global__\s+(?:\w+\s+)*void\s+\w+\s*\(([^)]*)\)", text, re.S)
    if not match:
        return []
    return _split_params_blob(match.group(1))


def _parse_kernel_name(text: str) -> str:
    match = re.search(r"__global__\s+(?:\w+\s+)*void\s+(\w+)\s*\(", text, re.S)
    return match.group(1) if match else "kernel"


def _np_dtype_for_cpp(cpp_type: str) -> str:
    mapping = {
        "float": "np.float32",
        "float4_e1m2x2_t": "np.uint8",
        "float4_e2m1x2_t": "np.uint8",
        "float8_e4m3_t": "np.uint8",
        "float8_e5m2_t": "np.uint8",
        "float8_e8m0_t": "np.uint8",
        "half": "np.float16",
        "hifloat8_t": "np.uint8",
        "aclFloat16": "np.float16",
        "__bf16": "np.uint16",
        "bfloat16_t": "np.uint16",
        "int8_t": "np.int8",
        "uint8_t": "np.uint8",
        "int16_t": "np.int16",
        "uint16_t": "np.uint16",
        "int32_t": "np.int32",
        "uint32_t": "np.uint32",
        "int64_t": "np.int64",
        "uint64_t": "np.uint64",
    }
    return mapping.get(cpp_type, "np.float32")


def _cpp_host_type(cpp_type: str) -> str:
    if cpp_type == "half":
        return "aclFloat16"
    if cpp_type in {"__bf16", "bfloat16_t"}:
        return "uint16_t"
    return cpp_type


def _is_bf16_cpp_type(cpp_type: str) -> bool:
    return cpp_type in {"__bf16", "bfloat16_t"}


def _rewrite_host_unsupported_types(text: str) -> str:
    # `bisheng -xcce` performs a host-side pass that parses kernel launch code.
    # Some device-only builtin types (e.g. `__bf16`) are rejected there.
    return text.replace("__bf16", "bfloat16_t")


def _default_eps_for_cpp_type(cpp_type: str) -> float:
    # CPU golden vs NPU results may have small floating-point differences.
    if cpp_type in {"half", "aclFloat16"}:
        return 1e-2
    if cpp_type in {"float"}:
        return 1e-4
    return 0.0


def _default_bf16_max_ulp_for_cpp_type(cpp_type: str) -> int:
    return 1 if _is_bf16_cpp_type(cpp_type) else 0


def _integer_scalar_default_value(
    testcase: str,
    name: str,
    host_type: str,
    sample_name: str = "",
) -> Optional[int]:
    sample_override = (
        SAMPLE_CASE_INT_SCALAR_DEFAULTS
        .get(sample_name.lower(), {})
        .get(testcase, {})
        .get(name)
    )
    override = sample_override
    if override is None:
        override = CASE_INT_SCALAR_DEFAULTS.get(testcase, {}).get(name)
    if override is not None:
        return int(override)
    if re.match(r"^(u?int)(8|16|32|64)_t$", host_type) or host_type in {"int", "unsigned", "size_t"}:
        return 1
    return None


def _bool_scalar_default_value(testcase: str, name: str) -> Optional[bool]:
    override = CASE_BOOL_SCALAR_DEFAULTS.get(testcase, {}).get(name)
    if override is None:
        return None
    return bool(override)


def _derive_testcase_name(input_cpp: Path) -> str:
    name = input_cpp.stem
    if name.endswith("-pto"):
        name = name[:-4]
    if name.endswith("_pto"):
        name = name[:-4]
    return name


def _resolve_sample_root(input_cpp: Path) -> Path:
    parent = input_cpp.parent
    if parent.name == "npu_validation":
        return parent.parent
    if parent.parent.name == "npu_validation":
        return parent.parent.parent
    return parent


def _find_custom_case_asset(sample_root: Path, testcase: str, filename: str) -> Optional[Path]:
    candidates = (
        sample_root / f"{testcase}_{filename}",
        sample_root / "npu_validation" / testcase / filename,
        sample_root / "npu_validation" / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _use_custom_golden_for_case(testcase: str, soc_version: str) -> bool:
    testcase_lc = testcase.lower()
    soc_lc = (soc_version or "").lower()
    is_a3 = (
        any(token in soc_lc for token in ("910a", "910proa", "910b"))
        or os.environ.get("PTOAS_BOARD_IS_A3") == "1"
    )
    if is_a3 and testcase_lc in UNSTABLE_A3_CUSTOM_GOLDEN_CASES:
        return False
    return True


def _copy_asset_if_needed(src: Path, dst: Path):
    if src.resolve() == dst.resolve():
        return
    shutil.copy2(src, dst)


def _copy_custom_golden_helpers(sample_root: Path, output_dir: Path):
    for helper in sample_root.glob("*_golden_*.py"):
        _copy_asset_if_needed(helper, output_dir / helper.name)


def _replace_includes(text: str) -> str:
    if "#include \"common/pto_instr.hpp\"" in text:
        return text.replace("#include \"common/pto_instr.hpp\"", INCLUDE_REPLACEMENT.rstrip())
    if "#include <pto/pto-inst.hpp>" in text:
        return text
    return INCLUDE_REPLACEMENT + "\n" + text


def _inject_packed_pred_mask_preload(
    kernel_text: str,
    *,
    dst_tile: str,
    output_ptr: str,
    output_cpp_type: str,
    rows: int,
    cols: int,
    logical_elem_count: int,
) -> str:
    """
    pto.tcmp / pto.tcmps write a packed predicate mask and may leave parts of the
    destination tile undefined (UB garbage). Our validation harness compares
    two NPU runs for determinism; undefined bytes make the compare flaky.

    Inject a TLOAD(dst, GM_output) before the first PIPE_MTE2->PIPE_V barrier so
    the whole dst tile starts from deterministic contents (the output buffer is
    initialized from .bin files on the host).
    """
    if "PTOAS_PACKED_MASK_PRELOAD" in kernel_text:
        return kernel_text

    if not dst_tile or not output_ptr:
        return kernel_text

    # Find a reasonable insertion point: before the first MTE2->V set_flag.
    m = re.search(r"^(\s*)set_flag\s*\(\s*PIPE_MTE2\s*,\s*PIPE_V\s*,", kernel_text, re.M)
    if m:
        indent = m.group(1)
        insert_at = m.start()
    else:
        # Fallback: insert right before the first TCMP/TCMPS call.
        m2 = re.search(r"^(\s*)TCMPS?\s*\(", kernel_text, re.M)
        if not m2:
            return kernel_text
        indent = m2.group(1)
        insert_at = m2.start()

    # We don't rely on the kernel's existing GlobalTensor aliases here; keep
    # names unique to avoid collisions.
    preload_lines = [
        f"{indent}// PTOAS_PACKED_MASK_PRELOAD: init packed predicate dst from GM",
        f"{indent}{{",
        f"{indent}  using __ptoas_mask_gt_shape = pto::Shape<1, 1, 1, {rows}, {cols}>;",
        f"{indent}  using __ptoas_mask_gt_stride = pto::Stride<{logical_elem_count}, {logical_elem_count}, {logical_elem_count}, {cols}, 1>;",
        f"{indent}  constexpr pto::Layout __ptoas_mask_gt_layout = pto::Layout::ND;",
        f"{indent}  __ptoas_mask_gt_shape __ptoas_mask_shape = __ptoas_mask_gt_shape();",
        f"{indent}  __ptoas_mask_gt_stride __ptoas_mask_stride = __ptoas_mask_gt_stride();",
        f"{indent}  using __ptoas_mask_gt = GlobalTensor<{output_cpp_type}, __ptoas_mask_gt_shape, __ptoas_mask_gt_stride, __ptoas_mask_gt_layout>;",
        f"{indent}  __ptoas_mask_gt __ptoas_mask_src = __ptoas_mask_gt((__gm__ {output_cpp_type}*){output_ptr}, __ptoas_mask_shape, __ptoas_mask_stride);",
        f"{indent}  TLOAD({dst_tile}, __ptoas_mask_src);",
        f"{indent}}}",
        "",
    ]
    block = "\n".join(preload_lines)
    return kernel_text[:insert_at] + block + kernel_text[insert_at:]


def _infer_aicore_arch(kernel_text: str, soc_version: str) -> str:
    # Heuristic: kernels that touch cube/L0/L1 tile types or cbuf memories need
    # the "cube" arch; pure vector kernels can use the vector arch.
    #
    # IMPORTANT: the default arch depends on the Ascend SoC.
    has_mix_macros = "__DAV_CUBE__" in kernel_text and "__DAV_VEC__" in kernel_text
    has_intra_block_sync = "set_intra_block(" in kernel_text or "wait_intra_block(" in kernel_text
    has_mixed_section_sync = has_mix_macros and has_intra_block_sync
    cube_markers = (
        "TileType::Mat",
        "TileType::Left",
        "TileType::Right",
        "TileType::Acc",
        "__cbuf__",
        "__ca__",
        "__cb__",
        "__cc__",
        "copy_gm_to_cbuf",
        "copy_cbuf_to_gm",
        "mad(",
        "mmad(",
        "TMMAD",
    )
    needs_cube = any(m in kernel_text for m in cube_markers)

    sv = (soc_version or "").lower()
    if "950" in sv or "a5" in sv:
        # Sectioned kernels that synchronize across DAV cube/vector regions
        # need PTO-ISA's mixed-kernel compile mode so the toolchain chooses
        # the correct pipe restrictions and DAV macro ownership.
        if has_mixed_section_sync:
            return "dav-c310"
        # Ascend950 (A5) uses A5 instruction set. pto-isa examples build A5
        # kernels with dav-c310-{vec|cube}.
        return "dav-c310-cube" if needs_cube else "dav-c310-vec"
    if "910b" in sv:
        # A3 board validation follows the official a2a3 PTO-ISA ST setup:
        # build vec/cube kernels with dav-c220 and MEMORY_BASE rather than
        # the A5 dav-c310/REGISTER_BASE path.
        if has_mixed_section_sync:
            return "dav-c220"
        return "dav-c220-cube" if needs_cube else "dav-c220-vec"

    # Default to Ascend910 (dav-c220) when SoC is unknown.
    return "dav-c220-cube" if needs_cube else "dav-c220-vec"


def _infer_launch_block_count(kernel_text: str, testcase: str) -> int:
    # Inter-core sync functional cases need at least two cores:
    # one producer core does sync.set, one consumer core does sync.wait.
    if testcase.startswith("test_intercore_sync_") and "get_block_idx()" in kernel_text:
        return 2
    return 1


def _parse_int_list(blob: str):
    items = []
    for part in blob.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            items.append(int(p, 0))
        except ValueError:
            return None
    return items


def _infer_mrgsort_block_len(kernel_text: str) -> Optional[int]:
    """
    Try to infer the compile-time blockLen argument passed to:
        TMRGSORT(dst, src, blockLen)

    Most PTOAS-generated kernels use a constant like:
        int32_t v3 = 64;
        TMRGSORT(v22, v21, v3);
    """
    call = re.search(r"\bTMRGSORT\s*\(\s*\w+\s*,\s*\w+\s*,\s*([^)]+?)\s*\)", kernel_text)
    if not call:
        return None
    arg = call.group(1).strip()
    # Direct literal.
    if re.fullmatch(r"(?:0x[0-9A-Fa-f]+|\d+)", arg):
        try:
            return int(arg, 0)
        except ValueError:
            return None

    # Identifier that is defined as a constant earlier in the kernel.
    if not re.fullmatch(r"[A-Za-z_]\w*", arg):
        return None
    match = re.search(rf"\b(?:int32_t|uint32_t|int|unsigned)\s+{re.escape(arg)}\s*=\s*(0x[0-9A-Fa-f]+|\d+)\s*;", kernel_text)
    if not match:
        return None
    try:
        return int(match.group(1), 0)
    except ValueError:
        return None


def _required_elements_for_shape_stride(shape_dims, stride_dims) -> Optional[int]:
    if not shape_dims or not stride_dims:
        return None
    n = min(len(shape_dims), len(stride_dims))
    req = 1
    for i in range(n):
        dim = shape_dims[i]
        stride = stride_dims[i]
        if not isinstance(dim, int) or not isinstance(stride, int):
            return None
        if dim < 0:
            return None
        if dim <= 1:
            continue
        if stride < 0:
            return None
        req += (dim - 1) * stride
    return max(req, 1)


def _sanitize_int_expr(expr: str) -> str:
    expr = expr.strip()
    # Strip common C-style integer casts found in PTOAS-generated code.
    expr = re.sub(
        r"\(\s*(?:unsigned|int|long|size_t|int(?:8|16|32|64)_t|uint(?:8|16|32|64)_t)\s*\)",
        "",
        expr,
    )
    # Strip integer literal suffixes (u/l/ul/ull...).
    expr = re.sub(r"(\b0x[0-9A-Fa-f]+|\b\d+)(?:[uUlL]+)\b", r"\1", expr)
    return expr.strip()


def _safe_eval_int_expr(expr: str, env: dict) -> Optional[int]:
    """
    Best-effort evaluate a C-like integer expression using values from `env`.

    Returns None if the expression contains unknown identifiers or unsupported
    constructs.
    """
    expr = _sanitize_int_expr(expr)
    if not expr:
        return None

    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, bool)):
                return int(node.value)
            return None
        if isinstance(node, ast.Name):
            if node.id in env and env[node.id] is not None:
                return int(env[node.id])
            return None
        if isinstance(node, ast.UnaryOp):
            val = ev(node.operand)
            if val is None:
                return None
            if isinstance(node.op, ast.UAdd):
                return +val
            if isinstance(node.op, ast.USub):
                return -val
            return None
        if isinstance(node, ast.BinOp):
            left = ev(node.left)
            right = ev(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.FloorDiv):
                return left // right if right != 0 else None
            if isinstance(node.op, ast.Mod):
                return left % right if right != 0 else None
            if isinstance(node.op, ast.LShift):
                return left << right
            if isinstance(node.op, ast.RShift):
                return left >> right
            if isinstance(node.op, ast.BitAnd):
                return left & right
            if isinstance(node.op, ast.BitOr):
                return left | right
            if isinstance(node.op, ast.BitXor):
                return left ^ right
            return None
        return None

    return ev(parsed)


def _infer_int_var_maxima(kernel_text: str, seed_env: Optional[dict] = None) -> dict:
    """
    Infer max values for simple integer temporaries (e.g. v23) used in pointer
    arithmetic, by evaluating constant-ish assignments and simple for-loop ranges.

    This is used to size GM buffers conservatively for CPU/NPU runs.
    """
    assigns = []

    int_vars = set()
    for m in re.finditer(
        r"\b(?:bool|unsigned|int|long|size_t|int(?:8|16|32|64)_t|uint(?:8|16|32|64)_t)\s+(\w+)\s*(?:=\s*[^;]+)?;",
        kernel_text,
    ):
        int_vars.add(m.group(1))

    # Typed initialization (non-hoisted case).
    for m in re.finditer(
        r"\b(?:bool|unsigned|int|long|size_t|int(?:8|16|32|64)_t|uint(?:8|16|32|64)_t)\s+(\w+)\s*=\s*([^;]+);",
        kernel_text,
    ):
        name = m.group(1)
        expr = m.group(2).strip()
        assigns.append((name, expr))

    # declareVariablesAtTop hoists declarations, leaving untyped assignments like:
    #   v34 = v29 + v33;
    for m in re.finditer(r"\b(\w+)\s*=\s*([^;]+);", kernel_text):
        name = m.group(1)
        if name not in int_vars:
            continue
        expr = m.group(2).strip()
        assigns.append((name, expr))

    loops = []
    for m in re.finditer(r"\bfor\s*\(", kernel_text):
        open_paren = kernel_text.find("(", m.start())
        if open_paren < 0:
            continue
        close_paren = _find_matching_paren(kernel_text, open_paren)
        if close_paren is None:
            continue
        header = kernel_text[open_paren + 1:close_paren]
        parts = _split_top_level(header, ";")
        if len(parts) != 3:
            continue
        init, cond, step = parts
        init_m = re.match(
            r"^\s*(?:unsigned|int|long|size_t|int(?:8|16|32|64)_t|uint(?:8|16|32|64)_t)\s+(\w+)\s*=\s*(.+?)\s*$",
            init,
        )
        if not init_m:
            continue
        ind = init_m.group(1)
        cond_m = re.match(rf"^\s*{re.escape(ind)}\s*<\s*(.+?)\s*$", cond)
        step_m = re.match(rf"^\s*{re.escape(ind)}\s*\+=\s*(.+?)\s*$", step)
        if not cond_m or not step_m:
            continue
        loops.append((ind, init_m.group(2).strip(), cond_m.group(1).strip(), step_m.group(1).strip()))

    maxima: dict[str, Optional[int]] = {
        k: (None if v is None else int(v))
        for k, v in (seed_env or {}).items()
    }

    def set_max(name: str, value: int) -> bool:
        cur = maxima.get(name)
        if cur is None or value > cur:
            maxima[name] = value
            return True
        return False

    changed = True
    for _ in range(64):
        if not changed:
            break
        changed = False
        for name, expr in assigns:
            val = _safe_eval_int_expr(expr, maxima)
            if val is None:
                continue
            if set_max(name, val):
                changed = True

        for ind, start, end, step in loops:
            start_v = _safe_eval_int_expr(start, maxima)
            end_v = _safe_eval_int_expr(end, maxima)
            step_v = _safe_eval_int_expr(step, maxima)
            if start_v is None or end_v is None or step_v is None:
                continue
            if step_v == 0:
                continue
            if step_v > 0:
                if end_v <= start_v:
                    max_ind = start_v
                else:
                    span = end_v - start_v - 1
                    max_ind = start_v + (span // step_v) * step_v
            else:
                # Rare in these kernels; approximate with start.
                max_ind = start_v
            if set_max(ind, max_ind):
                changed = True

    # Replace None with 0 for downstream best-effort arithmetic.
    return {k: (0 if v is None else int(v)) for k, v in maxima.items()}


def _infer_gm_pointer_elem_counts(kernel_text: str, pointer_param_names, seed_int_env: Optional[dict] = None):
    """
    Infer minimum element counts for each __gm__ pointer param from GlobalTensor
    shape/stride metadata found in PTOAS-generated kernels.

    This fixes cases where the logical shape is small (e.g. 32x32) but the GM
    tensor uses padded strides (e.g. row stride 256), so the kernel accesses a
    much larger linear range.
    """
    if not pointer_param_names:
        return {}

    pointer_params = set(pointer_param_names)

    int_max = _infer_int_var_maxima(kernel_text, seed_env=seed_int_env)

    pointer_like = set(pointer_param_names)
    for m in re.finditer(r"__gm__\s+[\w:<>]+\s*\*\s*(\w+)\s*(?:=[^;]+)?;", kernel_text):
        pointer_like.add(m.group(1))

    ptr_to_base_offset = {}
    for m in re.finditer(
        r"__gm__\s+[\w:<>]+\s*\*\s*(\w+)\s*=\s*(\w+)\s*\+\s*([^;]+);",
        kernel_text,
    ):
        ptr_to_base_offset[m.group(1)] = (m.group(2), m.group(3).strip())

    # declareVariablesAtTop form:
    #   __gm__ float* v35;
    #   v35 = v1 + v34;
    for m in re.finditer(r"\b(\w+)\s*=\s*(\w+)\s*\+\s*([^;]+);", kernel_text):
        lhs = m.group(1)
        base = m.group(2)
        if lhs not in pointer_like:
            continue
        if base not in pointer_like and base not in pointer_params:
            continue
        ptr_to_base_offset[lhs] = (base, m.group(3).strip())

    ptr_to_param = {}
    for m in re.finditer(
        r"__gm__\s+[\w:<>]+\s*\*\s*(\w+)\s*=\s*\(__gm__\s+[\w:<>]+\s*\*\)\s*(\w+)\b",
        kernel_text,
    ):
        ptr_to_param[m.group(1)] = m.group(2)

    for m in re.finditer(r"\b(\w+)\s*=\s*\(__gm__\s+[\w:<>]+\s*\*\)\s*(\w+)\b", kernel_text):
        lhs = m.group(1)
        rhs = m.group(2)
        if lhs not in pointer_like:
            continue
        if rhs not in pointer_like and rhs not in pointer_params:
            continue
        ptr_to_param[lhs] = rhs

    def resolve_param_and_offset(ptr: str):
        cur = ptr
        offset = 0
        seen = set()
        for _ in range(16):
            if cur in pointer_params:
                return cur, offset
            if cur in seen:
                break
            seen.add(cur)
            mapped = ptr_to_param.get(cur)
            if mapped:
                cur = mapped
                continue
            base_off = ptr_to_base_offset.get(cur)
            if base_off:
                base, off_expr = base_off
                off_val = _safe_eval_int_expr(off_expr, int_max)
                if off_val is not None:
                    offset += max(off_val, 0)
                cur = base
                continue
            break
        return None, None

    def resolve_param_and_offset_expr(ptr_expr: str):
        """
        Resolve a pointer expression passed to GlobalTensor(...) back to a GM
        pointer param name plus a conservative constant offset in elements.

        Handles common PTOAS patterns like:
          v1
          v1 + (expr)
          (v1 + expr0) + expr1
          reinterpret_cast<__gm__ float*>(v1 + expr)
          (__gm__ float*)(v1 + expr)
        """
        expr = _strip_enclosing_parens(ptr_expr.strip())
        if not expr:
            return None, None

        m = re.match(r"^(?:reinterpret_cast|static_cast)<[^>]+>\((.*)\)$", expr)
        if m:
            expr = _strip_enclosing_parens(m.group(1).strip())

        # C-style cast prefix: (__gm__ float*)expr / (float*)expr
        m = re.match(r"^\(\s*__gm__[^)]*\)\s*(.+)$", expr)
        if m:
            expr = _strip_enclosing_parens(m.group(1).strip())
        else:
            m = re.match(r"^\(\s*[\w:<> ]+\*\s*\)\s*(.+)$", expr)
            if m:
                expr = _strip_enclosing_parens(m.group(1).strip())

        plus_terms = _split_top_level(expr, "+")
        if len(plus_terms) > 1:
            for idx, term in enumerate(plus_terms):
                param, off0 = resolve_param_and_offset_expr(term)
                if not param or off0 is None:
                    continue
                offset = off0
                for other_idx, other_term in enumerate(plus_terms):
                    if other_idx == idx:
                        continue
                    off_val = _safe_eval_int_expr(other_term, int_max)
                    if off_val is None:
                        continue
                    offset += max(off_val, 0)
                return param, offset

        m = re.match(r"^(\w+)\s*\+\s*(.+)$", expr)
        if m:
            base = m.group(1)
            off_expr = m.group(2).strip()
            param, off0 = resolve_param_and_offset(base)
            if not param or off0 is None:
                return None, None
            off_val = _safe_eval_int_expr(off_expr, int_max)
            if off_val is None:
                return param, off0
            return param, off0 + max(off_val, 0)

        return resolve_param_and_offset(expr)

    def eval_int_arg_list(args_blob: str):
        values = []
        for arg in _split_cpp_args(args_blob):
            value = _safe_eval_int_expr(arg, int_max)
            if value is None:
                return None
            values.append(value)
        return values

    # Parse aliases: GTShape_*=pto::Shape<...>; GTStride_*=pto::Stride<...>;
    shape_aliases = {}
    for m in re.finditer(r"using\s+(\w+)\s*=\s*pto::Shape<([^>]*)>;", kernel_text):
        dims = _parse_int_list(m.group(2))
        if dims:
            shape_aliases[m.group(1)] = dims

    stride_aliases = {}
    for m in re.finditer(r"using\s+(\w+)\s*=\s*pto::Stride<([^>]*)>;", kernel_text):
        dims = _parse_int_list(m.group(2))
        if dims:
            stride_aliases[m.group(1)] = dims

    def collect_shape_or_stride_objects(kind: str):
        # New tile-native EmitC may materialize dynamic GM views as either:
        #   pto::Shape<..., -1, -1> s = pto::Shape<...>(..., rows, cols);
        # or declare-at-top form:
        #   pto::Shape<..., -1, -1> s;
        #   s = pto::Shape<...>(..., rows, cols);
        # The template carries `-1` placeholders, while constructor arguments
        # carry the concrete validation footprint.
        objects = {}
        ctor_pat = rf"(?:pto::)?{kind}<([^;\n]*)>\s*\(([^;]*)\)"
        init_pat = rf"\b(?:pto::)?{kind}<[^;\n]*>\s+(\w+)\s*=\s*{ctor_pat}\s*;"
        for m in re.finditer(init_pat, kernel_text):
            dims = eval_int_arg_list(m.group(3))
            if not dims:
                dims = _parse_int_list(m.group(2))
            if dims:
                objects[m.group(1)] = dims

        assign_pat = rf"\b(\w+)\s*=\s*{ctor_pat}\s*;"
        for m in re.finditer(assign_pat, kernel_text):
            dims = eval_int_arg_list(m.group(3))
            if not dims:
                dims = _parse_int_list(m.group(2))
            if dims:
                objects[m.group(1)] = dims
        return objects

    shape_objects = collect_shape_or_stride_objects("Shape")
    stride_objects = collect_shape_or_stride_objects("Stride")

    # Map GT_* alias -> (shape_alias, stride_alias)
    gt_alias_to_shape_stride = {}
    for m in re.finditer(
        # Matches both:
        #   using GT = GlobalTensor<T, ShapeAlias, StrideAlias>;
        # and the 4-param layout form:
        #   using GT = GlobalTensor<T, ShapeAlias, StrideAlias, LayoutAlias>;
        r"using\s+(\w+)\s*=\s*GlobalTensor<\s*[^,>]+\s*,\s*(\w+)\s*,\s*(\w+)\s*(?:,\s*[^>]+)?\s*>;",
        kernel_text,
    ):
        gt_alias = m.group(1)
        shape_alias = m.group(2)
        stride_alias = m.group(3)
        gt_alias_to_shape_stride[gt_alias] = (shape_alias, stride_alias)

    # Find instantiations: GT_xxx v = GT_xxx(ptr, ...)
    param_elem_counts = {}
    gt_object_to_param_offset = {}

    def record_global_tensor_footprint(gt_object: str, base_ptr_expr: str, shape_dims, stride_dims):
        req = _required_elements_for_shape_stride(shape_dims, stride_dims)
        if not req:
            return
        param, off = resolve_param_and_offset_expr(base_ptr_expr)
        if not param or off is None:
            return
        gt_object_to_param_offset[gt_object] = (param, off)
        param_elem_counts[param] = max(param_elem_counts.get(param, 0), req + max(off, 0))

    for m in re.finditer(r"\b(\w+)\s+(\w+)\s*=\s*\1\s*\(\s*(\w+)\s*,", kernel_text):
        gt_alias = m.group(1)
        gt_object = m.group(2)
        base_ptr = m.group(3)
        shape_stride = gt_alias_to_shape_stride.get(gt_alias)
        if not shape_stride:
            continue
        shape_dims = shape_aliases.get(shape_stride[0])
        stride_dims = stride_aliases.get(shape_stride[1])
        record_global_tensor_footprint(gt_object, base_ptr, shape_dims, stride_dims)

    for m in re.finditer(
        r"\b(?:pto::)?GlobalTensor<[^;\n]*>\s+(\w+)\s*=\s*(?:pto::)?GlobalTensor<[^;\n]*>\s*\(([^;]*)\);",
        kernel_text,
    ):
        gt_object = m.group(1)
        args = _split_cpp_args(m.group(2))
        if len(args) < 3:
            continue
        shape_dims = shape_objects.get(args[1].strip())
        stride_dims = stride_objects.get(args[2].strip())
        record_global_tensor_footprint(gt_object, args[0], shape_dims, stride_dims)

    # Follow aliases produced by PTOAS__GLOBAL_TENSOR_DATA(parentView). This is
    # needed for partitioned tensor views whose constructor base is derived from
    # a previously materialized GlobalTensor object.
    for m in re.finditer(
        r"__gm__\s+[\w:<>]+\s*\*\s*(\w+)\s*=\s*PTOAS__GLOBAL_TENSOR_DATA\(\s*(\w+)\s*\);",
        kernel_text,
    ):
        alias, tensor = m.group(1), m.group(2)
        mapped = gt_object_to_param_offset.get(tensor)
        if mapped:
            ptr_to_base_offset[alias] = (mapped[0], str(mapped[1]))

    for m in re.finditer(
        r"\b(?:pto::)?GlobalTensor<[^;\n]*>\s+(\w+)\s*=\s*(?:pto::)?GlobalTensor<[^;\n]*>\s*\(([^;]*)\);",
        kernel_text,
    ):
        gt_object = m.group(1)
        args = _split_cpp_args(m.group(2))
        if len(args) < 3:
            continue
        shape_dims = shape_objects.get(args[1].strip())
        stride_dims = stride_objects.get(args[2].strip())
        record_global_tensor_footprint(gt_object, args[0], shape_dims, stride_dims)

    # Newer PTOAS EmitC output (especially with declareVariablesAtTop) may avoid
    # `using GTShape = ...; using GTStride = ...;` aliases and instead embeds
    # pto::Shape/pto::Stride directly in the GlobalTensor template.
    for m in re.finditer(
        r"\b(?:pto::)?GlobalTensor<[^;\n]*(?:pto::)?Shape<([^>]*)>[^;\n]*(?:pto::)?Stride<([^>]*)>[^;\n]*>\s*\(\s*([^,]+?)\s*,",
        kernel_text,
    ):
        shape_dims = _parse_int_list(m.group(1))
        stride_dims = _parse_int_list(m.group(2))
        req = _required_elements_for_shape_stride(shape_dims, stride_dims)
        if not req:
            continue
        base_ptr_expr = m.group(3).strip()
        param, off = resolve_param_and_offset_expr(base_ptr_expr)
        if not param or off is None:
            continue
        param_elem_counts[param] = max(param_elem_counts.get(param, 0), req + max(off, 0))

    return param_elem_counts


def generate_testcase(
    input_cpp: Path,
    output_root: Optional[Path],
    testcase: str,
    run_mode: str,
    soc_version: str,
    aicore_arch: Optional[str] = None,
):
    sample_root = _resolve_sample_root(input_cpp)
    if output_root:
        output_dir = output_root / sample_root.name / testcase
    else:
        output_dir = sample_root / "npu_validation" / testcase
    output_dir.mkdir(parents=True, exist_ok=True)

    use_custom_golden = _use_custom_golden_for_case(testcase, soc_version)
    custom_golden = _find_custom_case_asset(sample_root, testcase, "golden.py") if use_custom_golden else None
    custom_compare = _find_custom_case_asset(sample_root, testcase, "compare.py") if use_custom_golden else None
    shared_validation_runtime = sample_root.parent / "validation_runtime.py"

    raw_kernel = input_cpp.read_text(encoding="utf-8")
    raw_kernel_for_analysis = raw_kernel
    kernel_info = _describe_kernel_source(raw_kernel_for_analysis)
    # pto.tcmp / pto.tcmps produce packed predicate masks and leave parts of the
    # logical u8 tile undefined. This can make byte-wise compares flaky.
    has_packed_pred_mask = re.search(r"\bTCMPS?\s*\(", raw_kernel_for_analysis) is not None
    has_dav_cube = "__DAV_CUBE__" in raw_kernel
    has_dav_vec = "__DAV_VEC__" in raw_kernel
    has_intra_block_sync = "set_intra_block(" in raw_kernel or "wait_intra_block(" in raw_kernel
    has_mixed_section_sync = has_dav_cube and has_dav_vec and has_intra_block_sync
    has_cube_only_section = has_dav_cube and not has_dav_vec
    has_vec_only_section = has_dav_vec and not has_dav_cube

    is_mixed_kernel = kernel_info["kind"] == "mixed"
    raw_params = kernel_info["raw_params"]
    pointer_param_names = [_extract_cpp_name(p) for p in raw_params if _is_gm_pointer_param(p)]
    prefetch_workspace_param_names = _detect_prefetch_workspace_pointer_params(
        raw_kernel_for_analysis, pointer_param_names
    )
    uses_prefetch_async_runtime = bool(prefetch_workspace_param_names) and "TPREFETCH_ASYNC(" in raw_kernel_for_analysis

    if aicore_arch is None:
        if is_mixed_kernel:
            sv = (soc_version or "").lower()
            if "950" in sv or "a5" in sv or "910b" in sv:
                aicore_arch = "dav-c310"
            else:
                aicore_arch = "dav-c220"
        # Sectioned kernels contain `#if defined(__DAV_CUBE__)` / `__DAV_VEC__`
        # blocks. If they also carry explicit pipe synchronization, align to
        # PTO-ISA mix-kernel compile mode (`dav-c310`) so the toolchain owns
        # DAV macro definition and pipe legality checks.
        elif has_mixed_section_sync:
            sv = (soc_version or "").lower()
            if "950" in sv or "a5" in sv:
                aicore_arch = "dav-c310"
            elif "910b" in sv:
                aicore_arch = "dav-c220"
            else:
                aicore_arch = "dav-c220"
        elif has_cube_only_section:
            # A cube-only section must keep the cube arch. Building it as vec
            # while forcing `__DAV_CUBE__` makes AIC pipe synchronization fail
            # legality checks on A5.
            sv = (soc_version or "").lower()
            if "950" in sv or "a5" in sv:
                aicore_arch = "dav-c310-cube"
            else:
                aicore_arch = "dav-c220-cube"
        elif has_vec_only_section:
            sv = (soc_version or "").lower()
            if "950" in sv or "a5" in sv:
                aicore_arch = "dav-c310-vec"
            elif "910b" in sv:
                aicore_arch = "dav-c220-vec"
            else:
                aicore_arch = "dav-c220-vec"
        elif has_dav_cube or has_dav_vec:
            # Generic multi-section kernels without mixed-kernel sync keep the
            # historical vec-arch + forced-macro path.
            sv = (soc_version or "").lower()
            if "950" in sv or "a5" in sv:
                aicore_arch = "dav-c310-vec"
            elif "910b" in sv:
                aicore_arch = "dav-c220-vec"
            else:
                aicore_arch = "dav-c220-vec"
        else:
            aicore_arch = _infer_aicore_arch(raw_kernel, soc_version)

    is_a5_soc = "950" in (soc_version or "").lower() or "a5" in (soc_version or "").lower()

    if uses_prefetch_async_runtime and (not is_a5_soc) and aicore_arch.startswith("dav-c310"):
        if aicore_arch.endswith("-cube"):
            aicore_arch = "dav-c220-cube"
        elif aicore_arch == "dav-c310":
            aicore_arch = "dav-c220"
        else:
            aicore_arch = "dav-c220-vec"

    # For single-section kernels, force-define DAV macro(s) to keep section
    # bodies visible to the selected compile arch.
    # For mix-kernel arch (dav-c310/dav-c220), do not force-define macros.
    dav_defines = ""
    is_mix_arch = aicore_arch in {"dav-c310", "dav-c220"}
    if not is_mix_arch:
        if has_dav_cube:
            dav_defines += " -D__DAV_CUBE__"
        if has_dav_vec:
            dav_defines += " -D__DAV_VEC__"

    rows, cols = _parse_shape(kernel_info["call_text"])
    logical_elem_count = rows * cols
    kernel_name = kernel_info["kernel_name"]
    mrgsort_block_len = _infer_mrgsort_block_len(raw_kernel_for_analysis) if "TMRGSORT" in raw_kernel_for_analysis else None
    inferred_void_ptr_types = {}
    for raw in raw_params:
        if not _is_gm_pointer_param(raw):
            continue
        name = _extract_cpp_name(raw)
        cpp_type = _extract_cpp_type(raw)
        if cpp_type == "void":
            inferred = _infer_void_gm_pointee_type(raw_kernel_for_analysis, name)
            if inferred:
                inferred_void_ptr_types[name] = inferred

    ffts_param_names = _detect_set_ffts_pointer_params(raw_kernel_for_analysis, pointer_param_names)
    non_runtime_pointer_param_names = [
        n
        for n in pointer_param_names
        if n not in ffts_param_names and n not in prefetch_workspace_param_names
    ]

    output_param_names = []
    for writer_text in kernel_info["writer_texts"]:
        output_param_names.extend(_detect_output_pointer_params(writer_text, non_runtime_pointer_param_names))
    output_param_names = _ordered_unique(output_param_names)
    sample_output_param_names = (
        SAMPLE_CASE_OUTPUT_PARAMS
        .get(sample_root.name.lower(), {})
        .get(testcase)
    )
    if sample_output_param_names is not None:
        output_param_names = list(sample_output_param_names)
    if not output_param_names and non_runtime_pointer_param_names:
        output_param_names = [
            non_runtime_pointer_param_names[0]
            if len(non_runtime_pointer_param_names) == 1
            else non_runtime_pointer_param_names[-1]
        ]
    output_param_name_set = set(output_param_names)

    params = []
    for raw in raw_params:
        name = _extract_cpp_name(raw)
        cpp_type = _extract_cpp_type(raw)
        if cpp_type == "void" and name in inferred_void_ptr_types:
            cpp_type = inferred_void_ptr_types[name]
        if _is_gm_pointer_param(raw):
            params.append(
                {
                    "kind": "ptr",
                    "raw": raw,
                    "name": name,
                    "cpp_type": cpp_type,
                    "host_type": _cpp_host_type(cpp_type),
                    "role": (
                        "ffts"
                        if name in ffts_param_names
                        else (
                            "prefetch_workspace"
                            if name in prefetch_workspace_param_names
                            else ("output" if name in output_param_name_set else "input")
                        )
                    ),
                }
            )
        else:
            params.append(
                {
                    "kind": "scalar",
                    "raw": raw,
                    "name": name,
                    "cpp_type": cpp_type,
                    "host_type": _cpp_host_type(cpp_type),
                }
            )

    # Initialize every GM pointer from a host-side .bin file.
    #
    # Rationale:
    # - Some kernels are in-place (single pointer param) or may read from an
    #   "output" pointer as scratch. Leaving buffers uninitialized leads to
    #   non-determinism between CPU golden and real NPU.
    data_ptrs = [p for p in params if p["kind"] == "ptr" and p["role"] not in {"ffts", "prefetch_workspace"}]
    ffts_ptrs = [p for p in params if p["kind"] == "ptr" and p["role"] == "ffts"]
    prefetch_workspace_ptrs = [p for p in params if p["kind"] == "ptr" and p["role"] == "prefetch_workspace"]
    init_ptrs = list(data_ptrs)
    output_ptrs = [p for p in data_ptrs if p["role"] == "output"]

    scalar_int_defaults = {
        p["name"]: default_value
        for p in params
        if p["kind"] == "scalar"
        for default_value in [
            _integer_scalar_default_value(
                testcase,
                p["name"],
                p["host_type"],
                sample_root.name,
            )
        ]
        if default_value is not None
    }
    inferred_counts = {}
    for analysis_text in kernel_info["analysis_texts"]:
        partial_counts = _infer_gm_pointer_elem_counts(analysis_text, pointer_param_names, seed_int_env=scalar_int_defaults)
        for name, count in partial_counts.items():
            inferred_counts[name] = max(inferred_counts.get(name, 0), count)
    pointer_count_minimums = _pointer_count_minimums_for_case(
        sample_root.name,
        testcase,
    )
    for name, count in pointer_count_minimums.items():
        inferred_counts[name] = max(inferred_counts.get(name, 0), int(count))
    ptr_elem_counts = {}
    for p in data_ptrs:
        inferred = inferred_counts.get(p["name"])
        ptr_elem_counts[p["name"]] = int(inferred) if inferred and int(inferred) > 0 else logical_elem_count
    if testcase in {"rmsnorm_incore_0", "decode_projection_incore_0"}:
        # These repro kernels partition a [16, hidden] ND view with a row
        # offset. Board validation runs a single-block case, so keep bf16
        # input/output buffers large enough for the full 16xhidden window.
        required_elems = 16 * (5120 if testcase == "rmsnorm_incore_0" else 8192)
        for p in data_ptrs:
            if p["host_type"] != "uint16_t":
                continue
            cur = int(ptr_elem_counts.get(p["name"], logical_elem_count))
            ptr_elem_counts[p["name"]] = max(cur, required_elems)
        if testcase == "decode_projection_incore_0":
            # decode_projection_incore_0 also reads gamma as f32[1, 8192].
            for p in data_ptrs:
                if p["host_type"] != "float":
                    continue
                cur = int(ptr_elem_counts.get(p["name"], logical_elem_count))
                ptr_elem_counts[p["name"]] = max(cur, 8192)

    templates_root = Path(__file__).resolve().parents[1] / "templates"
    template = (templates_root / "main_template.cpp").read_text(encoding="utf-8")
    case_name = f"case_{rows}x{cols}"

    launch_name = f"Launch{kernel_name[0].upper()}{kernel_name[1:]}"

    launch_decl_params = []
    launch_call_args = []
    for p in params:
        if p["kind"] == "ptr":
            launch_decl_params.append(f"{p['host_type']} *{p['name']}")
            launch_call_args.append(f"{p['name']}Device")
        else:
            launch_decl_params.append(f"{p['host_type']} {p['name']}")
            launch_call_args.append(p["name"])

    param_decls_lines = []
    if data_ptrs:
        for p in data_ptrs:
            elem_cnt = ptr_elem_counts.get(p["name"], logical_elem_count)
            param_decls_lines.append(f"    size_t elemCount_{p['name']} = {elem_cnt};")
            param_decls_lines.append(
                f"    size_t fileSize_{p['name']} = elemCount_{p['name']} * sizeof({p['host_type']});"
            )

    for p in params:
        if p["kind"] != "scalar":
            continue
        t = p["host_type"]
        if testcase in {"rmsnorm_incore_0", "decode_projection_incore_0"} and t in {
            "int8_t",
            "uint8_t",
            "int16_t",
            "uint16_t",
            "int32_t",
            "uint32_t",
            "int64_t",
            "uint64_t",
            "int",
            "unsigned",
            "size_t",
        }:
            # These kernels use this scalar as row offset (%arg3).
            # Keep it at 0 for single-block validation to avoid shifted windows.
            value = "0"
            param_decls_lines.append(f"    {t} {p['name']} = {value};")
            continue
        # Some PTO-ISA APIs use small POD structs as scalar parameters.
        # Example: pto::MrgSortExecutedNumList (used by TMRGSORT multi-list variants).
        if t.endswith("MrgSortExecutedNumList"):
            # A zero-initialized executed list can lead to illegal configurations
            # and runtime exceptions for TMRGSORT format2 on NPU. Default to "all
            # lists full" for our generated samples (each list holds 128 packed
            # structures in the standard 1x256 f32 representation).
            param_decls_lines.append(f"    {t} {p['name']}{{128, 128, 128, 128}};")
            continue
        if t == "bool":
            bool_override = _bool_scalar_default_value(testcase, p["name"])
            value = "true" if bool_override is None else ("true" if bool_override else "false")
        elif re.match(r"^(u?int)(8|16|32|64)_t$", t) or t in {"int", "unsigned", "size_t"}:
            int_override = _integer_scalar_default_value(
                testcase,
                p["name"],
                t,
                sample_root.name,
            )
            value = "1" if int_override is None else str(int_override)
        elif t in {"float"}:
            value = "1.0f"
        elif t in {"double"}:
            value = "1.0"
        else:
            value = "0"
        param_decls_lines.append(f"    {t} {p['name']} = {value};")

    for p in params:
        if p["kind"] != "ptr":
            continue
        if p["role"] == "ffts":
            param_decls_lines.append(f"    {p['host_type']} *{p['name']}Device = nullptr;")
            param_decls_lines.append(f"    uint64_t {p['name']}FftsAddr = 0;")
            param_decls_lines.append(f"    uint32_t {p['name']}FftsLen = 0;")
        elif p["role"] == "prefetch_workspace":
            param_decls_lines.append(f"    {p['host_type']} *{p['name']}Device = nullptr;")
        else:
            param_decls_lines.append(f"    {p['host_type']} *{p['name']}Host = nullptr;")
            param_decls_lines.append(f"    {p['host_type']} *{p['name']}Device = nullptr;")

    alloc_host = []
    alloc_device = []
    init_runtime_ptrs = []
    free_host = []
    free_device = []
    for p in data_ptrs:
        size_var = f"fileSize_{p['name']}"
        alloc_host.append(
            f"    ACL_CHECK(aclrtMallocHost((void **)(&{p['name']}Host), {size_var}));"
        )
        alloc_device.append(
            f"    ACL_CHECK(aclrtMalloc((void **)&{p['name']}Device, {size_var}, ACL_MEM_MALLOC_HUGE_FIRST));"
        )
        free_device.append(f"    aclrtFree({p['name']}Device);")
        free_host.append(f"    aclrtFreeHost({p['name']}Host);")
    for p in ffts_ptrs:
        init_runtime_ptrs.append(
            f"    if (const rtError_t _rt = rtGetC2cCtrlAddr(&{p['name']}FftsAddr, &{p['name']}FftsLen); _rt != RT_ERROR_NONE) {{"
        )
        init_runtime_ptrs.append(
            f"        std::fprintf(stderr, \"[ERROR] rtGetC2cCtrlAddr failed for {p['name']}: %d (%s:%d)\\n\", (int)_rt, __FILE__, __LINE__);"
        )
        init_runtime_ptrs.append("        rc = 1;")
        init_runtime_ptrs.append("        goto cleanup;")
        init_runtime_ptrs.append("    }")
        init_runtime_ptrs.append(
            f"    {p['name']}Device = reinterpret_cast<{p['host_type']} *>({p['name']}FftsAddr);"
        )
    if prefetch_workspace_ptrs:
        param_decls_lines.append("    pto::comm::sdma::SdmaWorkspaceManager sdmaMgr;")
        param_decls_lines.append("    bool sdmaWorkspaceOk = false;")
        param_decls_lines.append('    const char *sdmaSocVersion = std::getenv("SOC_VERSION");')
        param_decls_lines.append('    const char *ptoasBoardIsA3 = std::getenv("PTOAS_BOARD_IS_A3");')
        param_decls_lines.append(
            '    const bool skipSdmaWorkspaceInit = (std::getenv("PTO_DISABLE_SDMA_WORKSPACE_INIT") != nullptr) || '
            '(ptoasBoardIsA3 != nullptr && std::strcmp(ptoasBoardIsA3, "1") == 0) || '
            '(sdmaSocVersion != nullptr && (std::strstr(sdmaSocVersion, "950") != nullptr || '
            'std::strstr(sdmaSocVersion, "A5") != nullptr || std::strstr(sdmaSocVersion, "a5") != nullptr));'
        )
        init_runtime_ptrs.append("    if (skipSdmaWorkspaceInit) {")
        init_runtime_ptrs.append(
            '        std::fprintf(stderr, "[WARN] Skip SdmaWorkspaceManager::Init on this platform - TPREFETCH_ASYNC will fall back to no-op prefetch\\n");'
        )
        init_runtime_ptrs.append("    } else {")
        init_runtime_ptrs.append("        sdmaWorkspaceOk = sdmaMgr.Init();")
        init_runtime_ptrs.append("    }")
        init_runtime_ptrs.append("    if (!skipSdmaWorkspaceInit && !sdmaWorkspaceOk) {")
        init_runtime_ptrs.append(
            '        std::fprintf(stderr, "[WARN] SdmaWorkspaceManager::Init failed - TPREFETCH_ASYNC will fall back to no-op prefetch\\n");'
        )
        init_runtime_ptrs.append("    }")
        for p in prefetch_workspace_ptrs:
            init_runtime_ptrs.append(
                f"    {p['name']}Device = sdmaWorkspaceOk ? reinterpret_cast<{p['host_type']} *>(sdmaMgr.GetWorkspaceAddr()) : nullptr;"
            )
            init_runtime_ptrs.append(f"    if (sdmaWorkspaceOk && {p['name']}Device == nullptr) {{")
            init_runtime_ptrs.append(
                f'        std::fprintf(stderr, "[ERROR] SDMA workspace address is null for {p["name"]}\\n");'
            )
            init_runtime_ptrs.append("        rc = 1;")
            init_runtime_ptrs.append("        goto cleanup;")
            init_runtime_ptrs.append("    }")
        free_device.append("    sdmaMgr.Finalize();")

    read_inputs = []
    copy_inputs = []
    for p in init_ptrs:
        size_var = f"fileSize_{p['name']}"
        read_inputs.append(
            f"    ReadFile(\"./{p['name']}.bin\", {size_var}, {p['name']}Host, {size_var});"
        )
        copy_inputs.append(
            f"    ACL_CHECK(aclrtMemcpy({p['name']}Device, {size_var}, {p['name']}Host, {size_var}, ACL_MEMCPY_HOST_TO_DEVICE));"
        )

    output_copy_back = []
    output_write = []
    for p in output_ptrs:
        size_var = f"fileSize_{p['name']}"
        output_copy_back.append(
            f"    ACL_CHECK(aclrtMemcpy({p['name']}Host, {size_var}, {p['name']}Device, {size_var}, ACL_MEMCPY_DEVICE_TO_HOST));"
        )
        output_write.append(
            f"    WriteFile(\"./{p['name']}.bin\", {p['name']}Host, {size_var});"
        )

    runtime_rt_include = '#include "runtime/rt.h"' if ffts_ptrs else ""
    runtime_host_include_dirs = ""
    if ffts_ptrs:
        runtime_host_include_dirs = "    ${ASCEND_HOME_PATH}/pkg_inc/runtime\n"

    param_decls = "\n".join(param_decls_lines)
    runtime_rt_include = ""
    if ffts_ptrs:
        # `rtGetC2cCtrlAddr` is provided by CANN runtime. Use ccelib runtime
        # header here instead of `runtime/rt.h` to avoid environment-specific
        # include path issues on some board images.
        runtime_rt_include = '#include <stdint.h>\n#include <ccelib/common/runtime.h>'
    if prefetch_workspace_ptrs:
        runtime_rt_include = (
            runtime_rt_include + '\n#include "pto/npu/comm/async/sdma/sdma_workspace_manager.hpp"'
            if runtime_rt_include
            else '#include "pto/npu/comm/async/sdma/sdma_workspace_manager.hpp"'
        )
    cann_extra_link_dirs = """set(PTO_CANN_EXTRA_LINK_DIRS "")
if(DEFINED ENV{PTO_CANN_EXTRA_LINK_DIRS} AND NOT "$ENV{PTO_CANN_EXTRA_LINK_DIRS}" STREQUAL "")
    string(REPLACE ":" ";" PTO_CANN_EXTRA_LINK_DIRS "$ENV{PTO_CANN_EXTRA_LINK_DIRS}")
endif()
"""
    main_cpp = (
        template
        .replace("@RUNTIME_RT_INCLUDE@", runtime_rt_include)
        .replace("@TEST_SUITE@", testcase.upper())
        .replace("@CASE_NAME@", case_name)
        .replace("@RUNTIME_RT_INCLUDE@", runtime_rt_include)
        .replace(
            "@LAUNCH_DECL@",
            f'extern "C" void {launch_name}({", ".join(launch_decl_params + ["void *stream"])});',
        )
        .replace("@PARAM_DECLS@", param_decls)
        .replace("@ALLOC_HOST@", "\n".join(alloc_host))
        .replace("@ALLOC_DEVICE@", "\n".join(alloc_device))
        .replace("@INIT_RUNTIME_PTRS@", "\n".join(init_runtime_ptrs))
        .replace("@READ_INPUTS@", "\n".join(read_inputs))
        .replace("@COPY_TO_DEVICE@", "\n".join(copy_inputs))
        .replace(
            "@LAUNCH_CALL@",
            f"    {launch_name}({', '.join(launch_call_args + ['stream'])});",
        )
        .replace("@COPY_BACK@", "\n".join(output_copy_back))
        .replace("@WRITE_OUTPUT@", "\n".join(output_write))
        .replace("@FREE_DEVICE@", "\n".join(free_device))
        .replace("@FREE_HOST@", "\n".join(free_host))
    )
    (output_dir / "main.cpp").write_text(main_cpp, encoding="utf-8")

    golden_template = (templates_root / "golden_template.py").read_text(encoding="utf-8")
    input_generate = []
    elem_count = logical_elem_count
    kernel_has_tscatter = "TSCATTER" in raw_kernel
    kernel_has_tgather = "TGATHER" in raw_kernel
    kernel_has_tgatherb = "TGATHERB" in raw_kernel
    kernel_has_mscatter = "MSCATTER" in raw_kernel
    kernel_has_mgather = "MGATHER" in raw_kernel
    # Some kernels use an integer tensor as "indices". The safe in-range domain
    # depends on the op semantics:
    # - TSCATTER: use a deterministic, collision-free permutation so NPU-vs-NPU
    #   golden mode stays stable across runs.
    # - TGATHER: indices are linear indices in [0, rows*cols).
    # - TGATHERB: offsets are block addresses (bytes), not per-element indices.
    index_mod = None
    if kernel_has_tscatter:
        index_mod = max(elem_count, 1)
    elif kernel_has_tgather and not kernel_has_tgatherb:
        index_mod = max(elem_count, 1)
    mgather_table_input = None
    if kernel_has_mgather:
        for p in init_ptrs:
            if p.get("role") == "input":
                mgather_table_input = p
                break
    mscatter_indices_input = None
    mscatter_output = output_ptrs[0] if kernel_has_mscatter and output_ptrs else None
    if kernel_has_mscatter:
        for p in reversed(init_ptrs):
            p_dtype = _np_dtype_for_cpp(p["cpp_type"])
            if p.get("role") == "input" and (
                p_dtype.startswith("np.int") or p_dtype.startswith("np.uint")
            ):
                mscatter_indices_input = p
                break
        if mscatter_output is not None:
            index_mod = max(
                int(ptr_elem_counts.get(mscatter_output["name"], logical_elem_count)),
                1,
            )
    mrgsort_packed = "TMRGSORT" in raw_kernel
    for p in init_ptrs:
        np_dtype = _np_dtype_for_cpp(p["cpp_type"])
        name = p["name"]
        size = ptr_elem_counts.get(name, elem_count)
        is_output = p.get("role") == "output"
        is_integer = np_dtype.startswith("np.int") or np_dtype.startswith("np.uint")
        is_tscatter_indices = kernel_has_tscatter and p.get("role") == "input" and is_integer and size == elem_count
        is_mscatter_indices = (
            kernel_has_mscatter
            and mscatter_indices_input is not None
            and name == mscatter_indices_input["name"]
        )
        is_mgather_indices = (
            kernel_has_mgather
            and mgather_table_input is not None
            and p.get("role") == "input"
            and is_integer
            and name != mgather_table_input["name"]
        )
        is_tgatherb_offset = kernel_has_tgatherb and p.get("role") == "input" and is_integer and size < elem_count
        is_tgatherb_src = kernel_has_tgatherb and p.get("role") == "input" and not is_tgatherb_offset
        # If the kernel has both inputs and outputs, default to zero-init for
        # output buffers to match pto-isa ST conventions (and improve determinism).
        zero_init = is_output and len(init_ptrs) > 1

        if zero_init:
            input_generate.append(f"    {name} = np.zeros(({size},), dtype={np_dtype})")
            input_generate.append(f"    {name}.tofile(\"{name}.bin\")")
        elif mrgsort_packed and (not is_output) and np_dtype in ("np.float32", "np.float16"):
            input_generate.append(f"    # TMRGSORT expects packed (value, index) structures (8 bytes each).")
            input_generate.append(f"    # Generate per-block sorted inputs to match pto-isa ST data layout.")
            if np_dtype == "np.float32":
                input_generate.append(f"    {name}__words_per_struct = 2  # float32(4B) + uint32(4B)")
                input_generate.append(f"    {name}__struct_dtype = np.dtype([('v', np.float32), ('i', np.uint32)])")
                input_generate.append(f"    {name}__value_dtype = np.float32")
            else:
                input_generate.append(f"    {name}__words_per_struct = 4  # float16(2B) + pad(2B) + uint32(4B)")
                input_generate.append(
                    f"    {name}__struct_dtype = np.dtype([('v', np.float16), ('pad', np.uint16), ('i', np.uint32)])"
                )
                input_generate.append(f"    {name}__value_dtype = np.float16")

            input_generate.append(f"    {name}__struct_count = {size} // {name}__words_per_struct")
            # Two modes:
            #   - Single-list format (TMRGSORT(dst, src, blockLen)): input is arranged in
            #     4 blocks and each block is sorted independently.
            #   - Multi-list format (TMRGSORT(dst, executed, tmp, src0..)): each input list
            #     is fully sorted.
            mrgsort_single = mrgsort_block_len is not None
            if mrgsort_single:
                input_generate.append(f"    {name}__block_len = {mrgsort_block_len}")
                input_generate.append(f"    {name}__structs_per_block = {name}__block_len // {name}__words_per_struct")
            input_generate.append(
                f"    {name}__values = np.random.uniform(low=0, high=1, size=({name}__struct_count,)).astype({name}__value_dtype)"
            )
            input_generate.append(f"    {name}__idx = np.arange({name}__struct_count, dtype=np.uint32)")
            if mrgsort_single:
                input_generate.append(f"    if {name}__structs_per_block > 0 and {name}__struct_count > 0:")
                input_generate.append(f"        pad = (-{name}__struct_count) % {name}__structs_per_block")
                input_generate.append(f"        if pad:")
                input_generate.append(
                    f"            {name}__values = np.concatenate(({name}__values, np.zeros(pad, dtype={name}__values.dtype)))"
                )
                input_generate.append(
                    f"            {name}__idx = np.concatenate(({name}__idx, np.zeros(pad, dtype={name}__idx.dtype)))"
                )
                input_generate.append(f"        v = {name}__values.reshape(-1, {name}__structs_per_block)")
                input_generate.append(f"        i = {name}__idx.reshape(-1, {name}__structs_per_block)")
                input_generate.append(f"        order = np.argsort(-v, kind='stable', axis=1)")
                input_generate.append(
                    f"        {name}__values = np.take_along_axis(v, order, axis=1).reshape(-1)[:{name}__struct_count]"
                )
                input_generate.append(
                    f"        {name}__idx = np.take_along_axis(i, order, axis=1).reshape(-1)[:{name}__struct_count]"
                )
            else:
                input_generate.append(f"    if {name}__struct_count > 0:")
                input_generate.append(f"        order = np.argsort(-{name}__values, kind='stable')")
                input_generate.append(f"        {name}__values = {name}__values[order]")
                input_generate.append(f"        {name}__idx = {name}__idx[order]")
            input_generate.append(f"    {name}__packed = np.empty(({name}__struct_count,), dtype={name}__struct_dtype)")
            input_generate.append(f"    {name}__packed['v'] = {name}__values")
            if np_dtype == "np.float16":
                input_generate.append(f"    {name}__packed['pad'] = np.uint16(0)")
            input_generate.append(f"    {name}__packed['i'] = {name}__idx")
            input_generate.append(f"    {name}__packed.tofile(\"{name}.bin\")")
        elif is_tscatter_indices:
            input_generate.append(f"    {name}__cols = np.arange({cols}, dtype=np.int64).reshape(1, {cols})")
            input_generate.append(f"    {name}__row_perm = np.random.permutation({rows}).astype(np.int64).reshape({rows}, 1)")
            input_generate.append(
                f"    {name} = ({name}__row_perm * {cols} + {name}__cols).astype({np_dtype}).reshape(-1)"
            )
            input_generate.append(f"    {name}.tofile(\"{name}.bin\")")
        elif is_mscatter_indices:
            out_count = (
                int(ptr_elem_counts.get(mscatter_output["name"], logical_elem_count))
                if mscatter_output is not None
                else max(size, 1)
            )
            input_generate.append(
                f"    {name} = (np.arange({size}, dtype=np.int64) % {out_count}).astype({np_dtype}, copy=False)"
            )
            input_generate.append(f"    {name}.tofile(\"{name}.bin\")")
        elif is_mgather_indices:
            table_count = (
                int(ptr_elem_counts.get(mgather_table_input['name'], logical_elem_count))
                if mgather_table_input is not None
                else max(size, 1)
            )
            input_generate.append(
                f"    {name} = (np.arange({size}, dtype=np.int64) % {table_count}).astype({np_dtype}, copy=False)"
            )
            input_generate.append(f"    {name}.tofile(\"{name}.bin\")")
        elif is_tgatherb_offset:
            input_generate.append(f"    {name} = (np.arange({size}, dtype=np.uint32) * 32).astype({np_dtype})")
            input_generate.append(f"    {name}.tofile(\"{name}.bin\")")
        elif is_tgatherb_src:
            if is_integer:
                input_generate.append(f"    {name} = np.arange({size}, dtype=np.int64).astype({np_dtype})")
            else:
                input_generate.append(f"    {name} = np.arange({size}, dtype=np.float32).astype({np_dtype})")
            input_generate.append(f"    {name}.tofile(\"{name}.bin\")")
        elif is_integer:
            if index_mod is not None:
                input_generate.append(
                    f"    {name} = (np.arange({size}, dtype=np.int64) % {index_mod}).astype({np_dtype})"
                )
            else:
                input_generate.append(f"    {name} = np.zeros(({size},), dtype={np_dtype})")
            input_generate.append(f"    {name}.tofile(\"{name}.bin\")")
        else:
            input_generate.append(f"    {name} = np.random.random(size=({size},)).astype({np_dtype})")
            input_generate.append(f"    {name}.tofile(\"{name}.bin\")")

    golden_dst = output_dir / "golden.py"
    if custom_golden is not None:
        _copy_asset_if_needed(custom_golden, golden_dst)
    else:
        golden_py = golden_template.replace("@INPUT_GENERATE@", "\n".join(input_generate))
        golden_dst.write_text(golden_py, encoding="utf-8")
    if custom_golden is not None or custom_compare is not None:
        _copy_custom_golden_helpers(sample_root, output_dir)
        if shared_validation_runtime.is_file():
            _copy_asset_if_needed(shared_validation_runtime, output_dir / "validation_runtime.py")

    # Emit the kernel source, optionally injecting a packed-predicate preload to
    # make TCMP/TCMPS outputs deterministic for byte-wise compares.
    kernel_text_out = raw_kernel_for_analysis
    if has_packed_pred_mask and output_ptrs:
        # Only handle the common packed-mask case (u8 output).
        mask_out = next((p for p in output_ptrs if p["cpp_type"] == "uint8_t"), None)
        if mask_out is not None:
            m = re.search(r"\bTCMPS?\s*\(\s*(\w+)\s*,", raw_kernel_for_analysis)
            if m:
                kernel_text_out = _inject_packed_pred_mask_preload(
                    kernel_text_out,
                    dst_tile=m.group(1),
                    output_ptr=mask_out["name"],
                    output_cpp_type=mask_out["cpp_type"],
                    rows=rows,
                    cols=cols,
                    logical_elem_count=logical_elem_count,
                )

    if kernel_info.get("needs_global_wrapper"):
        kernel_text_out = _append_single_kernel_global_wrapper(
            kernel_text_out,
            kernel_name,
            raw_params,
        )

    if is_mixed_kernel:
        kernel_text_out = _append_mixed_kernel_wrapper(
            kernel_text_out,
            kernel_name,
            raw_params,
            kernel_info["aic_text"],
            kernel_info["aiv_text"],
            kernel_info.get("aiv_local_param"),
        )

    kernel_definition = next(
        (
            function
            for function in _extract_aicore_functions(kernel_text_out)
            if function["name"] == kernel_name and function["is_global"]
        ),
        None,
    )
    kernel_linkage = (
        'extern "C" '
        if kernel_definition is not None and kernel_definition["is_extern_c"]
        else ""
    )

    kernel_out = output_dir / f"{testcase}_kernel.cpp"
    kernel_out.write_text(_replace_includes(kernel_text_out), encoding="utf-8")

    launch_fn_params = ", ".join(launch_decl_params + ["void *stream"])
    kernel_call_args_device = []
    kernel_call_args_host = []
    for p in params:
        if p["kind"] == "ptr":
            cast_ty = _strip_param_name(p["raw"], p["name"])
            kernel_call_args_device.append(f"({cast_ty}){p['name']}")
            kernel_call_args_host.append(f"({_rewrite_host_unsupported_types(cast_ty)}){p['name']}")
        else:
            kernel_call_args_device.append(p["name"])
            kernel_call_args_host.append(p["name"])
    kernel_call_args_device = ", ".join(kernel_call_args_device)
    kernel_call_args_host = ", ".join(kernel_call_args_host)
    raw_params_host = [_rewrite_host_unsupported_types(p) for p in raw_params]
    launch_block_count = _infer_launch_block_count(raw_kernel_for_analysis, testcase)
    launch_cpp = (
        INCLUDE_REPLACEMENT
        + "\n"
        "#if defined(__CCE_AICORE__)\n"
        f"{kernel_linkage}__global__ AICORE void {kernel_name}({', '.join(raw_params)});\n"
        "#else\n"
        f"{kernel_linkage}__global__ AICORE void {kernel_name}({', '.join(raw_params_host)});\n"
        "#endif\n\n"
        f"extern \"C\" void {launch_name}({launch_fn_params}) {{\n"
        "#if defined(__CCE_AICORE__)\n"
        f"    {kernel_name}<<<{launch_block_count}, nullptr, stream>>>({kernel_call_args_device});\n"
        "#else\n"
        f"    {kernel_name}<<<{launch_block_count}, nullptr, stream>>>({kernel_call_args_host});\n"
        "#endif\n"
        f"}}\n"
    )
    (output_dir / "launch.cpp").write_text(launch_cpp, encoding="utf-8")

    # pto-isa selects instruction implementations based on MEMORY_BASE vs
    # REGISTER_BASE. A3 board validation follows the official a2a3 ST setup,
    # so Ascend910B still uses MEMORY_BASE. Only A5-class targets use
    # REGISTER_BASE here.
    mem_base_define = "MEMORY_BASE"
    sv = (soc_version or "").lower()
    if "950" in sv or "a5" in sv:
        mem_base_define = "REGISTER_BASE"
    if uses_prefetch_async_runtime and not is_a5_soc:
        mem_base_define = "MEMORY_BASE"

    # CCE printing support is gated behind `--cce-enable-print` on some bisheng
    # toolchains. Only enable it when kernels emit printf.
    needs_cce_print = bool(re.search(r"\b(?:bisheng::)?cce::printf\s*\(", raw_kernel_for_analysis))
    cce_enable_print_opt = "    --cce-enable-print" if needs_cce_print else ""
    cce_print_define_opt = "    -DPTOAS_ENABLE_CCE_PRINT=1" if needs_cce_print else ""

    cce_stack_size_opt = ""
    # `-mllvm -cce-aicore-stack-size=...` is rejected on some targets (e.g.
    # dav-l310 / dav-l311).
    if not aicore_arch.startswith(("dav-l310", "dav-l311")):
        cce_stack_size_opt = '    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"\n'

    cmake_content = f"""
cmake_minimum_required(VERSION 3.16)

# Prefer setting compilers before project() so CMake picks up bisheng correctly.
set(CMAKE_C_COMPILER bisheng)
set(CMAKE_CXX_COMPILER bisheng)

project({testcase}_npu_validation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION Ascend910)
endif()
option(ENABLE_SIM_GOLDEN "Build Ascend simulator (camodel) executable" ON)

if(NOT DEFINED ENV{{ASCEND_HOME_PATH}})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please source the CANN set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{{ASCEND_HOME_PATH}})
endif()

set(PTO_ISA_ROOT "" CACHE PATH "Path to pto-isa repo")
if(NOT PTO_ISA_ROOT)
    set(_PTO_ISA_CANDIDATES
        "${{CMAKE_CURRENT_LIST_DIR}}/../../../../pto-isa"
        "${{CMAKE_CURRENT_LIST_DIR}}/../../../../../pto-isa"
        "${{CMAKE_CURRENT_LIST_DIR}}/../../../../../../pto-isa"
    )
    foreach(_cand IN LISTS _PTO_ISA_CANDIDATES)
        if(EXISTS "${{_cand}}/include" AND EXISTS "${{_cand}}/tests/common")
            set(PTO_ISA_ROOT "${{_cand}}" CACHE PATH "Path to pto-isa repo" FORCE)
            break()
        endif()
    endforeach()
endif()
if(NOT PTO_ISA_ROOT)
    message(FATAL_ERROR "Cannot find PTO_ISA_ROOT, please pass -DPTO_ISA_ROOT=/path/to/pto-isa.")
endif()

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong
    -fPIC
)
add_link_options(
    -s
    -Wl,-z,relro
    -Wl,-z,now
)

	set(CMAKE_CCE_COMPILE_OPTIONS
	    -xcce
	    -fenable-matrix
	    --cce-aicore-enable-tl
	{cce_enable_print_opt}
	{cce_print_define_opt}
	    -fPIC
	    -Xhost-start -Xhost-end
	{cce_stack_size_opt}\
	    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
	    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

include_directories(
    ${{PTO_ISA_ROOT}}/include
    ${{PTO_ISA_ROOT}}/tests/common
    ${{ASCEND_HOME_PATH}}/include
    ${{ASCEND_DRIVER_PATH}}/kernel/inc
)

{cann_extra_link_dirs}

	add_library({testcase}_kernel SHARED {testcase}_kernel.cpp launch.cpp)
	target_compile_options({testcase}_kernel PRIVATE ${{CMAKE_CCE_COMPILE_OPTIONS}} --cce-aicore-arch={aicore_arch}{dav_defines} -D{mem_base_define} -std=c++17)
	target_include_directories({testcase}_kernel PRIVATE
	    ${{ASCEND_HOME_PATH}}/pkg_inc/
	    ${{ASCEND_HOME_PATH}}/pkg_inc/profiling/
	    ${{ASCEND_HOME_PATH}}/pkg_inc/runtime/runtime
	)
target_link_options({testcase}_kernel PRIVATE --cce-fatobj-link)

add_executable({testcase} main.cpp)
target_compile_options({testcase} PRIVATE ${{CMAKE_CPP_COMPILE_OPTIONS}})
target_include_directories({testcase} PRIVATE
    ${{PTO_ISA_ROOT}}/include
    ${{PTO_ISA_ROOT}}/tests/common
{runtime_host_include_dirs})

target_link_directories({testcase} PUBLIC
    ${{ASCEND_HOME_PATH}}/lib64
    ${{PTO_CANN_EXTRA_LINK_DIRS}}
)

find_library(PTO_NNOPBASE_LIB
    NAMES nnopbase
    HINTS ${{PTO_CANN_EXTRA_LINK_DIRS}} ${{ASCEND_HOME_PATH}}/lib64
    NO_DEFAULT_PATH
)
if(NOT PTO_NNOPBASE_LIB)
    find_library(PTO_NNOPBASE_LIB NAMES nnopbase)
endif()
if(NOT PTO_NNOPBASE_LIB)
    message(FATAL_ERROR "Cannot find libnnopbase.so. Set PTO_CANN_EXTRA_LINK_DIRS or fix ASCEND_HOME_PATH.")
endif()

target_link_libraries({testcase} PRIVATE
    {testcase}_kernel
    runtime
    stdc++ ascendcl m tiling_api platform c_sec dl ${{PTO_NNOPBASE_LIB}}
)
target_link_options({testcase} PRIVATE -Wl,--allow-shlib-undefined)

if(ENABLE_SIM_GOLDEN)
    # Simulator executable: used to generate golden outputs (Ascend camodel).
    add_executable({testcase}_sim main.cpp)
    target_compile_options({testcase}_sim PRIVATE ${{CMAKE_CPP_COMPILE_OPTIONS}})
    target_include_directories({testcase}_sim PRIVATE
        ${{PTO_ISA_ROOT}}/include
        ${{PTO_ISA_ROOT}}/tests/common
{runtime_host_include_dirs})
    target_link_directories({testcase}_sim PUBLIC
        ${{ASCEND_HOME_PATH}}/lib64
        ${{PTO_CANN_EXTRA_LINK_DIRS}}
        ${{ASCEND_HOME_PATH}}/aarch64-linux/simulator/${{SOC_VERSION}}/lib
        ${{ASCEND_HOME_PATH}}/x86_64-linux/simulator/${{SOC_VERSION}}/lib
        ${{ASCEND_HOME_PATH}}/simulator/${{SOC_VERSION}}/lib
        ${{ASCEND_HOME_PATH}}/tools/simulator/${{SOC_VERSION}}/lib
    )
    target_link_libraries({testcase}_sim PRIVATE
        {testcase}_kernel
        runtime_camodel
        stdc++ ascendcl m tiling_api platform c_sec dl ${{PTO_NNOPBASE_LIB}}
    )
    target_link_options({testcase}_sim PRIVATE -Wl,--allow-shlib-undefined)
endif()
"""
    (output_dir / "CMakeLists.txt").write_text(cmake_content.strip() + "\n", encoding="utf-8")

    compare_template = (templates_root / "compare_template.py").read_text(encoding="utf-8")
    compare_lines = ["    ok = True"]
    compare_prefix_counts = {}
    scatter_indices_input = None
    if kernel_has_tscatter:
        for p in init_ptrs:
            p_dtype = _np_dtype_for_cpp(p["cpp_type"])
            if p.get("role") == "input" and (p_dtype.startswith("np.int") or p_dtype.startswith("np.uint")):
                scatter_indices_input = p
                break
    elif kernel_has_mscatter and mscatter_indices_input is not None:
        scatter_indices_input = mscatter_indices_input
    for p in output_ptrs:
        name = p["name"]
        req = inferred_counts.get(name)
        if req is None:
            continue
        try:
            req = int(req)
        except Exception:
            continue
        if req <= 0:
            continue
        file_cnt = ptr_elem_counts.get(name, logical_elem_count)
        if file_cnt and req < int(file_cnt):
            compare_prefix_counts[name] = req
    for name, count in CASE_COMPARE_PREFIX_COUNTS.get(testcase, {}).items():
        file_cnt = int(ptr_elem_counts.get(name, logical_elem_count))
        compare_prefix_counts[name] = min(file_cnt, int(count))
    # TMRGSORT format2 testcase writes three contiguous regions:
    # 2-way (256) + 3-way (384) + 4-way (up to 512).
    # With 4-way exhausted mode, the stable worst-case valid prefix for 4-way
    # is 128 elements, so compare 256 + 384 + 128 = 768 elements.
    testcase_lc = testcase.lower()
    if testcase_lc == "mrgsort_format2":
        for p in output_ptrs:
            name = p["name"]
            file_cnt = int(ptr_elem_counts.get(name, logical_elem_count))
            compare_prefix_counts[name] = min(file_cnt, 768)
    for p in output_ptrs:
        np_dtype = _np_dtype_for_cpp(p["cpp_type"])
        name = p["name"]
        eps = _default_eps_for_cpp_type(p["cpp_type"])
        is_bf16_output = _is_bf16_cpp_type(p["cpp_type"])
        bf16_max_ulp = _default_bf16_max_ulp_for_cpp_type(p["cpp_type"])
        if (kernel_has_tscatter or kernel_has_mscatter) and scatter_indices_input is not None:
            if is_bf16_output:
                compare_lines.append(
                    f"    ok = compare_bf16_bin_at_indices(\"golden_{name}.bin\", \"{name}.bin\", {bf16_max_ulp}, "
                    f"\"{scatter_indices_input['name']}.bin\", {_np_dtype_for_cpp(scatter_indices_input['cpp_type'])}) and ok"
                )
            else:
                compare_lines.append(
                    f"    ok = compare_bin_at_indices(\"golden_{name}.bin\", \"{name}.bin\", {np_dtype}, {eps}, "
                    f"\"{scatter_indices_input['name']}.bin\", {_np_dtype_for_cpp(scatter_indices_input['cpp_type'])}) and ok"
                )
        elif has_packed_pred_mask and p["cpp_type"] in {"uint8_t", "int8_t"}:
            compare_lines.append(
                f"    ok = compare_packed_pred_mask(\"golden_{name}.bin\", \"{name}.bin\", {rows}, {cols}) and ok"
            )
        else:
            prefix_cnt = compare_prefix_counts.get(name)
            if prefix_cnt is not None:
                if is_bf16_output:
                    compare_lines.append(
                        f"    ok = compare_bf16_bin_prefix(\"golden_{name}.bin\", \"{name}.bin\", {bf16_max_ulp}, {prefix_cnt}) and ok"
                    )
                else:
                    compare_lines.append(
                        f"    ok = compare_bin_prefix(\"golden_{name}.bin\", \"{name}.bin\", {np_dtype}, {eps}, {prefix_cnt}) and ok"
                    )
            else:
                if is_bf16_output:
                    compare_lines.append(
                        f"    ok = compare_bf16_bin(\"golden_{name}.bin\", \"{name}.bin\", {bf16_max_ulp}) and ok"
                    )
                else:
                    compare_lines.append(
                        f"    ok = compare_bin(\"golden_{name}.bin\", \"{name}.bin\", {np_dtype}, {eps}) and ok"
                    )
    if testcase in {"test_intercore_sync_a5_functional", "test_intercore_sync_a5_ptoisa_vec"}:
        # Extra functional check (not just run-to-run determinism):
        # core0 writes 2.0 to output[0], core1 waits then mirrors to output[1].
        out_name = output_ptrs[0]["name"] if output_ptrs else "v1"
        compare_lines.append(f"    __inter_out = np.fromfile(\"{out_name}.bin\", dtype=np.float32)")
        compare_lines.append("    if __inter_out.size < 2:")
        compare_lines.append("        print(f\"[ERROR] intercore check requires >=2 elements, got {__inter_out.size}\")")
        compare_lines.append("        ok = False")
        compare_lines.append("    else:")
        compare_lines.append("        if abs(float(__inter_out[0]) - 2.0) > 1e-6:")
        compare_lines.append(
            "            print(f\"[ERROR] intercore check failed: out[0]={float(__inter_out[0])}, expect 2.0\")"
        )
        compare_lines.append("            ok = False")
        compare_lines.append("        if abs(float(__inter_out[1]) - 2.0) > 1e-6:")
        compare_lines.append(
            "            print(f\"[ERROR] intercore check failed: out[1]={float(__inter_out[1])}, expect 2.0\")"
        )
        compare_lines.append("            ok = False")
    compare_dst = output_dir / "compare.py"
    if custom_compare is not None:
        _copy_asset_if_needed(custom_compare, compare_dst)
    else:
        compare_py = compare_template.replace("@COMPARES@", "\n".join(compare_lines))
        compare_dst.write_text(compare_py, encoding="utf-8")
    (output_dir / "validation_meta.env").write_text(
        "\n".join(
            [
                f"CUSTOM_GOLDEN={1 if custom_golden is not None else 0}",
                f"CUSTOM_COMPARE={1 if custom_compare is not None else 0}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Let the runner know which bins are outputs (for sim->golden copying).
    (output_dir / "outputs.txt").write_text(
        "\n".join([p["name"] for p in output_ptrs]) + ("\n" if output_ptrs else ""),
        encoding="utf-8",
    )

    run_sh = (templates_root / "run_sh_template.sh").read_text(encoding="utf-8")
    run_sh = run_sh.replace("@EXECUTABLE@", testcase)
    run_sh = run_sh.replace("@RUN_MODE@", run_mode)
    run_sh = run_sh.replace("@SOC_VERSION@", soc_version)
    run_path = output_dir / "run.sh"
    run_path.write_text(run_sh, encoding="utf-8")
    run_path.chmod(0o755)


def main():
    parser = argparse.ArgumentParser(description="Generate NPU validation testcase from PTOAS kernel.")
    parser.add_argument("--input", required=True, help="Input PTOAS .cpp file")
    parser.add_argument("--testcase", default=None, help="Testcase name (default: derived from input filename)")
    parser.add_argument("--output-root", default=None, help="Output testcases root directory")
    parser.add_argument("--run-mode", default="npu", choices=["sim", "npu"], help="Run mode for run.sh")
    parser.add_argument("--soc-version", default="Ascend910", help="SOC version for run.sh")
    parser.add_argument(
        "--aicore-arch",
        default=None,
        help="Override AICore arch passed to bisheng (e.g. dav-c220-vec|dav-c220-cube|dav-c310-vec|dav-c310-cube)",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root else None
    testcase = args.testcase or _derive_testcase_name(Path(args.input))
    generate_testcase(
        Path(args.input),
        output_root,
        testcase,
        args.run_mode,
        args.soc_version,
        aicore_arch=args.aicore_arch,
    )


if __name__ == "__main__":
    main()
