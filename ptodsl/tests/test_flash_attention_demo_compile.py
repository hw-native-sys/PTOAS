#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
from ptodsl._bootstrap import make_context
from mlir.ir import Module


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_parse_roundtrip_and_verify(text: str, label: str) -> None:
    with make_context() as ctx:
        parsed = Module.parse(text, ctx)
        parsed.operation.verify()
        roundtrip_text = str(parsed)
    expect(
        roundtrip_text == text,
        f"{label} should survive Module.parse(...) round-trip without textual drift",
    )


def load_flash_attention_demo():
    demo_candidates = [
        REPO_ROOT / "ptodsl" / "examples" / "flash_attention_sketch.py",
        REPO_ROOT / "ptodsl" / "demos" / "flash_attention_sketch.py",
    ]
    for demo_path in demo_candidates:
        if demo_path.is_file():
            break
    else:
        raise AssertionError(
            "canonical flash attention demo is missing: "
            + ", ".join(str(path) for path in demo_candidates)
        )

    spec = spec_from_file_location("ptodsl_flash_attention_demo", demo_path)
    expect(spec is not None and spec.loader is not None, f"unable to create import spec for {demo_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    demo = load_flash_attention_demo()

    expect(hasattr(demo, "emit_flash_attention_mlir"), "flash attention demo should export emit_flash_attention_mlir(...)")
    expect(hasattr(demo, "flash_attention_kernel"), "flash attention demo should export flash_attention_kernel")

    wrapper_text = demo.emit_flash_attention_mlir(head_dim=128, causal=False, block_q=128, block_kv=128)
    expect_parse_roundtrip_and_verify(wrapper_text, "flash attention wrapper-emitted MLIR")
    expect("func.func @flash_attention_kernel" in wrapper_text, "wrapper compile should emit the flash_attention_kernel entry")
    expect(wrapper_text.count("module") >= 2, "wrapper compile should emit an outer container plus child modules")
    expect(
        'pto.backend = "vpto"' in wrapper_text
        and 'pto.target_arch = "a5"' in wrapper_text
        and 'pto.kernel_kind = #pto.kernel_kind<vector>' in wrapper_text,
        "flash attention wrapper compile should encode the VPTO backend directly on the child module",
    )
    expect("func.func @materialize_tile_bounds" in wrapper_text, "wrapper compile should emit the SIMT helper function")
    expect("pto.simt_launch" in wrapper_text, "wrapper compile should materialize SIMT launch ops")
    expect("pto.barrier <PIPE_ALL>" in wrapper_text, "demo phase boundaries should lower to pipe_barrier(Pipe.ALL)")

    compiled = demo.flash_attention_kernel.compile(
        BLOCK_Q=64,
        BLOCK_KV=128,
        HEAD_DIM=128,
        CAUSAL=True,
    )
    compiled.verify()

    expect(
        compiled.constexpr_bindings == {
            "BLOCK_Q": 64,
            "BLOCK_KV": 128,
            "HEAD_DIM": 128,
            "CAUSAL": True,
            "NUM_STAGES": 2,
        },
        f"unexpected constexpr bindings: {compiled.constexpr_bindings!r}",
    )

    specialized_text = compiled.mlir_text()
    expect_parse_roundtrip_and_verify(specialized_text, "flash attention specialized MLIR")
    expect("func.func @flash_attention_kernel" in specialized_text, "direct compile should emit the flash_attention_kernel entry")
    expect(specialized_text.count("module") >= 2, "direct compile should keep the backend-partitioned container shape")
    expect(
        'pto.backend = "vpto"' in specialized_text
        and 'pto.target_arch = "a5"' in specialized_text
        and 'pto.kernel_kind = #pto.kernel_kind<vector>' in specialized_text,
        "direct compile should encode the VPTO backend directly on the child module",
    )
    expect("!pto.tile_buf<mat, 64x128xf32" in specialized_text, "BLOCK_Q=64 specialization should change the physical Q tile shape")
    expect("pto.simt_launch @materialize_tile_bounds" in specialized_text, "direct compile should route SIMT helpers through launch ops")

    cached = demo.flash_attention_kernel.cached_specializations()
    expect(len(cached) >= 2, "wrapper compile plus explicit compile should populate at least two cached specializations")
    print("ptodsl_flash_attention_demo_compile: PASS")


if __name__ == "__main__":
    main()
