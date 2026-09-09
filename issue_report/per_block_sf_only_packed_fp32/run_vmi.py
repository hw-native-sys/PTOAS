#!/usr/bin/env python3
"""Reproduce the leftover at the production VMI entry."""
from __future__ import annotations

import argparse
import os
import sys

KIND = "block_sf_only_fp32"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--device", default=os.environ.get("NPU_TEST_DEVICE", "npu:0"))
    args = ap.parse_args()
    print(f"LEFTOVER_KIND={KIND} compile_only={args.compile_only} device={args.device}")
    try:
        import torch
        from tile_kernels.quant.common import get_cast_input_config_impl, get_cast_output_config
        if KIND == "h128":
            from tile_kernels_vmi.quant.per_token_cast_vmi import get_per_token_cast_kernel_vmi
            kn = get_per_token_cast_kernel_vmi(
                512, 128, get_cast_input_config_impl(torch.bfloat16),
                get_cast_output_config("e4m3", (1, 32), True, True, True),
            )
        elif KIND == "tma_unpacked":
            from tile_kernels_vmi.quant.per_token_cast_vmi import get_per_token_cast_kernel_vmi
            kn = get_per_token_cast_kernel_vmi(
                512, 3072, get_cast_input_config_impl(torch.bfloat16),
                get_cast_output_config("e4m3", (1, 32), True, True, False),
            )
        elif KIND == "fp4_unpacked":
            from tile_kernels_vmi.quant.per_token_cast_vmi import get_per_token_cast_kernel_vmi
            kn = get_per_token_cast_kernel_vmi(
                512, 3072, get_cast_input_config_impl(torch.bfloat16),
                get_cast_output_config("e2m1", (1, 32), False, True, False),
            )
        elif KIND == "sf_only_packed":
            from tile_kernels_vmi.quant.per_token_cast_vmi import get_per_token_cast_kernel_vmi
            kn = get_per_token_cast_kernel_vmi(
                512, 3072, get_cast_input_config_impl(torch.bfloat16),
                get_cast_output_config("e4m3", (1, 32), False, True, True),
                sf_only=True,
            )
        elif KIND == "rescale_row":
            from tile_kernels_vmi.quant.per_token_cast_vmi import get_per_token_cast_kernel_vmi
            kn = get_per_token_cast_kernel_vmi(
                512, 3072,
                get_cast_input_config_impl(torch.float8_e4m3fn, (32, 32), False, True),
                get_cast_output_config("e4m3", (1, 32), False, True, True),
            )
        elif KIND == "channel_unpacked_in":
            from tile_kernels_vmi.quant.per_channel_cast_vmi import get_per_channel_cast_kernel_vmi
            kn = get_per_channel_cast_kernel_vmi(
                512, 3072,
                get_cast_input_config_impl(torch.float8_e4m3fn, (1, 32), False, False),
                get_cast_output_config("e4m3", (32, 1), False, True, True),
            )
        elif KIND == "block_sf_only_fp32":
            if args.compile_only:
                print("SKIP runtime compare in compile-only; see recorded.log (1024-byte SF mismatch)")
                return 0
            torch.npu.set_device(int(args.device.split(":")[-1]))
            from tile_kernels.quant.common import alloc_scaling_factors
            from tile_kernels.quant.per_block_cast_asc import get_per_block_cast_kernel_asc
            from tile_kernels_vmi.quant.per_block_cast_vmi import get_per_block_cast_kernel_vmi
            in_cfg = get_cast_input_config_impl(torch.float32)
            out_cfg = get_cast_output_config("e4m3", (32, 32), False, True, True)
            x = torch.randn(512, 2048, dtype=torch.float32, device=args.device)
            a = get_per_block_cast_kernel_asc(hidden=2048, in_config=in_cfg, out_config=out_cfg, sf_only=True, cast_only=False, num_vec_cores=72)
            v = get_per_block_cast_kernel_vmi(512, 2048, in_cfg, out_cfg, sf_only=True)
            sf_a = alloc_scaling_factors((512, 2048), out_cfg, device=args.device)
            sf_v = alloc_scaling_factors((512, 2048), out_cfg, device=args.device)
            a(x, x.new_empty(512, 2048, dtype=out_cfg.torch_dtype), sf_a)
            v(x, sf_v)
            torch.npu.synchronize()
            mismatch = int((sf_a.view(torch.uint8) != sf_v.view(torch.uint8)).sum().item())
            print(f"SF_MISMATCH_BYTES={mismatch}")
            return 0 if mismatch == 0 else 2
        else:
            raise SystemExit(f"unknown KIND={KIND}")
        print(f"UNEXPECTED_OK {type(kn).__name__}")
        return 0
    except Exception as exc:
        print(f"REPRO {type(exc).__name__}: {exc}")
        return 0 if args.compile_only else 2


if __name__ == "__main__":
    raise SystemExit(main())
