# Golden for vmi_new/vmula-bf16-vl8-unaligned-stateful-load.
#
# Same input samples as vmi_new/vmula-bf16-vl8 (the framework case
# vmi_vmula_bf16_vl8_g1_a8_n_nomask_t01): the unaligned-stateful case exists
# to prove the vldas+vldus lowering (#1374) reads the same bits the aligned
# path would, so the numerical golden is identical.
#
# Each GM operand is a 32-byte staging image of its UB slot: the 16-byte
# payload is written at the element offset the kernel later loads from
# (acc at lane 4, lhs at lane 1, rhs at lane 6), so the misaligned effective
# addresses (1032/2050/3084) land exactly on the payload.
#
# Semantics: vmula computes acc + lhs*rhs in a wide domain and rounds once to
# bf16 (RNE) at the output -- it does NOT round the product to bf16 first.

import argparse
from pathlib import Path

import numpy as np

VL = 8
SLOT_LANES = 16  # 32-byte staging image, in bf16 lanes

ACC_BITS = np.array(
    [0xC087, 0x3F4F, 0x3F3D, 0xC00D, 0x3FE5, 0xC027, 0xC047, 0x401E],
    dtype=np.uint16,
)
LHS_BITS = np.array(
    [0xC024, 0xBFF9, 0x4092, 0xBF79, 0xC080, 0x3FDE, 0xBF95, 0xC006],
    dtype=np.uint16,
)
RHS_BITS = np.array(
    [0x4095, 0xC022, 0xC08A, 0xC07E, 0x409B, 0xBE68, 0xC057, 0xBFCB],
    dtype=np.uint16,
)

assert ACC_BITS.shape == (VL,) and LHS_BITS.shape == (VL,) and RHS_BITS.shape == (VL,), \
    f"input arrays must have {VL} lanes"


def bf16_bits_to_f32(bits):
    u = bits.astype(np.uint32) << np.uint32(16)
    return u.view(np.float32)


def to_bf16_bits(x):
    """f32 -> bfloat16 bit pattern (uint16), round-to-nearest-even."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    rounding_bias = np.uint32(0x7FFF) + lsb
    r = ((u + rounding_bias) >> np.uint32(16)).astype(np.uint16)
    r = np.where(np.isnan(x), np.uint16(0x7FC0), r)
    return r


def staging_image(payload_bits, lane_offset):
    """32-byte UB slot image with the payload at the given lane offset."""
    slot = np.zeros(SLOT_LANES, dtype=np.uint16)
    slot[lane_offset:lane_offset + VL] = payload_bits
    return slot


def generate(output_dir: Path) -> None:
    acc = bf16_bits_to_f32(ACC_BITS)
    lhs = bf16_bits_to_f32(LHS_BITS)
    rhs = bf16_bits_to_f32(RHS_BITS)
    # Wide-domain accumulate: compute in f64, round once to bf16 at output.
    wide = acc.astype(np.float64) + lhs.astype(np.float64) * rhs.astype(np.float64)
    golden = to_bf16_bits(wide.astype(np.float32))

    # Sentinel so a completely-missed store shows up as garbage, not coincidence.
    dst = np.full(VL, 0xAAAA, dtype=np.uint16)

    output_dir.mkdir(parents=True, exist_ok=True)
    staging_image(ACC_BITS, 4).tofile(output_dir / "v1.bin")
    staging_image(LHS_BITS, 1).tofile(output_dir / "v2.bin")
    staging_image(RHS_BITS, 6).tofile(output_dir / "v3.bin")
    dst.tofile(output_dir / "v4.bin")
    golden.tofile(output_dir / "golden_v4.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
