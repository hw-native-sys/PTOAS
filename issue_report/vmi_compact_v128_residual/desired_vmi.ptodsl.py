"""Desired VMI for a compact V<128 x bf16> leftover strip.

Design §1.1: K_raw = 128 * 16 / 2048 = 1; one physical vreg, all 128
bf16 lanes valid. create_mask(128) is legal. Host-pad to 256 is not a port.

ASC tiles hidden with block_k aligned to 128 (H=128 one tile, H=384 three).
PTOAS currently reports VMI-RESIDUAL-OP when this strip is used at H=384.
"""
from ptodsl import pto


@pto.jit(name="compact_v128_cast", kernel_kind="vector", target="a5", mode="explicit")
def compact_v128_cast(
    out: pto.ptr(pto.f8e4m3, "ub"),
    x: pto.ptr(pto.bf16, "ub"),
):
    with pto.vecscope():
        mask = pto.vmi.create_mask(128, size=128)
        row = pto.vmi.vload(x, 0, size=128)
        abs_x = pto.vmi.vand(row, pto.vmi.vbrc(pto.ui16(32767), size=128), mask)
        amax = pto.vmi.vcmax(abs_x, mask)
        # Remainder tile is still V<128 x T>, not a residual opcode.
        pto.vmi.vstore(row, pto.castptr(out, pto.ptr(pto.bf16, "ub")), 0, mask)
        _ = amax
