"""Desired VMI for unpacked float scale load/store (row and TMA-col).

ASC Nightly L1 stores/loads V<L x f32> scales without packing. TMA-col is
a compiler dist-mode / address map, not a host permute of row-major SF.

Production VMI asserts packed-only on these adapters and never emits IR.
"""
from ptodsl import pto


@pto.jit(name="unpacked_float_sf_move", kernel_kind="vector", target="a5", mode="explicit")
def unpacked_float_sf_move(
    sf_row: pto.ptr(pto.f32, "ub"),
    sf_tma: pto.ptr(pto.f32, "ub"),
    scale: pto.ptr(pto.f32, "ub"),
):
    with pto.vecscope():
        mask = pto.vmi.create_mask(64, size=64)
        s = pto.vmi.vload(scale, 0, size=64)
        # Row-major unpacked store: logical contiguous V<64 x f32>.
        pto.vmi.vstore(s, sf_row, 0, mask)
        # TMA-col unpacked store: same logical vector; pto.as holds dist.
        pto.vmi.vstore(s, sf_tma, 0, mask)
        # Rescale input: load unpacked row-major in-SF (per_channel L1).
        _in = pto.vmi.vload(sf_row, 0, size=64)
        _ = _in
