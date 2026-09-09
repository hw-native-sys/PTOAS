"""Desired VMI for amax -> UE8M0 pack (scale-only, no payload store).

bf16 per-block sf_only+packed already matches ASC bitwise. fp32 of the
same path mismatches 1024 SF bytes. per-token sf_only+packed never emits
IR (assert: sf_only requires unpacked).

One issue: pack scales from an fp32 (or bf16-widened) amax into UE8M0
and vstore packed bytes. Payload store is optional.
"""
from ptodsl import pto


@pto.jit(name="ue8m0_pack_from_fp32_amax", kernel_kind="vector", target="a5", mode="explicit")
def ue8m0_pack_from_fp32_amax(
    out_sf: pto.ptr(pto.ui8, "ub"),
    x: pto.ptr(pto.f32, "ub"),
):
    with pto.vecscope():
        mask = pto.vmi.create_mask(64, size=64)
        row = pto.vmi.vload(x, 0, size=64)
        amax = pto.vmi.vcmax(row, mask)
        # Exponent-only UE8M0: (bits >> 23) + rounding, pack to u8.
        bits = pto.vmi.vinterpret_cast(amax, to_dtype=pto.ui32)
        exp = pto.vmi.vshrs(bits, 23, mask)
        packed = pto.vmi.vcvt(exp, to_dtype=pto.ui8)
        pto.vmi.vstore(packed, out_sf, 0, pto.vmi.create_mask(8, size=8))
