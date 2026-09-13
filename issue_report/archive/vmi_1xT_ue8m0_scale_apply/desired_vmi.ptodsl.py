"""Desired VMI for UE8M0 1xT extract + scale apply.

Legal compact extract: vload size=1 (one UE8M0 byte) -> shift into a
bf16/f32 exponent -> vbrc across the payload group -> vcvt + vmul.

No PART / ONEPT / PK impersonation. Layout (pack / interleave / dist) is
pto.as's job. See PTO-vmi-design.en.md §1.1 compact vregs and §0.2 Category A.

PTOAS must lower this to the ASC pattern in asc_pattern.mi
(vlds_brc_elem + vshl + vcvt + vmul).
"""
from ptodsl import pto


@pto.jit(name="ue8m0_1xT_scale_apply", kernel_kind="vector", target="a5", mode="explicit")
def ue8m0_1xT_scale_apply(
    out: pto.ptr(pto.f32, "ub"),
    x: pto.ptr(pto.f8e4m3, "ub"),
    packed_sf: pto.ptr(pto.ui8, "ub"),
):
    with pto.vecscope():
        one = pto.vmi.create_mask(1, size=1)
        mask = pto.vmi.create_mask(64, size=64)
        # Compact 1xT extract: one UE8M0 byte, not a 32-lane PK impersonation.
        raw = pto.vmi.vload(packed_sf, 0, size=1)
        u16 = pto.vmi.vcvt(raw, to_dtype=pto.ui16)
        exp = pto.vmi.vshls(u16, 7, one)
        scale_bf = pto.vmi.vinterpret_cast(exp, to_dtype=pto.bf16)
        scale = pto.vmi.vbrc(scale_bf, size=64)
        payload = pto.vmi.vload(x, 0, size=64)
        value = pto.vmi.vcvt(payload, to_dtype=pto.f32)
        scale_f32 = pto.vmi.vcvt(scale, to_dtype=pto.f32)
        pto.vmi.vstore(pto.vmi.vmul(value, scale_f32, mask), out, 0, mask)
