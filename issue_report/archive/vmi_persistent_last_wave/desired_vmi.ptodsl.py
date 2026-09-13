"""Desired VMI for one Persistent remainder wave (per_channel TMA-in).

The 1xT TMA-col *input* unpack is the same legal family as issue A.
This file is the *last incomplete software-pipeline wave* only: one
remainder group, not the full 7k-hidden kernel.

Recorded fused mismatch is confined to remainder / periodic m-tile waves
(512x7168 m-tile 7 x k-tiles 20-27; 8064x2048 period 27). Compose is
issue A (cast_back TMA npt=1), not this file.
"""
from ptodsl import pto


@pto.jit(name="persistent_last_wave", kernel_kind="vector", target="a5", mode="explicit")
def persistent_last_wave(
    out: pto.ptr(pto.bf16, "ub"),
    x: pto.ptr(pto.f8e4m3, "ub"),
    packed_sf: pto.ptr(pto.ui8, "ub"),
):
    with pto.vecscope():
        one = pto.vmi.create_mask(1, size=1)
        mask = pto.vmi.create_mask(128, size=128)
        raw = pto.vmi.vload(packed_sf, 0, size=1)
        exp = pto.vmi.vshls(pto.vmi.vcvt(raw, to_dtype=pto.ui16), 7, one)
        scale = pto.vmi.vbrc(pto.vmi.vinterpret_cast(exp, to_dtype=pto.bf16), size=128)
        payload = pto.vmi.vload(x, 0, size=128)
        # Last wave: same VF as a full wave. pto.as must not drop/skew
        # the remainder group's 1xT scales or payload.
        pto.vmi.vstore(pto.vmi.vmul(payload, scale, mask), out, 0, mask)
