"""Desired VMI for last-tile fp8 zero (ceildiv tail, no host pad).

ASC fused rescale at M=8001 uses in-kernel ceildiv and zeros the leftover
e4m3/e2m1 lanes. Design vbrc tables list i8-i32 / f16 / bf16 / f32 —
fp8_e4m3 / fp8_e2m1 is the hole. T.clear on e4m3 UB also fails
(Bad bit-width).

Legal: vbrc of a zero fp8 scalar into V<L x fp8_e4m3>, or a predicated
vstore of zeros. Host-pad M to a multiple of 16 is not a port.
"""
from ptodsl import pto


@pto.jit(name="fp8_tail_zero", kernel_kind="vector", target="a5", mode="explicit")
def fp8_tail_zero(out: pto.ptr(pto.f8e4m3, "ub")):
    with pto.vecscope():
        mask = pto.vmi.create_mask(64, size=64)
        # Spec gap: vbrc must accept fp8_e4m3 the same way it accepts i8.
        zeros = pto.vmi.vbrc(pto.f8e4m3(0), size=64)
        pto.vmi.vstore(zeros, out, 0, mask)
