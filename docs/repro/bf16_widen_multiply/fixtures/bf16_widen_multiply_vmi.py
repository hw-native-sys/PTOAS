"""Standalone PTODSL VMI implementation of BF16 widen and FP32 multiply."""

from ptodsl import pto

ELEMENTS_PER_CORE = 2048
UB_BYTES = ELEMENTS_PER_CORE * 8


@pto.jit(name="bf16_widen_multiply_vmi", kernel_kind="vector", target="a5", mode="explicit")
def bf16_widen_multiply_vmi(a: pto.ptr(pto.bf16, "gm"), b: pto.ptr(pto.bf16, "gm"), out: pto.ptr(pto.f32, "gm")):
    raw = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
    a_ub = pto.castptr(raw, pto.ptr(pto.bf16, "ub"))
    b_ub = pto.castptr(pto.addptr(raw, ELEMENTS_PER_CORE * 2), pto.ptr(pto.bf16, "ub"))
    out_ub = pto.castptr(pto.addptr(raw, ELEMENTS_PER_CORE * 4), pto.ptr(pto.f32, "ub"))
    offset = pto.get_block_idx() * ELEMENTS_PER_CORE
    pto.mte_gm_ub(pto.addptr(a, offset), a_ub, 0, ELEMENTS_PER_CORE * 2, nburst=(1, ELEMENTS_PER_CORE * 2, ELEMENTS_PER_CORE * 2))
    pto.mte_gm_ub(pto.addptr(b, offset), b_ub, 0, ELEMENTS_PER_CORE * 2, nburst=(1, ELEMENTS_PER_CORE * 2, ELEMENTS_PER_CORE * 2))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)
    mask = pto.vmi.create_mask(64, size=64)
    for lane in range(0, ELEMENTS_PER_CORE, 64):
        a_f32 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(a_ub, lane), 0, size=64), to_dtype=pto.f32)
        b_f32 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(b_ub, lane), 0, size=64), to_dtype=pto.f32)
        pto.vmi.vstore(pto.vmi.vmul(a_f32, b_f32, mask), pto.addptr(out_ub, lane), 0, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_ub_gm(out_ub, pto.addptr(out, offset), ELEMENTS_PER_CORE * 4, nburst=(1, ELEMENTS_PER_CORE * 4, ELEMENTS_PER_CORE * 4))
