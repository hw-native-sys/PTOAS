"""PTO VMI reduce epilogue: vcadd(group=1) + vstore(group=1) compact [N_ACC].

Hinted ONEPT / 1PT path: reduce to a 1-lane score, then unit-stride group
store (no mask). On a PTOAS build that lowers compact reduction stores to
point stores, this should become vsts ... 1PT_B32 — unlike mask1 (masked
1-lane store) or pad-8 (vbrc×8).
"""

from ptodsl import pto

_VL = 64

_UB = {
    "acc": 0x0000,
    "out": 0x2000,
}


def _build(n_acc: int, name: str):
    @pto.jit(
        name=name,
        target="a5",
        backend="vpto",
        mode="explicit",
        kernel_kind="vector",
        insert_sync=False,
    )
    def kernel(
        acc_gm: pto.ptr(pto.f32, "gm"),
        out_gm: pto.ptr(pto.f32, "gm"),
    ):
        acc = pto.castptr(pto.const(_UB["acc"], dtype=pto.ui64), pto.ptr(pto.f32, "ub"))
        out = pto.castptr(pto.const(_UB["out"], dtype=pto.ui64), pto.ptr(pto.f32, "ub"))

        nbytes_acc = n_acc * _VL * 4
        nbytes_out = n_acc * 4
        pto.mte_gm_ub(acc_gm, acc, 0, nbytes_acc, nburst=(1, nbytes_acc, nbytes_acc))
        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
        pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)

        mask64 = pto.vmi.create_mask(_VL, size=_VL)

        for i in pto.static_range(n_acc):
            acc_v = pto.vmi.vload(acc, i * _VL, size=_VL)
            red = pto.vmi.vcadd(acc_v, mask64, group=1, reassoc=True)
            pto.vmi.vstore(red, out, i, group=1, stride=1)

        pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=1)
        pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=1)
        pto.mte_ub_gm(out, out_gm, nbytes_out, nburst=(1, nbytes_out, nbytes_out))

    return kernel


store_group1_vmi_large = _build(20, "store_group1_vmi_large")
store_group1_vmi_small = _build(4, "store_group1_vmi_small")
