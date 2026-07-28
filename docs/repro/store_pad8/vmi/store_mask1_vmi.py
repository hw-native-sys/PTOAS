"""PTO VMI reduce epilogue: vcadd + 1-lane masked store (compact [N_ACC]).

Same math as the pad-8 workaround and CCE ONEPT: sum each of N_ACC length-64
f32 rows to one scalar. Unlike pad-8, this does not vbrc to 8 lanes — it stores
under a 1-lane mask into a compact [N_ACC] buffer (CCE GM layout).

On vmi-v0.1.3 this lowers (vdup/vadd + masked vsts) but is not CCE ONEPT /
1PT_B32. Use it to separate kernel formulation from compiler lowering cost.
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
        mask1 = pto.vmi.create_mask(1, size=1)

        for i in pto.static_range(n_acc):
            acc_v = pto.vmi.vload(acc, i * _VL, size=_VL)
            red = pto.vmi.vcadd(acc_v, mask64, reassoc=True)
            pto.vmi.vstore(red, out, i, mask1)

        pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=1)
        pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=1)
        pto.mte_ub_gm(out, out_gm, nbytes_out, nburst=(1, nbytes_out, nbytes_out))

    return kernel


store_mask1_vmi_large = _build(20, "store_mask1_vmi_large")
store_mask1_vmi_small = _build(4, "store_mask1_vmi_small")
