"""PTO VMI port of the store_pad8 reduce epilogue (pad-8 workaround).

ASC/CCE: vcadd + scalar vsts ONEPT_B32 per accumulator (see
../cce/csrc/store_pad8_vf_sim_kernel.cpp). Product VMI workaround: broadcast
the reduced value to 8 lanes (vbrc size=8) and store under an 8-wide mask.
See store_mask1_vmi.py for the compact 1-lane store form.

Output is the full padded [N_ACC, 8] buffer (lane 0 of each row holds the
reduced value). Host-side compaction is out of scope here.
"""

from ptodsl import pto

_VL = 64
_REDUCE_PAD = 8

_UB = {
    "acc": 0x0000,
    "pad": 0x2000,
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
        pad_gm: pto.ptr(pto.f32, "gm"),
    ):
        acc = pto.castptr(pto.const(_UB["acc"], dtype=pto.ui64), pto.ptr(pto.f32, "ub"))
        pad = pto.castptr(pto.const(_UB["pad"], dtype=pto.ui64), pto.ptr(pto.f32, "ub"))

        nbytes_acc = n_acc * _VL * 4
        nbytes_pad = n_acc * _REDUCE_PAD * 4
        pto.mte_gm_ub(acc_gm, acc, 0, nbytes_acc, nburst=(1, nbytes_acc, nbytes_acc))
        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
        pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)

        mask64 = pto.vmi.create_mask(_VL, size=_VL)
        mask8 = pto.vmi.create_mask(_REDUCE_PAD, size=_REDUCE_PAD)

        for i in pto.static_range(n_acc):
            acc_v = pto.vmi.vload(acc, i * _VL, size=_VL)
            red = pto.vmi.vcadd(acc_v, mask64, reassoc=True)
            pto.vmi.vstore(pto.vmi.vbrc(red, size=_REDUCE_PAD), pad, i * _REDUCE_PAD, mask8)

        pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=1)
        pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=1)
        pto.mte_ub_gm(pad, pad_gm, nbytes_pad, nburst=(1, nbytes_pad, nbytes_pad))

    return kernel


store_pad8_vmi_large = _build(20, "store_pad8_vmi_large")
store_pad8_vmi_small = _build(4, "store_pad8_vmi_small")
