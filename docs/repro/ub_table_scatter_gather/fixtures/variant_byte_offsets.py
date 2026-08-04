# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Ruled-out variant: scale offsets by 4 (byte offsets instead of element offsets).

Spec says per-lane element offset. Byte scaling is also wrong (still stale / wrong
values). Kept here so the hypothesis stays closed.
"""

from ptodsl import pto

_VL = 64
_POOL = 256


@pto.jit(
    name="ub_table_scatter_gather_byte_offsets",
    kernel_kind="vector",
    target="a5",
    backend="vpto",
    mode="explicit",
)
def ub_table_scatter_gather_byte_offsets(
    val_gm: pto.ptr(pto.f32, "gm"),
    out_gm: pto.ptr(pto.f32, "gm"),
):
    ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    pool = ub
    val_ub = pto.addptr(ub, _POOL)
    out_ub = pto.addptr(ub, _POOL + _VL)
    nbytes = _VL * 4

    pto.mte_gm_ub(val_gm, val_ub, 0, nbytes, nburst=(1, nbytes, nbytes))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(_VL, size=_VL)
    idx = pto.vmi.vci(pto.i32(0), size=_VL, order="ASC")
    idx4 = pto.vmi.vmuls(idx, 4, mask)
    off0 = pto.const(0, dtype=pto.index)

    neg_inf = pto.vmi.vbrc(pto.f32(float("-inf")), size=_VL)
    pto.vmi.vstore(neg_inf, pool, off0, mask)
    pto.pipe_barrier("V")
    val = pto.vmi.vload(val_ub, off0, size=_VL)
    pto.vmi.vscatter(val, pool, idx4, mask)
    pto.pipe_barrier("V")
    got = pto.vmi.vgather(pool, idx4, mask)
    pto.vmi.vstore(got, out_ub, off0, mask)

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_ub_gm(out_ub, out_gm, nbytes, nburst=(1, nbytes, nbytes))


if __name__ == "__main__":
    print(ub_table_scatter_gather_byte_offsets.compile().mlir_text()[:300])
