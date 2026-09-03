# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ptodsl import pto, scalar


@pto.jit(
    name="main_kernel",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    ast_rewrite=False,
)
def main_kernel(
    RSTD: pto.ptr(pto.f32, "gm"),
    W: pto.ptr(pto.f32, "gm"),
    X: pto.ptr(pto.f32, "gm"),
    Y: pto.ptr(pto.f32, "gm"),
    eps: pto.f32,
):
  bx = pto.get_block_idx()
  buf_dyn_shmem = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.i8, "ub"))

  pto.set_flag("V", "MTE2", event_id=0)
  pto.set_flag("V", "MTE2", event_id=1)
  pto.set_flag("MTE3", "V", event_id=0)
  pto.set_flag("MTE3", "V", event_id=1)

  pto.mte_gm_ub(
      W,
      pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")),
      0,
      20480,
      nburst=(1, 20480, 20480),
  )
  pto.set_flag("MTE2", "V", event_id=2)
  pto.wait_flag("MTE2", "V", event_id=2)

  # The physical fragment has 32 elements, while only 20 are valid for d=5120.
  w_frag = pto.alloc_buffer((32,), pto.f32)
  with pto.simt(256, 1, 1):
    simtvf_tx = pto.get_tid_x()
    with pto.for_(0, 10, step=1, unroll="full") as weight_i:
      scalar.store(
          scalar.load(
              pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")),
              (weight_i * 512) + (simtvf_tx * 2),
              contiguous=2,
          ),
          w_frag,
          weight_i * 2,
      )

  with pto.for_(0, 64, step=1) as t:
    pto.wait_flag("V", "MTE2", event_id=t & 1)
    pto.mte_gm_ub(
        pto.addptr(X, (t * 327680) + (bx * 5120)),
        pto.addptr(
            pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")),
            ((t & 1) * 8192) + 5120,
        ),
        0,
        20480,
        nburst=(1, 20480, 20480),
    )
    pto.set_flag("MTE2", "V", event_id=t & 1)
    pto.wait_flag("MTE3", "V", event_id=t & 1)
    pto.wait_flag("MTE2", "V", event_id=t & 1)

    with pto.simt(256, 1, 1):
      x_frag = pto.alloc_buffer((32,), pto.f32)
      sum_sq = pto.alloc_buffer((1,), pto.f32)
      simtvf_tx = pto.get_tid_x()

      for input_i in pto.static_range(0, 16):
        scalar.store(
            scalar.load(
                pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")),
                ((((t & 1) * 8192) + (input_i * 512)) + (simtvf_tx * 2))
                + 5120,
                contiguous=2,
            ),
            x_frag,
            input_i * 2,
        )

      scalar.store(float.fromhex("0x0p+0"), sum_sq, 0)
      for reduce_i in pto.static_range(0, 20):
        scalar.store(
            scalar.load(sum_sq, 0)
            + (scalar.load(x_frag, reduce_i) * scalar.load(x_frag, reduce_i)),
            sum_sq,
            0,
        )

      scalar.store(
          pto.simt_allreduce_sum(
              scalar.load(sum_sq, 0),
              threads=256,
              scale=1,
              thread_offset=0,
              scratch=pto.addptr(
                  pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), 37904
              ),
          ),
          sum_sq,
          0,
      )
      var = (scalar.load(sum_sq, 0) / float.fromhex("0x1.4p+12")) + eps
      rstd_val = float.fromhex("0x1p+0") / pto.sqrt(var)
      scalar.store(
          rstd_val,
          pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")),
          ((t & 1) * 8) + 37888,
      )

      with pto.for_(0, 10, step=1, unroll="full") as output_i:
        scalar.store(
            (
                scalar.load(x_frag, output_i * 2, contiguous=2)
                * pto.Vec(pto.f32, 2, init=rstd_val)
            )
            * scalar.load(w_frag, output_i * 2, contiguous=2),
            pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")),
            ((((t & 1) * 8192) + (output_i * 512)) + (simtvf_tx * 2))
            + 21504,
        )

    pto.set_flag("V", "MTE3", event_id=t & 1)
    pto.set_flag("V", "MTE2", event_id=t & 1)
    pto.wait_flag("V", "MTE3", event_id=t & 1)
    pto.mte_ub_gm(
        pto.addptr(
            pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")),
            ((t & 1) * 8) + 37888,
        ),
        pto.addptr(RSTD, (t * 64) + bx),
        4,
        nburst=(1, 4, 4),
    )
    pto.mte_ub_gm(
        pto.addptr(
            pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")),
            ((t & 1) * 8192) + 21504,
        ),
        pto.addptr(Y, (t * 327680) + (bx * 5120)),
        20480,
        nburst=(1, 20480, 20480),
    )
    pto.set_flag("MTE3", "V", event_id=t & 1)

  pto.wait_flag("V", "MTE2", event_id=0)
  pto.wait_flag("V", "MTE2", event_id=1)
  pto.wait_flag("MTE3", "V", event_id=0)
  pto.wait_flag("MTE3", "V", event_id=1)
