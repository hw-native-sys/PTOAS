# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ptoas.mlir.dialects.pto as pto


@pto.vkernel(target="a5", name="template_abs_kernel")
def template_abs_kernel(src: pto.Tile, dst: pto.Tile):
    total = src.shape[0] * src.shape[1]
    step = 256 // src.ub_ptr.elem_bytes

    with pto.strict_vecscope(src.ub_ptr, dst.ub_ptr, 0, total, step, total) as (
        vin,
        vout,
        lb,
        ub,
        vec_step,
        remaining,
    ):
        for offset in range(lb, ub, vec_step):
            mask, remaining = pto.plt_b32(remaining)
            vec_in = pto.vlds(vin, offset)
            vec_out = pto.vabs(vec_in, mask)
            pto.vsts(vec_out, vout, offset, mask)


template_abs_kernel_f32 = template_abs_kernel.jit(
    src=pto.Tile(
        ub_ptr=pto.ptr(pto.f32, "ub"),
        shape=pto.const([32, 32]),
    ),
    dst=pto.Tile(
        ub_ptr=pto.ptr(pto.f32, "ub"),
        shape=pto.const([32, 32]),
    ),
)

template_abs_kernel_f16 = template_abs_kernel.jit(
    src=pto.Tile(
        ub_ptr=pto.ptr(pto.f16, "ub"),
        shape=pto.const([32, 32]),
    ),
    dst=pto.Tile(
        ub_ptr=pto.ptr(pto.f16, "ub"),
        shape=pto.const([32, 32]),
    ),
)


if __name__ == "__main__":
    print(template_abs_kernel_f32.mlir_text(), end="")
