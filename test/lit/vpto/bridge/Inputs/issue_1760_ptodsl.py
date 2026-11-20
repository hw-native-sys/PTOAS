#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Issue #1760 reduced to two VPTO children that each generate a pipe wrapper.

Only initialization is needed to expose the duplicate symbol at link time.
This is a compiler input, not a runnable data-transfer or numerical test.
Running this script prints frontend IR without invoking CANN or a simulator.
"""

from ptodsl import pto


@pto.jit(name="vec_child", kernel_kind="vector", target="a5", backend="vpto", entry=False)
def vec_child():
    fifo = pto.import_reserved_buffer("fifo", peer_func="cube_child")
    pipe = pto.pipe.v2c(slot_size=1024, consumer_buf=fifo, id=0)
    pipe.init_simd()


@pto.jit(name="cube_child", kernel_kind="cube", target="a5", backend="vpto", entry=False)
def cube_child():
    fifo = pto.reserve_buffer("fifo", size=8192, location="mat")
    pipe = pto.pipe.v2c(slot_size=1024, consumer_buf=fifo, id=0)
    pipe.init_cube()


@pto.jit(name="cv_entry", target="a5", backend="emitc")
def cv_entry():
    vec_child()
    cube_child()


if __name__ == "__main__":
    print(cv_entry.compile().mlir_text())
