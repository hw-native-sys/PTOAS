#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# coding=utf-8

"""Runtime numeric regression for the ``pto.ttrans`` tile templates (issue #1604).

Four shapes cover all four b8/b16 ttrans templates on the VPTO path:

- i8  [32,32] tall: b8 colwise (vgather2 zero-extend + PK_B16 packed store)
- ui8 [32,32] tall: b8 colwise
- f16 [16,32] wide: b16 rowwise (vlds + 16-bit-offset vscatter)
- i8  [32,64] wide: b8 rowwise (UNPK_B8 load + even-byte vscatter)

The host side talks to libascendcl through ctypes directly so the case does
not depend on torch/torch_npu.
"""

import ctypes
import os
import zlib

import numpy as np

from ptodsl import pto


ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def _ttrans_body(src_ptr, dst_ptr, *, rows, cols, dtype):
    src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[cols, rows], strides=[rows, 1])
    src = pto.alloc_tile(shape=[rows, cols], dtype=dtype, valid_shape=[rows, cols], blayout="RowMajor")
    tmp = pto.alloc_tile(shape=[rows, cols], dtype=dtype, valid_shape=[rows, cols], blayout="RowMajor")
    dst = pto.alloc_tile(shape=[cols, rows], dtype=dtype, valid_shape=[cols, rows], blayout="RowMajor")
    pto.tile.load(src_view, src)
    pto.tile.transpose(src, tmp, dst)
    pto.tile.store(dst, dst_view)


def _make_ttrans_kernel(name, dtype, rows, cols):
    @pto.jit(name=name, target="a5")
    def _kernel(
        src_ptr: pto.ptr(dtype, "gm"),
        dst_ptr: pto.ptr(dtype, "gm"),
    ):
        _ttrans_body(src_ptr, dst_ptr, rows=rows, cols=cols, dtype=dtype)

    return _kernel


SHAPES = (
    ("ttrans_i8_tall_32x32", pto.i8, np.int8, 32, 32),
    ("ttrans_ui8_tall_32x32", pto.ui8, np.uint8, 32, 32),
    ("ttrans_f16_wide_16x32", pto.f16, np.float16, 16, 32),
    ("ttrans_i8_wide_32x64", pto.i8, np.int8, 32, 64),
)

KERNELS = {
    name: _make_ttrans_kernel(name, dtype, rows, cols)
    for name, dtype, _np_dtype, rows, cols in SHAPES
}


def _load_acl():
    # Load by soname so an msprof simulator run resolves the simulator-backed
    # libascendcl from its runtime environment; fall back to the toolkit copy
    # (set_env.sh puts lib64 on LD_LIBRARY_PATH for direct runs).
    try:
        lib = ctypes.CDLL("libascendcl.so")
    except OSError:
        home = os.environ["ASCEND_HOME_PATH"]
        lib = ctypes.CDLL(os.path.join(home, "lib64", "libascendcl.so"))
    lib.aclInit.argtypes = [ctypes.c_void_p]
    lib.aclrtSetDevice.argtypes = [ctypes.c_int32]
    lib.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
    lib.aclrtMalloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.aclrtMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
    lib.aclrtFree.argtypes = [ctypes.c_void_p]
    if lib.aclInit(None) != 0:
        raise RuntimeError("aclInit failed")
    if lib.aclrtSetDevice(0) != 0:
        raise RuntimeError("aclrtSetDevice(0) failed")
    stream = ctypes.c_void_p()
    if lib.aclrtCreateStream(ctypes.byref(stream)) != 0:
        raise RuntimeError("aclrtCreateStream failed")
    return lib, stream


def _make_source(name, np_dtype, rows, cols):
    rng = np.random.default_rng(zlib.crc32(name.encode("utf-8")))
    if np_dtype == np.float16:
        return rng.uniform(-4.0, 4.0, size=(rows, cols)).astype(np.float16)
    if np_dtype == np.uint8:
        return rng.integers(0, 256, size=(rows, cols)).astype(np.uint8)
    return rng.integers(-100, 101, size=(rows, cols)).astype(np.int8)


def main():
    lib, stream = _load_acl()
    failed = False
    for name, _dtype, np_dtype, rows, cols in SHAPES:
        src = _make_source(name, np_dtype, rows, cols)
        golden = np.ascontiguousarray(src.T)
        nbytes = src.nbytes

        dev_src = ctypes.c_void_p()
        dev_dst = ctypes.c_void_p()
        if lib.aclrtMalloc(ctypes.byref(dev_src), nbytes, ACL_MEM_MALLOC_HUGE_FIRST) != 0:
            raise RuntimeError(f"{name}: aclrtMalloc(src) failed")
        if lib.aclrtMalloc(ctypes.byref(dev_dst), nbytes, ACL_MEM_MALLOC_HUGE_FIRST) != 0:
            raise RuntimeError(f"{name}: aclrtMalloc(dst) failed")
        try:
            host_src = src.tobytes()
            if lib.aclrtMemcpy(dev_src, nbytes, host_src, nbytes, ACL_MEMCPY_HOST_TO_DEVICE) != 0:
                raise RuntimeError(f"{name}: H2D memcpy failed")

            compiled = KERNELS[name].compile()
            compiled[1, stream](dev_src, dev_dst)
            if lib.aclrtSynchronizeStream(stream) != 0:
                raise RuntimeError(f"{name}: stream sync failed")

            out = np.empty((cols, rows), dtype=np_dtype)
            if lib.aclrtMemcpy(out.ctypes.data, nbytes, dev_dst, nbytes, ACL_MEMCPY_DEVICE_TO_HOST) != 0:
                raise RuntimeError(f"{name}: D2H memcpy failed")
        finally:
            lib.aclrtFree(dev_src)
            lib.aclrtFree(dev_dst)

        mismatched = int((out != golden).sum())
        ok = mismatched == 0
        print(
            f"{'PASS' if ok else 'FAIL'} {name} shape={rows}x{cols} "
            f"mismatched={mismatched}/{out.size}"
        )
        if not ok:
            print(f"  golden row0: {golden[0, :8].tolist()}")
            print(f"  actual row0: {out[0, :8].tolist()}")
            failed = True

    if failed:
        return 1
    print("All ttrans cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
