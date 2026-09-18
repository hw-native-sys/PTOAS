# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""``pto.init_core`` – standard-library core execution-state initialization.

This module is loaded lazily by the stdlib export catalog; kernels only see
the public ``pto.init_core()`` surface.  It emits the canonical A5 VPTO
core-state initialization sequence in source order, composed from the
standard ``pto.*`` interfaces:

1. ``pto.get_ctrl`` → standard scalar ``&``/``|`` operators (emitted as
   ``pto.and``/``pto.or``) → ``pto.set_ctrl`` – preserve the running
   CTRL bits selected by ``_CTRL_KEEP_MASK`` and force the bits selected by
   ``_CTRL_PRESET_BITS``.
2. Vector kernels: ``pto.set_loop_size_ubtoout(1, 1)`` /
   ``pto.set_loop_size_outtoub(1, 1)`` to restore default DMA loop sizes.
   Explicit ``kernel_kind="cube"`` kernels receive ``pto.set_mov_pad_val(0)``
   instead, mirroring the reference ``__DAV_CUBE__`` / ``__DAV_VEC__``
   initialization split.
3. ``pto.set_store_atomic_cfg(0b00100100)`` – restore the default scalar
   store-atomic configuration.

Every emitted operation keeps its IR side-effect semantics, so later
semantic, layout, scheduling, and code-generation passes cannot remove or
reorder the initialization incorrectly.
"""

from .. import pto
from .._ops_common import (
    _require_backend,
    _require_explicit_mode,
    _require_target_arch,
)

_CTRL_KEEP_MASK = 0x1000000000000
_CTRL_PRESET_BITS = 0x1000000000000008
_DEFAULT_ST_ATOMIC_CFG = 0b00100100
_DEFAULT_DMA_LOOP_COUNT = 1
_PAD_VALUE = 0


def _authored_kernel_kind():
    from .._tracing.active import current_session

    session = current_session()
    if session is None:
        return None
    module_spec = getattr(session, "current_function_module_spec", None)
    if module_spec is None:
        module_spec = session.module_spec
    return getattr(module_spec, "kernel_kind", None)


@pto.func(returns=None, ast_rewrite=False)
def init_core():
    """Initialize A5 VPTO core execution state at the call site."""
    _require_explicit_mode("pto.init_core()")
    _require_target_arch("pto.init_core()", {"a5"})
    _require_backend("pto.init_core()", {"vpto"})

    ctrl = pto.get_ctrl()
    pto.set_ctrl((ctrl & _CTRL_KEEP_MASK) | _CTRL_PRESET_BITS)

    if _authored_kernel_kind() == "cube":
        pto.set_mov_pad_val(_PAD_VALUE)
    else:
        pto.set_loop_size_ubtoout(_DEFAULT_DMA_LOOP_COUNT, _DEFAULT_DMA_LOOP_COUNT)
        pto.set_loop_size_outtoub(_DEFAULT_DMA_LOOP_COUNT, _DEFAULT_DMA_LOOP_COUNT)

    pto.set_store_atomic_cfg(_DEFAULT_ST_ATOMIC_CFG)
