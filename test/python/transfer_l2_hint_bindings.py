#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise direction-specific A5 hint values, builders and IR round trips."""

from ptoas.mlir.dialects import pto
from ptoas.mlir.ir import Context, InsertionPoint, Location, Module


LOAD_MODES = (
    ("NormalFirstVictim", "normal_first_victim", 0),
    ("NormalLastVictim", "normal_last_victim", 1),
    ("NormalPersistent", "normal_persistent", 2),
    ("NotAllocKeep", "not_alloc_keep", 4),
    ("NotAllocClean", "not_alloc_clean", 5),
    ("NotAllocDrop", "not_alloc_drop", 6),
)
STORE_MODES = LOAD_MODES[:3] + (("NotAllocClean", "not_alloc_clean", 4),)


def check_modes(enum_type, attr_type, spelling, modes):
    """Check every enum encoding and the C API's sparse-value validation."""
    for name, keyword, encoding in modes:
        mode = getattr(enum_type, name)
        attr = attr_type.get(mode)
        expected = f"#pto.{spelling}<{keyword}>"
        if mode.value != encoding or attr.value != encoding or str(attr) != expected:
            raise RuntimeError(f"incorrect {spelling} mode: {name}")
        if attr_type.get(encoding) != attr:
            raise RuntimeError(f"integer builder differs for {name}")
    valid_values = {encoding for _, _, encoding in modes}
    for encoding in (-1, 3, 4, 5, 6, 7, 2**31 - 1):
        if encoding not in valid_values and attr_type.get(encoding) is not None:
            raise RuntimeError(f"accepted invalid {spelling} encoding {encoding}")


def check_builders():
    """Preserve optional attributes, explicit zero, and every per-op selection."""
    module = Module.parse("""
        module attributes {pto.target_arch = "a5"} {
          func.func @transfers(%view: !pto.partition_tensor_view<16x16xf32>,
                               %tile: !pto.tile_buf<vec, 16x16xf32>) {
            return
          }
        }
    """)
    block = module.body.operations[0].regions[0].blocks[0]
    view, tile = block.arguments
    with InsertionPoint.at_block_begin(block):
        plain_load = pto.TLoadOp(None, view, tile)
        plain_store = pto.TStoreOp(None, tile, view)
        for name, _, _ in LOAD_MODES:
            pto.TLoadOp(None, view, tile, l2_hint=getattr(pto.LoadL2Hint, name))
        for name, _, _ in STORE_MODES:
            pto.TStoreOp(None, tile, view, l2_hint=getattr(pto.StoreL2Hint, name))
    if plain_load.l2_hint is not None or plain_store.l2_hint is not None:
        raise RuntimeError("ordinary transfers acquired an implicit hint")
    if not module.operation.verify():
        raise RuntimeError("hinted transfer builders did not verify")
    text = str(module)
    round_trip = Module.parse(text)
    if not round_trip.operation.verify() or str(round_trip) != text:
        raise RuntimeError("hinted transfers did not survive a round trip")
    for spelling, modes in (("load_l2_hint", LOAD_MODES), ("store_l2_hint", STORE_MODES)):
        for _, keyword, _ in modes:
            if text.count(f"#pto.{spelling}<{keyword}>") != 1:
                raise RuntimeError(f"lost or duplicated {spelling} {keyword}")


def main():
    with Context() as ctx, Location.unknown(ctx):
        pto.register_dialect(ctx, load=True)
        check_modes(pto.LoadL2Hint, pto.LoadL2HintAttr, "load_l2_hint", LOAD_MODES)
        check_modes(pto.StoreL2Hint, pto.StoreL2HintAttr, "store_l2_hint", STORE_MODES)
        check_builders()
    print("transfer_l2_hint_bindings: PASS")


if __name__ == "__main__":
    main()
