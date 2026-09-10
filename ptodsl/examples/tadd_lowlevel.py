# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Builds the MLIR IR module equivalent to expand_tileop_to_vpto_result.pto using
low-level MLIR Python bindings.

Target IR (expand_tileop_to_vpto_result.pto):
  module attributes {pto.target_arch = "a5"} {
    module attributes {pto.kernel_kind = #pto.kernel_kind<vector>, pto.target_arch = "a5"} {
      func.func @TADD() {
        %c0_i64 = arith.constant 0 : i64
        %c16 = arith.constant 16 : index
        %c4096_i64 = arith.constant 4096 : i64
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c64_i32 = arith.constant 64 : i32
        %c64 = arith.constant 64 : index
        pto.vecscope {
          %0 = pto.castptr %c4096_i64 : i64 -> !pto.ptr<f32, ub>
          %1 = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>
          scf.for %arg0 = %c0 to %c16 step %c1 {
            %mask, %scalar_out = pto.plt_b32 %c64_i32 : i32 -> !pto.mask<b32>, i32
            %2 = arith.muli %arg0, %c64 : index
            %3 = pto.addptr %0, %2 : <f32, ub> -> <f32, ub>
            %4 = pto.vlds %3[%c0] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
            %5 = pto.addptr %1, %2 : <f32, ub> -> <f32, ub>
            %6 = pto.vlds %5[%c0] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
            %7 = pto.vadd %4, %6, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
            pto.vsts %7, %5[%c0], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
          }
        }
        return
      }
    }
  }
"""

from ptoas.mlir.ir import (
    Attribute,
    Context,
    F32Type,
    IntegerType,
    IndexType,
    InsertionPoint,
    Location,
    Module,
    Operation,
    StringAttr,
    Type,
)
from ptoas.mlir.dialects import arith, func, pto, scf


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown():
            # ── Types ────────────────────────────────────────────────────────
            i32 = IntegerType.get_signless(32)
            i64 = IntegerType.get_signless(64)
            idx = IndexType.get()
            f32 = F32Type.get()

            # !pto.ptr<f32, ub>  –  pointer to f32 in the UB (VEC) address space
            ptr_f32_ub = pto.PtrType.get(
                f32, memory_space=pto.AddressSpaceAttr.get(pto.AddressSpace.VEC)
            )

            # VReg and Mask types have no Python-binding constructors yet;
            # Type.parse is the only available path for these two.
            vreg_64f32 = Type.parse("!pto.vreg<64xf32>")
            mask_b32   = Type.parse("!pto.mask<b32>")

            # ── Shared attributes ─────────────────────────────────────────
            target_arch_attr = StringAttr.get("a5")
            kernel_kind_attr = Attribute.parse("#pto.kernel_kind<vector>")

            # ── Outer module ─────────────────────────────────────────────
            outer_mod = Module.create()
            outer_mod.operation.attributes["pto.target_arch"] = target_arch_attr

            with InsertionPoint(outer_mod.body):
                # ── Inner module  ─────────────────────────────────────────
                # Module.create() does not use the active InsertionPoint, so we
                # use Operation.create("builtin.module") directly instead.
                inner_op = Operation.create("builtin.module", regions=1)
                inner_op.attributes["pto.target_arch"] = target_arch_attr
                inner_op.attributes["pto.kernel_kind"] = kernel_kind_attr

                # builtin.module needs exactly one block in its body region.
                inner_body = inner_op.regions[0].blocks.append()

                with InsertionPoint(inner_body):
                    # ── func @TADD() ──────────────────────────────────────
                    fn_ty = func.FunctionType.get([], [])
                    fn = func.FuncOp("TADD", fn_ty)
                    entry = fn.add_entry_block()

                    with InsertionPoint(entry):
                        # Constants live outside vecscope; they are visible
                        # inside because vecscope is not a new scope for SSA.
                        c0_i64 = arith.ConstantOp(i64, 0).result
                        c16 = arith.ConstantOp(idx, 16).result
                        c4096_i64 = arith.ConstantOp(i64, 4096).result
                        c0 = arith.ConstantOp(idx, 0).result
                        c1 = arith.ConstantOp(idx, 1).result
                        c64_i32 = arith.ConstantOp(i32, 64).result
                        c64 = arith.ConstantOp(idx, 64).result

                        # ── pto.vecscope { … } ────────────────────────────
                        vecscope_op = pto.VecScopeOp()
                        # vecscope has one region; we must append its entry block.
                        vs_block = vecscope_op.body.blocks.append()

                        with InsertionPoint(vs_block):
                            # %0 = pto.castptr %c4096_i64 : i64 -> !pto.ptr<f32, ub>
                            ptr0 = pto.CastPtrOp(ptr_f32_ub, c4096_i64).result

                            # %1 = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>
                            ptr1 = pto.CastPtrOp(ptr_f32_ub, c0_i64).result

                            # scf.for %arg0 = %c0 to %c16 step %c1 { … }
                            for_op = scf.ForOp(c0, c16, c1)
                            with InsertionPoint(for_op.body):
                                arg0 = for_op.induction_variable

                                # %mask, %scalar_out = pto.plt_b32 %c64_i32
                                plt = pto.PltB32Op(mask_b32, i32, c64_i32)
                                mask = plt.mask
                                # scalar_out is unused in this kernel

                                # %2 = arith.muli %arg0, %c64 : index
                                off = arith.MulIOp(arg0, c64).result

                                # %3 = pto.addptr %0, %2
                                ptr3 = pto.AddPtrOp(ptr0, off).result

                                # %4 = pto.vlds %3[%c0] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
                                vreg4 = pto.VldsOp(vreg_64f32, ptr3, c0).result

                                # %5 = pto.addptr %1, %2
                                ptr5 = pto.AddPtrOp(ptr1, off).result

                                # %6 = pto.vlds %5[%c0] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
                                vreg6 = pto.VldsOp(vreg_64f32, ptr5, c0).result

                                # %7 = pto.vadd %4, %6, %mask
                                vreg7 = pto.VaddOp(vreg_64f32, vreg4, vreg6, mask).result

                                # pto.vsts %7, %5[%c0], %mask
                                pto.VstsOp(vreg7, ptr5, c0, mask)

                                scf.YieldOp([])

                        func.ReturnOp([])

            outer_mod.operation.verify()
            return outer_mod


if __name__ == "__main__":
    print(build())
