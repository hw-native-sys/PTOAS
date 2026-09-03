# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import unittest
import inspect
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, call, patch

import ptodsl._ops as _ops
import ptodsl._pipe_namespace as _pipe_namespace
from ptodsl._context import make_context
from ptodsl._diagnostics import PTODSLDeprecationWarning
from ptodsl import pto
from ptoas.mlir.ir import F32Type, IntegerType, Type

import ptodsl._ops_common as _ops_common
import ptodsl._ops_core as _ops_core
import ptodsl._ops_vmem as _ops_vmem
import ptodsl._ops_mask as _ops_mask
import ptodsl._ops_vmath as _ops_vmath
import ptodsl._ops_tile as _ops_tile
import ptodsl._ops_mte as _ops_mte
import ptodsl._ops_simt as _ops_simt
from contextlib import ExitStack, contextmanager

_OPS_FAMILY = (
    _ops, _ops_common, _ops_core, _ops_vmem, _ops_mask,
    _ops_vmath, _ops_tile, _ops_mte, _ops_simt,
)


@contextmanager
def patch_ops(name, **kwargs):
    """Patch ``name`` across every ``_ops``-family module that binds it.

    The op implementations were split out of ``ptodsl._ops`` into the
    ``_ops_*`` modules, so a name can be resolved from any of them. One
    mock is installed everywhere and yielded once, so call assertions see
    every lookup site, matching the pre-split ``_ops`` patch seam.
    """
    mock = MagicMock(**kwargs)
    with ExitStack() as stack:
        targets = [module for module in _OPS_FAMILY if hasattr(module, name)]
        assert targets, f"no _ops-family module binds {name!r}"
        for module in targets:
            stack.enter_context(patch.object(module, name, mock))
        yield mock


def _identity(value):
    return value


class VectorCubeSurfaceTest(unittest.TestCase):
    def test_mte_mx_full_operands_preserve_all_values(self):
        source = object()
        destination = object()
        controls = {
            "x_start": 3,
            "y_start": 5,
            "x_step": 16,
            "y_step": 2,
            "src_stride": 8,
            "dst_stride": 2,
        }

        def coerce(value, *, context):
            return f"{context}:{value}"

        with patch_ops("_require_explicit_mode"), \
             patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("_coerce_i64", side_effect=coerce), \
             patch.object(_ops._pto, "MteL1L0aMxOp") as load_ca:
            pto.mte_l1_l0a_mx(source, destination, **controls)
        load_ca.assert_called_once_with(
            source,
            destination,
            x_start="mte_l1_l0a_mx x_start:3",
            y_start="mte_l1_l0a_mx y_start:5",
            x_step="mte_l1_l0a_mx x_step:16",
            y_step="mte_l1_l0a_mx y_step:2",
            src_stride="mte_l1_l0a_mx src_stride:8",
            dst_stride="mte_l1_l0a_mx dst_stride:2",
        )

        with patch_ops("_require_explicit_mode"), \
             patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("_coerce_i64", side_effect=coerce), \
             patch.object(_ops._pto, "MteL1L0bMxOp") as load_cb:
            pto.mte_l1_l0b_mx(source, destination, **controls)
        load_cb.assert_called_once_with(
            source,
            destination,
            x_start="mte_l1_l0b_mx x_start:3",
            y_start="mte_l1_l0b_mx y_start:5",
            x_step="mte_l1_l0b_mx x_step:16",
            y_step="mte_l1_l0b_mx y_step:2",
            src_stride="mte_l1_l0b_mx src_stride:8",
            dst_stride="mte_l1_l0b_mx dst_stride:2",
        )

    def test_mte_mx_shape_operands_are_named(self):
        source = object()
        destination = object()

        def coerce(value, *, context):
            return f"{context}:{value}"

        with patch_ops("_require_explicit_mode"), \
             patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("_coerce_i64", side_effect=coerce), \
             patch.object(_ops._pto, "MteL1L0aMxOp") as load_ca:
            pto.mte_l1_l0a_mx(
                source, destination, 128, 256, start_row=3, start_col=5
            )
        load_ca.assert_called_once_with(
            source,
            destination,
            m="mte_l1_l0a_mx m:128",
            k="mte_l1_l0a_mx k:256",
            start_row="mte_l1_l0a_mx start_row:3",
            start_col="mte_l1_l0a_mx start_col:5",
        )

        with patch_ops("_require_explicit_mode"), \
             patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("_coerce_i64", side_effect=coerce), \
             patch.object(_ops._pto, "MteL1L0bMxOp") as load_cb:
            pto.mte_l1_l0b_mx(
                source, destination, 256, 128, start_row=5, start_col=3
            )
        load_cb.assert_called_once_with(
            source,
            destination,
            k="mte_l1_l0b_mx k:256",
            n="mte_l1_l0b_mx n:128",
            start_row="mte_l1_l0b_mx start_row:5",
            start_col="mte_l1_l0b_mx start_col:3",
        )

    def test_mte_mx_shape_operands_default_starts_to_zero(self):
        source = object()
        destination = object()

        def coerce(value, *, context):
            return f"{context}:{value}"

        with patch_ops("_require_explicit_mode"), \
             patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("_coerce_i64", side_effect=coerce), \
             patch.object(_ops._pto, "MteL1L0aMxOp") as load_ca:
            pto.mte_l1_l0a_mx(source, destination, 128, 256)
        load_ca.assert_called_once_with(
            source,
            destination,
            m="mte_l1_l0a_mx m:128",
            k="mte_l1_l0a_mx k:256",
            start_row="mte_l1_l0a_mx start_row:0",
            start_col="mte_l1_l0a_mx start_col:0",
        )

    def test_mte_mx_full_operands_require_a_complete_mode(self):
        source = object()
        destination = object()
        complete_controls = {
            "x_start": 3,
            "y_start": 5,
            "x_step": 16,
            "y_step": 2,
            "src_stride": 8,
            "dst_stride": 2,
        }

        with patch_ops("_require_explicit_mode"):
            with self.assertRaisesRegex(TypeError, "require x_start"):
                pto.mte_l1_l0a_mx(source, destination, x_start=3)
            with self.assertRaisesRegex(TypeError, "either k/n or full"):
                pto.mte_l1_l0b_mx(source, destination, 128, 256, **complete_controls)
        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("_coerce_i64", side_effect=lambda value, *, context: f"{context}:{value}"), \
             patch.object(_ops._pto, "MteL1L0aMxOp") as load_ca:
            pto.mte_l1_l0a_mx(
                source, destination, start_row=0, start_col=0,
                **complete_controls
            )
        load_ca.assert_called_once_with(
            source,
            destination,
            x_start="mte_l1_l0a_mx x_start:3",
            y_start="mte_l1_l0a_mx y_start:5",
            x_step="mte_l1_l0a_mx x_step:16",
            y_step="mte_l1_l0a_mx y_step:2",
            src_stride="mte_l1_l0a_mx src_stride:8",
            dst_stride="mte_l1_l0a_mx dst_stride:2",
        )

        self.assertFalse(hasattr(pto, "load_cbuf_to_ca_mx"))
        self.assertFalse(hasattr(pto, "load_cbuf_to_cb_mx"))

    def test_row_reduction_auto_tmp_prefers_surface_metadata(self):
        src = SimpleNamespace(
            type="ignored_type",
            surface_metadata={
                "shape": (8, 64),
                "dtype": "f32_type",
                "memory_space": "vec",
                "valid_shape": ("rows", "cols"),
            },
        )
        auto_tmp = object()
        dst = object()
        sentinel = object()

        with patch_ops("parse_tile_type_metadata") as parse_tile_type_metadata, \
             patch_ops("alloc_tile", return_value=auto_tmp) as alloc_tile, \
             patch_ops("trowmax", return_value=sentinel) as trowmax:
            result = pto.tile.rowmax(src, dst=dst, tmp=None)

        self.assertIs(result, sentinel)
        parse_tile_type_metadata.assert_not_called()
        alloc_tile.assert_called_once_with(
            shape=[8, 64],
            dtype="f32_type",
            memory_space="vec",
            valid_shape=["rows", "cols"],
            blayout="RowMajor",
            slayout="NoneBox",
        )
        trowmax.assert_called_once_with(src, auto_tmp, dst)

    def test_row_reduction_auto_tmp_uses_row_major_layout(self):
        src = SimpleNamespace(type="!pto.tile_buf<vec, 8x64xf32, valid=8x64, blayout=col_major>")
        auto_tmp = object()
        dst = object()
        sentinel = object()
        element_type = object()
        metadata = {
            "shape_dims": (8, 64),
            "element_type": element_type,
            "memory_space": "vec",
            "valid_dims": (8, 64),
        }

        with patch_ops("unwrap_surface_value", return_value=src), \
             patch_ops("parse_tile_type_metadata", return_value=metadata), \
             patch_ops("alloc_tile", return_value=auto_tmp) as alloc_tile, \
             patch_ops("trowmax", return_value=sentinel) as trowmax:
            result = pto.tile.rowmax(src, dst=dst, tmp=None)

        self.assertIs(result, sentinel)
        alloc_tile.assert_called_once_with(
            shape=[8, 64],
            dtype=element_type,
            memory_space="vec",
            valid_shape=[8, 64],
            blayout="RowMajor",
            slayout="NoneBox",
        )
        trowmax.assert_called_once_with(src, auto_tmp, dst)

    def test_public_namespace_exports_new_vector_and_cube_apis(self):
        names = [
            "vsub", "vmin", "vand", "vor", "vxor", "vshl", "vshr",
            "vln", "vsqrt", "vabs", "vneg", "vrec", "vrsqrt", "vrelu", "vnot",
            "vsqz",
            "vcmin", "vcgmin", "vcpadd",
            "vadds", "vmuls", "vmaxs", "vmins", "vlrelu", "vshls", "vshrs", "vands", "vors", "vxors",
            "vaxpy", "vmula", "vci", "vaddrelu", "vsubrelu", "vsel",
            "mte_gm_l1", "raw_fill_l1", "mte_l1_ub", "mte_gm_l1_frac", "mte_l1_bt", "mte_l1_fb",
            "mad_acc", "mad_bias", "mad_mx", "mad_mx_acc", "mad_mx_bias",
            "FractalMode", "AccStoreUnitFlagCtrl", "MadUnitFlagMode", "SatMode", "Tf32Mode", "SplitMode",
        ]

        for name in names:
            self.assertTrue(hasattr(pto, name), name)

    def test_raw_fill_l1_preserves_all_control_fields(self):
        destination = object()

        def coerce(value, *, context):
            return f"{context}:{value}"

        with patch_ops("_require_explicit_mode"), \
             patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("_coerce_i64", side_effect=coerce), \
             patch_ops("_coerce_i32", side_effect=coerce), \
             patch_ops("IntegerAttr") as integer_attr, \
             patch_ops("IntegerType") as integer_type, \
             patch.object(_ops._pto, "RawFillL1Op") as raw_fill:
            pto.raw_fill_l1(
                destination,
                32,
                0x12345678,
                repeat_times=3,
                block_num_32b=7,
                dst_gap_32b=11,
                fill_word_bits=16,
            )

        integer_type.get_signless.assert_called_once_with(64)
        integer_attr.get.assert_called_once_with(ANY, 16)
        raw_fill.assert_called_once_with(
            destination,
            "raw_fill_l1 byte_offset:32",
            "raw_fill_l1 raw_value:305419896",
            "raw_fill_l1 repeat_times:3",
            "raw_fill_l1 block_num_32b:7",
            "raw_fill_l1 dst_gap_32b:11",
            integer_attr.get.return_value,
        )

    def test_raw_fill_l1_rejects_positional_control_fields(self):
        destination = object()
        with patch_ops("_require_explicit_mode"):
            with self.assertRaises(TypeError):
                pto.raw_fill_l1(destination, 32, 0x12345678, 3, 7, 11, 16)

    def test_raw_fill_l1_validates_fill_word_bits(self):
        destination = object()
        with patch_ops("_require_explicit_mode"):
            with self.assertRaises(ValueError):
                pto.raw_fill_l1(
                    destination, 32, 0,
                    repeat_times=1, block_num_32b=1, dst_gap_32b=0,
                    fill_word_bits=8,
                )
            with self.assertRaises(TypeError):
                pto.raw_fill_l1(
                    destination, 32, 0,
                    repeat_times=1, block_num_32b=1, dst_gap_32b=0,
                    fill_word_bits=True,
                )
            with self.assertRaises(TypeError):
                pto.raw_fill_l1(
                    destination, 32, 0,
                    repeat_times=1, block_num_32b=1, dst_gap_32b=0,
                    fill_word_bits=16.0,
                )

    def test_raw_fill_l1_validates_static_geometry(self):
        destination = object()
        with patch_ops("_require_explicit_mode"):
            with self.assertRaises(ValueError):
                pto.raw_fill_l1(
                    destination, 2, 0,
                    repeat_times=1, block_num_32b=1, dst_gap_32b=0,
                    fill_word_bits=16,
                )
            with self.assertRaises(ValueError):
                pto.raw_fill_l1(
                    destination, -1, 0,
                    repeat_times=1, block_num_32b=1, dst_gap_32b=0,
                    fill_word_bits=16,
                )
            with self.assertRaises(ValueError):
                pto.raw_fill_l1(
                    destination, 32, 0,
                    repeat_times=32768, block_num_32b=1, dst_gap_32b=0,
                    fill_word_bits=16,
                )
            with self.assertRaises(ValueError):
                pto.raw_fill_l1(
                    destination, 32, 0,
                    repeat_times=1, block_num_32b=1, dst_gap_32b=-2,
                    fill_word_bits=16,
                )

    def test_tile_bitwise_aliases_are_exposed_without_legacy_names(self):
        preferred_names = [
            "bit_not", "bit_and", "bit_ands", "bit_or", "bit_ors",
            "bit_xor", "bit_xors", "bit_shl", "bit_shls", "bit_shr", "bit_shrs",
        ]
        legacy_names = [
            "not_", "and_", "ands", "or_", "ors", "xor", "xors", "shl", "shls", "shr", "shrs",
        ]

        for name in preferred_names:
            with self.subTest(name=name):
                self.assertTrue(hasattr(pto.tile, name), name)

        for name in legacy_names:
            with self.subTest(name=name):
                self.assertFalse(hasattr(pto.tile, name), name)

    def test_tile_partial_and_fillpad_names_are_exposed_without_legacy_names(self):
        preferred_names = [
            "partadd", "partmul", "partmax", "partmin",
            "fillpad",
        ]
        legacy_names = [
            "part_add", "part_mul", "part_max", "part_min",
            "fill_pad", "fill_pad_expand", "fill_pad_inplace",
            "fillpad_expand", "fillpad_inplace",
        ]

        for name in preferred_names:
            with self.subTest(name=name):
                self.assertTrue(hasattr(pto.tile, name), name)

        for name in legacy_names:
            with self.subTest(name=name):
                self.assertFalse(hasattr(pto.tile, name), name)

    def test_tile_fillpad_dispatches_one_op_without_mode(self):
        src = object()
        dst = object()

        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch.object(_ops._pto, "tfillpad") as tfillpad:
            pto.tile.fillpad(src, dst)

        tfillpad.assert_called_once_with(src, dst)

    def test_tile_fillpad_rejects_mode_argument(self):
        with self.assertRaises(TypeError):
            pto.tile.fillpad(object(), object(), mode="expand")

    def test_sync_flag_names_are_exposed_without_legacy_aliases(self):
        preferred_names = [
            "set_cross_block", "wait_cross_block",
            "set_intra_block", "wait_intra_block",
        ]
        legacy_names = [
            "set_cross_flag", "wait_cross_flag",
            "set_intra_flag", "wait_intra_flag",
        ]

        for name in preferred_names:
            with self.subTest(name=name):
                self.assertTrue(hasattr(pto, name), name)

        for name in legacy_names:
            with self.subTest(name=name):
                self.assertFalse(hasattr(pto, name), name)

    def test_cross_block_sync_rejects_ffts_mode(self):
        with self.assertRaises(TypeError):
            _ops.set_cross_block(pto.Pipe.FIX, 0, ffts_mode=2)

    def test_direct_vector_wrappers_dispatch_to_generated_ops(self):
        lhs = SimpleNamespace(type="vec_ty")
        rhs = SimpleNamespace(type="vec_ty")
        mask = SimpleNamespace(type="mask_ty")
        result = object()

        binary_cases = [
            ("vsub", "VsubOp", (lhs, rhs, mask)),
            ("vmin", "VminOp", (lhs, rhs, mask)),
            ("vand", "VandOp", (lhs, rhs, mask)),
            ("vor", "VorOp", (lhs, rhs, mask)),
            ("vxor", "VxorOp", (lhs, rhs, mask)),
            ("vshl", "VshlOp", (lhs, rhs, mask)),
            ("vshr", "VshrOp", (lhs, rhs, mask)),
        ]
        unary_cases = [
            ("vln", "VlnOp", (lhs, mask)),
            ("vsqrt", "VsqrtOp", (lhs, mask)),
            ("vabs", "VabsOp", (lhs, mask)),
            ("vneg", "VnegOp", (lhs, mask)),
            ("vrelu", "VreluOp", (lhs, mask)),
            ("vnot", "VnotOp", (lhs, mask)),
            ("vcmin", "VcminOp", (lhs, mask)),
            ("vcpadd", "VcpaddOp", (lhs, mask)),
        ]

        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("wrap_surface_value", side_effect=_identity):
            for func_name, op_name, args in binary_cases + unary_cases:
                with self.subTest(func=func_name):
                    fake_op = SimpleNamespace(result=result)
                    with patch.object(_ops._pto, op_name, return_value=fake_op) as op_ctor:
                        output = getattr(_ops, func_name)(*args)
                    self.assertIs(output, result)
                    self.assertEqual(op_ctor.call_args.args[0], "vec_ty")

    def test_vec_scalar_wrappers_and_vaxpy_coerce_scalar_operands(self):
        vec = SimpleNamespace(type="vec_ty")
        other = SimpleNamespace(type="vec_ty")
        mask = SimpleNamespace(type="mask_ty")
        scalar = object()
        coerced_scalar = object()
        result = object()

        vec_scalar_cases = [
            ("vadds", "VaddsOp"),
            ("vmuls", "VmulsOp"),
            ("vmaxs", "VmaxsOp"),
            ("vmins", "VminsOp"),
            ("vlrelu", "VlreluOp"),
        ]

        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("wrap_surface_value", side_effect=_identity), \
             patch_ops("_coerce_scalar_like_vector_element", return_value=coerced_scalar) as coerce_scalar:
            for func_name, op_name in vec_scalar_cases:
                with self.subTest(func=func_name):
                    fake_op = SimpleNamespace(result=result)
                    with patch.object(_ops._pto, op_name, return_value=fake_op) as op_ctor:
                        output = getattr(_ops, func_name)(vec, scalar, mask)
                    self.assertIs(output, result)
                    self.assertEqual(op_ctor.call_args.args, ("vec_ty", vec, coerced_scalar, mask))

            fake_op = SimpleNamespace(result=result)
            with patch.object(_ops._pto, "VaxpyOp", return_value=fake_op) as op_ctor:
                output = _ops.vaxpy(scalar, vec, other, mask)
            self.assertIs(output, result)
            self.assertEqual(op_ctor.call_args.args, ("vec_ty", vec, other, coerced_scalar, mask))
            self.assertGreaterEqual(coerce_scalar.call_count, len(vec_scalar_cases) + 1)

    def test_composed_vector_wrappers_chain_existing_primitives(self):
        vec = object()
        rhs = object()
        mask = object()
        zero_vec = object()
        one_vec = object()
        sqrt_vec = object()
        add_vec = object()
        sub_vec = object()
        relu_vec = object()
        reciprocal_vec = object()

        with patch_ops("vmuls", return_value=zero_vec) as vmuls, \
             patch_ops("vadds", return_value=one_vec) as vadds, \
             patch_ops("vdiv", return_value=reciprocal_vec) as vdiv:
            self.assertIs(_ops.vrec(vec, mask), reciprocal_vec)
            vmuls.assert_called_once_with(vec, 0, mask)
            vadds.assert_called_once_with(zero_vec, 1, mask)
            vdiv.assert_called_once_with(one_vec, vec, mask)

        with patch_ops("vsqrt", return_value=sqrt_vec) as vsqrt, \
             patch_ops("vrec", return_value=reciprocal_vec) as vrec:
            self.assertIs(_ops.vrsqrt(vec, mask), reciprocal_vec)
            vsqrt.assert_called_once_with(vec, mask)
            vrec.assert_called_once_with(sqrt_vec, mask)

        with patch_ops("vadd", return_value=add_vec) as vadd, \
             patch_ops("vrelu", return_value=relu_vec) as vrelu:
            self.assertIs(_ops.vaddrelu(vec, rhs, mask), relu_vec)
            vadd.assert_called_once_with(vec, rhs, mask)
            vrelu.assert_called_once_with(add_vec, mask)

        with patch_ops("vsub", return_value=sub_vec) as vsub, \
             patch_ops("vrelu", return_value=relu_vec) as vrelu:
            self.assertIs(_ops.vsubrelu(vec, rhs, mask), relu_vec)
            vsub.assert_called_once_with(vec, rhs, mask)
            vrelu.assert_called_once_with(sub_vec, mask)

    def test_vsqz_dispatches_to_generated_op_with_source_type(self):
        inp = SimpleNamespace(type="vec_ty")
        mask = SimpleNamespace(type="mask_ty")
        result = object()

        with patch_ops(
            "unwrap_surface_value", side_effect=_identity
        ), patch_ops(
            "wrap_surface_value", side_effect=_identity
        ), patch.object(
            _ops._pto,
            "VsqzOp",
            return_value=SimpleNamespace(result=result),
        ) as op_ctor:
            output = _ops.vsqz(inp, mask)

        self.assertIs(output, result)
        op_ctor.assert_called_once_with("vec_ty", inp, mask)

    def test_vcgmin_and_vsel_dispatch_correctly(self):
        vec = SimpleNamespace(type="vec_ty")
        other = SimpleNamespace(type="vec_ty")
        mask = SimpleNamespace(type="mask_ty")
        reduced = object()
        selected = object()

        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("wrap_surface_value", side_effect=_identity), \
             patch.object(_ops._pto, "VcgminOp", return_value=SimpleNamespace(result=reduced)) as vcgmin_op:
            output = _ops.vcgmin(vec, mask)
        self.assertIs(output, reduced)
        self.assertEqual(vcgmin_op.call_args.args, ("vec_ty", vec, mask))

        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("wrap_surface_value", side_effect=_identity), \
             patch.object(_ops._pto, "VselOp", return_value=SimpleNamespace(result=selected)) as vsel_op:
            output = _ops.vsel(vec, other, mask)
        self.assertIs(output, selected)
        self.assertEqual(vsel_op.call_args.args, ("vec_ty", vec, other, mask))

    def test_vector_scalar_helper_dispatches_cover_shift_and_bitwise_scalar_helpers(self):
        vec = SimpleNamespace(type="vec_ty")
        mask = SimpleNamespace(type="mask_ty")
        scalar = SimpleNamespace(type="scalar_ty")
        scalar_i16 = SimpleNamespace(type="i16_ty")
        shifted = object()
        anded = object()
        ored = object()
        xored = object()
        broadcast = object()

        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("wrap_surface_value", side_effect=_identity), \
             patch_ops("_reject_low_precision_vreg_operands") as reject_lp, \
             patch.object(_ops.IntegerType, "get_signless", return_value="i16_ty") as get_signless, \
             patch_ops("coerce_scalar_to_type", return_value=scalar_i16) as coerce_i16, \
             patch.object(_ops._pto, "VshlsOp", return_value=SimpleNamespace(result=shifted)) as vshls_op, \
             patch.object(_ops._pto, "VshrsOp", return_value=SimpleNamespace(result=shifted)) as vshrs_op:
            self.assertIs(_ops.vshls(vec, scalar, mask), shifted)
            self.assertIs(_ops.vshrs(vec, scalar, mask), shifted)
        self.assertEqual(reject_lp.call_count, 2)
        self.assertEqual(get_signless.call_count, 2)
        self.assertEqual(coerce_i16.call_count, 2)
        self.assertEqual(vshls_op.call_args.args, ("vec_ty", vec, scalar_i16, mask))
        self.assertEqual(vshrs_op.call_args.args, ("vec_ty", vec, scalar_i16, mask))

        with patch_ops("_coerce_scalar_like_vector_element", return_value=scalar) as coerce_scalar, \
             patch_ops("vbr", return_value=broadcast) as vbr, \
             patch_ops("vand", return_value=anded) as vand:
            self.assertIs(_ops.vands(vec, scalar, mask), anded)
        coerce_scalar.assert_called_once_with(vec, scalar, context="vands")
        vbr.assert_called_once_with(scalar)
        vand.assert_called_once_with(vec, broadcast, mask)

        with patch_ops("_coerce_scalar_like_vector_element", return_value=scalar) as coerce_scalar, \
             patch_ops("vbr", return_value=broadcast) as vbr, \
             patch_ops("vor", return_value=ored) as vor:
            self.assertIs(_ops.vors(vec, scalar, mask), ored)
        coerce_scalar.assert_called_once_with(vec, scalar, context="vors")
        vbr.assert_called_once_with(scalar)
        vor.assert_called_once_with(vec, broadcast, mask)

        with patch_ops("_coerce_scalar_like_vector_element", return_value=scalar) as coerce_scalar, \
             patch_ops("vbr", return_value=broadcast) as vbr, \
             patch_ops("vxor", return_value=xored) as vxor:
            self.assertIs(_ops.vxors(vec, scalar, mask), xored)
        coerce_scalar.assert_called_once_with(vec, scalar, context="vxors")
        vbr.assert_called_once_with(scalar)
        vxor.assert_called_once_with(vec, broadcast, mask)


    def test_vlds_accepts_extended_distribution_tokens(self):
        ptr = SimpleNamespace(type="ptr_ty")
        vec = object()

        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("wrap_surface_value", side_effect=_identity), \
             patch_ops("_infer_vreg_type_from_address_source", return_value="vec_ty"), \
             patch_ops("_coerce_index", return_value="idx"), \
             patch_ops("_normalize_post_update_mode", return_value="NO_POST_UPDATE"), \
             patch_ops("_normalize_dist_token", side_effect=lambda dist, *, allowed, context: dist) as normalize_dist, \
             patch.object(_ops._pto, "VldsOp", return_value=SimpleNamespace(result=vec)) as vlds_op:
            self.assertIs(_ops.vlds(ptr, 0, dist="E2B_B16"), vec)
            self.assertIs(_ops.vlds(ptr, 0, dist="BRC_BLK"), vec)
        self.assertEqual(normalize_dist.call_args_list[0].args[0], "E2B_B16")
        self.assertEqual(normalize_dist.call_args_list[1].args[0], "BRC_BLK")
        self.assertEqual(vlds_op.call_count, 2)

    def test_f32_to_fp8_vcvt_contract_accepts_rahz_rounding(self):
        contract = _ops._VCVT_CONTRACTS[("f32", "f8e4m3")]
        for rnd in ("R", "A", "H", "Z"):
            with self.subTest(rnd=rnd):
                _ops._validate_vcvt_attrs(
                    "f32",
                    "f8e4m3",
                    contract,
                    rnd=rnd,
                    sat="SAT",
                    part="P0",
                    context="pto.vcvt",
                )
        with self.assertRaisesRegex(ValueError, r"expected one of A, H, R, Z"):
            _ops._validate_vcvt_attrs(
                "f32",
                "f8e4m3",
                contract,
                rnd="F",
                sat="SAT",
                part="P0",
                context="pto.vcvt",
            )

    def test_cube_variant_wrappers_dispatch_to_generated_ops(self):
        lhs = object()
        rhs = object()
        dst = object()
        bias = object()
        lhs_scale = object()
        rhs_scale = object()
        acc_in = object()

        cube_cases = [
            ("mad_acc", "MadAccOp", (lhs, rhs, dst, 1, 2, 3), (lhs, rhs, dst, "i64:1", "i64:2", "i64:3")),
            ("mad_bias", "MadBiasOp", (lhs, rhs, dst, bias, 1, 2, 3), (lhs, rhs, dst, bias, "i64:1", "i64:2", "i64:3")),
            ("mad_mx", "MadMxOp", (lhs, rhs, dst, 1, 2, 3), (lhs, rhs, dst, "i64:1", "i64:2", "i64:3")),
            ("mad_mx_acc", "MadMxAccOp", (lhs, rhs, dst, 1, 2, 3), (lhs, rhs, dst, "i64:1", "i64:2", "i64:3")),
            ("mad_mx_bias", "MadMxBiasOp", (lhs, rhs, dst, bias, 1, 2, 3), (lhs, rhs, dst, bias, "i64:1", "i64:2", "i64:3")),
        ]

        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("_coerce_i64", side_effect=lambda value, *, context: f"i64:{value}"):
            for func_name, op_name, args, expected_call in cube_cases:
                with self.subTest(func=func_name):
                    op_ctor = MagicMock()
                    with patch.object(_ops._pto, op_name, op_ctor):
                        getattr(_ops, func_name)(*args)
                    self.assertEqual(op_ctor.call_args.args, expected_call)

        mx_tileop_cases = [
            ("tgemv", "TGemvOp", (lhs, rhs, dst), (None, lhs, rhs, dst)),
            ("tgemv_acc", "TGemvAccOp", (acc_in, lhs, rhs, dst), (None, acc_in, lhs, rhs, dst)),
            ("tgemv_bias", "TGemvBiasOp", (lhs, rhs, bias, dst), (None, lhs, rhs, bias, dst)),
            ("tmatmul_mx", "TMatmulMxOp", (lhs, lhs_scale, rhs, rhs_scale, dst), (None, lhs, lhs_scale, rhs, rhs_scale, dst)),
            ("tmatmul_mx_acc", "TMatmulMxAccOp", (acc_in, lhs, lhs_scale, rhs, rhs_scale, dst), (None, acc_in, lhs, lhs_scale, rhs, rhs_scale, dst)),
            ("tmatmul_mx_bias", "TMatmulMxBiasOp", (lhs, lhs_scale, rhs, rhs_scale, bias, dst), (None, lhs, lhs_scale, rhs, rhs_scale, bias, dst)),
            ("tgemv_mx", "TGemvMxOp", (lhs, lhs_scale, rhs, rhs_scale, dst), (None, lhs, lhs_scale, rhs, rhs_scale, dst)),
            ("tgemv_mx_acc", "TGemvMxAccOp", (acc_in, lhs, lhs_scale, rhs, rhs_scale, dst), (None, acc_in, lhs, lhs_scale, rhs, rhs_scale, dst)),
            ("tgemv_mx_bias", "TGemvMxBiasOp", (lhs, lhs_scale, rhs, rhs_scale, bias, dst), (None, lhs, lhs_scale, rhs, rhs_scale, bias, dst)),
        ]

        with patch_ops("unwrap_surface_value", side_effect=_identity):
            for func_name, op_name, args, expected_call in mx_tileop_cases:
                with self.subTest(func=func_name):
                    op_ctor = MagicMock()
                    with patch.object(_ops._pto, op_name, op_ctor):
                        getattr(_ops, func_name)(*args)
                    self.assertEqual(op_ctor.call_args.args, expected_call)

    def test_mte_l1_l0_explicit_controls_emit_wrapper_ops(self):
        source = object()
        destination = object()
        controls = {
            "m_start": 3,
            "k_start": 5,
            "m_step": 16,
            "k_step": 2,
            "src_stride": 8,
            "dst_stride": 2,
        }

        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("_coerce_i64", side_effect=lambda value, *, context: f"{context}:{value}"):
            ca_op = MagicMock()
            with patch.object(_ops._pto, "MteL1L0aOp", ca_op):
                _ops.mte_l1_l0a(source, destination, **controls, transpose=True)
            self.assertEqual(
                ca_op.call_args.args,
                (
                    source,
                    destination,
                ),
            )
            self.assertEqual(
                ca_op.call_args.kwargs,
                {
                    "m_start": "mte_l1_l0a m_start:3",
                    "k_start": "mte_l1_l0a k_start:5",
                    "m_step": "mte_l1_l0a m_step:16",
                    "k_step": "mte_l1_l0a k_step:2",
                    "src_stride": "mte_l1_l0a src_stride:8",
                    "dst_stride": "mte_l1_l0a dst_stride:2",
                    "transpose": True,
                },
            )

            cb_op = MagicMock()
            with patch.object(_ops._pto, "MteL1L0bOp", cb_op):
                _ops.mte_l1_l0b(source, destination, **controls, transpose=True)
            self.assertEqual(
                cb_op.call_args.args,
                (
                    source,
                    destination,
                ),
            )
            self.assertEqual(
                cb_op.call_args.kwargs,
                {
                    "m_start": "mte_l1_l0b m_start:3",
                    "k_start": "mte_l1_l0b k_start:5",
                    "m_step": "mte_l1_l0b m_step:16",
                    "k_step": "mte_l1_l0b k_step:2",
                    "src_stride": "mte_l1_l0b src_stride:8",
                    "dst_stride": "mte_l1_l0b dst_stride:2",
                    "transpose": True,
                },
            )

    def test_mte_l1_l0_explicit_fp4_controls_stay_on_wrapper_ops(self):
        controls = {
            "m_start": 0,
            "k_start": 4,
            "m_step": 16,
            "k_step": 4,
            "src_stride": 16,
            "dst_stride": 16,
        }

        with make_context():
            for dtype in ("f4E1M2x2", "f4E2M1x2"):
                source = SimpleNamespace(
                    type=Type.parse(f"!pto.ptr<!pto.{dtype}, l1>")
                )
                destination_a = SimpleNamespace(
                    type=Type.parse(f"!pto.ptr<!pto.{dtype}, l0a>")
                )
                destination_b = SimpleNamespace(
                    type=Type.parse(f"!pto.ptr<!pto.{dtype}, l0b>")
                )
                with self.subTest(dtype=dtype), \
                     patch_ops("unwrap_surface_value", side_effect=_identity), \
                     patch_ops("_coerce_i64", side_effect=lambda value, *, context: f"{context}:{value}"), \
                     patch.object(_ops._pto, "MteL1L0aOp") as ca_op, \
                     patch.object(_ops._pto, "MteL1L0bOp") as cb_op:
                    _ops.mte_l1_l0a(
                        source, destination_a, **controls, transpose=0
                    )
                    _ops.mte_l1_l0b(
                        source, destination_b, **controls, transpose=1
                    )

                ca_op.assert_called_once_with(
                    source,
                    destination_a,
                    m_start="mte_l1_l0a m_start:0",
                    k_start="mte_l1_l0a k_start:4",
                    m_step="mte_l1_l0a m_step:16",
                    k_step="mte_l1_l0a k_step:4",
                    src_stride="mte_l1_l0a src_stride:16",
                    dst_stride="mte_l1_l0a dst_stride:16",
                    transpose=False,
                )
                cb_op.assert_called_once_with(
                    source,
                    destination_b,
                    m_start="mte_l1_l0b m_start:0",
                    k_start="mte_l1_l0b k_start:4",
                    m_step="mte_l1_l0b m_step:16",
                    k_step="mte_l1_l0b k_step:4",
                    src_stride="mte_l1_l0b src_stride:16",
                    dst_stride="mte_l1_l0b dst_stride:16",
                    transpose=True,
                )

    def test_mte_l1_l0_explicit_controls_normalize_legacy_transpose(self):
        controls = {
            "m_start": 0,
            "k_start": 4,
            "m_step": 16,
            "k_step": 4,
            "src_stride": 8,
            "dst_stride": 16,
        }

        source = object()
        destination = object()
        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("_coerce_i64", side_effect=lambda value, *, context: value), \
             patch.object(_ops._pto, "MteL1L0aOp") as ca_op:
            _ops.mte_l1_l0a(source, destination, **controls, transpose=1)
        self.assertEqual(
            ca_op.call_args.kwargs,
            {**controls, "transpose": True},
        )

        with self.assertRaisesRegex(
            TypeError, "mte_l1_l0a transpose expects bool or integer 0/1"
        ):
            _ops.mte_l1_l0a(source, destination, **controls, transpose=2)
        with self.assertRaisesRegex(
            TypeError, "mte_l1_l0b transpose expects bool or integer 0/1"
        ):
            _ops.mte_l1_l0b(object(), object(), **controls, transpose="true")

    def test_mte_l1_l0_shape_transpose_accepts_legacy_zero_one(self):
        source = object()
        destination = object()
        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("_coerce_i64", side_effect=lambda value, *, context: value), \
             patch.object(_ops._pto, "MteL1L0aOp") as ca_op:
            _ops.mte_l1_l0a(source, destination, 16, 32, transpose=0)
        self.assertEqual(
            ca_op.call_args.kwargs,
            {"m": 16, "k": 32, "start_row": 0, "start_col": 0, "transpose": False},
        )

        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("_coerce_i64", side_effect=lambda value, *, context: value), \
             patch.object(_ops._pto, "MteL1L0bOp") as cb_op:
            _ops.mte_l1_l0b(source, destination, 32, 16, transpose=1)
        self.assertEqual(
            cb_op.call_args.kwargs,
            {"k": 32, "n": 16, "start_row": 0, "start_col": 0, "transpose": True},
        )

    def test_mte_l1_l0_explicit_controls_defer_fp4_type_validation_to_vpto(self):
        controls = {
            "m_start": 0,
            "k_start": 0,
            "m_step": 1,
            "k_step": 1,
            "src_stride": 1,
            "dst_stride": 1,
        }
        with make_context():
            source = SimpleNamespace(type=Type.parse("!pto.ptr<i8, l1>"))
            destination = SimpleNamespace(
                type=Type.parse("!pto.ptr<!pto.f4E2M1x2, l0a>")
            )
            with patch_ops("unwrap_surface_value", side_effect=_identity), \
                 patch_ops("_coerce_i64", side_effect=lambda value, *, context: value), \
                 patch.object(_ops._pto, "MteL1L0aOp") as ca_op:
                _ops.mte_l1_l0a(source, destination, **controls, transpose=False)
            ca_op.assert_called_once_with(
                source,
                destination,
                m_start=0,
                k_start=0,
                m_step=1,
                k_step=1,
                src_stride=1,
                dst_stride=1,
                transpose=False,
            )

    def test_mte_l1_l0_legacy_forms_preserve_keyword_compatibility(self):
        source = object()
        destination = object()

        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("_coerce_i64", side_effect=lambda value, *, context: f"{context}:{value}"):
            ca_op = MagicMock()
            with patch.object(_ops._pto, "MteL1L0aOp", ca_op):
                _ops.mte_l1_l0a(
                    source,
                    destination,
                    m=128,
                    k=64,
                    start_row=2,
                    start_col=4,
                    transpose=True,
                )
            self.assertEqual(
                ca_op.call_args.args,
                (
                    source,
                    destination,
                ),
            )
            self.assertEqual(
                ca_op.call_args.kwargs,
                {
                    "m": "mte_l1_l0a m:128",
                    "k": "mte_l1_l0a k:64",
                    "start_row": "mte_l1_l0a start_row:2",
                    "start_col": "mte_l1_l0a start_col:4",
                    "transpose": True,
                },
            )

            cb_op = MagicMock()
            with patch.object(_ops._pto, "MteL1L0bOp", cb_op):
                _ops.mte_l1_l0b(
                    source,
                    destination,
                    k=64,
                    n=128,
                    start_row=4,
                    start_col=2,
                    transpose=True,
                )
            self.assertEqual(
                cb_op.call_args.args,
                (
                    source,
                    destination,
                ),
            )
            self.assertEqual(
                cb_op.call_args.kwargs,
                {
                    "k": "mte_l1_l0b k:64",
                    "n": "mte_l1_l0b n:128",
                    "start_row": "mte_l1_l0b start_row:4",
                    "start_col": "mte_l1_l0b start_col:2",
                    "transpose": True,
                },
            )

    def test_mte_l1_l0_explicit_controls_reject_ambiguous_forms(self):
        source = object()
        destination = object()
        complete_controls = {
            "m_start": 3,
            "k_start": 5,
            "m_step": 16,
            "k_step": 2,
            "src_stride": 8,
            "dst_stride": 2,
        }

        invalid_cases = [
            lambda: _ops.mte_l1_l0a(source, destination, m_start=3),
            lambda: _ops.mte_l1_l0b(source, destination, m_start=3),
            lambda: _ops.mte_l1_l0a(source, destination, m=128, k=64, **complete_controls),
            lambda: _ops.mte_l1_l0b(source, destination, k=64, n=128, **complete_controls),
            lambda: _ops.mte_l1_l0a(source, destination, start_row=1, **complete_controls),
            lambda: _ops.mte_l1_l0b(source, destination, start_col=1, **complete_controls),
        ]
        for invalid_call in invalid_cases:
            with self.subTest(call=invalid_call):
                with self.assertRaises(TypeError):
                    invalid_call()

        self.assertFalse(hasattr(pto, "load_cbuf_to_ca"))
        self.assertFalse(hasattr(pto, "load_cbuf_to_cb"))

    def test_mad_option_wrappers_dispatch_to_generated_ops(self):
        lhs = object()
        rhs = object()
        dst = object()
        bias = object()
        mad_options = {
            "unit_flag_mode": "unit_flag_attr",
            "disable_gemv": True,
            "sat_mode": "sat_attr",
            "tf32_mode": "tf32_attr",
            "n_dir": True,
        }
        mad_mx_options = {
            "unit_flag_mode": "unit_flag_attr",
            "disable_gemv": True,
            "sat_mode": "sat_attr",
            "n_dir": True,
        }

        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("_coerce_i64", side_effect=lambda value, *, context: f"i64:{value}"), \
             patch_ops("_mad_options", return_value=mad_options) as normalize_mad, \
             patch.object(_ops._pto, "MadOp", MagicMock()) as mad_op:
            _ops.mad(
                lhs,
                rhs,
                dst,
                1,
                2,
                3,
                unit_flag=pto.MadUnitFlagMode.CHECK_ONLY,
                disable_gemv=True,
                sat=pto.SatMode.OFF,
                tf32_mode=pto.Tf32Mode.ROUND_EVEN,
                n_dir=True,
            )
        normalize_mad.assert_called_once_with(
            unit_flag=pto.MadUnitFlagMode.CHECK_ONLY,
            disable_gemv=True,
            sat=pto.SatMode.OFF,
            tf32_mode=pto.Tf32Mode.ROUND_EVEN,
            n_dir=True,
        )
        self.assertEqual(mad_op.call_args.args, (lhs, rhs, dst, "i64:1", "i64:2", "i64:3"))
        self.assertEqual(mad_op.call_args.kwargs, mad_options)

        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("_coerce_i64", side_effect=lambda value, *, context: f"i64:{value}"), \
             patch_ops("_mad_mx_options", return_value=mad_mx_options) as normalize_mx, \
             patch.object(_ops._pto, "MadMxBiasOp", MagicMock()) as mad_mx_bias_op:
            _ops.mad_mx_bias(
                lhs,
                rhs,
                dst,
                bias,
                1,
                2,
                3,
                unit_flag=pto.MadUnitFlagMode.CHECK_AND_SET,
                disable_gemv=True,
                sat=pto.SatMode.ON,
                n_dir=True,
            )
        normalize_mx.assert_called_once_with(
            unit_flag=pto.MadUnitFlagMode.CHECK_AND_SET,
            disable_gemv=True,
            sat=pto.SatMode.ON,
            n_dir=True,
        )
        self.assertEqual(mad_mx_bias_op.call_args.args, (lhs, rhs, dst, bias, "i64:1", "i64:2", "i64:3"))
        self.assertEqual(mad_mx_bias_op.call_args.kwargs, mad_mx_options)

    def test_mte_l0c_ub_dst_mode_accepts_enum_like_subblock_value(self):
        enum_like = SimpleNamespace(value=1)
        with patch_ops("_acc_store_ub_dst_mode_attr", return_value="single_attr"), \
             patch_ops("_coerce_i64", side_effect=lambda value, *, context: f"i64:{value}"):
            attr, sub_blockid = _ops._mte_l0c_ub_dst_mode(enum_like)
        self.assertEqual(attr, "single_attr")
        self.assertEqual(sub_blockid, "i64:1")

    def test_mte_l0c_ub_dst_mode_accepts_split_enum(self):
        with patch_ops("_acc_store_ub_dst_mode_attr", side_effect=lambda mode: f"{mode}_attr"):
            attr, sub_blockid = _ops._mte_l0c_ub_dst_mode(split=pto.SplitMode.N)
        self.assertEqual(attr, "split_n_attr")
        self.assertIsNone(sub_blockid)

    def test_cube_sat_modes_map_to_backend_tokens(self):
        with patch_ops("Attribute") as attr:
            attr.parse.side_effect = lambda text: text
            self.assertEqual(_ops._mad_sat_attr(pto.SatMode.ON), "#pto<mad_sat_mode sat>")
            self.assertEqual(_ops._mad_sat_attr(pto.SatMode.OFF), "#pto<mad_sat_mode nosat>")
            self.assertEqual(_ops._acc_store_sat_attr(pto.SatMode.PRESERVE_NAN), "#pto<acc_store_sat_mode sat_preserve_nan>")

    def test_acc_store_no_convert_skips_payload_kind_check(self):
        payload = object()
        with patch_ops("Attribute") as attr:
            attr.parse.side_effect = lambda text: text
            value, mode = _ops._acc_store_pre_quant((payload, "no_convert"))

        self.assertIs(value, payload)
        self.assertEqual(mode, "#pto<quant_pre_mode no_convert>")

    def test_tile_selection_surface_exposes_optional_tmp(self):
        for func, expected in [
            (_ops.tsel, ["mask", "src0", "src1", "dst", "tmp"]),
            (_ops.tsels, ["mask", "src", "scalar", "dst", "tmp"]),
            (pto.tile.sel, ["mask", "src0", "src1", "dst", "tmp"]),
            (pto.tile.sels, ["mask", "src", "scalar", "dst", "tmp"]),
        ]:
            with self.subTest(func=func):
                signature = inspect.signature(func)
                self.assertEqual(list(signature.parameters.keys()), expected)
                self.assertEqual(signature.parameters["tmp"].kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIsNone(signature.parameters["tmp"].default)

    def test_tile_remainder_surface_exposes_optional_tmp(self):
        for func, expected in [
            (pto.tile.rem, ["src0", "src1", "dst", "tmp", "precision"]),
            (pto.tile.rems, ["src", "scalar", "dst", "tmp"]),
        ]:
            with self.subTest(func=func):
                signature = inspect.signature(func)
                self.assertEqual(list(signature.parameters), expected)
                self.assertEqual(signature.parameters["tmp"].kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIsNone(signature.parameters["tmp"].default)

    def test_tile_prelu_surface_exposes_explicit_tmp(self):
        self.assertEqual(
            list(inspect.signature(pto.tile.prelu).parameters),
            ["src0", "src1", "dst", "tmp"],
        )
        self.assertEqual(
            inspect.signature(pto.tile.prelu).parameters["tmp"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )

        src0, src1, dst, tmp = object(), object(), object(), object()
        with patch_ops("tprelu") as tprelu_op:
            pto.tile.prelu(src0, src1, dst, tmp=tmp)
        tprelu_op.assert_called_once_with(src0, src1, dst, tmp=tmp)

        with patch_ops("tprelu") as tprelu_op:
            pto.tile.prelu(src0, src1, dst)
        tprelu_op.assert_called_once_with(src0, src1, dst, tmp=None)

    def test_tile_precision_surfaces_use_common_precision_parameter(self):
        self.assertIs(pto.Precision, pto.DivPrecision)
        self.assertIs(pto.Precision, pto.RemPrecision)
        legacy_keywords = {
            pto.tile.div: "div_precision",
            pto.tile.divs: "div_precision",
            pto.tile.exp: "exp_precision",
            pto.tile.log: "log_precision",
            pto.tile.sqrt: "sqrt_precision",
            pto.tile.rsqrt: "rsqrt_precision",
            pto.tile.recip: "recip_precision",
            pto.tile.rowexpanddiv: "div_precision",
            pto.tile.colexpanddiv: "div_precision",
        }
        for func, legacy_keyword in legacy_keywords.items():
            with self.subTest(func=func):
                self.assertIn("precision", inspect.signature(func).parameters)
                self.assertIn(legacy_keyword, inspect.signature(func).parameters)
                self.assertIs(
                    inspect.signature(func).parameters[legacy_keyword].default,
                    _ops._UNSET,
                )

    def test_legacy_precision_keyword_warns_and_forwards(self):
        src0, src1, dst = object(), object(), object()
        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("_normalize_enum_attr", return_value="high_precision") as normalize, \
             patch.object(_ops._pto, "tdiv") as tdiv_op:
            with self.assertWarns(PTODSLDeprecationWarning):
                pto.tile.div(src0, src1, dst, div_precision="high_precision")

        normalize.assert_called_once_with(
            "high_precision",
            enum_cls=_ops._pto.DivPrecision,
            attr_cls=_ops._pto.DivPrecisionAttr,
            context="tdiv precision",
        )
        tdiv_op.assert_called_once_with(src0, src1, dst, precision_type="high_precision")

    def test_legacy_precision_keyword_cannot_be_combined_with_precision(self):
        with self.assertRaisesRegex(TypeError, "either precision or div_precision"):
            pto.tile.div(object(), object(), object(), precision="default", div_precision="default")

    def test_tprelu_is_exported(self):
        self.assertIn("tprelu", _ops.__all__)

    def test_trandom_requires_i32_runtime_scalars(self):
        operands = [object()] * 6
        with patch_ops("unwrap_surface_value", side_effect=_identity):
            with self.assertRaisesRegex(TypeError, "key0 must be a signless i32 runtime scalar"):
                _ops.trandom(*operands, object())

        with make_context():
            i16 = SimpleNamespace(type=IntegerType.get_signless(16))
            f32 = SimpleNamespace(type=F32Type.get())
            i32 = IntegerType.get_signless(32)
            si32 = IntegerType.get_signed(32)
            ui32 = IntegerType.get_unsigned(32)
            valid_operands = [SimpleNamespace(type=i32) for _ in range(6)]
            with patch_ops("unwrap_surface_value", side_effect=_identity):
                for name, invalid_operand in (
                    ("key0", i16),
                    ("key1", f32),
                    ("counter0", SimpleNamespace(type=si32)),
                    ("counter1", SimpleNamespace(type=ui32)),
                ):
                    operands = list(valid_operands)
                    operand_index = ("key0", "key1", "counter0", "counter1").index(name)
                    operands[operand_index] = invalid_operand
                    with self.subTest(name=name):
                        with self.assertRaisesRegex(
                            TypeError, f"{name} must be a signless i32 runtime scalar"
                        ):
                            _ops.trandom(*operands, object())

    def test_trandom_forwards_valid_i32_runtime_scalars(self):
        with make_context():
            scalar = SimpleNamespace(type=IntegerType.get_signless(32))
        dst = object()
        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch.object(_ops._pto, "TRandomOp") as trandom_op:
            _ops.trandom(scalar, scalar, scalar, scalar, scalar, scalar, dst, rounds=7)
        trandom_op.assert_called_once_with(
            scalar, scalar, scalar, scalar, scalar, scalar, dst, rounds=7
        )

    def test_tile_fmod_surfaces_expose_public_precision_contract(self):
        self.assertEqual(
            list(inspect.signature(pto.tile.fmod).parameters),
            ["src0", "src1", "dst", "precision"],
        )
        self.assertEqual(
            list(inspect.signature(pto.tile.fmods).parameters),
            ["src", "scalar", "dst"],
        )

        src0, src1, dst = object(), object(), object()
        with patch_ops("tfmod") as tfmod_op:
            pto.tile.fmod(src0, src1, dst, precision="high_precision")
        tfmod_op.assert_called_once_with(src0, src1, dst, precision="high_precision")

        src, scalar, dst = object(), 3.0, object()
        with patch_ops("tfmods") as tfmods_op:
            pto.tile.fmods(src, scalar, dst)
        tfmods_op.assert_called_once_with(src, scalar, dst)

    def test_tile_remainder_wrappers_forward_tmp_scalar_and_precision(self):
        src0 = object()
        src1 = object()
        dst = object()
        tmp = object()
        scalar = 3.0
        with patch_ops("trem") as trem_op:
            pto.tile.rem(src0, src1, dst, tmp=tmp, precision="high_precision")
        trem_op.assert_called_once_with(src0, src1, dst, tmp=tmp, precision="high_precision")

        with patch_ops("trems") as trems_op:
            pto.tile.rems(src0, scalar, dst)
        trems_op.assert_called_once_with(src0, scalar, dst, tmp=None)

    def test_tile_remainder_wrappers_omit_implicit_tmp(self):
        src0, src1, dst, scalar = object(), object(), object(), object()
        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch.object(_ops._pto, "trem") as trem_op, \
             patch_ops("_normalize_enum_attr", return_value="default"):
            _ops.trem(src0, src1, dst)
        trem_op.assert_called_once_with(src0, src1, dst, precision_type="default")

        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch.object(_ops._pto, "trems") as trems_op, \
             patch_ops("_coerce_tile_scalar_operand", return_value=scalar):
            _ops.trems(src0, scalar, dst)
        trems_op.assert_called_once_with(src0, scalar, dst)

        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch.object(_ops._pto, "TPReluOp") as tprelu_op:
            _ops.tprelu(src0, src1, dst)
        tprelu_op.assert_called_once_with(src0, src1, dst)

    def test_tile_selection_wrappers_use_explicit_tmp_or_synthesize_one(self):
        mask = object()
        src0 = object()
        src1 = object()
        src = object()
        dst = object()
        tmp = object()
        scalar = object()
        coerced_scalar = object()
        synthesized_tmp = object()

        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("_coerce_tile_scalar_operand", return_value=coerced_scalar):
            with patch_ops("_resolve_selection_tmp", return_value=synthesized_tmp) as resolve_tmp, \
                 patch.object(_ops._pto, "tsel") as tsel_op:
                _ops.tsel(mask, src0, src1, dst)
            resolve_tmp.assert_called_once_with(dst, None, context="tsel")
            tsel_op.assert_called_once_with(mask, src0, src1, dst, tmp=synthesized_tmp)

            with patch_ops("_resolve_selection_tmp", side_effect=AssertionError("should not synthesize")), \
                 patch.object(_ops._pto, "tsel") as tsel_op:
                _ops.tsel(mask, src0, src1, dst, tmp=tmp)
            tsel_op.assert_called_once_with(mask, src0, src1, dst, tmp=tmp)

            with patch_ops("_resolve_selection_tmp", return_value=synthesized_tmp) as resolve_tmp, \
                 patch.object(_ops._pto, "tsels") as tsels_op:
                _ops.tsels(mask, src, scalar, dst)
            resolve_tmp.assert_called_once_with(dst, None, context="tsels")
            tsels_op.assert_called_once_with(mask, src, coerced_scalar, dst, tmp=synthesized_tmp)

            with patch_ops("_resolve_selection_tmp", side_effect=AssertionError("should not synthesize")), \
                 patch.object(_ops._pto, "tsels") as tsels_op:
                _ops.tsels(mask, src, scalar, dst, tmp=tmp)
            tsels_op.assert_called_once_with(mask, src, coerced_scalar, dst, tmp=tmp)

    def test_tile_row_reductions_expose_optional_tmp_and_synthesize_one(self):
        src = SimpleNamespace(type="src_ty")
        dst = object()
        tmp = object()
        synthesized_tmp = object()

        row_cases = [
            ("rowsum", "trowsum"),
            ("rowmax", "trowmax"),
            ("rowmin", "trowmin"),
            ("rowprod", "trowprod"),
            ("rowargmax", "trowargmax"),
            ("rowargmin", "trowargmin"),
        ]

        for name, low_level_name in row_cases:
            with self.subTest(func=name):
                signature = inspect.signature(getattr(pto.tile, name))
                self.assertEqual(list(signature.parameters.keys()), ["src", "dst", "tmp"])
                self.assertEqual(signature.parameters["tmp"].kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIsNone(signature.parameters["tmp"].default)

                with patch_ops("unwrap_surface_value", side_effect=_identity), \
                     patch_ops("alloc_tile", return_value=synthesized_tmp) as alloc_tile, \
                     patch_ops(low_level_name) as low_level_op:
                    getattr(pto.tile, name)(src, dst)
                alloc_tile.assert_called_once_with(tile_type="src_ty")
                low_level_op.assert_called_once_with(src, synthesized_tmp, dst)

                with patch_ops("unwrap_surface_value", side_effect=_identity), \
                     patch_ops("alloc_tile", side_effect=AssertionError("should not synthesize")), \
                     patch_ops(low_level_name) as low_level_op:
                    getattr(pto.tile, name)(src, dst, tmp=tmp)
                low_level_op.assert_called_once_with(src, tmp, dst)

    def test_tile_transpose_wrapper_uses_tmp_keyword_builder(self):
        src = object()
        tmp = object()
        dst = object()

        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch.object(_ops._pto, "TTransOp") as ttrans_op:
            pto.tile.transpose(src, tmp, dst)

        ttrans_op.assert_called_once_with(src, dst, tmp=tmp)

    def test_tile_sort_gather_wrappers_call_low_level_ops(self):
        src = object()
        idx = object()
        dst = object()
        tmp = object()
        block_len = object()

        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch.object(_ops._pto, "tsort32") as tsort32_op:
            pto.tile.sort32(src, idx, dst)
        self.assertEqual(tsort32_op.call_args.args, (src, idx, dst))
        self.assertEqual(tsort32_op.call_args.kwargs, {"tmp": None})

        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch.object(_ops._pto, "tsort32") as tsort32_op:
            pto.tile.sort32(src, idx, dst, tmp=tmp)
        self.assertEqual(tsort32_op.call_args.args, (src, idx, dst))
        self.assertEqual(tsort32_op.call_args.kwargs, {"tmp": tmp})

        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch.object(_ops._pto, "tmrgsort") as tmrgsort_op:
            pto.tile.mrgsort(src, dst, block_len)
        self.assertEqual(tmrgsort_op.call_args.args, ([src], [dst]))
        self.assertEqual(
            tmrgsort_op.call_args.kwargs,
            {"block_len": block_len, "tmp": None, "excuted": None, "exhausted": None},
        )

        parsed_pattern = object()
        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("_tile_mask_pattern_attr", return_value=parsed_pattern) as mask_attr, \
             patch.object(_ops._pto, "tgather") as tgather_op:
            pto.tile.gather(src, dst, mask_pattern="P0101", axis="row")
        mask_attr.assert_called_once_with("P0101")
        self.assertEqual(tgather_op.call_args.args, (src, dst))
        self.assertEqual(tgather_op.call_args.kwargs["mask_pattern"], parsed_pattern)
        self.assertEqual(tgather_op.call_args.kwargs["axis"], "row")

        with self.assertRaisesRegex(ValueError, "unsupported tile mask pattern"):
            pto.tile.gather(src, dst, mask_pattern="PAT_ALL", axis="row")

    def test_tile_mov_accepts_acc_to_vec_mode(self):
        src = object()
        dst = object()
        parsed_mode = object()

        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch.object(_ops.Attribute, "parse", return_value=parsed_mode) as parse_attr, \
             patch.object(_ops._pto, "TMovOp") as tmov_op:
            pto.tile.mov(src, dst, mode="split_n")

        parse_attr.assert_called_once_with("#pto<acc_to_vec_mode dual_mode_split_n>")
        self.assertEqual(tmov_op.call_args.args, (None, src, dst))
        self.assertEqual(tmov_op.call_args.kwargs, {"accToVecMode": parsed_mode})
    def test_tile_extract_dispatches_row_and_col_indices(self):
        src = object()
        dst = object()

        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops(
                 "_coerce_index",
                 side_effect=lambda value, *, context: f"idx:{context}:{value}") as coerce_index, \
             patch.object(_ops._pto, "TExtractOp") as textract_op:
            pto.tile.extract(src, dst, 7, 11)

        self.assertEqual(
            textract_op.call_args.args,
            (src, "idx:textract(index_row):7", "idx:textract(index_col):11", dst),
        )
        self.assertEqual(coerce_index.call_count, 2)

    def test_tile_fp_forms_dispatch_through_unified_ops(self):
        src = object()
        dst = object()
        fp = object()

        with patch_ops("unwrap_surface_value", side_effect=_identity), \
             patch_ops("_is_partition_tensor_view", return_value=True), \
             patch_ops("_coerce_index", side_effect=lambda value, *, context: value), \
             patch.object(_ops._pto, "TStoreOp") as tstore_op, \
             patch.object(_ops._pto, "TMovOp") as tmov_op, \
             patch.object(_ops._pto, "TExtractOp") as textract_op, \
             patch.object(_ops._pto, "TInsertOp") as tinsert_op:
            pto.tile.store(src, dst, fp=fp)
            pto.tile.mov(src, dst, fp=fp)
            pto.tile.extract(src, dst, 3, 5, fp=fp)
            pto.tile.insert(src, dst, 3, 5, fp=fp)

        tstore_op.assert_called_once_with(None, src, dst, fp=fp)
        tmov_op.assert_called_once_with(None, src, dst, fp=fp)
        textract_op.assert_called_once_with(src, 3, 5, dst, fp=fp)
        tinsert_op.assert_called_once_with(src, 3, 5, dst, fp=fp)

    def test_sync_event_id_rejects_out_of_range_static_values(self):
        cases = [
            (_ops.set_flag, ("MTE2", "V"), {"event_id": 8}, "set_flag(..., event_id=...)"),
            (_ops.wait_flag, ("MTE2", "V"), {"event_id": -1}, "wait_flag(..., event_id=...)"),
            (_ops.set_cross_block, (pto.Pipe.FIX, 16), {}, "set_cross_block(..., event_id=...)", "[0, 15]"),
            (_ops.wait_cross_block, (pto.Pipe.FIX, -1), {}, "wait_cross_block(..., event_id=...)", "[0, 15]"),
            (_ops.set_intra_block, (pto.Pipe.FIX, 32), {}, "set_intra_block(..., event_id=...)", "[0, 31]"),
            (_ops.set_intra_block, (pto.Pipe.MTE3, 32), {}, "set_intra_block(..., event_id=...)", "[0, 31]"),
            (_ops.wait_intra_block, (pto.Pipe.V, -2), {}, "wait_intra_block(..., event_id=...)", "[0, 31]"),
            (_ops.wait_intra_block, (pto.Pipe.FIX, 32), {}, "wait_intra_block(..., event_id=...)", "[0, 31]"),
            (_ops.wait_intra_block, (pto.Pipe.MTE3, 32), {}, "wait_intra_block(..., event_id=...)", "[0, 31]"),
        ]

        with patch.object(_ops._pto, "set_flag") as set_flag_op, \
             patch.object(_ops._pto, "set_flag_dyn") as set_flag_dyn_op, \
             patch.object(_ops._pto, "wait_flag") as wait_flag_op, \
             patch.object(_ops._pto, "wait_flag_dyn") as wait_flag_dyn_op, \
             patch.object(_ops._pto, "sync_set") as sync_set_op, \
             patch.object(_ops._pto, "sync_wait") as sync_wait_op:
            for case in cases:
                func, args, kwargs, context, *expected_range = case
                with self.subTest(func=func.__name__, event_id=kwargs.get("event_id", args[-1])):
                    with self.assertRaises(ValueError) as exc:
                        func(*args, **kwargs)
                    message = str(exc.exception)
                    self.assertIn(context, message)
                    self.assertIn(expected_range[0] if expected_range else "[0, 7]", message)

        set_flag_op.assert_not_called()
        set_flag_dyn_op.assert_not_called()
        wait_flag_op.assert_not_called()
        wait_flag_dyn_op.assert_not_called()
        sync_set_op.assert_not_called()
        sync_wait_op.assert_not_called()

    def test_sync_facades_reject_illegal_pipe_endpoints(self):
        cases = [
            (
                _ops.set_cross_block,
                ("M", 0),
                "set_cross_block(pipe, event_id)",
                "<PIPE_FIX>, <PIPE_MTE1>, <PIPE_MTE2>, <PIPE_MTE3>, <PIPE_V>",
                "<PIPE_M>",
            ),
            (
                _ops.wait_cross_block,
                (pto.Pipe.M, 0),
                "wait_cross_block(pipe, event_id)",
                "<PIPE_FIX>, <PIPE_MTE1>, <PIPE_MTE2>, <PIPE_MTE3>, <PIPE_V>",
                "<PIPE_M>",
            ),
            (
                _ops.set_intra_block,
                ("M", 0),
                "set_intra_block(pipe, event_id)",
                "<PIPE_FIX>, <PIPE_MTE1>, <PIPE_MTE2>, <PIPE_MTE3>, <PIPE_V>",
                "<PIPE_M>",
            ),
            (
                _ops.wait_intra_block,
                (pto.Pipe.M, 0),
                "wait_intra_block(pipe, event_id)",
                "<PIPE_FIX>, <PIPE_MTE1>, <PIPE_MTE2>, <PIPE_MTE3>, <PIPE_V>",
                "<PIPE_M>",
            ),
        ]

        with patch.object(_ops._pto, "sync_set") as sync_set_op, \
             patch.object(_ops._pto, "sync_wait") as sync_wait_op:
            for func, args, context, expected, actual in cases:
                with self.subTest(func=func.__name__, pipe=args[0]):
                    with self.assertRaises(ValueError) as exc:
                        func(*args)
                    message = str(exc.exception)
                    self.assertIn(context, message)
                    self.assertIn(expected, message)
                    self.assertIn(actual, message)

        sync_set_op.assert_not_called()
        sync_wait_op.assert_not_called()

    def test_intra_sync_mixed_writeback_event_ranges(self):
        dynamic_event = object()
        dynamic_event_operand = object()
        with patch_ops("_pipe_attr", side_effect=lambda pipe: f"pipe:{pipe}") as pipe_attr, \
             patch_ops("unwrap_surface_value", return_value=dynamic_event_operand) as unwrap_surface_value, \
             patch.object(_ops._pto, "set_intra_block") as set_intra_block_op, \
             patch.object(_ops._pto, "wait_intra_block") as wait_intra_block_op:
            _ops.set_intra_block(pto.Pipe.FIX, 31)
            _ops.set_intra_block(pto.Pipe.MTE3, 31)
            _ops.wait_intra_block(pto.Pipe.FIX, 16)
            _ops.wait_intra_block(pto.Pipe.V, 31)
            _ops.wait_intra_block(pto.Pipe.MTE3, 31)
            _ops.wait_intra_block(pto.Pipe.MTE3, dynamic_event)

        self.assertEqual(pipe_attr.call_count, 6)
        self.assertEqual(set_intra_block_op.call_count, 2)
        wait_intra_block_op.assert_has_calls([
            call(f"pipe:{pto.Pipe.FIX}", 16),
            call(f"pipe:{pto.Pipe.V}", 31),
            call(f"pipe:{pto.Pipe.MTE3}", 31),
            call(f"pipe:{pto.Pipe.MTE3}", dynamic_event_operand),
        ])
        unwrap_surface_value.assert_called_once_with(dynamic_event)

    def test_pipe_namespace_and_buffer_helpers_are_exposed(self):
        names = [
            "c2v", "v2c", "bidirectional",
        ]
        for name in names:
            with self.subTest(name=name):
                self.assertTrue(hasattr(pto.pipe, name), name)
        old_names = [
            "c2v_global", "v2c_global",
            "c2v_local", "v2c_local", "bidirectional_local",
        ]
        for name in old_names:
            with self.subTest(name=name):
                self.assertFalse(hasattr(pto.pipe, name), name)

        for name in ["reserve_buffer", "import_reserved_buffer"]:
            with self.subTest(name=name):
                self.assertTrue(hasattr(pto, name), name)

        self.assertTrue(hasattr(pto, "gm_ptr"), "gm_ptr")

    def test_global_pipe_methods_dispatch_to_matching_frontend_ops(self):
        alloc_entry = object()
        pop_entry = object()

        with make_context():
            gm_slot_type = _pipe_namespace._pto.TensorViewType.get([16, 16], F32Type.get())
        gm_slot = SimpleNamespace(type=gm_slot_type)

        with patch.object(_pipe_namespace, "unwrap_surface_value", side_effect=_identity), \
             patch.object(_pipe_namespace, "wrap_surface_value", side_effect=_identity), \
             patch.object(_pipe_namespace, "_infer_unambiguous_global_slot_size", return_value=1024):
            pipe = pto.pipe.c2v(
                gm_slot_tensor=gm_slot,
                id=7,
                nosplit=True,
            )

        self.assertEqual(pipe.id, 7)
        self.assertEqual(pipe.slot_size, 1024)
        self.assertEqual(pipe.entry_type, gm_slot_type)

        with patch.object(_pipe_namespace._pto, "AicInitializePipeOp") as aic_init, \
             patch.object(_pipe_namespace._pto, "AivInitializePipeOp") as aiv_init, \
             patch.object(_pipe_namespace._pto, "TAllocToAivOp", return_value=SimpleNamespace(result=alloc_entry)) as alloc_op, \
             patch.object(_pipe_namespace._pto, "TPushToAivOp") as push_op, \
             patch.object(_pipe_namespace._pto, "TPopFromAicOp", return_value=SimpleNamespace(result=pop_entry)) as pop_op, \
             patch.object(_pipe_namespace._pto, "TFreeFromAicOp") as free_op, \
             patch.object(_pipe_namespace, "wrap_surface_value", side_effect=_identity), \
             patch.object(_pipe_namespace, "unwrap_surface_value", side_effect=_identity):
            pipe.init_cube()
            pipe.init_simd()
            alloc_result = pipe.alloc()
            pipe.push(alloc_result, split=1)
            pop_result = pipe.pop()
            pipe.free(pop_result, split=2)

        expected_init_kwargs = {
            "id": 7,
            "nosplit": True,
            "gm_slot_tensor": gm_slot,
        }
        aic_init.assert_called_once_with(1, 1024, **expected_init_kwargs)
        aiv_init.assert_called_once_with(1, 1024, **expected_init_kwargs)
        alloc_op.assert_called_once_with(gm_slot_type, 0, id=7)
        push_op.assert_called_once_with(alloc_entry, 1, id=7)
        pop_op.assert_called_once_with(gm_slot_type, 0, id=7)
        free_op.assert_called_once_with(2, entry=pop_entry, id=7)
        self.assertIs(alloc_result, alloc_entry)
        self.assertIs(pop_result, pop_entry)

    def test_local_pipe_constructors_route_consumer_buffers(self):
        c2v_buf = object()
        v2c_buf = object()
        gm_slot_buffer = object()

        with patch.object(_pipe_namespace, "unwrap_surface_value", side_effect=_identity), \
             patch.object(_pipe_namespace, "wrap_surface_value", side_effect=_identity):
            c2v = pto.pipe.c2v(
                slot_size=1024,
                consumer_buf=c2v_buf,
                gm_slot_buffer=gm_slot_buffer,
                id=3,
                local_slot_num=2,
                nosplit=True,
            )
            v2c = pto.pipe.v2c(
                slot_size=2048,
                consumer_buf=v2c_buf,
                gm_slot_buffer=gm_slot_buffer,
                id=4,
                local_slot_num=5,
            )
            bidi = pto.pipe.bidirectional(
                slot_size=4096,
                c2v_consumer_buf=c2v_buf,
                v2c_consumer_buf=v2c_buf,
                gm_slot_buffer=gm_slot_buffer,
                id=5,
            )

        self.assertIsNone(c2v.entry_type)
        self.assertIsNone(v2c.entry_type)
        self.assertIsNone(bidi.entry_type)
        self.assertEqual(bidi.c2v.id, 5)
        self.assertEqual(bidi.v2c.id, 5)

        with patch.object(_pipe_namespace._pto, "AicInitializePipeOp") as aic_init, \
             patch.object(_pipe_namespace._pto, "AivInitializePipeOp") as aiv_init:
            c2v.init_cube()
            c2v.init_simd()
            v2c.init_cube()
            bidi.init_simd()

        self.assertEqual(aic_init.call_args_list[0].args, (1, 1024))
        self.assertEqual(aic_init.call_args_list[0].kwargs, {
            "id": 3,
            "local_slot_num": 2,
            "nosplit": True,
            "gm_slot_buffer": gm_slot_buffer,
            "c2v_consumer_buf": c2v_buf,
        })
        self.assertEqual(aiv_init.call_args_list[0].args, (1, 1024))
        self.assertEqual(aiv_init.call_args_list[0].kwargs, {
            "id": 3,
            "local_slot_num": 2,
            "nosplit": True,
            "gm_slot_buffer": gm_slot_buffer,
            "c2v_consumer_buf": c2v_buf,
        })
        self.assertEqual(aic_init.call_args_list[1].args, (2, 2048))
        self.assertEqual(aic_init.call_args_list[1].kwargs, {
            "id": 4,
            "local_slot_num": 5,
            "gm_slot_buffer": gm_slot_buffer,
            "v2c_consumer_buf": v2c_buf,
        })
        self.assertEqual(aiv_init.call_args_list[1].args, (3, 4096))
        self.assertEqual(aiv_init.call_args_list[1].kwargs, {
            "id": 5,
            "gm_slot_buffer": gm_slot_buffer,
            "c2v_consumer_buf": c2v_buf,
            "v2c_consumer_buf": v2c_buf,
        })

    def test_local_pipe_constructors_still_require_consumer_buffers(self):
        with self.assertRaises(TypeError) as c2v_exc:
            pto.pipe.c2v(slot_size=1024, id=3)
        self.assertIn("requires consumer_buf for local pipes", str(c2v_exc.exception))

        with self.assertRaises(TypeError) as v2c_exc:
            pto.pipe.v2c(slot_size=2048, id=4)
        self.assertIn("requires consumer_buf for local pipes", str(v2c_exc.exception))

    def test_pipe_constructors_require_explicit_stable_ids(self):
        buf = object()
        with make_context():
            gm_slot_type = _pipe_namespace._pto.TensorViewType.get([16, 16], F32Type.get())
        gm_slot = SimpleNamespace(type=gm_slot_type)

        with patch.object(_pipe_namespace, "unwrap_surface_value", side_effect=_identity), \
             patch.object(_pipe_namespace, "_infer_unambiguous_global_slot_size", return_value=1024):
            cases = [
                lambda: pto.pipe.c2v(gm_slot_tensor=gm_slot),
                lambda: pto.pipe.v2c(gm_slot_tensor=gm_slot),
                lambda: pto.pipe.c2v(slot_size=1024, consumer_buf=buf),
                lambda: pto.pipe.v2c(slot_size=1024, consumer_buf=buf),
                lambda: pto.pipe.bidirectional(
                    slot_size=1024,
                    c2v_consumer_buf=buf,
                    v2c_consumer_buf=buf,
                ),
            ]
            for case in cases:
                with self.subTest(case=case):
                    with self.assertRaises(TypeError) as exc:
                        case()
                    self.assertIn("requires an explicit stable id", str(exc.exception))

            invalid_global_cases = [
                lambda: pto.pipe.c2v(gm_slot_tensor=gm_slot, consumer_buf=buf, id=1),
                lambda: pto.pipe.v2c(gm_slot_tensor=gm_slot, gm_slot_buffer=buf, id=1),
                lambda: pto.pipe.c2v(gm_slot_tensor=gm_slot, local_slot_num=2, id=1),
            ]
            for case in invalid_global_cases:
                with self.subTest(case=case):
                    with self.assertRaises(TypeError):
                        case()

    def test_global_pipe_slot_size_inference_requires_unambiguous_nosplit(self):
        with make_context():
            gm_slot_type = _pipe_namespace._pto.TensorViewType.get([16, 16], F32Type.get())
        gm_slot = SimpleNamespace(type=gm_slot_type)

        with patch.object(_pipe_namespace, "unwrap_surface_value", side_effect=_identity):
            for case in (
                lambda: pto.pipe.c2v(gm_slot_tensor=gm_slot, id=11),
                lambda: pto.pipe.v2c(gm_slot_tensor=gm_slot, id=12),
            ):
                with self.subTest(case=case):
                    with self.assertRaises(TypeError) as exc:
                        case()
                    self.assertIn("requires explicit slot_size", str(exc.exception))
                    self.assertIn("nosplit=True", str(exc.exception))

    def test_global_pipe_allows_explicit_full_slot_size_for_split_consumer_shape(self):
        with make_context():
            gm_slot_type = _pipe_namespace._pto.TensorViewType.get([4, 64, 128], F32Type.get())
        gm_slot = SimpleNamespace(type=gm_slot_type)

        with patch.object(_pipe_namespace, "unwrap_surface_value", side_effect=_identity), \
             patch.object(_pipe_namespace, "wrap_surface_value", side_effect=_identity):
            pipe = pto.pipe.c2v(
                gm_slot_tensor=gm_slot,
                slot_size=262144,
                id=13,
            )

        self.assertEqual(pipe.slot_size, 262144)
        self.assertEqual(pipe.entry_type, gm_slot_type)

    def test_local_pipe_transactions_dispatch_to_tile_entry_frontend_ops(self):
        c2v_buf = object()
        v2c_buf = object()
        c2v_tile = object()
        v2c_tile = object()
        c2v_result = object()
        v2c_result = object()
        c2v_type = "c2v_tile_ty"
        v2c_type = "v2c_tile_ty"

        with patch.object(_pipe_namespace, "unwrap_surface_value", side_effect=_identity), \
             patch.object(_pipe_namespace, "wrap_surface_value", side_effect=_identity):
            c2v = pto.pipe.c2v(slot_size=1024, consumer_buf=c2v_buf, id=6)
            v2c = pto.pipe.v2c(slot_size=2048, consumer_buf=v2c_buf, id=7)

        with patch.object(_pipe_namespace._pto, "TPushToAivOp") as c2v_push, \
             patch.object(_pipe_namespace._pto, "TPopFromAicOp", return_value=SimpleNamespace(result=c2v_result)) as c2v_pop, \
             patch.object(_pipe_namespace._pto, "TFreeFromAicOp") as c2v_free, \
             patch.object(_pipe_namespace._pto, "TPushToAicOp") as v2c_push, \
             patch.object(_pipe_namespace._pto, "TPopFromAivOp", return_value=SimpleNamespace(result=v2c_result)) as v2c_pop, \
             patch.object(_pipe_namespace._pto, "TFreeFromAivOp") as v2c_free, \
             patch.object(_pipe_namespace, "unwrap_surface_value", side_effect=_identity), \
             patch.object(_pipe_namespace, "wrap_surface_value", side_effect=_identity):
            c2v.push(c2v_tile, split=1)
            c2v_output = c2v.pop(result_type=c2v_type, split=2)
            c2v.free(split=0)

            v2c.push(v2c_tile, split=2)
            v2c_output = v2c.pop(result_type=v2c_type, split=1)
            v2c.free(split=0)

        c2v_push.assert_called_once_with(c2v_tile, 1, id=6)
        c2v_pop.assert_called_once_with(c2v_type, 2, id=6)
        c2v_free.assert_called_once_with(0, entry=None, id=6)
        self.assertIs(c2v_output, c2v_result)

        v2c_push.assert_called_once_with(v2c_tile, 2, id=7)
        v2c_pop.assert_called_once_with(v2c_type, 1, id=7)
        v2c_free.assert_called_once_with(0, entry=None, id=7)
        self.assertIs(v2c_output, v2c_result)

    def test_local_pipe_pop_can_carry_runtime_valid_shape(self):
        c2v_buf = object()
        c2v_type = "c2v_tile_ty"
        row = object()
        col = object()
        result = object()

        with patch.object(_pipe_namespace, "unwrap_surface_value", side_effect=_identity), \
             patch.object(_pipe_namespace, "wrap_surface_value", side_effect=_identity):
            c2v = pto.pipe.c2v(slot_size=1024, consumer_buf=c2v_buf, id=8)

        with patch.object(_pipe_namespace._pto, "TPopFromAicOp", return_value=SimpleNamespace(result=result)) as pop_op, \
             patch.object(_pipe_namespace, "_coerce_index", side_effect=lambda value, *, context: value), \
             patch.object(_pipe_namespace, "wrap_surface_value", side_effect=_identity):
            output = c2v.pop(result_type=c2v_type, valid_shape=[row, col])

        pop_op.assert_called_once_with(
            c2v_type,
            0,
            valid_row=row,
            valid_col=col,
            id=8,
        )
        self.assertIs(output, result)

    def test_pipe_surface_rejects_ambiguous_or_invalid_transactions(self):
        c2v_buf = object()
        v2c_buf = object()

        with patch.object(_pipe_namespace, "unwrap_surface_value", side_effect=_identity), \
             patch.object(_pipe_namespace, "wrap_surface_value", side_effect=_identity):
            c2v = pto.pipe.c2v(slot_size=1024, consumer_buf=c2v_buf, id=9)
            bidi = pto.pipe.bidirectional(
                slot_size=1024,
                c2v_consumer_buf=c2v_buf,
                v2c_consumer_buf=v2c_buf,
                id=10,
            )

        with self.assertRaises(TypeError):
            c2v.alloc()
        with self.assertRaises(TypeError):
            c2v.pop()
        with self.assertRaises(TypeError):
            bidi.push(object())
        with self.assertRaises(TypeError):
            bidi.pop(result_type="tile_ty")
        with self.assertRaises(TypeError):
            bidi.free()

    def test_bidirectional_pipe_endpoints_dispatch_transactions(self):
        c2v_buf = object()
        v2c_buf = object()
        c2v_tile = object()
        v2c_tile = object()
        c2v_result = object()
        v2c_result = object()

        with patch.object(_pipe_namespace, "unwrap_surface_value", side_effect=_identity), \
             patch.object(_pipe_namespace, "wrap_surface_value", side_effect=_identity):
            bidi = pto.pipe.bidirectional(
                slot_size=1024,
                c2v_consumer_buf=c2v_buf,
                v2c_consumer_buf=v2c_buf,
                id=10,
            )

        with patch.object(_pipe_namespace._pto, "TPushToAivOp") as c2v_push, \
             patch.object(_pipe_namespace._pto, "TPopFromAicOp", return_value=SimpleNamespace(result=c2v_result)) as c2v_pop, \
             patch.object(_pipe_namespace._pto, "TFreeFromAicOp") as c2v_free, \
             patch.object(_pipe_namespace._pto, "TPushToAicOp") as v2c_push, \
             patch.object(_pipe_namespace._pto, "TPopFromAivOp", return_value=SimpleNamespace(result=v2c_result)) as v2c_pop, \
             patch.object(_pipe_namespace._pto, "TFreeFromAivOp") as v2c_free, \
             patch.object(_pipe_namespace, "unwrap_surface_value", side_effect=_identity), \
             patch.object(_pipe_namespace, "wrap_surface_value", side_effect=_identity):
            bidi.c2v.push(c2v_tile, split=0)
            c2v_output = bidi.c2v.pop(result_type="c2v_ty", split=1)
            bidi.c2v.free(split=2)
            bidi.v2c.push(v2c_tile, split=2)
            v2c_output = bidi.v2c.pop(result_type="v2c_ty", split=1)
            bidi.v2c.free(split=0)

        c2v_push.assert_called_once_with(c2v_tile, 0, id=10)
        c2v_pop.assert_called_once_with("c2v_ty", 1, id=10)
        c2v_free.assert_called_once_with(2, entry=None, id=10)
        v2c_push.assert_called_once_with(v2c_tile, 2, id=10)
        v2c_pop.assert_called_once_with("v2c_ty", 1, id=10)
        v2c_free.assert_called_once_with(0, entry=None, id=10)
        self.assertIs(c2v_output, c2v_result)
        self.assertIs(v2c_output, v2c_result)

    def test_reserved_buffer_helpers_normalize_location_and_peer_func(self):
        with make_context():
            with patch.object(_ops._pto, "ReserveBufferOp", return_value=SimpleNamespace(result=object())) as reserve_op, \
                 patch.object(_ops._pto, "ImportReservedBufferOp", return_value=SimpleNamespace(result=object())) as import_op, \
                 patch_ops("wrap_surface_value", side_effect=_identity):
                reserve_result = _ops.reserve_buffer("fifo", size=8192, location="vec")
                import_result = _ops.import_reserved_buffer(
                    "fifo",
                    peer_func=SimpleNamespace(spec=SimpleNamespace(symbol_name="vector_kernel")),
                )

        self.assertIsNotNone(reserve_result)
        self.assertIsNotNone(import_result)
        self.assertEqual(reserve_op.call_args.args[0], "fifo")
        self.assertEqual(reserve_op.call_args.args[1], 8192)
        self.assertEqual(str(reserve_op.call_args.args[2]), "#pto.address_space<vec>")
        self.assertEqual(reserve_op.call_args.args[3], True)
        self.assertEqual(import_op.call_args.args, ("fifo", "vector_kernel"))


if __name__ == "__main__":
    unittest.main()
