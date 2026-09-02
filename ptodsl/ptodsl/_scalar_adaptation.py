# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared scalar adaptation helpers for PTODSL frontend lowering."""

from __future__ import annotations

from ._types import _materialize_integer_literal

from ptoas.mlir.ir import (
    Attribute,
    BF16Type,
    DenseElementsAttr,
    F16Type,
    F32Type,
    Float8E4M3FNType,
    Float8E5M2Type,
    FloatAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    Operation,
    VectorType,
)
from ptoas.mlir.dialects import pto as _pto


def _pto_constant(type_obj, value_attr):
    return Operation.create(
        "pto.constant", results=[type_obj], attributes={"value": value_attr}
    ).results[0]


def _pto_cast(
    value, target_type, *, index=False, attributes=None, signedness=None
):
    source_type = value.type
    source_kind = classify_runtime_scalar_type(source_type)
    target_kind = classify_runtime_scalar_type(target_type)
    op_attributes = dict(attributes or {})

    if index or _is_index_integer_conversion(source_type, target_type):
        if op_attributes:
            raise ValueError(
                "index/integer casts do not accept numeric conversion attributes"
            )
        return _emit_index_cast(value, target_type, signedness=signedness)

    inferred_signedness = _conversion_signedness(source_type, target_type)
    signedness = signedness or inferred_signedness
    if source_kind == "integer" and target_kind == "integer":
        source_width = _integer_element_type(source_type).width
        target_width = _integer_element_type(target_type).width
        if source_width < target_width:
            mnemonic = "pto.exti"
            op_attributes["signedness"] = _signedness_attr(signedness)
        elif source_width > target_width:
            mnemonic = "pto.trunci"
        else:
            if op_attributes:
                raise ValueError(
                    "same-width integer casts do not accept conversion attributes"
                )
            return _restore_authored_integer_type(
                _to_common_integer_value(value), target_type
            )
    elif source_kind == "integer" and target_kind == "float":
        mnemonic = "pto.itof"
        op_attributes["signedness"] = _signedness_attr(signedness)
    elif source_kind == "float" and target_kind == "integer":
        mnemonic = "pto.ftoi"
        op_attributes["signedness"] = _signedness_attr(signedness)
    elif source_kind == "float" and target_kind == "float":
        mnemonic = "pto.ftof"
    else:
        raise TypeError(
            f"unsupported numeric conversion {source_type} -> {target_type}"
        )

    common_value = _to_common_integer_value(value)
    common_target_type = _signless_integer_type(target_type)
    result = Operation.create(
        mnemonic,
        results=[common_target_type],
        operands=[common_value],
        attributes=op_attributes,
    ).results[0]
    return _restore_authored_integer_type(result, target_type)


def _is_index_integer_conversion(source_type, target_type):
    source_element = _scalar_or_vector_element_type(source_type)
    target_element = _scalar_or_vector_element_type(target_type)
    return (
        IndexType.isinstance(source_element)
        and IntegerType.isinstance(target_element)
    ) or (
        IntegerType.isinstance(source_element)
        and IndexType.isinstance(target_element)
    )


def _index_conversion_signedness(source_type, target_type):
    source_element = _scalar_or_vector_element_type(source_type)
    target_element = _scalar_or_vector_element_type(target_type)
    if IndexType.isinstance(source_element) and IntegerType.isinstance(target_element):
        return _integer_signedness(target_type) or "signed"
    if IntegerType.isinstance(source_element) and IndexType.isinstance(target_element):
        return _integer_signedness(source_type) or "signed"
    raise TypeError(
        "index cast requires exactly one index type and one integer type, got "
        f"{source_type} -> {target_type}"
    )


def _validate_index_cast_types(source_type, target_type):
    source_vector = VectorType.isinstance(source_type)
    target_vector = VectorType.isinstance(target_type)
    if source_vector != target_vector:
        raise TypeError(
            "index/integer casts require both types to be scalar or both to be "
            f"builtin vectors, got {source_type} -> {target_type}"
        )
    if source_vector:
        source_shape = VectorType(source_type)
        target_shape = VectorType(target_type)
        if (
            tuple(source_shape.shape) != tuple(target_shape.shape)
            or tuple(source_shape.scalable_dims) != tuple(target_shape.scalable_dims)
        ):
            raise TypeError(
                "index/integer casts require matching builtin-vector shapes, got "
                f"{source_type} -> {target_type}"
            )
    if not _is_index_integer_conversion(source_type, target_type):
        raise TypeError(
            "index cast requires exactly one index type and one integer type, got "
            f"{source_type} -> {target_type}"
        )


def _emit_index_cast(value, target_type, *, signedness=None):
    source_type = value.type
    _validate_index_cast_types(source_type, target_type)
    signedness = signedness or _index_conversion_signedness(
        source_type, target_type
    )
    source = (
        _to_common_integer_value(value)
        if IntegerType.isinstance(_scalar_or_vector_element_type(source_type))
        else value
    )
    target = (
        _signless_integer_type(target_type)
        if IntegerType.isinstance(_scalar_or_vector_element_type(target_type))
        else target_type
    )
    result = Operation.create(
        "pto.index_cast",
        results=[target],
        operands=[source],
        attributes={"signedness": _signedness_attr(signedness)},
    ).results[0]
    return _restore_authored_integer_type(result, target_type)


def _unrealized_type_cast(value, target_type):
    return Operation.create(
        "builtin.unrealized_conversion_cast",
        results=[target_type],
        operands=[value],
    ).results[0]


def _integer_element_type(type_obj):
    element_type = _scalar_or_vector_element_type(type_obj)
    return (
        IntegerType(element_type)
        if IntegerType.isinstance(element_type)
        else None
    )


def _scalar_or_vector_element_type(type_obj):
    return (
        VectorType(type_obj).element_type
        if VectorType.isinstance(type_obj)
        else type_obj
    )


def _integer_type_with_element(type_obj, converted_element):
    if not VectorType.isinstance(type_obj):
        return converted_element
    vector_type = VectorType(type_obj)
    return VectorType.get(
        list(vector_type.shape),
        converted_element,
        scalable=list(vector_type.scalable_dims),
    )


def _signless_integer_type(type_obj):
    element_type = _integer_element_type(type_obj)
    if element_type is None:
        return type_obj
    return _integer_type_with_element(
        type_obj, IntegerType.get_signless(element_type.width)
    )


def _integer_signedness(type_obj):
    element_type = _scalar_or_vector_element_type(type_obj)
    if IndexType.isinstance(element_type):
        return "signed"
    integer_type = _integer_element_type(type_obj)
    if integer_type is None:
        return None
    if integer_type.width == 1:
        return "unsigned"
    if integer_type.is_unsigned:
        return "unsigned"
    return "signed"


def _signedness_attr(signedness):
    if signedness not in {"signed", "unsigned"}:
        raise ValueError(f"expected signed or unsigned semantics, got {signedness!r}")
    return Attribute.parse(f"#pto.signedness<{signedness}>")


def _conversion_signedness(source_type, target_type):
    source_signedness = _integer_signedness(source_type)
    if source_signedness is not None:
        return source_signedness
    return _integer_signedness(target_type)


def _to_common_integer_value(value):
    common_type = _signless_integer_type(value.type)
    if common_type == value.type:
        return value
    return _unrealized_type_cast(value, common_type)


def _restore_authored_integer_type(value, target_type):
    if value.type == target_type:
        return value
    if _signless_integer_type(target_type) != value.type:
        return value
    return _unrealized_type_cast(value, target_type)


def classify_runtime_scalar_type(type_obj):
    if IndexType.isinstance(type_obj):
        return "index"
    if IntegerType.isinstance(type_obj):
        return "integer"
    if any(cls.isinstance(type_obj) for cls in (BF16Type, F16Type, F32Type,
                                                Float8E4M3FNType, Float8E5M2Type)):
        return "float"
    if any(
        getattr(_pto, name, None) is not None and
        getattr(_pto, name).isinstance(type_obj)
        for name in ("HiF8Type", "HiF8x2Type", "F4E1M2x2Type",
                     "F4E2M1x2Type", "BF16x2Type", "F8E8M0Type")
    ):
        return "float"
    if VectorType.isinstance(type_obj):
        element_type = VectorType(type_obj).element_type
        if IndexType.isinstance(element_type):
            return "index"
        if any(cls.isinstance(element_type) for cls in (BF16Type, F16Type, F32Type,
                                                        Float8E4M3FNType, Float8E5M2Type)):
            return "float"
        if any(
            getattr(_pto, name, None) is not None and
            getattr(_pto, name).isinstance(element_type)
            for name in ("HiF8Type", "F4E1M2x2Type", "F4E2M1x2Type")
        ):
            return "float"
        if IntegerType.isinstance(element_type):
            return "integer"
    raise TypeError(f"runtime scalar operators only support index/int/float values, got {type_obj}")


def is_mlir_value(value) -> bool:
    return not isinstance(value, (bool, int, float)) and hasattr(value, "type")


def materialize_scalar_literal(value, target_type, *, context: str):
    """Materialize one Python literal as an MLIR scalar constant of *target_type*."""
    if isinstance(value, bool):
        raise TypeError(f"{context} does not accept bool literals")

    target_kind = classify_runtime_scalar_type(target_type)
    if isinstance(value, float) and target_kind != "float":
        raise TypeError(
            f"{context} cannot materialize a floating-point literal against non-floating "
            f"target type {target_type}"
        )

    if target_kind == "float":
        if VectorType.isinstance(target_type):
            element_type = VectorType(target_type).element_type
            return _pto_constant(
                target_type,
                DenseElementsAttr.get_splat(
                    target_type,
                    FloatAttr.get(element_type, float(value)),
                ),
            )
        return _pto_constant(
            target_type, FloatAttr.get(target_type, float(value))
        )
    if target_kind == "index":
        return _pto_constant(
            target_type, IntegerAttr.get(target_type, int(value))
        )

    if VectorType.isinstance(target_type):
        common_type = _signless_integer_type(target_type)
        element_type = VectorType(common_type).element_type
        result = _pto_constant(
            common_type,
            DenseElementsAttr.get_splat(
                common_type,
                IntegerAttr.get(element_type, int(value)),
            ),
        )
        return _restore_authored_integer_type(result, target_type)
    return _materialize_integer_literal(target_type, value)


def coerce_scalar_value_to_type(value, target_type, *, context: str):
    """Normalize one already-unwrapped authored scalar value/literal to *target_type*."""
    if not hasattr(value, "type"):
        return materialize_scalar_literal(value, target_type, context=context)

    if value.type == target_type:
        return value

    source_kind = classify_runtime_scalar_type(value.type)
    target_kind = classify_runtime_scalar_type(target_type)

    if source_kind == "index" and target_kind == "integer":
        return coerce_integer_like(value, target_type)
    if source_kind == "integer" and target_kind == "index":
        return _pto_cast(value, target_type, index=True)
    if source_kind == "integer" and target_kind == "integer":
        return coerce_integer_like(value, target_type)
    if source_kind == "float" and target_kind == "float":
        return _coerce_float_like(value, target_type)

    raise TypeError(
        f"{context} cannot coerce the authored value to the expected scalar type: "
        f"got {value.type}, expected {target_type}"
    )


def coerce_runtime_index_value(value, *, context: str):
    """Normalize one already-unwrapped authored value/literal to an MLIR index value."""
    if isinstance(value, bool):
        raise TypeError(f"{context} does not accept bool values")
    if isinstance(value, int):
        index_type = IndexType.get()
        return _pto_constant(index_type, IntegerAttr.get(index_type, value))
    if not hasattr(value, "type"):
        raise TypeError(
            f"{context} expects a Python int, an index value, or an integer runtime scalar; "
            f"got {value!r}"
        )

    value_type = value.type
    if IndexType.isinstance(value_type):
        return value
    if IntegerType.isinstance(value_type):
        return _pto_cast(value, IndexType.get(), index=True)

    raise TypeError(f"{context} expects an index or integer runtime scalar, got {value_type}")


def coerce_runtime_loop_bounds(start, stop, step, *, context: str):
    """Normalize loop bounds while preserving safe signed-integer loops.

    ``scf.for`` accepts signless integer bounds, but its loop condition is
    signed.  Therefore same-width signless/signed integer bounds remain in
    that integer type, while index bounds remain index. Unsigned bounds are
    rejected because converting them to index would not provide unsigned-loop
    semantics.
    Python literals follow the selected runtime bound type.
    """
    raw_values = (start, stop, step)
    if any(isinstance(value, bool) for value in raw_values):
        raise TypeError(f"{context} does not accept bool values")
    typed_values = tuple(value for value in raw_values if hasattr(value, "type"))
    if not typed_values:
        index_type = IndexType.get()
        return tuple(
            materialize_scalar_literal(value, index_type, context=context)
            for value in raw_values
        )

    bound_types = tuple(value.type for value in typed_values)
    if any(not (IndexType.isinstance(ty) or IntegerType.isinstance(ty)) for ty in bound_types):
        raise TypeError(
            f"{context} expects an index or integer runtime scalar, got "
            + ", ".join(str(value.type) for value in typed_values)
        )

    integer_widths = {
        IntegerType(ty).width for ty in bound_types if IntegerType.isinstance(ty)
    }
    if len(integer_widths) > 1:
        raise TypeError(
            f"{context} requires integer bounds with matching widths, got "
            + ", ".join(str(value.type) for value in typed_values)
        )

    if any(_integer_signedness(value.type) == "unsigned" for value in typed_values):
        raise TypeError(
            f"{context} does not support unsigned integer bounds; "
            "convert the bound to index or a signed integer explicitly"
        )

    if any(IndexType.isinstance(ty) for ty in bound_types):
        target_type = IndexType.get()
    else:
        target_type = IntegerType.get_signless(integer_widths.pop())

    normalized = []
    for value in raw_values:
        if not hasattr(value, "type"):
            normalized.append(materialize_scalar_literal(value, target_type, context=context))
            continue
        value_kind = classify_runtime_scalar_type(value.type)
        if value_kind == "index" and IndexType.isinstance(target_type):
            normalized.append(value)
        elif value_kind == "integer" and IntegerType.isinstance(target_type):
            if _integer_element_type(value.type).width != IntegerType(target_type).width:
                raise TypeError(f"{context} requires integer bounds with matching widths")
            normalized.append(_to_common_integer_value(value))
        elif value_kind == "integer" and IndexType.isinstance(target_type):
            normalized.append(_emit_index_cast(value, target_type))
        elif value_kind == "index" and IntegerType.isinstance(target_type):
            raise TypeError(f"{context} cannot mix index and integer bounds")
        else:
            raise TypeError(f"{context} expects index or integer bounds, got {value.type}")
    return tuple(normalized)


def coerce_runtime_integer_value(value, target_type, *, context: str):
    """Normalize one authored integer-like value/literal to *target_type*."""
    if isinstance(value, bool):
        raise TypeError(f"{context} does not accept bool values")
    if isinstance(value, int):
        return _materialize_integer_literal(target_type, value)
    if not hasattr(value, "type"):
        raise TypeError(f"{context} expects an integer-like scalar, got {value!r}")

    kind = classify_runtime_scalar_type(value.type)
    if kind == "float":
        raise TypeError(f"{context} expects an integer-like scalar, got {value.type}")
    return coerce_integer_like(value, target_type)


def coerce_runtime_integer_to_i1(value, *, context: str):
    """Convert one runtime integer-like value to ``i1`` using nonzero truthiness."""
    if not hasattr(value, "type"):
        raise TypeError(f"{context} expects an integer-like runtime scalar, got {value!r}")

    if IndexType.isinstance(value.type):
        zero = _pto_constant(IndexType.get(), IntegerAttr.get(IndexType.get(), 0))
        return Operation.create(
            "pto.cmpi",
            results=[IntegerType.get_signless(1)],
            operands=[value, zero],
            attributes={
                "predicate": Attribute.parse("#pto.scalar_cmp_predicate<ne>"),
                "signedness": _signedness_attr("signed"),
            },
        ).results[0]

    if not IntegerType.isinstance(value.type):
        raise TypeError(f"{context} expects an integer-like runtime scalar, got {value.type}")

    signless_type = _signless_integer_type(value.type)
    signless_value = _to_common_integer_value(value)
    if IntegerType(signless_type).width == 1:
        return signless_value
    zero = _pto_constant(signless_type, IntegerAttr.get(signless_type, 0))
    return Operation.create(
        "pto.cmpi",
        results=[IntegerType.get_signless(1)],
        operands=[signless_value, zero],
        attributes={
            "predicate": Attribute.parse("#pto.scalar_cmp_predicate<ne>"),
            "signedness": _signedness_attr(_integer_signedness(value.type)),
        },
    ).results[0]


def coerce_runtime_i1_value(value, *, context: str):
    """Normalize one authored bool/integer-like value/literal to signless i1."""
    i1_type = IntegerType.get_signless(1)
    if isinstance(value, bool):
        return _materialize_integer_literal(i1_type, int(value))
    if isinstance(value, int):
        if value not in (0, 1):
            raise ValueError(f"{context} expects a bool or 0/1 integer, got {value}")
        return _materialize_integer_literal(i1_type, value)
    if not hasattr(value, "type"):
        raise TypeError(f"{context} expects a bool or integer-like scalar, got {value!r}")

    kind = classify_runtime_scalar_type(value.type)
    if kind == "float":
        raise TypeError(f"{context} expects a bool or integer-like scalar, got {value.type}")
    return coerce_runtime_integer_to_i1(value, context=context)


def normalize_runtime_binary_operands(lhs, rhs, *, require_matching_signedness=False):
    lhs_is_value = is_mlir_value(lhs)
    rhs_is_value = is_mlir_value(rhs)

    if not lhs_is_value and not rhs_is_value:
        raise TypeError("runtime scalar operators require at least one traced runtime operand")

    if lhs_is_value and rhs_is_value:
        return reconcile_typed_runtime_binary_operands(
            lhs, rhs, require_matching_signedness=require_matching_signedness)

    anchor_type = lhs.type if lhs_is_value else rhs.type
    lhs = lhs if lhs_is_value else _materialize_runtime_literal(lhs, anchor_type)
    rhs = rhs if rhs_is_value else _materialize_runtime_literal(rhs, anchor_type)
    return reconcile_typed_runtime_binary_operands(
        lhs, rhs, require_matching_signedness=require_matching_signedness)


def reconcile_typed_runtime_binary_operands(lhs, rhs, *, require_matching_signedness=False):
    lhs_type = lhs.type
    rhs_type = rhs.type

    if lhs_type == rhs_type:
        return lhs, rhs, classify_runtime_scalar_type(lhs_type)

    if IndexType.isinstance(lhs_type) and IntegerType.isinstance(rhs_type):
        rhs = _pto_cast(rhs, IndexType.get(), index=True)
        return lhs, rhs, "index"

    if IntegerType.isinstance(lhs_type) and IndexType.isinstance(rhs_type):
        lhs = _pto_cast(lhs, IndexType.get(), index=True)
        return lhs, rhs, "index"

    if IntegerType.isinstance(lhs_type) and IntegerType.isinstance(rhs_type):
        if (require_matching_signedness and lhs_type != rhs_type and
                _integer_signedness(lhs_type) != _integer_signedness(rhs_type)):
            raise TypeError(
                "sign-sensitive runtime scalar operators require matching signedness; "
                f"got {lhs_type} and {rhs_type}; cast explicitly before the operation"
            )
        lhs_width = IntegerType(lhs_type).width
        rhs_width = IntegerType(rhs_type).width
        target_type = lhs_type if lhs_width >= rhs_width else rhs_type
        lhs = coerce_integer_like(lhs, target_type)
        rhs = coerce_integer_like(rhs, target_type)
        return lhs, rhs, "integer"

    raise TypeError(
        "runtime scalar operators require matching scalar types or an index/integer pair; "
        f"got {lhs_type} and {rhs_type}"
    )


def coerce_integer_like(value, target_type):
    if IndexType.isinstance(value.type):
        return _pto_cast(value, target_type, index=True)

    source_type = value.type
    source_width = IntegerType(source_type).width
    target_width = IntegerType(target_type).width
    if source_width != target_width or source_type != target_type:
        # i1 carries boolean truth values; widening must preserve 0/1 storage.
        signedness = "unsigned" if source_width == 1 else None
        return _pto_cast(value, target_type, signedness=signedness)
    return value


def _materialize_runtime_literal(value, anchor_type):
    if isinstance(value, bool):
        raise TypeError("runtime scalar operators do not accept bool literals")

    kind = classify_runtime_scalar_type(anchor_type)
    if isinstance(value, float) and kind != "float":
        raise TypeError(
            "runtime scalar operators cannot materialize a floating-point literal "
            f"against non-floating operand type {anchor_type}"
        )

    return materialize_scalar_literal(
        value, anchor_type, context="runtime scalar operator"
    )


def _coerce_float_like(value, target_type):
    if value.type == target_type:
        return value

    source_shape = _float_shape(value.type)
    target_shape = _float_shape(target_type)
    if source_shape != target_shape:
        raise TypeError(
            "cannot coerce floating-point values with different shapes: "
            f"{value.type} and {target_type}"
        )

    source_width = _float_bytewidth(value.type)
    target_width = _float_bytewidth(target_type)
    if source_width < target_width:
        return _pto_cast(value, target_type)
    if source_width > target_width:
        return _pto_cast(value, target_type)
    raise TypeError(
        "cannot coerce between different floating-point types of the same width: "
        f"{value.type} and {target_type}"
    )


def _float_shape(type_obj):
    if VectorType.isinstance(type_obj):
        return tuple(VectorType(type_obj).shape)
    return ()


def _float_bytewidth(type_obj):
    if VectorType.isinstance(type_obj):
        return _float_bytewidth(VectorType(type_obj).element_type)
    if Float8E4M3FNType.isinstance(type_obj) or Float8E5M2Type.isinstance(type_obj):
        return 1
    if BF16Type.isinstance(type_obj) or F16Type.isinstance(type_obj):
        return 2
    if F32Type.isinstance(type_obj):
        return 4
    raise TypeError(f"unsupported floating-point type {type_obj}")


__all__ = [
    "classify_runtime_scalar_type",
    "coerce_integer_like",
    "coerce_runtime_i1_value",
    "coerce_runtime_integer_to_i1",
    "coerce_runtime_index_value",
    "coerce_runtime_integer_value",
    "coerce_scalar_value_to_type",
    "is_mlir_value",
    "materialize_scalar_literal",
    "normalize_runtime_binary_operands",
    "reconcile_typed_runtime_binary_operands",
]
