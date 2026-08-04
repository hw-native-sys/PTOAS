# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""In-process PTODSL TileLib materialization entry point."""

from __future__ import annotations

import json

from ._render_runtime import _TemplateTrace
from ._selection import _select_descriptor_and_specs, metadata_request


def metadata(
    target: str,
    op: str,
    operand_specs_json: str,
    context_attrs_json: str,
) -> str:
    """Return candidate metadata JSON without any daemon transport."""
    try:
        operand_specs = json.loads(operand_specs_json)
        context_attrs = json.loads(context_attrs_json or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid TileLib metadata request: {exc}") from exc
    return json.dumps(
        metadata_request(target, op, operand_specs, context_attrs),
        separators=(",", ":"),
        sort_keys=True,
    )


def materialize(
    target: str,
    op: str,
    operand_specs_json: str,
    context_attrs_json: str,
    candidate_id: str | None,
    context,
):
    """Return ``(source_module, entry_symbol)`` in *context*.

    JSON is retained only for the compact pure-data specialization request. No
    MLIR text is produced or parsed by this path.
    """
    try:
        operand_specs = json.loads(operand_specs_json)
        context_attrs = json.loads(context_attrs_json or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid TileLib materialization request: {exc}") from exc

    descriptor, tile_specs = _select_descriptor_and_specs(
        target,
        op,
        operand_specs,
        context_attrs,
        candidate_id or None,
    )
    module = _TemplateTrace(
        descriptor,
        tile_specs,
        context_attrs=context_attrs,
    ).build_module_in_context(context)
    module.operation.verify()
    return module, descriptor.name


__all__ = ["materialize", "metadata"]
