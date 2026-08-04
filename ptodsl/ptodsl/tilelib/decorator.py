# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""The ``@tile_template`` decorator + the registered descriptor / specialization artifact."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

from . import registry as _registry
from ._render_runtime import _TemplateTrace
from .metadata import TemplateMetadata
from .._tracing import ModuleArtifact


@dataclass(frozen=True)
class TileTemplate:
    """A registered template version: the Python body + its metadata + parameter order."""

    py_fn: object
    metadata: TemplateMetadata
    param_names: tuple

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def target(self) -> str:
        return self.metadata.target

    @property
    def op(self) -> str:
        return self.metadata.op

    def specialize(self, context_attrs=None, **tile_specs) -> "SpecializedTileTemplate":
        return SpecializedTileTemplate(self, tile_specs, context_attrs=context_attrs)


class SpecializedTileTemplate(ModuleArtifact):
    """A ``TileTemplate`` bound to concrete ``TileSpec``s; ``.mlir_text()`` renders it."""

    def __init__(self, descriptor: TileTemplate, tile_specs: dict, context_attrs=None):
        super().__init__(
            descriptor.name,
            module_factory=lambda: _TemplateTrace(
                descriptor, tile_specs, context_attrs=context_attrs
            ).build_standalone_module(),
        )
        self.descriptor = descriptor
        self.tile_specs = tile_specs
        self.context_attrs = dict(context_attrs or {})

def tile_template(*, op, target="a5", name=None, dtypes=(), layouts=(),
                  memory_spaces=(), constraints=(), priority=0, fusible=False,
                  loop_depth=None, id=None, Tail=None, is_post_update=False,
                  iteration_axis="none", op_engine="other", op_class="other",
                  tags=(), register=True):
    """Register a Python function as a TileLib implementation of *op* for *target*."""
    if target != "a5":
        raise ValueError("tile-template tracing currently only supports target='a5'")

    def decorator(fn):
        descriptor = TileTemplate(
            py_fn=fn,
            metadata=TemplateMetadata.build(
                op=op,
                target=target,
                name=name or fn.__name__,
                dtypes=dtypes,
                layouts=layouts,
                memory_spaces=memory_spaces,
                constraints=constraints,
                priority=priority,
                fusible=fusible,
                loop_depth=loop_depth,
                id=id,
                Tail=Tail,
                is_post_update=is_post_update,
                iteration_axis=iteration_axis,
                op_engine=op_engine,
                op_class=op_class,
                tags=tags,
            ),
            param_names=tuple(inspect.signature(fn).parameters.keys()),
        )
        if register:
            _registry.register(descriptor)
        return descriptor

    return decorator


__all__ = ["TileTemplate", "SpecializedTileTemplate", "tile_template"]
