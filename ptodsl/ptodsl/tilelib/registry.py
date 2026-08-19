# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TileLib registry + version selection.

Mirrors the *logic* of tilelang-dsl's ``select_kernel`` (filter legal candidates, rank by
priority) but is engine-agnostic: it returns the chosen ``TileTemplate`` descriptor, which
the caller renders via ptodsl's engine. Selection order:

    1. filter by op + target (hard)
    2. delegate all candidate legality to constraints.py
    3. no legal candidate  -> error
    4. one legal candidate -> choose it
    5. several             -> highest priority wins; remaining ties -> error
"""

from __future__ import annotations

from . import constraints as _constraints


class NoMatchingTemplate(Exception):
    pass


class AmbiguousTemplate(Exception):
    pass


def candidate_sort_key(descriptor):
    """Return the deterministic legal-candidate reporting order.

    Priority is the only selection rank. Name is a stable tie ordering for
    diagnostics and non-winning candidates; a top-priority tie is still an
    ambiguity and is rejected by ``select``.
    """

    return (-descriptor.metadata.priority, descriptor.name)


def _has_tag(descriptor, tag: str) -> bool:
    return tag in getattr(descriptor.metadata, "tags", ())


def _is_default_hidden(descriptor) -> bool:
    return _has_tag(descriptor, "vmi")


class TileTemplateRegistry:
    def __init__(self):
        self._descriptors: list = []

    def register(self, descriptor) -> None:
        # Re-registration (e.g. module reload) replaces the prior entry with the same name.
        self._descriptors = [
            d for d in self._descriptors
            if not (d.op == descriptor.op and d.target == descriptor.target and d.name == descriptor.name)
        ]
        self._descriptors.append(descriptor)

    def all(self) -> tuple:
        return tuple(self._descriptors)

    def lookup(self, op: str, target: str) -> list:
        return [d for d in self._descriptors if d.op == op and d.target == target]

    def legal_candidates(self, op: str, target: str, tile_specs: dict,
                         context_attrs: dict | None = None,
                         include_hidden: bool = False) -> list:
        candidates = self.lookup(op, target)
        if not candidates:
            raise NoMatchingTemplate(f"no template registered for op={op!r} target={target!r}")

        evaluated = [
            (
                descriptor,
                _constraints.evaluate_candidate(
                    descriptor,
                    tile_specs,
                    target,
                    op,
                    context_attrs,
                ),
            )
            for descriptor in candidates
        ]
        legal = [
            descriptor
            for descriptor, result in evaluated
            if result.legal
        ]
        if not include_hidden:
            legal = [
                descriptor
                for descriptor in legal
                if not _is_default_hidden(descriptor)
            ]
        if not legal:
            reasons = "; ".join(
                f"{descriptor.name}: {result.reason}"
                for descriptor, result in evaluated
            )
            raise NoMatchingTemplate(
                f"no legal template for op={op!r} target={target!r}; {reasons}"
            )

        legal.sort(key=candidate_sort_key)
        return legal

    def select(self, op: str, target: str, tile_specs: dict,
               context_attrs: dict | None = None, candidate_id: str | None = None):
        if candidate_id:
            candidates = self.lookup(op, target)
            matched = [descriptor for descriptor in candidates if descriptor.name == candidate_id]
            if not matched:
                names = ", ".join(descriptor.name for descriptor in candidates)
                raise NoMatchingTemplate(
                    f"candidate {candidate_id!r} is not registered for op={op!r} "
                    f"target={target!r}; registered candidates: {names}"
                )
            descriptor = matched[0]
            legality = _constraints.evaluate_candidate(
                descriptor,
                tile_specs,
                target,
                op,
                context_attrs,
            )
            if legality.legal:
                return descriptor
            raise NoMatchingTemplate(
                f"candidate {candidate_id!r} is not a legal template for op={op!r} "
                f"target={target!r}: {legality.reason}"
            )

        legal = self.legal_candidates(
            op,
            target,
            tile_specs,
            context_attrs,
            include_hidden=bool(candidate_id),
        )

        if len(legal) == 1:
            return legal[0]

        top_priority = legal[0].metadata.priority
        winners = [d for d in legal if d.metadata.priority == top_priority]
        if len(winners) > 1:
            names = ", ".join(d.name for d in winners)
            raise AmbiguousTemplate(
                f"multiple templates tie at priority {top_priority} for op={op!r} "
                f"target={target!r}: {names}; assign distinct priorities or make "
                "their constraints mutually exclusive"
            )
        return legal[0]


# Process-wide default registry (the decorator registers into this one).
_DEFAULT_REGISTRY = TileTemplateRegistry()


def default_registry() -> TileTemplateRegistry:
    return _DEFAULT_REGISTRY


def register(descriptor) -> None:
    _DEFAULT_REGISTRY.register(descriptor)


def _load_default_templates(op: str, target: str) -> None:
    # Import lazily to avoid a registry/templates import cycle during package
    # initialization. The loader is cached and registers descriptors as a
    # module-import side effect.
    from ._template_package import load_template

    load_template(op, target)


def legal_candidates(op: str, target: str, tile_specs: dict,
                     context_attrs: dict | None = None,
                     include_hidden: bool = False):
    _load_default_templates(op, target)
    return _DEFAULT_REGISTRY.legal_candidates(
        op,
        target,
        tile_specs,
        context_attrs,
        include_hidden=include_hidden,
    )


def select(op: str, target: str, tile_specs: dict, context_attrs: dict | None = None,
           candidate_id: str | None = None):
    _load_default_templates(op, target)
    return _DEFAULT_REGISTRY.select(op, target, tile_specs, context_attrs, candidate_id)


__all__ = [
    "TileTemplateRegistry",
    "NoMatchingTemplate",
    "AmbiguousTemplate",
    "candidate_sort_key",
    "default_registry",
    "legal_candidates",
    "register",
    "select",
]
