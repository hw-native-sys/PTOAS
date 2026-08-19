# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib daemon for the ExpandTileOp Unix-socket RPC contract.

The daemon owns template discovery, selection, specialization, rendering, and an
in-memory instance cache. PTODSL templates are loaded from the Python package, so
the daemon does not scan or depend on an external template directory.

Run it with:

    python3 -m ptodsl.tilelib.serving.daemon --socket <path>
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socketserver
import threading

from .. import constraints as _constraints
from .. import registry as _registry
from ..metadata import ScalarSpec, ScalarType, TileSpec, VectorSpec, ViewSpec
from ..templates import load_template
from .wire import recv_message, send_message


def _remove_socket_path(socket_path: str) -> None:
    """Remove an existing socket entry, including a broken symlink."""
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass


def _build_tile_specs(descriptor, operand_specs: list) -> dict:
    """Map positional daemon operands onto a template's parameter names."""
    if not isinstance(operand_specs, list):
        raise TypeError("operand_specs must be a list")
    if len(operand_specs) != len(descriptor.param_names):
        raise ValueError(
            f"template {descriptor.name!r} expects {len(descriptor.param_names)} "
            f"operands, got {len(operand_specs)}"
        )

    specs = {}
    for index, (name, spec) in enumerate(zip(descriptor.param_names, operand_specs)):
        if not isinstance(spec, dict):
            raise TypeError(f"operand_specs[{index}] must be an object")

        kind = spec.get("kind")
        if kind == "scalar":
            try:
                specs[name] = ScalarSpec(
                    dtype=ScalarType(spec["dtype"]),
                    value=spec.get("value"),
                )
            except KeyError as exc:
                raise ValueError(
                    f"scalar operand {index} ({name!r}) is missing {exc.args[0]!r}"
                ) from exc
            continue

        if kind == "vector":
            try:
                specs[name] = VectorSpec(
                    shape=tuple(spec["shape"]),
                    dtype=ScalarType(spec["dtype"]),
                )
            except KeyError as exc:
                raise ValueError(
                    f"vector operand {index} ({name!r}) is missing {exc.args[0]!r}"
                ) from exc
            continue

        if kind == "view":
            config = spec.get("config") or {}
            if not isinstance(config, dict):
                raise TypeError(f"operand_specs[{index}].config must be an object")
            try:
                strides = spec.get("strides")
                specs[name] = ViewSpec(
                    shape=tuple(spec["shape"]),
                    dtype=ScalarType(spec["dtype"]),
                    memory_space=spec.get("memory_space", "gm"),
                    strides=tuple(strides) if strides is not None else None,
                    layout=config.get("layout"),
                )
            except KeyError as exc:
                raise ValueError(
                    f"view operand {index} ({name!r}) is missing {exc.args[0]!r}"
                ) from exc
            continue

        if kind != "tile":
            raise NotImplementedError(
                "PTODSL TileLib daemon currently supports tile, scalar, view, "
                f"and vector operands; "
                f"operand {index} ({name!r}) has kind {kind!r}"
            )

        config = spec.get("config") or {}
        if not isinstance(config, dict):
            raise TypeError(f"operand_specs[{index}].config must be an object")

        try:
            shape = tuple(spec["shape"])
            dtype = ScalarType(spec["dtype"])
        except KeyError as exc:
            raise ValueError(
                f"tile operand {index} ({name!r}) is missing {exc.args[0]!r}"
            ) from exc

        valid_shape = spec.get("valid_shape")
        specs[name] = TileSpec(
            shape=shape,
            dtype=dtype,
            memory_space=spec.get("memory_space", "ub"),
            valid_shape=tuple(valid_shape) if valid_shape is not None else None,
            b_layout=config.get("b_layout", "row_major"),
            s_layout=config.get("s_layout", "none_box"),
            pad_value=spec.get("pad_value", config.get("pad_value", "Null")),
        )
    return specs


def _constraint_name(predicate) -> str:
    return getattr(predicate, "__name__", repr(predicate))


def _metadata_value(value):
    if callable(value):
        return {"callable": _constraint_name(value)}
    return value


def _metadata_for_descriptor(descriptor, constraint_context: dict) -> dict:
    metadata = descriptor.metadata
    if callable(metadata.Tail):
        has_tail = _constraints.passes((metadata.Tail,), constraint_context)
    else:
        has_tail = bool(metadata.Tail)
    return {
        "op": metadata.op,
        "target": metadata.target,
        "name": metadata.name,
        "dtypes": [list(signature) for signature in metadata.dtypes],
        "layouts": list(metadata.layouts),
        "memory_spaces": list(metadata.memory_spaces),
        "constraints": [
            _constraint_name(predicate) for predicate in metadata.constraints
        ],
        "priority": metadata.priority,
        "fusible": metadata.fusible,
        "loop_depth": metadata.loop_depth,
        "id": metadata.id,
        "Tail": _metadata_value(metadata.Tail),
        "has_tail": has_tail,
        "is_post_update": metadata.is_post_update,
        "iteration_axis": metadata.iteration_axis,
        "op_engine": metadata.op_engine,
        "op_class": metadata.op_class,
        "tags": list(metadata.tags),
        "resource_scope": metadata.resource_scope,
        "resource_vector_values": metadata.resource_vector_values,
        "resource_chunk_streaming": metadata.resource_chunk_streaming,
    }


def _registered_candidates(target: str, op: str) -> list:
    # Import only this op's template module. Registration happens as an import
    # side effect and repeated requests are no-ops because the loader is cached.
    load_template(op, target)
    candidates = _registry.default_registry().lookup(op, target)
    if not candidates:
        raise _registry.NoMatchingTemplate(
            f"no template registered for op={op!r} target={target!r}"
        )
    return candidates


def _is_vmi_descriptor(descriptor) -> bool:
    return "vmi" in getattr(descriptor.metadata, "tags", ())


def _legal_candidate_specs(
    target: str,
    op: str,
    operand_specs: list,
    context_attrs: dict | None = None,
) -> list:
    """Return legal ``(descriptor, specs)`` pairs for this concrete request.

    Different template versions may have different parameter counts/order.  The
    wire operands are positional, so bind them against each descriptor before
    asking the Python constraint legalizer.
    """
    evaluated = []
    for descriptor in _registered_candidates(target, op):
        try:
            specs = _build_tile_specs(descriptor, operand_specs)
        except Exception as exc:
            evaluated.append((descriptor, None, f"operand binding failed: {exc}"))
            continue

        legality = _constraints.evaluate_candidate(
            descriptor,
            specs,
            target,
            op,
            context_attrs,
        )
        evaluated.append(
            (
                descriptor,
                specs,
                legality.reason if not legality.legal else None,
            )
        )

    legal = [
        (descriptor, specs)
        for descriptor, specs, reason in evaluated
        if specs is not None and reason is None
    ]
    if not legal:
        reasons = "; ".join(
            f"{descriptor.name}: {reason}"
            for descriptor, _, reason in evaluated
        )
        raise _registry.NoMatchingTemplate(
            f"no legal template for op={op!r} target={target!r}; {reasons}"
        )

    legal.sort(key=lambda pair: pair[0].metadata.priority, reverse=True)
    return legal


def _select_descriptor_and_specs(
    target: str,
    op: str,
    operand_specs: list,
    context_attrs: dict | None = None,
    candidate_id: str | None = None,
):
    if candidate_id:
        for descriptor in _registered_candidates(target, op):
            if descriptor.name != candidate_id:
                continue
            try:
                specs = _build_tile_specs(descriptor, operand_specs)
            except Exception as exc:
                raise _registry.NoMatchingTemplate(
                    f"candidate {candidate_id!r} cannot bind operands for "
                    f"op={op!r} target={target!r}: {exc}"
                ) from exc

            legality = _constraints.evaluate_candidate(
                descriptor,
                specs,
                target,
                op,
                context_attrs,
            )
            if legality.legal:
                return descriptor, specs
            raise _registry.NoMatchingTemplate(
                f"candidate {candidate_id!r} is not a legal template for "
                f"op={op!r} target={target!r}: {legality.reason}"
            )

        registered_names = ", ".join(
            descriptor.name for descriptor in _registered_candidates(target, op)
        )
        raise _registry.NoMatchingTemplate(
            f"candidate {candidate_id!r} is not registered for op={op!r} "
            f"target={target!r}; registered candidates: {registered_names}"
        )

    legal = _legal_candidate_specs(target, op, operand_specs, context_attrs)

    legal = [
        (descriptor, specs)
        for descriptor, specs in legal
        if not _is_vmi_descriptor(descriptor)
    ]
    if not legal:
        raise _registry.NoMatchingTemplate(
            f"no public TileLib template for op={op!r} target={target!r}"
        )

    if len(legal) == 1:
        return legal[0]

    top_priority = legal[0][0].metadata.priority
    winners = [
        (descriptor, specs)
        for descriptor, specs in legal
        if descriptor.metadata.priority == top_priority
    ]
    if len(winners) > 1:
        names = ", ".join(descriptor.name for descriptor, _ in winners)
        raise _registry.AmbiguousTemplate(
            f"multiple templates tie at priority {top_priority} for op={op!r} "
            f"target={target!r}: {names}"
        )
    return legal[0]


def metadata_request(
    target: str,
    op: str,
    operand_specs: list,
    context_attrs: dict | None = None,
    include_vmi_candidates: bool = False,
) -> dict:
    """Return every legal candidate and its selection metadata."""
    legal = _legal_candidate_specs(target, op, operand_specs, context_attrs)
    if not include_vmi_candidates:
        legal = [
            (descriptor, specs)
            for descriptor, specs in legal
            if not _is_vmi_descriptor(descriptor)
        ]
    return {
        "target": target,
        "op": op,
        "candidates": {
            descriptor.name: _metadata_for_descriptor(
                descriptor,
                {
                    **_constraints.build_context(specs, target, op),
                    **(context_attrs or {}),
                },
            )
            for descriptor, specs in legal
        },
    }


def render_request(
    target: str,
    op: str,
    operand_specs: list,
    context_attrs: dict | None = None,
    candidate_id: str | None = None,
) -> str:
    """Select and render one PTODSL template as MLIR text."""
    descriptor, tile_specs = _select_descriptor_and_specs(
        target,
        op,
        operand_specs,
        context_attrs,
        candidate_id,
    )
    return descriptor.specialize(
        context_attrs=context_attrs or {},
        **tile_specs,
    ).mlir_text()


class TileLibDaemonServer(socketserver.UnixStreamServer):
    """Sequential Unix-socket RPC server with an in-memory render cache."""

    def __init__(self, socket_path: str, max_entries: int = 1000):
        if max_entries <= 0:
            raise ValueError("max_entries must be greater than zero")
        super().__init__(socket_path, _Handler)
        os.chmod(socket_path, 0o600)
        self._cache: dict[str, str] = {}
        self._max_entries = max_entries
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    @property
    def stats(self) -> dict:
        """Return a snapshot of cache counters for diagnostics and tests."""
        return dict(self._stats)

    def dispatch(self, request: dict) -> dict:
        if not isinstance(request, dict):
            return {"success": False, "error": "request must be a JSON object"}

        method = request.get("method")
        params = request.get("params") or {}
        if not isinstance(params, dict):
            return {"success": False, "error": "request params must be a JSON object"}

        try:
            if method == "instantiate":
                result = self._instantiate(**params)
            elif method == "get_metadata":
                result = self._get_metadata(**params)
            elif method == "ping":
                result = "pong"
            elif method == "get_stats":
                result = self._get_stats()
            elif method == "clear":
                result = self._clear()
            else:
                return {"success": False, "error": f"unknown method {method!r}"}
            return {"success": True, "result": result}
        except Exception as exc:
            return {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _get_metadata(
        self,
        target,
        op,
        operand_specs,
        context_attrs=None,
        include_vmi_candidates=False,
    ):
        return metadata_request(
            target,
            op,
            operand_specs,
            context_attrs,
            include_vmi_candidates=include_vmi_candidates,
        )

    def _get_stats(self):
        requests = self._stats["hits"] + self._stats["misses"]
        total_entries = len(self._cache)
        return {
            **self._stats,
            "entries": total_entries,
            "total_entries": total_entries,
            "max_entries": self._max_entries,
            "hit_rate": self._stats["hits"] / requests if requests else 0.0,
        }

    def _clear(self):
        self._cache.clear()
        return {"cleared": True}

    def _instantiate(
        self,
        target,
        op,
        operand_specs,
        context_attrs=None,
        candidate_id=None,
    ):
        key = json.dumps(
            {
                "target": target,
                "op": op,
                "operand_specs": operand_specs,
                "context_attrs": context_attrs,
                "candidate_id": candidate_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

        cached = self._cache.get(key)
        if cached is not None:
            self._stats["hits"] += 1
            return cached
        self._stats["misses"] += 1

        mlir_text = render_request(
            target,
            op,
            operand_specs,
            context_attrs,
            candidate_id,
        )

        if len(self._cache) >= self._max_entries:
            self._cache.pop(next(iter(self._cache)))
            self._stats["evictions"] += 1
        self._cache[key] = mlir_text
        return mlir_text


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            request = recv_message(self.request)
        except (ConnectionError, UnicodeDecodeError, ValueError):
            return
        send_message(self.request, self.server.dispatch(request))


def _parse_args(argv):
    parser = argparse.ArgumentParser(prog="ptodsl.tilelib.serving.daemon")
    parser.add_argument("--socket", required=True)
    parser.add_argument(
        "--template-dir",
        default=None,
        help="accepted during migration but ignored; PTODSL templates are in-package",
    )
    parser.add_argument("--max-entries", type=int, default=1000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    _remove_socket_path(args.socket)

    server = TileLibDaemonServer(args.socket, max_entries=args.max_entries)
    stop = threading.Event()

    def _request_shutdown(*_):
        stop.set()

    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    if args.verbose:
        print(f"PTODSL TileLib daemon listening on {args.socket}", flush=True)

    try:
        stop.wait()
    finally:
        server.shutdown()
        server.server_close()
        _remove_socket_path(args.socket)


if __name__ == "__main__":
    main()


__all__ = ["TileLibDaemonServer", "main", "metadata_request", "render_request"]
