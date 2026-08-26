# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Host-side additions to the native build of one ``@pto.jit`` kernel.

PTODSL builds a kernel into a shared library by compiling generated launch code
and the ptoas-produced kernel object, then linking the two. That is a closed set,
which is a problem for a kernel whose host side already exists in C++: the only
ways in were to reimplement it in Python or to ship it as a second library the
caller loads separately.

``native_options`` opens that up. It names host C++ sources to compile and link
into the same library, the include directories they need, and the libraries to
link against. Those sources become part of the build's identity, so editing one
rebuilds the library rather than silently reusing it.

The public surface is a plain mapping, matching ``frontend_options`` on the same
decorator. It is normalized here into a frozen, hashable record, because
``KernelModuleSpec`` is frozen and the build cache compares these values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_SUPPORTED_NATIVE_OPTION_KEYS = frozenset(
    {"host_sources", "include_dirs", "link_libraries", "library_dirs"}
)

# A library name goes onto the link line as -lNAME. Anything that could be read
# as a second argument, a path, or a flag is rejected rather than pasted.
_FORBIDDEN_LIBRARY_CHARS = frozenset({"/", "\\", " ", "\t", "\n", "\r", '"', "'", ";", "&", "|", "$"})


@dataclass(frozen=True)
class NativeBuildOptions:
    """Resolved host-side additions to one kernel's native build."""

    host_sources: tuple[Path, ...] = ()
    include_dirs: tuple[Path, ...] = ()
    link_libraries: tuple[str, ...] = ()
    library_dirs: tuple[Path, ...] = ()

    def is_empty(self) -> bool:
        return not (
            self.host_sources or self.include_dirs or self.link_libraries or self.library_dirs
        )


EMPTY_NATIVE_OPTIONS = NativeBuildOptions()


def _require_path_sequence(key: str, value) -> tuple[str, ...]:
    """Accept one path or an iterable of paths, and reject anything else clearly."""
    if isinstance(value, (str, Path)):
        return (str(value),)
    if isinstance(value, (bytes, bytearray)):
        raise TypeError(f"@pto.jit native_options[{key!r}] must be str or os.PathLike, not bytes")
    try:
        items = list(value)
    except TypeError as exc:
        raise TypeError(
            f"@pto.jit native_options[{key!r}] must be a path or an iterable of paths"
        ) from exc
    resolved = []
    for item in items:
        if not isinstance(item, (str, Path)):
            raise TypeError(
                f"@pto.jit native_options[{key!r}] entries must be str or os.PathLike, "
                f"got {type(item).__name__}"
            )
        text = str(item)
        if not text:
            raise ValueError(f"@pto.jit native_options[{key!r}] contains an empty path")
        resolved.append(text)
    return tuple(resolved)


def _resolve_against(base_dir: Path | None, raw: str) -> Path:
    """Anchor a relative path at the file that declared the kernel.

    Relative to the declaring module rather than the process working directory,
    so a kernel keeps building wherever it is invoked from. This is the rule
    ``@pto.jit(source=...)`` already uses for its IR path.
    """
    candidate = Path(raw).expanduser()
    if candidate.is_absolute() or base_dir is None:
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def _normalize_paths(key: str, value, *, base_dir: Path | None) -> tuple[Path, ...]:
    resolved = []
    for raw in _require_path_sequence(key, value):
        path = _resolve_against(base_dir, raw)
        if path not in resolved:
            resolved.append(path)
    return tuple(resolved)


def _normalize_libraries(value) -> tuple[str, ...]:
    names = []
    for raw in _require_path_sequence("link_libraries", value):
        # -l takes a bare name; the caller means library_dirs if they have a path.
        bad = sorted(_FORBIDDEN_LIBRARY_CHARS.intersection(raw))
        if bad:
            raise ValueError(
                f"@pto.jit native_options['link_libraries'] entry {raw!r} contains "
                f"unsupported characters {bad!r}; pass a bare library name such as 'dl' "
                "and put directories in native_options['library_dirs']"
            )
        if raw.startswith("-"):
            raise ValueError(
                f"@pto.jit native_options['link_libraries'] entry {raw!r} looks like a linker "
                "flag; pass a bare library name such as 'dl'"
            )
        if raw not in names:
            names.append(raw)
    return tuple(names)


def normalize_native_options(
    native_options: Mapping | None,
    *,
    declaring_file: str | None = None,
    function_name: str | None = None,
) -> NativeBuildOptions:
    """Validate the public mapping and resolve it against the declaring file.

    Paths are resolved but not required to exist: a kernel is often declared
    before its host sources are generated, and the build reports a missing source
    with the command that needed it.
    """
    if native_options is None:
        return EMPTY_NATIVE_OPTIONS
    if not isinstance(native_options, Mapping):
        raise TypeError("@pto.jit native_options must be a mapping when provided")

    unknown = set(native_options) - _SUPPORTED_NATIVE_OPTION_KEYS
    if unknown:
        raise ValueError(
            f"@pto.jit native_options has unsupported keys: {sorted(unknown)!r}; "
            f"supported keys are {sorted(_SUPPORTED_NATIVE_OPTION_KEYS)!r}"
        )

    base_dir = Path(declaring_file).resolve().parent if declaring_file else None

    options = NativeBuildOptions(
        host_sources=_normalize_paths(
            "host_sources", native_options.get("host_sources", ()), base_dir=base_dir
        ),
        include_dirs=_normalize_paths(
            "include_dirs", native_options.get("include_dirs", ()), base_dir=base_dir
        ),
        link_libraries=_normalize_libraries(native_options.get("link_libraries", ())),
        library_dirs=_normalize_paths(
            "library_dirs", native_options.get("library_dirs", ()), base_dir=base_dir
        ),
    )

    # Include and library directories only mean something for sources being
    # compiled or symbols being linked. Accepting them alone would silently do
    # nothing, which is worth a diagnostic rather than a quiet no-op.
    if options.include_dirs and not options.host_sources:
        where = f" for {function_name}" if function_name else ""
        raise ValueError(
            f"@pto.jit native_options['include_dirs'] was set without 'host_sources'{where}; "
            "include directories apply to the host sources this option compiles"
        )

    return options


__all__ = [
    "EMPTY_NATIVE_OPTIONS",
    "NativeBuildOptions",
    "normalize_native_options",
]
