#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Strip the CANN delivery wheel and remove all ELF runtime search paths.

Run after auditwheel has collected and renamed external libraries. Package-local
DT_NEEDED entries use explicit $ORIGIN-relative filenames, which glibc resolves
relative to the requesting ELF, without RPATH, RUNPATH or LD_LIBRARY_PATH. This
also works for direct Python imports and cyclic dependency graphs. Ordinary pip
wheels and development builds keep their existing loading policy.
"""

from __future__ import annotations

import argparse
import filecmp
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
from typing import NamedTuple

from elftools.elf.dynamic import DynamicSegment
from elftools.elf.elffile import ELFFile
from wheel.wheelfile import WheelFile


class ElfInfo(NamedTuple):
    needed: tuple
    soname: str
    search_paths: tuple
    sections: tuple


def inspect_elf(path: Path) -> ElfInfo:
    """Read ELF metadata independently of patchelf's mutation commands."""
    needed, search_paths, sections = [], [], []
    soname = ""
    with path.open("rb") as stream:
        elf = ELFFile(stream)
        sections = [section.name for section in elf.iter_sections()]
        # The loader consumes PT_DYNAMIC even when section headers are absent.
        for segment in elf.iter_segments():
            if not isinstance(segment, DynamicSegment):
                continue
            for tag in segment.iter_tags():
                if tag.entry.d_tag == "DT_NEEDED":
                    needed.append(tag.needed)
                elif tag.entry.d_tag == "DT_SONAME":
                    soname = tag.soname
                elif tag.entry.d_tag in ("DT_RPATH", "DT_RUNPATH"):
                    search_paths.append(tag.entry.d_tag)
    return ElfInfo(tuple(needed), soname, tuple(search_paths), tuple(sections))


def _member_path(root: Path, member) -> Path:
    name = PurePosixPath(member.filename)
    kind = stat.S_IFMT(member.external_attr >> 16)
    if (name.is_absolute() or ".." in name.parts or "\\" in member.filename
            or kind not in (0, stat.S_IFREG, stat.S_IFDIR)):
        raise ValueError(f"unsafe wheel member: {member.filename}")
    return root.joinpath(*name.parts)


def _extract(wheel: Path, root: Path) -> None:
    with WheelFile(str(wheel)) as archive:
        members = archive.infolist()
        if len({member.filename for member in members}) != len(members):
            raise ValueError("duplicate wheel member names")
        if sum(member.file_size for member in members) > 4 * 1024 ** 3:
            raise ValueError("wheel payload exceeds 4 GiB")
        for member in members:
            target = _member_path(root, member)
            if member.filename.endswith(("/RECORD.jws", "/RECORD.p7s")):
                raise ValueError("cannot rewrite a signed wheel")
            if member.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            target.chmod((member.external_attr >> 16) & 0o755 or 0o644)


def _elf_files(root: Path) -> dict:
    binaries = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                if stream.read(4) == b"\x7fELF":
                    binaries[path] = inspect_elf(path)
    if not binaries:
        raise ValueError("wheel contains no ELF files")
    return binaries


def _library_index(binaries: dict) -> dict:
    libraries = {}
    for path, info in binaries.items():
        for name in {path.name, info.soname} - {""}:
            previous = libraries.get(name)
            if previous is None:
                libraries[name] = path
                continue
            # Wheel builders materialize .so symlinks as identical files. Only
            # aliases in one directory have the same $ORIGIN dependency base.
            if (path.parent != previous.parent
                    or not filecmp.cmp(path, previous, shallow=False)):
                raise ValueError(f"ambiguous packaged library: {name}")
            if path.name == name:
                libraries[name] = path
    return libraries


def _origin_target(binary: Path, needed: str, root: Path) -> Path:
    if not needed.startswith("$ORIGIN/"):
        raise ValueError(f"non-relocatable dependency in {binary.name}: {needed}")
    target = (binary.parent / needed[len("$ORIGIN/"):]).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"dependency escapes wheel: {needed}") from error
    return target


def validate_tree(root: Path) -> int:
    """Reject residual paths/symbols and broken package-local dependency edges."""
    binaries = _elf_files(root)
    libraries = _library_index(binaries)
    for path, info in binaries.items():
        debug = [name for name in info.sections
                 if name == ".symtab" or name.startswith((".debug_", ".zdebug_"))]
        if info.search_paths or debug:
            raise ValueError(f"unhardened ELF {path.relative_to(root)}: {info.search_paths}, {debug}")
        for needed in info.needed:
            if "/" in needed:
                if _origin_target(path, needed, root) not in binaries:
                    raise ValueError(f"missing packaged dependency: {needed}")
            elif needed in libraries:
                raise ValueError(f"packaged dependency still needs a search path: {needed}")
    return len(binaries)


def _tool(name: str) -> str:
    executable = shutil.which(name)
    if not executable:
        raise RuntimeError(f"required ELF tool not found: {name}")
    return executable


def _run(command: list) -> None:
    subprocess.run(command, check=True, timeout=120)


def _harden_tree(root: Path) -> int:
    binaries = _elf_files(root)
    libraries = _library_index(binaries)
    strip, patchelf = _tool("strip"), _tool("patchelf")
    for path, info in binaries.items():
        # Strip before patchelf expands ELF sections (auditwheel uses this order
        # too), then validate the final bytes after all modifications.
        _run([strip, "--strip-unneeded", str(path)])
        replacements = []
        for needed in info.needed:
            target = libraries.get(needed)
            if target is not None:
                relative = os.path.relpath(target, path.parent)
                replacements.extend(["--replace-needed", needed, "$ORIGIN/" + relative])
        _run([patchelf, *replacements, "--remove-rpath", str(path)])
    return validate_tree(root)


def _pack(root: Path, destination: Path) -> None:
    with WheelFile(str(destination), "w") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.relative_to(root).as_posix() != archive.record_path:
                archive.write(str(path), path.relative_to(root).as_posix())
        # WheelFile regenerates RECORD hashes/sizes, including every patched ELF.


def process_wheel(wheel: Path, output_dir: Path = None) -> int:
    """Check a wheel, or write a hardened copy atomically to another directory."""
    wheel = wheel.resolve(strict=True)
    if output_dir is not None:
        output_dir = output_dir.resolve()
        if output_dir == wheel.parent:
            raise ValueError("output directory must preserve the original wheel")
        output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ptoas-cann-", dir=output_dir) as temporary:
        root = Path(temporary) / "payload"
        root.mkdir()
        _extract(wheel, root)
        if output_dir is None:
            return validate_tree(root)
        count = _harden_tree(root)
        staged_wheel = Path(temporary) / wheel.name
        _pack(root, staged_wheel)
        # Reopen the final archive and validate RECORD hashes as well as ELF data.
        process_wheel(staged_wheel)
        os.replace(str(staged_wheel), str(output_dir / wheel.name))
        return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir", type=Path)
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    count = process_wheel(args.wheel, args.output_dir)
    print(f"validated {count} stripped ELF files without RPATH/RUNPATH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
