#!/usr/bin/env python3
"""bisheng --shared build for store_pad8 *_vf_sim_kernel.cpp shells."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _ascend_home() -> Path:
    home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if not home:
        raise EnvironmentError("ASCEND_HOME_PATH is not set.")
    return Path(home)


def _bisheng() -> str:
    candidate = _ascend_home() / "bin" / "bisheng"
    if candidate.is_file():
        return str(candidate)
    found = shutil.which("bisheng")
    if found:
        return found
    raise FileNotFoundError("bisheng compiler not found")


def _npu_arch() -> str:
    return os.environ.get("ASCEND_NPU_ARCH", "dav-3510")


def _repro_root(cce_root: Path) -> Path:
    # <store_pad8>/cce -> parent == store_pad8 repro root
    return cce_root.resolve().parent


def _so_path(cce_root: Path, sources: list[Path]) -> Path:
    build_dir = cce_root / "build"
    if len(sources) == 1:
        stem = sources[0].stem
        if stem.endswith("_kernel"):
            stem = stem[: -len("_kernel")]
        return build_dir / f"lib{stem}.so"
    return build_dir / "libstore_pad8_vf_sim.so"


def build_cce_root(cce_root: Path, force: bool = False) -> Path:
    """Build the VF sim .so from all *_vf_sim_kernel.cpp under cce_root/csrc."""
    cce_root = cce_root.resolve()
    csrc = cce_root / "csrc"
    build_dir = cce_root / "build"
    shared_inc = _repro_root(cce_root) / "common" / "inc"
    sources = sorted(csrc.glob("*_vf_sim_kernel.cpp"))
    if not sources:
        raise FileNotFoundError(f"no *_vf_sim_kernel.cpp under {csrc}")

    so_path = _so_path(cce_root, sources)
    build_dir.mkdir(parents=True, exist_ok=True)
    if so_path.is_file() and not force:
        return so_path

    asc_files: list[str] = []
    for src in sources:
        asc = build_dir / f"{src.stem}.asc"
        if asc.is_symlink() or asc.exists():
            asc.unlink()
        asc.symlink_to(src.resolve())
        asc_files.append(str(asc))

    cmd = [
        _bisheng(),
        "-fPIC",
        "--shared",
        f"--npu-arch={_npu_arch()}",
        "-std=c++17",
        f"-I{shared_inc}",
        f"-I{csrc}",
        f"-I{csrc / 'inc'}",
        "-O2",
        "-Wno-ignored-attributes",
        "-Wno-unknown-attributes",
        "-Wno-macro-redefined",
        *asc_files,
        "-o",
        str(so_path),
    ]
    print("==>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=build_dir, check=True)
    print(f"Built {so_path}", flush=True)
    return so_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cce_root",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="Path to cce/ (default: cwd)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        build_cce_root(args.cce_root.resolve(), force=args.force)
    except (EnvironmentError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
