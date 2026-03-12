#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mlir import ir
from mlir.dialects import pto as raw_pto


REPO_ROOT = Path(__file__).resolve().parents[1]
OPS_TD = REPO_ROOT / "include" / "PTO" / "IR" / "PTOOps.td"
TYPES_TD = REPO_ROOT / "include" / "PTO" / "IR" / "PTOTypeDefs.td"
ATTRS_TD = REPO_ROOT / "include" / "PTO" / "IR" / "PTOAttrs.td"
PYTHON_BINDINGS = REPO_ROOT / "lib" / "Bindings" / "Python" / "PTOModule.cpp"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_ops(text: str) -> list[dict[str, str]]:
    pattern = re.compile(r'def\s+([A-Za-z0-9_]+)\s*:\s*PTO_[A-Za-z0-9_]+<"([^"]+)"', re.M)
    ops = []
    for def_name, mnemonic in pattern.findall(text):
        ops.append(
            {
                "def_name": def_name,
                "mnemonic": mnemonic,
                "python_symbol": f"{def_name}",
            }
        )
    return ops


def _extract_types(text: str) -> list[dict[str, str]]:
    pattern = re.compile(r'def\s+([A-Za-z0-9_]+)\s*:\s*TypeDef<PTO_Dialect,\s*"([^"]+)">', re.M)
    return [
        {
            "def_name": def_name,
            "dialect_name": dialect_name,
            "python_symbol": f"{def_name}",
        }
        for def_name, dialect_name in pattern.findall(text)
    ]


def _extract_attrs(text: str) -> list[dict[str, str]]:
    pattern = re.compile(r'def\s+([A-Za-z0-9_]+)\s*:\s*PTO_Attr<"([^"]+)",\s*"([^"]+)"', re.M)
    return [
        {
            "def_name": def_name,
            "dialect_name": dialect_name,
            "mnemonic": mnemonic,
            "python_symbol": def_name,
        }
        for def_name, dialect_name, mnemonic in pattern.findall(text)
    ]


def _extract_enums(text: str) -> list[dict[str, str]]:
    pattern = re.compile(r'def\s+([A-Za-z0-9_]+Enum)\s*:\s*PTO_I32Enum<\s*"([^"]+)"', re.M)
    return [
        {
            "def_name": def_name,
            "dialect_name": dialect_name,
            "python_symbol": dialect_name,
        }
        for def_name, dialect_name in pattern.findall(text)
    ]


def _extract_python_type_methods(text: str) -> dict[str, list[str]]:
    type_block_pattern = re.compile(
        r'mlir_type_subclass\(\s*m,\s*"([A-Za-z0-9_]+)"(.*?)(?=^\s*mlir_type_subclass\(\s*m,\s*"[A-Za-z0-9_]+"|^\s*populatePTODialectSubmodule\(m\);)',
        re.M | re.S,
    )
    method_pattern = re.compile(r'\.(?:def|def_classmethod)\(\s*"([A-Za-z0-9_]+)"')
    methods: dict[str, list[str]] = {}
    for type_name, block in type_block_pattern.findall(text):
        methods[type_name] = sorted(set(method_pattern.findall(block)))
    return methods


def build_manifest() -> dict[str, object]:
    ops = _extract_ops(_read(OPS_TD))
    types = _extract_types(_read(TYPES_TD))
    attrs_text = _read(ATTRS_TD)
    attrs = _extract_attrs(attrs_text)
    enums = _extract_enums(attrs_text)
    with ir.Context() as ctx:
        raw_pto.register_dialect(ctx, load=True)
        python_symbols = sorted(name for name in dir(raw_pto) if not name.startswith("_"))
    python_type_methods = _extract_python_type_methods(_read(PYTHON_BINDINGS))

    return {
        "source": "PTOAS",
        "canonical_frontend_type": "TileType",
        "internal_tile_buffer_type": "TileBufType",
        "ops": ops,
        "types": types,
        "attrs": attrs,
        "enums": enums,
        "python_symbols": python_symbols,
        "python_type_methods": python_type_methods,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate PTOAS capability manifest")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    manifest = build_manifest()
    data = json.dumps(manifest, indent=2, sort_keys=True)
    if args.output is None:
        print(data)
    else:
        args.output.write_text(data + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
