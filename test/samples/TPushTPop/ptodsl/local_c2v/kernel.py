# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL local C2V TPush/Tpop sample for A5.

This mirrors the local FIFO shape used by ``test/samples/TPushTPop/test1``:
the Cube side imports the Vector-owned FIFO buffer, pushes one tile, and the
SIMD side pops/frees the tile before storing it back to GM.
"""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import List, Optional, Tuple

from ptodsl import pto


_ROWS = 16
_COLS = 16
_SLOT_SIZE = _ROWS * _COLS * 4
_FIFO_SIZE = _SLOT_SIZE * 8


@pto.jit(name="ptodsl_tpush_tpop_local_c2v_cube", target="a5", kernel_kind="cube")
def ptodsl_tpush_tpop_local_c2v_cube(
    A_ptr: pto.ptr(pto.f32, "gm"),
):
    c0 = pto.const(0)
    c1 = pto.const(1)
    c_rows = pto.const(_ROWS)
    c_cols = pto.const(_COLS)

    a_view = pto.make_tensor_view(A_ptr, shape=[c_rows, c_cols], strides=[c_cols, c1])
    a_part = pto.partition_view(a_view, offsets=[c0, c0], sizes=[c_rows, c_cols])

    # The standalone Cube compile needs a local peer declaration to satisfy
    # frontend verification. emit_mlir() removes this compile-only reserve and
    # rewrites the peer to the real Vector function in the merged sample module.
    pto.reserve_buffer("c2v_fifo", size=_FIFO_SIZE, location="vec")
    c2v_import = pto.import_reserved_buffer(
        "c2v_fifo",
        peer_func="ptodsl_tpush_tpop_local_c2v_cube",
    )
    c2v = pto.pipe.c2v(
        slot_size=_SLOT_SIZE,
        consumer_buf=c2v_import,
        id=0,
    )
    src_tile = pto.alloc_tile(shape=[_ROWS, _COLS], dtype=pto.f32)
    pto.tile.load(a_part, src_tile)
    with pto.tileop():
        c2v.init_cube()
        c2v.push(src_tile, split=0)


@pto.jit(name="ptodsl_tpush_tpop_local_c2v_vector", target="a5", kernel_kind="vector")
def ptodsl_tpush_tpop_local_c2v_vector(
    O_ptr: pto.ptr(pto.f32, "gm"),
):
    c0 = pto.const(0)
    c1 = pto.const(1)
    c_rows = pto.const(_ROWS)
    c_cols = pto.const(_COLS)

    o_view = pto.make_tensor_view(O_ptr, shape=[c_rows, c_cols], strides=[c_cols, c1])
    o_part = pto.partition_view(o_view, offsets=[c0, c0], sizes=[c_rows, c_cols])

    c2v_buf = pto.reserve_buffer("c2v_fifo", size=_FIFO_SIZE, location="vec")
    c2v = pto.pipe.c2v(
        slot_size=_SLOT_SIZE,
        consumer_buf=c2v_buf,
        id=0,
    )
    dst_type = pto.alloc_tile(shape=[_ROWS, _COLS], dtype=pto.f32)
    with pto.tileop():
        c2v.init_simd()
        fifo_tile = c2v.pop(result_type=dst_type, split=0)
        pto.tile.store(fifo_tile, o_part)
        c2v.free(split=0)


def emit_mlir() -> str:
    cube_text = _function_module_body(
        ptodsl_tpush_tpop_local_c2v_cube.compile().mlir_text(),
        kernel_kind="cube",
        drop_compile_only_peer_reserve=True,
        peer_rewrite=(
            "peer_func = @ptodsl_tpush_tpop_local_c2v_cube",
            "peer_func = @ptodsl_tpush_tpop_local_c2v_vector",
        ),
    )
    vector_text = _function_module_body(
        ptodsl_tpush_tpop_local_c2v_vector.compile().mlir_text(),
        kernel_kind="vector",
    )
    return "module {\n" + cube_text + "\n" + vector_text + "\n}\n"


def _function_module_body(
    mlir_text: str,
    *,
    kernel_kind: str,
    peer_rewrite: Optional[Tuple[str, str]] = None,
    drop_compile_only_peer_reserve: bool = False,
) -> str:
    lines = [line for line in mlir_text.strip().splitlines()]
    if not lines or not lines[0].startswith("module attributes"):
        raise ValueError("expected a PTODSL single-function module")
    if lines[-1] != "}":
        raise ValueError("expected a PTODSL module closing brace")

    body = "\n".join(lines[1:-1])
    body = body.replace(
        "attributes {pto.aicore}",
        f"attributes {{pto.aicore, pto.kernel_kind = #pto.kernel_kind<{kernel_kind}>}}",
        1,
    )
    if drop_compile_only_peer_reserve:
        body = _drop_compile_only_peer_reserve(body)
    if peer_rewrite is not None:
        old, new = peer_rewrite
        body = body.replace(old, new, 1)
    return body


def _drop_compile_only_peer_reserve(body: str) -> str:
    return re.sub(
        r"\n    %\d+ = pto\.reserve_buffer\{name = \"c2v_fifo\", size = \d+, "
        r"location = <vec>, auto = true\} -> i32",
        "",
        body,
        count=1,
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _resolve_ptoas() -> Path:
    root = _repo_root()
    candidates = [
        root / "build" / "tools" / "ptoas" / "ptoas",
        root / "install" / "bin" / "ptoas",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    from_path = shutil.which("ptoas")
    if from_path:
        return Path(from_path)
    raise FileNotFoundError("unable to locate ptoas under build/, install/, or PATH")


def verify_ptoas() -> str:
    mlir_text = emit_mlir()
    ptoas = _resolve_ptoas()
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False, encoding="utf-8") as handle:
        handle.write(mlir_text)
        input_path = Path(handle.name)

    try:
        result = subprocess.run(
            [str(ptoas), "--pto-arch=a5", str(input_path), "--emit-pto-ir", "-o", "-"],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        input_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(
            "ptoas frontend verification failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-mlir", action="store_true", help="print the PTODSL-generated MLIR")
    parser.add_argument("--verify-ptoas", action="store_true", help="run PTOAS frontend verification")
    args = parser.parse_args(argv)

    if args.emit_mlir:
        print(emit_mlir())
        return 0

    if args.verify_ptoas:
        verify_ptoas()
        print("PASS ptodsl_tpush_tpop_local_c2v ptoas frontend")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
