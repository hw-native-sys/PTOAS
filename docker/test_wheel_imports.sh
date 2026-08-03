#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Test wheel installation by verifying the installed Python contract works.
#
# Usage: ./test_wheel_imports.sh
#
# This script tests that the installed wheel can import:
#   - ptoas.mlir.ir
#   - ptoas.mlir.dialects.pto
#   - ptodsl
#   - from ptodsl import pto, scalar
#   - ptoas CLI entry
# and that a minimal PTODSL compile-only probe succeeds.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

WHEEL_GLOB="${PTO_TEST_WHEEL_PATH:-}"
if [[ -z "${WHEEL_GLOB}" && -n "${PTO_WHEEL_DIST_DIR:-}" ]]; then
  WHEEL_GLOB="${PTO_WHEEL_DIST_DIR}/ptoas*.whl"
fi
if [[ -z "${WHEEL_GLOB}" && -d "${REPO_ROOT}/build/wheel-dist" ]]; then
  WHEEL_GLOB="${REPO_ROOT}/build/wheel-dist/ptoas*.whl"
fi

TEST_TMPDIR=""
cleanup() {
  if [[ -n "${TEST_TMPDIR}" && -d "${TEST_TMPDIR}" ]]; then
    rm -rf "${TEST_TMPDIR}"
  fi
}
trap cleanup EXIT

echo "Testing wheel imports..."

if [[ -n "${WHEEL_GLOB}" ]] && compgen -G "${WHEEL_GLOB}" >/dev/null 2>&1; then
  TEST_WHEEL="$(compgen -G "${WHEEL_GLOB}" | sort | tail -n 1)"
  TEST_TMPDIR="$(mktemp -d /tmp/ptoas-wheel-test.XXXXXX)"
  echo "Installing wheel into isolated venv: ${TEST_WHEEL}"
  echo "Checking wheel payload before installation..."
  "${PYTHON_BIN}" "${REPO_ROOT}/docker/validate_wheel_payload.py" "${TEST_WHEEL}"
  "${PYTHON_BIN}" -m venv "${TEST_TMPDIR}/venv"
  source "${TEST_TMPDIR}/venv/bin/activate"
  python -m pip install --no-deps --force-reinstall "${TEST_WHEEL}"
  PYTHON_BIN="python"
else
  echo "No wheel artifact supplied; testing the currently installed environment."
fi

if [[ -z "${TEST_TMPDIR}" ]]; then
  TEST_TMPDIR="$(mktemp -d /tmp/ptoas-wheel-test.XXXXXX)"
fi

unset PYTHONPATH

# Test in a clean directory to avoid local imports
cd /tmp

PTOAS_ENTRYPOINT="$(command -v ptoas)"
PYTHON_ENTRYPOINT="$(command -v "${PYTHON_BIN}")"

echo "Testing ptoas.mlir.ir import..."
"$PYTHON_BIN" -c "from ptoas.mlir import ir; print('ptoas.mlir.ir imported successfully')"

echo "Testing pto dialect import..."
"$PYTHON_BIN" -c "from ptoas.mlir.dialects import pto; print('pto dialect imported successfully')"

echo "Testing that PTOAS does not claim the top-level mlir namespace..."
"$PYTHON_BIN" - <<'PY'
try:
    import mlir  # noqa: F401
except ImportError:
    pass
else:
    raise SystemExit("the ptoas wheel unexpectedly provides a top-level mlir package")
PY

echo "Testing ptodsl import..."
"$PYTHON_BIN" -c "import ptodsl; print(f'ptodsl imported successfully from {ptodsl.__file__}')"

echo "Testing ptodsl public imports..."
"$PYTHON_BIN" -c "from ptodsl import pto, scalar; print('ptodsl public imports imported successfully')"

echo "Testing installed ptoas console entry..."
"$PYTHON_BIN" -c "from ptoas import _core; print(f'ptoas native extension imported successfully from {_core.__file__}')"
PTOAS_PACKAGE_ROOT="$($PYTHON_BIN -c 'from pathlib import Path; from ptoas import _core; print(Path(_core.__file__).parent)')"
NATIVE_DEPENDENCY_ARGS=("${PTOAS_PACKAGE_ROOT}")
if [[ "${PTO_EXPECT_STATIC_LLVM:-0}" == "1" ]]; then
  NATIVE_DEPENDENCY_ARGS+=(--expect-static-llvm)
fi
"$PYTHON_BIN" "${REPO_ROOT}/docker/validate_installed_native_dependencies.py" \
  "${NATIVE_DEPENDENCY_ARGS[@]}"
PTOAS_VERSION_OUTPUT="$(ptoas --version | tr -d '\r')"
echo "${PTOAS_VERSION_OUTPUT}"
EXPECTED_PTOAS_CLI_VERSION="${PTOAS_CLI_VERSION:-${PTOAS_VERSION:-}}"
if [[ -n "${EXPECTED_PTOAS_CLI_VERSION}" ]]; then
  EXPECTED_VERSION_OUTPUT="ptoas ${EXPECTED_PTOAS_CLI_VERSION}"
  if [[ "${PTOAS_VERSION_OUTPUT}" != "${EXPECTED_VERSION_OUTPUT}" ]]; then
    echo "Error: expected '${EXPECTED_VERSION_OUTPUT}', got '${PTOAS_VERSION_OUTPUT}'" >&2
    exit 1
  fi
else
  echo "${PTOAS_VERSION_OUTPUT}" | grep -Eq '^ptoas [0-9]+\.[0-9]+$'
fi

echo "Testing installed ptoas console entry in a clean environment..."
CLEAN_ENV_DIR="${TEST_TMPDIR}/clean-env"
CLEAN_ENV_PTO="${CLEAN_ENV_DIR}/wheel-clean-env-probe.pto"
CLEAN_ENV_PTO_IR="${CLEAN_ENV_DIR}/wheel-clean-env-probe.pto.ir"
CLEAN_ENV_LOG="${CLEAN_ENV_DIR}/wheel-clean-env-probe.log"
mkdir -p "${CLEAN_ENV_DIR}"

CLEAN_PTOAS_VERSION_OUTPUT="$(
  env -i \
    HOME="${CLEAN_ENV_DIR}" \
    TMPDIR="${CLEAN_ENV_DIR}" \
    PATH="/usr/bin:/bin" \
    "${PTOAS_ENTRYPOINT}" --version | tr -d '\r'
)"
echo "${CLEAN_PTOAS_VERSION_OUTPUT}"
if [[ -n "${EXPECTED_PTOAS_CLI_VERSION}" ]]; then
  EXPECTED_VERSION_OUTPUT="ptoas ${EXPECTED_PTOAS_CLI_VERSION}"
  if [[ "${CLEAN_PTOAS_VERSION_OUTPUT}" != "${EXPECTED_VERSION_OUTPUT}" ]]; then
    echo "Error: expected '${EXPECTED_VERSION_OUTPUT}', got '${CLEAN_PTOAS_VERSION_OUTPUT}'" >&2
    exit 1
  fi
else
  echo "${CLEAN_PTOAS_VERSION_OUTPUT}" | grep -Eq '^ptoas [0-9]+\.[0-9]+$'
fi

env -i \
  HOME="${CLEAN_ENV_DIR}" \
  TMPDIR="${CLEAN_ENV_DIR}" \
  PATH="/usr/bin:/bin" \
  CLEAN_ENV_PTO="${CLEAN_ENV_PTO}" \
  "${PYTHON_ENTRYPOINT}" - <<'PY'
import os
from pathlib import Path

from ptodsl import pto


@pto.jit(target="a5")
def wheel_clean_env_probe(
    a_ptr: pto.ptr(pto.f32, "gm"),
    o_ptr: pto.ptr(pto.f32, "gm"),
):
    a_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32, addr=0)
    o_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32, addr=0)
    a_view = pto.make_tensor_view(
        a_ptr,
        shape=[1, 16],
        strides=[16, 1],
    )
    o_view = pto.make_tensor_view(
        o_ptr,
        shape=[1, 16],
        strides=[16, 1],
    )
    pto.tile.load(a_view, a_tile)
    pto.tile.store(o_tile, o_view)


output_path = Path(os.environ["CLEAN_ENV_PTO"])
mlir_text = wheel_clean_env_probe.compile().mlir_text()
if "func.func @wheel_clean_env_probe" not in mlir_text:
    raise SystemExit("PTODSL clean-environment probe did not preserve the kernel symbol")
if "pto.tload" not in mlir_text or "pto.tstore" not in mlir_text:
    raise SystemExit("PTODSL clean-environment probe did not emit the expected tile ops")
output_path.write_text(mlir_text, encoding="utf-8")
print(f"Wrote PTODSL-authored PTO IR to {output_path}")
PY

if ! env -i \
  HOME="${CLEAN_ENV_DIR}" \
  TMPDIR="${CLEAN_ENV_DIR}" \
  PATH="/usr/bin:/bin" \
  "${PTOAS_ENTRYPOINT}" --pto-arch=a5 --pto-backend=vpto --pto-level=level3 --enable-tile-op-expand --emit-pto-ir "${CLEAN_ENV_PTO}" -o "${CLEAN_ENV_PTO_IR}" \
  >"${CLEAN_ENV_LOG}" 2>&1
then
  echo "Error: clean-environment ptoas smoke failed; log follows:" >&2
  if [[ -f "${CLEAN_ENV_LOG}" ]]; then
    cat "${CLEAN_ENV_LOG}" >&2
  fi
  exit 1
fi

if [[ ! -s "${CLEAN_ENV_PTO_IR}" ]]; then
  echo "Error: clean-environment ptoas smoke did not produce ${CLEAN_ENV_PTO_IR}" >&2
  exit 1
fi
grep -q "wheel_clean_env_probe" "${CLEAN_ENV_PTO_IR}" || {
  echo "Error: clean-environment ptoas smoke output is missing wheel_clean_env_probe" >&2
  exit 1
}
grep -q "pto.tload" "${CLEAN_ENV_PTO_IR}" || {
  echo "Error: clean-environment ptoas smoke output is missing pto.tload" >&2
  exit 1
}
grep -q "pto.tstore" "${CLEAN_ENV_PTO_IR}" || {
  echo "Error: clean-environment ptoas smoke output is missing pto.tstore" >&2
  exit 1
}
grep -q "candidates = " "${CLEAN_ENV_PTO_IR}" || {
  echo "Error: clean-environment ptoas smoke output is missing TileLib candidate metadata" >&2
  exit 1
}
echo "All wheel import tests passed!"
