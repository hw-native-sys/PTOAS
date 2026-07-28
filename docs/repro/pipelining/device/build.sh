#!/usr/bin/env bash
# Build launchable CCE stages=2 scale host (relative CANN / PTOAS paths).
set -euo pipefail
DEVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${DEVICE_DIR}/../scripts/env.sh"

AICORE_ARCH="${PTO_AICORE_ARCH:-dav-c310}"
OUT="${DEVICE_DIR}/build"
mkdir -p "${OUT}"

echo "bisheng=${BISHENG}"
echo "ASCEND=${ASCEND}"

"${BISHENG}" -c -fPIC -O2 -std=c++17 -xcce \
  -Xhost-start -Xhost-end \
  -mllvm -cce-aicore-stack-size=0x8000 \
  -mllvm -cce-aicore-function-stack-size=0x8000 \
  -mllvm -cce-aicore-record-overflow=true \
  -mllvm -cce-aicore-addr-transform \
  -mllvm -cce-aicore-dcci-insert-for-scalar=false \
  --cce-aicore-arch="${AICORE_ARCH}" \
  -I"${ASCEND}/include" \
  -I"${ASCEND}/compiler/tikcpp/tikcfw" \
  -I"${ASCEND}/compiler/tikcpp/tikcfw/impl" \
  -I"${ASCEND}/compiler/tikcpp/tikcfw/interface" \
  "${DEVICE_DIR}/scale_stages2.asc" \
  -o "${OUT}/scale_stages2.o"

"${BISHENG}" -fPIC -shared --cce-fatobj-link \
  -o "${OUT}/libscale_stages2.so" \
  "${OUT}/scale_stages2.o" \
  -Wl,--no-as-needed -L"${ASCEND}/lib64" -lruntime

"${BISHENG}" -O2 -std=c++17 -xc++ \
  -I"${ASCEND}/include" \
  "${DEVICE_DIR}/main.cpp" \
  -o "${OUT}/scale_stages2_host" \
  -L"${OUT}" -L"${ASCEND}/lib64" \
  -lscale_stages2 -lascendcl -lruntime -ltiling_api -lplatform -lc_sec -ldl -lnnopbase \
  -Wl,-rpath,"${OUT}" -Wl,-rpath,"${ASCEND}/lib64"

echo "OK: ${OUT}/scale_stages2_host"
