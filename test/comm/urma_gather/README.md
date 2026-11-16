<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# A5 URMA Gather

This standalone test compiles PTO IR with PTOAS, launches the generated device
functions on rank 0, and verifies the compacted data and trailing canary on each
receiving rank. It uses the PTO-ISA URMA workspace manager and MPI helper.

Use CANN Toolkit 9.1.0 or newer with A5 (3510) vector and URMA support, MPICH,
and at least two visible NPU devices on one machine. Source the CANN environment
before building. If multiple MPI libraries are installed, set `MPI_LIB_PATH` to
the MPICH library used by `mpirun`.
Use the default GitCode PTO-ISA revision from CI or a compatible newer revision.

```sh
cmake -S test/comm/urma_gather -B /tmp/ptoas-urma-scalar \
  -DCMAKE_CXX_COMPILER="$(command -v bisheng)" \
  -DPTOAS_BIN=/path/to/build/tools/ptoas/ptoas \
  -DPTO_ISA_ROOT=/path/to/pto-isa -DPTO_URMA_GATHER_PATH=scalar
cmake --build /tmp/ptoas-urma-scalar --parallel
mpirun -n 2 /tmp/ptoas-urma-scalar/urma_gather
mpirun -n 3 /tmp/ptoas-urma-scalar/urma_gather
```

Use a second build directory with `PTO_URMA_GATHER_PATH=simt` to verify the
same payload with forced SIMT filling. This mode runs the kernel with scratch
and requires `PTO_URMA_GATHER_SIMT_AVAILABLE`. Each rank verifies its local
result and returns a nonzero exit status on failure.

The test covers one and two destinations, discontiguous 4 KiB/8 KiB segments,
12/13/25 segment boundaries, repeated `NoCqe` and empty `Final` completion in
one session, rejection of a dynamic zero-count payload, deferred `OneCqe`
polling and completion, and a subsequent default-policy submission. The
`NoCqe` prefix, deferred `OneCqe`, and default immediate transfer use separate
receive regions. The test checks the payload and canary in each region.
