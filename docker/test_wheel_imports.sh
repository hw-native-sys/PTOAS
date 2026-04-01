#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Test wheel installation by verifying imports work correctly.
#
# Usage: ./test_wheel_imports.sh
#
# This script tests that the installed wheel can import:
#   - mlir.ir
#   - mlir.dialects.pto

set -e

echo "Testing wheel imports..."

# Test in a clean directory to avoid local imports
cd /tmp

echo "Testing mlir.ir import..."
python -c "import mlir.ir; print('mlir.ir imported successfully')"

echo "Testing pto dialect import..."
python -c "from mlir.dialects import pto; print('pto dialect imported successfully')"

echo "All wheel import tests passed!"
