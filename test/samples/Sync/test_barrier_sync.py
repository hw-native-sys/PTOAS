# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

if __name__ == "__main__":
    # Regression for zhangstevenunity/PTOAS#185:
    # barrier_sync should support any SyncOpType that can be mapped to a PIPE
    # (not just TMATMUL/TVEC).
    print(
        r"""module {
  func.func @test_barrier_sync_py() {
    pto.barrier_sync[<TLOAD>]
    pto.barrier_sync[<TSTORE_VEC>]
    pto.barrier_sync[<TVEC>]
    return
  }
}
"""
    )
