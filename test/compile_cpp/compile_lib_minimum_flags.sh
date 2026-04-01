# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

bisheng \
    -I${PTO_LIB_PATH}/include \
    -fPIC -shared -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    ./abs_vec_core.cpp \
    -o ./abs_kernel_min_flg.so

# NOTE: with CANN installation, even the `-I${PTO_LIB_PATH}/include` can be omitted
# because `$ASCEND_TOOLKIT_HOME/include/pto` also contains PTO headers (not most recent version)
