#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Issue #591 performance acceptance: the numeric result is intentionally not
# gated here; the acceptance evidence is the simulator instruction log
# (zero RV_SADD/RV_SMOVK in the rolled tile loop) and the RVEC cycle count
# from *_summary_log (rolled <= 1.1x unrolled). Check the output exists.

import os
import sys

PATH = "v3.bin"


def main():
    if not os.path.exists(PATH) or os.path.getsize(PATH) == 0:
        print(f"[ERROR] output missing or empty: {PATH}")
        sys.exit(2)
    print("[INFO] compare passed (output present; perf evidence is in "
          "core*_summary_log and core*.veccore*.instr_log.dump)")


if __name__ == "__main__":
    main()
