#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Return a single-arch macOS wheel platform tag."""

import argparse
import sysconfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_arch", choices=["x86_64", "arm64"])
    args = parser.parse_args()

    plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    if plat.endswith("universal2"):
        plat = plat[: -len("universal2")] + args.target_arch
    print(plat)


if __name__ == "__main__":
    main()
