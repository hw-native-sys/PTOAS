# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Emit ranked candidate metadata for the InsertTemplateAttributes lit test."""

import argparse
import json


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--method", default="instantiate")
parser.add_argument("--candidate-id")
args, _ = parser.parse_known_args()

if args.method == "get_metadata":
    print(
        json.dumps(
            {
                "target": "a5",
                "op": "pto.tadd",
                "candidates": [
                    {
                        "id": 99,
                        "name": "preferred_1d",
                        "loop_depth": 1,
                        "is_post_update": False,
                        "has_tail": True,
                    },
                    {
                        "id": 0,
                        "name": "fallback_2d",
                        "loop_depth": 2,
                        "is_post_update": False,
                        "has_tail": False,
                    },
                ],
            }
        ),
        end="",
    )
else:
    candidate = args.candidate_id or "missing_candidate"
    print(
        f"""
module {{
  func.func @{candidate}(
      %src0: !pto.tile_buf<vec, 8x64xf32>,
      %src1: !pto.tile_buf<vec, 8x64xf32>,
      %dst: !pto.tile_buf<vec, 8x64xf32>)
      attributes {{pto.tilelang.instance,
                  pto.kernel_kind = #pto.kernel_kind<vector>}} {{
    return
  }}
}}
""",
        end="",
    )
