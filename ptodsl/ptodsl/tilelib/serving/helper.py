# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""One-shot command-line client for the ExpandTileOp daemon contract.

Example:

    python3 -m ptodsl.tilelib.serving.helper --socket <path> --target a5 \
        --op pto.tadd --operand-specs '[...]'
"""

from __future__ import annotations

import argparse
import json
import sys

from .client import DaemonClient, DaemonError


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ptodsl.tilelib.serving.helper")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--op", required=True)
    parser.add_argument("--operand-specs", required=True)
    parser.add_argument("--context-attrs", default=None)
    parser.add_argument(
        "--method",
        choices=("instantiate", "get_metadata"),
        default="instantiate",
    )
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument(
        "--include-vmi-candidates",
        action="store_true",
        help="Include internal VMI TileLib candidates in metadata responses.",
    )
    args = parser.parse_args(argv)

    try:
        operand_specs = json.loads(args.operand_specs)
        context_attrs = json.loads(args.context_attrs) if args.context_attrs else {}
    except json.JSONDecodeError as exc:
        parser.error(f"invalid JSON input: {exc}")

    try:
        client = DaemonClient(args.socket)
        if args.method == "get_metadata":
            result = client.get_metadata(
                args.target,
                args.op,
                operand_specs,
                context_attrs,
                include_vmi_candidates=args.include_vmi_candidates,
            )
            sys.stdout.write(json.dumps(result))
            return

        result = client.instantiate(
            args.target,
            args.op,
            operand_specs,
            context_attrs,
            args.candidate_id,
        )
    except (DaemonError, OSError) as exc:
        sys.stderr.write(f"Error: daemon RPC failed: {exc}\n")
        raise SystemExit(1) from exc

    sys.stdout.write(result)


if __name__ == "__main__":
    main()
