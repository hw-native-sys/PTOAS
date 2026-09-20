# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Deterministic protocol adapter; deliberately makes no performance prediction."""
import argparse

from pto_costmodel.contract import capabilities, envelope, header
from pto_costmodel.package import read_package
from pto_costmodel.wire import encode, fingerprint, read_json, require
from pto_costmodel.validation import validate_request


def model_info():
    return dict(name="reference", revision="1", adapter_version="2.0", config_fingerprint=fingerprint({}))


def evaluate(package, request):
    validate_request(request, package)
    rows = []
    for candidate in request["candidates"]:
        rows.append(dict(candidate_id=candidate["candidate_id"], configuration=candidate["configuration"],
                         schedule_fingerprint=candidate["schedule"]["fingerprint"],
                         latency=dict(value=None, unit="us"), resources=candidate["resources"],
                         coverage=dict(status="reference", operations="not_modeled", communication="not_modeled",
                                       approximations=[], unsupported=[]),
                         evidence=dict(buffer_requirements=candidate["schedule"]["buffer_requirements"])))
    return dict(**header("result"), identity=package["identity"], candidates=rows,
                model=model_info())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capabilities", "evaluate", "propose"))
    parser.add_argument("--package")
    parser.add_argument("--request")
    args = parser.parse_args()
    if args.action == "capabilities":
        result = capabilities("reference")
        result["model"] = model_info()
    else:
        result = evaluate(read_package(args.package), read_json(args.request))
    print(encode(result), end="")


if __name__ == "__main__":
    main()
