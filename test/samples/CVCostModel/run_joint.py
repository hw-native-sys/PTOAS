# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Run the fixed micro through a real external adapter and retain all evidence.

This compiler-side script uses the explicitly supplied TileSim interpreter.
It does not perform device execution or claim that a candidate is profitable.
"""
import argparse
from pathlib import Path
import tempfile

import pto_costmodel
from ptoas._cv_cli import selected_plan
from ptoas._cv_exchange import export_v2
from ptoas._cv_session import apply_candidate, run_model
from pto_costmodel.package import read_package
from pto_costmodel.wire import encode, read_json


def run(args):
    sample = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=False)
    source, profile = sample / "four_stage_serial.pto", read_json(sample / "a5_profile.json")
    with tempfile.TemporaryDirectory(prefix="pto-joint-bindings-") as temporary:
        initial = Path(temporary) / "initial"
        export_v2(source, profile, initial)
        bindings = read_package(initial)["bindings"]
    bindings.update(alias_contract="disjoint", scenario="micro_distinct_Q_K_V_output_allocations")
    package = args.output / "package"
    export_v2(source, profile, package, bindings)
    sdk = Path(pto_costmodel.__file__).resolve().parent.parent
    adapter = dict(argv=[str(args.model_python.absolute()), "-m", "core.frontend.adaptor.ptoas_exchange"],
                   cwd=str(args.model_root.resolve()), environment=dict(PYTHONPATH=str(sdk)))
    (args.output / "adapter.json").write_text(encode(adapter), encoding="utf-8")
    selection = run_model(package, adapter, args.output / "model-run", [0, 1, 2, 3])
    # Explicitly inspect the P=2 candidate; model predictions have not selected it for deployment.
    candidate = selected_plan(args.output / "model-run" / "plan-2.json", read_package(package))
    annotations = apply_candidate(package, candidate["configuration"], args.output / "annotations")
    compilation = apply_candidate(package, candidate["configuration"], args.output / "compiled", "compile")
    summary = dict(selection=selection, annotations=annotations, compilation=compilation,
                   selection_reason="explicit P=2 integration fixture; not a performance selection",
                   device_execution="not_run", performance_certification="not_run")
    (args.output / "summary.json").write_text(encode(summary), encoding="utf-8")
    print(encode(summary), end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-python", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
