# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Compile each candidate from its own checkpoint copy through the native backend."""
from pathlib import Path
import hashlib
import tempfile
from collections import Counter

from ptoas._cv_materialize import materialize, serial_baseline
from ptoas._cv_memory import physical_memory
from ptoas._cv_completion import verify_completion
from ptoas.costmodel import run_native
from ptoas._loader import ensure_core
from ptoas._cv_ir import walk
from ptoas.mlir import ir
from pto_costmodel.wire import encode, fingerprint, read_text


def compiler_identity():
    core = Path(ensure_core().__file__).resolve()
    libraries = core.parent / "mlir" / "_mlir_libs"
    paths = [core] + sorted(p for p in libraries.iterdir() if p.suffix in (".so", ".dylib", ".dll"))
    components = {str(p.relative_to(core.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return dict(fingerprint=fingerprint(components), components=components)


def compile_candidate(graph, package, candidate, report, serial=False):
    text, trace = (serial_baseline if serial else materialize)(graph, package, candidate)
    with tempfile.TemporaryDirectory(prefix="pto-cv-compile-") as tmp:
        source, output = Path(tmp) / "candidate.pto", Path(tmp) / "candidate.cpp"
        lowered_path = Path(tmp) / "lowered.pto"
        source.write_text(text, encoding="utf-8")
        flags = ["--pto-arch=a5", "--pto-level=level2", "--enable-insert-sync"]
        run_native(flags + ["--cv-costmodel-final-ir-file=" + str(lowered_path), str(source), "-o", str(output)])
        generated = read_text(output)
        lowered = read_text(lowered_path)
        module = ir.Module.parse(lowered)
        summary = dict(Counter(op.name for op in walk(module.operation)))
        memory = physical_memory(module, package, candidate)
        completion = verify_completion(module, package, candidate, memory)
    report.update(status="compiled_baseline" if serial else "compiled_candidate", optimization_applied=not serial,
                  physical_feasibility="native_pipeline_passed", numerical_correctness="not_checked",
                  performance_certification="not_run", automatic_application=False,
                  re_evaluation_required=True)
    report["candidate_artifact_fingerprint"] = hashlib.sha256(generated.encode("utf-8")).hexdigest()
    compiler = compiler_identity()
    report["compiler_binary_fingerprint"] = compiler["fingerprint"]
    report["compiler_components"] = compiler["components"]
    report["native_flags"] = flags
    report["lowered_operation_counts"] = summary
    binding = dict(identity=package["identity"], candidate_id=candidate["candidate_id"],
                   schedule_fingerprint=candidate["schedule"]["fingerprint"],
                   compiler_binary_fingerprint=report["compiler_binary_fingerprint"],
                   compiler_components=compiler["components"],
                   cpp_fingerprint=report["candidate_artifact_fingerprint"],
                   final_ir_fingerprint=hashlib.sha256(lowered.encode("utf-8")).hexdigest(), native_flags=flags)
    memory["binding"] = binding
    memory["fingerprint"] = fingerprint(memory)
    validation = dict(schema_version="pto.compiler_validation.v1", binding=binding,
                      memory_plan_fingerprint=memory["fingerprint"], completion=completion,
                      physical_layout="pass", static_gm_bounds="pass", G2=completion["status"],
                      G3="not_run", G4="not_run")
    report["validation"]["G2"] = completion["status"]
    report["memory_plan_fingerprint"] = memory["fingerprint"]
    report["final_ir_fingerprint"] = binding["final_ir_fingerprint"]
    return {"candidate.pto": text, "candidate.cpp": generated, "lowered.pto": lowered,
            "materialization_trace.json": encode(trace), "memory_plan.json": encode(memory),
            "validation_report.json": encode(validation)}
