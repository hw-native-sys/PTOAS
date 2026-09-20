# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""G2 feedback failures and iteration-distinct independent runtime fixtures."""
from copy import deepcopy
from pathlib import Path
import hashlib
import sys
import tempfile
import unittest
from unittest.mock import patch

from ptoas.mlir import ir
from ptoas.mlir.dialects import pto
from ptoas._cv_exchange import export_v2
from ptoas._cv_session import apply_candidate
from ptoas._cv_completion import _compact_proofs, verify_completion
from ptoas._cv_memory import physical_memory, _usage
from ptoas._cv_ir import walk
from pto_costmodel.contract import configuration
from pto_costmodel.package import read_package
from pto_costmodel.wire import encode, read_json

SAMPLE = Path(__file__).resolve().parents[3] / "samples" / "CVCostModel"
sys.path.insert(0, str(SAMPLE))
from prepare_runtime import prepare_case
from runtime_fixture import input_arrays, source_text
from build_runtime import verify_artifacts


class FeedbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="pto-g2-test-")
        cls.root = Path(cls.temporary.name)
        prepare_case(cls.root, 4, True, "g2-test")
        cls.case = cls.root / "crossing-n4"
        cls.variant = cls.case / "p2"
        cls.package = read_package(cls.case / "package")
        cls.candidate = read_json(cls.variant / "candidate.json")
        cls.memory = read_json(cls.variant / "memory_plan.json")
        cls.final_ir = (cls.variant / "lowered.pto").read_text()
        files = {str(p.relative_to(cls.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in cls.root.rglob("*") if p.is_file()}
        (cls.root / "manifest.json").write_text(encode(dict(files=files)))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def verify(self, mutate=None, candidate=None, memory=None):
        with ir.Context() as context:
            pto.register_dialect(context, load=True)
            module = ir.Module.parse(self.final_ir)
            if mutate:
                mutate(module)
            return verify_completion(module, self.package, candidate or self.candidate, memory or self.memory)

    def test_completion_and_three_live_slots(self):
        self.assertEqual(self.verify()["status"], "pass")
        requirements = self.candidate["schedule"]["buffer_requirements"]
        self.assertIn(3, [v for k, v in requirements.items() if k.startswith("vector.")])
        key = next(k for k, v in requirements.items() if k.startswith("vector.") and v == 3)
        rows = [r for r in self.memory["allocations"] if r["source_buffer_id"] == key]
        self.assertEqual({r["slot"] for r in rows}, {0, 1, 2})
        self.assertTrue(all(r["offset_bytes"] is not None for r in rows))

    def test_same_compile_and_replay(self):
        from ptoas import _cv_compile
        with patch.object(_cv_compile, "run_native", wraps=_cv_compile.run_native) as native:
            apply_candidate(self.case / "package", self.candidate["configuration"], self.root / "replay", "compile")
        self.assertEqual(native.call_count, 1)
        self.assertTrue(any("--cv-costmodel-final-ir-file=" in a for a in native.call_args[0][0]))
        self.assertEqual((self.root / "replay" / "candidate.cpp").read_text(),
                         (self.variant / "candidate.cpp").read_text())
        self.assertEqual(read_json(self.root / "replay" / "memory_plan.json"), self.memory)

    def test_missing_completion_rejected(self):
        def remove_flags(module):
            for op in list(walk(module.operation)):
                if op.name in ("pto.set_flag", "pto.wait_flag"):
                    op.erase()
        report = self.verify(remove_flags)
        self.assertEqual(report["status"], "unknown")
        self.assertIn("UNPROVEN_COMPLETION", {e["code"] for e in report["unresolved"]})

    def test_wrong_slot_rejected(self):
        def wrong_slot(module):
            for op in walk(module.operation):
                if op.name == "pto.alloc_tile" and "pto.costmodel.buffer_id" in op.attributes:
                    op.attributes["pto.costmodel.slot"] = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 99)
                    break
        self.assertEqual(self.verify(wrong_slot)["status"], "fail")

    def test_wrong_transaction_split_rejected(self):
        def wrong_split(module):
            for op in walk(module.operation):
                if op.name == "pto.tpush":
                    op.attributes["split"] = ir.IntegerAttr.get(ir.IntegerType.get_signless(8), 0)
                    break
        report = self.verify(wrong_split)
        self.assertEqual(report["status"], "fail")
        self.assertIn("TRANSACTION_MAPPING", {e["code"] for e in report["unresolved"]})

    def test_reuse_wait_cycle_rejected(self):
        candidate = deepcopy(self.candidate)
        raw = next(v for v in candidate["schedule"]["buffer_versions"] if v["readers"])
        candidate["schedule"]["dependencies"].append(
            dict(source=raw["readers"][0], target=raw["writer"], kind="fifo_capacity"))
        report = self.verify(candidate=candidate)
        self.assertEqual(report["status"], "fail")
        self.assertIn("COMPLETION_CYCLE", {e["code"] for e in report["unresolved"]})

    def test_overlapping_live_slots_rejected(self):
        memory = deepcopy(self.memory)
        vector = [r for r in memory["allocations"] if r["owner"] == "local" and r["memory_space"] == "vec"]
        for row in vector:
            row["offset_bytes"] = vector[0]["offset_bytes"]
        report = self.verify(memory=memory)
        self.assertEqual(report["status"], "fail")
        self.assertIn("LIVE_STORAGE_OVERLAP", {e["code"] for e in report["unresolved"]})

    def test_two_aiv_domains_and_legal_reuse(self):
        row = next(r for r in self.memory["allocations"] if r["owner"] == "local" and r["memory_space"] == "vec")
        alias = dict(row, source_buffer_id="different-logical-buffer", allocation_id="different")
        usage = _usage([row, alias], self.package["target"]["compiler_budget"])
        self.assertEqual({u["core"] for u in usage}, {"AIV0", "AIV1"})
        self.assertTrue(all(u["union_reserved_bytes"] == row["reserved_bytes"] for u in usage))

    def test_native_legal_reuse_keeps_identities(self):
        from ptoas._cv_ir import attr
        prepare_case(self.root, 1, False, "reuse-proof")
        case = self.root / "basic-n1"
        package = read_package(case / "package")
        candidate = read_json(case / "p0" / "candidate.json")
        text = (case / "p0" / "lowered.pto").read_text()
        with ir.Context() as context:
            pto.register_dialect(context, load=True)
            module = ir.Module.parse(text)
            locals_ = [op for op in walk(module.operation) if op.name == "pto.alloc_tile"
                       and str(attr(op, "pto.costmodel.buffer_id", "")).startswith("vector.")]
            first, last = locals_[0], locals_[-1]
            last.operands[0] = first.operands[0]
            memory = physical_memory(module, package, candidate)
            report = verify_completion(module, package, candidate, memory)
            a, c = attr(first, "pto.costmodel.buffer_id"), attr(last, "pto.costmodel.buffer_id")
        self.assertEqual(report["status"], "pass")
        rows = [r for r in memory["allocations"] if r["source_buffer_id"] in (a, c)]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["offset_bytes"], rows[1]["offset_bytes"])
        usage = _usage(rows, package["target"]["compiler_budget"])
        self.assertTrue(all(u["union_reserved_bytes"] == rows[0]["reserved_bytes"] for u in usage))

    def test_feedback_tamper_rejected(self):
        verify_artifacts(self.variant)
        path = self.variant / "validation_report.json"
        original = path.read_text()
        try:
            report = read_json(path)
            report["binding"]["candidate_id"] = "stale"
            path.write_text(encode(report))
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                verify_artifacts(self.variant)
        finally:
            path.write_text(original)

    def test_large_completion_evidence_is_bounded_and_bound(self):
        proofs = [dict(source=f"s{i}", target=f"t{i}", kind="physical_overlap") for i in range(5000)]
        retained, summary = _compact_proofs(proofs)
        self.assertEqual(len(retained), 4096)
        self.assertEqual(summary["total"], 5000)
        self.assertEqual(summary["by_kind"], {"physical_overlap": 5000})
        self.assertTrue(summary["truncated"])
        changed = [dict(row) for row in proofs]
        changed[-1]["target"] = "different"
        self.assertNotEqual(summary["sha256"], _compact_proofs(changed)[1]["sha256"])

    def test_native_memory_replay(self):
        with ir.Context() as context:
            pto.register_dialect(context, load=True)
            memory = physical_memory(ir.Module.parse(self.final_ir), self.package, self.candidate)
        self.assertEqual(memory["allocations"], self.memory["allocations"])
        self.assertEqual(memory["usage"], self.memory["usage"])

    def test_undersized_native_nz_layout_rejected(self):
        profile = read_json(SAMPLE / "a5_profile.json")
        export_v2(SAMPLE / "four_stage_serial.pto", profile, self.root / "tiny")
        bindings = read_package(self.root / "tiny")["bindings"]
        bindings["alias_contract"] = "disjoint"
        export_v2(SAMPLE / "four_stage_serial.pto", profile, self.root / "tiny-bound", bindings)
        package = read_package(self.root / "tiny-bound")
        report = apply_candidate(self.root / "tiny-bound", configuration(package["program"], 0),
                                 self.root / "tiny-out", "compile_serial")
        self.assertEqual(report["validation"]["G2"], "fail")
        validation = read_json(self.root / "tiny-out" / "validation_report.json")
        self.assertIn("NATIVE_LAYOUT_OVERFLOW", {e["code"] for e in validation["completion"]["unresolved"]})

    def test_native_library_changes_compiler_identity(self):
        from types import SimpleNamespace
        from ptoas import _cv_compile
        root = self.root / "identity-test"
        libraries = root / "mlir" / "_mlir_libs"
        libraries.mkdir(parents=True)
        core = root / "_core.so"
        core.write_bytes(b"unchanged-python-entry")
        backend = libraries / "libPTOASCompiler.so"
        backend.write_bytes(b"backend-v1")
        with patch.object(_cv_compile, "ensure_core", return_value=SimpleNamespace(__file__=str(core))):
            before = _cv_compile.compiler_identity()
            backend.write_bytes(b"backend-v2")
            after = _cv_compile.compiler_identity()
        self.assertNotEqual(before["fingerprint"], after["fingerprint"])

    def test_empty_compile_without_model(self):
        source = self.root / "empty.pto"
        source.write_text(source_text(0))
        profile = read_json(SAMPLE / "a5_profile.json")
        export_v2(source, profile, self.root / "empty-inferred")
        bindings = read_package(self.root / "empty-inferred")["bindings"]
        bindings["alias_contract"] = "disjoint"
        export_v2(source, profile, self.root / "empty-package", bindings)
        package = read_package(self.root / "empty-package")
        report = apply_candidate(self.root / "empty-package", configuration(package["program"], 0),
                                 self.root / "empty-compiled", "compile")
        self.assertEqual(report["validation"]["G2"], "pass")
        candidate = read_json(self.root / "empty-compiled" / "candidate.json")
        self.assertFalse(candidate["schedule"]["events"])

    def test_independent_golden_detects_versions(self):
        import numpy as np
        for crossing in (False, True):
            for seed in (0, 1, 2):
                data = input_arrays(4, seed, crossing)
                for name in ("q", "k", "v", "golden"):
                    self.assertFalse(np.array_equal(data[name][0], data[name][1]))
                self.assertFalse(np.array_equal(data["golden"], np.roll(data["golden"], 1, axis=0)))
                for i in range(4):
                    q, k, v = [data[name][i].astype(np.int64) for name in ("q", "k", "v")]
                    # Scalar dot products independently check the NumPy batch expression.
                    p00 = -sum(int(q[0,j])*int(k[j,0]) for j in range(32))
                    pv00 = sum(-sum(int(q[0,j])*int(k[j,t]) for j in range(32))*int(v[t,0]) for t in range(32))
                    self.assertEqual(data["golden"][i,0,0], pv00+p00 if crossing else 2*pv00)


if __name__ == "__main__":
    unittest.main()
