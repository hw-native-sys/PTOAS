# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""G1 tests against the real native checkpoint and MLIR-backed exchange API."""
from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ptoas._cv_common import ContractError, read_json
from ptoas.costmodel import export_package, import_plan

FIXTURES = Path(__file__).resolve().parents[3] / "samples" / "CVCostModel"


class ExchangeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix="cv-contract-tests-")
        cls.root = Path(cls.workspace.name)
        cls.profile = read_json(FIXTURES / "a5_profile.json")
        cls.source = FIXTURES / "four_stage_serial.pto"
        cls.package = cls.root / "package"
        export_package(cls.source, cls.profile, {}, cls.package)
        cls.manifest = read_json(cls.package / "manifest.json")
        cls.plan = cls.make_plan()

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    @classmethod
    def make_plan(cls):
        manifest = cls.manifest
        keys = ("schema_version", "input_fingerprint", "target_profile_fingerprint",
                "bindings_fingerprint", "pipeline_id", "schedule_kind", "preload_semantics", "local_schedule")
        plan = {key: manifest[key] for key in keys}
        plan["preload_count"] = 2
        plan["buffers"] = [dict(buffer_id=b["id"], count=1) for b in manifest["buffers"]
                           if b["multi_buffer_eligible"]]
        return plan

    def output(self, suffix="result"):
        return self.root / (self.id().split(".")[-1] + "-" + suffix)

    def reject(self, plan, code, **kwargs):
        output = self.output("rejected")
        with self.assertRaises(ContractError) as caught:
            import_plan(self.package, plan, output, **kwargs)
        self.assertEqual(caught.exception.code, code, str(caught.exception))
        self.assertFalse(output.exists())

    def test_c01_expected_graph(self):
        expected = read_json(FIXTURES / "expected_graph.json")
        manifest = self.manifest
        self.assertEqual([row["trip_count"] for row in manifest["loops"]], expected["trip_counts"])
        self.assertEqual(len(manifest["pipes"]), expected["logical_pipes"])
        self.assertEqual(sum(p["core_expanded_edges"] for p in manifest["pipes"]), expected["core_expanded_edges"])
        self.assertEqual([t["id"] for t in manifest["tasks"]], expected["tasks"])
        for scope in ("cube", "vector"):
            local = [b for b in manifest["buffers"] if b["owner"] == "local" and b["source_scope"] == scope]
            self.assertEqual(len(local), expected["local_buffers"][scope])
            self.assertEqual([b["allocation_bytes"] for b in local], expected["local_allocation_bytes"][scope])
        self.assertEqual(sum(b["owner"] == "borrowed_entry" for b in manifest["buffers"]), 3)
        self.assertEqual(sum(b["owner"] == "pipe_backing" for b in manifest["buffers"]), 3)
        self.assertEqual(len(manifest["transactions"]), 9)
        self.assertEqual({(p["producer_task_id"], p["consumer_task_id"]) for p in manifest["pipes"]},
                         {("C_QK", "V_P"), ("V_P", "C_PV"), ("C_PV", "V_O")})
        for pipe in manifest["pipes"]:
            self.assertEqual((pipe["split"], pipe["effective_slot_num"], pipe["slot_size_bytes"]), (1, 4, 1024))
        for scope, size in expected["reserved_bytes"].items():
            self.assertEqual(sum(b["allocation_bytes"] for b in manifest["buffers"]
                                 if b["owner"] == "pipe_backing" and b["source_scope"] == scope), size)

    def test_c02_ssa_rename(self):
        source = self.output("renamed.pto")
        source.write_text(self.source.read_text(encoding="utf-8").replace("%a", "%renamed_a"), encoding="utf-8")
        package = self.output("package")
        export_package(source, self.profile, {}, package)
        self.assertEqual(read_json(package / "manifest.json"), self.manifest)

    def test_c03_counts_and_no_transformation(self):
        plan = copy.deepcopy(self.plan)
        vector_ids = [b["id"] for b in self.manifest["buffers"]
                      if b["owner"] == "local" and b["source_scope"] == "vector"]
        for row in plan["buffers"]:
            if row["buffer_id"] in vector_ids:
                row["count"] = vector_ids.index(row["buffer_id"]) + 1
        output = self.output()
        report = import_plan(self.package, plan, output)
        text = (output / "annotated.pto").read_text(encoding="utf-8")
        self.assertEqual(report["status"], "annotation_only")
        self.assertFalse(report["optimization_applied"])
        self.assertNotIn("pto.alloc_multi_tile", text)
        for count in (1, 2, 3):
            self.assertIn(f"pto.pipeline.multi_buffer_count = {count} : i64", text)
        self.assertEqual(text.count("scf.for"), 2)
        from ptoas.mlir import ir
        from ptoas.mlir.dialects import pto

        with ir.Context() as context:
            pto.register_dialect(context, load=True)
            module = ir.Module.parse(text)
            actual = {}
            for func in module.body.operations:
                for op in func.regions[0].blocks[0].operations:
                    if op.operation.name == "pto.alloc_tile":
                        key = ir.StringAttr(op.attributes["pto.costmodel.buffer_id"]).value
                        actual[key] = ir.IntegerAttr(op.attributes["pto.pipeline.multi_buffer_count"]).value
            self.assertEqual(actual, {r["buffer_id"]: r["count"] for r in plan["buffers"]})

    def test_c03_idempotent(self):
        first = self.output("first")
        second = self.output("second")
        import_plan(self.package, self.plan, first)
        import_plan(self.package, self.plan, second, current_input=first / "annotated.pto")
        self.assertEqual((first / "annotated.pto").read_bytes(), (second / "annotated.pto").read_bytes())

    def test_metrics_do_not_change_annotations(self):
        plan = copy.deepcopy(self.plan)
        first = self.output("first")
        second = self.output("second")
        import_plan(self.package, plan, first)
        plan["metrics"] = dict(latency_ns=100)
        import_plan(self.package, plan, second)
        self.assertEqual((first / "annotated.pto").read_bytes(), (second / "annotated.pto").read_bytes())

    def test_unknown_id(self):
        plan = copy.deepcopy(self.plan)
        plan["buffers"][0]["buffer_id"] = "absent"
        self.reject(plan, "UNKNOWN_ID")

    def test_duplicate_id(self):
        plan = copy.deepcopy(self.plan)
        plan["buffers"].append(plan["buffers"][0])
        self.reject(plan, "DUPLICATE_ID")

    def test_missing_buffer(self):
        plan = copy.deepcopy(self.plan)
        plan["buffers"].pop()
        self.reject(plan, "MISSING_BUFFER")

    def test_zero_count(self):
        plan = copy.deepcopy(self.plan)
        plan["buffers"][0]["count"] = 0
        self.reject(plan, "RANGE")

    def test_bool_count(self):
        plan = copy.deepcopy(self.plan)
        plan["buffers"][0]["count"] = True
        self.reject(plan, "RANGE")

    def test_oversized_count(self):
        plan = copy.deepcopy(self.plan)
        plan["buffers"][0]["count"] = 1 << 40
        self.reject(plan, "RANGE")

    def test_negative_preload(self):
        plan = copy.deepcopy(self.plan)
        plan["preload_count"] = -1
        self.reject(plan, "RANGE")

    def test_unknown_schedule(self):
        plan = copy.deepcopy(self.plan)
        plan["schedule_kind"] = "arbitrary_task_order"
        self.reject(plan, "SCHEDULE")

    def test_task_depth_cannot_be_iteration_distance(self):
        plan = copy.deepcopy(self.plan)
        plan["preload_semantics"] = "task_preissue_depth"
        self.reject(plan, "SCHEDULE")

    def test_unknown_schema(self):
        plan = copy.deepcopy(self.plan)
        plan["schema_version"] = "v999"
        self.reject(plan, "SCHEMA")

    def test_physical_address_injection(self):
        plan = copy.deepcopy(self.plan)
        plan["buffers"][0]["pto.multi_buffer_addrs"] = [0, 512]
        self.reject(plan, "SCHEMA")

    def test_ownership(self):
        plan = copy.deepcopy(self.plan)
        borrowed = next(b for b in self.manifest["buffers"] if b["owner"] == "borrowed_entry")
        plan["buffers"].append(dict(buffer_id=borrowed["id"], count=2))
        self.reject(plan, "OWNERSHIP")

    def test_stale_hash(self):
        plan = copy.deepcopy(self.plan)
        plan["input_fingerprint"] = "stale"
        self.reject(plan, "STALE_PLAN")

    def test_bindings_changed(self):
        self.reject(self.plan, "STALE_PLAN", bindings={"N": 3})

    def test_profile_changed(self):
        profile = copy.deepcopy(self.profile)
        profile["capacity_bytes"]["vec"] -= 256
        self.reject(self.plan, "STALE_PLAN", profile=profile)

    def test_trip_count_changed(self):
        source = self.output("changed.pto")
        source.write_text(self.source.read_text(encoding="utf-8").replace("constant 2 : index", "constant 3 : index"),
                          encoding="utf-8")
        self.reject(self.plan, "STALE_PLAN", current_input=source)

    def test_c08_effective_preload(self):
        plan = copy.deepcopy(self.plan)
        plan["preload_count"] = 3
        first = import_plan(self.package, self.plan, self.output("p2"))
        second = import_plan(self.package, plan, self.output("p3"))
        self.assertEqual(first["effective_preload"], 2)
        self.assertEqual(first["effective_preload"], second["effective_preload"])

    def test_json_duplicate_key(self):
        path = self.output("duplicate.json")
        path.write_text('{"count":1,"count":2}', encoding="utf-8")
        with self.assertRaises(ContractError) as caught:
            read_json(path)
        self.assertEqual(caught.exception.code, "SCHEMA")

    def test_single_buffer_capacity(self):
        plan = copy.deepcopy(self.plan)
        plan["buffers"][0]["count"] = 100000
        self.reject(plan, "CAPACITY")

    def test_unknown_semantic_field(self):
        plan = copy.deepcopy(self.plan)
        plan["prefetch_everything"] = True
        self.reject(plan, "SCHEMA")

    def test_no_overwrite(self):
        output = self.output()
        import_plan(self.package, self.plan, output)
        before = (output / "annotated.pto").read_bytes()
        with self.assertRaises(ContractError) as caught:
            import_plan(self.package, self.plan, output)
        self.assertEqual(caught.exception.code, "OUTPUT_EXISTS")
        self.assertEqual(before, (output / "annotated.pto").read_bytes())

    def test_unknown_op_rejected(self):
        source = self.output("unknown.pto")
        text = self.source.read_text(encoding="utf-8")
        text = text.replace(
            "%c2 = arith.constant 2 : index",
            "%c2 = arith.constant 2 : index\n    %u = arith.divui %c2, %c1 : index")
        source.write_text(text, encoding="utf-8")
        with self.assertRaises(ContractError) as caught:
            export_package(source, self.profile, {}, self.output())
        self.assertEqual(caught.exception.code, "UNSUPPORTED")

    def test_profile_boolean_topology(self):
        profile = copy.deepcopy(self.profile)
        profile["aic_count"] = True
        with self.assertRaises(ContractError) as caught:
            export_package(self.source, profile, {}, self.output())
        self.assertEqual(caught.exception.code, "TARGET")

    def test_profile_arch_rejected(self):
        profile = copy.deepcopy(self.profile)
        profile["arch"] = "a3"
        with self.assertRaises(ContractError) as caught:
            export_package(self.source, profile, {}, self.output())
        self.assertEqual(caught.exception.code, "TARGET")

    def test_trip_count_mismatch(self):
        source = self.output("mismatch.pto")
        text = self.source.read_text(encoding="utf-8").replace("constant 2 : index", "constant 3 : index", 1)
        source.write_text(text, encoding="utf-8")
        with self.assertRaises(ContractError) as caught:
            export_package(source, self.profile, {}, self.output())
        self.assertEqual(caught.exception.code, "LEGALITY")

    def test_package_tamper(self):
        import shutil

        package = self.output("tampered")
        shutil.copytree(self.package, package)
        manifest = package / "manifest.json"
        text = manifest.read_text(encoding="utf-8").replace('"trip_count": 2', '"trip_count": 3')
        manifest.write_text(text, encoding="utf-8")
        with self.assertRaises(ContractError) as caught:
            import_plan(package, self.plan, self.output())
        self.assertEqual(caught.exception.code, "PACKAGE_INTEGRITY")

    def test_alias_root(self):
        source = self.output("alias.pto")
        text = self.source.read_text(encoding="utf-8")
        line = next(row for row in text.splitlines() if "pto.tneg ins(%qk :" in row)
        typ = line.split("ins(%qk : ", 1)[1].split(") outs", 1)[0]
        alias = f"      %qk_alias = pto.treshape %qk : {typ} -> {typ}\n"
        text = text.replace(line, alias + line.replace("%qk", "%qk_alias"))
        source.write_text(text, encoding="utf-8")
        export_package(source, self.profile, {}, self.output())
        manifest = read_json(self.output() / "manifest.json")
        borrowed = [b for b in manifest["buffers"] if b["owner"] == "borrowed_entry"]
        self.assertEqual(len(borrowed), 3)
        self.assertEqual(sum(len(b.get("aliases", [])) for b in borrowed), 1)

    def test_shape_changed(self):
        source = self.output("shape.pto")
        text = self.source.read_text(encoding="utf-8")
        replacements = (("16x16", "32x32"), ("8x16", "16x32"),
                        ("rows=16", "rows=32"), ("cols=16", "cols=32"),
                        ("v_row=16", "v_row=32"), ("v_col=16", "v_col=32"),
                        ("rows=8", "rows=16"), ("v_row=8", "v_row=16"),
                        ("constant 16 : index", "constant 32 : index"),
                        ("constant 8 : index", "constant 16 : index"),
                        ("size=4096,", "size=16384,"), ("slot_size=1024", "slot_size=4096"))
        for before, after in replacements:
            text = text.replace(before, after)
        source.write_text(text, encoding="utf-8")
        self.reject(self.plan, "STALE_PLAN", current_input=source)

    def test_borrowed_use_after_free(self):
        source = self.output("lifetime.pto")
        text = self.source.read_text(encoding="utf-8")
        use = next(row for row in text.splitlines() if "pto.tneg ins(%qk :" in row)
        text = text.replace(use + "\n", "")
        free = "      pto.tfree_from_aic {id=0,split=1}"
        self.assertIn(free, text)
        source.write_text(text.replace(free, free + "\n" + use), encoding="utf-8")
        with self.assertRaises(ContractError) as caught:
            export_package(source, self.profile, {}, self.output())
        self.assertIn(caught.exception.code, ("COMPILER", "LEGALITY"))
        self.assertFalse(self.output().exists())

    def test_export_without_pythonpath(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"PYTHONPATH": ""}):
            export_package(self.source, self.profile, {}, self.output())
        self.assertEqual(read_json(self.output() / "manifest.json"), self.manifest)

    def test_cli_rejection_report(self):
        import json

        plan = self.output("bad.json")
        plan.write_text('{"physical_address":0}', encoding="utf-8")
        result = subprocess.run([sys.executable, "-m", "ptoas._cli", "costmodel", "import",
                                 str(self.package), "--plan", str(plan), "--output", str(self.output())],
                                capture_output=True, text=True, timeout=120, check=False)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr)["status"], "rejected")
        self.assertEqual(json.loads(result.stderr)["code"], "SCHEMA")
        self.assertFalse(self.output().exists())

    def test_valid_shape_bytes(self):
        from ptoas.mlir import ir
        from ptoas.mlir.dialects import pto
        from ptoas._cv_ir import tile_info

        with ir.Context() as context:
            pto.register_dialect(context, load=True)
            typ = ir.Type.parse("!pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, "
                                "v_row=15, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>")
            row = tile_info(typ, self.profile)
            self.assertEqual(row["logical_bytes"], 960)
            self.assertEqual(row["allocation_bytes"], 1024)
            self.assertEqual(row["slot_stride_bytes"], 1024)

    def test_serial_compiles(self):
        result = subprocess.run([sys.executable, "-m", "ptoas._cli", "--pto-arch=a5", str(self.source),
                                 "-o", str(self.output("kernel.cpp"))],
                                capture_output=True, text=True, timeout=120, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("TPUSH", self.output("kernel.cpp").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
