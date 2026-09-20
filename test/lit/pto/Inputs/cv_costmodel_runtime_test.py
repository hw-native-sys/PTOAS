# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Reject ambiguous profiling boundaries before reporting diagnostic latency."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

SAMPLE = Path(__file__).resolve().parents[3] / "samples" / "CVCostModel"
sys.path.insert(0, str(SAMPLE))
from summarize_runtime import duration, summarize
from summarize_ab import paired_statistics
from run_ab import performance_order
from run_mechanism import ORDERS, correctness_order as mechanism_correctness_order
from run_mechanism import performance_order as mechanism_performance_order
from runtime_fixture import multibuffer_stress_source_text


class TimingBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pto-timing-test-")
        self.root = Path(self.temporary.name)
        (self.root / "profile").mkdir()
        self.row = {"Op Name": "fixed", "Task Type": "MIX_AIC", "Device_id": "0",
                    "Block Num": "1", "Mix Block Num": "2", "Task Duration(us)": "12.5"}

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, rows):
        with (self.root / "profile" / "op_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.row))
            writer.writeheader()
            writer.writerows(rows)

    def test_whole_mixed_kernel(self):
        self.save([self.row])
        self.assertEqual(duration(self.root, "fixed", 0)["duration_us"], 12.5)

    def test_unknown_and_nonfinite_timing(self):
        for value in ("nan", "inf", "0", "-1", "N/A"):
            self.save([dict(self.row, **{"Task Duration(us)": value})])
            with self.assertRaises(ValueError):
                duration(self.root, "fixed", 0)

    def test_partial_or_multiple_tasks_rejected(self):
        for rows in ([self.row, self.row], [dict(self.row, **{"Task Type": "AI_VECTOR_CORE"})],
                     [dict(self.row, **{"Mix Block Num": "1"})]):
            self.save(rows)
            with self.assertRaises(ValueError):
                duration(self.root, "fixed", 0)

    def test_two_baselines_and_overlap(self):
        def bucket(values):
            return dict(samples=[dict(duration_us=v) for v in values])
        groups = {("case", "serial"): bucket([20, 21, 19, 20, 20]),
                  ("case", "p0"): bucket([10, 11, 9, 10, 10]),
                  ("case", "p2"): bucket([10, 10, 10, 10, 10])}
        row = next(r for r in summarize(groups) if r["variant"] == "p2")
        self.assertEqual(row["ratio_to_serial"], 0.5)
        self.assertEqual(row["ratio_to_p0"], 1.0)
        self.assertEqual(row["comparison_to_p0"], "cannot_reliably_distinguish")

    def test_paired_bootstrap_is_reproducible(self):
        baseline = [10.0] * 20
        candidate = [9.0] * 20
        first = paired_statistics(baseline, candidate)
        self.assertEqual(first, paired_statistics(baseline, candidate))
        self.assertAlmostEqual(first["mean_gain"], 0.1)
        self.assertGreater(first["confidence_interval"][0], 0)

    def test_ab_ba_order_and_baseline_retention(self):
        variant = lambda name: {"variant": name}
        manifest = {"cases": [
            {"case": "optimized", "candidates": [variant("A"), variant("P0"), variant("B")]},
            {"case": "retained", "candidates": [variant("A"), variant("P0")]}]}
        rows = list(performance_order(manifest))
        paired = [row for row in rows if row[4] == "paired_profile"]
        self.assertEqual(len(paired), 40)
        for block in range(20):
            selected = [row for row in paired if row[3] == block]
            expected = ["A", "B"] if block % 2 == 0 else ["B", "A"]
            self.assertEqual([row[1]["variant"] for row in selected], expected)
            self.assertTrue(all(row[2] == block % 3 for row in selected))
        self.assertFalse(any(row[0]["case"] == "retained" for row in rows))

    def test_stress_fixture_keeps_odd_negation_result(self):
        for repetitions in (129, 257, 513):
            text = multibuffer_stress_source_text(repetitions)
            self.assertEqual(text.count("pto.tneg"), repetitions)
            self.assertEqual(text.count("%d = pto.alloc_tile"), 1)
            operations = [line.strip() for line in text.splitlines() if "pto.tneg" in line]
            self.assertIn("ins(%qk", operations[0])
            self.assertIn("outs(%a", operations[-1])

    def test_four_way_mechanism_matrix(self):
        variant = lambda name: {"variant": name}
        manifest = {"cases": [{"case": "stress", "candidates": [
            variant(name) for name in ("A", "P0", "M", "B")]}]}
        correctness = list(mechanism_correctness_order(manifest))
        self.assertEqual(len(correctness), 24)
        self.assertEqual({row[1]["variant"] for row in correctness}, {"A", "P0", "M", "B"})
        rows = list(mechanism_performance_order(manifest))
        self.assertEqual(len(rows), 84)
        paired = [row for row in rows if row[4] == "paired_profile"]
        self.assertEqual(len(paired), 80)
        for block in range(20):
            selected = [row for row in paired if row[3] == block]
            self.assertEqual(tuple(row[1]["variant"] for row in selected), ORDERS[block % 4])
            self.assertTrue(all(row[2] == block % 3 for row in selected))


if __name__ == "__main__":
    unittest.main()
