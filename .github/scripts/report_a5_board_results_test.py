#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import contextlib
import importlib.util
import io
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = pathlib.Path(__file__).with_name("report_a5_board_results.py")
SPEC = importlib.util.spec_from_file_location("report_a5_board_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


class BoardResultReportTest(unittest.TestCase):
    def test_load_and_render_results(self) -> None:
        with tempfile.TemporaryDirectory(prefix="a5-board-report-") as temp_dir:
            results = pathlib.Path(temp_dir) / "results.tsv"
            results.write_text(
                "testcase\tstatus\tstage\tinfo\n"
                "Abs/abs\tOK\trun\tvalidation=independent-golden\n"
                "Add/add\tFAIL\trun\texit=1\n"
                "Print/print\tSKIP\trun\tin SKIP_CASES\n",
                encoding="utf-8",
            )

            summary = REPORT.load_results(results)
            self.assertTrue(summary.results_found)
            self.assertEqual(summary.total, 3)
            self.assertEqual(summary.counts["OK"], 1)
            self.assertEqual(summary.counts["FAIL"], 1)
            self.assertEqual(summary.failed_cases, ("Add/add",))

            markdown = REPORT.render_markdown(
                summary,
                conclusion="failure",
                run_url="https://github.example/actions/runs/1",
                sha="0123456789abcdef",
            )
            self.assertIn("Status: **FAIL**", markdown)
            self.assertIn("`Add/add`", markdown)
            self.assertIn("`0123456789ab`", markdown)

    def test_log_errors_are_attached_to_failed_cases(self) -> None:
        with tempfile.TemporaryDirectory(prefix="a5-board-report-") as temp_dir:
            root = pathlib.Path(temp_dir)
            results = root / "results.tsv"
            results.write_text(
                "testcase\tstatus\tstage\tinfo\n"
                "Rowexpandsub/rowexpandsub\tFAIL\trun\texit=2\n"
                "DeepseekV4DecodeA5/rope_cs\tFAIL\trun\texit=139\n"
                "TquantMx/tquant_mx\tFAIL\trun\texit=2\n",
                encoding="utf-8",
            )
            log = root / "board-validation.log"
            log.write_text(
                "[time] === CASE: /tmp/payload/test/samples/Rowexpandsub/rowexpandsub-pto.cpp ===\n"
                "[ERROR] Mismatch: golden_v3.bin vs v3.bin, max diff=5.5\n"
                "[time] ERROR: testcase failed (exit 2): rowexpandsub\n"
                "[time] === CASE: /tmp/payload/test/samples/DeepseekV4DecodeA5/rope_cs-pto.cpp ===\n"
                "run_remote_npu_validation.sh: line 792: Segmentation fault (core dumped)\n"
                "[time] === CASE: /tmp/payload/test/samples/TquantMx/tquant_mx-pto.cpp ===\n"
                "error: no member named 'assignData' in 'pto::Tile'\n",
                encoding="utf-8",
            )

            summary = REPORT.load_results(results, log)
            details = dict(summary.failure_details)
            self.assertIn("Mismatch: golden_v3.bin", details["Rowexpandsub/rowexpandsub"])
            self.assertIn("Segmentation fault", details["DeepseekV4DecodeA5/rope_cs"])
            self.assertIn("no member named 'assignData'", details["TquantMx/tquant_mx"])
            markdown = REPORT.render_markdown(
                summary, conclusion="failure", run_url="", sha=""
            )
            self.assertIn("### Concrete errors", markdown)

            payload = REPORT.build_feishu_payload(
                summary, conclusion="failure", run_url="", sha=""
            )
            content = payload["card"]["elements"][0]["text"]["content"]
            self.assertIn("**Concrete errors**", content)
            self.assertIn("Segmentation fault", content)

            detail_payloads = REPORT.build_feishu_detail_payloads(
                summary, run_url="https://github.example/actions/runs/3"
            )
            self.assertGreaterEqual(len(detail_payloads), 1)
            detail_content = "\n".join(
                card["card"]["elements"][0]["text"]["content"]
                for card in detail_payloads
            )
            for testcase in summary.failed_cases:
                self.assertIn(f"`{testcase}`", detail_content)

    def test_missing_results_are_reported_as_failure(self) -> None:
        summary = REPORT.load_results(pathlib.Path("/path/that/does/not/exist"))
        self.assertFalse(summary.results_found)
        self.assertFalse(summary.passed)
        payload = REPORT.build_feishu_payload(
            summary,
            conclusion="failure",
            run_url="",
            sha="",
        )
        self.assertEqual(payload["card"]["header"]["template"], "red")
        self.assertIn("夜间看护", payload["card"]["header"]["title"]["content"])

    def test_successful_results_are_reported_as_pass(self) -> None:
        summary = REPORT.BoardSummary(
            True,
            REPORT.Counter({"OK": 2, "SKIP": 1}),
            (),
        )
        self.assertTrue(summary.passed)
        payload = REPORT.build_feishu_payload(
            summary,
            conclusion="success",
            run_url="https://github.example/actions/runs/2",
            sha="abcdef0123456789",
        )
        self.assertEqual(payload["card"]["header"]["template"], "green")

    def test_invalid_webhook_does_not_mask_successful_summary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="a5-board-report-") as temp_dir:
            results = pathlib.Path(temp_dir) / "results.tsv"
            results.write_text(
                "testcase\tstatus\tstage\tinfo\nAbs/abs\tOK\trun\tpassed\n",
                encoding="utf-8",
            )
            argv = [
                str(SCRIPT),
                "--results",
                str(results),
                "--conclusion",
                "success",
            ]
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.dict(os.environ, {"A5_FEISHU_WEBHOOK_URL": "invalid"}),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                self.assertEqual(REPORT.main(), 0)
            self.assertIn("Status: **PASS**", stdout.getvalue())
            self.assertIn("WARNING: failed to send Feishu notification", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
