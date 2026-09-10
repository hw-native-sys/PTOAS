#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import hashlib
import importlib.util
import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock


try:
    from packaging.tags import Tag
except ImportError:  # pragma: no cover - exercised only in minimal build environments.
    try:
        from pip._vendor.packaging.tags import Tag
    except ImportError:
        Tag = None


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "install_nightly_wheel.py"
SPEC = importlib.util.spec_from_file_location("install_nightly_wheel", SCRIPT)
assert SPEC and SPEC.loader
INSTALLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALLER)


class NightlyWheelSelectionTests(unittest.TestCase):
    @unittest.skipIf(Tag is None, "packaging is not available in this Python environment")
    def test_falls_back_to_pip_vendor_packaging(self):
        from pip._vendor.packaging.tags import Tag as PipTag

        compatible = PipTag("cp312", "cp312", "manylinux_2_34_x86_64")
        release = {
            "tag_name": "nightly",
            "assets": [
                {
                    "name": "ptoas-0.57-cp312-cp312-manylinux_2_34_x86_64.whl",
                    "browser_download_url": "https://example.invalid/new.whl",
                }
            ],
        }
        from pip._vendor.packaging import tags as pip_tags

        with mock.patch.dict(
            sys.modules,
            {"packaging": None, "packaging.tags": None, "packaging.utils": None},
        ):
            with mock.patch.object(
                pip_tags, "sys_tags", return_value=iter([compatible])
            ):
                selection = INSTALLER.select_wheel(release)

        self.assertEqual(selection.name, "ptoas-0.57-cp312-cp312-manylinux_2_34_x86_64.whl")

    @unittest.skipIf(Tag is None, "packaging is not available in this Python environment")
    def test_selects_latest_compatible_version(self):
        compatible = Tag("cp312", "cp312", "manylinux_2_34_x86_64")
        release = {
            "tag_name": "nightly",
            "assets": [
                {
                    "name": "ptoas-0.56-cp312-cp312-manylinux_2_34_x86_64.whl",
                    "browser_download_url": "https://example.invalid/old.whl",
                    "updated_at": "2026-08-06T00:00:00Z",
                },
                {
                    "name": "ptoas-0.57-cp312-cp312-manylinux_2_34_x86_64.whl",
                    "browser_download_url": "https://example.invalid/new.whl",
                    "updated_at": "2026-08-06T01:00:00Z",
                },
                {
                    "name": "ptoas-0.58-cp311-cp311-manylinux_2_34_x86_64.whl",
                    "browser_download_url": "https://example.invalid/wrong-python.whl",
                },
            ],
        }

        with mock.patch("packaging.tags.sys_tags", return_value=iter([compatible])):
            selection = INSTALLER.select_wheel(release)

        self.assertEqual(selection.name, "ptoas-0.57-cp312-cp312-manylinux_2_34_x86_64.whl")
        self.assertEqual(selection.url, "https://example.invalid/new.whl")

    @unittest.skipIf(Tag is None, "packaging is not available in this Python environment")
    def test_prefers_newer_asset_over_higher_version(self):
        compatible = Tag("cp312", "cp312", "manylinux_2_34_x86_64")
        release = {
            "tag_name": "nightly",
            "assets": [
                {
                    "name": "ptoas-0.58-cp312-cp312-manylinux_2_34_x86_64.whl",
                    "browser_download_url": "https://example.invalid/old.whl",
                    "updated_at": "2026-08-05T00:00:00Z",
                },
                {
                    "name": "ptoas-0.57-cp312-cp312-manylinux_2_34_x86_64.whl",
                    "browser_download_url": "https://example.invalid/new.whl",
                    "updated_at": "2026-08-06T01:00:00Z",
                },
            ],
        }

        with mock.patch("packaging.tags.sys_tags", return_value=iter([compatible])):
            selection = INSTALLER.select_wheel(release, "ptoas")

        self.assertEqual(selection.name, "ptoas-0.57-cp312-cp312-manylinux_2_34_x86_64.whl")

    @unittest.skipIf(Tag is None, "packaging is not available in this Python environment")
    def test_prefers_higher_build_number(self):
        compatible = Tag("cp312", "cp312", "manylinux_2_34_x86_64")
        release = {
            "tag_name": "nightly",
            "assets": [
                {
                    "name": "ptoas-0.57-1-cp312-cp312-manylinux_2_34_x86_64.whl",
                    "browser_download_url": "https://example.invalid/build1.whl",
                    "updated_at": "2026-08-06T01:00:00Z",
                },
                {
                    "name": "ptoas-0.57-2-cp312-cp312-manylinux_2_34_x86_64.whl",
                    "browser_download_url": "https://example.invalid/build2.whl",
                    "updated_at": "2026-08-06T01:00:00Z",
                },
            ],
        }

        with mock.patch("packaging.tags.sys_tags", return_value=iter([compatible])):
            selection = INSTALLER.select_wheel(release, "ptoas")

        self.assertEqual(selection.name, "ptoas-0.57-2-cp312-cp312-manylinux_2_34_x86_64.whl")

    @unittest.skipIf(Tag is None, "packaging is not available in this Python environment")
    def test_normalizes_distribution_name(self):
        compatible = Tag("cp312", "cp312", "manylinux_2_34_x86_64")
        release = {
            "tag_name": "nightly",
            "assets": [
                {
                    "name": "pto_as-0.57-cp312-cp312-manylinux_2_34_x86_64.whl",
                    "browser_download_url": "https://example.invalid/pto-as.whl",
                }
            ],
        }

        with mock.patch("packaging.tags.sys_tags", return_value=iter([compatible])):
            selection = INSTALLER.select_wheel(release, "pto_as")

        self.assertEqual(selection.name, "pto_as-0.57-cp312-cp312-manylinux_2_34_x86_64.whl")

    @unittest.skipIf(Tag is None, "packaging is not available in this Python environment")
    def test_prefers_better_supported_tag(self):
        platform_tag = Tag("cp312", "cp312", "manylinux_2_34_x86_64")
        universal_tag = Tag("py3", "none", "any")
        release = {
            "tag_name": "nightly",
            "assets": [
                {
                    "name": "ptoas-0.57-py3-none-any.whl",
                    "browser_download_url": "https://example.invalid/universal.whl",
                    "updated_at": "2026-08-06T01:00:00Z",
                },
                {
                    "name": "ptoas-0.57-cp312-cp312-manylinux_2_34_x86_64.whl",
                    "browser_download_url": "https://example.invalid/platform.whl",
                    "updated_at": "2026-08-06T01:00:00Z",
                },
            ],
        }

        with mock.patch("packaging.tags.sys_tags", return_value=iter([platform_tag, universal_tag])):
            selection = INSTALLER.select_wheel(release, "ptoas")

        self.assertEqual(selection.name, "ptoas-0.57-cp312-cp312-manylinux_2_34_x86_64.whl")

    @unittest.skipIf(Tag is None, "packaging is not available in this Python environment")
    def test_rejects_missing_compatible_wheel(self):
        release = {"tag_name": "nightly", "assets": []}
        with self.assertRaisesRegex(RuntimeError, "no compatible ptoas wheel"):
            with mock.patch("packaging.tags.sys_tags", return_value=iter(())):
                INSTALLER.select_wheel(release, "ptoas")

    def test_download_verifies_sha256(self):
        payload = b"nightly wheel"
        expected = hashlib.sha256(payload).hexdigest()
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.side_effect = [payload, b""]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "wheel.whl"
            with mock.patch("urllib.request.urlopen", return_value=response):
                INSTALLER.download("https://example.invalid/wheel.whl", destination, expected)
            self.assertEqual(destination.read_bytes(), payload)

    def test_download_rejects_wrong_sha256(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.side_effect = [b"nightly wheel", b""]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "wheel.whl"
            with mock.patch("urllib.request.urlopen", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                    INSTALLER.download("https://example.invalid/wheel.whl", destination, "0" * 64)


if __name__ == "__main__":
    unittest.main()
