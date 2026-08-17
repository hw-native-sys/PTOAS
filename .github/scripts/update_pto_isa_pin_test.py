#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pathlib
import subprocess
import sys
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).with_name("update_pto_isa_pin.py")


def run(command: list[str], cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def git(repo: pathlib.Path, *args: str) -> str:
    result = run(["git", *args], repo)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


class PinUpdaterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="pto-isa-pin-test-")
        self.root = pathlib.Path(self.temp_dir.name)
        self.repo_a = self.create_repo("repo-a")
        self.base = self.commit(self.repo_a, "base")
        self.child = self.commit(self.repo_a, "child")
        git(self.repo_a, "switch", "-q", "-c", "side", self.base)
        self.side = self.commit(self.repo_a, "side")
        git(self.repo_a, "switch", "-q", "main")

        self.repo_b = self.create_repo("repo-b")
        self.other = self.commit(self.repo_b, "other")
        self.fixture_dir = self.root / "fixtures"
        self.fixture_dir.mkdir()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def create_repo(self, name: str) -> pathlib.Path:
        repo = self.root / name
        repo.mkdir()
        git(repo, "init", "-q", "-b", "main")
        git(repo, "config", "user.name", "Pin Test")
        git(repo, "config", "user.email", "pin-test@example.com")
        return repo

    def commit(self, repo: pathlib.Path, value: str) -> str:
        (repo / "value.txt").write_text(value, encoding="utf-8")
        git(repo, "add", "value.txt")
        git(repo, "commit", "-q", "-m", value)
        return git(repo, "rev-parse", "HEAD")

    def write_default_fixtures(self, commit: str) -> list[str]:
        ci = self.fixture_dir / "ci.yml"
        ci.write_text(
            "pto_isa_commit:\n  description: pin\n  default: "
            + commit
            + "\nenv:\n  PTO_ISA_COMMIT: ${{ github.event.inputs.pto_isa_commit || '"
            + commit
            + "' }}\n",
            encoding="utf-8",
        )
        dockerfile = self.fixture_dir / "Dockerfile"
        dockerfile.write_text(
            f"ARG PTO_ISA_COMMIT={commit}\n"
            f"# pinned: https://gitcode.com/cann/pto-isa/commit/{commit}\n",
            encoding="utf-8",
        )
        guide = self.fixture_dir / "guide.md"
        guide.write_text(
            f"export PTO_ISA_COMMIT={commit}\nexport PTO_ISA_COMMIT={commit}\n",
            encoding="utf-8",
        )
        remote = self.fixture_dir / "remote.sh"
        remote.write_text(
            f'PTO_ISA_COMMIT="${{PTO_ISA_COMMIT:-{commit}}}"\n',
            encoding="utf-8",
        )
        return [
            "--ci-workflow",
            str(ci),
            "--dockerfile",
            str(dockerfile),
            "--compile-only-guide",
            str(guide),
            "--remote-validation-script",
            str(remote),
        ]

    def invoke(
        self,
        target: str,
        repo: pathlib.Path,
        commit: str,
        fixture_args: list[str],
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(SCRIPT),
            "--target",
            target,
            "--repo-url",
            str(repo),
            "--commit",
            commit,
            *fixture_args,
        ]
        if check:
            command.append("--check")
        return run(command, self.root)

    def test_equal_is_a_checked_noop(self) -> None:
        args = self.write_default_fixtures(self.base)
        result = self.invoke(
            "gitcode-default", self.repo_a, self.base, args, check=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), self.base)

    def test_descendant_updates_all_default_locations(self) -> None:
        args = self.write_default_fixtures(self.base)
        result = self.invoke("gitcode-default", self.repo_a, self.child, args)
        self.assertEqual(result.returncode, 0, result.stderr)
        for path in self.fixture_dir.iterdir():
            self.assertNotIn(self.base, path.read_text(encoding="utf-8"))
            self.assertIn(self.child, path.read_text(encoding="utf-8"))

    def test_ancestor_and_non_fast_forward_are_rejected(self) -> None:
        args = self.write_default_fixtures(self.child)
        ancestor = self.invoke("gitcode-default", self.repo_a, self.base, args)
        self.assertNotEqual(ancestor.returncode, 0)
        self.assertIn("refusing non-fast-forward", ancestor.stderr)

        args = self.write_default_fixtures(self.side)
        sibling = self.invoke("gitcode-default", self.repo_a, self.child, args)
        self.assertNotEqual(sibling.returncode, 0)
        self.assertIn("refusing non-fast-forward", sibling.stderr)

    def test_unknown_revision_is_rejected(self) -> None:
        args = self.write_default_fixtures(self.base)
        result = self.invoke("gitcode-default", self.repo_a, "f" * 40, args)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("is not reachable", result.stderr)

    def test_targets_update_only_their_own_files(self) -> None:
        ci_sim = self.fixture_dir / "ci_sim.yml"
        ci_sim.write_text(
            f"env:\n  PTO_ISA_COMMIT: {self.base}\n", encoding="utf-8"
        )
        dev = self.fixture_dir / "Dockerfile.dev"
        dev.write_text(
            f"ARG PTO_ISA_COMMIT={self.base}\n", encoding="utf-8"
        )

        ci_result = self.invoke(
            "github-ci-sim",
            self.repo_a,
            self.child,
            ["--ci-sim-workflow", str(ci_sim), "--dev-dockerfile", str(dev)],
        )
        self.assertEqual(ci_result.returncode, 0, ci_result.stderr)
        self.assertIn(self.child, ci_sim.read_text(encoding="utf-8"))
        self.assertIn(self.base, dev.read_text(encoding="utf-8"))

    def test_cross_remote_revision_is_rejected_without_writes(self) -> None:
        ci_sim = self.fixture_dir / "ci_sim.yml"
        ci_sim.write_text(
            f"env:\n  PTO_ISA_COMMIT: {self.base}\n", encoding="utf-8"
        )
        result = self.invoke(
            "github-ci-sim",
            self.repo_b,
            self.other,
            ["--ci-sim-workflow", str(ci_sim)],
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("is not reachable", result.stderr)
        self.assertIn(self.base, ci_sim.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
