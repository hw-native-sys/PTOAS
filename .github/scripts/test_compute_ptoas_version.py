#!/usr/bin/env python3

import pathlib
import subprocess
import sys
import tempfile


SCRIPT_PATH = pathlib.Path(__file__).with_name("compute_ptoas_version.py")


def write_cmake(path: pathlib.Path, version: str) -> None:
    path.write_text(f"cmake_minimum_required(VERSION 3.20.0)\nproject(ptoas VERSION {version})\n", encoding="utf-8")


def run_ok(*args: str) -> str:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run_fail(*args: str) -> str:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        raise AssertionError("expected failure, but command succeeded")
    return completed.stderr.strip()


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        cmake_file = pathlib.Path(tmpdir) / "CMakeLists.txt"

        write_cmake(cmake_file, "0.10")
        assert run_ok("--cmake-file", str(cmake_file), "--mode", "dev") == "0.10"
        assert run_ok("--cmake-file", str(cmake_file), "--mode", "release") == "0.11"
        assert (
            run_ok(
                "--cmake-file",
                str(cmake_file),
                "--mode",
                "release",
                "--check-tag",
                "v0.11",
            )
            == "0.11"
        )

        write_cmake(cmake_file, "0.21")
        assert (
            run_ok(
                "--cmake-file",
                str(cmake_file),
                "--mode",
                "release",
                "--check-tag",
                "v0.21",
            )
            == "0.21"
        )
        assert (
            run_ok(
                "--cmake-file",
                str(cmake_file),
                "--mode",
                "release",
                "--check-tag",
                "v0.22",
            )
            == "0.22"
        )

        error = run_fail(
            "--cmake-file",
            str(cmake_file),
            "--mode",
            "release",
            "--check-tag",
            "v0.20",
        )
        assert "does not match next release version '0.22' or current base version '0.21'" in error

    print("compute_ptoas_version.py tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
