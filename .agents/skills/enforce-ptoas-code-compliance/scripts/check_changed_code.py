#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run deterministic compliance checks on changed PTOAS code."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess


LINE_LIMIT = 120
MAX_TEXT_FILE_SIZE = 2 * 1024 * 1024
LICENSE_MARKER = "Copyright (c)"
DELIVERY_MARKERS = ("TO" + "DO", "T" + "BD", "FIX" + "ME")
DELIVERY_MARKER_PATTERN = re.compile(r"\b(?:" + "|".join(DELIVERY_MARKERS) + r")\b")
TEST_DIRECTORY_NAMES = frozenset({"test", "tests", "unittest", "unittests"})
LICENSED_LANGUAGES = frozenset({"cpp", "python", "shell", "cmake"})


@dataclass(frozen=True)
class ChangedLine:
    path: Path
    number: int
    text: str


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    severity: str
    rule: str
    message: str


@dataclass(frozen=True)
class TextRule:
    languages: frozenset[str]
    expression: re.Pattern[str]
    severity: str
    rule: str
    message: str


CPP = frozenset({"cpp"})
PYTHON = frozenset({"python"})
SHELL = frozenset({"shell"})
CMAKE = frozenset({"cmake"})
CPP_CMAKE = frozenset({"cpp", "cmake"})
SCRIPT_LANGUAGES = frozenset({"python", "shell", "cmake", "batch"})

TEXT_RULES = (
    TextRule(CPP, re.compile(r"(?<!static_)\bassert\s*\("), "error", "G.AST.02/G.TES.01",
             "use explicit runtime handling or the approved debug-only assertion macro"),
    TextRule(CPP, re.compile(r"\bNULL\b"), "error", "G.EXP.35-CPP", "use nullptr"),
    TextRule(CPP, re.compile(r"\b(?:strcpy|strcat|sprintf|vsprintf|gets)\s*\("), "error",
             "G.FUU.21", "do not introduce an unbounded memory or string function"),
    TextRule(CPP, re.compile(r"\b(?:realloc|alloca)\s*\("), "error", "G.FUU.09/G.FUU.10",
             "use a scoped container or checked allocation strategy"),
    TextRule(CPP, re.compile(r"\b(?:atoi|atol|atoll|atof)\s*\("), "warning", "G.FUU.22",
             "use a conversion API that reports invalid input and range errors"),
    TextRule(CPP, re.compile(r"\b(?:reinterpret_cast|const_cast)\s*<"), "warning",
             "G.EXP.15-CPP/G.EXP.16-CPP", "prove that the cast is necessary and safe"),
    TextRule(CPP, re.compile(r"\bgoto\b"), "warning", "G.EXP.42-CPP", "prefer structured control flow"),
    TextRule(CPP, re.compile(r"\b(?:memcpy|memmove|memset)\s*\("), "warning",
             "G.MEM.03/EChecker_BufferSize", "prove all source and destination bounds"),
    TextRule(CPP, re.compile(r"\b(?:malloc|calloc)\s*\("), "warning", "G.MEM.01/G.MEM.02",
             "prove size arithmetic and allocation-failure handling"),
    TextRule(CPP, re.compile(r"\.(?:lock|unlock)\s*\("), "warning", "G.CON.02-CPP",
             "prefer a scoped lock wrapper"),
    TextRule(CPP, re.compile(r"\b(?:0[xX][0-9A-Fa-f]+|\d+)l+\b"), "warning", "G.CNS.01-CPP",
             "use an uppercase L integer suffix"),
    TextRule(CPP, re.compile(r"^\s*(?:if|for|while)\s*\([^;]*\)\s*$"), "error", "G.FMT.11-CPP",
             "put the control-statement body in braces"),
    TextRule(CPP, re.compile(r"^\s*(?:else|do)\s*$"), "error", "G.FMT.11-CPP",
             "put the control-statement body in braces"),
    TextRule(CPP, re.compile(r"^\s*(?:if|for|while)\s*\([^;]*\)\s+[^{\s].*$"), "error",
             "G.FMT.11-CPP", "put the control-statement body in braces"),
    TextRule(PYTHON, re.compile(r"\b(?:eval|exec)\s*\("), "error", "G.EDV.01",
             "do not evaluate code from data"),
    TextRule(PYTHON, re.compile(r"shell\s*=\s*" + r"True"), "error", "G.EDV.04",
             "invoke an argument vector without a command shell"),
    TextRule(PYTHON, re.compile(r"\bos\.system\s*\("), "error", "G.EDV.02",
             "invoke a validated argument vector without a command shell"),
    TextRule(PYTHON, re.compile(r"\btempfile\.mktemp\s*\("), "error", "G.FIO.03",
             "use a secure tempfile context manager"),
    TextRule(PYTHON, re.compile(r"\b(?:pickle|_pickle)\.load\s*\("), "error", "G.SER.01",
             "do not deserialize untrusted pickle data"),
    TextRule(PYTHON, re.compile(r"\byaml\.load\s*\("), "error", "G.SER.03",
             "use yaml.safe_load for untrusted input"),
    TextRule(PYTHON, re.compile(r"\bshelve\.open\s*\("), "warning", "G.SER.01",
             "prove the shelf is trusted or use a non-executable format"),
    TextRule(PYTHON, re.compile(r"def\s+\w+\s*\([^)]*=\s*(?:\[\]|\{\}|set\(\))"), "error",
             "G.FNM.01", "do not use a mutable default argument"),
    TextRule(PYTHON, re.compile(r"\b(?:logging|logger)\.(?:debug|info)\s*\(\s*f[\"']"), "warning",
             "G.LOG.01", "use logging lazy interpolation"),
    TextRule(SHELL, re.compile(r"(^|[;&|]\s*)ev" + r"al(?:\s|$)"), "error", "G.EDV.01/G.EDV.02",
             "do not evaluate a constructed command"),
    TextRule(SHELL, re.compile(r"\b(?:bash|sh)\s+-c(?:\s|$)"), "warning", "G.EDV.02",
             "prefer direct argument-vector execution"),
    TextRule(SHELL, re.compile(r"\brm\s+-[A-Za-z]*[rf][A-Za-z]*\s+[^\n]*[*?]"), "error",
             "G.EDV.03", "do not combine recursive deletion with an unresolved wildcard"),
    TextRule(CMAKE, re.compile(r"(?<!\w)-fpermissive\b"), "error", "G.C&C++.LANG.04",
             "do not weaken C++ language enforcement"),
    TextRule(CMAKE, re.compile(r"(^|\s)-w(\s|$)"), "error", "G.C&C++.WARN.04",
             "do not suppress all compiler warnings"),
    TextRule(CMAKE, re.compile(r"-Wno-error\s*="), "error", "G.C&C++.WARN.06",
             "do not downgrade an explicitly promoted warning"),
    TextRule(CMAKE, re.compile(r"include\s*\([^)]*CMakeLists\.txt", re.IGNORECASE), "error",
             "G.CMake.03/G.CMake.07", "use add_subdirectory instead of including CMakeLists.txt"),
    TextRule(CMAKE, re.compile(r"^\s*[A-Z][A-Z0-9_]*\s*\("), "warning", "G.CMake.19",
             "use lowercase CMake command names"),
    TextRule(CMAKE, re.compile(r"\b(?:BUILD_)?RPATH\b"), "warning", "G.C&C++.SEC.06",
             "review runtime search paths for the target and release context"),
    TextRule(SCRIPT_LANGUAGES, re.compile(r"(?:^|[\"'])/(?:home|Users|root)/"), "error",
             "G.SCRIPT.05/G.SCRIPT.06", "derive paths instead of hard-coding a user installation path"),
)


def run_git(repo: Path, arguments: list[str]) -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable was not found")
    result = subprocess.run(
        [git, "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout


def parse_base(base: str) -> str:
    if not base or base.startswith("-") or any(character.isspace() for character in base):
        raise argparse.ArgumentTypeError("must be a non-option Git revision without whitespace")
    return base


def language_for(path: Path) -> str | None:
    name = path.name
    suffix = path.suffix.lower()
    if name == "CMakeLists.txt" or suffix == ".cmake":
        return "cmake"
    if name.lower() == "dockerfile":
        return "docker"
    if suffix in {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inc", ".td"}:
        return "cpp"
    if suffix == ".py":
        return "python"
    if suffix in {".sh", ".bash"}:
        return "shell"
    if suffix in {".bat", ".cmd"}:
        return "batch"
    return None


def nul_paths(text: str) -> set[Path]:
    return {Path(item) for item in text.split("\0") if item}


def changed_paths(repo: Path, base: str) -> tuple[set[Path], set[Path]]:
    tracked = nul_paths(run_git(repo, ["diff", "--name-only", "-z", base, "--"]))
    added = nul_paths(run_git(repo, ["diff", "--name-only", "--diff-filter=A", "-z", base, "--"]))
    untracked = nul_paths(run_git(repo, ["ls-files", "--others", "--exclude-standard", "-z"]))
    return tracked | untracked, added | untracked


def parse_diff(diff: str) -> list[ChangedLine]:
    lines: list[ChangedLine] = []
    current_path: Path | None = None
    line_number = 0
    hunk = re.compile(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
    for raw_line in diff.splitlines():
        if raw_line.startswith("+++ "):
            label = raw_line[4:]
            current_path = Path(label[2:]) if label.startswith("b/") else None
            continue
        match = hunk.match(raw_line)
        if match is not None:
            line_number = int(match.group(1))
            continue
        if current_path is None:
            continue
        if raw_line.startswith("+"):
            lines.append(ChangedLine(current_path, line_number, raw_line[1:]))
            line_number += 1
        elif raw_line.startswith(" "):
            line_number += 1
    return lines


def resolve_repo_file(repo: Path, relative: Path) -> Path:
    resolved = (repo / relative).resolve()
    try:
        resolved.relative_to(repo)
    except ValueError as error:
        raise RuntimeError(f"changed path escapes repository: {relative}") from error
    return resolved


def read_text_file(repo: Path, relative: Path) -> str | None:
    resolved = resolve_repo_file(repo, relative)
    if not resolved.is_file() or resolved.stat().st_size > MAX_TEXT_FILE_SIZE:
        return None
    data = resolved.read_bytes()
    if b"\0" in data:
        return None
    return data.decode("utf-8")


def untracked_lines(repo: Path, paths: set[Path]) -> list[ChangedLine]:
    lines: list[ChangedLine] = []
    for path in sorted(paths):
        if language_for(path) is None:
            continue
        try:
            content = read_text_file(repo, path)
        except UnicodeDecodeError:
            continue
        if content is None:
            continue
        lines.extend(ChangedLine(path, number, text) for number, text in enumerate(content.splitlines(), 1))
    return lines


def is_test_path(path: Path) -> bool:
    return any(part.lower() in TEST_DIRECTORY_NAMES for part in path.parts)


def scan_changed_lines(lines: list[ChangedLine]) -> list[Finding]:
    findings: list[Finding] = []
    for changed in lines:
        language = language_for(changed.path)
        if language is None:
            continue
        if len(changed.text) > LINE_LIMIT:
            findings.append(Finding(changed.path, changed.number, "error", "G.FMT.02",
                                    f"line has {len(changed.text)} columns; limit is {LINE_LIMIT}"))
        if DELIVERY_MARKER_PATTERN.search(changed.text):
            severity = "warning" if is_test_path(changed.path) else "error"
            findings.append(Finding(changed.path, changed.number, severity, "G.CMT.05",
                                    "remove unfinished-work marker from delivery code"))
        for rule in TEXT_RULES:
            if language in rule.languages and rule.expression.search(changed.text):
                findings.append(Finding(changed.path, changed.number, rule.severity, rule.rule, rule.message))
    return findings


def first_content_line(content: str) -> str:
    lines = content.splitlines()
    return lines[0] if lines else ""


def scan_file_constraints(repo: Path, paths: set[Path], added: set[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(paths):
        language = language_for(path)
        if language is None:
            continue
        try:
            content = read_text_file(repo, path)
        except UnicodeDecodeError:
            findings.append(Finding(path, 1, "error", "G.PRJ.04", "source file is not valid UTF-8"))
            continue
        if content is None:
            continue
        if language in LICENSED_LANGUAGES and LICENSE_MARKER not in "\n".join(content.splitlines()[:12]):
            findings.append(Finding(path, 1, "error", "G.CMT.03", "add the repository license header"))
        if "\r\n" in content and "\n" in content.replace("\r\n", ""):
            findings.append(Finding(path, 1, "error", "G.FMT.10", "do not mix LF and CRLF line endings"))
        if language == "shell" and first_content_line(content) not in {"#!/usr/bin/env bash", "#!/bin/bash"}:
            findings.append(Finding(path, 1, "error", "G.SH.01", "declare a Bash interpreter on line 1"))
        if language == "batch" and first_content_line(content).strip().lower() != "@echo off":
            findings.append(Finding(path, 1, "error", "G.BAT.03", "put @echo off on line 1"))
        if language == "docker":
            if not re.search(r"(?im)^\s*USER\s+(?!root(?:\s|$))\S+", content):
                findings.append(Finding(path, 1, "error", "G.DOCKER.06", "declare a non-root USER"))
            if re.search(r"(?im)^\s*ADD\s+", content):
                findings.append(Finding(path, 1, "error", "G.DOCKER.08", "use COPY instead of ADD"))
        if path in added and language == "cpp" and path.suffix.lower() in {".h", ".hh", ".hpp", ".hxx"}:
            header = "\n".join(content.splitlines()[:60])
            if "#pragma once" not in header and not re.search(r"(?m)^\s*#ifndef\s+\w+", header):
                findings.append(Finding(path, 1, "error", "G.INC.04-CPP", "add an include guard"))
    return findings


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="repository root")
    parser.add_argument("--base", required=True, type=parse_base,
                        help="target branch or commit used as the diff base")
    parser.add_argument("--fail-on", choices=("error", "warning", "none"), default="error")
    return parser.parse_args()


def should_fail(findings: list[Finding], fail_on: str) -> bool:
    if fail_on == "none":
        return False
    if fail_on == "warning":
        return bool(findings)
    return any(finding.severity == "error" for finding in findings)


def main() -> int:
    arguments = parse_arguments()
    repo = Path(arguments.repo).expanduser().resolve(strict=True)
    base = arguments.base
    run_git(repo, ["rev-parse", "--verify", f"{base}^{{commit}}"])
    paths, added = changed_paths(repo, base)
    supported = {path for path in paths if language_for(path) is not None}
    tracked_diff = run_git(repo, ["diff", "--unified=0", "--no-color", "--no-ext-diff",
                                  base, "--"])
    untracked = added - nul_paths(run_git(repo, ["diff", "--name-only", "--diff-filter=A", "-z",
                                                 base, "--"]))
    lines = parse_diff(tracked_diff) + untracked_lines(repo, untracked)
    findings = scan_changed_lines(lines) + scan_file_constraints(repo, supported, added)
    severity_order = {"error": 0, "warning": 1, "note": 2}
    findings.sort(key=lambda item: (item.path.as_posix(), item.line, severity_order[item.severity], item.rule))
    for finding in findings:
        print(f"{finding.path}:{finding.line}: {finding.severity} {finding.rule} {finding.message}")
    errors = sum(finding.severity == "error" for finding in findings)
    warnings = sum(finding.severity == "warning" for finding in findings)
    print(f"checked_files={len(supported)} errors={errors} warnings={warnings}")
    return 1 if should_fail(findings, arguments.fail_on) else 0


if __name__ == "__main__":
    raise SystemExit(main())
