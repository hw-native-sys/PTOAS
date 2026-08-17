#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import dataclasses
import pathlib
import re
import subprocess
import sys
import tempfile


@dataclasses.dataclass(frozen=True)
class PinTarget:
    repo_url: str
    compatibility: str


TARGETS = {
    "gitcode-default": PinTarget(
        repo_url="https://gitcode.com/cann/pto-isa.git",
        compatibility="default CANN CI, container, compile-only, and remote validation",
    ),
    "github-ci-sim": PinTarget(
        repo_url="https://github.com/hw-native-sys/pto-isa.git",
        compatibility="GitHub CPU-simulator CI",
    ),
    "cann90-dev": PinTarget(
        repo_url="https://gitcode.com/cann/pto-isa.git",
        compatibility="CANN 9.0 development container",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update one repo-aware pto-isa pin target without rollback."
    )
    parser.add_argument(
        "--target",
        choices=TARGETS,
        required=True,
        help="Pin target to verify or update.",
    )
    parser.add_argument(
        "--repo-url",
        help="Override the target repository URL (primarily for testing).",
    )
    parser.add_argument(
        "--commit",
        help="Commit SHA to pin. If omitted, resolves the current remote HEAD.",
    )
    parser.add_argument(
        "--ci-workflow",
        default=".github/workflows/ci.yml",
        help="Path to the CI workflow file.",
    )
    parser.add_argument(
        "--dockerfile",
        default="docker/Dockerfile",
        help="Path to the Dockerfile that vendors pto-isa.",
    )
    parser.add_argument(
        "--compile-only-guide",
        default="docs/no_npu_compile_only_guide_zh.md",
        help="Path to the no-NPU compile-only guide.",
    )
    parser.add_argument(
        "--remote-validation-script",
        default="test/npu_validation/scripts/run_remote_npu_validation.sh",
        help="Path to the remote NPU validation runner that falls back to a pinned pto-isa commit.",
    )
    parser.add_argument(
        "--ci-sim-workflow",
        default=".github/workflows/ci_sim.yml",
        help="Path to the GitHub CPU-simulator workflow.",
    )
    parser.add_argument(
        "--dev-dockerfile",
        default="docker/Dockerfile.dev",
        help="Path to the CANN 9.0 development Dockerfile.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that all pinned locations already match the target commit.",
    )
    return parser.parse_args()


def resolve_head_commit(repo_url: str) -> str:
    out = subprocess.check_output(
        ["git", "ls-remote", repo_url, "HEAD"],
        text=True,
    ).strip()
    sha = out.split()[0] if out else ""
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RuntimeError(f"failed to resolve HEAD for {repo_url!r}: {out!r}")
    return sha


def validate_commit(commit: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError(f"expected a full lowercase commit SHA, got {commit!r}")


def run_git(args: list[str], cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", "protocol.file.allow=always", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def verify_descendant(repo_url: str, current: str, candidate: str) -> None:
    validate_commit(current)
    validate_commit(candidate)
    with tempfile.TemporaryDirectory(prefix="pto-isa-pin-") as temp_dir:
        repo_dir = pathlib.Path(temp_dir)
        init = run_git(["init", "--bare", "--quiet"], repo_dir)
        if init.returncode != 0:
            raise RuntimeError(f"failed to initialize temporary repository: {init.stderr.strip()}")

        for revision in dict.fromkeys((current, candidate)):
            fetched = run_git(
                ["fetch", "--quiet", "--no-tags", repo_url, revision], repo_dir
            )
            if fetched.returncode != 0:
                raise RuntimeError(
                    f"revision {revision} is not reachable from {repo_url}: "
                    f"{fetched.stderr.strip()}"
                )

        ancestry = run_git(
            ["merge-base", "--is-ancestor", current, candidate], repo_dir
        )
        if ancestry.returncode == 1:
            raise RuntimeError(
                f"refusing non-fast-forward pin update: {candidate} is not a "
                f"descendant of current pin {current} in {repo_url}"
            )
        if ancestry.returncode != 0:
            raise RuntimeError(
                f"failed to compare {current} and {candidate} in {repo_url}: "
                f"{ancestry.stderr.strip()}"
            )


def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: pathlib.Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def replace_exactly_once(
    text: str, pattern: str, replacement: str, path: pathlib.Path
) -> str:
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one match for pattern {pattern!r} in {path}, got {count}"
        )
    return new_text


def replace_exact_count(
    text: str,
    pattern: str,
    replacement: str,
    path: pathlib.Path,
    expected_count: int,
) -> str:
    new_text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != expected_count:
        raise RuntimeError(
            f"expected {expected_count} matches for pattern {pattern!r} in {path}, got {count}"
        )
    return new_text


def update_ci_workflow(path: pathlib.Path, commit: str) -> bool:
    original = read_text(path)
    updated = original
    updated = replace_exactly_once(
        updated,
        r"(pto_isa_commit:\n(?:\s+.*\n){0,8}?\s+default:\s*)([0-9a-f]{40})",
        rf"\g<1>{commit}",
        path,
    )
    updated = replace_exactly_once(
        updated,
        r"(PTO_ISA_COMMIT:\s*\$\{\{\s*github\.event\.inputs\.pto_isa_commit\s*\|\|\s*')([^']*)('\s*\}\})",
        rf"\g<1>{commit}\g<3>",
        path,
    )
    if updated != original:
        write_text(path, updated)
        return True
    return False


def update_dockerfile(path: pathlib.Path, commit: str) -> bool:
    original = read_text(path)
    updated = original
    updated = replace_exactly_once(
        updated,
        r"^(ARG PTO_ISA_COMMIT=)([0-9a-f]{40})$",
        rf"\g<1>{commit}",
        path,
    )
    updated = replace_exactly_once(
        updated,
        r"^(# pinned: https://gitcode\.com/cann/pto-isa/commit/)([0-9a-f]{40})$",
        rf"\g<1>{commit}",
        path,
    )
    if updated != original:
        write_text(path, updated)
        return True
    return False


def update_compile_only_guide(path: pathlib.Path, commit: str) -> bool:
    original = read_text(path)
    updated = replace_exact_count(
        original,
        r"^(export PTO_ISA_COMMIT=)([0-9a-f]{40})$",
        rf"\g<1>{commit}",
        path,
        expected_count=2,
    )
    if updated != original:
        write_text(path, updated)
        return True
    return False


def update_remote_validation_script(path: pathlib.Path, commit: str) -> bool:
    original = read_text(path)
    updated = replace_exactly_once(
        original,
        r'^(PTO_ISA_COMMIT="\$\{PTO_ISA_COMMIT:-)([0-9a-f]{40})(\}")$',
        rf"\g<1>{commit}\g<3>",
        path,
    )
    if updated != original:
        write_text(path, updated)
        return True
    return False


def extract_ci_commit(path: pathlib.Path) -> tuple[str, str]:
    text = read_text(path)
    default_match = re.search(
        r"pto_isa_commit:\n(?:\s+.*\n){0,8}?\s+default:\s*([0-9a-f]{40})",
        text,
        flags=re.MULTILINE,
    )
    env_match = re.search(
        r"PTO_ISA_COMMIT:\s*\$\{\{\s*github\.event\.inputs\.pto_isa_commit\s*\|\|\s*'([0-9a-f]{40})'\s*\}\}",
        text,
    )
    if not default_match or not env_match:
        raise RuntimeError(f"failed to read pinned pto-isa commit from {path}")
    return default_match.group(1), env_match.group(1)


def extract_docker_commit(path: pathlib.Path) -> tuple[str, str]:
    text = read_text(path)
    arg_match = re.search(r"^ARG PTO_ISA_COMMIT=([0-9a-f]{40})$", text, flags=re.MULTILINE)
    comment_match = re.search(
        r"^# pinned: https://gitcode\.com/cann/pto-isa/commit/([0-9a-f]{40})$",
        text,
        flags=re.MULTILINE,
    )
    if not arg_match or not comment_match:
        raise RuntimeError(f"failed to read pinned pto-isa commit from {path}")
    return arg_match.group(1), comment_match.group(1)


def extract_compile_only_commits(path: pathlib.Path) -> tuple[str, str]:
    matches = re.findall(
        r"^export PTO_ISA_COMMIT=([0-9a-f]{40})$",
        read_text(path),
        flags=re.MULTILINE,
    )
    if len(matches) != 2:
        raise RuntimeError(
            f"expected two pinned pto-isa commits in {path}, got {len(matches)}"
        )
    return matches[0], matches[1]


def extract_remote_validation_commit(path: pathlib.Path) -> str:
    text = read_text(path)
    match = re.search(
        r'^PTO_ISA_COMMIT="\$\{PTO_ISA_COMMIT:-([0-9a-f]{40})\}"$',
        text,
        flags=re.MULTILINE,
    )
    if not match:
        raise RuntimeError(f"failed to read pinned pto-isa commit from {path}")
    return match.group(1)


def extract_ci_sim_commit(path: pathlib.Path) -> str:
    match = re.search(
        r"^\s*PTO_ISA_COMMIT:\s*([0-9a-f]{40})$",
        read_text(path),
        flags=re.MULTILINE,
    )
    if not match:
        raise RuntimeError(f"failed to read pinned pto-isa commit from {path}")
    return match.group(1)


def extract_dev_docker_commit(path: pathlib.Path) -> str:
    match = re.search(
        r"^ARG PTO_ISA_COMMIT=([0-9a-f]{40})$",
        read_text(path),
        flags=re.MULTILINE,
    )
    if not match:
        raise RuntimeError(f"failed to read pinned pto-isa commit from {path}")
    return match.group(1)


def update_single_commit(
    path: pathlib.Path, pattern: str, replacement: str
) -> bool:
    original = read_text(path)
    updated = replace_exactly_once(original, pattern, replacement, path)
    if updated == original:
        return False
    write_text(path, updated)
    return True


def read_target_commit(args: argparse.Namespace) -> str:
    if args.target == "github-ci-sim":
        return extract_ci_sim_commit(pathlib.Path(args.ci_sim_workflow))
    if args.target == "cann90-dev":
        return extract_dev_docker_commit(pathlib.Path(args.dev_dockerfile))

    ci_default, ci_env = extract_ci_commit(pathlib.Path(args.ci_workflow))
    docker_arg, docker_comment = extract_docker_commit(pathlib.Path(args.dockerfile))
    guide_setup, guide_run = extract_compile_only_commits(
        pathlib.Path(args.compile_only_guide)
    )
    remote_validation = extract_remote_validation_commit(
        pathlib.Path(args.remote_validation_script)
    )
    values = {
        ci_default,
        ci_env,
        docker_arg,
        docker_comment,
        guide_setup,
        guide_run,
        remote_validation,
    }
    if len(values) != 1:
        raise RuntimeError(
            "gitcode-default pin locations disagree: " + ", ".join(sorted(values))
        )
    return values.pop()


def update_target(args: argparse.Namespace, commit: str) -> None:
    if args.target == "github-ci-sim":
        update_single_commit(
            pathlib.Path(args.ci_sim_workflow),
            r"^(\s*PTO_ISA_COMMIT:\s*)([0-9a-f]{40})$",
            rf"\g<1>{commit}",
        )
        return
    if args.target == "cann90-dev":
        update_single_commit(
            pathlib.Path(args.dev_dockerfile),
            r"^(ARG PTO_ISA_COMMIT=)([0-9a-f]{40})$",
            rf"\g<1>{commit}",
        )
        return

    update_ci_workflow(pathlib.Path(args.ci_workflow), commit)
    update_dockerfile(pathlib.Path(args.dockerfile), commit)
    update_compile_only_guide(pathlib.Path(args.compile_only_guide), commit)
    update_remote_validation_script(pathlib.Path(args.remote_validation_script), commit)


def verify(
    ci_path: pathlib.Path,
    docker_path: pathlib.Path,
    compile_only_guide_path: pathlib.Path,
    remote_validation_path: pathlib.Path,
    commit: str,
) -> None:
    ci_default, ci_env = extract_ci_commit(ci_path)
    docker_arg, docker_comment = extract_docker_commit(docker_path)
    guide_setup, guide_run = extract_compile_only_commits(compile_only_guide_path)
    remote_validation_commit = extract_remote_validation_commit(
        remote_validation_path
    )
    values = {
        f"{ci_path}:workflow_dispatch_default": ci_default,
        f"{ci_path}:runtime_default": ci_env,
        f"{docker_path}:arg": docker_arg,
        f"{docker_path}:comment": docker_comment,
        f"{compile_only_guide_path}:setup": guide_setup,
        f"{compile_only_guide_path}:run": guide_run,
        f"{remote_validation_path}:fallback": remote_validation_commit,
    }
    mismatches = {name: value for name, value in values.items() if value != commit}
    if mismatches:
        detail = ", ".join(f"{name}={value}" for name, value in mismatches.items())
        raise RuntimeError(f"pto-isa pin mismatch, expected {commit}: {detail}")


def main() -> int:
    args = parse_args()
    target = TARGETS[args.target]
    repo_url = args.repo_url or target.repo_url
    current = read_target_commit(args)
    commit = args.commit or resolve_head_commit(repo_url)
    validate_commit(commit)
    verify_descendant(repo_url, current, commit)

    if args.check:
        if current != commit:
            raise RuntimeError(
                f"{args.target} pin mismatch, expected {commit}, found {current}"
            )
        print(commit)
        return 0

    update_target(args, commit)
    updated = read_target_commit(args)
    if updated != commit:
        raise RuntimeError(
            f"failed to update {args.target} pin to {commit}; found {updated}"
        )
    print(commit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
