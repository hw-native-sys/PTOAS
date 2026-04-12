---
name: llvm-test-tool-fallback
description: When `lit` or `FileCheck` is missing from the current shell, look for the corresponding LLVM test tools in the environment or existing LLVM workspace before treating it as a repo issue.
---

# LLVM Test Tool Fallback

Use this skill when:
- `python3 -m lit` fails because `lit` is missing
- `FileCheck` is not in `PATH`
- a test command fails only because LLVM test tools are not available in the current shell

Rule:
- do not stop at `command not found`
- first try to find `lit` / `FileCheck` from the environment's LLVM toolchain or an existing LLVM workspace
- treat missing `lit` / `FileCheck` as an environment-tool issue, not as a PTOAS regression
