---
name: ptoas-bisheng-asm-from-object-cmd
description: Use when you need assembly for a PTOAS VPTO case that already compiles to a device object. First find the exact command that produced the `.o`, then derive the `.s` command by replacing `-c` with `-S`. Do not guess a fresh Bisheng command line.
metadata:
  short-description: Derive `.s` from real `.o` command
---

# PTOAS Bisheng ASM From Object Command

Use this skill when the task is to inspect generated assembly for a VPTO case and the case already has a known `.o` build path.

## Rule

- Do not invent a new `bisheng` command.
- First find the exact command that built the `.o`.
- Then derive the `.s` command from that exact command by changing `-c` to `-S`.
- Keep the rest of the arguments unchanged unless the original command already wrote to a conflicting output path.

## Preferred Sources

- Validation script logs
- Build scripts such as `test/vpto/scripts/run_host_vpto_validation.sh`
- Saved shell history or generated compile traces in the case workspace

## Procedure

1. Locate the real `.o` compile command for the target case.
2. Copy that command exactly.
3. Replace `-c` with `-S`.
4. Point `-o` to a `.s` path.
5. Run the derived command.
6. Inspect the generated assembly instead of guessing from LLVM IR.

## Anti-Pattern

- Do not hand-write a new `bisheng -S ...` command from memory.
- Do not drop flags such as `--target`, `-march`, `--cce-aicore-arch`, `--cce-aicore-only`, `-O2`, include paths, or wrapper options that were present in the real `.o` command.
