# ptobc maintenance notes (PTOAS)

This tool encodes/decodes PTO-BC v0.

## v0 compatibility contract

PTO-BC v0 files are expected to remain readable across PTOAS builds. Never
change the payload schema of an assigned opcode while keeping version 0. When
an IR op gains optional operands, preserve the shipped opcode payload and use
either a new opcode, a legacy wire alias, or the generic v0 compatibility
encoding for forms that do not fit the old schema.

`tools/ptobc/generated/ptobc_opcodes_v0.h` is the checked-in authoritative
schema table. The generator named by older header comments is not present in
this repository, so table changes are currently maintained and reviewed by
hand.

## When you change the PTO dialect / IR
If you change any of the following:
- `include/PTO/IR/PTOOps.td` (add/remove ops, rename mnemonics)
- operand counts / region structure / immediates semantics

…then you **must** update the PTO-BC v0 opcode/schema table without changing
existing wire payloads and ensure tests pass.

## Required gates
Run (or rely on CI):
- `ctest -R ptobc_stage9_e2e`
- `ctest -R ptobc_to_ptoas_smoke`
- `ctest -R ptobc_opcode_coverage_check`
- `ctest -R ptobc_v0_fp_schema_compatibility_check`
- `ctest -R ptobc_tfillpad_legacy_v0_decode`
- `ctest -R ptobc_fp_extended_v0_compatibility`

CI additionally builds the last pre-unification v0 reader at commit
`9c49c3697de35d4b36e2abc5a00da0b264ae1bb6` and runs the extended FP test as
new writer -> legacy reader -> legacy lowering. Update that pin only when the
minimum supported v0 reader changes deliberately.

## Notes
- `ptobc_opcode_coverage_check` is a heuristic based on `mnemonic = "..."` occurrences.
  If PTOOps.td patterns change, update `tools/ptobc/tests/opcode_coverage_check.py`.
