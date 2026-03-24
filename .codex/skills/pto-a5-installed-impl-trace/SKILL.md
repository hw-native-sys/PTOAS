---
name: pto-a5-installed-impl-trace
description: Trace PTO A5 behavior from the installed CANN/PTO implementation under ASCEND_HOME_PATH before trusting repo-local lowering or emitter code. Use when the user asks which installed A5 implementation branch is authoritative, which builtin wrapper a PTO/A5 op maps to, or whether repo lowering matches installed PTO behavior.
---

# PTO A5 Installed Implementation Trace

Use this skill when the task is specifically about:
- checking what an A5 PTO op really does on the installed machine
- mapping PTO/A5 behavior to builtins or LLVM/HIVM intrinsics
- tracing PTO wrappers down to CCE builtin wrappers such as `__builtin_cce_*`
- deciding whether repo-local lowering is correct or only a guess
- resolving conflicts between generated repo IR and installed PTO headers
- tracing `Cmp`, `Cmps`, predicate, pack, store, or typed vector behavior

## Strong Rule

If you are about to change repo code for an A5 op, stop and inspect the
installed PTO implementation first. Treat the installed PTO library under
`ASCEND_HOME_PATH` as the semantic source of truth.

Only make a repo-local substitution after you have confirmed one of:
- the installed PTO headers already express that replacement relationship
- the frontend/compiler intrinsic contract proves two forms are equivalent at
  the intrinsic layer

Do not guess behavior from repo-local lowering, emitter code, or from what
"seems plausible" for an intrinsic sequence.

Do not start from repo-local lowering when the question is about real A5
behavior. The installed PTO implementation under `ASCEND_HOME_PATH` is the
first source of truth.

## Required Search Order

Always follow this order:

1. `source /usr/local/Ascend/cann/set_env.sh`
2. confirm `ASCEND_HOME_PATH`
3. inspect installed PTO dispatch headers:
   - `$ASCEND_HOME_PATH/aarch64-linux/include/pto/common/pto_instr_impl.hpp`
4. inspect the matching A5 implementation:
   - `$ASCEND_HOME_PATH/aarch64-linux/include/pto/npu/a5/T*.hpp`
5. inspect typed helpers:
   - `$ASCEND_HOME_PATH/aarch64-linux/include/pto/npu/a5/utils.hpp`
6. inspect builtin wrapper headers when the question is about the real compiler-facing builtin:
   - `$ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/*/include/__clang_cce_vector_intrinsics.h`
   - `$ASCEND_HOME_PATH/tools/bisheng_compiler/lib/clang/*/include/npu_arch_*/__clang_cce_vector_intrinsics.h`
7. only then compare against repo-local code under `lib/PTO/Transforms/`

## Trace By The Real Type Split

Do not infer the active implementation from the final storage type alone.
Follow the source element type and the installed dispatch branch.

Example:
- for `Cmp` with `f32 -> ui8`, inspect the `sizeof(src) == 4` branch, not the
  `ui8` destination branch
- for scalar or packed outputs, treat pack/store ops separately from compare
  predicate generation

Typical A5 compare split:
- 32-bit source elements -> `TCmp_32B` / `TCmps_32B`
- 16-bit source elements -> 16-bit branch
- 8-bit source elements -> 8-bit branch

## What To Extract

When tracing an op, capture:
- the installed PTO entrypoint that handles it
- the exact typed branch that matches the user case
- the builtins used in order
- any typed helper that explains `pset/plt` or store packing selection
- the compiler builtin wrapper if it is visible in installed Clang headers

For compare-family questions, separate:
- predicate generation
- compare builtin
- predicate pack/interleave
- predicate store

Stop at the builtin wrapper layer if the lower compiler implementation is not
available. That is still enough to answer questions such as:
- `pset_b32 -> __builtin_cce_pset_b32`
- `plt_b32 -> __builtin_cce_plt_b32_v300`

## When The Builtin Name Is Still Not Enough

If the installed PTO headers tell you the wrapper builtin but that still does
not answer the LLVM/HIVM operand contract, do not guess from repo-local
lowering. Extend the trace using the generated sample kernel and the real
compiler frontend:

1. generate the sample testcase kernel that already compiles on this machine
2. inspect the testcase build flags from:
   - `<testcase>/build/CMakeFiles/<target>.dir/flags.make`
   - `<testcase>/build/CMakeFiles/<target>.dir/build.make`
3. rerun the same `bisheng` compile with `-v` and `-save-temps`
4. inspect:
   - `*.ccei` for the exact installed PTO wrapper call sequence
   - `strings *.bc | rg 'llvm.hivm\\.'` to see which HIVM intrinsics survived
5. if needed, rerun the same frontend compile with `-S`, `-emit-llvm`, or the
   equivalent `cc1` invocation from `-v` to inspect the real LLVM IR emitted by
   the compiler frontend before instruction selection

This is the required fallback when the question is really:
- what exact `llvm.hivm.*` intrinsic shape the compiler expects
- whether a hand-written LLVM IR call shape is valid
- whether a selector failure is caused by a guessed mask/value form

Prefer this real-frontend route over inventing mask constants or argument
shapes from memory.

## Reporting Back

When you use this skill, report:
- the exact installed header paths inspected
- which typed branch was the authoritative one
- the builtin sequence observed there
- the builtin wrapper name if you found one in the installed Clang headers
- whether repo-local lowering matches or diverges
- the first concrete mismatch, if any
