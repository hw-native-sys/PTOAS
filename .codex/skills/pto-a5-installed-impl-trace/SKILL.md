---
name: pto-a5-installed-impl-trace
description: Guide LLVM IR discovery for A5 VPTO lowering from the installed CANN/PTO implementation under ASCEND_HOME_PATH. Use when the user does not yet know which `llvm.hivm.*` intrinsic, builtin wrapper, or operand contract a VPTO/A5 op should lower to.
---

# PTO A5 Installed Implementation Trace

Use this skill when the task is specifically about:
- checking what an A5 PTO op really does on the installed machine
- mapping PTO/A5 behavior to builtins or LLVM/HIVM intrinsics
- tracing PTO wrappers down to CCE builtin wrappers such as `__builtin_cce_*`
- deciding whether repo-local lowering is correct or only a guess
- resolving conflicts between generated repo IR and installed PTO headers
- tracing `Cmp`, `Cmps`, predicate, pack, store, or typed vector behavior

This skill answers:
- what LLVM IR a VPTO op should lower to
- what the authoritative intrinsic name is
- what operand list or mask form the installed toolchain expects
- whether repo-local lowering or emission diverges from installed behavior

This skill does not answer:
- how to build or link a finished LLVM-path artifact end to end
- how to package `.o`, `fatobj`, or `.so`
- how to run board validation

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
7. inspect intrinsic name availability directly from the installed compiler binary before guessing LLVM/HIVM spellings:
   - `strings $ASCEND_HOME_PATH/bin/bisheng | rg 'llvm\\.hivm\\.'`
   - narrow to the op under investigation, for example:
     - `strings $ASCEND_HOME_PATH/bin/bisheng | rg 'llvm\\.hivm\\.(vneg|vrsqrt|vnot|vmov)'`
8. only then compare against repo-local code under `lib/PTO/Transforms/`

## Practical Fast Path

For VPTO LLVM emission work, prefer this concrete order instead of jumping
straight to ad hoc compiler probes:

1. confirm the op exists in installed PTO/A5 headers
2. confirm the builtin wrapper shape in installed Clang headers
3. confirm the intrinsic name family with:
   - `strings $ASCEND_HOME_PATH/bin/bisheng | rg 'llvm\\.hivm\\.<op>'`
4. patch repo-local emitter/lowering as little as possible
5. generate real repo-driven LLVM IR through the existing VPTO validation path:
   - `source scripts/ptoas_env.sh`
   - `WORK_SPACE=/tmp/<token> CASE_NAME=<case> DEVICE=SIM COMPILE_ONLY=1 test/vpto/scripts/run_host_vpto_validation.sh`
6. inspect:
   - `<workspace>/<case-token>/*.ll`
   - `<workspace>/<case-token>/validation.log`
7. only after seeing the real generated `.ll` and Bisheng failure should you
   refine the call shape

This route is preferred because it preserves the real PTOAS lowering context,
the real case structure, and the exact driver invocation used by the repo.

## Probe Strategy

Use probes in this order:

1. installed headers
2. `strings bisheng`
3. repo-generated VPTO LLVM IR from `run_host_vpto_validation.sh`
4. only then minimal handwritten `.ll` probes
5. handwritten `.cce` frontend probes are last resort

Handwritten `.ll` probes are acceptable for quick ABI sanity checks such as:
- whether Bisheng recognizes a specific `llvm.hivm.*` name
- whether a guessed argument count immediately crashes or verifies

But they are not the primary source of truth for semantic or frontend wrapper
behavior.

## Avoid These Traps

Do not default to handwritten `.cce` probes when repo-driven IR is available.
On this machine, bare `.cce` probes often fail before reaching the real
question because they are missing the exact frontend driver mode, target
features, wrapper setup, or host/device compilation context used by the repo.

In particular, treat these as warning signs that you have started too low in
the stack:
- errors around `[aicore]`
- errors around `__cce_half`
- builtin alias attribute failures
- missing target feature or wrapper environment failures

When these happen, step back to the repo-driven compile-only flow instead of
trying to repair the ad hoc frontend invocation from scratch.

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
lowering. Extend the trace using the generated repo testcase first, and only
after that the real compiler frontend:

1. run an existing repo case with:
   - `WORK_SPACE=/tmp/<token> CASE_NAME=<case> DEVICE=SIM COMPILE_ONLY=1 test/vpto/scripts/run_host_vpto_validation.sh`
2. inspect the generated `.ll` and `validation.log`
3. if the repo-generated LLVM IR still leaves the contract ambiguous, inspect
   the testcase build flags from:
   - `<testcase>/build/CMakeFiles/<target>.dir/flags.make`
   - `<testcase>/build/CMakeFiles/<target>.dir/build.make`
4. rerun the same `bisheng` compile with `-v` and `-save-temps`
5. inspect:
   - `*.ccei` for the exact installed PTO wrapper call sequence
   - `strings *.bc | rg 'llvm.hivm\\.'` to see which HIVM intrinsics survived
6. if needed, rerun the same frontend compile with `-S`, `-emit-llvm`, or the
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
- whether `strings $ASCEND_HOME_PATH/bin/bisheng` confirmed the intrinsic name
- which typed branch was the authoritative one
- the builtin sequence observed there
- the builtin wrapper name if you found one in the installed Clang headers
- whether repo-generated `.ll` matched the guessed call shape
- whether repo-local lowering matches or diverges
- the first concrete mismatch, if any
