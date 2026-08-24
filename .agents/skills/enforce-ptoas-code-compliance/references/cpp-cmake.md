# C, C++, And CMake Rules

## Contents

- [Taint, bounds, and memory](#taint-bounds-and-memory)
- [Arithmetic and expressions](#arithmetic-and-expressions)
- [Pointers, resources, and strings](#pointers-resources-and-strings)
- [Classes, exceptions, and concurrency](#classes-exceptions-and-concurrency)
- [Control flow, declarations, and macros](#control-flow-declarations-and-macros)
- [Headers, formatting, and comments](#headers-formatting-and-comments)
- [API, signals, and release behavior](#api-signals-and-release-behavior)
- [CMake and compiler configuration](#cmake-and-compiler-configuration)
- [Review checklist](#review-checklist)

## Taint, Bounds, And Memory

Treat external values as tainted until validated. This covers `EChecker_BufferSize`,
`EChecker_OutOfBound` (product-specific), `EChecker_Overrun`, `EChecker_TaintedArgument`,
`EChecker_TaintedLoopBound`, `EChecker_TaintedPtrDereference`, `SecK_OutOfBoundsChecker`, and
`SecK_NullPointerDereferenceChecker`.

- `G.ARR.01`, `G.RES.01-CPP`, `G.STD.08-CPP`: validate every external array/container index against
  the exact current bound before use.
- `G.FUD.07`, `G.RES.03-CPP`: pass array extent with a pointer, or use a size-carrying view supported
  by C++17, such as `llvm::ArrayRef` or an existing container reference. Do not introduce
  `std::span` without an intentional language-standard upgrade.
- `G.ARR.02`, `G.ARR.03`: never infer an array parameter or pointed-to allocation size with
  `sizeof`.
- `G.MEM.01`, `G.RES.02-CPP`: validate allocation size, including multiplication/addition overflow,
  before allocation.
- `G.MEM.02`, `G.RES.13-CPP`: handle allocation failure according to the selected allocation API.
- `G.MEM.03`: validate externally derived copy/set/compare lengths against source and destination.
- `G.MEM.04`: clear sensitive memory with an operation the compiler cannot optimize away.
- `G.STR.01`, `G.STR.02`, `G.STD.05-CPP`: reserve the terminator and prove termination for C strings.
- `G.STD.09-CPP`, `G.STD.11-CPP`, `G.STD.12-CPP`: preserve iterator validity and destination
  capacity; use erase after remove algorithms.
- `G.FMT.08`, `G.FMT.11-CPP`: use braces for selection and loops, including one-line bodies.

Test zero, one, maximum valid, first invalid, and overflow-adjacent values. A check after pointer
arithmetic or dereference is too late.

## Arithmetic And Expressions

- `G.INT.01`, `G.EXP.20-CPP`: prevent signed overflow.
- `G.INT.02`, `G.EXP.21-CPP`: prevent unintended unsigned wraparound.
- `G.INT.03`, `G.OPR.01`, `G.EXP.22-CPP`: guard division and remainder by zero.
- `G.INT.04`, `G.EXP.26-CPP`: widen operands before evaluation, not only the result.
- `G.INT.05`, `G.EXP.23-CPP`: perform bitwise operations on unsigned integers.
- `G.INT.06`: range-check external integers before conversion or use.
- `G.INT.07`, `G.EXP.24-CPP`: validate shift counts against zero and the promoted left-operand width.
- `G.INT.09`: keep enum values unique unless aliases are intentional and documented.
- `G.INT.10`, `G.EXP.17-CPP`, `G.EXP.25-CPP`: make narrowing and signed/unsigned conversions explicit
  and prove their range.
- `G.EXP.01`, `G.TYP.02`: use compatible basic types for arithmetic and comparisons.
- `G.EXP.04`, `G.EXP.30-CPP`: use parentheses where precedence is not immediately obvious.
- `G.EXP.05`: do not pass side-effecting expressions to `sizeof`.
- `G.EXP.06`: do not assume a particular bit-field layout.
- `G.EXP.12-CPP`: bit-copy only trivially copyable objects.
- `G.EXP.13-CPP`: use character types for characters.
- `G.EXP.14-CPP`: use C++ casts; avoid `reinterpret_cast` and `const_cast`
  (`G.EXP.15-CPP`, `G.EXP.16-CPP`).
- `G.EXP.18-CPP`, `G.ARR.04`: avoid integer/pointer conversion.
- `G.ARR.05`: do not force-convert unrelated object-pointer types.
- `G.ARR.06`: do not introduce variable-length arrays.
- `G.TYP.02`, `G.C&C++.WARN.14`: do not use direct floating-point equality for approximate values.

## Pointers, Resources, And Strings

- `G.ARR.08`, `G.FUD.08`, `G.RES.15-CPP`: establish non-nullness before every nullable dereference.
- `G.MEM.05`, `SecK_UseAfterFreeChecker`: never access released storage.
- `G.PRM.03`, `G.VAR.08`, `SecK_MemoryAndResourceLeakChecker`: pair acquisition and release on all
  normal and exceptional exits.
- `G.RES.04-CPP`, `G.VAR.06`: do not let local addresses escape their lifetime.
- `G.RES.05-CPP`, `G.RES.06-CPP`: escaping lambdas must not capture locals by reference; avoid
  default capture.
- `G.RES.07-CPP`, `G.VAR.05`: reset non-owning handles after release.
- `G.RES.08-CPP`: use RAII for ownership.
- `G.RES.09-CPP`, `G.RES.10-CPP`: use `make_unique` and `make_shared`.
- `G.RES.11-CPP`, `G.RES.12-CPP`: pair allocation/deallocation forms and custom operators.
- `G.STD.02-CPP`: prefer `std::string` for ordinary text.
- `G.STD.03-CPP`: do not construct `std::string` from a nullable pointer.
- `G.STD.04-CPP`: do not retain `c_str()` or `data()` pointers across invalidating operations.
- `G.STD.07-CPP`: do not retain secrets in ordinary strings.
- `G.FUU.09`, `G.FUU.10`: do not use `realloc` or `alloca`.
- `G.FUU.21` and unsafe-function metrics: do not introduce unbounded C memory/string operations.

When a low-level API is unavoidable, prove destination capacity, source availability, overlap
requirements, and return-value handling at the call site.

## Classes, Exceptions, And Concurrency

- `G.CLS.01-CPP`, `G.CLS.02-CPP`: initialize every member at declaration or in the constructor
  initialization list.
- `G.CLS.03-CPP`: mark converting single-argument constructors `explicit`.
- `G.CLS.04-CPP`, `G.CLS.05-CPP`: define/delete copy and move operation pairs consistently.
- `G.CLS.06-CPP`: do not dispatch virtual functions from constructors or destructors.
- `G.CLS.07-CPP`: prevent public copying/moving of polymorphic bases unless explicitly safe.
- `G.CLS.09-CPP`: leave moved-from owners valid and resource-safe.
- `G.CLS.10-CPP`: give polymorphic bases virtual destructors when deletion through the base is
  supported.
- `G.CLS.11-CPP`: do not redefine inherited virtual default arguments.
- `G.CLS.12-CPP`: use `override` or `final`.
- `G.CLS.13-CPP`, `G.CLS.14-CPP`, `G.CLS.15-CPP`: do not hide inherited non-virtual APIs, decay
  derived arrays to base pointers, or overload comma/logical operators.
- `G.CNS.03-CPP`, `G.CNS.04-CPP`: apply `const` to observers and read-only pointees/references.
- `G.ERR.01-CPP` through `G.ERR.07-CPP`: throw standard-exception-derived objects by value, catch by
  reference, order handlers most-derived first, do not throw from destructors, and do not use
  dynamic exception specifications.
- `G.CON.01-CPP`: wait on condition variables with a predicate or a loop.
- `G.CON.02-CPP`: prefer scoped lock wrappers over direct mutex lock/unlock calls.
- `SecL_DataRace`: synchronize every shared mutable access with a documented ownership/locking rule.

## Control Flow, Declarations, And Macros

- `G.CTL.01`, `G.EXP.36-CPP`: control expressions are boolean.
- `G.CTL.02`, `G.EXP.31-CPP`, `G.EXP.32-CPP`, `G.EXP.33-CPP`: do not rely on skipped operands or
  combine side effects with short-circuiting/increment expressions.
- `G.CTL.03`: every loop has a provable exit or an intentional service-loop contract.
- `G.CTL.04`, `G.EXP.40-CPP`: never use floating-point loop counters.
- `G.CTL.06`, `G.EXP.42-CPP`: avoid `goto`; if legacy code requires it, never jump into a scope or
  upward into repeated execution.
- `G.CTL.07`, `G.EXP.37-CPP`: include a deliberate `default`, even when it only reports an invalid
  state.
- `G.CTL.08`, `G.EXP.38-CPP`: do not use a switch for a single condition.
- `G.DCL.01`, `G.EXP.01-CPP`: do not define reserved identifiers.
- `G.EXP.02-CPP`, `G.TYP.01`: do not redefine fundamental types.
- `G.EXP.03-CPP`: prefer `using` aliases.
- `G.EXP.04-CPP`: preserve the one-definition rule.
- `G.EXP.07-CPP`: do not depend on cross-translation-unit global initialization order.
- `G.EXP.08-CPP`, `G.EXP.09-CPP`, `G.VAR.01`: initialize before use and declare near first use.
- `G.EXP.10-CPP`, `G.VAR.02`: do not shadow names in nested scopes.
- `G.EXP.19-CPP`: do not move from const objects.
- `G.EXP.35-CPP`: use `nullptr`.
- `G.EXP.43-CPP`, `G.OTH.01`, `G.PRJ.05`: delete dead code instead of commenting it out.
- `G.ENU.01-CPP`, `G.ENU.02-CPP`: prefer named scoped enums.
- `G.PRE.01-CPP`: use typed constants, not constant macros.
- `G.PRE.02-CPP`: prefer functions to function-like macros.
- `G.PRE.03-CPP` through `G.PRE.05-CPP`: make preprocessor conditions explicitly boolean, guard
  identifiers with `defined`, and keep matching branches in one file.
- General `G.PRE.*`: parenthesize macro parameters/results, avoid side-effecting arguments and
  control-flow macros, do not shadow keywords, do not embed directives in arguments, and omit a
  trailing semicolon.

## Headers, Formatting, And Comments

- `G.ARR.07`: specify the bound on externally linked array declarations.
- `G.INC.01-CPP` through `G.INC.12-CPP`: prevent cycles, include only needed self-contained headers,
  use guards or the established equivalent, do not include inside `extern "C"`, order includes,
  avoid global using-directives and header-local anonymous/static definitions, and hide
  translation-unit-only symbols.
- `G.FUD.01`, `G.FUN.02-CPP`: keep declaration/definition names and qualifiers identical.
- `G.FUD.02`: prefer return values to output parameters.
- `G.FUD.09`: avoid modifying parameter variables; use a local value when transformation is needed.
- `G.FUN.01-CPP`: keep functions single-purpose.
- `G.FUN.03-CPP`: remove unused parameters or use a framework-approved explicit marker.
- `G.FUN.04-CPP`: avoid C-style variadic functions.
- `G.FUN.07-CPP`: do not `std::move` a returned local.
- `G.FMT.*`: use four-space indentation, one statement per line, braces, consistent line endings,
  project-consistent brace/pointer style, useful spacing, and a 120-column maximum unless a
  non-wrappable token makes that impossible.
- `G.CMT.03-CPP`, `G.CMT.04-CPP`, `G.CMT.05-CPP`: use the repository copyright header, avoid empty
  ceremonial comments, and do not ship TODO/TBD/FIXME markers.
- `G.CNS.01-CPP`: use uppercase `L`, not lowercase `l`, for integer suffixes.
- `G.CNS.02-CPP`: replace unexplained literals with named typed constants.
- `G.NAM.03-CPP`: follow one naming style within the component.
- `G.STD.01-CPP`: use current standard-library headers.
- `G.TMP.01-CPP`: keep template definitions and explicit specializations with their template.
- `G.VAR.03`: avoid large stack allocations.

## API, Signals, And Release Behavior

- `G.FUU.01`, `G.FUU.11`, `G.FUU.12`: check relevant return values and pass the true destination
  capacity to bounded APIs.
- `G.FUU.13` through `G.FUU.15`: do not wrap or macro-rename approved secure functions; use only
  the project-approved safe-function implementation when that policy applies.
- `G.FUU.04` through `G.FUU.08`, `G.STD.16-CPP`: do not use `atexit`, abort-style termination, or
  process/thread exit functions outside an approved program entrypoint.
- `G.FUU.05`, `G.STD.17-CPP`: do not directly terminate another process.
- `G.FUU.16`, `G.FUU.17`, `G.STD.15-CPP`: validate externally influenced process arguments and
  dynamic-module names.
- `G.FUU.19`, `G.OTH.02`, `G.STD.19-CPP`: call only async-signal-safe operations in signal handlers
  and do not access unsafe shared objects.
- `G.FUU.20`, `G.STD.18-CPP`: avoid time-of-check/time-of-use and library-call races.
- `G.STD.13-CPP`, `G.STD.14-CPP`: use valid, trusted format strings.
- `G.PRJ.03`: do not ship product debug entrypoints.
- `G.PRJ.04`: keep text source in the project's UTF-8 encoding.
- `G.OTH.03`: never use weak pseudo-random generators for security.
- `G.OTH.04`: do not expose object or function addresses in release output.
- `G.OTH.05`, `G.PRJ.07`: do not embed unapproved public endpoints.

## CMake And Compiler Configuration

- `G.CMake.01` through `G.CMake.07`: keep each `CMakeLists.txt` with its source directory, recurse
  with `add_subdirectory`, include only `.cmake` modules, and never include a `CMakeLists.txt`.
- `G.CMake.10` through `G.CMake.13`: use explicit compiler-specific toolchain files through
  `CMAKE_TOOLCHAIN_FILE`; keep project options out of them.
- `G.CMake.17`, `G.CMake.18`: make `cmake_minimum_required` and `project` the first project commands
  after the required license header and comments.
- `G.CMake.19`, `G.CMake.20`: use lowercase commands, uppercase built-in properties, and do not
  prefix custom variables with `CMAKE`.
- `G.CMake.22`, `G.CMake.24`: avoid deprecated syntax; place file-scope commands before targets and
  prefer target-scoped commands.
- `G.CMake.25`, `G.CMake.26`: use target sources with unambiguous paths across directories.
- `G.CMake.27`: use source, binary, and current-list directory variables for their intended trees.
- `G.C&C++.01`, `.04`, `.07`, `.08`, `.09`, `.12`, `G.COM.01`, `.02`, `.03`: preserve out-of-source
  builds, arbitrary install prefixes, one build entrypoint, target selection, parallel builds, a
  clean target, and no source-tree mutation.
- `G.C&C++.LANG.01`: explicitly select the language standard.
- `G.C&C++.LANG.04`: never add `-fpermissive`.
- `G.C&C++.WARN.01`, `.02`, `.04`, `.05`, `.06`, `.14`: enable useful warnings; do not use `-w`,
  blanket `-Wno-*`, or warning-error downgrades. Inspect every narrow suppression.
- `G.C&C++.SEC.01` through `.06`, `.09`: apply supported stack, PIE/ASLR, RELRO, non-executable
  stack, symbol-stripping, runtime-search-path, and SafeSEH policies to the appropriate release
  platform. Do not pass unsupported flags to every toolchain.
- `G.C&C++.CDG.01`: use `-fno-common` where supported for C targets.
- `G.COM.07`, `.08`: produce concise leveled logs and keep per-build logs distinguishable.
- `G.COM.10`: build as a non-system user in managed build environments.

## Review Checklist

- Taint sources and all sensitive sinks are mapped.
- Bounds checks occur before address computation and dereference.
- Size arithmetic is checked in the type that performs the operation.
- Nullability, ownership, iterator validity, and cleanup are explicit.
- Error paths are tested and do not depend on assertions.
- New code introduces no unsafe functions, warning suppression, or undefined behavior.
- Return values, signal behavior, process/module inputs, and release diagnostics are safe.
- CMake changes are target-scoped and preserve supported toolchains.
- Relevant compiler, sanitizer/static-analysis, unit, lit, and board tests pass.
