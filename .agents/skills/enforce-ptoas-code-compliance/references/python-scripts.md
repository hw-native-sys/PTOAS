# Python And Script Rules

## Contents

- [Python structure and style](#python-structure-and-style)
- [Python errors and types](#python-errors-and-types)
- [Untrusted data and external execution](#untrusted-data-and-external-execution)
- [Files, serialization, logging, and secrets](#files-serialization-logging-and-secrets)
- [Shell and batch](#shell-and-batch)
- [Build, container, and ecosystem files](#build-container-and-ecosystem-files)

## Python Structure And Style

- `G.CLS.01`, `G.CLS.05`: call the parent initializer correctly, normally with `super()`.
- `G.CLS.02`: keep an override's signature compatible with the base method.
- `G.CLS.08`: define instance attributes in `__init__`, preferably with type annotations.
- `G.CLS.09`: magic methods return the protocol-required type.
- `G.CLS.10`: unsupported numeric magic methods return `NotImplemented`.
- `G.CMT.01`, `G.CMT.03`: place module and public-function docstrings in their canonical locations.
- `G.CMT.03`, `G.PRJ.06`: use the repository header and omit personal information.
- `G.CMT.04`, `G.CMT.05`: keep comments consistent and do not ship TODO/FIXME markers.
- `G.AST.01` through `G.AST.05`, `G.TES.01`: use assertions only for one debug-only internal
  invariant; never mutate state in an assertion or use one for a possible runtime failure.
- `G.FMT.01` through `G.FMT.12`: use four spaces, logical blank lines, one import and statement per
  line, consistent line endings, readable spacing, and at most 120 columns.
- `G.FNM.01`: never use a mutable default argument.
- `G.FNM.02`: do not accidentally close over a changing loop variable.
- `G.FNM.03`, `G.FNM.05`: group large related parameter/result sets in named types.
- `G.FNM.04`: do not assign the result of a no-return function.
- `G.FNM.06`: return from generators instead of raising `StopIteration`.
- `G.IMP.01` through `G.IMP.03`: prefer explicit absolute package imports and do not use
  `__import__`.
- `G.NAM.01`, `G.NAM.03`, `G.NAM.05`: keep naming consistent, use `self`/`cls`, and reserve
  double-underscore magic names for protocols.
- `G.PY.01`, `G.PRJ.04`: use UTF-8 for Python and project text files.
- `G.VAR.01` through `G.VAR.03`: keep a variable's type stable and avoid shadowing/global leakage.
- `G.CTL.01` through `G.CTL.05`: keep branch return shapes consistent, remove unreachable code,
  make loops terminate, keep conditions small, iterate directly, and use `_` for unused loop values.
- `G.EXP.03`, `G.EXP.04`: use named functions for nontrivial behavior and keep comprehensions simple.

## Python Errors And Types

- `G.ERR.03`, `G.ERR.05`, `G.ERR.06`: raise instantiated, business-specific `Exception`
  subclasses.
- `G.ERR.04`: preserve traceback context when translating exceptions.
- `G.ERR.07`: do not swallow exceptions.
- `G.ERR.08`: do not expose secrets in errors.
- `G.ERR.09`, `G.ERR.10`: avoid duplicate catches and order specific handlers before broad ones.
- `G.ERR.11`: use `sys.exit` only at the main entrypoint.
- `G.ERR.13`: propagate with bare `raise` when appropriate; do not use `raise exc`.
- `G.ERR.14`: let `finally` complete normally.
- `G.OPR.01`: guard zero divisors.
- `G.OPR.02`, `G.OPR.03`, `G.OPR.05`, `G.OPR.06`: compare `None` with `is`, values with equality,
  and use `is not`/`not in` idiomatically.
- `G.TYP.01`: construct `Decimal` from exact strings/integers, never binary floats.
- `G.TYP.02`: compare approximate floats with a tolerance, not `==`.
- `G.TYP.04`, `G.TYP.06`, `G.TYP.08`: use truthiness for sequence emptiness, unique dict keys, and
  `isinstance` for runtime type tests.
- `G.PSL.01`, `G.PSL.02`: avoid deprecated APIs and use timezone-aware datetimes when timezones
  matter.

## Untrusted Data And External Execution

- `G.EDV.01`: never evaluate untrusted text with `eval` or `exec`.
- `G.EDV.02`, `G.EDV.04`: do not pass untrusted data through a command interpreter or
  `subprocess(..., shell=True)`.
- `G.EDV.03`: avoid interpreter-expanded wildcards.
- `G.EDV.05`, `G.CNP.01`: resolve executables, pass an argument vector, set a timeout where a child
  can stall, and handle the return code.
- `G.EDV.06`, `G.FUU.18`: parameterize SQL.
- `G.EDV.07`: do not let untrusted templates drive `.format`.
- `G.EDV.08`: bound input and avoid catastrophic-backtracking regexes.
- `G.EDV.09`, `G.EDV.10`: use safe XML construction and disable external entities.
- `G.FUU.02`, `G.FUU.03`: keep format strings trusted and type-correct.
- `G.FUU.16`, `G.FUU.17`: validate externally influenced process arguments and module names.
- `G.FUU.01`: inspect and handle meaningful return values instead of silently discarding failure.

## Files, Serialization, Logging, And Secrets

- `G.FIL.01`, `G.FIO.01`: create files with the minimum required permissions.
- `G.FIL.02`, `G.FIO.02`: resolve/normalize externally influenced paths, constrain them to an
  allowed root, and reject traversal before use.
- `G.FIL.03`: use a private temporary location, not a predictable shared path.
- `G.FIO.03`: use `TemporaryFile`, `NamedTemporaryFile`, or `TemporaryDirectory`, never
  `tempfile.mktemp`.
- `G.FIO.04`: clean temporary files on success and failure.
- `G.FIO.06`: validate archive member paths, types, sizes, and extraction destination.
- `G.SER.01`: do not load untrusted pickle, `_pickle`, or shelve data.
- `G.SER.02`: encrypt authenticated sensitive serialized data.
- `G.SER.03`: use `yaml.safe_load`, not `yaml.load`.
- `G.SER.04`: do not use jsonpickle for untrusted or sensitive data.
- `G.LOG.01`: use logging's lazy interpolation for debug/info paths.
- `G.LOG.02`: use the project logging facility rather than ad-hoc application prints.
- `G.LOG.03`, `G.LOG.04`: sanitize external log data and never log credentials, tokens, or keys.
- `G.DSP.01` through `G.DSP.03`: sign and encrypt sensitive outbound objects, use cryptographically
  secure randomness, and use TLS sockets in security-sensitive network code.
- `G.OTH.03`: never use `rand`-style randomness for security.
- `G.OTH.04`: do not expose object/function addresses in release output.
- `G.OTH.05`, `G.PRJ.07`: do not embed unapproved public endpoints.

## Shell And Batch

- `G.SH.01`: put the selected shell interpreter on the first line; use Bash for Bash syntax.
- `G.SCRIPT.02`, `.04`: keep call and variable-expansion nesting shallow.
- `G.SCRIPT.05`, `.06`: derive paths from the script/repository and avoid fixed installation paths.
- `G.SCRIPT.07`: do not depend on network drives.
- `G.SCRIPT.08`: return zero only on success.
- `G.SCRIPT.09`: include purpose and repository copyright.
- `G.SCRIPT.11`, `.12`: keep the delivery unit's script-language set small and consistent.
- Quote expansions, use arrays for command arguments, reject unsafe external values, and avoid
  `eval`, `bash -c`, wildcard deletion, and command-string construction.
- `G.BAT.01` through `.06`, `.08`: for batch files, use `.bat`, `@echo off`, `rem` comments,
  lowercase snake-case filenames/variables, uppercase constants/environment variables, and `call`
  for subroutines or batch files.

## Build, Container, And Ecosystem Files

Apply these only when the corresponding artifact exists:

- `G.DOCKER.01`, `.02`, `.04` through `.13`: use the approved base/build environment, declared
  tools, `COPY`, a non-root `USER`, maintainer metadata, configurable tool locations, explicit
  `ENV`, and deterministic install order.
- `G.BI.*`, `G.VM.*`, `G.ENV.*`, `G.TOOL.*`: treat base image, managed environment, OS, and tool
  lifecycle requirements as release/platform policy, not ordinary source-style rules.
- `G.PLAYBOOK.*`: preserve the prescribed role layout. Governed installer scripts require exact
  `#!/bin/bash`, `install_dir=$1`, local pre-provisioned packages, validation, and cleanup.
- `G.GO.*`: keep one root build entrypoint, use modules, lock explicit versions, and fail builds on
  task failure. Verify the catalog's legacy `verdor.json` wording against the current approved Go
  policy before adding such a file.
- `G.GRADLE.*`, `G.MAVEN.*`: use conventional root entrypoints, central dependency/version
  management, fixed release versions, dependency locks, UTF-8 POMs, and zero unresolved warnings.
- `G.JS.*`: commit package manifests and lockfiles, use the approved registry for release builds,
  and never commit generated build products.
- `G.MF.*`: apply manifest/dependency/playbook repository rules only to product-release manifests.
- `G.COM.*`, `G.C&C++.*`: provide one build entrypoint, clean/target/parallel support, separated
  source/build/install trees, deterministic configuration, concise per-run logs, and no source-tree
  mutation.
