# Quality Gates

## Contents

- [Changed-code gates](#changed-code-gates)
- [Repository trend metrics](#repository-trend-metrics)
- [Named design findings](#named-design-findings)
- [Evidence requirements](#evidence-requirements)

## Changed-Code Gates

Require new or materially changed code to meet these targets:

- average file length below 300 logical code lines;
- average function length below 30 logical code lines;
- average cyclomatic complexity below 5;
- total changed-code duplication below 10%;
- changed source-file duplication below 4%;
- duplicated source lines below 10%;
- redundant-code-block density of zero;
- unsafe-function density of zero;
- no new compiler-warning suppression without a reviewed, narrow toolchain justification.

Do not game averages by splitting coherent code into meaningless fragments. Prefer cohesive types,
single-purpose functions, shared helpers with clear ownership, and removal of dead/redundant code.

## Repository Trend Metrics

Track analyzer-provided metrics by language:

- `code_duplication_ratio`, `file_duplication_ratio`, `non_hfile_code_duplication_ratio`,
  `non_hfile_duplication_ratio`, `duplication_file`;
- `cyclomatic_complexity_per_method`, `huge_cyclomatic_complexity`;
- `lines_per_file`, `lines_per_method`, `huge_method`, `huge_headerfile`,
  `huge_non_headerfile`, `huge_folder`, `huge_depth`;
- `redundant_code`, `redundant_code_kloc`;
- `unsafe_function`, `unsafe_functions_kloc`;
- `warning_suppression`.

Continuous-improvement thresholds from the catalog:

- oversized directory ratio below `0.04%`;
- oversized header ratio below `1%`;
- oversized source-file ratio below `1%`;
- oversized function ratio below `4%`;
- very-high-cyclomatic-complexity function ratio below `1%`.

Use the analyzer's configured definitions for “oversized” and “very high complexity”; the supplied
catalog does not define their absolute thresholds. Do not invent them.

## Named Design Findings

Treat these as review prompts that require structural evidence:

- god file/class, complex file/class, split-personality file/class;
- traditional breaker, shotgun surgery, feature envy, data clumps;
- refused bequest, unstable dependency, confused inheritance hierarchy, cyclic dependency;
- constructor allocation without destructor release;
- misplaced allocation arithmetic parentheses;
- accidental precision loss through integer division;
- an intended override that fails because its signature differs;
- unsafe cryptography, random seeding, key reuse, padding, and service configuration findings.

Apply the catalog's concrete security constraints when the corresponding API is present:

- `SecA_Ascend_GEDeprecatedLowPerformanceInterface` and
  `SecA_Ascend_GERecommandHighPerformanceInterface`: replace an API only after proving semantic and
  supported-version equivalence.
- For password hashing with scrypt, require `N >= 2^14`, salt length at least 16 bytes, `r >= 8`,
  `p >= 1`, and output length at least 256 bits.
- Use an approved modern algorithm and mode. Apply CMS-Padding or ISO-Padding when the selected
  block mode requires padding.
- Do not reuse one symmetric key for encryption and MAC operations.
- Do not seed security randomness from system time or post-process CSPRNG output in a way that
  reduces its security.
- Treat IPSI algorithm findings, weak-algorithm findings, and common-service configuration findings
  as blocking until the approved product policy confirms the configuration.
- Release deliverables that are required to be native binaries must have the expected ELF or PE
  format; do not apply that requirement to scripts, data, or documentation.

Translate tool labels into a concrete path, symbol, dependency edge, or data flow. Do not report a
translated smell name alone.

## Evidence Requirements

For EChecker, SecK, SecL, Ascend API recommendations, and security TOP findings:

1. Preserve the exact analyzer rule name and source location.
2. Reproduce or trace the path from source to sink.
3. Identify validation, bounds, lifetime, lock, or cryptographic invariants already present.
4. Classify the finding as real, conditional, or false positive with evidence.
5. Fix the root cause and add a focused test when real.
6. Use suppression only when the project has an approved mechanism and the justification is local,
   stable, and reviewable.

Do not claim a rule passed merely because the fast changed-code checker emitted no finding.
