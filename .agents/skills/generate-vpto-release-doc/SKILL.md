---
name: generate-vpto-release-doc
description: Generate or refresh bundled PTO ISA reference docs. Packages `docs/vpto-spec.md` together with `docs/isa/micro-isa/*.md` into the main repo doc `docs/PTO-micro-Instruction-SPEC.md`, packages `docs/isa/tile-op/*.md` into the companion repo doc `docs/PTO-tile-Instruction-SPEC.md`, and still supports emitting a legacy versioned snapshot under `docs/release/`.
---

# Generate VPTO Release Doc

Use this skill when the task is specifically about:
- creating or refreshing the main bundled micro-spec `docs/PTO-micro-Instruction-SPEC.md`
- creating or refreshing the standalone Tile Instruction bundle `docs/PTO-tile-Instruction-SPEC.md`
- emitting a legacy versioned snapshot under `docs/release/`
- regenerating downstream bundled docs from PTOAS sources through the same script

The main bundled micro-spec starts from `docs/vpto-spec.md`, strips draft-only metadata and appendix content, then inlines every chapter from `docs/isa/micro-isa/` under a dedicated detailed-reference section. `docs/PTO-tile-Instruction-SPEC.md` stays separate and all cross-doc links are rewritten to point at the bundled filenames.

Do not hand-edit bundled outputs; regenerate them through the script so link rewriting stays reproducible.

## Canonical Workflow

1. Pick the target version. The repo-doc bundle filenames do not carry a version suffix; the version is recorded in the version-history bullets inside each file.

Default output paths:

```bash
docs/PTO-micro-Instruction-SPEC.md   # main bundled micro-spec
docs/PTO-tile-Instruction-SPEC.md    # standalone Tile Instruction bundle
docs/release/vpto-spec-v<version>.md # optional legacy versioned snapshot
```

2. Run the bundled generator script.

Generate both repo-doc bundles (default):

```bash
python3 .agents/skills/generate-vpto-release-doc/scripts/generate_release_vpto_spec.py --version 0.4
```

Generate just one repo-doc bundle:

```bash
python3 .agents/skills/generate-vpto-release-doc/scripts/generate_release_vpto_spec.py --version 0.4 --target micro
python3 .agents/skills/generate-vpto-release-doc/scripts/generate_release_vpto_spec.py --version 0.4 --target tileop
```

Generate only the legacy versioned snapshot:

```bash
python3 .agents/skills/generate-vpto-release-doc/scripts/generate_release_vpto_spec.py --version 0.4 --target merged
```

Generate downstream docs into another directory while preserving old PTO-Gym filenames:

```bash
python3 .agents/skills/generate-vpto-release-doc/scripts/generate_release_vpto_spec.py \
  --version 0.4 \
  --output-dir 3rdparty/PTO-Gym/docs \
  --micro-output-name PTO-micro-Instruction-SPEC.md \
  --tileop-output-name PTO-tile-Instruction-SPEC.md

python3 .agents/skills/generate-vpto-release-doc/scripts/generate_release_vpto_spec.py \
  --version 0.4 \
  --target merged \
  --output-dir 3rdparty/PTO-Gym/docs \
  --micro-output-name PTO-micro-Instruction-SPEC.md \
  --tileop-output-name PTO-tile-Instruction-SPEC.md \
  --merged-output-name vpto-spec.md
```

Custom version-bullet text:

```bash
python3 .agents/skills/generate-vpto-release-doc/scripts/generate_release_vpto_spec.py \
  --version 0.4 \
  --version-note 'Custom release note for this run'
```

3. Review each generated file.

Invariants for the main bundle (`docs/PTO-micro-Instruction-SPEC.md`):
- exactly one `#` level title at the top
- `[toc]` is present near the top
- the requested-version bullet is at the top of the version-history list
- beginning-of-file draft metadata (`Status`, `Base`, `Updated`) is removed
- appendix content is removed
- `docs/vpto-spec.md` contributes the overview / notation / summary sections
- `## Detailed ISA Group Reference` exists and inlines every chapter from `docs/isa/micro-isa/` in sorted order
- chapter headings are demoted by two levels so each `# N. ...` becomes `### N. ...`
- source-tree links into `isa/micro-isa/...` are rewritten to in-document `#micro-...` anchors
- source-tree links into `isa/tile-op/...` are rewritten to `PTO-tile-Instruction-SPEC.md#tile-...`

Invariants for the Tile Instruction bundle (`docs/PTO-tile-Instruction-SPEC.md`):
- exactly one `#` level title at the top
- `[toc]` is present near the top
- every chapter file from `docs/isa/tile-op/` is inlined in sorted order
- intra-bundle links resolve to `<a id="tile-XX-name"></a>` anchors
- cross-bundle links are rewritten to `PTO-micro-Instruction-SPEC.md#micro-XX-name`

Invariants for the legacy versioned snapshot:
- it carries the same bundled micro-spec content as the main repo doc
- it is emitted under `docs/release/` by default unless `--output-dir` overrides the destination
- Tile Instruction links are rewritten relative to the snapshot location

4. If the user wants extra release-note wording, patch only the version bullets or other small wording around the generated content. Prefer rerunning the script over hand-merging large sections.

## Source Mapping

| Source | Target |
|--------|--------|
| `docs/vpto-spec.md` + `docs/isa/micro-isa/*.md` | `docs/PTO-micro-Instruction-SPEC.md` |
| `docs/isa/tile-op/*.md` | `docs/PTO-tile-Instruction-SPEC.md` |
| `docs/vpto-spec.md` + `docs/isa/micro-isa/*.md` | `docs/release/vpto-spec-v<version>.md` (legacy versioned snapshot) |

## Merge Rules

For the main bundled micro-spec the script:
- emits a single top-level title
- prepends a target-specific version-bullet list
- inserts a `[toc]` marker
- starts from `docs/vpto-spec.md`
- strips draft metadata, appendix content, and now-misleading "see individual files" prose
- preserves the high-level overview and summary sections
- rewrites `isa/micro-isa/*.md` links to in-document anchors
- rewrites `isa/tile-op/*.md` links to the companion bundle filename + anchor
- appends a `## Detailed ISA Group Reference` section that inlines all `docs/isa/micro-isa/*.md` chapters in sorted order

For the standalone Tile Instruction bundle the script:
- emits a single top-level title
- prepends a target-specific version-bullet list
- inserts a `[toc]` marker
- inlines all `docs/isa/tile-op/*.md` files in sorted order
- demotes chapter headings by one level
- emits stable HTML anchors like `<a id="tile-XX-name"></a>`
- rewrites intra-bundle relative links to those anchors
- rewrites cross-bundle relative links to the bundled micro-spec filename + anchor

## Notes

- Repo-doc bundle filenames intentionally drop the version suffix; the version is recorded only in the per-file version-history bullets.
- Default version notes for known versions live inside the script; pass `--version-note` to add or override the note for the requested target.
- When chapter filenames or numbering change in `docs/isa/micro-isa/` or `docs/isa/tile-op/`, regenerate both repo-doc bundles and any legacy snapshots so links stay synchronized.
- If downstream consumers still require older filenames such as `pto-micro-instruction.md`, use `--micro-output-name` / `--tileop-output-name` instead of hand-renaming the generated files.
