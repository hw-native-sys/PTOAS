---
name: generate-vpto-release-doc
description: Generate or refresh `docs/release/vpto-spec-v*.md` by merging `docs/vpto-spec.md` with `docs/isa/*.md`, following the release-doc naming and layout. Use when the user asks to create or update a merged VPTO release spec, inline ISA Markdown into one release document, add TOC and version bullets, move `Quick Reference by Category` to the end, or strip update, appendix, and correspondence content from the merged release doc.
---

# Generate VPTO Release Doc

Use this skill when the task is specifically about:
- creating a new merged release document under `docs/release/`
- refreshing an existing `vpto-spec-v*.md` release doc from `docs/vpto-spec.md` and `docs/isa/*.md`
- keeping the merged release doc aligned with the naming and structure used in `docs/release/vpto-spec-v0.1.md`

## Canonical Workflow

1. Pick the target version and output path.

Default output path:

```bash
docs/release/vpto-spec-v<version>.md
```

2. Run the bundled generator script.

```bash
python3 .codex/skills/generate-vpto-release-doc/scripts/generate_release_vpto_spec.py --version 0.2
```

If you need an explicit note for the new version bullet:

```bash
python3 .codex/skills/generate-vpto-release-doc/scripts/generate_release_vpto_spec.py \
  --version 0.2 \
  --version-note 'Merge `docs/vpto-spec.md` with `docs/isa/*.md`; add TOC; move `Quick Reference by Category` to the end; remove update, appendix, and correspondence content'
```

3. Review the generated file before finalizing.

Check these invariants:
- exactly one `#` level title in the whole file
- `[toc]` is present near the top
- the top version bullet for the requested version was added
- `## Quick Reference by Category` is the final top-level section
- no `Updated:` / review-status boilerplate remains at the beginning
- no appendix sections remain
- no `## Correspondence Categories` section remains
- no `CCE correspondence` / builtin-mapping blocks remain

4. If the user wants extra release-note wording, patch only the version bullets or other small wording around the generated content. Prefer rerunning the script over hand-merging large sections.

## Source Mapping

Use `docs/vpto-spec.md` for:
- `Part I: Architecture Overview`
- `Part II: Notation Convention`
- `C-Style Semantics Convention`
- `Template Placeholder Conventions`
- `Instruction Groups`
- `Supported Data Types`
- `Common Patterns`
- `Quick Reference by Category`

Use `docs/isa/*.md` for:
- the inlined `Detailed ISA Group Reference`

## Merge Rules

The merged release document should:
- keep the release-doc title and version-bullet style
- preserve the `Instruction Groups` summary table
- inline `docs/isa/*.md` under `Detailed ISA Group Reference`
- convert `docs/isa/*.md` links into in-document anchors like `#isa-03-vector-load-store`
- demote the inlined ISA headings by two levels so the merged TOC stays stable
- place `Quick Reference by Category` at the end

The merged release document must remove:
- beginning-of-file update/review metadata from `docs/vpto-spec.md`
- `## Correspondence Categories`
- all `CCE correspondence` blocks and related builtin/token mapping lines
- the sentence `For detailed semantics, C-style pseudocode, and CCE mappings, see the individual group documentation files.`
- appendix sections

## Notes

- The script assumes the source headings in `docs/vpto-spec.md` keep their current names. If extraction fails, inspect the heading names there before patching the script.
- The script is deterministic and is the preferred path for regenerating large merged release docs.
