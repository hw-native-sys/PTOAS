#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate bundled PTO ISA reference docs.

Default repo docs:
- ``docs/vpto-spec.md`` + ``docs/isa/micro-isa/*.md``
  -> ``docs/PTO-micro-Instruction-SPEC.md``
- ``docs/isa/tile-op/*.md`` -> ``docs/PTO-tile-Instruction-SPEC.md``

Legacy versioned snapshot:
- ``docs/vpto-spec.md`` + ``docs/isa/micro-isa/*.md``
  -> ``docs/release/vpto-spec-v<version>.md``
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DOCS_DIR = ROOT / "docs"
ISA_DIR = DOCS_DIR / "isa"
RELEASE_DIR = DOCS_DIR / "release"
SOURCE_SPEC = DOCS_DIR / "vpto-spec.md"

MICRO_TARGET = "micro"
TILEOP_TARGET = "tileop"
MERGED_TARGET = "merged"

SPLIT_BUNDLES: dict[str, dict[str, str | Path]] = {
    MICRO_TARGET: {
        "source_dir": ISA_DIR / "micro-isa",
        "output_name": "PTO-micro-Instruction-SPEC.md",
        "title": "# PTO micro Instruction Spec — Draft (A5)",
        "anchor_prefix": "micro",
        "source_dir_name": "micro-isa",
        "peer_target": TILEOP_TARGET,
        "peer_dir": "tile-op",
    },
    TILEOP_TARGET: {
        "source_dir": ISA_DIR / "tile-op",
        "output_name": "PTO-tile-Instruction-SPEC.md",
        "title": "# PTO Tile Instruction SPEC (A5)",
        "anchor_prefix": "tile",
        "source_dir_name": "tile-op",
        "peer_target": MICRO_TARGET,
        "peer_dir": "micro-isa",
    },
}

MICRO_DOC_TITLE = "# PTO micro Instruction Spec — Draft (A5)"
MERGED_OUTPUT_NAME = "vpto-spec-v{version}.md"
PART_IV_HEADING = "## Part IV: PTO Tile Instruction"

MICRO_REQUIRED_SECTIONS = [
    "## Part I: Architecture Overview",
    "## Part II: Notation Convention",
    "## Part III: ISA Instruction Reference",
    "## Instruction Groups",
    "## Supported Data Types",
    "## Common Patterns",
    "## Quick Reference by Category",
]

VERSION_NOTES = {
    MICRO_TARGET: {
        "0.1": "Doc Init",
        "0.2": "Update micro Instruction latency and throughput",
        "0.3": (
            "Add runtime block query and vector-interval legality notes; "
            "Normalize load/store distribution families; Update get_buf/rls_buf details"
        ),
        "0.4": "Update DMA instruction docs and add PTO Tile Instruction SPEC",
        "0.5": "Add CUBE instruction docs; Rename MTE instruction and address space",
    },
    TILEOP_TARGET: {
        "0.4": "Initial PTO Tile Instruction SPEC covering core TileOps",
    },
    MERGED_TARGET: {
        "0.1": "Doc Init",
        "0.2": "Update micro Instruction latency and throughput",
        "0.3": (
            "Add runtime block query and vector-interval legality notes; "
            "Normalize load/store distribution families; Update get_buf/rls_buf details"
        ),
        "0.4": "Update DMA instruction docs and add PTO Tile Instruction SPEC",
        "0.5": "Add CUBE instruction docs; Rename MTE instruction and address space",
    },
}

# In-directory chapter link inside one ISA folder, for example
# ``[Foo](02-types-and-attributes.md)`` or ``[Foo](02-types-and-attributes.md#anchor)``.
INTRA_LINK_RE = re.compile(
    r"\]\((?P<chapter>[0-9]{2}-[A-Za-z0-9-]+)\.md(?P<anchor>#[A-Za-z0-9_-]+)?\)"
)
# Cross-bundle link from one ISA folder to its sibling, for example
# ``[Foo](../micro-isa/01-pipeline-sync.md)``.
PEER_LINK_RE_TEMPLATE = (
    r"\]\(\.\./{peer_dir}/(?P<chapter>[0-9]{{2}}-[A-Za-z0-9-]+)\.md"
    r"(?P<anchor>#[A-Za-z0-9_-]+)?\)"
)
# Link from the top-level vpto-spec.md to a chapter inside an ISA folder, for example
# ``[Foo](isa/tile-op/01-tile-overview.md)``.
SPEC_LINK_RE_TEMPLATE = (
    r"\]\((?:\.\./)?(?:docs/)?isa/{source_dir_name}/(?P<chapter>[0-9]{{2}}-[A-Za-z0-9-]+)\.md"
    r"(?P<anchor>#[A-Za-z0-9_-]+)?\)"
)


def render_version_bullets(target: str, version: str, version_note: str | None) -> str:
    notes = dict(VERSION_NOTES[target])
    if version_note:
        notes[version] = version_note
    elif version not in notes:
        notes[version] = "Release refresh"

    def key_fn(item: str) -> tuple[int, ...]:
        return tuple(int(part) for part in item.split("."))

    return "\n".join(
        f"- v{ver}: {notes[ver]}" for ver in sorted(notes, key=key_fn, reverse=True)
    )


def extract_sections(markdown: str) -> dict[str, str]:
    headings = list(re.finditer(r"^## .*$", markdown, flags=re.MULTILINE))
    sections: dict[str, str] = {}
    for index, match in enumerate(headings):
        heading = match.group(0).strip()
        start = match.start()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        sections[heading] = markdown[start:end].strip() + "\n"
    return sections


def trim_trailing_rule(text: str) -> str:
    return re.sub(r"\n---\s*\Z", "\n", text.strip() + "\n").rstrip()


def demote_headings(text: str, levels: int = 1) -> str:
    """Increase ATX heading level by ``levels``, capped at H6."""

    def replace(match: re.Match[str]) -> str:
        hashes = match.group(1)
        heading = match.group(2)
        new_level = min(6, len(hashes) + levels)
        return f"{'#' * new_level} {heading}"

    return re.sub(r"^(#{1,6})\s+(.*)$", replace, text, flags=re.MULTILINE)


def strip_spec_unwanted_lines(markdown: str) -> str:
    lines = markdown.splitlines()
    kept: list[str] = []
    skip_correspondence = False

    for line in lines:
        if re.match(r"^## Correspondence Categories\b", line):
            skip_correspondence = True
            continue
        if skip_correspondence:
            if re.match(r"^## ", line):
                skip_correspondence = False
            else:
                continue
        if line.startswith("> **Status:**"):
            continue
        if line.startswith("> **Base:**"):
            continue
        if line.startswith("> **Additions from:**"):
            continue
        if line.startswith("> **Updated:**"):
            continue
        if "For detailed semantics, C-style pseudocode, and CCE mappings" in line:
            continue
        if "CCE correspondence" in line or "builtin mapping" in line.lower():
            continue
        kept.append(line)

    text = "\n".join(kept).strip() + "\n"
    text = re.sub(r"\n## Appendix\b.*\Z", "\n", text, flags=re.DOTALL)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def normalize_part_three_heading(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].startswith("## Part III: ISA Instruction Reference"):
        if lines[1].startswith("# Part III: ISA Instruction Reference"):
            lines = ["## Part III: ISA Instruction Reference — Summary"] + lines[2:]
    return "\n".join(lines).strip() + "\n"


def resolve_bundle_ref(
    from_output_target: str,
    to_target: str,
    split_output_names: dict[str, str],
    output_dir_overridden: bool,
) -> str:
    bundle_name = split_output_names[to_target]
    if from_output_target == MERGED_TARGET and not output_dir_overridden:
        return f"../{bundle_name}"
    return bundle_name


def rewrite_intra_links(text: str, anchor_prefix: str) -> str:
    """Rewrite same-bundle chapter links to in-document anchors."""

    def repl(match: re.Match[str]) -> str:
        chapter = match.group("chapter").lower()
        anchor_suffix = match.group("anchor") or ""
        if anchor_suffix:
            return f"]({anchor_suffix})"
        return f"](#{anchor_prefix}-{chapter})"

    return INTRA_LINK_RE.sub(repl, text)


def rewrite_cross_bundle_links(
    text: str,
    source_target: str,
    from_output_target: str,
    split_output_names: dict[str, str],
    output_dir_overridden: bool,
) -> str:
    """Rewrite cross-bundle chapter links to the companion bundle."""

    cfg = SPLIT_BUNDLES[source_target]
    peer_target: str = cfg["peer_target"]  # type: ignore[assignment]
    peer_dir: str = cfg["peer_dir"]  # type: ignore[assignment]
    peer_cfg = SPLIT_BUNDLES[peer_target]
    peer_anchor_prefix: str = peer_cfg["anchor_prefix"]  # type: ignore[assignment]
    peer_bundle_ref = resolve_bundle_ref(
        from_output_target,
        peer_target,
        split_output_names,
        output_dir_overridden,
    )
    peer_link_re = re.compile(PEER_LINK_RE_TEMPLATE.format(peer_dir=re.escape(peer_dir)))

    def repl(match: re.Match[str]) -> str:
        chapter = match.group("chapter").lower()
        anchor_suffix = match.group("anchor") or ""
        if anchor_suffix:
            return f"]({peer_bundle_ref}{anchor_suffix})"
        return f"]({peer_bundle_ref}#{peer_anchor_prefix}-{chapter})"

    return peer_link_re.sub(repl, text)


def rewrite_spec_chapter_links(
    text: str,
    same_doc_targets: set[str],
    from_output_target: str,
    split_output_names: dict[str, str],
    output_dir_overridden: bool,
) -> str:
    """Rewrite ``docs/vpto-spec.md`` chapter links to bundled outputs."""

    for target, cfg in SPLIT_BUNDLES.items():
        source_dir_name: str = cfg["source_dir_name"]  # type: ignore[assignment]
        anchor_prefix: str = cfg["anchor_prefix"]  # type: ignore[assignment]
        link_re = re.compile(
            SPEC_LINK_RE_TEMPLATE.format(source_dir_name=re.escape(source_dir_name))
        )

        if target in same_doc_targets:

            def repl(match: re.Match[str]) -> str:
                chapter = match.group("chapter").lower()
                anchor_suffix = match.group("anchor") or ""
                if anchor_suffix:
                    return f"]({anchor_suffix})"
                return f"](#{anchor_prefix}-{chapter})"

        else:
            bundle_ref = resolve_bundle_ref(
                from_output_target,
                target,
                split_output_names,
                output_dir_overridden,
            )

            def repl(match: re.Match[str]) -> str:
                chapter = match.group("chapter").lower()
                anchor_suffix = match.group("anchor") or ""
                if anchor_suffix:
                    return f"]({bundle_ref}{anchor_suffix})"
                return f"]({bundle_ref}#{anchor_prefix}-{chapter})"

        text = link_re.sub(repl, text)
    return text


def build_chapter_blocks(
    source_target: str,
    heading_levels: int,
    from_output_target: str,
    split_output_names: dict[str, str],
    output_dir_overridden: bool,
) -> tuple[list[Path], list[str]]:
    cfg = SPLIT_BUNDLES[source_target]
    source_dir: Path = cfg["source_dir"]  # type: ignore[assignment]
    if not source_dir.is_dir():
        raise SystemExit(f"source directory not found: {source_dir}")

    chapter_files = sorted(source_dir.glob("*.md"))
    if not chapter_files:
        raise SystemExit(f"no .md files found in {source_dir}")

    anchor_prefix: str = cfg["anchor_prefix"]  # type: ignore[assignment]
    blocks: list[str] = []

    for path in chapter_files:
        chapter_id = f"{anchor_prefix}-{path.stem.lower()}"
        text = path.read_text().strip() + "\n"
        text = rewrite_intra_links(text, anchor_prefix)
        text = rewrite_cross_bundle_links(
            text,
            source_target,
            from_output_target,
            split_output_names,
            output_dir_overridden,
        )
        text = demote_headings(text, levels=heading_levels)
        text = f'<a id="{chapter_id}"></a>\n\n{text}'
        blocks.append(trim_trailing_rule(text))

    return chapter_files, blocks


def build_micro_bundle(
    version_target: str,
    version: str,
    version_note: str | None,
    from_output_target: str,
    split_output_names: dict[str, str],
    output_dir_overridden: bool,
) -> str:
    source_text = strip_spec_unwanted_lines(SOURCE_SPEC.read_text())
    sections = extract_sections(source_text)

    missing = [name for name in MICRO_REQUIRED_SECTIONS if name not in sections]
    if missing:
        raise SystemExit(f"missing expected headings in {SOURCE_SPEC}: {missing}")

    rendered_sections: list[str] = []
    for name in MICRO_REQUIRED_SECTIONS[:4]:
        text = sections[name]
        if name == "## Part III: ISA Instruction Reference":
            text = normalize_part_three_heading(text)
        text = rewrite_spec_chapter_links(
            text,
            same_doc_targets={MICRO_TARGET},
            from_output_target=from_output_target,
            split_output_names=split_output_names,
            output_dir_overridden=output_dir_overridden,
        )
        rendered_sections.append(trim_trailing_rule(text))

    chapter_files, chapter_blocks = build_chapter_blocks(
        MICRO_TARGET,
        heading_levels=2,
        from_output_target=from_output_target,
        split_output_names=split_output_names,
        output_dir_overridden=output_dir_overridden,
    )
    detailed_section = "\n".join(
        [
            "## Detailed ISA Group Reference",
            "",
            (
                f"This section inlines the {len(chapter_files)} ISA group documents so the "
                "architectural overview, notation, summary table, and per-group semantics can "
                "be read in a single file."
            ),
            "",
            "\n\n".join(chapter_blocks),
        ]
    ).strip()

    for name in MICRO_REQUIRED_SECTIONS[4:]:
        text = rewrite_spec_chapter_links(
            sections[name],
            same_doc_targets={MICRO_TARGET},
            from_output_target=from_output_target,
            split_output_names=split_output_names,
            output_dir_overridden=output_dir_overridden,
        )
        rendered_sections.append(trim_trailing_rule(text))

    if PART_IV_HEADING in sections:
        part_four = rewrite_spec_chapter_links(
            sections[PART_IV_HEADING],
            same_doc_targets={MICRO_TARGET},
            from_output_target=from_output_target,
            split_output_names=split_output_names,
            output_dir_overridden=output_dir_overridden,
        )
        rendered_sections.append(trim_trailing_rule(part_four))

    body = rendered_sections[:4] + [detailed_section] + rendered_sections[4:]
    parts = [
        MICRO_DOC_TITLE,
        "",
        render_version_bullets(version_target, version, version_note),
        "",
        "[toc]",
        "",
        "---",
        "",
        "\n\n".join(body),
        "",
    ]
    return "\n".join(parts)


def build_tileop_bundle(
    version: str,
    version_note: str | None,
    split_output_names: dict[str, str],
    output_dir_overridden: bool,
) -> str:
    cfg = SPLIT_BUNDLES[TILEOP_TARGET]
    _, blocks = build_chapter_blocks(
        TILEOP_TARGET,
        heading_levels=1,
        from_output_target=TILEOP_TARGET,
        split_output_names=split_output_names,
        output_dir_overridden=output_dir_overridden,
    )

    parts = [
        cfg["title"],
        "",
        render_version_bullets(TILEOP_TARGET, version, version_note),
        "",
        "[toc]",
        "",
        "---",
        "",
        "\n\n".join(blocks),
        "",
    ]
    return "\n".join(parts)


def validate_micro_bundle(text: str) -> None:
    if text.count(MICRO_DOC_TITLE) != 1:
        raise SystemExit(f"expected exactly one top-level title: {MICRO_DOC_TITLE!r}")
    if len(re.findall(r"^# ", text, flags=re.MULTILINE)) != 1:
        raise SystemExit("expected exactly one top-level heading in micro bundle")
    if "\n[toc]\n" not in text:
        raise SystemExit("missing [toc] near top")
    if "## Detailed ISA Group Reference" not in text:
        raise SystemExit("missing Detailed ISA Group Reference section")
    if re.search(r"^> \*\*(Status|Base|Updated):", text, flags=re.MULTILINE):
        raise SystemExit("beginning metadata must not remain in micro bundle")
    if re.search(r"^## Appendix\b", text, flags=re.MULTILINE):
        raise SystemExit("appendix content must not remain in micro bundle")
    if "../micro-isa/" in text or "../tile-op/" in text:
        raise SystemExit("stale relative ISA directory links remain in micro bundle")
    if "isa/micro-isa/" in text or "isa/tile-op/" in text or "docs/isa/" in text:
        raise SystemExit("stale source-tree ISA links remain in micro bundle")


def validate_tileop_bundle(text: str) -> None:
    title: str = SPLIT_BUNDLES[TILEOP_TARGET]["title"]  # type: ignore[assignment]
    if text.count(title) != 1:
        raise SystemExit(f"expected exactly one top-level title: {title!r}")
    if len(re.findall(r"^# ", text, flags=re.MULTILINE)) != 1:
        raise SystemExit("expected exactly one top-level heading in tileop bundle")
    if "\n[toc]\n" not in text:
        raise SystemExit("missing [toc] near top")
    if "../micro-isa/" in text or "../tile-op/" in text:
        raise SystemExit("stale relative ISA directory links remain in tileop bundle")


def resolve_split_output_names(args: argparse.Namespace) -> dict[str, str]:
    return {
        MICRO_TARGET: args.micro_output_name
        or SPLIT_BUNDLES[MICRO_TARGET]["output_name"],  # type: ignore[index]
        TILEOP_TARGET: args.tileop_output_name
        or SPLIT_BUNDLES[TILEOP_TARGET]["output_name"],  # type: ignore[index]
    }


def resolve_output_name(
    target: str,
    version: str,
    split_output_names: dict[str, str],
    merged_output_name: str | None,
) -> str:
    if target == MERGED_TARGET:
        if merged_output_name:
            return merged_output_name
        return MERGED_OUTPUT_NAME.format(version=version)
    return split_output_names[target]


def resolve_output_dir(target: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    if target == MERGED_TARGET:
        return RELEASE_DIR
    return DOCS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Release version, for example 0.4")
    parser.add_argument(
        "--version-note",
        help="Version bullet text for the requested target and version",
    )
    parser.add_argument(
        "--target",
        choices=sorted(SPLIT_BUNDLES.keys()) + [MERGED_TARGET, "all"],
        default="all",
        help="Which target to generate. 'all' keeps the repo-doc default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Override the output directory. By default, repo-doc bundles go to docs/ "
            "and the legacy versioned snapshot goes to docs/release/."
        ),
    )
    parser.add_argument(
        "--micro-output-name",
        help="Override the main bundled micro-spec filename.",
    )
    parser.add_argument(
        "--tileop-output-name",
        help="Override the PTO Tile Instruction bundle filename.",
    )
    parser.add_argument(
        "--merged-output-name",
        help="Override the legacy merged-doc output filename, for example vpto-spec.md",
    )
    args = parser.parse_args()

    if args.target == "all":
        targets = [MICRO_TARGET, TILEOP_TARGET]
    else:
        targets = [args.target]

    output_dir_overridden = args.output_dir is not None
    split_output_names = resolve_split_output_names(args)

    for target in targets:
        if target == MICRO_TARGET:
            text = build_micro_bundle(
                version_target=MICRO_TARGET,
                version=args.version,
                version_note=args.version_note,
                from_output_target=MICRO_TARGET,
                split_output_names=split_output_names,
                output_dir_overridden=output_dir_overridden,
            )
            validate_micro_bundle(text)
        elif target == TILEOP_TARGET:
            text = build_tileop_bundle(
                version=args.version,
                version_note=args.version_note,
                split_output_names=split_output_names,
                output_dir_overridden=output_dir_overridden,
            )
            validate_tileop_bundle(text)
        else:
            text = build_micro_bundle(
                version_target=MERGED_TARGET,
                version=args.version,
                version_note=args.version_note,
                from_output_target=MERGED_TARGET,
                split_output_names=split_output_names,
                output_dir_overridden=output_dir_overridden,
            )
            validate_micro_bundle(text)

        output_dir = resolve_output_dir(target, args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_name = resolve_output_name(
            target,
            args.version,
            split_output_names,
            args.merged_output_name,
        )
        output = output_dir / output_name
        output.write_text(text)
        try:
            shown = output.relative_to(ROOT)
        except ValueError:
            shown = output
        print(f"wrote {shown}")


if __name__ == "__main__":
    main()
