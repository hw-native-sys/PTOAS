#!/usr/bin/env python3
"""Generate merged VPTO release spec from docs/vpto-spec.md and docs/isa/*.md."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DOCS_DIR = ROOT / "docs"
SOURCE_SPEC = DOCS_DIR / "vpto-spec.md"
ISA_DIR = DOCS_DIR / "isa"
RELEASE_DIR = DOCS_DIR / "release"

TITLE = "# PTO micro Instruction Spec \u2014 Draft (A5)"
DEFAULT_VERSION_NOTES = {
    "0.1": "Doc Init",
    "0.2": "Update micro Instruction latency and throughput",
    "0.3": "Refresh VPTO ISA specification",
}

KEEP_SECTIONS = [
    "## Part I: Architecture Overview",
    "## Part II: Notation Convention",
    "## Instruction Groups",
    "## Supported Data Types",
    "## Common Patterns",
    "## Quick Reference by Category",
]

ISA_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:\.\./)?(?:isa/)?([0-9]{2}-[A-Za-z0-9-]+)\.md\)")


def extract_sections(markdown: str) -> dict[str, str]:
    headings = list(re.finditer(r"^## .*$", markdown, flags=re.MULTILINE))
    sections: dict[str, str] = {}
    for index, match in enumerate(headings):
        heading = match.group(0).strip()
        start = match.start()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        sections[heading] = markdown[start:end].strip() + "\n"
    return sections


def rewrite_isa_links(text: str) -> str:
    return ISA_LINK_RE.sub(lambda m: f"[{m.group(1)}](#isa-{m.group(2).lower()})", text)


def trim_trailing_rule(text: str) -> str:
    return re.sub(r"\n---\s*\Z", "\n", text.strip() + "\n").rstrip()


def strip_unwanted_lines(text: str) -> str:
    lines = text.splitlines()
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
        if line.startswith("> **Status:**") or line.startswith("> **Base:**") or line.startswith("> **Additions from:**") or line.startswith("> **Updated:**"):
            continue
        if "For detailed semantics, C-style pseudocode, and CCE mappings" in line:
            continue
        if "CCE correspondence" in line or "builtin mapping" in line.lower():
            continue
        kept.append(line)
    text = "\n".join(kept).strip() + "\n"
    text = re.sub(r"\n## Appendix [A-Z]:.*\Z", "\n", text, flags=re.DOTALL)
    return text


def demote_headings(text: str, levels: int = 2) -> str:
    def replace(match: re.Match[str]) -> str:
        hashes = match.group(1)
        heading = match.group(2)
        new_level = min(6, len(hashes) + levels)
        return f"{'#' * new_level} {heading}"

    return re.sub(r"^(#{1,6})\s+(.*)$", replace, text, flags=re.MULTILINE)


def render_version_bullets(version: str, version_note: str | None) -> str:
    notes = dict(DEFAULT_VERSION_NOTES)
    if version_note:
        notes[version] = version_note
    elif version not in notes:
        notes[version] = "Release refresh"

    def key_fn(item: str) -> tuple[int, ...]:
        return tuple(int(part) for part in item.split("."))

    lines = [f"- v{ver}: {notes[ver]}" for ver in sorted(notes, key=key_fn, reverse=True)]
    return "\n".join(lines)


def build_release_doc(version: str, version_note: str | None) -> str:
    source_text = strip_unwanted_lines(SOURCE_SPEC.read_text())
    sections = extract_sections(source_text)

    missing = [name for name in KEEP_SECTIONS if name not in sections]
    if missing:
        raise SystemExit(f"missing expected headings in docs/vpto-spec.md: {missing}")

    content_sections = [trim_trailing_rule(rewrite_isa_links(sections[name])) for name in KEEP_SECTIONS[:-1]]

    isa_blocks: list[str] = ["## Detailed ISA Group Reference"]
    for isa_path in sorted(ISA_DIR.glob("*.md")):
        isa_text = rewrite_isa_links(isa_path.read_text().strip() + "\n")
        isa_blocks.append(trim_trailing_rule(demote_headings(isa_text)))

    quick_reference = trim_trailing_rule(rewrite_isa_links(sections["## Quick Reference by Category"]))

    parts = [
        TITLE,
        "",
        render_version_bullets(version, version_note),
        "",
        "[toc]",
        "",
        "---",
        "",
        "\n\n".join(content_sections),
        "\n\n".join(isa_blocks),
        quick_reference,
        "",
    ]
    return "\n".join(part for part in parts if part is not None)


def validate_release_doc(text: str) -> None:
    if text.count("# PTO micro Instruction Spec") != 1:
        raise SystemExit("expected exactly one top-level title")
    if "\n[toc]\n" not in text:
        raise SystemExit("missing [toc] near top")
    if re.search(r"^## Quick Reference by Category\b", text, flags=re.MULTILINE) is None:
        raise SystemExit("missing Quick Reference by Category")
    if re.search(r"^## Quick Reference by Category\b[\s\S]*\Z", text, flags=re.MULTILINE) is None:
        raise SystemExit("Quick Reference by Category must be present at end")
    if re.search(r"^## Appendix\b", text, flags=re.MULTILINE):
        raise SystemExit("appendix content must not remain")
    if "Updated:" in text or "review" in text.splitlines()[:8]:
        raise SystemExit("beginning metadata must not remain")
    if "## Correspondence Categories" in text or "CCE correspondence" in text:
        raise SystemExit("correspondence content must not remain")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="Release version, e.g. 0.2")
    parser.add_argument("--version-note", help="Version bullet text for the requested version")
    parser.add_argument("--output", help="Explicit output path")
    args = parser.parse_args()

    output = Path(args.output) if args.output else RELEASE_DIR / f"vpto-spec-v{args.version}.md"
    output.parent.mkdir(parents=True, exist_ok=True)

    text = build_release_doc(args.version, args.version_note)
    validate_release_doc(text)
    output.write_text(text)


if __name__ == "__main__":
    main()
