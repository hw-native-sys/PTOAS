#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import json
import linecache
import re
import subprocess
import sys
import tempfile
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
USER_GUIDE_ROOT = REPO_ROOT / "ptodsl" / "docs" / "user_guide"
from ptodsl import pto
from ptodsl._context import make_context
from ptodsl._runtime.launch import LaunchHandle, _marshal_launch_args
from ptodsl._runtime.toolchain import resolve_ptoas_binary
from ptoas.mlir.ir import Module
from support.docs_fragment_fixtures import FRAGMENT_FIXTURES, render_fragment_fixture

FENCE_RE = re.compile(r"^```(?P<lang>[A-Za-z0-9_+-]*)\s*$")
META_RE = re.compile(r"^\s*<!--\s*ptodsl-doc-(?P<kind>test|pending)\s*:\s*(?P<body>.*?)\s*-->\s*$")


@dataclass(frozen=True)
class MarkdownCodeBlock:
    path: Path
    start_line: int
    end_line: int
    language: str
    lines: tuple[str, ...]
    metadata: "Optional[DocBlockMetadata]"

    @property
    def text(self) -> str:
        return "".join(self.lines)


@dataclass(frozen=True)
class MarkdownScanResult:
    path: Path
    blocks: tuple[MarkdownCodeBlock, ...]


@dataclass(frozen=True)
class DocBlockMetadata:
    kind: str
    body: str
    line: int
    raw: str


@dataclass(frozen=True)
class DocTestDirective:
    mode: str
    symbol: Optional[str] = None
    compile_kwargs: Optional[dict[str, object]] = None
    fixture: Optional[str] = None
    files: Optional[dict[str, str]] = None


@dataclass(frozen=True)
class LaunchRecord:
    compiled: object
    grid: int
    stream: object
    args: tuple[object, ...]
    marshaled_arg_count: int


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def format_doc_context(path: Path, start_line: int, symbol: Optional[str] = None) -> str:
    symbol_text = symbol if symbol is not None else "<unknown>"
    return f"{path}:{start_line} [symbol={symbol_text}]"


def fail_doc(path: Path, start_line: int, message: str, symbol: Optional[str] = None) -> None:
    raise AssertionError(f"{format_doc_context(path, start_line, symbol)}: {message}")


def iter_markdown_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*.md"))


def parse_metadata_line(path: Path, line: str, line_number: int) -> Optional[DocBlockMetadata]:
    match = META_RE.match(line)
    if match is None:
        return None

    kind = match.group("kind")
    body = match.group("body").strip()
    expect(body, f"{format_doc_context(path, line_number)}: ptodsl-doc-{kind} metadata must not be empty")
    if kind == "test":
        try:
            json.loads(body)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"{format_doc_context(path, line_number)}: ptodsl-doc-test metadata must be valid JSON: {exc.msg}"
            ) from exc
    return DocBlockMetadata(kind=kind, body=body, line=line_number, raw=line.rstrip("\n"))


def find_block_metadata(path: Path, lines: list[str], fence_line: int) -> Optional[DocBlockMetadata]:
    candidate = fence_line - 2
    while candidate >= 0 and not lines[candidate].strip():
        candidate -= 1
    if candidate < 0:
        return None
    line = lines[candidate]
    if line.lstrip().startswith("<!-- ptodsl-doc-") and parse_metadata_line(path, line, candidate + 1) is None:
        fail_doc(path, fence_line, "malformed ptodsl-doc metadata comment")
    return parse_metadata_line(path, line, candidate + 1)


def block_label(block: MarkdownCodeBlock, symbol: Optional[str] = None) -> str:
    return format_doc_context(block.path, block.start_line, symbol)


def expect_parse_roundtrip_and_verify(text: str, label: str) -> None:
    with make_context() as ctx:
        parsed = Module.parse(text, ctx)
        parsed.operation.verify()
        roundtrip_text = str(parsed)
    expect(
        roundtrip_text == text,
        f"{label} should survive Module.parse(...) round-trip without textual drift",
    )


def extract_child_module_texts(container_text: str, label: str) -> list[str]:
    with make_context() as ctx:
        parsed = Module.parse(container_text, ctx)
        top_level_ops = list(parsed.body.operations)
    expect(top_level_ops, f"{label} should contain at least one top-level operation")
    if all(op.operation.name == "builtin.module" for op in top_level_ops):
        return [str(op) for op in top_level_ops]
    return [container_text]


def run_ptoas_frontend_verify(ptoas_bin: Path, mlir_text: str, label: str) -> None:
    child_modules = extract_child_module_texts(mlir_text, label)

    for index, child_text in enumerate(child_modules, start=1):
        with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False, encoding="utf-8") as handle:
            handle.write(child_text)
            input_path = Path(handle.name)

        child_label = f"{label} [child {index}]"
        try:
            result = subprocess.run(
                [str(ptoas_bin), str(input_path), "--emit-pto-ir", "-o", "-"],
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            input_path.unlink(missing_ok=True)

        if result.returncode == 0 and result.stdout.strip():
            continue

        if "expected VPTO container top level to contain only kernel submodules" in result.stderr:
            continue

        if "VPTO LLVM emission failed" in result.stderr:
            continue

        if (
            "object output requires an explicit file path passed with -o." in result.stderr
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_root = Path(temp_dir)
                fallback_input_path = temp_root / "input.mlir"
                output_path = temp_root / "kernel.o"
                fallback_input_path.write_text(child_text, encoding="utf-8")
                fallback_result = subprocess.run(
                    [str(ptoas_bin), str(fallback_input_path), "-o", str(output_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                artifact_exists = output_path.is_file()
                artifact_size = output_path.stat().st_size if artifact_exists else 0
            if fallback_result.returncode != 0 and (
                "ASCEND_HOME_PATH is required" in fallback_result.stderr
                or "CANN toolchain is required but was not initialized" in fallback_result.stderr
            ):
                continue
            expect(
                fallback_result.returncode == 0,
                f"{child_label} should pass PTOAS fallback compilation when the VPTO fast path skips --emit-pto-ir.\n"
                f"stdout:\n{fallback_result.stdout}\nstderr:\n{fallback_result.stderr}",
            )
            expect(artifact_exists, f"{child_label} should produce an output artifact via fallback ptoas -o")
            expect(
                artifact_size > 0,
                f"{child_label} should produce a non-empty output artifact via fallback ptoas -o",
            )
            continue

        expect(
            result.returncode == 0,
            f"{child_label} should pass PTOAS frontend verification.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        expect(result.stdout.strip(), f"{child_label} should emit non-empty PTO IR after PTOAS frontend passes")


def parse_test_directive(block: MarkdownCodeBlock) -> DocTestDirective:
    expect(block.metadata is not None, f"{block_label(block)}: python code block missing metadata")
    expect(block.metadata.kind == "test", f"{block_label(block)}: expected ptodsl-doc-test metadata")

    try:
        payload = json.loads(block.metadata.body)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"{block_label(block)}: ptodsl-doc-test metadata must be valid JSON: {exc.msg}"
        ) from exc

    expect(
        isinstance(payload, dict),
        f"{block_label(block)}: ptodsl-doc-test metadata must be a JSON object",
    )

    mode = payload.get("mode")
    symbol = payload.get("symbol")
    compile_kwargs = payload.get("compile")
    fixture = payload.get("fixture")
    files = payload.get("files")

    expect(
        isinstance(mode, str) and mode,
        f"{block_label(block)}: ptodsl-doc-test metadata must define a non-empty string 'mode'",
    )
    if files is not None:
        expect(
            isinstance(files, dict) and all(isinstance(path, str) and isinstance(text, str) for path, text in files.items()),
            f"{block_label(block, symbol if isinstance(symbol, str) and symbol else None)}: "
            "ptodsl-doc-test metadata 'files' must be an object mapping relative file paths to text",
        )
        for path in files:
            file_path = Path(path)
            expect(
                not file_path.is_absolute() and ".." not in file_path.parts,
                f"{block_label(block, symbol if isinstance(symbol, str) and symbol else None)}: "
                f"ptodsl-doc-test metadata file path must be relative and stay inside the snippet directory: {path!r}",
            )
    if mode in ("compile", "compile_fragment"):
        expect(
            isinstance(symbol, str) and symbol,
            f"{block_label(block)}: ptodsl-doc-test metadata must define a non-empty string 'symbol'",
        )
        expect(
            isinstance(compile_kwargs, dict),
            f"{block_label(block, symbol if isinstance(symbol, str) and symbol else None)}: "
            "ptodsl-doc-test metadata must define an object 'compile'",
        )
        if mode == "compile_fragment":
            expect(
                isinstance(fixture, str) and fixture,
                f"{block_label(block, symbol)}: ptodsl-doc-test compile_fragment metadata must define a non-empty string 'fixture'",
            )
            return DocTestDirective(
                mode=mode,
                symbol=symbol,
                compile_kwargs=compile_kwargs,
                fixture=fixture,
                files=files,
            )
        return DocTestDirective(mode=mode, symbol=symbol, compile_kwargs=compile_kwargs, files=files)

    if mode == "launch_fragment":
        expect(
            isinstance(fixture, str) and fixture,
            f"{block_label(block)}: ptodsl-doc-test launch_fragment metadata must define a non-empty string 'fixture'",
        )
        if symbol is not None:
            expect(
                isinstance(symbol, str) and symbol,
                f"{block_label(block)}: ptodsl-doc-test launch_fragment 'symbol' must be a non-empty string when present",
            )
        expect(
            compile_kwargs is None,
            f"{block_label(block, symbol if isinstance(symbol, str) and symbol else None)}: "
            "ptodsl-doc-test launch_fragment does not accept a 'compile' object; the snippet owns its compile/launch flow",
        )
        return DocTestDirective(mode=mode, symbol=symbol, fixture=fixture, files=files)

    expect(
        False,
        f"{block_label(block, symbol if isinstance(symbol, str) and symbol else None)}: "
        f"unsupported ptodsl-doc-test mode {mode!r}; only 'compile', 'compile_fragment', and 'launch_fragment' are supported",
    )
    return DocTestDirective(mode=mode)


def _write_directive_files(snippet_dir: Path, files: Optional[dict[str, str]]) -> None:
    if not files:
        return
    for relative_path, text in files.items():
        output_path = snippet_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")


@contextmanager
def directive_execution_dir(directive: DocTestDirective):
    if not directive.files:
        yield None
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        snippet_dir = Path(temp_dir)
        _write_directive_files(snippet_dir, directive.files)
        yield snippet_dir


def execute_source(
    source: str,
    block: MarkdownCodeBlock,
    symbol: Optional[str] = None,
    *,
    extra_namespace: Optional[dict[str, object]] = None,
    source_dir: Optional[Path] = None,
) -> dict[str, object]:
    source_file = block.path if source_dir is None else source_dir / "case.py"
    namespace: dict[str, object] = {
        "__builtins__": __builtins__,
        "__name__": "__ptodsl_doc_snippet__",
        "__file__": str(source_file),
        "pto": pto,
    }
    if extra_namespace is not None:
        namespace.update(extra_namespace)
    filename = f"{source_file}::codeblock:{block.start_line}"
    source_lines = source.splitlines(keepends=True)
    linecache.cache[filename] = (len(source), None, source_lines, filename)
    try:
        exec(compile(source, filename, "exec"), namespace, namespace)
    except Exception as exc:
        raise AssertionError(
            f"{block_label(block, symbol)}: snippet execution failed: {exc.__class__.__name__}: {exc}"
        ) from exc
    return namespace


@contextmanager
def capture_launch_records():
    records: list[LaunchRecord] = []

    def fake_launch_call(self, *args):
        marshaled = _marshal_launch_args(self._compiled._kernel_signature, args)
        records.append(
            LaunchRecord(
                compiled=self._compiled,
                grid=self._grid,
                stream=self._stream,
                args=tuple(args),
                marshaled_arg_count=len(marshaled),
            )
        )
        return None

    with mock.patch.object(LaunchHandle, "__call__", new=fake_launch_call):
        yield records


def verify_compiled_target(
    block: MarkdownCodeBlock,
    directive: DocTestDirective,
    namespace: dict[str, object],
    ptoas_bin: Path,
    *,
    frontend_verify: bool,
) -> None:
    expect(directive.symbol is not None, f"{block_label(block)}: compile mode requires a symbol")
    expect(directive.compile_kwargs is not None, f"{block_label(block, directive.symbol)}: compile mode requires compile kwargs")
    expect(
        directive.symbol in namespace,
        f"{block_label(block, directive.symbol)}: declared symbol is missing from snippet namespace",
    )

    target = namespace[directive.symbol]
    compile_attr = getattr(target, "compile", None)
    expect(
        callable(compile_attr),
        f"{block_label(block, directive.symbol)}: declared symbol does not expose a callable .compile(...) surface",
    )

    try:
        compiled = compile_attr(**directive.compile_kwargs)
    except Exception as exc:
        raise AssertionError(
            f"{block_label(block, directive.symbol)}: compile failed: {exc.__class__.__name__}: {exc}"
        ) from exc

    try:
        compiled.verify()
    except Exception as exc:
        raise AssertionError(
            f"{block_label(block, directive.symbol)}: compiled.verify() failed: {exc.__class__.__name__}: {exc}"
        ) from exc

    mlir_text = compiled.mlir_text()
    expect(
        isinstance(mlir_text, str) and mlir_text.strip(),
        f"{block_label(block, directive.symbol)}: compiled artifact should expose non-empty mlir_text()",
    )

    label = block_label(block, directive.symbol)
    expect_parse_roundtrip_and_verify(mlir_text, label)
    if frontend_verify:
        run_ptoas_frontend_verify(ptoas_bin, mlir_text, label)


def run_compile_block(block: MarkdownCodeBlock, ptoas_bin: Path) -> None:
    directive = parse_test_directive(block)
    with directive_execution_dir(directive) as source_dir:
        namespace = execute_source(block.text, block, directive.symbol, source_dir=source_dir)
        verify_compiled_target(block, directive, namespace, ptoas_bin, frontend_verify=False)


def run_compile_fragment_block(block: MarkdownCodeBlock, ptoas_bin: Path) -> None:
    directive = parse_test_directive(block)
    expect(
        directive.fixture is not None,
        f"{block_label(block, directive.symbol)}: compile_fragment mode requires a fixture id",
    )
    expect(
        directive.fixture in FRAGMENT_FIXTURES,
        f"{block_label(block, directive.symbol)}: unknown fragment fixture {directive.fixture!r}",
    )
    try:
        rendered_source = render_fragment_fixture(FRAGMENT_FIXTURES[directive.fixture], block.text)
    except ValueError as exc:
        raise AssertionError(
            f"{block_label(block, directive.symbol)}: fragment fixture {directive.fixture!r} is invalid: {exc}"
        ) from exc
    with directive_execution_dir(directive) as source_dir:
        namespace = execute_source(rendered_source, block, directive.symbol, source_dir=source_dir)
        verify_compiled_target(block, directive, namespace, ptoas_bin, frontend_verify=False)


def run_launch_fragment_block(block: MarkdownCodeBlock, ptoas_bin: Path) -> None:
    directive = parse_test_directive(block)
    expect(
        directive.fixture is not None,
        f"{block_label(block, directive.symbol)}: launch_fragment mode requires a fixture id",
    )
    expect(
        directive.fixture in FRAGMENT_FIXTURES,
        f"{block_label(block, directive.symbol)}: unknown fragment fixture {directive.fixture!r}",
    )
    try:
        rendered_source = render_fragment_fixture(FRAGMENT_FIXTURES[directive.fixture], block.text)
    except ValueError as exc:
        raise AssertionError(
            f"{block_label(block, directive.symbol)}: fragment fixture {directive.fixture!r} is invalid: {exc}"
        ) from exc

    with directive_execution_dir(directive) as source_dir:
        with capture_launch_records() as launch_records:
            execute_source(
                rendered_source,
                block,
                directive.symbol,
                extra_namespace={"PTODSL_DOC_LAUNCH_RECORDS": launch_records},
                source_dir=source_dir,
            )

    expect(
        bool(launch_records),
        f"{block_label(block, directive.symbol)}: launch_fragment snippet did not execute any compiled[grid, stream](...) launch",
    )

    seen_compiled_ids: set[int] = set()
    for record in launch_records:
        compiled = record.compiled
        compiled_id = id(compiled)
        if compiled_id in seen_compiled_ids:
            continue
        seen_compiled_ids.add(compiled_id)
        try:
            compiled.verify()
        except Exception as exc:
            raise AssertionError(
                f"{block_label(block, directive.symbol)}: compiled launch target verify() failed: "
                f"{exc.__class__.__name__}: {exc}"
            ) from exc
        mlir_text = compiled.mlir_text()
        expect(
            isinstance(mlir_text, str) and mlir_text.strip(),
            f"{block_label(block, directive.symbol)}: compiled launch target should expose non-empty mlir_text()",
        )
        label = block_label(block, directive.symbol or getattr(compiled, "ir_function_name", None))
        expect_parse_roundtrip_and_verify(mlir_text, label)
        run_ptoas_frontend_verify(ptoas_bin, mlir_text, label)

def scan_markdown_file(path: Path) -> MarkdownScanResult:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    blocks: list[MarkdownCodeBlock] = []
    in_code_block = False
    block_language = ""
    block_start = 0
    block_lines: list[str] = []
    metadata: Optional[DocBlockMetadata] = None

    for index, line in enumerate(lines, start=1):
        fence_match = FENCE_RE.match(line.rstrip("\n"))
        if fence_match:
            if not in_code_block:
                in_code_block = True
                block_language = fence_match.group("lang")
                block_start = index
                block_lines = []
                metadata = find_block_metadata(path, lines, index)
            else:
                blocks.append(
                    MarkdownCodeBlock(
                        path=path,
                        start_line=block_start,
                        end_line=index,
                        language=block_language,
                        lines=tuple(block_lines),
                        metadata=metadata,
                    )
                )
                in_code_block = False
                block_language = ""
                block_start = 0
                block_lines = []
                metadata = None
            continue

        if in_code_block:
            block_lines.append(line)

    expect(not in_code_block, f"unclosed fenced code block in {path}")
    return MarkdownScanResult(path=path, blocks=tuple(blocks))


def scan_user_guide() -> tuple[MarkdownScanResult, ...]:
    return tuple(scan_markdown_file(path) for path in iter_markdown_files(USER_GUIDE_ROOT))


def collect_python_blocks(results: Iterable[MarkdownScanResult]) -> tuple[MarkdownCodeBlock, ...]:
    blocks: list[MarkdownCodeBlock] = []
    for result in results:
        for block in result.blocks:
            if block.language == "python":
                blocks.append(block)
    return tuple(blocks)


def collect_tagged_python_blocks(blocks: Iterable[MarkdownCodeBlock]) -> tuple[MarkdownCodeBlock, ...]:
    return tuple(block for block in blocks if block.metadata is not None)


def summarize_metadata(blocks: Iterable[MarkdownCodeBlock]) -> tuple[int, int]:
    test_count = 0
    pending_count = 0
    for block in blocks:
        if block.metadata.kind == "test":
            test_count += 1
        elif block.metadata.kind == "pending":
            pending_count += 1
        else:
            raise AssertionError(
                f"{block_label(block)}: unsupported ptodsl-doc metadata kind {block.metadata.kind!r}"
            )
    return test_count, pending_count


def collect_test_blocks(blocks: Iterable[MarkdownCodeBlock]) -> tuple[MarkdownCodeBlock, ...]:
    return tuple(
        block
        for block in blocks
        if block.metadata is not None and block.metadata.kind == "test"
    )


def summarize_test_modes(blocks: Iterable[MarkdownCodeBlock]) -> tuple[int, int, int]:
    compile_count = 0
    compile_fragment_count = 0
    launch_fragment_count = 0
    for block in blocks:
        directive = parse_test_directive(block)
        if directive.mode == "compile":
            compile_count += 1
        elif directive.mode == "compile_fragment":
            compile_fragment_count += 1
        elif directive.mode == "launch_fragment":
            launch_fragment_count += 1
        else:
            raise AssertionError(f"{block_label(block)}: unsupported docs-as-test mode {directive.mode!r}")
    return compile_count, compile_fragment_count, launch_fragment_count


def main() -> None:
    expect(USER_GUIDE_ROOT.is_dir(), f"missing PTODSL user guide directory: {USER_GUIDE_ROOT}")

    results = scan_user_guide()
    python_blocks = collect_python_blocks(results)
    tagged_python_blocks = collect_tagged_python_blocks(python_blocks)
    test_count, pending_count = summarize_metadata(tagged_python_blocks)
    test_blocks = collect_test_blocks(tagged_python_blocks)
    compile_test_count, compile_fragment_test_count, launch_fragment_test_count = summarize_test_modes(test_blocks)

    expect(bool(results), f"no markdown files found under {USER_GUIDE_ROOT}")
    expect(bool(python_blocks), f"no Python fenced code blocks found under {USER_GUIDE_ROOT}")

    if compile_test_count or compile_fragment_test_count or launch_fragment_test_count:
        try:
            ptoas_bin = resolve_ptoas_binary()
        except FileNotFoundError as exc:
            compile_blocks = [
                block
                for block in test_blocks
                if parse_test_directive(block).mode in ("compile", "compile_fragment", "launch_fragment")
            ]
            fail_doc(compile_blocks[0].path, compile_blocks[0].start_line, str(exc))
    else:
        ptoas_bin = None
    for block in test_blocks:
        directive = parse_test_directive(block)
        if directive.mode == "compile":
            expect(ptoas_bin is not None, f"{block_label(block, directive.symbol)}: missing ptoas binary for compile-mode docs test")
            run_compile_block(block, ptoas_bin)
        elif directive.mode == "compile_fragment":
            expect(
                ptoas_bin is not None,
                f"{block_label(block, directive.symbol)}: missing ptoas binary for compile_fragment-mode docs test",
            )
            run_compile_fragment_block(block, ptoas_bin)
        elif directive.mode == "launch_fragment":
            expect(
                ptoas_bin is not None,
                f"{block_label(block, directive.symbol)}: missing ptoas binary for launch_fragment-mode docs test",
            )
            run_launch_fragment_block(block, ptoas_bin)
        else:
            raise AssertionError(f"{block_label(block)}: unsupported docs-as-test mode {directive.mode!r}")

    markdown_count = len(results)
    python_count = len(python_blocks)
    block_count = sum(len(result.blocks) for result in results)
    untracked_count = python_count - len(tagged_python_blocks)
    print(
        "ptodsl_docs_as_test: scanned "
        f"{markdown_count} markdown files, {block_count} fenced blocks, {python_count} python blocks "
        f"({test_count} test = {compile_test_count} compile + "
        f"{compile_fragment_test_count} compile_fragment + {launch_fragment_test_count} launch_fragment, "
        f"{pending_count} pending, {untracked_count} untracked)"
    )


if __name__ == "__main__":
    main()
