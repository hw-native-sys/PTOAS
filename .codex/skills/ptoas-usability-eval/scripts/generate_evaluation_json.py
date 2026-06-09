#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import copy
import json
import logging
import sys
from pathlib import Path


LOGGER = logging.getLogger(__name__)

TOP_LEVEL_FIELDS = [
    "evaluation_id",
    "operator_name",
    "batch_id",
    "development_success",
    "test_results",
    "summary",
    "documentation_retrieval",
    "code_examples",
    "build_configuration",
    "functional_testing",
    "key_findings",
    "dimension_tables",
    "evaluated_at",
    "report_file",
]

SUMMARY_FIELDS = [
    "support_score",
    "measured_score",
    "documentation_score",
    "example_score",
    "build_score",
    "debugging_score",
    "efficiency_score",
    "prompt_tokens",
    "completion_tokens",
]

TEST_LEVEL_FIELDS = ("level_0", "level_1", "level_2", "level_3")
TEST_RESULT_FIELDS = [*TEST_LEVEL_FIELDS, "passed_count", "total_count", "pass_rate"]
DOCUMENTATION_FIELDS = [
    "total_searches",
    "effective_searches",
    "effectiveness_rate",
]
CODE_EXAMPLE_FIELDS = [
    "sampled_case_count",
    "sampled_cases",
    "modified_examples",
    "modification_rate",
]
BUILD_CONFIGURATION_FIELDS = ["config_lines", "macro_count", "macros"]
FUNCTIONAL_TEST_FIELDS = [
    "compile_run_count",
    "cycles",
    "sampled_case_count",
    "sampled_cases",
]

LEVEL_LABELS = {
    "level_0": "L1 文档审阅层",
    "level_1": "L2 本地最小运行层",
    "level_2": "L3 Linux compile-only 层",
    "level_3": "L4 NPU 上板层",
}

DIMENSION_TABLES = {
    "discoverability": [
        "doc_search",
        "navigation_depth",
        "multi_entry_access",
        "version_lookup",
    ],
    "consistency": [
        "document_structure",
        "concept_alignment",
        "interface_layering",
    ],
    "accuracy": [
        "doc_correctness",
        "version_matrix",
        "build_signal_accuracy",
    ],
    "completeness": [
        "document_coverage",
        "sample_coverage",
        "tool_coverage",
        "deliverable_completeness",
    ],
    "learnability": [
        "progressive_guidance",
        "cognitive_load",
        "key_path_visibility",
    ],
    "practicability": [
        "one_shot_success",
        "example_reuse",
        "operation_steps",
        "deployment_readiness",
    ],
    "debuggability": [
        "error_context",
        "remediation_hint",
        "signal_to_noise",
    ],
}

REPRESENTATIVE_CASES = [
    "test/samples/AddPtr",
    "test/samples/AllocTile",
    "test/samples/AsyncComm",
    "test/samples/Bf16",
    "test/samples/CommSync",
    "test/samples/Complex",
    "test/samples/ControlFlow",
    "test/samples/Cvt",
    "test/samples/Dequant",
    "test/samples/DynamicTailMatmul",
    "test/samples/FFN",
    "test/samples/FlashAttention",
    "test/samples/Gather",
    "test/samples/Gemv",
    "test/samples/GQA",
    "test/samples/LayoutInference",
    "test/samples/MatMul",
    "test/samples/Mgather",
    "test/samples/Mscatter",
    "test/samples/Partition5D",
    "test/samples/PyPTOIRParser",
    "test/samples/Quant",
    "test/samples/Qwen3DecodeA3",
    "test/samples/Qwen3DecodeA5",
    "test/samples/Scatter",
    "test/samples/SetValidShape",
    "test/samples/Sync",
    "test/samples/SyncAll",
    "test/samples/TPrefetch",
    "test/samples/TPrefetchAsync",
    "test/samples/TPushTPop",
    "test/samples/planmemory",
]


def _build_dimension_properties():
    return {
        dimension_name: {
            "type": "object",
            "required": subdimensions,
            "properties": {
                subdimension: {"type": "integer", "minimum": 1, "maximum": 10}
                for subdimension in subdimensions
            },
        }
        for dimension_name, subdimensions in DIMENSION_TABLES.items()
    }


def _build_schema_properties():
    return {
        "evaluation_id": {"type": "string"},
        "operator_name": {"type": "string"},
        "batch_id": {"type": "string"},
        "development_success": {"type": "boolean"},
        "test_results": {"type": "object", "required": TEST_RESULT_FIELDS},
        "summary": {"type": "object", "required": SUMMARY_FIELDS},
        "documentation_retrieval": {
            "type": "object",
            "required": DOCUMENTATION_FIELDS,
        },
        "code_examples": {"type": "object", "required": CODE_EXAMPLE_FIELDS},
        "build_configuration": {
            "type": "object",
            "required": BUILD_CONFIGURATION_FIELDS,
        },
        "functional_testing": {
            "type": "object",
            "required": FUNCTIONAL_TEST_FIELDS,
        },
        "key_findings": {"type": "array"},
        "dimension_tables": {
            "type": "object",
            "required": list(DIMENSION_TABLES.keys()),
            "properties": _build_dimension_properties(),
        },
        "evaluated_at": {"type": "string"},
        "report_file": {"type": "string"},
    }


def _build_evaluation_json_schema():
    return {
        "type": "object",
        "required": TOP_LEVEL_FIELDS,
        "properties": _build_schema_properties(),
    }


def _clamp(number, lower_bound, upper_bound):
    return max(lower_bound, min(upper_bound, number))


def _safe_ratio(numerator, denominator):
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _normalized_ratio(numerator, denominator):
    return round(_clamp(_safe_ratio(numerator, denominator), 0.0, 1.0), 4)


def _coerce_non_negative_int(value):
    return max(0, int(value))


def _normalize_level_result(level_result):
    passed_count = _coerce_non_negative_int(level_result.get("passed_count", 0))
    total_count = _coerce_non_negative_int(level_result.get("total_count", 0))
    if passed_count > total_count:
        passed_count = total_count
    level_result["passed_count"] = passed_count
    level_result["total_count"] = total_count
    return passed_count, total_count


def _normalize_test_results(test_results):
    passed_total = 0
    case_total = 0
    for level_name in TEST_LEVEL_FIELDS:
        level_passed, level_cases = _normalize_level_result(test_results[level_name])
        passed_total += level_passed
        case_total += level_cases
    test_results["passed_count"] = passed_total
    test_results["total_count"] = case_total
    test_results["pass_rate"] = _normalized_ratio(passed_total, case_total)


def _normalize_summary(summary):
    for score_field in SUMMARY_FIELDS:
        if score_field.endswith("_tokens"):
            summary[score_field] = _coerce_non_negative_int(summary.get(score_field, 0))
            continue
        summary[score_field] = round(
            _clamp(float(summary.get(score_field, 1)), 1.0, 10.0),
            2,
        )


def _normalize_documentation(documentation):
    documentation["total_searches"] = _coerce_non_negative_int(
        documentation.get("total_searches", 0)
    )
    documentation["effective_searches"] = _coerce_non_negative_int(
        documentation.get("effective_searches", 0)
    )
    if documentation["effective_searches"] > documentation["total_searches"]:
        documentation["effective_searches"] = documentation["total_searches"]
    documentation["effectiveness_rate"] = _normalized_ratio(
        documentation["effective_searches"],
        documentation["total_searches"],
    )


def _normalize_code_examples(code_examples):
    code_examples["sampled_case_count"] = _coerce_non_negative_int(
        code_examples.get("sampled_case_count", 0)
    )
    code_examples["modified_examples"] = _coerce_non_negative_int(
        code_examples.get("modified_examples", 0)
    )
    if code_examples["modified_examples"] > code_examples["sampled_case_count"]:
        code_examples["modified_examples"] = code_examples["sampled_case_count"]
    code_examples["modification_rate"] = _normalized_ratio(
        code_examples["modified_examples"],
        code_examples["sampled_case_count"],
    )


def _normalize_build_configuration(build_configuration):
    build_configuration["config_lines"] = _coerce_non_negative_int(
        build_configuration.get("config_lines", 0)
    )
    build_configuration["macro_count"] = _coerce_non_negative_int(
        build_configuration.get("macro_count", 0)
    )


def _normalize_functional_testing(functional_testing):
    functional_testing["compile_run_count"] = _coerce_non_negative_int(
        functional_testing.get("compile_run_count", 0)
    )
    functional_testing["cycles"] = _coerce_non_negative_int(
        functional_testing.get("cycles", 0)
    )
    functional_testing["sampled_case_count"] = _coerce_non_negative_int(
        functional_testing.get("sampled_case_count", 0)
    )


def _normalize_dimension_tables(dimension_tables):
    for dimension_name, subdimensions in DIMENSION_TABLES.items():
        for subdimension in subdimensions:
            dimension_tables[dimension_name][subdimension] = int(
                _clamp(
                    int(dimension_tables[dimension_name].get(subdimension, 1)),
                    1,
                    10,
                )
            )


def _postprocess_evaluation_json(payload):
    processed = copy.deepcopy(payload)
    _normalize_test_results(processed["test_results"])
    _normalize_summary(processed["summary"])
    _normalize_documentation(processed["documentation_retrieval"])
    _normalize_code_examples(processed["code_examples"])
    _normalize_build_configuration(processed["build_configuration"])
    _normalize_functional_testing(processed["functional_testing"])
    _normalize_dimension_tables(processed["dimension_tables"])
    return processed


def _require_fields(container, fields, container_name):
    missing_fields = [field for field in fields if field not in container]
    if missing_fields:
        raise ValueError(f"missing {container_name}: {missing_fields}")


def _validate_string_fields(payload):
    for field_name in (
        "evaluation_id",
        "operator_name",
        "batch_id",
        "evaluated_at",
        "report_file",
    ):
        if not isinstance(payload[field_name], str):
            raise ValueError(f"{field_name} must be a string")


def _validate_summary(summary):
    _require_fields(summary, SUMMARY_FIELDS, "summary fields")
    for score_field in SUMMARY_FIELDS:
        if score_field.endswith("_tokens"):
            continue
        score = float(summary[score_field])
        if not 1.0 <= score <= 10.0:
            raise ValueError(f"{score_field} out of range: {score}")


def _validate_dimension_tables(dimension_tables):
    _require_fields(dimension_tables, DIMENSION_TABLES.keys(), "dimensions")
    for dimension_name, subdimensions in DIMENSION_TABLES.items():
        dimension_values = dimension_tables[dimension_name]
        if not 3 <= len(dimension_values) <= 4:
            raise ValueError(
                f"dimension {dimension_name} must contain 3-4 subdimensions"
            )
        _require_fields(
            dimension_values,
            subdimensions,
            f"{dimension_name} subdimensions",
        )
        for subdimension in subdimensions:
            score = dimension_values[subdimension]
            if not isinstance(score, int) or not 1 <= score <= 10:
                raise ValueError(
                    f"subdimension score out of range: {dimension_name}.{subdimension}"
                )


def _collect_non_negative_counts(payload):
    return [
        payload["test_results"]["passed_count"],
        payload["test_results"]["total_count"],
        payload["documentation_retrieval"]["total_searches"],
        payload["documentation_retrieval"]["effective_searches"],
        payload["code_examples"]["sampled_case_count"],
        payload["code_examples"]["modified_examples"],
        payload["build_configuration"]["config_lines"],
        payload["build_configuration"]["macro_count"],
        payload["functional_testing"]["compile_run_count"],
        payload["functional_testing"]["cycles"],
        payload["functional_testing"]["sampled_case_count"],
        payload["summary"]["prompt_tokens"],
        payload["summary"]["completion_tokens"],
    ]


def _validate_non_negative_counts(payload):
    count_fields = _collect_non_negative_counts(payload)
    if any(int(value) < 0 for value in count_fields):
        raise ValueError("token/count fields must be non-negative")


def _validate_test_results(test_results):
    _require_fields(test_results, TEST_RESULT_FIELDS, "test_results fields")
    pass_rate = float(test_results["pass_rate"])
    if not 0.0 <= pass_rate <= 1.0:
        raise ValueError(f"pass_rate out of range: {pass_rate}")
    if int(test_results["passed_count"]) > int(test_results["total_count"]):
        raise ValueError("passed_count cannot exceed total_count")
    recomputed_pass_rate = round(
        _safe_ratio(int(test_results["passed_count"]), int(test_results["total_count"])),
        4,
    )
    if abs(recomputed_pass_rate - pass_rate) > 0.01:
        raise ValueError(
            "pass_rate mismatch: "
            f"expected {recomputed_pass_rate}, got {test_results['pass_rate']}"
        )


def _validate_documentation(documentation):
    _require_fields(documentation, DOCUMENTATION_FIELDS, "documentation fields")
    if int(documentation["effective_searches"]) > int(documentation["total_searches"]):
        raise ValueError("effective_searches cannot exceed total_searches")


def _validate_evaluation_json(payload):
    _require_fields(payload, TOP_LEVEL_FIELDS, "top-level fields")
    _validate_string_fields(payload)
    if not isinstance(payload["development_success"], bool):
        raise ValueError("development_success must be a bool")
    _validate_summary(payload["summary"])
    _validate_dimension_tables(payload["dimension_tables"])
    _validate_non_negative_counts(payload)
    _validate_test_results(payload["test_results"])
    _validate_documentation(payload["documentation_retrieval"])


def _existing_representative_cases(repo_root):
    return [
        case_path for case_path in REPRESENTATIVE_CASES if (repo_root / case_path).exists()
    ]


def _resolve_report_file(repo_root, output_path):
    try:
        return output_path.relative_to(repo_root).as_posix()
    except ValueError:
        return output_path.as_posix()


def _build_test_results_section():
    test_results = {
        level_name: {
            "label": level_label,
            "passed_count": 0,
            "total_count": 0,
        }
        for level_name, level_label in LEVEL_LABELS.items()
    }
    test_results.update({"passed_count": 0, "total_count": 0, "pass_rate": 0.0})
    return test_results


def _build_summary_section():
    summary = {field_name: 5 for field_name in SUMMARY_FIELDS}
    summary["prompt_tokens"] = 0
    summary["completion_tokens"] = 0
    return summary


def _build_key_findings():
    return [
        {
            "rank": 1,
            "finding": (
                "This file is a starter template. Replace placeholder scores and "
                "counts after actual touchpoint scoring."
            ),
            "category": "process",
        },
        {
            "rank": 2,
            "finding": (
                "Representative case pack is preselected from PTOAS samples to cover "
                "compile, validation, shape, sync, layout, precision and model-style "
                "paths."
            ),
            "category": "sampling",
        },
        {
            "rank": 3,
            "finding": (
                "Use real retrieval counts, compile cycles and run outcomes before "
                "treating this JSON as a measured report."
            ),
            "category": "validation",
        },
    ]


def _build_dimension_table_defaults():
    return {
        dimension_name: {subdimension: 5 for subdimension in subdimensions}
        for dimension_name, subdimensions in DIMENSION_TABLES.items()
    }


def _build_template_payload(repo_root, output_path):
    representative_cases = _existing_representative_cases(repo_root)
    representative_case_count = len(representative_cases)
    payload = {
        "evaluation_id": "ptoas-touchpoint-eval-template",
        "operator_name": "PTOAS",
        "batch_id": f"representative-samples-{representative_case_count}",
        "development_success": False,
        "test_results": _build_test_results_section(),
        "summary": _build_summary_section(),
        "documentation_retrieval": {
            "total_searches": 0,
            "effective_searches": 0,
            "effectiveness_rate": 0.0,
            "tracked_queries": [],
        },
        "code_examples": {
            "sampled_case_count": representative_case_count,
            "sampled_cases": representative_cases,
            "modified_examples": 0,
            "modification_rate": 0.0,
        },
        "build_configuration": {
            "config_lines": 0,
            "macro_count": 0,
            "macros": [],
        },
        "functional_testing": {
            "compile_run_count": 0,
            "cycles": 0,
            "sampled_case_count": representative_case_count,
            "sampled_cases": representative_cases,
        },
        "key_findings": _build_key_findings(),
        "dimension_tables": _build_dimension_table_defaults(),
        "evaluated_at": "1970-01-01T00:00:00Z",
        "report_file": _resolve_report_file(repo_root, output_path),
    }
    processed = _postprocess_evaluation_json(payload)
    _validate_evaluation_json(processed)
    return processed


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate PTOAS touchpoint evaluation JSON template."
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="PTOAS repository root.",
    )
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parents[1]
            / "assets"
            / "ptoas_touchpoint_evaluation_template.json"
        ),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--emit-schema",
        action="store_true",
        help="Print the JSON schema-like contract instead of the template payload.",
    )
    return parser.parse_args()


def _build_output_payload(arguments, output_path):
    if arguments.emit_schema:
        return _build_evaluation_json_schema()
    repo_root = Path(arguments.repo_root).resolve()
    return _build_template_payload(repo_root, output_path)


def _write_payload(output_path, payload):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def _configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )


def main():
    _configure_logging()
    arguments = _parse_arguments()
    output_path = Path(arguments.output).resolve()
    payload = _build_output_payload(arguments, output_path)
    _write_payload(output_path, payload)
    LOGGER.info("%s", output_path)


if __name__ == "__main__":
    main()
