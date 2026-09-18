# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Exact-workload certification from explicitly supplied measurement evidence.

This validates evidence; it neither manufactures board measurements nor extends
a certificate to unmeasured shapes, compiler artifacts, devices, or models.
"""
import math
import random
from statistics import mean

from pto_costmodel.wire import fields, fingerprint, integer, require


def certify(policy, evidence):
    fields(policy, ("minimum_samples", "minimum_speedup", "maximum_regression", "maximum_prediction_error",
                    "absolute_tolerance", "relative_tolerance"),
           ("confidence_level", "bootstrap_resamples", "bootstrap_seed"))
    integer(policy["minimum_samples"], "minimum_samples", 3)
    for key in set(policy) - {"minimum_samples", "bootstrap_resamples", "bootstrap_seed"}:
        _number(policy[key], key)
    paired = {"confidence_level", "bootstrap_resamples", "bootstrap_seed"}
    require(not (set(policy) & paired) or paired.issubset(policy), "POLICY",
            "paired bootstrap fields must be provided together")
    if paired.issubset(policy):
        require(0 < policy["confidence_level"] < 1, "POLICY", "confidence level must be in (0,1)")
        integer(policy["bootstrap_resamples"], "bootstrap_resamples", 1)
        integer(policy["bootstrap_seed"], "bootstrap_seed")
    require(policy["maximum_regression"] < 1, "POLICY", "regression limit must be below 1")
    fields(evidence, ("identity", "candidate_id", "schedule_fingerprint", "model", "device", "measurement_environment",
                      "baseline_artifact_fingerprint", "candidate_artifact_fingerprint", "predicted_latency_us",
                      "baseline_us", "candidate_us", "correctness", "runner_revision"))
    for key in ("device", "measurement_environment", "baseline_artifact_fingerprint",
                "candidate_artifact_fingerprint", "runner_revision"):
        require(isinstance(evidence[key], str) and bool(evidence[key]), "EVIDENCE", f"missing {key}")
    _correctness(policy, evidence["correctness"])
    baseline, candidate = evidence["baseline_us"], evidence["candidate_us"]
    require(isinstance(baseline, list) and isinstance(candidate, list)
            and len(baseline) == len(candidate) >= policy["minimum_samples"], "EVIDENCE", "paired samples required")
    for value in baseline + candidate:
        _number(value, "latency", positive=True)
    _number(evidence["predicted_latency_us"], "prediction", positive=True)
    pair_gains = [1 - c / b for b, c in zip(baseline, candidate)]
    speedup = mean(pair_gains) if paired.issubset(policy) else mean(baseline) / mean(candidate) - 1
    worst_regression = max(c / b - 1 for b, c in zip(baseline, candidate))
    error = abs(evidence["predicted_latency_us"] - mean(candidate)) / mean(candidate)
    confidence_interval = None
    if paired.issubset(policy):
        rng = random.Random(policy["bootstrap_seed"])
        distribution = [mean(pair_gains[rng.randrange(len(pair_gains))] for _ in pair_gains)
                        for _ in range(policy["bootstrap_resamples"])]
        alpha = (1 - policy["confidence_level"]) / 2
        confidence_interval = [_percentile(distribution, alpha), _percentile(distribution, 1 - alpha)]
    passed = (speedup >= policy["minimum_speedup"] and worst_regression <= policy["maximum_regression"]
              and error <= policy["maximum_prediction_error"]
              and (confidence_interval is None or confidence_interval[0] > 0))
    report = dict(status="certified_exact_workload" if passed else "not_certified",
                  evidence_fingerprint=fingerprint(evidence),
                  policy_fingerprint=fingerprint(policy), identity=evidence["identity"],
                  candidate_id=evidence["candidate_id"], model=evidence["model"], device=evidence["device"],
                  candidate_artifact_fingerprint=evidence["candidate_artifact_fingerprint"],
                  measurement_environment=evidence["measurement_environment"],
                  metrics=dict(mean_speedup=speedup, worst_paired_regression=worst_regression,
                               prediction_error=error, bootstrap_confidence_interval=confidence_interval))
    report["certificate_id"] = fingerprint(report)
    report["evidence_origin"] = "external_measurement_report"
    report["automatic_application"] = False
    return report


def _number(value, name, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0),
            "EVIDENCE", f"invalid {name}")


def _correctness(policy, evidence):
    fields(evidence, ("golden_fingerprint", "passed", "absolute_tolerance", "relative_tolerance",
                      "deadlock_free", "bounds_checked"))
    require(evidence["passed"] is True and evidence["deadlock_free"] is True and evidence["bounds_checked"] is True,
            "CORRECTNESS", "correctness validation incomplete")
    require(isinstance(evidence["golden_fingerprint"], str) and evidence["golden_fingerprint"],
            "CORRECTNESS", "independent golden missing")
    for key in ("absolute_tolerance", "relative_tolerance"):
        _number(evidence[key], key)
        require(evidence[key] <= policy[key], "CORRECTNESS", "measurement tolerance exceeds policy")


def _percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
