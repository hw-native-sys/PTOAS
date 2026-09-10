#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import inspect
import warnings

from ptodsl import pto
from ptodsl._diagnostics import PTODSLDeprecationWarning, deprecated


@deprecated("use replacement() instead")
def old_function(value, *, scale=1):
    """Return a scaled value."""
    return value * scale


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    expect(pto.deprecated is deprecated, "deprecated should be exported on pto")
    expect(old_function.__name__ == "old_function", "decorator should preserve function metadata")
    expect(old_function.__doc__ == "Return a scaled value.", "decorator should preserve the docstring")
    expect(
        str(inspect.signature(old_function)) == "(value, *, scale=1)",
        "decorator should preserve the callable signature",
    )
    expect(
        old_function.__deprecated__ == "use replacement() instead",
        "decorator should expose its migration reason",
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = old_function(3, scale=2)

    expect(result == 6, "decorator should preserve the wrapped result")
    expect(len(captured) == 1, "deprecated call should emit one warning")
    warning = captured[0]
    expect(warning.category is PTODSLDeprecationWarning, "warning should use the PTODSL category")
    expect("old_function is deprecated" in str(warning.message), "warning should name the old API")
    expect("use replacement() instead" in str(warning.message), "warning should include the migration reason")
    expect(warning.filename == __file__, "warning should point at the caller")

    for invalid_reason in ("", 123):
        try:
            deprecated(invalid_reason)
        except TypeError:
            pass
        else:
            raise AssertionError("deprecated() should reject an invalid reason")

    print("ptodsl_deprecated: PASS")


if __name__ == "__main__":
    main()
