# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Lazy loader for ported, per-architecture TileLib templates."""

from functools import lru_cache
from importlib import import_module


_TEMPLATE_MODULES = {
    ("a5", "pto.tabs"): ".a5.unary.tabs",
    ("a5", "pto.tadd"): ".a5.elementwise.tadd",
    ("a5", "pto.tadds"): ".a5.elementwise.tadds",
    ("a5", "pto.tand"): ".a5.elementwise.tand",
    ("a5", "pto.tands"): ".a5.elementwise.tands",
    ("a5", "pto.tci"): ".a5.unary.tci",
    ("a5", "pto.tcmp"): ".a5.tcmp",
    ("a5", "pto.tcmps"): ".a5.tcmps",
    ("a5", "pto.tcolargmax"): ".a5.reduce.tcolargmax",
    ("a5", "pto.tcolargmin"): ".a5.reduce.tcolargmin",
    ("a5", "pto.tcolmax"): ".a5.tcolmax",
    ("a5", "pto.tcolexpand"): ".a5.expand.tcolexpand",
    ("a5", "pto.tcolexpandadd"): ".a5.expand.tcolexpandadd",
    ("a5", "pto.tcolexpanddiv"): ".a5.tcolexpanddiv",
    ("a5", "pto.tcolexpandexpdif"): ".a5.expand.tcolexpandexpdif",
    ("a5", "pto.tcolexpandmax"): ".a5.expand.tcolexpandmax",
    ("a5", "pto.tcolexpandmin"): ".a5.expand.tcolexpandmin",
    ("a5", "pto.tcolexpandmul"): ".a5.expand.tcolexpandmul",
    ("a5", "pto.tcolexpandsub"): ".a5.expand.tcolexpandsub",
    ("a5", "pto.tcolmin"): ".a5.tcolmin",
    ("a5", "pto.tcolprod"): ".a5.reduce.tcolprod",
    ("a5", "pto.tcolsum"): ".a5.reduce.tcolsum",
    ("a5", "pto.tconcat"): ".a5.memory.tconcat",
    ("a5", "pto.tdiv"): ".a5.tdiv",
    ("a5", "pto.tdivs"): ".a5.tdivs",
    ("a5", "pto.tcvt"): ".a5.tcvt",
    ("a5", "pto.tdequant"): ".a5.tdequant",
    ("a5", "pto.texp"): ".a5.unary.texp",
    ("a5", "pto.texpands"): ".a5.expand.texpand",
    ("a5", "pto.textract"): (".a5.memory.textract", ".a5.memory.textract_fp"),
    ("a5", "pto.tfmod"): ".a5.elementwise.tfmod",
    ("a5", "pto.tfmods"): ".a5.elementwise.tfmods",
    ("a5", "pto.tfillpad"): ".a5.memory.tfillpad",
    ("a5", "pto.tgather"): ".a5.tgather",
    ("a5", "pto.tgatherb"): ".a5.memory.tgatherb",
    ("a5", "pto.tgemv"): ".a5.matmul.tgemv",
    ("a5", "pto.tgemv.acc"): ".a5.matmul.tgemv_acc",
    ("a5", "pto.tgemv.bias"): ".a5.matmul.tgemv_bias",
    ("a5", "pto.tgemv.mx"): ".a5.matmul.tgemv_mx",
    ("a5", "pto.tgemv.mx.acc"): ".a5.matmul.tgemv_mx",
    ("a5", "pto.tgemv.mx.bias"): ".a5.matmul.tgemv_mx",
    ("a5", "pto.thistogram"): ".a5.thistogram",
    ("a5", "pto.tinsert"): ".a5.tinsert",
    ("a5", "pto.tlrelu"): ".a5.tlrelu",
    ("a5", "pto.tlog"): ".a5.tlog",
    ("a5", "pto.tload"): ".a5.memory.tload",
    ("a5", "pto.tmax"): ".a5.elementwise.tmax",
    ("a5", "pto.tmaxs"): ".a5.elementwise.tmaxs",
    ("a5", "pto.tmin"): ".a5.elementwise.tmin",
    ("a5", "pto.tmins"): ".a5.elementwise.tmins",
    ("a5", "pto.tmatmul"): ".a5.matmul.tmatmul",
    ("a5", "pto.tmatmul.acc"): ".a5.matmul.tmatmul_acc",
    ("a5", "pto.tmatmul.bias"): ".a5.matmul.tmatmul_bias",
    ("a5", "pto.tmatmul.mx"): ".a5.matmul.tmatmul_mx",
    ("a5", "pto.tmatmul.mx.acc"): ".a5.matmul.tmatmul_mx",
    ("a5", "pto.tmatmul.mx.bias"): ".a5.matmul.tmatmul_mx",
    ("a5", "pto.tmov"): (
        ".a5.memory.tmov",
        ".a5.memory.tmov2bias",
        ".a5.memory.tmov2left",
        ".a5.memory.tmov2right",
        ".a5.memory.tmov2scale",
        ".a5.memory.tmov2vec",
        ".a5.memory.tmov_fp",
    ),
    ("a5", "pto.tmul"): ".a5.elementwise.tmul",
    ("a5", "pto.tmuls"): ".a5.elementwise.tmuls",
    ("a5", "pto.tneg"): ".a5.unary.tneg",
    ("a5", "pto.tnot"): ".a5.unary.tnot",
    ("a5", "pto.tor"): ".a5.elementwise.tor",
    ("a5", "pto.tors"): ".a5.elementwise.tors",
    ("a5", "pto.tpartadd"): ".a5.reduce.tpartadd",
    ("a5", "pto.tpartmax"): ".a5.reduce.tpartmax",
    ("a5", "pto.tpartmin"): ".a5.reduce.tpartmin",
    ("a5", "pto.tpartmul"): ".a5.reduce.tpartmul",
    ("a5", "pto.tprelu"): ".a5.tprelu",
    ("a5", "pto.trelu"): ".a5.unary.trelu",
    ("a5", "pto.trecip"): ".a5.trecip",
    ("a5", "pto.trem"): ".a5.elementwise.trem",
    ("a5", "pto.trems"): ".a5.elementwise.trems",
    ("a5", "pto.trandom"): ".a5.trandom",
    ("a5", "pto.trsqrt"): ".a5.unary.trsqrt",
    ("a5", "pto.trowargmax"): ".a5.reduce.trowargmax",
    ("a5", "pto.trowargmin"): ".a5.reduce.trowargmin",
    ("a5", "pto.trowexpand"): ".a5.expand.trowexpand",
    ("a5", "pto.trowexpandadd"): ".a5.expand.trowexpandadd",
    ("a5", "pto.trowexpanddiv"): ".a5.expand.trowexpanddiv",
    ("a5", "pto.trowexpandexpdif"): ".a5.expand.trowexpandexpdif",
    ("a5", "pto.trowexpandmax"): ".a5.expand.trowexpandmax",
    ("a5", "pto.trowexpandmin"): ".a5.expand.trowexpandmin",
    ("a5", "pto.trowexpandmul"): ".a5.expand.trowexpandmul",
    ("a5", "pto.trowexpandsub"): ".a5.expand.trowexpandsub",
    ("a5", "pto.trowmax"): ".a5.reduce.trowmax",
    ("a5", "pto.trowmin"): ".a5.reduce.trowmin",
    ("a5", "pto.trowprod"): ".a5.reduce.trowprod",
    ("a5", "pto.trowsum"): ".a5.reduce.trowsum",
    ("a5", "pto.tscatter"): ".a5.tscatter",
    ("a5", "pto.tsel"): ".a5.tsel",
    ("a5", "pto.tsels"): ".a5.tsels",
    ("a5", "pto.tshl"): ".a5.elementwise.tshl",
    ("a5", "pto.tshls"): ".a5.elementwise.tshls",
    ("a5", "pto.tshr"): ".a5.elementwise.tshr",
    ("a5", "pto.tshrs"): ".a5.elementwise.tshrs",
    ("a5", "pto.tmrgsort"): ".a5.tmrgsort",
    ("a5", "pto.tsort32"): ".a5.tsort32",
    ("a5", "pto.tstore"): ".a5.tstore",
    ("a5", "pto.tsub"): ".a5.elementwise.tsub",
    ("a5", "pto.tsubs"): ".a5.elementwise.tsubs",
    ("a5", "pto.tsqrt"): ".a5.unary.tsqrt",
    ("a5", "pto.ttrans"): ".a5.ttrans",
    ("a5", "pto.ttri"): ".a5.unary.ttri",
    ("a5", "pto.txor"): ".a5.elementwise.txor",
    ("a5", "pto.txors"): ".a5.elementwise.txors",
}


@lru_cache(maxsize=None)
def load_template(op: str, target: str) -> bool:
    """Import and register only the template module for ``(target, op)``.

    Both this cache and Python's module cache make repeated requests no-ops.
    Returns ``False`` when this TileLib has no module for the requested pair.
    """
    module_names = _TEMPLATE_MODULES.get((target, op))
    if module_names is None:
        return False
    if isinstance(module_names, str):
        module_names = (module_names,)
    for module_name in module_names:
        import_module(module_name, package=__name__)
    return True


__all__ = ["load_template"]
