// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- DialectPTO.cpp -----------------------------------------------------===//
//
// Python bindings for the PTO dialect types embedded in PTOASCompiler. The
// thin ptoas._core entry point calls this implementation, while the public
// Python facade remains ptoas.mlir.dialects.pto.
//
//===----------------------------------------------------------------------===//

#include <stdexcept>
#include <string>

#include "PTOModule.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "pto-c/Dialect/PTO.h"
#include "pto-c/Dialect/PTOEnums.h"
#include "pybind11/stl.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

static std::vector<int64_t> toInt64Vector(const py::sequence &seq) {
  std::vector<int64_t> out;
  out.reserve(seq.size());
  for (py::handle h : seq) {
    out.push_back(py::cast<int64_t>(h));
  }
  return out;
}

static std::vector<int64_t> toShapeVectorOrDynamicRank(py::object shapeOrRank) {
  if (py::isinstance<py::int_>(shapeOrRank)) {
    auto rank = shapeOrRank.cast<int64_t>();
    if (rank < 0) {
      throw py::value_error("rank must be non-negative");
    }
    return std::vector<int64_t>(static_cast<size_t>(rank),
                                mlirShapedTypeGetDynamicSize());
  }
  return toInt64Vector(shapeOrRank.cast<py::sequence>());
}

static MlirContext inferContextFromElementType(MlirContext context,
                                               MlirType elementType) {
  if (!mlirContextIsNull(context)) {
    return context;
  }
  if (mlirTypeIsNull(elementType)) {
    throw py::value_error("context is required when element_type is null");
  }
  return mlirTypeGetContext(elementType);
}

static int32_t enumValueFromPy(py::object value, const char *attrName,
                               const char *enumName) {
  if (py::isinstance<py::int_>(value)) {
    return value.cast<int32_t>();
  }
  if (py::hasattr(value, "value")) {
    return value.attr("value").cast<int32_t>();
  }
  throw std::runtime_error(std::string(attrName) + ".get expects int or " +
                           enumName + " enum");
}

static void bindPTOEnumAttr(pybind11::module &m, const char *attrName,
                            const char *enumName,
                            bool (*isA)(MlirAttribute),
                            MlirAttribute (*get)(MlirContext, int32_t),
                            int32_t (*getValue)(MlirAttribute)) {
  mlir_attribute_subclass(m, attrName, isA)
      .def_classmethod(
          "get",
          [attrName, enumName, get](py::object cls, py::object value,
                                    MlirContext ctx) -> py::object {
            int32_t v = enumValueFromPy(value, attrName, enumName);
            MlirAttribute a = get(ctx, v);
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls.attr("__call__")(a);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
      .def_property_readonly("value", [getValue](MlirAttribute self) {
        return getValue(self);
      });
}

static py::list shapeToPyList(const int64_t *data, intptr_t n) {
  py::list lst;
  for (intptr_t i = 0; i < n; ++i) {
    lst.append(py::int_(data[i]));
  }
  return lst;
}

static py::object wrapAttributeAs(const py::module_ &m, const char *className,
                                  MlirAttribute attr) {
  if (mlirAttributeIsNull(attr)) {
    return py::none();
  }
  py::object cls = m.attr(className);
  return cls.attr("__call__")(attr);
}

static MlirAttribute optionalAttributeFromPy(py::object attr) {
  if (attr.is_none()) {
    return MlirAttribute{nullptr};
  }
  return py::cast<MlirAttribute>(attr);
}

static void populatePTODialectSubmodule(const pybind11::module &m) {
  (void)m;
}

static void registerPTOEnums1(py::module_ &m) {
    // [保留 HEAD]: AddressSpace 枚举定义
    py::enum_<MlirPTOAddressSpace>(m, "AddressSpace")
    .value("Zero", MlirPTOAddressSpace_Zero)
    .value("GM",   MlirPTOAddressSpace_GM)
    .value("MAT",   MlirPTOAddressSpace_MAT)
    .value("LEFT",  MlirPTOAddressSpace_LEFT)
    .value("RIGHT",  MlirPTOAddressSpace_RIGHT)
    .value("ACC",  MlirPTOAddressSpace_ACC)
    .value("VEC",   MlirPTOAddressSpace_VEC)
    .value("BIAS",   MlirPTOAddressSpace_BIAS)
    .value("SCALING", MlirPTOAddressSpace_SCALING)
    .export_values();
    py::enum_<MlirPTOFenceScope>(m, "FenceScope")
    .value("LocalMemory", MlirPTOFenceScope_LocalMemory)
    .value("GM", MlirPTOFenceScope_GM)
    .value("All", MlirPTOFenceScope_All)
    .export_values();
    py::enum_<MlirPTOLoadCachePolicy>(m, "LoadCachePolicy")
    .value("Default", MlirPTOLoadCachePolicy_Default)
    .value("L2Bypass", MlirPTOLoadCachePolicy_L2Bypass);
    py::enum_<MlirPTOBLayout>(m, "BLayout")
    .value("RowMajor", MlirPTOBLayout_RowMajor)
    .value("ColMajor", MlirPTOBLayout_ColMajor);

    py::enum_<MlirPTOSLayout>(m, "SLayout")
    .value("NoneBox", MlirPTOSLayout_NoneBox)
    .value("RowMajor", MlirPTOSLayout_RowMajor)
    .value("ColMajor", MlirPTOSLayout_ColMajor);

    py::enum_<MlirPTOPadValue>(m, "PadValue")
    .value("Null", MlirPTOPadValue_Null)
    .value("Zero", MlirPTOPadValue_Zero)
    .value("Max", MlirPTOPadValue_Max)
    .value("Min", MlirPTOPadValue_Min);

    py::enum_<MlirPTOCompactMode>(m, "CompactMode")
    .value("Null", MlirPTOCompactMode_Null)
    .value("Normal", MlirPTOCompactMode_Normal)
    .value("RowPlusOne", MlirPTOCompactMode_RowPlusOne);

    py::enum_<MlirPTORoundMode>(m, "RoundMode")
    .value("NONE", MlirPTORoundMode_NONE)
    .value("RINT", MlirPTORoundMode_RINT)
    .value("ROUND", MlirPTORoundMode_ROUND)
    .value("FLOOR", MlirPTORoundMode_FLOOR)
    .value("CEIL", MlirPTORoundMode_CEIL)
    .value("TRUNC", MlirPTORoundMode_TRUNC)
    .value("ODD", MlirPTORoundMode_ODD)
    .value("CAST_RINT", MlirPTORoundMode_CAST_RINT);

}

static void registerPTOEnums2(py::module_ &m) {
    py::enum_<MlirPTODivPrecision>(m, "DivPrecision")
    .value("Default", MlirPTODivPrecision_Default)
    .value("HighPrecision", MlirPTODivPrecision_HighPrecision);

    py::enum_<MlirPTOExpPrecision>(m, "ExpPrecision")
    .value("Default", MlirPTOExpPrecision_Default)
    .value("HighPrecision", MlirPTOExpPrecision_HighPrecision);

    py::enum_<MlirPTOLogPrecision>(m, "LogPrecision")
    .value("Default", MlirPTOLogPrecision_Default)
    .value("HighPrecision", MlirPTOLogPrecision_HighPrecision);

    py::enum_<MlirPTORecipPrecision>(m, "RecipPrecision")
    .value("Default", MlirPTORecipPrecision_Default)
    .value("HighPrecision", MlirPTORecipPrecision_HighPrecision);

    py::enum_<MlirPTORemPrecision>(m, "RemPrecision")
    .value("Default", MlirPTORemPrecision_Default)
    .value("HighPrecision", MlirPTORemPrecision_HighPrecision);

    py::enum_<MlirPTORsqrtPrecision>(m, "RsqrtPrecision")
    .value("Default", MlirPTORsqrtPrecision_Default)
    .value("HighPrecision", MlirPTORsqrtPrecision_HighPrecision);

    py::enum_<MlirPTOSqrtPrecision>(m, "SqrtPrecision")
    .value("Default", MlirPTOSqrtPrecision_Default)
    .value("HighPrecision", MlirPTOSqrtPrecision_HighPrecision);

    py::enum_<MlirPTOFmodPrecision>(m, "FmodPrecision")
    .value("Default", MlirPTOFmodPrecision_Default)
    .value("HighPrecision", MlirPTOFmodPrecision_HighPrecision);

    py::enum_<MlirPTOSaturationMode>(m, "SaturationMode")
    .value("ON", MlirPTOSaturationMode_ON)
    .value("OFF", MlirPTOSaturationMode_OFF);

    py::enum_<MlirPTOCmpMode>(m, "CmpMode")
      .value("EQ", MlirPTOCmpMode_EQ)
      .value("NE", MlirPTOCmpMode_NE)
      .value("LT", MlirPTOCmpMode_LT)
      .value("LE", MlirPTOCmpMode_LE)
      .value("GT", MlirPTOCmpMode_GT)
      .value("GE", MlirPTOCmpMode_GE)
      .export_values();

}

static void registerPTOEnums3(py::module_ &m) {
    py::enum_<MlirPTOPIPE>(m, "PIPE")
      .value("PIPE_S", MlirPTOPIPE_PIPE_S)
      .value("PIPE_V", MlirPTOPIPE_PIPE_V)
      .value("PIPE_M", MlirPTOPIPE_PIPE_M)
      .value("PIPE_MTE1", MlirPTOPIPE_PIPE_MTE1)
      .value("PIPE_MTE2", MlirPTOPIPE_PIPE_MTE2)
      .value("PIPE_MTE3", MlirPTOPIPE_PIPE_MTE3)
      .value("PIPE_ALL", MlirPTOPIPE_PIPE_ALL)
      .value("PIPE_MTE4", MlirPTOPIPE_PIPE_MTE4)
      .value("PIPE_MTE5", MlirPTOPIPE_PIPE_MTE5)
      .value("PIPE_V2", MlirPTOPIPE_PIPE_V2)
      .value("PIPE_FIX", MlirPTOPIPE_PIPE_FIX)
      .value("VIRTUAL_PIPE_MTE2_L1A", MlirPTOPIPE_VIRTUAL_PIPE_MTE2_L1A)
      .value("VIRTUAL_PIPE_MTE2_L1B", MlirPTOPIPE_VIRTUAL_PIPE_MTE2_L1B)
      .value("PIPE_NUM", MlirPTOPIPE_PIPE_NUM)
      .value("PIPE_UNASSIGNED", MlirPTOPIPE_PIPE_UNASSIGNED);

    py::enum_<MlirPTOLayout>(m, "Layout")
      .value("ND", MlirPTOLayout_ND)
      .value("DN", MlirPTOLayout_DN)
      .value("NZ", MlirPTOLayout_NZ)
      .value("MX_A_ZZ", MlirPTOLayout_MX_A_ZZ)
      .value("MX_B_NN", MlirPTOLayout_MX_B_NN);

    py::enum_<MlirPTOAccToVecMode>(m, "AccToVecMode")
      .value("SingleModeVec0", MlirPTOAccToVecMode_SingleModeVec0)
      .value("SingleModeVec1", MlirPTOAccToVecMode_SingleModeVec1)
      .value("DualModeSplitM", MlirPTOAccToVecMode_DualModeSplitM)
      .value("DualModeSplitN", MlirPTOAccToVecMode_DualModeSplitN)
      .export_values();

    py::enum_<MlirPTOTInsertMode>(m, "TInsertMode")
      .value("SPLIT2", MlirPTOTInsertMode_SPLIT2)
      .value("SPLIT4", MlirPTOTInsertMode_SPLIT4)
      .export_values();

    py::enum_<MlirPTOReluPreMode>(m, "ReluPreMode")
      .value("NoRelu", MlirPTOReluPreMode_NoRelu)
      .value("NormalRelu", MlirPTOReluPreMode_NormalRelu)
      .export_values();

    py::enum_<MlirPTOAtomicType>(m, "AtomicType")
      .value("AtomicNone", MlirPTOAtomicType_AtomicNone)
      .value("AtomicAdd", MlirPTOAtomicType_AtomicAdd)
      .export_values();

}

static void registerPTOEnums4(py::module_ &m) {
    py::enum_<MlirPTONotifyOp>(m, "NotifyOp")
      .value("AtomicAdd", MlirPTONotifyOp_AtomicAdd)
      .value("Set", MlirPTONotifyOp_Set)
      .export_values();

    py::enum_<MlirPTOWaitCmp>(m, "WaitCmp")
      .value("EQ", MlirPTOWaitCmp_EQ)
      .value("NE", MlirPTOWaitCmp_NE)
      .value("GT", MlirPTOWaitCmp_GT)
      .value("GE", MlirPTOWaitCmp_GE)
      .value("LT", MlirPTOWaitCmp_LT)
      .value("LE", MlirPTOWaitCmp_LE)
      .export_values();

    py::enum_<MlirPTOReduceOp>(m, "ReduceOp")
      .value("Sum", MlirPTOReduceOp_Sum)
      .value("Max", MlirPTOReduceOp_Max)
      .value("Min", MlirPTOReduceOp_Min)
      .export_values();

    py::enum_<MlirPTOSyncOpType>(m, "SyncOpType")
      .value("TLOAD", MlirPTOSyncOpType_TLOAD)
      .value("TSTORE_ACC", MlirPTOSyncOpType_TSTORE_ACC)
      .value("TSTORE_VEC", MlirPTOSyncOpType_TSTORE_VEC)
      .value("TMOV_M2L", MlirPTOSyncOpType_TMOV_M2L)
      .value("TMOV_M2S", MlirPTOSyncOpType_TMOV_M2S)
      .value("TMOV_M2B", MlirPTOSyncOpType_TMOV_M2B)
      .value("TMOV_M2V", MlirPTOSyncOpType_TMOV_M2V)
      .value("TMOV_V2M", MlirPTOSyncOpType_TMOV_V2M)
      .value("TMATMUL", MlirPTOSyncOpType_TMATMUL)
      .value("TVEC", MlirPTOSyncOpType_TVEC)
      .value("TVECWAIT_EVENT", MlirPTOSyncOpType_TVECWAIT_EVENT)
      .export_values();

}

static void registerPTOEnums5(py::module_ &m) {
    py::enum_<MlirPTOEVENT>(m, "EVENT")
      .value("EVENT_ID0", MlirPTOEVENT_EVENT_ID0)
      .value("EVENT_ID1", MlirPTOEVENT_EVENT_ID1)
      .value("EVENT_ID2", MlirPTOEVENT_EVENT_ID2)
      .value("EVENT_ID3", MlirPTOEVENT_EVENT_ID3)
      .value("EVENT_ID4", MlirPTOEVENT_EVENT_ID4)
      .value("EVENT_ID5", MlirPTOEVENT_EVENT_ID5)
      .value("EVENT_ID6", MlirPTOEVENT_EVENT_ID6)
      .value("EVENT_ID7", MlirPTOEVENT_EVENT_ID7)
      .export_values();

    py::enum_<MlirPTOMaskPattern>(m, "MaskPattern")
      .value("P0101", MlirPTOMaskPattern_P0101)
      .value("P1010", MlirPTOMaskPattern_P1010)
      .value("P0001", MlirPTOMaskPattern_P0001)
      .value("P0010", MlirPTOMaskPattern_P0010)
      .value("P0100", MlirPTOMaskPattern_P0100)
      .value("P1000", MlirPTOMaskPattern_P1000)
      .value("P1111", MlirPTOMaskPattern_P1111)
      .export_values();
}

static void registerPTOSimpleAttrs1(py::module_ &m) {
    mlir_attribute_subclass(m, "BLayoutAttr",
                        [](MlirAttribute a) -> bool {
                          return mlirPTOAttrIsABLayoutAttr(a);
                        })
    .def_classmethod(
        "get",
        [](py::object cls, MlirPTOBLayout value, MlirContext ctx) -> py::object {
          MlirAttribute a = mlirPTOBLayoutAttrGet(ctx, static_cast<int32_t>(value));
          if (mlirAttributeIsNull(a)) {
            return py::none();
          }
          return cls(a);
        },
        py::arg("cls"), py::arg("value"), py::arg("context") = py::none());

    mlir_attribute_subclass(m, "SLayoutAttr",
                            [](MlirAttribute a) -> bool {
                            return mlirPTOAttrIsASLayoutAttr(a);
                            })
        .def_classmethod(
            "get",
            [](py::object cls, MlirPTOSLayout value, MlirContext ctx) -> py::object {
            MlirAttribute a = mlirPTOSLayoutAttrGet(ctx, static_cast<int32_t>(value));
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls(a);
            },
            py::arg("cls"), py::arg("value"), py::arg("context") = py::none());

    mlir_attribute_subclass(m, "PadValueAttr",
                            [](MlirAttribute a) -> bool {
                            return mlirPTOAttrIsAPadValueAttr(a);
                            })
        .def_classmethod(
            "get",
            [](py::object cls, MlirPTOPadValue value, MlirContext ctx) -> py::object {
            MlirAttribute a = mlirPTOPadValueAttrGet(ctx, static_cast<int32_t>(value));
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls(a);
            },
            py::arg("cls"), py::arg("value"), py::arg("context") = py::none());

}

static void registerPTOSimpleAttrs2(py::module_ &m) {
    mlir_attribute_subclass(m, "CompactModeAttr",
                            [](MlirAttribute a) -> bool {
                            return mlirPTOAttrIsACompactModeAttr(a);
                            })
        .def_classmethod(
            "get",
            [](py::object cls, MlirPTOCompactMode value, MlirContext ctx) -> py::object {
            MlirAttribute a = mlirPTOCompactModeAttrGet(ctx, static_cast<int32_t>(value));
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls(a);
            },
            py::arg("cls"), py::arg("value"), py::arg("context") = py::none());

    mlir_attribute_subclass(m, "AccToVecModeAttr",
                            [](MlirAttribute a) -> bool {
                            return mlirPTOAttrIsAAccToVecModeAttr(a);
                            })
        .def_classmethod(
            "get",
            [](py::object cls, MlirPTOAccToVecMode value, MlirContext ctx) -> py::object {
            MlirAttribute a = mlirPTOAccToVecModeAttrGet(ctx, static_cast<int32_t>(value));
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls(a);
            },
            py::arg("cls"), py::arg("value"), py::arg("context") = py::none());

    mlir_attribute_subclass(m, "TInsertModeAttr",
                            [](MlirAttribute a) -> bool {
                            return mlirPTOAttrIsATInsertModeAttr(a);
                            })
        .def_classmethod(
            "get",
            [](py::object cls, MlirPTOTInsertMode value, MlirContext ctx) -> py::object {
            MlirAttribute a = mlirPTOTInsertModeAttrGet(ctx, static_cast<int32_t>(value));
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls(a);
            },
            py::arg("cls"), py::arg("value"), py::arg("context") = py::none());

}

static void registerPTOSimpleAttrs3(py::module_ &m) {
    mlir_attribute_subclass(m, "ReluPreModeAttr",
                            [](MlirAttribute a) -> bool {
                            return mlirPTOAttrIsAReluPreModeAttr(a);
                            })
        .def_classmethod(
            "get",
            [](py::object cls, MlirPTOReluPreMode value, MlirContext ctx) -> py::object {
            MlirAttribute a = mlirPTOReluPreModeAttrGet(ctx, static_cast<int32_t>(value));
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls(a);
            },
            py::arg("cls"), py::arg("value"), py::arg("context") = py::none());

    mlir_attribute_subclass(m, "AtomicTypeAttr",
                            [](MlirAttribute a) -> bool {
                            return mlirPTOAttrIsAAtomicTypeAttr(a);
                            })
        .def_classmethod(
            "get",
            [](py::object cls, MlirPTOAtomicType value, MlirContext ctx) -> py::object {
            MlirAttribute a = mlirPTOAtomicTypeAttrGet(ctx, static_cast<int32_t>(value));
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls(a);
            },
            py::arg("cls"), py::arg("value"), py::arg("context") = py::none());

    mlir_attribute_subclass(m, "NotifyOpAttr",
                            [](MlirAttribute a) -> bool {
                            return mlirPTOAttrIsANotifyOpAttr(a);
                            })
        .def_classmethod(
            "get",
            [](py::object cls, MlirPTONotifyOp value, MlirContext ctx) -> py::object {
            MlirAttribute a = mlirPTONotifyOpAttrGet(ctx, static_cast<int32_t>(value));
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls(a);
            },
            py::arg("cls"), py::arg("value"), py::arg("context") = py::none());

}

static void registerPTOSimpleAttrs4(py::module_ &m) {
    mlir_attribute_subclass(m, "WaitCmpAttr",
                            [](MlirAttribute a) -> bool {
                            return mlirPTOAttrIsAWaitCmpAttr(a);
                            })
        .def_classmethod(
            "get",
            [](py::object cls, MlirPTOWaitCmp value, MlirContext ctx) -> py::object {
            MlirAttribute a = mlirPTOWaitCmpAttrGet(ctx, static_cast<int32_t>(value));
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls(a);
            },
            py::arg("cls"), py::arg("value"), py::arg("context") = py::none());

    mlir_attribute_subclass(m, "ReduceOpAttr",
                            [](MlirAttribute a) -> bool {
                            return mlirPTOAttrIsAReduceOpAttr(a);
                            })
        .def_classmethod(
            "get",
            [](py::object cls, MlirPTOReduceOp value, MlirContext ctx) -> py::object {
            MlirAttribute a = mlirPTOReduceOpAttrGet(ctx, static_cast<int32_t>(value));
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls(a);
            },
            py::arg("cls"), py::arg("value"), py::arg("context") = py::none());
}

static void registerPTOValueAttrs1(py::module_ &m) {
    // [保留 HEAD]: AddressSpaceAttr 定义
    mlir_attribute_subclass(
        m, "AddressSpaceAttr",
        [](MlirAttribute a) { return mlirPTOAttrIsAAddressSpaceAttr(a); })
    .def_classmethod(
        "get",
        [](py::object cls, py::object value, MlirContext context) -> py::object {
        // 支持传 enum 或 int
        int32_t v = 0;
        if (py::isinstance<py::int_>(value)) {
            v = py::cast<int32_t>(value);
        } else {
            // enum: pto.AddressSpace.UB -> 转成 int
            v = py::cast<int32_t>(value.attr("value").cast<py::int_>());
        }
        MlirAttribute a = mlirPTOAddressSpaceAttrGet(context, v);
        return cls.attr("__call__")(a);
        },
        py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
    .def_property_readonly(
        "value",
        [](MlirAttribute self) -> int32_t {
        return mlirPTOAddressSpaceAttrGetValue(self);
        });

    mlir_attribute_subclass(
        m, "FenceScopeAttr",
        [](MlirAttribute a) { return mlirPTOAttrIsAFenceScopeAttr(a); })
    .def_classmethod(
        "get",
        [](py::object cls, py::object value, MlirContext context) -> py::object {
        int32_t v = 0;
        if (py::isinstance<py::int_>(value)) {
            v = py::cast<int32_t>(value);
        } else {
            v = py::cast<int32_t>(value.attr("value").cast<py::int_>());
        }
        MlirAttribute a = mlirPTOFenceScopeAttrGet(context, v);
        if (mlirAttributeIsNull(a)) {
          return py::none();
        }
        return cls.attr("__call__")(a);
        },
        py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
    .def_property_readonly(
        "value",
        [](MlirAttribute self) -> int32_t {
        return mlirPTOFenceScopeAttrGetValue(self);
        });

}

static void registerPTOValueAttrs2(py::module_ &m) {
    bindPTOEnumAttr(m, "LoadCachePolicyAttr", "LoadCachePolicy",
                    mlirPTOAttrIsALoadCachePolicyAttr,
                    mlirPTOLoadCachePolicyAttrGet,
                    mlirPTOLoadCachePolicyAttrGetValue);

    mlir_attribute_subclass(
        m, "RoundModeAttr",
        [](MlirAttribute a) { return mlirPTOAttrIsARoundModeAttr(a); })
     .def_classmethod(
         "get",
        [](py::object cls, py::object value, MlirContext ctx) -> py::object {
        int32_t v = 0;
        if (py::isinstance<py::int_>(value)) {
            v = value.cast<int32_t>();
        } else if (py::hasattr(value, "value")) {
            // 通用：py::enum_ 通常有 .value
            v = value.attr("value").cast<int32_t>();
        } else {
            throw std::runtime_error("RoundModeAttr.get expects int or RoundMode enum");
        }

        MlirAttribute a = mlirPTORoundModeAttrGet(ctx, v);
        if (mlirAttributeIsNull(a)) {
          return py::none();
        }
        return cls.attr("__call__")(a);
         },
        py::arg("cls"), py::arg("value"), py::arg("context") = py::none())

    .def_property_readonly(
        "value",
        [](MlirAttribute self) -> int32_t {
        return mlirPTORoundModeAttrGetValue(self);
        });

}

static void registerPTOValueAttrs3(py::module_ &m) {
    bindPTOEnumAttr(m, "DivPrecisionAttr", "DivPrecision",
                    mlirPTOAttrIsADivPrecisionAttr,
                    mlirPTODivPrecisionAttrGet,
                    mlirPTODivPrecisionAttrGetValue);
    bindPTOEnumAttr(m, "ExpPrecisionAttr", "ExpPrecision",
                    mlirPTOAttrIsAExpPrecisionAttr,
                    mlirPTOExpPrecisionAttrGet,
                    mlirPTOExpPrecisionAttrGetValue);
    bindPTOEnumAttr(m, "LogPrecisionAttr", "LogPrecision",
                    mlirPTOAttrIsALogPrecisionAttr,
                    mlirPTOLogPrecisionAttrGet,
                    mlirPTOLogPrecisionAttrGetValue);
    bindPTOEnumAttr(m, "RecipPrecisionAttr", "RecipPrecision",
                    mlirPTOAttrIsARecipPrecisionAttr,
                    mlirPTORecipPrecisionAttrGet,
                    mlirPTORecipPrecisionAttrGetValue);
    bindPTOEnumAttr(m, "RemPrecisionAttr", "RemPrecision",
                    mlirPTOAttrIsARemPrecisionAttr,
                    mlirPTORemPrecisionAttrGet,
                    mlirPTORemPrecisionAttrGetValue);
    bindPTOEnumAttr(m, "RsqrtPrecisionAttr", "RsqrtPrecision",
                    mlirPTOAttrIsARsqrtPrecisionAttr,
                    mlirPTORsqrtPrecisionAttrGet,
                    mlirPTORsqrtPrecisionAttrGetValue);
    bindPTOEnumAttr(m, "SqrtPrecisionAttr", "SqrtPrecision",
                    mlirPTOAttrIsASqrtPrecisionAttr,
                    mlirPTOSqrtPrecisionAttrGet,
                    mlirPTOSqrtPrecisionAttrGetValue);
    bindPTOEnumAttr(m, "FmodPrecisionAttr", "FmodPrecision",
                    mlirPTOAttrIsAFmodPrecisionAttr,
                    mlirPTOFmodPrecisionAttrGet,
                    mlirPTOFmodPrecisionAttrGetValue);

}

static void registerPTOValueAttrs4(py::module_ &m) {
    mlir_attribute_subclass(
        m, "SaturationModeAttr",
        [](MlirAttribute a) { return mlirPTOAttrIsASaturationModeAttr(a); })
     .def_classmethod(
         "get",
        [](py::object cls, py::object value, MlirContext ctx) -> py::object {
        int32_t v = 0;
        if (py::isinstance<py::int_>(value)) {
            v = value.cast<int32_t>();
        } else if (py::hasattr(value, "value")) {
            v = value.attr("value").cast<int32_t>();
        } else {
            throw std::runtime_error("SaturationModeAttr.get expects int or SaturationMode enum");
        }

        MlirAttribute a = mlirPTOSaturationModeAttrGet(ctx, v);
        if (mlirAttributeIsNull(a)) {
          return py::none();
        }
        return cls.attr("__call__")(a);
         },
        py::arg("cls"), py::arg("value"), py::arg("context") = py::none())

    .def_property_readonly(
        "value",
        [](MlirAttribute self) -> int32_t {
        return mlirPTOSaturationModeAttrGetValue(self);
        });

}

static void registerPTOValueAttrs5(py::module_ &m) {
    mlir_attribute_subclass(
        m, "PipeAttr",
        [](MlirAttribute a) { return mlirPTOAttrIsAPipeAttr(a); })
      .def_classmethod(
          "get",
          [](py::object cls, py::object value, MlirContext ctx) -> py::object {
            int32_t v = 0;
            if (py::isinstance<py::int_>(value)) {
              v = value.cast<int32_t>();
            } else if (py::hasattr(value, "value")) {
              v = value.attr("value").cast<int32_t>();
            } else {
              throw std::runtime_error("PipeAttr.get expects int or PIPE enum");
            }
            MlirAttribute a = mlirPTOPipeAttrGet(ctx, v);
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls.attr("__call__")(a);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
      .def_property_readonly(
          "value",
          [](MlirAttribute self) -> int32_t {
            return mlirPTOPipeAttrGetValue(self);
          });

}

static void registerPTOValueAttrs6(py::module_ &m) {
    mlir_attribute_subclass(
        m, "LayoutAttr",
        [](MlirAttribute a) { return mlirPTOAttrIsALayoutAttr(a); })
      .def_classmethod(
          "get",
          [](py::object cls, py::object value, MlirContext ctx) -> py::object {
            int32_t v = 0;
            if (py::isinstance<py::int_>(value)) {
              v = value.cast<int32_t>();
            } else if (py::hasattr(value, "value")) {
              v = value.attr("value").cast<int32_t>();
            } else {
              throw std::runtime_error("LayoutAttr.get expects int or Layout enum");
            }
            MlirAttribute a = mlirPTOLayoutAttrGet(ctx, v);
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls.attr("__call__")(a);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
      .def_property_readonly(
          "value",
          [](MlirAttribute self) -> int32_t {
            return mlirPTOLayoutAttrGetValue(self);
          });

    mlir_attribute_subclass(m, "CmpModeAttr", mlirAttributeIsAPTOCmpModeAttr)
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctx, MlirPTOCmpMode value) {
            return cls(mlirPTOCmpModeAttrGet(ctx, value));
          },
          "cls"_a, "context"_a, "value"_a)
      .def_property_readonly(
          "value",
          [](MlirAttribute self) {
            return mlirPTOCmpModeAttrGetValue(self);
          });

}

static void registerPTOValueAttrs7(py::module_ &m) {
    mlir_attribute_subclass(
        m, "SyncOpTypeAttr",
        [](MlirAttribute a) { return mlirPTOAttrIsASyncOpTypeAttr(a); })
      .def_classmethod(
          "get",
          [](py::object cls, py::object value, MlirContext ctx) -> py::object {
            int32_t v = 0;
            if (py::isinstance<py::int_>(value)) {
              v = py::cast<int32_t>(value);
            } else if (py::hasattr(value, "value")) {
              v = value.attr("value").cast<int32_t>();
            } else {
              throw std::runtime_error("SyncOpTypeAttr.get expects int or SyncOpType enum");
            }
            MlirAttribute a = mlirPTOSyncOpTypeAttrGet(ctx, v);
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls.attr("__call__")(a);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
      .def_property_readonly(
          "value",
          [](MlirAttribute self) -> int32_t {
            return mlirPTOSyncOpTypeAttrGetValue(self);
          });

}

static void registerPTOValueAttrs8(py::module_ &m) {
    mlir_attribute_subclass(
        m, "EventAttr",
        [](MlirAttribute a) { return mlirPTOAttrIsAEventAttr(a); })
      .def_classmethod(
          "get",
          [](py::object cls, py::object value, MlirContext ctx) -> py::object {
            int32_t v = 0;
            if (py::isinstance<py::int_>(value)) {
              v = py::cast<int32_t>(value);
            } else if (py::hasattr(value, "value")) {
              v = value.attr("value").cast<int32_t>();
            } else {
              throw std::runtime_error("EventAttr.get expects int or EVENT enum");
            }
            MlirAttribute a = mlirPTOEventAttrGet(ctx, v);
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls.attr("__call__")(a);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
      .def_property_readonly(
          "value",
          [](MlirAttribute self) -> int32_t {
            return mlirPTOEventAttrGetValue(self);
          });

    py::enum_<MlirPTOCoalesce>(m, "Coalesce")
      .value("Elem", MlirPTOCoalesce_Elem)
      .value("Row", MlirPTOCoalesce_Row)
      .export_values();

}

static void registerPTOValueAttrs9(py::module_ &m) {
    mlir_attribute_subclass(
        m, "CoalesceAttr",
        [](MlirAttribute a) { return mlirPTOAttrIsACoalesceAttr(a); })
      .def_classmethod(
          "get",
          [](py::object cls, py::object value, MlirContext ctx) -> py::object {
            int32_t v = 0;
            if (py::isinstance<py::int_>(value)) {
              v = py::cast<int32_t>(value);
            } else if (py::hasattr(value, "value")) {
              v = value.attr("value").cast<int32_t>();
            } else {
              throw std::runtime_error("CoalesceAttr.get expects int or Coalesce enum");
            }
            MlirAttribute a =
                mlirPTOCoalesceAttrGet(ctx, static_cast<MlirPTOCoalesce>(v));
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls.attr("__call__")(a);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
      .def_property_readonly(
          "value",
          [](MlirAttribute self) -> int32_t {
            return mlirPTOCoalesceAttrGetValue(self);
          });

    py::enum_<MlirPTOQuantType>(m, "QuantType")
      .value("INT8_SYM",  MlirPTOQuantType_INT8_SYM)
      .value("INT8_ASYM", MlirPTOQuantType_INT8_ASYM)
      .value("MXFP8",     MlirPTOQuantType_MXFP8)
      .value("MXFP4_E2M1", MlirPTOQuantType_MXFP4_E2M1)
      .export_values();

}

static void registerPTOValueAttrs10(py::module_ &m) {
    py::enum_<MlirPTOQuantScaleAlg>(m, "QuantScaleAlg")
      .value("OCP", MlirPTOQuantScaleAlg_OCP)
      .value("NV", MlirPTOQuantScaleAlg_NV)
      .export_values();

    py::enum_<MlirPTOMxGroupAxis>(m, "MxGroupAxis")
      .value("Axis0", MlirPTOMxGroupAxis_Axis0)
      .value("Axis1", MlirPTOMxGroupAxis_Axis1)
      .export_values();

    py::enum_<MlirPTOVecStoreMode>(m, "VecStoreMode")
      .value("ND", MlirPTOVecStoreMode_ND)
      .value("NZ", MlirPTOVecStoreMode_NZ)
      .export_values();

    mlir_attribute_subclass(
        m, "QuantTypeAttr",
        [](MlirAttribute a) { return mlirPTOAttrIsAQuantTypeAttr(a); })
      .def_classmethod(
          "get",
          [](py::object cls, py::object value, MlirContext ctx) -> py::object {
            int32_t v = 0;
            if (py::isinstance<py::int_>(value)) {
              v = py::cast<int32_t>(value);
            } else if (py::hasattr(value, "value")) {
              v = value.attr("value").cast<int32_t>();
            } else {
              throw std::runtime_error("QuantTypeAttr.get expects int or QuantType enum");
            }
            MlirAttribute a = mlirPTOQuantTypeAttrGet(ctx, v);
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls.attr("__call__")(a);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
      .def_property_readonly(
          "value",
          [](MlirAttribute self) -> int32_t {
            return mlirPTOQuantTypeAttrGetValue(self);
          });

}

static void registerPTOValueAttrs11(py::module_ &m) {
    mlir_attribute_subclass(
        m, "QuantScaleAlgAttr",
        [](MlirAttribute a) { return mlirPTOAttrIsAQuantScaleAlgAttr(a); })
      .def_classmethod(
          "get",
          [](py::object cls, py::object value, MlirContext ctx) -> py::object {
            int32_t v = 0;
            if (py::isinstance<py::int_>(value)) {
              v = py::cast<int32_t>(value);
            } else if (py::hasattr(value, "value")) {
              v = value.attr("value").cast<int32_t>();
            } else {
              throw std::runtime_error("QuantScaleAlgAttr.get expects int or QuantScaleAlg enum");
            }
            MlirAttribute a = mlirPTOQuantScaleAlgAttrGet(ctx, v);
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls.attr("__call__")(a);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
      .def_property_readonly(
          "value",
          [](MlirAttribute self) -> int32_t {
            return mlirPTOQuantScaleAlgAttrGetValue(self);
          });

}

static void registerPTOValueAttrs12(py::module_ &m) {
    mlir_attribute_subclass(
        m, "MxGroupAxisAttr",
        [](MlirAttribute a) { return mlirPTOAttrIsAMxGroupAxisAttr(a); })
      .def_classmethod(
          "get",
          [](py::object cls, py::object value, MlirContext ctx) -> py::object {
            int32_t v = 0;
            if (py::isinstance<py::int_>(value)) {
              v = py::cast<int32_t>(value);
            } else if (py::hasattr(value, "value")) {
              v = value.attr("value").cast<int32_t>();
            } else {
              throw std::runtime_error("MxGroupAxisAttr.get expects int or MxGroupAxis enum");
            }
            MlirAttribute a = mlirPTOMxGroupAxisAttrGet(ctx, v);
            if (mlirAttributeIsNull(a)) return py::none();
            return cls.attr("__call__")(a);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
      .def_property_readonly(
          "value",
          [](MlirAttribute self) -> int32_t {
            return mlirPTOMxGroupAxisAttrGetValue(self);
          });

}

static void registerPTOValueAttrs13(py::module_ &m) {
    mlir_attribute_subclass(
        m, "VecStoreModeAttr",
        [](MlirAttribute a) { return mlirPTOAttrIsAVecStoreModeAttr(a); })
      .def_classmethod(
          "get",
          [](py::object cls, py::object value, MlirContext ctx) -> py::object {
            int32_t v = 0;
            if (py::isinstance<py::int_>(value)) {
              v = py::cast<int32_t>(value);
            } else if (py::hasattr(value, "value")) {
              v = value.attr("value").cast<int32_t>();
            } else {
              throw std::runtime_error("VecStoreModeAttr.get expects int or VecStoreMode enum");
            }
            MlirAttribute a = mlirPTOVecStoreModeAttrGet(ctx, v);
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls.attr("__call__")(a);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
      .def_property_readonly(
          "value",
          [](MlirAttribute self) -> int32_t {
            return mlirPTOVecStoreModeAttrGetValue(self);
          });

}

static void addMaskPatternGetMethods(pure_subclass &c, py::module_ m) {
  c
      .def_classmethod(
          "get",
          [m](py::object cls, py::object value,
                                MlirContext ctx) -> py::object {
            MlirAttribute a{nullptr};
            if (py::isinstance(value, m.attr("MaskPattern"))) {
              auto v =
                  static_cast<MlirPTOMaskPattern>(value.attr("value").cast<int32_t>());
              a = mlirPTOMaskPatternAttrGetEnum(ctx, v);
            } else if (py::isinstance<py::int_>(value)) {
              int32_t v = py::cast<int32_t>(value);
              a = mlirPTOMaskPatternAttrGet(ctx, v);
              if (mlirAttributeIsNull(a)) {
                throw std::runtime_error(
                    "MaskPatternAttr.get(int, ...) only accepts unambiguous values {0,3,6,7}; "
                    "use MaskPattern enum for ISA values and get_legacy_raw(...) for historical raw encodings");
              }
            } else {
              throw std::runtime_error("MaskPatternAttr.get expects int or MaskPattern enum");
            }
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls.attr("__call__")(a);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
      .def_classmethod(
          "get_legacy_raw",
          [](py::object cls, int32_t value, MlirContext ctx) -> py::object {
            MlirAttribute a = mlirPTOMaskPatternAttrGetLegacyRaw(ctx, value);
            if (mlirAttributeIsNull(a)) {
              throw std::runtime_error(
                  "MaskPatternAttr.get_legacy_raw(...) only accepts historical raw values {0,3,4,5}");
            }
            return cls.attr("__call__")(a);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none());
}

static void registerMaskPatternAttr(py::module_ &m) {
  mlir_attribute_subclass maskPattern(
      m, "MaskPatternAttr",
      [](MlirAttribute a) -> bool { return mlirPTOAttrIsAMaskPatternAttr(a); });
  addMaskPatternGetMethods(maskPattern, m);
  maskPattern
      .def_property_readonly(
          "value",
          [](MlirAttribute self) -> MlirPTOMaskPattern {
            return mlirPTOMaskPatternAttrGetEnumValue(self);
          })
      .def_property_readonly(
          "int_value",
          [](MlirAttribute self) -> int32_t {
            return mlirPTOMaskPatternAttrGetValue(self);
          });;
}

static void registerPTOTypes1(py::module_ &m) {
    // --------------------------------------------------------------------------
    // !pto.ptr<elem>
    // --------------------------------------------------------------------------
    mlir_type_subclass(
        m, "PtrType",
        [](MlirType type) -> bool { return mlirPTOTypeIsAPtrType(type); })
        .def_classmethod(
            "get",
            [](py::object cls, MlirType elementType, py::object memorySpace,
               MlirContext context) -> py::object {
                MlirContext ctx = context;
                if (!ctx.ptr) {
                  ctx = mlirTypeGetContext(elementType);
                }
                MlirType t = {nullptr};
                if (memorySpace.is_none()) {
                  t = mlirPTOPtrTypeGet(ctx, elementType);
                } else {
                  MlirAttribute memorySpaceAttr =
                      py::cast<MlirAttribute>(memorySpace);
                  t = mlirPTOPtrTypeGetWithMemorySpace(ctx, elementType,
                                                       memorySpaceAttr);
                }
                return cls.attr("__call__")(t);
            },
            py::arg("cls"), py::arg("element_type"),
            py::arg("memory_space") = py::none(),
            py::arg("context") = py::none())
        .def_property_readonly(
            "element_type",
            [](MlirType self) -> MlirType {
                return mlirPTOPtrTypeGetElementType(self);
            })
        .def_property_readonly(
            "memory_space",
            [](MlirType self) -> MlirAttribute {
                return mlirPTOPtrTypeGetMemorySpace(self);
            });

}

static void registerPTOTypes2(py::module_ &m) {
    mlir_type_subclass(
        m, "VRegType",
        [](MlirType type) -> bool { return mlirPTOTypeIsAVRegType(type); })
        .def_classmethod(
            "get",
            [](py::object cls, int64_t elementCount, MlirType elementType,
               MlirContext context) -> py::object {
                context = inferContextFromElementType(context, elementType);
                MlirType t = mlirPTOVRegTypeGet(context, elementCount, elementType);
                return cls.attr("__call__")(t);
            },
            py::arg("cls"), py::arg("element_count"), py::arg("element_type"),
            py::arg("context") = py::none())
        .def_property_readonly(
            "element_count",
            [](MlirType self) -> int64_t {
                return mlirPTOVRegTypeGetElementCount(self);
            })
        .def_property_readonly(
            "element_type",
            [](MlirType self) -> MlirType {
                return mlirPTOVRegTypeGetElementType(self);
            });

    mlir_type_subclass(
        m, "MaskType",
        [](MlirType type) -> bool { return mlirPTOTypeIsAMaskType(type); })
        .def_classmethod(
            "get",
            [](py::object cls, std::string granularity, MlirContext context) -> py::object {
                MlirType t = mlirPTOMaskTypeGet(
                    context, mlirStringRefCreate(granularity.data(), granularity.size()));
                return cls.attr("__call__")(t);
            },
            py::arg("cls"), py::arg("granularity"),
            py::arg("context") = py::none())
        .def_property_readonly(
            "granularity",
            [](MlirType self) -> std::string {
                MlirStringRef s = mlirPTOMaskTypeGetGranularity(self);
                return std::string(s.data, s.length);
            });

}

static void registerPTOTypes3(py::module_ &m) {
    mlir_type_subclass(
        m, "VMIVRegType",
        [](MlirType type) -> bool {
            return mlirPTOTypeIsAVMIVRegType(type);
        })
        .def_classmethod(
            "get",
            [](py::object cls, int64_t elementCount, MlirType elementType,
               py::object layout, MlirContext context) -> py::object {
                context = inferContextFromElementType(context, elementType);
                MlirAttribute layoutAttr = optionalAttributeFromPy(layout);
                MlirType t = mlirPTOVMIVRegTypeGet(
                    context, elementCount, elementType, layoutAttr);
                return cls.attr("__call__")(t);
            },
            py::arg("cls"), py::arg("element_count"), py::arg("element_type"),
            py::arg("layout") = py::none(), py::arg("context") = py::none())
        .def_property_readonly(
            "element_count",
            [](MlirType self) -> int64_t {
                return mlirPTOVMIVRegTypeGetElementCount(self);
            })
        .def_property_readonly(
            "element_type",
            [](MlirType self) -> MlirType {
                return mlirPTOVMIVRegTypeGetElementType(self);
            })
        .def_property_readonly(
            "layout",
            [](MlirType self) -> py::object {
                MlirAttribute attr = mlirPTOVMIVRegTypeGetLayout(self);
                if (mlirAttributeIsNull(attr)) {
                  return py::none();
                }
                return py::cast(attr);
            });

}

static void registerPTOTypes4(py::module_ &m) {
    mlir_type_subclass(
        m, "VMIMaskType",
        [](MlirType type) -> bool {
            return mlirPTOTypeIsAVMIMaskType(type);
        })
        .def_classmethod(
            "get",
            [](py::object cls, int64_t elementCount, std::string granularity,
               py::object layout, MlirContext context) -> py::object {
                MlirAttribute layoutAttr = optionalAttributeFromPy(layout);
                MlirType t = mlirPTOVMIMaskTypeGet(
                    context, elementCount,
                    mlirStringRefCreate(granularity.data(), granularity.size()),
                    layoutAttr);
                return cls.attr("__call__")(t);
            },
            py::arg("cls"), py::arg("element_count"),
            py::arg("granularity") = "pred", py::arg("layout") = py::none(),
            py::arg("context") = py::none())
        .def_property_readonly(
            "element_count",
            [](MlirType self) -> int64_t {
                return mlirPTOVMIMaskTypeGetElementCount(self);
            })
        .def_property_readonly(
            "granularity",
            [](MlirType self) -> std::string {
                MlirStringRef s = mlirPTOVMIMaskTypeGetGranularity(self);
                return std::string(s.data, s.length);
            })
        .def_property_readonly(
            "layout",
            [](MlirType self) -> py::object {
                MlirAttribute attr = mlirPTOVMIMaskTypeGetLayout(self);
                if (mlirAttributeIsNull(attr)) {
                  return py::none();
                }
                return py::cast(attr);
            });
}

static void registerPTOTypes4b(py::module_ &m) {
    mlir_type_subclass(
        m, "AlignType",
        [](MlirType type) -> bool { return mlirPTOTypeIsAAlignType(type); })
        .def_classmethod(
            "get",
            [](py::object cls, MlirContext context) -> py::object {
                MlirType t = mlirPTOAlignTypeGet(context);
                return cls.attr("__call__")(t);
            },
            py::arg("cls"), py::arg("context") = py::none());

}

static void registerPTOTypes5(py::module_ &m) {
    mlir_type_subclass(
        m, "StructType",
        [](MlirType type) -> bool { return mlirPTOTypeIsAStructType(type); })
        .def_classmethod(
            "get",
            [](py::object cls, py::sequence fieldTypes, MlirContext context) -> py::object {
                std::vector<MlirType> fields;
                fields.reserve(fieldTypes.size());
                for (py::handle field : fieldTypes) {
                  fields.push_back(py::cast<MlirType>(field));
                }
                if (fields.empty()) {
                  throw py::value_error("StructType.get requires at least one field type");
                }
                if (!context.ptr) {
                  context = mlirTypeGetContext(fields.front());
                }
                MlirType t = mlirPTOStructTypeGet(
                    context, static_cast<intptr_t>(fields.size()), fields.data());
                if (mlirTypeIsNull(t)) {
                  throw py::value_error(
                      "StructType.get received invalid struct field types");
                }
                return cls.attr("__call__")(t);
            },
            py::arg("cls"), py::arg("field_types"), py::arg("context") = py::none())
        .def_property_readonly(
            "field_types",
            [](MlirType self) -> py::list {
                intptr_t n = mlirPTOStructTypeGetNumFieldTypes(self);
                py::list fields;
                for (intptr_t i = 0; i < n; ++i) {
                  fields.append(mlirPTOStructTypeGetFieldType(self, i));
                }
                return fields;
            });

    mlir_type_subclass(
        m, "AsyncSessionType",
        [](MlirType type) -> bool { return mlirPTOTypeIsAAsyncSessionType(type); })
        .def_classmethod(
            "get",
            [](py::object cls, MlirContext context) -> py::object {
                MlirType t = mlirPTOAsyncSessionTypeGet(context);
                return cls.attr("__call__")(t);
            },
            py::arg("cls"), py::arg("context") = py::none());

}

static void registerPTOTypes6(py::module_ &m) {
    mlir_type_subclass(
        m, "AsyncEventType",
        [](MlirType type) -> bool { return mlirPTOTypeIsAAsyncEventType(type); })
        .def_classmethod(
            "get",
            [](py::object cls, MlirContext context) -> py::object {
                MlirType t = mlirPTOAsyncEventTypeGet(context);
                return cls.attr("__call__")(t);
            },
            py::arg("cls"), py::arg("context") = py::none());

    mlir_type_subclass(
        m, "PrefetchAsyncContextType",
        [](MlirType type) -> bool {
            return mlirPTOTypeIsAPrefetchAsyncContextType(type);
        })
        .def_classmethod(
            "get",
            [](py::object cls, MlirContext context) -> py::object {
                MlirType t = mlirPTOPrefetchAsyncContextTypeGet(context);
                return cls.attr("__call__")(t);
            },
            py::arg("cls"), py::arg("context") = py::none());

    mlir_type_subclass(
        m, "HiF8Type",
        [](MlirType type) -> bool { return mlirPTOTypeIsAHiF8Type(type); })
        .def_classmethod(
            "get",
            [](py::object cls, MlirContext context) -> py::object {
                MlirType t = mlirPTOHiF8TypeGet(context);
                return cls.attr("__call__")(t);
            },
            py::arg("cls"), py::arg("context") = py::none());

    mlir_type_subclass(
        m, "F8E8M0Type",
        [](MlirType type) -> bool { return mlirPTOTypeIsAF8E8M0Type(type); })
        .def_classmethod(
            "get",
            [](py::object cls, MlirContext context) -> py::object {
                MlirType t = mlirPTOF8E8M0TypeGet(context);
                return cls.attr("__call__")(t);
            },
            py::arg("cls"), py::arg("context") = py::none());

}

static void registerPTOTypes7(py::module_ &m) {
    mlir_type_subclass(
        m, "HiF8x2Type",
        [](MlirType type) -> bool { return mlirPTOTypeIsAHiF8x2Type(type); })
        .def_classmethod(
            "get",
            [](py::object cls, MlirContext context) -> py::object {
                MlirType t = mlirPTOHiF8x2TypeGet(context);
                return cls.attr("__call__")(t);
            },
            py::arg("cls"), py::arg("context") = py::none());

    mlir_type_subclass(
        m, "BF16x2Type",
        [](MlirType type) -> bool { return mlirPTOTypeIsABF16x2Type(type); })
        .def_classmethod(
            "get",
            [](py::object cls, MlirContext context) -> py::object {
                MlirType t = mlirPTOBF16x2TypeGet(context);
                return cls.attr("__call__")(t);
            },
            py::arg("cls"), py::arg("context") = py::none());

    mlir_type_subclass(
        m, "F4E1M2x2Type",
        [](MlirType type) -> bool { return mlirPTOTypeIsAF4E1M2x2Type(type); })
        .def_classmethod(
            "get",
            [](py::object cls, MlirContext context) -> py::object {
                MlirType t = mlirPTOF4E1M2x2TypeGet(context);
                return cls.attr("__call__")(t);
            },
            py::arg("cls"), py::arg("context") = py::none());

    mlir_type_subclass(
        m, "F4E2M1x2Type",
        [](MlirType type) -> bool { return mlirPTOTypeIsAF4E2M1x2Type(type); })
        .def_classmethod(
            "get",
            [](py::object cls, MlirContext context) -> py::object {
                MlirType t = mlirPTOF4E2M1x2TypeGet(context);
                return cls.attr("__call__")(t);
            },
            py::arg("cls"), py::arg("context") = py::none());

    // --------------------------------------------------------------------------
    // !pto.tensor_view<shape x elem>
    // --------------------------------------------------------------------------
}

static void registerPTOTypes8(py::module_ &m) {
    mlir_type_subclass(
        m, "TensorViewType",
        [](MlirType type) -> bool { return mlirPTOTypeIsATensorViewType(type); })
        .def_classmethod(
            "get",
            [](py::object cls, py::object shape_or_rank, MlirType elementType, MlirContext context) -> py::object {
                std::vector<int64_t> shp = toShapeVectorOrDynamicRank(shape_or_rank);
                context = inferContextFromElementType(context, elementType);
                MlirType t = mlirPTOTensorViewTypeGet(
                    context, static_cast<intptr_t>(shp.size()), shp.data(), elementType);
                return cls.attr("__call__")(t);
            },
            py::arg("cls"), py::arg("shape_or_rank"), py::arg("element_type"),
            py::arg("context") = py::none())
        .def_property_readonly(
            "rank",
            [](MlirType self) -> intptr_t { return mlirPTOTensorViewTypeGetRank(self); })
        .def_property_readonly(
            "element_type",
            [](MlirType self) -> MlirType {
                return mlirPTOTensorViewTypeGetElementType(self);
            })
        .def_property_readonly(
            "shape",
            [](MlirType self) -> py::list {
                intptr_t n = 0;
                const int64_t *data = mlirPTOTensorViewTypeGetShape(self, &n);
                return shapeToPyList(data, n);
            });
        // --------------------------------------------------------------------------
    // !pto.tile_view<shape x elem>
    // --------------------------------------------------------------------------
}

static void registerPTOTypes9(py::module_ &m) {
    mlir_type_subclass(
        m, "PartitionTensorViewType",
        [](MlirType t) -> bool { return mlirPTOTypeIsAPartitionTensorViewType(t); })
    .def_classmethod(
        "get",
        [](py::object cls, py::object shape_or_rank, MlirType elementType, MlirContext context) -> py::object {
        std::vector<int64_t> shp = toShapeVectorOrDynamicRank(shape_or_rank);
        context = inferContextFromElementType(context, elementType);
        MlirType t = mlirPTOPartitionTensorViewTypeGet(context,
                                            static_cast<intptr_t>(shp.size()),
                                            shp.data(),
                                            elementType);
        return cls.attr("__call__")(t);
        },
        py::arg("cls"), py::arg("shape_or_rank"), py::arg("element_type"),
        py::arg("context") = py::none())
    .def_property_readonly(
        "rank",
        [](MlirType self) -> intptr_t { return mlirPTOPartitionTensorViewTypeGetRank(self); })
    .def_property_readonly(
        "element_type",
        [](MlirType self) -> MlirType { return mlirPTOPartitionTensorViewTypeGetElementType(self); })
    .def_property_readonly(
        "shape",
        [](MlirType self) -> py::list {
        intptr_t n = 0;
        const int64_t *data = mlirPTOPartitionTensorViewTypeGetShape(self, &n);
        return shapeToPyList(data, n);
        });

    // --------------------------------------------------------------------------
    // !pto.tile<shape x elem>
    // --------------------------------------------------------------------------
}

static void registerPTOTypes10(py::module_ &m) {
    mlir_type_subclass(
        m, "TileType",
        [](MlirType t) -> bool { return mlirPTOTypeIsATileType(t); })
    .def_classmethod(
        "get",
        [](py::object cls, py::sequence shape, MlirType elementType, MlirContext context) -> py::object {
        auto shp = toInt64Vector(shape);
        MlirType t = mlirPTOTileTypeGet(context,
                                        static_cast<intptr_t>(shp.size()),
                                        shp.data(),
                                        elementType);
        return cls.attr("__call__")(t);
        },
        py::arg("cls"), py::arg("shape"), py::arg("element_type"),
        py::arg("context") = py::none())
    .def_property_readonly(
        "rank",
        [](MlirType self) -> intptr_t { return mlirPTOTileTypeGetRank(self); })
    .def_property_readonly(
        "element_type",
        [](MlirType self) -> MlirType { return mlirPTOTileTypeGetElementType(self); })
    .def_property_readonly(
        "shape",
        [](MlirType self) -> py::list {
        intptr_t n = 0;
        const int64_t *data = mlirPTOTileTypeGetShape(self, &n);
        return shapeToPyList(data, n);
        });

}

static void addTileBufConfigGetMethod(pure_subclass &c) {
  c
        .def_classmethod(
            "get",
            [](py::object cls,
                MlirAttribute blayout,
                MlirAttribute slayout,
                int32_t s_fractal_size,
                MlirAttribute pad,
                MlirContext ctx,
                py::object compactModeObj) -> py::object {
                MlirType i32 = mlirIntegerTypeGet(ctx, 32);
                MlirAttribute sz = mlirIntegerAttrGet(i32, s_fractal_size);
                MlirAttribute compactMode = mlirPTOCompactModeAttrGet(
                    ctx, static_cast<int32_t>(MlirPTOCompactMode_Null));
                if (!compactModeObj.is_none()) {
                  if (py::isinstance<py::int_>(compactModeObj)) {
                    compactMode = mlirPTOCompactModeAttrGet(
                        ctx, compactModeObj.cast<int32_t>());
                  } else if (py::hasattr(compactModeObj, "value")) {
                    compactMode = mlirPTOCompactModeAttrGet(
                        ctx, compactModeObj.attr("value").cast<int32_t>());
                  } else {
                    compactMode = compactModeObj.cast<MlirAttribute>();
                  }
                }
                MlirAttribute a = mlirPTOTileBufConfigAttrGetWithCompactMode(
                    ctx, blayout, slayout, sz, pad, compactMode);
                if (mlirAttributeIsNull(a)) {
                  return py::none();
                }
                return cls(a);
            },
            py::arg("cls"),
            py::arg("blayout"),
            py::arg("slayout"),
            py::arg("s_fractal_size"),
            py::arg("pad"),
            py::arg("context") = py::none(),
            py::arg("compact_mode") = py::none());
}

static void registerTileBufConfigAttr(py::module_ &m) {
  mlir_attribute_subclass tileBufConfig(
      m, "TileBufConfigAttr",
      [](MlirAttribute a) -> bool {
        return mlirPTOAttrIsATileBufConfigAttr(a);
      });
  tileBufConfig
        .def_classmethod(
            "get_default",
            [](py::object cls, MlirContext ctx) -> py::object {
                MlirAttribute a = mlirPTOTileBufConfigAttrGetDefault(ctx);
                if (mlirAttributeIsNull(a)) {
                  return py::none();
                }
                return cls(a);
            },
            py::arg("cls"), py::arg("context") = py::none());
  addTileBufConfigGetMethod(tileBufConfig);
}

static void addTileBufTypeGetMethod(pure_subclass &c) {
  c
        .def_classmethod(
            "get",
            [](py::object cls, std::vector<int64_t> shape, MlirType elementType, MlirAttribute memorySpace,
               py::object validShapeObj, py::object configObj, MlirContext ctx) -> py::object {
              // 1) 计算 validShape（默认=shape）
              std::vector<int64_t> validShape = shape;

              if (!validShapeObj.is_none()) {
                // 支持 valid_shape 为 list[int] 或 list[Optional[int]]
                py::list lst = validShapeObj.cast<py::list>();
                if (static_cast<size_t>(lst.size()) != shape.size()) {
                  throw std::runtime_error("valid_shape rank must match shape rank");
                }
                validShape.resize(lst.size());
                for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(lst.size()); ++i) {
                  py::object e = lst[i];
                  if (e.is_none()) {
                    validShape[i] = -1; // None -> dynamic
                  } else {
                    validShape[i] = e.cast<int64_t>();
                  }
                }
              }

              // 2) 调 CAPI
              MlirType ty;
              if (!configObj.is_none()) {
                MlirAttribute cfg = configObj.cast<MlirAttribute>();
                ty = mlirPTOTileBufTypeGetWithValidShapeAndConfig(
                    ctx, static_cast<intptr_t>(shape.size()), shape.data(), elementType, memorySpace,
                    static_cast<intptr_t>(validShape.size()), validShape.data(), cfg);
              } else {
                ty = mlirPTOTileBufTypeGetWithValidShape(ctx, static_cast<intptr_t>(shape.size()), shape.data(),
                                                         elementType, memorySpace,
                                                         static_cast<intptr_t>(validShape.size()), validShape.data());
              }

              if (mlirTypeIsNull(ty)) {
                return py::none();
              }
              return cls(ty);
            },
            py::arg("cls"), py::arg("shape"), py::arg("element_type"), py::arg("memory_space"),
            py::arg("valid_shape") = py::none(), py::arg("config") = py::none(), py::arg("context") = py::none());
}

static void addTileBufTypeAccessors1(pure_subclass &c, py::module_ m) {
  c
        .def_classmethod(
            "upcast_type",
            [](py::object cls, MlirType t) -> py::object {
              if (mlirPTOTypeIsATileBufType(t)) {
                return cls(t);
              }
              return py::none();
            },
            py::arg("cls"), py::arg("type"))
        .def_property_readonly("rank",
                               [](MlirType self) -> intptr_t {
                                 return mlirPTOTileBufTypeGetRank(self);
                               })
        .def_property_readonly(
            "element_type",
            [](MlirType self) -> MlirType { return mlirPTOTileBufTypeGetElementType(self); })
        .def_property_readonly("memory_space",
                               [m](MlirType self) -> py::object {
                                 MlirAttribute attr = mlirPTOTileBufTypeGetMemorySpace(self);
                                 return wrapAttributeAs(m, "AddressSpaceAttr", attr);
                               })
        .def_property_readonly("shape",
                               [](MlirType self) -> py::list {
                                 intptr_t n = 0;
                                 const int64_t *d = mlirPTOTileBufTypeGetShape(self, &n);
                                 return shapeToPyList(d, n);
                               })
        .def_property_readonly("valid_shape",
                               [](MlirType self) -> py::list {
                                 intptr_t n = 0;
                                 const int64_t *d = mlirPTOTileBufTypeGetValidShape(self, &n);
                                 return shapeToPyList(d, n);
                               });
}

static void addTileBufTypeAccessors2(pure_subclass &c, py::module_ m) {
  c
        .def_property_readonly("blayout_attr",
                               [m](MlirType self) -> py::object {
                                 MlirAttribute attr = mlirPTOTileBufTypeGetBLayoutAttr(self);
                                 return wrapAttributeAs(m, "BLayoutAttr", attr);
                               })
        .def_property_readonly("slayout_attr",
                               [m](MlirType self) -> py::object {
                                 MlirAttribute attr = mlirPTOTileBufTypeGetSLayoutAttr(self);
                                 return wrapAttributeAs(m, "SLayoutAttr", attr);
                               })
        .def_property_readonly(
            "blayout_value",
            [](MlirType self) -> int32_t { return mlirPTOTileBufTypeGetBLayoutValue(self); })
        .def_property_readonly(
            "slayout_value",
            [](MlirType self) -> int32_t { return mlirPTOTileBufTypeGetSLayoutValue(self); })
        .def_property_readonly(
            "pad_value",
            [](MlirType self) -> int32_t { return mlirPTOTileBufTypeGetPadValue(self); })
        .def_property_readonly(
            "compact_mode",
            [](MlirType self) -> int32_t { return mlirPTOTileBufTypeGetCompactMode(self); })
        .def_property_readonly("s_fractal_size", [](MlirType self) -> int32_t {
          return mlirPTOTileBufTypeGetSFractalSize(self);
        });
}

static void registerTileBufType(py::module_ &m) {
  mlir_type_subclass tileBuf(
      m, "TileBufType",
      [](MlirType t) -> bool { return mlirPTOTypeIsATileBufType(t); });
  addTileBufTypeGetMethod(tileBuf);
  addTileBufTypeAccessors1(tileBuf, m);
  addTileBufTypeAccessors2(tileBuf, m);
}

void mlir::pto::python::populatePTODialectBindings(pybind11::module_ &m) {
    // --------------------------------------------------------------------------
    // Dialect registration helper
    // --------------------------------------------------------------------------
    m.def(
        "register_dialect",
        [](MlirContext context, bool load) {
            MlirDialectHandle handle = mlirGetDialectHandle__pto__();
            mlirDialectHandleRegisterDialect(handle, context);
            if (load) {
              mlirDialectHandleLoadDialect(handle, context);
            }
        },
        py::arg("context"), py::arg("load") = true);
    registerPTOEnums1(m);
    registerPTOEnums2(m);
    registerPTOEnums3(m);
    registerPTOEnums4(m);
    registerPTOEnums5(m);
    registerPTOSimpleAttrs1(m);
    registerPTOSimpleAttrs2(m);
    registerPTOSimpleAttrs3(m);
    registerPTOSimpleAttrs4(m);
    registerPTOValueAttrs1(m);
    registerPTOValueAttrs2(m);
    registerPTOValueAttrs3(m);
    registerPTOValueAttrs4(m);
    registerPTOValueAttrs5(m);
    registerPTOValueAttrs6(m);
    registerPTOValueAttrs7(m);
    registerPTOValueAttrs8(m);
    registerPTOValueAttrs9(m);
    registerPTOValueAttrs10(m);
    registerPTOValueAttrs11(m);
    registerPTOValueAttrs12(m);
    registerPTOValueAttrs13(m);
    registerMaskPatternAttr(m);
    registerPTOTypes1(m);
    registerPTOTypes2(m);
    registerPTOTypes3(m);
    registerPTOTypes4(m);
    registerPTOTypes4b(m);
    registerPTOTypes5(m);
    registerPTOTypes6(m);
    registerPTOTypes7(m);
    registerPTOTypes8(m);
    registerPTOTypes9(m);
    registerPTOTypes10(m);
    registerTileBufConfigAttr(m);
    registerTileBufType(m);
    populatePTODialectSubmodule(m);
}
