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

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "pto-c/Dialect/PTO.h"
#include "pybind11/stl.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinTypes.h"
#include "PTO/IR/PTO.h"
#include "PTOModule.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;
using llvm::cast;
using llvm::isa;

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
                                mlir::ShapedType::kDynamic);
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

static void populatePTODialectSubmodule(pybind11::module &m) {
  (void)m;
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

    // [保留 HEAD]: AddressSpace 枚举定义
    py::enum_<mlir::pto::AddressSpace>(m, "AddressSpace")
    .value("Zero", mlir::pto::AddressSpace::Zero)
    .value("GM",   mlir::pto::AddressSpace::GM)
    .value("MAT",   mlir::pto::AddressSpace::MAT)
    .value("LEFT",  mlir::pto::AddressSpace::LEFT)
    .value("RIGHT",  mlir::pto::AddressSpace::RIGHT)
    .value("ACC",  mlir::pto::AddressSpace::ACC)
    .value("VEC",   mlir::pto::AddressSpace::VEC)
    .value("BIAS",   mlir::pto::AddressSpace::BIAS)
    .value("SCALING", mlir::pto::AddressSpace::SCALING)
    .export_values();
    py::enum_<mlir::pto::FenceScope>(m, "FenceScope")
    .value("LocalMemory", mlir::pto::FenceScope::LocalMemory)
    .value("GM", mlir::pto::FenceScope::GM)
    .value("All", mlir::pto::FenceScope::All)
    .export_values();
    py::enum_<mlir::pto::BLayout>(m, "BLayout")
    .value("RowMajor", mlir::pto::BLayout::RowMajor)
    .value("ColMajor", mlir::pto::BLayout::ColMajor);

    py::enum_<mlir::pto::SLayout>(m, "SLayout")
    .value("NoneBox", mlir::pto::SLayout::NoneBox)
    .value("RowMajor", mlir::pto::SLayout::RowMajor)
    .value("ColMajor", mlir::pto::SLayout::ColMajor);

    py::enum_<mlir::pto::PadValue>(m, "PadValue")
    .value("Null", mlir::pto::PadValue::Null)
    .value("Zero", mlir::pto::PadValue::Zero)
    .value("Max", mlir::pto::PadValue::Max)
    .value("Min", mlir::pto::PadValue::Min);

    py::enum_<mlir::pto::CompactMode>(m, "CompactMode")
    .value("Null", mlir::pto::CompactMode::Null)
    .value("Normal", mlir::pto::CompactMode::Normal)
    .value("RowPlusOne", mlir::pto::CompactMode::RowPlusOne);

    py::enum_<mlir::pto::RoundMode>(m, "RoundMode")
    .value("NONE", mlir::pto::RoundMode::NONE)
    .value("RINT", mlir::pto::RoundMode::RINT)
    .value("ROUND", mlir::pto::RoundMode::ROUND)
    .value("FLOOR", mlir::pto::RoundMode::FLOOR)
    .value("CEIL", mlir::pto::RoundMode::CEIL)
    .value("TRUNC", mlir::pto::RoundMode::TRUNC)
    .value("ODD", mlir::pto::RoundMode::ODD)
    .value("CAST_RINT", mlir::pto::RoundMode::CAST_RINT);

    py::enum_<mlir::pto::DivPrecision>(m, "DivPrecision")
    .value("Default", mlir::pto::DivPrecision::Default)
    .value("HighPrecision", mlir::pto::DivPrecision::HighPrecision);

    py::enum_<mlir::pto::ExpPrecision>(m, "ExpPrecision")
    .value("Default", mlir::pto::ExpPrecision::Default)
    .value("HighPrecision", mlir::pto::ExpPrecision::HighPrecision);

    py::enum_<mlir::pto::LogPrecision>(m, "LogPrecision")
    .value("Default", mlir::pto::LogPrecision::Default)
    .value("HighPrecision", mlir::pto::LogPrecision::HighPrecision);

    py::enum_<mlir::pto::RecipPrecision>(m, "RecipPrecision")
    .value("Default", mlir::pto::RecipPrecision::Default)
    .value("HighPrecision", mlir::pto::RecipPrecision::HighPrecision);

    py::enum_<mlir::pto::RemPrecision>(m, "RemPrecision")
    .value("Default", mlir::pto::RemPrecision::Default)
    .value("HighPrecision", mlir::pto::RemPrecision::HighPrecision);

    py::enum_<mlir::pto::RsqrtPrecision>(m, "RsqrtPrecision")
    .value("Default", mlir::pto::RsqrtPrecision::Default)
    .value("HighPrecision", mlir::pto::RsqrtPrecision::HighPrecision);

    py::enum_<mlir::pto::SqrtPrecision>(m, "SqrtPrecision")
    .value("Default", mlir::pto::SqrtPrecision::Default)
    .value("HighPrecision", mlir::pto::SqrtPrecision::HighPrecision);

    py::enum_<mlir::pto::FmodPrecision>(m, "FmodPrecision")
    .value("Default", mlir::pto::FmodPrecision::Default)
    .value("HighPrecision", mlir::pto::FmodPrecision::HighPrecision);

    py::enum_<mlir::pto::SaturationMode>(m, "SaturationMode")
    .value("ON", mlir::pto::SaturationMode::ON)
    .value("OFF", mlir::pto::SaturationMode::OFF);

    py::enum_<MlirPTOCmpMode>(m, "CmpMode")
      .value("EQ", MlirPTOCmpMode_EQ)
      .value("NE", MlirPTOCmpMode_NE)
      .value("LT", MlirPTOCmpMode_LT)
      .value("LE", MlirPTOCmpMode_LE)
      .value("GT", MlirPTOCmpMode_GT)
      .value("GE", MlirPTOCmpMode_GE)
      .export_values();

    py::enum_<mlir::pto::PIPE>(m, "PIPE")
      .value("PIPE_S", mlir::pto::PIPE::PIPE_S)
      .value("PIPE_V", mlir::pto::PIPE::PIPE_V)
      .value("PIPE_M", mlir::pto::PIPE::PIPE_M)
      .value("PIPE_MTE1", mlir::pto::PIPE::PIPE_MTE1)
      .value("PIPE_MTE2", mlir::pto::PIPE::PIPE_MTE2)
      .value("PIPE_MTE3", mlir::pto::PIPE::PIPE_MTE3)
      .value("PIPE_ALL", mlir::pto::PIPE::PIPE_ALL)
      .value("PIPE_MTE4", mlir::pto::PIPE::PIPE_MTE4)
      .value("PIPE_MTE5", mlir::pto::PIPE::PIPE_MTE5)
      .value("PIPE_V2", mlir::pto::PIPE::PIPE_V2)
      .value("PIPE_FIX", mlir::pto::PIPE::PIPE_FIX)
      .value("VIRTUAL_PIPE_MTE2_L1A", mlir::pto::PIPE::VIRTUAL_PIPE_MTE2_L1A)
      .value("VIRTUAL_PIPE_MTE2_L1B", mlir::pto::PIPE::VIRTUAL_PIPE_MTE2_L1B)
      .value("PIPE_NUM", mlir::pto::PIPE::PIPE_NUM)
      .value("PIPE_UNASSIGNED", mlir::pto::PIPE::PIPE_UNASSIGNED);

    py::enum_<mlir::pto::Layout>(m, "Layout")
      .value("ND", mlir::pto::Layout::ND)
      .value("DN", mlir::pto::Layout::DN)
      .value("NZ", mlir::pto::Layout::NZ)
      .value("MX_A_ZZ", mlir::pto::Layout::MX_A_ZZ)
      .value("MX_B_NN", mlir::pto::Layout::MX_B_NN);

    py::enum_<mlir::pto::AccToVecMode>(m, "AccToVecMode")
      .value("SingleModeVec0", mlir::pto::AccToVecMode::SingleModeVec0)
      .value("SingleModeVec1", mlir::pto::AccToVecMode::SingleModeVec1)
      .value("DualModeSplitM", mlir::pto::AccToVecMode::DualModeSplitM)
      .value("DualModeSplitN", mlir::pto::AccToVecMode::DualModeSplitN)
      .export_values();

    py::enum_<mlir::pto::TInsertMode>(m, "TInsertMode")
      .value("SPLIT2", mlir::pto::TInsertMode::SPLIT2)
      .value("SPLIT4", mlir::pto::TInsertMode::SPLIT4)
      .export_values();

    py::enum_<mlir::pto::ReluPreMode>(m, "ReluPreMode")
      .value("NoRelu", mlir::pto::ReluPreMode::NoRelu)
      .value("NormalRelu", mlir::pto::ReluPreMode::NormalRelu)
      .export_values();

    py::enum_<mlir::pto::AtomicType>(m, "AtomicType")
      .value("AtomicNone", mlir::pto::AtomicType::AtomicNone)
      .value("AtomicAdd", mlir::pto::AtomicType::AtomicAdd)
      .export_values();

    py::enum_<mlir::pto::NotifyOp>(m, "NotifyOp")
      .value("AtomicAdd", mlir::pto::NotifyOp::AtomicAdd)
      .value("Set", mlir::pto::NotifyOp::Set)
      .export_values();

    py::enum_<mlir::pto::WaitCmp>(m, "WaitCmp")
      .value("EQ", mlir::pto::WaitCmp::EQ)
      .value("NE", mlir::pto::WaitCmp::NE)
      .value("GT", mlir::pto::WaitCmp::GT)
      .value("GE", mlir::pto::WaitCmp::GE)
      .value("LT", mlir::pto::WaitCmp::LT)
      .value("LE", mlir::pto::WaitCmp::LE)
      .export_values();

    py::enum_<mlir::pto::ReduceOp>(m, "ReduceOp")
      .value("Sum", mlir::pto::ReduceOp::Sum)
      .value("Max", mlir::pto::ReduceOp::Max)
      .value("Min", mlir::pto::ReduceOp::Min)
      .export_values();

    py::enum_<mlir::pto::SyncOpType>(m, "SyncOpType")
      .value("TLOAD", mlir::pto::SyncOpType::TLOAD)
      .value("TSTORE_ACC", mlir::pto::SyncOpType::TSTORE_ACC)
      .value("TSTORE_VEC", mlir::pto::SyncOpType::TSTORE_VEC)
      .value("TMOV_M2L", mlir::pto::SyncOpType::TMOV_M2L)
      .value("TMOV_M2S", mlir::pto::SyncOpType::TMOV_M2S)
      .value("TMOV_M2B", mlir::pto::SyncOpType::TMOV_M2B)
      .value("TMOV_M2V", mlir::pto::SyncOpType::TMOV_M2V)
      .value("TMOV_V2M", mlir::pto::SyncOpType::TMOV_V2M)
      .value("TMATMUL", mlir::pto::SyncOpType::TMATMUL)
      .value("TVEC", mlir::pto::SyncOpType::TVEC)
      .value("TVECWAIT_EVENT", mlir::pto::SyncOpType::TVECWAIT_EVENT)
      .export_values();

    py::enum_<mlir::pto::EVENT>(m, "EVENT")
      .value("EVENT_ID0", mlir::pto::EVENT::EVENT_ID0)
      .value("EVENT_ID1", mlir::pto::EVENT::EVENT_ID1)
      .value("EVENT_ID2", mlir::pto::EVENT::EVENT_ID2)
      .value("EVENT_ID3", mlir::pto::EVENT::EVENT_ID3)
      .value("EVENT_ID4", mlir::pto::EVENT::EVENT_ID4)
      .value("EVENT_ID5", mlir::pto::EVENT::EVENT_ID5)
      .value("EVENT_ID6", mlir::pto::EVENT::EVENT_ID6)
      .value("EVENT_ID7", mlir::pto::EVENT::EVENT_ID7)
      .export_values();

    py::enum_<mlir::pto::MaskPattern>(m, "MaskPattern")
      .value("P0101", mlir::pto::MaskPattern::P0101)
      .value("P1010", mlir::pto::MaskPattern::P1010)
      .value("P0001", mlir::pto::MaskPattern::P0001)
      .value("P0010", mlir::pto::MaskPattern::P0010)
      .value("P0100", mlir::pto::MaskPattern::P0100)
      .value("P1000", mlir::pto::MaskPattern::P1000)
      .value("P1111", mlir::pto::MaskPattern::P1111)
      .export_values();
    py::object maskPatternEnumType = m.attr("MaskPattern");

    mlir_attribute_subclass(m, "BLayoutAttr",
                        [](MlirAttribute a) -> bool {
                          return mlirPTOAttrIsABLayoutAttr(a);
                        })
    .def_classmethod(
        "get",
        [](py::object cls, mlir::pto::BLayout value, MlirContext ctx) -> py::object {
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
            [](py::object cls, mlir::pto::SLayout value, MlirContext ctx) -> py::object {
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
            [](py::object cls, mlir::pto::PadValue value, MlirContext ctx) -> py::object {
            MlirAttribute a = mlirPTOPadValueAttrGet(ctx, static_cast<int32_t>(value));
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls(a);
            },
            py::arg("cls"), py::arg("value"), py::arg("context") = py::none());

    mlir_attribute_subclass(m, "CompactModeAttr",
                            [](MlirAttribute a) -> bool {
                            return mlirPTOAttrIsACompactModeAttr(a);
                            })
        .def_classmethod(
            "get",
            [](py::object cls, mlir::pto::CompactMode value, MlirContext ctx) -> py::object {
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
            [](py::object cls, mlir::pto::AccToVecMode value, MlirContext ctx) -> py::object {
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
            [](py::object cls, mlir::pto::TInsertMode value, MlirContext ctx) -> py::object {
            MlirAttribute a = mlirPTOTInsertModeAttrGet(ctx, static_cast<int32_t>(value));
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls(a);
            },
            py::arg("cls"), py::arg("value"), py::arg("context") = py::none());

    mlir_attribute_subclass(m, "ReluPreModeAttr",
                            [](MlirAttribute a) -> bool {
                            return mlirPTOAttrIsAReluPreModeAttr(a);
                            })
        .def_classmethod(
            "get",
            [](py::object cls, mlir::pto::ReluPreMode value, MlirContext ctx) -> py::object {
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
            [](py::object cls, mlir::pto::AtomicType value, MlirContext ctx) -> py::object {
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
            [](py::object cls, mlir::pto::NotifyOp value, MlirContext ctx) -> py::object {
            MlirAttribute a = mlirPTONotifyOpAttrGet(ctx, static_cast<int32_t>(value));
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls(a);
            },
            py::arg("cls"), py::arg("value"), py::arg("context") = py::none());

    mlir_attribute_subclass(m, "WaitCmpAttr",
                            [](MlirAttribute a) -> bool {
                            return mlirPTOAttrIsAWaitCmpAttr(a);
                            })
        .def_classmethod(
            "get",
            [](py::object cls, mlir::pto::WaitCmp value, MlirContext ctx) -> py::object {
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
            [](py::object cls, mlir::pto::ReduceOp value, MlirContext ctx) -> py::object {
            MlirAttribute a = mlirPTOReduceOpAttrGet(ctx, static_cast<int32_t>(value));
            if (mlirAttributeIsNull(a)) {
              return py::none();
            }
            return cls(a);
            },
            py::arg("cls"), py::arg("value"), py::arg("context") = py::none());
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

    py::enum_<mlir::pto::Coalesce>(m, "Coalesce")
      .value("Elem", mlir::pto::Coalesce::Elem)
      .value("Row", mlir::pto::Coalesce::Row)
      .export_values();

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

    py::enum_<mlir::pto::QuantType>(m, "QuantType")
      .value("INT8_SYM",  mlir::pto::QuantType::INT8_SYM)
      .value("INT8_ASYM", mlir::pto::QuantType::INT8_ASYM)
      .value("MXFP8",     mlir::pto::QuantType::MXFP8)
      .value("MXFP4_E2M1", mlir::pto::QuantType::MXFP4_E2M1)
      .export_values();

    py::enum_<mlir::pto::QuantScaleAlg>(m, "QuantScaleAlg")
      .value("OCP", mlir::pto::QuantScaleAlg::OCP)
      .value("NV", mlir::pto::QuantScaleAlg::NV)
      .export_values();

    py::enum_<mlir::pto::MxGroupAxis>(m, "MxGroupAxis")
      .value("Axis0", mlir::pto::MxGroupAxis::Axis0)
      .value("Axis1", mlir::pto::MxGroupAxis::Axis1)
      .export_values();

    py::enum_<mlir::pto::VecStoreMode>(m, "VecStoreMode")
      .value("ND", mlir::pto::VecStoreMode::ND)
      .value("NZ", mlir::pto::VecStoreMode::NZ)
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

    mlir_attribute_subclass(
        m, "MaskPatternAttr",
        [](MlirAttribute a) { return mlirPTOAttrIsAMaskPatternAttr(a); })
      .def_classmethod(
          "get",
          [maskPatternEnumType](py::object cls, py::object value,
                                MlirContext ctx) -> py::object {
            MlirAttribute a{nullptr};
            if (py::isinstance(value, maskPatternEnumType)) {
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
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
      .def_property_readonly(
          "value",
          [](MlirAttribute self) -> mlir::pto::MaskPattern {
            return static_cast<mlir::pto::MaskPattern>(
                mlirPTOMaskPatternAttrGetEnumValue(self));
          })
      .def_property_readonly(
          "int_value",
          [](MlirAttribute self) -> int32_t {
            return mlirPTOMaskPatternAttrGetValue(self);
          });

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

    // ---- TileBufConfigAttr ----
    mlir_attribute_subclass(m, "TileBufConfigAttr",
                            [](MlirAttribute a) -> bool {
                                return mlirPTOAttrIsATileBufConfigAttr(a);
                            })
        .def_classmethod(
            "get_default",
            [](py::object cls, MlirContext ctx) -> py::object {
                MlirAttribute a = mlirPTOTileBufConfigAttrGetDefault(ctx);
                if (mlirAttributeIsNull(a)) {
                  return py::none();
                }
                return cls(a);
            },
            py::arg("cls"), py::arg("context") = py::none())
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
                    ctx, static_cast<int32_t>(mlir::pto::CompactMode::Null));
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

    // ---- TileBufType ----
    mlir_type_subclass(m, "TileBufType", [](MlirType t) -> bool { return mlirPTOTypeIsATileBufType(t); })
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
            py::arg("valid_shape") = py::none(), py::arg("config") = py::none(), py::arg("context") = py::none())
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
                               })
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

    populatePTODialectSubmodule(m);
}
