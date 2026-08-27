// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Support/CodeConstants.h"
#include "ptoas.h"

#include "PTO/Transforms/SoftLibService.h"
#include "PTO/Transforms/TileLibService.h"
#include "PTOModule.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

namespace py = pybind11;

namespace {

class PythonTileLibService final : public mlir::pto::TileLibService {
public:
  mlir::FailureOr<std::string> getMetadata(
      const mlir::pto::TileLibMaterializationRequest &request) override {
    py::gil_scoped_acquire acquire;
    try {
      return py::cast<std::string>(getCompilerRuntime().attr("metadata")(
          request.target, request.op, request.operandSpecsJson,
          request.contextAttrsJson));
    } catch (const py::error_already_set &error) {
      llvm::errs() << "TileLib: PTODSL metadata query raised Python "
                      "exception:\n"
                   << error.what() << "\n";
      return mlir::failure();
    }
  }

  mlir::LogicalResult
  materialize(const mlir::pto::TileLibMaterializationRequest &request,
              mlir::MLIRContext &context,
              mlir::pto::TileLibMaterializationCallback callback) override {
    py::gil_scoped_acquire acquire;
    try {
      py::object contextOwner = getPythonContext(context);
      MlirContext pythonContext = py::cast<MlirContext>(contextOwner);
      if (unwrap(pythonContext) != &context) {
        llvm::errs() << "TileLib: Python context does not match the PTOAS "
                        "MLIRContext\n";
        return mlir::failure();
      }

      py::tuple result = getCompilerRuntime().attr("materialize")(
          request.target, request.op, request.operandSpecsJson,
          request.contextAttrsJson, request.candidateId, contextOwner);
      const bool hasExpectedResultArity = result.size() == mlir::pto::kValue2;
      if (!hasExpectedResultArity) {
        throw py::value_error(
            "PTODSL materialize() must return (module, entry_symbol)");
      }

      // MlirModule is a non-owning handle. Keep result[0] alive until the
      // complete source module has been cloned into C++ ownership.
      py::object moduleOwner = result[0];
      MlirModule rawModule = py::cast<MlirModule>(moduleOwner);
      if (!mlirContextEqual(mlirModuleGetContext(rawModule), pythonContext)) {
        llvm::errs() << "TileLib: PTODSL returned a module from a different "
                        "MLIRContext\n";
        return mlir::failure();
      }

      mlir::ModuleOp source = unwrap(rawModule);
      return callback(source, py::cast<std::string>(result[1]));
    } catch (const py::error_already_set &error) {
      llvm::errs() << "TileLib: PTODSL materialization raised Python "
                      "exception:\n"
                   << error.what() << "\n";
      return mlir::failure();
    } catch (const std::exception &error) {
      llvm::errs() << "TileLib: invalid PTODSL materialization result: "
                   << error.what() << "\n";
      return mlir::failure();
    }
  }

  // Shared by the TileLib and SoftLib Python bridges.  The returned Python
  // object owns only the capsule wrapper; the C++ pass owns the MLIR context.
  static py::module_ getCompilerRuntime() {
    // Python's sys.modules cache makes this a process-wide runtime module
    // without storing a py::object whose destructor could outlive CPython.
    return py::module_::import("ptodsl.tilelib._compiler_runtime");
  }

  static py::object getPythonContext(mlir::MLIRContext &context) {
    py::object capsule = py::reinterpret_steal<py::object>(
        mlirPythonContextToCapsule(wrap(&context)));
    return py::module_::import("ptoas.mlir.ir")
        .attr("Context")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule);
  }
};

class PythonSoftLibService final : public mlir::pto::SoftLibService {
public:
  mlir::LogicalResult
  materialize(const mlir::pto::SoftLibMaterializationRequest &request,
              mlir::MLIRContext &context,
              mlir::pto::SoftLibMaterializationCallback callback) override {
    py::gil_scoped_acquire acquire;
    try {
      py::object contextOwner = PythonTileLibService::getPythonContext(context);
      MlirContext pythonContext = py::cast<MlirContext>(contextOwner);
      py::tuple result =
          py::module_::import("ptodsl.softlib._compiler_runtime")
              .attr("materialize")(request.target, request.op,
                                   request.operandSpecsJson, contextOwner);
      const bool hasExpectedResultCount =
          result.size() == mlir::pto::kValue2;
      if (!hasExpectedResultCount) {
        throw py::value_error(
            "SoftLib materialize() must return (module, entry_symbol)");
      }
      py::object moduleOwner = result[0];
      MlirModule rawModule = py::cast<MlirModule>(moduleOwner);
      if (!mlirContextEqual(mlirModuleGetContext(rawModule), pythonContext)) {
        return mlir::failure();
      }
      return callback(unwrap(rawModule), py::cast<std::string>(result[1]));
    } catch (const py::error_already_set &error) {
      llvm::errs()
          << "SoftLib: PTODSL materialization raised Python exception:\n"
          << error.what() << "\n";
      return mlir::failure();
    } catch (const std::exception &error) {
      llvm::errs() << "SoftLib: invalid PTODSL materialization result: "
                   << error.what() << "\n";
      return mlir::failure();
    }
  }
};

constexpr char kRuntimeRegistrationCapsuleName[] =
    "ptoas.TileLibRuntimeRegistration";

class PythonTileLibRuntimeRegistration {
public:
  PythonTileLibRuntimeRegistration()
      : service(std::make_shared<PythonTileLibService>()) {
    mlir::pto::TileLibRuntime::install(service);
    softService = std::make_shared<PythonSoftLibService>();
    mlir::pto::SoftLibRuntime::install(softService);
  }

  ~PythonTileLibRuntimeRegistration() {
    mlir::pto::TileLibRuntime::uninstall(service.get());
    mlir::pto::SoftLibRuntime::uninstall(softService.get());
  }

private:
  std::shared_ptr<PythonTileLibService> service;
  std::shared_ptr<PythonSoftLibService> softService;
};

void destroyRuntimeRegistration(PyObject *capsule) {
  void *pointer =
      PyCapsule_GetPointer(capsule, kRuntimeRegistrationCapsuleName);
  if (!pointer) {
    PyErr_Clear();
    return;
  }
  delete static_cast<PythonTileLibRuntimeRegistration *>(pointer);
}

int runPTOASFromPython(const std::vector<std::string> &arguments) {
  std::vector<std::string> storage = arguments;
  std::vector<char *> argv;
  argv.reserve(storage.size());
  for (std::string &argument : storage) {
    argv.push_back(argument.data());
  }

  py::object contextOwner =
      py::module_::import("ptoas.mlir.ir").attr("Context")();
  MlirContext rawContext = py::cast<MlirContext>(contextOwner);

  int result;
  {
    py::gil_scoped_release release;
    result = mlir::pto::runPTOAS(static_cast<int>(argv.size()), argv.data(),
                                 *unwrap(rawContext));
  }
  return result;
}

} // namespace

PYBIND11_MODULE(_core, module) {
  module.doc() = "PTOAS compiler and PTO dialect native bindings";
  py::module_::import("ptoas.mlir.ir");
  mlir::pto::python::populatePTODialectBindings(module);
  module.add_object("_tilelib_runtime_registration",
                    py::capsule(new PythonTileLibRuntimeRegistration(),
                                kRuntimeRegistrationCapsuleName,
                                destroyRuntimeRegistration));
  module.def("main", &runPTOASFromPython, py::arg("argv"));
}
