// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "ptoas.h"

#include "PTOModule.h"
#include "PTO/Transforms/TileLibService.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"

#include "llvm/Support/raw_ostream.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include <string>
#include <vector>

namespace py = pybind11;

namespace {

class PythonTileLibService final : public mlir::pto::TileLibService {
public:
  explicit PythonTileLibService(py::object contextOwner)
      : contextOwner(std::move(contextOwner)) {}

  mlir::FailureOr<std::string>
  getMetadata(const mlir::pto::TileLibMaterializationRequest &request) override {
    py::gil_scoped_acquire acquire;
    try {
      return py::cast<std::string>(getRuntime().attr("metadata")(
          request.target, request.op, request.operandSpecsJson,
          request.contextAttrsJson));
    } catch (const py::error_already_set &error) {
      llvm::errs() << "TileLib: PTODSL metadata query raised Python "
                      "exception:\n"
                   << error.what() << "\n";
      return mlir::failure();
    }
  }

  mlir::FailureOr<mlir::pto::TileLibMaterialization>
  materialize(const mlir::pto::TileLibMaterializationRequest &request,
              mlir::MLIRContext &context) override {
    py::gil_scoped_acquire acquire;
    try {
      MlirContext pythonContext = py::cast<MlirContext>(contextOwner);
      if (unwrap(pythonContext) != &context) {
        llvm::errs() << "TileLib: Python context does not match the PTOAS "
                        "MLIRContext\n";
        return mlir::failure();
      }

      py::tuple result = getRuntime().attr("materialize")(
          request.target, request.op, request.operandSpecsJson,
          request.contextAttrsJson, request.candidateId, contextOwner);
      if (result.size() != 2)
        throw py::value_error(
            "PTODSL materialize() must return (module, entry_symbol)");

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
      auto cloned = mlir::cast<mlir::ModuleOp>(source->clone());
      mlir::pto::TileLibMaterialization materialization{
          mlir::OwningOpRef<mlir::ModuleOp>(cloned),
          py::cast<std::string>(result[1])};
      return materialization;
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

private:
  py::object &getRuntime() {
    if (!runtime)
      runtime = py::module_::import("ptodsl.tilelib._compiler_runtime");
    return runtime;
  }

  // These objects are created and destroyed by runPTOASFromPython while the
  // calling thread owns the GIL. materialize() reacquires it for every DSL
  // invocation because the native compiler releases it around the driver.
  py::object contextOwner;
  py::object runtime;
};

int runPTOASFromPython(const std::vector<std::string> &arguments) {
  std::vector<std::string> storage = arguments;
  std::vector<char *> argv;
  argv.reserve(storage.size());
  for (std::string &argument : storage)
    argv.push_back(argument.data());

  py::object contextOwner =
      py::module_::import("ptoas.mlir.ir").attr("Context")();
  MlirContext rawContext = py::cast<MlirContext>(contextOwner);
  auto tileLibService =
      std::make_shared<PythonTileLibService>(contextOwner);

  int result;
  {
    py::gil_scoped_release release;
    result = mlir::pto::runPTOAS(static_cast<int>(argv.size()), argv.data(),
                                 *unwrap(rawContext), tileLibService);
  }
  return result;
}

} // namespace

PYBIND11_MODULE(_core, module) {
  module.doc() = "PTOAS compiler and PTO dialect native bindings";
  py::module_::import("ptoas.mlir.ir");
  mlir::pto::python::populatePTODialectBindings(module);
  module.def("main", &runPTOASFromPython, py::arg("argv"));
}
