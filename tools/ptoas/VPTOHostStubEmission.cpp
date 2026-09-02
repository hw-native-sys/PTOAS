// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "VPTOHostStubEmission.h"

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>

using namespace mlir;

namespace {

constexpr unsigned kBooleanBitWidth = 1;
constexpr unsigned kByteBitWidth = 8;
constexpr unsigned kShortBitWidth = 16;
constexpr unsigned kIntBitWidth = 32;
constexpr unsigned kLongLongBitWidth = 64;

struct VPTOKernelStubDecl {
  std::string logicalName;
  SmallVector<std::string> argTypes;
};

static std::string getLogicalKernelName(llvm::StringRef symbol) {
  if (symbol.ends_with("_mix_aiv")) {
    return symbol.drop_back(strlen("_mix_aiv")).str();
  }
  if (symbol.ends_with("_mix_aic")) {
    return symbol.drop_back(strlen("_mix_aic")).str();
  }
  return symbol.str();
}

static std::string getStubScalarCType(Type type) {
  if (isa<IndexType>(type)) {
    return "long long";
  }
  if (auto intType = dyn_cast<IntegerType>(type)) {
    switch (intType.getWidth()) {
    case kBooleanBitWidth:
    case kByteBitWidth:
      return "signed char";
    case kShortBitWidth:
      return "short";
    case kIntBitWidth:
      return "int";
    case kLongLongBitWidth:
      return "long long";
    default:
      return "long long";
    }
  }
  if (auto floatType = dyn_cast<FloatType>(type)) {
    if (floatType.isF32()) {
      return "float";
    }
    if (floatType.isF64()) {
      return "double";
    }
    return "short";
  }
  return "long long";
}

static std::string getStubCType(Type type) {
  if (isa<pto::PtrType, MemRefType>(type)) {
    return "__gm__ void *";
  }
  return getStubScalarCType(type);
}

} // namespace

static LogicalResult collectVPTOKernelStubDecls(
    ArrayRef<ModuleOp> modules, SmallVectorImpl<VPTOKernelStubDecl> &decls,
    llvm::raw_ostream &diagOS) {
  bool hadError = false;
  llvm::StringMap<unsigned> logicalNameToIndex;

  for (ModuleOp module : modules) {
    module.walk([&decls, &logicalNameToIndex, &hadError,
                 &diagOS](func::FuncOp func) {
      if (!pto::isPTOEntryFunction(func)) {
        return;
      }

      std::string logicalName = getLogicalKernelName(func.getSymName());
      SmallVector<std::string> argTypes;
      argTypes.reserve(func.getNumArguments());
      for (Type type : func.getArgumentTypes()) {
        argTypes.push_back(getStubCType(type));
      }

      auto [it, inserted] =
          logicalNameToIndex.try_emplace(logicalName, decls.size());
      if (inserted) {
        decls.push_back(VPTOKernelStubDecl{logicalName, std::move(argTypes)});
        return;
      }

      VPTOKernelStubDecl &existing = decls[it->second];
      if (existing.argTypes != argTypes) {
        diagOS << "Error: mixed kernel variants disagree on host stub signature "
               << "for '" << logicalName << "'.\n";
        hadError = true;
      }
    });
  }

  return hadError ? failure() : success();
}

LogicalResult mlir::pto::emitVPTOHostStubSource(ModuleOp module,
                                                std::string &stubSource,
                                                llvm::raw_ostream &diagOS) {
  return emitVPTOHostStubSource(ArrayRef<ModuleOp>(module), stubSource, diagOS);
}

LogicalResult mlir::pto::emitVPTOHostStubSource(ArrayRef<ModuleOp> modules,
                                                std::string &stubSource,
                                                llvm::raw_ostream &diagOS) {
  SmallVector<VPTOKernelStubDecl> stubDecls;
  if (failed(collectVPTOKernelStubDecls(modules, stubDecls, diagOS))) {
    return failure();
  }

  if (stubDecls.empty()) {
    diagOS << "Error: no PTO entry functions found for host stub emission.\n";
    return failure();
  }

  stubSource.clear();
  llvm::raw_string_ostream os(stubSource);
  os << "#ifndef AICORE\n#define AICORE [aicore]\n#endif\n\n";
  for (const VPTOKernelStubDecl &decl : stubDecls) {
    os << "extern \"C\" __global__ AICORE void " << decl.logicalName << "(";
    for (size_t i = 0; i < decl.argTypes.size(); ++i) {
      if (i != 0) {
        os << ", ";
      }
      os << decl.argTypes[i] << " arg" << i;
    }
    os << ") {}\n";
  }
  os.flush();
  return success();
}
