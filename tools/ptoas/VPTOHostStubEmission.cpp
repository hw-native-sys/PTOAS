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
  SmallVector<unsigned> ptrElemBytes;
  SmallVector<bool> syntheticArgs;
  std::string kernelSuffix; // "_mix_aiv" or "_mix_aic" (ABI name suffix)
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

// A stub arg type string of "__gm__ void *" marks a pointer (tensor) param;
// everything else is a scalar. Mirrors the ptr/scalar split the EmitC wrapper
// does (pypto pto_backend.py:571 tensor_params vs scalar_params).
static bool isPtrArgType(llvm::StringRef cType) {
  return cType.contains("__gm__ void *");
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

static unsigned getPointerElementBytes(Type type) {
  auto ptr = dyn_cast<pto::PtrType>(type);
  if (!ptr)
    return 1;
  Type elem = ptr.getElementType();
  if (elem.isBF16() || elem.isF16() || elem.isInteger(16)) return 2;
  if (elem.isF32() || elem.isInteger(32)) return 4;
  if (elem.isF64() || elem.isInteger(64)) return 8;
  return 1;
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
      SmallVector<unsigned> ptrElemBytes;
      SmallVector<bool> syntheticArgs;
      argTypes.reserve(func.getNumArguments());
      ptrElemBytes.reserve(func.getNumArguments());
      syntheticArgs.reserve(func.getNumArguments());
      unsigned argNo = 0;
      for (BlockArgument arg : func.getArguments()) {
        Type type = arg.getType();
        argTypes.push_back(getStubCType(type));
        ptrElemBytes.push_back(getPointerElementBytes(type));
        // PTOAS's parsed func arguments do not retain SSA name hints.  The
        // PyPTO ABI appends synthetic dispatch parameters in canonical order
        // (block_idx, block_num, optional subblock_idx), so identify only the
        // trailing i32 parameters and leave user scalars untouched.
        bool trailingDispatch = argNo + 2 >= func.getNumArguments() &&
                                isa<IntegerType>(type) &&
                                cast<IntegerType>(type).getWidth() == 32;
        syntheticArgs.push_back(trailingDispatch);
        ++argNo;
      }
      // Derive the ABI suffix from the kernel_kind attr on the owning child
      // ModuleOp (Vector -> "_mix_aiv", Cube -> "_mix_aic"). The PTO IR func
      // name has no suffix yet (rename happens later in the lowering pipeline),
      // so the wrapper forward-call target = logicalName + suffix.
      std::string suffix;
      if (auto *parentOp = func->getParentOp()) {
        if (auto kindAttr = parentOp->getAttrOfType<pto::FunctionKernelKindAttr>(
                pto::FunctionKernelKindAttr::name)) {
          if (kindAttr.getKernelKind() == pto::FunctionKernelKind::Vector) {
            suffix = "_mix_aiv";
          } else if (kindAttr.getKernelKind() == pto::FunctionKernelKind::Cube) {
            suffix = "_mix_aic";
          }
        }
      }

      auto [it, inserted] =
          logicalNameToIndex.try_emplace(logicalName, decls.size());
      if (inserted) {
        decls.push_back(VPTOKernelStubDecl{
            logicalName, std::move(argTypes), std::move(ptrElemBytes),
            std::move(syntheticArgs), std::move(suffix)});
        return;
      }

      VPTOKernelStubDecl &existing = decls[it->second];
      if (existing.argTypes != argTypes || existing.ptrElemBytes != ptrElemBytes ||
          existing.syntheticArgs != syntheticArgs) {
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

// ---------------------------------------------------------------------------
// Device-side kernel_entry wrapper (M1: VPTO <-> simpler scheduler ABI)
// ---------------------------------------------------------------------------

LogicalResult mlir::pto::emitVPTODeviceWrapperSource(
    ArrayRef<ModuleOp> modules, std::string &wrapperSource,
    llvm::raw_ostream &diagOS) {
  SmallVector<VPTOKernelStubDecl> stubDecls;
  if (failed(collectVPTOKernelStubDecls(modules, stubDecls, diagOS))) {
    return failure();
  }
  if (stubDecls.empty()) {
    diagOS << "Error: no PTO entry functions found for device wrapper emission.\n";
    return failure();
  }
  if (stubDecls.size() != 1) {
    diagOS << "Error: merged device wrapper emission currently requires exactly "
               "one PTO entry function per device ELF; got "
            << stubDecls.size() << ".\n";
    return failure();
  }

  wrapperSource.clear();
  llvm::raw_string_ostream os(wrapperSource);
  // The simpler scheduler fills PTO2DispatchPayload.args[] with ChipTensor*
  // (tensors) and uint64 (scalars); the wrapper unpacks them the way the EmitC
  // route does (pypto pto_backend.py:571-629). We inline a minimal view of the
  // ChipTensor layout (buffer.addr @0, start_offset @24) instead of including
  // simpler's tensor.h — PTOAS must not depend on the simpler runtime tree.
  os << "#include <cstddef>\n"
        "#include <cstdint>\n"
        "#include <pto/pto-inst.hpp>\n"
        "using namespace pto;\n";
  os << "// Minimal ChipTensor layout (buffer.addr + start_offset only).\n"
        "// Matches simpler's ChipTensor (PTOBufferHandle{addr,size}, "
        "owner_task_id, start_offset @offset 24).\n"
        "struct VptoWrapperTensor {\n"
        "  struct { uint64_t addr; uint64_t size; } buffer;\n"
        "  uint64_t owner_task_id;\n"
        "  uint64_t start_offset;\n"
        "};\n"
        "static_assert(offsetof(VptoWrapperTensor, start_offset) == 24, \"ChipTensor ABI mismatch\");\n"
        "struct VptoWrapperLocalContext { int32_t block_idx; int32_t block_num; };\n"
        "struct VptoWrapperGlobalContext { int32_t sub_block_id; };\n"
        "static __aicore__ inline int32_t vpto_get_block_idx(__gm__ int64_t* args) {\n"
        "  return reinterpret_cast<__gm__ VptoWrapperLocalContext*>(args[48])->block_idx;\n"
        "}\n"
        "static __aicore__ inline int32_t vpto_get_block_num(__gm__ int64_t* args) {\n"
        "  return reinterpret_cast<__gm__ VptoWrapperLocalContext*>(args[48])->block_num;\n"
        "}\n"
        "static __aicore__ inline int32_t vpto_get_sub_block_id(__gm__ int64_t* args) {\n"
        "  return reinterpret_cast<__gm__ VptoWrapperGlobalContext*>(args[49])->sub_block_id;\n"
        "}\n"
        "#ifndef __aicore__\n"
        "#define __aicore__ [aicore]\n"
        "#endif\n"
        "#ifndef __gm__\n"
        "#define __gm__\n"
        "#endif\n";
  // Forward declarations for the VPTO typed bodies (compiled separately as
  // LLVM IR → bisheng -x ir → .o, then ld.lld-merged with this wrapper .o).
  // Must be at global scope — bisheng rejects `extern "C"` inside a device
  // function body. The body's own definition carries __aicore__; the extern
  // declaration here just needs the symbol name and signature.
  for (const VPTOKernelStubDecl &decl : stubDecls) {
    os << "extern \"C\" __aicore__ __attribute__((always_inline)) void "
       << decl.logicalName << decl.kernelSuffix << "(";
    for (size_t i = 0; i < decl.argTypes.size(); ++i) {
      if (i) {
        os << ", ";
      }
      os << decl.argTypes[i];
    }
    os << ");\n";
  }
  os << "extern \"C\" __aicore__\n"
        "void kernel_entry(__gm__ int64_t* args) {\n"
        "  // Match the EmitC entry contract: a preceding task may leave the\n"
        "  // hardware atomic mode enabled, so every scheduler entry resets it.\n"
        "  set_atomic_none();\n";
  // Merged-device mode intentionally accepts one PTO entry per ELF.  Simpler
  // selects the ELF by func_id, then enters this ELF's sole kernel_entry.
  for (const VPTOKernelStubDecl &decl : stubDecls) {
    // Split args[] into tensors-first (ChipTensor*) and scalars-second,
    // mirroring the EmitC wrapper (pypto pto_backend.py:571-629).
    SmallVector<unsigned> tensorIdx;
    SmallVector<unsigned> scalarIdx;
    for (unsigned i = 0; i < decl.argTypes.size(); ++i) {
      if (decl.syntheticArgs[i])
        continue;
      if (isPtrArgType(decl.argTypes[i])) {
        tensorIdx.push_back(i);
      } else {
        scalarIdx.push_back(i);
      }
    }

    // Build the forward-call argument list: unpack tensors then scalars.
    SmallVector<std::string> callArgs(decl.argTypes.size());
    unsigned tensorSlot = 0;
    for (unsigned i : tensorIdx) {
      // Each tensor arg arrives as args[tensorSlot] pointing at a ChipTensor;
      // extract buffer.addr + start_offset, cast to the declared __gm__ T*.
      std::string cType = decl.argTypes[i]; // "__gm__ void *"
      // Strip "__gm__ void *" -> "__gm__ void*" then take the pointee type;
      // the body expects "__gm__ void*" (generic GM pointer) for ptr args.
      os << "  __gm__ VptoWrapperTensor* _t" << i
         << "_tensor = reinterpret_cast<__gm__ VptoWrapperTensor*>(args["
         << tensorSlot << "]);\n";
      os << "  " << cType << " _t" << i << " = reinterpret_cast<" << cType
         << ">(reinterpret_cast<__gm__ uint8_t*>(_t" << i
         << "_tensor->buffer.addr) + _t" << i << "_tensor->start_offset * "
         << decl.ptrElemBytes[i] << ");\n";
      callArgs[i] = "_t" + std::to_string(i);
      ++tensorSlot;
    }
    unsigned scalarSlot = tensorIdx.size();
    for (unsigned i : scalarIdx) {
      const std::string &cType = decl.argTypes[i];
      os << "  union { uint64_t u64; " << cType << " val; } _s" << i
         << "_conv;\n";
      os << "  _s" << i << "_conv.u64 = args[" << scalarSlot << "];\n";
      os << "  " << cType << " _s" << i << " = _s" << i << "_conv.val;\n";
      callArgs[i] = "_s" + std::to_string(i);
      ++scalarSlot;
    }

    bool hasSpmd = false;
    for (unsigned i = 0; i < decl.argTypes.size(); ++i)
      hasSpmd |= decl.syntheticArgs[i];
    if (hasSpmd) {
      os << "  int32_t __pypto_spmd_block_idx = vpto_get_block_idx(args);\n"
            "  int32_t __pypto_spmd_block_num = vpto_get_block_num(args);\n";
      unsigned syntheticOrdinal = 0;
      for (unsigned i = 0; i < decl.argTypes.size(); ++i) {
        if (!decl.syntheticArgs[i])
          continue;
        if (syntheticOrdinal == 0)
          callArgs[i] = "__pypto_spmd_block_idx";
        else if (syntheticOrdinal == 1)
          callArgs[i] = "__pypto_spmd_block_num";
        else
          callArgs[i] = "vpto_get_sub_block_id(args)";
        ++syntheticOrdinal;
      }
    }

    // Forward to the typed VPTO body. (Single-kernel-per-module case: this is
    //  the only path the scheduler will reach via kernel_entry.)
    os << "  " << decl.logicalName << decl.kernelSuffix << "(";
    for (size_t i = 0; i < callArgs.size(); ++i) {
      if (i) {
        os << ", ";
      }
      os << callArgs[i];
    }
    os << ");\n";
  }
  os << "}\n";
  os.flush();
  return success();
}

LogicalResult mlir::pto::emitVPTODeviceWrapperSource(
    ModuleOp module, std::string &wrapperSource, llvm::raw_ostream &diagOS) {
  return emitVPTODeviceWrapperSource(ArrayRef<ModuleOp>(module), wrapperSource,
                                      diagOS);
}
