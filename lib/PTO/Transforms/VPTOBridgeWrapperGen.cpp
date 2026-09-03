// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOBridgeWrapperGen.cpp - bridge wrapper source generation -------===//
//===----------------------------------------------------------------------===//
//
// Resolves logical bridge entries plus structured specializations into
// concrete wrapper instances. Family renderers own the final C++ spelling;
// external route policy is deliberately not consulted here.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VPTOBridgeRegistry.h"
#include "PTO/Transforms/VPTOBridgeTokens.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace mlir {
namespace pto {

#define GEN_PASS_DECL_VPTOBRIDGEWRAPPERGEN
#define GEN_PASS_DEF_VPTOBRIDGEWRAPPERGEN
#include "PTO/Transforms/Passes.h.inc"

namespace {

static FailureOr<std::string> renderStructuredTile(DictionaryAttr tile) {
  auto element = tile.getAs<TypeAttr>("element_type");
  auto shape = tile.getAs<DenseI64ArrayAttr>("shape");
  auto valid = tile.getAs<DenseI64ArrayAttr>("valid_shape");
  auto memory = tile.getAs<AddressSpaceAttr>("memory_space");
  auto bLayout = tile.getAs<IntegerAttr>("b_layout");
  auto sLayout = tile.getAs<IntegerAttr>("s_layout");
  auto fractal = tile.getAs<IntegerAttr>("s_fractal");
  if (!element || !shape || !valid || !memory || !bLayout || !sLayout ||
      !fractal || shape.size() != 2 || valid.size() != 2) {
    return failure();
  }
  llvm::StringRef tileKind;
  switch (memory.getAddressSpace()) {
  case AddressSpace::LEFT: tileKind = "Left"; break;
  case AddressSpace::RIGHT: tileKind = "Right"; break;
  case AddressSpace::ACC: tileKind = "Acc"; break;
  case AddressSpace::MAT: tileKind = "Mat"; break;
  case AddressSpace::VEC: tileKind = "Vec"; break;
  case AddressSpace::BIAS: tileKind = "Bias"; break;
  case AddressSpace::SCALING: tileKind = "Scaling"; break;
  default: return failure();
  }
  llvm::StringRef blockLayout = bLayout.getInt() == 0 ? "RowMajor" : "ColMajor";
  std::string token = "pto::Tile<pto::TileType::" + tileKind.str() + ", " +
      buildBridgeElementTypeToken(element.getValue()) + ", " +
      std::to_string(shape[0]) + ", " + std::to_string(shape[1]) +
      ", pto::BLayout::" + blockLayout.str() + ", " +
      std::to_string(valid[0]) + ", " + std::to_string(valid[1]);
  if (sLayout.getInt() != 0) {
    llvm::StringRef storageLayout =
        sLayout.getInt() == 1 ? "RowMajor" : "ColMajor";
    token += ", pto::SLayout::" + storageLayout.str() + ", " +
             std::to_string(fractal.getInt());
  }
  return token + ">";
}

static FailureOr<std::string> renderCubeInstance(BridgeCallOp call,
                                                  llvm::StringRef symbol) {
  auto entryId = call->getAttrOfType<StringAttr>("entry_id");
  auto specAttr = call->getAttrOfType<BridgeCubeSpecAttr>("spec");
  DictionaryAttr spec = specAttr ? specAttr.getValue() : DictionaryAttr();
  if (!entryId || !spec) {
    return failure();
  }
  auto result = renderStructuredTile(spec.getAs<DictionaryAttr>("result_tile"));
  auto left = renderStructuredTile(spec.getAs<DictionaryAttr>("left_tile"));
  auto right = renderStructuredTile(spec.getAs<DictionaryAttr>("right_tile"));
  if (failed(result) || failed(left) || failed(right)) {
    return failure();
  }
  const BridgeFunctionDesc *desc =
      findBridgeFunctionById(entryId.getValue());
  if (!desc || desc->renderer != BridgeRendererKind::CubeDirect ||
      desc->callSpelling.empty()) {
    return failure();
  }
  StringRef callName = desc->callSpelling;
  callName.consume_front("pto::");
  std::string source;
  llvm::raw_string_ostream os(source);
  os << "#include <pto/pto-inst.hpp>\n#include <stdint.h>\n"
     << "#ifdef __DAV_CUBE__\n"
     << "extern \"C\" [aicore] void " << symbol
     << "(uint64_t dstAddress, uint64_t lhsAddress, uint64_t rhsAddress) {\n"
     << "  using ResultTile = " << *result << ";\n"
     << "  using LeftTile = " << *left << ";\n"
     << "  using RightTile = " << *right << ";\n"
     << "  ResultTile dst; LeftTile lhs; RightTile rhs;\n"
     << "  pto::TASSIGN_IMPL(dst, dstAddress);\n"
     << "  pto::TASSIGN_IMPL(lhs, lhsAddress);\n"
     << "  pto::TASSIGN_IMPL(rhs, rhsAddress);\n"
     << "  pto::" << callName << "(dst, lhs, rhs);\n}\n#endif\n";
  os.flush();
  return source;
}


struct ResolvedPipeSymbols {
  std::string init;
  std::string size;
  std::string push;
  std::string pop;
  std::string free;
};

static std::string pipeSymbol(BridgeEntryId id, unsigned instanceId) {
  const BridgeFunctionDesc *desc = findBridgeFunction(id);
  return desc->symbolBase.str() + "__" + std::to_string(instanceId);
}

static FailureOr<std::string>
renderPipeInstance(BridgeObjectCreateOp create,
                   const ResolvedPipeSymbols &symbols, unsigned instanceId,
                   ArrayRef<BridgeCallOp> calls) {
  auto spec = create->getAttrOfType<BridgePipeSpecAttr>("spec");
  if (!spec) {
    return create.emitError("resolved Pipe object has no structured spec");
  }
  DictionaryAttr fields = spec.getValue();
  auto pipe = fields.getAs<StringAttr>(kBridgeSpecPipeKey);
  auto split = fields.getAs<StringAttr>(kBridgeSpecSplitKey);
  auto producer = fields.getAs<StringAttr>(kBridgeSpecProducerTileKey);
  auto consumer = fields.getAs<StringAttr>(kBridgeSpecConsumerTileKey);
  if (!pipe) {
    return create.emitError("Pipe bridge spec is missing the pipe configuration");
  }

  bool needsPush = false;
  bool needsPop = false;
  bool needsFree = false;
  for (BridgeCallOp call : calls) {
    auto entryId = call->getAttrOfType<StringAttr>("entry_id");
    if (!entryId) {
      return call.emitError("Pipe bridge call is missing its logical entry ID");
    }
    needsPush |= entryId.getValue() == "pipe.push";
    needsPop |= entryId.getValue() == "pipe.pop";
    needsFree |= entryId.getValue() == "pipe.free";
  }
  if ((needsPush || needsPop || needsFree) && !split) {
    return create.emitError("Pipe bridge spec is missing the split axis");
  }
  if (needsPush && !producer) {
    return create.emitError("Pipe bridge spec is missing its producer tile");
  }
  if (needsPop && !consumer) {
    return create.emitError("Pipe bridge spec is missing its consumer tile");
  }

  func::FuncOp func = create->getParentOfType<func::FuncOp>();
  auto kind = func->getAttrOfType<FunctionKernelKindAttr>(
      FunctionKernelKindAttr::name);
  if (!kind || (kind.getKernelKind() != FunctionKernelKind::Cube &&
                kind.getKernelKind() != FunctionKernelKind::Vector)) {
    return create.emitError(
        "Pipe bridge instance requires a cube or vector kernel kind");
  }
  StringRef guard = kind.getKernelKind() == FunctionKernelKind::Cube
                        ? "__DAV_CUBE__"
                        : "__DAV_VEC__";
  std::string suffix = std::to_string(instanceId);
  std::string pipeType = "Pipe__" + suffix;
  std::string producerType = "ProducerTile__" + suffix;
  std::string consumerType = "ConsumerTile__" + suffix;

  std::string source;
  llvm::raw_string_ostream os(source);
  os << "using " << pipeType << " = " << pipe.getValue() << ";\n";
  if (producer) {
    os << "using " << producerType << " = " << producer.getValue() << ";\n";
  }
  if (consumer) {
    os << "using " << consumerType << " = " << consumer.getValue() << ";\n";
  }
  os << "extern \"C\" [aicore] void " << symbols.init
     << "(void *storage, uint32_t localBuffer) {\n"
     << "  new (storage) " << pipeType << "(nullptr, localBuffer, 0);\n}\n"
     << "extern \"C\" [aicore] size_t " << symbols.size
     << "() { return sizeof(" << pipeType << "); }\n"
     << "#ifdef " << guard << "\n";
  if (needsPush) {
    os << "extern \"C\" [aicore] void " << symbols.push
       << "(void *storage, uint64_t producerAddress) {\n"
       << "  auto &pipe = *reinterpret_cast<" << pipeType
       << " *>(storage);\n"
       << "  " << producerType << " tile;\n"
       << "  pto::TASSIGN_IMPL(tile, producerAddress);\n"
       << "  pto::TPUSH<" << pipeType << ", " << producerType << ", "
       << split.getValue() << ">(pipe, tile);\n}\n";
  }
  if (needsPop) {
    os << "extern \"C\" [aicore] uint64_t " << symbols.pop
       << "(void *storage) {\n"
       << "  auto &pipe = *reinterpret_cast<" << pipeType
       << " *>(storage);\n"
       << "  " << consumerType << " tile;\n"
       << "  pto::TPOP<" << pipeType << ", " << consumerType << ", "
       << split.getValue() << ">(pipe, tile);\n"
       << "  pipe_barrier(PIPE_ALL);\n"
       << "  return reinterpret_cast<uint64_t>(tile.data());\n}\n";
  }
  if (needsFree) {
    os << "extern \"C\" [aicore] void " << symbols.free
       << "(void *storage) {\n"
       << "  auto &pipe = *reinterpret_cast<" << pipeType
       << " *>(storage);\n"
       << "  pto::TFREE<" << pipeType << ", " << split.getValue()
       << ">(pipe);\n}\n";
  }
  os << "#endif\n";
  os.flush();
  return source;
}

static FailureOr<std::string> resolvePipeInstances(ModuleOp module) {
  std::string source;
  llvm::StringMap<std::pair<unsigned, ResolvedPipeSymbols>> instances;
  unsigned nextId = 0;
  bool failedResolve = false;
  module.walk([&](BridgeObjectCreateOp create) {
    auto entryId = create->getAttrOfType<StringAttr>("entry_id");
    if (!entryId || entryId.getValue() != "pipe.init") {
      return;
    }
    SmallVector<BridgeCallOp> calls;
    for (Operation *user : create.getResult().getUsers()) {
      auto call = dyn_cast<BridgeCallOp>(user);
      if (!call) {
        create.emitError("Pipe bridge object has a non-bridge lifecycle user");
        failedResolve = true;
        return;
      }
      calls.push_back(call);
    }
    llvm::sort(calls, [](BridgeCallOp lhs, BridgeCallOp rhs) {
      return lhs->isBeforeInBlock(rhs);
    });

    std::string canonical;
    llvm::raw_string_ostream keyOS(canonical);
    create->getAttr("spec").print(keyOS);
    create->getParentOfType<func::FuncOp>()
        ->getAttr(FunctionKernelKindAttr::name)
        .print(keyOS);
    for (BridgeCallOp call : calls) {
      call->getAttr("entry_id").print(keyOS);
    }
    keyOS.flush();

    auto found = instances.find(canonical);
    if (found == instances.end()) {
      const unsigned instanceId = nextId++;
      ResolvedPipeSymbols newSymbols{
          pipeSymbol(BridgeEntryId::PipeInit, instanceId),
          pipeSymbol(BridgeEntryId::PipeSize, instanceId),
          pipeSymbol(BridgeEntryId::PipePush, instanceId),
          pipeSymbol(BridgeEntryId::PipePop, instanceId),
          pipeSymbol(BridgeEntryId::PipeFree, instanceId)};
      FailureOr<std::string> rendered =
          renderPipeInstance(create, newSymbols, instanceId, calls);
      if (failed(rendered)) {
        failedResolve = true;
        return;
      }
      source += *rendered;
      found = instances
                  .try_emplace(canonical, instanceId, std::move(newSymbols))
                  .first;
    }
    const ResolvedPipeSymbols &symbols = found->second.second;
    create->setAttr("entry", StringAttr::get(module.getContext(), symbols.init));
    create->setAttr("size_callee",
                    StringAttr::get(module.getContext(), symbols.size));
    for (BridgeCallOp call : calls) {
      StringRef id = call->getAttrOfType<StringAttr>("entry_id").getValue();
      StringRef symbol;
      if (id == "pipe.push") {
        symbol = symbols.push;
      } else if (id == "pipe.pop") {
        symbol = symbols.pop;
      } else if (id == "pipe.free") {
        symbol = symbols.free;
      } else {
        call.emitError("unsupported Pipe bridge lifecycle entry");
        failedResolve = true;
        continue;
      }
      call.setCalleeAttr(StringAttr::get(module.getContext(), symbol));
    }
  });
  if (failedResolve) {
    return failure();
  }
  if (!source.empty()) {
    source.insert(0,
        "// Generated by ptoas (pto-emit-vpto-bridge-wrapper). Do not edit.\n"
        "#include <pto/pto-inst.hpp>\n"
        "#include <pto/npu/a5/TFree.hpp>\n"
        "#include <pto/npu/a5/TPop.hpp>\n"
        "#include <pto/npu/a5/TPush.hpp>\n"
        "#include <stddef.h>\n#include <stdint.h>\n"
        "[aicore] inline void *operator new(size_t, void *ptr) noexcept { "
        "return ptr; }\n");
  }
  return source;
}

static FailureOr<std::string> resolveCubeInstances(ModuleOp module) {
  llvm::StringMap<std::pair<std::string, std::string>> instances;
  std::string source;
  unsigned nextId = 0;
  bool failedRender = false;
  module.walk([&](BridgeCallOp call) {
    auto key = call->getAttrOfType<StringAttr>("instance_key");
    auto entryId = call->getAttrOfType<StringAttr>("entry_id");
    if (!key || !entryId || !entryId.getValue().starts_with("cube.")) {
      return;
    }
    auto found = instances.find(key.getValue());
    if (found == instances.end()) {
      const BridgeFunctionDesc *desc =
          findBridgeFunctionBySymbol(call.getCallee());
      if (!desc) {
        failedRender = true;
        return;
      }
      std::string symbol = desc->symbolBase.str() + "__" +
                           std::to_string(nextId++);
      auto rendered = renderCubeInstance(call, symbol);
      if (failed(rendered)) {
        failedRender = true;
        return;
      }
      found = instances.try_emplace(key.getValue(), symbol, *rendered).first;
      source += found->second.second;
    }
    call.setCalleeAttr(StringAttr::get(module.getContext(), found->second.first));
  });
  if (failedRender) {
    module.emitError("cannot resolve structured Cube bridge instance");
    return failure();
  }
  return source;
}

struct VPTOBridgeWrapperGenPass final
    : public impl::VPTOBridgeWrapperGenBase<VPTOBridgeWrapperGenPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VPTOBridgeWrapperGenPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    FailureOr<std::string> pipeSource = resolvePipeInstances(module);
    if (failed(pipeSource)) {
      signalPassFailure();
      return;
    }
    FailureOr<std::string> cubeSource = resolveCubeInstances(module);
    if (failed(cubeSource)) {
      signalPassFailure();
      return;
    }
    std::string source = *pipeSource + *cubeSource;
    if (source.empty()) {
      return;
    }
    OpBuilder builder(module);
    module->setAttr(kBridgeWrapperSourceAttrName,
                    builder.getStringAttr(source));
  }
};

} // namespace

std::unique_ptr<Pass> createVPTOBridgeWrapperGenPass() {
  return std::make_unique<VPTOBridgeWrapperGenPass>();
}

} // namespace pto
} // namespace mlir
