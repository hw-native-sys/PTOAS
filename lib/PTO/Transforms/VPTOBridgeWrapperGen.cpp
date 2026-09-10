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
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace mlir {
namespace pto {

#define GEN_PASS_DECL_VPTOBRIDGEWRAPPERGEN
#define GEN_PASS_DEF_VPTOBRIDGEWRAPPERGEN
#include "PTO/Transforms/Passes.h.inc"

namespace {

static FailureOr<llvm::StringRef> renderBridgeCType(BridgeValueKind kind) {
  switch (kind) {
  case BridgeValueKind::Pointer:
    return llvm::StringRef("void *");
  case BridgeValueKind::I32:
    return llvm::StringRef("uint32_t");
  case BridgeValueKind::I64:
    return llvm::StringRef("uint64_t");
  }
  return failure();
}

static FailureOr<std::string>
renderBridgeCSignature(const BridgeFunctionDesc &desc, StringRef symbol,
                       ArrayRef<StringRef> argumentNames) {
  bool invalidSignature = argumentNames.size() != desc.arguments.size() ||
                          desc.results.size() > 1;
  if (invalidSignature) {
    return failure();
  }
  FailureOr<StringRef> resultType =
      desc.results.empty() ? FailureOr<StringRef>(StringRef("void"))
                           : renderBridgeCType(desc.results.front());
  if (failed(resultType)) {
    return failure();
  }
  std::string signature;
  llvm::raw_string_ostream os(signature);
  os << *resultType << " " << symbol << "(";
  for (auto [index, name] : llvm::enumerate(argumentNames)) {
    auto argumentType = renderBridgeCType(desc.arguments[index]);
    if (failed(argumentType)) {
      return failure();
    }
    if (index != 0) {
      os << ", ";
    }
    os << *argumentType;
    if (!argumentType->ends_with("*")) {
      os << " ";
    }
    os << name;
  }
  os << ")";
  os.flush();
  return signature;
}

static FailureOr<llvm::StringRef> renderTileKind(AddressSpace space) {
  switch (space) {
  case AddressSpace::LEFT:
    return llvm::StringRef("Left");
  case AddressSpace::RIGHT:
    return llvm::StringRef("Right");
  case AddressSpace::ACC:
    return llvm::StringRef("Acc");
  case AddressSpace::MAT:
    return llvm::StringRef("Mat");
  case AddressSpace::VEC:
    return llvm::StringRef("Vec");
  case AddressSpace::BIAS:
    return llvm::StringRef("Bias");
  case AddressSpace::SCALING:
    return llvm::StringRef("Scaling");
  default:
    return failure();
  }
}

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
  auto tileKind = renderTileKind(memory.getAddressSpace());
  if (failed(tileKind)) {
    return failure();
  }
  llvm::StringRef blockLayout = bLayout.getInt() == 0 ? "RowMajor" : "ColMajor";
  auto elementToken = buildBridgeElementTypeToken(element.getValue());
  if (failed(elementToken)) {
    return failure();
  }
  std::string token =
      "pto::Tile<pto::TileType::" + tileKind->str() + ", " + *elementToken +
      ", " + std::to_string(shape[0]) + ", " + std::to_string(shape[1]) +
      ", pto::BLayout::" + blockLayout.str() + ", " + std::to_string(valid[0]) +
      ", " + std::to_string(valid[1]);
  const bool hasStorageLayout = sLayout.getInt() != 0;
  if (hasStorageLayout) {
    llvm::StringRef storageLayout =
        sLayout.getInt() == 1 ? "RowMajor" : "ColMajor";
    token += ", pto::SLayout::" + storageLayout.str() + ", " +
             std::to_string(fractal.getInt());
  }
  return token + ">";
}

static FailureOr<std::string> renderPipeConfig(DictionaryAttr config) {
  auto flag = config.getAs<IntegerAttr>("flag_base");
  auto dir = config.getAs<IntegerAttr>("dir_mask");
  auto slotSize = config.getAs<IntegerAttr>("slot_size");
  auto slotNum = config.getAs<IntegerAttr>("slot_num");
  auto localSlot = config.getAs<IntegerAttr>("local_slot_num");
  auto nosplit = config.getAs<BoolAttr>("nosplit");
  if (!flag || !dir || !slotSize || !slotNum || !localSlot || !nosplit) {
    return failure();
  }
  StringRef direction;
  switch (dir.getInt()) {
  case 1:
    direction = "C2V";
    break;
  case 2:
    direction = "V2C";
    break;
  case 3:
    direction = "BOTH";
    break;
  default:
    return failure();
  }
  return ("pto::TPipe<" + std::to_string(flag.getInt()) +
          ", pto::Direction::DIR_" + direction.str() + ", " +
          std::to_string(slotSize.getInt()) + ", " +
          std::to_string(slotNum.getInt()) + ", " +
          std::to_string(localSlot.getInt()) + ", " +
          (nosplit.getValue() ? "true>" : "false>"));
}

static FailureOr<std::string> renderPipeSplit(IntegerAttr split) {
  if (!split) {
    return failure();
  }
  switch (split.getInt()) {
  case 0:
    return std::string("pto::TileSplitAxis::TILE_NO_SPLIT");
  case 1:
    return std::string("pto::TileSplitAxis::TILE_UP_DOWN");
  case 2:
    return std::string("pto::TileSplitAxis::TILE_LEFT_RIGHT");
  case 3:
    return std::string("pto::TileSplitAxis::TILE_UP_DOWN_ODD");
  case 4:
    return std::string("pto::TileSplitAxis::TILE_LEFT_RIGHT_ODD");
  default:
    return failure();
  }
}

static FailureOr<std::string> renderCubeInstance(BridgeCallOp call,
                                                 llvm::StringRef symbol,
                                                 unsigned instanceId) {
  BridgeEntryId entryId = call.getEntry();
  auto specAttr = dyn_cast_or_null<BridgeCubeSpecAttr>(call.getSpecAttr());
  DictionaryAttr spec = specAttr ? specAttr.getValue() : DictionaryAttr();
  if (!spec) {
    return failure();
  }
  auto result = renderStructuredTile(spec.getAs<DictionaryAttr>("result_tile"));
  auto left = renderStructuredTile(spec.getAs<DictionaryAttr>("left_tile"));
  auto right = renderStructuredTile(spec.getAs<DictionaryAttr>("right_tile"));
  const bool hasInvalidTile =
      failed(result) || failed(left) || failed(right);
  if (hasInvalidTile) {
    return failure();
  }
  const BridgeFunctionDesc *desc = findBridgeFunction(entryId);
  if (!desc || desc->core != BridgeCoreKind::Cube ||
      desc->renderer != BridgeRendererKind::CubeDirect ||
      desc->callSpelling.empty()) {
    return failure();
  }
  auto signature = renderBridgeCSignature(
      *desc, symbol, {"dstAddress", "lhsAddress", "rhsAddress"});
  if (failed(signature)) {
    return failure();
  }
  StringRef callName = desc->callSpelling;
  callName.consume_front("pto::");
  std::string suffix = "__" + std::to_string(instanceId);
  std::string resultType = "ResultTile" + suffix;
  std::string leftType = "LeftTile" + suffix;
  std::string rightType = "RightTile" + suffix;
  std::string source;
  llvm::raw_string_ostream os(source);
  os << "#include <pto/pto-inst.hpp>\n#include <stdint.h>\n"
     << "#ifdef __DAV_CUBE__\n"
     << "extern \"C\" [aicore] " << *signature << " {\n"
     << "  using " << resultType << " = " << *result << ";\n"
     << "  using " << leftType << " = " << *left << ";\n"
     << "  using " << rightType << " = " << *right << ";\n"
     << "  " << resultType << " dst; " << leftType << " lhs; " << rightType
     << " rhs;\n"
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

struct PipeUsageFlags {
  bool needsPush = false;
  bool needsPop = false;
  bool needsFree = false;
};

static PipeUsageFlags collectPipeUsageFlags(ArrayRef<BridgeCallOp> calls) {
  PipeUsageFlags flags;
  for (BridgeCallOp call : calls) {
    BridgeEntryId entryId = call.getEntry();
    flags.needsPush |= entryId == BridgeEntryId::PipePush;
    flags.needsPop |= entryId == BridgeEntryId::PipePop;
    flags.needsFree |= entryId == BridgeEntryId::PipeFree;
  }
  return flags;
}

static FailureOr<llvm::StringRef>
getPipeInstanceGuard(BridgeObjectCreateOp create) {
  func::FuncOp func = create->getParentOfType<func::FuncOp>();
  auto kind =
      func->getAttrOfType<FunctionKernelKindAttr>(FunctionKernelKindAttr::name);
  const bool isSupportedKind =
      kind && (kind.getKernelKind() == FunctionKernelKind::Cube ||
               kind.getKernelKind() == FunctionKernelKind::Vector);
  if (!isSupportedKind) {
    return create.emitError(
        "Pipe bridge instance requires a cube or vector kernel kind");
  }
  return kind.getKernelKind() == FunctionKernelKind::Cube
             ? FailureOr<llvm::StringRef>(llvm::StringRef("__DAV_CUBE__"))
             : FailureOr<llvm::StringRef>(llvm::StringRef("__DAV_VEC__"));
}

struct PipeSignatures {
  std::string init;
  std::string size;
  std::string push;
  std::string pop;
  std::string free;
};

static FailureOr<PipeSignatures>
renderPipeInstanceSignatures(BridgeObjectCreateOp create,
                             const ResolvedPipeSymbols &symbols,
                             const PipeUsageFlags &flags) {
  const BridgeFunctionDesc *initDesc =
      findBridgeFunction(BridgeEntryId::PipeInit);
  const BridgeFunctionDesc *sizeDesc =
      findBridgeFunction(BridgeEntryId::PipeSize);
  const BridgeFunctionDesc *pushDesc =
      findBridgeFunction(BridgeEntryId::PipePush);
  const BridgeFunctionDesc *popDesc =
      findBridgeFunction(BridgeEntryId::PipePop);
  const BridgeFunctionDesc *freeDesc =
      findBridgeFunction(BridgeEntryId::PipeFree);
  if (!initDesc || !sizeDesc || !pushDesc || !popDesc || !freeDesc) {
    return create.emitError("Pipe bridge registry is incomplete");
  }
  auto initSignature = renderBridgeCSignature(
      *initDesc, symbols.init, {"storage", "localBuffer"});
  auto sizeSignature = renderBridgeCSignature(*sizeDesc, symbols.size, {});
  auto pushSignature = renderBridgeCSignature(
      *pushDesc, symbols.push, {"storage", "producerAddress"});
  auto popSignature =
      renderBridgeCSignature(*popDesc, symbols.pop, {"storage"});
  auto freeSignature =
      renderBridgeCSignature(*freeDesc, symbols.free, {"storage"});
  bool invalidSignature =
      failed(initSignature) || failed(sizeSignature) ||
      (flags.needsPush && failed(pushSignature)) ||
      (flags.needsPop && failed(popSignature)) ||
      (flags.needsFree && failed(freeSignature));
  if (invalidSignature) {
    return create.emitError("Pipe bridge registry has an invalid ABI");
  }
  PipeSignatures signatures;
  signatures.init = *initSignature;
  signatures.size = *sizeSignature;
  if (flags.needsPush) {
    signatures.push = *pushSignature;
  }
  if (flags.needsPop) {
    signatures.pop = *popSignature;
  }
  if (flags.needsFree) {
    signatures.free = *freeSignature;
  }
  return signatures;
}

static FailureOr<std::string> renderPipeInstanceBody(
    llvm::StringRef guard, const std::string &pipeType,
    const std::string &producerType, const std::string &consumerType,
    const PipeSignatures &signatures, const PipeUsageFlags &flags,
    llvm::StringRef pipeToken, llvm::StringRef producerToken,
    llvm::StringRef consumerToken, llvm::StringRef splitToken) {
  std::string source;
  llvm::raw_string_ostream os(source);
  os << "using " << pipeType << " = " << pipeToken << ";\n";
  if (!producerToken.empty()) {
    os << "using " << producerType << " = " << producerToken << ";\n";
  }
  if (!consumerToken.empty()) {
    os << "using " << consumerType << " = " << consumerToken << ";\n";
  }
  os << "extern \"C\" [aicore] " << signatures.init << " {\n"
     << "  new (storage) " << pipeType << "(nullptr, localBuffer, 0);\n}\n"
     << "extern \"C\" [aicore] " << signatures.size
     << " { return sizeof(" << pipeType << "); }\n"
     << "#ifdef " << guard << "\n";
  if (flags.needsPush) {
    os << "extern \"C\" [aicore] " << signatures.push << " {\n"
       << "  auto &pipe = *reinterpret_cast<" << pipeType << " *>(storage);\n"
       << "  " << producerType << " tile;\n"
       << "  pto::TASSIGN_IMPL(tile, producerAddress);\n"
       << "  pto::TPUSH<" << pipeType << ", " << producerType << ", "
       << splitToken << ">(pipe, tile);\n}\n";
  }
  if (flags.needsPop) {
    os << "extern \"C\" [aicore] " << signatures.pop << " {\n"
       << "  auto &pipe = *reinterpret_cast<" << pipeType << " *>(storage);\n"
       << "  " << consumerType << " tile;\n"
       << "  pto::TPOP<" << pipeType << ", " << consumerType << ", "
       << splitToken << ">(pipe, tile);\n"
       << "  pipe_barrier(PIPE_ALL);\n"
       << "  return reinterpret_cast<uint64_t>(tile.data());\n}\n";
  }
  if (flags.needsFree) {
    os << "extern \"C\" [aicore] " << signatures.free << " {\n"
       << "  auto &pipe = *reinterpret_cast<" << pipeType << " *>(storage);\n"
       << "  pto::TFREE<" << pipeType << ", " << splitToken << ">(pipe);\n}\n";
  }
  os << "#endif\n";
  os.flush();
  return source;
}

struct PipeSpecParts {
  DictionaryAttr config;
  IntegerAttr splitAttr;
  DictionaryAttr producer;
  DictionaryAttr consumer;
};

static FailureOr<PipeSpecParts>
getPipeInstanceSpecParts(BridgeObjectCreateOp create) {
  auto spec = dyn_cast_or_null<BridgePipeSpecAttr>(create.getSpecAttr());
  if (!spec) {
    return create.emitError("resolved Pipe object has no structured spec");
  }
  DictionaryAttr fields = spec.getValue();
  PipeSpecParts parts;
  parts.config = fields.getAs<DictionaryAttr>(kBridgeSpecPipeKey);
  parts.splitAttr = fields.getAs<IntegerAttr>(kBridgeSpecSplitKey);
  parts.producer = fields.getAs<DictionaryAttr>(kBridgeSpecProducerTileKey);
  parts.consumer = fields.getAs<DictionaryAttr>(kBridgeSpecConsumerTileKey);
  return parts;
}

struct PipeRenderTokens {
  std::string pipe;
  std::string split;
  std::string producer;
  std::string consumer;
};

static FailureOr<PipeRenderTokens>
renderPipeInstanceTokens(BridgeObjectCreateOp create,
                         const PipeSpecParts &parts,
                         const PipeUsageFlags &flags) {
  auto pipe = parts.config ? renderPipeConfig(parts.config)
                           : FailureOr<std::string>(failure());
  auto split = renderPipeSplit(parts.splitAttr);
  const bool hasValidPipeConfig = parts.config && succeeded(pipe);
  if (!hasValidPipeConfig) {
    return create.emitError(
        "Pipe bridge spec is missing structured pipe configuration");
  }
  const bool needsSplit = flags.needsPush || flags.needsPop || flags.needsFree;
  if (needsSplit && failed(split)) {
    return create.emitError("Pipe bridge spec is missing the split axis");
  }
  if (flags.needsPush && !parts.producer) {
    return create.emitError("Pipe bridge spec is missing its producer tile");
  }
  if (flags.needsPop && !parts.consumer) {
    return create.emitError("Pipe bridge spec is missing its consumer tile");
  }
  auto producerToken = parts.producer
                           ? renderStructuredTile(parts.producer)
                           : FailureOr<std::string>(failure());
  auto consumerToken = parts.consumer
                           ? renderStructuredTile(parts.consumer)
                           : FailureOr<std::string>(failure());
  const bool invalidProducer = flags.needsPush && failed(producerToken);
  const bool invalidConsumer = flags.needsPop && failed(consumerToken);
  if (invalidProducer || invalidConsumer) {
    return create.emitError(
        "Pipe bridge spec contains an invalid structured tile");
  }
  PipeRenderTokens tokens;
  tokens.pipe = *pipe;
  if (succeeded(split)) {
    tokens.split = *split;
  }
  if (parts.producer && succeeded(producerToken)) {
    tokens.producer = *producerToken;
  }
  if (parts.consumer && succeeded(consumerToken)) {
    tokens.consumer = *consumerToken;
  }
  return tokens;
}

static FailureOr<std::string>
renderPipeInstance(BridgeObjectCreateOp create,
                   const ResolvedPipeSymbols &symbols, unsigned instanceId,
                   ArrayRef<BridgeCallOp> calls) {
  FailureOr<PipeSpecParts> parts = getPipeInstanceSpecParts(create);
  if (failed(parts)) {
    return failure();
  }
  PipeUsageFlags flags = collectPipeUsageFlags(calls);
  FailureOr<PipeRenderTokens> tokens =
      renderPipeInstanceTokens(create, *parts, flags);
  if (failed(tokens)) {
    return failure();
  }
  FailureOr<llvm::StringRef> guard = getPipeInstanceGuard(create);
  if (failed(guard)) {
    return failure();
  }
  FailureOr<PipeSignatures> signatures =
      renderPipeInstanceSignatures(create, symbols, flags);
  if (failed(signatures)) {
    return failure();
  }
  std::string suffix = std::to_string(instanceId);
  std::string pipeType = "Pipe__" + suffix;
  std::string producerType = "ProducerTile__" + suffix;
  std::string consumerType = "ConsumerTile__" + suffix;
  return renderPipeInstanceBody(*guard, pipeType, producerType, consumerType,
                                *signatures, flags, tokens->pipe,
                                tokens->producer, tokens->consumer,
                                tokens->split);
}

static WalkResult renderPipeInstanceGroup(
    BridgeObjectCreateOp create, llvm::StringSet<> &rendered,
    unsigned &nextId, std::string &source, bool &failedRender) {
  bool isPipeInit = create.getEntry() == BridgeEntryId::PipeInit;
  if (!isPipeInit) {
    return WalkResult::advance();
  }
  StringAttr key = create.getInstanceKeyAttr();
  StringAttr init = create.getCalleeAttr();
  StringAttr size = create.getSizeCalleeAttr();
  if (!key || !init || !size) {
    create.emitError("Pipe bridge instance has not been resolved");
    failedRender = true;
    return WalkResult::advance();
  }
  if (!rendered.insert(key.getValue()).second) {
    return WalkResult::advance();
  }
  SmallVector<BridgeCallOp> calls;
  ResolvedPipeSymbols symbols;
  symbols.init = init.getValue().str();
  symbols.size = size.getValue().str();
  for (Operation *user : create.getResult().getUsers()) {
    auto call = dyn_cast<BridgeCallOp>(user);
    if (!call || !call.getCalleeAttr()) {
      create.emitError("Pipe bridge lifecycle has not been resolved");
      failedRender = true;
      return WalkResult::advance();
    }
    calls.push_back(call);
    bool isPush = call.getEntry() == BridgeEntryId::PipePush;
    if (isPush) {
      symbols.push = call.getCalleeAttr().getValue().str();
    } else if (call.getEntry() == BridgeEntryId::PipePop) {
      symbols.pop = call.getCalleeAttr().getValue().str();
    } else if (call.getEntry() == BridgeEntryId::PipeFree) {
      symbols.free = call.getCalleeAttr().getValue().str();
    }
  }
  FailureOr<std::string> renderedSource =
      renderPipeInstance(create, symbols, nextId++, calls);
  if (failed(renderedSource)) {
    failedRender = true;
    return WalkResult::advance();
  }
  source += *renderedSource;
  return WalkResult::advance();
}

static FailureOr<std::string> renderPipeInstances(ModuleOp module) {
  std::string source;
  llvm::StringSet<> rendered;
  unsigned nextId = 0;
  bool failedRender = false;
  module.walk([&](BridgeObjectCreateOp create) {
    return renderPipeInstanceGroup(create, rendered, nextId, source,
                                   failedRender);
  });
  if (failedRender) {
    return failure();
  }
  if (!source.empty()) {
    source.insert(
        0,
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

static FailureOr<std::string> renderCubeInstances(ModuleOp module) {
  llvm::StringSet<> rendered;
  std::string source;
  unsigned nextId = 0;
  bool failedRender = false;
  module.walk([&](BridgeCallOp call) {
    const BridgeFunctionDesc *desc = findBridgeFunction(call.getEntry());
    if (!desc || desc->family != BridgeFamily::Cube) {
      return;
    }
    StringAttr key = call.getInstanceKeyAttr();
    StringAttr callee = call.getCalleeAttr();
    if (!key || !callee) {
      call.emitError("Cube bridge instance has not been resolved");
      failedRender = true;
      return;
    }
    if (!rendered.insert(key.getValue()).second) {
      return;
    }
    auto renderedSource = renderCubeInstance(call, callee.getValue(), nextId++);
    if (failed(renderedSource)) {
      failedRender = true;
      return;
    }
    source += *renderedSource;
  });
  if (failedRender) {
    module.emitError("cannot render structured Cube bridge instance");
    return failure();
  }
  return source;
}

struct VPTOBridgeWrapperGenPass final
    : public impl::VPTOBridgeWrapperGenBase<VPTOBridgeWrapperGenPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VPTOBridgeWrapperGenPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    FailureOr<std::string> pipeSource = renderPipeInstances(module);
    if (failed(pipeSource)) {
      signalPassFailure();
      return;
    }
    FailureOr<std::string> cubeSource = renderCubeInstances(module);
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
