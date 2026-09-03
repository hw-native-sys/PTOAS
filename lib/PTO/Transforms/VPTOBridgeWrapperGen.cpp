// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

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
#include "llvm/ADT/StringSet.h"
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
  case AddressSpace::LEFT:
    tileKind = "Left";
    break;
  case AddressSpace::RIGHT:
    tileKind = "Right";
    break;
  case AddressSpace::ACC:
    tileKind = "Acc";
    break;
  case AddressSpace::MAT:
    tileKind = "Mat";
    break;
  case AddressSpace::VEC:
    tileKind = "Vec";
    break;
  case AddressSpace::BIAS:
    tileKind = "Bias";
    break;
  case AddressSpace::SCALING:
    tileKind = "Scaling";
    break;
  default:
    return failure();
  }
  llvm::StringRef blockLayout = bLayout.getInt() == 0 ? "RowMajor" : "ColMajor";
  auto elementToken = buildBridgeElementTypeToken(element.getValue());
  if (failed(elementToken)) {
    return failure();
  }
  std::string token =
      "pto::Tile<pto::TileType::" + tileKind.str() + ", " + *elementToken +
      ", " + std::to_string(shape[0]) + ", " + std::to_string(shape[1]) +
      ", pto::BLayout::" + blockLayout.str() + ", " + std::to_string(valid[0]) +
      ", " + std::to_string(valid[1]);
  if (sLayout.getInt() != 0) {
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
  StringAttr entryId = call.getEntryAttr();
  auto specAttr = dyn_cast_or_null<BridgeCubeSpecAttr>(call.getSpecAttr());
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
  const BridgeFunctionDesc *desc = findBridgeFunctionById(entryId.getValue());
  if (!desc || desc->core != BridgeCoreKind::Cube ||
      desc->renderer != BridgeRendererKind::CubeDirect ||
      desc->callSpelling.empty()) {
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
     << "extern \"C\" [aicore] void " << symbol
     << "(uint64_t dstAddress, uint64_t lhsAddress, uint64_t rhsAddress) {\n"
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

static FailureOr<std::string>
renderPipeInstance(BridgeObjectCreateOp create,
                   const ResolvedPipeSymbols &symbols, unsigned instanceId,
                   ArrayRef<BridgeCallOp> calls) {
  auto spec = dyn_cast_or_null<BridgePipeSpecAttr>(create.getSpecAttr());
  if (!spec) {
    return create.emitError("resolved Pipe object has no structured spec");
  }
  DictionaryAttr fields = spec.getValue();
  auto pipeConfig = fields.getAs<DictionaryAttr>(kBridgeSpecPipeKey);
  auto splitAttr = fields.getAs<IntegerAttr>(kBridgeSpecSplitKey);
  auto producer = fields.getAs<DictionaryAttr>(kBridgeSpecProducerTileKey);
  auto consumer = fields.getAs<DictionaryAttr>(kBridgeSpecConsumerTileKey);
  auto pipe = pipeConfig ? renderPipeConfig(pipeConfig)
                         : FailureOr<std::string>(failure());
  auto split = renderPipeSplit(splitAttr);
  bool hasValidPipeConfig = pipeConfig && succeeded(pipe);
  if (!hasValidPipeConfig) {
    return create.emitError(
        "Pipe bridge spec is missing structured pipe configuration");
  }

  bool needsPush = false;
  bool needsPop = false;
  bool needsFree = false;
  for (BridgeCallOp call : calls) {
    StringAttr entryId = call.getEntryAttr();
    if (!entryId) {
      return call.emitError("Pipe bridge call is missing its logical entry ID");
    }
    needsPush |= entryId.getValue() == "pipe.push";
    needsPop |= entryId.getValue() == "pipe.pop";
    needsFree |= entryId.getValue() == "pipe.free";
  }
  bool needsSplit = needsPush || needsPop || needsFree;
  if (needsSplit && failed(split)) {
    return create.emitError("Pipe bridge spec is missing the split axis");
  }
  if (needsPush && !producer) {
    return create.emitError("Pipe bridge spec is missing its producer tile");
  }
  if (needsPop && !consumer) {
    return create.emitError("Pipe bridge spec is missing its consumer tile");
  }

  func::FuncOp func = create->getParentOfType<func::FuncOp>();
  auto kind =
      func->getAttrOfType<FunctionKernelKindAttr>(FunctionKernelKindAttr::name);
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
  auto producerToken = producer ? renderStructuredTile(producer)
                                : FailureOr<std::string>(failure());
  auto consumerToken = consumer ? renderStructuredTile(consumer)
                                : FailureOr<std::string>(failure());
  bool invalidProducer = needsPush && failed(producerToken);
  bool invalidConsumer = needsPop && failed(consumerToken);
  if (invalidProducer || invalidConsumer) {
    return create.emitError(
        "Pipe bridge spec contains an invalid structured tile");
  }
  os << "using " << pipeType << " = " << *pipe << ";\n";
  if (producer) {
    os << "using " << producerType << " = " << *producerToken << ";\n";
  }
  if (consumer) {
    os << "using " << consumerType << " = " << *consumerToken << ";\n";
  }
  os << "extern \"C\" [aicore] void " << symbols.init
     << "(void *storage, uint32_t localBuffer) {\n"
     << "  new (storage) " << pipeType << "(nullptr, localBuffer, 0);\n}\n"
     << "extern \"C\" [aicore] size_t " << symbols.size << "() { return sizeof("
     << pipeType << "); }\n"
     << "#ifdef " << guard << "\n";
  if (needsPush) {
    os << "extern \"C\" [aicore] void " << symbols.push
       << "(void *storage, uint64_t producerAddress) {\n"
       << "  auto &pipe = *reinterpret_cast<" << pipeType << " *>(storage);\n"
       << "  " << producerType << " tile;\n"
       << "  pto::TASSIGN_IMPL(tile, producerAddress);\n"
       << "  pto::TPUSH<" << pipeType << ", " << producerType << ", " << *split
       << ">(pipe, tile);\n}\n";
  }
  if (needsPop) {
    os << "extern \"C\" [aicore] uint64_t " << symbols.pop
       << "(void *storage) {\n"
       << "  auto &pipe = *reinterpret_cast<" << pipeType << " *>(storage);\n"
       << "  " << consumerType << " tile;\n"
       << "  pto::TPOP<" << pipeType << ", " << consumerType << ", " << *split
       << ">(pipe, tile);\n"
       << "  pipe_barrier(PIPE_ALL);\n"
       << "  return reinterpret_cast<uint64_t>(tile.data());\n}\n";
  }
  if (needsFree) {
    os << "extern \"C\" [aicore] void " << symbols.free << "(void *storage) {\n"
       << "  auto &pipe = *reinterpret_cast<" << pipeType << " *>(storage);\n"
       << "  pto::TFREE<" << pipeType << ", " << *split << ">(pipe);\n}\n";
  }
  os << "#endif\n";
  os.flush();
  return source;
}

static FailureOr<std::string> renderPipeInstances(ModuleOp module) {
  std::string source;
  llvm::StringSet<> rendered;
  unsigned nextId = 0;
  bool failedRender = false;
  module.walk([&](BridgeObjectCreateOp create) {
    bool isPipeInit = create.getEntry() == "pipe.init";
    if (!isPipeInit) {
      return;
    }
    StringAttr key = create.getInstanceKeyAttr();
    StringAttr init = create.getCalleeAttr();
    StringAttr size = create.getSizeCalleeAttr();
    if (!key || !init || !size) {
      create.emitError("Pipe bridge instance has not been resolved");
      failedRender = true;
      return;
    }
    if (!rendered.insert(key.getValue()).second) {
      return;
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
        return;
      }
      calls.push_back(call);
      bool isPush = call.getEntry() == "pipe.push";
      if (isPush) {
        symbols.push = call.getCalleeAttr().getValue().str();
      } else if (call.getEntry() == "pipe.pop") {
        symbols.pop = call.getCalleeAttr().getValue().str();
      } else if (call.getEntry() == "pipe.free") {
        symbols.free = call.getCalleeAttr().getValue().str();
      }
    }
    FailureOr<std::string> renderedSource =
        renderPipeInstance(create, symbols, nextId++, calls);
    if (failed(renderedSource)) {
      failedRender = true;
      return;
    }
    source += *renderedSource;
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
    const BridgeFunctionDesc *desc = findBridgeFunctionById(call.getEntry());
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
