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
// Renders each resolved bridge instance into C++ wrapper source. The extern
// "C" prototype comes from the registry descriptor; the renderer owns only
// the body and the template arguments. Sources are grouped into translation
// units by (renderer, core), which is what object emission compiles.
//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/VPTOBridgeRegistry.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VPTOBridgeTokens.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::pto {
#define GEN_PASS_DEF_VPTOBRIDGEWRAPPERGEN
#include "PTO/Transforms/Passes.h.inc"

namespace {
FailureOr<StringRef> renderValueType(BridgeValueType value) {
  switch (value.kind) {
  case BridgeValueKind::Pointer:
    return value.space == BridgePointerSpace::Global
               ? StringRef("__gm__ void *")
               : StringRef("void *");
  case BridgeValueKind::ObjectHandle:
    return StringRef("void *");
  case BridgeValueKind::F32:
    return StringRef("float");
  case BridgeValueKind::I32:
    return StringRef("uint32_t");
  case BridgeValueKind::I64:
    return StringRef("int64_t");
  case BridgeValueKind::U64:
    return StringRef("uint64_t");
  }
  return failure();
}

/// The prototype is a property of the descriptor, so every renderer and the
/// call lowering agree on it by construction.
FailureOr<std::string> renderPrototype(const BridgeFunctionDesc &desc,
                                       StringRef symbol,
                                       ArrayRef<StringRef> argumentNames) {
  const bool unsupportedShape = argumentNames.size() != desc.arguments.size() ||
                                desc.results.size() > 1;
  if (unsupportedShape) {
    return failure();
  }
  FailureOr<StringRef> resultType =
      desc.results.empty() ? FailureOr<StringRef>(StringRef("void"))
                           : renderValueType(desc.results.front());
  if (failed(resultType)) {
    return failure();
  }
  std::string prototype;
  llvm::raw_string_ostream os(prototype);
  os << *resultType << " " << symbol << "(";
  for (auto [index, name] : llvm::enumerate(argumentNames)) {
    FailureOr<StringRef> argumentType = renderValueType(desc.arguments[index]);
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
  return prototype;
}

FailureOr<StringRef> renderTileKind(AddressSpace space) {
  switch (space) {
  case AddressSpace::LEFT:
    return StringRef("Left");
  case AddressSpace::RIGHT:
    return StringRef("Right");
  case AddressSpace::ACC:
    return StringRef("Acc");
  case AddressSpace::VEC:
    return StringRef("Vec");
  case AddressSpace::MAT:
    return StringRef("Mat");
  default:
    return failure();
  }
}

// Tile specs are verified (rank 2, static shape, known kind and dtype).
FailureOr<std::string> renderTile(BridgeTileSpecAttr spec) {
  TileBufType tile = spec.getTile();
  auto memory = llvm::cast<AddressSpaceAttr>(tile.getMemorySpace());
  FailureOr<StringRef> kind = renderTileKind(memory.getAddressSpace());
  auto element = buildBridgeElementTypeToken(tile.getElementType());
  const bool unrenderable = failed(kind) || failed(element);
  if (unrenderable) {
    return failure();
  }
  const StringRef block = tile.getBLayoutValueI32() == 0 ? "RowMajor" : "ColMajor";
  std::string result;
  llvm::raw_string_ostream os(result);
  os << "pto::Tile<pto::TileType::" << *kind << ", " << *element << ", "
     << tile.getShape()[0] << ", " << tile.getShape()[1]
     << ", pto::BLayout::" << block << ", " << tile.getValidShape()[0]
     << ", " << tile.getValidShape()[1];
  const bool hasSecondaryLayout = tile.getSLayoutValueI32() != 0;
  if (hasSecondaryLayout) {
    os << ", pto::SLayout::"
       << (tile.getSLayoutValueI32() == 1 ? "RowMajor" : "ColMajor")
       << ", " << tile.getSFractalSizeI32();
  }
  os << ">";
  return result;
}

FailureOr<std::string> renderPipeType(BridgePipeSpecAttr spec)
{
    StringRef direction;
    switch (spec.getDirMask()) {
        case 1:
            direction = "DIR_C2V";
            break;
        case mlir::pto::kValue2:
            direction = "DIR_V2C";
            break;
        default:
            return failure();
    }
    std::string result;
    llvm::raw_string_ostream os(result);
    os << "pto::TPipe<" << spec.getFlagBase() << ", pto::Direction::" << direction << ", " << spec.getSlotSize() << ", "
       << spec.getSlotNum() << ", " << spec.getLocalSlotNum() << ", "
       << (spec.getNosplit().getValue() ? "true" : "false") << ">";
    return result;
}

FailureOr<StringRef> renderSplit(BridgePipeSpecAttr spec)
{
    if (!spec.getSplit()) {
        return failure();
    }
    switch (*spec.getSplit()) {
        case 0:
            return StringRef("pto::TileSplitAxis::TILE_NO_SPLIT");
        case 1:
            return StringRef("pto::TileSplitAxis::TILE_UP_DOWN");
        case mlir::pto::kValue2:
            return StringRef("pto::TileSplitAxis::TILE_LEFT_RIGHT");
        case mlir::pto::kValue3:
            return StringRef("pto::TileSplitAxis::TILE_UP_DOWN_ODD");
        case mlir::pto::kValue4:
            return StringRef("pto::TileSplitAxis::TILE_LEFT_RIGHT_ODD");
        default:
            return failure();
    }
}

/// One Pipe lifecycle entry. The object storage bound is checked here so an
/// oversized pto::TPipe fails at wrapper compile time, not at run time.
/// Pipe init: construct the object in the caller's storage and hold the
/// registry's bound to the real type at wrapper compile time.
FailureOr<std::string> renderPipeInit(const BridgeFunctionDesc &desc,
                                      StringRef symbol, StringRef pipeType,
                                      BridgePipeSpecAttr spec) {
  auto prototype = renderPrototype(desc, symbol, {"storage", "localBuffer"});
  if (failed(prototype)) {
    return failure();
  }
  std::string source;
  llvm::raw_string_ostream os(source);
  os << "extern \"C\" [aicore] " << *prototype << " {\n"
     << "  using Pipe = " << pipeType << ";\n"
     << "  static_assert(sizeof(Pipe) <= " << desc.storageSize
     << ", \"pto::TPipe exceeds the bridge object storage bound\");\n"
     << "  static_assert(alignof(Pipe) <= " << desc.storageAlign
     << ", \"pto::TPipe exceeds the bridge object alignment bound\");\n"
     << "  new (storage) Pipe(nullptr, "
        << (spec.getDirMask() == mlir::pto::kValue2 ? "0, localBuffer" : "localBuffer, 0")
     << ");\n}\n";
  return source;
}

FailureOr<std::string> renderPipePush(const BridgeFunctionDesc &desc,
                                      StringRef symbol, StringRef pipeType,
                                      BridgePipeSpecAttr spec,
                                      StringRef split) {
  auto prototype = renderPrototype(desc, symbol, {"storage", "tileAddress"});
  const bool unrenderable = failed(prototype) || !spec.getProducer();
  if (unrenderable) {
    return failure();
  }
  auto tile = renderTile(spec.getProducer());
  if (failed(tile)) {
    return failure();
  }
  std::string source;
  llvm::raw_string_ostream os(source);
  os << "extern \"C\" [aicore] " << *prototype << " {\n"
     << "  using Pipe = " << pipeType << ";\n"
     << "  using ProducerTile = " << *tile << ";\n"
     << "  auto &pipe = *reinterpret_cast<Pipe *>(storage);\n"
     << "  ProducerTile tile;\n"
     << "  pto::TASSIGN_IMPL(tile, tileAddress);\n"
     << "  pto::TPUSH<Pipe, ProducerTile, " << split << ">(pipe, tile);\n}\n";
  return source;
}

FailureOr<std::string> renderPipePop(const BridgeFunctionDesc &desc,
                                     StringRef symbol, StringRef pipeType,
                                     BridgePipeSpecAttr spec, StringRef split) {
  auto prototype = renderPrototype(desc, symbol, {"storage"});
  const bool unrenderable = failed(prototype) || !spec.getConsumer();
  if (unrenderable) {
    return failure();
  }
  auto tile = renderTile(spec.getConsumer());
  if (failed(tile)) {
    return failure();
  }
  std::string source;
  llvm::raw_string_ostream os(source);
  os << "extern \"C\" [aicore] " << *prototype << " {\n"
     << "  using Pipe = " << pipeType << ";\n"
     << "  using ConsumerTile = " << *tile << ";\n"
     << "  auto &pipe = *reinterpret_cast<Pipe *>(storage);\n"
     << "  ConsumerTile tile;\n"
     << "  pto::TPOP<Pipe, ConsumerTile, " << split << ">(pipe, tile);\n"
     << "  pipe_barrier(PIPE_ALL);\n"
     << "  return reinterpret_cast<uint64_t>(tile.data());\n}\n";
  return source;
}

FailureOr<std::string> renderPipeFree(const BridgeFunctionDesc &desc,
                                      StringRef symbol, StringRef pipeType,
                                      StringRef split) {
  auto prototype = renderPrototype(desc, symbol, {"storage"});
  if (failed(prototype)) {
    return failure();
  }
  std::string source;
  llvm::raw_string_ostream os(source);
  os << "extern \"C\" [aicore] " << *prototype << " {\n"
     << "  using Pipe = " << pipeType << ";\n"
     << "  auto &pipe = *reinterpret_cast<Pipe *>(storage);\n"
     << "  pto::TFREE<Pipe, " << split << ">(pipe);\n}\n";
  return source;
}

/// One Pipe lifecycle entry. The split axis is a property of the object, so
/// every entry but init needs it.
FailureOr<std::string> renderPipeInstance(BridgeInstanceOp instance,
                                          const BridgeFunctionDesc &desc) {
  auto spec = llvm::dyn_cast_or_null<BridgePipeSpecAttr>(instance.getSpecAttr());
  if (!spec) {
    return failure();
  }
  FailureOr<std::string> pipeType = renderPipeType(spec);
  if (failed(pipeType)) {
    return failure();
  }
  StringRef symbol = instance.getSymName();
  if (desc.id == BridgeEntryId::PipeInit) {
    return renderPipeInit(desc, symbol, *pipeType, spec);
  }
  FailureOr<StringRef> split = renderSplit(spec);
  if (failed(split)) {
    return failure();
  }
  switch (desc.id) {
  case BridgeEntryId::PipePush:
    return renderPipePush(desc, symbol, *pipeType, spec, *split);
  case BridgeEntryId::PipePop:
    return renderPipePop(desc, symbol, *pipeType, spec, *split);
  case BridgeEntryId::PipeFree:
    return renderPipeFree(desc, symbol, *pipeType, *split);
  default:
    return failure();
  }
}

/// The accumulation phase as a template argument, empty when unspecified.
StringRef renderAccPhase(AccPhaseAttr phase) {
  switch (phase.getValue()) {
  case AccPhase::Partial:
    return "<pto::AccPhase::Partial>";
  case AccPhase::Final:
    return "<pto::AccPhase::Final>";
  case AccPhase::Unspecified:
    break;
  }
  return StringRef();
}

/// A direct Cube entry: assign each tile to its planned address, then call the
/// PTO-ISA instruction. The accumulate form carries one more tile.
FailureOr<std::string> renderCubeInstance(BridgeInstanceOp instance,
                                          const BridgeFunctionDesc &desc) {
  auto spec = llvm::dyn_cast_or_null<BridgeCubeSpecAttr>(instance.getSpecAttr());
  if (!spec) {
    return failure();
  }
  const bool accumulates = desc.id == BridgeEntryId::CubeTMatmulAcc;
  if (accumulates != static_cast<bool>(spec.getAccIn())) {
    return failure();
  }
  SmallVector<StringRef> argumentNames = {"dstAddress"};
  if (accumulates) {
    argumentNames.push_back("accAddress");
  }
  argumentNames.append({StringRef("lhsAddress"), StringRef("rhsAddress")});
  auto prototype = renderPrototype(desc, instance.getSymName(), argumentNames);
  auto dst = renderTile(spec.getDst());
  auto lhs = renderTile(spec.getLhs());
  auto rhs = renderTile(spec.getRhs());
  const bool unrenderable =
      failed(prototype) || failed(dst) || failed(lhs) || failed(rhs);
  if (unrenderable) {
    return failure();
  }
  const StringRef phase = renderAccPhase(spec.getAccPhase());
  std::string source;
  llvm::raw_string_ostream os(source);
  os << "extern \"C\" [aicore] " << *prototype << " {\n"
     << "  using Dst = " << *dst << ";\n"
     << "  using Lhs = " << *lhs << ";\n"
     << "  using Rhs = " << *rhs << ";\n";
  if (!accumulates) {
    os << "  Dst dst; Lhs lhs; Rhs rhs;\n"
       << "  pto::TASSIGN_IMPL(dst, dstAddress);\n"
       << "  pto::TASSIGN_IMPL(lhs, lhsAddress);\n"
       << "  pto::TASSIGN_IMPL(rhs, rhsAddress);\n"
       << "  pto::TMATMUL" << phase << "(dst, lhs, rhs);\n}\n";
    return source;
  }
  auto accIn = renderTile(spec.getAccIn());
  if (failed(accIn)) {
    return failure();
  }
  os << "  using AccIn = " << *accIn << ";\n"
     << "  Dst dst; Lhs lhs; Rhs rhs; AccIn accIn;\n"
     << "  pto::TASSIGN_IMPL(dst, dstAddress);\n"
     << "  pto::TASSIGN_IMPL(accIn, accAddress);\n"
     << "  pto::TASSIGN_IMPL(lhs, lhsAddress);\n"
     << "  pto::TASSIGN_IMPL(rhs, rhsAddress);\n"
     << "  pto::TMATMUL_ACC" << phase << "(dst, accIn, lhs, rhs);\n}\n";
  return source;
}

/// A print value shim: cce::printf takes the format bytes opaquely, so each
/// entry only has to forward its value in the right carrier type.
FailureOr<std::string> renderPrintValueShim(const BridgeFunctionDesc &desc,
                                            StringRef symbol,
                                            StringRef valueExpression) {
  auto prototype = renderPrototype(desc, symbol, {"fmt", "value"});
  if (failed(prototype)) {
    return failure();
  }
  std::string source;
  llvm::raw_string_ostream os(source);
  os << "extern \"C\" [aicore] __attribute__((always_inline)) " << *prototype
     << " {\n  cce::printf((__gm__ const char *)fmt, " << valueExpression
     << ");\n}\n";
  return source;
}

/// A DebugTunnel lifecycle shim. init also writes the fix-stack object that
/// cce::printf's instance lookup reads back.
FailureOr<std::string> renderPrintLifecycleShim(const BridgeFunctionDesc &desc,
                                                StringRef symbol,
                                                bool initializes) {
  auto prototype = renderPrototype(desc, symbol, {"dtData"});
  if (failed(prototype)) {
    return failure();
  }
  std::string source;
  llvm::raw_string_ostream os(source);
  os << "extern \"C\" [aicore] __attribute__((always_inline)) " << *prototype
     << " {\n";
  if (initializes) {
    os << "  cce::internal::DebugTunnel::OnKernelInitialize(\n"
       << "      (__gm__ cce::internal::DebugTunnelData *)dtData);\n"
       << "  *(uint64_t *)pto_vpto_print_fix_stack_object() = (uint64_t)dtData;\n}\n";
    return source;
  }
  os << "  cce::internal::DebugTunnel::OnKernelFinish(\n"
     << "      (__gm__ cce::internal::DebugTunnelData *)dtData);\n}\n";
  return source;
}

/// The VPTO emitter cannot instantiate the CCE print templates, so each print
/// entry becomes a shim that forwards to cce::printf. The host DebugTunnel
/// decoder is what reads the format string.
FailureOr<std::string> renderPrintInstance(BridgeInstanceOp instance,
                                           const BridgeFunctionDesc &desc) {
  StringRef symbol = instance.getSymName();
  switch (desc.id) {
  case BridgeEntryId::PrintString: {
    auto prototype = renderPrototype(desc, symbol, {"fmt"});
    if (failed(prototype)) {
      return failure();
    }
    std::string source;
    llvm::raw_string_ostream os(source);
    os << "extern \"C\" [aicore] __attribute__((always_inline)) " << *prototype
       << " {\n  cce::printf((__gm__ const char *)fmt);\n}\n";
    return source;
  }
  case BridgeEntryId::PrintF32:
  case BridgeEntryId::PrintI64:
    return renderPrintValueShim(desc, symbol, "value");
  case BridgeEntryId::PrintU64:
    // cce::printf has no 64-bit unsigned overload, so the value travels as
    // long long: the record carries the same eight bytes and the host format
    // string reinterprets them.
    return renderPrintValueShim(desc, symbol, "(long long)value");
  case BridgeEntryId::PrintInit:
    return renderPrintLifecycleShim(desc, symbol, /*initializes=*/true);
  case BridgeEntryId::PrintFinish:
    return renderPrintLifecycleShim(desc, symbol, /*initializes=*/false);
  default:
    return failure();
  }
}

StringRef stringifyRenderer(BridgeRendererKind renderer) {
  switch (renderer) {
  case BridgeRendererKind::Pipe:
    return "pipe";
  case BridgeRendererKind::CubeDirect:
    return "cube_direct";
  case BridgeRendererKind::Print:
    return "print";
  }
  return "<unknown>";
}

/// Includes and device-local declarations a translation unit needs. The print
/// marker is also what object emission looks for when it adds the CCE print
/// compile flags.
std::string renderPreamble(BridgeRendererKind renderer) {
  switch (renderer) {
  case BridgeRendererKind::Pipe:
    // The pipe lifecycle headers are not pulled in by pto-inst.hpp, and the
    // device compile has no <new>, so placement new is declared here.
    return "#include <pto/pto-inst.hpp>\n"
           "#include <pto/npu/a5/TFree.hpp>\n"
           "#include <pto/npu/a5/TPop.hpp>\n"
           "#include <pto/npu/a5/TPush.hpp>\n"
           "#include <stddef.h>\n"
           "#include <stdint.h>\n"
           "[aicore] inline void *operator new(size_t, void *storage) noexcept"
           " { return storage; }\n";
  case BridgeRendererKind::Print:
    return "// VPTO print bridge wrapper: cce::printf shims.\n"
           "#include \"ccelib/__ccelib.h\"\n"
           "#include <stddef.h>\n"
           "#include <stdint.h>\n"
           "extern \"C\" [aicore] void *pto_vpto_print_fix_stack_object()\n"
           "    asm(\"llvm.hivm.get.sycl.fix.stack.object\");\n";
  case BridgeRendererKind::CubeDirect:
    return "#include <pto/pto-inst.hpp>\n#include <stdint.h>\n";
  }
  return std::string();
}

/// Dispatches one instance to the renderer its entry declares.
FailureOr<std::string> renderInstance(BridgeInstanceOp instance,
                                      const BridgeFunctionDesc &desc) {
  switch (desc.renderer) {
  case BridgeRendererKind::Pipe:
    return renderPipeInstance(instance, desc);
  case BridgeRendererKind::CubeDirect:
    return renderCubeInstance(instance, desc);
  case BridgeRendererKind::Print:
    return renderPrintInstance(instance, desc);
  }
  return failure();
}

/// Translation units are keyed by (renderer, core); MapVector keeps the
/// emission order deterministic.
using WrapperUnits = llvm::MapVector<std::pair<unsigned, unsigned>, std::string>;

Attribute buildWrapperUnitAttr(MLIRContext *context,
                               std::pair<unsigned, unsigned> key,
                               StringRef body) {
  const auto renderer = static_cast<BridgeRendererKind>(key.first);
  const auto core = static_cast<BridgeCoreKind>(key.second);
  SmallVector<NamedAttribute> fields = {
      NamedAttribute(StringAttr::get(context, "renderer"),
                     StringAttr::get(context, stringifyRenderer(renderer))),
      NamedAttribute(StringAttr::get(context, "core"),
                     BridgeCoreKindAttr::get(context, core)),
      NamedAttribute(StringAttr::get(context, "source"),
                     StringAttr::get(context, renderPreamble(renderer) +
                                                  body.str()))};
  return DictionaryAttr::get(context, fields);
}

struct WrapperGen final : impl::VPTOBridgeWrapperGenBase<WrapperGen> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WrapperGen)

  LogicalResult collectUnits(ModuleOp module, WrapperUnits &units) const {
    for (BridgeInstanceOp instance : module.getOps<BridgeInstanceOp>()) {
      const BridgeFunctionDesc *desc = findBridgeFunction(instance.getEntry());
      if (!desc) {
        return instance.emitError("bridge instance has no registered entry");
      }
      FailureOr<std::string> rendered = renderInstance(instance, *desc);
      if (failed(rendered)) {
        return instance.emitError()
               << "cannot render bridge instance for the "
               << stringifyRenderer(desc->renderer) << " renderer";
      }
      auto key = std::make_pair(static_cast<unsigned>(desc->renderer),
                                static_cast<unsigned>(instance.getCore()));
      units[key] += *rendered;
    }
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    WrapperUnits units;
    if (failed(collectUnits(module, units))) {
      signalPassFailure();
      return;
    }
    if (units.empty()) {
      return;
    }
    SmallVector<Attribute> wrappers;
    for (auto &unit : units) {
      wrappers.push_back(
          buildWrapperUnitAttr(&getContext(), unit.first, unit.second));
    }
    module->setAttr(kBridgeWrapperSourceAttrName,
                    ArrayAttr::get(&getContext(), wrappers));
  }
};

} // namespace

std::unique_ptr<Pass> createVPTOBridgeWrapperGenPass() {
  return std::make_unique<WrapperGen>();
}
} // namespace mlir::pto
