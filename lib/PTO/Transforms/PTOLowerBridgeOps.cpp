// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// The one pass that routes operations into bridge IR. It owns the route
// policy: the policy is parsed once in initialize(), and the typed family
// lowerings below consume the parsed result instead of re-reading it.
#include "PTO/Transforms/Passes.h"
#include "PTO/IR/VPTOBridgeRegistry.h"
#include "PTO/Transforms/PrintEncoding.h"
#include "PTO/Transforms/Tf32ModeSupport.h"
#include "PTO/Transforms/VPTOBridgeWhitelist.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::pto {
#define GEN_PASS_DEF_PTOLOWERBRIDGEOPS
#include "PTO/Transforms/Passes.h.inc"

namespace {
/// Collects the planned address and the verified tile spec of each operand of
/// a direct Cube operation. Shared by the adapters below, which differ only in
/// which tiles they pass and in what the renderer calls.
LogicalResult collectCubeOperands(Operation *op, ArrayRef<Value> tiles,
                                  ArrayRef<StringRef> roles,
                                  SmallVectorImpl<Value> &addresses,
                                  SmallVectorImpl<BridgeTileSpecAttr> &specs) {
  auto emitError = [&] { return op->emitError(); };
  for (auto [index, operand] : llvm::enumerate(tiles)) {
    auto tile = dyn_cast<TileBufType>(operand.getType());
    auto alloc = operand.getDefiningOp<AllocTileOp>();
    const bool validAddress = tile && alloc && alloc.getAddr() &&
                              alloc.getAddr().getType().isInteger(64);
    if (!validAddress) {
      return op->emitError()
             << "matmul bridge " << roles[index]
             << " requires an alloc_tile with a planned i64 address";
    }
    auto spec = BridgeTileSpecAttr::getChecked(emitError, op->getContext(), tile);
    if (!spec) {
      return failure();
    }
    addresses.push_back(alloc.getAddr());
    specs.push_back(spec);
  }
  return success();
}

/// Preconditions shared by the Cube bridge adapters. The bridge ABI carries the
/// tile specs and the acc phase only, so a routed operation must be destination
/// style and must not ask for per-operation HF32 rounding: either would change
/// what the emitted instruction computes.
static LogicalResult checkCubeBridgePreconditions(Operation *op)
{
    const bool hasTensorResults = op->getNumResults() != 0;
    if (hasTensorResults) {
        return op->emitError(
            "matmul bridge requires destination-style tiles without tensor "
            "results");
    }
    if (op->hasAttr("tf32_mode")) {
        return op->emitOpError()
            << "tf32_mode is not supported by the Cube bridge; drop the "
            << "--vpto-bridge-policy route for this op or use "
            << "--pto-backend=emitc";
    }
    return success();
}

/// Direct Cube adapter: take the tiles and planned addresses from the typed
/// getters and build one structured call.
struct TMatmulBridgeAdapter {
    static LogicalResult lower(TMatmulOp op)
    {
        if (failed(checkCubeBridgePreconditions(op))) {
            return failure();
        }
        SmallVector<Value> addresses;
        SmallVector<BridgeTileSpecAttr> specs;
        const Value tiles[] = {op.getDst(), op.getLhs(), op.getRhs()};
        const StringRef roles[] = {"dst", "lhs", "rhs"};
        if (failed(collectCubeOperands(op, tiles, roles, addresses, specs))) {
            return failure();
        }
        auto spec = BridgeCubeSpecAttr::getChecked(
            [&] { return op.emitError(); }, op.getContext(), specs[0],
            specs[1], specs[2],
            AccPhaseAttr::get(op.getContext(), op.getAccPhase()),
            BridgeTileSpecAttr());
        if (!spec) {
            return failure();
        }
        OpBuilder builder(op);
        builder.create<BridgeCallOp>(op.getLoc(), TypeRange{},
            BridgeEntryId::CubeTMatmul, spec, nullptr, addresses);
        op.erase();
        return success();
    }
};

/// The accumulate form keeps its accumulator and destination apart, so it has
/// one more operand than the plain matmul and its own entry.
struct TMatmulAccBridgeAdapter {
    static LogicalResult lower(TMatmulAccOp op)
    {
        if (failed(checkCubeBridgePreconditions(op))) {
            return failure();
        }
        SmallVector<Value> addresses;
        SmallVector<BridgeTileSpecAttr> specs;
        // ABI order follows the PTO-ISA call: destination, accumulator, lhs, rhs.
        const Value tiles[] = {op.getDst(), op.getAccIn(), op.getLhs(),
            op.getRhs()};
        const StringRef roles[] = {"dst", "acc_in", "lhs", "rhs"};
        if (failed(collectCubeOperands(op, tiles, roles, addresses, specs))) {
            return failure();
        }
        auto spec = BridgeCubeSpecAttr::getChecked(
            [&] { return op.emitError(); }, op.getContext(), specs[0],
            specs[2], specs[3],
            AccPhaseAttr::get(op.getContext(), op.getAccPhase()), specs[1]);
        if (!spec) {
            return failure();
        }
        OpBuilder builder(op);
        builder.create<BridgeCallOp>(op.getLoc(), TypeRange{},
            BridgeEntryId::CubeTMatmulAcc, spec, nullptr, addresses);
        op.erase();
        return success();
    }
};

template <typename OpTy, typename Adapter>
LogicalResult lowerDirectBridgeOps(func::FuncOp func,
                                   const BridgeRoutePolicy &policy,
                                   StringRef family) {
  if (!policy.routesOp(family, OpTy::getOperationName())) {
    return success();
  }
  SmallVector<OpTy> operations;
  func.walk([&](OpTy op) { operations.push_back(op); });
  for (OpTy op : operations) {
    if (failed(Adapter::lower(op))) {
      return failure();
    }
  }
  return success();
}

/// Interns the print format strings of a module as private addrspace(1)
/// constants. The wrapper takes the bytes as an opaque stream, so one global
/// per distinct string is enough.
class PrintFormatInterner {
public:
  explicit PrintFormatInterner(ModuleOp module) : module(module) {}

  /// Returns the pointer to the interned copy of \p text, creating the global
  /// on first use. Globals go to the top of the module body so that every
  /// addressof below them refers to an already-declared symbol.
  Value getPointer(OpBuilder &builder, Location loc, StringRef text) {
    auto found = globals.find(text);
    if (found == globals.end()) {
      std::string name =
          ("pto_vpto_print_fmt_" + std::to_string(globals.size()));
      SmallVector<int8_t> bytes(text.begin(), text.end());
      bytes.push_back(0); // the host decoder reads a C string
      auto byteType = builder.getI8Type();
      auto arrayType =
          LLVM::LLVMArrayType::get(byteType, bytes.size());
      // The LLVM translation rejects a plain ArrayAttr initializer here.
      auto initType = RankedTensorType::get(
          {static_cast<int64_t>(bytes.size())}, byteType);
      OpBuilder moduleBuilder =
          OpBuilder::atBlockBegin(module.getBody());
      auto global = moduleBuilder.create<LLVM::GlobalOp>(
          module.getLoc(), arrayType, /*isConstant=*/true,
          LLVM::Linkage::Private, name,
          DenseIntElementsAttr::get(initType, bytes));
      global.setAddrSpace(1);
      found = globals.try_emplace(text, name).first;
    }
    return builder.create<LLVM::AddressOfOp>(
        loc, LLVM::LLVMPointerType::get(builder.getContext(), 1),
        found->second);
  }

private:
  ModuleOp module;
  llvm::StringMap<std::string> globals;
};

/// Widens one print value to the entry's carrier type and returns the entry.
/// The DebugTunnel record is fixed width, so f16/bf16 extend and f64 truncates
/// to f32, and narrow integers widen by the conversion's signedness.
FailureOr<std::pair<BridgeEntryId, Value>>
buildPrintValue(OpBuilder &builder, Location loc, Value scalar,
                const PrintFormatInfo &format, Operation *op) {
  Type type = scalar.getType();
  if (auto floatType = dyn_cast<FloatType>(type)) {
    if (format.conversion != PrintConversionKind::Float) {
      return op->emitError()
             << "print format expects an integer conversion but the value is "
             << type;
    }
    Value value = scalar;
    const bool needsExtend = floatType.isF16() || floatType.isBF16();
    if (needsExtend) {
      value = builder.create<LLVM::FPExtOp>(loc, builder.getF32Type(), scalar);
    } else if (!floatType.isF32()) {
      // f64 prints with f32 precision; the record carries four bytes.
      value =
          builder.create<LLVM::FPTruncOp>(loc, builder.getF32Type(), scalar);
    }
    return std::make_pair(BridgeEntryId::PrintF32, value);
  }
  auto intType = dyn_cast<IntegerType>(type);
  if (!intType) {
    return op->emitError() << "print does not support the value type " << type;
  }
  if (format.conversion == PrintConversionKind::Float) {
    return op->emitError()
           << "print format expects a float conversion but the value is "
           << type;
  }
  const bool isUnsigned =
      format.conversion == PrintConversionKind::UnsignedInt;
  const bool needsWideModifier = intType.getWidth() == 64;
  if (needsWideModifier && !format.hasWideLengthModifier) {
    return op->emitError()
           << "print format for a 64-bit integer needs a 64-bit length "
              "modifier, as in %lld";
  }
  Value value = scalar;
  const bool needsWidening = intType.getWidth() < 64;
  if (needsWidening) {
    Type wide = builder.getI64Type();
    value = isUnsigned
                ? builder.create<LLVM::ZExtOp>(loc, wide, scalar).getResult()
                : builder.create<LLVM::SExtOp>(loc, wide, scalar).getResult();
  }
  return std::make_pair(
      isUnsigned ? BridgeEntryId::PrintU64 : BridgeEntryId::PrintI64, value);
}

/// Literal records pto.tprint prints around its element values. The host
/// DebugTunnel decoder renders them as the tile header and shape lines.
std::string tprintDtypeName(Type element) {
  if (element.isF16()) {
    return "float16";
  }
  if (element.isF32()) {
    return "float32";
  }
  if (element.isBF16()) {
    return "bfloat16";
  }
  return "int" + std::to_string(element.getIntOrFloatBitWidth());
}

std::string tprintHeaderText(Type element) {
  return "=== [TPRINT Tile] Data Type: " + tprintDtypeName(element) +
         ", Layout: ND, TileType: Vec ===\n";
}

std::string tprintShapeText(ArrayRef<int64_t> shape, ArrayRef<int64_t> valid) {
  const std::string rows = std::to_string(shape[0]);
  const std::string cols = std::to_string(shape[1]);
  return "  Shape: [" + rows + ", " + cols + "], Valid Shape: [" +
         std::to_string(valid[0]) + ", " + std::to_string(valid[1]) + "]\n";
}

/// Element offset of the tile data base. When the module declares the system
/// address intrinsic the UB window is addressed through it; otherwise the tile
/// pointer stays at the plain VPTO placeholder base.
Value buildTprintBaseElementOffset(OpBuilder &builder, Location loc,
                                   ModuleOp module, int64_t elementBytes) {
  constexpr int64_t kUbBaseByteOffset = 0x80000;
  Type i64Type = builder.getI64Type();
  auto systemBase =
      module.lookupSymbol<LLVM::LLVMFuncOp>("llvm.hivm.GET.SYS.VA.BASE");
  if (!systemBase) {
    return builder.create<LLVM::ConstantOp>(loc, i64Type,
                                            builder.getI64IntegerAttr(0));
  }
  Value base = builder
                   .create<LLVM::CallOp>(loc, i64Type, systemBase.getSymName(),
                                         ValueRange{})
                   .getResult();
  Value window = builder.create<LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(kUbBaseByteOffset));
  Value address = builder.create<LLVM::AddOp>(loc, i64Type, base, window);
  Value elementSize = builder.create<LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(elementBytes));
  return builder.create<LLVM::UDivOp>(loc, i64Type, address, elementSize);
}

/// The literal header and shape records that frame a tile dump.
void emitTPrintLiteralRecords(OpBuilder &builder, Location loc,
                              PrintFormatInterner &formats, Type element,
                              ArrayRef<int64_t> shape, ArrayRef<int64_t> valid) {
  Value header = formats.getPointer(builder, loc, tprintHeaderText(element));
  builder.create<BridgeCallOp>(loc, TypeRange{}, BridgeEntryId::PrintString,
                               /*spec=*/nullptr, /*instance=*/nullptr,
                               ValueRange{header});
  Value shapeText = formats.getPointer(builder, loc, tprintShapeText(shape, valid));
  builder.create<BridgeCallOp>(loc, TypeRange{}, BridgeEntryId::PrintString,
                               /*spec=*/nullptr, /*instance=*/nullptr,
                               ValueRange{shapeText});
}

/// Widens one tile element to its print carrier, the same way a scalar print
/// of that type would.
std::pair<BridgeEntryId, Value> buildTPrintElementValue(OpBuilder &builder,
                                                        Location loc,
                                                        Type element,
                                                        Value value) {
  if (!isa<FloatType>(element)) {
    const bool needsWidening = element.getIntOrFloatBitWidth() < 64;
    if (needsWidening) {
      value = builder.create<LLVM::SExtOp>(loc, builder.getI64Type(), value);
    }
    return {BridgeEntryId::PrintI64, value};
  }
  auto floatType = cast<FloatType>(element);
  const bool needsExtend = floatType.isF16() || floatType.isBF16();
  if (needsExtend) {
    value = builder.create<LLVM::FPExtOp>(loc, builder.getF32Type(), value);
  }
  return {BridgeEntryId::PrintF32, value};
}

/// One value record per element, in a row/column loop nest over the tile.
void emitTPrintElementLoop(OpBuilder &builder, Location loc, ModuleOp module,
                           Type element, ArrayRef<int64_t> shape,
                           ArrayRef<int64_t> valid, Value address,
                           Value valueFormat) {
  MLIRContext *context = builder.getContext();
  Type i64Type = builder.getI64Type();
  Value baseOffset = buildTprintBaseElementOffset(
      builder, loc, module, element.getIntOrFloatBitWidth() / 8);
  auto tilePointerType = LLVM::LLVMPointerType::get(context, 6);
  Value tileBase = builder.create<LLVM::IntToPtrOp>(loc, tilePointerType, address);
  Value colCount = builder.create<LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(shape[1]));
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value rowBound = builder.create<arith::ConstantIndexOp>(loc, valid[0]);
  Value colBound = builder.create<arith::ConstantIndexOp>(loc, valid[1]);

  auto rowLoop = builder.create<scf::ForOp>(loc, zero, rowBound, one);
  OpBuilder::InsertionGuard rowGuard(builder);
  builder.setInsertionPointToStart(rowLoop.getBody());
  auto colLoop = builder.create<scf::ForOp>(loc, zero, colBound, one);
  OpBuilder::InsertionGuard colGuard(builder);
  builder.setInsertionPointToStart(colLoop.getBody());

  Value row = builder.create<arith::IndexCastOp>(loc, i64Type,
                                                 rowLoop.getInductionVar());
  Value col = builder.create<arith::IndexCastOp>(loc, i64Type,
                                                 colLoop.getInductionVar());
  Value rowOffset = builder.create<LLVM::MulOp>(loc, i64Type, row, colCount);
  Value elementIndex =
      builder.create<LLVM::AddOp>(loc, i64Type, rowOffset, col);
  Value offset =
      builder.create<LLVM::AddOp>(loc, i64Type, baseOffset, elementIndex);
  Value elementPointer = builder.create<LLVM::GEPOp>(
      loc, tilePointerType, element, tileBase, ValueRange{offset});
  Value loaded = builder.create<LLVM::LoadOp>(loc, element, elementPointer);
  auto [entry, value] =
      buildTPrintElementValue(builder, loc, element, loaded);
  builder.create<BridgeCallOp>(loc, TypeRange{}, entry, /*spec=*/nullptr,
                               /*instance=*/nullptr,
                               ValueRange{valueFormat, value});
}

/// pto.tprint expands to the header and shape records plus one value record
/// per element, all through the same fixed Print entries. It is a debug op, so
/// it does not participate in TileLib template expansion.
LogicalResult lowerTPrintOp(TPrintOp tprint, PrintFormatInterner &formats) {
  auto tile = dyn_cast<TileBufType>(tprint.getSrc().getType());
  const bool unsupportedSource = !tile || tile.getShape().size() != 2;
  if (unsupportedSource) {
    return tprint.emitError()
           << "tprint bridge supports only a rank-2 tile_buf source";
  }
  Type element = tile.getElementType();
  auto memory = dyn_cast_or_null<AddressSpaceAttr>(tile.getMemorySpace());
  const bool unsupportedLayout = !memory ||
      memory.getAddressSpace() != AddressSpace::VEC ||
      tile.getBLayoutValueI32() != 0 || tile.getSLayoutValueI32() != 0 ||
      tprint.getTmp() || tprint.getPrintFormatAttr();
  if (unsupportedLayout) {
    return tprint.emitError("tprint bridge supports only ND row-major Vec tiles without tmp or printFormat");
  }
  const bool supportedElement = element.isF16() || element.isBF16() ||
      element.isF32() || element.isSignlessInteger(8) ||
      element.isSignlessInteger(16) || element.isSignlessInteger(32);
  if (!supportedElement) {
    return tprint.emitError("tprint bridge has unsupported element type");
  }
  auto alloc = tprint.getSrc().getDefiningOp<AllocTileOp>();
  if (!alloc || !alloc.getAddr()) {
    return tprint.emitError("tprint bridge requires an alloc_tile with a planned i64 address");
  }
  ArrayRef<int64_t> valid = tile.getValidShape();
  if (llvm::any_of(valid, ShapedType::isDynamic)) {
    return tprint.emitError("tprint bridge requires static valid dimensions");
  }
  ArrayRef<int64_t> shape = tile.getShape();
  OpBuilder builder(tprint);
  Location loc = tprint.getLoc();
  emitTPrintLiteralRecords(builder, loc, formats, element, shape, valid);
  Value valueFormat = formats.getPointer(
      builder, loc, isa<FloatType>(element) ? "%6.2f" : "%6d");
  emitTPrintElementLoop(builder, loc, tprint->getParentOfType<ModuleOp>(),
                        element, shape, valid, alloc.getAddr(), valueFormat);
  tprint.erase();
  return success();
}

LogicalResult lowerPrintOp(PrintOp print, PrintFormatInterner &formats) {
  StringRef format = print.getFormat();
  FailureOr<PrintFormatInfo> info = analyzePrintFormat(format);
  if (failed(info)) {
    return print.emitError()
           << "print format \"" << format
           << "\" must hold exactly one supported conversion specifier";
  }
  OpBuilder builder(print);
  Location loc = print.getLoc();
  auto value = buildPrintValue(builder, loc, print.getScalar(), *info, print);
  if (failed(value)) {
    return failure();
  }
  Value formatPointer = formats.getPointer(builder, loc, format);
  // Print entries are fixed prototypes, so they carry no specialization.
  builder.create<BridgeCallOp>(loc, TypeRange{}, value->first,
                               /*spec=*/nullptr, /*instance=*/nullptr,
                               ValueRange{formatPointer, value->second});
  print.erase();
  return success();
}

/// Print family typed lowering. Print has no non-bridge VPTO path, so a
/// selected print op that cannot be lowered is an error, not a fallback.
LogicalResult lowerPrintFamilyOps(ModuleOp module,
                                  const BridgeRoutePolicy &policy) {
  const bool routesPrint =
      policy.routesOp("print", PrintOp::getOperationName());
  const bool routesTPrint =
      policy.routesOp("print", TPrintOp::getOperationName());
  SmallVector<PrintOp> prints;
  SmallVector<TPrintOp> tprints;
  module.walk([&](Operation *op) {
    if (auto print = dyn_cast<PrintOp>(op)) {
      if (routesPrint) {
        prints.push_back(print);
      }
    } else if (auto tprint = dyn_cast<TPrintOp>(op)) {
      if (routesTPrint) {
        tprints.push_back(tprint);
      }
    }
  });
  {
    // Report a print the policy excluded: unlike the other families, print has
    // no non-bridge VPTO lowering to fall back to, so leaving the op alone
    // would fail later with an unrelated translation error.
    WalkResult unroutable = module.walk([&](Operation *op) {
      const bool isUnroutedPrint =
          (isa<PrintOp>(op) && !routesPrint) ||
          (isa<TPrintOp>(op) && !routesTPrint);
      if (isUnroutedPrint) {
        op->emitError() << "the route policy excludes " << op->getName()
                        << ", which has no non-bridge VPTO lowering";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (unroutable.wasInterrupted()) {
      return failure();
    }
  }
  PrintFormatInterner formats(module);
  for (PrintOp print : prints) {
    if (failed(lowerPrintOp(print, formats))) {
      return failure();
    }
  }
  for (TPrintOp tprint : tprints) {
    if (failed(lowerTPrintOp(tprint, formats))) {
      return failure();
    }
  }
  return success();
}

/// A routed operation that no family pattern consumed would otherwise reach
/// emission as an unlowered op, or silently fall back to a different path.
LogicalResult verifyRoutedOpsWereLowered(ModuleOp module,
                                         const BridgeRoutePolicy &policy) {
  WalkResult result = module.walk([&](Operation *op) {
    // Only registry members can be routed, so unrelated ops such as
    // arith.constant or func.return are never reported here.
    const BridgeFunctionDesc *desc =
        findBridgeFunctionByOp(op->getName().getStringRef());
    if (!desc) {
      return WalkResult::advance();
    }
    const bool routed =
        policy.routesOp(stringifyBridgeFamily(desc->family),
                        op->getName().getStringRef());
    if (routed) {
      op->emitError() << "operation is routed to the "
                      << stringifyBridgeFamily(desc->family)
                      << " bridge but was not lowered to bridge IR";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

struct LowerBridgeOps final : impl::PTOLowerBridgeOpsBase<LowerBridgeOps> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerBridgeOps)

  LowerBridgeOps() = default;
  explicit LowerBridgeOps(const PTOLowerBridgeOpsOptions &options)
      : impl::PTOLowerBridgeOpsBase<LowerBridgeOps>(options) {}

  // The policy is a property of the compilation, not of a function: parse it
  // once here rather than per function body.
  LogicalResult initialize(MLIRContext *context) override {
    FailureOr<BridgeRoutePolicy> parsed =
        loadBridgeRoutePolicy(policyPath, llvm::errs());
    if (failed(parsed)) {
      return failure();
    }
    policy = *parsed;
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool lowered = succeeded(lowerPrintFamilyOps(module, policy));
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (!lowered) {
        break;
      }
      // Cube first: a direct Cube op consumes the tile's planned address, so
      // after it the tile's only user is the pipe lifecycle op, which lets the
      // Pipe lowering retire the tile. The other order leaves the Cube op
      // holding a tile the Pipe lowering is trying to erase.
      lowered = succeeded(lowerDirectBridgeOps<TMatmulOp, TMatmulBridgeAdapter>(
                    func, policy, "cube")) &&
                succeeded(
                    lowerDirectBridgeOps<TMatmulAccOp, TMatmulAccBridgeAdapter>(
                        func, policy, "cube")) &&
                succeeded(lowerPipeFamilyOps(func, policy));
    }
    if (lowered) {
      lowered = succeeded(verifyRoutedOpsWereLowered(module, policy));
    }
    if (!lowered) {
      signalPassFailure();
    }
  }

  BridgeRoutePolicy policy;
};
} // namespace

std::unique_ptr<Pass> createPTOLowerBridgeOpsPass(llvm::StringRef policyPath) {
  PTOLowerBridgeOpsOptions options;
  options.policyPath = policyPath.str();
  return std::make_unique<LowerBridgeOps>(options);
}
} // namespace mlir::pto
