//===- A5VMLLVMEmitter.cpp - A5VM to official LLVM IR text emitter -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/A5VMLLVMEmitter.h"

#include "PTO/IR/A5VM.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/HIVMIntrinsicNaming.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir::pto {
namespace {

struct QueriedTargetAttrs {
  std::string targetCPU;
  std::string targetFeatures;
};

struct ABIExpr {
  enum class Kind { Constant, FuncArg, Mul };

  Kind kind = Kind::Constant;
  uint64_t constant = 0;
  unsigned argIndex = 0;
  std::unique_ptr<ABIExpr> lhs;
  std::unique_ptr<ABIExpr> rhs;

  static ABIExpr constantExpr(uint64_t value) {
    ABIExpr expr;
    expr.kind = Kind::Constant;
    expr.constant = value;
    return expr;
  }

  static ABIExpr argExpr(unsigned argIndex) {
    ABIExpr expr;
    expr.kind = Kind::FuncArg;
    expr.argIndex = argIndex;
    return expr;
  }

  static ABIExpr mulExpr(ABIExpr lhs, ABIExpr rhs) {
    ABIExpr expr;
    expr.kind = Kind::Mul;
    expr.lhs = std::make_unique<ABIExpr>(std::move(lhs));
    expr.rhs = std::make_unique<ABIExpr>(std::move(rhs));
    return expr;
  }
};

struct ExternalMemRefABISpec {
  unsigned addressSpace = 1;
  int64_t rank = 0;
  ABIExpr offset = ABIExpr::constantExpr(0);
  ABIExpr totalSize = ABIExpr::constantExpr(1);
  ABIExpr stride = ABIExpr::constantExpr(1);
};

struct ExternalArgABISpec {
  bool isMemRef = false;
  ExternalMemRefABISpec memrefSpec;
};

struct FunctionABISpec {
  SmallVector<ExternalArgABISpec> args;
};

static Type getElementTypeFromVectorLike(Type type);

static std::optional<ABIExpr> buildABIExprFromValue(Value value);

static std::optional<ABIExpr> buildABIExprFromFoldResult(OpFoldResult ofr) {
  if (auto attr = ofr.dyn_cast<Attribute>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return ABIExpr::constantExpr(intAttr.getValue().getZExtValue());
    return std::nullopt;
  }
  return buildABIExprFromValue(ofr.get<Value>());
}

static std::optional<ABIExpr> buildABIExprFromValue(Value value) {
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    auto func = dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp());
    if (!func || blockArg.getOwner() != &func.getBody().front())
      return std::nullopt;
    return ABIExpr::argExpr(blockArg.getArgNumber());
  }

  if (auto constIndex = value.getDefiningOp<arith::ConstantIndexOp>())
    return ABIExpr::constantExpr(constIndex.value());
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return ABIExpr::constantExpr(intAttr.getValue().getZExtValue());
  }
  if (auto castOp = value.getDefiningOp<arith::IndexCastOp>())
    return buildABIExprFromValue(castOp.getIn());
  if (auto castOp = value.getDefiningOp<arith::IndexCastUIOp>())
    return buildABIExprFromValue(castOp.getIn());
  if (auto extOp = value.getDefiningOp<arith::ExtUIOp>())
    return buildABIExprFromValue(extOp.getIn());
  if (auto extOp = value.getDefiningOp<arith::ExtSIOp>())
    return buildABIExprFromValue(extOp.getIn());
  if (auto truncOp = value.getDefiningOp<arith::TruncIOp>())
    return buildABIExprFromValue(truncOp.getIn());
  if (auto mulOp = value.getDefiningOp<arith::MulIOp>()) {
    auto lhs = buildABIExprFromValue(mulOp.getLhs());
    auto rhs = buildABIExprFromValue(mulOp.getRhs());
    if (!lhs || !rhs)
      return std::nullopt;
    return ABIExpr::mulExpr(std::move(*lhs), std::move(*rhs));
  }

  return std::nullopt;
}

static unsigned getExternalPointerAddressSpace(MemRefType type) {
  if (auto addrAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(type.getMemorySpace())) {
    switch (addrAttr.getAddressSpace()) {
    case pto::AddressSpace::GM:
    case pto::AddressSpace::Zero:
      return 1;
    case pto::AddressSpace::VEC:
      return 6;
    default:
      break;
    }
  }
  return 1;
}

static std::optional<ABIExpr> deriveMemRefTotalSize(BlockArgument arg,
                                                    MemRefType type) {
  if (type.getRank() != 1)
    return std::nullopt;

  if (!type.isDynamicDim(0))
    return ABIExpr::constantExpr(type.getDimSize(0));

  for (Operation *user : arg.getUsers()) {
    auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(user);
    if (!reinterpret || reinterpret.getSource() != arg)
      continue;

    std::optional<ABIExpr> accum;
    for (OpFoldResult size : reinterpret.getMixedSizes()) {
      auto sizeExpr = buildABIExprFromFoldResult(size);
      if (!sizeExpr)
        return std::nullopt;
      accum = accum ? ABIExpr::mulExpr(std::move(*accum), std::move(*sizeExpr))
                    : std::move(*sizeExpr);
    }
    if (accum)
      return accum;
  }

  return std::nullopt;
}

static llvm::StringMap<FunctionABISpec> collectFunctionABISpecs(ModuleOp module) {
  llvm::StringMap<FunctionABISpec> specs;
  module.walk([&](func::FuncOp funcOp) {
    if (funcOp.isExternal())
      return;

    FunctionABISpec funcSpec;
    funcSpec.args.reserve(funcOp.getNumArguments());

    for (BlockArgument arg : funcOp.getArguments()) {
      ExternalArgABISpec argSpec;
      if (auto memrefType = dyn_cast<MemRefType>(arg.getType())) {
        if (memrefType.getRank() == 1) {
          auto totalSize = deriveMemRefTotalSize(arg, memrefType);
          if (totalSize) {
            argSpec.isMemRef = true;
            argSpec.memrefSpec.addressSpace =
                getExternalPointerAddressSpace(memrefType);
            argSpec.memrefSpec.rank = 1;
            argSpec.memrefSpec.offset = ABIExpr::constantExpr(0);
            argSpec.memrefSpec.totalSize = std::move(*totalSize);
            argSpec.memrefSpec.stride = ABIExpr::constantExpr(1);
          }
        }
      }
      funcSpec.args.push_back(std::move(argSpec));
    }

    specs[funcOp.getName().str()] = std::move(funcSpec);
  });
  return specs;
}

static std::optional<uint64_t> parsePipeImmediate(llvm::StringRef pipe) {
  if (pipe == "PIPE_S")
    return 0;
  if (pipe == "PIPE_V")
    return 1;
  if (pipe == "PIPE_M")
    return 2;
  if (pipe == "PIPE_MTE1")
    return 3;
  if (pipe == "PIPE_MTE2")
    return 4;
  if (pipe == "PIPE_MTE3")
    return 5;
  if (pipe == "PIPE_ALL")
    return 6;
  if (pipe == "PIPE_MTE4")
    return 7;
  if (pipe == "PIPE_MTE5")
    return 8;
  if (pipe == "PIPE_V2")
    return 9;
  if (pipe == "PIPE_FIX")
    return 10;
  if (pipe == "VIRTUAL_PIPE_MTE2_L1A")
    return 11;
  if (pipe == "VIRTUAL_PIPE_MTE2_L1B")
    return 12;
  return std::nullopt;
}

static std::optional<uint64_t> parseEventImmediate(llvm::StringRef event) {
  if (!event.consume_front("EVENT_ID"))
    return std::nullopt;
  uint64_t value = 0;
  if (event.getAsInteger(10, value))
    return std::nullopt;
  return value;
}

static std::optional<uint64_t> parseLoadDistImmediate(llvm::StringRef dist) {
  if (dist.empty() || dist == "NORM")
    return 0;
  if (dist == "BLK")
    return 15;
  if (dist == "UNPK_B16")
    return 14;
  if (dist == "DINTLV_B32")
    return 19;
  return std::nullopt;
}

static std::optional<uint64_t> parseStoreDistImmediate(Type valueType,
                                                       llvm::StringRef dist) {
  Type elementType = getElementTypeFromVectorLike(valueType);
  if (!elementType)
    return std::nullopt;

  if (dist.empty()) {
    unsigned bitWidth = 0;
    if (auto intType = dyn_cast<IntegerType>(elementType))
      bitWidth = intType.getWidth();
    else if (auto floatType = dyn_cast<FloatType>(elementType))
      bitWidth = floatType.getWidth();
    switch (bitWidth) {
    case 8:
      return 0;
    case 16:
      return 1;
    case 32:
      return 2;
    default:
      return std::nullopt;
    }
  }

  if (dist == "NORM_B8")
    return 0;
  if (dist == "NORM_B16")
    return 1;
  if (dist == "NORM_B32")
    return 2;
  if (dist == "ONEPT_B8")
    return 3;
  if (dist == "ONEPT_B16")
    return 4;
  if (dist == "ONEPT_B32")
    return 5;
  if (dist == "PK_B16")
    return 6;
  if (dist == "PK_B32")
    return 7;
  if (dist == "INTLV_B8")
    return 8;
  if (dist == "INTLV_B16")
    return 9;
  if (dist == "PK_B64")
    return 10;
  if (dist == "INTLV_B32")
    return 11;
  if (dist == "PK4_B32")
    return 12;
  if (dist == "MRG4CHN_B8")
    return 13;
  if (dist == "MRG2CHN_B8")
    return 14;
  if (dist == "MRG2CHN_B16")
    return 15;
  return std::nullopt;
}

static Type convertA5VMType(Type type, Builder &builder) {
  if (auto vecType = dyn_cast<a5vm::VecType>(type))
    return VectorType::get({vecType.getElementCount()}, vecType.getElementType());
  if (isa<a5vm::MaskType>(type))
    return VectorType::get({256}, builder.getI1Type());
  if (isa<a5vm::AlignType>(type))
    return builder.getI64Type();
  return type;
}

static Type getElementTypeFromVectorLike(Type type) {
  if (auto vecType = dyn_cast<a5vm::VecType>(type))
    return vecType.getElementType();
  if (auto vecType = dyn_cast<VectorType>(type))
    return vecType.getElementType();
  return {};
}

static Value castIntegerLikeTo(Operation *anchor, Value value, Type targetType) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);

  if (value.getType() == targetType)
    return value;

  auto targetInt = dyn_cast<IntegerType>(targetType);
  if (value.getType().isIndex() && targetInt)
    return builder.create<arith::IndexCastOp>(anchor->getLoc(), targetType, value);
  if (auto sourceInt = dyn_cast<IntegerType>(value.getType())) {
    if (targetInt) {
      if (sourceInt.getWidth() < targetInt.getWidth())
        return builder.create<arith::ExtUIOp>(anchor->getLoc(), targetType, value);
      if (sourceInt.getWidth() > targetInt.getWidth())
        return builder.create<arith::TruncIOp>(anchor->getLoc(), targetType, value);
      return value;
    }
    if (targetType.isIndex())
      return builder.create<arith::IndexCastOp>(anchor->getLoc(), targetType, value);
  }

  return {};
}

static FailureOr<Value> convertElementOffsetToBytes(Operation *anchor, Value offset,
                                                    Type elementType) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);

  Value offsetI32 = castIntegerLikeTo(anchor, offset, builder.getI32Type());
  if (!offsetI32)
    return failure();

  unsigned bitWidth = 0;
  if (auto intType = dyn_cast<IntegerType>(elementType))
    bitWidth = intType.getWidth();
  else if (auto floatType = dyn_cast<FloatType>(elementType))
    bitWidth = floatType.getWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return failure();

  Value scale = builder.create<arith::ConstantOp>(
      anchor->getLoc(), builder.getI32IntegerAttr(bitWidth / 8));
  return builder.create<arith::MulIOp>(anchor->getLoc(), offsetI32, scale)
      .getResult();
}

static Value getI64Constant(OpBuilder &builder, Location loc, uint64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(value))
      .getResult();
}

static Value getI32Constant(OpBuilder &builder, Location loc, uint64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(value))
      .getResult();
}

static FailureOr<Value> packLoopPair(Operation *anchor, Value low, Value high) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);

  Value lowI64 = castIntegerLikeTo(anchor, low, builder.getI64Type());
  Value highI64 = castIntegerLikeTo(anchor, high, builder.getI64Type());
  if (!lowI64 || !highI64)
    return failure();

  Value shift = getI64Constant(builder, anchor->getLoc(), 40);
  Value highShifted =
      builder.create<arith::ShLIOp>(anchor->getLoc(), highI64, shift).getResult();
  return builder.create<arith::OrIOp>(anchor->getLoc(), highShifted, lowI64)
      .getResult();
}

static FailureOr<Value> packLoopSize(Operation *anchor, Value loop2, Value loop1) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);

  Value loop2I64 = castIntegerLikeTo(anchor, loop2, builder.getI64Type());
  Value loop1I64 = castIntegerLikeTo(anchor, loop1, builder.getI64Type());
  if (!loop2I64 || !loop1I64)
    return failure();

  Value shift = getI64Constant(builder, anchor->getLoc(), 21);
  Value loop2Shifted =
      builder.create<arith::ShLIOp>(anchor->getLoc(), loop2I64, shift).getResult();
  return builder.create<arith::OrIOp>(anchor->getLoc(), loop2Shifted, loop1I64)
      .getResult();
}

static FailureOr<Value>
packCopyGmToUbConfig0(Operation *anchor, a5vm::CopyGmToUbufOp op,
                      ValueRange operands) {
  if (operands.size() != 12)
    return failure();

  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  auto getI64Operand = [&](unsigned idx) -> Value {
    return castIntegerLikeTo(anchor, operands[idx], builder.getI64Type());
  };

  Value sid = getI64Operand(4);
  Value nBurst = getI64Operand(5);
  Value lenBurst = getI64Operand(6);
  Value leftPadding = getI64Operand(7);
  Value rightPadding = getI64Operand(8);
  Value cacheCtl = getI64Operand(9);
  if (!sid || !nBurst || !lenBurst || !leftPadding || !rightPadding || !cacheCtl)
    return failure();

  Value dataSelect =
      getI64Constant(builder, loc,
                     op.getDataSelectBit().has_value() && *op.getDataSelectBit());

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value,
                                         getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value {
    return builder.create<arith::OrIOp>(loc, lhs, rhs);
  };

  Value config = sid;
  config = bitOr(config, shl(nBurst, 4));
  config = bitOr(config, shl(lenBurst, 25));
  config = bitOr(config, shl(leftPadding, 46));
  config = bitOr(config, shl(rightPadding, 52));
  config = bitOr(config, shl(dataSelect, 58));
  config = bitOr(config, shl(cacheCtl, 60));
  return config;
}

static FailureOr<Value>
packCopyGmToUbConfig1(Operation *anchor, ValueRange operands) {
  if (operands.size() != 12)
    return failure();
  return packLoopPair(anchor, operands[10], operands[11]);
}

static FailureOr<Value>
packCopyUbToGmConfig0(Operation *anchor, ValueRange operands) {
  if (operands.size() != 10)
    return failure();

  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  auto getI64Operand = [&](unsigned idx) -> Value {
    return castIntegerLikeTo(anchor, operands[idx], builder.getI64Type());
  };

  Value sid = getI64Operand(4);
  Value nBurst = getI64Operand(5);
  Value lenBurst = getI64Operand(6);
  Value reserved = getI64Operand(7);
  if (!sid || !nBurst || !lenBurst || !reserved)
    return failure();

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value,
                                         getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value {
    return builder.create<arith::OrIOp>(loc, lhs, rhs);
  };

  Value config = sid;
  config = bitOr(config, shl(nBurst, 4));
  config = bitOr(config, shl(lenBurst, 25));
  config = bitOr(config, shl(reserved, 60));
  return config;
}

static FailureOr<Value>
packCopyUbToGmConfig1(Operation *anchor, ValueRange operands) {
  if (operands.size() != 10)
    return failure();
  return packLoopPair(anchor, operands[8], operands[9]);
}

static func::FuncOp getOrCreateExternalFunc(ModuleOp module, StringRef name,
                                            FunctionType type) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(name))
    return existing;
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());
  auto func = builder.create<func::FuncOp>(module.getLoc(), name, type);
  func.setPrivate();
  return func;
}

static FailureOr<StringRef> getConfirmedAbsPathCallee(Operation *op) {
  if (isa<a5vm::SetLoop2StrideOutToUbOp>(op))
    return StringRef("llvm.hivm.SET.LOOP2.STRIDE.OUTTOUB");
  if (isa<a5vm::SetLoop1StrideOutToUbOp>(op))
    return StringRef("llvm.hivm.SET.LOOP1.STRIDE.OUTTOUB");
  if (isa<a5vm::SetLoopSizeOutToUbOp>(op))
    return StringRef("llvm.hivm.SET.LOOP.SIZE.OUTTOUB");
  if (isa<a5vm::SetLoop2StrideUbToOutOp>(op))
    return StringRef("llvm.hivm.SET.LOOP2.STRIDE.UBTOOUT");
  if (isa<a5vm::SetLoop1StrideUbToOutOp>(op))
    return StringRef("llvm.hivm.SET.LOOP1.STRIDE.UBTOOUT");
  if (isa<a5vm::SetLoopSizeUbToOutOp>(op))
    return StringRef("llvm.hivm.SET.LOOP.SIZE.UBTOOUT");
  if (isa<a5vm::CopyGmToUbufOp>(op))
    return StringRef("llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.f32.DV");
  if (isa<a5vm::CopyUbufToGmOp>(op))
    return StringRef("llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.DV");
  if (isa<a5vm::SetFlagOp>(op))
    return StringRef("llvm.hivm.SET.FLAG.IMM");
  if (isa<a5vm::WaitFlagOp>(op))
    return StringRef("llvm.hivm.WAIT.FLAG.IMM");
  if (isa<a5vm::PipeBarrierOp>(op))
    return StringRef("llvm.hivm.BARRIER");
  if (isa<a5vm::PltB32Op>(op))
    return StringRef("llvm.hivm.plt.b32.v300");
  if (isa<a5vm::VldsOp>(op))
    return StringRef("llvm.hivm.vldsx1");
  if (isa<a5vm::VabsOp>(op))
    return StringRef("llvm.hivm.vabs.v64f32.x");
  if (isa<a5vm::VstsOp>(op))
    return StringRef("llvm.hivm.vstsx1");
  return failure();
}

static LogicalResult rewriteA5VMOp(Operation *op, ModuleOp module,
                                   llvm::raw_ostream &diagOS) {
  auto calleeName = getConfirmedAbsPathCallee(op);
  if (failed(calleeName)) {
    diagOS << "A5VM LLVM emission failed: unsupported Abs-path op "
           << op->getName().getStringRef() << "\n";
    return failure();
  }

  IRRewriter builder(op->getContext());
  builder.setInsertionPoint(op);
  Location loc = op->getLoc();

  SmallVector<Type> resultTypes;
  for (Type type : op->getResultTypes())
    resultTypes.push_back(convertA5VMType(type, builder));

  SmallVector<Value> callArgs;

  if (isa<a5vm::SetLoop2StrideOutToUbOp, a5vm::SetLoop1StrideOutToUbOp,
          a5vm::SetLoop2StrideUbToOutOp, a5vm::SetLoop1StrideUbToOutOp>(op)) {
    auto packed = packLoopPair(op, op->getOperand(0), op->getOperand(1));
    if (failed(packed))
      return failure();
    callArgs.push_back(*packed);
  } else if (isa<a5vm::SetLoopSizeOutToUbOp, a5vm::SetLoopSizeUbToOutOp>(op)) {
    auto packed = packLoopSize(op, op->getOperand(0), op->getOperand(1));
    if (failed(packed))
      return failure();
    callArgs.push_back(*packed);
  } else if (auto copy = dyn_cast<a5vm::CopyGmToUbufOp>(op)) {
    auto config0 = packCopyGmToUbConfig0(op, copy, op->getOperands());
    auto config1 = packCopyGmToUbConfig1(op, op->getOperands());
    if (failed(config0) || failed(config1))
      return failure();
    callArgs.push_back(op->getOperand(1));
    callArgs.push_back(op->getOperand(0));
    callArgs.push_back(*config0);
    callArgs.push_back(*config1);
  } else if (isa<a5vm::CopyUbufToGmOp>(op)) {
    auto config0 = packCopyUbToGmConfig0(op, op->getOperands());
    auto config1 = packCopyUbToGmConfig1(op, op->getOperands());
    if (failed(config0) || failed(config1))
      return failure();
    callArgs.push_back(op->getOperand(1));
    callArgs.push_back(op->getOperand(0));
    callArgs.push_back(*config0);
    callArgs.push_back(*config1);
  } else if (auto setFlag = dyn_cast<a5vm::SetFlagOp>(op)) {
    auto src = parsePipeImmediate(setFlag.getSrcPipe());
    auto dst = parsePipeImmediate(setFlag.getDstPipe());
    auto event = parseEventImmediate(setFlag.getEventId());
    if (!src || !dst || !event)
      return failure();
    callArgs.push_back(getI64Constant(builder, loc, *src));
    callArgs.push_back(getI64Constant(builder, loc, *dst));
    callArgs.push_back(getI64Constant(builder, loc, *event));
  } else if (auto waitFlag = dyn_cast<a5vm::WaitFlagOp>(op)) {
    auto src = parsePipeImmediate(waitFlag.getSrcPipe());
    auto dst = parsePipeImmediate(waitFlag.getDstPipe());
    auto event = parseEventImmediate(waitFlag.getEventId());
    if (!src || !dst || !event)
      return failure();
    callArgs.push_back(getI64Constant(builder, loc, *src));
    callArgs.push_back(getI64Constant(builder, loc, *dst));
    callArgs.push_back(getI64Constant(builder, loc, *event));
  } else if (auto barrier = dyn_cast<a5vm::PipeBarrierOp>(op)) {
    auto pipe = parsePipeImmediate(barrier.getPipe());
    if (!pipe)
      return failure();
    callArgs.push_back(getI64Constant(builder, loc, *pipe));
  } else if (isa<a5vm::PltB32Op>(op)) {
    Value laneCount = castIntegerLikeTo(op, op->getOperand(0), builder.getI32Type());
    if (!laneCount)
      return failure();
    callArgs.push_back(laneCount);
  } else if (auto vlds = dyn_cast<a5vm::VldsOp>(op)) {
    Type elementType = getElementTypeFromVectorLike(vlds.getResult().getType());
    auto offsetBytes = convertElementOffsetToBytes(
        op, op->getOperand(1), elementType);
    auto dist = parseLoadDistImmediate(vlds.getDist().value_or("NORM"));
    if (!elementType || failed(offsetBytes) || !dist)
      return failure();
    callArgs.push_back(op->getOperand(0));
    callArgs.push_back(*offsetBytes);
    callArgs.push_back(getI32Constant(builder, loc, *dist));
    callArgs.push_back(getI32Constant(builder, loc, 0));
  } else if (auto vabs = dyn_cast<a5vm::VabsOp>(op)) {
    Value input = op->getOperand(0);
    Value mask = op->getOperand(1);
    Type vecType = resultTypes.front();
    Type maskType = convertA5VMType(mask.getType(), builder);
    if (input.getType() != vecType || mask.getType() != maskType) {
      diagOS << "A5VM LLVM emission failed: unexpected vabs operand types\n";
      return failure();
    }
    callArgs.push_back(input);
    callArgs.push_back(mask);
  } else if (auto vsts = dyn_cast<a5vm::VstsOp>(op)) {
    Type elementType = getElementTypeFromVectorLike(vsts.getValue().getType());
    auto offsetBytes = convertElementOffsetToBytes(
        op, op->getOperand(2), elementType);
    auto dist = parseStoreDistImmediate(vsts.getValue().getType(),
                                        vsts.getDist().value_or(""));
    if (!elementType || failed(offsetBytes) || !dist)
      return failure();
    callArgs.push_back(op->getOperand(0));
    callArgs.push_back(op->getOperand(1));
    callArgs.push_back(*offsetBytes);
    callArgs.push_back(getI32Constant(builder, loc, *dist));
    callArgs.push_back(getI32Constant(builder, loc, 0));
    callArgs.push_back(op->getOperand(3));
  } else {
    diagOS << "A5VM LLVM emission failed: Abs path does not yet support "
           << op->getName().getStringRef() << "\n";
    return failure();
  }

  SmallVector<Type> argTypes;
  for (Value arg : callArgs)
    argTypes.push_back(arg.getType());

  auto funcType = builder.getFunctionType(argTypes, resultTypes);
  auto callee = getOrCreateExternalFunc(module, *calleeName, funcType);
  auto call = builder.create<func::CallOp>(loc, callee, callArgs);
  if (op->getNumResults() == 0)
    builder.eraseOp(op);
  else
    builder.replaceOp(op, call.getResults());
  return success();
}

static LogicalResult rewriteA5VMOps(ModuleOp module, llvm::raw_ostream &diagOS) {
  SmallVector<Operation *> opsToRewrite;
  module.walk([&](Operation *op) {
    if (op->getName().getDialectNamespace() == "a5vm")
      opsToRewrite.push_back(op);
  });

  for (Operation *op : opsToRewrite) {
    if (failed(rewriteA5VMOp(op, module, diagOS)))
      return failure();
  }

  bool hasA5VM = false;
  module.walk([&](Operation *op) {
    if (op->getName().getDialectNamespace() == "a5vm")
      hasA5VM = true;
  });
  return success(!hasA5VM);
}

static llvm::StringMap<unsigned>
collectVecScopeLoopCounts(ModuleOp module) {
  llvm::StringMap<unsigned> counts;
  module.walk([&](scf::ForOp forOp) {
    if (!forOp->hasAttr("llvm.loop.aivector_scope"))
      return;
    auto func = forOp->getParentOfType<func::FuncOp>();
    if (!func)
      return;
    counts[func.getName().str()]++;
  });
  return counts;
}

static bool ensureDummyPredForAIVectorScopeLatch(llvm::Loop *loop) {
  llvm::BasicBlock *latch = loop->getLoopLatch();
  if (!latch)
    return false;

  llvm::SmallVector<llvm::BasicBlock *, 4> preds(llvm::predecessors(latch));
  if (preds.size() != 1)
    return false;

  llvm::BasicBlock *pred = preds.front();
  auto *predTerm = pred->getTerminator();
  if (!predTerm || predTerm->getNumSuccessors() <= 1)
    return false;

  llvm::Function *function = latch->getParent();
  if (!function)
    return false;

  llvm::BasicBlock *dummy =
      llvm::BasicBlock::Create(function->getContext(), "aivscope.dummy", function, latch);
  llvm::BranchInst::Create(latch, dummy);
  predTerm->replaceUsesOfWith(latch, dummy);
  return true;
}

static void attachAIVectorScopeMetadata(llvm::Module &llvmModule,
                                        const llvm::StringMap<unsigned> &counts) {
  for (llvm::Function &function : llvmModule) {
    auto it = counts.find(function.getName());
    if (it == counts.end() || it->second == 0)
      continue;
    if (it->second != 1)
      continue;

    llvm::DominatorTree dt(function);
    llvm::LoopInfo loopInfo(dt);
    if (loopInfo.empty())
      continue;

    llvm::Loop *loop = *loopInfo.begin();
    (void)ensureDummyPredForAIVectorScopeLatch(loop);

    dt.recalculate(function);
    loopInfo.releaseMemory();
    loopInfo.analyze(dt);
    if (loopInfo.empty())
      continue;
    loop = *loopInfo.begin();

    llvm::BasicBlock *latch = loop->getLoopLatch();
    if (!latch)
      continue;
    auto *terminator = latch->getTerminator();
    if (!terminator)
      continue;

    llvm::LLVMContext &ctx = llvmModule.getContext();
    llvm::Metadata *ops[] = {
        nullptr, llvm::MDNode::get(ctx, llvm::MDString::get(ctx, "llvm.loop.aivector_scope"))};
    auto *loopID = llvm::MDNode::getDistinct(ctx, ops);
    loopID->replaceOperandWith(0, loopID);
    terminator->setMetadata(llvm::LLVMContext::MD_loop, loopID);
  }
}

static FailureOr<std::string> extractQuotedLLVMFnAttr(llvm::StringRef ir,
                                                      llvm::StringRef key) {
  std::string pattern = "\"";
  pattern += key.str();
  pattern += "\"=\"";
  size_t start = ir.find(pattern);
  if (start == llvm::StringRef::npos)
    return failure();
  start += pattern.size();
  size_t end = ir.find('"', start);
  if (end == llvm::StringRef::npos || end <= start)
    return failure();
  return ir.slice(start, end).str();
}

static FailureOr<QueriedTargetAttrs>
queryDefaultTargetAttrs(const A5VMEmissionOptions &options,
                        llvm::raw_ostream &diagOS) {
  static llvm::StringMap<QueriedTargetAttrs> cache;

  if (options.targetTriple.empty() || options.march.empty() ||
      options.aicoreArch.empty()) {
    diagOS << "A5VM LLVM emission failed: missing target query options\n";
    return failure();
  }

  std::string cacheKey =
      options.targetTriple + "|" + options.march + "|" + options.aicoreArch;
  if (auto it = cache.find(cacheKey); it != cache.end())
    return it->second;

  auto bisheng = llvm::sys::findProgramByName("bisheng");
  if (!bisheng) {
    diagOS << "A5VM LLVM emission failed: unable to find 'bisheng' in PATH\n";
    return failure();
  }
  const std::string &bishengPath = *bisheng;

  llvm::SmallString<64> inputPath;
  llvm::SmallString<64> outputPath;
  int inputFD = -1;
  int outputFD = -1;
  if (auto ec = llvm::sys::fs::createTemporaryFile("ptoas-a5vm-target-query",
                                                   "c", inputFD, inputPath)) {
    diagOS << "A5VM LLVM emission failed: cannot create bisheng query input: "
           << ec.message() << "\n";
    return failure();
  }
  if (auto ec = llvm::sys::fs::createTemporaryFile("ptoas-a5vm-target-query",
                                                   "ll", outputFD, outputPath)) {
    llvm::sys::fs::remove(inputPath);
    llvm::sys::Process::SafelyCloseFileDescriptor(inputFD);
    diagOS << "A5VM LLVM emission failed: cannot create bisheng query output: "
           << ec.message() << "\n";
    return failure();
  }

  auto cleanup = llvm::make_scope_exit([&]() {
    llvm::sys::fs::remove(inputPath);
    llvm::sys::fs::remove(outputPath);
  });

  {
    llvm::raw_fd_ostream inputOS(inputFD, /*shouldClose=*/false);
    inputOS << "void f(void) {}\n";
  }
  llvm::sys::Process::SafelyCloseFileDescriptor(inputFD);
  llvm::sys::Process::SafelyCloseFileDescriptor(outputFD);

  llvm::SmallString<128> stderrPath;
  int stderrFD = -1;
  if (auto ec = llvm::sys::fs::createTemporaryFile("ptoas-a5vm-target-query",
                                                   "stderr", stderrFD,
                                                   stderrPath)) {
    diagOS << "A5VM LLVM emission failed: cannot create bisheng query stderr: "
           << ec.message() << "\n";
    return failure();
  }
  auto stderrCleanup = llvm::make_scope_exit([&]() {
    llvm::sys::fs::remove(stderrPath);
  });
  llvm::sys::Process::SafelyCloseFileDescriptor(stderrFD);

  llvm::SmallVector<std::string> argStorage = {
      bishengPath,
      ("--target=" + options.targetTriple),
      ("-march=" + options.march),
      ("--cce-aicore-arch=" + options.aicoreArch),
      "--cce-aicore-only",
      "-x",
      "c",
      inputPath.str().str(),
      "-S",
      "-emit-llvm",
      "-o",
      outputPath.str().str(),
  };
  llvm::SmallVector<llvm::StringRef> args;
  args.reserve(argStorage.size());
  for (const std::string &arg : argStorage)
    args.push_back(arg);

  std::string execErr;
  bool execFailed = false;
  int rc = llvm::sys::ExecuteAndWait(
      bishengPath, args, std::nullopt,
      {std::nullopt, std::nullopt, llvm::StringRef(stderrPath)}, 0, 0,
      &execErr, &execFailed);

  auto stderrBuffer = llvm::MemoryBuffer::getFile(stderrPath);
  llvm::StringRef stderrText =
      stderrBuffer ? stderrBuffer.get()->getBuffer() : llvm::StringRef();

  if (execFailed || rc != 0) {
    diagOS << "A5VM LLVM emission failed: bisheng target query failed\n";
    diagOS << "Command:";
    for (llvm::StringRef arg : args)
      diagOS << " " << arg;
    diagOS << "\n";
    if (!execErr.empty())
      diagOS << execErr << "\n";
    if (!stderrText.empty())
      diagOS << stderrText << "\n";
    return failure();
  }

  auto outputBuffer = llvm::MemoryBuffer::getFile(outputPath);
  if (!outputBuffer) {
    diagOS << "A5VM LLVM emission failed: cannot read bisheng query output\n";
    return failure();
  }

  FailureOr<std::string> targetCPU =
      extractQuotedLLVMFnAttr(outputBuffer.get()->getBuffer(), "target-cpu");
  FailureOr<std::string> targetFeatures = extractQuotedLLVMFnAttr(
      outputBuffer.get()->getBuffer(), "target-features");
  if (failed(targetCPU) || failed(targetFeatures)) {
    diagOS << "A5VM LLVM emission failed: cannot parse bisheng target attrs\n";
    diagOS << outputBuffer.get()->getBuffer() << "\n";
    return failure();
  }

  QueriedTargetAttrs attrs{*targetCPU, *targetFeatures};
  cache[cacheKey] = attrs;
  return attrs;
}

static LogicalResult
applyQueriedTargetAttrs(ModuleOp module, const A5VMEmissionOptions &options,
                        llvm::raw_ostream &diagOS) {
  FailureOr<QueriedTargetAttrs> attrs = queryDefaultTargetAttrs(options, diagOS);
  if (failed(attrs)) {
    if (options.defaultTargetCPU.empty() ||
        options.defaultTargetFeatures.empty())
      return failure();
    diagOS << "A5VM LLVM emission: falling back to configured default target attributes\n";
    attrs = QueriedTargetAttrs{options.defaultTargetCPU,
                               options.defaultTargetFeatures};
  }

  MLIRContext *ctx = module.getContext();
  StringAttr cpuAttr = StringAttr::get(ctx, attrs->targetCPU);
  LLVM::TargetFeaturesAttr featureAttr =
      LLVM::TargetFeaturesAttr::get(ctx, attrs->targetFeatures);
  module.walk([&](LLVM::LLVMFuncOp funcOp) {
    funcOp.setTargetCpuAttr(cpuAttr);
    funcOp.setTargetFeaturesAttr(featureAttr);
  });
  return success();
}

static llvm::Value *castABIValue(llvm::IRBuilder<> &builder, llvm::Value *value,
                                 llvm::Type *targetType) {
  if (value->getType() == targetType)
    return value;

  if (auto *targetPtr = dyn_cast<llvm::PointerType>(targetType)) {
    auto *sourcePtr = dyn_cast<llvm::PointerType>(value->getType());
    if (!sourcePtr)
      return nullptr;
    if (sourcePtr->getAddressSpace() == targetPtr->getAddressSpace())
      return builder.CreateBitCast(value, targetType);
    return builder.CreateAddrSpaceCast(value, targetType);
  }

  if (targetType->isIntegerTy()) {
    if (value->getType()->isIntegerTy()) {
      unsigned srcWidth = value->getType()->getIntegerBitWidth();
      unsigned dstWidth = targetType->getIntegerBitWidth();
      if (srcWidth == dstWidth)
        return value;
      if (srcWidth < dstWidth)
        return builder.CreateZExt(value, targetType);
      return builder.CreateTrunc(value, targetType);
    }
  }

  return nullptr;
}

static llvm::Value *materializeABIExpr(llvm::IRBuilder<> &builder,
                                       const ABIExpr &expr,
                                       llvm::Function *wrapper,
                                       llvm::Type *targetType) {
  switch (expr.kind) {
  case ABIExpr::Kind::Constant:
    return llvm::ConstantInt::get(targetType, expr.constant);
  case ABIExpr::Kind::FuncArg: {
    if (expr.argIndex >= wrapper->arg_size())
      return nullptr;
    return castABIValue(builder, wrapper->getArg(expr.argIndex), targetType);
  }
  case ABIExpr::Kind::Mul: {
    llvm::Value *lhs =
        materializeABIExpr(builder, *expr.lhs, wrapper, targetType);
    llvm::Value *rhs =
        materializeABIExpr(builder, *expr.rhs, wrapper, targetType);
    if (!lhs || !rhs)
      return nullptr;
    return builder.CreateMul(lhs, rhs);
  }
  }
  return nullptr;
}

static unsigned getMemRefExpandedArgCount(int64_t rank) {
  return 2u + 1u + static_cast<unsigned>(rank) + static_cast<unsigned>(rank);
}

static llvm::Value *resolveInsertedAggregateValue(llvm::Value *value,
                                                  llvm::ArrayRef<unsigned> idxs) {
  auto *insert = dyn_cast<llvm::InsertValueInst>(value);
  if (!insert)
    return nullptr;

  if (insert->getIndices() == idxs)
    return insert->getInsertedValueOperand();

  return resolveInsertedAggregateValue(insert->getAggregateOperand(), idxs);
}

static llvm::Value *resolveAddrSpaceRoundTrip(llvm::Value *value) {
  auto *outerCast = dyn_cast<llvm::AddrSpaceCastInst>(value);
  if (!outerCast)
    return nullptr;

  auto *innerCast = dyn_cast<llvm::AddrSpaceCastInst>(outerCast->getPointerOperand());
  if (!innerCast)
    return nullptr;

  llvm::Value *original = innerCast->getPointerOperand();
  if (original->getType() != outerCast->getType())
    return nullptr;

  auto *innerDstPtr = dyn_cast<llvm::PointerType>(innerCast->getType());
  auto *outerDstPtr = dyn_cast<llvm::PointerType>(outerCast->getType());
  auto *origPtr = dyn_cast<llvm::PointerType>(original->getType());
  if (!innerDstPtr || !outerDstPtr || !origPtr)
    return nullptr;

  if (innerDstPtr->getAddressSpace() == origPtr->getAddressSpace())
    return nullptr;
  if (outerDstPtr->getAddressSpace() != origPtr->getAddressSpace())
    return nullptr;

  return original;
}

static void simplifyAggregateCarrierOps(llvm::Function &function) {
  bool changed = true;
  while (changed) {
    changed = false;

    SmallVector<llvm::Instruction *> toErase;
    for (llvm::BasicBlock &block : function) {
      for (llvm::Instruction &inst : block) {
        if (auto *cast = dyn_cast<llvm::AddrSpaceCastInst>(&inst)) {
          if (llvm::Value *resolved = resolveAddrSpaceRoundTrip(cast)) {
            cast->replaceAllUsesWith(resolved);
            toErase.push_back(cast);
            changed = true;
            continue;
          }
        }

        if (auto *extract = dyn_cast<llvm::ExtractValueInst>(&inst)) {
          if (llvm::Value *resolved =
                  resolveInsertedAggregateValue(extract->getAggregateOperand(),
                                               extract->getIndices())) {
            extract->replaceAllUsesWith(resolved);
            toErase.push_back(extract);
            changed = true;
            continue;
          }
        }

        if (llvm::isInstructionTriviallyDead(&inst)) {
          toErase.push_back(&inst);
          changed = true;
        }
      }
    }

    for (llvm::Instruction *inst : toErase)
      if (!inst->isTerminator())
        inst->eraseFromParent();
  }
}

static LogicalResult rewriteFunctionsToEmitCStyleABI(
    llvm::Module &llvmModule, const llvm::StringMap<FunctionABISpec> &specs,
    llvm::raw_ostream &diagOS) {
  SmallVector<llvm::Function *> funcs;
  for (llvm::Function &function : llvmModule)
    if (!function.isDeclaration())
      funcs.push_back(&function);

  for (llvm::Function *function : funcs) {
    auto it = specs.find(function->getName());
    if (it == specs.end())
      continue;

    const FunctionABISpec &spec = it->second;
    if (spec.args.empty())
      continue;

    bool needsRewrite =
        llvm::any_of(spec.args, [](const ExternalArgABISpec &arg) {
          return arg.isMemRef;
        });
    if (!needsRewrite)
      continue;

    SmallVector<llvm::Type *> publicArgTypes;
    SmallVector<unsigned> oldArgBaseIndex(spec.args.size(), 0);
    unsigned oldArgCursor = 0;
    bool supported = true;
    for (auto [idx, argSpec] : llvm::enumerate(spec.args)) {
      oldArgBaseIndex[idx] = oldArgCursor;
      if (argSpec.isMemRef) {
        if (argSpec.memrefSpec.rank != 1) {
          supported = false;
          break;
        }
        publicArgTypes.push_back(llvm::PointerType::get(
            llvmModule.getContext(), argSpec.memrefSpec.addressSpace));
        oldArgCursor += getMemRefExpandedArgCount(argSpec.memrefSpec.rank);
      } else {
        if (oldArgCursor >= function->arg_size()) {
          supported = false;
          break;
        }
        publicArgTypes.push_back(function->getArg(oldArgCursor)->getType());
        ++oldArgCursor;
      }
    }

    if (!supported || oldArgCursor != function->arg_size()) {
      diagOS << "A5VM LLVM emission warning: skipping ABI rewrite for "
             << function->getName()
             << " because the lowered signature does not match the seam spec\n";
      continue;
    }

    std::string originalName = function->getName().str();
    std::string tempName = "__ptoas_old_" + originalName;
    function->setName(tempName);
    function->setLinkage(llvm::GlobalValue::InternalLinkage);

    auto *publicType = llvm::FunctionType::get(function->getReturnType(),
                                               publicArgTypes,
                                               function->isVarArg());
    llvm::Function *replacement = llvm::Function::Create(
        publicType, llvm::GlobalValue::ExternalLinkage, originalName, &llvmModule);
    replacement->copyAttributesFrom(function);
    replacement->setLinkage(llvm::GlobalValue::ExternalLinkage);

    unsigned publicArgIndex = 0;
    for (llvm::Argument &arg : replacement->args())
      arg.setName("arg" + std::to_string(publicArgIndex++));

    llvm::BasicBlock *bridgeEntry = llvm::BasicBlock::Create(
        llvmModule.getContext(), "entry", replacement);
    llvm::IRBuilder<> builder(bridgeEntry);

    llvm::ValueToValueMapTy vmap;
    for (auto [idx, argSpec] : llvm::enumerate(spec.args)) {
      llvm::Value *publicArg = replacement->getArg(idx);
      unsigned oldBase = oldArgBaseIndex[idx];
      if (!argSpec.isMemRef) {
        llvm::Value *casted = castABIValue(
            builder, publicArg, function->getArg(oldBase)->getType());
        if (!casted) {
          diagOS << "A5VM LLVM emission failed: cannot cast scalar arg for "
                 << originalName << "\n";
          return failure();
        }
        vmap[function->getArg(oldBase)] = casted;
        continue;
      }

      llvm::Type *oldPtrTy = function->getArg(oldBase)->getType();
      llvm::Type *oldAlignedPtrTy = function->getArg(oldBase + 1)->getType();
      llvm::Type *oldOffsetTy = function->getArg(oldBase + 2)->getType();
      llvm::Type *oldSizeTy = function->getArg(oldBase + 3)->getType();
      llvm::Type *oldStrideTy = function->getArg(oldBase + 4)->getType();

      llvm::Value *allocated = castABIValue(builder, publicArg, oldPtrTy);
      llvm::Value *aligned = castABIValue(builder, publicArg, oldAlignedPtrTy);
      llvm::Value *offset = materializeABIExpr(
          builder, argSpec.memrefSpec.offset, replacement, oldOffsetTy);
      llvm::Value *size = materializeABIExpr(
          builder, argSpec.memrefSpec.totalSize, replacement, oldSizeTy);
      llvm::Value *stride = materializeABIExpr(
          builder, argSpec.memrefSpec.stride, replacement, oldStrideTy);
      if (!allocated || !aligned || !offset || !size || !stride) {
        diagOS << "A5VM LLVM emission failed: cannot materialize direct ABI for "
               << originalName << "\n";
        return failure();
      }

      vmap[function->getArg(oldBase)] = allocated;
      vmap[function->getArg(oldBase + 1)] = aligned;
      vmap[function->getArg(oldBase + 2)] = offset;
      vmap[function->getArg(oldBase + 3)] = size;
      vmap[function->getArg(oldBase + 4)] = stride;
    }

    llvm::SmallVector<llvm::ReturnInst *, 4> returns;
    llvm::CloneFunctionInto(replacement, function, vmap,
                            llvm::CloneFunctionChangeType::LocalChangesOnly,
                            returns);

    llvm::BasicBlock *oldEntry = &replacement->getEntryBlock();
    llvm::BasicBlock *clonedEntry = oldEntry->getNextNode();
    if (!clonedEntry) {
      diagOS << "A5VM LLVM emission failed: cloned function body is empty for "
             << originalName << "\n";
      return failure();
    }
    builder.CreateBr(clonedEntry);

    function->eraseFromParent();
    simplifyAggregateCarrierOps(*replacement);
  }

  return success();
}

static std::unique_ptr<llvm::Module>
buildLLVMModuleFromA5VM(ModuleOp module, llvm::LLVMContext &llvmContext,
                        const A5VMEmissionOptions &options,
                        llvm::raw_ostream &diagOS) {
  OwningOpRef<ModuleOp> cloned(cast<ModuleOp>(module->clone()));
  auto vecScopeCounts = collectVecScopeLoopCounts(*cloned);
  auto abiSpecs = collectFunctionABISpecs(*cloned);

  if (failed(rewriteA5VMOps(*cloned, diagOS))) {
    diagOS << "A5VM LLVM emission failed: A5VM-to-call rewriting failed\n";
    return nullptr;
  }

  PassManager pm(cloned->getContext());
  pm.enableVerifier();
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  if (failed(pm.run(*cloned))) {
    diagOS << "A5VM LLVM emission failed: official lowering pipeline failed\n";
    return nullptr;
  }

  if (failed(applyQueriedTargetAttrs(*cloned, options, diagOS)))
    return nullptr;

  registerBuiltinDialectTranslation(*cloned->getContext());
  registerLLVMDialectTranslation(*cloned->getContext());
  auto llvmModule = translateModuleToLLVMIR(cloned.get(), llvmContext);
  if (!llvmModule) {
    diagOS << "A5VM LLVM emission failed: LLVM IR export failed\n";
    return nullptr;
  }

  attachAIVectorScopeMetadata(*llvmModule, vecScopeCounts);
  if (failed(rewriteFunctionsToEmitCStyleABI(*llvmModule, abiSpecs, diagOS)))
    return nullptr;
  llvmModule->setModuleIdentifier("ptoas.hivm.official");
  llvmModule->setSourceFileName("ptoas.hivm.official");
  return llvmModule;
}

} // namespace

LogicalResult
translateA5VMModuleToLLVMText(ModuleOp module, llvm::raw_ostream &os,
                              const A5VMEmissionOptions &options,
                              llvm::raw_ostream &diagOS) {
  llvm::LLVMContext llvmContext;
  auto llvmModule = buildLLVMModuleFromA5VM(module, llvmContext, options, diagOS);
  if (!llvmModule)
    return failure();
  llvmModule->print(os, nullptr);
  return success();
}

LogicalResult
translateA5VMModuleToLLVMBitcode(ModuleOp module, llvm::raw_ostream &os,
                                 const A5VMEmissionOptions &options,
                                 llvm::raw_ostream &diagOS) {
  llvm::LLVMContext llvmContext;
  auto llvmModule = buildLLVMModuleFromA5VM(module, llvmContext, options, diagOS);
  if (!llvmModule)
    return failure();
  llvm::WriteBitcodeToFile(*llvmModule, os);
  return success();
}

} // namespace mlir::pto
