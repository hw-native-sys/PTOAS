//===- VPTOLLVMEmitter.cpp - VPTO to official LLVM IR text emitter -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/VPTOLLVMEmitter.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VPTOLowering.h"
#include "PTO/Transforms/HIVMIntrinsicNaming.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/SmallString.h"
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
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

using namespace mlir;

namespace mlir::pto {
namespace {

constexpr StringLiteral kAIVScopeDummyCallee = "aivscope_dummy";

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
static Type getElementTypeFromPointerLike(Type type);
static std::optional<int64_t> getElementCountFromVectorLike(Type type);
static func::FuncOp getOrCreateExternalFunc(ModuleOp module, StringRef name,
                                            FunctionType type);
static Value castIntegerLikeTo(Operation *anchor, Value value, Type targetType);

static std::string getElementTypeFragment(Type type) {
  if (type.isF16())
    return "f16";
  if (type.isBF16())
    return "bf16";
  if (type.isF32())
    return "f32";
  if (auto intType = dyn_cast<IntegerType>(type))
    return (intType.isUnsigned() ? "u" : "s") + std::to_string(intType.getWidth());
  return {};
}

static std::optional<uint64_t> parseRoundModeImmediate(StringRef roundMode) {
  if (roundMode == "R" || roundMode == "ROUND_R")
    return 0; // __cce_simd::ROUND::R
  if (roundMode == "A" || roundMode == "ROUND_A")
    return 1; // __cce_simd::ROUND::A
  if (roundMode == "F" || roundMode == "ROUND_F")
    return 2; // __cce_simd::ROUND::F
  if (roundMode == "C" || roundMode == "ROUND_C")
    return 3; // __cce_simd::ROUND::C
  if (roundMode == "Z" || roundMode == "ROUND_Z")
    return 4; // __cce_simd::ROUND::Z
  if (roundMode == "O" || roundMode == "ROUND_O")
    return 5; // __cce_simd::ROUND::O
  return std::nullopt;
}

static std::optional<uint64_t> parseSaturationImmediate(StringRef sat) {
  if (sat == "SAT" || sat == "RS_ENABLE")
    return 0; // __cce_simd::RoundingSaturation::ENABLE
  if (sat == "NOSAT" || sat == "RS_DISABLE")
    return 1; // __cce_simd::RoundingSaturation::DISABLE
  return std::nullopt;
}

static std::optional<uint64_t> parsePartImmediate(StringRef part) {
  if (part == "EVEN" || part == "PART_EVEN")
    return 0; // __cce_simd::Part::EVEN
  if (part == "ODD" || part == "PART_ODD")
    return 1; // __cce_simd::Part::ODD
  return std::nullopt;
}

static FailureOr<Value> normalizeVdupScalarOperand(OpBuilder &builder, Location loc,
                                                   pto::VdupOp vdup) {
  Value input = vdup.getInput();
  Type scalarType = input.getType();
  auto intType = dyn_cast<IntegerType>(scalarType);
  if (!intType || intType.getWidth() != 8)
    return input;

  Type resultElemType = getElementTypeFromVectorLike(vdup.getResult().getType());
  std::string resultElemFragment = getElementTypeFragment(resultElemType);
  if (resultElemFragment != "s8" && resultElemFragment != "u8")
    return input;

  Type i16Type = builder.getIntegerType(16);
  if (resultElemFragment == "u8")
    return builder.create<arith::ExtUIOp>(loc, i16Type, input).getResult();
  return builder.create<arith::ExtSIOp>(loc, i16Type, input).getResult();
}

// VSQZ #st hint must only be set when the compacted vector feeds VSTUR.
// Emitting #st=1 without a matching VSTUR consumer can deadlock hardware queues.
static uint64_t determineVsqzStoreHint(pto::VsqzOp vsqz) {
  Value result = vsqz.getResult();
  for (Operation *user : result.getUsers()) {
    auto vstur = dyn_cast<pto::VsturOp>(user);
    if (!vstur)
      continue;
    if (vstur.getValue() == result)
      return 1;
  }
  return 0;
}

enum class VcvtElemKind {
  Invalid,
  F16,
  BF16,
  F32,
  S8,
  U8,
  S16,
  U16,
  S32,
  U32,
  S64,
};

struct VcvtContract {
  const char *intrinsic;
  bool requiresRnd;
  bool requiresSat;
  bool requiresPart;
  unsigned maskBitWidth;
};

static VcvtElemKind classifyVcvtElemType(Type type) {
  if (type.isF16())
    return VcvtElemKind::F16;
  if (type.isBF16())
    return VcvtElemKind::BF16;
  if (type.isF32())
    return VcvtElemKind::F32;
  if (auto intType = dyn_cast<IntegerType>(type)) {
    switch (intType.getWidth()) {
    case 8:
      return intType.isUnsigned() ? VcvtElemKind::U8 : VcvtElemKind::S8;
    case 16:
      return intType.isUnsigned() ? VcvtElemKind::U16 : VcvtElemKind::S16;
    case 32:
      return intType.isUnsigned() ? VcvtElemKind::U32 : VcvtElemKind::S32;
    case 64:
      return intType.isUnsigned() ? VcvtElemKind::Invalid : VcvtElemKind::S64;
    default:
      return VcvtElemKind::Invalid;
    }
  }
  return VcvtElemKind::Invalid;
}

static std::optional<VcvtContract> lookupVcvtContract(VcvtElemKind src,
                                                      VcvtElemKind dst) {
  switch (src) {
  case VcvtElemKind::F32:
    switch (dst) {
    case VcvtElemKind::F16:
      return VcvtContract{"llvm.hivm.vcvtff.f322f16.x", true, true, true, 32};
    case VcvtElemKind::BF16:
      return VcvtContract{"llvm.hivm.vcvtff.f322bf16.x", true, true, true, 32};
    case VcvtElemKind::S16:
      return VcvtContract{"llvm.hivm.vcvtfi.f322s16.x", true, true, true, 32};
    case VcvtElemKind::S32:
      return VcvtContract{"llvm.hivm.vcvtfi.f322s32.x", true, true, false, 32};
    case VcvtElemKind::S64:
      return VcvtContract{"llvm.hivm.vcvtfi.f322s64.x", true, true, true, 32};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::F16:
    switch (dst) {
    case VcvtElemKind::F32:
      return VcvtContract{"llvm.hivm.vcvtff.f162f32.x", false, false, true, 16};
    case VcvtElemKind::S32:
      return VcvtContract{"llvm.hivm.vcvtfi.f162s32.x", true, false, true, 16};
    case VcvtElemKind::S16:
      return VcvtContract{"llvm.hivm.vcvtfi.f162s16.x", true, true, false, 16};
    case VcvtElemKind::S8:
      return VcvtContract{"llvm.hivm.vcvtfi.f162s8.x", true, true, true, 16};
    case VcvtElemKind::U8:
      return VcvtContract{"llvm.hivm.vcvtfi.f162u8.x", true, true, true, 16};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::BF16:
    switch (dst) {
    case VcvtElemKind::F32:
      return VcvtContract{"llvm.hivm.vcvtff.bf162f32.x", false, false, true, 16};
    case VcvtElemKind::S32:
      return VcvtContract{"llvm.hivm.vcvtfi.bf162s32.x", true, true, true, 16};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::U8:
    switch (dst) {
    case VcvtElemKind::F16:
      return VcvtContract{"llvm.hivm.vcvtif.u82f16.x", false, false, true, 8};
    case VcvtElemKind::U16:
      return VcvtContract{"llvm.hivm.vcvtii.u82u16.x", false, false, true, 8};
    case VcvtElemKind::U32:
      return VcvtContract{"llvm.hivm.vcvtii.u82u32.x", false, false, true, 8};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::S8:
    switch (dst) {
    case VcvtElemKind::F16:
      return VcvtContract{"llvm.hivm.vcvtif.s82f16.x", false, false, true, 8};
    case VcvtElemKind::S16:
      return VcvtContract{"llvm.hivm.vcvtii.s82s16.x", false, false, true, 8};
    case VcvtElemKind::S32:
      return VcvtContract{"llvm.hivm.vcvtii.s82s32.x", false, false, true, 8};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::U16:
    switch (dst) {
    case VcvtElemKind::U8:
      return VcvtContract{"llvm.hivm.vcvtii.u162u8.x", false, true, true, 16};
    case VcvtElemKind::U32:
      return VcvtContract{"llvm.hivm.vcvtii.u162u32.x", false, false, true, 16};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::S16:
    switch (dst) {
    case VcvtElemKind::F16:
      return VcvtContract{"llvm.hivm.vcvtif.s162f16.x", true, false, false, 16};
    case VcvtElemKind::F32:
      return VcvtContract{"llvm.hivm.vcvtif.s162f32.x", false, false, true, 16};
    case VcvtElemKind::U8:
      return VcvtContract{"llvm.hivm.vcvtii.s162u8.x", false, true, true, 16};
    case VcvtElemKind::U32:
      return VcvtContract{"llvm.hivm.vcvtii.s162u32.x", false, false, true, 16};
    case VcvtElemKind::S32:
      return VcvtContract{"llvm.hivm.vcvtii.s162s32.x", false, false, true, 16};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::U32:
    switch (dst) {
    case VcvtElemKind::U8:
      return VcvtContract{"llvm.hivm.vcvtii.u322u8.x", false, true, true, 32};
    case VcvtElemKind::U16:
      return VcvtContract{"llvm.hivm.vcvtii.u322u16.x", false, true, true, 32};
    case VcvtElemKind::S16:
      return VcvtContract{"llvm.hivm.vcvtii.u322s16.x", false, true, true, 32};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::S32:
    switch (dst) {
    case VcvtElemKind::F32:
      return VcvtContract{"llvm.hivm.vcvtif.s322f32.x", true, false, false, 32};
    case VcvtElemKind::U8:
      return VcvtContract{"llvm.hivm.vcvtii.s322u8.x", false, true, true, 32};
    case VcvtElemKind::U16:
      return VcvtContract{"llvm.hivm.vcvtii.s322u16.x", false, true, true, 32};
    case VcvtElemKind::S16:
      return VcvtContract{"llvm.hivm.vcvtii.s322s16.x", false, true, true, 32};
    case VcvtElemKind::S64:
      return VcvtContract{"llvm.hivm.vcvtii.s322s64.x", false, false, true, 32};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::S64:
    switch (dst) {
    case VcvtElemKind::F32:
      return VcvtContract{"llvm.hivm.vcvtif.s642f32.x", true, false, true, 32};
    case VcvtElemKind::S32:
      return VcvtContract{"llvm.hivm.vcvtii.s642s32.x", false, true, true, 32};
    default:
      return std::nullopt;
    }
  case VcvtElemKind::Invalid:
    return std::nullopt;
  }
  return std::nullopt;
}

static std::optional<uint64_t> parseHiLoPartImmediate(StringRef part) {
  if (part == "LOWER")
    return 0; // __cce_simd::HiloPart::Lower
  if (part == "HIGHER")
    return 1; // __cce_simd::HiloPart::Higher
  return std::nullopt;
}

static std::optional<uint64_t> parsePredicatePatternImmediate(StringRef pattern) {
  if (pattern == "PAT_ALL")
    return 0;
  if (pattern == "PAT_VL1")
    return 1;
  if (pattern == "PAT_VL2")
    return 2;
  if (pattern == "PAT_VL3")
    return 3;
  if (pattern == "PAT_VL4")
    return 4;
  if (pattern == "PAT_VL8")
    return 5;
  if (pattern == "PAT_VL16")
    return 6;
  if (pattern == "PAT_VL32")
    return 7;
  if (pattern == "PAT_VL64")
    return 8;
  if (pattern == "PAT_VL128")
    return 9;
  if (pattern == "PAT_M3")
    return 10;
  if (pattern == "PAT_M4")
    return 11;
  if (pattern == "PAT_H")
    return 12;
  if (pattern == "PAT_Q")
    return 13;
  if (pattern == "PAT_ALLF")
    return 15;
  return std::nullopt;
}

static Type getSignlessIntegerTypeWithSameWidth(Type type, Builder &builder) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return builder.getIntegerType(intType.getWidth());
  if (auto floatType = dyn_cast<FloatType>(type))
    return builder.getIntegerType(floatType.getWidth());
  return {};
}

static std::string getVbrScalarFragment(Type type) {
  if (type.isF16())
    return "f16";
  if (type.isBF16())
    return "bf16";
  if (type.isF32())
    return "f32";
  if (auto intType = dyn_cast<IntegerType>(type))
    return (intType.isUnsigned() ? "u" : "s") + std::to_string(intType.getWidth());
  return {};
}

static std::string getCopyElementFragment(Type elementType) {
  if (!elementType)
    return {};
  if (elementType.isF16())
    return "f16";
  if (elementType.isBF16())
    return "bf16";
  if (elementType.isF32())
    return "f32";
  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    switch (intType.getWidth()) {
    case 8:
      return intType.isUnsigned() ? "u8" : "s8";
    case 16:
      return intType.isUnsigned() ? "u16" : "s16";
    case 32:
      return intType.isUnsigned() ? "u32" : "s32";
    default:
      return {};
    }
  }
  return {};
}

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

static std::optional<uint64_t> parseSprImmediate(llvm::StringRef spr) {
  if (spr == "AR")
    return 74;
  return std::nullopt;
}

static std::optional<unsigned> getDistElementWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();
  if (type.isF16() || type.isBF16())
    return 16;
  if (type.isF32())
    return 32;
  if (type.isF64())
    return 64;
  return std::nullopt;
}

static std::optional<uint64_t> parseLoadDistImmediate(llvm::StringRef dist,
                                                      Type elementType) {
  auto width = getDistElementWidth(elementType);
  if (dist.empty() || dist == "NORM")
    return 0;
  if (!width)
    return std::nullopt;
  if (dist == "BRC")
    return *width == 8   ? std::optional<uint64_t>(1)
           : *width == 16 ? std::optional<uint64_t>(2)
           : *width == 32 ? std::optional<uint64_t>(3)
                          : std::nullopt;
  if (dist == "US")
    return *width == 8   ? std::optional<uint64_t>(6)
           : *width == 16 ? std::optional<uint64_t>(7)
                          : std::nullopt;
  if (dist == "DS")
    return *width == 8   ? std::optional<uint64_t>(8)
           : *width == 16 ? std::optional<uint64_t>(9)
                          : std::nullopt;
  if (dist == "UNPK")
    return *width == 8   ? std::optional<uint64_t>(13)
           : *width == 16 ? std::optional<uint64_t>(14)
           : *width == 32 ? std::optional<uint64_t>(18)
                          : std::nullopt;
  if (dist == "BRC_BLK")
    return 15;
  if (dist == "E2B")
    return *width == 16 ? std::optional<uint64_t>(16)
           : *width == 32 ? std::optional<uint64_t>(17)
                          : std::nullopt;
  if (dist == "UNPK4")
    return *width == 8 ? std::optional<uint64_t>(20) : std::nullopt;
  if (dist == "SPLT4CHN")
    return *width == 8 ? std::optional<uint64_t>(21) : std::nullopt;
  if (dist == "SPLT2CHN")
    return *width == 8   ? std::optional<uint64_t>(22)
           : *width == 16 ? std::optional<uint64_t>(23)
                          : std::nullopt;
  return std::nullopt;
}

static std::optional<uint64_t> parseLoadX2DistImmediate(llvm::StringRef dist,
                                                        Type elementType) {
  auto width = getDistElementWidth(elementType);
  if (dist == "BDINTLV")
    return 10;
  if (!width)
    return std::nullopt;
  if (dist == "DINTLV")
    return *width == 8   ? std::optional<uint64_t>(11)
           : *width == 16 ? std::optional<uint64_t>(12)
           : *width == 32 ? std::optional<uint64_t>(19)
                          : std::nullopt;
  return std::nullopt;
}

static std::optional<uint64_t> parsePredicateLoadDistImmediate(llvm::StringRef dist) {
  if (dist.empty() || dist == "NORM")
    return 0; // Dist::DIST_NORM
  if (dist == "US")
    return 1; // Dist::DIST_US
  if (dist == "DS")
    return 2; // Dist::DIST_DS
  return std::nullopt;
}

static std::optional<uint64_t> parsePredicateStoreDistImmediate(llvm::StringRef dist) {
  if (dist.empty() || dist == "NORM")
    return 0; // Dist::DIST_NORM
  if (dist == "PK")
    return 1; // Dist::DIST_PK
  return std::nullopt;
}

static Value packBlockRepeatStride(Operation *anchor, Value blockStride,
                                   Value repeatStride) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);

  Value blockI32 = castIntegerLikeTo(anchor, blockStride, builder.getI32Type());
  Value repeatI32 =
      castIntegerLikeTo(anchor, repeatStride, builder.getI32Type());
  if (!blockI32 || !repeatI32)
    return {};

  auto c16 = builder.create<arith::ConstantIntOp>(anchor->getLoc(), 16, 32);
  auto blockShifted =
      builder.create<arith::ShLIOp>(anchor->getLoc(), blockI32, c16);
  return builder
      .create<arith::OrIOp>(anchor->getLoc(), blockShifted, repeatI32)
      .getResult();
}

static std::optional<uint64_t> parseOrderImmediate(llvm::StringRef order) {
  if (order.empty() || order == "ASC")
    return 0; // INC_ORDER
  if (order == "DESC")
    return 1; // DEC_ORDER
  return std::nullopt;
}

static std::optional<uint64_t> parseStoreDistImmediate(llvm::StringRef dist,
                                                       Type elementType) {
  auto width = getDistElementWidth(elementType);
  if (dist.empty() || dist == "NORM") {
    if (!width)
      return std::nullopt;
    if (*width == 8)
      return 0; // norm_b8
    if (*width == 16)
      return 1; // norm_b16
    if (*width == 32)
      return 2; // norm_b32
    return std::nullopt;
  }
  if (!width)
    return std::nullopt;
  if (dist == "1PT")
    return *width == 8   ? std::optional<uint64_t>(3)
           : *width == 16 ? std::optional<uint64_t>(4)
           : *width == 32 ? std::optional<uint64_t>(5)
                          : std::nullopt;
  if (dist == "PK")
    return *width == 16 ? std::optional<uint64_t>(6)
           : *width == 32 ? std::optional<uint64_t>(7)
           : *width == 64 ? std::optional<uint64_t>(10)
                          : std::nullopt;
  if (dist == "PK4")
    return *width == 32 ? std::optional<uint64_t>(12) : std::nullopt;
  if (dist == "MRG4CHN")
    return *width == 8 ? std::optional<uint64_t>(13) : std::nullopt;
  if (dist == "MRG2CHN")
    return *width == 8   ? std::optional<uint64_t>(14)
           : *width == 16 ? std::optional<uint64_t>(15)
                          : std::nullopt;
  return std::nullopt;
}

static std::optional<uint64_t> parseStoreX2DistImmediate(llvm::StringRef dist,
                                                         Type elementType) {
  auto width = getDistElementWidth(elementType);
  if (!width)
    return std::nullopt;
  if (dist == "INTLV")
    return *width == 8   ? std::optional<uint64_t>(8)
           : *width == 16 ? std::optional<uint64_t>(9)
           : *width == 32 ? std::optional<uint64_t>(11)
                          : std::nullopt;
  return std::nullopt;
}

static std::optional<int32_t> parsePostModeImmediate(StringRef mode) {
  if (mode == "NO_POST_UPDATE")
    return 0;
  if (mode == "POST_UPDATE")
    return 1;
  return std::nullopt;
}

static Type convertVPTOType(Type type, Builder &builder) {
  if (auto vecType = dyn_cast<pto::VRegType>(type))
    return VectorType::get({vecType.getElementCount()}, vecType.getElementType());
  if (isa<pto::MaskType>(type))
    return VectorType::get({256}, builder.getI1Type());
  if (isa<pto::AlignType>(type))
    return VectorType::get({32}, builder.getIntegerType(8));
  if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
    return LLVM::LLVMPointerType::get(
        builder.getContext(),
        static_cast<unsigned>(ptrType.getMemorySpace().getAddressSpace()));
  }
  return type;
}

static bool hasPtoPtrType(TypeRange types) {
  return llvm::any_of(types, [](Type type) { return isa<pto::PtrType>(type); });
}

static bool hasPtoAlignType(Type type) {
  if (isa<pto::AlignType>(type))
    return true;
  if (auto functionType = dyn_cast<FunctionType>(type))
    return llvm::any_of(functionType.getInputs(), hasPtoAlignType) ||
           llvm::any_of(functionType.getResults(), hasPtoAlignType);
  return false;
}

static bool hasPtoAlignType(TypeRange types) {
  return llvm::any_of(types, [](Type type) { return hasPtoAlignType(type); });
}

static bool hasPtoMemRefMemorySpace(Type type) {
  if (auto memRefType = dyn_cast<MemRefType>(type))
    return isa<pto::AddressSpaceAttr>(memRefType.getMemorySpace());
  if (auto functionType = dyn_cast<FunctionType>(type))
    return llvm::any_of(functionType.getInputs(), hasPtoMemRefMemorySpace) ||
           llvm::any_of(functionType.getResults(), hasPtoMemRefMemorySpace);
  return false;
}

static bool hasPtoMemRefMemorySpace(TypeRange types) {
  return llvm::any_of(types, [](Type type) {
    return hasPtoMemRefMemorySpace(type);
  });
}

struct ConvertPtoMemRefSpaceCarrierOp final : ConversionPattern {
  ConvertPtoMemRefSpaceCarrierOp(TypeConverter &typeConverter,
                                 MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!hasPtoMemRefMemorySpace(op->getOperandTypes()) &&
        !hasPtoMemRefMemorySpace(op->getResultTypes()))
      return failure();
    if (op->getNumRegions() != 0)
      return rewriter.notifyMatchFailure(
          op, "region ops with PTO memref spaces are handled structurally");

    FailureOr<Operation *> converted =
        convertOpResultTypes(op, operands, *typeConverter, rewriter);
    if (failed(converted))
      return failure();
    return success();
  }
};

struct ConvertMemRefReinterpretCastSpaceOp final
    : OpConversionPattern<memref::ReinterpretCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedResultType = getTypeConverter()->convertType(op.getType());
    auto memRefResultType = dyn_cast_or_null<MemRefType>(convertedResultType);
    if (!memRefResultType)
      return rewriter.notifyMatchFailure(op, "expected memref result type");

    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, memRefResultType, adaptor.getSource(), adaptor.getOffsets(),
        adaptor.getSizes(), adaptor.getStrides(), op.getStaticOffsets(),
        op.getStaticSizes(), op.getStaticStrides());
    return success();
  }
};

struct ConvertMemRefSubViewSpaceOp final
    : OpConversionPattern<memref::SubViewOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::SubViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedResultType = getTypeConverter()->convertType(op.getType());
    auto memRefResultType = dyn_cast_or_null<MemRefType>(convertedResultType);
    if (!memRefResultType)
      return rewriter.notifyMatchFailure(op, "expected memref result type");

    rewriter.replaceOpWithNewOp<memref::SubViewOp>(
        op, memRefResultType, adaptor.getSource(), op.getMixedOffsets(),
        op.getMixedSizes(), op.getMixedStrides());
    return success();
  }
};

struct ConvertMemRefSpaceUnrealizedCastOp final
    : OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();
    if (!hasPtoMemRefMemorySpace(op->getOperandTypes()) &&
        !hasPtoMemRefMemorySpace(op->getResultTypes()))
      return failure();

    Type convertedResultType = getTypeConverter()->convertType(op.getResult(0).getType());
    if (!convertedResultType)
      return failure();

    Value input = adaptor.getOperands().front();
    if (input.getType() == convertedResultType) {
      rewriter.replaceOp(op, input);
      return success();
    }
    return failure();
  }
};

static LogicalResult normalizePtoMemRefSpaces(ModuleOp module,
                                              llvm::raw_ostream &diagOS) {
  MLIRContext *context = module.getContext();
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addConversion([&](MemRefType type) -> Type {
    auto addrSpace = dyn_cast_or_null<pto::AddressSpaceAttr>(type.getMemorySpace());
    if (!addrSpace)
      return type;
    return MemRefType::get(
        type.getShape(), type.getElementType(), type.getLayout(),
        IntegerAttr::get(IntegerType::get(context, 64),
                         static_cast<int64_t>(addrSpace.getAddressSpace())));
  });
  typeConverter.addTypeAttributeConversion(
      [](MemRefType, pto::AddressSpaceAttr attr) -> Attribute {
        return IntegerAttr::get(IntegerType::get(attr.getContext(), 64),
                                static_cast<int64_t>(attr.getAddressSpace()));
      });
  auto materializeMemRefCast = [](OpBuilder &builder, Type resultType,
                                  ValueRange inputs, Location loc) -> Value {
    if (inputs.size() != 1)
      return {};
    return builder
        .create<UnrealizedConversionCastOp>(loc, TypeRange{resultType}, inputs)
        .getResult(0);
  };
  typeConverter.addSourceMaterialization(materializeMemRefCast);
  typeConverter.addTargetMaterialization(materializeMemRefCast);
  typeConverter.addArgumentMaterialization(materializeMemRefCast);

  ConversionTarget target(*context);
  target.addLegalOp<ModuleOp>();
  target.addDynamicallyLegalOp<func::FuncOp>(
      [&](func::FuncOp op) {
        return typeConverter.isSignatureLegal(op.getFunctionType()) &&
               typeConverter.isLegal(&op.getBody());
      });
  target.addDynamicallyLegalOp<func::CallOp>(
      [&](func::CallOp op) { return typeConverter.isLegal(op); });
  target.addDynamicallyLegalOp<func::ReturnOp>(
      [&](func::ReturnOp op) { return typeConverter.isLegal(op); });
  target.addDynamicallyLegalOp<cf::BranchOp, cf::CondBranchOp>(
      [&](Operation *op) {
        return isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                                typeConverter);
      });
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return typeConverter.isLegal(op->getOperandTypes()) &&
           typeConverter.isLegal(op->getResultTypes());
  });

  RewritePatternSet patterns(context);
  scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns,
                                                       target);
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
  patterns.add<ConvertMemRefReinterpretCastSpaceOp, ConvertMemRefSubViewSpaceOp,
               ConvertMemRefSpaceUnrealizedCastOp>(
      typeConverter, context);
  patterns.add<ConvertPtoMemRefSpaceCarrierOp>(typeConverter, context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    diagOS << "VPTO LLVM emission failed: memref address-space normalization "
              "failed\n";
    return failure();
  }

  SmallVector<UnrealizedConversionCastOp> castsToFold;
  module.walk([&](UnrealizedConversionCastOp castOp) {
    if (castOp->getNumOperands() != 1 || castOp->getNumResults() != 1)
      return;
    if (!hasPtoMemRefMemorySpace(castOp->getOperandTypes()) &&
        !hasPtoMemRefMemorySpace(castOp->getResultTypes()))
      return;
    Type convertedResultType = typeConverter.convertType(castOp.getResult(0).getType());
    if (convertedResultType && convertedResultType == castOp.getOperand(0).getType())
      castsToFold.push_back(castOp);
  });
  for (UnrealizedConversionCastOp castOp : castsToFold) {
    castOp.getResult(0).replaceAllUsesWith(castOp.getOperand(0));
    castOp.erase();
  }

  WalkResult leftover = module.walk([&](Operation *op) {
    if (hasPtoMemRefMemorySpace(op->getOperandTypes()) ||
        hasPtoMemRefMemorySpace(op->getResultTypes())) {
      diagOS << "VPTO LLVM emission failed: residual PTO memref address space on op "
             << op->getName().getStringRef() << "\n";
      op->print(diagOS);
      diagOS << "\n";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (leftover.wasInterrupted())
    return failure();
  return success();
}

struct ConvertPtoAddPtrOp final : OpConversionPattern<pto::AddPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto convertedResultType =
        getTypeConverter()->convertType(op.getResult().getType());
    auto llvmPtrType = dyn_cast_or_null<LLVM::LLVMPointerType>(convertedResultType);
    if (!llvmPtrType)
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer result type");

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex())
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(),
                                                     rewriter.getI64Type(), offset);

    auto gep = rewriter.create<LLVM::GEPOp>(
        op.getLoc(), llvmPtrType, cast<pto::PtrType>(op.getPtr().getType()).getElementType(),
        adaptor.getPtr(), ValueRange{offset});
    rewriter.replaceOp(op, gep.getResult());
    return success();
  }
};

struct ConvertPtoCastPtrOp final : OpConversionPattern<pto::CastPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::CastPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedResultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedResultType)
      return rewriter.notifyMatchFailure(op, "could not convert castptr result type");

    Value input = adaptor.getInput();
    Type inputType = input.getType();
    if (inputType == convertedResultType) {
      rewriter.replaceOp(op, input);
      return success();
    }

    if (auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(convertedResultType)) {
      if (isa<IntegerType>(inputType)) {
        auto intToPtr =
            rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), llvmPtrType, input);
        rewriter.replaceOp(op, intToPtr.getResult());
        return success();
      }
      auto sourcePtrType = dyn_cast<LLVM::LLVMPointerType>(inputType);
      if (!sourcePtrType)
        return rewriter.notifyMatchFailure(op, "expected integer or LLVM pointer input");
      if (sourcePtrType.getAddressSpace() == llvmPtrType.getAddressSpace()) {
        auto bitcast =
            rewriter.create<LLVM::BitcastOp>(op.getLoc(), llvmPtrType, input);
        rewriter.replaceOp(op, bitcast.getResult());
        return success();
      }
      return rewriter.notifyMatchFailure(op, "cross-address-space ptr casts are unsupported");
    }

    if (auto resultIntType = dyn_cast<IntegerType>(convertedResultType)) {
      if (auto inputPtrType = dyn_cast<LLVM::LLVMPointerType>(inputType)) {
        rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, resultIntType, input);
        return success();
      }
      if (auto inputIntType = dyn_cast<IntegerType>(inputType)) {
        unsigned srcWidth = inputIntType.getWidth();
        unsigned dstWidth = resultIntType.getWidth();
        if (srcWidth == dstWidth) {
          rewriter.replaceOp(op, input);
          return success();
        }
        if (srcWidth < dstWidth) {
          rewriter.replaceOpWithNewOp<LLVM::ZExtOp>(op, resultIntType, input);
          return success();
        }
        rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, resultIntType, input);
        return success();
      }
    }

    return rewriter.notifyMatchFailure(op, "unsupported castptr conversion");
  }
};

struct ConvertPtoLoadScalarOp final : OpConversionPattern<pto::LoadScalarOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::LoadScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!llvmPtrType)
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex())
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(),
                                                     rewriter.getI64Type(), offset);

    Value elemPtr = adaptor.getPtr();
    if (!matchPattern(offset, m_Zero())) {
      elemPtr = rewriter.create<LLVM::GEPOp>(op.getLoc(), llvmPtrType,
                                             op.getValue().getType(), adaptor.getPtr(),
                                             ValueRange{offset});
    }

    auto getNaturalAlignment = [&](Type type) -> unsigned {
      unsigned alignBytes = 0;
      if (auto intType = dyn_cast<IntegerType>(type)) {
        alignBytes = llvm::divideCeil(unsigned(intType.getWidth()), 8u);
      } else if (type.isF16() || type.isBF16()) {
        alignBytes = 2;
      } else if (type.isF32()) {
        alignBytes = 4;
      } else if (type.isF64()) {
        alignBytes = 8;
      }
      return alignBytes;
    };

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        op, op.getValue().getType(), elemPtr,
        getNaturalAlignment(op.getValue().getType()));
    return success();
  }
};

struct ConvertPtoStoreScalarOp final : OpConversionPattern<pto::StoreScalarOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::StoreScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!llvmPtrType)
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer operand");

    Value offset = adaptor.getOffset();
    if (offset.getType().isIndex())
      offset = rewriter.create<arith::IndexCastUIOp>(op.getLoc(),
                                                     rewriter.getI64Type(), offset);

    Value elemPtr = adaptor.getPtr();
    if (!matchPattern(offset, m_Zero())) {
      elemPtr = rewriter.create<LLVM::GEPOp>(op.getLoc(), llvmPtrType,
                                             adaptor.getValue().getType(),
                                             adaptor.getPtr(), ValueRange{offset});
    }

    auto getNaturalAlignment = [&](Type type) -> unsigned {
      unsigned alignBytes = 0;
      if (auto intType = dyn_cast<IntegerType>(type)) {
        alignBytes = llvm::divideCeil(unsigned(intType.getWidth()), 8u);
      } else if (type.isF16() || type.isBF16()) {
        alignBytes = 2;
      } else if (type.isF32()) {
        alignBytes = 4;
      } else if (type.isF64()) {
        alignBytes = 8;
      }
      return alignBytes;
    };

    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(
        op, adaptor.getValue(), elemPtr,
        getNaturalAlignment(adaptor.getValue().getType()));
    return success();
  }
};

struct ConvertPtoUnrealizedCastOp final
    : OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "only 1:1 casts are supported");

    Type convertedResultType =
        getTypeConverter()->convertType(op.getResult(0).getType());
    if (!convertedResultType)
      return rewriter.notifyMatchFailure(op, "could not convert cast result type");

    Value input = adaptor.getOperands().front();
    if (auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(convertedResultType)) {
      if (input.getType().isInteger(64)) {
        rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, llvmPtrType, input);
        return success();
      }
    }
    if (input.getType() == convertedResultType) {
      rewriter.replaceOp(op, input);
      return success();
    }

    auto cast = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), TypeRange{convertedResultType}, input);
    rewriter.replaceOp(op, cast.getResults());
    return success();
  }
};

struct ConvertPtoPtrCarrierOp final : ConversionPattern {
  ConvertPtoPtrCarrierOp(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<pto::AddPtrOp, pto::CastPtrOp, pto::LoadScalarOp, pto::StoreScalarOp,
            UnrealizedConversionCastOp>(op))
      return failure();
    if (!hasPtoPtrType(op->getOperandTypes()) && !hasPtoPtrType(op->getResultTypes()))
      return failure();
    if (op->getNumRegions() != 0)
      return rewriter.notifyMatchFailure(op, "region ops with pto.ptr are unsupported");

    SmallVector<Type> convertedResultTypes;
    if (failed(typeConverter->convertTypes(op->getResultTypes(), convertedResultTypes)))
      return failure();

    OperationState state(op->getLoc(), op->getName().getStringRef());
    state.addOperands(operands);
    state.addTypes(convertedResultTypes);
    state.addAttributes(op->getAttrs());
    state.addSuccessors(op->getSuccessors());

    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct ConvertPtoAlignUnrealizedCastOp final
    : OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();
    if (!hasPtoAlignType(op->getOperandTypes()) &&
        !hasPtoAlignType(op->getResultTypes()))
      return failure();

    Type convertedResultType =
        getTypeConverter()->convertType(op.getResult(0).getType());
    if (!convertedResultType)
      return failure();

    Value input = adaptor.getOperands().front();
    if (input.getType() == convertedResultType) {
      rewriter.replaceOp(op, input);
      return success();
    }
    return failure();
  }
};

struct ConvertPtoAlignCarrierOp final : ConversionPattern {
  ConvertPtoAlignCarrierOp(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<UnrealizedConversionCastOp>(op))
      return failure();
    if (!hasPtoAlignType(op->getOperandTypes()) &&
        !hasPtoAlignType(op->getResultTypes()))
      return failure();
    if (op->getNumRegions() != 0)
      return rewriter.notifyMatchFailure(op,
                                         "region ops with pto.align are handled structurally");

    SmallVector<Type> convertedResultTypes;
    if (failed(typeConverter->convertTypes(op->getResultTypes(),
                                           convertedResultTypes)))
      return failure();

    OperationState state(op->getLoc(), op->getName().getStringRef());
    state.addOperands(operands);
    state.addTypes(convertedResultTypes);
    state.addAttributes(op->getAttrs());
    state.addSuccessors(op->getSuccessors());

    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

static LogicalResult normalizePtoPtrsToLLVM(ModuleOp module, llvm::raw_ostream &diagOS) {
  MLIRContext *context = module.getContext();

  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    if (funcOp.isExternal())
      continue;
  }

  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addConversion([&](pto::PtrType type) -> Type {
    return LLVM::LLVMPointerType::get(
        context, static_cast<unsigned>(type.getMemorySpace().getAddressSpace()));
  });
  auto materializePtrCast = [](OpBuilder &builder, Type resultType,
                               ValueRange inputs, Location loc) -> Value {
    if (inputs.size() != 1)
      return {};
    return builder
        .create<UnrealizedConversionCastOp>(loc, TypeRange{resultType}, inputs)
        .getResult(0);
  };
  typeConverter.addSourceMaterialization(materializePtrCast);
  typeConverter.addTargetMaterialization(materializePtrCast);
  typeConverter.addArgumentMaterialization(materializePtrCast);

  ConversionTarget target(*context);
  target.addLegalOp<ModuleOp>();
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType()) &&
           typeConverter.isLegal(&op.getBody());
  });
  target.addDynamicallyLegalOp<func::CallOp>(
      [&](func::CallOp op) { return typeConverter.isLegal(op); });
  target.addDynamicallyLegalOp<func::ReturnOp>(
      [&](func::ReturnOp op) { return typeConverter.isLegal(op); });
  target.addDynamicallyLegalOp<cf::BranchOp, cf::CondBranchOp>(
      [&](Operation *op) {
        return isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                                typeConverter);
      });
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return typeConverter.isLegal(op->getOperandTypes()) &&
           typeConverter.isLegal(op->getResultTypes());
  });
  target.addIllegalOp<pto::AddPtrOp, pto::CastPtrOp, pto::LoadScalarOp,
                      pto::StoreScalarOp>();
  target.addDynamicallyLegalOp<UnrealizedConversionCastOp>([](UnrealizedConversionCastOp op) {
    return !hasPtoPtrType(op->getOperandTypes()) && !hasPtoPtrType(op->getResultTypes());
  });

  RewritePatternSet patterns(context);
  scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns,
                                                       target);
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
  patterns.add<ConvertPtoAddPtrOp, ConvertPtoCastPtrOp, ConvertPtoLoadScalarOp,
               ConvertPtoStoreScalarOp, ConvertPtoUnrealizedCastOp>(
      typeConverter, context);
  patterns.add<ConvertPtoPtrCarrierOp>(typeConverter, context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    diagOS << "VPTO LLVM emission failed: pto.ptr normalization failed\n";
    return failure();
  }

  SmallVector<UnrealizedConversionCastOp> castsToFold;
  module.walk([&](UnrealizedConversionCastOp castOp) {
    if (castOp->getNumOperands() != 1 || castOp->getNumResults() != 1)
      return;
    if (!hasPtoPtrType(castOp->getOperandTypes()) &&
        !hasPtoPtrType(castOp->getResultTypes()))
      return;
    Type convertedResultType = typeConverter.convertType(castOp.getResult(0).getType());
    if (convertedResultType && convertedResultType == castOp.getOperand(0).getType())
      castsToFold.push_back(castOp);
  });
  for (UnrealizedConversionCastOp castOp : castsToFold) {
    castOp.getResult(0).replaceAllUsesWith(castOp.getOperand(0));
    castOp.erase();
  }

  return success();
}

static LogicalResult normalizePtoAlignsToABI(ModuleOp module,
                                             llvm::raw_ostream &diagOS) {
  MLIRContext *context = module.getContext();

  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addConversion([&](pto::AlignType type) -> Type {
    return VectorType::get({32}, IntegerType::get(context, 8));
  });
  auto materializeAlignCast = [](OpBuilder &builder, Type resultType,
                                 ValueRange inputs, Location loc) -> Value {
    if (inputs.size() != 1)
      return {};
    return builder
        .create<UnrealizedConversionCastOp>(loc, TypeRange{resultType}, inputs)
        .getResult(0);
  };
  typeConverter.addSourceMaterialization(materializeAlignCast);
  typeConverter.addTargetMaterialization(materializeAlignCast);
  typeConverter.addArgumentMaterialization(materializeAlignCast);

  ConversionTarget target(*context);
  target.addLegalOp<ModuleOp>();
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType()) &&
           typeConverter.isLegal(&op.getBody());
  });
  target.addDynamicallyLegalOp<func::CallOp>(
      [&](func::CallOp op) { return typeConverter.isLegal(op); });
  target.addDynamicallyLegalOp<func::ReturnOp>(
      [&](func::ReturnOp op) { return typeConverter.isLegal(op); });
  target.addDynamicallyLegalOp<cf::BranchOp, cf::CondBranchOp>(
      [&](Operation *op) {
        return isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                                typeConverter);
      });
  target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
      [&](UnrealizedConversionCastOp op) {
        return !hasPtoAlignType(op->getOperandTypes()) &&
               !hasPtoAlignType(op->getResultTypes());
      });
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return typeConverter.isLegal(op->getOperandTypes()) &&
           typeConverter.isLegal(op->getResultTypes());
  });

  RewritePatternSet patterns(context);
  scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns,
                                                       target);
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
  patterns.add<ConvertPtoAlignUnrealizedCastOp, ConvertPtoAlignCarrierOp>(
      typeConverter, context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    diagOS << "VPTO LLVM emission failed: pto.align normalization failed\n";
    return failure();
  }

  SmallVector<UnrealizedConversionCastOp> castsToFold;
  module.walk([&](UnrealizedConversionCastOp castOp) {
    if (castOp->getNumOperands() != 1 || castOp->getNumResults() != 1)
      return;
    if (!hasPtoAlignType(castOp->getOperandTypes()) &&
        !hasPtoAlignType(castOp->getResultTypes()))
      return;
    Type convertedResultType =
        typeConverter.convertType(castOp.getResult(0).getType());
    if (convertedResultType &&
        convertedResultType == castOp.getOperand(0).getType())
      castsToFold.push_back(castOp);
  });
  for (UnrealizedConversionCastOp castOp : castsToFold) {
    castOp.getResult(0).replaceAllUsesWith(castOp.getOperand(0));
    castOp.erase();
  }

  WalkResult leftover = module.walk([&](Operation *op) {
    if (hasPtoAlignType(op->getOperandTypes()) ||
        hasPtoAlignType(op->getResultTypes())) {
      diagOS << "VPTO LLVM emission failed: residual pto.align type on op "
             << op->getName().getStringRef() << "\n";
      op->print(diagOS);
      diagOS << "\n";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (leftover.wasInterrupted())
    return failure();
  return success();
}

static Type getElementTypeFromVectorLike(Type type) {
  if (auto vecType = dyn_cast<pto::VRegType>(type))
    return vecType.getElementType();
  if (auto vecType = dyn_cast<VectorType>(type))
    return vecType.getElementType();
  return {};
}

static Type getElementTypeFromPointerLike(Type type) {
  if (auto ptrType = dyn_cast<pto::PtrType>(type))
    return ptrType.getElementType();
  if (auto memRefType = dyn_cast<MemRefType>(type))
    return memRefType.getElementType();
  return {};
}

static Type getElementTypeFromABIValue(Value value) {
  if (!value)
    return {};
  if (Type direct = getElementTypeFromPointerLike(value.getType()))
    return direct;
  return {};
}

static std::optional<int64_t> getElementCountFromVectorLike(Type type) {
  if (auto vecType = dyn_cast<pto::VRegType>(type))
    return vecType.getElementCount();
  if (auto vecType = dyn_cast<VectorType>(type)) {
    if (vecType.getRank() != 1)
      return std::nullopt;
    return vecType.getShape().front();
  }
  return std::nullopt;
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

static Value buildBridgeCast(OpBuilder &builder, Location loc, Value input,
                             Type targetType) {
  if (input.getType() == targetType)
    return input;
  if ((isa<pto::PtrType>(input.getType()) &&
       isa<LLVM::LLVMPointerType>(targetType)) ||
      (isa<LLVM::LLVMPointerType>(input.getType()) &&
       isa<pto::PtrType>(targetType))) {
    return builder
        .create<UnrealizedConversionCastOp>(loc, TypeRange{targetType}, input)
        .getResult(0);
  }
  return builder.create<arith::BitcastOp>(loc, targetType, input).getResult();
}

static FailureOr<Value> requirePointerABIAddress(Operation *anchor, Value address,
                                                 llvm::raw_ostream &diagOS) {
  if (isa<LLVM::LLVMPointerType>(address.getType()))
    return address;
  if (auto ptrType = dyn_cast<pto::PtrType>(address.getType())) {
    OpBuilder builder(anchor);
    builder.setInsertionPoint(anchor);
    auto llvmPtrType = LLVM::LLVMPointerType::get(
        builder.getContext(),
        static_cast<unsigned>(ptrType.getMemorySpace().getAddressSpace()));
    Value abiAddress = buildBridgeCast(builder, anchor->getLoc(), address, llvmPtrType);
    return abiAddress;
  }

  diagOS << "VPTO LLVM emission failed: expected pointer-ABI address after "
            "pre-emit canonicalization, but saw "
         << address.getType() << " on op ";
  anchor->print(diagOS);
  diagOS << "\n";
  return failure();
}

static FailureOr<Value> materializeAlignABIValue(Operation *anchor, Value align,
                                                 llvm::raw_ostream &diagOS) {
  if (!align)
    return failure();
  if (isa<VectorType>(align.getType()))
    return align;

  auto alignType = dyn_cast<pto::AlignType>(align.getType());
  if (!alignType) {
    diagOS << "VPTO LLVM emission failed: expected align ABI value, but saw "
           << align.getType() << "\n";
    return failure();
  }

  Operation *def = align.getDefiningOp();
  if (!def) {
    diagOS << "VPTO LLVM emission failed: unsupported non-ABI align producer "
           << "<block-argument>"
           << " for " << alignType << "\n";
    return failure();
  }

  auto defName = def->getName().getStringRef();
  if (defName != "pto.init_align" && defName != "ub.poison") {
    diagOS << "VPTO LLVM emission failed: unsupported non-ABI align producer ";
    diagOS << def->getName();
    diagOS << " for " << alignType << "\n";
    return failure();
  }

  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  auto abiType = cast<VectorType>(convertVPTOType(alignType, builder));
  auto zeroAttr = DenseElementsAttr::get(abiType, builder.getI8IntegerAttr(0));
  return builder.create<arith::ConstantOp>(anchor->getLoc(), abiType, zeroAttr)
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

static Value getI16Constant(OpBuilder &builder, Location loc, uint64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI16IntegerAttr(value))
      .getResult();
}

static Value buildAllTrueMask(OpBuilder &builder, Location loc) {
  auto maskType = VectorType::get({256}, builder.getI1Type());
  auto attr = DenseElementsAttr::get(maskType, true);
  return builder.create<arith::ConstantOp>(loc, maskType, attr).getResult();
}

static FailureOr<Value> buildPltB8Mask(IRRewriter &builder, ModuleOp module,
                                       Location loc, uint64_t laneCount,
                                       llvm::raw_ostream &diagOS) {
  Value laneCountValue = getI32Constant(builder, loc, laneCount);
  auto maskType = VectorType::get({256}, builder.getI1Type());
  auto funcType =
      builder.getFunctionType({builder.getI32Type()}, {maskType, builder.getI32Type()});
  auto callee =
      getOrCreateExternalFunc(module, "llvm.hivm.plt.b8.v300", funcType);
  auto call = builder.create<func::CallOp>(loc, callee, ValueRange{laneCountValue});
  return call.getResult(0);
}

static FailureOr<Value> buildPltB32Mask(IRRewriter &builder, ModuleOp module,
                                        Location loc, uint64_t laneCount,
                                        llvm::raw_ostream &diagOS) {
  // Keep this helper narrowly scoped to the verified HIVM form we have observed
  // in emitc-generated device IR. For Expands/TExpandS, installed PTO source
  // calls pset_b32(PAT_ALL), but save-temps from the working emitc path show
  // that the compiler frontend does not preserve a pset-shaped HIVM intrinsic
  // here. Instead, the full-lane mask is materialized in the final device IR as
  // llvm.hivm.plt.b32.v300(i32 64), i.e. a canonical "all 64 b32 lanes active"
  // form that the backend accepts. Reproduce that observed lowering here; do
  // not treat it as evidence that pset_b32 and plt_b32 are generally
  // interchangeable at the source or VPTO level.
  Value laneCountValue = getI32Constant(builder, loc, laneCount);
  auto maskType = VectorType::get({256}, builder.getI1Type());
  auto funcType =
      builder.getFunctionType({builder.getI32Type()}, {maskType, builder.getI32Type()});
  auto callee =
      getOrCreateExternalFunc(module, "llvm.hivm.plt.b32.v300", funcType);
  auto call = builder.create<func::CallOp>(loc, callee, ValueRange{laneCountValue});
  return call.getResult(0);
}

static FailureOr<Value> buildPltB16Mask(IRRewriter &builder, ModuleOp module,
                                        Location loc, uint64_t laneCount,
                                        llvm::raw_ostream &diagOS) {
  Value laneCountValue = getI32Constant(builder, loc, laneCount);
  auto maskType = VectorType::get({256}, builder.getI1Type());
  auto funcType =
      builder.getFunctionType({builder.getI32Type()}, {maskType, builder.getI32Type()});
  auto callee =
      getOrCreateExternalFunc(module, "llvm.hivm.plt.b16.v300", funcType);
  auto call = builder.create<func::CallOp>(loc, callee, ValueRange{laneCountValue});
  return call.getResult(0);
}

static FailureOr<Value> buildDynamicPltMask(IRRewriter &builder, ModuleOp module,
                                            Location loc, Value laneCount,
                                            Type vectorElemType,
                                            llvm::raw_ostream &diagOS) {
  Value laneCountI32 = laneCount;
  Type i32Type = builder.getI32Type();
  if (laneCountI32.getType() != i32Type) {
    if (laneCountI32.getType().isIndex()) {
      laneCountI32 =
          builder.create<arith::IndexCastOp>(loc, i32Type, laneCountI32);
    } else if (auto sourceInt = dyn_cast<IntegerType>(laneCountI32.getType())) {
      auto targetInt = cast<IntegerType>(i32Type);
      if (sourceInt.getWidth() < targetInt.getWidth()) {
        laneCountI32 =
            builder.create<arith::ExtUIOp>(loc, i32Type, laneCountI32);
      } else if (sourceInt.getWidth() > targetInt.getWidth()) {
        laneCountI32 =
            builder.create<arith::TruncIOp>(loc, i32Type, laneCountI32);
      }
    } else {
      return failure();
    }
  }

  auto maskType = VectorType::get({256}, builder.getI1Type());
  auto funcType =
      builder.getFunctionType({builder.getI32Type()}, {maskType, builder.getI32Type()});

  StringRef calleeName;
  if (vectorElemType.isF32()) {
    calleeName = "llvm.hivm.plt.b32.v300";
  } else if (vectorElemType.isF16() || vectorElemType.isBF16()) {
    calleeName = "llvm.hivm.plt.b16.v300";
  } else if (auto intType = dyn_cast<IntegerType>(vectorElemType)) {
    if (intType.getWidth() == 32)
      calleeName = "llvm.hivm.plt.b32.v300";
    else if (intType.getWidth() == 16)
      calleeName = "llvm.hivm.plt.b16.v300";
  }

  if (calleeName.empty()) {
    diagOS << "VPTO LLVM emission failed: unsupported dynamic plt mask element "
              "type "
           << vectorElemType << "\n";
    return failure();
  }

  auto callee = getOrCreateExternalFunc(module, calleeName, funcType);
  auto call = builder.create<func::CallOp>(loc, callee, ValueRange{laneCountI32});
  return call.getResult(0);
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
packCopyGmToUbConfig0(Operation *anchor, pto::CopyGmToUbufOp op,
                      ValueRange operands) {
  if (operands.size() != 11)
    return failure();

  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  auto getI64Operand = [&](unsigned idx) -> Value {
    return castIntegerLikeTo(anchor, operands[idx], builder.getI64Type());
  };

  Value sid = getI64Operand(2);
  Value nBurst = getI64Operand(3);
  Value lenBurst = getI64Operand(4);
  Value leftPadding = getI64Operand(5);
  Value rightPadding = getI64Operand(6);
  Value dataSelect = castIntegerLikeTo(anchor, operands[7], builder.getI64Type());
  Value cacheCtl = getI64Operand(8);
  if (!sid || !nBurst || !lenBurst || !leftPadding || !rightPadding ||
      !dataSelect || !cacheCtl)
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
  config = bitOr(config, shl(leftPadding, 46));
  config = bitOr(config, shl(rightPadding, 52));
  config = bitOr(config, shl(dataSelect, 58));
  config = bitOr(config, shl(cacheCtl, 60));
  return config;
}

static FailureOr<Value>
packCopyGmToUbConfig1(Operation *anchor, ValueRange operands) {
  if (operands.size() != 11)
    return failure();
  return packLoopPair(anchor, operands[9], operands[10]);
}

static FailureOr<Value>
packCopyUbToGmConfig0(Operation *anchor, ValueRange operands) {
  if (operands.size() != 8)
    return failure();

  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  auto getI64Operand = [&](unsigned idx) -> Value {
    return castIntegerLikeTo(anchor, operands[idx], builder.getI64Type());
  };

  Value sid = getI64Operand(2);
  Value nBurst = getI64Operand(3);
  Value lenBurst = getI64Operand(4);
  Value reserved = getI64Operand(5);
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
  if (operands.size() != 8)
    return failure();
  return packLoopPair(anchor, operands[6], operands[7]);
}

static FailureOr<Value> packVbitsortConfig(Operation *anchor, Value repeatTimes) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  Value repeatI64 = castIntegerLikeTo(anchor, repeatTimes, builder.getI64Type());
  if (!repeatI64)
    return failure();
  return builder
      .create<arith::ShLIOp>(loc, repeatI64, getI64Constant(builder, loc, 56))
      .getResult();
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

static FailureOr<std::string> getConfirmedCallee(Operation *op) {
  if (isa<pto::SetLoop2StrideOutToUbOp>(op))
    return std::string("llvm.hivm.SET.LOOP2.STRIDE.OUTTOUB");
  if (isa<pto::SetLoop1StrideOutToUbOp>(op))
    return std::string("llvm.hivm.SET.LOOP1.STRIDE.OUTTOUB");
  if (isa<pto::SetLoopSizeOutToUbOp>(op))
    return std::string("llvm.hivm.SET.LOOP.SIZE.OUTTOUB");
  if (isa<pto::SetLoop2StrideUbToOutOp>(op))
    return std::string("llvm.hivm.SET.LOOP2.STRIDE.UBTOOUT");
  if (isa<pto::SetLoop1StrideUbToOutOp>(op))
    return std::string("llvm.hivm.SET.LOOP1.STRIDE.UBTOOUT");
  if (isa<pto::SetLoopSizeUbToOutOp>(op))
    return std::string("llvm.hivm.SET.LOOP.SIZE.UBTOOUT");
  if (auto copy = dyn_cast<pto::CopyGmToUbufOp>(op)) {
    Type elementType = getElementTypeFromABIValue(copy.getSource());
    if (!elementType)
      elementType = getElementTypeFromABIValue(copy.getDestination());
    std::string elem = getCopyElementFragment(elementType);
    if (elem.empty())
      return failure();
    return "llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2." + elem + ".DV";
  }
  if (isa<pto::CopyUbufToGmOp>(op))
    return std::string("llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.DV");
  if (isa<pto::SetFlagOp>(op))
    return std::string("llvm.hivm.SET.FLAG.IMM");
  if (isa<pto::WaitFlagOp>(op))
    return std::string("llvm.hivm.WAIT.FLAG.IMM");
  if (isa<pto::BarrierOp>(op))
    return std::string("llvm.hivm.BARRIER");
  if (isa<pto::GetBlockIdxOp>(op))
    return std::string("llvm.hivm.GET.BLOCK.IDX");
  if (isa<pto::GetSubBlockIdxOp>(op))
    return std::string("llvm.hivm.GET.SUBBLOCKID");
  if (isa<pto::GetBlockNumOp>(op))
    return std::string("llvm.hivm.GET.BLOCK.NUM");
  if (isa<pto::GetSubBlockNumOp>(op))
    return std::string("llvm.hivm.GET.SUBBLOCKDIM");
  if (isa<pto::SprclrOp>(op))
    return std::string("llvm.hivm.sprclr");
  if (isa<pto::PltB8Op>(op))
    return std::string("llvm.hivm.plt.b8.v300");
  if (isa<pto::PltB32Op>(op))
    return std::string("llvm.hivm.plt.b32.v300");
  if (isa<pto::PltB16Op>(op))
    return std::string("llvm.hivm.plt.b16.v300");
  if (isa<pto::PsetB8Op>(op))
    return std::string("llvm.hivm.pset.b8");
  if (isa<pto::PsetB16Op>(op))
    return std::string("llvm.hivm.pset.b16");
  if (isa<pto::PsetB32Op>(op))
    return std::string("llvm.hivm.pset.b32");
  if (isa<pto::PgeB8Op>(op))
    return std::string("llvm.hivm.pge.b8");
  if (isa<pto::PgeB16Op>(op))
    return std::string("llvm.hivm.pge.b16");
  if (isa<pto::PgeB32Op>(op))
    return std::string("llvm.hivm.pge.b32");
  if (isa<pto::VldasOp>(op))
    return std::string("llvm.hivm.vldas");
  if (isa<pto::InitAlignOp>(op))
    return std::string("llvm.hivm.init.vector.align.data");
  if (auto vldus = dyn_cast<pto::VldusOp>(op)) {
    std::string vec = getElementTypeFragment(
        getElementTypeFromVectorLike(vldus.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(vldus.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vldus.v" + std::to_string(*lanes) + vec;
  }
  if (isa<pto::VstusOp>(op))
    return std::string("llvm.hivm.vstus");
  if (isa<pto::VsturOp>(op))
    return std::string("llvm.hivm.vstur");
  if (auto vlds = dyn_cast<pto::VldsOp>(op)) {
    std::string vec = getElementTypeFragment(getElementTypeFromVectorLike(vlds.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(vlds.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    std::string name = "llvm.hivm.vldsx1";
    name += ".v" + std::to_string(*lanes) + vec;
    return name;
  }
  if (auto vldsPost = dyn_cast<pto::VldsPostOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(vldsPost.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(vldsPost.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vldsx1.post.v" + std::to_string(*lanes) + vec;
  }
  if (auto vldsPost = dyn_cast<pto::VldsPostOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(vldsPost.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(vldsPost.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vldsx1.post.v" + std::to_string(*lanes) + vec;
  }
  if (auto vabs = dyn_cast<pto::VabsOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(vabs.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(vabs.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vabs.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto vexp = dyn_cast<pto::VexpOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(vexp.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(vexp.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vexp.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto vln = dyn_cast<pto::VlnOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(vln.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(vln.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vln.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto vneg = dyn_cast<pto::VnegOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(vneg.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(vneg.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vneg.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto vsqrt = dyn_cast<pto::VsqrtOp>(op)) {
    std::string vec = getElementTypeFragment(
        getElementTypeFromVectorLike(vsqrt.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(vsqrt.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vsqrt.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto vrelu = dyn_cast<pto::VreluOp>(op)) {
    std::string vec = getElementTypeFragment(
        getElementTypeFromVectorLike(vrelu.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(vrelu.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vrelu.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto vnot = dyn_cast<pto::VnotOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(vnot.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(vnot.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vnot.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto vdup = dyn_cast<pto::VdupOp>(op)) {
    Type inputType = vdup.getInput().getType();
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(vdup.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(vdup.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    if (isa<VectorType, pto::VRegType>(inputType)) {
      StringRef position = vdup.getPosition().value_or("LOWEST");
      StringRef family = position == "HIGHEST" ? "vdupm" : "vdup";
      return "llvm.hivm." + family.str() + ".v" + std::to_string(*lanes) + vec + ".z";
    }
    return "llvm.hivm.vdups.v" + std::to_string(*lanes) + vec + ".z";
  }
  if (auto vbr = dyn_cast<pto::VbrOp>(op)) {
    std::string scalar = getVbrScalarFragment(vbr.getValue().getType());
    if (scalar.empty())
      return failure();
    return "llvm.hivm.vbr." + scalar + ".v300";
  }
  if (auto binary = dyn_cast<pto::VaddOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(binary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vadd.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto binary = dyn_cast<pto::VsubOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(binary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vsub.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto binary = dyn_cast<pto::VmulOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(binary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vmul.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto binary = dyn_cast<pto::VmulsOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(binary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vmuls.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto binary = dyn_cast<pto::VaddsOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(binary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vadds.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto binary = dyn_cast<pto::VmaxsOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(binary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vmaxs.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto binary = dyn_cast<pto::VminsOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(binary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vmins.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto binary = dyn_cast<pto::VlreluOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(binary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vlrelu.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto binary = dyn_cast<pto::VshlsOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(binary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vshls.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto binary = dyn_cast<pto::VshrsOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(binary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vshrs.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto binary = dyn_cast<pto::VpreluOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(binary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vprelu.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto binary = dyn_cast<pto::VexpdiffOp>(op)) {
    std::string srcVec =
        getElementTypeFragment(getElementTypeFromVectorLike(binary.getInput().getType()));
    auto srcLanes = getElementCountFromVectorLike(binary.getInput().getType());
    std::string dstElem =
        getElementTypeFragment(getElementTypeFromVectorLike(binary.getResult().getType()));
    if (srcVec.empty() || dstElem.empty() || !srcLanes)
      return failure();
    return "llvm.hivm.vexpdif.v" + std::to_string(*srcLanes) + srcVec + dstElem;
  }
  if (auto binary = dyn_cast<pto::VdivOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(binary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vdiv.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto binary = dyn_cast<pto::VmaxOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(binary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vmax.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto binary = dyn_cast<pto::VminOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(binary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vmin.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto binary = dyn_cast<pto::VandOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(binary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vand.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto binary = dyn_cast<pto::VorOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(binary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vor.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto binary = dyn_cast<pto::VxorOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(binary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vxor.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto binary = dyn_cast<pto::VaddcOp>(op)) {
    std::string vec = getElementTypeFragment(
        getElementTypeFromVectorLike(binary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vaddc.v" + std::to_string(*lanes) + vec;
  }
  if (auto binary = dyn_cast<pto::VsubcOp>(op)) {
    std::string vec = getElementTypeFragment(
        getElementTypeFromVectorLike(binary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vsubc.v" + std::to_string(*lanes) + vec;
  }
  if (auto binary = dyn_cast<pto::VaddcsOp>(op)) {
    std::string vec = getElementTypeFragment(
        getElementTypeFromVectorLike(binary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vaddcs.v" + std::to_string(*lanes) + vec;
  }
  if (auto binary = dyn_cast<pto::VsubcsOp>(op)) {
    std::string vec = getElementTypeFragment(
        getElementTypeFromVectorLike(binary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vsubcs.v" + std::to_string(*lanes) + vec;
  }
  if (auto binary = dyn_cast<pto::VshlOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(binary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vshl.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto binary = dyn_cast<pto::VshrOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(binary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vshr.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto unary = dyn_cast<pto::VcaddOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(unary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(unary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vcadd.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto unary = dyn_cast<pto::VcmaxOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(unary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(unary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vcmax.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto unary = dyn_cast<pto::VcminOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(unary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(unary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vcmin.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto unary = dyn_cast<pto::VcgaddOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(unary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(unary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vcgadd.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto unary = dyn_cast<pto::VcgmaxOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(unary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(unary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vcgmax.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto unary = dyn_cast<pto::VcgminOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(unary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(unary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vcgmin.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto unary = dyn_cast<pto::VcpaddOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(unary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(unary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vcpadd.v" + std::to_string(*lanes) + vec + ".x";
  }
  if (auto ternary = dyn_cast<pto::VmulaOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(ternary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(ternary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vmula.v" + std::to_string(*lanes) + vec + ".m";
  }
  if (auto binary = dyn_cast<pto::VmullOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(binary.getLow().getType()));
    auto lanes = getElementCountFromVectorLike(binary.getLow().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vmull.v" + std::to_string(*lanes) + vec;
  }
  if (auto ternary = dyn_cast<pto::VaxpyOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(ternary.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(ternary.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vaxpy.v" + std::to_string(*lanes) + vec + ".m";
  }
  if (auto vci = dyn_cast<pto::VciOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(vci.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(vci.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    if (vec == "f16")
      return "llvm.hivm.vci.v" + std::to_string(*lanes) + vec + ".f16";
    if (vec == "f32")
      return "llvm.hivm.vci.v" + std::to_string(*lanes) + vec + ".f32";
    return "llvm.hivm.vci.v" + std::to_string(*lanes) + vec;
  }
  if (auto vbitsort = dyn_cast<pto::VbitsortOp>(op)) {
    Type sourceElemType = getElementTypeFromABIValue(vbitsort.getSource());
    if (!sourceElemType)
      return failure();
    if (sourceElemType.isF16())
      return std::string("llvm.hivm.VBS32.V300.f16");
    if (sourceElemType.isF32())
      return std::string("llvm.hivm.VBS32.V300.f32");
    return failure();
  }
  if (auto vtrc = dyn_cast<pto::VtrcOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(vtrc.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(vtrc.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vtrc." + vec + ".x";
  }
  if (auto vcvt = dyn_cast<pto::VcvtOp>(op)) {
    Type inputElemType = getElementTypeFromVectorLike(vcvt.getInput().getType());
    Type resultElemType = getElementTypeFromVectorLike(vcvt.getResult().getType());
    if (!inputElemType || !resultElemType)
      return failure();
    auto contract = lookupVcvtContract(classifyVcvtElemType(inputElemType),
                                       classifyVcvtElemType(resultElemType));
    if (contract)
      return std::string(contract->intrinsic);
    return failure();
  }
  if (isa<pto::VstarOp>(op))
    return std::string("llvm.hivm.vstar");
  if (isa<pto::VstasOp>(op))
    return std::string("llvm.hivm.vstas");
  if (auto vsqz = dyn_cast<pto::VsqzOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(vsqz.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(vsqz.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vsqz.v" + std::to_string(*lanes) + vec + ".x.v300";
  }
  if (auto vusqz = dyn_cast<pto::VusqzOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(vusqz.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(vusqz.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vusqz.v" + std::to_string(*lanes) + vec + ".m";
  }
  if (auto unpack = dyn_cast<pto::VsunpackOp>(op)) {
    Type inputElemType = getElementTypeFromVectorLike(unpack.getSrc().getType());
    Type resultElemType = getElementTypeFromVectorLike(unpack.getResult().getType());
    std::string input = getElementTypeFragment(inputElemType);
    std::string result = getElementTypeFragment(resultElemType);
    if (input.empty() || result.empty())
      return failure();
    return "llvm.hivm.vsunpack." + input + "2" + result;
  }
  if (auto unpack = dyn_cast<pto::VzunpackOp>(op)) {
    Type inputElemType = getElementTypeFromVectorLike(unpack.getSrc().getType());
    Type resultElemType = getElementTypeFromVectorLike(unpack.getResult().getType());
    std::string input = getElementTypeFragment(inputElemType);
    std::string result = getElementTypeFragment(resultElemType);
    if (input.empty() || result.empty())
      return failure();
    return "llvm.hivm.vzunpack." + input + "2" + result;
  }
  if (auto pack = dyn_cast<pto::VpackOp>(op)) {
    Type inputElemType = getElementTypeFromVectorLike(pack.getSrc().getType());
    Type resultElemType = getElementTypeFromVectorLike(pack.getResult().getType());
    std::string input = getElementTypeFragment(inputElemType);
    std::string result = getElementTypeFragment(resultElemType);
    auto part = parseHiLoPartImmediate(pack.getPart());
    if (input.empty() || result.empty() || !part)
      return failure();
    return "llvm.hivm.vpack." + input + "2" + result + ".x";
  }
  if (auto interleave = dyn_cast<pto::VintlvOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(interleave.getLow().getType()));
    auto lanes = getElementCountFromVectorLike(interleave.getLow().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vintlv.v" + std::to_string(*lanes) + vec;
  }
  if (auto deinterleave = dyn_cast<pto::VdintlvOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(deinterleave.getLow().getType()));
    auto lanes = getElementCountFromVectorLike(deinterleave.getLow().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vdintlv.v" + std::to_string(*lanes) + vec;
  }
  if (isa<pto::VsldbOp>(op))
    return std::string("llvm.hivm.vsldb");
  if (isa<pto::VsstbOp>(op))
    return std::string("llvm.hivm.vsstb");
  if (auto vldsx2 = dyn_cast<pto::Vldsx2Op>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(vldsx2.getLow().getType()));
    auto lanes = getElementCountFromVectorLike(vldsx2.getLow().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vldsx2.v" + std::to_string(*lanes) + vec;
  }
  if (auto vstsx2 = dyn_cast<pto::Vstsx2Op>(op)) {
    std::string vec = getElementTypeFragment(
        getElementTypeFromVectorLike(vstsx2.getLow().getType()));
    auto lanes = getElementCountFromVectorLike(vstsx2.getLow().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vstsx2.v" + std::to_string(*lanes) + vec;
  }
  if (auto vsts = dyn_cast<pto::VstsOp>(op)) {
    std::string vec = getElementTypeFragment(getElementTypeFromVectorLike(vsts.getValue().getType()));
    auto lanes = getElementCountFromVectorLike(vsts.getValue().getType());
    if (vec.empty() || !lanes)
      return failure();
    std::string name = "llvm.hivm.vstsx1";
    name += ".v" + std::to_string(*lanes) + vec;
    return name;
  }
  if (auto vstsPost = dyn_cast<pto::VstsPostOp>(op)) {
    std::string vec = getElementTypeFragment(
        getElementTypeFromVectorLike(vstsPost.getValue().getType()));
    auto lanes = getElementCountFromVectorLike(vstsPost.getValue().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vstsx1.post.v" + std::to_string(*lanes) + vec;
  }
  if (auto vstsPost = dyn_cast<pto::VstsPostOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(vstsPost.getValue().getType()));
    auto lanes = getElementCountFromVectorLike(vstsPost.getValue().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vstsx1.post.v" + std::to_string(*lanes) + vec;
  }
  if (auto vcmp = dyn_cast<pto::VcmpOp>(op)) {
    std::string elem = getElementTypeFragment(getElementTypeFromVectorLike(vcmp.getSrc0().getType()));
    if (elem.empty())
      return failure();
    return "llvm.hivm.vcmp." + vcmp.getCmpMode().str() + "." + elem + ".z";
  }
  if (auto vcmps = dyn_cast<pto::VcmpsOp>(op)) {
    std::string elem = getElementTypeFragment(getElementTypeFromVectorLike(vcmps.getSrc().getType()));
    if (elem.empty())
      return failure();
    return "llvm.hivm.vcmps." + vcmps.getCmpMode().str() + "." + elem + ".z";
  }
  if (auto vsel = dyn_cast<pto::VselOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(vsel.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(vsel.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vsel.v" + std::to_string(*lanes) + vec;
  }
  if (auto vselr = dyn_cast<pto::VselrOp>(op)) {
    Type elemType = getElementTypeFromVectorLike(vselr.getResult().getType());
    auto lanes = getElementCountFromVectorLike(vselr.getResult().getType());
    if (!elemType || !lanes)
      return failure();
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(vselr.getResult().getType()));
    if (auto floatType = dyn_cast<FloatType>(elemType); floatType && floatType.isF32())
      vec = "u32";
    if (vec.empty())
      return failure();
    return "llvm.hivm.vselr.v" + std::to_string(*lanes) + vec;
  }
  if (isa<pto::PpackOp>(op))
    return std::string("llvm.hivm.ppack.z");
  if (isa<pto::PunpackOp>(op))
    return std::string("llvm.hivm.punpack");
  if (isa<pto::PnotOp>(op))
    return std::string("llvm.hivm.pnot.z");
  if (isa<pto::PselOp>(op))
    return std::string("llvm.hivm.psel");
  if (isa<pto::PandOp>(op))
    return std::string("llvm.hivm.pand.z");
  if (isa<pto::PorOp>(op))
    return std::string("llvm.hivm.por.z");
  if (isa<pto::PxorOp>(op))
    return std::string("llvm.hivm.pxor.z");
  if (isa<pto::PdintlvB8Op>(op))
    return std::string("llvm.hivm.pdintlv.b8");
  if (isa<pto::PdintlvB16Op>(op))
    return std::string("llvm.hivm.pdintlv.b16");
  if (isa<pto::PdintlvB32Op>(op))
    return std::string("llvm.hivm.pdintlv.b32");
  if (isa<pto::PintlvB8Op>(op))
    return std::string("llvm.hivm.pintlv.b8");
  if (isa<pto::PintlvB16Op>(op))
    return std::string("llvm.hivm.pintlv.b16");
  if (isa<pto::PintlvB32Op>(op))
    return std::string("llvm.hivm.pintlv.b32");
  if (isa<pto::PldsOp>(op))
    return std::string("llvm.hivm.plds.b8");
  if (isa<pto::PldiOp>(op))
    return std::string("llvm.hivm.pldi.b8");
  if (isa<pto::PstsOp>(op))
    return std::string("llvm.hivm.psts.b8");
  if (op->getName().getStringRef() == "pto.pstu") {
    Type maskOperandType = op->getOperand(1).getType();
    if (auto maskType = dyn_cast<pto::MaskType>(maskOperandType)) {
      if (maskType.isB16())
        return std::string("llvm.hivm.pstu.b16");
      if (maskType.isB32())
        return std::string("llvm.hivm.pstu.b32");
    }
    if (Type baseElementType = getElementTypeFromABIValue(op->getOperand(2))) {
      if (auto intType = dyn_cast<IntegerType>(baseElementType)) {
        if (intType.getWidth() == 16)
          return std::string("llvm.hivm.pstu.b16");
        if (intType.getWidth() == 32)
          return std::string("llvm.hivm.pstu.b32");
      }
    }
    // Current repo coverage only exercises the installed `b32` surface. Keep
    // this fallback narrow to unblock those cases; `b16` still needs an
    // end-to-end testcase path before we can claim the generic surface works.
    return std::string("llvm.hivm.pstu.b32");
  }
  if (auto pstu = dyn_cast<pto::PstuOp>(op)) {
    if (auto maskType = dyn_cast<pto::MaskType>(pstu.getValue().getType())) {
      if (maskType.isB16())
        return std::string("llvm.hivm.pstu.b16");
      if (maskType.isB32())
        return std::string("llvm.hivm.pstu.b32");
    }
    if (Type baseElementType = getElementTypeFromABIValue(pstu.getBase())) {
      if (auto intType = dyn_cast<IntegerType>(baseElementType)) {
        if (intType.getWidth() == 16)
          return std::string("llvm.hivm.pstu.b16");
        if (intType.getWidth() == 32)
          return std::string("llvm.hivm.pstu.b32");
      }
    }
    return failure();
  }
  if (isa<pto::PstiOp>(op))
    return std::string("llvm.hivm.psti.b8");
  if (auto gather = dyn_cast<pto::Vgather2Op>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(gather.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(gather.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vgather2.v300.v" + std::to_string(*lanes) + vec;
  }
  if (auto gather = dyn_cast<pto::Vgather2BcOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(gather.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(gather.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vgather2.bc.v" + std::to_string(*lanes) + vec;
  }
  if (auto gather = dyn_cast<pto::VgatherbOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(gather.getResult().getType()));
    auto lanes = getElementCountFromVectorLike(gather.getResult().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vgatherb.v310.v" + std::to_string(*lanes) + vec;
  }
  if (auto scatter = dyn_cast<pto::VscatterOp>(op)) {
    std::string vec =
        getElementTypeFragment(getElementTypeFromVectorLike(scatter.getValue().getType()));
    auto lanes = getElementCountFromVectorLike(scatter.getValue().getType());
    if (vec.empty() || !lanes)
      return failure();
    return "llvm.hivm.vscatter.v" + std::to_string(*lanes) + vec + ".v300";
  }
  return failure();
}

static LogicalResult
guardNoMemRefIntrinsicArgs(Operation *op, StringRef calleeName,
                           ValueRange callArgs, llvm::raw_ostream &diagOS) {
  if (calleeName != "llvm.hivm.vldsx1" && calleeName != "llvm.hivm.vstsx1")
    return success();

  for (auto [idx, arg] : llvm::enumerate(callArgs)) {
    Type argType = arg.getType();
    if (!isa<MemRefType, UnrankedMemRefType>(argType))
      continue;
    diagOS << "VPTO LLVM emission failed: intrinsic ABI guard rejected memref "
              "argument #"
           << idx << " for " << calleeName << " from "
           << op->getName().getStringRef() << " (" << argType << ")\n";
    return failure();
  }
  return success();
}

static LogicalResult rewriteVPTOOp(Operation *op, ModuleOp module,
                                   llvm::raw_ostream &diagOS) {
  IRRewriter builder(op->getContext());
  builder.setInsertionPoint(op);
  Location loc = op->getLoc();

  if (auto vbr = dyn_cast<pto::VbrOp>(op)) {
    auto calleeName = getConfirmedCallee(op);
    if (failed(calleeName)) {
      diagOS << "VPTO LLVM emission failed: unsupported op "
             << op->getName().getStringRef() << "\n";
      return failure();
    }

    Type resultType = convertVPTOType(vbr.getResult().getType(), builder);
    Type scalarType = vbr.getValue().getType();
    if (!resultType || !scalarType) {
      diagOS << "VPTO LLVM emission failed: could not materialize vbr types\n";
      return failure();
    }

    auto funcType = builder.getFunctionType({scalarType}, {resultType});
    auto callee = getOrCreateExternalFunc(module, *calleeName, funcType);
    auto call =
        builder.create<func::CallOp>(loc, callee, ValueRange{vbr.getValue()});
    builder.replaceOp(op, call.getResults());
    return success();
  }

  if (isa<pto::GetBlockIdxOp, pto::GetSubBlockIdxOp, pto::GetBlockNumOp,
          pto::GetSubBlockNumOp>(op)) {
    SmallVector<Type> argTypes;
    auto funcType = builder.getFunctionType(argTypes, op->getResultTypes());
    auto callee = getOrCreateExternalFunc(module, *getConfirmedCallee(op), funcType);
    auto call = builder.create<func::CallOp>(loc, callee, ValueRange{});
    builder.replaceOp(op, call.getResults());
    return success();
  }

  auto calleeName = getConfirmedCallee(op);
  if (failed(calleeName)) {
    diagOS << "VPTO LLVM emission failed: unsupported op "
           << op->getName().getStringRef() << "\n";
    return failure();
  }

  SmallVector<Type> surfaceResultTypes(op->getResultTypes().begin(),
                                       op->getResultTypes().end());
  SmallVector<Type> loweredResultTypes;
  loweredResultTypes.reserve(surfaceResultTypes.size());
  for (Type type : surfaceResultTypes)
    loweredResultTypes.push_back(convertVPTOType(type, builder));
  SmallVector<Type> intrinsicResultTypes(loweredResultTypes.begin(),
                                         loweredResultTypes.end());
  if (auto vldus = dyn_cast<pto::VldusOp>(op)) {
    Type sourceType = convertVPTOType(vldus.getSource().getType(), builder);
    if (!sourceType) {
      diagOS << "VPTO LLVM emission failed: could not materialize vldus source type\n";
      return failure();
    }
    intrinsicResultTypes.push_back(sourceType);
  }

  SmallVector<Value> callArgs;

  if (isa<pto::SetLoop2StrideOutToUbOp, pto::SetLoop1StrideOutToUbOp,
          pto::SetLoop2StrideUbToOutOp, pto::SetLoop1StrideUbToOutOp>(op)) {
    auto packed = packLoopPair(op, op->getOperand(0), op->getOperand(1));
    if (failed(packed))
      return failure();
    callArgs.push_back(*packed);
  } else if (isa<pto::SetLoopSizeOutToUbOp, pto::SetLoopSizeUbToOutOp>(op)) {
    auto packed = packLoopSize(op, op->getOperand(0), op->getOperand(1));
    if (failed(packed))
      return failure();
    callArgs.push_back(*packed);
  } else if (auto copy = dyn_cast<pto::CopyGmToUbufOp>(op)) {
    auto config0 = packCopyGmToUbConfig0(op, copy, op->getOperands());
    auto config1 = packCopyGmToUbConfig1(op, op->getOperands());
    auto destination = requirePointerABIAddress(op, copy.getDestination(), diagOS);
    auto source = requirePointerABIAddress(op, copy.getSource(), diagOS);
    if (failed(config0) || failed(config1) || failed(destination) ||
        failed(source))
      return failure();
    callArgs.push_back(*destination);
    callArgs.push_back(*source);
    callArgs.push_back(*config0);
    callArgs.push_back(*config1);
  } else if (auto copy = dyn_cast<pto::CopyUbufToGmOp>(op)) {
    auto config0 = packCopyUbToGmConfig0(op, op->getOperands());
    auto config1 = packCopyUbToGmConfig1(op, op->getOperands());
    auto destination = requirePointerABIAddress(op, copy.getDestination(), diagOS);
    auto source = requirePointerABIAddress(op, copy.getSource(), diagOS);
    if (failed(config0) || failed(config1) || failed(destination) ||
        failed(source))
      return failure();
    callArgs.push_back(*destination);
    callArgs.push_back(*source);
    callArgs.push_back(*config0);
    callArgs.push_back(*config1);
  } else if (auto setFlag = dyn_cast<pto::SetFlagOp>(op)) {
    auto src = parsePipeImmediate(stringifyPIPE(setFlag.getSrcPipe().getPipe()));
    auto dst = parsePipeImmediate(stringifyPIPE(setFlag.getDstPipe().getPipe()));
    auto event = parseEventImmediate(stringifyEVENT(setFlag.getEventId().getEvent()));
    if (!src || !dst || !event)
      return failure();
    callArgs.push_back(getI64Constant(builder, loc, *src));
    callArgs.push_back(getI64Constant(builder, loc, *dst));
    callArgs.push_back(getI64Constant(builder, loc, *event));
  } else if (auto waitFlag = dyn_cast<pto::WaitFlagOp>(op)) {
    auto src =
        parsePipeImmediate(stringifyPIPE(waitFlag.getSrcPipe().getPipe()));
    auto dst =
        parsePipeImmediate(stringifyPIPE(waitFlag.getDstPipe().getPipe()));
    auto event =
        parseEventImmediate(stringifyEVENT(waitFlag.getEventId().getEvent()));
    if (!src || !dst || !event)
      return failure();
    callArgs.push_back(getI64Constant(builder, loc, *src));
    callArgs.push_back(getI64Constant(builder, loc, *dst));
    callArgs.push_back(getI64Constant(builder, loc, *event));
  } else if (auto barrier = dyn_cast<pto::BarrierOp>(op)) {
    auto pipe = parsePipeImmediate(stringifyPIPE(barrier.getPipe().getPipe()));
    if (!pipe)
      return failure();
    callArgs.push_back(getI64Constant(builder, loc, *pipe));
  } else if (auto sprclr = dyn_cast<pto::SprclrOp>(op)) {
    auto spr = parseSprImmediate(sprclr.getSpr());
    if (!spr) {
      diagOS << "VPTO LLVM emission failed: unsupported sprclr target "
             << sprclr.getSpr() << "\n";
      return failure();
    }
    callArgs.push_back(getI16Constant(builder, loc, *spr));
  } else if (isa<pto::PltB8Op, pto::PltB32Op, pto::PltB16Op>(op)) {
    Value laneCount = castIntegerLikeTo(op, op->getOperand(0), builder.getI32Type());
    if (!laneCount)
      return failure();
    callArgs.push_back(laneCount);
  } else if (auto pset = dyn_cast<pto::PsetB8Op>(op)) {
    auto pattern = parsePredicatePatternImmediate(pset.getPattern());
    if (!pattern) {
      diagOS << "VPTO LLVM emission failed: unsupported pset_b8 pattern "
             << pset.getPattern() << "\n";
      return failure();
    }
    callArgs.push_back(getI32Constant(builder, loc, *pattern));
  } else if (auto pset = dyn_cast<pto::PsetB16Op>(op)) {
    auto pattern = parsePredicatePatternImmediate(pset.getPattern());
    if (!pattern) {
      diagOS << "VPTO LLVM emission failed: unsupported pset_b16 pattern "
             << pset.getPattern() << "\n";
      return failure();
    }
    callArgs.push_back(getI32Constant(builder, loc, *pattern));
  } else if (auto pset = dyn_cast<pto::PsetB32Op>(op)) {
    auto pattern = parsePredicatePatternImmediate(pset.getPattern());
    if (!pattern) {
      diagOS << "VPTO LLVM emission failed: unsupported pset_b32 pattern "
             << pset.getPattern() << "\n";
      return failure();
    }
    callArgs.push_back(getI32Constant(builder, loc, *pattern));
  } else if (auto pge = dyn_cast<pto::PgeB8Op>(op)) {
    auto pattern = parsePredicatePatternImmediate(pge.getPattern());
    if (!pattern) {
      diagOS << "VPTO LLVM emission failed: unsupported pge_b8 pattern "
             << pge.getPattern() << "\n";
      return failure();
    }
    callArgs.push_back(getI32Constant(builder, loc, *pattern));
    callArgs.push_back(getI32Constant(builder, loc, 0));
  } else if (auto pge = dyn_cast<pto::PgeB16Op>(op)) {
    auto pattern = parsePredicatePatternImmediate(pge.getPattern());
    if (!pattern) {
      diagOS << "VPTO LLVM emission failed: unsupported pge_b16 pattern "
             << pge.getPattern() << "\n";
      return failure();
    }
    callArgs.push_back(getI32Constant(builder, loc, *pattern));
    callArgs.push_back(getI32Constant(builder, loc, 0));
  } else if (auto pge = dyn_cast<pto::PgeB32Op>(op)) {
    auto pattern = parsePredicatePatternImmediate(pge.getPattern());
    if (!pattern) {
      diagOS << "VPTO LLVM emission failed: unsupported pge_b32 pattern "
             << pge.getPattern() << "\n";
      return failure();
    }
    callArgs.push_back(getI32Constant(builder, loc, *pattern));
    callArgs.push_back(getI32Constant(builder, loc, 0));
  } else if (isa<pto::InitAlignOp>(op)) {
    // llvm.hivm.init.vector.align.data() has no operands.
  } else if (auto vldas = dyn_cast<pto::VldasOp>(op)) {
    auto source = requirePointerABIAddress(op, vldas.getSource(), diagOS);
    if (failed(source))
      return failure();
    callArgs.push_back(*source);
  } else if (auto vldus = dyn_cast<pto::VldusOp>(op)) {
    auto source = requirePointerABIAddress(op, vldus.getSource(), diagOS);
    if (failed(source))
      return failure();
    callArgs.push_back(*source);
    callArgs.push_back(vldus.getAlign());
  } else if (auto vstus = dyn_cast<pto::VstusOp>(op)) {
    Type elementType = getElementTypeFromVectorLike(vstus.getValue().getType());
    auto basePtr = requirePointerABIAddress(op, vstus.getBase(), diagOS);
    auto alignValue = materializeAlignABIValue(op, vstus.getAlignIn(), diagOS);
    if (!elementType || failed(basePtr))
      return failure();
    auto offsetBytes = convertElementOffsetToBytes(op, vstus.getOffset(), elementType);
    if (failed(offsetBytes) || failed(alignValue))
      return failure();
    callArgs.push_back(vstus.getValue());
    callArgs.push_back(*basePtr);
    callArgs.push_back(*offsetBytes);
    callArgs.push_back(*alignValue);
  } else if (auto vstur = dyn_cast<pto::VsturOp>(op)) {
    auto basePtr = requirePointerABIAddress(op, vstur.getBase(), diagOS);
    auto postMode = parsePostModeImmediate(vstur.getMode());
    auto alignValue = materializeAlignABIValue(op, vstur.getAlignIn(), diagOS);
    if (failed(basePtr) || !postMode) {
      if (!postMode)
        diagOS << "VPTO LLVM emission failed: unsupported vstur mode "
               << vstur.getMode() << "\n";
      return failure();
    }
    if (failed(alignValue))
      return failure();
    callArgs.push_back(vstur.getValue());
    callArgs.push_back(*basePtr);
    callArgs.push_back(*alignValue);
    callArgs.push_back(getI32Constant(builder, loc, *postMode));
    callArgs.push_back(getI32Constant(builder, loc, 0));
  } else if (auto vlds = dyn_cast<pto::VldsOp>(op)) {
    Type elementType = getElementTypeFromVectorLike(vlds.getResult().getType());
    auto offsetBytes = convertElementOffsetToBytes(
        op, op->getOperand(1), elementType);
    auto basePtr = requirePointerABIAddress(op, op->getOperand(0), diagOS);
    auto dist =
        parseLoadDistImmediate(vlds.getDist().value_or("NORM"), elementType);
    if (!elementType || failed(offsetBytes) || failed(basePtr) || !dist) {
      if (elementType && succeeded(basePtr) && !dist)
        diagOS << "VPTO LLVM emission failed: unsupported vlds dist immediate\n";
      return failure();
    }
    callArgs.push_back(*basePtr);
    callArgs.push_back(*offsetBytes);
    callArgs.push_back(getI32Constant(builder, loc, *dist));
    callArgs.push_back(getI32Constant(builder, loc, 0));
  } else if (auto vldsPost = dyn_cast<pto::VldsPostOp>(op)) {
    Type elementType = getElementTypeFromVectorLike(vldsPost.getResult().getType());
    auto offsetBytes = convertElementOffsetToBytes(
        op, vldsPost.getOffset(), elementType);
    auto basePtr = requirePointerABIAddress(op, vldsPost.getSource(), diagOS);
    auto dist =
        parseLoadDistImmediate(vldsPost.getDist().value_or("NORM"), elementType);
    if (!elementType || failed(offsetBytes) || failed(basePtr) || !dist)
      return failure();
    callArgs.push_back(*basePtr);
    callArgs.push_back(*offsetBytes);
    callArgs.push_back(getI32Constant(builder, loc, *dist));
    callArgs.push_back(getI32Constant(builder, loc, 1));
  } else if (auto vabs = dyn_cast<pto::VabsOp>(op)) {
    Value input = op->getOperand(0);
    Value mask = op->getOperand(1);
    Type vecType = loweredResultTypes.front();
    Type maskType = convertVPTOType(mask.getType(), builder);
    if (input.getType() != vecType || mask.getType() != maskType) {
      diagOS << "VPTO LLVM emission failed: unexpected vabs operand types\n";
      return failure();
    }
    callArgs.push_back(input);
    callArgs.push_back(mask);
  } else if (auto unary = dyn_cast<pto::VexpOp>(op)) {
    Value input = unary.getInput();
    Value mask = unary.getMask();
    Type vecType = loweredResultTypes.front();
    Type maskType = convertVPTOType(mask.getType(), builder);
    if (input.getType() != vecType || mask.getType() != maskType) {
      diagOS << "VPTO LLVM emission failed: unexpected "
             << op->getName().getStringRef() << " operand types\n";
      return failure();
    }
    callArgs.push_back(input);
    callArgs.push_back(mask);
  } else if (auto unary = dyn_cast<pto::VlnOp>(op)) {
    Value input = unary.getInput();
    Value mask = unary.getMask();
    Type vecType = loweredResultTypes.front();
    Type maskType = convertVPTOType(mask.getType(), builder);
    if (input.getType() != vecType || mask.getType() != maskType) {
      diagOS << "VPTO LLVM emission failed: unexpected "
             << op->getName().getStringRef() << " operand types\n";
      return failure();
    }
    callArgs.push_back(input);
    callArgs.push_back(mask);
  } else if (auto unary = dyn_cast<pto::VnegOp>(op)) {
    Value input = unary.getInput();
    Value mask = unary.getMask();
    Type vecType = loweredResultTypes.front();
    Type maskType = convertVPTOType(mask.getType(), builder);
    if (input.getType() != vecType || mask.getType() != maskType) {
      diagOS << "VPTO LLVM emission failed: unexpected "
             << op->getName().getStringRef() << " operand types\n";
      return failure();
    }
    callArgs.push_back(input);
    callArgs.push_back(mask);
  } else if (auto unary = dyn_cast<pto::VsqrtOp>(op)) {
    Value input = unary.getInput();
    Value mask = unary.getMask();
    Type vecType = loweredResultTypes.front();
    Type maskType = convertVPTOType(mask.getType(), builder);
    if (input.getType() != vecType || mask.getType() != maskType) {
      diagOS << "VPTO LLVM emission failed: unexpected "
             << op->getName().getStringRef() << " operand types\n";
      return failure();
    }
    callArgs.push_back(input);
    callArgs.push_back(mask);
  } else if (auto unary = dyn_cast<pto::VreluOp>(op)) {
    Value input = unary.getInput();
    Value mask = unary.getMask();
    Type vecType = loweredResultTypes.front();
    Type maskType = convertVPTOType(mask.getType(), builder);
    if (input.getType() != vecType || mask.getType() != maskType) {
      diagOS << "VPTO LLVM emission failed: unexpected "
             << op->getName().getStringRef() << " operand types\n";
      return failure();
    }
    callArgs.push_back(input);
    callArgs.push_back(mask);
  } else if (auto unary = dyn_cast<pto::VnotOp>(op)) {
    Value input = unary.getInput();
    Value mask = unary.getMask();
    Type vecType = loweredResultTypes.front();
    Type maskType = convertVPTOType(mask.getType(), builder);
    if (input.getType() != vecType || mask.getType() != maskType) {
      diagOS << "VPTO LLVM emission failed: unexpected "
             << op->getName().getStringRef() << " operand types\n";
      return failure();
    }
    callArgs.push_back(input);
    callArgs.push_back(mask);
  } else if (auto vdup = dyn_cast<pto::VdupOp>(op)) {
    Type scalarType = getElementTypeFromVectorLike(vdup.getResult().getType());
    bool vectorInput = isa<VectorType, pto::VRegType>(vdup.getInput().getType());
    if (!vectorInput && (!scalarType || vdup.getInput().getType() != scalarType)) {
      diagOS << "VPTO LLVM emission failed: unexpected vdup operand types\n";
      return failure();
    }
    if (vectorInput && vdup.getInput().getType() != loweredResultTypes.front()) {
      diagOS << "VPTO LLVM emission failed: vector-input vdup requires matching result type\n";
      return failure();
    }
    if (vectorInput) {
      callArgs.push_back(vdup.getInput());
    } else {
      FailureOr<Value> normalizedScalar = normalizeVdupScalarOperand(builder, loc, vdup);
      if (failed(normalizedScalar))
        return failure();
      callArgs.push_back(*normalizedScalar);
    }
    callArgs.push_back(vdup.getMask());
    callArgs.push_back(getI32Constant(builder, loc, 1));
  } else if (isa<pto::VaddOp, pto::VsubOp,
                 pto::VmulOp, pto::VdivOp, pto::VmaxOp, pto::VminOp,
                 pto::VandOp, pto::VorOp, pto::VxorOp, pto::VshlOp,
                 pto::VshrOp, pto::VshlsOp, pto::VshrsOp>(op)) {
    callArgs.append(op->operand_begin(), op->operand_end());
  } else if (isa<pto::VaddcOp, pto::VsubcOp>(op)) {
    callArgs.push_back(op->getOperand(0));
    callArgs.push_back(op->getOperand(1));
    callArgs.push_back(op->getOperand(2));
  } else if (isa<pto::VaddcsOp, pto::VsubcsOp>(op)) {
    callArgs.push_back(op->getOperand(0));
    callArgs.push_back(op->getOperand(1));
    callArgs.push_back(op->getOperand(2));
    callArgs.push_back(op->getOperand(3));
  } else if (auto vmula = dyn_cast<pto::VmulaOp>(op)) {
    callArgs.push_back(vmula.getAcc());
    callArgs.push_back(vmula.getLhs());
    callArgs.push_back(vmula.getRhs());
    callArgs.push_back(vmula.getMask());
  } else if (auto vmull = dyn_cast<pto::VmullOp>(op)) {
    callArgs.push_back(vmull.getLhs());
    callArgs.push_back(vmull.getRhs());
    callArgs.push_back(vmull.getMask());
  } else if (auto vaxpy = dyn_cast<pto::VaxpyOp>(op)) {
    auto laneCount = getElementCountFromVectorLike(vaxpy.getResult().getType());
    if (!laneCount) {
      diagOS << "VPTO LLVM emission failed: could not determine lane count for "
             << op->getName().getStringRef() << "\n";
      return failure();
    }
    Type elemType = getElementTypeFromVectorLike(vaxpy.getResult().getType());
    Value mask;
    if (elemType.isF32()) {
      auto fullMask = buildPltB32Mask(builder, module, loc, *laneCount, diagOS);
      if (failed(fullMask))
        return failure();
      mask = *fullMask;
    } else {
      auto fullMask = buildPltB16Mask(builder, module, loc, *laneCount, diagOS);
      if (failed(fullMask))
        return failure();
      mask = *fullMask;
    }
    // Installed wrapper surface is dst = alpha * src0 + dst. VPTO models this
    // as a pure op returning the updated addend vector.
    callArgs.push_back(vaxpy.getSrc1());
    callArgs.push_back(vaxpy.getSrc0());
    callArgs.push_back(vaxpy.getAlpha());
    callArgs.push_back(mask);
  } else if (auto vci = dyn_cast<pto::VciOp>(op)) {
    auto orderAttr = op->getAttrOfType<StringAttr>("order");
    auto order = parseOrderImmediate(orderAttr ? orderAttr.getValue() : StringRef("ASC"));
    if (!order) {
      diagOS << "VPTO LLVM emission failed: unsupported vci order ";
      if (orderAttr)
        diagOS << orderAttr.getValue();
      else
        diagOS << "<null>";
      diagOS << "\n";
      return failure();
    }
    callArgs.push_back(vci.getIndex());
    callArgs.push_back(getI32Constant(builder, loc, *order));
  } else if (isa<pto::VpreluOp>(op)) {
    callArgs.append(op->operand_begin(), op->operand_end());
    auto laneCount = getElementCountFromVectorLike(op->getResult(0).getType());
    if (!laneCount) {
      diagOS << "VPTO LLVM emission failed: could not determine lane count for "
             << op->getName().getStringRef() << "\n";
      return failure();
    }
    Value mask;
    if (getElementTypeFromVectorLike(op->getResult(0).getType()).isF32() ||
        getElementTypeFromVectorLike(op->getResult(0).getType()).isInteger(32)) {
      auto fullMask = buildPltB32Mask(builder, module, loc, *laneCount, diagOS);
      if (failed(fullMask))
        return failure();
      mask = *fullMask;
    } else {
      auto fullMask = buildPltB16Mask(builder, module, loc, *laneCount, diagOS);
      if (failed(fullMask))
        return failure();
      mask = *fullMask;
    }
    callArgs.push_back(mask);
  } else if (auto vexpdiff = dyn_cast<pto::VexpdiffOp>(op)) {
    callArgs.push_back(vexpdiff.getInput());
    callArgs.push_back(vexpdiff.getMax());
    auto srcLaneCount = getElementCountFromVectorLike(vexpdiff.getInput().getType());
    if (!srcLaneCount) {
      diagOS << "VPTO LLVM emission failed: could not determine lane count for "
             << op->getName().getStringRef() << "\n";
      return failure();
    }
    Value mask;
    Type inputElemType = getElementTypeFromVectorLike(vexpdiff.getInput().getType());
    if (inputElemType.isF32() || inputElemType.isInteger(32)) {
      auto fullMask = buildPltB32Mask(builder, module, loc, *srcLaneCount, diagOS);
      if (failed(fullMask))
        return failure();
      mask = *fullMask;
    } else {
      auto fullMask = buildPltB16Mask(builder, module, loc, *srcLaneCount, diagOS);
      if (failed(fullMask))
        return failure();
      mask = *fullMask;
    }
    auto part = parsePartImmediate(vexpdiff.getPart());
    if (!part) {
      diagOS << "VPTO LLVM emission failed: unsupported vexpdiff part ";
      diagOS << vexpdiff.getPart();
      diagOS << "\n";
      return failure();
    }
    callArgs.push_back(mask);
    callArgs.push_back(getI32Constant(builder, loc, *part));
  } else if (isa<pto::VmulsOp, pto::VaddsOp,
                 pto::VmaxsOp, pto::VminsOp,
                 pto::VlreluOp>(op)) {
    callArgs.append(op->operand_begin(), op->operand_end());
  } else if (isa<pto::VcaddOp, pto::VcmaxOp, pto::VcminOp,
                 pto::VcgaddOp, pto::VcgmaxOp, pto::VcgminOp,
                 pto::VcpaddOp>(op)) {
    callArgs.push_back(op->getOperand(0));
    callArgs.push_back(op->getOperand(1));
  } else if (auto vtrc = dyn_cast<pto::VtrcOp>(op)) {
    auto roundMode = parseRoundModeImmediate(vtrc.getRoundMode());
    if (!roundMode) {
      diagOS << "VPTO LLVM emission failed: unsupported round mode "
             << vtrc.getRoundMode() << "\n";
      return failure();
    }
    auto laneCount = getElementCountFromVectorLike(vtrc.getResult().getType());
    if (!laneCount) {
      diagOS << "VPTO LLVM emission failed: could not determine lane count for "
             << op->getName().getStringRef() << "\n";
      return failure();
    }
    auto mask = buildPltB32Mask(builder, module, loc, *laneCount, diagOS);
    if (failed(mask))
      return failure();
    callArgs.push_back(vtrc.getInput());
    callArgs.push_back(getI32Constant(builder, loc, *roundMode));
    callArgs.push_back(*mask);
  } else if (auto vcvt = dyn_cast<pto::VcvtOp>(op)) {
    Type inputElemType = getElementTypeFromVectorLike(vcvt.getInput().getType());
    Type resultElemType = getElementTypeFromVectorLike(vcvt.getResult().getType());
    auto inputLanes = getElementCountFromVectorLike(vcvt.getInput().getType());
    if (!inputElemType || !resultElemType || !inputLanes) {
      diagOS << "VPTO LLVM emission failed: could not determine vcvt type shape\n";
      return failure();
    }

    auto contract = lookupVcvtContract(classifyVcvtElemType(inputElemType),
                                       classifyVcvtElemType(resultElemType));
    if (!contract) {
      diagOS << "VPTO LLVM emission failed: unsupported vcvt type pair "
             << vcvt.getInput().getType() << " -> " << vcvt.getResult().getType()
             << "\n";
      return failure();
    }

    callArgs.push_back(vcvt.getInput());
    FailureOr<Value> mask = failure();
    switch (contract->maskBitWidth) {
    case 8:
      mask = buildPltB8Mask(builder, module, loc, *inputLanes, diagOS);
      break;
    case 16:
      mask = buildPltB16Mask(builder, module, loc, *inputLanes, diagOS);
      break;
    case 32:
      mask = buildPltB32Mask(builder, module, loc, *inputLanes, diagOS);
      break;
    default:
      diagOS << "VPTO LLVM emission failed: unsupported vcvt mask width "
             << contract->maskBitWidth << "\n";
      return failure();
    }
    if (failed(mask))
      return failure();
    callArgs.push_back(*mask);

    if (contract->requiresRnd) {
      auto roundMode = vcvt.getRndAttr()
                           ? parseRoundModeImmediate(*vcvt.getRnd())
                           : std::nullopt;
      if (!roundMode) {
        diagOS << "VPTO LLVM emission failed: vcvt requires valid rnd attr\n";
        return failure();
      }
      callArgs.push_back(getI32Constant(builder, loc, *roundMode));
    }
    if (contract->requiresSat) {
      auto sat =
          vcvt.getSatAttr() ? parseSaturationImmediate(*vcvt.getSat()) : std::nullopt;
      if (!sat) {
        diagOS << "VPTO LLVM emission failed: vcvt requires valid sat attr\n";
        return failure();
      }
      callArgs.push_back(getI32Constant(builder, loc, *sat));
    }
    if (contract->requiresPart) {
      auto part =
          vcvt.getPartAttr() ? parsePartImmediate(*vcvt.getPart()) : std::nullopt;
      if (!part) {
        diagOS << "VPTO LLVM emission failed: vcvt requires valid part attr\n";
        return failure();
      }
      callArgs.push_back(getI32Constant(builder, loc, *part));
    }
  } else if (auto vstar = dyn_cast<pto::VstarOp>(op)) {
    auto basePtr = requirePointerABIAddress(op, vstar.getDestination(), diagOS);
    auto alignValue = materializeAlignABIValue(op, vstar.getValue(), diagOS);
    if (failed(basePtr) || failed(alignValue))
      return failure();
    callArgs.push_back(*alignValue);
    callArgs.push_back(*basePtr);
    callArgs.push_back(getI32Constant(builder, loc, 0));
  } else if (auto vstas = dyn_cast<pto::VstasOp>(op)) {
    auto basePtr = requirePointerABIAddress(op, vstas.getDestination(), diagOS);
    auto alignValue = materializeAlignABIValue(op, vstas.getValue(), diagOS);
    Type elementType = getElementTypeFromABIValue(vstas.getDestination());
    if (failed(basePtr) || failed(alignValue) || !elementType) {
      diagOS << "VPTO LLVM emission failed: could not materialize vstas ABI "
                "inputs; destination type="
             << vstas.getDestination().getType() << ", element type="
             << (elementType ? elementType : Type()) << "\n";
      return failure();
    }
    auto offsetBytes = convertElementOffsetToBytes(op, vstas.getOffset(), elementType);
    if (failed(offsetBytes)) {
      diagOS << "VPTO LLVM emission failed: could not materialize vstas byte "
                "offset from "
             << vstas.getOffset().getType() << " using element type "
             << elementType << "\n";
      return failure();
    }
    callArgs.push_back(*alignValue);
    callArgs.push_back(*basePtr);
    callArgs.push_back(*offsetBytes);
    callArgs.push_back(getI32Constant(builder, loc, 0));
  } else if (auto vsqz = dyn_cast<pto::VsqzOp>(op)) {
    callArgs.push_back(vsqz.getInput());
    callArgs.push_back(vsqz.getMask());
    callArgs.push_back(
        getI32Constant(builder, loc, determineVsqzStoreHint(vsqz)));
  } else if (auto vusqz = dyn_cast<pto::VusqzOp>(op)) {
    callArgs.push_back(vusqz.getSrc());
    callArgs.push_back(vusqz.getMask());
  } else if (auto unpack = dyn_cast<pto::VsunpackOp>(op)) {
    Value part = castIntegerLikeTo(op, unpack.getPart(), builder.getI32Type());
    if (!part) {
      diagOS << "VPTO LLVM emission failed: could not materialize vsunpack part\n";
      return failure();
    }
    callArgs.push_back(unpack.getSrc());
    callArgs.push_back(part);
  } else if (auto unpack = dyn_cast<pto::VzunpackOp>(op)) {
    Value part = castIntegerLikeTo(op, unpack.getPart(), builder.getI32Type());
    if (!part) {
      diagOS << "VPTO LLVM emission failed: could not materialize vzunpack part\n";
      return failure();
    }
    callArgs.push_back(unpack.getSrc());
    callArgs.push_back(part);
  } else if (auto pack = dyn_cast<pto::VpackOp>(op)) {
    auto part = parseHiLoPartImmediate(pack.getPart());
    if (!part) {
      diagOS << "VPTO LLVM emission failed: unsupported vpack part "
             << pack.getPart() << "\n";
      return failure();
    }
    callArgs.push_back(pack.getSrc());
    callArgs.push_back(getI32Constant(builder, loc, *part));
  } else if (auto interleave = dyn_cast<pto::VintlvOp>(op)) {
    callArgs.push_back(interleave.getLhs());
    callArgs.push_back(interleave.getRhs());
  } else if (auto deinterleave = dyn_cast<pto::VdintlvOp>(op)) {
    callArgs.push_back(deinterleave.getLhs());
    callArgs.push_back(deinterleave.getRhs());
  } else if (auto vldsx2 = dyn_cast<pto::Vldsx2Op>(op)) {
    Type elementType = getElementTypeFromVectorLike(vldsx2.getLow().getType());
    auto offsetBytes = convertElementOffsetToBytes(op, vldsx2.getOffset(), elementType);
    auto basePtr = requirePointerABIAddress(op, vldsx2.getSource(), diagOS);
    auto dist = parseLoadX2DistImmediate(vldsx2.getDist(), elementType);
    if (!elementType || failed(offsetBytes) || failed(basePtr) || !dist) {
      if (elementType && succeeded(basePtr) && !dist)
        diagOS << "VPTO LLVM emission failed: unsupported vldsx2 dist immediate\n";
      return failure();
    }
    callArgs.push_back(*basePtr);
    callArgs.push_back(*offsetBytes);
    callArgs.push_back(getI32Constant(builder, loc, *dist));
    callArgs.push_back(getI32Constant(builder, loc, 0));
  } else if (auto vsldb = dyn_cast<pto::VsldbOp>(op)) {
    auto basePtr = requirePointerABIAddress(op, vsldb.getSource(), diagOS);
    Value packedStride = packBlockRepeatStride(
        op, vsldb.getBlockStride(), vsldb.getRepeatStride());
    if (failed(basePtr) || !packedStride) {
      if (succeeded(basePtr) && !packedStride)
        diagOS << "VPTO LLVM emission failed: could not pack vsldb control word\n";
      return failure();
    }
    callArgs.push_back(*basePtr);
    callArgs.push_back(packedStride);
    callArgs.push_back(getI32Constant(builder, loc, 0));
    callArgs.push_back(vsldb.getMask());
  } else if (auto vsstb = dyn_cast<pto::VsstbOp>(op)) {
    auto basePtr = requirePointerABIAddress(op, vsstb.getDestination(), diagOS);
    Value packedStride = packBlockRepeatStride(
        op, vsstb.getBlockStride(), vsstb.getRepeatStride());
    if (failed(basePtr) || !packedStride) {
      if (succeeded(basePtr) && !packedStride)
        diagOS << "VPTO LLVM emission failed: could not pack vsstb control word\n";
      return failure();
    }
    callArgs.push_back(vsstb.getValue());
    callArgs.push_back(*basePtr);
    callArgs.push_back(packedStride);
    callArgs.push_back(getI32Constant(builder, loc, 0));
    callArgs.push_back(vsstb.getMask());
  } else if (auto vstx2 = dyn_cast<pto::Vstsx2Op>(op)) {
    Type elementType = getElementTypeFromVectorLike(vstx2.getLow().getType());
    auto offsetBytes =
        convertElementOffsetToBytes(op, vstx2.getOffset(), elementType);
    auto basePtr =
        requirePointerABIAddress(op, vstx2.getDestination(), diagOS);
    auto dist = parseStoreX2DistImmediate(vstx2.getDist(), elementType);
    if (!elementType || failed(offsetBytes) || failed(basePtr) || !dist) {
      if (elementType && succeeded(basePtr) && !dist)
        diagOS
            << "VPTO LLVM emission failed: unsupported vstsx2 dist immediate\n";
      return failure();
    }
    Value offsetI32 = castIntegerLikeTo(op, *offsetBytes, builder.getI32Type());
    if (!offsetI32)
      return failure();
    callArgs.push_back(vstx2.getLow());
    callArgs.push_back(vstx2.getHigh());
    callArgs.push_back(*basePtr);
    callArgs.push_back(offsetI32);
    callArgs.push_back(getI32Constant(builder, loc, *dist));
    callArgs.push_back(getI32Constant(builder, loc, 0));
    callArgs.push_back(vstx2.getMask());
  } else if (auto vsts = dyn_cast<pto::VstsOp>(op)) {
    Type elementType = getElementTypeFromVectorLike(vsts.getValue().getType());
    auto offsetBytes = convertElementOffsetToBytes(
        op, op->getOperand(2), elementType);
    auto basePtr = requirePointerABIAddress(op, op->getOperand(1), diagOS);
    auto dist =
        parseStoreDistImmediate(vsts.getDist().value_or("NORM"), elementType);
    if (!elementType || failed(offsetBytes) || failed(basePtr) || !dist) {
      if (elementType && succeeded(basePtr) && !dist)
        diagOS << "VPTO LLVM emission failed: unsupported vsts dist immediate\n";
      return failure();
    }
    callArgs.push_back(op->getOperand(0));
    callArgs.push_back(*basePtr);
    callArgs.push_back(*offsetBytes);
    callArgs.push_back(getI32Constant(builder, loc, *dist));
    callArgs.push_back(getI32Constant(builder, loc, 0));
    callArgs.push_back(op->getOperand(3));
  } else if (auto vstsPost = dyn_cast<pto::VstsPostOp>(op)) {
    Type elementType = getElementTypeFromVectorLike(vstsPost.getValue().getType());
    auto offsetBytes = convertElementOffsetToBytes(op, vstsPost.getOffset(), elementType);
    auto basePtr = requirePointerABIAddress(op, vstsPost.getDestination(), diagOS);
    auto dist = parseStoreDistImmediate(vstsPost.getDist().value_or("NORM"),
                                        elementType);
    if (!elementType || failed(offsetBytes) || failed(basePtr) || !dist)
      return failure();
    callArgs.push_back(vstsPost.getValue());
    callArgs.push_back(*basePtr);
    callArgs.push_back(*offsetBytes);
    callArgs.push_back(getI32Constant(builder, loc, *dist));
    callArgs.push_back(getI32Constant(builder, loc, 1));
    callArgs.push_back(vstsPost.getMask());
  } else if (auto ppack = dyn_cast<pto::PpackOp>(op)) {
    auto part = parseHiLoPartImmediate(ppack.getPart());
    if (!part) {
      diagOS << "VPTO LLVM emission failed: unsupported ppack part "
             << ppack.getPart() << "\n";
      return failure();
    }
    callArgs.push_back(ppack.getInput());
    callArgs.push_back(getI32Constant(builder, loc, *part));
  } else if (auto punpack = dyn_cast<pto::PunpackOp>(op)) {
    auto part = parseHiLoPartImmediate(punpack.getPart());
    if (!part) {
      diagOS << "VPTO LLVM emission failed: unsupported punpack part "
             << punpack.getPart() << "\n";
      return failure();
    }
    callArgs.push_back(punpack.getInput());
    callArgs.push_back(getI32Constant(builder, loc, *part));
  } else if (auto vselr = dyn_cast<pto::VselrOp>(op)) {
    auto resultVecType = dyn_cast<VectorType>(loweredResultTypes.front());
    if (!resultVecType) {
      diagOS << "VPTO LLVM emission failed: unexpected vselr result type\n";
      return failure();
    }
    Type intrinsicVecType = resultVecType;
    if (auto resultFloat = dyn_cast<FloatType>(resultVecType.getElementType());
        resultFloat && resultFloat.isF32()) {
      intrinsicVecType =
          VectorType::get(resultVecType.getShape(), builder.getI32Type(),
                          resultVecType.getScalableDims());
    }
    intrinsicResultTypes[0] = intrinsicVecType;
    callArgs.push_back(buildBridgeCast(builder, loc, vselr.getSrc0(), intrinsicVecType));
    callArgs.push_back(vselr.getSrc1());
  } else if (isa<pto::VcmpOp, pto::VcmpsOp, pto::VselOp, pto::PnotOp,
                 pto::PselOp, pto::PandOp, pto::PorOp, pto::PxorOp,
                 pto::PdintlvB8Op, pto::PdintlvB16Op, pto::PdintlvB32Op,
                 pto::PintlvB8Op, pto::PintlvB16Op, pto::PintlvB32Op>(op)) {
    callArgs.append(op->operand_begin(), op->operand_end());
  } else if (auto plds = dyn_cast<pto::PldsOp>(op)) {
    auto basePtr = requirePointerABIAddress(op, plds.getSource(), diagOS);
    Value offset = castIntegerLikeTo(op, plds.getOffset(), builder.getI32Type());
    auto dist = parsePredicateLoadDistImmediate(plds.getDist());
    if (failed(basePtr) || !offset || !dist) {
      if (succeeded(basePtr) && offset && !dist)
        diagOS << "VPTO LLVM emission failed: unsupported plds dist immediate\n";
      return failure();
    }
    callArgs.push_back(*basePtr);
    callArgs.push_back(offset);
    callArgs.push_back(getI32Constant(builder, loc, *dist));
    callArgs.push_back(getI32Constant(builder, loc, 0));
  } else if (auto pldi = dyn_cast<pto::PldiOp>(op)) {
    auto basePtr = requirePointerABIAddress(op, pldi.getSource(), diagOS);
    Value offset = castIntegerLikeTo(op, pldi.getOffset(), builder.getI32Type());
    auto dist = parsePredicateLoadDistImmediate(pldi.getDist());
    if (failed(basePtr) || !offset || !dist) {
      if (succeeded(basePtr) && offset && !dist)
        diagOS << "VPTO LLVM emission failed: unsupported pldi dist immediate\n";
      return failure();
    }
    callArgs.push_back(*basePtr);
    callArgs.push_back(offset);
    callArgs.push_back(getI32Constant(builder, loc, *dist));
    callArgs.push_back(getI32Constant(builder, loc, 0));
  } else if (auto psts = dyn_cast<pto::PstsOp>(op)) {
    auto basePtr = requirePointerABIAddress(op, psts.getDestination(), diagOS);
    Value offset = castIntegerLikeTo(op, psts.getOffset(), builder.getI32Type());
    auto dist = parsePredicateStoreDistImmediate(psts.getDist());
    if (failed(basePtr) || !offset || !dist) {
      if (succeeded(basePtr) && offset && !dist)
        diagOS << "VPTO LLVM emission failed: unsupported psts dist immediate\n";
      return failure();
    }
    callArgs.push_back(psts.getValue());
    callArgs.push_back(*basePtr);
    callArgs.push_back(offset);
    callArgs.push_back(getI32Constant(builder, loc, *dist));
    callArgs.push_back(getI32Constant(builder, loc, 0));
  } else if (op->getName().getStringRef() == "pto.pstu") {
    auto basePtr = requirePointerABIAddress(op, op->getOperand(2), diagOS);
    auto alignValue = materializeAlignABIValue(op, op->getOperand(0), diagOS);
    if (failed(basePtr) || failed(alignValue))
      return failure();
    callArgs.push_back(op->getOperand(1));
    callArgs.push_back(*basePtr);
    callArgs.push_back(*alignValue);
  } else if (auto pstu = dyn_cast<pto::PstuOp>(op)) {
    auto basePtr = requirePointerABIAddress(op, pstu.getBase(), diagOS);
    auto alignValue = materializeAlignABIValue(op, pstu.getAlignIn(), diagOS);
    if (failed(basePtr) || failed(alignValue))
      return failure();
    callArgs.push_back(pstu.getValue());
    callArgs.push_back(*basePtr);
    callArgs.push_back(*alignValue);
  } else if (auto psti = dyn_cast<pto::PstiOp>(op)) {
    auto basePtr = requirePointerABIAddress(op, psti.getDestination(), diagOS);
    Value offset = castIntegerLikeTo(op, psti.getOffset(), builder.getI32Type());
    auto dist = parsePredicateStoreDistImmediate(psti.getDist());
    if (failed(basePtr) || !offset || !dist) {
      if (succeeded(basePtr) && offset && !dist)
        diagOS << "VPTO LLVM emission failed: unsupported psti dist immediate\n";
      return failure();
    }
    callArgs.push_back(psti.getValue());
    callArgs.push_back(*basePtr);
    callArgs.push_back(offset);
    callArgs.push_back(getI32Constant(builder, loc, *dist));
    callArgs.push_back(getI32Constant(builder, loc, 0));
  } else if (auto gather = dyn_cast<pto::Vgather2Op>(op)) {
    Type resultElemType = getElementTypeFromVectorLike(gather.getResult().getType());
    auto basePtr = requirePointerABIAddress(op, gather.getSource(), diagOS);
    auto mask = buildDynamicPltMask(builder, module, loc, gather.getActiveLanes(),
                                    resultElemType, diagOS);
    if (!resultElemType || failed(basePtr) || failed(mask))
      return failure();
    callArgs.push_back(*basePtr);
    callArgs.push_back(gather.getOffsets());
    callArgs.push_back(*mask);
  } else if (auto gather = dyn_cast<pto::Vgather2BcOp>(op)) {
    auto basePtr = requirePointerABIAddress(op, gather.getSource(), diagOS);
    if (failed(basePtr))
      return failure();
    callArgs.push_back(*basePtr);
    callArgs.push_back(gather.getOffsets());
    callArgs.push_back(gather.getMask());
  } else if (auto gather = dyn_cast<pto::VgatherbOp>(op)) {
    auto basePtr = requirePointerABIAddress(op, gather.getSource(), diagOS);
    if (failed(basePtr))
      return failure();
    callArgs.push_back(*basePtr);
    callArgs.push_back(gather.getOffsets());
    callArgs.push_back(gather.getMask());
  } else if (auto vbitsort = dyn_cast<pto::VbitsortOp>(op)) {
    auto destination = requirePointerABIAddress(op, vbitsort.getDestination(), diagOS);
    auto source = requirePointerABIAddress(op, vbitsort.getSource(), diagOS);
    auto indices = requirePointerABIAddress(op, vbitsort.getIndices(), diagOS);
    auto config = packVbitsortConfig(op, vbitsort.getRepeatTimes());
    if (failed(destination) || failed(source) || failed(indices) || failed(config))
      return failure();
    callArgs.push_back(*destination);
    callArgs.push_back(*source);
    callArgs.push_back(*indices);
    callArgs.push_back(*config);
  } else if (auto scatter = dyn_cast<pto::VscatterOp>(op)) {
    Type valueElemType = getElementTypeFromVectorLike(scatter.getValue().getType());
    auto basePtr = requirePointerABIAddress(op, scatter.getDestination(), diagOS);
    auto mask = buildDynamicPltMask(builder, module, loc, scatter.getActiveLanes(),
                                    valueElemType, diagOS);
    if (!valueElemType || failed(basePtr) || failed(mask))
      return failure();
    callArgs.push_back(scatter.getValue());
    callArgs.push_back(*basePtr);
    callArgs.push_back(scatter.getOffsets());
    callArgs.push_back(*mask);
  } else {
    diagOS << "VPTO LLVM emission failed: op lowering is not implemented for "
           << op->getName().getStringRef() << "\n";
    return failure();
  }

  SmallVector<Type> argTypes;
  for (Value arg : callArgs)
    argTypes.push_back(arg.getType());

  auto funcType = builder.getFunctionType(argTypes, intrinsicResultTypes);
  auto callee = getOrCreateExternalFunc(module, *calleeName, funcType);
  auto call = builder.create<func::CallOp>(loc, callee, callArgs);
  if (op->getNumResults() == 0)
    builder.eraseOp(op);
  else {
    SmallVector<Value> finalResults;
    finalResults.reserve(op->getNumResults());
    for (auto [idx, result] :
         llvm::enumerate(call.getResults().take_front(op->getNumResults()))) {
      Type surfaceType = surfaceResultTypes[idx];
      if (isa<LLVM::LLVMPointerType>(surfaceType)) {
        diagOS << "VPTO LLVM emission failed: unexpected LLVM pointer surface "
                  "result type on op ";
        op->print(diagOS);
        diagOS << "\n";
        return failure();
      }
      if (isa<pto::PtrType>(surfaceType)) {
        finalResults.push_back(buildBridgeCast(builder, loc, result, surfaceType));
        continue;
      }
      finalResults.push_back(result);
    }
    builder.replaceOp(op, finalResults);
  }
  return success();
}

static LogicalResult rewriteVPTOOps(ModuleOp module, llvm::raw_ostream &diagOS) {
  SmallVector<Operation *> opsToRewrite;
  module.walk([&](Operation *op) {
    if (op->getName().getDialectNamespace() != "pto")
      return;
    if (isa<pto::AddPtrOp, pto::CastPtrOp, pto::LoadScalarOp, pto::StoreScalarOp>(op))
      return;
    opsToRewrite.push_back(op);
  });

  for (Operation *op : opsToRewrite) {
    if (failed(rewriteVPTOOp(op, module, diagOS)))
      return failure();
  }

  bool hasVPTO = false;
  module.walk([&](Operation *op) {
    if (op->getName().getDialectNamespace() != "pto")
      return;
    if (isa<pto::AddPtrOp, pto::CastPtrOp, pto::LoadScalarOp,
            pto::StoreScalarOp>(op))
      return;
    hasVPTO = true;
  });

  SmallVector<Operation *> poisonOps;
  module.walk([&](Operation *op) {
    auto name = op->getName().getStringRef();
    if (name == "ub.poison" &&
        op->getNumResults() == 1 &&
        isa<pto::AlignType>(op->getResult(0).getType()))
      poisonOps.push_back(op);
  });
  for (Operation *op : poisonOps) {
    OpBuilder builder(op);
    auto abiType = cast<VectorType>(convertVPTOType(op->getResult(0).getType(), builder));
    auto zeroAttr = DenseElementsAttr::get(abiType, builder.getI8IntegerAttr(0));
    auto zero = builder.create<arith::ConstantOp>(op->getLoc(), abiType, zeroAttr);
    op->getResult(0).replaceAllUsesWith(zero.getResult());
    op->erase();
  }

  return success(!hasVPTO);
}

static Type normalizeTypeForOfficialLLVMLowering(Type type, Builder &builder) {
  type = convertVPTOType(type, builder);

  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    auto addrAttr =
        dyn_cast_or_null<pto::AddressSpaceAttr>(memrefType.getMemorySpace());
    if (!addrAttr)
      return type;
    unsigned addrSpace = getExternalPointerAddressSpace(memrefType);
    return MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                           memrefType.getLayout(),
                           builder.getI64IntegerAttr(addrSpace));
  }

  if (auto memrefType = dyn_cast<UnrankedMemRefType>(type)) {
    auto addrAttr =
        dyn_cast_or_null<pto::AddressSpaceAttr>(memrefType.getMemorySpace());
    if (!addrAttr)
      return type;
    // Official MemRef-to-LLVM conversion requires integer memory spaces.
    return UnrankedMemRefType::get(memrefType.getElementType(),
                                   builder.getI64IntegerAttr(
                                       static_cast<int64_t>(AddressSpace::GM)));
  }

  return type;
}

static void normalizeFuncSignaturesForOfficialLLVMLowering(ModuleOp module) {
  Builder builder(module.getContext());

  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    FunctionType oldType = funcOp.getFunctionType();
    SmallVector<Type> newInputs;
    SmallVector<Type> newResults;
    bool changed = false;

    newInputs.reserve(oldType.getNumInputs());
    for (Type input : oldType.getInputs()) {
      Type normalized = normalizeTypeForOfficialLLVMLowering(input, builder);
      changed |= (normalized != input);
      newInputs.push_back(normalized);
    }

    newResults.reserve(oldType.getNumResults());
    for (Type result : oldType.getResults()) {
      Type normalized = normalizeTypeForOfficialLLVMLowering(result, builder);
      changed |= (normalized != result);
      newResults.push_back(normalized);
    }

    if (!changed)
      continue;

    auto newType = builder.getFunctionType(newInputs, newResults);
    funcOp.setFunctionTypeAttr(TypeAttr::get(newType));

    if (funcOp.isExternal())
      continue;
    Block &entry = funcOp.getBody().front();
    for (auto [arg, newType] : llvm::zip(entry.getArguments(), newInputs))
      if (arg.getType() != newType)
        arg.setType(newType);
  }
}

static void ensureAIVScopeDummyDecl(ModuleOp module) {
  SymbolTable symbolTable(module);
  if (symbolTable.lookup<func::FuncOp>(kAIVScopeDummyCallee))
    return;

  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());
  auto funcType = builder.getFunctionType(TypeRange{}, TypeRange{});
  auto dummy = builder.create<func::FuncOp>(module.getLoc(),
                                            kAIVScopeDummyCallee, funcType);
  dummy.setPrivate();
}

static void materializeVecScopeCarrierLoops(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  (void)ctx->getOrLoadDialect<arith::ArithDialect>();
  (void)ctx->getOrLoadDialect<scf::SCFDialect>();
  ensureAIVScopeDummyDecl(module);

  SmallVector<pto::VecScopeOp, 16> scopes;
  module.walk([&](pto::VecScopeOp vecScope) { scopes.push_back(vecScope); });

  IRRewriter rewriter(module.getContext());
  for (pto::VecScopeOp vecScope : llvm::reverse(scopes)) {
    if (!vecScope || vecScope.getBody().empty())
      continue;

    rewriter.setInsertionPoint(vecScope);
    auto loc = vecScope.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    scf::ForOp carrier = rewriter.create<scf::ForOp>(loc, c0, c1, c1);

    Block &vecScopeBody = vecScope.getBody().front();
    Block *carrierBody = carrier.getBody();
    Operation *yield = carrierBody->getTerminator();
    carrierBody->getOperations().splice(Block::iterator(yield),
                                        vecScopeBody.getOperations(),
                                        vecScopeBody.begin(),
                                        vecScopeBody.end());
    rewriter.setInsertionPoint(yield);
    rewriter.create<func::CallOp>(loc, kAIVScopeDummyCallee, TypeRange{},
                                  ValueRange{});
    rewriter.eraseOp(vecScope);
  }

  SmallVector<pto::StrictVecScopeOp, 16> strictScopes;
  module.walk(
      [&](pto::StrictVecScopeOp strictVecScope) { strictScopes.push_back(strictVecScope); });

  for (pto::StrictVecScopeOp strictVecScope : llvm::reverse(strictScopes)) {
    if (!strictVecScope || strictVecScope.getBody().empty())
      continue;

    rewriter.setInsertionPoint(strictVecScope);
    auto loc = strictVecScope.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    scf::ForOp carrier = rewriter.create<scf::ForOp>(loc, c0, c1, c1);

    Block &strictBody = strictVecScope.getBody().front();
    Block *carrierBody = carrier.getBody();
    Operation *yield = carrierBody->getTerminator();

    IRMapping mapping;
    for (auto [blockArg, capture] :
         llvm::zip(strictBody.getArguments(), strictVecScope.getCaptures()))
      mapping.map(blockArg, capture);

    rewriter.setInsertionPoint(yield);
    for (Operation &nested : strictBody.getOperations())
      rewriter.clone(nested, mapping);
    rewriter.create<func::CallOp>(loc, kAIVScopeDummyCallee, TypeRange{},
                                  ValueRange{});

    rewriter.eraseOp(strictVecScope);
  }
}

static bool satisfiesAIVectorScopeLatchPostcondition(llvm::Loop *loop) {
  llvm::BasicBlock *latch = loop->getLoopLatch();
  if (!latch)
    return false;

  llvm::SmallVector<llvm::BasicBlock *, 4> preds(llvm::predecessors(latch));
  if (preds.size() != 1)
    return false;

  auto *predTerm = preds.front()->getTerminator();
  return predTerm && predTerm->getNumSuccessors() == 1 &&
         predTerm->getSuccessor(0) == latch;
}

// Bisheng imposes a strict CFG contract on loops carrying
// `llvm.loop.aivector_scope` metadata:
//   1. the latch must have exactly one predecessor
//   2. that predecessor must have exactly one successor, namely the latch
//
// The generic SCF/LLVM lowering pipeline does not preserve this shape for us.
// Therefore VPTO LLVM emission treats this as a required postcondition instead
// of a best-effort cleanup:
//   - if the loop already satisfies the contract, keep it as-is
//   - otherwise normalize all latch predecessors through a dummy block
//   - if normalization still cannot re-establish the contract, fail the export
//
// Failing loudly here is intentional. Silently attaching aivscope metadata to
// an unsupported latch shape only defers the problem into Bisheng as a backend
// crash, which makes future regressions harder to diagnose.
static LogicalResult ensureDummyPredForAIVectorScopeLatch(llvm::Loop *loop,
                                                          llvm::raw_ostream &diagOS) {
  if (satisfiesAIVectorScopeLatchPostcondition(loop))
    return success();

  llvm::BasicBlock *latch = loop->getLoopLatch();
  if (!latch) {
    diagOS << "VPTO LLVM emission failed: aivscope loop is missing a latch\n";
    return failure();
  }

  llvm::SmallVector<llvm::BasicBlock *, 4> preds(llvm::predecessors(latch));
  if (preds.empty()) {
    diagOS << "VPTO LLVM emission failed: aivscope latch has no predecessor\n";
    return failure();
  }

  auto *dummy = llvm::SplitBlockPredecessors(
      latch, preds, "aivscope.dummy", static_cast<llvm::DominatorTree *>(nullptr),
      static_cast<llvm::LoopInfo *>(nullptr), nullptr, /*PreserveLCSSA=*/false);
  if (!dummy) {
    diagOS << "VPTO LLVM emission failed: failed to normalize aivscope latch "
              "predecessors\n";
    return failure();
  }

  if (!satisfiesAIVectorScopeLatchPostcondition(loop)) {
    diagOS << "VPTO LLVM emission failed: normalized aivscope latch still does "
              "not satisfy the single-predecessor/single-successor contract\n";
    return failure();
  }
  return success();
}

static LogicalResult attachAIVectorScopeMetadata(
    llvm::Module &llvmModule, llvm::raw_ostream &diagOS) {
  llvm::Function *dummyCallee = llvmModule.getFunction(kAIVScopeDummyCallee);
  if (!dummyCallee)
    return success();

  for (llvm::Function &function : llvmModule) {
    if (function.isDeclaration())
      continue;
    llvm::DominatorTree dt(function);
    llvm::LoopInfo loopInfo(dt);

    // Stage 1: collect the lowered vecscope markers in this function. Each
    // marker should end up inside the final LLVM loop that carries one
    // `pto.vecscope` / `pto.strict_vecscope`.
    llvm::SmallVector<llvm::CallInst *, 4> dummyCalls;
    for (llvm::BasicBlock &block : function) {
      for (llvm::Instruction &inst : block) {
        auto *call = dyn_cast<llvm::CallInst>(&inst);
        if (call && call->getCalledFunction() == dummyCallee)
          dummyCalls.push_back(call);
      }
    }

    for (llvm::CallInst *dummyCall : dummyCalls) {
      llvm::BasicBlock *markedBlock = dummyCall->getParent();
      llvm::Loop *loop = loopInfo.getLoopFor(markedBlock);
      if (!loop) {
        diagOS << "VPTO LLVM emission failed: aivscope_dummy in function "
               << function.getName() << " does not belong to an LLVM loop\n";
        return failure();
      }

      // Stage 2: if the marker ended up in the loop latch, split the block so
      // the eventual latch stays as a clean backedge block instead of carrying
      // vector-thread side effects.
      if (markedBlock == loop->getLoopLatch() &&
          dummyCall != markedBlock->getTerminator()) {
        markedBlock->splitBasicBlock(dummyCall->getIterator(), "aivscope.latch");
        dt.recalculate(function);
        loopInfo.releaseMemory();
        loopInfo.analyze(dt);
        markedBlock = dummyCall->getParent();
        loop = loopInfo.getLoopFor(markedBlock);
        if (!loop) {
          diagOS << "VPTO LLVM emission failed: split aivscope latch in "
                 << function.getName()
                 << " no longer belongs to an LLVM loop\n";
          return failure();
        }
      }

      if (failed(ensureDummyPredForAIVectorScopeLatch(loop, diagOS)))
        return failure();

      // Stage 3: after any CFG surgery, re-query the loop and attach
      // `llvm.loop.aivector_scope` to the normalized latch backedge. The dummy
      // marker has served its purpose by this point and is removed.
      dt.recalculate(function);
      loopInfo.releaseMemory();
      loopInfo.analyze(dt);
      loop = loopInfo.getLoopFor(markedBlock);
      if (!loop) {
        diagOS << "VPTO LLVM emission failed: aivscope_dummy in function "
               << function.getName()
               << " lost its loop after latch normalization\n";
        return failure();
      }

      llvm::BasicBlock *latch = loop->getLoopLatch();
      auto *branch = dyn_cast_or_null<llvm::BranchInst>(
          latch ? latch->getTerminator() : nullptr);
      if (!branch || branch->isConditional()) {
        diagOS << "VPTO LLVM emission failed: normalized aivscope loop in "
               << function.getName()
               << " does not have an unconditional latch backedge\n";
        return failure();
      }

      llvm::LLVMContext &ctx = llvmModule.getContext();
      llvm::Metadata *ops[] = {
          nullptr, llvm::MDNode::get(ctx, llvm::MDString::get(ctx, "llvm.loop.aivector_scope"))};
      auto *loopID = llvm::MDNode::getDistinct(ctx, ops);
      loopID->replaceOperandWith(0, loopID);
      branch->setMetadata(llvm::LLVMContext::MD_loop, loopID);
      dummyCall->eraseFromParent();
    }
  }

  if (dummyCallee->use_empty())
    dummyCallee->eraseFromParent();
  return success();
}

static void attachHIVMKernelAnnotations(llvm::Module &llvmModule) {
  llvm::NamedMDNode *annotations = llvmModule.getOrInsertNamedMetadata(
      "hivm.annotations");
  llvm::LLVMContext &ctx = llvmModule.getContext();
  llvm::Type *i32Ty = llvm::Type::getInt32Ty(ctx);
  llvm::Constant *one = llvm::ConstantInt::get(i32Ty, 1);

  auto addAnnotation = [&](llvm::Function &function, llvm::StringRef kind) {
    llvm::Metadata *ops[] = {
        llvm::ValueAsMetadata::get(&function),
        llvm::MDString::get(ctx, kind),
        llvm::ConstantAsMetadata::get(one)};
    annotations->addOperand(llvm::MDNode::get(ctx, ops));
  };

  for (llvm::Function &function : llvmModule) {
    if (function.isDeclaration())
      continue;
    if (function.getLinkage() != llvm::GlobalValue::ExternalLinkage)
      continue;

    llvm::StringRef name = function.getName();
    if (name.contains(".extracted") || name.contains(".vector.thread"))
      continue;

    addAnnotation(function, "kernel");
    addAnnotation(function, "kernel_with_simd");
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
queryDefaultTargetAttrs(const VPTOEmissionOptions &options,
                        llvm::raw_ostream &diagOS) {
  static llvm::StringMap<QueriedTargetAttrs> cache;

  if (options.targetTriple.empty() || options.march.empty() ||
      options.aicoreArch.empty()) {
    diagOS << "VPTO LLVM emission failed: missing target query options\n";
    return failure();
  }

  std::string cacheKey =
      options.targetTriple + "|" + options.march + "|" + options.aicoreArch;
  if (auto it = cache.find(cacheKey); it != cache.end())
    return it->second;

  auto bisheng = llvm::sys::findProgramByName("bisheng");
  if (!bisheng) {
    diagOS << "VPTO LLVM emission failed: unable to find 'bisheng' in PATH\n";
    return failure();
  }
  const std::string &bishengPath = *bisheng;

  llvm::SmallString<64> inputPath;
  llvm::SmallString<64> outputPath;
  int inputFD = -1;
  int outputFD = -1;
  if (auto ec = llvm::sys::fs::createTemporaryFile("ptoas-vpto-target-query",
                                                   "c", inputFD, inputPath)) {
    diagOS << "VPTO LLVM emission failed: cannot create bisheng query input: "
           << ec.message() << "\n";
    return failure();
  }
  if (auto ec = llvm::sys::fs::createTemporaryFile("ptoas-vpto-target-query",
                                                   "ll", outputFD, outputPath)) {
    llvm::sys::fs::remove(inputPath);
    llvm::sys::Process::SafelyCloseFileDescriptor(inputFD);
    diagOS << "VPTO LLVM emission failed: cannot create bisheng query output: "
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
  if (auto ec = llvm::sys::fs::createTemporaryFile("ptoas-vpto-target-query",
                                                   "stderr", stderrFD,
                                                   stderrPath)) {
    diagOS << "VPTO LLVM emission failed: cannot create bisheng query stderr: "
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
    diagOS << "VPTO LLVM emission failed: bisheng target query failed\n";
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
    diagOS << "VPTO LLVM emission failed: cannot read bisheng query output\n";
    return failure();
  }

  FailureOr<std::string> targetCPU =
      extractQuotedLLVMFnAttr(outputBuffer.get()->getBuffer(), "target-cpu");
  FailureOr<std::string> targetFeatures = extractQuotedLLVMFnAttr(
      outputBuffer.get()->getBuffer(), "target-features");
  if (failed(targetCPU) || failed(targetFeatures)) {
    diagOS << "VPTO LLVM emission failed: cannot parse bisheng target attrs\n";
    diagOS << outputBuffer.get()->getBuffer() << "\n";
    return failure();
  }

  QueriedTargetAttrs attrs{*targetCPU, *targetFeatures};
  cache[cacheKey] = attrs;
  return attrs;
}

static LogicalResult
applyQueriedTargetAttrs(ModuleOp module, const VPTOEmissionOptions &options,
                        llvm::raw_ostream &diagOS) {
  FailureOr<QueriedTargetAttrs> attrs = queryDefaultTargetAttrs(options, diagOS);
  if (failed(attrs)) {
    if (options.defaultTargetCPU.empty() ||
        options.defaultTargetFeatures.empty())
      return failure();
    diagOS << "VPTO LLVM emission: falling back to configured default target attributes\n";
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
      diagOS << "VPTO LLVM emission warning: skipping ABI rewrite for "
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
          diagOS << "VPTO LLVM emission failed: cannot cast scalar arg for "
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
        diagOS << "VPTO LLVM emission failed: cannot materialize direct ABI for "
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
      diagOS << "VPTO LLVM emission failed: cloned function body is empty for "
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
buildLLVMModuleFromPreparedVPTO(ModuleOp module,
                                llvm::LLVMContext &llvmContext,
                                const VPTOEmissionOptions &options,
                                llvm::raw_ostream &diagOS) {
  materializeVecScopeCarrierLoops(module);

  if (failed(normalizePtoMemRefSpaces(module, diagOS)))
    return nullptr;

  if (failed(normalizePtoAlignsToABI(module, diagOS)))
    return nullptr;

  if (failed(rewriteVPTOOps(module, diagOS))) {
    diagOS << "VPTO LLVM emission failed: VPTO-to-call rewriting failed\n";
    return nullptr;
  }

  if (failed(normalizePtoPtrsToLLVM(module, diagOS)))
    return nullptr;

  normalizeFuncSignaturesForOfficialLLVMLowering(module);

  PassManager pm(module.getContext());
  pm.enableVerifier();
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  if (failed(pm.run(module))) {
    diagOS << "VPTO LLVM emission failed: official lowering pipeline failed\n";
    return nullptr;
  }

  if (failed(applyQueriedTargetAttrs(module, options, diagOS)))
    return nullptr;

  registerBuiltinDialectTranslation(*module.getContext());
  registerLLVMDialectTranslation(*module.getContext());
  auto llvmModule = translateModuleToLLVMIR(module.getOperation(), llvmContext);
  if (!llvmModule) {
    diagOS << "VPTO LLVM emission failed: LLVM IR export failed\n";
    return nullptr;
  }

  if (failed(attachAIVectorScopeMetadata(*llvmModule, diagOS)))
    return nullptr;
  attachHIVMKernelAnnotations(*llvmModule);
  llvmModule->setModuleIdentifier("ptoas.hivm.official");
  llvmModule->setSourceFileName("ptoas.hivm.official");
  return llvmModule;
}

} // namespace

LogicalResult
translateVPTOModuleToLLVMText(ModuleOp module, llvm::raw_ostream &os,
                              const VPTOEmissionOptions &options,
                              llvm::raw_ostream &diagOS) {
  llvm::LLVMContext llvmContext;
  auto llvmModule =
      buildLLVMModuleFromPreparedVPTO(module, llvmContext, options, diagOS);
  if (!llvmModule)
    return failure();
  llvmModule->print(os, nullptr);
  return success();
}

LogicalResult
translateVPTOModuleToLLVMBitcode(ModuleOp module, llvm::raw_ostream &os,
                                 const VPTOEmissionOptions &options,
                                 llvm::raw_ostream &diagOS) {
  llvm::LLVMContext llvmContext;
  auto llvmModule =
      buildLLVMModuleFromPreparedVPTO(module, llvmContext, options, diagOS);
  if (!llvmModule)
    return failure();
  llvm::WriteBitcodeToFile(*llvmModule, os);
  return success();
}

} // namespace mlir::pto
