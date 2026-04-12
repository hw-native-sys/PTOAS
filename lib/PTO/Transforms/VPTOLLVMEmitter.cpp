#include "PTO/Transforms/VPTOLLVMEmitter.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOSyncUtils.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/LLVMContext.h"

namespace mlir::pto {

void materializeVecScopeCarrierLoops(ModuleOp module);
LogicalResult normalizePtoMemRefSpaces(ModuleOp module,
                                       llvm::raw_ostream &diagOS);
LogicalResult applyQueriedTargetAttrs(ModuleOp module,
                                      const VPTOEmissionOptions &options,
                                      llvm::raw_ostream &diagOS);
LogicalResult attachAIVectorScopeMetadata(llvm::Module &llvmModule,
                                          llvm::raw_ostream &diagOS);
void attachHIVMKernelAnnotations(llvm::Module &llvmModule);

namespace {

static std::string getElementTypeFragment(Type type);
static Type getElementTypeFromVectorLike(Type type);
static std::optional<int64_t> getElementCountFromVectorLike(Type type);

static Type convertVPTOType(Type type, Builder &builder) {
  if (auto vecType = dyn_cast<pto::VRegType>(type))
    return VectorType::get({vecType.getElementCount()}, vecType.getElementType());
  if (isa<pto::MaskType>(type))
    return VectorType::get({256}, builder.getI1Type());
  if (isa<pto::AlignType>(type))
    return VectorType::get({32}, builder.getI8Type());
  if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
    return LLVM::LLVMPointerType::get(
        builder.getContext(),
        static_cast<unsigned>(ptrType.getMemorySpace().getAddressSpace()));
  }
  return type;
}

static bool hasVPTOConvertibleType(Type type) {
  return isa<pto::VRegType, pto::MaskType, pto::AlignType, pto::PtrType>(type);
}

static bool hasVPTOConvertibleType(TypeRange types) {
  return llvm::any_of(types, [](Type type) { return hasVPTOConvertibleType(type); });
}

static Value materializeVPTOCast(OpBuilder &builder, Type resultType,
                                 ValueRange inputs, Location loc) {
  if (inputs.size() != 1)
    return {};
  return builder
      .create<UnrealizedConversionCastOp>(loc, TypeRange{resultType}, inputs)
      .getResult(0);
}

class VPTOTypeConverter final : public TypeConverter {
public:
  explicit VPTOTypeConverter(MLIRContext *context) {
    addConversion([](Type type) { return type; });
    addConversion([](Type type) -> Type {
      // The conversion callback outlives this constructor, so build on demand
      // from the current type context instead of capturing a local Builder.
      Builder builder(type.getContext());
      return convertVPTOType(type, builder);
    });
    addSourceMaterialization(materializeVPTOCast);
    addTargetMaterialization(materializeVPTOCast);
    addArgumentMaterialization(materializeVPTOCast);
  }
};

struct PlannedDecl {
  std::string name;
  FunctionType type;
};

struct LoweringState {
  SmallVector<PlannedDecl> plannedDecls;
};

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

static Value getI64Constant(OpBuilder &builder, Location loc, uint64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(value))
      .getResult();
}

static Value getI32Constant(OpBuilder &builder, Location loc, uint64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(value))
      .getResult();
}

static FailureOr<StringRef> buildLaneTypedCallee(MLIRContext *context,
                                                 Type resultType,
                                                 StringRef stem,
                                                 StringRef suffix) {
  std::string vec =
      getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
    return failure();

  return StringAttr::get(context, "llvm.hivm." + stem.str() + ".v" +
                                      std::to_string(*lanes) + vec +
                                      suffix.str())
      .getValue();
}

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

static Type getElementTypeFromVectorLike(Type type) {
  if (auto vecType = dyn_cast<pto::VRegType>(type))
    return vecType.getElementType();
  if (auto vecType = dyn_cast<VectorType>(type))
    return vecType.getElementType();
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

static FailureOr<Value> normalizeVdupScalarOperand(OpBuilder &builder, Location loc,
                                                   pto::VdupOp op) {
  Value input = op.getInput();
  auto intType = dyn_cast<IntegerType>(input.getType());
  if (!intType || intType.getWidth() != 8)
    return input;

  Type resultElemType = getElementTypeFromVectorLike(op.getResult().getType());
  std::string resultElemFragment = getElementTypeFragment(resultElemType);
  if (resultElemFragment != "s8" && resultElemFragment != "u8")
    return input;

  Type i16Type = builder.getIntegerType(16);
  if (resultElemFragment == "u8")
    return builder.create<arith::ExtUIOp>(loc, i16Type, input).getResult();
  return builder.create<arith::ExtSIOp>(loc, i16Type, input).getResult();
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

static std::optional<uint64_t> parseHiLoPartImmediate(StringRef part) {
  if (part == "LOWER")
    return 0;
  if (part == "HIGHER")
    return 1;
  return std::nullopt;
}

static std::optional<uint64_t> parseRoundModeImmediate(StringRef roundMode) {
  if (roundMode == "R" || roundMode == "ROUND_R")
    return 0;
  if (roundMode == "A" || roundMode == "ROUND_A")
    return 1;
  if (roundMode == "F" || roundMode == "ROUND_F")
    return 2;
  if (roundMode == "C" || roundMode == "ROUND_C")
    return 3;
  if (roundMode == "Z" || roundMode == "ROUND_Z")
    return 4;
  if (roundMode == "O" || roundMode == "ROUND_O")
    return 5;
  return std::nullopt;
}

static std::optional<uint64_t> parseSaturationImmediate(StringRef sat) {
  if (sat == "SAT" || sat == "RS_ENABLE")
    return 0;
  if (sat == "NOSAT" || sat == "RS_DISABLE")
    return 1;
  return std::nullopt;
}

static std::optional<uint64_t> parsePartImmediate(StringRef part) {
  if (part == "EVEN" || part == "PART_EVEN")
    return 0;
  if (part == "ODD" || part == "PART_ODD")
    return 1;
  return std::nullopt;
}

static std::optional<uint64_t> parsePredicateStoreDistImmediate(StringRef dist) {
  if (dist == "NORM")
    return 0;
  if (dist == "PK")
    return 1;
  return std::nullopt;
}

static std::optional<uint64_t> parsePredicateLoadDistImmediate(StringRef dist) {
  if (dist.empty() || dist == "NORM")
    return 0;
  if (dist == "US")
    return 1;
  if (dist == "DS")
    return 2;
  return std::nullopt;
}

static std::optional<int32_t> parsePostModeImmediate(StringRef mode) {
  if (mode == "NO_POST_UPDATE")
    return 0;
  if (mode == "POST_UPDATE")
    return 1;
  return std::nullopt;
}

static std::optional<uint64_t> parsePipeImmediate(StringRef pipe) {
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

static std::optional<uint64_t> parseEventImmediate(StringRef event) {
  if (!event.consume_front("EVENT_ID"))
    return std::nullopt;
  uint64_t value = 0;
  if (event.getAsInteger(10, value))
    return std::nullopt;
  return value;
}

static std::optional<uint64_t> parseSprImmediate(StringRef spr) {
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

static std::optional<uint64_t> parseLoadDistImmediate(StringRef dist,
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

static std::optional<uint64_t> parseLoadX2DistImmediate(StringRef dist,
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

static std::optional<uint64_t> parseStoreDistImmediate(StringRef dist,
                                                       Type elementType) {
  auto width = getDistElementWidth(elementType);
  if (dist.empty() || dist == "NORM") {
    if (!width)
      return std::nullopt;
    if (*width == 8)
      return 0;
    if (*width == 16)
      return 1;
    if (*width == 32)
      return 2;
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

static std::optional<uint64_t> parseStoreX2DistImmediate(StringRef dist,
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

static std::optional<uint64_t> parseOrderImmediate(StringRef order) {
  if (order.empty() || order == "ASC")
    return 0;
  if (order == "DESC")
    return 1;
  return std::nullopt;
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
packCopyGmToUbConfig0(Operation *anchor, ValueRange operands) {
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

static FailureOr<Value> materializeDynamicPltMask(ConversionPatternRewriter &rewriter,
                                                  LoweringState &state,
                                                  Location loc,
                                                  Value laneCount,
                                                  Type vectorElemType) {
  Type i32Type = rewriter.getI32Type();
  Value laneCountI32 = laneCount;
  if (laneCountI32.getType() != i32Type) {
    laneCountI32 = castIntegerLikeTo(rewriter.getInsertionBlock()->getParentOp(),
                                     laneCountI32, i32Type);
    if (!laneCountI32)
      return failure();
  }

  StringRef calleeName;
  if (vectorElemType.isF32()) {
    calleeName = StringRef("llvm.hivm.plt.b32.v300");
  } else if (vectorElemType.isF16() || vectorElemType.isBF16()) {
    calleeName = StringRef("llvm.hivm.plt.b16.v300");
  } else if (auto intType = dyn_cast<IntegerType>(vectorElemType)) {
    if (intType.getWidth() == 32)
      calleeName = StringRef("llvm.hivm.plt.b32.v300");
    else if (intType.getWidth() == 16)
      calleeName = StringRef("llvm.hivm.plt.b16.v300");
    else if (intType.getWidth() == 8)
      calleeName = StringRef("llvm.hivm.plt.b8.v300");
  }
  if (calleeName.empty())
    return failure();

  Type maskType = VectorType::get({256}, rewriter.getI1Type());
  auto funcType =
      rewriter.getFunctionType(TypeRange{i32Type}, TypeRange{maskType, i32Type});
  auto call = rewriter.create<func::CallOp>(loc, calleeName, funcType.getResults(),
                                            ValueRange{laneCountI32});
  state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
  return call.getResult(0);
}

static FailureOr<StringRef> buildCarryBinaryCallee(MLIRContext *context,
                                                   Type resultType,
                                                   StringRef stem) {
  std::string vec =
      getElementTypeFragment(cast<pto::VRegType>(resultType).getElementType());
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
    return failure();
  return StringAttr::get(context, "llvm.hivm." + stem.str() + ".v" +
                                      std::to_string(*lanes) + vec)
      .getValue();
}

template <typename UnaryOp>
static StringRef getUnaryMaskedStem() {
  if constexpr (std::is_same_v<UnaryOp, pto::VabsOp>)
    return "vabs";
  if constexpr (std::is_same_v<UnaryOp, pto::VexpOp>)
    return "vexp";
  if constexpr (std::is_same_v<UnaryOp, pto::VlnOp>)
    return "vln";
  if constexpr (std::is_same_v<UnaryOp, pto::VnegOp>)
    return "vneg";
  if constexpr (std::is_same_v<UnaryOp, pto::VsqrtOp>)
    return "vsqrt";
  if constexpr (std::is_same_v<UnaryOp, pto::VreluOp>)
    return "vrelu";
  if constexpr (std::is_same_v<UnaryOp, pto::VnotOp>)
    return "vnot";
  return {};
}

template <typename BinaryOp>
static StringRef getBinaryMaskedStem() {
  if constexpr (std::is_same_v<BinaryOp, pto::VaddOp>)
    return "vadd";
  if constexpr (std::is_same_v<BinaryOp, pto::VsubOp>)
    return "vsub";
  if constexpr (std::is_same_v<BinaryOp, pto::VmulOp>)
    return "vmul";
  if constexpr (std::is_same_v<BinaryOp, pto::VdivOp>)
    return "vdiv";
  if constexpr (std::is_same_v<BinaryOp, pto::VmaxOp>)
    return "vmax";
  if constexpr (std::is_same_v<BinaryOp, pto::VminOp>)
    return "vmin";
  if constexpr (std::is_same_v<BinaryOp, pto::VandOp>)
    return "vand";
  if constexpr (std::is_same_v<BinaryOp, pto::VorOp>)
    return "vor";
  if constexpr (std::is_same_v<BinaryOp, pto::VxorOp>)
    return "vxor";
  if constexpr (std::is_same_v<BinaryOp, pto::VshlOp>)
    return "vshl";
  if constexpr (std::is_same_v<BinaryOp, pto::VshrOp>)
    return "vshr";
  return {};
}

template <typename CarryOp>
static StringRef getCarryBinaryStem() {
  if constexpr (std::is_same_v<CarryOp, pto::VaddcOp>)
    return "vaddc";
  if constexpr (std::is_same_v<CarryOp, pto::VsubcOp>)
    return "vsubc";
  if constexpr (std::is_same_v<CarryOp, pto::VaddcsOp>)
    return "vaddcs";
  if constexpr (std::is_same_v<CarryOp, pto::VsubcsOp>)
    return "vsubcs";
  return {};
}

template <typename CarryOp>
static constexpr bool hasCarryInput() {
  return std::is_same_v<CarryOp, pto::VaddcsOp> ||
         std::is_same_v<CarryOp, pto::VsubcsOp>;
}

static FailureOr<StringRef> buildVselCallee(MLIRContext *context,
                                            Type resultType) {
  std::string vec =
      getElementTypeFragment(cast<pto::VRegType>(resultType).getElementType());
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
    return failure();
  return StringAttr::get(context, "llvm.hivm.vsel.v" + std::to_string(*lanes) +
                                      vec)
      .getValue();
}

static FailureOr<StringRef> buildVselrCallee(MLIRContext *context,
                                             Type resultType) {
  Type elemType = getElementTypeFromVectorLike(resultType);
  auto lanes = getElementCountFromVectorLike(resultType);
  if (!elemType || !lanes)
    return failure();

  std::string vec = getElementTypeFragment(elemType);
  if (auto floatType = dyn_cast<FloatType>(elemType);
      floatType && floatType.isF32())
    vec = "u32";
  if (vec.empty())
    return failure();

  return StringAttr::get(context, "llvm.hivm.vselr.v" + std::to_string(*lanes) +
                                      vec)
      .getValue();
}

static FailureOr<StringRef> buildVdupCallee(MLIRContext *context, pto::VdupOp op) {
  Type inputType = op.getInput().getType();
  Type resultType = op.getResult().getType();
  std::string vec = getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
    return failure();

  if (isa<VectorType, pto::VRegType>(inputType)) {
    StringRef position = op.getPosition().value_or("LOWEST");
    StringRef family = position == "HIGHEST" ? "vdupm" : "vdup";
    return StringAttr::get(context, "llvm.hivm." + family.str() + ".v" +
                                        std::to_string(*lanes) + vec + ".z")
        .getValue();
  }

  return StringAttr::get(context, "llvm.hivm.vdups.v" + std::to_string(*lanes) +
                                      vec + ".z")
      .getValue();
}

static FailureOr<StringRef> buildVbrCallee(MLIRContext *context, Type scalarType) {
  std::string scalar = getVbrScalarFragment(scalarType);
  if (scalar.empty())
    return failure();
  return StringAttr::get(context, "llvm.hivm.vbr." + scalar + ".v300").getValue();
}

static FailureOr<StringRef> buildPstuCallee(MLIRContext *context, pto::PstuOp op) {
  if (auto maskType = dyn_cast<pto::MaskType>(op.getValue().getType())) {
    if (maskType.isB16())
      return StringAttr::get(context, "llvm.hivm.pstu.b16").getValue();
    if (maskType.isB32())
      return StringAttr::get(context, "llvm.hivm.pstu.b32").getValue();
  }
  return failure();
}

static StringRef buildVstusCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vstus").getValue();
}

static StringRef buildVsturCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vstur").getValue();
}

static StringRef buildInitAlignCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.init.vector.align.data").getValue();
}

static StringRef buildSprclrCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.sprclr").getValue();
}

static StringRef buildVstarCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vstar").getValue();
}

static StringRef buildVstasCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vstas").getValue();
}

static FailureOr<StringRef> buildVldsPostCallee(MLIRContext *context,
                                                Type resultType) {
  std::string vec = getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
    return failure();
  return StringAttr::get(context, "llvm.hivm.vldsx1.post.v" +
                                      std::to_string(*lanes) + vec)
      .getValue();
}

static FailureOr<StringRef> buildVstsPostCallee(MLIRContext *context,
                                                Type valueType) {
  std::string vec = getElementTypeFragment(getElementTypeFromVectorLike(valueType));
  auto lanes = getElementCountFromVectorLike(valueType);
  if (vec.empty() || !lanes)
    return failure();
  return StringAttr::get(context, "llvm.hivm.vstsx1.post.v" +
                                      std::to_string(*lanes) + vec)
      .getValue();
}

static StringRef buildVldasCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vldas").getValue();
}

static FailureOr<StringRef> buildVldusCallee(MLIRContext *context,
                                             Type resultType) {
  std::string vec = getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
    return failure();
  return StringAttr::get(context, "llvm.hivm.vldus.v" +
                                      std::to_string(*lanes) + vec)
      .getValue();
}

static FailureOr<StringRef> buildVcmpCallee(MLIRContext *context, Type inputType,
                                            StringRef cmpMode,
                                            bool isScalarCompare) {
  std::string elem = getElementTypeFragment(getElementTypeFromVectorLike(inputType));
  if (elem.empty())
    return failure();
  StringRef stem = isScalarCompare ? "vcmps" : "vcmp";
  return StringAttr::get(context, "llvm.hivm." + stem.str() + "." +
                                      cmpMode.str() + "." + elem + ".z")
      .getValue();
}

template <typename VecScalarOp>
static StringRef getVecScalarMaskedStem() {
  if constexpr (std::is_same_v<VecScalarOp, pto::VmulsOp>)
    return "vmuls";
  if constexpr (std::is_same_v<VecScalarOp, pto::VaddsOp>)
    return "vadds";
  if constexpr (std::is_same_v<VecScalarOp, pto::VmaxsOp>)
    return "vmaxs";
  if constexpr (std::is_same_v<VecScalarOp, pto::VminsOp>)
    return "vmins";
  if constexpr (std::is_same_v<VecScalarOp, pto::VlreluOp>)
    return "vlrelu";
  if constexpr (std::is_same_v<VecScalarOp, pto::VshlsOp>)
    return "vshls";
  if constexpr (std::is_same_v<VecScalarOp, pto::VshrsOp>)
    return "vshrs";
  return {};
}

template <typename ReductionOp>
static StringRef getReductionUnaryStem() {
  if constexpr (std::is_same_v<ReductionOp, pto::VcaddOp>)
    return "vcadd";
  if constexpr (std::is_same_v<ReductionOp, pto::VcmaxOp>)
    return "vcmax";
  if constexpr (std::is_same_v<ReductionOp, pto::VcminOp>)
    return "vcmin";
  if constexpr (std::is_same_v<ReductionOp, pto::VcgaddOp>)
    return "vcgadd";
  if constexpr (std::is_same_v<ReductionOp, pto::VcgmaxOp>)
    return "vcgmax";
  if constexpr (std::is_same_v<ReductionOp, pto::VcgminOp>)
    return "vcgmin";
  if constexpr (std::is_same_v<ReductionOp, pto::VcpaddOp>)
    return "vcpadd";
  return {};
}

static FailureOr<StringRef> buildCopyGmToUbCallee(MLIRContext *context,
                                                  pto::CopyGmToUbufOp op) {
  Type elementType = cast<pto::PtrType>(op.getSource().getType()).getElementType();
  std::string elem = getCopyElementFragment(elementType);
  if (elem.empty())
    return failure();
  return StringAttr::get(context, "llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2." + elem +
                                      ".DV")
      .getValue();
}

static StringRef buildCopyUbToGmCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.DV")
      .getValue();
}

static StringRef buildPstiCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.psti.b8").getValue();
}

static StringRef buildPstsCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.psts.b8").getValue();
}

static StringRef buildPldiCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pldi.b8").getValue();
}

static StringRef buildPldsCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.plds.b8").getValue();
}

static StringRef buildPnotCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pnot.z").getValue();
}

static StringRef buildPselCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.psel").getValue();
}

static StringRef buildPandCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pand.z").getValue();
}

static StringRef buildPorCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.por.z").getValue();
}

static StringRef buildPxorCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pxor.z").getValue();
}

static StringRef buildPpackCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.ppack.z").getValue();
}

static StringRef buildPunpackCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.punpack").getValue();
}

template <typename Op>
static StringRef buildPredicatePairReorderCallee(MLIRContext *context);

template <>
StringRef buildPredicatePairReorderCallee<pto::PdintlvB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pdintlv.b8").getValue();
}

template <>
StringRef buildPredicatePairReorderCallee<pto::PdintlvB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pdintlv.b16").getValue();
}

template <>
StringRef buildPredicatePairReorderCallee<pto::PdintlvB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pdintlv.b32").getValue();
}

template <>
StringRef buildPredicatePairReorderCallee<pto::PintlvB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pintlv.b8").getValue();
}

template <>
StringRef buildPredicatePairReorderCallee<pto::PintlvB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pintlv.b16").getValue();
}

template <>
StringRef buildPredicatePairReorderCallee<pto::PintlvB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pintlv.b32").getValue();
}

static FailureOr<StringRef> buildInterleaveCallee(MLIRContext *context,
                                                  Type resultType,
                                                  StringRef stem) {
  return buildLaneTypedCallee(context, resultType, stem, "");
}

static FailureOr<StringRef> buildUnpackCallee(MLIRContext *context,
                                              Type inputType,
                                              Type resultType,
                                              StringRef stem) {
  std::string input =
      getElementTypeFragment(getElementTypeFromVectorLike(inputType));
  std::string result =
      getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  if (input.empty() || result.empty())
    return failure();
  return StringAttr::get(context,
                         "llvm.hivm." + stem.str() + "." + input + "2" + result)
      .getValue();
}

static FailureOr<StringRef> buildVpackCallee(MLIRContext *context, Type inputType,
                                             Type resultType) {
  std::string input =
      getElementTypeFragment(getElementTypeFromVectorLike(inputType));
  std::string result =
      getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  if (input.empty() || result.empty())
    return failure();

  return StringAttr::get(context, "llvm.hivm.vpack." + input + "2" + result + ".x")
      .getValue();
}

static FailureOr<StringRef> buildVsqzCallee(MLIRContext *context,
                                            Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vsqz", ".x.v300");
}

static FailureOr<StringRef> buildVusqzCallee(MLIRContext *context,
                                             Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vusqz", ".m");
}

static FailureOr<StringRef> buildVmulaCallee(MLIRContext *context,
                                             Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vmula", ".m");
}

static FailureOr<StringRef> buildVmullCallee(MLIRContext *context,
                                             Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vmull", "");
}

template <typename StoreOp>
static StringRef getPredicateStoreCallee(MLIRContext *context);

template <>
StringRef getPredicateStoreCallee<pto::PstiOp>(MLIRContext *context) {
  return buildPstiCallee(context);
}

template <>
StringRef getPredicateStoreCallee<pto::PstsOp>(MLIRContext *context) {
  return buildPstsCallee(context);
}

template <typename LoadOp>
static StringRef getPredicateLoadCallee(MLIRContext *context);

template <>
StringRef getPredicateLoadCallee<pto::PldiOp>(MLIRContext *context) {
  return buildPldiCallee(context);
}

template <>
StringRef getPredicateLoadCallee<pto::PldsOp>(MLIRContext *context) {
  return buildPldsCallee(context);
}

template <typename PredicateMaskOp>
static StringRef getPredicateMaskCallee(MLIRContext *context);

template <>
StringRef getPredicateMaskCallee<pto::PnotOp>(MLIRContext *context) {
  return buildPnotCallee(context);
}

template <>
StringRef getPredicateMaskCallee<pto::PselOp>(MLIRContext *context) {
  return buildPselCallee(context);
}

template <>
StringRef getPredicateMaskCallee<pto::PandOp>(MLIRContext *context) {
  return buildPandCallee(context);
}

template <>
StringRef getPredicateMaskCallee<pto::PorOp>(MLIRContext *context) {
  return buildPorCallee(context);
}

template <>
StringRef getPredicateMaskCallee<pto::PxorOp>(MLIRContext *context) {
  return buildPxorCallee(context);
}

template <typename PackOp>
static StringRef getPredicatePackCallee(MLIRContext *context);

template <>
StringRef getPredicatePackCallee<pto::PpackOp>(MLIRContext *context) {
  return buildPpackCallee(context);
}

template <>
StringRef getPredicatePackCallee<pto::PunpackOp>(MLIRContext *context) {
  return buildPunpackCallee(context);
}

template <typename PltOp>
static StringRef buildPltCallee(MLIRContext *context);

template <>
StringRef buildPltCallee<pto::PltB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.plt.b8.v300").getValue();
}

template <>
StringRef buildPltCallee<pto::PltB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.plt.b16.v300").getValue();
}

template <>
StringRef buildPltCallee<pto::PltB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.plt.b32.v300").getValue();
}

template <typename PsetOp>
static StringRef buildPsetCallee(MLIRContext *context);

template <>
StringRef buildPsetCallee<pto::PsetB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pset.b8").getValue();
}

template <>
StringRef buildPsetCallee<pto::PsetB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pset.b16").getValue();
}

template <>
StringRef buildPsetCallee<pto::PsetB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pset.b32").getValue();
}

template <typename PgeOp>
static StringRef buildPgeCallee(MLIRContext *context);

template <>
StringRef buildPgeCallee<pto::PgeB8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pge.b8").getValue();
}

template <>
StringRef buildPgeCallee<pto::PgeB16Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pge.b16").getValue();
}

template <>
StringRef buildPgeCallee<pto::PgeB32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.pge.b32").getValue();
}

static FailureOr<StringRef> buildVldsCallee(MLIRContext *context, Type resultType) {
  std::string vec = getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
    return failure();
  return StringAttr::get(context, "llvm.hivm.vldsx1.v" + std::to_string(*lanes) +
                                      vec)
      .getValue();
}

static FailureOr<StringRef> buildVldsx2Callee(MLIRContext *context,
                                              Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vldsx2", "");
}

static StringRef buildVsldbCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vsldb").getValue();
}

static FailureOr<StringRef> buildVstsCallee(MLIRContext *context, Type valueType) {
  std::string vec = getElementTypeFragment(getElementTypeFromVectorLike(valueType));
  auto lanes = getElementCountFromVectorLike(valueType);
  if (vec.empty() || !lanes)
    return failure();
  return StringAttr::get(context, "llvm.hivm.vstsx1.v" + std::to_string(*lanes) +
                                      vec)
      .getValue();
}

static FailureOr<StringRef> buildVstsx2Callee(MLIRContext *context, Type valueType) {
  return buildLaneTypedCallee(context, valueType, "vstsx2", "");
}

static StringRef buildVsstbCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.vsstb").getValue();
}

static FailureOr<StringRef> buildVgather2Callee(MLIRContext *context,
                                                Type resultType) {
  std::string vec =
      getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
    return failure();
  return StringAttr::get(context, "llvm.hivm.vgather2.v300.v" +
                                      std::to_string(*lanes) + vec)
      .getValue();
}

static FailureOr<StringRef> buildVgather2BcCallee(MLIRContext *context,
                                                  Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vgather2.bc", "");
}

static FailureOr<StringRef> buildVgatherbCallee(MLIRContext *context,
                                                Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vgatherb.v310", "");
}

static FailureOr<StringRef> buildVscatterCallee(MLIRContext *context,
                                                Type valueType) {
  return buildLaneTypedCallee(context, valueType, "vscatter", ".v300");
}

static FailureOr<StringRef> buildVpreluCallee(MLIRContext *context,
                                              Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vprelu", ".x");
}

static FailureOr<StringRef> buildVaxpyCallee(MLIRContext *context,
                                             Type resultType) {
  return buildLaneTypedCallee(context, resultType, "vaxpy", ".m");
}

static FailureOr<StringRef> buildVciCallee(MLIRContext *context, Type resultType) {
  std::string vec =
      getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
    return failure();
  if (vec == "f16" || vec == "f32")
    return StringAttr::get(context, "llvm.hivm.vci.v" + std::to_string(*lanes) +
                                        vec + "." + vec)
        .getValue();
  return StringAttr::get(context,
                         "llvm.hivm.vci.v" + std::to_string(*lanes) + vec)
      .getValue();
}

static FailureOr<StringRef> buildVtrcCallee(MLIRContext *context, Type resultType) {
  std::string vec =
      getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes)
    return failure();
  return StringAttr::get(context, "llvm.hivm.vtrc." + vec + ".x").getValue();
}

static FailureOr<StringRef> buildVexpdiffCallee(MLIRContext *context,
                                                Type inputType,
                                                Type resultType) {
  std::string srcVec =
      getElementTypeFragment(getElementTypeFromVectorLike(inputType));
  auto srcLanes = getElementCountFromVectorLike(inputType);
  std::string dstElem =
      getElementTypeFragment(getElementTypeFromVectorLike(resultType));
  if (srcVec.empty() || dstElem.empty() || !srcLanes)
    return failure();
  return StringAttr::get(context, "llvm.hivm.vexpdif.v" +
                                      std::to_string(*srcLanes) + srcVec +
                                      dstElem)
      .getValue();
}

static FailureOr<StringRef> buildVbitsortCallee(MLIRContext *context,
                                                pto::VbitsortOp op) {
  Type sourceElemType = cast<pto::PtrType>(op.getSource().getType()).getElementType();
  if (sourceElemType.isF16())
    return StringAttr::get(context, "llvm.hivm.VBS32.V300.f16").getValue();
  if (sourceElemType.isF32())
    return StringAttr::get(context, "llvm.hivm.VBS32.V300.f32").getValue();
  return failure();
}

static FailureOr<VcvtContract> buildVcvtContract(pto::VcvtOp op) {
  Type inputElemType = getElementTypeFromVectorLike(op.getInput().getType());
  Type resultElemType = getElementTypeFromVectorLike(op.getResult().getType());
  if (!inputElemType || !resultElemType)
    return failure();
  auto contract = lookupVcvtContract(classifyVcvtElemType(inputElemType),
                                     classifyVcvtElemType(resultElemType));
  if (!contract)
    return failure();
  return *contract;
}

template <typename LoopOp>
static StringRef buildSetLoopCallee(MLIRContext *context);

template <>
StringRef buildSetLoopCallee<pto::SetLoop2StrideOutToUbOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP2.STRIDE.OUTTOUB")
      .getValue();
}

template <>
StringRef buildSetLoopCallee<pto::SetLoop1StrideOutToUbOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP1.STRIDE.OUTTOUB")
      .getValue();
}

template <>
StringRef buildSetLoopCallee<pto::SetLoopSizeOutToUbOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP.SIZE.OUTTOUB")
      .getValue();
}

template <>
StringRef buildSetLoopCallee<pto::SetLoop2StrideUbToOutOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP2.STRIDE.UBTOOUT")
      .getValue();
}

template <>
StringRef buildSetLoopCallee<pto::SetLoop1StrideUbToOutOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP1.STRIDE.UBTOOUT")
      .getValue();
}

template <>
StringRef buildSetLoopCallee<pto::SetLoopSizeUbToOutOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP.SIZE.UBTOOUT")
      .getValue();
}

template <typename SyncOp>
static StringRef buildSyncCallee(MLIRContext *context);

template <>
StringRef buildSyncCallee<pto::SetFlagOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.FLAG.IMM").getValue();
}

template <>
StringRef buildSyncCallee<pto::WaitFlagOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.WAIT.FLAG.IMM").getValue();
}

template <>
StringRef buildSyncCallee<pto::BarrierOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.BARRIER").getValue();
}

template <>
StringRef buildSyncCallee<pto::GetBufOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.BUFI.mode").getValue();
}

template <>
StringRef buildSyncCallee<pto::RlsBufOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.RLS.BUFI.mode").getValue();
}

template <typename QueryOp>
static StringRef buildRuntimeQueryCallee(MLIRContext *context);

template <>
StringRef buildRuntimeQueryCallee<pto::GetBlockIdxOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.BLOCK.IDX").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetSubBlockIdxOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.SUBBLOCKID").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetBlockNumOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.BLOCK.NUM").getValue();
}

template <>
StringRef buildRuntimeQueryCallee<pto::GetSubBlockNumOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.SUBBLOCKDIM").getValue();
}

static LogicalResult
materializeDecls(ModuleOp module, ArrayRef<PlannedDecl> plannedDecls,
                 llvm::raw_ostream &diagOS) {
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(&module.getBodyRegion().front());
  for (const PlannedDecl &decl : plannedDecls) {
    if (func::FuncOp existing = module.lookupSymbol<func::FuncOp>(decl.name)) {
      if (existing.getFunctionType() != decl.type) {
        diagOS << "VPTO LLVM emission failed: conflicting declaration for "
               << decl.name << "\n";
        return failure();
      }
      continue;
    }
    auto func =
        builder.create<func::FuncOp>(module.getLoc(), decl.name, decl.type);
    func.setPrivate();
  }
  return success();
}

template <typename UnaryOp>
class LowerUnaryMaskedOpPattern final : public OpConversionPattern<UnaryOp> {
public:
  explicit LowerUnaryMaskedOpPattern(TypeConverter &typeConverter,
                                     MLIRContext *context,
                                     LoweringState &state)
      : OpConversionPattern<UnaryOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(UnaryOp op, typename UnaryOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef stem = getUnaryMaskedStem<UnaryOp>();
    FailureOr<StringRef> calleeName =
        buildLaneTypedCallee(op.getContext(), op.getResult().getType(), stem, ".x");
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported unary VPTO signature");

    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert unary result type");

    Value input = adaptor.getOperands()[0];
    Value mask = adaptor.getOperands()[1];
    Type expectedMaskType =
        this->getTypeConverter()->convertType(op->getOperand(1).getType());
    if (!input || !mask || input.getType() != resultType ||
        mask.getType() != expectedMaskType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted unary VPTO operand types");
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{resultType},
                                              ValueRange{input, mask});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVsqzOpPattern final : public OpConversionPattern<pto::VsqzOp> {
public:
  explicit LowerVsqzOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::VsqzOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VsqzOp op, pto::VsqzOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName =
        buildVsqzCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vsqz VPTO signature");

    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType)
      return rewriter.notifyMatchFailure(op, "failed to convert vsqz types");

    Value input = adaptor.getInput();
    Value mask = adaptor.getMask();
    if (!input || !mask || input.getType() != resultType ||
        mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vsqz operand types");
    }

    Value storeHint =
        getI32Constant(rewriter, op.getLoc(), determineVsqzStoreHint(op));
    auto funcType = rewriter.getFunctionType(
        TypeRange{resultType, maskType, storeHint.getType()}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{input, mask, storeHint});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVusqzOpPattern final : public OpConversionPattern<pto::VusqzOp> {
public:
  explicit LowerVusqzOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VusqzOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VusqzOp op, pto::VusqzOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName =
        buildVusqzCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vusqz VPTO signature");

    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType)
      return rewriter.notifyMatchFailure(op, "failed to convert vusqz types");

    Value src = adaptor.getSrc();
    Value mask = adaptor.getMask();
    if (!src || !mask || src.getType() != resultType || mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vusqz operand types");
    }

    auto funcType =
        rewriter.getFunctionType(TypeRange{resultType, maskType}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType}, ValueRange{src, mask});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVmulaOpPattern final : public OpConversionPattern<pto::VmulaOp> {
public:
  explicit LowerVmulaOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                               LoweringState &state)
      : OpConversionPattern<pto::VmulaOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VmulaOp op, pto::VmulaOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName =
        buildVmulaCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vmula VPTO signature");

    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType)
      return rewriter.notifyMatchFailure(op, "failed to convert vmula types");

    Value acc = adaptor.getAcc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value mask = adaptor.getMask();
    if (!acc || !lhs || !rhs || !mask || acc.getType() != resultType ||
        lhs.getType() != resultType || rhs.getType() != resultType ||
        mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vmula operand types");
    }

    auto funcType = rewriter.getFunctionType(
        TypeRange{resultType, resultType, resultType, maskType},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{acc, lhs, rhs, mask});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVmullOpPattern final : public OpConversionPattern<pto::VmullOp> {
public:
  explicit LowerVmullOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                               LoweringState &state)
      : OpConversionPattern<pto::VmullOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VmullOp op, pto::VmullOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName =
        buildVmullCallee(op.getContext(), op.getLow().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vmull VPTO signature");

    Type inputType = this->getTypeConverter()->convertType(op.getLhs().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    SmallVector<Type> resultTypes;
    if (!inputType || !maskType ||
        failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert vmull types");
    }
    if (resultTypes.size() != 2 || resultTypes[0] != resultTypes[1])
      return rewriter.notifyMatchFailure(op, "unexpected converted vmull results");

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value mask = adaptor.getMask();
    if (!lhs || !rhs || !mask || lhs.getType() != inputType ||
        rhs.getType() != inputType || mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vmull operand types");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{inputType, inputType, maskType},
                                             resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, resultTypes,
                                              ValueRange{lhs, rhs, mask});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename BinaryOp>
class LowerBinaryMaskedOpPattern final : public OpConversionPattern<BinaryOp> {
public:
  explicit LowerBinaryMaskedOpPattern(TypeConverter &typeConverter,
                                      MLIRContext *context,
                                      LoweringState &state)
      : OpConversionPattern<BinaryOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(BinaryOp op, typename BinaryOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef stem = getBinaryMaskedStem<BinaryOp>();
    FailureOr<StringRef> calleeName =
        buildLaneTypedCallee(op.getContext(), op.getResult().getType(), stem, ".x");
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported binary VPTO signature");

    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert binary result type");

    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];
    Value mask = adaptor.getOperands()[2];
    Type expectedMaskType =
        this->getTypeConverter()->convertType(op->getOperand(2).getType());
    if (!lhs || !rhs || !mask || lhs.getType() != resultType ||
        rhs.getType() != resultType || mask.getType() != expectedMaskType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted binary VPTO operand types");
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{resultType},
                                              ValueRange{lhs, rhs, mask});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename CarryOp>
class LowerCarryBinaryOpPattern final : public OpConversionPattern<CarryOp> {
public:
  explicit LowerCarryBinaryOpPattern(TypeConverter &typeConverter,
                                     MLIRContext *context, LoweringState &state)
      : OpConversionPattern<CarryOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(CarryOp op, typename CarryOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef stem = getCarryBinaryStem<CarryOp>();
    FailureOr<StringRef> calleeName =
        buildCarryBinaryCallee(op.getContext(), op.getResult().getType(), stem);
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported carry VPTO signature");

    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    Type carryType =
        this->getTypeConverter()->convertType(op->getResult(1).getType());
    if (!resultType || !carryType)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert carry result types");

    SmallVector<Value> callArgs;
    callArgs.append(adaptor.getOperands().begin(), adaptor.getOperands().end());
    const size_t expectedArgCount = hasCarryInput<CarryOp>() ? 4 : 3;
    if (callArgs.size() != expectedArgCount || callArgs[0].getType() != resultType ||
        callArgs[1].getType() != resultType || callArgs.back().getType() != carryType)
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted carry operand types");
    if constexpr (hasCarryInput<CarryOp>()) {
      if (callArgs[2].getType() != carryType)
        return rewriter.notifyMatchFailure(
            op, "unexpected converted carry input operand type");
    }

    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType, carryType}, callArgs);
    state.plannedDecls.push_back(
        PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename CopyOp>
class LowerCopyOpPattern final : public OpConversionPattern<CopyOp> {
public:
  explicit LowerCopyOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<CopyOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(CopyOp op, typename CopyOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName = failure();
    if constexpr (std::is_same_v<CopyOp, pto::CopyGmToUbufOp>)
      calleeName = buildCopyGmToUbCallee(op.getContext(), op);
    else
      calleeName = buildCopyUbToGmCallee(op.getContext());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported copy VPTO signature");

    auto llvmSourceType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getOperands()[0].getType());
    auto llvmDestType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getOperands()[1].getType());
    if (!llvmSourceType || !llvmDestType)
      return rewriter.notifyMatchFailure(op, "expected LLVM pointer copy operands");

    FailureOr<Value> config0 = failure();
    FailureOr<Value> config1 = failure();
    if constexpr (std::is_same_v<CopyOp, pto::CopyGmToUbufOp>) {
      config0 = packCopyGmToUbConfig0(op, adaptor.getOperands());
      config1 = packCopyGmToUbConfig1(op, adaptor.getOperands());
    } else {
      config0 = packCopyUbToGmConfig0(op, adaptor.getOperands());
      config1 = packCopyUbToGmConfig1(op, adaptor.getOperands());
    }
    if (failed(config0) || failed(config1))
      return rewriter.notifyMatchFailure(op, "failed to materialize copy config");

    SmallVector<Value> args{adaptor.getOperands()[1], adaptor.getOperands()[0],
                            *config0, *config1};
    auto funcType = rewriter.getFunctionType(
        TypeRange{llvmDestType, llvmSourceType, rewriter.getI64Type(),
                  rewriter.getI64Type()},
        TypeRange{});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    (void)call;
    return success();
  }

private:
  LoweringState &state;
};

template <typename VecScalarOp>
class LowerVecScalarMaskedOpPattern final
    : public OpConversionPattern<VecScalarOp> {
public:
  explicit LowerVecScalarMaskedOpPattern(TypeConverter &typeConverter,
                                         MLIRContext *context,
                                         LoweringState &state)
      : OpConversionPattern<VecScalarOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(VecScalarOp op, typename VecScalarOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef stem = getVecScalarMaskedStem<VecScalarOp>();
    FailureOr<StringRef> calleeName =
        buildLaneTypedCallee(op.getContext(), op.getResult().getType(), stem, ".x");
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op,
                                         "unsupported vec-scalar VPTO signature");

    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert vec-scalar result type");

    Value input = adaptor.getOperands()[0];
    Value scalar = adaptor.getOperands()[1];
    Value mask = adaptor.getOperands()[2];
    Type expectedMaskType =
        this->getTypeConverter()->convertType(op->getOperand(2).getType());
    if (!input || !scalar || !mask || input.getType() != resultType ||
        mask.getType() != expectedMaskType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted vec-scalar VPTO operand types");
    }

    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{input, scalar, mask});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename ReductionOp>
class LowerReductionUnaryOpPattern final
    : public OpConversionPattern<ReductionOp> {
public:
  explicit LowerReductionUnaryOpPattern(TypeConverter &typeConverter,
                                        MLIRContext *context,
                                        LoweringState &state)
      : OpConversionPattern<ReductionOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(ReductionOp op, typename ReductionOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef stem = getReductionUnaryStem<ReductionOp>();
    FailureOr<StringRef> calleeName =
        buildLaneTypedCallee(op.getContext(), op.getResult().getType(), stem, ".x");
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op,
                                         "unsupported reduction VPTO signature");

    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert reduction result type");
    }

    Value input = adaptor.getInput();
    Value mask = adaptor.getMask();
    if (!input || !mask || input.getType() != resultType ||
        mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted reduction operand types");
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{resultType},
                                              ValueRange{input, mask});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVselOpPattern final : public OpConversionPattern<pto::VselOp> {
public:
  explicit LowerVselOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::VselOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VselOp op, pto::VselOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName =
        buildVselCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vsel VPTO signature");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType)
      return rewriter.notifyMatchFailure(op, "failed to convert vsel result type");

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value mask = adaptor.getMask();
    if (!src0 || !src1 || !mask || src0.getType() != resultType ||
        src1.getType() != resultType || mask.getType() != maskType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vsel operand types");
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{resultType},
                                              ValueRange{src0, src1, mask});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVdupOpPattern final : public OpConversionPattern<pto::VdupOp> {
public:
  explicit LowerVdupOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::VdupOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VdupOp op, pto::VdupOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName = buildVdupCallee(op.getContext(), op);
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vdup VPTO signature");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType = this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType)
      return rewriter.notifyMatchFailure(op, "failed to convert vdup result type");

    Value mask = adaptor.getMask();
    if (!mask || mask.getType() != maskType)
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vdup mask type");

    SmallVector<Value> callArgs;
    bool vectorInput = isa<VectorType, pto::VRegType>(op.getInput().getType());
    if (vectorInput) {
      Value input = adaptor.getInput();
      if (!input || input.getType() != resultType) {
        return rewriter.notifyMatchFailure(
            op, "vector-input vdup requires matching result type");
      }
      callArgs.push_back(input);
    } else {
      Type scalarType = getElementTypeFromVectorLike(op.getResult().getType());
      if (!scalarType || op.getInput().getType() != scalarType) {
        return rewriter.notifyMatchFailure(op,
                                           "unexpected scalar-input vdup type");
      }
      FailureOr<Value> normalizedScalar =
          normalizeVdupScalarOperand(rewriter, op.getLoc(), op);
      if (failed(normalizedScalar))
        return rewriter.notifyMatchFailure(op,
                                           "failed to normalize scalar vdup input");
      callArgs.push_back(*normalizedScalar);
    }

    callArgs.push_back(mask);
    callArgs.push_back(getI32Constant(rewriter, op.getLoc(), 1));

    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType}, callArgs);
    state.plannedDecls.push_back(
        PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVbrOpPattern final : public OpConversionPattern<pto::VbrOp> {
public:
  explicit LowerVbrOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                             LoweringState &state)
      : OpConversionPattern<pto::VbrOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VbrOp op, pto::VbrOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName =
        buildVbrCallee(op.getContext(), op.getValue().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vbr VPTO signature");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert vbr result type");

    Value scalar = adaptor.getValue();
    if (!scalar || scalar.getType() != op.getValue().getType())
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vbr operand type");

    auto funcType = rewriter.getFunctionType(TypeRange{scalar.getType()},
                                             TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{resultType},
                                              ValueRange{scalar});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVselrOpPattern final : public OpConversionPattern<pto::VselrOp> {
public:
  explicit LowerVselrOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                               LoweringState &state)
      : OpConversionPattern<pto::VselrOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VselrOp op, pto::VselrOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName =
        buildVselrCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vselr VPTO signature");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    auto resultVectorType = dyn_cast<VectorType>(resultType);
    if (!resultVectorType)
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vselr result type");

    Type intrinsicResultType = resultType;
    if (auto floatType = dyn_cast<FloatType>(resultVectorType.getElementType());
        floatType && floatType.isF32()) {
      intrinsicResultType = VectorType::get(
          resultVectorType.getShape(), rewriter.getI32Type(),
          resultVectorType.getScalableDims());
    }

    Type indexType = this->getTypeConverter()->convertType(op.getSrc1().getType());
    if (!indexType)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert vselr index type");

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    if (!src0 || !src1 || src1.getType() != indexType)
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vselr operand types");

    if (src0.getType() != intrinsicResultType) {
      if (src0.getType() != resultType)
        return rewriter.notifyMatchFailure(op,
                                           "unexpected converted vselr source type");
      src0 = rewriter.create<LLVM::BitcastOp>(op.getLoc(), intrinsicResultType, src0);
    }

    auto funcType = rewriter.getFunctionType(
        TypeRange{intrinsicResultType, indexType}, TypeRange{intrinsicResultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{intrinsicResultType},
        ValueRange{src0, src1});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});

    Value result = call.getResult(0);
    if (intrinsicResultType != resultType)
      result = rewriter.create<LLVM::BitcastOp>(op.getLoc(), resultType, result);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  LoweringState &state;
};

class LowerPnotOpPattern final : public OpConversionPattern<pto::PnotOp> {
public:
  explicit LowerPnotOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::PnotOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::PnotOp op, pto::PnotOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert pnot result type");

    Value input = adaptor.getInput();
    Value mask = adaptor.getMask();
    if (!input || !mask || input.getType() != resultType ||
        mask.getType() != resultType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted pnot operand types");
    }

    StringRef calleeName = getPredicateMaskCallee<pto::PnotOp>(op.getContext());
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              TypeRange{resultType},
                                              ValueRange{input, mask});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName.str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename InterleaveOp>
class LowerInterleaveOpPattern final
    : public OpConversionPattern<InterleaveOp> {
public:
  explicit LowerInterleaveOpPattern(TypeConverter &typeConverter,
                                    MLIRContext *context, LoweringState &state)
      : OpConversionPattern<InterleaveOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(InterleaveOp op, typename InterleaveOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef stem = std::is_same_v<InterleaveOp, pto::VintlvOp> ? "vintlv" : "vdintlv";
    FailureOr<StringRef> calleeName =
        buildInterleaveCallee(op.getContext(), op.getLow().getType(), stem);
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op,
                                         "unsupported interleave VPTO signature");

    Type lowType = this->getTypeConverter()->convertType(op.getLow().getType());
    Type highType = this->getTypeConverter()->convertType(op.getHigh().getType());
    if (!lowType || !highType || lowType != highType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert interleave result types");
    }

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (!lhs || !rhs || lhs.getType() != lowType || rhs.getType() != lowType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted interleave operand types");
    }

    auto funcType = rewriter.getFunctionType(TypeRange{lowType, lowType},
                                             TypeRange{lowType, highType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{lowType, highType}, ValueRange{lhs, rhs});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename PackOp>
class LowerPredicatePackOpPattern final : public OpConversionPattern<PackOp> {
public:
  explicit LowerPredicatePackOpPattern(TypeConverter &typeConverter,
                                       MLIRContext *context,
                                       LoweringState &state)
      : OpConversionPattern<PackOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(PackOp op, typename PackOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert predicate-pack result type");

    auto part = parseHiLoPartImmediate(op.getPart());
    if (!part)
      return rewriter.notifyMatchFailure(
          op, "unsupported predicate-pack part immediate");

    Value input = adaptor.getInput();
    if (!input || input.getType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "unexpected converted predicate-pack operand type");

    Value partValue = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(*part));
    StringRef calleeName = getPredicatePackCallee<PackOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(
        TypeRange{resultType, rewriter.getI32Type()}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), calleeName, TypeRange{resultType}, ValueRange{input, partValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename UnpackOp>
class LowerUnpackOpPattern final : public OpConversionPattern<UnpackOp> {
public:
  explicit LowerUnpackOpPattern(TypeConverter &typeConverter,
                                MLIRContext *context, LoweringState &state)
      : OpConversionPattern<UnpackOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(UnpackOp op, typename UnpackOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef stem = std::is_same_v<UnpackOp, pto::VsunpackOp> ? "vsunpack"
                                                               : "vzunpack";
    FailureOr<StringRef> calleeName = buildUnpackCallee(
        op.getContext(), op.getSrc().getType(), op.getResult().getType(), stem);
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op,
                                         "unsupported unpack VPTO signature");

    Type srcType = this->getTypeConverter()->convertType(op.getSrc().getType());
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!srcType || !resultType)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert unpack types");

    Value src = adaptor.getSrc();
    if (!src || src.getType() != srcType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted unpack source type");
    }

    Value part = castIntegerLikeTo(op, adaptor.getPart(), rewriter.getI32Type());
    if (!part)
      return rewriter.notifyMatchFailure(op, "failed to materialize unpack part");

    auto funcType = rewriter.getFunctionType(TypeRange{srcType, part.getType()},
                                             TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType}, ValueRange{src, part});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVpackOpPattern final : public OpConversionPattern<pto::VpackOp> {
public:
  explicit LowerVpackOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VpackOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VpackOp op, pto::VpackOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName =
        buildVpackCallee(op.getContext(), op.getSrc().getType(),
                         op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vpack VPTO signature");

    Type srcType = this->getTypeConverter()->convertType(op.getSrc().getType());
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!srcType || !resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert vpack types");

    auto partImm = parseHiLoPartImmediate(op.getPart());
    if (!partImm)
      return rewriter.notifyMatchFailure(op, "unsupported vpack part immediate");

    Value src = adaptor.getSrc();
    if (!src || src.getType() != srcType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted vpack source type");
    }

    Value part = getI32Constant(rewriter, op.getLoc(), *partImm);
    auto funcType = rewriter.getFunctionType(TypeRange{srcType, part.getType()},
                                             TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType}, ValueRange{src, part});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename PredicateMaskOp>
class LowerPredicateMaskBinaryOpPattern final
    : public OpConversionPattern<PredicateMaskOp> {
public:
  explicit LowerPredicateMaskBinaryOpPattern(TypeConverter &typeConverter,
                                             MLIRContext *context,
                                             LoweringState &state)
      : OpConversionPattern<PredicateMaskOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(PredicateMaskOp op, typename PredicateMaskOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert predicate-mask result type");

    Value src0 = adaptor.getSrc0();
    Value src1 = adaptor.getSrc1();
    Value mask = adaptor.getMask();
    if (!src0 || !src1 || !mask || src0.getType() != resultType ||
        src1.getType() != resultType || mask.getType() != resultType) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted predicate-mask operand types");
    }

    StringRef calleeName = getPredicateMaskCallee<PredicateMaskOp>(op.getContext());
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              TypeRange{resultType},
                                              ValueRange{src0, src1, mask});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName.str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename ReorderOp>
class LowerPredicatePairReorderOpPattern final
    : public OpConversionPattern<ReorderOp> {
public:
  explicit LowerPredicatePairReorderOpPattern(TypeConverter &typeConverter,
                                              MLIRContext *context,
                                              LoweringState &state)
      : OpConversionPattern<ReorderOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(ReorderOp op, typename ReorderOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(
          op, "failed to convert predicate-pair-reorder result types");
    if (resultTypes.size() != 2 || resultTypes[0] != resultTypes[1])
      return rewriter.notifyMatchFailure(
          op, "unexpected predicate-pair-reorder converted result types");

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (!lhs || !rhs || lhs.getType() != resultTypes[0] ||
        rhs.getType() != resultTypes[0]) {
      return rewriter.notifyMatchFailure(
          op, "unexpected converted predicate-pair-reorder operand types");
    }

    StringRef calleeName =
        buildPredicatePairReorderCallee<ReorderOp>(op.getContext());
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName, resultTypes,
                                              ValueRange{lhs, rhs});
    state.plannedDecls.push_back(
        PlannedDecl{calleeName.str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename CmpOp>
class LowerCmpOpPattern final : public OpConversionPattern<CmpOp> {
public:
  explicit LowerCmpOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                             LoweringState &state)
      : OpConversionPattern<CmpOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(CmpOp op, typename CmpOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    constexpr bool isScalarCompare = std::is_same_v<CmpOp, pto::VcmpsOp>;
    Type inputType = Type();
    if constexpr (isScalarCompare)
      inputType = op.getSrc().getType();
    else
      inputType = op.getSrc0().getType();
    FailureOr<StringRef> calleeName =
        buildVcmpCallee(op.getContext(), inputType, op.getCmpMode(),
                        isScalarCompare);
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op,
                                         "unsupported compare VPTO signature");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type maskType =
        this->getTypeConverter()->convertType(op.getMask().getType());
    if (!resultType || !maskType)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert compare result type");
    if (resultType != maskType)
      return rewriter.notifyMatchFailure(op,
                                         "unexpected compare mask conversion");

    SmallVector<Value> callArgs;
    callArgs.append(adaptor.getOperands().begin(), adaptor.getOperands().end());
    if constexpr (isScalarCompare) {
      if (callArgs.size() != 3 || !callArgs[0] || !callArgs[1] || !callArgs[2] ||
          callArgs[2].getType() != maskType) {
        return rewriter.notifyMatchFailure(
            op, "unexpected converted scalar-compare operand types");
      }
    } else {
      if (callArgs.size() != 3 || !callArgs[0] || !callArgs[1] || !callArgs[2] ||
          callArgs[0].getType() != callArgs[1].getType() ||
          callArgs[2].getType() != maskType) {
        return rewriter.notifyMatchFailure(
            op, "unexpected converted compare operand types");
      }
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              TypeRange{resultType}, callArgs);
    state.plannedDecls.push_back(
        PlannedDecl{calleeName->str(), call.getCalleeType()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename PltOp>
class LowerPltOpPattern final : public OpConversionPattern<PltOp> {
public:
  explicit LowerPltOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                             LoweringState &state)
      : OpConversionPattern<PltOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(PltOp op, typename PltOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value laneCount = castIntegerLikeTo(op, adaptor.getScalar(), rewriter.getI32Type());
    if (!laneCount)
      return rewriter.notifyMatchFailure(op, "failed to materialize plt lane count");

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "failed to convert plt result types");

    StringRef calleeName = buildPltCallee<PltOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{rewriter.getI32Type()},
                                             resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              resultTypes, ValueRange{laneCount});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename PsetOp>
class LowerPsetOpPattern final : public OpConversionPattern<PsetOp> {
public:
  explicit LowerPsetOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<PsetOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(PsetOp op, typename PsetOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto pattern = parsePredicatePatternImmediate(op.getPattern());
    if (!pattern)
      return rewriter.notifyMatchFailure(op, "unsupported pset pattern");

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "failed to convert pset result types");

    StringRef calleeName = buildPsetCallee<PsetOp>(op.getContext());
    Value patternValue = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(*pattern));
    auto funcType = rewriter.getFunctionType(TypeRange{rewriter.getI32Type()},
                                             resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              resultTypes, ValueRange{patternValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename PgeOp>
class LowerPgeOpPattern final : public OpConversionPattern<PgeOp> {
public:
  explicit LowerPgeOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                             LoweringState &state)
      : OpConversionPattern<PgeOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(PgeOp op, typename PgeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto pattern = parsePredicatePatternImmediate(op.getPattern());
    if (!pattern)
      return rewriter.notifyMatchFailure(op, "unsupported pge pattern");

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "failed to convert pge result types");

    StringRef calleeName = buildPgeCallee<PgeOp>(op.getContext());
    Value patternValue = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(*pattern));
    Value zero = rewriter.create<arith::ConstantOp>(op.getLoc(),
                                                    rewriter.getI32IntegerAttr(0));
    auto funcType = rewriter.getFunctionType(
        TypeRange{rewriter.getI32Type(), rewriter.getI32Type()}, resultTypes);
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), calleeName, resultTypes,
                                      ValueRange{patternValue, zero});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVldsOpPattern final : public OpConversionPattern<pto::VldsOp> {
public:
  explicit LowerVldsOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::VldsOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VldsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elementType = getElementTypeFromVectorLike(op.getResult().getType());
    if (!elementType)
      return rewriter.notifyMatchFailure(op, "unsupported vlds element type");
    auto offsetBytes = convertElementOffsetToBytes(op, adaptor.getOffset(), elementType);
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    auto dist =
        parseLoadDistImmediate(op.getDist().value_or("NORM"), elementType);
    if (failed(offsetBytes) || !basePtr || !dist)
      return rewriter.notifyMatchFailure(op, "failed to materialize vlds operands");

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "failed to convert vlds result types");

    FailureOr<StringRef> calleeName = buildVldsCallee(op.getContext(),
                                                      op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vlds signature");

    Value distValue = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(*dist));
    Value zero = rewriter.create<arith::ConstantOp>(op.getLoc(),
                                                    rewriter.getI32IntegerAttr(0));
    SmallVector<Value> args{adaptor.getSource(), *offsetBytes, distValue, zero};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSource().getType(), rewriter.getI32Type(),
                  rewriter.getI32Type(), rewriter.getI32Type()},
        resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              resultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVldsPostOpPattern final
    : public OpConversionPattern<pto::VldsPostOp> {
public:
  explicit LowerVldsPostOpPattern(TypeConverter &typeConverter,
                                  MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VldsPostOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::VldsPostOp op, pto::VldsPostOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elementType = getElementTypeFromVectorLike(op.getResult().getType());
    if (!elementType)
      return rewriter.notifyMatchFailure(op, "unsupported vlds_post element type");

    auto offsetBytes = convertElementOffsetToBytes(op, adaptor.getOffset(), elementType);
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    auto dist =
        parseLoadDistImmediate(op.getDist().value_or("NORM"), elementType);
    if (failed(offsetBytes) || !basePtr || !dist) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize vlds_post operands");
    }

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    Type updatedSourceType =
        this->getTypeConverter()->convertType(op.getUpdatedSource().getType());
    if (!resultType || !updatedSourceType || updatedSourceType != adaptor.getSource().getType()) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert vlds_post result types");
    }

    FailureOr<StringRef> calleeName =
        buildVldsPostCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vlds_post signature");

    Value distValue = getI32Constant(rewriter, op.getLoc(), *dist);
    Value postValue = getI32Constant(rewriter, op.getLoc(), 1);
    SmallVector<Value> args{adaptor.getSource(), *offsetBytes, distValue, postValue};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSource().getType(), (*offsetBytes).getType(),
                  distValue.getType(), postValue.getType()},
        TypeRange{resultType, updatedSourceType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType, updatedSourceType}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVldsx2OpPattern final : public OpConversionPattern<pto::Vldsx2Op> {
public:
  explicit LowerVldsx2OpPattern(TypeConverter &typeConverter,
                                MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::Vldsx2Op>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::Vldsx2Op op, pto::Vldsx2Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elementType = getElementTypeFromVectorLike(op.getLow().getType());
    if (!elementType)
      return rewriter.notifyMatchFailure(op, "unsupported vldsx2 element type");

    auto offsetBytes =
        convertElementOffsetToBytes(op, adaptor.getOffset(), elementType);
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    auto dist = parseLoadX2DistImmediate(op.getDist(), elementType);
    if (failed(offsetBytes) || !basePtr || !dist) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize vldsx2 operands");
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes)) ||
        resultTypes.size() != 2) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert vldsx2 result types");
    }

    FailureOr<StringRef> calleeName =
        buildVldsx2Callee(op.getContext(), op.getLow().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vldsx2 signature");

    Value distValue = getI32Constant(rewriter, op.getLoc(), *dist);
    Value zeroValue = getI32Constant(rewriter, op.getLoc(), 0);
    SmallVector<Value> args{adaptor.getSource(), *offsetBytes, distValue,
                            zeroValue};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSource().getType(), (*offsetBytes).getType(),
                  distValue.getType(), zeroValue.getType()},
        resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName,
                                              resultTypes, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVsldbOpPattern final : public OpConversionPattern<pto::VsldbOp> {
public:
  explicit LowerVsldbOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VsldbOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VsldbOp op, pto::VsldbOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    Value packedStride =
        packBlockRepeatStride(op, adaptor.getBlockStride(), adaptor.getRepeatStride());
    if (!basePtr || !packedStride)
      return rewriter.notifyMatchFailure(op, "failed to materialize vsldb operands");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert vsldb result type");

    StringRef calleeName = buildVsldbCallee(op.getContext());
    Value zeroValue = getI32Constant(rewriter, op.getLoc(), 0);
    SmallVector<Value> args{adaptor.getSource(), packedStride, zeroValue,
                            adaptor.getMask()};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSource().getType(), packedStride.getType(),
                  zeroValue.getType(), adaptor.getMask().getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              TypeRange{resultType}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerInitAlignOpPattern final
    : public OpConversionPattern<pto::InitAlignOp> {
public:
  explicit LowerInitAlignOpPattern(TypeConverter &typeConverter,
                                   MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::InitAlignOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::InitAlignOp op, pto::InitAlignOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert init_align result type");

    StringRef calleeName = buildInitAlignCallee(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{}, TypeRange{resultType});
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{resultType});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVldasOpPattern final : public OpConversionPattern<pto::VldasOp> {
public:
  explicit LowerVldasOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VldasOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VldasOp op, pto::VldasOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!sourceType || !resultType)
      return rewriter.notifyMatchFailure(op,
                                         "expected converted vldas operand/result types");

    StringRef calleeName = buildVldasCallee(op.getContext());
    auto funcType =
        rewriter.getFunctionType(TypeRange{adaptor.getSource().getType()},
                                 TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              TypeRange{resultType},
                                              ValueRange{adaptor.getSource()});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVldusOpPattern final : public OpConversionPattern<pto::VldusOp> {
public:
  explicit LowerVldusOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VldusOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VldusOp op, pto::VldusOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    SmallVector<Type> resultTypes;
    if (!sourceType ||
        failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)) ||
        resultTypes.size() != 2 || adaptor.getAlign().getType() != resultTypes[1]) {
      return rewriter.notifyMatchFailure(op,
                                         "expected converted vldus operand/result types");
    }

    FailureOr<StringRef> calleeName =
        buildVldusCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vldus signature");

    SmallVector<Type> intrinsicResultTypes(resultTypes.begin(), resultTypes.end());
    // The installed no-post A5 vldus intrinsic returns an extra hidden base ptr.
    intrinsicResultTypes.push_back(adaptor.getSource().getType());

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSource().getType(), adaptor.getAlign().getType()},
        intrinsicResultTypes);
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, intrinsicResultTypes,
        ValueRange{adaptor.getSource(), adaptor.getAlign()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults().take_front(resultTypes.size()));
    return success();
  }

private:
  LoweringState &state;
};

class LowerSprclrOpPattern final : public OpConversionPattern<pto::SprclrOp> {
public:
  explicit LowerSprclrOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                                LoweringState &state)
      : OpConversionPattern<pto::SprclrOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::SprclrOp op, pto::SprclrOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto spr = parseSprImmediate(op.getSpr());
    if (!spr)
      return rewriter.notifyMatchFailure(op, "unsupported sprclr target");

    StringRef calleeName = buildSprclrCallee(op.getContext());
    Value sprValue = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI16IntegerAttr(*spr));
    auto funcType = rewriter.getFunctionType(TypeRange{sprValue.getType()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{sprValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerVstsOpPattern final : public OpConversionPattern<pto::VstsOp> {
public:
  explicit LowerVstsOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::VstsOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VstsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elementType = getElementTypeFromVectorLike(op.getValue().getType());
    if (!elementType)
      return rewriter.notifyMatchFailure(op, "unsupported vsts element type");
    auto offsetBytes =
        convertElementOffsetToBytes(op, adaptor.getOffset(), elementType);
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    auto dist =
        parseStoreDistImmediate(op.getDist().value_or("NORM"), elementType);
    if (failed(offsetBytes) || !basePtr || !dist)
      return rewriter.notifyMatchFailure(op, "failed to materialize vsts operands");

    FailureOr<StringRef> calleeName =
        buildVstsCallee(op.getContext(), op.getValue().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vsts signature");

    Value distValue = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(*dist));
    Value zero = rewriter.create<arith::ConstantOp>(op.getLoc(),
                                                    rewriter.getI32IntegerAttr(0));
    SmallVector<Value> args{adaptor.getValue(), adaptor.getDestination(),
                            *offsetBytes, distValue, zero, adaptor.getMask()};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getValue().getType(), adaptor.getDestination().getType(),
                  rewriter.getI32Type(), rewriter.getI32Type(),
                  rewriter.getI32Type(), adaptor.getMask().getType()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerVsstbOpPattern final : public OpConversionPattern<pto::VsstbOp> {
public:
  explicit LowerVsstbOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VsstbOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VsstbOp op, pto::VsstbOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto basePtr =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    Value packedStride =
        packBlockRepeatStride(op, adaptor.getBlockStride(), adaptor.getRepeatStride());
    if (!basePtr || !packedStride)
      return rewriter.notifyMatchFailure(op, "failed to materialize vsstb operands");

    StringRef calleeName = buildVsstbCallee(op.getContext());
    Value zeroValue = getI32Constant(rewriter, op.getLoc(), 0);
    SmallVector<Value> args{adaptor.getValue(), adaptor.getDestination(),
                            packedStride, zeroValue, adaptor.getMask()};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getValue().getType(), adaptor.getDestination().getType(),
                  packedStride.getType(), zeroValue.getType(),
                  adaptor.getMask().getType()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerVstsPostOpPattern final
    : public OpConversionPattern<pto::VstsPostOp> {
public:
  explicit LowerVstsPostOpPattern(TypeConverter &typeConverter,
                                  MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VstsPostOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::VstsPostOp op, pto::VstsPostOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elementType = getElementTypeFromVectorLike(op.getValue().getType());
    if (!elementType)
      return rewriter.notifyMatchFailure(op, "unsupported vsts_post element type");

    auto offsetBytes = convertElementOffsetToBytes(op, adaptor.getOffset(), elementType);
    auto basePtr =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    auto dist =
        parseStoreDistImmediate(op.getDist().value_or("NORM"), elementType);
    if (failed(offsetBytes) || !basePtr || !dist) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize vsts_post operands");
    }

    Type updatedDestinationType =
        this->getTypeConverter()->convertType(op.getUpdatedDestination().getType());
    if (!updatedDestinationType || updatedDestinationType != adaptor.getDestination().getType()) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert vsts_post result type");
    }

    FailureOr<StringRef> calleeName =
        buildVstsPostCallee(op.getContext(), op.getValue().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vsts_post signature");

    Value distValue = getI32Constant(rewriter, op.getLoc(), *dist);
    Value postValue = getI32Constant(rewriter, op.getLoc(), 1);
    SmallVector<Value> args{adaptor.getValue(), adaptor.getDestination(), *offsetBytes,
                            distValue, postValue, adaptor.getMask()};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getValue().getType(), adaptor.getDestination().getType(),
                  (*offsetBytes).getType(), distValue.getType(), postValue.getType(),
                  adaptor.getMask().getType()},
        TypeRange{updatedDestinationType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{updatedDestinationType}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVstsx2OpPattern final : public OpConversionPattern<pto::Vstsx2Op> {
public:
  explicit LowerVstsx2OpPattern(TypeConverter &typeConverter,
                                MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::Vstsx2Op>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::Vstsx2Op op, pto::Vstsx2Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elementType = getElementTypeFromVectorLike(op.getLow().getType());
    if (!elementType)
      return rewriter.notifyMatchFailure(op, "unsupported vstsx2 element type");

    auto offsetBytes =
        convertElementOffsetToBytes(op, adaptor.getOffset(), elementType);
    auto basePtr =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    auto dist = parseStoreX2DistImmediate(op.getDist(), elementType);
    if (failed(offsetBytes) || !basePtr || !dist) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize vstsx2 operands");
    }

    FailureOr<StringRef> calleeName =
        buildVstsx2Callee(op.getContext(), op.getLow().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vstsx2 signature");

    Value distValue = getI32Constant(rewriter, op.getLoc(), *dist);
    Value zeroValue = getI32Constant(rewriter, op.getLoc(), 0);
    SmallVector<Value> args{adaptor.getLow(), adaptor.getHigh(),
                            adaptor.getDestination(), *offsetBytes, distValue,
                            zeroValue, adaptor.getMask()};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getLow().getType(), adaptor.getHigh().getType(),
                  adaptor.getDestination().getType(), (*offsetBytes).getType(),
                  distValue.getType(), zeroValue.getType(),
                  adaptor.getMask().getType()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), *calleeName, TypeRange{}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerPstuOpPattern final : public OpConversionPattern<pto::PstuOp> {
public:
  explicit LowerPstuOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::PstuOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::PstuOp op, pto::PstuOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<StringRef> calleeName = buildPstuCallee(op.getContext(), op);
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported pstu signature");

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "failed to convert pstu result types");
    if (resultTypes.size() != 2)
      return rewriter.notifyMatchFailure(op, "unexpected converted pstu result arity");

    auto baseType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getBase().getType());
    if (!baseType || adaptor.getAlignIn().getType() != resultTypes[0] ||
        adaptor.getBase().getType() != resultTypes[1]) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted pstu operand/result types");
    }

    SmallVector<Value> args{adaptor.getValue(), adaptor.getBase(), adaptor.getAlignIn()};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getValue().getType(), adaptor.getBase().getType(),
                  adaptor.getAlignIn().getType()},
        resultTypes);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), *calleeName, resultTypes,
                                              args);
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVstusOpPattern final : public OpConversionPattern<pto::VstusOp> {
public:
  explicit LowerVstusOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                               LoweringState &state)
      : OpConversionPattern<pto::VstusOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VstusOp op, pto::VstusOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elementType = getElementTypeFromVectorLike(op.getValue().getType());
    if (!elementType)
      return rewriter.notifyMatchFailure(op, "unsupported vstus element type");

    auto offsetBytes = convertElementOffsetToBytes(op, adaptor.getOffset(), elementType);
    if (failed(offsetBytes))
      return rewriter.notifyMatchFailure(op, "failed to convert vstus offset");

    Type resultType = this->getTypeConverter()->convertType(op.getAlignOut().getType());
    auto baseType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getBase().getType());
    if (!resultType || !baseType || adaptor.getAlignIn().getType() != resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vstus operand/result types");
    }

    StringRef calleeName = buildVstusCallee(op.getContext());
    SmallVector<Value> args{adaptor.getValue(), adaptor.getBase(), *offsetBytes,
                            adaptor.getAlignIn()};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getValue().getType(), adaptor.getBase().getType(),
                  (*offsetBytes).getType(), adaptor.getAlignIn().getType()},
        TypeRange{resultType});
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{resultType}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVsturOpPattern final : public OpConversionPattern<pto::VsturOp> {
public:
  explicit LowerVsturOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                               LoweringState &state)
      : OpConversionPattern<pto::VsturOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VsturOp op, pto::VsturOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto postMode = parsePostModeImmediate(op.getMode());
    if (!postMode)
      return rewriter.notifyMatchFailure(op, "unsupported vstur mode immediate");

    Type resultType = this->getTypeConverter()->convertType(op.getAlignOut().getType());
    auto baseType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getBase().getType());
    if (!resultType || !baseType || adaptor.getAlignIn().getType() != resultType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vstur operand/result types");
    }

    StringRef calleeName = buildVsturCallee(op.getContext());
    Value modeValue = getI32Constant(rewriter, op.getLoc(), *postMode);
    Value zeroValue = getI32Constant(rewriter, op.getLoc(), 0);
    SmallVector<Value> args{adaptor.getValue(), adaptor.getBase(), adaptor.getAlignIn(),
                            modeValue, zeroValue};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getValue().getType(), adaptor.getBase().getType(),
                  adaptor.getAlignIn().getType(), modeValue.getType(),
                  zeroValue.getType()},
        TypeRange{resultType});
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{resultType}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVstarOpPattern final : public OpConversionPattern<pto::VstarOp> {
public:
  explicit LowerVstarOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                               LoweringState &state)
      : OpConversionPattern<pto::VstarOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VstarOp op, pto::VstarOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto baseType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    Type alignType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!baseType || !alignType || adaptor.getValue().getType() != alignType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vstar operand types");
    }

    StringRef calleeName = buildVstarCallee(op.getContext());
    Value zeroValue = getI32Constant(rewriter, op.getLoc(), 0);
    SmallVector<Value> args{adaptor.getValue(), adaptor.getDestination(), zeroValue};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getValue().getType(), adaptor.getDestination().getType(),
                  zeroValue.getType()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerVstasOpPattern final : public OpConversionPattern<pto::VstasOp> {
public:
  explicit LowerVstasOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                               LoweringState &state)
      : OpConversionPattern<pto::VstasOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VstasOp op, pto::VstasOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto baseType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    Type alignType = this->getTypeConverter()->convertType(op.getValue().getType());
    auto dstType = dyn_cast<pto::PtrType>(op.getDestination().getType());
    if (!baseType || !alignType || adaptor.getValue().getType() != alignType || !dstType) {
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vstas operand types");
    }

    auto offsetBytes =
        convertElementOffsetToBytes(op, adaptor.getOffset(), dstType.getElementType());
    if (failed(offsetBytes))
      return rewriter.notifyMatchFailure(op, "failed to convert vstas offset");

    StringRef calleeName = buildVstasCallee(op.getContext());
    Value zeroValue = getI32Constant(rewriter, op.getLoc(), 0);
    SmallVector<Value> args{adaptor.getValue(), adaptor.getDestination(), *offsetBytes,
                            zeroValue};
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getValue().getType(), adaptor.getDestination().getType(),
                  (*offsetBytes).getType(), zeroValue.getType()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerVgather2OpPattern final
    : public OpConversionPattern<pto::Vgather2Op> {
public:
  explicit LowerVgather2OpPattern(TypeConverter &typeConverter,
                                  MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::Vgather2Op>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::Vgather2Op op, pto::Vgather2Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elemType = getElementTypeFromVectorLike(op.getResult().getType());
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    if (!elemType || !basePtr)
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vgather2 operand types");

    FailureOr<Value> mask = materializeDynamicPltMask(
        rewriter, state, op.getLoc(), adaptor.getActiveLanes(), elemType);
    if (failed(mask))
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize vgather2 mask");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert vgather2 result type");

    FailureOr<StringRef> calleeName =
        buildVgather2Callee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vgather2 signature");

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSource().getType(), adaptor.getOffsets().getType(),
                  (*mask).getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getSource(), adaptor.getOffsets(), *mask});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVgather2BcOpPattern final
    : public OpConversionPattern<pto::Vgather2BcOp> {
public:
  explicit LowerVgather2BcOpPattern(TypeConverter &typeConverter,
                                    MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::Vgather2BcOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::Vgather2BcOp op, pto::Vgather2BcOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!basePtr || !resultType)
      return rewriter.notifyMatchFailure(op,
          "unexpected converted vgather2_bc operand/result types");

    FailureOr<StringRef> calleeName =
        buildVgather2BcCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vgather2_bc signature");

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSource().getType(), adaptor.getOffsets().getType(),
                  adaptor.getMask().getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getSource(), adaptor.getOffsets(), adaptor.getMask()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVgatherbOpPattern final
    : public OpConversionPattern<pto::VgatherbOp> {
public:
  explicit LowerVgatherbOpPattern(TypeConverter &typeConverter,
                                  MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VgatherbOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::VgatherbOp op, pto::VgatherbOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto basePtr = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!basePtr || !resultType)
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vgatherb operand/result types");

    FailureOr<StringRef> calleeName =
        buildVgatherbCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vgatherb signature");

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSource().getType(), adaptor.getOffsets().getType(),
                  adaptor.getMask().getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getSource(), adaptor.getOffsets(), adaptor.getMask()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVscatterOpPattern final
    : public OpConversionPattern<pto::VscatterOp> {
public:
  explicit LowerVscatterOpPattern(TypeConverter &typeConverter,
                                  MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VscatterOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::VscatterOp op, pto::VscatterOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elemType = getElementTypeFromVectorLike(op.getValue().getType());
    auto basePtr =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    if (!elemType || !basePtr)
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vscatter operand types");

    FailureOr<Value> mask = materializeDynamicPltMask(
        rewriter, state, op.getLoc(), adaptor.getActiveLanes(), elemType);
    if (failed(mask))
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize vscatter mask");

    FailureOr<StringRef> calleeName =
        buildVscatterCallee(op.getContext(), op.getValue().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vscatter signature");

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getValue().getType(), adaptor.getDestination().getType(),
                  adaptor.getOffsets().getType(), (*mask).getType()},
        TypeRange{});
    rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{},
        ValueRange{adaptor.getValue(), adaptor.getDestination(),
                   adaptor.getOffsets(), *mask});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerVpreluOpPattern final : public OpConversionPattern<pto::VpreluOp> {
public:
  explicit LowerVpreluOpPattern(TypeConverter &typeConverter,
                                MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VpreluOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::VpreluOp op, pto::VpreluOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto laneCount = getElementCountFromVectorLike(op.getResult().getType());
    Type elemType = getElementTypeFromVectorLike(op.getResult().getType());
    if (!laneCount || !elemType)
      return rewriter.notifyMatchFailure(op, "unsupported vprelu signature");

    FailureOr<Value> mask = materializeDynamicPltMask(
        rewriter, state, op.getLoc(), getI32Constant(rewriter, op.getLoc(), *laneCount),
        elemType);
    if (failed(mask))
      return rewriter.notifyMatchFailure(op, "failed to materialize vprelu mask");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert vprelu result type");

    FailureOr<StringRef> calleeName =
        buildVpreluCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vprelu callee");

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getLhs().getType(), adaptor.getRhs().getType(),
                  (*mask).getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getLhs(), adaptor.getRhs(), *mask});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVaxpyOpPattern final : public OpConversionPattern<pto::VaxpyOp> {
public:
  explicit LowerVaxpyOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VaxpyOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VaxpyOp op, pto::VaxpyOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto laneCount = getElementCountFromVectorLike(op.getResult().getType());
    Type elemType = getElementTypeFromVectorLike(op.getResult().getType());
    if (!laneCount || !elemType)
      return rewriter.notifyMatchFailure(op, "unsupported vaxpy signature");

    FailureOr<Value> mask = materializeDynamicPltMask(
        rewriter, state, op.getLoc(), getI32Constant(rewriter, op.getLoc(), *laneCount),
        elemType);
    if (failed(mask))
      return rewriter.notifyMatchFailure(op, "failed to materialize vaxpy mask");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert vaxpy result type");

    FailureOr<StringRef> calleeName =
        buildVaxpyCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vaxpy callee");

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getSrc1().getType(), adaptor.getSrc0().getType(),
                  adaptor.getAlpha().getType(), (*mask).getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getSrc1(), adaptor.getSrc0(), adaptor.getAlpha(),
                   *mask});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVciOpPattern final : public OpConversionPattern<pto::VciOp> {
public:
  explicit LowerVciOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                             LoweringState &state)
      : OpConversionPattern<pto::VciOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VciOp op, pto::VciOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto order = parseOrderImmediate(op.getOrder().value_or("ASC"));
    if (!order)
      return rewriter.notifyMatchFailure(op, "unsupported vci order");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert vci result type");

    FailureOr<StringRef> calleeName =
        buildVciCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vci callee");

    Value orderValue = getI32Constant(rewriter, op.getLoc(), *order);
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getIndex().getType(), orderValue.getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getIndex(), orderValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVexpdiffOpPattern final
    : public OpConversionPattern<pto::VexpdiffOp> {
public:
  explicit LowerVexpdiffOpPattern(TypeConverter &typeConverter,
                                  MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VexpdiffOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::VexpdiffOp op, pto::VexpdiffOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto laneCount = getElementCountFromVectorLike(op.getInput().getType());
    Type elemType = getElementTypeFromVectorLike(op.getInput().getType());
    auto part = parsePartImmediate(op.getPart());
    if (!laneCount || !elemType || !part)
      return rewriter.notifyMatchFailure(op, "unsupported vexpdiff signature");

    FailureOr<Value> mask = materializeDynamicPltMask(
        rewriter, state, op.getLoc(), getI32Constant(rewriter, op.getLoc(), *laneCount),
        elemType);
    if (failed(mask))
      return rewriter.notifyMatchFailure(op, "failed to materialize vexpdiff mask");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert vexpdiff result type");

    FailureOr<StringRef> calleeName =
        buildVexpdiffCallee(op.getContext(), op.getInput().getType(),
                            op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vexpdiff callee");

    Value partValue = getI32Constant(rewriter, op.getLoc(), *part);
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getInput().getType(), adaptor.getMax().getType(),
                  (*mask).getType(), partValue.getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getInput(), adaptor.getMax(), *mask, partValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVbitsortOpPattern final
    : public OpConversionPattern<pto::VbitsortOp> {
public:
  explicit LowerVbitsortOpPattern(TypeConverter &typeConverter,
                                  MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VbitsortOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::VbitsortOp op, pto::VbitsortOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    auto srcType = dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    auto idxType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getIndices().getType());
    if (!dstType || !srcType || !idxType)
      return rewriter.notifyMatchFailure(op,
                                         "unexpected converted vbitsort operand types");

    FailureOr<Value> config = packVbitsortConfig(op, adaptor.getRepeatTimes());
    if (failed(config))
      return rewriter.notifyMatchFailure(op, "failed to pack vbitsort config");

    FailureOr<StringRef> calleeName = buildVbitsortCallee(op.getContext(), op);
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vbitsort signature");

    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getDestination().getType(), adaptor.getSource().getType(),
                  adaptor.getIndices().getType(), (*config).getType()},
        TypeRange{});
    rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{},
        ValueRange{adaptor.getDestination(), adaptor.getSource(),
                   adaptor.getIndices(), *config});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerVcvtOpPattern final : public OpConversionPattern<pto::VcvtOp> {
public:
  explicit LowerVcvtOpPattern(TypeConverter &typeConverter,
                              MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::VcvtOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VcvtOp op, pto::VcvtOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputLanes = getElementCountFromVectorLike(op.getInput().getType());
    if (!inputLanes)
      return rewriter.notifyMatchFailure(op, "unsupported vcvt input shape");

    FailureOr<VcvtContract> contract = buildVcvtContract(op);
    if (failed(contract))
      return rewriter.notifyMatchFailure(op, "unsupported vcvt type pair");

    Type maskElemType = rewriter.getIntegerType((*contract).maskBitWidth);
    FailureOr<Value> mask = materializeDynamicPltMask(
        rewriter, state, op.getLoc(),
        getI32Constant(rewriter, op.getLoc(), *inputLanes), maskElemType);
    if (failed(mask))
      return rewriter.notifyMatchFailure(op, "failed to materialize vcvt mask");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert vcvt result type");

    SmallVector<Value> callArgs;
    SmallVector<Type> argTypes;
    callArgs.push_back(adaptor.getInput());
    argTypes.push_back(adaptor.getInput().getType());
    callArgs.push_back(*mask);
    argTypes.push_back((*mask).getType());

    if ((*contract).requiresRnd) {
      auto roundMode =
          op.getRndAttr() ? parseRoundModeImmediate(*op.getRnd()) : std::nullopt;
      if (!roundMode)
        return rewriter.notifyMatchFailure(op, "vcvt requires valid rnd attr");
      Value roundValue = getI32Constant(rewriter, op.getLoc(), *roundMode);
      callArgs.push_back(roundValue);
      argTypes.push_back(roundValue.getType());
    }

    if ((*contract).requiresSat) {
      auto saturation =
          op.getSatAttr() ? parseSaturationImmediate(*op.getSat()) : std::nullopt;
      if (!saturation)
        return rewriter.notifyMatchFailure(op, "vcvt requires valid sat attr");
      Value satValue = getI32Constant(rewriter, op.getLoc(), *saturation);
      callArgs.push_back(satValue);
      argTypes.push_back(satValue.getType());
    }

    if ((*contract).requiresPart) {
      auto part = op.getPartAttr() ? parsePartImmediate(*op.getPart()) : std::nullopt;
      if (!part)
        return rewriter.notifyMatchFailure(op, "vcvt requires valid part attr");
      Value partValue = getI32Constant(rewriter, op.getLoc(), *part);
      callArgs.push_back(partValue);
      argTypes.push_back(partValue.getType());
    }

    auto funcType = rewriter.getFunctionType(argTypes, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), StringRef((*contract).intrinsic), TypeRange{resultType}, callArgs);
    state.plannedDecls.push_back(
        PlannedDecl{std::string((*contract).intrinsic), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class LowerVtrcOpPattern final : public OpConversionPattern<pto::VtrcOp> {
public:
  explicit LowerVtrcOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                              LoweringState &state)
      : OpConversionPattern<pto::VtrcOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(pto::VtrcOp op, pto::VtrcOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto roundMode = parseRoundModeImmediate(op.getRoundMode());
    if (!roundMode)
      return rewriter.notifyMatchFailure(op, "unsupported vtrc signature");

    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert vtrc result type");

    FailureOr<StringRef> calleeName =
        buildVtrcCallee(op.getContext(), op.getResult().getType());
    if (failed(calleeName))
      return rewriter.notifyMatchFailure(op, "unsupported vtrc callee");

    Value roundValue = getI32Constant(rewriter, op.getLoc(), *roundMode);
    auto funcType = rewriter.getFunctionType(
        TypeRange{adaptor.getInput().getType(), roundValue.getType(),
                  adaptor.getMask().getType()},
        TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), *calleeName, TypeRange{resultType},
        ValueRange{adaptor.getInput(), roundValue, adaptor.getMask()});
    state.plannedDecls.push_back(PlannedDecl{calleeName->str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename StoreOp>
class LowerPredicateStoreOpPattern final : public OpConversionPattern<StoreOp> {
public:
  explicit LowerPredicateStoreOpPattern(TypeConverter &typeConverter,
                                        MLIRContext *context,
                                        LoweringState &state)
      : OpConversionPattern<StoreOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(StoreOp op, typename StoreOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto llvmDestType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getDestination().getType());
    Type valueType = this->getTypeConverter()->convertType(op.getValue().getType());
    if (!llvmDestType || !valueType)
      return rewriter.notifyMatchFailure(
          op, "expected converted predicate-store operand types");

    auto dist = parsePredicateStoreDistImmediate(op.getDist());
    if (!dist)
      return rewriter.notifyMatchFailure(
          op, "unsupported predicate-store dist immediate");

    Value offset = castIntegerLikeTo(op, adaptor.getOffset(), rewriter.getI32Type());
    if (!offset)
      return rewriter.notifyMatchFailure(
          op, "failed to convert predicate-store offset to i32");

    StringRef calleeName = getPredicateStoreCallee<StoreOp>(op.getContext());
    SmallVector<Value> args;
    args.push_back(adaptor.getValue());
    args.push_back(adaptor.getDestination());
    args.push_back(offset);
    args.push_back(rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(*dist)));
    args.push_back(rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(0)));
    auto funcType = rewriter.getFunctionType(
        TypeRange{valueType, llvmDestType, rewriter.getI32Type(),
                  rewriter.getI32Type(), rewriter.getI32Type()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename LoadOp>
class LowerPredicateLoadOpPattern final : public OpConversionPattern<LoadOp> {
public:
  explicit LowerPredicateLoadOpPattern(TypeConverter &typeConverter,
                                       MLIRContext *context,
                                       LoweringState &state)
      : OpConversionPattern<LoadOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(LoadOp op, typename LoadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto llvmSourceType =
        dyn_cast<LLVM::LLVMPointerType>(adaptor.getSource().getType());
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!llvmSourceType || !resultType)
      return rewriter.notifyMatchFailure(
          op, "expected converted predicate-load operand/result types");

    auto dist = parsePredicateLoadDistImmediate(op.getDist());
    if (!dist)
      return rewriter.notifyMatchFailure(
          op, "unsupported predicate-load dist immediate");

    Value offset = castIntegerLikeTo(op, adaptor.getOffset(), rewriter.getI32Type());
    if (!offset)
      return rewriter.notifyMatchFailure(
          op, "failed to convert predicate-load offset to i32");

    StringRef calleeName = getPredicateLoadCallee<LoadOp>(op.getContext());
    SmallVector<Value> args;
    args.push_back(adaptor.getSource());
    args.push_back(offset);
    args.push_back(rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(*dist)));
    args.push_back(rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(0)));
    auto funcType = rewriter.getFunctionType(
        TypeRange{llvmSourceType, rewriter.getI32Type(), rewriter.getI32Type(),
                  rewriter.getI32Type()},
        TypeRange{resultType});
    auto call =
        rewriter.create<func::CallOp>(op.getLoc(), calleeName, resultType, args);
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

template <typename LoopOp>
class LowerSetLoopConfigOpPattern final : public OpConversionPattern<LoopOp> {
public:
  explicit LowerSetLoopConfigOpPattern(TypeConverter &typeConverter,
                                       MLIRContext *context,
                                       LoweringState &state)
      : OpConversionPattern<LoopOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(LoopOp op, typename LoopOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> packed = failure();
    if constexpr (std::is_same_v<LoopOp, pto::SetLoopSizeOutToUbOp> ||
                  std::is_same_v<LoopOp, pto::SetLoopSizeUbToOutOp>) {
      packed = packLoopSize(op, adaptor.getFirst(), adaptor.getSecond());
    } else {
      packed = packLoopPair(op, adaptor.getFirst(), adaptor.getSecond());
    }
    if (failed(packed))
      return rewriter.notifyMatchFailure(op,
                                         "failed to pack loop configuration");

    StringRef calleeName = buildSetLoopCallee<LoopOp>(op.getContext());
    auto funcType =
        rewriter.getFunctionType(TypeRange{rewriter.getI64Type()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{*packed});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename SyncOp>
class LowerPipeEventSyncOpPattern final : public OpConversionPattern<SyncOp> {
public:
  explicit LowerPipeEventSyncOpPattern(TypeConverter &typeConverter,
                                       MLIRContext *context,
                                       LoweringState &state)
      : OpConversionPattern<SyncOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(SyncOp op, typename SyncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto src = parsePipeImmediate(stringifyPIPE(op.getSrcPipe().getPipe()));
    auto dst = parsePipeImmediate(stringifyPIPE(op.getDstPipe().getPipe()));
    auto event = parseEventImmediate(stringifyEVENT(op.getEventId().getEvent()));
    if (!src || !dst || !event)
      return rewriter.notifyMatchFailure(op, "unsupported sync immediate");

    StringRef calleeName = buildSyncCallee<SyncOp>(op.getContext());
    Value srcValue = getI64Constant(rewriter, op.getLoc(), *src);
    Value dstValue = getI64Constant(rewriter, op.getLoc(), *dst);
    Value eventValue = getI64Constant(rewriter, op.getLoc(), *event);
    auto funcType = rewriter.getFunctionType(
        TypeRange{rewriter.getI64Type(), rewriter.getI64Type(),
                  rewriter.getI64Type()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{srcValue, dstValue, eventValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

class LowerBarrierOpPattern final : public OpConversionPattern<pto::BarrierOp> {
public:
  explicit LowerBarrierOpPattern(TypeConverter &typeConverter,
                                 MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::BarrierOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto pipe = parsePipeImmediate(stringifyPIPE(op.getPipe().getPipe()));
    if (!pipe)
      return rewriter.notifyMatchFailure(op, "unsupported barrier pipe");

    StringRef calleeName = buildSyncCallee<pto::BarrierOp>(op.getContext());
    Value pipeValue = getI64Constant(rewriter, op.getLoc(), *pipe);
    auto funcType =
        rewriter.getFunctionType(TypeRange{rewriter.getI64Type()}, TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{pipeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename BufSyncOp>
class LowerBufSyncOpPattern final : public OpConversionPattern<BufSyncOp> {
public:
  explicit LowerBufSyncOpPattern(TypeConverter &typeConverter,
                                 MLIRContext *context, LoweringState &state)
      : OpConversionPattern<BufSyncOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(BufSyncOp op, typename BufSyncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    PIPE pipe = PIPE::PIPE_UNASSIGNED;
    if (auto pipeAttr = dyn_cast<PipeAttr>(op.getOpTypeAttr())) {
      pipe = pipeAttr.getPipe();
    } else {
      auto opTypeOr = parseSyncOpTypeLikeAttr(op.getOpTypeAttr());
      if (failed(opTypeOr))
        return rewriter.notifyMatchFailure(
            op, "buffer sync expects pipe/sync_op_type/pipe_event_type attr");
      pipe = mapSyncOpTypeToPipe(*opTypeOr);
    }
    if (!isConcreteSyncPipe(pipe))
      return rewriter.notifyMatchFailure(op,
                                         "buffer sync op_type cannot map to concrete pipe");

    auto pipeImm = parsePipeImmediate(stringifyPIPE(pipe));
    if (!pipeImm)
      return rewriter.notifyMatchFailure(op, "unsupported buffer sync pipe");

    StringRef calleeName = buildSyncCallee<BufSyncOp>(op.getContext());
    Value pipeValue = getI64Constant(rewriter, op.getLoc(), *pipeImm);
    Value bufIdValue =
        getI64Constant(rewriter, op.getLoc(), op.getBufIdAttr().getInt());
    Value modeValue =
        getI64Constant(rewriter, op.getLoc(), op.getModeAttr().getInt());
    auto funcType = rewriter.getFunctionType(
        TypeRange{rewriter.getI64Type(), rewriter.getI64Type(),
                  rewriter.getI64Type()},
        TypeRange{});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                  ValueRange{pipeValue, bufIdValue, modeValue});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

template <typename QueryOp>
class LowerRuntimeQueryOpPattern final : public OpConversionPattern<QueryOp> {
public:
  explicit LowerRuntimeQueryOpPattern(TypeConverter &typeConverter,
                                      MLIRContext *context,
                                      LoweringState &state)
      : OpConversionPattern<QueryOp>(typeConverter, context), state(state) {}

  LogicalResult
  matchAndRewrite(QueryOp op, typename QueryOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert runtime-query result type");

    StringRef calleeName = buildRuntimeQueryCallee<QueryOp>(op.getContext());
    auto funcType = rewriter.getFunctionType(TypeRange{}, TypeRange{resultType});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), calleeName,
                                              TypeRange{resultType}, ValueRange{});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  LoweringState &state;
};

class ConvertVPTOUnrealizedCastOp final
    : public OpConversionPattern<UnrealizedConversionCastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();
    if (!hasVPTOConvertibleType(op->getOperandTypes()) &&
        !hasVPTOConvertibleType(op->getResultTypes()))
      return failure();

    Type convertedResultType =
        getTypeConverter()->convertType(op.getResult(0).getType());
    if (!convertedResultType)
      return failure();

    Value input = adaptor.getOperands().front();
    if (input.getType() != convertedResultType)
      return failure();

    rewriter.replaceOp(op, input);
    return success();
  }
};

class ConvertPtoAddPtrOp final : public OpConversionPattern<pto::AddPtrOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedResultType = getTypeConverter()->convertType(op.getResult().getType());
    auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(convertedResultType);
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

class ConvertPtoCastPtrOp final : public OpConversionPattern<pto::CastPtrOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::CastPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedResultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedResultType)
      return rewriter.notifyMatchFailure(op,
                                         "could not convert castptr result type");

    Value input = adaptor.getInput();
    Type inputType = input.getType();
    if (inputType == convertedResultType) {
      rewriter.replaceOp(op, input);
      return success();
    }

    if (auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(convertedResultType)) {
      if (isa<IntegerType>(inputType)) {
        rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, llvmPtrType, input);
        return success();
      }
      auto sourcePtrType = dyn_cast<LLVM::LLVMPointerType>(inputType);
      if (!sourcePtrType)
        return rewriter.notifyMatchFailure(op,
                                           "expected integer or LLVM pointer input");
      if (sourcePtrType.getAddressSpace() == llvmPtrType.getAddressSpace()) {
        rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, llvmPtrType, input);
        return success();
      }
      return rewriter.notifyMatchFailure(
          op, "cross-address-space ptr casts are unsupported");
    }

    if (auto resultIntType = dyn_cast<IntegerType>(convertedResultType)) {
      if (isa<LLVM::LLVMPointerType>(inputType)) {
        rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, resultIntType, input);
        return success();
      }
    }

    return rewriter.notifyMatchFailure(op, "unsupported castptr conversion");
  }
};

class ConvertPtoLoadScalarOp final
    : public OpConversionPattern<pto::LoadScalarOp> {
public:
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
      if (auto intType = dyn_cast<IntegerType>(type))
        alignBytes = llvm::divideCeil(unsigned(intType.getWidth()), 8u);
      else if (type.isF16() || type.isBF16())
        alignBytes = 2;
      else if (type.isF32())
        alignBytes = 4;
      else if (type.isF64())
        alignBytes = 8;
      return alignBytes;
    };

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        op, op.getValue().getType(), elemPtr,
        getNaturalAlignment(op.getValue().getType()));
    return success();
  }
};

class ConvertPtoStoreScalarOp final
    : public OpConversionPattern<pto::StoreScalarOp> {
public:
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
      if (auto intType = dyn_cast<IntegerType>(type))
        alignBytes = llvm::divideCeil(unsigned(intType.getWidth()), 8u);
      else if (type.isF16() || type.isBF16())
        alignBytes = 2;
      else if (type.isF32())
        alignBytes = 4;
      else if (type.isF64())
        alignBytes = 8;
      return alignBytes;
    };

    rewriter.create<LLVM::StoreOp>(op.getLoc(), adaptor.getValue(), elemPtr,
                                   getNaturalAlignment(adaptor.getValue().getType()));
    rewriter.eraseOp(op);
    return success();
  }
};

class ConvertVPTOTypedCarrierOp final : public ConversionPattern {
public:
  ConvertVPTOTypedCarrierOp(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<pto::CastPtrOp>(op))
      return failure();
    if (!hasVPTOConvertibleType(op->getOperandTypes()) &&
        !hasVPTOConvertibleType(op->getResultTypes()))
      return failure();
    if (op->getNumRegions() != 0)
      return rewriter.notifyMatchFailure(
          op, "region ops with VPTO types are handled structurally");

    FailureOr<Operation *> converted =
        convertOpResultTypes(op, operands, *typeConverter, rewriter);
    if (failed(converted))
      return failure();
    return success();
  }
};

static void populateVPTOOpLoweringPatterns(VPTOTypeConverter &typeConverter,
                                           RewritePatternSet &patterns,
                                           LoweringState &state) {
  patterns.add<LowerUnaryMaskedOpPattern<pto::VabsOp>,
               LowerUnaryMaskedOpPattern<pto::VexpOp>,
               LowerUnaryMaskedOpPattern<pto::VlnOp>,
               LowerUnaryMaskedOpPattern<pto::VnegOp>,
               LowerUnaryMaskedOpPattern<pto::VsqrtOp>,
               LowerUnaryMaskedOpPattern<pto::VreluOp>,
               LowerUnaryMaskedOpPattern<pto::VnotOp>,
               LowerVsqzOpPattern, LowerVusqzOpPattern,
               LowerVmulaOpPattern, LowerVmullOpPattern,
               LowerBinaryMaskedOpPattern<pto::VaddOp>,
               LowerBinaryMaskedOpPattern<pto::VsubOp>,
               LowerBinaryMaskedOpPattern<pto::VmulOp>,
               LowerBinaryMaskedOpPattern<pto::VdivOp>,
               LowerBinaryMaskedOpPattern<pto::VmaxOp>,
               LowerBinaryMaskedOpPattern<pto::VminOp>,
               LowerBinaryMaskedOpPattern<pto::VandOp>,
               LowerBinaryMaskedOpPattern<pto::VorOp>,
               LowerBinaryMaskedOpPattern<pto::VxorOp>,
               LowerCarryBinaryOpPattern<pto::VaddcOp>,
               LowerCarryBinaryOpPattern<pto::VsubcOp>,
               LowerCarryBinaryOpPattern<pto::VaddcsOp>,
               LowerCarryBinaryOpPattern<pto::VsubcsOp>,
               LowerBinaryMaskedOpPattern<pto::VshlOp>,
               LowerBinaryMaskedOpPattern<pto::VshrOp>,
               LowerVecScalarMaskedOpPattern<pto::VmulsOp>,
               LowerVecScalarMaskedOpPattern<pto::VaddsOp>,
               LowerVecScalarMaskedOpPattern<pto::VmaxsOp>,
               LowerVecScalarMaskedOpPattern<pto::VminsOp>,
               LowerVecScalarMaskedOpPattern<pto::VlreluOp>,
               LowerVecScalarMaskedOpPattern<pto::VshlsOp>,
               LowerVecScalarMaskedOpPattern<pto::VshrsOp>,
               LowerReductionUnaryOpPattern<pto::VcaddOp>,
               LowerReductionUnaryOpPattern<pto::VcmaxOp>,
               LowerReductionUnaryOpPattern<pto::VcminOp>,
               LowerReductionUnaryOpPattern<pto::VcgaddOp>,
               LowerReductionUnaryOpPattern<pto::VcgmaxOp>,
               LowerReductionUnaryOpPattern<pto::VcgminOp>,
               LowerReductionUnaryOpPattern<pto::VcpaddOp>,
               LowerVdupOpPattern,
               LowerVbrOpPattern,
               LowerPredicatePackOpPattern<pto::PpackOp>,
               LowerPredicatePackOpPattern<pto::PunpackOp>,
               LowerVselOpPattern, LowerVselrOpPattern, LowerPnotOpPattern,
               LowerPredicateMaskBinaryOpPattern<pto::PselOp>,
               LowerPredicateMaskBinaryOpPattern<pto::PandOp>,
               LowerPredicateMaskBinaryOpPattern<pto::PorOp>,
               LowerPredicateMaskBinaryOpPattern<pto::PxorOp>,
               LowerPredicatePairReorderOpPattern<pto::PdintlvB8Op>,
               LowerPredicatePairReorderOpPattern<pto::PdintlvB16Op>,
               LowerPredicatePairReorderOpPattern<pto::PdintlvB32Op>,
               LowerPredicatePairReorderOpPattern<pto::PintlvB8Op>,
               LowerPredicatePairReorderOpPattern<pto::PintlvB16Op>,
               LowerPredicatePairReorderOpPattern<pto::PintlvB32Op>,
               LowerUnpackOpPattern<pto::VsunpackOp>,
               LowerUnpackOpPattern<pto::VzunpackOp>,
               LowerVpackOpPattern,
               LowerInterleaveOpPattern<pto::VintlvOp>,
               LowerInterleaveOpPattern<pto::VdintlvOp>,
               LowerCmpOpPattern<pto::VcmpOp>,
               LowerCmpOpPattern<pto::VcmpsOp>,
               LowerPltOpPattern<pto::PltB8Op>,
               LowerPltOpPattern<pto::PltB16Op>,
               LowerPltOpPattern<pto::PltB32Op>,
               LowerPsetOpPattern<pto::PsetB8Op>,
               LowerPsetOpPattern<pto::PsetB16Op>,
               LowerPsetOpPattern<pto::PsetB32Op>,
               LowerPgeOpPattern<pto::PgeB8Op>,
               LowerPgeOpPattern<pto::PgeB16Op>,
               LowerPgeOpPattern<pto::PgeB32Op>,
               LowerSetLoopConfigOpPattern<pto::SetLoop2StrideOutToUbOp>,
               LowerSetLoopConfigOpPattern<pto::SetLoop1StrideOutToUbOp>,
               LowerSetLoopConfigOpPattern<pto::SetLoopSizeOutToUbOp>,
               LowerSetLoopConfigOpPattern<pto::SetLoop2StrideUbToOutOp>,
               LowerSetLoopConfigOpPattern<pto::SetLoop1StrideUbToOutOp>,
               LowerSetLoopConfigOpPattern<pto::SetLoopSizeUbToOutOp>,
               LowerPipeEventSyncOpPattern<pto::SetFlagOp>,
               LowerPipeEventSyncOpPattern<pto::WaitFlagOp>,
               LowerBarrierOpPattern,
               LowerBufSyncOpPattern<pto::GetBufOp>,
               LowerBufSyncOpPattern<pto::RlsBufOp>,
               LowerRuntimeQueryOpPattern<pto::GetBlockIdxOp>,
               LowerRuntimeQueryOpPattern<pto::GetSubBlockIdxOp>,
               LowerRuntimeQueryOpPattern<pto::GetBlockNumOp>,
               LowerRuntimeQueryOpPattern<pto::GetSubBlockNumOp>,
               LowerVldsOpPattern, LowerVldsPostOpPattern,
               LowerVldsx2OpPattern, LowerVsldbOpPattern,
               LowerVldasOpPattern, LowerInitAlignOpPattern,
               LowerVldusOpPattern, LowerSprclrOpPattern,
               LowerVstsOpPattern, LowerVsstbOpPattern,
               LowerVstsPostOpPattern, LowerVstsx2OpPattern,
               LowerVstarOpPattern, LowerVstasOpPattern,
               LowerVgather2OpPattern, LowerVgather2BcOpPattern,
               LowerVgatherbOpPattern, LowerVscatterOpPattern,
               LowerVpreluOpPattern, LowerVaxpyOpPattern,
               LowerVciOpPattern, LowerVexpdiffOpPattern,
               LowerVbitsortOpPattern, LowerVtrcOpPattern, LowerVcvtOpPattern,
               LowerPredicateLoadOpPattern<pto::PldiOp>,
               LowerPredicateLoadOpPattern<pto::PldsOp>,
               LowerPredicateStoreOpPattern<pto::PstiOp>,
               LowerPredicateStoreOpPattern<pto::PstsOp>,
               LowerPstuOpPattern, LowerVstusOpPattern, LowerVsturOpPattern,
               LowerCopyOpPattern<pto::CopyGmToUbufOp>,
               LowerCopyOpPattern<pto::CopyUbufToGmOp>>(
      typeConverter, patterns.getContext(), state);
}

static void configureVPTOOpLoweringTarget(ConversionTarget &target,
                                          VPTOTypeConverter &typeConverter) {
  (void)typeConverter;
  target.addLegalOp<ModuleOp>();
  target.addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect,
                         func::FuncDialect, scf::SCFDialect>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addIllegalOp<pto::SetFlagOp, pto::WaitFlagOp, pto::BarrierOp,
                      pto::GetBufOp, pto::RlsBufOp>();
  target.addIllegalOp<pto::GetBlockIdxOp, pto::GetSubBlockIdxOp,
                      pto::GetBlockNumOp, pto::GetSubBlockNumOp>();
  target.addIllegalOp<pto::SetLoop2StrideOutToUbOp, pto::SetLoop1StrideOutToUbOp,
                      pto::SetLoopSizeOutToUbOp, pto::SetLoop2StrideUbToOutOp,
                      pto::SetLoop1StrideUbToOutOp, pto::SetLoopSizeUbToOutOp>();
  target.addIllegalOp<pto::VldsOp, pto::VldsPostOp, pto::Vldsx2Op,
                      pto::VsldbOp, pto::VldasOp, pto::InitAlignOp,
                      pto::VldusOp, pto::SprclrOp, pto::VstsOp,
                      pto::VsstbOp, pto::VstsPostOp, pto::Vstsx2Op,
                      pto::VstarOp, pto::VstasOp, pto::Vgather2Op,
                      pto::Vgather2BcOp, pto::VgatherbOp, pto::VscatterOp,
                      pto::PldiOp, pto::PldsOp, pto::PstiOp, pto::PstsOp,
                      pto::PstuOp, pto::VstusOp, pto::VsturOp>();
  target.addIllegalOp<pto::PltB8Op, pto::PltB16Op, pto::PltB32Op,
                      pto::PsetB8Op, pto::PsetB16Op, pto::PsetB32Op,
                      pto::PgeB8Op, pto::PgeB16Op, pto::PgeB32Op>();
  target.addIllegalOp<pto::VabsOp, pto::VexpOp, pto::VlnOp, pto::VnegOp,
                      pto::VsqrtOp, pto::VreluOp, pto::VnotOp, pto::VsqzOp,
                      pto::VusqzOp, pto::VmulaOp, pto::VmullOp, pto::VaddOp,
                      pto::VsubOp, pto::VmulOp,
                      pto::VdivOp, pto::VmaxOp, pto::VminOp, pto::VandOp,
                      pto::VorOp, pto::VxorOp, pto::VaddcOp, pto::VsubcOp,
                      pto::VaddcsOp, pto::VsubcsOp, pto::VshlOp, pto::VshrOp,
                      pto::VmulsOp, pto::VaddsOp, pto::VmaxsOp,
                      pto::VminsOp, pto::VlreluOp, pto::VshlsOp, pto::VshrsOp,
                      pto::VcaddOp, pto::VcmaxOp, pto::VcminOp,
                      pto::VcgaddOp, pto::VcgmaxOp, pto::VcgminOp, pto::VcpaddOp,
                      pto::VdupOp, pto::VbrOp,
                      pto::PpackOp, pto::PunpackOp, pto::VselOp, pto::VselrOp,
                      pto::PnotOp, pto::PselOp, pto::PandOp, pto::PorOp, pto::PxorOp,
                      pto::PdintlvB8Op, pto::PdintlvB16Op, pto::PdintlvB32Op,
                      pto::PintlvB8Op, pto::PintlvB16Op, pto::PintlvB32Op,
                      pto::VsunpackOp, pto::VzunpackOp, pto::VpackOp,
                      pto::VintlvOp, pto::VdintlvOp, pto::VpreluOp,
                      pto::VaxpyOp, pto::VciOp, pto::VexpdiffOp,
                      pto::VbitsortOp, pto::VtrcOp, pto::VcvtOp,
                      pto::VcmpOp, pto::VcmpsOp,
                      pto::CopyGmToUbufOp, pto::CopyUbufToGmOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
}

static void populateVPTOStructuralTypePatterns(
    VPTOTypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns,
                                                       target);
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
}

static void foldVPTOTypeCasts(ModuleOp module, TypeConverter &typeConverter) {
  SmallVector<UnrealizedConversionCastOp> castsToFold;
  module.walk([&](UnrealizedConversionCastOp castOp) {
    if (castOp->getNumOperands() != 1 || castOp->getNumResults() != 1)
      return;
    if (!hasVPTOConvertibleType(castOp->getOperandTypes()) &&
        !hasVPTOConvertibleType(castOp->getResultTypes()))
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
}

static LogicalResult lowerVPTOOps(ModuleOp module, llvm::raw_ostream &diagOS) {
  MLIRContext *context = module.getContext();
  VPTOTypeConverter typeConverter(context);
  ConversionTarget target(*context);
  RewritePatternSet patterns(context);
  LoweringState state;

  configureVPTOOpLoweringTarget(target, typeConverter);
  populateVPTOOpLoweringPatterns(typeConverter, patterns, state);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    diagOS << "VPTO LLVM emission failed: VPTO op lowering failed\n";
    return failure();
  }
  if (failed(materializeDecls(module, state.plannedDecls, diagOS)))
    return failure();
  return success();
}

static LogicalResult lowerVPTOTypes(ModuleOp module, llvm::raw_ostream &diagOS) {
  MLIRContext *context = module.getContext();
  VPTOTypeConverter typeConverter(context);
  ConversionTarget target(*context);
  RewritePatternSet patterns(context);

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
  target.addIllegalOp<pto::AddPtrOp, pto::CastPtrOp, pto::LoadScalarOp,
                      pto::StoreScalarOp>();
  target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
      [&](UnrealizedConversionCastOp op) {
        return !hasVPTOConvertibleType(op->getOperandTypes()) &&
               !hasVPTOConvertibleType(op->getResultTypes());
      });
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return typeConverter.isLegal(op->getOperandTypes()) &&
           typeConverter.isLegal(op->getResultTypes());
  });

  populateVPTOStructuralTypePatterns(typeConverter, patterns, target);
  patterns.add<ConvertPtoAddPtrOp, ConvertPtoCastPtrOp, ConvertPtoLoadScalarOp,
               ConvertPtoStoreScalarOp>(typeConverter, context);
  patterns.add<ConvertVPTOUnrealizedCastOp>(typeConverter, context);
  patterns.add<ConvertVPTOTypedCarrierOp>(typeConverter, context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    diagOS << "VPTO LLVM emission failed: VPTO type lowering failed\n";
    return failure();
  }
  foldVPTOTypeCasts(module, typeConverter);
  return success();
}

static Type normalizeTypeForOfficialLLVMLowering(Type type, Builder &builder) {
  type = convertVPTOType(type, builder);
  return type;
}

static void normalizeFuncSignaturesForOfficialLLVMLowering(ModuleOp module) {
  Builder builder(module.getContext());

  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    FunctionType oldType = funcOp.getFunctionType();
    SmallVector<Type> newInputs;
    SmallVector<Type> newResults;
    bool changed = false;

    for (Type input : oldType.getInputs()) {
      Type normalized = normalizeTypeForOfficialLLVMLowering(input, builder);
      changed |= (normalized != input);
      newInputs.push_back(normalized);
    }
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

template <typename EmitFn>
static LogicalResult runPipeline(ModuleOp module, llvm::raw_ostream &diagOS,
                                 const VPTOEmissionOptions &options,
                                 EmitFn &&emit) {
  OwningOpRef<Operation *> clonedOp(module->clone());
  ModuleOp clonedModule = cast<ModuleOp>(*clonedOp);

  materializeVecScopeCarrierLoops(clonedModule);

  if (failed(normalizePtoMemRefSpaces(clonedModule, diagOS))) {
    diagOS << "VPTO LLVM emission failed: normalizePtoMemRefSpaces failed\n";
    return failure();
  }
  if (failed(lowerVPTOOps(clonedModule, diagOS))) {
    diagOS << "VPTO LLVM emission failed: lowerVPTOOps failed\n";
    return failure();
  }
  if (failed(lowerVPTOTypes(clonedModule, diagOS))) {
    diagOS << "VPTO LLVM emission failed: lowerVPTOTypes failed\n";
    return failure();
  }

  normalizeFuncSignaturesForOfficialLLVMLowering(clonedModule);

  PassManager pm(clonedModule.getContext());
  pm.enableVerifier();
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  if (failed(pm.run(clonedModule))) {
    diagOS << "VPTO LLVM emission failed: official lowering pipeline failed\n";
    return failure();
  }

  if (failed(applyQueriedTargetAttrs(clonedModule, options, diagOS)))
    return failure();

  llvm::LLVMContext llvmContext;
  registerBuiltinDialectTranslation(*clonedModule.getContext());
  registerLLVMDialectTranslation(*clonedModule.getContext());
  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(clonedModule.getOperation(), llvmContext);
  if (!llvmModule) {
    diagOS << "VPTO LLVM emission failed: LLVM IR export failed\n";
    return failure();
  }

  if (failed(attachAIVectorScopeMetadata(*llvmModule, diagOS)))
    return failure();
  attachHIVMKernelAnnotations(*llvmModule);
  llvmModule->setModuleIdentifier("ptoas.hivm.official");
  llvmModule->setSourceFileName("ptoas.hivm.official");
  return emit(*llvmModule);
}

} // namespace

LogicalResult
translateVPTOModuleToLLVMText(ModuleOp module, llvm::raw_ostream &os,
                              const VPTOEmissionOptions &options,
                              llvm::raw_ostream &diagOS) {
  return runPipeline(module, diagOS, options, [&](llvm::Module &llvmModule) {
    llvmModule.print(os, nullptr);
    return success();
  });
}

LogicalResult
translateVPTOModuleToLLVMBitcode(ModuleOp module, llvm::raw_ostream &os,
                                 const VPTOEmissionOptions &options,
                                 llvm::raw_ostream &diagOS) {
  return runPipeline(module, diagOS, options, [&](llvm::Module &llvmModule) {
    llvm::WriteBitcodeToFile(llvmModule, os);
    return success();
  });
}

} // namespace mlir::pto
