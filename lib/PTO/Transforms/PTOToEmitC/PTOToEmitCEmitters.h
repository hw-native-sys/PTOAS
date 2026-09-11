// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitC.cpp - PTO to EmitC conversion pass ----------------------===//
//===----------------------------------------------------------------------===//

//===- PTOToEmitCEmitters.h - shared helpers for PTO->EmitC ------------===//
//===----------------------------------------------------------------------===//
//
// Internal header shared by the PTOToEmitC*.cpp translation units that
// together implement the PTO-to-EmitC conversion pass. Not part of any
// public API; do not include this outside lib/PTO/Transforms/PTOToEmitC/.
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef DEBUG_TYPE
#define DEBUG_TYPE "pto-emitc"
#endif

#include <cassert>

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOLayoutUtils.h"
#include "PTO/IR/PTOSyncUtils.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/MemoryConsistencyAttrs.h"
#include "../Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
namespace pto {

using namespace mlir;
using namespace mlir::pto;

// Build the <Algorithm::HIGH_PRECISION> template-args shared by every
// precision-tunable math lowering (Recip/Rem/Fmod/Pow/Rsqrt/Sqrt/Div and
// friends). Returns a null ArrayAttr for the default precision so the plain
// call overload is emitted.
template <typename Precision>
inline ArrayAttr buildPrecisionTemplateArgs(Builder &builder,
                                            Precision precision, Precision def,
                                            const char *algorithm) {
  if (precision == def)
    return ArrayAttr{};
  return builder.getArrayAttr(
      {emitc::OpaqueAttr::get(builder.getContext(),
                              (llvm::Twine("pto::") + algorithm +
                               "::HIGH_PRECISION")
                                  .str())});
}

// Three-input arithmetic ops without a native pto-isa implementation
// (TADDC/TSUBC/TSUBSC) decompose into two opaque calls:
// firstOp(dst, a, b) then TADD(dst, dst, c).
inline void emitDecomposedPairAndErase(Operation *op,
                                        ConversionPatternRewriter &rewriter,
                                        StringRef firstOp, Value dst, Value a,
                                        Value b, Value c) {
  auto loc = op->getLoc();
  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, firstOp, ArrayAttr{},
                                       ArrayAttr{}, ValueRange{dst, a, b});
  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TADD", ArrayAttr{},
                                       ArrayAttr{}, ValueRange{dst, dst, c});
  rewriter.eraseOp(op);
}

// Coerce a pointer-like value to a uint64_t address: FFTs-style pointer types
// go through reinterpret_cast<uint64_t>, other types through a plain cast.
inline Value coerceToU64Address(ConversionPatternRewriter &rewriter,
                                Location loc, Value value, Type u64Ty) {
  auto *ctx = rewriter.getContext();
  if (isa<emitc::PointerType>(value.getType()) ||
      (isa<emitc::OpaqueType>(value.getType()) &&
       cast<emitc::OpaqueType>(value.getType()).getValue().ends_with("*"))) {
    auto rcU64 =
        rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
    return rewriter
        .create<emitc::CallOpaqueOp>(loc, u64Ty, "reinterpret_cast",
                                     ArrayAttr{}, rcU64, ValueRange{value})
        .getResult(0);
  }
  if (value.getType() != u64Ty)
    return rewriter.create<emitc::CastOp>(loc, u64Ty, value).getResult();
  return value;
}

inline constexpr llvm::StringLiteral kGlobalTensorStridesAttrName =
    "__pto.globaltensor_strides";
inline constexpr llvm::StringLiteral kPipePeerOwnerFuncAttrName =
    "__pto.peer_owner_func";
inline constexpr llvm::StringLiteral kPipePeerReserveNameAttrName =
    "__pto.peer_reserve_name";
inline constexpr llvm::StringLiteral kPipePeerDirMaskAttrName =
    "__pto.peer_dir_mask";
inline constexpr llvm::StringLiteral kEmitCScalarOutTypeAttrName =
    "__pto.emitc_scalar_out_type";
inline constexpr llvm::StringLiteral kLastUseAttrName = "pto.last_use";
inline constexpr llvm::StringLiteral kLastUseMarkerPrefix = "PTOAS__LAST_USE__";
inline constexpr unsigned kPTOIndexBitWidth =
    64; // keep consistent with IndexType conversion
inline constexpr llvm::StringLiteral kAutoSyncTailPendingModeAttr =
    "__pto.auto_sync_tail_mode";
inline constexpr llvm::StringLiteral kAutoSyncTailBarrierAttr =
    "pto.auto_sync_tail_barrier";
inline constexpr llvm::StringLiteral kAutoSyncTailHintAttr =
    "pto.auto_sync_tail_hint";
inline constexpr llvm::StringLiteral kAutoSyncTailPolicyBarrierAll =
    "barrier_all";
inline constexpr llvm::StringLiteral kAutoSyncTailPolicyMte3ToSEvent0 =
    "setwait_mte3_to_s_event0";
inline constexpr llvm::StringLiteral kAutoSyncTailModeBarrierAllToken =
    "PTOAutoSyncTailMode::kBarrierAll";
inline constexpr llvm::StringLiteral kAutoSyncTailModeMte3ToSEvent0Token =
    "PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0";

struct InterCoreSyncCallDesc {
  const char *callee = nullptr;
  ArrayAttr args;
  SmallVector<Value, 2> operands;
};

struct GlobalTensorTypeNames {
  std::string shapeTypeName;
  std::string strideTypeName;
  std::string tensorTypeName;
  std::string layoutConstName;
};

struct SpecialGlobalTensorTypeSpec {
  std::string shapeTypeExpr;
  std::string strideTypeExpr;
  std::string layoutEnum;
};

//---- shared helper declarations (definitions in PTOToEmitC*.cpp) --------//
FailureOr<Value> adaptCallOperandForEmitC(const TypeConverter *typeConverter, ConversionPatternRewriter &rewriter, Location loc, Type originalCalleeArgTy, Value originalOperand, Value loweredOperand) ;
const char *addrSpaceQualifier(pto::AddressSpace as) ;
void appendRawLocationNameHints(Location loc, SmallVectorImpl<std::string> &hints) ;
Value applyStaticMemrefOffset(ConversionPatternRewriter &rewriter, Location loc, Value basePtr, int64_t offset) ;
ArrayAttr buildAccPhaseTemplateArgs(ConversionPatternRewriter &rewriter, pto::AccPhase phase) ;
FailureOr<Value> buildAsyncScratchTileValue( ConversionPatternRewriter &rewriter, Location loc, Value originalScratch, Value emittedScratch) ;
FailureOr<Value> buildCollectiveParallelGroup( ConversionPatternRewriter &rewriter, Location loc, ArrayRef<Value> groupGTs, int64_t root) ;
FailureOr<Value> buildCommGlobalTensorValue( ConversionPatternRewriter &rewriter, Location loc, Value originalValue, Value emittedValue, Operation *anchor) ;
template <typename OpTy>
FailureOr<SmallVector<Value>> buildCommGroupGlobalTensors(
    ConversionPatternRewriter &rewriter, Location loc, OpTy op,
    ValueRange originalGroup, ValueRange emittedGroup);
FailureOr<Value> buildCommTileValue(ConversionPatternRewriter &rewriter, Location loc, Value originalValue, Value emittedValue) ;
SmallVector<unsigned, 4> buildDefaultLastUseTileSlotOrder(Operation *op) ;
FailureOr<std::string> buildEmitCOpaqueConstantLiteral(Type targetType, Attribute valueAttr) ;
std::string buildFixpipeConfigAliasName(int32_t pipeId) ;
FailureOr<std::string> buildFixpipeConfigTypeToken(AccPushEpilogueAttr accPushEpilogue) ;
Value buildGlobalTensorFromMemref(ConversionPatternRewriter &rewriter, Location loc, Value basePtr, MemRefType mrTy, Operation *anchor, StringRef tag = {});
void buildGlobalTensorShapeAndStride(ArrayRef<int64_t> shape, ArrayRef<int64_t> strides, SmallVectorImpl<int64_t> &shape5D, SmallVectorImpl<int64_t> &stride5D) ;
FailureOr<Value> buildGlobalTensorViewFromPointer(
    ConversionPatternRewriter &rewriter, Location loc, Value ptr, Type elemTy,
    ArrayRef<int64_t> shape, ArrayRef<int64_t> strides = {},
    std::optional<SpecialGlobalTensorTypeSpec> specialSpec = std::nullopt,
    StringRef layoutEnum = "pto::Layout::ND");
InterCoreSyncCallDesc buildInterCoreSyncSetCall( ConversionPatternRewriter &rewriter, Location loc, PTOArch targetArch, pto::PipeAttr pipeAttr, IntegerAttr eventIdAttr, int64_t fftsMode) ;
InterCoreSyncCallDesc buildInterCoreSyncSetCallDyn( ConversionPatternRewriter &rewriter, Location loc, PTOArch targetArch, pto::PipeAttr pipeAttr, Value eventIdVal, int64_t fftsMode) ;
InterCoreSyncCallDesc buildInterCoreSyncWaitCall( ConversionPatternRewriter &rewriter, PTOArch targetArch, pto::PipeAttr pipeAttr, IntegerAttr eventIdAttr) ;
InterCoreSyncCallDesc buildInterCoreSyncWaitCallDyn( ConversionPatternRewriter &rewriter, Location loc, PTOArch targetArch, pto::PipeAttr pipeAttr, Value eventIdVal) ;
std::optional<std::string> buildLastUseMarkerCallee(Operation *op, StringRef callee, ArrayRef<unsigned> tileSlotOrder = {}) ;
SmallVector<int64_t> buildRowMajorStrides(ArrayRef<int64_t> shape) ;
FailureOr<Value> buildRuntimeGlobalTensor( ConversionPatternRewriter &rewriter, Location loc, Value ptr, Type elemTy, ArrayRef<int64_t> staticShape, ValueRange runtimeShape, ValueRange runtimeStrides, StringRef layoutEnum = "pto::Layout::ND") ;
FailureOr<Value> buildSyncAllGlobalTensorFromPointer( ConversionPatternRewriter &rewriter, Location loc, Value ptr, Type elemTy) ;
std::string buildTPipeToken(int32_t flagBase, llvm::StringRef dirTok, int32_t slotSize, int32_t slotNum, int32_t localSlotNum, bool nosplit) ;
FailureOr<std::string> buildTPipeTokenFromInitOp(Operation *op, PTOArch targetArch) ;
Value castInterCoreEventIdToI32(ConversionPatternRewriter &rewriter, Location loc, Value eventId) ;
Value castSignlessIntToUnsignedSameWidth(ConversionPatternRewriter &rewriter, Location loc, Value v, unsigned bitWidth) ;
Value castToGMBytePointer(ConversionPatternRewriter &rewriter, Location loc, Value value) ;
Value castViewIndexToEmitC(ConversionPatternRewriter &rewriter, Location loc, Value value) ;
void collectStructTypes(Type t, llvm::SetVector<pto::StructType> &out) ;
SmallVector<unsigned, 4> collectTileOperandNumbers(Operation *op) ;
Value createFFTSMsg(ConversionPatternRewriter &rewriter, Location loc, Value eventId, int64_t fftsMode) ;
void createOpaqueCall(ConversionPatternRewriter &rewriter, Location loc,
                      TypeRange resultTypes, StringRef callee, ArrayAttr args,
                      ArrayAttr templateArgs, ValueRange operands);
void createLastUseAwareOpaqueCall(
    ConversionPatternRewriter &rewriter, Operation *op, TypeRange resultTypes,
    StringRef callee, ValueRange operands, ArrayAttr args = ArrayAttr{},
    ArrayAttr templateArgs = ArrayAttr{},
    ArrayRef<unsigned> tileSlotOrder = {});
Value emitCCast(ConversionPatternRewriter &rewriter, Location loc, Type dstType, Value src) ;
void emitConservativeGmFencePipeDrains( ConversionPatternRewriter &rewriter, Location loc) ;
void emitDsbDdr(ConversionPatternRewriter &rewriter, Location loc) ;
void emitInvalidateGmCacheAll(ConversionPatternRewriter &rewriter, Location loc) ;
void emitInvalidateGmCacheSingleLine(ConversionPatternRewriter &rewriter, Location loc, Value addr) ;
void emitPipeBarrier(ConversionPatternRewriter &rewriter, Location loc, StringRef pipeTok) ;
void emitTNotifyReleaseActions(ConversionPatternRewriter &rewriter, Location loc, bool drainMte2, bool drainMte3) ;
void eraseDeadPureEmitCValueOps(ModuleOp module) ;
std::string evtTokFromEventAttr(mlir::pto::EventAttr a);
std::string maskPatternTok(mlir::pto::MaskPatternAttr a);
[[maybe_unused]] std::string evtTokFromEventEnum(mlir::pto::EVENT e);
LogicalResult extractSyncTripletTokens(Operation *op, std::string &srcTok,
                                       std::string &dstTok, std::string &evtTok,
                                       ConversionPatternRewriter &rewriter);
FailureOr<Operation *> findPeerFixpipeConsumerInit(Operation *producerInit) ;
int64_t getAPIntSignedValue(const APInt &value) ;
uint64_t getAPIntUnsignedValue(const APInt &value) ;
pto::AddressSpace getAddressSpaceOrGM(Attribute memorySpace) ;
std::string getAutoSyncTailModeToken(Operation *op) ;
std::string getElemTypeStringForGT(Type elemTy) ;
emitc::PointerType getEmitCPointerType(MLIRContext *ctx,
                                      StringRef pointeeTypeStr);
emitc::PointerType getEmitCPointerType(MLIRContext *ctx, StringRef qualifier,
                                      StringRef elemTypeStr);
int64_t getEmitCScalarByteWidth(Type elemTy) ;
std::string getEmitCScalarTypeToken(Type elemTy) ;
std::optional<std::string> getEmitCTileTypeString(pto::TileBufType type) ;
Type getEmitCVariableResultType(Type valueType) ;
Attribute getFFTSModeCodegenArg(ConversionPatternRewriter &rewriter, int64_t fftsMode) ;
FailureOr<std::string> getFixpipeLayoutToken(FixpipeLayout layout) ;
FailureOr<std::string> getFixpipeQuantToken(FixpipeQuant quant) ;
FailureOr<std::string> getFixpipeReluToken(FixpipeRelu relu) ;
int getGlobalTensorElementBytes(Type elemTy) ;
std::string getGlobalTensorTypeStringFromShapeAndStrides(
    Type elemTy, ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
    StringRef layoutEnum = "pto::Layout::ND");
Location getIndexedNameHintLoc(Location fallbackLoc, unsigned index) ;
int64_t getIntegerAttrSignedValue(IntegerAttr attr) ;
std::optional<StringRef> getKernelKindMacro(func::FuncOp funcOp) ;
StringRef getLastUseAwareCallee(Operation *op, StringRef callee, std::string &storage, ArrayRef<unsigned> tileSlotOrder = {}) ;
std::optional<mlir::pto::Layout> getLayoutAttrFromOp(Operation *op) ;
std::optional<mlir::pto::Layout> getLayoutAttrFromViewType(Type type) ;
FailureOr<std::string> getPipeDataTypeToken(Value value) ;
Type getPointerLikeElementType(Type type) ;
Value getRuntimeGlobalTensorMetadata( ConversionPatternRewriter &rewriter, Location loc, Value tensor, Value logicalDim, int64_t rank, bool isStride) ;
emitc::OpaqueType getRuntimeGlobalTensorOpaqueType( MLIRContext *ctx, Type elemTy, ArrayRef<int64_t> shape, StringRef layoutEnum) ;
emitc::OpaqueType getSignedIntOpaqueType(MLIRContext *ctx, unsigned bitWidth) ;
Value getSourceEmitCVariable(Value value) ;
std::optional<SpecialGlobalTensorTypeSpec> getSpecialGlobalTensorTypeSpecForLayout(std::optional<mlir::pto::Layout> layout, ArrayRef<int64_t> shape, Type elemTy) ;
std::optional<SpecialGlobalTensorTypeSpec> getSpecialScaleGlobalTensorTypeSpec(Operation *anchor, MemRefType mrTy) ;
std::optional<SpecialGlobalTensorTypeSpec> getSpecialScaleGlobalTensorTypeSpecForTileValue(Value dstValue, ArrayRef<int64_t> shape, Type elemTy) ;
std::optional<int64_t> getStaticIndexLikeValue(Value value) ;
bool getStaticMemrefLayout(MemRefType mrTy, SmallVectorImpl<int64_t> &strides, int64_t &offset) ;
LogicalResult getStaticTensorViewStrides( Value source, Value convertedSource, int64_t rank, SmallVectorImpl<int64_t> &strides) ;
std::string getStructTypeName(pto::StructType st) ;
FailureOr<std::string> getTPipeTokenFromValue(Value pipeHandle, PTOArch targetArch) ;
pto::BLayout getTileBufBLayoutValue(pto::TileBufConfigAttr configAttr) ;
pto::SLayout getTileBufSLayoutValue(pto::TileBufConfigAttr configAttr) ;
Type getTileDataResultType(MLIRContext *ctx, pto::AddressSpace as, StringRef elemTok) ;
FailureOr<std::string> getTileSplitToken(int64_t split) ;
emitc::OpaqueType getUnsignedIntOpaqueType(MLIRContext *ctx, unsigned bitWidth) ;
emitc::OpaqueType getWiderSignedIntOpaqueType(MLIRContext *ctx, unsigned bitWidth) ;
emitc::OpaqueType getWiderUnsignedIntOpaqueType(MLIRContext *ctx, unsigned bitWidth) ;
bool hasInterCoreSyncOp(func::FuncOp func) ;
bool hasSetFFTsOp(func::FuncOp func) ;
bool hasStaticShape(MemRefType mrTy) ;
std::string inferFallbackGlobalTensorLayout(ArrayRef<int64_t> shape, ArrayRef<int64_t> strides, Type elemTy) ;
const char *inferScalingRoleFromValue(Value value) ;
LogicalResult insertFixpipeConfigAliases(ModuleOp mop) ;
bool isDpsInitOperand(OpOperand &operand) ;
bool isEmitCGlobalTensorLikeType(Type ty) ;
bool isEmitCPointerLikeType(Type ty) ;
bool isEmitCTileLikeType(Type ty) ;
bool isF8E8M0ElemType(Type elemTy) ;
bool isGmCmoSpace(pto::AddressSpace space) ;
bool isInVectorKernel(Operation *op) ;
bool isLowPrecisionCubeOperandType(Type elemTy) ;
bool isSetFFTsPointerLikeType(Type ty) ;
bool isTriviallyInlineableExecuteRegion(scf::ExecuteRegionOp op) ;
std::string joinIntTemplateParams(ArrayRef<int64_t> values) ;
std::string layoutToEmitCString(mlir::pto::Layout layout) ;
Value loadEmitCVariableIfNeeded(OpBuilder &builder, Location loc, Value value) ;
Value makeEmitCIntConstant(ConversionPatternRewriter &rewriter, Location loc, Type type, int64_t value) ;
Value makeEmitCOpaqueConstant(ConversionPatternRewriter &rewriter, Location loc, Type type, llvm::StringRef literal) ;
Value makeViewIndexConstant(ConversionPatternRewriter &rewriter, Location loc, int64_t value) ;
std::string mangleStructFieldType(Type t) ;
Value materializeAddressAsPointer(ConversionPatternRewriter &rewriter, Location loc, Value addr, pto::AddressSpace as, StringRef elemTok) ;
Value materializeGlobalTensorDataPointer( ConversionPatternRewriter &rewriter, Location loc, Value value, Type sourceType) ;
Value materializeTileDataValue(ConversionPatternRewriter &rewriter, Location loc, Value tile, pto::AddressSpace as, StringRef elemTok) ;
Value maybeWrapGlobalMemrefAsGlobalTensor( ConversionPatternRewriter &rewriter, Location loc, Value loweredValue, Type originalType, Operation *anchor, StringRef tag = {});
int64_t multiplyOrDynamic(int64_t lhs, int64_t rhs) ;
bool isA5NoSplitPipeOp(Operation *op);
bool hasExplicitSubblockControl(Operation *op);
bool needsA5NoSplitVectorGuard(Operation *op);
bool needsWholeFunctionSCFToCF(func::FuncOp func) ;
std::string notifyOpTok(pto::NotifyOp op) ;
bool parseIntegerTemplateList(StringRef token, StringRef marker, SmallVectorImpl<int64_t> &values) ;
bool partitionViewHasStaticResultShape(pto::PartitionViewOp op) ;
Value peelGlobalTensorConversionBridge(Value value) ;
std::string pipeTokFromPipeAttr(mlir::pto::PipeAttr a) ;
std::string pipeTokFromPipeEnum(mlir::pto::PIPE p) ;
std::string reduceOpTok(pto::ReduceOp op) ;
LogicalResult rematerializeFixpipeQuantBindings(ModuleOp mop) ;
std::string renderStructDef(pto::StructType st) ;
std::string renderStructFieldDecl(Type fieldTy, const std::string &name) ;
int64_t renderTileTemplateDim(int64_t rawDim, Type elemTy, pto::BLayout blayout, int dimIdx) ;
void replaceOrEraseWithOpaqueCallAndReturnDst(Operation *op, Value dst, StringRef callee, ArrayRef<Value> args, ArrayAttr templateArgs, ConversionPatternRewriter &rewriter) ;
FailureOr<TileBufType> resolveFixpipeConsumerTileType(Value pipeHandle) ;
std::string resolveGlobalTensorLayout(Operation *anchor, Value basePtr, ArrayRef<int64_t> shape, ArrayRef<int64_t> strides, Type elemTy) ;
std::optional<mlir::pto::Layout> resolveLayoutForGlobalTensor(Operation *anchor, Value basePtr) ;
std::optional<mlir::pto::Layout> resolveLayoutFromValueChain(Value v) ;
std::string sanitizeIdentifier(std::string s) ;
const char *scalingRoleToken(Type elemTy, pto::TileBufConfigAttr configAttr) ;
std::string tileBufBLayoutToken(pto::TileBufConfigAttr configAttr) ;
std::string tileBufCompactToken(pto::TileBufConfigAttr configAttr) ;
std::string tileBufPadToken(pto::TileBufConfigAttr configAttr) ;
std::string tileBufSLayoutToken(pto::TileBufConfigAttr configAttr) ;
bool tileDataReturnsIntegralAddress(pto::AddressSpace as) ;
const char *tileRoleToken(Attribute memorySpace, std::optional<Type> elemType = std::nullopt, std::optional<pto::TileBufConfigAttr> configAttr = std::nullopt) ;
std::string waitCmpTok(pto::WaitCmp cmp) ;

class PTOToEmitCTypeConverter : public TypeConverter {
public:
  PTOToEmitCTypeConverter(MLIRContext *Ctx, PTOArch targetArch);

private:
  // Register the scalar/builtin -> EmitC type conversions (f32, i32, index,
  // vectors and low-precision floats).
  void registerBasicConversions(MLIRContext *Ctx);
  void registerFloatConversions(MLIRContext *Ctx);
  void registerIntegerConversions(MLIRContext *Ctx);
  // Register conversions for PTO-specific value types (tile buffers, pipes,
  // views, structs, ...).
  void registerPTOValueConversions(MLIRContext *Ctx);
  void registerPointerAndStructConversions(MLIRContext *Ctx);
  void registerRuntimeValueConversions(MLIRContext *Ctx);
  void registerViewAndHandleConversions(MLIRContext *Ctx);
  void registerTileConversions(MLIRContext *Ctx);
  // Register the memref -> address-space-qualified pointer conversion.
  void registerMemRefConversions(MLIRContext *Ctx);
  // Register function-type conversion and source/target materializations.
  void registerFunctionAndMaterializations();
};;

enum class Role { A, B, C, Unknown };

template <typename MatmulLikeOp>
static std::optional<Role> inferMatmulLikeSubviewRole(MatmulLikeOp op,
                                                      Value buffer) {
  if (op.getLhs() == buffer)
    return Role::A;
  if (op.getRhs() == buffer)
    return Role::B;
  return std::nullopt;
}

static std::optional<Role> inferSubviewRoleFromLoadUser(mlir::pto::TLoadOp load) {
  Value buffer = load.getDst();
  if (!buffer)
    return std::nullopt;
  for (Operation *user : buffer.getUsers()) {
    if (auto matmul = dyn_cast<mlir::pto::TMatmulOp>(user)) {
      if (auto role = inferMatmulLikeSubviewRole(matmul, buffer))
        return role;
      continue;
    }
    if (auto matmulAcc = dyn_cast<mlir::pto::TMatmulAccOp>(user)) {
      if (auto role = inferMatmulLikeSubviewRole(matmulAcc, buffer))
        return role;
    }
  }
  return std::nullopt;
}

static std::optional<Role> inferSubviewRoleFromUser(Operation *user, Value result) {
  if (auto load = dyn_cast<mlir::pto::TLoadOp>(user))
    return inferSubviewRoleFromLoadUser(load);
  if (auto store = dyn_cast<mlir::pto::TStoreOp>(user)) {
    if (store.getDst() == result)
      return Role::C;
  }
  return std::nullopt;
}

[[maybe_unused]] static Role inferSubviewRole(memref::SubViewOp sv) {
  Value result = sv.getResult();
  for (Operation *user : result.getUsers()) {
    if (auto role = inferSubviewRoleFromUser(user, result))
      return *role;
  }
  return Role::Unknown;
}


} // namespace pto
} // namespace mlir

