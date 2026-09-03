// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOSyncUtils.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VPTOMemoryDist.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VPTOLLVMEmitter.h"
#include "PTO/Transforms/VPTOLLVMEmitterHelper.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

namespace mlir::pto {

void materializeVecScopeCarrierLoops(ModuleOp module);
LogicalResult applyQueriedTargetAttrs(ModuleOp module, const VPTOEmissionOptions &options, llvm::raw_ostream &diagOS);
LogicalResult attachAIVectorScopeMetadata(llvm::Module &llvmModule, llvm::raw_ostream &diagOS);
void attachHIVMKernelAnnotations(llvm::Module &llvmModule, ModuleOp sourceModule);

namespace detail {

inline constexpr llvm::StringLiteral kVectorSuffix = "_mix_aiv";
inline constexpr llvm::StringLiteral kCubeSuffix = "_mix_aic";

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
  F8E4M3,
  F8E5M2,
  HiF8,
  F4E1M2x2,
  F4E2M1x2,
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
  bool satBeforeRnd = false;
};

struct MadCalleeContract {
  StringRef lhs;
  StringRef rhs;
  StringRef dst;
  StringRef callee;
};

struct LowpPayloadABI {
  Type llvmElementType;
  StringRef intrinsicElementFragment;
};

Type convertVPTOType(Type type, Builder &builder);
Value materializeVPTOCast(OpBuilder &builder, Type resultType, ValueRange inputs, Location loc);

class VPTOTypeConverter final : public TypeConverter {
public:
  explicit VPTOTypeConverter(MLIRContext *context) {
    addConversion([](Type type) { return type; });
    addConversion([](Type type) -> Type {
      Builder builder(type.getContext());
      return convertVPTOType(type, builder);
    });
    addSourceMaterialization(materializeVPTOCast);
    addTargetMaterialization(materializeVPTOCast);
  }
};

Type getLowPrecisionLLVMType(Type type, MLIRContext *context);
bool isLLVMExtensionVectorElementType(Type type);
Type getLLVMCompatibleVectorType(ArrayRef<int64_t> shape, Type elementType, ArrayRef<bool> scalableDims);
Type normalizePayloadTypeForLLVMLowering(Type type, Builder &builder);
Type normalizeGEPElementTypeForLLVMLowering(Type type, Builder &builder);
Type convertVPTOType(Type type, Builder &builder);
unsigned getNaturalByteAlignment(Type type);
bool hasVPTOConvertibleType(Type type);
bool hasVPTOConvertibleType(TypeRange types);
Value materializeVPTOCast(OpBuilder &builder, Type resultType, ValueRange inputs, Location loc);
LLVM::LLVMStructType getVPTOStructStorageType(pto::StructType structType, Builder &builder);
FailureOr<Value> getVPTOStructFieldAddress(ConversionPatternRewriter &rewriter, Location loc, Value root,
                                           pto::StructType rootType, ArrayRef<int64_t> path);
Value getI64Constant(OpBuilder &builder, Location loc, uint64_t value);
Value getI32Constant(OpBuilder &builder, Location loc, uint64_t value);
Value getI1Constant(OpBuilder &builder, Location loc, bool value);
bool isMxElementType(Type ty);
std::string getMadMxElementFragment(Type type);
FailureOr<StringRef> buildMadMxCalleeName(MLIRContext *context, Type lhsElem, Type rhsElem);
bool isSignedOrSignlessInteger(IntegerType intType, unsigned width);
std::string getMadRhsFragment(Type type);
bool isMadE4M3ElementType(Type type);
bool isMadE5M2ElementType(Type type);
std::string getMadDstFragment(Type type);
ArrayRef<MadCalleeContract> getMadCalleeContracts();
std::string getMadLhsFragment(Type type);
FailureOr<StringRef> buildMadTypedCalleeName(MLIRContext *context, Type lhsElem, Type rhsElem, Type dstElem);
FailureOr<StringRef> buildLaneTypedCallee(MLIRContext *context, Type resultType, StringRef stem, StringRef suffix);
std::string getCANN900VectorElementFragment(Type type);
std::string getCANN900VectorTypeFragment(Type vectorType);
std::string getCANN900SignednessFragment(Type elemType);
FailureOr<StringRef> buildCANN900ModeTypedCallee(MLIRContext *context, Type vectorType, StringRef stem, StringRef mode);
FailureOr<StringRef> buildCANN900SignedModeTypedCallee(MLIRContext *context, Type vectorType, StringRef stem,
                                                       StringRef mode);
FailureOr<StringRef> buildCANN900WideningReductionCallee(MLIRContext *context, Type inputType, Type resultType,
                                                         StringRef stem, StringRef mode);
std::string getElementTypeFragment(Type type);
std::string getLowPrecisionElementFragment(Type type);
std::string getMemoryElementTypeFragment(Type type);
bool isLowpPayloadElementType(Type type);
std::optional<LowpPayloadABI> getLowpPayloadABI(Type elementType, MLIRContext *context);
std::string getDirectLowpVLogicElementFragment(Type type);
FailureOr<StringRef> buildDirectLowpVLogicCallee(MLIRContext *context, Type vectorType, StringRef stem, StringRef mode);
FailureOr<StringRef> buildLowpPayloadVLogicCallee(MLIRContext *context, Type vectorType, StringRef stem,
                                                  StringRef mode);
Type getLowpPayloadCarrierType(Type vectorLikeType, MLIRContext *context);
Type getPayloadABIType(Type semanticType, Type convertedType, MLIRContext *context);
Value castToPayloadABI(Location loc, Value value, Type semanticType, ConversionPatternRewriter &rewriter);
Value castFromPayloadABI(Location loc, Value value, Type semanticType, Type convertedType,
                         ConversionPatternRewriter &rewriter);
std::string getAtomicElementTypeFragment(Type type, Attribute signednessAttr);
std::string getL0LoadElementFragment(Type type);
std::string getShuffleIntrinsicTypeFragment(Type type);
std::string getReduxIntrinsicTypeFragment(Type type, Attribute signednessAttr);
Type getElementTypeFromVectorLike(Type type);
std::optional<int64_t> getElementCountFromVectorLike(Type type);
Value castIntegerLikeTo(Operation *anchor, Value value, Type targetType);
FailureOr<Value> reinterpretPointerToAddrSpace(Operation *anchor, Value value, unsigned targetAddressSpace);
FailureOr<Value> normalizeVdupScalarOperand(OpBuilder &builder, Location loc, Value input, Type resultType);
Value normalizeByteScalarOperandForCANN900VectorCall(OpBuilder &builder, Location loc, Value input,
                                                     Type semanticElementType);
bool isCompatibleScalarForSemanticType(Type semanticType, Type scalarType);
std::string getCopyElementFragment(Type elementType);
std::string getNd2NzCopyElementFragment(Type elementType);
std::optional<uint64_t> parsePredicatePatternImmediate(StringRef pattern);
std::optional<uint64_t> parseHiLoPartImmediate(StringRef part);
std::optional<uint64_t> parseRoundModeImmediate(StringRef roundMode);
std::optional<uint64_t> parseSaturationImmediate(StringRef sat);
std::optional<uint64_t> parsePartImmediate(StringRef part);
std::optional<uint64_t> parseVcvtPartImmediate(StringRef part);
std::optional<uint64_t> parsePredicateStoreDistImmediate(StringRef dist);
std::optional<uint64_t> parsePredicateLoadDistImmediate(StringRef dist);
std::optional<int32_t> parsePostModeImmediate(StringRef mode);
std::optional<uint64_t> parsePipeImmediate(StringRef pipe);
std::optional<uint64_t> parseEventImmediate(StringRef event);
std::optional<uint64_t> parseSprImmediate(StringRef spr);
std::optional<unsigned> getDistElementWidth(Type type);
VcvtElemKind classifyVcvtElemType(Type type);
std::optional<VcvtContract> lookupVcvtContract(VcvtElemKind src, VcvtElemKind dst);
uint64_t determineVsqzStoreHint(pto::VsqzOp vsqz);
std::optional<uint64_t> parseLoadDistImmediate(StringRef dist, Type elementType);
FailureOr<Value> packShiftedFields(Operation *anchor, Value base, ArrayRef<std::pair<Value, uint64_t>> fields);
std::optional<uint64_t> parseLoadX2DistImmediate(StringRef dist, Type elementType);
std::optional<uint64_t> parseStoreDistImmediate(StringRef dist, Type elementType);
bool isOnePointStoreDist(StringRef dist);
bool isMaskOnlyUsedByOnePointStores(Value mask);
std::optional<uint64_t> parseStoreX2DistImmediate(StringRef dist, Type elementType);
Value packBlockRepeatStride(Operation *anchor, Value blockStride, Value repeatStride);
std::optional<uint64_t> parseOrderImmediate(StringRef order);
FailureOr<Value> packLoopPair(Operation *anchor, Value low, Value high);
FailureOr<Value> packLoopSize(Operation *anchor, Value loop2, Value loop1);
FailureOr<Value> packCopyGmToUbConfig0(Operation *anchor, ValueRange operands);
FailureOr<Value> packCopyGmToUbConfig1(Operation *anchor, ValueRange operands);
FailureOr<Value> packCopyGmToUbConfig0(Operation *anchor, Value sid, Value nBurst, Value lenBurst, Value leftPadding,
                                       Value rightPadding, Value dataSelect, Value cacheCtl);
FailureOr<Value> packCopyUbToGmConfig0(Operation *anchor, ValueRange operands);
FailureOr<Value> packCopyUbToGmConfig1(Operation *anchor, ValueRange operands);
FailureOr<Value> packCopyUbToGmConfig0(Operation *anchor, Value sid, Value nBurst, Value lenBurst, Value l2CacheCtl);
FailureOr<Value> packCopyUbToUbConfig(Operation *anchor, ValueRange operands);
FailureOr<Value> packCopyCbufToUbConfig(Operation *anchor, ValueRange operands);
FailureOr<Value> packCopyUbToCbufConfig(Operation *anchor, ValueRange operands);
FailureOr<Value> packCopyGmToCbufConfig0(Operation *anchor, Value nBurst, Value lenBurst);
FailureOr<Value> packCopyGmToCbufConfig1(Operation *anchor, Value srcStride, Value dstStride);
FailureOr<Value> packCopyGmToCbufMultiConfig0(Operation *anchor, Value sid, Value loop1SrcStride, Value l2CacheCtl,
                                              Value nValue);
FailureOr<Value> packCopyGmToCbufMultiConfig1(Operation *anchor, Value dValue, Value loop4SrcStride, Value smallC0En);
FailureOr<Value> packCopyCbufToBtConfig(Operation *anchor, Value convControl, Value nBurst, Value lenBurst,
                                        Value sourceGap, Value dstGap);
FailureOr<Value> packCopyCbufToFbufConfig(Operation *anchor, Value nBurst, Value lenBurst, Value sourceGap,
                                          Value dstGap);
FailureOr<Value> packLoadCbufToS4Config0(Operation *anchor, Value mStart, Value kStart, Value mStep, Value kStep);
FailureOr<Value> packLoadCbufToS4Config1(Operation *anchor, Value srcStride, Value dstStride);
FailureOr<Value> packLoadCbufToCaConfig0(Operation *anchor, Value mStart, Value kStart, Value mStep, Value kStep);
FailureOr<Value> packLoadCbufToCaConfig1(Operation *anchor, Value srcStride, Value dstStride);
FailureOr<Value> packLoadCbufToCbConfig0(Operation *anchor, Value mStart, Value kStart, Value mStep, Value kStep);
FailureOr<Value> packLoadCbufToCbConfig1(Operation *anchor, Value srcStride, Value dstStride);
Value buildMadBiasDestination(Operation *anchor, ConversionPatternRewriter &rewriter, Value dst, Value bias);
FailureOr<Value> packVbitsortConfig(Operation *anchor, Value repeatTimes);
FailureOr<Value> materializeDynamicPltMask(ConversionPatternRewriter &rewriter, LoweringState &state, Location loc,
                                           Value laneCount, Type vectorElemType);
FailureOr<StringRef> buildCarryBinaryCallee(MLIRContext *context, Type resultType, StringRef stem);
FailureOr<StringRef> buildVselCallee(MLIRContext *context, Type resultType);
FailureOr<StringRef> buildVselrCallee(MLIRContext *context, Type resultType);
FailureOr<StringRef> buildVdupCallee(MLIRContext *context, pto::VdupOp op);
FailureOr<StringRef> buildVbrCallee(MLIRContext *context, Type resultType);
FailureOr<StringRef> buildPstuCallee(MLIRContext *context, pto::PstuOp op);
FailureOr<StringRef> buildVstusCallee(MLIRContext *context, Type valueType);
FailureOr<StringRef> buildVstusPostCallee(MLIRContext *context, Type valueType);
StringRef buildVsturCallee(MLIRContext *context);
StringRef buildInitAlignCallee(MLIRContext *context);
StringRef buildSprclrCallee(MLIRContext *context);
StringRef buildSprstiCallee(MLIRContext *context, bool post);
StringRef buildSprstsCallee(MLIRContext *context, bool post);
StringRef buildStoreVfSimtInfoCallee(MLIRContext *context);
StringRef buildSyncthreadsCallee(MLIRContext *context);
StringRef buildThreadfenceCallee(MLIRContext *context);
StringRef buildThreadfenceBlockCallee(MLIRContext *context);
StringRef buildVstarCallee(MLIRContext *context);
StringRef buildVstasCallee(MLIRContext *context, bool post);
Value buildShuffleControlValue(OpBuilder &builder, Location loc, Value controlValue, int64_t widthValue,
                               unsigned controlMask);
FailureOr<StringRef> buildAtomicCalleeName(MLIRContext *context, Type ptrType, Type valueType, Attribute signednessAttr,
                                           StringRef opName);
FailureOr<StringRef> buildL1CacheLoadCallee(MLIRContext *context, Type resultType, pto::L1Cache l1cache);
FailureOr<StringRef> buildL1CacheStoreCallee(MLIRContext *context, Type valueType, pto::L1Cache l1cache);
FailureOr<StringRef> buildMulhiCallee(MLIRContext *context, Type resultType, pto::Signedness signedness);
FailureOr<StringRef> buildMulI32ToI64Callee(MLIRContext *context, pto::Signedness signedness);
std::string getScalarFloatBuiltinFragment(Type type);
std::string getLLVMFloatBuiltinFragment(Type type);
std::string getHIVMFloatBuiltinFragment(Type type);
FailureOr<StringRef> buildSqrtCallee(MLIRContext *context, Type valueType);
std::string getScalarHIVMFloatShortFragment(Type type);
FailureOr<StringRef> buildFmaCallee(MLIRContext *context, Type valueType);
std::string getConvertScalarFragment(Type type, Attribute signednessAttr);
FailureOr<StringRef> buildConvertCallee(MLIRContext *context, Type srcType, Type dstType, Attribute signednessAttr);
FailureOr<StringRef> buildVldsPostCallee(MLIRContext *context, Type resultType);
FailureOr<StringRef> buildVstsPostCallee(MLIRContext *context, Type valueType);
StringRef buildVldasCallee(MLIRContext *context);
FailureOr<StringRef> buildVldusCallee(MLIRContext *context, Type resultType);
FailureOr<StringRef> buildVldusPostCallee(MLIRContext *context, Type resultType);
FailureOr<StringRef> buildVcmpCallee(MLIRContext *context, Type inputType, StringRef cmpMode, bool isScalarCompare);
FailureOr<StringRef> buildCopyGmToUbCallee(MLIRContext *context, Type sourceType);
StringRef buildCopyUbToGmCallee(MLIRContext *context);
StringRef buildCopyUbToUbCallee(MLIRContext *context);
StringRef buildCopyCbufToUbCallee(MLIRContext *context);
StringRef buildCopyUbToCbufCallee(MLIRContext *context);
FailureOr<StringRef> buildOrdinaryMadCallee(MLIRContext *context, pto::MadRawOpInterface op);
FailureOr<StringRef> buildMxMadCallee(MLIRContext *context, pto::MadRawOpInterface op);
FailureOr<StringRef> buildCopyGmToCbufCallee(MLIRContext *context, Type sourceType);
FailureOr<StringRef> buildCopyGmToCbufMultiNd2NzCallee(MLIRContext *context, Type sourceType);
std::string getDn2NzCopyElementFragment(Type type);
FailureOr<StringRef> buildCopyGmToCbufMultiDn2NzCallee(MLIRContext *context, Type sourceType);
FailureOr<StringRef> buildLoadCbufToCaCallee(MLIRContext *context, Type sourceType);
FailureOr<StringRef> buildLoadCbufToCbCallee(MLIRContext *context, Type sourceType);
FailureOr<StringRef> buildLoadCbufToCaS4Callee(MLIRContext *context, Type sourceType);
FailureOr<StringRef> buildLoadCbufToCbS4Callee(MLIRContext *context, Type sourceType);
StringRef buildLoadCbufToCaMxCallee(MLIRContext *context);
StringRef buildLoadCbufToCbMxCallee(MLIRContext *context);
StringRef buildCopyMatrixCcToGmCallee(MLIRContext *context);
StringRef buildCopyMatrixCcToCbufCallee(MLIRContext *context);
FailureOr<StringRef> buildCopyMatrixCcToUbCallee(MLIRContext *context, Type destinationType);
FailureOr<StringRef> buildCopyCbufToBtCallee(pto::CopyCbufToBtOp op);
StringRef buildCopyCbufToFbufCallee(MLIRContext *context);
StringRef buildPstiCallee(MLIRContext *context, bool post);
StringRef buildPstsCallee(MLIRContext *context, bool post);
StringRef buildPldiCallee(MLIRContext *context, bool post);
StringRef buildPldsCallee(MLIRContext *context, bool post);
StringRef buildPnotCallee(MLIRContext *context);
StringRef buildPselCallee(MLIRContext *context);
StringRef buildPandCallee(MLIRContext *context);
StringRef buildPorCallee(MLIRContext *context);
StringRef buildPxorCallee(MLIRContext *context);
StringRef buildPpackCallee(MLIRContext *context);
StringRef buildPunpackCallee(MLIRContext *context);
FailureOr<StringRef> buildInterleaveCallee(MLIRContext *context, Type resultType, StringRef stem);
FailureOr<StringRef> buildUnpackCallee(MLIRContext *context, Type inputType, Type resultType, StringRef stem);
FailureOr<StringRef> buildVpackCallee(MLIRContext *context, Type inputType, Type resultType);
FailureOr<StringRef> buildVsqzCallee(MLIRContext *context, Type resultType);
FailureOr<StringRef> buildVusqzCallee(MLIRContext *context, Type resultType);
FailureOr<StringRef> buildVmulaCallee(MLIRContext *context, Type resultType);
FailureOr<StringRef> buildVmullCallee(MLIRContext *context, Type resultType);
FailureOr<StringRef> buildVldsCallee(MLIRContext *context, Type resultType);
FailureOr<StringRef> buildVldsx2Callee(MLIRContext *context, Type resultType, bool post);
FailureOr<StringRef> buildBlockStridedMemoryCallee(MLIRContext *context, Type vectorType, StringRef stem, bool post);
FailureOr<StringRef> buildVsldbCallee(MLIRContext *context, Type resultType, bool post);
FailureOr<StringRef> buildVstsCallee(MLIRContext *context, Type valueType);
FailureOr<StringRef> buildVstsx2Callee(MLIRContext *context, Type valueType);
FailureOr<StringRef> buildVsstbCallee(MLIRContext *context, Type valueType, bool post);
Type getVgather2SourceElementType(Type sourceType);
FailureOr<StringRef> buildVgather2Callee(MLIRContext *context, Type sourceType, Type resultType);
std::optional<uint64_t> getFixedVectorBitWidth(Type type);
FailureOr<Type> getVgather2OffsetsCarrierType(PatternRewriter &rewriter, Type sourceType, Type resultType,
                                              Type offsetsType);
FailureOr<StringRef> buildVgather2BcCallee(MLIRContext *context, Type resultType);
FailureOr<StringRef> buildVgatherbCallee(MLIRContext *context, Type resultType);
FailureOr<StringRef> buildVscatterCallee(MLIRContext *context, Type valueType);
FailureOr<Type> getVscatterOffsetsCarrierType(Type offsetsType);
FailureOr<StringRef> buildVaxpyCallee(MLIRContext *context, Type resultType);
FailureOr<StringRef> buildVmulscvtCallee(MLIRContext *context, Type inputType, Type resultType);
FailureOr<StringRef> buildVciCallee(MLIRContext *context, Type resultType);
FailureOr<StringRef> buildVtrcCallee(MLIRContext *context, Type resultType);
FailureOr<StringRef> buildVexpdifCallee(MLIRContext *context, Type inputType, Type resultType);
FailureOr<StringRef> buildVbitsortCallee(MLIRContext *context, pto::VbitsortOp op);
FailureOr<StringRef> buildVmrgsort4Callee(MLIRContext *context, pto::Vmrgsort4Op op);
FailureOr<Value> packVmrgsort4SourceAddr(Operation *anchor, Value source0, Value source1, Value source2, Value source3,
                                         Type elemType);
FailureOr<VcvtContract> buildVcvtContract(pto::VcvtOp op);
bool needsV300CtrlModeForVPTOFunc(func::FuncOp funcOp);
FailureOr<Value> encodeMovPadValue(Location loc, Value value, ConversionPatternRewriter &rewriter);
StringRef buildMemBarCallee(MemBarKind kind, MLIRContext *context);
uint64_t getDsbMemImmediate(DsbMem kind);
uint64_t getDcciCacheLineImmediate(DcciCacheLine kind);
uint64_t getDcciDstImmediate(DcciDst kind);
StringRef buildDcciCallee(unsigned addressSpace, bool hasDst, MLIRContext *context);
StringRef buildBufDynSyncCallee(MLIRContext *context, bool isGetBuf);
LogicalResult materializeDecls(ModuleOp module, ArrayRef<PlannedDecl> plannedDecls, llvm::raw_ostream &diagOS);

void populateVPTOArithmeticPatterns(VPTOTypeConverter &typeConverter, RewritePatternSet &patterns,
                                    LoweringState &state);
void populateVPTOMemoryPatterns(VPTOTypeConverter &typeConverter, RewritePatternSet &patterns, LoweringState &state);
void populateVPTOVectorMemoryPatterns(VPTOTypeConverter &typeConverter, RewritePatternSet &patterns,
                                      LoweringState &state);
void populateVPTOScalarPatterns(VPTOTypeConverter &typeConverter, RewritePatternSet &patterns, LoweringState &state);
void populateVPTOTypePatterns(VPTOTypeConverter &typeConverter, RewritePatternSet &patterns, ConversionTarget &target,
                              LoweringState &state);
void populateVPTOStructuralTypePatterns(VPTOTypeConverter &typeConverter, RewritePatternSet &patterns,
                                        ConversionTarget &target);
LogicalResult lowerCANN900Module(ModuleOp module, const VPTOEmissionOptions &options, EmittedLLVMModule &cubeModule,
                                 EmittedLLVMModule &vectorModule, llvm::raw_ostream &diagOS);

} // namespace detail
} // namespace mlir::pto
