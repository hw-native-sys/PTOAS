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

struct PlannedDecl {
  std::string name;
  FunctionType type;
};

struct LoweringState {
  SmallVector<PlannedDecl> plannedDecls;
};

Value getI64Constant(OpBuilder &builder, Location loc, uint64_t value);
Value getI32Constant(OpBuilder &builder, Location loc, uint64_t value);
Value packShiftedI64Fields(OpBuilder &builder, Location loc, Value config,
                           ArrayRef<std::pair<Value, uint64_t>> fields);
Value packMaskedI64Fields(OpBuilder &builder, Location loc, Value config,
                          ArrayRef<std::pair<Value, uint64_t>> fields,
                          uint64_t mask);
Type convertVPTOType(Type type, Builder &builder);
Value materializeVPTOCast(OpBuilder &builder, Type resultType, ValueRange inputs, Location loc);
Type getLowPrecisionLLVMType(Type type, MLIRContext *context);
bool isLLVMExtensionVectorElementType(Type type);
Type getLLVMCompatibleVectorType(ArrayRef<int64_t> shape, Type elementType, ArrayRef<bool> scalableDims);
Type normalizePayloadTypeForLLVMLowering(Type type, Builder &builder);
Type normalizeGEPElementTypeForLLVMLowering(Type type, Builder &builder);
unsigned getNaturalByteAlignment(Type type);
bool hasVPTOConvertibleType(Type type);
bool hasVPTOConvertibleType(TypeRange types);
LLVM::LLVMStructType getVPTOStructStorageType(pto::StructType structType, Builder &builder);
FailureOr<Value> getVPTOStructFieldAddress(ConversionPatternRewriter &rewriter, Location loc, Value root,
                                           pto::StructType rootType, ArrayRef<int64_t> path);
std::string getElementTypeFragment(Type type);
std::string getLowPrecisionElementFragment(Type type);
std::string getMemoryElementTypeFragment(Type type);
std::string getCopyElementFragment(Type type);
std::string getDn2NzCopyElementFragment(Type type);
std::string getMadLhsFragment(Type type);
std::string getMadDstFragment(Type type);
Type getElementTypeFromVectorLike(Type type);
std::optional<int64_t> getElementCountFromVectorLike(Type type);
bool isOnePointStoreDist(StringRef dist);
std::optional<uint64_t> parseRoundModeImmediate(StringRef roundMode);
std::optional<uint64_t> parsePartImmediate(StringRef part);
std::optional<uint64_t> parseVcvtPartImmediate(StringRef part);
std::optional<uint64_t> parseSaturationImmediate(StringRef sat);
std::optional<uint64_t> parsePredicateStoreDistImmediate(StringRef dist);
std::optional<uint64_t> parsePredicateLoadDistImmediate(StringRef dist);
Value castIntegerLikeTo(Operation *anchor, Value value, Type targetType);
FailureOr<SmallVector<Value, 7>> castIntegerLikeOperands(Operation *anchor, ValueRange operands,
                                                         ArrayRef<unsigned> indices, Type targetType);
FailureOr<Value> reinterpretPointerToAddrSpace(Operation *anchor, Value value, unsigned targetAddressSpace);
FailureOr<SmallVector<Value, 2>> reinterpretPointerOperands(Operation *anchor, ArrayRef<Value> values,
                                                           ArrayRef<unsigned> addressSpaces);
FailureOr<Value> packLoopPair(Operation *anchor, Value low, Value high);
FailureOr<Value> packLoopSize(Operation *anchor, Value loop2, Value loop1);

class VPTOTypeConverter final : public TypeConverter {
public:
  explicit VPTOTypeConverter(MLIRContext *context);
};

namespace ubuf {
void populateVPTOUbufPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns, LoweringState &state,
                              const std::string &march);
}

namespace detail {

inline constexpr llvm::StringLiteral kVectorSuffix = "_mix_aiv";
inline constexpr llvm::StringLiteral kCubeSuffix = "_mix_aic";

// Canonical integer bit widths used across callee-name selection, ABI payload
// packing, and type classification in the CANN900 emitter.
inline constexpr unsigned kBits2 = 2;
inline constexpr unsigned kBits4 = 4;
inline constexpr unsigned kBits8 = 8;
inline constexpr unsigned kBits16 = 16;
inline constexpr unsigned kBits32 = 32;
inline constexpr unsigned kBits64 = 64;

// Byte-addressable element granularity: element bit widths must be a whole
// number of bytes for the byte-oriented load/store intrinsics.
inline constexpr unsigned kBitsPerByte = 8;

// SIMT keep/resume slot and TPER/TPERL register window constraints:
// 123 addressable slots, register file index cap of 126, and 2-register
// (64-bit) payloads that must start on an even slot.
inline constexpr int64_t kSimtKeepResumeSlotCount = 123;
inline constexpr int64_t kSimtKeepResumeLastBaseRegister = 126;
inline constexpr unsigned kSimtKeepResumePairRegisterCount = 2;

// get_vms4_sr packs four 16-bit counters into one i64 runtime query result.
inline constexpr unsigned kVms4SrCountFieldBits = 16;
inline constexpr unsigned kVms4SrCountFieldCount = 4;

// Hardware immediate encodings for the SPR store and CBUF matrix fill.
inline constexpr uint64_t kSprArImmediate = 74;
inline constexpr uint64_t kFillWordWidth16 = 16;
inline constexpr uint64_t kFillWordWidth32 = 32;

// Vector-pair ABI shapes shared by the widening/elementwise callee builders.
inline constexpr unsigned kVmullResultCount = 2;
inline constexpr unsigned kVectorPairLaneCount = 2;

// Named vector shapes of the specialized widening intrinsics (f16<->f32).
inline constexpr int64_t kVexpdifInterleaveLanes = 128;
inline constexpr int64_t kVexpdifLanes = 64;
inline constexpr int64_t kVmulscvtInputLanes = 64;
inline constexpr int64_t kVmulscvtResultLanes = 128;

// VGather2 packs two sub-byte source elements per result lane.
inline constexpr int64_t kVgather2LaneMultiplier = 2;
inline constexpr uint64_t kVgather2PackShift16 = 16;
inline constexpr uint64_t kVgather2PackShift32 = 32;
inline constexpr uint64_t kVgather2PackShift48 = 48;

// MOV.PAD payloads are limited to byte, halfword, and word widths.
inline constexpr unsigned kMovPadWidth8 = 8;
inline constexpr unsigned kMovPadWidth16 = 16;
inline constexpr unsigned kMovPadWidth32 = 32;

// Intrinsic call shapes for the memory patterns: compare calls take
// (lhs, rhs, mask); post-update loads/stores append the updated base pointer
// as the third intrinsic result; predicate-pair reorder and PSTU return the
// (value, mask) or (loaded, mask) pair.
inline constexpr unsigned kVcmpsCallArgCount = 3;
inline constexpr unsigned kUpdatedBaseResultIndex = 2;
inline constexpr unsigned kPredicatePairResultCount = 2;
inline constexpr unsigned kPostUpdateResultCount = 2;
inline constexpr unsigned kPstuResultCount = 2;

// Copy/load config word layouts. Shifts are the hardware field bit positions
// of each config operand, grouped per intrinsic layout.
inline constexpr unsigned kGmToUbConfig0OperandCount = 11;
inline constexpr uint64_t kGmToUbBurstNumShift = 4;
inline constexpr uint64_t kGmToUbBurstLenShift = 25;
inline constexpr uint64_t kGmToUbLeftPaddingShift = 46;
inline constexpr uint64_t kGmToUbRightPaddingShift = 52;
inline constexpr uint64_t kGmToUbDataSelectShift = 58;
inline constexpr uint64_t kGmToUbCacheCtlShift = 60;

inline constexpr unsigned kUbToGmConfig0OperandCount = 8;
inline constexpr uint64_t kUbToGmBurstNumShift = 4;
inline constexpr uint64_t kUbToGmBurstLenShift = 25;
inline constexpr uint64_t kUbToGmL2CacheCtrlShift = 60;

inline constexpr unsigned kUbToUbConfigOperandCount = 7;
inline constexpr uint64_t kUbToUbNBurstShift = 16;
inline constexpr uint64_t kUbToUbLenBurstShift = 32;
inline constexpr uint64_t kUbToUbDstGapShift = 48;

inline constexpr unsigned kCbufToUbConfigOperandCount = 7;
inline constexpr uint64_t kCbufToUbNBurstShift = 4;
inline constexpr uint64_t kCbufToUbLenBurstShift = 16;
inline constexpr uint64_t kCbufToUbSrcGapShift = 32;
inline constexpr uint64_t kCbufToUbDstGapShift = 48;

// copy_gm_to_cbuf config0 packs burst_num[24:4] and burst_len[45:25]; config1
// packs burst_src_stride[39:0] and burst_dst_stride[60:40].
inline constexpr uint64_t kGmToCbufBurstNumShift = 4;
inline constexpr uint64_t kGmToCbufBurstLenShift = 25;
inline constexpr uint64_t kGmToCbufSrcStrideShift = 40;
inline constexpr uint64_t kGmToCbufDstStrideShift = 40;

// copy_gm_to_cbuf_multi config0: sid, loop1SrcStride, l2CacheCtrl, nValue;
// config1: dValue, loop4SrcStride, smallC0En.
inline constexpr uint64_t kGmToCbufMultiLoop1SrcStrideShift = 4;
inline constexpr uint64_t kGmToCbufMultiL2CacheCtrlShift = 44;
inline constexpr uint64_t kGmToCbufMultiNValueShift = 48;
inline constexpr uint64_t kGmToCbufMultiLoop4SrcStrideShift = 21;
inline constexpr uint64_t kGmToCbufMultiSmallC0EnShift = 61;

// copy_cbuf_to_bt: convControl, nBurst, lenBurst, sourceGap, dstGap.
inline constexpr uint64_t kCbufToBtConvControlShift = 3;
inline constexpr uint64_t kCbufToBtNBurstShift = 4;
inline constexpr uint64_t kCbufToBtLenBurstShift = 16;
inline constexpr uint64_t kCbufToBtSourceGapShift = 32;
inline constexpr uint64_t kCbufToBtDstGapShift = 48;

// copy_cbuf_to_fbuf: nBurst, lenBurst, sourceGap, dstGap.
inline constexpr uint64_t kCbufToFbufNBurstShift = 4;
inline constexpr uint64_t kCbufToFbufLenBurstShift = 16;
inline constexpr uint64_t kCbufToFbufSourceGapShift = 32;
inline constexpr uint64_t kCbufToFbufDstGapShift = 48;

// load_cbuf_to_{s4,ca,cb} tile config0 packs the tile descriptor
// [mStart | kStart<<16 | mStep<<32 | kStep<<40]; config1 shares the same
// dst-stride field position as kStart.
inline constexpr uint64_t kLoadCbufKStartShift = 16;
inline constexpr uint64_t kLoadCbufMStepShift = 32;
inline constexpr uint64_t kLoadCbufKStepShift = 40;
inline constexpr uint64_t kLoadCbufDstStrideShift = kLoadCbufKStartShift;

// vbitsort config packs the repeat count into the top byte of the word.
inline constexpr uint64_t kBitsortRepeatShift = 56;

// TPER/TPERL packing: block stride occupies the high half of a 32-bit word.
inline constexpr uint64_t kBlockRepeatStrideBlockShift = 16;

// MAD bias packing: the low 32 bits of the destination/bias addresses share
// one 64-bit word.
inline constexpr uint64_t kMadBiasAddressMask = 0xffffffffULL;
inline constexpr uint64_t kMadBiasHighWordShift = 32;

// PLT mask vector length for the V300 dynamic predicate mask intrinsic.
inline constexpr int64_t kPltMaskVectorLength = 256;

// SIMT keep/resume groups handle at most four logical slots per inline-asm
// group.
inline constexpr unsigned kSimtKeepResumeGroupCapacity = 4;

// get_vms4_sr result selection: four 16-bit counters, each fetched by
// shifting the raw query result right by a multiple of 16 bits.
inline constexpr unsigned kVms4SrCounterCount = 4;

// Decimal radix for parsing numeric immediate suffixes from attribute strings.
inline constexpr unsigned kRadixDecimal = 10;

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

Value getI1Constant(OpBuilder &builder, Location loc, bool value);
bool isMxElementType(Type ty);
std::string getMadMxElementFragment(Type type);
FailureOr<StringRef> buildMadMxCalleeName(MLIRContext *context, Type lhsElem, Type rhsElem);
bool isSignedOrSignlessInteger(IntegerType intType, unsigned width);
std::string getMadRhsFragment(Type type);
bool isMadE4M3ElementType(Type type);
bool isMadE5M2ElementType(Type type);
ArrayRef<MadCalleeContract> getMadCalleeContracts();
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
std::string getCANN900MemoryElementTypeFragment(Type type);
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
FailureOr<Value> normalizeVdupScalarOperand(OpBuilder &builder, Location loc, Value input, Type resultType);
Value normalizeByteScalarOperandForCANN900VectorCall(OpBuilder &builder, Location loc, Value input,
                                                     Type semanticElementType);
bool isCompatibleScalarForSemanticType(Type semanticType, Type scalarType);
std::string getNd2NzCopyElementFragment(Type elementType);
std::optional<uint64_t> parsePredicatePatternImmediate(StringRef pattern);
std::optional<uint64_t> parseHiLoPartImmediate(StringRef part);
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
bool isMaskOnlyUsedByOnePointStores(Value mask);
std::optional<uint64_t> parseStoreX2DistImmediate(StringRef dist, Type);
Value packBlockRepeatStride(Operation *anchor, Value blockStride, Value repeatStride);
std::optional<uint64_t> parseOrderImmediate(StringRef order);
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
bool needsV300CtrlModeForCANN900Func(func::FuncOp funcOp);
FailureOr<Value> encodeMovPadValue(Location loc, Value value, ConversionPatternRewriter &rewriter);
StringRef buildMemBarCallee(MemBarKind kind, MLIRContext *context);
uint64_t getDsbMemImmediate(DsbMem kind);
uint64_t getDcciCacheLineImmediate(DcciCacheLine kind);
uint64_t getDcciDstImmediate(DcciDst kind);
StringRef buildDcciCallee(unsigned addressSpace, bool hasDst, MLIRContext *context);
StringRef buildBufDynSyncCallee(MLIRContext *context, bool isGetBuf);
LogicalResult materializeDecls(ModuleOp module, ArrayRef<PlannedDecl> plannedDecls, llvm::raw_ostream &diagOS);

void populateVPTOArithmeticPatterns(const VPTOTypeConverter &typeConverter, RewritePatternSet &patterns,
                                    LoweringState &state);
void populateVPTOMemoryPatterns(const VPTOTypeConverter &typeConverter, RewritePatternSet &patterns,
                                LoweringState &state);
void populateVPTOVectorMemoryPatterns(const VPTOTypeConverter &typeConverter, RewritePatternSet &patterns,
                                      LoweringState &state);
void populateVPTOScalarPatterns(const VPTOTypeConverter &typeConverter, RewritePatternSet &patterns,
                                LoweringState &state);
void populateVPTOTypePatterns(const VPTOTypeConverter &typeConverter, RewritePatternSet &patterns,
                              LoweringState &state);
void populateVPTOStructuralTypePatterns(VPTOTypeConverter &typeConverter, RewritePatternSet &patterns,
                                        ConversionTarget &target);
LogicalResult lowerCANN900Module(ModuleOp module, const VPTOEmissionOptions &options, EmittedLLVMModule &cubeModule,
                                 EmittedLLVMModule &vectorModule, llvm::raw_ostream &diagOS);

} // namespace detail
} // namespace mlir::pto
