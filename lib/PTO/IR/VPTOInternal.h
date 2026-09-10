// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Internal shared helpers for VPTO split files.
// This header is internal to lib/PTO/IR and not installed.

#ifndef PTO_IR_VPTO_INTERNAL_H
#define PTO_IR_VPTO_INTERNAL_H

#include <algorithm>
#include <optional>
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VPTOMemoryDist.h"
#include "PTO/Support/CodeConstants.h"


using namespace mlir;
using namespace mlir::pto;

extern llvm::cl::opt<bool> disableVPTOAlignChainVerification;

LogicalResult verifyAlignTypeLike(Operation *op, Type type,
                                  StringRef roleDescription);
LogicalResult verifyStoreAlignChain(Value align, Operation *user,
                                    StringRef roleDescription);
LogicalResult verifyLoadAlignChain(Value align, Operation *user,
                                   StringRef roleDescription);


enum class MemoryRole {
  Unknown,
  GM,
  UB,
  Other,
};

MemoryRole classifyMemoryRole(Type type);

[[maybe_unused]] static bool isBufferLike(Type type) {
  return isa<BaseMemRefType, pto::PtrType>(type);
}

bool isForbiddenSynchronizationInsideVecScope(Operation *op);

Operation *findForbiddenSyncInRegion(Region &body);



LogicalResult verifyMaskTypeLike(Operation *op, Type type, StringRef roleDescription);
LogicalResult verifyMaskTypeWithGranularityLike(Operation *op, Type type,
                                                StringRef roleDescription,
                                                StringRef granularity);
std::optional<StringRef> normalizeRoundModeToken(StringRef token);
std::optional<StringRef> normalizeSaturationToken(StringRef token);
ParseResult normalizeNamedStringAttr(
    OpAsmParser &parser, NamedAttrList &attrs, StringRef sourceName,
    StringRef canonicalName,
    std::optional<StringRef> (*normalizeFn)(StringRef));


std::optional<StringRef> normalizeEvenOddPartToken(StringRef token);

// Batch1: RawFill 对齐常量
constexpr uint64_t kRawFillByteOffsetAlignment = 32;
constexpr uint64_t kRawFillControlFieldMax = 32767;

// Batch1: 由 VPTO.cpp 上移的文件局部类型(StructuredAccStore/CubeBridge 域共用)

struct StructuredAccStoreAsmState {
  std::optional<AccStoreUnitFlagCtrl> unitFlag;
  std::optional<AccStoreQuantPreMode> preQuantMode;
  std::optional<ReluPreMode> preReluMode;
  std::optional<AccStoreMode> mode;
  std::optional<AccStoreAtomicType> atomicType;
  std::optional<AccStoreAtomicOp> atomicOp;
  std::optional<AccStoreSatMode> satMode;

  SmallVector<OpAsmParser::UnresolvedOperand, 1> preQuantOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> preReluOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> clipValueOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> splitOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> loop0SrcStrideOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> loop3CountOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> loop3SrcStrideOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> loop3DstStrideOperands;

  SmallVector<Type, 1> preQuantTypes;
  SmallVector<Type, 1> preReluTypes;
  SmallVector<Type, 1> clipValueTypes;
  SmallVector<Type, 1> splitTypes;
  SmallVector<Type, 1> loop0SrcStrideTypes;
  SmallVector<Type, 1> loop3CountTypes;
  SmallVector<Type, 1> loop3SrcStrideTypes;
  SmallVector<Type, 1> loop3DstStrideTypes;
};

struct CubeBridgeLoadAsmOperand {
  OpAsmParser::UnresolvedOperand operand;
  Type type;
  bool present = false;
};

// Batch1: 跨文件共享函数声明(定义分布在 VPTO/VPTOMte/VPTOMteAsm/VPTODma/VPTOCubeBridge/VPTOStructuredAcc/VPTOVecOp/VPTOMemOp)
LogicalResult verifyIntegerVRegTypeLike(Operation *op, Type type, StringRef roleDescription);
LogicalResult checkConstAlignment(Operation *op, Value value, StringRef name, uint64_t alignment);
LogicalResult checkConstMax(Operation *op, Value value, StringRef name, uint64_t max);
std::string formatVRegType(int64_t elementCount, Type elementType);
StringRef getAddressSpaceDiagnosticName(pto::AddressSpace space);
unsigned getIntOrFloatBitWidth(Type type);
std::optional<int64_t> getVRegStorageBitWidth(Type type);
bool isInsideSimtExecutionScope(Operation *op);
bool isIntegerOrFloatLike(Type type);
bool isMxElementType(Type type);
bool isSupportedMovPadScalarType(Type type);
bool isSupportedPostMode(StringRef mode);
bool isSupportedPredicatePattern(StringRef pattern);
bool isVector2Of(Type type, llvm::function_ref<bool(Type)> elementPred);
ParseResult parseCubeBridgeOptionalOperands( OpAsmParser &parser, ArrayRef<StringRef> shapeNames, ArrayRef<StringRef> fullNames, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &legacyOperands, SmallVectorImpl<CubeBridgeLoadAsmOperand> &namedOperands, SmallVectorImpl<unsigned> &namedOperandOrder, bool &usesNamedOperands);
ParseResult parseCubeBridgeOptionalTypes( OpAsmParser &parser, bool usesNamedOperands, SmallVectorImpl<unsigned> &namedOperandOrder, SmallVectorImpl<CubeBridgeLoadAsmOperand> &namedOperands, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &legacyOperands, SmallVectorImpl<Type> &legacyTypes);
FailureOr<CubeLoadFracMode> parseCubeLoadFracModeKeyword(StringRef keyword);
ParseResult parseCubeLoadFracSrcLayoutGroup( OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands);
ParseResult parseCubeLoadFracSrcLayoutTypes(OpAsmParser &parser, SmallVectorImpl<Type> &types);
ParseResult parseDmaLoopAndPadTypeGroups( OpAsmParser &parser, SmallVectorImpl<Type> &loopCountTypes, SmallVectorImpl<Type> &loopSrcStrideTypes, SmallVectorImpl<Type> &loopDstStrideTypes, SmallVectorImpl<Type> &padTypes);
ParseResult parseDmaLoopOperandGroups( OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopCountOperands, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopSrcStrideOperands, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopDstStrideOperands);
ParseResult parseDmaLoopTypeGroups( OpAsmParser &parser, SmallVectorImpl<Type> &loopCountTypes, SmallVectorImpl<Type> &loopSrcStrideTypes, SmallVectorImpl<Type> &loopDstStrideTypes);
ParseResult parseDmaPadOperandGroup( OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &padOperands);
ParseResult parseDmaTripleGroup( OpAsmParser &parser, StringRef keyword, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands);
ParseResult parseDmaTripleTypes(OpAsmParser &parser, SmallVectorImpl<Type> &types);
ParseResult parseFixedKeywordOperandGroup( OpAsmParser &parser, StringRef keyword, int operandCount, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands);
ParseResult parseFixedKeywordTypes(OpAsmParser &parser, StringRef keyword, int typeCount, SmallVectorImpl<Type> &types);
ParseResult parseMteGmL1FracBasicOperands( OpAsmParser &parser, OpAsmParser::UnresolvedOperand &source, OpAsmParser::UnresolvedOperand &destination, StringRef &modeKeyword, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &shapeOperands, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &srcLayoutOperands, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dstGroupOperands, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &ctrlOperands);
ParseResult parseMteGmL1FracBasicTypes( OpAsmParser &parser, Type &sourceType, Type &destinationType, StringRef modeKeyword, SmallVectorImpl<Type> &shapeTypes, SmallVectorImpl<Type> &srcLayoutTypes, SmallVectorImpl<Type> &dstGroupTypes, SmallVectorImpl<Type> &ctrlTypes);
ParseResult parseMteGmUbBasicOperands( OpAsmParser &parser, OpAsmParser::UnresolvedOperand &source, OpAsmParser::UnresolvedOperand &destination, OpAsmParser::UnresolvedOperand &l2CacheCtl, OpAsmParser::UnresolvedOperand &lenBurst, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &nburstOperands);
ParseResult parseMteGmUbBasicTypes( OpAsmParser &parser, Type &sourceType, Type &destinationType, Type &l2CacheCtlType, Type &lenBurstType, SmallVectorImpl<Type> &nburstTypes);
ParseResult parseMteL0cGmTypes( OpAsmParser &parser, Type &sourceType, Type &destinationType, Type &mType, Type &nType, Type &srcStrideType, Type &dstStrideType, Type &sidType, Type &l2CacheCtrlType, StructuredAccStoreAsmState &state);
ParseResult parseMteL0cL1Types( OpAsmParser &parser, Type &sourceType, Type &destinationType, Type &mType, Type &nType, Type &srcStrideType, Type &dstStrideType, StructuredAccStoreAsmState &state);
ParseResult parseMteL0cUbBasicOperands( OpAsmParser &parser, OpAsmParser::UnresolvedOperand &source, OpAsmParser::UnresolvedOperand &destination, OpAsmParser::UnresolvedOperand &m, OpAsmParser::UnresolvedOperand &n, OpAsmParser::UnresolvedOperand &srcStride, OpAsmParser::UnresolvedOperand &dstStride);
ParseResult parseMteL0cUbDstMode(OpAsmParser &parser, AccStoreUbDstMode &dstMode, OpAsmParser::UnresolvedOperand &subBlockId, bool &hasSubBlockId);
ParseResult parseMteL0cUbTypes( OpAsmParser &parser, Type &sourceType, Type &destinationType, Type &mType, Type &nType, Type &srcStrideType, Type &dstStrideType, bool hasSubBlockId, Type &subBlockIdType, StructuredAccStoreAsmState &state);
ParseResult parseMteUbGmBasicOperands( OpAsmParser &parser, OpAsmParser::UnresolvedOperand &source, OpAsmParser::UnresolvedOperand &destination, OpAsmParser::UnresolvedOperand &lenBurst, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &nburstOperands);
ParseResult parseMteUbGmBasicTypes( OpAsmParser &parser, Type &sourceType, Type &destinationType, Type &lenBurstType, SmallVectorImpl<Type> &nburstTypes);
ParseResult parseMteUbGmL2CacheCtlOperand( OpAsmParser &parser, OpAsmParser::UnresolvedOperand &l2CacheCtl, bool &hasL2CacheCtl);
ParseResult parseOptionalDmaTripleGroupAlias( OpAsmParser &parser, ArrayRef<StringRef> keywords, StringRef &parsedKeyword, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands);
ParseResult parseStructuredAccStoreClauses( OpAsmParser &parser, StructuredAccStoreAsmState &state);
ParseResult parseStructuredAccStoreTailTypes( OpAsmParser &parser, StructuredAccStoreAsmState &state);
void printCubeLoadFracSrcLayoutGroup(OpAsmPrinter &printer, Value srcInnerStride, Value srcOuterStride);
void printCubeLoadFracSrcLayoutTypes(OpAsmPrinter &printer, Type srcInnerStrideType, Type srcOuterStrideType);
void printDmaPadGroup(OpAsmPrinter &printer, Value value, Value left, Value right);
void printDmaPadTypes(OpAsmPrinter &printer, Type valueType, Type leftType, Type rightType);
void printDmaTripleGroup(OpAsmPrinter &printer, StringRef keyword, Value first, Value second, Value third);
void printDmaTripleTypes(OpAsmPrinter &printer, StringRef keyword, Type first, Type second, Type third);
void printMteL1L0OptionalOperandsOp( OpAsmPrinter &printer, Operation *operation, Value source, Value destination, ArrayRef<Value> shapeOperands, ArrayRef<StringRef> shapeNames, ArrayRef<Value> fullOperands, ArrayRef<StringRef> fullNames);
void printStructuredAccStoreClauses( OpAsmPrinter &printer, std::optional<AccStoreUnitFlagCtrl> unitFlag, Value preQuant, std::optional<AccStoreQuantPreMode> preQuantMode, Value preRelu, std::optional<ReluPreMode> preReluMode, Value clipValue, std::optional<AccStoreMode> mode, Value split, Value loop0SrcStride, Value loop3Count, Value loop3SrcStride, Value loop3DstStride, std::optional<AccStoreSatMode> satMode, std::optional<AccStoreAtomicType> atomicType, std::optional<AccStoreAtomicOp> atomicOp);
void printStructuredAccStoreOptionalTypes( OpAsmPrinter &printer, Value preQuant, Value preRelu, Value clipValue, Value split, Value loop0SrcStride, Value loop3Count, Value loop3SrcStride, Value loop3DstStride);
ParseResult resolveCubeBridgeOperands( OpAsmParser &parser, OperationState &result, bool usesNamedOperands, OpAsmParser::UnresolvedOperand source, Type sourceType, OpAsmParser::UnresolvedOperand destination, Type destinationType, SmallVectorImpl<CubeBridgeLoadAsmOperand> &namedOperands, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &legacyOperands, SmallVectorImpl<Type> &legacyTypes, SmallVectorImpl<int32_t> &segmentSizes);
ParseResult resolveDmaBasicOperands( OpAsmParser &parser, OperationState &result, OpAsmParser::UnresolvedOperand source, Type sourceType, OpAsmParser::UnresolvedOperand destination, Type destinationType, OpAsmParser::UnresolvedOperand lenBurst, Type lenBurstType, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &nburstOperands, SmallVectorImpl<Type> &nburstTypes);
ParseResult resolveDmaLoopOperands( OpAsmParser &parser, OperationState &result, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopCountOperands, SmallVectorImpl<Type> &loopCountTypes, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopSrcStrideOperands, SmallVectorImpl<Type> &loopSrcStrideTypes, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopDstStrideOperands, SmallVectorImpl<Type> &loopDstStrideTypes);
ParseResult resolveDmaTripleOperands( OpAsmParser &parser, OperationState &result, bool hasL2CacheCtl, OpAsmParser::UnresolvedOperand l2CacheCtl, Type l2CacheCtlType, OpAsmParser::UnresolvedOperand source, Type sourceType, OpAsmParser::UnresolvedOperand destination, Type destinationType, OpAsmParser::UnresolvedOperand lenBurst, Type lenBurstType, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &nburstOperands, SmallVectorImpl<Type> &nburstTypes, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopCountOperands, SmallVectorImpl<Type> &loopCountTypes, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopSrcStrideOperands, SmallVectorImpl<Type> &loopSrcStrideTypes, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopDstStrideOperands, SmallVectorImpl<Type> &loopDstStrideTypes);
ParseResult resolveMteGmL1FracOperands( OpAsmParser &parser, OperationState &result, OpAsmParser::UnresolvedOperand source, Type sourceType, OpAsmParser::UnresolvedOperand destination, Type destinationType, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &shapeOperands, SmallVectorImpl<Type> &shapeTypes, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &srcLayoutOperands, SmallVectorImpl<Type> &srcLayoutTypes, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dstGroupOperands, SmallVectorImpl<Type> &dstGroupTypes, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &ctrlOperands, SmallVectorImpl<Type> &ctrlTypes);
ParseResult resolveMteGmUbOperands( OpAsmParser &parser, OperationState &result, OpAsmParser::UnresolvedOperand source, Type sourceType, OpAsmParser::UnresolvedOperand destination, Type destinationType, OpAsmParser::UnresolvedOperand l2CacheCtl, Type l2CacheCtlType, OpAsmParser::UnresolvedOperand lenBurst, Type lenBurstType, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &nburstOperands, SmallVectorImpl<Type> &nburstTypes, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopCountOperands, SmallVectorImpl<Type> &loopCountTypes, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopSrcStrideOperands, SmallVectorImpl<Type> &loopSrcStrideTypes, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &loopDstStrideOperands, SmallVectorImpl<Type> &loopDstStrideTypes, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &padOperands, SmallVectorImpl<Type> &padTypes);
ParseResult resolveMteL0cGmOperands( OpAsmParser &parser, OperationState &result, OpAsmParser::UnresolvedOperand source, Type sourceType, OpAsmParser::UnresolvedOperand destination, Type destinationType, OpAsmParser::UnresolvedOperand m, Type mType, OpAsmParser::UnresolvedOperand n, Type nType, OpAsmParser::UnresolvedOperand srcStride, Type srcStrideType, OpAsmParser::UnresolvedOperand dstStride, Type dstStrideType, OpAsmParser::UnresolvedOperand sid, Type sidType, OpAsmParser::UnresolvedOperand l2CacheCtrl, Type l2CacheCtrlType, StructuredAccStoreAsmState &state);
ParseResult resolveMteL0cL1Operands( OpAsmParser &parser, OperationState &result, OpAsmParser::UnresolvedOperand source, Type sourceType, OpAsmParser::UnresolvedOperand destination, Type destinationType, OpAsmParser::UnresolvedOperand m, Type mType, OpAsmParser::UnresolvedOperand n, Type nType, OpAsmParser::UnresolvedOperand srcStride, Type srcStrideType, OpAsmParser::UnresolvedOperand dstStride, Type dstStrideType, StructuredAccStoreAsmState &state);
ParseResult resolveMteL0cUbOperands( OpAsmParser &parser, OperationState &result, OpAsmParser::UnresolvedOperand source, Type sourceType, OpAsmParser::UnresolvedOperand destination, Type destinationType, OpAsmParser::UnresolvedOperand m, Type mType, OpAsmParser::UnresolvedOperand n, Type nType, OpAsmParser::UnresolvedOperand srcStride, Type srcStrideType, OpAsmParser::UnresolvedOperand dstStride, Type dstStrideType, bool hasSubBlockId, OpAsmParser::UnresolvedOperand subBlockId, Type subBlockIdType, const StructuredAccStoreAsmState &state);
void setMteGmUbSegmentSizes(OperationState &result, int32_t loopGroupCount, size_t padOperandCount);
void setMteL0cGmSegmentSizes(OperationState &result, const StructuredAccStoreAsmState &st);
void setMteL0cL1SegmentSizes(OperationState &result, const StructuredAccStoreAsmState &st);
void setMteUbGmSegmentSizes(OperationState &result, bool hasL2CacheCtl, size_t loopGroupCount);
ParseResult validateMteGmL1FracOperands( OpAsmParser &parser, size_t shapeOps, size_t shapeTypes, size_t srcLayoutOps, size_t srcLayoutTypes, size_t dstGroupOps, size_t dstGroupTypes, size_t ctrlOps, size_t ctrlTypes);
LogicalResult verifyDmaLoadStoreLoopGroups(Operation *op, ValueRange loopCounts, ValueRange loopSrcStrides, ValueRange loopDstStrides);
ParseResult verifyDmaLoopGroupConsistency( OpAsmParser &parser, size_t countOperands, size_t srcStrideOperands, size_t dstStrideOperands, size_t countTypes, size_t srcStrideTypes, size_t dstStrideTypes);
LogicalResult verifyMxLoadAlignment(Operation *op, Value source, Value destination);
LogicalResult verifyMxLoadOperands(Operation *op, ArrayRef<Value> shapeOperands, ArrayRef<StringRef> shapeNames, ArrayRef<Value> fullOperands);
LogicalResult verifyNestedInVecScope(Operation *op, StringRef opNameForDiag);
LogicalResult verifyNonLowPrecisionVRegElementTypeLike( Operation *op, Type type, StringRef roleDescription);
LogicalResult verifyNotNestedInVecScope(Operation *op, StringRef opNameForDiag);
LogicalResult verifyStructuredAccStoreLike( Operation *op, Type srcType, Type dstType, Value preQuant, Value preRelu, Value clipValue, Value split, Value loop0SrcStride, Value loop3Count, Value loop3SrcStride, Value loop3DstStride, std::optional<AccStoreUnitFlagCtrl> unitFlag, std::optional<AccStoreQuantPreMode> preQuantMode, std::optional<ReluPreMode> preReluMode, std::optional<AccStoreMode> mode, std::optional<AccStoreAtomicType> atomicType, std::optional<AccStoreAtomicOp> atomicOp, bool allowAtomic);

// Batch1: VPTO.cpp/MTE/DMA 共用小函数
[[maybe_unused]] 
static LogicalResult verifyVRegTypeLike(Operation *op, Type type,
                                       StringRef roleDescription) {
  auto vecType = dyn_cast<VRegType>(type);
  if (!vecType) {
    return op->emitOpError() << roleDescription << " must be !pto.vreg<...>";
  }

  return VRegType::verify(
      [&]() { return op->emitOpError() << roleDescription << " "; },
      vecType.getElementCount(), vecType.getElementType());
}

[[maybe_unused]] 

static int64_t getBufferElementByteSize(Type type) {
  Type elementType;
  if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
    elementType = ptrType.getElementType();
  } else if (auto memrefType = dyn_cast<BaseMemRefType>(type)) {
    elementType = memrefType.getElementType();
  } else {
    return 0;
  }

  return getPTOStorageElemByteSize(elementType);
}

[[maybe_unused]] 
static Type getBufferElementType(Type type) {
  if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
    return ptrType.getElementType();
  }
  if (auto memrefType = dyn_cast<BaseMemRefType>(type)) {
    return memrefType.getElementType();
  }
  return {};
}

[[maybe_unused]] 
static std::optional<AddressSpace> getBufferAddressSpace(Type type) {
  if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
    return ptrType.getMemorySpace().getAddressSpace();
  }
  if (auto memrefType = dyn_cast<BaseMemRefType>(type)) {
    if (auto space =
            dyn_cast_or_null<pto::AddressSpaceAttr>(memrefType.getMemorySpace())) {
      return space.getAddressSpace();
    }
    if (auto intSpace = dyn_cast_or_null<IntegerAttr>(memrefType.getMemorySpace())) {
      return static_cast<AddressSpace>(intSpace.getInt());
    }
  }
  return std::nullopt;
}

template <typename BridgeLoadOp>
static LogicalResult verifyCubeBridgeLoadLikeOp(BridgeLoadOp op,
                                                AddressSpace expectedDstSpace,
                                                StringRef dstName) {
  if (!isBufferLike(op.getSource().getType()) ||
      !isBufferLike(op.getDestination().getType())) {
    return op.emitOpError("requires buffer-like source and destination");
  }

  if (getBufferAddressSpace(op.getSource().getType()) != AddressSpace::MAT) {
    return op.emitOpError("requires MAT source");
  }
  if (getBufferAddressSpace(op.getDestination().getType()) != expectedDstSpace) {
    return op.emitOpError()
           << "requires " << dstName << " destination";
  }

  int64_t sourceElemBytes = getBufferElementByteSize(op.getSource().getType());
  int64_t destinationElemBytes =
      getBufferElementByteSize(op.getDestination().getType());
  if (sourceElemBytes <= 0 || destinationElemBytes <= 0) {
    return op.emitOpError(
        "requires source and destination element types with known byte width");
  }
  if (sourceElemBytes != destinationElemBytes) {
    return op.emitOpError(
        "requires source and destination element byte widths to match");
  }

  return success();
}

[[maybe_unused]] 
static ParseResult parseRequiredOperandWithComma(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &operand) {
  if (parser.parseOperand(operand)) {
    return failure();
  }
  (void)parser.parseOptionalComma();
  return success();
}

[[maybe_unused]] 

static LogicalResult checkNonNegativeConst(Operation *op, Value value,
                                           StringRef name) {
  if (!value) {
    return success();
  }
  APInt intValue;
  if (matchPattern(value, m_ConstantInt(&intValue)) && intValue.isNegative()) {
    return op->emitOpError() << name << " must be non-negative";
  }
  return success();
}

[[maybe_unused]] 
static LogicalResult verifyCubeBridgeLoadStart(Operation *op, Value firstStart,
                                               StringRef firstName,
                                               Value secondStart,
                                               StringRef secondName) {
  auto checkNonNegativeConst = [&](Value value, StringRef name) -> LogicalResult {
    APInt intValue;
    if (matchPattern(value, m_ConstantInt(&intValue)) && intValue.isNegative()) {
      return op->emitOpError() << name << " must be non-negative";
    }
    return success();
  };
  if (failed(checkNonNegativeConst(firstStart, firstName)) ||
      failed(checkNonNegativeConst(secondStart, secondName))) {
    return failure();
  }
  return success();
}

template <typename OpTy>
static LogicalResult verifyCubeBridgeLoadStart(OpTy op) {
  return verifyCubeBridgeLoadStart(op.getOperation(), op.getStartRow(),
                                   "start_row", op.getStartCol(), "start_col");
}

[[maybe_unused]] 
static LogicalResult verifyStaticControlRange(Operation *op, Value value,
                                              StringRef name, int64_t min,
                                              int64_t max) {
  APInt intValue;
  if (!matchPattern(value, m_ConstantInt(&intValue))) {
    return success();
}
  int64_t signedValue = intValue.getSExtValue();
  if (signedValue < min) {
    return op->emitOpError() << name
                             << (min == 0 ? " must be non-negative"
                                          : " must be greater than zero");
}
  if (signedValue > max) {
    return op->emitOpError() << name << " must be <= " << max
                             << " to fit the hardware control field";
}
  return success();
}


// Batch1: 跨文件实例化的模板定义(本体必须在头文件)
template <typename OpTy>
[[maybe_unused]] static void addStructuredAccStoreAttrs(OperationState &result,
                                       Builder &builder,
                                       const StructuredAccStoreAsmState &state) {
  if (state.mode) {
    result.addAttribute("mode", AccStoreModeAttr::get(builder.getContext(),
                                                      *state.mode));
  }
  if (state.unitFlag) {
    result.addAttribute("unit_flag",
                        AccStoreUnitFlagCtrlAttr::get(builder.getContext(),
                                                      *state.unitFlag));
  }
  if (state.preQuantMode) {
    result.addAttribute("pre_quant_mode",
                        AccStoreQuantPreModeAttr::get(builder.getContext(),
                                                      *state.preQuantMode));
  }
  if (state.preReluMode) {
    result.addAttribute("pre_relu_mode",
                        ReluPreModeAttr::get(builder.getContext(),
                                             *state.preReluMode));
  }
  if (state.atomicType) {
    result.addAttribute("atomic_type",
                        AccStoreAtomicTypeAttr::get(builder.getContext(),
                                                    *state.atomicType));
  }
  if (state.atomicOp) {
    result.addAttribute("atomic_op",
                        AccStoreAtomicOpAttr::get(builder.getContext(),
                                                  *state.atomicOp));
  }
  if (state.satMode) {
    result.addAttribute("sat_mode",
                        AccStoreSatModeAttr::get(builder.getContext(),
                                                 *state.satMode));
  }
}

template <typename OpTy>
[[maybe_unused]] static void setStructuredAccStoreSegmentSizes(OperationState &result,
                                              ArrayRef<int32_t> segmentSizes) {
  auto &segments = result.getOrAddProperties<typename OpTy::Properties>()
                       .operandSegmentSizes;
  llvm::copy(segmentSizes, segments.begin());
}

template <typename OpTy>
[[maybe_unused]] static void setCubeBridgeLoadOperandSegmentSizes(
    OperationState &result, ArrayRef<int32_t> segmentSizes) {
  auto &segments = result.getOrAddProperties<typename OpTy::Properties>()
                       .operandSegmentSizes;
  llvm::copy(segmentSizes, segments.begin());
}

template <typename OpTy>
[[maybe_unused]] static ParseResult parseMteL1L0OptionalOperandsOp(
    OpAsmParser &parser, OperationState &result, ArrayRef<StringRef> shapeNames,
    ArrayRef<StringRef> fullNames, StringRef operandDescription = "operands") {
  OpAsmParser::UnresolvedOperand source;
  OpAsmParser::UnresolvedOperand destination;
  if (parser.parseOperand(source) || parser.parseComma() ||
      parser.parseOperand(destination)) {
    return failure();
  }
  SmallVector<OpAsmParser::UnresolvedOperand, 6> legacyOperands;
  SmallVector<CubeBridgeLoadAsmOperand, 10> namedOperands(10);
  SmallVector<unsigned, 10> namedOperandOrder;
  bool usesNamedOperands = false;
  if (failed(parseCubeBridgeOptionalOperands(
          parser, shapeNames, fullNames, legacyOperands, namedOperands,
          namedOperandOrder, usesNamedOperands))) {
    return failure();
  }
  if (!usesNamedOperands && legacyOperands.size() != 4 &&
      legacyOperands.size() != 6) {
    return parser.emitError(
               parser.getCurrentLocation(),
               "expects either four shape-derived or six full positional ")
           << operandDescription;
  }
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }
  Type sourceType;
  Type destinationType;
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType)) {
    return failure();
  }
  SmallVector<Type, mlir::pto::kValue6> legacyTypes;
  if (failed(parseCubeBridgeOptionalTypes(parser, usesNamedOperands,
                                          namedOperandOrder, namedOperands,
                                          legacyOperands, legacyTypes))) {
    return failure();
  }
  SmallVector<int32_t, 12> segmentSizes(12, 0);
  segmentSizes[0] = 1;
  segmentSizes[1] = 1;
  if (failed(resolveCubeBridgeOperands(
          parser, result, usesNamedOperands, source, sourceType,
          destination, destinationType, namedOperands, legacyOperands,
          legacyTypes, segmentSizes))) {
    return failure();
  }
  setCubeBridgeLoadOperandSegmentSizes<OpTy>(result, segmentSizes);
  return success();
}

// Batch1: 跨文件实例化的模板定义 II
template <typename CopyOp>
LogicalResult verifyCopyElemByteWidths(CopyOp op) {
  int64_t sourceElemBytes = getBufferElementByteSize(op.getSource().getType());
  int64_t destinationElemBytes =
      getBufferElementByteSize(op.getDestination().getType());
  if (sourceElemBytes <= 0 || destinationElemBytes <= 0) {
    return op.emitOpError("requires copy source and destination element types with known byte width");
  }
  if (sourceElemBytes != destinationElemBytes) {
    return op.emitOpError("requires source and destination element byte widths to match");
  }

  return success();
}


template <typename CopyOp>
LogicalResult verifyCopyGmToUbufOp(CopyOp op, bool expectSourceGM) {
  if (!isBufferLike(op.getSource().getType()) ||
      !isBufferLike(op.getDestination().getType())) {
    return op.emitOpError(
        "requires typed !pto.ptr or memref source and destination");
  }

  MemoryRole sourceRole = classifyMemoryRole(op.getSource().getType());
  MemoryRole destinationRole = classifyMemoryRole(op.getDestination().getType());
  bool directionMatches = true;
  if (expectSourceGM) {
    directionMatches &= sourceRole != MemoryRole::UB;
    directionMatches &= destinationRole != MemoryRole::GM;
  } else {
    directionMatches &= sourceRole != MemoryRole::GM;
    directionMatches &= destinationRole != MemoryRole::UB;
  }

  if (!directionMatches) {
    return op.emitOpError()
           << "requires "
           << (expectSourceGM ? "GM source and UB destination"
                              : "UB source and GM destination");
  }

  return verifyCopyElemByteWidths(op);
}

template <typename CopyOp>
LogicalResult verifyCopyCbufToUbufLikeOp(CopyOp op) {
  if (!isBufferLike(op.getSource().getType()) ||
      !isBufferLike(op.getDestination().getType())) {
    return op.emitOpError(
        "requires typed !pto.ptr or memref source and destination");
  }

  if (classifyMemoryRole(op.getSource().getType()) != MemoryRole::Other ||
      classifyMemoryRole(op.getDestination().getType()) != MemoryRole::UB) {
    return op.emitOpError("requires CBUF source and UB destination");
  }

  return verifyCopyElemByteWidths(op);
}

#endif // PTO_IR_VPTO_INTERNAL_H
