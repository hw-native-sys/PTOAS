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

extern llvm::cl::opt<bool> disableVPTOAlignChainVerification;

mlir::LogicalResult verifyAlignTypeLike(mlir::Operation *op, mlir::Type type,
                                  llvm::StringRef roleDescription);
mlir::LogicalResult verifyStoreAlignChain(mlir::Value align, mlir::Operation *user,
                                    llvm::StringRef roleDescription);
mlir::LogicalResult verifyLoadAlignChain(mlir::Value align, mlir::Operation *user,
                                   llvm::StringRef roleDescription);

enum class MemoryRole {
  Unknown,
  GM,
  UB,
  Other,
};

MemoryRole classifyMemoryRole(mlir::Type type);

[[maybe_unused]] inline bool isBufferLike(mlir::Type type) {
  return mlir::isa<mlir::BaseMemRefType, mlir::pto::PtrType>(type);
}

bool isForbiddenSynchronizationInsideVecScope(mlir::Operation *op);

mlir::Operation *findForbiddenSyncInRegion(mlir::Region &body);

mlir::LogicalResult verifyMaskTypeLike(mlir::Operation *op, mlir::Type type, llvm::StringRef roleDescription);
mlir::LogicalResult verifyMaskTypeWithGranularityLike(mlir::Operation *op, mlir::Type type,
                                                llvm::StringRef roleDescription,
                                                llvm::StringRef granularity);
std::optional<llvm::StringRef> normalizeRoundModeToken(llvm::StringRef token);
std::optional<llvm::StringRef> normalizeSaturationToken(llvm::StringRef token);
mlir::ParseResult normalizeNamedStringAttr(
    mlir::OpAsmParser &parser, mlir::NamedAttrList &attrs, llvm::StringRef sourceName,
    llvm::StringRef canonicalName,
    std::optional<llvm::StringRef> (*normalizeFn)(llvm::StringRef));


std::optional<llvm::StringRef> normalizeEvenOddPartToken(llvm::StringRef token);

// Batch1: RawFill 对齐常量
constexpr uint64_t kRawFillByteOffsetAlignment = 32;
constexpr uint64_t kRawFillControlFieldMax = 32767;

// Batch1: 由 VPTO.cpp 上移的文件局部类型(StructuredAccStore/CubeBridge 域共用)

struct StructuredAccStoreAsmState {
  std::optional<mlir::pto::AccStoreUnitFlagCtrl> unitFlag;
  std::optional<mlir::pto::AccStoreQuantPreMode> preQuantMode;
  std::optional<mlir::pto::ReluPreMode> preReluMode;
  std::optional<mlir::pto::AccStoreMode> mode;
  std::optional<mlir::pto::AccStoreAtomicType> atomicType;
  std::optional<mlir::pto::AccStoreAtomicOp> atomicOp;
  std::optional<mlir::pto::AccStoreSatMode> satMode;

  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 1> preQuantOperands;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 1> preReluOperands;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 1> clipValueOperands;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 1> splitOperands;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 1> loop0SrcStrideOperands;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 1> loop3CountOperands;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 1> loop3SrcStrideOperands;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 1> loop3DstStrideOperands;

  llvm::SmallVector<mlir::Type, 1> preQuantTypes;
  llvm::SmallVector<mlir::Type, 1> preReluTypes;
  llvm::SmallVector<mlir::Type, 1> clipValueTypes;
  llvm::SmallVector<mlir::Type, 1> splitTypes;
  llvm::SmallVector<mlir::Type, 1> loop0SrcStrideTypes;
  llvm::SmallVector<mlir::Type, 1> loop3CountTypes;
  llvm::SmallVector<mlir::Type, 1> loop3SrcStrideTypes;
  llvm::SmallVector<mlir::Type, 1> loop3DstStrideTypes;
};

struct CubeBridgeLoadAsmOperand {
  mlir::OpAsmParser::UnresolvedOperand operand;
  mlir::Type type;
  bool present = false;
};

// Batch1: 跨文件共享函数声明(定义分布在 VPTO/VPTOMte/VPTOMteAsm/VPTODma/VPTOCubeBridge/VPTOStructuredAcc/VPTOVecOp/VPTOMemOp)
mlir::LogicalResult verifyIntegerVRegTypeLike(mlir::Operation *op, mlir::Type type, llvm::StringRef roleDescription);
mlir::LogicalResult checkConstAlignment(mlir::Operation *op, mlir::Value value, llvm::StringRef name, uint64_t alignment);
mlir::LogicalResult checkConstMax(mlir::Operation *op, mlir::Value value, llvm::StringRef name, uint64_t max);
std::string formatVRegType(int64_t elementCount, mlir::Type elementType);
llvm::StringRef getAddressSpaceDiagnosticName(mlir::pto::AddressSpace space);
unsigned getIntOrFloatBitWidth(mlir::Type type);
std::optional<int64_t> getVRegStorageBitWidth(mlir::Type type);
bool isInsideSimtExecutionScope(mlir::Operation *op);
bool isIntegerOrFloatLike(mlir::Type type);
bool isMxElementType(mlir::Type type);
bool isSupportedMovPadScalarType(mlir::Type type);
bool isSupportedPostMode(llvm::StringRef mode);
bool isCompatibleScalarForSemanticType(mlir::Type semanticType,
                                       mlir::Type scalarType);
bool isSupportedPredicatePattern(llvm::StringRef pattern);
bool isVector2Of(mlir::Type type, llvm::function_ref<bool(mlir::Type)> elementPred);
mlir::ParseResult parseCubeBridgeOptionalOperands( mlir::OpAsmParser &parser, llvm::ArrayRef<llvm::StringRef> shapeNames, llvm::ArrayRef<llvm::StringRef> fullNames, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &legacyOperands, llvm::SmallVectorImpl<CubeBridgeLoadAsmOperand> &namedOperands, llvm::SmallVectorImpl<unsigned> &namedOperandOrder, bool &usesNamedOperands);
mlir::ParseResult parseCubeBridgeOptionalTypes( mlir::OpAsmParser &parser, bool usesNamedOperands, llvm::SmallVectorImpl<unsigned> &namedOperandOrder, llvm::SmallVectorImpl<CubeBridgeLoadAsmOperand> &namedOperands, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &legacyOperands, llvm::SmallVectorImpl<mlir::Type> &legacyTypes);
llvm::FailureOr<mlir::pto::CubeLoadFracMode> parseCubeLoadFracModeKeyword(llvm::StringRef keyword);
mlir::ParseResult parseCubeLoadFracSrcLayoutGroup( mlir::OpAsmParser &parser, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &operands);
mlir::ParseResult parseCubeLoadFracSrcLayoutTypes(mlir::OpAsmParser &parser, llvm::SmallVectorImpl<mlir::Type> &types);
mlir::ParseResult parseDmaLoopAndPadTypeGroups( mlir::OpAsmParser &parser, llvm::SmallVectorImpl<mlir::Type> &loopCountTypes, llvm::SmallVectorImpl<mlir::Type> &loopSrcStrideTypes, llvm::SmallVectorImpl<mlir::Type> &loopDstStrideTypes, llvm::SmallVectorImpl<mlir::Type> &padTypes);
mlir::ParseResult parseDmaLoopOperandGroups( mlir::OpAsmParser &parser, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &loopCountOperands, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &loopSrcStrideOperands, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &loopDstStrideOperands);
mlir::ParseResult parseDmaLoopTypeGroups( mlir::OpAsmParser &parser, llvm::SmallVectorImpl<mlir::Type> &loopCountTypes, llvm::SmallVectorImpl<mlir::Type> &loopSrcStrideTypes, llvm::SmallVectorImpl<mlir::Type> &loopDstStrideTypes);
mlir::ParseResult parseDmaPadOperandGroup( mlir::OpAsmParser &parser, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &padOperands);
mlir::ParseResult parseDmaTripleGroup( mlir::OpAsmParser &parser, llvm::StringRef keyword, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &operands);
mlir::ParseResult parseDmaTripleTypes(mlir::OpAsmParser &parser, llvm::SmallVectorImpl<mlir::Type> &types);
mlir::ParseResult parseFixedKeywordOperandGroup( mlir::OpAsmParser &parser, llvm::StringRef keyword, int operandCount, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &operands);
mlir::ParseResult parseFixedKeywordTypes(mlir::OpAsmParser &parser, llvm::StringRef keyword, int typeCount, llvm::SmallVectorImpl<mlir::Type> &types);
mlir::ParseResult parseMteGmL1FracBasicOperands( mlir::OpAsmParser &parser, mlir::OpAsmParser::UnresolvedOperand &source, mlir::OpAsmParser::UnresolvedOperand &destination, llvm::StringRef &modeKeyword, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &shapeOperands, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &srcLayoutOperands, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &dstGroupOperands, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &ctrlOperands);
mlir::ParseResult parseMteGmL1FracBasicTypes( mlir::OpAsmParser &parser, mlir::Type &sourceType, mlir::Type &destinationType, llvm::StringRef modeKeyword, llvm::SmallVectorImpl<mlir::Type> &shapeTypes, llvm::SmallVectorImpl<mlir::Type> &srcLayoutTypes, llvm::SmallVectorImpl<mlir::Type> &dstGroupTypes, llvm::SmallVectorImpl<mlir::Type> &ctrlTypes);
mlir::ParseResult parseMteGmUbBasicOperands( mlir::OpAsmParser &parser, mlir::OpAsmParser::UnresolvedOperand &source, mlir::OpAsmParser::UnresolvedOperand &destination, mlir::OpAsmParser::UnresolvedOperand &l2CacheCtl, mlir::OpAsmParser::UnresolvedOperand &lenBurst, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &nburstOperands);
mlir::ParseResult parseMteGmUbBasicTypes( mlir::OpAsmParser &parser, mlir::Type &sourceType, mlir::Type &destinationType, mlir::Type &l2CacheCtlType, mlir::Type &lenBurstType, llvm::SmallVectorImpl<mlir::Type> &nburstTypes);
mlir::ParseResult parseMteL0cGmTypes( mlir::OpAsmParser &parser, mlir::Type &sourceType, mlir::Type &destinationType, mlir::Type &mType, mlir::Type &nType, mlir::Type &srcStrideType, mlir::Type &dstStrideType, mlir::Type &sidType, mlir::Type &l2CacheCtrlType, StructuredAccStoreAsmState &state);
mlir::ParseResult parseMteL0cL1Types( mlir::OpAsmParser &parser, mlir::Type &sourceType, mlir::Type &destinationType, mlir::Type &mType, mlir::Type &nType, mlir::Type &srcStrideType, mlir::Type &dstStrideType, StructuredAccStoreAsmState &state);
mlir::ParseResult parseMteL0cUbBasicOperands( mlir::OpAsmParser &parser, mlir::OpAsmParser::UnresolvedOperand &source, mlir::OpAsmParser::UnresolvedOperand &destination, mlir::OpAsmParser::UnresolvedOperand &m, mlir::OpAsmParser::UnresolvedOperand &n, mlir::OpAsmParser::UnresolvedOperand &srcStride, mlir::OpAsmParser::UnresolvedOperand &dstStride);
mlir::ParseResult parseMteL0cUbDstMode(mlir::OpAsmParser &parser, mlir::pto::AccStoreUbDstMode &dstMode, mlir::OpAsmParser::UnresolvedOperand &subBlockId, bool &hasSubBlockId);
mlir::ParseResult parseMteL0cUbTypes( mlir::OpAsmParser &parser, mlir::Type &sourceType, mlir::Type &destinationType, mlir::Type &mType, mlir::Type &nType, mlir::Type &srcStrideType, mlir::Type &dstStrideType, bool hasSubBlockId, mlir::Type &subBlockIdType, StructuredAccStoreAsmState &state);
mlir::ParseResult parseMteUbGmBasicOperands( mlir::OpAsmParser &parser, mlir::OpAsmParser::UnresolvedOperand &source, mlir::OpAsmParser::UnresolvedOperand &destination, mlir::OpAsmParser::UnresolvedOperand &lenBurst, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &nburstOperands);
mlir::ParseResult parseMteUbGmBasicTypes( mlir::OpAsmParser &parser, mlir::Type &sourceType, mlir::Type &destinationType, mlir::Type &lenBurstType, llvm::SmallVectorImpl<mlir::Type> &nburstTypes);
mlir::ParseResult parseMteUbGmL2CacheCtlOperand( mlir::OpAsmParser &parser, mlir::OpAsmParser::UnresolvedOperand &l2CacheCtl, bool &hasL2CacheCtl);
mlir::ParseResult parseOptionalDmaTripleGroupAlias( mlir::OpAsmParser &parser, llvm::ArrayRef<llvm::StringRef> keywords, llvm::StringRef &parsedKeyword, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &operands);
mlir::ParseResult parseStructuredAccStoreClauses( mlir::OpAsmParser &parser, StructuredAccStoreAsmState &state);
mlir::ParseResult parseStructuredAccStoreTailTypes( mlir::OpAsmParser &parser, StructuredAccStoreAsmState &state);
void printCubeLoadFracSrcLayoutGroup(mlir::OpAsmPrinter &printer, mlir::Value srcInnerStride, mlir::Value srcOuterStride);
void printCubeLoadFracSrcLayoutTypes(mlir::OpAsmPrinter &printer, mlir::Type srcInnerStrideType, mlir::Type srcOuterStrideType);
void printDmaPadGroup(mlir::OpAsmPrinter &printer, mlir::Value value, mlir::Value left, mlir::Value right);
void printDmaPadTypes(mlir::OpAsmPrinter &printer, mlir::Type valueType, mlir::Type leftType, mlir::Type rightType);
void printDmaTripleGroup(mlir::OpAsmPrinter &printer, llvm::StringRef keyword, mlir::Value first, mlir::Value second, mlir::Value third);
void printDmaTripleTypes(mlir::OpAsmPrinter &printer, llvm::StringRef keyword, mlir::Type first, mlir::Type second, mlir::Type third);
void printMteL1L0OptionalOperandsOp( mlir::OpAsmPrinter &printer, mlir::Operation *operation, mlir::Value source, mlir::Value destination, llvm::ArrayRef<mlir::Value> shapeOperands, llvm::ArrayRef<llvm::StringRef> shapeNames, llvm::ArrayRef<mlir::Value> fullOperands, llvm::ArrayRef<llvm::StringRef> fullNames);
void printStructuredAccStoreClauses( mlir::OpAsmPrinter &printer, std::optional<mlir::pto::AccStoreUnitFlagCtrl> unitFlag, mlir::Value preQuant, std::optional<mlir::pto::AccStoreQuantPreMode> preQuantMode, mlir::Value preRelu, std::optional<mlir::pto::ReluPreMode> preReluMode, mlir::Value clipValue, std::optional<mlir::pto::AccStoreMode> mode, mlir::Value split, mlir::Value loop0SrcStride, mlir::Value loop3Count, mlir::Value loop3SrcStride, mlir::Value loop3DstStride, std::optional<mlir::pto::AccStoreSatMode> satMode, std::optional<mlir::pto::AccStoreAtomicType> atomicType, std::optional<mlir::pto::AccStoreAtomicOp> atomicOp);
void printStructuredAccStoreOptionalTypes( mlir::OpAsmPrinter &printer, mlir::Value preQuant, mlir::Value preRelu, mlir::Value clipValue, mlir::Value split, mlir::Value loop0SrcStride, mlir::Value loop3Count, mlir::Value loop3SrcStride, mlir::Value loop3DstStride);
mlir::ParseResult resolveCubeBridgeOperands( mlir::OpAsmParser &parser, mlir::OperationState &result, bool usesNamedOperands, mlir::OpAsmParser::UnresolvedOperand source, mlir::Type sourceType, mlir::OpAsmParser::UnresolvedOperand destination, mlir::Type destinationType, llvm::SmallVectorImpl<CubeBridgeLoadAsmOperand> &namedOperands, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &legacyOperands, llvm::SmallVectorImpl<mlir::Type> &legacyTypes, llvm::SmallVectorImpl<int32_t> &segmentSizes);
mlir::ParseResult resolveDmaBasicOperands( mlir::OpAsmParser &parser, mlir::OperationState &result, mlir::OpAsmParser::UnresolvedOperand source, mlir::Type sourceType, mlir::OpAsmParser::UnresolvedOperand destination, mlir::Type destinationType, mlir::OpAsmParser::UnresolvedOperand lenBurst, mlir::Type lenBurstType, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &nburstOperands, llvm::SmallVectorImpl<mlir::Type> &nburstTypes);
mlir::ParseResult resolveDmaLoopOperands( mlir::OpAsmParser &parser, mlir::OperationState &result, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &loopCountOperands, llvm::SmallVectorImpl<mlir::Type> &loopCountTypes, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &loopSrcStrideOperands, llvm::SmallVectorImpl<mlir::Type> &loopSrcStrideTypes, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &loopDstStrideOperands, llvm::SmallVectorImpl<mlir::Type> &loopDstStrideTypes);
mlir::ParseResult resolveDmaTripleOperands( mlir::OpAsmParser &parser, mlir::OperationState &result, bool hasL2CacheCtl, mlir::OpAsmParser::UnresolvedOperand l2CacheCtl, mlir::Type l2CacheCtlType, mlir::OpAsmParser::UnresolvedOperand source, mlir::Type sourceType, mlir::OpAsmParser::UnresolvedOperand destination, mlir::Type destinationType, mlir::OpAsmParser::UnresolvedOperand lenBurst, mlir::Type lenBurstType, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &nburstOperands, llvm::SmallVectorImpl<mlir::Type> &nburstTypes, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &loopCountOperands, llvm::SmallVectorImpl<mlir::Type> &loopCountTypes, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &loopSrcStrideOperands, llvm::SmallVectorImpl<mlir::Type> &loopSrcStrideTypes, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &loopDstStrideOperands, llvm::SmallVectorImpl<mlir::Type> &loopDstStrideTypes);
mlir::ParseResult resolveMteGmL1FracOperands( mlir::OpAsmParser &parser, mlir::OperationState &result, mlir::OpAsmParser::UnresolvedOperand source, mlir::Type sourceType, mlir::OpAsmParser::UnresolvedOperand destination, mlir::Type destinationType, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &shapeOperands, llvm::SmallVectorImpl<mlir::Type> &shapeTypes, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &srcLayoutOperands, llvm::SmallVectorImpl<mlir::Type> &srcLayoutTypes, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &dstGroupOperands, llvm::SmallVectorImpl<mlir::Type> &dstGroupTypes, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &ctrlOperands, llvm::SmallVectorImpl<mlir::Type> &ctrlTypes);
mlir::ParseResult resolveMteGmUbOperands( mlir::OpAsmParser &parser, mlir::OperationState &result, mlir::OpAsmParser::UnresolvedOperand source, mlir::Type sourceType, mlir::OpAsmParser::UnresolvedOperand destination, mlir::Type destinationType, mlir::OpAsmParser::UnresolvedOperand l2CacheCtl, mlir::Type l2CacheCtlType, mlir::OpAsmParser::UnresolvedOperand lenBurst, mlir::Type lenBurstType, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &nburstOperands, llvm::SmallVectorImpl<mlir::Type> &nburstTypes, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &loopCountOperands, llvm::SmallVectorImpl<mlir::Type> &loopCountTypes, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &loopSrcStrideOperands, llvm::SmallVectorImpl<mlir::Type> &loopSrcStrideTypes, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &loopDstStrideOperands, llvm::SmallVectorImpl<mlir::Type> &loopDstStrideTypes, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &padOperands, llvm::SmallVectorImpl<mlir::Type> &padTypes);
mlir::ParseResult resolveMteL0cGmOperands( mlir::OpAsmParser &parser, mlir::OperationState &result, mlir::OpAsmParser::UnresolvedOperand source, mlir::Type sourceType, mlir::OpAsmParser::UnresolvedOperand destination, mlir::Type destinationType, mlir::OpAsmParser::UnresolvedOperand m, mlir::Type mType, mlir::OpAsmParser::UnresolvedOperand n, mlir::Type nType, mlir::OpAsmParser::UnresolvedOperand srcStride, mlir::Type srcStrideType, mlir::OpAsmParser::UnresolvedOperand dstStride, mlir::Type dstStrideType, mlir::OpAsmParser::UnresolvedOperand sid, mlir::Type sidType, mlir::OpAsmParser::UnresolvedOperand l2CacheCtrl, mlir::Type l2CacheCtrlType, StructuredAccStoreAsmState &state);
mlir::ParseResult resolveMteL0cL1Operands( mlir::OpAsmParser &parser, mlir::OperationState &result, mlir::OpAsmParser::UnresolvedOperand source, mlir::Type sourceType, mlir::OpAsmParser::UnresolvedOperand destination, mlir::Type destinationType, mlir::OpAsmParser::UnresolvedOperand m, mlir::Type mType, mlir::OpAsmParser::UnresolvedOperand n, mlir::Type nType, mlir::OpAsmParser::UnresolvedOperand srcStride, mlir::Type srcStrideType, mlir::OpAsmParser::UnresolvedOperand dstStride, mlir::Type dstStrideType, StructuredAccStoreAsmState &state);
mlir::ParseResult resolveMteL0cUbOperands( mlir::OpAsmParser &parser, mlir::OperationState &result, mlir::OpAsmParser::UnresolvedOperand source, mlir::Type sourceType, mlir::OpAsmParser::UnresolvedOperand destination, mlir::Type destinationType, mlir::OpAsmParser::UnresolvedOperand m, mlir::Type mType, mlir::OpAsmParser::UnresolvedOperand n, mlir::Type nType, mlir::OpAsmParser::UnresolvedOperand srcStride, mlir::Type srcStrideType, mlir::OpAsmParser::UnresolvedOperand dstStride, mlir::Type dstStrideType, bool hasSubBlockId, mlir::OpAsmParser::UnresolvedOperand subBlockId, mlir::Type subBlockIdType, const StructuredAccStoreAsmState &state);
void setMteGmUbSegmentSizes(mlir::OperationState &result, int32_t loopGroupCount, size_t padOperandCount);
void setMteL0cGmSegmentSizes(mlir::OperationState &result, const StructuredAccStoreAsmState &st);
void setMteL0cL1SegmentSizes(mlir::OperationState &result, const StructuredAccStoreAsmState &st);
void setMteUbGmSegmentSizes(mlir::OperationState &result, bool hasL2CacheCtl, size_t loopGroupCount);
mlir::ParseResult validateMteGmL1FracOperands( mlir::OpAsmParser &parser, size_t shapeOps, size_t shapeTypes, size_t srcLayoutOps, size_t srcLayoutTypes, size_t dstGroupOps, size_t dstGroupTypes, size_t ctrlOps, size_t ctrlTypes);
mlir::LogicalResult verifyDmaLoadStoreLoopGroups(mlir::Operation *op, mlir::ValueRange loopCounts, mlir::ValueRange loopSrcStrides, mlir::ValueRange loopDstStrides);
mlir::ParseResult verifyDmaLoopGroupConsistency( mlir::OpAsmParser &parser, size_t countOperands, size_t srcStrideOperands, size_t dstStrideOperands, size_t countTypes, size_t srcStrideTypes, size_t dstStrideTypes);
mlir::LogicalResult verifyMxLoadAlignment(mlir::Operation *op, mlir::Value source, mlir::Value destination);
mlir::LogicalResult verifyMxLoadOperands(mlir::Operation *op, llvm::ArrayRef<mlir::Value> shapeOperands, llvm::ArrayRef<llvm::StringRef> shapeNames, llvm::ArrayRef<mlir::Value> fullOperands);
mlir::LogicalResult verifyNestedInVecScope(mlir::Operation *op, llvm::StringRef opNameForDiag);
mlir::LogicalResult verifyNonLowPrecisionVRegElementTypeLike( mlir::Operation *op, mlir::Type type, llvm::StringRef roleDescription);
mlir::LogicalResult verifyNotNestedInVecScope(mlir::Operation *op, llvm::StringRef opNameForDiag);
mlir::LogicalResult verifyStructuredAccStoreLike( mlir::Operation *op, mlir::Type srcType, mlir::Type dstType, mlir::Value preQuant, mlir::Value preRelu, mlir::Value clipValue, mlir::Value split, mlir::Value loop0SrcStride, mlir::Value loop3Count, mlir::Value loop3SrcStride, mlir::Value loop3DstStride, std::optional<mlir::pto::AccStoreUnitFlagCtrl> unitFlag, std::optional<mlir::pto::AccStoreQuantPreMode> preQuantMode, std::optional<mlir::pto::ReluPreMode> preReluMode, std::optional<mlir::pto::AccStoreMode> mode, std::optional<mlir::pto::AccStoreAtomicType> atomicType, std::optional<mlir::pto::AccStoreAtomicOp> atomicOp, bool allowAtomic);

// Batch1: VPTO.cpp/MTE/DMA 共用小函数
[[maybe_unused]] 
inline mlir::LogicalResult verifyVRegTypeLike(mlir::Operation *op, mlir::Type type,
                                       llvm::StringRef roleDescription) {
  auto vecType = mlir::dyn_cast<mlir::pto::VRegType>(type);
  if (!vecType) {
    return op->emitOpError() << roleDescription << " must be !pto.vreg<...>";
  }

  return mlir::pto::VRegType::verify(
      [&]() { return op->emitOpError() << roleDescription << " "; },
      vecType.getElementCount(), vecType.getElementType());
}

[[maybe_unused]] 

inline int64_t getBufferElementByteSize(mlir::Type type) {
  mlir::Type elementType;
  if (auto ptrType = mlir::dyn_cast<mlir::pto::PtrType>(type)) {
    elementType = ptrType.getElementType();
  } else if (auto memrefType = mlir::dyn_cast<mlir::BaseMemRefType>(type)) {
    elementType = memrefType.getElementType();
  } else {
    return 0;
  }

  return mlir::pto::getPTOStorageElemByteSize(elementType);
}

[[maybe_unused]] 
inline mlir::Type getBufferElementType(mlir::Type type) {
  if (auto ptrType = mlir::dyn_cast<mlir::pto::PtrType>(type)) {
    return ptrType.getElementType();
  }
  if (auto memrefType = mlir::dyn_cast<mlir::BaseMemRefType>(type)) {
    return memrefType.getElementType();
  }
  return {};
}

[[maybe_unused]] 
inline std::optional<mlir::pto::AddressSpace> getBufferAddressSpace(mlir::Type type) {
  if (auto ptrType = mlir::dyn_cast<mlir::pto::PtrType>(type)) {
    return ptrType.getMemorySpace().getAddressSpace();
  }
  if (auto memrefType = mlir::dyn_cast<mlir::BaseMemRefType>(type)) {
    if (auto space =
            mlir::dyn_cast_or_null<mlir::pto::AddressSpaceAttr>(memrefType.getMemorySpace())) {
      return space.getAddressSpace();
    }
    if (auto intSpace = mlir::dyn_cast_or_null<mlir::IntegerAttr>(memrefType.getMemorySpace())) {
      return static_cast<mlir::pto::AddressSpace>(intSpace.getInt());
    }
  }
  return std::nullopt;
}

template <typename BridgeLoadOp>
static mlir::LogicalResult verifyCubeBridgeLoadLikeOp(BridgeLoadOp op,
                                                mlir::pto::AddressSpace expectedDstSpace,
                                                llvm::StringRef dstName) {
  if (!isBufferLike(op.getSource().getType()) ||
      !isBufferLike(op.getDestination().getType())) {
    return op.emitOpError("requires buffer-like source and destination");
  }

  if (getBufferAddressSpace(op.getSource().getType()) != mlir::pto::AddressSpace::MAT) {
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

  return mlir::success();
}

[[maybe_unused]] 
inline mlir::ParseResult parseRequiredOperandWithComma(
    mlir::OpAsmParser &parser, mlir::OpAsmParser::UnresolvedOperand &operand) {
  if (parser.parseOperand(operand)) {
    return mlir::failure();
  }
  (void)parser.parseOptionalComma();
  return mlir::success();
}

[[maybe_unused]] 

inline mlir::LogicalResult checkNonNegativeConst(mlir::Operation *op, mlir::Value value,
                                           llvm::StringRef name) {
  if (!value) {
    return mlir::success();
  }
  llvm::APInt intValue;
  if (matchPattern(value, mlir::m_ConstantInt(&intValue)) && intValue.isNegative()) {
    return op->emitOpError() << name << " must be non-negative";
  }
  return mlir::success();
}

[[maybe_unused]] 
inline mlir::LogicalResult verifyCubeBridgeLoadStart(mlir::Operation *op, mlir::Value firstStart,
                                               llvm::StringRef firstName,
                                               mlir::Value secondStart,
                                               llvm::StringRef secondName) {
  auto checkNonNegativeConst = [op](mlir::Value value, llvm::StringRef name) -> mlir::LogicalResult {
    llvm::APInt intValue;
    if (matchPattern(value, mlir::m_ConstantInt(&intValue)) && intValue.isNegative()) {
      return op->emitOpError() << name << " must be non-negative";
    }
    return mlir::success();
  };
  if (mlir::failed(checkNonNegativeConst(firstStart, firstName)) ||
      mlir::failed(checkNonNegativeConst(secondStart, secondName))) {
    return mlir::failure();
  }
  return mlir::success();
}

template <typename OpTy>
static mlir::LogicalResult verifyCubeBridgeLoadStart(OpTy op) {
  return verifyCubeBridgeLoadStart(op.getOperation(), op.getStartRow(),
                                   "start_row", op.getStartCol(), "start_col");
}

[[maybe_unused]] 
inline mlir::LogicalResult verifyStaticControlRange(mlir::Operation *op, mlir::Value value,
                                              llvm::StringRef name, int64_t min,
                                              int64_t max) {
  llvm::APInt intValue;
  if (!matchPattern(value, mlir::m_ConstantInt(&intValue))) {
    return mlir::success();
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
  return mlir::success();
}


// Batch1: 跨文件实例化的模板定义(本体必须在头文件)
template <typename OpTy>
[[maybe_unused]] static void addStructuredAccStoreAttrs(mlir::OperationState &result,
                                       mlir::Builder &builder,
                                       const StructuredAccStoreAsmState &state) {
  if (state.mode) {
    result.addAttribute("mode", mlir::pto::AccStoreModeAttr::get(builder.getContext(),
                                                      *state.mode));
  }
  if (state.unitFlag) {
    result.addAttribute("unit_flag",
                        mlir::pto::AccStoreUnitFlagCtrlAttr::get(builder.getContext(),
                                                      *state.unitFlag));
  }
  if (state.preQuantMode) {
    result.addAttribute("pre_quant_mode",
                        mlir::pto::AccStoreQuantPreModeAttr::get(builder.getContext(),
                                                      *state.preQuantMode));
  }
  if (state.preReluMode) {
    result.addAttribute("pre_relu_mode",
                        mlir::pto::ReluPreModeAttr::get(builder.getContext(),
                                             *state.preReluMode));
  }
  if (state.atomicType) {
    result.addAttribute("atomic_type",
                        mlir::pto::AccStoreAtomicTypeAttr::get(builder.getContext(),
                                                    *state.atomicType));
  }
  if (state.atomicOp) {
    result.addAttribute("atomic_op",
                        mlir::pto::AccStoreAtomicOpAttr::get(builder.getContext(),
                                                  *state.atomicOp));
  }
  if (state.satMode) {
    result.addAttribute("sat_mode",
                        mlir::pto::AccStoreSatModeAttr::get(builder.getContext(),
                                                 *state.satMode));
  }
}

template <typename OpTy>
[[maybe_unused]] static void setStructuredAccStoreSegmentSizes(mlir::OperationState &result,
                                              llvm::ArrayRef<int32_t> segmentSizes) {
  auto &segments = result.getOrAddProperties<typename OpTy::Properties>()
                       .operandSegmentSizes;
  llvm::copy(segmentSizes, segments.begin());
}

template <typename OpTy>
[[maybe_unused]] static void setCubeBridgeLoadOperandSegmentSizes(
    mlir::OperationState &result, llvm::ArrayRef<int32_t> segmentSizes) {
  auto &segments = result.getOrAddProperties<typename OpTy::Properties>()
                       .operandSegmentSizes;
  llvm::copy(segmentSizes, segments.begin());
}

template <typename OpTy>
[[maybe_unused]] static mlir::ParseResult parseMteL1L0OptionalOperandsOp(
    mlir::OpAsmParser &parser, mlir::OperationState &result, llvm::ArrayRef<llvm::StringRef> shapeNames,
    llvm::ArrayRef<llvm::StringRef> fullNames, llvm::StringRef operandDescription = "operands") {
  mlir::OpAsmParser::UnresolvedOperand source;
  mlir::OpAsmParser::UnresolvedOperand destination;
  if (parser.parseOperand(source) || parser.parseComma() ||
      parser.parseOperand(destination)) {
    return mlir::failure();
  }
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 6> legacyOperands; // 6:six full positional
  llvm::SmallVector<CubeBridgeLoadAsmOperand, 10> namedOperands(10); // 10: 数组长度
  llvm::SmallVector<unsigned, 10> namedOperandOrder;
  bool usesNamedOperands = false;
  if (mlir::failed(parseCubeBridgeOptionalOperands(
          parser, shapeNames, fullNames, legacyOperands, namedOperands,
          namedOperandOrder, usesNamedOperands))) {
    return mlir::failure();
  }
  if (!usesNamedOperands && legacyOperands.size() != mlir::pto::kValue4 &&
      legacyOperands.size() != mlir::pto::kValue6) {
    return parser.emitError(
               parser.getCurrentLocation(),
               "expects either four shape-derived or six full positional ")
           << operandDescription;
  }
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return mlir::failure();
  }
  mlir::Type sourceType;
  mlir::Type destinationType;
  if (parser.parseType(sourceType) || parser.parseComma() ||
      parser.parseType(destinationType)) {
    return mlir::failure();
  }
  llvm::SmallVector<mlir::Type, mlir::pto::kValue6> legacyTypes;
  if (mlir::failed(parseCubeBridgeOptionalTypes(parser, usesNamedOperands,
                                          namedOperandOrder, namedOperands,
                                          legacyOperands, legacyTypes))) {
    return mlir::failure();
  }
  llvm::SmallVector<int32_t, 12> segmentSizes(12, 0); // 12:数组长度
  segmentSizes[0] = 1;
  segmentSizes[1] = 1;
  if (mlir::failed(resolveCubeBridgeOperands(
          parser, result, usesNamedOperands, source, sourceType,
          destination, destinationType, namedOperands, legacyOperands,
          legacyTypes, segmentSizes))) {
    return mlir::failure();
  }
  setCubeBridgeLoadOperandSegmentSizes<OpTy>(result, segmentSizes);
  return mlir::success();
}

// Batch1: 跨文件实例化的模板定义 II
template <typename CopyOp>
mlir::LogicalResult verifyCopyElemByteWidths(CopyOp op) {
  int64_t sourceElemBytes = getBufferElementByteSize(op.getSource().getType());
  int64_t destinationElemBytes =
      getBufferElementByteSize(op.getDestination().getType());
  if (sourceElemBytes <= 0 || destinationElemBytes <= 0) {
    return op.emitOpError("requires copy source and destination element types with known byte width");
  }
  if (sourceElemBytes != destinationElemBytes) {
    return op.emitOpError("requires source and destination element byte widths to match");
  }

  return mlir::success();
}


template <typename CopyOp>
mlir::LogicalResult verifyCopyGmToUbufOp(CopyOp op, bool expectSourceGM) {
  if (!isBufferLike(op.getSource().getType()) ||
      !isBufferLike(op.getDestination().getType())) {
    return op.emitOpError(
        "requires typed !pto.ptr or memref source and destination");
  }

  MemoryRole sourceRole = classifyMemoryRole(op.getSource().getType());
  MemoryRole destinationRole = classifyMemoryRole(op.getDestination().getType());
  bool directionMatches;
  if (expectSourceGM) {
    directionMatches =
        sourceRole != MemoryRole::UB && destinationRole != MemoryRole::GM;
  } else {
    directionMatches =
        sourceRole != MemoryRole::GM && destinationRole != MemoryRole::UB;
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
mlir::LogicalResult verifyCopyCbufToUbufLikeOp(CopyOp op) {
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
