// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// the CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. This software is provided on an "AS IS" BASIS.

#pragma once
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/VPTOLLVMEmitter.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::pto {
struct PlannedDecl { std::string name; FunctionType type; };
struct LoweringState { SmallVector<PlannedDecl> plannedDecls; };
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
FailureOr<Value> getVPTOStructFieldAddress(ConversionPatternRewriter &rewriter, Location loc, Value root, pto::StructType rootType, ArrayRef<int64_t> path);
std::string getElementTypeFragment(Type type);
std::string getLowPrecisionElementFragment(Type type);
std::string getMemoryElementTypeFragment(Type type);
std::string getCopyElementFragment(Type type);
Type getElementTypeFromVectorLike(Type type);
std::optional<int64_t> getElementCountFromVectorLike(Type type);
bool isOnePointStoreDist(StringRef dist);
Value castIntegerLikeTo(Operation *anchor, Value value, Type targetType);
FailureOr<Value> reinterpretPointerToAddrSpace(Operation *anchor, Value value,
                                                unsigned targetAddressSpace);
FailureOr<Value> packLoopPair(Operation *anchor, Value low, Value high);
FailureOr<Value> packLoopSize(Operation *anchor, Value loop2, Value loop1);
void populateVPTOBasicPatterns(TypeConverter &typeConverter,
                                RewritePatternSet &patterns,
                                LoweringState &state);
void populateVPTOVectorUnaryPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     LoweringState &state);
void populateVPTOVectorCompactionPatterns(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          LoweringState &state);
void populateVPTOVectorMulaPatterns(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    LoweringState &state);
void populateVPTOVectorBinaryPatterns(TypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      LoweringState &state);
void populateVPTOVectorVmullPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     LoweringState &state);
void populateVPTOVectorCarryPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     LoweringState &state);
void populateVPTOVectorReductionPatterns(TypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         LoweringState &state);
void populateVPTOVectorPredicatePatterns(TypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         LoweringState &state);
void populateVPTOMemoryMaskPatterns(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    LoweringState &state);
void populateVPTOMemoryUbufPatterns(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    LoweringState &state);
void populateVPTOCubeMadPatterns(TypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 LoweringState &state);
void populateVPTOCubeMemoryPatterns(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    LoweringState &state);
void populateVPTOVectorMemoryPatterns(TypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      LoweringState &state);
void populateVPTOVectorGatherPatterns(TypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      LoweringState &state);
void populateVPTOUbufPatterns(TypeConverter &typeConverter,
                              RewritePatternSet &patterns,
                              LoweringState &state, const std::string &march);
void populateVPTOScalarAndRuntimePatterns(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          LoweringState &state);
void populateVPTOSyncAndConfigPatterns(TypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       LoweringState &state);
void populateVPTOVcvtPatterns(TypeConverter &typeConverter,
                              RewritePatternSet &patterns,
                              LoweringState &state);
bool needsV300CtrlModeForVPTOFunc(func::FuncOp funcOp);
class VPTOTypeConverter final : public TypeConverter { public: explicit VPTOTypeConverter(MLIRContext *context); };
} // namespace mlir::pto
