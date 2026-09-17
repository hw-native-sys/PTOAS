// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTO.cpp - PTO Dialect ----------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"

#include "mlir/Interfaces/InferIntRangeInterface.h"

// PTO IR implementation is grouped by semantic responsibility. The files
// remain textual fragments to preserve declaration order and internal linkage.

// Shared names, type support, and target properties.
#include "PTOCore/PTOCoreImports.cpp"
#include "PTOCore/PTOShapeTypeSupport.cpp"
#include "PTOCore/PTOTargetAndAlignment.cpp"
#include "PTOCore/PTOLowPrecisionTypes.cpp"

// Custom assembly for types, gather/scatter, and collective operations.
#include "PTOCore/PTOTypeAssembly.cpp"
#include "PTOCore/PTOGatherScatterAssemblyInput.cpp"
#include "PTOCore/PTOGatherScatterAssemblyResult.cpp"
#include "PTOCore/PTOCollectiveAssembly.cpp"

// Views, layout inference, pointer types, and dialect-level verification.
#include "PTOCore/PTOPartitionViewAssembly.cpp"
#include "PTOCore/PTOTileLayoutVerification.cpp"
#include "PTOCore/PTORowReductionLayout.cpp"
#include "PTOCore/PTOReductionScratchLayout.cpp"
#include "PTOCore/PTORegionAndViewVerification.cpp"
#include "PTOCore/PTOPointerAndStructVerification.cpp"
#include "PTOCore/PTODialectAndEntryVerification.cpp"

// Allocation, memory movement, synchronization, and shared tile checks.
#include "PTOTile/PTOTileAllocationAndLoad.cpp"
#include "PTOTile/PTOPrefetchAndSyncSet.cpp"
#include "PTOTile/PTOSynchronizationOps.cpp"
#include "PTOTile/PTOStoreVerification.cpp"
#include "PTOTile/PTOCommunicationShapeVerification.cpp"
#include "PTOTile/PTOExtentCompatibility.cpp"
#include "PTOTile/PTOTileBufferVerification.cpp"

// Arithmetic, matrix, column, conversion, and data-movement verification.
#include "PTOTile/PTOArithmeticVerification.cpp"
#include "PTOTile/PTOMatrixOperandVerification.cpp"
#include "PTOTile/PTOMatrixQuantAndAdd.cpp"
#include "PTOTile/PTOAxpyBitwiseConcat.cpp"
#include "PTOTile/PTOConcatSequenceTriangle.cpp"
#include "PTOTile/PTOCompareAndColumnExpand.cpp"
#include "PTOTile/PTOColumnExpandAndReduction.cpp"
#include "PTOTile/PTOColumnReductionConvertRandom.cpp"
#include "PTOTile/PTOExpAndTransferVerification.cpp"
#include "PTOTile/PTOExtractVerification.cpp"
#include "PTOTile/PTOInsertVerification.cpp"
#include "PTOTile/PTOInsertAndFillPad.cpp"
#include "PTOTile/PTOGatherVerification.cpp"
#include "PTOTile/PTOGatherAndExtremaOps.cpp"
#include "PTOTile/PTOMoveLayoutVerification.cpp"
#include "PTOTile/PTOMoveScalarAndCacheOps.cpp"

// Cache, buffer, gather/scatter, merge, and quantization operations.
#include "PTOTile/PTOCacheBarrierAndFlagOps.cpp"
#include "PTOTile/PTOBufferSyncAndMatrixOps.cpp"
#include "PTOTile/PTOMatrixSetValueHistogram.cpp"
#include "PTOTile/PTOHistogramScaleAndScatter.cpp"
#include "PTOTile/PTOMemoryGatherAndConvert.cpp"
#include "PTOTile/PTOConvertAndMergeSortAssembly.cpp"
#include "PTOTile/PTOMergeSortMultiplyShift.cpp"
#include "PTOTile/PTOUnaryBitwiseAndPartialOps.cpp"
#include "PTOTile/PTOPartialAndPreluQuantAssembly.cpp"
#include "PTOTile/PTOQuantizationAssembly.cpp"
#include "PTOTile/PTOQuantizationVerification.cpp"
#include "PTOTile/PTOMxQuantShapeVerification.cpp"
#include "PTOTile/PTOMxQuantAndRemainderOps.cpp"

// Scalar power/remainder, row expansion, selection, and transpose operations.
#include "PTOTile/PTORemainderAndPowerOps.cpp"
#include "PTOTile/PTOPowerShapeAndReshapeOps.cpp"
#include "PTOTile/PTORowExpandAndSortAssembly.cpp"
#include "PTOTile/PTORowExpandBinaryAssembly.cpp"
#include "PTOTile/PTORowExpandBinaryVerification.cpp"
#include "PTOTile/PTORowExpandReductionAssembly.cpp"
#include "PTOTile/PTOOptionalScratchAssembly.cpp"
#include "PTOTile/PTODeinterleaveAndRowProduct.cpp"
#include "PTOTile/PTOScatterAndSelect.cpp"
#include "PTOTile/PTOShiftSortSubtractOps.cpp"
#include "PTOTile/PTOTransposeXorPrintMatrix.cpp"

// Concrete types, subviews, and memory-effect interfaces.
#include "PTOInterfaces/PTOMatrixInferenceAndTileTypes.cpp"
#include "PTOInterfaces/PTOStructTypesAndStridedLayout.cpp"
#include "PTOInterfaces/PTOSubViewAssembly.cpp"
#include "PTOInterfaces/PTOSubViewInference.cpp"
#include "PTOInterfaces/PTOSubViewAndCoreEffects.cpp"
#include "PTOInterfaces/PTOMemoryAndElementwiseEffects.cpp"
#include "PTOInterfaces/PTOQuantReductionSelectionEffects.cpp"
#include "PTOInterfaces/PTOMatrixEffectsAndPipelineScopes.cpp"

// Frontend/internal pipelines, asynchronous communication, and SIMT support.
#include "PTOPipeline/PTOFrontendPipelineAssembly.cpp"
#include "PTOPipeline/PTOFrontendPipelineResources.cpp"
#include "PTOPipeline/PTOFrontendPipelineLookup.cpp"
#include "PTOPipeline/PTOFrontendPipelineDirection.cpp"
#include "PTOPipeline/PTOFixpipeTypeCompatibility.cpp"
#include "PTOPipeline/PTOAsyncCommunicationVerification.cpp"
#include "PTOPipeline/PTOCollectiveAndPipelineVerification.cpp"
#include "PTOPipeline/PTOAivPipelineVerification.cpp"
#include "PTOPipeline/PTOFrontendPipelineTransferOps.cpp"
#include "PTOPipeline/PTOInternalPipelineOps.cpp"
#include "PTOPipeline/PTOSimtVerificationAndAsyncEffects.cpp"
#include "PTOPipeline/PTOCollectivePipelineEffectsAndConvertAssembly.cpp"

// Remaining custom assembly hooks and generated operation definitions.
#include "PTOPipeline/PTOCachePolicyAssemblyAndGeneratedOps.cpp"

//===----------------------------------------------------------------------===//
// InferIntRangeInterface: PTO runtime query ops (i64)
//===----------------------------------------------------------------------===//

// Bit width of the i64 storage used for block-query range inference.
constexpr unsigned kBlockQueryRangeBitWidth = 64;

// get_block_idx returns the linear index of the current block within the task,
// documented as [0, BlockNum - 1]. The value is a 32-bit quantity on the
// hardware, and the LLVM lowering rounds it through i32 on every path (the
// tpe intrinsic is i32; the AIC path truncates the i64 intrinsic and
// zero-extends back), matching the CCE frontend where
// __builtin_cce_get_block_idx is int32_t. Report [0, INT32_MAX] accordingly:
// this is what the lowered code actually computes, and it is tight enough for
// range-driven consumers to fold address arithmetic that adds the block index
// to bounded offsets, while a multiple of the index still fails u32 bounds
// and stays wide.
void pto::GetBlockIdxOp::inferResultRanges(
    ::llvm::ArrayRef<::mlir::ConstantIntRanges> operandRanges,
    ::mlir::SetIntRangeFn setResultRange) {
  setResultRange(
      getResult(),
      ConstantIntRanges::fromUnsigned(APInt::getMinValue(kBlockQueryRangeBitWidth),
                                      APInt(kBlockQueryRangeBitWidth, INT32_MAX)));
}

// get_subblock_idx returns the vector-core ID, documented as [0, 1].
void pto::GetSubBlockIdxOp::inferResultRanges(
    ::llvm::ArrayRef<::mlir::ConstantIntRanges> operandRanges,
    ::mlir::SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 ConstantIntRanges::fromUnsigned(APInt(kBlockQueryRangeBitWidth, 0),
                                                 APInt(kBlockQueryRangeBitWidth, 1)));
}

// Block/subblock counts are non-negative. Do NOT claim >= 1: existing
// kernels guard with `cmpi sge block_num, 1` and rely on that comparison
// staying dynamic (a >= 1 range would fold the guard away).
void pto::GetBlockNumOp::inferResultRanges(
    ::llvm::ArrayRef<::mlir::ConstantIntRanges> operandRanges,
    ::mlir::SetIntRangeFn setResultRange) {
  setResultRange(
      getResult(),
      ConstantIntRanges::fromUnsigned(APInt::getMinValue(kBlockQueryRangeBitWidth),
                                      APInt::getSignedMaxValue(kBlockQueryRangeBitWidth)));
}

void pto::GetSubBlockNumOp::inferResultRanges(
    ::llvm::ArrayRef<::mlir::ConstantIntRanges> operandRanges,
    ::mlir::SetIntRangeFn setResultRange) {
  setResultRange(
      getResult(),
      ConstantIntRanges::fromUnsigned(APInt::getMinValue(kBlockQueryRangeBitWidth),
                                      APInt::getSignedMaxValue(kBlockQueryRangeBitWidth)));
}
