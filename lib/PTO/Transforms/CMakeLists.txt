# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(_ptoas_transforms_private_link_libs)
if(TARGET vfsim::native_core)
  list(APPEND _ptoas_transforms_private_link_libs
    $<BUILD_INTERFACE:vfsim::native_core>)
endif()
if(TARGET vfsim::ir_planner)
  list(APPEND _ptoas_transforms_private_link_libs
    $<BUILD_INTERFACE:vfsim::ir_planner>)
endif()

# VMIToVPTO lowering internals are textual include units of VMIToVPTO.cpp.
# They are listed here so LLVM per-directory source check is satisfied and
# marked HEADER_FILE_ONLY so they are not compiled as separate translation
# units (VMIToVPTO.cpp includes them inside its own anonymous namespace).
set(_ptoas_vmi_split_units
  VMIToVPTOConversionInternals.cpp
  VMIToVPTO/VMIToVPTOMemoryInternals.cpp
  VMIToVPTO/VMIToVPTOMaskInternals.cpp
  VMIToVPTO/VMIToVPTODataLayoutInternals.cpp
  VMIToVPTO/VMIToVPTOPatternInternals0.cpp
  VMIToVPTO/VMIToVPTOPatternInternals1.cpp
  VMIToVPTO/VMIToVPTOPatternInternals2.cpp
  VMIToVPTO/VMIToVPTOPatternInternals3.cpp
  VMIToVPTO/VMIToVPTOPatternInternals4.cpp
  VMIToVPTO/VMIToVPTOPatternInternals5.cpp
  VMIToVPTO/VMIToVPTOPatternInternals6.cpp
  VMIToVPTO/VMIToVPTOPatternInternals7.cpp
  VMIToVPTO/VMIToVPTOPatternInternals8.cpp
  VMIToVPTO/VMIToVPTOPatternInternals9.cpp
)
set_source_files_properties(${_ptoas_vmi_split_units} PROPERTIES HEADER_FILE_ONLY TRUE)

add_mlir_dialect_library(PTOTransforms
  ENABLE_AGGREGATION

  TileFusion/FusionAnalysis.cpp
  TileFusion/FusionOpSemantics.cpp
  TileFusion/PTOPreFusionAnalysis.cpp
  TileFusion/PTOPrintPreFusionAnalysis.cpp
  TileFusion/PTOFusionPlan.cpp
  TileFusion/PTOOpScheduling.cpp
  TileFusion/PTOMarkLastUse.cpp
  TileFusion/PTOFusionRegionGen.cpp
  TileFusion/PTOLowLevelLoopFusion.cpp
  TileFusion/PTOFusionPredicateElision.cpp
  TileFusion/PTOFusionLoadStoreElision.cpp
  TileFusion/PTOVexpdifFusion.cpp
  TileFusion/PTOUnrollAfterLoopFusion.cpp
  TileFusion/PTOFlattenFusionRegion.cpp
  VPTO/VPTOCANN900LLVMEmitterCommon.cpp
  VPTO/VPTOCANN900LLVMEmitterUbuf.cpp
  VPTO/VPTOCANN900LLVMEmitter.cpp
  VPTOCANN900LLVMEmitterArithmeticPatterns.cpp
  VPTO/VPTOCANN900LLVMEmitterCalleeCore.cpp
  VPTO/VPTOCANN900LLVMEmitterCalleeMemory.cpp
  VPTOCANN900LLVMEmitterMemoryPatterns.cpp
  VPTO/VPTOCANN900LLVMEmitterPacking.cpp
  VPTO/VPTOCANN900LLVMEmitterPipeline.cpp
  VPTOCANN900LLVMEmitterScalarPatterns.cpp
  VPTO/VPTOCANN900LLVMEmitterTypeHelpers.cpp
  VPTOCANN900LLVMEmitterTypePatterns.cpp
  VPTO/VPTOLLVMEmitterDispatcher.cpp
  VPTOLLVMEmitterHelper.cpp
  VPTOPtrNormalize.cpp
  VPTO/VPTOPtrCastCleanup.cpp
  VPTO/VPTOCombineReductions.cpp
  VPTO/VPTOOptimizeVcvt.cpp
  VPTO/VPTOMaskSimplify.cpp
  VPTOExpandWrapperOps.cpp
  VPTOSoftPostUpdate.cpp
  VPTO/VPTOGuardedLICM.cpp
  Passes/PTOPrintAddressAnalysis.cpp
  VPTOScheduler/VPTORegPressureTracker.cpp
  VPTOScheduler/VPTOSchedBoundary.cpp
  VPTOScheduler/VPTOSchedDAG.cpp
  VPTOScheduler/VPTOSchedDAGBuilder.cpp
  VPTOScheduler/VPTOSchedModel.cpp
  VPTOScheduler/VPTOSchedRegion.cpp
  VPTOScheduler/VPTOSchedResourceTracker.cpp
  VPTOScheduler/VPTOSchedStrategy.cpp
  VPTOScheduler/VPTOScheduler.cpp
  VPTOScheduler/VPTOSchedulerPass.cpp
  PTOVPTOPtrBoundary.cpp
  VPTO/VPTOBufferMaterialization.cpp
  PTOValidateVPTOIR.cpp
  Passes/PTONarrowVPTOLoopCounters.cpp
  Passes/PTOConvertSCFToCFWithLoopHintsPass.cpp
  Passes/PTOPromotePersistentFragmentLoopsPass.cpp
  Passes/PTOUnrollLoopsPass.cpp
  PTOValidateVMIIR.cpp
  VMIPreAssignmentCombine.cpp
  VMI/VMIControlFlowSupport.cpp
  VMI/VMILegalizeArithSelect.cpp
  VMI/VMIMaskGranularityAssignment.cpp
  VMILowerUnifiedToLegacy.cpp
  VMI/VMINormalizeSignlessIntToUnsigned.cpp
  VMI/VMILayoutRematerializeWeakProducers.cpp
  VMILayoutAssignment.cpp
  VMI/VMILayoutFold.cpp
  VMILayoutPropagation.cpp
  VMILayoutSupport.cpp
  VMI/VMILayoutRematerialize.cpp
  VMILayoutSinkMaterialization.cpp
  VMIToVPTO/VMIToVPTO.cpp
  ${_ptoas_vmi_split_units}
  VPTOStatefulStreamFusion.cpp
  Lowering/PTOExpandSoftLib.cpp
  SIMTPersistentFragmentAnalysis.cpp
  Passes/PTOAnalyzeSIMTPersistentFragment.cpp
  Passes/PTOMaterializeSIMTPersistentFragment.cpp
  Passes/PTOOutlineSIMTSections.cpp
  VPTO/PTOInferVPTOVecScope.cpp

  InsertSync/PTOInsertSync.cpp
  Passes/PTOInjectBarrierAllSync.cpp
  InsertSync/InsertSyncDebug.cpp
  PTOMaterializeImplicitTmp.cpp
  Lowering/PTORematerializeFixpipeVectorQuant.cpp
  Passes/PTOValidateIntToPtrUses.cpp
  TileOp/TileLibService.cpp
  SoftLibService.cpp
  InsertTemplateAttributes.cpp
  ExpandTileOp.cpp
  TileOp/FoldTileBufIntrinsics.cpp
  Lowering/LowerPTOToUBufOps.cpp
  TileOp/PTOLowerToOpLibCalls.cpp
  TileOp/PTOInstantiateAndInlineOpLib.cpp
  PTOToEmitC/PTOToEmitCCommon.cpp
  PTOToEmitC/PTOToEmitCArith.cpp
  PTOToEmitC/PTOToEmitCMemref.cpp
  PTOToEmitC/PTOToEmitCLoadStore.cpp
  PTOToEmitC/PTOToEmitCSyncComm.cpp
  PTOToEmitC/PTOToEmitCScalarMisc.cpp
  PTOToEmitC/PTOToEmitCTile.cpp
  PTOToEmitC/PTOToEmitCTensor.cpp
  PTOToEmitC/PTOToEmitCTensorReduce.cpp
  PTOToEmitC/PTOToEmitCPass.cpp
  Lowering/CppPostprocess.cpp
  Utils.cpp
  Passes/OptMemPlanForPipeline.cpp
  InferPTOMemScope.cpp
  PTOPlanMemoryModern.cpp
  Passes/PTORemoveRedundantBarrier.cpp
  InferPTOLayout.cpp
  Lowering/PTOA5NormalizeTMovPass.cpp
  Lowering/PTORemoveIdentityTMov.cpp
  Lowering/PTOCanonicalizeIR.cpp
  Lowering/BufferizableOpInterfaceImpl.cpp
  Lowering/ConvertToPTOOp.cpp
  Passes/PTOAssignDefaultFrontendPipeIdPass.cpp
  Lowering/PTOLowerFrontendPipeOpsPass.cpp
  Passes/PTOInferValidatePipeInitPass.cpp
  Passes/PTOResolveBufferSelect.cpp
  PTOResolveReservedBuffersPass.cpp
  Passes/SlotAffineAnalysis.cpp
  Passes/PTOWrapFunctionsInSectionsPass.cpp
  PTONormalizeUncoveredTileSections.cpp
  Passes/PTOValidatePhysicalSectionBoundaries.cpp
  PTOMaterializeTileOpSections.cpp
  VPTO/VPTONormalizeContainer.cpp
  VPTOSplitCVModule.cpp
  InsertSync/PTOIRTranslator.cpp
  InsertSync/SyncMacroModel.cpp
  InsertSync/SyncCommon.cpp
  InsertSync/InsertSyncAnalysis.cpp
  InsertSync/MemoryDependentAnalyzer.cpp
  InsertSync/MoveSyncState.cpp
  InsertSync/RemoveRedundantSync.cpp
  InsertSync/SyncEventIdAllocation.cpp
  InsertSync/SyncCodegen.cpp
  BufidSync/BufidSyncPass.cpp
  BufidSync/BufidSyncAnalysis.cpp
  BufidSync/BufidSyncIdAlloc.cpp
  BufidSync/BufidSyncCodegen.cpp
  Passes/LoweringSyncToPipe.cpp
  Passes/PTOVerifyTFreePass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/PTO

  DEPENDS
  PTOPassesIncGen
  PTOOpsIncGen

  LINK_COMPONENTS
  Analysis

  LINK_LIBS
  PUBLIC
  PTOAnalysis
  PTOIR
  MLIRAnalysis
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRAffineDialect
  MLIREmitCDialect
  MLIRTargetCpp
  MLIRTransformUtils
  MLIRTransforms
  MLIRTensorDialect
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRVectorDialect
  MLIRParser
  MLIRArithTransforms
  MLIRSCFToEmitC
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRConvertToLLVMPass
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
  PRIVATE
  ${_ptoas_transforms_private_link_libs}
)

install(TARGETS PTOTransforms
  EXPORT PTOASTargets      # 将此 Target 加入导出集
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

# 设置头文件包含路径，确保自动获得头文件路径
target_include_directories(PTOTransforms INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Sources grouped in subdirectories keep including headers that live directly in
# this directory by bare name (e.g. "Utils.h"), so keep it on the private
# include path.
target_include_directories(PTOTransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(TARGET obj.PTOTransforms)
  target_include_directories(obj.PTOTransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# MLIR conversion patterns legitimately hide base-class virtual
# matchAndRewrite overloads; upstream advice is to silence the warning.
# https://discourse.llvm.org/t/matchandrewrite-hiding-virtual-functions/84933/6
# (Replaces the per-file #pragma GCC diagnostic ignored "-Woverloaded-virtual"
# that PTOToEmitC.cpp carried before the split.)
if(TARGET obj.PTOTransforms)
  target_compile_options(obj.PTOTransforms PRIVATE -Wno-overloaded-virtual)
endif()
target_compile_options(PTOTransforms PRIVATE -Wno-overloaded-virtual)

if(TARGET vfsim::native_core AND TARGET obj.PTOTransforms)
  # Preserve the simulator target's compile usage requirements on the object
  # library; its link dependency is already captured in aggregation metadata.
  target_link_libraries(obj.PTOTransforms PRIVATE
    $<BUILD_INTERFACE:vfsim::native_core>)
endif()

if(TARGET vfsim::ir_planner)
  target_compile_definitions(PTOTransforms PRIVATE PTO_ENABLE_VFSIM_IR_PLANNER=1)
  if(TARGET obj.PTOTransforms)
    target_link_libraries(obj.PTOTransforms PRIVATE
      $<BUILD_INTERFACE:vfsim::ir_planner>)
    target_compile_definitions(obj.PTOTransforms PRIVATE PTO_ENABLE_VFSIM_IR_PLANNER=1)
  endif()
endif()
