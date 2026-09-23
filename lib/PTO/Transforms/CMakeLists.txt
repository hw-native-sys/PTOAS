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
  VMIToVPTO/VMIToVPTOConversionInternals.cpp
  VMIToVPTO/VMIToVPTOMemoryInternals.cpp
  VMIToVPTO/VMIToVPTOMaskInternals.cpp
  VMIToVPTO/VMIToVPTODataLayoutInternals.cpp
  VMIToVPTO/VMIToVPTOUnifiedMaskInternals.cpp
  VMIToVPTO/VMIToVPTOPatternInternals0.cpp
  VMIToVPTO/VMIToVPTOPatternInternals1.cpp
  VMIToVPTO/VMIToVPTOPatternInternals2.cpp
  VMIToVPTO/VMIToVPTOPatternInternals3.cpp
  VMIToVPTO/VMIToVPTOPatternInternals4.cpp
  VMIToVPTO/VMIToVPTOPatternInternals5.cpp
  VMIToVPTO/VMIToVPTOPatternInternals10.cpp
  VMIToVPTO/VMIToVPTOPatternInternals6.cpp
  VMIToVPTO/VMIToVPTOUnifiedPatternInternals.cpp
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
  VPTO/VPTOCANN900LLVMEmitterArithmeticPatterns.cpp
  VPTO/VPTOCANN900LLVMEmitterCalleeCore.cpp
  VPTO/VPTOCANN900LLVMEmitterCalleeMemory.cpp
  VPTO/VPTOCANN900LLVMEmitterMemoryPatterns.cpp
  VPTO/VPTOCANN900LLVMEmitterPacking.cpp
  VPTO/VPTOCANN900LLVMEmitterPipeline.cpp
  VPTO/VPTOCANN900LLVMEmitterScalarPatterns.cpp
  VPTO/VPTOCANN900LLVMEmitterTypeHelpers.cpp
  VPTO/VPTOCANN900LLVMEmitterTypePatterns.cpp
  VPTO/VPTOLLVMEmitterDispatcher.cpp
  VPTO/VPTOLLVMEmitterHelper.cpp
  VPTO/VPTOPtrNormalize.cpp
  VPTO/VPTOPtrCastCleanup.cpp
  VPTO/VPTOCombineReductions.cpp
  VPTO/VPTOOptimizeVcvt.cpp
  VPTO/VPTOMaskSimplify.cpp
  VPTOExpandMadOps.cpp
  VPTO/VPTOExpandWrapperOps.cpp
  VPTOOptimizeCtrlState.cpp
  VPTO/VPTOSoftPostUpdate.cpp
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
  VPTOScheduler/VPTOSchedulerRematCandidates.cpp
  VPTOScheduler/VPTOSchedulerRematPlanner.cpp
  VPTOScheduler/VPTOSchedulerRematPressure.cpp
  VPTOScheduler/VPTOSchedulerRematerialization.cpp
  VPTOScheduler/VPTOSchedulerPass.cpp
  Passes/PTOVPTOPtrBoundary.cpp
  VPTO/VPTOBufferMaterialization.cpp
  Passes/PTOValidateVPTOIR.cpp
  Passes/PTONarrowVPTOLoopCounters.cpp
  Passes/PTOConvertSCFToCFWithLoopHintsPass.cpp
  Passes/PTOPromotePersistentFragmentLoopsPass.cpp
  Passes/PTOUnrollLoopsPass.cpp
  Passes/PTOValidateVMIIR.cpp
  VMI/VMIExpandImplicitEnsureLayouts.cpp
  VMI/VMIPreAssignmentCombine.cpp
  VMI/VMIControlFlowSupport.cpp
  VMI/VMILegalizeArithSelect.cpp
  VMI/VMIMaskGranularityAssignment.cpp
  VMI/VMILowerUnifiedToLegacy.cpp
  VMI/VMINormalizeSignlessIntToUnsigned.cpp
  VMI/VMILayoutRematerializeWeakProducers.cpp
  VMI/VMILayoutAssignment.cpp
  VMI/VMILayoutSpineAnalysis.cpp
  VMI/VMILayoutFold.cpp
  VMI/VMILayoutPropagation.cpp
  VMI/VMILayoutSupport.cpp
  VMI/VMILayoutSupportGroupSlots.cpp
  VMI/VMILayoutRematerialize.cpp
  VMI/VMILayoutSinkMaterialization.cpp
  VMIToVPTO/VMIToVPTO.cpp
  ${_ptoas_vmi_split_units}
  VPTO/VPTOStatefulStreamFusion.cpp
  Lowering/PTOExpandSoftLib.cpp
  Passes/SIMTPersistentFragmentAnalysis.cpp
  Passes/PTOAnalyzeSIMTPersistentFragment.cpp
  Passes/PTOMaterializeSIMTPersistentFragment.cpp
  Passes/PTOOutlineSIMTSections.cpp
  VPTO/PTOInferVPTOVecScope.cpp

  InsertSync/PTOInsertSync.cpp
  Passes/PTOInjectBarrierAllSync.cpp
  InsertSync/InsertSyncDebug.cpp
  Passes/PTOMaterializeImplicitTmp.cpp
  Lowering/PTORematerializeFixpipeVectorQuant.cpp
  TileOp/TileLibService.cpp
  Support/SoftLibService.cpp
  TileOp/InsertTemplateAttributes.cpp
  TileOp/ExpandTileOp.cpp
  TileOp/FoldTileBufIntrinsics.cpp
  Lowering/LowerPTOToUBufOps.cpp
  TileOp/PTOLowerToOpLibCalls.cpp
  TileOp/PTOInstantiateAndInlineOpLib.cpp
  PTOToEmitC/PTOToEmitCCommon.cpp
  PTOToEmitC/PTOToEmitCTypeConverter.cpp
  PTOToEmitC/PTOToEmitCPass.cpp
  PTOToEmitC/Arith/ArithBinary.cpp
  PTOToEmitC/Arith/ArithCast.cpp
  PTOToEmitC/Arith/ArithCmp.cpp
  PTOToEmitC/Arith/ArithFloatMinMax.cpp
  PTOToEmitC/Arith/ArithIntBinary.cpp
  PTOToEmitC/Arith/ArithIntDiv.cpp
  PTOToEmitC/Arith/ArithIntMinMax.cpp
  PTOToEmitC/Arith/ArithMisc.cpp
  PTOToEmitC/Arith/ArithShift.cpp
  PTOToEmitC/Arith/ArithSupport.cpp
  PTOToEmitC/Arith/ArithPatterns.cpp
  PTOToEmitC/Tensor/TActivation.cpp
  PTOToEmitC/Tensor/TAdd.cpp
  PTOToEmitC/Tensor/TAnd.cpp
  PTOToEmitC/Tensor/TCI.cpp
  PTOToEmitC/Tensor/TCmp.cpp
  PTOToEmitC/Tensor/TColExpand.cpp
  PTOToEmitC/Tensor/TColReduce.cpp
  PTOToEmitC/Tensor/TConcat.cpp
  PTOToEmitC/Tensor/TCvt.cpp
  PTOToEmitC/Tensor/TDiv.cpp
  PTOToEmitC/Tensor/TElemwise.cpp
  PTOToEmitC/Tensor/TExp.cpp
  PTOToEmitC/Tensor/TExtractInsert.cpp
  PTOToEmitC/Tensor/TFillPad.cpp
  PTOToEmitC/Tensor/TGather.cpp
  PTOToEmitC/Tensor/TMinMax.cpp
  PTOToEmitC/Tensor/TMov.cpp
  PTOToEmitC/Tensor/TMul.cpp
  PTOToEmitC/Tensor/TQuant.cpp
  PTOToEmitC/Tensor/TRandom.cpp
  PTOToEmitC/Tensor/TSort.cpp
  PTOToEmitC/Tensor/TSub.cpp
  PTOToEmitC/Tensor/TTrans.cpp
  PTOToEmitC/Tensor/TTri.cpp
  PTOToEmitC/Tensor/TensorSupport.cpp
  PTOToEmitC/Tensor/TensorPatterns.cpp
  PTOToEmitC/Reduce/ArithCmpI.cpp
  PTOToEmitC/Reduce/ReduceMisc.cpp
  PTOToEmitC/Reduce/TInterleave.cpp
  PTOToEmitC/Reduce/TMath.cpp
  PTOToEmitC/Reduce/TPart.cpp
  PTOToEmitC/Reduce/TPrintTrap.cpp
  PTOToEmitC/Reduce/TRowExpand.cpp
  PTOToEmitC/Reduce/TRowReduce.cpp
  PTOToEmitC/Reduce/TScatter.cpp
  PTOToEmitC/Reduce/TSel.cpp
  PTOToEmitC/Reduce/TShiftScalar.cpp
  PTOToEmitC/Reduce/ReduceSupport.cpp
  PTOToEmitC/Reduce/ReducePatterns.cpp
  PTOToEmitC/ScalarMisc/AsyncSession.cpp
  PTOToEmitC/ScalarMisc/Comm.cpp
  PTOToEmitC/ScalarMisc/GlobalEventArray.cpp
  PTOToEmitC/ScalarMisc/LocalArray.cpp
  PTOToEmitC/ScalarMisc/RuntimeQuery.cpp
  PTOToEmitC/ScalarMisc/ScalarMiscMisc.cpp
  PTOToEmitC/ScalarMisc/ScalarMiscOps.cpp
  PTOToEmitC/ScalarMisc/ScalarPtr.cpp
  PTOToEmitC/ScalarMisc/Struct.cpp
  PTOToEmitC/ScalarMisc/ScalarMiscSupport.cpp
  PTOToEmitC/ScalarMisc/ScalarMiscPatterns.cpp
  PTOToEmitC/Tile/PartitionView.cpp
  PTOToEmitC/Tile/Section.cpp
  PTOToEmitC/Tile/TensorView.cpp
  PTOToEmitC/Tile/TileAlloc.cpp
  PTOToEmitC/Tile/TileMisc.cpp
  PTOToEmitC/Tile/TileSupport.cpp
  PTOToEmitC/Tile/TilePatterns.cpp
  PTOToEmitC/SyncComm/BlockIdx.cpp
  PTOToEmitC/SyncComm/BufPool.cpp
  PTOToEmitC/SyncComm/FuncCall.cpp
  PTOToEmitC/SyncComm/Sync.cpp
  PTOToEmitC/SyncComm/SyncCommSupport.cpp
  PTOToEmitC/SyncComm/SyncCommPatterns.cpp
  PTOToEmitC/LoadStore/TLoad.cpp
  PTOToEmitC/LoadStore/TMatmul.cpp
  PTOToEmitC/LoadStore/TStore.cpp
  PTOToEmitC/LoadStore/LoadStoreSupport.cpp
  PTOToEmitC/LoadStore/LoadStorePatterns.cpp
  PTOToEmitC/Memref/ReinterpretCast.cpp
  PTOToEmitC/Memref/Subview.cpp
  PTOToEmitC/Memref/MemrefSupport.cpp
  PTOToEmitC/Memref/MemrefPatterns.cpp
  Lowering/CppPostprocess.cpp
  Support/Utils.cpp
  Passes/OptMemPlanForPipeline.cpp
  Passes/InferPTOMemScope.cpp
  Passes/PTOPlanMemoryModern.cpp
  Passes/PTORemoveRedundantBarrier.cpp
  Passes/InferPTOLayout.cpp
  Lowering/PTOA5NormalizeTMovPass.cpp
  Lowering/PTORemoveIdentityTMov.cpp
  Lowering/PTOArithRangeOptimize.cpp
  Lowering/PTOCanonicalizeIR.cpp
  Lowering/BufferizableOpInterfaceImpl.cpp
  Lowering/PTOLowerGenericOps.cpp
  Passes/PTOAssignDefaultFrontendPipeIdPass.cpp
  Lowering/PTOLowerFrontendPipeOpsPass.cpp
  Passes/PTOInferValidatePipeInitPass.cpp
  Passes/PTOResolveBufferSelect.cpp
  Passes/PTOResolveReservedBuffersPass.cpp
  Passes/SlotAffineAnalysis.cpp
  Passes/PTOWrapFunctionsInSectionsPass.cpp
  Passes/PTONormalizeUncoveredTileSections.cpp
  Passes/PTOValidatePhysicalSectionBoundaries.cpp
  Passes/PTOMaterializeTileOpSections.cpp
  VPTO/VPTONormalizeContainer.cpp
  VPTO/VPTOSplitCVModule.cpp
  InsertSync/PTOIRTranslator.cpp
  InsertSync/SyncMacroModel.cpp
  InsertSync/SyncCommon.cpp
  InsertSync/InsertSyncAnalysis.cpp
  InsertSync/MemoryDependentAnalyzer.cpp
  InsertSync/MoveSyncState.cpp
  InsertSync/RemoveRedundantSync.cpp
  InsertSync/RemoveSlotRedundantSync.cpp
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
  MLIRLLVMDialect
  MLIRPass
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRMathDialect
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
  MLIRMathToLLVM
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
target_include_directories(PTOTransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Support)
# PTOValidateVPTOIR.cpp shares the SIMT keep/resume verify helpers with the
# PTO IR implementation (lib/PTO/IR/PTOPipeline); expose that internal dir.
target_include_directories(PTOTransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../IR)
if(TARGET obj.PTOTransforms)
  target_include_directories(obj.PTOTransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_include_directories(obj.PTOTransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Support)
  target_include_directories(obj.PTOTransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../IR)
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
