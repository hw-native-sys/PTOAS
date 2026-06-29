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

add_mlir_dialect_library(PTOTransforms
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
  TileFusion/PTOFlattenFusionRegion.cpp
  VPTOLLVMEmitter.cpp
  VPTOCANN900LLVMEmitter.cpp
  VPTOLLVMEmitterDispatcher.cpp
  VPTOLLVMEmitterHelper.cpp
  VPTOPtrNormalize.cpp
  VPTOPtrCastCleanup.cpp
  VPTOExpandWrapperOps.cpp
  PTOVPTOPtrBoundary.cpp
  VPTOBufferMaterialization.cpp
  PTOValidateVPTOIR.cpp
  PTOUnrollSIMTForPass.cpp
  PTOInferVPTOVecScope.cpp

  InsertSync/PTOInsertSync.cpp
  PTOInjectBarrierAllSync.cpp
  InsertSync/InsertSyncDebug.cpp
  PTOInsertVMemBar.cpp
  PTOViewToMemref.cpp
  PTOValidateIntToPtrUses.cpp
  ExpandTileOp.cpp
  FoldTileBufIntrinsics.cpp
  PTOLowerToOpLibCalls.cpp
  PTOInstantiateAndInlineOpLib.cpp
  PTOToEmitC.cpp
  CppPostprocess.cpp
  Utils.cpp
  OptMemPlanForPipeline.cpp
  AllocToPointerCast.cpp
  InferPTOMemScope.cpp
  PTOPlanMemory.cpp
  PTORemoveRedundantBarrier.cpp
  InferPTOLayout.cpp
  PTOA5NormalizeTMovPass.cpp
  PTOCanonicalizeIR.cpp
  PTOMaterializeTileHandles.cpp
  BufferizableOpInterfaceImpl.cpp
  ConvertToPTOOp.cpp
  PTOAssignDefaultFrontendPipeIdPass.cpp
  PTOLowerFrontendPipeOpsPass.cpp
  PTOInferValidatePipeInitPass.cpp
  PTOResolveReservedBuffersPass.cpp
  PTOWrapFunctionsInSectionsPass.cpp
  PTONormalizeUncoveredTileSections.cpp
  VPTONormalizeContainer.cpp
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
  GraphSyncSolver/PTOGraphSyncSolver.cpp
  GraphSyncSolver/SyncSolverIR.cpp
  GraphSyncSolver/SyncSolverIRTranslator.cpp
  GraphSyncSolver/Utility.cpp
  GraphSyncSolver/MemInfo.cpp
  GraphSyncSolver/GraphSolver.cpp
  GraphSyncSolver/EventIdSolver.cpp
  GraphSyncSolver/SyncSolver.cpp
  GraphSyncSolver/SyncSolverCodeGen.cpp
  LoweringSyncToPipe.cpp
  PTOVerifyTFreePass.cpp
  PTOVerifySubkernelPipeContractPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/PTO

  DEPENDS
  PTOPassesIncGen
  PTOOpsIncGen

  LINK_COMPONENTS
  Analysis

  LINK_LIBS PUBLIC
  PTOIR
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
  MLIRVectorDialect
  MLIRParser
  MLIRArithTransforms
  MLIRSCFToEmitC
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRConvertToLLVMPass
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
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
