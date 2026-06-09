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
  InsertSync/PTOInsertSync.cpp
  PTOInjectBarrierAllSync.cpp
  InsertSync/InsertSyncDebug.cpp
  PTOViewToMemref.cpp
  PTOViewToMemrefCompute.cpp
  PTOValidateIntToPtrUses.cpp
  PTOToEmitC.cpp
  PTOToEmitCSubview.cpp
  PTOToEmitCArith.cpp
  PTOToEmitCTilePatterns.cpp
  PTOToEmitCTilePatternsExtra.cpp
  PTOToEmitCTileMaterialization.cpp
  PTOToEmitCSync.cpp
  PTOToEmitCComm.cpp
  PTOToEmitCKernelOps.cpp
  PTOToEmitCControlFlow.cpp
  PTOToEmitCSimpleOps.cpp
  PTOToEmitCRuntimeOps.cpp
  PTOToEmitCMemoryOps.cpp
  Utils.cpp
  OptMemPlanForPipeline.cpp
  AllocToPointerCast.cpp
  InferPTOMemScope.cpp
  PTOPlanMemory.cpp
  PTOPlanMemoryReuse.cpp
  PTORemoveRedundantBarrier.cpp
  InferPTOLayout.cpp
  PTOA5NormalizeTMovPass.cpp
  PTOMaterializeTileHandles.cpp
  BufferizableOpInterfaceImpl.cpp
  ConvertToPTOOp.cpp
  PTOAssignDefaultFrontendPipeIdPass.cpp
  PTOLowerFrontendPipeOpsPass.cpp
  PTOInferValidatePipeInitPass.cpp
  PTOResolveReservedBuffersPass.cpp
  PTOWrapFunctionsInSectionsPass.cpp
  InsertSync/PTOIRTranslator.cpp
  InsertSync/SyncCommon.cpp
  InsertSync/InsertSyncAnalysis.cpp
  InsertSync/MemoryDependentAnalyzer.cpp
  InsertSync/MoveSyncState.cpp
  InsertSync/RemoveRedundantSync.cpp
  InsertSync/SyncEventIdAllocation.cpp
  InsertSync/SyncCodegen.cpp
  GraphSyncSolver/PTOGraphSyncSolver.cpp
  GraphSyncSolver/SyncSolverIR.cpp
  GraphSyncSolver/SyncSolverIRTranslator.cpp
  GraphSyncSolver/Utility.cpp
  GraphSyncSolver/MemInfo.cpp
  GraphSyncSolver/GraphSolver.cpp
  GraphSyncSolver/EventIdSolver.cpp
  GraphSyncSolver/SyncSolver.cpp
  GraphSyncSolver/SyncSolverMerge.cpp
  GraphSyncSolver/SyncSolverCodeGen.cpp
  LoweringSyncToPipe.cpp
  PTOVerifyTFreePass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/PTO

  DEPENDS
  PTOPassesIncGen
  PTOOpsIncGen

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
  MLIRSCFToEmitC
  MLIRSCFDialect
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
