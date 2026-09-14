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

# [关键] 库名重命名为 PTOIR，避免与 LLVM 里的 PTODialect/MLIRPTODialect 冲突
add_mlir_dialect_library(PTOIR
  ENABLE_AGGREGATION

  PTO.cpp
  VPTO.cpp
  VPTOAddressSemantics.cpp
  VPTOMemoryDist.cpp
  VPTOScheduling.cpp
  VMI/VMIHelpers.cpp
  VMI/VMI.cpp
  VMI/VMIMem.cpp
  VMI/VMICvt.cpp
  VMI/VMIGroup.cpp
  VMI/VMIArith.cpp
  VPTO/ubops/VPTOUBSetMask.cpp
  VPTO/ubops/VPTOUBSetMaskCount.cpp
  VPTO/ubops/VPTOUBSetMaskNorm.cpp
  VPTO/vecscope/VPTOVecScope.cpp
  VPTO/vecscope/VPTOStrictVecScope.cpp
  VPTO/ubv/VPTOUBVabs.cpp
  VPTO/ubv/VPTOUBVadd.cpp
  VPTO/ubv/VPTOUBVaddRelu.cpp
  VPTO/ubv/VPTOUBVaddS.cpp
  VPTO/ubv/VPTOUBVand.cpp
  VPTO/ubv/VPTOUBVdiv.cpp
  VPTO/ubv/VPTOUBVdup.cpp
  VPTO/ubv/VPTOUBVexp.cpp
  VPTO/ubv/VPTOUBVgather.cpp
  VPTO/ubv/VPTOUBVgatherb.cpp
  VPTO/ubv/VPTOUBVln.cpp
  VPTO/ubv/VPTOUBVmax.cpp
  VPTO/ubv/VPTOUBVmaxS.cpp
  VPTO/ubv/VPTOUBVmin.cpp
  VPTO/ubv/VPTOUBVminS.cpp
  VPTO/ubv/VPTOUBVmul.cpp
  VPTO/ubv/VPTOUBVmulS.cpp
  VPTO/ubv/VPTOUBVnot.cpp
  VPTO/ubv/VPTOUBVor.cpp
  VPTO/ubv/VPTOUBVrelu.cpp
  VPTO/ubv/VPTOUBVrsqrt.cpp
  VPTO/ubv/VPTOUBVshl.cpp
  VPTO/ubv/VPTOUBVshr.cpp
  VPTO/ubv/VPTOUBVsqrt.cpp
  VPTO/ubv/VPTOUBVsub.cpp
  VPTO/align/VPTOAlign.cpp
  VPTO/align/VPTOInitAlign.cpp
  VPTO/align/VPTOPstu.cpp
  VPTO/align/VPTOVstus.cpp
  VPTO/align/VPTOVstur.cpp
  VPTO/align/VPTOVstas.cpp
  VPTO/align/VPTOVstar.cpp
  VPTO/align/VPTOVldas.cpp
  VPTO/align/VPTOVldus.cpp
  VPTO/vcvt/VPTOVcvt.cpp
  VPTO/vcvt/VPTOVcvtShared.cpp
  VPTO/mte/VPTOMteGmL1.cpp
  VPTO/mte/VPTOMteGmL1Frac.cpp
  VPTO/mte/VPTOMteGmUb.cpp
  VPTO/mte/VPTOMteL0cGm.cpp
  VPTO/mte/VPTOMteL0cL1.cpp
  VPTO/mte/VPTOMteL0cUb.cpp
  VPTO/mte/VPTOMteL1Bt.cpp
  VPTO/mte/VPTOMteL1Fb.cpp
  VPTO/mte/VPTOMteL1L0a.cpp
  VPTO/mte/VPTOMteL1L0aMx.cpp
  VPTO/mte/VPTOMteL1L0b.cpp
  VPTO/mte/VPTOMteL1L0bMx.cpp
  VPTO/mte/VPTOMteL1Ub.cpp
  VPTO/mte/VPTOMteUbGm.cpp
  VPTO/mte/VPTOMteUbL1.cpp
  VPTO/mte/VPTOMteUbUb.cpp
  VPTO/mteasm/VPTOMteGmL1FracAsm.cpp
  VPTO/mteasm/VPTOMteGmUbAsm.cpp
  VPTO/mteasm/VPTOMteL0cGmAsm.cpp
  VPTO/mteasm/VPTOMteL0cL1Asm.cpp
  VPTO/mteasm/VPTOMteL0cUbAsm.cpp
  VPTO/mteasm/VPTOMteL1L0Asm.cpp
  VPTO/mteasm/VPTOMteUbGmAsm.cpp
  VPTO/cbridge/VPTOCubeBridge.cpp
  VPTO/cbridge/VPTOLoadCbufToCa.cpp
  VPTO/cbridge/VPTOLoadCbufToCb.cpp
  VPTO/cbridge/VPTOLoadCbufToCaS4.cpp
  VPTO/cbridge/VPTOLoadCbufToCbS4.cpp
  VPTO/cbridge/VPTOLoadCbufToCaMx.cpp
  VPTO/cbridge/VPTOLoadCbufToCbMx.cpp
  VPTO/structuredacc/VPTOStructuredAcc.cpp
  VPTO/structuredacc/VPTOStructuredAccParse.cpp
  VPTO/structuredacc/VPTOStructuredAccPrint.cpp
  VPTO/structuredacc/VPTOStructuredAccVerify.cpp
  VPTO/dma/VPTODma.cpp
  VPTO/dma/VPTOSetMovPadVal.cpp
  VPTO/dma/VPTOCopyGmToUbuf.cpp
  VPTO/dma/VPTOCopyUbufToUbuf.cpp
  VPTO/dma/VPTOCopyCbufToUbuf.cpp
  VPTO/dma/VPTOCopyUbufToCbuf.cpp
  VPTO/dma/VPTOCopyUbufToGm.cpp
  VPTO/dma/VPTOVlds.cpp
  VPTO/dma/VPTOVldsx2.cpp
  VPTO/dma/VPTOVsts.cpp
  VPTO/dma/VPTOVstsx2.cpp
  VPTO/mad/VPTOMad.cpp
  VPTO/mad/VPTOMadOp.cpp
  VPTO/mad/VPTOMadAccOp.cpp
  VPTO/mad/VPTOMadBiasOp.cpp
  VPTO/mad/VPTOMadMxOp.cpp
  VPTO/mad/VPTOMadMxAccOp.cpp
  VPTO/mad/VPTOMadMxBiasOp.cpp
  VPTO/mad/VPTOMadRawOp.cpp
  VPTO/mad/VPTOMadBiasRawOp.cpp
  VPTO/mad/VPTOMadMxRawOp.cpp
  VPTO/mad/VPTOMadMxBiasRawOp.cpp
  VPTO/vecop/pred/VPTOPand.cpp
  VPTO/vecop/pred/VPTOPbitcast.cpp
  VPTO/vecop/pred/VPTOPdintlvB16.cpp
  VPTO/vecop/pred/VPTOPdintlvB32.cpp
  VPTO/vecop/pred/VPTOPdintlvB8.cpp
  VPTO/vecop/pred/VPTOPgeB16.cpp
  VPTO/vecop/pred/VPTOPgeB32.cpp
  VPTO/vecop/pred/VPTOPgeB8.cpp
  VPTO/vecop/pred/VPTOPintlvB16.cpp
  VPTO/vecop/pred/VPTOPintlvB32.cpp
  VPTO/vecop/pred/VPTOPintlvB8.cpp
  VPTO/vecop/pred/VPTOPltB16.cpp
  VPTO/vecop/pred/VPTOPltB32.cpp
  VPTO/vecop/pred/VPTOPltB8.cpp
  VPTO/vecop/pred/VPTOPltmB16.cpp
  VPTO/vecop/pred/VPTOPltmB32.cpp
  VPTO/vecop/pred/VPTOPltmB8.cpp
  VPTO/vecop/pred/VPTOPnot.cpp
  VPTO/vecop/pred/VPTOPor.cpp
  VPTO/vecop/pred/VPTOPpack.cpp
  VPTO/vecop/pred/VPTOPsel.cpp
  VPTO/vecop/pred/VPTOPsetB16.cpp
  VPTO/vecop/pred/VPTOPsetB32.cpp
  VPTO/vecop/pred/VPTOPsetB8.cpp
  VPTO/vecop/pred/VPTOPunpack.cpp
  VPTO/vecop/pred/VPTOPxor.cpp
  VPTO/vecop/arith/VPTOVabs.cpp
  VPTO/vecop/arith/VPTOVadd.cpp
  VPTO/vecop/arith/VPTOVaddc.cpp
  VPTO/vecop/arith/VPTOVaddcs.cpp
  VPTO/vecop/arith/VPTOVaddreluconv.cpp
  VPTO/vecop/arith/VPTOVadds.cpp
  VPTO/vecop/arith/VPTOVaxpy.cpp
  VPTO/vecop/arith/VPTOVcadd.cpp
  VPTO/vecop/arith/VPTOVcbmax.cpp
  VPTO/vecop/arith/VPTOVcbmin.cpp
  VPTO/vecop/arith/VPTOVcgadd.cpp
  VPTO/vecop/arith/VPTOVcgmax.cpp
  VPTO/vecop/arith/VPTOVcgmin.cpp
  VPTO/vecop/arith/VPTOVcmax.cpp
  VPTO/vecop/arith/VPTOVcmin.cpp
  VPTO/vecop/arith/VPTOVcmp.cpp
  VPTO/vecop/arith/VPTOVcmps.cpp
  VPTO/vecop/arith/VPTOVcpadd.cpp
  VPTO/vecop/arith/VPTOVdiv.cpp
  VPTO/vecop/arith/VPTOVexp.cpp
  VPTO/vecop/arith/VPTOVexpdif.cpp
  VPTO/vecop/arith/VPTOVln.cpp
  VPTO/vecop/arith/VPTOVlrelu.cpp
  VPTO/vecop/arith/VPTOVmadd.cpp
  VPTO/vecop/arith/VPTOVmax.cpp
  VPTO/vecop/arith/VPTOVmaxs.cpp
  VPTO/vecop/arith/VPTOVmin.cpp
  VPTO/vecop/arith/VPTOVmins.cpp
  VPTO/vecop/arith/VPTOVmul.cpp
  VPTO/vecop/arith/VPTOVmula.cpp
  VPTO/vecop/arith/VPTOVmulconv.cpp
  VPTO/vecop/arith/VPTOVmull.cpp
  VPTO/vecop/arith/VPTOVmuls.cpp
  VPTO/vecop/arith/VPTOVmulscvt.cpp
  VPTO/vecop/arith/VPTOVneg.cpp
  VPTO/vecop/arith/VPTOVprelu.cpp
  VPTO/vecop/arith/VPTOVrelu.cpp
  VPTO/vecop/arith/VPTOVsqrt.cpp
  VPTO/vecop/arith/VPTOVsub.cpp
  VPTO/vecop/arith/VPTOVsubc.cpp
  VPTO/vecop/arith/VPTOVsubcs.cpp
  VPTO/vecop/bitwise/VPTOVand.cpp
  VPTO/vecop/bitwise/VPTOVbitcast.cpp
  VPTO/vecop/bitwise/VPTOVbitsort.cpp
  VPTO/vecop/bitwise/VPTOVnot.cpp
  VPTO/vecop/bitwise/VPTOVor.cpp
  VPTO/vecop/bitwise/VPTOVshl.cpp
  VPTO/vecop/bitwise/VPTOVshls.cpp
  VPTO/vecop/bitwise/VPTOVshr.cpp
  VPTO/vecop/bitwise/VPTOVshrs.cpp
  VPTO/vecop/bitwise/VPTOVxor.cpp
  VPTO/vecop/datamov/VPTOVbr.cpp
  VPTO/vecop/datamov/VPTOVci.cpp
  VPTO/vecop/datamov/VPTOVdintlv.cpp
  VPTO/vecop/datamov/VPTOVdintlvv2.cpp
  VPTO/vecop/datamov/VPTOVdup.cpp
  VPTO/vecop/datamov/VPTOVintlv.cpp
  VPTO/vecop/datamov/VPTOVintlvv2.cpp
  VPTO/vecop/datamov/VPTOVpack.cpp
  VPTO/vecop/datamov/VPTOVsel.cpp
  VPTO/vecop/datamov/VPTOVselr.cpp
  VPTO/vecop/datamov/VPTOVselrv2.cpp
  VPTO/vecop/datamov/VPTOVsldb.cpp
  VPTO/vecop/datamov/VPTOVsqz.cpp
  VPTO/vecop/datamov/VPTOVstar.cpp
  VPTO/vecop/datamov/VPTOVsunpack.cpp
  VPTO/vecop/datamov/VPTOVtrc.cpp
  VPTO/vecop/datamov/VPTOVusqz.cpp
  VPTO/vecop/datamov/VPTOVzunpack.cpp
  VPTO/vecop/mulred/VPTOMulI32ToI64.cpp
  VPTO/vecop/mulred/VPTOMulhi.cpp
  VPTO/vecop/mulred/VPTOReduxAdd.cpp
  VPTO/vecop/mulred/VPTOReduxMax.cpp
  VPTO/vecop/mulred/VPTOReduxMin.cpp
  VPTO/vecop/mulred/VPTOShuffleBfly.cpp
  VPTO/vecop/mulred/VPTOShuffleDown.cpp
  VPTO/vecop/mulred/VPTOShuffleIdx.cpp
  VPTO/vecop/mulred/VPTOShuffleUp.cpp
  VPTO/vecop/VPTOVecOp.cpp
  VPTO/memop/VPTOMemOp.cpp
  VPTO/memop/VPTOAtomicAdd.cpp
  VPTO/memop/VPTOAtomicAnd.cpp
  VPTO/memop/VPTOAtomicCas.cpp
  VPTO/memop/VPTOAtomicExch.cpp
  VPTO/memop/VPTOAtomicMax.cpp
  VPTO/memop/VPTOAtomicMin.cpp
  VPTO/memop/VPTOAtomicOr.cpp
  VPTO/memop/VPTOAtomicSub.cpp
  VPTO/memop/VPTOAtomicXor.cpp
  VPTO/memop/VPTOChistv2.cpp
  VPTO/memop/VPTOCreateCbufMatrix.cpp
  VPTO/memop/VPTODhistv2.cpp
  VPTO/memop/VPTOPTOLdDev.cpp
  VPTO/memop/VPTOPTOLdg.cpp
  VPTO/memop/VPTOPTOLoad.cpp
  VPTO/memop/VPTOPTOStDev.cpp
  VPTO/memop/VPTOPTOStg.cpp
  VPTO/memop/VPTOPTOStore.cpp
  VPTO/memop/VPTOPldi.cpp
  VPTO/memop/VPTOPlds.cpp
  VPTO/memop/VPTOPsti.cpp
  VPTO/memop/VPTOPsts.cpp
  VPTO/memop/VPTOPstu.cpp
  VPTO/memop/VPTORawFillL1.cpp
  VPTO/memop/VPTOSimtLaunch.cpp
  VPTO/memop/VPTOSprclr.cpp
  VPTO/memop/VPTOSprsti.cpp
  VPTO/memop/VPTOSprsts.cpp
  VPTO/memop/VPTOTensorViewAddr.cpp
  VPTO/memop/VPTOTileBufAddr.cpp
  VPTO/memop/VPTOUvld.cpp
  VPTO/memop/VPTOVgather2.cpp
  VPTO/memop/VPTOVgather2Bc.cpp
  VPTO/memop/VPTOVgatherb.cpp
  VPTO/memop/VPTOVldas.cpp
  VPTO/memop/VPTOVldus.cpp
  VPTO/memop/VPTOVscatter.cpp
  VPTO/memop/VPTOVsstb.cpp
  VPTO/memop/VPTOVstas.cpp
  VPTO/memop/VPTOVstur.cpp
  VPTO/memop/VPTOVstus.cpp
  PTOAttrs.cpp
  PTOLayoutUtils.cpp
  PTOSyncUtils.cpp
  PTOTypeDefs.cpp
  PTOTypeUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/PTO

  DEPENDS
  PTOOpsIncGen  # 对应 include/PTO/IR/CMakeLists.txt 里定义的那个目标

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)

install(TARGETS PTOIR
  EXPORT PTOASTargets      # 将此 Target 加入导出集
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

# 设置头文件包含路径，确保自动获得头文件路径
target_include_directories(PTOIR INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# VMI/ VPTO/ 子目录源文件仍按文件名裸引用本目录内部头 (VMIInternal.h /
# VPTOInternal.h), 将本目录加入 PRIVATE 搜索路径保持自包含解析
target_include_directories(PTOIR PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
