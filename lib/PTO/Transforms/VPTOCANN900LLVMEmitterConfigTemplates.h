// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Config, loop, synchronization, and runtime-query callee-name templates
// for the CANN900 LLVM emitter. Split from VPTOCANN900LLVMEmitterTemplates.h
// to keep each header under the 500-line header-size limit.

#pragma once

#include "VPTOCANN900LLVMEmitterInternal.h"

namespace mlir::pto::detail {

template <typename LoopOp> StringRef buildSetLoopCallee(MLIRContext *context);

template <typename ConfigOp> StringRef buildUnaryConfigCallee(MLIRContext *context);

template <typename ConfigOp> StringRef buildNullaryConfigCallee(MLIRContext *context);

template <> inline StringRef buildSetLoopCallee<pto::SetLoop2StrideOutToUbOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP2.STRIDE.OUTTOUB").getValue();
}

template <> inline StringRef buildSetLoopCallee<pto::SetLoop1StrideOutToUbOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP1.STRIDE.OUTTOUB").getValue();
}

template <> inline StringRef buildSetLoopCallee<pto::SetLoopSizeOutToUbOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP.SIZE.OUTTOUB").getValue();
}

template <> inline StringRef buildSetLoopCallee<pto::SetLoop2StrideUbToOutOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP2.STRIDE.UBTOOUT").getValue();
}

template <> inline StringRef buildSetLoopCallee<pto::SetLoop1StrideUbToOutOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP1.STRIDE.UBTOOUT").getValue();
}

template <> inline StringRef buildSetLoopCallee<pto::SetLoopSizeUbToOutOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP.SIZE.UBTOOUT").getValue();
}

template <> inline StringRef buildSetLoopCallee<pto::SetLoop3ParaOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP3.PARA").getValue();
}

template <> inline StringRef buildSetLoopCallee<pto::SetChannelParaOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.CHANNEL.PARA").getValue();
}

template <> inline StringRef buildUnaryConfigCallee<pto::SetMovPadValOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.MOV.PAD.VAL").getValue();
}

template <> inline StringRef buildUnaryConfigCallee<pto::SetQuantPreOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.QUANT.PRE.v300").getValue();
}

template <> inline StringRef buildUnaryConfigCallee<pto::SetReluAlphaOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.RELU.ALPHA").getValue();
}

template <> inline StringRef buildUnaryConfigCallee<pto::SetFixClipReluOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.FIX.CLIP.RELU").getValue();
}

template <> inline StringRef buildUnaryConfigCallee<pto::SetLoop2StrideOutToL1Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP2.STRIDE.OUTTOL1").getValue();
}

template <> inline StringRef buildUnaryConfigCallee<pto::SetLoop1StrideOutToL1Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP1.STRIDE.OUTTOL1").getValue();
}

template <> inline StringRef buildUnaryConfigCallee<pto::SetLoopSizeOutToL1Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.LOOP.SIZE.OUTTOL1").getValue();
}

template <> inline StringRef buildUnaryConfigCallee<pto::SetMte2NzParaOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.MTE2.NZ.PARA").getValue();
}

template <> inline StringRef buildUnaryConfigCallee<pto::SetPadValOutToL1Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.PAD.VAL.OUTTOL1").getValue();
}

template <> inline StringRef buildUnaryConfigCallee<pto::SetFpcOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.FPC").getValue();
}

template <> inline StringRef buildUnaryConfigCallee<pto::SetStoreAtomicCfgOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.ST.ATOMIC.CFG").getValue();
}

template <> inline StringRef buildNullaryConfigCallee<pto::SetAtomicS32Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.ATOMIC.S32").getValue();
}

template <> inline StringRef buildNullaryConfigCallee<pto::SetAtomicS8Op>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.ATOMIC.S8").getValue();
}

template <typename SyncOp> StringRef buildSyncCallee(MLIRContext *context);

template <> inline StringRef buildSyncCallee<pto::SetFlagOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.FLAG.IMM").getValue();
}

template <> inline StringRef buildSyncCallee<pto::WaitFlagOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.WAIT.FLAG.IMM").getValue();
}

template <> inline StringRef buildSyncCallee<pto::SetFlagDynOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.FLAG.REG").getValue();
}

template <> inline StringRef buildSyncCallee<pto::WaitFlagDynOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.WAIT.FLAG.REG").getValue();
}

template <> inline StringRef buildSyncCallee<pto::BarrierOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.BARRIER").getValue();
}

template <> inline StringRef buildSyncCallee<pto::SyncSetOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.CROSS.CORE").getValue();
}

template <> inline StringRef buildSyncCallee<pto::SyncWaitOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.WAIT.FLAG.DEV.REG").getValue();
}

template <> inline StringRef buildSyncCallee<pto::SetIntraBlockOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.SET.INTRA.BLOCK.mode").getValue();
}

template <> inline StringRef buildSyncCallee<pto::WaitIntraBlockOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.WAIT.INTRA.BLOCK.mode").getValue();
}

template <> inline StringRef buildSyncCallee<pto::GetBufOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.BUFI.mode").getValue();
}

template <> inline StringRef buildSyncCallee<pto::RlsBufOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.RLS.BUFI.mode").getValue();
}

template <typename QueryOp> StringRef buildRuntimeQueryCallee(MLIRContext *context);

template <> inline StringRef buildRuntimeQueryCallee<pto::GetBlockIdxOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.BLOCK.IDX").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetSubBlockIdxOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.SUBBLOCKID").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetBlockNumOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.BLOCK.NUM").getValue();
}

template <> inline StringRef buildRuntimeQueryCallee<pto::GetSubBlockNumOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.GET.SUBBLOCKDIM").getValue();
}

template <typename QueryOp> StringRef buildSimtBlockQueryCallee(MLIRContext *context);

template <> inline StringRef buildSimtBlockQueryCallee<pto::GetBlockIdxOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.tpe.get.BLOCK.IDX").getValue();
}

template <> inline StringRef buildSimtBlockQueryCallee<pto::GetBlockNumOp>(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.tpe.get.BLOCK.NUM").getValue();
}


} // namespace mlir::pto::detail
