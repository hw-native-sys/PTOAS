// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- SyncCommSupport.cpp - SyncComm lowering helpers --------------------------------===//
//===----------------------------------------------------------------------===//

#include "../PTOToEmitCEmitters.h"
#include "SyncCommInternal.h"

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

// CANN Open Software License Agreement Version 2.0 (the "License").
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

//===- PTOToEmitCSyncComm.cpp - sync/barrier/comm/async/declare lowering ---------===//
//===----------------------------------------------------------------------===//





//===----------------------------------------------------------------------===//
// Return lowering
//===----------------------------------------------------------------------===

//===----------------------------------------------------------------------===//
// Sync lowering
//===----------------------------------------------------------------------===


[[maybe_unused]] static std::string getPipeName(pto::PIPE pipe) {
  switch (pipe) {
    case pto::PIPE::PIPE_S: return "PIPE_S";
    case pto::PIPE::PIPE_V: return "PIPE_V";
    case pto::PIPE::PIPE_M: return "PIPE_M";
    case pto::PIPE::PIPE_MTE1: return "PIPE_MTE1";
    case pto::PIPE::PIPE_MTE2: return "PIPE_MTE2";
    case pto::PIPE::PIPE_MTE3: return "PIPE_MTE3";
    case pto::PIPE::PIPE_ALL: return "PIPE_ALL";
    case pto::PIPE::PIPE_MTE4: return "PIPE_MTE4";
    case pto::PIPE::PIPE_MTE5: return "PIPE_MTE5";
    case pto::PIPE::PIPE_V2: return "PIPE_V2";
    case pto::PIPE::PIPE_FIX: return "PIPE_FIX";
    case pto::PIPE::VIRTUAL_PIPE_MTE2_L1A: return "VIRTUAL_PIPE_MTE2_L1A";
    case pto::PIPE::VIRTUAL_PIPE_MTE2_L1B: return "VIRTUAL_PIPE_MTE2_L1B";
    // 默认回退
    default: return "PIPE_ALL"; 
  }
}

//===----------------------------------------------------------------------===//
// pto.barrier lowering -> pipe_barrier(...)

//===----------------------------------------------------------------------===//

// Sync lowering (robust for bracket form pto.set_flag[...] / pto.wait_flag[...])
// Replace your PTOSyncToRuntimeCall with the code below.
//===----------------------------------------------------------------------===//

bool tryConvertPipeAttrToToken(Attribute attr, std::string &token) {
  if (!attr)
    return false;
  if (auto pipe = dyn_cast<mlir::pto::PipeAttr>(attr)) {
    token = mlir::pto::stringifyPIPE(pipe.getPipe()).str();
    return true;
  }
  if (auto stringAttr = dyn_cast<StringAttr>(attr)) {
    token = stringAttr.getValue().str();
    return true;
  }
  return false;
}

bool tryConvertEventAttrToToken(Attribute attr, std::string &token) {
  if (!attr)
    return false;
  if (auto event = dyn_cast<mlir::pto::EventAttr>(attr)) {
    token = mlir::pto::stringifyEVENT(event.getEvent()).str();
    return true;
  }
  if (auto stringAttr = dyn_cast<StringAttr>(attr)) {
    token = stringAttr.getValue().str();
    return true;
  }
  return false;
}

bool tryAssignSyncTokens(Attribute srcAttr, Attribute dstAttr,
                                Attribute evtAttr, std::string &srcTok,
                                std::string &dstTok, std::string &evtTok) {
  std::string localSrc;
  std::string localDst;
  std::string localEvt;
  if (!tryConvertPipeAttrToToken(srcAttr, localSrc) ||
      !tryConvertPipeAttrToToken(dstAttr, localDst) ||
      !tryConvertEventAttrToToken(evtAttr, localEvt)) {
    return false;
  }
  srcTok = std::move(localSrc);
  dstTok = std::move(localDst);
  evtTok = std::move(localEvt);
  return true;
}

bool tryExtractSyncTokensFromNamedAttrs(Operation *op,
                                               StringRef srcName,
                                               StringRef dstName,
                                               StringRef evtName,
                                               std::string &srcTok,
                                               std::string &dstTok,
                                               std::string &evtTok) {
  return tryAssignSyncTokens(op->getAttr(srcName), op->getAttr(dstName),
                             op->getAttr(evtName), srcTok, dstTok, evtTok);
}

bool tryExtractSyncTokensFromArrayAttr(Operation *op, StringRef attrName,
                                              std::string &srcTok,
                                              std::string &dstTok,
                                              std::string &evtTok) {
  auto arrayAttr = op->getAttrOfType<ArrayAttr>(attrName);
  if (!arrayAttr || arrayAttr.size() < 3)
    return false;
  return tryAssignSyncTokens(arrayAttr[0], arrayAttr[1], arrayAttr[2], srcTok,
                             dstTok, evtTok);
}

bool tryExtractFallbackSyncTokens(Operation *op, std::string &srcTok,
                                         std::string &dstTok,
                                         std::string &evtTok) {
  SmallVector<std::string, 2> pipes;
  std::string event;
  for (NamedAttribute namedAttr : op->getAttrs()) {
    std::string token;
    if (tryConvertPipeAttrToToken(namedAttr.getValue(), token)) {
      pipes.push_back(std::move(token));
      continue;
    }
    if (event.empty() &&
        tryConvertEventAttrToToken(namedAttr.getValue(), token)) {
      event = std::move(token);
    }
  }
  if (pipes.size() < 2 || event.empty())
    return false;
  srcTok = pipes[0];
  dstTok = pipes[1];
  evtTok = event;
  return true;
}

LogicalResult extractSyncTripletTokens(Operation *op,
                                             std::string &srcTok,
                                             std::string &dstTok,
                                             std::string &evtTok,
                                             ConversionPatternRewriter &rewriter) {
  if (tryExtractSyncTokensFromNamedAttrs(op, "src_pipe", "dst_pipe", "event_id",
                                         srcTok, dstTok, evtTok) ||
      tryExtractSyncTokensFromNamedAttrs(op, "srcPipe", "dstPipe", "eventId",
                                         srcTok, dstTok, evtTok) ||
      tryExtractSyncTokensFromNamedAttrs(op, "src", "dst", "event", srcTok,
                                         dstTok, evtTok)) {
    return success();
  }

  for (StringRef attrName : {"args", "pipes", "sync", "triplet", "attrs"}) {
    if (tryExtractSyncTokensFromArrayAttr(op, attrName, srcTok, dstTok,
                                          evtTok)) {
      return success();
    }
  }

  if (tryExtractFallbackSyncTokens(op, srcTok, dstTok, evtTok))
    return success();
  return rewriter.notifyMatchFailure(
      op, "cannot extract PIPE/PIPE/EVENT tokens from pto.{set,wait}_flag");
}
// set_flag/wait_flag lowering: resolve the PIPE/PIPE/EVENT token triplet and
// emit the matching runtime call.

// GetBlockIdxOp Lowering (pto.get_block_idx -> get_block_idx())

// GetBlockNumOp Lowering (pto.get_block_num -> get_block_num())

// GetSubBlockIdxOp Lowering (pto.get_block_idx -> get_subblockid())

// GetSubBlockNumOp Lowering.



} // namespace pto
} // namespace mlir
