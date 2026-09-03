// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOBridgeRegistry.cpp - Strongly typed bridge ABI registry -------===//

#include "PTO/Transforms/VPTOBridgeRegistry.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr BridgeValueKind kPipeInitArgs[] = {BridgeValueKind::I32};
constexpr BridgeValueKind kPipeInitResults[] = {BridgeValueKind::PipeObject};
constexpr BridgeValueKind kPipePushArgs[] = {BridgeValueKind::PipeObject,
                                              BridgeValueKind::I64};
constexpr BridgeValueKind kPipePopArgs[] = {BridgeValueKind::PipeObject};
constexpr BridgeValueKind kPipePopResults[] = {BridgeValueKind::I64};
constexpr BridgeValueKind kPipeFreeArgs[] = {BridgeValueKind::PipeObject};
constexpr BridgeValueKind kMatmulArgs[] = {BridgeValueKind::I64,
                                           BridgeValueKind::I64,
                                           BridgeValueKind::I64};
constexpr BridgeValueKind kGemvArgs[] = {BridgeValueKind::I64,
                                         BridgeValueKind::I64,
                                         BridgeValueKind::I64};

const BridgeFunctionDesc kRegistry[] = {
    {BridgeEntryId::PipeInit, BridgeFamily::Pipe, BridgeRendererKind::Pipe,
     "pto.initialize_l2l_pipe", "pto_vpto_pipe_init", kPipeInitArgs,
     kPipeInitResults, true, 32},
    {BridgeEntryId::PipeSize, BridgeFamily::Pipe, BridgeRendererKind::Pipe,
     "", "pto_vpto_pipe_size", {}, {}, false},
    {BridgeEntryId::PipePush, BridgeFamily::Pipe, BridgeRendererKind::Pipe,
     "pto.tpush", "pto_vpto_pipe_push", kPipePushArgs, {}, false},
    {BridgeEntryId::PipePop, BridgeFamily::Pipe, BridgeRendererKind::Pipe,
     "pto.tpop", "pto_vpto_pipe_pop", kPipePopArgs, kPipePopResults, false},
    {BridgeEntryId::PipeFree, BridgeFamily::Pipe, BridgeRendererKind::Pipe,
     "pto.tfree", "pto_vpto_pipe_free", kPipeFreeArgs, {}, false},
    {BridgeEntryId::CubeTMatmul, BridgeFamily::Cube,
     BridgeRendererKind::CubeDirect, "pto.tmatmul", "pto_vpto_tmatmul",
     kMatmulArgs, {}, false},
    {BridgeEntryId::CubeTgemv, BridgeFamily::Cube,
     BridgeRendererKind::CubeDirect, "pto.tgemv", "pto_vpto_tgemv", kGemvArgs,
     {}, false},
};

} // namespace

llvm::ArrayRef<BridgeFunctionDesc> pto::getBridgeFunctionRegistry() {
  return kRegistry;
}

const BridgeFunctionDesc *pto::findBridgeFunction(BridgeEntryId id) {
  auto it = llvm::find_if(kRegistry,
                          [id](const BridgeFunctionDesc &desc) {
                            return desc.id == id;
                          });
  return it == std::end(kRegistry) ? nullptr : &*it;
}

const BridgeFunctionDesc *
pto::findBridgeFunctionByOp(llvm::StringRef opName) {
  auto it = llvm::find_if(kRegistry,
                          [opName](const BridgeFunctionDesc &desc) {
                            return desc.opName == opName;
                          });
  return it == std::end(kRegistry) ? nullptr : &*it;
}

const BridgeFunctionDesc *
pto::findBridgeFunctionBySymbol(llvm::StringRef symbol) {
  auto it = llvm::find_if(kRegistry,
                          [symbol](const BridgeFunctionDesc &desc) {
                            return desc.symbolBase == symbol;
                          });
  return it == std::end(kRegistry) ? nullptr : &*it;
}

llvm::StringRef pto::stringifyBridgeEntryId(BridgeEntryId id) {
  switch (id) {
  case BridgeEntryId::PipeInit:
    return "pipe.init";
  case BridgeEntryId::PipeSize:
    return "pipe.size";
  case BridgeEntryId::PipePush:
    return "pipe.push";
  case BridgeEntryId::PipePop:
    return "pipe.pop";
  case BridgeEntryId::PipeFree:
    return "pipe.free";
  case BridgeEntryId::CubeTMatmul:
    return "cube.tmatmul";
  case BridgeEntryId::CubeTgemv:
    return "cube.tgemv";
  }
  llvm_unreachable("unknown bridge entry ID");
}

llvm::StringRef pto::stringifyBridgeFamily(BridgeFamily family) {
  switch (family) {
  case BridgeFamily::Pipe:
    return "pipe";
  case BridgeFamily::Cube:
    return "cube";
  }
  llvm_unreachable("unknown bridge family");
}
