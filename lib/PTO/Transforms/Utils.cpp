// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "Utils.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "pto-utils"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define DBGSNL() (llvm::dbgs() << "\n")

namespace mlir {
namespace pto {

static constexpr llvm::StringLiteral kFrontendPipeIdAttrName =
    "__pto.frontend_id";

FailureOr<bool> hasTFillPadExpandedPhysicalShape(TFillPadOp op) {
  auto srcType = dyn_cast<TileBufType>(op.getSrc().getType());
  auto dstType = dyn_cast<TileBufType>(op.getDst().getType());
  if (!srcType || !dstType || srcType.getRank() != dstType.getRank())
    return failure();

  bool expanded = false;
  for (auto [srcDim, dstDim] :
       llvm::zip_equal(srcType.getShape(), dstType.getShape())) {
    if (srcDim == dstDim)
      continue;
    if (ShapedType::isDynamic(srcDim) || ShapedType::isDynamic(dstDim) ||
        dstDim < srcDim)
      return failure();
    expanded = true;
  }
  return expanded;
}

static Value peelTFillPadStorageAlias(Value value) {
  constexpr unsigned kMaxDepth = 32;
  for (unsigned depth = 0; value && depth < kMaxDepth; ++depth) {
    Operation *def = value.getDefiningOp();
    if (!def)
      break;
    if (auto cast = dyn_cast<UnrealizedConversionCastOp>(def)) {
      if (cast.getNumOperands() != 1 || cast.getNumResults() != 1)
        break;
      value = cast.getOperand(0);
      continue;
    }
    if (auto bitcast = dyn_cast<BitcastOp>(def)) {
      value = bitcast.getSrc();
      continue;
    }
    if (auto reshape = dyn_cast<TReshapeOp>(def)) {
      value = reshape.getSrc();
      continue;
    }
    break;
  }
  return value;
}

static bool haveSameKnownTFillPadStartAddress(Value src, Value dst) {
  src = peelTFillPadStorageAlias(src);
  dst = peelTFillPadStorageAlias(dst);
  if (src == dst)
    return true;

  auto srcAlloc = src.getDefiningOp<AllocTileOp>();
  auto dstAlloc = dst.getDefiningOp<AllocTileOp>();
  if (!srcAlloc || !dstAlloc || !srcAlloc.getAddr() || !dstAlloc.getAddr())
    return false;

  Value srcAddr = srcAlloc.getAddr();
  Value dstAddr = dstAlloc.getAddr();
  if (srcAddr == dstAddr)
    return true;

  IntegerAttr srcConst;
  IntegerAttr dstConst;
  return matchPattern(srcAddr, m_Constant(&srcConst)) &&
         matchPattern(dstAddr, m_Constant(&dstConst)) &&
         srcConst.getValue() == dstConst.getValue();
}

FailureOr<TFillPadLoweringKind>
inferTFillPadLoweringKindAfterMemoryPlanning(TFillPadOp op) {
  FailureOr<bool> expanded = hasTFillPadExpandedPhysicalShape(op);
  if (failed(expanded))
    return failure();

  auto srcSpace = GetBufferSpaceAttr(op.getSrc());
  auto dstSpace = GetBufferSpaceAttr(op.getDst());
  bool isVec = srcSpace && dstSpace &&
               srcSpace->getAddressSpace() == AddressSpace::VEC &&
               dstSpace->getAddressSpace() == AddressSpace::VEC;

  if (*expanded) {
    if (!isVec)
      return failure();
    return TFillPadLoweringKind::Expand;
  }
  if (isVec &&
      haveSameKnownTFillPadStartAddress(op.getSrc(), op.getDst()))
    return TFillPadLoweringKind::InPlace;
  return TFillPadLoweringKind::Normal;
}

std::optional<PhysicalSectionKind>
inferPhysicalSectionKindFromPipe(Operation *op) {
  auto pipeOp = dyn_cast_or_null<OpPipeInterface>(op);
  if (!pipeOp)
    return std::nullopt;

  switch (pipeOp.getPipe()) {
  case PIPE::PIPE_M:
  case PIPE::PIPE_MTE1:
    return PhysicalSectionKind::Cube;
  case PIPE::PIPE_V:
  case PIPE::PIPE_V2:
  case PIPE::PIPE_S:
    return PhysicalSectionKind::Vector;
  default:
    return std::nullopt;
  }
}

func::ReturnOp getAssumedUniqueReturnOp(func::FuncOp funcOp) {
  func::ReturnOp returnOp;
  for (Block &b : funcOp.getBody()) {
    if (auto candidateOp = dyn_cast<func::ReturnOp>(b.getTerminator())) {
      if (returnOp)
        return nullptr;
      returnOp = candidateOp;
    }
  }
  return returnOp;
}

Value peelUnrealized(Value value) {
  if (auto castOp = value.getDefiningOp<UnrealizedConversionCastOp>())
    return castOp.getOperand(0);
  return value;
}

bool isScalarFixpipeQuant(FixpipeQuant quant) {
  switch (quant) {
  case FixpipeQuant::DEQF16Scalar:
  case FixpipeQuant::REQ8Scalar:
  case FixpipeQuant::QF322B8PreScalar:
  case FixpipeQuant::QF322F16PreScalar:
  case FixpipeQuant::QF322BF16PreScalar:
  case FixpipeQuant::QS322BF16PreScalar:
  case FixpipeQuant::QF322HIF8PreScalar:
  case FixpipeQuant::QF322FP8PreScalar:
    return true;
  default:
    return false;
  }
}

bool isVectorFixpipeQuant(FixpipeQuant quant) {
  switch (quant) {
  case FixpipeQuant::DEQF16Vec:
  case FixpipeQuant::REQ8Vec:
  case FixpipeQuant::QF322B8PreVec:
  case FixpipeQuant::QS322BF16PreVec:
    return true;
  default:
    return false;
  }
}

Operation *getPipeInitDef(Value pipeHandle) {
  pipeHandle = peelUnrealized(pipeHandle);
  return pipeHandle ? pipeHandle.getDefiningOp() : nullptr;
}

AccPushEpilogueAttr getPipeInitAccPushEpilogue(Operation *initOp) {
  if (auto init = dyn_cast_or_null<InitializeL2LPipeOp>(initOp))
    return init.getAccPushEpilogueAttr();
  if (auto init = dyn_cast_or_null<InitializeL2G2LPipeOp>(initOp))
    return init.getAccPushEpilogueAttr();
  return {};
}

std::optional<int32_t> getFrontendPipeIdFromInit(Operation *initOp) {
  if (!initOp)
    return std::nullopt;
  if (auto attr = initOp->getAttrOfType<IntegerAttr>(kFrontendPipeIdAttrName))
    return static_cast<int32_t>(attr.getInt());
  return std::nullopt;
}

std::optional<int32_t> getFrontendPipeIdFromHandle(Value pipeHandle) {
  return getFrontendPipeIdFromInit(getPipeInitDef(pipeHandle));
}

// New helper function to get the updated BaseMemRefType
BaseMemRefType getBaseMemRefTypeWithNewScope(BaseMemRefType type,
                                             AddressSpaceAttr targetMemScope) {
  if (auto memRefType = dyn_cast<MemRefType>(type)) {
    return MemRefType::Builder(memRefType).setMemorySpace(targetMemScope);
  } else if (auto unrankedMemRefType = dyn_cast<UnrankedMemRefType>(type)) {
    return UnrankedMemRefType::get(unrankedMemRefType.getElementType(),
                                   targetMemScope);
  }
  llvm_unreachable("Unexpected BaseMemRefType");
  return type;
}

void setBaseMemRefTypeScope(Value val, AddressSpaceAttr targetMemScope) {
  Type type = val.getType();
  if (!isa<BaseMemRefType>(type)) {
    return;
  }

  if (auto curMemScope = dyn_cast_if_present<AddressSpaceAttr>(
          dyn_cast<BaseMemRefType>(type).getMemorySpace())) {
    if (curMemScope != targetMemScope)
      llvm::report_fatal_error("memref scope mismatch while propagating PTO address space");
    return;
  }

  auto memRefType = cast<BaseMemRefType>(type);
  auto newMemRefType =
      getBaseMemRefTypeWithNewScope(memRefType, targetMemScope);
  val.setType(newMemRefType);
}


std::optional<AddressSpaceAttr> GetBufferSpaceAttr(Value operand) {
  if (auto tileTy = dyn_cast<pto::TileBufType>(operand.getType())) {
    if (auto memorySpaceAttr =
            dyn_cast_or_null<AddressSpaceAttr>(tileTy.getMemorySpace()))
      return memorySpaceAttr;
    return std::nullopt;
  }
  if (auto multiTy = dyn_cast<pto::MultiTileBufType>(operand.getType())) {
    if (auto memorySpaceAttr = dyn_cast_or_null<AddressSpaceAttr>(
            multiTy.getSlotType().getMemorySpace()))
      return memorySpaceAttr;
    return std::nullopt;
  }

  if (!llvm::isa<MemRefType>(operand.getType())) {
    return std::nullopt;
  }
  auto memRefType = cast<MemRefType>(operand.getType());
  auto memorySpace = memRefType.getMemorySpace();
  if (!memorySpace)
    return std::nullopt;
  auto memorySpaceAttr = dyn_cast<AddressSpaceAttr>(memorySpace);
  if (!memorySpaceAttr) {
    return std::nullopt;
  }
  return memorySpaceAttr;
}

std::optional<std::pair<Value, Value>> getOperationAliasInfo(Operation *op) {
  if (auto subViewOp = dyn_cast<memref::SubViewOp>(op)) {
    return std::make_pair(subViewOp.getResult(), subViewOp.getViewSource());
  } else if (auto makeViewOp = dyn_cast<pto::MakeTensorViewOp>(op)) {
    return std::make_pair(makeViewOp.getResult(), makeViewOp.getPtr());
  } else if (auto partViewOp = dyn_cast<pto::PartitionViewOp>(op)) {
    return std::make_pair(partViewOp.getResult(), partViewOp.getSource());
  } else if (auto addPtrOp = dyn_cast<pto::AddPtrOp>(op)) {
    return std::make_pair(addPtrOp.getResult(), addPtrOp.getPtr());
  } else if (auto ptrToIntOp = dyn_cast<pto::PtrToIntOp>(op)) {
    return std::make_pair(ptrToIntOp.getResult(), ptrToIntOp.getPtr());
  } else if (auto intToPtrOp = dyn_cast<pto::IntToPtrOp>(op)) {
    return std::make_pair(intToPtrOp.getResult(), intToPtrOp.getAddr());
  } else if (auto castPtrOp = dyn_cast<pto::CastPtrOp>(op)) {
    return std::make_pair(castPtrOp.getResult(), castPtrOp.getInput());
  } else if (auto subViewOp = dyn_cast<pto::SubViewOp>(op)) {
    return std::make_pair(subViewOp.getResult(), subViewOp.getSource());
  } else if (auto bitcastOp = dyn_cast<pto::BitcastOp>(op)) {
    return std::make_pair(bitcastOp.getResult(), bitcastOp.getSrc());
  } else if (auto reshapeOp = dyn_cast<pto::TReshapeOp>(op)) {
    return std::make_pair(reshapeOp.getResult(), reshapeOp.getSrc());
  } else if (auto multiGetOp = dyn_cast<pto::MultiTileGetOp>(op)) {
    return std::make_pair(multiGetOp.getResult(), multiGetOp.getSource());
  } else if (auto extSliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
    return std::make_pair(extSliceOp.getResult(), extSliceOp.getSource());
  } else if (auto collapseShapeOp = dyn_cast<memref::CollapseShapeOp>(op)) {
    return std::make_pair(collapseShapeOp.getResult(),
                          collapseShapeOp.getViewSource());
  } else if (auto expandShapeOp = dyn_cast<memref::ExpandShapeOp>(op)) {
    return std::make_pair(expandShapeOp.getResult(),
                          expandShapeOp.getViewSource());
  } else if (auto viewOp = dyn_cast<memref::ViewOp>(op)) {
    return std::make_pair(viewOp.getResult(), viewOp.getViewSource());
  } else if (auto reinterpretCastOp = dyn_cast<memref::ReinterpretCastOp>(op)) {
    return std::make_pair(reinterpretCastOp.getResult(),
                          reinterpretCastOp.getViewSource());
  } else if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(op)) {
    return std::make_pair(reshapeOp.getResult(), reshapeOp.getViewSource());
  } else if (auto castOp = dyn_cast<memref::CastOp>(op)) {
    return std::make_pair(castOp.getResult(), castOp.getViewSource());
  } else if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
    if (castOp.getNumOperands() == 1 && castOp.getNumResults() == 1)
      return std::make_pair(castOp.getResult(0), castOp.getOperand(0));
  } else if (auto extractStridedMetadataOp =
                 dyn_cast<memref::ExtractStridedMetadataOp>(op)) {
    return std::make_pair(extractStridedMetadataOp.getBaseBuffer(),
                          extractStridedMetadataOp.getViewSource());
  } else if (auto toBufferOp = dyn_cast<bufferization::ToBufferOp>(op)) {
    return std::make_pair(toBufferOp.getBuffer(), toBufferOp.getTensor());
  } else if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(op)) {
    return std::make_pair(toTensorOp.getResult(), toTensorOp.getOperand());
  }
  return std::nullopt;
}

Value tracebackImpl(Value memrefVal) {
  // case 1: v is the iter_arg of a scf.for
  if (auto arg = dyn_cast<BlockArgument>(memrefVal)) {
    if (auto forOp =
            dyn_cast<scf::ForOp>(arg.getParentRegion()->getParentOp())) {
      if (arg.getArgNumber() > 0 &&
          forOp.getInitArgs().size() > arg.getArgNumber() - 1) {
        return forOp.getInitArgs()[arg.getArgNumber() - 1];
      }
    }
  }

  Value result;
  Operation *def = memrefVal.getDefiningOp();
  if (!def) {
    // failed to trace back
    return result;
  }

  // case 2: v is the result of cast-like ops
  //  - memref.cast
  //  - memref.collapse_shape
  //  - memref.expand_shape
  //  - memref.memory_space_cast
  //  - memref.reinterpret_cast
  //  - memref.reshape
  //  - memref.transpose
  if (auto op = dyn_cast<memref::CastOp>(def)) {
    result = op.getSource();
  } else if (auto op = dyn_cast<memref::CollapseShapeOp>(def)) {
    result = op.getSrc();
  } else if (auto op = dyn_cast<memref::ExpandShapeOp>(def)) {
    result = op.getSrc();
  } else if (auto op = dyn_cast<memref::MemorySpaceCastOp>(def)) {
    result = op.getSource();
  } else if (auto op = dyn_cast<memref::ReinterpretCastOp>(def)) {
    result = op.getSource();
  } else if (auto op = dyn_cast<memref::ReshapeOp>(def)) {
    result = op.getSource();
  } else if (auto op = dyn_cast<memref::TransposeOp>(def)) {
    result = op.getIn();
  } else if (auto op = dyn_cast<UnrealizedConversionCastOp>(def)) {
    result = op.getOperand(cast<OpResult>(memrefVal).getResultNumber());
  } else if (auto op = dyn_cast<scf::ForOp>(def)) {
    // trace back memref.alloc support scf.for
    result = op.getInitArgs()[cast<OpResult>(memrefVal).getResultNumber()];
  }

  if (result) {
    return result;
  }

  // case 3: v is the result of the view-like ops
  //  - memref::view
  //  - memref::subview
  if (auto op = dyn_cast<memref::ViewOp>(def)) {
    result = op.getViewSource();
  } else if (auto op = dyn_cast<memref::SubViewOp>(def)) {
    result = op.getViewSource();
  }

  return result;
}

bool isAllocLikeOp(Operation *op) {
  if (!op)
    return false;
  return isa<memref::AllocOp>(op) || isa<memref::AllocaOp>(op);
}

bool isAllocLikeOp(Value val) {
  return isAllocLikeOp(val.getDefiningOp());
}

std::optional<int64_t> getStaticTotalSize(const ArrayRef<int64_t> &shapes) {
  int64_t totalSize = 1;
  for (const auto &shape : shapes) {
    if (ShapedType::isDynamic(shape)) {
      return std::nullopt;
    }
    totalSize = totalSize * shape;
  }
  return totalSize;
}

uint64_t AlignUp(uint64_t lhs, uint64_t rhs) {
  if (rhs == 0)
    return lhs;
  if (lhs % rhs != 0) {
    lhs += rhs - (lhs % rhs);
  }
  return lhs;
}

Value tracebackMemRef(Value memrefVal) {
  int loopBound = 256;
  while (memrefVal && !isAllocLikeOp(memrefVal)) {
    auto upward = tracebackImpl(memrefVal);
    if (!upward) {
      break;
    }

    memrefVal = upward;

    // avoid infinite loop
    if (loopBound-- < 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "tracebackMemRef exceeds loopBound(" << loopBound << ")!");
      break;
    }
  }

  return memrefVal;
}

std::optional<memref::AllocOp> tracebackMemRefToAlloc(Value memrefVal) {
  auto tracedValue = tracebackMemRef(memrefVal);
  return isAllocLikeOp(tracedValue)
             ? tracedValue.getDefiningOp<memref::AllocOp>()
             : std::optional<memref::AllocOp>();
}

/// trace value and judge if it is function argument
bool isFromFunctionArg(mlir::Value v) {
  return tracebackMemRef(v).getDefiningOp() == nullptr;
}

bool isLocalBuffer(std::optional<AddressSpaceAttr> memorySpaceAttr) {
  if (!memorySpaceAttr.has_value()) {
    return false;
  }

  if (memorySpaceAttr.value().getAddressSpace() == pto::AddressSpace::GM) {
    return false;
  }
  if (LocalBufferSpace.count(memorySpaceAttr.value().getAddressSpace())) {
    return true;
  }
  llvm_unreachable("Currently only support (UB | L1 | L0C) allocation");
}

SmallVector<Value> getOpTouchBuffer(Operation *op) {
  SmallVector<Value> touchBuffer;
  touchBuffer.insert(touchBuffer.end(), op->getResults().begin(),
                     op->getResults().end());
  for (OpOperand &operand : op->getOpOperands()) {
    touchBuffer.push_back(operand.get());
  }
  return touchBuffer;
}

bool isOpTouchLocalBuffer(Operation *op) {
  auto touchBuffer = getOpTouchBuffer(op);
  for (Value buffer : touchBuffer) {
    auto bufferSpace = GetBufferSpaceAttr(buffer);
    if (isLocalBuffer(bufferSpace)) {
      return true;
    }
  }
  return false;
}

ModuleOp getTopLevelModuleOp(Operation *op) {
  ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
  while (moduleOp && moduleOp->getParentOp()) {
    moduleOp = moduleOp->getParentOfType<ModuleOp>();
  }
  return moduleOp;
}

/// Index of yielded value where is alias of targetVal.
std::optional<int> getYieldValueIdx(Value targetVal, ValueRange yieldedValues) {
  auto it = std::find(yieldedValues.begin(), yieldedValues.end(), targetVal);
  if (it != yieldedValues.end()) {
    return it - yieldedValues.begin();
  }

  return std::nullopt;
}

LoopLikeOpInterface getParentLoop(Value val) {
  if (!val.getDefiningOp())
    return nullptr;

  // Firstly, get parent loop
  LoopLikeOpInterface parentLoop =
      val.getDefiningOp()->getParentOfType<LoopLikeOpInterface>();
  if (!parentLoop) {
    return nullptr;
  }

  // Need to determine whether val is yielded by the loop.
  auto yieldedValues = parentLoop.getYieldedValues();
  if (yieldedValues.empty())
    return parentLoop;

  auto idxLoopRes = getYieldValueIdx(val, yieldedValues);
  if (idxLoopRes.has_value()) {
    // The val is yielded by loop, so need to find parent of parent loop.
    auto res = parentLoop.getLoopResults().value()[*idxLoopRes];
    return getParentLoop(res);
  }

  // Need to determine whether val is yielded by if/else.
  auto parentIf = val.getDefiningOp()->getParentOfType<scf::IfOp>();
  if (!parentIf || parentIf.getResults().empty())
    return parentLoop;

  auto thenYieldOp = parentIf.thenYield();
  auto thenYieldOpers = thenYieldOp.getOperands();

  auto idxThenYielded = getYieldValueIdx(val, thenYieldOpers);
  if (idxThenYielded.has_value()) {
    // The val is yielded by ifOp, need to find parent loop of ifOp's result
    auto res = parentIf.getResults()[*idxThenYielded];
    return getParentLoop(res);
  }

  auto elseYieldOp = parentIf.elseYield();
  auto elseYieldOpers = elseYieldOp.getOperands();
  auto idxElseYielded = getYieldValueIdx(val, elseYieldOpers);
  if (idxElseYielded.has_value()) {
    auto res = parentIf.getResults()[*idxElseYielded];
    return getParentLoop(res);
  }

  return parentLoop;
}

}
}
