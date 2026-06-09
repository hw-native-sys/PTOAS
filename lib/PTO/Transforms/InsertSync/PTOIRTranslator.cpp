// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/AsmState.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
// [P0 新增] 引入副作用接口和 PTO 接口
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <optional>

#define DEBUG_TYPE "pto-ir-translator"

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr size_t kTileRank2D = 2;
constexpr unsigned kStrideInlineCapacity = 4;
constexpr unsigned kMemoryEffectInlineCapacity = 4;

using StrideVector = SmallVector<int64_t, kStrideInlineCapacity>;
using MemoryEffectVector =
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>,
                kMemoryEffectInlineCapacity>;

} // namespace

static bool getConstIndexValue(Value value, int64_t &out) {
  while (true) {
    if (auto castOp = value.getDefiningOp<arith::IndexCastOp>()) {
      value = castOp.getIn();
      continue;
    }
    if (auto extOp = value.getDefiningOp<arith::ExtSIOp>()) {
      value = extOp.getIn();
      continue;
    }
    if (auto extOp = value.getDefiningOp<arith::ExtUIOp>()) {
      value = extOp.getIn();
      continue;
    }
    if (auto truncOp = value.getDefiningOp<arith::TruncIOp>()) {
      value = truncOp.getIn();
      continue;
    }
    break;
  }
  if (auto constIndex = value.getDefiningOp<arith::ConstantIndexOp>()) {
    out = constIndex.value();
    return true;
  }
  if (auto constInt = value.getDefiningOp<arith::ConstantIntOp>()) {
    out = constInt.value();
    return true;
  }
  auto constOp = value.getDefiningOp<arith::ConstantOp>();
  auto intAttr =
      constOp ? dyn_cast<IntegerAttr>(constOp.getValue()) : IntegerAttr();
  if (!intAttr) return false;
  out = intAttr.getInt();
  return true;
}

static bool isStaticRank2Shape(ArrayRef<int64_t> shape) {
  return shape.size() == kTileRank2D &&
         llvm::none_of(shape, [](int64_t dim) {
           return dim == ShapedType::kDynamic;
         });
}

static int64_t getTileMajorStride(pto::TileBufType type) {
  ArrayRef<int64_t> shape = type.getShape();
  int64_t rows = shape[0];
  int64_t cols = shape[1];
  bool rowMajor =
      type.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
  int64_t stride = rowMajor ? cols : rows;
  if (type.getCompactModeI32() == static_cast<int32_t>(pto::CompactMode::RowPlusOne))
    ++stride;
  return stride;
}

static std::optional<SmallVector<uint64_t>>
getPtoSubViewBaseAddresses(pto::SubViewOp op, pto::TileBufType sourceType,
                           int64_t elemBytes) {
  if (!isStaticRank2Shape(sourceType.getShape())) return std::nullopt;
  if (sourceType.getSLayoutValueI32() !=
      static_cast<int32_t>(pto::SLayout::NoneBox))
    return std::nullopt;
  if (op.getOffsets().size() != kTileRank2D) return std::nullopt;

  int64_t rowOffset = 0;
  int64_t colOffset = 0;
  if (!getConstIndexValue(op.getOffsets()[0], rowOffset) ||
      !getConstIndexValue(op.getOffsets()[1], colOffset))
    return std::nullopt;
  if (rowOffset < 0 || colOffset < 0) return std::nullopt;

  auto sizesAttr = op.getSizes();
  if (!sizesAttr || sizesAttr.size() != kTileRank2D) return std::nullopt;
  int64_t rowSize = cast<IntegerAttr>(sizesAttr[0]).getInt();
  int64_t colSize = cast<IntegerAttr>(sizesAttr[1]).getInt();
  if (rowSize <= 0 || colSize <= 0) return std::nullopt;

  bool rowMajor =
      sourceType.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
  int64_t majorStride = getTileMajorStride(sourceType);

  SmallVector<uint64_t> addresses;
  if (rowMajor) {
    addresses.reserve(static_cast<size_t>(rowSize));
    for (int64_t row = 0; row < rowSize; ++row) {
      int64_t elemOffset = (rowOffset + row) * majorStride + colOffset;
      addresses.push_back(static_cast<uint64_t>(elemOffset * elemBytes));
    }
  } else {
    addresses.reserve(static_cast<size_t>(colSize));
    for (int64_t col = 0; col < colSize; ++col) {
      int64_t elemOffset = (colOffset + col) * majorStride + rowOffset;
      addresses.push_back(static_cast<uint64_t>(elemOffset * elemBytes));
    }
  }

  return addresses;
}

static std::optional<SmallVector<uint64_t>>
buildMemrefSubViewBaseAddresses(ArrayRef<int64_t> strides, int64_t rowOffset,
                                int64_t colOffset, int64_t rowSize,
                                int64_t colSize, int64_t elemBytes) {
  SmallVector<uint64_t> addresses;
  if (strides[1] == 1) {
    addresses.reserve(static_cast<size_t>(rowSize));
    for (int64_t row = 0; row < rowSize; ++row) {
      int64_t elemOffset =
          (rowOffset + row) * strides[0] + colOffset * strides[1];
      addresses.push_back(static_cast<uint64_t>(elemOffset * elemBytes));
    }
    return addresses;
  }
  if (strides[0] == 1) {
    addresses.reserve(static_cast<size_t>(colSize));
    for (int64_t col = 0; col < colSize; ++col) {
      int64_t elemOffset =
          rowOffset * strides[0] + (colOffset + col) * strides[1];
      addresses.push_back(static_cast<uint64_t>(elemOffset * elemBytes));
    }
    return addresses;
  }
  return std::nullopt;
}

static bool parseStaticMemrefSubViewWindow(memref::SubViewOp op,
                                           int64_t &rowOffset,
                                           int64_t &colOffset,
                                           int64_t &rowSize,
                                           int64_t &colSize) {
  ArrayRef<int64_t> staticOffsets = op.getStaticOffsets();
  ArrayRef<int64_t> staticSizes = op.getStaticSizes();
  ArrayRef<int64_t> staticSubViewStrides = op.getStaticStrides();
  if (staticOffsets.size() != kTileRank2D || staticSizes.size() != kTileRank2D ||
      staticSubViewStrides.size() != kTileRank2D)
    return false;
  if (llvm::is_contained(staticOffsets, ShapedType::kDynamic) ||
      llvm::is_contained(staticSizes, ShapedType::kDynamic) ||
      llvm::is_contained(staticSubViewStrides, ShapedType::kDynamic))
    return false;
  if (staticSubViewStrides[0] != 1 || staticSubViewStrides[1] != 1)
    return false;
  rowOffset = staticOffsets[0];
  colOffset = staticOffsets[1];
  rowSize = staticSizes[0];
  colSize = staticSizes[1];
  return rowOffset >= 0 && colOffset >= 0 && rowSize > 0 && colSize > 0;
}

static std::optional<SmallVector<uint64_t>>
getMemrefSubViewBaseAddresses(memref::SubViewOp op, MemRefType sourceType,
                              int64_t elemBytes) {
  if (!sourceType.hasStaticShape() || sourceType.getRank() != static_cast<int64_t>(kTileRank2D))
    return std::nullopt;

  SmallVector<int64_t> strides;
  int64_t baseOffset = ShapedType::kDynamic;
  if (failed(mlir::getStridesAndOffset(sourceType, strides, baseOffset)) ||
      strides.size() != kTileRank2D ||
      llvm::is_contained(strides, ShapedType::kDynamic))
    return std::nullopt;

  int64_t rowOffset = 0;
  int64_t colOffset = 0;
  int64_t rowSize = 0;
  int64_t colSize = 0;
  if (!parseStaticMemrefSubViewWindow(op, rowOffset, colOffset, rowSize,
                                      colSize))
    return std::nullopt;
  return buildMemrefSubViewBaseAddresses(strides, rowOffset, colOffset, rowSize,
                                         colSize, elemBytes);
}

// [辅助函数] 尝试从 Operation 中计算相对于 Source 的字节偏移量和新大小
// 返回值: pair<offsetInBytes, sizeInBytes>
// 如果无法计算静态值，返回 {-1, -1} 表示这是动态的
static std::pair<int64_t, int64_t> getStaticOffsetAndSize(Operation *op, Value src) {
  auto srcType = dyn_cast<MemRefType>(src.getType());
  if (!srcType) return {0, 0};

  int64_t elemSize =
      static_cast<int64_t>(pto::getPTOStorageElemByteSize(srcType.getElementType()));
  if (elemSize == 0)
    return {-1, -1};

  // === Case 1: memref.subview ===
  if (auto subView = dyn_cast<memref::SubViewOp>(op)) {
    int64_t baseOffset;
    StrideVector strides;
    if (failed(mlir::getStridesAndOffset(srcType, strides, baseOffset))) {
        return {-1, -1};
    }

    int64_t newSize = 1;
    for (int64_t s : subView.getStaticSizes()) {
      if (s == ShapedType::kDynamic) return {-1, -1};
      newSize *= s;
    }
    newSize *= elemSize;

    int64_t totalOffset = 0;
    auto staticOffsets = subView.getStaticOffsets();
    if (staticOffsets.empty()) return {-1, -1};
    if (staticOffsets.size() > strides.size()) return {-1, -1};

    for (size_t i = 0; i < staticOffsets.size(); ++i) {
      int64_t off = staticOffsets[i];
      if (off == ShapedType::kDynamic) return {-1, -1};

      int64_t stride = 1;
      if (i < strides.size() && strides[i] != ShapedType::kDynamic) {
          stride = strides[i];
      } else {
          return {-1, -1};
      }

      totalOffset += off * stride;
    }

    return {totalOffset * elemSize, newSize};
  }

  // === Case 2: memref.reinterpret_cast ===
  if (auto castOp = dyn_cast<memref::ReinterpretCastOp>(op)) {
    auto staticOffsets = castOp.getStaticOffsets();
    if (staticOffsets.empty() || staticOffsets[0] == ShapedType::kDynamic) {
        return {0, 0};
    }
    return {staticOffsets[0] * elemSize, 0};
  }

  return {0, 0};
}

// ============================================================================
// 1. 构建入口
// ============================================================================
void PTOIRTranslator::Build() {
  Region &funcRegion = func_.getBody();
  UpdateKernelArgMemInfo();
  RecursionIR(&funcRegion);
}

// ============================================================================
// 2. 更新 Kernel 参数内存信息 (GM Global Memory)
// ============================================================================
void PTOIRTranslator::UpdateKernelArgMemInfo() {
  auto funcParamSize = func_.getNumArguments();
  for (size_t i = 0; i < funcParamSize; i++) {
    Value funcArg = func_.getArgument(i);
    Type argType = funcArg.getType();
    if (!isa<pto::PtrType>(argType) && !isa<MemRefType>(argType)) {
      continue;
    }

    std::unique_ptr<BaseMemInfo> newMemInfo = std::make_unique<BaseMemInfo>(
        funcArg,                  // baseBuffer
        funcArg,                  // rootBuffer
        pto::AddressSpace::GM,    // Scope
        SmallVector<uint64_t>{0}, // Base Addresses
        0                         // Allocate Size
    );

    buffer2MemInfoMap_[funcArg].emplace_back(newMemInfo->clone());
  }
}

// ============================================================================
// 3. 递归遍历 IR (核心分发逻辑)
// ============================================================================
void PTOIRTranslator::RecursionIR(Region *region) {
  auto result = region->walk<WalkOrder::PreOrder>(
      [this](Operation *op) { return TranslateOperation(op); });
  if (result == WalkResult::interrupt()) {
    llvm_unreachable("PTO InjectSync Traverse IR Failed!");
  }
}

std::optional<WalkResult> PTOIRTranslator::HandleControlFlowOp(Operation *op) {
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    UpdateForOpInfo(forOp);
    return WalkResult::skip();
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    UpdateWhileOpInfo(whileOp);
    return WalkResult::skip();
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    UpdateIfOpInfo(ifOp);
    return WalkResult::skip();
  }
  if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
    UpdateYieldOpInfo(yieldOp);
    return WalkResult::advance();
  }
  return std::nullopt;
}

LogicalResult PTOIRTranslator::HandleAllocLikeOp(Operation *op, bool &handled) {
  handled = true;
  if (auto allocOp = dyn_cast<pto::AllocTileOp>(op))
    return UpdateAllocTileOpMemInfo(allocOp);
  if (auto memAllocOp = dyn_cast<memref::AllocOp>(op))
    return UpdateMemrefAllocOpMemInfo(memAllocOp);
  if (auto declareOp = dyn_cast<pto::DeclareTileMemRefOp>(op))
    return UpdateDeclareTileMemRefOpMemInfo(declareOp);
  if (auto castOp = dyn_cast<pto::PointerCastOp>(op))
    return UpdatePointerCastOpMemInfo(castOp);
  handled = false;
  return success();
}

bool PTOIRTranslator::HandleAliasLikeOp(Operation *op) {
  if (auto makeViewOp = dyn_cast<pto::MakeTensorViewOp>(op)) {
    UpdateAliasBufferInfo(makeViewOp.getResult(), makeViewOp.getPtr());
    return true;
  }
  if (auto bindTileOp = dyn_cast<pto::BindTileOp>(op)) {
    UpdateAliasBufferInfo(bindTileOp.getResult(), bindTileOp.getSource());
    return true;
  }
  if (auto partitionOp = dyn_cast<pto::PartitionViewOp>(op)) {
    UpdateAliasBufferInfo(partitionOp.getResult(), partitionOp.getSource());
    return true;
  }
  if (auto subViewOp = dyn_cast<pto::SubViewOp>(op)) {
    UpdateTileSubViewAliasBufferInfo(subViewOp);
    return true;
  }
  if (auto memrefSubView = dyn_cast<memref::SubViewOp>(op)) {
    UpdateMemrefSubViewAliasBufferInfo(memrefSubView);
    return true;
  }
  if (auto castOp = dyn_cast<memref::ReinterpretCastOp>(op)) {
    UpdateAliasBufferInfo(castOp.getResult(), castOp.getSource());
    return true;
  }
  if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(op)) {
    UpdateAliasBufferInfo(collapseOp.getResult(), collapseOp.getSrc());
    return true;
  }
  if (auto expandOp = dyn_cast<memref::ExpandShapeOp>(op)) {
    UpdateAliasBufferInfo(expandOp.getResult(), expandOp.getSrc());
    return true;
  }
  return false;
}

WalkResult PTOIRTranslator::TranslateOperation(Operation *op) {
  if (auto ctrl = HandleControlFlowOp(op))
    return *ctrl;
  bool handled = false;
  if (failed(HandleAllocLikeOp(op, handled)))
    return WalkResult::interrupt();
  if (handled || HandleAliasLikeOp(op))
    return WalkResult::advance();
  if (isa<pto::OpPipeInterface>(op))
    UpdatePTOOpInfo(op);
  return WalkResult::advance();
}

// ============================================================================
// 4. 处理 AllocTile / PointerCast
// ============================================================================
LogicalResult PTOIRTranslator::UpdateAllocTileOpMemInfo(pto::AllocTileOp op) {
  Value res = op.getResult();

  auto tileType = dyn_cast<pto::TileBufType>(res.getType());
  uint64_t sizeInBytes = 0;
  uint64_t baseAddr = 0;

  // If alloc_tile carries an explicit address, record it when it's a constant.
  if (Value addr = op.getAddr()) {
    llvm::APInt apIntValue;
    if (matchPattern(addr, m_ConstantInt(&apIntValue))) {
        // 将 APInt 转换为 int64_t，再转为 uint64_t
        int64_t c = apIntValue.getSExtValue();  // 有符号扩展转换
        // 如果确定是无符号值，也可以用：apIntValue.getZExtValue()
        baseAddr = static_cast<uint64_t>(c);
    }
  }

  // 1. 计算大小
  if (tileType) {
    ArrayRef<int64_t> shape = tileType.getShape();
    bool isStatic = true;
    for (int64_t dim : shape) {
      if (dim == ShapedType::kDynamic) {
        isStatic = false;
        break;
      }
    }

    if (isStatic) {
      int64_t elemSize = static_cast<int64_t>(
          pto::getPTOStorageElemByteSize(tileType.getElementType()));
      if (elemSize == 0)
        return failure();
      int64_t numElements = 1;
      for (auto dim : shape) numElements *= dim;
      sizeInBytes = numElements * elemSize;
    }
  }

  // 2. 解析地址空间
  // 默认设为 MAT (Matrix Buffer)，但优先读取 Type 中的属性
  pto::AddressSpace space = pto::AddressSpace::MAT;

  if (tileType) {
      if (auto attr = tileType.getMemorySpace()) {
          // 尝试转换为 PTO 的 AddressSpaceAttr
          if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(attr)) {
              space = ptoAttr.getAddressSpace();
          }
      }
  }

  // 3. 注册 Buffer 信息
  auto newMemInfo = std::make_unique<BaseMemInfo>(
      res,
      res,
      space, // 使用解析出的 space
      SmallVector<uint64_t>{baseAddr},
      sizeInBytes
  );

  buffer2MemInfoMap_[res].emplace_back(newMemInfo->clone());
  return success();
}

LogicalResult PTOIRTranslator::UpdatePointerCastOpMemInfo(pto::PointerCastOp op) {
  Value res = op.getResult();
  auto memRefType = dyn_cast<MemRefType>(res.getType());
  if (!memRefType) return failure();

  if (op.getAddrs().empty()) {
    return op.emitError("PointerCast must have at least one address operand");
  }
  Value rootSrc = op.getAddrs().front();

  uint64_t sizeInBytes = 0;
  if (memRefType.hasStaticShape()) {
    int64_t elemSize = static_cast<int64_t>(
        pto::getPTOStorageElemByteSize(memRefType.getElementType()));
    if (elemSize == 0)
      return failure();
    int64_t numElements = 1;
    for (auto dim : memRefType.getShape()) numElements *= dim;
    sizeInBytes = numElements * elemSize;
  }

  pto::AddressSpace space = pto::AddressSpace::GM;
  if (auto attr = memRefType.getMemorySpace()) {
    if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(attr)) {
      space = ptoAttr.getAddressSpace();
    }
  }

  auto newMemInfo = std::make_unique<BaseMemInfo>(
      res,
      rootSrc,
      space,
      SmallVector<uint64_t>{0},
      sizeInBytes
  );

  buffer2MemInfoMap_[res].emplace_back(newMemInfo->clone());
  return success();
}

LogicalResult
PTOIRTranslator::UpdateDeclareTileMemRefOpMemInfo(pto::DeclareTileMemRefOp op) {
  Value res = op.getResult();
  auto memRefType = dyn_cast<MemRefType>(res.getType());
  if (!memRefType)
    return failure();

  uint64_t sizeInBytes = 0;
  if (memRefType.hasStaticShape()) {
    int64_t elemSize = static_cast<int64_t>(
        pto::getPTOStorageElemByteSize(memRefType.getElementType()));
    if (elemSize == 0)
      return failure();

    int64_t numElements = 1;
    for (auto dim : memRefType.getShape())
      numElements *= dim;
    sizeInBytes = numElements * elemSize;
  }

  pto::AddressSpace space = pto::AddressSpace::MAT;
  if (auto attr = memRefType.getMemorySpace()) {
    if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(attr))
      space = ptoAttr.getAddressSpace();
  }

  // declare_tile_memref is only a symbolic placeholder. Use its SSA result as
  // both base/root so later bind_tile aliases and tpop consumers can be
  // connected by InsertSync without inventing a fake allocation.
  auto newMemInfo = std::make_unique<BaseMemInfo>(
      res,
      res,
      space,
      SmallVector<uint64_t>{0},
      sizeInBytes);

  buffer2MemInfoMap_[res].emplace_back(newMemInfo->clone());
  return success();
}

// ============================================================================
// 5. [P0 修改] 更新 PTO Op 信息 (通用接口版)
// ============================================================================
void PTOIRTranslator::UpdatePTOOpInfo(Operation *op) {
  // 1. 获取流水线类型 (现在通过 Interface)
  pto::PipelineType pipe = getOpPipeline(op);
  // 如果 Op 不属于任何关心的流水线，直接跳过，不建立 Sync 节点
  if (pipe == pto::PipelineType::PIPE_UNASSIGNED) return;

  SmallVector<const BaseMemInfo *> defVec;
  SmallVector<const BaseMemInfo *> useVec;
  // 2. [关键] 使用 MemoryEffects 接口自动获取读写依赖
  if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
    MemoryEffectVector effects;
    memEffect.getEffects(effects);
    for (auto &effect : effects) {
      Value val = effect.getValue();
      if (!val) continue;

       // 只有当 Value 在我们的 BufferMap 中有记录时，才视为有效依赖
       // (过滤掉比如 Loop Iterator 或其他标量)
       if (isa<MemoryEffects::Read>(effect.getEffect())) {
          UpdateDefUseVec({val}, useVec);
       } else if (isa<MemoryEffects::Write>(effect.getEffect())) {
          UpdateDefUseVec({val}, defVec);
       }
     }
  } else {
    // 如果算子有 Pipe 属性但没实现 MemoryEffects，这是一个定义错误
    // 我们可以打印个 Warning 或者保持为空 (认为无副作用)
    LLVM_DEBUG(llvm::dbgs() << "Warning: Op " << op->getName()
                            << " has Pipe but no MemoryEffects interface.\n");
  }

  // 3. 构建 Compound Node
  auto compoundElement = std::make_unique<CompoundInstanceElement>(
      index, defVec, useVec, pipe, op->getName());
  compoundElement->elementOp = op;

  // 4. 设置 Core Type (用于区分 Cube/Vector 资源)
  // Matmul (M) 和 L1->L0 搬运 (MTE1) 通常涉及 Cube 资源
  if (pipe == pto::PipelineType::PIPE_M || pipe == pto::PipelineType::PIPE_MTE1) {
    compoundElement->compoundCoreType = pto::TCoreType::CUBE;
  } else {
    // MTE2, MTE3, Vector 归类为 Vector Core (或者对应 MTE 资源)
    compoundElement->compoundCoreType = pto::TCoreType::VECTOR;
  }

  syncIR_.emplace_back(std::move(compoundElement));
  index++;
}

// ============================================================================
// 6. [P0 修改] 获取 Op 的 Pipeline 类型
// ============================================================================
pto::PipelineType PTOIRTranslator::getOpPipeline(Operation *op) const {
  // 1. 优先尝试通过接口获取
  if (auto pipeOp = dyn_cast<pto::OpPipeInterface>(op)) {
    // 注意：假设 pto::Pipe (ODS Enum) 和 pto::PipelineType (C++ Enum) 的数值定义是一致的
    // 或者在这里做一个 switch-case 映射
    // 目前假设直接 cast 是安全的 (0=S, 1=V, 2=M ...)
    return static_cast<pto::PipelineType>(pipeOp.getPipe());
  }
  // 2. 如果没实现接口，返回 Unassigned
  return pto::PipelineType::PIPE_UNASSIGNED;
}

// ============================================================================
// 7. 控制流处理 (SCF Support)
// ============================================================================

void PTOIRTranslator::UpdateForOpInfo(scf::ForOp forOp) {
  auto forBeginElement = std::make_unique<LoopInstanceElement>(index, index, index);
  forBeginElement->elementOp = forOp.getOperation();
  syncIR_.emplace_back(std::move(forBeginElement));

  std::unique_ptr<InstanceElement> &forElement = syncIR_[index];
  index++;

  auto *forBeginPtr = dyn_cast<LoopInstanceElement>(forElement.get());
  assert(forBeginPtr != nullptr && "Sync IR Construction failed.");

  if (!forOp.getInitArgs().empty()) {
    assert(forOp.getInitArgs().size() == forOp.getRegionIterArgs().size());
    for (auto [i, arg] : llvm::enumerate(forOp.getInitArgs())) {
      UpdateAliasBufferInfo(forOp.getRegionIterArgs()[i], arg);
    }
  }

  RecursionIR(&forOp.getRegion());

  forBeginPtr->endId = index;
  auto forEnd = forBeginPtr->CloneFor(KindOfLoop::LOOP_END);
  forEnd->elementOp = forOp.getOperation();
  syncIR_.emplace_back(std::move(forEnd));
  index++;
}

void PTOIRTranslator::UpdateWhileOpInfo(scf::WhileOp whileOp) {
  auto loopBeginElement = std::make_unique<LoopInstanceElement>(index, index, index);
  loopBeginElement->elementOp = whileOp.getOperation();
  syncIR_.emplace_back(std::move(loopBeginElement));

  auto *loopBeginPtr = dyn_cast<LoopInstanceElement>(syncIR_.back().get());
  index++;

  if (!whileOp.getInits().empty()) {
    for (auto [initArg, blockArg] : llvm::zip(whileOp.getInits(), whileOp.getBeforeArguments())) {
      UpdateAliasBufferInfo(blockArg, initArg);
    }
    auto conditionOp = whileOp.getConditionOp();
    for (auto [yieldedArg, blockArg] : llvm::zip(conditionOp.getArgs(), whileOp.getAfterArguments())) {
      UpdateAliasBufferInfo(blockArg, yieldedArg);
    }
  }

  RecursionIR(&whileOp.getBefore());
  RecursionIR(&whileOp.getAfter());

  loopBeginPtr->endId = index;
  auto forEnd = loopBeginPtr->CloneFor(KindOfLoop::LOOP_END);
  forEnd->elementOp = whileOp.getOperation();
  syncIR_.emplace_back(std::move(forEnd));
  index++;
}

void PTOIRTranslator::UpdateIfOpInfo(scf::IfOp ifOp) {
  auto ifBeginElement = std::make_unique<BranchInstanceElement>(index, index, KindOfBranch::IF_BEGIN);
  ifBeginElement->elementOp = ifOp.getOperation();
  auto *ifPtr = ifBeginElement.get();

  syncIR_.emplace_back(std::move(ifBeginElement));
  index++;

  // 1. 处理 Then 区域
  RecursionIR(&ifOp.getThenRegion());

  // Then 的结束占位符
  auto placeHolder = std::make_unique<PlaceHolderInstanceElement>(index, ifPtr->GetIndex());

  // 直接指向Then Block的yieldop
  placeHolder->elementOp = ifOp.getThenRegion().front().getTerminator();

  syncIR_.emplace_back(std::move(placeHolder));
  index++;

  ifPtr->branchId = index;

  // 2. 处理 Else 区域 (总是创建 SyncIR 节点，即使 IR 中没有 Else)
  auto ifElseElement = ifPtr->CloneBranch(KindOfBranch::ELSE_BEGIN);
  ifElseElement->elementOp = ifOp.getOperation();
  auto *elsePtr = ifElseElement.get();

  syncIR_.emplace_back(std::move(ifElseElement));
  index++;

  if (ifOp.elseBlock()) {
    RecursionIR(&ifOp.getElseRegion());
  }

  // Else 的结束占位符
  auto elsePlaceHolder = std::make_unique<PlaceHolderInstanceElement>(index, elsePtr->GetIndex());

  if (ifOp.elseBlock()) {
      // 如果有真实的 Else Block，映射到 ifOp (CodeGen 需定位到 Else Yield 前)
      elsePlaceHolder->elementOp = ifOp.getElseRegion().front().getTerminator();
      elsePlaceHolder->isVirtualElse = false;
  } else {
      // 如果没有 Else Block，标记为虚拟，映射到 ifOp
      elsePlaceHolder->elementOp = ifOp.getOperation();
      elsePlaceHolder->isVirtualElse = true;
      elsePlaceHolder->parentIfOp = ifOp.getOperation();
  }

  syncIR_.emplace_back(std::move(elsePlaceHolder));
  index++;

  elsePtr->endId = index;
  ifPtr->endId = index;

  // 3. If End
  auto ifEndElement = ifPtr->CloneBranch(KindOfBranch::IF_END);
  ifEndElement->elementOp = ifOp.getOperation();
  syncIR_.emplace_back(std::move(ifEndElement));
  index++;
}

void PTOIRTranslator::UpdateYieldOpInfo(scf::YieldOp yieldOp) {
  auto *parentOp = yieldOp->getParentOp();
  if (!parentOp || isa<scf::WhileOp>(parentOp)) return;

  assert(parentOp->getResults().size() == yieldOp->getOpOperands().size());
  for (auto [yieldVal, resultVal] : llvm::zip(yieldOp->getOpOperands(), parentOp->getResults())) {
    UpdateAliasBufferInfo(resultVal, yieldVal.get());
  }
}

// ============================================================================
// 8. 辅助函数
// ============================================================================
void PTOIRTranslator::UpdateAliasBufferInfo(Value result, Value source) {
  if (!result || !source) return;
  if (!buffer2MemInfoMap_.contains(source)) return;

  int64_t deltaOffset = 0;
  int64_t newSize = -1;

  if (auto op = result.getDefiningOp()) {
    auto info = getStaticOffsetAndSize(op, source);
    if (info.first != -1) {
        deltaOffset = info.first;
        if (info.second > 0) newSize = info.second;
    }
  }

  auto &resultMemInfoVec = buffer2MemInfoMap_[result];

  for (auto &parentInfo : buffer2MemInfoMap_[source]) {
    auto newInfo = parentInfo->clone(result);
    if (!newInfo->baseAddresses.empty()) {
        newInfo->baseAddresses[0] += deltaOffset;
    } else {
        newInfo->baseAddresses.push_back(deltaOffset);
    }

    if (newSize > 0) {
        newInfo->allocateSize = newSize;
    }

    resultMemInfoVec.emplace_back(std::move(newInfo));
  }
}

void PTOIRTranslator::UpdateConservativeAliasBufferInfo(Value result,
                                                        Value source) {
  if (!result || !source) return;
  if (!buffer2MemInfoMap_.contains(source)) return;

  auto &resultMemInfoVec = buffer2MemInfoMap_[result];
  for (auto &parentInfo : buffer2MemInfoMap_[source])
    resultMemInfoVec.emplace_back(parentInfo->clone(result));
}

bool PTOIRTranslator::CanMaterializeSubviewAliases(Value source) const {
  auto it = buffer2MemInfoMap_.find(source);
  if (it == buffer2MemInfoMap_.end())
    return false;
  return llvm::all_of(it->second, [](const std::unique_ptr<BaseMemInfo> &info) {
    return info && info->baseAddresses.size() == 1 && info->allocateSize != 0;
  });
}

void PTOIRTranslator::CloneSubviewAliasInfos(Value result, Value source,
                                             ArrayRef<uint64_t> subViewAddresses,
                                             uint64_t segmentSize) {
  auto &resultMemInfoVec = buffer2MemInfoMap_[result];
  for (auto &parentInfo : buffer2MemInfoMap_[source]) {
    auto newInfo = parentInfo->clone(result);
    SmallVector<uint64_t> addresses;
    addresses.reserve(subViewAddresses.size());
    uint64_t parentBase = parentInfo->baseAddresses[0];
    for (uint64_t offset : subViewAddresses)
      addresses.push_back(parentBase + offset);
    newInfo->baseAddresses = std::move(addresses);
    newInfo->allocateSize = segmentSize;
    resultMemInfoVec.emplace_back(std::move(newInfo));
  }
}

void PTOIRTranslator::UpdateMemrefSubViewAliasBufferInfo(memref::SubViewOp op) {
  Value result = op.getResult();
  Value source = op.getSource();
  if (!result || !source) return;
  if (!buffer2MemInfoMap_.contains(source)) return;

  auto sourceType = dyn_cast<MemRefType>(source.getType());
  if (!sourceType) {
    UpdateConservativeAliasBufferInfo(result, source);
    return;
  }

  unsigned elemBytes = pto::getPTOStorageElemByteSize(sourceType.getElementType());
  auto subViewAddresses =
      elemBytes == 0 ? std::nullopt
                     : getMemrefSubViewBaseAddresses(
                           op, sourceType, static_cast<int64_t>(elemBytes));
  if (!subViewAddresses || subViewAddresses->empty()) {
    UpdateConservativeAliasBufferInfo(result, source);
    return;
  }

  auto getSegmentSize =
      [&op, sourceType, elemBytes]() -> std::optional<uint64_t> {
    SmallVector<int64_t> strides;
    int64_t baseOffset = ShapedType::kDynamic;
    if (failed(mlir::getStridesAndOffset(sourceType, strides, baseOffset)) ||
        strides.size() != kTileRank2D) {
      return std::nullopt;
    }
    ArrayRef<int64_t> staticSizes = op.getStaticSizes();
    if (strides[1] == 1) {
      return static_cast<uint64_t>(
          staticSizes[1] * static_cast<int64_t>(elemBytes));
    }
    if (strides[0] == 1) {
      return static_cast<uint64_t>(
          staticSizes[0] * static_cast<int64_t>(elemBytes));
    }
    return std::nullopt;
  };

  std::optional<uint64_t> segmentSize = getSegmentSize();
  if (!segmentSize || !CanMaterializeSubviewAliases(source)) {
    UpdateConservativeAliasBufferInfo(result, source);
    return;
  }

  CloneSubviewAliasInfos(result, source, *subViewAddresses, *segmentSize);
}

void PTOIRTranslator::UpdateTileSubViewAliasBufferInfo(pto::SubViewOp op) {
  Value result = op.getResult();
  Value source = op.getSource();
  if (!result || !source) return;
  if (!buffer2MemInfoMap_.contains(source)) return;

  auto sourceType = dyn_cast<pto::TileBufType>(source.getType());
  if (!sourceType) {
    UpdateConservativeAliasBufferInfo(result, source);
    return;
  }

  unsigned elemBytes = pto::getPTOStorageElemByteSize(sourceType.getElementType());
  auto subViewAddresses =
      elemBytes == 0 ? std::nullopt
                     : getPtoSubViewBaseAddresses(
                           op, sourceType, static_cast<int64_t>(elemBytes));
  if (!subViewAddresses || subViewAddresses->empty()) {
    UpdateConservativeAliasBufferInfo(result, source);
    return;
  }

  auto sizesAttr = op.getSizes();
  int64_t rowSize = cast<IntegerAttr>(sizesAttr[0]).getInt();
  int64_t colSize = cast<IntegerAttr>(sizesAttr[1]).getInt();
  bool rowMajor =
      sourceType.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
  uint64_t segmentSize =
      static_cast<uint64_t>((rowMajor ? colSize : rowSize) *
                            static_cast<int64_t>(elemBytes));

  for (auto &parentInfo : buffer2MemInfoMap_[source]) {
    if (!parentInfo || parentInfo->baseAddresses.size() != 1 ||
        parentInfo->allocateSize == 0) {
      UpdateConservativeAliasBufferInfo(result, source);
      return;
    }
  }

  auto &resultMemInfoVec = buffer2MemInfoMap_[result];
  for (auto &parentInfo : buffer2MemInfoMap_[source]) {
    auto newInfo = parentInfo->clone(result);
    SmallVector<uint64_t> addresses;
    addresses.reserve(subViewAddresses->size());
    uint64_t parentBase = parentInfo->baseAddresses[0];
    for (uint64_t offset : *subViewAddresses)
      addresses.push_back(parentBase + offset);
    newInfo->baseAddresses = std::move(addresses);
    newInfo->allocateSize = segmentSize;
    resultMemInfoVec.emplace_back(std::move(newInfo));
  }
}

// ============================================================================
// 实现 UpdateMemrefAllocOpMemInfo
// ============================================================================
LogicalResult PTOIRTranslator::UpdateMemrefAllocOpMemInfo(memref::AllocOp op) {
  Value res = op.getResult();
  auto memRefType = dyn_cast<MemRefType>(res.getType());
  if (!memRefType) return failure();

  // 1. 计算大小 (Bytes)
  uint64_t sizeInBytes = 0;
  if (memRefType.hasStaticShape()) {
    int64_t elemSize = static_cast<int64_t>(
        pto::getPTOStorageElemByteSize(memRefType.getElementType()));
    if (elemSize == 0)
      return failure();

    int64_t numElements = 1;
    for (auto dim : memRefType.getShape()) numElements *= dim;
    sizeInBytes = numElements * elemSize;
  }

  // 2. 解析地址空间 (Scope)
  // 默认视为 MAT/UB (Local Memory)，这是 alloc 的常见用途
  // 如果有显式属性，则覆盖
  pto::AddressSpace space = pto::AddressSpace::MAT;

  if (auto attr = memRefType.getMemorySpace()) {
    if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(attr)) {
      space = ptoAttr.getAddressSpace();
    }
  }

  // 3. 注册 Buffer 信息
  // 对于 alloc，它自己就是 Root
  auto newMemInfo = std::make_unique<BaseMemInfo>(
      res,                      // baseBuffer
      res,                      // rootBuffer (Self is root)
      space,
      SmallVector<uint64_t>{0}, // Base Addresses (Offset 0)
      sizeInBytes
  );

  buffer2MemInfoMap_[res].emplace_back(newMemInfo->clone());
  return success();
}

void PTOIRTranslator::UpdateDefUseVec(ValueRange values, SmallVector<const BaseMemInfo *> &vec) {
  for (Value v : values) {
    if (buffer2MemInfoMap_.contains(v)) {
      for (auto &memInfo : buffer2MemInfoMap_[v]) {
        vec.push_back(memInfo.get());
      }
    }
  }
}

// ============================================================================
// 9. 调试与打印支持
// ============================================================================

std::string PTOIRTranslator::getPipelineName(pto::PipelineType pipe) const {
  switch (pipe) {
  case pto::PipelineType::PIPE_MTE1: return "MTE1";
  case pto::PipelineType::PIPE_MTE2: return "MTE2";
  case pto::PipelineType::PIPE_MTE3: return "MTE3";
  case pto::PipelineType::PIPE_M:    return "CUBE";
  case pto::PipelineType::PIPE_V:    return "VECTOR";
  case pto::PipelineType::PIPE_S:    return "SCALAR";
  case pto::PipelineType::PIPE_ALL:  return "BARRIER";
  default: return "UNKNOWN";
  }
}

void PTOIRTranslator::printMemInfoList(llvm::raw_ostream &os,
                                       const SmallVector<const BaseMemInfo *> &list,
                                       AsmState &state) const {
  os << "[";
  bool first = true;
  for (const auto *info : list) {
    if (!first) os << ", ";
    info->rootBuffer.printAsOperand(os, state);
    // [Fix] 打印 MAT 或 VEC 或 GM
    if (info->scope == pto::AddressSpace::GM) os << "(GM)";
    else if (info->scope == pto::AddressSpace::MAT) os << "(MAT)";
    else if (info->scope == pto::AddressSpace::VEC) os << "(VEC)";
    else os << "(Other)"; // 处理 LEFT/RIGHT/ACC 等其他情况
    first = false;
  }
  os << "]";
}

void PTOIRTranslator::print() {
  llvm::errs() << "\n=== PTO IR Translator Dump ===\n";

  AsmState state(func_);

  llvm::errs() << "--- Buffer Analysis (Value -> Root) ---\n";
  for (auto &it : buffer2MemInfoMap_) {
    Value v = it.first;
    auto &infoList = it.second;

    llvm::errs() << "  ";
    v.printAsOperand(llvm::errs(), state);
    llvm::errs() << " -> ";

    for (auto &mem : infoList) {
        mem->rootBuffer.printAsOperand(llvm::errs(), state);
        llvm::errs() << " ";
    }
    llvm::errs() << "\n";
  }

  llvm::errs() << "\n--- SyncIR Structure ---\n";
  for (const auto &element : syncIR_) {
    unsigned id = element->GetIndex();
    llvm::errs() << llvm::formatv("{0,4}: ", id);

    switch (element->GetKind()) {
    case InstanceElement::KindTy::COMPOUND: {
      auto *comp = dyn_cast<CompoundInstanceElement>(element.get());
      llvm::errs() << "COMPOUND [" << getPipelineName(comp->kPipeValue) << "] ";
      llvm::errs() << comp->opName.getStringRef() << "\n";

      llvm::errs() << "      DEF: ";
      printMemInfoList(llvm::errs(), comp->defVec, state);
      llvm::errs() << "\n      USE: ";
      printMemInfoList(llvm::errs(), comp->useVec, state);
      llvm::errs() << "\n";
      break;
    }
    case InstanceElement::KindTy::LOOP:
        llvm::errs() << "LOOP\n"; break;
    case InstanceElement::KindTy::BRANCH:
        llvm::errs() << "BRANCH\n"; break;
    case InstanceElement::KindTy::PLACE_HOLDER:
        llvm::errs() << "PLACE_HOLDER\n"; break;
    }
  }
  llvm::errs() << "==============================\n\n";
}
