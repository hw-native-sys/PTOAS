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
#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
// [P0 新增] 引入副作用接口和 PTO 接口
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <optional>

#define DEBUG_TYPE "pto-ir-translator"

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr size_t kTileRank2D = 2;
constexpr unsigned kMemoryEffectInlineCapacity = 4;
constexpr llvm::StringLiteral kTileOpEffectsAttr = "pto.tileop.effects";

using MemoryEffectVector =
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>,
                kMemoryEffectInlineCapacity>;

static uint64_t getStaticBufferSizeInBytes(ArrayRef<int64_t> shape,
                                           Type elementType) {
  uint64_t size = pto::getPTOStorageElemByteSize(elementType);
  if (size == 0) {
    return 0;
  }
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic) {
      return 0;
    }
    size *= static_cast<uint64_t>(dim);
  }
  return size;
}

static uint64_t getTileBufferFootprintBytes(pto::TileBufType type) {
  ArrayRef<int64_t> shape = type.getShape();
  uint64_t elemBytes = pto::getPTOStorageElemByteSize(type.getElementType());
  if (elemBytes == 0) {
    return 0;
  }
  if (type.getCompactModeI32() !=
      static_cast<int32_t>(pto::CompactMode::RowPlusOne)) {
    return getStaticBufferSizeInBytes(shape, type.getElementType());
  }
  if (shape.size() != kTileRank2D ||
      llvm::is_contained(shape, ShapedType::kDynamic)) {
    return 0;
  }

  bool rowMajor =
      type.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
  uint64_t major = static_cast<uint64_t>(rowMajor ? shape[0] : shape[1]);
  uint64_t minor = static_cast<uint64_t>(rowMajor ? shape[1] : shape[0]);
  if (major == 0 || minor == 0) {
    return 0;
  }
  return ((major - 1) * (minor + 1) + minor) * elemBytes;
}

static pto::AddressSpace getTileAddressSpace(pto::TileBufType type) {
  if (auto attr =
          dyn_cast_or_null<pto::AddressSpaceAttr>(type.getMemorySpace())) {
    return attr.getAddressSpace();
  }
  return pto::AddressSpace::MAT;
}

static void
appendUniqueMemInfo(SmallVectorImpl<std::unique_ptr<BaseMemInfo>> &infos,
                    std::unique_ptr<BaseMemInfo> candidate) {
  if (!candidate) {
    return;
  }
  if (llvm::any_of(infos, [&](const std::unique_ptr<BaseMemInfo> &info) {
        return info && *info == *candidate;
      })) {
    return;
  }
  infos.emplace_back(std::move(candidate));
}

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
  if (!intAttr) {
    return false;
  }
  out = intAttr.getInt();
  return true;
}

static std::optional<uint64_t> getKnownPhysicalAddress(Value value) {
  int64_t address = 0;
  if (!getConstIndexValue(value, address) || address < 0) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(address);
}

static bool isLocalAddressSpace(pto::AddressSpace space) {
  return space != pto::AddressSpace::GM &&
         space != pto::AddressSpace::Zero;
}

static pto::AddressSpace getPointerLikeAddressSpace(Type type) {
  if (auto space = pto::getPTOAddressSpaceAttr(type)) {
    return space.getAddressSpace();
  }
  return pto::AddressSpace::GM;
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
  if (type.getCompactModeI32() == static_cast<int32_t>(pto::CompactMode::RowPlusOne)) {
    ++stride;
  }
  return stride;
}

static std::optional<SmallVector<uint64_t>>
getPtoSubViewBaseAddresses(pto::SubViewOp op, pto::TileBufType sourceType,
                           int64_t elemBytes) {
  if (!isStaticRank2Shape(sourceType.getShape())) {
    return std::nullopt;
  }
  if (sourceType.getSLayoutValueI32() !=
      static_cast<int32_t>(pto::SLayout::NoneBox)) {
    return std::nullopt;
  }
  if (op.getOffsets().size() != kTileRank2D) {
    return std::nullopt;
  }

  int64_t rowOffset = 0;
  int64_t colOffset = 0;
  if (!getConstIndexValue(op.getOffsets()[0], rowOffset) ||
      !getConstIndexValue(op.getOffsets()[1], colOffset)) {
    return std::nullopt;
  }
  if (rowOffset < 0 || colOffset < 0) {
    return std::nullopt;
  }

  auto sizesAttr = op.getSizes();
  if (!sizesAttr || sizesAttr.size() != kTileRank2D) {
    return std::nullopt;
  }
  int64_t rowSize = cast<IntegerAttr>(sizesAttr[0]).getInt();
  int64_t colSize = cast<IntegerAttr>(sizesAttr[1]).getInt();
  if (rowSize <= 0 || colSize <= 0) {
    return std::nullopt;
  }

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

namespace {

static func::FuncOp lookupSyncHelper(func::CallOp callOp) {
  auto module = callOp->getParentOfType<ModuleOp>();
  if (!module || callOp.getCallee().empty()) {
    return {};
  }
  auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
  if (!callee) {
    return {};
  }
  if (!callee->hasAttr("pto.tileop.helper") &&
      !callee->hasAttr("pto.ptodsl.subkernel_helper")) {
    return {};
  }
  return callee;
}

static std::optional<pto::PipelineType> getSyncHelperPipe(func::FuncOp callee) {
  if (auto roleAttr = callee->getAttrOfType<mlir::StringAttr>(
          "pto.ptodsl.subkernel_helper")) {
    return llvm::StringSwitch<std::optional<pto::PipelineType>>(
               roleAttr.getValue())
        .Case("cube", pto::PipelineType::PIPE_M)
        .Case("simd", pto::PipelineType::PIPE_V)
        .Default(std::nullopt);
  }

  auto kindAttr = callee->getAttrOfType<mlir::StringAttr>("pto.tileop.kind");
  if (!kindAttr) {
    return std::nullopt;
  }
  return llvm::StringSwitch<std::optional<pto::PipelineType>>(
             kindAttr.getValue())
      .Case("cube", pto::PipelineType::PIPE_M)
      .Case("vector", pto::PipelineType::PIPE_V)
      .Default(std::nullopt);
}

static bool isSyncHelperMemoryOperand(Type type) {
  return isa<pto::PtrType, pto::TileBufType, pto::TensorViewType,
             pto::PartitionTensorViewType>(type);
}

static pto::TCoreType getSyncHelperCoreType(pto::PipelineType pipe) {
  return pipe == pto::PipelineType::PIPE_M ? pto::TCoreType::CUBE
                                           : pto::TCoreType::VECTOR;
}

} // namespace

// ============================================================================
// 1. 构建入口
// ============================================================================
void PTOIRTranslator::Build() {
  Region &funcRegion = func_.getBody();
  UpdateKernelArgMemInfo();
  RecursionIR(&funcRegion);
}

// ============================================================================
// 2. 更新 Kernel 参数内存信息
// ============================================================================
void PTOIRTranslator::UpdateKernelArgMemInfo() {
  for (Value funcArg : func_.getArguments()) {
    Type argType = funcArg.getType();
    pto::AddressSpace space = pto::AddressSpace::Zero;
    uint64_t sizeInBytes = 0;
    SmallVector<uint64_t> baseAddresses{0};

    if (auto ptrType = dyn_cast<pto::PtrType>(argType)) {
      space = ptrType.getMemorySpace().getAddressSpace();
    } else if (auto tileType = dyn_cast<pto::TileBufType>(argType)) {
      space = getTileAddressSpace(tileType);
      sizeInBytes = getTileBufferFootprintBytes(tileType);
    } else if (auto multiType = dyn_cast<pto::MultiTileBufType>(argType)) {
      pto::TileBufType slotType = multiType.getSlotType();
      space = getTileAddressSpace(slotType);
      sizeInBytes = getTileBufferFootprintBytes(slotType);
      baseAddresses.clear();
      for (uint32_t slot = 0; slot < multiType.getCount(); ++slot) {
        baseAddresses.push_back(sizeInBytes == 0 ? 0 : slot * sizeInBytes);
      }
    } else if (auto viewType = dyn_cast<pto::TensorViewType>(argType)) {
      space = pto::AddressSpace::GM;
      sizeInBytes = getStaticBufferSizeInBytes(viewType.getShape(),
                                               viewType.getElementType());
    } else if (auto viewType =
                   dyn_cast<pto::PartitionTensorViewType>(argType)) {
      space = pto::AddressSpace::GM;
      sizeInBytes = getStaticBufferSizeInBytes(viewType.getShape(),
                                               viewType.getElementType());
    } else {
      continue;
    }

    buffer2MemInfoMap_[funcArg].emplace_back(std::make_unique<BaseMemInfo>(
        funcArg, funcArg, space, std::move(baseAddresses), sizeInBytes));
  }
}

// ============================================================================
// 3. 递归遍历 IR (核心分发逻辑)
// ============================================================================
void PTOIRTranslator::RecursionIR(Region *region) {
  auto result = region->walk<WalkOrder::PreOrder>([&](Operation *op) {
    // --- Case A: 内存分配 (AllocTile) ---
    if (auto allocOp = dyn_cast<pto::AllocTileOp>(op)) {
      if (failed(UpdateAllocTileOpMemInfo(allocOp))) {
        return WalkResult::interrupt();
      }
    }
    else if (auto allocMultiOp = dyn_cast<pto::AllocMultiTileOp>(op)) {
      if (failed(UpdateAllocMultiTileOpMemInfo(allocMultiOp))) {
        return WalkResult::interrupt();
      }
    }
    else if (auto declareOp = dyn_cast<pto::DeclareTileOp>(op)) {
      if (failed(UpdateDeclareTileOpMemInfo(declareOp))) {
        return WalkResult::interrupt();
      }
    }
    else if (auto declareGlobalOp = dyn_cast<pto::DeclareGlobalOp>(op)) {
      if (failed(UpdateDeclareGlobalOpMemInfo(declareGlobalOp))) {
        return WalkResult::interrupt();
      }
    }
    // --- Case B: 别名/视图操作 ---
    else if (auto makeViewOp = dyn_cast<pto::MakeTensorViewOp>(op)) {
      UpdateAliasBufferInfo(makeViewOp.getResult(), makeViewOp.getPtr());
    }
    else if (auto subViewOp = dyn_cast<pto::PartitionViewOp>(op)) {
      UpdateAliasBufferInfo(subViewOp.getResult(), subViewOp.getSource());
    }
    else if (auto addPtrOp = dyn_cast<pto::AddPtrOp>(op)) {
      UpdateAliasBufferInfo(addPtrOp.getResult(), addPtrOp.getPtr());
    }
    else if (auto ptrToIntOp = dyn_cast<pto::PtrToIntOp>(op)) {
      UpdateAliasBufferInfo(ptrToIntOp.getResult(), ptrToIntOp.getPtr());
    }
    else if (auto intToPtrOp = dyn_cast<pto::IntToPtrOp>(op)) {
      if (failed(UpdateIntToPtrOpMemInfo(intToPtrOp))) {
        return WalkResult::interrupt();
      }
    }
    else if (auto castPtrOp = dyn_cast<pto::CastPtrOp>(op)) {
      if (isa<pto::PtrType>(castPtrOp.getInput().getType()) &&
          isa<pto::PtrType>(castPtrOp.getResult().getType())) {
        UpdateAliasBufferInfo(castPtrOp.getResult(), castPtrOp.getInput());
      }
    }
    else if (auto subViewOp = dyn_cast<pto::SubViewOp>(op)) {
      UpdateTileSubViewAliasBufferInfo(subViewOp);
    } else if (auto multiGet = dyn_cast<pto::MultiTileGetOp>(op)) {
      UpdateMultiTileGetAliasBufferInfo(multiGet);
    }
    else if (auto reshape = dyn_cast<pto::TReshapeOp>(op)) {
      UpdateAliasBufferInfo(reshape.getResult(), reshape.getSrc());
    }
    else if (auto bitcast = dyn_cast<pto::BitcastOp>(op)) {
      UpdateAliasBufferInfo(bitcast.getResult(), bitcast.getSrc());
    }
    else if (auto select = dyn_cast<arith::SelectOp>(op)) {
      UpdateAliasBufferInfo(select.getResult(), select.getTrueValue());
      UpdateAliasBufferInfo(select.getResult(), select.getFalseValue());
    }

    // --- Case C: 控制流 (SCF) ---
    else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      UpdateForOpInfo(forOp);
      return WalkResult::skip();
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      UpdateWhileOpInfo(whileOp);
      return WalkResult::skip();
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      UpdateIfOpInfo(ifOp);
      return WalkResult::skip();
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      UpdateYieldOpInfo(yieldOp);
    } else if (getSyncMacroModel(op)) {
      UpdateMacroOpInfo(op);
    } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
      UpdateHelperCallInfo(callOp);
    } else if (isa<pto::LoadScalarOp, pto::StoreScalarOp>(op)) {
      // Scalar GM pointer accesses do not implement OpPipeInterface, but they
      // execute on PIPE_S and can race with async MTE/FIX tile stores touching
      // the same GM payload.
      UpdatePTOOpInfoWithPipeline(op, pto::PipelineType::PIPE_S,
                                  /*skipIfNoMemInfo=*/true);
    } else if (isa<pto::OpPipeInterface>(op)) {
      // --- Case D: 带有 OpPipeInterface 的计算/搬运指令 ---
      UpdatePTOOpInfo(op);
    }
    return WalkResult::advance();
  });
  if (result == WalkResult::interrupt()) {
    llvm_unreachable("PTO InjectSync Traverse IR Failed!");
  }
}

// ============================================================================
// 4. 处理本地 tile 分配
// ============================================================================
LogicalResult PTOIRTranslator::UpdateAllocTileOpMemInfo(pto::AllocTileOp op) {
  Value res = op.getResult();

  auto tileType = dyn_cast<pto::TileBufType>(res.getType());
  uint64_t sizeInBytes = 0;
  uint64_t baseAddr = 0;
  std::optional<uint64_t> knownPhysicalAddress;

  // If alloc_tile carries an explicit address, record it when it's a constant.
  if (Value addr = op.getAddr()) {
    knownPhysicalAddress = getKnownPhysicalAddress(addr);
    if (knownPhysicalAddress) {
      baseAddr = *knownPhysicalAddress;
    }
  }

  // 1. 计算大小
  if (tileType) {
    sizeInBytes = getTileBufferFootprintBytes(tileType);
    if (sizeInBytes == 0) {
      return failure();
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
      res, res, space, SmallVector<uint64_t>{baseAddr}, sizeInBytes,
      knownPhysicalAddress.has_value() && isLocalAddressSpace(space));

  buffer2MemInfoMap_[res].emplace_back(newMemInfo->clone());
  return success();
}

LogicalResult
PTOIRTranslator::UpdateAllocMultiTileOpMemInfo(pto::AllocMultiTileOp op) {
  Value result = op.getResult();
  auto multiType = op.getResult().getType();
  pto::TileBufType slotType = multiType.getSlotType();

  uint64_t slotBytes = getTileBufferFootprintBytes(slotType);
  if (slotBytes == 0) {
    return failure();
  }

  pto::AddressSpace space = pto::AddressSpace::MAT;
  if (auto attr = dyn_cast_or_null<pto::AddressSpaceAttr>(
          slotType.getMemorySpace())) {
    space = attr.getAddressSpace();
  }

  SmallVector<uint64_t> addresses;
  bool hasKnownAddresses = false;
  if (auto planned = op->getAttrOfType<DenseI64ArrayAttr>(
          pto::kPtoMultiBufferAddrsAttrName)) {
    if (planned.size() != multiType.getCount()) {
      return op.emitError("planned address count does not match slot count");
    }
    for (int64_t address : planned.asArrayRef()) {
      addresses.push_back(static_cast<uint64_t>(address));
    }
    hasKnownAddresses = true;
  } else if (Value base = op.getAddr()) {
    if (std::optional<uint64_t> knownBase = getKnownPhysicalAddress(base)) {
      for (uint32_t slot = 0; slot < multiType.getCount(); ++slot) {
        addresses.push_back(*knownBase + slot * slotBytes);
      }
      hasKnownAddresses = true;
    }
  }

  if (addresses.empty()) {
    return op.emitError(
        "requires planner-assigned slot addresses or a constant level3 base");
  }

  auto info = std::make_unique<BaseMemInfo>(
      result, result, space, std::move(addresses), slotBytes,
      hasKnownAddresses && isLocalAddressSpace(space));
  buffer2MemInfoMap_[result].emplace_back(info->clone());
  return success();
}

LogicalResult
PTOIRTranslator::UpdateDeclareTileOpMemInfo(pto::DeclareTileOp op) {
  Value result = op.getTile();
  auto tileType = dyn_cast<pto::TileBufType>(result.getType());
  if (!tileType) {
    return op.emitError("requires a tile_buf result for sync analysis");
  }

  uint64_t sizeInBytes = getTileBufferFootprintBytes(tileType);
  if (sizeInBytes == 0) {
    return failure();
  }

  pto::AddressSpace space = pto::AddressSpace::MAT;
  if (auto attr = dyn_cast_or_null<pto::AddressSpaceAttr>(
          tileType.getMemorySpace())) {
    space = attr.getAddressSpace();
  }

  auto info = std::make_unique<BaseMemInfo>(
      result, result, space, SmallVector<uint64_t>{0}, sizeInBytes);
  buffer2MemInfoMap_[result].emplace_back(info->clone());
  return success();
}

LogicalResult
PTOIRTranslator::UpdateDeclareGlobalOpMemInfo(pto::DeclareGlobalOp op) {
  Value res = op.getEntry();
  auto tensorViewType = dyn_cast<pto::TensorViewType>(res.getType());
  if (!tensorViewType) {
    return failure();
  }

  uint64_t sizeInBytes = 0;
  ArrayRef<int64_t> shape = tensorViewType.getShape();
  bool isStatic = llvm::all_of(shape, [](int64_t dim) {
    return dim != ShapedType::kDynamic;
  });
  if (isStatic) {
    int64_t elemSize = static_cast<int64_t>(
        pto::getPTOStorageElemByteSize(tensorViewType.getElementType()));
    if (elemSize == 0) {
      return failure();
    }
    int64_t numElements = 1;
    for (int64_t dim : shape) {
      numElements *= dim;
    }
    sizeInBytes = static_cast<uint64_t>(numElements * elemSize);
  }

  // declare_global is a symbolic GM entry placeholder whose address gets bound
  // later by talloc/tpop. Using the SSA result as both base/root lets
  // partition_view/tstore/tload and tpush/tpop share one alias chain so
  // InsertSync can recover the GM-slot handshake ordering.
  auto newMemInfo = std::make_unique<BaseMemInfo>(
      res,
      res,
      pto::AddressSpace::GM,
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
  UpdatePTOOpInfoWithPipeline(op, getOpPipeline(op));
}

void PTOIRTranslator::UpdatePTOOpInfoWithPipeline(Operation *op,
                                                  pto::PipelineType pipe,
                                                  bool skipIfNoMemInfo) {
  // 如果 Op 不属于任何关心的流水线，直接跳过，不建立 Sync 节点
  if (pipe == pto::PipelineType::PIPE_UNASSIGNED) {
    return;
  }

  SmallVector<const BaseMemInfo *> defVec;
  SmallVector<const BaseMemInfo *> useVec;
  // 2. [关键] 使用 MemoryEffects 接口自动获取读写依赖
  if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
    MemoryEffectVector effects;
    memEffect.getEffects(effects);
    for (auto &effect : effects) {
      Value val = effect.getValue();
      if (!val) {
        continue;
      }

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

  if (skipIfNoMemInfo && defVec.empty() && useVec.empty()) {
    return;
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

void PTOIRTranslator::MakeMacroCompound(Operation *op, PipelineType pipe,
                                        ValueRange defValues,
                                        ValueRange useValues,
                                        int macroPhaseId) {
  SmallVector<const BaseMemInfo *> defVec;
  SmallVector<const BaseMemInfo *> useVec;
  UpdateDefUseVec(defValues, defVec);
  UpdateDefUseVec(useValues, useVec);

  auto compoundElement = std::make_unique<CompoundInstanceElement>(
      index, std::move(defVec), std::move(useVec), pipe, op->getName());
  compoundElement->elementOp = op;
  compoundElement->macroOpInstanceId = macroPhaseId;
  compoundElement->compoundCoreType =
      pipe == PipelineType::PIPE_M ? pto::TCoreType::CUBE
                                   : pto::TCoreType::VECTOR;
  syncIR_.emplace_back(std::move(compoundElement));
  index++;
}

void PTOIRTranslator::UpdateMacroOpInfo(Operation *op) {
  auto model = getSyncMacroModel(op);
  if (!model) {
    return;
  }
  for (const auto &phase : model->phases) {
    MakeMacroCompound(op, phase.pipe, ValueRange(phase.defValues),
                      ValueRange(phase.useValues), phase.phaseId);
  }
}

void PTOIRTranslator::UpdateHelperCallInfo(func::CallOp callOp) {
  func::FuncOp callee = lookupSyncHelper(callOp);
  if (!callee) {
    return;
  }

  std::optional<pto::PipelineType> pipe = getSyncHelperPipe(callee);
  if (!pipe || *pipe == pto::PipelineType::PIPE_UNASSIGNED) {
    return;
  }

  SmallVector<const BaseMemInfo *> defVec;
  SmallVector<const BaseMemInfo *> useVec;
  auto effects = callee->getAttrOfType<ArrayAttr>(kTileOpEffectsAttr);
  bool hasPreciseEffects = effects && effects.size() == callOp.getNumOperands();
  for (auto [operandIndex, operand] : llvm::enumerate(callOp.getOperands())) {
    if (!isSyncHelperMemoryOperand(operand.getType())) {
      continue;
    }

    StringRef effect = "readwrite";
    if (hasPreciseEffects) {
      if (auto effectAttr = dyn_cast<StringAttr>(effects[operandIndex])) {
        effect = effectAttr.getValue();
      }
    }
    if (effect == "read" || effect == "readwrite") {
      UpdateDefUseVec({operand}, useVec);
    }
    if (effect == "write" || effect == "readwrite") {
      UpdateDefUseVec({operand}, defVec);
    }
  }

  if (defVec.empty() && useVec.empty()) {
    return;
  }

  auto compoundElement = std::make_unique<CompoundInstanceElement>(
      index, defVec, useVec, *pipe, callOp->getName());
  compoundElement->elementOp = callOp;
  compoundElement->compoundCoreType = getSyncHelperCoreType(*pipe);
  syncIR_.emplace_back(std::move(compoundElement));
  index++;
}

// ============================================================================
// 6. [P0 修改] 获取 Op 的 Pipeline 类型
// ============================================================================
pto::PipelineType PTOIRTranslator::getOpPipeline(Operation *op) {
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
  if (!parentOp || isa<scf::WhileOp>(parentOp)) {
    return;
  }

  assert(parentOp->getResults().size() == yieldOp->getOpOperands().size());
  for (auto [yieldVal, resultVal] : llvm::zip(yieldOp->getOpOperands(), parentOp->getResults())) {
    UpdateAliasBufferInfo(resultVal, yieldVal.get());
  }
}

// ============================================================================
// 8. 辅助函数
// ============================================================================
void PTOIRTranslator::UpdateAliasBufferInfo(Value result, Value source) {
  if (!result || !source) {
    return;
  }
  if (!buffer2MemInfoMap_.contains(source)) {
    return;
  }

  auto &resultMemInfoVec = buffer2MemInfoMap_[result];
  for (auto &parentInfo : buffer2MemInfoMap_[source]) {
    appendUniqueMemInfo(resultMemInfoVec, parentInfo->clone(result));
  }
}

void PTOIRTranslator::UpdateConservativeAliasBufferInfo(Value result,
                                                        Value source) {
  UpdateAliasBufferInfo(result, source);
}

LogicalResult PTOIRTranslator::UpdateIntToPtrOpMemInfo(pto::IntToPtrOp op) {
  Value result = op.getResult();
  if (!result) {
    return failure();
  }

  // Preserve provenance across the explicit byte-address round trip:
  //   %addr = pto.ptrtoint %ptr
  //   %ptr2 = pto.inttoptr %addr
  // `ptr2` is the same address as `ptr`, possibly with a different element
  // type, so scalar memory ops must participate in the same GM dependency
  // chain as tile stores/loads through the original pointer.
  if (auto ptrToInt = op.getAddr().getDefiningOp<pto::PtrToIntOp>()) {
    UpdateConservativeAliasBufferInfo(result, ptrToInt.getPtr());
    if (buffer2MemInfoMap_.contains(result)) {
      return success();
    }
  }

  // A raw integer address may still point at any object in its address space.
  // Keep it in the sync IR instead of letting scalar ops be dropped by
  // skipIfNoMemInfo=true.
  auto space = getPointerLikeAddressSpace(result.getType());
  auto newMemInfo = std::make_unique<BaseMemInfo>(
      result, result, space, SmallVector<uint64_t>{0},
      /*allocateSize=*/0, /*hasKnownPhysicalAddresses=*/false,
      /*aliasesUnknownRange=*/true);
  buffer2MemInfoMap_[result].emplace_back(newMemInfo->clone());
  return success();
}

void PTOIRTranslator::UpdateMultiTileGetAliasBufferInfo(
    pto::MultiTileGetOp op) {
  UpdateSlotSelectedAliasBufferInfo(op.getResult(), op.getSource(),
                                    op.getSlot());
}

void PTOIRTranslator::UpdateSlotSelectedAliasBufferInfo(Value result,
                                                        Value source,
                                                        Value slot) {
  if (!result || !source) {
    return;
  }
  if (!buffer2MemInfoMap_.contains(source)) {
    return;
  }

  IntegerAttr constAttr;
  bool isConstSlot = matchPattern(slot, m_Constant(&constAttr));
  int64_t constSlotIdx = isConstSlot ? constAttr.getValue().getSExtValue() : -1;

  auto &resultMemInfoVec = buffer2MemInfoMap_[result];
  for (auto &parentInfo : buffer2MemInfoMap_[source]) {
    auto newInfo = parentInfo->clone(result);
    // Multi-buffer parent: `baseAddresses` lists every physical slot's
    // planner-assigned offset. For a constant slot index, narrow it to just
    // that one slot so MemAlias's range-overlap check returns false when
    // two const-slot uses pick different slots. For a dynamic slot index,
    // keep all addresses -- the runtime SSA could resolve to any slot, so
    // sync must conservatively treat the use as touching all of them.
    if (isConstSlot && constSlotIdx >= 0 &&
        constSlotIdx < static_cast<int64_t>(newInfo->baseAddresses.size()) &&
        newInfo->baseAddresses.size() > 1) {
      uint64_t pickAddr =
          newInfo->baseAddresses[static_cast<size_t>(constSlotIdx)];
      newInfo->baseAddresses.clear();
      newInfo->baseAddresses.push_back(pickAddr);
    }
    resultMemInfoVec.emplace_back(std::move(newInfo));
  }
}

void PTOIRTranslator::UpdateTileSubViewAliasBufferInfo(pto::SubViewOp op) {
  Value result = op.getResult();
  Value source = op.getSource();
  if (!result || !source) {
    return;
  }
  if (!buffer2MemInfoMap_.contains(source)) {
    return;
  }

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
    for (uint64_t offset : *subViewAddresses) {
      addresses.push_back(parentBase + offset);
    }
    newInfo->baseAddresses = std::move(addresses);
    newInfo->allocateSize = segmentSize;
    resultMemInfoVec.emplace_back(std::move(newInfo));
  }
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

std::string PTOIRTranslator::getPipelineName(pto::PipelineType pipe) {
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
                                       AsmState &state) {
  os << "[";
  bool first = true;
  for (const auto *info : list) {
    if (!first) {
      os << ", ";
    }
    info->rootBuffer.printAsOperand(os, state);
    // [Fix] 打印 MAT 或 VEC 或 GM
    if (info->scope == pto::AddressSpace::GM) {
      os << "(GM)";
    } else if (info->scope == pto::AddressSpace::MAT) {
      os << "(MAT)";
    } else if (info->scope == pto::AddressSpace::VEC) {
      os << "(VEC)";
    } else {
      os << "(Other)"; // 处理 LEFT/RIGHT/ACC 等其他情况
    }
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
