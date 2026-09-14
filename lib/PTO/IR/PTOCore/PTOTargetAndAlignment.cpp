// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

#define GET_ATTRDEF_CLASSES
#include "PTO/IR/PTOAttrs.cpp.inc"

#include "PTO/IR/PTODialect.cpp.inc"

[[maybe_unused]] static LogicalResult parseShapeAndElemStable(mlir::AsmParser &parser,
                                             llvm::SmallVectorImpl<int64_t> &shape,
                                             mlir::Type &elementType) {
  if (failed(parser.parseLess())) {
    return failure();
  }

  if (failed(parser.parseDimensionList(shape, /*allowDynamic=*/true))) {
    return failure();
  }

  if (failed(parser.parseType(elementType))) {
    return failure();
  }

  if (failed(parser.parseGreater())) {
    return failure();
  }

  return success();
}

static int64_t getPTOTypeRank(Type type) {
  // 1. 处理标准的 MLIR 类型 (Tensor, Vector)
  if (auto shapedTy = dyn_cast<ShapedType>(type)) {
    if (shapedTy.hasRank()) {
      return shapedTy.getRank();
    }
    return -1; // Unranked type
  }

  // 2. 处理 PTO 自定义类型
  if (auto tvTy = dyn_cast<pto::TensorViewType>(type)) {
    return tvTy.getRank();
  }

  if (auto tileTy = dyn_cast<pto::TileType>(type)) {
    return tileTy.getRank();
  }

  if (auto tileViewTy = dyn_cast<pto::PartitionTensorViewType>(type)) {
    return tileViewTy.getRank();
  }

  if (auto tileBufTy = dyn_cast<pto::TileBufType>(type)) {
    return tileBufTy.getRank();
  }

  // 3. 不支持的类型
  return -1;
}

func::FuncOp mlir::pto::lookupPeerFuncAcrossContainer(Operation *op,
                                                      FlatSymbolRefAttr peerAttr) {
  if (!op || !peerAttr) {
    return {};
  }

  auto currentFunc = op->getParentOfType<func::FuncOp>();
  if (!currentFunc) {
    return {};
  }

  auto currentChildModule = currentFunc->getParentOfType<ModuleOp>();
  if (!currentChildModule) {
    return {};
  }

  StringRef target = peerAttr.getValue();
  for (func::FuncOp funcOp : currentChildModule.getOps<func::FuncOp>()) {
    if (funcOp.getSymName() == target) {
      return funcOp;
    }
  }
  if (auto localPeer = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupSymbolIn(currentChildModule, target))) {
    return localPeer;
  }

  Operation *maybeOuter = currentChildModule->getParentOp();
  auto outerModule = dyn_cast_or_null<ModuleOp>(maybeOuter);
  if (!outerModule) {
    return {};
  }

  SmallVector<func::FuncOp> fallbackMatches;
  outerModule.walk([&](func::FuncOp funcOp) {
    auto visibility = funcOp->getAttrOfType<StringAttr>("sym_visibility");
    if (visibility && visibility.getValue() == "private") {
      return WalkResult::advance();
    }

    StringRef symbolName = funcOp.getSymName();
    if (symbolName == target ||
        (funcOp->hasAttr(kPTODSLLogicalNameAttrName) &&
         getPTODSLLogicalNameOrSymbolName(funcOp) == target)) {
      fallbackMatches.push_back(funcOp);
    }
    return WalkResult::advance();
  });

  if (fallbackMatches.size() == 1) {
    return fallbackMatches.front();
  }
  return {};
}

static bool isA5DeviceSpec(StringRef spec) {
  return spec.starts_with("Ascend950") || spec.starts_with("Ascend910_95");
}

static bool isA5ModuleTarget(ModuleOp module) {
  if (!module) {
    return false;
  }
  if (auto arch = module->getAttrOfType<StringAttr>(kPTOTargetArchAttrName)) {
    if (arch.getValue().equals_insensitive("a5")) {
      return true;
    }
  }
  if (auto spec = module->getAttrOfType<StringAttr>("pto.device-spec")) {
    return isA5DeviceSpec(spec.getValue());
  }
  return false;
}

PTOArch mlir::pto::getTargetArch(ModuleOp module) {
  if (isA5ModuleTarget(module)) {
    return PTOArch::A5;
  }

  switch (getPTOParserTargetArch(module ? module.getContext() : nullptr)) {
  case PTOParserTargetArch::A5:
    return PTOArch::A5;
  case PTOParserTargetArch::A3:
  case PTOParserTargetArch::Unspecified:
    break;
  }
  return PTOArch::A3;
}

PTOArch mlir::pto::getTargetArch(Operation *op) {
  if (!op) {
    return PTOArch::A3;
  }
  if (auto module = op->getParentOfType<ModuleOp>()) {
    return getTargetArch(module);
  }
  switch (getPTOParserTargetArch(op->getContext())) {
  case PTOParserTargetArch::A5:
    return PTOArch::A5;
  case PTOParserTargetArch::A3:
  case PTOParserTargetArch::Unspecified:
    break;
  }
  return PTOArch::A3;
}

bool mlir::pto::isTargetArchA3(ModuleOp module) {
  return getTargetArch(module) == PTOArch::A3;
}

bool mlir::pto::isTargetArchA5(ModuleOp module) {
  return getTargetArch(module) == PTOArch::A5;
}

bool mlir::pto::isTargetArchA3(Operation *op) {
  return getTargetArch(op) == PTOArch::A3;
}

bool mlir::pto::isTargetArchA5(Operation *op) {
  return getTargetArch(op) == PTOArch::A5;
}

constexpr int64_t kA5VectorLengthBytes = 256;

enum class PredicateLoadDist {
  Norm,
  Us,
  Ds,
};

enum class PredicateStoreDist {
  Norm,
  Pk,
};

struct PredicateLoadAlignmentRule {
  PredicateLoadDist dist;
  int64_t alignmentBytes;
};

struct PredicateStoreAlignmentRule {
  PredicateStoreDist dist;
  int64_t alignmentBytes;
};

constexpr PredicateLoadAlignmentRule kA5PredicateLoadAlignmentRules[] = {
    {PredicateLoadDist::Norm, kA5VectorLengthBytes / 8},
    {PredicateLoadDist::Us, kA5VectorLengthBytes / 16},
    {PredicateLoadDist::Ds, std::min<int64_t>(32, kA5VectorLengthBytes / 4)},
};

constexpr PredicateStoreAlignmentRule kA5PredicateStoreAlignmentRules[] = {
    {PredicateStoreDist::Norm, kA5VectorLengthBytes / 8},
    {PredicateStoreDist::Pk, kA5VectorLengthBytes / 16},
};

static std::optional<PredicateLoadDist>
parsePredicateLoadDist(StringRef dist) {
  if (dist == "NORM") {
    return PredicateLoadDist::Norm;
  }
  if (dist == "US") {
    return PredicateLoadDist::Us;
  }
  if (dist == "DS") {
    return PredicateLoadDist::Ds;
  }
  return std::nullopt;
}

static std::optional<PredicateStoreDist>
parsePredicateStoreDist(StringRef dist) {
  if (dist == "NORM") {
    return PredicateStoreDist::Norm;
  }
  if (dist == "PK") {
    return PredicateStoreDist::Pk;
  }
  return std::nullopt;
}

template <typename Rule, typename Dist, size_t N>
static std::optional<int64_t> findAlignmentSize(const Rule (&rules)[N],
                                                Dist dist) {
  auto rule = llvm::find_if(
      rules, [&](const Rule &entry) { return entry.dist == dist; });
  if (rule == std::end(rules)) {
    return std::nullopt;
  }
  return rule->alignmentBytes;
}

std::optional<int64_t>
mlir::pto::getLoadStoreVecAlignmentSize(Operation *op) {
  if (!op || getTargetArch(op) != PTOArch::A5) {
    return std::nullopt;
  }

  if (auto pldi = dyn_cast<PldiOp>(op)) {
    auto dist = parsePredicateLoadDist(pldi.getDist());
    return dist ? findAlignmentSize(kA5PredicateLoadAlignmentRules, *dist)
                : std::nullopt;
  }
  if (auto psti = dyn_cast<PstiOp>(op)) {
    auto dist = parsePredicateStoreDist(psti.getDist());
    return dist ? findAlignmentSize(kA5PredicateStoreAlignmentRules, *dist)
                : std::nullopt;
  }
  if (auto sprsti = dyn_cast<SprstiOp>(op)) {
    if (sprsti.getSpr() == "AR") {
        return mlir::pto::kValue4;
    }
  }
  return std::nullopt;
}

static llvm::TypeSize getOneByteTypeSize() { return llvm::TypeSize::getFixed(mlir::pto::kValue8); }
