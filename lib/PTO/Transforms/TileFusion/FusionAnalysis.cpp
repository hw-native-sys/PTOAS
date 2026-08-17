// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/TileFusion/FusionAnalysis.h"

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace pto {

namespace {

static int64_t getConstantIndexOrDynamic(Value value) {
  if (!value) {
    return ShapedType::kDynamic;
  }
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>()) {
    return cst.value();
  }
  if (auto cst = value.getDefiningOp<arith::ConstantIntOp>()) {
    return cst.value();
  }
  return ShapedType::kDynamic;
}

static SmallVector<int64_t, mlir::pto::kValue4> getValidShapeVec(Type type) {
  if (auto tileType = dyn_cast<pto::TileBufType>(type)) {
    return SmallVector<int64_t, mlir::pto::kValue4>(tileType.getValidShape().begin(),
                                   tileType.getValidShape().end());
  }
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return SmallVector<int64_t, mlir::pto::kValue4>(shapedType.getShape().begin(),
                                   shapedType.getShape().end());
  }
  return {};
}

/// Returns true if \p op is a pure, regionless operation that can participate
/// in structural-equivalence-based shape canonicalization.  We only allow ops
/// that are memory-effect-free, have no regions, and produce at most one
/// result, so that two such ops with the same name, attributes and equivalent
/// operands are guaranteed to produce the same value.
static bool isShapeComputableOp(Operation *op) {
  if (!op) {
    return false;
  }
  if (op->getNumRegions() != 0) {
    return false;
  }
  if (op->getNumResults() != 1) {
    return false;
  }
  if (!isMemoryEffectFree(op)) {
    return false;
  }

  // Only allow arith ops that appear in typical valid-shape computations
  // (minsi, maxsi, cmpi, select, addi, subi, muli, divsi, divui,
  //  index_cast, etc.) and
  // pto ops that are pure and regionless.
  StringRef opName = op->getName().getStringRef();
  if (opName.starts_with("arith.")) {
    return opName == "arith.minsi" || opName == "arith.maxsi" ||
           opName == "arith.minui" || opName == "arith.maxui" ||
           opName == "arith.cmpi" || opName == "arith.select" ||
           opName == "arith.addi" || opName == "arith.subi" ||
           opName == "arith.muli" || opName == "arith.divsi" ||
           opName == "arith.divui" || opName == "arith.index_cast";
  }

  return false;
}


/// A structural signature map that deduplicates computable shape expressions.
/// Maps (opName opaque pointer, canonicalOperands, attrs) → representative Value.
class StructuralSignatureMap {
public:
  struct Key {
    void *opNamePtr; // OperationName::getAsOpaquePointer()
    SmallVector<Value, mlir::pto::kValue4> operands;
    Attribute attrs;

    bool operator==(const Key &rhs) const {
      if (opNamePtr != rhs.opNamePtr) {
        return false;
      }
      if (operands.size() != rhs.operands.size()) {
        return false;
      }
      for (auto [l, r] : llvm::zip(operands, rhs.operands)) {
        if (l != r) {
          return false;
        }
      }
      return attrs == rhs.attrs;
    }
  };

  struct KeyInfo : public DenseMapInfo<Key> {
    static Key getEmptyKey() {
      return Key{DenseMapInfo<void *>::getEmptyKey(), {}, Attribute()};
    }
    static Key getTombstoneKey() {
      return Key{DenseMapInfo<void *>::getTombstoneKey(), {}, Attribute()};
    }
    static unsigned getHashValue(const Key &key) {
      unsigned h = DenseMapInfo<void *>::getHashValue(key.opNamePtr);
      for (Value v : key.operands) {
        h = llvm::hash_combine(h, DenseMapInfo<Value>::getHashValue(v));
      }
      h = llvm::hash_combine(h, DenseMapInfo<Attribute>::getHashValue(key.attrs));
      return h;
    }
    static bool isEqual(const Key &lhs, const Key &rhs) { return lhs == rhs; }
  };

  /// Try to find an existing representative for the given structural key.
  /// If found, return it.  Otherwise, register \p value as the representative
  /// and return it.
  Value getOrCreateRepresentative(const Key &key, Value value) {
    auto [it, inserted] = map.try_emplace(key, value);
    return it->second;
  }

private:
  DenseMap<Key, Value, KeyInfo> map;
};

/// Canonicalize a shape value using structural equivalence deduplication.
/// This version uses a StructuralSignatureMap to find the canonical
/// representative for computable expressions.
static Value canonicalizeValue(Value value,
                               DenseMap<Value, Value> &canonicalByValue,
                               StructuralSignatureMap &signatureMap) {
  if (!value) {
    return value;
  }

  auto cachedIt = canonicalByValue.find(value);
  if (cachedIt != canonicalByValue.end()) {
    return cachedIt->second;
  }

  // BlockArguments are their own canonical form.
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    canonicalByValue.try_emplace(value, value);
    return value;
  }

  Operation *op = value.getDefiningOp();
  if (!op) {
    canonicalByValue.try_emplace(value, value);
    return value;
  }

  // Constants are their own canonical form (handled by bindConstant).
  if (isa<arith::ConstantIndexOp, arith::ConstantIntOp>(op)) {
    canonicalByValue.try_emplace(value, value);
    return value;
  }

  // Non-computable ops are their own canonical form.
  if (!isShapeComputableOp(op)) {
    canonicalByValue.try_emplace(value, value);
    return value;
  }

  // Recursively canonicalize all operands.
  SmallVector<Value, mlir::pto::kValue4> canonicalOperands;
  canonicalOperands.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    canonicalOperands.push_back(
        canonicalizeValue(operand, canonicalByValue, signatureMap));
  }

  // Build structural key and look up / create representative.
  StructuralSignatureMap::Key key;
  key.opNamePtr = op->getName().getAsOpaquePointer();
  key.operands = std::move(canonicalOperands);
  key.attrs = op->getAttrDictionary();

  Value representative = signatureMap.getOrCreateRepresentative(key, value);
  canonicalByValue.try_emplace(value, representative);
  return representative;
}

static constexpr unsigned kInvalidShapeDim = ~0U;

struct ShapeValueDims {
  unsigned rows = kInvalidShapeDim;
  unsigned cols = kInvalidShapeDim;

  bool isValid() const {
    return rows != kInvalidShapeDim && cols != kInvalidShapeDim;
  }
};

class ShapeConstraintSolver {
public:
  unsigned createDim() {
    unsigned id = parent.size();
    parent.push_back(id);
    rank.push_back(0);
    constants.push_back(std::nullopt);
    conflicts.push_back(false);
    return id;
  }

  unsigned find(unsigned dim) {
    assert(dim < parent.size() && "shape dim out of range");
    if (parent[dim] == dim) {
      return dim;
    }
    parent[dim] = find(parent[dim]);
    return parent[dim];
  }

  void merge(unsigned lhs, unsigned rhs) {
    if (lhs == kInvalidShapeDim || rhs == kInvalidShapeDim) {
      return;
    }

    unsigned lhsRoot = find(lhs);
    unsigned rhsRoot = find(rhs);
    if (lhsRoot == rhsRoot) {
      return;
    }

    if (rank[lhsRoot] < rank[rhsRoot]) {
      std::swap(lhsRoot, rhsRoot);
    }
    parent[rhsRoot] = lhsRoot;
    if (rank[lhsRoot] == rank[rhsRoot]) {
      ++rank[lhsRoot];
    }

    conflicts[lhsRoot] = conflicts[lhsRoot] || conflicts[rhsRoot];
    if (constants[lhsRoot] && constants[rhsRoot] &&
        *constants[lhsRoot] != *constants[rhsRoot]) {
      conflicts[lhsRoot] = true;
    } else if (!constants[lhsRoot]) {
      constants[lhsRoot] = constants[rhsRoot];
    }
  }

  void bindConstant(unsigned dim, int64_t value) {
    if (dim == kInvalidShapeDim || value == ShapedType::kDynamic) {
      return;
    }

    unsigned root = find(dim);
    if (constants[root] && *constants[root] != value) {
      conflicts[root] = true;
    } else {
      constants[root] = value;
    }
  }

  bool hasConflict(unsigned dim) {
    if (dim == kInvalidShapeDim) {
      return true;
    }
    return conflicts[find(dim)];
  }

  std::optional<int64_t> getConstant(unsigned dim) {
    if (dim == kInvalidShapeDim) {
      return std::nullopt;
    }
    return constants[find(dim)];
  }

  /// Force-mark a dim's equivalence class as conflicting, preventing
  /// buildIterationDomainInfo from proving a consistent shape.  Used when a
  /// runtime set_validshape invalidates alloc-time shape assumptions.
  void markConflict(unsigned dim) {
    if (dim == kInvalidShapeDim) {
      return;
    }
    conflicts[find(dim)] = true;
  }

private:
  SmallVector<unsigned, mlir::pto::kValue32> parent;
  SmallVector<unsigned, mlir::pto::kValue32> rank;
  SmallVector<std::optional<int64_t>, mlir::pto::kValue32> constants;
  SmallVector<bool, mlir::pto::kValue32> conflicts;
};

static void bindDimToValue(ShapeConstraintSolver &solver,
                           DenseMap<Value, unsigned> &symbolDimByValue,
                           DenseMap<Value, Value> &canonicalByValue,
                           StructuralSignatureMap &signatureMap,
                           unsigned dim, Value value) {
  if (!value || dim == kInvalidShapeDim) {
    return;
  }

  int64_t constant = getConstantIndexOrDynamic(value);
  if (constant != ShapedType::kDynamic) {
    solver.bindConstant(dim, constant);
    return;
  }

  // Canonicalize the value so structurally equivalent SSA values map to
  // the same representative, enabling the solver to merge their dims.
  Value canonical =
      canonicalizeValue(value, canonicalByValue, signatureMap);
  auto [it, inserted] =
      symbolDimByValue.try_emplace(canonical, kInvalidShapeDim);
  if (inserted) {
    it->second = solver.createDim();
  }
  solver.merge(dim, it->second);
}

static void bindExplicitValidDims(ShapeConstraintSolver &solver,
                                  DenseMap<Value, unsigned> &symbolDimByValue,
                                  DenseMap<Value, Value> &canonicalByValue,
                                  StructuralSignatureMap &signatureMap,
                                  Value value, ShapeValueDims dims) {
  if (auto alloc = value.getDefiningOp<pto::AllocTileOp>()) {
    bindDimToValue(solver, symbolDimByValue, canonicalByValue, signatureMap,
                   dims.rows, alloc.getValidRow());
    bindDimToValue(solver, symbolDimByValue, canonicalByValue, signatureMap,
                   dims.cols, alloc.getValidCol());
    return;
  }
  if (auto subview = value.getDefiningOp<pto::SubViewOp>()) {
    bindDimToValue(solver, symbolDimByValue, canonicalByValue, signatureMap,
                   dims.rows, subview.getValidRow());
    bindDimToValue(solver, symbolDimByValue, canonicalByValue, signatureMap,
                   dims.cols, subview.getValidCol());
    return;
  }

  // pto.get_validshape reads runtime valid_row/valid_col from a tile_buf and
  // produces two index-typed results.  If the tile value is consumed by a
  // get_validshape op, bind the dims to that op's results so the solver can
  // track runtime-updated shape values rather than relying on stale alloc-time
  // metadata.
  for (Operation *user : value.getUsers()) {
    if (auto getVS = dyn_cast<pto::GetValidShapeOp>(user)) {
      bindDimToValue(solver, symbolDimByValue, canonicalByValue, signatureMap,
                     dims.rows, getVS.getValidRow());
      bindDimToValue(solver, symbolDimByValue, canonicalByValue, signatureMap,
                     dims.cols, getVS.getValidCol());
      return;
    }
  }
}

static ShapeValueDims getValueDims(
    ShapeConstraintSolver &solver, DenseMap<Value, ShapeValueDims> &dimsByValue,
    DenseMap<Value, unsigned> &symbolDimByValue,
    DenseMap<Value, Value> &canonicalByValue,
    StructuralSignatureMap &signatureMap, Value value) {
  auto existing = dimsByValue.find(value);
  if (existing != dimsByValue.end()) {
    return existing->second;
  }

  ShapeValueDims dims;
  SmallVector<int64_t, mlir::pto::kValue4> validShape = getValidShapeVec(value.getType());
  if (validShape.size() >= mlir::pto::kValue2) {
    dims.rows = solver.createDim();
    dims.cols = solver.createDim();
    if (!ShapedType::isDynamic(validShape[0])) {
      solver.bindConstant(dims.rows, validShape[0]);
    }
    if (!ShapedType::isDynamic(validShape[1])) {
      solver.bindConstant(dims.cols, validShape[1]);
    }
    bindExplicitValidDims(solver, symbolDimByValue, canonicalByValue,
                         signatureMap, value, dims);
  }

  dimsByValue.try_emplace(value, dims);
  return dims;
}

static void mergeRows(ShapeConstraintSolver &solver, ShapeValueDims lhs,
                      ShapeValueDims rhs) {
  solver.merge(lhs.rows, rhs.rows);
}

static void mergeCols(ShapeConstraintSolver &solver, ShapeValueDims lhs,
                      ShapeValueDims rhs) {
  solver.merge(lhs.cols, rhs.cols);
}

static void mergeShapes(ShapeConstraintSolver &solver, ShapeValueDims lhs,
                        ShapeValueDims rhs) {
  mergeRows(solver, lhs, rhs);
  mergeCols(solver, lhs, rhs);
}

static void mergeAllShapes(
    ShapeConstraintSolver &solver, DenseMap<Value, ShapeValueDims> &dimsByValue,
    DenseMap<Value, unsigned> &symbolDimByValue,
    DenseMap<Value, Value> &canonicalByValue,
    StructuralSignatureMap &signatureMap, ArrayRef<Value> values) {
  if (values.empty()) {
    return;
  }
  ShapeValueDims anchor = getValueDims(solver, dimsByValue, symbolDimByValue,
                                        canonicalByValue, signatureMap,
                                        values.front());
  for (Value value : values.drop_front()) {
    mergeShapes(solver, anchor,
                getValueDims(solver, dimsByValue, symbolDimByValue,
                             canonicalByValue, signatureMap, value));
  }
}

static void applyShapeConstraintsForNode(
    ShapeConstraintSolver &solver, DenseMap<Value, ShapeValueDims> &dimsByValue,
    DenseMap<Value, unsigned> &symbolDimByValue,
    DenseMap<Value, Value> &canonicalByValue,
    StructuralSignatureMap &signatureMap,
    const FusionComputeNode &node) {
  const FusionOpSemantics &semantics = node.semantics;
  switch (semantics.computeFamily) {
  case FusionComputeFamily::Elementwise: {
    SmallVector<Value, mlir::pto::kValue6> values;
    values.append(semantics.tileInputs.begin(), semantics.tileInputs.end());
    values.append(semantics.tileOutputs.begin(), semantics.tileOutputs.end());
    mergeAllShapes(solver, dimsByValue, symbolDimByValue, canonicalByValue,
                   signatureMap, values);
    return;
  }
  case FusionComputeFamily::ScalarExpand:
    mergeAllShapes(solver, dimsByValue, symbolDimByValue, canonicalByValue,
                   signatureMap, semantics.tileOutputs);
    return;
  case FusionComputeFamily::RowBroadcastBinary: {
    if (semantics.tileOutputs.empty()) {
      return;
    }
    ShapeValueDims output = getValueDims(
        solver, dimsByValue, symbolDimByValue, canonicalByValue, signatureMap,
        semantics.tileOutputs.front());
    if (!semantics.tileInputs.empty()) {
      mergeShapes(solver,
                  getValueDims(solver, dimsByValue, symbolDimByValue,
                               canonicalByValue, signatureMap,
                               semantics.tileInputs[0]),
                  output);
    }
    if (semantics.tileInputs.size() >= mlir::pto::kValue2) {
      ShapeValueDims rowInput = getValueDims(
          solver, dimsByValue, symbolDimByValue, canonicalByValue, signatureMap,
          semantics.tileInputs[1]);
      mergeRows(solver, rowInput, output);
      solver.bindConstant(rowInput.cols, 1);
    }
    for (Value extraOutput : ArrayRef<Value>(semantics.tileOutputs).drop_front()) {
      mergeShapes(solver, output,
                  getValueDims(solver, dimsByValue, symbolDimByValue,
                               canonicalByValue, signatureMap, extraOutput));
    }
    return;
  }
  case FusionComputeFamily::ReduceRow:
  case FusionComputeFamily::ReduceCol: {
    mergeAllShapes(solver, dimsByValue, symbolDimByValue, canonicalByValue,
                   signatureMap, semantics.tileInputs);
    if (semantics.tileInputs.empty() || semantics.tileOutputs.empty()) {
      return;
    }
    ShapeValueDims input = getValueDims(
        solver, dimsByValue, symbolDimByValue, canonicalByValue, signatureMap,
        semantics.tileInputs.front());
    ShapeValueDims output = getValueDims(
        solver, dimsByValue, symbolDimByValue, canonicalByValue, signatureMap,
        semantics.tileOutputs.front());
    if (semantics.computeFamily == FusionComputeFamily::ReduceRow) {
      mergeRows(solver, input, output);
      solver.bindConstant(output.cols, 1);
    } else {
      solver.bindConstant(output.rows, 1);
      mergeCols(solver, input, output);
    }
    for (Value extraOutput : ArrayRef<Value>(semantics.tileOutputs).drop_front()) {
      mergeShapes(solver, output,
                  getValueDims(solver, dimsByValue, symbolDimByValue,
                               canonicalByValue, signatureMap, extraOutput));
    }
    return;
  }
  case FusionComputeFamily::Unknown:
    return;
  }
}

static ShapeValueDims getIterationDomainDimsForNode(
    ShapeConstraintSolver &solver, DenseMap<Value, ShapeValueDims> &dimsByValue,
    DenseMap<Value, unsigned> &symbolDimByValue,
    DenseMap<Value, Value> &canonicalByValue,
    StructuralSignatureMap &signatureMap,
    const FusionComputeNode &node) {
  const FusionOpSemantics &semantics = node.semantics;
  switch (semantics.computeFamily) {
  case FusionComputeFamily::Elementwise:
  case FusionComputeFamily::ScalarExpand:
  case FusionComputeFamily::RowBroadcastBinary:
    if (!semantics.tileOutputs.empty()) {
      return getValueDims(solver, dimsByValue, symbolDimByValue,
                          canonicalByValue, signatureMap,
                          semantics.tileOutputs.front());
    }
    if (!semantics.tileInputs.empty()) {
      return getValueDims(solver, dimsByValue, symbolDimByValue,
                          canonicalByValue, signatureMap,
                          semantics.tileInputs.front());
    }
    break;
  case FusionComputeFamily::ReduceRow:
  case FusionComputeFamily::ReduceCol:
    if (!semantics.tileInputs.empty()) {
      return getValueDims(solver, dimsByValue, symbolDimByValue,
                          canonicalByValue, signatureMap,
                          semantics.tileInputs.front());
    }
    break;
  case FusionComputeFamily::Unknown:
    break;
  }
  return ShapeValueDims();
}

static IterationDomainInfo
buildIterationDomainInfo(ShapeConstraintSolver &solver, ShapeValueDims dims) {
  IterationDomainInfo info;
  if (!dims.isValid()) {
    return info;
  }
  if (solver.hasConflict(dims.rows) || solver.hasConflict(dims.cols)) {
    info.unprovenReason = IterationDomainUnprovenReason::InconsistentShape;
    return info;
  }

  info.proof = IterationDomainProof::Proven;
  info.unprovenReason = IterationDomainUnprovenReason::None;
  if (std::optional<int64_t> row = solver.getConstant(dims.rows)) {
    info.vRow = *row;
  }
  if (std::optional<int64_t> col = solver.getConstant(dims.cols)) {
    info.vCol = *col;
  }
  return info;
}

static unsigned assignShapeInferredDomainClass(
    ShapeConstraintSolver &solver,
    SmallVectorImpl<IterationDomainClass> &classes,
    DenseMap<std::pair<unsigned, unsigned>, unsigned> &provenClassByRoot,
    ShapeValueDims dims, const IterationDomainInfo &info, unsigned nodeId) {
  if (info.proof == IterationDomainProof::Proven) {
    std::pair<unsigned, unsigned> key{solver.find(dims.rows),
                                     solver.find(dims.cols)};
    auto it = provenClassByRoot.find(key);
    if (it != provenClassByRoot.end()) {
      classes[it->second].members.push_back(nodeId);
      return it->second;
    }

    unsigned classId = classes.size();
    IterationDomainClass klass;
    klass.id = classId;
    klass.info = info;
    klass.members.push_back(nodeId);
    classes.push_back(std::move(klass));
    provenClassByRoot.try_emplace(key, classId);
    return classId;
  }

  unsigned classId = classes.size();
  IterationDomainClass klass;
  klass.id = classId;
  klass.info = info;
  klass.members.push_back(nodeId);
  classes.push_back(std::move(klass));
  return classId;
}

//===----------------------------------------------------------------------===//
// Static iteration-domain inference.
//
// When shape inference (ShapeConstraintSolver) is disabled, the fusion pipeline
// falls back to this conservative inference, which only reasons about the
// statically known / directly-bound valid-shape of each tile value.  Two tiles
// are proven to share an iteration domain only when their valid-row/valid-col
// are identical compile-time constants.  Ops whose valid-shape is dynamic
// (valid=?x?) are left Unproven, so each becomes its own domain class and is
// never fused with another op on shape grounds.
//===----------------------------------------------------------------------===//

struct Rank2IterationSpace {
  int64_t rows = ShapedType::kDynamic;
  int64_t cols = ShapedType::kDynamic;
};

static std::optional<Rank2IterationSpace> getRank2IterationSpace(Value value) {
  SmallVector<int64_t, mlir::pto::kValue4> validShape = getValidShapeVec(value.getType());
  if (validShape.size() < mlir::pto::kValue2) {
    return std::nullopt;
  }
  return Rank2IterationSpace{validShape[0], validShape[1]};
}

static void mergeIterationDim(int64_t &mergedDim, int64_t dim,
                              IterationDomainInfo &info) {
  if (mergedDim == ShapedType::kDynamic || dim == ShapedType::kDynamic) {
    mergedDim = ShapedType::kDynamic;
    if (info.unprovenReason == IterationDomainUnprovenReason::None) {
      info.unprovenReason = IterationDomainUnprovenReason::DynamicShape;
    }
    return;
  }

  if (mergedDim != dim) {
    mergedDim = ShapedType::kDynamic;
    info.unprovenReason = IterationDomainUnprovenReason::InconsistentShape;
  }
}

static IterationDomainInfo
inferConsensusIterationDomain(ArrayRef<Value> anchorValues) {
  IterationDomainInfo info;
  info.unprovenReason = IterationDomainUnprovenReason::None;

  if (anchorValues.empty()) {
    return info;
  }

  std::optional<Rank2IterationSpace> firstSpace =
      getRank2IterationSpace(anchorValues.front());
  if (!firstSpace) {
    return info;
  }

  info.vRow = firstSpace->rows;
  info.vCol = firstSpace->cols;

  if (info.vRow == ShapedType::kDynamic || info.vCol == ShapedType::kDynamic) {
    info.unprovenReason = IterationDomainUnprovenReason::DynamicShape;
  }

  for (Value value : ArrayRef<Value>(anchorValues).drop_front()) {
    std::optional<Rank2IterationSpace> space = getRank2IterationSpace(value);
    if (!space) {
      info.vRow = ShapedType::kDynamic;
      info.vCol = ShapedType::kDynamic;
      info.unprovenReason = IterationDomainUnprovenReason::MissingTileDomain;
      return info;
    }
    mergeIterationDim(info.vRow, space->rows, info);
    mergeIterationDim(info.vCol, space->cols, info);
  }

  if (info.unprovenReason == IterationDomainUnprovenReason::None &&
      info.vRow != ShapedType::kDynamic && info.vCol != ShapedType::kDynamic) {
    info.proof = IterationDomainProof::Proven;
    return info;
  }

  if (info.unprovenReason == IterationDomainUnprovenReason::None) {
    info.unprovenReason = IterationDomainUnprovenReason::DynamicShape;
  }
  return info;
}

static IterationDomainInfo
inferIterationDomainInfo(const FusionOpSemantics &semantics) {
  switch (semantics.computeFamily) {
  case FusionComputeFamily::Elementwise: {
    SmallVector<Value, mlir::pto::kValue6> anchors;
    anchors.append(semantics.tileInputs.begin(), semantics.tileInputs.end());
    anchors.append(semantics.tileOutputs.begin(), semantics.tileOutputs.end());
    return inferConsensusIterationDomain(anchors);
  }
  case FusionComputeFamily::ScalarExpand:
  case FusionComputeFamily::RowBroadcastBinary:
    return inferConsensusIterationDomain(semantics.tileOutputs);
  case FusionComputeFamily::ReduceRow:
  case FusionComputeFamily::ReduceCol:
    return inferConsensusIterationDomain(semantics.tileInputs);
  case FusionComputeFamily::Unknown:
    return IterationDomainInfo();
  }
  return IterationDomainInfo();
}

static unsigned assignStaticIterationDomainClass(
    SmallVectorImpl<IterationDomainClass> &classes,
    DenseMap<std::pair<int64_t, int64_t>, unsigned> &provenClassByKey,
    const IterationDomainInfo &info, unsigned nodeId) {
  if (info.proof == IterationDomainProof::Proven) {
    std::pair<int64_t, int64_t> key{info.vRow, info.vCol};
    auto it = provenClassByKey.find(key);
    if (it != provenClassByKey.end()) {
      classes[it->second].members.push_back(nodeId);
      return it->second;
    }

    unsigned classId = classes.size();
    IterationDomainClass klass;
    klass.id = classId;
    klass.info = info;
    klass.members.push_back(nodeId);
    classes.push_back(std::move(klass));
    provenClassByKey.try_emplace(key, classId);
    return classId;
  }

  unsigned classId = classes.size();
  IterationDomainClass klass;
  klass.id = classId;
  klass.info = info;
  klass.members.push_back(nodeId);
  classes.push_back(std::move(klass));
  return classId;
}

/// Static fallback for iteration-domain inference: assigns each compute node
/// to an IterationDomainClass using only the directly-bound valid-shape of its
/// tiles.  Used when shape inference (--enable-shape-inference) is disabled.
static LogicalResult inferStaticIterationDomain(FusionBlockAnalysis &analysis) {
  analysis.iterationDomainClasses.clear();
  DenseMap<std::pair<int64_t, int64_t>, unsigned> provenClassByKey;
  for (FusionComputeNode &node : analysis.computeNodes) {
    IterationDomainInfo domainInfo = inferIterationDomainInfo(node.semantics);
    node.iterationDomainClass = assignStaticIterationDomainClass(
        analysis.iterationDomainClasses, provenClassByKey, domainInfo, node.id);
  }
  return success();
}

static LogicalResult inferDynamicIterationDomain(FusionBlockAnalysis &analysis) {
  ShapeConstraintSolver solver;
  DenseMap<Value, ShapeValueDims> dimsByValue;
  DenseMap<Value, unsigned> symbolDimByValue;
  DenseMap<Value, Value> canonicalByValue;
  StructuralSignatureMap signatureMap;

  for (const FusionComputeNode &node : analysis.computeNodes) {
    for (Value input : node.semantics.tileInputs) {
      (void)getValueDims(solver, dimsByValue, symbolDimByValue,
                         canonicalByValue, signatureMap, input);
    }
    for (Value output : node.semantics.tileOutputs) {
      (void)getValueDims(solver, dimsByValue, symbolDimByValue,
                         canonicalByValue, signatureMap, output);
    }
  }

  for (const FusionComputeNode &node : analysis.computeNodes) {
    applyShapeConstraintsForNode(solver, dimsByValue, symbolDimByValue,
                                 canonicalByValue, signatureMap, node);
  }

  // pto.set_validshape mutates runtime valid-row/valid-col metadata in-place
  // on a tile_buf.  If a set_validshape modifies a tile that participates in
  // the fusion block, the alloc-time valid shape is stale and the solver's
  // constraints for that tile may be incorrect.  Mark the affected tile's
  // dims as conflicting so buildIterationDomainInfo will report
  // InconsistentShape and the fusion pipeline will conservatively refuse to
  // fuse ops whose domain depends on the stale shape.
  if (analysis.block) {
    for (Operation &op : *analysis.block) {
      auto setVS = dyn_cast<pto::SetValidShapeOp>(op);
      if (!setVS) {
        continue;
      }
      Value source = setVS.getSource();
      auto dimsIt = dimsByValue.find(source);
      if (dimsIt != dimsByValue.end()) {
        solver.markConflict(dimsIt->second.rows);
        solver.markConflict(dimsIt->second.cols);
      }
    }
  }

  analysis.iterationDomainClasses.clear();
  DenseMap<std::pair<unsigned, unsigned>, unsigned> provenClassByRoot;
  for (FusionComputeNode &node : analysis.computeNodes) {
    ShapeValueDims domainDims = getIterationDomainDimsForNode(
        solver, dimsByValue, symbolDimByValue, canonicalByValue, signatureMap,
        node);
    IterationDomainInfo info = buildIterationDomainInfo(solver, domainDims);
    node.iterationDomainClass = assignShapeInferredDomainClass(
        solver, analysis.iterationDomainClasses, provenClassByRoot, domainDims,
        info, node.id);
  }
  return success();
}

struct MutableLiveness {
  FusionValueLiveness live;
};

struct MutableWriteInstance {
  FusionWriteInstanceLiveness live;
  unsigned producerBlockOrder = 0;
};

static FusionWriteInstanceEscapeClass classifyEscapeClass(
    const FusionWriteInstanceLiveness &live) {
  if (live.hasExternalUsers || live.escapesBlock ||
      live.hasLocalHardBoundaryUsers) {
    return FusionWriteInstanceEscapeClass::HardExternal;
  }
  if (live.hasLocalBoundaryUsers) {
    return FusionWriteInstanceEscapeClass::LocalBoundaryExternal;
  }
  return FusionWriteInstanceEscapeClass::Internal;
}

static Value getWriteInstanceStorageValue(Operation *op, unsigned outputIndex,
                                          Value output) {
  if (auto dpsIface = dyn_cast<pto::PTO_DpsInitOpInterface>(op)) {
    unsigned tileOutputIndex = 0;
    for (Value init : dpsIface.getDpsInits()) {
      if (!isa<pto::TileBufType>(init.getType())) {
        continue;
      }
      if (tileOutputIndex == outputIndex) {
        return init;
      }
      ++tileOutputIndex;
    }
  }
  return output;
}

static unsigned getOrCreateLivenessSlot(DenseMap<Value, unsigned> &slotByValue,
                                        SmallVectorImpl<MutableLiveness> &slots,
                                        Value value) {
  auto [it, inserted] = slotByValue.try_emplace(value, slots.size());
  if (inserted) {
    MutableLiveness state;
    state.live.value = value;
    slots.push_back(std::move(state));
  }
  return it->second;
}

static void appendUniqueNode(SmallVectorImpl<unsigned> &nodes, unsigned nodeId) {
  if (!llvm::is_contained(nodes, nodeId)) {
    nodes.push_back(nodeId);
  }
}

static void recordLastLocalConsumer(std::optional<unsigned> &lastLocalConsumer,
                                    unsigned consumerId) {
  if (!lastLocalConsumer || consumerId > *lastLocalConsumer) {
    lastLocalConsumer = consumerId;
  }
}

static void finalizeBlockLiveness(
    Block &block, DenseMap<Operation *, FusionOpKind> &kindByOp,
    DenseMap<Operation *, unsigned> &computeNodeByOp,
    SmallVectorImpl<MutableLiveness> &mutableLiveness) {
  for (MutableLiveness &state : mutableLiveness) {
    for (OpOperand &use : state.live.value.getUses()) {
      Operation *user = use.getOwner();
      if (user->getBlock() != &block) {
        state.live.hasExternalUsers = true;
        state.live.escapesBlock = true;
        continue;
      }

      auto kindIt = kindByOp.find(user);
      if (kindIt == kindByOp.end()) {
        continue;
      }

      if (user->hasTrait<OpTrait::IsTerminator>()) {
        state.live.escapesBlock = true;
      }

      switch (kindIt->second) {
      case FusionOpKind::Compute: {
        auto nodeIt = computeNodeByOp.find(user);
        if (nodeIt == computeNodeByOp.end()) {
          continue;
        }
        unsigned consumerId = nodeIt->second;
        appendUniqueNode(state.live.consumerNodes, consumerId);
        recordLastLocalConsumer(state.live.lastLocalConsumer, consumerId);
        break;
      }
      case FusionOpKind::LocalBoundary:
        state.live.hasLocalBoundaryUsers = true;
        break;
      case FusionOpKind::HardBoundary:
        state.live.hasLocalHardBoundaryUsers = true;
        break;
      }
    }
  }
}

static std::optional<unsigned> findReachingWriteInstance(
    ArrayRef<unsigned> writeInstanceIds,
    ArrayRef<MutableWriteInstance> mutableWriteInstances,
    std::optional<unsigned> userBlockOrder) {
  if (writeInstanceIds.empty()) {
    return std::nullopt;
  }

  if (!userBlockOrder) {
    return writeInstanceIds.back();
  }

  for (unsigned writeInstanceId : llvm::reverse(writeInstanceIds)) {
    if (mutableWriteInstances[writeInstanceId].producerBlockOrder <
        *userBlockOrder) {
      return writeInstanceId;
    }
  }
  return std::nullopt;
}

static bool isDpsInitOperandUse(OpOperand &use) {
  auto dpsIface = dyn_cast<pto::PTO_DpsInitOpInterface>(use.getOwner());
  if (!dpsIface) {
    return false;
  }

  for (OpOperand &dpsInit : dpsIface.getDpsInitsMutable()) {
    if (&dpsInit == &use) {
      return true;
    }
  }
  return false;
}

static void finalizeWriteInstances(
    Block &block, DenseMap<Operation *, FusionOpKind> &kindByOp,
    DenseMap<Operation *, unsigned> &computeNodeByOp,
    DenseMap<Operation *, unsigned> &blockOrderByOp,
    ArrayRef<MutableLiveness> mutableLiveness,
    SmallVectorImpl<MutableWriteInstance> &mutableWriteInstances) {
  for (const MutableLiveness &storageState : mutableLiveness) {
    if (storageState.live.writeInstances.empty()) {
      continue;
    }

    for (OpOperand &use : storageState.live.value.getUses()) {
      if (isDpsInitOperandUse(use)) {
        continue;
      }

      Operation *user = use.getOwner();
      bool isInBlock = user->getBlock() == &block;
      std::optional<unsigned> userBlockOrder;
      if (isInBlock) {
        auto orderIt = blockOrderByOp.find(user);
        if (orderIt != blockOrderByOp.end()) {
          userBlockOrder = orderIt->second;
        }
      }

      std::optional<unsigned> writeInstanceId = findReachingWriteInstance(
          storageState.live.writeInstances, mutableWriteInstances,
          userBlockOrder);
      if (!writeInstanceId) {
        continue;
      }

      FusionWriteInstanceLiveness &writeLive =
          mutableWriteInstances[*writeInstanceId].live;

      if (!isInBlock) {
        writeLive.hasExternalUsers = true;
        writeLive.escapesBlock = true;
        continue;
      }

      auto kindIt = kindByOp.find(user);
      if (kindIt == kindByOp.end()) {
        continue;
      }

      if (user->hasTrait<OpTrait::IsTerminator>()) {
        writeLive.escapesBlock = true;
      }

      switch (kindIt->second) {
      case FusionOpKind::Compute: {
        auto nodeIt = computeNodeByOp.find(user);
        if (nodeIt == computeNodeByOp.end()) {
          continue;
        }
        unsigned consumerId = nodeIt->second;
        appendUniqueNode(writeLive.consumerNodes, consumerId);
        recordLastLocalConsumer(writeLive.lastLocalConsumer, consumerId);
        break;
      }
      case FusionOpKind::LocalBoundary:
        writeLive.hasLocalBoundaryUsers = true;
        break;
      case FusionOpKind::HardBoundary:
        writeLive.hasLocalHardBoundaryUsers = true;
        break;
      }
    }
  }

  for (MutableWriteInstance &state : mutableWriteInstances) {
    state.live.escapeClass = classifyEscapeClass(state.live);
  }
}

/// Build the shared dataflow graph (compute nodes, DFG edges, value liveness,
/// write instances) for a single block, WITHOUT inferring iteration-domain
/// classes.  Domain inference is a separable, cheap follow-up step run by
/// inferIterationDomainClasses; it is deliberately kept out of the
/// analysis-manager-cached DFG so the cached graph can be shared across passes.
static FailureOr<FusionBlockAnalysis> analyzeBlockDFG(Block &block) {
  FusionBlockAnalysis analysis;
  analysis.block = &block;

  DenseMap<Value, unsigned> producerByValue;
  DenseMap<Value, unsigned> livenessSlotByValue;
  SmallVector<MutableLiveness, mlir::pto::kValue8> mutableLiveness;
  SmallVector<MutableWriteInstance, mlir::pto::kValue8> mutableWriteInstances;
  DenseMap<Operation *, FusionOpKind> kindByOp;
  DenseMap<Operation *, unsigned> computeNodeByOp;
  DenseMap<Operation *, unsigned> blockOrderByOp;

  unsigned blockOrder = 0;
  for (Operation &op : block) {
    FailureOr<FusionOpSemantics> semanticsOr = getFusionOpSemantics(&op);
    if (failed(semanticsOr)) {
      op.emitError("failed to normalize fusion op semantics");
      return failure();
    }
    blockOrderByOp[&op] = blockOrder;
    kindByOp[&op] = semanticsOr->kind;

    if (semanticsOr->kind == FusionOpKind::LocalBoundary) {
      for (Value input : semanticsOr->tileInputs) {
        getOrCreateLivenessSlot(livenessSlotByValue, mutableLiveness, input);
      }
      for (Value output : semanticsOr->tileOutputs) {
        getOrCreateLivenessSlot(livenessSlotByValue, mutableLiveness, output);
      }
      ++blockOrder;
      continue;
    }

    if (semanticsOr->kind != FusionOpKind::Compute) {
      ++blockOrder;
      continue;
    }

    FusionComputeNode node;
    node.id = analysis.computeNodes.size();
    node.blockOrder = blockOrder;
    node.op = &op;
    node.semantics = *semanticsOr;
    computeNodeByOp[&op] = node.id;

    for (auto [outputIdx, output] : llvm::enumerate(node.semantics.tileOutputs)) {
      producerByValue[output] = node.id;
      unsigned liveSlot =
          getOrCreateLivenessSlot(livenessSlotByValue, mutableLiveness, output);
      mutableLiveness[liveSlot].live.producerNode = node.id;

      MutableWriteInstance writeInstance;
      writeInstance.live.id = mutableWriteInstances.size();
      writeInstance.live.value = output;
      writeInstance.live.storageValue =
          getWriteInstanceStorageValue(&op, outputIdx, output);
      writeInstance.live.producerNode = node.id;
      writeInstance.producerBlockOrder = blockOrder;
      mutableLiveness[liveSlot].live.writeInstances.push_back(
          writeInstance.live.id);
      mutableWriteInstances.push_back(std::move(writeInstance));
    }

    for (Value input : node.semantics.tileInputs) {
      unsigned liveSlot =
          getOrCreateLivenessSlot(livenessSlotByValue, mutableLiveness, input);
      appendUniqueNode(mutableLiveness[liveSlot].live.consumerNodes, node.id);
      recordLastLocalConsumer(mutableLiveness[liveSlot].live.lastLocalConsumer,
                              node.id);

      auto producerIt = producerByValue.find(input);
      if (producerIt == producerByValue.end()) {
        continue;
      }

      FusionDFGEdge edge;
      edge.producerNode = producerIt->second;
      edge.consumerNode = node.id;
      edge.value = input;

      unsigned edgeId = analysis.edges.size();
      analysis.edges.push_back(edge);
      node.incomingEdges.push_back(edgeId);
      if (edge.producerNode < analysis.computeNodes.size()) {
        analysis.computeNodes[edge.producerNode].outgoingEdges.push_back(edgeId);
      }
    }

    analysis.computeNodes.push_back(std::move(node));
    ++blockOrder;
  }

  finalizeBlockLiveness(block, kindByOp, computeNodeByOp, mutableLiveness);
  finalizeWriteInstances(block, kindByOp, computeNodeByOp, blockOrderByOp,
                         mutableLiveness, mutableWriteInstances);

  analysis.liveness.reserve(mutableLiveness.size());
  for (MutableLiveness &state : mutableLiveness) {
    analysis.liveness.push_back(std::move(state.live));
  }
  analysis.writeInstances.reserve(mutableWriteInstances.size());
  for (MutableWriteInstance &state : mutableWriteInstances) {
    analysis.writeInstances.push_back(std::move(state.live));
  }

  return std::move(analysis);
}

static LogicalResult analyzeRegionDFG(Region &region,
                                      SmallVectorImpl<FusionBlockAnalysis> &blocks) {
  for (Block &block : region.getBlocks()) {
    FailureOr<FusionBlockAnalysis> blockAnalysis = analyzeBlockDFG(block);
    if (failed(blockAnalysis)) {
      return failure();
    }
    blocks.push_back(std::move(*blockAnalysis));
    for (Operation &op : block) {
      for (Region &nested : op.getRegions()) {
        if (failed(analyzeRegionDFG(nested, blocks))) {
          return failure();
        }
      }
    }
  }
  return success();
}

} // namespace

FailureOr<PreFusionAnalysisResult>
buildPreFusionAnalysisDFG(func::FuncOp func) {
  PreFusionAnalysisResult result;
  if (failed(analyzeRegionDFG(func.getRegion(), result.blocks))) {
    return failure();
  }
  return std::move(result);
}

LogicalResult inferIterationDomainClasses(PreFusionAnalysisResult &result,
                                          bool enableShapeInference) {
  for (FusionBlockAnalysis &block : result.blocks) {
    if (enableShapeInference) {
      if (failed(inferDynamicIterationDomain(block))) {
        return failure();
      }
    } else {
      if (failed(inferStaticIterationDomain(block))) {
        return failure();
      }
    }
  }
  return success();
}

FailureOr<PreFusionAnalysisResult>
buildPreFusionAnalysis(func::FuncOp func, bool enableShapeInference) {
  FailureOr<PreFusionAnalysisResult> result = buildPreFusionAnalysisDFG(func);
  if (failed(result)) {
    return failure();
  }
  if (failed(inferIterationDomainClasses(*result, enableShapeInference))) {
    return failure();
  }
  return std::move(*result);
}

} // namespace pto
} // namespace mlir
