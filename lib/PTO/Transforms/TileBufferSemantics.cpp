#include "TileBufferSemantics.h"

#include "Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "pto-tile-buffer-semantics"

namespace mlir {
namespace pto {

// Reads planning address-space from either memref or tile-buffer values.
std::optional<AddressSpaceAttr> getPlanningBufferSpaceAttr(Value operand) {
  if (auto memRefType = dyn_cast<MemRefType>(operand.getType())) {
    auto memorySpace = memRefType.getMemorySpace();
    if (!memorySpace)
      return std::nullopt;
    return dyn_cast<AddressSpaceAttr>(memorySpace);
  }

  if (auto tileBufType = dyn_cast<pto::TileBufType>(operand.getType())) {
    auto memorySpace = tileBufType.getMemorySpace();
    if (!memorySpace)
      return std::nullopt;
    return dyn_cast<AddressSpaceAttr>(memorySpace);
  }
  return std::nullopt;
}

// Returns (alias_result, source) for planning aliases, including tile views.
std::optional<std::pair<Value, Value>> getBufferAliasInfo(Operation *op) {
  if (auto genericAlias = getOperationAliasInfo(op))
    return genericAlias;

  if (auto bindOp = dyn_cast<pto::BindTileOp>(op))
    return std::make_pair(bindOp.getResult(), bindOp.getSource());
  if (auto subsetOp = dyn_cast<pto::SubsetOp>(op))
    return std::make_pair(subsetOp.getResult(), subsetOp.getSource());
  if (auto bitcastOp = dyn_cast<pto::BitcastOp>(op))
    return std::make_pair(bitcastOp.getResult(), bitcastOp.getSrc());
  if (auto treshapeOp = dyn_cast<pto::TReshapeOp>(op))
    return std::make_pair(treshapeOp.getResult(), treshapeOp.getSrc());
  return std::nullopt;
}

// Classifies view semantics of planning-relevant operations.
TileViewKind getTileViewKind(Operation *op) {
  if (!op)
    return TileViewKind::Unknown;

  if (auto bindOp = dyn_cast<pto::BindTileOp>(op)) {
    if (auto semantics =
            bindOp->getAttrOfType<StringAttr>("pto.view_semantics")) {
      if (semantics.getValue() == "subset")
        return TileViewKind::Subset;
      if (semantics.getValue() == "bitcast")
        return TileViewKind::Bitcast;
      if (semantics.getValue() == "treshape")
        return TileViewKind::TReshape;
    }
    return TileViewKind::BindTile;
  }

  if (isa<pto::SubsetOp>(op))
    return TileViewKind::Subset;
  if (isa<pto::BitcastOp>(op))
    return TileViewKind::Bitcast;
  if (isa<pto::TReshapeOp>(op))
    return TileViewKind::TReshape;
  if (isa<memref::SubViewOp, memref::ViewOp, memref::ReinterpretCastOp,
          memref::CastOp, memref::CollapseShapeOp, memref::ExpandShapeOp,
          memref::ReshapeOp, memref::ExtractStridedMetadataOp>(op))
    return TileViewKind::MemRefViewLike;
  return TileViewKind::Unknown;
}

// Best-effort constant folding for index-like integers used by valid-shape
// propagation.
static std::optional<int64_t> getConstIndexLike(Value v) {
  if (!v)
    return std::nullopt;
  if (auto cOp = v.getDefiningOp<arith::ConstantIndexOp>())
    return cOp.value();
  if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>())
    return cInt.value();
  if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(cOp.getValue()))
      return ia.getInt();
  }
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>())
    return getConstIndexLike(castOp.getIn());
  if (auto extOp = v.getDefiningOp<arith::ExtSIOp>())
    return getConstIndexLike(extOp.getIn());
  if (auto extOp = v.getDefiningOp<arith::ExtUIOp>())
    return getConstIndexLike(extOp.getIn());
  if (auto truncOp = v.getDefiningOp<arith::TruncIOp>())
    return getConstIndexLike(truncOp.getIn());
  return std::nullopt;
}

// Single-step traceback through planning alias/view-like constructs.
static Value tracebackRootOneStep(Value value) {
  // Case 1: value is the iter_arg of a scf.for.
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (auto forOp =
            dyn_cast<scf::ForOp>(arg.getParentRegion()->getParentOp())) {
      if (arg.getArgNumber() > 0 &&
          forOp.getInitArgs().size() > arg.getArgNumber() - 1) {
        return forOp.getInitArgs()[arg.getArgNumber() - 1];
      }
    }
  }

  Operation *def = value.getDefiningOp();
  if (!def)
    return Value{};

  if (auto aliasPair = getBufferAliasInfo(def)) {
    auto [aliasValue, sourceValue] = *aliasPair;
    if (aliasValue == value)
      return sourceValue;
  }

  // Case 2: cast-like memref ops.
  if (auto op = dyn_cast<memref::MemorySpaceCastOp>(def))
    return op.getSource();
  if (auto op = dyn_cast<memref::TransposeOp>(def))
    return op.getIn();
  if (auto op = dyn_cast<UnrealizedConversionCastOp>(def))
    return op.getOperand(cast<OpResult>(value).getResultNumber());
  if (auto op = dyn_cast<scf::ForOp>(def))
    return op.getInitArgs()[cast<OpResult>(value).getResultNumber()];

  return Value{};
}

// Traces to the ultimate storage root used by planning and alias checks.
Value tracebackBufferRoot(Value value) {
  int loopBound = 256;
  while (value) {
    auto upward = tracebackRootOneStep(value);
    if (!upward)
      break;
    value = upward;
    if (loopBound-- < 0) {
      LLVM_DEBUG(llvm::dbgs() << "tracebackBufferRoot exceeds loopBound("
                              << loopBound << ")!");
      break;
    }
  }
  return value;
}

// Converts element type to bit-width for static-size estimation.
static int64_t getElemBitWidth(Type elemTy) {
  if (!elemTy)
    return -1;
  if (auto intTy = dyn_cast<IntegerType>(elemTy))
    return intTy.getWidth();
  if (auto floatTy = dyn_cast<FloatType>(elemTy))
    return floatTy.getWidth();
  if (isa<IndexType>(elemTy))
    return 64;
  return -1;
}

// Decodes shape/valid/config from either memref or tile type.
static bool decodeTypeSemantics(Type type, Type &elemTy,
                                SmallVectorImpl<int64_t> &shape,
                                SmallVectorImpl<int64_t> &validShape,
                                TileBufConfigAttr &config) {
  if (auto memRefType = dyn_cast<MemRefType>(type)) {
    elemTy = memRefType.getElementType();
    shape.assign(memRefType.getShape().begin(), memRefType.getShape().end());
    validShape = shape;
    return true;
  }
  if (auto tileBufType = dyn_cast<pto::TileBufType>(type)) {
    elemTy = tileBufType.getElementType();
    shape.assign(tileBufType.getShape().begin(), tileBufType.getShape().end());
    validShape.assign(tileBufType.getValidShape().begin(),
                      tileBufType.getValidShape().end());
    config = tileBufType.getConfigAttr();
    return true;
  }
  return false;
}

// Overrides valid-shape from bind_tile operands when constants are available.
static void applyBindTileValidShape(pto::BindTileOp bindOp,
                                    SmallVectorImpl<int64_t> &validShape) {
  auto ensureRank2 = [&]() {
    if (validShape.size() < 2)
      validShape.resize(2, ShapedType::kDynamic);
  };

  if (bindOp.getValidRow()) {
    ensureRank2();
    validShape[0] = getConstIndexLike(bindOp.getValidRow())
                        .value_or(ShapedType::kDynamic);
  }
  if (bindOp.getValidCol()) {
    ensureRank2();
    validShape[1] = getConstIndexLike(bindOp.getValidCol())
                        .value_or(ShapedType::kDynamic);
  }
}

// Collects all SSA buffers that an op reads/writes/produces.
static SmallVector<Value> getOpTouchBuffer(Operation *op) {
  SmallVector<Value> touchBuffer;
  touchBuffer.insert(touchBuffer.end(), op->getResults().begin(),
                     op->getResults().end());
  for (OpOperand &operand : op->getOpOperands())
    touchBuffer.push_back(operand.get());
  return touchBuffer;
}

// Returns true when any touched SSA value resolves to local planning space.
bool isOpTouchPlannableLocalBuffer(Operation *op) {
  auto touchBuffer = getOpTouchBuffer(op);
  for (Value buffer : touchBuffer) {
    auto bufferSpace = getPlanningBufferSpaceAttr(buffer);
    if (isLocalBuffer(bufferSpace))
      return true;
  }
  return false;
}

// Records semantic inference failure reason when callers request diagnostics.
static LogicalResult failSemantics(std::string *failureReason,
                                   llvm::StringRef message) {
  if (failureReason) {
    *failureReason = message.str();
  }
  return failure();
}

// Builds normalized planning semantics from a value:
// - root: traced storage owner
// - scope: local memory space
// - shape/valid/config/view-kind
// - constBits: static bytes in bits
LogicalResult inferTileBufferSemantics(Value value, TileBufferSemantics &out,
                                       std::string *failureReason) {
  if (!value)
    return failSemantics(failureReason, "value is null");

  out = TileBufferSemantics{};
  out.value = value;
  out.root = tracebackBufferRoot(value);
  if (auto as = getPlanningBufferSpaceAttr(value)) {
    out.scope = as->getAddressSpace();
  } else if (auto as = getPlanningBufferSpaceAttr(out.root)) {
    out.scope = as->getAddressSpace();
  } else {
    return failSemantics(failureReason,
                         "failed to resolve address-space from value/root");
  }

  // Prefer root storage type for size calculation and keep queried type as
  // fallback when root cannot provide a shaped type.
  bool decoded = decodeTypeSemantics(out.root ? out.root.getType() : Type{},
                                     out.elementType, out.shape,
                                     out.validShape, out.config);
  if (!decoded) {
    decoded = decodeTypeSemantics(value.getType(), out.elementType, out.shape,
                                  out.validShape, out.config);
  }
  if (!decoded)
    return failSemantics(
        failureReason,
        "failed to decode shape/element/config from root/value type");

  if (out.shape.empty()) {
    return failSemantics(failureReason, "decoded shape is empty");
  }
  for (int64_t dim : out.shape) {
    if (ShapedType::isDynamic(dim)) {
      return failSemantics(
          failureReason,
          "dynamic shape is unsupported for PlanMemory static sizing");
    }
    if (dim <= 0) {
      return failSemantics(failureReason,
                           "shape dimensions must be positive");
    }
  }

  if (auto def = value.getDefiningOp()) {
    out.viewKind = getTileViewKind(def);
    if (auto bindOp = dyn_cast<pto::BindTileOp>(def)) {
      if (!out.config)
        out.config = bindOp.getConfigAttr();
      applyBindTileValidShape(bindOp, out.validShape);
    }
  }

  auto staticSize = getStaticTotalSize(out.shape);
  int64_t elemBits = getElemBitWidth(out.elementType);
  if (!staticSize.has_value()) {
    return failSemantics(failureReason,
                         "failed to compute static element count from shape");
  }
  if (staticSize.value() <= 0) {
    return failSemantics(failureReason,
                         "static element count must be positive");
  }
  if (elemBits <= 0) {
    return failSemantics(failureReason,
                         "unsupported element type bit-width for sizing");
  }
  out.constBits = staticSize.value() * elemBits;
  return success();
}

} // namespace pto
} // namespace mlir
