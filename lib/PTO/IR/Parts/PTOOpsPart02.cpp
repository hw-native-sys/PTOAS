// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.

static LogicalResult verifyColReductionValidRegion(Operation *op, Type srcTy,
                                                   Type dstTy,
                                                   bool requireNonZeroSrc) {
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != 2 || dstValid.size() != 2)
    return op->emitOpError("expects src and dst to have rank-2 valid_shape");
  // Fully-empty dst valid region (0x0): dual-AIV no-op replay marker. The op
  // writes no elements; accept and skip the non-empty constraints. One-sided
  // empties still fall through. See pto-isa#143 for hardware Rv=0 no-op.
  // Col arg reductions (tcolargmax/tcolargmin) never reach this point with a
  // 0x0 dst: verifyColArgReductionDstLayout enforces dst valid_shape[0] == 1
  // first, so they stay strict without needing a flag here (unlike the row
  // path, whose dst-layout check does not constrain valid).
  if (dstValid[0] == 0 && dstValid[1] == 0)
    return success();
  if (requireNonZeroSrc) {
    if (srcValid[0] != ShapedType::kDynamic && srcValid[0] == 0)
      return op->emitOpError("expects src valid_shape[0] to be non-zero");
    if (srcValid[1] != ShapedType::kDynamic && srcValid[1] == 0)
      return op->emitOpError("expects src valid_shape[1] to be non-zero");
  }
  if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1])
    return op->emitOpError("expects src and dst to have the same valid_shape[1]");
  return success();
}

static LogicalResult verifyColArgReductionDstLayout(Operation *op, Type ty,
                                                    StringRef name) {
  if (failed(verifyNDStyleVecTile(op, ty, name)))
    return failure();
  auto valid = getValidShapeVec(ty);
  if (valid.size() != 2)
    return op->emitOpError() << "expects " << name
                             << " to have rank-2 valid_shape";
  if (valid[0] != ShapedType::kDynamic && valid[0] != 1)
    return op->emitOpError() << "expects " << name
                             << " valid_shape[0] to be 1";
  return success();
}

static std::optional<int64_t> getConstantIntegerValue(Value value) {
  if (!value)
    return std::nullopt;
  if (auto arithCst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(arithCst.getValue()))
      return intAttr.getInt();
  }
  return std::nullopt;
}

LogicalResult mlir::pto::FusionRegionOp::verify() {
  Region &bodyRegion = getBody();
  if (bodyRegion.empty())
    return emitOpError("expects a non-empty body region");

  Block &body = bodyRegion.front();
  if (body.getNumArguments() != 0)
    return emitOpError() << "expects body block to have no arguments, got "
                         << body.getNumArguments();

  if (body.empty() || !body.back().hasTrait<OpTrait::IsTerminator>())
    return emitOpError("expects body to terminate with pto.yield");

  auto yield = dyn_cast<YieldOp>(&body.back());
  if (!yield)
    return emitOpError("expects body to terminate with pto.yield");

  if (yield.getValues().size() != getOutputs().size())
    return emitOpError() << "expects pto.yield to return "
                         << getOutputs().size() << " values, got "
                         << yield.getValues().size();

  for (auto [idx, pair] :
       llvm::enumerate(llvm::zip(yield.getValues(), getOutputs()))) {
    Value yielded = std::get<0>(pair);
    Value output = std::get<1>(pair);
    if (yielded.getType() != output.getType())
      return emitOpError() << "expects yielded value #" << idx << " to have "
                           << "type " << output.getType() << ", got "
                           << yielded.getType();
  }

  return success();
}

LogicalResult mlir::pto::YieldOp::verify() {
  auto parent = dyn_cast_or_null<FusionRegionOp>(getOperation()->getParentOp());
  if (!parent)
    return emitOpError("expects parent op to be pto.fusion_region");

  if (getValues().size() != parent.getOutputs().size())
    return emitOpError() << "expects " << parent.getOutputs().size()
                         << " yielded values to match parent results, got "
                         << getValues().size();

  for (auto [idx, pair] :
       llvm::enumerate(llvm::zip(getValues(), parent.getOutputs()))) {
    Value yielded = std::get<0>(pair);
    Value output = std::get<1>(pair);
    if (yielded.getType() != output.getType())
      return emitOpError() << "expects yielded value #" << idx << " to have "
                           << "type " << output.getType() << ", got "
                           << yielded.getType();
  }

  return success();
}

LogicalResult mlir::pto::MakeTensorViewOp::verify() {
  auto tvTy = dyn_cast<mlir::pto::TensorViewType>(getResult().getType());
  if (!tvTy)
    return emitOpError("result must be pto.tensor_view<...>");

  auto pty = dyn_cast<mlir::pto::PtrType>(getPtr().getType());
  if (!pty)
    return emitOpError("ptr operand must be !pto.ptr<...>");

  if (pty.getElementType() != tvTy.getElementType())
    return emitOpError() << "ptr element type must match tensor_view element type, but got ptr="
                         << pty.getElementType() << " view=" << tvTy.getElementType();

  int64_t rank = tvTy.getRank();

  if ((int64_t)getShape().size() != rank || (int64_t)getStrides().size() != rank)
    return emitOpError() << "shape/strides operand counts must match tensor_view rank="
                         << rank;

  // Detect dynamic shape/stride.
  bool hasDynamicShape = llvm::any_of(tvTy.getShape(), [](int64_t v) {
    return v == ShapedType::kDynamic;
  });
  bool hasDynamicStride = llvm::any_of(getStrides(), [](Value s) {
    return !getConstIndexValue(s).has_value();
  });

  auto layoutAttr = getLayoutAttr();

  // 1) Dynamic shape/stride without explicit layout: warn and keep going.
  if ((hasDynamicShape || hasDynamicStride) && !layoutAttr) {
    return success();
  }

  // 2) Static shape/stride with explicit layout: verify correctness.
  bool allStaticStride = true;
  SmallVector<int64_t> strideInts;
  strideInts.reserve(getStrides().size());
  for (Value s : getStrides()) {
    auto val = getConstIndexValue(s);
    if (!val) {
      allStaticStride = false;
      break;
    }
    strideInts.push_back(*val);
  }

  bool allStaticShape =
      llvm::none_of(tvTy.getShape(), [](int64_t v) { return v == ShapedType::kDynamic; });

  if (layoutAttr && allStaticShape && allStaticStride) {
    SmallVector<int64_t> shapeInts(tvTy.getShape().begin(), tvTy.getShape().end());
    if (auto inferred = inferLayout(shapeInts, strideInts,
                                    getElemByteSize(tvTy.getElementType()))) {
      (void)inferred;
    }
  }

  return success();
}

LogicalResult mlir::pto::PartitionViewOp::verify() {
  auto srcTy = dyn_cast<mlir::pto::TensorViewType>(getSource().getType());
  auto resTy = dyn_cast<mlir::pto::PartitionTensorViewType>(getResult().getType());
  if (!srcTy || !resTy)
    return emitOpError("expects tensor_view source and partition_tensor_view result");

  if (srcTy.getElementType() != resTy.getElementType())
    return emitOpError() << "element type mismatch between source and result: src="
                         << srcTy.getElementType() << " result="
                         << resTy.getElementType();

  int64_t srcRank = srcTy.getRank();
  if ((int64_t)getOffsets().size() != srcRank)
    return emitOpError() << "offset count (" << getOffsets().size()
                         << ") must match source rank (" << srcRank << ")";

  if ((int64_t)getSizes().size() != srcRank)
    return emitOpError() << "size count (" << getSizes().size()
                         << ") must match source rank (" << srcRank << ")";

  ArrayRef<int64_t> srcShape = srcTy.getShape();
  ArrayRef<int64_t> resShape = resTy.getShape();
  bool sameRank = resTy.getRank() == srcRank;

  for (int64_t i = 0; i < srcRank; ++i) {
    auto offVal = getConstIndexValue(getOffsets()[i]);
    auto sizeVal = getConstIndexValue(getSizes()[i]);

    if (offVal && *offVal < 0)
      return emitOpError() << "offset at dim " << i
                           << " must be non-negative, got " << *offVal;

    if (sizeVal && *sizeVal <= 0)
      return emitOpError() << "size at dim " << i
                           << " must be positive, got " << *sizeVal;

    if (sameRank && sizeVal) {
      int64_t resDim = resShape[i];
      if (resDim != ShapedType::kDynamic && *sizeVal != resDim)
        return emitOpError() << "size/result mismatch at dim " << i
                             << ": size operand=" << *sizeVal
                             << " result type dim=" << resDim;
    }

    int64_t srcDim = srcShape[i];
    if (srcDim == ShapedType::kDynamic)
      continue;

    if (sizeVal && *sizeVal > srcDim)
      return emitOpError() << "size at dim " << i << " (" << *sizeVal
                           << ") exceeds static source dim (" << srcDim << ")";

    if (offVal && sizeVal && (*offVal + *sizeVal > srcDim))
      return emitOpError() << "offset+size at dim " << i << " ("
                           << (*offVal + *sizeVal)
                           << ") exceeds static source dim (" << srcDim << ")";
  }

  return success();
}

LogicalResult mlir::pto::AddPtrOp::verify() {
  Value ptr = getOperation()->getOperand(0);
  Value result = getOperation()->getResult(0);

  auto ptrTy = dyn_cast<mlir::pto::PtrType>(ptr.getType());
  if (!ptrTy)
    return emitOpError("ptr operand must be !pto.ptr<...>");

  auto resTy = dyn_cast<mlir::pto::PtrType>(result.getType());
  if (!resTy)
    return emitOpError("result must be !pto.ptr<...>");

  if (ptrTy != resTy)
    return emitOpError("result type must match ptr operand type");

  return success();
}

static LogicalResult verifyPtrLikeForAddressCast(Operation *op, Type type,
                                                 StringRef name) {
  if (isa<mlir::pto::PtrType>(type))
    return success();

  auto memTy = dyn_cast<MemRefType>(type);
  if (!memTy)
    return op->emitOpError()
           << "expects " << name << " to be !pto.ptr<...> or a GM memref";

  if (memTy.getRank() != 1)
    return op->emitOpError()
           << "expects lowered memref " << name << " to be rank-1";

  if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
    return op->emitOpError()
           << "expects lowered memref " << name << " to use GM address space";

  return success();
}

static Type getPointerLikeElementType(Type type) {
  if (auto ptrTy = dyn_cast<mlir::pto::PtrType>(type))
    return ptrTy.getElementType();
  if (auto memTy = dyn_cast<MemRefType>(type))
    return memTy.getElementType();
  return Type();
}

static bool isEmitCSupportedScalarType(Type type) {
  if (!type)
    return false;
  if (type.isF16() || type.isBF16() || type.isF32() || type.isF64())
    return true;
  if (auto intTy = dyn_cast<IntegerType>(type))
    return intTy.getWidth() == 8 || intTy.getWidth() == 16 ||
           intTy.getWidth() == 32 || intTy.getWidth() == 64;
  if (mlir::pto::isPTOFloat8Type(type))
    return true;
  if (isa<mlir::pto::HiF8Type, mlir::pto::F4E1M2x2Type,
          mlir::pto::F4E2M1x2Type>(type))
    return true;
  return false;
}

LogicalResult mlir::pto::PtrToIntOp::verify() {
  Type resultTy = getResult().getType();
  auto intTy = dyn_cast<IntegerType>(resultTy);
  if (!intTy || intTy.getWidth() != 64)
    return emitOpError("result must be i64");

  return verifyPtrLikeForAddressCast(getOperation(), getPtr().getType(),
                                     "ptr operand");
}

LogicalResult mlir::pto::IntToPtrOp::verify() {
  auto addrTy = dyn_cast<IntegerType>(getAddr().getType());
  if (!addrTy || addrTy.getWidth() != 64)
    return emitOpError("address operand must be i64");

  if (failed(verifyPtrLikeForAddressCast(getOperation(), getResult().getType(),
                                         "result")))
    return failure();

  Type dstElem = getPointerLikeElementType(getResult().getType());
  if (!isEmitCSupportedScalarType(dstElem))
    return emitOpError("result element type is not supported by EmitC: ")
           << dstElem;

  return success();
}

LogicalResult mlir::pto::LocalArrayGetOp::verify() {
  auto arrayTy = getArray().getType();
  int64_t rank = arrayTy.getRank();
  int64_t numIdx = static_cast<int64_t>(getIndices().size());
  if (numIdx != rank)
    return emitOpError() << "expects " << rank
                         << " indices for !pto.local_array of rank " << rank
                         << ", got " << numIdx;
  if (getResult().getType() != arrayTy.getElementType())
    return emitOpError()
           << "result type " << getResult().getType()
           << " does not match array element type "
           << arrayTy.getElementType();
  return success();
}

LogicalResult mlir::pto::LocalArraySetOp::verify() {
  auto arrayTy = getArray().getType();
  int64_t rank = arrayTy.getRank();
  int64_t numIdx = static_cast<int64_t>(getIndices().size());
  if (numIdx != rank)
    return emitOpError() << "expects " << rank
                         << " indices for !pto.local_array of rank " << rank
                         << ", got " << numIdx;
  if (getValue().getType() != arrayTy.getElementType())
    return emitOpError() << "value type " << getValue().getType()
                         << " does not match array element type "
                         << arrayTy.getElementType();
  return success();
}

LogicalResult mlir::pto::CastPtrOp::verify() {
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();

  auto inputPtrType = dyn_cast<mlir::pto::PtrType>(inputType);
  auto resultPtrType = dyn_cast<mlir::pto::PtrType>(resultType);
  auto inputMemRefType = dyn_cast<BaseMemRefType>(inputType);
  bool inputIsInteger = isa<IntegerType>(inputType);
  bool resultIsInteger = isa<IntegerType>(resultType);

  if (!inputPtrType && !inputMemRefType && !inputIsInteger)
    return emitOpError("input must be an integer, memref, or !pto.ptr<...>");
  if (!resultPtrType && !resultIsInteger)
    return emitOpError("result must be an integer or !pto.ptr<...>");

  if (inputIsInteger && resultIsInteger)
    return emitOpError("integer-to-integer cast is not a ptr cast");

  if (inputMemRefType && resultIsInteger)
    return emitOpError("memref-to-integer cast is unsupported");

  if (inputMemRefType && resultPtrType) {
    auto memrefSpace = dyn_cast_or_null<mlir::pto::AddressSpaceAttr>(
        inputMemRefType.getMemorySpace());
    auto resultSpace = resultPtrType.getMemorySpace();
    if (memrefSpace && memrefSpace != resultSpace)
      return emitOpError("memref-to-ptr cast must stay within the same PTO memory space");
  }

  if (inputPtrType && resultPtrType &&
      inputPtrType.getMemorySpace() != resultPtrType.getMemorySpace()) {
    return emitOpError("ptr-to-ptr cast must stay within the same PTO memory space");
  }

  return success();
}




void PTODialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "PTO/IR/PTOTypeDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "PTO/IR/PTOOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "PTO/IR/PTOAttrs.cpp.inc"
      >();
}


AddressSpaceAttr mlir::pto::getPTOAddressSpaceAttr(Type type) {
  if (auto ptrType = dyn_cast<PtrType>(type))
    return ptrType.getMemorySpace();
  auto memRefType = dyn_cast<BaseMemRefType>(type);
  if (!memRefType)
    return {};
  auto scopeAttr = dyn_cast<AddressSpaceAttr>(memRefType.getMemorySpace());
  if (!scopeAttr)
    return {};
  return scopeAttr;
}

bool mlir::pto::isScalarPtrOrMemRef(Type type) {
  if (auto pty = dyn_cast<mlir::pto::PtrType>(type))
    return static_cast<bool>(pty);
  if (auto memTy = dyn_cast<MemRefType>(type))
    return isGmAddressSpaceAttr(memTy.getMemorySpace());
  return false;
}

bool mlir::pto::hasExplicitPTOEntryAttr(func::FuncOp func) {
  return func && (func->hasAttrOfType<UnitAttr>(kPTOEntryAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kLegacyHACCEntryAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kPTOKernelAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kLegacyPTOAICoreAttrName));
}

bool mlir::pto::hasExplicitPTOEntryAttr(LLVM::LLVMFuncOp func) {
  return func && (func->hasAttrOfType<UnitAttr>(kPTOEntryAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kLegacyHACCEntryAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kPTOKernelAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kLegacyPTOAICoreAttrName));
}

bool mlir::pto::isPTOEntryFunction(func::FuncOp func) {
  if (!func || func.isDeclaration())
    return false;
  return hasExplicitPTOEntryAttr(func);
}

bool mlir::pto::isPTOEntryFunction(LLVM::LLVMFuncOp func) {
  if (!func || func.isDeclaration())
    return false;
  return hasExplicitPTOEntryAttr(func);
}

bool mlir::pto::hasExternalArtifactVisibility(func::FuncOp func) {
  if (!func || func.isDeclaration())
    return false;
  if (isPTOEntryFunction(func))
    return true;
  auto attr = func->getAttrOfType<StringAttr>(kPTOVisibilityAttrName);
  if (!attr)
    return false;
  return attr.getValue() == kPTOVisibilityExternalValue;
}

void mlir::pto::setExternalArtifactVisibility(func::FuncOp func,
                                              bool isExternal) {
  if (!func)
    return;
  if (isExternal) {
    func->setAttr(kPTOVisibilityAttrName,
                  StringAttr::get(func.getContext(),
                                  kPTOVisibilityExternalValue));
    return;
  }
  func->removeAttr(kPTOVisibilityAttrName);
}

LogicalResult mlir::pto::validatePTOEntryFunctions(ModuleOp module) {
  if (!module)
    return success();

  for (auto func : module.getOps<func::FuncOp>()) {
    if (!hasExplicitPTOEntryAttr(func))
      continue;
    if (func.isDeclaration()) {
      return func.emitOpError()
             << "`" << kPTOEntryAttrName
             << "` is only valid on function definitions";
    }
  }

  for (auto func : module.getOps<func::FuncOp>()) {
    if (!isPTOEntryFunction(func))
      continue;
    if (func.getFunctionType().getNumResults() != 0) {
      return func.emitOpError()
             << "PTO entry functions must return void";
    }
  }
  return success();
}

void mlir::pto::annotatePTOEntryFunctions(ModuleOp module) {
  (void)module;
}

//===----------------------------------------------------------------------===//
// PTO Load/Store/Addf (non-DPS polymorphic) verification + inference.
//  - If operands are memref/tensor: verify strictly.
//  - Otherwise (tile_view/tile etc): accept (so old IR can still parse).
//===----------------------------------------------------------------------===//

[[maybe_unused]] static LogicalResult verifyMemrefToTensorLoad(Operation *op, Value src, Value res) {
  auto mr = dyn_cast<MemRefType>(src.getType());
  auto rt = dyn_cast<RankedTensorType>(res.getType());
  if (!mr)
    return success(); // non-memref case: don't block old IR
  if (!rt)
    return op->emitOpError("when src is memref, result must be ranked tensor");

  if (mr.getElementType() != rt.getElementType())
    return op->emitOpError() << "memref/tensor element type mismatch: memref="
                             << mr.getElementType() << " tensor=" << rt.getElementType();

  if (mr.getRank() != rt.getRank())
    return op->emitOpError() << "rank mismatch: memref rank=" << mr.getRank()
                             << " tensor rank=" << rt.getRank();

  if (mr.hasStaticShape()) {
    if (!rt.hasStaticShape())
      return op->emitOpError("memref has static shape but result tensor is not static");
    if (mr.getShape() != rt.getShape())
      return op->emitOpError() << "shape mismatch: memref=" << mr << " tensor=" << rt;
  } else {
    // For dynamic memref dims: if tensor dim is static, allow it; if it's dynamic too, also fine.
    // We only reject when a memref static dim conflicts with tensor static dim.
    for (int64_t i = 0; i < mr.getRank(); ++i) {
      int64_t md = mr.getDimSize(i);
      int64_t td = rt.getDimSize(i);
      if (md != ShapedType::kDynamic && td != ShapedType::kDynamic && md != td)
        return op->emitOpError() << "dim mismatch at " << i << ": memref=" << md << " tensor=" << td;
    }
  }
  return success();
}

[[maybe_unused]] static LogicalResult verifyMemrefTensorStore(Operation *op, Value dst, Value src) {
  auto mr = dyn_cast<MemRefType>(dst.getType());
  if (!mr)
    return success(); // non-memref case: old tile IR allowed
  auto rt = dyn_cast<RankedTensorType>(src.getType());
  if (!rt)
    return op->emitOpError("when dst is memref, src must be ranked tensor");

  if (mr.getElementType() != rt.getElementType())
    return op->emitOpError() << "memref/tensor element type mismatch: memref="
                             << mr.getElementType() << " tensor=" << rt.getElementType();

  if (mr.getRank() != rt.getRank())
    return op->emitOpError() << "rank mismatch: memref rank=" << mr.getRank()
                             << " tensor rank=" << rt.getRank();

  for (int64_t i = 0; i < mr.getRank(); ++i) {
    int64_t md = mr.getDimSize(i);
    int64_t td = rt.getDimSize(i);
    if (md != ShapedType::kDynamic && td != ShapedType::kDynamic && md != td)
      return op->emitOpError() << "dim mismatch at " << i << ": memref=" << md << " tensor=" << td;
  }
  return success();
}

LogicalResult AllocTileOp::verify() {
  auto ty = getResult().getType(); // TileBufType

  if (failed(verifyTileBufLayoutConstraints(*this, ty, "result")))
    return failure();

  // op 上有没有传 operands
  bool hasVR = getValidRow() != nullptr;
  bool hasVC = getValidCol() != nullptr;

  // type 上的 validShape
  auto vs = ty.getValidShape();
  if (vs.size() != 2)
    return emitOpError("result tile_buf must have rank-2 validShape");

  // TileBuf valid dims use a negative sentinel (e.g. '?' / -1). Be robust to
  // any negative value (some code may materialize MLIR dynamic sentinels).
  bool needVR = (vs[0] < 0);
  bool needVC = (vs[1] < 0);

  // 你要求的：v_row=?, v_col=? 时必须同时给两个
  // （这条规则由下面两句自然实现）
  if (hasVR != needVR)
    return emitOpError() << "valid_row operand "
                         << (needVR ? "is required" : "must be absent")
                         << " because result type v_row is "
                         << (needVR ? "?" : std::to_string(vs[0]));

  if (hasVC != needVC)
    return emitOpError() << "valid_col operand "
                         << (needVC ? "is required" : "must be absent")
                         << " because result type v_col is "
                         << (needVC ? "?" : std::to_string(vs[1]));

  return success();
}

LogicalResult MaterializeTileOp::verify() {
  auto sourceTy = cast<MemRefType>(getSource().getType());
  auto resultTy = cast<TileBufType>(getResult().getType());

  if (sourceTy.getRank() != 2)
    return emitOpError("source memref must be rank-2 to materialize a tile handle");
  if (resultTy.getRank() != 2)
    return emitOpError("result tile_buf must be rank-2");
  if (failed(verifyTileBufLayoutConstraints(*this, resultTy, "result")))
    return failure();

  auto viewSemantics = (*this)->getAttrOfType<StringAttr>("pto.view_semantics");
  bool isSubview = viewSemantics && viewSemantics.getValue() == "subview";
  if (!isSubview && sourceTy.getShape() != resultTy.getShape())
    return emitOpError() << "source/result shape mismatch: source="
                         << sourceTy << " result=" << resultTy;

  if (sourceTy.getElementType() != resultTy.getElementType())
    return emitOpError() << "source/result element type mismatch: source="
                         << sourceTy.getElementType()
                         << " result=" << resultTy.getElementType();

  if (sourceTy.getMemorySpace() != resultTy.getMemorySpace())
    return emitOpError() << "source/result memory space mismatch";

  if (getConfig() != resultTy.getConfigAttr())
    return emitOpError("config attribute must match the result tile_buf config");

  auto shape = resultTy.getShape();
  auto validShape = resultTy.getValidShape();
  if (validShape.size() != 2)
    return emitOpError("result tile_buf must have rank-2 validShape");
  for (unsigned i = 0; i < 2; ++i) {
    if (shape[i] != ShapedType::kDynamic &&
        validShape[i] != ShapedType::kDynamic && validShape[i] > shape[i]) {
      return emitOpError() << "valid_shape[" << i << "] must be <= shape["
                           << i << "]";
    }
  }

  return success();
}

LogicalResult TAssignOp::verify() {
  if (getTile().getType() != getResult().getType()) {
    return emitOpError("result type must match tile operand type");
  }
  return success();
}

LogicalResult TLoadOp::verify() {
  auto verifyCommon =
      [&](bool allowLowPrecision)
      -> FailureOr<std::pair<pto::PartitionTensorViewType, pto::TileBufType>> {
    auto srcPart = dyn_cast<pto::PartitionTensorViewType>(getSrc().getType());
    auto dstTile = dyn_cast<pto::TileBufType>(getDst().getType());
    if (!srcPart || !dstTile) {
      emitOpError("expects src to be !pto.partition_tensor_view and dst to be !pto.tile_buf");
      return failure();
    }
    if (failed(verifyTileBufCommon(*this, dstTile, "dst", allowLowPrecision)))
      return failure();

    auto srcShape = srcPart.getShape();
    for (unsigned i = 0; i < srcShape.size(); ++i) {
      if (srcShape[i] != ShapedType::kDynamic && srcShape[i] <= 0) {
        emitOpError() << "expects src shape[" << i << "] to be positive";
        return failure();
      }
    }
    auto dstValid = dstTile.getValidShape();
    for (unsigned i = 0; i < dstValid.size(); ++i) {
      if (dstValid[i] != ShapedType::kDynamic && dstValid[i] < 0) {
        emitOpError() << "expects dst valid_shape[" << i << "] to be non-negative";
        return failure();
      }
    }
    return std::make_pair(srcPart, dstTile);
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/false);
    if (failed(common))
      return failure();
    auto [srcPart, dstTile] = *common;

    Type srcElem = srcPart.getElementType();
    Type dstElem = dstTile.getElementType();
    if (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem))
      return emitOpError("expects A2/A3 tload low-precision element types to be unsupported");
    if (!(dstElem.isInteger(8) || dstElem.isInteger(16) || dstElem.isInteger(32) ||
          dstElem.isInteger(64) || dstElem.isF16() || dstElem.isBF16() || dstElem.isF32()))
      return emitOpError("expects A2/A3 tload dst element type to be i8/i16/i32/i64/u64/f16/bf16/f32");

    auto dstSpace = getPTOMemorySpaceEnum(dstTile);
    if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                      *dstSpace != pto::AddressSpace::MAT))
      return emitOpError("expects A2/A3 tload dst to use loc=vec or loc=mat");

    if (getElemByteSize(srcElem) != getElemByteSize(dstElem))
      return emitOpError("expects src and dst element types to have the same bitwidth");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/true);
    if (failed(common))
      return failure();
    auto [srcPart, dstTile] = *common;

    Type srcElem = srcPart.getElementType();
    Type dstElem = dstTile.getElementType();
    unsigned srcBytes = getElemByteSize(srcElem);
    unsigned dstBytes = getElemByteSize(dstElem);
    if (srcBytes != dstBytes)
      return emitOpError("expects src and dst element types to have the same element size");
    if (!(dstBytes == 1 || dstBytes == 2 || dstBytes == 4 || dstBytes == 8))
      return emitOpError("expects A5 tload dst element size to be 1, 2, 4, or 8 bytes");
    if (!isA5TLoadStoreTransferElemType(srcElem))
      return emitOpError("expects A5 tload src element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
    if (!isA5TLoadStoreTransferElemType(dstElem))
      return emitOpError("expects A5 tload dst element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");

    if (dstElem.isInteger(64)) {
      auto pad = dstTile.getPadValueI32();
      if (pad != static_cast<int32_t>(pto::PadValue::Null) &&
          pad != static_cast<int32_t>(pto::PadValue::Zero))
        return emitOpError("expects A5 i64/u64 tload dst pad to be null or zero");
    }

    auto dstSpace = getPTOMemorySpaceEnum(dstTile);
    if (dstSpace && *dstSpace == pto::AddressSpace::VEC) {
      int32_t bl = dstTile.getBLayoutValueI32();
      int32_t sl = dstTile.getSLayoutValueI32();
      bool isND = (bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::NoneBox));
      bool isDN = (bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::NoneBox));
      bool isNZ = (bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::RowMajor));
      if (!isND && !isDN && !isNZ)
        return emitOpError("expects A5 tload vec dst layout to be ND, DN, or NZ");
    }

    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TPrefetchOp::verify() {
  auto verifyImpl = [&](bool allowLowPrecision) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();

    Type srcElem;
    Type dstElem;

    if (auto srcPart = dyn_cast<pto::PartitionTensorViewType>(srcTy)) {
      auto srcShape = srcPart.getShape();
      for (unsigned i = 0; i < srcShape.size(); ++i) {
        if (srcShape[i] != ShapedType::kDynamic && srcShape[i] <= 0)
          return emitOpError() << "expects src shape[" << i << "] to be positive";
      }
      srcElem = srcPart.getElementType();
    } else if (auto srcMr = dyn_cast<MemRefType>(srcTy)) {
      if (!srcMr.hasRank())
        return emitOpError("expects src memref to be ranked");
      for (int64_t dim : srcMr.getShape()) {
        if (dim != ShapedType::kDynamic && dim <= 0)
          return emitOpError("expects src memref shape to be positive");
      }
      srcElem = srcMr.getElementType();
    } else {
      return emitOpError("expects src to be !pto.partition_tensor_view or memref");
    }

    if (auto dstTile = dyn_cast<pto::TileBufType>(dstTy)) {
      if (failed(verifyTileBufCommon(*this, dstTile, "dst", allowLowPrecision)))
        return failure();
      auto dstValid = dstTile.getValidShape();
      for (unsigned i = 0; i < dstValid.size(); ++i) {
        if (dstValid[i] != ShapedType::kDynamic && dstValid[i] < 0)
          return emitOpError() << "expects dst valid_shape[" << i
                               << "] to be non-negative";
      }
      auto dstSpace = getPTOMemorySpaceEnum(dstTile);
      if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                        *dstSpace != pto::AddressSpace::MAT))
        return emitOpError("expects dst to use loc=vec or loc=mat");
      dstElem = dstTile.getElementType();
    } else if (auto dstMr = dyn_cast<MemRefType>(dstTy)) {
      auto dstSpace = getPTOMemorySpaceEnum(dstMr);
      if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                        *dstSpace != pto::AddressSpace::MAT))
        return emitOpError("expects dst memref to use loc=vec or loc=mat");
      if (!dstMr.hasRank())
        return emitOpError("expects dst memref to be ranked");
      if (failed(verifyTileBufCommon(*this, dstMr, "dst", allowLowPrecision)))
        return failure();
      dstElem = dstMr.getElementType();
    } else {
      return emitOpError("expects dst to be !pto.tile_buf or memref");
    }

    if (getElemByteSize(srcElem) != getElemByteSize(dstElem))
      return emitOpError("expects src and dst element types to have the same element size");
    if (!allowLowPrecision &&
        (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem)))
      return emitOpError("expects A2/A3 tprefetch low-precision element types to be unsupported");
    if (allowLowPrecision &&
        (!isA5TLoadStoreTransferElemType(srcElem) ||
         !isA5TLoadStoreTransferElemType(dstElem)))
      return emitOpError("expects A5 tprefetch element types to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyImpl(/*allowLowPrecision=*/false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyImpl(/*allowLowPrecision=*/true);
  };
  switch (getVerifierTargetArch(getOperation())) {
  case VerifierTargetArch::A2A3:
    return verifyA2A3();
  case VerifierTargetArch::A5:
    return verifyA5();
  }
  return failure();
}

LogicalResult MakePrefetchAsyncContextOp::verify() {
  Type workspaceTy = getWorkspace().getType();
  Type elemTy = nullptr;
  if (auto ptrTy = dyn_cast<pto::PtrType>(workspaceTy)) {
    elemTy = ptrTy.getElementType();
  } else if (auto memTy = dyn_cast<MemRefType>(workspaceTy)) {
    if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
      return emitOpError("expects workspace memref to be in GM address space");
    elemTy = memTy.getElementType();
  } else {
    return emitOpError("expects workspace to be !pto.ptr<i8> or GM memref<i8>");
  }
  if (!isByteIntegerType(elemTy))
    return emitOpError("expects workspace element type to be an 8-bit integer");
  return success();
}

LogicalResult TPrefetchAsyncOp::verify() {
  if (failed(verifyAsyncFlatContiguous1DGMViewLike(getOperation(), getSrc(),
                                                   "src")))
    return failure();
  return success();
}

LogicalResult mlir::pto::SetFFTsOp::verify() {
  auto mr = llvm::dyn_cast<mlir::MemRefType>(getFfts().getType());
  if (!mr)
    return emitOpError("expects a memref operand");

  if (!mr.getElementType().isInteger(64) && !mr.getElementType().isInteger(8))
    return emitOpError("expects element type i64 (or i8)");

  return mlir::success();
}

ParseResult mlir::pto::SyncSetOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                SyncSetOp::getPipeAttrName(result.name),
                                SyncSetOp::getEventIdAttrName(result.name));
}

void mlir::pto::SyncSetOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

LogicalResult mlir::pto::SyncSetOp::verify() {
  bool hasStatic = getEventIdAttr() != nullptr;
  bool hasDynamic = static_cast<bool>(getEventIdDyn());
  if (hasStatic == hasDynamic)
    return emitOpError()
           << "expects exactly one event-id form: static attr or dynamic index operand";
  if (IntegerAttr fftsModeAttr = getFftsModeAttr()) {
    int64_t fftsMode = fftsModeAttr.getInt();
    if (fftsMode < 0 || fftsMode > 2)
      return emitOpError() << "requires ffts_mode in range [0, 2], but got "
                           << fftsMode;
  }

  auto verifyA2A3 = [&]() -> LogicalResult { return success(); };
  auto verifyA5 = [&]() -> LogicalResult {
    switch (getPipe().getPipe()) {
    case PIPE::PIPE_FIX:
    case PIPE::PIPE_MTE3:
      return success();
    default:
      return emitOpError()
             << "A5 sync.set expects pipe to be one of <PIPE_FIX>, <PIPE_MTE3>";
    }
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

ParseResult mlir::pto::SyncWaitOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                SyncWaitOp::getPipeAttrName(result.name),
                                SyncWaitOp::getEventIdAttrName(result.name));
}

void mlir::pto::SyncWaitOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

ParseResult mlir::pto::SyncAllOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  SmallVector<Type, 4> operandTypes;
  Attribute modeAttr;
  Attribute coreTypeAttr;

  if (parser.parseLParen())
    return failure();

  if (failed(parser.parseOptionalRParen())) {
    if (parser.parseOperandList(operands) || parser.parseColonTypeList(operandTypes) ||
        parser.parseRParen())
      return failure();
    if (operands.size() != operandTypes.size())
      return parser.emitError(parser.getCurrentLocation())
             << "expects the same number of operands and operand types";
  }

  if (parser.parseKeyword("mode") || parser.parseEqual() ||
      parser.parseAttribute(modeAttr) || parser.parseComma() ||
      parser.parseKeyword("core_type") || parser.parseEqual() ||
      parser.parseAttribute(coreTypeAttr))
    return failure();

  auto mode = dyn_cast<pto::SyncAllModeAttr>(modeAttr);
  if (!mode)
    return parser.emitError(parser.getCurrentLocation())
           << "expects mode to be #pto.sync_all_mode<...>";

  auto coreType = dyn_cast<pto::SyncCoreTypeAttr>(coreTypeAttr);
  if (!coreType)
    return parser.emitError(parser.getCurrentLocation())
           << "expects core_type to be #pto.sync_core_type<...>";

  result.addAttribute("mode", mode);
  result.addAttribute("core_type", coreType);

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  auto addSegmentSizes = [&](int32_t gm, int32_t ub, int32_t l1,
                             int32_t used) {
    result.addAttribute("operandSegmentSizes",
                        parser.getBuilder().getDenseI32ArrayAttr(
                            {gm, ub, l1, used}));
  };

  switch (mode.getValue()) {
  case pto::SyncAllMode::Hard:
    if (!operands.empty())
      return parser.emitError(parser.getCurrentLocation())
             << "expects hard syncall to have no operands";
    addSegmentSizes(0, 0, 0, 0);
    return success();
  case pto::SyncAllMode::Soft:
    break;
  }

  switch (coreType.getValue()) {
  case pto::SyncCoreType::AIVOnly:
    if (operands.size() != 2 && operands.size() != 3)
      return parser.emitError(parser.getCurrentLocation())
             << "expects soft AIV-only syncall to have gm_workspace, "
                "ub_workspace, and optional used_cores";
    if (parser.resolveOperand(operands[0], operandTypes[0], result.operands) ||
        parser.resolveOperand(operands[1], operandTypes[1], result.operands))
      return failure();
    if (operands.size() == 3 &&
        parser.resolveOperand(operands[2], operandTypes[2], result.operands))
      return failure();
    addSegmentSizes(1, 1, 0, operands.size() == 3 ? 1 : 0);
    return success();
  case pto::SyncCoreType::AICOnly:
    if (operands.size() != 2 && operands.size() != 3)
      return parser.emitError(parser.getCurrentLocation())
             << "expects soft AIC-only syncall to have gm_workspace, "
                "l1_workspace, and optional used_cores";
    if (parser.resolveOperand(operands[0], operandTypes[0], result.operands) ||
        parser.resolveOperand(operands[1], operandTypes[1], result.operands))
      return failure();
    if (operands.size() == 3 &&
        parser.resolveOperand(operands[2], operandTypes[2], result.operands))
      return failure();
    addSegmentSizes(1, 0, 1, operands.size() == 3 ? 1 : 0);
    return success();
  case pto::SyncCoreType::Mix:
    if (operands.size() != 3 && operands.size() != 4)
      return parser.emitError(parser.getCurrentLocation())
             << "expects soft mixed syncall to have gm_workspace, "
                "ub_workspace, l1_workspace, and optional used_cores";
    if (parser.resolveOperand(operands[0], operandTypes[0], result.operands) ||
        parser.resolveOperand(operands[1], operandTypes[1], result.operands) ||
        parser.resolveOperand(operands[2], operandTypes[2], result.operands))
      return failure();
    if (operands.size() == 4 &&
        parser.resolveOperand(operands[3], operandTypes[3], result.operands))
      return failure();
    addSegmentSizes(1, 1, 1, operands.size() == 4 ? 1 : 0);
    return success();
  }

  llvm_unreachable("unhandled SyncCoreType");
}

void mlir::pto::SyncAllOp::print(OpAsmPrinter &p) {
  SmallVector<Value, 4> operands;
  if (getGmWorkspace())
    operands.push_back(getGmWorkspace());
  if (getUbWorkspace())
    operands.push_back(getUbWorkspace());
  if (getL1Workspace())
    operands.push_back(getL1Workspace());
  if (getUsedCores())
    operands.push_back(getUsedCores());

  p << "(";
  if (!operands.empty()) {
    p.printOperands(operands);
    p << " : ";
    llvm::interleaveComma(operands, p,
                          [&](Value operand) { p.printType(operand.getType()); });
  }
  p << ") mode = " << getMode() << ", core_type = " << getCoreType();
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes", "mode",
                                           "core_type"});
}

LogicalResult mlir::pto::SyncWaitOp::verify() {
  bool hasStatic = getEventIdAttr() != nullptr;
  bool hasDynamic = static_cast<bool>(getEventIdDyn());
  if (hasStatic == hasDynamic)
    return emitOpError()
           << "expects exactly one event-id form: static attr or dynamic index operand";

  auto verifyA2A3 = [&]() -> LogicalResult { return success(); };
  auto verifyA5 = [&]() -> LogicalResult {
    switch (getPipe().getPipe()) {
    case PIPE::PIPE_FIX:
    case PIPE::PIPE_MTE1:
    case PIPE::PIPE_MTE2:
    case PIPE::PIPE_MTE3:
    case PIPE::PIPE_V:
      return success();
    default:
      return emitOpError() << "A5 sync.wait expects pipe to be one of "
                              "<PIPE_FIX>, <PIPE_MTE1>, <PIPE_MTE2>, "
                              "<PIPE_MTE3>, <PIPE_V>";
    }
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TStoreOp::verify() {
  auto verifyCommon =
      [&](bool allowLowPrecision)
      -> FailureOr<std::pair<pto::TileBufType, pto::PartitionTensorViewType>> {
    auto srcTile = dyn_cast<pto::TileBufType>(getSrc().getType());
    auto dstPart = dyn_cast<pto::PartitionTensorViewType>(getDst().getType());
    if (!srcTile || !dstPart) {
      emitOpError("expects src to be !pto.tile_buf and dst to be !pto.partition_tensor_view");
      return failure();
    }
    if (failed(verifyTileBufCommon(*this, srcTile, "src", allowLowPrecision)))
      return failure();
    for (auto [idx, dim] : llvm::enumerate(dstPart.getShape())) {
      if (dim != ShapedType::kDynamic && dim <= 0) {
        emitOpError() << "expects dst shape[" << idx << "] to be positive";
        return failure();
      }
    }
    auto srcValid = srcTile.getValidShape();
    for (auto [idx, dim] : llvm::enumerate(srcValid)) {
      if (dim != ShapedType::kDynamic && dim < 0) {
        emitOpError() << "expects src valid_shape[" << idx << "] to be non-negative";
        return failure();
      }
    }

    // Keep TSTORE contract explicit while preserving existing legal layout
    // reinterpretation paths (e.g. 1x1024 <-> 32x32, 5D partition views).
    // When both sides are fully static, require equal element counts between
    // dst shape and src valid_shape.
    auto getStaticElemCount = [](ArrayRef<int64_t> shape) -> std::optional<int64_t> {
      int64_t total = 1;
      for (int64_t dim : shape) {
        if (dim == ShapedType::kDynamic)
          return std::nullopt;
        if (dim <= 0)
          return std::nullopt;
        if (total > std::numeric_limits<int64_t>::max() / dim)
          return std::nullopt;
        total *= dim;
      }
      return total;
    };

    auto dstElemCount = getStaticElemCount(dstPart.getShape());
    auto srcValidElemCount = getStaticElemCount(srcValid);
    if (dstElemCount && srcValidElemCount && *dstElemCount != *srcValidElemCount) {
      emitOpError() << "expects dst static element count (" << *dstElemCount
                    << ") to match src valid_shape static element count ("
                    << *srcValidElemCount << ")";
      return failure();
    }
    return std::make_pair(srcTile, dstPart);
  };

  auto isLoadStoreElemType = [&](Type ty) -> bool {
    return ty.isInteger(8) || ty.isInteger(16) || ty.isInteger(32) ||
           ty.isInteger(64) || ty.isF16() || ty.isBF16() || ty.isF32();
  };
  auto isI8Like = [&](Type ty) -> bool { return ty.isInteger(8); };
  bool hasPreQuant = static_cast<bool>(getPreQuantScalar());
  auto reluMode = getReluPreMode();

  auto verifyA2A3 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/false);
    if (failed(common))
      return failure();
    auto [srcTile, dstPart] = *common;
    auto srcSpace = getPTOMemorySpaceEnum(srcTile);
    if (!srcSpace || (*srcSpace != pto::AddressSpace::VEC &&
                      *srcSpace != pto::AddressSpace::MAT &&
                      *srcSpace != pto::AddressSpace::ACC))
      return emitOpError("expects A2/A3 tstore src to use loc=vec, loc=mat, or loc=acc");
    if (hasPreQuant && *srcSpace != pto::AddressSpace::ACC)
      return emitOpError("expects preQuantScalar form to use loc=acc src");
    if (reluMode != pto::ReluPreMode::NoRelu && *srcSpace != pto::AddressSpace::ACC)
      return emitOpError("expects reluPreMode form to use loc=acc src");

    Type srcElem = srcTile.getElementType();
    Type dstElem = dstPart.getElementType();
    if (*srcSpace == pto::AddressSpace::VEC || *srcSpace == pto::AddressSpace::MAT) {
      if (hasPreQuant)
        return emitOpError("expects preQuantScalar form to use loc=acc src");
      if (isPTOLowPrecisionType(dstElem))
        return emitOpError("expects A2/A3 vec/mat tstore low-precision dst element types to be unsupported");
      if (!isLoadStoreElemType(srcElem))
        return emitOpError("expects A2/A3 vec/mat tstore src element type to be i8/i16/i32/i64/u64/f16/bf16/f32");
      if (getElemByteSize(srcElem) != getElemByteSize(dstElem))
        return emitOpError("expects A2/A3 vec/mat tstore src and dst element types to have the same bitwidth");
      return success();
    }

    if (!(srcElem.isInteger(32) || srcElem.isF32()))
      return emitOpError("expects A2/A3 acc tstore src element type to be i32 or f32");
    if (hasPreQuant) {
      if (srcElem.isInteger(32)) {
        if (!(isI8Like(dstElem) || dstElem.isF16()))
          return emitOpError("expects A2/A3 acc preQuantScalar tstore dst type to be i8/ui8/f16");
      } else if (srcElem.isF32()) {
        if (!isI8Like(dstElem))
          return emitOpError("expects A2/A3 acc preQuantScalar tstore dst type to be i8/ui8");
      }
    } else {
      if (!(dstElem.isInteger(32) || dstElem.isF32() || dstElem.isF16() ||
            dstElem.isBF16()))
        return emitOpError("expects A2/A3 acc tstore dst element type to be i32/f32/f16/bf16");
    }

    auto srcShape = srcTile.getShape();
    if (srcShape[1] != ShapedType::kDynamic &&
        (srcShape[1] < 1 || srcShape[1] > 4095))
      return emitOpError("expects A2/A3 acc tstore src cols to be in [1, 4095]");
    auto srcValid = srcTile.getValidShape();
    if (srcValid[1] != ShapedType::kDynamic &&
        (srcValid[1] < 0 || srcValid[1] > 4095))
      return emitOpError("expects A2/A3 acc tstore src valid_shape[1] to be in [0, 4095]");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/true);
    if (failed(common))
      return failure();
    auto [srcTile, dstPart] = *common;
    auto srcSpace = getPTOMemorySpaceEnum(srcTile);
    if (!srcSpace || (*srcSpace != pto::AddressSpace::VEC &&
                      *srcSpace != pto::AddressSpace::ACC))
      return emitOpError("expects A5 tstore src to use loc=vec or loc=acc");
    if (hasPreQuant && *srcSpace != pto::AddressSpace::ACC)
      return emitOpError("expects preQuantScalar form to use loc=acc src");
    if (reluMode != pto::ReluPreMode::NoRelu && *srcSpace != pto::AddressSpace::ACC)
      return emitOpError("expects reluPreMode form to use loc=acc src");

    Type srcElem = srcTile.getElementType();
    Type dstElem = dstPart.getElementType();
    if (*srcSpace == pto::AddressSpace::VEC) {
      if (hasPreQuant)
        return emitOpError("expects preQuantScalar form to use loc=acc src");
      if (!isA5TLoadStoreTransferElemType(srcElem))
        return emitOpError("expects A5 vec tstore src element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
      if (getElemByteSize(srcElem) != getElemByteSize(dstElem))
        return emitOpError("expects A5 vec tstore src and dst element types to have the same bitwidth");

      int32_t bl = srcTile.getBLayoutValueI32();
      int32_t sl = srcTile.getSLayoutValueI32();
      bool isND = (bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::NoneBox));
      bool isDN = (bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::NoneBox));
      bool isNZ = (bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::RowMajor));
      auto srcShape = srcTile.getShape();
      bool isSpecialCase = (srcShape.size() == 2 && (srcShape[0] == 1 || srcShape[1] == 1));
      if (!isSpecialCase && !isND && !isDN && !isNZ)
        return emitOpError("expects A5 vec tstore src layout to be ND, DN, or NZ (or special case with 1 row/col)");
      return success();
    }

    if (!(srcElem.isInteger(32) || srcElem.isF32()))
      return emitOpError("expects A5 acc tstore src element type to be i32 or f32");
    if (hasPreQuant) {
      if (!isA5AccStorePreQuantDstType(srcElem, dstElem))
        return emitOpError("expects A5 acc preQuantScalar tstore dst type to be i8/ui8/f16/bf16/f32/hif8/f8E4M3");
    } else {
      if (!(dstElem.isInteger(32) || dstElem.isF32() || dstElem.isF16() ||
            dstElem.isBF16()))
        return emitOpError("expects A5 acc tstore dst element type to be i32/f32/f16/bf16");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAbsOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
      failed(verifyVecTileCommon(*this, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
    return failure();

  Type elemTy;
  if (auto tb = dyn_cast<pto::TileBufType>(srcTy))
    elemTy = tb.getElementType();
  else if (auto mr = dyn_cast<MemRefType>(srcTy))
    elemTy = mr.getElementType();
  if (!(elemTy.isF16() || elemTy.isF32()))
    return emitOpError() << "expects element type to be f16 or f32";

  return success();
}
// PTO.cpp

static bool isPTOShapedLike(Type ty) {
  return mlir::isa<MemRefType, RankedTensorType,
                pto::TensorViewType, pto::TileBufType,
                pto::PartitionTensorViewType>(ty);
}

static bool isTileLikeType(Type ty) {
  return isa<pto::TileBufType, MemRefType>(ty);
}

static Type getElemTy(Type ty) {
  if (auto mr = mlir::dyn_cast<MemRefType>(ty)) return mr.getElementType();
  if (auto tt = mlir::dyn_cast<RankedTensorType>(ty)) return tt.getElementType();
  if (auto tv = mlir::dyn_cast<pto::TensorViewType>(ty)) return tv.getElementType();
  if (auto tb = mlir::dyn_cast<pto::TileBufType>(ty)) return tb.getElementType();
  if (auto tv = mlir::dyn_cast<pto::PartitionTensorViewType>(ty)) return tv.getElementType();
  return Type();
}

static SmallVector<int64_t, 4> getShapeVec(Type ty) {
  SmallVector<int64_t, 4> s;
  if (auto mr = mlir::dyn_cast<MemRefType>(ty))
    return SmallVector<int64_t,4>(mr.getShape().begin(), mr.getShape().end());
  if (auto tt = mlir::dyn_cast<RankedTensorType>(ty))
    return SmallVector<int64_t,4>(tt.getShape().begin(), tt.getShape().end());
  if (auto tv = mlir::dyn_cast<pto::TensorViewType>(ty))
    return SmallVector<int64_t,4>(tv.getShape().begin(), tv.getShape().end());
  if (auto tb = mlir::dyn_cast<pto::TileBufType>(ty))
    return SmallVector<int64_t,4>(tb.getShape().begin(), tb.getShape().end());
  if (auto tv = mlir::dyn_cast<pto::PartitionTensorViewType>(ty))
    return SmallVector<int64_t,4>(tv.getShape().begin(), tv.getShape().end());
  return {};
}

static SmallVector<int64_t, 4> getValidShapeVec(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty))
    return SmallVector<int64_t, 4>(tb.getValidShape().begin(), tb.getValidShape().end());
  return getShapeVec(ty);
}

static int64_t getLogicalTileDim(int64_t rawDim, Type elemTy,
                                 std::optional<pto::BLayout> blayout,
                                 unsigned dimIdx) {
  if (rawDim == ShapedType::kDynamic || !isPTOFloat4PackedType(elemTy))
    return rawDim;
  pto::BLayout layout = blayout.value_or(pto::BLayout::RowMajor);
  unsigned packedDim = layout == pto::BLayout::ColMajor ? 0 : 1;
  return dimIdx == packedDim ? rawDim * 2 : rawDim;
}

static std::optional<pto::BLayout> getTileBufBLayout(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty))
    return static_cast<pto::BLayout>(tb.getBLayoutValueI32());
  return std::nullopt;
}

static SmallVector<int64_t, 4> getLogicalTileExtentVec(Type ty,
                                                       bool useValidShape) {
  SmallVector<int64_t, 4> dims =
      useValidShape ? getValidShapeVec(ty) : getShapeVec(ty);
  if (!isTileLikeType(ty) || dims.size() != 2)
    return dims;

  Type elemTy = getElemTy(ty);
  auto blayout = getTileBufBLayout(ty);
  for (unsigned i = 0; i < dims.size(); ++i)
    dims[i] = getLogicalTileDim(dims[i], elemTy, blayout, i);
  return dims;
}

static int64_t getConstantIndexOrDynamic(Value value) {
  if (!value)
    return ShapedType::kDynamic;
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  return ShapedType::kDynamic;
}

static SmallVector<int64_t, 4> getValidShapeVec(Value value) {
  if (!value)
    return {};
  auto valid = getValidShapeVec(value.getType());
  if (auto bind = value.getDefiningOp<pto::BindTileOp>()) {
    if (valid.size() >= 1 && bind.getValidRow())
      valid[0] = getConstantIndexOrDynamic(bind.getValidRow());
    if (valid.size() >= 2 && bind.getValidCol())
      valid[1] = getConstantIndexOrDynamic(bind.getValidCol());
  }
  return valid;
}

static SmallVector<int64_t, 4> getMatmulLogicalShapeVec(Type ty) {
  auto shape = getShapeVec(ty);
  auto valid = getValidShapeVec(ty);
  if (!isa<pto::TileBufType>(ty) || shape.size() != valid.size())
    return shape;

  for (size_t i = 0, e = shape.size(); i < e; ++i) {
    if (valid[i] != ShapedType::kDynamic)
      shape[i] = valid[i];
  }
  return shape;
}

static bool isByteIntegerType(Type ty) {
  auto intTy = dyn_cast<IntegerType>(ty);
  return intTy && intTy.getWidth() == 8;
}

static LogicalResult verifyAsyncFlatContiguous1DGMMemRef(Operation *op,
                                                         Value value,
                                                         StringRef name) {
  auto memTy = dyn_cast<MemRefType>(value.getType());
  if (!memTy)
    return op->emitOpError() << "expects " << name << " to be a memref";
  if (!memTy.hasRank())
    return op->emitOpError() << "expects " << name << " to be a ranked memref";
  if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
    return op->emitOpError() << "expects " << name
                             << " to be in GM address space";

  ArrayRef<int64_t> shape = memTy.getShape();
  if (shape.empty())
    return op->emitOpError() << "expects " << name
                             << " to have rank >= 1";
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic)
      return op->emitOpError() << "expects " << name
                               << " to have a static shape";
  }

  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (failed(getStridesAndOffset(memTy, strides, offset)))
    return op->emitOpError() << "expects " << name
                             << " to be a strided memref with a known layout";

  bool hasDynamicLayout =
      offset == ShapedType::kDynamic ||
      llvm::any_of(strides, [](int64_t stride) {
        return stride == ShapedType::kDynamic;
      });
  if (hasDynamicLayout)
    return success();

  bool packed = !strides.empty() && strides.back() == 1;
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0 && packed; --i)
    packed &= strides[i] == strides[i + 1] * shape[i + 1];
  if (!packed)
    return op->emitOpError()
           << "expects " << name
           << " to be a static flat contiguous logical 1D GM memref";

  bool logical1D = true;
  for (int i = 0, e = static_cast<int>(shape.size()) - 1; i < e; ++i)
    logical1D &= shape[i] == 1;
  if (!logical1D)
    return op->emitOpError()
           << "expects " << name
           << " to be a static flat contiguous logical 1D GM memref";

  return success();
}

static LogicalResult verifyAsyncFlatContiguous1DGMViewLike(Operation *op,
                                                           Value value,
                                                           StringRef name) {
  Type ty = value.getType();
  if (isa<MemRefType>(ty))
    return verifyAsyncFlatContiguous1DGMMemRef(op, value, name);

  if (!isa<pto::TensorViewType, pto::PartitionTensorViewType>(ty))
    return op->emitOpError() << "expects " << name
                             << " to be a memref/tensor_view/partition_view";

  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty())
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic)
      return op->emitOpError() << "expects " << name
                               << " to have a static shape";
  }

  bool logical1D = true;
  for (int i = 0, e = static_cast<int>(shape.size()) - 1; i < e; ++i)
    logical1D &= shape[i] == 1;
  if (!logical1D)
    return op->emitOpError()
           << "expects " << name
           << " to be a static flat contiguous logical 1D GM view";

  return success();
}

static bool isCommGlobalLikeType(Type ty) {
  if (auto memTy = dyn_cast<MemRefType>(ty))
    return isGmAddressSpaceAttr(memTy.getMemorySpace());
  return isa<pto::TensorViewType, pto::PartitionTensorViewType>(ty);
}

static LogicalResult verifyCommGlobalLike(Operation *op, Value value,
                                          StringRef name) {
  Type ty = value.getType();
  if (!isCommGlobalLikeType(ty))
    return op->emitOpError() << "expects " << name
                             << " to be a GM memref/tensor_view/partition_view";

  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty())
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim <= 0)
      return op->emitOpError() << "expects " << name
                               << " to have a positive static shape";
  }
  return success();
}

static LogicalResult verifyCommSignalLike(Operation *op, Value value,
                                          StringRef name) {
  if (failed(verifyCommGlobalLike(op, value, name)))
    return failure();
  Type elemTy = getElemTy(value.getType());
  if (!elemTy || !elemTy.isSignlessInteger(32))
    return op->emitOpError() << "expects " << name
                             << " element type to be i32";
  return success();
}

static LogicalResult verifyCommStagingTileLike(Operation *op, Value value,
                                               StringRef name) {
  Type ty = value.getType();
  if (!isa<pto::TileBufType, MemRefType>(ty))
    return op->emitOpError() << "expects " << name
                             << " to be a tile_buf or memref tile";
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name
                             << " to be in vec address space";
  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty())
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim <= 0)
      return op->emitOpError() << "expects " << name
                               << " to have a positive static shape";
  }
  return success();
}

static LogicalResult verifyCommGlobalGroup(Operation *op, ValueRange group,
                                           StringRef name) {
  if (group.empty())
    return op->emitOpError() << "expects at least one " << name << " operand";
  Type groupTy = group.front().getType();
  for (auto it : llvm::enumerate(group)) {
    if (failed(verifyCommGlobalLike(op, it.value(),
                                    (name + "[" + Twine(it.index()) + "]").str())))
      return failure();
    if (it.value().getType() != groupTy)
      return op->emitOpError() << "expects all " << name
                               << " operands to have identical types";
  }
  return success();
}

static LogicalResult verifyCommPingPongSameType(Operation *op, Value ping,
                                                Value pong, StringRef pingName,
                                                StringRef pongName) {
  if (!pong)
    return success();
  if (failed(verifyCommStagingTileLike(op, ping, pingName)) ||
      failed(verifyCommStagingTileLike(op, pong, pongName)))
    return failure();
  if (ping.getType() != pong.getType())
    return op->emitOpError() << "expects " << pingName << " and " << pongName
                             << " to have identical types";
  return success();
}

static std::optional<uint64_t> getStaticByteSize(Type ty) {
  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty())
    return std::nullopt;
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim < 0)
      return std::nullopt;
  }

  Type elemTy = getElemTy(ty);
  uint64_t elemBytes = getElemByteSize(elemTy);
  if (elemBytes == 0)
    return std::nullopt;

  uint64_t total = elemBytes;
  for (int64_t dim : shape) {
    total *= static_cast<uint64_t>(dim);
  }
  return total;
}

static std::optional<pto::AddressSpace> getPTOMemorySpaceEnum(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (auto as = dyn_cast_or_null<pto::AddressSpaceAttr>(tb.getMemorySpace()))
      return as.getAddressSpace();
    return std::nullopt;
  }
  if (auto mr = dyn_cast<MemRefType>(ty)) {
    if (auto as = dyn_cast_or_null<pto::AddressSpaceAttr>(mr.getMemorySpace()))
      return as.getAddressSpace();
    if (!mr.getMemorySpace())
      return pto::AddressSpace::GM;
  }
  return std::nullopt;
}

[[maybe_unused]] static bool isRank2TileBuf(Type ty) {
  auto tb = dyn_cast<pto::TileBufType>(ty);
  return tb && tb.getRank() == 2 && tb.getValidShape().size() == 2;
}

static bool isSupportedVecElemType(Type ty, bool allowBf16,
                                   bool allowInt8) {
  if (ty.isF16() || ty.isF32())
    return true;
  if (allowBf16 && ty.isBF16())
    return true;
  if (auto it = dyn_cast<IntegerType>(ty)) {
    switch (it.getWidth()) {
    case 32:
    case 16:
      return true;
    case 8:
      return allowInt8;
    default:
      return false;
    }
  }
  return false;
}

static bool isSupportedMGatherMScatterIndexElemType(Type ty) {
  auto it = dyn_cast<IntegerType>(ty);
  if (!it || it.getWidth() != 32)
    return false;
  return true;
}

static bool isSupportedMGatherMScatterPayloadElemType(Operation *op, Type ty) {
  if (isSupportedVecElemType(ty, /*allowBf16=*/true, /*allowInt8=*/true))
    return true;
  if (!isTargetArchA5(op))
    return false;
  if (isPTOHiFloat8Type(ty))
    return true;
  return ty.isFloat8E4M3() || ty.isFloat8E4M3FN() || ty.isFloat8E4M3FNUZ() ||
         ty.isFloat8E4M3B11FNUZ() || ty.isFloat8E5M2() || ty.isFloat8E5M2FNUZ();
}

static bool isSupportedMScatterAtomicPayloadElemType(Type ty,
                                                     pto::ScatterAtomicOp atomic) {
  auto intTy = dyn_cast<IntegerType>(ty);
  switch (atomic) {
  case pto::ScatterAtomicOp::None:
    return true;
  case pto::ScatterAtomicOp::Add:
    return ty.isF16() || ty.isF32() ||
           (intTy && intTy.getWidth() == 32);
  case pto::ScatterAtomicOp::Max:
  case pto::ScatterAtomicOp::Min:
    return ty.isF32() ||
           (intTy && intTy.getWidth() == 32);
  }
  llvm_unreachable("Unknown ScatterAtomicOp");
}

static LogicalResult verifyMGatherMScatterMemOperand(Operation *op,
                                                     Value memValue,
                                                     Type dataElemTy,
                                                     StringRef dataOperandLabel) {
  Type memTy = memValue.getType();
  Type memElem = getElemTy(memTy);
  if (!memElem || memElem != dataElemTy)
    return op->emitOpError() << "expects mem element type to match "
                             << dataOperandLabel << " element type";

  if (isa<pto::PartitionTensorViewType>(memTy)) {
    if (auto layout = getLogicalViewLayout(memValue)) {
      if (*layout != pto::Layout::ND)
        return op->emitOpError(
            "expects mem partition view to use ND logical layout when layout "
            "can be inferred");
    }
    return success();
  }

  if (auto mr = dyn_cast<MemRefType>(memTy)) {
    auto as = getPTOMemorySpaceEnum(mr);
    if (!as || (*as != pto::AddressSpace::GM &&
                 *as != pto::AddressSpace::Zero))
      return op->emitOpError(
          "expects mem memref to use GM or zero address space");
    if (mr.getRank() == 5) {
      auto shape = mr.getShape();
      bool allStatic = true;
      for (int64_t d : shape)
        if (d == ShapedType::kDynamic)
          allStatic = false;
      if (allStatic && (shape[0] != 1 || shape[1] != 1 || shape[2] != 1))
        return op->emitOpError(
            "expects rank-5 GM memref leading dimensions to be [1,1,1,...] "
            "(GlobalTensor table shape)");
    }
    return success();
  }

  return op->emitOpError(
      "expects mem to be !pto.partition_tensor_view or a GM/ZERO memref");
}

static bool hasCompatibleKnownExtent(int64_t lhs, int64_t rhs);
static bool isKnownUnitExtent(int64_t value);
static bool isKnownZeroOrUnitExtent(int64_t value);
static bool hasCompatibleKnownExtentOrZero(int64_t lhs, int64_t rhs);
