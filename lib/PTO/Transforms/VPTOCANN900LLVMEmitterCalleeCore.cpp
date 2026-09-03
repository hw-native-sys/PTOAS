// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "VPTOCANN900LLVMEmitterInternal.h"

namespace mlir::pto::detail {

FailureOr<StringRef> buildCarryBinaryCallee(MLIRContext *context, Type resultType, StringRef stem) {
  std::string vec = getElementTypeFragment(cast<pto::VRegType>(resultType).getElementType());
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm." + stem.str() + ".v" + std::to_string(*lanes) + vec).getValue();
}

FailureOr<StringRef> buildVselCallee(MLIRContext *context, Type resultType) {
  std::string vec = getCANN900VectorTypeFragment(resultType);
  if (vec.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vsel." + vec).getValue();
}

FailureOr<StringRef> buildVselrCallee(MLIRContext *context, Type resultType) {
  Type elementType = getElementTypeFromVectorLike(resultType);
  auto lanes = getElementCountFromVectorLike(resultType);
  if (!elementType || !lanes) {
    return failure();
  }

  std::optional<LowpPayloadABI> abi = getLowpPayloadABI(elementType, context);
  std::string vec = abi ? "v" + std::to_string(*lanes) + abi->intrinsicElementFragment.str()
                        : getCANN900VectorTypeFragment(resultType);
  if (vec.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vselr." + vec).getValue();
}

FailureOr<StringRef> buildVdupCallee(MLIRContext *context, pto::VdupOp op) {
  Type inputType = op.getInput().getType();
  Type resultType = op.getResult().getType();
  std::string vec = getCANN900VectorTypeFragment(resultType);
  if (vec.empty()) {
    return failure();
  }

  if (isa<VectorType, pto::VRegType>(inputType)) {
    StringRef position = op.getPosition().value_or("LOWEST");
    StringRef family = position == "HIGHEST" ? "vdupm" : "vdup";
    return StringAttr::get(context, "llvm.hivm." + family.str() + ".z." + vec).getValue();
  }

  return StringAttr::get(context, "llvm.hivm.vdups.z." + vec).getValue();
}

FailureOr<StringRef> buildVbrCallee(MLIRContext *context, Type resultType) {
  std::string vec = getCANN900VectorTypeFragment(resultType);
  if (vec.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vbr." + vec).getValue();
}

FailureOr<StringRef> buildPstuCallee(MLIRContext *context, pto::PstuOp op) {
  if (auto maskType = dyn_cast<pto::MaskType>(op.getValue().getType())) {
    if (maskType.isB16()) {
      return StringAttr::get(context, "llvm.hivm.pstu.b16").getValue();
    }
    if (maskType.isB32()) {
      return StringAttr::get(context, "llvm.hivm.pstu.b32").getValue();
    }
  }
  return failure();
}

FailureOr<StringRef> buildVstusCallee(MLIRContext *context, Type valueType) {
  std::string vec = getMemoryElementTypeFragment(getElementTypeFromVectorLike(valueType));
  auto lanes = getElementCountFromVectorLike(valueType);
  if (vec.empty() || !lanes) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vstus.v" + std::to_string(*lanes) + vec).getValue();
}

FailureOr<StringRef> buildVstusPostCallee(MLIRContext *context, Type valueType) {
  std::string vec = getMemoryElementTypeFragment(getElementTypeFromVectorLike(valueType));
  auto lanes = getElementCountFromVectorLike(valueType);
  if (vec.empty() || !lanes) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vstus.post.v" + std::to_string(*lanes) + vec).getValue();
}

StringRef buildVsturCallee(MLIRContext *context) { return StringAttr::get(context, "llvm.hivm.vstur").getValue(); }

StringRef buildInitAlignCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.init.vector.align.data").getValue();
}

StringRef buildSprclrCallee(MLIRContext *context) { return StringAttr::get(context, "llvm.hivm.sprclr").getValue(); }

StringRef buildSprstiCallee(MLIRContext *context, bool post) {
  return StringAttr::get(context, post ? "llvm.hivm.sprsti.post" : "llvm.hivm.sprsti").getValue();
}

StringRef buildSprstsCallee(MLIRContext *context, bool post) {
  return StringAttr::get(context, post ? "llvm.hivm.sprsts.post" : "llvm.hivm.sprsts").getValue();
}

StringRef buildStoreVfSimtInfoCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.store.vfsimt.info").getValue();
}

StringRef buildSyncthreadsCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.sync.workitems").getValue();
}

StringRef buildThreadfenceCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.fence.workitems").getValue();
}

StringRef buildThreadfenceBlockCallee(MLIRContext *context) {
  return StringAttr::get(context, "llvm.hivm.fenceblock.workitems").getValue();
}

StringRef buildVstarCallee(MLIRContext *context) { return StringAttr::get(context, "llvm.hivm.vstar").getValue(); }

StringRef buildVstasCallee(MLIRContext *context, bool post) {
  return StringAttr::get(context, post ? "llvm.hivm.vstas.post" : "llvm.hivm.vstas").getValue();
}

Value buildShuffleControlValue(OpBuilder &builder, Location loc, Value controlValue, int64_t widthValue,
                               unsigned controlMask) {
  Value lowBits = builder.create<arith::AndIOp>(loc, controlValue, getI32Constant(builder, loc, 0x1f));
  Value encodedWidth = getI32Constant(builder, loc, static_cast<uint32_t>(32 - widthValue) << 16);
  Value encodedMask = getI32Constant(builder, loc, static_cast<uint32_t>(controlMask) << 8);
  Value highBits = builder.create<arith::OrIOp>(loc, encodedWidth, encodedMask);
  return builder.create<arith::OrIOp>(loc, highBits, lowBits);
}

FailureOr<StringRef> buildAtomicCalleeName(MLIRContext *context, Type ptrType, Type valueType, Attribute signednessAttr,
                                           StringRef opName) {
  std::string elem = getAtomicElementTypeFragment(valueType, signednessAttr);
  if (elem.empty()) {
    return failure();
  }
  auto ptrTy = dyn_cast<pto::PtrType>(ptrType);
  if (!ptrTy) {
    return failure();
  }

  StringRef space;
  switch (ptrTy.getMemorySpace().getAddressSpace()) {
  case pto::AddressSpace::GM:
    space = "G";
    break;
  case pto::AddressSpace::VEC:
    if (valueType.isInteger(64)) {
      return failure();
    }
    space = "S";
    break;
  default:
    return failure();
  }

  return StringAttr::get(context, "llvm.hivm.atom." + opName.str() + "." + space.str() + "." + elem).getValue();
}

FailureOr<StringRef> buildL1CacheLoadCallee(MLIRContext *context, Type resultType, pto::L1Cache l1cache) {
  std::string elem;
  if (auto intType = dyn_cast<IntegerType>(resultType)) {
    if (intType.getWidth() == 8) {
      elem = "s8";
    } else if (intType.getWidth() == 16) {
      elem = "s16";
    } else if (intType.getWidth() == 32) {
      elem = "s32";
    } else if (intType.getWidth() == 64) {
      elem = "s64";
    }
  } else if (resultType.isF16() || resultType.isBF16()) {
    elem = "s16";
  } else if (resultType.isF32()) {
    elem = "s32";
  } else if (resultType.isF64()) {
    elem = "s64";
  } else if (pto::isPTOFloat8Type(resultType) || pto::isPTOHiFloat8Type(resultType)) {
    elem = "s8";
  } else if (pto::isPTOPackedLdgStgVectorType(resultType)) {
    unsigned totalBits = pto::getPTOPackedLdgStgTotalBits(resultType);
    if (totalBits == 16) {
      elem = "s16";
    } else if (totalBits == 32) {
      elem = "s32";
    } else if (totalBits == 64) {
      elem = "s64";
    }
  }
  if (elem.empty()) {
    return failure();
  }
  StringRef l1cacheName = l1cache == pto::L1Cache::Cache ? "cache" : "uncache";
  return StringAttr::get(context, "llvm.hivm.ldg." + l1cacheName.str() + "." + elem).getValue();
}

FailureOr<StringRef> buildL1CacheStoreCallee(MLIRContext *context, Type valueType, pto::L1Cache l1cache) {
  std::string elem;
  if (auto intType = dyn_cast<IntegerType>(valueType)) {
    if (intType.getWidth() == 8) {
      elem = "b8";
    } else if (intType.getWidth() == 16) {
      elem = "b16";
    } else if (intType.getWidth() == 32) {
      elem = "b32";
    } else if (intType.getWidth() == 64) {
      elem = "b64";
    }
  } else if (valueType.isF16() || valueType.isBF16()) {
    elem = "b16";
  } else if (valueType.isF32()) {
    elem = "b32";
  } else if (valueType.isF64()) {
    elem = "b64";
  } else if (pto::isPTOFloat8Type(valueType) || pto::isPTOHiFloat8Type(valueType)) {
    elem = "b8";
  } else if (pto::isPTOPackedLdgStgVectorType(valueType)) {
    unsigned totalBits = pto::getPTOPackedLdgStgTotalBits(valueType);
    if (totalBits == 16) {
      elem = "b16";
    } else if (totalBits == 32) {
      elem = "b32";
    } else if (totalBits == 64) {
      elem = "b64";
    }
  }
  if (elem.empty()) {
    return failure();
  }
  StringRef l1cacheName = l1cache == pto::L1Cache::Cache ? "cache" : "uncache";
  return StringAttr::get(context, "llvm.hivm.stg." + l1cacheName.str() + "." + elem).getValue();
}

FailureOr<StringRef> buildMulhiCallee(MLIRContext *context, Type resultType, pto::Signedness signedness) {
  if (resultType.isInteger(32)) {
    return StringAttr::get(context,
                           signedness == pto::Signedness::Unsigned ? "llvm.hivm.mulhi.ui" : "llvm.hivm.mulhi.i")
        .getValue();
  }
  if (resultType.isInteger(64) && signedness == pto::Signedness::Unsigned) {
    return StringAttr::get(context, "llvm.hivm.mul64hi.ui").getValue();
  }
  return failure();
}

FailureOr<StringRef> buildMulI32ToI64Callee(MLIRContext *context, pto::Signedness signedness) {
  return StringAttr::get(context, signedness == pto::Signedness::Unsigned ? "llvm.hivm.mul.i32toi64.ui"
                                                                          : "llvm.hivm.mul.i32toi64.i")
      .getValue();
}

std::string getScalarFloatBuiltinFragment(Type type) {
  if (type.isF32()) {
    return "f32";
  }
  if (type.isF16()) {
    return "f16";
  }
  if (type.isBF16()) {
    return "bf16";
  }
  return {};
}

std::string getLLVMFloatBuiltinFragment(Type type) {
  std::string scalar = getScalarFloatBuiltinFragment(type);
  if (!scalar.empty()) {
    return scalar;
  }

  auto vecType = dyn_cast<VectorType>(type);
  if (!vecType || vecType.getRank() != 1 || vecType.getDimSize(0) != 2) {
    return {};
  }
  Type elementType = vecType.getElementType();
  if (elementType.isF16()) {
    return "v2f16";
  }
  if (elementType.isBF16()) {
    return "v2bf16";
  }
  return {};
}

std::string getHIVMFloatBuiltinFragment(Type type) {
  std::string scalar = getScalarFloatBuiltinFragment(type);
  if (!scalar.empty()) {
    return scalar;
  }

  auto vecType = dyn_cast<VectorType>(type);
  if (!vecType || vecType.getRank() != 1 || vecType.getDimSize(0) != 2) {
    return {};
  }
  Type elementType = vecType.getElementType();
  if (elementType.isF16()) {
    return "f16x2";
  }
  if (elementType.isBF16()) {
    return "bf16x2";
  }
  return {};
}

FailureOr<StringRef> buildSqrtCallee(MLIRContext *context, Type valueType) {
  std::string elem = getLLVMFloatBuiltinFragment(valueType);
  if (elem != "f32" && elem != "f16" && elem != "v2f16") {
    return failure();
  }
  return StringAttr::get(context, "llvm.sqrt." + elem).getValue();
}

std::string getScalarHIVMFloatShortFragment(Type type) {
  if (type.isF32()) {
    return "f";
  }
  if (type.isF16()) {
    return "h";
  }
  if (type.isBF16()) {
    return "y";
  }
  return {};
}

FailureOr<StringRef> buildFmaCallee(MLIRContext *context, Type valueType) {
  std::string elem = getHIVMFloatBuiltinFragment(valueType);
  if (elem.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.ffma." + elem + ".rrr").getValue();
}

std::string getConvertScalarFragment(Type type, Attribute signednessAttr) {
  if (auto vecType = dyn_cast<VectorType>(type)) {
    if (vecType.getRank() != 1 || vecType.getDimSize(0) != 2) {
      return {};
    }
    Type elementType = vecType.getElementType();
    if (std::string elem = getLowPrecisionElementFragment(elementType);
        !elem.empty() && !pto::isPTOFloat4PackedType(elementType)) {
      return elem + "x2";
    }
    if (elementType.isF32()) {
      return "f32x2";
    }
    if (elementType.isF16()) {
      return "f16x2";
    }
    if (elementType.isBF16()) {
      return "bf16x2";
    }
    return {};
  }
  if (type.isF32()) {
    return "fp32";
  }
  if (type.isF16()) {
    return "fp16";
  }
  if (type.isBF16()) {
    return "bf16";
  }
  if (std::string elem = getLowPrecisionElementFragment(type); !elem.empty()) {
    return elem;
  }
  auto intType = dyn_cast<IntegerType>(type);
  if (!intType || (intType.getWidth() != 32 && intType.getWidth() != 64) || !signednessAttr) {
    return {};
  }
  auto signedness = cast<pto::SignednessAttr>(signednessAttr).getValue();
  return std::string(signedness == pto::Signedness::Unsigned ? "u" : "s") + std::to_string(intType.getWidth());
}

FailureOr<StringRef> buildConvertCallee(MLIRContext *context, Type srcType, Type dstType, Attribute signednessAttr) {
  std::string src = getConvertScalarFragment(srcType, signednessAttr);
  std::string dst = getConvertScalarFragment(dstType, signednessAttr);
  if (src.empty() || dst.empty()) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm." + src + ".to." + dst).getValue();
}

FailureOr<StringRef> buildVldsPostCallee(MLIRContext *context, Type resultType) {
  std::string vec = getMemoryElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vldsx1.post.v" + std::to_string(*lanes) + vec).getValue();
}

FailureOr<StringRef> buildVstsPostCallee(MLIRContext *context, Type valueType) {
  std::string vec = getMemoryElementTypeFragment(getElementTypeFromVectorLike(valueType));
  auto lanes = getElementCountFromVectorLike(valueType);
  if (vec.empty() || !lanes) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vstsx1.post.v" + std::to_string(*lanes) + vec).getValue();
}

StringRef buildVldasCallee(MLIRContext *context) { return StringAttr::get(context, "llvm.hivm.vldas").getValue(); }

FailureOr<StringRef> buildVldusCallee(MLIRContext *context, Type resultType) {
  std::string vec = getMemoryElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vldus.v" + std::to_string(*lanes) + vec).getValue();
}

FailureOr<StringRef> buildVldusPostCallee(MLIRContext *context, Type resultType) {
  std::string vec = getMemoryElementTypeFragment(getElementTypeFromVectorLike(resultType));
  auto lanes = getElementCountFromVectorLike(resultType);
  if (vec.empty() || !lanes) {
    return failure();
  }
  return StringAttr::get(context, "llvm.hivm.vldus.post.v" + std::to_string(*lanes) + vec).getValue();
}

FailureOr<StringRef> buildVcmpCallee(MLIRContext *context, Type inputType, StringRef cmpMode, bool isScalarCompare) {
  std::string vec = getCANN900VectorTypeFragment(inputType);
  std::string signedness = getCANN900SignednessFragment(getElementTypeFromVectorLike(inputType));
  if (vec.empty() || signedness.empty()) {
    return failure();
  }
  StringRef stem = isScalarCompare ? "vcmps" : "vcmp";
  return StringAttr::get(context, "llvm.hivm." + stem.str() + "." + cmpMode.str() + "." + signedness + ".z." + vec)
      .getValue();
}

} // namespace mlir::pto::detail
