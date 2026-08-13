// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOOpCommon.td - shared VPTO operation definitions -----*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// Shared constraints and base classes used by more than one VPTO operation
// family. Concrete operations belong in one of the family-specific files.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_VPTOOPCOMMON
#define MLIR_DIALECT_PTO_IR_VPTOOPCOMMON

include "PTO/IR/PTOOpCommon.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "PTO/IR/VPTOInterfaces.td"

def PTO_VectorType : Type<CPred<"::llvm::isa<::mlir::pto::VRegType>($_self)">,
                           "PTO low-level vector type">;
def PTO_MaskTypeConstraint : Type<CPred<"::llvm::isa<::mlir::pto::MaskType>($_self)">,
                                   "PTO low-level mask type">;
def PTO_B8MaskTypeConstraint : Type<
  CPred<"::llvm::isa<::mlir::pto::MaskType>($_self) && ::llvm::cast<::mlir::pto::MaskType>($_self).isB8()">,
  "PTO low-level b8 mask type">;
def PTO_B16MaskTypeConstraint : Type<
  CPred<"::llvm::isa<::mlir::pto::MaskType>($_self) && ::llvm::cast<::mlir::pto::MaskType>($_self).isB16()">,
  "PTO low-level b16 mask type">;
def PTO_B32MaskTypeConstraint : Type<
  CPred<"::llvm::isa<::mlir::pto::MaskType>($_self) && ::llvm::cast<::mlir::pto::MaskType>($_self).isB32()">,
  "PTO low-level b32 mask type">;
def PTO_AlignTypeConstraint : Type<CPred<"::llvm::isa<::mlir::pto::AlignType>($_self)">,
                                    "PTO low-level align type">;

def PTO_ScalingPtrType : Type<
  CPred<"::llvm::isa<::mlir::pto::PtrType>($_self) && "
        "::llvm::cast<::mlir::pto::PtrType>($_self).getMemorySpace().getAddressSpace() == ::mlir::pto::AddressSpace::SCALING">,
  "scaling pointer type">;
def PTO_BufferLikeType : AnyTypeOf<[AnyMemRef, PTO_BufferType],
                                    "memref or pointer-like buffer type">;
def PTO_AccStorePayloadType : AnyTypeOf<[AnySignlessInteger, AnyFloat, PTO_ScalingPtrType],
                                        "signless integer, float scalar, or scaling pointer">;
def PTO_AccStoreClipPayloadType : AnyTypeOf<[AnyInteger, AnyFloat],
                                            "integer or float clip scalar">;
def PTO_V2F16Type : Type<
  CPred<"::llvm::isa<::mlir::VectorType>($_self) && "
        "::llvm::cast<::mlir::VectorType>($_self).getRank() == 1 && "
        "::llvm::cast<::mlir::VectorType>($_self).getDimSize(0) == 2 && "
        "::llvm::cast<::mlir::VectorType>($_self).getElementType().isF16()">,
  "vector<2xf16>">;
def PTO_V2BF16Type : Type<
  CPred<"::llvm::isa<::mlir::VectorType>($_self) && "
        "::llvm::cast<::mlir::VectorType>($_self).getRank() == 1 && "
        "::llvm::cast<::mlir::VectorType>($_self).getDimSize(0) == 2 && "
        "::llvm::cast<::mlir::VectorType>($_self).getElementType().isBF16()">,
  "vector<2xbf16>">;
def PTO_V2F32Type : Type<
  CPred<"::llvm::isa<::mlir::VectorType>($_self) && "
        "::llvm::cast<::mlir::VectorType>($_self).getRank() == 1 && "
        "::llvm::cast<::mlir::VectorType>($_self).getDimSize(0) == 2 && "
        "::llvm::cast<::mlir::VectorType>($_self).getElementType().isF32()">,
  "vector<2xf32>">;
def PTO_V2F8E4M3FNType : Type<
  CPred<"::llvm::isa<::mlir::VectorType>($_self) && "
        "::llvm::cast<::mlir::VectorType>($_self).getRank() == 1 && "
        "::llvm::cast<::mlir::VectorType>($_self).getDimSize(0) == 2 && "
        "::llvm::isa<::mlir::Float8E4M3FNType>(::llvm::cast<::mlir::VectorType>($_self).getElementType())">,
  "vector<2xf8E4M3FN>">;
def PTO_V4F8E4M3FNType : Type<
  CPred<"::llvm::isa<::mlir::VectorType>($_self) && "
        "::llvm::cast<::mlir::VectorType>($_self).getRank() == 1 && "
        "::llvm::cast<::mlir::VectorType>($_self).getDimSize(0) == 4 && "
        "::llvm::isa<::mlir::Float8E4M3FNType>(::llvm::cast<::mlir::VectorType>($_self).getElementType())">,
  "vector<4xf8E4M3FN>">;
def PTO_V8F8E4M3FNType : Type<
  CPred<"::llvm::isa<::mlir::VectorType>($_self) && "
        "::llvm::cast<::mlir::VectorType>($_self).getRank() == 1 && "
        "::llvm::cast<::mlir::VectorType>($_self).getDimSize(0) == 8 && "
        "::llvm::isa<::mlir::Float8E4M3FNType>(::llvm::cast<::mlir::VectorType>($_self).getElementType())">,
  "vector<8xf8E4M3FN>">;
def PTO_V2F8E5M2Type : Type<
  CPred<"::llvm::isa<::mlir::VectorType>($_self) && "
        "::llvm::cast<::mlir::VectorType>($_self).getRank() == 1 && "
        "::llvm::cast<::mlir::VectorType>($_self).getDimSize(0) == 2 && "
        "::llvm::isa<::mlir::Float8E5M2Type>(::llvm::cast<::mlir::VectorType>($_self).getElementType())">,
  "vector<2xf8E5M2>">;
def PTO_V4F8E5M2Type : Type<
  CPred<"::llvm::isa<::mlir::VectorType>($_self) && "
        "::llvm::cast<::mlir::VectorType>($_self).getRank() == 1 && "
        "::llvm::cast<::mlir::VectorType>($_self).getDimSize(0) == 4 && "
        "::llvm::isa<::mlir::Float8E5M2Type>(::llvm::cast<::mlir::VectorType>($_self).getElementType())">,
  "vector<4xf8E5M2>">;
def PTO_V8F8E5M2Type : Type<
  CPred<"::llvm::isa<::mlir::VectorType>($_self) && "
        "::llvm::cast<::mlir::VectorType>($_self).getRank() == 1 && "
        "::llvm::cast<::mlir::VectorType>($_self).getDimSize(0) == 8 && "
        "::llvm::isa<::mlir::Float8E5M2Type>(::llvm::cast<::mlir::VectorType>($_self).getElementType())">,
  "vector<8xf8E5M2>">;
def PTO_V2HiF8Type : Type<
  CPred<"::llvm::isa<::mlir::pto::HiF8x2Type>($_self)">,
  "!pto.hif8x2">;
def PTO_LowPrecisionPackedType : AnyTypeOf<[PTO_V2F8E4M3FNType,
                                            PTO_V4F8E4M3FNType,
                                            PTO_V8F8E4M3FNType,
                                            PTO_V2F8E5M2Type,
                                            PTO_V4F8E5M2Type,
                                            PTO_V8F8E5M2Type,
                                            PTO_V2HiF8Type],
                                           "supported FP8/HiF8 packed type">;
def PTO_SimtF4ValueType : Type<
  CPred<"::llvm::isa<::mlir::pto::F4E1M2x2Type, ::mlir::pto::F4E2M1x2Type>($_self)">,
  "!pto.f4E1M2x2 or !pto.f4E2M1x2">;
def PTO_SimtShuffleValueType : AnyTypeOf<[I32, I64, F16, F32, PTO_V2F16Type],
                                         "i32, i64, f16, f32 or vector<2xf16>">;
def PTO_SimtUnaryFloatValueType : AnyTypeOf<[F16, F32, BF16, PTO_V2F16Type, PTO_V2BF16Type],
                                           "f16, f32, bf16, vector<2xf16> or vector<2xbf16>">;
def PTO_SimtUnaryF16F32ValueType : AnyTypeOf<[F16, F32, PTO_V2F16Type],
                                            "f16, f32 or vector<2xf16>">;
def PTO_SimtBinaryF32BF16ValueType : AnyTypeOf<[F32, BF16, PTO_V2F16Type, PTO_V2BF16Type],
                                              "f32, bf16, vector<2xf16> or vector<2xbf16>">;
def PTO_SimtBinaryFloatValueType : AnyTypeOf<[F16, F32, BF16, PTO_V2F16Type, PTO_V2BF16Type],
                                             "floating-point scalar or vector binary value">;
def PTO_SimtBinaryF16F32ValueType : AnyTypeOf<[F16, F32, PTO_V2F16Type],
                                             "f16, f32 or vector<2xf16>">;
def PTO_SimtConvertValueType : AnyTypeOf<[I32, I64, F16, BF16, F32,
                                          PTO_V2F16Type, PTO_V2BF16Type,
                                          PTO_V2F32Type, PTO_LowPrecisionPackedType,
                                          PTO_SimtF4ValueType],
                                         "standard scalar, low-precision scalar payload, or supported vector<2xT> conversion type">;
def PTO_AtomicValueType : AnyTypeOf<[I32, I64, F16, BF16, F32, PTO_V2F16Type, PTO_V2BF16Type],
                                    "i32, i64, f16, bf16, f32, vector<2xf16> or vector<2xbf16>">;

class PTO_MicroOp<string mnemonic, list<Trait> traits = []>
    : PTO_Op<mnemonic, !listconcat(traits, [MicroOpInterface])>;

class PTO_VectorMicroOp<string mnemonic, list<Trait> traits = []>
    : PTO_MicroOp<mnemonic,
                  !listconcat(traits, [VectorMicroOpInterface])>;

class PTO_CubeMicroOp<string mnemonic, list<Trait> traits = []>
    : PTO_MicroOp<mnemonic,
                  !listconcat(traits, [CubeMicroOpInterface])>;

class PTO_MteOp<string mnemonic, list<Trait> traits = []>
    : PTO_MicroOp<mnemonic, !listconcat(traits, [MteOpInterface])>;

class PTO_SimtOp<string mnemonic, list<Trait> traits = []>
    : PTO_MicroOp<mnemonic, !listconcat(traits, [SimtOpInterface])>;

class PTO_DirectSimtNullaryI64PureOp<string mnemonic> : PTO_SimtOp<mnemonic, [Pure]> {
  let arguments = (ins);
  let results = (outs I64:$result);
  let assemblyFormat = [{ attr-dict `:` type($result) }];
}

class PTO_DirectSimtNullaryI32PureOp<string mnemonic> : PTO_SimtOp<mnemonic, [Pure]> {
  let arguments = (ins);
  let results = (outs I32:$result);
  let assemblyFormat = [{ attr-dict `:` type($result) }];
}

#endif // MLIR_DIALECT_PTO_IR_VPTOOPCOMMON
