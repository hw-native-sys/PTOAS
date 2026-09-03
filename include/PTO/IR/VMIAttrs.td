// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIAttrs.td - PTO VMI attributes definitions --------*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_VMIATTRS
#define MLIR_DIALECT_PTO_IR_VMIATTRS

def VMILayoutAttr : PTO_Attr<"VMILayout", "vmi.layout"> {
  let summary = "VMI logical vector register layout";
  let parameters = (ins
    StringRefParameter<"layout kind">:$kind,
    "int64_t":$factor,
    "int64_t":$blockElems,
    "int64_t":$slots,
    "int64_t":$laneStride
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    static VMILayoutAttr getContiguous(::mlir::MLIRContext *context,
                                       int64_t laneStride = 1);
    static VMILayoutAttr getDeinterleaved(::mlir::MLIRContext *context,
                                          int64_t factor,
                                          int64_t laneStride = 1);
    static VMILayoutAttr getBlockDeinterleaved(
        ::mlir::MLIRContext *context, int64_t factor);
    static VMILayoutAttr getGroupSlots(::mlir::MLIRContext *context,
                                       int64_t numGroups,
                                       int64_t slots = 0,
                                       int64_t laneStride = 1);

    bool isContiguous() const { return getKind() == "contiguous"; }
    bool isDeinterleaved() const { return getKind() == "deinterleaved"; }
    bool isBlockDeinterleaved() const {
      return getKind() == "block_deinterleaved";
    }
    bool isGroupSlots() const { return getKind() == "num_groups"; }
    bool isDenseSplit() const {
      return isDeinterleaved() || isBlockDeinterleaved();
    }
    bool isDense() const { return isContiguous() || isDenseSplit(); }
    int64_t getNumGroups() const { return getFactor(); }
    bool hasDenseLaneStride() const {
      return isDense() && getLaneStride() != 1;
    }
    bool hasGroupSlotLaneStride() const {
      return isGroupSlots() && getLaneStride() != 1;
    }
    bool hasLaneStride() const { return getLaneStride() != 1; }
  }];
}

//===----------------------------------------------------------------------===//
// VMI Predication Mode — [pmode]
// Controls inactive/predicated-off lane behavior on vector operations.
//===----------------------------------------------------------------------===//

def VMI_PMode_Merge : I32EnumAttrCase<"Merge", 0, "merge">;
def VMI_PMode_Zero  : I32EnumAttrCase<"Zero",  1, "zero">;
def VMIPredicationMode : I32EnumAttr<"PredicationMode",
    "VMI predication mode: merge (keep old value) or zero (fill with 0)", [
  VMI_PMode_Merge, VMI_PMode_Zero
]> {
  let cppNamespace = "::mlir::pto";
}

//===----------------------------------------------------------------------===//
// VMI Distribution Mode — {dist-mode}
// Controls the data-layout interpretation of load/store operations.
//===----------------------------------------------------------------------===//

def VMI_DistMode_Continuous : I32EnumAttrCase<"Continuous", 0, "continuous">;
def VMI_DistMode_Dintlv     : I32EnumAttrCase<"Dintlv",     2, "dintlv">;
def VMI_DistMode_Intlv      : I32EnumAttrCase<"Intlv",      3, "intlv">;
def VMI_DistMode_Brc        : I32EnumAttrCase<"Brc",        4, "brc">;
def VMIDistMode : I32EnumAttr<"DistMode",
    "VMI distribution mode for load/store", [
  VMI_DistMode_Continuous, VMI_DistMode_Dintlv,
  VMI_DistMode_Intlv, VMI_DistMode_Brc
]> {
  let cppNamespace = "::mlir::pto";
}

//===----------------------------------------------------------------------===//
// VMI Rounding Mode — {rnd=}
// Controls the rounding direction for narrowing convert operations.
//===----------------------------------------------------------------------===//

def VMI_Round_RTZ : I32EnumAttrCase<"RTZ", 0, "rtz">;
def VMI_Round_RNE : I32EnumAttrCase<"RNE", 1, "rne">;
def VMI_Round_Odd : I32EnumAttrCase<"Odd", 2, "odd">;
def VMIConvRoundMode : I32EnumAttr<"ConvRoundMode",
    "VMI rounding mode for narrowing conversions", [
  VMI_Round_RTZ, VMI_Round_RNE, VMI_Round_Odd
]> {
  let cppNamespace = "::mlir::pto";
}

//===----------------------------------------------------------------------===//
// VMI Saturation Mode — {sat=}
// Controls whether narrowing conversions saturate on overflow.
//===----------------------------------------------------------------------===//

def VMI_Sat_None : I32EnumAttrCase<"None", 0, "none">;
def VMI_Sat_Sat  : I32EnumAttrCase<"Sat",  1, "sat">;
def VMISatMode : I32EnumAttr<"SatMode",
    "VMI saturation mode for narrowing conversions", [
  VMI_Sat_None, VMI_Sat_Sat
]> {
  let cppNamespace = "::mlir::pto";
}

#endif // MLIR_DIALECT_PTO_IR_VMIATTRS
