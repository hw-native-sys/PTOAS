// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOGenericAttrs.td - PTO generic value attributes ----*- tablegen -*-===//
//
// Semantic attributes shared by execution-domain-independent PTO values.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTOGENERICATTRS
#define MLIR_DIALECT_PTO_IR_PTOGENERICATTRS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Generic value-operation semantic attributes
//===----------------------------------------------------------------------===//

def PTO_IOFnone : I32BitEnumAttrCaseNone<"none">;
def PTO_IOFnsw : I32BitEnumAttrCaseBit<"nsw", 0>;
def PTO_IOFnuw : I32BitEnumAttrCaseBit<"nuw", 1>;
def PTO_IntegerOverflowFlags : I32BitEnumAttr<
    "IntegerOverflowFlags", "Integer overflow promises",
    [PTO_IOFnone, PTO_IOFnsw, PTO_IOFnuw]> {
  let separator = ", ";
  let cppNamespace = "::mlir::pto";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
}
def PTO_IntegerOverflowAttr :
    EnumAttr<PTO_Dialect, PTO_IntegerOverflowFlags, "overflow"> {
  let assemblyFormat = "`<` $value `>`";
}

def PTO_FMFnone : I32BitEnumAttrCaseNone<"none">;
def PTO_FMFreassoc : I32BitEnumAttrCaseBit<"reassoc", 0>;
def PTO_FMFnnan : I32BitEnumAttrCaseBit<"nnan", 1>;
def PTO_FMFninf : I32BitEnumAttrCaseBit<"ninf", 2>;
def PTO_FMFnsz : I32BitEnumAttrCaseBit<"nsz", 3>;
def PTO_FMFarcp : I32BitEnumAttrCaseBit<"arcp", 4>;
def PTO_FMFcontract : I32BitEnumAttrCaseBit<"contract", 5>;
def PTO_FMFafn : I32BitEnumAttrCaseBit<"afn", 6>;
def PTO_FMFfast : I32BitEnumAttrCaseGroup<"fast", [
    PTO_FMFreassoc, PTO_FMFnnan, PTO_FMFninf, PTO_FMFnsz,
    PTO_FMFarcp, PTO_FMFcontract, PTO_FMFafn]>;
def PTO_FastMathFlags : I32BitEnumAttr<
    "FastMathFlags", "Floating-point optimization permissions",
    [PTO_FMFnone, PTO_FMFreassoc, PTO_FMFnnan, PTO_FMFninf, PTO_FMFnsz,
     PTO_FMFarcp, PTO_FMFcontract, PTO_FMFafn, PTO_FMFfast]> {
  let separator = ",";
  let cppNamespace = "::mlir::pto";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
}
def PTO_FastMathAttr : EnumAttr<PTO_Dialect, PTO_FastMathFlags, "fastmath"> {
  let assemblyFormat = "`<` $value `>`";
}

def PTO_RToNearestTiesToEven : I32EnumAttrCase<"to_nearest_even", 0>;
def PTO_RDownward : I32EnumAttrCase<"downward", 1>;
def PTO_RUpward : I32EnumAttrCase<"upward", 2>;
def PTO_RTowardZero : I32EnumAttrCase<"toward_zero", 3>;
def PTO_RToNearestTiesAwayFromZero : I32EnumAttrCase<"to_nearest_away", 4>;
def PTO_RToOdd : I32EnumAttrCase<"to_odd", 5>;
def PTO_RHybrid : I32EnumAttrCase<"hybrid", 6>;
def PTO_FloatRoundingModeAttr : I32EnumAttr<
    "FloatRoundingMode", "Floating-point rounding mode", [
      PTO_RToNearestTiesToEven, PTO_RDownward, PTO_RUpward,
      PTO_RTowardZero, PTO_RToNearestTiesAwayFromZero, PTO_RToOdd,
      PTO_RHybrid
    ]> {
  let cppNamespace = "::mlir::pto";
}

//===----------------------------------------------------------------------===//
// Generic scalar and builtin-vector comparison predicates
//===----------------------------------------------------------------------===//

def PTO_ScalarCmpPredicate_Eq : I32EnumAttrCase<"Eq", 0, "eq">;
def PTO_ScalarCmpPredicate_Ne : I32EnumAttrCase<"Ne", 1, "ne">;
def PTO_ScalarCmpPredicate_Lt : I32EnumAttrCase<"Lt", 2, "lt">;
def PTO_ScalarCmpPredicate_Le : I32EnumAttrCase<"Le", 3, "le">;
def PTO_ScalarCmpPredicate_Gt : I32EnumAttrCase<"Gt", 4, "gt">;
def PTO_ScalarCmpPredicate_Ge : I32EnumAttrCase<"Ge", 5, "ge">;
def PTO_ScalarCmpPredicate_AlwaysFalse
    : I32EnumAttrCase<"AlwaysFalse", 6, "false">;
def PTO_ScalarCmpPredicate_OEq : I32EnumAttrCase<"OEq", 7, "oeq">;
def PTO_ScalarCmpPredicate_OGt : I32EnumAttrCase<"OGt", 8, "ogt">;
def PTO_ScalarCmpPredicate_OGe : I32EnumAttrCase<"OGe", 9, "oge">;
def PTO_ScalarCmpPredicate_OLt : I32EnumAttrCase<"OLt", 10, "olt">;
def PTO_ScalarCmpPredicate_OLe : I32EnumAttrCase<"OLe", 11, "ole">;
def PTO_ScalarCmpPredicate_ONe : I32EnumAttrCase<"ONe", 12, "one">;
def PTO_ScalarCmpPredicate_Ord : I32EnumAttrCase<"Ord", 13, "ord">;
def PTO_ScalarCmpPredicate_UEq : I32EnumAttrCase<"UEq", 14, "ueq">;
def PTO_ScalarCmpPredicate_UGt : I32EnumAttrCase<"UGt", 15, "ugt">;
def PTO_ScalarCmpPredicate_UGe : I32EnumAttrCase<"UGe", 16, "uge">;
def PTO_ScalarCmpPredicate_ULt : I32EnumAttrCase<"ULt", 17, "ult">;
def PTO_ScalarCmpPredicate_ULe : I32EnumAttrCase<"ULe", 18, "ule">;
def PTO_ScalarCmpPredicate_UNe : I32EnumAttrCase<"UNe", 19, "une">;
def PTO_ScalarCmpPredicate_Uno : I32EnumAttrCase<"Uno", 20, "uno">;
def PTO_ScalarCmpPredicate_AlwaysTrue
    : I32EnumAttrCase<"AlwaysTrue", 21, "true">;
def PTO_ScalarCmpPredicateEnum : PTO_I32Enum<
  "ScalarCmpPredicate", "PTO generic scalar comparison predicate", [
    PTO_ScalarCmpPredicate_Eq,
    PTO_ScalarCmpPredicate_Ne,
    PTO_ScalarCmpPredicate_Lt,
    PTO_ScalarCmpPredicate_Le,
    PTO_ScalarCmpPredicate_Gt,
    PTO_ScalarCmpPredicate_Ge,
    PTO_ScalarCmpPredicate_AlwaysFalse,
    PTO_ScalarCmpPredicate_OEq,
    PTO_ScalarCmpPredicate_OGt,
    PTO_ScalarCmpPredicate_OGe,
    PTO_ScalarCmpPredicate_OLt,
    PTO_ScalarCmpPredicate_OLe,
    PTO_ScalarCmpPredicate_ONe,
    PTO_ScalarCmpPredicate_Ord,
    PTO_ScalarCmpPredicate_UEq,
    PTO_ScalarCmpPredicate_UGt,
    PTO_ScalarCmpPredicate_UGe,
    PTO_ScalarCmpPredicate_ULt,
    PTO_ScalarCmpPredicate_ULe,
    PTO_ScalarCmpPredicate_UNe,
    PTO_ScalarCmpPredicate_Uno,
    PTO_ScalarCmpPredicate_AlwaysTrue
  ]>;

def PTO_ScalarCmpPredicateAttr
    : PTO_Attr<"ScalarCmpPredicate", "scalar_cmp_predicate"> {
  let parameters = (ins EnumParameter<PTO_ScalarCmpPredicateEnum>:$value);
  let assemblyFormat = "`<` params `>`";
  let summary = "Type-independent predicate for generic scalar comparison";
}

#endif // MLIR_DIALECT_PTO_IR_PTOGENERICATTRS
