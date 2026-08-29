// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOGenericOps.td - PTO generic value operations -------*- tablegen -*-===//
//
// Execution-domain-independent scalar and builtin-vector operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTOGENERICOPS
#define MLIR_DIALECT_PTO_IR_PTOGENERICOPS

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

// Generic value operations keep PTODSL-authored scalar and builtin-vector
// semantics in the PTO dialect until execution-scope validation has completed.
// They deliberately do not implement MicroOpInterface: a dedicated
// legalization pass lowers them to arith/math/LLVM before VPTO backend emission.
// This base class categorizes that execution-domain-independent operation family.
class PTO_GenericOp<string mnemonic, list<Trait> traits = []>
    : PTO_Op<mnemonic, traits>;

def PTO_GenericNumericType : AnyTypeOf<[
    AnySignlessInteger,
    AnyFloat,
    Index,
    VectorOfAnyRankOf<[AnySignlessInteger, AnyFloat]>
  ], "integer, float, index, or builtin vector thereof">;
def PTO_GenericFloatType : AnyTypeOf<[
    AnyFloat,
    VectorOfAnyRankOf<[AnyFloat]>
  ], "float or builtin vector of floats">;
def PTO_GenericConversionFloatType : AnyTypeOf<[
    AnyFloat,
    VectorOfAnyRankOf<[AnyFloat]>,
    Type<CPred<"::mlir::pto::isPTOLowPrecisionType($_self) || "
               "(::llvm::isa<::mlir::VectorType>($_self) && "
               "::mlir::pto::isPTOLowPrecisionType("
               "::llvm::cast<::mlir::VectorType>($_self).getElementType()))">,
         "PTO packed floating-point value">
  ], "standard or PTO packed floating-point value">;
def PTO_GenericIntegerLikeType : AnyTypeOf<[
    AnySignlessInteger,
    Index,
    VectorOfAnyRankOf<[AnySignlessInteger]>
  ], "integer, index, or builtin vector of integers">;
def PTO_GenericIntegerType : AnyTypeOf<[
    AnySignlessInteger,
    VectorOfAnyRankOf<[AnySignlessInteger]>
  ], "integer or builtin vector of integers">;
def PTO_GenericBoolLikeType : AnyTypeOf<[
    I1,
    VectorOfAnyRankOf<[I1]>
  ], "i1 or builtin vector of i1">;

def PTO_ConstantOp : PTO_GenericOp<"constant", [
    Pure,
    AllTypesMatch<["value", "result"]>
  ]> {
  let summary = "Execution-domain-independent scalar or builtin-vector constant";
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs PTO_GenericNumericType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "attr-dict $value";
}

class PTO_GenericBinaryIntegerLikeOpWithOverflow<
    string mnemonic, string opSummary, list<Trait> traits = []>
    : PTO_GenericOp<mnemonic, !listconcat([
        Pure,
        AllTypesMatch<["lhs", "rhs", "result"]>
      ], traits)> {
  let summary = opSummary;
  let arguments = (ins
    PTO_GenericIntegerLikeType:$lhs,
    PTO_GenericIntegerLikeType:$rhs,
    DefaultValuedAttr<PTO_IntegerOverflowAttr,
                      "::mlir::pto::IntegerOverflowFlags::none">:$overflowFlags
  );
  let results = (outs PTO_GenericIntegerLikeType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $lhs `,` $rhs (`overflow` `` $overflowFlags^)? attr-dict `:` type($lhs)
  }];
}

class PTO_GenericBinaryFloatOp<string mnemonic, string opSummary,
                              list<Trait> traits = []>
    : PTO_GenericOp<mnemonic, !listconcat([
        Pure,
        AllTypesMatch<["lhs", "rhs", "result"]>
      ], traits)> {
  let summary = opSummary;
  let arguments = (ins
    PTO_GenericFloatType:$lhs,
    PTO_GenericFloatType:$rhs,
    DefaultValuedAttr<PTO_FastMathAttr,
                      "::mlir::pto::FastMathFlags::none">:$fastmath
  );
  let results = (outs PTO_GenericFloatType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $lhs `,` $rhs (`fastmath` `` $fastmath^)? attr-dict `:` type($lhs)
  }];
}

def PTO_AddIOp : PTO_GenericBinaryIntegerLikeOpWithOverflow<
    "addi", "Execution-domain-independent integer addition", [Commutative]>;
def PTO_AddFOp : PTO_GenericBinaryFloatOp<
    "addf", "Execution-domain-independent floating-point addition", [Commutative]>;
def PTO_SubIOp : PTO_GenericBinaryIntegerLikeOpWithOverflow<
    "subi", "Execution-domain-independent integer subtraction">;
def PTO_SubFOp : PTO_GenericBinaryFloatOp<
    "subf", "Execution-domain-independent floating-point subtraction">;
def PTO_MulIOp : PTO_GenericBinaryIntegerLikeOpWithOverflow<
    "muli", "Execution-domain-independent integer multiplication", [Commutative]>;
def PTO_MulFOp : PTO_GenericBinaryFloatOp<
    "mulf", "Execution-domain-independent floating-point multiplication", [Commutative]>;

def PTO_AddUIExtendedOp : PTO_GenericOp<"addui_extended", [
    Pure,
    Commutative,
    AllTypesMatch<["lhs", "rhs", "sum"]>,
    TypesMatchWith<
      "overflow has i1 element type and the same shape as operands",
      "lhs", "overflow", "::getPTOI1SameShape($_self)">
  ]> {
  let summary = "Unsigned addition returning the wrapped sum and overflow bit";
  let arguments = (ins
    PTO_GenericIntegerLikeType:$lhs,
    PTO_GenericIntegerLikeType:$rhs
  );
  let results = (outs
    PTO_GenericIntegerLikeType:$sum,
    PTO_GenericBoolLikeType:$overflow
  );
  let hasVerifier = 1;
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($sum) `,` type($overflow)
  }];
}

def PTO_MulExtendedOp : PTO_GenericOp<"mul_extended", [
        Pure,
        Commutative,
        AllTypesMatch<["lhs", "rhs", "low", "high"]>
      ]> {
  let summary = "Full-width multiplication split into low and high halves";
  let arguments = (ins
    PTO_GenericIntegerLikeType:$lhs,
    PTO_GenericIntegerLikeType:$rhs,
    PTO_SignednessAttr:$signedness
  );
  let results = (outs
    PTO_GenericIntegerLikeType:$low,
    PTO_GenericIntegerLikeType:$high
  );
  let hasVerifier = 1;
  let assemblyFormat = [{
    $lhs `,` $rhs custom<Signedness>($signedness) attr-dict `:` type($low)
  }];
}

def PTO_NegIOp : PTO_GenericOp<"negi", [
    Pure,
    AllTypesMatch<["value", "result"]>
  ]> {
  let summary = "Execution-domain-independent integer negation";
  let arguments = (ins
    PTO_GenericIntegerLikeType:$value,
    DefaultValuedAttr<PTO_IntegerOverflowAttr,
                      "::mlir::pto::IntegerOverflowFlags::none">:$overflowFlags
  );
  let results = (outs PTO_GenericIntegerLikeType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $value (`overflow` `` $overflowFlags^)? attr-dict `:` type($value)
  }];
}

def PTO_NegFOp : PTO_GenericOp<"negf", [
    Pure,
    AllTypesMatch<["value", "result"]>
  ]> {
  let summary = "Execution-domain-independent floating-point negation";
  let arguments = (ins
    PTO_GenericFloatType:$value,
    DefaultValuedAttr<PTO_FastMathAttr,
                      "::mlir::pto::FastMathFlags::none">:$fastmath
  );
  let results = (outs PTO_GenericFloatType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $value (`fastmath` `` $fastmath^)? attr-dict `:` type($value)
  }];
}

def PTO_DivFOp : PTO_GenericOp<"divf", [
    Pure,
    AllTypesMatch<["lhs", "rhs", "result"]>
  ]> {
  let summary = "Execution-domain-independent floating-point value division";
  let arguments = (ins
    PTO_GenericFloatType:$lhs,
    PTO_GenericFloatType:$rhs,
    DefaultValuedAttr<PTO_FastMathAttr,
                      "::mlir::pto::FastMathFlags::none">:$fastmath
  );
  let results = (outs PTO_GenericFloatType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $lhs `,` $rhs (`fastmath` `` $fastmath^)? attr-dict `:` type($lhs)
  }];
}

class PTO_GenericBinaryIntegerLikeOp<string mnemonic, string opSummary>
    : PTO_GenericOp<mnemonic, [
        Pure,
        AllTypesMatch<["lhs", "rhs", "result"]>
      ]> {
  let summary = opSummary;
  let arguments = (ins
    PTO_GenericIntegerLikeType:$lhs,
    PTO_GenericIntegerLikeType:$rhs
  );
  let results = (outs PTO_GenericIntegerLikeType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs)
  }];
}

class PTO_GenericSignedBinaryIntegerLikeOp<string mnemonic, string opSummary>
    : PTO_GenericOp<mnemonic, [
        Pure,
        AllTypesMatch<["lhs", "rhs", "result"]>
      ]> {
  let summary = opSummary;
  let arguments = (ins
    PTO_GenericIntegerLikeType:$lhs,
    PTO_GenericIntegerLikeType:$rhs,
    PTO_SignednessAttr:$signedness
  );
  let results = (outs PTO_GenericIntegerLikeType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $lhs `,` $rhs custom<Signedness>($signedness) attr-dict `:` type($lhs)
  }];
}

def PTO_DivIOp : PTO_GenericSignedBinaryIntegerLikeOp<
    "divi", "Integer division rounding toward zero">;
def PTO_FloorDivOp : PTO_GenericSignedBinaryIntegerLikeOp<
    "floordiv", "Integer floor division">;
def PTO_CeilDivOp : PTO_GenericSignedBinaryIntegerLikeOp<
    "ceildiv", "Integer ceiling division">;
def PTO_RemIOp : PTO_GenericSignedBinaryIntegerLikeOp<
    "remi", "Integer remainder">;
def PTO_RemFOp : PTO_GenericOp<"remf", [
    Pure,
    AllTypesMatch<["lhs", "rhs", "result"]>
  ]> {
  let summary = "Floating-point remainder";
  let arguments = (ins
    PTO_GenericFloatType:$lhs,
    PTO_GenericFloatType:$rhs,
    DefaultValuedAttr<PTO_FastMathAttr,
                      "::mlir::pto::FastMathFlags::none">:$fastmath
  );
  let results = (outs PTO_GenericFloatType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $lhs `,` $rhs (`fastmath` `` $fastmath^)? attr-dict `:` type($lhs)
  }];
}
def PTO_AndOp : PTO_GenericBinaryIntegerLikeOp<
    "and", "Execution-domain-independent scalar bitwise and">;
def PTO_OrOp : PTO_GenericBinaryIntegerLikeOp<
    "or", "Execution-domain-independent scalar bitwise or">;
def PTO_XorOp : PTO_GenericBinaryIntegerLikeOp<
    "xor", "Execution-domain-independent scalar bitwise xor">;

class PTO_GenericBinaryIntegerOp<string mnemonic, string opSummary>
    : PTO_GenericOp<mnemonic, [
        Pure,
        AllTypesMatch<["lhs", "rhs", "result"]>
      ]> {
  let summary = opSummary;
  let arguments = (ins
    PTO_GenericIntegerType:$lhs,
    PTO_GenericIntegerType:$rhs
  );
  let results = (outs PTO_GenericIntegerType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs)
  }];
}

def PTO_ShlOp : PTO_GenericOp<"shl", [
    Pure,
    AllTypesMatch<["lhs", "rhs", "result"]>
  ]> {
  let summary = "Execution-domain-independent integer left shift";
  let arguments = (ins
    PTO_GenericIntegerType:$lhs,
    PTO_GenericIntegerType:$rhs,
    DefaultValuedAttr<PTO_IntegerOverflowAttr,
                      "::mlir::pto::IntegerOverflowFlags::none">:$overflowFlags
  );
  let results = (outs PTO_GenericIntegerType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $lhs `,` $rhs (`overflow` `` $overflowFlags^)? attr-dict `:` type($lhs)
  }];
}
def PTO_ShrOp : PTO_GenericOp<"shr", [
    Pure,
    AllTypesMatch<["lhs", "rhs", "result"]>
  ]> {
  let summary = "Signed arithmetic or unsigned logical right shift";
  let arguments = (ins
    PTO_GenericIntegerType:$lhs,
    PTO_GenericIntegerType:$rhs,
    PTO_SignednessAttr:$signedness
  );
  let results = (outs PTO_GenericIntegerType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $lhs `,` $rhs custom<Signedness>($signedness) attr-dict `:` type($lhs)
  }];
}

class PTO_GenericCmpOp<string mnemonic>
    : PTO_GenericOp<mnemonic, [
    Pure,
    AllTypesMatch<["lhs", "rhs"]>,
    TypesMatchWith<
      "result has i1 element type and the same shape as operands",
      "lhs", "result", "::getPTOI1SameShape($_self)">
  ]>;

def PTO_CmpIOp : PTO_GenericCmpOp<"cmpi"> {
  let summary = "Execution-domain-independent integer comparison";
  let arguments = (ins
    PTO_ScalarCmpPredicateAttr:$predicate,
    PTO_GenericIntegerLikeType:$lhs,
    PTO_GenericIntegerLikeType:$rhs,
    PTO_SignednessAttr:$signedness
  );
  let results = (outs PTO_GenericBoolLikeType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    custom<ScalarCmpPredicate>($predicate) $lhs `,` $rhs
    custom<Signedness>($signedness) attr-dict `:` type($lhs)
  }];
}

def PTO_CmpFOp : PTO_GenericCmpOp<"cmpf"> {
  let summary = "Execution-domain-independent floating-point comparison";
  let arguments = (ins
    PTO_ScalarCmpPredicateAttr:$predicate,
    PTO_GenericFloatType:$lhs,
    PTO_GenericFloatType:$rhs,
    DefaultValuedAttr<PTO_FastMathAttr,
                      "::mlir::pto::FastMathFlags::none">:$fastmath
  );
  let results = (outs PTO_GenericBoolLikeType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    custom<ScalarCmpPredicate>($predicate) $lhs `,` $rhs
    (`fastmath` `` $fastmath^)? attr-dict `:` type($lhs)
  }];
}

class PTO_GenericIntegerExtremumOp<string mnemonic, string opSummary>
    : PTO_GenericOp<mnemonic, [
    Pure,
    Commutative,
    AllTypesMatch<["lhs", "rhs", "result"]>
  ]> {
  let summary = opSummary;
  let arguments = (ins
    PTO_GenericIntegerLikeType:$lhs,
    PTO_GenericIntegerLikeType:$rhs,
    PTO_SignednessAttr:$signedness
  );
  let results = (outs PTO_GenericIntegerLikeType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $lhs `,` $rhs custom<Signedness>($signedness) attr-dict `:` type($lhs)
  }];
}

def PTO_MaxIOp : PTO_GenericIntegerExtremumOp<
    "maxi", "Integer maximum with explicit signedness">;
def PTO_MinIOp : PTO_GenericIntegerExtremumOp<
    "mini", "Integer minimum with explicit signedness">;

class PTO_GenericFloatExtremumOp<string mnemonic, string opSummary>
    : PTO_GenericOp<mnemonic, [
        Pure,
        Commutative,
        AllTypesMatch<["lhs", "rhs", "result"]>
      ]> {
  let summary = opSummary;
  let arguments = (ins
    PTO_GenericFloatType:$lhs,
    PTO_GenericFloatType:$rhs,
    DefaultValuedAttr<PTO_FastMathAttr,
                      "::mlir::pto::FastMathFlags::none">:$fastmath
  );
  let results = (outs PTO_GenericFloatType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $lhs `,` $rhs (`fastmath` `` $fastmath^)? attr-dict `:` type($lhs)
  }];
}

def PTO_MaximumOp : PTO_GenericFloatExtremumOp<
    "maximum", "NaN-propagating floating-point maximum">;
def PTO_MinimumOp : PTO_GenericFloatExtremumOp<
    "minimum", "NaN-propagating floating-point minimum">;

def PTO_MaxFOp : PTO_GenericFloatExtremumOp<
    "maxf", "Floating-point maximum with maxNum semantics">;
def PTO_MinFOp : PTO_GenericFloatExtremumOp<
    "minf", "Floating-point minimum with minNum semantics">;

def PTO_AbsIOp : PTO_GenericOp<"absi", [
    Pure,
    AllTypesMatch<["value", "result"]>
  ]> {
  let summary = "Execution-domain-independent integer absolute value";
  let arguments = (ins
    PTO_GenericIntegerLikeType:$value,
    PTO_SignednessAttr:$signedness
  );
  let results = (outs PTO_GenericIntegerLikeType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $value custom<Signedness>($signedness) attr-dict `:` type($value)
  }];
}

def PTO_AbsFOp : PTO_GenericOp<"absf", [
    Pure,
    AllTypesMatch<["value", "result"]>
  ]> {
  let summary = "Execution-domain-independent floating-point absolute value";
  let arguments = (ins PTO_GenericFloatType:$value);
  let results = (outs PTO_GenericFloatType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $value attr-dict `:` type($value)
  }];
}

class PTO_GenericConversionOp<string mnemonic, string opSummary,
                             dag opArguments, dag opResults>
    : PTO_GenericOp<mnemonic, [Pure]> {
  let summary = opSummary;
  let arguments = opArguments;
  let results = opResults;
  let hasVerifier = 1;
}

def PTO_ExtIOp : PTO_GenericConversionOp<
    "exti", "Extend an integer value with explicit signedness",
    (ins PTO_GenericIntegerType:$src, PTO_SignednessAttr:$signedness),
    (outs PTO_GenericIntegerType:$dst)> {
  let assemblyFormat = [{
    $src custom<Signedness>($signedness) attr-dict
    `:` type($src) `->` type($dst)
  }];
}

def PTO_TruncIOp : PTO_GenericConversionOp<
    "trunci", "Truncate an integer value",
    (ins PTO_GenericIntegerType:$src,
         DefaultValuedAttr<PTO_IntegerOverflowAttr,
                           "::mlir::pto::IntegerOverflowFlags::none">:$overflowFlags),
    (outs PTO_GenericIntegerType:$dst)> {
  let assemblyFormat = [{
    $src (`overflow` `` $overflowFlags^)? attr-dict
    `:` type($src) `->` type($dst)
  }];
}

def PTO_FToFOp : PTO_GenericConversionOp<
    "ftof", "Convert between floating-point formats",
    (ins PTO_GenericConversionFloatType:$src,
         OptionalAttr<PTO_FloatRoundingModeAttr>:$roundingmode,
         OptionalAttr<PTO_FastMathAttr>:$fastmath,
         DefaultValuedAttr<PTO_SaturationAttr,
                           "::mlir::pto::Saturation::Disable">:$saturation),
    (outs PTO_GenericConversionFloatType:$dst)> {
  let assemblyFormat = [{
    $src (custom<OptionalGenericRounding>($roundingmode)^)?
    (`fastmath` `` $fastmath^)?
    (custom<OptionalSaturation>($saturation)^)?
    attr-dict
    `:` type($src) `->` type($dst)
  }];
}

def PTO_FToIOp : PTO_GenericConversionOp<
    "ftoi", "Convert floating point to an integer with explicit signedness",
    (ins PTO_GenericConversionFloatType:$src,
         PTO_SignednessAttr:$signedness,
         OptionalAttr<PTO_FloatRoundingModeAttr>:$roundingmode,
         DefaultValuedAttr<PTO_SaturationAttr,
                           "::mlir::pto::Saturation::Disable">:$saturation),
    (outs PTO_GenericIntegerType:$dst)> {
  let assemblyFormat = [{
    $src custom<Signedness>($signedness)
    (custom<OptionalGenericRounding>($roundingmode)^)?
    (custom<OptionalSaturation>($saturation)^)? attr-dict
    `:` type($src) `->` type($dst)
  }];
}

def PTO_IToFOp : PTO_GenericConversionOp<
    "itof", "Convert an integer to floating point with explicit signedness",
    (ins PTO_GenericIntegerType:$src,
         PTO_SignednessAttr:$signedness,
         OptionalAttr<PTO_FloatRoundingModeAttr>:$roundingmode,
         DefaultValuedAttr<PTO_SaturationAttr,
                           "::mlir::pto::Saturation::Disable">:$saturation),
    (outs PTO_GenericConversionFloatType:$dst)> {
  let assemblyFormat = [{
    $src custom<Signedness>($signedness)
    (custom<OptionalGenericRounding>($roundingmode)^)?
    (custom<OptionalSaturation>($saturation)^)? attr-dict
    `:` type($src) `->` type($dst)
  }];
}

def PTO_GenericIndexOrIntegerType : AnyTypeOf<[
    AnySignlessInteger,
    Index,
    VectorOfAnyRankOf<[AnySignlessInteger, Index]>
  ], "integer, index, or builtin vector thereof">;

class PTO_GenericIndexCastBase<string mnemonic, string opSummary>
    : PTO_GenericOp<mnemonic, [Pure]> {
  let summary = opSummary;
  let arguments = (ins
    PTO_GenericIndexOrIntegerType:$src,
    PTO_SignednessAttr:$signedness
  );
  let results = (outs PTO_GenericIndexOrIntegerType:$dst);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $src custom<Signedness>($signedness) attr-dict
    `:` type($src) `->` type($dst)
  }];
}

def PTO_IndexCastOp : PTO_GenericIndexCastBase<
    "index_cast", "Signedness-aware conversion between index and integer values">;

def PTO_SelectOp : PTO_GenericOp<"select", [
    Pure,
    AllTypesMatch<["true_value", "false_value", "result"]>
  ]> {
  let summary = "Select between two execution-domain-independent values";
  let arguments = (ins
    PTO_GenericBoolLikeType:$condition,
    PTO_GenericNumericType:$true_value,
    PTO_GenericNumericType:$false_value
  );
  let results = (outs PTO_GenericNumericType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $condition `,` $true_value `,` $false_value attr-dict `:`
    custom<SelectType>(type($condition), type($result))
  }];
}

#endif // MLIR_DIALECT_PTO_IR_PTOGENERICOPS
