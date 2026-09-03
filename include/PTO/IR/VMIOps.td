// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIOps.td - PTO VMI semantic operations -------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_VMIOPS
#define MLIR_DIALECT_PTO_IR_VMIOPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def VMI_VRegTypeConstraint : Type<
  CPred<"::llvm::isa<::mlir::pto::VMIVRegType>($_self)">,
  "VMI logical vector register type">;

def VMI_MaskTypeConstraint : Type<
  CPred<"::llvm::isa<::mlir::pto::VMIMaskType>($_self)">,
  "VMI logical mask type">;

def VMI_ValueTypeConstraint : Type<
  CPred<"::llvm::isa<::mlir::pto::VMIVRegType, ::mlir::pto::VMIMaskType>($_self)">,
  "VMI logical vector or mask type">;

def PTO_PhysicalVRegTypeConstraint : Type<
  CPred<"::llvm::isa<::mlir::pto::VRegType>($_self)">,
  "PTO physical vector register type">;

def PTO_PhysicalMaskTypeConstraint : Type<
  CPred<"::llvm::isa<::mlir::pto::MaskType>($_self)">,
  "PTO physical mask type">;

def PTO_PhysicalVMIPartTypeConstraint : AnyTypeOf<
  [PTO_PhysicalVRegTypeConstraint, PTO_PhysicalMaskTypeConstraint],
  "PTO physical vector register or mask type">;

class VMI_Op<string mnemonic, list<Trait> traits = []>
    : PTO_Op<"vmi." # mnemonic,
             !listconcat(traits, [MicroOpInterface, VectorMicroOpInterface])>;

//===--- Legacy (old) VMI ops ---===//

def VMIConstantOp : VMI_Op<"constant", [Pure]> {
  let summary = "VMI logical vector constant";
  let arguments = (ins AnyAttr:$value);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
}

def VMIBroadcastOp : VMI_Op<"broadcast", [Pure]> {
  let summary = "Broadcast one scalar or 1-lane VMI vector to a VMI logical vector";
  let arguments = (ins AnyType:$value);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

def VMIIotaOp : VMI_Op<"iota", [Pure]> {
  let summary = "Create a VMI logical index vector from a scalar base";
  let description = [{
    Produces a continuous ramp over the full logical length L:
    `dst[i] = base + i` (ASC). For L=128 that is `base..base+127`
    (physical parts `base+0..63` then `base+64..127`).
  }];
  let arguments = (ins
    AnyTypeOf<[AnyInteger, AnyFloat], "integer/float scalar">:$base,
    OptionalAttr<StrAttr>:$order
  );
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$base attr-dict `:` type($base) `->` type($result)";
}

def VMIGroupIotaOp : VMI_Op<"group_iota", [Pure]> {
  let summary = "Create a contiguous group-periodic VMI index vector";
  let description = [{
    Internal legacy form for public `vci(..., group=C)` with C > 1. For logical
    length L and group size S=L/C, produces
    `dst[g*S + j] = base +/- j`. This producer is contiguous-only; consumers
    requesting another layout receive an explicit `ensure_layout`.
  }];
  let arguments = (ins
    AnyTypeOf<[AnyInteger, AnyFloat], "integer/float scalar">:$base,
    OptionalAttr<StrAttr>:$order,
    I64Attr:$group
  );
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$base attr-dict `:` type($base) `->` type($result)";
}

def VMICreateMaskOp : VMI_Op<"create_mask", [Pure]> {
  let summary = "Create a VMI logical prefix predicate mask";
  let arguments = (ins Index:$active_lanes);
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$active_lanes attr-dict `:` type($active_lanes) `->` type($result)";
}

def VMICreateGroupMaskOp : VMI_Op<"create_group_mask", [Pure]> {
  let summary = "Create a VMI logical grouped predicate mask";
  let description = [{
    Creates a mask where lane i is active iff
    `(i % group_size) < active_elems_per_group`.
  }];
  let arguments = (ins
    Index:$active_elems_per_group,
    I64Attr:$num_groups,
    I64Attr:$group_size
  );
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$active_elems_per_group attr-dict `:` type($active_elems_per_group) `->` type($result)";
}

def VMIConstantMaskOp : VMI_Op<"constant_mask", [Pure]> {
  let summary = "VMI logical predicate mask constant";
  let arguments = (ins AnyAttr:$value);
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
}

def VMIMaskAndOp : VMI_Op<"mask_and", [Pure]> {
  let summary = "VMI logical predicate mask and";
  let arguments = (ins VMI_MaskTypeConstraint:$lhs, VMI_MaskTypeConstraint:$rhs);
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIMaskOrOp : VMI_Op<"mask_or", [Pure]> {
  let summary = "VMI logical predicate mask or";
  let arguments = (ins VMI_MaskTypeConstraint:$lhs, VMI_MaskTypeConstraint:$rhs);
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIMaskXOrOp : VMI_Op<"mask_xor", [Pure]> {
  let summary = "VMI logical predicate mask xor";
  let arguments = (ins VMI_MaskTypeConstraint:$lhs, VMI_MaskTypeConstraint:$rhs);
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIMaskNotOp : VMI_Op<"mask_not", [Pure]> {
  let summary = "VMI logical predicate mask not";
  let arguments = (ins VMI_MaskTypeConstraint:$source);
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIAddFOp : VMI_Op<"addf", [Pure]> {
  let summary = "VMI floating-point elementwise add";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIAddIOp : VMI_Op<"addi", [Pure]> {
  let summary = "VMI integer elementwise add";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMISubFOp : VMI_Op<"subf", [Pure]> {
  let summary = "VMI floating-point elementwise subtract";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMISubIOp : VMI_Op<"subi", [Pure]> {
  let summary = "VMI integer elementwise subtract";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIMulFOp : VMI_Op<"mulf", [Pure]> {
  let summary = "VMI floating-point elementwise multiply";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIMulIOp : VMI_Op<"muli", [Pure]> {
  let summary = "VMI integer elementwise multiply";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIFmaOp : VMI_Op<"fma", [Pure]> {
  let summary = "VMI fused floating-point multiply-add";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs,
                       VMI_VRegTypeConstraint:$acc);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs `,` $acc attr-dict `:` type($lhs) `,` type($rhs) `,` type($acc) `->` type($result)";
}

def VMIDivFOp : VMI_Op<"divf", [Pure]> {
  let summary = "VMI floating-point elementwise divide";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIMinFOp : VMI_Op<"minf", [Pure]> {
  let summary = "VMI floating-point elementwise minimum";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIMaxFOp : VMI_Op<"maxf", [Pure]> {
  let summary = "VMI floating-point elementwise maximum";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIMinIOp : VMI_Op<"mini", [Pure]> {
  let summary = "VMI integer elementwise minimum";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIMaxIOp : VMI_Op<"maxi", [Pure]> {
  let summary = "VMI integer elementwise maximum";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMINegFOp : VMI_Op<"negf", [Pure]> {
  let summary = "VMI floating-point elementwise negate";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMINegIOp : VMI_Op<"negi", [Pure]> {
  let summary = "VMI integer elementwise negate";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIAbsFOp : VMI_Op<"absf", [Pure]> {
  let summary = "VMI floating-point elementwise absolute value";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIAbsIOp : VMI_Op<"absi", [Pure]> {
  let summary = "VMI integer elementwise absolute value";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMISqrtOp : VMI_Op<"sqrt", [Pure]> {
  let summary = "VMI floating-point elementwise square root";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIExpOp : VMI_Op<"exp", [Pure]> {
  let summary = "VMI floating-point elementwise exponential";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMILnOp : VMI_Op<"ln", [Pure]> {
  let summary = "VMI floating-point elementwise natural logarithm";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIReluOp : VMI_Op<"relu", [Pure]> {
  let summary = "VMI floating-point or signed 32-bit integer elementwise ReLU";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIAndIOp : VMI_Op<"andi", [Pure]> {
  let summary = "VMI integer elementwise bitwise and";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIOrIOp : VMI_Op<"ori", [Pure]> {
  let summary = "VMI integer elementwise bitwise or";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIXOrIOp : VMI_Op<"xori", [Pure]> {
  let summary = "VMI integer elementwise bitwise xor";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIShLIOp : VMI_Op<"shli", [Pure]> {
  let summary = "VMI integer elementwise left shift";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIShRUIOp : VMI_Op<"shrui", [Pure]> {
  let summary = "VMI unsigned integer elementwise right shift";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIShRSIOp : VMI_Op<"shrsi", [Pure]> {
  let summary = "VMI signed integer elementwise arithmetic right shift";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMINotOp : VMI_Op<"not", [Pure]> {
  let summary = "VMI integer elementwise bitwise not";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMICmpFOp : VMI_Op<"cmpf", [Pure]> {
  let summary = "VMI floating-point elementwise compare";
  let arguments = (ins StrAttr:$predicate, VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMICmpIOp : VMI_Op<"cmpi", [Pure]> {
  let summary = "VMI integer elementwise compare";
  let arguments = (ins StrAttr:$predicate, VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMISelectOp : VMI_Op<"select", [Pure]> {
  let summary = "VMI elementwise select";
  let arguments = (ins VMI_MaskTypeConstraint:$mask, VMI_VRegTypeConstraint:$true_value,
                       VMI_VRegTypeConstraint:$false_value);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$mask `,` $true_value `,` $false_value attr-dict `:` type($mask) `,` type($true_value) `,` type($false_value) `->` type($result)";
}

def VMIActivePrefixIndexOp : VMI_Op<"active_prefix_index"> {
  let summary = "VMI per-lane active-prefix index from a predicate mask";
  let arguments = (ins VMI_MaskTypeConstraint:$mask);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$mask attr-dict `:` type($mask) `->` type($result)";
}

def VMICompressOp : VMI_Op<"compress"> {
  let summary = "VMI compact active source lanes according to a predicate mask";
  let arguments = (ins VMI_VRegTypeConstraint:$source, VMI_MaskTypeConstraint:$mask);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $mask attr-dict `:` type($source) `,` type($mask) `->` type($result)";
}

def VMICompressStoreOp : VMI_Op<"compress_store", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI store active source lanes contiguously according to a predicate mask";
  let arguments = (ins VMI_VRegTypeConstraint:$value, PtrOrMemRef:$destination,
                       Index:$offset, VMI_MaskTypeConstraint:$mask);
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = "$value `,` $destination `[` $offset `]` `,` $mask attr-dict `:` type($value) `,` type($destination) `,` type($mask)";
}

def VMIReduceAddIOp : VMI_Op<"reduce_addi"> {
  let summary = "VMI masked integer add reduction";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       VMI_MaskTypeConstraint:$mask);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $mask attr-dict `:` type($source) `,` type($mask) `->` type($result)";
}

def VMIReduceAddFOp : VMI_Op<"reduce_addf"> {
  let summary = "VMI masked floating-point add reduction with explicit reassociation permission";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       VMI_MaskTypeConstraint:$mask,
                       OptionalAttr<UnitAttr>:$reassoc);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $mask attr-dict `:` type($source) `,` type($mask) `->` type($result)";
}

def VMIReduceMaxFOp : VMI_Op<"reduce_maxf"> {
  let summary = "VMI masked floating-point maximum reduction";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       VMI_MaskTypeConstraint:$mask);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $mask attr-dict `:` type($source) `,` type($mask) `->` type($result)";
}

def VMIReduceMinFOp : VMI_Op<"reduce_minf"> {
  let summary = "VMI masked floating-point minimum reduction";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       VMI_MaskTypeConstraint:$mask);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $mask attr-dict `:` type($source) `,` type($mask) `->` type($result)";
}

def VMIReduceMaxIOp : VMI_Op<"reduce_maxi"> {
  let summary = "VMI masked integer maximum reduction";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       VMI_MaskTypeConstraint:$mask);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $mask attr-dict `:` type($source) `,` type($mask) `->` type($result)";
}

def VMIReduceMinIOp : VMI_Op<"reduce_mini"> {
  let summary = "VMI masked integer minimum reduction";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       VMI_MaskTypeConstraint:$mask);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $mask attr-dict `:` type($source) `,` type($mask) `->` type($result)";
}

def VMIGroupReduceAddFOp : VMI_Op<"group_reduce_addf"> {
  let summary = "VMI masked floating-point add reduction within fixed logical groups";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       VMI_MaskTypeConstraint:$mask,
                       I64Attr:$num_groups,
                       OptionalAttr<UnitAttr>:$reassoc);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $mask attr-dict `:` type($source) `,` type($mask) `->` type($result)";
}

def VMIGroupReduceMaxFOp : VMI_Op<"group_reduce_maxf"> {
  let summary = "VMI masked floating-point maximum reduction within fixed logical groups";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       VMI_MaskTypeConstraint:$mask,
                       I64Attr:$num_groups);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $mask attr-dict `:` type($source) `,` type($mask) `->` type($result)";
}

def VMIGroupReduceMinFOp : VMI_Op<"group_reduce_minf"> {
  let summary = "VMI masked floating-point minimum reduction within fixed logical groups";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       VMI_MaskTypeConstraint:$mask,
                       I64Attr:$num_groups);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $mask attr-dict `:` type($source) `,` type($mask) `->` type($result)";
}

def VMIGroupReduceAddIOp : VMI_Op<"group_reduce_addi"> {
  let summary = "VMI masked integer add reduction within fixed logical groups";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       VMI_MaskTypeConstraint:$mask,
                       I64Attr:$num_groups);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $mask attr-dict `:` type($source) `,` type($mask) `->` type($result)";
}

def VMIGroupReduceMaxIOp : VMI_Op<"group_reduce_maxi"> {
  let summary = "VMI masked integer maximum reduction within fixed logical groups";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       VMI_MaskTypeConstraint:$mask,
                       I64Attr:$num_groups);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $mask attr-dict `:` type($source) `,` type($mask) `->` type($result)";
}

def VMIGroupReduceMinIOp : VMI_Op<"group_reduce_mini"> {
  let summary = "VMI masked integer minimum reduction within fixed logical groups";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       VMI_MaskTypeConstraint:$mask,
                       I64Attr:$num_groups);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $mask attr-dict `:` type($source) `,` type($mask) `->` type($result)";
}

def VMIGroupBroadcastOp : VMI_Op<"group_broadcast"> {
  let summary = "VMI broadcast group-slot values back to each logical group";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       I64Attr:$num_groups);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

class VMIHistogramOp<string mnemonic, string summaryText>
    : VMI_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins VMI_VRegTypeConstraint:$acc,
                       VMI_VRegTypeConstraint:$source,
                       VMI_MaskTypeConstraint:$mask);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$acc `,` $source `,` $mask attr-dict `:` type($acc) `,` type($source) `,` type($mask) `->` type($result)";
}

def VMIVdhistOp : VMIHistogramOp<"vdhist",
  "VMI distribution histogram (dhistv2) over unsigned or signless 8-bit source lanes (interpreted as unsigned)">;

def VMIVchistOp : VMIHistogramOp<"vchist",
  "VMI cumulative histogram (chistv2 half-axis) over unsigned or signless 8-bit source lanes (interpreted as unsigned)">;

def VMIExtFOp : VMI_Op<"extf", [Pure]> {
  let summary = "VMI floating-point elementwise extension";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMITruncFOp : VMI_Op<"truncf", [Pure]> {
  let summary = "VMI floating-point elementwise truncation";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                   OptionalAttr<StrAttr>:$rounding,
                   OptionalAttr<StrAttr>:$saturate);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIFPToSIOp : VMI_Op<"fptosi", [Pure]> {
  let summary = "VMI floating-point to signed integer elementwise conversion";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                   OptionalAttr<StrAttr>:$rounding,
                   OptionalAttr<StrAttr>:$saturate);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIFPToUIOp : VMI_Op<"fptoui", [Pure]> {
  let summary = "VMI floating-point to unsigned integer elementwise conversion";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                   OptionalAttr<StrAttr>:$rounding,
                   OptionalAttr<StrAttr>:$saturate);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMISIToFPOp : VMI_Op<"sitofp", [Pure]> {
  let summary = "VMI signed integer to floating-point elementwise conversion";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIExtSIOp : VMI_Op<"extsi", [Pure]> {
  let summary = "VMI signed integer elementwise extension";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIExtUIOp : VMI_Op<"extui", [Pure]> {
  let summary = "VMI unsigned integer elementwise extension";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMITruncIOp : VMI_Op<"trunci", [Pure]> {
  let summary = "VMI saturating integer elementwise truncation";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                   OptionalAttr<StrAttr>:$saturate);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIBitcastOp : VMI_Op<"bitcast", [Pure]> {
  let summary = "VMI bitwise vector reinterpretation";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMILoadOp : VMI_Op<"load", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI logical vector load";
  let arguments = (ins PtrOrMemRef:$source, Index:$offset);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `[` $offset `]` attr-dict `:` type($source) `->` type($result)";
}

def VMIDeinterleaveLoadOp : VMI_Op<"deinterleave_load", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI two-way logical deinterleave load";
  let arguments = (ins PtrOrMemRef:$source, Index:$offset);
  let results = (outs VMI_VRegTypeConstraint:$low, VMI_VRegTypeConstraint:$high);
  let hasVerifier = 1;
  let assemblyFormat = "$source `[` $offset `]` attr-dict `:` type($source) `->` type($low) `,` type($high)";
}

def VMIGroupLoadOp : VMI_Op<"group_load", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI logical grouped vector load with a row stride between groups";
  let arguments = (ins PtrOrMemRef:$source, Index:$offset, Index:$row_stride,
                       I64Attr:$num_groups);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `[` $offset `]` `,` $row_stride attr-dict `:` type($source) `->` type($result)";
}

def VMIGroupSlotLoadOp : VMI_Op<"group_slot_load", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI load one scalar value per logical group into group slots";
  let arguments = (ins PtrOrMemRef:$source, Index:$offset, Index:$source_group_stride,
                       I64Attr:$num_groups);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `[` $offset `]` `,` $source_group_stride attr-dict `:` type($source) `->` type($result)";
}

def VMIGroupBroadcastLoadOp : VMI_Op<"group_broadcast_load", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI load one scalar value per logical group and broadcast it to group lanes";
  let arguments = (ins PtrOrMemRef:$source, Index:$offset, Index:$source_group_stride,
                       I64Attr:$num_groups);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `[` $offset `]` `,` $source_group_stride attr-dict `:` type($source) `->` type($result)";
}

def VMIStrideLoadOp : VMI_Op<"stride_load", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI block-strided vector load";
  let arguments = (ins PtrOrMemRef:$source, Index:$offset,
                       I16:$block_stride, VMI_MaskTypeConstraint:$mask);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source `[` $offset `]` `,` $block_stride `,` $mask attr-dict `:`
    type($source) `,` type($block_stride) `,` type($mask) `->` type($result)
  }];
}

def VMIMaskedLoadOp : VMI_Op<"masked_load", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI logical masked vector load with passthrough lanes";
  let arguments = (ins PtrOrMemRef:$source, Index:$offset,
                       VMI_MaskTypeConstraint:$mask,
                       VMI_VRegTypeConstraint:$passthru);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `[` $offset `]` `,` $mask `,` $passthru attr-dict `:` type($source) `,` type($mask) `,` type($passthru) `->` type($result)";
}

def VMIGatherOp : VMI_Op<"gather", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI logical masked indexed gather with passthrough lanes";
  let description = [{
    Gathers elements from UB memory using an index vector.
    i8/ui8 sources may be gathered into matching i16/ui16 results. The
    8-to-16 promotion is a zero-extension: one byte is read per lane and the
    upper 8 bits are zero-filled, regardless of whether the source element
    type is signed (i8) or unsigned (ui8). Sign-extension is not supported.
  }];
  let arguments = (ins PtrOrMemRef:$source,
                       VMI_VRegTypeConstraint:$indices,
                       VMI_MaskTypeConstraint:$mask,
                       VMI_VRegTypeConstraint:$passthru);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `[` $indices `]` `,` $mask `,` $passthru attr-dict `:` type($source) `,` type($indices) `,` type($mask) `,` type($passthru) `->` type($result)";
}

def VMIExpandLoadOp : VMI_Op<"expand_load", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI load a dense active-lane stream into masked logical lanes";
  let arguments = (ins PtrOrMemRef:$source, Index:$offset,
                       VMI_MaskTypeConstraint:$mask,
                       VMI_VRegTypeConstraint:$passthru);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `[` $offset `]` `,` $mask `,` $passthru attr-dict `:` type($source) `,` type($mask) `,` type($passthru) `->` type($result)";
}

def VMIStoreOp : VMI_Op<"store", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI logical vector store";
  let arguments = (ins VMI_VRegTypeConstraint:$value, PtrOrMemRef:$destination, Index:$offset);
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = "$value `,` $destination `[` $offset `]` attr-dict `:` type($value) `,` type($destination)";
}

def VMIInterleaveStoreOp : VMI_Op<"interleave_store", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI two-way logical interleave store";
  let arguments = (ins VMI_VRegTypeConstraint:$low, VMI_VRegTypeConstraint:$high,
                       PtrOrMemRef:$destination, Index:$offset);
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = "$low `,` $high `,` $destination `[` $offset `]` attr-dict `:` type($low) `,` type($high) `,` type($destination)";
}

def VMIGroupStoreOp : VMI_Op<"group_store", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI logical grouped vector store with a row stride between groups";
  let arguments = (ins VMI_VRegTypeConstraint:$value, PtrOrMemRef:$destination,
                       Index:$offset, Index:$row_stride, I64Attr:$num_groups);
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = "$value `,` $destination `[` $offset `]` `,` $row_stride attr-dict `:` type($value) `,` type($destination)";
}

def VMIMaskedStoreOp : VMI_Op<"masked_store", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI logical masked vector store";
  let arguments = (ins VMI_VRegTypeConstraint:$value, PtrOrMemRef:$destination,
                       Index:$offset, VMI_MaskTypeConstraint:$mask);
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = "$value `,` $destination `[` $offset `]` `,` $mask attr-dict `:` type($value) `,` type($destination) `,` type($mask)";
}

def VMIStrideStoreOp : VMI_Op<"stride_store", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI block-strided vector store";
  let arguments = (ins VMI_VRegTypeConstraint:$value, PtrOrMemRef:$destination,
                       Index:$offset, I16:$block_stride,
                       VMI_MaskTypeConstraint:$mask);
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $value `,` $destination `[` $offset `]` `,` $block_stride `,` $mask attr-dict `:`
    type($value) `,` type($destination) `,` type($block_stride) `,` type($mask)
  }];
}

def VMIScatterOp : VMI_Op<"scatter", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI logical masked indexed scatter";
  let arguments = (ins VMI_VRegTypeConstraint:$value,
                       PtrOrMemRef:$destination,
                       VMI_VRegTypeConstraint:$indices,
                       VMI_MaskTypeConstraint:$mask);
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = "$value `,` $destination `[` $indices `]` `,` $mask attr-dict `:` type($value) `,` type($destination) `,` type($indices) `,` type($mask)";
}

def VMIShuffleOp : VMI_Op<"shuffle", [Pure]> {
  let summary = "VMI static lane shuffle";
  let arguments = (ins VMI_VRegTypeConstraint:$source, DenseI64ArrayAttr:$indices);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `[` $indices `]` attr-dict `:` type($source) `->` type($result)";
}

def VMIChannelSplitOp : VMI_Op<"channel_split"> {
  let summary = "VMI split interleaved logical channels";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs Variadic<VMI_VRegTypeConstraint>:$results);
  let hasVerifier = 1;
}

def VMIChannelMergeOp : VMI_Op<"channel_merge"> {
  let summary = "VMI merge logical channels by interleaving";
  let arguments = (ins Variadic<VMI_VRegTypeConstraint>:$inputs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
}

def VMIEnsureLayoutOp : VMI_Op<"ensure_layout", [Pure]> {
  let summary = "Internal VMI data layout materialization helper";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIEnsureMaskLayoutOp : VMI_Op<"ensure_mask_layout", [Pure]> {
  let summary = "Internal VMI mask layout materialization helper";
  let arguments = (ins VMI_MaskTypeConstraint:$source);
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIEnsureMaskGranularityOp : VMI_Op<"ensure_mask_granularity", [Pure]> {
  let summary = "Internal VMI mask granularity materialization helper";
  let arguments = (ins VMI_MaskTypeConstraint:$source);
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIUnpackOp : VMI_Op<"unpack"> {
  let summary = "Internal VMI value projection to physical parts";
  let arguments = (ins VMI_ValueTypeConstraint:$source);
  let results = (outs Variadic<PTO_PhysicalVMIPartTypeConstraint>:$parts);
  let hasVerifier = 1;
}

def VMIPackOp : VMI_Op<"pack"> {
  let summary = "Internal physical parts materialized as one VMI value";
  let arguments = (ins Variadic<PTO_PhysicalVMIPartTypeConstraint>:$parts);
  let results = (outs VMI_ValueTypeConstraint:$result);
  let hasVerifier = 1;
}

//===--- Unified (new) VMI ops ---===//

def VMIVbrcOp : VMI_Op<"vbrc", [Pure]> {
  let summary = "VMI broadcast: scalar/1-lane to vector, or group-slot to dense vector";
  let arguments = (ins AnyType:$value, OptionalAttr<I64Attr>:$group);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

def VMIVciOp : VMI_Op<"vci", [Pure]> {
  let summary = "Create a VMI logical index vector from a scalar base";
  let description = [{
    Unified form of `iota`. Without `{group}`, continuous ramp over L lanes.
    With `{group = C}`, group-periodic restart at `base` per group (see
    `iota`); identical group runs share one physical index register after
    lowering. Non-contiguous result layouts rewrite to contiguous iota +
    `ensure_layout` in unified→legacy. Lowers to `iota` preserving `{group}`.
  }];
  let arguments = (ins
    AnyTypeOf<[AnyInteger, AnyFloat], "integer/float scalar">:$base,
    OptionalAttr<StrAttr>:$order,
    OptionalAttr<I64Attr>:$group
  );
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$base attr-dict `:` type($base) `->` type($result)";
}

def VMIPsetOp : VMI_Op<"pset", [Pure]> {
  let summary = "Create a predicate mask with all lanes active";
  let description = [{
    Creates a mask where all lanes are active (set to true).
    Replacement for the all-active mode of `create_mask`.

    When {group = C} is present, creates a grouped all-active mask
    (all lanes active within each of the C groups).

    Example:
    ```
    %m = pto.vmi.pset "PAT_ALL" : !pto.vmi.mask<16>
    %m2 = pto.vmi.pset "PAT_ALL" {group = 8} : !pto.vmi.mask<256xpred>
    ```
  }];
  let arguments = (ins StrAttr:$pattern, OptionalAttr<I64Attr>:$group);
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$pattern attr-dict `:` type($result)";
}

def VMIPgeOp : VMI_Op<"pge", [Pure]> {
  let summary = "Create a tail predicate mask with the first N logical lanes active";
  let description = [{
    Creates a mask where the first N logical lanes are active and the
    remaining lanes are inactive. Replacement for the tail mode of
    `create_mask`.

    When {group = C} is present, creates a grouped tail mask where
    the first N lanes within each of the C groups are active.
    Replacement for `create_group_mask`.

    Example:
    ```
    %m = pto.vmi.pge "PAT_VL16" : !pto.vmi.mask<32>
    %m2 = pto.vmi.pge "PAT_VL25" {group = 8} : !pto.vmi.mask<256xpred>
    ```
  }];
  let arguments = (ins StrAttr:$pattern, OptionalAttr<I64Attr>:$group);
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$pattern attr-dict `:` type($result)";
}

def VMIPltOp : VMI_Op<"plt", [Pure]> {
  let summary = "Data-dependent tail mask from scalar remainder";
  let description = [{
    Creates a mask where the first min(rem, L) lanes are active
    and returns the remainder for the next chunk.

    The mask granularity (b8/b16/b32) is inferred from the result type.
    Chaining:

    ```
    %m1, %next1 = pto.vmi.plt %rem : i32 -> !pto.vmi.mask<128>, i32
    %m2, %next2 = pto.vmi.plt %next1 : i32 -> !pto.vmi.mask<128>, i32
    ```

    Example:
    ```
    // 250 elements across 128-lane chunks
    // Round 1: rem=250 → mask all-active (clamped to 128), next=122
    // Round 2: rem=122 → mask VL122, next=0
    %m1, %n1 = pto.vmi.plt %rem : i32 -> !pto.vmi.mask<128>, i32
    %m2, %n2 = pto.vmi.plt %n1  : i32 -> !pto.vmi.mask<128>, i32
    ```
  }];
  let arguments = (ins I32:$scalar);
  let results = (outs VMI_MaskTypeConstraint:$mask, I32:$scalar_out);
  let hasVerifier = 1;
  let assemblyFormat = "$scalar attr-dict `:` type($scalar) `->` type($mask) `,` type($scalar_out)";
}

def VMIVaddOp : VMI_Op<"vadd", [Pure]> {
  let summary = "VMI elementwise add (unified fp/int)";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs,
                       Variadic<VMI_MaskTypeConstraint>:$mask,
                       OptionalAttr<StrAttr>:$pmode);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs (`,` $mask^)? attr-dict `:` type($lhs) `,` type($rhs) (`,` type($mask)^)? `->` type($result)";
}

def VMIVaddcOp : VMI_Op<"vaddc", [Pure]> {
  let summary = "VMI integer add with per-lane carry output";
  let description = [{
    Adds two integer vectors and returns both the truncated result and the
    per-lane carry predicate.  The operation is restricted to the 32-bit
    integer forms supported by the underlying VPTO carry instruction.
  }];
  let arguments = (ins
    VMI_VRegTypeConstraint:$lhs,
    VMI_VRegTypeConstraint:$rhs,
    VMI_MaskTypeConstraint:$mask
  );
  let results = (outs
    VMI_VRegTypeConstraint:$result,
    VMI_MaskTypeConstraint:$carry
  );
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs `,` $mask attr-dict `:` type($lhs) `,` type($rhs) `,` type($mask) `->` type($result) `,` type($carry)";
}

def VMIVaddcsOp : VMI_Op<"vaddcs", [Pure]> {
  let summary = "VMI integer add with carry-in and carry-out";
  let description = [{
    Adds two integer vectors and a per-lane carry-in predicate, returning both
    the truncated result and the per-lane carry-out predicate.  The operation
    is restricted to the 32-bit integer forms supported by the underlying VPTO
    carry instruction.
  }];
  let arguments = (ins
    VMI_VRegTypeConstraint:$lhs,
    VMI_VRegTypeConstraint:$rhs,
    VMI_MaskTypeConstraint:$carry_in,
    VMI_MaskTypeConstraint:$mask
  );
  let results = (outs
    VMI_VRegTypeConstraint:$result,
    VMI_MaskTypeConstraint:$carry
  );
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs `,` $carry_in `,` $mask attr-dict `:` type($lhs) `,` type($rhs) `,` type($carry_in) `,` type($mask) `->` type($result) `,` type($carry)";
}

def VMIVsubOp : VMI_Op<"vsub", [Pure]> {
  let summary = "VMI elementwise subtract (unified fp/int)";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs,
                       Variadic<VMI_MaskTypeConstraint>:$mask,
                       OptionalAttr<StrAttr>:$pmode);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs (`,` $mask^)? attr-dict `:` type($lhs) `,` type($rhs) (`,` type($mask)^)? `->` type($result)";
}

def VMIVmulOp : VMI_Op<"vmul", [Pure]> {
  let summary = "VMI elementwise multiply (unified fp/int)";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs,
                       Variadic<VMI_MaskTypeConstraint>:$mask,
                       OptionalAttr<StrAttr>:$pmode);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs (`,` $mask^)? attr-dict `:` type($lhs) `,` type($rhs) (`,` type($mask)^)? `->` type($result)";
}

def VMIVdivOp : VMI_Op<"vdiv", [Pure]> {
  let summary = "VMI elementwise divide";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs,
                       Variadic<VMI_MaskTypeConstraint>:$mask,
                       OptionalAttr<StrAttr>:$pmode);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs (`,` $mask^)? attr-dict `:` type($lhs) `,` type($rhs) (`,` type($mask)^)? `->` type($result)";
}

def VMIVminOp : VMI_Op<"vmin", [Pure]> {
  let summary = "VMI elementwise minimum";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs,
                       Variadic<VMI_MaskTypeConstraint>:$mask,
                       OptionalAttr<StrAttr>:$pmode);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs (`,` $mask^)? attr-dict `:` type($lhs) `,` type($rhs) (`,` type($mask)^)? `->` type($result)";
}

def VMIVmaxOp : VMI_Op<"vmax", [Pure]> {
  let summary = "VMI elementwise maximum";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs,
                       Variadic<VMI_MaskTypeConstraint>:$mask,
                       OptionalAttr<StrAttr>:$pmode);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs (`,` $mask^)? attr-dict `:` type($lhs) `,` type($rhs) (`,` type($mask)^)? `->` type($result)";
}

def VMIVnegOp : VMI_Op<"vneg", [Pure]> {
  let summary = "VMI elementwise negate";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       Variadic<VMI_MaskTypeConstraint>:$mask,
                       OptionalAttr<StrAttr>:$pmode);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source (`,` $mask^)? attr-dict `:` type($source) (`,` type($mask)^)? `->` type($result)";
}

def VMIVabsOp : VMI_Op<"vabs", [Pure]> {
  let summary = "VMI elementwise absolute value (float/int unified)";
  let description = [{
    Computes the elementwise absolute value of a logical vector.
    Supports floating-point and explicitly signed integer element types.

    pmode governs inactive-lane behavior:
    - "merge" (default): inactive lanes pass through the source value.
    - "zero": inactive lanes are zeroed.

    Example:
    ```
    %r = pto.vmi.vabs %v, %m {pmode = "zero"}
      : !pto.vmi.vreg<128xsi32>, !pto.vmi.mask<128xpred> -> !pto.vmi.vreg<128xsi32>
    ```
  }];
  let arguments = (ins
    VMI_VRegTypeConstraint:$source,
    Variadic<VMI_MaskTypeConstraint>:$mask,
    OptionalAttr<StrAttr>:$pmode
  );
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source (`,` $mask^)? attr-dict `:` type($source) (`,` type($mask)^)? `->` type($result)";
}

def VMIVsqrtOp : VMI_Op<"vsqrt", [Pure]> {
  let summary = "VMI elementwise square root";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       Variadic<VMI_MaskTypeConstraint>:$mask,
                       OptionalAttr<StrAttr>:$pmode);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source (`,` $mask^)? attr-dict `:` type($source) (`,` type($mask)^)? `->` type($result)";
}

def VMIVexpOp : VMI_Op<"vexp", [Pure]> {
  let summary = "VMI elementwise exponential";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       Variadic<VMI_MaskTypeConstraint>:$mask,
                       OptionalAttr<StrAttr>:$pmode);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source (`,` $mask^)? attr-dict `:` type($source) (`,` type($mask)^)? `->` type($result)";
}

def VMIVlnOp : VMI_Op<"vln", [Pure]> {
  let summary = "VMI elementwise natural logarithm";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       Variadic<VMI_MaskTypeConstraint>:$mask,
                       OptionalAttr<StrAttr>:$pmode);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source (`,` $mask^)? attr-dict `:` type($source) (`,` type($mask)^)? `->` type($result)";
}

def VMIVreluOp : VMI_Op<"vrelu", [Pure]> {
  let summary = "VMI elementwise ReLU";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       Variadic<VMI_MaskTypeConstraint>:$mask,
                       OptionalAttr<StrAttr>:$pmode);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source (`,` $mask^)? attr-dict `:` type($source) (`,` type($mask)^)? `->` type($result)";
}

def VMIVandOp : VMI_Op<"vand", [Pure]> {
  let summary = "VMI elementwise bitwise and";
  let arguments = (ins VMI_ValueTypeConstraint:$lhs, VMI_ValueTypeConstraint:$rhs,
                       Variadic<VMI_MaskTypeConstraint>:$mask,
                       OptionalAttr<StrAttr>:$pmode);
  let results = (outs VMI_ValueTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs (`,` $mask^)? attr-dict `:` type($lhs) `,` type($rhs) (`,` type($mask)^)? `->` type($result)";
}

def VMIVorOp : VMI_Op<"vor", [Pure]> {
  let summary = "VMI elementwise bitwise or";
  let arguments = (ins VMI_ValueTypeConstraint:$lhs, VMI_ValueTypeConstraint:$rhs,
                       Variadic<VMI_MaskTypeConstraint>:$mask,
                       OptionalAttr<StrAttr>:$pmode);
  let results = (outs VMI_ValueTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs (`,` $mask^)? attr-dict `:` type($lhs) `,` type($rhs) (`,` type($mask)^)? `->` type($result)";
}

def VMIVxorOp : VMI_Op<"vxor", [Pure]> {
  let summary = "VMI elementwise bitwise xor";
  let arguments = (ins VMI_ValueTypeConstraint:$lhs, VMI_ValueTypeConstraint:$rhs,
                       Variadic<VMI_MaskTypeConstraint>:$mask,
                       OptionalAttr<StrAttr>:$pmode);
  let results = (outs VMI_ValueTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs (`,` $mask^)? attr-dict `:` type($lhs) `,` type($rhs) (`,` type($mask)^)? `->` type($result)";
}

def VMIVshlOp : VMI_Op<"vshl", [Pure]> {
  let summary = "VMI elementwise left shift";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs,
                       Variadic<VMI_MaskTypeConstraint>:$mask,
                       OptionalAttr<StrAttr>:$pmode);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs (`,` $mask^)? attr-dict `:` type($lhs) `,` type($rhs) (`,` type($mask)^)? `->` type($result)";
}

def VMIVshrOp : VMI_Op<"vshr", [Pure]> {
  let summary = "VMI signedness-aware elementwise right shift";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs,
                       Variadic<VMI_MaskTypeConstraint>:$mask,
                       OptionalAttr<StrAttr>:$pmode);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs (`,` $mask^)? attr-dict `:` type($lhs) `,` type($rhs) (`,` type($mask)^)? `->` type($result)";
}

def VMIVnotOp : VMI_Op<"vnot", [Pure]> {
  let summary = "VMI elementwise bitwise not";
  let arguments = (ins VMI_ValueTypeConstraint:$source,
                       Variadic<VMI_MaskTypeConstraint>:$mask,
                       OptionalAttr<StrAttr>:$pmode);
  let results = (outs VMI_ValueTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source (`,` $mask^)? attr-dict `:` type($source) (`,` type($mask)^)? `->` type($result)";
}

def VMIVcmpOp : VMI_Op<"vcmp", [Pure]> {
  let summary = "VMI elementwise compare (fp/int unified) → predicate mask";
  let description = [{
    Compares two logical vector registers elementwise and produces a predicate
    mask. The comparison mode is given by the `cmp` attribute. The `seed`
    operand provides the governing predicate Pg: where seed[i]=0 the result
    lane is zero (pmode="zeroing", default), and where seed[i]=1 the comparison
    result is written. Floating-point comparisons support eq/ne/lt/le/gt/ge
    and ordered forms oeq/one/olt/ole/ogt/oge. Integer comparisons use
    eq/ne/lt/le/gt/ge; signedness is selected by the integer element type.
    Signless integers use unsigned semantics (signless/unsigned versus signed).

    Example:
    ```
    %r = pto.vmi.vcmp %a, %b, %seed {cmp = "lt"}
      : !pto.vmi.vreg<128xf32>, !pto.vmi.vreg<128xf32>, !pto.vmi.mask<128xb32>
      -> !pto.vmi.mask<128xb32>
    ```
  }];

  let arguments = (ins
    VMI_VRegTypeConstraint:$lhs,
    VMI_VRegTypeConstraint:$rhs,
    VMI_MaskTypeConstraint:$seed,
    StrAttr:$cmp,
    OptionalAttr<StrAttr>:$pmode
  );
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $lhs `,` $rhs `,` $seed attr-dict `:` type($lhs) `,` type($rhs) `,` type($seed) `->` type($result)
  }];
}

def VMIVcmpsOp : VMI_Op<"vcmps", [Pure]> {
  let summary = "VMI elementwise vector-scalar compare → predicate mask";
  let description = [{
    Compares each lane of a logical vector against a scalar and produces a
    predicate mask. The scalar type must match the vector element type. The
    `seed` operand provides the governing predicate Pg.

    Example:
    ```
    %r = pto.vmi.vcmps %v, %s, %seed {cmp = "ge"}
      : !pto.vmi.vreg<64xf16>, f16, !pto.vmi.mask<64xpred>
      -> !pto.vmi.mask<64xpred>
    ```
  }];

  let arguments = (ins
    VMI_VRegTypeConstraint:$src,
    AnyType:$scalar,
    VMI_MaskTypeConstraint:$seed,
    StrAttr:$cmp,
    OptionalAttr<StrAttr>:$pmode
  );
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $src `,` $scalar `,` $seed attr-dict `:` type($src) `,` type($scalar) `,` type($seed) `->` type($result)
  }];
}

class VMI_VecScalarOp<string mnemonic, string summaryText,
                      Type scalarType = AnyType>
    : VMI_Op<mnemonic, [Pure]> {
  let summary = summaryText;
  let description = [{
    Performs elementwise operation between a logical vector and a scalar.
    The scalar is implicitly broadcast to all lanes (R6/M2).

    pmode governs inactive-lane behavior on the result:
    - "merge" (default): inactive lanes pass through the source value.
    - "zero": inactive lanes are zeroed.

    Example:
    ```
    %r = pto.vmi.} # mnemonic # [{ %v, %s, %m {pmode = "merge"}
      : !pto.vmi.vreg<64xf16>, f16, !pto.vmi.mask<64xpred> -> !pto.vmi.vreg<64xf16>
    ```
  }];
  let arguments = (ins
    VMI_VRegTypeConstraint:$src,
    scalarType:$scalar,
    VMI_MaskTypeConstraint:$mask,
    OptionalAttr<StrAttr>:$pmode
  );
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $src `,` $scalar `,` $mask attr-dict `:` type($src) `,` type($scalar) `,` type($mask) `->` type($result)
  }];
}

def VMIAddSOp : VMI_VecScalarOp<"vadds", "VMI vector-scalar elementwise add">;
def VMIMulSOp : VMI_VecScalarOp<"vmuls", "VMI vector-scalar elementwise multiply">;
def VMIMaxSOp : VMI_VecScalarOp<"vmaxs", "VMI vector-scalar elementwise maximum">;
def VMIMinSOp : VMI_VecScalarOp<"vmins", "VMI vector-scalar elementwise minimum">;
def VMIShlSOp : VMI_VecScalarOp<"vshls", "VMI vector-scalar elementwise shift left", I16>;
def VMIShrSOp : VMI_VecScalarOp<"vshrs", "VMI vector-scalar elementwise shift right", I16>;

def VMIvSelOp : VMI_Op<"vsel", [Pure]> {
  let summary = "VMI elementwise select";
  let description = [{
    Per-lane selection driven by a predicate mask.  For each lane i:

      result[i] = mask[i] ? true_value[i] : false_value[i]

    pmode governs inactive-lane behavior on the result:
    - "merge" (default): inactive lanes retain the false_value element.
    - "zero": inactive lanes are zeroed.

    Example:
    ```mlir
    %r = pto.vmi.vsel %m, %a, %b {pmode = "zero"}
      : !pto.vmi.mask<256xpred>, !pto.vmi.vreg<256xui16>, !pto.vmi.vreg<256xui16> -> !pto.vmi.vreg<256xui16>
    ```
  }];
  let arguments = (ins VMI_MaskTypeConstraint:$mask, VMI_VRegTypeConstraint:$true_value,
                       VMI_VRegTypeConstraint:$false_value,
                       OptionalAttr<StrAttr>:$pmode);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$mask `,` $true_value `,` $false_value attr-dict `:` type($mask) `,` type($true_value) `,` type($false_value) `->` type($result)";
}

def VMIvcaddOp : VMI_Op<"vcadd", [Pure]> {
  let summary = "VMI add-reduction: sum lanes to a scalar or compact group vector";
  let description = [{
    Reduces source lanes by summation, governed by a mask.

    - No init operand: inactive lanes contribute 0 (semantically nil in
      addition).  Result reaches a single scalar when `group` is absent,
      or `C` compact scalars when `{group=C}` is specified.
    - Floating-point sources MUST carry the `reassoc` attribute (hardware
      vcadd uses pair-wise reassociation).
    - `pmode` controls inactive-result behaviour: "zero" (default) writes 0;
      "merge" preserves the destination register's old value.
  }];
  let arguments = (ins
    VMI_VRegTypeConstraint:$source,
    VMI_MaskTypeConstraint:$mask,
    OptionalAttr<I64Attr>:$group,
    OptionalAttr<StrAttr>:$pmode,
    OptionalAttr<UnitAttr>:$reassoc
  );
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $mask attr-dict `:` type($source) `,` type($mask) `->` type($result)";
}

def VMIvcmaxOp : VMI_Op<"vcmax", [Pure]> {
  let summary = "VMI max-reduction: maximum across lanes to a scalar or compact group vector";
  let description = [{
    Reduces source lanes by taking the element-wise maximum, governed by a mask.

    - Inactive lanes are treated as -INF (floating-point) or the type's
      minimum representable value (integer).
    - `pmode` controls inactive-result behaviour as for `vcadd`.
  }];
  let arguments = (ins
    VMI_VRegTypeConstraint:$source,
    VMI_MaskTypeConstraint:$mask,
    OptionalAttr<I64Attr>:$group,
    OptionalAttr<StrAttr>:$pmode
  );
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $mask attr-dict `:` type($source) `,` type($mask) `->` type($result)";
}

def VMIvcminOp : VMI_Op<"vcmin", [Pure]> {
  let summary = "VMI min-reduction: minimum across lanes to a scalar or compact group vector";
  let description = [{
    Reduces source lanes by taking the element-wise minimum, governed by a mask.

    - Inactive lanes are treated as +INF (floating-point) or the type's
      maximum representable value (integer).
    - `pmode` controls inactive-result behaviour as for `vcadd`.
  }];
  let arguments = (ins
    VMI_VRegTypeConstraint:$source,
    VMI_MaskTypeConstraint:$mask,
    OptionalAttr<I64Attr>:$group,
    OptionalAttr<StrAttr>:$pmode
  );
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $mask attr-dict `:` type($source) `,` type($mask) `->` type($result)";
}

def VMICvtOp : VMI_Op<"vcvt", [Pure]> {
  let summary = "VMI unified elementwise type conversion";
  let description = [{
    Converts a VMI logical vector from one element type to another.
    The conversion direction and semantics are derived from the source and result
    element types:

    - fp → fp, |dst| > |src|:  floating-point widening   (replaces extf)
    - fp → fp, |dst| < |src|:  floating-point narrowing   (replaces truncf)
    - fp → int:                 signedness follows the result type
    - int → fp:                 currently requires an explicitly signed source
    - int → int, |dst| > |src|: integer extension          (replaces extsi/extui)
    - int → int, |dst| < |src|: integer truncation         (replaces trunci)

    Attributes:
    - `rounding`: rounding mode for fp narrowing and fp-to-integer
      conversions. Narrowing accepts R=nearest-even, A=away-from-zero,
      H=half-up, Z=toward-zero; fp-to-integer accepts the hardware modes
      R/A/F/C/Z.
    - `saturate`: "SAT" or "NOSAT"; required for fp narrowing, integer
      narrowing, and fp-to-int conversion.
    - `pmode`: predication mode ("merge" | "zero").

    Signless integer element types use unsigned semantics. Thus fp→signless
    follows the unsigned conversion path, and signless int→int widening uses
    zero extension. Signed conversion semantics require explicit `siN` types.

    Example:
    ```mlir
    %w = pto.vmi.vcvt %x : !pto.vmi.vreg<64xf16> -> !pto.vmi.vreg<64xf32>
    %n = pto.vmi.vcvt %y {rounding="H"} : !pto.vmi.vreg<64xf32> -> !pto.vmi.vreg<64xf16>
    %z = pto.vmi.vcvt %a : !pto.vmi.vreg<256xui8> -> !pto.vmi.vreg<256xui16>
    ```
  }];
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                   OptionalAttr<StrAttr>:$rounding,
                   OptionalAttr<StrAttr>:$saturate,
                   OptionalAttr<StrAttr>:$pmode);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIvLoadOp : VMI_Op<"vload",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     AttrSizedOperandSegments]> {
  let summary = "VMI unified logical vector load";
  let description = [{
    Unified vector load.  dist-mode selects the memory-access pattern.

    dist-mode values:
    - "continuous" (default): contiguous stride-1 load  → 1 result;
      1/2/4/8-lane results alias group=1/2/4/8 with unit stride
    - "dintlv":             deinterleaved dual load      → 2 results (%lo, %hi)
    - "brc":                broadcast load               → 1 result

    group mode (mutually exclusive with dist-mode):
    - {group = C}: grouped vector load with row stride  → 1 result

    block-stride mode (mutually exclusive with dist-mode and group):
    - {block_stride = B}: block-strided masked load → 1 result

    pmode ("zero"|"merge") governs inactive-lane behavior on the result.
    vload itself has no mask operand; A5 loads are not predicable — masks
    migrate to consumer ops or vstore.
  }];
  let arguments = (ins
    PtrOrMemRef:$source,
    Index:$offset,
    Optional<Index>:$stride,
    Optional<I16>:$block_stride,
    OptionalAttr<StrAttr>:$dist_mode,
    OptionalAttr<I64Attr>:$group,
    OptionalAttr<StrAttr>:$pmode
  );
  let results = (outs Variadic<VMI_VRegTypeConstraint>:$results);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def VMIvStoreOp : VMI_Op<"vstore",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     AttrSizedOperandSegments]> {
  let summary = "VMI unified logical vector store";
  let description = [{
    Unified vector store replacing the individual store, interleave_store,
    masked_store, stride_store, and group_store ops.

    dist-mode controls the memory access pattern:
    - "continuous" (default): contiguous stride-1 store  → 1 value;
      1/2/4/8-lane inputs alias group=1/2/4/8 with unit stride
    - "intlv":              interleaved dual store        → 2 values (%lo, %hi)

    group mode (mutually exclusive with dist-mode):
    - {group = C}: grouped vector store with row stride

    block-stride mode (mutually exclusive with dist-mode and group):
    - {block_stride = B}: block-strided masked store

    pmode governs inactive lane behavior:
    - "zero" (default):  inactive lanes store 0
    - "merge":           inactive lanes skip write (needs mask)

    mask is variadic: 0 or 1 mask operand.
  }];
  let arguments = (ins
    Variadic<VMI_VRegTypeConstraint>:$values,
    PtrOrMemRef:$destination,
    Index:$offset,
    Optional<Index>:$stride,
    Optional<I16>:$block_stride,
    Variadic<VMI_MaskTypeConstraint>:$mask,
    OptionalAttr<StrAttr>:$dist_mode,
    OptionalAttr<I64Attr>:$group,
    OptionalAttr<StrAttr>:$pmode
  );
  let results = (outs);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def VMIVsstbOp : VMI_Op<"vsstb",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI zero-repeat-stride block store";
  let arguments = (ins VMI_VRegTypeConstraint:$value,
                       PtrOrMemRef:$destination, Index:$offset,
                       I16:$block_stride, VMI_MaskTypeConstraint:$mask,
                       OptionalAttr<StrAttr>:$pmode);
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = "$value `,` $destination `[` $offset `]` `,` $block_stride `,` $mask attr-dict `:` type($value) `,` type($destination) `,` type($mask)";
}

def VMIVinterpretCastOp : VMI_Op<"vinterpret_cast", [Pure]> {
  let summary = "VMI bitwise vector reinterpretation";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIVselrOp : VMI_Op<"vselr", [Pure]> {
  let summary = "VMI dynamic lane select";
  let description = [{
    Dynamic lane permutation: for each lane i, result[i] = source[index[i]].
    Replaces the static-index `shuffle` op with a dynamic index vector operand.
    Lowers 1:1 to `pto.vselr` for contiguous vectors with 64, 128, or 256 lanes
    for 8-bit elements; 64 or 128 lanes for 16-bit elements; or 64 lanes for
    32-bit elements.  Source, index, and result use the same lane count, and
    the index element storage width must match the source element storage
    width.  Other lane counts are not supported by this operation.

    Example:
    ```
    %r = pto.vmi.vselr %src, %idx : !pto.vmi.vreg<128xf16>, !pto.vmi.vreg<128xi16> -> !pto.vmi.vreg<128xf16>
    ```
  }];
  let arguments = (ins
    VMI_VRegTypeConstraint:$source,
    VMI_VRegTypeConstraint:$index
  );
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source `,` $index attr-dict `:` type($source) `,` type($index) `->` type($result)
  }];
}

def VMIVintlvOp : VMI_Op<"vintlv"> {
  let summary = "VMI interleave two vectors lane-by-lane (Category A: layout-transparent)";
  let arguments = (ins
    VMI_VRegTypeConstraint:$lhs,
    VMI_VRegTypeConstraint:$rhs,
    VMI_MaskTypeConstraint:$mask,
    OptionalAttr<StrAttr>:$pmode
  );
  let results = (outs
    VMI_VRegTypeConstraint:$low,
    VMI_VRegTypeConstraint:$high
  );
  let hasVerifier = 1;
  let assemblyFormat = [{
    $lhs `,` $rhs `,` $mask attr-dict `:` type($lhs) `,` type($rhs) `,` type($mask) `->` type($low) `,` type($high)
  }];
}

def VMIVdintlvOp : VMI_Op<"vdintlv"> {
  let summary = "VMI deinterleave two vectors by even/odd lanes (Category A: layout-transparent)";
  let arguments = (ins
    VMI_VRegTypeConstraint:$lhs,
    VMI_VRegTypeConstraint:$rhs,
    VMI_MaskTypeConstraint:$mask,
    OptionalAttr<StrAttr>:$pmode
  );
  let results = (outs
    VMI_VRegTypeConstraint:$low,
    VMI_VRegTypeConstraint:$high
  );
  let hasVerifier = 1;
  let assemblyFormat = [{
    $lhs `,` $rhs `,` $mask attr-dict `:` type($lhs) `,` type($rhs) `,` type($mask) `->` type($low) `,` type($high)
  }];
}

def VMIVgatherOp : VMI_Op<"vgather", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI logical masked indexed gather";
  let description = [{
    Gathers elements from UB memory using an index vector.
    i8/ui8 sources may be gathered into matching i16/ui16 results. The
    8-to-16 promotion is a zero-extension: one byte is read per lane and the
    upper 8 bits are zero-filled, regardless of whether the source element
    type is signed (i8) or unsigned (ui8). Sign-extension is not supported.
    Inactive lanes are controlled by pmode (no explicit passthru operand).

    pmode controls inactive-lane behavior:
    - "zero" (default): inactive lanes are zeroed.
    - "merge": inactive lanes preserve the prior destination value.

    Example:
    ```
    %g = pto.vmi.vgather %src, %offsets, %mask
      : !pto.ptr<f32, ub>, !pto.vmi.vreg<64xi32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>
    ```
  }];
  let arguments = (ins
    PtrOrMemRef:$source,
    VMI_VRegTypeConstraint:$offsets,
    VMI_MaskTypeConstraint:$mask,
    OptionalAttr<StrAttr>:$pmode
  );
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $offsets `,` $mask attr-dict `:` type($source) `,` type($offsets) `,` type($mask) `->` type($result)";
}

def VMIVgatherbOp : VMI_Op<"vgatherb", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI logical masked byte-granularity indexed gather";
  let description = [{
    Gathers elements from UB memory using byte-offset indices.
    Mask lane count equals result lane count (may differ from offset lane count).

    Example:
    ```
    %gb = pto.vmi.vgatherb %src, %offsets, %mask
      : !pto.ptr<i32, ub>, !pto.vmi.vreg<64xi32>, !pto.vmi.mask<256> -> !pto.vmi.vreg<256xi32>
    ```
  }];
  let arguments = (ins
    PtrOrMemRef:$source,
    VMI_VRegTypeConstraint:$offsets,
    VMI_MaskTypeConstraint:$mask,
    OptionalAttr<StrAttr>:$pmode
  );
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $offsets `,` $mask attr-dict `:` type($source) `,` type($offsets) `,` type($mask) `->` type($result)";
}

def VMIVscatterOp : VMI_Op<"vscatter", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI logical masked indexed scatter";
  let description = [{
    Scatters elements to UB memory using an index vector.
    Inactive lanes are controlled by pmode.

    Example:
    ```
    pto.vmi.vscatter %v, %dest, %offsets, %mask
      : !pto.vmi.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.vmi.vreg<64xi32>, !pto.vmi.mask<64>
    ```
  }];
  let arguments = (ins
    VMI_VRegTypeConstraint:$value,
    PtrOrMemRef:$destination,
    VMI_VRegTypeConstraint:$offsets,
    VMI_MaskTypeConstraint:$mask,
    OptionalAttr<StrAttr>:$pmode
  );
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = "$value `,` $destination `,` $offsets `,` $mask attr-dict `:` type($value) `,` type($destination) `,` type($offsets) `,` type($mask)";
}

def VMIVexpdifOp : VMI_Op<"vexpdif", [Pure]> {
  let summary = "VMI fused exp(x - max) for softmax numerical stability";
  let description = [{
    Computes exp(x - max). x and max use the same f16 or f32 element type.
    The result preserves the logical lane count and always uses f32 elements.

    Example:
    ```
    %e = pto.vmi.vexpdif %x, %max, %mask
      : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>
    ```
  }];
  let arguments = (ins
    VMI_VRegTypeConstraint:$x,
    VMI_VRegTypeConstraint:$max,
    VMI_MaskTypeConstraint:$mask,
    OptionalAttr<StrAttr>:$pmode
  );
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$x `,` $max `,` $mask attr-dict `:` type($x) `,` type($max) `,` type($mask) `->` type($result)";
}

def VMIVaxpyOp : VMI_Op<"vaxpy", [Pure]> {
  let summary = "VMI fused alpha*x + y (scale-add)";
  let description = [{
    Computes y = alpha * x + y in a single hardware instruction.

    Example:
    ```
    %y = pto.vmi.vaxpy %x, %acc, %alpha, %mask
      : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, f32, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>
    ```
  }];
  let arguments = (ins
    VMI_VRegTypeConstraint:$x,
    VMI_VRegTypeConstraint:$acc,
    AnyTypeOf<[AnyFloat]>:$alpha,
    VMI_MaskTypeConstraint:$mask,
    OptionalAttr<StrAttr>:$pmode
  );
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
}

def VMIVlreluOp : VMI_Op<"vlrelu", [Pure]> {
  let summary = "VMI leaky ReLU activation";
  let description = [{
    Computes y = x > 0 ? x : slope * x using a single hardware instruction.
    The slope is a scalar shared across all lanes.

    Example:
    ```
    %lr = pto.vmi.vlrelu %x, %slope, %mask
      : !pto.vmi.vreg<64xf32>, f32, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>
    ```
  }];
  let arguments = (ins
    VMI_VRegTypeConstraint:$x,
    AnyTypeOf<[AnyFloat]>:$slope,
    VMI_MaskTypeConstraint:$mask,
    OptionalAttr<StrAttr>:$pmode
  );
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
}

def VMIVpreluOp : VMI_Op<"vprelu", [Pure]> {
  let summary = "VMI parametric ReLU activation";
  let description = [{
    Computes y = max(x, 0) + alpha * min(x, 0) where alpha is a per-lane
    parameter vector (not a shared scalar).

    Example:
    ```
    %pr = pto.vmi.vprelu %x, %alpha, %mask
      : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64xf32>
    ```
  }];
  let arguments = (ins
    VMI_VRegTypeConstraint:$x,
    VMI_VRegTypeConstraint:$alpha,
    VMI_MaskTypeConstraint:$mask,
    OptionalAttr<StrAttr>:$pmode
  );
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$x `,` $alpha `,` $mask attr-dict `:` type($x) `,` type($alpha) `,` type($mask) `->` type($result)";
}

def VMIVmullOp : VMI_Op<"vmull", [Pure]> {
  let summary = "VMI widening 32x32 multiply with low/high results";
  let description = [{
    Computes a per-lane 32-bit x 32-bit product and returns its low and high
    32-bit halves. Inactive lanes in both results are zero. The initial
    contract accepts exactly 64, 128, or 256 lanes of i32 or ui32 data.

    Example:
    ```
    %low, %high = pto.vmi.vmull %a, %b, %mask
      : !pto.vmi.vreg<64xi32>, !pto.vmi.vreg<64xi32>,
        !pto.vmi.mask<64xpred>
        -> !pto.vmi.vreg<64xi32>, !pto.vmi.vreg<64xi32>
    ```
  }];
  let arguments = (ins
    VMI_VRegTypeConstraint:$a,
    VMI_VRegTypeConstraint:$b,
    VMI_MaskTypeConstraint:$mask,
    OptionalAttr<StrAttr>:$pmode
  );
  let results = (outs
    VMI_VRegTypeConstraint:$low,
    VMI_VRegTypeConstraint:$high
  );
  let hasVerifier = 1;
  let assemblyFormat = "$a `,` $b `,` $mask attr-dict `:` type($a) `,` type($b) `,` type($mask) `->` type($low) `,` type($high)";
}

def VMIVmulaOp : VMI_Op<"vmula", [Pure]> {
  let summary = "VMI fused multiply-add (acc = acc + lhs * rhs)";
  let description = [{
    Computes acc = acc + lhs * rhs in a single hardware instruction.
    The accumulator operand comes first to match hardware FMA semantics.

    Supports both floating-point (f16/bf16/f32) and integer (i8-i32) types.
    The mask operand is optional; when present it is only a semantic annotation
    and is discarded during lowering to legacy fma.

    Example:
    ```
    %acc1 = pto.vmi.vmula %acc, %a, %b, %mask
      : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64>
        -> !pto.vmi.vreg<64xf32>

    // Without mask:
    %acc2 = pto.vmi.vmula %acc, %a, %b
      : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>
        -> !pto.vmi.vreg<64xf32>
    ```
  }];
  let arguments = (ins
    VMI_VRegTypeConstraint:$acc,
    VMI_VRegTypeConstraint:$lhs,
    VMI_VRegTypeConstraint:$rhs,
    Variadic<VMI_MaskTypeConstraint>:$mask,
    OptionalAttr<StrAttr>:$pmode
  );
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$acc `,` $lhs `,` $rhs (`,` $mask^)? attr-dict `:` type($acc) `,` type($lhs) `,` type($rhs) (`,` type($mask)^)? `->` type($result)";
}
#endif // MLIR_DIALECT_PTO_IR_VMIOPS
