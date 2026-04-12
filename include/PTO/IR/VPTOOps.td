//===- VPTOOps.td - PTO low-level operations ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_VPTOOPS
#define MLIR_DIALECT_PTO_IR_VPTOOPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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

def PTO_BufferType : Type<
  CPred<"::llvm::isa<::mlir::pto::PtrType>($_self)">,
  "pointer-like buffer type">;
def PTO_BufferLikeType : AnyTypeOf<[AnyMemRef, PTO_BufferType],
                                    "memref or pointer-like buffer type">;

def VecScopeOp : PTO_Op<"vecscope", [SingleBlock, NoTerminator]> {
  let summary = "Structured region container for one VPTO vector scope";
  let description = [{
    `pto.vecscope` marks a structured vector-scope interval without overloading
    a dummy carrier loop with scope metadata. Lowering and emission passes may
    use the region boundary to preserve loop shape while treating the enclosed
    body as one VPTO vector interval.
  }];

  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let assemblyFormat = "$body attr-dict";
}

def StrictVecScopeOp : PTO_Op<"strict_vecscope", [SingleBlock, NoTerminator,
                                                  IsolatedFromAbove]> {
  let summary = "Structured VPTO vector scope with explicit captures only";
  let description = [{
    `pto.strict_vecscope` is the strict form of `pto.vecscope`. Values used by
    the body must be passed explicitly through op operands and corresponding
    block arguments; implicit SSA capture from the surrounding scope is
    rejected.
  }];

  let arguments = (ins Variadic<AnyType>:$captures);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` $captures `)` $body attr-dict `:` functional-type($captures, results)
  }];
}

class PTO_BinaryI64ConfigOp<string mnemonic> : PTO_Op<mnemonic> {
  let arguments = (ins
    I64:$first,
    I64:$second
  );

  let results = (outs);

  let assemblyFormat = [{
    $first `,` $second attr-dict `:` type($first) `,` type($second)
  }];
}

def PTO_SetLoop2StrideOutToUbOp : PTO_BinaryI64ConfigOp<"set_loop2_stride_outtoub">;
def PTO_SetLoop1StrideOutToUbOp : PTO_BinaryI64ConfigOp<"set_loop1_stride_outtoub">;
def PTO_SetLoopSizeOutToUbOp : PTO_BinaryI64ConfigOp<"set_loop_size_outtoub">;
def PTO_SetLoop2StrideUbToOutOp : PTO_BinaryI64ConfigOp<"set_loop2_stride_ubtoout">;
def PTO_SetLoop1StrideUbToOutOp : PTO_BinaryI64ConfigOp<"set_loop1_stride_ubtoout">;
def PTO_SetLoopSizeUbToOutOp : PTO_BinaryI64ConfigOp<"set_loop_size_ubtoout">;

def PTO_CopyGmToUbufOp : PTO_Op<"copy_gm_to_ubuf", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$sid,
    I64:$n_burst,
    I64:$len_burst,
    I64:$left_padding_count,
    I64:$right_padding_count,
    I1:$data_select_bit,
    I64:$l2_cache_ctl,
    I64:$gm_stride,
    I64:$ub_stride
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $sid `,` $n_burst `,` $len_burst `,`
    $left_padding_count `,` $right_padding_count `,` $data_select_bit `,` $l2_cache_ctl `,` $gm_stride `,` $ub_stride
    attr-dict `:` type($source) `,` type($destination) `,` type($sid) `,` type($n_burst) `,`
    type($len_burst) `,` type($left_padding_count) `,`
    type($right_padding_count) `,` type($data_select_bit) `,` type($l2_cache_ctl) `,` type($gm_stride) `,` type($ub_stride)
  }];
}

def PTO_CopyUbufToUbufOp : PTO_Op<"copy_ubuf_to_ubuf"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$sid,
    I64:$n_burst,
    I64:$len_burst,
    I64:$src_stride,
    I64:$dst_stride
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $sid `,` $n_burst `,` $len_burst `,` $src_stride `,` $dst_stride
    attr-dict `:` type($source) `,` type($destination) `,` type($sid) `,` type($n_burst) `,`
    type($len_burst) `,` type($src_stride) `,` type($dst_stride)
  }];
}

def PTO_VldsOp : PTO_Op<"vlds", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    Index:$offset,
    OptionalAttr<StrAttr>:$dist
  );

  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $offset `]` attr-dict `:` type($source) `->` type($result)
  }];
}

def PTO_VldsPostOp : PTO_Op<"vlds_post", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    Index:$offset,
    OptionalAttr<StrAttr>:$dist
  );

  let results = (outs PTO_VectorType:$result,
                      PTO_BufferLikeType:$updated_source);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $offset `]` attr-dict `:` type($source) `->` type($result) `,` type($updated_source)
  }];
}

def PTO_Vldsx2Op : PTO_Op<"vldsx2", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    Index:$offset,
    StrAttr:$dist
  );
  let results = (outs PTO_VectorType:$low, PTO_VectorType:$high);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $offset `]` `,` $dist attr-dict `:` type($source) `,` type($offset) `->` type($low) `,` type($high)
  }];
}

def PTO_VldasOp : PTO_Op<"vldas", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$source
  );

  let results = (outs PTO_AlignTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` type($result)
  }];
}

def PTO_InitAlignOp : PTO_Op<"init_align", []> {
  let arguments = (ins);

  let results = (outs PTO_AlignTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

def PTO_SprclrOp : PTO_Op<"sprclr", []> {
  let arguments = (ins StrAttr:$spr);

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $spr attr-dict
  }];
}

def PTO_VldusOp : PTO_Op<"vldus", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_AlignTypeConstraint:$align
  );

  let results = (outs
    PTO_VectorType:$result,
    PTO_AlignTypeConstraint:$updated_align
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $align attr-dict `:` type($source) `,` type($align) `->` type($result) `,` type($updated_align)
  }];
}

def PTO_UvldOp : PTO_Op<"uvld", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    Index:$offset
  );

  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $offset `]` attr-dict `:` type($source) `->` type($result)
  }];
}

def PTO_VbrOp : PTO_Op<"vbr", [Pure]> {
  let arguments = (ins AnyType:$value);
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result)
  }];
}

def PTO_VdupOp : PTO_Op<"vdup", [Pure]> {
  let arguments = (ins
    AnyType:$input,
    PTO_MaskTypeConstraint:$mask,
    OptionalAttr<StrAttr>:$position
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $mask attr-dict `:` type($input) `,` type($mask) `->` type($result)
  }];
}

def PTO_PsetB8Op : PTO_Op<"pset_b8", [Pure]> {
  let arguments = (ins StrAttr:$pattern);
  let results = (outs PTO_B8MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $pattern attr-dict `:` type($result)
  }];
}

def PTO_PsetB16Op : PTO_Op<"pset_b16", [Pure]> {
  let arguments = (ins StrAttr:$pattern);
  let results = (outs PTO_B16MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $pattern attr-dict `:` type($result)
  }];
}

// NOTE: The op families introduced below are intentionally marked as
// unvalidated scaffolding. They are added to preserve missing CCE builtin
// semantics at the dialect layer, but they have not yet been validated through
// PTO lowering or end-to-end sample execution.
def PTO_PsetB32Op : PTO_Op<"pset_b32", [Pure]> {
  let arguments = (ins StrAttr:$pattern);
  let results = (outs PTO_B32MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $pattern attr-dict `:` type($result)
  }];
}

def PTO_PgeB8Op : PTO_Op<"pge_b8", [Pure]> {
  let arguments = (ins StrAttr:$pattern);
  let results = (outs PTO_B8MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $pattern attr-dict `:` type($result)
  }];
}

def PTO_PgeB16Op : PTO_Op<"pge_b16", [Pure]> {
  let arguments = (ins StrAttr:$pattern);
  let results = (outs PTO_B16MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $pattern attr-dict `:` type($result)
  }];
}

def PTO_PgeB32Op : PTO_Op<"pge_b32", [Pure]> {
  let arguments = (ins StrAttr:$pattern);
  let results = (outs PTO_B32MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $pattern attr-dict `:` type($result)
  }];
}

def PTO_PltB8Op : PTO_Op<"plt_b8", [Pure]> {
  let arguments = (ins I32:$scalar);
  let results = (outs PTO_B8MaskTypeConstraint:$mask, I32:$scalar_out);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $scalar attr-dict `:` type($scalar) `->` type($mask) `,` type($scalar_out)
  }];
}

def PTO_PltB16Op : PTO_Op<"plt_b16", [Pure]> {
  let arguments = (ins I32:$scalar);
  let results = (outs PTO_B16MaskTypeConstraint:$mask, I32:$scalar_out);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $scalar attr-dict `:` type($scalar) `->` type($mask) `,` type($scalar_out)
  }];
}

def PTO_PltB32Op : PTO_Op<"plt_b32", [Pure]> {
  let arguments = (ins I32:$scalar);
  let results = (outs PTO_B32MaskTypeConstraint:$mask, I32:$scalar_out);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $scalar attr-dict `:` type($scalar) `->` type($mask) `,` type($scalar_out)
  }];
}

class PTO_MaskUnaryOp<string mnemonic> : PTO_Op<mnemonic, [Pure]> {
  let arguments = (ins PTO_MaskTypeConstraint:$input);
  let results = (outs PTO_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

def PTO_PpackOp : PTO_MaskUnaryOp<"ppack"> {
  let arguments = (ins PTO_MaskTypeConstraint:$input, StrAttr:$part);
  let assemblyFormat = [{
    $input `,` $part attr-dict `:` type($input) `->` type($result)
  }];
}

def PTO_PunpackOp : PTO_MaskUnaryOp<"punpack"> {
  let arguments = (ins PTO_MaskTypeConstraint:$input, StrAttr:$part);
  let assemblyFormat = [{
    $input `,` $part attr-dict `:` type($input) `->` type($result)
  }];
}

def PTO_PnotOp : PTO_Op<"pnot", [Pure]> {
  let arguments = (ins
    PTO_MaskTypeConstraint:$input,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $mask attr-dict `:` type($input) `,` type($mask) `->` type($result)
  }];
}

def PTO_PselOp : PTO_Op<"psel", [Pure]> {
  let arguments = (ins
    PTO_MaskTypeConstraint:$src0,
    PTO_MaskTypeConstraint:$src1,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 `,` $mask attr-dict `:` type($src0) `,` type($src1) `,` type($mask) `->` type($result)
  }];
}

def PTO_PandOp : PTO_Op<"pand", [Pure]> {
  let arguments = (ins
    PTO_MaskTypeConstraint:$src0,
    PTO_MaskTypeConstraint:$src1,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 `,` $mask attr-dict `:` type($src0) `,` type($src1) `,` type($mask) `->` type($result)
  }];
}

def PTO_PorOp : PTO_Op<"por", [Pure]> {
  let arguments = (ins
    PTO_MaskTypeConstraint:$src0,
    PTO_MaskTypeConstraint:$src1,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 `,` $mask attr-dict `:` type($src0) `,` type($src1) `,` type($mask) `->` type($result)
  }];
}

def PTO_PxorOp : PTO_Op<"pxor", [Pure]> {
  let arguments = (ins
    PTO_MaskTypeConstraint:$src0,
    PTO_MaskTypeConstraint:$src1,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 `,` $mask attr-dict `:` type($src0) `,` type($src1) `,` type($mask) `->` type($result)
  }];
}

def PTO_PldsOp : PTO_Op<"plds", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    Index:$offset,
    StrAttr:$dist
  );
  let results = (outs PTO_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $offset `]` `,` $dist attr-dict `:` type($source) `,` type($offset) `->` type($result)
  }];
}

def PTO_PldiOp : PTO_Op<"pldi", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    Index:$offset,
    StrAttr:$dist
  );
  let results = (outs PTO_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $offset `]` `,` $dist attr-dict `:` type($source) `,` type($offset) `->` type($result)
  }];
}

def PTO_PstiOp : PTO_Op<"psti", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_MaskTypeConstraint:$value,
    PTO_BufferLikeType:$destination,
    Index:$offset,
    StrAttr:$dist
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `[` $offset `]` `,` $dist attr-dict `:` type($value) `,` type($destination) `,` type($offset)
  }];
}

def PTO_VabsOp : PTO_Op<"vabs", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$input,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $mask attr-dict `:` type($input) `,` type($mask) `->` type($result)
  }];
}

class PTO_UnaryVecOp<string mnemonic> : PTO_Op<mnemonic, [Pure]> {
  let arguments = (ins
    PTO_VectorType:$input,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $mask attr-dict `:` type($input) `,` type($mask) `->` type($result)
  }];
}

def PTO_VexpOp : PTO_UnaryVecOp<"vexp">;
def PTO_VlnOp : PTO_UnaryVecOp<"vln">;
def PTO_VsqrtOp : PTO_UnaryVecOp<"vsqrt">;
def PTO_VnegOp : PTO_UnaryVecOp<"vneg">;
def PTO_VrsqrtOp : PTO_UnaryVecOp<"vrsqrt">;
def PTO_VrecOp : PTO_UnaryVecOp<"vrec">;
def PTO_VreluOp : PTO_UnaryVecOp<"vrelu">;
def PTO_VnotOp : PTO_UnaryVecOp<"vnot">;
def PTO_VcaddOp : PTO_UnaryVecOp<"vcadd">;
def PTO_VcmaxOp : PTO_UnaryVecOp<"vcmax">;
def PTO_VcminOp : PTO_UnaryVecOp<"vcmin">;
def PTO_VcgaddOp : PTO_UnaryVecOp<"vcgadd">;
def PTO_VcgmaxOp : PTO_UnaryVecOp<"vcgmax">;
def PTO_VcgminOp : PTO_UnaryVecOp<"vcgmin">;
def PTO_VcpaddOp : PTO_UnaryVecOp<"vcpadd">;

class PTO_BinaryVecOp<string mnemonic> : PTO_Op<mnemonic, [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $mask attr-dict `:` type($lhs) `,` type($rhs) `,` type($mask) `->` type($result)
  }];
}

def PTO_VaddOp : PTO_BinaryVecOp<"vadd">;
def PTO_VsubOp : PTO_BinaryVecOp<"vsub">;
def PTO_VsaddOp : PTO_BinaryVecOp<"vsadd">;
def PTO_VssubOp : PTO_BinaryVecOp<"vssub">;
def PTO_VmulOp : PTO_BinaryVecOp<"vmul">;
def PTO_VdivOp : PTO_BinaryVecOp<"vdiv">;
def PTO_VmaxOp : PTO_BinaryVecOp<"vmax">;
def PTO_VminOp : PTO_BinaryVecOp<"vmin">;
def PTO_VandOp : PTO_BinaryVecOp<"vand">;
def PTO_VorOp : PTO_BinaryVecOp<"vor">;
def PTO_VxorOp : PTO_BinaryVecOp<"vxor">;

def PTO_VaddcOp : PTO_Op<"vaddc", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs
    PTO_VectorType:$result,
    PTO_MaskTypeConstraint:$carry
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $mask attr-dict `:` type($lhs) `,` type($rhs) `,` type($mask) `->` type($result) `,` type($carry)
  }];
}

def PTO_VsubcOp : PTO_Op<"vsubc", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs
    PTO_VectorType:$result,
    PTO_MaskTypeConstraint:$carry
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $mask attr-dict `:` type($lhs) `,` type($rhs) `,` type($mask) `->` type($result) `,` type($carry)
  }];
}

def PTO_VaddcsOp : PTO_Op<"vaddcs", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs,
    PTO_MaskTypeConstraint:$carry_in,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs
    PTO_VectorType:$result,
    PTO_MaskTypeConstraint:$carry
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $carry_in `,` $mask attr-dict `:` type($lhs) `,` type($rhs) `,` type($carry_in) `,` type($mask) `->` type($result) `,` type($carry)
  }];
}

def PTO_VsubcsOp : PTO_Op<"vsubcs", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs,
    PTO_MaskTypeConstraint:$carry_in,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs
    PTO_VectorType:$result,
    PTO_MaskTypeConstraint:$carry
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $carry_in `,` $mask attr-dict `:` type($lhs) `,` type($rhs) `,` type($carry_in) `,` type($mask) `->` type($result) `,` type($carry)
  }];
}

def PTO_VbcntOp : PTO_UnaryVecOp<"vbcnt">;
def PTO_VclsOp : PTO_UnaryVecOp<"vcls">;

def PTO_VshlOp : PTO_BinaryVecOp<"vshl">;
def PTO_VshrOp : PTO_BinaryVecOp<"vshr">;

def PTO_VselOp : PTO_Op<"vsel", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$src0,
    PTO_VectorType:$src1,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 `,` $mask attr-dict `:` type($src0) `,` type($src1) `,` type($mask) `->` type($result)
  }];
}

def PTO_VcmpOp : PTO_Op<"vcmp", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$src0,
    PTO_VectorType:$src1,
    PTO_MaskTypeConstraint:$mask,
    StrAttr:$cmp_mode
  );
  let results = (outs PTO_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 `,` $mask `,` $cmp_mode attr-dict `:` type($src0) `,` type($src1) `,` type($mask) `->` type($result)
  }];
}

def PTO_VcmpsOp : PTO_Op<"vcmps", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$src,
    AnyType:$scalar,
    PTO_MaskTypeConstraint:$mask,
    StrAttr:$cmp_mode
  );
  let results = (outs PTO_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src `,` $scalar `,` $mask `,` $cmp_mode attr-dict `:` type($src) `,` type($scalar) `,` type($mask) `->` type($result)
  }];
}

class PTO_PredicatePairReorderOp<string mnemonic, Type operandTy>
    : PTO_Op<mnemonic, [Pure]> {
  let arguments = (ins
    operandTy:$lhs,
    operandTy:$rhs
  );
  let results = (outs
    operandTy:$low,
    operandTy:$high
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($low) `,` type($high)
  }];
}

def PTO_PdintlvB8Op : PTO_PredicatePairReorderOp<"pdintlv_b8",
    PTO_B8MaskTypeConstraint>;
def PTO_PdintlvB16Op : PTO_PredicatePairReorderOp<"pdintlv_b16",
    PTO_B16MaskTypeConstraint>;
def PTO_PdintlvB32Op : PTO_PredicatePairReorderOp<"pdintlv_b32",
    PTO_B32MaskTypeConstraint>;

def PTO_PintlvB8Op : PTO_PredicatePairReorderOp<"pintlv_b8",
    PTO_B8MaskTypeConstraint>;
def PTO_PintlvB16Op : PTO_PredicatePairReorderOp<"pintlv_b16",
    PTO_B16MaskTypeConstraint>;
def PTO_PintlvB32Op : PTO_PredicatePairReorderOp<"pintlv_b32",
    PTO_B32MaskTypeConstraint>;

class PTO_VecScalarOp<string mnemonic> : PTO_Op<mnemonic, [Pure]> {
  let arguments = (ins
    PTO_VectorType:$input,
    AnyType:$scalar
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $scalar attr-dict `:` type($input) `,` type($scalar) `->` type($result)
  }];
}

class PTO_VecScalarMaskedOp<string mnemonic> : PTO_Op<mnemonic, [Pure]> {
  let arguments = (ins
    PTO_VectorType:$input,
    AnyType:$scalar,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $scalar `,` $mask attr-dict `:` type($input) `,` type($scalar) `,` type($mask) `->` type($result)
  }];
}

def PTO_VtrcOp : PTO_Op<"vtrc", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$input,
    StrAttr:$round_mode
  );
  let results = (outs PTO_VectorType:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def PTO_VcvtOp : PTO_Op<"vcvt", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$input,
    OptionalAttr<StrAttr>:$rnd,
    OptionalAttr<StrAttr>:$sat,
    OptionalAttr<StrAttr>:$part
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_VciOp : PTO_Op<"vci", [Pure]> {
  let arguments = (ins
    AnyInteger:$index,
    OptionalAttr<StrAttr>:$order
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $index attr-dict `:` type($index) `->` type($result)
  }];
}

def PTO_VbitsortOp : PTO_Op<"vbitsort", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$destination,
    PTO_BufferType:$source,
    PTO_BufferType:$indices,
    Index:$repeat_times
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $destination `,` $source `,` $indices `,` $repeat_times attr-dict `:` type($destination) `,`
    type($source) `,` type($indices) `,` type($repeat_times)
  }];
}

def PTO_Vmrgsort4Op : PTO_Op<"vmrgsort4"> {
  let arguments = (ins
    PTO_BufferType:$destination,
    PTO_BufferType:$source0,
    PTO_BufferType:$source1,
    PTO_BufferType:$source2,
    PTO_BufferType:$source3,
    I64:$count,
    I64:$config
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $destination `,` $source0 `,` $source1 `,` $source2 `,` $source3 `,` $count `,` $config
    attr-dict `:` type($destination) `,` type($source0) `,` type($source1) `,` type($source2) `,`
    type($source3) `,` type($count) `,` type($config)
  }];
}

def PTO_Vgather2Op : PTO_Op<"vgather2", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_VectorType:$offsets,
    Index:$active_lanes
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $offsets `,` $active_lanes attr-dict `:` type($source) `,` type($offsets) `,` type($active_lanes) `->` type($result)
  }];
}

def PTO_VgatherbOp : PTO_Op<"vgatherb", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_VectorType:$offsets,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $offsets `,` $mask attr-dict `:` type($source) `,` type($offsets) `,` type($mask) `->` type($result)
  }];
}

// NOTE: Unvalidated new gather/select/interleave-family abstractions. Added to
// cover CCE builtin families not yet exercised through end-to-end PTO seams.
def PTO_Vgather2BcOp : PTO_Op<"vgather2_bc", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_VectorType:$offsets,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $offsets `,` $mask attr-dict `:` type($source) `,` type($offsets) `,` type($mask) `->` type($result)
  }];
}

def PTO_VmulsOp : PTO_VecScalarMaskedOp<"vmuls">;
def PTO_VaddsOp : PTO_VecScalarMaskedOp<"vadds">;
def PTO_VsaddsOp : PTO_VecScalarMaskedOp<"vsadds">;
def PTO_VmaxsOp : PTO_VecScalarMaskedOp<"vmaxs">;
def PTO_VminsOp : PTO_VecScalarMaskedOp<"vmins">;
def PTO_VlreluOp : PTO_VecScalarMaskedOp<"vlrelu">;
def PTO_VshlsOp : PTO_VecScalarMaskedOp<"vshls">;
def PTO_VshrsOp : PTO_VecScalarMaskedOp<"vshrs">;

def PTO_VstsOp : PTO_Op<"vsts", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_VectorType:$value,
    PTO_BufferLikeType:$destination,
    Index:$offset,
    OptionalAttr<StrAttr>:$dist,
    PTO_MaskTypeConstraint:$mask
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `[` $offset `]` `,` $mask attr-dict `:` type($value) `,` type($destination) `,` type($mask)
  }];
}

def PTO_VstsPostOp : PTO_Op<"vsts_post", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_VectorType:$value,
    PTO_BufferLikeType:$destination,
    Index:$offset,
    OptionalAttr<StrAttr>:$dist,
    PTO_MaskTypeConstraint:$mask
  );

  let results = (outs PTO_BufferLikeType:$updated_destination);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `[` $offset `]` `,` $mask attr-dict `:` type($value) `,` type($destination) `,` type($mask) `->` type($updated_destination)
  }];
}

def PTO_VscatterOp : PTO_Op<"vscatter", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_VectorType:$value,
    PTO_BufferType:$destination,
    PTO_VectorType:$offsets,
    Index:$active_lanes
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `,` $offsets `,` $active_lanes attr-dict `:` type($value) `,` type($destination) `,` type($offsets) `,` type($active_lanes)
  }];
}

def PTO_PstsOp : PTO_Op<"psts", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_MaskTypeConstraint:$value,
    PTO_BufferLikeType:$destination,
    Index:$offset,
    StrAttr:$dist
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `[` $offset `]` `,` $dist attr-dict `:` type($value) `,` type($destination) `,` type($offset)
  }];
}

def PTO_CopyUbufToGmOp : PTO_Op<"copy_ubuf_to_gm", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$sid,
    I64:$n_burst,
    I64:$len_burst,
    I64:$reserved,
    I64:$burst_dst_stride,
    I64:$burst_src_stride
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $sid `,` $n_burst `,` $len_burst `,`
    $reserved `,` $burst_dst_stride `,` $burst_src_stride
    attr-dict `:` type($source) `,` type($destination) `,` type($sid) `,` type($n_burst) `,`
    type($len_burst) `,` type($reserved) `,`
    type($burst_dst_stride) `,` type($burst_src_stride)
  }];
}

// NOTE: Unvalidated new x2 / pair / align-store-family abstractions. Added to
// reflect CCE builtin families but not yet end-to-end validated.
def PTO_VselrOp : PTO_Op<"vselr", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$src0,
    PTO_VectorType:$src1
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 attr-dict `:` type($src0) `,` type($src1) `->` type($result)
  }];
}

def PTO_VslideOp : PTO_Op<"vslide", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$src0,
    PTO_VectorType:$src1,
    I16:$amt
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 `,` $amt attr-dict `:` type($src0) `,` type($src1) `,` type($amt) `->` type($result)
  }];
}

def PTO_VsqzOp : PTO_UnaryVecOp<"vsqz">;

def PTO_VusqzOp : PTO_Op<"vusqz", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$src,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src `,` $mask attr-dict `:` type($src) `,` type($mask) `->` type($result)
  }];
}

def PTO_VpackOp : PTO_Op<"vpack", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$src,
    StrAttr:$part
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src `,` $part attr-dict `:` type($src) `->` type($result)
  }];
}

def PTO_VsunpackOp : PTO_Op<"vsunpack", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$src,
    Index:$part
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src `,` $part attr-dict `:` type($src) `->` type($result)
  }];
}

def PTO_VzunpackOp : PTO_Op<"vzunpack", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$src,
    Index:$part
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src `,` $part attr-dict `:` type($src) `->` type($result)
  }];
}

def PTO_Vselrv2Op : PTO_Op<"vselrv2", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$src0,
    PTO_VectorType:$src1
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 attr-dict `:` type($src0) `,` type($src1) `->` type($result)
  }];
}

def PTO_VintlvOp : PTO_Op<"vintlv", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs
  );
  let results = (outs PTO_VectorType:$low, PTO_VectorType:$high);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($low) `,` type($high)
  }];
}

def PTO_VdintlvOp : PTO_Op<"vdintlv", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs
  );
  let results = (outs PTO_VectorType:$low, PTO_VectorType:$high);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($low) `,` type($high)
  }];
}

def PTO_Vintlvv2Op : PTO_Op<"vintlvv2", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs,
    StrAttr:$part
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $part attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def PTO_Vdintlvv2Op : PTO_Op<"vdintlvv2", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs,
    StrAttr:$part
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $part attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def PTO_VmullOp : PTO_Op<"vmull", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$low, PTO_VectorType:$high);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $mask attr-dict `:` type($lhs) `,` type($rhs) `,` type($mask) `->` type($low) `,` type($high)
  }];
}

def PTO_VmulaOp : PTO_Op<"vmula", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$acc,
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $acc `,` $lhs `,` $rhs `,` $mask attr-dict `:` type($acc) `,` type($lhs) `,` type($rhs) `,` type($mask) `->` type($result)
  }];
}

class PTO_UnmaskedBinaryVecOp<string mnemonic> : PTO_Op<mnemonic, [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def PTO_VpreluOp : PTO_UnmaskedBinaryVecOp<"vprelu">;
def PTO_VexpdiffOp : PTO_Op<"vexpdiff", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$input,
    PTO_VectorType:$max,
    StrAttr:$part
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $max `,` $part attr-dict `:` type($input) `,` type($max) `->` type($result)
  }];
}

def PTO_VaxpyOp : PTO_Op<"vaxpy", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$src0,
    PTO_VectorType:$src1,
    AnyType:$alpha
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 `,` $alpha attr-dict `:` type($src0) `,` type($src1) `,` type($alpha) `->` type($result)
  }];
}

def PTO_VaddreluconvOp : PTO_Op<"vaddreluconv", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def PTO_VmulconvOp : PTO_Op<"vmulconv", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$lhs,
    PTO_VectorType:$rhs
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def PTO_Vstsx2Op : PTO_Op<"vstsx2", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_VectorType:$low,
    PTO_VectorType:$high,
    PTO_BufferLikeType:$destination,
    Index:$offset,
    StrAttr:$dist,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $low `,` $high `,` $destination `[` $offset `]` `,` $dist `,` $mask attr-dict `:` type($low) `,` type($high) `,` type($destination) `,` type($offset) `,` type($mask)
  }];
}

def PTO_VsldbOp : PTO_Op<"vsldb", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    I16:$block_stride,
    I16:$repeat_stride,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $block_stride `,` $repeat_stride `,` $mask attr-dict `:` type($source) `,` type($block_stride) `,` type($repeat_stride) `,` type($mask) `->` type($result)
  }];
}

def PTO_VsstbOp : PTO_Op<"vsstb", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_VectorType:$value,
    PTO_BufferLikeType:$destination,
    I16:$block_stride,
    I16:$repeat_stride,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `,` $block_stride `,` $repeat_stride `,` $mask attr-dict `:` type($value) `,` type($destination) `,` type($block_stride) `,` type($repeat_stride) `,` type($mask)
  }];
}

def PTO_VstasOp : PTO_Op<"vstas", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_AlignTypeConstraint:$value,
    PTO_BufferLikeType:$destination,
    I32:$offset
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `,` $offset attr-dict `:` type($value) `,` type($destination) `,` type($offset)
  }];
}

def PTO_VstarOp : PTO_Op<"vstar", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_AlignTypeConstraint:$value,
    PTO_BufferLikeType:$destination
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination attr-dict `:` type($value) `,` type($destination)
  }];
}

// NOTE: Unvalidated stateful store-family abstractions. These preserve
// align/base/offset update results explicitly in SSA form instead of relying on
// implicit CCE reference updates.
// Keep `base/base_out` pointer-only (`PTO_BufferType`): memref semantics for
// stateful post-update addresses are intentionally out of scope in this change.
def PTO_PstuOp : PTO_Op<"pstu", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_AlignTypeConstraint:$align_in,
    PTO_MaskTypeConstraint:$value,
    PTO_BufferType:$base
  );
  let results = (outs PTO_AlignTypeConstraint:$align_out, PTO_BufferType:$base_out);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $align_in `,` $value `,` $base attr-dict `:` type($align_in) `,` type($value) `,` type($base) `->` type($align_out) `,` type($base_out)
  }];
}

def PTO_VstusOp : PTO_Op<"vstus", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_AlignTypeConstraint:$align_in,
    I32:$offset,
    PTO_VectorType:$value,
    PTO_BufferType:$base
  );
  let results = (outs PTO_AlignTypeConstraint:$align_out);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $align_in `,` $offset `,` $value `,` $base attr-dict `:` type($align_in) `,` type($offset) `,` type($value) `,` type($base) `->` type($align_out)
  }];
}

def PTO_VsturOp : PTO_Op<"vstur", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_AlignTypeConstraint:$align_in,
    PTO_VectorType:$value,
    PTO_BufferType:$base,
    StrAttr:$mode
  );
  let results = (outs PTO_AlignTypeConstraint:$align_out);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $align_in `,` $value `,` $base `,` $mode attr-dict `:` type($align_in) `,` type($value) `,` type($base) `->` type($align_out)
  }];
}

#endif // MLIR_DIALECT_PTO_IR_VPTOOPS
