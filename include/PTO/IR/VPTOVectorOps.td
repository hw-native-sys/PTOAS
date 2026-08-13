// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details.

//===- VPTOVectorOps - semantic operation family -*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_VPTOVECTOROPS
#define MLIR_DIALECT_PTO_IR_VPTOVECTOROPS

include "PTO/IR/VPTOOpCommon.td"

def PTO_VldsOp : PTO_VectorMicroOp<"vlds", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    Index:$offset,
    OptionalAttr<StrAttr>:$dist
  );

  let results = (outs PTO_VectorType:$result,
                      Optional<PTO_BufferLikeType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $offset `]` attr-dict `:` type($source) `->` type($result)
    (`,` type($updated_base)^)?
  }];
}

def PTO_Vldsx2Op : PTO_VectorMicroOp<"vldsx2", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    Index:$offset,
    StrAttr:$dist
  );
  let results = (outs PTO_VectorType:$low,
                      PTO_VectorType:$high,
                      Optional<PTO_BufferLikeType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $offset `]` `,` $dist attr-dict `:` type($source) `,` type($offset) `->` type($low) `,` type($high)
    (`,` type($updated_base)^)?
  }];
}

def PTO_VldasOp : PTO_VectorMicroOp<"vldas", [
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

def PTO_InitAlignOp : PTO_VectorMicroOp<"init_align", []> {
  let arguments = (ins);

  let results = (outs PTO_AlignTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

def PTO_SprclrOp : PTO_VectorMicroOp<"sprclr", []> {
  let arguments = (ins StrAttr:$spr);

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $spr attr-dict
  }];
}

def PTO_SprstiOp : PTO_VectorMicroOp<"sprsti", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    StrAttr:$spr,
    PTO_BufferType:$destination,
    I32:$offset
  );

  let results = (outs Optional<PTO_BufferType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $spr `,` $destination `[` $offset `]` attr-dict `:` type($destination) `,` type($offset)
    (`->` type($updated_base)^)?
  }];
}

def PTO_SprstsOp : PTO_VectorMicroOp<"sprsts", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    StrAttr:$spr,
    PTO_BufferType:$destination,
    I32:$offset
  );

  let results = (outs Optional<PTO_BufferType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $spr `,` $destination `[` $offset `]` attr-dict `:` type($destination) `,` type($offset)
    (`->` type($updated_base)^)?
  }];
}

def PTO_VldusOp : PTO_VectorMicroOp<"vldus", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_AlignTypeConstraint:$align,
    Optional<Index>:$increment
  );

  let results = (outs
    PTO_VectorType:$result,
    PTO_AlignTypeConstraint:$updated_align,
    Optional<PTO_BufferType>:$updated_base
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $align (`,` $increment^)? attr-dict `:` type($source) `,` type($align)
    (`,` type($increment)^)? `->` type($result) `,` type($updated_align)
    (`,` type($updated_base)^)?
  }];
}

def PTO_UvldOp : PTO_VectorMicroOp<"uvld", [
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

def PTO_VbrOp : PTO_VectorMicroOp<"vbr", [Pure]> {
  let arguments = (ins AnyType:$value);
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result)
  }];
}

def PTO_VdupOp : PTO_VectorMicroOp<"vdup", [Pure]> {
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

def PTO_PsetB8Op : PTO_VectorMicroOp<"pset_b8", [Pure]> {
  let arguments = (ins StrAttr:$pattern);
  let results = (outs PTO_B8MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $pattern attr-dict `:` type($result)
  }];
}

def PTO_PsetB16Op : PTO_VectorMicroOp<"pset_b16", [Pure]> {
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
def PTO_PsetB32Op : PTO_VectorMicroOp<"pset_b32", [Pure]> {
  let arguments = (ins StrAttr:$pattern);
  let results = (outs PTO_B32MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $pattern attr-dict `:` type($result)
  }];
}

def PTO_PgeB8Op : PTO_VectorMicroOp<"pge_b8", [Pure]> {
  let arguments = (ins StrAttr:$pattern);
  let results = (outs PTO_B8MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $pattern attr-dict `:` type($result)
  }];
}

def PTO_PgeB16Op : PTO_VectorMicroOp<"pge_b16", [Pure]> {
  let arguments = (ins StrAttr:$pattern);
  let results = (outs PTO_B16MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $pattern attr-dict `:` type($result)
  }];
}

def PTO_PgeB32Op : PTO_VectorMicroOp<"pge_b32", [Pure]> {
  let arguments = (ins StrAttr:$pattern);
  let results = (outs PTO_B32MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $pattern attr-dict `:` type($result)
  }];
}

def PTO_PltB8Op : PTO_VectorMicroOp<"plt_b8", [Pure]> {
  let arguments = (ins I32:$scalar);
  let results = (outs PTO_B8MaskTypeConstraint:$mask, I32:$scalar_out);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $scalar attr-dict `:` type($scalar) `->` type($mask) `,` type($scalar_out)
  }];
}

def PTO_PltB16Op : PTO_VectorMicroOp<"plt_b16", [Pure]> {
  let arguments = (ins I32:$scalar);
  let results = (outs PTO_B16MaskTypeConstraint:$mask, I32:$scalar_out);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $scalar attr-dict `:` type($scalar) `->` type($mask) `,` type($scalar_out)
  }];
}

def PTO_PltB32Op : PTO_VectorMicroOp<"plt_b32", [Pure]> {
  let arguments = (ins I32:$scalar);
  let results = (outs PTO_B32MaskTypeConstraint:$mask, I32:$scalar_out);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $scalar attr-dict `:` type($scalar) `->` type($mask) `,` type($scalar_out)
  }];
}

def PTO_PltmB8Op : PTO_VectorMicroOp<"pltm_b8", [Pure]> {
  let arguments = (ins I16:$loop, I32:$bound);
  let results = (outs PTO_B8MaskTypeConstraint:$mask);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $loop `,` $bound attr-dict `:` type($loop) `,` type($bound) `->` type($mask)
  }];
}

def PTO_PltmB16Op : PTO_VectorMicroOp<"pltm_b16", [Pure]> {
  let arguments = (ins I16:$loop, I32:$bound);
  let results = (outs PTO_B16MaskTypeConstraint:$mask);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $loop `,` $bound attr-dict `:` type($loop) `,` type($bound) `->` type($mask)
  }];
}

def PTO_PltmB32Op : PTO_VectorMicroOp<"pltm_b32", [Pure]> {
  let arguments = (ins I16:$loop, I32:$bound);
  let results = (outs PTO_B32MaskTypeConstraint:$mask);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $loop `,` $bound attr-dict `:` type($loop) `,` type($bound) `->` type($mask)
  }];
}

class PTO_MaskUnaryOp<string mnemonic> : PTO_VectorMicroOp<mnemonic, [Pure]> {
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

def PTO_PbitcastOp : PTO_MaskUnaryOp<"pbitcast">;

def PTO_PnotOp : PTO_VectorMicroOp<"pnot", [Pure]> {
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

def PTO_PselOp : PTO_VectorMicroOp<"psel", [Pure]> {
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

def PTO_PandOp : PTO_VectorMicroOp<"pand", [Pure]> {
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

def PTO_PorOp : PTO_VectorMicroOp<"por", [Pure]> {
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

def PTO_PxorOp : PTO_VectorMicroOp<"pxor", [Pure]> {
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

def PTO_PldsOp : PTO_VectorMicroOp<"plds", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    Index:$offset,
    StrAttr:$dist
  );
  let results = (outs PTO_MaskTypeConstraint:$result,
                      Optional<PTO_BufferLikeType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $offset `]` `,` $dist attr-dict `:` type($source) `,` type($offset) `->` type($result)
    (`,` type($updated_base)^)?
  }];
}

def PTO_PldiOp : PTO_VectorMicroOp<"pldi", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    Index:$offset,
    StrAttr:$dist
  );
  let results = (outs PTO_MaskTypeConstraint:$result,
                      Optional<PTO_BufferLikeType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $offset `]` `,` $dist attr-dict `:` type($source) `,` type($offset) `->` type($result)
    (`,` type($updated_base)^)?
  }];
}

def PTO_PstiOp : PTO_VectorMicroOp<"psti", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_MaskTypeConstraint:$value,
    PTO_BufferLikeType:$destination,
    Index:$offset,
    StrAttr:$dist
  );
  let results = (outs Optional<PTO_BufferLikeType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `[` $offset `]` `,` $dist attr-dict `:` type($value) `,` type($destination) `,` type($offset)
    (`->` type($updated_base)^)?
  }];
}

def PTO_VabsOp : PTO_VectorMicroOp<"vabs", [Pure]> {
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

class PTO_UnaryVecOp<string mnemonic> : PTO_VectorMicroOp<mnemonic, [Pure]> {
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
def PTO_VreluOp : PTO_UnaryVecOp<"vrelu">;
def PTO_VnotOp : PTO_UnaryVecOp<"vnot">;
def PTO_VcaddOp : PTO_UnaryVecOp<"vcadd">;

def PTO_VcmaxOp : PTO_UnaryVecOp<"vcmax">;
def PTO_VcminOp : PTO_UnaryVecOp<"vcmin">;
def PTO_VcgaddOp : PTO_UnaryVecOp<"vcgadd">;
def PTO_VcgmaxOp : PTO_UnaryVecOp<"vcgmax">;
def PTO_VcgminOp : PTO_UnaryVecOp<"vcgmin">;
def PTO_VcpaddOp : PTO_UnaryVecOp<"vcpadd">;

class PTO_HistogramOp<string mnemonic> : PTO_VectorMicroOp<mnemonic, [Pure]> {
  let arguments = (ins
    PTO_VectorType:$acc,
    PTO_VectorType:$source,
    PTO_B8MaskTypeConstraint:$mask,
    I32:$bin
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $acc `,` $source `,` $mask `,` $bin attr-dict `:` type($acc) `,` type($source) `,` type($mask) `,` type($bin) `->` type($result)
  }];
}

def PTO_Chistv2Op : PTO_HistogramOp<"chistv2">;
def PTO_Dhistv2Op : PTO_HistogramOp<"dhistv2">;

class PTO_ExtremaPredicateOp<string mnemonic>
    : PTO_VectorMicroOp<mnemonic, [Pure]> {
  let arguments = (ins
    PTO_VectorType:$input,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs
    PTO_VectorType:$value,
    PTO_MaskTypeConstraint:$predicate
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $mask attr-dict `:` type($input) `,` type($mask) `->` type($value) `,` type($predicate)
  }];
}

def PTO_VcbmaxOp : PTO_ExtremaPredicateOp<"vcbmax">;
def PTO_VcbminOp : PTO_ExtremaPredicateOp<"vcbmin">;

class PTO_BinaryVecOp<string mnemonic> : PTO_VectorMicroOp<mnemonic, [Pure]> {
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
def PTO_VmulOp : PTO_BinaryVecOp<"vmul">;
def PTO_VdivOp : PTO_BinaryVecOp<"vdiv">;
def PTO_VmaxOp : PTO_BinaryVecOp<"vmax">;
def PTO_VminOp : PTO_BinaryVecOp<"vmin">;
def PTO_VandOp : PTO_BinaryVecOp<"vand">;
def PTO_VorOp : PTO_BinaryVecOp<"vor">;
def PTO_VxorOp : PTO_BinaryVecOp<"vxor">;

class PTO_TernaryVecOp<string mnemonic> : PTO_VectorMicroOp<mnemonic, [Pure]> {
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

def PTO_VmaddOp : PTO_TernaryVecOp<"vmadd">;

def PTO_VaddcOp : PTO_VectorMicroOp<"vaddc", [Pure]> {
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

def PTO_VsubcOp : PTO_VectorMicroOp<"vsubc", [Pure]> {
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

def PTO_VaddcsOp : PTO_VectorMicroOp<"vaddcs", [Pure]> {
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

def PTO_VsubcsOp : PTO_VectorMicroOp<"vsubcs", [Pure]> {
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

def PTO_VshlOp : PTO_BinaryVecOp<"vshl">;
def PTO_VshrOp : PTO_BinaryVecOp<"vshr">;

def PTO_VselOp : PTO_VectorMicroOp<"vsel", [Pure]> {
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

def PTO_VcmpOp : PTO_VectorMicroOp<"vcmp", [Pure]> {
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

def PTO_VcmpsOp : PTO_VectorMicroOp<"vcmps", [Pure]> {
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
    : PTO_VectorMicroOp<mnemonic, [Pure]> {
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

class PTO_VecScalarOp<string mnemonic> : PTO_VectorMicroOp<mnemonic, [Pure]> {
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

class PTO_VecScalarMaskedOp<string mnemonic>
    : PTO_VectorMicroOp<mnemonic, [Pure]> {
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

def PTO_VtrcOp : PTO_VectorMicroOp<"vtrc", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$input,
    PTO_MaskTypeConstraint:$mask,
    StrAttr:$round_mode
  );
  let results = (outs PTO_VectorType:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def PTO_VcvtOp : PTO_VectorMicroOp<"vcvt", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$input,
    PTO_MaskTypeConstraint:$mask,
    OptionalAttr<StrAttr>:$rnd,
    OptionalAttr<StrAttr>:$sat,
    OptionalAttr<StrAttr>:$part
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_VmulscvtOp : PTO_VectorMicroOp<"vmulscvt", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$input,
    AnyType:$scalar,
    PTO_MaskTypeConstraint:$mask,
    StrAttr:$rnd,
    StrAttr:$part
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $scalar `,` $mask `,` $rnd `,` $part attr-dict `:` type($input) `,` type($scalar) `,` type($mask) `->` type($result)
  }];
}

def PTO_VbitcastOp : PTO_VectorMicroOp<"vbitcast", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$input
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

def PTO_VciOp : PTO_VectorMicroOp<"vci", [Pure]> {
  let arguments = (ins
    AnyTypeOf<[AnyInteger, AnyFloat], "integer/float scalar">:$index,
    OptionalAttr<StrAttr>:$order
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $index attr-dict `:` type($index) `->` type($result)
  }];
}

def PTO_VbitsortOp : PTO_VectorMicroOp<"vbitsort", [
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

def PTO_Vmrgsort4Op : PTO_VectorMicroOp<"vmrgsort4"> {
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

def PTO_Vgather2Op : PTO_VectorMicroOp<"vgather2", [
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

def PTO_VgatherbOp : PTO_VectorMicroOp<"vgatherb", [
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
def PTO_Vgather2BcOp : PTO_VectorMicroOp<"vgather2_bc", [
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
def PTO_VmaxsOp : PTO_VecScalarMaskedOp<"vmaxs">;
def PTO_VminsOp : PTO_VecScalarMaskedOp<"vmins">;
def PTO_VlreluOp : PTO_VecScalarMaskedOp<"vlrelu">;
def PTO_VshlsOp : PTO_VecScalarMaskedOp<"vshls">;
def PTO_VshrsOp : PTO_VecScalarMaskedOp<"vshrs">;

def PTO_VstsOp : PTO_VectorMicroOp<"vsts", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_VectorType:$value,
    PTO_BufferLikeType:$destination,
    Index:$offset,
    OptionalAttr<StrAttr>:$dist,
    PTO_MaskTypeConstraint:$mask
  );

  let results = (outs Optional<PTO_BufferLikeType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `[` $offset `]` `,` $mask attr-dict `:` type($value) `,` type($destination) `,` type($mask)
    (`->` type($updated_base)^)?
  }];
}

def PTO_VscatterOp : PTO_VectorMicroOp<"vscatter", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_VectorType:$value,
    PTO_BufferType:$destination,
    PTO_VectorType:$offsets,
    PTO_MaskTypeConstraint:$mask
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `,` $offsets `,` $mask attr-dict `:` type($value) `,` type($destination) `,` type($offsets) `,` type($mask)
  }];
}

def PTO_PstsOp : PTO_VectorMicroOp<"psts", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_MaskTypeConstraint:$value,
    PTO_BufferLikeType:$destination,
    Index:$offset,
    StrAttr:$dist
  );

  let results = (outs Optional<PTO_BufferLikeType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `[` $offset `]` `,` $dist attr-dict `:` type($value) `,` type($destination) `,` type($offset)
    (`->` type($updated_base)^)?
  }];
}

def PTO_VselrOp : PTO_VectorMicroOp<"vselr", [Pure]> {
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

def PTO_VsqzOp : PTO_UnaryVecOp<"vsqz">;

def PTO_VusqzOp : PTO_VectorMicroOp<"vusqz", [Pure]> {
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

def PTO_VpackOp : PTO_VectorMicroOp<"vpack", [Pure]> {
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

def PTO_VsunpackOp : PTO_VectorMicroOp<"vsunpack", [Pure]> {
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

def PTO_VzunpackOp : PTO_VectorMicroOp<"vzunpack", [Pure]> {
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

def PTO_Vselrv2Op : PTO_VectorMicroOp<"vselrv2", [Pure]> {
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

def PTO_VintlvOp : PTO_VectorMicroOp<"vintlv", [Pure]> {
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

def PTO_VdintlvOp : PTO_VectorMicroOp<"vdintlv", [Pure]> {
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

def PTO_Vintlvv2Op : PTO_VectorMicroOp<"vintlvv2", [Pure]> {
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

def PTO_Vdintlvv2Op : PTO_VectorMicroOp<"vdintlvv2", [Pure]> {
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

def PTO_VmullOp : PTO_VectorMicroOp<"vmull", [Pure]> {
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

def PTO_VmulaOp : PTO_VectorMicroOp<"vmula", [Pure]> {
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

class PTO_UnmaskedBinaryVecOp<string mnemonic>
    : PTO_VectorMicroOp<mnemonic, [Pure]> {
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

class PTO_BinaryVecMaskedOp<string mnemonic>
    : PTO_VectorMicroOp<mnemonic, [Pure]> {
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

def PTO_VpreluOp : PTO_BinaryVecMaskedOp<"vprelu">;
def PTO_VexpdifOp : PTO_VectorMicroOp<"vexpdif", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$input,
    PTO_VectorType:$max,
    PTO_MaskTypeConstraint:$mask,
    StrAttr:$part
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $max `,` $mask `,` $part attr-dict `:` type($input) `,` type($max) `,` type($mask) `->` type($result)
  }];
}

def PTO_VaxpyOp : PTO_VectorMicroOp<"vaxpy", [Pure]> {
  let arguments = (ins
    PTO_VectorType:$src0,
    PTO_VectorType:$src1,
    AnyType:$alpha,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 `,` $alpha `,` $mask attr-dict `:` type($src0) `,` type($src1) `,` type($alpha) `,` type($mask) `->` type($result)
  }];
}

def PTO_VaddreluconvOp : PTO_VectorMicroOp<"vaddreluconv", [Pure]> {
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

def PTO_VmulconvOp : PTO_VectorMicroOp<"vmulconv", [Pure]> {
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

def PTO_Vstsx2Op : PTO_VectorMicroOp<"vstsx2", [
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

def PTO_VsldbOp : PTO_VectorMicroOp<"vsldb", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    I16:$block_stride,
    I16:$repeat_stride,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs PTO_VectorType:$result,
                      Optional<PTO_BufferLikeType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $block_stride `,` $repeat_stride `,` $mask attr-dict `:` type($source) `,` type($block_stride) `,` type($repeat_stride) `,` type($mask) `->` type($result)
    (`,` type($updated_base)^)?
  }];
}

def PTO_VsstbOp : PTO_VectorMicroOp<"vsstb", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_VectorType:$value,
    PTO_BufferLikeType:$destination,
    I16:$block_stride,
    I16:$repeat_stride,
    PTO_MaskTypeConstraint:$mask
  );
  let results = (outs Optional<PTO_BufferLikeType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `,` $block_stride `,` $repeat_stride `,` $mask attr-dict `:` type($value) `,` type($destination) `,` type($block_stride) `,` type($repeat_stride) `,` type($mask)
    (`->` type($updated_base)^)?
  }];
}

def PTO_VstasOp : PTO_VectorMicroOp<"vstas", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_AlignTypeConstraint:$value,
    PTO_BufferLikeType:$destination,
    I32:$offset
  );
  let results = (outs Optional<PTO_BufferLikeType>:$updated_base);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `,` $offset attr-dict `:` type($value) `,` type($destination) `,` type($offset)
    (`->` type($updated_base)^)?
  }];
}

def PTO_VstarOp : PTO_VectorMicroOp<"vstar", [
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
def PTO_PstuOp : PTO_VectorMicroOp<"pstu", [
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

def PTO_VstusOp : PTO_VectorMicroOp<"vstus", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_AlignTypeConstraint:$align_in,
    I32:$offset,
    PTO_VectorType:$value,
    PTO_BufferType:$base
  );
  let results = (outs PTO_AlignTypeConstraint:$align_out,
                      Optional<PTO_BufferType>:$base_out);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $align_in `,` $offset `,` $value `,` $base attr-dict `:` type($align_in) `,` type($offset) `,` type($value) `,` type($base) `->` type($align_out)
    (`,` type($base_out)^)?
  }];
}

def PTO_VsturOp : PTO_VectorMicroOp<"vstur", [
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

#endif // MLIR_DIALECT_PTO_IR_VPTOVECTOROPS
