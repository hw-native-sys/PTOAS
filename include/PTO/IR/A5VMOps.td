//===- A5VMOps.td - A5VM operations -------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_A5VMOPS
#define MLIR_DIALECT_PTO_IR_A5VMOPS

include "PTO/IR/A5VMDialect.td"
include "PTO/IR/A5VMTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def A5VM_VectorType : Type<CPred<"::llvm::isa<::mlir::a5vm::VecType>($_self)">,
                           "A5VM vector type">;
def A5VM_MaskTypeConstraint : Type<CPred<"::llvm::isa<::mlir::a5vm::MaskType>($_self)">,
                                   "A5VM mask type">;
def A5VM_AlignTypeConstraint : Type<CPred<"::llvm::isa<::mlir::a5vm::AlignType>($_self)">,
                                    "A5VM align type">;

def A5VM_BufferType : Type<
  CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">,
  "LLVM pointer type">;

class A5VM_Op<string mnemonic, list<Trait> traits = []>
    : Op<A5VM_Dialect, mnemonic, traits>;

def A5VM_SetFlagOp : A5VM_Op<"set_flag"> {
  let arguments = (ins
    StrAttr:$src_pipe,
    StrAttr:$dst_pipe,
    StrAttr:$event_id
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `[` $src_pipe `,` $dst_pipe `,` $event_id `]` attr-dict
  }];
}

def A5VM_WaitFlagOp : A5VM_Op<"wait_flag"> {
  let arguments = (ins
    StrAttr:$src_pipe,
    StrAttr:$dst_pipe,
    StrAttr:$event_id
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `[` $src_pipe `,` $dst_pipe `,` $event_id `]` attr-dict
  }];
}

def A5VM_PipeBarrierOp : A5VM_Op<"pipe_barrier"> {
  let arguments = (ins StrAttr:$pipe);

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = "$pipe attr-dict";
}

class A5VM_BufTokenOp<string mnemonic> : A5VM_Op<mnemonic> {
  let arguments = (ins
    StrAttr:$pipe,
    I64:$buf_id,
    I64:$mode
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $pipe `,` $buf_id `,` $mode attr-dict `:` type($buf_id) `,` type($mode)
  }];
}

def A5VM_GetBufOp : A5VM_BufTokenOp<"get_buf">;
def A5VM_RlsBufOp : A5VM_BufTokenOp<"rls_buf">;

class A5VM_BinaryI64ConfigOp<string mnemonic> : A5VM_Op<mnemonic> {
  let arguments = (ins
    I64:$first,
    I64:$second
  );

  let results = (outs);

  let assemblyFormat = [{
    $first `,` $second attr-dict `:` type($first) `,` type($second)
  }];
}

def A5VM_SetLoop2StrideOutToUbOp : A5VM_BinaryI64ConfigOp<"set_loop2_stride_outtoub">;
def A5VM_SetLoop1StrideOutToUbOp : A5VM_BinaryI64ConfigOp<"set_loop1_stride_outtoub">;
def A5VM_SetLoopSizeOutToUbOp : A5VM_BinaryI64ConfigOp<"set_loop_size_outtoub">;
def A5VM_SetLoop2StrideUbToOutOp : A5VM_BinaryI64ConfigOp<"set_loop2_stride_ubtoout">;
def A5VM_SetLoop1StrideUbToOutOp : A5VM_BinaryI64ConfigOp<"set_loop1_stride_ubtoout">;
def A5VM_SetLoopSizeUbToOutOp : A5VM_BinaryI64ConfigOp<"set_loop_size_ubtoout">;

def A5VM_CopyGmToUbufOp : A5VM_Op<"copy_gm_to_ubuf", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_BufferType:$source,
    A5VM_BufferType:$destination,
    I64:$valid_rows,
    I64:$valid_cols,
    I64:$sid,
    I64:$n_burst,
    I64:$len_burst,
    I64:$left_padding_count,
    I64:$right_padding_count,
    I64:$l2_cache_ctl,
    I64:$gm_stride,
    I64:$ub_stride,
    OptionalAttr<StrAttr>:$layout,
    OptionalAttr<BoolAttr>:$data_select_bit,
    OptionalAttr<BoolAttr>:$ub_pad
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $valid_rows `,` $valid_cols `,` $sid `,` $n_burst `,` $len_burst `,`
    $left_padding_count `,` $right_padding_count `,` $l2_cache_ctl `,` $gm_stride `,` $ub_stride
    attr-dict `:` type($source) `,` type($destination) `,` type($valid_rows) `,` type($valid_cols) `,`
    type($sid) `,` type($n_burst) `,` type($len_burst) `,` type($left_padding_count) `,`
    type($right_padding_count) `,` type($l2_cache_ctl) `,` type($gm_stride) `,` type($ub_stride)
  }];
}

def A5VM_CopyUbufToUbufOp : A5VM_Op<"copy_ubuf_to_ubuf"> {
  let arguments = (ins
    A5VM_BufferType:$source,
    A5VM_BufferType:$destination,
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

def A5VM_VldsOp : A5VM_Op<"vlds", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_BufferType:$source,
    Index:$offset,
    OptionalAttr<StrAttr>:$dist
  );

  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $offset `]` attr-dict `:` type($source) `->` type($result)
  }];
}

def A5VM_VldsPostOp : A5VM_Op<"vlds_post", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_BufferType:$source,
    Index:$offset,
    OptionalAttr<StrAttr>:$dist
  );

  let results = (outs A5VM_VectorType:$result, A5VM_BufferType:$updated_source);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $offset `]` attr-dict `:` type($source) `->` type($result) `,` type($updated_source)
  }];
}

def A5VM_VldasOp : A5VM_Op<"vldas", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_BufferType:$source
  );

  let results = (outs A5VM_AlignTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` type($result)
  }];
}

def A5VM_VldusOp : A5VM_Op<"vldus", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_BufferType:$source,
    A5VM_AlignTypeConstraint:$align
  );

  let results = (outs
    A5VM_VectorType:$result,
    A5VM_AlignTypeConstraint:$updated_align,
    A5VM_BufferType:$updated_source
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $align attr-dict `:` type($source) `,` type($align) `->` type($result) `,` type($updated_align) `,` type($updated_source)
  }];
}

def A5VM_VbrOp : A5VM_Op<"vbr", [Pure]> {
  let arguments = (ins AnyType:$value);
  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result)
  }];
}

def A5VM_VdupOp : A5VM_Op<"vdup", [Pure]> {
  let arguments = (ins
    AnyType:$input,
    OptionalAttr<StrAttr>:$position,
    OptionalAttr<StrAttr>:$mode
  );
  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

def A5VM_PsetB8Op : A5VM_Op<"pset_b8", [Pure]> {
  let arguments = (ins StrAttr:$pattern);
  let results = (outs A5VM_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $pattern attr-dict `:` type($result)
  }];
}

def A5VM_PsetB16Op : A5VM_Op<"pset_b16", [Pure]> {
  let arguments = (ins StrAttr:$pattern);
  let results = (outs A5VM_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $pattern attr-dict `:` type($result)
  }];
}

// NOTE: The op families introduced below are intentionally marked as
// unvalidated scaffolding. They are added to preserve missing CCE builtin
// semantics at the dialect layer, but they have not yet been validated through
// PTO lowering or end-to-end sample execution.
def A5VM_PsetB32Op : A5VM_Op<"pset_b32", [Pure]> {
  let arguments = (ins StrAttr:$pattern);
  let results = (outs A5VM_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $pattern attr-dict `:` type($result)
  }];
}

class A5VM_PgeOp<string mnemonic> : A5VM_Op<mnemonic, [Pure]> {
  let arguments = (ins StrAttr:$pattern);
  let results = (outs A5VM_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $pattern attr-dict `:` type($result)
  }];
}

def A5VM_PgeB8Op : A5VM_PgeOp<"pge_b8">;
def A5VM_PgeB16Op : A5VM_PgeOp<"pge_b16">;
def A5VM_PgeB32Op : A5VM_PgeOp<"pge_b32">;

class A5VM_PltOp<string mnemonic> : A5VM_Op<mnemonic, [Pure]> {
  let arguments = (ins I32:$scalar);
  let results = (outs A5VM_MaskTypeConstraint:$mask, I32:$scalar_out);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $scalar attr-dict `:` type($scalar) `->` type($mask) `,` type($scalar_out)
  }];
}

def A5VM_PltB8Op : A5VM_PltOp<"plt_b8">;
def A5VM_PltB16Op : A5VM_PltOp<"plt_b16">;
def A5VM_PltB32Op : A5VM_PltOp<"plt_b32">;

class A5VM_MaskUnaryOp<string mnemonic> : A5VM_Op<mnemonic, [Pure]> {
  let arguments = (ins A5VM_MaskTypeConstraint:$input);
  let results = (outs A5VM_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

def A5VM_PpackOp : A5VM_MaskUnaryOp<"ppack"> {
  let arguments = (ins A5VM_MaskTypeConstraint:$input, StrAttr:$part);
  let assemblyFormat = [{
    $input `,` $part attr-dict `:` type($input) `->` type($result)
  }];
}

def A5VM_PunpackOp : A5VM_MaskUnaryOp<"punpack"> {
  let arguments = (ins A5VM_MaskTypeConstraint:$input, StrAttr:$part);
  let assemblyFormat = [{
    $input `,` $part attr-dict `:` type($input) `->` type($result)
  }];
}

def A5VM_PnotOp : A5VM_Op<"pnot", [Pure]> {
  let arguments = (ins
    A5VM_MaskTypeConstraint:$input,
    A5VM_MaskTypeConstraint:$mask
  );
  let results = (outs A5VM_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $mask attr-dict `:` type($input) `,` type($mask) `->` type($result)
  }];
}

def A5VM_PselOp : A5VM_Op<"psel", [Pure]> {
  let arguments = (ins
    A5VM_MaskTypeConstraint:$src0,
    A5VM_MaskTypeConstraint:$src1,
    A5VM_MaskTypeConstraint:$mask
  );
  let results = (outs A5VM_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 `,` $mask attr-dict `:` type($src0) `,` type($src1) `,` type($mask) `->` type($result)
  }];
}

def A5VM_PldsOp : A5VM_Op<"plds", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_BufferType:$source,
    Index:$offset,
    OptionalAttr<StrAttr>:$dist
  );
  let results = (outs A5VM_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $offset `]` attr-dict `:` type($source) `->` type($result)
  }];
}

// NOTE: Unvalidated predicate load/store-family abstractions. Added to cover
// direct CCE predicate memory builtins not yet end-to-end exercised.
def A5VM_PldOp : A5VM_Op<"pld", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_BufferType:$source,
    Index:$offset,
    StrAttr:$dist
  );
  let results = (outs A5VM_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $offset `]` `,` $dist attr-dict `:` type($source) `,` type($offset) `->` type($result)
  }];
}

def A5VM_PldiOp : A5VM_Op<"pldi", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_BufferType:$source,
    I32:$offset,
    StrAttr:$dist
  );
  let results = (outs A5VM_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $offset `,` $dist attr-dict `:` type($source) `,` type($offset) `->` type($result)
  }];
}

def A5VM_PstOp : A5VM_Op<"pst", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_MaskTypeConstraint:$value,
    A5VM_BufferType:$destination,
    Index:$offset,
    StrAttr:$dist
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `[` $offset `]` `,` $dist attr-dict `:` type($value) `,` type($destination) `,` type($offset)
  }];
}

def A5VM_PstiOp : A5VM_Op<"psti", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_MaskTypeConstraint:$value,
    A5VM_BufferType:$destination,
    I32:$offset,
    StrAttr:$dist
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `,` $offset `,` $dist attr-dict `:` type($value) `,` type($destination) `,` type($offset)
  }];
}

def A5VM_VabsOp : A5VM_Op<"vabs", [Pure]> {
  let arguments = (ins
    A5VM_VectorType:$input,
    A5VM_MaskTypeConstraint:$mask,
    OptionalAttr<StrAttr>:$mode
  );
  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $mask attr-dict `:` type($input) `,` type($mask) `->` type($result)
  }];
}

class A5VM_UnaryVecOp<string mnemonic> : A5VM_Op<mnemonic, [Pure]> {
  let arguments = (ins
    A5VM_VectorType:$input,
    A5VM_MaskTypeConstraint:$mask,
    OptionalAttr<StrAttr>:$mode
  );
  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $mask attr-dict `:` type($input) `,` type($mask) `->` type($result)
  }];
}

def A5VM_VexpOp : A5VM_UnaryVecOp<"vexp">;
def A5VM_VlnOp : A5VM_UnaryVecOp<"vln">;
def A5VM_VsqrtOp : A5VM_UnaryVecOp<"vsqrt">;
def A5VM_VrecOp : A5VM_UnaryVecOp<"vrec">;
def A5VM_VreluOp : A5VM_UnaryVecOp<"vrelu">;
def A5VM_VnotOp : A5VM_UnaryVecOp<"vnot">;
def A5VM_VcaddOp : A5VM_UnaryVecOp<"vcadd">;
def A5VM_VcmaxOp : A5VM_UnaryVecOp<"vcmax">;
def A5VM_VcminOp : A5VM_UnaryVecOp<"vcmin">;

class A5VM_BinaryVecOp<string mnemonic> : A5VM_Op<mnemonic, [Pure]> {
  let arguments = (ins
    A5VM_VectorType:$lhs,
    A5VM_VectorType:$rhs
  );
  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def A5VM_VaddOp : A5VM_BinaryVecOp<"vadd">;
def A5VM_VsubOp : A5VM_BinaryVecOp<"vsub">;
def A5VM_VmulOp : A5VM_BinaryVecOp<"vmul">;
def A5VM_VdivOp : A5VM_BinaryVecOp<"vdiv">;
def A5VM_VmaxOp : A5VM_BinaryVecOp<"vmax">;
def A5VM_VminOp : A5VM_BinaryVecOp<"vmin">;
def A5VM_VandOp : A5VM_BinaryVecOp<"vand">;
def A5VM_VorOp : A5VM_BinaryVecOp<"vor">;
def A5VM_VxorOp : A5VM_BinaryVecOp<"vxor">;

def A5VM_VaddcOp : A5VM_Op<"vaddc", [Pure]> {
  let arguments = (ins
    A5VM_VectorType:$lhs,
    A5VM_VectorType:$rhs,
    A5VM_MaskTypeConstraint:$mask
  );
  let results = (outs
    A5VM_VectorType:$result,
    A5VM_MaskTypeConstraint:$carry
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $mask attr-dict `:` type($lhs) `,` type($rhs) `,` type($mask) `->` type($result) `,` type($carry)
  }];
}

def A5VM_VsubcOp : A5VM_Op<"vsubc", [Pure]> {
  let arguments = (ins
    A5VM_VectorType:$lhs,
    A5VM_VectorType:$rhs,
    A5VM_MaskTypeConstraint:$mask
  );
  let results = (outs
    A5VM_VectorType:$result,
    A5VM_MaskTypeConstraint:$carry
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $mask attr-dict `:` type($lhs) `,` type($rhs) `,` type($mask) `->` type($result) `,` type($carry)
  }];
}

def A5VM_VaddcsOp : A5VM_Op<"vaddcs", [Pure]> {
  let arguments = (ins
    A5VM_VectorType:$lhs,
    A5VM_VectorType:$rhs,
    A5VM_MaskTypeConstraint:$carry_in,
    A5VM_MaskTypeConstraint:$mask
  );
  let results = (outs
    A5VM_VectorType:$result,
    A5VM_MaskTypeConstraint:$carry
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $carry_in `,` $mask attr-dict `:` type($lhs) `,` type($rhs) `,` type($carry_in) `,` type($mask) `->` type($result) `,` type($carry)
  }];
}

def A5VM_VsubcsOp : A5VM_Op<"vsubcs", [Pure]> {
  let arguments = (ins
    A5VM_VectorType:$lhs,
    A5VM_VectorType:$rhs,
    A5VM_MaskTypeConstraint:$carry_in,
    A5VM_MaskTypeConstraint:$mask
  );
  let results = (outs
    A5VM_VectorType:$result,
    A5VM_MaskTypeConstraint:$carry
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $carry_in `,` $mask attr-dict `:` type($lhs) `,` type($rhs) `,` type($carry_in) `,` type($mask) `->` type($result) `,` type($carry)
  }];
}

def A5VM_VbcntOp : A5VM_UnaryVecOp<"vbcnt">;
def A5VM_VclsOp : A5VM_UnaryVecOp<"vcls">;

def A5VM_VshlOp : A5VM_BinaryVecOp<"vshl">;
def A5VM_VshrOp : A5VM_BinaryVecOp<"vshr">;

def A5VM_VselOp : A5VM_Op<"vsel", [Pure]> {
  let arguments = (ins
    A5VM_VectorType:$src0,
    A5VM_VectorType:$src1,
    A5VM_MaskTypeConstraint:$mask
  );
  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 `,` $mask attr-dict `:` type($src0) `,` type($src1) `,` type($mask) `->` type($result)
  }];
}

def A5VM_VcmpOp : A5VM_Op<"vcmp", [Pure]> {
  let arguments = (ins
    A5VM_VectorType:$src0,
    A5VM_VectorType:$src1,
    A5VM_MaskTypeConstraint:$mask,
    StrAttr:$cmp_mode
  );
  let results = (outs A5VM_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 `,` $mask `,` $cmp_mode attr-dict `:` type($src0) `,` type($src1) `,` type($mask) `->` type($result)
  }];
}

def A5VM_VcmpsOp : A5VM_Op<"vcmps", [Pure]> {
  let arguments = (ins
    A5VM_VectorType:$src,
    AnyType:$scalar,
    A5VM_MaskTypeConstraint:$mask,
    StrAttr:$cmp_mode
  );
  let results = (outs A5VM_MaskTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src `,` $scalar `,` $mask `,` $cmp_mode attr-dict `:` type($src) `,` type($scalar) `,` type($mask) `->` type($result)
  }];
}

def A5VM_PdintlvB8Op : A5VM_Op<"pdintlv_b8", [Pure]> {
  let arguments = (ins
    A5VM_MaskTypeConstraint:$lhs,
    A5VM_MaskTypeConstraint:$rhs
  );
  let results = (outs
    A5VM_MaskTypeConstraint:$low,
    A5VM_MaskTypeConstraint:$high
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($low) `,` type($high)
  }];
}

def A5VM_PintlvB16Op : A5VM_Op<"pintlv_b16", [Pure]> {
  let arguments = (ins
    A5VM_MaskTypeConstraint:$lhs,
    A5VM_MaskTypeConstraint:$rhs
  );
  let results = (outs
    A5VM_MaskTypeConstraint:$low,
    A5VM_MaskTypeConstraint:$high
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($low) `,` type($high)
  }];
}

class A5VM_VecScalarOp<string mnemonic> : A5VM_Op<mnemonic, [Pure]> {
  let arguments = (ins
    A5VM_VectorType:$input,
    AnyType:$scalar
  );
  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $scalar attr-dict `:` type($input) `,` type($scalar) `->` type($result)
  }];
}

def A5VM_VtrcOp : A5VM_Op<"vtrc", [Pure]> {
  let arguments = (ins
    A5VM_VectorType:$input,
    StrAttr:$round_mode
  );
  let results = (outs A5VM_VectorType:$result);

  let assemblyFormat = [{
    $input `,` $round_mode attr-dict `:` type($input) `->` type($result)
  }];
}

def A5VM_VcvtOp : A5VM_Op<"vcvt", [Pure]> {
  let arguments = (ins
    A5VM_VectorType:$input,
    OptionalAttr<StrAttr>:$round_mode,
    OptionalAttr<StrAttr>:$sat,
    OptionalAttr<StrAttr>:$part
  );
  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

def A5VM_VciOp : A5VM_Op<"vci", [Pure]> {
  let arguments = (ins
    AnyInteger:$index,
    OptionalAttr<StrAttr>:$order
  );
  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $index attr-dict `:` type($index) `->` type($result)
  }];
}

def A5VM_VbitsortOp : A5VM_Op<"vbitsort"> {
  let arguments = (ins
    A5VM_BufferType:$destination,
    A5VM_BufferType:$source,
    A5VM_BufferType:$indices,
    Index:$repeat_times
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $destination `,` $source `,` $indices `,` $repeat_times attr-dict `:` type($destination) `,`
    type($source) `,` type($indices) `,` type($repeat_times)
  }];
}

def A5VM_Vmrgsort4Op : A5VM_Op<"vmrgsort4"> {
  let arguments = (ins
    A5VM_BufferType:$destination,
    A5VM_BufferType:$source0,
    A5VM_BufferType:$source1,
    A5VM_BufferType:$source2,
    A5VM_BufferType:$source3,
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

def A5VM_Vgather2Op : A5VM_Op<"vgather2", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_BufferType:$source,
    A5VM_VectorType:$offsets,
    Index:$active_lanes
  );
  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $offsets `,` $active_lanes attr-dict `:` type($source) `,` type($offsets) `,` type($active_lanes) `->` type($result)
  }];
}

def A5VM_VgatherbOp : A5VM_Op<"vgatherb", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_BufferType:$source,
    A5VM_VectorType:$offsets,
    Index:$active_lanes
  );
  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $offsets `,` $active_lanes attr-dict `:` type($source) `,` type($offsets) `,` type($active_lanes) `->` type($result)
  }];
}

// NOTE: Unvalidated new gather/select/interleave-family abstractions. Added to
// cover CCE builtin families not yet exercised through end-to-end PTO seams.
def A5VM_Vgather2BcOp : A5VM_Op<"vgather2_bc", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_BufferType:$source,
    A5VM_VectorType:$offsets,
    A5VM_MaskTypeConstraint:$mask
  );
  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $offsets `,` $mask attr-dict `:` type($source) `,` type($offsets) `,` type($mask) `->` type($result)
  }];
}

def A5VM_VmulsOp : A5VM_VecScalarOp<"vmuls">;
def A5VM_VaddsOp : A5VM_VecScalarOp<"vadds">;
def A5VM_VmaxsOp : A5VM_VecScalarOp<"vmaxs">;
def A5VM_VminsOp : A5VM_VecScalarOp<"vmins">;
def A5VM_VlreluOp : A5VM_VecScalarOp<"vlrelu">;
def A5VM_VshlsOp : A5VM_VecScalarOp<"vshls">;
def A5VM_VshrsOp : A5VM_VecScalarOp<"vshrs">;

def A5VM_VsldOp : A5VM_Op<"vsld", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_BufferType:$source,
    Index:$offset,
    StrAttr:$stride
  );

  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $offset `]` `,` $stride attr-dict `:` type($source) `->` type($result)
  }];
}

def A5VM_VstsOp : A5VM_Op<"vsts", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_VectorType:$value,
    A5VM_BufferType:$destination,
    Index:$offset,
    OptionalAttr<StrAttr>:$dist,
    A5VM_MaskTypeConstraint:$mask
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `[` $offset `]` `,` $mask attr-dict `:` type($value) `,` type($destination) `,` type($mask)
  }];
}

def A5VM_VstsPostOp : A5VM_Op<"vsts_post", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_VectorType:$value,
    A5VM_BufferType:$destination,
    Index:$offset,
    OptionalAttr<StrAttr>:$dist,
    A5VM_MaskTypeConstraint:$mask
  );

  let results = (outs A5VM_BufferType:$updated_destination);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `[` $offset `]` `,` $mask attr-dict `:` type($value) `,` type($destination) `,` type($mask) `->` type($updated_destination)
  }];
}

def A5VM_VscatterOp : A5VM_Op<"vscatter", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_VectorType:$value,
    A5VM_BufferType:$destination,
    A5VM_VectorType:$offsets,
    Index:$active_lanes
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `,` $offsets `,` $active_lanes attr-dict `:` type($value) `,` type($destination) `,` type($offsets) `,` type($active_lanes)
  }];
}

def A5VM_PstsOp : A5VM_Op<"psts", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_MaskTypeConstraint:$value,
    A5VM_BufferType:$destination,
    Index:$offset
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `[` $offset `]` attr-dict `:` type($value) `,` type($destination)
  }];
}

// NOTE: Unvalidated new store-family abstraction. Added for missing CCE
// `vsst` semantics, but not yet exercised through end-to-end PTO lowering.
def A5VM_VsstOp : A5VM_Op<"vsst", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_VectorType:$value,
    A5VM_BufferType:$destination,
    Index:$offset,
    StrAttr:$stride
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `[` $offset `]` `,` $stride attr-dict `:` type($value) `,` type($destination)
  }];
}

def A5VM_CopyUbufToGmOp : A5VM_Op<"copy_ubuf_to_gm", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_BufferType:$source,
    A5VM_BufferType:$destination,
    I64:$valid_rows,
    I64:$valid_cols,
    I64:$sid,
    I64:$n_burst,
    I64:$len_burst,
    I64:$reserved,
    I64:$burst_dst_stride,
    I64:$burst_src_stride,
    OptionalAttr<StrAttr>:$layout
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $valid_rows `,` $valid_cols `,` $sid `,` $n_burst `,` $len_burst `,`
    $reserved `,` $burst_dst_stride `,` $burst_src_stride
    attr-dict `:` type($source) `,` type($destination) `,` type($valid_rows) `,` type($valid_cols) `,`
    type($sid) `,` type($n_burst) `,` type($len_burst) `,` type($reserved) `,`
    type($burst_dst_stride) `,` type($burst_src_stride)
  }];
}

// NOTE: Unvalidated new x2 / pair / align-store-family abstractions. Added to
// reflect CCE builtin families but not yet end-to-end validated.
def A5VM_VselrOp : A5VM_Op<"vselr", [Pure]> {
  let arguments = (ins
    A5VM_VectorType:$src0,
    A5VM_VectorType:$src1
  );
  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 attr-dict `:` type($src0) `,` type($src1) `->` type($result)
  }];
}

def A5VM_Vselrv2Op : A5VM_Op<"vselrv2", [Pure]> {
  let arguments = (ins
    A5VM_VectorType:$src0,
    A5VM_VectorType:$src1
  );
  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src0 `,` $src1 attr-dict `:` type($src0) `,` type($src1) `->` type($result)
  }];
}

def A5VM_VintlvOp : A5VM_Op<"vintlv", [Pure]> {
  let arguments = (ins
    A5VM_VectorType:$lhs,
    A5VM_VectorType:$rhs
  );
  let results = (outs A5VM_VectorType:$low, A5VM_VectorType:$high);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($low) `,` type($high)
  }];
}

def A5VM_VdintlvOp : A5VM_Op<"vdintlv", [Pure]> {
  let arguments = (ins
    A5VM_VectorType:$lhs,
    A5VM_VectorType:$rhs
  );
  let results = (outs A5VM_VectorType:$low, A5VM_VectorType:$high);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($low) `,` type($high)
  }];
}

def A5VM_Vintlvv2Op : A5VM_Op<"vintlvv2", [Pure]> {
  let arguments = (ins
    A5VM_VectorType:$lhs,
    A5VM_VectorType:$rhs,
    StrAttr:$part
  );
  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $part attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def A5VM_Vdintlvv2Op : A5VM_Op<"vdintlvv2", [Pure]> {
  let arguments = (ins
    A5VM_VectorType:$lhs,
    A5VM_VectorType:$rhs,
    StrAttr:$part
  );
  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $part attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def A5VM_VmullOp : A5VM_Op<"vmull", [Pure]> {
  let arguments = (ins
    A5VM_VectorType:$lhs,
    A5VM_VectorType:$rhs,
    A5VM_MaskTypeConstraint:$mask
  );
  let results = (outs A5VM_VectorType:$low, A5VM_VectorType:$high);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $mask attr-dict `:` type($lhs) `,` type($rhs) `,` type($mask) `->` type($low) `,` type($high)
  }];
}

def A5VM_VmulaOp : A5VM_Op<"vmula", [Pure]> {
  let arguments = (ins
    A5VM_VectorType:$acc,
    A5VM_VectorType:$lhs,
    A5VM_VectorType:$rhs,
    A5VM_MaskTypeConstraint:$mask,
    OptionalAttr<StrAttr>:$mode
  );
  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $acc `,` $lhs `,` $rhs `,` $mask attr-dict `:` type($acc) `,` type($lhs) `,` type($rhs) `,` type($mask) `->` type($result)
  }];
}

def A5VM_Vldx2Op : A5VM_Op<"vldx2", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_BufferType:$source,
    Index:$offset,
    StrAttr:$dist
  );
  let results = (outs A5VM_VectorType:$low, A5VM_VectorType:$high);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $offset `]` `,` $dist attr-dict `:` type($source) `,` type($offset) `->` type($low) `,` type($high)
  }];
}

def A5VM_Vstx2Op : A5VM_Op<"vstx2", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_VectorType:$low,
    A5VM_VectorType:$high,
    A5VM_BufferType:$destination,
    Index:$offset,
    StrAttr:$dist,
    A5VM_MaskTypeConstraint:$mask
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $low `,` $high `,` $destination `[` $offset `]` `,` $dist `,` $mask attr-dict `:` type($low) `,` type($high) `,` type($destination) `,` type($offset) `,` type($mask)
  }];
}

def A5VM_VsldbOp : A5VM_Op<"vsldb", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_BufferType:$source,
    I32:$offset,
    A5VM_MaskTypeConstraint:$mask
  );
  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $offset `,` $mask attr-dict `:` type($source) `,` type($offset) `,` type($mask) `->` type($result)
  }];
}

def A5VM_VsstbOp : A5VM_Op<"vsstb", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_VectorType:$value,
    A5VM_BufferType:$destination,
    I32:$offset,
    A5VM_MaskTypeConstraint:$mask
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `,` $offset `,` $mask attr-dict `:` type($value) `,` type($destination) `,` type($offset) `,` type($mask)
  }];
}

def A5VM_VstaOp : A5VM_Op<"vsta", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_AlignTypeConstraint:$value,
    A5VM_BufferType:$destination,
    Index:$offset
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `[` $offset `]` attr-dict `:` type($value) `,` type($destination) `,` type($offset)
  }];
}

def A5VM_VstasOp : A5VM_Op<"vstas", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_AlignTypeConstraint:$value,
    A5VM_BufferType:$destination,
    I32:$offset
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `,` $offset attr-dict `:` type($value) `,` type($destination) `,` type($offset)
  }];
}

def A5VM_VstarOp : A5VM_Op<"vstar", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_AlignTypeConstraint:$value,
    A5VM_BufferType:$destination
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
def A5VM_PstuOp : A5VM_Op<"pstu", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_AlignTypeConstraint:$align_in,
    A5VM_MaskTypeConstraint:$value,
    A5VM_BufferType:$base
  );
  let results = (outs A5VM_AlignTypeConstraint:$align_out, A5VM_BufferType:$base_out);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $align_in `,` $value `,` $base attr-dict `:` type($align_in) `,` type($value) `,` type($base) `->` type($align_out) `,` type($base_out)
  }];
}

def A5VM_VstuOp : A5VM_Op<"vstu", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_AlignTypeConstraint:$align_in,
    Index:$offset_in,
    A5VM_VectorType:$value,
    A5VM_BufferType:$base,
    StrAttr:$mode
  );
  let results = (outs A5VM_AlignTypeConstraint:$align_out, Index:$offset_out);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $align_in `,` $offset_in `,` $value `,` $base `,` $mode attr-dict `:` type($align_in) `,` type($offset_in) `,` type($value) `,` type($base) `->` type($align_out) `,` type($offset_out)
  }];
}

def A5VM_VstusOp : A5VM_Op<"vstus", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_AlignTypeConstraint:$align_in,
    I32:$offset,
    A5VM_VectorType:$value,
    A5VM_BufferType:$base,
    StrAttr:$mode
  );
  let results = (outs A5VM_AlignTypeConstraint:$align_out, A5VM_BufferType:$base_out);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $align_in `,` $offset `,` $value `,` $base `,` $mode attr-dict `:` type($align_in) `,` type($offset) `,` type($value) `,` type($base) `->` type($align_out) `,` type($base_out)
  }];
}

def A5VM_VsturOp : A5VM_Op<"vstur", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_AlignTypeConstraint:$align_in,
    A5VM_VectorType:$value,
    A5VM_BufferType:$base,
    StrAttr:$mode
  );
  let results = (outs A5VM_AlignTypeConstraint:$align_out);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $align_in `,` $value `,` $base `,` $mode attr-dict `:` type($align_in) `,` type($value) `,` type($base) `->` type($align_out)
  }];
}

#endif // MLIR_DIALECT_PTO_IR_A5VMOPS
