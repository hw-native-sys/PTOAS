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

def A5VM_VldasOp : A5VM_Op<"vldas", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_BufferType:$source,
    Index:$offset
  );

  let results = (outs A5VM_AlignTypeConstraint:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $offset `]` attr-dict `:` type($source) `->` type($result)
  }];
}

def A5VM_VldusOp : A5VM_Op<"vldus", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_AlignTypeConstraint:$align,
    A5VM_BufferType:$source,
    Index:$offset
  );

  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $align `,` $source `[` $offset `]` attr-dict `:` type($align) `,` type($source) `->` type($result)
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

def A5VM_VabsOp : A5VM_Op<"vabs", [Pure]> {
  let arguments = (ins A5VM_VectorType:$input);
  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

class A5VM_UnaryVecOp<string mnemonic> : A5VM_Op<mnemonic, [Pure]> {
  let arguments = (ins A5VM_VectorType:$input);
  let results = (outs A5VM_VectorType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
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

def A5VM_VmulsOp : A5VM_VecScalarOp<"vmuls">;
def A5VM_VaddsOp : A5VM_VecScalarOp<"vadds">;
def A5VM_VmaxsOp : A5VM_VecScalarOp<"vmaxs">;
def A5VM_VminsOp : A5VM_VecScalarOp<"vmins">;
def A5VM_VlreluOp : A5VM_VecScalarOp<"vlrelu">;

def A5VM_VstsOp : A5VM_Op<"vsts", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_VectorType:$value,
    A5VM_BufferType:$destination,
    Index:$offset,
    OptionalAttr<StrAttr>:$dist
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `[` $offset `]` attr-dict `:` type($value) `,` type($destination)
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

def A5VM_VstsPredOp : A5VM_Op<"vsts_pred", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_VectorType:$value,
    A5VM_BufferType:$destination,
    Index:$offset,
    Index:$active_lanes,
    OptionalAttr<StrAttr>:$dist
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $destination `[` $offset `]` `,` $active_lanes attr-dict `:` type($value) `,` type($destination) `,` type($active_lanes)
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

#endif // MLIR_DIALECT_PTO_IR_A5VMOPS
