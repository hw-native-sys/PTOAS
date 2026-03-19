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

def A5VM_VldsOp : A5VM_Op<"vlds", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    A5VM_BufferType:$source,
    Index:$offset
  );

  let results = (outs A5VM_VectorType:$result);

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
