// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOMteOps - semantic operation family -*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_VPTOMTEOPS
#define MLIR_DIALECT_PTO_IR_VPTOMTEOPS

include "PTO/IR/VPTOOpCommon.td"

def PTO_CopyGmToUbufOp : PTO_MteOp<"copy_gm_to_ubuf", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
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

def PTO_MteGmUbOp : PTO_MteOp<"mte_gm_ub", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    I64:$l2_cache_ctl,
    I64:$len_burst,
    I64:$n_burst,
    I64:$nburst_src_stride,
    I64:$nburst_dst_stride,
    Variadic<I64>:$loop_counts,
    Variadic<I64>:$loop_src_strides,
    Variadic<I64>:$loop_dst_strides,
    Optional<AnyTypeOf<[AnyInteger, AnyFloat], "integer/float scalar">>:$pad_value,
    Optional<I64>:$left_padding_count,
    Optional<I64>:$right_padding_count
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::Value":$destination,
      "::mlir::Value":$l2CacheCtl,
      "::mlir::Value":$lenBurst,
      "::mlir::pto::DmaLoopConfig":$nburst,
      "::llvm::ArrayRef<::mlir::pto::DmaLoopConfig>":$loops,
      "::std::optional<::mlir::pto::DmaPadConfig>":$pad
    )>,
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::Value":$destination,
      "::mlir::Value":$l2CacheCtl,
      "::mlir::Value":$lenBurst,
      "::mlir::pto::DmaLoopConfig":$nburst,
      "::std::optional<::mlir::pto::DmaLoopConfig>":$loop1,
      "::std::optional<::mlir::pto::DmaLoopConfig>":$loop2,
      "::std::optional<::mlir::pto::DmaPadConfig>":$pad
    )>
  ];

}

def PTO_CopyUbufToUbufOp : PTO_MteOp<"copy_ubuf_to_ubuf"> {
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

def PTO_CopyCbufToUbufOp : PTO_MteOp<"copy_cbuf_to_ubuf", [
    CubeMicroOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
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

def PTO_CopyUbufToCbufOp : PTO_MteOp<"copy_ubuf_to_cbuf", [
    CubeMicroOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
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

def PTO_MteUbUbOp : PTO_MteOp<"mte_ub_ub", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$n_burst,
    I64:$len_burst,
    I64:$src_stride,
    I64:$dst_stride
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $len_burst
    `nburst` `(` $n_burst `,` $src_stride `,` $dst_stride `)`
    attr-dict `:` type($source) `,` type($destination) `,` type($n_burst) `,`
    type($len_burst) `,` type($src_stride) `,`
    type($dst_stride)
  }];
}

def PTO_MteUbL1Op : PTO_MteOp<"mte_ub_l1", [
    CubeMicroOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$n_burst,
    I64:$len_burst,
    I64:$src_stride,
    I64:$dst_stride
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $len_burst
    `nburst` `(` $n_burst `,` $src_stride `,` $dst_stride `)`
    attr-dict `:` type($source) `,` type($destination) `,` type($n_burst) `,`
    type($len_burst) `,` type($src_stride) `,`
    type($dst_stride)
  }];
}

def PTO_CopyGmToCbufOp : PTO_MteOp<"copy_gm_to_cbuf"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$n_burst,
    I64:$len_burst,
    I64:$src_stride,
    I64:$dst_stride
  );

  let results = (outs);

  let assemblyFormat = [{
    $source `,` $destination `,` $n_burst `,` $len_burst `,` $src_stride `,` $dst_stride
    attr-dict `:` type($source) `,` type($destination) `,` type($n_burst) `,` type($len_burst) `,` type($src_stride) `,` type($dst_stride)
  }];
}

def PTO_CopyGmToCbufMultiNd2NzOp : PTO_MteOp<"copy_gm_to_cbuf_multi_nd2nz"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$sid,
    I64:$loop1_src_stride,
    I64:$l2_cache_ctrl,
    I64:$n_value,
    I64:$d_value,
    I64:$loop4_src_stride,
    I1:$smallc0_en
  );

  let results = (outs);

  let assemblyFormat = [{
    $source `,` $destination `,` $sid `,` $loop1_src_stride `,` $l2_cache_ctrl `,`
    $n_value `,` $d_value `,` $loop4_src_stride `,` $smallc0_en
    attr-dict `:` type($source) `,` type($destination) `,` type($sid) `,`
    type($loop1_src_stride) `,` type($l2_cache_ctrl) `,` type($n_value) `,`
    type($d_value) `,` type($loop4_src_stride) `,` type($smallc0_en)
  }];
}

def PTO_CopyGmToCbufMultiDn2NzOp : PTO_MteOp<"copy_gm_to_cbuf_multi_dn2nz"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$sid,
    I64:$loop1_src_stride,
    I64:$l2_cache_ctrl,
    I64:$n_value,
    I64:$d_value,
    I64:$loop4_src_stride,
    I1:$smallc0_en
  );

  let results = (outs);

  let assemblyFormat = [{
    $source `,` $destination `,` $sid `,` $loop1_src_stride `,` $l2_cache_ctrl `,`
    $n_value `,` $d_value `,` $loop4_src_stride `,` $smallc0_en
    attr-dict `:` type($source) `,` type($destination) `,` type($sid) `,`
    type($loop1_src_stride) `,` type($l2_cache_ctrl) `,` type($n_value) `,`
    type($d_value) `,` type($loop4_src_stride) `,` type($smallc0_en)
  }];
}

def PTO_CopyCbufToBtOp : PTO_MteOp<"copy_cbuf_to_bt"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I1:$conv_control,
    I64:$n_burst,
    I64:$len_burst,
    I64:$source_gap,
    I64:$dst_gap
  );
  let results = (outs);

  let assemblyFormat = [{
    $source `,` $destination `,` $conv_control `,` $n_burst `,` $len_burst `,`
    $source_gap `,` $dst_gap
    attr-dict `:` type($source) `,` type($destination) `,` type($conv_control) `,`
    type($n_burst) `,` type($len_burst) `,` type($source_gap) `,` type($dst_gap)
  }];
}

def PTO_CopyCbufToFbufOp : PTO_MteOp<"copy_cbuf_to_fbuf"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$n_burst,
    I64:$len_burst,
    I64:$source_gap,
    I64:$dst_gap
  );
  let results = (outs);

  let assemblyFormat = [{
    $source `,` $destination `,` $n_burst `,` $len_burst `,`
    $source_gap `,` $dst_gap
    attr-dict `:` type($source) `,` type($destination) `,` type($n_burst) `,`
    type($len_burst) `,` type($source_gap) `,` type($dst_gap)
  }];
}

def PTO_LoadCbufToCaOp : PTO_MteOp<"load_cbuf_to_ca"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$m_start,
    I64:$k_start,
    I64:$m_step,
    I64:$k_step,
    I64:$src_stride,
    I64:$dst_stride,
    DefaultValuedAttr<BoolAttr, "false">:$transpose
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $m_start `,` $k_start `,` $m_step `,`
    $k_step `,` $src_stride `,` $dst_stride attr-dict `:` type($source) `,`
    type($destination) `,` type($m_start) `,` type($k_start) `,`
    type($m_step) `,` type($k_step) `,` type($src_stride) `,`
    type($dst_stride)
  }];
}

def PTO_LoadCbufToCaS4Op : PTO_MteOp<"load_cbuf_to_ca_s4"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$m_start,
    I64:$k_start,
    I64:$m_step,
    I64:$k_step,
    I64:$src_stride,
    I64:$dst_stride,
    I64:$transpose
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $m_start `,` $k_start `,` $m_step `,`
    $k_step `,` $src_stride `,` $dst_stride `,` $transpose
    attr-dict `:` type($source) `,` type($destination) `,` type($m_start) `,`
    type($k_start) `,` type($m_step) `,` type($k_step) `,` type($src_stride) `,`
    type($dst_stride) `,` type($transpose)
  }];
}

def PTO_LoadCbufToCbOp : PTO_MteOp<"load_cbuf_to_cb"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$m_start,
    I64:$k_start,
    I64:$m_step,
    I64:$k_step,
    I64:$src_stride,
    I64:$dst_stride,
    DefaultValuedAttr<BoolAttr, "false">:$transpose
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $m_start `,` $k_start `,` $m_step `,`
    $k_step `,` $src_stride `,` $dst_stride attr-dict `:` type($source) `,`
    type($destination) `,` type($m_start) `,` type($k_start) `,`
    type($m_step) `,` type($k_step) `,` type($src_stride) `,`
    type($dst_stride)
  }];
}

def PTO_LoadCbufToCbS4Op : PTO_MteOp<"load_cbuf_to_cb_s4"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$m_start,
    I64:$k_start,
    I64:$m_step,
    I64:$k_step,
    I64:$src_stride,
    I64:$dst_stride,
    I64:$transpose
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $m_start `,` $k_start `,` $m_step `,`
    $k_step `,` $src_stride `,` $dst_stride `,` $transpose
    attr-dict `:` type($source) `,` type($destination) `,` type($m_start) `,`
    type($k_start) `,` type($m_step) `,` type($k_step) `,` type($src_stride) `,`
    type($dst_stride) `,` type($transpose)
  }];
}

def PTO_LoadCbufToCaMxOp : PTO_MteOp<"load_cbuf_to_ca_mx"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$x_start_position,
    I64:$y_start_position,
    I64:$x_step,
    I64:$y_step,
    I64:$src_stride,
    I64:$dst_stride
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $x_start_position `,` $y_start_position `,`
    $x_step `,` $y_step `,` $src_stride `,` $dst_stride attr-dict `:`
    type($source) `,` type($destination) `,` type($x_start_position) `,`
    type($y_start_position) `,` type($x_step) `,` type($y_step) `,`
    type($src_stride) `,` type($dst_stride)
  }];
}

def PTO_LoadCbufToCbMxOp : PTO_MteOp<"load_cbuf_to_cb_mx"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$x_start_position,
    I64:$y_start_position,
    I64:$x_step,
    I64:$y_step,
    I64:$src_stride,
    I64:$dst_stride
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $x_start_position `,` $y_start_position `,`
    $x_step `,` $y_step `,` $src_stride `,` $dst_stride attr-dict `:`
    type($source) `,` type($destination) `,` type($x_start_position) `,`
    type($y_start_position) `,` type($x_step) `,` type($y_step) `,`
    type($src_stride) `,` type($dst_stride)
  }];
}

def PTO_CopyMatrixCcToGmOp : PTO_MteOp<"copy_matrix_cc_to_gm"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$xm,
    I64:$xt
  );

  let results = (outs);

  let assemblyFormat = [{
    $source `,` $destination `,` $xm `,` $xt attr-dict `:` type($source) `,` type($destination) `,` type($xm) `,` type($xt)
  }];
}

def PTO_CopyMatrixCcToCbufOp : PTO_MteOp<"copy_matrix_cc_to_cbuf"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$config0,
    I64:$config1
  );

  let results = (outs);

  let assemblyFormat = [{
    $source `,` $destination `,` $config0 `,` $config1 attr-dict `:` type($source) `,`
    type($destination) `,` type($config0) `,` type($config1)
  }];
}

def PTO_CopyMatrixCcToUbOp : PTO_MteOp<"copy_matrix_cc_to_ub"> {
  let arguments = (ins
    PTO_BufferType:$source,
    PTO_BufferType:$destination,
    I64:$config0,
    I64:$config1
  );

  let results = (outs);

  let assemblyFormat = [{
    $source `,` $destination `,` $config0 `,` $config1 attr-dict `:` type($source) `,`
    type($destination) `,` type($config0) `,` type($config1)
  }];
}

def PTO_CopyUbufToGmOp : PTO_MteOp<"copy_ubuf_to_gm", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    I64:$sid,
    I64:$n_burst,
    I64:$len_burst,
    I64:$l2_cache_ctl,
    I64:$burst_dst_stride,
    I64:$burst_src_stride
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $destination `,` $sid `,` $n_burst `,` $len_burst `,`
    $l2_cache_ctl `,` $burst_dst_stride `,` $burst_src_stride
    attr-dict `:` type($source) `,` type($destination) `,` type($sid) `,` type($n_burst) `,`
    type($len_burst) `,` type($l2_cache_ctl) `,`
    type($burst_dst_stride) `,` type($burst_src_stride)
  }];
}

def PTO_MteUbGmOp : PTO_MteOp<"mte_ub_gm", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    I64:$len_burst,
    I64:$n_burst,
    I64:$nburst_src_stride,
    I64:$nburst_dst_stride,
    Optional<I64>:$l2_cache_ctl,
    Variadic<I64>:$loop_counts,
    Variadic<I64>:$loop_src_strides,
    Variadic<I64>:$loop_dst_strides
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::Value":$destination,
      "::mlir::Value":$lenBurst,
      "::mlir::pto::DmaLoopConfig":$nburst,
      "::mlir::Value":$l2CacheCtl,
      "::llvm::ArrayRef<::mlir::pto::DmaLoopConfig>":$loops
    )>,
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::Value":$destination,
      "::mlir::Value":$lenBurst,
      "::mlir::pto::DmaLoopConfig":$nburst,
      "::mlir::Value":$l2CacheCtl,
      "::std::optional<::mlir::pto::DmaLoopConfig>":$loop1,
      "::std::optional<::mlir::pto::DmaLoopConfig>":$loop2
    )>
  ];
}

// Cube bridge wrappers: fuse common register config sequences with the core
// cube transfer/load/store ops, similar to dma_load/dma_store wrappers.
def PTO_MteGmL1Op : PTO_MteOp<"mte_gm_l1", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    I64:$len_burst,
    I64:$n_burst,
    I64:$nburst_src_stride,
    I64:$nburst_dst_stride,
    Variadic<I64>:$loop_counts,
    Variadic<I64>:$loop_src_strides,
    Variadic<I64>:$loop_dst_strides
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::Value":$destination,
      "::mlir::Value":$lenBurst,
      "::mlir::pto::DmaLoopConfig":$nburst,
      "::llvm::ArrayRef<::mlir::pto::DmaLoopConfig>":$loops
    )>,
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::Value":$destination,
      "::mlir::Value":$lenBurst,
      "::mlir::pto::DmaLoopConfig":$nburst,
      "::std::optional<::mlir::pto::DmaLoopConfig>":$loop1,
      "::std::optional<::mlir::pto::DmaLoopConfig>":$loop2
    )>
  ];
}

def PTO_MteL1UbOp : PTO_MteOp<"mte_l1_ub", [
    CubeMicroOpInterface,
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    I64:$len_burst,
    I64:$n_burst,
    I64:$nburst_src_stride,
    I64:$nburst_dst_stride,
    Variadic<I64>:$loop_counts,
    Variadic<I64>:$loop_src_strides,
    Variadic<I64>:$loop_dst_strides
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::Value":$destination,
      "::mlir::Value":$lenBurst,
      "::mlir::pto::DmaLoopConfig":$nburst,
      "::llvm::ArrayRef<::mlir::pto::DmaLoopConfig>":$loops
    )>,
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::Value":$destination,
      "::mlir::Value":$lenBurst,
      "::mlir::pto::DmaLoopConfig":$nburst,
      "::std::optional<::mlir::pto::DmaLoopConfig>":$loop1,
      "::std::optional<::mlir::pto::DmaLoopConfig>":$loop2
    )>
  ];
}

def PTO_MteL1BtOp : PTO_MteOp<"mte_l1_bt", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    I64:$len_burst,
    I64:$n_burst,
    I64:$nburst_src_gap,
    I64:$nburst_dst_gap
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::Value":$destination,
      "::mlir::Value":$lenBurst,
      "::mlir::pto::DmaLoopConfig":$nburst
    )>
  ];
}

def PTO_MteL1FbOp : PTO_MteOp<"mte_l1_fb", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    I64:$len_burst,
    I64:$n_burst,
    I64:$nburst_src_gap,
    I64:$nburst_dst_gap
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::Value":$destination,
      "::mlir::Value":$lenBurst,
      "::mlir::pto::DmaLoopConfig":$nburst
    )>
  ];
}

def PTO_MteGmL1FracOp : PTO_MteOp<"mte_gm_l1_frac", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    I64:$n_value,
    I64:$d_value,
    I64:$src_inner_stride,
    I64:$group_count,
    I64:$dst_loop2_stride,
    I64:$dst_loop3_stride,
    I64:$dst_loop4_stride,
    I64:$l2_cache_ctrl,
    I1:$smallc0_en,
    Optional<I64>:$src_outer_stride,
    PTO_CubeLoadFracModeAttr:$mode
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::Value":$destination,
      "::mlir::pto::CubeLoadFracMode":$mode,
      "::mlir::pto::CubeLoadFracShapeConfig":$shape,
      "::mlir::pto::CubeLoadFracSrcLayoutConfig":$srcLayout,
      "::mlir::pto::CubeLoadFracDstGroupConfig":$dstGroup,
      "::mlir::pto::CubeLoadFracCtrlConfig":$ctrl
    )>
  ];
}

def PTO_MteL1L0aOp : PTO_MteOp<"mte_l1_l0a", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    Optional<I64>:$m,
    Optional<I64>:$k,
    Optional<I64>:$start_row,
    Optional<I64>:$start_col,
    Optional<I64>:$m_start,
    Optional<I64>:$k_start,
    Optional<I64>:$m_step,
    Optional<I64>:$k_step,
    Optional<I64>:$src_stride,
    Optional<I64>:$dst_stride,
    DefaultValuedAttr<BoolAttr, "false">:$transpose
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MteL1L0bOp : PTO_MteOp<"mte_l1_l0b", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    Optional<I64>:$k,
    Optional<I64>:$n,
    Optional<I64>:$start_row,
    Optional<I64>:$start_col,
    Optional<I64>:$m_start,
    Optional<I64>:$k_start,
    Optional<I64>:$m_step,
    Optional<I64>:$k_step,
    Optional<I64>:$src_stride,
    Optional<I64>:$dst_stride,
    DefaultValuedAttr<BoolAttr, "false">:$transpose
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MteL1L0aMxOp : PTO_MteOp<"mte_l1_l0a_mx", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    Optional<I64>:$m,
    Optional<I64>:$k,
    Optional<I64>:$start_row,
    Optional<I64>:$start_col,
    Optional<I64>:$x_start,
    Optional<I64>:$y_start,
    Optional<I64>:$x_step,
    Optional<I64>:$y_step,
    Optional<I64>:$src_stride,
    Optional<I64>:$dst_stride
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MteL1L0bMxOp : PTO_MteOp<"mte_l1_l0b_mx", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    Optional<I64>:$k,
    Optional<I64>:$n,
    Optional<I64>:$start_row,
    Optional<I64>:$start_col,
    Optional<I64>:$x_start,
    Optional<I64>:$y_start,
    Optional<I64>:$x_step,
    Optional<I64>:$y_step,
    Optional<I64>:$src_stride,
    Optional<I64>:$dst_stride
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MteL0cL1Op : PTO_MteOp<"mte_l0c_l1", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    I64:$m,
    I64:$n,
    I64:$src_stride,
    I64:$dst_stride,
    Optional<PTO_AccStorePayloadType>:$pre_quant,
    Optional<PTO_AccStorePayloadType>:$pre_relu,
    Optional<PTO_AccStoreClipPayloadType>:$clip_value,
    Optional<I64>:$split,
    Optional<I64>:$loop0_src_stride,
    Optional<I64>:$loop3_count,
    Optional<I64>:$loop3_src_stride,
    Optional<I64>:$loop3_dst_stride,
    OptionalAttr<PTO_AccStoreModeAttr>:$mode,
    OptionalAttr<PTO_UnitFlagCtrlAttr>:$unit_flag,
    OptionalAttr<PTO_AccStoreQuantPreModeAttr>:$pre_quant_mode,
    OptionalAttr<PTO_ReluPreModeAttr>:$pre_relu_mode,
    OptionalAttr<PTO_AccStoreAtomicTypeAttr>:$atomic_type,
    OptionalAttr<PTO_AccStoreAtomicOpAttr>:$atomic_op,
    OptionalAttr<PTO_AccStoreSatModeAttr>:$sat_mode
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MteL0cGmOp : PTO_MteOp<"mte_l0c_gm", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    I64:$m,
    I64:$n,
    I64:$src_stride,
    I64:$dst_stride,
    Optional<PTO_AccStorePayloadType>:$pre_quant,
    Optional<PTO_AccStorePayloadType>:$pre_relu,
    Optional<PTO_AccStoreClipPayloadType>:$clip_value,
    I64:$sid,
    I64:$l2_cache_ctrl,
    Optional<I64>:$split,
    Optional<I64>:$loop0_src_stride,
    Optional<I64>:$loop3_count,
    Optional<I64>:$loop3_src_stride,
    Optional<I64>:$loop3_dst_stride,
    OptionalAttr<PTO_AccStoreModeAttr>:$mode,
    OptionalAttr<PTO_UnitFlagCtrlAttr>:$unit_flag,
    OptionalAttr<PTO_AccStoreQuantPreModeAttr>:$pre_quant_mode,
    OptionalAttr<PTO_ReluPreModeAttr>:$pre_relu_mode,
    OptionalAttr<PTO_AccStoreAtomicTypeAttr>:$atomic_type,
    OptionalAttr<PTO_AccStoreAtomicOpAttr>:$atomic_op,
    OptionalAttr<PTO_AccStoreSatModeAttr>:$sat_mode
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MteL0cUbOp : PTO_MteOp<"mte_l0c_ub", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferLikeType:$source,
    PTO_BufferLikeType:$destination,
    I64:$m,
    I64:$n,
    I64:$src_stride,
    I64:$dst_stride,
    Optional<PTO_AccStorePayloadType>:$pre_quant,
    Optional<PTO_AccStorePayloadType>:$pre_relu,
    Optional<PTO_AccStoreClipPayloadType>:$clip_value,
    Optional<I64>:$sub_blockid,
    Optional<I64>:$split,
    Optional<I64>:$loop0_src_stride,
    Optional<I64>:$loop3_count,
    Optional<I64>:$loop3_src_stride,
    Optional<I64>:$loop3_dst_stride,
    PTO_AccStoreUbDstModeAttr:$dst_mode,
    OptionalAttr<PTO_AccStoreModeAttr>:$mode,
    OptionalAttr<PTO_UnitFlagCtrlAttr>:$unit_flag,
    OptionalAttr<PTO_AccStoreQuantPreModeAttr>:$pre_quant_mode,
    OptionalAttr<PTO_ReluPreModeAttr>:$pre_relu_mode,
    OptionalAttr<PTO_AccStoreSatModeAttr>:$sat_mode
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

// NOTE: Unvalidated new x2 / pair / align-store-family abstractions. Added to
// reflect CCE builtin families but not yet end-to-end validated.

#endif // MLIR_DIALECT_PTO_IR_VPTOMTEOPS
