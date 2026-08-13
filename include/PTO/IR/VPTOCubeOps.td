// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOCubeOps - semantic operation family -*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_VPTOCUBEOPS
#define MLIR_DIALECT_PTO_IR_VPTOCUBEOPS

include "PTO/IR/VPTOOpCommon.td"

def PTO_MadOp : PTO_CubeMicroOp<"mad", [
    DeclareOpInterfaceMethods<PTO_MadSemanticOpInterface, [
      "isMadMxFamily", "hasBiasOperand", "readsAccumulator",
      "supportsTf32Mode", "getBiasOrNull"
    ]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$lhs,
    PTO_BufferType:$rhs,
    PTO_BufferType:$dst,
    I64:$m,
    I64:$n,
    I64:$k,
    OptionalAttr<PTO_MadUnitFlagModeAttr>:$unit_flag_mode,
    UnitAttr:$disable_gemv,
    OptionalAttr<PTO_MadSatModeAttr>:$sat_mode,
    OptionalAttr<PTO_Tf32ModeAttr>:$tf32_mode,
    UnitAttr:$n_dir
  );
  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MadAccOp : PTO_CubeMicroOp<"mad_acc", [
    DeclareOpInterfaceMethods<PTO_MadSemanticOpInterface, [
      "isMadMxFamily", "hasBiasOperand", "readsAccumulator",
      "supportsTf32Mode", "getBiasOrNull"
    ]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$lhs,
    PTO_BufferType:$rhs,
    PTO_BufferType:$dst,
    I64:$m,
    I64:$n,
    I64:$k,
    OptionalAttr<PTO_MadUnitFlagModeAttr>:$unit_flag_mode,
    UnitAttr:$disable_gemv,
    OptionalAttr<PTO_MadSatModeAttr>:$sat_mode,
    OptionalAttr<PTO_Tf32ModeAttr>:$tf32_mode,
    UnitAttr:$n_dir
  );
  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MadBiasOp : PTO_CubeMicroOp<"mad_bias", [
    DeclareOpInterfaceMethods<PTO_MadSemanticOpInterface, [
      "isMadMxFamily", "hasBiasOperand", "readsAccumulator",
      "supportsTf32Mode", "getBiasOrNull"
    ]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$lhs,
    PTO_BufferType:$rhs,
    PTO_BufferType:$dst,
    PTO_BufferType:$bias,
    I64:$m,
    I64:$n,
    I64:$k,
    OptionalAttr<PTO_MadUnitFlagModeAttr>:$unit_flag_mode,
    UnitAttr:$disable_gemv,
    OptionalAttr<PTO_MadSatModeAttr>:$sat_mode,
    OptionalAttr<PTO_Tf32ModeAttr>:$tf32_mode,
    UnitAttr:$n_dir
  );
  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MadMxOp : PTO_CubeMicroOp<"mad_mx", [
    DeclareOpInterfaceMethods<PTO_MadSemanticOpInterface, [
      "isMadMxFamily", "hasBiasOperand", "readsAccumulator",
      "supportsTf32Mode", "getBiasOrNull", "getTf32ModeAttr"
    ]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$lhs,
    PTO_BufferType:$rhs,
    PTO_BufferType:$dst,
    I64:$m,
    I64:$n,
    I64:$k,
    OptionalAttr<PTO_MadUnitFlagModeAttr>:$unit_flag_mode,
    UnitAttr:$disable_gemv,
    OptionalAttr<PTO_MadSatModeAttr>:$sat_mode,
    UnitAttr:$n_dir
  );
  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MadMxAccOp : PTO_CubeMicroOp<"mad_mx_acc", [
    DeclareOpInterfaceMethods<PTO_MadSemanticOpInterface, [
      "isMadMxFamily", "hasBiasOperand", "readsAccumulator",
      "supportsTf32Mode", "getBiasOrNull", "getTf32ModeAttr"
    ]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$lhs,
    PTO_BufferType:$rhs,
    PTO_BufferType:$dst,
    I64:$m,
    I64:$n,
    I64:$k,
    OptionalAttr<PTO_MadUnitFlagModeAttr>:$unit_flag_mode,
    UnitAttr:$disable_gemv,
    OptionalAttr<PTO_MadSatModeAttr>:$sat_mode,
    UnitAttr:$n_dir
  );
  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MadMxBiasOp : PTO_CubeMicroOp<"mad_mx_bias", [
    DeclareOpInterfaceMethods<PTO_MadSemanticOpInterface, [
      "isMadMxFamily", "hasBiasOperand", "readsAccumulator",
      "supportsTf32Mode", "getBiasOrNull", "getTf32ModeAttr"
    ]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$lhs,
    PTO_BufferType:$rhs,
    PTO_BufferType:$dst,
    PTO_BufferType:$bias,
    I64:$m,
    I64:$n,
    I64:$k,
    OptionalAttr<PTO_MadUnitFlagModeAttr>:$unit_flag_mode,
    UnitAttr:$disable_gemv,
    OptionalAttr<PTO_MadSatModeAttr>:$sat_mode,
    UnitAttr:$n_dir
  );
  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def PTO_MadRawOp : PTO_CubeMicroOp<"mad_raw", [
    DeclareOpInterfaceMethods<PTO_MadRawOpInterface, [
      "isMadMxFamily", "hasBiasOperand", "getBiasOrNull"
    ]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$lhs,
    PTO_BufferType:$rhs,
    PTO_BufferType:$dst,
    I64:$xt
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $dst `,` $xt attr-dict `:` type($lhs) `,` type($rhs) `,` type($dst) `,` type($xt)
  }];
}

def PTO_MadBiasRawOp : PTO_CubeMicroOp<"mad_bias_raw", [
    DeclareOpInterfaceMethods<PTO_MadRawOpInterface, [
      "isMadMxFamily", "hasBiasOperand", "getBiasOrNull"
    ]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$lhs,
    PTO_BufferType:$rhs,
    PTO_BufferType:$dst,
    PTO_BufferType:$bias,
    I64:$xt
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $dst `,` $bias `,` $xt attr-dict `:` type($lhs) `,` type($rhs) `,` type($dst) `,` type($bias) `,` type($xt)
  }];
}

def PTO_MadMxRawOp : PTO_CubeMicroOp<"mad_mx_raw", [
    DeclareOpInterfaceMethods<PTO_MadRawOpInterface, [
      "isMadMxFamily", "hasBiasOperand", "getBiasOrNull"
    ]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$lhs,
    PTO_BufferType:$rhs,
    PTO_BufferType:$dst,
    I64:$xt
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $dst `,` $xt attr-dict `:` type($lhs) `,` type($rhs) `,` type($dst) `,` type($xt)
  }];
}

def PTO_MadMxBiasRawOp : PTO_CubeMicroOp<"mad_mx_bias_raw", [
    DeclareOpInterfaceMethods<PTO_MadRawOpInterface, [
      "isMadMxFamily", "hasBiasOperand", "getBiasOrNull"
    ]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins
    PTO_BufferType:$lhs,
    PTO_BufferType:$rhs,
    PTO_BufferType:$dst,
    PTO_BufferType:$bias,
    I64:$xt
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $dst `,` $bias `,` $xt attr-dict `:` type($lhs) `,` type($rhs) `,` type($dst) `,` type($bias) `,` type($xt)
  }];
}

#endif // MLIR_DIALECT_PTO_IR_VPTOCUBEOPS
