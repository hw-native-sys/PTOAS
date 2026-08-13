// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details.

//===- PTOTileCubeOps - semantic operation family -*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTOTILECUBEOPS
#define MLIR_DIALECT_PTO_IR_PTOTILECUBEOPS

include "PTO/IR/PTOOpCommon.td"

def TMatmulBiasOp : PTO_TOp<"tmatmul.bias", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Matmul with bias add (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$a,
    PTODpsType:$b,
    PTODpsType:$bias,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_AccPhaseAttr, "::mlir::pto::AccPhase::Unspecified">:$accPhase
  );

  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;

  //let builders = [
  //  OpBuilder<(ins "TypeRange":$resultTypes, "Value":$a, "Value":$b, "Value":$bias, "Value":$dst), [{
  //    build($_builder, $_state, resultTypes, a, b, bias, dst, ValueRange{});
  //  }]>
  //];

  let assemblyFormat = [{
    `ins` `(` $a `,` $b `,` $bias `:` qualified(type($a)) `,` qualified(type($b)) `,` qualified(type($bias)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TMATMUL_BIAS"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

// pto.tmatmul.mx ins(%a,%a_scale,%b,%b_scale) outs(%dst) -> tensor?

def TMatmulMxOp : PTO_TOp<"tmatmul.mx", [
    PTO_DpsInitOpInterface,
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Matmul mx (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$a,
    PTODpsType:$a_scale,
    PTODpsType:$b,
    PTODpsType:$b_scale,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_AccPhaseAttr, "::mlir::pto::AccPhase::Unspecified">:$accPhase);

  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $a `,` $a_scale `,` $b `,` $b_scale
          `:` type($a) `,` type($a_scale) `,` type($b) `,` type($b_scale) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TMATMUL_MX"; }
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_M;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

// pto.tmatmul.mx.acc ins(%c_in,%a,%a_scale,%b,%b_scale) outs(%dst) -> tensor?

def TMatmulMxAccOp : PTO_TOp<"tmatmul.mx.acc", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Matmul mx accumulate (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$c_in,
    PTODpsType:$a,
    PTODpsType:$a_scale,
    PTODpsType:$b,
    PTODpsType:$b_scale,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_AccPhaseAttr, "::mlir::pto::AccPhase::Unspecified">:$accPhase);

  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $c_in `,` $a `,` $a_scale `,` $b `,` $b_scale
          `:` type($c_in) `,` type($a) `,` type($a_scale) `,` type($b) `,` type($b_scale) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TMATMUL_MX"; }
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_M;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

// pto.tmatmul.mx.bias ins(%a,%a_scale,%b,%b_scale,%bias) outs(%dst)  -> tensor?

def TMatmulMxBiasOp : PTO_TOp<"tmatmul.mx.bias",[
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Matmul mx with bias (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$a,
    PTODpsType:$a_scale,
    PTODpsType:$b,
    PTODpsType:$b_scale,
    PTODpsType:$bias,
    PTODpsType:$dst);

  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;


  let assemblyFormat = [{
    `ins` `(` $a `,` $a_scale `,` $b `,` $b_scale `,` $bias
          `:` type($a) `,` type($a_scale) `,` type($b) `,` type($b_scale) `,` qualified(type($bias)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TMATMUL_MX"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TMatmulOp : PTO_TOp<"tmatmul", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO matrix multiplication operation (optional bias), destination-style (DPS) in tile world.";

  let arguments = (ins
    PTODpsType:$lhs,
    PTODpsType:$rhs,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_AccPhaseAttr, "::mlir::pto::AccPhase::Unspecified">:$accPhase
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` qualified(type($lhs)) `,` qualified(type($rhs)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let extraClassDeclaration = [{
    static StringRef getOpName() { return "matmul_dps"; }
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_M;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TMatmulAccOp : PTO_TOp<"tmatmul.acc", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO matrix multiplication accumulate operation, destination-style (DPS, no bias).";

  let arguments = (ins
    PTODpsType:$acc_in,
    PTODpsType:$lhs,
    PTODpsType:$rhs,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_AccPhaseAttr, "::mlir::pto::AccPhase::Unspecified">:$accPhase
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $acc_in `,` $lhs `,` $rhs `:` qualified(type($acc_in) ) `,` qualified(type($lhs)) `,` qualified(type($rhs)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let extraClassDeclaration = [{
      ::mlir::pto::PIPE getPipe() {
        return ::mlir::pto::PIPE::PIPE_M;
      }
      ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGemvOp : PTO_TOp<"tgemv", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO matrix-vector multiplication operation (optional bias), destination-style (DPS) in tile world.";

  let arguments = (ins
    PTODpsType:$lhs,
    PTODpsType:$rhs,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_AccPhaseAttr, "::mlir::pto::AccPhase::Unspecified">:$accPhase
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` qualified(type($lhs)) `,` qualified(type($rhs)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let extraClassDeclaration = [{
    static StringRef getOpName() { return "gemv"; }
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_M;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGemvAccOp : PTO_TOp<"tgemv.acc", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO matrix-vector multiplication accumulate operation, destination-style (DPS, no bias).";

  let arguments = (ins
    PTODpsType:$acc_in,
    PTODpsType:$lhs,
    PTODpsType:$rhs,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_AccPhaseAttr, "::mlir::pto::AccPhase::Unspecified">:$accPhase
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $acc_in `,` $lhs `,` $rhs `:` qualified(type($acc_in) ) `,` qualified(type($lhs)) `,` qualified(type($rhs)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let extraClassDeclaration = [{
      ::mlir::pto::PIPE getPipe() {
        return ::mlir::pto::PIPE::PIPE_M;
      }
      ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGemvBiasOp : PTO_TOp<"tgemv.bias", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "GEMV with bias add (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType :$a,
    PTODpsType :$b,
    PTODpsType :$bias,
    PTODpsType :$dst,
    DefaultValuedAttr<PTO_AccPhaseAttr, "::mlir::pto::AccPhase::Unspecified">:$accPhase
  );

  let results = (outs Optional<AnyRankedTensor>:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $a `,` $b `,` $bias `:` qualified(type($a)) `,` qualified(type($b)) `,` qualified(type($bias)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TGEMV_BIAS"; }
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_M;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGemvMxOp : PTO_TOp<"tgemv.mx", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Mixed-precision GEMV with scale tiles (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$a,
    PTODpsType:$a_scale,
    PTODpsType:$b,
    PTODpsType:$b_scale,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_AccPhaseAttr, "::mlir::pto::AccPhase::Unspecified">:$accPhase
  );

  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $a `,` $a_scale `,` $b `,` $b_scale
          `:` type($a) `,` type($a_scale) `,` type($b) `,` type($b_scale) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TGEMV_MX"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGemvMxAccOp : PTO_TOp<"tgemv.mx.acc", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Mixed-precision GEMV accumulate with scale tiles (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$c_in,
    PTODpsType:$a,
    PTODpsType:$a_scale,
    PTODpsType:$b,
    PTODpsType:$b_scale,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_AccPhaseAttr, "::mlir::pto::AccPhase::Unspecified">:$accPhase
  );

  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $c_in `,` $a `,` $a_scale `,` $b `,` $b_scale
          `:` type($c_in) `,` type($a) `,` type($a_scale) `,` type($b) `,` type($b_scale) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TGEMV_MX"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGemvMxBiasOp : PTO_TOp<"tgemv.mx.bias", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Mixed-precision GEMV with bias and scale tiles (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$a,
    PTODpsType:$a_scale,
    PTODpsType:$b,
    PTODpsType:$b_scale,
    PTODpsType:$bias,
    PTODpsType:$dst
  );

  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $a `,` $a_scale `,` $b `,` $b_scale `,` $bias
          `:` type($a) `,` type($a_scale) `,` type($b) `,` type($b_scale) `,` qualified(type($bias)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TGEMV_MX"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

#endif // MLIR_DIALECT_PTO_IR_PTOTILECUBEOPS
