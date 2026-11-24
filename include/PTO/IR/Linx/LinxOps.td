// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

include "PTO/IR/Linx/LinxTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
class LinxPTO_Op<string name, list<Trait> traits = []> :
    Op<PTO_Dialect, name, traits>;
def LinxPTO_AllocTileOp : LinxPTO_Op<"alloc_tile", [MemoryEffects<[MemAlloc]>]> {
  let results = (outs LinxPTO_TileBuf:$tile);
  let assemblyFormat = "attr-dict `:` type($tile)";
}
def LinxPTO_TLoadOp : LinxPTO_Op<"tload", [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = (ins LinxPTO_GMView:$src, LinxPTO_TileBuf:$dst);
  let assemblyFormat = "`ins` `(` $src `:` type($src) `)` `outs` `(` $dst `:` type($dst) `)` attr-dict";
  let hasVerifier = 1;
}
def LinxPTO_TStoreOp : LinxPTO_Op<"tstore", [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = (ins LinxPTO_TileBuf:$src, LinxPTO_GMView:$dst);
  let assemblyFormat = "`ins` `(` $src `:` type($src) `)` `outs` `(` $dst `:` type($dst) `)` attr-dict";
  let hasVerifier = 1;
}
def LinxPTO_TAddOp : LinxPTO_Op<"tadd", [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = (ins LinxPTO_TileBuf:$lhs, LinxPTO_TileBuf:$rhs, LinxPTO_TileBuf:$dst);
  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $dst `:` type($dst) `)` attr-dict
  }];
  let hasVerifier = 1;
}

def LinxPTO_TMatmulOp : LinxPTO_Op<"tmatmul", [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = (ins LinxPTO_TileBuf:$lhs, LinxPTO_TileBuf:$rhs, LinxPTO_TileBuf:$dst);
  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $dst `:` type($dst) `)` attr-dict
  }];
  let hasVerifier = 1;
}

class LinxPTO_UnaryOp<string name> : LinxPTO_Op<name, [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = (ins LinxPTO_TileBuf:$src, LinxPTO_TileBuf:$dst);
  let assemblyFormat = "`ins` `(` $src `:` type($src) `)` `outs` `(` $dst `:` type($dst) `)` attr-dict";
  let hasVerifier = 1;
}
def LinxPTO_TExpOp : LinxPTO_UnaryOp<"texp">;
def LinxPTO_TRowMaxOp : LinxPTO_UnaryOp<"trowmax">;
def LinxPTO_TRowSumOp : LinxPTO_UnaryOp<"trowsum">;
class LinxPTO_RowExpandOp<string name> : LinxPTO_Op<name, [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = (ins LinxPTO_TileBuf:$src, LinxPTO_TileBuf:$row, LinxPTO_TileBuf:$dst);
  let assemblyFormat = [{
    `ins` `(` $src `,` $row `:` type($src) `,` type($row) `)`
    `outs` `(` $dst `:` type($dst) `)` attr-dict
  }];
  let hasVerifier = 1;
}
def LinxPTO_TRowExpandSubOp : LinxPTO_RowExpandOp<"trowexpandsub">;
def LinxPTO_TRowExpandDivOp : LinxPTO_RowExpandOp<"trowexpanddiv">;
