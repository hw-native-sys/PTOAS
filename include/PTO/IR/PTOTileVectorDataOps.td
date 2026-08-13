// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOTileVectorDataOps - semantic operation family -*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTOTILEVECTORDATAOPS
#define MLIR_DIALECT_PTO_IR_PTOTILEVECTORDATAOPS

include "PTO/IR/PTOOpCommon.td"

def TTransOp : PTO_TOp<"ttrans", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO matrix trans operation (destination-style, DPS).";
  let description = [{
    DPS form:
      tile buf:  pto.ttrans ins(%src, %tmp) outs(%dst)

    The current PTO IR form always carries a tmp operand, but PTO IR does not
    require tmp to match the shape or valid shape of src/dst.
  }];
  let arguments = (ins
    PTODpsType:$src,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );
  let results = (outs);

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_V; // Vector Pipe
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getDstMutable();
    }
  }];
}

def TMovOp  : PTO_TOp<"tmov", [
  AttrSizedOperandSegments,
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Move data between domains (DPS version).";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    Optional<PTODpsType>:$fp,
    Optional<I64>:$preQuantScalar,
    OptionalAttr<PTO_AccToVecModeAttr>:$accToVecMode,
    DefaultValuedAttr<PTO_ReluPreModeAttr, "::mlir::pto::ReluPreMode::NoRelu">:$reluPreMode
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src))
    (`,` $fp^ `:` qualified(type($fp)))?
    (`,` $preQuantScalar^ `:` type($preQuantScalar))?
    `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      // TMOV spans multiple hardware pipelines depending on the source/dest
      // domains. Most tile-domain moves are executed by MTE1 (e.g. MAT->L0),
      // while UB->UB copies are vector-pipe operations.
      auto getASFromType = [](Type ty)
          -> std::optional<::mlir::pto::AddressSpace> {
        if (auto tb = llvm::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = llvm::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace();
          return std::nullopt;
        }
        return std::nullopt;
      };

      auto sOpt = getASFromType(getSrc().getType());
      auto dOpt = getASFromType(getDst().getType());
      if (!sOpt.has_value() || !dOpt.has_value())
        return ::mlir::pto::PIPE::PIPE_V;

      const auto s = sOpt.value();
      const auto d = dOpt.value();

      // UB -> UB copy is vector pipe.
      if (s == ::mlir::pto::AddressSpace::VEC && d == ::mlir::pto::AddressSpace::VEC) {
        return ::mlir::pto::PIPE::PIPE_V;
      }

      // MAT -> L0A/L0B/Bias moves run on MTE1.
      if ((s == ::mlir::pto::AddressSpace::MAT &&
           (d == ::mlir::pto::AddressSpace::LEFT || d == ::mlir::pto::AddressSpace::RIGHT ||
            d == ::mlir::pto::AddressSpace::BIAS))) {
        return ::mlir::pto::PIPE::PIPE_MTE1;
      }

      if (s == ::mlir::pto::AddressSpace::MAT &&
          d == ::mlir::pto::AddressSpace::SCALING) {
        // PTOAS uses SCALING for two distinct PTO-ISA destinations. A5 MX
        // scale tiles are ScaleLeft/ScaleRight loads into L0A/L0B and run on
        // MTE1. Regular Scaling tiles are FBUF loads (TMOV_M2S) and run on
        // Distinguish the MX-only f8E8M0 type from regular scaling tiles.
        Type dstElemTy;
        if (auto tb = llvm::dyn_cast<::mlir::pto::TileBufType>(getDst().getType()))
          dstElemTy = tb.getElementType();
        if (dstElemTy && llvm::isa<::mlir::pto::F8E8M0Type>(dstElemTy))
          return ::mlir::pto::PIPE::PIPE_MTE1;
        return ::mlir::pto::PIPE::PIPE_FIX;
      }

      // ACC -> MAT / VEC use FIX in PTO-ISA sync mapping (TMOV_A2M/TMOV_A2V).
      if (s == ::mlir::pto::AddressSpace::ACC &&
          (d == ::mlir::pto::AddressSpace::MAT ||
           d == ::mlir::pto::AddressSpace::VEC)) {
        return ::mlir::pto::PIPE::PIPE_FIX;
      }

      // Fallback: treat as vector pipe (safe default for most intra-domain moves).
      return ::mlir::pto::PIPE::PIPE_V;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}


// =============================================================================
// System/Runtime Query Ops
// =============================================================================

def TileValidRowsOp : PTO_Op<"tile_valid_rows", [Pure]> {
  let summary = "Extract valid row count from a tile_buf.";
  let description = [{
    Returns the valid row count (v_row) of a `tile_buf` as an `index` value.
    When the tile_buf has a static v_row, FoldTileBufIntrinsics folds this
    into `arith.constant`. When v_row is dynamic (`?`), the fold resolves
    it to the runtime index value carried by the tile_buf.
  }];

  let arguments = (ins TileBufType:$src);
  let results = (outs Index:$result);

  let assemblyFormat = [{
    $src attr-dict `:` qualified(type($src)) `->` type($result)
  }];
}

def TileValidColsOp : PTO_Op<"tile_valid_cols", [Pure]> {
  let summary = "Extract valid column count from a tile_buf.";
  let description = [{
    Returns the valid column count (v_col) of a `tile_buf` as an `index` value.
    When the tile_buf has a static v_col, FoldTileBufIntrinsics folds this
    into `arith.constant`. When v_col is dynamic (`?`), the fold resolves
    it to the runtime index value carried by the tile_buf.
  }];

  let arguments = (ins TileBufType:$src);
  let results = (outs Index:$result);

  let assemblyFormat = [{
    $src attr-dict `:` qualified(type($src)) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// FFT Configuration Operation
//===----------------------------------------------------------------------===//

def MGatherOp : PTO_TOp<"mgather", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  MteOpInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Gather-load elements from memory into a tile using per-element indices (tile world, ins/outs).";
  let description = [{
    Two destination forms are supported:

    - **GM -> UB (VEC)**: `mem` is a GM table, `idx` is a UB index tile, and
      `dst` is a UB (`loc=vec`) tile. This is the default path.
    - **GM -> L1 (MAT)**: `dst` is an L1 / cube (`loc=mat`) tile in NZ layout,
      `idx` is supplied as a **GM** tensor (the cube core cannot read UB on
      A5), and `Coalesce::Elem` additionally requires a contiguous **GM
      `scratch`** workspace used to stage the gathered elements into NZ layout
      before the bulk GM -> L1 copy. `Coalesce::Row` needs no scratch. The path
      is selected by `dst` being a `loc=mat` tile and mirrors the pto-isa
      `MGATHER` GM -> L1 overloads.
  }];

  let arguments = (ins
    PTODpsType:$mem,
    PTODpsType:$idx,
    PTODpsType:$dst,
    Optional<PTODpsType>:$scratch,
    PTO_CoalesceAttr:$coalesce,
    DefaultValuedAttr<PTO_GatherOOBAttr, "::mlir::pto::GatherOOB::Undefined">:$gatherOob);

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "MGATHER"; }
    ::mlir::pto::PIPE getPipe() {
      // GM -> L1 gather: the destination is an L1 / cube (loc=mat) tile and the
      // gather is issued as a GM -> L1 DMA load (copy_gm_to_cbuf / nd2nz) on the
      // MTE2 pipe on every arch, mirroring pto.tload. The GM -> UB (VEC) path
      // below keeps its A5 SIMT (PIPE_V) / A2A3 MTE2 selection.
      //
      // Resolve the destination address space for InsertSync and
      // GraphSyncSolver through the OpPipeInterface.
      auto getAddressSpace =
          [](::mlir::Type ty) -> std::optional<::mlir::pto::AddressSpace> {
        if (auto tb = ::mlir::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = ::mlir::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace();
          return std::nullopt;
        }
        return std::nullopt;
      };
      if (auto as = getAddressSpace(getDst().getType());
          as && *as == ::mlir::pto::AddressSpace::MAT)
        return ::mlir::pto::PIPE::PIPE_MTE2;

      auto isA5Target = [&]() -> bool {
        auto moduleOp = getOperation()->getParentOfType<::mlir::ModuleOp>();
        if (!moduleOp)
          return false;

        if (auto arch =
                moduleOp->getAttrOfType<::mlir::StringAttr>("pto.target_arch")) {
          if (arch.getValue().equals_insensitive("a5"))
            return true;
        }

        if (auto spec =
                moduleOp->getAttrOfType<::mlir::StringAttr>("pto.device-spec")) {
          auto s = spec.getValue();
          if (s.starts_with("Ascend950") || s.starts_with("Ascend910_95"))
            return true;
        }

        return false;
      };

      return isA5Target() ? ::mlir::pto::PIPE::PIPE_V
                          : ::mlir::pto::PIPE::PIPE_MTE2;
    }
    ::mlir::Value getSource() { return getMem(); }
    ::mlir::Value getDestination() { return getDst(); }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}



// ---- tile-world TOp version ----
// pto.tsetval ins(%offset, %val) outs(%dst) : PTODpsType, index, T

def TSetValOp : PTO_TOp<"tsetval", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Write a scalar value into a single element of dst at offset (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$dst,
    Index:$offset,
    ScalarType:$val
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $offset `,` $val `:` type($offset) `,` type($val) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "SETVAL"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_S; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

// ---- tile-world TOp version ----
// pto.tgetval ins(%src, %offset) outs(%dst) : PTODpsType, index -> T

def TGetValOp : PTO_TOp<"tgetval", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Read a single element from tile-like src at offset into a scalar (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$src,
    Index:$offset
  );

  let results = (outs ScalarType:$dst);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $offset `:` qualified(type($src)) `,` type($offset) `)`
    `outs` `:` qualified(type($dst) )
    attr-dict
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "GETVAL"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_S; }
  }];
}

def THistogramOp : PTO_TOp<"thistogram", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "THISTOGRAM: accumulate per-row 256-bin histograms.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$idx,
    PTODpsType:$dst,
    DefaultValuedOptionalAttr<I32Attr, "1">:$byte
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $idx `:` qualified(type($src)) `,` qualified(type($idx)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGetScaleAddrOp : PTO_TOp<"tget_scale_addr", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Bind a scaling tile to the scaled address of a source tile.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_S; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}


// ---- tile-world TOp version (with 't') ----
// pto.mscatter ins(%src, %idx) outs(%mem) [ ...]

def MScatterOp : PTO_TOp<"mscatter", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  MteOpInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Scatter-store elements from a tile into memory using per-element indices (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$idx,
    PTODpsType:$mem,               // outs target
    OptionalAttr<PTO_CoalesceAttr>:$coalesce,
    DefaultValuedAttr<PTO_ScatterAtomicOpAttr, "::mlir::pto::ScatterAtomicOp::None">:$scatterAtomicOp,
    DefaultValuedAttr<PTO_ScatterOOBAttr, "::mlir::pto::ScatterOOB::Undefined">:$scatterOob,
    OptionalAttr<PTO_ScatterConflictAttr>:$scatterConflict
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "MSCATTER"; }
    ::mlir::pto::PIPE getPipe() {
      auto isA5Target = [&]() -> bool {
        auto moduleOp = getOperation()->getParentOfType<::mlir::ModuleOp>();
        if (!moduleOp)
          return false;

        if (auto arch =
                moduleOp->getAttrOfType<::mlir::StringAttr>("pto.target_arch")) {
          if (arch.getValue().equals_insensitive("a5"))
            return true;
        }

        if (auto spec =
                moduleOp->getAttrOfType<::mlir::StringAttr>("pto.device-spec")) {
          auto s = spec.getValue();
          if (s.starts_with("Ascend950") || s.starts_with("Ascend910_95"))
            return true;
        }

        return false;
      };

      return isA5Target() ? ::mlir::pto::PIPE::PIPE_V
                          : ::mlir::pto::PIPE::PIPE_MTE3;
    }
    ::mlir::Value getSource() { return getSrc(); }
    ::mlir::Value getDestination() { return getMem(); }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getMemMutable(); }
  }];
}

//===----------------------------------------------------------------------===//
// PTO_TOPs(Tilebuffer, DPS)
//===----------------------------------------------------------------------===//

def TConcatOp: PTO_TOp<"tconcat", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TCONCAT: Concatenate two tiles along the column dimension.";
  let description = [{
    Semantics (per row):
      dst[:, 0:validCol0] = src0
      dst[:, validCol0:validCol0+validCol1] = src1

    Constraints are implementation-defined; see verifier for current checks.
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TConcatidxOp: PTO_TOp<"tconcatidx", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TCONCAT (indexed): Concatenate two tiles along the column dimension with per-row index control.";
  let description = [{
    Indexed form of tconcat. Takes two additional index tiles (src0Idx, src1Idx)
    that specify the number of columns to copy from each source on a per-row basis.

    Semantics (per row):
      idx0_num = src0Idx[row, 0]
      idx1_num = src1Idx[row, 0]
      copy from src0: min(idx0_num, src0_valid_col, dst_valid_col) columns
      copy from src1: min(idx1_num, src1_valid_col, dst_valid_col - copied_from_src0) columns
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$src0Idx,
    PTODpsType:$src1Idx,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `,` $src0Idx `,` $src1Idx `:`
      qualified(type($src0)) `,` qualified(type($src1)) `,`
      qualified(type($src0Idx)) `,` qualified(type($src1Idx)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TRandomOp : PTO_TOp<"trandom", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Generate random values into dst tile using key/counter words.";

  let arguments = (ins
    AnySignlessInteger:$key0,
    AnySignlessInteger:$key1,
    AnySignlessInteger:$counter0,
    AnySignlessInteger:$counter1,
    AnySignlessInteger:$counter2,
    AnySignlessInteger:$counter3,
    PTODpsType:$dst,
    DefaultValuedAttr<I32Attr, "10">:$rounds
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $key0 `,` $key1 `,` $counter0 `,` $counter1 `,` $counter2 `,` $counter3
    attr-dict
    `:` type($key0) `,` type($key1) `,` type($counter0) `,` type($counter1) `,` type($counter2) `,` type($counter3) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TExtractOp : PTO_TOp<"textract", [
  AttrSizedOperandSegments,
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Extract sub-tile window from src into dst (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    Index:$indexRow,
    Index:$indexCol,
    PTODpsType:$dst,
    Optional<PTODpsType>:$fp,
    Optional<I64>:$preQuantScalar,
    OptionalAttr<PTO_AccToVecModeAttr>:$accToVecMode,
    DefaultValuedAttr<PTO_ReluPreModeAttr, "::mlir::pto::ReluPreMode::NoRelu">:$reluPreMode
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $indexRow `,` $indexCol
    (`,` $preQuantScalar^)? `:` qualified(type($src)) `,` type($indexRow) `,` type($indexCol)
    (`,` type($preQuantScalar)^)?
    (`fp` $fp^ `:` qualified(type($fp)))? `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      // Align with pto-isa TEXTRACT op classes:
      //   - TEXTRACT_M2LR : MAT -> LEFT/RIGHT (and scaling-like targets) (MTE1)
      //   - TEXTRACT_V2M  : VEC -> MAT       (FIX)
      //   - TEXTRACT_A2M  : ACC -> MAT       (FIX)
      // For pure UB slices (VEC -> VEC), treat as vector pipe.
      auto getASFromType = [](Type ty)
          -> std::optional<::mlir::pto::AddressSpace> {
        if (auto tb = llvm::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = llvm::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace();
          return std::nullopt;
        }
        return std::nullopt;
      };

      auto sOpt = getASFromType(getSrc().getType());
      auto dOpt = getASFromType(getDst().getType());
      if (!sOpt.has_value() || !dOpt.has_value())
        return ::mlir::pto::PIPE::PIPE_V;

      const auto s = sOpt.value();
      const auto d = dOpt.value();

      if (s == ::mlir::pto::AddressSpace::MAT &&
          (d == ::mlir::pto::AddressSpace::LEFT ||
           d == ::mlir::pto::AddressSpace::RIGHT ||
           d == ::mlir::pto::AddressSpace::BIAS ||
           d == ::mlir::pto::AddressSpace::SCALING)) {
        return ::mlir::pto::PIPE::PIPE_MTE1;
      }

      if ((s == ::mlir::pto::AddressSpace::VEC &&
           d == ::mlir::pto::AddressSpace::MAT) ||
          (s == ::mlir::pto::AddressSpace::ACC &&
           (d == ::mlir::pto::AddressSpace::MAT ||
            d == ::mlir::pto::AddressSpace::VEC))) {
        return ::mlir::pto::PIPE::PIPE_FIX;
      }

      if (s == ::mlir::pto::AddressSpace::VEC &&
          d == ::mlir::pto::AddressSpace::VEC) {
        return ::mlir::pto::PIPE::PIPE_V;
      }

      // Default to vector pipe for unmatched combinations.
      return ::mlir::pto::PIPE::PIPE_V;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TInsertOp : PTO_TOp<"tinsert", [
  AttrSizedOperandSegments,
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Insert src sub-tile into dst at (indexRow, indexCol) (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    Index:$indexRow,
    Index:$indexCol,
    PTODpsType:$dst,
    Optional<PTODpsType>:$fp,
    Optional<I64>:$preQuantScalar,
    OptionalAttr<PTO_AccToVecModeAttr>:$accToVecMode,
    DefaultValuedAttr<PTO_ReluPreModeAttr, "::mlir::pto::ReluPreMode::NoRelu">:$reluPreMode,
    OptionalAttr<PTO_TInsertModeAttr>:$tinsertMode
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $indexRow `,` $indexCol `:` qualified(type($src)) `,` type($indexRow) `,` type($indexCol)
    (`fp` $fp^ `:` qualified(type($fp)))?
    (`pre_quant` $preQuantScalar^ `:` type($preQuantScalar))?
    `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    // TINSERT runs on different pipes depending on target/path:
    // - Vec(UB)->Vec(UB): PIPE_V.
    // - A5 Vec(UB)->Mat(L1): PIPE_MTE3 (custom UB->L1 copy path).
    // - Acc->Mat / Acc->Vec (A2/A3/A5 regular path): PIPE_FIX.
    ::mlir::pto::PIPE getPipe() {
      auto isA5Target = [&]() -> bool {
        auto moduleOp = getOperation()->getParentOfType<::mlir::ModuleOp>();
        if (!moduleOp)
          return false;

        if (auto arch =
                moduleOp->getAttrOfType<::mlir::StringAttr>("pto.target_arch")) {
          if (arch.getValue().equals_insensitive("a5"))
            return true;
        }

        if (auto spec =
                moduleOp->getAttrOfType<::mlir::StringAttr>("pto.device-spec")) {
          auto s = spec.getValue();
          if (s.starts_with("Ascend950") || s.starts_with("Ascend910_95"))
            return true;
        }
        return false;
      };

      auto getASFromType = [](Type ty)
          -> std::optional<::mlir::pto::AddressSpace> {
        if (auto tb = ::mlir::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = ::mlir::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace();
          return std::nullopt;
        }
        return std::nullopt;
      };

      auto sOpt = getASFromType(getSrc().getType());
      auto dOpt = getASFromType(getDst().getType());
      if (sOpt.has_value() && dOpt.has_value() &&
          sOpt.value() == ::mlir::pto::AddressSpace::VEC &&
          dOpt.value() == ::mlir::pto::AddressSpace::VEC)
        return ::mlir::pto::PIPE::PIPE_V;
      if (isA5Target() && sOpt.has_value() && dOpt.has_value() &&
          sOpt.value() == ::mlir::pto::AddressSpace::VEC &&
          dOpt.value() == ::mlir::pto::AddressSpace::MAT)
        return ::mlir::pto::PIPE::PIPE_MTE3;

      // Default and Acc->Mat path.
      return ::mlir::pto::PIPE::PIPE_FIX;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TFillPadOp : PTO_TOp<"tfillpad", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Fill padding with compiler-inferred lowering semantics (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    OptionalAttr<PTO_PadValueAttr>:$padValue
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins`  `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      auto getASFromType = [](Type ty)
          -> std::optional<::mlir::pto::AddressSpace> {
        if (auto tb = ::mlir::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = ::mlir::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace();
          return std::nullopt;
        }
        return std::nullopt;
      };

      auto sOpt = getASFromType(getSrc().getType());
      auto dOpt = getASFromType(getDst().getType());
      if (sOpt.has_value() && dOpt.has_value() &&
          sOpt.value() == ::mlir::pto::AddressSpace::MAT &&
          dOpt.value() == ::mlir::pto::AddressSpace::MAT)
        return ::mlir::pto::PIPE::PIPE_MTE1;
      return ::mlir::pto::PIPE::PIPE_V;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGatherOp : PTO_TOp<"tgather", [
  AttrSizedOperandSegments,
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Gather/select elements using index+tmp, scalar-compare+tmp, or a mask pattern (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    Optional<PTODpsType>:$cdst,
    Optional<PTODpsType>:$indices,
    Optional<PTODpsType>:$tmp,
    Optional<ScalarType>:$kValue,
    OptionalAttr<PTO_MaskPatternAttr>:$maskPattern,
    OptionalAttr<StrAttr>:$axis,
    OptionalAttr<PTO_CmpModeAttr>:$cmpMode,
    OptionalAttr<I32Attr>:$offset
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    bool hasCompareForm() { return static_cast<bool>(getCdst()) || static_cast<bool>(getKValue()); }
    bool hasIndexForm() { return static_cast<bool>(getIndices()); }
    bool hasMaskForm() { return static_cast<bool>(getMaskPatternAttr()); }
    bool hasAxis() { return static_cast<bool>(getAxisAttr()); }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return ::mlir::MutableOperandRange(getOperation(), 1, 1 + (getCdst() ? 1 : 0));
    }
  }];
}

def TGatherBOp : PTO_TOp<"tgatherb", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Gather elements using byte offsets (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$offsets,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $offsets `:` qualified(type($src)) `,` type($offsets) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TReshapeOp: PTO_TOp<"treshape", [
  OpPipeInterface,
  Pure,
  ViewLikeOpInterface
]> {
  let summary = "TRESHAPE: Reinterpret a tile buffer view (SSA; aliases src storage)";

  let arguments = (ins
    PTODpsType:$src
  );

  let results = (outs
    PTODpsType:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = "$src attr-dict `:` qualified(type($src)) `->` qualified(type($result))";

  let extraClassDeclaration = [{
    // Keep consistent with pto-isa opPipeList: TRESHAPE is modeled on PIPE_S.
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_S; }
    ::mlir::Value getViewSource() { return getSrc(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWEXPAND TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TScatterOp: PTO_TOp<"tscatter", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSCATTER: Scatter elements of a source tile into a destination tile using indices or a mask pattern.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    Optional<PTODpsType>:$indexes,
    OptionalAttr<StrAttr>:$axis,
    OptionalAttr<PTO_MaskPatternAttr>:$maskPattern
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    bool hasIndexForm() { return static_cast<bool>(getIndexes()); }
    bool hasMaskForm() { return static_cast<bool>(getMaskPatternAttr()); }
    bool hasAxis() { return static_cast<bool>(getAxisAttr()); }
    ::mlir::pto::PIPE getPipe() {
      if (hasMaskForm())
        return ::mlir::pto::PIPE::PIPE_V;

      // NOTE:
      // - On A2/A3, indexed TSCATTER lowers to a scalar UB loop (PIPE_S).
      // - On A5, pto-isa TSCATTER is vector scatter (PIPE_V).
      // We accept either `pto.target_arch=a5` or A5 `pto.device-spec`.
      auto isA5Target = [&]() -> bool {
        auto moduleOp = getOperation()->getParentOfType<::mlir::ModuleOp>();
        if (!moduleOp)
          return false;

        if (auto arch =
                moduleOp->getAttrOfType<::mlir::StringAttr>("pto.target_arch")) {
          if (arch.getValue().equals_insensitive("a5"))
            return true;
        }

        if (auto spec =
                moduleOp->getAttrOfType<::mlir::StringAttr>("pto.device-spec")) {
          auto s = spec.getValue();
          if (s.starts_with("Ascend950") || s.starts_with("Ascend910_95"))
            return true;
        }

        return false;
      };

      return isA5Target() ? ::mlir::pto::PIPE::PIPE_V
                          : ::mlir::pto::PIPE::PIPE_S;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSEL TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TPrintOp: PTO_TOp<"tprint", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TPRINT: Print the contents of a Tile or GlobalTensor for debugging purposes directly from device code.";
  let description = [{
    pto-isa overloads support `TPRINT(src)` and `TPRINT<Format>(src, tmp)`.
    The optional tmp operand is used when printing Mat/Acc tiles through a
    scratch GlobalTensor, while Vec tiles and GlobalTensors can be printed
    without tmp.
  }];

  let arguments = (ins
    PTODpsType:$src,
    Optional<PTODpsType>:$tmp,
    OptionalAttr<PTO_PrintFormatAttr>:$printFormat
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
 }

#endif // MLIR_DIALECT_PTO_IR_PTOTILEVECTORDATAOPS
