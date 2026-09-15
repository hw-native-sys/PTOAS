// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOEnums.h ---------------------------------------------------------===//
//
// C++ scoped-enum mirror of the PTO dialect enums generated in
// PTO/IR/PTOEnums.h.inc.
//
// The Python bindings translation unit (PTOModule.cpp) must be compilable
// online with only the shipped C-API headers (no LLVM/MLIR C++ tree). These
// mirror enums replace the C++ `enum class mlir::pto::*` types that pybind
// otherwise binds. Integer values MUST stay in lock-step with the generated
// C++ enums; lib/CAPI/Dialect/PTO.cpp holds static_asserts that break the
// build on any drift.
//
// `enum class` is intentional rather than a C-style unscoped enum: the only
// consumers are C++ translation units (the Python bindings and the CAPI
// value-drift asserts), no exported C API signature uses these enum types, and
// scoped enums avoid leaking the enumerator names into the enclosing namespace.
//
// CmpMode, Coalesce and MaskPattern are already mirrored in pto-c/Dialect/PTO.h
// and are intentionally NOT redefined here.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_PTO_ENUMS_H
#define MLIR_C_DIALECT_PTO_ENUMS_H

enum class MlirPTOAddressSpace {
  Zero = 0,
  GM = 1,
  MAT = 2,
  LEFT = 3,
  RIGHT = 4,
  ACC = 5,
  VEC = 6,
  BIAS = 7,
  SCALING = 8,
};

enum class MlirPTOFenceScope {
  LocalMemory = 0,
  GM = 1,
  All = 2,
};

enum class MlirPTOLoadCachePolicy {
  Default = 0,
  L2Bypass = 1,
};

enum class MlirPTOBLayout {
  RowMajor = 0,
  ColMajor = 1,
};

enum class MlirPTOSLayout {
  NoneBox = 0,
  RowMajor = 1,
  ColMajor = 2,
};

enum class MlirPTOPadValue {
  Null = 0,
  Zero = 1,
  Max = 2,
  Min = 3,
};

enum class MlirPTOCompactMode {
  Null = 0,
  Normal = 1,
  RowPlusOne = 2,
};

enum class MlirPTORoundMode {
  NONE = 0,
  RINT = 1,
  ROUND = 2,
  FLOOR = 3,
  CEIL = 4,
  TRUNC = 5,
  ODD = 6,
  CAST_RINT = 7,
};

enum class MlirPTODivPrecision {
  Default = 0,
  HighPrecision = 1,
};

enum class MlirPTOExpPrecision {
  Default = 0,
  HighPrecision = 1,
};

enum class MlirPTOLogPrecision {
  Default = 0,
  HighPrecision = 1,
};

enum class MlirPTORecipPrecision {
  Default = 0,
  HighPrecision = 1,
};

enum class MlirPTORemPrecision {
  Default = 0,
  HighPrecision = 1,
};

enum class MlirPTORsqrtPrecision {
  Default = 0,
  HighPrecision = 1,
};

enum class MlirPTOSqrtPrecision {
  Default = 0,
  HighPrecision = 1,
};

enum class MlirPTOFmodPrecision {
  Default = 0,
  HighPrecision = 1,
};

enum class MlirPTOSaturationMode {
  ON = 0,
  OFF = 1,
};

enum class MlirPTOPIPE {
  PIPE_S = 0,
  PIPE_V = 1,
  PIPE_M = 2,
  PIPE_MTE1 = 3,
  PIPE_MTE2 = 4,
  PIPE_MTE3 = 5,
  PIPE_ALL = 6,
  PIPE_MTE4 = 7,
  PIPE_MTE5 = 8,
  PIPE_V2 = 9,
  PIPE_FIX = 10,
  VIRTUAL_PIPE_MTE2_L1A = 11,
  VIRTUAL_PIPE_MTE2_L1B = 12,
  PIPE_NUM = 13,
  PIPE_UNASSIGNED = 99,
};

enum class MlirPTOLayout {
  ND = 0,
  DN = 1,
  NZ = 2,
  MX_A_ZZ = 3,
  MX_B_NN = 4,
};

enum class MlirPTOAccToVecMode {
  SingleModeVec0 = 0,
  SingleModeVec1 = 1,
  DualModeSplitM = 2,
  DualModeSplitN = 3,
};

enum class MlirPTOTInsertMode {
  SPLIT2 = 2,
  SPLIT4 = 3,
};

enum class MlirPTOReluPreMode {
  NoRelu = 0,
  NormalRelu = 1,
  ScalarRelu = 2,
  VectorRelu = 3,
  Pwl = 4,
};

enum class MlirPTOAtomicType {
  AtomicNone = 0,
  AtomicAdd = 1,
};

enum class MlirPTONotifyOp {
  AtomicAdd = 0,
  Set = 1,
};

enum class MlirPTOWaitCmp {
  EQ = 0,
  NE = 1,
  GT = 2,
  GE = 3,
  LT = 4,
  LE = 5,
};

enum class MlirPTOReduceOp {
  Sum = 0,
  Max = 1,
  Min = 2,
};

enum class MlirPTOSyncOpType {
  TLOAD = 0,
  TSTORE_ACC = 1,
  TSTORE_VEC = 2,
  TMOV_M2L = 3,
  TMOV_M2S = 4,
  TMOV_M2B = 5,
  TMOV_M2V = 6,
  TMOV_V2M = 7,
  TMATMUL = 8,
  TVEC = 9,
  TVECWAIT_EVENT = 10,
};

enum class MlirPTOEVENT {
  EVENT_ID0 = 0,
  EVENT_ID1 = 1,
  EVENT_ID2 = 2,
  EVENT_ID3 = 3,
  EVENT_ID4 = 4,
  EVENT_ID5 = 5,
  EVENT_ID6 = 6,
  EVENT_ID7 = 7,
};

enum class MlirPTOQuantType {
  INT8_SYM = 0,
  INT8_ASYM = 1,
  MXFP8 = 2,
  MXFP4_E2M1 = 3,
};

enum class MlirPTOQuantScaleAlg {
  OCP = 0,
  NV = 1,
};

enum class MlirPTOMxGroupAxis {
  Axis0 = 0,
  Axis1 = 1,
};

enum class MlirPTOVecStoreMode {
  ND = 0,
  NZ = 1,
};

#endif // MLIR_C_DIALECT_PTO_ENUMS_H
