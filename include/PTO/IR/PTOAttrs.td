// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOAttrs.td - PTO dialect attributes definitions ----*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// This is the definition file for PTO dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTOATTRS
#define MLIR_DIALECT_PTO_IR_PTOATTRS

include "PTO/IR/PTODialect.td"

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// PTO Enums and Attributes.
//===----------------------------------------------------------------------===//

class PTO_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::pto";
}

class PTO_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<PTO_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Address Space
//===----------------------------------------------------------------------===//

def PTO_AddressSpace_Default : I32EnumAttrCase<"Zero", 0, "zero">;
def PTO_AddressSpace_GM : I32EnumAttrCase<"GM", 1, "gm">;
def PTO_AddressSpace_MAT : I32EnumAttrCase<"MAT", 2, "mat">;
def PTO_AddressSpace_LEFT : I32EnumAttrCase<"LEFT", 3, "left">;
def PTO_AddressSpace_RIGHT : I32EnumAttrCase<"RIGHT", 4, "right">;
def PTO_AddressSpace_ACC : I32EnumAttrCase<"ACC", 5, "acc">;
def PTO_AddressSpace_VEC : I32EnumAttrCase<"VEC", 6, "vec">;
def PTO_AddressSpace_BIAS : I32EnumAttrCase<"BIAS", 7, "bias">;
def PTO_AddressSpace_SCALING : I32EnumAttrCase<"SCALING", 8, "scaling">;

def PTO_AddressSpaceEnum : PTO_I32Enum<
  "AddressSpace", "PTO Address Space", [
    PTO_AddressSpace_Default,
    PTO_AddressSpace_GM,
    PTO_AddressSpace_MAT,
    PTO_AddressSpace_LEFT,
    PTO_AddressSpace_RIGHT,
    PTO_AddressSpace_ACC,
    PTO_AddressSpace_VEC,
    PTO_AddressSpace_BIAS,
    PTO_AddressSpace_SCALING
  ]>;

def PTO_AddressSpaceAttr : PTO_Attr<"AddressSpace", "address_space"> {
  let parameters = (ins EnumParameter<PTO_AddressSpaceEnum>:$address_space);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    PTO address space mapping attribute. Maps to GM, L1, L0A, L0B, L0C and UB.
  }];
}

//===----------------------------------------------------------------------===//
// Signedness
//===----------------------------------------------------------------------===//

def PTO_Signedness_Signed : I32EnumAttrCase<"Signed", 0, "signed">;
def PTO_Signedness_Unsigned : I32EnumAttrCase<"Unsigned", 1, "unsigned">;

def PTO_SignednessEnum : PTO_I32Enum<
  "Signedness", "PTO integer signedness", [
    PTO_Signedness_Signed,
    PTO_Signedness_Unsigned
  ]>;

def PTO_SignednessAttr : PTO_Attr<"Signedness", "signedness"> {
  let parameters = (ins EnumParameter<PTO_SignednessEnum>:$value);
  let assemblyFormat = "`<` params `>`";
  let summary = "Integer signedness control for semantic ops";
}

//===----------------------------------------------------------------------===//
// SIMT GM load/store L1 and L2 cache controls
//===----------------------------------------------------------------------===//

def PTO_LdL2CacheEnum : PTO_I32Enum<
  "LdL2Cache", "PTO load L2 cache control", [
    I32EnumAttrCase<"NMFV", 0, "nmfv">,
    I32EnumAttrCase<"NMLV", 1, "nmlv">,
    I32EnumAttrCase<"NMPRS", 2, "nmprs">,
    I32EnumAttrCase<"NMPREF", 3, "nmpref">,
    I32EnumAttrCase<"NAKEEP", 4, "nakeep">,
    I32EnumAttrCase<"NACLEAN", 5, "naclean">,
    I32EnumAttrCase<"NADROP", 6, "nadrop">,
    I32EnumAttrCase<"IDSFV", 8, "idsfv">,
    I32EnumAttrCase<"IDSLV", 9, "idslv">,
    I32EnumAttrCase<"IDSPRS", 10, "idsprs">,
    I32EnumAttrCase<"IDSPREF", 11, "idspref">,
    I32EnumAttrCase<"EXFV", 12, "exfv">,
    I32EnumAttrCase<"EXLV", 13, "exlv">,
    I32EnumAttrCase<"EXPRS", 14, "exprs">,
    I32EnumAttrCase<"EXPREF", 15, "expref">
  ]>;

def PTO_LdL2CacheAttr
    : EnumAttr<PTO_Dialect, PTO_LdL2CacheEnum, "ld_l2cache"> {
  let assemblyFormat = "`<` params `>`";
  let summary = "load L2 cache control";
}

def PTO_StL2CacheEnum : PTO_I32Enum<
  "StL2Cache", "PTO store/atomic L2 cache control", [
    I32EnumAttrCase<"NMFV", 0, "nmfv">,
    I32EnumAttrCase<"NMLV", 1, "nmlv">,
    I32EnumAttrCase<"NMPRS", 2, "nmprs">,
    I32EnumAttrCase<"NMRED", 3, "nmred">,
    I32EnumAttrCase<"NACI", 4, "naci">,
    I32EnumAttrCase<"NAPW", 5, "napw">,
    I32EnumAttrCase<"NAPI", 6, "napi">,
    I32EnumAttrCase<"NARED", 7, "nared">,
    I32EnumAttrCase<"WBHFV", 8, "wbhfv">,
    I32EnumAttrCase<"WBHLV", 9, "wbhlv">,
    I32EnumAttrCase<"WBHPRS", 10, "wbhprs">,
    I32EnumAttrCase<"WBHRED", 11, "wbhred">,
    I32EnumAttrCase<"WTSFV", 12, "wtsfv">,
    I32EnumAttrCase<"WTSLV", 13, "wtslv">,
    I32EnumAttrCase<"WTSPRS", 14, "wtsprs">,
    I32EnumAttrCase<"WTSRED", 15, "wtsred">
  ]>;

def PTO_StL2CacheAttr
    : EnumAttr<PTO_Dialect, PTO_StL2CacheEnum, "st_l2cache"> {
  let assemblyFormat = "`<` params `>`";
  let summary = "store and atomic L2 cache control";
}

def PTO_L1CacheEnum : PTO_I32Enum<
  "L1Cache", "PTO L1 cache control", [
    I32EnumAttrCase<"Uncache", 0, "uncache">,
    I32EnumAttrCase<"Cache", 1, "cache">
  ]>;

def PTO_L1CacheAttr
    : EnumAttr<PTO_Dialect, PTO_L1CacheEnum, "l1cache"> {
  let assemblyFormat = "`<` params `>`";
  let summary = "L1 cache control";
}

//===----------------------------------------------------------------------===//
// SIMT conversion controls
//===----------------------------------------------------------------------===//

// CANN ROUND semantics confirmed from installed beta.1 headers:
//   R = round to nearest, ties to even
//   A = round away from zero
//   F = round toward minus infinity
//   C = round toward plus infinity
//   Z = round toward zero
//   O = round to odd
//   H = CAST_CEIL (special cast mode observed on H8/HiF8 conversion paths)
def PTO_RoundingEnum : PTO_I32Enum<
  "Rounding", "PTO SIMT conversion rounding selector", [
    // nearest-even
    I32EnumAttrCase<"R", 0, "r">,
    // away-zero
    I32EnumAttrCase<"A", 1, "a">,
    // floor / toward minus infinity
    I32EnumAttrCase<"F", 2, "f">,
    // ceil / toward plus infinity
    I32EnumAttrCase<"C", 3, "c">,
    // trunc / toward zero
    I32EnumAttrCase<"Z", 4, "z">,
    // round to odd
    I32EnumAttrCase<"O", 5, "o">,
    // CAST_CEIL on the confirmed H8/HiF8 conversion slice
    I32EnumAttrCase<"H", 6, "h">
  ]>;

def PTO_RoundingAttr : EnumAttr<PTO_Dialect, PTO_RoundingEnum, "rounding"> {
  let assemblyFormat = "`<` params `>`";
  let summary = "SIMT conversion rounding selector";
}

def PTO_SaturationEnum : PTO_I32Enum<
  "Saturation", "PTO SIMT conversion saturation selector", [
    I32EnumAttrCase<"Disable", 0, "nosat">,
    I32EnumAttrCase<"Enable", 1, "sat">
  ]>;

def PTO_SaturationAttr
    : EnumAttr<PTO_Dialect, PTO_SaturationEnum, "saturation"> {
  let assemblyFormat = "`<` params `>`";
  let summary = "SIMT conversion saturation selector";
}

//===----------------------------------------------------------------------===//
// Pipe
//===----------------------------------------------------------------------===//

def PTO_PIPE_S : I32EnumAttrCase<"PIPE_S", 0>;
def PTO_PIPE_V : I32EnumAttrCase<"PIPE_V", 1>;
def PTO_PIPE_M : I32EnumAttrCase<"PIPE_M", 2>;
def PTO_PIPE_MTE1 : I32EnumAttrCase<"PIPE_MTE1", 3>;
def PTO_PIPE_MTE2 : I32EnumAttrCase<"PIPE_MTE2", 4>;
def PTO_PIPE_MTE3 : I32EnumAttrCase<"PIPE_MTE3", 5>;
def PTO_PIPE_ALL : I32EnumAttrCase<"PIPE_ALL", 6>;
def PTO_PIPE_MTE4 : I32EnumAttrCase<"PIPE_MTE4", 7>;
def PTO_PIPE_MTE5 : I32EnumAttrCase<"PIPE_MTE5", 8>;
def PTO_PIPE_V2 : I32EnumAttrCase<"PIPE_V2", 9>;
def PTO_PIPE_FIX : I32EnumAttrCase<"PIPE_FIX", 10>;
def PTO_VIRTUAL_PIPE_MTE2_L1A : I32EnumAttrCase<"VIRTUAL_PIPE_MTE2_L1A", 11>;
def PTO_VIRTUAL_PIPE_MTE2_L1B : I32EnumAttrCase<"VIRTUAL_PIPE_MTE2_L1B", 12>;
def PTO_PIPE_NUM : I32EnumAttrCase<"PIPE_NUM", 13>;
def PTO_PIPE_UNASSIGNED : I32EnumAttrCase<"PIPE_UNASSIGNED", 99>;

def PTO_PipeEnum : PTO_I32Enum<
  "PIPE", "PTO Op Pipe", [
    PTO_PIPE_S,
    PTO_PIPE_V,
    PTO_PIPE_M,
    PTO_PIPE_MTE1,
    PTO_PIPE_MTE2,
    PTO_PIPE_MTE3,
    PTO_PIPE_ALL,
    PTO_PIPE_MTE4,
    PTO_PIPE_MTE5,
    PTO_PIPE_V2,
    PTO_PIPE_FIX,
    PTO_VIRTUAL_PIPE_MTE2_L1A,
    PTO_VIRTUAL_PIPE_MTE2_L1B,
    PTO_PIPE_NUM,
    PTO_PIPE_UNASSIGNED
  ]>;

def PTO_PipeAttr : PTO_Attr<"Pipe", "pipe"> {
  let parameters = (ins EnumParameter<PTO_PipeEnum>:$pipe);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    PTO Op pipe attribute.
  }];
}

//===----------------------------------------------------------------------===//
// MemBar
//===----------------------------------------------------------------------===//

def PTO_MEMBAR_VV_ALL : I32EnumAttrCase<"VV_ALL", 0>;
def PTO_MEMBAR_VST_VLD : I32EnumAttrCase<"VST_VLD", 1>;
def PTO_MEMBAR_VLD_VST : I32EnumAttrCase<"VLD_VST", 2>;
def PTO_MEMBAR_VST_VST : I32EnumAttrCase<"VST_VST", 3>;
def PTO_MEMBAR_VS_ALL : I32EnumAttrCase<"VS_ALL", 4>;
def PTO_MEMBAR_VST_LD : I32EnumAttrCase<"VST_LD", 5>;
def PTO_MEMBAR_VLD_ST : I32EnumAttrCase<"VLD_ST", 6>;
def PTO_MEMBAR_VST_ST : I32EnumAttrCase<"VST_ST", 7>;
def PTO_MEMBAR_SV_ALL : I32EnumAttrCase<"SV_ALL", 8>;
def PTO_MEMBAR_ST_VLD : I32EnumAttrCase<"ST_VLD", 9>;
def PTO_MEMBAR_LD_VST : I32EnumAttrCase<"LD_VST", 10>;
def PTO_MEMBAR_ST_VST : I32EnumAttrCase<"ST_VST", 11>;
def PTO_MEMBAR_SS_ALL : I32EnumAttrCase<"SS_ALL", 12>;
def PTO_MEMBAR_ST_LD : I32EnumAttrCase<"ST_LD", 13>;
def PTO_MEMBAR_LD_ST : I32EnumAttrCase<"LD_ST", 14>;
def PTO_MEMBAR_ST_ST : I32EnumAttrCase<"ST_ST", 15>;

def PTO_MemBarEnum : PTO_I32Enum<
  "MemBarKind", "PTO low-level memory barrier kind", [
    PTO_MEMBAR_VV_ALL,
    PTO_MEMBAR_VST_VLD,
    PTO_MEMBAR_VLD_VST,
    PTO_MEMBAR_VST_VST,
    PTO_MEMBAR_VS_ALL,
    PTO_MEMBAR_VST_LD,
    PTO_MEMBAR_VLD_ST,
    PTO_MEMBAR_VST_ST,
    PTO_MEMBAR_SV_ALL,
    PTO_MEMBAR_ST_VLD,
    PTO_MEMBAR_LD_VST,
    PTO_MEMBAR_ST_VST,
    PTO_MEMBAR_SS_ALL,
    PTO_MEMBAR_ST_LD,
    PTO_MEMBAR_LD_ST,
    PTO_MEMBAR_ST_ST
  ]>;

def PTO_MemBarAttr : PTO_Attr<"MemBar", "membar"> {
  let parameters = (ins EnumParameter<PTO_MemBarEnum>:$kind);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Low-level memory barrier kind for VPTO `pto.mem_bar`.
  }];
}

//===----------------------------------------------------------------------===//
// Sync Op Type (High Level Abstraction)
//===----------------------------------------------------------------------===//

def PTO_SYNC_TLOAD        : I32EnumAttrCase<"TLOAD", 0>;
def PTO_SYNC_TSTORE_ACC   : I32EnumAttrCase<"TSTORE_ACC", 1>;
def PTO_SYNC_TSTORE_VEC   : I32EnumAttrCase<"TSTORE_VEC", 2>;
def PTO_SYNC_TMOV_M2L     : I32EnumAttrCase<"TMOV_M2L", 3>;
def PTO_SYNC_TMOV_M2S     : I32EnumAttrCase<"TMOV_M2S", 4>;
def PTO_SYNC_TMOV_M2B     : I32EnumAttrCase<"TMOV_M2B", 5>;
def PTO_SYNC_TMOV_M2V     : I32EnumAttrCase<"TMOV_M2V", 6>;
def PTO_SYNC_TMOV_V2M     : I32EnumAttrCase<"TMOV_V2M", 7>;
def PTO_SYNC_TMATMUL      : I32EnumAttrCase<"TMATMUL", 8>;
def PTO_SYNC_TVEC         : I32EnumAttrCase<"TVEC", 9>;
def PTO_SYNC_TVECWAIT     : I32EnumAttrCase<"TVECWAIT_EVENT", 10>;

def PTO_SyncOpTypeEnum : PTO_I32Enum<
  "SyncOpType", "PTO High Level Sync Op Type", [
    PTO_SYNC_TLOAD,
    PTO_SYNC_TSTORE_ACC,
    PTO_SYNC_TSTORE_VEC,
    PTO_SYNC_TMOV_M2L,
    PTO_SYNC_TMOV_M2S,
    PTO_SYNC_TMOV_M2B,
    PTO_SYNC_TMOV_M2V,
    PTO_SYNC_TMOV_V2M,
    PTO_SYNC_TMATMUL,
    PTO_SYNC_TVEC,
    PTO_SYNC_TVECWAIT
  ]>;

// Canonical sync endpoint type attribute used by record_event/wait_event.
def PTO_PipeEventTypeAttr : PTO_Attr<"PipeEventType", "pipe_event_type"> {
  let parameters = (ins EnumParameter<PTO_SyncOpTypeEnum>:$op_type);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    High-level synchronization endpoint type. Lowering pass maps it to hardware pipes.
  }];
}

// Legacy alias kept for compatibility with older code.
def PTO_SyncOpTypeAttr : PTO_Attr<"SyncOpType", "sync_op_type"> {
  let parameters = (ins EnumParameter<PTO_SyncOpTypeEnum>:$op_type);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Deprecated alias of pipe_event_type. Prefer #pto.pipe_event_type<...>.
  }];
}

//===----------------------------------------------------------------------===//
// SYNCALL enums
//===----------------------------------------------------------------------===//

def PTO_SYNCALL_MODE_HARD : I32EnumAttrCase<"Hard", 0, "hard">;
def PTO_SYNCALL_MODE_SOFT : I32EnumAttrCase<"Soft", 1, "soft">;

def PTO_SyncAllModeEnum : PTO_I32Enum<
  "SyncAllMode", "PTO SYNCALL mode", [
    PTO_SYNCALL_MODE_HARD,
    PTO_SYNCALL_MODE_SOFT
  ]>;

def PTO_SyncAllModeAttr : PTO_Attr<"SyncAllMode", "sync_all_mode"> {
  let parameters = (ins EnumParameter<PTO_SyncAllModeEnum>:$value);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    SYNCALL mode: hard barrier or workspace-backed soft barrier.
  }];
}

def PTO_SYNC_CORE_AIV_ONLY : I32EnumAttrCase<"AIVOnly", 0, "aiv_only">;
def PTO_SYNC_CORE_AIC_ONLY : I32EnumAttrCase<"AICOnly", 1, "aic_only">;
def PTO_SYNC_CORE_MIX : I32EnumAttrCase<"Mix", 2, "mix">;

def PTO_SyncCoreTypeEnum : PTO_I32Enum<
  "SyncCoreType", "PTO SYNCALL core participation kind", [
    PTO_SYNC_CORE_AIV_ONLY,
    PTO_SYNC_CORE_AIC_ONLY,
    PTO_SYNC_CORE_MIX
  ]>;

def PTO_SyncCoreTypeAttr : PTO_Attr<"SyncCoreType", "sync_core_type"> {
  let parameters = (ins EnumParameter<PTO_SyncCoreTypeEnum>:$value);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    SYNCALL core participation kind: AIV only, AIC only, or mixed AIC/AIV.
  }];
}

//===----------------------------------------------------------------------===//
// Layout (GlobalTensor layout inference)
//===----------------------------------------------------------------------===//

def PTO_Layout_ND : I32EnumAttrCase<"ND", 0, "nd">;
def PTO_Layout_DN : I32EnumAttrCase<"DN", 1, "dn">;
def PTO_Layout_NZ : I32EnumAttrCase<"NZ", 2, "nz">;

def PTO_LayoutEnum : PTO_I32Enum<
  "Layout", "Global tensor layout (row/col/fractal)", [
    PTO_Layout_ND,
    PTO_Layout_DN,
    PTO_Layout_NZ
  ]>;

def PTO_LayoutAttr : PTO_Attr<"Layout", "layout"> {
  let parameters = (ins EnumParameter<PTO_LayoutEnum>:$layout);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Layout inferred from shape/stride for GlobalTensor:
      ND (row-major), DN (col-major), NZ (fractal).
  }];
}

//===----------------------------------------------------------------------===//
// Function and Module Core Type
//===----------------------------------------------------------------------===//

def PTO_FunctionKernelKindCube : I32EnumAttrCase<"Cube", 0, "cube">;
def PTO_FunctionKernelKindVector : I32EnumAttrCase<"Vector", 1, "vector">;

def PTO_FunctionKernelKindEnum : PTO_I32Enum<
  "FunctionKernelKind", "PTO Function Kernel Kind", [
    PTO_FunctionKernelKindCube,
    PTO_FunctionKernelKindVector
  ]>;

def PTO_FunctionKernelKindAttr : PTO_Attr<"FunctionKernelKind", "kernel_kind"> {
  let parameters = (ins EnumParameter<PTO_FunctionKernelKindEnum>:$kernelKind);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Stable frontend-visible function attribute for cube or vector kernel kind.
  }];
}

def PTO_AIC : I32EnumAttrCase<"AIC", 1>;
def PTO_AIV : I32EnumAttrCase<"AIV", 2>;
def PTO_MIX : I32EnumAttrCase<"MIX", 3>;
def PTO_AIC_OR_AIV : I32EnumAttrCase<"AIC_OR_AIV", 4>;

def PTO_TFuncCoreTypeEnum : PTO_I32Enum<
  "TFuncCoreType", "PTO Function Core Type", [
    PTO_AIC,
    PTO_AIV,
    PTO_MIX,
    PTO_AIC_OR_AIV
  ]>;

def PTO_TFuncCoreTypeAttr : PTO_Attr<"TFuncCoreType", "func_core_type"> {
  let parameters = (ins EnumParameter<PTO_TFuncCoreTypeEnum>:$funcCoreType);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    PTO function core type attribute.
  }];
}

def PTO_TModuleCoreTypeEnum : PTO_I32Enum<
  "TModuleCoreType", "PTO Module Core Type", [
    PTO_AIC,
    PTO_AIV,
    PTO_MIX
  ]>;

def PTO_TModuleCoreTypeAttr : PTO_Attr<"TModuleCoreType", "module_core_type"> {
  let parameters = (ins EnumParameter<PTO_TModuleCoreTypeEnum>:$moduleCoreType);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    PTO module core type attribute.

    If all of the functions within the module has `AIV` func core type , the
    module core type is `AIV`.

    If all of the functions within the module has `AIC` func core type , the
    module core type is `AIC`.

    Otherwise, the module core type is `MIX`.
  }];
}

//===----------------------------------------------------------------------===//
// Event ID
//===----------------------------------------------------------------------===//

def PTO_EVENT_ID0 : I32EnumAttrCase<"EVENT_ID0", 0>;
def PTO_EVENT_ID1 : I32EnumAttrCase<"EVENT_ID1", 1>;
def PTO_EVENT_ID2 : I32EnumAttrCase<"EVENT_ID2", 2>;
def PTO_EVENT_ID3 : I32EnumAttrCase<"EVENT_ID3", 3>;
def PTO_EVENT_ID4 : I32EnumAttrCase<"EVENT_ID4", 4>;
def PTO_EVENT_ID5 : I32EnumAttrCase<"EVENT_ID5", 5>;
def PTO_EVENT_ID6 : I32EnumAttrCase<"EVENT_ID6", 6>;
def PTO_EVENT_ID7 : I32EnumAttrCase<"EVENT_ID7", 7>;

def PTO_EventEnum : PTO_I32Enum<
  "EVENT", "Event ID for Synchronization", [
    PTO_EVENT_ID0,
    PTO_EVENT_ID1,
    PTO_EVENT_ID2,
    PTO_EVENT_ID3,
    PTO_EVENT_ID4,
    PTO_EVENT_ID5,
    PTO_EVENT_ID6,
    PTO_EVENT_ID7
  ]>;

def PTO_EventAttr : PTO_Attr<"Event", "event"> {
  let parameters = (ins EnumParameter<PTO_EventEnum>:$event);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    PTO event attribute for synchronization.
  }];
}

//===----------------------------------------------------------------------===//
// Plan Memory Mode
//===----------------------------------------------------------------------===//

def PTO_LOCALMEMPLAN : I32EnumAttrCase<"LOCAL_MEM_PLAN", 0>;
def PTO_GLOBALWORKSPACEPLAN : I32EnumAttrCase<"GLOBAL_WORKSPACE_PLAN", 1>;

def PTO_MemPlanModeEnum : PTO_I32Enum<
  "MemPlanMode", "Mem Plan Mode", [
    PTO_LOCALMEMPLAN,
    PTO_GLOBALWORKSPACEPLAN
]>;

//===----------------------------------------------------------------------===//
// Pad Mode for LoadOp
//===----------------------------------------------------------------------===//

// No padding
def PTO_PadNull : I32EnumAttrCase<"PadNull", 0>;
// Pad using the first element
def PTO_PadFirstElem : I32EnumAttrCase<"PadFirstElem", 1>;
// Pad using specified pad value
def PTO_PadValue : I32EnumAttrCase<"PadValue", 2>;

def PTO_PadModeEnum : PTO_I32Enum<
  "PadMode", "Pad Mode for LoadOp", [
    PTO_PadNull,
    PTO_PadFirstElem,
    PTO_PadValue
  ]>;

def PTO_PadModeAttr : PTO_Attr<"PadMode", "padmode"> {
  let parameters = (ins EnumParameter<PTO_PadModeEnum>:$padmode);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    PTO pad mode attribute.
  }];
}

def PTO_VFAttr : PTO_Attr<"VectorFunction", "vector_function"> {
  let description = [{PTO vector function attribute}];
}
//===----------------------------------------------------------------------===//
// CmpMode
//===----------------------------------------------------------------------===//

def PTO_CmpModeEnum : PTO_I32Enum<
  "CmpMode", "PTO compare mode", [
    I32EnumAttrCase<"EQ", 0, "eq">,
    I32EnumAttrCase<"NE", 1, "ne">,
    I32EnumAttrCase<"LT", 2, "lt">,
    I32EnumAttrCase<"LE", 3, "le">,
    I32EnumAttrCase<"GT", 4, "gt">,
    I32EnumAttrCase<"GE", 5, "ge">
  ]>;

def PTO_CmpModeAttr : EnumAttr<PTO_Dialect, PTO_CmpModeEnum, "cmp"> {
  let summary = "cmp mode attribute";
}
//===----------------------------------------------------------------------===//
// RoundMode
//===----------------------------------------------------------------------===//

def PTO_RoundModeEnum : PTO_I32Enum<
  "RoundMode", "PTO rounding mode", [
    I32EnumAttrCase<"NONE",      0>,
    I32EnumAttrCase<"RINT",      1>,
    I32EnumAttrCase<"ROUND",     2>,
    I32EnumAttrCase<"FLOOR",     3>,
    I32EnumAttrCase<"CEIL",      4>,
    I32EnumAttrCase<"TRUNC",     5>,
    I32EnumAttrCase<"ODD",       6>,
    I32EnumAttrCase<"CAST_RINT", 7>
  ]>;

def PTO_RoundModeAttr : EnumAttr<PTO_Dialect, PTO_RoundModeEnum, "round_mode"> {
  let summary = "rounding mode attribute";
}

//===----------------------------------------------------------------------===//
// MAD semantic controls
//===----------------------------------------------------------------------===//

def PTO_MadUnitFlagModeEnum : PTO_I32Enum<
  "MadUnitFlagMode", "PTO MAD producer unit-flag mode", [
    I32EnumAttrCase<"CheckOnly", 0, "check_only">,
    I32EnumAttrCase<"CheckAndSet", 1, "check_and_set">
  ]>;

def PTO_MadUnitFlagModeAttr
    : EnumAttr<PTO_Dialect, PTO_MadUnitFlagModeEnum, "mad_unit_flag_mode"> {
  let summary = "MAD producer unit-flag mode attribute";
}

def PTO_Tf32ModeEnum : PTO_I32Enum<
  "Tf32Mode", "PTO MAD TF32 rounding mode", [
    I32EnumAttrCase<"RoundEven", 0, "round_even">,
    I32EnumAttrCase<"RoundAway", 1, "round_away">
  ]>;

def PTO_Tf32ModeAttr : EnumAttr<PTO_Dialect, PTO_Tf32ModeEnum, "tf32_mode"> {
  let summary = "MAD TF32 rounding mode attribute";
}

def PTO_MadSatModeEnum : PTO_I32Enum<
  "MadSatMode", "PTO MAD floating exceptional-value saturation mode", [
    I32EnumAttrCase<"Sat", 0, "sat">,
    I32EnumAttrCase<"NoSat", 1, "nosat">
  ]>;

def PTO_MadSatModeAttr
    : EnumAttr<PTO_Dialect, PTO_MadSatModeEnum, "mad_sat_mode"> {
  let summary = "MAD floating exceptional-value saturation mode attribute";
}

//===----------------------------------------------------------------------===//
// SaturationMode
//===----------------------------------------------------------------------===//

def PTO_SaturationModeEnum : PTO_I32Enum<
  "SaturationMode", "PTO saturation mode", [
    I32EnumAttrCase<"ON",  0>,
    I32EnumAttrCase<"OFF", 1>
  ]>;

def PTO_SaturationModeAttr : EnumAttr<PTO_Dialect, PTO_SaturationModeEnum, "saturation_mode"> {
  let summary = "saturation mode attribute";
}
//===----------------------------------------------------------------------===//
// TDIV template controls
//===----------------------------------------------------------------------===//

def PTO_DivPrecisionEnum : PTO_I32Enum<
  "DivPrecision", "PTO TDIV precision mode", [
    I32EnumAttrCase<"Default", 0, "default">,
    I32EnumAttrCase<"HighPrecision", 1, "high_precision">
  ]>;

def PTO_DivPrecisionAttr : EnumAttr<PTO_Dialect, PTO_DivPrecisionEnum, "div_precision"> {
  let summary = "TDIV precision mode attribute";
}
//===----------------------------------------------------------------------===//
// TEXP template controls
//===----------------------------------------------------------------------===//

def PTO_ExpPrecisionEnum : PTO_I32Enum<
  "ExpPrecision", "PTO TEXP precision mode", [
    I32EnumAttrCase<"Default", 0, "default">,
    I32EnumAttrCase<"HighPrecision", 1, "high_precision">
  ]>;

def PTO_ExpPrecisionAttr : EnumAttr<PTO_Dialect, PTO_ExpPrecisionEnum, "exp_precision"> {
  let summary = "TEXP precision mode attribute";
}
//===----------------------------------------------------------------------===//
// TLOG template controls
//===----------------------------------------------------------------------===//

def PTO_LogPrecisionEnum : PTO_I32Enum<
  "LogPrecision", "PTO TLOG precision mode", [
    I32EnumAttrCase<"Default", 0, "default">,
    I32EnumAttrCase<"HighPrecision", 1, "high_precision">
  ]>;

def PTO_LogPrecisionAttr : EnumAttr<PTO_Dialect, PTO_LogPrecisionEnum, "log_precision"> {
  let summary = "TLOG precision mode attribute";
}
//===----------------------------------------------------------------------===//
// TRECIP template controls
//===----------------------------------------------------------------------===//

def PTO_RecipPrecisionEnum : PTO_I32Enum<
  "RecipPrecision", "PTO TRECIP precision mode", [
    I32EnumAttrCase<"Default", 0, "default">,
    I32EnumAttrCase<"HighPrecision", 1, "high_precision">
  ]>;

def PTO_RecipPrecisionAttr : EnumAttr<PTO_Dialect, PTO_RecipPrecisionEnum, "recip_precision"> {
  let summary = "TRECIP precision mode attribute";
}
//===----------------------------------------------------------------------===//
// TREM template controls
//===----------------------------------------------------------------------===//

def PTO_RemPrecisionEnum : PTO_I32Enum<
  "RemPrecision", "PTO TREM precision mode", [
    I32EnumAttrCase<"Default", 0, "default">,
    I32EnumAttrCase<"HighPrecision", 1, "high_precision">
  ]>;

def PTO_RemPrecisionAttr : EnumAttr<PTO_Dialect, PTO_RemPrecisionEnum, "rem_precision"> {
  let summary = "TREM precision mode attribute";
}
//===----------------------------------------------------------------------===//
// TRSQRT template controls
//===----------------------------------------------------------------------===//

def PTO_RsqrtPrecisionEnum : PTO_I32Enum<
  "RsqrtPrecision", "PTO TRSQRT precision mode", [
    I32EnumAttrCase<"Default", 0, "default">,
    I32EnumAttrCase<"HighPrecision", 1, "high_precision">
  ]>;

def PTO_RsqrtPrecisionAttr : EnumAttr<PTO_Dialect, PTO_RsqrtPrecisionEnum, "rsqrt_precision"> {
  let summary = "TRSQRT precision mode attribute";
}
//===----------------------------------------------------------------------===//
// TSQRT template controls
//===----------------------------------------------------------------------===//

def PTO_SqrtPrecisionEnum : PTO_I32Enum<
  "SqrtPrecision", "PTO TSQRT precision mode", [
    I32EnumAttrCase<"Default", 0, "default">,
    I32EnumAttrCase<"HighPrecision", 1, "high_precision">
  ]>;

def PTO_SqrtPrecisionAttr : EnumAttr<PTO_Dialect, PTO_SqrtPrecisionEnum, "sqrt_precision"> {
  let summary = "TSQRT precision mode attribute";
}
//===----------------------------------------------------------------------===//
// TFMOD template controls
//===----------------------------------------------------------------------===//

def PTO_FmodPrecisionEnum : PTO_I32Enum<
  "FmodPrecision", "PTO TFMOD precision mode", [
    I32EnumAttrCase<"Default", 0, "default">,
    I32EnumAttrCase<"HighPrecision", 1, "high_precision">
  ]>;

def PTO_FmodPrecisionAttr : EnumAttr<PTO_Dialect, PTO_FmodPrecisionEnum, "fmod_precision"> {
  let summary = "TFMOD precision mode attribute";
}
//===----------------------------------------------------------------------===//
// TStore template controls
//===----------------------------------------------------------------------===//

def PTO_STPhaseEnum : PTO_I32Enum<
  "STPhase", "PTO TSTORE store phase", [
    I32EnumAttrCase<"Unspecified", 0, "unspecified">,
    I32EnumAttrCase<"Partial", 2, "partial">,
    I32EnumAttrCase<"Final", 3, "final">
  ]>;

def PTO_STPhaseAttr : EnumAttr<PTO_Dialect, PTO_STPhaseEnum, "st_phase"> {
  let summary = "TSTORE phase attribute";
}

//===----------------------------------------------------------------------===//
// TMatmul accumulation phase
//===----------------------------------------------------------------------===//

// Mirrors the device-side `AccPhase` template parameter on TMATMUL /
// TMATMUL_ACC. `Unspecified` keeps backward compatibility: the lowering emits
// the call without any template argument (i.e. `TMATMUL(...)`), which is also
// the only form generated before this attribute existed. `Partial` / `Final`
// instruct lowering to emit the templated form (`TMATMUL<AccPhase::Final>(...)`)
// so a K-loop scheduler upstream can drive last-iteration vs. mid-iteration
// behaviour without resorting to per-call switch statements at codegen time.
def PTO_AccPhaseEnum : PTO_I32Enum<
  "AccPhase", "PTO matmul accumulation phase", [
    I32EnumAttrCase<"Unspecified", 0, "unspecified">,
    I32EnumAttrCase<"Partial", 2, "partial">,
    I32EnumAttrCase<"Final", 3, "final">
  ]>;

def PTO_AccPhaseAttr : EnumAttr<PTO_Dialect, PTO_AccPhaseEnum, "acc_phase"> {
  let summary = "TMATMUL / TMATMUL_ACC accumulation phase attribute";
}

def PTO_AtomicTypeEnum : PTO_I32Enum<
  "AtomicType", "PTO TSTORE atomic mode", [
    I32EnumAttrCase<"AtomicNone", 0, "atomic_none">,
    I32EnumAttrCase<"AtomicAdd", 1, "atomic_add">
  ]>;

def PTO_AtomicTypeAttr : EnumAttr<PTO_Dialect, PTO_AtomicTypeEnum, "atomic_type"> {
  let summary = "TSTORE atomic type attribute";
}

def PTO_NotifyOpEnum : PTO_I32Enum<
  "NotifyOp", "PTO communication notify op", [
    I32EnumAttrCase<"AtomicAdd", 0, "atomic_add">,
    I32EnumAttrCase<"Set", 1, "set">
  ]>;

def PTO_NotifyOpAttr : EnumAttr<PTO_Dialect, PTO_NotifyOpEnum, "notify_op"> {
  let summary = "communication notify operation attribute";
}

def PTO_WaitCmpEnum : PTO_I32Enum<
  "WaitCmp", "PTO communication wait/test compare", [
    I32EnumAttrCase<"EQ", 0, "eq">,
    I32EnumAttrCase<"NE", 1, "ne">,
    I32EnumAttrCase<"GT", 2, "gt">,
    I32EnumAttrCase<"GE", 3, "ge">,
    I32EnumAttrCase<"LT", 4, "lt">,
    I32EnumAttrCase<"LE", 5, "le">
  ]>;

def PTO_WaitCmpAttr : EnumAttr<PTO_Dialect, PTO_WaitCmpEnum, "wait_cmp"> {
  let summary = "communication wait/test comparison attribute";
}

def PTO_ReduceOpEnum : PTO_I32Enum<
  "ReduceOp", "PTO communication reduce operation", [
    I32EnumAttrCase<"Sum", 0, "sum">,
    I32EnumAttrCase<"Max", 1, "max">,
    I32EnumAttrCase<"Min", 2, "min">
  ]>;

def PTO_ReduceOpAttr : EnumAttr<PTO_Dialect, PTO_ReduceOpEnum, "reduce_op"> {
  let summary = "communication reduce operation attribute";
}

def PTO_ReluPreModeEnum : PTO_I32Enum<
  "ReluPreMode", "PTO TSTORE relu pre mode", [
    I32EnumAttrCase<"NoRelu", 0, "no_relu">,
    I32EnumAttrCase<"NormalRelu", 1, "normal_relu">,
    I32EnumAttrCase<"ScalarRelu", 2, "scalar_relu">,
    I32EnumAttrCase<"VectorRelu", 3, "vector_relu">,
    I32EnumAttrCase<"Pwl", 4, "pwl">
  ]>;

def PTO_ReluPreModeAttr : EnumAttr<PTO_Dialect, PTO_ReluPreModeEnum, "relu_pre_mode"> {
  let summary = "TSTORE relu pre mode attribute";
}

def PTO_UnitFlagCtrlEnum : PTO_I32Enum<
  "AccStoreUnitFlagCtrl", "PTO accumulator store unit-flag mode", [
    I32EnumAttrCase<"Off", 0, "off">,
    I32EnumAttrCase<"CheckOnly", 2, "check_only">,
    I32EnumAttrCase<"CheckAndClear", 3, "check_and_clear">
  ]>;

def PTO_UnitFlagCtrlAttr : EnumAttr<PTO_Dialect, PTO_UnitFlagCtrlEnum, "unit_flag_ctrl"> {
  let summary = "acc_store unit-flag control attribute";
}

def PTO_AccStoreQuantPreModeEnum : PTO_I32Enum<
  "AccStoreQuantPreMode", "PTO accumulator store pre-quantization mode", [
    I32EnumAttrCase<"NoConvert", 0, "no_convert">,
    I32EnumAttrCase<"F32F16", 1, "f32_f16">,
    I32EnumAttrCase<"QF322HIF8PreVec", 2, "qf322hif8_pre_vec">,
    I32EnumAttrCase<"QF322HIF8PreScalar", 3, "qf322hif8_pre_scalar">,
    I32EnumAttrCase<"QF322HIF8PreHybridVec", 4, "qf322hif8_pre_hybrid_vec">,
    I32EnumAttrCase<"QF322HIF8PreHybridScalar", 5, "qf322hif8_pre_hybrid_scalar">,
    I32EnumAttrCase<"DEQS32IntVec", 6, "deqs32_int_vec">,
    I32EnumAttrCase<"DEQS32IntScalar", 7, "deqs32_int_scalar">,
    I32EnumAttrCase<"REQ8Vec", 8, "req8_vec">,
    I32EnumAttrCase<"REQ8Scalar", 9, "req8_scalar">,
    I32EnumAttrCase<"DEQF16Vec", 10, "deqf16_vec">,
    I32EnumAttrCase<"DEQF16Scalar", 11, "deqf16_scalar">,
    I32EnumAttrCase<"QF322FP8PreVec", 12, "qf322fp8_pre_vec">,
    I32EnumAttrCase<"QF322FP8PreScalar", 13, "qf322fp8_pre_scalar">,
    I32EnumAttrCase<"QF322F32PreVec", 14, "qf322f32_pre_vec">,
    I32EnumAttrCase<"QF322F32PreScalar", 15, "qf322f32_pre_scalar">,
    I32EnumAttrCase<"F32BF16", 16, "f32_bf16">,
    I32EnumAttrCase<"QF162B8PreVec", 17, "qf162b8_pre_vec">,
    I32EnumAttrCase<"QF162B8PreScalar", 18, "qf162b8_pre_scalar">,
    I32EnumAttrCase<"QF162S4PreVec", 19, "qf162s4_pre_vec">,
    I32EnumAttrCase<"QF162S4PreScalar", 20, "qf162s4_pre_scalar">,
    I32EnumAttrCase<"REQ4Vec", 21, "req4_vec">,
    I32EnumAttrCase<"REQ4Scalar", 22, "req4_scalar">,
    I32EnumAttrCase<"QF322B8PreVec", 23, "qf322b8_pre_vec">,
    I32EnumAttrCase<"QF322B8PreScalar", 24, "qf322b8_pre_scalar">,
    I32EnumAttrCase<"QF322S4PreVec", 25, "qf322s4_pre_vec">,
    I32EnumAttrCase<"QF322S4PreScalar", 26, "qf322s4_pre_scalar">,
    I32EnumAttrCase<"DEQS16Vec", 27, "deqs16_vec">,
    I32EnumAttrCase<"DEQS16Scalar", 28, "deqs16_scalar">,
    I32EnumAttrCase<"QF162S16PreVec", 29, "qf162s16_pre_vec">,
    I32EnumAttrCase<"QF162S16PreScalar", 30, "qf162s16_pre_scalar">,
    I32EnumAttrCase<"QF322F16PreVec", 31, "qf322f16_pre_vec">,
    I32EnumAttrCase<"QF322F16PreScalar", 32, "qf322f16_pre_scalar">,
    I32EnumAttrCase<"QF322BF16PreVec", 33, "qf322bf16_pre_vec">,
    I32EnumAttrCase<"QF322BF16PreScalar", 34, "qf322bf16_pre_scalar">,
    I32EnumAttrCase<"QS322BF16PreVec", 35, "qs322bf16_pre_vec">,
    I32EnumAttrCase<"QS322BF16PreScalar", 36, "qs322bf16_pre_scalar">
  ]>;

def PTO_AccStoreQuantPreModeAttr : EnumAttr<PTO_Dialect, PTO_AccStoreQuantPreModeEnum, "quant_pre_mode"> {
  let summary = "acc_store pre-quantization mode attribute";
}

def PTO_AccStoreAtomicTypeEnum : PTO_I32Enum<
  "AccStoreAtomicType", "PTO accumulator store atomic type", [
    I32EnumAttrCase<"F32", 1, "f32">,
    I32EnumAttrCase<"F16", 2, "f16">,
    I32EnumAttrCase<"S16", 3, "s16">,
    I32EnumAttrCase<"S32", 4, "s32">,
    I32EnumAttrCase<"S8", 5, "s8">,
    I32EnumAttrCase<"BF16", 6, "bf16">
  ]>;

def PTO_AccStoreAtomicTypeAttr : EnumAttr<PTO_Dialect, PTO_AccStoreAtomicTypeEnum, "acc_store_atomic_type"> {
  let summary = "acc_store atomic type attribute";
}

def PTO_AccStoreAtomicOpEnum : PTO_I32Enum<
  "AccStoreAtomicOp", "PTO accumulator store atomic op", [
    I32EnumAttrCase<"Add", 0, "add">,
    I32EnumAttrCase<"Max", 1, "max">,
    I32EnumAttrCase<"Min", 2, "min">
  ]>;

def PTO_AccStoreAtomicOpAttr : EnumAttr<PTO_Dialect, PTO_AccStoreAtomicOpEnum, "acc_store_atomic_op"> {
  let summary = "acc_store atomic op attribute";
}

def PTO_AccStoreModeEnum : PTO_I32Enum<
  "AccStoreMode", "PTO accumulator store layout conversion mode", [
    I32EnumAttrCase<"Nz2nd", 0, "nz2nd">,
    I32EnumAttrCase<"Nz2dn", 1, "nz2dn">,
    I32EnumAttrCase<"Nz2nz", 2, "nz2nz">
  ]>;

def PTO_AccStoreModeAttr : EnumAttr<PTO_Dialect, PTO_AccStoreModeEnum, "acc_store_mode"> {
  let summary = "acc_store layout conversion mode";
}

def PTO_AccStoreUbDstModeEnum : PTO_I32Enum<
  "AccStoreUbDstMode", "PTO accumulator store UB destination mode", [
    I32EnumAttrCase<"Single", 0, "single">,
    I32EnumAttrCase<"SplitM", 1, "split_m">,
    I32EnumAttrCase<"SplitN", 2, "split_n">
  ]>;

def PTO_AccStoreUbDstModeAttr
    : EnumAttr<PTO_Dialect, PTO_AccStoreUbDstModeEnum, "acc_store_ub_dst_mode"> {
  let summary = "acc_store_ub destination mode attribute";
}

def PTO_AccStoreSatModeEnum : PTO_I32Enum<
  "AccStoreSatMode", "PTO accumulator store saturation mode", [
    I32EnumAttrCase<"Sat", 0, "sat">,
    I32EnumAttrCase<"NoSat", 1, "nosat">,
    I32EnumAttrCase<"SatPreserveNan", 2, "sat_preserve_nan">
  ]>;

def PTO_AccStoreSatModeAttr
    : EnumAttr<PTO_Dialect, PTO_AccStoreSatModeEnum, "acc_store_sat_mode"> {
  let summary = "acc_store saturation mode attribute";
}

def PTO_CubeLoadFracModeEnum : PTO_I32Enum<
  "CubeLoadFracMode", "PTO cube fractal load source layout mode", [
    I32EnumAttrCase<"Nd2nz", 0, "nd2nz">,
    I32EnumAttrCase<"Dn2nz", 1, "dn2nz">
  ]>;

def PTO_CubeLoadFracModeAttr : EnumAttr<PTO_Dialect, PTO_CubeLoadFracModeEnum, "cube_load_frac_mode"> {
  let summary = "cube_load_frac source layout conversion mode";
}

def PTO_GatherOOBEnum : PTO_I32Enum<
  "GatherOOB", "PTO MGATHER out-of-bounds mode", [
    I32EnumAttrCase<"Undefined", 0, "undefined">,
    I32EnumAttrCase<"Clamp", 1, "clamp">,
    I32EnumAttrCase<"Wrap", 2, "wrap">,
    I32EnumAttrCase<"Zero", 3, "zero">
  ]>;

def PTO_GatherOOBAttr : EnumAttr<PTO_Dialect, PTO_GatherOOBEnum, "gather_oob"> {
  let summary = "MGATHER out-of-bounds handling mode";
}

def PTO_ScatterAtomicOpEnum : PTO_I32Enum<
  "ScatterAtomicOp", "PTO MSCATTER atomic mode", [
    I32EnumAttrCase<"None", 0, "none">,
    I32EnumAttrCase<"Add", 1, "add">,
    I32EnumAttrCase<"Max", 2, "max">,
    I32EnumAttrCase<"Min", 3, "min">
  ]>;

def PTO_ScatterAtomicOpAttr : EnumAttr<PTO_Dialect, PTO_ScatterAtomicOpEnum, "scatter_atomic_op"> {
  let summary = "MSCATTER atomic mode";
}

def PTO_ScatterOOBEnum : PTO_I32Enum<
  "ScatterOOB", "PTO MSCATTER out-of-bounds mode", [
    I32EnumAttrCase<"Undefined", 0, "undefined">,
    I32EnumAttrCase<"Skip", 1, "skip">,
    I32EnumAttrCase<"Clamp", 2, "clamp">,
    I32EnumAttrCase<"Wrap", 3, "wrap">
  ]>;

def PTO_ScatterOOBAttr : EnumAttr<PTO_Dialect, PTO_ScatterOOBEnum, "scatter_oob"> {
  let summary = "MSCATTER out-of-bounds handling mode";
}

def PTO_AccToVecMode_Enum : PTO_I32Enum<"AccToVecMode", "TMOV acc-to-vec mode", [
  I32EnumAttrCase<"SingleModeVec0", 0, "single_mode_vec0">,
  I32EnumAttrCase<"SingleModeVec1", 1, "single_mode_vec1">,
  I32EnumAttrCase<"DualModeSplitM", 2, "dual_mode_split_m">,
  I32EnumAttrCase<"DualModeSplitN", 3, "dual_mode_split_n">
  ]>;

def PTO_AccToVecModeAttr : PTO_Attr<"AccToVecMode", "acc_to_vec_mode"> {
  let parameters = (ins EnumParameter<PTO_AccToVecMode_Enum>:$value);
  let assemblyFormat = "`<` params `>`";
}
//===----------------------------------------------------------------------===//
// MaskPattern
//===----------------------------------------------------------------------===//

def PTO_MaskPatternEnum : PTO_I32Enum<
  "MaskPattern", "PTO mask pattern", [
    I32EnumAttrCase<"P0101", 1>,
    I32EnumAttrCase<"P1010", 2>,
    I32EnumAttrCase<"P0001", 3>,
    I32EnumAttrCase<"P0010", 4>,
    I32EnumAttrCase<"P0100", 5>,
    I32EnumAttrCase<"P1000", 6>,
    I32EnumAttrCase<"P1111", 7>
  ]
>;

def PTO_MaskPatternAttr : PTO_Attr<"MaskPattern", "mask_pattern"> {
  let summary = "mask pattern attribute";
  let parameters = (ins EnumParameter<PTO_MaskPatternEnum>:$value);
  let assemblyFormat = "`<` params `>`";
}

// ---------- BLayout / SLayout / PadValue (same style as AddressSpace via PTO_I32Enum) ----------
def PTO_BLayout_Enum : PTO_I32Enum<"BLayout", "Base layout", [
  I32EnumAttrCase<"RowMajor", 0, "row_major">,
  I32EnumAttrCase<"ColMajor", 1, "col_major">
  ]>;

def PTO_BLayoutAttr : PTO_Attr<"BLayout", "blayout"> {
  let parameters = (ins EnumParameter<PTO_BLayout_Enum>:$value);
  let assemblyFormat = "`<` params `>`";
}

def PTO_SLayout_Enum : PTO_I32Enum<"SLayout", "Secondary layout", [
  I32EnumAttrCase<"NoneBox",  0, "none_box">,
  I32EnumAttrCase<"RowMajor", 1, "row_major">,
  I32EnumAttrCase<"ColMajor", 2, "col_major">
  ]>;

def PTO_SLayoutAttr : PTO_Attr<"SLayout", "slayout"> {
  let parameters = (ins EnumParameter<PTO_SLayout_Enum>:$value);
  let assemblyFormat = "`<` params `>`";
}

def PTO_PadValue_Enum : PTO_I32Enum<"PadValue", "Pad value policy", [
  I32EnumAttrCase<"Null", 0, "null">,
  I32EnumAttrCase<"Zero", 1, "zero">,
  I32EnumAttrCase<"Max",  2, "max">,
  I32EnumAttrCase<"Min",  3, "min">
  ]>;

def PTO_PadValueAttr : PTO_Attr<"PadValue", "pad_value"> {
  let parameters = (ins EnumParameter<PTO_PadValue_Enum>:$value);
  let assemblyFormat = "`<` params `>`";
}

def PTO_CompactMode_Enum : PTO_I32Enum<"CompactMode", "Tile compact mode", [
  I32EnumAttrCase<"Null", 0, "null">,
  I32EnumAttrCase<"Normal", 1, "normal">,
  I32EnumAttrCase<"RowPlusOne", 2, "row_plus_one">
  ]>;

def PTO_CompactModeAttr : PTO_Attr<"CompactMode", "compact_mode"> {
  let parameters = (ins EnumParameter<PTO_CompactMode_Enum>:$value);
  let assemblyFormat = "`<` params `>`";
}

// ---------- tile_buf_config (NO b_fractal / s_fractal) ----------
def TileBufConfigAttr : AttrDef<PTO_Dialect, "TileBufConfig"> {
  let mnemonic = "tile_buf_config";
  let parameters = (ins
    "BLayoutAttr":$bLayout,
    "SLayoutAttr":$sLayout,
    "mlir::IntegerAttr":$sFractalSize, // i32
    "PadValueAttr":$pad,
    "CompactModeAttr":$compactMode
  );

  let hasCustomAssemblyFormat = 1;

  let builders = [
    AttrBuilder<(ins
      "BLayoutAttr":$bLayout,
      "SLayoutAttr":$sLayout,
      "mlir::IntegerAttr":$sFractalSize,
      "PadValueAttr":$pad,
      "CompactModeAttr":$compactMode
    )>
  ];

  let extraClassDeclaration = [{
    static TileBufConfigAttr getDefault(MLIRContext *ctx);
    bool isDefault() const;

    static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                                mlir::Attribute bLayout,
                                mlir::Attribute sLayout,
                                mlir::IntegerAttr sFractalSize,
                                mlir::Attribute pad,
                                mlir::Attribute compactMode);
  }];
}

//===----------------------------------------------------------------------===//
// QuantType
//===----------------------------------------------------------------------===//

def PTO_QuantTypeEnum : PTO_I32Enum<
  "QuantType", "PTO quantization type", [
    I32EnumAttrCase<"INT8_SYM",  0>,
    I32EnumAttrCase<"INT8_ASYM", 1>
  ]>;

def PTO_QuantTypeAttr : EnumAttr<PTO_Dialect, PTO_QuantTypeEnum, "quant_type"> {
  let summary = "quantization type attribute";
}

//===----------------------------------------------------------------------===//
// TInsertMode (tinsert FP split mode for A5)
//===----------------------------------------------------------------------===//

def PTO_TInsertMode_Enum : PTO_I32Enum<"TInsertMode", "TINSERT split mode (A5)", [
  I32EnumAttrCase<"SPLIT2", 2, "split2">,
  I32EnumAttrCase<"SPLIT4", 3, "split4">
]>;

def PTO_TInsertModeAttr : PTO_Attr<"TInsertMode", "tinsert_mode"> {
  let parameters = (ins EnumParameter<PTO_TInsertMode_Enum>:$value);
  let assemblyFormat = "`<` params `>`";
}

#endif // MLIR_DIALECT_PTO_IR_PTOATTRS
