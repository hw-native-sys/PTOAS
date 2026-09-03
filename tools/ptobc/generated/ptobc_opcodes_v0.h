// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// PTO-BC v0 schema table.
//
// This checked-in table is authoritative. The generator referenced by older
// revisions is not shipped in this repository, so changes must be reviewed
// against tools/ptobc/MAINTENANCE.md and the v0 compatibility tests.
#pragma once

#include <cstdint>
#include <optional>

#include <llvm/ADT/StringRef.h>

namespace ptobc::v0 {

inline constexpr uint8_t kVariantDefault = 0;
inline constexpr uint8_t kVariantAcc = 1;
inline constexpr uint8_t kVariantBias = 2;
inline constexpr uint8_t kVariantMx = 3;
inline constexpr uint8_t kVariantMxAcc = 4;
inline constexpr uint8_t kVariantMxBias = 5;
inline constexpr uint8_t kSectionCubeVariant = 0;
inline constexpr uint8_t kSectionVectorVariant = 1;
inline constexpr uint8_t kHasVariant = 1;
inline constexpr uint16_t kTscatterMaskOpcode = 0x109C;

inline constexpr int kTgemvOperandCount = 3;
inline constexpr int kTgemvAccOperandCount = 4;
inline constexpr int kTgemvBiasOperandCount = 4;
inline constexpr int kTgemvMxOperandCount = 5;
inline constexpr int kTgemvMxAccOperandCount = 6;
inline constexpr int kTgemvMxBiasOperandCount = 6;
inline constexpr int kTmatmulOperandCount = 3;
inline constexpr int kTmatmulAccOperandCount = 4;
inline constexpr int kTmatmulBiasOperandCount = 4;
inline constexpr int kTmatmulMxOperandCount = 5;
inline constexpr int kTmatmulMxAccOperandCount = 6;
inline constexpr int kTmatmulMxBiasOperandCount = 6;

struct OpInfo {
  uint16_t opcode;
  const char *name;
  uint8_t has_variant_u8;
  uint8_t result_type_mode;
  uint8_t operand_mode;
  uint16_t num_operands;
  uint16_t num_results;
  uint16_t num_regions;
  uint8_t imm_kind;
};

inline constexpr OpInfo kOpTable[] = {
  {0x0000, "pto.get_block_idx", 0, 0x01, 0x00, 0, 1, 0, 0x00},
  {0x0001, "pto.get_block_num", 0, 0x01, 0x00, 0, 1, 0, 0x00},
  {0x0002, "pto.get_subblock_idx", 0, 0x01, 0x00, 0, 1, 0, 0x00},
  {0x0003, "pto.get_subblock_num", 0, 0x01, 0x00, 0, 1, 0, 0x00},
  {0x0004, "pto.make_tensor_view", 0, 0x01, 0x03, 1, 1, 0, 0x06},
  {0x0005, "pto.partition_view", 0, 0x01, 0x03, 1, 1, 0, 0x07},
  {0x0006, "pto.section", 1, 0x00, 0x00, 0, 0, 1, 0x00},
  {0x1000, "pto.addptr", 0, 0x01, 0x00, 2, 1, 0, 0x00},
  {0x1001, "pto.alloc_tile", 0, 0x01, 0x04, 0, 1, 0, 0x08},
  {0x1002, "pto.barrier", 0, 0x00, 0x00, 0, 0, 0, 0x00},
  {0x1003, "pto.mgather", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1004, "pto.mscatter", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1005, "pto.record_event", 0, 0x00, 0x00, 0, 0, 0, 0x02},
  {0x1006, "pto.tabs", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x1007, "pto.tadd", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1008, "pto.taddc", 0, 0x00, 0x00, 4, 0, 0, 0x00},
  {0x1009, "pto.tadds", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x100A, "pto.taddsc", 0, 0x00, 0x00, 4, 0, 0, 0x00},
  {0x100B, "pto.tand", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x100C, "pto.tands", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x100D, "pto.tci", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x100E, "pto.tcmp", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x100F, "pto.tcmps", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1010, "pto.tcolexpand", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x1011, "pto.tcolexpandadd", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1012, "pto.tcolexpanddiv", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1013, "pto.tcolexpandexpdif", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1014, "pto.tcolexpandmax", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1015, "pto.tcolexpandmin", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1016, "pto.tcolexpandmul", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1017, "pto.tcolexpandsub", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1018, "pto.tcolmax", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x1019, "pto.tcolmin", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x101A, "pto.tcolprod", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x101B, "pto.tcolsum", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x101C, "pto.tcvt", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x101D, "pto.tdiv", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x101E, "pto.tdivs", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x101F, "pto.texp", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x1020, "pto.texpands", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x1021, "pto.textract", 0, 0x00, 0x00, 4, 0, 0, 0x00},
  // Legacy textract_fp wire opcode; decoded as the unified pto.textract op.
  {0x1022, "pto.textract", 0, 0x00, 0x00, 5, 0, 0, 0x00},
  {0x1023, "pto.tfillpad", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  // Legacy tfillpad wire opcodes; decoded as the unified pto.tfillpad op.
  {0x1024, "pto.tfillpad", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x1025, "pto.tfillpad", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x1026, "pto.tfmod", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1027, "pto.tfmods", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1028, "pto.tgather", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x1029, "pto.tgatherb", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x102A, "pto.tgemv", 1, 0x00, 0x01, 0, 0, 0, 0x00},
  {0x102B, "pto.tgetval", 0, 0x01, 0x00, 2, 1, 0, 0x00},
  {0x102C, "pto.timg2col", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x102D, "pto.tinsert", 0, 0x00, 0x00, 4, 0, 0, 0x00},
  // Legacy tinsert_fp wire opcode; decoded as the unified pto.tinsert op.
  {0x102E, "pto.tinsert", 0, 0x00, 0x00, 5, 0, 0, 0x00},
  {0x102F, "pto.tload", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x1030, "pto.tlog", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x1031, "pto.tlrelu", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1032, "pto.tmatmul", 1, 0x00, 0x01, 0, 0, 0, 0x00},
  {0x1033, "pto.tmatmul.mx", 1, 0x00, 0x01, 0, 0, 0, 0x00},
  {0x1034, "pto.tmax", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1035, "pto.tmaxs", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1036, "pto.tmin", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1037, "pto.tmins", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1038, "pto.tmov", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  // Legacy tmov.fp wire opcode; decoded as the unified pto.tmov op.
  {0x1039, "pto.tmov", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x103A, "pto.tmrgsort", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x103B, "pto.tmul", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x103C, "pto.tmuls", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x103D, "pto.tneg", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x103E, "pto.tnot", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x103F, "pto.tor", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1040, "pto.tors", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1041, "pto.tpartadd", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1042, "pto.tpartmax", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1043, "pto.tpartmin", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1044, "pto.tpartmul", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1045, "pto.tprefetch", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x1046, "pto.tprelu", 0, 0x00, 0x00, 4, 0, 0, 0x00},
  {0x1047, "pto.tquant", 0, 0x00, 0x02, 3, 0, 0, 0x00},
  {0x1048, "pto.trecip", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x1049, "pto.trelu", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x104A, "pto.trem", 0, 0x00, 0x00, 4, 0, 0, 0x00},
  {0x104B, "pto.trems", 0, 0x00, 0x00, 4, 0, 0, 0x00},
  {0x104C, "pto.treshape", 0, 0x01, 0x00, 1, 1, 0, 0x00},
  {0x104D, "pto.trowexpand", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x104E, "pto.trowexpandadd", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x104F, "pto.trowexpandexpdif", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x1050, "pto.trowexpandmax", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x1051, "pto.trowexpandmin", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x1052, "pto.trowmax", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1053, "pto.trowmin", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1054, "pto.trowsum", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1055, "pto.trsqrt", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x1056, "pto.tscatter", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1057, "pto.tsel", 0, 0x00, 0x00, 5, 0, 0, 0x00},
  {0x1058, "pto.tsels", 0, 0x00, 0x00, 5, 0, 0, 0x00},
  {0x1059, "pto.tset_img2col_padding", 0, 0x00, 0x00, 1, 0, 0, 0x00},
  {0x105A, "pto.tset_img2col_rpt", 0, 0x00, 0x00, 1, 0, 0, 0x00},
  {0x105B, "pto.tsetfmatrix", 0, 0x00, 0x00, 1, 0, 0, 0x00},
  {0x105C, "pto.tsethf32mode", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x105D, "pto.tsettf32mode", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x105E, "pto.tsetval", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x105F, "pto.tshl", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1060, "pto.tshls", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1061, "pto.tshr", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1062, "pto.tshrs", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1063, "pto.tsort32", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x1064, "pto.tsqrt", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x1065, "pto.tstore", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  // Legacy tstore_fp wire opcode; decoded as the unified pto.tstore op.
  {0x1066, "pto.tstore", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1067, "pto.tsub", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1068, "pto.tsubc", 0, 0x00, 0x00, 4, 0, 0, 0x00},
  {0x1069, "pto.tsubs", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x106A, "pto.tsubsc", 0, 0x00, 0x00, 4, 0, 0, 0x00},
  {0x106B, "pto.trowexpandsub", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x106C, "pto.ttrans", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x106D, "pto.ttri", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x106E, "pto.txor", 0, 0x00, 0x00, 4, 0, 0, 0x00},
  {0x106F, "pto.txors", 0, 0x00, 0x00, 4, 0, 0, 0x00},
  {0x1070, "pto.wait_event", 0, 0x00, 0x00, 0, 0, 0, 0x02},
  {0x1071, "pto.tprint", 0, 0x00, 0x00, 1, 0, 0, 0x00},
  {0x1072, "pto.subview", 0, 0x01, 0x02, 0, 1, 0, 0x00},
  {0x1073, "pto.trowexpanddiv", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x1074, "pto.trowexpandmul", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x1075, "pto.tdequant", 0, 0x00, 0x00, 4, 0, 0, 0x00},
  {0x1076, "pto.taxpy", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1077, "pto.thistogram", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1078, "pto.tget_scale_addr", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x1079, "pto.trowargmax", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x107A, "pto.trowargmin", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x107B, "pto.tcolargmax", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x107C, "pto.tcolargmin", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x107D, "pto.tsync", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x107E, "pto.reserve_buffer", 0, 0x01, 0x00, 0, 1, 0, 0x00},
  {0x107F, "pto.import_reserved_buffer", 0, 0x01, 0x00, 0, 1, 0, 0x00},
  {0x1080, "pto.aic_initialize_pipe", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x1081, "pto.aiv_initialize_pipe", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x1082, "pto.tpush_to_aiv", 0, 0x00, 0x00, 1, 0, 0, 0x00},
  {0x1083, "pto.tpush_to_aic", 0, 0x00, 0x00, 1, 0, 0, 0x00},
  {0x1084, "pto.tpop_from_aic", 0, 0x01, 0x02, 0, 1, 0, 0x00},
  {0x1085, "pto.tpop_from_aiv", 0, 0x01, 0x02, 0, 1, 0, 0x00},
  {0x1086, "pto.tfree_from_aic", 0, 0x00, 0x00, 0, 0, 0, 0x00},
  {0x1087, "pto.tfree_from_aiv", 0, 0x00, 0x00, 0, 0, 0, 0x00},
  {0x1088, "pto.set_validshape", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x1089, "pto.tconcat", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x108A, "pto.trowprod", 0, 0x00, 0x00, 3, 0, 0, 0x00},
  {0x108B, "pto.initialize_l2g2l_pipe", 0, 0x01, 0x02, 0, 1, 0, 0x00},
  {0x108C, "pto.initialize_l2l_pipe", 0, 0x01, 0x02, 0, 1, 0, 0x00},
  {0x108D, "pto.tpush", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x108E, "pto.declare_tile", 0, 0x01, 0x00, 0, 1, 0, 0x00},
  {0x108F, "pto.tpop", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x1090, "pto.tfree", 0, 0x00, 0x00, 1, 0, 0, 0x00},
  {0x1091, "pto.comm.tput", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x1092, "pto.comm.tget", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x1093, "pto.comm.tnotify", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x1094, "pto.comm.twait", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x1095, "pto.comm.ttest", 0, 0x01, 0x02, 0, 1, 0, 0x00},
  {0x1096, "pto.comm.tbroadcast", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x1097, "pto.comm.tgather", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x1098, "pto.comm.tscatter", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x1099, "pto.comm.treduce", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x109A, "pto.tpartargmax", 0, 0x00, 0x00, 6, 0, 0, 0x00},
  {0x109B, "pto.tpartargmin", 0, 0x00, 0x00, 6, 0, 0, 0x00},
  {0x109C, "pto.tscatter.maskpattern", 0, 0x00, 0x00, 2, 0, 0, 0x00},
  {0x109D, "pto.fusion_region", 0, 0x02, 0x00, 0, 0, 1, 0x00},
  {0x109E, "pto.yield", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x2000, "arith.addi", 0, 0x01, 0x00, 2, 1, 0, 0x00},
  {0x2001, "arith.ceildivsi", 0, 0x01, 0x00, 2, 1, 0, 0x00},
  {0x2002, "arith.cmpi", 0, 0x01, 0x00, 2, 1, 0, 0x01},
  {0x2003, "arith.constant", 0, 0x01, 0x00, 0, 1, 0, 0x05},
  {0x2004, "arith.index_cast", 0, 0x01, 0x00, 1, 1, 0, 0x00},
  {0x2005, "arith.minui", 0, 0x01, 0x00, 2, 1, 0, 0x00},
  {0x2006, "arith.muli", 0, 0x01, 0x00, 2, 1, 0, 0x00},
  {0x2007, "arith.select", 0, 0x01, 0x00, 3, 1, 0, 0x00},
  {0x2008, "arith.subi", 0, 0x01, 0x00, 2, 1, 0, 0x00},
  {0x4000, "scf.for", 0, 0x00, 0x00, 3, 0, 1, 0x00},
  {0x4001, "scf.if", 0, 0x00, 0x00, 1, 0, 2, 0x00},
  {0x4002, "scf.yield", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x6000, "func.func", 0, 0x00, 0x00, 0, 0, 0, 0x00},
  {0x6001, "func.return", 0, 0x00, 0x02, 0, 0, 0, 0x00},
  {0x6002, "func.call", 0, 0x02, 0x02, 0, 0, 0, 0x00},
};

inline const OpInfo *lookupByOpcode(uint16_t opcode) {
  // Binary search on kOpTable (sorted by opcode).
  size_t lo = 0, hi = sizeof(kOpTable) / sizeof(kOpTable[0]);
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    uint16_t v = kOpTable[mid].opcode;
    if (v == opcode) return &kOpTable[mid];
    if (v < opcode) lo = mid + 1; else hi = mid;
  }
  return nullptr;
}

inline const OpInfo *lookupByName(llvm::StringRef name) {
  for (const OpInfo &entry : kOpTable) {
    if (name == entry.name) {
      return &entry;
    }
  }
  return nullptr;
}

struct OpcodeAndVariant { uint16_t opcode; uint8_t hasVariant; uint8_t variant; };

struct VariantName {
  uint16_t opcode;
  uint8_t variant;
  const char *name;
};

// Family-op variant names (ops whose full name differs per variant). The
// base full names are already present in kOpTable, so only the suffixed
// variants are listed here. `pto.tscatter` resolves to the indexed form via
// kOpTable; the mask form uses the distinct name `pto.tscatter.maskpattern`.
inline constexpr VariantName kVariantNameTable[] = {
    {0x0006, kSectionCubeVariant, "pto.section.cube"},
    {0x0006, kSectionVectorVariant, "pto.section.vector"},
    {0x102A, kVariantAcc, "pto.tgemv.acc"},
    {0x102A, kVariantBias, "pto.tgemv.bias"},
    {0x102A, kVariantMx, "pto.tgemv.mx"},
    {0x102A, kVariantMxAcc, "pto.tgemv.mx.acc"},
    {0x102A, kVariantMxBias, "pto.tgemv.mx.bias"},
    {0x1032, kVariantAcc, "pto.tmatmul.acc"},
    {0x1032, kVariantBias, "pto.tmatmul.bias"},
    {0x1033, kVariantDefault, "pto.tmatmul.mx"},
    {0x1033, kVariantAcc, "pto.tmatmul.mx.acc"},
    {0x1033, kVariantBias, "pto.tmatmul.mx.bias"},
};

// For non-family ops, variant is 0. For family ops, variant is the assigned u8.
// NOTE: `pto.section` is not a real op name; use `pto.section.cube`/`pto.section.vector`.
inline std::optional<OpcodeAndVariant> lookupOpcodeAndVariantByFullName(llvm::StringRef fullName) {
  // Neither name was resolvable in the original per-name switch and both must
  // stay unresolvable: `pto.section` is not a registered op (only its
  // .cube/.vector variants are), and `pto.tscatter.maskpattern` is a
  // decode-only wire alias. Without these guards the kOpTable scan below
  // would compact-encode an unregistered op with either name instead of
  // falling through to the generic record.
  if (fullName == "pto.section" || fullName == "pto.tscatter.maskpattern") {
    return std::nullopt;
  }
  for (const OpInfo &entry : kOpTable) {
    if (fullName == entry.name) {
      return OpcodeAndVariant{entry.opcode, entry.has_variant_u8, 0};
    }
  }
  for (const VariantName &entry : kVariantNameTable) {
    if (fullName == entry.name) {
      return OpcodeAndVariant{entry.opcode, kHasVariant, entry.variant};
    }
  }
  return std::nullopt;
}

inline const char *fullNameFromOpcodeVariant(uint16_t opcode, uint8_t variant) {
  const OpInfo *info = lookupByOpcode(opcode);
  if (!info) {
    return nullptr;
  }
  if (opcode == kTscatterMaskOpcode) {
    return "pto.tscatter";
  }
  if (info->has_variant_u8 == 0) {
    return info->name;
  }
  for (const VariantName &entry : kVariantNameTable) {
    if (entry.opcode == opcode && entry.variant == variant) {
      return entry.name;
    }
  }
  return info->name;
}

inline std::optional<int> lookupOperandsByVariant(uint16_t opcode, uint8_t variant) {
  switch (opcode) {
  case 0x102A:
    switch (variant) {
    case kVariantDefault: return kTgemvOperandCount;
    case kVariantAcc: return kTgemvAccOperandCount;
    case kVariantBias: return kTgemvBiasOperandCount;
    case kVariantMx: return kTgemvMxOperandCount;
    case kVariantMxAcc: return kTgemvMxAccOperandCount;
    case kVariantMxBias: return kTgemvMxBiasOperandCount;
    default: return std::nullopt;
    }
  case 0x1032:
    switch (variant) {
    case kVariantDefault: return kTmatmulOperandCount;
    case kVariantAcc: return kTmatmulAccOperandCount;
    case kVariantBias: return kTmatmulBiasOperandCount;
    default: return std::nullopt;
    }
  case 0x1033:
    switch (variant) {
    case kVariantDefault: return kTmatmulMxOperandCount;
    case kVariantAcc: return kTmatmulMxAccOperandCount;
    case kVariantBias: return kTmatmulMxBiasOperandCount;
    default: return std::nullopt;
    }
  default: return std::nullopt;
  }
}

} // namespace ptobc::v0
