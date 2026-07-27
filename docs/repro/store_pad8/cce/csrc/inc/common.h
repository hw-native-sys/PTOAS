#pragma once

#include "acl/acl.h"
#include "kernel_operator.h"
#include "simt_api/asc_bf16.h"
#include "simt_api/asc_fp16.h"

#include "numeric_limits.h"
#include "simd_inst.h"

// Reduce N_ACC live f32 accumulator vectors (VL=64 lanes each) to N_ACC
// scalars on one core. N_ACC=20 (LARGE) matches a typical residual-mix
// epilogue. N_ACC=4 (SMALL) is a quick smoke case.
constexpr int32_t kVL = 64;
constexpr int32_t kMaxNAcc = 20;

// UB layout (bytes):
//   acc[N_ACC * kVL] f32      @0
//   reduced[N_ACC] f32        @kMaxNAcc*kVL*4  (compact ONEPT store target)
constexpr int32_t kAccUb = 0;
constexpr int32_t kReducedUb = kMaxNAcc * kVL * 4;
constexpr int32_t kUbBytes = kReducedUb + kMaxNAcc * 4;
