#pragma once
#include <stdint.h>
#include "kernel_operator.h"
#include "simt_api/asc_bf16.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_fp8.h"
#include "simt_api/asc_simt.h"
using fp8_e4_t = float8_e4m3_t;
struct fp8_e5_t { uint8_t bits; };
using fp8_e8_t = uint8_t;
