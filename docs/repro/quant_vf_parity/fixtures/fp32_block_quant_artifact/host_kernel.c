// TVM host target: {"kind":"c","tag":"","keys":["cpu"]}
#define TVM_EXPORTS
#include "tvm/runtime/base.h"
#include "tvm/runtime/c_backend_api.h"
#include "tvm/ffi/c_api.h"
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#if defined(_MSC_VER)
#define TL_ALIGN(N) __declspec(align(N))
#else
#define TL_ALIGN(N) __attribute__((aligned(N)))
#endif
#ifdef __OBJC__
#include "tvm/runtime/device_api.h"
#include "tvm/ffi/function.h"
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <torch/mps.h>
#endif
void* __tvm_ffi__library_ctx = NULL;
static void* __tvm_set_device_packed = NULL;
static void* per_block_cast_kernel_kernel_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_per_block_cast_kernel(void* self_handle, void* args, int32_t num_args, void* result);
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_per_block_cast_kernel(void* self_handle, void* args, int32_t num_args, void* result) {
  TL_ALIGN(128) TVMFFIAny stack[8];
  void* stack_ffi_any = stack;
  if (!((num_args == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "per_block_cast_kernel: num_args should be 3", (long long)(num_args), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(args == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "per_block_cast_kernel: args pointer is NULL");
    return -1;
  }
  int32_t x_handle_type_index = (((TVMFFIAny*)args)[0].type_index);
  if (!(((((x_handle_type_index == 0) || (x_handle_type_index == 4)) || (x_handle_type_index == 7)) || (64 <= x_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel per_block_cast_kernel input x expected pointer or tensor handle");
    return -1;
  }
  int32_t out_handle_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((out_handle_type_index == 0) || (out_handle_type_index == 4)) || (out_handle_type_index == 7)) || (64 <= out_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel per_block_cast_kernel input out expected pointer or tensor handle");
    return -1;
  }
  int32_t out_sf_handle_type_index = (((TVMFFIAny*)args)[2].type_index);
  if (!(((((out_sf_handle_type_index == 0) || (out_sf_handle_type_index == 4)) || (out_sf_handle_type_index == 7)) || (64 <= out_sf_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel per_block_cast_kernel input output_sf expected pointer or tensor handle");
    return -1;
  }
  void* x_handle = ((x_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  void* out_handle = ((out_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  void* out_sf_handle = ((out_sf_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[2].v_ptr) + 24)) : (((TVMFFIAny*)args)[2].v_ptr));
  bool per_block_cast_kernel_x_is_null = (x_handle == NULL);
  if (!(!per_block_cast_kernel_x_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "per_block_cast_kernel.x is expected to have non-NULL pointer");
    return -1;
  }
  bool per_block_cast_kernel_out_is_null = (out_handle == NULL);
  if (!(!per_block_cast_kernel_out_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "per_block_cast_kernel.out is expected to have non-NULL pointer");
    return -1;
  }
  bool per_block_cast_kernel_output_sf_is_null = (out_sf_handle == NULL);
  if (!(!per_block_cast_kernel_output_sf_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "per_block_cast_kernel.output_sf is expected to have non-NULL pointer");
    return -1;
  }
  void* per_block_cast_kernel_x_shape = (((DLTensor*)x_handle)[0].shape);
  void* per_block_cast_kernel_out_shape = (((DLTensor*)out_handle)[0].shape);
  void* per_block_cast_kernel_output_sf_shape = (((DLTensor*)out_sf_handle)[0].shape);
  if (!(((((DLTensor*)x_handle)[0].ndim) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel per_block_cast_kernel input x ndim mismatch, expected 2", (long long)((((DLTensor*)x_handle)[0].ndim)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((bool)1)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Symbolic shape variable num_tokens requires at least one non-null buffer among: per_block_cast_kernel.x, per_block_cast_kernel.out");
    return -1;
  }
  int32_t num_tokens = ((int32_t)((int64_t*)per_block_cast_kernel_x_shape)[0]);
  void* per_block_cast_kernel_x_strides = (((DLTensor*)x_handle)[0].strides);
  int32_t dev_id = (((DLTensor*)x_handle)[0].device.device_id);
  void* x = (((DLTensor*)x_handle)[0].data);
  if (!(((((DLTensor*)out_handle)[0].ndim) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel per_block_cast_kernel input out ndim mismatch, expected 2", (long long)((((DLTensor*)out_handle)[0].ndim)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* per_block_cast_kernel_out_strides = (((DLTensor*)out_handle)[0].strides);
  void* out = (((DLTensor*)out_handle)[0].data);
  if (!(((((DLTensor*)out_sf_handle)[0].ndim) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel per_block_cast_kernel input output_sf ndim mismatch, expected 2", (long long)((((DLTensor*)out_sf_handle)[0].ndim)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* per_block_cast_kernel_output_sf_strides = (((DLTensor*)out_sf_handle)[0].strides);
  int32_t condval;
  if ((per_block_cast_kernel_output_sf_strides == NULL)) {
    condval = 1;
  } else {
    condval = ((int32_t)((int64_t*)per_block_cast_kernel_output_sf_strides)[0]);
  }
  int32_t sf_stride = condval;
  void* output_sf = (((DLTensor*)out_sf_handle)[0].data);
  if (!(((((((DLTensor*)x_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)x_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)x_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel per_block_cast_kernel input x dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)per_block_cast_kernel_x_shape)[1]) == 2048))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel per_block_cast_kernel input x shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)per_block_cast_kernel_x_shape)[1])), (long long)(2048));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_1;
  if ((per_block_cast_kernel_x_strides == NULL)) {
    condval_1 = 1;
  } else {
    condval_1 = ((int32_t)((int64_t*)per_block_cast_kernel_x_strides)[1]);
  }
  if (!((condval_1 == 1))) {
    int32_t condval_2;
    if ((per_block_cast_kernel_x_strides == NULL)) {
      condval_2 = 1;
    } else {
      condval_2 = ((int32_t)((int64_t*)per_block_cast_kernel_x_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel per_block_cast_kernel input x strides[1] violates packed ABI constraint", (long long)(condval_2), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_3;
  if ((per_block_cast_kernel_x_strides == NULL)) {
    condval_3 = 1;
  } else {
    condval_3 = ((int32_t)((int64_t*)per_block_cast_kernel_x_strides)[0]);
  }
  if (!(((condval_3 == 2048) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel per_block_cast_kernel input x strides[0] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)x_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel per_block_cast_kernel input x byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)x_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)x_handle)[0].device.device_type) == 12))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel per_block_cast_kernel input x device_type mismatch, expected ext_dev", (long long)((((DLTensor*)x_handle)[0].device.device_type)), (long long)(12));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((num_tokens == 0) || !(x == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel per_block_cast_kernel input x data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)out_handle)[0].dtype.code) == (uint8_t)10) && ((((DLTensor*)out_handle)[0].dtype.bits) == (uint8_t)8)) && ((((DLTensor*)out_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel per_block_cast_kernel input out dtype mismatch, expected float8_e4m3fn");
    return -1;
  }
  if (!((((int32_t)((int64_t*)per_block_cast_kernel_x_shape)[0]) == ((int32_t)((int64_t*)per_block_cast_kernel_out_shape)[0])))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel per_block_cast_kernel input out shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)per_block_cast_kernel_x_shape)[0])), (long long)(((int32_t)((int64_t*)per_block_cast_kernel_out_shape)[0])));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)per_block_cast_kernel_out_shape)[1]) == 2048))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel per_block_cast_kernel input out shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)per_block_cast_kernel_out_shape)[1])), (long long)(2048));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_4;
  if ((per_block_cast_kernel_out_strides == NULL)) {
    condval_4 = 1;
  } else {
    condval_4 = ((int32_t)((int64_t*)per_block_cast_kernel_out_strides)[1]);
  }
  if (!((condval_4 == 1))) {
    int32_t condval_5;
    if ((per_block_cast_kernel_out_strides == NULL)) {
      condval_5 = 1;
    } else {
      condval_5 = ((int32_t)((int64_t*)per_block_cast_kernel_out_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel per_block_cast_kernel input out strides[1] violates packed ABI constraint", (long long)(condval_5), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_6;
  if ((per_block_cast_kernel_out_strides == NULL)) {
    condval_6 = 1;
  } else {
    condval_6 = ((int32_t)((int64_t*)per_block_cast_kernel_out_strides)[0]);
  }
  if (!(((condval_6 == 2048) || (num_tokens == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel per_block_cast_kernel input out strides[0] violates packed ABI constraint");
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)out_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel per_block_cast_kernel input out byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)out_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)out_handle)[0].device.device_id) == (((DLTensor*)x_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel per_block_cast_kernel input out device_id violates packed ABI constraint", (long long)((((DLTensor*)out_handle)[0].device.device_id)), (long long)((((DLTensor*)x_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)out_handle)[0].device.device_type) == 12))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel per_block_cast_kernel input out device_type mismatch, expected ext_dev", (long long)((((DLTensor*)out_handle)[0].device.device_type)), (long long)(12));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((num_tokens == 0) || !(out == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel per_block_cast_kernel input out data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)out_sf_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)out_sf_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)out_sf_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel per_block_cast_kernel input output_sf dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)per_block_cast_kernel_output_sf_shape)[0]) == ((num_tokens + 31) >> 5)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel per_block_cast_kernel input output_sf shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)per_block_cast_kernel_output_sf_shape)[0])), (long long)(((num_tokens + 31) >> 5)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)per_block_cast_kernel_output_sf_shape)[1]) == 64))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel per_block_cast_kernel input output_sf shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)per_block_cast_kernel_output_sf_shape)[1])), (long long)(64));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_7;
  if ((per_block_cast_kernel_output_sf_strides == NULL)) {
    condval_7 = 1;
  } else {
    condval_7 = ((int32_t)((int64_t*)per_block_cast_kernel_output_sf_strides)[1]);
  }
  if (!((condval_7 == 1))) {
    int32_t condval_8;
    if ((per_block_cast_kernel_output_sf_strides == NULL)) {
      condval_8 = 1;
    } else {
      condval_8 = ((int32_t)((int64_t*)per_block_cast_kernel_output_sf_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel per_block_cast_kernel input output_sf strides[1] violates packed ABI constraint", (long long)(condval_8), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)out_sf_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel per_block_cast_kernel input output_sf byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)out_sf_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)out_sf_handle)[0].device.device_id) == (((DLTensor*)x_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel per_block_cast_kernel input output_sf device_id violates packed ABI constraint", (long long)((((DLTensor*)out_sf_handle)[0].device.device_id)), (long long)((((DLTensor*)x_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)out_sf_handle)[0].device.device_type) == 12))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel per_block_cast_kernel input output_sf device_type mismatch, expected ext_dev", (long long)((((DLTensor*)out_sf_handle)[0].device.device_type)), (long long)(12));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((num_tokens + 31) >> 5) == 0) || !(output_sf == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel per_block_cast_kernel input output_sf data pointer is NULL");
    return -1;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = ((int64_t)12);
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)dev_id);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
  if (__tvm_set_device_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_set_device", &__tvm_set_device_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_1;
  result_1.type_index = kTVMFFINone;
  result_1.zero_padding = 0;
  result_1.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_set_device_packed, (TVMFFIAny*) stack_ffi_any, 2, &result_1) != 0) {
    return -1;
  }
  if (out == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = out;
  if (output_sf == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = output_sf;
  if (x == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = x;
  (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)num_tokens);
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)sf_stride);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)72);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = (int64_t)163968;
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = (int64_t)0;
  if (per_block_cast_kernel_kernel_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "per_block_cast_kernel_kernel", &per_block_cast_kernel_kernel_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_2;
  result_2.type_index = kTVMFFINone;
  result_2.zero_padding = 0;
  result_2.v_int64 = 0;
  if (TVMFFIFunctionCall(per_block_cast_kernel_kernel_packed, (TVMFFIAny*) stack_ffi_any, 7, &result_2) != 0) {
    return -1;
  }
  return 0;
}

// CodegenC: NOTE: Auto-generated entry function
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_main(void* self, void* args,int num_args, void* result) {
  return __tvm_ffi_per_block_cast_kernel(self, args, num_args, result);
}
