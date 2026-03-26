// RUN: ./build/tools/ptoas/ptoas %s -o - | FileCheck %s

// CHECK-LABEL: @copy_gm_to_ubuf
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: a5vm.copy_gm_to_ubuf %arg0, %arg1, %[[ROWS:[^,]+]], %[[COLS:[^,]+]], %[[ZERO:[^,]+]], %[[NBURST:[^,]+]], %[[LEN:[^,]+]], %[[ZERO]], %[[ZERO]], %[[FALSE]], %[[ZERO]], %[[GMSTRIDE:[^,]+]], %[[UBSTRIDE:[^ ]+]]
// CHECK-SAME: : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i64, i64, i1, i64, i64, i64
module {
  func.func @copy_gm_to_ubuf(%src: !pto.ptr<f32, gm>, %dst: !pto.ptr<f32, ub>) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %cfalse = arith.constant false
    a5vm.copy_gm_to_ubuf %src, %dst, %c32_i64, %c32_i64, %c0_i64, %c32_i64, %c128_i64, %c0_i64, %c0_i64, %cfalse, %c0_i64, %c128_i64, %c128_i64 : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i64, i64, i1, i64, i64, i64
    return
  }
}

// CHECK: error: 'a5vm.copy_gm_to_ubuf' op requires GM source and UB destination
module {
  func.func @copy_gm_to_ubuf_wrong_direction(%src: !pto.ptr<f32, ub>, %dst: !pto.ptr<f32, gm>) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %cfalse = arith.constant false
    a5vm.copy_gm_to_ubuf %src, %dst, %c32_i64, %c32_i64, %c0_i64, %c32_i64, %c128_i64, %c0_i64, %c0_i64, %cfalse, %c0_i64, %c128_i64, %c128_i64 : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64, i64, i1, i64, i64, i64
    return
  }
}

// CHECK: error: 'a5vm.copy_gm_to_ubuf' op requires source and destination element byte widths to match
module {
  func.func @copy_gm_to_ubuf_width_mismatch(%src: !pto.ptr<i8, gm>, %dst: !pto.ptr<f32, ub>) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %cfalse = arith.constant false
    a5vm.copy_gm_to_ubuf %src, %dst, %c32_i64, %c32_i64, %c0_i64, %c32_i64, %c128_i64, %c0_i64, %c0_i64, %cfalse, %c0_i64, %c128_i64, %c128_i64 : !pto.ptr<i8, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i64, i64, i1, i64, i64, i64
    return
  }
}
