// RUN: ./build/tools/ptoas/ptoas %s -o - | FileCheck %s

// CHECK-LABEL: @copy_ubuf_to_gm
// CHECK: a5vm.copy_ubuf_to_gm %arg0, %arg1, %[[ROWS:[^,]+]], %[[COLS:[^,]+]], %[[ZERO:[^,]+]], %[[NBURST:[^,]+]], %[[LEN:[^,]+]], %[[ZERO]], %[[DSTSTRIDE:[^,]+]], %[[SRCSTRIDE:[^ ]+]]
// CHECK-SAME: : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64, i64, i64
module {
  func.func @copy_ubuf_to_gm(%src: !pto.ptr<f32, ub>, %dst: !pto.ptr<f32, gm>) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %c4_i64 = arith.constant 4 : i64
    a5vm.copy_ubuf_to_gm %src, %dst, %c32_i64, %c32_i64, %c0_i64, %c32_i64, %c128_i64, %c0_i64, %c4_i64, %c128_i64 : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64, i64, i64
    return
  }
}

// CHECK: error: 'a5vm.copy_ubuf_to_gm' op requires UB source and GM destination
module {
  func.func @copy_ubuf_to_gm_wrong_direction(%src: !pto.ptr<f32, gm>, %dst: !pto.ptr<f32, ub>) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %c4_i64 = arith.constant 4 : i64
    a5vm.copy_ubuf_to_gm %src, %dst, %c32_i64, %c32_i64, %c0_i64, %c32_i64, %c128_i64, %c0_i64, %c4_i64, %c128_i64 : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i64, i64, i64
    return
  }
}

// CHECK: error: 'a5vm.copy_ubuf_to_gm' op requires source and destination element byte widths to match
module {
  func.func @copy_ubuf_to_gm_width_mismatch(%src: !pto.ptr<f32, ub>, %dst: !pto.ptr<i8, gm>) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %c4_i64 = arith.constant 4 : i64
    a5vm.copy_ubuf_to_gm %src, %dst, %c32_i64, %c32_i64, %c0_i64, %c32_i64, %c128_i64, %c0_i64, %c4_i64, %c128_i64 : !pto.ptr<f32, ub>, !pto.ptr<i8, gm>, i64, i64, i64, i64, i64, i64, i64, i64
    return
  }
}
