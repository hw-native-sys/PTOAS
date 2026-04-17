// RUN: ptoas %s | FileCheck %s

module {
  func.func @single_kernel(%arg0: !pto.ptr<f32>) {
    return
  }
}

// CHECK-LABEL: __global__ AICORE void single_kernel(
