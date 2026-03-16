// RUN: not ptoas %s 2>&1 | FileCheck %s

module {
  func.func @helper() {
    return
  }

  func.func @kernel() {
    return
  }
}

// CHECK: module with multiple function definitions requires at least one `pto.entry` function
