#pragma once

#include <cstdint>

struct TensorBuffer {
  void *addr;
};

struct Tensor {
  TensorBuffer buffer;
  int64_t start_offset;
};
