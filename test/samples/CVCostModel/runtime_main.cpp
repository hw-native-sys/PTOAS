// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <acl/acl.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void launch_cv(float *, float *, float *, float *, void *);

namespace {
constexpr size_t kGuard = 64;
constexpr float kGuardValue = 1234567.0F;
constexpr float kSentinel = -7654321.0F;

void check(aclError code, const char *action) {
  if (code != ACL_SUCCESS) {
    throw std::runtime_error(std::string(action) + " failed: " + std::to_string(code));
  }
}

struct Runtime {
  aclrtStream stream = nullptr;
  std::array<void *, 4> allocations = {};
  bool initialized = false;
  bool selected = false;
  int device = 0;
  ~Runtime() {
    if (stream != nullptr) {
      aclrtDestroyStream(stream);
    }
    for (void *p : allocations) {
      if (p != nullptr) {
        aclrtFree(p);
      }
    }
    if (selected) {
      aclrtResetDevice(device);
    }
    if (initialized) {
      aclFinalize();
    }
  }
  void finish() {
    check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    stream = nullptr;
    for (void *&p : allocations) {
      check(aclrtFree(p), "aclrtFree");
      p = nullptr;
    }
    check(aclrtResetDevice(device), "aclrtResetDevice");
    selected = false;
    check(aclFinalize(), "aclFinalize");
    initialized = false;
  }
  void start(int index) {
    check(aclInit(nullptr), "aclInit");
    initialized = true;
    device = index;
    check(aclrtSetDevice(index), "aclrtSetDevice");
    selected = true;
    check(aclrtCreateStream(&stream), "aclrtCreateStream");
  }
};

std::vector<float> read(const std::string &path, size_t count) {
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path.c_str(), "rb"), &std::fclose);
  if (!file) {
    throw std::runtime_error("cannot open input");
  }
  std::vector<float> data(count);
  const size_t actual = std::fread(data.data(), sizeof(float), count, file.get());
  const int extra = std::fgetc(file.get());
  const int closed = std::fclose(file.release());
  if (actual != count || extra != EOF || closed != 0) {
    throw std::runtime_error("input length/read failure");
  }
  return data;
}

using Inputs = std::array<std::vector<float>, 4>;

void save(const char *path, const float *data, size_t count) {
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path, "wb"), &std::fclose);
  if (!file) {
    throw std::runtime_error("cannot save output");
  }
  const size_t written = std::fwrite(data, sizeof(float), count, file.get());
  const int closed = std::fclose(file.release());
  if (written != count || closed != 0) {
    throw std::runtime_error("output write failure");
  }
}

Inputs prepare(Runtime &runtime, size_t count, const std::string &inputs) {
  const size_t total = count + 2 * kGuard;
  const size_t bytes = total * sizeof(float);
  const std::array<std::string, 4> names = {"q", "k", "v", "output"};
  Inputs initial;
  for (size_t i = 0; i < initial.size(); ++i) {
    initial[i].assign(total, kGuardValue);
    std::vector<float> payload(count, kSentinel);
    if (i < 3) {
      payload = read(inputs + "/" + names[i] + ".bin", count);
    }
    std::copy(payload.begin(), payload.end(), initial[i].begin() + kGuard);
    check(aclrtMalloc(&runtime.allocations[i], bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
    check(aclrtMemcpy(runtime.allocations[i], bytes, initial[i].data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
  }
  return initial;
}

bool verify(Runtime &runtime, const Inputs &initial, const std::vector<float> &golden) {
  const size_t count = golden.size();
  const size_t total = count + 2 * kGuard;
  const size_t bytes = total * sizeof(float);
  for (size_t i = 0; i < initial.size(); ++i) {
    std::vector<float> actual(total);
    check(aclrtMemcpy(actual.data(), bytes, runtime.allocations[i], bytes, ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
    for (size_t j = 0; j < total; ++j) {
      const bool output = i == 3 && j >= kGuard && j < count + kGuard;
      const float expected = output ? golden[j - kGuard] : initial[i][j];
      // Both independent golden and device arithmetic produce exact FP32 integers.
      if (actual[j] != expected) {
        std::fprintf(stderr, "mismatch buffer=%zu element=%zu expected=%g actual=%g\n", i, j, expected, actual[j]);
        save("mismatch.bin", actual.data(), total);
        return false;
      }
    }
    if (i == 3) {
      save("output.bin", actual.data() + kGuard, count);
    }
  }
  return true;
}

int run(int device, size_t iterations, const std::string &inputs) {
  const size_t count = iterations * 1024;
  Runtime runtime;
  runtime.start(device);
  const char *soc = aclrtGetSocName();
  const bool supportedSoc = soc != nullptr && std::string(soc).find("950") != std::string::npos;
  if (!supportedSoc) {
    throw std::runtime_error("unexpected device SOC");
  }
  std::printf("soc=%s\n", soc);
  std::fflush(stdout);
  const auto initial = prepare(runtime, count, inputs);
  launch_cv(static_cast<float *>(runtime.allocations[0]) + kGuard,
            static_cast<float *>(runtime.allocations[1]) + kGuard,
            static_cast<float *>(runtime.allocations[2]) + kGuard,
            static_cast<float *>(runtime.allocations[3]) + kGuard, runtime.stream);
  check(aclrtSynchronizeStream(runtime.stream), "aclrtSynchronizeStream");
  const bool passed = verify(runtime, initial, read(inputs + "/golden.bin", count));
  runtime.finish();
  if (!passed) {
    return 3;
  }
  std::puts("GOLDEN_PASS guards=pass input_unchanged=pass");
  return 0;
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 4) {
      throw std::runtime_error("usage: runner DEVICE ITERATIONS INPUT_DIRECTORY");
    }
    size_t consumed = 0;
    const int device = std::stoi(argv[1], &consumed);
    const bool deviceParsed = consumed == std::string(argv[1]).size();
    if (!deviceParsed || device < 0 || device > 15) {
      throw std::runtime_error("invalid device");
    }
    const int iterations = std::stoi(argv[2], &consumed);
    const bool countParsed = consumed == std::string(argv[2]).size();
    const bool countSupported = iterations == 1 || iterations == 2 || iterations == 4 || iterations == 8;
    if (!countParsed || !countSupported) {
      throw std::runtime_error("invalid iterations");
    }
    return run(device, static_cast<size_t>(iterations), argv[3]);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 2;
  }
}
