// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/SoftLibService.h"

#include <mutex>

namespace {
std::mutex &runtimeMutex() {
  static std::mutex mutex;
  return mutex;
}
std::shared_ptr<mlir::pto::SoftLibService> &runtimeService() {
  static std::shared_ptr<mlir::pto::SoftLibService> service;
  return service;
}
} // namespace

void mlir::pto::SoftLibRuntime::install(std::shared_ptr<SoftLibService> service) {
  std::lock_guard<std::mutex> lock(runtimeMutex());
  runtimeService() = std::move(service);
}

void mlir::pto::SoftLibRuntime::uninstall(SoftLibService *service) {
  std::lock_guard<std::mutex> lock(runtimeMutex());
  if (runtimeService().get() == service)
    runtimeService().reset();
}

std::shared_ptr<mlir::pto::SoftLibService>
mlir::pto::SoftLibRuntime::getService() {
  std::lock_guard<std::mutex> lock(runtimeMutex());
  return runtimeService();
}
