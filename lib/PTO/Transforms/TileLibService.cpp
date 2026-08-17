// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/TileLibService.h"

#include <mutex>

namespace {

std::mutex &getRuntimeMutex() {
  static std::mutex mutex;
  return mutex;
}

std::shared_ptr<mlir::pto::TileLibService> &getRuntimeService() {
  static std::shared_ptr<mlir::pto::TileLibService> service;
  return service;
}

} // namespace

void mlir::pto::TileLibRuntime::install(
    std::shared_ptr<TileLibService> service) {
  std::lock_guard<std::mutex> lock(getRuntimeMutex());
  getRuntimeService() = std::move(service);
}

void mlir::pto::TileLibRuntime::uninstall(TileLibService *service) {
  std::lock_guard<std::mutex> lock(getRuntimeMutex());
  if (getRuntimeService().get() == service) {
    getRuntimeService().reset();
  }
}

std::shared_ptr<mlir::pto::TileLibService>
mlir::pto::TileLibRuntime::getService() {
  std::lock_guard<std::mutex> lock(getRuntimeMutex());
  return getRuntimeService();
}
