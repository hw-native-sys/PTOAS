// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "ptobc/ptobc_format.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>

#include <PTO/IR/PTO.h>

#include <iostream>
#include <optional>
#include <vector>

namespace ptobc {
mlir::OwningOpRef<mlir::ModuleOp> parsePTOFile(mlir::MLIRContext& ctx, const std::string& path);
PTOBCFile encodeFromMLIRModule(mlir::ModuleOp module);
void decodeFileToPTO(const std::string& inPath, const std::string& outPath);
}

static void usage() {
  std::cerr << "ptobc (v0)\n\n"
            << "Usage:\n"
            << "  ptobc encode <input.pto> -o <out.ptobc>\n"
            << "  ptobc decode <input.ptobc> -o <out.pto>\n";
}

struct CommandLineOptions {
  std::string cmd;
  std::string input;
  std::string output;
};

namespace {

constexpr size_t kCommandArgumentCount = 2;
constexpr size_t kFullCommandArgumentCount = 5;
constexpr size_t kCommandArgumentIndex = 1;
constexpr size_t kInputArgumentIndex = 2;
constexpr size_t kFirstOptionArgumentIndex = 3;
constexpr size_t kNextArgumentOffset = 1;
constexpr int kUsageExitCode = 2;

static std::vector<std::string> collectArguments(int argc, char *argv[]) {
  std::vector<std::string> args;
  args.reserve(static_cast<size_t>(argc));
  for (int i = 0; i < argc; ++i) {
    args.emplace_back(argv[i]);
  }
  return args;
}

} // namespace

static std::optional<CommandLineOptions>
parseCommandLine(const std::vector<std::string> &args) {
  if (args.size() < kCommandArgumentCount) {
    return std::nullopt;
  }

  CommandLineOptions options{args[kCommandArgumentIndex], "", ""};
  if (options.cmd != "encode" && options.cmd != "decode") {
    return options;
  }
  if (args.size() < kFullCommandArgumentCount) {
    return std::nullopt;
  }

  options.input = args[kInputArgumentIndex];
  for (size_t i = kFirstOptionArgumentIndex; i < args.size(); ++i) {
    const std::string &arg = args[i];
    if (arg == "-o" && i + kNextArgumentOffset < args.size()) {
      options.output = args[++i];
    }
  }
  if (options.output.empty()) {
    return std::nullopt;
  }
  return options;
}

static mlir::DialectRegistry buildRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::pto::PTODialect>();
  return registry;
}

static void preloadDialects(mlir::MLIRContext &ctx) {
  (void)ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  (void)ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  (void)ctx.getOrLoadDialect<mlir::affine::AffineDialect>();
  (void)ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  (void)ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  (void)ctx.getOrLoadDialect<mlir::pto::PTODialect>();
}

static int runEncode(const CommandLineOptions &options) {
  mlir::MLIRContext ctx(buildRegistry());
  ctx.allowUnregisteredDialects(true);
  preloadDialects(ctx);

  auto module = ptobc::parsePTOFile(ctx, options.input);
  auto file = ptobc::encodeFromMLIRModule(*module);
  auto bytes = file.serialize();
  ptobc::writeFile(options.output, bytes);
  return 0;
}

static int runDecode(const CommandLineOptions &options) {
  ptobc::decodeFileToPTO(options.input, options.output);
  return 0;
}

int main(int argc, char **argv) {
  auto args = collectArguments(argc, argv);
  auto options = parseCommandLine(args);
  if (!options) {
    usage();
    return kUsageExitCode;
  }

  try {
    if (options->cmd == "encode") {
      return runEncode(*options);
    }
    if (options->cmd == "decode") {
      return runDecode(*options);
    }
    usage();
    return kUsageExitCode;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
