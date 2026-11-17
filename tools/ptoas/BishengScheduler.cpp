// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Support/CodeConstants.h"
#include "BishengScheduler.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>

namespace {
struct StackUsage {
    std::map<std::string, uint64_t> functions;
    uint64_t total = 0;
};

bool addUsageLine(llvm::StringRef line, StackUsage& usage)
{
    auto [name, properties] = line.split(": Stack size:");
    auto [bytes, remainder] = properties.trim().split(" bytes");
    uint64_t size = 0;
    bool invalidNumber = bytes.getAsInteger(10, size);
    bool invalidFields = name.trim().empty() || !properties.contains(" bytes") ||
                         (!remainder.empty() && !remainder.starts_with(","));
    if (invalidFields || invalidNumber) {
        return false;
    }
    auto [entry, inserted] = usage.functions.emplace(name.trim().str(), size);
    if (!inserted) {
        // Bisheng can repeat an identical VF report. Never count it twice.
        return entry->second == size;
    }
    uint64_t available = std::numeric_limits<uint64_t>::max() - usage.total;
    if (size > available) {
        return false;
    }
    usage.total += size;
    return true;
}

std::optional<StackUsage> readStackUsage(llvm::StringRef logPath)
{
    auto buffer = llvm::MemoryBuffer::getFile(logPath);
    if (!buffer) {
        return std::nullopt;
    }
    StackUsage usage;
    llvm::StringRef remaining = buffer.get()->getBuffer();
    while (!remaining.empty()) {
        auto [line, tail] = remaining.split('\n');
        remaining = tail;
        llvm::StringRef report = line.trim();
        if (!report.consume_front("[BISHENG] SIMD VF Function properties for ")) {
            continue;
        }
        if (!addUsageLine(report, usage)) {
            return std::nullopt;
        }
    }
    if (usage.functions.empty()) {
        return std::nullopt;
    }
    return usage;
}

bool sameFunctions(const StackUsage& on, const StackUsage& off)
{
    bool sameCount = on.functions.size() == off.functions.size();
    if (!sameCount) {
        return false;
    }
    for (const auto& entry : on.functions) {
        bool found = off.functions.count(entry.first) != 0;
        if (!found) {
            return false;
        }
    }
    return true;
}

bool selectOffObject(llvm::StringRef offObject, llvm::StringRef output, llvm::raw_ostream& diagnostics)
{
    if (std::error_code error = llvm::sys::fs::copy_file(offObject, output)) {
        diagnostics << "Error: cannot select Bisheng off object: " << error.message() << "\n";
        return false;
    }
    return true;
}

bool compareAndSelect(
    const StackUsage& on, llvm::StringRef offLog, llvm::StringRef offObject, llvm::StringRef output,
    llvm::raw_ostream& diagnostics)
{
    auto off = readStackUsage(offLog);
    if (!off || !sameFunctions(on, *off)) {
        diagnostics << "Warning: Bisheng scheduler auto: off SIMD VF stack report unavailable or "
                       "incomparable; keeping on.\n";
        return true;
    }
    bool selectOff = off->total < on.total;
    diagnostics << "Bisheng scheduler auto: SIMD VF stack bytes on=" << on.total << ", off=" << off->total
                << "; selected " << (selectOff ? "off" : "on") << ".\n";
    if (selectOff) {
        return selectOffObject(offObject, output, diagnostics);
    }
    return true;
}

bool reportRetrySetupFailure(std::error_code error, bool keepOn, llvm::raw_ostream& diagnostics)
{
    diagnostics << (keepOn ? "Warning: Bisheng scheduler auto: " : "Error: Bisheng scheduler auto: ")
                << error.message() << (keepOn ? "; keeping on.\n" : "; cannot retry off.\n");
    return keepOn;
}

void reportRecoveredFailure(llvm::StringRef messages, llvm::raw_ostream& diagnostics)
{
    llvm::StringRef summary = messages.trim().split('\n').first;
    while (!messages.empty()) {
        auto [line, tail] = messages.split('\n');
        messages = tail;
        bool failureLine = line.contains("error:") || line.contains("LLVM ERROR") ||
                           line.contains("Assertion") || line.contains("Unknown command line argument");
        if (failureLine) {
            summary = line.trim();
            break;
        }
    }
    if (!summary.empty()) {
        diagnostics << "Warning: recovered Bisheng on compilation failure: " << summary << "\n";
    }
}

void reportReportUnavailable(llvm::raw_ostream& diagnostics)
{
    diagnostics << "Warning: Bisheng scheduler auto: SIMD VF stack report unavailable; "
                   "keeping on, so auto behaves like --bisheng-vec-misched=on and no "
                   "stack size is compared. Check that this Bisheng toolchain can emit "
                   "SIMD VF stack reports with -mllvm --cce-res-usage.\n";
}

bool retryWithoutScheduler(
    const StackUsage* on, llvm::StringRef objectPath, mlir::pto::CompileBishengVariant compile,
    llvm::raw_ostream& diagnostics)
{
    bool keepOn = on != nullptr;
    llvm::SmallString<mlir::pto::kValue128> offObject;
    llvm::SmallString<mlir::pto::kValue128> offLog;
    if (std::error_code error = llvm::sys::fs::createTemporaryFile("ptoas-bisheng-off", "o", offObject)) {
        return reportRetrySetupFailure(error, keepOn, diagnostics);
    }
    llvm::FileRemover objectCleanup(offObject);
    if (std::error_code error = llvm::sys::fs::createTemporaryFile("ptoas-bisheng-off", "log", offLog)) {
        return reportRetrySetupFailure(error, keepOn, diagnostics);
    }
    llvm::FileRemover logCleanup(offLog);
    std::string retryErrors;
    llvm::raw_string_ostream retryDiagnostics(retryErrors);
    // After an on failure there is no stack report to compare, so a successful
    // off compile is sufficient and does not need resource-reporting support.
    if (!compile(false, keepOn, offObject, offLog, retryDiagnostics)) {
        diagnostics << (keepOn ? "Warning: Bisheng scheduler auto: retry failed; keeping on.\n"
                               : "Error: Bisheng scheduler auto: both on and off compilation failed.\n")
                    << "Bisheng off compilation diagnostics:\n" << retryErrors;
        return keepOn;
    }
    diagnostics << retryErrors;
    if (on) {
        return compareAndSelect(*on, offLog, offObject, objectPath, diagnostics);
    }
    if (!selectOffObject(offObject, objectPath, diagnostics)) {
        return false;
    }
    diagnostics << "Bisheng scheduler auto: on compilation failed; selected off.\n";
    return true;
}
} // namespace

mlir::pto::BishengSchedulerMode mlir::pto::getBishengSchedulerModeForTarget(
    BishengSchedulerMode mode, bool vectorTarget)
{
    if (vectorTarget || mode == BishengSchedulerMode::On) {
        return mode;
    }
    return BishengSchedulerMode::Off;
}

bool mlir::pto::compileWithBishengScheduler(
    BishengSchedulerMode mode, llvm::StringRef objectPath, llvm::StringRef logPath, CompileBishengVariant compile,
    llvm::raw_ostream& diagnostics)
{
    bool automatic = mode == BishengSchedulerMode::Auto;
    if (!automatic) {
        return compile(mode != BishengSchedulerMode::Off, false, objectPath, logPath, diagnostics);
    }
    std::string onMessages;
    llvm::raw_string_ostream onDiagnostics(onMessages);
    if (!compile(true, true, objectPath, logPath, onDiagnostics)) {
        // Asking for --cce-res-usage can fail on a toolchain without the resource
        // report while scheduler-on compilation itself works. Retry on without
        // the report first, so a missing report capability neither degrades auto
        // to off nor masquerades as a scheduler-on compilation failure.
        std::string schedulerOnMessages;
        llvm::raw_string_ostream schedulerOnDiagnostics(schedulerOnMessages);
        if (compile(true, false, objectPath, logPath, schedulerOnDiagnostics)) {
            diagnostics << schedulerOnMessages;
            reportReportUnavailable(diagnostics);
            return true;
        }
        diagnostics << "Warning: Bisheng scheduler auto: on compilation failed; retrying off.\n";
        bool recovered = retryWithoutScheduler(nullptr, objectPath, compile, diagnostics);
        if (recovered) {
            reportRecoveredFailure(schedulerOnMessages, diagnostics);
        } else {
            diagnostics << "Bisheng on compilation diagnostics:\n" << schedulerOnMessages;
        }
        return recovered;
    }
    diagnostics << onMessages;
    auto on = readStackUsage(logPath);
    if (!on) {
        reportReportUnavailable(diagnostics);
        return true;
    }
    if (on->total == 0) {
        diagnostics << "Bisheng scheduler auto: SIMD VF stack bytes on=0; "
                       "selected on without retry.\n";
        return true;
    }
    return retryWithoutScheduler(&*on, objectPath, compile, diagnostics);
}
