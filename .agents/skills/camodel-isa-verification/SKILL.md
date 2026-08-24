---
name: camodel-isa-verification
description: Create, run, and analyze PTO-ISA ST tests on the CANN CA model simulator. Use when verifying A5/Ascend PTO-ISA instruction behavior, inspecting simulator instruction logs, measuring vector instruction latency, or comparing UB dump hex output against expected values.
---

# CA Model ISA Verification

## Overview

Use this skill to verify PTO-ISA instructions with the CANN cycle-accurate CA model. Work from the PTO-ISA checkout, create or select an ST test, run it in simulator mode, then inspect instruction and UB dump logs for latency and correctness.

## Environment

Run from Linux or WSL with CANN initialized:

```bash
cd ~/pto-isa
source /usr/local/Ascend/cann/set_env.sh
```

For CANN 9.0.0-alpha.1 namespace issues, check whether `TCvt.hpp` already uses `__cce_simd::RoundRType`:

```bash
grep '__cce_simd::RoundRType' ~/pto-isa/include/pto/npu/a5/TCvt.hpp
```

If the fix is missing, update the affected `using ::RoundRType;` style declarations to the `__cce_simd::` namespace before building.

## Create Or Select A Test

ST tests normally live under `tests/st/` and use `t<operation>` names such as `tadd`, `tmul`, `tload`, `tgather`, and `tcvt`.

Each test directory usually contains:

```text
tests/st/tadd/
  CMakeLists.txt
  test_tadd.cpp
  inc/test_tadd.hpp
```

To create a new operation test, copy the closest existing test and rename the suite, files, operation include, parameters, and `CMakeLists.txt` target:

```bash
cp -r tests/st/tadd tests/st/tnewop
```

Keep the case small enough to isolate the instruction or addressing behavior being verified.

## Run On CA Model

Run a full ST test:

```bash
python3 tests/script/run_st.py -r sim -v a5 -t <testcase>
```

Run one GoogleTest case:

```bash
python3 tests/script/run_st.py -r sim -v a5 -t <testcase> -g "<TestSuite>.<case_name>"
```

Capture a run log for later inspection:

```bash
python3 tests/script/run_st.py -r sim -v a5 -t tmul 2>&1 | tee logs/tmul.log
```

Use `a5` for the A5 architecture. Use `Ascend910_9599` only when the full chip model is required.

## Locate Simulator Logs

After a simulator run, inspect files under `build/tests/st/<testcase>/`:

```text
core0.veccore0.instr_log.dump
core0.veccore0.instr_popped_log.dump
core0.veccore0.ub.rd_log.dump
core0.veccore0.ub.wr_log.dump
```

Copy logs for a report or deeper analysis:

```bash
mkdir -p ~/npu_skills/pto-isa/verification/logs/<test_name>
cp build/tests/st/<testcase>/core0.veccore0.*.dump \
  ~/npu_skills/pto-isa/verification/logs/<test_name>/
```

## Analyze Instruction Latency

The instruction log contains cycle, PC, pipe, binary encoding, instruction ID, mnemonic, and operands. Focus on dependent instruction pairs, not isolated timestamps.

Common pipe and latency expectations:

| Instruction | Pipe | Typical latency |
| --- | --- | --- |
| `RV_VLD` normal load | `RVECLD` | 9 cycles |
| `RV_VST` normal store | `RVECST` | 9 cycles |
| `RV_VADD` F32 | `RVECEX` | 7 cycles |
| `RV_VMUL` F32 | `RVECEX` | 7 cycles |
| `RV_VGATHER2` B32 | `RVECLD` | 28 cycles |

Extract the relevant trace:

```bash
LOG=build/tests/st/<testcase>/core0.veccore0.instr_log.dump
grep -E "RV_VLD|RV_VADD|RV_VMUL|RV_VST|RV_VGATHER2" "$LOG" | head -40
```

For a load-compute-store chain, compute latency from the producing instruction cycle to the first dependent consumer cycle. For example, if `RV_VLD` appears at cycle 2664 and dependent `RV_VADD` appears at 2669, the observed dependency gap is 5 cycles in that trace.

Find one instruction by simulator ID:

```bash
grep "ID: 000115" core0.veccore0.instr_log.dump
```

Use the `Binary: 0x...` field with the A5 ISA reference when a low-level decode is needed.

## Verify UB Dumps

Use `core0.veccore0.ub.wr_log.dump` to verify output buffers. Check that write addresses match the expected output region, then decode the hex values.

Common float32 encodings:

| Hex | Float32 |
| --- | --- |
| `3f800000` | 1.0 |
| `40000000` | 2.0 |
| `40400000` | 3.0 |
| `40800000` | 4.0 |
| `40a00000` | 5.0 |
| `40c00000` | 6.0 |
| `40e00000` | 7.0 |
| `41000000` | 8.0 |
| `41100000` | 9.0 |

Use a small Python snippet when many values must be checked:

```python
import struct

def hex_to_float(hex_str):
    return struct.unpack("!f", bytes.fromhex(hex_str))[0]

def verify_ub_dump(log_path, expected_values, start_addr=0):
    with open(log_path) as f:
        for line in f:
            if "Address" not in line:
                continue
            for part in line.split("Address")[1:]:
                if "=" not in part:
                    continue
                addr_hex, val_hex = part.split("=")
                addr = int(addr_hex.strip(), 16)
                val = hex_to_float(val_hex.strip().strip("[]"))
                idx = (addr - start_addr) // 4
                if 0 <= idx < len(expected_values):
                    if abs(val - expected_values[idx]) > 1e-5:
                        print(f"MISMATCH {addr:08x}: got {val}, expected {expected_values[idx]}")
                    else:
                        print(f"OK {addr:08x}: {val}")
```

## Quick Checks

- `NOT SUPPORT TASK TYPE 3` is usually a simulator warning and can often be ignored.
- If a test hangs, increase timeout or reduce tile size to isolate the instruction.
- If no dump files appear, confirm `-r sim` was used and inspect `build/tests/st/<testcase>/`.
- If namespace build errors appear on CANN 9.0.0-alpha.1, re-check the `TCvt.hpp` namespace fix.
