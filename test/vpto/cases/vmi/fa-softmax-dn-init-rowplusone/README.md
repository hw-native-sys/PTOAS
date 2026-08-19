# fa-softmax-dn-init-rowplusone 复现速查

> 在本 case 上跑 VMI 融合 / 非 VMI 两路，以及带 `--enable-vecscope-mem-bar` 的组合：性能（summary + 指令时间序列 + hazard 计数）、device 汇编（shim 捕获 IR → bisheng 出 `.s`）、MLIR（各 pass 中间 IR）。
>
> 本文档只记录"怎么跑 + 各产物的期待形态"。所有路径以仓库根为基准的相对路径表示（环境变量里的绝对路径按本机实际情况替换）。

## 0. 提示词（复制粘贴驱动跑产物）

以下提示词可粘给 AI 助手或人执行。每个独立成块，按需选用。执行前先 source §2 的环境变量。

### 0.1 跑性能（sim）

```
在 fa-softmax-dn-init-rowplusone case 上跑 VMI 融合和非 VMI 两路 sim 性能。
参照 README.md §2 准备环境、§3 跑 validation 脚本（切 flags + 跑完恢复）。
两路都跑完后，按 §3 的取数命令给出：rvec_busy、kernal ticks、mte2/mte3 busy cycle、
hazard 总数及类型分解（overlaps with）、compare 是否 PASS。
把 core0_summary_log、core0.veccore0.instr_log.dump、validation.log 三类文件存档到
log/ 下。若要看 membar pass 效果，额外加一路 --enable-vecscope-mem-bar 的组合。
```

期待产物：`core0_summary_log`（SU busy cycle + ticks）、`core0.veccore0.instr_log.dump`（逐指令时间序列）、`validation.log`（含 hazard warning 行）。hazard 数与 compare PASS/FAIL 是正确性关键指标。

### 0.2 出汇编（device IR + `.s`）

```
参照 README.md §5 构造 shim capture 工具，捕获 VMI / 非 VMI 两路真正喂给 bisheng 的
device IR（device_input.ll）。然后用 §5.3 的 bisheng 命令编出 aicore .s（可读汇编）
和 .o（校验用）。验证 .o 是 ELF arch 0x1029。
如需 vf-fusion 关闭的汇编对照，按 §5.4 用 -mllvm 后端选项（7 个，false 分支）出 .s，
并对照 vf-fusion on 版本。两版 .s 行数和 SMEM_BAR 数会不同。
```

期待产物：`vmi_device_input.ll` / `novmi_device_input.ll`（shim 捕获的 IR）、`rpo_vmi.s` / `rpo_novmi.s`（可读汇编）、`*.aicore.o`（ELF 0x1029，`.text` size 校验）。vf-fusion off 版 `.s` 用 `output_vfoff.s` 命名，对照 `output_vfon.s`。

### 0.3 出中间 MLIR

```
参照 README.md §7，用 --mlir-print-ir-after-all 先抓全 pipeline pass 列表，再用
--mlir-print-ir-after=<pass-arg> dump 单个 pass 后的 IR。重点：dump
pto-insert-vecscope-mem-bar 的 before（PTOInferVPTOVecScope 后）和 after，看
pto.mem_bar 插在哪、什么 kind。也 dump vmi-loop-fusion / vmi-load-store-elision 看
融合/消除效果。
```

期待产物：各 pass 的 `.mlir` 文件。membar pass 前后 diff 应只有 `pto.mem_bar` 行的插入（无指令重排）。

## 1. 路由

- **VMI 路**：`--enable-vmi --enable-op-fusion=true`。ptoas 做 VF 融合（`VmiLoopFusion` + `VmiLoadStoreElision`）。
- **非 VMI 路**：`--enable-op-fusion=false`。全 fusion 关闭。

两者互斥。`--enable-vmi` 单独置 true 无效，必须配 `--enable-op-fusion=true` 才进 VMI 路径。

### 1.1 注意：bisheng 后端 VF-fusion 与 ptoas VMI 融合是两层独立的东西

`--enable-op-fusion` 只控制 **ptoas 层** 的 VMI 融合。bisheng 编 aicore 时还有它**自己**的 SIMD VF-fusion 流水线（`-cce-vf-enable-vf-fusion` 等 7 个后端 pass），二者独立：

| 路由 | ptoas VMI 融合 | bisheng VF-fusion |
|---|---|---|
| VMI 路 | ✓ 开 | ✗ **ptoas 强制关**（防二次优化内存流量） |
| 非 VMI 路 | ✗ 关 | ✓ **保留开启**（bisheng 默认） |

**所以"非 VMI 路"的 rvec busy 不是干净的"只关 ptoas 融合"对照**——它同时让 bisheng 自身的 VF-fusion 跑了起来，二者开销叠加。这点在做性能对照时必须清楚。

ptoas 在 VMI 路**自动**注入这 7 个 `-cce-vf-*=false`（`ObjectEmission.cpp` 的 `disableBishengVFFusion`，由 `useVMIFusionPipeline` 触发）；非 VMI 路不注入。可 strace 确认：

```bash
strace -f -e trace=execve -s 4000 -o /tmp/strace.log \
  "$PTOAS_BIN" --pto-arch=a5 --pto-backend=vpto --pto-level=level2 \
  --tile-lib-backend=ptodsl --enable-op-fusion=false \
  "$CASE/kernel.pto" -o /tmp/x.o 2>/dev/null
# aicore 那条 execve 应只有 --cce-auto-sync=off，没有 -cce-vf-*
grep 'execve.*bisheng' /tmp/strace.log | grep 'cce-bitcode-is-aicore' \
  | grep -oE '\-\-cce-auto-sync=[a-z]+|\-cce-vf-[a-z-]+=[a-z]+' | sort | uniq -c
```

**要做"ptoas 不融合 + bisheng VF 也关"的干净对照**，用独立开关 `--disable-bisheng-vf-fusion`（`ptoas.cpp` 的 `cl::opt`，与 VMI 无关）：

```bash
$PTOAS_BIN --pto-arch=a5 --pto-backend=vpto --pto-level=level2 \
  --tile-lib-backend=ptodsl --enable-op-fusion=false \
  --disable-bisheng-vf-fusion \
  "$CASE/kernel.pto" -o /tmp/rpo_novmi_vfoff.fatobj.o
```

> 仅供出 IR/汇编或 strace 看 argv 时直接调 ptoas（§4）。要走 sim 则仍得照 §3 临时改 `ptoas.flags`（validation 脚本读 flags 文件优先于 env），在 flags 里加 `--disable-bisheng-vf-fusion` 即可，跑完恢复。

## 2. 环境（每次新 shell 都要 source）

```bash
cd <PTOAS_REPO_ROOT>
export ASCEND_HOME_PATH=<your CANN root, e.g. .../cann-9.0.1>
source "$ASCEND_HOME_PATH/bin/setenv.bash"
export PTODSL_PYTHON_ROOT=<PTOAS_REPO_ROOT>/ptodsl
# 注意 PYTHONPATH 必须在 source setenv.bash 之后重置：setenv.bash 会把
# PYTHONPATH 设成 CANN 的 python/site-packages，且 ptoas 的 python wrapper 会
# 自己把 build/python insert 到 sys.path[0]，所以这里不要再放 build/python
# （否则 wrapper 见它已在 sys.path 就跳过 insert，反而让 ptodsl 下的占位
# ptoas 包优先于带 _core.so 的 build/python/ptoas）。
export PYTHONPATH=<PTOAS_REPO_ROOT>/ptodsl:${PYTHONPATH:-}
export PTO_ISA_ROOT=<your pto-isa root>
export PTOAS_BIN=<PTOAS_REPO_ROOT>/build/tools/ptoas/ptoas
CASE=test/vpto/cases/vmi/fa-softmax-dn-init-rowplusone
```

> build 用 `build/tools/ptoas/ptoas`（python wrapper + `runtime-staging/lib/ptoas.so`）。重编：`cmake --build build -j4`，同步：`cp -f build/python/ptoas/mlir/_mlir_libs/libPTOASCompiler.so ~/.local/lib/python3.12/site-packages/ptoas/mlir/_mlir_libs/`（或对应 editable install 路径）。

### 2.1 排错：`ptoas` 报 `No module named 'ptoas.mlir.ir'` / `cannot import name '_core'`

两类 import 失败根因不同，但都和 PYTHONPATH / stale 副本有关：

**A. `cannot import name '_core' from 'ptoas'`** —— `ptoas` 解析到了 `ptodsl/ptoas/`（只有占位 `__init__.py`，没有 `_core.so`）。原因：PYTHONPATH 里把 `build/python` 放在了 `ptodsl` 后面，或两个目录都放了。修复：**PYTHONPATH 只放 `ptodsl`**（见 §2），让 wrapper 自己把 `build/python` insert 到 sys.path[0]；不要在 PYTHONPATH 里出现 `build/python`。

**B. `No module named 'ptoas.mlir.ir'` / `initialization failed`** —— `build/python/mlir/` 这个**陈旧构建残留目录**（来自旧的 CMake 配置，当前 `lib/Bindings/Python/CMakeLists.txt` 已修正装到 `ptoas/mlir/`，但旧产物不会被 `git clean` 触及）作为顶层 `mlir` 命名空间包，劫持了 `ptoas/mlir/__init__.py` 里的 `import mlir`，导致 `ptoas.mlir` 被指到一个一个缺 `ir.py` 的空包。修复：

```bash
# 确认是 stale 残留（应缺 __init__.py / ir.py，只有 dialects/ + _mlir_libs/_pto.so）
ls build/python/mlir/ build/python/ptoas/mlir/
# 直接删掉 stale 残留（gitignore 产物，删后干净重编不会再生成）
rm -rf build/python/mlir
# 验证
"$PTOAS_BIN" --version   # 应输出 ptoas <ver>
```

> 该残留只在 build 树被旧配置写过时存在；`rm -rf build/ && cmake --build build` 后不会复现。

## 3. 两路性能（VPTO 路径，sim）

validation 脚本 `test/vpto/scripts/run_host_vpto_validation.sh` 读 case 的 `ptoas.flags`（**优先于** `PTOAS_FLAGS` 环境变量），所以切路径要临时改 flags、跑完恢复。case 默认是 VMI 路（`--enable-vmi --enable-op-fusion=true`）；非 VMI 路需临时改成 `--enable-op-fusion=false`（去掉 `--enable-vmi`）。

```bash
cd <PTOAS_REPO_ROOT>
# 接 §1 的环境变量
CASE_DIR=test/vpto/cases/vmi/fa-softmax-dn-init-rowplusone
FLAGS_FILE="$CASE_DIR/ptoas.flags"
cp "$FLAGS_FILE" /tmp/rpo_flags.bak
trap 'cp /tmp/rpo_flags.bak "$FLAGS_FILE"' EXIT   # 退出必恢复

# --- 测1：VMI 融合（case 默认 flags，直接跑即可）---
WS=/tmp/fa_rpo_t1; rm -rf "$WS"; mkdir -p "$WS"
WORK_SPACE="$WS" CASES_ROOT=$PWD/test/vpto/cases/vmi \
  ASCEND_HOME_PATH="$ASCEND_HOME_PATH" PTOAS_BIN="$PTOAS_BIN" \
  CASE_NAME=fa-softmax-dn-init-rowplusone DEVICE=SIM \
  PTODSL_SIM_SOC_VERSION=Ascend950PR_9599 \
  bash test/vpto/scripts/run_host_vpto_validation.sh > /tmp/rpo_t1.run.log 2>&1

# --- 测2：非 VMI（临时改成 --enable-op-fusion=false）---
echo "--pto-arch a5 --pto-backend=vpto --tile-lib-backend=ptodsl --enable-op-fusion=false" > "$FLAGS_FILE"
WS=/tmp/fa_rpo_t2; rm -rf "$WS"; mkdir -p "$WS"
WORK_SPACE="$WS" CASES_ROOT=$PWD/test/vpto/cases/vmi \
  ASCEND_HOME_PATH="$ASCEND_HOME_PATH" PTOAS_BIN="$PTOAS_BIN" \
  CASE_NAME=fa-softmax-dn-init-rowplusone DEVICE=SIM \
  PTODSL_SIM_SOC_VERSION=Ascend950PR_9599 \
  bash test/vpto/scripts/run_host_vpto_validation.sh > /tmp/rpo_t2.run.log 2>&1

cp /tmp/rpo_flags.bak "$FLAGS_FILE"   # 手动恢复（trap 兜底）
```

**取数 + 查 hazard**：

> 产物目录名 = `case_name` 本身（脚本 `case_output_token` 把空格/斜杠换下划线），**没有** `vmi_` 前缀；即 `$WS/fa-softmax-dn-init-rowplusone/`。

```bash
for n in 1 2; do
  WS=/tmp/fa_rpo_t$n
  DIR="$WS/fa-softmax-dn-init-rowplusone"   # 目录名 = CASE_NAME，无 vmi_ 前缀
  SUMMARY="$DIR/core0_summary_log"
  INSTLOG="$DIR/core0.veccore0.instr_log.dump"
  echo "=== 测$n ==="
  grep -E "rvec_veccore0_busy_cycle|kernal total ticks|mte2_veccore0_su_busy_cycle|mte3_veccore0_su_busy_cycle" "$SUMMARY"
  echo "hazard count: $(grep -c 'overlaps with' /tmp/rpo_t$n.run.log)"
  grep "overlaps with" /tmp/rpo_t$n.run.log \
    | sed -E 's/.*cur_instr (RV_[A-Z]+).*pre_instr (RV_[A-Z]+).*/\1 vs \2/' | sort | uniq -c
  grep -E "compare passed|nz compare" /tmp/rpo_t$n.run.log | tail -2
done
```

### 性能数据产物

两个文件都要存：

| 文件 | 内容 | 看什么 |
|---|---|---|
| `core0_summary_log` | 各 SU busy cycle 汇总 + kernal/system ticks | mte2/mte3/rvec busy cycle，宏观性能 |
| `core0.veccore0.instr_log.dump` | 每条指令的执行时间序列 | 逐指令分析：哪条 RV_VMAX/VLD/VST 占用、时刻、PC、二进制编码 |

`instr_log.dump` 格式（每行一条指令）：
```
[info] [00002676] (PC: 0x10d0d214) RVECEX   : (Binary: 0x82180782) (ID: 000103) RV_VMAX Dtype: F32
[info] [00002677] (PC: 0x10d0d210) RVECLD   : (Binary: 0x00280008) (ID: 000112) RV_VLD
```
方括号内是 tick 时刻，RVECEX/RVECLD/SCALAR 是执行单元，ID 是指令序号。

存档两路性能数据：
```bash
DEST=log/fa_vmi_rowplusone_full
mkdir -p "$DEST"
for n in 1 2; do  # 1=VMI, 2=非VMI
  DIR=/tmp/fa_rpo_t$n/fa-softmax-dn-init-rowplusone   # 目录名 = CASE_NAME，无 vmi_ 前缀
  tag=$([ "$n" = 1 ] && echo vmi || echo novmi)
  cp "$DIR/core0_summary_log" "$DEST/perf_${tag}_core0_summary.txt"
  cp "$DIR/core0.veccore0.instr_log.dump" "$DEST/perf_${tag}_instr_log.dump"
done
```

## 4. 直接 ptoas 出 IR / 汇编（不走 validation 脚本）

只看 IR 或汇编、不需要跑 sim 时，直接调 ptoas（绕过 `ptoas.flags` 临时改文件）：

```bash
# VMI 路：全量 MLIR（每个 pass 后的 IR）
$PTOAS_BIN --pto-arch=a5 --pto-backend=vpto --pto-level=level2 \
  --tile-lib-backend=ptodsl --enable-vmi --enable-op-fusion=true \
  --mlir-print-ir-after-all \
  "$CASE/kernel.pto" -o /tmp/rpo_vmi.fatobj.o > /tmp/rpo_vmi_mlir.log 2>&1

# VMI 路：最终汇编（VPTO→LLVM IR）
$PTOAS_BIN --pto-arch=a5 --pto-backend=vpto --pto-level=level2 \
  --tile-lib-backend=ptodsl --enable-vmi --enable-op-fusion=true \
  --emit-vpto-llvm-ir \
  "$CASE/kernel.pto" -o /tmp/rpo_vmi.ll

# 非 VMI 路：把 --enable-vmi --enable-op-fusion=true 换成 --enable-op-fusion=false
```

`--pto-level=level2` 必带（fusion 流水线在 level2/3 才跑；level1 不跑且有 warning）。

> `--emit-vpto-llvm-ir` 出的是 ptoas 内部 dump 的 IR，**不等于**真正喂给 bisheng 编 aicore 的 IR。要拿能编 aicore `.o` 的 device-only IR，用 §5 的 shim 捕获。

## 5. 汇编捕获（shim capture）：device IR + aicore `.o` / `.s`

ptoas 编 VPTO fatobj 时把 device LLVM IR 经 stdin 喂给 bisheng（带 `-cce-bitcode-is-aicore`）。用一个 shim 假 bisheng 把这路 stdin IR `tee` 落盘，就能拿到**真正喂给 bisheng 的那版 device IR**，再用真 bisheng 编出 aicore `.o` / `.s`。

### 5.1 构造 capture shim（一次性）

shim = 真 CANN 的全量软链，只把 `bin/bisheng` 换成捕获脚本：

```bash
CANN="$ASCEND_HOME_PATH"
SHIM=/tmp/perf/asm/shim_capture
rm -rf "$SHIM"; mkdir -p "$SHIM/bin"

# 顶层所有非 bin 目录软链到真 CANN
for d in "$CANN"/*/; do
  b=$(basename "$d"); [ "$b" = "bin" ] && continue
  ln -sfn "$d" "$SHIM/$b"
done
# bin/ 下除 bisheng 外全部软链
for e in "$CANN"/bin/*; do
  b=$(basename "$e"); [ "$b" = "bisheng" ] && continue
  ln -sfn "$e" "$SHIM/bin/$b"
done

# bin/bisheng = 捕获脚本（只在 -cce-bitcode-is-aicore + stdin 时落盘 IR，其余原样转发）
cat > "$SHIM/bin/bisheng" <<'EOF'
#!/usr/bin/env bash
set -o pipefail
real_bisheng="<your CANN>/bin/bisheng"
capture_file="${PTOAS_CAPTURE_LL:-/tmp/perf/asm/captured.ll}"
capture=0; stdin_ir=0
for arg in "$@"; do
  [[ "$arg" == "-cce-bitcode-is-aicore" ]] && capture=1
  [[ "$arg" == "-" ]] && stdin_ir=1
done
if [[ "$capture" == "1" && "$stdin_ir" == "1" ]]; then
  tee "$capture_file" | "$real_bisheng" "$@"
  exit "${PIPESTATUS[1]}"
fi
exec "$real_bisheng" "$@"
EOF
chmod +x "$SHIM/bin/bisheng"
```

> 必须软链整个 CANN 目录树（顶层所有 dir + bin 下除 bisheng 外所有条目）。漏 `tools/`（含 `bisheng_compiler/bin/bisheng` cc1 前端）报 `unable to locate bisheng cc1 frontend`。

### 5.2 捕获 device IR

```bash
cd <PTOAS_REPO_ROOT>
# 接 §1 的环境变量
PTOAS=build/tools/ptoas/ptoas
K=test/vpto/cases/vmi/fa-softmax-dn-init-rowplusone/kernel.pto
SHIM=/tmp/perf/asm/shim_capture

# VMI 路:
ART=/tmp/perf/asm/vmi; rm -rf "$ART"; mkdir -p "$ART"
PTOAS_CAPTURE_LL="$ART/device_input.ll" \
  ASCEND_HOME_PATH="$SHIM" \
  "$PTOAS" --pto-arch=a5 --pto-backend=vpto --pto-level=level2 \
  --tile-lib-backend=ptodsl --enable-vmi --enable-op-fusion=true \
  "$K" -o "$ART/fa.fatobj.o"

# 非 VMI 路:
ART=/tmp/perf/asm/novmi; rm -rf "$ART"; mkdir -p "$ART"
PTOAS_CAPTURE_LL="$ART/device_input.ll" \
  ASCEND_HOME_PATH="$SHIM" \
  "$PTOAS" --pto-arch=a5 --pto-backend=vpto --pto-level=level2 \
  --tile-lib-backend=ptodsl --enable-op-fusion=false \
  "$K" -o "$ART/fa.fatobj.o"

ls -la /tmp/perf/asm/vmi/device_input.ll /tmp/perf/asm/novmi/device_input.ll
```

> 关键 env：`PTOAS_CAPTURE_LL` 命名落盘文件，`ASCEND_HOME_PATH="$SHIM"` 让 ptoas 调 shim 而非真 bisheng。`--pto-level=level2` 要带（fusion 流水线才跑）。

### 5.3 用 bisheng 编 aicore `.o` / `.s`（汇编）

捕获的 device IR 可编 aicore object，也可直接出汇编文本。**优先出 `.s`**（人可读的汇编指令），`.o` 留作体积校验：

```bash
# object (校验用):
bisheng --target=hiipu64-hisilicon-cce -march=dav-c310-vec \
  --cce-aicore-arch=dav-c310-vec --cce-aicore-only -O2 \
  -c -x ir <device_input.ll> -o rpo_vmi.aicore.o

# assembly (看指令用), 只改 -c→-S:
bisheng --target=hiipu64-hisilicon-cce -march=dav-c310-vec \
  --cce-aicore-arch=dav-c310-vec --cce-aicore-only -O2 \
  -S -x ir <device_input.ll> -o rpo_vmi.s
# 非 VMI 路: 同上, 把 vmi 换成 novmi (IR 文件 + -o 输出名)
```

验证 `.o`：`file rpo_vmi.aicore.o` 应为 `ELF 64-bit LSB relocatable, *unknown arch 0x1029*`。

### 5.4 vf-fusion off 的可读汇编：用后端 `-mllvm` 选项

> **注意**：PTOAS 编 fatobj 时已默认传 `--cce-auto-sync=off`（关闭 bisheng 的 auto-sync，避免在循环内插入冗余 SMEM_BAR）。手动用 bisheng 编 `.s` 时也需加上，否则会看到 128+ 个循环内 SMEM_BAR（VLD_VST），rvec busy 从 ~727 飙到 ~4262。

要看 bisheng SIMD VF 融合关闭后的汇编，有个限制：**可读 `.s` 和 vf-fusion 生效在简单 argv 下互斥**——前端 `--cce-simd-vf-fusion=false` 在 IR 路径下报 `argument unused`。

破解：用 `-mllvm` 后端选项，不走前端语法糖。`--cce-simd-vf-fusion=false` 实际展开成这 7 个后端 pass 选项（false 分支，按编译器源码）：

```
-mllvm -cce-vf-enable-vf-fusion=false
-mllvm -cce-vf-enable-vf-loop-extender=false
-mllvm -cce-vf-enable-loop-fusion=false
-mllvm -cce-vf-enable-vf-ldst-elimination=false
-mllvm -cce-vf-enable-ub-dead-st-elimination=false
-mllvm -cce-vf-auto-sync=off
-mllvm -cce-vf-enable-vf-ifelse-extender=false
```

这些走 LLVM 后端 pass，在简单 argv `-S` 路径也跑，所以同时拿到可读 `.s` 和 vf-fusion off 效果：

```bash
# vf-fusion OFF 的可读汇编（简单 argv -S + -mllvm 后端选项，false 分支）
bisheng --target=hiipu64-hisilicon-cce -march=dav-c310-vec \
  --cce-aicore-arch=dav-c310-vec --cce-aicore-only -O2 \
  --cce-auto-sync=off \
  -mllvm -cce-vf-enable-vf-fusion=false \
  -mllvm -cce-vf-enable-vf-loop-extender=false \
  -mllvm -cce-vf-enable-loop-fusion=false \
  -mllvm -cce-vf-enable-vf-ldst-elimination=false \
  -mllvm -cce-vf-enable-ub-dead-st-elimination=false \
  -mllvm -cce-vf-auto-sync=off \
  -mllvm -cce-vf-enable-vf-ifelse-extender=false \
  -S -x ir <device_input.ll> -o output_vfoff.s
# 对照：vf-fusion ON 的可读汇编（同简单 argv -S，不带 vf 选项，但仍需关 auto-sync）
bisheng --target=hiipu64-hisilicon-cce -march=dav-c310-vec \
  --cce-aicore-arch=dav-c310-vec --cce-aicore-only -O2 \
  --cce-auto-sync=off \
  -S -x ir <device_input.ll> -o output_vfon.s
```

> `--cce-simd-vf-fusion=true` 分支额外 push `-cce-vf-remove-membar=true` + `-cce-vf-auto-sync=fused`；`=false` 分支 push `-cce-vf-auto-sync=off` 但不 push `-cce-vf-remove-membar`（用后端默认）。
>
> 简单 argv（无 `-cce-bitcode-is-aicore`）`-S` 可用；full ptoas argv（带 `-cce-bitcode-is-aicore`）`-S` 报 `unsupported option '-S' on device side`，只能 `-c` 出 `.o`，且 `llvm-objdump -d` 对 arch 0x1029 全 `<not available>`（无 disassembler backend）。

### 5.5 从 ptoas 注入 vf-fusion 选项跑 sim

ptoas 调 bisheng 的 argv 由 `tools/ptoas/ObjectEmission.cpp` 构造，无原生透传机制。要让 sim 也跑 vf-fusion off，可在该文件的 vec-misched 注入之后加 env 读取，把 `PTOAS_BISHENG_VF_ARGS`（空格分隔）的 token 追加进 bisheng argv，然后 rebuild。跑 sim 时：

```bash
export PTOAS_BISHENG_VF_ARGS="-mllvm -cce-vf-enable-vf-fusion=false -mllvm -cce-vf-enable-vf-loop-extender=false -mllvm -cce-vf-enable-loop-fusion=false -mllvm -cce-vf-enable-vf-ldst-elimination=false -mllvm -cce-vf-enable-ub-dead-st-elimination=false -mllvm -cce-vf-auto-sync=off -mllvm -cce-vf-enable-vf-ifelse-extender=false"
# 然后照 §3 跑 validation 脚本（该 env 会被 ptoas 读到，注入 bisheng）
```

`PTOAS_BISHENG_VF_ARGS` 传前端 `--cce-simd-vf-fusion=false` 在 sim 路也没用（同样的 unused），必须传后端 `-mllvm` 选项。用 strace 确认选项落到 ptoas 调 bisheng 的 argv 里：

```bash
strace -f -e trace=execve -s 4000 -o /tmp/strace_vf.log \
  "$PTOAS_BIN" <full compile cmd>
grep 'execve.*bisheng' /tmp/strace_vf.log   # 看 argv 是否含 -cce-vf-*
```

## 6. hazard 计数 + 运行日志保存

sim 的 `overlaps with` warning = 运行时访存流水 hazard。每行带完整字段：

```
[warning] [00002857] cur_instr RV_VLDI (id:548 pc:0x10d0d230 vloop_id:1 vloop_pc:0x10d0d228) overlaps with pre_instr RV_VSTI (id:545 pc:0x10d0d224 vloop_id:0 vloop_pc:0x0)
```

字段：方括号内 tick，`cur_instr`/`pre_instr` 是冲突对，`id` 指令序号，`pc` 真实地址，`vloop_id`/`vloop_pc` 虚拟循环位置。两类典型冲突：
- **VLDI vs VSTI**：上个 vloop 写 UB 与下个 vloop 读同一 UB 无同步。
- **VSTI vs VSTI**：相邻两次 UB 写冲突。

### 6.1 运行日志保存 + hazard 计数

每次跑 sim 的运行日志（validation 脚本的 `validation.log`，含 hazard warning 行）必须存档，不能只存 summary。hazard 个数是性能/正确性关键指标。

```bash
DEST=log/fa_vmi_rowplusone_full
for n in 1 2; do  # 1=VMI, 2=非VMI
  DIR=/tmp/fa_rpo_t$n/fa-softmax-dn-init-rowplusone   # 目录名 = CASE_NAME，无 vmi_ 前缀
  tag=$([ "$n" = 1 ] && echo vmi || echo novmi)
  cp "$DIR/validation.log" "$DEST/perf_${tag}_run.log"
done

# hazard 计数 + 类型分解
for tag in vmi novmi; do
  LOG="$DEST/perf_${tag}_run.log"
  echo "=== $tag ==="
  echo "hazard 总数: $(grep -c 'overlaps with' "$LOG")"
  grep "overlaps with" "$LOG" \
    | sed -E 's/.*cur_instr (RV_[A-Z]+).*pre_instr (RV_[A-Z]+).*/\1 vs \2/' | sort | uniq -c
done
```

## 7. Dump 各 pass 中间 MLIR

```bash
# 抓全 pipeline pass 列表（先做这步知道有哪些 pass 可 dump）
$PTOAS_BIN --pto-arch=a5 --pto-backend=vpto --pto-level=level2 \
  --tile-lib-backend=ptodsl --enable-vmi --enable-op-fusion=true \
  --emit-vpto -o /dev/null --mlir-print-ir-after-all 2>&1 \
  | grep -oE "IR Dump After [A-Za-z0-9_]+ \([a-z0-9-]+\)" | awk '!seen[$0]++'
```

dump 单个 pass：

```bash
$PTOAS_BIN --pto-arch=a5 --pto-backend=vpto --pto-level=level2 \
  --tile-lib-backend=ptodsl --enable-vmi --enable-op-fusion=true \
  --emit-vpto "$CASE/kernel.pto" -o /dev/null \
  --mlir-print-ir-after="<pass-arg>" 2>&1 \
  | grep -v '^TileLib daemon\|^Info: ptodsl' \
  | awk '/^\/\/ -----.*IR Dump/ {p=1} p {print}' > stage.mlir
```

`--enable-vecscope-mem-bar` 的 pass 名 `pto-insert-vecscope-mem-bar`，跑在 `PTOInferVPTOVecScope` 之后、`VPTOExpandWrapperOps` 之前。dump 它的 before/after 看 membar 插入：

```bash
$PTOAS_BIN --pto-arch=a5 --pto-backend=vpto --pto-level=level2 \
  --tile-lib-backend=ptodsl --enable-op-fusion=false --enable-vecscope-mem-bar \
  --emit-vpto "$CASE/kernel.pto" -o /dev/null \
  --mlir-print-ir-after="pto-insert-vecscope-mem-bar" 2>&1 \
  | grep -v '^TileLib daemon\|^Info: ptodsl' > membar_after.mlir
```

## 8. 产物清单

跑完上述各步，典型产物：

```
log/fa_vmi_rowplusone_full/
  perf_vmi_core0_summary.txt       # VMI 路 summary
  perf_novmi_core0_summary.txt     # 非 VMI 路 summary
  perf_vmi_instr_log.dump          # VMI 路指令时间序列
  perf_novmi_instr_log.dump        # 非 VMI 路指令时间序列
  perf_vmi_run.log                 # VMI 路 validation 日志（含 hazard warning）
  perf_novmi_run.log               # 非 VMI 路
  vmi_device_input.ll              # VMI 路 device IR（shim 捕获）
  novmi_device_input.ll            # 非 VMI 路 device IR（shim 捕获）
  rpo_vmi.s / rpo_novmi.s          # 两路汇编
  rpo_vmi.aicore.o / rpo_novmi.aicore.o   # aicore obj（.text 校验）
```
