# PTOAS CI 与上板测试说明

## 1. 文档目的

本文说明 PTOAS 当前与开发者最相关的 4 条验证链路：

1. 本地最小复现：`build -> runop -> 生成 pto/cpp`
2. GitHub Actions：`Build Wheel` 与 `CI`
3. A3/A5 self-hosted runner 每日板测
4. PR 评论触发的 A3/A5 手动板测

本文只描述当前仓库已经存在、并且日常开发会直接用到的流程，不展开板测机器人内部实现。

## 2. 当前链路总览

### 2.1 `Build Wheel`

对应 workflow：`.github/workflows/build_wheel.yml`

补充：

- macOS wheel 走独立 workflow：`.github/workflows/build_wheel_mac.yml`
- 评论板测监测器当前依赖的是 Linux `Build Wheel` 产物

用途：

- 构建 Linux wheel
- 构建 `ptoas` 二进制分发包
- 产出评论板测监测器会下载的 artifact

关键 artifact：

- `ptoas-wheel-py<version>-x86_64`
- `ptoas-bin-x86_64`

结论：

- 如果 PR 对应 head SHA 没有成功的 `Build Wheel`，评论触发的板测通常无法准备工具链。
- 板测机器人报 `no successful workflow run named 'Build Wheel' found` 时，先补跑这个 workflow。

### 2.2 `CI`

对应 workflow：`.github/workflows/ci.yml`

用途：

- 在 GitHub runner 上构建 LLVM/MLIR 与 PTOAS
- 执行 `test/samples/runop.sh --enablebc all`
- 在 `workflow_dispatch` / `schedule` 时打包 payload，由 A3 self-hosted runner 通过 `task-submit` 执行 `run_remote_npu_validation.sh`

触发差异：

- `push` / `pull_request`：只跑 GitHub runner 上的构建和样例生成，不会把 PR-controlled workflow 调度到 A3 runner
- `workflow_dispatch` / `schedule`：除上述步骤外，还会在 A3 runner 跑板测 job `remote-npu-validation`

### 2.3 `A5 Nightly Board`

对应 workflow：`.github/workflows/ci_a5.yml`

用途：

- 每天北京时间 `03:00` 在 A5 self-hosted runner 上构建 `main`
- 生成全量 A5 payload，并通过 `task-submit` 排队执行 NPU 板测
- 在 GitHub Actions Summary 中展示 `OK / FAIL / SKIP` 汇总并上传结果 artifact
- 配置 `A5_FEISHU_WEBHOOK_URL` secret 后，同步发送飞书结果卡片

该 workflow 直接在 A5 主机运行，不使用 SSH secret，也不依赖板测监测器的私有脚本。runner 尚未在线时，定时任务会保持排队状态。

### 2.4 评论触发板测

这部分不在本仓库内实现，但当前开发流程依赖它。

用途：

- 在 PR 评论区手动触发 A3 或 A5 板测
- 结果会回到 GitHub 评论区，并同步飞书通知

常用命令：

- `/run a3`
- `/run all`
- `/run a5 <case>`
- `/run a5 case1,case2 --pto-level=level3`

约束：

- A5 手动板测当前应显式给出 case 列表，不建议直接 `/run a5` 空跑。
- `/run all` 当前主要用于 A3 这套全量手动板测。
- 如果 PR 与 `origin/main` 有冲突，监测器应直接提示冲突并跳过，不应继续执行。

## 3. 本地最小复现

下面这套命令尽量对齐 `CI` workflow，适合开发阶段先在本地确认 `py -> pto -> cpp` 没有问题。

### 3.1 构建 LLVM/MLIR

```bash
git clone https://github.com/vpto-dev/llvm-project.git
cd llvm-project
git checkout feature-vpto

cmake -G Ninja -S llvm -B llvm/build-shared \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DBUILD_SHARED_LIBS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE=python3 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD="host"

ninja -C llvm/build-shared
```

### 3.2 构建 PTOAS

在 PTOAS 仓库根目录执行：

```bash
export LLVM_DIR=$PWD/llvm-project/llvm/build-shared
export PTO_INSTALL_DIR=$PWD/install
python3 -m pip install 'pybind11<3'

cmake -G Ninja -S . -B build \
  -DLLVM_DIR="$LLVM_DIR/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_DIR/lib/cmake/mlir" \
  -DPython3_EXECUTABLE=python3 \
  -DPython3_FIND_STRATEGY=LOCATION \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DCMAKE_INSTALL_PREFIX="$PTO_INSTALL_DIR" \
  -DCMAKE_BUILD_TYPE=Release

ninja -C build ptoas
ninja -C build ptobc
ninja -C build install
```

### 3.3 跑样例生成链路

```bash
LLVM_BUILD_DIR="$LLVM_DIR" ./quick_install.sh
export PTOAS_BIN="$(command -v ptoas)"

bash test/samples/runop.sh --enablebc all
```

只跑单个目录时：

```bash
bash test/samples/runop.sh --enablebc -t Sync
```

只跑单个 Python 样例时，直接进入对应目录执行也可以，但要自己保证 `PYTHONPATH` / `LD_LIBRARY_PATH` / `PTOAS_BIN` 已经对齐。

如果要本地复现 A5 / Qwen 一类 case，建议显式补齐 `ptoas` 参数：

```bash
export PTOAS_FLAGS="--pto-arch=a5 --pto-level=level3"
bash test/samples/runop.sh --enablebc -t Qwen3Decode
```

说明：

- `runop.sh` 支持通过 `PTOAS_FLAGS` 透传额外 `ptoas` 参数
- `runop.sh` 默认会追加 `--enable-insert-sync`
- 本地复现板测问题时，优先让 `PTOAS_FLAGS` 与评论触发板测保持一致

### 3.4 直接跑远端板测脚本

如果你已经在板机上，或者 GitHub Actions payload 已经准备好，可以直接运行：

```bash
STAGE=run \
RUN_MODE=npu \
SOC_VERSION=Ascend910B1 \
DEVICE_ID=2 \
SKIP_CASES='mix_kernel,vadd_validshape,vadd_validshape_dynamic,print,storefp' \
bash test/npu_validation/scripts/run_remote_npu_validation.sh
```

A5 例子：

```bash
STAGE=run \
RUN_MODE=npu \
SOC_VERSION=Ascend950 \
RUN_ONLY_CASES='qwen3_decode_layer_incore_0,qwen3_decode_layer_incore_1' \
DEVICE_ID=1 \
bash test/npu_validation/scripts/run_remote_npu_validation.sh
```

说明：

- `RUN_ONLY_CASES` 和 `SKIP_CASES` 都支持逗号或空格分隔。
- `run_remote_npu_validation.sh` 会自动 source 常见 Ascend 环境脚本，并尝试探测 `ASCEND_HOME_PATH`。
- `ci.yml` 会在全量跑时根据 `SOC_VERSION` 自动排除 A3-only / A5-only case；手工跑时你自己要关注目标架构是否匹配。

## 4. GitHub Actions 怎么跑

### 4.1 PR 默认会跑什么

PR 创建或推送后，至少会涉及两类 workflow：

1. `Build Wheel`
2. `CI`

推荐检查顺序：

1. `Build Wheel` 是否成功
2. `CI` 的 `build-and-test` 是否成功
3. 如果要上板，再决定走 `workflow_dispatch` 还是评论触发监测器

### 4.2 手动触发 `Build Wheel`

当评论板测缺少工具链 artifact 时，先手动补跑：

```bash
gh workflow run build_wheel.yml \
  --repo hw-native-sys/PTOAS \
  --ref <your-branch>
```

检查产物：

```bash
gh run list --repo hw-native-sys/PTOAS --workflow 'Build Wheel' --limit 5
```

### 4.3 配置和触发 A3 每日板测

`CI` 的 `remote-npu-validation` 只会在 `workflow_dispatch` 或定时任务下执行。
它固定使用以下 self-hosted runner 标签：

- `self-hosted`
- `Linux`
- `ARM64`
- `ptoas-a3`

runner 用户需要能够执行 `task-submit`，并提供 `cmake`、`git`、`make`、
`python3`、`tar` 和 Python `numpy`。workflow 不再使用 `SSH_KEY`、
`SSH_KNOWN_HOSTS` 或远端主机参数；payload 会下载到 A3 runner 的临时目录，
通过 TaskQueue 获取设备后在本机执行。默认 `device_id=auto`，由 TaskQueue
选择空闲卡并通过 `TASK_DEVICE` 传给板测脚本。

需要单独检查 A3 runner 的异步 TaskQueue 接口时，手动触发
`.github/workflows/a3_taskqueue_smoke.yml`。该 workflow 不 checkout 源码、
不申请 NPU，只验证 `task-submit --env-file`、任务 ID 解析、`--wait`、
`--status` 和退出码传播。它不接受 `pull_request` 事件，避免 PR-controlled
workflow 在持久化 self-hosted runner 上执行。

命令行例子：

```bash
gh workflow run ci.yml \
  --repo hw-native-sys/PTOAS \
  --ref main \
  -f stage=run \
  -f run_mode=npu \
  -f soc_version=Ascend910 \
  -f device_id=auto \
  -f skip_cases='mix_kernel,vadd_validshape,vadd_validshape_dynamic,print,storefp' \
  -f run_only_cases=''
```

关键输入解释：

- `stage`：`build` 或 `run`
- `run_mode`：`npu` 或 `sim`
- `soc_version`：A3 编译目标，默认 `Ascend910`
- `device_id`：传给 `task-submit --device` 的选择器，默认 `auto`
- `skip_cases`：跳过列表
- `run_only_cases`：只跑列表
- `pto_isa_repo` / `pto_isa_commit`：指定板测使用的 `pto-isa`

定时任务固定使用默认分支；GitHub cron 使用 UTC，因此配置为 `0 19 * * *`，
对应北京时间次日 `03:00`。同一时间只允许一个 A3 nightly job，新的运行会排队且不会取消正在进行的板测。

### 4.4 配置和触发 A5 每日板测

A5 runner 需要注册到 `hw-native-sys/PTOAS`，并具有以下标签：

- `self-hosted`
- `Linux`
- `ARM64`
- `ptoas-a5`

runner 用户需要能够执行 `task-submit`，并能读取预编译 LLVM、CANN 和 Python 环境。建议把 runner service 放入现有 `ptoasboard.slice`，让 runner 侧构建继续受 `35 CPU / 48 GiB` 资源上限约束。`task-submit` 会复用板机任务队列，因此它能和评论板测机器人共享设备而不直接抢卡。

workflow 支持以下 repository variables；未设置时使用 A5 板机当前默认值：

- `PTOAS_A5_LLVM_BUILD_DIR`：预编译 LLVM/MLIR build 目录
- `PTOAS_A5_PYTHON`：能导入 `numpy`、`pybind11` 和该 LLVM Python binding 的 Python
- `PTOAS_A5_BUILD_JOBS`：PTOAS 并行编译数，默认 `4`
- `PTOAS_A5_DEVICE_ID`：传给 `task-submit --device` 的设备号，默认 `1`

飞书通知是可选能力。需要时新增 repository secret `A5_FEISHU_WEBHOOK_URL`；不配置不会影响板测和 GitHub Summary。

payload 生成阶段暂时排除以下使用显式本地内存地址的 Level-3 repro：
`decode_projection_incore_0`、`rmsnorm_incore_0`、
`test_tmov_col_major_16x1_align_a5` 和
`test_tmov_row_major_1x16_control_a5`。这些快照尚未适配当前 Level-3
VPTO/TMOV 输入契约，不应阻断其余 A5 常规板测；完成迁移后应从
`A5_INCOMPATIBLE_LEVEL3_CASES` 中移除并重新纳入 nightly。

手动触发全量 A5 板测：

```bash
gh workflow run ci_a5.yml \
  --repo hw-native-sys/PTOAS \
  --ref main
```

只跑指定 case：

```bash
gh workflow run ci_a5.yml \
  --repo hw-native-sys/PTOAS \
  --ref main \
  -f run_only_cases='qwen3_decode_layer_incore_0,qwen3_decode_layer_incore_1'
```

定时任务固定使用默认分支；GitHub cron 使用 UTC，因此配置为 `0 19 * * *`，对应北京时间次日 `03:00`。同一时间只允许一个 A5 nightly job，新的运行会排队且不会取消正在进行的板测。

## 5. PR 评论触发板测

### 5.1 触发前检查

先检查这几件事：

1. PR 能否干净合到 `origin/main`
2. 对应 head SHA 或 merge SHA 是否有成功的 `Build Wheel`
3. 要跑的 case 是否已经具备可比较的 golden / compare 资产
4. A3 / A5 是否需要按架构拆 case

### 5.2 常用命令

A3 全量：

```text
/run a3
```

A3 手动全量入口：

```text
/run all
```

A5 单 case：

```text
/run a5 qwen3_decode_layer_incore_0 --pto-level=level3
```

A5 多 case：

```text
/run a5 qwen3_decode_layer_incore_0,qwen3_decode_layer_incore_1,qwen3_decode_layer_incore_2 --pto-level=level3
```

说明：

- case 列表支持逗号分隔；为了减少解析歧义，优先使用逗号。
- A5 当前建议总是显式给 case 列表。
- Qwen 一类 A5 case 常常还会带 `--pto-level=level3`。

### 5.3 结果怎么看

通常会有两路结果：

1. GitHub 评论区汇总
2. 飞书机器人消息

常见状态：

- `OK / FAIL / SKIP`
- `fetch-source`：拉源码失败
- `prepare-toolchain`：下载或解压板测工具链失败
- `sample-build-and-test`：样例生成、编译或运行阶段失败
- `internal`：监测器自身异常

## 6. 新增 case 时要注意什么

### 6.1 `runop.sh` 会执行哪些文件

`test/samples/runop.sh` 会遍历样例目录下的 `*.py`，但会跳过：

- `*_golden.py`
- `*_compare.py`

这意味着：

- 纯 helper 脚本不要随便命名成普通 `*.py`
- 如果 helper 只是给 golden 复用，优先命名成 `*_golden.py`，或者放到不会被 `runop.sh` 当成入口脚本执行的路径

否则 `runop.sh all` 会把它当成样例生成入口，导致空 IR 或错误 IR 被送进 `ptoas`。

### 6.2 golden / compare 资产放哪里

当前样例链路会自动拷贝：

- 样例目录下的 `*_golden.py`
- 样例目录下的 `*_compare.py`
- 样例目录下 `npu_validation/golden.py`
- 样例目录下 `npu_validation/compare.py`

建议：

- case 独立的 golden，用 `<case>_golden.py`
- case 独立的 compare，用 `<case>_compare.py`
- 多个 case 共享逻辑时，公共代码不要命名成会被入口匹配到的普通脚本名

### 6.3 A3 / A5 分流

如果 case 只适用于某个架构，需要同步更新 `.github/workflows/ci.yml` 和 `.github/workflows/ci_a5.yml` 中的分流列表：

- `A3_ONLY_CASES`
- `A5_ONLY_CASES`

否则：

- A3 可能错误地去跑 A5 case
- A5 可能错误地去跑 A3 case
- 全量板测会出现和功能本身无关的误报

## 7. 常见问题

### 7.1 为什么 PR 的 `CI` 绿了，但评论板测还是起不来？

最常见原因不是 case 本身，而是工具链前置条件不满足：

- `Build Wheel` 没成功
- PR 没有可用 merge ref
- 板测监测器拉源码或下载 artifact 失败

先看机器人回的失败阶段，不要直接重试很多次。

### 7.2 为什么 A5 手动板测不建议直接 `/run a5`？

当前 A5 监测器更适合显式 case 列表，尤其是 Qwen、Tilelet 这类只跑局部回归的场景。直接空跑会把“我要验证的内容”说不清楚，也容易碰到非目标 case 干扰。

### 7.3 为什么新增了 helper 脚本后 `runop all` 挂了？

因为 `runop.sh` 默认把普通 `*.py` 视为样例入口。helper 如果不属于 golden/compare 命名模式，就会被误执行。

### 7.4 什么时候用 `workflow_dispatch`，什么时候用评论触发板测？

建议：

- 想直接验证远端脚本、payload 或 `run_remote_npu_validation.sh` 时，用 `workflow_dispatch`
- 想对 PR 做日常板测、拿飞书/GitHub 评论反馈时，用评论触发监测器

## 8. 参考文件

- `.github/workflows/build_wheel.yml`
- `.github/workflows/ci.yml`
- `.github/workflows/ci_a5.yml`
- `.github/scripts/report_a5_board_results.py`
- `test/samples/runop.sh`
- `test/npu_validation/scripts/run_remote_npu_validation.sh`
- `test/npu_validation/scripts/generate_testcase.py`
