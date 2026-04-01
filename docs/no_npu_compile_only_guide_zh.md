# 无 NPU 环境的 Compile-Only 指南

本文档说明如何在**没有 `/dev/davinci*` 设备节点**的机器上，使用 `ptoas + CANN(bisheng) + pto-isa` 对生成的 `.cpp` 做**仅编译验证**。

本文档适用于以下场景：

- 本地开发机没有 NPU 卡，只想验证 `ptoas` 生成的 C++ 能否通过 `bisheng` 编译。
- CI 或评审机只有 CANN 工具链，没有运行环境，不需要执行 kernel。
- 需要在上板前完成一轮 host-side compile-only 预检查。

本文档不适用于以下场景：

- 需要真正执行 kernel。
- 需要生成 golden 并做数值比对。
- 需要验证运行时 ACL / 驱动 / 权限问题。

## 1. 概述

- **可以**在无卡环境做 compile-only。
- 需要的不是 NPU 卡，而是：
  - `bisheng` 可用
  - `ASCEND_HOME_PATH` 正确
  - `pto-isa` 头文件和公共测试头可用
- `STAGE=build` 不会检查 `/dev/davinci*`，因此可以复用现有验证脚本完成编译验证。
- A5 case 对 `CANN` 与 `pto-isa` 版本对齐更敏感；如果遇到 A5 静态检查或头文件命名空间错误，需要优先检查版本匹配，而不是默认认为 `ptoas` 代码生成有问题。

## 2. 依赖准备

### 2.1 安装 CANN Toolkit

无卡环境至少需要安装带 `bisheng` 的 CANN Toolkit。安装完成后，确认下面几项存在：

```bash
which bisheng
bisheng --version
ls /usr/local/Ascend
```

常见路径包括：

- `/usr/local/Ascend/cann`
- `/usr/local/Ascend/cann-<version>`
- `/usr/local/Ascend/ascend-toolkit/latest`

加载环境：

```bash
source /usr/local/Ascend/cann/set_env.sh
# 或
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

如果没有自动导出，也可以手动指定：

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann
export PATH="$ASCEND_HOME_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64:${LD_LIBRARY_PATH:-}"
```

### 2.2 准备 pto-isa

`generate_testcase.py` 生成的工程会直接包含 `pto-isa`：

- `${PTO_ISA_ROOT}/include`
- `${PTO_ISA_ROOT}/tests/common`

因此 `PTO_ISA_ROOT` 至少需要满足：

```bash
ls $PTO_ISA_ROOT/include
ls $PTO_ISA_ROOT/tests/common
```

建议直接使用当前 CI pin 的版本，避免本地和 CI 结果不一致。

## 3. 单个 case 的 compile-only

### 3.1 A3 示例

```bash
mkdir -p /tmp/ptoas_compile_only_inputs/Addc
./build/tools/ptoas/ptoas \
  test/samples/Addc/addc.pto \
  -o /tmp/ptoas_compile_only_inputs/Addc/addc-pto.cpp
```

### 3.2 A3 示例生成验证工程并执行编译

```bash
python3 test/npu_validation/scripts/generate_testcase.py \
  --input /tmp/ptoas_compile_only_inputs/Addc/addc-pto.cpp \
  --testcase addc \
  --output-root /tmp/ptoas_compile_only \
  --run-mode npu \
  --soc-version Ascend910

cd /tmp/ptoas_compile_only/Addc/addc
cmake -S . -B build \
  -DENABLE_SIM_GOLDEN=OFF \
  -DSOC_VERSION=Ascend910 \
  -DPTO_ISA_ROOT=$PTO_ISA_ROOT
cmake --build build --parallel
```

### 3.3 A5 示例

以下示例假定本地 `CANN` 与 `pto-isa` 的 A5 版本已经对齐。

```bash
mkdir -p /tmp/ptoas_compile_only_inputs/Sync
./build/tools/ptoas/ptoas \
  test/samples/Sync/test_a5_buf_sync.pto \
  --pto-arch a5 \
  -o /tmp/ptoas_compile_only_inputs/Sync/test_a5_buf_sync-pto.cpp

python3 test/npu_validation/scripts/generate_testcase.py \
  --input /tmp/ptoas_compile_only_inputs/Sync/test_a5_buf_sync-pto.cpp \
  --testcase test_a5_buf_sync \
  --output-root /tmp/ptoas_compile_only_a5 \
  --run-mode npu \
  --soc-version Ascend950

cd /tmp/ptoas_compile_only_a5/Sync/test_a5_buf_sync
cmake -S . -B build \
  -DENABLE_SIM_GOLDEN=OFF \
  -DSOC_VERSION=Ascend950 \
  -DPTO_ISA_ROOT=$PTO_ISA_ROOT
cmake --build build --parallel
```

生成后目录类似：

```text
/tmp/ptoas_compile_only/
└── Addc/
    └── addc/
        ├── CMakeLists.txt
        ├── addc_kernel.cpp
        ├── launch.cpp
        ├── main.cpp
        └── ...
```

这里不会访问 `/dev/davinci*`，因此无卡环境也可以完成。

## 4. 批量编译验证流程

对于一批已经生成的 `*-pto.cpp`，建议直接复用仓库中的现有脚本：

- `test/npu_validation/scripts/run_remote_npu_validation.sh`

这个脚本在 `STAGE=build` 下：

- 会生成 testcase
- 会执行 `cmake` 和 `cmake --build`
- **不会**做设备检查
- **不会**运行可执行文件

### 4.1 准备输入目录

脚本默认扫描：

- `test/samples/**/*.cpp` 中名字匹配 `*-pto.cpp` 的文件

如果你要复用 CI 的样例生成链路，可以先执行：

```bash
export PAYLOAD_ROOT=/tmp/ptoas_payload
export TARGET_SOC_VERSION=Ascend910
export PTO_ISA_COMMIT=893e4b00a826231e776539512b5b17f82fc11838

rm -rf "$PAYLOAD_ROOT"
mkdir -p "$PAYLOAD_ROOT/test/samples"
mkdir -p "$PAYLOAD_ROOT/test/npu_validation/scripts"
mkdir -p "$PAYLOAD_ROOT/test/npu_validation/templates"

cp test/npu_validation/scripts/generate_testcase.py \
  "$PAYLOAD_ROOT/test/npu_validation/scripts/"
cp test/npu_validation/scripts/run_remote_npu_validation.sh \
  "$PAYLOAD_ROOT/test/npu_validation/scripts/"
cp test/npu_validation/templates/* \
  "$PAYLOAD_ROOT/test/npu_validation/templates/"
chmod +x "$PAYLOAD_ROOT/test/npu_validation/scripts/run_remote_npu_validation.sh"

export PTOAS_BIN=$PWD/build/tools/ptoas/ptoas
export PTOBC_BIN=$PWD/build/tools/ptobc/ptobc
export PYTHON_BIN=/usr/bin/python3
export PTOAS_OUT_DIR="$PAYLOAD_ROOT/test/samples"
export PYTHONPATH="$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core:$PTO_INSTALL_DIR:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$LLVM_BUILD_DIR/lib:$PTO_INSTALL_DIR/lib:${LD_LIBRARY_PATH:-}"
export SOC_VERSION="$TARGET_SOC_VERSION"

bash test/samples/runop.sh --enablebc all

# `runop.sh` 对 Sync 目录下 direct `.pto` regression case 的输出命名为 `*.cpp`，
# 但批量验证脚本只扫描 `*-pto.cpp`。这里额外生成一份带唯一后缀的
# `*-pto.cpp`，以便后续 compile-only 流程覆盖这些 direct `.pto` 用例，
# 同时避免覆盖 Python 样例已经生成的同名 `*-pto.cpp`。
sv_lc="$(printf '%s' "$TARGET_SOC_VERSION" | tr '[:upper:]' '[:lower:]')"
for f in "$PAYLOAD_ROOT"/test/samples/Sync/*.cpp; do
  [[ -f "$f" ]] || continue
  [[ "$f" == *-pto.cpp ]] && continue
  base="$(basename "$f" .cpp)"
  if [[ "$base" == "test_a5_buf_sync" && "$sv_lc" != *"950"* && "$sv_lc" != *"a5"* ]]; then
    continue
  fi
  cp "$f" "$PAYLOAD_ROOT/test/samples/Sync/${base}_direct_pto-pto.cpp"
done
```

### 4.2 批量执行 compile-only

完成样例生成后，需要切换到 payload 根目录执行脚本，使其扫描
`$PAYLOAD_ROOT/test/samples/**/*.cpp`，与 CI 的目录布局保持一致：

```bash
cd "$PAYLOAD_ROOT"
export STAGE=build
export RUN_MODE=npu
export SOC_VERSION="$TARGET_SOC_VERSION"
export PTO_ISA_REPO=https://gitcode.com/cann/pto-isa.git
export PTO_ISA_COMMIT=893e4b00a826231e776539512b5b17f82fc11838

# 参照 CI 的做法，按目标 SoC 排除非匹配的 A3/A5 变体。
A3_ONLY_CASES="partition5d,partition5d_dynamic,mrgsort,tmatmulk_autosync"
A5_ONLY_CASES="partition5d_a5,partition5d_dynamic_a5,mrgsort_a5,tmatmulk_autosync_a5,test_a5_buf_sync_direct_pto"
sv_lc="$(printf '%s' "$SOC_VERSION" | tr '[:upper:]' '[:lower:]')"
if [[ "$sv_lc" == *"950"* || "$sv_lc" == *"a5"* ]]; then
  export SKIP_CASES="$A3_ONLY_CASES"
else
  export SKIP_CASES="$A5_ONLY_CASES"
fi

bash ./test/npu_validation/scripts/run_remote_npu_validation.sh
```

在 `STAGE=build` 下，脚本会向 CMake 传递 `-DENABLE_SIM_GOLDEN=OFF`，
因此不会构建 `_sim` 目标，也不依赖 simulator 组件。

如果本地已经有 vendored 的 `pto-isa/` 目录，也可以不走网络 clone，脚本会优先使用本地目录。

### 4.3 指定测试用例

以下命令默认继续在 `$PAYLOAD_ROOT` 目录下执行。

```bash
export STAGE=build
export RUN_ONLY_CASES=abs,gather,scatter
bash test/npu_validation/scripts/run_remote_npu_validation.sh
```

### 4.4 排除特定测试用例

```bash
export STAGE=build
export SKIP_CASES=mix_kernel,print,storefp
bash test/npu_validation/scripts/run_remote_npu_validation.sh
```

## 5. A3 / A5 的注意事项

### 5.1 A3 目标

A3 compile-only 一般只要求：

- 生成链路正确
- `bisheng` 可用
- `pto-isa` include 对齐

### 5.2 A5 目标

A5 case 常见两类失败：

1. **CANN 头文件 / intrinsics 与 pto-isa 不匹配**

典型报错：

```text
no member named 'RoundZType' in namespace '__cce_simd'
```

此类问题通常不是 `.cpp` 语法本身错误，而是当前 `CANN` 与 `pto-isa` 的 A5 头文件接口不一致。

建议处理顺序：

1. 先对齐到 CI 使用的 `pto-isa` commit
2. 再升级到与板端一致的 CANN 版本
3. 最后再判断是否是 `ptoas` 代码生成问题

2. **pto-isa A5 静态约束失败**

典型报错：

```text
static assertion failed: Non-conforming matrix fractal
```

这类通常说明：

- tile layout / fractal / pad 与 A5 要求不一致
- 或者 `.py/.pto` 中的 A5 配置在 lowering 过程中被改写了

此类情况需要检查生成出来的 `Tile<...>` 模板参数，而不是只看前端输入。

## 6. 排障建议

出现 compile-only 失败时，按下面顺序看：

1. `which bisheng` / `bisheng --version`
2. `echo $ASCEND_HOME_PATH`
3. `ls $PTO_ISA_ROOT/include`
4. 确认 `pto-isa` commit 是否与 CI 对齐
5. 确认生成 `.cpp` 时使用的 `--pto-arch` 是否正确
6. 对 A5 case，直接看生成的 `Tile<...>` 参数是否已经偏离预期

## 7. 说明与限制

compile-only 能证明的是：

- `ptoas` 生成的 C++ 能否被当前 `bisheng` + `pto-isa` + CANN 头文件接受

compile-only **不能**证明：

- kernel 在真实 NPU 上一定可运行
- runtime / ACL / 驱动环境正确
- 输出数值一定正确
- 自动同步 / event 分配在真机上一定不会死锁

建议采用以下验证顺序：

1. 本地或无卡机先做 compile-only
2. 通过后再上板做 `STAGE=run`
3. 最终以板测结果为准
