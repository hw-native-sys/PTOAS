# UF 同步循环验证

该用例通过 `--enable-insert-sync` 编译，复用 L0A/L0B 和地址为零的 ACC，
128 轮 matmul/store 分别写入 128 个输出 tile。输入为小整数 FP16，CPU
参考结果可由 FP32 精确表示；比较要求 `rtol=0, atol=0` 并检查完整输出长度。

```bash
"$PTOAS_BIN" --pto-arch=a5 --pto-level=level3 --enable-insert-sync uf_sync_loop.pto -o uf_sync_loop-pto.cpp
python3 ../../npu_validation/scripts/generate_testcase.py \
  --input uf_sync_loop-pto.cpp --testcase uf_sync_loop \
  --output-root "$VALIDATION_ROOT" --run-mode npu \
  --soc-version Ascend950DT_9574 --aicore-arch dav-c310-cube
```

L1 输入使用 col_major 的 blayout 与 row_major 的 slayout，以满足 A5 的
ND→NZ 加载约束；随后搬入 L0A/L0B 的布局由各自 tile 类型指定。

使用生成项目的 CMake 构建流程及真机 CANN 环境。运行前核对生成 C++：ACC
链不得残留 M/FIX 双向事件和 PIPE_M barrier；输入加载所需 MTE 事件仍须存在。
在生成的 `UFSyncLoop/uf_sync_loop` 目录中重复至少 10 次以下步骤，每次独立
生成输入、运行并比较；任何超时或非零退出均算失败：

```bash
python3 golden.py
timeout 60 ./build/uf_sync_loop
python3 compare.py
```

`PTO_ISA_ROOT` 必须指向与目标 CANN 兼容的 pto-isa。编译器可在构建机生成 C++，
真机只需接收该 C++、样例资产与 testcase 生成器，无需部署整套 LLVM。
