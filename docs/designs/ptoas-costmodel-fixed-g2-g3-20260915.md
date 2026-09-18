# 固定候选 G2→G3 验收记录（2026-09-15）

冻结运行版本 `20ae930a44d066f3e77d6f8916e23642280921f8`：47 个合法变体通过受限 G2 和设备 G3，三种子、两种执行顺序共 282 次运行全部通过。另有 3 个槽数不足配置在编译阶段被拒绝，未送设备。诊断性能完成 130 个计时样本；无法可靠区分候选收益，G4 未认证。

## 范围与证据身份

- A5、32×32 FP32、静态单轴 N=1/2/4/8、固定四阶段图、1 AIC＋2 AIV，核内调度关闭。
- 主机 `ptoas-a5-52`、设备 0、实际 SOC `Ascend950PR_9589`；CANN 9.2.0-beta.1、Bisheng clang 15.0.5、目标 `dav-c310`。
- 原生编译器指纹 `5ecce95a8b7753b009bbcc73454a636b7541a098e292b66f65937a47f29ffefb`；完整原生库摘要位于矩阵 manifest。
- `ptoas-a5-39` 的 ACL 探测未在等待期限内调度，取消该待运行任务后选用 52；没有合并两台主机的数据。
- ACL 探测 0/1 均通过，按计划选择设备 0。状态为 `ACL_RUNTIME_AVAILABLE`；缺少 npu-smi，不声明完整健康遥测或对非协作进程的独占证明。
- 所有设备变体在同一冻结提交上生成，47 个 Bisheng 构建均无警告。后续离线分析和测试提交不改变运行对象。

## 逐候选 G2/G3

`serial` 是原始串行循环；`p0` 是 P=0 静态展开对照。`slots123` 使用 P=min(2,N)，按 Vector local A/B/C 指定 1/2/3 槽。每个合法变体执行 6 次；被拒绝配置不执行。

| 用例 | 变体 | P | Vector A/B/C 槽 | G2 | G3 | 运行数 |
|---|---|---:|---|---|---|---:|
| basic-n1 | serial | 0 | 1/1/1 | pass | pass | 6 |
| basic-n1 | p0 | 0 | 1/1/1 | pass | pass | 6 |
| basic-n1 | p1 | 1 | 1/1/1 | pass | pass | 6 |
| basic-n1 | p2 | 2 | 1/1/1 | pass | pass | 6 |
| basic-n1 | slots123 | 1 | 1/2/3 | pass | pass | 6 |
| basic-n2 | serial | 0 | 1/1/1 | pass | pass | 6 |
| basic-n2 | p0 | 0 | 1/1/1 | pass | pass | 6 |
| basic-n2 | p1 | 1 | 1/1/1 | pass | pass | 6 |
| basic-n2 | p2 | 2 | 1/1/1 | pass | pass | 6 |
| basic-n2 | p3 | 3 | 1/1/1 | pass | pass | 6 |
| basic-n2 | slots123 | 2 | 1/2/3 | pass | pass | 6 |
| basic-n4 | serial | 0 | 1/1/1 | pass | pass | 6 |
| basic-n4 | p0 | 0 | 1/1/1 | pass | pass | 6 |
| basic-n4 | p1 | 1 | 1/1/1 | pass | pass | 6 |
| basic-n4 | p2 | 2 | 1/1/1 | pass | pass | 6 |
| basic-n4 | p4 | 4 | 1/1/1 | pass | pass | 6 |
| basic-n4 | p5 | 5 | 1/1/1 | pass | pass | 6 |
| basic-n4 | slots123 | 2 | 1/2/3 | pass | pass | 6 |
| basic-n8 | serial | 0 | 1/1/1 | pass | pass | 6 |
| basic-n8 | p0 | 0 | 1/1/1 | pass | pass | 6 |
| basic-n8 | p1 | 1 | 1/1/1 | pass | pass | 6 |
| basic-n8 | p2 | 2 | 1/1/1 | pass | pass | 6 |
| basic-n8 | p8 | 8 | 1/1/1 | pass | pass | 6 |
| basic-n8 | p9 | 9 | 1/1/1 | pass | pass | 6 |
| basic-n8 | slots123 | 2 | 1/2/3 | pass | pass | 6 |
| crossing-n1 | serial | 0 | 1/1/1 | pass | pass | 6 |
| crossing-n1 | p0 | 0 | 1/1/1 | pass | pass | 6 |
| crossing-n1 | p1 | 1 | 1/1/1 | pass | pass | 6 |
| crossing-n1 | p2 | 2 | 1/1/1 | pass | pass | 6 |
| crossing-n1 | slots123 | 1 | 1/2/3 | pass | pass | 6 |
| crossing-n2 | serial | 0 | 1/1/1 | pass | pass | 6 |
| crossing-n2 | p0 | 0 | 1/1/1 | pass | pass | 6 |
| crossing-n2 | p1 | 1 | 2/1/1 | pass | pass | 6 |
| crossing-n2 | p2 | 2 | 2/1/1 | pass | pass | 6 |
| crossing-n2 | p3 | 3 | 2/1/1 | pass | pass | 6 |
| crossing-n2 | slots123 | 2 | 1/2/3 | rejected | not_run | 0 |
| crossing-n4 | serial | 0 | 1/1/1 | pass | pass | 6 |
| crossing-n4 | p0 | 0 | 1/1/1 | pass | pass | 6 |
| crossing-n4 | p1 | 1 | 2/1/1 | pass | pass | 6 |
| crossing-n4 | p2 | 2 | 3/1/1 | pass | pass | 6 |
| crossing-n4 | p4 | 4 | 4/1/1 | pass | pass | 6 |
| crossing-n4 | p5 | 5 | 4/1/1 | pass | pass | 6 |
| crossing-n4 | slots123 | 2 | 1/2/3 | rejected | not_run | 0 |
| crossing-n8 | serial | 0 | 1/1/1 | pass | pass | 6 |
| crossing-n8 | p0 | 0 | 1/1/1 | pass | pass | 6 |
| crossing-n8 | p1 | 1 | 2/1/1 | pass | pass | 6 |
| crossing-n8 | p2 | 2 | 3/1/1 | pass | pass | 6 |
| crossing-n8 | p8 | 8 | 8/1/1 | pass | pass | 6 |
| crossing-n8 | p9 | 9 | 8/1/1 | pass | pass | 6 |
| crossing-n8 | slots123 | 2 | 1/2/3 | rejected | not_run | 0 |

跨阶段 N=4、P=2 的 A 需要三槽；实际配置 A/B/C=3/1/1，每个 AIV 的 vector arena 为 43,008 字节，可用预算 253,952 字节。相同数值地址按 AIV0/AIV1 各自核算。

## 正确性与 G2 覆盖

三种固定种子的逐轮 Q/K/V 从 [-2,2] 整数生成并保存为 FP32；CPU 使用独立 NumPy 整数矩阵运算生成可精确表示的 golden。basic 计算 O=2×((-QK)×V)，crossing 计算 O=((-QK)×V)+(-QK)。两个 AIV 写入不同半区，四个 GM 参数独立分配。所有输出逐元素精确一致；重复输出稳定，输入未改变、外围 guard 未修改，运行无错误和超时。GM guard 与静态访问范围检查共同构成证据，不替代全部设备地址访问跟踪。

同一次原生调用生成 C++、最终 PTO、memory_plan.json 和 validation_report.json。反馈绑定版本、槽位、物理域、实际 engine/set-wait 完成顺序及产物身份。unknown 不能通过 G2。检查范围限于已映射 FP32 操作、N≤pipe 容量、无 local/backing 重叠及匹配的 PTO 头文件语义。

这 47 个设备变体没有不同 allocation 之间的地址重叠。跨迭代槽位轮转已在设备验证，跨 Buffer 地址复用仅有编译器正负测试；不能把本轮 G3 扩大为通用地址复用、batched FIFO 或任意 Bisheng 后续降低证明。

## 回归与发现

12 项基础回归、38 项 v1、31 项 v2、15 项 G2 和 4 项计时边界测试通过。覆盖来源保留、双 AIV 核算、合法复用、异步覆盖拒绝、复用环拒绝、未知完成条件、错误槽位/事务 split、反馈篡改、编译器身份以及 N=0 空执行。快速代码规范检查 42 个文件，0 错误、0 警告；完整静态分析工具未齐备，不宣称完成全套静态分析。

bringup 发现原 16×16 接口 micro 每个 AIV 只有 8 行 NZ allocation，A5 ND→NZ 实际按 16 行跨度写入，首个串行试验出现 expected 88 / actual 96 后立即停止。新增 NATIVE_LAYOUT_OVERFLOW 编译拒绝检查，并使用 32×32 设备 fixture（每 AIV 16 行）完成本轮验收。原接口 micro 保留；未放宽容差。

## 诊断性能

仅 N=4/8、同一 G3 设备对象：26 个变体各 1 次预热和 5 次串行 msprof 采样，共 156 次运行，全部通过 golden、输入及 guard 检查。130 份计时记录均核对为恰好一个匹配 kernel 符号的 `MIX_AIC` task，Block Num=1、Mix Block Num=2、Device_id=0，单位为 `Task Duration(us)`。

所有非自身比较与 serial、P0 的实测范围均重叠，结论为**无法可靠区分收益**。不能按最低单点或中位数选出“最佳优化”。表中比值是候选中位时延/基线中位时延，小于 1 表示中位数较低，不等于收益已认证。

| 用例 | 变体 | 中位数（µs） | 范围（µs） | /serial | /P0 |
|---|---|---:|---|---:|---:|
| basic-n4 | serial | 9.718 | 9.583–10.211 | 1.000 | 0.974 |
| basic-n4 | p0 | 9.976 | 9.607–10.503 | 1.027 | 1.000 |
| basic-n4 | p1 | 9.905 | 9.788–10.308 | 1.019 | 0.993 |
| basic-n4 | p2 | 10.103 | 9.650–10.200 | 1.040 | 1.013 |
| basic-n4 | p4 | 9.785 | 9.541–10.010 | 1.007 | 0.981 |
| basic-n4 | p5 | 9.779 | 9.521–10.054 | 1.006 | 0.980 |
| basic-n4 | slots123 | 9.749 | 9.083–10.291 | 1.003 | 0.977 |
| basic-n8 | serial | 14.010 | 13.731–14.753 | 1.000 | 0.991 |
| basic-n8 | p0 | 14.135 | 13.364–14.827 | 1.009 | 1.000 |
| basic-n8 | p1 | 13.667 | 13.269–14.193 | 0.976 | 0.967 |
| basic-n8 | p2 | 14.039 | 13.709–14.968 | 1.002 | 0.993 |
| basic-n8 | p8 | 13.592 | 13.232–13.967 | 0.970 | 0.962 |
| basic-n8 | p9 | 13.734 | 13.270–13.972 | 0.980 | 0.972 |
| basic-n8 | slots123 | 13.914 | 13.161–14.357 | 0.993 | 0.984 |
| crossing-n4 | serial | 9.951 | 9.242–10.295 | 1.000 | 1.012 |
| crossing-n4 | p0 | 9.832 | 9.677–10.570 | 0.988 | 1.000 |
| crossing-n4 | p1 | 10.063 | 9.236–10.244 | 1.011 | 1.023 |
| crossing-n4 | p2 | 10.063 | 9.493–10.331 | 1.011 | 1.023 |
| crossing-n4 | p4 | 9.527 | 9.445–10.595 | 0.957 | 0.969 |
| crossing-n4 | p5 | 9.981 | 9.607–10.183 | 1.003 | 1.015 |
| crossing-n8 | serial | 13.644 | 13.123–14.270 | 1.000 | 0.962 |
| crossing-n8 | p0 | 14.181 | 13.628–14.433 | 1.039 | 1.000 |
| crossing-n8 | p1 | 13.974 | 13.662–14.702 | 1.024 | 0.985 |
| crossing-n8 | p2 | 13.765 | 13.447–14.248 | 1.009 | 0.971 |
| crossing-n8 | p8 | 13.933 | 13.894–14.134 | 1.021 | 0.983 |
| crossing-n8 | p9 | 13.330 | 12.943–14.156 | 0.977 | 0.940 |

本轮为小规模 micro 的诊断采样，每次恢复输入并启动独立 runner 进程。上表是混合 kernel 的设备 task 时间，未用包含 ACL/msprof 启动的进程 wall time 代替 kernel 时延，也未拆分测量两个子核。五次样本不构成统计认证；没有完整 TileSim 预测、预测误差评估或全量健康遥测。静态展开影响和 preload/multi-buffer 影响在当前数据中均不能可靠排序，更不能外推到真实 FA。

`analysis/summary.json` 保存全部单次时间、profiler CSV 路径及 SHA-256、标准差和两种基线比较状态；原始 profiler 数据保留在各运行目录的 `profile/` 下。

## 后续仍未完成

完整 M01–M12、有界内存规划搜索、TileSim 全程序计时及计时槽数搜索、编译反馈消费与重新评估、真实 FA、G4 认证与自动启用均保留未完成状态。默认 annotation-only 及 v1/2.0 协议行为不变。

## 原始证据与重放

本地证据根（均位于开发者私有目录，不纳入源码仓库）：

```text
/Users/lishengtao/Documents/PTO/_private_ptoas_lab/20260915-cv-g3-acceptance-04
/Users/lishengtao/Documents/PTO/_private_ptoas_lab/20260915-cv-diagnostic-perf-04
/Users/lishengtao/Documents/PTO/_ptoas_tech_lab_records/profiles/cv-costmodel/experiments/20260915-003-g2-g3-runtime
```

矩阵 manifest 绑定源码、bindings、target、候选、布局和 C++；各 build_manifest.json 绑定头文件、Bisheng、设备对象与 launcher；每次运行保存 command.json、result.json、输入、golden、输出及日志。重放必须使用冻结提交和对应编译环境，重新生成到独立目录；不能在已冻结目录覆盖产物。操作命令见 [fixture README](../../test/samples/CVCostModel/README.md)。


原始证据完成离线审计：47 份 build manifest、282 次 G3 和 156 次性能运行的对象/输入/输出/候选身份全部一致；每次输出文件摘要与独立 golden 一致。远端 G3 的 3,162 个文件和性能阶段的 10,951 个文件已逐文件核对 SHA-256，两个远端实验已结束并冻结只读。

- 矩阵 SHA-256：`e46e4253e85860e08ba845dc6939971bd58858dca2d51629120ede32318ec132`。
- G3 报告 SHA-256：`fa559bbf4fc04bdc84b4f902d7e4003b75ed888feb92715dc3f39351effda5a1`。
- 性能报告 SHA-256：`c51872a37e50b02c5aacae78839bc33a27295321b95364e7aa1214e1e74a3307`。
- `compiler-snapshot/` 保留冻结源码归档和原生组件；7,771 个源码文件重新计算出的指纹与矩阵一致。该 macOS/Python 3.9 原生组件快照仍依赖匹配的 LLVM 19.1.7，不是可移植工具包。

运行冻结提交为 `20ae930a4`；离线计时工具提交为 `cd10e0563`，新增编译器合法地址复用测试为 `e42a61178`。最终文档提交不改变已验收的 kernel 和 launcher。全部提交仅保留本地，未推送远端。
