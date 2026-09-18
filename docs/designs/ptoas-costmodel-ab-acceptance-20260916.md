# PTOAS × TileSim 固定推荐 A/B 验收记录（2026-09-16）

本轮完成“先由 TileSim 冻结推荐，再由 PTOAS 编译并上板验证”的闭环。最终四个 workload
全部得到 `BASELINE_RETAINED`：TileSim 判断候选收益不足固定的 2% 门槛，因此 PTOAS 没有
生成 B，也没有采集 A/B profiler 样本。这是预先约定的有效验收结果，不是性能实验缺失。

## 冻结身份

| 项目 | 值 |
|---|---|
| PTOAS | `170e78a2bbcec23f1f8a560e898d6924ec8dd117` |
| TileSim | `0ea5ba8f548595771c0c92f51f797e0dac7e0ee9` |
| 输入矩阵 SHA-256 | `1c85de81d906eba6051211faadfde0f400324fcc71944047a3adc536ddee6034` |
| 主机/设备 | `ptoas-a5-39` / device 0 |
| 运行时 SOC | `Ascend950DT_9592` |
| 设备状态证据 | `ACL_RUNTIME_AVAILABLE`；`npu-smi` 不可用，未宣称完整健康遥测 |

模型推荐在任何新设备测量前冻结。模型、Plan、schedule、A/P0 编译产物和运行报告均绑定上述
输入摘要；本轮未把此前 130 个诊断样本用于候选选择、拟合或替代验收。

## 模型选择结果

| Workload | P=0 预测时延 | 最好候选 | 最好预测时延 | 预测收益 | 冻结决定 |
|---|---:|---:|---:|---:|---|
| basic, N=4 | 10.726510 us | P=1 | 10.648191 us | 0.730% | `BASELINE_RETAINED` |
| crossing, N=4 | 10.726510 us | P=1 | 10.648191 us | 0.730% | `BASELINE_RETAINED` |
| basic, N=8 | 21.186408 us | P=1 | 21.029771 us | 0.739% | `BASELINE_RETAINED` |
| crossing, N=8 | 21.186408 us | P=1 | 21.029771 us | 0.739% | `BASELINE_RETAINED` |

所有候选都有完整模型覆盖。P=1/2/N/N+1 的预测差异位于 0.5% tie 区间时，模型依次按总内存、
有效 preload 和 candidate ID 排序；P=1 成为排名第一的优化候选。它相对 P=0 的收益仍低于
2%，所以最终推荐 ID 指向 baseline。crossing 用例的 preload 候选会增加多槽内存，但不改变结论。

## G2 与 G3

每个 workload 只生成 A（原始串行）和 P0（P=0 静态展开），共八个独立符号和产物；没有伪造 B。
八个产物的候选身份、schedule 指纹、wrapper、memory plan 和 validation report 均通过 G2。

G3 在全新实验 `20260916-cv-ab-acceptance-04` 中运行。A/P0 覆盖三个固定种子和两种执行顺序，
共 48 次运行，结果全部满足：

- CPU 独立 golden 精确相等；
- 输入未修改，GM guard 未修改；
- 相同 case/variant/seed 的两次输出摘要一致；
- 无超时、运行错误或 SOC 身份错配。

正式矩阵第一次尝试发现期望 SOC 与运行时 SOC 不一致，身份门正确停止实验；另一次长队列任务超时。
二者没有并入正式结果。验收使用随后创建的全新实验并重新构建全部产物。

## 性能与 G4 结论

性能实验 `20260916-cv-ab-performance-01` 先验证了相同矩阵摘要和完整 G3 报告。因为四个 workload
都选择 baseline，协议要求不生成 B，也不执行 20 个 AB/BA block。性能报告因此包含零条样本，
汇总报告逐 workload 输出 `BASELINE_RETAINED` 和 `G4: not_applicable`。

本轮结论是模型的保守选择得到端到端验证：当预测收益不足 2% 时，PTOAS 保留串行基线。
没有实测优化候选，不能计算实测收益、bootstrap 区间或 TileSim 对 B 的绝对预测误差，因而没有
`AB_BENEFIT_PASS`、`AB_BENEFIT_FAIL` 或 `G4_EXACT_WORKLOAD_PASS`。默认自动优化仍关闭。

本地证据目录：

- G3：`_private_ptoas_lab/experiments/20260916-cv-ab-acceptance-04`
- 选择与性能汇总：`_private_ptoas_lab/experiments/20260916-cv-ab-performance-01/summary`

远端两个正式实验已通过 `finalize_experiment.sh` 冻结为只读。后续若要取得真正的 A/B 收益结论，
必须使用一个在测量前预测收益达到 2% 的 workload，或另立协议修改门槛；不能在看到本轮设备结果后
改选候选。
