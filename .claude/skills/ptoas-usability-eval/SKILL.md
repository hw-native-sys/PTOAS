---
name: ptoas-usability-eval
description: Evaluate PTOAS repository usability by the 30 touch points from the 10-point scorecard. Always classify the evaluation by environment layer first, score by touch point instead of scene, use only repo-native docs/scripts/samples/CI as primary evidence, and mark unsupported or untested items as 未实测 or N/A.
---

# PTOAS Usability Eval

当用户要评估 `hw-native-sys/PTOAS` 的易用性，或要按 `PTOAS-易用性评估指标.xlsx` 里的 `30` 个 `Touch-Point` 给 PTOAS 打分时，使用这个 Skill。

## 默认范围

- 默认以 `references/ptoas-usability-scorecard-10pt.md` 的 `30` 个 `Touch-Point` 作为唯一分项基线。
- 默认做 repo 级触点评估，不再按旧场景编号拆分主流程。
- 旧场景字样如果出现在 Excel/CSV 的“说明”里，只作为该触点的适用提示，不再作为评分入口或总分权重依据。
- `03 builtin 算子定制修改` 这类明显超出 PTOAS repo 可自证范围的需求，仍然按 `N/A` 处理。

先读 [references/touchpoint-selection.md](references/touchpoint-selection.md) 选定适用触点，再读 [references/scope.md](references/scope.md) 确认 repo 边界、层级边界和 `未实测/N/A` 规则。

## 先判层级

开始评分前，必须先声明本次评估覆盖到哪一层。没有层级，不能直接混着打分。

可选层级：
- `L1 文档审阅层`：只看仓库文档、脚本、样例、CI，不做运行。
- `L2 本地最小运行层`：当前机器已有 `ptoas` / `ptobc` / Python 绑定，可做最小命令验证。
- `L3 Linux compile-only 层`：需要 Linux + CANN/bisheng + `PTO_ISA_ROOT`，不要求带卡。
- `L4 NPU 上板层`：需要带卡 Linux、驱动、权限、`/dev/davinci*` 与对应用户组。

约束：
- 没进入某层，就把该层指标记为 `未实测`，不能因为当前机器缺环境就给 PTOAS 低分。
- `bisheng` / CANN compile-only 一般属于 `L3`，不应在本地 Mac 上硬打低分。
- 带卡运行、设备权限、驱动、ACL、用户组属于 `L4`。

## 证据来源

优先只用仓库内证据，不把仓外经验当成主证据。固定入口见 [references/evidence-checklist.md](references/evidence-checklist.md)。

高优先级证据：
- `README.md`
- `docs/no_npu_compile_only_guide_zh.md`
- `docs/PTO_IR_manual.md`
- `test/samples/runop.sh`
- `test/npu_validation/scripts/generate_testcase.py`
- `test/npu_validation/scripts/run_remote_npu_validation.sh`
- `test/samples/PyPTOIRParser/README.md`
- `test/samples/FlashAttention/`, `test/samples/GQA/`, `test/samples/FFN/`
- `test/samples/SetValidShape/`, `test/samples/LayoutInference/`, `test/samples/Partition5D/`, `test/samples/planmemory/`
- `.github/workflows/ci.yml`
- `.github/ISSUE_TEMPLATE/performance_issue.yml`

## 工作流

1. 先判断本次覆盖层级：`L1/L2/L3/L4`。输出中必须显式写出来。
2. 再读 [references/touchpoint-selection.md](references/touchpoint-selection.md)，按 `触点类别`、证据可得性和用户关注点选定本次 `Touch-Point` 范围。
3. 读 [references/ptoas-usability-scorecard-10pt.md](references/ptoas-usability-scorecard-10pt.md)，以 Excel 同步下来的 `30` 个 `Touch-Point` 定义、量化指标、打分规则与 VOD 备注为准。
4. 从仓库内收集证据，记录每次检索轮次、文档跳转次数、执行命令、耗时、成功/失败结果。
5. 读 [references/evidence-checklist.md](references/evidence-checklist.md) 确认证据入口和每类触点的推荐取证路径。
6. 需要判断 repo 边界、未实测/N/A 口径时，读 [references/scope.md](references/scope.md)。
7. 需要汇总总分或分类小计时，读 [references/scoring.md](references/scoring.md)。
8. 对每个触点都输出：原始观测值、评分、证据路径、说明。没有实测的数据不要猜，记为 `未实测` 或 `N/A`。
9. 明确区分：
    - PTOAS 仓库已提供的能力
    - 外部前置条件，例如 LLVM、CANN、`pto-isa`、NPU、驱动/权限、业务 baseline
10. 若文档描述与实际运行冲突，以实际命令结果为准，并指出冲突位置。
11. 默认给两个总分：`总分（支撑）` 和 `总分（实测）`。如果用户只要分项，不强制输出总分。
12. 如果用户要结构化 JSON，输出必须是**触点优先**：
    - 优先对齐 `scripts/generate_evaluation_json.py` 里的 `14` 个顶层字段
    - 用 `summary` / `dimension_tables` / `documentation_retrieval` / `functional_testing` 承接触点评分结果
    - `dimension_tables` 固定用 `7` 个维度，且每个维度保留 `3~4` 个子维度整数分
    - 不要再设计 `scene_scores`、`scenes`、旧场景权重这类字段

## 计量规则

- 单项、分类小计、总分统一使用 `10 分制`。
- 分项字段定义、适用层级、量化指标、打分规则、VOD 备注，以 [references/ptoas-usability-scorecard-10pt.md](references/ptoas-usability-scorecard-10pt.md) 为准。
- `检索轮次`：每次新的定向搜索或定位尝试算 1 轮。
- `文档跳转次数`：命中首个目标文档后，每跨一个文档/README/脚本入口算 1 次。
- `耗时`：尽量记录真实墙钟时间；拿不到就写 `未实测`，不要臆测。
- `成功率`：只基于当前任务里真实执行或真实定位到的结果计算。
- `未实测`：当前会话未覆盖到对应环境层级，或该层级前置条件不存在，或缺少前后对照 baseline。
- `N/A`：只用于超出 PTOAS 能力边界，或当前任务明确不纳入本次评估范围的项。

## 输出格式

按下面顺序输出：

1. `评估层级`
2. `触点选择`
3. `总分（支撑）`
4. `总分（实测）`
5. `分触点评分`
6. `分类小计`
7. `覆盖说明`
8. `关键证据`
9. `主要短板`
10. `建议动作`

如果用户只要简版结论，也要至少保留：评估层级、已选触点、总评、最低分项、证据路径。

## 结构化 JSON

- 用户要 agent 侧结构化结果时，直接使用 `scripts/generate_evaluation_json.py`。
- 该脚本会：
  - 生成固定 `14` 个顶层字段
  - 对齐 `summary` 的 `9` 个子字段
  - 固定 `dimension_tables` 的 `7` 个维度
  - 自动执行 `_postprocess_evaluation_json()` 风格的修正
  - 自动执行 `_validate_evaluation_json()` 风格的校验
- 默认模板输出会带一组 PTOAS 代表性 case 清单；当前 repo 里默认选择几十个典型样例，作为批量评估的 starter pack。
