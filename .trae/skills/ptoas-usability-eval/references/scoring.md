# 总分汇总规则

本文件只定义 **触点优先** 的汇总方法。`30` 个 `Touch-Point` 的定义、量化指标、打分规则与 `VOD` 备注，统一以 `ptoas-usability-scorecard-10pt.md` 为准；这里不再做分场景加权。

## 1. 基本原则

- 单项、分类小计、维度小计、总分全部使用 `10 分制`。
- 默认先算 `Touch-Point` 单项分，再算 `触点类别` 小计，最后算 repo 级总分。
- 默认输出两个总分：
  - `总分（支撑）`
  - `总分（实测）`
- 没有证据的项必须写成 `未实测` 或 `N/A`，不能为了凑总分补猜。

## 2. 特殊值处理

- `未实测`：不进入任何分母。
- `N/A`：不进入任何分母。
- 某个小计下如果全部是 `未实测/N/A`：
  - 该小计不输出数值
  - 标记为 `本次无有效样本`

## 3. 单项分

每个 `Touch-Point` 都必须保留四个最小字段：

- `score`
- `raw_observation`
- `evidence`
- `note`

其中：

- `score` 为 `1~10`，或 `未实测`，或 `N/A`
- `raw_observation` 必须保留原始观测值，例如检索轮次、跳转次数、步骤数、成功率、报错样本数
- `evidence` 必须给出仓内路径、命令或日志来源
- `note` 用于说明层级、前置条件、例外情况

## 4. 分类小计

默认按 `触点类别` 汇总，小计直接对 **已评分** 的触点做简单平均：

- `资料/文档`
- `API/接口`
- `源码&示例类`
- `工具`
- `版本`
- `运行反馈`

公式：

```text
分类小计 = 同类别下所有数值型 Touch-Point 分数的平均值
```

不纳入：

- `未实测`
- `N/A`

## 5. 维度小计

如果用户要看更抽象的维度，可以按 Excel 的 `维度` 列再做一层小计，例如：

- `易获取性`
- `一致性`
- `准确性`
- `完整性/完备性`
- `易实践性`
- `易部署性`
- `易调试性`
- `易学/易理解`

公式同样是简单平均：

```text
维度小计 = 同维度下所有数值型 Touch-Point 分数的平均值
```

如果用户没要求，这层可以不输出。

## 6. 总分

### 6.1 总分（支撑）

定义：所有 **已评分** `Touch-Point` 的 repo 级平均分。

纳入范围：

- 文档、接口、样例、脚本、配置、日志格式等仓内直接可观察证据
- 以及本次真实执行后得到的数值项

公式：

```text
总分（支撑） = 所有数值型 Touch-Point 分数的平均值
```

用途：衡量 PTOAS 仓库本身提供的支撑质量。

### 6.2 总分（实测）

定义：只对 **有真实执行证据** 的 `Touch-Point` 做平均。

至少满足一种证据：

- 真实命令执行
- 真实 build / compile-only
- 真实 sample 运行
- 真实 validation / compare / board 结果
- 真实报错日志采样

不纳入：

- 纯文档可发现性分
- 纯文件存在性分
- `未实测`
- `N/A`

公式：

```text
总分（实测） = 所有“带真实执行证据”的数值型 Touch-Point 分数平均值
```

用途：衡量当前会话里真正被跑到、被验证到的易用性。

## 7. 高条件触点

下面这些触点默认不是“只读仓库就能稳定量化”的项目，缺证据时应直接记 `未实测`：

- `Touch-Point017`
- `Touch-Point020`
- `Touch-Point021`
- `Touch-Point022`
- `Touch-Point024`
- `Touch-Point028`
- `Touch-Point029`
- `Touch-Point030`

原因分别见 `touchpoint-selection.md` 与 `scope.md`。

## 8. 输出要求

建议固定输出以下内容：

- `评估层级`
- `已选 Touch-Points`
- `总分（支撑）`
- `总分（实测）`
- `分类小计`
- `维度小计`（可选）
- `最低分项`
- `未实测项`
- `关键证据`

## 9. 推荐展示格式

```text
评估层级: L1 + L2
已选 Touch-Points: Touch-Point001-030
总分（支撑）: 7.8/10
总分（实测）: 6.5/10
分类小计:
- 资料/文档: 8.1/10
- API/接口: 7.4/10
- 源码&示例类: 7.0/10
- 工具: 7.9/10
- 版本: 8.0/10
- 运行反馈: 6.2/10
说明:
- `Touch-Point028-030` 只纳入了真实日志样本
- `Touch-Point021-022` 因缺 baseline 记 `未实测`
```

## 10. 结构化 JSON 对齐

如果结果要喂给 agent 侧的结构化评估链路，固定按下面口径输出：

- 顶层字段固定为 `14` 个
- `summary` 固定 `9` 个子字段
- `dimension_tables` 固定 `7` 个维度：
  - `discoverability`
  - `consistency`
  - `accuracy`
  - `completeness`
  - `learnability`
  - `practicability`
  - `debuggability`
- `test_results.pass_rate` 必须由 `passed_count / total_count` 自动回算
- `documentation_retrieval.effectiveness_rate` 必须由 `effective_searches / total_searches` 自动回算
- 取值越界时先 clamp，再校验

具体模板与校验逻辑见 `scripts/generate_evaluation_json.py`。
