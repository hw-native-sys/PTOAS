# PTOAS Touch-Point 选型

本文件定义 `30` 个 `Touch-Point` 在 PTOAS repo 里的默认选法。原则是 **先按触点类别与证据可得性选型，再按层级决定哪些项能实测**，不再按旧场景编号拆分。

## 1. 选型原则

- `ptoas-usability-scorecard-10pt.md` 是分项定义与打分规则的唯一基线。
- 默认做 repo 级评估，不按场景切分主流程。
- 每次评估都要先声明层级，再声明本次纳入哪些触点。
- 没有证据的项记 `未实测`；超出 PTOAS repo 边界的项记 `N/A`。

## 2. 30 个 Touch-Point 的分组

### 2.1 资料 / 文档

- `Touch-Point001` 检索命中成功率
- `Touch-Point002` 文档跳转次数
- `Touch-Point003` 多入口可达率
- `Touch-Point004` 单次任务文档跳转浏览率
- `Touch-Point005` 知识渐进式发布
- `Touch-Point006` 文档结构风格一致率
- `Touch-Point007` 概念跨文档冲突数
- `Touch-Point008` 文档错误点位密度
- `Touch-Point009` 文档场景 / 内容覆盖缺失率
- `Touch-Point010` 版本配套关系准确性
- `Touch-Point011` 资料交付件完备率

### 2.2 API / 接口

- `Touch-Point012` 目标接口平均查找检索轮次
- `Touch-Point013` 渐进式复杂披露覆盖度

### 2.3 源码 & 示例类

- `Touch-Point014` 示例代码一键编译运行成功率
- `Touch-Point015` quick_start / sample 一次跑通率
- `Touch-Point016` 样例覆盖度
- `Touch-Point017` 最小功能实现 Demo 覆盖率
- `Touch-Point018` 命令示例覆盖度
- `Touch-Point019` API 调用样例覆盖率
- `Touch-Point020` 样例代码编译错误检出与修复效率
- `Touch-Point021` 业务代码直接复用改编比例
- `Touch-Point022` 关键链路显性化率
- `Touch-Point023` 认知理解步数

### 2.4 工具 / 版本 / 运行反馈

- `Touch-Point024` 首次安装部署一次性成功率
- `Touch-Point025` 标准任务平均操作步骤数
- `Touch-Point026` 功能 / 场景覆盖率
- `Touch-Point027` 版本检索命中成功率
- `Touch-Point028` 报错携带环境 / 版本 / 上下文信息完整率
- `Touch-Point029` 报错自带排障建议比例
- `Touch-Point030` 无效冗余信息占比

## 3. 默认选型包

### 3.1 `Full Pack`

适用：用户要完整 repo 级总评。

- 默认纳入 `Touch-Point001-030`
- 其中高条件项如果缺证据，保留在清单里但记 `未实测`

### 3.2 `Doc-First Pack`

适用：只做文档和仓内静态证据审阅。

- `Touch-Point001-013`
- `Touch-Point018-019`
- `Touch-Point023`
- `Touch-Point025-027`

默认不实测：

- `Touch-Point014-017`
- `Touch-Point020-022`
- `Touch-Point024`
- `Touch-Point028-030`

### 3.3 `Build/Run Pack`

适用：用户关心 sample、compile-only、validation、日志反馈。

- `Touch-Point014-020`
- `Touch-Point024-030`
- 视需要补 `Touch-Point001-005` 与 `Touch-Point010`

### 3.4 `High-Condition Pack`

适用：用户明确要看迁移复用、关键链路、安装成功率、日志质量这类高条件触点。

- `Touch-Point017`
- `Touch-Point020`
- `Touch-Point021`
- `Touch-Point022`
- `Touch-Point024`
- `Touch-Point028-030`

前提：必须先说明证据来源，不然直接记 `未实测`。

## 4. 高条件触点的纳入规则

- `Touch-Point017`：需要对照样例、baseline、功能清单或可验证的目标功能集合。
- `Touch-Point020`：需要真实编译或 validation 过程，不能只看 README。
- `Touch-Point021`：需要 PR diff、迁移前后 case，或明确的复用前后材料。
- `Touch-Point022`：需要关键链路入口、脚本或流程证据，常见于 PyPTO、性能、验证链路。
- `Touch-Point024`：需要真实从零安装部署记录。
- `Touch-Point028-030`：需要真实错误日志样本，不能只凭“脚本看起来写得不错”打高分。

## 5. 默认排除项

本 Skill 只覆盖 `ptoas-usability-scorecard-10pt.md` 这 `30` 个 `Touch-Point`。任何不在这张表里的生态级、产品矩阵级、友商对标级指标，默认都不纳入。

## 6. 使用要求

- 每次正式评估前，先在输出里给出 `评估层级` 和 `触点选择`。
- 如果用户没指定，就默认用 `Full Pack`。
- 如果证据明显不足，降级到 `Doc-First Pack` 或 `Build/Run Pack`，不要硬开全量实测。
- 同一个触点只能按当前层级与当前证据给分，不能拿其他会话经验补分。
