# PTOAS Usability Eval Skill

这是 PTOAS 仓库内的通用 Skill 源目录。

支持的客户端入口：
- Codex: `.codex/skills/ptoas-usability-eval/`
- Cursor: `.cursor/skills/ptoas-usability-eval/`
- Trae: `.trae/skills/ptoas-usability-eval/`
- Claude Code: `.claude/skills/ptoas-usability-eval/`

当前覆盖的评估对象：
- `references/ptoas-usability-scorecard-10pt.md` 中的 `30` 个 `Touch-Point`
- `资料/文档`、`API/接口`、`源码&示例类`、`工具`、`版本`、`运行反馈` 六类触点
- `L1/L2/L3/L4` 四层环境能力

当前附带的评分基线：
- `references/ptoas-usability-scorecard-10pt.md` 直接对齐 `PTOAS-易用性评估指标.xlsx`
- 全表共 `30` 个 `Touch-Point`
- 单项、分类小计、总分统一使用 `10 分制`
- `未实测/N/A` 不进入总分分母
- 默认输出 `总分（支撑）` 和 `总分（实测）`

当前附带的结构化输出：
- `scripts/generate_evaluation_json.py`：生成 `14` 顶层字段的结构化评估 JSON
- `assets/ptoas_touchpoint_evaluation_template.json`：默认模板输出
- 模板内预置 `32` 个 PTOAS 代表性 sample 目录，作为 starter pack

当前评估逻辑：
- 先以 `references/ptoas-usability-scorecard-10pt.md` 作为分项定义与打分规则的唯一基线
- 再按 `触点类别`、证据可得性、环境层级选择本次要评的 `Touch-Point`
- 不再按旧场景编号拆分，也不再做分场景权重汇总
- 默认只把本次真正有证据的触点放进 repo 级总分

约定：
- `skills/ptoas-usability-eval/` 作为仓库内的通用主副本
- 各客户端目录提供可直接发现的副本，便于不同工具开箱即用
- 修改 Skill 内容时，应同步更新上述四个客户端目录
- 对 `L3/L4` 依赖 Linux/CANN/NPU 的指标，未实测时必须标 `未实测`，不能因为当前机器缺环境直接给 PTOAS 低分
