---
name: resolve-dsl-issue
description: 根据用户提供的 issue 链接，提取 DSL 与 PTO IR 复现最小用例，运行 PTOAS 复现并分析日志，在用户指导下完成修复、提交并自动创建关联 issue 的 PR。
---

# Resolve DSL Issue

当任务满足以下任一条件时使用本 skill：
- 用户明确提供了要处理的 issue 链接
- 用户希望“按 issue 内容复现 DSL 问题并定位根因”

不建议作为主入口的场景：
- 仅做编译/构建，不涉及 issue 复现
- 仅做 NPU 运行验证，不涉及 DSL/PTO IR 复现

## 目标

从 issue 中抽取可执行复现输入（DSL + PTO IR），在仓库内构造最小复现并定位根因；在用户确认修复方向后完成代码修复、验证、提交，并自动创建关联原始 issue 的 PR。

## 前置条件

- 当前目录是 PTOAS 仓库根目录
- `build/` 目录可写
- `ptoas` 可执行（已在 PATH 或有明确绝对路径）
- 能访问 issue 内容（网页、API、或用户粘贴）
- 若需要自动创建 PR：`gh` CLI 已安装并登录（`gh auth status` 成功）

## 标准流程

### 1. 解析 issue，提取两个代码片段

必须提取到两类片段：
- DSL 代码片段（`.py`）
- PTO IR 代码片段（`.pto`）（如果是纯DSL前端问题，可以不需要 PTO IR）

推荐提取顺序：
1. issue 正文
2. issue 评论
3. issue 附件/粘贴内容

如果任一片段缺失，停止后续复现，直接在 issue 请求补充（模板见“评论模板”）。

### 2. 在仓库中落盘复现文件

文件位置固定为：
- DSL: `lib/TileOps/`
- PTO IR: `test/dsl/`

命名建议使用 issue 编号，避免冲突，例如：
- `lib/TileOps/issue_<id>_repro.py`
- `test/dsl/issue_<id>_repro.pto`

要求：
- 原样写入，避免“自动修复”代码导致偏离用户输入
- 保留 issue 中的关键注释和输入形状信息

### 3. 执行编译并保存日志

标准命令：

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --enable-tile-op-expand --vpto-emit-hivm-llvm <pto_file> &> <log_file>
```

推荐日志路径：
- `build/issue_<id>_repro.log`

示例：

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --enable-tile-op-expand --vpto-emit-hivm-llvm \
  test/dsl/issue_1234_repro.pto \
  &> build/issue_1234_repro.log
```

### 4. 分析日志，判断是否复现

先定位关键错误信息（error/fatal/assert/traceback），再判断是否与 issue 描述一致。

推荐快速检索：

```bash
rg -n "error|fatal|assert|traceback|failed" build/issue_<id>_repro.log
```

分支处理：
- 未复现：在 issue 中请求更完整的复现信息（环境、命令、输入、预期/实际）
- 已复现：进入根因定位

### 5. 根因定位与方案建议

定位输出应至少包含：
- 触发错误的阶段（前端解析/TileOp 展开/Lowering/LLVM 发射等）
- 直接触发点（具体报错行、pass、或输入约束不满足）
- 根因判断（1-2 条最可能原因，标注置信度）
- 修复建议（最小改动优先）

如果无法在当前上下文完成修复实现，也需要给出：
- 建议修改文件范围
- 建议新增/补充的测试用例

### 6. 与用户确认修复方向（必须）

在进入代码修改前，先向用户同步：
- 复现文件路径
- 复现命令
- 关键报错摘要
- 根因与建议
- 待确认项（如环境差异）

只有在用户明确同意修复方向后，才进入第 7 步。

### 7. 实施修复并本地验证

修复要求：
- 仅改动与该 issue 直接相关的最小文件集合
- 优先补充或更新回归测试（如 `test/dsl` 相关用例）
- 保留复现输入，避免把“复现文件”误删

验证要求：
- 至少重新执行一次复现命令，确认错误消失或行为符合预期
- 将关键验证日志保存到 `build/issue_<id>_fix_verify.log`
- 跑一次完整的dsl测试集，确认无其他回归

### 8. 提交代码（在用户确认后执行）

分支命名建议：
- `fix/issue-<id>-dsl`

提交信息建议（至少包含 issue 编号）：
- `fix(dsl): <简要修复描述> (#<id>)`

示例命令：

```bash
git checkout -b fix/issue_1234_dsl
git add <changed_files>
git commit -m "fix(dsl): handle <case> in <op> (#1234)"
git push -u origin fix/issue_1234_dsl
```

### 9. 自动创建 PR 并关联原始 issue

目标仓库：https://github.com/mouliangyu/PTOAS/
目标分支：feature-vpto-backend

关联规则（GitHub）：
- 在 PR 描述中包含 `Closes #<id>` 或 `Fixes #<id>`
- 若是跨仓库 issue，使用 `Closes <owner>/<repo>#<id>`
- 合并后删除分支

推荐使用 `gh pr create`：

```bash
gh pr create \
  --base main \
  --head fix/issue_1234_dsl \
  --title "fix(dsl): <简要修复标题>" \
  --body "$(cat <<'EOF'
## Summary
- <修改点1>
- <修改点2>

## Repro
- issue: #1234
- repro cmd: `ptoas --pto-arch=a5 --pto-backend=vpto --enable-tile-op-expand --vpto-emit-hivm-llvm test/dsl/issue_1234_repro.pto`

## Validation
- <验证命令/结果>

Closes #1234
EOF
)"
```

创建 PR 后需要回填：
- PR 链接
- 关联 issue 语句是否生效（是否显示 “linked issues”）

### 10. 结果同步并等待 review

向用户同步：
- 修复文件列表
- 提交 hash
- PR 链接
- 关联 issue 状态
- 后续待办（例如 reviewer 关注点）

## 评论模板

### 模板 A：缺少 DSL 或 PTO IR 片段

```text
为了准确复现该问题，还需要完整的最小复现输入。请补充以下两段代码：
1) DSL Python 片段（可直接运行到生成该 PTO 的部分）
2) 对应的 PTO IR 片段（完整函数/入口，不要省略关键上下文）

建议同时提供：执行命令、实际报错、期望行为。
```

### 模板 B：当前未复现

```text
我已按当前 issue 信息完成复现尝试，但暂未在本地复现相同报错。
请补充以下信息以便继续定位：
1) 完整执行命令（含所有 flags）
2) 运行环境（分支/commit、CANN 版本、是否自定义环境变量）
3) 实际报错全文（建议粘贴日志片段）
4) 期望结果与当前结果差异
```

### 模板 C：已复现并给出建议

```text
已使用 issue 中输入复现成功，关键报错位于：<阶段/文件/日志行号>。
初步根因：<根因描述>。
建议修复：<最小修复方案>。

如果你同意该方向，我会继续补充对应测试并提交修复实现供 review。
```

### 模板 D：修复完成，准备提交与开 PR

```text
修复已完成并通过本地验证。
计划执行：
1) 提交分支：fix/issue-<id>-dsl
2) 创建 PR 并在描述中添加 `Closes #<id>` 自动关联 issue

请确认是否按该方案提交并创建 PR。
```

### 模板 E：PR 已创建并关联 issue

```text
PR 已创建：<pr_url>
已在 PR 描述中添加 `Closes #<id>`，原始 issue 已自动关联。

本次提交：
- Commit: <commit_hash>
- 关键修改：<summary>
- 验证结果：<summary>
```

## 执行注意事项

- 不要在未确认复现之前改动用户原始输入语义
- 优先保留最小复现，不做无关重构
- 若 issue 信息不完整，先补信息再继续，不要猜测输入
- 日志分析时优先使用首次错误点，不要只看最后一行报错
- 未经用户确认，不要直接执行 `git commit`、`git push`、`gh pr create`
- PR 关联语句建议统一放在 PR body 末尾，避免被模板覆盖
- 若 `gh` 未登录或无权限，输出完整 PR 标题/body 草稿供用户手动创建

## 最终输出格式（给用户）

建议按以下顺序输出：
1. 是否复现成功
2. 复现文件路径与命令
3. 日志关键错误（1-3 条）
4. 根因判断
5. 修复建议与下一步计划
6.（若完成修复）提交信息、PR 链接、issue 关联状态
