# 证据清单

## 固定证据入口

| 路径 | 主要用途 | 层级 | 主要触点 | 备注 |
| --- | --- | --- | --- | --- |
| `README.md` | 官方构建、环境变量、CLI、Python 绑定、sample 运行、compile-only/上板验证主入口 | `L1-L4` | `Touch-Point001-005`, `Touch-Point008-011`, `Touch-Point014-015`, `Touch-Point018`, `Touch-Point024-027` | repo 级首入口 |
| `docs/no_npu_compile_only_guide_zh.md` | 无卡 compile-only 流程、批量验证流程、`pto-isa`/CANN 依赖说明 | `L1`, `L3` | `Touch-Point005`, `Touch-Point008-010`, `Touch-Point018`, `Touch-Point024-026`, `Touch-Point028-030` | Linux/CANN 依赖主说明 |
| `docs/PTO_IR_manual.md` | IR 层级、tile/view/valid-shape、layout、dynamic shape、Level-2/3 语义 | `L1-L4` | `Touch-Point007`, `Touch-Point012-013`, `Touch-Point017`, `Touch-Point019`, `Touch-Point021-023` | API / IR 入口主证据 |
| `test/samples/runop.sh` | 批量样例生成、`ptoas`/`ptobc` 运行、A3/A5 默认参数策略 | `L1-L4` | `Touch-Point014-020`, `Touch-Point023`, `Touch-Point025-026`, `Touch-Point028-030` | sample 主执行器 |
| `test/npu_validation/scripts/generate_testcase.py` | 从 `*-pto.cpp` 生成验证工程，观察 golden/compare/兼容层处理 | `L1`, `L3`, `L4` | `Touch-Point016`, `Touch-Point018-020`, `Touch-Point022-023`, `Touch-Point026`, `Touch-Point028-030` | validation 生成入口 |
| `test/npu_validation/scripts/run_remote_npu_validation.sh` | compile-only / sim / npu 运行链路、日志格式、设备与 `pto-isa` 检查 | `L1`, `L3`, `L4` | `Touch-Point024-030` | 运行反馈主证据 |
| `test/samples/PyPTOIRParser/README.md` | 来自 pypto `ir_parser` 的 vendored `.pto` 快照说明 | `L1` | `Touch-Point001`, `Touch-Point003`, `Touch-Point012-013`, `Touch-Point017`, `Touch-Point019`, `Touch-Point021` | 迁移 / API 样例入口 |
| `test/samples/MatMul/` | README 直接引用的基准样例，适合作为默认复现模板 | `L1-L4` | `Touch-Point014-016`, `Touch-Point018-021`, `Touch-Point023` | 默认 sample |
| `test/samples/FlashAttention/` | attention 类固定 shape 样例 | `L1-L4` | `Touch-Point019`, `Touch-Point021-023`, `Touch-Point026`, `Touch-Point028-030` | 性能 / 复杂链路样例 |
| `test/samples/GQA/` | attention 组合样例 | `L1-L4` | `Touch-Point019`, `Touch-Point021-023`, `Touch-Point026`, `Touch-Point028-030` | 性能 / 复杂链路样例 |
| `test/samples/FFN/` | 算子组合样例 | `L1-L4` | `Touch-Point019`, `Touch-Point021-023`, `Touch-Point026`, `Touch-Point028-030` | 性能 / 复杂链路样例 |
| `test/samples/SetValidShape/` | dynamic/valid-shape 相关样例 | `L1-L4` | `Touch-Point019`, `Touch-Point021-023`, `Touch-Point026`, `Touch-Point028-030` | 泛化 shape 证据 |
| `test/samples/LayoutInference/` | layout 推断相关样例 | `L1-L4` | `Touch-Point012-013`, `Touch-Point019`, `Touch-Point021-023`, `Touch-Point026`, `Touch-Point028-030` | 泛化 shape / IR 证据 |
| `test/samples/Partition5D/` | 多维 partition / shape 泛化相关样例 | `L1-L4` | `Touch-Point017`, `Touch-Point019`, `Touch-Point021-023`, `Touch-Point026`, `Touch-Point028-030` | 多 shape / 迁移对照可选 |
| `test/samples/planmemory/` | alias/planmemory/shape 相关样例 | `L1-L4` | `Touch-Point019`, `Touch-Point021-023`, `Touch-Point026`, `Touch-Point028-030` | shape / plan 复杂链路 |
| `.github/workflows/ci.yml` | CI 中的 LLVM/PTOAS 构建、lit、sample test、remote validation 参考配置 | `L1`, `L3`, `L4` | `Touch-Point003`, `Touch-Point008-010`, `Touch-Point024-027`, `Touch-Point028-030` | 版本与执行链路旁证 |
| `.github/ISSUE_TEMPLATE/performance_issue.yml` | 性能问题受理模板，可用来评估性能数据/复现要求的完备性 | `L1` | `Touch-Point003`, `Touch-Point022`, `Touch-Point026`, `Touch-Point029` | 关键链路显性化旁证 |

说明：
- 不要把当前分支不存在的样例 README 当成固定证据源。
- 没有“前后对照 / baseline / 真实运行日志”时，不要硬算 `Touch-Point017`、`Touch-Point021`、`Touch-Point022`、`Touch-Point028-030`。

## 推荐检索顺序

1. `README.md`
2. `docs/no_npu_compile_only_guide_zh.md`
3. `docs/PTO_IR_manual.md`
4. `test/samples/MatMul/` 或用户指定样例目录
5. `test/samples/PyPTOIRParser/`, `FlashAttention/`, `GQA/`, `FFN/`, `SetValidShape/`, `LayoutInference/`, `Partition5D/`, `planmemory/`
6. `test/samples/runop.sh`
7. `test/npu_validation/scripts/*.py` / `*.sh`
8. `.github/workflows/ci.yml`
9. `.github/ISSUE_TEMPLATE/performance_issue.yml`

## 推荐检索命令

```bash
rg -n "构建|运行测试|compile-only|runop|generate_testcase|run_remote_npu_validation|level3" README.md docs test .github
rg -n "valid_shape|layout|partition|reshape|dynamic shape|Level-2|Level-3" docs/PTO_IR_manual.md docs test
rg -n "FlashAttention|GQA|FFN|MatMul|SetValidShape|LayoutInference|Partition5D|planmemory" test .github
rg --files test/samples
find test/samples -maxdepth 2 -type f \( -name '*.py' -o -name '*.pto' -o -name 'README.md' \)
```

## 高条件触点的补充记录项

如果本次纳入以下触点，额外记录：
- `Touch-Point017` / `Touch-Point021`：是否存在迁移前/迁移后对照物、PR diff、baseline 来源路径
- `Touch-Point022`：是否存在 PyPTO / 性能 / 精度关键链路文档或脚本入口
- `Touch-Point028-030`：是否存在真实错误日志、日志来源路径、是否来自 `L2/L3/L4` 实测
- `Touch-Point024-026`：是否真实跑过部署/编译/validation 链路，还是只做文档审阅
- 统一记录：当前停在哪一层、哪些分数是实测、哪些是文档侧支撑分

## 记录要求

每个评分项至少要落这些证据字段：

- `证据路径`
- `检索/执行命令`
- `检索轮次`
- `文档跳转次数`
- `评估层级`
- `耗时`
- `结果`
- `评分`
- `备注`

## 默认样例

若用户没有指定具体算子或样例，优先使用：

- `test/samples/MatMul/tmatmulk.py`
- `test/samples/MatMul/tmatmulk.pto`
- `test/samples/Addc/addc.py`
- `test/samples/PyPTOIRParser/`
- `test/samples/FlashAttention/`
- `test/samples/SetValidShape/`

理由：这些路径要么被 `README.md` 直接引用，要么能稳定覆盖文档、API、样例、工具、运行反馈等多类触点。
