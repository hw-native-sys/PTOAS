# Issue #1004：TileOp 默认 Kernel Kind 设计

## 状态

本文是 [hw-native-sys/PTOAS#1004](https://github.com/hw-native-sys/PTOAS/issues/1004)
的拟议实现设计。它只描述默认 TileOp 路径：不新增 `mixed` kernel kind，也不要求
用户选择物理计算核类型。

## 问题

`@pto.jit(..., backend="vpto")` 保留 `"vector"` 作为实际生效的默认
`kernel_kind`，以便 native build 路径沿用原有的 host 编译配置选择。decorator 同时
记录调用者是否传入过该参数：

```python
@pto.jit(target="a5", backend="vpto")
def kernel():
    ...
```

对此形式，前端生成：

```text
KernelModuleSpec.kernel_kind          == "vector"
KernelModuleSpec.kernel_kind_explicit == False
```

这两个字段的含义不同：前者是内部的实际默认值；后者是判断 PTO IR 是否包含用户
编写的物理计算核约束所需的来源信息。

当前 `_apply_child_module_attrs` 会为每个 VPTO child 写入
`pto.kernel_kind = #pto.kernel_kind<vector>`。它把内部默认值变成了显式 IR 契约。
PTOAS 随后物化 Vector TileOp section 和 Cube TileOp section 时，
`VPTOSplitCVModule` 会正确拒绝 Cube section，因为它与已显式标为 Vector 的 child
冲突。失败发生在 VPTO emission 之前。

## 目标

标准的公开编程模型应为：

```python
@pto.jit(target="a5", backend="vpto")
def gemm_epilogue(...):
    # kernel 编排位于这里。
    with pto.tileop():
        # Vector 单核计算。
        ...
    with pto.tileop():
        # Cube 单核计算。
        ...
```

用户不写 `kernel_kind="vector"`、`kernel_kind="cube"`、section hint 或
`kernel_kind="mixed"`。PTOAS 推导每个 TileOp 的单一计算域，物化相应 section，并将
mixed entry 拆分为 Vector 和 Cube backend child。

修改必须保留 `KernelModuleSpec` 中的默认值 `"vector"`，因为 native build 仍会使用
该值；但不能再把默认值序列化为用户编写的 VPTO IR intent。

## 非目标

- 不向 PTODSL 或 PTO IR 添加 `kernel_kind="mixed"`。
- 不添加公开的 TileOp section 或物理计算核 hint。
- 不修改 TileOp 单核推导、section ABI，或 `VPTOSplitCVModule` 的无 kind 拆分算法。
- 不为手写 orchestration kernel 中的 raw MTE、flag/sync 或任意 raw VPTO 指令序列
  推导 C/V 归属。TileOp 有意将 MTE 和跨 pipe 同步留在调用者中；此类推导有独立的
  正确性契约，必须单独提案。
- 不在本修改中删除已有的显式 kind 支持。它仍是向后兼容的高级路径，但新的 TileOp
  文档和示例不能要求使用它。

## 现有流水线契约

相关职责已经存在：

1. `@pto.jit` 使用私有默认 sentinel，计算实际生效的 `kernel_kind`，并记录
   `kernel_kind_explicit`。
2. PTODSL tracing 生成 TileOp helper 时不预先选择 section kind。
3. PTOAS 在 backend pipeline 之前运行 `PTOMaterializeTileOpSectionsPass`，再运行
   `PTONormalizeUncoveredTileSectionsPass`。前一个 pass 验证单一纯 TileOp 计算域，并
   物化 `pto.section.vector` 或 `pto.section.cube`。
4. `VPTOSplitCVModule` 针对每种 section kind 克隆一次无 kind module，移除相反的
   section，并且只在生成的 backend child 上设置 `pto.kernel_kind`。
5. `VPTONormalizeContainer` 接收 VPTO lowering 所需的规范 child module。

唯一错误的衔接位于第 1 步与第 4 步之间：默认值过早地写在 child-module 边界。

```text
默认 @pto.jit
  -> 实际 kind="vector", explicit=False
  -> 不带 pto.kernel_kind 的 VPTO child
  -> TileOp section 物化
  -> 无 kind 的 C/V section 拆分
  -> vector child + cube child，分别带有生成的 pto.kernel_kind
```

## 拟议修改

只修改 `ptodsl/ptodsl/_tracing/module_builder.py::_apply_child_module_attrs`。

当前 VPTO 条件将每个有效实际 kind 都视为用户编写的 kind。修正后的 VPTO 条件必须
要求其来源是显式的：

```python
if (
    spec.kernel_kind in {"cube", "vector"}
    and (
        (spec.backend == "vpto" and spec.kernel_kind_explicit)
        or (spec.backend == "emitc" and not spec.entry)
    )
):
    child_op.attributes["pto.kernel_kind"] = Attribute.parse(
        f"#pto.kernel_kind<{spec.kernel_kind}>"
    )
```

`_build_backend_partitioned_module` 中已有的 function-level 条件，会为普通 entry
检查 `spec.kernel_kind_explicit`。child-level 条件必须与其一致，以保持“实际默认值”与
“用户编写的 IR 约束”之间的唯一事实来源。

本 issue 不需要 C++ pass 修改。特别是不能放宽
`verifyExplicitKernelKindMatchesSections`：一个真正显式单 kind module 中包含相反
section，仍必须被诊断为无效。

## IR 示例

### PTOAS 物化 section 前的默认 TileOp entry

VPTO child 带有 backend 路由元数据，但不带物理 kind：

```mlir
module attributes {pto.target_arch = "a5"} {
  module attributes {pto.backend = "vpto", pto.target_arch = "a5"} {
    func.func @kernel() attributes {pto.entry} {
      // pto.tileop.helper 函数和/或 inline TileOp 调用
      return
    }
  }
}
```

### TileOp 物化与 C/V 拆分后

外层 container 对每个推导出的 kind 有一个 child。只有这些生成的 child 带有 kind：

```mlir
module attributes {pto.target_arch = "a5"} {
  module attributes {
    pto.backend = "vpto",
    pto.kernel_kind = #pto.kernel_kind<vector>,
    pto.target_arch = "a5"
  } { ... vector 代码 ... }
  module attributes {
    pto.backend = "vpto",
    pto.kernel_kind = #pto.kernel_kind<cube>,
    pto.target_arch = "a5"
  } { ... cube 代码 ... }
}
```

这是现有的规范 VPTO 输入形态。VPTO emitter 与 host stub 生成不需要感知新的 mixed
类型。

## 兼容性

- 默认 `@pto.jit` 在内部继续保留实际 `kernel_kind="vector"`；native build 的选择
  不变。
- 显式 `kernel_kind="vector"` 与 `kernel_kind="cube"` 为兼容而继续接受，并保留
  当前的显式 IR 行为。
- 默认纯 Vector 或纯 Cube TileOp 保持有效；其物理 kind 由 PTOAS 推导，而非用户断言。
- 默认 mixed TileOp entry 在其推导出的各 section 分别有效时变为有效。
- 显式 kind 与相反的推导 section 组合时，继续产生现有冲突诊断。编译器不得为满足
  自相矛盾的显式声明而静默丢弃用户代码。

## 测试计划

后续实现 PR 必须在两个层级添加聚焦覆盖。

### PTODSL tracing 回归

新增一个合法、最小的默认 `@pto.jit(backend="vpto")` fixture，其中包含一个 Vector
TileOp 和一个 Cube TileOp。断言生成的 backend child：

- 包含 `pto.backend = "vpto"`；
- 在经过 PTOAS 处理前不包含 `pto.kernel_kind`；以及
- 其私有 module spec/cache signature 保留实际 `"vector"` 和
  `kernel_kind_explicit=False`。

该测试证明前端没有将默认值序列化为 intent，且不得要求用户传入 `kernel_kind`。

### PTOAS 端到端回归

通过以下命令编译 trace 后的 IR：

```text
ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto <input> -o -
```

检查结果恰有一个 Vector child 与一个 Cube child；每个 child 必须带有生成的匹配
`pto.kernel_kind`、包含自身的 section body，且不残留 `pto.section.*`。现有
`test/lit/vpto/section_sugar_mixed.pto` 覆盖 splitter 的 raw IR 契约；新测试必须覆盖
导致 #1004 的 PTODSL 默认 child 契约。

还应保留或新增：

- 默认纯 Vector TileOp：一个 Vector child；
- 默认纯 Cube TileOp：一个 Cube child；
- 显式 Vector 加 Cube section：冲突诊断；
- 显式 Cube 加 Vector section：冲突诊断。

issue POC 可以有意省略 `pto.vecscope`；它足以证明此前的 split 失败，但不是端到端
成功用例。正向回归必须使用通过 verifier 的 Vector TileOp，确保后续 vector
verification 失败不会掩盖 split 正确性。

## 验收标准

1. 文档中的默认 TileOp 示例从不写 `kernel_kind`。
2. PTODSL tracing 完成后的 IR，其 VPTO child 不包含由默认值导出的
   `pto.kernel_kind`。
3. PTOAS 能够推导并将一个有效的默认 C/V TileOp entry 拆分为恰好两个规范 child
   module。
4. native build 的实际默认值仍为 `"vector"`。
5. 显式且矛盾的 kind/section 输入仍产生带位置的诊断。
6. 不引入公开的 `mixed` kind、section hint 或新的 TileOp 参数。
