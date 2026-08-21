<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# PTODSL Struct 成员访问表层设计

**状态：** 设计稿 v4（Issue #1129，已按三轮 Codex review 修订）

> 修订说明：v2 将字段名→位置 path 的解析**完全移到 AST 重写期**并直接发射
> canonical `struct_get` / `struct_set`，消除运行时私有 helper、descriptor→value
> 传播、source-less 静默失效等 P1 问题。
>
> v3（本轮，针对第二轮 review）：
> - **P1 #1**：新增局部 descriptor 的**白名单静态求值**机制（不对任意 AST 用
>   `eval()`），并删除"全局需源码可定位"的错误约束（直接查 `static_env`）。
> - **P1 #2**：混合规则改为——进入 positional 层后整条成员表达式非法，须改写为完整
>   canonical path。
> - **P1 #3**：`_StructMemberRewriter` 与 `_ControlFlowRewriter` 同受
>   `rewrite_control_flow` 门控，并同步 `ast_rewrite` 选项文档。
> - **P1 #4**：EmitC 验收改为断言位置字段访问 `.f0` / `.f1.f0`，不期待源字段名。
> - **P2**：签名固定 `dict[str, FieldType]`；修正 rebinding 继承、`state.field: T`
>   写入语义、多目标赋值临时变量；删除非 struct base 的普遍诊断承诺；补充公开导出
>   与 path 断言方式。
>
> v4（本轮，针对第三轮 review）：
> - **P1**：修正多目标赋值顺序——按 `node.targets` 源码顺序从左到右逐个执行目标，
>   RHS 只求值一次；补充 `state.x = a = rhs` 与多个成员目标的反例。
> - **P2**：静态求值白名单覆盖 `static_env` 中的合法 MLIR scalar `Type`（与
>   `struct_type` 字段规则一致）；选项门控统一为公开的 `ast_rewrite` /
>   `frontend_options["ast_rewrite"]`，并明确 `rewrite_part` 只接受
>   `{"control_flow"}`；补 `AnnAssign` 注解的丢弃 / 一致性校验 / 重写期错误。

**范围：** PTODSL Python 前端（`_types.py` / `_ops.py` / `_surface_values.py`）、
AST 重写、用户手册与回归测试

**依赖：** 已合并的 `ptodsl-struct-frontend-design.md`（Issue #1104）中定义的
`!pto.struct`、`pto.struct_type`、`pto.declare_struct`、`pto.struct_get` 与
`pto.struct_set` canonical API。

## 1. 背景与动机

PR #1104 引入了与 PTO IR 对齐的 canonical struct API：用位置 path
`pto.struct_get(state, (1, 0))` 与 `pto.struct_set(state, (1, 0), value)` 读写
`pto.declare_struct(...)` 声明的栈上异构聚合 `!pto.struct<...>`。该 API 最贴近 IR
形态，但 review 指出其易用性不足：

> 希望 DSL 能够直接提供一个类似 class 的抽象，用户可以像访问 class 成员一样访问
> struct 成员，不用 `struct_set` / `struct_get`。

本设计为 PTODSL 增加一个"命名字段 + Python 成员访问"的表层，作为 canonical
path API 之上的薄、无损包装，并保留 canonical API 作为底层实现。

### 核心矛盾

- Python 的 `obj.field` 由 `__getattr__` / `__setattr__` 在运行时解析，天然支持
  任意字段名，且无法在调用 `__setattr__` 前天然区分"写入整块 struct"与"写入成员"。
- PTO IR 的字段身份是**位置**（`DenseI64ArrayAttr` path），不是名字。
- `!pto.struct` 的字段类型是**位置顺序**的；若支持任意命名，必须把名字可靠地映射
  回位置，且不能依赖 Python 运行时把字符串当作 IR path。

因此本设计引入**显式命名字段声明**（`pto.struct({...})`），由前端在 tracing 前把
名字静态解析为位置 path，再复用现有 `struct_get` / `struct_set` 的 path 检查和
lowering。字段名是编译期常量，不进入 IR。

## 2. 目标与非目标

### 2.1 目标

- 提供可声明命名字段的 PTODSL struct 表层。
- 支持成员读取 `state.field`、写入 `state.field = value` 与嵌套成员访问
  `state.inner.field`。
- 成员访问无损 lower 到现有 `pto.struct_get` / `pto.struct_set`，生成与显式 path
  API **逐位等价**的 PTO IR。
- 保持显式 path API 可用，作为底层 canonical API。
- 明确字段命名、Python class 声明方式、字面量 materialization 与诊断规则。

### 2.2 非目标

- 不改变 `!pto.struct` 的存储模型、ABI、生命周期或后端 lowering。
- 不支持动态字段名（`getattr(state, name)` / `state.__dict__` 注入）、动态索引。
- 不将 struct 用作函数 ABI（`@pto.jit` entry / `entry=False` module、
  `@pto.tileop` / `@pto.simt` 参数）。
- 不引入通用 `pto.f64` surface、Tile/TensorView/ptr 容器字段（沿用既有类型约束）。
- 不修改 canonical path API 的签名或语义。

## 3. 用户 API

### 3.1 类型构造：`pto.struct({...})`

用命名字段构造与 `pto.struct_type(...)` 等价的惰性描述符：

```python
Point = pto.struct({"x": pto.i32, "y": pto.f32})
Nested = pto.struct({
    "id": pto.i32,
    "point": pto.struct({"x": pto.i32, "y": pto.f32}),
})
```

公开签名为（**固定为 `dict[str, FieldType]` 单参数**，不接受位置列表、不接受
`*args`、不接受裸描述符）：

```python
pto.struct(fields: dict[str, FieldType]) -> _StructDescriptor
```

- `fields` 必须为 `dict[str, field_type]`（Python 3.7+ 保序）。字段类型规则与
  `pto.struct_type(...)` 完全一致：PTODSL 标量 dtype、已 materialize 的 MLIR
  scalar type，或另一个 `pto.struct({...})` / `pto.struct_type(...)` 描述符。
- 返回一个新的 `_StructDescriptor`，其字段顺序即 dict 插入顺序。因此由
  `pto.struct({...})` 构造的描述符与等价的 `pto.struct_type(...)` 解析为**同一个**
  `!pto.struct<...>` 文本表示。
- 名字是**编译期常量**，仅由前端 / AST 重写器用来把成员访问映射为位置 path；名字
  本身不进入 IR 文本、不参与 lowering。
- 空 dict、非 dict 参数、非法字段类型沿用既有 `struct_type` 诊断并补充 struct
  专属上下文。

**重名键诊断的边界**：普通 Python `dict` 在求值前就丢失重复键
（`{"x": i32, "x": f32}` 只剩一个），因此**不能承诺在运行时检测重名键**。重复键
检测只在 AST 重写期对源码中的字面量 `{...}` AST 节点进行（见 §6.4）；运行时传入
已构造 dict 的重复键无法且不检测。文档与诊断表相应改为"仅 AST 字面量 dict 检测
重名键"。

**混用规则（明确，P1 #2 修订）**：成员访问要求整条字段名链上的**每一层**都是
`pto.struct({...})` 命名的。规则如下：

- **一旦路径进入 positional struct（`pto.struct_type(...)` 层），整条成员表达式
  不合法**，用户必须改写为到达标量叶子的完整 canonical path。因为
  `struct_get` / `struct_set` 只允许 path 到达标量叶子，不能返回或写入整个嵌套
  struct；`state.pt` 本身也不是 IR 值（见 §4）。
- 因此 `state.inner.x` 只有在 `inner` 及其祖先都是 named 时才可解析；若 `inner` 是
  positional，`state.inner` 与 `state.inner.x` **都**在重写期报错，用户应改写为
  `pto.struct_get(state, (1, 0))` 这类完整标量 path。
- 允许 named 与 positional 混用，但混用只影响"能够用成员访问到达多深"，不改变
  `!pto.struct<...>` 的表示。

### 3.2 声明与成员访问

```python
@pto.jit(target="a5", mode="explicit")
def accumulate(x: pto.i32, y: pto.f32):
    S = pto.struct({"n": pto.i32, "sum": pto.f32})
    state = pto.declare_struct(S)

    state.n = x            # -> pto.struct_set(state, 0, x)
    state.sum = y          # -> pto.struct_set(state, 1, y)
    count = state.n        # -> pto.struct_get(state, 0)
    total = state.sum      # -> pto.struct_get(state, 1)
    _ = count, total
```

嵌套成员访问：

```python
P = pto.struct({
    "id": pto.i32,
    "pt": pto.struct({"x": pto.i32, "y": pto.f32}),
})

@pto.jit(target="a5", mode="explicit")
def kernel():
    s = pto.declare_struct(P)
    s.pt.x = 1            # -> pto.struct_set(s, (1, 0), 1)
    v = s.pt.y            # -> pto.struct_get(s, (1, 1))
```

### 3.3 与 canonical API 的关系

`state.field` 仅是 `pto.struct_get(state, path)` / `pto.struct_set(state, path, v)`
的语法糖，其中 `path` 由字段名静态解析。两者可混写；`struct_get` / `struct_set`
保持可用并作为唯一 canonical 底层。

## 4. 关键设计决策

1. **字段名是编译期常量，由结构性（structural）类型携带，不进入 IR。**
   名字存储于 `_StructDescriptor` 的字段元数据中；`!pto.struct` 本身仍是位置聚合。
   这保证了与既有 PTO IR / VPTO / EmitC lowering 的零改动。

2. **成员访问通过 AST 重写实现，且字段名→位置 path 在重写期完全解析并直接发射
   canonical `struct_get` / `struct_set`。**
   原因：
   - `state.field = value` 在 Python 里必然先触发 `__setattr__`，无法在求值前把
     字段名转成位置 path 并让 `state.field`（读）与 `state.field = v`（写）共享同一
     套解析逻辑。
   - 运行时 `__getattr__` 返回的标量 SSA value 无法携带"来自哪个 struct 的哪个
     成员"这一信息，无法在 `state.pt.x = v` 时把 `state.pt` 中间结果当作可写
     handle 再回填。
   - PTODSL 已有 `@pto.jit(ast_rewrite=True)` 的 source-to-source 重写基础设施
     （`_ast_rewrite.py`），可把 `attribute` 节点静态替换为对 `pto.struct_get` /
     `pto.struct_set` 的调用。
   - **重写器在重写期用 `field_index(name)` 把名字链解析为整数 tuple**，生成的
     代码是 `pto.struct_set(state, (0,), value)` / `pto.struct_get(state, (0,))`，
     **不含任何运行时私有 helper、不携带 descriptor 到 value**。这样：
     - 不需要 `state -> descriptor` 的运行时绑定（P1 #1 消除）；
     - 不需要注入/引用私有 helper（P1 #2 消除）；
     - 生成的代码直接落在 canonical surface 上，可追踪、可验证、与手写 path 一致。

3. **标量成员就是 `struct_get` 的标量 result，不引入新的值类别。**
   因此 `state.field` 可直接参与 arith 运算、作为 `if_` 条件等，与
   `pto.struct_get(state, path)` 的返回值完全等价。

4. **嵌套 struct 成员（`state.pt`）在前端 typing 期就解析为位置 path，不产生
   IR。** 它只在「基 struct + path 前缀」的意义上存在，用于把 `state.pt.x`
   折叠为 `struct_get(state, (1, 0))`。对 `state.pt` 本身求值（非继续访问标量）在
   重写期报错，因为它不是 IR 值；用户必须给出到达标量叶子的完整 path。

## 5. 命名与语法分歧点

### 5.1 字段命名规则

字段名必须同时满足以下条件，否则在 `pto.struct(...)` 声明期报错：

- **合法 Python 标识符**（`str.isidentifier()`），且**不是 Python 关键字**
  （`keyword.iskeyword(name)`）。`isidentifier()` 会接受 `class`、`await` 等关键字，
  但 `state.class` / `state.await` 不是合法 Python 语法；因此必须额外排除关键字。
- **不以 `_` 开头**（避免与内部属性冲突，预留扩展）。
- **不在实现维护的保留集合内**。保留集合由 `_StructDescriptor` / struct surface 的
  内部属性名决定，是可执行的、需随实现维护的常量集合，至少包含：
  `value`、`type`、`surface_metadata`、`field_descriptors`、`field_names`、
  `field_index`、`resolve`、`declared_value` 以及所有 `_SurfaceValue` 的既有属性。
  保留集合应在 `_types.py` 中定义为模块级 frozenset，并在测试中逐一断言冲突报错。
- 非标识符 dict 键（如 `"a-b"`）在声明期即报错，避免产生无法通过 `state.a-b`
  访问的名字。

### 5.2 Python class 声明方式（备选，均不采用）

被拒绝的备选：

- **`class` + `__annotations__`**：PTODSL 的 dtype descriptor 用作注解类型并非
  真正的 Python 类型，`__annotations__` 在 `@pto.jit` 前即可读取，但 `class` 语句
  会创建真正的 Python 类型对象，与运行时 `_SurfaceValue` 语义耦合，且 dict 保序
  与继承/`__slots__` 语义带来额外复杂度。本设计不引入新的 Python 类型层次。
- **dataclass**：引入 dataclass + 字段映射的完整对象模型，超出"薄、无损"目标。
- **`pto.struct_type` + 独立 names 参数**：`struct_type(pto.i32, pto.f32,
  names=("n","sum"))` 可用，但把名字与位置分离，嵌套时容易错位；`pto.struct({...})`
  把名字与类型绑在一起，更不易错。

因此选择 `pto.struct({...})` 这种"名字与类型同处"的声明式 dict 形式。

## 6. 前端实现

### 6.1 类型层（`_types.py`）

扩展 `_StructDescriptor`，使其同时支持两种构造路径：

- positional 构造（`pto.struct_type(...)`）：`field_descriptors` 保序，无名字。
- named 构造（`pto.struct({...})`）：保存 `_field_names: tuple[str, ...]` 与
  `_name_to_index: dict[str, int]`，`field_descriptors` 仍为保序字段类型。

新增公开 `struct(fields)`，**固定单参数**：

```python
def struct(fields: dict[str, FieldType]) -> _StructDescriptor:
    if not isinstance(fields, dict):
        raise TypeError("pto.struct(...): expected a dict[str, field_type]")
    return _StructDescriptor.from_named(fields)
```

不接受位置列表、`*args` 或裸描述符；这些误用报错并提示用 `struct_type` 或 dict。
**签名固定为 `dict[str, FieldType]`（与伪实现一致），不是 `Mapping`**：字段顺序取自
`dict` 的插入序，且 `Mapping` 的迭代序未定义、普通 `dict` 之外的其他 `Mapping` 子类
不在支持范围内。文档统一使用 `dict[str, FieldType]`。

`_StructDescriptor` 增加：

- `field_names` property（positional 构造返回 `None`）。
- `field_index(name) -> int`：查 `_name_to_index`，未知名报错。
- `field_descriptor_at(i) -> (name, type)`：返回第 `i` 个字段的名字（可能为 `None`）
  与类型，供 AST 重写器逐层解析嵌套。
- `resolve()` / `field_descriptors` 保持与 positional 构造完全一致，保证生成相同
  `!pto.struct<...>`。

### 6.2 成员访问解析（全部在 AST 重写期，无运行时 helper）

**不引入 `_StructMemberRef` 或任何运行时私有 helper。** 字段名→位置 path 的解析
完全发生在 §6.4 的 AST 重写器内：

- 重写器持有 `state -> _StructDescriptor` 的静态绑定（见 §6.4）。
- 对 `state.a.b`，重写器用 `field_index` / `field_descriptor_at` 逐层把名字解析为
  整数 tuple `indices`。
- **直接发射** `pto.struct_get(state, indices)` 或 `pto.struct_set(state, indices,
  rhs)`——即 canonical API 本身，路径已预解析好。

因此 `_ops.py` 无需新增任何可被源码引用的名称；生成的代码只引用 `pto.struct_get` /
`pto.struct_set`（已在 `pto` 命名空间及 `exec` 的 globals 中可用）。这消除了 P1 #1
的 descriptor→value 传播与 P1 #2 的 helper 注入/命名冲突问题。

### 6.3 运行时 surface 与 source-less fallback（`_surface_values.py`）

不新增**公开**的 struct surface 符号；`declare_struct` 返回内部
`StructValue`（`RuntimeValue` 子类，仍属 `_SurfaceValue` 体系）。

**source-less / 未重写路径的策略（明确，非静默）**：成员语法依赖 AST 源码可获取。
当 `inspect.getsource()` 失败（exec / REPL / notebook）或 `ast_rewrite=False` 时，
`_ast_rewrite.py` 现有逻辑返回原函数（见 `rewrite_jit_function` 的 source-less
fallback）。此时 `state.field = value` 会退化为普通 Python 实例属性赋值（不会生成
`struct_set`），读取则得到 `AttributeError`——这比显式失败更危险。

因此实现引入 `StructValue(RuntimeValue)`（`_surface_values.py`），作为
`declare_struct` 的返回类型；其 `__getattr__` / `__setattr__` **仅用于诊断**：

- 访问非内部属性时抛出明确错误：内容为"struct 成员访问需要 AST 源码重写
  （默认开启）；请保留 `@pto.jit` 的源码可获取性，或改用
  `pto.struct_get` / `pto.struct_set`"。
- **绝不**通过这些魔术方法静默生成 IR。
- 普通 `_SurfaceValue`（非 struct）不额外拦截，保持既有行为。

> 注：v4 之前的草稿曾计划"不新增 surface 类、用 `_is_struct_value` 私有标记
> 挂在 `_SurfaceValue` 上"；最终实现改用 `StructValue` 子类，诊断面相同、
> 类型边界更清晰。

这样三条路径都有确定行为：AST 重写成功（推荐）、明确报错（source-less /
`ast_rewrite=False`）、显式 canonical path API（始终可用）。

### 6.4 AST 重写（`_ast_rewrite.py`）

在 `rewrite_jit_function` 的 transformer 中新增 `_StructMemberRewriter`。它需要精确的
**符号环境**与 **rebinding 规则**，以免误重写普通 Python 属性。

#### 局部 descriptor 的静态求值（P1 #1 核心）

AST 重写发生在函数执行前，此时局部 `S` **尚未构造**，重写器不能调用
`S.field_index()`。因此重写器必须把 `pto.struct({...})` / `pto.struct_type(...)`
的赋值**静态求值**为独立的 descriptor metadata，而不是直接对任意 AST 用 `eval()`。

**静态求值机制（限定可解析语法，逐层安全求值）**：

重写器解析 `pto.struct({...})` 的 AST，为每个字段名递归求值其类型，得到元数据
`{name -> resolved_field_metadata}`。可解析的值集合（白名单）为：

- `pto.struct({...})` 字面量：字段名是字符串字面量，字段类型递归用白名单求值。
- `pto.struct_type(...)` 字面量：位置字段，无名字（`field_names is None`）。
- `pto.*` 标量 dtype：`pto.i32`、`pto.f32` 等**常量属性**——通过
  `static_env` 中已有的 `pto` + 属性名常量解析，不执行任意代码。
- 已存在于 `static_env` 的 descriptor 名（全局 / closure 中已构造的 `P`）。
- 已存在于 `static_env` 的 **合法 MLIR scalar `Type`**（P2 修订）：命名 `pto.struct`
  承诺与 `struct_type` 完全相同的字段类型规则，而 §3.1 允许字段是"已 materialize
  的 MLIR scalar type"。因此白名单必须覆盖 `static_env` 中 `ptoas.mlir.ir.Type` 的
  合法标量实例（如 `IntegerType.get_signless(32)`），与 `_resolve_struct_field_type`
  的既有校验一致；不支持的 MLIR 类型照旧报错。
- 局部 type 别名：`T = pto.struct({...})` 后 `T` 可被引用（重写器在同函数内
  AST 前瞻解析）。

**不支持**的表达式（动态表达式、函数调用返回值、下标、运算、非 `pto.*` 名称、
非 `static_env` 中已知的 MLIR `Type`）在重写期报错，提示用字面量形式或
`static_env` 中的名 descriptor / 名 MLIR Type。重写器**不**对任意 AST 调用
`eval()`——只对白名单形式做结构化求值，避免执行任意代码。

这一步产出重写器内部的 `descriptors: {name -> _DescriptorMetadata}`，其中
`_DescriptorMetadata` 是纯数据（字段名列表 + 每层类型），与运行时 `_StructDescriptor`
结构一致，但可在无 context、函数未执行时构造。

#### 全局 / closure descriptor（P1 #1 修正）

全局 / closure 的 `P = pto.struct(...)` **不需要源码可定位到赋值**。`static_env`
（`fn.__globals__` + `nonlocals`）已经包含实际的 `P` descriptor 对象，重写器直接
检查 `static_env[name]` 是否为 `_StructDescriptor` 即可。因此文档第 3.2 节的全局
`P` 示例可正常解析。上一版"源码可定位到赋值"的限制删除。

#### 符号环境（struct 绑定来源）

- **函数局部**：`pto.struct({...})` / `pto.struct_type(...)` 的赋值 /
  `AnnAssign`，经上面的静态求值产出 `var_name -> _DescriptorMetadata`。
- **全局 / closure**：`static_env[name]` 是 `_StructDescriptor` 则直接使用（见上）。
- **不支持**在分支/循环内**合并** descriptor 绑定（即不同分支赋不同 struct 类型给
  同名变量）；这类情形重写器报错，提示将 struct 类型在函数入口统一声明，或用
  canonical path。

**结构性绑定 vs 值绑定**：`state = pto.declare_struct(S)` 建立 `state` 的"struct
值"身份，指向 `S` 的 descriptor。重写器维护两类映射：

- `type_bindings: var_name -> _DescriptorMetadata`（来自 `S = pto.struct(...)`）。
- `value_bindings: var_name -> _DescriptorMetadata`（来自 `state = declare_struct(S)`）。

`declare_struct(...)` 的实参既可以是直接 `S`，也可以是内联
`pto.declare_struct(pto.struct({...}))`（后者建立 `state -> 内联 metadata` 绑定）。
`declare_struct` 的其他实参（非 struct 类型）不应建立 value binding。

**Rebinding 与取消身份（精确规则，P2 修订）**：

- 变量被重新赋值（`state = other`、`state = some_value`、`state += ...`）后，其
  struct 值身份**取消**；后续 `state.field` 不再按 struct 重写。重写器在赋值点更
  `value_bindings`：若 `other` **是 struct value binding**（即 `other` 本身在
  `value_bindings` 中）则继承其绑定，否则移除 `state` 的绑定。
- 对 `state.field += x`、链式赋值（`a = b = state.field`）、`AnnAssign`、`del
  state.field`（P2 修订）：
  - `state.field += x`：重写为 `tmp = pto.struct_get(state, path); new = tmp + x;
    pto.struct_set(state, path, new)`（读→运算→写）。
  - `state.field: T = value`（带值的注解赋值）是**写入**，lower 为
    `pto.struct_set(state, path, value)`。
  - `state.field: T`（无值注解）**不支持**，明确拒绝或定义为无操作（不做
    `struct_get`）；文档选择**拒绝**，避免无端读取。
  - **`AnnAssign` 注解处理（P2 增补）**：成员注解只允许**静态、可忽略的类型标注**。
    实现重写为 `struct_set` 时**不保留 annotation**（丢弃 `node.annotation`）。
    注解表达式须为静态形式（`pto.*` dtype 或 `static_env` 中已知 type dispatch）；
    动态注解表达式在重写期报错。若注解与字段实际类型不一致（如
    `state.field: pto.i32 = ...` 而字段是 `f32`），在**重写期**给出明确错误，而非
    仅当语法检查通过。无值注解 `state.field: T` 的重写期拒绝同时覆盖：注解即使合法
    也不产生读取。
  - `del state.field`：不支持，报错（struct 无字段删除语义）。
  - **多目标赋值（P1 修订）**：对 `Assign`（如 `a = state.x = rhs`），RHS **只求值
    一次**到临时变量，然后**按 `node.targets` 的源码顺序从左到右**逐个执行目标
    赋值。对 struct 成员目标生成 `struct_set`，对普通名称目标生成普通赋值。因此
    `a = state.x = rhs` 的正确顺序是：
    ```python
    _tmp = rhs        # RHS 求值一次
    a = _tmp          # 目标 a（源码顺序在前）
    pto.struct_set(state, path, _tmp)   # 目标 state.x
    ```
    而 `state.x = a = rhs` 的顺序是：
    ```python
    _tmp = rhs
    pto.struct_set(state, path, _tmp)   # 目标 state.x（源码顺序在前）
    a = _tmp
    ```
    多个成员目标同样按 `node.targets` 顺序逐个生成 `struct_set`。普通名称目标（包括
    其值等于 RHS 的临时变量）照常赋 `_tmp`。成员写入不产生可赋给其他目标的新值，
    所有普通目标统一取 `_tmp`。

**避免误重写普通属性（P2 修订）**：只有当 `base` 是 `value_bindings` 中的已知
struct 值变量，且整条字段名链都能解析时才重写。`pto.for_`、`loop.state`、第三方
对象属性等**不**在 `value_bindings` 中，保持原样。**不做**对任意 `obj.field` 的
猜测重写。

**非 struct base 的普遍诊断不承诺**：当 `state` 被 rebinding 出 `value_bindings`
（如 `state = pto.i32(1)` 后 `state.field = 2`），AST 不会重写，普通非 struct
`_SurfaceValue` 也不拦截，可能只是设置 Python 实例属性。因此**不承诺**在
`value_bindings` 之外对"成员访问但 base 非 struct"普遍报错。诊断只覆盖**已知为
struct value binding** 时的未知字段等情形；对已脱离 struct 身份的 base，行为与普通
Python 属性一致（不重写、不报错）。可选增强（不承诺）：若前端能跟踪"某变量曾是
struct value binding 但被重赋值"，可对后续成员访问给出提示，但这是非阻塞增强。

**实际重写产出**（P1 #1/#2 的闭合方案）：

- 读 `x = state.a.b` → `x = pto.struct_get(state, (1, 0))`（重写器已把 `["a","b"]`
  解析为整数 tuple）。
- 写 `state.a.b = v` → `pto.struct_set(state, (1, 0), v)`。
- 生成的代码只引用 `pto.struct_get` / `pto.struct_set`，二者已在 `pto` 命名空间，
  且 `exec` 的 globals 已含 `pto`；无需注入私有 helper。

**AST 字面量 dict 的重名键检测**：对 `pto.struct({...})` 的字面量 `{...}` 节点，若
其中出现重复键，在重写期报错（运行时 dict 无法检测，见 §3.1）。

**错误处理**：

- 未知字段名：抛 `PTODSLAstRewriteError`，信息含 struct 上下文与合法字段名列表。
- `state.pt` 单独求值（非成员，且 `pt` 是嵌套 named struct）：报错，提示继续访问
  到标量叶子或用 `pto.struct_get` / canonical path。
- 链上某层是 positional（`field_names is None`）却继续访问成员：报错，提示该层
  用 canonical path。

**选项门控（P2 修订，对齐公开 API 名称）**：公开 `@pto.jit` 选项没有
`rewrite_control_flow` 参数；实际 API 是 `ast_rewrite=...` 与
`frontend_options={"ast_rewrite": ...}`。`_StructMemberRewriter` **不引入新的公开
选项名**，而是与 `_ControlFlowRewriter` 同受 `ast_rewrite` /
`frontend_options["ast_rewrite"]` 门控，因为成员赋值重写依赖 control-flow 重写对
`state.field += x` 等生成的临时变量与 SSA 语义保持一致：

```python
if ast_rewrite:   # 即 frontend_options.get("ast_rewrite", True)
    function_def = _StructMemberRewriter(...).visit(function_def)
    rewriter = _ControlFlowRewriter(...)
    function_def.body = rewriter.rewrite_block(...)
```

当 `ast_rewrite=False` 时，`_StructMemberRewriter` **不运行**，成员访问不重写，落入
§6.3 的 `_SurfaceValue` 诊断路径。这与"`ast_rewrite=False` 下成员访问必须走诊断"
的设计一致。

**`rewrite_part` 的语义**：`frontend_options["rewrite_part"]` 当前只接受
`{"control_flow"}`，用于在 control-flow 重写内细分。成员重写不单独暴露 rewrite
part；若未来要独立控制成员重写，应另行设计新的 rewrite part 名，而不是复用未公开
名称。**公开文档更新**：`ast_rewrite` / `frontend_options["ast_rewrite"]` 不再只
控制 control flow，还控制 struct 成员访问重写；需在 `@pto.jit` 选项文档注明该语义
扩展。

重写发生在 `_ControlFlowRewriter` 之前，保证成员赋值不会被后续控制流重写误判。
重写后的函数体即 canonical `struct_get` / `struct_set`，因此既有 path 验证、字面量
materialization 与诊断全部复用。

### 6.5 字面量 materialization

`state.n = 7` 复用 `struct_set` 的 `_materialize_struct_value`：整型字面量写整型
字段、也可 materialize 为浮点字段；`float` 仅可写浮点字段；`bool` / 字符串 /
typed-SSA 精确匹配规则全部沿用。AST 重写只负责把 `state.n` 变成
`struct_set(state, path, rhs)`，不额外引入类型转换。

### 6.6 诊断规则

| 场景 | 诊断 |
|---|---|
| `pto.struct({})` 空 dict | 至少需要一个字段 |
| `pto.struct([...])` 位置列表 / 非 dict | 提示用 `pto.struct_type(...)` 或 dict |
| 字段名非标识符 / 是 Python 关键字 / 以 `_` 开头 / 保留名 | 声明期报错，列出合法形式 |
| 重名键 | 仅对 AST 字面量 `{...}` 在重写期报错；运行时 dict 无法检测 |
| 访问未知字段 | 报字段名 + 合法字段名列表 |
| 链上某层是 positional nested 却继续访问成员 | 提示改写为到达标量叶子的完整 canonical path |
| `state.pt` 求值（非继续访问标量） | 提示继续访问到标量或用完整 canonical path |
| 分支/循环内合并 struct 类型绑定 | 报错，提示入口统一声明或用 canonical path |
| `del state.field` | 不支持，报错（struct 无字段删除语义） |
| `state.field: T`（无值注解） | 不支持，拒绝（不做 `struct_get`） |
| 成员访问但 base 已脱离 struct 身份 | **不承诺普遍报错**（见 §6.4 避免误重写）；与普通属性一致 |
| 写入类型不匹配 | 复用 `struct_set` 现有诊断 |
| struct 作 ABI / carry state | 复用既有拒绝逻辑 |
| source-less / `ast_rewrite=False` 下访问成员 | 明确报错，提示保留源码或用 canonical path |

## 7. 兼容性与后端

- 由 `pto.struct({...})` 构造的描述符与等价 `pto.struct_type(...)` 解析为同一个
  `!pto.struct<...>`，因此对 EmitC / VPTO LLVM lowering **零改动**。
- 既有 `struct_type` / `declare_struct` / `struct_get` / `struct_set` 签名与语义
  不变；`pto.struct` 是新增命名空间符号。
- 现有 `test_struct.py` 全部保持通过；新增用例只覆盖命名与成员访问表层。

**公开导出（P2 增补）**：新增 `pto.struct` 需同步三处导出，并加测试：

- `ptodsl/ptodsl/_types.py`：将 `struct` 加入 `__all__`。
- `ptodsl/ptodsl/pto.py`：从 `._types` 显式 import 并导出 `struct`（沿用现有
  named-import 风格，见 `pto.py` 中 `struct_type` 的导入）。
- 新增 public namespace export 测试：断言 `hasattr(pto, "struct")` 且
  `"struct" in _types.__all__`（与现有 `test_struct.py::test_public_namespace_exports_struct_surface`
  的 `struct_type` 断言并列）。

**命名冲突的申明（supersede #1104 决定）**：Issue #1104 的设计文档曾以"避免与
Python `struct` 标准库和未来结构化 value helper 混淆"为由选择 `struct_type` 而非
`struct`。本设计新增 `pto.struct` 是**有意的需求演进**，在此明确 supersede 该命名
决定的一部分：

- `pto.struct_type` **保留**并继续作为 positional canonical 构造器（`#1104` 的
  既定 API 不变）。
- `pto.struct` **新增**作为 field mapping 构造器，用于成员访问表层。
- 二者在 `pto` 命名空间下靠 `_` 与参数形态区分，Python 用户经 `pto.` 前缀调用，
  不会与标准库 `struct` 模块（需 `import struct`）冲突。
- 若后续需要，`pto.struct` 也可作为未来结构化 value helper 的命名空间入口，这
  与它承担"命名字段构造器"的角色不冲突。

一句话：`struct_type` 的"避免与 `struct` 混淆"约束针对的是**不使用 `pto.` 前缀的
裸名**；在 `pto.` 命名空间内新增 `struct` 不产生该混淆，且 `struct_type` 仍保留。

## 8. 文档与测试

### 8.1 用户手册

在 `ptodsl/docs/user_guide/04-type-system-and-buffer.md` 的 Struct 小节增加"命名字段
与成员访问"小节：`pto.struct({...})` 声明、读 / 写 / 嵌套示例、与 canonical path
API 的关系、字段名与保留字规则、AST 重写开启要求。示例纳入
`docs_fragment_fixtures.py` docs-as-tests。

### 8.2 回归测试

`ptodsl/tests/test_struct.py` 新增：

- `pto.struct({...})` 与 `pto.struct_type(...)` 解析为同一 `!pto.struct` 文本；
- **AST 重写**：`state.field` 读 / 写被重写为 `pto.struct_get(state, (0,))` /
  `pto.struct_set(state, (0,), v)`，断言生成的 canonical op 的 **path 属性**
  与手写等价，并用 path 属性值相等而非整段 MLIR 文本比较（避免 SSA 名 / location
  差异）。path 从 op 的 `DenseI64ArrayAttr` 读取：`op.operation.attributes["path"]`
  （仓库无 `CompositeType`，不引用它）；
- 嵌套成员访问（≥ 2 层）生成 `[1, 0]` / `[1, 1]` 形式 path；
- 成员访问与显式 path API 混写；
- 字段名不入 IR：断言生成的 canonical op 的 path 是整数、且 `!pto.struct` 类型文本
  不含字段名字符串；
- 未知字段、非标识符/关键字/保留名、`state.pt` 单独求值、positional-nested 继续
  访问、`del state.field`、分支合并类型绑定等在重写期报错；
- rebinding 后 `state.field` 不再按 struct 重写；
- 与现有 `--emit-pto-ir` frontend verification 一起通过。

**EmitC / VPTO 端到端编译（新增，证明重写确实工作）**：

- 一个使用 `pto.struct({...})` + `state.field` 的 `backend="emitc"` 嵌套例子：
  parse + verify，并断言生成的 C++ 包含 **EmitC 按位置生成的字段访问 `.f0` /
  `.f1.f0`**（见 `test/lit/pto/struct_nested_emitc.pto`）。字段名不进入 IR，因此
  **不**期待源字段名 `n` / `pt.x` 出现在 C++ 中——只检查位置成员访问；
- 一个使用默认 VPTO backend 的等同例子：parse + verify，并断言 `pto.struct_get` /
  `pto.struct_set` 在 VPTO LLVM lowering 后变为 GEP/load/store；
- 与手写 `struct_get` / `struct_set` 的等价例子对比，确认两者生成**相同**的
  field-path / 类型 / lowering 结果。

## 9. 验收矩阵

| 层级 | 验收内容 |
|---|---|
| Python 单测 | 命名 / 成员访问在重写期映射为等价 canonical path，字段名不入 IR |
| AST 重写 | 重写产物只含 `pto.struct_get` / `pto.struct_set`，无私有 helper |
| 文档 | 用户示例在 docs-as-tests 中编译 |
| PTOAS frontend | 生成 module 在 `--emit-pto-ir` 下验证通过 |
| EmitC | 嵌套成员例子 parse/verify，C++ 输出含位置字段访问 `.f0` / `.f1.f0`（非源字段名） |
| VPTO | 嵌套成员例子 parse/verify，get/set lower 为 GEP/load/store，与手写 path 一致 |

## 10. 需要保持的决定

- 字段名是编译期常量，由描述符携带，不进入 IR。
- 字段名→位置 path 的解析**完全在 AST 重写期**完成，重写产物直接是
  `pto.struct_get` / `pto.struct_set`，无运行时私有 helper、无 descriptor→value
  传播。
- 局部 descriptor 在重写期经**白名单静态求值**（字面量 `pto.struct(...)` /
  `pto.struct_type(...)` / `pto.*` dtype / `static_env` 中的 descriptor / 合法 MLIR
  scalar `Type` / 局部别名），不对任意 AST 调 `eval()`。
- 全局 / closure descriptor 直接查 `static_env`，不要求源码可定位到赋值。
- 成员访问通过 AST 重写实现，受公开选项 `ast_rewrite` /
  `frontend_options["ast_rewrite"]` 门控（不引入 `rewrite_control_flow` 等未公开
  名称；`rewrite_part` 只接受 `{"control_flow"}`）；`ast_rewrite=False` 时不重写、
  落入 `_SurfaceValue` 明确诊断。`_SurfaceValue` 的 `__getattr__` / `__setattr__`
  仅用于诊断，绝不静默生成 IR。
- 多目标赋值按 `node.targets` **源码顺序从左到右**逐个执行目标，RHS 只求值一次；
  成员目标生成 `struct_set`，普通目标赋 RHS 临时值。
- 成员 `AnnAssign` 注解只允许静态可忽略标注，重写为 `struct_set` 时不保留 annotation，
  注解与字段类型不一致或无值注解在重写期报错。
- 嵌套成员在前端即折叠为位置 path，`state.pt` 本身不是 IR 值；一旦进入 positional
  层，整条成员表达式不合法，须改写为完整 canonical path。
- `pto.struct({...})` 与 `pto.struct_type(...)` 解析为同一 `!pto.struct`；成员访问
  要求链上每层都是 named。
- 标量成员即 `struct_get` 的标量 result，无新值类别。
- EmitC 生成位置字段访问（`.f0` / `.f1.f0`），**不**含源字段名；验收按此断言。
- struct 生命周期与 ABI 约束沿用既有规则，本设计不改变。