# Tile Lib 向量库方案设计

## 第一章 背景与问题

### 1.1 当前编译栈与编译时长问题

当前从 DSL 到硬件二进制的完整编译栈如下：

```
PTO DSL (TileLang...)
       ↓
     PTOAS (MLIR)
       ↓
  Tile Lib (CCE)          ← C++ 模板库
       ↓
     CCEC                 ← C++ 编译器
       ↓
    LLVM IR
       ↓
    BiSheng
       ↓
  Davinci Binary
```

这条编译栈层次较深。PTOAS 生成 CCE C++ 代码后，需要经过 C++ 模板实例化和 CCEC 编译才能产生 LLVM IR，再由 BiSheng 编译器生成最终的 Davinci 二进制。其中 **C++ 模板实例化和 CCE 编译** 是主要的编译时间瓶颈。

我们希望简化编译栈，跳过 CCE 代码生成和编译的过程，直接从 PTOAS 输出 LLVM IR：

```
PTO DSL (TileLang...) + Tile Lib
       ↓
     PTOAS (MLIR)         ← 直接输出 LLVM IR，跳过 CCE
       ↓
    LLVM IR
       ↓
    BiSheng
       ↓
  Davinci Binary
```

这样可以显著缩短编译时间。但当前的 Tile Lib 是基于 CCE 和 C++ 模板开发的，因此需要用其它方式重新实现 Tile Lib。

### 1.2 PTOAS 中向量库实现的挑战

PTOAS 中目前设计两层粒度的 IR：

- **PTO TileOp**：面向上层用户的高层抽象，操作对象是 `tile_buf`，一条指令表达完整的 tile 语义（如 `pto.tadd`、`pto.tmul`、`pto.tload`）。
- **Vector IR (vPTO)**：面向底层硬件的指令接口，操作对象是 `vreg`/`ptr`，需要显式循环、显式寄存器宽度、显式 mask 处理（如 `pto.vadd`、`pto.vlds`、`pto.vsts`）。

Tile Lib 的一种实现方式是直接使用 Vector IR 编写。以 `pto.tadd`（逐元素加法）为例，在 TileOp层只需一条指令：

```mlir
pto.tadd ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, ...>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, ...>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, ...>)
```

而用 Vector IR 实现同样的语义（`dtype=f32, rows=16, cols=64`），需要展开为完整的向量循环：

```mlir
func.func @TADD(
    %a: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %b: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %c: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>) {
  %vecA = pto.tile_buf_addr %a : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
      v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
  %vecB = pto.tile_buf_addr %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
      v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
  %vecC = pto.tile_buf_addr %c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
      v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index

  pto.vecscope {
    scf.for %arg0 = %c0 to %c16 step %c1 {           // 遍历 rows
      scf.for %arg1 = %c0 to %c64 step %c64 {         // 遍历 cols，步长=vector_width
        %va = pto.vlds %vecA[%arg0, %arg1] : memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>> -> !pto.vreg<64xf32>
        %vb = pto.vlds %vecB[%arg0, %arg1] : memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>> -> !pto.vreg<64xf32>
        %vc = pto.vadd %va, %vb : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>
        pto.vsts %vc, %vecC[%arg0, %arg1] : !pto.vreg<64xf32>, memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
      }
    }
  }
  return
}
```

直接基于 Vector IR 开发 Tile Lib 面临以下困难：

1. **MLIR 语法门槛高**：需要熟悉 `memref`、`index`、`strided` 等 MLIR 数据类型和语法，使用 MLIR 的方式定义变量、表达运算和控制流。
2. **参数组合无法穷举**：`dtype` 有 f16/f32/bf16 等，`rows`/`cols` 可以是任意正整数，为每种 `(op, dtype, rows, cols, layout)` 组合手写向量实现不可行。

因此，直接基于 PTO Vector IR 开发 Tile Lib，技术难度大且工作无法收敛。

## 第二章 方案：使用 Python 开发 Tile Lib

### 2.1 总体思路

为了降低开发门槛并解决参数组合的穷举问题，我们采用 **TileLang Python DSL** 来编写
Tile Lib 的向量库实现。库开发者使用 Python 编写 vkernel 函数，PTOAS 编译器在编译时
根据具体的 Tile op 以及操作数类型进行匹配、特化（specialization）和实例化（instantiation）。

TileLang DSL 的完整语法定义在 `tilelang-dsl/docs/tilelang-dsl-guide.md`，本章在该文档
基础上，聚焦于本方案所依赖的语言子集及其语义约束。

整体方案：

1. **用 TileLang Python DSL 编写 vkernel**：以 `@pto.vkernel` 装饰器声明匹配元数据
   （`target` / `op` 或 `ops` / `dtypes` / `constraints` / `priority`），函数体使用
   `pto.Tile` 数据类型和基础向量指令（`make_mask` / `vlds` / `vsts` / `vadd` / …）
   按 Tile 指令语义编写向量实现。
2. **编译器匹配并特化 vkernel**：PTOAS 遇到 Tile op 时，通过 DSL 提供的
   `pto.select_kernel(target, concrete_op, operand_types, …)` 匹配候选 vkernel，按 DSL Guide
   §Kernel Selection Mechanism 的规则（target → op → dtypes → constraints → priority）
   选出一条，再以调用点的具体 `tile_buf` 类型作为 specialization key 进行特化，生成
   以 `tile_buf` 为形参的向量实现函数。
3. **inline 到调用点**：特化后的向量 IR 以 `func.call` 形式插入到原 Tile op 的位置，
   随后由 `PTOInlineLibCall` pass inline 到调用点，继续后续优化和 lowering 流程。

### 2.2 TADD 模板示例

以 `pto.tadd`（逐元素加法）为例，TileLang DSL 编写的 vkernel 如下（`PAT` 是
`pto.MaskPattern` 的别名；算子名按 DSL Guide §Kernel Declaration 约定，不带 `pto.` 前缀）：

```python
from pto import MaskPattern as PAT

@pto.vkernel(
    target="a5",
    op="tadd",                              # 匹配 pto.tadd
    dtypes=[(pto.f32, pto.f32, pto.f32)],   # 操作数类型签名 (src0, src1, dst)
    advanced=True,                          # 启用隐式 vecscope 推断
)
def template_tadd(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile) -> None:
    dtype = dst.element_type                 # 编译期静态
    valid_rows, valid_cols = dst.valid_shape # 静态或动态

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            summed = pto.vadd(lhs, rhs, mask)
            pto.vsts(summed, dst[row, col:], mask)
    return None
```

代码解读：

- **`@pto.vkernel`** 装饰器声明本 kernel 匹配 `a5` 架构下的 `tadd` 算子、操作数签名
  `(f32, f32, f32)`。`advanced=True` 让编译器对函数体内的 `vlds`/`vadd`/`vsts` 序列
  自动推断 `pto.vecscope`，无需显式 `with pto.vecscope():` 包裹
  （详见 DSL Guide §Implicit Scope Inference）。
- **kernel 参数**为 3 个 `pto.Tile` 对象（2 个输入 `src0` / `src1`，1 个输出 `dst`），
  对应 VPTO IR 中的 `!pto.tile_buf` 类型，它们是实例化时被特化的 symbolic value。
  **参数顺序必须与 PTOAS 中对应指令的操作数顺序一致**（即 `ins` 在前、`outs` 在后），
  因为 `ExpandTileOp` 按位置索引直接传递操作数。
- 通过 **`Tile` 属性接口**读取元素类型 `element_type` 和 `valid_shape`。参考 DSL Guide
  §Tile Attributes：`shape` / `element_type` / `memory_space` / `config` 都是编译期静态
  值，`valid_shape` 允许为静态或动态。
- **2 层循环**分别遍历 tile 的行和列。外层步长 1，内层步长为 `pto.get_lanes(dtype)`
  （单个向量寄存器可容纳的元素数，f32→64，f16→128）。
- **`pto.make_mask(dtype, remained)`** 按 DSL Guide §Typed Masks 的 tail-processing 语义，
  返回 `(mask, new_remaining)`，并根据 `dtype` 自动选择正确的 mask 粒度
  （`f32` → `mask_b32`、`f16` → `mask_b16`、`i8` → `mask_b8`）。
- **Tile 元素级索引语法糖** `src0[row, col:]` 实现向量宽度的 load/store
  （DSL Guide §Address Generation Syntax Sugar）：`col:` 后缀表示以 `col` 为起点、按
  向量宽度连续读取；编译器按 `element_size` 和 layout 自动计算字节偏移，避免手写
  `i * cols * 4` 之类易错的算术。
- **`pto.vadd(lhs, rhs, mask)`** 执行逐元素加法；**`pto.vsts(summed, dst[row, col:], mask)`**
  将结果带 mask 写回 `dst`。

### 2.3 值模型与 Staging 语义

TileLang DSL 按 DSL Guide §Value Model 的定义，采用 **symbolic value** 模型——函数体中的
值并非 Python 运行时的 `int`/`float`，而是编译器构造的 SSA 值或编译期常量。在 vkernel
实例化过程中，`pto.Tile` 参数的属性按两种 stage 区分处理：

#### 编译期静态值（Compile-time Static）

以下属性在 vkernel 实例化时已经确定，由 TileLang Codegen 在编译期折叠为字面量，
**不会**生成 MLIR SSA 值：

| 属性 | 来源 | 说明 |
|------|------|------|
| `element_type` | `tile_buf` 的 `dtype` 字段 | 决定 vreg 类型和向量宽度；参与 specialization key |
| `element_size` | 由 `dtype` 推导 | f32→4, f16→2, i8→1 |
| `shape` | `tile_buf` 的 `rows`, `cols` 字段 | **必须是编译期静态值**，参与 specialization key |
| `memory_space` | `tile_buf` 的 `loc` 字段 | `MemorySpace.GM` / `MemorySpace.UB`；参与 specialization key |
| `config` | `tile_buf` 的 blayout / slayout / fractal / pad | 决定 stride 模式和偏移计算方式 |

这些值在 Python 层直接参与运算（如 `pto.get_lanes(dtype)`、`rows * cols * element_size`），
结果在编译期确定。DSL Guide §Tile Types 明确规定 **Static Shape Requirement**：
`shape` 必须是 compile-time constant。

#### 运行时 SSA 值（Runtime Dynamic）

以下属性可能在编译期未知，生成为实例化函数的参数或 SSA 值：

| 属性 | 来源 | 说明 |
|------|------|------|
| `valid_shape` | `tile_buf` 的 `v_row`, `v_col` 字段 | **可以是静态也可以是动态**（DSL Guide §Tile Shape Concepts） |

当 `valid_shape` 为静态值时，TileLang Codegen 在编译期折叠（与 `shape` 相同处理方式）；
当为动态值时，生成为实例化函数的 `index` 类型参数，循环边界等依赖它的地方生成
`scf.for`。该参数在 PTOAS 侧由 `pto.bind_tile` 的 `valid_row` / `valid_col`
操作数承载（参见第三章）。

#### 正式约束

1. **`shape` 必须是编译期静态值**，并参与 specialization key。若 `shape` 为动态值，
   vkernel 实例化应报错拒绝。
2. **`valid_shape` 可以是静态也可以是动态**。当为静态值时，TileLang Codegen 应检查
   `valid_shape[i] ≤ shape[i]`（逐维度），对齐 DSL Guide §Tile Shape Concepts 的约束。
3. **`element_type`、`element_size`、`memory_space`、`config` 必须是编译期静态值**，
   它们决定了函数体的结构（vreg 类型、向量宽度、stride 模式等）。

#### 对控制流的影响

```python
rows, cols = src0.shape           # 编译期静态 → Python 层直接展开或折叠
v_rows, v_cols = src0.valid_shape # 可能是动态 → 生成 scf.for

for i in range(0, v_rows, 1):    # v_rows 动态 → scf.for %i = 0 to %v_rows
    for j in range(0, v_cols, 64): # v_cols 动态 → scf.for %j = 0 to %v_cols step 64
        ...

# 对比：如果用 shape（静态），Python 层可以直接展开
for i in range(0, rows, 1):       # rows=16 静态 → Python 展开 16 次迭代
    ...
```

### 2.4 TileLang DSL 语法参考

本节摘录本方案所依赖的 DSL 子集；完整定义见 `tilelang-dsl/docs/tilelang-dsl-guide.md`。

#### 2.4.1 基础标量类型

| DSL 类型 | 说明 | 位宽 |
|----------|------|------|
| `pto.i1`  | 布尔 | 1 |
| `pto.i8`  | 8 位整数 | 8 |
| `pto.i16` | 16 位整数 | 16 |
| `pto.i32` | 32 位整数 | 32 |
| `pto.i64` | 64 位整数 | 64 |
| `pto.f16` | 半精度浮点 | 16 |
| `pto.bf16`| Brain float 16 | 16 |
| `pto.f32` | 单精度浮点 | 32 |

Python 字面量自动推导类型：`bool` → `pto.i1`，`int` → 上下文决定（通常 `pto.i32`/`pto.i64`），
`float` → `pto.f32`。需要显式类型时可用 `x = pto.i32(1024)` 或类型注解。

DSL 还提供类型通配符 `pto.AnyFloat` / `pto.AnyInt` / `pto.AnyType` / `pto.AnyMask`
和类型变量 `pto.TypeVar(...)`，用于在 `dtypes=` 中写多态签名。

#### 2.4.2 向量与 Mask 类型

向量寄存器固定 **256 字节** 宽度：

```python
pto.vreg(64,  pto.f32)   # 64 lanes × 32 bit = 2048 bit
pto.vreg(128, pto.f16)   # 128 lanes × 16 bit = 2048 bit
```

约束：`lanes × bitwidth(element_type) == 2048`。可用 `pto.get_lanes(dtype)` 获得 lane 数。

Mask 按位粒度分型（DSL Guide §Typed Masks），必须与 vreg 元素族匹配：

| DSL 类型 | VPTO 类型 | 对应元素族 |
|----------|-----------|-----------|
| `pto.mask_b8`  | `!pto.mask<b8>`  | `i8` 向量 |
| `pto.mask_b16` | `!pto.mask<b16>` | `f16` / `bf16` / `i16` 向量 |
| `pto.mask_b32` | `!pto.mask<b32>` | `f32` / `i32` 向量 |

粒度不匹配（例如 `f32` 向量配 `mask_b16`）会在类型检查阶段报错。

#### 2.4.3 Tile 数据类型

`pto.Tile` 表示一个带有布局和配置信息的数据块，对应 VPTO IR 中的 `!pto.tile_buf` 类型。

**Tile 属性接口**（DSL Guide §Tile Attributes）：

| 属性 | 类型 | 说明 |
|------|------|------|
| `shape` | `tuple[int, ...]` | **编译期静态**的物理维度（rows, cols） |
| `valid_shape` | `tuple[int, ...]` | 有效数据维度（v_row, v_col），可为静态或动态，须 ≤ `shape` |
| `element_type` | `Type` | 元素类型，如 `pto.f32` |
| `element_size` | `int` | 元素字节大小 |
| `memory_space` | `MemorySpace` | `MemorySpace.GM` / `MemorySpace.UB` |
| `config` | `TileConfig` | 布局与 padding 配置 |
| `rank` / `num_elements` / `valid_elements` | `int` | 派生属性 |

**Tile 配置枚举**：

```python
pto.BLayout.ROW_MAJOR / pto.BLayout.COL_MAJOR        # 基础布局
pto.SLayout.NONE_BOX / pto.SLayout.ROW_MAJOR / pto.SLayout.COL_MAJOR
pto.PadValue.NULL / pto.PadValue.ZERO / pto.PadValue.MAX / pto.PadValue.MIN
```

**地址生成语法糖**（DSL Guide §Address Generation Syntax Sugar）——向量级读写使用
元素索引语法，编译器自动按 layout 计算字节偏移：

| 语法 | 含义 |
|------|------|
| `tile[row, col:]` | 行主序：从 `(row, col)` 起按向量宽度连续读 |
| `tile[row:, col]` | 列主序：从 `(row, col)` 起按向量宽度连续读 |
| `tile[start:]` | 1D tile：从 `start` 起按向量宽度连续读 |
| `tile[row, col]` | 单元素（仅 `pto.vsld` 等 broadcast load 使用） |

#### 2.4.4 向量操作接口

本方案依赖 DSL Guide §Operations 中列在 **`stable`** tier 的 base vector ops：

**Mask 生成**（DSL Guide §`pto.make_mask`）：

| 形式 | 说明 |
|------|------|
| `pto.make_mask(dtype, remaining: pto.i32)` | Tail processing：返回 `(mask, new_remaining)` |
| `pto.make_mask(dtype, PAT.ALL)` | 固定 pattern：返回单值 `mask`。其它 pattern 包括 `PAT.EVEN`/`PAT.ODD` 等 |

**向量 Load / Store**：

| 操作 | 说明 |
|------|------|
| `pto.vlds(tile[row, col:])` | 从 tile 的 `(row, col)` 按向量宽度加载到 vreg |
| `pto.vsts(vec, tile[row, col:], mask)` | 将 vreg 按 mask 写入 tile 的 `(row, col)` |

上述两条也支持 DSL Guide 中的 byte-offset 形式 `pto.vlds(buf, offset)` / `pto.vsts(vec, buf, offset, mask)`
（Advanced Tier），但模板库优先使用元素索引语法。

**基础二元/一元算子**（用于常见 Tile op 的展开）：

| 操作 | 说明 |
|------|------|
| `pto.vadd / vsub / vmul / vdiv(vec1, vec2, mask)` | 逐元素二元运算 |
| `pto.vmax / vmin(vec1, vec2, mask)` | 逐元素比较 |
| `pto.vabs / vexp / vln / vsqrt / vrelu(vec, mask)` | 逐元素一元运算 |
| `pto.vmuls / vadds(vec, scalar, mask)` | 向量-标量运算 |

#### 2.4.5 控制流

**循环**使用 Python 的 `range` 语法：

```python
for i in range(0, valid_rows, 1):
    for j in range(0, valid_cols, pto.get_lanes(dtype)):
        ...
```

当循环边界来自 `shape`（编译期常量）时，DSL 在 Python 层直接展开循环；当来自
`valid_shape`（可能是动态值）时，生成 `scf.for` MLIR 循环。

**向量作用域**：本方案的 vkernel 统一使用 `advanced=True`，由编译器的 Scope Inference Pass
对连续、数据依赖的 `vlds`/`vadd`/`vsts` 序列自动推断 `pto.vecscope` 边界，库开发者无需
显式书写 `with pto.vecscope(): ...`。需要精确控制时可使用 `strict_vecscope`（Advanced Tier）。

#### 2.4.6 多算子模板（template slots）

对于计算骨架相同、仅核心算子不同的一组 Tile op（如 `tadd`/`tsub`/`tmul`/`tdiv`），
可用 DSL Guide §Template-based Kernel Authoring 的 `ops=[...]` + `templates=` + `pto.tpl(...)`
在一个 vkernel 中共享实现：

```python
@pto.vkernel(
    target="a5",
    ops=["tadd", "tsub", "tmul", "tdiv"],
    dtypes=[(T, T, T)],
    advanced=True,
    templates={
        "core": {"tadd": "vadd", "tsub": "vsub",
                  "tmul": "vmul", "tdiv": "vdiv"},
    },
)
def elementwise_arithmetic(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    dtype = dst.element_type
    rows, cols = dst.valid_shape
    for row in range(0, rows, 1):
        remained = cols
        for col in range(0, cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            out = pto.tpl("core", lhs, rhs, mask)   # 按选中的具体算子替换
            pto.vsts(out, dst[row, col:], mask)
```

编译期 `pto.select_kernel(...)` 会把具体的 `tadd`/`tsub`/… 绑定到 `selected_op`，
`pto.tpl("core", ...)` 再按 `templates["core"]` 的映射展开为真正的 `vadd`/`vsub`/… 调用。
这样本方案的 Tile Lib 可以用一份模板覆盖四条逐元素算子，显著收敛维护成本。

## 第三章 PTOAS 编译器：TileOp Expand

### 3.1 编译流程

PTOAS 编译器的输入可以是 Tile 指令、向量指令、或两者的混合。TileLang/VPTO 与
PTODSL VMI provider 在 Expand 前共享 Tile frontend，但 Expand 后进入不同的融合路径。
PTODSL VMI 的 RFC 目标 pipeline 如下：

```
输入：TileOp / 向量指令 / TileOp + 向量指令混合
       ↓
  PlanMemory                ← UB 内存分配规划
       ↓
  InsertSync                ← 管线同步插入
       ↓
  Expand TileOp             ← 将 TileOp 替换为对实例化模板函数的调用
       ↓
  Inline                    ← 将模板函数体 inline 到调用点
       ↓
  Fold TileBuf Intrinsics   ← 先折叠 shape family，暴露循环边界
       ↓
  VMI Fusion                ← 识别 TileLib unit，保守合并兼容 VMI 循环
       ↓
  VMI Mem2Reg               ← 融合后消除同 location 的中间 UB store/load
       ↓
  Fold TileBuf Intrinsics   ← 再折叠 address family
       ↓
  VMI Layout / VMIToVPTO    ← 选择物理 layout 并降到 VPTO
       ↓
  Infer VPTO VecScope       ← emission boundary 统一推断物理 vecscope
```

VMI Fusion 的详细设计见 [`vmi-vf-fusion-design.md`](vmi-vf-fusion-design.md)。
现有 Tile-native `FusionPlan/FusionRegionGen` 和 VPTO `PTOLowLevelLoopFusion` 不直接
充当 VMI Fusion pass；它们的分析思想或通用 utility 可以复用。

Tile 指令到向量指令的展开由三个 pass 协作完成：

1. **Expand TileOp**：核心 pass。调用配置的 TileOp provider 实例化模板库，生成以 `tile_buf` 为参数的实现函数，将原 Tile op 替换为对该函数的 `func.call`。
2. **Inline**：将模板函数体 inline 到调用点，使模板函数的 `tile_buf` 形参与调用点的实际 `tile_buf` 值绑定。
3. **Fold TileBuf Intrinsics**：折叠 inline 后留下的 tile_buf 系列（`pto.tile_buf_addr`、`pto.tile_valid_rows`、`pto.tile_valid_cols`）和 tensor_view 系列（`pto.tensor_view_addr`、`pto.get_tensor_view_dim`、`pto.get_tensor_view_stride`）intrinsic，将 `tile_buf` / `partition_tensor_view` 的属性折叠为具体的 memref、常量和 SSA 值。

`ExpandTileOp` 当前支持三种 TileLib backend：

| backend | 选项 | 实现来源 | 输出 IR |
|---|---|---|---|
| PTODSL | `--tile-lib-backend=ptodsl`（默认） | 现有 PTODSL TileLib registry/daemon | 由已注册 candidate 决定 |
| Composite PTODSL VMI | `--tile-lib-backend=ptodsl-vmi` | `PIPE_V` 使用 `ptodsl.vmi_tilelib`；其他 Pipe 使用现有 PTODSL TileLib registry/daemon | 向量计算为 Unified VMI，搬运/Cube 边界沿用现有 PTODSL 实现 |
| TileLang | `--tile-lib-backend=tilelang` | legacy `@pto.vkernel` 模板目录 | VPTO/MI `pto.vlds/vadd/vsts` |

PTODSL VMI provider 还支持 `--ptodsl-vmi-provider-module`、`--ptodsl-pkg-path` 和
`--ptodsl-python-exe`。它要求 VPTO backend 且 `--enable-vmi=true`。该选项是组合式
provider：通过 `OpPipeInterface::getPipe()` 逐个选择实现，`PIPE_V` TileOp 必须匹配
PTODSL VMI candidate，`PIPE_MTE1/MTE2/MTE3/FIX/M` 等非向量 TileOp 继续使用现有
PTODSL TileLib daemon。该 backend 不把非向量 TileOp 回退到 TileLang。
因此 `tload/tstore` 等 GM/UB 搬运不会要求 VMI candidate，而 UB 内部向量计算不会因
candidate 缺失而静默回退到 MI。当前 VMI TileLib 已覆盖静态 Softmax compute harness 的
`tadd/tadds/tsub/tmul/tmuls/tmax/tmaxs/tmins/tdivs/tmov/texp/trowmax/trowsum/`
`trowexpandsub/tcvt`：dense f32 Tile 的 physical inner 固定为 64 lanes，每行恰好一个
logical block；RowReduce 使用单个 row loop 和单个输入 block，`[rows,1]` compact result
作为辅助 Tile；Convert 支持 `64xf32 -> 64xf16` 并沿用源 iteration contract。
`tdivs` 当前只覆盖 tile/scalar 的 f32 default
precision，通过 `vbrc + vdiv` 组合实现。
当前仍要求静态 Shape 和静态 valid shape，尚未覆盖动态 tail mask。在新的 VMI Fusion
pipeline 接入之前，该 provider 会拒绝 `--enable-op-fusion`，防止误入旧的 VPTO/MI
loop-fusion pass。

`test/samples/FlashAttention/flash_attention_softmax.pto` 的静态 `[64,64]xf32` 多项式计算
路径已经可以完成 Expand、Inline 和 VMI-to-VPTO。该样例不包含标准 Softmax 的 RowReduce
归一化，因此不能替代完整 FA/Online Softmax 验收；动态 row/column、tail mask、row-wise
归一化和全部 Convert/高精度除法变体仍未覆盖。

RFC 首期还要求每个 `(target, PIPE_V TileOp)` 恰好存在一个 canonical VMI
implementation。该实现必须在无融合时独立正确执行。PTODSL VMI helper 不做多个模板间
的 priority 选择，也不存在 candidate locking；多 candidate、schedule family 和
region-aware cost model 均属于后续性能迭代。

VMI provider module 必须显式暴露 `VMI_TILELIB_REGISTRY`。模板通过
`@canonical_vmi_template` 在模块导入时注册，Expand helper 只从该 registry 按
`(target, op)` 查询，不扫描 Python module 全局变量。VMI 使用独立 registry，避免与普通
PTODSL TileLib 中同名的 VPTO/MI candidate 混合选择。

### 3.2 Expand TileOp Pass 的工作流程

以编译时遇到 `pto.tadd` 为例，Expand TileOp pass 的处理步骤如下：

```
Step 1: 识别 Tile Op 并分类操作数
────────────────────────────────
  遍历函数体中所有 Tile op（pto.tadd, pto.tload, ...）
  每个操作数按 IR 类型分为三类：
    Tile   — TileBufType（如 pto.tadd 的输入/输出 tile_buf）
    View   — MemRefType（如 pto.tload 的 src，由 PTOViewToMemref 降级的 partition_tensor_view）
    Scalar — 标量类型（如 pto.tadds 的 scalar 操作数）

Step 2: 构造 Specialization Key + 查询缓存
──────────────────────────────────────────
  根据 Tile op 的所有操作数构造 specialization key（见 3.2.1）
  查询实例化缓存：
    如果缓存命中，直接复用已实例化的函数，跳到 Step 4

Step 3: 选择 backend 并实例化模板（缓存未命中时执行）
────────────────────────────────────────────────────
  --tile-lib-backend=tilelang：所有 TileOp 调用 legacy TileLang helper
  --tile-lib-backend=ptodsl：所有 TileOp 调用 PTODSL TileLib daemon
  --tile-lib-backend=ptodsl-vmi：
    PIPE_V                         → PTODSL VMI helper
    PIPE_MTE1/MTE2/MTE3/FIX/M... → PTODSL TileLib daemon
  传入 op 名称、target、各操作数 schema 和静态 context attrs
  backend 查找匹配的 @vkernel 或 TileTemplate，填入具体参数进行特化
  输出实例化后的 MLIR 函数，解析文本，克隆到目标 Module，写入缓存

Step 4: 生成调用并替换原 Tile Op
───────────────────────────────
  在原 Tile op 位置插入 backend-specific func.call
    TileLang:    @__pto_tilelang_...
    PTODSL VMI: @__pto_ptodsl_vmi_...
  - Tile 操作数：类型一致，直接传递
  - View 操作数：调用方类型为 memref，模板参数类型为 partition_tensor_view，
    插入 builtin.unrealized_conversion_cast 桥接（由后续 FoldTileBufIntrinsics 消除）
  - Scalar 操作数：直接传递
  删除原 Tile op
```

#### 3.2.1 Specialization Key 与缓存

模板展开本质上是一个特化过程。当同一个 module 中存在多个相同类型的 Tile op（如多处 `pto.tadd` 且所有操作数类型完全相同），应复用已实例化的结果而非重复展开。

**重要**：SpecKey 必须基于 **所有操作数** 的类型构建，而不仅仅是第一个操作数。因为同一个 op 的不同操作数可能有不同的类型（如不同的 dtype 或 shape），仅用第一个操作数无法区分这些情况。

操作数按 IR 类型分为三类，每类参与 SpecKey 的字段不同：

| 操作数类型 | IR 类型 | 参与 SpecKey 的字段 | 不参与 SpecKey 但传给 Python DSL 的字段 |
|-----------|---------|--------------------|-----------------------------------------|
| **Tile** | `TileBufType` | `dtype` + `shape` + `valid_shape` + `memorySpace` + `config`（blayout/slayout/fractal/pad） | — |
| **View** | `MemRefType`（降级后的 `PartitionTensorViewType`） | `dtype` | `shape`、`strides`、`memorySpace`（仅用于约束检查） |
| **Scalar** | 标量类型 | `dtype` | — |

**View 操作数的特化策略**：View 对应的模板参数类型为 `!pto.partition_tensor_view<?x?x...xdtype>`，维度全部动态，shape/strides 通过 intrinsic 在运行时查询。因此不同 view shape 的 Tile op 可以共享同一份模板实例——`shape`/`strides`/`memorySpace` 不参与 SpecKey 的判等和 hash。这些字段通过 `--operand-specs` JSON 传给 Python DSL 的 `expand_helper`，先按操作数位置构造成 `arg0_*`、`arg1_*` 一类的位置化上下文，再在 constraint evaluation 阶段按模板参数顺序映射到当前参数名（如 `src` / `dst`）后参与约束检查；它们不直接影响模板代码生成。

**Tile 操作数的特化策略**：当前实现中，`valid_shape` 参与 SpecKey，并与 `shape`、`memorySpace`、`config` 一起决定模板实例和缓存 key。也就是说，相同 `(op, operand_types)` 但不同 `valid_shape` 的 Tile op 当前会生成不同的实例化结果。约束检查和缓存命名都基于这一实现语义。

#### 3.2.2 模板实例化过程

Expand TileOp 通过调用 Python 子进程来实例化模板。具体流程：

1. **调用 Python helper**：`python3 -m tilelang_dsl.expand_helper --target <arch> --op pto.<op> --operand-specs <JSON>`，其中 JSON 描述每个操作数的类型信息。
2. **Python 端处理**：
   - 扫描模板目录下的 `.py` 文件，查找标注了 `@pto.vkernel` 装饰器的模板函数
   - 先按操作数个数和参数种类（`tile` / `view` / `scalar`）做 schema 预过滤
   - 基于 `operand_specs` 构造按位置组织的上下文属性（如 `arg0_shape`、`arg0_strides`、`arg1_config`）
   - 调用 `pto.select_kernel(target, concrete_op, operand_types, context_attrs, registry)` 按 `target → op → dtypes → constraints → priority` 规则选择模板
   - 对 `pto.Tile` 参数使用给定的 shape / valid_shape / memory_space / config 进行特化
   - 对 `pto.PartitionTensorView` 参数，不做 `specialize()`，而是通过位置化上下文把 shape/strides/memorySpace 提供给前置条件检查（参数类型保持全动态）
   - 输出特化后的 MLIR 文本
3. **C++ 端处理**：
   - 解析 MLIR 文本为 `ModuleOp`
   - 提取 `func.func`，克隆到目标 Module 末尾
   - 重命名为 `__pto_tilelang_<target>_<op>_tile_<dtype>_<dim0>_<dim1>_view_<dtype>_...`（Tile 操作数拼 shape/valid_shape/config，View/Scalar 只拼 dtype），设为 `private` 可见性
   - 按 `target + op + operand schema` 存入 specCache

**关键约束**：Python DSL 实例化输出的函数需要满足以下要求：

1. **参数类型**可以是 `!pto.tile_buf`、`!pto.partition_tensor_view` 或标量类型。DSL 在实例化时将 Tile 参数的元素类型、静态 shape、布局配置等信息编码进 `tile_buf` 类型；View 参数保持全动态维度（`!pto.partition_tensor_view<?x?x...xdtype>`）。
2. **函数必须带有 `pto.tilelang.instance` 属性**（UnitAttr）。Inline pass 通过此属性识别需要内联的模板实例函数。

函数体内部通过以下 intrinsic 提取信息：

**tile_buf 系列**（从 `!pto.tile_buf` 提取）：

| Intrinsic | 功能 | 输出类型 |
|-----------|------|----------|
| `pto.tile_buf_addr` | 提取数据区域的 memref 指针 | `memref<RxCxdtype, strided<...>, #pto.address_space<...>>` |
| `pto.tile_valid_rows` | 提取有效行数 | `index` |
| `pto.tile_valid_cols` | 提取有效列数 | `index` |

**tensor_view 系列**（从 `!pto.partition_tensor_view` 提取）：

| Intrinsic | 功能 | 输出类型 |
|-----------|------|----------|
| `pto.tensor_view_addr` | 提取 memref/ptr 基地址 | `memref<...>` 或 `!pto.ptr<...>` |
| `pto.get_tensor_view_dim` | 按维度索引提取 shape 大小 | `index` |
| `pto.get_tensor_view_stride` | 按维度索引提取 stride | `index` |

对于 Tile 操作数，Expand TileOp 直接将 `tile_buf` 透传。对于 View 操作数，调用方类型为 `memref`，模板参数类型为 `!pto.partition_tensor_view`，因此 Expand TileOp 在调用点插入 `builtin.unrealized_conversion_cast` 桥接。类型转换和 intrinsic 折叠统一在后续的 Fold pass 中处理。

### 3.3 实例化模板函数的 IR 结构

TileLang DSL 实例化后，生成的 MLIR 函数结构如下（以 `pto.tadd`、`dtype=f32`、`shape=(16,64)` 为例）：

```mlir
func.func @__pto_tilelang_tadd_f32_16_64(
    %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=?, v_col=?,
                         blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=?, v_col=?,
                         blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %dst:  !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=?, v_col=?,
                         blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    attributes { pto.tilelang.instance }
  {

  // 1. 从 tile_buf 提取 memref 地址
  %mSrc0 = pto.tile_buf_addr %src0 : ... -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
  %mSrc1 = pto.tile_buf_addr %src1 : ... -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
  %mDst  = pto.tile_buf_addr %dst  : ... -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>

  // 2. 从 tile_buf 提取有效形状（inline 后由 Fold pass 折叠为常量或绑定到实际动态值）
  %v_rows = pto.tile_valid_rows %src0 : ... -> index
  %v_cols = pto.tile_valid_cols %src0 : ... -> index
  %v_cols_i32 = arith.index_cast %v_cols : index to i32  // plt_b32 需要 i32

  // 3. dtype=f32 → vector_width=64（256B / 4B），这是在模板实例化时固化的常量
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index

  // 4. 向量循环体：按行遍历，按 vreg 宽度分块，带尾部 mask
  pto.vecscope {
    scf.for %i = %c0 to %v_rows step %c1 {
      scf.for %j = %c0 to %v_cols step %c64 iter_args(%remain = %v_cols_i32) -> (i32) {
        %mask, %next = pto.plt_b32 %remain : i32 -> !pto.mask, i32

        %va = pto.vlds %mSrc0[%i, %j]
            : memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>> -> !pto.vreg<64xf32>
        %vb = pto.vlds %mSrc1[%i, %j]
            : memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>> -> !pto.vreg<64xf32>
        %vc = pto.vadd %va, %vb, %mask
            : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
        pto.vsts %vc, %mDst[%i, %j], %mask
            : !pto.vreg<64xf32>, memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>, !pto.mask

        scf.yield %next : i32
      }
    }
  }
  return
}
```

当 `valid_shape` 为静态已知值时（如 `v_row=16, v_col=64`），`pto.tile_valid_rows`/`pto.tile_valid_cols` 在 Fold pass 中会被直接折叠为常量 `arith.constant 16 : index`。当 `valid_shape` 为动态值时（`v_row=?, v_col=?`），Fold pass 将其替换为调用点传入的实际动态 `index` 值。

### 3.4 三个 Pass 的输入/输出示例

以下展示一个完整的 `pto.tadd` 从 TileOp 到向量 IR 的变换过程。

#### 3.4.1 输入（TileOp）

```mlir
func.func @TADD(
    %a: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %b: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %c: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>) {
  pto.tadd ins(%a, %b : ...) outs(%c : ...)
  return
}
```

#### 3.4.2 经过 Expand TileOp 后

`pto.tadd` 被替换为 `func.call`，操作数直接传递（类型不变）：

```mlir
func.func @TADD(%a: !pto.tile_buf<...>, %b: !pto.tile_buf<...>, %c: !pto.tile_buf<...>) {
  // pto.tadd 被替换为函数调用，tile_buf 直接透传
  call @__pto_tilelang_tadd_f32_16_64(%a, %b, %c) : (...) -> ()
  return
}

// TileLang DSL 实例化的模板函数（参数为 tile_buf 类型，带 pto.tilelang.instance 属性）
func.func private @__pto_tilelang_tadd_f32_16_64(
    %src0: !pto.tile_buf<...>, %src1: !pto.tile_buf<...>, %dst: !pto.tile_buf<...>)
    attributes { pto.tilelang.instance } {
  %mSrc0 = pto.tile_buf_addr %src0 : ... -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
  %mSrc1 = pto.tile_buf_addr %src1 : ...
  %mDst  = pto.tile_buf_addr %dst  : ...
  %v_rows = pto.tile_valid_rows %src0 : ... -> index
  %v_cols = pto.tile_valid_cols %src0 : ... -> index
  %v_cols_i32 = arith.index_cast %v_cols : index to i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  pto.vecscope {
    scf.for %i = %c0 to %v_rows step %c1 {
      scf.for %j = %c0 to %v_cols step %c64 iter_args(%remain = %v_cols_i32) -> (i32) {
        %mask, %next = pto.plt_b32 %remain : i32 -> !pto.mask, i32
        %va = pto.vlds %mSrc0[%i, %j] : memref<...> -> !pto.vreg<64xf32>
        %vb = pto.vlds %mSrc1[%i, %j] : memref<...> -> !pto.vreg<64xf32>
        %vc = pto.vadd %va, %vb, %mask : ...
        pto.vsts %vc, %mDst[%i, %j], %mask : ...
        scf.yield %next : i32
      }
    }
  }
  return
}
```

#### 3.4.3 经过 Inline 后

模板函数体被 inline 到 `@TADD` 函数中，形参 `%src0`/`%src1`/`%dst` 与实参 `%a`/`%b`/`%c` 绑定：

```mlir
func.func @TADD(%a: !pto.tile_buf<...>, %b: !pto.tile_buf<...>, %c: !pto.tile_buf<...>) {
  // inline 后，tile_buf_addr / tile_valid_rows / tile_valid_cols 的操作数
  // 绑定到调用点的实际 tile_buf 值
  %mA = pto.tile_buf_addr %a : ... -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
  %mB = pto.tile_buf_addr %b : ...
  %mC = pto.tile_buf_addr %c : ...
  %v_rows = pto.tile_valid_rows %a : ... -> index
  %v_cols = pto.tile_valid_cols %a : ... -> index
  %v_cols_i32 = arith.index_cast %v_cols : index to i32
  ...
  pto.vecscope {
    scf.for %i = %c0 to %v_rows step %c1 { ... }
  }
  return
}
```

#### 3.4.4 经过 Fold TileBuf Intrinsics 后

Fold pass 处理两族 intrinsic，通过严格的模式匹配将它们解析回调用点的具体 SSA 值。

##### tile_buf 系列折叠

每一个被折叠的 tile_buf intrinsic，其 `tile_buf` 操作数必须能解析到调用点
的 materialized tile handle，否则 pass 直接报错并失败：

```mlir
%0 = pto.pointer_cast(%addr) {config = ...}
       : memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
%1 = pto.bind_tile %0, %v_row, %v_col {config = ...}
       : memref<16x64xf32, strided<[64, 1]>, ...>
      -> memref<16x64xf32, strided<[64, 1], offset: ?>, ...>
%2 = builtin.unrealized_conversion_cast %1
       : memref<...> to !pto.tile_buf<vec, 16x64xf32>
```

也即：`tile_buf ← unrealized_conversion_cast ← pto.bind_tile ← pto.pointer_cast`。

**三条折叠规则**（均锚定到 `pto.bind_tile`）：

- `pto.tile_buf_addr %a` → 折叠为 `bind_tile` 的 **第一个操作数**（即 `pto.pointer_cast` 的结果）。
  注意这里**绕过**了 `bind_tile` 自身产出的、带 `offset: ?` 的动态布局 memref，
  直接复用上游的 `strided<[64, 1]>` 静态布局 memref。这样下游的 `pto.vlds`/`pto.vsts`
  在被规范化、最终下沉到 VPTO 后端时，看到的始终是干净的 `strided<[..], offset: 0>` 布局，
  避免了 `pto.vlds does not support dynamic memref layout offsets` 这类下游错误。
  若 `tile_buf_addr` 声明的结果类型与 `bind_tile` 源 memref 的实际布局不一致，
  会就地把结果类型替换为源 memref 的真实类型——下游向量算子对相同 element type / shape
  的 strided 布局是多态的。
- `pto.tile_valid_rows %a` → 优先按 `TileBufType.validShape[0]` 静态折叠：
  若是静态值（如 `v_row=16`），折叠为 `arith.constant 16 : index`；
  若是动态值（`v_row=?`），折叠为 `bind_tile` 的 **第二个操作数**（`valid_row`，已经是 `index` 类型）。
- `pto.tile_valid_cols %a` → 同理，使用 `validShape[1]` 或 `bind_tile` 的 **第三个操作数**。

##### tensor_view 系列折叠

每一个被折叠的 tensor_view intrinsic，其 `partition_tensor_view` 操作数必须由如下固定链定义
（由 `ExpandTileOp` 和 `PTOViewToMemref` pass 保证），否则 pass 直接报错并失败：

```mlir
%rc = memref.reinterpret_cast %arg0
    to offset: [0], sizes: [%c1, %c1, %c1, %c16, %c64],
       strides: [%c1024, %c1024, %c1024, %c64, %c1]
  : memref<?xf32, gm> → memref<?x?x?x?x?xf32, strided<[?,?,?,?,?], offset:?>, gm>

%sv = memref.subview %rc [0,0,0,0,0] [1,1,1,16,64] [1,1,1,1,1]
  : → memref<1x1x1x16x64xf32, strided<[?,?,?,?,?], offset:?>, gm>

%tv = builtin.unrealized_conversion_cast %sv
  : memref<...> → !pto.partition_tensor_view<...>
```

也即：`partition_tensor_view ← unrealized_conversion_cast ← memref.subview ← memref.reinterpret_cast`。

pass 贯穿整条链，**一步到位**折叠到最终结果，不生成中间的 `memref.dim`、`memref.extract_strided_metadata` 或 `pto.castptr %subview`：

- `pto.get_tensor_view_dim %tv, %cN` →
  - subview 结果类型 shape[N] 是静态的：折叠为 `arith.constant`（如 dim 3 → `arith.constant 16`）
  - shape[N] 是动态的：取 subview 的 `getMixedSizes()[N]`（可能追溯到 reinterpret_cast 的 size operand）

- `pto.get_tensor_view_stride %tv, %cN` →
  直接取 reinterpret_cast 的 stride operand（通过 `getMixedStrides()[N]`）。
  若 subview 的 stride[N] 不为 1，则生成 `arith.muli(rc_stride, sv_stride)`。
  reinterpret_cast 的 stride 可以是静态属性（生成 `arith.constant`）或动态 SSA 值（直接复用）。

- `pto.tensor_view_addr %tv` →
  - subview 和 reinterpret_cast 的 offset 均为 0：折叠为 `pto.castptr %arg0`（直接用 base memref）
  - 有非零 offset：折叠为 `pto.addptr(pto.castptr %arg0, linear_offset)`，
    其中 `linear_offset = rc_offset + sum(sv_offset[i] * rc_stride[i])`

##### 通用规则

**跳过 TileLang 模板实例**：被 `PTOInlineLibCall` 内联完且作为 dead callee 删除之前，
带 `pto.tilelang.instance` 属性的私有模板函数仍可能保留在 module 中。这些函数体内的
`pto.tile_buf_addr` 等 intrinsic 直接作用在 `tile_buf` 类型的 BlockArgument 上，
没有 `bind_tile` 可供折叠——pass 通过检测 `pto.tilelang.instance` 属性跳过这些函数，
留给下游 DCE 清理。

折叠后得到最终的纯向量 IR，不再包含任何 tile_buf 引用：

```mlir
func.func @TADD(
    %a: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %b: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %c: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>) {

  %vecA = pto.tile_buf_addr %a : ... -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
  %vecB = pto.tile_buf_addr %b : ... -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
  %vecC = pto.tile_buf_addr %c : ... -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index    // ← tile_valid_rows 折叠为常量
  %c64 = arith.constant 64 : index    // ← tile_valid_cols 折叠为常量
  %c64_i32 = arith.constant 64 : i32

  pto.vecscope {
    scf.for %i = %c0 to %c16 step %c1 {
      scf.for %j = %c0 to %c64 step %c64 iter_args(%remain = %c64_i32) -> (i32) {
        %mask, %next = pto.plt_b32 %remain : i32 -> !pto.mask, i32
        %va = pto.vlds %vecA[%i, %j]
            : memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>> -> !pto.vreg<64xf32>
        %vb = pto.vlds %vecB[%i, %j]
            : memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>> -> !pto.vreg<64xf32>
        %vc = pto.vadd %va, %vb, %mask
            : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
        pto.vsts %vc, %vecC[%i, %j], %mask
            : !pto.vreg<64xf32>, memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>, !pto.mask
        scf.yield %next : i32
      }
    }
  }
  return
}
```

这与 1.2 节中手写的 Vector IR 实现等价，但由 Python DSL 模板自动生成。

### 3.5 模板目录与部署

TileLang DSL 模板文件（`.py`）部署在 PTOAS 工程的固定位置：

```
lib/TileOp/                     ← 模板库根目录
├── tadd_template.py            ← pto.tadd 的模板
├── tsub_template.py            ← pto.tsub 的模板
├── tmul_template.py            ← pto.tmul 的模板
└── ...
```

`tilelang_dsl` Python 包在安装后位于固定的 Python 包路径下。Expand TileOp pass 无需额外的 CLI 选项指定路径——模板目录和包路径在编译器构建时确定。

### 3.6 添加新算子的模板

在模板目录下创建 `.py` 文件，使用 `@pto.vkernel` 装饰器定义模板：

```python
@pto.vkernel(
    op="pto.<op_name>",           # 匹配的 Tile 算子名
    dtypes=[(<dtype>, ...)],      # 支持的 dtype 签名
    advanced=True,                # 启用隐式 vecscope 推断
    name="template_<op_name>",
)
def template_xxx(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    # 向量化实现体
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            # ... 向量操作 ...
    return None
```

**关键约束：模板参数顺序必须与 PTOAS 中对应指令的操作数顺序严格一致。**
`ExpandTileOp` 按位置索引将指令操作数直接传递给模板函数参数。对于 DPS 风格的
算子，这意味着 `ins` 操作数在前、`outs` 在后。例如 `pto.tadd ins(%a, %b) outs(%c)`
的操作数顺序为 `(src0, src1, dst)`，模板参数必须为 `(src0, src1, dst)`。

`expand_helper.py` 自动扫描目录下所有 `.py` 文件，先按参数 schema 过滤候选，再通过 `select_kernel()` 按 `target`、`op`、`dtype`、`constraints` 和 `priority` 选择模板。模板约束读取的位置化上下文由 `argN_*` 键提供，并在 constraint evaluation 阶段按参数顺序映射到模板自己的参数名。


## 第四章 前置工作

### 4.1 Python DSL 扩展

| 工作项 | 说明 |
|--------|------|
| `@pto.tile_template` 装饰器 | 标记模板函数，指定对应的 Tile op 和 target |
| `pto.Tile` 属性接口 | 支持 `shape`、`valid_shape`、`element_type`、`element_size` 等属性访问 |
| `Tile` 下标访问 | 支持 `tile[i, j]` 语法用于 `vlds`/`vsts` 的地址计算 |
| 动态循环边界 | 当 `valid_shape` 为运行时动态值时，`range` 生成 `scf.for` |

### 4.2 PTOAS 编译器：Expand TileOp Pass

| 工作项 | 说明 |
|--------|------|
| 模板查找机制 | 根据 Tile op 种类和 dtype 匹配 Python DSL 模板 |
| 模板实例化 | 调用 Python DSL，传入具体 `tile_buf` 类型，获取实例化后的 MLIR |
| MLIR 解析与 inline | 解析生成的 MLIR 文本，inline 到调用点，绑定参数 |
| Cleanup | 实例化后运行 canonicalize 清理冗余 |

### 4.3 PTOAS 编译器：Fold TileBuf Intrinsics Pass

**tile_buf 系列**：

| 工作项 | 说明 |
|--------|------|
| 严格模式匹配 | 要求 `tile_buf` 由 `unrealized_conversion_cast ← pto.bind_tile` 链定义，否则 emit error 并 fail pass |
| `tile_buf_addr` 折叠 | 替换为 `bind_tile.getSource()`（即 `pto.pointer_cast` 的静态布局 memref），绕过 `bind_tile` 产出的动态 offset 布局 |
| 结果类型自适应 | 若 `tile_buf_addr` 声明类型与 source memref 实际布局不一致，就地更新结果类型 |
| `tile_valid_rows/cols` 折叠 | 优先按 `TileBufType.validShape` 静态折叠为 `arith.constant`；动态时取 `bind_tile` 的 `valid_row`/`valid_col` 操作数 |

**tensor_view 系列**：

| 工作项 | 说明 |
|--------|------|
| 严格模式匹配 | 要求 `partition_tensor_view` 由 `unrealized_conversion_cast ← memref.subview ← memref.reinterpret_cast` 链定义，否则 emit error 并 fail pass |
| `tensor_view_addr` 折叠 | 贯穿 subview → reinterpret_cast 链，折叠为 `pto.castptr %base_memref`；有非零 offset 时生成 `pto.addptr` |
| `get_tensor_view_dim` 折叠 | 静态 shape 维度折叠为 `arith.constant`；动态维度取 subview 的 `getMixedSizes()` operand |
| `get_tensor_view_stride` 折叠 | 直接取 reinterpret_cast 的 stride operand（`getMixedStrides()`），乘以 subview stride（通常为 1 可短路） |
| Dead op 清理 | 折叠完成后清理无 user 的 `unrealized_conversion_cast`、`memref.subview`、`memref.reinterpret_cast` |

**通用**：

| 工作项 | 说明 |
|--------|------|
| 跳过模板实例 | 检测 `pto.tilelang.instance` 属性，跳过 `PTOInlineLibCall` 删除前残留的私有模板函数 |

### 4.4 测试与文档

- Python DSL 模板编写和实例化的单元测试
  以当前 `lib/TileOps/tadd_template.py` 为例，新增/维护
  `test/lit/vpto/expand_tile_op_tilelang.pto`
  作为 `pto.tadd` TileLang 模板实例化的基础回归。该用例覆盖：
  1. `ExpandTileOp` 是否能匹配 `pto.tadd` 并调用 Python DSL helper；
  2. 模板实例化后的 `func.call` 是否能被 inline；
  3. `FoldTileBufIntrinsics` 之后是否得到 `pto.vlds` / `pto.vadd` / `pto.vsts` 形式的 Vector IR。

  当前 `pto.tadd` 的向量库模板实现如下：

  ```python
  import sys
  from pathlib import Path
  import tilelang_dsl as pto


  @pto.vkernel(
      target="a5",
      op="pto.tadd"
  )
  def template_tadd(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
      dtype = dst.element_type
      valid_rows, valid_cols = dst.valid_shape

      for row in range(0, valid_rows, 1):
          remained = valid_cols
          for col in range(0, valid_cols, pto.get_lanes(dtype)):
              mask, remained = pto.make_mask(dtype, remained)
              lhs = pto.vlds(src0[row, col:])
              rhs = pto.vlds(src1[row, col:])
              summed = pto.vadd(lhs, rhs, mask)
              pto.vsts(summed, dst[row, col:], mask)
      return
  ```

  对应的单元测试用例如下：

  ```mlir
  // Test that ExpandTileOp + InlineLibCall + FoldTileBufIntrinsics pipeline
  // expands pto.tadd via the default TileLang Python DSL template
  // lib/TileOps/tadd_template.py.
  //
  // Pipeline: PTOMaterializeTileHandles -> ExpandTileOp -> InlineLibCall -> FoldTileBufIntrinsics
  //
  // RUN: ptoas --pto-arch=a5 --pto-backend=vpto --enable-tile-op-expand %s -o - 2>/dev/null | FileCheck %s

  // After the full tile-op-expand path on the VPTO backend, the original
  // pto.tadd should be lowered to vector-style VPTO IR.
  // CHECK: func.func @TADD
  // CHECK-NOT: pto.tadd ins
  // CHECK: pto.vecscope
  // CHECK: pto.castptr
  // CHECK: pto.vlds
  // CHECK: pto.vadd
  // CHECK: pto.vsts

  module {
    func.func @TADD() {
      %a = pto.alloc_tile
        : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                        blayout=row_major, slayout=none_box, fractal=512, pad=0>
      %b = pto.alloc_tile
        : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                        blayout=row_major, slayout=none_box, fractal=512, pad=0>
      %tile_buf = pto.alloc_tile
        : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                        blayout=row_major, slayout=none_box, fractal=512, pad=0>

      pto.tadd ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                                blayout=row_major, slayout=none_box, fractal=512, pad=0>,
                            !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                                blayout=row_major, slayout=none_box, fractal=512, pad=0>)
               outs(%tile_buf : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                                     blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      return
    }
  }
  ```
- Expand TileOp pass 的端到端测试（`pto.tadd` → VPTO fatobj）
  使用以下命令生成最终 fatobj，并由 `ptoas` 内部完成 VPTO lowering、device 编译、stub 生成和打包：

  ```bash
  ./build/tools/ptoas/ptoas test/lit/vpto/expand_tile_op_tilelang.pto \
    --pto-arch=a5 \
    --pto-backend=vpto \
    --enable-tile-op-expand \
    -o add.o
  ```

  说明：
  - `add.o` 是 host 可链接的 fatobj 对象。
  - 若上述命令成功生成 `add.o`，则说明当前 `pto.tadd` 的向量库模板已经完成：
  - TileLang 模板实例化；
  - `pto.tadd -> VPTO -> LLVM` 的端到端 lowering；
  - device 编译、stub 生成和 fatobj 打包。
- 融合场景测试（多个 Tile op 连续使用后的 VF Fusion）
- 更新 `PTO_IR_manual.md` 和 TileLang DSL Guide

#### 4.4.1 ST 精度验证

IR 回归测试只能验证"模板展开后 IR 长什么样"，无法回答"最终在 simulator / NPU 上跑出来的数值是否正确"。
`test/tilelang_st` 框架提供了端到端精度验证能力，详细设计参见 [`tilelang-st-framework.md`](tilelang-st-framework.md)。

本节面向库开发者，说明在完成一个新 TileLang 库实现（如 `lib/TileOps/<op>_template.py`）后，如何接入 ST 框架验证精度。

##### 完整执行链路概览

ST 框架的统一入口是：

```bash
python3 test/tilelang_st/script/run_st.py -r sim -v a5 -t tadd
```

它不是只做“编译 `.pto`”，而是把编译、生成输入、运行二进制和精度比较串成一条完整流水线：

```text
run_st.py
  ├─ set_env_variables()
  │   └─ 配置 simulator / NPU 运行环境
  ├─ build_project()
  │   ├─ cmake -DRUN_MODE=... -DSOC_VERSION=... -DTEST_CASE=... -DPTOAS_BIN=...
  │   ├─ ptoas: <op>.pto -> <op>_kernel.o
  │   │    flags:
  │   │      --pto-arch=a5
  │   │      --pto-backend=vpto
  │   │      --enable-insert-sync
  │   │      --enable-tile-op-expand
  │   ├─ bisheng -xcce: launch.cpp + <op>_kernel.o -> lib<op>_kernel.so
  │   └─ bisheng -xc++: main.cpp -> <op>
  ├─ run_gen_data()
  │   └─ 在 build/testcase/<testcase>/ 下生成每个 case 的 input/golden
  ├─ run_binary()
  │   └─ 在 build/testcase/<testcase>/ 下执行 ../../bin/<testcase> [case]
  └─ run_compare()
      └─ 在 build/testcase/<testcase>/ 下逐 case 比较 golden/output
```

其中编译子链可以单独理解为：

```text
<op>.pto
  ──ptoas──> <op>_kernel.o                     (host-linkable fatobj)
  ──bisheng -xcce launch.cpp + <op>_kernel.o──> lib<op>_kernel.so
                                                (共享库)
  ──bisheng -xc++ main.cpp + .so──> <op>       (host 可执行文件)
```

`ptoas` 直接产出 host 可链接的 fatobj。TileLang ST 不再维护 `kernel.ll -> device.o -> repack`
这条旧中间链路，`launch.cpp` 只负责 host 侧 kernel 声明和 wrapper，最终由 `bisheng -xcce`
把 `launch.cpp` 与 fatobj 链接成共享库。

运行阶段同样是 ST 框架的一部分，而不是“编译完以后开发者手工处理”的额外步骤：

- `gen_data.py` 会基于 `cases.py` 中的 `CASES` 为每个 case 生成 `input*.bin` 和 `golden.bin`
- host 可执行文件会按 `main.cpp` 中的 case table 逐个读取 `./<case_name>/input*.bin`，运行 kernel，并写回 `./<case_name>/output.bin`
- `compare.py` 再基于同一份 `CASES` 定义逐 case 读取并裁剪需要比较的数据，最后调用公共 `result_cmp()`
- 若传入 `-c <case_name>`，则运行和比较都只针对单个 case

因此，TileLang ST 的验证对象不是“某一份中间 IR 是否长得对”，而是：

1. TileLang 模板是否成功展开并编译到可执行产物；
2. 生成的数据、运行时读取的 case 目录、以及 compare 使用的 golden/output 是否保持一致；
3. 最终 simulator / NPU 上的数值结果是否正确。

编译子链由 `testcase/CMakeLists.txt` 中的 `pto_tilelang_vec_st()` 宏自动接管，整条执行链路则由
`run_st.py` 统一调度。

##### 新增 testcase 所需文件（七个文件 + 一个注册修改）

以新增 `pto.tsub` 为例，需在 `test/tilelang_st/npu/a5/src/st/testcase/tsub/` 下准备
7 个文件，并修改 1 个注册文件：

**1. `CMakeLists.txt`** — 通常只有一行：

```cmake
pto_tilelang_vec_st(tsub)
```

宏自动查找同目录下的 `tsub.pto`、`launch.cpp`、`main.cpp`，串联上述五步编译。

**2. `cases.py`** — **case 定义的单一来源**，`gen_data.py` 和 `compare.py` 均从此导入：

```python
import numpy as np

CASES = [
    {
        "name": "f32_16x64",
        "dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-6,
    },
]
```

常规 case 必须包含 `name`/`dtype`/`shape`/`valid_shape`/`eps` 五个字段，`valid_shape` 为必填。
如果输出 shape 与输入不同（如 `trowsum`），再额外补 `dst_shape`/`dst_valid_shape`，供
`compare.py` 和 `gen_data.py` 使用。

**3. `tsub.pto`** — kernel 描述，一个文件中放多个 case 对应的函数。每个函数
代表一种 dtype/shape 组合。以 tadd 为参考，kernel 结构为：

```mlir
module {
  // Case: f32 16x64
  func.func @TSUB_f32_16x64(%a_ptr: !pto.ptr<f32>, %b_ptr: !pto.ptr<f32>,
                              %c_ptr: !pto.ptr<f32>) {
    // 1. make_tensor_view: 从 !pto.ptr 构造 5D tensor_view (1×1×1×rows×cols)
    // 2. partition_view: 提取 tile 区域
    // 3. alloc_tile: 分配 UB 上的 tile_buf
    // 4. tload: 从 GM 加载到 UB
    // 5. pto.tsub: 执行计算
    // 6. tstore: 从 UB 写回 GM
    return
  }
  // Case: f32 32x32
  func.func @TSUB_f32_32x32(...) { ... }
}
```

函数命名约定：`<OP>_<dtype>_<rows>x<cols>`，例如 `TSUB_f32_16x64`、`TSUB_bf16_32x32`。

注意：`.pto` 中 `make_tensor_view` 的 shape 维度是 5D（`1×1×1×rows×cols`），strides 需要
与 shape 一致（最内维 stride=1，逐维累乘）。函数参数顺序决定了后续所有文件的参数顺序。

**4. `launch.cpp`** — 为每个 kernel 声明 entry 和 launch wrapper：

```cpp
#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void TSUB_f32_16x64(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTSUB_f32_16x64(float *a, float *b, float *c, void *stream) {
    TSUB_f32_16x64<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}
```

关键约束：
- `extern "C" __global__ AICORE void ...` 声明需要和 `.pto` 中的 `pto.kernel` 函数签名对应
- kernel 参数类型和顺序必须与 `.pto` 中函数签名一致
- `<<<1, nullptr, stream>>>` 表示单核启动

**5. `main.cpp`** — host driver，核心是 case table 和 `RunCase()` 函数：

```cpp
#include "acl/acl.h"
#include "test_common.h"   // PtoTestCommon::ReadFile / WriteFile + ACL_CHECK

using LaunchFn = void (*)(float *, float *, float *, void *);

struct TestCase {
    const char *name;      // 对应 cases.py 中的 name 和运行时子目录
    LaunchFn    launch;
    size_t      rows;       // allocated tile rows
    size_t      cols;       // allocated tile cols
    size_t      validRows;  // effective computation rows  (<= rows)
    size_t      validCols;  // effective computation cols  (<= cols)
    size_t      elemSize;
};

static const TestCase kCases[] = {
    {"f32_16x64", LaunchTSUB_f32_16x64, 16, 64, 16, 64, sizeof(float)},
    {"f32_32x32", LaunchTSUB_f32_32x32, 32, 32, 32, 32, sizeof(float)},
};
```

注意：`ACL_CHECK` 宏由公共头 `test_common.h` 提供（需在 `acl/acl.h` 之后包含），无需在每个 testcase 中重复定义。

`RunCase()` 的职责：
1. 从 `./<case>/input*.bin` 读取输入到 host 内存
2. `aclrtMemcpy` 拷贝到 device
3. 调用 `tc.launch(...)` 启动 kernel
4. `aclrtSynchronizeStream` 等待完成
5. 拷贝结果回 host
6. 写 `./<case>/output.bin`

`main()` 支持可选 `argv[1]` 作为 case filter，实现单 case 执行。

**6. `gen_data.py`** — 生成每个 case 的输入和 golden，从 `cases.py` 导入 `CASES`：

```python
from cases import CASES
from st_common import validate_cases, setup_case_rng, save_case_data

validate_cases(CASES)

for case in CASES:
    setup_case_rng(case)  # per-case seed，新增 case 不影响已有数据
    dtype, shape = case["dtype"], case["shape"]
    valid_shape = case["valid_shape"]

    input1 = np.random.randint(1, 10, size=shape).astype(dtype)
    input2 = np.random.randint(1, 10, size=shape).astype(dtype)
    golden = np.zeros(shape, dtype=dtype)
    vr, vc = valid_shape
    golden[:vr, :vc] = (input1[:vr, :vc] - input2[:vr, :vc]).astype(dtype, copy=False)  # tsub: 减法

    save_case_data(case["name"], {"input1": input1, "input2": input2, "golden": golden})
```

注意 golden 的计算逻辑必须与 op 语义一致（tadd 是加法，tsub 是减法），且只在 `valid_shape` 区域内计算。

**7. `compare.py`** — 每个 testcase 自己维护比较脚本。公共层只提供
`st_common.result_cmp(golden, output, eps)`，具体比较哪些数据由 testcase 自己决定。

以 `tsub` 这种输入输出 shape 一致的 case 为例，核心逻辑通常是：

```python
from cases import CASES
from st_common import result_cmp, style_fail, style_pass, validate_cases

validate_cases(CASES)

for case in CASES:
    shape = case["shape"]
    vr, vc = case["valid_shape"]
    golden = np.fromfile(os.path.join(case["name"], "golden.bin"), dtype=case["dtype"]).reshape(shape)
    output = np.fromfile(os.path.join(case["name"], "output.bin"), dtype=case["dtype"]).reshape(shape)
    ok = result_cmp(golden[:vr, :vc], output[:vr, :vc], case["eps"])
```

如果是 `trowsum` 这类输出 shape 不同的 op，则 `compare.py` 可以自己按 `dst_shape` reshape，
并只比较 `dst_valid_shape` 对应的有效区域。exit code 2 表示失败。

精度阈值参考：

| dtype | 建议 eps |
|---|---|
| `float32` | `1e-6` |
| `float16` | `1e-3` |
| `bfloat16` | `1e-2` |
| `int8/int16/int32` | `0`（精确匹配） |

**8. 注册** — 修改 `testcase/CMakeLists.txt`，将新 op 加入 `ALL_TESTCASES`：

```cmake
set(ALL_TESTCASES
    tadd
    tsub    # ← 新增
)
```

##### 文件间一致性约束

新增 testcase 时最容易出错的是以下几处必须严格一致：

| 约束 | 涉及文件 | 示例 |
|---|---|---|
| kernel 函数名 | `.pto` ↔ `launch.cpp` | `@TSUB_f32_16x64` ↔ `TSUB_f32_16x64` |
| Launch wrapper 名 | `launch.cpp` ↔ `main.cpp` | `LaunchTSUB_f32_16x64` |
| case 名 | `cases.py` ↔ `main.cpp` kCases[] ↔ 运行时目录 | `f32_16x64` |
| 参数顺序 | `.pto` → `launch.cpp` → `main.cpp` 的 launch 调用 | `(a, b) → c` |
| shape / valid_shape | `cases.py` ↔ `.pto` tile shape ↔ `main.cpp` rows/cols/validRows/validCols | `16×64` / `(16, 64)` |

Python 侧的 case 名、dtype、shape、valid_shape、eps（以及必要时的 `dst_shape` /
`dst_valid_shape`）已通过 `cases.py` 收敛为单一来源。但 C++ 侧 `main.cpp` 的 `kCases[]`
和 `.pto` 仍需手动与 `cases.py` 保持一致。

任何一处不一致都可能导致：编译成功但运行时 segfault，或运行成功但比较结果错误且难以定位。

##### 运行方式

统一入口为 `test/tilelang_st/script/run_st.py`。前置条件：
- `ptoas` 已通过 `./quick_install.sh` 或 editable install 安装到当前
  Python 环境（也可通过 `-p` 指定或 `PTOAS_BIN` 环境变量覆盖）
- `ASCEND_HOME_PATH` 已设置
- simulator / NPU 运行前已加载 CANN `setenv.bash`

```bash
# simulator 上跑 tsub 全部 case
python3 test/tilelang_st/script/run_st.py -r sim -v a5 -t tsub

# NPU 上跑 tsub 全部 case
python3 test/tilelang_st/script/run_st.py -r npu -v a5 -t tsub

# 只跑单个 case
python3 test/tilelang_st/script/run_st.py -r sim -v a5 -t tsub -c f32_16x64

# 复用已有 build，跳过重新编译（只重新生成数据、执行、比较）
python3 test/tilelang_st/script/run_st.py -r sim -v a5 -t tsub -c f32_16x64 -w
```

`run_st.py` 执行顺序：`set_env_variables()` → `build_project()` → `run_gen_data()` →
`run_binary()` → `run_compare()`。产物输出到
`test/tilelang_st/npu/a5/src/st/build/testcase/<testcase>/` 下：

```text
build/testcase/tsub/
├── st_common.py     # 从 testcase/ 公共目录拷贝
├── cases.py         # 从 testcase/tsub/ 拷贝
├── gen_data.py      # 从 testcase/tsub/ 拷贝
├── compare.py       # 从 testcase/tsub/ 拷贝
├── f32_16x64/
│   ├── input1.bin
│   ├── input2.bin
│   ├── golden.bin
│   └── output.bin
└── f32_32x32/
    └── ...
```

##### 建议的开发验证节奏

1. **最小 case 先行**：先写一个最小 case（如 `f32_16x64`），在 simulator 上跑通：
   ```bash
   python3 test/tilelang_st/script/run_st.py -r sim -v a5 -t tsub -c f32_16x64
   ```

2. **快速迭代**：修改 `.pto` 或 host 代码后，用 `-w` 跳过 cmake/make 重编译。
   注意：如果改了 `.pto` 本身，仍需重新编译（不加 `-w`），`-w` 只适合改 `gen_data.py` /
   `compare.py` / `main.cpp` 中非编译相关逻辑的情况。

3. **扩充 case**：单 case 稳定后，补充更多 shape / dtype 组合。建议覆盖：
   - 不同 dtype（f32 / f16 / bf16）
   - 不同 tile 形状（正方形、长条形）
   - 边界情况（valid 行列不是整 tile 的场景）

4. **全量验证**：跑全量 case 确认无回归。

5. **NPU 验证**：切到 `-r npu` 在真实硬件上验证。simulator 和 NPU 的行为可能存在差异。

##### 调试建议

| 阶段 | 排查方向 |
|---|---|
| `ptoas` 编译失败 | 检查 `.pto` 语法、TileLang 模板是否匹配、是否缺少 `--enable-tile-op-expand` |
| fatobj 生成失败 | 检查 `ptoas` stderr、`.pto` 语义和 `pto.kernel` 函数签名 |
| 链接失败 | 检查共享库符号名一致性、ACL 运行时依赖 |
| kernel 执行失败 | 确认 `build/testcase/<op>/<case>/input*.bin` 是否已生成 |
| compare fail | 先检查 `output.bin` vs `golden.bin` 差异，再检查 `.pto` 语义和参数顺序 |

##### 已有 testcase 下新增 case

如果只是在已有 testcase（如 `tadd`）下新增一个 case（如 `f32_8x128`），只需同步修改 4 个文件：

| 文件 | 修改内容 |
|---|---|
| `cases.py` | 在 `CASES` 中加入 `{"name": "f32_8x128", "dtype": np.float32, "shape": (8, 128), "valid_shape": (8, 128), "eps": 1e-6}` |
| `tadd.pto` | 新增 `func.func @TADD_f32_8x128(...)` 函数体 |
| `launch.cpp` | 新增 `extern "C"` kernel 声明和 `LaunchTADD_f32_8x128` wrapper |
| `main.cpp` | 在 `kCases[]` 中加入 `{"f32_8x128", LaunchTADD_f32_8x128, 8, 128, 8, 128, sizeof(float)}` |

`gen_data.py` 和 `compare.py` 无需修改，自动从 `cases.py` 读取。
