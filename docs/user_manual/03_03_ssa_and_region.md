# 3.3 SSA 值与 Region

## 1. 范围

本页描述 PTO 程序中的 SSA 绑定、block 参数、region 和控制流组合方式，不重复模块、函数和类型本体定义。

## 2. SSA 值

### 概述

PTO 程序使用标准 MLIR SSA 语法连接常量、view、tile 和各类 PTO 操作。

### 语法

```mlir
%c0 = arith.constant 0 : index
%tile = pto.alloc_tile
  : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16,
                  blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

### 规则

- 每个 SSA 值只定义一次
- 后续使用通过 `%name` 引用
- SSA 值的类型由定义它的操作结果确定

## 3. Block 与 Block 参数

### 概述

block 是 region 内的顺序执行单元；block 参数用于表达控制流边界上的值传递。

### 语法

```mlir
^bb0(%arg0: i1):
  cf.br ^bb1(%arg0 : i1)

^bb1(%flag: i1):
  return
```

### 说明

- block 参数本质上是该 block 的 SSA 入口值
- 分支操作传入的实参与目标 block 参数一一对应
- 大多数 PTO 样例只有单 block 函数体，但控制流增强后可出现多 block

## 4. Region

### 概述

region 是一组 block 的容器。函数体、`scf.for`、`scf.if` 等都通过 region 承载嵌套语义。

### 示例

```mlir
%final_alive = scf.for %i = %c0 to %c4 step %c1
    iter_args(%alive = %true) -> (i1) {
  %next_alive = scf.if %alive -> (i1) {
    scf.yield %false : i1
  } else {
    scf.yield %true : i1
  }
  scf.yield %next_alive : i1
}
```

### 说明

- PTO 操作可以直接出现在 `scf` 或 `cf` 的 region 内
- region 终结操作必须满足其宿主操作要求
- `iter_args` 和 `yield` 共同定义循环携带值的 SSA 形态

## 5. Constraints

- SSA 值在使用前必须先定义
- 分支实参与 block 参数的个数和类型必须匹配
- 每个 region 内必须以合法 terminator 结束
- 控制流结构之外，PTO 操作的副作用顺序仍按程序顺序与依赖关系解释

## 6. Example

```mlir
func.func @loop_kernel() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %true = arith.constant true

  %flag = scf.for %i = %c0 to %c4 step %c1
      iter_args(%alive = %true) -> (i1) {
    scf.yield %alive : i1
  }
  return
}
```
