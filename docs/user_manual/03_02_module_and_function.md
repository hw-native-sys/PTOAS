# 3.2 模块与函数语法

## 1. 范围

本页只描述 PTO 程序中 `module` 与 `func.func` 的文本形式、组成部分和约束。

## 2. `module`

### 概述

`module` 是 PTO 程序的顶层容器，用于承载符号、函数和顶层属性。

### 语法

```mlir
module {
  ...
}
```

带模块属性的常见形式：

```mlir
module attributes {
  ...
}
```

### 说明

- `module` 内部是一个 region
- PTO 程序通常至少包含一个 `func.func`
- 符号可见性、符号名唯一性和嵌套规则沿用 MLIR 机制

### 目标选择

目标相关行为以 `ptoas` 命令行参数 `--pto-arch` 为准，例如：

- `--pto-arch=a3`
- `--pto-arch=a5`

用户在编写 `module` 时，不需要再额外通过模块属性重复声明目标设备信息。

## 3. `func.func`

### 概述

PTO ISA 不重新定义函数语法，而是直接复用标准 `func.func`。

### 语法

```mlir
func.func @kernel() {
  return
}
```

```mlir
func.func @kernel(%src: !pto.ptr<f16>, %dst: !pto.ptr<f16>, %n: i32) {
  return
}
```

```mlir
func.func @get_i32() -> i32 {
  %c0 = arith.constant 0 : i32
  return %c0 : i32
}
```

### 参数与返回值

- 参数使用 `%name: type` 形式声明
- 返回值使用 `-> type` 或 `-> (type0, type1, ...)` 形式声明
- PTO 自定义类型和 MLIR 内建类型可以混用
- PTO kernel 更常见的是通过指针、view 或 tile 相关对象完成结果写回，而不是返回大对象

## 4. 常见组织方式

### 4.1 仅使用指针参数

```mlir
func.func @copy(%src: !pto.ptr<f16>, %dst: !pto.ptr<f16>) {
  return
}
```

### 4.2 混合指针与标量控制参数

```mlir
func.func @kernel(%src: !pto.ptr<f16>, %dst: !pto.ptr<f16>, %m: i32, %n: i32) {
  return
}
```

### 4.3 带函数属性

```mlir
func.func @kernel(%src: !pto.ptr<f16>) attributes {sym_visibility = "public"} {
  return
}
```

## 5. Constraints

- 函数名在所在 `module` 的符号表内必须唯一
- 入口 block 的参数列表必须与函数签名一致
- `return` 的返回值个数和类型必须与函数声明匹配
- 函数体中的 PTO 操作仍需满足各自的语义约束

## 6. Example

```mlir
module {
  func.func @kernel(%src: !pto.ptr<f16>, %dst: !pto.ptr<f16>) {
    return
  }
}
```

```mlir
module {
  func.func @kernel(%src: !pto.ptr<f16>, %dst: !pto.ptr<f16>) {
    return
  }
}
```
