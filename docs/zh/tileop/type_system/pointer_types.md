# 指针类型

## 概述

`!pto.ptr<T, space>` 表示指定地址空间中指向元素 `T` 的指针。省略 `space` 的 `!pto.ptr<T>` 默认为全局内存（GM）指针。

## 语法

```mlir
!pto.ptr<f16>
!pto.ptr<i32>
!pto.ptr<!pto.hif8>
!pto.ptr<f32, ub>
!pto.ptr<f16, l1>
```

## 参数

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `T` | 元素类型 | 指针所指向的元素类型 |
| `space` | 地址空间关键字，可选 | 默认为 `gm`；也支持 `ub`、`l1`、`l0a`、`l0b`、`l0c`、`bt`、`fb` |

本地空间也可写为 `vec`、`mat`、`left`、`right`、`acc`、`bias`、`scaling`，分别对应上表中的 `ub`、`l1`、`l0a`、`l0b`、`l0c`、`bt`、`fb`。

## 常见构造路径

- 作为函数参数出现
- 作为 `pto.addptr` 的结果
- 作为 `pto.castptr` 的结果

## 常见消费者

- `pto.make_tensor_view`
- `pto.load`
- `pto.store`
- `pto.castptr`

## 使用角色

在 PTO 程序中，`!pto.ptr<T>` 通常不直接表达 tile 级计算语义，而是承担以下角色：

- 全局内存入口
- 标量访存入口
- 构造 `tensor_view` 的基础句柄

## Constraints

- 指针元素类型必须是合法的元素类型
- 将整数与指针互转时，整数地址必须为 signless `i64`；指针之间的类型重解释必须保留地址空间。
- 指针包含元素类型和地址空间，不包含张量的 shape、stride、layout 或 Tile 有效区域元数据。

## Example

```mlir
func.func @kernel(%src: !pto.ptr<f16>, %dst: !pto.ptr<f16>) {
  return
}
```
