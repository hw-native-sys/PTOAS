# 4.3 指针类型

## 1. 概述

`!pto.ptr<T>` 表示指向全局内存元素 `T` 的指针，是 PTO 程序从外部地址空间进入 PTO 对象模型的基础类型。

## 2. 语法

```mlir
!pto.ptr<f16>
!pto.ptr<i32>
!pto.ptr<!pto.hif8>
```

## 3. 参数

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `T` | 元素类型 | 指针所指向的元素类型 |

## 4. 常见构造路径

- 作为函数参数出现
- 作为 `pto.addptr` 的结果
- 作为 `pto.inttoptr` 的结果

## 5. 常见消费者

- `pto.make_tensor_view`
- `pto.load_scalar`
- `pto.store_scalar`
- `pto.ptrtoint`

## 6. 使用角色

在 PTO 程序中，`!pto.ptr<T>` 通常不直接表达 tile 级计算语义，而是承担以下角色：

- 全局内存入口
- 标量访存入口
- 构造 `tensor_view` 的基础句柄

## 7. Constraints

- 指针元素类型必须是合法的元素类型
- 将整数与指针互转时，后续用途必须满足相关验证约束
- 仅有指针本身并不包含 shape、stride、layout 或 tile 位置语义

## 8. Example

```mlir
func.func @kernel(%src: !pto.ptr<f16>, %dst: !pto.ptr<f16>) {
  return
}
```
