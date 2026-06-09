# PTO 与 PTOAS 用户手册

## 1. 概述

- [1.1 什么是 PTO](./01_01_what_is_pto.md)
- [1.2 什么是 PTOAS](./01_02_what_is_ptoas.md)

## 2. 编程模型

- [2.1 PTO 机器模型](./02_01_machine_model.md)
- [2.2 PTO 执行模型](./02_02_execution_model.md)
- [2.3 Tile 与 Tensor View](./02_03_tile_and_tensor_view.md)
- [2.4 PTO ISA 层级](./02_04_ir_levels.md)
- [2.5 PTO 同步模型](./02_05_sync_model.md)
- [2.6 CV Pipe](./02_06_cv_pipe.md)
- [2.7 目标架构与使用差异](./02_07_architecture_notes.md)

## 3. 语法

- [3.1 PTO ISA 语法总览](./03_01_pto_isa_syntax_reference.md)
- [3.2 模块与函数语法](./03_02_module_and_function.md)
- [3.3 SSA 值与 Region](./03_03_ssa_and_region.md)
- [3.4 类型语法](./03_04_type_syntax.md)
- [3.5 属性语法](./03_05_attribute_syntax.md)
- [3.6 Operation 汇编格式](./03_06_operation_assembly.md)

## 4. 类型系统

- [4.1 PTO ISA 类型系统总览](./04_01_pto_isa_type_system_reference.md)
- [4.2 元素类型](./04_02_element_types.md)
- [4.3 指针类型](./04_03_pointer_types.md)
- [4.4 Tensor View 类型](./04_04_tensor_view_types.md)
- [4.5 Tile Buffer 类型](./04_05_tile_buffer_types.md)

## 5. 内存模型

- [5.1 地址空间](./05_01_address_spaces.md)
- [5.2 PTO 存储模型](./05_02_storage_model.md)
- [5.3 Tile 分配](./05_03_tile_allocation.md)
- [5.4 Reserved Buffer](./05_04_reserved_buffers.md)
- [5.5 A3 与 A5 的内存使用差异](./05_05_a3_a5_memory_diff.md)

## 6. 操作

- [6.1 资源绑定操作](./06_01_resource_binding_ops.md)
- [6.2 指针与视图操作](./06_02_pointer_and_view_ops.md)
- [6.3 数据搬运操作](./06_03_data_movement_ops.md)
- [6.4 矩阵计算操作](./06_04_matrix_compute_ops.md)
- [6.5 逐元素操作（Tile-Tile）](./06_05_elementwise_tilt_tile_op.md)
- [6.6 Tile-标量/Tile-立即数操作](./06_06_tile_scalar_immediate_ops.md)
- [6.7 轴规约与广播操作](./06_07_reduction_and_broadcast_ops.md)
- [6.8 重排与转换操作](./06_08_relayout_and_convert_ops.md)
- [6.9 同步操作](./06_09_sync_ops.md)
- [6.10 核内 CV Pipe 前端接口](./06_10_cv_pipe_ops.md)
- [6.11 运行时与调试操作](./06_11_runtime_and_debug_ops.md)
- [6.12 复杂操作](./06_12_complex_ops.md)

## 7. 编译选项指南

- [7.1 编译选项总览](./07_01_pipeline_overview.md)
- [7.2 目标架构选项](./07_02_layout_inference.md)
- [7.3 PTO 层级选项](./07_03_memory_planning.md)
- [7.4 布局相关选项](./07_04_auto_sync.md)
- [7.5 自动内存规划](./07_05_codegen_notes.md)
- [7.6 自动同步选项](./07_06_auto_sync_options.md)
- [7.7 调试与排查](./07_07_debugging_and_inspection.md)

## 8. 示例

- [8.1 从 PTO 到输出代码](./08_01_build_and_run_overview.md)
- [8.2 MatMul 示例](./08_02_matmul_example.md)
- [8.3 Softmax 示例](./08_03_softmax_example.md)
- [8.4 Flash Attention 示例](./08_04_flash_attention_example.md)
