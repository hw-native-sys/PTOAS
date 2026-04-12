# TileLang DSL Syntax Sugar Proposals

## Overview

This document proposes syntax sugar enhancements for the TileLang Python DSL to improve programming ergonomics while maintaining close correspondence with the underlying VPTO IR. The current DSL design closely mirrors VPTO instructions, which can lead to verbose and error-prone code. These proposals aim to provide higher-level abstractions that compile down to the existing VPTO operations.

## Current Usability Challenges

### 1. **Low-Level Pointer Operations**
```python
# Current: manual byte offset management
ub_in = pto.castptr(0, pto.ptr(pto.f32, MemorySpace.UB))
ub_out = pto.castptr(4096, pto.ptr(pto.f32, MemorySpace.UB))
next_ptr = pto.addptr(ub_ptr, 4096)
```
**Problem**: Users must manage byte offsets and memory spaces manually.

### 2. **Verbose Copy Operations**
The `pto.copy_ubuf_to_ubuf` operation has 7 parameters:
- `src_offset`, `src_stride0`, `src_stride1`
- `dst_offset`, `dst_stride0`, `dst_stride1`

**Problem**: Correctly setting stride parameters is error-prone, especially for multi-dimensional data.

### 3. **Precise Mask Type Matching**
```python
# Must ensure mask granularity matches element type
mask32 = pto.pset_b32("PAT_ALL")  # f32 requires b32 mask
mask16 = pto.pset_b16("PAT_ALL")  # f16 requires b16 mask
```
**Problem**: Type error messages are not intuitive and easy to confuse.

### 4. **Strict Vector Scope Requirements**
```python
# strict_vecscope requires explicit capture of all variables
with pto.strict_vecscope(src_ptr, dst_ptr, start, end) as (s, d, lb, ub):
    # Can only use captured variables
```
**Problem**: Increases boilerplate code, especially when multiple variables need capture.

### 5. **Manual Synchronization Management**
```python
pto.set_flag(PIPE.MTE2, PIPE.V, EVENT.ID0)
pto.wait_flag(PIPE.MTE2, PIPE.V, EVENT.ID0)
```
**Problem**: Easy to forget synchronization or use wrong event IDs.

### 6. **Byte Offsets vs. Element Indices**
```python
# Need to calculate byte offsets
vec = pto.vlds(ub_ptr, lane * 256)  # Assuming f32, 4 bytes per element
```
**Problem**: Users must understand underlying memory layout.

## Proposed Syntax Sugar Enhancements

### 1. **Array View Abstraction**

#### Current API
```python
# Low-level pointer operations
ub_ptr = pto.castptr(0, pto.ptr(pto.f32, MemorySpace.UB))
vec = pto.vlds(ub_ptr, 64 * 4)  # Load 64th f32 element
```

#### Proposed Syntax Sugar
```python
# Create array views
ub_array = pto.ub_array(256, pto.f32, base_offset=0)  # 256-element f32 UB array
gm_array = pto.gm_array(1024, pto.f32, src)           # GM pointer array view

# Element access with automatic offset calculation
element = ub_array[64]          # Get 64th element (auto-calculates byte offset)
slice = ub_array[128:256]       # Slice operation

# Array assignment (compiles to appropriate copy operations)
ub_array[0:64] = gm_array[0:64]  # Compiles to copy_gm_to_ubuf

# Multi-dimensional arrays
ub_2d = pto.ub_array((256, 128), pto.f32)  # 2D array
row = ub_2d[32, :]                         # Row slice
col = ub_2d[:, 64]                         # Column slice
```

#### Implementation Notes
- `ub_array[64]` → `pto.vlds(ub_ptr, 64 * sizeof(f32))`
- `ub_array[0:64] = gm_array[0:64]` → Appropriate `copy_gm_to_ubuf` call with stride calculations
- Array views are compile-time constructs with no runtime overhead

### 2. **Simplified Copy Operations**

#### Current API
```python
pto.copy_gm_to_ubuf(src, dst, 0, 32, 128, 0, 0, False, 0, 128, 128)
```

#### Proposed Syntax Sugar
```python
# Full array copy
pto.copy_gm_to_ub(gm_array, ub_array)

# Slice copy with automatic stride calculation
pto.copy_gm_to_ub(gm_array[0:64], ub_array[128:192])

# Copy with element count
pto.copy_gm_to_ub(gm_array, ub_array, count=64)

# Transpose copy
pto.copy_gm_to_ub(gm_array, ub_array, transpose=True)

# Multi-dimensional copy with automatic stride inference
pto.copy_gm_to_ub(gm_2d[0:32, :], ub_2d[:, 0:64])

# Chained operations
(pto.copy_gm_to_ub(gm_array, ub_array)
 .then(pto.copy_ub_to_ub(ub_array, ub_temp))
 .then(pto.copy_ub_to_gm(ub_temp, dst_array)))
```

### 3. **Automatic Mask Inference**

#### Current API
```python
# Must specify mask type explicitly
mask32 = pto.pset_b32("PAT_ALL")
vec_f32 = pto.vlds(ptr, offset)
out = pto.vabs(vec_f32, mask32)
```

#### Proposed Syntax Sugar
```python
# Automatic mask type inference
mask = pto.pset("PAT_ALL")          # Inferred as mask_b32 for f32 vectors
out = pto.vabs(vec_f32, mask)       # Type-safe, auto-matched

# Vector method syntax (more Pythonic)
out = vec_f32.abs(mask="PAT_ALL")
out = vec_f32.add(other_vec, mask=pto.pset("PAT_EVEN"))
out = vec_f32.max(scalar, mask="PAT_ALL")

# Mask creation from comparison
mask = vec_f32 >= pto.f32(0.0)      # Creates appropriate mask_b32
mask = vec_f32 < threshold          # Auto-infers mask type

# Mask operations with auto-typing
combined = mask1 & mask2            # Bitwise AND with type preservation
inverted = ~mask                    # Logical NOT
```

### 4. **Simplified Synchronization Primitives**

#### Current API
```python
pto.set_flag(PIPE.MTE2, PIPE.V, EVENT.ID0)
# ... computation ...
pto.wait_flag(PIPE.MTE2, PIPE.V, EVENT.ID0)
```

#### Proposed Syntax Sugar
```python
# Context manager for automatic synchronization
with pto.sync_between(PIPE.MTE2, PIPE.V, event=EVENT.ID0):
    # set_flag called on entry, wait_flag on exit
    pto.copy_gm_to_ub(src, dst)
    compute_block()

# Decorator for function-level synchronization
@pto.synchronized(from_pipe=PIPE.MTE2, to_pipe=PIPE.V)
def compute_block():
    # Automatic synchronization before and after
    pass

# Pipeline synchronization chain
with pto.pipeline([
    (PIPE.MTE2, PIPE.V, EVENT.ID0),
    (PIPE.V, PIPE.MTE3, EVENT.ID1),
    (PIPE.MTE3, PIPE.S, EVENT.ID2)
]):
    # Multi-stage synchronization
    stage1()
    stage2()
    stage3()
```

### 5. **Element-Level Indexing Operations**

#### Current API
```python
# Byte offset calculation required
vec = pto.vlds(ub_ptr, lane * 256)  # Need to know f32 is 4 bytes
```

#### Proposed Syntax Sugar
```python
# Element-level indexing
vec = pto.vlde(ub_array, lane)      # Automatic byte offset calculation
pto.vste(vec, ub_array, lane)       # Element-level store

# Array view methods
vec = ub_array.load_element(lane)
ub_array.store_element(lane, vec)

# Batch operations
vectors = ub_array.load_elements([0, 64, 128, 192])
ub_array.store_elements([256, 320, 384], vectors)

# Strided access
stride = ub_array.load_stride(start=0, end=1024, step=64)
```

### 6. **Type Inference Simplification**

#### Current API
```python
# Explicit type annotations required
remaining: pto.i32 = 1024
# or
remaining = pto.i32(1024)
```

#### Proposed Syntax Sugar
```python
# Automatic type inference for constants
remaining = pto.constant(1024)      # Inferred as i32 or i64 from context
step = pto.constant(64, type=pto.i32)  # Explicit type specification

# Typed range with automatic inference
for i in pto.range(0, 1024, 64):    # i automatically gets correct machine type
    # i is pto.i32

# Function argument type inference
@pto.vkernel
def kernel(x):  # Type inferred from usage
    return x * pto.constant(2)  # x type inferred from multiplication

# Variable type inference from operations
result = pto.constant(10) + pto.constant(20)  # result is pto.i32
```

### 7. **More Flexible Vector Scopes**

#### Current API
```python
# Explicit capture required
with pto.strict_vecscope(src_ptr, dst_ptr, start, end) as (s, d, lb, ub):
    for i in range(lb, ub, step):
        vec = pto.vlds(s, i)
        pto.vsts(vec, d, i, mask)
```

#### Proposed Syntax Sugar
```python
# Automatic variable capture
with pto.vector_scope():
    # Variables used in scope are automatically captured
    for i in pto.range(start, end, step):
        vec = src_array.load_element(i)
        dst_array.store_element(i, vec.abs())

# Decorator for vectorized functions
@pto.vectorize
def compute_element(src, dst, index):
    vec = src.load_element(index)
    dst.store_element(index, vec.abs())

# Apply vectorized function across range
pto.vector_map(compute_element, src_array, dst_array, range(0, 1024, 64))

# Lambda support
pto.vector_map(lambda x: x.abs(), src_array, dst_array)
```

### 8. **Built-in Utility Functions**

#### Common Pattern Encapsulation
```python
# Vector map/reduce operations
result = pto.vector_map(abs, src_array, dst_array)          # Element-wise mapping
sum = pto.vector_reduce(add, array)                         # Reduction
max_val = pto.vector_reduce(max, array)                     # Maximum reduction

# Vector zip/unzip
zipped = pto.vector_zip(src1, src2, dst)                    # Interleave
unzipped1, unzipped2 = pto.vector_unzip(src, dst1, dst2)    # Deinterleave

# Mathematical functions
result = pto.vector_sin(array)
result = pto.vector_exp(array)
result = pto.vector_relu(array)
result = pto.vector_sigmoid(array)

# Statistical operations
mean = pto.vector_mean(array)
variance = pto.vector_variance(array)
min_val, max_val = pto.vector_minmax(array)

# Linear algebra (small-scale)
dot_product = pto.vector_dot(vec1, vec2)
norm = pto.vector_norm(array)
```

## Implementation Strategy

These syntax sugar enhancements can be implemented through:

1. **Python Decorators and Context Managers**: For synchronization and vector scopes
2. **Wrapper Classes**: `UBArray`, `GMArray`, `Vector` classes that encapsulate low-level operations
3. **Operator Overloading**: Support for `[]`, `:`, arithmetic operators on wrapper classes
4. **Type Inference System**: Context-based machine type inference
5. **Compile-time Transformation**: Conversion of high-level syntax to low-level VPTO operations before IR generation

## Compatibility with VPTO IR

**Key Principle**: All syntax sugar must ultimately lower to existing VPTO operations.

### Lowering Examples

| Syntax Sugar | VPTO IR Equivalent |
|--------------|-------------------|
| `ub_array[64]` | `pto.vlds(ub_ptr, 64 * sizeof(f32))` |
| `pto.copy_gm_to_ub(src_array, dst_array)` | Appropriate `copy_gm_to_ubuf` call with calculated strides |
| `with pto.sync_between(...):` | `set_flag` + `wait_flag` pair |
| `mask = vec_f32 >= pto.f32(0.0)` | `pto.pge_b32(vec_f32, pto.f32(0.0))` |
| `vec_f32.abs(mask="PAT_ALL")` | `pto.vabs(vec_f32, pto.pset_b32("PAT_ALL"))` |

## Prioritization

### High Priority (Immediate Value)
1. Array view abstraction
2. Simplified copy operations  
3. Automatic mask inference

### Medium Priority (Significant Ergonomics Improvement)
4. Element-level indexing
5. Type inference simplification
6. Flexible vector scopes

### Low Priority (Advanced Features)
7. Enhanced synchronization primitives
8. Built-in utility functions

## Migration Path

The existing low-level API will remain available for performance-critical code or direct VPTO IR correspondence. Syntax sugar will be provided as an optional layer that can be mixed with low-level operations.

```python
# Mixed usage example
@pto.vkernel
def mixed_kernel(src: pto.ptr(pto.f32, MemorySpace.GM),
                 dst: pto.ptr(pto.f32, MemorySpace.GM)):
    # Low-level: manual pointer setup
    ub_in = pto.castptr(0, pto.ptr(pto.f32, MemorySpace.UB))
    
    # High-level: array view for computation
    ub_array = pto.ub_array(256, pto.f32, base_ptr=ub_in)
    
    # Mixed: low-level copy, high-level computation
    pto.copy_gm_to_ubuf(src, ub_in, 0, 32, 128, 0, 0, False, 0, 128, 128)
    
    with pto.vector_scope():
        for i in pto.range(0, 256, 64):
            vec = ub_array.load_element(i)
            result = vec.abs(mask="PAT_ALL")
            ub_array.store_element(i, result)
    
    # Low-level: copy back
    pto.copy_ubuf_to_gm(ub_in, dst, 0, 32, 128, 0, 128, 128)
```

## Next Steps

1. **Prototype Implementation**: Start with array view abstraction and simplified copy operations
2. **User Feedback**: Gather feedback from performance engineers on the proposed syntax
3. **Gradual Rollout**: Implement enhancements in phases, starting with high-priority items
4. **Documentation**: Update DSL guide with syntax sugar examples and migration guides
5. **Testing**: Ensure all syntax sugar correctly lowers to VPTO IR and maintains performance

These enhancements will significantly improve the TileLang DSL's usability while maintaining the close correspondence with underlying VPTO IR that performance engineers require.

1. 软件流水线（Software Pipelining）的表达成本
在 NPU 上写 Vector 级算子，最难的往往不是数值计算，而是利用 UB (Unified Buffer) 进行 Double/Multi-Buffering（乒乓缓存），并手动排布内存搬运与计算的流水线。

现状挑战：如果开发者全靠手写 set_flag、wait_flag，以及手动维护 Ping-Pong 缓冲的偏移量，代码会迅速膨胀且极易死锁或读写冲突。

优化建议：DSL 在保留底层原语的同时，可以提供稍微高级一点的流水线抽象。例如，引入 pto.CircularBuffer(tile, num_stages=2) 的概念，让开发者可以专注于“当前 stage 的计算”，而由底层生成器自动完成不同 stage 的指针轮转和 Flag 同步。

2. Python 宿主变量 vs MLIR SSA 变量的心智模型边界
因为 DSL 的本质是用 Python 元编程来生成 MLIR（静态图），开发者在写代码时很容易混淆“Python 运行期的值”和“NPU 运行期的值”。

现状挑战：手册中提到“变量的自动合并”（比如 if 分支产生合并），这涉及到复杂的 SSA 转换。特别是在 for 循环中，**循环携带状态（Loop-carried state）**的处理往往是个痛点。如果开发者在循环外定义了一个 Python 列表或字典，在循环内去修改它，这在生成 MLIR 的 scf.for 时是无法正确映射的。

优化建议：需要有极其明确的类型系统提示或语法边界，强制区分编译期求值的变量（Meta-variables）和生成的 MLIR Value。可以考虑借鉴 Triton 的方式，提供类似 tl.constexpr 的装饰或类型，让开发者清楚哪些分支在生成 MLIR 时会被静态展开，哪些会真正生成 scf.if。

3. 地址计算（Address Generation）的易错性
即使是对底层开发者，手动计算字节偏移也是痛苦且容易出 Bug 的。

现状挑战：i * cols * 4 这种强依赖 f32 占用 4 字节的硬编码，在泛型算子开发中会带来负担（比如想写一个同时兼容 f16 和 f32 的模板算子）。

优化建议：提供基于语义的视图（View）操作。保留控制力不代表必须算字节。可以提供类似 tile.get_vector_slice(row_idx, vec_idx) 的接口，它在内部自动 Emit（发射）对应的 MLIR 乘法和加法指令来计算 offset。这不仅防呆，还能让生成的 MLIR 结构更规范。

4. Mask 的隐式推导（针对边界处理）
NPU 算子经常要处理尾部不对齐的数据（Tail processing）。

优化建议：虽然底层需要具体的 Mask 寄存器配置（如 PAT_ALL），但在 for 循环的最后一步边界处理时，能否提供一个类似 pto.make_mask(remaining_elements) 的宏/内联函数？让它在生成 MLIR 时，自动展开为对应的硬件 plt_b32 等指令，这样可以大幅减少手写冗长边界判断的样板代码。