# TileLang Python DSL Guide

The TileLang Python DSL provides a high-level, Pythonic interface for authoring vector compute kernels targeting the Ascend NPU hardware. This guide is intended for library developers and performance engineers who need to write efficient, hardware-aware kernels using the PTO micro instruction set.

The DSL is designed to generate MLIR function libraries rather than direct binary executables. These MLIR libraries are intended to be consumed by other compilation frameworks that transform high-level tile semantics into low-level vector operations. This enables library developers to focus on hardware-aware kernel authoring while relying on upstream compilers for tile-level optimizations and code generation.

## Quick Start

**Note on mask pattern enums**: For brevity, examples in this guide use `PAT` as an alias for `pto.MaskPattern` (e.g., `PAT.ALL` instead of `pto.MaskPattern.PAT_ALL`). You can create this alias with `from pto import MaskPattern as PAT` or `PAT = pto.MaskPattern`.

Here's a minimal example of a tile scaling kernel using the new Tile type:

```python
import pto

@pto.vkernel(target="a5", op="scale", dtypes=[(pto.AnyFloat, pto.AnyFloat)], priority=10)
def tile_scale(input_tensor: pto.TensorView,   # Input tensor view (shape: 256x128, f32, GM)
               output_tensor: pto.TensorView,  # Output tensor view (same shape and type)
               scale_factor: pto.f32): # Scaling factor
    # Access tensor properties
    rows, cols = input_tensor.shape      # (256, 128)
    dtype = input_tensor.element_type    # pto.f32
    
    # Create a temporary tile in UB for computation
    ub_tile = pto.tile((rows, cols), dtype, pto.MemorySpace.UB)
    
    # Load input tensor from GM to UB using high-level DMA operation
    pto.dma_load(input_tensor, ub_tile)
    
    # Vector computation: scale all elements in the tile
    all_mask = pto.make_mask(dtype, PAT.ALL)
    
    # Process tile in row-major order
    for row in range(0, rows):
        # Process each row in vector chunks
        # Vector width is hardware-defined: 256 bytes / element size
        # For f32: 256/4 = 64 lanes, for f16: 256/2 = 128 lanes
        vector_lanes = pto.get_lanes(dtype)  # Compute vector lanes based on element type (e.g., 64 for f32, 128 for f16)
        for col_start in range(0, cols, vector_lanes):
            # Load vector using element-indexing syntax (no manual byte calculation)
            vec = pto.vlds(ub_tile[row, col_start:])
            
            # Scale vector
            scaled = pto.vmuls(vec, scale_factor, all_mask)
            
            # Store result back using element-indexing syntax
            pto.vsts(scaled, ub_tile[row, col_start:], all_mask)
    
    # Store result from UB back to GM output tensor using high-level DMA operation
    pto.dma_store(ub_tile, output_tensor)
```

This example demonstrates:
1. **TensorView parameters** in kernel declaration
2. **TensorView property access** (shape, element_type)  
3. **Tile creation** for temporary buffers
4. **High-level DMA operations** (`dma_load`/`dma_store`) for data movement
5. **Implicit tile→UBRef conversion** in vector load/store operations
6. **Automatic DMA parameter inference** from tensor slices and tile properties

For an even more concise example showing pure computation on UB tiles (assuming data is already in UB):

```python
@pto.vkernel(target="a5", op="elementwise", dtypes=[(pto.AnyFloat, pto.AnyFloat, pto.AnyFloat)], priority=10)
def ub_tile_computation(a: pto.Tile,  # UB tile
                        b: pto.Tile,  # UB tile  
                        c: pto.Tile): # UB tile (output)
    dtype = a.element_type

    # All tiles are in UB memory space
    all_mask = pto.make_mask(dtype, PAT.ALL)
    rows, cols = a.shape
    
    # Element-wise: c = a + b * 2.0
    for i in range(0, rows * cols, 64):
        # Load vectors from UB tiles using element-indexing syntax
        vec_a = pto.vlds(a[i:])         # Implicit tile→UBRef with automatic offset calculation
        vec_b = pto.vlds(b[i:])
        
        # Compute: b * 2.0
        scaled_b = pto.vmuls(vec_b, 2.0, all_mask)
        
        # Compute: a + scaled_b
        result = pto.vadd(vec_a, scaled_b, all_mask)
        
        # Store result to output tile using element-indexing syntax
        pto.vsts(result, c[i:], all_mask)
```

## Core Concepts

### Kernel Declaration

Kernels are defined using the `@pto.vkernel` decorator with enhanced matching capabilities for PTO operations. The decorator specifies matching criteria for target architecture, operation type, data types, and additional constraints, along with a priority for disambiguation when multiple kernels match.

#### Basic Syntax

```python
@pto.vkernel(
    target="a5",                     # Target architecture
    op="matmul",                    # PTO operation name to match
    dtypes=[(pto.f16, pto.f16, pto.f32)],  # Type signatures
    constraints=[                    # Additional constraints
        AnyOf(k_dim_aligned_64, continuous_memory),
        Not(requires_ub_memory)
    ],
    priority=100                    # Priority for selection
)
def matmul_fallback(a: pto.Tile, b: pto.Tile, c: pto.Tile) -> None:
    # kernel implementation
```

#### Decorator Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `target` | `str` | Yes | Target hardware architecture (e.g., `"a5"` for Ascend 950). |
| `op` | `str` | Yes | Name of the PTO operation to match (e.g., `"matmul"`, `"conv2d"`, `"add"`). |
| `dtypes` | `List[Tuple[Type, ...]]` | Yes | List of type signatures. Each tuple specifies the expected data types for the operation's operands (inputs and outputs) in order. |
| `constraints` | `List[Constraint]` | No | Additional constraints that must be satisfied for the kernel to be selected. Can include logical combinations (`AnyOf`, `AllOf`, `Not`). Default: empty list. |
| `priority` | `int` | No | Selection priority when multiple kernels match. Higher values have higher priority. Default: `0`. |
| `name` | `str` | No | Kernel name (used for debugging and profiling). Defaults to the decorated function's name. |

#### Type Matching Rules

The `dtypes` parameter supports flexible type matching:

1. **Concrete Types**: Exact type matches using DSL scalar types:
   - `pto.f16`, `pto.f32`, `pto.bf16`
   - `pto.i8`, `pto.i16`, `pto.i32`, `pto.i64`
   - `pto.mask_b8`, `pto.mask_b16`, `pto.mask_b32`

2. **Type Wildcards**: Generic type patterns:
   - `pto.AnyFloat`: Matches any floating-point type (`f16`, `bf16`, `f32`)
   - `pto.AnyInt`: Matches any integer type (`i8`, `i16`, `i32`, `i64`)
   - `pto.AnyType`: Matches any scalar type
   - `pto.AnyMask`: Matches any mask type (`mask_b8`, `mask_b16`, `mask_b32`)

3. **Type Variables**: Named type variables that enforce consistency within a signature:
   ```python
   T = pto.TypeVar('T')  # Define a type variable
   
   @pto.vkernel(
       target="a5",
       op="elementwise",
       dtypes=[(T, T, T)],  # All three operands must have the same type
       constraints=[]
   )
   def elementwise_same_type(x: pto.Tile, y: pto.Tile, out: pto.Tile) -> None:
       # x, y, and out must have identical element types
       pass
   ```

4. **Mixed Signatures**: Multiple type signatures for the same operation:
   ```python
   @pto.vkernel(
       target="a5",
       op="add",
       dtypes=[
           (pto.AnyFloat, pto.AnyFloat, pto.AnyFloat),  # Float addition
           (pto.AnyInt, pto.AnyInt, pto.AnyInt)         # Integer addition
       ]
   )
   def generic_add(a: pto.Tile, b: pto.Tile, c: pto.Tile) -> None:
       # Supports both float and integer types
       pass
   ```

#### Constraint System

Constraints are compile-time predicates that refine kernel selection. The system supports logical combinations of constraints.

##### Predefined Constraints

| Constraint | Description |
|------------|-------------|
| `k_dim_aligned_64` | K dimension is aligned to 64 elements (for matmul kernels). |
| `continuous_memory` | Operands reside in contiguous memory regions. |
| `requires_ub_memory` | Operation requires Unified Buffer memory (vs. Global Memory). |
| `tensor_rank(rank)` | Operand tensor has specified rank (e.g., `tensor_rank(2)` for 2D tensors). |
| `broadcastable` | Operands are broadcastable according to NumPy-style broadcasting rules. |
| `static_shape` | All tensor dimensions are known at compile time (no dynamic shapes). |

##### Logical Constraint Combinators

| Combinator | Description | Example |
|------------|-------------|---------|
| `AnyOf(c1, c2, ...)` | At least one of the constraints must be satisfied. | `AnyOf(k_dim_aligned_64, continuous_memory)` |
| `AllOf(c1, c2, ...)` | All constraints must be satisfied. | `AllOf(tensor_rank(2), static_shape)` |
| `Not(c)` | The constraint must not be satisfied. | `Not(requires_ub_memory)` |

##### Custom Constraints

Users can define custom constraints using predicate functions:

```python
# Define a custom constraint
def large_batch(batch_size: pto.i32) -> pto.Constraint:
    """Batch size must be ≥ 1024."""
    return pto.Constraint(lambda op: op.batch_size >= batch_size)

@pto.vkernel(
    target="a5",
    op="matmul",
    dtypes=[(pto.AnyFloat, pto.AnyFloat, pto.AnyFloat)],
    constraints=[large_batch(1024)]
)
def large_batch_matmul(a: pto.Tile, b: pto.Tile, c: pto.Tile) -> None:
    # Optimized for large batch sizes
    pass
```

#### Kernel Selection Mechanism

When a PTO operation needs implementation, the system performs the following matching process:

1. **Target Filtering**: Select kernels with matching `target` architecture.
2. **Operation Filtering**: Select kernels with matching `op` name.
3. **Type Matching**: For each kernel's `dtypes` list, check if any signature matches the operation's operand types:
   - Concrete types must match exactly.
   - Wildcard types match according to their category.
   - Type variables must be consistent within the signature.
4. **Constraint Validation**: For each matching kernel, evaluate all `constraints`. If any constraint fails, the kernel is rejected.
5. **Priority Selection**: From the remaining kernels, select the one with the highest `priority` value.
6. **Fallback**: If no kernel matches, compilation fails with an error.

#### Examples

##### Matmul with Multiple Implementations

```python
# High-performance kernel for aligned K dimension
@pto.vkernel(
    target="a5",
    op="matmul",
    dtypes=[(pto.f16, pto.f16, pto.f32)],
    constraints=[k_dim_aligned_64],
    priority=200
)
def matmul_aligned_k(a: pto.Tile, b: pto.Tile, c: pto.Tile) -> None:
    # Optimized implementation for aligned K
    pass

# General-purpose fallback
@pto.vkernel(
    target="a5",
    op="matmul",
    dtypes=[(pto.AnyFloat, pto.AnyFloat, pto.AnyFloat)],
    constraints=[],
    priority=100
)
def matmul_general(a: pto.Tile, b: pto.Tile, c: pto.Tile) -> None:
    # Generic implementation
    pass
```

##### Elementwise Operation with Type Polymorphism

```python
@pto.vkernel(
    target="a5",
    op="add",
    dtypes=[
        (pto.AnyFloat, pto.AnyFloat, pto.AnyFloat),
        (pto.AnyInt, pto.AnyInt, pto.AnyInt)
    ],
    constraints=[broadcastable]
)
def polymorphic_add(a: pto.Tile, b: pto.Tile, out: pto.Tile) -> None:
    # Single implementation handles both float and integer types
    dtype = a.element_type
    all_mask = pto.make_mask(dtype, PAT.ALL)
    # ... implementation using generic vector operations
    pass
```

##### Constrained Convolution Kernel

```python
@pto.vkernel(
    target="a5",
    op="conv2d",
    dtypes=[(pto.f16, pto.f16, pto.f32)],
    constraints=[
        AllOf(
            tensor_rank(4),          # NHWC format
            static_shape,            # No dynamic dimensions
            Not(requires_ub_memory)  # GM memory preferred
        )
    ],
    priority=150
)
def conv2d_nhwc_f16_f32(input: pto.Tile, filter: pto.Tile, output: pto.Tile) -> None:
    # Optimized for NHWC layout with static shapes
    pass
```

### Value Model

The DSL operates on symbolic values, not Python runtime values:
- **Constants**: Python literals that are typed to machine types
- **Operation results**: Values produced by DSL operations
- **Block arguments**: Values introduced by control flow structures

### Memory Spaces

The DSL supports different memory spaces:
- `MemorySpace.GM`: Global Memory
- `MemorySpace.UB`: Unified Buffer (local storage for vector computation)

## Type System

### Scalar Types

| DSL Type | Description | Bit Width |
|----------|-------------|-----------|
| `pto.i1` | Boolean | 1 |
| `pto.i8` | 8-bit integer | 8 |
| `pto.i16` | 16-bit integer | 16 |
| `pto.i32` | 32-bit integer | 32 |
| `pto.i64` | 64-bit integer | 64 |
| `pto.f16` | Half precision float | 16 |
| `pto.bf16` | Brain float 16 | 16 |
| `pto.f32` | Single precision float | 32 |

Python literals are automatically typed:
- `bool` → `pto.i1`
- `int` → Context-dependent (typically `pto.i32` or `pto.i64`)
- `float` → `pto.f32`

For explicit typing, use type constructors:
```python
x = pto.i32(1024)      # Explicit i32 constant
y: pto.i32 = 1024      # Type annotation
```

### Vector Types

Vector registers have fixed 256-byte width:

```python
v64_f32 = pto.vreg(64, pto.f32)    # 64 lanes of f32 (64 * 32b = 2048b)
v128_f16 = pto.vreg(128, pto.f16)  # 128 lanes of f16 (128 * 16b = 2048b)
```

Constraint: `lanes × bitwidth(element_type) = 2048`

### Typed Masks

Masks are typed by their bit granularity:

| DSL Type | VPTO Type | Description |
|----------|-----------|-------------|
| `pto.mask_b8` | `!pto.mask<b8>` | 8-bit granularity mask |
| `pto.mask_b16` | `!pto.mask<b16>` | 16-bit granularity mask |
| `pto.mask_b32` | `!pto.mask<b32>` | 32-bit granularity mask |

Mask operations must match the vector element family:
- `f32` vectors use `mask_b32`
- `f16` vectors use `mask_b16`
- `i8` vectors use `mask_b8`

```python
# Correct: f32 vector with b32 mask
mask32 = pto.make_mask(pto.f32, PAT.ALL)
vec_f32 = pto.vlds(ptr, offset)
out = pto.vabs(vec_f32, mask32)

# Error: mismatched mask granularity
mask16 = pto.make_mask(pto.f16, PAT.ALL)
out = pto.vabs(vec_f32, mask16)  # Type error!
```

### Pointer Types

Pointers combine element type and memory space:

```python
from pto import MemorySpace

ptr_gm = pto.ptr(pto.f32, MemorySpace.GM)    # GM pointer to f32
ptr_ub = pto.ptr(pto.f16, MemorySpace.UB)    # UB pointer to f16
```

The `MemorySpace` enum provides type-safe memory space specification:

| Enum Value | Description |
|------------|-------------|
| `MemorySpace.GM` | Global Memory (off-chip HBM/DDR) |
| `MemorySpace.UB` | Unified Buffer (on-chip SRAM, 256KB) |

This replaces string literals (`MemorySpace.GM`/`MemorySpace.UB`) with compile-time checked enums.

### Pointer Type Aliases

For clarity in API documentation, the following type aliases are used:

| Alias | Equivalent Type | Description |
|-------|----------------|-------------|
| `GMPtr` | `ptr(..., MemorySpace.GM)` | Pointer to Global Memory |
| `UBPtr` | `ptr(..., MemorySpace.UB)` | Pointer to Unified Buffer |
| `UBRef` | `Union[MemRefType, UBPtr]` | UB buffer or pointer (accepted by load/store ops) |
| `Tile` | `pto.tile(...)` | Tile buffer with layout and configuration |

### MemRef Types

For buffer-like authoring, use memref types:

```python
buf1d = pto.memref(256, pto.f32, MemorySpace.UB)          # 1D: 256-element f32 buffer in UB
buf2d = pto.memref((256, 128), pto.f32, MemorySpace.UB)   # 2D: 256x128 f32 buffer in UB
```

- **1D shapes**: Use a scalar integer (e.g., `256`)
- **Multi-dimensional shapes**: Use a tuple (e.g., `(256, 128)`)

MemRefs are used for stateless load/store operations that accept `buf_like` operands in VPTO.


### TensorView Types

TensorView types represent views into tensors residing in Global Memory (GM). They are used as kernel parameters for describing GM data and support slicing operations to create logical partitions for DMA load/store operations.

### TensorView Type Definition

TensorView types are parameterized by shape and element type:

```python
# Kernel parameter using TensorView
@pto.vkernel(target="a5", op="custom", dtypes=[(pto.AnyFloat, pto.AnyFloat, pto.AnyFloat)], priority=10)
def tiled_kernel(
    input_tensor: pto.TensorView,   # GM tensor view
    output_tensor: pto.TensorView,  # GM tensor view
    tile_buf: pto.Tile              # UB tile
):
    # Access tensor view properties
    rows, cols = input_tensor.shape      # (dynamic or static)
    dtype = input_tensor.element_type    # e.g., pto.f32
    strides = input_tensor.strides       # stride in elements
```

**Important Notes:**
- TensorView is a **read-only descriptor** for GM data (though DMA store operations can write to it)
- Shape can be **static** (compile-time constants) or **dynamic** (determined at runtime)
- Strides are expressed in elements, not bytes
- Memory space is always GM (Global Memory)

### TensorView Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple[int, ...]` | Tensor dimensions (2D only in current profile) |
| `element_type` | `Type` | Element data type (e.g., `pto.f32`, `pto.f16`) |
| `strides` | `tuple[int, ...]` | Stride in elements for each dimension |
| `offset` | `pto.i64` | Byte offset from base pointer (internal) |

### Padding Mode Enum

Padding mode controls how out-of-bounds accesses are handled during DMA load/store operations:

| Enum Value | Description |
|------------|-------------|
| `PadMode.PadNull` | No padding (out-of-bounds access is invalid) |
| `PadMode.PadFirstElem` | Pad using the first element of the source |
| `PadMode.PadValue` | Pad using a specified value (requires `pad_value` parameter) |

**Usage:**
```python
from pto import PadMode

# Load with zero padding
pto.dma_load(src_partition, dst_tile, 
          pad_mode=PadMode.PadValue, 
          pad_value=pto.f32(0.0))

# Load with first-element padding
pto.dma_load(src_partition, dst_tile, pad_mode=PadMode.PadFirstElem)

# Load without padding (default)
pto.dma_load(src_partition, dst_tile)  # pad_mode=PadMode.PadNull
```

### Slicing Syntax

TensorView supports Python slicing syntax to create logical partitions:

```python
# Create a partition from a tensor view
partition = tensor_view[row_start:row_end, col_start:col_end]

# Example: extract a 16x16 tile from a larger tensor
tile_view = large_tensor[0:16, 0:16]

# Dynamic offsets and sizes
start_row = pto.i32(0)
start_col = pto.i32(0)
dynamic_partition = tensor_view[start_row:start_row+16, start_col:start_col+16]
```

**Constraints:**
- Slicing returns a new TensorView representing the logical partition
- The partition must be within the original tensor bounds
- Slices can be static (constant bounds) or dynamic (runtime values)

### Alignment Type

The `pto.align` type is used for alignment carrier operations and maps to `!pto.align`.

### Tile Types

Tile types represent data blocks in memory with layout and configuration information, corresponding to `!pto.tile_buf` in the VPTO IR. Tiles are commonly used as kernel parameters for tiled computations.

#### Tile Type Definition

```python
# Create a tile with shape, element type, and memory space
tile = pto.tile((256, 128), pto.f32, MemorySpace.UB)

# With explicit configuration
config = pto.tile_config(
    b_layout=pto.BLayout.ROW_MAJOR,
    s_layout=pto.SLayout.NONE_BOX,
    s_fractal_size=pto.i32(16),
    pad_value=pto.PadValue.ZERO
)
tile = pto.tile((256, 128), pto.f32, MemorySpace.UB, config=config)

# With valid shape (actual data dimensions within tile)
tile = pto.tile((256, 128), pto.f32, MemorySpace.UB, valid_shape=(240, 120))
```

**Important Notes on Shape and Valid Shape:**
- **Static Shape Requirement**: The `shape` parameter must be a compile-time constant. Tile dimensions are fixed at compilation time and cannot change at runtime.
- **Valid Shape Constraints**: The `valid_shape` parameter can be either static (compile-time constant) or dynamic (determined at runtime). It must be less than or equal to the physical `shape` in each dimension. This allows for variable-sized data within a fixed tile allocation.
- **Default Behavior**: When `valid_shape` is not specified, it defaults to the full `shape`.

#### Tile Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple[int, ...]` | **Static** full tile dimensions (compile-time constant) |
| `element_type` | `Type` | Element data type (e.g., `pto.f32`) |
| `memory_space` | `MemorySpace` | Memory space (GM, UB, etc.) |
| `valid_shape` | `tuple[int, ...]` | Actual data dimensions within tile (can be static/compile-time or dynamic/runtime). Must be ≤ shape in each dimension. |
| `config` | `TileConfig` | Layout and padding configuration |

#### Tile Configuration

The tile configuration includes layout and padding information:

```python
# Layout enums
pto.BLayout.ROW_MAJOR     # 0: row-major base layout
pto.BLayout.COL_MAJOR     # 1: column-major base layout

pto.SLayout.NONE_BOX      # 0: no secondary layout
pto.SLayout.ROW_MAJOR     # 1: row-major secondary layout  
pto.SLayout.COL_MAJOR     # 2: column-major secondary layout

pto.PadValue.NULL         # 0: no padding
pto.PadValue.ZERO         # 1: zero padding
pto.PadValue.MAX          # 2: maximum value padding
pto.PadValue.MIN          # 3: minimum value padding
```

#### Tile Shape Concepts

- **Static Physical Shape**: The `shape` parameter represents the **static physical dimensions** of the tile allocated in memory. This must be a **compile-time constant** because tile memory allocation is fixed during compilation. The shape determines the total memory footprint and cannot change at runtime.

- **Valid Shape**: The `valid_shape` parameter represents the logical dimensions of actual data within the tile. It can be either **static** (compile-time constant) or **dynamic** (determined at runtime). It must be less than or equal to the physical `shape` in each dimension. When `valid_shape` is not specified, it defaults to the full `shape`.

- **Key Distinction**:
  - `shape`: **Static, compile-time** - Fixed tile allocation
  - `valid_shape`: **Static or Dynamic** - Actual data region (must be ≤ shape)

- **Constraints**:
  - `valid_shape[i] ≤ shape[i]` for each dimension i
  - `shape` must be compile-time constants
  - `valid_shape` can be compile-time constants or runtime values

- **Use Cases**:
  - Fixed-size tile buffers with variable data (e.g., batch processing with different input sizes)
  - Padding scenarios where physical allocation is larger than actual data
  - Partial tile utilization in tiled algorithms

- **Fractal Layout**: The `s_fractal_size` in tile configuration specifies the size of fractal blocks for secondary layout. This is used for optimized memory access patterns in matrix operations.

- **Padding Behavior**: The `pad_value` determines how out-of-bounds accesses are handled when reading beyond `valid_shape` but within `shape`. Padding values are used for accesses in the padded region (between valid_shape and shape).

> **⚠️ Important: Shape Constraints**
> 
> The tile `shape` must be **compile-time constants**. `valid_shape` can be compile-time constants or determined at runtime, but must satisfy `valid_shape[i] ≤ shape[i]` for all dimensions i.

### Tile Operations

#### Basic Access Operations

```python
# Get tile properties
shape = tile.shape                    # (256, 128)
elem_type = tile.element_type         # pto.f32
mem_space = tile.memory_space         # MemorySpace.UB
valid_shape = tile.valid_shape        # (240, 120) or same as shape

# Get configuration properties
config = tile.config
b_layout = config.b_layout            # pto.BLayout.ROW_MAJOR
s_layout = config.s_layout            # pto.SLayout.NONE_BOX
s_fractal = config.s_fractal_size     # pto.i32(16)
pad = config.pad_value                # pto.PadValue.ZERO

# Dynamic properties
rank = tile.rank                      # 2
num_elements = tile.num_elements      # 32768 (256 * 128)
valid_elements = tile.valid_elements  # 28800 (240 * 120)
```

#### Layout and Stride Queries

```python
# Get layout descriptors
layout_desc = tile.layout_descriptor  # Returns layout description object

# Get strides (in elements)
strides = tile.strides                # (128, 1) for row-major 256x128

# Get byte strides
byte_strides = tile.byte_strides      # (512, 4) for f32 row-major

# Get base offset (in bytes)
offset = tile.offset                  # pto.i64(0) or specified offset
```

#### Conversion Operations

Tiles support both explicit and implicit conversion to UBRef. When a tile is used in operations expecting a UBRef (e.g., `pto.vlds`, `pto.vsts`), it is automatically converted.

```python
# Convert to UBRef (implicit in vector operations)
ub_ref = tile.to_ubref()              # Explicit conversion
# or use tile as UBRef directly in vector ops
vec = pto.vlds(tile, offset)          # Implicit conversion

# Convert to typed pointer
ptr = tile.as_ptr()                   # Returns pto.ptr(pto.f32, MemorySpace.UB)

# Convert to MemRef (for compatibility)
memref = tile.to_memref()             # Returns pto.memref((256, 128), pto.f32, MemorySpace.UB)

# Extract slice of tile
slice_tile = tile.slice((0, 0), (64, 128))  # 64x128 slice from top-left corner

# Reshape tile (logical reshape, no data movement)
reshaped = tile.reshape((32768,))     # 1D reshape of 256x128 tile
```

#### Kernel Parameter Usage

```python
@pto.vkernel(target="a5", op="scale", dtypes=[(pto.AnyFloat, pto.AnyFloat)], priority=10)
def tiled_kernel(
    input_tile: pto.Tile,              # Tile parameter
    output_tile: pto.Tile,             # Another tile parameter
    scale: pto.f32
):
    # Convert tiles to UBRef for vector operations
    ub_in = input_tile.to_ubref()
    ub_out = output_tile.to_ubref()
    
    # Or use tiles directly (implicit conversion)
    all_mask = pto.make_mask(pto.f32, PAT.ALL)
    for i in range(0, 256, 64):
        # tile implicitly converts to UBRef in vlds with element-indexing syntax
        vec = pto.vlds(input_tile[i, 0:])        # Load from row i, columns 0 to vector_lanes-1
        scaled = pto.vmuls(vec, scale, all_mask)
        pto.vsts(scaled, output_tile[i, 0:], all_mask)  # Store to same position
```

#### Tile Creation from Existing Buffers

```python
# Create tile from existing pointer with shape
ptr = pto.castptr(0, pto.ptr(pto.f32, MemorySpace.UB))
tile = pto.tile_from_ptr(ptr, (256, 128), pto.f32)

# Create tile from memref
memref = pto.memref((256, 128), pto.f32, MemorySpace.UB)
tile = pto.tile_from_memref(memref)

# Create tile with explicit stride
tile = pto.tile_with_strides((256, 128), pto.f32, MemorySpace.UB, 
                             strides=(256, 1))  # Column-major strides
```

## Control Flow

### Vector Scopes

The TileLang DSL supports implicit vector scope inference, allowing developers to write vector operations directly without explicit `pto.vecscope()` blocks. The compiler automatically groups consecutive, data-dependent vector operations into implicit vector scopes during lowering.

#### Implicit Scope Inference

**Note:** The explicit `pto.vecscope()` construct is deprecated. Vector operations are automatically grouped into implicit scopes by the compiler's Scope Inference Pass.

When you write vector operations like `pto.vlds`, `pto.vadd`, `pto.vsts` directly in your code, the compiler's **Scope Inference Pass** analyzes the control flow graph and automatically creates vector scopes:

```python
# No explicit vecscope needed - compiler infers scope boundaries
vec = pto.vlds(outer_ptr, offset)
result = pto.vadd(vec, vec, all_mask)
pto.vsts(result, dst_ptr, offset, all_mask)
```

The compiler automatically groups these three operations into a single implicit vector scope because they form a data-dependent chain.

**Scope boundary rules:**
1. **Control flow boundaries**: Branches (`if`/`else`), loops (`for`/`while`), and function calls create implicit scope boundaries
2. **Scalar operations**: Non-vector operations (e.g., scalar arithmetic, pointer arithmetic) create boundaries
3. **Explicit strict_vecscope**: User-defined `strict_vecscope` blocks create hard boundaries

#### Explicit Scope Boundaries with `strict_vecscope`

For precise control over scope boundaries, use explicit `strict_vecscope` blocks. These create hard boundaries that prevent the compiler from merging operations across the block boundary:

```python
with pto.strict_vecscope(src_ptr, dst_ptr, start, end) as (s, d, lb, ub):
    # Operations inside this block are isolated from outside
    # Compiler will not merge operations across this boundary
    for i in range(lb, ub, 64):
        vec = pto.vlds(s, i)
        pto.vsts(vec, d, i, all_mask)
```

**Use cases for strict_vecscope:**
- Performance optimization: Isolate critical vector computation regions
- Debugging: Create explicit boundaries to isolate vector operations
- Resource management: Control vector register allocation boundaries
- Compatibility: Ensure deterministic scope placement for hardware constraints

### Loops

Counted loops use Python's `range` syntax:

```python
for i in range(lb, ub, step):
    # Loop body
    mask, rem = pto.make_mask(pto.f32, remaining)
    # ...
```

Loop-carried state is automatically handled through variable updates within the loop.

### Conditionals

`if` statements support value merging:

```python
flag: pto.i1 = some_condition
step: pto.i32 = 0

if flag:
    step = pto.i32(64)
else:
    step = pto.i32(128)

# 'step' here is the merged result from both branches
```

Variables defined in only one branch are local to that branch.

## Operations

The DSL provides operations grouped by functionality. All operations use the `pto.` prefix. Operations are organized by functional families following the VPTO instruction set architecture.

### Pointer Construction

Operations for creating and manipulating typed pointers.

#### `pto.castptr(offset: pto.i64, ptr_type: Type) -> PtrType`

**Description**: Creates a pointer with the specified offset and type.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `offset` | `pto.i64` | Byte offset from base address |
| `ptr_type` | `Type` | Target pointer type (e.g., `pto.ptr(pto.f32, MemorySpace.GM)`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `ptr` | `PtrType` | Typed pointer value |

**Example**:
```python
ub_ptr = pto.castptr(0, pto.ptr(pto.f32, MemorySpace.UB))
```

#### `pto.addptr(ptr: PtrType, offset: pto.i64) -> PtrType`

**Description**: Adds an offset to an existing pointer.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `ptr` | `PtrType` | Source pointer |
| `offset` | `pto.i64` | Byte offset to add |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `new_ptr` | `PtrType` | Pointer with offset applied |

**Example**:
```python
next_ptr = pto.addptr(ub_ptr, 4096)
```

### Synchronization & Buffer Control

Operations for pipeline synchronization and buffer management.

#### `pto.set_flag(pipe_from: PIPE, pipe_to: PIPE, event: EVENT) -> None`

**Description**: Sets a synchronization flag between hardware pipelines.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe_from` | `PIPE` | Source pipeline (e.g., `PIPE.MTE2`) |
| `pipe_to` | `PIPE` | Destination pipeline (e.g., `PIPE.V`) |
| `event` | `EVENT` | Event identifier (e.g., `EVENT.ID0`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import PIPE, EVENT

pto.set_flag(PIPE.MTE2, PIPE.V, EVENT.ID0)
```

#### `pto.wait_flag(pipe_from: PIPE, pipe_to: PIPE, event: EVENT) -> None`

**Description**: Waits for a synchronization flag between hardware pipelines.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe_from` | `PIPE` | Source pipeline (e.g., `PIPE.MTE2`) |
| `pipe_to` | `PIPE` | Destination pipeline (e.g., `PIPE.V`) |
| `event` | `EVENT` | Event identifier (e.g., `EVENT.ID0`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import PIPE, EVENT

pto.wait_flag(PIPE.MTE2, PIPE.V, EVENT.ID0)
```

#### `pto.pipe_barrier(pipes: PIPE) -> None`

**Description**: Executes a barrier across specified pipelines.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pipes` | `PIPE` | Pipeline specification (e.g., `PIPE.ALL`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import PIPE

pto.pipe_barrier(PIPE.ALL)
```

#### `pto.get_buf(op_type: SyncOpType, buf_id: pto.i32, mode: pto.i32 = 0) -> None`

**Description**: Acquires a buffer for producer-consumer synchronization.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `op_type` | `SyncOpType` | Operation type (e.g., `SyncOpType.TLOAD`) |
| `buf_id` | `pto.i32` | Buffer identifier |
| `mode` | `pto.i32` | Acquisition mode (default: 0) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import SyncOpType

# Acquire buffer for DMA load operation
pto.get_buf(SyncOpType.TLOAD, 0)
```

#### `pto.rls_buf(op_type: SyncOpType, buf_id: pto.i32, mode: pto.i32 = 0) -> None`

**Description**: Releases a previously acquired buffer.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `op_type` | `SyncOpType` | Operation type (e.g., `SyncOpType.TLOAD`) |
| `buf_id` | `pto.i32` | Buffer identifier |
| `mode` | `pto.i32` | Release mode (default: 0) |

**Returns**: None (side-effect操作)

**Example**:
```python
from pto import SyncOpType

# Release buffer for DMA load operation
pto.rls_buf(SyncOpType.TLOAD, 0)
```

### Low-level DMA Programming (Legacy)

**Note**: These low-level DMA programming operations are automatically handled by `pto.dma_load` and `pto.dma_store` in most cases. They expose hardware DMA engine parameters directly and should only be used when the automatic inference provided by the high-level API is insufficient for specific optimization needs.

This section contains both DMA configuration operations (setting loop strides and sizes) and DMA execution operations (copying data). Prefer the high-level `pto.dma_load` and `pto.dma_store` operations which automatically infer all parameters from TensorView slices and Tile properties.

#### When to Use Low-level DMA Programming

Consider using these low-level operations only in the following scenarios:

1. **Performance micro-optimization**: When specific DMA parameter tuning is required for performance-critical code
2. **Non-standard access patterns**: When TensorView slicing syntax cannot express the desired memory access pattern
3. **Hardware-specific optimizations**: When targeting specific DMA engine characteristics not captured by the high-level API

For 99% of use cases, `pto.dma_load` and `pto.dma_store` with TensorView slicing provide sufficient control and are much easier to use correctly.

#### Manual Configuration Example

```python
# Manual DMA configuration (discouraged for normal use)
pto.set_loop2_stride_outtoub(32, 128)    # Outer loop strides
pto.set_loop1_stride_outtoub(1, 32)      # Inner loop strides  
pto.set_loop_size_outtoub(16, 16)        # Transfer size
pto.copy_gm_to_ubuf(gm_ptr, ub_ptr, ...)

# Equivalent using high-level API (recommended)
pto.dma_load(input_tensor[0:16, 0:16], ub_tile)
# All loop strides and sizes automatically inferred
```

#### `pto.set_loop2_stride_outtoub(stride0: pto.i64, stride1: pto.i64) -> None`

**Description**: Configures DMA stride parameters for GM → UB transfers (loop2).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `stride0` | `pto.i64` | First dimension stride |
| `stride1` | `pto.i64` | Second dimension stride |

**Returns**: None (side-effect operation)

#### `pto.set_loop1_stride_outtoub(stride0: pto.i64, stride1: pto.i64) -> None`

**Description**: Configures DMA stride parameters for GM → UB transfers (loop1).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `stride0` | `pto.i64` | First dimension stride |
| `stride1` | `pto.i64` | Second dimension stride |

**Returns**: None (side-effect operation)

#### `pto.set_loop_size_outtoub(size0: pto.i64, size1: pto.i64) -> None`

**Description**: Configures DMA transfer size for GM → UB transfers.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `size0` | `pto.i64` | First dimension size |
| `size1` | `pto.i64` | Second dimension size |

**Returns**: None (side-effect operation)

**Example**:
```python
pto.set_loop_size_outtoub(1, 1)
```

#### `pto.set_loop2_stride_ubtoout(stride0: pto.i64, stride1: pto.i64) -> None`

**Description**: Configures DMA stride parameters for UB → GM transfers (loop2).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `stride0` | `pto.i64` | First dimension stride |
| `stride1` | `pto.i64` | Second dimension stride |

**Returns**: None (side-effect operation)

#### `pto.set_loop1_stride_ubtoout(stride0: pto.i64, stride1: pto.i64) -> None`

**Description**: Configures DMA stride parameters for UB → GM transfers (loop1).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `stride0` | `pto.i64` | First dimension stride |
| `stride1` | `pto.i64` | Second dimension stride |

**Returns**: None (side-effect operation)

#### `pto.set_loop_size_ubtoout(size0: pto.i64, size1: pto.i64) -> None`

**Description**: Configures DMA transfer size for UB → GM transfers.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `size0` | `pto.i64` | First dimension size |
| `size1` | `pto.i64` | Second dimension size |

**Returns**: None (side-effect operation)

#### DMA Execution Operations

**Note**: These operations execute DMA transfers but require manual configuration of DMA parameters (loop strides, loop sizes) using the `set_loop*_stride_*` and `set_loop_size_*` operations described above. The high-level `pto.dma_load` and `pto.dma_store` operations automatically handle both configuration and execution.

The following operations provide direct control over DMA transfers but require manual stride and size configuration. Prefer the high-level Tile Data Movement operations for most use cases.

#### `pto.copy_gm_to_ubuf(src: GMPtr, dst: UBPtr, src_offset: pto.i64, src_stride0: pto.i64, src_stride1: pto.i64, dst_offset: pto.i64, dst_stride0: pto.i64, transpose: pto.i1, pad_left: pto.i64, pad_right: pto.i64, pad_value: pto.i64) -> None`

**Description**: Copies data from Global Memory (GM) to Unified Buffer (UB).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `GMPtr` | Source GM pointer |
| `dst` | `UBPtr` | Destination UB pointer |
| `src_offset` | `pto.i64` | Source offset |
| `src_stride0` | `pto.i64` | Source stride dimension 0 |
| `src_stride1` | `pto.i64` | Source stride dimension 1 |
| `dst_offset` | `pto.i64` | Destination offset |
| `dst_stride0` | `pto.i64` | Destination stride dimension 0 |
| `transpose` | `pto.i1` | Transpose flag |
| `pad_left` | `pto.i64` | Left padding size |
| `pad_right` | `pto.i64` | Right padding size |
| `pad_value` | `pto.i64` | Padding value |

**Returns**: None (side-effect operation)

**Example**:
```python
pto.copy_gm_to_ubuf(gm_ptr, ub_ptr, 0, 32, 128, 0, 0, False, 0, 128, 128)
```

#### `pto.copy_ubuf_to_ubuf(src: UBPtr, dst: UBPtr, src_offset: pto.i64, src_stride0: pto.i64, src_stride1: pto.i64, dst_offset: pto.i64, dst_stride0: pto.i64, dst_stride1: pto.i64) -> None`

**Description**: Copies data within Unified Buffer (UB → UB).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `UBPtr` | Source UB pointer |
| `dst` | `UBPtr` | Destination UB pointer |
| `src_offset` | `pto.i64` | Source offset |
| `src_stride0` | `pto.i64` | Source stride dimension 0 |
| `src_stride1` | `pto.i64` | Source stride dimension 1 |
| `dst_offset` | `pto.i64` | Destination offset |
| `dst_stride0` | `pto.i64` | Destination stride dimension 0 |
| `dst_stride1` | `pto.i64` | Destination stride dimension 1 |

**Returns**: None (side-effect operation)

#### `pto.copy_ubuf_to_gm(src: UBPtr, dst: GMPtr, src_offset: pto.i64, src_stride0: pto.i64, src_stride1: pto.i64, dst_offset: pto.i64, dst_stride0: pto.i64, dst_stride1: pto.i64) -> None`

**Description**: Copies data from Unified Buffer (UB) to Global Memory (GM).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `UBPtr` | Source UB pointer |
| `dst` | `GMPtr` | Destination GM pointer |
| `src_offset` | `pto.i64` | Source offset |
| `src_stride0` | `pto.i64` | Source stride dimension 0 |
| `src_stride1` | `pto.i64` | Source stride dimension 1 |
| `dst_offset` | `pto.i64` | Destination offset |
| `dst_stride0` | `pto.i64` | Destination stride dimension 0 |
| `dst_stride1` | `pto.i64` | Destination stride dimension 1 |

**Returns**: None (side-effect operation)

**Example**:
```python
pto.copy_ubuf_to_gm(ub_ptr, gm_ptr, 0, 32, 128, 0, 128, 128)
```

### Tile Data Movement Operations

High-level operations for moving data between TensorView partitions (GM) and Tile buffers (UB), as well as between Tile buffers. These operations **automatically handle all low-level DMA configuration** and provide an intuitive interface based on tile semantics.

#### Automatic DMA Parameter Inference

The `pto.dma_load` and `pto.dma_store` operations automatically infer DMA transfer parameters (loop strides, loop sizes) from:

1. **TensorView slices** - Python slicing syntax captures stride information:
   ```python
   # Contiguous slice: [0:16, 0:16]
   pto.dma_load(input_tensor[0:16, 0:16], ub_tile)
   
   # Strided slice: [0:64:2, 0:32] → stride=2 in first dimension
   pto.dma_load(input_tensor[0:64:2, 0:32], ub_tile)
   ```

2. **Tile properties** - Layout and memory space determine destination patterns:
   ```python
   # Row-major vs column-major layouts affect stride computation
   row_major_tile = pto.tile((16, 16), pto.f32, pto.MemorySpace.UB, b_layout=pto.BLayout.ROW_MAJOR)
   col_major_tile = pto.tile((16, 16), pto.f32, pto.MemorySpace.UB, b_layout=pto.BLayout.COL_MAJOR)
   ```

3. **Transpose and padding requirements** - Specified via operation parameters.

#### Benefits of Automatic Inference

- **Simplified API**: No need to manually call `set_loop*_stride_*` and `set_loop_size_*` operations
- **Reduced errors**: Automatic parameter validation and consistency checking
- **Hardware abstraction**: Focus on data movement semantics, not DMA engine details
- **Portable code**: Same TileLang code works across different DMA implementations

For advanced use cases requiring manual DMA parameter control, see the [Low-level DMA Programming (Legacy)](#low-level-dma-programming-legacy) section.

#### `pto.dma_load(src: TensorView, dst: Tile, pad_mode: PadMode = PadMode.PadNull, pad_value: ScalarType = None, left_padding: Index = 0, right_padding: Index = 0, init_out_buffer: bool = False) -> None`

**Description**: Loads data from a TensorView partition (GM) into a Tile buffer (UB). This maps to `pto.copy_gm_to_ubuf` operation in VPTO IR.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `TensorView` | Source tensor view partition (must be in GM) |
| `dst` | `Tile` | Destination tile buffer (must be in UB memory space) |
| `pad_mode` | `PadMode` | Padding mode (PadNull, PadFirstElem, PadValue) |
| `pad_value` | `ScalarType` | Padding value (required if `pad_mode == PadValue`) |
| `left_padding` | `Index` | Left padding element count |
| `right_padding` | `Index` | Right padding element count |
| `init_out_buffer` | `bool` | Initialize output buffer before loading |

**Returns**: None (side-effect operation)

**Constraints**:
- Destination tile must have `memory_space = MemorySpace.UB`
- Element types of source and destination must have same bitwidth
- Source partition shape must match destination tile valid shape (after accounting for padding)

**Example**:
```python
# Load a 16x16 partition into a UB tile
pto.dma_load(input_tensor[0:16, 0:16], ub_tile)

# Load with zero padding
pto.dma_load(input_tensor[0:16, 0:16], ub_tile, 
          pad_mode=PadMode.PadValue, 
          pad_value=pto.f32(0.0),
          left_padding=2, 
          right_padding=2)
```

#### `pto.dma_store(src: Tile, dst: TensorView, pad_mode: PadMode = PadMode.PadNull, pad_value: ScalarType = None, left_padding: Index = 0, right_padding: Index = 0) -> None`

**Description**: Stores data from a Tile buffer (UB) to a TensorView partition (GM). This maps to `pto.copy_ubuf_to_gm` operation in VPTO IR.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `Tile` | Source tile buffer (must be in UB memory space) |
| `dst` | `TensorView` | Destination tensor view partition (must be in GM) |
| `pad_mode` | `PadMode` | Padding mode (PadNull, PadFirstElem, PadValue) |
| `pad_value` | `ScalarType` | Padding value (required if `pad_mode == PadValue`) |
| `left_padding` | `Index` | Left padding element count |
| `right_padding` | `Index` | Right padding element count |

**Returns**: None (side-effect operation)

**Constraints**:
- Source tile must have `memory_space = MemorySpace.UB`
- Element types of source and destination must have same bitwidth
- Source tile valid shape must match destination partition shape (after accounting for padding)

**Example**:
```python
# Store a UB tile to a GM partition
pto.dma_store(ub_tile, output_tensor[0:16, 0:16])

# Store with padding
pto.dma_store(ub_tile, output_tensor[0:16, 0:16],
           pad_mode=PadMode.PadValue,
           pad_value=pto.f32(0.0),
           left_padding=1,
           right_padding=1)
```

#### `pto.dma_copy(src: Tile, dst: Tile, src_offset: tuple[Index, Index] = (0, 0), dst_offset: tuple[Index, Index] = (0, 0), copy_shape: tuple[Index, Index] = None) -> None`

**Description**: Copies data between Tile buffers within Unified Buffer (UB → UB). This maps to `pto.copy_ubuf_to_ubuf` operation in VPTO IR.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `Tile` | Source tile buffer (must be in UB memory space) |
| `dst` | `Tile` | Destination tile buffer (must be in UB memory space) |
| `src_offset` | `tuple[Index, Index]` | Offset within source tile (row, col) in elements |
| `dst_offset` | `tuple[Index, Index]` | Offset within destination tile (row, col) in elements |
| `copy_shape` | `tuple[Index, Index]` | Shape of region to copy (rows, cols) in elements. If None, copies the maximum valid region starting from offsets. |

**Returns**: None (side-effect operation)

**Constraints**:
- Both tiles must have `memory_space = MemorySpace.UB`
- Element types of source and destination must match
- Source and destination regions must be within tile valid shapes

**Example**:
```python
# Copy entire tile
pto.dma_copy(src_tile, dst_tile)

# Copy subregion: copy 8x8 block from (2,2) in src to (0,0) in dst
pto.dma_copy(src_tile, dst_tile, 
              src_offset=(2, 2), 
              dst_offset=(0, 0), 
              copy_shape=(8, 8))
```

**Note**: These high-level operations automatically handle DMA stride and size configuration based on tile shapes, layouts, and offsets. For low-level control, see the [Low-level DMA Programming (Legacy)](#low-level-dma-programming-legacy) section.

#### VPTO IR Mapping

The high-level DMA operations in TileLang DSL map to corresponding operations in VPTO IR:

| TileLang DSL Operation | VPTO IR Operation | Description |
|------------------------|-------------------|-------------|
| `pto.dma_load` | `pto.copy_gm_to_ubuf` | Loads data from GM tensor view to UB tile buffer |
| `pto.dma_store` | `pto.copy_ubuf_to_gm` | Stores data from UB tile buffer to GM tensor view |
| `pto.dma_copy` | `pto.copy_ubuf_to_ubuf` | Copies data between UB tile buffers |

These mappings allow the TileLang compiler to generate efficient VPTO IR code while providing a higher-level, more intuitive API for developers. The compiler automatically handles the conversion between Tile/TensorView abstractions and the low-level pointer/stride representation required by VPTO IR operations.


### Address Generation Syntax Sugar

To simplify address calculation and reduce manual byte offset computation errors, TileLang DSL provides syntactic sugar for vector load/store operations using element-based indexing. This syntax automatically computes the byte offset based on tile shape, element type, and layout.

#### Indexing Syntax

The syntax supports two indexing modes for different operations:

1. **Vector-range indexing** (for vector load/store operations):
   - **Row-major layout (default)**: `tile[row_index, col_start:]`
     - `row_index`: Row index (0-based)
     - `col_start:`: Starting column index followed by colon, indicating a vector-width contiguous region starting from this column
     - The colon (`:`) indicates an implicit vector-width range determined by hardware vector size (256 bytes) and element type
   
   - **Column-major layout**: `tile[row_start:, col_index]`
     - `row_start:`: Starting row index followed by colon, indicating a vector-width contiguous region starting from this row
     - `col_index`: Column index (0-based)
     - Used for column-major tiles (`BLayout.COL_MAJOR`) where elements are stored column-wise
   
   - **1D tile indexing**: `tile[start:]` (or equivalently `tile[0, start:]` for row-major or `tile[start:, 0]` for column-major)
     - `start:`: Starting element index followed by colon

2. **Single-element indexing** (for scalar load operations like `pto.vsld`):
   - **Row-major layout (default)**: `tile[row_index, col_index]`
     - `row_index`: Row index (0-based)
     - `col_index`: Column index (0-based)
     - Loads a single element at the specified position and broadcasts it to all vector lanes
   
   - **Column-major layout**: `tile[row_index, col_index]` (same syntax)
     - `row_index`: Row index (0-based)
     - `col_index`: Column index (0-based)
     - Same syntax as row-major; the layout determines how the offset is computed
   
   - **1D tile indexing**: `tile[pos]`
     - `pos`: Element index (0-based)
     - Loads a single element at the specified position and broadcasts it to all vector lanes

#### Vector Width Calculation

The number of elements loaded/stored in a single vector operation is determined by:

```
vector_lanes = 256 // element_size_bytes(element_type)
```

**Convenience API**: Use `pto.get_lanes(dtype)` to compute vector lanes for a given element type (e.g., `pto.get_lanes(pto.f32)` returns 64, `pto.get_lanes(pto.f16)` returns 128).

Where `element_size_bytes` is:
- 1 byte for `i8`
- 2 bytes for `i16`, `f16`, `bf16`
- 4 bytes for `i32`, `f32`
- 8 bytes for `i64`

#### Offset Computation

The byte offset is automatically computed based on tile layout:

- **Row-major layout** (`BLayout.ROW_MAJOR`):
  ```
  offset = (row_index * stride_row + col_start) * element_size_bytes
  ```
  where `stride_row` is the row stride in elements (typically `tile.shape[1]` for contiguous tiles).

- **Column-major layout** (`BLayout.COL_MAJOR`):
  - For syntax `tile[row_start:, col_index]`:
    ```
    offset = (col_index * stride_col + row_start) * element_size_bytes
    ```
  - For backward compatibility with traditional offset calculation:
    ```
    offset = (col_start * stride_col + row_index) * element_size_bytes
    ```
  where `stride_col` is the column stride in elements (typically `tile.shape[0]` for contiguous tiles), `row_start` is the starting row index, and `col_index` is the column index.

**Note**: 
- For single-element indexing (`tile[row, col]` or `tile[pos]`), the same offset formulas apply with `col_start` replaced by `col_index` (or `start` replaced by `pos` for 1D tiles).
- For column-major vector-range indexing (`tile[row_start:, col_index]`), the offset formula uses `row_start` as the starting position along the contiguous dimension.
- The compiler automatically handles the appropriate substitution based on the indexing syntax and tile layout.

#### Constraints

1. **Boundary checks**: The requested region must be within tile bounds:
   - **For vector-range indexing** (`:` syntax):
     - **Row-major layout** (`tile[row_index, col_start:]`):
       - `row_index < tile.shape[0]` and `col_start + vector_lanes <= tile.shape[1]`
     - **Column-major layout** (`tile[row_start:, col_index]`):
       - `row_start + vector_lanes <= tile.shape[0]` and `col_index < tile.shape[1]`
     - **1D tile indexing**: `tile[start:]`
       - `start + vector_lanes <= tile.shape[0]` (or `tile.shape[1]` for 1D tiles)
   - **For single-element indexing** (no `:` syntax):
     - 2D: `row_index < tile.shape[0]` and `col_index < tile.shape[1]` (same for both layouts)
     - 1D: `pos < tile.shape[0]` (or `tile.shape[1]` for 1D tiles)

2. **Alignment**: The computed offset must satisfy hardware alignment requirements for the operation.

3. **Full vectors only**: The `:` syntax always loads/stores a full vector width. For partial vectors, use the traditional byte offset approach with explicit mask handling.

4. **Single-element operations**: The single-element indexing syntax (`tile[row, col]` or `tile[pos]`) is only supported for scalar load operations like `pto.vsld`. For other operations, use vector-range indexing with `:` syntax.

#### Supported Operations

The indexing syntax is supported for all vector load and store operations with the following syntax mapping:

- **Vector-range indexing** (`tile[row, col:]` or `tile[start:]`):
  - Load operations: `vlds`, `vldas`, `vldus`, `vplds`, `vldx2`
  - Store operations: `vsts`, `vsta`, `psts`, `vsst`, `vstx2`

- **Single-element indexing** (`tile[row, col]` or `tile[pos]`):
  - Load operations: `vsld` (scalar load with broadcast)

#### Examples

The following examples use row-major layout syntax. For column-major tiles, use `tile[row_start:, col_index]` syntax instead of `tile[row_index, col_start:]`.

```python
# 2D tile indexing (row-major layout)
vec = pto.vlds(tile[i, j:])          # Load vector from row i, columns j to j+vector_lanes-1
pto.vsts(vec, tile[i, j:], mask)     # Store vector with mask

# 1D tile indexing  
vec = pto.vlds(tile[k:])             # Load vector from elements k to k+vector_lanes-1
pto.vsts(vec, tile[k:], mask)        # Store vector with mask

# Dual load with indexing
vec1, vec2 = pto.vldx2(tile_a[i, j:], tile_b[i, j:])

# Aligned load with indexing
vec = pto.vldas(tile[i, j:], align)

# Scalar load (broadcast)
vec = pto.vsld(tile[i, j])          # Load scalar at tile[i,j] and broadcast to vector
```

#### Comparison with Manual Offset Calculation

**Traditional approach (error-prone):**
```python
# Manual byte offset calculation for f32 tile
rows, cols = tile.shape
row_offset = i * cols * 4  # Hard-coded 4 bytes for f32
col_offset = j * 4
offset = row_offset + col_offset
vec = pto.vlds(tile, offset)
```

**New syntax (type-safe):**
```python
# Automatic offset calculation
vec = pto.vlds(tile[i, j:])  # Compiler computes correct offset for any element type
```

The syntax sugar eliminates manual byte calculations, reduces errors, and makes code generic across different element types (e.g., the same kernel works for both `f16` and `f32` without modification).

### Vector Load Operations

Operations for loading data from memory into vector registers.

#### `pto.vlds(buf: UBRef, offset: Index) -> VRegType`  
#### `pto.vlds(tile[row, col:]) -> VRegType`
#### `pto.vlds(tile[start:]) -> VRegType`

**Description**: Stateless vector load from buffer. Supports both traditional byte-offset syntax and new element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `UBRef` | Buffer or pointer (UB memory space) |
| `offset` | `Index` | Byte offset |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column (vector-width range) |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index (vector-width range) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Loaded vector register |

**Constraints**:
- Buffer must be in UB memory space
- For byte-offset syntax: offset must be properly aligned based on element type
- For element-indexing syntax: the requested vector region must be within tile bounds and satisfy alignment requirements

**Examples**:
```python
# Traditional byte-offset syntax
vec = pto.vlds(ub_ptr, lane * 256)

# New element-indexing syntax
vec = pto.vlds(tile[i, j:])      # Load from row i, columns j to j+vector_lanes-1
vec = pto.vlds(tile[k:])         # Load from 1D tile, elements k to k+vector_lanes-1

# Generic kernel that works for both f16 and f32
@pto.vkernel(target="a5", op="scale", dtypes=[(pto.AnyFloat, pto.AnyFloat)], priority=10)
def generic_scale(src: pto.Tile, dst: pto.Tile, scale: pto.f32):
    rows, cols = src.shape
    all_mask = pto.make_mask(src.element_type, PAT.ALL)
    for i in range(0, rows):
        for j in range(0, cols, vector_lanes):  # vector_lanes computed from element type
            # No manual byte calculation needed!
            vec = pto.vlds(src[i, j:])
            scaled = pto.vmuls(vec, scale, all_mask)
            pto.vsts(scaled, dst[i, j:], all_mask)
```

#### `pto.vldas(buf: UBRef, offset: Index, align: pto.align) -> VRegType`  
#### `pto.vldas(tile[row, col:], align: pto.align) -> VRegType`  
#### `pto.vldas(tile[start:], align: pto.align) -> VRegType`

**Description**: Aligned vector load with explicit alignment carrier. Supports both byte-offset and element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `UBRef` | Buffer or pointer (UB memory space) |
| `offset` | `Index` | Byte offset |
| `align` | `pto.align` | Alignment specification |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index |
| `align` | `pto.align` | Alignment specification |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Loaded vector register |

**Examples**:
```python
# Byte-offset syntax
vec = pto.vldas(ub_ptr, offset, align)

# Element-indexing syntax
vec = pto.vldas(tile[i, j:], align)
vec = pto.vldas(tile[k:], align)
```

#### `pto.vldus(buf: UBRef, offset: Index) -> VRegType`  
#### `pto.vldus(tile[row, col:]) -> VRegType`  
#### `pto.vldus(tile[start:]) -> VRegType`

**Description**: Unaligned vector load. Supports both byte-offset and element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `UBRef` | Buffer or pointer (UB memory space) |
| `offset` | `Index` | Byte offset |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Loaded vector register |

**Examples**:
```python
# Byte-offset syntax
vec = pto.vldus(ub_ptr, offset)

# Element-indexing syntax
vec = pto.vldus(tile[i, j:])
vec = pto.vldus(tile[k:])
```

#### `pto.vplds(buf: UBRef, offset: Index, pred: MaskType) -> VRegType`  
#### `pto.vplds(tile[row, col:], pred: MaskType) -> VRegType`  
#### `pto.vplds(tile[start:], pred: MaskType) -> VRegType`

**Description**: Predicated vector load stateless. Supports both byte-offset and element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `UBRef` | Buffer or pointer (UB memory space) |
| `offset` | `Index` | Byte offset |
| `pred` | `MaskType` | Predicate mask |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index |
| `pred` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Loaded vector register |

**Examples**:
```python
# Byte-offset syntax
vec = pto.vplds(ub_ptr, offset, mask)

# Element-indexing syntax
vec = pto.vplds(tile[i, j:], mask)
vec = pto.vplds(tile[k:], mask)
```

#### `pto.vldx2(buf1: UBRef, buf2: UBRef, offset: Index) -> (VRegType, VRegType)`  
#### `pto.vldx2(tile1[row, col:], tile2[row, col:]) -> (VRegType, VRegType)`  
#### `pto.vldx2(tile1[start:], tile2[start:]) -> (VRegType, VRegType)`

**Description**: Dual vector load from two buffers. Supports both byte-offset and element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf1` | `UBRef` | First buffer or pointer |
| `buf2` | `UBRef` | Second buffer or pointer |
| `offset` | `Index` | Byte offset (applied to both buffers) |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `tile1[row, col:]` | `Tile` with indexing | First 2D tile with row index and starting column |
| `tile2[row, col:]` | `Tile` with indexing | Second 2D tile with row index and starting column |
| _or_ | | |
| `tile1[start:]` | `Tile` with indexing | First 1D tile with starting element index |
| `tile2[start:]` | `Tile` with indexing | Second 1D tile with starting element index |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec1` | `VRegType` | Vector from first buffer |
| `vec2` | `VRegType` | Vector from second buffer |

**Examples**:
```python
# Byte-offset syntax
vec1, vec2 = pto.vldx2(ub_ptr1, ub_ptr2, offset)

# Element-indexing syntax
vec1, vec2 = pto.vldx2(tile_a[i, j:], tile_b[i, j:])
vec1, vec2 = pto.vldx2(tile_a[k:], tile_b[k:])
```

#### `pto.vsld(buf: UBRef, offset: Index) -> VRegType`  
#### `pto.vsld(tile[row, col]) -> VRegType`  
#### `pto.vsld(tile[pos]) -> VRegType`

**Description**: Scalar load to vector (broadcast scalar to all lanes). Supports both byte-offset and element-indexing syntax. The element-indexing syntax loads a single element (not a vector) and broadcasts it to all lanes.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `UBRef` | Buffer or pointer (UB memory space) |
| `offset` | `Index` | Byte offset |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col]` | `Tile` with indexing | 2D tile with row and column indices (single element) |
| `tile[pos]` | `Tile` with indexing | 1D tile with element index (single element) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Vector with scalar broadcast to all lanes |

**Examples**:
```python
# Byte-offset syntax
vec = pto.vsld(ub_ptr, offset)

# Element-indexing syntax
vec = pto.vsld(tile[i, j])    # Load single element at (i,j) and broadcast
vec = pto.vsld(tile[k])       # Load single element at position k and broadcast
```

### Predicate Operations

Operations for creating and manipulating typed masks.

**Recommended API**: For most use cases, prefer the unified `pto.make_mask()` function which automatically selects the appropriate mask granularity based on element type and supports both tail processing (remaining element count) and pattern-based mask generation. This eliminates the need to manually choose between `plt_b8`/`plt_b16`/`plt_b32` (tail processing) and `pset_b8`/`pset_b16`/`pset_b32` (pattern generation) operations.

**Pattern alias**: For brevity in examples, the documentation uses `PAT` as an alias for `pto.MaskPattern` (e.g., `PAT.ALL` instead of `pto.MaskPattern.PAT_ALL`). In practice, you can create this alias with `from pto import MaskPattern as PAT` or `PAT = pto.MaskPattern`.

#### `pto.pset_b8(pattern: pto.MaskPattern) -> pto.mask_b8`

**Description**: Creates an 8-bit granularity mask from a pattern.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pattern` | `pto.MaskPattern` | Mask pattern enum (e.g., `pto.MaskPattern.PAT_ALL`, `pto.MaskPattern.PAT_EVEN`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b8` | 8-bit granularity mask |

**Constraints**:
- Used with `i8` vector operations

**Example**:
```python
mask8 = pto.make_mask(pto.i8, PAT.ALL)
```

#### `pto.pset_b16(pattern: pto.MaskPattern) -> pto.mask_b16`

**Description**: Creates a 16-bit granularity mask from a pattern.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pattern` | `pto.MaskPattern` | Mask pattern enum (e.g., `pto.MaskPattern.PAT_ALL`, `pto.MaskPattern.PAT_EVEN`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b16` | 16-bit granularity mask |

**Constraints**:
- Used with `f16`/`bf16`/`i16` vector operations

**Example**:
```python
mask16 = pto.make_mask(pto.f16, PAT.ALL)
```

#### `pto.pset_b32(pattern: pto.MaskPattern) -> pto.mask_b32`

**Description**: Creates a 32-bit granularity mask from a pattern.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pattern` | `pto.MaskPattern` | Mask pattern enum (e.g., `pto.MaskPattern.PAT_ALL`, `pto.MaskPattern.PAT_EVEN`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b32` | 32-bit granularity mask |

**Constraints**:
- Used with `f32`/`i32` vector operations

**Example**:
```python
mask32 = pto.make_mask(pto.f32, PAT.ALL)
```

#### `pto.pge_b8(vec: VRegType, scalar: ScalarType) -> pto.mask_b8`

**Description**: Creates 8-bit mask where vector elements ≥ scalar.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector (element type must match mask granularity) |
| `scalar` | `ScalarType` | Scalar comparison value |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b8` | 8-bit granularity mask |

**Constraints**:
- Vector element type must be `i8` or compatible

#### `pto.pge_b16(vec: VRegType, scalar: ScalarType) -> pto.mask_b16`

**Description**: Creates 16-bit mask where vector elements ≥ scalar.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector (element type must match mask granularity) |
| `scalar` | `ScalarType` | Scalar comparison value |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b16` | 16-bit granularity mask |

**Constraints**:
- Vector element type must be `f16`/`bf16`/`i16`

#### `pto.pge_b32(vec: VRegType, scalar: ScalarType) -> pto.mask_b32`

**Description**: Creates 32-bit mask where vector elements ≥ scalar.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector (element type must match mask granularity) |
| `scalar` | `ScalarType` | Scalar comparison value |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b32` | 32-bit granularity mask |

**Constraints**:
- Vector element type must be `f32`/`i32`

**Example**:
```python
mask = pto.pge_b32(vec_f32, pto.f32(0.0))
```

#### `pto.plt_b8(vec: VRegType, scalar: ScalarType) -> pto.mask_b8`

**Description**: Creates 8-bit mask where vector elements < scalar.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector (element type must match mask granularity) |
| `scalar` | `ScalarType` | Scalar comparison value |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b8` | 8-bit granularity mask |

#### `pto.plt_b16(vec: VRegType, scalar: ScalarType) -> pto.mask_b16`

**Description**: Creates 16-bit mask where vector elements < scalar.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector (element type must match mask granularity) |
| `scalar` | `ScalarType` | Scalar comparison value |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b16` | 16-bit granularity mask |

#### `pto.plt_b32(vec: VRegType, scalar: ScalarType) -> (pto.mask_b32, pto.i32)`

**Description**: Creates 32-bit mask where vector elements < scalar, returns mask and remaining count.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector (element type must match mask granularity) |
| `scalar` | `ScalarType` | Scalar comparison value |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b32` | 32-bit granularity mask |
| `remaining` | `pto.i32` | Remaining element count |

**Example**:
```python
mask, remaining = pto.plt_b32(vec_f32, pto.f32(10.0))
```

#### `pto.make_mask(element_type: Type, value: pto.i32 | pto.MaskPattern) -> MaskType | (MaskType, pto.i32)`

**Description**: Creates a mask with appropriate bitwidth (8, 16, or 32) based on element type, automatically inferring whether to perform tail processing or pattern-based mask generation based on the `value` parameter type. This convenience function eliminates the need to manually choose between `plt_b8`/`plt_b16`/`plt_b32` and `pset_b8`/`pset_b16`/`pset_b32` operations.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `element_type` | `Type` | Element type (e.g., `pto.f32`, `pto.f16`, `pto.i8`) |
| `value` | `pto.i32` \| `pto.MaskPattern` | Either: <br>- Remaining element count (as `pto.i32`) for tail processing <br>- Mask pattern enum value for fixed mask generation (e.g., `pto.MaskPattern.PAT_ALL`, `pto.MaskPattern.PAT_VL32`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `MaskType` | Generated mask with appropriate granularity |
| `remaining` | `pto.i32` | Updated remaining element count (only returned when `value` is a `pto.i32` for tail processing) |

**Constraints**:
- The `element_type` must be one of: `f32`, `i32`, `f16`, `bf16`, `i16`, `i8`
- The returned mask granularity matches the element type: 32-bit for `f32`/`i32`, 16-bit for `f16`/`bf16`/`i16`, 8-bit for `i8`
- The function infers the operation mode from the `value` parameter type at compile time:
  - `pto.i32` value → tail processing mode (returns `(mask, updated_remaining)`)
  - `pto.MaskPattern` enum value → pattern mode (returns `mask` only)

**Implementation Note**: This function is a DSL macro that performs type-based dispatch at compile time:
- When `value` is a `pto.i32` expression: expands to corresponding `plt_b` instruction (`plt_b32`, `plt_b16`, or `plt_b8`)
- When `value` is a `pto.MaskPattern` enum value: expands to corresponding `pset_b` instruction (`pset_b32`, `pset_b16`, or `pset_b8`)

**Example**:
```python
# Tail processing with f32 vectors: value is pto.i32 → expands to plt_b32
mask_f32, remaining_f32 = pto.make_mask(pto.f32, remaining_elements)

# Tail processing with f16 vectors: value is pto.i32 → expands to plt_b16  
mask_f16, remaining_f16 = pto.make_mask(pto.f16, remaining_elements)

# Tail processing with i8 vectors: value is pto.i32 → expands to plt_b8
mask_i8, remaining_i8 = pto.make_mask(pto.i8, remaining_elements)

# Pattern-based mask with f32 vectors: value is MaskPattern enum → expands to pset_b32
mask_all_f32 = pto.make_mask(pto.f32, PAT.ALL)

# Pattern-based mask with f16 vectors: value is MaskPattern enum → expands to pset_b16  
mask_even_f16 = pto.make_mask(pto.f16, PAT.EVEN)

# Pattern-based mask with i8 vectors: value is MaskPattern enum → expands to pset_b8
mask_all_i8 = pto.make_mask(pto.i8, PAT.ALL)

# Type annotations help clarify expected parameter types
remaining: pto.i32 = 1024
mask1, updated = pto.make_mask(pto.f32, remaining)     # tail processing
mask2 = pto.make_mask(pto.f32, PAT.ALL)              # pattern mode
```

#### `pto.ppack(mask: MaskType) -> pto.i32`

**Description**: Packs mask bits into a 32-bit integer.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Input mask (`mask_b8`, `mask_b16`, or `mask_b32`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `packed` | `pto.i32` | Packed mask bits |

#### `pto.punpack(packed: pto.i32) -> MaskType`

**Description**: Unpacks 32-bit integer to mask (granularity determined by context).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `packed` | `pto.i32` | Packed mask bits |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `MaskType` | Unpacked mask |

#### `pto.pnot(mask: MaskType) -> MaskType`

**Description**: Logical negation of mask bits.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Input mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `negated` | `MaskType` | Negated mask |

#### `pto.psel(mask: MaskType, true_val: ScalarType, false_val: ScalarType) -> ScalarType`

**Description**: Selects between two scalar values based on mask.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Selection mask |
| `true_val` | `ScalarType` | Value selected when mask bit is 1 |
| `false_val` | `ScalarType` | Value selected when mask bit is 0 |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `ScalarType` | Selected scalar value |

### Unary Vector Operations

Element-wise unary operations on vector registers.

#### `pto.vabs(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Absolute value of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask (granularity must match vector element type) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Absolute values |

**Constraints**:
- Mask granularity must match vector element type (e.g., `f32` requires `mask_b32`)

**Example**:
```python
abs_vec = pto.vabs(vec_f32, mask32)
```

#### `pto.vexp(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Exponential of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Exponential values |

#### `pto.vln(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Natural logarithm of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Natural logarithm values |

#### `pto.vsqrt(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Square root of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Square root values |

#### `pto.vrelu(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: ReLU activation (max(0, x)) of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | ReLU-activated values |

#### `pto.vnot(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Bitwise NOT of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Bitwise NOT values |

#### `pto.vcadd(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Complex addition of vector elements (treating pairs as complex numbers).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector (interpreted as complex pairs) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Complex addition result |

#### `pto.vcmax(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Complex maximum of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector (interpreted as complex pairs) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Complex maximum result |

### Binary Vector Operations

Element-wise binary operations on vector registers.

#### `pto.vadd(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise addition of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Sum of vectors |

**Example**:
```python
sum_vec = pto.vadd(vec_a, vec_b, mask32)
```

#### `pto.vsub(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise subtraction of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Difference of vectors |

#### `pto.vmul(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise multiplication of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Product of vectors |

#### `pto.vdiv(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise division of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Quotient of vectors |

#### `pto.vmax(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise maximum of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Element-wise maximum |

#### `pto.vmin(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise minimum of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Element-wise minimum |

#### `pto.vand(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise bitwise AND of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Bitwise AND result |

#### `pto.vor(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise bitwise OR of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Bitwise OR result |

#### `pto.vxor(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise bitwise XOR of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Bitwise XOR result |

#### `pto.vshl(vec: VRegType, shift: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise shift left (vector shift amounts).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `shift` | `VRegType` | Shift amounts (per element) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Shifted values |

#### `pto.vshr(vec: VRegType, shift: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise shift right (vector shift amounts).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `shift` | `VRegType` | Shift amounts (per element) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Shifted values |

### Vector-Scalar Operations

Operations between vectors and scalars.

#### `pto.vmuls(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`

**Description**: Vector multiplied by scalar (broadcast).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Scalar multiplier |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Scaled vector |

**Example**:
```python
scaled = pto.vmuls(vec_f32, pto.f32(2.0), mask32)
```

#### `pto.vadds(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`

**Description**: Vector plus scalar (broadcast).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Scalar addend |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Result vector |

#### `pto.vmaxs(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`

**Description**: Element-wise maximum of vector and scalar.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Scalar value |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Maximum values |

#### `pto.vmins(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`

**Description**: Element-wise minimum of vector and scalar.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Scalar value |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Minimum values |

#### `pto.vlrelu(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`

**Description**: Leaky ReLU activation (max(αx, x)).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Alpha coefficient |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Leaky ReLU activated values |

#### `pto.vshls(vec: VRegType, shift: ScalarType, mask: MaskType) -> VRegType`

**Description**: Vector shift left by scalar (uniform shift).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `shift` | `ScalarType` | Shift amount (same for all elements) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Shifted values |

#### `pto.vshrs(vec: VRegType, shift: ScalarType, mask: MaskType) -> VRegType`

**Description**: Vector shift right by scalar (uniform shift).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `shift` | `ScalarType` | Shift amount (same for all elements) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Shifted values |

### Carry & Select Operations

Operations with carry propagation and selection.

#### `pto.vaddc(vec1: VRegType, vec2: VRegType, carry_in: ScalarType, mask: MaskType) -> (VRegType, ScalarType)`

**Description**: Vector addition with carry input and output.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `carry_in` | `ScalarType` | Input carry bit |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Sum vector |
| `carry_out` | `ScalarType` | Output carry bit |

#### `pto.vsubc(vec1: VRegType, vec2: VRegType, borrow_in: ScalarType, mask: MaskType) -> (VRegType, ScalarType)`

**Description**: Vector subtraction with borrow input and output.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `borrow_in` | `ScalarType` | Input borrow bit |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Difference vector |
| `borrow_out` | `ScalarType` | Output borrow bit |

#### `pto.vsel(mask: MaskType, true_vec: VRegType, false_vec: VRegType) -> VRegType`

**Description**: Vector select based on mask.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Selection mask |
| `true_vec` | `VRegType` | Vector selected when mask bit is 1 |
| `false_vec` | `VRegType` | Vector selected when mask bit is 0 |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Selected vector |

**Example**:
```python
result = pto.vsel(mask32, scaled_vec, original_vec)
```

### Data Rearrangement

Operations for rearranging data within vectors.

#### `pto.pdintlv_b8(mask: pto.mask_b8) -> pto.mask_b8`

**Description**: Deinterleave 8-bit mask.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `pto.mask_b8` | Input 8-bit mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `pto.mask_b8` | Deinterleaved mask |

#### `pto.pintlv_b16(mask: pto.mask_b16) -> pto.mask_b16`

**Description**: Interleave 16-bit mask.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `pto.mask_b16` | Input 16-bit mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `pto.mask_b16` | Interleaved mask |

#### `pto.vintlv(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Interleave two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Interleaved vector |

#### `pto.vdintlv(vec: VRegType, mask: MaskType) -> (VRegType, VRegType)`

**Description**: Deinterleave vector into two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec1` | `VRegType` | First deinterleaved vector |
| `vec2` | `VRegType` | Second deinterleaved vector |

### Conversion & Special Operations

Type conversion and specialized operations.

#### `pto.vtrc(vec: VRegType, mask: MaskType, rnd: str) -> VRegType`

**Description**: Truncate/round floating-point vector elements to integer-valued
floating-point results under an explicit predicate mask.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask; granularity must match element width |
| `rnd` | `str` | Round mode: `R`, `A`, `F`, `C`, or `Z` |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Rounded result with the same floating-point element type |

#### `pto.vcvt(vec: VRegType, to_type: Type, mask: MaskType) -> VRegType`

**Description**: Type conversion of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `to_type` | `Type` | Target element type |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Converted vector |

#### `pto.vbitsort(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Bitonic sort of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Sorted vector |

#### `pto.vmrgsort4(vec1: VRegType, vec2: VRegType, vec3: VRegType, vec4: VRegType, mask: MaskType) -> VRegType`

**Description**: 4-way merge sort of vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `vec3` | `VRegType` | Third input vector |
| `vec4` | `VRegType` | Fourth input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Merged and sorted vector |

### Stateless Store Operations

Operations for storing data from vector registers to memory (stateless).

#### `pto.vsts(vec: VRegType, buf: UBRef, offset: Index, mask: MaskType) -> None`  
#### `pto.vsts(vec: VRegType, tile[row, col:], mask: MaskType) -> None`  
#### `pto.vsts(vec: VRegType, tile[start:], mask: MaskType) -> None`

**Description**: Stateless vector store to buffer. Supports both byte-offset and element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `buf` | `UBRef` | Destination buffer or pointer (UB memory space) |
| `offset` | `Index` | Byte offset |
| `mask` | `MaskType` | Predicate mask |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

**Constraints**:
- Buffer must be in UB memory space
- For byte-offset syntax: offset must be properly aligned based on element type
- For element-indexing syntax: the destination vector region must be within tile bounds and satisfy alignment requirements

**Examples**:
```python
# Byte-offset syntax
pto.vsts(vec_f32, ub_ptr, lane * 256, mask32)

# Element-indexing syntax
pto.vsts(vec, tile[i, j:], mask)      # Store to row i, columns j to j+vector_lanes-1
pto.vsts(vec, tile[k:], mask)         # Store to 1D tile, elements k to k+vector_lanes-1

# In a generic kernel
@pto.vkernel(target="a5", op="copy", dtypes=[(pto.AnyFloat, pto.AnyFloat)], priority=10)
def generic_store(src: pto.Tile, dst: pto.Tile):
    rows, cols = src.shape
    all_mask = pto.make_mask(src.element_type, PAT.ALL)
    for i in range(0, rows):
        for j in range(0, cols, vector_lanes):
            vec = pto.vlds(src[i, j:])
            pto.vsts(vec, dst[i, j:], all_mask)  # No manual offset calculation
```

#### `pto.psts(mask: MaskType, buf: UBRef, offset: Index) -> None`  
#### `pto.psts(mask: MaskType, tile[row, col:]) -> None`  
#### `pto.psts(mask: MaskType, tile[start:]) -> None`

**Description**: Predicate store to buffer. Supports both traditional byte-offset syntax and new element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Mask to store |
| `buf` | `UBRef` | Destination buffer or pointer |
| `offset` | `Index` | Byte offset |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Mask to store |
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column (vector-width range) |

**Parameters (1D element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Mask to store |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index (vector-width range) |

**Returns**: None (side-effect operation)

#### `pto.vsst(scalar: ScalarType, buf: UBRef, offset: Index, mask: MaskType) -> None`  
#### `pto.vsst(scalar: ScalarType, tile[row, col:], mask: MaskType) -> None`  
#### `pto.vsst(scalar: ScalarType, tile[start:], mask: MaskType) -> None`

**Description**: Scalar to vector store (broadcast scalar to all lanes). Supports both traditional byte-offset syntax and new element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | `ScalarType` | Scalar value |
| `buf` | `UBRef` | Destination buffer or pointer |
| `offset` | `Index` | Byte offset |
| `mask` | `MaskType` | Predicate mask |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | `ScalarType` | Scalar value |
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column (vector-width range) |
| `mask` | `MaskType` | Predicate mask |

**Parameters (1D element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | `ScalarType` | Scalar value |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index (vector-width range) |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

#### `pto.vstx2(vec1: VRegType, vec2: VRegType, buf1: UBRef, buf2: UBRef, offset: Index, mask: MaskType) -> None`  
#### `pto.vstx2(vec1: VRegType, vec2: VRegType, tile1[row, col:], tile2[row, col:], mask: MaskType) -> None`  
#### `pto.vstx2(vec1: VRegType, vec2: VRegType, tile1[start:], tile2[start:], mask: MaskType) -> None`

**Description**: Dual vector store to two buffers. Supports both traditional byte-offset syntax and new element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First vector to store |
| `vec2` | `VRegType` | Second vector to store |
| `buf1` | `UBRef` | First destination buffer |
| `buf2` | `UBRef` | Second destination buffer |
| `offset` | `Index` | Byte offset (applied to both buffers) |
| `mask` | `MaskType` | Predicate mask |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First vector to store |
| `vec2` | `VRegType` | Second vector to store |
| `tile1[row, col:]` | `Tile` with indexing | First 2D tile with row index and starting column (vector-width range) |
| `tile2[row, col:]` | `Tile` with indexing | Second 2D tile with row index and starting column (vector-width range) |
| `mask` | `MaskType` | Predicate mask |

**Parameters (1D element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First vector to store |
| `vec2` | `VRegType` | Second vector to store |
| `tile1[start:]` | `Tile` with indexing | First 1D tile with starting element index (vector-width range) |
| `tile2[start:]` | `Tile` with indexing | Second 1D tile with starting element index (vector-width range) |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

#### `pto.vsta(vec: VRegType, buf: UBRef, offset: Index, align: pto.align, mask: MaskType) -> None`  
#### `pto.vsta(vec: VRegType, tile[row, col:], align: pto.align, mask: MaskType) -> None`  
#### `pto.vsta(vec: VRegType, tile[start:], align: pto.align, mask: MaskType) -> None`

**Description**: Aligned vector store with explicit alignment carrier. Supports both traditional byte-offset syntax and new element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `buf` | `UBRef` | Destination buffer or pointer |
| `offset` | `Index` | Byte offset |
| `align` | `pto.align` | Alignment specification |
| `mask` | `MaskType` | Predicate mask |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column (vector-width range) |
| `align` | `pto.align` | Alignment specification |
| `mask` | `MaskType` | Predicate mask |

**Parameters (1D element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index (vector-width range) |
| `align` | `pto.align` | Alignment specification |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

### Stateful Store Operations

Operations for storing data with stateful semantics.

#### `pto.pstu(mask: MaskType, buf: UBRef, offset: Index) -> None`

**Description**: Predicate stateful store.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Mask to store |
| `buf` | `UBRef` | Destination buffer or pointer |
| `offset` | `Index` | Byte offset |

**Returns**: None (side-effect operation)

#### `pto.vstu(vec: VRegType, buf: UBRef, offset: Index, mask: MaskType) -> None`

**Description**: Vector stateful store.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `buf` | `UBRef` | Destination buffer or pointer |
| `offset` | `Index` | Byte offset |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

#### `pto.vstus(align_in: AlignType, offset: i32, vec: VRegType, buf: UBRef) -> AlignType`

**Description**: No-post unaligned vector store with scalar offset.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `align_in` | `AlignType` | Incoming unaligned-store state |
| `offset` | `i32` | Stream advance offset in elements |
| `vec` | `VRegType` | Vector to store |
| `buf` | `UBRef` | UB destination base pointer |

**Returns**: Updated align-state token for a later flush op such as `pto.vstas`.

#### `pto.vstur(align_in: AlignType, vec: VRegType, buf: UBRef, mode: str) -> AlignType`

**Description**: Unaligned vector store using the SPR-AR-driven stateful form.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `align_in` | `AlignType` | Incoming unaligned-store state |
| `vec` | `VRegType` | Vector to store |
| `buf` | `UBRef` | UB destination base pointer |
| `mode` | `str` | `POST_UPDATE` or `NO_POST_UPDATE` |

**Returns**: Updated align-state token for a later flush op such as `pto.vstar`.

## Examples

### Simple Vector Copy

```python
@pto.vkernel(...)
def vector_copy(src: pto.memref(256, pto.f32, MemorySpace.UB),
                dst: pto.memref(256, pto.f32, MemorySpace.UB)):
    all_mask = pto.make_mask(pto.f32, PAT.ALL)
    for offset in range(0, 256, 64):
        vec = pto.vlds(src, offset)
        pto.vsts(vec, dst, offset, all_mask)
```

### Conditional Computation

```python
@pto.vkernel(...)
def conditional_scale(src: pto.ptr(pto.f32, MemorySpace.GM),
                      dst: pto.ptr(pto.f32, MemorySpace.GM),
                      threshold: pto.f32):
    # ... setup ...

    with pto.strict_vecscope(ub_in, ub_out, threshold) as (vin, vout, thresh):
        for i in range(0, 1024, 64):
            vec = pto.vlds(vin, i)

            # Compare with threshold
            mask = pto.pge_b32(vec, thresh)

            # Scale values above threshold
            scaled = pto.vmuls(vec, pto.f32(2.0), mask)

            # Keep original values below threshold
            result = pto.vsel(mask, scaled, vec)

            pto.vsts(result, vout, i, all_mask)
```

### Loop with Carry

```python
@pto.vkernel(...)
def prefix_sum(src: pto.ptr(pto.i32, MemorySpace.UB),
               dst: pto.ptr(pto.i32, MemorySpace.UB)):
    all_mask = pto.make_mask(pto.i32, PAT.ALL)
    carry = pto.i32(0)

    for i in range(0, 256, 64):
        vec = pto.vlds(src, i)
        result, carry = pto.vaddcs(vec, carry, all_mask)
        pto.vsts(result, dst, i, all_mask)
```

## Common Errors

### Typed Mask Mismatch

```
Error: f32 vector operation cannot consume mask_b16
```

**Solution:** Ensure mask granularity matches vector element size:
- `f32` vectors use `mask_b32`
- `f16` vectors use `mask_b16`
- `i8` vectors use `mask_b8`

### Strict Scope Implicit Capture

```
Error: strict_vecscope body cannot capture outer value 'ub_in' implicitly
```

**Solution:** Pass all required values in the capture list:

```python
# Wrong:
with pto.strict_vecscope() as ():
    vec = pto.vlds(ub_in, offset)  # ub_in from outer scope

# Correct:
with pto.strict_vecscope(ub_in) as (ub):
    vec = pto.vlds(ub, offset)
```

### Untyped Loop Carried State

```
Error: loop-carried value must have explicit machine type
```

**Solution:** Add type annotations to loop-carried variables:

```python
# Wrong:
remaining = 1024  # Plain Python int
for i in range(0, N, step):
    mask, remaining = pto.make_mask(pto.f32, remaining)

# Correct:
remaining: pto.i32 = 1024
# or
remaining = pto.i32(1024)
```

## Compatibility Notes

The current experimental implementation in `python/pto/dialects/pto.py` differs from this specification in several ways:

1. **Mask types**: The experimental version uses untyped `mask` instead of `mask_b8`/`mask_b16`/`mask_b32`
2. **Barrier operation**: Uses `pto.barrier()` instead of `pto.pipe_barrier()`
3. **MemRef support**: Does not yet support `pto.memref()` types
4. **Operation coverage**: Implements only a subset of operations

When implementing new code, follow this specification. The experimental implementation will be updated to match over time.

## Next Steps

- Explore the ISA documentation in `docs/isa/` for detailed operation semantics
- Check `test/samples/` for example kernels
- Refer to `docs/vpto-spec.md` for the underlying VPTO instruction specification

For compiler developers, see `docs/PTO_IR_manual.md` for MLIR-level details.
