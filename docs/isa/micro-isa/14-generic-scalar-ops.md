# 14. Generic Scalar and Builtin Vector Operations

> **Category:** Common value computation used around PTO operations
> **Dialect:** `pto`

PTO micro Instruction programs use scalar values for dimensions, offsets,
predicates, and addresses, and builtin vectors for short elementwise value
groups. This chapter is the authoring reference for those computations. It
covers common PTO value operations whose meaning is independent of SIMT
workitem state. The public authoring surface uses `pto.*` consistently;
standard MLIR arithmetic operations are not a parallel public interface.

These operations may be used in ordinary AICore code or in a SIMT body. Merely
using a scalar value operation does not select SIMT execution. Scalar memory
access is documented separately in
[Special Scalar Operations](18-special-scalar.md).

---

## PTO Common Value Operations

Use the `pto.*` forms below for public PTO value operations. Arithmetic,
integer division, bitwise operations, comparisons, extrema, and absolute value
accept either scalars or builtin vectors. Vector forms are elementwise and
require matching shapes and element types.

Common integer values use signless `i*` element types. Operations whose result
depends on integer interpretation write `signed` or `unsigned` explicitly in
their assembly. Signed and unsigned vreg element types belong to the
hardware-facing operation surface and are not accepted as common value types.

| Family | Operations | Supported values |
|--------|------------|------------------|
| Constants | `pto.constant` | integer, floating-point, `index`, or builtin vector value |
| Arithmetic | `pto.addi/addf`, `pto.subi/subf`, `pto.muli/mulf`, `pto.negi/negf`, `pto.addui_extended`, `pto.mul_extended` | category-specific integer, floating-point, `index`, or builtin vector values |
| Division | `pto.divi`, `pto.divf`, `pto.floordiv`, `pto.ceildiv` | matching values of the operation's numeric category |
| Remainder | `pto.remi`, `pto.remf` | matching values of the operation's numeric category |
| Bitwise | `pto.and`, `pto.or`, `pto.xor` | matching integer or `index` scalars, or integer builtin vectors |
| Shift | `pto.shl`, `pto.shr` | matching integer scalars or builtin vectors |
| Comparison | `pto.cmpi`, `pto.cmpf` | matching integer/index or floating-point values |
| Extrema | `pto.maxi`, `pto.maxf`, `pto.mini`, `pto.minf`, `pto.maximum`, `pto.minimum` | matching values of the operation's numeric category |
| Absolute value | `pto.absi`, `pto.absf` | integer/index or floating-point value |
| Conversion | `pto.exti`, `pto.trunci`, `pto.ftof`, `pto.ftoi`, `pto.itof`, `pto.bitcast`, `pto.index_cast` | compatible scalar or shape-preserving builtin-vector numeric types described below |
| Selection | `pto.select` | scalar `i1` or same-shape builtin-vector `i1` condition and matching alternatives; scalar `i1` may select whole vectors |
| Floating math | `pto.exp`, `pto.log`, `pto.sqrt`, `pto.pow`, `pto.fma` | supported floating-point values |

### Constants and primitive arithmetic

```mlir
%c0 = pto.constant 0 : index
%scale = pto.constant 2.000000e+00 : f32
%offset = pto.addi %base, %c0 : index
%scaled = pto.mulf %value, %scale : f32
%quotient = pto.divf %scaled, %denominator : f32
%bias = pto.constant dense<1.000000e+00> : vector<4xf32>
%sum4 = pto.addf %value4, %bias : vector<4xf32>
%negative4 = pto.negf %sum4 : vector<4xf32>
```

- `pto.constant` creates an integer, floating-point, `index`, or dense builtin
  vector value of the written type.
- Integer `*i` and floating-point `*f` operations require both operands and the
  result to have the same category and scalar or builtin-vector type.
- `pto.negi` and `pto.negf` compute the additive inverse of a scalar or each builtin-vector
  element. Integer overflow follows the element type's ordinary wraparound
  behavior.
- `pto.divf` accepts floating-point scalars or builtin vectors and performs
  floating-point division elementwise.

Integer overflow and floating-point exceptional values follow the ordinary
semantics of the corresponding element type. Optional semantic attributes are
written only when their non-default contract is intended:

```mlir
%sum = pto.addi %lhs, %rhs overflow<nsw, nuw> : i32
%fast = pto.addf %flhs, %frhs fastmath<nnan,ninf> : f32
%half = pto.ftof %value round(z) fastmath<nnan> : f32 -> f16
```

`overflow<nsw>` and `overflow<nuw>` promise that signed or unsigned overflow
does not occur; violating an asserted promise produces poison. They are
available on integer `pto.addi/subi/muli/negi/shl` and `pto.trunci`.
`fastmath<...>` carries the standard floating-point optimization permissions;
it can relax NaN, infinity, signed-zero, reassociation, reciprocal, contraction,
or approximation behavior. With no attribute, strict behavior is preserved.

### Division, remainder, and bitwise operations

```mlir
%tiles = pto.floordiv %elements, %tile_size signed : index
%groups = pto.ceildiv %elements, %group_size signed : index
%tail = pto.remi %elements, %tile_size signed : index
%fraction = pto.remf %value, %period : f32
%enabled = pto.and %flags, %mask : i32
%packed = pto.shl %enabled, %shift : i32
%high = pto.shr %packed, %shift unsigned : i32
```

The signedness clause records integer interpretation explicitly. `pto.divi`
rounds toward zero; `pto.floordiv` rounds toward negative infinity for signed
inputs; and `pto.ceildiv` rounds toward positive infinity. For unsigned
inputs, floor division has the same quotient as unsigned division. `index`
authoring conventionally uses `signed`, while explicit PTO IR may select
`unsigned` when unsigned interpretation is required. `pto.remi` uses the same
signedness clause, while `pto.remf` is the floating-point remainder operation.

`pto.and`, `pto.or`, and `pto.xor` apply bitwise operations to matching integer
or `index` values. For `i1`, they act as boolean conjunction, disjunction, and
exclusive disjunction. Integer builtin-vector forms apply the same operation
independently to each element.

`pto.shl` shifts left and fills low bits with zero. `pto.shr ... signed` is an
arithmetic right shift and `pto.shr ... unsigned` is a logical right shift. Shift operands must be
fixed-width integers; builtin-vector forms apply the corresponding shift
independently to each element.

### `pto.cmpi` and `pto.cmpf`

```mlir
%has_work = pto.cmpi gt %remaining, %zero signed : index
%in_range = pto.cmpi le %value, %limit unsigned : i32
%ordered = pto.cmpf ne %lhs, %rhs : f32
%lanes = pto.cmpf gt %lhs4, %rhs4 : vector<4xf32>
```

Integer operands use `eq`, `ne`, `lt`, `le`, `gt`, or `ge` followed by the
required `signed` or `unsigned` clause. Predicate and integer interpretation
are deliberately separate, as on the other sign-sensitive operations.
Floating-point operands use the ordered shorthand `eq`, `ne`, `lt`, `le`,
`gt`, and `ge`, and also accept `false`,
`oeq`, `ogt`, `oge`, `olt`, `ole`, `one`, `ord`, `ueq`, `ugt`, `uge`, `ult`,
`ule`, `une`, `uno`, and `true`. Both operands must have the same integer,
floating-point, `index`, or builtin vector type.
Scalar operands produce `i1`; builtin-vector operands produce a same-shape
builtin vector with `i1` elements. Floating-point
shortcuts are ordered and correspond to `oeq`, `one`, `olt`, `ole`, `ogt`, and
`oge`. Explicit `o*` predicates are true only for ordered inputs; explicit
`u*` predicates are true when either input is NaN or the stated relation holds.
`ord` tests that neither input is NaN, while `uno` tests that at least one is
NaN. `false` and `true` produce their named constant result.

### Extrema

- **Purpose:** Select the greater or lesser of two scalar or builtin-vector values.
- **Syntax:**

  ```mlir
  %hi = pto.maxi %lhs, %rhs signed : i32
  %lo = pto.mini %ulhs, %urhs unsigned : i32
  %fhi = pto.maxf %flhs, %frhs : f32
  ```

- **Operands and result:** operands and result have the same scalar or
  builtin-vector type. Vector forms are elementwise.
- **Integer semantics:** the required `signed` or `unsigned` clause selects the
  ordering.
- **Index semantics:** `index` values use the ordering selected by the required
  `signed` or `unsigned` clause. PTODSL-generated index extrema use `signed` by
  default.
- **Floating-point semantics:** The operations use maxNum/minNum behavior. If
  exactly one operand is NaN, the non-NaN operand is returned. If both operands
  are NaN, the result is NaN. Other values are compared numerically.
- **NaN-propagating variants:** `pto.maximum/minimum` return NaN when either
  operand is NaN and order signed zero as `-0.0 < +0.0`. They are separate
  because this observable behavior differs from `pto.maxf/minf`.

```text
max(lhs, rhs) = lhs if lhs >= rhs else rhs
min(lhs, rhs) = lhs if lhs <= rhs else rhs
```

Examples:

```mlir
%bounded = pto.mini %requested, %capacity signed : index
%largest = pto.maxf %lhs, %rhs : f32
%strict = pto.maximum %lhs, %rhs : f32
```

### Common floating math

`pto.pow` computes elementwise floating-point power. `pto.fma` computes fused
`lhs * rhs + acc` with one final rounding. Like `pto.exp`, `pto.log`, and
`pto.sqrt`, these are value operations independent of SIMT workitem state.
Scalar and builtin-vector forms share the same public numerical contract.

```mlir
%powered = pto.pow %base, %exponent : f32, f32 -> f32
%fused = pto.fma %lhs, %rhs, %acc : f32, f32, f32 -> f32
```

### `pto.absi` and `pto.absf`

- **Purpose:** Compute a scalar absolute value.
- **Syntax:** `%result = pto.absi %value signedness : T` or `%result = pto.absf %value : T`
- **Operand and result:** `%value` and the result have the same integer,
  floating-point, or `index` type `T`.
- **Semantics:** Integer and `index` operands require `signed` or `unsigned`.
  Signed values, including signed-interpreted `index`, produce their absolute
  value; unsigned values are unchanged.
  Floating-point operands omit signedness and produce their absolute value.
- **Constraints:** For the most-negative signed integer, whose positive value
  is not representable in the same width, the result follows the scalar integer
  absolute-value contract and must not be assumed to saturate.

```mlir
%magnitude = pto.absi %delta signed : i32
%distance = pto.absi %offset signed : index
```

### Numeric conversion

- **Purpose:** Perform a standard scalar or builtin-vector numeric conversion
  with an operation name that identifies the source and destination category.
- **Operations:** `pto.exti`, `pto.trunci`, `pto.ftof`, `pto.ftoi`, and
  `pto.itof`.
- **Legal forms:**

  | Source | Destination | Behavior |
  |--------|-------------|----------|
  | floating-point | floating-point | `pto.ftof`; convert to the destination format |
  | narrower integer | wider integer | `pto.exti`; sign-extend or zero-extend according to signedness |
  | wider integer | narrower integer | `pto.trunci`; keep the low destination bits |
  | integer | floating-point | `pto.itof`; convert the signed or unsigned integer value to floating point |
  | floating-point | integer | `pto.ftoi`; convert to the signed or unsigned integer value, truncating toward zero |
  | `index` | integer | convert the index representation to the requested integer width |
  | integer | `index` | convert the integer representation to `index` |

  Builtin-vector conversion accepts all four fixed-width numeric rows. Source
  and destination vectors must have the same shape, and the conversion is
  applied independently to each element.

- **Integer width changes:** `pto.exti` requires `signed` or `unsigned`.
  `pto.trunci` is signedness-independent and may carry explicit overflow
  promises. Equal-width carrier reinterpretation requires no numeric operation.
- **Cross-category conversion:** The required signedness clause selects signed
  or unsigned integer interpretation. Ordinary generic floating-point to
  integer conversion follows the corresponding `arith.fptosi/fptoui`
  non-saturating contract. SIMT hardware conversion uses the explicit
  `round(...)` and `sat`/`nosat` controls required by its instruction contract.
  PTODSL authors these category operations through `pto.cast`, which carries
  those controls when the selected conversion form supports them.

```mlir
%wide = pto.exti %small signed : i16 -> i32
%half = pto.ftof %value : f32 -> f16
%count = pto.ftoi %ratio unsigned : f32 -> i32
%wide4 = pto.ftof %half4 : vector<4xf16> -> vector<4xf32>
```

Floating truncation accepts `round(r)`, `round(a)`, `round(f)`, `round(c)`,
`round(z)`, `round(o)`, or `round(h)` for nearest-even, nearest-away,
downward, upward, toward-zero, to-odd, and hybrid modes. Outside SIMT, rounding
is only meaningful for narrowing `ftof`; generic `ftoi` and `itof` reject it.
SIMT hardware conversions accept only modes supported by their instruction
contract, with `to_odd` and `hybrid` restricted to the corresponding SIMT
forms. Floating extension/truncation may carry `fastmath<...>`.

### Extended integer arithmetic

`pto.addui_extended` returns the wrapped sum and an `i1` overflow result.
`pto.mul_extended` returns the low and high halves of a signed or unsigned
full-width product, selected by its signedness clause. Builtin-vector forms return
same-shape results.

```mlir
%sum, %overflow = pto.addui_extended %lhs, %rhs : i32, i1
%low, %high = pto.mul_extended %lhs, %rhs signed : i32
```

### `pto.bitcast`

- **Purpose:** Reinterpret a numeric scalar or builtin vector without numeric
  conversion.
- **Syntax:** `%result = pto.bitcast %value : Src -> Dst`
- **Constraints:** Source and destination must both be numeric scalars or both
  be builtin vectors. Their element bit widths must match; vector shapes must
  also match. Integer and floating-point types may be mixed.
- **Semantics:** Every source bit is preserved. The destination type only
  changes how that bit pattern is interpreted.

```mlir
%bits = pto.bitcast %value : f32 -> i32
%values = pto.bitcast %bits4 : vector<4xi32> -> vector<4xf32>
```

### `pto.index_cast`

- **Purpose:** Make a conversion between an `index` value and an integer value
  explicit.
- **Syntax:** `%result = pto.index_cast %value signedness : Src -> Dst`
- **Constraints:** Exactly one of `Src` and `Dst` must be `index`; the other
  must be an integer type. Integer-to-integer and index-to-index forms are
  invalid.

```mlir
%offset = pto.index_cast %offset_i32 signed : i32 -> index
%count = pto.index_cast %count_idx unsigned : index -> i64
```

### `pto.select`

- **Purpose:** Choose one of two scalar or builtin-vector values.
- **Syntax:** `%result = pto.select %condition, %true_value, %false_value : T`
- **Operands:** For scalar `T`, `%condition` is `i1`. For builtin-vector `T`,
  `%condition` is a same-shape builtin vector with `i1` elements.
  `%true_value` and `%false_value` have the same integer, floating-point,
  `index`, or builtin-vector type `T`.
- **Result:** The result has type `T` and equals `%true_value` when the
  condition is true, otherwise `%false_value`.

```text
result = condition ? true_value : false_value
```

```mlir
%active_count = pto.select %is_tail, %tail_count, %full_count : index
%active4 = pto.select %lane_mask, %lhs4, %rhs4 : vector<4xf32>
```

### `pto.exp` / `pto.log` / `pto.sqrt`

- **Purpose:** Compute the natural exponential, natural logarithm, or square
  root of a floating-point value.
- **Syntax:**

  ```mlir
  %e = pto.exp %x : T -> T
  %l = pto.log %x : T -> T
  %s = pto.sqrt %x : T -> T
  ```

- **Operand and result:** The operand and result have the same type. `T` is
  `f16`, `f32`, or `vector<2xf16>`. In the packed form the operation is applied
  independently to each element.
- **Semantics:** `pto.exp` computes `e ** x`; `pto.log` computes `ln(x)`; and
  `pto.sqrt` computes the principal square root. Overflow, underflow,
  infinities, NaNs, and out-of-domain inputs follow the target floating-point
  rules.

```mlir
%scale = pto.exp %delta : f32 -> f32
%root = pto.sqrt %variance : f32 -> f32
```

---

## Scalar and control-flow boundary

- Use the `pto.*` operations in this chapter for scalar values in PTO source.
- Standard `arith`, `math`, and LLVM operations are compiler legalization IR,
  not a second public scalar authoring surface.
- Use `scf` with `pto.cmpi` or `pto.cmpf` results for structured control flow; see
  [SCF](15-shared-scf.md).
- Use vector or tile PTO operations for vector/tile payload computation rather
  than applying scalar operations to PTO vector-register values.
