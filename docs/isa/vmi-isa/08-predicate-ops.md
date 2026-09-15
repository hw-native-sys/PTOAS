# 8. Predicate Generation Ops

> **Category:** gen (mask producers — take no input mask).
> **Mask in:** none (they generate masks).
>
> Mask generation is expressed with two ops: `create_mask` (prefix / first-N
> tail) and `create_group_mask` (grouped prefix / grouped first-N tail). Mask
> granularity (`b8`/`b16`/`b32`) is derived from the result type, not spelled in
> the op name.
>
> `create_mask` takes a single `index` operand `active_lanes`. When
> `active_lanes ≥ L` it yields an all-active mask; when `active_lanes = N < L`
> it yields a first-N tail mask. `create_group_mask` repeats the first-N pattern
> within each of `num_groups` equal groups (group size `group_size`).

```mlir
%act  = arith.minsi %rem, %cL   // min(rem, L)
%aidx = arith.index_cast %act   // i32 -> index
%mask = pto.vmi.create_mask %aidx : index -> !pto.vmi.mask<128×b32>
%next = arith.subi %rem, %act   // rem - min(rem, L)
```


---

## `pto.vmi.create_mask`

- **syntax:**
  ```mlir
  %m = pto.vmi.create_mask %active_lanes : index -> !pto.vmi.mask<L>
  ```
- **semantics:** Create a predicate mask where the first `active_lanes` logical
  lanes are active and the rest are inactive. `active_lanes ≥ L` produces an
  all-active mask; `active_lanes = N` produces a first-N tail mask.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = (i < active_lanes) ? 1 : 0;
  ```

- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `active_lanes` | `index` | Number of leading active lanes |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.mask<L>` | Predicate mask |

- **example:**
  ```mlir
  // All-active mask (active_lanes >= L)
  %all = pto.vmi.create_mask %c128 : index -> !pto.vmi.mask<128×b32>

  // First-N tail mask (N = 64)
  %tail = pto.vmi.create_mask %c64 : index -> !pto.vmi.mask<128×b32>
  ```


---

## `pto.vmi.create_group_mask`

- **syntax:**
  ```mlir
  %m = pto.vmi.create_group_mask %active_elems_per_group {num_groups = C, group_size = S}
      : index -> !pto.vmi.mask<L>
  ```
- **semantics:** Create a grouped predicate mask. The mask is divided into
  `num_groups` equal groups of `group_size` lanes each; lane `i` is active iff
  `(i % group_size) < active_elems_per_group`. When
  `active_elems_per_group ≥ group_size` all lanes are active within every group
  (grouped all-active); otherwise the first `active_elems_per_group` lanes are
  active within each group (grouped first-N tail).

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = ((i % group_size) < active_elems_per_group) ? 1 : 0;
  ```

- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `active_elems_per_group` | `index` | Active lanes within each group |

- **attributes:**

  | Attribute | Values | Description |
  |---|---|---|
  | `num_groups` | positive integer | Number of equal groups |
  | `group_size` | positive integer | Lanes per group (`L / num_groups`) |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.mask<L>` | Grouped predicate mask |

- **example:**
  ```mlir
  // Grouped all-active: 8 groups, group size 32, all lanes active per group
  %all = pto.vmi.create_group_mask %c32 {num_groups = 8, group_size = 32}
      : index -> !pto.vmi.mask<256×b32>

  // Grouped first-N tail: first 25 lanes per group, 8 groups
  %tail = pto.vmi.create_group_mask %c25 {num_groups = 8, group_size = 32}
      : index -> !pto.vmi.mask<256×b32>
  ```

`num_groups` is logically legal for any positive divisor of the result mask
lane count. A backend may impose a narrower materialization limit separately.


---

## Mask Boolean Ops (`vand` / `vor` / `vxor` / `vnot` on masks)

The elementwise bitwise ops are reused directly on mask operands, treated as a
per-lane bit-wise boolean op on the predicate.

- **example:**
  ```mlir
  // Predicate boolean ops on masks
  %and = pto.vmi.vand %lt, %gt
      : !pto.vmi.mask<128xpred>, !pto.vmi.mask<128xpred>
      -> !pto.vmi.mask<128xpred>
  %or = pto.vmi.vor %lt, %gt
      : !pto.vmi.mask<128xpred>, !pto.vmi.mask<128xpred>
      -> !pto.vmi.mask<128xpred>
  %xor = pto.vmi.vxor %lt, %gt
      : !pto.vmi.mask<128xpred>, !pto.vmi.mask<128xpred>
      -> !pto.vmi.mask<128xpred>
  %not = pto.vmi.vnot %lt
      : !pto.vmi.mask<128xpred> -> !pto.vmi.mask<128xpred>
  ```

- **lowering:** mask logic is normalised onto the mask interface by `pto.as` in
  `vmi-lower-unified-to-legacy`: `vand` / `vor` / `vxor` / `vnot` on masks
  become `pto.vmi.mask_and` / `mask_or` / `mask_xor` / `mask_not`, which lower
  to `pto.pand` / `por` / `pxor` / `pnot`. The vreg form of the same spellings
  becomes the vreg interface (`pto.vmi.andi` / `ori` / `xori` / `not`), which
  keeps its governing predicate mask.
