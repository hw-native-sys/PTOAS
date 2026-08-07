# Feature request: reliable compact f32 group-store on device

Audience: PTOAS/VMI maintainers.

## Requested contract

A f32 VMI value whose logical layout contains one group/slot should support:

```python
pto.vmi.vstore(value, dst, offset, group=1, stride=1)
```

at every naturally aligned f32 destination. It should have the same observable
result as native ASC/CCE `vsts(value, dst, 0, ONEPT_B32, mask)`: lane 0 is
written and adjacent elements are unchanged.

## Observed mislowering

The desired fixture enters layout assignment with a contiguous 64-lane result:

```text
pto.vmi.group_broadcast_load ... ->
  !pto.vmi.vreg<64xf32, #pto.vmi.layout<contiguous>>
pto.vmi.group_store ... {num_groups = 1}
```

The emitted VPTO incorrectly contains:

```text
%value = pto.vlds ... {dist = "BRC_B32"}
...
pto.vsts %value, %compact_destination[%zero], %all_mask
```

There is no `dist = "1PT_B32"`. The `group=1` store constraint did not force a
compact group-slot representation or point-store lowering, so a dense store is
issued at each 4-byte destination.

## Why existing compile-only coverage is insufficient

`test/lit/vmi_new/vmi_to_vpto_group_store_compact_small.pto` verifies
`1PT_B32` for inputs already typed with a compact group-slot layout. It does
not cover a full broadcast vector feeding `group_store(num_groups=1)`. Native
compilation therefore succeeds even though the emitted store is unsafe. The
consequence appears when synchronizing the launched kernel:

```text
RuntimeError: npuSynchronizeDevice: ... NPU function error:
device error type 3, error code is 507035
ERR00100 PTA call acl api failed
```

Both a focused layout/lowering test for the broadcast-to-compact transition and
an on-device regression are needed.

## Acceptance criteria

1. `scripts/check_compile.sh` builds the native baseline and lowers both VMI
   variants through PTOAS.
2. Emitted VPTO for the desired case contains `BRC_B32` for its loads and
   `1PT_B32` for each logical scalar store; it must not emit a dense full-mask
   store into the compact destination.
3. `scripts/run_device.sh DEVICE` reports a numeric pass for the native
   baseline.
4. The desired compact VMI fixture reports a numeric pass and never emits
   507035.
5. The padded workaround also passes, demonstrating parity of the surrounding
   broadcast/multiply/add computation.
6. The device test is retained in PTOAS CI so a compile-success/device-fault
   regression cannot recur.

## Related fix boundary

Commit `823ec908` fixed a closely related but earlier failure: a logical
one-slot VMI load lowered to `vsldb`, which required 32-byte source alignment.
It now lowers to a scalar broadcast load. This fixture already emits
`BRC_B32`; its separate defect is the missing layout conversion/point-store
lowering at the following `group=1` store.

## Non-goals

- Exercising a frontend outside this repository.
- Reproducing an end-to-end application kernel.
- Treating the 64× UB allocation and SIMT extraction as an acceptable final
  implementation.
