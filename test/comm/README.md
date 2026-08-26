# Async communication host helpers

Host-side support for `pto.sdma_gm_gm` and the session template a kernel loads
with `pto.session_init`.

| File | Role |
| --- | --- |
| `AsyncWorkspace.h` | Builds the async workspace: streams, the AICPU STARS query, and the channel record table a kernel reads. Resolves every CANN symbol with `dlopen`. |
| `AsyncWorkspaceShim.cpp` | `extern "C"` surface over that header, so Python can drive it without a second copy of the session ABI. |
| `async_workspace.py` | `ctypes` wrapper over the shim. |
| `spike_sdma_gm_gm_engine.py` | The engine-path spike described below. |
| `build_async_shim.sh` | Standalone build of the shim into `build/libpto_async_shim.so`. Not needed by the spike, which gets the shim through the kernel build; useful to check the shim compiles on a machine with no card. |

## How the shim reaches Python

The spike does not build or load a separate library. It lists the shim in its
kernel declaration:

```python
@pto.jit(
    name="sdma_gm_gm_engine",
    native_options={
        "host_sources": ["AsyncWorkspaceShim.cpp"],
        "include_dirs": [".", "../../include"],
        "link_libraries": ["dl"],
    },
    ...
)
```

PTODSL compiles that as host C++ and links it into the kernel's own shared
library, so `compiled.native_library()` hands back one library carrying both the
launch entry and the shim. One build, one artifact, and the shim's contents are
part of the build's cache key — editing it rebuilds rather than silently reusing.
See `ptodsl/docs/user_guide/03-kernel-entry-and-subkernels.md`, "Host C++ in the
kernel's library".

## Why a shim rather than a Python port

Every session field index, slot width, SQE constant and channel-record offset
lives in `include/PTO/Support/AsyncSessionABI.h`, and the expansion in
`lib/PTO/Transforms/VPTOExpandWrapperOps.cpp` reads them from there. A Python
test that filled a session template by index would be a second, unchecked copy
of that layout, and the header exists specifically to keep there from being two.

So the shim takes named values and returns an opaque byte image. It asks the C++
side how large a session template is; it never says. `async_workspace.py`
contains no offsets at all, and adding a session field cannot silently break it.

The shim needs no CANN headers or libraries to build, because `AsyncWorkspace`
resolves the toolkit at run time through `dlopen`. `-ldl` is the only link
dependency, so it compiles on a machine with no driver and no card.

## The engine-path spike

`test/vpto/cases/async-comm/sdma_gm_gm.py` covers the A5 `{soft_put}` form of
`pto.sdma_gm_gm` as an ordinary golden ST case. It can do that because
`{soft_put}` expands to a synchronous GM→UB→GM copy: the transfer is finished
when the kernel returns.

The engine form is not like that. The kernel writes SQEs, publishes the queue
tail and rings a doorbell; the SDMA engine moves the bytes afterwards. Running it
under the PTODSL ST harness needs three things to be true, and none of them can
be checked without a card:

1. `AsyncWorkspace` can attach to the device and context `torch_npu` already set
   up, rather than needing to own `aclrtSetDevice` itself.
2. Memory the workspace allocates through the runtime is addressable by a kernel
   the harness launched, so a raw device address can be handed over inside a
   session template.
3. A destination the engine fills can be observed from the harness, which
   synchronizes the kernel and knows nothing about the engine.

`spike_sdma_gm_gm_engine.py` is what answers that. It is deliberately not under
`test/vpto/cases/`: `run_host_vpto_validation.sh` discovers and runs every `.py`
in that tree, and a case needing a hand-built shim and a CANN 9.0 toolkit would
fail the whole validation run everywhere else. It also has no skip path, because
a spike that quietly passes when it could not run answers nothing.

If all three hold, the engine path needs no change to the ST framework, and the
C++ and shell harness these cases used to require has nothing left to do. If one
fails, the spike reports which.

### Running it

Requires CANN ≥ 9.0.0 for `aclnnShmemSdmaStarsQuery`, a real device, and a
`torch_npu` the harness can import. The shim is built as part of the kernel, so
there is no separate build step.

`PTO_ASYNC_ARCH` has to name the generation of the card, and both are worth
running: the three assumptions above are generation-independent, but the doorbell
is not, so A5 and A2/A3 exercise different final writes.

```bash
PTO_ASYNC_ARCH=a5 python3 test/comm/spike_sdma_gm_gm_engine.py
PTO_ASYNC_ARCH=a3 python3 test/comm/spike_sdma_gm_gm_engine.py
```

Start with A5. It is where `AsyncWorkspace` has already been through a live STARS
query — the 2049-for-2048 ring workaround in `AsyncWorkspace.h` was observed
there on CANN 9.2.0 — and the A2/A3 doorbell path is the one that can take a card
down if the sequence is wrong.

To check that the shim itself compiles without a card or a kernel:

```bash
test/comm/build_async_shim.sh
```

Useful before touching a card:

```bash
PTO_ASYNC_ARCH=a5 python3 test/comm/spike_sdma_gm_gm_engine.py --list       # case names
PTO_ASYNC_ARCH=a5 python3 test/comm/spike_sdma_gm_gm_engine.py --emit-mlir  # the kernel, no device needed
```

Knobs:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PTO_ASYNC_ARCH` | **none, required** | `a2`, `a3` or `a5` — the generation of the card being pointed at. The doorbell is the one part of the post that differs: A2/A3 reaches it only by MTE and stages the tail in UB, A5 writes it with `st_dev` at a different offset. Deliberately not defaulted, because a store aimed at `sq_reg_base` the wrong way can leave an A2/A3 card in an unrecoverable RAS state. `a2` and `a3` generate the same code. |
| `PTO_ASYNC_POLL_TIMEOUT_MS` | `2000` | How long to wait for the engine to drain before calling the destination wrong. |
| `PTO_ASYNC_SHIM` | `test/comm/build/libpto_async_shim.so` | Shim path, for a caller loading a standalone build rather than taking it from a kernel library. |
| `CXX` | `g++` | Host compiler for `build_async_shim.sh`. |

### What each case separates

All four run the same kernel and differ only in the session template the host
writes, which is the point: the transfer is described by session data, not by the
kernel. `block_bytes` is what splits one transfer into SQEs, so it decides how
many entries the queue should gain.

| Case | Transfer | Entries |
| --- | --- | --- |
| `single_entry` | 4096 bytes in 4096-byte blocks | 1 |
| `even_split` | 4096 bytes in 1024-byte blocks | 4 |
| `ragged_tail` | 4096 bytes in 1536-byte blocks | 3, the last one short |
| `comm_block_offset` | 2048 bytes at offset 2048 of an 8 KiB buffer | 1 |

Each case checks the queue tail as well as the data. Correct bytes at the
destination only say the transfer happened; one oversized entry moves the same
bytes as a correct split, and the tail is what separates them. A correct entry
count with an unfilled destination is reported differently from a wrong entry
count, because the first means the kernel did its part and the engine did not.

### If the spike passes

Promoting it is a rename into `test/vpto/cases/async-comm/`; the shim needs no
step in `run_host_vpto_validation.sh`, because the kernel build already carries
it. After that, the two-device window shapes in `test/lit/vpto/async_*.pto` become
the next thing to give a runtime case. That needs `HcclWindows.h`, which went away
with the C++ cases that were its only caller; it comes back through the same shim
pattern and the same `native_options` entry, so nothing here has to change to
accommodate it.
