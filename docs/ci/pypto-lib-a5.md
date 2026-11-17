# pypto-lib A5 board nightly

pypto-lib is the model and example library built on PyPTO, and PyPTO consumes
PTOAS as a released wheel. A PTOAS regression therefore reaches pypto-lib one
hop later than PyPTO's own gate catches it. This nightly runs pypto-lib's A5
on-board selection against the PTOAS build under test, on the same board host
and card as the other nightlies.

It answers one question: **does the PTOAS build we are about to ship still run
pypto-lib's A5 board cases?**

## Why a host timer instead of a GitHub runner

pypto-lib's CI is GitHub Actions. Its A5 jobs ask for
`runs-on: [self-hosted, linux, npu-a5]`:

| Job | Selection |
| --- | --- |
| `ci.yml` `device (a5)` | `detect_changes.py --root examples --platform a5` (PR-scoped; 11 entries on main) |
| `daily_ci.yml` `Model tests (a5)` | `detect_changes.py --root models --platform a5` (32 entries on main) |
| `daily_ci.yml` `Operator performance (a5)` | `ci_perf.py ci --platform a5` (timing only, not a correctness gate) |

The runners those jobs land on report names of the form `238PR-npu-a5-*` /
`238DT-npu-a5-*`, but they are **not** the A5 board host the pto-as nightlies
use: that host has no GitHub Actions runner registered, no
`/opt/pypto/toolchain` bundle, and no `task-submit` -- the card-borrowing queue
every pypto-lib device step calls. Moving those jobs onto our host as GitHub
jobs is separate work that needs a runner registration token for the repository
and root to install the bundle and the queue tool.

So this nightly uses the same pattern as the A5 sweep and the PyPTO
compatibility run: a host-side systemd timer that runs the downstream selection
directly on the reserved card and reports on Feishu. It is a nightly gate, not a
GitHub status check.

## What runs

`pypto_compat.sh --stage lib` adds two selections to the driver the PyPTO
compatibility nightly already uses:

| Key | Mirrors | Entries (main, 2026-11) |
| --- | --- | --- |
| `lib-a5-examples` | `ci.yml` `device (a5)` | 11 |
| `lib-a5-models` | `daily_ci.yml` `Model tests (a5)` | 32-34 (moves with main) |

The file list is produced by **pypto-lib's own**
`.github/scripts/detect_changes.py` against the checkout under test, not by a
copy kept here. The point is to reproduce the CI job; a list that drifts from it
reports cases CI never runs.

Measured on the board host, 2026-11 (ptoas 0.73, pypto-lib main): 29 pass, 6
unvalidated for the comm-domain reason below, 10 unvalidated for card count,
**15-26 min of case time** depending on card contention (examples ~2 min;
models 32-34 entries, and the count tracks pypto-lib main).

Two kinds of entry cannot be validated with one reserved card, and both are
reported as `未验证(key): <case> — <reason>` in `summary.md` and on the card --
never silently dropped, because a suite that shrinks reads as green:

- `# ci: devices=N` with N > 1: needs N cards, and this host reserves one
  (`examples/advanced/allreduce.py` plus 7-8 `decode_*` / `prefill_*` model
  entries, depending on the pypto-lib revision).
- A comm domain that cannot be initialized: `prefill_c1a_*` / `prefill_c2a_*` /
  `prefill_swa` abort in `comm_init` with `HcclCommInitRootInfo failed: 19`
  before any kernel runs. Verified under **root as well** as a normal user, so
  it is not a privilege issue: `/etc/hccn.conf` on this host is 0 bytes, i.e. no
  HCCL comm domain is configured. pto-as is not involved at that point, so the
  runner reports the entry as unvalidated with the error rather than red. Giving
  the host a comm domain makes those entries run; nothing in the repo changes.

## Attribution

A red entry is re-run with the baseline ptoas (the release PyPTO pins) before
the card goes out:

- candidate green -> 通过
- candidate red, baseline green -> 阻塞（pto-as 回归）
- both red -> 不阻塞（pypto-lib / host environment）
- no baseline installed -> 无结论（未归因）

The stage is deliberately not part of `--stage all`: a pypto-lib-side model
failure must not move the PTOAS release verdict computed by the PyPTO
compatibility run, so it runs as its own nightly and its own card.

## Host layout

Everything lives in the PyPTO compatibility root (`/data/ptoas-pypto` by
default), so both nightlies share the Python 3.10 environment and the ptoas
installs under test:

| Item | Value |
| --- | --- |
| pypto-lib checkout | `<root>/src/pypto-lib` (shallow, `main`, advanced each run) |
| Python env | `<root>/mamba/envs/pypto` |
| Candidate / baseline | `<root>/ptoas/v<ver>` (same as the compatibility run) |
| Results | `<root>/results-lib/<run-id>/`, `latest` symlink |
| Device | card 2 |
| Timer | `ptoas-pypto-lib-a5.timer`, daily 05:00 Asia/Shanghai |

05:00 is after the 00:00 A5 sweep and the 04:00 compatibility run. The unit
also waits for `ptoas-a5-nightly.service` and `ptoas-pypto-compat.service`
(`PTOAS_WAIT_FOR_UNITS`), so a slow night delays this run instead of colliding
with it for the card.

## Install

```bash
sudo .gitcode/scripts/install_pypto_lib_timer.sh \
  --repo-dir /data/ptoas-ci/src --pypto-root /data/ptoas-pypto
```

Render the units without touching the host:

```bash
.gitcode/scripts/install_pypto_lib_timer.sh --dry-run
```

Run one selection without sending a card:

```bash
PTOAS_PYPTO_ROOT=/data/ptoas-pypto \
  bash .gitcode/scripts/pypto_compat.sh --stage lib --only lib-a5-examples --no-notify
```

## Failure modes

1. **GitHub is unreachable** -- `update_pypto_lib` retries once, then records
   `clone-failed`/`fetch-failed` and continues on the existing checkout, exactly
   as a PyPTO fetch failure does. `github.com:443` from this host has been seen
   to time out for minutes while `git ls-remote` still worked an hour earlier.
2. **Card contention** -- another tenant on card 2 makes cases fail with a
   device error. Check `npu-smi info -t proc-mem -i 2` before reading a red
   night as a regression.
3. **pypto-lib main moves** -- a model-side failure shows up as `both red` in
   attribution, which points at pypto-lib rather than pto-as.
4. **Non-root manual runs against the installed env** fail on the root-owned
   `simpler_setup/_assets/build/.pto-isa.lock`. The unit runs as root; manual
   non-root runs need a writable copy of that lock path.
