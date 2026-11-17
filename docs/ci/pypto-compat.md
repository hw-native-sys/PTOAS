# PyPTO compatibility nightly

PTOAS is consumed by [PyPTO](https://github.com/hw-native-sys/pypto) as a
released wheel pinned in `toolchain/versions.env` (`PTOAS_VERSION` plus a
sha256 per architecture). PyPTO's own CI is what proves a PTOAS release is
usable downstream, but it only runs after the fact: when a PTOAS change breaks
PyPTO, the failure surfaces in PyPTO's repository, on PyPTO's schedule, with
PyPTO triaging it.

This nightly moves that signal upstream. It installs the newest published
pto-as release wheel (falling back to a branch build when the release has no
wheel for the host architecture) and runs the parts of PyPTO's gate that
exercise PTOAS, on our own hardware, so a regression is caught before it
becomes a PyPTO incident.

The goal is **not** "run PyPTO's whole CI". It is "run the PTOAS-dependent
subset, with attribution, cheaply enough to run every night".

Introduced on branch `ci/a5-nightly-board-migration` (the A5 nightly migration
MR, 354), on top of the same driver layout the A5 nightly uses.

## Host and schedule

Everything runs on the A5 board host `238` (`x86_64`, 8x Ascend950PR, CANN 9.1).

| Item | Value |
| --- | --- |
| Root | `/data/ptoas-pypto` |
| PyPTO checkout | `/data/ptoas-pypto/src/pypto` (git `main`, Python 3.10 env) |
| Candidate ptoas | `/data/ptoas-pypto/ptoas/v<version>` for the release named in `ptoas_release.txt` (`v073` today) |
| Fallback candidate | `/data/ptoas-pypto/ptoas/candidate` (shim over `/data/ptoas-ci/src/install`), only when the release wheel is unavailable |
| Baseline ptoas | the release PyPTO pins in `toolchain/versions.env` (`v072` today), used only to attribute a red candidate |
| Results | `/data/ptoas-pypto/results/<run-id>/`, `latest` symlink |
| Timer | `ptoas-pypto-compat.timer`, daily 04:00 Asia/Shanghai |
| Device | card 2 only |

04:00 is late enough that the 00:00 A5 board sweep is usually finished. The
service additionally waits for `ptoas-a5-nightly.service` to go inactive, so a
slow sweep delays the run instead of colliding with it for card 2.

Measured 2026-11-02: the A5 sweep ran 00:00:01–03:25:40, and a full compatibility
run took 5m55s (systemd, warm) to 7m33s (cold). Those numbers predate the growth
of PyPTO's parametrisation: the same five selections were 142 cases on 11-02 and
446 on 2026-11-09 (439 pass, 7 skip), a full run of about 13 minutes. Expect the
card around 04:14 today; when the sweep overruns, the run waits for card 2 and
then needs its own 13 minutes.

The sibling `ptoas-pypto-lib-a5.timer` runs the same driver with `--stage lib`
at 05:00 for pypto-lib's A5 gate; it waits for both units above and writes to
`results-lib/`. See [pypto-lib-a5.md](pypto-lib-a5.md).

## What runs

Selections mirror PyPTO's CI jobs instead of "all of `tests/st`". PyPTO's
simulator jobs run fixed file lists and markers; the rest of `tests/st` is
device-only or needs multi-card batching, and running it here reports failures
PyPTO never gates on.

| Selection | PyPTO job it mirrors | Time | Cases |
| --- | --- | --- | --- |
| `a5sim-core` | `system-tests-a5sim` file list (`-k "not TestMscatter"`) | ~80 s | 33 pass, 7 skip |
| `sim-fillpad` | `test_fillpad_inplace.py --memory-planner=ptoas` (both sims) | ~30 s | 18 |
| `a5sim-crosscore` | `test_cross_core.py --platform=a5sim` | ~4.5 min | 163 |
| `a2a3sim-cast` | `test_cast_saturation.py` on `a2a3sim` | ~70 s | 44 |
| `a5-board` | `onboard-tests-a5` five-file subset, card 2 | ~6 min | 181 |

The case counts are what the *current* PyPTO main parametrises, not a fixed
number: `a5sim-crosscore` was 11 and `a5-board` 29 until 2026-11, when PyPTO
expanded those files. Re-read the counts from a run instead of trusting the
table.

`--stage sim` runs the four simulator selections, `--stage a5` runs the board
selection, `--stage all` (default) runs everything. `--only KEY[,KEY]` narrows
it further.

## Candidate and baseline

The candidate is **the newest pto-as release**, because that is what PyPTO
consumes: it installs the wheel pinned in its `toolchain/versions.env`, and a
release that breaks PyPTO must be caught before PyPTO picks it up. The version
is a reviewed file in this repository, `.gitcode/scripts/ptoas_release.txt`:

```bash
$ cat .gitcode/scripts/ptoas_release.txt
0.73
```

Bump that file when a release lands; the nightly installs
`<root>/ptoas/v<version>` from the release wheel on the next run and keeps using
it. PyPTO's own `toolchain/versions.env` is read as well: when it names a newer
version than the pin, that newer one wins, so the nightly follows PyPTO as soon
as PyPTO starts consuming the release. The pin stays as the fallback for when
the checkout cannot be read. Wheel resolution goes through GitHub and its
mirrors with retries, and the sha256 is checked against the digest the release
API reports. Autodiscovery is deliberately not used: the release API seen from
this host does not expose every published tag (v0.72 was invisible while v0.70
was listed), so a pin file is the only reliable way to say "this is the version
we are testing".

### Release cadence and the promotion rule

Releases currently land roughly **twice a week**, and the A5 gates run one
release. The order is fixed:

1. A release candidate appears with its wheels published (not a draft -- check
   that an anonymous request for the asset returns 200, see below).
2. Run this gate and the pypto-lib stage against the candidate release, before
   announcing it. A hand-run with `--candidate` or a bumped pin both count, as
   long as both stages are green; a red one is attributed with the baseline.
3. Only after that verification passes, bump `ptoas_release.txt` and **commit
   and push it**. The A5 CI is then on the new version from its next run.

Bumping first inverts the purpose of the nightly: the A5 card would become the
first place a regression appears, after the version was already announced. The
pin must be pushed, not just edited: both units run the repository checkout and
`git checkout -f` restores whatever the pin file says in git, so an unpushed
bump is silently reverted on the next run (this happened between 0.72 and 0.73,
which the 11-07 and 11-09 cards still tested as 0.72).

The pin says "what upstream announced", which is not always "what is
machine-downloadable". v0.72 was announced on 2026-11-02 and stayed a **draft**
until later on 2026-11-03: the tag page returned 404, both asset URLs 404'd
through every mirror, and `git ls-remote --tags` on GitHub and GitCode both
stopped at `v0.70`, while a logged-in maintainer saw "0.72" on the releases page
linking to `/releases/tag/untagged-b84bfa9bd1d699fc8769`. It was published the
next day, and PyPTO now pins `v0.72` itself in `toolchain/versions.env`, so this
nightly runs one release ahead of its consumer (0.73).

The reason that distinction matters is worth keeping in mind for the next
release: GitHub shows drafts only to users with write access, and a draft with
no tag yet carries the temporary ref `untagged-<hash>`. The real tag appears
when someone presses *Publish release*, and only then do the asset URLs,
`git ls-remote --tags`, and therefore 238/68.6 see anything. The question that
decides whether this nightly can test a version is not "is it on the releases
page" but "does an anonymous request for the asset return 200".

Artifacts of the current pin (v0.73, cp310 wheels, published 2026-11-07):

| Arch | Size | sha256 |
| --- | --- | --- |
| x86_64 | 31,225,314 | `065660abf13fc6b5622df7a84e984b0b83f9340f9ef6b01e59db8ec1ef96dfee` |
| aarch64 | 30,159,225 | `3c902bcdba051c2214bf621f797668ad036b89e1ca0ae8f30b7285229d9ddccb` |

Artifacts of v0.72, for reference:

| Arch | Size | sha256 |
| --- | --- | --- |
| x86_64 | 31,209,537 | `5f82586e9e211a890bd4b625848a731339a7e6bb320c6f5640558452f36cbf94` |
| aarch64 | 30,139,815 | `618d7ffcbc70d7e0d6997556205140f27a62838de795d20d306c74cbdc227216` |

The wheel fetched on 11-02 and tested on 238 has a different digest
(`848df643...3b3e2`) but every one of its 463 archive entries hashes identically
to the published x86_64 wheel -- only the zip timestamps differ, so the tested
build and the published build are the same content.

When the pinned wheel cannot be fetched the nightly first uses a wheel placed by
hand in `<root>/ptoas/` (same file name), then falls back to the branch install
(`<root>/ptoas/candidate`); the card says `[release <version> unavailable;
testing branch build]` instead of producing no result at all.

`--candidate DIR` (or `PTOAS_CANDIDATE_ROOT`) overrides the pin for a one-off
run against any tree -- for example the branch build that the A5 nightly leaves
in `/data/ptoas-ci/src/install`. That shim runs the install with the CPython
3.12 interpreter it was built for:

```bash
#!/bin/bash
export LD_LIBRARY_PATH=<llvm-shared-lib>:/data/ptoas-ci/src/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec /usr/bin/python3.12 /data/ptoas-ci/src/install/bin/ptoas "$@"
```

`baseline` stays at the release PyPTO currently pins (`v072` today; the script
defaults to `/data/ptoas-pypto/ptoas/v070` and is moved to PyPTO's version when
that release is installed). It exists only for attribution: if a selection
fails with the candidate, the same selection is re-run with the baseline, so a
failure that already exists in the pinned release is not reported as a new
regression.

The PyPTO checkout is refreshed on every run (`git fetch` + fast checkout of
`origin/main`, then `pip install` of the tree and its `runtime`). The card shows
the resulting head and one of `unchanged` / `updated` / `fetch-failed` /
`checkout-failed` / `rebuild-failed`. A fetch or rebuild failure keeps the
previous revision and still produces a result, so a network problem on one night
does not erase the nightly.

## A3 half

> The 68.6 development host described below is retired (2026-11-12). The
> replacement is the self-hosted A3 runner driven by
> `.gitcode/workflows/a3_pypto_smoke.yml`; this section is kept for the
> selection and result-file contract that the runner port reuses.

PyPTO's A3 (`a2a3`) gate runs on Ascend910, which is a separate host
(`101.245.68.6`). The board host cannot reach it inbound, so the direction is
reversed: the A3 host runs first and pushes its result to the board host, and
the 04:00 run folds it into the same card.

| Item | Value |
| --- | --- |
| Script | `.gitcode/scripts/pypto_compat_a3.sh` |
| Root | supplied by `PTOAS_PYPTO_ROOT` on the A3 host |
| Result pushed to | `PTOAS_PUBLISH_HOST:$PTOAS_PUBLISH_DIR` (publishing disabled when unset) |
| Schedule | user crontab, 22:00 and 01:00 CST (06:00/09:00 host time) |
| Card line | `A3(68.6): <VERDICT> a3-direct OK=n FAIL=n SKIP=n | ptoas X | pypto <sha> | <time>` |

The A3 host has no GitHub route, so it pulls the PyPTO tree from the board host
over the same SSH channel it publishes on and rebuilds it there; if that fails
it keeps its current tree and says so.

The A3 half does not have to run just before the card. It counts if it finished
**after the previous card** (the board host writes `state/a3.consumed` when it
builds one) and is no older than 14 h (`PTOAS_A3_MAX_AGE`). So the A3 job can
run early in the night and the board run picks up whatever is there; a silent A3
host, or a result already used by the previous card, renders `未就绪` instead of
a stale green.

The A3 candidate follows the same release pin. When the release has no aarch64
wheel installed under `ptoas-v<version-without-dots>` (`ptoas-v073`), the
runner falls back to the branch
build in `pto-as-install` and labels the card `release X not installed for
aarch64; testing branch build` -- an honest label, not a silent substitution.

Install it as a user crontab on that host (no root there):

```bash
R=<the A3 compat root, PTOAS_PYPTO_ROOT>
install -m 0755 pypto_compat_a3.sh "$R/pypto_compat_a3.sh"
printf '0.73\n' > "$R/ptoas_release.txt"
(crontab -l 2>/dev/null; echo "0 6,9 * * * $R/pypto_compat_a3.sh >> $R/cron.log 2>&1") | crontab -
```

That host runs on `America/Los_Angeles`, sixteen hours behind the board host, so
06:00 there is 22:00 CST and 09:00 is 01:00 CST -- well before any plausible
card, which is all the board run needs. The second shot is a retry: the runner
exits immediately when a result was produced in the last three hours
(`PTOAS_A3_FORCE=1` bypasses that), so the two scheduled shots do not overlap in
practice (the script takes no lock of its own).

The host is shared with other tenants, so `task-submit` may wait hours for a
card; `PTOAS_DEVICE_WAIT` defaults to 7200 s and a timeout reports `NO_DEVICE`
rather than hanging the night. The queue also enforces its own runtime cap
(300 s by default, which killed the first real run), so the submission passes
`--max-time 1800`.

Two build details on this host are not optional:

- The runtime must be rebuilt with CANN sourced (`set_env.sh` and
  `ASCEND_HOME_PATH`). Without it the build silently skips the on-board variants
  and `build/lib/pto_isa_build.json` is never written; every on-card test then
  fails with `Missing PTO-ISA build metadata`.
- `simpler_setup/_assets/build/pto-isa` must contain the checkout the runtime
  pins. With GitHub unreachable an empty directory is not "missing" to the
  resolver: it starts a clone that hangs until the queue kills the job. The
  runner seeds it from the synced source tree whenever it is empty. The A3 half pulls whatever revision
the board host holds at that moment; because the board host refreshes its
checkout as part of the 04:00 run, a same-night A3 result normally runs the
previous day's PyPTO commit. Both heads are printed on the card, so the gap is
visible rather than implied.

## Verdicts

| Verdict | Meaning | Card |
| --- | --- | --- |
| `GREEN` | every selection passed with the candidate | ✅ 可发版 |
| `REGRESSION` | candidate red, same selection green with the baseline | ❌ 阻塞发版 |
| `PYTPO_ISSUE` | candidate red and baseline red too — not a pto-as regression | ⚠️ 不阻塞 |
| `UNATTRIBUTED` | candidate red and no baseline installed | ⛔ 无结论 |

The card goes to the existing Feishu group `PTOAS A5 nightly CI`, prefixed
`[PTOAS × PyPTO 兼容]`, and lists per-selection counts plus the first failing
cases. `--no-notify` renders a card without sending it.

## Install and operate

```bash
# units (idempotent)
bash .gitcode/scripts/install_pypto_compat_timer.sh \
  --repo-dir /data/ptoas-ci/src --pypto-root /data/ptoas-pypto

systemctl list-timers ptoas-pypto-compat.timer
journalctl -u ptoas-pypto-compat.service -n 100

# manual run (respects the A5 nightly only when PTOAS_WAIT_FOR_UNITS is set)
PTOAS_PYPTO_ROOT=/data/ptoas-pypto \
  bash /data/ptoas-ci/src/.gitcode/scripts/pypto_compat.sh --stage all

# bounded re-run while debugging
./pypto_compat.sh --stage sim --only a5sim-crosscore --no-notify
```

Environment knobs:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PTOAS_PYPTO_ROOT` | `/data/ptoas-pypto` | environment and results root |
| `PTOAS_CANDIDATE_ROOT` | unset | explicit ptoas under test; unset resolves to the newest release wheel, with the branch build as the fallback |
| `PTOAS_BASELINE_ROOT` | `<root>/ptoas/v070` | attribution baseline |
| `PTOAS_WAIT_FOR_UNITS` | empty | systemd units to wait out first |
| `PTOAS_A5_DEVICE_ID` | `2` | card used by `a5-board` |
| `A5_FEISHU_WEBHOOK_URL`, `A5_FEISHU_SECRET` | – | shared notifier, `/etc/ptoas-a5-ci.env` |
| `PTOAS_A3_INCOMING` | `<root>/incoming/a3` | where the A3 host publishes its result |
| `PTOAS_A3_MAX_AGE` | `50400` | seconds before an A3 result counts as not ready |

## Environment notes

The PyPTO environment is a conda env (Python 3.10) with PyPTO `main` and its
`runtime` submodule built in place. Two non-obvious requirements:

- The runtime must be built with CANN on `PATH` and `ASCEND_HOME_PATH` set, or
  `detect_buildable_platforms()` skips the on-board variants and
  `build/lib/pto_isa_build.json` is never written. Without that file every
  on-board lookup fails with `Missing PTO-ISA build metadata`.
- Conda's compiler package exports `CFLAGS`/`CPPFLAGS` with x86 tuning flags
  that the AICPU cross-compiler rejects (`-march=nocona`). Clear
  `CFLAGS CXXFLAGS CPPFLAGS LDFLAGS` and put a system `gcc`/`g++` shim first on
  `PATH` before installing `runtime`.
- PyPTO's on-board code needs CANN 9.2 headers (`comm_hccl.cpp` uses the
  channel-config API that 9.1 does not declare), so the units point
  `PTOAS_CANN_ENV` at a side-by-side SDK under `cann-rel/` (release 9.2.0,
  installed from `Ascend-cann-toolkit_9.2.0` + `Ascend-cann-950-ops_9.2.0`).
  A 2026-10-15 internal build used before that shipped an unsigned
  `opp/built-in/op_impl/aicpu/kernel/asccomm_aicpu.tar.gz`; the card runs with
  package signature verification enabled, so every `rtSetDevice()` failed with
  `507033` / `Package_Error_Verify_Package(E30009)`. The release builds on this
  host (2026-10-08, 10-10) and the 9.3.0 weekly (10-28) do not contain that
  package at all — check the `cann-hcomm` sub-package before trusting any
  side-by-side tree. Do not turn off signature verification on a shared host.
- The 9.2.0 release tree also bundles its own `pypto` (0.2.1) under
  `python/site-packages`, and CANN's `set_env.sh` puts that directory on
  `PYTHONPATH`, so it shadows the PyPTO under test. In the host copy the
  `pypto`, `pypto-0.2.1.dist-info` and `pypto_pro` entries are renamed to
  `*.cann-bundled`; redo that after a reinstall and re-check
  `python -c "import pypto; print(pypto.__file__)"`.

## Coverage and limits

Covered today:

- PyPTO's A5 on-board gate, on real hardware, in the same CANN environment its
  own A5 runner uses.
- The simulator selections PyPTO runs for A5 and A2A3 code paths, plus the
  PTOAS-planner `fillpad` and `cast_saturation` cases.

Not covered (needs resources we do not have yet):

- PyPTO's A3 (`a2a3`) `system-tests` / `system-tests-direct` shards in full.
  A bounded in-process selection runs there today (see "A3 half"); the batched
  shards need PyPTO's own device batching, which is a separate step. The host
  environment is prepared under `PTOAS_PYPTO_ROOT` (a writable workspace on that
  host):
  - `pto-as-shallow` + `pto-as-build` + `pto-as-install`: the branch builds with
    the VPTO LLVM 19 fork already on the host
    (`ptoas-board-monitor/cache/llvm-project-vpto-feature/build-shared`, 19.1.7).
    Upstream LLVM 19 lacks `CallingConv::SimtEntry` and LLVM 21 is rejected by
    `CMakeLists.txt`, so only that fork works.
  - `venv`: isolated (no system site packages -- a `~/.local` editable `.pth`
    for another project shadows `pypto`/`simpler`), pypto + runtime built, the
    pinned pto-isa checkout seeded into `simpler_setup/_assets/build/pto-isa`
    because GitHub is unreachable there.
  - Collection matches the A5 host: `a2a3sim 1721/3952`.
  - One on-card smoke (`test_tri` FP16) passes via `task-submit --device auto`;
    the command must export `ASCEND_HOME_PATH` itself, the task sandbox does not.
  - Unattended runs use a user crontab (the host has no root); the publish key is
    `~/.ssh/id_ed25519_ptoas238` and does not need a Feishu secret there,
    because the board host sends the single card.
- PyPTO's full `tests/st` device matrix — needs batched task-submit execution
  against a card pool.
- PyPTO's serving jobs — they need model weights and an A3 card pool.
- `pypto-lib` — covered by a separate nightly (`pypto_compat.sh --stage lib`,
  see [pypto-lib-a5.md](pypto-lib-a5.md)). It runs as its own unit and card so
  a pypto-lib-side model failure cannot move this run's release verdict.

Because the simulator selections are a curated subset, a green night means "the
PTOAS-dependent gate PyPTO runs is still green with this branch", not "PyPTO is
green everywhere".
