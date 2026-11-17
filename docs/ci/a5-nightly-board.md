# A5 nightly board validation (GitCode CI)

A5 (Ascend950PR) board validation used to run on board host `52`. That host was
powered off over the national holiday and its accelerator cards failed on
power-up, so the nightly moved to the shared A5 machine `238` (ask the CI owner
for its access details; they are deliberately not recorded here).

This document covers the GitCode pipeline, the automation script, the host
prerequisites, and the one-time setup that the infrastructure/engineering CI
owner has to perform.

## What runs, and when

| Item | Value |
| --- | --- |
| Workflow | `.gitcode/workflows/a5_nightly_board.yml` |
| Driver | `.gitcode/scripts/a5_board_nightly.sh` |
| Schedule | the host `systemd` timer runs at **00:00 Asia/Shanghai**; the workflow itself is `workflow_dispatch`-only until a GitCode runner with `ptoas-a5` is registered (its cron is then `0 16 * * *` UTC) |
| Manual trigger | `workflow_dispatch` (`device`, `skip_cases`, `run_only_cases`, `pto_isa_commit`, `allow_daytime`) |
| PR / PR-comment trigger | **none by design** |
| Runner label | `self-hosted` + `ptoas-a5` |
| Job timeout | 600 min (validation step 450 min) |

The pipeline is deliberately independent from `pto-as_action.yml`: it defines
only `workflow_dispatch` today (and `schedule` once a GitCode A5 runner exists),
so a `/compile` comment on a pull request can never start a board run.

The nightly starts at 00:00 because the machine is quietest overnight; the
full sweep needs about four hours, so results land around 04:00-05:00. Three
guard checkpoints watch it (00:05 / 02:00 / 04:30) and a final watchdog runs at
08:00; see "Failure handling" below.

The driver refuses to start between 06:00 and 22:00 local time. If the platform
ever stops treating cron as UTC, or the host timezone is not UTC+8, the 16:00
UTC slot lands in the working day; the driver then fails fast and alerts
instead of occupying card 2 during working hours. Set `allow_daytime=true` only
for a manual daytime triage run.

## Host and architecture notes

`52` was ARM64; `238` is x86_64 with 8x Ascend950PR (CANN 9.1.0, driver
25.7.rc1). This does not change what is validated:

* PTOAS is built natively on the board host. Both `x86_64` and `aarch64` are
  supported by `build.sh` and by `run_remote_npu_validation.sh` (it resolves
  the CANN host library directory per architecture).
* Device code is always cross-compiled for the NPU by the CANN toolchain
  (`bisheng` / `bishengir-compile-a5`), so the host CPU architecture only
  affects the host-side binaries, not the generated kernel.
* The only architecture-sensitive historical assumption was the GitHub
  workflow's `runs-on: [self-hosted, Linux, ARM64, ptoas-a5]` plus its
  `uname -m == aarch64` guard. Both are gone here; the new workflow matches the
  machine by the `ptoas-a5` label instead.

## Host prerequisites (what engineering CI needs installed)

All of the following are already present on `238` except the Python packages
that the one-time setup installs into `${PTOAS_CI_ROOT}/venv` (the driver only
reuses that virtualenv; it does not create it).

| Requirement | Version on 238 | Notes |
| --- | --- | --- |
| OS / CPU | Ubuntu 24.04, x86_64 | ARM64 is equally supported |
| NPU driver + firmware | 25.7.rc1 / 9.0.0.100.200 | `npu-smi` must report A5 cards |
| CANN toolkit | 9.1.0 (`/usr/local/Ascend/ascend-toolkit/latest`) | must ship `bisheng`, `bishengir-compile-a5`, `ccec` |
| `cmake` | 3.28.3 (>= 3.20) | |
| `ninja` | 1.11.1 | |
| `gcc` / `g++` | 13.3.0 | C++17 required |
| `git`, `patch`, `tar`, `flock`, `rsync` | present | `rsync` falls back to `tar` if absent |
| `ccache` | present (optional) | cache lives under `${PTOAS_CI_ROOT}/cache/ccache` |
| Python | 3.12 venv at `/data/ptoas-ci/venv` | see package list below |
| Disk | `/data` (197 GiB free) | a cold run needs ~30 GiB; `/` only has ~19 GiB, so **never** build under `/` |

Python packages (the actual answer to "which Python packages does the board
test depend on"):

```
numpy            # sample generation and golden comparison
pybind11==2.13.6 # REQUIRED: MLIR 19 bindings do not compile against pybind11 3.x
```

`pybind11` is only needed to build the MLIR Python bindings; `numpy` is needed
both at configure time (`Python3_NumPy_INCLUDE_DIRS`) and at runtime.

## One-time host setup

```bash
# 1. Working root on the large filesystem
mkdir -p /data/ptoas-ci/{3rd-party,work,logs,cache,locks,state}

# 2. Python environment
python3 -m venv /data/ptoas-ci/venv
/data/ptoas-ci/venv/bin/pip install 'numpy==1.26.4' 'pybind11==2.13.6'

# 3. cann-cmake (small helper repo used by CMake)
git clone --depth 1 -b master-055 https://gitcode.com/cann/cmake.git \
  /data/ptoas-ci/3rd-party/cann-cmake

# 4. vpto LLVM patch. The GitCode release URL returns 401 for anonymous
#    downloads, so cache it once on a machine that can reach it.
curl -L -o /data/ptoas-ci/3rd-party/feature-vpto-last3.patch \
  https://gitcode.com/cann-src-third-party/llvm/releases/download/19.1.7-h0/feature-vpto-last3.patch
sha256sum /data/ptoas-ci/3rd-party/feature-vpto-last3.patch
# expected: a49c1d3dd8ab78e93264712bc0d46deb536196a54abb2c2ee02abd914cd385e2

# 5. GitCode read-only credential. Both paths use it: the host-side timer
#    through PTOAS_GIT_CREDENTIALS_FILE, and the GitCode workflow through the
#    same variable. Use a CI-owned file, not ~/.git-credentials: on a shared
#    host that file usually holds several tenants' tokens for gitcode.com and
#    git's store helper answers with the first match, i.e. the wrong account.
printf 'https://<user>:<read-only-token>@gitcode.com\n' > /data/ptoas-ci/.git-credentials
chmod 600 /data/ptoas-ci/.git-credentials

# 6. Host checkout for the fallback timer. The GitCode runner path uses its
#    own checkout and mirrors it into /data/ptoas-ci/src; this clone is only
#    needed when the systemd timer is the scheduler.
git clone https://gitcode.com/cann/pto-as.git /data/ptoas-ci/src
```

The first run additionally downloads the LLVM 19.1.7 source tarball from the
public OBS mirror and builds LLVM/MLIR in
`/data/ptoas-ci/3rd-party/lib_cache/llvm_19.1.7/`. That takes roughly 30-60
minutes on 238; later runs are incremental.

## Runner registration (infrastructure / engineering CI)

The scheduled workflow needs a **host runner** on `238` carrying the custom
label `ptoas-a5`:

1. Project (or organization) → **Settings → Actions → Runners → Host Runner**.
2. Add a runner named e.g. `ptoas-a5-board`, OS image `Ubuntu`, custom label
   `ptoas-a5`.
3. Run the generated install script on `238` with `sudo`. It installs the
   octopus runner service and registers it.
4. Confirm the runner is **online** and that its label set contains
   `self-hosted` and `ptoas-a5`.
5. The runner service user must be able to read `/dev/davinci*`, run
   `npu-smi`, write `${PTOAS_CI_ROOT}`, and load the CANN environment. The
   simplest setup on `238` is root, which is what the systemd units use too.
6. Configure the repository secrets/variables:

   | Kind | Name | Purpose |
   | --- | --- | --- |
   | secret | `GIT_TOKEN` | checkout |
   | secret | `A5_FEISHU_WEBHOOK_URL` | result card / failure alert |
   | secret | `A5_FEISHU_SECRET` | optional, only with 签名校验 |
   | secret | `AK`, `SK` | optional, only with OBS upload |
   | variable | `A5_BOARD_OBS_UPLOAD=true` | optional OBS upload |
   | variable | `PTOAS_A5_DEVICE_ID` | optional; default card 2 |

GitCode quirk found with the workflow validator: a step `if` cannot read
`vars.*` (`if表达式无法解析 ... 不支持的关键字`), although the context reference
lists it. The OBS upload step therefore tests `env.A5_BOARD_OBS_UPLOAD`, which
the workflow mirrors from the variable at workflow level.

The runner work directory lives under `/opt/runner_*/` on a small filesystem.
That is expected: the driver mirrors the checkout into
`/data/ptoas-ci/src` and keeps every build artifact there. Do not move
`PTOAS_CI_ROOT` onto `/`.

### What a runner is, and what it does not replace

A runner is an **agent process on `238`**, not a cloud machine: GitCode only
schedules the job and streams the logs. The A5 cards, the build, the driver and
the payload all still run on `238`. What you can remove once the runner is
stable is the host-side `systemd` nightly timer; keep the 08:00 watchdog. The
three guard timers call `systemctl` and are specific to the host-timer path, so
disable them when the runner schedules the job. Both paths take the same
`flock`, so a duplicate start exits instead of sharing card 2.

### First run with the runner

1. Merge this workflow to the default branch. It ships without a `schedule`, so
   merging cannot queue jobs while no A5 runner exists; `workflow_dispatch` is
   read from the default branch.
2. Run `workflow_dispatch` with `device=2` and
   `run_only_cases=VectorAddition` (plus `allow_daytime=true` if you test
   during the day). This exercises checkout, credentials, build, card
   selection, one case and the Feishu card.
3. Check the run page and the summary; then run a full manual sweep.
4. Only after the full sweep passes, add the `0 16 * * *` trigger and disable
   the host timer in the same change.

## Manual run

```bash
# Full path (build + payload + board run), auto-selecting a healthy idle card
/data/ptoas-ci/src/.gitcode/scripts/a5_board_nightly.sh \
  --workspace /data/ptoas-ci/src --no-sync --ci-root /data/ptoas-ci

# Only verify/prepare the toolchain
.../a5_board_nightly.sh --prepare-only

# Triage a subset on a specific physical card
.../a5_board_nightly.sh --device 6 --run-only Qwen3-8B-A5/decode_projection \
  --no-build
```

The four entry-point scripts are executable in git. If an older checkout still
has mode 0644, prefix the commands with `bash`.

Manual runs default to `--scope manual`, so they write
`state/last-manual.env` and never overwrite the nightly state that the
watchdog and guard read. `--prepare-only` writes `state/last-prepare.env`.

`--device auto` only accepts a card whose `npu-smi` health is `OK` **and** that
has no running processes, so the nightly never preempts another user on this
shared machine. If every healthy card is busy the run fails with a clear
message instead of sharing a card.

The shipped timer and workflow pin `--device 2`, the card reserved for PTOAS CI
on 238. Pinning matters for runtime, not just politeness: a card shared with
other tenants adds seconds of device-open/TSD startup latency to every case,
and the suite runs one process per case. Measure a single case with
`.../a5_board_nightly.sh --device 2 --run-only Abs --no-build` before changing
the pin.

## Offline / no-runner fallback (systemd timer)

If the GitCode runner cannot be registered immediately, the same driver can be
scheduled from the host itself:

```bash
# Requires the host checkout from one-time setup step 6.
sudo /data/ptoas-ci/src/.gitcode/scripts/install_a5_nightly_timer.sh \
  --repo-dir /data/ptoas-ci/src \
  --ci-root /data/ptoas-ci

# Dry-run: render the units into a temp directory and print them (no root).
bash /data/ptoas-ci/src/.gitcode/scripts/install_a5_nightly_timer.sh \
  --repo-dir /data/ptoas-ci/src --ci-root /data/ptoas-ci --dry-run
```

This installs `ptoas-a5-nightly.timer` (`OnCalendar=*-*-* 00:00:00`) running
`.gitcode/scripts/a5_nightly_cron.sh`, which updates the checkout and then
calls the driver. The units load `/etc/ptoas-a5-ci.env`; put the Feishu
webhook/secret (and `PTOAS_GIT_URL` if using a mirror) there and
`chmod 600` it. Run only one of the two schedulers at a time; both paths take
the same `flock` on `${PTOAS_CI_ROOT}/locks/nightly.lock`, so a duplicate start
exits immediately rather than corrupting a run. When the runner takes over,
disable the nightly timer **and** the three guard timers; keep the watchdog.

### Which code a run actually validates

The nightly is a regression signal for `master`, not for the CI branch that
carries the driver:

* **PTOAS source.** The runner path checks out the default branch, so it is
  current by construction. The host-timer path tracks `PTOAS_GIT_REF` and then
  merges the canonical `master` (`PTOAS_UPSTREAM_URL`, default
  `https://gitcode.com/cann/pto-as.git`) into a throwaway local tree before
  every run; that merge is never pushed. The upstream is fetched separately on
  purpose: 238's `origin` is a personal fork used for authentication, and a
  fork's `master` can be weeks behind, so merging `origin/master` would quietly
  reproduce the same stale-snapshot problem. A conflict fails the run with an
  alert instead of validating stale code. The commit written to the card and to
  `state/last-run.env` is the canonical master commit under test.
* **pto-isa.** The nightly tracks the newest `cann/pto-isa` master commit
  instead of the repository pin (`PTO_ISA_TRACK_LATEST=1`, the default). The
  pin advanced only through a PR plus a verified run, which lagged the isa
  fixes the board cases need by weeks; pto-isa runs its own daily guard, so a
  bad isa commit shows up there first. The commit actually used is written to
  `state/last-run.env` as `PTO_ISA_COMMIT` and logged at the start of the board
  stage, so a red night can be attributed without guessing.

  Set `PTO_ISA_TRACK_LATEST=0` (or pass `--pto-isa-commit <sha>`) to go back to
  the pinned value. If the remote cannot be reached, the driver logs that and
  falls back to the pin rather than skipping the night. The pin itself is still
  maintained for the other CI paths:

  ```bash
  python .github/scripts/update_pto_isa_pin.py --target gitcode-default \
    --commit <40-hex-sha>
  ```

  The script refuses a non-descendant update and rewrites every
  `gitcode-default` location consistently (`ci.yml`, `Dockerfile`, the
  compile-only guide, and `run_remote_npu_validation.sh`).

## Failure handling

A nightly that dies before the board stage used to be silent — under the old
03:00 schedule a run exited after one second with a git auth error and nobody
noticed until someone asked the next morning. Three mechanisms now cover that:

* **State file** — the driver writes `${PTOAS_CI_ROOT}/state/last-run.env` for
  `--scope nightly` (the cron/workflow path) and `last-manual.env` /
  `last-prepare.env` for CLI/manual/prepare runs. It records `SCOPE`,
  `RUN_DATE`, `RUN_ID`, `START`, `END`, `RC`, `REPORTED`, `DEVICE` and
  `WORK_ROOT` at start (`RC=running`) and on every exit path, including
  SIGTERM. Only `last-run.env` is watchdog/guard input, so a manual subset or
  `--prepare-only` cannot mask a failed nightly. A failed run also drops
  `work/<run>/FAILED` with the exit code and log path.
* **Failure alert** — when `A5_FEISHU_WEBHOOK_URL` is set, every non-zero exit
  (and every failure before the driver starts, e.g. the checkout) posts one
  short text message with the run id and log path. Nothing is sent when the
  variable is unset; the state file and log remain.
* **Watchdog** — `ptoas-a5-nightly-watchdog.timer` fires at 08:00 and runs
  `a5_nightly_cron.sh --check-last-run 7 --max-success-age-hours 24`. It
  reports *missing* (no nightly record), *not-nightly* (only a manual/prepare
  record), *stale-record* (the newest record is not from tonight), *failed*
  (`RC!=0`), *stuck* (`RC=running` for more than 7 h) or *stale* (last success
  ended more than 24 h ago) and alerts the same way.

  The *stale-record* check exists because of 2026-10-30: a call to
  `prune_heavy_artifacts` sat above the function's own definition, so every
  start exited 127 within seconds, before the driver could write its state. The
  watchdog then read the previous night's `RC=1, REPORTED=1` record, concluded
  "the run already sent its own card", and stayed silent — the one failure it
  existed to catch. Two changes prevent a repeat: the driver installs its state
  file and EXIT trap before any fallible work, and the watchdog compares the
  record's start date with today before it looks at `RC`.

### Three-stage guard

`ptoas-a5-guard@.service` runs `a5_nightly_guard.sh --stage {start,middle,end}`
at three checkpoints. It only takes actions that can be automated; everything
else becomes an alert.

| Stage | Time | What it does |
| --- | --- | --- |
| `start` | 00:05 | Waits up to 5 min for the 00:00 run to appear, starts it if the timer did not, then watches the run for 30 minutes. A run that died there is restarted up to twice, each restart observed for 10 minutes. A run that already finished successfully is left alone. |
| `middle` | 02:00 | If running, compares cases completed with elapsed time (a 710-case sweep should be ≈50 % through) and reports "slow" under 30 %; if it already exited with a non-zero rc — or there is no current-night record — restarts it and watches the restart for 10 minutes before giving up. |
| `end` | 04:30 | Waits for completion (up to 3 h), then reports PASS with the OK/FAIL/SKIP counts and the first failing cases, or FAIL with the same detail. A previous night's (or a manual run's) PASS is never reused. |

The guard only applies to the host-timer path (it calls `systemctl`). With the
GitCode runner, keep the state-file watchdog and leave the guard timers
disabled; otherwise the guard sees no host service and raises false alarms.

The guard never holds the nightly's lock and never runs cases itself. It calls
`systemctl start --no-block`, so a restart is a fresh run: progress from the
dead attempt is lost. Resuming from the last completed case would need the
driver to reuse a previous payload (`--reuse-work`), which is not implemented —
a 02:00 restart therefore finishes around 06:15 instead of 04:15.

Two smaller retries: `--wait-idle MIN` (default 30, `0` disables) waits for the
target card to become healthy and idle instead of failing instantly when
another tenant is on it, and the checkout retries a failed `git fetch` three
times before giving up.

### Feishu notifications

Every run posts one card to the CI group: run id, OK/FAIL/SKIP counts, the
failing cases with the concrete compare error, and a **not-validated** block
(SKIPs grouped by reason plus any case excluded before generation, with the
generated + not-generated = A5 inventory total). Early failures — before a
summary exists — and the guard/watchdog send a short text message instead; to
avoid duplicates only one of them speaks per event.

1. In the Feishu group: **设置 → 群机器人 → 添加机器人 → 自定义机器人**.
   A custom bot needs no tenant-admin approval; an app bot does.
2. Security setting: enable **签名校验**. A keyword such as `PTOAS` only proves
   the sender knew the word, while the signature also proves it holds the
   secret; both senders sign with
   `base64(HMAC-SHA256(key="<timestamp>\n<secret>", msg=""))` and stay backward
   compatible when the secret is unset.
3. Copy the webhook URL and the secret into `/etc/ptoas-a5-ci.env` on the board
   host (the service templates load it via `EnvironmentFile=-`):

   ```bash
   A5_FEISHU_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxxxx
   A5_FEISHU_SECRET=xxxxxxxx
   ```

   `chmod 600 /etc/ptoas-a5-ci.env`, then `systemctl daemon-reload`. Both values
   are credentials: keep them on the host only — never in the repo, an issue or
   this document. (Only a bot configured without 签名校验 may omit the secret
   line.)
4. Verify that systemd parsed the file; `EnvironmentFile=-` makes a *missing*
   file non-fatal, but it does not validate the contents:

   ```bash
   systemd-run --unit=probe --collect --wait \
     --property=EnvironmentFile=-/etc/ptoas-a5-ci.env \
     /bin/bash -c 'echo url=${#A5_FEISHU_WEBHOOK_URL} secret=${#A5_FEISHU_SECRET}'
   journalctl -u probe -o cat
   ```

   `--wait` makes the probe finish before `journalctl` reads it; the command
   prints only the two lengths, never the values themselves.

5. Test without waiting for a run:

   ```bash
   set -a; . /etc/ptoas-a5-ci.env; set +a
   bash -c '. /data/ptoas-ci/src/.gitcode/scripts/a5_notify.sh; a5_notify_text "PTOAS A5 通知测试"'
   ```

   `a5_notify_text` swallows the HTTP body; to read Feishu's own verdict, POST
   the same payload with `curl` and require
   `{"StatusCode":0,"StatusMessage":"success"}`.

The GitCode workflow passes `A5_FEISHU_WEBHOOK_URL` and `A5_FEISHU_SECRET`
from repository secrets; the `/etc/ptoas-a5-ci.env` file is for the host-timer
path.

## Failure modes already seen on 238

| Symptom | Cause | Fix |
| --- | --- | --- |
| `def_property family does not currently support keep_alive` while building `_mlir` | pybind11 3.x with MLIR 19 bindings | pin `pybind11==2.13.6` |
| `libMLIRMlirOptMain.so.19.1 ... missing and no known rule` | interrupted LLVM build left a configured-but-incomplete cache; `build.sh` only checks CMake package files | the driver runs an incremental `cmake --build` on every cached LLVM tree before building PTOAS |
| `could not read Username for 'https://gitcode.com'` | anonymous access blocked, or a wrong tenant token was picked from `~/.git-credentials` | use the CI-owned file from one-time setup step 5; inherited credential helpers are disabled, and public repos also work anonymously |
| `No space left on device` | build produced under `/` (`/opt` runner work dir) | keep `PTOAS_CI_ROOT` on `/data` |
| payload generation exits non-zero without further output | `runop.sh` returns 1 when any sample fails, and `set -euo pipefail` aborts the driver | the driver now reports the failing sample log explicitly |
| `COVERAGE_MISSING` lists `DeepseekV4*/...A3/...` or `Qwen3_*A3/...` cases | `runop.sh` derives the model glob from `SOC_VERSION`; without it the A5 payload's expected list falls back to `*A3` | the driver exports `SOC_VERSION` when generating the payload (850 expected → 710) |

### Known sample breakage: `Img2col/timg2col_runtime.py`

Commit `4fd44233c` introduced this sample together with the conv ops, but the
sample was never buildable: it calls
`ConvTileBufType.get(..., conv_cfg, ctx)` although the type only has five
parameters (shape, element, bufferSize, memorySpace, layout). Commit
`2a2bad22d` later added the optional `valid_shape` parameter and made
`TImg2colOp`'s `pos_m`/`pos_k` keyword-only operands, which did not help. It
currently fails in three steps:

1. `ConvTileBufType.get(..., conv_cfg, ctx)` — `conv_cfg` was never a type
   parameter; the geometry belongs on `pto.alloc_tile {config = ...}`.
2. `TImg2colOp(dst, src, c1, c8)` — `posM`/`posK` are keyword arguments now.
3. After those two fixes the verifier rejects the `tload` pair: an A5 conv
   `tload` requires conv layouts on both sides (e.g. `nhwc -> nc1hwc0`), while
   the sample's GM view has no layout.

Fixing (3) changes how the sample's GM buffer is interpreted, so it must be
done together with `timg2col_runtime_golden.py`; that is the sample owner's
change, not a CI change. Until it lands, the driver passes
`timg2col_runtime` to `PTOAS_SKIP_CASES`
(`PTOAS_A5_GENERATION_SKIP_CASES` overrides the list) so one stale sample
cannot block the entire nightly. Remove the entry once the sample is ported.

**The exclusion is reported, never silent.** A skipped sample does not reach the
manifest, so the suite would look one case smaller. The driver writes
`work/<run>/not-generated.tsv` and the summary plus the Feishu card carry a
"not validated" block, so the inventory stays explicit:

```
710 generated (OK 655 / SKIP 54 / FAIL 1) + 1 not generated = 711 A5 cases
```

SKIPs are grouped by the reason the runner recorded (multi-card harness,
`SKIP_CASES`, EmitC incompatibilities, ...). Nothing is filtered out of the
report: the point of the nightly is to surface gaps, not to make the chart look
green.

## Verification performed on 238

2026-10-24, commit `b8fa2e93b`, x86_64 host, eight Ascend950PR cards:

* Cold LLVM/MLIR 19.1.7 build plus PTOAS build: ~45 min wall clock.
* `--prepare-only`: passes (CANN detected, python deps validated, cached LLVM
  completed, build environment loaded).
* Subset run (`--run-only VectorAddition`) on an A5 card:
  `OK 3 / FAIL 0 / SKIP 2`, board exit code 0, `compare passed`, summary
  rendered.
* Full run: 710 expected cases. Measured on 238, card 2, quiet host: 18-21 s
  per case for the unit samples (222 cases in 66 min) and roughly 20-30 s for
  the model samples, i.e. **4-5 hours** including payload generation. That is
  why the job budget is 600 minutes: the old 480-minute cap killed a run at
  607/850 on a contended card.
* `bash -n` on the five shell entry points: clean. (`shellcheck` was not
  available on the authoring host.)

A second, static verification pass on 2026-10-28 covered the current revision:
`bash -n` on all scripts, workflow YAML parsing, a 192-check local consistency
harness (guard/watchdog state transitions, installer `--dry-run`, driver
fail-fast paths, Feishu signature), and a cross-check against the GitCode
trigger/context/expression docs (cron is UTC and default-branch-only; missing
context properties evaluate to an empty string; `inputs` and `||` are
supported) and the shared CANN CI examples (`atomgit.workflow_run_id`,
`steps.process_checkout.outputs.path`). The workflow also passes GitCode's own
`actions/workflows/validate` API (`valid=true`); that validator caught the
unsupported `vars.*` use in a step `if`, which is now mirrored through `env`.
The GitCode runner path still needs one manual `workflow_dispatch` and one
scheduled run to validate the runner, secrets and the actual board execution.

The host-side systemd timer was installed during the same session
(`systemctl is-enabled ptoas-a5-nightly.timer` -> `enabled`, next run 00:00 CST).
Disable it — and the three guard timers — once the GitCode `ptoas-a5` runner is
online, so the nightly is not executed twice. Keep the watchdog.

## Results and retention

Per run, under `${PTOAS_CI_ROOT}/work/<YYYYmmdd-HHMMSS>/`:

* `remote_npu_validation_results.tsv` — per-case `status` (`OK`/`FAIL`/`SKIP`)
* `board-validation.log`, `sample-build.log`, `nightly.log`
* `summary.md` — rendered by the shared
  `.github/scripts/report_a5_board_results.py` (same formatter as the GitHub
  A5 midnight job), optionally pushed to the Feishu webhook
* `payload/` — generated samples and the vendored pto-isa
* `latest-artifacts/` — copies of the five files above for the current run;
  the GitCode upload step reads this directory so it never uploads an older
  run. It is refreshed on every exit path.

```bash
# newest run, and just the non-OK rows
run=$(ls -1dt /data/ptoas-ci/work/20*/ | head -1)
awk -F'\t' '$2!="OK"' "${run}remote_npu_validation_results.tsv"
grep -nE 'ERROR|COVERAGE_MISSING' "${run}board-validation.log" | head
```

The host-timer path also appends to `/data/ptoas-ci/logs/nightly-cron.log`
(the systemd unit's stdout), which is the place to look when a run dies before
it creates a work directory. With the GitCode runner the same files exist on
the host, and the run page additionally shows the step output.

`npu-validation/` and `payload/` are the only large trees (~17 GiB and ~66 MiB
per run); every other file above is a few megabytes. They are pruned from all
runs except the current one at both the start and the end of each run, so a
week of history costs megabytes instead of ~120 GiB. Set
`PTOAS_A5_KEEP_HEAVY_RUNS=N` to keep the heavy trees of the N newest earlier
runs for debugging. Whole run directories older than 7 days are removed at the
end of each run (`--retention-days`). The process exit code is the board
validation exit code.
