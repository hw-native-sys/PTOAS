# External Integrations

**Analysis Date:** 2026-03-18

## APIs & External Services

**Source Control & Release Distribution:**
- GitHub - Primary source host, Actions runner, and release asset source.
  - SDK/Client: `git`, GitHub Actions, GitHub Releases HTTP endpoints
  - Auth: `SSH_KEY` and `SSH_KNOWN_HOSTS` secrets in `.github/workflows/ci.yml`; standard GitHub token/permissions in `.github/workflows/build_wheel.yml` and `.github/workflows/build_wheel_mac.yml`
- GitHub Releases API - Installer resolves the latest release tag and downloads prebuilt tarballs in `install.sh`.
  - SDK/Client: `curl` or `wget`
  - Auth: none detected for public release access

**Upstream Toolchain Dependencies:**
- LLVM Project - Upstream source checkout for the required `llvmorg-19.1.7` toolchain in `.github/workflows/build_wheel.yml`, `.github/workflows/build_wheel_mac.yml`, `docker/Dockerfile`, and `README.md`.
  - SDK/Client: `git`
  - Auth: none detected
- PTO-ISA - External runtime/header dependency used for generated C++ compilation and remote validation in `docker/Dockerfile` and `test/npu_validation/scripts/run_remote_npu_validation.sh`.
  - SDK/Client: `git`
  - Auth: `PTO_ISA_REPO` can point to any Git remote; no credential variable is defined in-repo

**Container & Package Registries:**
- Quay.io `pypa/manylinux` images - Linux wheel build container source in `docker/Dockerfile` and `.github/workflows/build_wheel.yml`.
  - SDK/Client: Docker/OCI pull
  - Auth: none detected
- Quay.io Ascend CANN images - Runtime/on-device validation container source in `docker/Dockerfile`.
  - SDK/Client: Docker/OCI pull
  - Auth: none detected
- PyPI / pip ecosystem - Pulls Python dependencies such as `numpy`, `pybind11`, `nanobind`, `setuptools`, `wheel`, `auditwheel`, `delocate`, `pyyaml`, and `torch-npu` in workflow and Docker scripts.
  - SDK/Client: `pip`
  - Auth: none detected
- PyTorch wheel index - CPU Torch package source in `docker/Dockerfile`.
  - SDK/Client: `pip --index-url https://download.pytorch.org/whl/cpu`
  - Auth: none detected

**Remote Execution:**
- Remote Ascend NPU host over SSH/SCP - CI can upload payloads and execute validation on a remote machine in `.github/workflows/ci.yml`.
  - SDK/Client: `ssh`, `scp`
  - Auth: `SSH_KEY`, `SSH_KNOWN_HOSTS`

## Data Storage

**Databases:**
- Not detected
  - Connection: Not applicable
  - Client: Not applicable

**File Storage:**
- Local filesystem artifacts only for build output, wheelhouse, payload tarballs, and generated validation projects in `build/`, `install/`, `wheelhouse/`, and `test/npu_validation/`.
- GitHub Releases asset storage for published wheel/tarball artifacts in `.github/workflows/build_wheel.yml` and `.github/workflows/build_wheel_mac.yml`.

**Caching:**
- GitHub Actions cache for LLVM build directories in `.github/workflows/build_wheel.yml`, `.github/workflows/build_wheel_mac.yml`, and `.github/workflows/ci.yml`.

## Authentication & Identity

**Auth Provider:**
- Custom/none for the application itself. This repo is a compiler toolchain and does not expose end-user auth flows.
  - Implementation: CI-only SSH authentication for remote board access in `.github/workflows/ci.yml`

## Monitoring & Observability

**Error Tracking:**
- None detected

**Logs:**
- Build/test logs are emitted to stdout/stderr from shell scripts and GitHub Actions steps in `.github/workflows/*.yml`, `docker/*.sh`, and `test/npu_validation/scripts/run_remote_npu_validation.sh`.
- Remote validation writes a TSV summary file at `remote_npu_validation_results.tsv` in `test/npu_validation/scripts/run_remote_npu_validation.sh`.

## CI/CD & Deployment

**Hosting:**
- GitHub Actions for CI and release automation in `.github/workflows/build_wheel.yml`, `.github/workflows/build_wheel_mac.yml`, and `.github/workflows/ci.yml`.
- GitHub Releases for published wheels and binary tarballs in `.github/workflows/build_wheel.yml` and `install.sh`.

**CI Pipeline:**
- Linux wheel build and release pipeline in `.github/workflows/build_wheel.yml`
- macOS wheel build and release pipeline in `.github/workflows/build_wheel_mac.yml`
- Main build, sample translation test, payload packaging, and remote NPU validation pipeline in `.github/workflows/ci.yml`

## Environment Configuration

**Required env vars:**
- Build/package variables: `PTO_SOURCE_DIR`, `PTO_INSTALL_DIR`, `LLVM_BUILD_DIR`, `LLVM_DIR`, `MLIR_DIR`, `Python3_EXECUTABLE`, `pybind11_DIR`, `MLIR_PYTHON_PACKAGE_DIR`
- Runtime library variables: `PYTHONPATH`, `LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`, `MLIR_PYTHON_ROOT`, `PTO_PYTHON_ROOT`
- Installer variables: `PTOAS_REPO`, `PTOAS_INSTALL_ROOT`, `PTOAS_BIN_DIR`
- Wheel metadata variables: `PTOAS_PYTHON_PACKAGE_VERSION`, `PTOAS_VERSION`, `WHEEL_PLAT_NAME`
- Remote validation variables: `ASCEND_HOME_PATH`, `DEVICE_ID`, `SOC_VERSION`, `RUN_MODE`, `GOLDEN_MODE`, `PTO_ISA_REPO`, `PTO_ISA_COMMIT`, `SKIP_CASES`, `RUN_ONLY_CASES`, `RESULTS_TSV`, `OUTPUT_ROOT`
- CI secrets: `SSH_KEY`, `SSH_KNOWN_HOSTS`

**Secrets location:**
- GitHub Actions encrypted secrets for `SSH_KEY` and `SSH_KNOWN_HOSTS` in `.github/workflows/ci.yml`
- Local shell environment or CI environment for non-secret path/config variables in `README.md`, `env.sh`, `docker/create_wheel.sh`, and `test/npu_validation/scripts/run_remote_npu_validation.sh`

## Webhooks & Callbacks

**Incoming:**
- GitHub Actions event triggers only: `push`, `pull_request`, `schedule`, `workflow_dispatch`, and `release` in `.github/workflows/build_wheel.yml`, `.github/workflows/build_wheel_mac.yml`, and `.github/workflows/ci.yml`

**Outgoing:**
- GitHub release asset uploads from workflow jobs in `.github/workflows/build_wheel.yml` and `.github/workflows/build_wheel_mac.yml`
- SSH/SCP calls from CI to the configured remote NPU host in `.github/workflows/ci.yml`

---

*Integration audit: 2026-03-18*
