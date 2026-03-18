# Technology Stack

**Analysis Date:** 2026-03-18

## Languages

**Primary:**
- C++17 - Core compiler, MLIR dialect implementation, transforms, and CLI tooling in `CMakeLists.txt`, `lib/`, `include/`, and `tools/`.
- Python 3.9+ - Packaging metadata, MLIR Python dialect bindings, sample generators, and validation scripts in `pyproject.toml`, `python/`, `docker/`, and `test/`.

**Secondary:**
- Bash - Build, install, packaging, and validation automation in `install.sh`, `docker/*.sh`, `test/samples/runop.sh`, and `test/npu_validation/scripts/run_remote_npu_validation.sh`.
- CMake - Top-level build orchestration and package export in `CMakeLists.txt`, `include/CMakeLists.txt`, `lib/CMakeLists.txt`, `python/CMakeLists.txt`, and `tools/CMakeLists.txt`.
- YAML - CI/CD workflows in `.github/workflows/build_wheel.yml`, `.github/workflows/build_wheel_mac.yml`, and `.github/workflows/ci.yml`.

## Runtime

**Environment:**
- Native CLI/compiler runtime on Linux and macOS. Build targets are defined in `CMakeLists.txt`; release install flow targets Linux prebuilt binaries in `install.sh`.
- Python runtime `>=3.9` for the wheel package in `pyproject.toml`.
- Containerized build/runtime supported through `docker/Dockerfile` using `quay.io/pypa/manylinux_2_34_*` and `quay.io/ascend/cann:8.5.0-910b-ubuntu22.04-py3.11`.

**Package Manager:**
- Python `pip` for wheel dependencies and runtime packages in `.github/workflows/build_wheel.yml`, `.github/workflows/build_wheel_mac.yml`, and `docker/Dockerfile`.
- System packages via `dnf` in `docker/Dockerfile` and `.github/workflows/build_wheel.yml`.
- System packages via `apt-get` in `.github/workflows/ci.yml`.
- Homebrew via `brew` on macOS in `.github/workflows/build_wheel_mac.yml`.
- Lockfile: missing

## Frameworks

**Core:**
- LLVM 19 / MLIR 19 (`llvmorg-19.1.7`) - Required compiler framework and Python binding host in `README.md`, `CMakeLists.txt`, `.github/workflows/build_wheel.yml`, `.github/workflows/build_wheel_mac.yml`, and `docker/Dockerfile`.
- Out-of-tree MLIR dialect architecture - PTO dialect is built as an external MLIR project with install/export support in `CMakeLists.txt`, `include/PTO/`, `lib/PTO/`, and `python/pto/dialects/pto.py`.
- CTest - Enabled for test targets in `CMakeLists.txt`.

**Testing:**
- GitHub Actions CI - Main automation runner for build, sample translation tests, wheel smoke tests, and remote NPU validation in `.github/workflows/ci.yml`, `.github/workflows/build_wheel.yml`, and `.github/workflows/build_wheel_mac.yml`.
- Shell-based sample tests - End-to-end `python -> .pto -> ptoas -> .cpp` flow in `docker/test_ptoas_cli.sh` and `test/samples/runop.sh`.

**Build/Dev:**
- CMake 3.20+ - Required build generator front-end in `CMakeLists.txt` and `README.md`.
- Ninja - Primary build executor in `README.md`, `.github/workflows/build_wheel.yml`, `.github/workflows/build_wheel_mac.yml`, `.github/workflows/ci.yml`, and `docker/Dockerfile`.
- pybind11 - Required for Python extension binding build in `CMakeLists.txt`, `.github/workflows/build_wheel.yml`, `.github/workflows/build_wheel_mac.yml`, and `README.md`.
- nanobind - Installed alongside Python packaging dependencies in `.github/workflows/build_wheel.yml`, `.github/workflows/build_wheel_mac.yml`, and `docker/Dockerfile`.
- auditwheel - Linux wheel repair in `.github/workflows/build_wheel.yml` and `docker/Dockerfile`.
- delocate - macOS wheel repair in `.github/workflows/build_wheel_mac.yml`.
- ccache - Installed in Linux and macOS wheel pipelines in `.github/workflows/build_wheel.yml`, `.github/workflows/build_wheel_mac.yml`, and `docker/Dockerfile`.

## Key Dependencies

**Critical:**
- `llvm-project` tag `llvmorg-19.1.7` - Mandatory upstream dependency; the whole build expects matching LLVM/MLIR CMake packages and Python package layout in `README.md`, `CMakeLists.txt`, `.github/workflows/build_wheel.yml`, `.github/workflows/build_wheel_mac.yml`, and `docker/Dockerfile`.
- `pybind11<3` - Explicitly pinned because LLVM/MLIR Python bindings are incompatible with pybind11 3.x in `.github/workflows/build_wheel.yml`, `.github/workflows/build_wheel_mac.yml`, and `.github/workflows/ci.yml`.
- `setuptools` / `wheel` - Used to build the `ptoas` Python wheel in `docker/create_wheel.sh`, `docker/setup.py`, and `docker/setup_mac.py`.
- `numpy` - Required for sample/test generation and validation scripts in `.github/workflows/build_wheel.yml`, `.github/workflows/build_wheel_mac.yml`, `.github/workflows/ci.yml`, `docker/Dockerfile`, and `test/npu_validation/templates/*.py`.

**Infrastructure:**
- `auditwheel` - Bundles Linux shared-library dependencies into wheels in `.github/workflows/build_wheel.yml` and `docker/Dockerfile`.
- `delocate` - Bundles and validates macOS wheel dylib dependencies in `.github/workflows/build_wheel_mac.yml`.
- `torch==2.9.0` and `torch-npu==2.9.0` - Added only in the Ascend runtime container to facilitate on-device testing in `docker/Dockerfile`.
- `pyyaml` - Installed in the runtime container for device testing support in `docker/Dockerfile`.
- Ascend `bisheng` compiler and CANN runtime - Expected on device/container validation paths in `docker/Dockerfile`, `docker/README.md`, and `test/npu_validation/scripts/run_remote_npu_validation.sh`.

## Configuration

**Environment:**
- Build configuration is environment-variable driven. Core variables are `LLVM_DIR`, `MLIR_DIR`, `Python3_EXECUTABLE`, `pybind11_DIR`, `MLIR_PYTHON_PACKAGE_DIR`, and `CMAKE_INSTALL_PREFIX` in `README.md`, `.github/workflows/build_wheel.yml`, `.github/workflows/build_wheel_mac.yml`, and `.github/workflows/ci.yml`.
- Runtime Python integration relies on `PYTHONPATH` and shared-library discovery via `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH` in `README.md`, `env.sh`, `docker/test_ptoas_cli.sh`, and `docker/test_wheel_imports.sh`.
- Packaging/install scripts also use `PTOAS_REPO`, `PTOAS_INSTALL_ROOT`, `PTOAS_BIN_DIR`, `PTOAS_PYTHON_PACKAGE_VERSION`, and `WHEEL_PLAT_NAME` in `install.sh` and `docker/create_wheel.sh`.
- Remote NPU validation uses `ASCEND_HOME_PATH`, `DEVICE_ID`, `PTO_ISA_REPO`, `PTO_ISA_COMMIT`, `RUN_MODE`, `SOC_VERSION`, `SKIP_CASES`, and `RUN_ONLY_CASES` in `test/npu_validation/scripts/run_remote_npu_validation.sh` and `.github/workflows/ci.yml`.

**Build:**
- Primary build config: `CMakeLists.txt`
- Packaging metadata: `pyproject.toml`
- Wheel builders: `docker/create_wheel.sh`, `docker/setup.py`, `docker/setup_mac.py`
- Container build: `docker/Dockerfile`
- CI/CD pipelines: `.github/workflows/build_wheel.yml`, `.github/workflows/build_wheel_mac.yml`, `.github/workflows/ci.yml`
- Local environment helper: `env.sh`

## Platform Requirements

**Development:**
- Linux is the documented primary development environment in `README.md`; macOS is additionally supported for wheel production in `.github/workflows/build_wheel_mac.yml`.
- Requires a prebuilt LLVM/MLIR shared build at `llvmorg-19.1.7`, CMake 3.20+, Ninja, a C++17 compiler, and Python 3.9+ in `README.md` and `CMakeLists.txt`.
- NPU validation requires Ascend CANN environment setup and `bisheng` availability in `docker/README.md`, `docker/Dockerfile`, and `test/npu_validation/scripts/run_remote_npu_validation.sh`.

**Production:**
- Binary distribution target: Linux x86_64 and Linux aarch64 tarballs from GitHub Releases in `install.sh` and `.github/workflows/build_wheel.yml`.
- Python wheel distribution target: Linux manylinux 2.34 and macOS single-arch wheels uploaded by GitHub Actions in `.github/workflows/build_wheel.yml` and `.github/workflows/build_wheel_mac.yml`.
- On-device execution target: Ascend/CANN runtime environment with optional simulator/NPU execution paths in `docker/Dockerfile` and `test/npu_validation/scripts/generate_testcase.py`.

---

*Stack analysis: 2026-03-18*
