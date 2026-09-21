# CANN `.run` wheel payload

`bash build.sh --pkg` builds a PTOAS wheel, collects its shared dependencies with
`auditwheel repair`, and creates a separate CANN delivery copy before CPack
embeds it in the installer. The original repaired wheel remains in
`build/wheelhouse`; the delivery wheel is in `.ptoas-wheelhouse`.

The delivery copy strips every ELF file, including the libraries copied by
auditwheel, and removes both `DT_RPATH` and `DT_RUNPATH`. Package-local
`DT_NEEDED` entries become explicit `$ORIGIN`-relative filenames. The Linux
glibc loader resolves these relative to the requesting ELF, so relocating the
installation works for the CLI and direct Python imports without setting
`LD_LIBRARY_PATH`. External system dependencies keep their ordinary SONAMEs.
This processing is specific to CANN installers; ordinary pip wheels and editable
builds retain their existing loading policy.

The postprocessor accepts identical library aliases in the same directory and
rejects ambiguous library names, package dependencies that
escape the wheel or no longer exist, residual symbol/debug sections, and
remaining runtime search paths. It preserves the input archive, regenerates
wheel `RECORD` hashes and sizes, and publishes the output only after reopening
and validating it. CMake repeats validation when staging the wheel, including
when `PTOAS_WHEEL_FILE` is supplied directly to CPack.

To check an extracted installer's wheel on a Linux build host with `wheel` and
`pyelftools` installed:

```bash
python3 scripts/package/harden_cann_wheel.py path/to/ptoas.whl --check
```

To create a delivery copy, also put native `strip` and `patchelf` on `PATH`:

```bash
python3 scripts/package/harden_cann_wheel.py path/to/ptoas.whl --output-dir delivery
```

The architecture of `strip` must match the wheel. The normal CANN build runs
natively on each target architecture. Run the regression on Linux with a C
compiler, Python development headers, `wheel`, `pyelftools`, and `patchelf`:

```bash
python3 test/python/test_harden_cann_wheel.py -v
```

The regression compiles real ELF fixtures with transitive and circular library
dependencies, checks both RPATH and RUNPATH removal, and loads a Python native
module and executable from an unrelated installation prefix with no loader
environment variables. The CANN package build runs this regression before
processing the delivery wheel.
