# a2a3 llvm lowering spec
These specifications are for lowering on A2 and A3, i.e. pto -> vpto -> llvm.

## instruction mapping pto -> pto.ub.* 
- pto.tadd to pto.ub.vadd
- pto.tload to pto.ub.gm_to_ub
- pto.tstore to pto.ub.ub_to_gm

to lower tile level instructions use the npu-coding mcp to get the raw cce implemtation that maps very closely to pto.ub.* instructions

## instruction mapping pto.ub.* -> llvm
- at the llvm level use ony intrisics valid for dav-c220-vec as listed in /docs/designs/a2a3-intrinsics.md and /docs/designs/a2a3-dma-builtins.md
- address space for gm is 1
- address space for ub is 6

## instruction for compiling llvm
- bisheng should use dav-c220-vec as the target arch (the one for A2 and A3)

# General guidelines
- try to reuse most to the A5 pipeline
- do not use ExpandTileOp
- do not use dav-c310-vec for A2 A3 pipeline
- use dav-c220-vec for A2 A3 pipeline
- never use tilelang
- test running e2e example
```sh
python ptodsl/examples/tadd_launch_a3.py
```
# Testing
- add a test for every pto.ub.* op
- add a test for every lowering pto.* -> pto.ub.*
- tests should use captures to check arguments of the ops not just a mnemonic check
- use this docker image cce-mlir-dev-npu-aarch64-llvm19-cann900
- full command  
```sh
docker run --rm -i -v $(pwd):/workspace -v /tmp/opencode/ptoas-cce-build:/tmp/ptoas-build -v /tmp/opencode/ptoas-cce-install:/tmp/ptoas-install --ipc=host --privileged --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro --user $(id -u):$(id -g) -e HOME=/tmp --entrypoint bash cce-mlir-dev-npu-aarch64-llvm19-cann900 
```