#!/usr/bin/env python3
"""Live torch_npu benchmark for the reduced grouped vector pipeline."""
from __future__ import annotations
import ctypes, os, subprocess
from pathlib import Path
import torch
import torch_npu  # noqa: F401

HERE = Path(__file__).parent
OUT = HERE / "outputs"
DEVICE = f"npu:{os.environ.get('LIVE_DEVICE', os.environ.get('ACL_DEVICE_ID', '0'))}"

def build_vmi():
    OUT.mkdir(exist_ok=True)
    env = os.environ.copy(); env.pop("PYTHONPATH", None)
    ptoas = os.environ.get("PTOAS_BIN") or subprocess.check_output(["conda","run","-n","cann91_dev","which","ptoas"], text=True).strip().splitlines()[-1]
    subprocess.run([ptoas,"--pto-arch=a5","--pto-backend=vpto","--pto-level=level3",str(HERE/"fixtures/grouped_scale_vmi.pto"),"-o",str(OUT/"live_vmi.o")], check=True, env=env)
    launch = OUT / "live_launch.cpp"
    launch.write_text('#include <stdint.h>\nextern "C" __global__ [aicore] void grouped_scale_body(__gm__ uint16_t*,__gm__ uint8_t*,__gm__ uint16_t*);\nextern "C" void live_launch(void*x,void*y,void*s,void*st){grouped_scale_body<<<1,nullptr,st>>>((__gm__ uint16_t*)x,(__gm__ uint8_t*)y,(__gm__ uint16_t*)s);}\n')
    b = os.environ.get("BISHENG", f"{os.environ['ASCEND_HOME_PATH']}/bin/bisheng")
    subprocess.run([b,"-xcce","-Xhost-start","-Xhost-end","-fPIC","-O2","-std=c++17","--cce-aicore-arch=dav-c310","-c",str(launch),"-o",str(OUT/"live_launch.o")], check=True)
    subprocess.run([b,"-fPIC","-shared","--cce-fatobj-link","-o",str(OUT/"live_vmi.so"),str(OUT/"live_vmi.o"),str(OUT/"live_launch.o"),"-L"+os.environ["ASCEND_HOME_PATH"]+"/aarch64-linux/lib64","-Wl,-rpath,"+os.environ["ASCEND_HOME_PATH"]+"/aarch64-linux/lib64","-Wl,--no-as-needed","-lruntime"], check=True)

def stream_ptr():
    p = torch.npu.current_stream()._as_parameter_
    return p.value if hasattr(p, "value") else int(p)

def main():
    torch.npu.set_device(DEVICE); build_vmi()
    lib = ctypes.CDLL(str(OUT/"live_vmi.so")); fn = lib.live_launch; fn.argtypes=[ctypes.c_void_p]*4
    # Ones make both the BF16 value and the E4M3FN bit pattern exact: 0x38.
    x=torch.ones(256,dtype=torch.bfloat16,device=DEVICE); y=torch.empty(256,dtype=torch.uint8,device=DEVICE); s=torch.empty(8,dtype=torch.bfloat16,device=DEVICE)
    def vmi(): fn(ctypes.c_void_p(x.data_ptr()),ctypes.c_void_p(y.data_ptr()),ctypes.c_void_p(s.data_ptr()),ctypes.c_void_p(stream_ptr()))
    for _ in range(20): vmi()
    torch.npu.synchronize(); samples=[]
    for _ in range(30):
        e0=torch.npu.Event(enable_timing=True); e1=torch.npu.Event(enable_timing=True); e0.record(); vmi(); e1.record(); e1.synchronize(); samples.append(e0.elapsed_time(e1)*1000)
    torch.npu.synchronize(); got=y.cpu(); got_scale=s.cpu()
    if not bool(torch.all(got == 0x38)) or not bool(torch.all(got_scale == 1)):
        raise RuntimeError(f"unexpected output: fp8={got[:8].tolist()} scale={got_scale.tolist()}")
    print(f"device={DEVICE} samples={len(samples)} warmup=20 launches_per_sample=1 VMI_us={sorted(samples)[len(samples)//2]:.3f}")
    print("correctness=PASS fp8_e4m3fn_one=0x38 grouped_scale=1")
if __name__ == "__main__": main()
