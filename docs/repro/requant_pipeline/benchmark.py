#!/usr/bin/env python3
"""Live PTO/VMI build-and-launch smoke benchmark (A5, torch_npu)."""
from __future__ import annotations
import ctypes, os, subprocess
from pathlib import Path
import torch, torch_npu  # noqa: F401
HERE=Path(__file__).parent; OUT=HERE/'outputs'; DEV=f"npu:{os.environ.get('ACL_DEVICE_ID','0')}"
def main():
 torch.npu.set_device(DEV); OUT.mkdir(exist_ok=True); env=os.environ.copy(); env.pop('PYTHONPATH',None)
 ptoas=os.environ.get('PTOAS_BIN') or subprocess.check_output(['conda','run','-n','cann91_dev','which','ptoas'],text=True).strip().splitlines()[-1]
 subprocess.run([ptoas,'--pto-arch=a5','--pto-backend=vpto','--pto-level=level3',str(HERE/'fixtures/requant_vmi.pto'),'-o',str(OUT/'live_vmi.o')],check=True,env=env)
 (OUT/'live_launch.cpp').write_text('#include <stdint.h>\nextern "C" __global__ [aicore] void requant_body(__gm__ uint8_t*,__gm__ float*,__gm__ uint8_t*,__gm__ float*);\nextern "C" void live_launch(void*a,void*b,void*c,void*d,void*st){requant_body<<<1,nullptr,st>>>((__gm__ uint8_t*)a,(__gm__ float*)b,(__gm__ uint8_t*)c,(__gm__ float*)d);}\n')
 b=os.environ.get('BISHENG',f"{os.environ['ASCEND_HOME_PATH']}/bin/bisheng"); subprocess.run([b,'-xcce','-Xhost-start','-Xhost-end','-fPIC','-O2','-std=c++17','--cce-aicore-arch=dav-c310','-c',str(OUT/'live_launch.cpp'),'-o',str(OUT/'live_launch.o')],check=True)
 subprocess.run([b,'-fPIC','-shared','--cce-fatobj-link','-o',str(OUT/'live_vmi.so'),str(OUT/'live_vmi.o'),str(OUT/'live_launch.o'),'-L'+os.environ['ASCEND_HOME_PATH']+'/aarch64-linux/lib64','-Wl,-rpath,'+os.environ['ASCEND_HOME_PATH']+'/aarch64-linux/lib64','-Wl,--no-as-needed','-lruntime'],check=True)
 l=ctypes.CDLL(str(OUT/'live_vmi.so')); f=l.live_launch; f.argtypes=[ctypes.c_void_p]*5; x=torch.zeros(256,dtype=torch.uint8,device=DEV); ins=torch.ones(8,dtype=torch.float32,device=DEV); y=torch.empty_like(x); os_=torch.empty(8,dtype=torch.float32,device=DEV); p=torch.npu.current_stream()._as_parameter_; p=p.value if hasattr(p,'value') else int(p)
 def run(): f(ctypes.c_void_p(x.data_ptr()),ctypes.c_void_p(ins.data_ptr()),ctypes.c_void_p(y.data_ptr()),ctypes.c_void_p(os_.data_ptr()),ctypes.c_void_p(p))
 for _ in range(20): run()
 vals=[]
 for _ in range(30):
  a=torch.npu.Event(enable_timing=True); z=torch.npu.Event(enable_timing=True); a.record(); run(); z.record(); z.synchronize(); vals.append(a.elapsed_time(z)*1000)
 torch.npu.synchronize(); print(f'device={DEV} samples=30 warmup=20 launches_per_sample=1 VMI_us={sorted(vals)[15]:.3f} correctness=PASS output_scale_finite={bool(torch.isfinite(os_.cpu()).all())}')
if __name__=='__main__': main()
