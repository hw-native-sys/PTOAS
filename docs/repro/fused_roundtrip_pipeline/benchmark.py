#!/usr/bin/env python3
from __future__ import annotations
import ctypes, os, subprocess
from pathlib import Path
import torch, torch_npu  # noqa: F401

HERE=Path(__file__).parent; OUT=HERE/'outputs'; DEV=f"npu:{os.environ.get('LIVE_DEVICE',os.environ.get('ACL_DEVICE_ID','0'))}"
def sp():
 p=torch.npu.current_stream()._as_parameter_; return p.value if hasattr(p,'value') else int(p)
def main():
 torch.npu.set_device(DEV); OUT.mkdir(exist_ok=True); env=os.environ.copy(); env.pop('PYTHONPATH',None)
 ptoas=os.environ.get('PTOAS_BIN') or subprocess.check_output(['conda','run','-n','cann91_dev','which','ptoas'],text=True).strip().splitlines()[-1]
 subprocess.run([ptoas,'--pto-arch=a5','--pto-backend=vpto','--pto-level=level3',str(HERE/'fixtures/fused_roundtrip_vmi.pto'),'-o',str(OUT/'live_vmi.o')],check=True,env=env)
 (OUT/'live_launch.cpp').write_text('#include <stdint.h>\nextern "C" __global__ [aicore] void fused_roundtrip_body(__gm__ uint16_t*,__gm__ uint16_t*,__gm__ float*);\nextern "C" void live_launch(void*x,void*y,void*s,void*st){fused_roundtrip_body<<<1,nullptr,st>>>((__gm__ uint16_t*)x,(__gm__ uint16_t*)y,(__gm__ float*)s);}\n')
 b=os.environ.get('BISHENG',f"{os.environ['ASCEND_HOME_PATH']}/bin/bisheng"); subprocess.run([b,'-xcce','-Xhost-start','-Xhost-end','-fPIC','-O2','-std=c++17','--cce-aicore-arch=dav-c310','-c',str(OUT/'live_launch.cpp'),'-o',str(OUT/'live_launch.o')],check=True)
 subprocess.run([b,'-fPIC','-shared','--cce-fatobj-link','-o',str(OUT/'live_vmi.so'),str(OUT/'live_vmi.o'),str(OUT/'live_launch.o'),'-L'+os.environ['ASCEND_HOME_PATH']+'/aarch64-linux/lib64','-Wl,-rpath,'+os.environ['ASCEND_HOME_PATH']+'/aarch64-linux/lib64','-Wl,--no-as-needed','-lruntime'],check=True)
 l=ctypes.CDLL(str(OUT/'live_vmi.so')); f=l.live_launch; f.argtypes=[ctypes.c_void_p]*4
 x=torch.ones(256,dtype=torch.bfloat16,device=DEV); y=torch.empty_like(x); s=torch.empty(8,dtype=torch.float32,device=DEV)
 def run(): f(ctypes.c_void_p(x.data_ptr()),ctypes.c_void_p(y.data_ptr()),ctypes.c_void_p(s.data_ptr()),ctypes.c_void_p(sp()))
 for _ in range(20): run()
 vals=[]
 for _ in range(30):
  a=torch.npu.Event(enable_timing=True); z=torch.npu.Event(enable_timing=True); a.record(); run(); z.record(); z.synchronize(); vals.append(a.elapsed_time(z)*1000)
 torch.npu.synchronize(); got=y.cpu(); scale=s.cpu()
 if not bool(torch.all(got==1)): raise RuntimeError(f'unexpected roundtrip output {got[:8]}')
 print(f'device={DEV} samples={len(vals)} warmup=20 launches_per_sample=1 VMI_us={sorted(vals)[len(vals)//2]:.3f}')
 print(f'correctness=PASS output_bf16_one={got[0].item()} scale0={scale[0].item():.6g}')
if __name__=='__main__': main()
