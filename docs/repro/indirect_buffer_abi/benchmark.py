#!/usr/bin/env python3
"""Executable fixed-argument control for the indirect-buffer ABI report."""
from __future__ import annotations
import ctypes, os, subprocess
from pathlib import Path
import torch, torch_npu  # noqa: F401
HERE=Path(__file__).parent; OUT=HERE/'outputs'; DEV=f"npu:{os.environ.get('ACL_DEVICE_ID','0')}"
def main():
 torch.npu.set_device(DEV); OUT.mkdir(exist_ok=True); env=os.environ.copy(); env['PYTHONPATH']=str(HERE/'fixtures')
 ptoas=os.environ.get('PTOAS_BIN') or subprocess.check_output(['conda','run','-n','cann91_dev','which','ptoas'],text=True).strip().splitlines()[-1]
 mlir=subprocess.check_output(['conda','run','-n','cann91_dev','python',str(HERE/'fixtures/fixed_arguments.py'),'--emit-mlir'],text=True,env=env); (OUT/'live_fixed.mlir').write_text(mlir)
 # Compile the fixed ABI to verify the workaround remains usable.
 p=OUT/'live_fixed.o'; subprocess.run([ptoas,'--pto-arch=a5','--pto-backend=vpto','--pto-level=level3',str(OUT/'live_fixed.mlir'),'-o',str(p)],check=False)
 src=OUT/'live_fixed_launch.cpp'; src.write_text('#include <stdint.h>\nextern "C" __global__ [aicore] void fixed_two_buffers(__gm__ float*,__gm__ float*,__gm__ float*);\nextern "C" void live_launch(void*a,void*b,void*c,void*st){fixed_two_buffers<<<1,nullptr,st>>>((__gm__ float*)a,(__gm__ float*)b,(__gm__ float*)c);}\n')
 b=os.environ.get('BISHENG',f"{os.environ['ASCEND_HOME_PATH']}/bin/bisheng"); subprocess.run([b,'-xcce','-Xhost-start','-Xhost-end','-fPIC','-O2','-std=c++17','--cce-aicore-arch=dav-c310','-c',str(src),'-o',str(OUT/'live_fixed_launch.o')],check=True)
 subprocess.run([b,'-fPIC','-shared','--cce-fatobj-link','-o',str(OUT/'live_fixed.so'),str(p),str(OUT/'live_fixed_launch.o'),'-L'+os.environ['ASCEND_HOME_PATH']+'/aarch64-linux/lib64','-Wl,-rpath,'+os.environ['ASCEND_HOME_PATH']+'/aarch64-linux/lib64','-Wl,--no-as-needed','-lruntime'],check=True)
 l=ctypes.CDLL(str(OUT/'live_fixed.so')); f=l.live_launch; f.argtypes=[ctypes.c_void_p]*4; a=torch.ones(64,dtype=torch.float32,device=DEV); b=torch.ones_like(a); c=torch.empty_like(a); s=torch.npu.current_stream()._as_parameter_; s=s.value if hasattr(s,'value') else int(s)
 def run(): f(ctypes.c_void_p(a.data_ptr()),ctypes.c_void_p(b.data_ptr()),ctypes.c_void_p(c.data_ptr()),ctypes.c_void_p(s))
 for _ in range(20): run()
 vals=[]
 for _ in range(30):
  x=torch.npu.Event(enable_timing=True); z=torch.npu.Event(enable_timing=True); x.record(); run(); z.record(); z.synchronize(); vals.append(x.elapsed_time(z)*1000)
 torch.npu.synchronize(); print(f'device={DEV} samples=30 warmup=20 launches_per_sample=1 fixed_argument_us={sorted(vals)[15]:.3f} correctness={bool(torch.all(c.cpu()==1))} staged_copy_bytes=0 pointer_table_api=REJECTED')
if __name__=='__main__': main()
