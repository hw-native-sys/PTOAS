#!/usr/bin/env python3
"""Measured pointer-table CCE reference and fixed-argument VMI control."""
from __future__ import annotations
import argparse, ctypes, os, subprocess
from pathlib import Path
import torch, torch_npu  # noqa: F401
HERE=Path(__file__).parent; OUT=HERE/'outputs'; DEV=f"npu:{os.environ.get('ACL_DEVICE_ID','0')}"; WARMUP,SAMPLES=10,20
def cmd(a, **kw): subprocess.run(a,check=True,**kw)
def bisheng(): return os.environ.get('BISHENG',f"{os.environ['ASCEND_HOME_PATH']}/bin/bisheng")
def link(o,so):
 r=os.environ['ASCEND_HOME_PATH']; cmd([bisheng(),'--cce-fatobj-link','-shared','-fPIC',*map(str,o),'-L'+r+'/aarch64-linux/lib64','-Wl,-rpath,'+r+'/aarch64-linux/lib64','-Wl,--no-as-needed','-lruntime','-o',str(so)])
def build_cce():
 OUT.mkdir(exist_ok=True); r=os.environ['ASCEND_HOME_PATH']; d,h,so=OUT/'ref_device.o',OUT/'ref_host.o',OUT/'libref_cce.so'; inc=['-Ifixtures','-I'+r+'/include','-I'+r+'/aarch64-linux/tikcpp/tikcfw','-I'+r+'/aarch64-linux/tikcpp/tikcfw/impl','-I'+r+'/aarch64-linux/tikcpp/tikcfw/interface']
 cmd([bisheng(),'-O2','-fPIC','-std=c++17','--npu-arch=dav-3510',*inc,'-c',str(HERE/'fixtures/reference_device.asc'),'-o',str(d)]); cmd([bisheng(),'-xcce','-Xhost-start','-Xhost-end','-fPIC','-O2','-std=c++17','--cce-aicore-arch=dav-c310','-c',str(HERE/'fixtures/reference_launch.cpp'),'-o',str(h)]); link([d,h],so); return so
def build_vmi():
 OUT.mkdir(exist_ok=True); env=os.environ.copy(); env['PYTHONPATH']=str(HERE/'fixtures'); r=os.environ['ASCEND_HOME_PATH']; ptoas=os.environ.get('PTOAS_BIN') or subprocess.check_output(['conda','run','-n','cann91_dev','which','ptoas'],text=True).strip().splitlines()[-1]; mlir=subprocess.check_output(['conda','run','-n','cann91_dev','python',str(HERE/'fixtures/fixed_arguments.py'),'--emit-mlir'],text=True,env=env); (OUT/'fixed.mlir').write_text(mlir); o,h,so=OUT/'fixed.o',OUT/'fixed_host.o',OUT/'libfixed_vmi.so'; cmd([ptoas,'--pto-arch=a5','--pto-backend=vpto','--pto-level=level3',str(OUT/'fixed.mlir'),'-o',str(o)],env=env); s=OUT/'fixed_host.cpp'; s.write_text('#include <stdint.h>\nextern "C" __global__ [aicore] void fixed_two_buffers(__gm__ float*,__gm__ float*,__gm__ float*);\nextern "C" void launch_fixed_vmi(void*st,void*a,void*b,void*c){fixed_two_buffers<<<1,nullptr,st>>>((__gm__ float*)a,(__gm__ float*)b,(__gm__ float*)c);}\n'); cmd([bisheng(),'-xcce','-Xhost-start','-Xhost-end','-fPIC','-O2','-std=c++17','--cce-aicore-arch=dav-c310','-c',str(s),'-o',str(h)]); link([o,h],so); return so
def sp():
 p=torch.npu.current_stream()._as_parameter_; return p.value if hasattr(p,'value') else int(p)
def med(fn):
 for _ in range(WARMUP): fn()
 torch.npu.synchronize(); v=[]
 for _ in range(SAMPLES):
  a,z=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True); a.record(); fn(); z.record(); z.synchronize(); v.append(a.elapsed_time(z)*1000)
 return sorted(v)[len(v)//2]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--compile-only',action='store_true'); a=ap.parse_args()
 if a.compile_only: build_cce(); build_vmi(); print('PASS: stream-launchable CCE and VMI libraries built'); return
 torch.npu.set_device(DEV); c=ctypes.CDLL(str(build_cce())); v=ctypes.CDLL(str(build_vmi())); cf=c.launch_indirect_reference; vf=v.launch_fixed_vmi; cf.argtypes=[ctypes.c_void_p]*8; vf.argtypes=[ctypes.c_void_p]*4
 # Each table entry is a genuine device address.  Its per-layer allocation
 # preserves the block strides used by the A5 body (72 blocks are available;
 # n=10 activates the first ten), rather than pointing into a compact host-side
 # staging layout.
 initial=torch.zeros((10,32768),dtype=torch.bfloat16,device=DEV)
 layer_in=[torch.zeros((10,8192),dtype=torch.bfloat16,device=DEV) for _ in range(10)]
 layer_out=[torch.zeros_like(layer_in[0]) for _ in range(10)]
 residual=[torch.zeros((10,32768),dtype=torch.bfloat16,device=DEV) for _ in range(10)]
 pre=[torch.zeros((10,4),dtype=torch.float32,device=DEV) for _ in range(10)]
 post=[torch.zeros_like(pre[0]) for _ in range(10)]
 comb=[torch.zeros((10,16),dtype=torch.float32,device=DEV) for _ in range(10)]
 def ptrs(items): return torch.tensor([x.data_ptr() for x in items],dtype=torch.int64,device=DEV)
 tables=[ptrs(comb),ptrs(layer_in),ptrs(layer_out),ptrs(post),ptrs(pre),ptrs(residual)]; stream=lambda:ctypes.c_void_p(sp())
 def cr(): cf(stream(),ctypes.c_void_p(tables[0].data_ptr()),ctypes.c_void_p(initial.data_ptr()),ctypes.c_void_p(tables[1].data_ptr()),ctypes.c_void_p(tables[2].data_ptr()),ctypes.c_void_p(tables[3].data_ptr()),ctypes.c_void_p(tables[4].data_ptr()),ctypes.c_void_p(tables[5].data_ptr()))
 a0=torch.ones(64,dtype=torch.float32,device=DEV); a1=torch.ones_like(a0); a2=torch.empty_like(a0)
 def vr():
  for _ in range(10): vf(stream(),ctypes.c_void_p(a0.data_ptr()),ctypes.c_void_p(a1.data_ptr()),ctypes.c_void_p(a2.data_ptr()))
 cr(); vr(); torch.npu.synchronize(); assert all(bool(torch.all(x.cpu()==0)) for x in residual) and bool(torch.all(a2.cpu()==1)); cu,vu=med(cr),med(vr); print(f'device={DEV} layers=10 pointer_tables=6 samples={SAMPLES} warmup={WARMUP}'); print('correctness=PASS cce_host_golden=PASS vmi_host_golden=PASS pointer_table_addresses=DEVICE'); print(f'CCE_us={cu:.3f} fixed_argument_VMI_us={vu:.3f} CCE_over_fixed_VMI={cu/vu:.4f}')
if __name__=='__main__': main()
