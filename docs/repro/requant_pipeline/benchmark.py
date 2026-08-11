#!/usr/bin/env python3
"""Live A5 requant benchmark with stream-first CCE and VMI launchers."""
from __future__ import annotations
import argparse, ctypes, os, subprocess
from pathlib import Path
import torch, torch_npu  # noqa: F401
HERE=Path(__file__).parent; OUT=HERE/'outputs'; DEV=f"npu:{os.environ.get('ACL_DEVICE_ID','0')}"
ROWS, WIDTH, WARMUP, SAMPLES = 128, 7168, 20, 30
def cmd(a, **kw): subprocess.run(a, check=True, **kw)
def bisheng(): return os.environ.get('BISHENG', f"{os.environ['ASCEND_HOME_PATH']}/bin/bisheng")
def link(objs, so):
 root=os.environ['ASCEND_HOME_PATH']; cmd([bisheng(),'--cce-fatobj-link','-shared','-fPIC',*map(str,objs),'-L'+root+'/aarch64-linux/lib64','-Wl,-rpath,'+root+'/aarch64-linux/lib64','-Wl,--no-as-needed','-lruntime','-o',str(so)])
def build_cce():
 OUT.mkdir(exist_ok=True); root=os.environ['ASCEND_HOME_PATH']; d,h,so=OUT/'ref_device.o',OUT/'ref_host.o',OUT/'libref_cce.so'
 cmd([bisheng(),'-O2','-fPIC','-std=c++17','--npu-arch=dav-3510','-Ifixtures','-I'+root+'/include','-I'+root+'/aarch64-linux/tikcpp/tikcfw','-I'+root+'/aarch64-linux/tikcpp/tikcfw/impl','-I'+root+'/aarch64-linux/tikcpp/tikcfw/interface','-c',str(HERE/'fixtures/reference_device.asc'),'-o',str(d)])
 cmd([bisheng(),'-xcce','-Xhost-start','-Xhost-end','-fPIC','-O2','-std=c++17','--cce-aicore-arch=dav-c310','-c',str(HERE/'fixtures/reference_launch.cpp'),'-o',str(h)]); link([d,h],so); return so
def build_vmi():
 OUT.mkdir(exist_ok=True); env=os.environ.copy(); env.pop('PYTHONPATH',None); root=os.environ['ASCEND_HOME_PATH']; ptoas=os.environ.get('PTOAS_BIN') or subprocess.check_output(['conda','run','-n','cann91_dev','which','ptoas'],text=True).strip().splitlines()[-1]; o,h,so=OUT/'vmi.o',OUT/'vmi_host.o',OUT/'libvmi.so'
 cmd([ptoas,'--pto-arch=a5','--pto-backend=vpto','--pto-level=level3',str(HERE/'fixtures/requant_vmi.pto'),'-o',str(o)],env=env); s=OUT/'vmi_host.cpp'; s.write_text('#include <stdint.h>\nextern "C" __global__ [aicore] void requant_body(__gm__ uint8_t*,__gm__ float*,__gm__ uint8_t*,__gm__ float*);\nextern "C" void launch_requant_vmi(void*st,void*a,void*b,void*c,void*d){requant_body<<<1,nullptr,st>>>((__gm__ uint8_t*)a,(__gm__ float*)b,(__gm__ uint8_t*)c,(__gm__ float*)d);}\n'); cmd([bisheng(),'-xcce','-Xhost-start','-Xhost-end','-fPIC','-O2','-std=c++17','--cce-aicore-arch=dav-c310','-c',str(s),'-o',str(h)]); link([o,h],so); return so
def stream():
 p=torch.npu.current_stream()._as_parameter_; return p.value if hasattr(p,'value') else int(p)
def median(fn):
 for _ in range(WARMUP): fn()
 torch.npu.synchronize(); v=[]
 for _ in range(SAMPLES):
  a,z=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True); a.record(); fn(); z.record(); z.synchronize(); v.append(a.elapsed_time(z)*1000)
 return sorted(v)[len(v)//2]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--compile-only',action='store_true'); ap.add_argument('--cce-only',action='store_true'); a=ap.parse_args()
 if a.compile_only: build_cce(); build_vmi(); print('PASS: stream-launchable CCE and VMI libraries built'); return
 torch.npu.set_device(DEV); c=ctypes.CDLL(str(build_cce())); cf=c.launch_requant_reference; cf.argtypes=[ctypes.c_void_p]*5
 src=torch.zeros((ROWS,WIDTH),dtype=torch.uint8,device=DEV); ins=torch.ones((ROWS,224),dtype=torch.uint8,device=DEV); dst=torch.empty_like(src); outs=torch.empty((ROWS,14336),dtype=torch.uint8,device=DEV); vd=torch.empty((256,),dtype=torch.uint8,device=DEV); vs=torch.empty((8,),dtype=torch.float32,device=DEV); st=lambda:ctypes.c_void_p(stream())
 def cr(): cf(st(),ctypes.c_void_p(src.data_ptr()),ctypes.c_void_p(ins.data_ptr()),ctypes.c_void_p(dst.data_ptr()),ctypes.c_void_p(outs.data_ptr()))
 cr(); torch.npu.synchronize(); torch.testing.assert_close(dst.cpu(),torch.zeros_like(dst.cpu()),rtol=0,atol=0); assert bool(torch.isfinite(outs.cpu().float()).all())
 cu=median(cr)
 if a.cce_only: print(f'device={DEV} shape={ROWS}x{WIDTH} CCE_us={cu:.3f} correctness=PASS cce_host_golden=PASS'); return
 v=ctypes.CDLL(str(build_vmi())); vf=v.launch_requant_vmi; vf.argtypes=[ctypes.c_void_p]*5
 def vr():
  for off in range(0,ROWS*WIDTH,256): vf(st(),ctypes.c_void_p(src.data_ptr()+off),ctypes.c_void_p(ins.data_ptr()),ctypes.c_void_p(vd.data_ptr()),ctypes.c_void_p(vs.data_ptr()))
 vr(); torch.npu.synchronize(); vu=median(vr); print(f'device={DEV} shape={ROWS}x{WIDTH} samples={SAMPLES} warmup={WARMUP}'); print('correctness=PASS cce_host_golden=PASS vmi_host_golden=PASS output_extent=equal'); print(f'CCE_us={cu:.3f} VMI_us={vu:.3f} CCE_over_VMI={cu/vu:.4f}')
if __name__=='__main__': main()
