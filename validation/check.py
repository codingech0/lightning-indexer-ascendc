import ctypes as C, json, os, sys, argparse
import torch, torch_npu
P=C.c_void_p; I=C.c_int64; U=C.c_uint64
ap=argparse.ArgumentParser()
ap.add_argument("--shape",default="1,1,4,16")
ap.add_argument("--dtype",default="fp16",choices=["fp16","bf16"])
ap.add_argument("--gamma",type=int,default=1)
ap.add_argument("--seed",type=int,default=42)
ap.add_argument("--repeat",type=int,default=100)
ap.add_argument("--warmup",type=int,default=20)
ap.add_argument("--mode",default="random",choices=["random","zero","sigmoid_extreme"])
ap.add_argument("--benchmark-failed", action="store_true",
                help="Diagnostic baseline only: time a failed kernel but retain FAIL and exit 2")
a=ap.parse_args(); B,S,n,D=map(int,a.shape.split(",")); M=n*n+2*n
torch.manual_seed(a.seed); torch.npu.set_device(0)
lib=C.CDLL(os.environ["ASCEND_CUSTOM_OPP_PATH"].split(":")[0]+"/op_api/lib/libcust_opapi.so",mode=C.RTLD_GLOBAL)
meta=C.CDLL("libnnopbase.so",mode=C.RTLD_GLOBAL)
meta.aclCreateTensor.argtypes=[C.POINTER(I),U,C.c_int,C.POINTER(I),I,C.c_int,C.POINTER(I),U,P]
meta.aclCreateTensor.restype=P
meta.aclDestroyTensor.argtypes=[P]; meta.aclDestroyTensor.restype=C.c_int
lib.aclnnMhcPreGetWorkspaceSize.argtypes=[P]*5+[C.c_double]*2+[P]*3+[C.POINTER(U),C.POINTER(P)]
lib.aclnnMhcPreGetWorkspaceSize.restype=C.c_int
lib.aclnnMhcPre.argtypes=[P,U,P,P]; lib.aclnnMhcPre.restype=C.c_int
dtype=torch.float16 if a.dtype=="fp16" else torch.bfloat16
x=torch.randn(B,S,n,D).to(dtype)
if a.mode=="zero": x.zero_()
phi=torch.randn(M,n*D)*0.05
alpha=torch.tensor([1.,0.8,1.2])
bias=torch.randn(M)*0.1
if a.mode=="sigmoid_extreme":
 # Isolate activation inputs, including exp overflow and sigmoid saturation.
 phi.zero_()
 activation_bias=torch.tensor([-100.,-80.,-20.,-1.,0.,1.,20.,80.])
 bias[:2*n]=activation_bias.repeat((2*n+7)//8)[:2*n]
gamma=torch.randn(n,D)*0.1+1 if a.gamma else None
xf=x.double(); xg=xf if gamma is None else xf*gamma.double()
mix=(xg.reshape(B*S,n*D)@phi.double().T)*torch.rsqrt(xf.square().mean((-2,-1)).reshape(-1,1)+1e-6)
pre=torch.sigmoid(mix[:,:n]*alpha[0].double()+bias[:n].double())+1e-6
ref=[(pre.reshape(B,S,n,1)*xf).sum(-2).to(dtype),
 (2*torch.sigmoid(mix[:,n:2*n]*alpha[1].double()+bias[n:2*n].double())).reshape(B,S,n).float(),
 (mix[:,2*n:]*alpha[2].double()+bias[2*n:].double()).reshape(B,S,n,n).float()]
host=[x,phi,alpha,bias,gamma]
dev=[t.npu() if t is not None else None for t in host]
out=[torch.full(r.shape,float("nan"),dtype=r.dtype,device="npu") for r in ref]
def tensor(t):
 if t is None: return None
 dims=(I*t.ndim)(*t.shape); strides=(I*t.ndim)(*t.stride())
 ptr=meta.aclCreateTensor(dims,t.ndim,{torch.float16:1,torch.float32:0,torch.bfloat16:27}[t.dtype],strides,0,2,dims,t.ndim,t.data_ptr())
 if not ptr: raise RuntimeError("aclCreateTensor returned null")
 return ptr
desc=[tensor(t) for t in dev+out]
torch.npu.synchronize()
stream=torch.npu.current_stream().npu_stream
buffers=[]
def launch():
 ws=U(); ex=P()
 rc=lib.aclnnMhcPreGetWorkspaceSize(*desc[:5],1e-6,1e-6,*desc[5:],C.byref(ws),C.byref(ex))
 if rc: raise RuntimeError("GetWorkspaceSize="+str(rc))
 work=torch.empty(ws.value,dtype=torch.uint8,device="npu") if ws.value else None
 if work is not None: buffers.append(work)
 rc=lib.aclnnMhcPre(work.data_ptr() if work is not None else None,ws,ex,stream)
 if rc: raise RuntimeError("aclnnMhcPre="+str(rc))
launch(); torch.npu.synchronize()
def validate_outputs(stage):
 stats=[]; passed=True
 for name,t,r in zip(["hIn","hPost","hRes"],out,ref):
  got=t.cpu().double(); expected=r.double(); error=(got-expected).abs()
  # Preserve the same strict thresholds for initial and replay outputs.
  rtol=(0.002 if a.dtype=="fp16" else 0.016) if name=="hIn" else 2e-5
  atol=2e-5
  ok=bool(torch.isfinite(got).all() and (error<=atol+rtol*expected.abs()).all())
  stats.append(dict(output=name,pass_=ok,max_abs=float(error.max()),max_rel=float((error/expected.abs().clamp_min(1e-8)).max()),mean_abs=float(error.mean()),atol=atol,rtol=rtol))
  passed &= ok
 record=dict(shape=[B,S,n,D],dtype=a.dtype,gamma=a.gamma,seed=a.seed,mode=a.mode,device=torch.npu.get_device_name(0),stage=stage,correctness="PASS" if passed else "FAIL",errors=stats)
 print(json.dumps(record),flush=True)
 return passed
passed=validate_outputs("initial")
if not passed and not a.benchmark_failed: sys.exit(2)
for i in range(a.warmup): launch()
torch.npu.synchronize()
for i in range(a.repeat): launch()
torch.npu.synchronize()
if a.warmup or a.repeat:
 passed = validate_outputs("after_replay") and passed
print(json.dumps(dict(completed=True,warmup=a.warmup,measured=a.repeat,latency_source="msprof kernel task duration required; no CPU timing")),flush=True)
for ptr in desc:
 if ptr: meta.aclDestroyTensor(ptr)
if not passed: sys.exit(2)
