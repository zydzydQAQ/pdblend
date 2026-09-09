"""Original resident55 or fixed heterogeneous native gate at an explicit maximum.

No profile, baseline policy, request, ACK, native limit, or local timeout changes.
This gate saves clocks but does not by itself certify sustained loaded frequency.
"""
import argparse,asyncio,hashlib,importlib.util,json,os,socket,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE.parent.parent

def require(v,message):
    if not v:raise ValueError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def checked(r):
    require(sha(r['path'])==r['sha256'],'frozen reference changed');return read(r['path'])
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def manifest(r):
    value=checked(r);root=Path(r['path']).parent
    return {str(Path(p) if Path(p).is_absolute() else root/p):h for p,h in value['files'].items()}
def check_runtime_directory(binding,runtime_dir):
    require({str(Path(read(i['engine_config'])['runtime_dir']).resolve()) for i in binding['instances']}=={str(Path(runtime_dir).resolve())},'actual native event directory differs from declared runtime directory')

def freeze_check(spec):
    require(spec['schema']=='distributed14b-original-native-gate-qualified-max-v1' and spec['authorized'] is True,'unknown gate authorization')
    require(type(spec['maximum']) is int and spec['maximum'] in (2400,2520),'only declared hardware maxima')
    for p,h in spec['files'].items():require(sha(p)==h,'gate source/input changed: '+p)
    binding=checked(spec['binding']);require(binding['model']=='14b','original 14B model required')
    require(ref(Path(binding['host_release'])/'manifest.json')==spec['host_manifest'],'actual serving source differs')
    cpu=checked(spec['source_cpu']);require(cpu['passed'] and any(x['manifest']==spec['host_manifest'] for x in cpu['sources']),'actual maximum-aware source missing from frozen CPU proof')
    require(spec['files'].get(spec['binding']['path'])==spec['binding']['sha256'],'binding outside declaration closure')
    for p,h in binding['files'].items():require(spec['files'].get(p)==h,'binding closure omitted')
    require(spec['kind'] in ('resident','heterogeneous'),'unknown layout gate')
    check_runtime_directory(binding,spec['runtime_dir'])
    return binding

def prepare(binding,runtime_dir,out,*,heterogeneous=False,maximum=2400):
    binding=Path(binding).resolve();b=read(binding);out=Path(out).resolve()
    require(not out.exists(),'new immutable gate declaration required')
    host=ref(Path(b['host_release'])/'manifest.json');cpu=ref(R/'cpu-validation-p10.json')
    files={**b['files'],**manifest(host),**manifest(ref(HERE/'manifest.json'))}
    refs=[ref(binding),host,cpu,ref(HERE/'manifest.json')]
    for x in refs:files[x['path']]=x['sha256']
    spec=dict(schema='distributed14b-original-native-gate-qualified-max-v1',authorized=True,binding=ref(binding),hostname=b['hostname'],host_manifest=host,source_cpu=cpu,source=ref(HERE/'manifest.json'),maximum=maximum,kind='heterogeneous' if heterogeneous else 'resident',runtime_dir=str(Path(runtime_dir).resolve()),files=files,request_and_native_gates_unchanged=True,work_timeout_s=390,cleanup_timeout_s=90,global_deadline=None,loaded_frequency_qualification_inferred=False)
    freeze_check(spec);write(out,spec);return ref(out)

def gate_module(spec):
    path=HERE/(spec['kind']+'.py');name='original_qualified_max_'+spec['kind'];sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    spec=read(a.spec);binding=freeze_check(spec);gate=gate_module(spec)
    host=Path(binding['host_release']);sys.path[:0]=[str(host/'src'),str(host),'/root/workspace/pdblend/.runtime-deps']
    gate.validate_scope(binding);gate.common().validate_binding(binding)
    if not a.run:print(json.dumps(dict(passed=True,cpu_only=True,maximum=spec['maximum'],kind=spec['kind'])));return
    require(socket.gethostname()==spec['hostname'] and not a.out.exists() and 'PDBLEND_NODE_LOCK_FD' not in os.environ,'fresh target host/node owner required')
    from ecopadg.serving.campaign import node_lease
    args=argparse.Namespace(out=a.out,runtime_dir=Path(spec['runtime_dir']),maximum=spec['maximum'])
    invocation=dict(spec=ref(a.spec),source=ref(HERE/'manifest.json'),binding=spec['binding'],maximum=spec['maximum'],started_s=time.time(),pid=os.getpid(),argv=list(sys.argv),request_and_native_gates_unchanged=True,loaded_frequency_qualification_inferred=False)
    ip=a.out.parent/(a.out.name+'-invocation.json');write(ip,invocation)
    with node_lease():asyncio.run(gate.execute(args,binding))
    print(json.dumps(dict(passed=True,maximum=spec['maximum'],status=ref(a.out/'status.json'),invocation=ref(ip))))
if __name__=='__main__':main()
