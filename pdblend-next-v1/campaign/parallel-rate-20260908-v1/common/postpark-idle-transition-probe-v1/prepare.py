import argparse,json
from pathlib import Path
import run as m

def manifest_files(p):
 x=m.read(p);root=p.parent;files={str(Path(k) if Path(k).is_absolute() else root/k):v for k,v in x['files'].items()};files[str(p)]=m.sha(p);return files

def prepare(binding,node,out):
 b=m.read(binding);m.need(node in ('B','C') and b['hostname']=={'B':'iZwz9i5bte3xkpmcoes3t2Z','C':'iZwz9gfq11hx1sbob59yrgZ'}[node],'wrong assigned actual node')
 manifest=m.HERE/'manifest.json';host=Path(b['host_release'])/'manifest.json';adapter=m.R/'A/isolated-power-v2/manifest.json';hooks=m.R/'A/dynamic-execution-isolated-power-002/sampler_hooks.py'
 files=dict(b['files']);files.update(manifest_files(manifest));files.update(manifest_files(host));files.update(manifest_files(adapter));files[str(hooks)]=m.sha(hooks);files[str(binding.resolve())]=m.sha(binding)
 spec=dict(schema='postpark-idle-transition-diagnostic-input-v1',authorized=True,node=node,hostname=b['hostname'],binding=m.ref(binding),host_manifest=m.ref(host),target_mhz=2100,observed_default_mhz=2520,observation_limit_s=2.0,tolerance_mhz=15,stable_interval_s=.05,sample_interval_s=.01,model_requests=0,service_timeouts_changed=False,automatic_retries=False,measurement_adapter=m.ref(adapter),measurement_hooks=m.ref(hooks),points=[dict(point_id=f'gpu{g}-repeat{j}',gpu=g,repeat=j,instance_id=next(i['id'] for i in b['instances'] if i['gpus']==[g])) for g in (6,7) for j in (1,2,3)],files=files,scope='diagnostic native-empty reset to observed default then owned2100; no profile/performance result publication')
 m.spec_check(spec);m.need(not out.exists(),'immutable new diagnostic declaration required');m.write(out,spec);print(json.dumps(m.ref(out)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--node',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();prepare(a.binding,a.node,a.out)
