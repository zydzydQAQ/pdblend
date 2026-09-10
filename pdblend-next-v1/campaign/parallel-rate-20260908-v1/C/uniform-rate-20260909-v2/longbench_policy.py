"""Complete the missing C7B LongBench policy using its unchanged historical mapping."""
import argparse
import copy
from pathlib import Path
import sys

ROOT=Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'common/uniform-rate-20260909-v2'))
import support as p


def config(parent,out):
    policy=p.load(p.REPO/'campaign/AC-baseline-100s-preparation-v1/bind.py','C7B_missing_LB_policy')
    system=parent['system'];strategy='dynamollm-resident' if system=='dynamollm' else system
    historical,_=policy.configuration('7b','longbench',strategy)
    roles={(i['tp'],tuple(i['gpus'])):i['role'] for i in historical['instances']}
    layout=[dict(id=i['id'],tp=i['tp'],gpus=i['gpus'],port=i['port'],kv_port=i['kv_port'],url=i['url'],
        container_name=i['container']['name'],role=roles[i['tp'],tuple(i['gpus'])]) for i in parent['instances']]
    mapped,note=policy.configuration('7b','longbench',strategy,deployment=dict(model='7b',historical_engine_observation_only=True,
        layouts={strategy+':longbench':layout},topology_source_path_verified=False),out=out/'unused')
    mapped.update(comparison_system=system,controller_source_release=parent['host_release'])
    return mapped,note


def reconstruct(parent_ref,out):
    meter=p.load(HERE/'meter_binding.py','C7B_fresh_native_auditor')
    parent=meter.native(parent_ref)
    p.need(parent['model']=='7b' and parent['system'] in ('mixed','distserve','dynamollm'),'wrong qualification family')
    p.need('longbench' not in parent['configs'],'not a missing LongBench policy')
    mapped,note=config(parent,out)
    b=copy.deepcopy(parent)
    b['configs']['longbench']=str(out/'configs/longbench.json')
    b['output']=str(out/'results')
    b['longbench_policy_addition']=dict(parent_binding=parent_ref,policy_source=p.ref(p.REPO/'campaign/AC-baseline-100s-preparation-v1/bind.py'),
        historical_policy=note,native_gate_reused_same_fresh_8TP1=True,performance_measurements_inherited=False)
    return b,mapped


def complete_files(binding):
    for ref in (binding['longbench_policy_addition']['parent_binding'],binding['longbench_policy_addition']['policy_source'],p.ref(__file__)):
        binding['files'][ref['path']]=ref['sha256']
    configuration=Path(binding['configs']['longbench'])
    binding['files'][str(configuration)]=p.sha(configuration)
    def external(value,key=''):
        if isinstance(value,dict):
            for k,v in value.items():external(v,k)
        elif isinstance(value,list):
            for v in value:external(v,key)
        elif isinstance(value,str) and value.startswith('/') and key!='journal' and Path(value).is_file():
            binding['files'][value]=p.sha(value)
    external(p.read(configuration))
    return binding


def verify(reference):
    actual=p.checked(reference);out=Path(reference['path']).parent
    expected,cfg=reconstruct(actual['longbench_policy_addition']['parent_binding'],out)
    p.need(p.read(out/'configs/longbench.json')==cfg,'historical LongBench policy changed')
    expected=complete_files(expected)
    p.need(actual==expected,'LongBench binding differs from exact mechanical reconstruction')
    for path,digest in actual['files'].items():p.need(p.sha(path)==digest,'frozen source changed: '+path)
    return dict(passed=True,independently_recomputed=True,binding=reference)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();p.need(not args.out.exists(),'fresh policy qualification required')
    binding,cfg=reconstruct(p.ref(args.binding),args.out)
    p.save(args.out/'configs/longbench.json',cfg);complete_files(binding);p.save(args.out/'binding.json',binding)
    print(verify(p.ref(args.out/'binding.json')))
