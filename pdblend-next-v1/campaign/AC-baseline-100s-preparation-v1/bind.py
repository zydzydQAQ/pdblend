"""Bind frozen historical baseline policies to explicit observed deployments.

This module does no HTTP, Docker, GPU, source rewriting, or profile synthesis.
The caller must acquire the node lease and verify the bound deployment before
running any configuration. Binding is a CPU preparation result, not readiness.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
PROTOCOL='per-dataset-slo-five-system-fixed-window-v1'
SLOS={'alpaca':(1.,.1),'sharegpt':(5.,.15),'longbench':(15.,.2)}
SYSTEMS={'mixed','distserve','ecoserve','dynamollm','dynamollm-resident'}

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok: raise ValueError(message)

def configuration(model,dataset,strategy,*,scale=1.,deployment=None,out=None):
    require(model in ('14b','7b') and dataset in SLOS and strategy in SYSTEMS,'unknown baseline identity')
    require(scale in (.5,1.,2.) and type(scale) in (int,float),'unsupported SLO scale')
    index=read(ROOT/'historical-index.json')
    rows=[r for r in index['records'] if (r['model'],r['dataset'],r['strategy'])==(model,dataset,strategy)]
    require(len(rows)==1,'no historical implementation for this model/system; resident cannot be promoted to full Dynamo')
    record=rows[0]; path=ROOT/record['template']
    require(sha(path)==record['template_sha256'],'historical template changed')
    cfg=copy.deepcopy(read(path)); original=copy.deepcopy(cfg)
    mapping={i['id']:i['id'] for i in cfg['instances']}
    if deployment is not None:
        require(deployment.get('model')==model,'deployment model differs')
        require(deployment.get('historical_engine_observation_only') is True,
                'deployment must explicitly declare the separately verified observation-only engine branch')
        layout=deployment.get('layouts',{}).get(strategy+':'+dataset,
            deployment.get('layouts',{}).get(strategy,deployment.get('layouts',{}).get('default')))
        require(isinstance(layout,list),'explicit layout is required')
        expected={(i['tp'],tuple(i['gpus'])):i for i in cfg['instances']}
        actual={(i.get('tp'),tuple(i.get('gpus',[]))):i for i in layout}
        require(len(actual)==len(layout)==len(expected) and set(actual)==set(expected),
                'physical TP/GPU layout differs from the historical selected placement')
        new=[]; ports=[]; ids=[]
        for old in cfg['instances']:
            instance=copy.deepcopy(actual[old['tp'],tuple(old['gpus'])])
            require(isinstance(instance.get('id'),str) and re.fullmatch('[A-Za-z0-9_-]+',instance['id']),'unsafe instance ID')
            require(re.fullmatch(r'(pdb-v2-|pdb-next-)[A-Za-z0-9_-]+',instance.get('container_name','')),'explicit experiment container required')
            for key in ('port','kv_port'):
                require(type(instance.get(key)) is int and 1024<=instance[key]<65536,'invalid explicit port')
            require(instance.get('url')==f"http://127.0.0.1:{instance['port']}",'loopback URL/port mismatch')
            require(instance.get('role',old['role'])==old['role'],'historical PD role cannot be changed by binding')
            instance['role']=old['role']; mapping[old['id']]=instance['id']; new.append(instance)
            ports.extend((instance['port'],instance['kv_port']));ids.append(instance['id'])
        require(len(set(ports))==len(ports) and len(set(ids))==len(ids),'duplicate instance or HTTP/KV port')
        cfg['instances']=new
        if 'dynamo_assignments' in cfg:
            cfg['dynamo_assignments']={mapping[k]:v for k,v in cfg['dynamo_assignments'].items()}
        # Only relocate external inputs. Their bytes are independently bound in
        # the deployment record; replacing profile values/costs is not allowed.
        relocations=deployment.get('path_map',{})
        def relocated(value):
            if isinstance(value,str): return relocations.get(value,value)
            if isinstance(value,list): return [relocated(x) for x in value]
            if isinstance(value,dict): return {k:relocated(v) for k,v in value.items()}
            return value
        for key in ('profiles','interconnect','transfer_evidence','frequency_evidence','retained_weights','topology'):
            if key in cfg: cfg[key]=relocated(cfg[key])
        if strategy=='dynamollm':
            require(deployment.get('topology_source_path_verified') is True,
                    'full Dynamo requires an explicit observation engine source in every lifecycle start')
            require(cfg.get('topology') and cfg.get('topology_costs') and cfg.get('retained_weights'),
                    'full Dynamo physical reconfiguration prerequisites absent')
    cfg.update(evaluation_protocol='evaluation-v3',measurement_window_protocol=PROTOCOL,
        experiment_protocol=PROTOCOL,arrival_window_s=100.,slo_scale=float(scale),
        slo_ttft_s=SLOS[dataset][0]*scale,slo_tpot_s=SLOS[dataset][1]*scale,
        slo_attainment_target=.9,power_mode='instant')
    if out is not None:cfg['journal']=str(Path(out).resolve()/'control.jsonl')
    # These are carried in the wrapper receipt, never used to alter routing.
    note=dict(schema=1,model=model,dataset=dataset,strategy=strategy,
        canonical_system='dynamollm' if strategy.startswith('dynamollm') else strategy,
        implementation_variant=strategy,historical_config_sha256=record['template_sha256'],
        historical_config_path=record['historical_config_path'],
        frozen_historical_prior=original['output_prior'],
        dataset_calibrated_historical_policy=True,initial_id_mapping=mapping,
        historical_layout_preserved=True,measurement_protocol=PROTOCOL,
        binding_is_live_readiness=False,arrival_window_s=100.,arrival_seed=701,
        dynamo_scope=('resident class routing/frequency; no topology scaling' if strategy=='dynamollm-resident'
          else 'full implementation; 100s does not normally reach 300s shard / 1800s instance periods' if strategy=='dynamollm' else None),
        original_source_and_profile_scope='paper-mechanism reproduction, not official project deployment',
        formal_eligible=False)
    return cfg,note

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=('14b','7b'),required=True)
    p.add_argument('--deployment',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); require(not a.out.exists(),'output already exists; no overwrite')
    dep=read(a.deployment) if a.deployment else None
    records=[];payloads={}
    strategies=('mixed','distserve','ecoserve','dynamollm' if a.model=='14b' else 'dynamollm-resident')
    for dataset in SLOS:
        for strategy in strategies:
            for scale in (.5,1.,2.):
                name=f'{dataset}.{strategy}.scale{scale:g}'
                cfg,note=configuration(a.model,dataset,strategy,scale=scale,deployment=dep,out=a.out/name)
                payloads[name+'.config.json']=cfg;records.append(dict(**note,config=name+'.config.json'))
    a.out.mkdir(parents=True)
    for name,value in payloads.items():(a.out/name).write_text(json.dumps(value,indent=2)+'\n')
    result=dict(schema=1,ready_to_execute=False,reason='root wrapper must verify fresh actual engine/model/source identities and own the node lease',
        deployment_path=str(a.deployment.resolve()) if a.deployment else None,
        deployment_sha256=sha(a.deployment) if a.deployment else None,
        configurations=records,files={p.name:sha(p) for p in a.out.glob('*.config.json')})
    (a.out/'binding.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(configurations=len(records),output=str(a.out.resolve()),ready_to_execute=False)))

if __name__=='__main__':main()
