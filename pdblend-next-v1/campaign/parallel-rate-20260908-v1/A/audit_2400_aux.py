"""Read-only actual pure-prefill and switching evidence validation."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
def require(ok,why):
    if not ok:raise ValueError(why)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m

def audit(kind,spec_path,micro,out):
    spec_path,micro,out=map(lambda p:Path(p).resolve(),(spec_path,micro,out))
    require(not out.exists(),'fresh evidence output required')
    spec=read(spec_path);sources=dict(spec['files']);sources[str(spec_path)]=sha(spec_path)
    sources[str(Path(__file__).resolve())]=sha(__file__)
    require(all(sha(p)==h for p,h in sources.items()),'frozen source changed')
    def used(p):
        p=Path(p).resolve();h=sha(p)
        require(str(p) not in sources or sources[str(p)]==h,'input changed during audit')
        sources[str(p)]=h;return p
    state=read(used(micro/'status.json'))
    require(state['complete'] and state['passed'] and state['cleanup_complete'] and not state['failed']
        and state['node_lease_held'] is False and not Path('/proc/'+str(state['pid'])).exists(),
        'complete native work, measurement and owner cleanup required')
    require(read(used(micro/'spec-reference.json'))==dict(path=str(spec_path),sha256=sha(spec_path)),
        'actual specification differs')
    require(read(used(micro/'final-restoration.json'))['complete'],'native final restoration missing')
    code=next(Path(p).parent for p in spec['files'] if p.endswith('/run_current.py'))
    sys.path[:0]=[str(code),str(Path(spec['host_release'])/'src')]
    order=load('source_order',code/'source_order.py')
    before,after=[read(used(micro/f'source-order.{phase}.json')) for phase in ('before','after')]
    receipt=read(used(state['measurement']['path']))
    require(sha(state['measurement']['path'])==state['measurement']['sha256'],'receipt changed')
    for p,h in receipt['artifacts'].items():require(sha(used(p))==h,'actual power/clock artifact changed')
    pp=next(p for p in receipt['artifacts'] if Path(p).name=='power.csv')
    with Path(pp).open() as f:power=[(float(r['t_s']),[float(r[f'gpu{g}_w']) for g in range(8)]) for r in csv.DictReader(f)]
    clocks=read(next(p for p in receipt['artifacts'] if Path(p).name=='clocks.json'))
    if isinstance(clocks,dict):clocks=clocks.get('samples',clocks.get('frequency_samples'))
    results=[]
    if kind=='prefill':
        module=load('prefill_phase_measure',code/'prefill_phase_measure.py')
        declared=module.point_specs()
        require(len(state['completed'])==spec['planned_points']==len(declared)==9,'exact nine pure-prefill cases required')
        references={};all_ids=set()
        for point in declared:
            path=micro/'results'/point['point_id'];raw=read(used(path/'raw.json'))
            require(raw['spec']==point and not raw.get('error'),'actual prefill declaration differs')
            events=[json.loads(x) for x in used(path/'events.jsonl').read_text().splitlines() if x]
            proof=order.validate_pair(before,after,'nextv3a6',measurement_start_s=raw['measurement_start_s'],
                measurement_end_s=raw['measurement_end_s'],contract_path=code/'source-order-contract.json')
            value=module.derive(raw,events,power,clocks)
            for key,n in [('declared_warmup',32),('request',64)]:
                request=raw[key];rid=request['request_id']
                require(rid not in all_ids and request['success'] and request['done_marker']
                    and len(request['output_token_ids'])==n and request['usage']['completion_tokens']==n,
                    'actual independent complete work missing')
                all_ids.add(rid)
            n=point['input_tokens'];tokens=raw['request']['output_token_ids']
            require(n not in references or references[n]==tokens,'same actual input output differs across repeats')
            references[n]=tokens
            results.append(dict(point_id=point['point_id'],raw=dict(path=str(path/'raw.json'),sha256=sha(path/'raw.json')),
                                evidence=value,source_order=proof))
    else:
        module=load('frequency_switch_measure',code/'frequency_switch_measure.py')
        raw=read(used(micro/'switches.raw.json'))
        require(raw['complete'] and not raw.get('error') and len(raw['requests'])==2,'complete actual switching work required')
        reference,request=raw['requests']
        require(reference['success'] and request['success'] and reference['done_marker'] and request['done_marker']
            and len(reference['output_token_ids'])==32 and len(request['output_token_ids'])==1024
            and request['output_token_ids'][:32]==reference['output_token_ids']
            and reference['usage']['completion_tokens']==32 and request['usage']['completion_tokens']==1024,
            'all complete output token identity required')
        require(reference['request_id']!=request['request_id'],'separate numerical reference required')
        events=[json.loads(x) for x in used(micro/'switches.events.jsonl').read_text().splitlines() if x]
        rid=request['request_id'];prefill=decode=0;previous=None
        for event in events:
            start,end=event['started_s'],event['finished_s']
            require(set(event['request_ids'])<={rid} and len(event['request_ids'])==event['prefill']+event['decode']
                and event['generation']==raw['runtime_service']['generation'] and event['role']=='mixed'
                and event['mode']=='continuous' and (previous is None or start>=previous-1e-6)
                and raw['active_start_s']<=start<end<=raw['finished_s'],'foreign or unordered actual owner work')
            previous=end;decode+=event['decode'];prefill+=event['tokens']-event['decode']
        require(prefill==128 and decode==1023,'complete native source-ordered work missing')
        require(sorted((s['repeat'],s['source_mhz'],s['target_mhz']) for s in raw['switches'])==
                sorted((rep,a,b) for rep in range(1,4) for a,b in module.pairs()),'all declared transition repeats required')
        for switch in raw['switches']:
            span=[e for e in events if e['started_s']<switch['finished_s'] and e['finished_s']>switch['started_s']]
            require(span and all(e['prefill']==0 and e['decode']==1 for e in span),
                'actual transition must overlap only real owned decode steps')
        proof=order.validate_pair(before,after,'nextv3a6',measurement_start_s=raw['active_start_s'],
            measurement_end_s=raw['active_end_s'],contract_path=code/'source-order-contract.json')
        results.append(dict(evidence=module.derive(raw,power,clocks),source_order=proof,
            raw=dict(path=str(micro/'switches.raw.json'),sha256=sha(micro/'switches.raw.json'))))
    require(all(sha(p)==h for p,h in sources.items()),'source changed at completion')
    result=dict(schema='A2400-aux-independent-evidence-v1',kind=kind,passed=True,
        results=results,source_sha256=sources,profile_publication_allowed=False)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,kind=kind,records=len(results))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['prefill','switch'],required=True)
    p.add_argument('--spec',required=True);p.add_argument('--micro',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();audit(a.kind,a.spec,a.micro,a.out)
