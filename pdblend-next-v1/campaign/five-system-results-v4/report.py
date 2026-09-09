"""Read-only single-seed100s tables and line/scatter plots; no serving imports."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import time

HERE=Path(__file__).resolve().parent
COMMON=HERE.parent/'five-system-fixed-window-v1'
PROTOCOL='per-dataset-slo-five-system-fixed-window-v1'
MODELS=('7b','14b','32b');DATASETS=('alpaca','sharegpt','longbench')
SYSTEMS=('pdblend','mixed','distserve','dynamollm','ecoserve')
DEFAULT_SOURCES={
    '14b':(COMMON/'sources/A14B/manifest.json','a0a2193e9504b77bcef1113b0fc7de13a8c2f3f7838f24e27f5a91e24106f327'),
    '32b':(COMMON/'sources/B32B/manifest.json','4b9494c6b0a38cb9d44dbc490530d88e9a0eec76b44854f6e40db0328bbfe5ed'),
    '7b':(COMMON/'sources/C7B/manifest.json','b5beb606905cbb944434b4e8ca06d0c08d8a258eb35921e9250d120980f16176')}
LABELS=dict(pdblend='PDBlend',mixed='Mixed',distserve='DistServe',dynamollm='DynamoLLM',ecoserve='EcoServe')
METRICS=(
    ('energy_j','Primary energy, all8 GPUs (kJ)',.001),
    ('energy_per_good_request_j','Primary energy / good request (J)',1.),
    ('slo_attainment','Joint SLO attainment (%)',100.),
    ('gpu_util','Utilization, mean of all8 GPUs (%)',100.),
    ('ttft_avg_s','Average TTFT (s)',1.),
    ('tpot_avg_s','Average TPOT (s)',1.),
    ('goodput_measurement_rps','Goodput over energy window (req/s)',1.),
    ('completion_fraction','Completed prescribed work (%)',100.))
COUNTS=('n_expected','offered_requests','completed','completed_work_requests','good_requests','failed_requests',
        'request_timeouts','admission_rejections','input_tokens','generated_tokens','expected_generated_tokens')
OBSERVATIONS=tuple(k for k,_,_ in METRICS if k!='completion_fraction')+COUNTS+(
    'measurement_valid','fixed_window_valid','work_complete','gpu_util_per_gpu','measurement_start_s',
    'measurement_end_s','measurement_duration_s','goodput_fixed_arrival_window_rps','goodput_rps',
    'implementation_variant','implementation_scope','comparison_system','runtime_error','incomplete_drain')


def require(ok,message):
    if not ok:raise ValueError(message)


class PendingProducer(ValueError):
    """A frozen in-progress invocation does not yet attribute this checkpoint."""


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda:handle.read(1024**2),b''):h.update(data)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def finite(v):return type(v) in (int,float) and math.isfinite(v)
def integer(v):return type(v) is int and v>=0


def csv_value(value):
    if value is None:return ''
    if isinstance(value,(dict,list)):return json.dumps(value,ensure_ascii=False,allow_nan=False)
    return value


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        w.writerows({k:csv_value(r.get(k)) for k in keys} for r in rows)


class Evidence:
    """Track exact bytes read for this snapshot; no live source/device calls."""
    def __init__(self):self.files={};self.invocation_observations={}
    def digest(self,path,expected=None):
        path=Path(path).resolve();value=sha(path)
        if expected is not None:require(value==expected,'artifact SHA differs: '+str(path))
        if str(path) in self.files:require(self.files[str(path)]==value,'source changed during snapshot: '+str(path))
        self.files[str(path)]=value
        return value
    def read(self,path,expected=None):
        path=Path(path).resolve();data=path.read_bytes();value=hashlib.sha256(data).hexdigest()
        if expected is not None:require(value==expected,'JSON SHA differs: '+str(path))
        if str(path) in self.files:require(self.files[str(path)]==value,'source changed during snapshot: '+str(path))
        self.files[str(path)]=value
        return json.loads(data)

    def invocation(self,path):
        """Seal the observed bytes even if an active producer later appends CPs."""
        path=Path(path).resolve();data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
        self.files[str(path)]=digest
        self.invocation_observations[digest]=data
        return dict(path=str(path),sha256=digest,observed_s=time.time(),state=json.loads(data))


def capture_binding(path,evidence,expected=None):
    path=Path(path).resolve();b=evidence.read(path,expected)
    b.pop('_host_verified',None) # Only this Evidence instance may create a successful verification cache.
    require(b.get('model') in MODELS and b.get('system') in SYSTEMS and b.get('protocol_id')==PROTOCOL,
        'different model/system/protocol binding')
    source,source_sha=DEFAULT_SOURCES[b['model']]
    require(b['files'].get(str(source))==source_sha,'binding lacks original shared declaration')
    b['_sha256']=evidence.files[str(path)];b['_path']=str(path)
    b['_producer_observations']=[evidence.invocation(p) for p in sorted((Path(b['output'])/'invocations').glob('*.json'))]
    b['_invocations']=[o['state'] for o in b['_producer_observations']]
    return b


def select_bindings(declarations,bindings,evidence,selection=None):
    """Explicit per-cell partition; never pick a successful or low-energy version."""
    indexed={r['cell_id']:r for r in declarations};selected={};cache={}
    groups=[]
    if selection is not None:
        require(not bindings,'use explicit selections or whole bindings, not both')
        spec=evidence.read(selection)
        require(spec.get('schema')=='five-system-per-cell-selection-v1','unknown result selection schema')
        groups=spec['groups']
    else:
        groups=[dict(binding=dict(path=str(Path(p).resolve()),sha256=sha(p)),cell_ids=None) for p in bindings]
    for g in groups:
        reference=g['binding'];path=str(Path(reference['path']).resolve())
        require(reference['path']==path and isinstance(reference.get('sha256'),str),'absolute pinned binding required')
        key=(path,reference['sha256'])
        b=cache.setdefault(key,None)
        if b is None:b=capture_binding(path,evidence,reference['sha256']);cache[key]=b
        ids=g.get('cell_ids')
        if ids is None:
            require(selection is None,'explicit selection needs exact cell IDs')
            ids=[r['cell_id'] for r in declarations if (r['model'],r['system'])==(b['model'],b['system']) and r['dataset'] in b['configs']]
        require(ids and len(ids)==len(set(ids)),'empty or duplicate selection')
        for cid in ids:
            require(cid in indexed and cid not in selected,'foreign or overlapping cell source selection')
            row=indexed[cid]
            require((row['model'],row['system'])==(b['model'],b['system']) and row['dataset'] in b['configs'],
                'selected source does not serve this exact model/system/dataset')
            selected[cid]=b
    return selected,cache


def verify_producer(row,binding,evidence,cp,receipt):
    """A CP's producer fixes its source; an invocation that skipped it does not."""
    matches=[o for o in binding['_producer_observations'] if row['cell_id'] in o['state'].get('completed',[])]
    source,source_sha=DEFAULT_SOURCES[row['model']]
    execution=binding.get('execution_manifest',dict(path=str(source),sha256=source_sha))
    if not matches:
        pending=[o for o in binding['_producer_observations'] if o['state'].get('binding_sha256')==binding['_sha256']
            and o['state'].get('manifest_sha256')==execution['sha256'] and o['state'].get('protocol_id')==PROTOCOL
            and o['state'].get('system')==row['system'] and o['state'].get('phase')==row['phase']
            and o['state'].get('finished_s') is None and not o['state'].get('error') and not o['state'].get('complete')]
        if pending:raise PendingProducer('checkpoint awaits attribution in the frozen invocation observation; no current process-liveness claim')
    require(len(matches)==1,'checkpoint needs one actual executing producer; a skip is not execution')
    observation=matches[0];inv=observation['state'];end=inv.get('finished_s') or observation['observed_s']
    require(inv.get('binding_sha256')==binding['_sha256'] and inv.get('protocol_id')==PROTOCOL
        and inv.get('system')==row['system'] and inv.get('phase')==row['phase']
        and row['dataset'] in inv.get('selected_datasets',DATASETS)
        and inv['started_s']<=receipt['started_s']<=receipt['finished_s']<=cp['completed_s']<=end,
        'executing invocation binding/system/phase/timestamps differ')
    require(binding['files'].get(execution['path'])==execution['sha256'] and inv.get('manifest_sha256')==execution['sha256'],
        'executing workload manifest differs from actual binding')
    declared=evidence.read(execution['path'],execution['sha256'])
    require(declared.get('model')==row['model'] and declared.get('protocol_id')==PROTOCOL
        and len([r for r in declared['cells'] if r==row])==1,'actual row missing from executing source')
    if execution['sha256']!=source_sha:
        require(declared.get('parent_manifest')==str(source) and declared.get('parent_manifest_sha256')==source_sha,
            'subset source has a different original declaration')
        original={r['cell_id']:r for r in evidence.read(source,source_sha)['cells']}
        require(len(declared['cells'])==len({r['cell_id'] for r in declared['cells']})
            and all(r==original.get(r['cell_id']) for r in declared['cells']),'subset altered original work')
    host=Path(binding['host_release']);manifest_path=str(host/'manifest.json')
    require(manifest_path in binding['files'],'host manifest not in actual binding')
    manifest=evidence.read(manifest_path,binding['files'][manifest_path])
    if '_host_verified' not in binding:
        for name,digest in manifest['files'].items():
            p=str(host/name);require(binding['files'].get(p)==digest,'host source not fully frozen in binding');evidence.digest(p,digest)
        binding['_host_verified']=True
    output=Path(binding['output']);actual_identity={}
    before=evidence.read(output/'operations'/row['cell_id']/'identity.before.json')
    after=evidence.read(output/'operations'/row['cell_id']/'identity.after.json')
    a={i['provenance']['instance_id']:i for i in before};z={i['provenance']['instance_id']:i for i in after}
    require(len(a)==len(before)==len(z)==len(after)==len(binding['instances']) and set(a)==set(z)=={i['id'] for i in binding['instances']},
        'incomplete actual process identity')
    for i in binding['instances']:
        x,y=a[i['id']],z[i['id']]
        require(x['provenance']==y['provenance'] and all(x['provenance'].get(k)==v for k,v in i['provenance'].items()),'actual engine provenance differs')
        require(x['container']['Id']==y['container']['Id']==i['container']['id']
            and x['container']['Image']==y['container']['Image']==i['container']['image']
            and x['container']['State']['StartedAt']==y['container']['State']['StartedAt']==i['container']['StartedAt']
            and type(x['container']['State']['Pid']) is int and x['container']['State']['Pid']>0
            and x['container']['State']['Pid']==y['container']['State']['Pid'],'actual engine process changed or mismatches binding')
        actual_identity[i['id']]=dict(container_id=i['container']['id'],started_at=i['container']['StartedAt'],host_pid=x['container']['State']['Pid'])
    failure=None
    if inv.get('error'):
        require(inv.get('finished_s') and inv.get('complete') is False and inv['current_cell']!=row['cell_id'],
            'failed producer does not prove a successful prefix')
        failed_id=inv['current_cell'];failed_receipt=output/'operations'/failed_id/'receipt.json'
        bad=evidence.read(failed_receipt)
        require(not (output/'checkpoints'/(failed_id+'.json')).exists() and bad.get('measurement_valid') is False
            and bad.get('finished_s') and bad.get('child_stopped') is True,'failed successor relabelled or not terminal')
        files={}
        for root in (failed_receipt.parent,output/'cells'/failed_id):
            for p in root.rglob('*'):
                if p.is_file():files[str(p)]=evidence.digest(p)
        failure=dict(cell_id=failed_id,receipt=str(failed_receipt),raw_files=files,
            primary_observed_energy_j=bad.get('summary',{}).get('energy_j'),full_operation_energy_j=bad.get('full_operation_energy_j'),
            energy_windows_overlap_not_added=True,counts_as_completed=False)
    return dict(binding_path=binding['_path'],binding_sha256=binding['_sha256'],host_release=str(host),
        host_manifest_sha256=binding['files'][manifest_path],execution_manifest=execution,
        invocation_source_path=observation['path'],invocation_sha256=observation['sha256'],
        invocation_snapshot='invocation-observations/'+observation['sha256']+'.json',
        invocation_terminal=inv.get('finished_s') is not None,invocation_error=inv.get('error'),
        invocation_observed_s=observation['observed_s'],actual_engine_identity=actual_identity,retained_failed_successor=failure)


def validate_manifest(m):
    require(m.get('protocol_id')==PROTOCOL and m.get('model') in MODELS
        and m.get('arrival_window_s')==100 and m.get('arrival_seeds')==[701]
        and m.get('comparison_systems')==list(SYSTEMS),'not the shared100s/seed701 declaration')
    rows=m['cells'];require(len(rows)==240 and len({r['cell_id'] for r in rows})==240,'wrong/duplicate model declaration')
    indexed={r['cell_id']:r for r in rows}
    for row in rows:
        require(row['model']==m['model'] and row['system'] in SYSTEMS and row['dataset'] in DATASETS
            and row['seed']==row['arrival_seed']==701 and row['trace_duration_s']==100
            and row['phase'] in ('main','scale') and integer(row['n_requests']) and row['n_requests']>0,
            'invalid shared workload identity')
        require(row['slo_scale'] in ((1.,) if row['phase']=='main' else (.5,2.)), 'wrong phase/scale')
        if row['phase']=='scale':
            main=indexed.get(row['reuse_main_cell_id'])
            require(main and main['phase']=='main' and all(main[k]==row[k] for k in
                ('system','model','dataset','rate_rps','seed','trace_sha256','content_pairing_sha256','n_requests')),
                'scale does not reference same-system exact main trace')
    for dataset in DATASETS:
        rates={r['rate_rps'] for r in rows if r['dataset']==dataset and r['phase']=='main'}
        require(len(rates)==10,'ten actual absolute rates required')
        for rate in rates:
            group=[r for r in rows if r['dataset']==dataset and r['rate_rps']==rate and r['phase']=='main']
            require(len(group)==5 and {r['system'] for r in group}==set(SYSTEMS)
                and len({(r['trace_sha256'],r['content_pairing_sha256'],r['n_requests']) for r in group})==1,
                'five systems lack identical workload SHA')
    return rows


def workload_counts(manifests,evidence):
    rows=[]
    for m in manifests:
        require(len(m['workloads'])==30,'thirty actual traces required')
        for w in m['workloads']:
            t=evidence.read(w['trace'],w['trace_sha256']);requests=t['requests'];n=len(requests)
            require(n==w['n_requests']==t['n_requests'] and len(t['prompts'])==n
                and t['protocol_id']==PROTOCOL and t['seed']==701 and t['arrival_window_s']==100,
                'actual shared trace identity/count differs')
            require(requests[0]['arrival_s']==0 and all(finite(q['arrival_s']) and 0<=q['arrival_s']<100 for q in requests),
                'arrival outside fixed window')
            rows.append(dict(model=m['model'],dataset=w['dataset'],rate_rps=w['rate_rps'],seed=701,
                arrival_window_s=100,planned_last_arrival_s=requests[-1]['arrival_s'],n_requests=n,
                input_tokens=sum(q['prompt_len'] for q in requests),
                prescribed_output_tokens=sum(q['output_len'] for q in requests),
                input_avg_tokens=statistics.fmean(q['prompt_len'] for q in requests),
                output_avg_tokens=statistics.fmean(q['output_len'] for q in requests),
                unique_pool_records=len(set(t['source_pool_indices'])),
                within_trace_resampling=t['within_trace_resampling'],sparse_screen=n<30,
                trace_sha256=w['trace_sha256'],content_pairing_sha256=w['content_pairing_sha256'],
                same_trace_systems=list(SYSTEMS),independent_arrival_realizations=1))
    return sorted(rows,key=lambda r:(r['model'],r['dataset'],r['rate_rps']))


def metric_observations(point,summary,receipt):
    """Retain failed-run observations; plotting uses a separate verified flag."""
    for field in OBSERVATIONS:point[field]=summary.get(field)
    for field in ('full_operation_energy_j','operation_start_s','operation_end_s','sampling_error',
                  'power_evidence','outer_cleanup_errors','clock_restore_complete','child_exitcode','child_stopped'):
        point[field]=receipt.get(field)
    n=summary.get('n_expected');complete=summary.get('completed_work_requests')
    point['completion_fraction']=complete/n if finite(n) and n>0 and finite(complete) else None
    point['receipt_measurement_valid']=receipt.get('measurement_valid')
    point['receipt_error']=receipt.get('error')
    point.update(observed_outer_sample_energy_j=None,observed_outer_sample_count=None,
        observed_outer_start_s=None,observed_outer_end_s=None,observed_outer_max_gap_s=None,
        observed_outer_scope=None,observed_outer_error=None)
    for key,_,_ in METRICS:
        if not finite(point[key]):point[key]=None
    point['gpu_util_vector_valid']=(isinstance(point['gpu_util_per_gpu'],list)
        and len(point['gpu_util_per_gpu'])==8 and all(finite(v) and 0<=v<=1 for v in point['gpu_util_per_gpu']))
    point['gpu_util_eight_board_consistent']=bool(point['gpu_util_vector_valid'] and finite(point['gpu_util'])
        and math.isclose(point['gpu_util'],statistics.fmean(point['gpu_util_per_gpu']),abs_tol=1e-9))
    point['metric_notes']=[]
    if summary.get('good_requests')==0:point['metric_notes'].append('zero good requests: J/good undefined')
    if not point['gpu_util_eight_board_consistent']:point['metric_notes'].append('eight-board utilization unavailable/inconsistent')


def verify_summary(row,summary,receipt):
    require(receipt.get('measurement_valid') is True and receipt.get('child_exitcode')==0
        and receipt.get('child_stopped') is True and not receipt.get('error')
        and not receipt.get('outer_cleanup_errors') and not receipt.get('sampling_error')
        and receipt.get('clock_restore_complete') is True and not receipt.get('integration_error')
        and receipt.get('power_evidence',{}).get('power_source_verified') is True,
        'receipt is not a successful measured/cleaned execution')
    require(receipt.get('restoration') and all(v.get('complete') is True for v in receipt['restoration'].values()),
        'native restoration not complete')
    w=summary.get('fixed_window',{})
    require(summary.get('measurement_valid') is True and summary.get('measurement_schema')==3
        and summary.get('measurement_window_protocol')==PROTOCOL and summary.get('fixed_window_valid') is True
        and summary.get('post_measurement_cleanup',{}).get('cleanup_complete') is True
        and summary.get('trace_sha256')==row['trace_sha256'] and summary.get('comparison_system')==row['system']
        and summary.get('slo_scale')==row['slo_scale'] and w.get('arrival_window_s')==100
        and w.get('effective_slo_s')==dict(ttft=row['slo_ttft_s'],tpot=row['slo_tpot_s']),
        'summary protocol/workload/SLO differs')
    n=row['n_requests'];good=summary.get('good_requests');complete=summary.get('completed_work_requests')
    require(summary.get('n_expected')==summary.get('offered_requests')==n
        and integer(good) and integer(complete) and 0<=good<=complete<=n
        and summary.get('failed_requests')==n-complete,'offered/completed/good denominators differ')
    start=summary.get('measurement_start_s');end=summary.get('measurement_end_s')
    energy=summary.get('energy_j');jg=summary.get('energy_per_good_request_j')
    require(finite(start) and finite(end) and end>start and finite(energy) and energy>=0
        and finite(w.get('arrival_epoch_s')) and finite(w.get('arrival_window_end_s'))
        and math.isclose(start,w['arrival_epoch_s'],abs_tol=1e-5,rel_tol=0)
        and math.isclose(w['arrival_window_end_s']-w['arrival_epoch_s'],100,abs_tol=1e-5,rel_tol=0)
        and end>=w['arrival_window_end_s'],'energy boundary does not cover fixed100s idle suffix')
    for value,expected in ((summary.get('slo_attainment'),good/n),
        (summary.get('goodput_measurement_rps'),good/(end-start)),
        (summary.get('goodput_fixed_arrival_window_rps'),good/100)):
        require(finite(value) and math.isclose(value,expected,rel_tol=1e-8,abs_tol=1e-10),'SLO/goodput denominator differs')
    require(jg is None if good==0 else finite(jg) and math.isclose(jg,energy/good,rel_tol=1e-8),
        'J/good numerator or zero-good handling differs')
    require(finite(receipt.get('full_operation_energy_j')) and receipt['full_operation_energy_j']>=0,
        'outer operation energy is missing')


def verify_artifacts(cp,row,output,evidence):
    require(cp.get('row')==row and cp.get('measurement_valid') is True,'checkpoint row differs from declaration')
    receipt=output/'operations'/row['cell_id']/'receipt.json'
    require(Path(cp['receipt']).resolve()==receipt.resolve(),'checkpoint receipt path escaped this cell')
    expected={str(p.resolve()) for base in (receipt.parent,output/'cells'/row['cell_id'])
        for p in base.rglob('*') if p.is_file()}
    required={str((output/'cells'/row['cell_id']/name).resolve()) for name in
        ('summary.json','runtime_config.json','bench.csv','power.csv','power_source.json','power_metadata.jsonl',
         'arrival_window.json','cleanup.json','control.jsonl')}
    required.update(str((receipt.parent/name).resolve()) for name in
        ('receipt.json','job.json','identity.before.json','controls.before.json','actual-epoch.json','dispatch.jsonl',
         'child.log','power/power.csv','power/power_source.json','power/power_metadata.jsonl','power/clocks.csv'))
    require(required<=expected,'checkpoint lacks required raw/clock/epoch/cleanup artifacts')
    artifacts=cp.get('artifacts')
    require(isinstance(artifacts,dict) and set(artifacts)==expected and expected,'full checkpoint artifact set differs/missing')
    for name,digest in artifacts.items():evidence.digest(name,digest)
    return evidence.read(receipt,cp['receipt_sha256'])


def observed_outer_energy(path,evidence):
    """Observed sample span only: never fill a missing full-operation boundary."""
    result=dict(observed_outer_sample_energy_j=None,observed_outer_sample_count=0,
        observed_outer_start_s=None,observed_outer_end_s=None,observed_outer_max_gap_s=None,
        observed_outer_scope='raw observed sample span only; sampling may be invalid or prefixes/tails missing; not full operation')
    if not path.exists():return result
    try:
        data=path.read_bytes();digest=hashlib.sha256(data).hexdigest();evidence.digest(path,digest)
        samples=[]
        for row in csv.DictReader(io.StringIO(data.decode())):
            t=float(row['t_s']);watts=[float(row[f'gpu{i}_w']) for i in range(8)]
            require(finite(t) and all(finite(w) and w>=0 for w in watts),'invalid actual8GPU power sample')
            require(not samples or t>=samples[-1][0],'power time regressed')
            samples.append((t,sum(watts)))
        result['observed_outer_sample_count']=len(samples)
        if len(samples)>=2:
            result.update(observed_outer_start_s=samples[0][0],observed_outer_end_s=samples[-1][0],
                observed_outer_max_gap_s=max(b[0]-a[0] for a,b in zip(samples,samples[1:])),
                observed_outer_sample_energy_j=sum((b[0]-a[0])*(a[1]+b[1])*.5 for a,b in zip(samples,samples[1:])))
    except Exception as exc:result['observed_outer_error']=repr(exc)
    return result


def terminal_invocation(output,cid,binding,evidence):
    if '_invocations' not in binding:
        binding['_invocations']=[evidence.read(path) for path in sorted((output/'invocations').glob('*.json'),reverse=True)]
    for state in binding['_invocations']:
        if (state.get('binding_sha256')==binding['_sha256'] and state.get('current_cell')==cid
                and state.get('finished_s') is not None and state.get('error')):return state['error']
    return None


def point(row,binding,evidence):
    p={key:row.get(key) for key in ('model','dataset','system','phase','cell_id','workload_id','rate_rps','seed',
        'slo_scale','slo_ttft_s','slo_tpot_s','trace_sha256','content_pairing_sha256','reuse_main_cell_id','n_requests','sparse_screen')}
    p.update(status='not_bound' if binding is None else 'pending',checkpoint_verified=False,metrics_verified=False,
        error=None,receipt_path=None,checkpoint_path=None,implementation_variant=None,executed_source=None)
    metric_observations(p,{}, {})
    if binding is None:return p
    output=Path(binding['output']).resolve();cid=row['cell_id'];cp_path=output/'checkpoints'/(cid+'.json')
    receipt_path=output/'operations'/cid/'receipt.json';summary_path=output/'cells'/cid/'summary.json'
    p.update(receipt_path=str(receipt_path),checkpoint_path=str(cp_path),binding_sha256=binding['_sha256'])
    try:
        config_path=Path(binding['configs'][row['dataset']]).resolve()
        config=evidence.read(config_path,binding['files'][str(config_path)])
        p['implementation_variant']=config['strategy']
        receipt=evidence.read(receipt_path) if receipt_path.exists() else {}
        summary=evidence.read(summary_path) if summary_path.exists() else receipt.get('summary',{})
        metric_observations(p,summary,receipt)
        p['implementation_variant']=summary.get('implementation_variant',config['strategy'])
        if not cp_path.exists():
            terminal_error=terminal_invocation(output,cid,binding,evidence)
            if receipt.get('finished_s') is not None:
                p['status']='awaiting_checkpoint' if receipt.get('measurement_valid') is True else 'technical_failed'
            elif (output/'operations'/cid).exists():p['status']='running'
            if terminal_error:p.update(status='technical_failed',error=terminal_error)
            if p['status']=='technical_failed':
                p['error']=receipt.get('error') or p['error'] or repr(dict(
                    sampling_error=receipt.get('sampling_error'),cleanup_errors=receipt.get('outer_cleanup_errors'),
                    integration_error=receipt.get('integration_error')))
                if receipt.get('full_operation_energy_j') is None:
                    p.update(observed_outer_energy(output/'operations'/cid/'power/power.csv',evidence))
            return p
        cp=evidence.read(cp_path);verified_receipt=verify_artifacts(cp,row,output,evidence)
        require(receipt==verified_receipt and receipt.get('summary')==summary,'receipt and disk summary differ')
        evidence.digest(row['trace'],row['trace_sha256'])
        actual=evidence.read(output/'cells'/cid/'runtime_config.json')
        expected=copy.deepcopy(config);expected.update(journal=str(output/'cells'/cid/'control.jsonl'),
            slo_scale=row['slo_scale'],slo_protocol='per-dataset-slo-v1',slo_attainment_target=.9,
            slo_ttft_s=row['slo_ttft_s'],slo_tpot_s=row['slo_tpot_s'],comparison_system=row['system'])
        require(actual==expected,'actual config differs from bound system/config/SLO')
        canonical='pdblend' if config['strategy'].startswith('pdblend') else 'dynamollm' if config['strategy']=='dynamollm-resident' else config['strategy']
        require(canonical==row['system'] and summary.get('implementation_variant')==config['strategy'],
            'implementation label differs from actual system')
        verify_summary(row,summary,receipt)
        p['executed_source']=verify_producer(row,binding,evidence,cp,receipt)
        p.update(status='completed',checkpoint_verified=True,metrics_verified=True)
    except PendingProducer as exc:
        p.update(status='awaiting_producer',checkpoint_verified=False,metrics_verified=False,error=None,source_observation_note=str(exc))
    except Exception as exc:
        p.update(status='evidence_error',error=repr(exc))
    return p


def plot_value(point,key):
    if not point['metrics_verified']:return float('nan')
    if key=='gpu_util' and not point['gpu_util_eight_board_consistent']:return float('nan')
    return point.get(key) if finite(point.get(key)) else float('nan')


def pairwise(points):
    index={(p['model'],p['dataset'],p['rate_rps'],p['slo_scale'],p['system']):p for p in points}
    rows=[]
    for pdb in points:
        if pdb['system']!='pdblend':continue
        for baseline in SYSTEMS[1:]:
            b=index[(pdb['model'],pdb['dataset'],pdb['rate_rps'],pdb['slo_scale'],baseline)]
            same=(pdb['trace_sha256']==b['trace_sha256'] and pdb['content_pairing_sha256']==b['content_pairing_sha256']
                  and pdb['n_requests']==b['n_requests'] and pdb['seed']==b['seed'])
            valid=bool(same and pdb['metrics_verified'] and b['metrics_verified'])
            row=dict(model=pdb['model'],dataset=pdb['dataset'],rate_rps=pdb['rate_rps'],slo_scale=pdb['slo_scale'],
                baseline=baseline,baseline_implementation=b.get('implementation_variant'),same_trace_sha256=same,
                both_verified=valid,pdb_status=pdb['status'],baseline_status=b['status'],
                primary_energy_ratio=None,j_per_good_ratio=None,slo_attainment_difference=None,
                statistical_scope='single paired arrival realization; no independent-seed confidence interval')
            for metric,out in (('energy_j','primary_energy_ratio'),('energy_per_good_request_j','j_per_good_ratio')):
                if valid and finite(pdb[metric]) and finite(b[metric]) and b[metric]>0:row[out]=pdb[metric]/b[metric]
            if valid:row['slo_attainment_difference']=pdb['slo_attainment']-b['slo_attainment']
            rows.append(row)
    return rows


def scale_display(points):
    index={(p['model'],p['cell_id']):p for p in points};result=[];seen=set()
    for p in points:
        if p['phase']!='scale':continue
        result.append(dict(p,display_source='separate_scale_cell'))
        key=(p['model'],p['reuse_main_cell_id'])
        if key not in seen:
            main=index[key]
            require(main['system']==p['system'] and main['trace_sha256']==p['trace_sha256'], 'scale display work differs')
            result.append(dict(main,display_source='reused_main_checkpoint'));seen.add(key)
    return result


def figures(points,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'figures';folder.mkdir()
    for model in MODELS:
        for dataset in DATASETS:
            # Plot the frozen declaration's full domain, including unmeasured rates.
            rates=sorted({p['rate_rps'] for p in points if p['model']==model
                          and p['dataset']==dataset and p['phase']=='main'})
            if not rates:continue
            require(len(rates)==10,'publication plot requires all ten declared main rates')
            for phase,scale,display in [('main',1.,points),('scale',.5,scale_display(points)),('scale',2.,scale_display(points))]:
                selected=[p for p in display if p['model']==model and p['dataset']==dataset and p['slo_scale']==scale
                          and (p['phase']=='main' if phase=='main' else p['phase']=='scale')]
                if not selected:continue
                fig,axes=plt.subplots(2,4,figsize=(19,8))
                for ax,(key,title,factor) in zip(axes.flat,METRICS):
                    for system in SYSTEMS:
                        group=[p for p in selected if p['system']==system]
                        variants={p.get('implementation_variant') for p in group if p.get('implementation_variant')}
                        if not variants:variants={None}
                        for variant in sorted(variants,key=str):
                            indexed={p['rate_rps']:p for p in group if p.get('implementation_variant') in (variant,None)}
                            values=[plot_value(indexed[x],key)*factor if x in indexed else float('nan') for x in rates]
                            label=LABELS[system]
                            if variant=='dynamollm-resident':label+=' (resident only)'
                            elif (system=='dynamollm' and variant is None and model in ('7b','32b')
                                  and not any(p['metrics_verified'] for p in indexed.values())):
                                # Frozen three-dataset baseline configuration declarations,
                                # source/SHA mapping in implementation-declarations.json.
                                label+='-resident'
                            if not any(p['metrics_verified'] for p in indexed.values()):label+=' (pending)'
                            ax.plot(rates,values,marker='o',markersize=4,linewidth=1.2,label=label)
                    ax.set_xlabel('Offered request rate (req/s)');ax.set_title(title);ax.grid(alpha=.2)
                    ax.set_xlim(rates[0],rates[-1])
                    if key in ('slo_attainment','gpu_util','completion_fraction'):
                        ax.set_ylim(0,100)
                    if key=='slo_attainment':
                        ax.axhline(90,color='0.4',linestyle='--',linewidth=.9,label='_nolegend_')
                        ax.text(.99,.90,'90% target',transform=ax.transAxes,ha='right',va='bottom',
                                fontsize=8,color='0.3')
                handles,labels=axes.flat[0].get_legend_handles_labels()
                fig.legend(handles,labels,loc='lower center',ncol=5,fontsize=9)
                fig.suptitle(f'{model.upper()} / {dataset} / SLO scale {scale:g} / seed701 / 100s\n'
                    'Full declared rate range; gaps are missing/unverified/not scheduled at this scale\n'
                    'pending = no verified point in this view; valid low-SLO and incomplete work retained',fontsize=12)
                fig.tight_layout(rect=(0,.07,1,.91));name=f'{model}-{dataset}-{phase}-slo{scale:g}'
                fig.savefig(folder/(name+'.png'),dpi=170);fig.savefig(folder/(name+'.pdf'));plt.close(fig)


def snapshot(out,*,bindings=(),selection=None,retained_attempt_bindings=(),counts_only=False,plots=False):
    out=Path(out).resolve();require(not out.exists(),'new snapshot directory required')
    evidence=Evidence();manifests=[evidence.read(path,digest) for path,digest in DEFAULT_SOURCES.values()]
    declarations=[]
    for m in manifests:declarations.extend(validate_manifest(m))
    counts=workload_counts(manifests,evidence)
    selected,binding_cache=select_bindings(declarations,bindings,evidence,selection)
    points=[] if counts_only else [point(r,selected.get(r['cell_id']),evidence) for r in declarations]
    attempts=[]
    for path in retained_attempt_bindings:
        key=(str(Path(path).resolve()),sha(path));b=binding_cache.get(key)
        if b is None:b=capture_binding(path,evidence,key[1]);binding_cache[key]=b
        for row in declarations:
            if (row['model']==b['model'] and row['system']==b['system']
                    and (Path(b['output'])/'operations'/row['cell_id']).exists()):
                p=point(row,b,evidence);p.update(selected_for_comparison=False,
                    retained_attempt_binding=str(Path(path).resolve()))
                attempts.append(p)
    # Scale measurements can only be paired after their explicit main reference is verified.
    indexed={(p['model'],p['cell_id']):p for p in points}
    for p in points:
        if p['phase']=='scale' and p['metrics_verified']:
            ref=indexed[(p['model'],p['reuse_main_cell_id'])]
            if not ref['metrics_verified']:
                p.update(metrics_verified=False,checkpoint_verified=False,status='evidence_error',
                    error='scale1 checkpoint not verified in this snapshot')
    by_model=[]
    for model in MODELS:
        group=[p for p in points if p['model']==model]
        old_attempts=[p for p in attempts if p['model']==model]
        bound=sorted({b['system'] for b in selected.values() if b['model']==model})
        by_model.append(dict(model=model,declared_main=150,declared_scale=90,
            bound_systems=bound,unbound_systems=[s for s in SYSTEMS if s not in bound],
            result_scan_performed=not counts_only,
            verified_main=sum(p['phase']=='main' and p['metrics_verified'] for p in group),
            verified_scale=sum(p['phase']=='scale' and p['metrics_verified'] for p in group),
            valid_below_90=sum(p['metrics_verified'] and p['slo_attainment']<.9 for p in group),
            technical_failed=sum(p['status']=='technical_failed' for p in group),
            retained_nonselected_attempts=len(old_attempts),
            retained_technical_failures=sum(p['status']=='technical_failed' for p in old_attempts),
            point_errors=[dict(cell_id=p['cell_id'],error=p['error']) for p in group if p['error']] ))
    out.mkdir(parents=True)
    write_csv(out/'request-counts.csv',counts)
    failed={p['receipt_path']:p for p in [*points,*attempts] if p['status']=='technical_failed'}
    for p in points:
        failure=(p.get('executed_source') or {}).get('retained_failed_successor')
        if failure and failure['receipt'] not in failed:failed[failure['receipt']]=failure
    document=dict(schema=2,protocol_id=PROTOCOL,created_s=time.time(),counts_only=counts_only,
        models=by_model,points=points,request_counts=counts,retained_nonselected_attempts=attempts,
        failed_attempts=list(failed.values()),
        validation_scope='v3 raw/metric checks plus per-cell executing invocation, exact workload, complete host SHA inventory and actual before/after engine process identity; native validity inherited from actual execution wrapper, not re-executed',
        notes=['Only new100s seed701 comparisons sharing exact trace SHA; old300 and historical64 are excluded.',
            'Single realization: no independent-seed confidence interval or fabricated repeat mean.',
            'Missing values are JSON null / CSV empty / plotted NaN; failed-run observed energy remains in the table.',
            'Valid incomplete work and low-SLO points remain in curves. Unverified/technical failure observations stay in tables.',
            'Primary all8 GPU-board energy and overlapping outer full-operation energy are separate; never added.',
            'No pipeline or SM occupancy is inferred from eight-board GPU utilization.',
            'Source selection is explicit per cell; no minimum-energy or successful-version selection occurs.',
            'Each completed point retains its executing binding, host revision and frozen observed invocation bytes.',
            'A verified CP may precede a later producer failure; that failure and its energy remain explicitly retained.',
            'This provenance report does not certify that different policies or host revisions are scientifically equivalent.'])
    (out/'results.json').write_text(json.dumps(document,indent=2,allow_nan=False)+'\n')
    if not counts_only:
        write_csv(out/'points.csv',points);write_csv(out/'paired-comparisons.csv',pairwise(points))
        write_csv(out/'scale-display.csv',scale_display(points))
    if attempts:write_csv(out/'retained-attempts.csv',attempts)
    if failed:write_csv(out/'failed-attempts.csv',list(failed.values()))
    if evidence.invocation_observations:
        folder=out/'invocation-observations';folder.mkdir()
        for digest,data in evidence.invocation_observations.items():(folder/(digest+'.json')).write_bytes(data)
    if plots:
        require(not counts_only,'counts-only cannot fabricate performance plots');figures(points,out)
    manifest=dict(schema=1,code_sha256=sha(__file__),protocol_id=PROTOCOL,sources=evidence.files,
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},
        no_gpu_or_network=True,no_synthetic_performance=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return document


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binding',type=Path,action='append',default=[])
    p.add_argument('--retained-attempt-binding',type=Path,action='append',default=[]);p.add_argument('--selection',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--counts-only',action='store_true');p.add_argument('--plots',action='store_true')
    a=p.parse_args();result=snapshot(a.out,bindings=a.binding,selection=a.selection,retained_attempt_bindings=a.retained_attempt_binding,
        counts_only=a.counts_only,plots=a.plots)
    print(json.dumps(dict(out=str(a.out),models=result['models'],actual_workload_rows=len(result['request_counts'])),indent=2))


if __name__=='__main__':main()
