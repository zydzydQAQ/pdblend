#!/usr/bin/env python3
"""Independent three-arm replay of the ShareGPT low-load energy exception.

Uses the original immutable runtime bundles and the common exclusive GPU queue.
The only candidate change is PDblend's existing 1500 MHz safety ceiling.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT / 'results/2026-09-25/sharegpt-lowload-retest-v1'
QUEUE = ROOT / 'results/2026-09-22/three-model/queue.json'
ARMS = ('pd2100', 'eco', 'pd1500')
ORDER = (('pd2100', 'eco', 'pd1500'), ('eco', 'pd1500', 'pd2100'),
         ('pd1500', 'pd2100', 'eco'))


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def verify(ref):
    assert binding(ref['path'])['sha256'] == ref['sha256'], ref['path']
    return read(ref['path'])


def prepare(package):
    if (package/'protocol.json').exists():
        raise FileExistsError(package)
    package.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((ROOT / 'results/compare.csv').open()))
    old_pd = next(x for x in rows if x['point_id'] == '7b-pdblend-sharegpt-x0.25-seed701'
                  and x['revision'].startswith('d4fe') and x['status'] == 'measured')
    eco_ref = next(x for x in json.loads(old_pd['available_baseline_receipts']).values()
                   if x['system'] == 'ecoserve')
    old_eco = next(x for x in rows if x['receipt_sha256'] == eco_ref['sha256'])
    originals = {}
    for arm, row in [('pd2100', old_pd), ('eco', old_eco)]:
        path = Path(row['receipt_path']).parent
        originals[arm] = read(path / 'point.json')
    assert originals['pd2100']['trace'] == originals['eco']['trace']
    diagnosis = dict(
        schema='lowload-exception-diagnosis/v1', created_s=time.time(),
        original_receipts={k: binding(x['receipt_path']) for k, x in [('pd2100', old_pd), ('eco', old_eco)]},
        original_metrics={k: {f: x.get(f) for f in (
            'energy_service_j', 'energy_tail_j', 'energy_service_tail_j', 'service_mean_power_w',
            'offered_requests', 'successful_requests', 'joint_slo_rate', 'ttft_p99_s', 'tpot_p99_s')}
            for k, x in [('pd2100', old_pd), ('eco', old_eco)]},
        extra_service_j=float(old_pd['energy_service_j'])-float(old_eco['energy_service_j']),
        extra_tail_j=float(old_pd['energy_tail_j'])-float(old_eco['energy_tail_j']),
        cause_evidence='PDblend keeps M4, EcoServe removes six of eight instances; PD clock probes return 900 to 2100 MHz.',
        causal_qualification='historical mechanism evidence; candidate effect requires new replay',
    )
    pd_dir = Path(old_pd['receipt_path']).parent
    meter = read(pd_dir / 'run/comparison-metering.json')
    start, end = meter['service_start_s'], meter['service_end_s']
    plans = [x for x in map(json.loads, (pd_dir/'run/controller.jsonl').read_text().splitlines()) if x['kind']=='plan']
    diagnosis['plans'] = [dict(offset_s=x['t']-start, counts=x['counts'], f_M=x['f_M'], shield_level=x.get('shield_level')) for x in plans]
    freqs = [x for x in map(json.loads, (pd_dir/'run/freq.jsonl').read_text().splitlines()) if start <= x[0] <= end]
    diagnosis['gpu0_clock_sample_counts'] = dict(Counter(str(round(x[1][0])) for x in freqs))
    save(package/'diagnosis.json', diagnosis)
    protocol = dict(schema='lowload-retest-protocol/v1', run_id=package.name,
        arms=list(ARMS), repeats_per_arm=3, order=ORDER,
        trace=originals['pd2100']['trace'], seed=701, duration_s=150,
        expected_requests=262, sole_candidate_control_change=dict(safety_max_freq=[2100,1500]),
        repeated_same_trace=True, new_independent_workloads=False,
        evaluation_informed_tuning=True, blind_holdout=False, formal_eligible=False,
        energy_scope='eight_gpu_boards_service_and_tail',
        per_point_fresh_session=True, preserve_original_outputs=True,
        acceptance='Report all attempts. Compare energy only with complete metering and SLO; never select the best repeat.',
        candidate_scope='Single low-load configuration experiment; no global policy/default changes.',
        preparer=binding(__file__))
    save(package/'protocol.json', protocol)
    for arm in ARMS:
        point = deepcopy(originals['eco' if arm=='eco' else 'pd2100'])
        point.update(name=point['name']+'-'+package.name+'-'+arm, run_id=package.name,
                     status='prepared', formal_eligible=False)
        point['retest'] = dict(protocol=binding(package/'protocol.json'), arm=arm,
                              historical_point=originals['eco' if arm=='eco' else 'pd2100']['name'])
        if arm == 'pd1500':
            config = verify(point['inputs']['system_config'])
            choice = verify(point['inputs']['offline_choice'])
            config['pdblend_runtime']['safety_max_freq'] = 1500
            choice['runtime_options']['safety_max_freq'] = 1500
            for role in ('P','D','M'):
                choice['plan']['f_'+role] = min(choice['plan']['f_'+role],1500)
            choice.pop('startup_contract',None)
            choice['startup_contract_pending'] = 'cpu_preview_for_explicit_1500_ceiling'
            choice['candidate_protocol'] = binding(package/'protocol.json')
            save(package/'candidate-runtime-options.json',config['pdblend_runtime'])
            choice['runtime_options_source'] = binding(package/'candidate-runtime-options.json')
            save(package/'candidate-config.json',config)
            save(package/'candidate-choice.json',choice)
            point['inputs']['system_config'] = binding(package/'candidate-config.json')
            point['inputs']['offline_choice'] = binding(package/'candidate-choice.json')
            point['optimization_version'] = dict(source_manifest=point['source_manifest'],
                profile=point['inputs']['profiles'][0], requested=config['pdblend_runtime'],
                qualification='development_only', evaluation_informed_tuning=True)
        save(package/'templates'/(arm+'.json'),point)
    print(json.dumps(dict(package=str(package),prepared_templates=3)))


def image_check(package, arm):
    from types import SimpleNamespace
    point = read(package/'templates'/(arm+'.json'))
    source_ref = point['inputs']['source_manifest']
    manifest = verify(source_ref)
    source = Path(source_ref['path']).parent
    for name, sha in manifest['files'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == sha,name
    import pdblend
    assert Path(pdblend.__file__).resolve().is_relative_to('/opt/pdblend-src')
    result = dict(arm=arm, status='passed', hardware_executed=False, source=source_ref)
    if arm == 'eco':
        from pdblend.bench.comparison_ecoserve_inputs import validate_ecoserve_inputs
        check = validate_ecoserve_inputs(point,point['engine_identity'],source_manifest=source_ref)
        assert check['preflight_ready'],check['gate_failures']
        result['input_gates'] = check
    else:
        from pdblend.bench.comparison_pdblend_observation import validate_observation_inputs
        from pdblend.bench.comparison_runtime import pdblend_window_resources
        from pdblend.bench.pdblend_runtime_options import comparison_options
        from pdblend.bench.independent_dispatch import request_rows
        validate_observation_inputs(point,point['inputs'])
        specs = [SimpleNamespace(tp=x['tp'],pp=x['pp'],generation=0) for x in point['engine_identity']['instances']]
        loaded,plan = pdblend_window_resources(point,specs)
        config = verify(point['inputs']['system_config'])
        choice = verify(point['inputs']['offline_choice'])
        ref = config['startup_helper']
        assert binding(ref['path'])['sha256'] == ref['sha256']
        spec = importlib.util.spec_from_file_location('pdblend.bench.comparison_startup',ref['path'])
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        opts=comparison_options(config,point['inputs']['system_config']['path'],point=point)['values']
        actual=helper.preview_startup(point,loaded.model,plan,opts,request_rows(verify(point['inputs']['planning_trace'])))
        if arm=='pd1500':
            choice['startup_contract']=helper.startup_contract(point,loaded.model,plan,opts,
                request_rows(verify(point['inputs']['planning_trace'])),profile=config['profile'],
                planning_trace=point['inputs']['planning_trace'])
            choice.pop('startup_contract_pending',None)
            save(package/'candidate-choice.json',choice)
            point['inputs']['offline_choice']=binding(package/'candidate-choice.json')
            save(package/'templates'/(arm+'.json'),point)
        helper.validate_contract_bindings(choice['startup_contract'],config['profile'],point['inputs']['planning_trace'],opts)
        assert choice['startup_contract']['expected_first_plan']==actual
        assert max(actual['f_'+role] for role in ('P','D','M')) <= (1500 if arm=='pd1500' else 2100)
        result.update(actual_first_plan=actual,requested_options=config['pdblend_runtime'])
    save(package/'preflight'/(arm+'.json'),result)
    print(json.dumps(dict(arm=arm,status='passed',hardware_executed=False)))


def preflight(package):
    for arm in ARMS:
        point=read(package/'templates'/(arm+'.json'))
        source=str(Path(point['inputs']['source_manifest']['path']).parent)
        cmd=['docker','run','--rm','--network','none','--cpus','1','--memory','4g',
             '--entrypoint','/opt/venv/bin/python','-v',source+':/opt/pdblend-src:ro',
             '-v',str(ROOT)+':'+str(ROOT)+':ro','-v',str(package)+':'+str(package)+':rw',
             '-e','PYTHONPATH=/opt/pdblend-src','-e','PYTHONDONTWRITEBYTECODE=1',
             '-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1',
             point['engine_identity']['image_digest'],'-B',str(Path(__file__).resolve()),
             '--stage','image-check','--package',str(package),'--arm',arm]
        done=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
        save(package/'preflight'/(arm+'-execution.json'),dict(argv=cmd,returncode=done.returncode,stdout=done.stdout,stderr=done.stderr))
        if done.returncode:raise RuntimeError(arm+': '+done.stderr[-5000:])
        print(done.stdout.strip(),flush=True)
    sys.path.insert(0,str(ROOT/'src'))
    from pdblend.bench.comparison_jobs import resident_job
    from pdblend.bench.resident_session import engine_signature,write_new
    points=[];groups=[];jobs=[]
    for repeat,arms in enumerate(ORDER,1):
        for arm in arms:
            point=read(package/'templates'/(arm+'.json'))
            point['name']+='-r'+str(repeat)
            point['retest']['repeat']=repeat
            identity=point['engine_identity']
            group=dict(session_id=package.name+'-'+arm+'-r'+str(repeat),model_id=point['model_id'],
                engine_identity=identity,engine_signature=engine_signature(identity),points=[point],
                gpu_count=8,exclusive=True,reserve_host=True)
            path=package/'groups'/(group['session_id']+'.json');write_new(path,group)
            cfg=verify(point['inputs']['system_config'])
            verification=(cfg.get('model_verification_receipt') or str(ROOT/'results/2026-09-22/three-model/profile-receipts/model-verification-99fabb0721f21aa50eb2a8518877acdf05cc76df32f0f769900be7b4d4471fc8.json'))
            job=resident_job(group,path,root=ROOT,source=Path(point['inputs']['source_manifest']['path']).parent,
                image=identity['image_digest'],verification=verification,campaign=package/'campaign.json',priority=3500)
            job['payload'].update(run_id=package.name,system=point['system'],model_id=point['model_id'],
                after_terminal=[jobs[-1]['job_id']] if jobs else [],formal_eligible=False,timeout_s=1800)
            jobs.append(job);groups.append(group);points.append(point)
    write_new(package/'campaign.json',dict(schema='independent-lowload-three-arm/v1',run_id=package.name,
        campaign_id=package.name,points=points,groups=groups,protocol=binding(package/'protocol.json')))
    write_new(package/'jobs.json',jobs)
    print(json.dumps(dict(preflight='passed',jobs=len(jobs),points=len(points))))


def run(package):
    import fcntl
    sys.path.insert(0,str(ROOT/'src'))
    from pdblend.experimentation.lease import GPULeaseQueue
    from pdblend.experimentation.worker import run_one
    class OwnedQueue(GPULeaseQueue):
        def _deps_ready(self,state,job):
            return job.get('payload',{}).get('run_id')==package.name and super()._deps_ready(state,job)
    lock=(package/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    queue=OwnedQueue(QUEUE)
    jobs=read(package/'jobs.json')
    for job in jobs:
        queue.enqueue(job['job_id'],job['payload'],priority=job['priority'],max_attempts=job['max_attempts'])
    save(package/'launch.json',dict(started_s=time.time(),jobs=binding(package/'jobs.json'),worker=binding(__file__)))
    while True:
        state=queue.snapshot();statuses={j['job_id']:state['jobs'][j['job_id']]['status'] for j in jobs}
        save(package/'status.json',dict(updated_s=time.time(),jobs=statuses,counts=dict(Counter(statuses.values()))))
        if all(v in ('succeeded','failed','cancelled','blocked') for v in statuses.values()):break
        if any(v=='blocked' for v in statuses.values()):
            save(package/'blocked.json',dict(at_s=time.time(),reason='blocked predecessor prevents after_terminal dependencies',jobs=statuses))
            break
        if not run_one(queue):time.sleep(5)
    analyze(package)


def analyze(package):
    state=read(QUEUE);records=[]
    for job in read(package/'jobs.json'):
        point=next(g for g in read(package/'campaign.json')['groups'] if g['session_id']==job['payload']['session_id'])['points'][0]
        root=QUEUE.parent/'queue-attempts'/job['job_id']
        results=list(root.glob('attempt-*/session/windows/*/result.json'))
        row=dict(arm=point['retest']['arm'],repeat=point['retest']['repeat'],job_id=job['job_id'],
                 status=state['jobs'][job['job_id']]['status'],result=None)
        if len(results)==1:
            result=read(results[0]);m=result.get('metrics',{})
            row.update(result=binding(results[0]),metrics={k:m.get(k) for k in (
                'offered_requests','successful_requests','failed_requests','slo_pass','joint_slo_rate',
                'energy_comparable','gpu_count','energy_service_j','energy_tail_j','energy_service_tail_j',
                'ttft_p99_s','tpot_p99_s','cohort_goodput_request_s')},
                formal_eligible=result.get('formal_eligible'),measurement_evidence_valid=result.get('measurement_evidence_valid'))
            completion=results[0].parents[2]/'completion.json'
            cleanup=read(completion).get('cleanup',{}) if completion.exists() else {}
            values=[m.get(k) for k in ('energy_service_j','energy_tail_j','energy_service_tail_j')]
            reasons=[]
            if not (all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in values)
                    and math.isclose(values[0]+values[1],values[2],rel_tol=1e-8,abs_tol=0.01)):
                reasons.append('incomplete_or_inconsistent_energy')
            if m.get('energy_comparable') is not True or m.get('gpu_count')!=8:reasons.append('eight_gpu_metering')
            if m.get('offered_requests')!=262 or m.get('successful_requests')!=262:reasons.append('request_completion')
            if m.get('slo_pass') is not True:reasons.append('slo_failure')
            if cleanup.get('passed') is not True:reasons.append('session_cleanup_unverified')
            row.update(observationally_eligible=not reasons,observational_exclusions=reasons,
                       qualification_pending=result.get('measurement_evidence_valid') is not True)
        records.append(row)
    summary={}
    for arm in ARMS:
        rr=[r for r in records if r['arm']==arm and r.get('metrics')]
        energy=[r['metrics']['energy_service_tail_j'] for r in rr if isinstance(r['metrics'].get('energy_service_tail_j'),(int,float))]
        eligible=[r['metrics']['energy_service_tail_j'] for r in rr if r.get('observationally_eligible')]
        ttfts=[r['metrics']['ttft_p99_s'] for r in rr if type(r['metrics'].get('ttft_p99_s')) in (int,float)]
        summary[arm]=dict(recorded=len(rr),slo_pass=sum(r['metrics'].get('slo_pass') is True for r in rr),
            energy_complete=len(energy),mean_energy_j=statistics.mean(energy) if energy else None,
            energy_stdev_j=statistics.stdev(energy) if len(energy)>1 else None,
            mean_ttft_p99_s=statistics.mean(ttfts) if ttfts else None,
            observationally_eligible=len(eligible),all_three_eligible=len(eligible)==3,
            comparable_mean_energy_j=statistics.mean(eligible) if len(eligible)==3 else None)
    report=dict(schema='lowload-three-arm-summary/v1',created_s=time.time(),protocol=binding(package/'protocol.json'),
        records=records,summary=summary,formal_eligible=False,scope='repeated_same_evaluation_trace_configuration_tuning')
    save(package/'report.json',report)
    print(json.dumps(summary))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','preflight','image-check','run','analyze'],required=True)
    p.add_argument('--package',type=Path,default=DEFAULT_PACKAGE)
    p.add_argument('--arm',choices=ARMS)
    args=p.parse_args();package=args.package.resolve()
    if args.stage=='image-check':image_check(package,args.arm)
    else:globals()[args.stage](package)


if __name__=='__main__':main()
