"""Collect current, field-specific baseline evidence without touching hardware.

CPU contracts certify algorithms only. In particular, no CPU result certifies
DistServe's executed phase batches or hardware KV admission boundary. Missing
hardware evidence remains explicit even when other fields pass.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from .campaign_followup_setup import read,write
from .evidence import REQUIRED_MECHANISMS,baseline_gaps,freeze_files,sha256,validate_freeze


CONTRACTS=('test_distserve_independent_tp_and_instance_search','test_dynamo_causal_length_prediction',
    'test_dynamo_nine_logical_pools_and_fragmentation','test_eco_joint_ttft_credit_and_kv_constraints',
    'test_mixed_frequency_uses_feasible_energy_minimum')


def sources():
    from .calibration import implementation_sources
    tests=Path(__file__).resolve().parents[3]/'tests/serving'
    return freeze_files(implementation_sources()|set(tests.glob('*.py')))


def cpu_contracts(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    before=sources();write(out/'source.before.json',before)
    test=Path(__file__).resolve().parents[3]/'tests/serving/test_mechanism_contracts.py'
    command=[sys.executable,'-m','pytest',str(test),'-q','--junitxml',str(out/'tests.xml')]
    result=subprocess.run(command,capture_output=True,text=True,timeout=120,check=False)
    (out/'output.txt').write_text(result.stdout+result.stderr)
    cases={}
    if (out/'tests.xml').exists():
        for case in ET.parse(out/'tests.xml').iter('testcase'):
            cases[case.attrib['name']]=not any(case.find(k) is not None for k in ('failure','error','skipped'))
    after=sources();write(out/'source.after.json',after)
    checked=dict(passed=result.returncode==0 and before==after and set(cases)==set(CONTRACTS) and all(cases.values()),
        cases=cases,source_unchanged=before==after,command=command,returncode=result.returncode,
        scope='executed CPU algorithm counterexamples; not hardware phase-batching or KV-boundary evidence',
        source_files=before,artifacts=freeze_files([str(p) for p in out.iterdir() if p.is_file()]))
    write(out/'summary.json',checked)
    return checked


def current_sources(recorded):
    current={str(p.resolve()):sha256(p) for p in Path(__file__).parent.glob('*.py')}
    if not set(current)<=set(recorded) or validate_freeze(recorded):
        raise ValueError('execution source inventory is incomplete or changed')


def engine_provenance(records,image):
    from .calibration_setup import current_engine_sources
    current=current_engine_sources()
    if not records or any(p.get('image_id')!=image or p.get('model')!='/models/Qwen2.5-14B-Instruct'
            or p.get('engine_version')!='0.9.2' or p.get('source_files_at_import')!=current for p in records):
        raise ValueError('hardware image, model, version or imported execution sources differ')


def events(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def rows(path):
    with Path(path).open() as handle:return list(csv.DictReader(handle))


def checked_power(directory):
    from .measurement import power_evidence
    directory=Path(directory);power=[(float(r['t_s']),[float(r[f'gpu{g}_w']) for g in range(8)])
        for r in rows(directory/'power.csv')]
    result=power_evidence(power,read(directory/'power_source.json'),events(directory/'power_metadata.jsonl'))
    if not result['power_source_verified']:raise ValueError('raw power is not verified instantaneous eight-card measurement')
    return power


def checked_search(cal):
    from .calibration_setup import validated_inputs
    from .interconnect import InterconnectTopology
    from .planner import TransferCost
    from .topology import InstanceSpec,validate_layout
    _,bundle,search,_,_,_,_,artifacts=validated_inputs(cal)
    if (bundle.get('instant_power_costs_verified') is not True
            or bundle.get('receiver_transfer_energy_included') is not True):
        raise ValueError('transfer energy lacks certified instantaneous sender and receiver costs')
    topology=InterconnectTopology.parse(Path(cal['interconnect']).read_text())
    links=[TransferCost(**r) for r in bundle['links']];counts={}
    for dataset in ('alpaca','sharegpt','longbench'):
        entry=search['datasets'][dataset];choices=entry['distserve']
        if not choices:raise ValueError('no executed DistServe search candidates for '+dataset)
        counts[dataset]=len(choices)
        for choice in choices:
            np=choice['prefill_count'];nd=choice['decode_count'];groups=choice['gpus']
            degrees=[choice['prefill_tp']]*np+[choice['decode_tp']]*nd
            if len(groups)!=len(degrees) or min(np,nd,choice['prefill_batch'],choice['decode_batch'])<1:
                raise ValueError('search has malformed independent stage dimensions')
            validate_layout([InstanceSpec(str(i),tp,tuple(g),20000+i,22000+i)
                for i,(tp,g) in enumerate(zip(degrees,groups))],range(8))
            for p in groups[:np]:
                for d in groups[np:]:
                    if not any(t.validated and t.source_sha256 and (t.source_tp,t.target_tp)==
                        (choice['prefill_tp'],choice['decode_tp']) and t.max_input_tokens>=entry['shape']['input_tokens']
                        and t.profile_batch>=choice['prefill_batch'] and t.matches_placement(p,d,topology) for t in links):
                        raise ValueError('candidate placement lacks its measured transport path')
    return dict(candidate_counts=counts,artifacts=artifacts,scope='executed offline search on certified profiles; capacities still require calibration')


def checked_eco(path,image):
    from .eco_validation import verify_execution
    raw=read(path)
    if not raw.get('complete') or not raw.get('passed') or raw.get('errors') or raw.get('cleanup_errors'):
        raise ValueError('EcoServe hardware validation did not pass')
    current_sources(raw['source_files'])
    engine_provenance(raw['engine_provenance'],image)
    if sha256(raw['config']['profiles'])!=raw['profile_sha256']:raise ValueError('EcoServe profiles changed')
    admissions=raw['admissions'];clients={e['client_request_id']:e for e in admissions}
    routes={e['request_id']:e['plan']['routes'][0]['decode_id'] for e in admissions}
    if len(admissions)!=4 or len(clients)!=4 or len(routes)!=4:raise ValueError('EcoServe admissions are incomplete')
    verify_execution(raw['events'],routes,raw['kv_observations'])
    if len(raw['outputs'])!=4:raise ValueError('EcoServe output set is incomplete')
    for i,output in enumerate(raw['outputs']):
        destination=routes[clients[str(i)]['request_id']]
        length=str(clients[str(i)]['input_tokens'])
        expected=raw['reference'][destination][length]
        if (not output.get('success') or not expected or output.get('token_ids')!=expected
                or output.get('generated_tokens')!=len(expected)):
            raise ValueError('EcoServe output differs from ordinary reference')
    checks=raw['checks']
    if any(checks.get(k) is not True for k in ('engine_temporal_exclusion','rolling_activation','macro_split_merge','output_correctness')):
        raise ValueError('EcoServe specific hardware mechanism check absent')
    windows=clients['1']['plan']['windows']
    if not {False,True}<={w['admit_prefill'] for w in windows}:raise ValueError('rolling window transition absent')
    operations={r.get('operation') for r in raw['changes']}
    if not {'split','merge'}<=operations:raise ValueError('actual macro split/merge absent')
    for change in raw['changes']:
        expected=[2,2] if change['operation']=='split' else [3]
        if sorted(map(len,change['after']))!=expected or not any(change['before_kv']['owners'].values()):
            raise ValueError('macro membership did not change while real KV remained live')
        if not change['acknowledgements'] or any(a['generation']!=a['acknowledged_generation'] for a in change['acknowledgements']):
            raise ValueError('macro change lacks engine generation acknowledgements')
        if change['operation']=='merge' and (not change['removed'] or any(change['removed_state'].get(k) for k in
                ('active','running','waiting','kv_allocations','transfer_allocations'))):
            raise ValueError('macro merge did not remove a drained member')
    if raw['engine_provenance_after']!=raw['engine_provenance']:raise ValueError('EcoServe resident engines changed')
    return {k:True for k in ('engine_temporal_exclusion','rolling_activation','macro_split_merge','output_correctness')}


def checked_dynamo(directory,image):
    from .calibration_setup import verify_artifacts
    from .dynamo_validation_setup import audit
    directory=Path(directory);reported=read(directory/'mechanisms.json');verify_artifacts(reported['artifacts'])
    raw=read(directory/'raw.json');current_sources(raw['source_files'])
    engine_provenance(raw['provenance_before'],image);engine_provenance(raw['provenance_after'],image)
    if raw.get('errors'):raise ValueError('Dynamo hardware execution errors')
    actual=audit(raw,events(directory/'control.jsonl'),rows(directory/'bench.csv'),read(directory/'summary.json'),checked_power(directory))
    if not actual['passed'] or actual['proposed_mechanism_fields']!=reported['proposed_mechanism_fields']:
        raise ValueError('Dynamo actual cycles/actions/output audit incomplete or changed')
    return actual['proposed_mechanism_fields']


def checked_smoke(directory,image):
    from .calibration_setup import verify_artifacts
    from .controller_smoke import audit_cell,STRATEGIES
    directory=Path(directory);report=read(directory/'summary.json');setup=read(directory.parent/'setup.json')
    verify_artifacts(setup['artifacts']);current_sources(setup['source_files'])
    if (not report.get('passed') or report.get('status')!='controller_smoke_passed' or report.get('errors')
            or set(report.get('cells',{}))!=set(STRATEGIES)):
        raise ValueError('controller smoke incomplete')
    reference=read(directory/'references.json');checked={}
    for strategy in ('mixed','mixed_dvfs','distserve'):
        for label in ('provenance_before','provenance_after'):
            engine_provenance(report['cells'][strategy][label],image)
        root=directory/strategy;checked_power(root)
        result=audit_cell(read(root/'summary.json'),rows(root/'bench.csv'),reference,
            events(root/'control.jsonl'),read(directory/(strategy+'.clocks.json')))
        if not result['passed']:raise ValueError('controller output/power audit failed: '+strategy)
        checked[strategy]=dict(output_correctness=True)
        if strategy=='mixed':
            outcomes=result['actual_clock_actions']['admission_outcomes']
            checked[strategy]['full_frequency']=bool(outcomes) and all(o.get('commanded')==2520 and
                o.get('requested')==2520 and not o.get('conservative_fallback') for o in outcomes)
    return checked


def checked_calibration(path):
    from .formal_setup import checked_calibration as check
    records,capacity,files=check(path)
    for result in records:
        if result.get('passed'):
            directory=Path(result['confirmation']['artifact']).parent
            checked_power(directory)
            files.update(directory/name for name in ('power.csv','power_source.json','power_metadata.jsonl'))
    return dict(systems={s:all(any(r['system']==s and r['dataset']==d and r.get('passed') for r in records)
        for d in ('alpaca','sharegpt','longbench')) for s in REQUIRED_MECHANISMS},
        capacity=capacity,artifacts=freeze_files(files))


def checked_batches(cal,calibration_path):
    """P and D batching can be demonstrated by separate real engine runs."""
    from .calibration_setup import verify_artifacts
    profiles=read(cal['profiles']);transfers=read(cal['transfers'])
    if (transfers.get('instant_power_costs_verified') is not True
            or transfers.get('receiver_transfer_energy_included') is not True):
        raise ValueError('transfer energy lacks certified instantaneous sender and receiver costs')
    artifacts=dict(profiles['certification_artifacts']);artifacts.update(transfers['certification_artifacts'])
    verify_artifacts(artifacts)
    current_engine=sha256(Path(__file__).with_name('engine.py'))
    required={'prefill':set(),'decode':set()}
    calibration=read(calibration_path)
    for result in calibration['results']:
        if result['system']=='distserve' and result.get('passed'):
            for instance in read(result['config'])['instances']:
                required[instance['role']].add(instance['tp'])
    if any(not tps for tps in required.values()):raise ValueError('no independently calibrated DistServe phase layout')
    examples={'prefill':{},'decode':{}};examined=[]
    for path,digest in artifacts.items():
        if Path(path).name!='raw.json':continue
        raw=read(path)
        if not raw.get('complete') or raw.get('sampling_error') or not raw.get('runs') or not raw.get('frequency_samples'):continue
        if any(len(values)!=8 or any(not isinstance(f,(int,float)) or not math.isfinite(f) or f<=0 for f in values)
               for _,values in raw['frequency_samples']):continue
        provenance=raw.get('engine_provenance',[])
        # This proves unchanged engine batching code on the same immutable
        # image, not that unrelated subsequently added helper modules existed.
        if not provenance or any(p.get('image_id')!=cal['image'] or p.get('engine_version')!='0.9.2'
                or p.get('model')!='/models/Qwen2.5-14B-Instruct'
                or {v for k,v in p.get('source_files_at_import',{}).items() if k.endswith('/serving/engine.py')}!={current_engine}
                for p in provenance):continue
        # Phase execution is a capability claim, not an energy result. An
        # older raw file with average-power readings can prove real batch
        # steps; none of its power/energy values enters this certificate.
        topology=raw.get('topology',{});examined.append(path)
        for index,run in enumerate(raw['runs']):
            if run.get('skipped'):continue
            step=run.get('step',{});source=topology.get('prefill',{})
            command=run.get('commanded_frequencies',raw.get('commanded_frequencies',{}))
            def full(instance):
                return instance and all(command.get(str(g),command.get(g))==2520 for g in instance['gpus'])
            if (source and full(source) and step.get('role')=='prefill' and step.get('prefill',0)>=2
                    and not step.get('decode') and step.get('tokens',0)>0
                    and len(set(step.get('request_ids',[])))==step['prefill']
                    and run.get('output_matches') is True and run.get('send') and run.get('receive')):
                examples['prefill'].setdefault(source['tp'],dict(artifact=path,sha256=digest,run_index=index,
                    batch=step['prefill'],frequency_mhz=2520,step=step))
            target=topology.get('decode',{})
            if not target or run.get('layout')!='pd' or not full(target):continue
            ids=set(run.get('held_kv_tokens',{}));batch=len(ids);count=run.get('output_tokens',raw.get('output_tokens'))
            tokens=run.get('token_ids',[]);usages=run.get('usages',[])
            if (batch<2 or not count or len(tokens)!=batch or len(usages)!=batch
                    or any(len(t)!=count or u.get('completion_tokens')!=count for t,u in zip(tokens,usages))):continue
            actual=[e for e in run.get('events',[]) if e.get('instance')==target['id'] and e.get('role')=='decode'
                and e.get('decode')==batch and not e.get('prefill') and e.get('tokens',0)>0
                and set(e.get('request_ids',[]))==ids]
            if len(actual)==count-1:
                examples['decode'].setdefault(target['tp'],dict(artifact=path,sha256=digest,run_index=index,
                    batch=batch,frequency_mhz=2520,executed_decode_steps=len(actual),request_ids=sorted(ids)))
    missing={role:sorted(tps-set(examples[role])) for role,tps in required.items() if tps-set(examples[role])}
    if missing:raise ValueError('missing real multi-request phase steps for calibrated TP: '+json.dumps(missing))
    return dict(passed=True,required_tps={k:sorted(v) for k,v in required.items()},examples=examples,
        engine_source_sha256=current_engine,examined_artifacts=examined,
        energy_values_imported=False,
        scope='independent real P multi-request steps and D simultaneous decode batches, on calibrated TPs at full frequency')


def checked_admission(path,image):
    from .admission_validation import verify_raw
    raw=read(path)
    if not raw.get('passed') or not raw.get('complete'):raise ValueError('real KV boundary stage did not complete')
    current_sources(raw['source_files']);engine_provenance(raw['provenance_before'],image)
    engine_provenance(raw['provenance_after'],image)
    if sha256(raw['profiles'])!=raw['profile_sha256']:raise ValueError('admission profiles changed')
    return verify_raw(raw)


def collect(plan,out):
    out=Path(out).resolve()
    if out.exists():raise ValueError('refusing to overwrite mechanism evidence')
    out.mkdir(parents=True)
    evidence={s:{k:dict(passed=False,artifact=None,sha256=None,reason='missing executed field-specific evidence')
        for k in fields} for s,fields in REQUIRED_MECHANISMS.items()}
    report=dict(complete=False,formal_eligible=False,started_s=time.time(),components={},source_files=sources())
    def record(system,field,artifact,scope,details=None):
        if field not in evidence[system]:return
        evidence[system][field]=dict(passed=True,artifact=str(Path(artifact).resolve()),sha256=sha256(artifact),
            scope=scope,details=details)
    def component(name,callback):
        try:
            value=callback();report['components'][name]=dict(passed=value.get('passed',True) if isinstance(value,dict) else True)
            return value
        except (OSError,ValueError,KeyError,RuntimeError,TypeError,subprocess.SubprocessError) as exc:
            report['components'][name]=dict(passed=False,error=str(exc));return None
    cpu=component('cpu_contracts',lambda:cpu_contracts(out/'cpu-contracts'))
    cpu_ok=bool(cpu and cpu['passed']);cpu_artifact=out/'cpu-contracts/summary.json'
    if cpu_ok:
        for system,field in (('dynamollm','length_prediction'),('dynamollm','nine_logical_pools'),('ecoserve','unified_constraints')):
            record(system,field,cpu_artifact,'CPU algorithm contract; hardware execution and independent calibration separate')
    followup=read(plan['followup']);cal=read(followup['calibration_template'])
    search=component('certified_search',lambda:checked_search(cal))
    if search and cpu_ok:
        checked=out/'search.checked.json';write(checked,dict(search,cpu_artifact=str(cpu_artifact),cpu_sha256=sha256(cpu_artifact)))
        for field in ('independent_parallel_search','instance_search','interconnect_placement'):
            record('distserve',field,checked,'executed CPU search plus current certified TP/transport measurements')
    eco=component('eco_hardware',lambda:checked_eco(plan['eco_result'],cal['image']))
    if eco:
        for field in eco:record('ecoserve',field,plan['eco_result'],'engine timeline, stationary KV and ordinary-output comparison')
    dynamo_path=Path(plan['dynamo_setup'])/'run'
    dynamo=component('dynamo_hardware',lambda:checked_dynamo(dynamo_path,cal['image']))
    if dynamo:
        for field,passed in dynamo.items():
            if passed is True:record('dynamollm',field,dynamo_path/'mechanisms.json','recomputed real-period action/output/power audit')
    smoke_path=Path(plan['smoke_out'])/'run'
    smoke=component('controller_hardware',lambda:checked_smoke(smoke_path,cal['image']))
    if smoke:
        for system,checks in smoke.items():
            for field,passed in checks.items():
                if passed:record(system,field,smoke_path/'summary.json','recomputed ordinary-token, clock and instantaneous-power audit')
        if cpu_ok:record('mixed_dvfs','feasible_energy_frequency',cpu_artifact,
            'CPU energy-minimum/slack contract with successful actual mixed-DVFS output/clock smoke',
            dict(hardware_artifact=str(smoke_path/'summary.json'),hardware_sha256=sha256(smoke_path/'summary.json')))
    calibration_path=Path(followup['calibration_out'])/'summary.json'
    calibration=component('independent_calibration',lambda:checked_calibration(calibration_path))
    if calibration:
        checked=out/'calibration.checked.json';write(checked,calibration)
        for system,passed in calibration['systems'].items():
            if passed:record(system,'independent_calibration',checked,'three calibration datasets with real full confirmations and current execution source')
    batches=component('hardware_phase_batching',lambda:checked_batches(cal,calibration_path))
    if batches:
        checked=out/'phase-batching.checked.json';write(checked,batches)
        record('distserve','phase_batching',checked,batches['scope'])
    admission=component('real_kv_boundary',lambda:checked_admission(Path(plan['admission_out'])/'raw.json',cal['image']))
    if admission and admission.get('distserve_pd_boundary') and admission.get('target_kv_reserved_before_producer') and admission.get('target_staging_reserved_before_producer'):
        record('distserve','kv_admission',Path(plan['admission_out'])/'raw.json',
            'actual mixed and decode-target KV boundaries; acknowledged shared KV/staging reservations precede the real producer step')
    else:
        evidence['distserve']['kv_admission']['reason']=('real mixed KV boundary verified; complete DistServe target-reservation path still missing'
            if admission else 'dedicated real-engine KV boundary evidence missing; CPU checks and ordinary smoke are insufficient')
    if report['source_files']!=sources():
        for fields in evidence.values():
            for proof in fields.values():proof.update(passed=False,reason='source changed during evidence collection')
    destination=Path(plan.get('mechanisms_out',str(out/'mechanisms.certified-v2.json'))).resolve()
    if destination.exists():raise ValueError('refusing to overwrite existing mechanism registry')
    write(destination,evidence)
    report.update(complete=True,finished_s=time.time(),registry=str(destination),registry_sha256=sha256(destination),
        missing=baseline_gaps(evidence),
        baseline_mechanisms_complete=not baseline_gaps(evidence),
        note='collection complete does not imply all mechanisms or formal performance targets passed')
    write(out/'summary.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=collect(read(args.manifest),args.out)
    print(json.dumps({k:result[k] for k in ('complete','baseline_mechanisms_complete','missing','registry')}))


if __name__=='__main__':main()
