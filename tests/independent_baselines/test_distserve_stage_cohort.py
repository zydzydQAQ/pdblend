"""Bound 2+2+4 preparation and actual six-engine barrier logic, without GPUs."""
import asyncio
import importlib.util
import json
from pathlib import Path
import shutil
import threading
import time
from types import SimpleNamespace

import pytest

from pdblend_baselines.distserve import stage_cohort as module
from pdblend_baselines.distserve import stage_collect

ROOT=Path(__file__).parents[2]


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    path=ROOT/'scripts/2026-09-24_prepare_distserve_cohort.py'
    spec=importlib.util.spec_from_file_location('prepare_distserve_three_model',path)
    prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)
    project=tmp_path/'project';(project/'scripts').mkdir(parents=True);(project/'src').mkdir()
    (project/'src/stub.py').write_text('source frozen by real freezer\n')
    name='2026-09-22_enqueue_parallel_profiles.py'
    shutil.copyfile(ROOT/'scripts'/name,project/'scripts'/name)
    monkeypatch.setattr(prepare,'ROOT',project)
    corpus=project/'corpus'
    for index,(member,(model,tp)) in enumerate(module.MODELS.items()):
        root=corpus/('2026-09-22-'+member.removeprefix('distserve-')+'-v1');root.mkdir(parents=True)
        for dataset in module.DATASETS:
            (root/(dataset+'.json')).write_text(json.dumps(dict(model_name=model,
                calibration=[dict(input_tokens=30+index,output_tokens=16),dict(input_tokens=7168,output_tokens=512)],
                tuning=[dict(input_tokens=128+index,output_tokens=32)],
                evaluation=[dict(input_tokens=1,output_tokens=8191)])))
    verification=project/'verification.json';verification.write_text('{}')
    out=project/'package'
    jobs,cohort,source=prepare.prepare(out,corpus,verification,'sha256:test')
    monkeypatch.setenv('PDBLEND_PROFILE_WAVE',str(out/'coord'))
    monkeypatch.setenv('PDBLEND_DIST_PORT_GUARD','1')
    return prepare,out,jobs,cohort,source,monkeypatch


def arguments(out,member):
    model,tp=module.MODELS[member]
    return ['--model',model,'--tp',str(tp),'--gpus',','.join(str(i) for i in range(2*tp)),
        '--point-plan',str(out/'plans'/member/'points.json'),
        '--input-manifest',str(out/'plans'/member/'input-manifest.json'),'--out',str(out/'cpu'/member),
        '--base-port','19000','--preflight-only']


def test_real_freeze_binds_three_model_owned_plans_and_cpu_commands_hide_all_gpus(prepared):
    prepare,out,jobs,cohort,source,monkeypatch=prepared
    assert [job['payload']['gpu_count'] for job in jobs]==[2,2,4]
    assert len({job['payload']['sampling_cohort'] for job in jobs})==1
    assert [row['raw_windows'] for row in cohort['members'].values()]==[414]*3
    for index,job in enumerate(jobs):
        member=job['payload']['cohort_member'];monkeypatch.setenv('PDBLEND_PROFILE_MEMBER',member)
        receipt,_=module.validate_cohort(out/'cohort-inputs.json',member,arguments(out,member))
        assert not receipt['hardware_executed'] and not receipt['formal_eligible'] and not receipt['parallel_qualified']
        cpu=prepare.cpu_command(job,out/'cpu'/member,index)
        image='pdblend:l20-cu128-vllm-v1@sha256:test';docker=cpu[:cpu.index(image)]
        assert '--gpus' not in docker and '--cap-add' not in docker
        assert 'NVIDIA_VISIBLE_DEVICES=void' in docker and cpu[-1]=='--preflight-only'
        assert job['payload']['cohort_dir']==str(out/'coord')
        assert job['payload']['argv'][1]=='-B' and 'stage_ports.py' in job['payload']['argv'][2]
        assert job['payload']['container_name']==job['job_id']
        assert '/proc:/host/proc:ro' in job['payload']['argv']
        plan=json.loads((out/'plans'/member/'points.json').read_text())
        assert min(min(p['lengths']) for p in plan['points'])==30+index
        assert len(plan['points'])==138 and all(p['repeats']==3 for p in plan['points'])
        for f in module.DIST_FREQS:
            shapes=[p['lengths'] for p in plan['points'] if p['role']=='decode' and p['purpose']=='training' and p['frequency_mhz']==f]
            assert all([n] in shapes for n in (4096,6144,7168,7808))
        for role in ('prefill','decode'):
            for purpose in ('training','holdout'):
                for f in module.DIST_FREQS:
                    assert {len(p['lengths']) for p in plan['points'] if (p['role'],p['purpose'],p['frequency_mhz'])==(role,purpose,f)}=={1,2,4}


@pytest.mark.parametrize('bad',['wave','point','corpus','foreign_model','partition','member','missing_peer'])
def test_bound_cohort_cannot_change_members_shapes_or_corpus(prepared,bad):
    _,out,jobs,cohort,_,monkeypatch=prepared
    member='distserve-7b';monkeypatch.setenv('PDBLEND_PROFILE_MEMBER',member)
    args=arguments(out,member)
    if bad=='wave':(out/'coord/wave.json').write_text('{}')
    elif bad=='point':(out/'plans/distserve-14b/points.json').write_text('{}')
    elif bad=='corpus':
        plan=json.loads((out/'plans'/member/'points.json').read_text())
        path=Path(plan['inputs'][0]['path']);value=json.loads(path.read_text());value['tuning'][0]['input_tokens']+=1
        path.write_text(json.dumps(value))
    elif bad=='foreign_model':args[args.index('--model')+1]='Qwen2.5-32B-Instruct'
    elif bad=='partition':args[args.index('--gpus')+1]='0,1,2,3'
    elif bad=='member':monkeypatch.setenv('PDBLEND_PROFILE_MEMBER','distserve-14b')
    elif bad=='missing_peer':
        cohort['members'].pop('distserve-14b');path=out/'cohort-inputs.json';path.write_text(json.dumps(cohort))
        manifest=out/'plans'/member/'input-manifest.json';value=json.loads(manifest.read_text())
        value['cohort_sha256']=module.sha(path);manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError):module.validate_cohort(out/'cohort-inputs.json',member,args)


def test_evaluation_values_never_select_profile_shapes(prepared):
    _,out,_,_,_,_=prepared
    plan=json.loads((out/'plans/distserve-7b/points.json').read_text());corpus=Path(plan['inputs'][0]['path']).parent
    for path in corpus.glob('*.json'):
        value=json.loads(path.read_text());value['evaluation']=[dict(input_tokens=7900,output_tokens=1)]
        path.write_text(json.dumps(value))
    fresh=module.point_plan(corpus,'Qwen2.5-7B-Instruct',1)
    assert fresh['points']==plan['points'] and fresh['selected_envelope']==plan['selected_envelope']
    assert fresh['inputs']!=plan['inputs'], 'whole-corpus bindings still report changed bytes'


@pytest.mark.parametrize('capacity', [dict(max_num_seqs=3,total_kv_tokens=100000,max_num_batched_tokens=8192),
    dict(max_num_seqs=32,total_kv_tokens=30000,max_num_batched_tokens=8192),
    dict(max_num_seqs=32,total_kv_tokens=100000,max_num_batched_tokens=4096)])
def test_actual_native_capacity_keeps_unsupported_candidate_out_of_measurement_and_fit(tmp_path,monkeypatch,capacity):
    calls=[]
    def call(url,method,path,*args):
        calls.append(path)
        assert path=='/baseline/capability'
        return dict(tp=2,state=capacity)
    monkeypatch.setattr(stage_collect,'call',call)
    spec=SimpleNamespace(base_url='cpu://native',tp=2)
    point=dict(role='decode' if capacity['total_kv_tokens']==30000 else 'prefill',
               frequency_mhz=2520,lengths=[7168]*4,purpose='training',repeat=0)
    row=stage_collect.window(spec,None,point,tmp_path/'unsupported.json')
    assert row['status']=='unsupported_engine' and row['capability']['state']==capacity
    assert stage_collect.rows_from_window(row)==[] and calls==['/baseline/capability']


@pytest.mark.parametrize('interference',[False,True])
@pytest.mark.asyncio
async def test_original_native_wave_measures_all_six_engines_and_serializes_every_role_if_unqualified(tmp_path,monkeypatch,interference):
    root=tmp_path/'coord';root.mkdir();members=list(module.MODELS)
    (root/'wave.json').write_text(json.dumps(dict(members=members,cohort_id='cpu-test-cohort',coordinator=True,
        synchronize_parallel_windows=True,keep_peers_resident_until_all_done=True)))
    settled={};lock=threading.Lock();parallel_seen=[];isolated_active=set();isolated_max=[0]
    def fake_window(spec,meter,point,path,*,before_measure=None):
        phase='parallel' if before_measure else 'isolated';repeat=point['repeat']
        if phase=='parallel':
            with lock:settled.setdefault(repeat,set()).add(spec.instance_id)
            before_measure()
            with lock:parallel_seen.append(set(settled[repeat]))
        else:
            with lock:
                isolated_active.add(spec.instance_id);isolated_max[0]=max(isolated_max[0],len(isolated_active))
            time.sleep(.001)
            with lock:isolated_active.remove(spec.instance_id)
        at=time.time();latency=11. if interference and phase=='parallel' and spec.instance_id.endswith('32b-0') else 10.
        return dict(start_s=at,end_s=at+5.,rows=[dict(latency_ms=latency,power_w=100.)])
    monkeypatch.setattr(stage_collect,'window',fake_window)
    monkeypatch.setattr(stage_collect,'rows_from_window',lambda raw:raw['rows'])
    waves=[];profilers=[]
    offset=0
    for member in members:
        count=module.MODELS[member][1]*2
        out=tmp_path/member;out.mkdir()
        profiler=SimpleNamespace(specs=[SimpleNamespace(instance_id=member+'-'+str(i)) for i in range(2)],
            meter=None,out_dir=out,raw=dict(environment=dict(gpu_uuids=[f'GPU-{n}' for n in range(offset,offset+count)])),
            profile_key=SimpleNamespace(as_dict=lambda m=member:dict(member=m)),_checkpoint=lambda:None)
        offset+=count;profilers.append(profiler);waves.append(stage_collect.NativeStageWave(root,member,timeout_s=5))
    await asyncio.gather(*(w.qualify_external(p) for w,p in zip(waves,profilers)))
    assert isolated_max==[1] and len(parallel_seen)==18 and all(len(ids)==6 for ids in parallel_seen)
    assert all(w.parallel is (not interference) for w in waves)
    for p in profilers:
        value=json.loads((p.out_dir/'samples/external-interference.json').read_text())
        assert len(value['comparisons'])==6 and value['common_measurement_windows']['passed']
        assert len(value['common_measurement_windows']['overlap_seconds'])==3
    execution=[]
    async def sample(wave):
        async with wave.measurement():
            execution.append(('start',wave.member))
            await asyncio.sleep(.001)
            execution.append(('done',wave.member))
    await asyncio.gather(*(sample(w) for w in waves))
    if interference:
        assert execution==[(stage,member) for member in members for stage in ('start','done')]
    assert all((root/(member+'.done.json')).is_file() for member in members)


def test_retry_overlays_only_private_ports_and_preserves_every_other_base_byte(prepared):
    prepare,out,_,_,source,_=prepared
    project=out.parent
    for name in ('stage_collect','stage_cohort','stage_ports'):
        relative=Path('pdblend_baselines/distserve')/(name+'.py')
        target=project/'src'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/'src'/relative,target)
    base=json.loads((source/'manifest.json').read_text())
    _,_,new=prepare.prepare(project/'retry',project/'corpus',project/'verification.json','sha256:test',source/'manifest.json')
    actual=json.loads((new/'manifest.json').read_text())
    assert all(actual['files'][name]==value for name,value in base['files'].items())
    assert set(actual['files'])-set(base['files'])=={
        'pdblend_baselines/distserve/'+name+'.py' for name in ('stage_collect','stage_cohort','stage_ports')}
    assert json.loads((source/'manifest.json').read_text())==base
    assert json.loads((project/'retry/source-base-overlay.json').read_text())['all_other_source_files_unchanged']
