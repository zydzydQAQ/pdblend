"""CPU-only builder tests: immutable inputs, source lineage and local import closure."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT=Path(__file__).resolve().parents[2]/'scripts/2026-09-24_prepare_pdblend_native_timing.py'
spec=importlib.util.spec_from_file_location('prepare_pd_native_timing_test',SCRIPT)
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return builder.binding(path)


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    from pdblend.profile.collection import native_serving_cycles
    project=tmp_path/'project';source=tmp_path/'repaired-source';source.mkdir()
    (source/'protected.py').write_text('fixed_base = True\n')
    put(source/'manifest.json',dict(files={'protected.py':builder.binding(source/'protected.py')['sha256']}))
    scripts=project/'scripts';scripts.mkdir(parents=True)
    (scripts/'2026-09-22_enqueue_parallel_profiles.py').write_text('''
import json,hashlib,shutil
from pathlib import Path
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify_snapshot(path,expected):
    actual={str(p.relative_to(path)):sha(p) for p in path.rglob('*') if p.is_file() and p.name!='manifest.json'}
    if actual!=expected:raise ValueError('snapshot checksum mismatch')
def freeze_source(staging,out):
    dest=out/'synthetic-source';shutil.copytree(staging,dest)
    files={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (dest/'manifest.json').write_text(json.dumps(dict(files=files,source_sha256='synthetic')))
    return dest,'synthetic'
''')
    modules=set(builder.OVERLAYS+builder.CYCLE_OVERLAYS+builder.LAYOUT_OVERLAYS+builder.STAGE_OVERLAYS
                +builder.FREQUENCY_OVERLAYS+builder.FREQUENCY_QUERY_OVERLAYS+builder.FREQUENCY_RUNTIME_OVERLAYS)
    modules|={'pdblend/profile/collection/native_runtime_'+s+'.py' for s in ('collect','audit')}
    modules|={'pdblend/profile/collection/native_timing_'+s+'.py' for s in ('plan_v2','capacity','replay','replay_v2')}
    for name in modules:
        p=project/'src'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('value = 1\n')
    # An indirect dependency is absent from the old source. It must be copied,
    # while imports pointing at existing frozen bytes preserve those bytes.
    p=project/'src/pdblend/profile/query/native_cycle_model.py'
    p.write_text('from .cycle_dependency import VALUE\n')
    p=project/'src/pdblend/profile/query/cycle_dependency.py';p.write_text('VALUE = 42\n')
    verification=tmp_path/'verification.json';put(verification,{})
    ledger=tmp_path/'ledger.json';put(ledger,{'development':True})
    trace=put(tmp_path/'trace.json',{'requests':[]});confirmation=put(tmp_path/'confirmation.json',{})
    anchor=put(tmp_path/'anchor.json',{})
    provenance=tmp_path/'bindings.json';put(provenance,dict(inputs={'7b':dict(rate_anchor=anchor,
        datasets={'alpaca':dict(confirmation=confirmation,tuning_trace=trace)})}))
    plan=dict(model_id='Qwen2.5-7B-Instruct',tp=1,pp=1,query_ledger=builder.binding(ledger),
        query_bindings=builder.binding(provenance),scope='native_timing_component_only',
        required_remaining=['power'],points=[dict(purpose='training'),dict(purpose='holdout')])
    cycle=dict(model_id=plan['model_id'],tp=1,pp=1,query_ledger=plan['query_ledger'],query_provenance=plan['query_bindings'],
        points=[dict(parent_trace=trace,purpose=phase,duration_s=60) for phase in ('training','holdout') for _ in range(8)])
    cycle_path=tmp_path/'cycle.json';put(cycle_path,cycle)
    monkeypatch.setattr(builder,'ROOT',project);monkeypatch.setattr(builder,'BASE',source);monkeypatch.setattr(builder,'VERIFY',verification)
    monkeypatch.setattr(builder,'build_plan',lambda *args:deepcopy(plan))
    monkeypatch.setattr(native_serving_cycles,'validate_cycle_plan',lambda p:p)
    return dict(root=project,source=source,ledger=ledger,provenance=provenance,cycle=cycle_path,plan=plan)


def test_default_preparation_retains_no_cycle_and_does_not_enqueue(tmp_path,prepared):
    p=prepared;out=tmp_path/'out'
    review=builder.prepare(out,p['ledger'],p['provenance'])
    inputs=json.loads((out/'inputs.json').read_text());jobs=json.loads((out/'jobs.json').read_text())
    assert not review['collect_request_cycles'] and review['timing_only']
    assert 'request_cycle_plan' not in inputs and '--request-cycle-plan' not in jobs[0]['payload']['argv']
    assert review['base_manifest']==builder.binding(p['source']/'manifest.json')
    assert review['enqueued'] is False and jobs[0]['priority']==690


def test_cycle_is_byte_frozen_input_bound_and_has_complete_source_dependency_closure(tmp_path,prepared):
    p=prepared;out=tmp_path/'out'
    review=builder.prepare(out,p['ledger'],p['provenance'],request_cycle_plan=p['cycle'],source_base=p['source'])
    inputs=json.loads((out/'inputs.json').read_text());job=json.loads((out/'jobs.json').read_text())[0]
    frozen=out/'request-cycle-plan.json'
    assert frozen.read_bytes()==p['cycle'].read_bytes()
    assert inputs['request_cycle_plan']==builder.binding(frozen)
    assert inputs['request_cycle_original_plan']==builder.binding(p['cycle'])
    refs=inputs['request_cycle_raw_inputs'];assert len(refs)==6
    assert {r['path'] for r in refs}>={str(p[k]) for k in ('ledger','provenance','cycle')}
    source=Path(review['source_manifest']['path']).parent
    assert (source/'protected.py').read_bytes()==(p['source']/'protected.py').read_bytes()
    dependency='pdblend/profile/query/cycle_dependency.py'
    assert dependency in review['dependency_overlays'] and (source/dependency).read_text()=='VALUE = 42\n'
    assert '--request-cycle-plan' in job['payload']['argv'] and str(frozen) in job['payload']['argv']
    assert 'pdblend/profile/collection/native_runtime_topology.py' in review['overlays']
    assert job['payload']['input_manifest']==builder.binding(out/'inputs.json') and job['priority']==800
    assert review['request_cycle_windows']==16 and review['request_cycle_service_wall_s']==960
    assert not review['timing_only'] and not review['formal_eligible'] and not review['enqueued']
    for ref in refs:
        assert ref['path']+':'+ref['path']+':ro' in job['payload']['argv']


@pytest.mark.parametrize('change',['model','tp','ledger','provenance','parent_sha'])
def test_mismatched_cycle_input_fails_before_creating_preparation(tmp_path,prepared,change):
    p=prepared;value=json.loads(p['cycle'].read_text())
    if change=='model':value['model_id']='Qwen2.5-14B-Instruct'
    elif change=='tp':value['tp']=2
    elif change=='ledger':value['query_ledger']={'path':'wrong','sha256':'wrong'}
    elif change=='provenance':value['query_provenance']={'path':'wrong','sha256':'wrong'}
    else:value['points'][0]['parent_trace']['sha256']='wrong'
    put(p['cycle'],value)
    with pytest.raises(ValueError):builder.prepare(tmp_path/'out',p['ledger'],p['provenance'],request_cycle_plan=p['cycle'])
    assert not (tmp_path/'out').exists()


def test_bad_source_base_cannot_produce_a_job_or_mutate_old_source(tmp_path,prepared):
    p=prepared;(p['source']/'protected.py').write_text('tampered\n')
    with pytest.raises(ValueError,match='snapshot checksum'):
        builder.prepare(tmp_path/'out',p['ledger'],p['provenance'],source_base=p['source'])
    assert not (tmp_path/'out').exists()


def test_timing_first_is_explicitly_frozen_with_its_replay_dependencies(tmp_path,prepared,monkeypatch):
    from pdblend.profile.collection import native_timing_plan_v2
    p=prepared;plan=dict(p['plan'],schema='pdblend-native-timing-plan/v2')
    path=tmp_path/'v2.json';put(path,plan)
    monkeypatch.setattr(native_timing_plan_v2,'validate_plan',lambda value:value)
    out=tmp_path/'out'
    review=builder.prepare(out,p['ledger'],p['provenance'],point_plan=path,
        request_cycle_plan=p['cycle'],timing_first=True)
    inputs=json.loads((out/'inputs.json').read_text());job=json.loads((out/'jobs.json').read_text())[0]
    assert inputs['timing_first'] is True and inputs['phase_order']==['timing','request_cycles']
    assert review['timing_first'] is True and review['phase_order']==inputs['phase_order']
    assert '--timing-first' in job['payload']['argv']
    assert set(builder.STAGE_OVERLAYS)<=set(review['overlays'])
    source=Path(review['source_manifest']['path']).parent
    assert all((source/name).read_bytes()==(p['root']/'src'/name).read_bytes() for name in builder.STAGE_OVERLAYS)


def test_timing_first_cannot_silently_reorder_a_legacy_plan(tmp_path,prepared):
    p=prepared
    with pytest.raises(ValueError,match='explicit v2'):
        builder.prepare(tmp_path/'out',p['ledger'],p['provenance'],timing_first=True)
    assert not (tmp_path/'out').exists()


def frequency_plan(tmp_path, prepared, monkeypatch):
    from pdblend.profile.collection import native_timing_plan_v2
    from pdblend.profile.collection.native_frequency_domain import make_domain,domain_fields
    p=prepared
    domain=make_domain(model_id=p['plan']['model_id'],high_mhz=2100,revision='new-test-domain')
    ref=put(tmp_path/'new-domain.json',domain)
    plan=dict(p['plan'],schema='pdblend-native-timing-plan/v2',frequency_domain_ref=ref,**domain_fields(domain))
    path=tmp_path/'new-domain-plan.json';put(path,plan)
    monkeypatch.setattr(native_timing_plan_v2,'validate_plan',lambda value:value)
    return path,plan


def test_new_frequency_revision_is_bound_in_inputs_source_and_external_mount(tmp_path,prepared,monkeypatch):
    p=prepared;path,plan=frequency_plan(tmp_path,p,monkeypatch);out=tmp_path/'new-out'
    review=builder.prepare(out,p['ledger'],p['provenance'],point_plan=path,timing_first=True)
    inputs=json.loads((out/'inputs.json').read_text());job=json.loads((out/'jobs.json').read_text())[0]
    for key in ('model_id','tp','pp','frequency_domain_ref','frequency_domain','frequency_domain_sha256'):
        assert inputs[key]==plan[key]
    ref=plan['frequency_domain_ref']
    assert ref['path']+':'+ref['path']+':ro' in job['payload']['argv']
    assert set(builder.FREQUENCY_OVERLAYS)<=set(review['overlays'])
    assert set(builder.FREQUENCY_QUERY_OVERLAYS)<=set(review['overlays'])
    assert review['frequency_domain_sha256']==plan['frequency_domain_sha256']
    assert review['timing_only'] and not review['enqueued'] and not review['formal_eligible']


def test_new_frequency_runtime_is_scoped_and_bound_before_source_copy(tmp_path,prepared,monkeypatch):
    from pdblend.profile.collection.native_runtime_collect import build_runtime_plan
    from pdblend.profile.collection.native_frequency_domain import validate_collection_inputs
    p=prepared;path,plan=frequency_plan(tmp_path,p,monkeypatch);out=tmp_path/'runtime-new-out'
    review=builder.prepare(out,p['ledger'],p['provenance'],point_plan=path,timing_first=True,collect_runtime=True)
    inputs=json.loads((out/'inputs.json').read_text())
    assert inputs['runtime_plan']==build_runtime_plan(plan['frequency_domain_ref'])
    assert inputs['runtime_include_transfer'] is False
    assert inputs['runtime_scope']=='capacity_static_clock_L1_off_wake_only'
    assert inputs['phase_order']==['runtime','timing']
    assert set(builder.FREQUENCY_RUNTIME_OVERLAYS)<=set(review['overlays'])
    for field in ('runtime_include_transfer','runtime_plan','runtime_scope'):
        changed=deepcopy(inputs);changed.pop(field)
        with pytest.raises(ValueError,match='bound scoped plan'):
            validate_collection_inputs(plan,changed,collect_runtime=True)
    changed=deepcopy(inputs);changed['runtime_include_transfer']=True
    with pytest.raises(ValueError,match='bound scoped plan'):
        validate_collection_inputs(plan,changed,collect_runtime=True)


@pytest.mark.parametrize('unsupported',['power','cycles','layout','legacy_order','changed_binding'])
def test_new_domain_rejects_unintegrated_supplements_before_source_copy(tmp_path,prepared,monkeypatch,unsupported):
    p=prepared;path,plan=frequency_plan(tmp_path,p,monkeypatch)
    kw=dict(point_plan=path,timing_first=True)
    if unsupported=='power':kw['collect_power_pilot']=True
    elif unsupported=='cycles':kw['request_cycle_plan']=p['cycle']
    elif unsupported=='layout':kw['layout_energy_plan']=p['cycle']
    elif unsupported=='legacy_order':kw['timing_first']=False
    else:Path(plan['frequency_domain_ref']['path']).write_text('{}')
    out=tmp_path/'bad-new-out'
    with pytest.raises(ValueError):builder.prepare(out,p['ledger'],p['provenance'],**kw)
    assert not out.exists()


def test_import_closure_handles_relative_imports_lazy_branches_and_existing_frozen_modules(tmp_path,monkeypatch):
    workspace=tmp_path/'workspace';staging=tmp_path/'staging';staging.mkdir()
    files={'pdblend/pkg/start.py':'from . import peer\ndef later():\n from ..other.leaf import f\n',
           'pdblend/pkg/peer.py':'from pdblend.public import SENTINEL\n',
           'pdblend/other/leaf.py':'def f(): return 1\n',
           'pdblend/public.py':'SENTINEL = "workspace drift"\n',
           'pdblend/__init__.py':'','pdblend/pkg/__init__.py':''}
    for name,text in files.items():
        p=workspace/'src'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
    for name,text in {'pdblend/pkg/start.py':files['pdblend/pkg/start.py'],'pdblend/public.py':'SENTINEL = "frozen"\n'}.items():
        p=staging/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
    monkeypatch.setattr(builder,'ROOT',workspace)
    added,closure=builder._close_imports(staging,['pdblend/pkg/start.py'])
    assert {'pdblend/pkg/peer.py','pdblend/other/leaf.py','pdblend/__init__.py','pdblend/pkg/__init__.py'}<=set(added)
    assert 'pdblend/public.py' in closure and 'pdblend/public.py' not in added
    assert (staging/'pdblend/public.py').read_text()=='SENTINEL = "frozen"\n'


@pytest.fixture
def layout_prepared(tmp_path,prepared,monkeypatch):
    from pdblend.profile.collection import native_layout_energy,native_timing_plan_v2
    p=prepared
    provenance=json.loads(p['provenance'].read_text());provenance['inputs']['32b']=provenance['inputs'].pop('7b')
    put(p['provenance'],provenance)
    plan=dict(p['plan'],schema='pdblend-native-timing-plan/v2',model_id='Qwen2.5-32B-Instruct',tp=2,
              query_provenance=builder.binding(p['provenance']))
    plan.pop('query_bindings');path=tmp_path/'timing-plan.json';put(path,plan)
    layout=json.loads(p['cycle'].read_text());layout.update(model_id=plan['model_id'],tp=2,
        query_provenance=plan['query_provenance'])
    layout['points']=[dict(layout['points'][0],purpose=phase,duration_s=duration)
                      for phase,duration,count in [('training',60.,12),('holdout',150.,24)] for _ in range(count)]
    layout_path=tmp_path/'layout.json';put(layout_path,layout)
    monkeypatch.setattr(native_layout_energy,'validate_layout_plan',lambda v:v)
    monkeypatch.setattr(native_timing_plan_v2,'validate_plan',lambda v:v)
    p.update(point_plan=path,layout=layout_path);return p


def test_layout_component_has_separate_frozen_inputs_and_omits_legacy_pilots(tmp_path,layout_prepared):
    p=layout_prepared;out=tmp_path/'out'
    review=builder.prepare(out,p['ledger'],p['provenance'],point_plan=p['point_plan'],
        collect_runtime=True,layout_energy_plan=p['layout'])
    inputs=json.loads((out/'inputs.json').read_text());job=json.loads((out/'jobs.json').read_text())[0]
    assert inputs['layout_energy_plan']==builder.binding(out/'layout-energy-plan.json')
    assert (out/'layout-energy-plan.json').read_bytes()==p['layout'].read_bytes()
    assert review['collect_layout_energy'] and not review['collect_request_cycles'] and not review['collect_power_pilot']
    assert review['layout_energy_windows']==36 and review['layout_energy_service_wall_s']==4320
    assert '--layout-energy-plan' in job['payload']['argv']
    assert '--request-cycle-plan' not in job['payload']['argv'] and '--power-pilot-plan' not in job['payload']['argv']
    assert job['payload']['formal_eligible'] is False and review['enqueued'] is False
    assert all(name in review['overlays'] for name in builder.LAYOUT_OVERLAYS)


@pytest.mark.parametrize('fault',['no_runtime','legacy_pilot','wrong_ledger','mutated_trace'])
def test_layout_invocation_rejects_incompatible_collection_before_freezing(tmp_path,layout_prepared,fault):
    p=layout_prepared;kw=dict(point_plan=p['point_plan'],collect_runtime=True,layout_energy_plan=p['layout'])
    if fault=='no_runtime':kw['collect_runtime']=False
    elif fault=='legacy_pilot':kw['collect_power_pilot']=True
    else:
        value=json.loads(p['layout'].read_text())
        if fault=='wrong_ledger':value['query_ledger']={'path':'other','sha256':'other'}
        else:value['points'][0]['parent_trace']['sha256']='0'*64
        put(p['layout'],value)
    with pytest.raises(ValueError):builder.prepare(tmp_path/'out',p['ledger'],p['provenance'],**kw)
    assert not (tmp_path/'out').exists()
