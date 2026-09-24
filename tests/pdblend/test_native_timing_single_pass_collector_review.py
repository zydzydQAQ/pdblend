"""Independent CPU checks of single-pass execution, ownership and cleanup."""
import asyncio
from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest

from pdblend.profile.collection import native_timing_collect as collector
from pdblend.profile.collection import native_timing_capacity as capacity
from pdblend.profile.collection.native_timing_single_pass import SCHEMA, COLLECTION_SCHEMA


@pytest.mark.parametrize('fault', [None, 'window', 'cleanup'])
def test_single_pass_keeps_holdouts_uses_all_owners_and_releases_failed_fleet(tmp_path, monkeypatch, fault):
    from pdblend.engine import launcher
    from pdblend.bench import metering
    from pdblend_baselines import resident_campaign
    from pdblend_runtime import cleanup, probe

    points = [dict(role='decode', batch=1, prompt_tokens=256+i*16, output_tokens=64,
                   frequency_mhz=1500, purpose='training' if i<8 else 'holdout',
                   seed=9701 if i<8 else 9702, repeats=1) for i in range(16)]
    plan = dict(schema=SCHEMA, model_id='Qwen2.5-7B-Instruct', tp=1, pp=1, resident_instances=8,
                capacity_policy=capacity.CAPACITY_POLICY, points=points,
                holdout_limits={}, required_remaining=[], qualification_level='single_pass_development')
    plan_path = tmp_path/'plan.json'; plan_path.write_text(json.dumps(plan))
    out = tmp_path/'raw'; out.mkdir()
    args = SimpleNamespace(model='/models/Qwen2.5-7B-Instruct', gpus=list(range(8)), base_port=20000,
        out=out, point_plan=plan_path, collect_runtime=False, timing_first=False,
        power_pilot_plan=None, request_cycle_plan=None, layout_energy_plan=None)
    calls=[]; active=set(); peak=[]; gate=asyncio.Event()

    class Instance:
        def __init__(self, spec): self.spec=spec; self.events=[]; self.live=False
        def start(self): self.live=True; self.events.append(dict(kind='start',instance=self.spec.instance_id))
        def wait_ready(self, **kwargs): pass

    class Fleet:
        def __init__(self, specs, path): self.instances={s.instance_id:Instance(s) for s in specs}
        def __getitem__(self, iid): return self.instances[iid]

    class Sampler:
        def start(self): calls.append('sampler_started')

    class Meter:
        def __init__(self, *args, **kwargs): pass
        def sampler(self, **kwargs): return Sampler()

    async def capabilities(specs):
        keys=('model_id','model_hash','tokenizer_hash','engine_revision','source_revision','image_digest')
        return {s.instance_id:dict({k:k for k in keys},tp=1,pp=1,gpu_uuids=[f'GPU-{i}'])
                for i,s in enumerate(specs)}

    async def drains(specs): return []
    async def warmup(*args): return []

    async def pre_capacity(spec, point, path, plan, identity):
        original={k:v for k,v in point.items() if k!='repeat'}
        assert spec.instance_id == f'pd-timing-{points.index(original)%8}'
        assert point['repeat']==0 and point['repeats']==1
        return None

    async def window(spec, meter, point, path, before_measure=None):
        assert point['purpose'] in ('training','holdout')  # No isolation probes.
        assert before_measure is None
        calls.append(('window',spec.instance_id,point['prompt_tokens'],point['purpose']))
        active.add(spec.instance_id); peak.append(len(active))
        if len(active)==8: gate.set()
        try:
            await gate.wait()
            if fault=='window' and spec.instance_id=='pd-timing-2': raise RuntimeError('raw window failed')
            await asyncio.sleep(0)
            raw=dict(point=point,start_s=10.,end_s=20.,power_samples=[(11.,[100.])])
            path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(raw))
            return raw
        finally:
            active.remove(spec.instance_id)

    def partition(plan, windows, *, identities):
        rows=list(windows); assert len(identities)==8
        assert len(rows)==16 and sum(r['raw']['point']['purpose']=='holdout' for r in rows)==8
        return dict(measured=rows,unsupported=[])

    def fit(partition, **kwargs):
        q=kwargs['measurement_qualification']
        assert q['qualified'] is False and q['parallel_qualified'] is False
        assert q['mode']=='parallel_development_unqualified' and q['checks']==[]
        assert len(kwargs['raw_bindings'])==16
        return dict(component_qualified=False, component={'holdout_diagnostics_retained':True})

    def clean(fleet, *args):
        assert not active
        calls.append('cleanup')
        for instance in fleet.instances.values(): instance.live=False
        return ['physical cleanup failed'] if fault=='cleanup' else []

    async def empty(fleet, *args):
        assert not any(i.live for i in fleet.instances.values())
        calls.append('physical_cleanup')
        return dict(passed=fault!='cleanup')

    async def no_probe(*args, **kwargs): raise AssertionError('single-pass must not run interference probes')

    monkeypatch.setenv('PDBLEND_GPU_UUIDS',','.join(f'GPU-{i}' for i in range(8)))
    monkeypatch.setattr(launcher,'Fleet',Fleet); monkeypatch.setattr(metering,'Gpus',Meter)
    monkeypatch.setattr(resident_campaign,'model_load_lock',nullcontext)
    monkeypatch.setattr(resident_campaign,'verify_endpoints',capabilities)
    monkeypatch.setattr(resident_campaign,'drain_endpoints',drains)
    monkeypatch.setattr(resident_campaign,'warmup_endpoints',warmup)
    monkeypatch.setattr(collector,'adopt_warmup_generations',lambda specs,*args:specs)
    monkeypatch.setattr(collector,'capacity_before_window',pre_capacity)
    monkeypatch.setattr(collector,'window',window)
    monkeypatch.setattr(collector,'audit_window',lambda *args,**kwargs:[dict(latency_ms=1.)])
    monkeypatch.setattr(capacity,'partition_windows',partition); monkeypatch.setattr(capacity,'fit_measured_partition',fit)
    monkeypatch.setattr(cleanup,'cleanup_owned',clean); monkeypatch.setattr(collector,'verify_compute_empty',empty)
    monkeypatch.setattr(probe,'call',no_probe)

    result=asyncio.run(collector.collect(args,plan))
    assert result['schema']==COLLECTION_SCHEMA and result['complete'] is (fault is None)
    assert result['status']==('passed' if fault is None else 'failed')
    assert calls[-2:]==['cleanup','physical_cleanup'] and not active and max(peak)==8
    assert not result['formal_eligible'] and not result['independent_parallel_qualification']
    assert [e['phase'] for e in result['phase_events']]==['startup','timing']
    if fault!='window':
        assert len([c for c in calls if isinstance(c,tuple)])==16
        assert result['component_qualified'] is False
    else:
        assert result['failed_phase']=='timing' and not result.get('timing_component')
    assert json.loads((out/'completion.json').read_text())['complete'] is (fault is None)
