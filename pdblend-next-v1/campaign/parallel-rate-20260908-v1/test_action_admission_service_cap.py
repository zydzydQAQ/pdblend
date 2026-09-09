"""Original profiles, real planner/reservation/backend, only physical NVML mocked."""
import asyncio,csv,json,time
from dataclasses import replace
from pathlib import Path
import pytest
import os,sys
HOST=Path(os.environ['PDB_TEST_HOST'])
sys.path[:0]=[str(HOST/'src'),str(HOST),'/root/workspace/pdblend/.runtime-deps',str(Path('/root/workspace/pdblend-next-v1/campaign/main-slo-improvement-v1/common'))]
from test_fixed_runtime_v6 import fake_clocks as old_fake_clocks,close

def fake_clocks(c):
    x,w=old_fake_clocks(c)
    x.max_frequency=2520
    x.pending_physical_commands={};x.physical_command_uncertainty=[];x.physical_command_sequence=0
    x.transaction_writes=[];x.write_guard_journal=None
    return x,w

from ecopadg.serving.runtime import Controller
from ecopadg.serving.backend import HttpEngineBackend,ClockWriteUncertain,ClockEligibilityExpired
from ecopadg.serving.state import ExpiredPlan
from ecopadg.serving.types import InstanceState,RuntimeSnapshot,RequestBudget
from ecopadg.serving.idle_admission import current_clock_first_plan

R=Path('/root/workspace/pdblend-next-v1/campaign/main-slo-improvement-v7')


def fixture(host='B'):
    cell=next((R/host/'fixed-screen-001/results/cells').glob('*alpaca*repeat1'))
    config=json.loads((cell/'runtime_config.json').read_text())
    config['journal']='/tmp/unused-p2-cpu-clock'
    config['observed_idle_admission_frequency_v2']=True
    controller=Controller(config);now=time.time()
    controller.state.snapshot=RuntimeSnapshot(1,now,tuple(InstanceState(i['id'],'mixed',i['tp'],
        tuple(i['gpus']),now,0,2520,60000,0,0) for i in config['instances']))
    request=RequestBudget('first',now,38,min(config['output_prior'],85),1,.1,
        output_limit=85,hard_deadline_s=now+100)
    clocks,writes=fake_clocks(controller)
    clocks.gpus=tuple(g for i in config['instances'] for g in i['gpus'])
    clocks.epochs={g:0 for g in clocks.gpus};clocks.applied={g:2520 for g in clocks.gpus}
    clocks.hardware.current_freq=lambda g:2520;clocks.state_lock=controller.state.lock
    controller.backend=HttpEngineBackend(config['instances'],None,clocks)
    controller.backend.exact_frequency_confirmation=True
    controller.backend.unconfirmed_retained_admission_deferral=True
    controller.backend.last={i['id']:{'active':0} for i in config['instances']}
    original=controller.planner.plan(controller.state.snapshot,(request,),now=now)
    assert original.feasible
    return controller,clocks,writes,request,original


@pytest.mark.parametrize('host',['A','B','C'])
def test_current_covered_first_plan_preserves_actual_request_and_confirms_full_tp(host):
    async def case():
        c,x,w,r,p=fixture(host)
        try:
            async with c.action_lock:
                chosen=await current_clock_first_plan(c,p,r)
                assert chosen.frequencies[0].frequency_mhz==2520
                assert not c.state.reservations and not w
                assert chosen.routes[0].predicted_ttft_s<=r.ttft_remaining(time.time())
                await c.state.reserve(chosen,time.time(),r)
                await c.backend.execute(chosen)
                assert await c.backend.confirm(chosen)
                admitted=next(i.requests[0] for i in c.state.snapshot.instances if i.requests)
                assert admitted.pending_frequency_mhz==2520
                assert (admitted.arrival_s,admitted.input_tokens,admitted.output_limit,admitted.ttft_s,admitted.tpot_s)==(r.arrival_s,r.input_tokens,r.output_limit,r.ttft_s,r.tpot_s)
                assert not w
        finally:await close(c,x)
    asyncio.run(case())


@pytest.mark.parametrize('kind',['tp_disagreement','unprofiled_idle405','stale_native','native_active','missing_native','ownership_missing','read_error','read_takes_past_expiry'])
def test_unknown_first_clock_never_writes_or_reserves(kind):
    async def case():
        c,x,w,r,p=fixture();iid=p.routes[0].decode_id
        try:
            if kind=='tp_disagreement':x.hardware.current_freq=lambda g:1500 if g==0 else 2520
            elif kind=='unprofiled_idle405':x.hardware.current_freq=lambda g:405
            elif kind=='stale_native':c.state.snapshot=replace(c.state.snapshot,instances=tuple(replace(i,timestamp_s=time.time()-10) for i in c.state.snapshot.instances))
            elif kind=='native_active':c.backend.last[iid]['active']=1
            elif kind=='missing_native':c.backend.last[iid]={}
            elif kind=='ownership_missing':x.gpus=()
            elif kind=='read_error':
                def fail(g):raise RuntimeError('actual read failed')
                x.hardware.current_freq=fail
            elif kind=='read_takes_past_expiry':
                p=replace(p,expires_s=time.time()+.001)
                def slow(g):time.sleep(.005);return 2520
                x.hardware.current_freq=slow
            async with c.action_lock:
                with pytest.raises(ExpiredPlan):await current_clock_first_plan(c,p,r)
            assert not w and not c.state.reservations and c.failure is None
        finally:await close(c,x)
    asyncio.run(case())


def test_current_clock_still_requires_actual_new_request_profile_and_deadline():
    async def case():
        c,x,w,r,p=fixture()
        try:
            async with c.action_lock:
                for invalid in (replace(r,input_tokens=100000),replace(r,ttft_s=.00001)):
                    with pytest.raises(ExpiredPlan):await current_clock_first_plan(c,p,invalid)
            assert not w and not c.state.reservations
        finally:await close(c,x)
    asyncio.run(case())


@pytest.mark.parametrize('kind',['reserved_first_token','native_work','kv_owner','transfer_owner'])
def test_existing_work_and_promises_are_never_replaced(kind):
    async def case():
        c,x,w,r,p=fixture();iid=p.routes[0].decode_id
        try:
            changes=dict(requests=(replace(r,request_id='older',pending_frequency_mhz=1500),))
            if kind=='native_work':changes=dict(running=1)
            if kind=='kv_owner':changes=dict(kv_allocations=(('old',16),))
            if kind=='transfer_owner':changes=dict(transfer_allocations=(('old',16),))
            c.state.snapshot=replace(c.state.snapshot,instances=tuple(replace(i,**changes) if i.instance_id==iid else i for i in c.state.snapshot.instances))
            before=c.state.snapshot
            def no_read(g):raise AssertionError('existing phase must not be reselected')
            x.hardware.current_freq=no_read
            async with c.action_lock:assert await current_clock_first_plan(c,p,r) is p
            assert c.state.snapshot is before and not w
        finally:await close(c,x)
    asyncio.run(case())


def test_default_off_has_no_new_nvml_reads_or_policy_changes():
    async def case():
        c,x,w,r,p=fixture();c.config['measured_frequency_write_guard_v1']=False
        try:
            x.hardware.current_freq=lambda g:pytest.fail('default off cannot observe hardware')
            assert await current_clock_first_plan(c,p,r) is p
        finally:await close(c,x)
    asyncio.run(case())


def test_new_dynamic_third_uses_current_published_topology_and_full_owner():
    async def case():
        c,x,w,r,p=fixture('A')
        try:
            original=c.state.snapshot.instances[0]
            third=replace(original,instance_id='actual-owned-third',gpus=(5,))
            c.backend.instances[third.instance_id]=dict(id=third.instance_id,gpus=[5],tp=1)
            c.backend.last[third.instance_id]={'active':0};x.gpus=(*x.gpus,5);x.epochs[5]=0;x.applied[5]=2520
            c.state.snapshot=replace(c.state.snapshot,instances=(*c.state.snapshot.instances,third))
            p=next(p for p in c.planner.candidates(c.state.snapshot,r,time.time()) if p.routes[0].decode_id==third.instance_id)
            async with c.action_lock:
                chosen=await current_clock_first_plan(c,p,r)
                assert chosen.routes[0].decode_id==third.instance_id and chosen.frequencies[0].frequency_mhz==2520
                await c.state.reserve(chosen,time.time(),r);await c.backend.execute(chosen)
                assert await c.backend.confirm(chosen)
            assert not w
        finally:await close(c,x)
    asyncio.run(case())


def test_observation_holds_actual_writer_and_state_locks_before_reservation(tmp_path):
    async def case():
        c,x,w,r,p=fixture();seen=[];x.write_guard_journal=str(tmp_path/'events.jsonl')
        def observe(g):
            seen.append((c.action_lock.locked(),x.lock.locked(),c.state.lock.locked(),not c.state.reservations))
            return 2520
        x.hardware.current_freq=observe
        try:
            async with c.action_lock:await current_clock_first_plan(c,p,r)
            assert len(seen)==2 and all(all(v) for v in seen)
            e=json.loads((tmp_path/'events.jsonl').read_text());assert e['allowed'] and len(e['observations'])==2 and e['physical_write_attempts']==[]
        finally:await close(c,x)
    asyncio.run(case())


def test_later_real_frequency_uncertainty_is_not_hidden_by_first_selection():
    async def case():
        c,x,w,r,p=fixture()
        try:
            async with c.action_lock:
                chosen=await current_clock_first_plan(c,p,r)
                await c.state.reserve(chosen,time.time(),r)
                x.hardware.current_freq=lambda g:1500
                # Actual verification/fallback remains the v7 physical path.
                with pytest.raises(ClockEligibilityExpired):await c.backend.execute(chosen)
            assert not await c.backend.confirm(chosen)
        finally:await close(c,x)
    asyncio.run(case())


def test_frozen_common_source_and_parent_feature_defaults():
    import hashlib,importlib.util
    root=Path(__file__).resolve().parent
    manifests=[json.loads((root/'hosts'/f'{m}-fixed-p4/manifest.json').read_text()) for m in ('7b','14b','32b')]
    assert all(x['files']==manifests[0]['files'] for x in manifests)
    assert len({x['common_controller_sha256'] for x in manifests})==1
    manifest=json.loads((HOST/'manifest.json').read_text())
    assert all(hashlib.sha256((HOST/n).read_bytes()).hexdigest()==d for n,d in manifest['files'].items())
    parent=Path(manifest['parent_manifest']['path']).parent
    spec=importlib.util.spec_from_file_location('build_four',root/'build_p4.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    for name,transform in (('runtime.py',mod.runtime),('backend.py',mod.backend)):
        assert (HOST/'src/ecopadg/serving'/name).read_text()==transform((parent/'src/ecopadg/serving'/name).read_text())
    for name in ('idle_admission.py','planner.py','profiles.py','frequency.py','tails.py'):
        assert (HOST/'src/ecopadg/serving'/name).read_bytes()==(parent/'src/ecopadg/serving'/name).read_bytes()


@pytest.mark.parametrize('wakeup',['measured_target','still_unconfirmed'])
def test_actual_all_tp_idle405_preserves_original_deferred_wakeup_then_rechecks(tmp_path,wakeup):
    async def case():
        c,x,w,r,p=fixture();x.write_guard_journal=str(tmp_path/'idle.jsonl')
        x.hardware.current_freq=lambda g:405;x.hardware.clock_idle=lambda g:True
        try:
            async with c.action_lock:
                chosen=await current_clock_first_plan(c,p,r)
                assert chosen.frequencies==p.frequencies and chosen.frequencies[0].frequency_mhz!=405
                await c.state.reserve(chosen,time.time(),r);await c.backend.execute(chosen)
                members=c.backend.instances[chosen.routes[0].decode_id]['gpus']
                assert all(g in x.deferred for g in members)
                assert all(x.applied[g]==chosen.frequencies[0].frequency_mhz for g in members)
                events=[json.loads(line) for line in (tmp_path/'idle.jsonl').read_text().splitlines()]
                event=next(e for e in events if e['kind']=='idle_first_admission_clock')
                assert not event['observed_at_target'] and event['requires_deferred_wakeup_verification']
                assert {v['observed_mhz'] for v in event['observations']}=={405}
                x.hardware.clock_idle=lambda g:False
                if wakeup=='measured_target':
                    x.hardware.current_freq=lambda g:chosen.frequencies[0].frequency_mhz
                    await x.verify_deferred();assert not x.deferred
                else:
                    with pytest.raises(ClockWriteUncertain):await x.verify_deferred()
                    assert x.deferred
        finally:await close(c,x)
    asyncio.run(case())


def test_only_one_tp_member_idle_is_not_sufficient_for_legacy_wakeup():
    async def case():
        c,x,w,r,p=fixture();x.hardware.current_freq=lambda g:405
        x.hardware.clock_idle=lambda g:g==0
        try:
            async with c.action_lock:
                with pytest.raises(ExpiredPlan):await current_clock_first_plan(c,p,r)
            assert not w and not c.state.reservations
        finally:await close(c,x)
    asyncio.run(case())

@pytest.mark.parametrize('kind',['pending','uncertain','both'])
def test_prior_physical_uncertainty_blocks_even_reads(kind):
    async def case():
        c,x,w,r,p=fixture()
        try:
            if kind in ('pending','both'):x.pending_physical_commands={1:object()}
            if kind in ('uncertain','both'):x.physical_command_uncertainty=[dict(physical_state_unknown=True)]
            x.hardware.current_freq=lambda g:pytest.fail('unresolved physical state must block reads')
            async with c.action_lock:
                with pytest.raises(ClockWriteUncertain):await current_clock_first_plan(c,p,r)
            assert not w and not c.state.reservations
        finally:await close(c,x)
    asyncio.run(case())


def test_new_feature_requires_explicit_opt_in():
    async def case():
        c,x,w,r,p=fixture();c.config.pop('observed_idle_admission_frequency_v2')
        try:
            x.hardware.current_freq=lambda g:pytest.fail('default-off must not read')
            assert await current_clock_first_plan(c,p,r) is p
        finally:await close(c,x)
    asyncio.run(case())
