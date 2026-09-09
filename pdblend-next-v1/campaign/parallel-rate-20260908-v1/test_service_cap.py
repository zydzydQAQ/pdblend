"""CPU semantics only: synthetic 2400 fixtures are never hardware evidence."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
import pytest
from test_action_admission_service_cap import fixture, close, Controller, HttpEngineBackend
from ecopadg.serving.backend import ClockOwner, ClockEligibilityExpired, ClockWriteUncertain
from ecopadg.serving.frequency import FrequencyPlanner
from ecopadg.serving.profiles import ProfileStore
from ecopadg.serving.types import ControlPlan, FrequencyAction


@pytest.mark.parametrize('ceiling',[2400,2100])
@pytest.mark.parametrize('tp_host',['A','B'])
def test_physical_owner_obeys_ceiling_and_rejects_higher_before_write(ceiling,tp_host):
    async def case():
        c,x,w,r,p=fixture(tp_host)
        try:
            x.max_frequency=ceiling
            x.write_guard=None
            await x.set(x.gpus,ceiling,verify_rise=False)
            assert set(x.applied.values())=={ceiling}
            before=list(w)
            with pytest.raises(ValueError):await x.set(x.gpus,2520)
            assert w==before
            b=HttpEngineBackend(c.config['instances'],None,x,max_frequency=ceiling)
            assert set(b.frequency.values())=={ceiling}
            with pytest.raises(ValueError):HttpEngineBackend(c.config['instances'],None,x)
        finally:await close(c,x)
    asyncio.run(case())


def test_config_requires_real_frequency_and_all_reachable_costs():
    async def case():
        c,x,*_=fixture('A')
        try:
            config=dict(c.config,max_service_frequency_mhz=2400)
            with pytest.raises(ValueError,match='lacks a measured profile'):Controller(config)
            # This cloned store is a named CPU mock, never a registered profile.
            mocked=ProfileStore(tuple(replace(p,frequency_mhz=2400,source_sha256='cpu-only-synthetic')
                if p.frequency_mhz==2520 else p for p in c.profiles.points))
            with patch.object(ProfileStore,'load',return_value=mocked):
                with pytest.raises(ValueError,match='transition costs'):Controller(config)
                costs=[dict(k,source_mhz=2400 if k['source_mhz']==2520 else k['source_mhz'],
                    target_mhz=2400 if k['target_mhz']==2520 else k['target_mhz'],
                    source_sha256='cpu-only-synthetic') for k in config['frequency_costs']]
                d=Controller(dict(config,frequency_costs=costs))
                assert d.planner.max_frequency==2400
                await d.planning_executor.close()
        finally:await close(c,x)
    asyncio.run(case())


@pytest.mark.parametrize('change',[{'strategy':'mixed'},{'allow_pd':True},
    {'dynamic_pools':True},{'measured_frequency_write_guard_v1':False},
    {'max_service_frequency_mhz':True},{'max_service_frequency_mhz':0},
    {'max_service_frequency_mhz':2600}])
def test_unsupported_ceiling_configuration_rejected(change):
    async def case():
        c,x,*_=fixture('A')
        try:
            with pytest.raises(ValueError):Controller(dict(c.config,max_service_frequency_mhz=2400,**change)
                if 'max_service_frequency_mhz' not in change else dict(c.config,**change))
        finally:await close(c,x)
    asyncio.run(case())


def test_original_profiles_cannot_offer_above_ceiling_and_recovery_uses_ceiling():
    async def case():
        c,x,w,r,p=fixture('A')
        try:
            c.planner.max_frequency=2100
            now=time.time()
            plans=c.planner.candidates(c.state.snapshot,replace(r,ttft_s=100,tpot_s=10),now)
            assert plans and all(a.frequency_mhz<=2100 for p in plans for a in p.frequencies)
            assert c.planner.point(c.state.snapshot.instances[0],r,2520,1) is None
            # Stale decode telemetry demands recovery; it must honor the cap.
            inst=replace(c.state.snapshot.instances[0],timestamp_s=now-10,
                requests=(replace(r,emitted=1,first_token_s=now-.01),),frequency_mhz=900)
            snap=replace(c.state.snapshot,instances=(inst,))
            recovery=FrequencyPlanner(c.planner,c.planner.frequency_costs).plan(snap,now)
            assert recovery.frequencies and {a.frequency_mhz for a in recovery.frequencies}=={2100}
        finally:await close(c,x)
    asyncio.run(case())
