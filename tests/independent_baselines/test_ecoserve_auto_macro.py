import asyncio
import importlib.util
from pathlib import Path

import pytest

from pdblend_baselines.ecoserve.auto_macro import commit_baselines, qualification_checks, specs_and_config
from pdblend_baselines.ecoserve.auto_macro_replay import replay


@pytest.mark.parametrize('model,tp', [('7B',1),('14B',1),('32B',2)])
def test_four_native_engines_keep_original_macro_parameters(tmp_path, model, tp):
    profile=tmp_path/'author.csv';profile.write_text('Length,Prefill Time\n16,20\n4096,526\n')
    specs,config=specs_and_config(f'Qwen2.5-{model}-Instruct',tp,list(range(4*tp)),18000,profile)
    assert len(specs)==4 and sorted(g for spec in specs for g in spec.gpus)==list(range(4*tp))
    assert all('pdblend_runtime.serve' in spec.command() for spec in specs)
    assert config['eco_initial_instances']==3
    assert (config['eco_scale_period_s'],config['eco_history_window_s'])==(5,60)
    assert (config['eco_macro_lower'],config['eco_macro_upper'])==(2,3)
    assert (config['slo_ttft_s'],config['slo_tpot_s'])==(5,.15)


def test_quiet_duration_and_outputs_do_not_qualify_automatic_macro():
    checks,actions=qualification_checks([dict(kind='eco_scale_observation')],[],{},[dict(ok=True)])
    assert not actions and not checks['automatic_split'] and not checks['automatic_merge']
    assert not checks['actual_hold'] and not checks['split_live_kv_ack']


def test_merge_continuity_uses_survivor_observation_after_removed_member_drain():
    old = dict(native_at_s=10, kv_allocations={'finished_during_drain': [[1]]})
    recent = dict(native_at_s=35, kv_allocations={'still_running_at_commit': [[2]]})
    rows = [dict(kind='eco_membership_prepare', observed_engine_states={'a':old}),
            dict(kind='eco_control_confirmed', instance_id='a', noop=True, observed_engine_state=recent),
            dict(kind='eco_control_confirmed', instance_id='a', noop=True, observed_engine_state=old)]
    before = commit_baselines(rows)
    assert before['a'] is recent
    assert rows[0]['observed_engine_states']['a'] is old


def test_automatic_split_merge_require_own_clock_receipts_live_kv_and_flush():
    journal=[]
    for operation,trigger,version,before,after,path in [
            ('add','mean_ttft',1,[['a','b','c']],[['a','b'],['c','d']],'/baseline/clock'),
            ('remove','saved_tpot',2,[['a','b'],['c','d']],[['a','b','c']],'/baseline/park')]:
        common=dict(origin='paper_supplement',operation=operation,trigger=trigger,instance_id='d',before=before)
        journal += [dict(kind='eco_membership_prepare',**common),
                    dict(kind='eco_http_receipt',instance_id='d',path=path,response=dict(acknowledged=True)),
                    dict(kind='eco_membership_commit',version=version,after=after,split=operation=='add',
                         merge=operation=='remove',**common)]
    journal += [dict(kind='eco_admission',controls=[dict(send_output=False)]),
                dict(kind='eco_client_sse',request_id='r',at_s=2,payload=dict(token_index=1,token_ids=[123]))]
    held={('r',1):dict(request_id='r',token_index=1,token_ids=[123],observed_s=1)}
    checks,_=qualification_checks(journal,[dict(version=1,observed=True),dict(version=2,observed=True)],held,[dict(ok=True)])
    assert all(checks.values())
    checks,_=qualification_checks(journal,[dict(version=1,observed=True)],held,[dict(ok=True)])
    assert not checks['merge_live_kv_ack']


def test_cpu_replay_triggers_original_candidates_and_labels_modeled_progress(tmp_path):
    profile=tmp_path/'author.csv'
    profile.write_text('Length,Prefill Time\n16,20\n128,24\n4096,526\n7168,974\n')
    _,config=specs_and_config('Qwen2.5-7B-Instruct',1,list(range(4)),18000,profile)
    rows=[(t,[1]*7168,512) for t,n in ((0,48),(20,24),(40,24)) for _ in range(n)]
    rows += [(t,[1]*128,512) for t in range(110,286,2)]
    result=asyncio.run(replay(config,rows))
    assert result['split_candidate'] and result['merge_candidate']
    assert not result['hardware_executed'] and not result['native_evidence']
    assert all(row['at_s']%5==0 for row in result['candidates'])


@pytest.mark.asyncio
async def test_real_http_quiet_window_is_inconclusive_without_automatic_actions(tmp_path,monkeypatch):
    pytest.importorskip('fastapi')
    spec=importlib.util.spec_from_file_location('auto_macro_http_fixture',Path(__file__).with_name('test_ecoserve_run_native_http.py'))
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    from pdblend_baselines.ecoserve.auto_macro import execute_window
    config,identity,trace=fixture.inputs(tmp_path,monkeypatch,count=4)
    config.update(eco_scale_period_s=5,eco_history_window_s=60,eco_initial_instances=3,
                  eco_macro_lower=2,eco_macro_upper=3)
    async with fixture.services(monkeypatch,identity,count=4) as (_,endpoints,_):
        result=await execute_window(config,endpoints,trace,tmp_path/'automatic',.08)
    assert result['status']=='inconclusive' and not result['complete']
    assert result['checks']['continuous_complete_output']
    assert result['checks']['all_native_drains_acknowledged']
    assert not result['automatic_actions']
    assert not result['checks']['automatic_split']
