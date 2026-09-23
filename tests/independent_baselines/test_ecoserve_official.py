"""Decision traces compared with the pinned author implementation, not a mock rule."""
import ast
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import random
import hashlib
import json

import pytest

from pdblend_baselines.ecoserve.policy import OfficialMacro, PrefillProfile, OutputBuffer

ROOT = Path(__file__).resolve().parent/'fixtures'


def test_pinned_reference_files_have_license_and_matching_hashes():
    root=ROOT/'baselines/ecoserve/references'
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['revision']=='e7d7f7fe29e20c0218afac305f157dde4513de76'
    assert 'LICENSE' in {row['path'] for row in manifest['files']}
    for row in manifest['files']:
        assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256']


def oracle(table, count=3, now=10000):
    source = ROOT / 'baselines/ecoserve/references/ecoserve/macro_instance.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MacroInstance')
    names = {'_check_constraints', '_switch_instance', '_get_predict_time', '_update_state', 'schedule'}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for method in methods:
        method.returns = None
        for arg in (*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs):
            arg.annotation = None
    ns = dict(time=SimpleNamespace(time=lambda: now / 1000),
              logger=SimpleNamespace(debug=lambda *a: None), BLOCK_SIZE=16,
              RequestState=lambda *a: SimpleNamespace(**dict(zip(
                  ('request_id', 'arrival_time', 'num_iterations', 'ttft', 'predict_time',
                   'predict_length', 'prefill_blocks'), a))))
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), str(source), 'exec'), ns)
    typ = type('PinnedMacroOracle', (), {n: ns[n] for n in names})
    result = typ()
    result.TTFT = 1000; result.TPOT = 100; result.instance_count = count
    result.prefill_instance = 0; result.prefill_data = table; result.writer = None
    result.controls = []
    result._send_control_info = lambda send, i: result.controls.append((i, send))
    result.instance_states = [SimpleNamespace(requests=deque(), waiting_queue=[], free_blocks=10000,
                             prefill_mode=False, schedule_time=now) for _ in range(count)]
    return result


def test_profile_formula_exact_keys_scaling_and_integer_truncation():
    table = {16: 2.5, 32: 8.8, 4096: 403.9, 7168: 650.2}
    reference = oracle(table)
    profile = PrefillProfile(table)
    for n in (1, 15, 16, 17, 31, 32, 128, 4095, 4096, 5000, 7168, 8192):
        assert profile.predict_ms(n) == reference._get_predict_time(n)
    assert profile.predict_ms(1) == 0  # no invented 1 ms floor


def test_profile_missing_required_anchor_fails_loading():
    with pytest.raises(ValueError, match='16.*4096'):
        PrefillProfile({16: 1})


def test_official_snapshot_differential_including_max_and_next_branch():
    rng = random.Random(701)
    table = {16: 2., 128: 24., 4096: 400.}
    for case in range(250):
        reference = oracle(table)
        policy = OfficialMacro(('a', 'b', 'c'), PrefillProfile(table), 1000, 100, now_ms=lambda: 10000.)
        for i in range(3):
            state = reference.instance_states[i]
            state.free_blocks = rng.choice((0, 10, 10000))
            for j in range(rng.randrange(5)):
                rid = f'{case}-{i}-{j}'
                state.requests.append(SimpleNamespace(request_id=rid, arrival_time=9000.,
                    num_iterations=rng.randrange(20), ttft=rng.randrange(500), predict_time=rng.randrange(600),
                    predict_length=-1, prefill_blocks=rng.randrange(200)))
                if rng.random() < .4: state.waiting_queue.append(rid)
        policy.instance_states = deepcopy(reference.instance_states)
        n = rng.choice((16, 128, 512, 4096, 7168))
        request = SimpleNamespace(request_id=f'new-{case}', prompt_len=n)
        assert policy.schedule(request) == reference.schedule(request)
        assert policy.controls == reference.controls
        assert [list(x.waiting_queue) for x in policy.instance_states] == [list(x.waiting_queue) for x in reference.instance_states]
        for actual, expected in zip(policy.instance_states, reference.instance_states):
            assert [vars(r) for r in actual.requests] == [vars(r) for r in expected.requests]
            assert actual.free_blocks == expected.free_blocks  # no output-length or speculative block reservation


def test_native_step_updates_match_official_second_token_progress():
    table = {16: 2., 4096: 400.}
    ref = oracle(table); policy = OfficialMacro(('a','b','c'), PrefillProfile(table),1000,100,now_ms=lambda:10000.)
    policy.instance_states = deepcopy(ref.instance_states)
    request = SimpleNamespace(request_id='r', prompt_len=16)
    policy.schedule(request); ref.schedule(request)
    for event in [dict(instance_id=0,free_blocks=88,prefill_mode=True,schedule_queue=['r'],all_queue=['r'],schedule_time=10001),
                  dict(instance_id=0,free_blocks=88,prefill_mode=False,schedule_queue=['r'],all_queue=['r'],schedule_time=10025),
                  dict(instance_id=0,free_blocks=87,prefill_mode=False,schedule_queue=['r'],all_queue=['r'],schedule_time=10039),
                  dict(instance_id=0,free_blocks=100,prefill_mode=False,schedule_queue=[],all_queue=[],schedule_time=10040)]:
        ref._update_state(SimpleNamespace(**event)); policy.update_state(SimpleNamespace(**event))
        assert [vars(r) for r in policy.instance_states[0].requests] == [vars(r) for r in ref.instance_states[0].requests]


def test_instance_output_hold_has_original_strict_deadline_and_flush():
    buffer = OutputBuffer()
    buffer.control(False, 1000, now_ms=10)
    assert buffer.receive('a', {'token_ids':[1]}, now_ms=11, unfinished=2) == []
    assert buffer.receive('b', {'token_ids':[2]}, now_ms=1010, unfinished=1) == []
    assert buffer.receive('b', {'token_ids':[3]}, now_ms=1011, unfinished=1) == [
        ('a',{'token_ids':[1]}),('b',{'token_ids':[2]}),('b',{'token_ids':[3]})]
    buffer.control(False,1000,now_ms=2000)
    assert buffer.receive('a',{'token_ids':[4]},now_ms=2001,unfinished=0) == [('a',{'token_ids':[4]})]
    buffer.control(False,1000,now_ms=3000)
    assert buffer.receive('a',{'token_ids':[5]},now_ms=3001,unfinished=1)==[]
    buffer.control(True,1000,now_ms=3002)
    assert buffer.pending  # author flushes on output, not merely receipt of control
    assert buffer.receive('a',{'token_ids':[6]},now_ms=3003,unfinished=1)==[('a',{'token_ids':[5]}),('a',{'token_ids':[6]})]


def test_output_flush_trace_matches_pinned_instance_method():
    source=ROOT/'baselines/ecoserve/references/ecoserve/instance.py'
    cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Instance')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_send_outputs')
    method.returns=None
    for arg in method.args.args:arg.annotation=None
    current={'now':0.,'unfinished':1};wire=[]
    ns=dict(time=SimpleNamespace(time=lambda:current['now']/1000),
            pickle=SimpleNamespace(dumps=lambda value:value),
            RPCOutput=lambda rid,text,finished:(rid,text,finished))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source),'exec'),ns)
    ref=SimpleNamespace(send_output=True,TTFT=0,prefill_time=0,output_list=[],
        engine=SimpleNamespace(scheduler=[SimpleNamespace(get_num_unfinished_seq_groups=lambda:current['unfinished'])]),
        output_socket=SimpleNamespace(send_multipart=lambda frames,copy=False:wire.extend(frames[0])))
    buffer=OutputBuffer();actual=[]
    rng=random.Random(702)
    for index in range(300):
        current['now']+=rng.randrange(30)
        if rng.random()<.2:
            send=rng.choice((True,False));ref.send_output=send;ref.TTFT=100;ref.prefill_time=current['now']
            buffer.control(send,100,now_ms=current['now'])
        current['unfinished']=rng.randrange(4)
        rid=str(index%4);text=str(index);finished=current['unfinished']==0
        upstream=SimpleNamespace(request_id=rid,outputs=[SimpleNamespace(text=text)],finished=finished)
        ns['_send_outputs'](ref,[upstream])
        actual.extend((rid,event['text'],event['finished']) for rid,event in buffer.receive(
            rid,dict(text=text,finished=finished),now_ms=current['now'],unfinished=current['unfinished']))
        assert actual==wire
