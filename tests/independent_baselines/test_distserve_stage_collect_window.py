"""Native scope changes must precede live decode; retain the 2s/5s protocol."""
import json
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from pdblend_baselines.distserve import stage_collect as module


@pytest.fixture
def harness(monkeypatch):
    class Clock:
        now=100.
        def time(self):return self.now
        def monotonic(self):return self.now
        def sleep(self,seconds):self.now+=seconds
    clock=Clock()
    monkeypatch.setattr(module,'time',clock)
    state=dict(active=False,measuring=False,calls=[],states={},samples=[],sample_calls=0)
    point=dict(role='decode',frequency_mhz=2100,lengths=[1024,1024],purpose='training',repeat=0)
    spec=SimpleNamespace(base_url='http://native',gpus=(0,1),tp=2)

    class Sampler:
        samples=[];frequency_samples=[];power_metadata=[];error=None
        def start(self):
            self.stopped=False
            state['service_start']=clock.time()
            self.samples=[(clock.time(),[100.,100.])]
            self.frequency_samples=[(clock.time(),[2100,2100])]
        def stop(self):
            if state.get('service_start') is not None and not self.stopped:
                self.samples.append((clock.time(),[100.,100.]))
                self.frequency_samples.append((clock.time(),[2100,2100]))
                self.stopped=True
    sampler=Sampler()
    meter=SimpleNamespace(sampler=lambda *args,**kwargs:sampler)

    def sample():
        began=state['service_start'];end=clock.time()
        # Acquisition starts before settle. Preserve those raw samples while
        # ensuring only service-window steps reach the independent fit.
        stamps=[began-1.,*[began+.25*(i+1) for i in range(state.get('service_steps',8))],end+.1]
        return dict(ranks=[dict(rank=rank,samples=[dict(system='distserve',role=point['role'],
            measurement_scope='runner',tp=2,pp=1,rank=rank,batch=2,request_ids=['a','b'],
            prompt_lengths=[1024,1024],context_lengths=[1024+i,1024+i] if point['role']=='decode' else [1024,1024],
            scheduled_lengths=[1,1] if point['role']=='decode' else [1024,1024],
            gpu_elapsed_ms=999. if i in (0,len(stamps)-1) else 10.+rank,at_s=at)
            for i,at in enumerate(stamps)]) for rank in range(2)])

    def call(url,method,path,payload=None):
        state['calls'].append((path,clock.time()))
        if path.endswith('/capability'):
            return dict(tp=2,state=dict(max_num_seqs=32,total_kv_tokens=100000,max_num_batched_tokens=8192))
        if path.endswith('/clock'):return dict(acknowledged=True)
        if path.endswith('/measurement/start'):
            if state['active']:
                raise HTTPError(url,409,'scope change requires an empty native queue',{},None)
            state['measuring']=True
            return dict(acknowledged=not state.get('reject_ack'),ranks=[dict(rank=i,acknowledged=True) for i in range(2)])
        if path.endswith('/measurement/samples'):
            assert state['measuring']
            state['sample_calls']+=1
            return sample()
        if path.endswith('/measurement/stop'):
            state['measuring']=False
            return dict(acknowledged=True)
        if path.endswith('/cancel'):
            state['states'][payload['request_id']]['finished']=True
            state['active']=False
            return dict(acknowledged=True)
        if path.endswith('/drain'):
            assert not state['active']
            return dict(acknowledged=True,drained=True)
        if path.endswith('/control'):return dict(acknowledged=True)
        raise AssertionError(path)

    def submit(url,lengths,role,pool,prefix):
        state['calls'].append(('submit',clock.time()))
        state['active']=role=='decode'
        rows={prefix+'-'+str(i):dict(tokens=1,finished=role=='prefill') for i in range(len(lengths))}
        state['states'].update(rows)
        if role=='prefill':clock.sleep(.5)
        return rows,[]
    monkeypatch.setattr(module,'call',call)
    monkeypatch.setattr(module,'_submit_batch',submit)
    return spec,meter,point,state,clock


def test_decode_arms_idle_then_settles_and_excludes_warmup_cuda_steps(harness,tmp_path):
    spec,meter,point,state,clock=harness
    def barrier():
        assert state['active'] and state['measuring']
        clock.sleep(.75)  # Cross-engine coordination remains outside service.
    result=module.window(spec,meter,point,tmp_path/'window.json',before_measure=barrier)
    assert result['status']=='measured',result.get('error')
    paths=[path for path,_ in state['calls']]
    assert paths.index('/baseline/measurement/start')<paths.index('submit')
    assert result['decode_active_s']==result['settle_started_s']
    assert result['settle_finished_s']-result['settle_started_s']==2.
    assert result['start_s']-result['settle_finished_s']==.75
    assert result['end_s']-result['start_s']==5.
    assert result['decode_steps']==8 and state['sample_calls']==1
    assert len(result['sample']['ranks'][0]['samples'])==10
    rows=module.rows_from_window(result)
    assert len(rows)==8 and {r['latency_ms'] for r in rows}=={11.}
    assert not state['active'] and not state['measuring'] and not result['cleanup_errors']
    assert json.loads((tmp_path/'window.json').read_text())['status']=='measured'


def test_prefill_keeps_full_five_second_service_and_idle_arming(harness,tmp_path):
    spec,meter,point,state,_=harness
    point['role']='prefill'
    result=module.window(spec,meter,point,tmp_path/'prefill.json')
    assert result['status']=='measured',result.get('error')
    assert result['settle_finished_s']-result['settle_started_s']==2.
    assert result['end_s']-result['start_s']==5.
    assert len([p for p,_ in state['calls'] if p=='submit'])==10
    assert len(module.rows_from_window(result))==8


def test_settle_cuda_steps_cannot_satisfy_minimum_eight_service_steps(harness,tmp_path):
    spec,meter,point,state,_=harness
    state['service_steps']=7
    result=module.window(spec,meter,point,tmp_path/'short.json')
    assert result['status']=='failed' and 'insufficient actual CUDA' in result['error']
    assert module.rows_from_window(result)==[]
    assert not state['active'] and not state['measuring'] and not result['cleanup_errors']


def test_missing_start_ack_never_submits_requests_and_still_closes_measurement(harness,tmp_path):
    spec,meter,point,state,_=harness
    state['reject_ack']=True
    result=module.window(spec,meter,point,tmp_path/'no-ack.json')
    assert result['status']=='failed' and 'not acknowledged' in result['error']
    assert not any(path=='submit' for path,_ in state['calls'])
    assert not state['active'] and not state['measuring'] and not result['cleanup_errors']
