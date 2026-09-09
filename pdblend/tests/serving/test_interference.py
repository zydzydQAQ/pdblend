import pytest
from ecopadg.serving.interference import interference_interval,measured_grid


def test_supplemental_training_covers_its_own_context_and_batch():
    def point(n,b,role='mixed'):
        return dict(role=role,tp=1,frequency_mhz=1500,input_tokens=n,context_tokens=n+512,batch=b)
    tables=[dict(points=[point(7168,4),point(128,1),point(128,8,'prefill')]),
            dict(points=[point(6144,6),point(7168,4)])]
    grid=measured_grid(tables,1,128,[128,7168])
    assert grid==[(1500,128,3,7552),(1500,128,5,6528),
                  (1500,7168,3,7552),(1500,7168,5,6528)]


def test_prefill_interference_uses_actual_neighboring_decode_tokens():
    events=[dict(prefill=0,decode=1,request_ids=['old'],started_s=t-.03,finished_s=t)
            for t in (.04,.08,.12,.46)]
    events.append(dict(prefill=1,decode=0,request_ids=['probe'],started_s=.13,finished_s=.41))
    measured=interference_interval(events,['old'],'probe')
    assert measured['baseline_iteration_s']==pytest.approx(.04)
    assert measured['interrupted_token_interval_s']==pytest.approx(.34)
    assert measured['incremental_delay_s']==pytest.approx(.30)
    assert measured['prefill_s']==pytest.approx(.28)
    with pytest.raises(ValueError,match='live decode'):
        interference_interval([e for e in events if e['finished_s']<.45],['old'],'probe')
    events[-1].update(decode=1,request_ids=['old','probe'])
    measured=interference_interval(events,['old'],'probe')
    assert measured['interrupted_token_interval_s']==pytest.approx(.29)
    assert measured['incremental_delay_s']==pytest.approx(.25)
