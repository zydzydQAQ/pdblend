import pytest

from pdblend.engine.client import Completion
from pdblend.engine.handoff_timing import measure_handoff
from pdblend_runtime.public_pd_probe import check_completion


def test_handoff_compares_second_output_not_visible_first_token():
    mixed=Completion('m','d',10,first_token_s=11,finished_s=11.3,
        token_times_s=[11,11.1,11.2,11.3],completion_tokens=4)
    pre=Completion('pd','p',20,first_token_s=21)
    combined=Completion('pd','d',20,first_token_s=21,decode_first_token_s=21.4,
        pd_protocol='carry_first_token')
    result=measure_handoff(mixed,pre,combined)
    assert result['overhead_s']==pytest.approx(.3)
    assert result['physical_copy_time'] is False
    # Both logical TTFT values equal one second, yet the handoff costs time.
    assert combined.ttft_s == mixed.ttft_s


def test_handoff_rejects_missing_or_coalesced_token_arrivals():
    mixed=Completion('m','d',10,token_times_s=[11,11.2],completion_tokens=4)
    pre=Completion('pd','p',20)
    combined=Completion('pd','d',20,decode_first_token_s=21.4,pd_protocol='carry_first_token')
    with pytest.raises(ValueError,match='aligned'):
        measure_handoff(mixed,pre,combined)
    combined.pd_protocol=None
    with pytest.raises(ValueError,match='protocol'):
        measure_handoff(mixed,pre,combined)


def test_public_golden_requires_exact_special_token_ids_and_usage():
    value=Completion('r','p',0,first_token_s=1,finished_s=2,prompt_tokens=512,
                     completion_tokens=2,token_ids=[151643,151645],text='',stream_done=True,usage_received=True)
    assert all(check_completion(value,[151643,151645],prompt_tokens=512,output_tokens=2).values())
    value.token_ids=[151645,151643]
    assert not all(check_completion(value,[151643,151645],prompt_tokens=512,output_tokens=2).values())
    value.token_ids=None
    assert not all(check_completion(value,None,prompt_tokens=512,output_tokens=2).values())
