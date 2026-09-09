import pytest
from ecopadg.serving.stream import CompletionAccumulator


def test_empty_text_tokens_and_final_usage_are_not_lost():
    stream=CompletionAccumulator()
    stream.add(dict(id='r',token_ids=[12],choices=[dict(text='',index=0)]),10)
    stream.add(dict(id='r',token_ids=[13,14],choices=[dict(text='ok',finish_reason='length')],
        usage=dict(prompt_tokens=3,completion_tokens=3,total_tokens=6)),11)
    result=stream.result()
    assert result['token_ids']==[12,13,14]
    assert result['token_received_s']==[10,11,11]
    assert result['choices']==[dict(text='ok',index=0,finish_reason='length')]


def test_truncated_stream_is_a_failure():
    stream=CompletionAccumulator()
    stream.add(dict(token_ids=[1],usage=dict(completion_tokens=2)),10)
    with pytest.raises(RuntimeError,match='incomplete'):
        stream.result()
