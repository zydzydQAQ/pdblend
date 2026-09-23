"""CPU test of vLLM's optional lossless sampled-token-ID response path."""
import inspect
import pytest
vllm=pytest.importorskip('vllm')
protocol=pytest.importorskip('vllm.entrypoints.openai.protocol')
serving=pytest.importorskip('vllm.entrypoints.openai.serving_completion')
def make_request():
    cls=protocol.CompletionRequest; fields=set(inspect.signature(cls).parameters)
    kwargs={'model':'test','prompt':[1,2,3],'max_tokens':3,'logprobs':0,'return_tokens_as_token_ids':True}
    return cls(**{k:v for k,v in kwargs.items() if k in fields})
def test_completion_request_and_logprobs_preserve_special_empty_text_ids():
    request=make_request(); assert request.logprobs==0; assert request.return_tokens_as_token_ids is True
    obj=object.__new__(serving.OpenAIServingCompletion)
    # The helper reads this constructor field; set it explicitly so this
    # focused protocol test does not need to instantiate the full server.
    obj.return_tokens_as_token_ids = True
    ids=[151643,151645,151643]
    class _Tokenizer:
        def decode(self, token_id):
            return ''
    result=obj._create_completion_logprobs(
        ids, [None] * len(ids), 0, _Tokenizer(), return_as_token_id=True)
    assert list(result.tokens)==[f'token_id:{x}' for x in ids]; assert len(result.tokens)==len(ids)
