"""Measured latency injection; replaces upstream OPT/A100 fit, never extrapolates PP."""
from contextvars import ContextVar

LATENCY=ContextVar('distserve_measured_latency')
PIPE_STAGE=ContextVar('distserve_measured_pipeline_stage',default=None)


def measured(role,tp,pp,batch,inputs,contexts):
    provider=LATENCY.get()
    if hasattr(provider,'stage_latency'):
        stage=PIPE_STAGE.get()
        if type(stage) is not int or not 0<=stage<pp:raise ValueError('actual simulator pipeline stage identity required')
        return provider.stage_latency(role,tp,pp,stage,batch,inputs,contexts)
    return provider(role,tp,pp,batch,inputs,contexts)


def get_prefill_time(num_tokens, *, bs, decode_bs, pp, model_type, TP,
                     prefill_len_list, engine_type, **kwargs):
    if decode_bs:raise ValueError('DistServe disaggregated simulator received a mixed batch')
    return measured('prefill',TP,pp,bs,prefill_len_list,prefill_len_list)


def get_decode_time(batch_size, *, pp, model_type, TP, token_generated_list,
                    engine_type, **kwargs):
    # The author estimator uses contexts, not each original input length, for D.
    return measured('decode',TP,pp,batch_size,(),token_generated_list)
