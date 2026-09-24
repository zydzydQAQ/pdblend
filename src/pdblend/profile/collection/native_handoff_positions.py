"""Observed logical token positions, never a physical-transfer predictor.

The existing native client receives P's first token before it submits D.
Its first-to-second gap is therefore a different estimand from first-token
latency. These CPU reducers preserve the raw signed contrast for diagnostics;
they do not add it to a CUDA prefill estimate or claim proxy-client SLO scope.
"""
from __future__ import annotations
import math

from .native_runtime_audit import validate_transfer_request
from .native_transfer_diagnostics import decompose
from .native_timing_plan import digest


def observe_endpoint_positions(row, *, specs, training_seed=9701, holdout_seed=9702):
    """Replay a complete native epoch/request and return its real positions.

    Input is an existing transfer_request journal row. The original validator
    enforces exact carry token IDs, native peer epochs and complete streams.
    The output remains observation-only even if all segment values are positive.
    """
    timing=validate_transfer_request(row,specs=specs,training_seed=training_seed,holdout_seed=holdout_seed)
    epoch=row['native_epoch'];p,d=epoch['prefill_instance'],epoch['decode_instance']
    parts=decompose(row['result'],input_tokens=row['input_tokens'],prefill_instance=p,decode_instance=d)
    pre,combined=row['result']['prefill'],row['result']['combined']
    first=pre['first_token_s']-pre['submitted_s']
    gap=combined['decode_first_token_s']-pre['first_token_s']
    second=combined['decode_first_token_s']-pre['submitted_s']
    if not (math.isclose(first+gap,second,rel_tol=0,abs_tol=1e-12)
            and math.isclose(gap,parts['client_bridge_s']+parts['decode_submit_to_next_output_s'],rel_tol=0,abs_tol=1e-12)):
        raise ValueError('native carried-token endpoint partition does not conserve elapsed time')
    return dict(schema='pdblend-native-handoff-endpoint-observation/v1',request_sha256=digest(row),
        request_id=pre['request_id'],input_tokens=row['input_tokens'],output_tokens=combined['completion_tokens'],
        native_peers=dict(prefill=p,decode=d,generation=specs[p]['generation'],
            prefill_gpus=list(specs[p]['gpus']),decode_gpus=list(specs[d]['gpus'])),
        first_output_latency_s=first,first_to_second_gap_s=gap,second_output_latency_s=second,
        client_bridge_s=parts['client_bridge_s'],decode_submit_to_next_output_s=parts['decode_submit_to_next_output_s'],
        signed_second_output_contrast_s=timing['overhead_s'],negative_contrast_preserved=True,
        observed_scope='direct_native_client_P_response_then_D_continuation',
        proxy_client_ttft_observed=False,physical_copy_time_s=None,physical_copy_time_identifiable=False,
        first_output_includes_prefill_and_handoff_work_before_P_response=True,
        first_to_second_includes_HTTP_D_queue_receive_install_and_compute=True,
        predictor_qualified=False,planner_transfer_seconds_compatible=False,formal_eligible=False)


def require_planner_handoff_prediction(observation):
    """Fail explicitly at the current observation/prediction scope boundary.

    A future independently replayed predictor needs its own interface/schema.
    Positive endpoint observations or a favorable contrast are insufficient.
    """
    if observation.get('schema')!='pdblend-native-handoff-endpoint-observation/v1':
        raise ValueError('unknown handoff observation schema')
    raise ValueError('missing_profile: endpoint observations are not a qualified proxy-client latency predictor; '
                     'signed second-output contrast cannot be transfer_seconds or a TTFT increment')
