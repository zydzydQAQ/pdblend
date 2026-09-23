"""Compare the same logical output position after a carried prefill token."""
from __future__ import annotations

import math

PROTOCOL = 'carry_first_token_second_output_v1'


def measure_handoff(mixed, prefill, combined):
    if combined.pd_protocol != 'carry_first_token':
        raise ValueError('handoff timing requires the carry-first-token protocol')
    if (mixed.completion_tokens < 2 or len(mixed.token_times_s) != mixed.completion_tokens or
            combined.decode_first_token_s is None or prefill.submitted_s is None):
        raise ValueError('handoff timing requires aligned ordinary token arrivals and D continuation')
    mixed_second = mixed.token_times_s[1] - mixed.submitted_s
    pd_second = combined.decode_first_token_s - prefill.submitted_s
    if not all(math.isfinite(t) and t >= 0 for t in (mixed_second, pd_second)):
        raise ValueError('invalid second-output latency')
    return dict(protocol=PROTOCOL, mixed_second_output_s=mixed_second,
                pd_second_output_s=pd_second, overhead_s=pd_second-mixed_second,
                includes_http_and_scheduling=True, physical_copy_time=False)
