"""Pure incremental long-context sampling plan; never executes or fits a model."""
from __future__ import annotations

import math

FREQUENCIES = (900, 1200, 1500, 1800, 2100, 2520)


def long_context_sampling_plan(raw: dict) -> dict:
    if raw.get('system') != 'pdblend' or raw.get('pp') != 1 or raw.get('tp') not in (1, 2, 4):
        raise ValueError('requires an independent PDBlend PP1 topology profile')
    capacity = int(raw['kv_capacity_tokens'])
    if capacity <= 0 or tuple(sorted(raw['freqs'])) != FREQUENCIES:
        raise ValueError('requires measured capacity and all six frequency tiers')

    def max_batch(context):
        # Reserve the full 1024-token completion, rather than only the initial
        # context+64. Long prompts otherwise pass admission and later OOM.
        return min(256, math.floor(.9 * capacity / (context + min(1024, 8192 - context))))

    training = []
    for frequency in FREQUENCIES:
        for context in (5120, 7168):
            maximum = max_batch(context)
            for batch in sorted({b for b in (1, 4, 8, maximum) if 1 <= b <= maximum}):
                training.append(dict(freq_mhz=frequency, batch=batch, context_tokens=context,
                                     max_tokens=1024, repeats=3, settle_s=2, measure_s=5,
                                     purpose='training_extension'))
    if not training:
        raise ValueError('no memory-feasible long-context training shape')
    holdout = []
    for frequency in FREQUENCIES:
        maximum_mid = max_batch(6144)
        # Two unseen-context shapes, including an interior batch, plus fresh
        # B1 and maximum-batch windows at the long endpoint. Never use these
        # observations for fitting or choosing among candidate models.
        middle = min(maximum_mid, max(3, math.floor(math.sqrt(2 * maximum_mid))))
        shapes = [(2, 6144), (middle, 6144), (1, 7168), (max_batch(7168), 7168)]
        for batch, context in dict.fromkeys(shapes):
            if 1 <= batch <= max_batch(context):
                holdout.append(dict(freq_mhz=frequency, batch=batch, context_tokens=context,
                                    max_tokens=1024, repeats=3, settle_s=2, measure_s=5,
                                    purpose='independent_holdout_after_candidate_freeze'))
    return dict(schema=1, system='pdblend', model_id=raw['model_id'], tp=raw['tp'], pp=1,
                kv_capacity_tokens=capacity, training=training, holdout=holdout,
                minimum_window_seconds=21 * (len(training) + len(holdout)),
                window_lower_bound_excludes='prefill, startup, cleanup, interference checks, and failed windows',
                runtime_capacity_recheck=True, reuse_existing_complete_samples=True,
                fit_existing_holdout=False, formal_eligible=False,
                domain_note='Only actual observed contexts are covered. A 7168-token prompt with 512 output '
                            'tokens needs decode coverage through 7679; a 7168 anchor alone does not prove it.',
                endpoint_completion_rule='If measured windows do not reach the campaign maximum context, '
                                         'append only the missing endpoint training/holdout windows; never '
                                         'extend the domain based on requested max_tokens alone.')
