"""Preserve producer totals while identifying missing usage on partial requests."""
import csv
from pathlib import Path
import raw_metrics_v2 as previous


def token_accounting(rows, reported_total):
    verified = 0
    unknown = []
    for row in rows:
        count = int(row['generated_tokens'])
        previous.require(count >= 0, 'negative recorded generated tokens')
        terminal = (row.get('token_count_source') == 'server_usage'
                    and previous.truth(row.get('token_ids_verified')))
        rejected_before_output = (previous.admission_rejected(row) and count == 0
            and not previous.truth(row.get('success'))
            and int(row.get('n_text_chunks') or 0) == 0
            and row.get('first_token_s') in (None, '')
            and row.get('last_token_s') in (None, ''))
        if terminal or rejected_before_output:
            verified += count
        else:
            unknown.append(row)
    previous.require(sum(int(r['generated_tokens']) for r in rows) == reported_total,
                     'producer recorded generated-token total differs from raw rows')
    lower_bound = bool(unknown) and all(int(r['generated_tokens']) == 0 for r in unknown)
    partial = [r for r in unknown if int(r.get('n_text_chunks') or 0) > 0]
    semantics = ('exact verified output token count' if not unknown else
        'recorded verified tokens are a lower bound; missing terminal usage is not zero generation'
        if lower_bound else 'producer-recorded output count includes unverified usage; exact total unknown')
    return dict(generated_token_count_complete=not unknown,
        generated_tokens_semantics=semantics,
        recorded_generated_tokens_are_lower_bound=lower_bound,
        verified_generated_tokens=verified,
        unverified_output_requests=len(unknown),
        unverified_partial_output_requests=len(partial),
        partial_output_chunks_without_verified_usage=sum(int(r.get('n_text_chunks') or 0) for r in partial))


def audit_additional_metrics(summary, directory):
    result = previous.audit_additional_metrics(summary, directory)
    with (Path(directory) / 'bench.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    result['normalized_metrics'].update(token_accounting(rows, summary['generated_tokens']))
    result['token_accounting_note'] = 'Text chunks are retained as chunks and never substituted for token counts.'
    return result
