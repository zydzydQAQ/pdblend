"""Derive workload totals omitted by legacy rows from their exact frozen trace."""


def metadata(p, row):
    trace = p.checked(dict(path=row['trace_path'], sha256=row['trace_sha256']))
    count = len(trace['requests'])
    tokens = sum(x['output_len'] for x in trace['requests'])
    p.need(row['n_requests'] == count, 'baseline declaration count differs from exact trace')
    p.need(row.get('n_expected', count) == count
           and row.get('expected_generated_tokens', tokens) == tokens,
           'baseline declared workload totals differ from exact trace')
    return dict(row, n_expected=count, expected_generated_tokens=tokens)
