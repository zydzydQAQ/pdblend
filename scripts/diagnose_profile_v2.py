#!/usr/bin/env python3
"""Replay failed profile-v2 points and classify repeat noise versus model residual."""
import json
import statistics
import sys
from pathlib import Path


def main():
    raw_path, audit_path, out_path = map(Path, sys.argv[1:4])
    raw = json.loads(raw_path.read_text())
    audit = json.loads(audit_path.read_text())
    rows = []
    failed = {tuple((p.get('freq_mhz'), p.get('context_tokens'), p.get('batch')))
              for f in audit.get('failures', []) if f.get('metric', '').startswith('decode_time@')
              for p in [f.get('point', {})]}
    for row in raw.get('decode', []):
        key = (row.get('freq_mhz'), row.get('context_tokens'), row.get('batch'))
        if key not in failed:
            continue
        values = row.get('step_repeats', [])
        cv = statistics.stdev(values) / statistics.mean(values) if len(values) >= 2 else None
        rows.append(dict(point=dict(freq_mhz=key[0], context_tokens=key[1], batch=key[2]),
                         repeats=values, repeat_cv=cv,
                         windows=[x.get('steady_window_s') for x in row.get('repeats', [])],
                         steps=[x.get('steps') for x in row.get('repeats', [])],
                         diagnosis='stable_measurement_model_fit_bias' if cv is not None and cv <= .01
                         else 'measurement_variation'))
    result = dict(failed_points=rows, conclusion=(
        'failed points have stable repeated measurements; revise/validate decode-time basis before changing gate'
        if rows and all(x['diagnosis'] == 'stable_measurement_model_fit_bias' for x in rows)
        else 'additional measurement investigation required'))
    out_path.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
