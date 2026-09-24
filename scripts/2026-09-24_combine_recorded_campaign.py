#!/usr/bin/env python3
"""Combine prepared unexecuted windows without revising recorded baseline points."""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path

from pdblend.bench.comparison_campaign import binding, group_points, load_bound
from pdblend.bench.resident_session import write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--eco', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    baseline_ref, eco_ref = binding(args.baseline), binding(args.eco)
    baseline, eco = load_bound(baseline_ref), load_bound(eco_ref)
    recorded = list(csv.DictReader(args.csv.open()))
    seen = {row['point_id'] for row in recorded if row.get('measurement_usable') == 'True'}
    selection = load_bound(eco['unattempted_selection'])
    replacements = {p['name']:p for p in eco['points'] if p['name'] in selection['points']}
    points = [deepcopy(replacements.get(p['name'], p)) for p in baseline['points']]
    if len(points) != 180 or len({p['name'] for p in points}) != 180:
        raise ValueError('incomplete or duplicated campaign inventory')
    jobs, planned = [], set()
    for campaign_path in (args.baseline, args.eco):
        jobs_ref = binding(campaign_path.parent/'jobs.json')
        for job in load_bound(jobs_ref):
            argv = job['payload']['argv']
            group = load_bound(binding(argv[argv.index('--group')+1]))
            names = {p['name'] for p in group['points']}
            if names & (seen | planned):
                raise ValueError('a prepared job repeats recorded or already queued work')
            if any(p != next(item for item in points if item['name'] == p['name']) for p in group['points']):
                raise ValueError('job window differs from the combined campaign')
            planned |= names
            item = deepcopy(job)
            item['payload'].update(prepared_jobs=jobs_ref,
                scheduling_reason='Run each unmeasured window once; retain all recorded baseline outcomes regardless of qualification.')
            jobs.append(item)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    ancestry = set(baseline.get('execution_campaigns', [])) | set(eco.get('execution_campaigns', []))
    ancestry.update((str(args.baseline.resolve()), str(args.eco.resolve())))
    campaign = dict(baseline, campaign_id=out.name, parent_campaign=baseline_ref,
        combined_campaigns=[baseline_ref, eco_ref], execution_campaigns=sorted(ancestry),
        points=points, groups=group_points(points), result_analysis_policy='all_recorded_windows/v1',
        summary=dict(points=180, newly_scheduled_points=len(planned), newly_scheduled_jobs=len(jobs),
                     existing_recorded_windows=sum(r.get('measurement_usable') == 'True' for r in recorded),
                     existing_recorded_points=len(seen), recorded_points_retried=0))
    write_new(out/'campaign.json', campaign)
    write_new(out/'jobs.json', jobs)
    write_new(out/'execution-inputs.json', load_bound(baseline['execution_inputs']))
    write_new(out/'recorded-exclusion.json', dict(analysis_policy='all_recorded_windows/v1',
        recorded_points=sorted(seen), new_points=sorted(planned),
        recorded_receipts=[dict(point=r['point_id'], path=r['receipt_path'], sha256=r['receipt_sha256'])
            for r in recorded if r.get('measurement_usable') == 'True']))
    print(json.dumps(campaign['summary']))


if __name__ == '__main__':
    main()
