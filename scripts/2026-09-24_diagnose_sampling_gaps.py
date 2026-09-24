#!/usr/bin/env python3
"""Read-only diagnosis of one already recorded window; never rewrites evidence."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time


def binding(path):
    return dict(path=str(path.resolve()), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def diagnose(csv_path, point_id, *, native_journal=False):
    rows = [r for r in csv.DictReader(csv_path.open()) if r['point_id'] == point_id and r.get('receipt_path')]
    if len(rows) != 1:
        raise ValueError('exactly one recorded point is required')
    root = Path(rows[0]['receipt_path']).parent/'run'
    metering = json.loads((root/'comparison-metering.json').read_text())
    start, end = metering['service_start_s'], metering['service_end_s']
    raw = root/'power.samples.jsonl.gz'
    previous = None; largest = []
    with gzip.open(raw, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('kind') != 'power_metadata':
                continue
            current = row['values']
            if previous and current['read_finished_s'][0] > start and previous['read_finished_s'][0] < end:
                gap = current['read_finished_s'][0] - previous['read_finished_s'][0]
                largest.append(dict(gap_s=gap, gap_start_s=previous['read_finished_s'][0],
                    gap_end_s=current['read_finished_s'][0],
                    at_service_s=previous['read_finished_s'][0]-start,
                    between_rows_s=current['read_started_s'][0]-previous['read_finished_s'][-1],
                    acquisition_wall_s=[b-a for a,b in zip(current['read_started_s'], current['read_finished_s'])],
                    nvml_latency_us=current.get('nvml_latency_us')))
                largest.sort(key=lambda r:r['gap_s'], reverse=True)
                del largest[5:]
            previous = current
    output = dict(point_id=point_id, csv_status=rows[0]['status'],
        source_power=binding(raw), source_metering=binding(root/'comparison-metering.json'),
        power_error=metering['power_error'],
        power_error_affects_window=metering['power_error_affects_window'],
        power_source_verified=metering['power_source_verified'],
        service_power={k:v for k,v in metering['service']['power'].items() if k!='per_gpu'},
        largest_acquisition_gaps=largest,
        historical_evidence_modified=False, complete_energy_recovery_established=False)
    if native_journal:
        from pdblend.results.journal import iter_journal
        journal=root/'events.jsonl.gz'; series={i:[] for i in range(8)}; errors=0
        for row in iter_journal(journal):
            if row.get('event') != 'dynamo_power':
                continue
            if row.get('error'):
                errors += 1; continue
            if row.get('gpu') in series:
                series[row['gpu']].append(row['timestamp'])
        devices={}
        for gpu, stamps in series.items():
            gaps=[b-a for a,b in zip(stamps,stamps[1:]) if b>start and a<end]
            devices[gpu]=dict(samples=sum(start<=t<end for t in stamps),
                max_gap_s=max(gaps, default=None), gaps_over_1s=sum(g>1 for g in gaps),
                bounds_bracketed=bool(stamps and stamps[0]<=start and stamps[-1]>=end),
                native_samples_inside_largest_common_gap=sum(
                    largest[0]['gap_start_s']<t<largest[0]['gap_end_s'] for t in stamps))
        output['native_telemetry']=dict(source=binding(journal), errors=errors, per_gpu=devices,
            same_host_process=True, collector='asyncio GroupTelemetry._sample')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=Path('results/compare.csv'))
    parser.add_argument('--point', default='7b-dynamollm-longbench-x1-seed701')
    parser.add_argument('--native-journal', action='store_true', help='Also scan the native event journal; schedule outside measurement')
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args(); started=time.monotonic()
    report=diagnose(args.csv,args.point,native_journal=args.native_journal)
    report['elapsed_s']=time.monotonic()-started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(report,stream,sort_keys=True,indent=2); stream.write('\n')
    print(json.dumps(dict(out=str(args.out.resolve()), point_id=args.point, elapsed_s=report['elapsed_s'])))


if __name__=='__main__': main()
