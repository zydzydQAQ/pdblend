"""Read-only original three-model Dynamo arrival-fidelity and controller-stall screen."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import final_selected_collect_v1 as collector


def main():
    protocol=collector.load_audit().p
    selected=[row for row in protocol.original_points() if row['system']=='dynamollm']
    assert len(selected)==90
    results=[]
    for old in selected:
        cp=protocol.read(old['checkpoint_path'])
        operation=Path(old['receipt_path']).parent
        cell=operation.parents[1]/'cells'/old['cell_id']
        files=[cell/name for name in ('bench.csv','summary.json','control.jsonl','power.csv')]
        for path in files:assert protocol.sha(path)==cp['artifacts'][str(path)]
        summary=protocol.read(cell/'summary.json')
        requests=list(csv.DictReader((cell/'bench.csv').open()))
        assert len(requests)==old['n_expected']
        dispatched=[row for row in requests if row['actual_dispatch_s']]
        lateness=[float(row['actual_dispatch_s'])-float(row['planned_arrival_s']) for row in dispatched]
        recorded=[float(row['dispatch_delay_s']) for row in dispatched]
        assert all(abs(a-b)<1e-6 for a,b in zip(lateness,recorded))
        control=[json.loads(line) for line in (cell/'control.jsonl').open() if line.strip()]
        times=sorted({float(row['at_s']) for row in control if isinstance(row.get('at_s'),(int,float))})
        gaps=sorted([(end-start,start,end) for start,end in zip(times,times[1:])],reverse=True)
        with_active=[]
        for gap,start,end in gaps:
            if gap<=1:break
            planned=[row for row in requests if start<float(row['planned_arrival_s'])<end]
            unfinished=[row for row in requests if float(row['planned_arrival_s'])<start
                        and float(row['finish_s'])>end]
            if planned or unfinished:
                with_active.append(dict(duration_s=gap,start_s=start,end_s=end,
                    planned_requests_inside=len(planned),requests_unfinished_through_gap=len(unfinished)))
        power=[float(row['t_s']) for row in csv.DictReader((cell/'power.csv').open())]
        limit=summary['dispatch_lateness_limit_s']
        epoch=min(float(row['planned_arrival_s']) for row in requests)
        latest_dispatch=max((float(row['actual_dispatch_s']) for row in dispatched),default=None)
        result=dict(cell_id=old['cell_id'],model=old['model'],dataset=old['dataset'],rate_rps=old['rate_rps'],
            checkpoint=protocol.ref(old['checkpoint_path']),receipt=protocol.ref(old['receipt_path']),
            source=old['executed_source'],raw_files={str(path):protocol.sha(path) for path in files},
            n_expected=len(requests),work_complete=old['work_complete'],failed_requests=old['failed_requests'],
            request_timeouts=old['request_timeouts'],slo_attainment=old['slo_attainment'],energy_j=old['energy_j'],
            declared_lateness_limit_s=limit,actual_dispatch_lateness_max_s=max(lateness,default=None),
            actual_dispatch_lateness_p99_s=sorted(lateness)[min(len(lateness)-1,int(.99*len(lateness)))] if lateness else None,
            dispatch_late_over_declared_limit=sum(v>limit for v in lateness) if limit is not None else None,
            dispatch_late_over_1s=sum(v>1 for v in lateness),dispatch_late_over_10s=sum(v>10 for v in lateness),
            missing_dispatch=len(requests)-len(dispatched),
            last_actual_dispatch_offset_s=latest_dispatch-epoch if latest_dispatch is not None else None,
            arrival_fidelity_valid_reported=summary['arrival_fidelity_valid'],
            open_loop_verified_reported=summary['open_loop_verified'],
            within_lateness_limit_reported=summary['dispatch_lateness_within_declared_limit'],
            planning=summary['admission_planning'],
            controller_max_event_gap_s=gaps[0][0] if gaps else None,
            largest_active_controller_gaps=with_active[:5],
            power_max_gap_s=max(b-a for a,b in zip(power,power[1:])),
            power_covers_measurement=power[0]<=summary['measurement_start_s']<summary['measurement_end_s']<=power[-1],
            scientific_measurement_not_relabelled=True)
        results.append(result)
    destination=ROOT/'C/all-model-original-Dynamo90-arrival-audit-v2.json'
    assert not destination.exists()
    destination.write_text(json.dumps(dict(schema='all-model-original-Dynamo90-arrival-readonly-screen-v2',created_s=time.time(),
        original_snapshot='snapshot-006',points=results,source=dict(path=str(Path(__file__)),sha256=protocol.sha(__file__)),
        original450_unchanged=True,gpu_actions=False,energy_not_reintegrated_in_this_timing_screen=True),indent=2)+'\n')
    print(json.dumps(dict(report=protocol.ref(destination),points=len(results),
        large_dispatch_delay=[{k:row[k] for k in ('cell_id','actual_dispatch_lateness_max_s','last_actual_dispatch_offset_s','work_complete','controller_max_event_gap_s')} for row in results if row['dispatch_late_over_1s']],
        max_dispatch_lateness=max(row['actual_dispatch_lateness_max_s'] for row in results))))


if __name__=='__main__':main()
