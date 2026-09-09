"""Separate verified observations from retired queues and conditional high rates."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def logical(cell):
    x=cell.get('original_point',cell.get('source_row',{}))
    return (x.get('model'),cell['dataset'],x.get('rate_rps'),x.get('seed'),
        cell['trace']['sha256'],x.get('content_pairing_sha256'),x.get('slo_ttft_s'),
        x.get('slo_tpot_s'),x.get('n_expected',x.get('n_requests')),
        x.get('expected_generated_tokens'),cell.get('repeat'),cell.get('arm'))

def audit(snapshot,out):
    if out.exists():raise FileExistsError(out)
    result=read(snapshot/'results.json');points={x['cell_id']:x for x in result['points']}
    nodes=[];origins=defaultdict(list);by_logical=defaultdict(list)
    for attempt in result['attempts']:
        root=Path(attempt['path']);status=read(root/'status.json');order=read(root/'declaration-order.json')
        state={k:status.get(k) for k in ('phase','pid','model','stage','completed','attempted','failed',
            'skipped','first_complete_breach','complete','error','current_cell','updated_s','finished_s')}
        nodes.append(dict(path=str(root),read_s=time.time(),status=state))
        for cell in order:
            entry=dict(root=root,status=state,cell=cell)
            origins[cell['cell_id']].append(entry);by_logical[logical(cell)].append(entry)
    rows=[]
    for cid,point in points.items():
        entries=origins[cid];cell=entries[0]['cell'];state='';reason=''
        if point['measurement_valid']:
            state='verified_complete_work' if point['work_complete'] else 'verified_incomplete_work_negative'
            reason='Raw workload and all-eight-GPU metrics independently checked.'
        elif point['status'] not in ('unmeasured','awaiting_mirror'):
            state='observed_invalid_requires_diagnosis';reason=point.get('error')
        elif any(cid in (e['status'].get('completed') or []) for e in entries):
            state='awaiting_raw_verification';reason='Producer reports terminal observation after independent snapshot or before full mirror.'
        else:
            lower=[e['status'].get('first_complete_breach',{}).get(cell['dataset'])
                for e in entries if e['status'].get('first_complete_breach')]
            if any(point['rate_rps']>v for v in lower if v is not None):
                state='not_required_above_first_loss';reason='Conditional higher rate canceled by the user stop rule; remains unmeasured.'
            elif all(e['status']['phase'] in ('stopped_at_boundary','complete') for e in entries):
                replacement=[e for e in by_logical[logical(cell)] if e['cell']['cell_id']!=cid
                    and (e['status']['phase']=='running' or e['cell']['cell_id'] in points
                         and points[e['cell']['cell_id']]['measurement_valid'])]
                if replacement:
                    state='superseded_declaration';reason='Same exact workload/arm/repeat was re-declared in the completion queue; no measurement reused or counted twice.'
                else:
                    state='retired_queue_awaiting_final_selection';reason='Old queue was stopped; node final configuration/continuation determines remaining required work.'
            else:
                state='required_queue_pending';reason='Selected queue has not produced a verified observation for this declaration.'
        rows.append(dict(model=point['model'],dataset=point['dataset'],rate_rps=point['rate_rps'],repeat=point['repeat'],
            cell_id=cid,original_cell_id=point['original_cell_id'],observation_status=point['status'],scope_status=state,
            reason=reason,source_declarations=[str(e['root']/'declaration-order.json') for e in entries]))
    out.mkdir(parents=True)
    with (out/'scope.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        w.writerows(dict(r,source_declarations=json.dumps(r['source_declarations'])) for r in rows)
    data=dict(created_s=time.time(),independent_snapshot=dict(path=str(snapshot),manifest_sha256=sha(snapshot/'manifest.json')),
        counts=dict(Counter(x['scope_status'] for x in rows)),rows=rows,status_snapshots=nodes,
        no_unmeasured_point_is_claimed_complete=True,not_a_final_configuration_selection=True)
    (out/'scope.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__))),indent=2)+'\n')
    print(json.dumps(data['counts']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.snapshot.resolve(),a.out.resolve())
