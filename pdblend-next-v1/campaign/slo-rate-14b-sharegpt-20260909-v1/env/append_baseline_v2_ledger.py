"""Append independently qualified baseline setup windows from small projections."""
from pathlib import Path
import json, sys, shutil
from build_setup_ledger import ROOT, ref, window

node=sys.argv[1]
assert node in ('A','C')
attempt='baseline-qualification-002' if node=='A' else 'baseline-002'
b=ROOT/node/attempt
projection=ref(b/'setup-receipt-projection.json')
proof=json.loads(Path(projection['path']).read_text())
assert proof['parent']['complete'] and not proof['parent'].get('error')
assert proof['ready_verified_systems']==['distserve','dynamollm','ecoserve','mixed']
p=ROOT/node/'setup-energy-ledger.json'
ledger=json.loads(p.read_text())
new=[]
for op,record in proof['operations'].items():
    state=record['state']
    assert state['measurement_valid'] and state['power_evidence']['power_source_verified']
    assert not state.get('sampling_error') and not state.get('cleanup_errors') and not state.get('errors')
    assert state.get('passed',state.get('complete')) and state.get('complete')
    assert all(f'gpu{i}_w' in record['power_header'].split(',') for i in range(8))
    if op=='deployment':
        key='all8_operation_energy_j';start,end=state['operation_start_s'],state['operation_end_s']
    else:
        key='full_operation_energy_j';start,end=state['measurement_start_s'],state['measurement_end_s']
        assert state['clock_restore_complete']
    new.append(dict(operation_id=node+'-'+attempt+'-'+op,operation=op,category='baseline_setup',
        operation_status='passed',energy_status='measured',energy_j=state[key],window=window(start,end),
        gpu_indices=list(range(8)),all8_gpu_coverage=True,measurement_valid=True,receipt=record['receipt'],
        status_projection=projection,power_source=state['power_evidence'],
        included_in_measured_setup_subtotal=True,shared_by_systems=['mixed','distserve','dynamollm','ecoserve'],
        qualification_passed=True,raw_replay='Original on-host independent validators passed for all four bindings; this ledger checks terminal compact evidence without repeating large raw replay.',
        accounting_note='Whole-node GPU window counted once; four policy bindings share this physical qualification. Retained-weight generation is inside frequency window, with no extra energy added.'))
expected=2 if node=='A' else 3
assert len(new)==expected
combined=[r for r in ledger['entries']+new if r['energy_status']=='measured']
combined.sort(key=lambda r:r['window']['start_s'])
assert all(a['window']['end_s']<=z['window']['start_s'] for a,z in zip(combined,combined[1:]))
assert len({r['receipt']['sha256'] for r in combined})==len(combined)
parent=proof['parent']
children=[r['operation_id'] for r in new]
new.append(dict(operation_id=node+'-'+attempt+'-orchestration',operation='baseline_orchestration',category='baseline_setup',
    operation_status='passed',energy_status='partial_child_operations_only',energy_j=None,
    window=window(parent['started_s'],parent['finished_s']),gpu_indices=None,all8_gpu_coverage=None,receipt=None,
    status=proof['parent_ref'],status_projection=projection,child_operations=children,
    reused_operations=['A-baseline-001-deployment'] if node=='A' else [],
    included_in_measured_setup_subtotal=False,
    accounting_note='No independent parent meter. Child windows counted once; A reuses the already-counted deployment and never adds it again.'))
cursor=parent['started_s'];gaps=[]
for r in sorted([r for r in new if r['energy_status']=='measured'],key=lambda r:r['window']['start_s']):
    w=r['window']
    if cursor<w['start_s']:gaps.append(window(cursor,w['start_s']))
    cursor=w['end_s']
if cursor<parent['finished_s']:gaps.append(window(cursor,parent['finished_s']))
new.append(dict(operation_id=node+'-'+attempt+'-unmetered-intervals',operation='baseline_setup_unmetered_intervals',
    category='baseline_setup',operation_status='completed',energy_status='unknown',energy_j=None,windows=gaps,
    gpu_indices=None,all8_gpu_coverage=None,receipt=None,included_in_measured_setup_subtotal=False,
    accounting_note='CPU registration, derive/replay, import, bookkeeping and gaps lack a dedicated energy meter; unknown, not zero. Formal service is accounted separately.'))
assert not ({r['operation_id'] for r in new}&{r['operation_id'] for r in ledger['entries']})
backup=p.with_name('setup-energy-ledger.before-'+attempt+'.json')
assert not backup.exists();shutil.copyfile(p,backup)
ledger['entries'].extend(new)
ledger['measured_operation_count']=len(combined)
ledger['measured_disjoint_setup_subtotal_j']=sum(r['energy_j'] for r in combined)
ledger['scope']='Terminal PDB and baseline preparation/qualification operations, including preserved failures; formal serving energy excluded'
ledger['complete_accounting']=False;ledger['complete_setup_energy_j']=None;ledger['whole_campaign_energy_j']=None
ledger['baseline_followup'].update(state='qualified_v2',qualified_attempt=attempt,
    qualified_attempt_status=proof['parent_ref'],qualified_attempt_measured_subtotal_j=sum(r['energy_j'] for r in new if r['energy_status']=='measured'),
    setup_receipt_projection=projection,ready=proof['ready_ref'],
    deployment_reused_and_previously_counted=node=='A')
ledger.setdefault('prior_ledger_snapshots',[]).append(ref(backup))
p.write_text(json.dumps(ledger,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(dict(node=node,ledger=ref(p),new_measured_operations=expected,
    new_known_window_energy_j=ledger['baseline_followup']['qualified_attempt_measured_subtotal_j'],
    all_known_setup_window_energy_j=ledger['measured_disjoint_setup_subtotal_j'],complete_setup_energy_j=None)))
