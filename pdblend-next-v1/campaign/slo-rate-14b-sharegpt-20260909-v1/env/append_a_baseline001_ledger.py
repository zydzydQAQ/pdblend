"""Record the terminal first baseline attempt; reporting only, no hardware."""
from pathlib import Path
import json, shutil
from build_setup_ledger import ROOT, ref, window

p = ROOT / 'A/setup-energy-ledger.json'
d = json.loads(p.read_text())
b = ROOT / 'A/baseline-001'
owner = json.loads((b/'status.json').read_text())
diagnostic = json.loads((b/'failure-inspection.json').read_text())
new = []
for op,rel,key,keys,header in [
    ('baseline-001-deployment','deployment/deployment-receipt.json','all8_operation_energy_j',['operation_start_s','operation_end_s'],'deployment/deployment-power/power.csv'),
    ('baseline-001-native-failed','qualification/native/status.json','full_operation_energy_j',['measurement_start_s','measurement_end_s'],'qualification/native/power/power.csv'),
]:
    q=b/rel; s=json.loads(q.read_text())
    assert s['measurement_valid'] and s['power_evidence']['power_source_verified'] and not s.get('sampling_error')
    assert all(f'gpu{i}_w' in diagnostic['power_headers'][str(b/header)].split(',') for i in range(8))
    passed=s.get('passed',s['complete'])
    if key=='all8_operation_energy_j':
        assert ref(q)==owner['deployment_receipt'] and s['complete'] and not s['errors']
    else:
        assert s['complete'] and not s['passed'] and s['native_cleanup_complete'] and s['clock_restore_complete'] and not s['cleanup_errors']
    new.append(dict(operation_id='A-'+op,operation=op,category='baseline_setup',
        operation_status='passed' if passed else 'failed',energy_status='measured',energy_j=s[key],
        window=window(s[keys[0]],s[keys[1]]),gpu_indices=list(range(8)),all8_gpu_coverage=True,
        measurement_valid=True,receipt=ref(q),power_source=s['power_evidence'],
        coverage_evidence=ref(b/'failure-inspection.json'),error=s.get('error'),
        included_in_measured_setup_subtotal=True,shared_by_systems=['mixed','distserve','dynamollm','ecoserve'],
        qualification_passed=passed if 'native' in op else None,
        raw_replay='Not replayed for this failed attempt; producer receipt SHA, power-source metadata and all-eight power header checked. Raw remains on A.',
        accounting_note='One physical window counted once. Failed gate energy retained; no performance eligibility or fourfold policy allocation inferred.'))
assert new[0]['window']['end_s'] <= new[1]['window']['start_s']
assert all(a['window']['end_s']<=new[0]['window']['start_s'] for a in d['entries'] if a['energy_status']=='measured')
new.append(dict(operation_id='A-baseline-001-orchestration',operation='baseline_orchestration',category='baseline_setup',
    operation_status='failed',energy_status='partial_child_operations_only',energy_j=None,
    window=window(owner['started_s'],owner['finished_s']),gpu_indices=None,all8_gpu_coverage=None,receipt=None,
    status=ref(b/'status.json'),qualification_status=ref(b/'qualification/status.json'),
    child_operations=[x['operation_id'] for x in new],included_in_measured_setup_subtotal=False,
    accounting_note='No independent parent meter. Deployment and failed native child windows counted once; no parent energy added.'))
cursor=owner['started_s']; gaps=[]
for row in new:
    if row['energy_status']!='measured': continue
    w=row['window']
    if cursor<w['start_s']: gaps.append(window(cursor,w['start_s']))
    cursor=w['end_s']
if cursor<owner['finished_s']: gaps.append(window(cursor,owner['finished_s']))
new.append(dict(operation_id='A-baseline-001-unmetered-intervals',operation='baseline_setup_unmetered_intervals',
    category='baseline_setup',operation_status='completed',energy_status='unknown',energy_j=None,windows=gaps,
    gpu_indices=None,all8_gpu_coverage=None,receipt=None,included_in_measured_setup_subtotal=False,
    accounting_note='Unmetered preflight, bind derivation, imports and bookkeeping. Subsequent diagnosis/wait energy remains unknown; no idle interpolation.'))
new.append(dict(operation_id='A-baseline-001-frequency-not-started',operation='baseline_frequency_qualification',
    category='baseline_setup',operation_status='not_started',energy_status='not_measured',energy_j=None,window=None,
    gpu_indices=None,all8_gpu_coverage=None,receipt=None,included_in_measured_setup_subtotal=False,
    accounting_note='Native gate failed before mechanism checks. Frequency never started; no energy value or qualification inferred.'))
assert not ({x['operation_id'] for x in d['entries']} & {x['operation_id'] for x in new})
old=ROOT/'A/setup-energy-ledger.pdb-only.json'
assert not old.exists(); shutil.copyfile(p,old)
d['entries'].extend(new); d['measured_operation_count']=6; d['pdb_only_ledger']=ref(old)
d['scope']='PDB migration and qualification, plus baseline-001 deployment and failed native gate; formal rate energy excluded'
d['measured_disjoint_setup_subtotal_j']=sum(x['energy_j'] for x in d['entries'] if x['energy_status']=='measured')
d['baseline_followup'].update(state='attempt_001_failed_native_cpu_interface',
    attempt_001_measured_subtotal_j=sum(x['energy_j'] for x in new if x['energy_status']=='measured'),
    attempt_001_status=ref(b/'status.json'),failure_inspection=ref(b/'failure-inspection.json'),
    resume_policy='Root coordinates fresh output. Preserve completed deployment and failed evidence; append future physical windows once, never recount this deployment.')
d['complete_setup_energy_j']=None; d['whole_campaign_energy_j']=None
p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(dict(ledger=ref(p),entries=len(d['entries']),
    baseline_measured_subtotal_j=d['baseline_followup']['attempt_001_measured_subtotal_j'],
    all_known_setup_windows_j=d['measured_disjoint_setup_subtotal_j'],complete_setup_energy_j=None)))
