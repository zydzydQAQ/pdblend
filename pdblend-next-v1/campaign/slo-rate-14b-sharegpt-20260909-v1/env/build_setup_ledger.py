"""Summarize terminal setup receipts locally without replaying large raw files."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json

ROOT = Path(__file__).resolve().parent.parent
GPUS = list(range(8))

def read(path):
    return json.loads(Path(path).read_text())

def ref(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def checked(reference):
    assert ref(reference['path']) == reference
    return read(reference['path'])

def window(start, end):
    return dict(start_s=start, end_s=end, duration_s=end-start,
                start_utc=datetime.fromtimestamp(start, timezone.utc).isoformat(),
                end_utc=datetime.fromtimestamp(end, timezone.utc).isoformat())

def build(node):
    root = ROOT / node
    projection_path = root / 'setup-status-projection.json'
    projections = read(projection_path)['operations']
    operations = []
    for name, description in [
        ('retirement', 'Prior resident services retired under the original node lease'),
        ('cold-bootstrap', 'Two native 14B TP1 engines cold-started; numerical reference and restoration'),
        ('fixed-qualification', '82 native shape cases, reference responses, two cancellations and restoration'),
        ('pdb-qualification', 'Six idle-domain recovery probes, controller drain and restoration'),
    ]:
        if name in projections:
            item = projections[name]
            state, state_ref = item['state'], item['status']
            qualification_ref = ref(root / name / 'qualified.json')
            assert checked(qualification_ref)['status'] == state_ref
        else:
            state_ref = ref(root / name / 'status.json')
            state = checked(state_ref)
            qualification_ref = None
        measurement = state.get('measurement', state.get('setup_measurement'))
        receipt = checked(measurement['receipt'])
        for key in ('measurement_valid', 'measurement_start_s', 'measurement_end_s', 'energy_j', 'gpu_indices'):
            assert measurement[key] == receipt[key]
        assert receipt['measurement_valid'] and receipt['gpu_indices'] == GPUS
        assert receipt['power_evidence']['power_source_verified']
        assert receipt['power_observer_stopped'] and receipt['memory_observer_stopped']
        samplers = []
        for sampler in receipt['isolated_samplers']:
            terminal = checked(sampler['receipt'])
            assert terminal['complete'] and terminal['child_exited'] and terminal['reader_stopped']
            samplers.append(dict(receipt=sampler['receipt'], complete=True,
                                 accounting_role='same measurement implementation; no additional energy term'))
        start, end = receipt['measurement_start_s'], receipt['measurement_end_s']
        operations.append(dict(operation_id=node+'-'+name, operation=name, description=description,
            category='pdb_setup', operation_status='passed', energy_status='measured',
            energy_j=receipt['energy_j'], window=window(start,end), gpu_indices=GPUS,
            all8_gpu_coverage=True, measurement_valid=True, receipt=measurement['receipt'],
            status=state_ref, status_projection=ref(projection_path) if name in projections else None,
            qualification=qualification_ref, sampler_receipts=samplers,
            power_source=receipt['power_evidence'],
            included_in_measured_setup_subtotal=True,
            accounting_note='Whole-node GPU energy in this receipt window; idle cards included. Not incremental deployment overhead and not an added serving-energy term.',
            raw_replay='Previously passed on the physical node before PDB handoff; this ledger checks terminal receipt SHA and metadata only.'))
    ordered = sorted(operations, key=lambda x: x['window']['start_s'])
    assert all(a['window']['end_s'] <= b['window']['start_s'] for a,b in zip(ordered,ordered[1:]))
    assert len({x['receipt']['sha256'] for x in operations}) == 4
    first = read(root/'environment-failed-001.json')
    ready = read(root/'pdb-ready.json')
    entries = list(operations)
    for filename, children, explanation in [
        ('environment-failed-001.json', ['retirement','cold-bootstrap'], 'CPU closure assertion after successful cold start; fixed native qualification had not begun.'),
        ('environment-failed-002.json', ['fixed-qualification'], 'CPU aiohttp import failure after the fixed gate passed; idle gate had not begun.'),
        ('environment-status.json', ['pdb-qualification'], 'Continuation supplied explicit dependency paths, replayed prior fixed evidence and completed only the idle gate.'),
    ]:
        state = read(root/filename)
        entries.append(dict(operation_id=node+'-'+filename.removesuffix('.json'),
            operation='environment_orchestration', category='pdb_setup',
            operation_status='passed' if state['complete'] else 'failed',
            energy_status='partial_child_operations_only', energy_j=None,
            window=window(state['started_s'],state['finished_s']),
            gpu_indices=None, all8_gpu_coverage=None, receipt=None, status=ref(root/filename),
            child_operations=[node+'-'+x for x in children],
            included_in_measured_setup_subtotal=False,
            accounting_note='No independent parent meter. Referenced child energy is already counted exactly once; parent energy is unknown, not zero.',
            description=explanation, error=state.get('error')))
    # Complement only the recorded preparation span. No assumption is made about
    # time or energy before that span, and parent windows above are never summed.
    span_start = first['started_s']
    span_end = ready['independent_handoff_verified_s']
    cursor = span_start
    gaps = []
    for operation in ordered:
        if cursor < operation['window']['start_s']:
            gaps.append(window(cursor,operation['window']['start_s']))
        cursor = operation['window']['end_s']
    if cursor < span_end:
        gaps.append(window(cursor,span_end))
    entries.append(dict(operation_id=node+'-unmetered-setup-intervals',
        operation='setup_unmetered_intervals',category='pdb_setup',operation_status='completed',
        energy_status='unknown',energy_j=None,gpu_indices=None,all8_gpu_coverage=None,receipt=None,
        windows=gaps,included_in_measured_setup_subtotal=False,
        accounting_note='Gaps include pre/post meter work, CPU checks, failure diagnosis, continuation delays and resident idle time. No continuous outer meter; no interpolation or zero filling.'))
    entries.append(dict(operation_id=node+'-cpu-preparation-and-staging',
        operation='model_sha_and_source_staging',category='pdb_setup',operation_status='completed',
        energy_status='unknown',energy_j=None,window=None,gpu_indices=None,all8_gpu_coverage=None,receipt=None,
        status=ref(root/'preparation/preflight.json'),source_audit=ref(ROOT/'env/source-audit.json'),
        included_in_measured_setup_subtotal=False,
        accounting_note='Model SHA, source packaging/distribution, CPU verification and dependency staging have no dedicated energy meter. Operation start/end and CPU/whole-server energy are unknown.'))
    future = dict(phase='baseline_after_PDB_boundary',state='pending_terminal_receipts',
        measured_energy_j=None,whole_campaign_energy_j=None,
        shared_by_systems=['mixed','distserve','dynamollm','ecoserve'],
        expected_operations=[
            dict(operation='baseline_deployment',receipt_path=str(root/'baseline-001/deployment/deployment-receipt.json'),energy_key='all8_operation_energy_j',window_keys=['operation_start_s','operation_end_s']),
            dict(operation='baseline_native_qualification',receipt_path=str(root/'baseline-001/qualification/native/status.json'),energy_key='full_operation_energy_j',window_keys=['measurement_start_s','measurement_end_s']),
            dict(operation='baseline_frequency_qualification',receipt_path=str(root/'baseline-001/qualification/frequency/status.json'),energy_key='full_operation_energy_j',window_keys=['measurement_start_s','measurement_end_s']),
        ],
        policy='Append each physical terminal operation once, including failed attempts. Four policy bindings share the same physical qualification; do not multiply its energy by four. Verify actual all-eight coverage and clock/power cleanup from each receipt; absent/invalid measurements stay unknown. Preserve overlapping diagnostic/primary/outer windows separately and never add them. CPU binding derivation and unmetered gaps remain unknown.')
    result = dict(schema='slo14-setup-energy-ledger-v1',node=node,
        scope='PDB migration and qualification through verified handoff; subsequent formal rate energy excluded',
        complete_accounting=False,ledger_status='known_setup_operations_recorded_with_explicit_gaps',
        energy_kind='whole-node GPU energy, not wall-plug or CPU energy',
        preparation_observation_span=window(span_start,span_end),
        measured_operation_count=4,
        measured_disjoint_setup_subtotal_j=sum(x['energy_j'] for x in operations),
        subtotal_is_complete_setup_energy=False,complete_setup_energy_j=None,whole_campaign_energy_j=None,
        nonoverlap_verified=True,all_measured_operations_cover_all8_gpus=True,
        methodology='Receipt-window SHA and terminal metadata checked locally. Prior on-host independent replay provides raw validation; no large raw read/transfer or GPU operation is performed by ledger creation.',
        entries=entries,baseline_followup=future,reporting_code=ref(__file__))
    destination=root/'setup-energy-ledger.json'
    destination.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(node=node,measured_subtotal_j=result['measured_disjoint_setup_subtotal_j'],
        measured_operations=4,unmetered_intervals=len(gaps),complete_setup_energy_j=None,ledger=ref(destination))))

if __name__ == '__main__':
    for node in ['A','C']:
        build(node)
