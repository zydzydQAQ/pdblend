"""C's frozen native-128 refusal proof, with explicit deadline accounting.

The original queue proof runs unchanged on the successful/refused cohort.
Every original request remains in the independent raw and arrival audits.
"""
import csv
import json
from pathlib import Path
import sys

ROOT = Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
sys.path.insert(0, str(ROOT / 'common/uniform-rate-20260909-v2'))
import support as p

ORIGINAL = dict(path=str(ROOT / 'common/token-evidence-v2/audit_native_queue_v3.py'),
    sha256='a9b18ec971053cd8fa53a08f42d2c79c58a2f3a55d135a0396499556490717ab')
RAW = dict(path=str(ROOT / 'common/token-evidence-v2/verify_native_raw_v3.py'),
    sha256='80ce376771525348453b0cf6d668fe20280cb941c7d20a823b9398971d628a9a')
TIMING = dict(path=str(ROOT / 'C/audit_eco_completed_v1.py'),
    sha256='c7aef592493a55c335ccfb0a50be6fe89f260d90845a6d99a48ad85de1d6fc18')
ENGINE = dict(path=str(ROOT.parent / 'AC-baseline-deployment-v1/engines/C/engine.py'),
    sha256='00fde2f6b0d3288feecfb6aacfccd0bee9362195ec90d2d38517caedb287ffc8')
IDENTITY = dict(path=str(ROOT / 'final_selected_baseline_v2.py'),
    sha256='6659f6270ef54994d1e478a93410c57aef14541f4dd804de9fd1ed61e5f524fc')
IMPORTED_SOURCES = {
    str(ROOT.parent / 'main-slo-improvement-v1/protocol.py'): 'c937f4126cec04d4db4721db6ab5b9d7be387d315f4dc68eef9df093454b3281',
    str(ROOT / 'source_identity_v3.py'): '9525d9c217322d35b2955b2bbb6f843ad4c98436520769a1b1cbb19c22be10fd',
    str(ROOT / 'source_identity.py'): 'bebb0b2adb0ea2d4760413ca8f38d6b56a6f718b3ce62b80a57bd02fb75fbbbf',
    str(ROOT / 'capacity_calibration_compatibility_v1.py'): '21c2fe4bb589799e76f74e732e0879b011cfa09fc8d5b933e30c98501b469edf',
    str(ROOT / 'final_baseline_overlay_v1.py'): '5b7b00b5cad7de7719cc2b2def895e13614c008f54556be7d565d0c875f4bacd',
    str(ROOT / 'source_identity_v2.py'): '65ff69a33f065f779e6e94f197981e828df30cce6a8ec1d069f407cd53cf3098',
}
NATIVE_ERROR = 'RuntimeError: HTTP 503: decode failed: bounded admission queue full'


def truth(value):
    return str(value).lower() in ('1', 'true')


def partition(rows):
    """Classify exact producer responses; no arbitrary HTTP failure is allowed."""
    good, refused, deadlines = [], [], []
    p.need(len({r['request_id'] for r in rows}) == len(rows), 'duplicate request identity')
    for row in rows:
        if truth(row.get('success')):
            p.need(not row.get('error') and not truth(row.get('request_timeout'))
                and row.get('token_count_source') == 'server_usage'
                and truth(row.get('token_ids_verified')), 'incomplete successful output')
            good.append(row)
        elif row.get('http_status') == '503':
            p.need(row.get('error') == NATIVE_ERROR and not truth(row.get('request_timeout'))
                and row.get('admission_rejection') in ('', None)
                and int(row.get('generated_tokens') or 0) == 0
                and int(row.get('n_text_chunks') or 0) == 0
                and not row.get('first_token_s') and not row.get('last_token_s'),
                '503 is not the exact zero-output native admission refusal')
            refused.append(row)
        else:
            p.need(truth(row.get('request_timeout'))
                and row.get('error') in ('request_hard_timeout', 'request_hard_timeout_before_dispatch')
                and row.get('http_status') in ('', '200')
                and row.get('admission_rejection') in ('', None, '0', 'False', 'false'),
                'failure is neither exact native refusal nor explicit request deadline')
            deadlines.append(row)
    return good, refused, deadlines


def queue_details(rows, events, original):
    good, refused, deadlines = partition(rows)
    p.need(refused, 'no native refusal to diagnose')
    ids = {r['request_id'] for r in rows}
    timing = [e for e in events if e.get('kind') == 'request_timing']
    p.need(len(timing) == len(ids) and {e['client_request_id'] for e in timing} == ids,
        'full original request timing cohort is incomplete or duplicated')
    cohort = good + refused
    selected = {r['request_id'] for r in cohort}
    selected_events = [e for e in events if e.get('client_request_id') in selected]
    details = original.queue_proof(cohort, selected_events, 128)
    p.need({d['request_id'] for d in details} == {r['request_id'] for r in refused},
        'native refusal proof does not cover exactly the native refusal cohort')
    return details, [r['request_id'] for r in deadlines]


def audit(checkpoint):
    reference = checkpoint if isinstance(checkpoint, dict) else p.ref(checkpoint)
    cp = p.checked(reference)
    for name in ('binding', 'receipt'):
        value = cp[name]
        if isinstance(value, str):
            value = dict(path=value, sha256=cp[name + '_sha256'])
        p.checked(value)
        cp[name] = value['path']
        cp[name + '_sha256'] = value['sha256']
    binding, receipt = p.read(cp['binding']), p.read(cp['receipt'])
    row, summary = cp['row'], receipt['summary']
    p.need(binding['hostname'] == 'iZwz9gfq11hx1sbob59yrgZ'
        and row['model'] == '7b' and row['system'] == 'ecoserve',
        'native refusal authority is only this actual C 7B EcoServe host')
    files = dict(binding['files'])
    files.update(IMPORTED_SOURCES)
    files.update(cp['artifacts'])
    files.update({reference['path']: reference['sha256'], cp['binding']: cp['binding_sha256'],
        cp['receipt']: cp['receipt_sha256']})
    for item in (ORIGINAL, RAW, TIMING, ENGINE, IDENTITY, p.ref(__file__)):
        files[item['path']] = item['sha256']
    for path, digest in files.items():
        p.need(p.sha(path) == digest, 'changed native evidence: ' + path)
    p.need(summary['measurement_valid'] and summary['fixed_window_valid']
        and summary['runtime_error'] is None and not summary['work_complete'], 'invalid native window')
    p.need(receipt['child_stopped'] and receipt['clock_restore_complete']
        and not receipt['outer_cleanup_errors']
        and all(v['complete'] for v in receipt['restoration'].values()), 'incomplete native cleanup')
    p.need(len(binding['instances']) == 8, 'original eight TP1 native members required')
    for instance in binding['instances']:
        imported = instance['provenance']['source_files_at_import']
        p.need(imported.get(ENGINE['path']) == ENGINE['sha256'] and instance['tp'] == 1,
            'actual imported native engine or TP differs from frozen refusal authority')
        config = p.read(instance['engine_config'])
        p.need(config.get('max_pending', 128) == 128, 'native admission bound changed')
    original = p.load(ORIGINAL, 'uniform_native_refusal_original')
    actual = original.identities.actual_identity(original.protocol, cp, binding, cp['receipt'])
    cell = Path(cp['receipt']).parents[2] / 'cells' / row['cell_id']
    with (cell / 'bench.csv').open() as stream:
        bench = list(csv.DictReader(stream))
    events = [json.loads(line) for line in (cell / 'control.jsonl').read_text().splitlines()]
    details, deadlines = queue_details(bench, events, original)
    timing = p.load(TIMING, 'uniform_native_refusal_full_timing').timing(bench, events)
    raw = p.load(RAW, 'uniform_native_refusal_full_raw').verify(reference['path'])
    p.need(raw['failed_requests'] == summary['failed_requests'] == len(details) + len(deadlines)
        and raw['request_timeouts'] == summary['request_timeouts'] == len(deadlines)
        and raw['completed_requests'] + len(details) + len(deadlines) == raw['expected_requests'],
        'native refusal and deadline counts do not partition the full denominator')
    stamps = sorted(e['at_s'] for e in events if 'at_s' in e)
    gap = max(z - a for a, z in zip(stamps, stamps[1:]))
    p.need(gap < 10, 'controller event gap invalidates native refusal evidence')
    return dict(schema='C-Eco-native128-refusal-and-deadline-audit-v1', passed=True,
        independently_recomputed=True, checkpoint=reference, binding=p.ref(cp['binding']),
        receipt=p.ref(cp['receipt']), original_engine_source=ENGINE, native_limit=128,
        native_rejections=len(details), request_timeouts=len(deadlines),
        failed_requests=len(details) + len(deadlines), native_rejection_request_ids=[d['request_id'] for d in details],
        timeout_request_ids=deadlines, refusal_details=details, native_rejections_are_not_timeouts=True,
        original_queue_proof_run_unchanged_on_success_and_refusal_cohort=True,
        all_original_requests_in_raw_and_arrival_audits=True,
        every_other_request_completed_exact_output=not deadlines,
        no_unknown_errors=True, cleanup_verified=True, measurement_valid=True, work_complete=False,
        equal_work_energy_comparison_eligible=False, not_hardware_saturation_proof=True,
        actual_identity=actual, timing=timing, raw=raw, controller_event_gap_max_s=gap,
        auditor_source=p.ref(__file__), files=files, no_gpu_actions=True)
