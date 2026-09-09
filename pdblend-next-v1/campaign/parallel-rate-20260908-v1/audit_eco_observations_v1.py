"""Recompute arrival fidelity and distinguish diagnosed EcoServe overload.

The caller independently verifies raw energy, exact workload, native cleanup,
clock release and actual engine identity. No failure is silently made eligible.
"""
import csv
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NEGATIVE = ROOT / 'C/audit_eco_native_queue_negative_v1.py'
NEGATIVE_SHA = 'c165006fdc260e48958d5be211320b8300abf431b7243510f8bab728dd386809'
B_QUALIFIER = ROOT / 'B/qualify_eco_drained_baseline_v2.py'
B_QUALIFIER_SHA = '8362c816f2c1e4d111b32f266748b7aa5be8c93fd76c0eeada21284d7a5d6f51'
B_QUALIFIED_BINDING = ROOT / 'B/eco-drain-qualification-001/ecoserve/binding.json'
B_QUALIFIED_SHA = 'ffb9393f0c8da9ddf629a2a868787c9ec9c4c996e03ffc88955f8e33da95d534'
B_RULES_SHA = '4bf519003d426cd4062152652dcc4833ff7d8d69a79bc1af2aa1a2a21c678ade'
C_QUALIFIED_BINDING = ROOT / 'C/eco-drain37-v1/qualified/binding.json'
C_QUALIFIED_SHA = 'efb40b15084f1965381d6161ab1344db97fbfaae702f6be041c686f4b2851932'
C_GATE_AUDITOR = ROOT.parent / 'AC-baseline-binding-v2/gate_evidence.py'
C_GATE_AUDITOR_SHA = 'b84a113563f1b064be5ef7a8cbf2006b0790f3b60bb1be3851ce66114dd9e9e6'
_cache = {}


def exact_module(p, path, digest, name):
    p.need(p.sha(path) == digest, 'EcoServe independent auditor source changed')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timing(p, bench, events):
    records = [e for e in events if e.get('kind') == 'request_timing']
    by_id = {e['client_request_id']: e for e in records}
    p.need(len(bench) > 0 and len(by_id) == len(records) == len(bench)
           and len({r['request_id'] for r in bench}) == len(bench)
           and set(by_id) == {r['request_id'] for r in bench}, 'EcoServe request timing IDs differ')
    lateness, handlers = [], []
    for r in bench:
        planned, actual, deadline = (float(r[k]) for k in
            ('planned_arrival_s', 'actual_dispatch_s', 'request_deadline_s'))
        p.need(all(math.isfinite(v) for v in (planned, actual, deadline))
               and abs(deadline - planned - 120) < 1e-5
               and r['open_loop_independent'] == 'True' and -1e-5 <= actual - planned <= 1,
               'EcoServe arrival or original request budget invalid')
        event = by_id[r['request_id']]
        p.need(event['planned_arrival_s'] == planned and event['actual_dispatch_s'] == actual
               and event['hard_deadline_s'] == deadline, 'EcoServe client/controller timing differs')
        delay = event['handler_arrival_s'] - actual
        p.need(math.isfinite(delay) and 0 <= delay <= 1, 'EcoServe handler starved')
        lateness.append(actual - planned)
        handlers.append(delay)
    ordered = sorted(lateness)
    q = (len(ordered) - 1) * .99
    k = int(q)
    p99 = ordered[k] + (ordered[min(k+1, len(ordered)-1)] - ordered[k]) * (q-k)
    p.need(p99 <= .1, 'EcoServe p99 dispatch latency exceeds the original engineering gate')
    return dict(dispatch_lateness_max_s=max(lateness), dispatch_lateness_p99_s=p99,
                handler_delay_max_s=max(handlers), request_budget_s=120)


def b_qualification(p, binding):
    reference = dict(path=str(B_QUALIFIED_BINDING), sha256=B_QUALIFIED_SHA)
    base = p.checked(reference)
    for key in ('qualification', 'qualifier_source', 'priority_qualification_inputs',
                'instances', 'host_release', 'mechanism_proof', 'oracle', 'identity_file'):
        p.need(binding[key] == base[key], 'EcoServe performance changed its fresh qualification: ' + key)
    p.need(binding['legacy_single_vs_pair_exact'] is False
           and binding['mechanism_proof']['overall_runtime_gate_passed'] is False,
           'EcoServe original legacy mismatch must remain visible')
    p.need(binding['files'][base['qualification']['path']] == base['qualification']['sha256'],
           'EcoServe native qualification not frozen in actual binding')
    if B_QUALIFIED_SHA not in _cache:
        module = exact_module(p, B_QUALIFIER, B_QUALIFIER_SHA, 'root_exact_eco_b_qualifier')
        module.audit_binding(reference)
        _cache[B_QUALIFIED_SHA] = True
    return dict(binding=reference, qualification=base['qualification'],
                auditor_source=dict(path=str(B_QUALIFIER), sha256=B_QUALIFIER_SHA),
                independently_recomputed=True, legacy_single_vs_pair_exact=False,
                numerical_scope='registered native-default trajectory at the same execution shape')


def c_qualification(p, binding):
    reference = dict(path=str(C_QUALIFIED_BINDING), sha256=C_QUALIFIED_SHA)
    base = p.checked(reference)
    for key in ('instances', 'host_release', 'correctness_evidence', 'mechanism_proof'):
        p.need(binding[key] == base[key], 'C performance changed its measured native qualification: ' + key)
    regenerated = set(base['configs'].values()) | {str(C_QUALIFIED_BINDING.parent / 'identity.json')}
    for path, digest in base['files'].items():
        p.need(p.sha(path) == digest and (binding['files'].get(path) == digest
               or path in regenerated and path not in binding['files']),
               'C fresh native qualification source/raw file changed')
    for dataset, original_path in base['configs'].items():
        actual_path = binding['configs'][dataset]
        p.need(binding['files'][actual_path] == p.sha(actual_path), 'C actual config not frozen')
        old, actual = p.read(original_path), p.read(actual_path)
        p.need({k:v for k,v in old.items() if k != 'journal'}
               == {k:v for k,v in actual.items() if k != 'journal'}, 'C continuation changed native serving configuration')
    p.need(p.sha(C_GATE_AUDITOR) == C_GATE_AUDITOR_SHA, 'C raw gate auditor changed')
    if C_QUALIFIED_SHA not in _cache:
        # Use the actual C release in an isolated interpreter, so an imported
        # A/B ecopadg module cannot silently supply the power validity rules.
        program = '''import hashlib,importlib.util,json,sys
from pathlib import Path
host,auditor,binding_path=sys.argv[1:]
sys.path.insert(0,str(Path(host)/'src'))
from ecopadg.serving import measurement
spec=importlib.util.spec_from_file_location('exact_C_gate_raw',auditor)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
binding=json.loads(Path(binding_path).read_text())
result,files=module.audit(Path(binding['correctness_evidence']),binding['instances'],'ecoserve',measurement.power_evidence)
requests=json.loads((Path(binding['correctness_evidence'])/'checks/checks.json').read_text())['requests']
print(json.dumps(dict(result=result,files=files,actual_request_records=len(requests),power_module=str(Path(measurement.__file__).resolve()),power_sha=hashlib.sha256(Path(measurement.__file__).read_bytes()).hexdigest())))
'''
        process = subprocess.run([sys.executable, '-I', '-c', program, base['host_release'],
            str(C_GATE_AUDITOR), str(C_QUALIFIED_BINDING)], capture_output=True, text=True)
        p.need(process.returncode == 0, 'C raw native gate failed: ' + process.stderr[-1500:])
        result = json.loads(process.stdout)
        expected_power = str(Path(base['host_release']) / 'src/ecopadg/serving/measurement.py')
        p.need(result['power_module'] == expected_power and result['power_sha'] == p.sha(expected_power)
               and result['result'] == base['mechanism_proof'] and result['actual_request_records'] == 55,
               'C actual native gate reconstruction differs')
        p.need(all(base['files'].get(path) == digest for path, digest in result['files'].items()),
               'C native gate was not completely pinned before performance')
        _cache[C_QUALIFIED_SHA] = result
    return dict(binding=reference, gate=base['correctness_evidence'],
        auditor_source=dict(path=str(C_GATE_AUDITOR), sha256=C_GATE_AUDITOR_SHA),
        independently_recomputed=True, actual_request_records=_cache[C_QUALIFIED_SHA]['actual_request_records'],
        verified=_cache[C_QUALIFIED_SHA]['result']['verified'])


def verify(p, cp, binding, summary, directory, checkpoint):
    bench = list(csv.DictReader((directory / 'bench.csv').open()))
    events = [json.loads(s) for s in (directory / 'control.jsonl').read_text().splitlines()]
    p.need(summary['runtime_error'] is None, 'EcoServe controller error is not a capacity observation')
    result = dict(timing=timing(p, bench, events), original_raw_unchanged=True)
    model = cp['row']['model']
    p.need(model in ('7b', '32b'), 'A EcoServe awaits its actual final fresh native qualification')
    if model == '7b':
        result['native_qualification'] = c_qualification(p, binding)
    if model == '32b':
        rules = cp['execution_rules']
        p.need(rules['sha256'] == B_RULES_SHA
               and binding['files'][rules['path']] == rules['sha256'], 'B EcoServe arrival rules changed')
        p.checked(rules)
        result['native_qualification'] = b_qualification(p, binding)
    if summary['work_complete']:
        p.need(summary['failed_requests'] == summary['request_timeouts'] == 0
               and all(r['success'] == '1' for r in bench), 'EcoServe complete-work flag differs')
        result.update(classification='valid_complete_work_observation', classified_native_queue_refusals=0)
        return result
    # A/B future failures remain unqualified until an explicit diagnosis is
    # independently verified. The known C error is one unique native branch.
    p.need(model == '7b', 'EcoServe incomplete observation awaits an independent diagnosis')
    try:
        module = exact_module(p, NEGATIVE, NEGATIVE_SHA, 'root_exact_eco_native_queue_auditor')
        negative = module.audit(Path(checkpoint))
    except AssertionError as exc:
        raise ValueError('EcoServe native queue diagnosis failed: ' + str(exc)) from exc
    p.need(negative['passed'] is True and negative['checkpoint'] == p.ref(checkpoint)
           and negative['classification'] == 'valid_incomplete_original_native_queue_capacity_negative'
           and negative['every_other_request_completed_exact_output'] is True
           and negative['no_accepted_request_stranded'] is True,
           'EcoServe failure is not a proven native capacity refusal')
    result.update(classification=negative['classification'], native_capacity_diagnosis=negative,
                  classified_native_queue_refusals=negative['failed_requests'],
                  diagnosed_refusal_generated_tokens=0,
                  not_hardware_saturation_proof=True,
                  equal_work_energy_comparison_eligible=False)
    return result
