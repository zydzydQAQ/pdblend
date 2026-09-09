"""Declare exactly three fresh P8 physical repetitions; keep all original workloads."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

A = Path(__file__).resolve().parent
R = A.parent
HOST = R / 'hosts/14b-capacity-p8'
CODE = A / 'load-p8-code-001'
OUT = A / 'p8-cold-recalibration-inputs-001'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p): return dict(path=str(Path(p).resolve()), sha256=sha(p))
def need(v, message):
    if not v: raise ValueError(message)
def checked(r):
    need(sha(r['path']) == r['sha256'], 'changed source reference')
    return json.loads(Path(r['path']).read_text())
def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    need(not OUT.exists(), 'new immutable recalibration declaration required')
    recovery = checked(ref(A / 'p7-failed900-recovery-001/status.json'))
    need(recovery['complete'] and recovery['native_original_budget_restored']
         and recovery['all_eight_clock_reset_complete'] and recovery['node_lease_held'] is False,
         'failed P7 attempt must have a separately successful actual recovery')
    cpu = checked(ref(CODE / 'cpu-validation.json'))
    need(cpu['passed'] and not cpu['gpu_qualified'] and all(sha(p) == h for p, h in cpu['files'].items()),
         'actual driver CPU source differs')
    common_cpu = ref(R / 'cpu-validation-p8.json')
    need(checked(common_cpu)['passed'], 'common actual-source CPU evidence required')
    manifest = checked(ref(HOST / 'manifest.json'))
    need(ref(HOST / 'manifest.json')['sha256'] == '54f3e3b609dde6452127ae9d4d040ff1f18715baaa9d220c652147bc8f5ca40f',
         'frozen P8 source differs')
    compatibility = ref(R / 'common/P6-numerical-P8-recalibration-bootstrap-v2.json')
    proof = checked(compatibility)
    oldspec = checked(ref(A / 'load-p6-gate-inputs-002/spec.json'))
    original_cap = checked(proof['measured_capacity_binding'])
    oldconfig = checked(oldspec['config'])
    oldinitial = checked(oldspec['original_binding'])
    needed = dict(original_cap['files'])
    needed.update(proof['files']); needed.update(cpu['files'])
    needed.update({str(HOST / name): digest for name, digest in manifest['files'].items()})
    for rr in [compatibility, proof['verifier'], ref(HOST / 'manifest.json'), ref(CODE / 'manifest.json'),
               ref(CODE / 'cpu-validation.json'), common_cpu, ref(__file__), ref(A / 'p7-failed900-recovery-001/status.json')]:
        needed[rr['path']] = rr['sha256']
    declaration = dict(schema='P8-actual-cold3-remove3-work-declaration-v1', authorized=True,
        created_s=time.time(), source=ref(HOST / 'manifest.json'), original_P6_certificate=proof['original_P6_certificate'],
        profile=oldspec['profiles'], compatibility=compatibility, original_752_trace=oldspec['cycles'][0]['under_load'],
        calibration_requests_unchanged=True, three_distinct_physical_instances=True,
        formal_eligible=False, automatic_retries=False, deadline_s=None,
        campaign_lifecycle='until_declared_complete_v1', request_timeout_s=120,
        calibration_action_budget_s=360, independent_physical_safety_timeout_s=120, runs=[])
    sys.path[:0] = [str(CODE), str(HOST / 'src'), str(HOST), str(R), '/root/workspace/pdblend/.runtime-deps']
    from calibration_compatibility import validate_compatibility
    for repeat in (1, 2, 3):
        inputs = OUT / f'repeat-{repeat}'
        output = A / f'p8-cold-recalibration-{repeat:03d}'
        need(not output.exists(), 'recalibration output must be unexecuted')
        cap = copy.deepcopy(original_cap)
        cap.update(owner_id=f'ap8cold{repeat}', http_port_base=31800+100*(repeat-1), kv_port_base=62000,
            runtime_dir=str(output / 'runtime'), physical_operation_timeout_s=120,
            controller_calibration_compatibility=compatibility, files=dict(needed))
        write(inputs / 'capacity-binding.json', cap)
        capref = ref(inputs / 'capacity-binding.json')
        config = copy.deepcopy(oldconfig)
        config.update(instances=oldinitial['instances'], capacity_integration_v1=False,
            capacity_binding_path=capref['path'], capacity_binding_sha256=capref['sha256'], port=31850+100*(repeat-1))
        write(inputs / 'config.json', config)
        actions = {}
        for operation in ('restore', 'remove'):
            action = checked(oldspec['cycles'][0][operation])
            need(action['work_budget_s'] == 360 and action['deadline_s'] is None, 'original calibration limits changed')
            action.update(repeat=repeat, purpose='P8 fresh measured transition bound after explicit physical120/planning separation')
            write(inputs / (operation + '.json'), action)
            actions[operation] = ref(inputs / (operation + '.json'))
        work = dict(schema='P8-physical-repetition-v1', authorized=True, repeat=repeat,
            source=declaration['source'], profile=oldspec['profiles'], trace=declaration['original_752_trace'],
            capacity_binding=capref, config=ref(inputs / 'config.json'), actions=actions,
            automatic_retries=False, arrival_duration_s=60, request_timeout_s=120)
        write(inputs / 'work-declaration.json', work)
        spec = copy.deepcopy(oldspec)
        spec.update(host_release=str(HOST), capacity_binding=capref, config=work['config'],
            input_declaration=ref(inputs / 'work-declaration.json'), controller_calibration_compatibility=compatibility,
            cpu_evidence=common_cpu, api_base='http://127.0.0.1:'+str(config['port']),
            stop_path=str(A / 'STOP_P8_RECALIBRATION'), files=dict(needed),
            cycles=[dict(under_load=declaration['original_752_trace'], **actions)])
        for rr in [capref, work['config'], spec['input_declaration'], *actions.values(), declaration['original_752_trace']]:
            spec['files'][rr['path']] = rr['sha256']
        validate_compatibility(spec, cap)
        write(inputs / 'spec.json', spec)
        declaration['runs'].append(dict(repeat=repeat, spec=ref(inputs / 'spec.json'), output=str(output)))
    write(OUT / 'declaration.json', declaration)
    print(json.dumps(ref(OUT / 'declaration.json')))


if __name__ == '__main__': main()
