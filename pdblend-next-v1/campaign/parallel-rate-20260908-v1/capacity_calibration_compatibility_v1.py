"""Narrow P6 numerical-evidence reuse for the one-call P7 live-clock correction."""
import hashlib
import json
from pathlib import Path
from build_capacity_p5 import default_equivalent

RUNTIME = 'src/ecopadg/serving/runtime.py'
OLD_CALL = 'self.capacity_service.tick(time.time())'
NEW_CALL = 'self.capacity_service.tick()'
MODULES = {'capacity_runtime.py','capacity_executor.py','capacity_backend.py',
           'capacity_certificate.py','capacity_planner.py'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(value,message):
    if not value:
        raise ValueError(message)


def checked(reference):
    need(sha(reference['path']) == reference['sha256'],'compatibility evidence changed: '+reference['path'])
    return json.loads(Path(reference['path']).read_text())


def checked_host(reference):
    manifest=checked(reference);root=Path(reference['path']).parent
    need(manifest.get('files') and all(sha(root/name)==h for name,h in manifest['files'].items()),
         'compatibility host source changed')
    return manifest,root


def verify(reference, actual_controller_manifest, capacity):
    proof=checked(reference)
    need(proof['schema']=='P6-numerical-calibration-P7-live-controller-compatibility-v1'
         and proof['authorized'] is True and proof['reuses_numerical_bounds_only'] is True
         and proof['claims_P7_was_measured_for_original_certificate'] is False
         and proof['requires_new_actual_autonomous_underload_gate'] is True
         and proof['requires_new_actual_candidate900'] is True,
         'explicit numerical reuse and separate P7 qualification required')
    need(proof['actual_controller_manifest']==actual_controller_manifest,
         'compatibility does not cover actual controller')
    for path,digest in proof['files'].items():
        need(sha(path)==digest,'compatibility dependency changed: '+path)
    measured,oldroot=checked_host(proof['measured_controller_manifest'])
    actual,newroot=checked_host(actual_controller_manifest)
    need(measured['implementation_series']=='parallel-p6' and actual['implementation_series']=='parallel-p7'
         and measured['model']==actual['model']=='14b', 'only measured A P6 to A P7 reuse is declared')
    need(set(measured['files'])==set(actual['files'])
         and [n for n in measured['files'] if measured['files'][n]!=actual['files'][n]]==[RUNTIME],
         'only the live capacity clock call may differ')
    original=(oldroot/RUNTIME).read_text();candidate=(newroot/RUNTIME).read_text()
    need(original.count(OLD_CALL)==1 and candidate==original.replace(OLD_CALL,NEW_CALL),
         'controller delta exceeds the diagnosed single clock invocation')
    p4,p4root=checked_host(proof['capacity_disabled_original_manifest'])
    need(p4['implementation_series']=='parallel-p4' and p4['model']=='14b',
         'wrong capacity-disabled original source')
    ast_sha=default_equivalent((p4root/RUNTIME).read_text(),candidate)
    need(ast_sha==proof['capacity_disabled_full_P4_ast_sha256'],'capacity-disabled source differs')
    old=checked(proof['measured_capacity_binding'])
    need(old['files'] and all(sha(p)==h for p,h in old['files'].items()), 'original capacity binding changed')
    need(capacity['identity']==old['identity']==proof['calibration_source_identity']
         and capacity['calibrated_source_semantics']==old['calibrated_source_semantics'],
         'P6 numerical source identity must remain unchanged')
    semantics=old['calibrated_source_semantics']
    need(semantics['candidate_manifest']==proof['measured_controller_manifest']
         and set(semantics['capacity_modules'])==MODULES
         and semantics['capacity_modules']==proof['capacity_module_hashes']
         and all(measured['files'][n]==actual['files'][n]==h for n,h in semantics['capacity_modules'].items()),
         'physical capacity or numerical validator code differs')
    need(capacity['calibration']==old['calibration']==proof['certificate'], 'different numerical certificate')
    certificate=checked(proof['certificate'])
    need(certificate['identity']==old['identity'] and certificate['measurement_verified'] is True,
         'original certificate identity or verification differs')
    checked(semantics['profile'])
    need(len(proof['measured_calibration_specs'])==2, 'full-layout and supplemental-idle source proofs required')
    modes=set()
    for spec_ref in proof['measured_calibration_specs']:
        spec=checked(spec_ref);config=checked(spec['config']);modes.add(spec['mode'])
        need(config.get('capacity_integration_v1') is False
             and Path(spec['host_release'])==oldroot,
             'reused numerical calibration must have autonomous control disabled on P6')
    need(modes=={'layout_calibration','idle_calibration'}, 'wrong reused calibration modes')
    cpu=checked(proof['clock_age_cpu_reproduction'])
    need(cpu['passed'] is True and cpu['tests']==6 and cpu['gpu_actions'] is False
         and cpu['uses_actual_P6_CapacityService_tick'] is True,
         'actual clock-age CPU reproduction missing')
    for path,digest in cpu['files'].items():
        need(sha(path)==digest,'clock-age CPU evidence changed')
    return proof
