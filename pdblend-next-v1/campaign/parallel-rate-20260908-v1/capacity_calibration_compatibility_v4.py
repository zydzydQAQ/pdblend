"""Keep P6 numerical and P8 transition measurements explicit under P9 control."""
import importlib.util
import json
import sys
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parent
MODULES = {'capacity_runtime.py','capacity_executor.py','capacity_backend.py',
           'capacity_certificate.py','capacity_planner.py'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(value, message):
    if not value:
        raise ValueError(message)


def checked(reference):
    need(sha(reference['path']) == reference['sha256'], 'P9 reuse reference changed')
    return json.loads(Path(reference['path']).read_text())


def bound_module(reference, expected, name):
    need(Path(reference['path']).resolve() == expected.resolve()
         and sha(expected) == reference['sha256'], 'wrong P9 reuse verifier source')
    spec=importlib.util.spec_from_file_location(name, expected)
    module=importlib.util.module_from_spec(spec)
    prior=list(sys.path)
    try:
        sys.path.insert(0,str(ROOT))
        spec.loader.exec_module(module)
    finally:
        sys.path[:]=prior
    need(Path(module.__file__).resolve() == expected.resolve(), 'wrong loaded verifier')
    return module


def verify(reference, actual_controller_manifest, capacity):
    proof=checked(reference)
    need(proof['verifier']==dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
         'explicit actual P9 verifier must be bound')
    need(capacity['files'].get(reference['path']) == reference['sha256']
         and capacity['files'].get(proof['verifier']['path']) == proof['verifier']['sha256'],
         'capacity must pin proof and actual verifier')
    need(proof['schema']=='P6-layout-P8-transition-P9-fresh-clock-controller-compatibility-v1'
         and proof['authorized'] is True and proof['original_measurements_not_relabelled'] is True
         and proof['requires_actual_P9_autonomous_gate_and900'] is True,
         'P9 runtime qualification cannot be inferred from old measurements')
    need(proof['files'] and all(sha(p)==h for p,h in proof['files'].items()),
         'P9 compatibility dependency changed')
    predecessor=checked(proof['predecessor_P8_compatibility'])
    old_capacity=checked(proof['predecessor_P8_capacity_binding'])
    prior_verifier=bound_module(predecessor['verifier'],ROOT/'capacity_calibration_compatibility_v3.py',
                                'P9_exact_P8_calibration_verifier')
    prior_verifier.verify(proof['predecessor_P8_compatibility'],
                          proof['actual_transition_controller_manifest'],old_capacity)
    need(proof['actual_controller_manifest']==actual_controller_manifest,
         'actual P9 source differs from explicit reuse declaration')
    actual,actual_root=prior_verifier.checked_host(actual_controller_manifest)
    old,old_root=prior_verifier.checked_host(proof['actual_transition_controller_manifest'])
    need(actual['model']==old['model']=='14b' and actual['implementation_series']=='parallel-p9'
         and old['implementation_series']=='parallel-p8', 'only measured A P8 to A P9 declared')
    builder=bound_module(proof['source_equivalence_verifier'],ROOT/'build_capacity_p9.py',
                         'P9_exact_source_equivalence_builder')
    changed={builder.BACKEND,builder.RUNTIME}
    need(set(actual['files'])==set(old['files']) and
         {p for p in actual['files'] if actual['files'][p]!=old['files'][p]}==changed,
         'P9 source delta exceeds the two explicitly reviewed files')
    functions={builder.BACKEND:builder.patched_backend,builder.RUNTIME:builder.patched_runtime}
    equality={}
    for name,patch in functions.items():
        before=(old_root/name).read_text();after=(actual_root/name).read_text()
        need(after==patch(before), 'actual P9 differs from exact optional confirmation patch')
        equality[name]=builder.default_equivalent(before,after)
    need(equality==proof['feature_disabled_complete_parent_ast_sha256']
         ==actual['feature_disabled_complete_parent_ast_sha256'],
         'default-disabled complete P8 source equivalence failed')
    need(all(actual['files'][name]==old['files'][name] for name in MODULES),
         'physical calibration, numerical validation or capacity policy changed')
    for field in ('measured_controller_manifest','measured_capacity_binding','original_P6_certificate',
                  'certificate','calibration_source_identity','transition_source_selection'):
        need(proof[field]==predecessor[field], 'measured P6/P8 evidence was relabelled: '+field)
    need(proof['actual_transition_controller_manifest']==predecessor['actual_controller_manifest']
         and proof['certificate_scope']==predecessor['certificate_scope']
             =='original_P6_layout_savings_with_actual_P8_transition_groups'
         and proof['new_transition_qualification_complete'] is True,
         'whole original P8 transition group must be retained')
    locations={'owner_id','http_port_base','kv_port_base','runtime_dir','files',
               'controller_calibration_compatibility'}
    need({k:v for k,v in capacity.items() if k not in locations}
         =={k:v for k,v in old_capacity.items() if k not in locations},
         'P9 cannot change empirical costs, domains, profile, identity or physical budget')
    need(capacity['identity']==proof['calibration_source_identity']
         and capacity['calibration']==proof['certificate']
         and capacity['physical_operation_timeout_s']==120,
         'P9 measured identity/certificate/bounded120 contract changed')
    # This repeats the real raw certificate validation using the actual P9
    # numerical modules, whose byte identity to measured P8 was checked above.
    prior_verifier.validate_certificate_bound(proof['certificate'],capacity['identity'],actual_controller_manifest)
    return proof
