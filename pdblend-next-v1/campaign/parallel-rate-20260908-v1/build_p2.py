"""Preserve the diagnostic binding when constraining every idle admission."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parent
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(source,before,after):
    if source.count(before)!=1:raise ValueError((before,source.count(before)))
    return source.replace(before,after)


def transform(source):
    source=source.replace('observed_first_admission_frequency_v1','observed_idle_admission_frequency_v2')
    source=once(source,"    used=getattr(controller,'startup_admission_used',set())\n",'')
    source=once(source,"    cold=[i for i in snapshot.instances if i.role=='mixed' and i.instance_id not in used\n",
        "    cold=[i for i in snapshot.instances if i.role=='mixed'\n")
    source=once(source,'    planner=copy.copy(original)\n', '''    planner=copy.copy(original)
    audit=getattr(original,'_admission_diagnostics',None)
    if audit is not None:
        # attach() stores an instance closure capturing the ORIGINAL bound
        # candidate method. A shallow copy must explicitly bind its own method,
        # retaining the same diagnostic scope and already wrapped profile store.
        bound=type(planner).candidates.__get__(planner,type(planner))
        def observed_candidates(snapshot,request,now):
            with audit.candidate(snapshot,request):
                return bound(snapshot,request,now)
        planner.candidates=observed_candidates
''')
    source=source.replace('first_admission_prior_physical_uncertainty','idle_admission_prior_physical_uncertainty')
    source=source.replace('first_admission_idle_wakeup_preserved','idle_admission_idle_wakeup_preserved')
    source=source.replace('first_admission_frequency_observation','idle_admission_frequency_observation')
    source=source.replace("scope='first admission only; profile/SLO and final physical confirmation still required'",
        "scope='every empty mixed admission; original profile/SLO, nonempty policy and final physical confirmation retained'")
    source=source.replace('First admission only: observe and retain a covered clock before reserving work.',
        'Every empty admission: observe and retain a covered clock before reserving work.')
    source=source.replace('No writes, no fallback, no retry classification. Each successful first admission\nconsumes the rule; later empty and existing-decode decisions keep their policy.',
        'No writes, no fallback, no retry classification. Diagnostic wrappers bind to\nthe constrained planner copy. Nonempty and existing-decode policies are retained.')
    ast.parse(source);return source


def build(model):
    parent=ROOT/'hosts'/f'{model}-fixed-p1';out=ROOT/'hosts'/f'{model}-fixed-p2'
    if out.exists():raise FileExistsError(out)
    manifest=json.loads((parent/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if sha(parent/name)!=digest:raise ValueError('changed p1 parent '+name)
    for name in manifest['files']:
        dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True)
        if name=='src/ecopadg/serving/startup_frequency.py':dst.write_text(transform((parent/name).read_text()))
        else:shutil.copyfile(parent/name,dst)
    refs={str(parent/'manifest.json'):sha(parent/'manifest.json'),str(Path(__file__)):sha(__file__)}
    result=dict(schema=2,model=model,implementation_series='parallel-p2',created_s=time.time(),
        files={name:sha(out/name) for name in manifest['files']},frozen_references=refs,
        parent_manifest=dict(path=str(parent/'manifest.json'),sha256=sha(parent/'manifest.json')),
        changed_parent_files=['src/ecopadg/serving/startup_frequency.py'],
        feature='observed_idle_admission_frequency_v2',default_enabled=False,
        diagnostic_candidate_rebound_to_constrained_clone=True,
        every_empty_mixed_admission_uses_observed_frequency=True,
        existing_decode_policy_preserved=True,profiles_unchanged=True,
        persistent_physical_uncertainty_barrier_preserved=True,gpu_qualified=False)
    with (out/'manifest.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(model,sha(out/'manifest.json'))


if __name__=='__main__':
    for model in ('7b','14b','32b'):build(model)
