"""Pure contract counterexamples, never measured qualification fixtures."""
import ast
import copy
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('successor_contract_test',HERE/'verify.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

def main():
    parent=ast.parse((HERE.parent/'distributed14b-qualification-v1/verify.py').read_text());actual=ast.parse((HERE/'verify.py').read_text())
    old={f.name:ast.dump(f,include_attributes=False) for f in parent.body if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef))}
    new={f.name:ast.dump(f,include_attributes=False) for f in actual.body if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef))}
    unchanged=[k for k in old if k not in ('policy_identity','verify')]
    assert all(old[k]==new[k] for k in unchanged)
    assert set(new)-set(old)=={'controller_successor'}
    cases=['all_'+str(len(unchanged))+'_old_raw_native_profile_and_CP_functions_AST_equal']
    r=lambda x:dict(path='/CPU-fixture/'+x,sha256=x)
    par=r('parent');par['sha256']='5dcae91099369a188f4aecb92f7a8c06b138e6ae0ae181b786f7b7011d034b84'
    host=r('actual');cpu=r('cpu');lineage=r('lineage');profile=r('profile');reg=r('registration')
    q=dict(controller_successor=lineage,host_manifest=host,profile=profile,profile_registration=dict(registration=reg))
    values={lineage['path']:dict(schema='distributed14b-P10-to-P11-idle-domain-successor-v1',host_manifest=host,parent_manifest=par,cpu_validation=cpu,profile=profile,profile_registration=reg),
      par['path']:dict(files={'src/ecopadg/serving/idle_admission.py':'old','src/ecopadg/serving/profiles.py':'same'}),
      host['path']:dict(files={'src/ecopadg/serving/idle_admission.py':'new','src/ecopadg/serving/idle_domain_reacquire.py':'helper','src/ecopadg/serving/profiles.py':'same'}),
      cpu['path']:dict(passed=True,sources=[dict(manifest=host)])}
    original_bound,original_manifest=v.bound,v.manifest
    v.bound=lambda reference,files:values[reference['path']]
    v.manifest=lambda reference:None
    try:
        assert v.controller_successor(q,dict(idle_domain_reacquire_v1=True),{})['original_numeric_and_measurement_sources_preserved']
        cases.append('exact_opt_in_successor_contract_CPU_fixture_only')
        base=copy.deepcopy(values)
        changes=[('numeric_source_changed',lambda:values[host['path']]['files'].update({'src/ecopadg/serving/profiles.py':'changed'})),
            ('missing_helper',lambda:values[host['path']]['files'].pop('src/ecopadg/serving/idle_domain_reacquire.py')),
            ('CPU_failed',lambda:values[cpu['path']].update(passed=False)),
            ('source_absent_CPU',lambda:values[cpu['path']].update(sources=[])),
            ('profile_changed',lambda:values[lineage['path']].update(profile=r('wrong'))),
            ('registration_changed',lambda:values[lineage['path']].update(profile_registration=r('wrong')))]
        for name,mutate in changes:
            values.clear();values.update(copy.deepcopy(base));mutate()
            try:v.controller_successor(q,dict(idle_domain_reacquire_v1=True),{})
            except (ValueError,KeyError):cases.append('reject_'+name)
            else:raise AssertionError(name)
        values.clear();values.update(copy.deepcopy(base))
        for flag in (None,False,1):
            try:v.controller_successor(q,dict(idle_domain_reacquire_v1=flag),{})
            except ValueError:cases.append('reject_unenabled_flag_'+str(flag))
            else:raise AssertionError(flag)
    finally:v.bound,v.manifest=original_bound,original_manifest
    import json
    print(json.dumps(dict(passed=True,cpu_only=True,tests=len(cases),cases=cases)))

if __name__=='__main__':main()
