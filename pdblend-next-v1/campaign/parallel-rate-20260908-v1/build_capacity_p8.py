"""Separate empirical planning costs from a declared bounded physical transaction timeout."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time
from build_capacity_p5 import default_equivalent

ROOT = Path(__file__).resolve().parent
RUNTIME = 'src/ecopadg/serving/runtime.py'
DRAFT = ROOT/'A/operation-budget-draft-001'
CHANGED = {'capacity_executor.py', 'capacity_runtime.py'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_equivalence(parent, candidate):
    def methods(path):
        tree=ast.parse(path.read_text())
        return {n.name:ast.dump(n,include_attributes=False) for cls in tree.body
                if isinstance(cls,ast.ClassDef) and cls.name=='PhysicalCapacityExecutor'
                for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    old,new=methods(parent/'capacity_executor.py'),methods(candidate/'capacity_executor.py')
    equal={name:old[name]==new[name] for name in ('calibrate','_execute','_rollback','finish_to_initial')}
    if not all(equal.values()):raise ValueError('actual physical/calibration methods changed')
    oldruntime=(parent/'capacity_runtime.py').read_text()
    expected=oldruntime.replace("            deadline_s=binding['deadline_s'], max_residents=8//self.identity.tp,",
        "            deadline_s=binding['deadline_s'], max_residents=8//self.identity.tp,\n            physical_operation_timeout_s=binding.get('physical_operation_timeout_s'),")
    if (candidate/'capacity_runtime.py').read_text()!=expected:raise ValueError('capacity runtime scope exceeds constructor parameter')
    return equal


def build():
    prepared = []
    for model in ('7b', '14b', '32b'):
        parent = ROOT/'hosts'/f'{model}-capacity-p7'
        out = ROOT/'hosts'/f'{model}-capacity-p8'
        if out.exists():
            raise FileExistsError(out)
        manifest = json.loads((parent/'manifest.json').read_text())
        for name, digest in manifest['files'].items():
            if sha(parent/name) != digest:
                raise ValueError('frozen P7 source changed: ' + name)
        runtime = (parent/RUNTIME).read_text()
        physical_equivalence(parent,DRAFT)
        proof = default_equivalent((ROOT/'hosts'/f'{model}-fixed-p4'/RUNTIME).read_text(), runtime)
        prepared.append((model,parent,out,manifest,runtime,proof))
    shared = None
    for model,parent,out,manifest,runtime,proof in prepared:
        for name in manifest['files']:
            dest = out/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            if name in CHANGED:
                shutil.copyfile(DRAFT/name,dest)
            else:
                shutil.copyfile(parent/name,dest)
        files = {name:sha(out/name) for name in sorted(manifest['files'])}
        if shared is not None and files != shared:
            raise ValueError('cross-model source differs')
        shared = files
        if set(name for name in files if files[name] != manifest['files'][name]) != CHANGED:
            raise ValueError('unexpected P8 source change')
        result = dict(schema=8,model=model,implementation_series='parallel-p8',created_s=time.time(),
            files=files,parent_manifest=dict(path=str(parent/'manifest.json'),sha256=sha(parent/'manifest.json')),
            frozen_references={str(p):sha(p) for p in (Path(__file__),ROOT/'build_capacity_p5.py',DRAFT/'cpu-validation.json',DRAFT/'capacity_executor.py',DRAFT/'capacity_runtime.py')},
            common_controller_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
            identical_source_files_all_models=True,feature='capacity_integration_v1',default_enabled=False,
            changed_parent_files=sorted(CHANGED),added_files=[],gpu_qualified=False,
            live_capacity_decision_time_after_snapshot=True,explicit_clock_simulation_interface_unchanged=True,
            freshness_future_unknown_and_busy_rejections_unchanged=True,
            physical_execution_and_calibration_methods_ast_identical_to_p7=physical_equivalence(parent,out),
            physical_operation_timeout_optional_default=None, proposed_A_physical_operation_timeout_s=120,
            original_request_deadline_s=120, original_calibration_budget_s=360,
            planning_empirical_duration_and_energy_unchanged=True,
            successful_empirical_bound_exceedance_recorded=True,
            capacity_disabled_complete_p4_ast_sha256=proof,
            inherited_first_admission_and_clock_cancellation_fixes=True,
            profiles_unchanged=True,calibration_required_at_start=True,
            workload_end_restores_initial_layout=True)
        (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
        print(model,sha(out/'manifest.json'),result['common_controller_sha256'])


if __name__ == '__main__':
    build()
