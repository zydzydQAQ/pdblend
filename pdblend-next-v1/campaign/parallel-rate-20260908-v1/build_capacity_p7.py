"""Take the live capacity decision timestamp after its asynchronous GPU snapshot."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time
from build_capacity_p5 import default_equivalent

ROOT = Path(__file__).resolve().parent
RUNTIME = 'src/ecopadg/serving/runtime.py'
OLD = 'self.capacity_service.tick(time.time())'
NEW = 'self.capacity_service.tick()'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patched(source):
    if source.count(OLD) != 1 or NEW in source:
        raise ValueError('unexpected live capacity invocation')
    value = source.replace(OLD, NEW)
    ast.parse(value)
    return value


def build():
    prepared = []
    for model in ('7b', '14b', '32b'):
        parent = ROOT/'hosts'/f'{model}-capacity-p6'
        out = ROOT/'hosts'/f'{model}-capacity-p7'
        if out.exists():
            raise FileExistsError(out)
        manifest = json.loads((parent/'manifest.json').read_text())
        for name, digest in manifest['files'].items():
            if sha(parent/name) != digest:
                raise ValueError('frozen P6 source changed: ' + name)
        runtime = patched((parent/RUNTIME).read_text())
        proof = default_equivalent((ROOT/'hosts'/f'{model}-fixed-p4'/RUNTIME).read_text(), runtime)
        prepared.append((model,parent,out,manifest,runtime,proof))
    shared = None
    for model,parent,out,manifest,runtime,proof in prepared:
        for name in manifest['files']:
            dest = out/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            if name == RUNTIME:
                dest.write_text(runtime)
            else:
                shutil.copyfile(parent/name,dest)
        files = {name:sha(out/name) for name in sorted(manifest['files'])}
        if shared is not None and files != shared:
            raise ValueError('cross-model source differs')
        shared = files
        if [name for name in files if files[name] != manifest['files'][name]] != [RUNTIME]:
            raise ValueError('unexpected P7 source change')
        result = dict(schema=7,model=model,implementation_series='parallel-p7',created_s=time.time(),
            files=files,parent_manifest=dict(path=str(parent/'manifest.json'),sha256=sha(parent/'manifest.json')),
            frozen_references={str(p):sha(p) for p in (Path(__file__),ROOT/'build_capacity_p5.py')},
            common_controller_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
            identical_source_files_all_models=True,feature='capacity_integration_v1',default_enabled=False,
            changed_parent_files=[RUNTIME],added_files=[],gpu_qualified=False,
            live_capacity_decision_time_after_snapshot=True,explicit_clock_simulation_interface_unchanged=True,
            freshness_future_unknown_and_busy_rejections_unchanged=True,
            capacity_modules_and_physical_transition_code_byte_identical_to_p6=True,
            capacity_disabled_complete_p4_ast_sha256=proof,
            inherited_first_admission_and_clock_cancellation_fixes=True,
            profiles_unchanged=True,calibration_required_at_start=True,
            workload_end_restores_initial_layout=True)
        (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
        print(model,sha(out/'manifest.json'),result['common_controller_sha256'])


if __name__ == '__main__':
    build()
