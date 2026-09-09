"""Freeze model-preserving hosts with the two reviewed frequency repairs."""
import argparse
import ast
import hashlib
import importlib.util
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent
STARTUP = CAMPAIGN / 'main-slo-v7-first-admission-frequency-candidate-v1'
UNCERTAIN = CAMPAIGN / 'B32B-v7-clock-uncertainty-successor-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(('unexpected source anchor', before, source.count(before)))
    return source.replace(before, after)


def startup_planner(source):
    insertion = '''            restriction=getattr(self,'startup_frequency_restrictions',{})
            if (d.instance_id in restriction and not d.requests and not d.running and not d.waiting):
                observed=restriction[d.instance_id]
                if (not observed or observed.get('snapshot_version')!=snapshot.version
                        or tuple(observed.get('gpus',()))!=tuple(d.gpus)
                        or not 0<=now-observed.get('observed_s',0)<=self.telemetry_ttl_s):
                    reject(self,'state_unavailable',instance_id=d.instance_id,detail='first admission has no fresh confirmed full-TP frequency')
                    continue
                fs=(observed['frequency_mhz'],)
'''
    source = once(source, '            for freq in fs:\n', insertion + '            for freq in fs:\n')
    maximum = 'frequencies=tuple(FrequencyAction(i.instance_id,self.max_frequency) for i in snapshot.instances),'
    if maximum in source:
        return once(source, maximum,
            "frequencies=tuple(FrequencyAction(i.instance_id,self.max_frequency) for i in snapshot.instances\n                    if i.instance_id not in getattr(self,'startup_frequency_restrictions',{})),")
    # C's existing recovery selects a measured frequency. Retain that policy
    # for eligible instances, excluding only unproved first-admission ones.
    return once(source,
        'recovery_actions(self,[(i,True) for i in snapshot.instances],',
        "recovery_actions(self,[(i,True) for i in snapshot.instances\n                if i.instance_id not in getattr(self,'startup_frequency_restrictions',{})],")


def startup_runtime(source):
    tree = ast.parse(source)
    if any(isinstance(node, ast.ImportFrom) and node.module == '__future__' for node in tree.body):
        raise ValueError('runtime future import requires an explicit insertion point')
    source = 'from .startup_frequency import admission_planner as startup_admission_planner, committed as startup_committed, failed as startup_failed\n' + source
    source = once(source,
        'async def recovery_plan(self, snapshot, request, now, *, pressure_pending=None):',
        'async def recovery_plan(self, snapshot, request, now, *, pressure_pending=None, admission_estimator=None):')
    source = once(source, 'observed_plan,capacity_admission_plan, self.planner, recovered, pending',
        'observed_plan,capacity_admission_plan, admission_estimator or self.planner, recovered, pending')
    source = once(source, '                    snapshot=self.state.snapshot\n                    planning_started=time.perf_counter()',
        '                    snapshot=self.state.snapshot\n                    admission_estimator=await startup_admission_planner(self,snapshot)\n                    now=time.time()\n                    planning_started=time.perf_counter()')
    source = once(source, 'observed_plan,capacity_admission_plan,self.planner,snapshot,',
        'observed_plan,capacity_admission_plan,admission_estimator,snapshot,')
    source = once(source, 'self.recovery_plan(snapshot, request, time.time(),pressure_pending=pressure_pending)',
        'self.recovery_plan(snapshot, request, time.time(),pressure_pending=pressure_pending,admission_estimator=admission_estimator)')
    source = once(source,
        "                                if self.evaluation_v3: active['timing']['backend_confirmed_s']=time.time()\n                            except BaseException:",
        "                                if self.evaluation_v3: active['timing']['backend_confirmed_s']=time.time()\n                                startup_committed(self,plan)\n                            except BaseException as startup_error:\n                                startup_failed(self,plan,startup_error)")
    return source


def helper():
    source = (STARTUP / 'candidate/startup_frequency.py').read_text()
    return once(source,
        '    async with controller.action_lock, clocks.lock, clocks.snapshot_guard():\n        if controller.state.snapshot.version!=snapshot.version:return planner\n',
        '''    async with controller.action_lock, clocks.lock, clocks.snapshot_guard():
        # A cancelled coroutine may leave a real executor write in progress.
        # Do not enqueue a frequency read behind it while holding state locks.
        if (getattr(clocks,'pending_physical_commands',{})
                or getattr(clocks,'physical_command_uncertainty',())):
            clocks.clock_event(dict(kind='first_admission_prior_physical_uncertainty',
                at_s=time.time(),safely_replannable=False,
                pending_sequences=list(getattr(clocks,'pending_physical_commands',{})),
                prior_uncertainty=list(getattr(clocks,'physical_command_uncertainty',())),
                requires_owned_close=True,measurement_confirmation=False))
            raise ClockWriteUncertain('startup observation blocked by unresolved physical command; owned close required')
        if controller.state.snapshot.version!=snapshot.version:return planner
''')


def build(model):
    parent = CAMPAIGN / f'main-slo-improvement-v1/hosts/{model}-fixed-v7'
    out = ROOT / f'hosts/{model}-fixed-p1'
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((parent / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if sha(parent / name) != digest:
            raise ValueError('parent host changed: ' + name)
    final_uncertain = UNCERTAIN / 'host-003/src/ecopadg/serving/backend.py'
    specification = importlib.util.spec_from_file_location('parallel_clock_build', UNCERTAIN / 'build.py')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    backend = module.transform((parent / 'src/ecopadg/serving/backend.py').read_text())
    if backend != final_uncertain.read_text():
        raise ValueError('backend composition differs from reviewed host-003')
    changed = {
        'src/ecopadg/serving/backend.py': backend,
        'src/ecopadg/serving/planner.py': startup_planner((parent / 'src/ecopadg/serving/planner.py').read_text()),
        'src/ecopadg/serving/runtime.py': startup_runtime((parent / 'src/ecopadg/serving/runtime.py').read_text()),
        'src/ecopadg/serving/startup_frequency.py': helper(),
    }
    for source in changed.values():
        ast.parse(source)
    for name in manifest['files']:
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name in changed:
            path.write_text(changed[name])
        else:
            shutil.copyfile(parent / name, path)
    (out / 'src/ecopadg/serving/startup_frequency.py').write_text(changed['src/ecopadg/serving/startup_frequency.py'])
    files = {name: sha(out / name) for name in sorted(set(manifest['files']) | set(changed))}
    refs = [parent / 'manifest.json', Path(__file__), STARTUP / 'manifest.json',
        STARTUP / 'candidate/startup_frequency.py', UNCERTAIN / 'manifest.json',
        UNCERTAIN / 'build.py', final_uncertain]
    result = dict(schema=1, implementation_series='parallel-p1', model=model,
        created_s=time.time(), parent_manifest=dict(path=str(parent / 'manifest.json'),sha256=sha(parent / 'manifest.json')),
        files=files, frozen_references={str(path):sha(path) for path in refs},
        changed_parent_files=sorted(set(changed) & set(manifest['files'])),
        added_files=sorted(set(changed)-set(manifest['files'])),
        features=['observed_first_admission_frequency_v1','persistent_physical_command_uncertainty'],
        model_specific_parent_policy_preserved=True, gpu_qualified=False,
        model_profiles_unchanged=True, unknown_physical_state_requires_owned_close=True)
    with (out / 'manifest.json').open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    return dict(model=model,host=str(out),manifest_sha256=sha(out / 'manifest.json'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=('7b','14b','32b'),required=True)
    print(json.dumps(build(parser.parse_args().model)))
