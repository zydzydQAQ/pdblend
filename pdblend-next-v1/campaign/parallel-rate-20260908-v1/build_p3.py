"""One controller source tree; preserve each model's explicit configuration."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def once(source,before,after):
    if source.count(before)!=1:raise ValueError((before,source.count(before)))
    return source.replace(before,after)

def sources():
    a=ROOT/'hosts/14b-fixed-p2';c=ROOT/'hosts/7b-fixed-p2'
    base='src/ecopadg/serving/'
    common={base+n:(c/base/n).read_text() for n in ('profiles.py','tails.py','frequency.py')}
    planner=(a/base/'planner.py').read_text()
    planner=once(planner,'from .frequency import FrequencyCost\n','from .frequency import FrequencyCost,recovery_actions\n')
    planner=once(planner,'                 independent_idle_mixed_on_stale_tail=False):',
        '                 independent_idle_mixed_on_stale_tail=False, coverage_aware_recovery=False):')
    planner=once(planner,'        self.protect_pending_decode = protect_pending_decode\n',
        '        self.protect_pending_decode = protect_pending_decode\n        self.coverage_aware_recovery = coverage_aware_recovery\n')
    planner=once(planner,'        return self.profiles.lookup(instance.role, instance.tp, freq,',
        '        return self.profiles.lookup_execution_phase(instance.role, instance.tp, freq,')
    planner=once(planner,'''            return ControlPlan(snapshot.version,now,now+self.telemetry_ttl_s,
                frequencies=tuple(FrequencyAction(i.instance_id,self.max_frequency) for i in snapshot.instances),
                reason="infeasible or unmeasured: restore capacity and retain admission queue",feasible=False)
''', '''            actions,expires,covered=recovery_actions(self,[(i,True) for i in snapshot.instances],
                now,now+self.telemetry_ttl_s,maximum=self.max_frequency)
            return ControlPlan(snapshot.version,now,expires,frequencies=actions,
                reason=("infeasible: measured coverage recovery and retain admission queue" if covered else
                        "infeasible or unmeasured: restore capacity and retain admission queue"),feasible=False)
''')
    common[base+'planner.py']=planner
    runtime=(a/base/'runtime.py').read_text()
    runtime=once(runtime,"            independent_idle_mixed_on_stale_tail=(self.strategy.startswith('pdblend')\n",
        "            coverage_aware_recovery=(self.strategy.startswith('pdblend')\n                and config.get('coverage_aware_recovery',False)),\n            independent_idle_mixed_on_stale_tail=(self.strategy.startswith('pdblend')\n")
    common[base+'runtime.py']=runtime
    for source in common.values():ast.parse(source)
    return common

def build():
    parents={m:ROOT/'hosts'/f'{m}-fixed-p2' for m in ('7b','14b','32b')}
    manifests={m:json.loads((p/'manifest.json').read_text()) for m,p in parents.items()}
    common=sources()
    for m,parent in parents.items():
        for n,h in manifests[m]['files'].items():
            if sha(parent/n)!=h:raise ValueError('changed parent '+str(parent/n))
    names=set(manifests['14b']['files'])
    if not all(set(v['files'])==names for v in manifests.values()):raise ValueError('unequal source inventories')
    differing={n for n in names if len({v['files'][n] for v in manifests.values()})>1}
    if differing!=set(common):raise ValueError(('unreviewed cross-model difference',differing^set(common)))
    frozen=[]
    for m,parent in parents.items():
        out=ROOT/'hosts'/f'{m}-fixed-p3'
        if out.exists():raise FileExistsError(out)
        for n in sorted(names):
            dst=out/n;dst.parent.mkdir(parents=True,exist_ok=True)
            if n in common:dst.write_text(common[n])
            else:shutil.copyfile(parent/n,dst)
        files={n:sha(out/n) for n in sorted(names)}
        if frozen and files!=frozen[0]:raise ValueError('controller files are not identical')
        frozen.append(files)
        refs=[p/'manifest.json' for p in parents.values()]+[Path(__file__)]
        result=dict(schema=3,model=m,implementation_series='parallel-p3',created_s=time.time(),files=files,
            frozen_references={str(p):sha(p) for p in refs},
            parent_manifest=dict(path=str(parent/'manifest.json'),sha256=sha(parent/'manifest.json')),
            common_controller_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
            identical_source_files_all_models=True,feature='observed_idle_admission_frequency_v2',
            original_model_switches_preserved=True,profiles_not_modified=True,
            existing_decode_policy_preserved=True,persistent_physical_uncertainty_barrier_preserved=True,
            completion_recovery_preserves_original_slo=True,gpu_qualified=False)
        with (out/'manifest.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(m,sha(out/'manifest.json'),result['common_controller_sha256'])

if __name__=='__main__':build()
