"""Freeze the final action-lock admission repair and persistent clock barrier."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'main-slo-improvement-v1'
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()

def once(source,before,after):
    if source.count(before)!=1:raise ValueError((before,source.count(before)))
    return source.replace(before,after)

def helper():
    source=(PARENT/'common/idle_admission_v8.py').read_text()
    source=once(source,'from .state import ExpiredPlan',
        'from .state import ExpiredPlan\nfrom .backend import ClockWriteUncertain\nfrom .completion_policy import recovery_budget, recovery_snapshot')
    source=once(source,'async def current_clock_first_plan(controller, plan, request):',
        'async def current_clock_first_plan(controller, plan, request, *, completion_recovery=False):')
    source=once(source,"    if controller.config.get('measured_frequency_write_guard_v1') is not True:\n",
        "    if (controller.config.get('observed_idle_admission_frequency_v2') is not True\n            or controller.config.get('measured_frequency_write_guard_v1') is not True):\n")
    source=once(source,'    async with clocks.lock, controller.state.lock:\n', '''    async with clocks.lock, controller.state.lock:
        # Executor cancellation does not establish a known physical state.
        # Never queue observations behind an unresolved command under locks.
        if (clocks.pending_physical_commands or clocks.physical_command_uncertainty):
            clocks.clock_event(dict(kind='idle_admission_prior_physical_uncertainty',
                at_s=time.time(),safely_replannable=False,
                pending_sequences=list(clocks.pending_physical_commands),
                prior_uncertainty=list(clocks.physical_command_uncertainty),
                requires_owned_close=True,measurement_confirmation=False))
            raise ClockWriteUncertain('idle observation blocked by unresolved physical command; owned close required')
''')
    source=once(source,'        candidates=controller.planner.candidates(snapshot,request,now)\n', '''        candidate_snapshot,candidate_request=snapshot,request
        if completion_recovery:
            if (not controller.evaluation_v3 or request.hard_deadline_s is None
                    or now>=request.hard_deadline_s):
                reject('completion recovery lacks a remaining original hard deadline')
            # Exactly the existing completion policy, only for planner inputs.
            # The caller still reserves and scores the original request budget.
            candidate_snapshot=recovery_snapshot(snapshot,now)
            candidate_request=recovery_budget(request,now)
        candidates=controller.planner.candidates(candidate_snapshot,candidate_request,now)
''')
    source=once(source,"        event.update(allowed=True,chosen_frequency_mhz=chosen.frequencies[0].frequency_mhz,",
        "        event.update(allowed=True,completion_recovery=completion_recovery,chosen_frequency_mhz=chosen.frequencies[0].frequency_mhz,")
    source=once(source,"        return replace(chosen,reason=('idle first admission preserves original measured target with all-TP idle-only throttle and deferred wakeup verification; '\n",
        "        return replace(chosen,expires_s=min(chosen.expires_s,plan.expires_s,\n                request.hard_deadline_s if request.hard_deadline_s is not None else float('inf')),\n            reason=('completion recovery; original SLO retained: ' if completion_recovery else '')+('idle first admission preserves original measured target with all-TP idle-only throttle and deferred wakeup verification; '\n")
    ast.parse(source);return source

def runtime(source):
    source=once(source,'from .tails import admission_budget\n',
        'from .tails import admission_budget\nfrom .idle_admission import current_clock_first_plan\n')
    source=once(source,'                    snapshot=self.state.snapshot\n                    planning_started=time.perf_counter()',
        '                    snapshot=self.state.snapshot\n                    completion_recovery=False\n                    planning_started=time.perf_counter()')
    source=once(source,'                            plan=recovered\n',
        '                            plan=recovered\n                            completion_recovery=True\n')
    source=once(source,'                            committed_s=time.time()\n',
        '                            plan=await current_clock_first_plan(self,plan,request,completion_recovery=completion_recovery)\n                            committed_s=time.time()\n')
    ast.parse(source);return source

def build(model):
    parent=PARENT/'hosts'/f'{model}-fixed-v7'
    clock=ROOT/'hosts'/f'{model}-fixed-p1'
    out=ROOT/'hosts'/f'{model}-fixed-p2'
    if out.exists():raise FileExistsError(out)
    manifest=json.loads((parent/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if sha(parent/name)!=digest:raise ValueError('changed v7 parent '+name)
    cm=json.loads((clock/'manifest.json').read_text())
    backend='src/ecopadg/serving/backend.py'
    if sha(clock/backend)!=cm['files'][backend]:raise ValueError('changed reviewed backend')
    changed={backend:(clock/backend).read_text(),
        'src/ecopadg/serving/runtime.py':runtime((parent/'src/ecopadg/serving/runtime.py').read_text()),
        'src/ecopadg/serving/idle_admission.py':helper()}
    for name in sorted(set(manifest['files'])|set(changed)):
        dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True)
        if name in changed:dst.write_text(changed[name])
        else:shutil.copyfile(parent/name,dst)
    refs=[parent/'manifest.json',clock/'manifest.json',Path(__file__),PARENT/'common/idle_admission_v8.py']
    result=dict(schema=2,model=model,implementation_series='parallel-p2',created_s=time.time(),
        files={name:sha(out/name) for name in sorted(set(manifest['files'])|set(changed))},
        frozen_references={str(p):sha(p) for p in refs},
        parent_manifest=dict(path=str(parent/'manifest.json'),sha256=sha(parent/'manifest.json')),
        changed_parent_files=sorted(set(changed)&set(manifest['files'])),added_files=sorted(set(changed)-set(manifest['files'])),
        feature='observed_idle_admission_frequency_v2',default_enabled=False,
        every_empty_mixed_admission_uses_final_action_lock=True,
        original_planner_and_diagnostic_binding_preserved=True,
        original_completion_recovery_policy_and_metrics_preserved=True,
        existing_decode_policy_preserved=True,profiles_unchanged=True,
        persistent_physical_uncertainty_barrier_preserved=True,gpu_qualified=False)
    with (out/'manifest.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(model,sha(out/'manifest.json'))

if __name__=='__main__':
    for model in ('7b','14b','32b'):build(model)
