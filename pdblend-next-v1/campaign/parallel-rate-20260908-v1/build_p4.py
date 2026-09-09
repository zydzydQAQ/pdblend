"""Retain queued work after an unconfirmed admission that changed no clocks."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def once(s,a,b):
    if s.count(a)!=1:raise ValueError((a,s.count(a)))
    return s.replace(a,b)

def backend(source):
    source=once(source,'    async def set(self, gpus, frequency,*,verify_rise=True,bootstrap=None):',
        '    async def set(self, gpus, frequency,*,verify_rise=True,bootstrap=None,admission_observation_deferral=False):')
    source=once(source,'                if pending and time.monotonic()>=deadline:\n', '''                if pending and time.monotonic()>=deadline:
                    # The admission has not been forwarded. If the complete
                    # clock transaction changed nothing, rollback its ledger
                    # reservation and retry under the original hard deadline.
                    # No uncertain observation authorizes work or a fallback.
                    if (admission_observation_deferral and self.write_guard is not None
                            and bootstrap is None and not self.transaction_writes
                            and not self.pending_physical_commands
                            and not self.physical_command_uncertainty
                            and all(self.applied.get(g)==frequency for g in gpus)
                            and not any(g in self.deferred for g in gpus)):
                        event=dict(kind='frequency_observation_deferred',at_s=time.time(),
                            gpus=list(gpus),pending_observations=list(pending),
                            rejected_target_mhz=frequency,
                            retained_commands={g:self.applied.get(g) for g in gpus},
                            physical_write_attempts=[],measurement_confirmation=False,
                            safely_replannable=True,original_deadline_retained=True,
                            reason='admission observation did not confirm unchanged full-TP command')
                        self.coverage_limits=(self.coverage_limits+[event])[-256:]
                        self.clock_event(event)
                        raise ClockEligibilityExpired('unconfirmed unchanged admission frequency; rollback reservation and retain original deadline')
''')
    source=once(source,'                    await self.clocks.set(self.instances[action.instance_id]["gpus"],desired)',
        '                    await self.clocks.set(self.instances[action.instance_id]["gpus"],desired,\n                        admission_observation_deferral=bool(plan.routes) and\n                            getattr(self,\'unconfirmed_retained_admission_deferral\',False))')
    ast.parse(source);return source

def runtime(source):
    anchor="            self.backend.exact_frequency_confirmation=True\n"
    source=once(source,anchor,anchor+"            self.backend.unconfirmed_retained_admission_deferral=(self.config.get('unconfirmed_retained_admission_deferral_v1') is True)\n")
    ast.parse(source);return source

def build():
    shared=None
    for m in ('7b','14b','32b'):
        parent=ROOT/'hosts'/f'{m}-fixed-p3';out=ROOT/'hosts'/f'{m}-fixed-p4'
        if out.exists():raise FileExistsError(out)
        manifest=json.loads((parent/'manifest.json').read_text())
        changed={'src/ecopadg/serving/backend.py':backend((parent/'src/ecopadg/serving/backend.py').read_text()),
            'src/ecopadg/serving/runtime.py':runtime((parent/'src/ecopadg/serving/runtime.py').read_text())}
        for n,h in manifest['files'].items():
            if sha(parent/n)!=h:raise ValueError('parent changed '+n)
            dst=out/n;dst.parent.mkdir(parents=True,exist_ok=True)
            if n in changed:dst.write_text(changed[n])
            else:shutil.copyfile(parent/n,dst)
        files={n:sha(out/n) for n in sorted(manifest['files'])}
        if shared and files!=shared:raise ValueError('cross-model source mismatch')
        shared=files
        result=dict(schema=4,model=m,implementation_series='parallel-p4',created_s=time.time(),files=files,
            frozen_references={str(p):sha(p) for p in (parent/'manifest.json',Path(__file__))},
            parent_manifest=dict(path=str(parent/'manifest.json'),sha256=sha(parent/'manifest.json')),
            common_controller_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
            identical_source_files_all_models=True,feature='unconfirmed_retained_admission_deferral_v1',
            default_enabled=False,admission_only=True,physical_write_uncertainty_remains_terminal=True,
            tolerance_settling_timeout_profiles_slo_and_hard_deadline_unchanged=True,
            changed_parent_files=sorted(changed),gpu_qualified=False)
        with (out/'manifest.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(m,sha(out/'manifest.json'),result['common_controller_sha256'])

if __name__=='__main__':build()
