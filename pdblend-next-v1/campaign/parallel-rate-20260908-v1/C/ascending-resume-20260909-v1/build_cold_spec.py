"""Freeze the C7B cold-recovery input package locally; never starts hardware."""
import copy
import json
from pathlib import Path
import time
import cold_restore as c


def main():
    out = c.HERE
    assert not (out/'cold-spec.json').exists(), 'immutable new package required'
    r = c.ROOT
    release_ref = c.ref(r/'C/p4-completion-release/release.json')
    release = c.checked(release_ref)
    target = copy.deepcopy(c.checked(release['binding']))
    host = r/'hosts/7b-capacity-p12'
    manifest = c.read(host/'manifest.json')
    target.update(host_release=str(host), deadline_s=None, campaign_lifecycle='until_declared_complete_v1',
                  configs={k:v['path'] for k,v in release['configs']['fixed2'].items()},
                  original_engine_source_preserved=True, frequency_and_cancel_qualification_pending=True,
                  output=str(out/'future-results'), experiment_scope='C7B ascending continuation; exact P4 policy/profile with P12 OFF source')
    target['files'].update(release['files'])
    target['files'].update({str(host/name):digest for name,digest in manifest['files'].items()})
    target['files'][str(host/'manifest.json')] = c.sha(host/'manifest.json')
    for path, digest in target['files'].items():
        assert c.sha(path) == digest, path
    c.save(out/'target-binding.json', target)
    refs = dict(original_release=release_ref, target_binding=c.ref(out/'target-binding.json'),
                host_manifest=c.ref(host/'manifest.json'),
                historical_inventory=c.ref(r.parents[1]/'campaign/C7B-retained-peer-repair-v1/actual-001/spec.json'),
                off_source_equivalence=c.ref(r/'common/idle-domain-off-existing95-reuse-declaration-v7.json'),
                restore_executor=c.ref(r/'B/baseline-return-after-external-source-v1/execution.py'),
                docker_equivalence=c.ref(out/'docker_equivalence.py'))
    files=dict(target['files'])
    for ref in refs.values():
        files[ref['path']] = ref['sha256']
    for path in (out/'cold_restore.py', out/'build_cold_spec.py',
                 r/'common/execution-until-complete-v1/run.py', r/'common/execution-until-complete-v1/child.py'):
        files[str(path)] = c.sha(path)
    spec=dict(schema='C7B-ascending-cold-recovery-v1', node='C', hostname=c.HOSTNAME,
              created_s=time.time(), frequencies_mhz=[900,1500,2100,2520],
              gpu_work_started=False, qualification_granted=False, performance_dispatch_allowed=False,
              **refs, files=files)
    c.validate(spec)
    c.save(out/'cold-spec.json', spec)
    print(json.dumps(dict(spec=c.ref(out/'cold-spec.json'), files=len(files),
                          gpu_work_started=False, serving_measurements_started=False)))


if __name__ == '__main__':
    main()
