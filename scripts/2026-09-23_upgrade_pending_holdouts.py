#!/usr/bin/env python3
"""Upgrade only unstarted TP4 holdouts to bounded shared-prefill windows."""
import importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pdblend.experimentation.lease import GPULeaseQueue


def main():
    root=ROOT/'results/2026-09-22/three-model';state=json.loads((root/'queue.json').read_text())
    old=[j for j in state['jobs'].values() if j['status']=='queued' and j['job_id'].startswith('holdout-')
         and j['payload'].get('tp')==4 and j['job_id'].endswith('-argvfix')]
    if len(old)!=3:raise ValueError('expected exactly three unstarted original TP4 holdouts')
    spec=importlib.util.spec_from_file_location('freeze',ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    source,sha=helper.freeze_source(ROOT/'src',root/'calibration-sources')
    ids={j['job_id']:j['job_id'].replace('-argvfix','-shared-'+sha[:12]) for j in old}
    jobs=[]
    for job in old:
        payload=dict(job['payload']);argv=list(payload['argv']);name=ids[job['job_id']]
        argv[argv.index('--name')+1]=name
        for i,value in enumerate(argv):
            if value.endswith(':/opt/pdblend-src:ro'):argv[i]=f'{source}:/opt/pdblend-src:ro'
            elif value.startswith('PDBLEND_SOURCE_SHA256='):argv[i]='PDBLEND_SOURCE_SHA256='+sha
            elif value.endswith(':/wave:rw'):
                before=Path(value[:-len(':/wave:rw')]);wave=json.loads((before/'wave.json').read_text())
                wave_id='holdout-shared-'+sha[:16]+'-'+before.name.rsplit('-',1)[1]
                after=root/'calibration-waves'/wave_id;after.mkdir(parents=True,exist_ok=True)
                wave['cohort_id']=wave_id;(after/'wave.json').write_text(json.dumps(wave,indent=2)+'\n')
                argv[i]=f'{after}:/wave:rw'
        candidate=Path(payload['candidate_dir']);manifest=json.loads((candidate/'manifest.json').read_text())
        raw=manifest['training_raw'];index=argv.index(payload['image_digest'])
        argv[index:index]=['-v',f'{raw}:{raw}:ro']
        argv+=['--shared-prefill-windows']
        payload.update(argv=argv,source_sha256=sha,container_name=name,shared_prefill_windows=True,
                       timeout_s=7200,depends_on=[ids.get(x,x) for x in payload['depends_on']])
        jobs.append(dict(job_id=name,payload=payload,priority=job['priority'],max_attempts=1))
    output=root/f'calibration-shared-window-spec-{sha[:16]}.json';output.write_text(json.dumps(jobs,indent=2)+'\n')
    queue=GPULeaseQueue(root/'queue.json')
    # Original immutable specs remain visible in history with explicit reason.
    for job in old:queue.block(job['job_id'],reason='Replaced before execution with bounded shared-prefill windows; same frozen fit and holdout plan')
    for job in jobs:queue.enqueue(**job)
    print(json.dumps(dict(source_sha256=sha,spec=str(output),jobs=[j['job_id'] for j in jobs]),indent=2))


if __name__=='__main__':main()
