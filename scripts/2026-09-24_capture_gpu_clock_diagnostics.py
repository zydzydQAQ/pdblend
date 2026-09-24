#!/usr/bin/env python3
"""Read-only NVML diagnostics alongside an already authorized lease workload."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--job',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-seconds',type=float,default=3600.)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    state=json.loads(args.queue.read_text());job=state['jobs'][args.job]
    leases=[l for l in state['leases'].values() if l['job_id']==args.job and l['status']=='active']
    if job['status']!='running' or len(leases)!=1:raise ValueError('one current owned lease required')
    lease=leases[0];uuids=lease['gpu_uuids']
    import pynvml as nv
    nv.nvmlInit();started=time.time();samples=0;error=None
    names={getattr(nv,name):name.removeprefix('nvmlClocksThrottleReason') for name in dir(nv)
           if name.startswith('nvmlClocksThrottleReason') and name not in ('nvmlClocksThrottleReasonAll','nvmlClocksThrottleReasonNone')}
    manifest=dict(schema='readonly-lease-clock-diagnostics/v1',job_id=args.job,lease_id=lease['lease_id'],
        attempt_dir=lease['attempt_dir'],gpu_uuids=uuids,job_payload=job['payload'],
        collector=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),started_s=started,
        interval_s=1.,gpu_workload_started=False,gpu_clock_changed=False,formal_energy_source=False,
        scope='concurrent_workload_diagnostics_not_reconstruction_of_earlier_EcoServe_window')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    try:
        handles=[nv.nvmlDeviceGetHandleByUUID(u) for u in uuids]
        with (out/'samples.jsonl').open('x') as stream:
            while time.time()-started<args.max_seconds:
                current=json.loads(args.queue.read_text())
                if current['jobs'][args.job]['status']!='running' or current['leases'][lease['lease_id']]['status']!='active':break
                row=dict(started_s=time.time(),devices=[])
                for uuid,handle in zip(uuids,handles):
                    device=dict(gpu_uuid=uuid,errors={})
                    queries={
                        'sm_clock_mhz':lambda:nv.nvmlDeviceGetClockInfo(handle,nv.NVML_CLOCK_SM),
                        'temperature_c':lambda:nv.nvmlDeviceGetTemperature(handle,nv.NVML_TEMPERATURE_GPU),
                        'power_limit_w':lambda:nv.nvmlDeviceGetPowerManagementLimit(handle)/1000.,
                        'enforced_power_limit_w':lambda:nv.nvmlDeviceGetEnforcedPowerLimit(handle)/1000.,
                        'device_power_usage_w_diagnostic':lambda:nv.nvmlDeviceGetPowerUsage(handle)/1000.,
                        'throttle_reasons_mask':lambda:nv.nvmlDeviceGetCurrentClocksThrottleReasons(handle),
                        'gpu_util_pct':lambda:nv.nvmlDeviceGetUtilizationRates(handle).gpu}
                    for key,fn in queries.items():
                        try:device[key]=fn()
                        except Exception as exc:device['errors'][key]=repr(exc)
                    if 'throttle_reasons_mask' in device:
                        device['throttle_reasons']=[name for bit,name in names.items() if bit&device['throttle_reasons_mask']]
                    row['devices'].append(device)
                row['finished_s']=time.time();stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush();samples+=1
                time.sleep(max(0.,1.-(time.time()-row['started_s'])))
    except Exception as exc:error=repr(exc)
    finally:
        nv.nvmlShutdown()
        report=dict(schema='readonly-lease-clock-diagnostics-completion/v1',started_s=started,
            finished_s=time.time(),samples=samples,error=error,gpu_workload_started=False,
            manifest_sha256=sha(out/'manifest.json'),samples_sha256=sha(out/'samples.jsonl'))
        (out/'completion.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
