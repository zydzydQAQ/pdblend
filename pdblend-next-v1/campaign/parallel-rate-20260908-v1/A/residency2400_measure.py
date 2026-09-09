"""Actual steady locked2400 residency after separately recorded clock settling."""
import asyncio
import json
import time

def require(ok,why):
    if not ok:raise ValueError(why)

async def observe(probe,clocks,hardware,out):
    raw=dict(schema='A2400-steady-residency-v1',complete=False,clock_command_mhz=2400,
        target_gpu=6,settle_observations=[],measurement_duration_declared_s=5,
        setup_and_clock_restoration_retained_in_whole_operation=True)
    try:
        raw['native_idle_before']=await probe.idle()
        await clocks.set([6],2400,verify_rise=False)
        until=time.monotonic()+4
        count=0
        while count<3:
            value=await asyncio.to_thread(hardware.current_freq,6)
            raw['settle_observations'].append(dict(at_s=time.time(),actual_sm_mhz=value))
            count=count+1 if abs(value-2400)<=15 else 0
            require(time.monotonic()<until,'actual idle locked2400 failed to settle')
            await asyncio.sleep(.02)
        raw['measurement_start_s']=time.time()
        await asyncio.sleep(5)
        raw['measurement_end_s']=time.time()
        raw['native_idle_after']=await probe.idle()
        raw['complete']=True
    except BaseException as exc:raw['error']=repr(exc)
    finally:(out/'residency.raw.json').write_text(json.dumps(raw,indent=2)+'\n')
    require(raw['complete'],'steady residency measurement failed')
    return raw

def derive(raw,power,clocks):
    from ecopadg.metrics import clip_power_window
    from ecopadg.measure.power import trapezoid_energy
    require(raw['complete'] and not raw.get('error'),'actual complete idle measurement required')
    start,end=raw['measurement_start_s'],raw['measurement_end_s']
    require(5<=end-start<5.5,'complete original5s idle interval required')
    values=clip_power_window(power,start,end,pad_s=0)
    actual=[v[6] for t,v in clocks if start<=t<=end]
    require(len(actual)>=3 and all(abs(f-2400)<=15 for f in actual),'full steady idle interval must be actual2400')
    require(all(len(v)==8 for t,v in values),'all-eight-card residency power required')
    target=trapezoid_energy([(t,[v[6]]) for t,v in values])
    return dict(valid=True,duration_s=end-start,target_gpu6_mean_w=target/(end-start),
        target_gpu6_max_w=max(v[6] for t,v in values),all_eight_gpu_j=trapezoid_energy(values),
        actual_sm_min_mhz=min(actual),actual_sm_max_mhz=max(actual),actual_clock_samples=len(actual),
        profile_publication_allowed=False)
