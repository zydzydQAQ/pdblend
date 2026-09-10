"""Fresh physical legacy-engine candidate qualification; no cost fitting or old-node proof."""
import argparse,asyncio,csv,json,os,signal,socket,subprocess,sys,time,uuid
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
U=ROOT/'A/uniform-rate-20260909-v1'
sys.path.insert(0,str(ROOT/'common/uniform-rate-20260909-v2'))
import support as p

def shapes(profile,tp):
 return sorted({(r['frequency_mhz'],r['input_tokens'],r['batch']) for r in profile['points'] if r['tp']==tp})

def progress(out,state):
    """Full request/SSE evidence is written once, after all owned streams stop."""
    compact={k:v for k,v in state.items() if k not in ('references','cases','idle_wakeup')}
    compact.update(completed_reference_count=len(state['references']),completed_case_count=len(state['cases']),
        completed_idle_wakeup_count=len(state['idle_wakeup']),full_request_evidence_pending=True)
    p.save(out/'status.json',compact)

def case_path(out,case):
    return out/'case-records'/('-'.join(str(case[k]) for k in ('instance_id','frequency','input_length','batch'))+'.json')

def case_record(case):
    result={k:v for k,v in case.items() if k!='requests'}
    result['request_ids']=[row['request_id'] for row in case['requests']]
    return result

def case_clocks(samples,gpus,target,rows):
    return [clock_window(samples,gpus,target,row['token_received_s'][0],row['token_received_s'][-1]) for row in rows]

async def joined_thread(function,*args):
    task=asyncio.create_task(asyncio.to_thread(function,*args));cancelled=None
    while not task.done():
        try:await asyncio.shield(task)
        except asyncio.CancelledError as exc:cancelled=exc
    result=task.result()
    if cancelled is not None:raise cancelled
    return result

def clock_window(samples,gpus,target,start,end):
 rows=[(t,f) for t,f in samples if start<=t<=end]
 p.need(end>start and len(rows)>=2,'loaded clock window missing')
 p.need(rows[0][0]-start<=.25 and end-rows[-1][0]<=.25 and max(b[0]-a[0] for a,b in zip(rows,rows[1:]))<=.25,'loaded clock continuity failed')
 extrema={str(g):[min(v[g] for _,v in rows),max(v[g] for _,v in rows)] for g in gpus}
 p.need(all(len(v)==8 and all(abs(v[g]-target)<=15 for g in gpus) for _,v in rows),'actual loaded clock outside declared +/-15 MHz: '+repr(extrema))
 return dict(gpus=gpus,target_mhz=target,start_s=start,end_s=end,samples=len(rows),actual_min_max_mhz=extrema)

def check(row,length,expected):
 p.need(row['success'] and row['http_status']==200 and row.get('done_marker') and not row.get('error'),'incomplete candidate response')
 p.need(row['prompt_token_ids']==([9707,1879,13]*(length//3+1))[:length] and row['output_token_ids']==expected and len(expected)==64,'numerical response differs')
 p.need(row['usage']['prompt_tokens']==length and row['usage']['completion_tokens']==64 and len(row['token_received_s'])==64 and row['token_received_s']==sorted(row['token_received_s']),'actual token work/timing differs')

def paths(binding):
 host=Path(binding['host_release']);sys.path[:0]=[str(host/'src'),str(host),'/root/workspace/pdblend/.runtime-deps']
 return p.load(ROOT/'common/execution-until-complete-v1/run.py','legacy_candidate_common')

async def execute(a,s):
 binding=p.checked(p.ref(a.binding));profile=p.checked(p.ref(a.profile));common=paths(binding)
 common.validate_binding(binding)
 p.need(socket.gethostname()==binding['hostname'],'wrong actual node')
 p.need(sorted({v[0] for i in binding['instances'] for v in shapes(profile,i['tp'])})==[900,1500,2100],'explicit supported domain required')
 streammod=p.load(U/'stream.py','legacy_candidate_original_sse')
 hooks=p.load(U/'meter-runtime/sampler_hooks.py','legacy_candidate_sampler_hooks')
 audit=p.load(ROOT.parent/'AC-baseline-binding-v2/gate_evidence.py','legacy_candidate_native_ack')
 from ecopadg.serving.campaign import node_lease
 from ecopadg.serving.backend import ClockOwner
 from ecopadg.measure.backends import PynvmlBackend
 from ecopadg.serving.measurement import save_raw,power_evidence
 from ecopadg.measure.power import trapezoid_energy
 from ecopadg.metrics import clip_power_window
 import aiohttp
 p.need('PDBLEND_NODE_LOCK_FD' not in os.environ,'new owner must acquire the node lease')
 with node_lease():
  a.out.mkdir(parents=True);s.update(node_lease_held=True,started_s=time.time(),binding=p.ref(a.binding),profile=p.ref(a.profile),pid=os.getpid())
  topology=subprocess.run(['nvidia-smi','topo','-m'],check=True,capture_output=True,text=True).stdout
  p.need(all('GPU'+str(i) in topology for i in range(8)),'actual eight-GPU interconnect incomplete')
  (a.out/'interconnect.txt').write_text(topology);s['topology']=p.ref(a.out/'interconnect.txt')
  progress(a.out,s)
  host=p.ref(Path(binding['host_release'])/'manifest.json');adapter=p.ref(U/'isolated-power/manifest.json')
  module=hooks.install(a.out/'isolated-samplers',host,adapter)
  sampler=owner=None;streams={};refs={};verified=False
  async with aiohttp.ClientSession(trust_env=False,connector=aiohttp.TCPConnector(limit=0)) as session:
   try:
    p.save(a.out/'identity.before.json',await common.identity(session,binding));verified=True
    hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
    sampler=module.IsolatedPowerSampler(range(8),interval=.02,backend=hardware,sample_clocks=True);sampler.start();await asyncio.to_thread(sampler.wait_ready)
    owner=await asyncio.to_thread(ClockOwner,hardware,tuple(range(8)),max_frequency=2100);s['measurement_start_s']=time.time()
    await owner.set(range(8),2100,verify_rise=False)
    for i in binding['instances']:
     await common.resume(session,dict(i,role='mixed'))
     stream=streammod.NaturalStream();stream.session=session;stream.args=SimpleNamespace(port=i['port'],seed=0);stream.issued=set()
     stream.stream_journal=(a.out/(i['id']+'.stream.jsonl')).open('x');stream.result_journal=(a.out/(i['id']+'.requests.jsonl')).open('x');streams[i['id']]=stream;refs[i['id']]={}
    async def reference(i):
     for length in sorted({v[1] for v in shapes(profile,i['tp'])}):
      row=await streams[i['id']].request(length,64);check(row,length,row['output_token_ids']);refs[i['id']][length]=row['output_token_ids'];s['references'].append(dict(instance_id=i['id'],input_length=length,request=row))
      await common.wait_idle(session,i)
    reference_tasks=[asyncio.create_task(reference(i)) for i in binding['instances']]
    try:await asyncio.gather(*reference_tasks)
    finally:
     for task in reference_tasks:
      if not task.done():task.cancel()
     await asyncio.gather(*reference_tasks,return_exceptions=True)
    progress(a.out,s)
    for target in (900,1500,2100):
     await owner.set(range(8),target,verify_rise=False)
     async def instance_cases(i):
      stream=streams[i['id']]
      warmup=await stream.request(128,64);check(warmup,128,refs[i['id']][128]);await common.wait_idle(session,i)
      for frequency,length,batch in [v for v in shapes(profile,i['tp']) if v[0]==target]:
       current=dict(instance_id=i['id'],tp=i['tp'],gpus=i['gpus'],frequency=frequency,input_length=length,batch=batch)
       s['active_cases'][i['id']]=current;progress(a.out,s)
       rows=await asyncio.gather(*(stream.request(length,64) for _ in range(batch)))
       case=dict(current,requests=rows,native_after=await common.wait_idle(session,i));s['cases'].append(case);progress(a.out,s)
       for row in rows:check(row,length,refs[i['id']][length])
       case['loaded_clocks']=await joined_thread(case_clocks,sampler.frequency_samples,i['gpus'],target,rows)
       await joined_thread(p.save,case_path(a.out,case),case_record(case))
       s['active_cases'].pop(i['id']);progress(a.out,s)
     work=[asyncio.create_task(instance_cases(i)) for i in binding['instances']]
     try:await asyncio.gather(*work)
     finally:
      for task in work:
       if not task.done():task.cancel()
      await asyncio.gather(*work,return_exceptions=True)
     # Every native engine is explicitly observed empty, then naturally idle and woken.
     start=time.time();polls=[]
     while time.time()-start<3:
      values=await asyncio.gather(*(common.http(session,i,'/runtime') for i in binding['instances']))
      for i,v in zip(binding['instances'],values):audit.ack(v,i)
      polls.append(dict(at_s=time.time(),native=values));await asyncio.sleep(.1)
     rows=await asyncio.gather(*(streams[i['id']].request(128,64) for i in binding['instances']))
     for i,row in zip(binding['instances'],rows):check(row,128,refs[i['id']][128]);clock_window(sampler.frequency_samples,i['gpus'],target,row['token_received_s'][0],row['token_received_s'][-1])
     s['idle_wakeup'].append(dict(frequency=target,start_s=start,polls=polls,requests=rows));progress(a.out,s)
    if a.retain_weights:
     i=binding['instances'][0];raw=await common.wait_idle(session,i);audit.ack(raw,i)
     transaction='uniform_'+uuid.uuid4().hex;payload=dict(expected_generation=raw['generation'],transaction=transaction)
     cache_root=Path(p.read(i['engine_config'])['weight_cache_root']);p.need(not (cache_root/transaction).exists(),'retained weight output must be fresh')
     s['retained_weights']=dict(instance_id=i['id'],native_before=raw,payload=payload,started_s=time.time());progress(a.out,s)
     async with session.post(i['url']+'/retain-weights',json=payload,timeout=aiohttp.ClientTimeout(total=620)) as response:
      body=await response.json();s['retained_weights'].update(http_status=response.status,response=body,finished_s=time.time());progress(a.out,s)
      p.need(response.status==200 and body['generation']==raw['generation'] and body['accepting'] is False,'actual native weight retention failed')
     cache=cache_root/transaction;p.need(body['retained_weights']==str(cache),'native retained cache path differs')
     manifest=p.ref(cache/'manifest.json');value=p.checked(manifest);p.need(value==body['manifest'] and value['complete'] and value['tp']==i['tp'],'actual cache manifest differs')
     rank_files={}
     for rank in value['ranks']:
      path=cache/rank['file'];p.need(path.is_file() and path.parent==cache and p.sha(path)==rank['sha256'],'actual retained rank bytes differ');rank_files[str(path)]=dict(sha256=rank['sha256'],stat=common.stat_identity(path))
     s['retained_weights'].update(manifest=manifest,rank_files=rank_files)
     resumed=await common.resume(session,dict(i,role='mixed'));s['retained_weights']['resumed']=resumed;progress(a.out,s)
    s['passed']=True
   finally:
    for i in binding['instances']:
     stream=streams.get(i['id'])
     if stream:
      for rid in stream.issued:
       try:await common.http(session,i,'/cancel',dict(request_id=rid))
       except BaseException as exc:s['cleanup_errors'].append(repr(exc))
      stream.stream_journal.close();stream.result_journal.close()
    if verified:
     restored=await asyncio.gather(*(common.restore(session,i) for i in binding['instances']),return_exceptions=True)
     s['restoration']={i['id']:dict(error=repr(v)) if isinstance(v,BaseException) else v for i,v in zip(binding['instances'],restored)}
    if owner:
     try:await owner.close();s['clock_restore_complete']=True
     except BaseException as exc:s['cleanup_errors'].append(repr(exc))
    s['measurement_end_s']=time.time()
    if sampler:
     await asyncio.sleep(.12);await asyncio.to_thread(sampler.stop)
     power=a.out/'power';power.mkdir();save_raw(power,[],sampler.samples,sampler.utilization_samples,power_source=sampler.power_source,power_metadata=sampler.power_metadata)
     with (power/'clocks.csv').open('x',newline='') as handle:
      writer=csv.writer(handle);writer.writerow(['t_s']+[f'gpu{i}_sm_mhz' for i in range(8)]);writer.writerows([t,*values] for t,values in sampler.frequency_samples)
     s['power_evidence']=power_evidence(sampler.samples,sampler.power_source,sampler.power_metadata);s['sampling_error']=sampler.error
     s['full_operation_energy_j']=trapezoid_energy(clip_power_window(sampler.samples,s['measurement_start_s'],s['measurement_end_s'],pad_s=0)) if s.get('measurement_start_s') else None
     s['isolated_artifacts']=hooks.completed_artifacts(hooks.directories(a.out/'isolated-samplers'),host,adapter)
    if verified:p.save(a.out/'identity.after.json',await common.identity(session,binding))
    s['measurement_valid']=bool(not s['cleanup_errors'] and s.get('clock_restore_complete') and not s.get('sampling_error') and s.get('full_operation_energy_j',0)>0 and s.get('power_evidence',{}).get('power_source_verified') and all(v.get('complete') for v in s.get('restoration',{}).values()))
    s['passed']=s['passed'] and s['measurement_valid'];progress(a.out,s)
  s['node_lease_held']=False

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--profile',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--retain-weights',action='store_true');parser.add_argument('--run',action='store_true');a=parser.parse_args()
 p.need(a.run and not a.out.exists(),'fresh explicit qualification only')
 s=dict(schema='fresh-legacy-frequency-qualification-v1',passed=False,complete=False,node_lease_held=False,references=[],cases=[],active_cases={},idle_wakeup=[],cleanup_errors=[],cost_values_recalibrated=False)
 async def controlled():
  task=asyncio.current_task()
  for sig in (signal.SIGINT,signal.SIGTERM):asyncio.get_running_loop().add_signal_handler(sig,task.cancel)
  await execute(a,s)
 try:asyncio.run(controlled());s['complete']=True
 except BaseException as exc:s.update(passed=False,error=repr(exc));raise
 finally:
  if a.out.exists():s.update(finished_s=time.time(),node_lease_held=False);p.save(a.out/'status.json',s)
 p.need(s['passed'],'fresh frequency candidate gate failed; preserve evidence')
if __name__=='__main__':main()
