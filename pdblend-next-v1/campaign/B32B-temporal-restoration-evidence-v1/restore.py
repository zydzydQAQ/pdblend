"""Finish native/prefix evidence only. Never launch a model or send inference."""
import argparse,asyncio,copy,csv,hashlib,json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;C=ROOT.parent;P=C/'B32B-temporal-observation-execution-v2-r4'
sys.path.insert(0,str(P));import common as c,run as r,archive
EXPECTED='cc45b0ca028f81a16d26e9be73e19bf069076f7c8970e5ce06295ba8ce77077d'
ATTEMPT=C/'B32B-temporal-observation-attempt-001';SPEC_SHA='79cae32269a0829be8221c17dd0bc69eb5e7a2ce60d2a415a032491aec01d646'

def preserve(before,after):
    for k in ('Id','Image','Config','HostConfig','Path','Args'):c.require(before[k]==after[k],'changed original '+k)
    def mounts(value):
        rows=value['Mounts'];c.require(len({x['Destination'] for x in rows})==len(rows),'duplicate mount destination')
        return sorted(json.dumps(x,sort_keys=True) for x in rows)
    c.require(mounts(before)==mounts(after),'changed full mount dictionaries')
    c.require(after['State']['Running'] is True and after['State']['Pid']>0 and after['State']['Pid']!=before['State']['Pid'] and after['State']['StartedAt']!=before['State']['StartedAt'],'not a fresh original process')

async def finish():
    import aiohttp
    from ecopadg.measure.backends import PynvmlBackend
    from ecopadg.measure.power import PowerSampler,trapezoid_energy
    from ecopadg.serving.measurement import save_raw,power_evidence
    from ecopadg.metrics import clip_power_window
    c.require(not (ROOT/'results').exists(),'one recovery evidence attempt only')
    op=r.Operation(c.read(ATTEMPT/'spec.json'));op.out=ROOT/'results';op.out.mkdir();op.deadline=min(c.DEADLINE,time.time()+90);op.restoring=True
    op.state=dict(schema=1,pid=os.getpid(),complete=False,errors=[],commands=[],performance_evidence=False,inference_requests_sent=0,original_failed_status=str(ATTEMPT/'results/status.json'),original_failed_status_sha256=c.sha(ATTEMPT/'results/status.json'));op.save()
    started=None;sampler=None;fresh=None
    async with aiohttp.ClientSession(trust_env=False) as session:
      try:
        oldstatus=c.read(ATTEMPT/'results/status.json');c.require(oldstatus['complete'] and oldstatus['process_terminal']['diagnostic_gpu_workers_gone'] and oldstatus['process_terminal']['http_child_exited'],'diagnostic workers not terminal')
        for pid in (oldstatus['pid'],oldstatus['child_pid']):c.require(not c.pid_live(pid),'diagnostic HTTP/owner process remains')
        diag=await op.inspected(['pdb-v2-temporalobsb1']);c.require(not diag[0]['State']['Running'] and diag[0]['State']['Pid']==0,'diagnostic container remains')
        original=c.read(op.spec['previous_binding']);before=c.read(ATTEMPT/'results/original-identity.before.json');inspections=await op.inspected([i['container']['name'] for i in original['instances']]);actual=[]
        for i,b,a in zip(original['instances'],before,inspections):
            preserve(b['container'],a);item=await op.source_identity(session,i,a);c.require(item['provenance']==b['provenance'],'original source/model provenance changed');raw=await op.engine.http(session,i,'/runtime',timeout=op.remaining(3));op.Checks.check_ack(raw,i);c.require(op.Checks.is_idle(raw) and raw['accepting'] is True,'original process not actually idle/accepting');item['runtime']=raw;actual.append(item)
        c.write(op.out/'identity.before.json',actual)
        fresh=copy.deepcopy(original);fresh.update(configs={},output=str(ROOT/'future-fresh-gate'),output_correctness_verified=False,formal_eligible=False,correctness_gate_required_before_performance=True,old_correctness_is_historical_only=True,restoration_evidence=str(op.out/'status.json'))
        fresh.pop('mechanism_proof',None);fresh.pop('correctness_evidence',None)
        for i,a in zip(fresh['instances'],actual):i['role']='mixed';i['container']['StartedAt']=a['container']['State']['StartedAt'];i['provenance']=a['provenance']
        op.hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant');sampler=PowerSampler(range(8),interval=.02,backend=op.hardware,sample_clocks=True);sampler.start();await op.deploy.await_power_ready(sampler,power_evidence);started=time.time()
        op.state['native']=await op.native(session,fresh,'all4-native');op.state['identity_after']=await op.engine.identity(session,fresh)
        c.write(op.out/'identity.after.json',op.state['identity_after'])
        op.state['runtime_prefix_after_restore']=archive.verify_prefix_after_restore(fresh,oldstatus['runtime_stopped'],op.state['native'],op.deadline)
        c.write(op.out/'runtime-prefix-after-restore.json',op.state['runtime_prefix_after_restore']);c.write(op.out/'restored-bootstrap.binding.json',fresh)
        op.state['restored_binding']=str(op.out/'restored-bootstrap.binding.json');op.state['restored_binding_sha256']=c.sha(op.out/'restored-bootstrap.binding.json');op.state['all_original_restored']=True
      except BaseException as exc:op.state['errors'].append(repr(exc))
      finally:
        ended=time.time()
        if sampler is not None:
          try:
            await asyncio.sleep(.15);await asyncio.to_thread(sampler.stop);out=op.out/'power';out.mkdir();save_raw(out,[],sampler.samples,sampler.utilization_samples,power_source=sampler.power_source,power_metadata=sampler.power_metadata)
            with (out/'clocks.csv').open('x',newline='') as f:
                w=csv.writer(f);w.writerow(['t_s']+[f'gpu{i}_sm_mhz' for i in range(8)]);w.writerows([t,*v] for t,v in sampler.frequency_samples)
            r.sample_window(sampler.samples,sampler.frequency_samples,started,ended);op.state['full_operation_energy_j']=trapezoid_energy(clip_power_window(sampler.samples,started,ended,pad_s=0));op.state['power_evidence']=power_evidence(sampler.samples,sampler.power_source,sampler.power_metadata);op.state['sampling_error']=sampler.error
          except BaseException as exc:op.state['errors'].append('power '+repr(exc))
        op.state.update(complete=True,finished_s=time.time(),operation_start_s=started,operation_end_s=ended,measurement_valid=bool(started and op.state.get('all_original_restored') and not op.state['errors'] and not op.state.get('sampling_error')),original_failure_unchanged=c.sha(ATTEMPT/'results/status.json')==op.state['original_failed_status_sha256'],clocks_changed=False);op.save()
    return op.state

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');args=p.parse_args();c.require(c.sha(P/'manifest.json')==EXPECTED,'frozen parent changed');c.package_check();c.require(c.sha(ATTEMPT/'spec.json')==SPEC_SHA,'actual attempted spec differs')
    if not args.run:print(json.dumps(dict(cpu_only=True,inference=False,container_restart=False)));return
    c.require(not os.environ.get('PDBLEND_NODE_LOCK_FD'),'fresh lease required')
    sys.path[:0]=[str(c.HOST/'src'),str(c.HOST),'/root/workspace/pdblend/.runtime-deps'];from ecopadg.serving.campaign import node_lease
    with node_lease():state=asyncio.run(finish())
    print(json.dumps({k:state.get(k) for k in ('complete','all_original_restored','measurement_valid','full_operation_energy_j','errors')}));c.require(state['measurement_valid'],'native recovery evidence failed')
if __name__=='__main__':main()
