"""Exactly two paired LongBench developer cells. No baseline or transport changes."""
import asyncio
import csv
import inspect
import json
import signal
import time
from pathlib import Path

import aiohttp
from ecopadg.serving.campaign import node_lease
from ecopadg.serving.runtime import Controller
from ecopadg.serving.backend import ClockOwner
from ecopadg.serving.measurement import power_evidence,save_raw
from ecopadg.measure.backends import PynvmlBackend
from ecopadg.measure.power import PowerSampler,trapezoid_energy
from ecopadg.metrics import clip_power_window
from safety import Safety,ROOT,OLD,RELEASE,PORTS,IDS,require,sha,write,check_ack,is_idle

HOST=ROOT.parents[1]/'releases/io-v1.2.1-runtime'
TRACE=ROOT.parent/'B32B-io-v1/longbench.trace.json'


def verify_policy(config,tokens):
    require(config['strategy']=='pdblend-joint' and config['output_prior']==211 and config['allow_pd'] is False
        and config['dynamic_pools'] is False and config['slow_topology'] is False and config['park_idle'] is False
        and config['evaluation_protocol']=='evaluation-v3' and config['slo_attainment_target']==.9
        and config['slo_ttft_s']==5. and config['slo_tpot_s']==.1 and config['request_timeout_s']==120,
        'original B2 policy/work/SLO differs')
    require(config['node_gpus']==list(range(8)) and [x['gpus'] for x in config['instances']]==[[0,1],[2,3]],'GPU topology differs')
    require(config['scheduler_budget_ablation']==dict(schema_version=1,max_num_batched_tokens=tokens,max_num_seqs=32),'budget differs')
    require(not config.get('topology') and not config.get('pd_topology') and not config.get('output_limit_aware_prediction',False),
        'unrelated dynamic topology/output predictor enabled')


class Pair(Safety):
    def __init__(self):
        require(not (ROOT/'status.json').exists(),'existing campaign evidence retained')
        self.log=(ROOT/'outer-http.jsonl').open('x',buffering=1)
        self.deadline=None;self.child=None;self.child_log=None;self.verified=False;self.hardware=None
        self.state=dict(complete=False,phase='preflight',started_s=time.time(),cells={},
            scope='two LongBench PDB-only fixed-budget developer ablation',
            full_runtime_gate='failed',temporal_gate='unfixed; no temporal execution allowed',
            budget_profile_certified=False,budgets=[8192,2048],rate=.125,seed=11,requests=64,
            baseline_execution=False,automatic_extra_cells=False)
        self.save()

    def save(self):write(ROOT/'status.json',self.state)

    def verify_frozen(self,label):
        m=json.loads((ROOT/'manifest.json').read_text())
        for p,h in m['files'].items():require(sha(ROOT/p)==h,'candidate changed: '+p)
        for p,h in m['frozen_inputs'].items():require(sha(p)==h,'historical/frozen input changed: '+p)
        for release in (HOST,RELEASE):
            for p,h in json.loads((release/'manifest.json').read_text())['files'].items():require(sha(release/p)==h,'frozen source changed: '+p)
        require(Path(inspect.getfile(Controller)).resolve()==HOST/'src/ecopadg/serving/runtime.py','wrong host import')
        gate=json.loads((ROOT.parent/'B32B-continuous-budget-validation-v1/status.json').read_text())
        require(gate.get('passed') and gate.get('measurement_valid') and all(x['exact_equal'] for x in gate['cross_replica_outputs']),
            'continuous correctness gate not passed')
        require(json.loads((OLD/'validation-status.json').read_text())['passed'] is False,'original temporal failure must remain recorded')
        configs=[json.loads((ROOT/f'budget{n}.config.json').read_text()) for n in (8192,2048)]
        for c,n in zip(configs,(8192,2048)):verify_policy(c,n)
        a,b=[dict(c) for c in configs]
        a.pop('scheduler_budget_ablation');b.pop('scheduler_budget_ablation');require(a==b,'paired config has non-budget difference')
        trace=json.loads(TRACE.read_text());require(len(trace['requests'])==64 and trace['seed']==11 and trace['rate']==.125,'wrong trace workload')
        sources={str(p):sha(p) for base in (Path('/root/workspace/pdblend/src'),Path('/root/workspace/pdblend/benchmarks'))
            for p in base.rglob('*.py') if '__pycache__' not in str(p)}
        histories={str(p):sha(p) for name in ('B32B-io-v1','B32B-idle-tail-v2','B32B-capacity3-v2','B32B-capacity2-v1')
            for p in (ROOT.parent/name).rglob('*') if p.is_file() and '__pycache__' not in str(p)
            and p.suffix in ('.json','.jsonl','.csv','.py','.md','.txt','.log')}
        if label=='before':self.old_source_hashes=sources;self.old_result_hashes=histories
        else:
            require(sources==self.old_source_hashes,'old B source changed')
            require(histories==self.old_result_hashes,'historical B results changed')
        write(ROOT/('historical-preservation.'+label+'.json'),dict(old_source_files=sources,
            historical_result_files=histories,checked_frozen_inputs=len(m['frozen_inputs']),baseline_files=91,all_unchanged=True))

    async def set_budget(self,tokens):
        rows={}
        for port in PORTS:
            r=await self.settled_idle(port)
            target=r['generation']+1
            payload=dict(generation=target,role='mixed',mode='continuous',admit_prefill=True,admit_decode=True,
                scheduler_budget=dict(schema_version=1,max_num_batched_tokens=tokens,max_num_seqs=32))
            response=await self.http(port,'/control',payload,label='fixed-cell-budget',timeout=25)
            after=await self.runtime(port,'budget-owner-cache-ack');check_ack(after)
            require(after['generation']==target and response['generation']==target and after.get('accepting') is True
                and after['scheduler_budget_effective']==dict(max_num_batched_tokens=tokens,max_num_seqs=32),'budget not actually ACKed')
            rows[str(port)]=dict(before=r,command=payload,ack=response,after=after)
        return rows

    def poll_events(self,tokens):
        for name in IDS:
            path=OLD/'runtime'/(name+'.control.events.jsonl')
            with path.open('rb') as f:f.seek(self.positions[name]);raw=f.read();self.positions[name]+=len(raw)
            data=self.tails[name]+raw;parts=data.split(b'\n');self.tails[name]=parts.pop()
            for line in parts:
                if not line:continue
                event=json.loads(line)
                self.owned[name].update(event.get('request_ids',[]))
                if event.get('tokens',0):
                    self.counts[name]+=1
                    require(event.get('mode')=='continuous' and event.get('role')=='mixed','forbidden temporal/non-mixed model execution')
                    require(event['tokens']<=tokens,'scheduled batch exceeds fixed token budget')

    async def stop_child(self):
        if self.child is None or self.child.returncode is not None:return
        self.child.terminate()
        try:await asyncio.wait_for(self.child.wait(),self.remaining(3))
        except asyncio.TimeoutError:self.child.kill();await asyncio.wait_for(self.child.wait(),self.remaining(2))

    def add_journal_ids(self,out):
        path=out/'control.jsonl'
        if not path.exists():return
        for line in path.read_bytes().splitlines(keepends=True):
            if not line.endswith(b'\n'):continue
            row=json.loads(line)
            if row.get('kind')=='admission':
                rid=row['request_id']
                require(isinstance(rid,str) and len(rid)==32 and all(c in '0123456789abcdef' for c in rid),'unexpected request ID')
                # A single mixed route forwards this UUID; cancelling its tombstone on the peer is harmless.
                for name in IDS:self.owned[name].add(rid)

    async def cleanup_cell(self,out,tokens):
        began=time.monotonic();self.deadline=began+80;self.state['cleanup']={}
        errors=[]
        try:
            await self.stop_child()
            try:self.poll_events(tokens);self.add_journal_ids(out)
            except BaseException as exc:errors.append('request/event ownership: '+repr(exc))
            if self.verified:
                async def cancel(port,rid):
                    try:await self.http(port,'/cancel',dict(request_id=rid),label='owned-cell-cleanup',timeout=8)
                    except BaseException as exc:errors.append('cancel '+rid+': '+repr(exc))
                await asyncio.gather(*(cancel(port,rid) for port,name in zip(PORTS,IDS) for rid in self.owned[name]))
                results=await asyncio.gather(*(self.restore_one(p) for p in PORTS),return_exceptions=True)
                errors.extend(repr(r) for r in results if isinstance(r,BaseException))
                if not errors:
                    after=await asyncio.wait_for(self.identity('after-'+str(tokens)),self.remaining(12))
                    for a,b in zip(self.identities,after):
                        require(a['container']['Id']==b['container']['Id'] and a['container']['State']['StartedAt']==b['container']['State']['StartedAt']
                            and a['provenance']==b['provenance'],'resident source/process changed')
        except BaseException as exc:errors.append(repr(exc))
        finally:
            self.deadline=began+90
            if self.verified:
                try:
                    # The host owned all eight clock locks. Reacquire only after its process exited.
                    clocks=await asyncio.to_thread(ClockOwner,self.hardware,tuple(range(8)))
                    await asyncio.wait_for(clocks.close(),self.remaining(10))
                except BaseException as exc:errors.append('clock release: '+repr(exc))
            self.deadline=None
        result=dict(complete=not errors,errors=errors,elapsed_s=time.monotonic()-began,native=self.state['cleanup'])
        write(self.operation/'outer-cleanup.json',result);return result

    async def cell(self,tokens):
        self.state.update(phase='running',current_budget=tokens);self.save()
        out=ROOT/f'cell-longbench-budget{tokens}';self.operation=ROOT/f'budget{tokens}-operation';self.operation.mkdir()
        self.verified=False;self.child=None;self.child_log=None
        self.initial={name:(OLD/'runtime'/(name+'.control.events.jsonl')).stat().st_size for name in IDS}
        self.positions=dict(self.initial);self.tails={name:b'' for name in IDS};self.owned={name:set() for name in IDS};self.counts={name:0 for name in IDS}
        sampler=PowerSampler(range(8),interval=.02,backend=self.hardware,sample_clocks=True);sampler.start()
        start=None;failure=None;summary=None
        record=self.state['cells'][str(tokens)]=dict(complete=False,started_s=time.time(),out=str(out),operation=str(self.operation))
        try:
            until=time.monotonic()+5
            while True:
                rows=list(sampler.samples);meta=list(sampler.power_metadata[:len(rows)])
                require(not sampler.error and time.monotonic()<until,'instant outer sampler not ready')
                if len(rows)>=2 and power_evidence(rows,sampler.power_source,meta)['power_source_verified']:break
                await asyncio.sleep(.02)
            start=record['observation_start_s']=time.time()
            self.identities=await self.identity('before-'+str(tokens));self.verified=True
            write(self.operation/'budget-before.json',await self.set_budget(tokens))
            self.child_log=(self.operation/'child.log').open('xb')
            self.child=await asyncio.create_subprocess_exec('python3','-u',str(ROOT/'child.py'),str(tokens),cwd=str(ROOT),
                stdin=asyncio.subprocess.DEVNULL,stdout=self.child_log,stderr=asyncio.subprocess.STDOUT,start_new_session=True)
            record['child_pid']=self.child.pid;self.save()
            waiter=asyncio.create_task(self.child.wait());until=time.monotonic()+800;last_save=0
            while not waiter.done():
                await asyncio.wait({waiter},timeout=.25)
                self.poll_events(tokens)
                require(time.monotonic()<until,'cell controller exceeded bounded original-trace drain window')
                if time.monotonic()-last_save>=5:
                    record.update(owner_model_steps=dict(self.counts),owned_requests={n:len(v) for n,v in self.owned.items()});self.save();last_save=time.monotonic()
            record['child_exitcode']=waiter.result()
            require((out/'summary.json').is_file(),'cell summary missing')
            summary=json.loads((out/'summary.json').read_text());record['summary']=summary
            after={str(p):await self.runtime(p,'after-cell-budget-ack') for p in PORTS}
            for r in after.values():
                check_ack(r);require(is_idle(r) and r['scheduler_budget_effective']==dict(max_num_batched_tokens=tokens,max_num_seqs=32),'budget changed or work residue after cell')
            write(self.operation/'budget-after.json',after)
            require(waiter.result()==0 and summary.get('measurement_valid') is True,'invalid cell; second point forbidden')
            require(summary.get('post_measurement_cleanup',{}).get('cleanup_complete') is True,'post-measurement controller cleanup failed')
            record['valid_primary_cell']=True
        except BaseException as exc:failure=exc;record['error']=repr(exc)
        finally:
            cleanup=await self.cleanup_cell(out,tokens);record['outer_cleanup']=cleanup
            end=record['observation_end_s']=time.time()
            await asyncio.sleep(.15)
            for name,offset in self.initial.items():
                try:
                    with (OLD/'runtime'/(name+'.control.events.jsonl')).open('rb') as f:f.seek(offset);raw=f.read()
                    (self.operation/(name+'.events.jsonl')).write_bytes(raw)
                except BaseException as exc:record.setdefault('event_capture_errors',[]).append(repr(exc))
            await asyncio.to_thread(sampler.stop)
            power=self.operation/'power';power.mkdir()
            save_raw(power,[],sampler.samples,sampler.utilization_samples,power_source=sampler.power_source,power_metadata=sampler.power_metadata)
            with (power/'clocks.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['t_s']+[f'gpu{i}_sm_mhz' for i in range(8)]);w.writerows([t]+list(v) for t,v in sampler.frequency_samples)
            evidence=power_evidence(sampler.samples,sampler.power_source,sampler.power_metadata)
            record.update(power_evidence=evidence,sampling_error=sampler.error)
            try:record['full_operation_energy_j']=trapezoid_energy(clip_power_window(sampler.samples,start,end,pad_s=0)) if start else None
            except BaseException as exc:record['integration_error']=repr(exc)
            record.update(complete=True,continued_allowed=bool(record.get('valid_primary_cell') and cleanup['complete']
                and not record.get('event_capture_errors') and not sampler.error and evidence['power_source_verified']
                and not record.get('integration_error')))
            self.save()
            if self.child_log:self.child_log.close()
        if failure is not None:raise failure
        require(record['continued_allowed'],'invalid measurement/cleanup; do not start next cell')

    async def run(self):
        failure=None
        try:
            self.verify_frozen('before')
            self.hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
            async with aiohttp.ClientSession(trust_env=False) as self.session:
                for tokens in (8192,2048):await self.cell(tokens)
            self.verify_frozen('after')
            self.state.update(phase='finished',complete=True,finished_s=time.time())
        except BaseException as exc:failure=exc;self.state.update(phase='failed',complete=True,error=repr(exc),finished_s=time.time())
        finally:self.save();self.log.close()
        if isinstance(failure,(KeyboardInterrupt,SystemExit,asyncio.CancelledError)):raise failure


async def main():
    task=asyncio.current_task();interrupted=False
    def cancel():
        nonlocal interrupted
        if not interrupted:interrupted=True;task.cancel()
    for sig in (signal.SIGINT,signal.SIGTERM):asyncio.get_running_loop().add_signal_handler(sig,cancel)
    await Pair().run()


if __name__=='__main__':
    with node_lease():asyncio.run(main())
