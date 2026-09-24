"""New opt-in source/target gateway for same-TP development qualification only.

Frozen source-only service is unchanged. A target has no public mutation route
and no activation operation. Goldens use the actual native generation pipeline.
"""
from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter,HTTPException,Request
from starlette.responses import JSONResponse

from .stationary_ipc import need,process_matches
from .stationary_owner_graph import read_bound
from .stationary_service import AdmissionFence,RequestNative,StationaryContext,StationaryCoordinator,PREFIX
from .stationary_target_bootstrap import TargetBootstrap


TARGET_PREFIX='/baseline/dynamollm/target/'
source_router=APIRouter();target_router=APIRouter()


class TargetAwareSourceCoordinator(StationaryCoordinator):
    async def execute(self,operation,payload):
        if operation not in ('export_to_target','target_release_ack'):
            return await super().execute(operation,payload)
        need(isinstance(payload,dict) and 'native_scheduler_drain' not in payload,
             'source gateway must collect the actual fresh native drain')
        async with self.context.lock:
            c=self.context
            identity=self._epoch(payload)
            need(identity['tp']==1 and c.phase=='released' and c.transaction is not None
                 and all(payload.get(k)==c.transaction[k] for k in ('transaction_id','expected_generation','expected_gpu_uuids')),
                 'target export/ACK requires the exact same-TP pinned source with absent KV')
            try:
                await self._drain(payload)
                if operation=='export_to_target':
                    ready=read_bound(payload['target_ready_ref'])
                    bootstrap=TargetBootstrap(ready['bootstrap_ref'])
                    need(ready['schema']=='dynamo-target-worker-bootstrap-ready/v1'
                         and ready['transaction_id']==payload['transaction_id']
                         and ready['gpu_uuid']==identity['identity']['gpu_uuids'][0]
                         and bootstrap.config['source_generation']==payload['expected_generation']
                         and bootstrap.config['plan']['plan_sha256']==ready['plan_sha256']
                         and ready['target_generation']==bootstrap.config['target_generation']
                         and ready['target_rank']==0 and process_matches(ready['process']),
                         'actual target bootstrap does not match pinned original source')
                    rows=await self._workers('export',dict(payload,target_rank=0,consumer=ready['process']))
                    self._rank_ack(rows,payload)
                    packet=rows[0]['packet']
                    need(packet['consumer']==ready['process'] and packet['plan_sha256']==ready['plan_sha256'],
                         'source exported another target/plan')
                    return dict(source=self.receipt(),packet=packet,formal_eligible=False)
                ack=read_bound(payload['release_ack_ref'])
                rows=await self._workers('consumer_release_ack',dict(payload,source_rank=0,receipt=ack))
                self._rank_ack(rows,payload)
                need(rows[0].get('acknowledged') is True,'original owner did not accept consumer clean ACK')
                return dict(source=self.receipt(),ranks=rows,consumer_process_exit_still_required=True,formal_eligible=False)
            except BaseException as error:
                q=await self._quarantine(error)
                raise HTTPException(503,detail=json.dumps(dict(error=repr(error),quarantine=q,source=self.receipt()))) from error


@source_router.post(PREFIX+'{operation}')
async def source_operation(request:Request,operation:str):
    try:
        return await TargetAwareSourceCoordinator(request.app.state.dynamo_stationary,RequestNative(request)).execute(
            operation,await request.json())
    except ValueError as error:raise HTTPException(409,str(error)) from error


class TargetPublicFence:
    async def __call__(self,scope,receive,send):
        if scope['type']=='http' and scope.get('method') not in ('GET','HEAD','OPTIONS') and not scope['path'].startswith(TARGET_PREFIX):
            return await JSONResponse(dict(error='stationary target only accepts bound private qualification requests'),
                status_code=409)(scope,receive,send)
        return await self.app(scope,receive,send)
    def __init__(self,app):self.app=app


def read_golden(bootstrap,identity):
    golden=read_bound(bootstrap.config['golden_ref']);raw=read_bound(golden['raw_ref'])
    need(golden.get('schema')=='dynamo-native-ordinary-golden/v1'
         and golden.get('tp')==golden.get('pp')==1
         and all(golden.get(k)==identity.get(k) for k in ('model_id','model_hash','tokenizer_hash','engine_revision','image_digest'))
         and golden.get('prompt') and all(type(t) is int and t>=0 for t in golden['prompt'])
         and type(golden.get('seed')) is int and golden.get('ignore_eos') is True
         and type(golden.get('max_tokens')) is int and 0<golden['max_tokens']<=64
         and raw.get('token_ids')==golden.get('token_ids') and len(golden['token_ids'])==golden['max_tokens']
         and raw.get('events') and raw['events'][-1].get('finished') is True
         and [t for event in raw['events'] for t in event['token_ids']]==golden['token_ids'],
         'bound actual ordinary same-TP golden is incomplete or has another model identity')
    return golden


@target_router.post(TARGET_PREFIX+'{operation}')
async def target_operation(request:Request,operation:str):
    from pdblend_runtime import serve
    from pdblend.online.native_control import validate_state
    bootstrap=request.app.state.dynamo_target_bootstrap;c=bootstrap.config
    p=await request.json()
    try:
        need(type(p) is dict and type(p.get('expected_generation')) is int
             and p==dict(transaction_id=c['transaction_id'],expected_generation=c['target_generation']),
             'target private request differs from bound transaction/epoch')
        need(operation in ('status','golden','close'), 'target public activation is not implemented')
        async with request.app.state.dynamo_target_lock:
            async def status():
                rows=await asyncio.wait_for(serve.workers(request,'dynamo_target_operation',operation='status',payload=p),35.)
                need(len(rows)==1 and rows[0]['rank']==0 and rows[0]['generation']==c['target_generation']
                     and rows[0]['gpu_uuid']==c['plan']['target_gpu_uuids'][0], 'real target worker identity differs')
                return rows[0]
            async def drain():
                started=time.time();r=await asyncio.wait_for(serve.drain_engine(request,dict(timeout_s=30)),35.)
                validate_state(r,generation=c['target_generation'],tp=1,pp=1,drained=True,observed_after_s=started)
                need(r.get('acknowledged') is True and r.get('accepting') is False,'target native drain/admission missing')
                return r
            if operation=='status':return await status()
            before=await drain()
            if operation=='close':
                prior=await status()
                rows=await asyncio.wait_for(serve.workers(request,'dynamo_target_operation',operation='close',
                    payload=dict(p,native_scheduler_drain=before)),35.)
                need(len(rows)==1 and rows[0].get('rank')==0 and rows[0].get('generation')==c['target_generation']
                     and rows[0].get('process')==prior['process']
                     and rows[0].get('gpu_uuid')==c['plan']['target_gpu_uuids'][0]
                     and rows[0].get('views_released') is True and rows[0].get('release_ack'),
                     'target did not return a real imported-view release ACK')
                ack=rows[0]['release_ack']
                need(ack.get('consumer')==prior['process'] and ack.get('gpu_uuid')==rows[0]['gpu_uuid']
                     and ack.get('views_released') is True and ack.get('cuda_synchronized') is True,
                     'target release ACK has another actual consumer/UUID or remains live')
                return dict(drain=before,ranks=rows,target_process_exit_required=True,formal_eligible=False)
            need(not request.app.state.dynamo_target_golden_started,'fixed target golden may execute only once')
            request.app.state.dynamo_target_golden_started=True
            golden=read_golden(bootstrap,request.app.state.native_identity)
            prior=await status();events=[];failure=None
            need(prior['KV_initialized'] is True and prior['binding_closed'] is False, 'target has no actual initialized native KV/binding')
            try:
                # The same native generator used by /baseline/generate, with
                # admission bypass restricted to this one bound qualification.
                async def collect():
                    async for event in serve.generate_events(request,dict(request_id=c['transaction_id']+'-target-golden',
                        prompt=golden['prompt'],max_tokens=golden['max_tokens'],seed=golden['seed'],ignore_eos=True),private=True):
                        events.append(event)
                await asyncio.wait_for(collect(),120.)
            except BaseException as error:failure=repr(error)
            final=after=None;cleanup_failure=None
            try:
                final=await drain();after=await status()
            except BaseException as error:cleanup_failure=repr(error)
            tokens=[t for e in events for t in e['token_ids']]
            passed=failure is None and cleanup_failure is None and bool(events) and events[-1].get('finished') is True and tokens==golden['token_ids']
            passed=passed and after['native_execute_model_completed_calls']>prior['native_execute_model_completed_calls']
            result=dict(status='passed' if passed else 'failed',error=failure,cleanup_error=cleanup_failure,
                process_isolation_required=cleanup_failure is not None,events=events,token_ids=tokens,
                golden_ref=c['golden_ref'],before=prior,after=after,final_native_drain=final,
                private_native_output_match=passed,target_public_admission=False,target_served=False,
                full_tp_switch_qualified=False,formal_eligible=False)
            bootstrap.record('native-golden',**result)
            return result
    except ValueError as error:raise HTTPException(409,str(error)) from error


def main():
    import argparse,sys
    from vllm.entrypoints.openai import api_server
    from pdblend_runtime import serve
    from .stationary_target_loader import LOAD_FORMAT
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--stationary-role',choices=('source','target'),required=True)
    parser.add_argument('--bootstrap-ref')
    options,argv=parser.parse_known_args()
    need('--enable-sleep-mode' not in argv and '--kv-transfer-config' not in argv,
         'private same-TP service forbids sleep and KV connectors')
    if '--enforce-eager' not in argv:argv+=['--enforce-eager']
    bootstrap=None
    settings={'--worker-extension-cls':'pdblend_baselines.dynamollm.stationary_target_worker.TargetAwareSourceExtension'}
    if options.stationary_role=='target':
        need(options.bootstrap_ref,'target bootstrap file/SHA binding required')
        ref=json.loads(options.bootstrap_ref);bootstrap=TargetBootstrap(ref)
        settings={'--worker-cls':'pdblend_baselines.dynamollm.stationary_target_worker.SameTpTargetWorker',
            '--scheduler-cls':'pdblend_baselines.dynamollm.stationary_target_worker.SameTpTargetScheduler',
            '--load-format':LOAD_FORMAT,'--model-loader-extra-config':json.dumps(dict(bootstrap_ref=ref))}
        need('--worker-extension-cls' not in argv,'target worker already owns the private operations')
    for key,value in settings.items():
        if key in argv:need(argv[argv.index(key)+1]==value,'private same-TP launch override differs: '+key)
        else:argv += [key,value]
    original=api_server.build_app
    def build_app(args):
        app=original(args)
        if options.stationary_role=='source':
            app.state.dynamo_stationary=StationaryContext();app.add_middleware(AdmissionFence);app.include_router(source_router)
        else:
            app.state.dynamo_target_bootstrap=bootstrap;app.state.dynamo_target_lock=asyncio.Lock()
            app.state.dynamo_target_golden_started=False;app.add_middleware(TargetPublicFence);app.include_router(target_router)
        return app
    api_server.build_app=build_app;sys.argv=[sys.argv[0],*argv];serve.main()


if __name__=='__main__':main()
