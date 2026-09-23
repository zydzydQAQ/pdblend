"""End-to-end EcoServe mechanism smoke against native-v1 services."""
import argparse, asyncio, json, time
from .runtime import EcoServeRuntime, MappedEcoServeTransport, validate_native_state

async def run(config, endpoints, prompt, request_id, out):
    journal_rows=[]
    async def journal(kind, **data): journal_rows.append({'kind':kind, **data})
    transport=MappedEcoServeTransport(endpoints)
    runtime=EcoServeRuntime(config, transport, journal)
    status='failed'; error=None; output=[]
    try:
        await runtime.start()
        async for event in runtime.handle({'prompt':prompt,'max_tokens':16}, request_id):
            output.append(event)
        if sum(len(row.get('token_ids',[])) for row in output)!=16 or not output[-1].get('finished'):
            raise RuntimeError('EcoServe output token count or terminal event differs')
        actions={row['kind'] for row in journal_rows}
        required={'eco_startup','eco_admission','eco_engine_output','eco_output_flush'}
        if not required <= actions: raise RuntimeError('missing EcoServe mechanism actions: '+str(required-actions))
        status='passed'
    except Exception as exc:
        error=repr(exc)
    finally:
        try: await runtime.close()
        except Exception as exc: error=error or repr(exc);status='failed'
    artifact={'schema':'ecoserve-mechanism-smoke-v1','status':status,
              'complete':status=='passed','mechanism_validated':status=='passed',
              'formal_eligible':False,'energy_comparable':False,
              'validated_actions':['admission','stream_output','output_flush'],
              'unvalidated_actions':['macro_rotation','split_merge','live_kv_continuity'],
              'request_id':request_id,'output_events':output,'journal':journal_rows,
              'endpoints':endpoints,'at_s':time.time()}
    if error: artifact['error']=error
    with open(out,'w') as f: json.dump(artifact,f,indent=2,sort_keys=True)
    if status!='passed': raise RuntimeError(error or 'EcoServe mechanism smoke failed')
    return artifact

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--endpoint',action='append',required=True,help='id=url')
    p.add_argument('--prompt-tokens',type=int,default=16);p.add_argument('--request-id',default='ecoserve-mechanism-smoke');p.add_argument('--out',required=True)
    args=p.parse_args(argv)
    with open(args.config) as f: config=json.load(f)
    endpoints={key:value for item in args.endpoint for key,value in [item.split('=',1)]}
    print(json.dumps(asyncio.run(run(config,endpoints,list(range(args.prompt_tokens)),args.request_id,args.out)),sort_keys=True))
if __name__=='__main__': main()
