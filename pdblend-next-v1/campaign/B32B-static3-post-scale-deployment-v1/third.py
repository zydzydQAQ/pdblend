"""Only the additional TP2 creation/peer handoff, using frozen restore primitives."""
import copy
import hashlib
import json
import socket
import time
from pathlib import Path
import contracts as c


def label(spec):
    return hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def preflight(spec,inventory):
    i=spec['third_instance'];future=spec['third_creation']
    c.require(i['container_name'] not in {r['Name'].lstrip('/') for r in inventory},'new third name already exists; no implicit retry')
    runtime=Path(c.read(i['config'])['runtime_dir'])
    c.require(not runtime.exists(),'third runtime already exists; preserve prior attempt')
    c.require(c.sha(i['config'])==spec['files'][i['config']],'third config changed')
    c.require(c.sha(future['entry_argv'][1])==future['engine_entry_sha256'],'actual v3 entry changed')
    for port in [i['port'],*range(i['kv_port'],i['kv_port']+32)]:
        with socket.socket() as s:
            try:s.bind(('127.0.0.1',port))
            except OSError as exc:raise RuntimeError('new HTTP/KV port occupied: '+str(port)) from exc


def argv(spec):
    i=spec['third_instance'];f=spec['third_creation'];hc=f['host_config']
    c.require(hc['NetworkMode']=='host' and hc['IpcMode']=='host'
        and hc['DeviceRequests']==[dict(Driver='',Count=-1,DeviceIDs=None,Capabilities=[['gpu']],Options={})],
        'unexpected original hardware/container template')
    args=['docker','run','-d','--name',i['container_name'],'--gpus','all','--network','host','--ipc','host',
          '--label','pdblend.static3.spec='+label(spec)]
    for value in hc.get('SecurityOpt') or []:args+=['--security-opt',value]
    for value in f['env']:args+=['-e',value]
    for value in hc['Binds']:args+=['-v',value]
    return args+[f['image']]+f['entry_argv']


def validate_created(spec,row):
    i=spec['third_instance'];f=spec['third_creation']
    c.require(row['Name'].lstrip('/')==i['container_name'] and row['Image']==f['image']
        and row['Config']['Cmd']==f['entry_argv'] and row['Path']=='python3','third image/name/entry differs')
    c.require(row['Config'].get('Labels',{}).get('pdblend.static3.spec')==label(spec),'third creation ownership not proven')
    env=lambda values:dict(x.split('=',1) for x in values)
    c.require(len(env(row['Config']['Env']))==len(row['Config']['Env'])
        and env(row['Config']['Env'])==env(f['env']),'third runtime environment differs')
    c.require(row['HostConfig']==f['host_config'],'third actual full Docker host settings differ')
    # Actual bind mounts must point to the original workspace and read-only model.
    mounts={r['Destination']:r for r in row['Mounts']}
    c.require(mounts['/root/workspace']['Source']=='/root/workspace' and mounts['/root/workspace']['RW'] is True
        and mounts['/models']['Source']=='/root/workspace/models' and mounts['/models']['RW'] is False,'third source/model mounts differ')
    s=row['State'];c.require(s['Running'] is True and type(s['Pid']) is int and s['Pid']>0
        and not any(s.get(k) for k in ('Paused','Restarting','Dead')),'third actual live process missing')
    return True


def peer_reply(reply,field,expected):
    rows=reply.get(field)
    c.require(isinstance(rows,list) and len(rows)==2
        and all(isinstance(r,dict) and type(r.get('rank')) is int for r in rows)
        and {r['rank'] for r in rows}=={0,1}
        and all(all(r.get(k)==v for k,v in expected.items()) for r in rows),
        'actual peer response must bind both distinct TP ranks and requested peer IDs')


async def create_and_prepare(spec,session,limit,result,common,m):
    i=spec['third_instance'];out=Path(spec['out'])
    # All create intents precede Docker: timeout may occur after the daemon created it.
    intent=dict(id=i['id'],name=i['container_name'],spec_canonical_sha256=label(spec),issued_s=time.time())
    result['creation_intents'].append(intent);m.write(out/'creation-intents'/(i['id']+'.json'),intent)
    cid=(await m.command(argv(spec),limit,result['commands'],cap=30)).strip()
    rows=json.loads(await m.command(['docker','inspect',i['container_name']],limit,result['commands']))
    c.require(len(rows)==1 and rows[0]['Id']==cid,'third actual ID differs from Docker acknowledgement')
    validate_created(spec,rows[0]);result['third_identity_initial']=rows[0]
    result['third_created']=dict(id=i['id'],name=i['container_name'],container_id=cid)
    result['new_containers_created']=True
    p,r=await m.ready(session,i,spec['third_expected_provenance'],limit,result['http_records'],common)
    result['new_provenance'][i['id']]=p;result['startup'][i['id']]=r
    all_instances=spec['instances']+[i]
    for owner in all_instances:await m.idle(session,owner,limit,result['http_records'],common)
    result['peer_registration']=[];result['peer_preparation']=[]
    for action in spec['third_creation']['register_new_peer_after_all_three_idle']:
        owner=next(x for x in all_instances if x['id']==action['instance'])
        reply=await m.http(session,owner,action['path'],action['json'],limit,result['http_records'])
        c.require(reply.get('id')==i['id'],'owner peer registration ID differs')
        peer_reply(reply,'registered',dict(id=i['id']))
        result['peer_registration'].append(dict(instance=owner['id'],request=action['json'],reply=reply))
    for action in spec['third_creation']['prepare_pairs']:
        owner=next(x for x in all_instances if x['id']==action['instance'])
        reply=await m.http(session,owner,'/prepare-peers',dict(peers=action['peers']),limit,result['http_records'])
        peer_reply(reply,'ready',dict(peers=action['peers']))
        result['peer_preparation'].append(dict(instance=owner['id'],request=dict(peers=action['peers']),reply=reply))
    # Re-drain and resume every owner after all newly established peer channels.
    result['native_after_peers']={x['id']:{} for x in all_instances}
    for owner in all_instances:
        await m.native(session,owner,limit,result['http_records'],common,result['native_after_peers'][owner['id']])
        result['new_native_restore'][owner['id']]=copy.deepcopy(result['native_after_peers'][owner['id']])
    result['fresh_free_kv_tokens']={x['id']:result['native_after_peers'][x['id']]['resumed']['after'].get('free_kv_tokens') for x in all_instances}
    c.require(all(type(v) is int and v>0 for v in result['fresh_free_kv_tokens'].values()),
              'actual positive fresh free-KV capacity must be recorded for each replica')


def verify_final(spec,result,by):
    i=spec['third_instance'];row=by[i['container_name']];validate_created(spec,row)
    initial=result['third_identity_initial']
    c.require(row['Id']==initial['Id'] and row['State']['Pid']==initial['State']['Pid']
        and row['State']['StartedAt']==initial['State']['StartedAt'],'third restarted during deployment')
    c.require(len({x['State']['Pid'] for x in by.values()})==3,'three distinct actual owner host processes required')
    c.require(set(result['native_after_peers'])=={x['id'] for x in spec['instances']+[i]}
        and all(r.get('complete') is True and not r.get('errors') for r in result['native_after_peers'].values()),'post-peer native drain/resume incomplete')


async def stop_intents(spec,limit,result,m):
    for intent in result['creation_intents']:
        try:
            # Inspect by preflight-absent unique name even if Docker timed out
            # before returning an ID. Never remove the created or retained object.
            rows=json.loads(await m.command(['docker','inspect',intent['name']],limit,result['commands'],cap=10))
            c.require(len(rows)==1 and rows[0]['Name'].lstrip('/')==intent['name']
                and rows[0]['Image']==spec['third_creation']['image']
                and rows[0]['Config'].get('Labels',{}).get('pdblend.static3.spec')==label(spec),
                'failed-create cleanup could not establish ownership')
            intent['actual_container_id']=rows[0]['Id']
            await m.command(['docker','stop','--time','10',rows[0]['Id']],limit,result['commands'],cap=20)
            intent['stopped_after_failure']=True
        except Exception as exc:
            result['errors'].append('stop third creation intent: '+repr(exc))


def bootstrap(spec,spec_path):
    out=Path(spec['out']);receipt=out/'deployment-receipt.json';r=c.read(receipt)
    c.require(r.get('complete') is True and r.get('measurement_valid') is True and not r.get('errors'),
              'actual measured terminal deployment required')
    c.require(all(c.sha(p)==h for p,h in r['artifacts'].items()),'actual deployment evidence changed')
    inventory=c.read(out/'containers.after.json');by={x['Name'].lstrip('/'):x for x in inventory}
    verify_final(spec,r,by)
    b=copy.deepcopy(c.read(c.PDB));third=copy.deepcopy(spec['third_instance'])
    third.pop('config',None);third.pop('container_name',None)
    b['instances'].append(third)
    identities=[]
    for i in b['instances']:
        name=next(x['container_name'] for x in spec['instances']+[spec['third_instance']] if x['id']==i['id'])
        row=by[name]
        i['container']=dict(name=name,id=row['Id'],image=row['Image'],StartedAt=row['State']['StartedAt'])
        i['provenance']=r['new_provenance'][i['id']]
        identities.append(dict(container=row,provenance=i['provenance'],
            runtime=r['native_after_peers'][i['id']]['resumed']['after']))
    identity=out/'identity.json'
    with identity.open('x') as f:json.dump(identities,f,indent=2,allow_nan=False);f.write('\n')
    b.update(configs={},identity_file=str(identity),output=str(out/'correctness-only-unused'),
             output_correctness_verified=False,fresh_three_replica_correctness_required=True,
             performance_authorized=False,static3_deployment=dict(spec=str(Path(spec_path).resolve()),spec_sha256=c.sha(spec_path),
                 receipt=str(receipt),receipt_sha256=c.sha(receipt)))
    c.merge(b['files'],spec['files']);c.merge(b['files'],r['artifacts'])
    c.merge(b['files'],{str(receipt):c.sha(receipt),str(Path(spec_path).resolve()):c.sha(spec_path),str(identity):c.sha(identity)})
    path=out/'restored-static3-bootstrap.json'
    with path.open('x') as f:json.dump(b,f,indent=2,allow_nan=False);f.write('\n')
    return dict(path=str(path),sha256=c.sha(path),configs_empty=True,ordinary_tested=False)
