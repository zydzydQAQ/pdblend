"""Read-only reconstruction of the declared legacy correctness mechanisms.

No energy is assigned to individual requests. Legacy sender counters remain
unknown; the actual synchronous PUT owner/rank proof is the required channel.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

def require(ok,why):
    if not ok:raise RuntimeError(why)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):
    raw=Path(p).read_bytes();require(not raw or raw.endswith(b'\n'),'truncated raw journal: '+str(p))
    return [json.loads(x) for x in raw.splitlines()]
def files(root):return {str(p.resolve()):sha(p) for p in sorted(Path(root).rglob('*')) if p.is_file()}

def ranks(value,tp):
    require(isinstance(value,list) and len(value)==tp,'all legacy ranks must be observed')
    for r in value:
        require(all(type(r.get(k)) is int and r[k]==0 for k in ('buffered_tensors','buffered_gpu_bytes','inflight_receives'))
            and r.get('listener_alive') is True and isinstance(r.get('allocations'),dict) and not r['allocations'],
            'legacy rank residue or missing real observation')

def ack(raw,i):
    require(raw.get('id')==i['id'] and type(raw.get('generation')) is int
        and raw['generation']==raw.get('acknowledged_generation') and not raw.get('error') and not raw.get('runtime_error')
        and raw.get('transport_healthy') is True,'real healthy legacy ACK missing')
    require(all(k in raw and not raw[k] for k in ('active','running','waiting','kv_allocations','transfer_allocations',
        'transfer_buffered_tensors','transfer_inflight_receives')),'legacy runtime residue or missing observation')
    require(all(type(raw.get(k)) in (int,float) and math.isfinite(raw[k]) for k in ('timestamp','transfer_observed_s'))
        and abs(raw['timestamp']-raw['transfer_observed_s'])<=1.,'raw legacy transfer snapshot not fresh at observation')

def identities(gate,instances):
    before=read(gate/'identity.before.json');after=read(gate/'identity.after.json')
    require(len(before)==len(after)==len(instances),'gate physical layout changed')
    for side in (before,after):
        by={x['provenance']['instance_id']:x for x in side};require(set(by)=={i['id'] for i in instances},'gate instance set differs')
        for i in instances:
            x=by[i['id']];c=x['container'];p=x['provenance']
            require(c['Id']==i['container']['id'] and c['Image']==i['container']['image']
                and c['State']['StartedAt']==i['container']['StartedAt'] and c['State']['Running'] is True,
                'gate ran on a different container process')
            require(all(p.get(k)==v for k,v in i['provenance'].items()),'gate model/source/PID differs from current identity')
            ack(x['runtime'],i)
    b={x['provenance']['instance_id']:x for x in before};a={x['provenance']['instance_id']:x for x in after}
    for iid in b:
        for k in ('Id','Image','Config','HostConfig'):
            require(b[iid]['container'].get(k)==a[iid]['container'].get(k),'gate container execution settings changed')
        # Docker may enumerate equivalent mounts in a different order. All
        # mount fields remain bound; only enumeration order is normalized.
        normalized=lambda c:sorted((json.dumps(x,sort_keys=True) for x in c.get('Mounts',[])))
        require(normalized(b[iid]['container'])==normalized(a[iid]['container']),'gate container mounts changed')
        require(b[iid]['container']['State']['Pid']==a[iid]['container']['State']['Pid'],'gate container PID changed')
    return before,after

def native(checks,instances):
    cleanup=checks.get('cleanup',{});require(cleanup.get('complete') is True and not cleanup.get('errors'),'raw cleanup incomplete')
    require(set(cleanup.get('instances',{}))=={i['id'] for i in instances},'native cleanup missing an instance')
    for i in instances:
        row=cleanup['instances'][i['id']];p=row.get('proof',{});r=row.get('resume',{})
        require(row.get('complete') is True and not row.get('errors') and p.get('drained') is True
            and p.get('accepting') is False and p.get('drain_proof_type')=='synchronous_put_owner_barrier'
            and type(p.get('generation')) is int,'actual native owner barrier missing')
        ranks(p.get('transfers'),i['tp']);ack(r['before'],i);ack(r['after'],i);ack(row['after'],i)
        command=r['command'];require(command['generation']==r['before']['generation']+1==r['after']['generation']
            and r['before']['generation']>=p['generation'] and row['after']['generation']==r['after']['generation'],
            'native resume generation was not actually ACKed')
        require(all(command.get(k)==v and r['after'].get(k)==v and row['after'].get(k)==v for k,v in
            dict(role='mixed',mode='continuous',admit_prefill=True,admit_decode=True).items())
            and row['after'].get('accepting') is True,'native cleanup did not resume mixed admission')

def integrate(rows,start,end):
    require(len(rows)>=2 and rows[0][0]<=start<end<=rows[-1][0],'whole energy/clock window not bracketed')
    total=0.
    for (t,x),(u,y) in zip(rows,rows[1:]):
        require(u>t,'sample time did not strictly advance')
        a=max(t,start);b=min(u,end)
        if a<b:
            pa=sum(x)+(sum(y)-sum(x))*(a-t)/(u-t);pb=sum(x)+(sum(y)-sum(x))*(b-t)/(u-t)
            total+=(pa+pb)*(b-a)/2
    return total

def power(gate,status,power_evidence):
    with (gate/'power/power.csv').open() as f:
        reader=csv.DictReader(f);require(all('gpu'+str(i)+'_w' in reader.fieldnames for i in range(8)),'all8 power columns required')
        rows=[(float(r['t_s']),[float(r['gpu'+str(i)+'_w']) for i in range(8)]) for r in reader]
    require(all(math.isfinite(t) and all(math.isfinite(w) and w>=0 for w in v) for t,v in rows),'invalid physical power')
    evidence=power_evidence(rows,read(gate/'power/power_source.json'),lines(gate/'power/power_metadata.jsonl'))
    require(evidence.get('power_source_verified') is True,'raw all8 instant provenance failed')
    start=status['measurement_start_s'];end=status['measurement_end_s'];energy=integrate(rows,start,end)
    require(math.isclose(energy,status['full_operation_energy_j'],rel_tol=1e-9,abs_tol=1e-5),'all8 raw energy differs')
    with (gate/'power/clocks.csv').open() as f:
        reader=csv.DictReader(f);require(reader.fieldnames==['t_s']+[f'gpu{i}_sm_mhz' for i in range(8)],'all8 actual clocks required')
        clocks=[(float(r['t_s']),[float(r[f'gpu{i}_sm_mhz']) for i in range(8)]) for r in reader]
    require(all(math.isfinite(t) and all(math.isfinite(w) and w>0 for w in v) for t,v in clocks),'actual clock observation invalid')
    integrate(clocks,start,end)
    return dict(all8_energy_j=energy,measurement_start_s=start,measurement_end_s=end,power_samples=len(rows),clock_samples=len(clocks),
        clock_scope='actual clock telemetry and successful owned-lock release; not a fixed-frequency profile certification')

def valid_request(row,http):
    body=row['body'];length=len(body['prompt']);n=body['max_tokens'];response=row['response'];ids=response.get('token_ids')
    require(body==dict(prompt=([9707,1879,13]*(length//3+1))[:length],max_tokens=n,temperature=0,top_p=1,ignore_eos=True,seed=0,stream=False),
        'original deterministic prompt/sampling changed')
    require(isinstance(ids,list) and len(ids)==n and all(type(x) is int for x in ids)
        and response.get('usage',{}).get('prompt_tokens')==length and response['usage'].get('completion_tokens')==n
        and not row.get('error') and row['dispatch_s']<=row['finished_s'],'prescribed response incomplete')
    matching=[r for r in http if r.get('route')=='/v1/completions' and r.get('instance')==row['instance_id'] and r.get('request_id')==row['request_id']]
    require(len(matching)==1 and matching[0].get('status')==200 and not matching[0].get('error')
        and matching[0].get('body')==body and matching[0].get('response')==response,'request raw HTTP evidence differs')
    return ids

def mechanisms(checks,http,events,instances,hetero=False):
    requests=checks.get('requests',[]);require(len({r['request_id'] for r in requests})==len(requests),'duplicate request IDs in gate')
    errors={};refs={}
    def evaluated(name,fn):
        try:fn();return True
        except (RuntimeError,KeyError,IndexError,TypeError) as exc:errors[name]=str(exc);return False
    def select(label,iid,length,n=64):
        found=[r for r in requests if r.get('label')==label and r.get('instance_id')==iid and len(r.get('body',{}).get('prompt',[]))==length and r['body'].get('max_tokens')==n]
        require(len(found)==1,'missing/duplicate original request '+label+':'+iid+':'+str(length))
        return found[0],valid_request(found[0],http)
    def ordinary():
        for n in (128,7168):
            if hetero:
                one=[select('ordinary-tp1-'+str(n),i['id'],n)[1] for i in instances[:-1]]
                two=[select('ordinary-tp2-local-repeat-'+str(j),instances[-1]['id'],n)[1] for j in range(2)]
                require(all(x==one[0] for x in one) and two[0]==two[1],'same-TP golden outputs differ')
                refs[n]=two[0]
                require(checks['ordinary'][str(n)]['tp1_token_ids']==one and checks['ordinary'][str(n)]['tp2_token_ids']==two,'ordinary summary differs from raw')
            else:
                replies=[select('ordinary-'+str(n),i['id'],n)[1] for i in instances]
                require(all(x==replies[0] for x in replies),'ordinary replicas differ');refs[n]=replies[0]
                require(checks['ordinary'][str(n)]['token_ids']==replies,'ordinary summary differs from raw')
    ordinary_ok=evaluated('ordinary',ordinary)
    pairs=[(s,instances[-1]) for s in instances[:-1]] if hetero else [(instances[0],d) for d in instances[1:]]
    def pd():
        require(ordinary_ok,'ordinary golden absent')
        require(len(checks.get('pd',[]))==2*len(pairs),'not all declared PD routes completed')
        for source,target in pairs:
            for n in (128,7168):
                found=[r for r in checks['pd'] if (r['source'],r['target'],r['prompt_length'])==(source['id'],target['id'],n)]
                require(len(found)==1 and found[0]['token_ids']==refs[n],'PD result differs from actual golden')
                prefix='cross-tp-' if hetero else 'pd-'
                prod=[r for r in requests if r.get('label')==prefix+'producer' and r['instance_id']==source['id'] and len(r['body']['prompt'])==n and r['request_id'].endswith(':'+source['id']+':'+target['id'])]
                con=[r for r in requests if r.get('label')==prefix+'consumer' and r['instance_id']==target['id'] and len(r['body']['prompt'])==n and r['request_id'].endswith(':'+source['id']+':'+target['id'])]
                require(len(prod)==len(con)==1 and ':p:' in prod[0]['request_id'] and prod[0]['request_id'].replace(':p:',':d:',1)==con[0]['request_id'],'true PD request pair missing')
                require(prod[0]['body']['max_tokens']==1 and con[0]['body']['max_tokens']==64,'PD work changed')
                valid_request(prod[0],http);require(valid_request(con[0],http)==refs[n],'PD HTTP output differs')
            cancellations=[r for r in checks.get('cancelled_unconsumed_kv',[]) if r['target']==target['id'] and (not hetero or r.get('source')==source['id'])]
            require(len(cancellations)==1,'unconsumed KV cancellation route missing');ranks(cancellations[0]['result'].get('transfers'),target['tp'])
            cancels=[r for r in http if r.get('route')=='/cancel' and r.get('instance')==target['id'] and r.get('status')==200
                and r.get('response')==cancellations[0]['result'] and r.get('body',{}).get('request_id','').endswith(':'+source['id']+':'+target['id'])]
            require(bool(cancels),'actual cancellation HTTP evidence absent')
    pd_ok=evaluated('pd',pd)
    def temporal():
        b=instances[1];phase=checks['temporal'];require(phase.get('complete') is True,'temporal pair incomplete')
        refs_t=[select('temporal-single-reference',b['id'],n)[1] for n in (96,192)]
        pair=[select('temporal-'+label,b['id'],n) for label,n in [('first',96),('second',192)]]
        require([x[1] for x in pair]==refs_t==phase['reference_token_ids']==phase['token_ids'],'temporal actual output differs')
        require(pair[0][0]['request_id'] in phase['first_allocated']['kv_allocations'] and phase['first_allocated']['running']==1,'original first allocation not observed')
        require(pair[1][0]['request_id'] not in phase['held']['kv_allocations'] and phase['held']['waiting']>=1
            and phase['held'].get('admit_prefill') is False,'original temporal prefill hold not observed')
        for label,admit in [('close_prefill',False),('open_prefill',True)]:
            item=phase[label];cmd=item['command']
            require(cmd.get('mode')=='temporal' and cmd.get('admit_prefill') is admit
                and cmd['generation']==item['before']['generation']+1==item['after']['generation']==item['after'].get('acknowledged_generation'),
                'temporal original control not ACKed')
        actual=[e for e in events[b['id']] if e.get('mode')=='temporal' and e.get('tokens',0)>0]
        require(actual and any(e.get('prefill') for e in actual) and any(e.get('decode') for e in actual)
            and not any(e.get('prefill') and e.get('decode') for e in actual),'actual temporal owner phase evidence absent/overlapped')
        for row,_ in pair:
            own=[e for e in actual if row['request_id'] in e.get('request_ids',[])]
            require(own and any(e.get('prefill') for e in own) and any(e.get('decode') for e in own),'temporal output lacks matching actual owner execution')
    result=dict(ordinary=ordinary_ok,pd=pd_ok)
    if not hetero:result['temporal']=evaluated('temporal',temporal)
    return result,errors

def audit(gate,instances,strategy,power_evidence,hetero=False):
    gate=Path(gate).resolve();hashes=files(gate);status=read(gate/'status.json');checks=read(gate/'checks/checks.json')
    require(status.get('complete') is True and status.get('finished_s') and status.get('measurement_valid') is True
        and status.get('native_cleanup_complete') is True and status.get('clock_restore_complete') is True
        and not status.get('cleanup_errors') and not status.get('sampling_error') and checks.get('complete') is True,
        'terminal clean measured gate required')
    identities(gate,instances);native(checks,instances);physical=power(gate,status,power_evidence)
    events={}
    for i in instances:
        candidates=[(p,r) for p,r in status.get('events',{}).items() if Path(p).name==i['id']+'.control.events.jsonl']
        require(len(candidates)==1,'owner event reference missing')
        original,record=candidates[0];path=gate/Path(original).name
        require(sha(path)==record['sha256'] and path.stat().st_size==record['offset_end']-record['offset_start'],'owner raw offset/hash differs')
        events[i['id']]=lines(path)
    actual,errors=mechanisms(checks,lines(gate/'checks/http.jsonl'),events,instances,hetero)
    require(status.get('mechanism_gate')==actual,'recorded mechanism gate differs from raw reconstruction')
    flags=checks.get('checks',{})
    require(bool(flags.get('ordinary_cross_replica_exact'))==actual['ordinary'] and
        bool(flags.get('pd_exact_all_declared_pairs') and flags.get('cancel_all_tp_ranks'))==actual['pd']
        and (hetero or bool(flags.get('temporal_exact'))==actual['temporal']),'checks header differs from raw reconstruction')
    needed={'mixed':['ordinary'],'dynamollm':['ordinary'],'dynamollm-resident':['ordinary'],
        'distserve':['ordinary','pd'],'ecoserve':['ordinary','temporal']}[strategy]
    require(all(actual.get(k) for k in needed),'required mechanism failed: '+str(needed)+' '+str(errors))
    require(files(gate)==hashes,'gate evidence changed during reconstruction')
    return dict(required=needed,verified=actual,raw_reconstruction_errors=errors,physical=physical,
        overall_runtime_gate_passed=status.get('passed'),original_failure_preserved=True,legacy_sender_counters='unknown; synchronous PUT native owner/rank proof',
        numerical_scope='exact prescribed output IDs within declared TP oracle; not bitwise KV or logits equivalence'),hashes
