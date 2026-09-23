"""Native-v1 baseline profile collector and artifact audit.

This collector talks only to the baseline service.  It does not import the
PDblend profiler or planner, and refuses to write a usable artifact when the
service returns incomplete evidence.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, statistics, sys, time
import uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

DIST_FREQS = (900, 1200, 1500, 1800, 2100, 2520)
ECO_LENGTHS = (16, 128, 512, 1024, 2048, 4096, 7168)


def _phase_max_tokens(role):
    """Use one token for a prefill sample; decode samples need a stream."""
    if role == 'prefill':
        return 1
    if role == 'decode':
        return 32
    raise ValueError('unsupported native profile role: ' + str(role))

def call(base, method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(base.rstrip('/') + path, data=data, method=method,
                  headers={'Content-Type':'application/json'})
    with urlopen(req, timeout=300) as response:
        return json.loads(response.read())

def generate(base, payload):
    req=Request(base.rstrip('/')+'/baseline/generate',data=json.dumps(payload).encode(),method='POST',headers={'Content-Type':'application/json','Accept':'text/event-stream'})
    with urlopen(req,timeout=300) as response:
        complete=False
        for raw in response:
            line=raw.decode().strip()
            if line.startswith('data:') and line[5:].strip() not in ('','[DONE]'):
                event=json.loads(line[5:].strip())
                if event.get('finished') or event.get('finish_reason'): complete=True
        if not complete: raise RuntimeError('generate ended without terminal event')

def generate_batch(base, payloads):
    ids={p['request_id'] for p in payloads}
    call(base,'POST','/baseline/control',dict(admit_prefill=False,admit_decode=False))
    with ThreadPoolExecutor(max_workers=len(payloads)) as pool:
        futures=[pool.submit(generate,base,p) for p in payloads]
        try:
            deadline=time.monotonic()+30
            while not ids <= set(call(base,'GET','/baseline/state')['all_queue']):
                for future in futures:
                    if future.done():future.result()
                if time.monotonic()>deadline:raise TimeoutError('native batch admission barrier timed out')
                time.sleep(.02)
        finally:
            call(base,'POST','/baseline/control',dict(admit_prefill=True,admit_decode=True))
        for future in futures:future.result()

def _meta(args):
    required = ('model_hash','tokenizer_hash','engine_version','image_digest','source_revision','gpu_uuids')
    out = {key: getattr(args, key) for key in required}
    if not out['gpu_uuids'] or any(not str(x) for x in out['gpu_uuids']):
        raise ValueError('gpu_uuids metadata is required')
    return out

def collect(args):
    meta = _meta(args)
    capability = call(args.url, 'GET', '/baseline/capability')
    for key in ('model_hash','tokenizer_hash','engine_version','image_digest','source_revision'):
        if capability.get(key) != meta[key]:
            raise ValueError('capability provenance mismatch: '+key)
    if not set(meta['gpu_uuids']).issubset(set(capability.get('gpu_uuids', []))):
        raise ValueError('capability GPU UUID inventory does not cover requested GPUs')
    tp, pp = int(capability['tp']), int(capability['pp'])
    if pp != 1:
        raise ValueError('unsupported_engine: rank CUDA events do not include PP wire/host service')
    if len(meta['gpu_uuids']) != tp or set(meta['gpu_uuids']) != set(capability['gpu_uuids']):
        raise ValueError('profile must measure exactly the native TP GPU group')
    meta.update(tp=tp, pp=pp)
    rows=[]; unsupported=[]
    if args.system == 'distserve':
        shapes=[{'role':role,'batch_size':batch,'context_tokens':context}
                for role in ('prefill','decode') for batch in args.batches for context in args.contexts]
        for frequency in DIST_FREQS:
            call(args.url,'POST','/baseline/clock',{'frequency_mhz':frequency})
            for shape in shapes:
                capacity=capability['state']
                if (shape['batch_size']*(shape['context_tokens']+32)>capacity['total_kv_tokens']*.9 or
                        shape['batch_size']>capacity['max_num_seqs'] or
                        (shape['role']=='prefill' and shape['batch_size']*shape['context_tokens']>capacity['max_num_batched_tokens'])):
                    unsupported.append(dict(frequency_mhz=frequency,**shape,status='unsupported_engine',
                                            reason='native KV/batch/full-prefill token budget'))
                    continue
                call(args.url,'POST','/baseline/measurement/start',
                     {'system':'distserve','scope':'runner','frequency_mhz':frequency,**shape})
                generate_batch(args.url, [{'request_id':'profile-'+uuid.uuid4().hex,'prompt':list(range(100,100+shape['context_tokens'])), 'system':'distserve','model':args.model,'role':shape['role'],
                    'context_tokens':shape['context_tokens'],'max_tokens':_phase_max_tokens(shape['role'])} for _ in range(shape['batch_size'])])
                sample=call(args.url,'GET','/baseline/measurement/samples')
                ranks=sample.get('ranks') if isinstance(sample,dict) else None
                if not ranks: raise RuntimeError('missing DistServe rank CUDA samples')
                selected = _select_samples(ranks, role=shape['role'], context=shape['context_tokens'], batch=shape['batch_size'], scope='runner', tp=tp)
                # Keep every native event, including warm-up/partial batches; only
                # exactly observed shapes enter the independent latency table.
                points = measured_points(selected, tp=tp, raw_sha256=_sha(sample))
                rows.append({'frequency_mhz':frequency,**shape,'sample':sample,
                             'sample_sha256':_sha(sample),'points':points})
    else:
        call(args.url,'POST','/baseline/clock',dict(frequency_mhz=args.frequency))
        meta['frequency_mhz']=args.frequency
        for length in ECO_LENGTHS:
            values=[]
            for repeat in range(5):
                call(args.url,'POST','/baseline/measurement/start',
                     {'system':'ecoserve','scope':'forward','input_tokens':length,
                      'batch_size':1,'repeat':repeat})
                generate(args.url, {'request_id':'profile-'+uuid.uuid4().hex,'prompt':list(range(100,100+length)), 'system':'ecoserve','model':args.model,'role':'prefill','input_tokens':length,
                    'batch_size':1,'max_tokens':1})
                sample=call(args.url,'GET','/baseline/measurement/samples')
                if not sample.get('ranks'): raise RuntimeError('missing EcoServe forward samples')
                _select_samples(sample['ranks'], role='prefill', context=length, batch=1, scope='forward', tp=tp)
                values.append(sample)
            rows.append({'input_tokens':length,'repetitions':values,
                         'minimum_ms':min(_sample_ms(v, context=length, tp=tp) for v in values),
                         'samples_sha256':[_sha(v) for v in values]})
    artifact={'schema':'pdblend-baseline-profile-v1','system':args.system,
              'model':args.model,'metadata':meta,'rows':rows,'unsupported':unsupported,
              'created_at_s':time.time(),'complete':True,'formal_eligible':False,
              'evidence_class':'native_cuda_stage_samples',
              'qualification_missing':['independent_holdout','power_calibration','parallel_interference'],
              'measurement_protocol':{'ecoserve':'minimum_of_five_full_prefill_forward_calls',
                                      'distserve':'measured_PP1_runner_events; no_PP_extrapolation'}}
    call(args.url,'POST','/baseline/measurement/stop',{})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)),exist_ok=True)
    if args.system == 'ecoserve':
        # The author controller consumes this exact CSV schema.  Measurement
        # provenance remains in a sidecar manifest rather than changing CSV.
        with open(args.out,'w') as f:
            f.write('Length,Prefill Time\n')
            for row in rows: f.write(f"{row['input_tokens']},{row['minimum_ms']}\n")
        with open(args.out+'.manifest.json','w') as f: json.dump(artifact,f,indent=2,sort_keys=True)
    else:
        with open(args.out,'w') as f: json.dump(artifact,f,indent=2,sort_keys=True)
    return artifact

def _sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _select_samples(ranks, *, role, context, batch, scope, tp):
    if (not isinstance(ranks,list) or len(ranks) != tp or
            {r.get('rank') for r in ranks} != set(range(tp))):
        raise ValueError('rank coverage is missing or duplicated')
    selected=[]
    for rank in sorted(ranks,key=lambda r:r['rank']):
        samples=rank.get('samples')
        if not isinstance(samples,list) or not samples:
            raise ValueError('rank samples missing')
        accepted=[]
        for sample in samples:
            for key in ('role','input_tokens','context_tokens','batch','gpu_elapsed_ms','measurement_scope','request_ids'):
                if key not in sample: raise ValueError('sample field missing: '+key)
            value=sample['gpu_elapsed_ms']
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('invalid CUDA elapsed time')
            if sample.get('failed'): raise ValueError('failed CUDA execution sample')
            if sample['measurement_scope'] != scope:
                raise ValueError('sample measurement scope does not match system')
            if sample['role'] != role or sample['batch'] != batch:
                continue
            if role == 'prefill':
                # Chunk fragments cannot be relabelled as a complete prefill.
                if sample['context_tokens'] != context or sample['input_tokens'] != context*batch:
                    continue
            elif not context < sample['context_tokens'] <= context+32:
                continue
            accepted.append(sample)
        if not accepted: raise ValueError('requested phase/full batch not observed in native events')
        selected.append(accepted)
    # Match ranks by exact request set and context, not response/order alone.
    def identity(s): return (tuple(s['request_ids']),s['role'],s['batch'],s['input_tokens'],s['context_tokens'])
    keys=[list(map(identity,rows)) for rows in selected]
    if any(k != keys[0] for k in keys[1:]):
        raise ValueError('rank event alignment differs')
    return selected


def measured_points(selected, *, tp, raw_sha256):
    points=[]
    for index, sample in enumerate(selected[0]):
        points.append(dict(role=sample['role'],tp=tp,pp=1,stage_index=0,
            batch=sample['batch'],max_input_tokens=sample.get('max_input_tokens',sample['context_tokens']),
            max_context_tokens=sample['context_tokens'],
            stage_latency_ms=max(rows[index]['gpu_elapsed_ms'] for rows in selected),
            source_sha256=raw_sha256,timing_scope='PP1_runner_CUDA'))
    return points


def _sample_ms(sample, *, context, tp):
    rows=_select_samples(sample['ranks'],role='prefill',context=context,batch=1,scope='forward',tp=tp)
    if any(len(rank)!=1 for rank in rows):
        raise ValueError('EcoServe requires one complete prefill forward per repetition')
    # A TP operation finishes when its slowest rank finishes. The paper's MIN
    # applies across five complete invocations, never across TP ranks.
    return max(rank[0]['gpu_elapsed_ms'] for rank in rows)


def distserve_latency(path, *, frequency):
    report=audit(path)
    if report['system'] != 'distserve': raise ValueError('DistServe owns its stage profile')
    with open(path) as stream: artifact=json.load(stream)
    points=[p for row in artifact['rows'] if row['frequency_mhz']==frequency for p in row['points']]
    if not points: raise ValueError('missing_profile: frequency')
    from .distserve.simulator import MeasuredLatency
    return MeasuredLatency(points)


def audit(path):
    path = os.fspath(path)
    manifest_path = path if path.endswith('.json') else path+'.manifest.json'
    with open(manifest_path) as f: artifact=json.load(f)
    if artifact.get('schema')!='pdblend-baseline-profile-v1' or artifact.get('complete') is not True:
        raise ValueError('invalid or incomplete profile schema')
    meta=artifact.get('metadata',{})
    required=('model_hash','tokenizer_hash','engine_version','image_digest','source_revision','gpu_uuids')
    missing=[key for key in required if not meta.get(key)]
    if missing: raise ValueError('missing provenance: '+','.join(missing))
    rows=artifact.get('rows',[])
    if not rows: raise ValueError('empty profile')
    if artifact['system']=='distserve':
        if {row.get('frequency_mhz') for row in rows} != set(DIST_FREQS): raise ValueError('DistServe requires six frequencies')
        for row in rows:
            if row.get('sample_sha256') != _sha(row.get('sample')): raise ValueError('raw sample checksum mismatch')
            selected=_select_samples(row['sample']['ranks'],role=row['role'],context=row['context_tokens'],
                                     batch=row['batch_size'],scope='runner',tp=meta['tp'])
            if row.get('points') != measured_points(selected,tp=meta['tp'],raw_sha256=row['sample_sha256']):
                raise ValueError('latency points differ from rank evidence')
    elif artifact['system']=='ecoserve':
        if {row.get('input_tokens') for row in rows} != set(ECO_LENGTHS): raise ValueError('EcoServe anchor sweep incomplete')
        if any(len(row.get('repetitions',[])) != 5 for row in rows): raise ValueError('EcoServe requires five repetitions per anchor')
        for row in rows:
            if row.get('samples_sha256') != [_sha(v) for v in row['repetitions']]: raise ValueError('raw sample checksum mismatch')
            actual=min(_sample_ms(v,context=row['input_tokens'],tp=meta['tp']) for v in row['repetitions'])
            if row.get('minimum_ms') != actual: raise ValueError('EcoServe MIN differs from full forward evidence')
    else: raise ValueError('unknown baseline system')
    return {'valid':True,'formal_eligible':False,'system':artifact['system'],'rows':len(rows),'metadata':meta}

def main(argv=None):
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('collect'); c.add_argument('--url',required=True); c.add_argument('--system',choices=('distserve','ecoserve'),required=True)
    c.add_argument('--model',required=True); c.add_argument('--out',required=True); c.add_argument('--model-hash',required=True)
    c.add_argument('--tokenizer-hash',required=True); c.add_argument('--engine-version',required=True); c.add_argument('--image-digest',required=True)
    c.add_argument('--source-revision',required=True); c.add_argument('--gpu-uuids',nargs='+',required=True)
    c.add_argument('--batches',type=int,nargs='+',default=[1,4,8]); c.add_argument('--contexts',type=int,nargs='+',default=[128,512,2048,4096])
    c.add_argument('--frequency',type=int,choices=DIST_FREQS,default=2520)
    a=sub.add_parser('audit'); a.add_argument('path')
    args=p.parse_args(argv)
    try: result=collect(args) if args.command=='collect' else audit(args.path)
    except Exception as exc: p.error(str(exc))
    print(json.dumps(result if args.command=='audit' else {'out':args.out,'rows':len(result['rows'])},sort_keys=True))
if __name__=='__main__': main()
