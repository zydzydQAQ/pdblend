"""Offline corpus preparation; no reference answer is exposed to online policy.

The answer supplies only a fixed public output-work limit, shared by every
system. LongBench keeps its official task instruction and question. Contexts
are truncated in the middle to the explicitly frozen engine budget.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random


FORMAL_SEEDS=(101,202,303)


def fingerprint(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()


def raw_examples(dataset,root,templates):
    if dataset=='alpaca':
        path=root/'alpaca_gpt4.json'
        for index,item in enumerate(json.loads(path.read_text())):
            text=item.get('instruction','')
            if item.get('input'): text+='\n\n'+item['input']
            yield str(path),index,[dict(role='user',content=text)],str(item.get('output',''))
    elif dataset=='sharegpt':
        path=root/'ShareGPT_V3_unfiltered_cleaned_split.json'
        for index,item in enumerate(json.loads(path.read_text())):
            messages=[];choices=[]
            for turn in item.get('conversations',[]):
                role={'human':'user','gpt':'assistant','system':'system'}.get(turn.get('from'))
                if role is None: continue
                if role=='assistant' and messages and messages[-1]['role']=='user':
                    choices.append((list(messages),str(turn.get('value',''))))
                messages.append(dict(role=role,content=str(turn.get('value',''))))
            if choices:
                prompt,answer=choices[-1]
                yield str(path),index,prompt,answer
    elif dataset=='longbench':
        for path in sorted((root/'longbench').glob('*.jsonl')):
            if path.stem.endswith('_e'): continue  # subset duplicates of original tasks
            if path.stem not in templates:
                raise ValueError('missing official LongBench template: '+path.stem)
            for index,line in enumerate(path.read_text().splitlines()):
                if not line.strip(): continue
                item=json.loads(line)
                answers=item.get('answers',[])
                if not answers: continue
                prompt=templates[path.stem].format(**item)
                yield str(path),index,[dict(role='user',content=prompt)],str(answers[0])
    else:
        raise ValueError('unknown dataset')


def encode_workload(tokenizer,messages,reference,max_input=7168,max_output=512):
    if max_input<2 or max_output<1:
        raise ValueError('positive output and at least two input tokens required')
    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
    original=len(ids)
    if not ids or not reference.strip(): return None
    reference_len=len(tokenizer.encode(reference,add_special_tokens=False))
    if reference_len<1: return None
    if len(ids)>max_input:
        left=max_input//2
        ids=ids[:left]+ids[-(max_input-left):]
    return dict(prompt=ids,input_tokens=len(ids),output_tokens=min(max_output,reference_len),
                original_input_tokens=original,truncated=original>max_input)


def make_trace(records,rate,seed,*,dataset,split,load):
    """Freeze a paired open-loop trace; references never enter this artifact."""
    if rate<=0 or not records or split not in ('calibration','development','formal'):
        raise ValueError('positive rate and a nonempty explicit split required')
    rng=random.Random(seed)
    requests=[];prompts=[];arrival=0.
    for index,record in enumerate(records):
        if index: arrival+=rng.expovariate(rate)
        prompts.append(record['prompt'])
        requests.append(dict(arrival_s=arrival,prompt_len=record['input_tokens'],
                             output_len=record['output_tokens']))
    return dict(schema=2,dataset=dataset,split=split,load=load,seed=seed,rate=rate,
        duration_s=arrival,requests=requests,prompts=prompts,
        source_shapes=[r['request_shape_sha256'] for r in records])


def make_dynamic_trace(corpora,capacities,phases,seed):
    """A paired 60-minute trace derived from independent static capacities.

    Mixture capacity is the harmonic service-demand approximation, explicitly
    recorded for subsequent dynamic validation. An arrival at each endpoint
    makes the prescribed observation span unambiguous, without future signals
    being exposed to online policies.
    """
    datasets=('alpaca','sharegpt','longbench')
    if (not phases or phases[0]['start_s']!=0 or phases[-1]['end_s']!=3600
            or any(a['end_s']!=b['start_s'] for a,b in zip(phases,phases[1:]))
            or any(capacities.get(d,0)<=0 or not corpora.get(d) for d in datasets)):
        raise ValueError('complete contiguous 60-minute phases and measured positive capacities required')
    rng=random.Random(seed);records=[];arrivals=[];metadata=[]
    for phase in phases:
        mix=phase['length_mix']
        if len(mix)!=3 or any(p<0 for p in mix) or abs(sum(mix)-1)>1e-9:
            raise ValueError('three normalized dataset mixture weights required')
        if not 0<phase['capacity_fraction']<=1 or phase['end_s']<=phase['start_s']:
            raise ValueError('positive duration and feasible load fraction required')
        rate=phase['capacity_fraction']/sum(p/capacities[d] for d,p in zip(datasets,mix))
        at=phase['start_s']
        while at<phase['end_s']:
            dataset=rng.choices(datasets,weights=mix,k=1)[0]
            record=dict(rng.choice(corpora[dataset]),dataset=dataset)
            records.append(record);arrivals.append(at)
            at+=rng.expovariate(rate)
        metadata.append(dict(phase,rate=rate))
    records.append(dict(records[-1]));arrivals.append(3600.)
    return dict(schema=2,dataset='dynamic',split='formal',load='changing',seed=seed,duration_s=3600,
        phases=metadata,capacity_model='harmonic mixture of independent baseline capacities; validate on the dynamic run',
        requests=[dict(arrival_s=at,prompt_len=r['input_tokens'],output_len=r['output_tokens'])
                  for r,at in zip(records,arrivals)],prompts=[r['prompt'] for r in records],
        source_shapes=[r['request_shape_sha256'] for r in records],
        source_datasets=[r['dataset'] for r in records])


def prepare(args):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    templates=json.loads(args.templates.read_text())
    args.out.mkdir(parents=True,exist_ok=False)
    manifest=dict(schema=2,model='Qwen2.5-14B-Instruct',input_limit=args.max_input,output_limit=args.max_output,
        output_work='min(reference token length, public output limit); ignore_eos=True',
        truncation='middle; preserve prompt head and final question/chat suffix',
        template_sha256=fingerprint(args.templates),formal_seeds=FORMAL_SEEDS,source_files={})
    summaries={}
    for dataset in ('alpaca','sharegpt','longbench'):
        examples=list(raw_examples(dataset,args.raw,templates))
        rng=random.Random(20260906+('alpaca','sharegpt','longbench').index(dataset))
        rng.shuffle(examples)
        records=[];seen=set()
        needed=args.calibration+args.development+args.formal_pool
        for source,index,messages,reference in examples:
            key=hashlib.sha256(json.dumps(messages,ensure_ascii=False).encode()).hexdigest()
            if key in seen: continue
            seen.add(key)
            encoded=encode_workload(tokenizer,messages,reference,args.max_input,args.max_output)
            if encoded is None: continue
            encoded.update(source_file=source,source_index=index,request_shape_sha256=key)
            records.append(encoded)
            if len(records)>=needed: break
        if len(records)<needed: raise ValueError(f'{dataset} has too few distinct usable requests')
        for source in {r['source_file'] for r in records}:
            if source not in manifest['source_files']: manifest['source_files'][source]=fingerprint(source)
        calibration=records[:args.calibration]
        development=records[args.calibration:args.calibration+args.development]
        formal=records[args.calibration+args.development:]
        payload=dict(schema=2,dataset=dataset,calibration=calibration,development=development,
                     formal={str(seed):random.Random(seed).sample(formal,500) for seed in FORMAL_SEEDS})
        path=args.out/(dataset+'.json');path.write_text(json.dumps(payload,ensure_ascii=False))
        def quantile(values,p): return sorted(values)[min(len(values)-1,int(p*len(values)))]
        summaries[dataset]=dict(records=len(records),truncated=sum(r['truncated'] for r in records),
            input_p50=quantile([r['input_tokens'] for r in records],.5),
            input_p90=quantile([r['input_tokens'] for r in records],.9),
            output_p50=quantile([r['output_tokens'] for r in records],.5),
            output_p90=quantile([r['output_tokens'] for r in records],.9),sha256=fingerprint(path))
        print(json.dumps(dict(dataset=dataset,**summaries[dataset])),flush=True)
    manifest['datasets']=summaries
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--templates',type=Path,required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-input',type=int,default=7168)
    p.add_argument('--max-output',type=int,default=512)
    p.add_argument('--calibration',type=int,default=256)
    p.add_argument('--development',type=int,default=256)
    p.add_argument('--formal-pool',type=int,default=1500)
    args=p.parse_args()
    if (args.formal_pool<500 or args.calibration<1 or args.development<1
            or args.max_input<2 or args.max_output<1 or args.max_input+args.max_output>8192):
        p.error('requires at least 500 formal examples and the common 8192-token engine limit')
    prepare(args)


if __name__=='__main__': main()
