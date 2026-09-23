#!/usr/bin/env python3
"""Read every evaluation shape against explicit bounded PDBlend components.

This is potential shape demand, independent of rate and actual scheduler batch.
It never extends a fitted domain or qualifies a baseline using PDBlend evidence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from pdblend.profile.power_calibration import write_immutable
from pdblend.profile.power_table import context_bounds,PowerCoverageError

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/2026-09-22/three-model/calibration-candidates'
FREQUENCIES=(900,1200,1500,1800,2100,2520)
BATCHES=(1,2,3,4,8,16,32,64,128,256)


def read(path):return json.loads(Path(path).read_text())
def bound(path):return dict(path=str(Path(path).resolve()),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def merge(intervals):
    result=[]
    for low,high in sorted(intervals):
        if high<low:continue
        if result and low<=result[-1][1]:result[-1][1]=max(high,result[-1][1])
        else:result.append([low,high])
    return result


def covered(low,high,intervals):
    """All requested integer decode contexts, without bridging unmeasured gaps."""
    next_token=low
    for left,right in merge(intervals):
        left,right=math.ceil(left),math.floor(right)
        if right<next_token:continue
        if left>next_token:return False
        next_token=right+1
        if next_token>high:return True
    return next_token>high


def intervals(candidate,long,f,b,*,power=False):
    d=candidate['decode_overrides'][str(f)]['domain'];out=[]
    if d['batch'][0]<=b<=d['batch'][1]:
        low,high=d['context'][0],min(d['context'][1],d['max_batch_context']/b)
        if power and candidate.get('decode_power_overrides'):
            try:a,z=context_bounds(candidate['decode_power_overrides'][str(f)],b);low,high=max(low,a),min(high,z)
            except PowerCoverageError:high=low-1
        if low<=high:out.append([low,high])
    if long and f'{f}/{b}' in long['nodes']:
        nodes=long['nodes'][f'{f}/{b}'];out.append([nodes[0]['context'],nodes[-1]['context']])
    return merge(out)


def build():
    versions={}
    for name in ('calibration-versions-v1','calibration-incremental-versions-v1'):
        registry=ROOT/'results/2026-09-23'/name/'registry.json'
        for row in read(registry)['versions']:versions[(row['model_id'],row['tp'])]=(row,registry)
    profiles=[]
    for root in sorted(BASE.iterdir()):
        if not (root/'candidate.json').is_file():continue
        candidate=read(root/'candidate.json');model=Path(candidate['model']).name;tp=candidate['tp'];long=None
        version=versions.get((model,tp));path=root/'candidate.json';provenance=dict(candidate=bound(path),qualification='original_candidate_not_assumed_passed')
        if version:
            row,registry=version;path=Path(row['evidence']['power_candidate']['path']);candidate=read(path)
            if row['evidence'].get('long_candidate'):long=read(row['evidence']['long_candidate']['path'])
            provenance=dict(candidate=bound(path),registry=bound(registry),version_id=row['version_id'],qualification='bounded_component_only')
        profiles.append(dict(model_id=model,tp=tp,pp=1,prefill_bounds=candidate['bounded_coverage']['prefill_tokens'],
            source=provenance,scenarios=[dict(frequency=f,batch=b,timing_intervals=intervals(candidate,long,f,b),
                power_and_timing_intervals=intervals(candidate,long,f,b,power=True)) for f in FREQUENCIES for b in BATCHES]))
    corpora=[];records=[]
    for size in ('7b','14b','32b'):
        root=ROOT/'datasets/prepared'/f'2026-09-22-{size}-v1';manifest=read(root/'manifest.json')
        for dataset in ('alpaca','sharegpt','longbench'):
            path=root/(dataset+'.json');binding=bound(path)
            if manifest['dataset_sha256'][dataset]!=binding['sha256']:raise ValueError('prepared dataset checksum changed')
            values=read(path)['evaluation'];shapes=[]
            for index,value in enumerate(values):
                n,o=value['input_tokens'],value['output_tokens']
                if type(n) is not int or type(o) is not int or n<=0 or o<=0 or len(value['prompt'])!=n:raise ValueError('invalid independent-tokenized evaluation shape')
                row=dict(model_id=manifest['model_name'],dataset=dataset,index=index,input_tokens=n,output_tokens=o,
                    decode_context_interval=[n,n+o-1],content_sha256=value['content_sha256'],request_shape_sha256=value['request_shape_sha256'])
                records.append(row);shapes.append(row)
            corpora.append(dict(model_id=manifest['model_name'],dataset=dataset,count=len(values),manifest=bound(root/'manifest.json'),
                data=binding,tokenizer_sha256=manifest['tokenizer_sha256'],output_work=manifest['output_work'],
                min_input=min(x['input_tokens'] for x in shapes),max_input=max(x['input_tokens'] for x in shapes),
                max_decode_context=max(x['decode_context_interval'][1] for x in shapes),
                input_below128=sum(x['input_tokens']<128 for x in shapes),decode_start_below256=sum(x['input_tokens']<256 for x in shapes),
                profiles=[dict(tp=profile['tp'],prefill_missing=sum(not profile['prefill_bounds'][0]<=x['input_tokens']<=profile['prefill_bounds'][1] for x in shapes),
                    scenarios=[dict(frequency=s['frequency'],batch=s['batch'],
                        timing_full_path_covered=sum(covered(*x['decode_context_interval'],s['timing_intervals']) for x in shapes),
                        energy_full_path_covered=sum(covered(*x['decode_context_interval'],s['power_and_timing_intervals']) for x in shapes))
                        for s in profile['scenarios']]) for profile in profiles if profile['model_id']==manifest['model_name']]))
    return dict(schema=1,kind='evaluation_shape_domain_ledger',system='pdblend',seeds=[701],records=records,corpora=corpora,profiles=profiles,
        total_records=len(records),formal_eligible=False,rate_anchor_pending=True,
        limits=['Batch scenarios are coverage probes, not actual scheduler batches or runtime traces.',
            'Every model uses its own tokenizer/manifest; equal resulting lengths do not imply borrowed tokenization.',
            'Context interval is possible fixed-output work from input through final output context.',
            'This ledger does not qualify original candidates or any independent baseline.',
            'Missing gaps remain unsupported; no input clamp or extrapolation.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if args.out.exists():raise FileExistsError('use a new ledger directory')
    data=build();args.out.mkdir(parents=True);write_immutable(args.out/'ledger.json',data)
    lines=['# Evaluation shape coverage','',f"All {data['total_records']} evaluation records inspected; single seed 701. Rates and actual batches remain pending.",'',
        '| Model | Dataset | Records | Input < 128 | Decode starts < 256 | Input range |','|---|---|---:|---:|---:|---|']
    for c in data['corpora']:lines.append(f"| {c['model_id']} | {c['dataset']} | {c['count']} | {c['input_below128']} | {c['decode_start_below256']} | {c['min_input']}–{c['max_input']} |")
    lines+=['','Current prefill profiles start at 128; short decode profiles start at 256. Long-domain unions retain the unmeasured gap and exact-batch limits.',
        '',*data['limits']]
    (args.out/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(out=str(args.out),records=data['total_records'],profiles=len(data['profiles']))))


if __name__=='__main__':main()
