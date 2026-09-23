#!/usr/bin/env python3
"""Narrow PDBlend coverage gaps to already frozen evaluation requests.

The input index binds the exact seed701 evaluation traces used by all systems.
Batch/frequency scenarios remain explicit planner queries, not observed native
scheduler geometry. This tool never constructs rates or changes a trace.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def missing_segments(low,high,covered):
    cursor=low;missing=[]
    for left,right in sorted((math.ceil(a),math.floor(b)) for a,b in covered):
        if right<cursor:continue
        if left>high:break
        if left>cursor:missing.append([cursor,min(high,left-1)])
        cursor=max(cursor,right+1)
        if cursor>high:break
    if cursor<=high:missing.append([cursor,high])
    return missing


def union(values):
    result=[]
    for left,right in sorted(values):
        if result and left<=result[-1][1]+1:result[-1][1]=max(right,result[-1][1])
        else:result.append([left,right])
    return result


def build(index,profile_ledger,*,batches):
    index,profile_ledger=Path(index),Path(profile_ledger)
    prepared=read(index);profiles=read(profile_ledger)['profiles'];traces=[]
    if prepared.get('kind')!='frozen_evaluation_trace_index' or prepared.get('seed')!=701:
        raise ValueError('explicit frozen seed701 evaluation trace index required')
    for binding in prepared['traces']:
        path=Path(binding['path'])
        if sha(path)!=binding['sha256'] or binding.get('split')!='evaluation' or binding.get('seed')!=701:
            raise ValueError('evaluation trace identity/split/checksum differs')
        data=read(path);rows=data['requests']
        if data['seed']!=701 or data.get('trace_sha256',canonical(rows))!=canonical(rows):
            raise ValueError('evaluation trace seed/content differs')
        if (not rows or len({r['idx'] for r in rows})!=len(rows) or
                any(not isinstance(r['max_tokens'],int) or not 2<=r['max_tokens']<=512 or
                    not 1<=len(r['prompt'])<=7168 or not 0<=r['arrival_s']<binding['duration_s'] for r in rows)):
            raise ValueError('invalid frozen evaluation requests')
        shapes=Counter((len(r['prompt']),r['max_tokens']) for r in rows)
        decisions=[]
        for p in profiles:
            if p['model_id']!=binding['model_id']:continue
            low,high=p['prefill_bounds'];prefill=[(n,o,count) for (n,o),count in shapes.items() if not low<=n<=high]
            scenarios=[]
            for scenario in p['scenarios']:
                if scenario['batch'] not in batches:continue
                out=dict(batch=scenario['batch'],frequency=scenario['frequency'])
                for metric,key in (('timing','timing_intervals'),('energy','power_and_timing_intervals')):
                    uncovered=[];affected=0
                    for (n,o),count in shapes.items():
                        # Preserve the planner's conservative possible context
                        # queries, not an assertion about measured native steps.
                        gaps=missing_segments(n,n+o-1,scenario[key])
                        if gaps:affected+=count;uncovered.extend(gaps)
                    out[metric]=dict(affected_requests=affected,missing_context_intervals=union(uncovered))
                scenarios.append(out)
            decisions.append(dict(tp=p['tp'],pp=p['pp'],source=p['source'],
                prefill_missing_requests=sum(row[2] for row in prefill),
                prefill_missing_shapes=[dict(input_tokens=n,output_tokens=o,requests=count) for n,o,count in sorted(prefill)],
                scenarios=scenarios))
        if not decisions:raise ValueError('no independent PDBlend profile declared for evaluation model')
        traces.append(dict(binding=binding,requests=len(rows),unique_request_shapes=len(shapes),
            shapes=[dict(input_tokens=n,output_tokens=o,requests=count) for (n,o),count in sorted(shapes.items())],
            profiles=decisions))
    return dict(kind='frozen_evaluation_profile_demand',system='pdblend',seed=701,trace_index=dict(path=str(index.resolve()),sha256=sha(index)),
        profile_ledger=dict(path=str(profile_ledger.resolve()),sha256=sha(profile_ledger)),traces=traces,
        batches=batches,actual_scheduler_batches_known=False,formal_eligible=False,
        sampling_points_generated=False,limits=['Missing intervals are demand, not an instruction to sample every integer.',
            'Planner candidate batches are explicit; runtime native batch observations remain separate.',
            'Experimental short mixed components do not silently replace missing pure-decode profile coverage.',
            'Every evaluation request remains identical to the bound five-system trace.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace-index',type=Path,required=True);p.add_argument('--profile-ledger',type=Path,required=True)
    p.add_argument('--batches',type=int,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('new immutable demand output required')
    if not a.batches or len(set(a.batches))!=len(a.batches) or min(a.batches)<1:raise ValueError('unique positive planner batches required')
    result=build(a.trace_index,a.profile_ledger,batches=a.batches);a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(dict(out=str(a.out),traces=len(result['traces']),requests=sum(t['requests'] for t in result['traces'])),indent=2))


if __name__=='__main__':main()
