"""Offline, calibration-only baseline configuration candidates, never results.

Predicted candidates order independent GPU calibration. They cannot certify
SLO capacity or enter formal energy statistics without executed confirmations.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics

from .baselines import DistServeSearch,online_profile_points
from .evidence import sha256
from .interconnect import InterconnectTopology
from .planner import TransferCost
from .profiles import ProfileStore


def calibration_shape(corpus,quantile=.9):
    records=corpus['calibration']
    if not records or not 0<quantile<=1: raise ValueError('nonempty calibration split and valid quantile required')
    def q(key):
        values=sorted(r[key] for r in records)
        return values[min(len(values)-1,int(quantile*len(values)))]
    return dict(input_tokens=q('input_tokens'),output_tokens=q('output_tokens'),
        mean_input_tokens=statistics.mean(r['input_tokens'] for r in records),
        mean_output_tokens=statistics.mean(r['output_tokens'] for r in records),quantile=quantile)


def candidates(profiles,links,topology,shape,capacities,ttft,tpot,rate):
    spatial=DistServeSearch(profiles,links,topology=topology).search(
        shape['input_tokens'],shape['output_tokens'],ttft,tpot,rate,capacities)
    # Equivalent rank-locality placements remain in the full search evidence,
    # but do not require repeated startup for every GPU permutation.
    best={}
    for choice in spatial:
        key=(choice.prefill_tp,choice.decode_tp,choice.prefill_count,choice.decode_count,
             choice.prefill_batch,choice.decode_batch)
        if key not in best or choice.transfer_upper_s<best[key].transfer_upper_s: best[key]=choice
    mixed=[]
    decode_points=online_profile_points(profiles,'mixed',shape['input_tokens'],
                                       shape['input_tokens']+shape['output_tokens'])
    for tp in sorted(capacities):
        # Admission queries the arriving prefill at batch one independently of
        # the resident decode batch/context. Use those same two phase buckets.
        prefill=profiles.lookup('mixed',tp,2520,shape['input_tokens'],shape['input_tokens']+1,1)
        if prefill is None: continue
        points=[p for p in decode_points if p.tp==tp
                and prefill.phase_time_bound('prefill')+p.iteration_s*p.bound<=ttft
                and p.iteration_s*p.bound<=tpot
                and p.batch*(shape['input_tokens']+shape['output_tokens'])<=capacities[tp]]
        if not points: continue
        def service(point):
            return (point.batch*prefill.phase_time_bound('prefill')+
                    max(shape['output_tokens']-1,0)*point.iteration_s*point.bound)
        points=[p for p in points if service(p)>0]
        if not points: continue
        for count in range(1,8//tp+1):
            point=max(points,key=lambda p:p.batch/service(p))
            capacity=count*point.batch/service(point)
            if capacity>=rate:
                mixed.append(dict(tp=tp,instance_count=count,batch=point.batch,capacity_rps=capacity,
                    gpus=[list(range(j*tp,(j+1)*tp)) for j in range(count)],source_sha256=point.source_sha256,
                    prefill_source_sha256=prefill.source_sha256))
    return dict(distserve=[asdict(c) for c in sorted(best.values(),key=lambda c:(-c.capacity_rps,sum(map(len,c.gpus))))],
                mixed=sorted(mixed,key=lambda c:(-c['capacity_rps'],c['tp']*c['instance_count'])),
                limitations=['shape quantiles are offline calibration-only summaries',
                    'batch operator capacity omits arrival variance and interference; hardware calibration required',
                    'mixed layouts also seed EcoServe and DynamoLLM; their distinct policies require independent calibration'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();manifest=json.loads(args.manifest.read_text())
    if args.out.exists(): parser.error('refusing to overwrite search evidence')
    profiles=ProfileStore.load(manifest['profiles'])
    bundle=json.loads(Path(manifest['transfers']).read_text())
    required=('certified','instant_power_costs_verified','receiver_transfer_energy_included')
    missing=[field for field in required if bundle.get(field) is not True]
    if missing:
        raise ValueError('configuration search needs certified instant transfer costs including receiver energy: '+
                         ', '.join(missing))
    links=[TransferCost(**t) for t in bundle['links']]
    topology=InterconnectTopology.parse(Path(manifest['interconnect']).read_text())
    capacities={};artifacts={}
    for path in manifest['capacity_measurements']:
        startup=json.loads(Path(path).read_text())
        if not startup.get('complete') or startup.get('errors'): raise ValueError('incomplete capacity measurement')
        artifacts[str(Path(path).resolve())]=sha256(path)
        for instance in startup['instances']:
            tp=instance['spec']['tp'];capacity=instance['measured_kv_capacity']
            capacities[tp]=min(capacities.get(tp,capacity),capacity)
    results={}
    for dataset in ('alpaca','sharegpt','longbench'):
        path=Path(manifest['corpus'])/(dataset+'.json')
        shape=calibration_shape(json.loads(path.read_text()),manifest.get('shape_quantile',.9))
        artifacts[str(path.resolve())]=sha256(path)
        results[dataset]=dict(shape=shape,**candidates(profiles,links,topology,shape,capacities,
            manifest['slo_ttft_s'],manifest['slo_tpot_s'],manifest.get('minimum_rate',0.)))
    for field in ('profiles','transfers','interconnect'):
        path=Path(manifest[field]).resolve();artifacts[str(path)]=sha256(path)
    args.out.write_text(json.dumps(dict(status='predicted_candidates_only',datasets=results,
        measured_kv_capacity=capacities,artifacts=artifacts),indent=2,allow_nan=False))
    print(json.dumps({d:{k:len(r[k]) for k in ('distserve','mixed')} for d,r in results.items()}))


if __name__=='__main__': main()
