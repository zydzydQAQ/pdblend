"""Freeze the declared 28 actual microbatches and 12 clock transitions; no GPU."""
import argparse,copy,json
from pathlib import Path
import validate as v
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent

def prepare(feasibility,out):
    feasibility=Path(feasibility).resolve();out=Path(out).resolve()
    v.require(not out.exists(),'new immutable microprofile spec required')
    manifest=v.read(HERE/'manifest.json');v.require(all(v.sha(p)==h for p,h in manifest['files'].items()),'profile driver changed')
    old=v.read(feasibility/'spec.json');evidence=v.read(feasibility/'evidence-manifest.json')
    v.require(evidence['passed'] and all(v.sha(p)==h for p,h in evidence['files'].items()),'feasibility raw evidence changed or failed')
    binding=v.fixed(old['binding']);profile=v.fixed(old['profile_reference'])
    spec=copy.deepcopy(old);files={**old['files'],**evidence['files'],**manifest['files']}
    references=[v.ref(HERE/'manifest.json'),v.ref(feasibility/'evidence-manifest.json'),v.ref(feasibility/'status.json'),v.ref(feasibility/'spec.json'),
        v.ref(HERE/'source-order-contract.json'),v.ref(Path(binding['host_release'])/'manifest.json'),
        v.ref(ROOT/'A/isolated-power-v2/manifest.json'),v.ref(ROOT/'A/dynamic-execution-isolated-power-002/sampler_hooks.py')]
    for reference in references:files[reference['path']]=reference['sha256']
    spec.update(schema='distributed14b-actual2400-profile-input-v1',authorized=True,points=v.declared_points(profile,binding),
        request_timeout_s=120,cleanup_timeout_s=120,frequency_tolerance_mhz=15,files=files,
        profile_publication_allowed=False,feasibility=v.ref(feasibility/'status.json'),feasibility_spec=v.ref(feasibility/'spec.json'),
        source_order_contract=v.ref(HERE/'source-order-contract.json'),host_manifest=v.ref(Path(binding['host_release'])/'manifest.json'),
        measurement_adapter=v.ref(ROOT/'A/isolated-power-v2/manifest.json'),measurement_hooks=v.ref(ROOT/'A/dynamic-execution-isolated-power-002/sampler_hooks.py'),
        samples_per_shape=2,samples_per_transition=2,transition_gpus=[6,7],
        transition_pairs=[[a,b] for low in (900,1500,2100) for a,b in ((low,2400),(2400,low))],
        empirical_not_future_guarantee=True,same_shape_mixed_only=True,new_cross_context_interference_points=0,
        original_performance_output_and_SLO_unchanged=True,automatic_retries=False)
    v.source_check(spec);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(spec,f,indent=2,allow_nan=False);f.write('\n')
    return v.ref(out)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--feasibility',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.feasibility,args.out)))
