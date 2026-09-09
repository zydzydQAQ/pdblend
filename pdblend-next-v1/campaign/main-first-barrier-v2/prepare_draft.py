"""Only C's actual9-success/null21-repair draft, never a host or global release."""
from pathlib import Path
import partition as p
P=Path(__file__).resolve().parent
C=P.parent
MIRROR=Path('/root/workspace/pdblend/new-results/campaigns/three-pool-v2/attribution-delivery-v1/C7B/frozen-inputs')
PATH_MAP={
 '/root/workspace/pdblend/new-results/campaigns/node-c-extensions-v1/quick-development-v1/profiles.development.json':str(MIRROR/'profiles.development.json'),
 '/root/workspace/pdblend/new-results/campaigns/node-c-extensions-v1/7b-qualification-v3/interconnect.txt':str(MIRROR/'interconnect.txt')}
PATH_MAP['/root/workspace/pdblend/new-results/campaigns/node-c-extensions-v1/7b-qualification-v3/clock-tp1/raw.json']=str(C/'C7B-source-evidence-mirror-v2/clock-tp1-raw.json')


def build(out):
    p.require(not out.exists(),'new draft directory required')
    bp=C/'C7B-baseline-main-first-v1/bindings/dynamollm-resident-resident/binding.json'
    source=C/'five-system-fixed-window-v1/sources/C7B/manifest.json'
    rows=[r for r in p.read(source)['cells'] if r['phase']=='main' and r['system']=='dynamollm']
    f=C/'C7B-main-after-dynamo-failure-v1/failure-boundary-proof.json';fr={'path':str(f),'sha256':p.sha(f)}
    reader=p.v1.Reader(PATH_MAP);failure=p.retained_failure(reader,fr)
    ids=[p.read(x)['row']['cell_id'] for x in failure['successful_original_checkpoints']]
    ref={'path':str(bp),'sha256':p.sha(bp)}
    g=dict(id='old-success9',system='dynamollm',datasets=sorted({r['dataset'] for r in rows if r['cell_id'] in ids}),binding=ref,policy_reference=ref,cell_ids=sorted(ids))
    compat,b=p.compatibility(g,reader)
    g['execution_source']=p.execution_source(g,b,reader,str(source),p.sha(source))
    records=[p.record(row,g,b,reader,[failure]) for row in rows if row['cell_id'] in ids]
    reader.stable();p.require(len(records)==9 and all(not r['execution']['invocation_succeeded'] for r in records),'actual old9 execution changed')
    new=dict(g,id='cooperative-pending21',binding=None,cell_ids=sorted(r['cell_id'] for r in rows if r['cell_id'] not in ids),
        expected_future_host=str(p.NEW_HOST),expected_future_host_manifest_sha256=p.NEW_MANIFEST)
    new.pop('execution_source');new['execution_manifest']=None
    draft=dict(schema=2,kind='partial-model-partition-draft-not-proof',model='7b',protocol_id=p.v1.PROTOCOL,deadline_s=p.v1.DEADLINE,
        source_manifest=str(source),source_sha256=p.sha(source),scope='C Dynamo30 only; the other120 original model cells must be provided before prove-main',
        groups=[g,new],retained_failure_refs=[fr],ready=False)
    p.partition_contract(draft['groups'],rows,require_bound=False)
    out.mkdir(parents=True)
    p.v1.write_new(out/'C-Dynamo-partition.draft.json',draft)
    p.v1.write_new(out/'actual-old9-validation.json',dict(cpu_only=True,gpu_executed=False,real_release_created=False,
        records=records,compatibility=compat,retained_failure=failure,input_files_sha256=reader.files,
        local_path_maps=PATH_MAP,future_binding_ready=False))
    return draft

if __name__=='__main__':
    import argparse,json
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);args=a.parse_args()
    d=build(args.out);print(json.dumps({'old_completed':len(d['groups'][0]['cell_ids']),'future_unbound':len(d['groups'][1]['cell_ids']),'ready':False}))
