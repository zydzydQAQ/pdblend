"""CPU-only audit of retained scale work and immutable binding inventory."""
import argparse
import json
from pathlib import Path
import time
import contract as c

ROOT=Path(__file__).resolve().parent
BINDINGS={
 '14b':{'pdblend':'A14B-five-system100-v1/pdblend/binding.json',
         'mixed':'A14B-five-system100-baselines-v1/bindings/mixed-resident/binding.json',
         'dynamollm':'A14B-baseline-main-first-v1/bindings/dynamollm-resident/binding.json'},
 '7b':{'pdblend':'C7B-five-system100-v1/pdblend-r2/binding.json',
        'mixed':'C7B-five-system100-baselines-v1/bindings/mixed-resident/binding.json',
        'dynamollm':'C7B-baseline-main-first-v1/bindings/dynamollm-resident-resident/binding.json'},
 '32b':{'pdblend':'B32B-five-system100-v1/binding.pdblend.r2.json',
         'mixed':'B32B-baseline-sequence-v1/attempt-001/bindings/mixed/binding.json',
         'dynamollm':'B32B-baseline-main-first-sequence-v1/attempt-001/bindings/dynamollm-resident/binding.json'}
}

def main(out):
    c.require(not out.exists(),'new read-only snapshot required')
    inputs={};records=[];inventory={};started=time.time()
    def bind(path):
        inputs[str(path)]=c.sha(path);return c.read(path)
    for model,systems in BINDINGS.items():
        source=c.CAMPAIGN/'five-system-fixed-window-v1/sources'/({'14b':'A14B','7b':'C7B','32b':'B32B'}[model])/'manifest.json'
        manifest=bind(source);c.require(inputs[str(source)]==c.barrier.SOURCE_SHA[model],'original model source changed')
        rows={r['cell_id']:r for r in manifest['cells']};inventory[model]={}
        for system,relative in systems.items():
            path=c.CAMPAIGN/relative
            if not path.exists():inventory[model][system]={'binding':str(path),'state':'not_observed'};continue
            b=bind(path);ref={'path':str(path),'sha256':inputs[str(path)]}
            configs={}
            for ds,p in b['configs'].items():
                cfg=bind(Path(p));c.require(b['files'][p]==inputs[p],'bound config differs')
                configs[ds]=dict(path=p,sha256=inputs[p],strategy=cfg['strategy'],
                    topology_runtime_dir=(cfg.get('topology') or {}).get('runtime_dir'),engine_template=(cfg.get('topology') or {}).get('engine_template'))
            cps=list((Path(b['output'])/'checkpoints').glob('*.json'))
            main_cps=[p for p in cps if c.read(p)['row']['phase']=='main']
            scale_cps=[p for p in cps if c.read(p)['row']['phase']=='scale']
            inventory[model][system]=dict(main_binding=ref,output=b['output'],configs=configs,
                physical_instances=[dict(id=i['id'],tp=i['tp'],gpus=i['gpus'],container=i['container']) for i in b['instances']],
                observed_main_cp_count=len(main_cps),observed_scale_cp_count=len(scale_cps),
                main_count_scope='read-only inventory count only; release still requires full endpoint/global proof',
                future_scale_identity='reuse_only' if system=='pdblend' else 'not_yet_qualified; final physical group and fresh process gate required')
            if system=='pdblend' or (model=='7b' and system=='mixed'):
                for cp in sorted(scale_cps):
                    record=bind(cp);row=rows[record['row']['cell_id']]
                    proof=c.verify_existing(row,[ref],c.barrier.SOURCE_SHA[model])
                    for p,h in record['artifacts'].items():inputs[p]=h
                    inputs[proof['executing_invocation']]=proof['executing_invocation_sha256']
                    records.append({k:v for k,v in proof.items() if k!='actual_processes'}|dict(model=model,system=system,
                        actual_processes_sha256=c.barrier.digest_json(proof['actual_processes'])))
            for missing in ('distserve','ecoserve'):
                inventory[model].setdefault(missing,dict(state='future_actual_binding_not_fixed_by_this_preparation',
                    note='B Eco temporal failure must remain missing, never released' if model=='32b' and missing=='ecoserve' else 'actual per-group main binding comes from eventual root main proof'))
    c.require(sum(r['system']=='pdblend' for r in records)==54,'original PDB3x18 not fully mirrored')
    c.require(sum(r['model']=='7b' and r['system']=='mixed' for r in records)==2,'expected original two C Mixed scale records changed')
    for p,h in inputs.items():c.require(c.sha(p)==h,'input changed during read-only audit: '+p)
    out.mkdir(parents=True)
    c.barrier.write_new(out/'actual-validation.json',dict(schema=1,cpu_only=True,gpu_actions=False,started_s=started,finished_s=time.time(),
        verified_reused_scale_cells=len(records),records=records,input_files_sha256=inputs,
        limitations='Original raw SHA/metrics/actual config/terminal invocation/identity/native verified; no new GPU sampling or independent token rescore/power integration in this audit. No global main release created.'))
    c.barrier.write_new(out/'binding-inventory.json',dict(schema=1,generated_s=time.time(),inventory=inventory,
        pending_global_release=True,source_scope='observed local mirrored bindings and checkpoint inventory; no remote or active-state assertion'))
    print(json.dumps(dict(verified_reused_scale_cells=len(records),input_files=len(inputs),out=str(out))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
