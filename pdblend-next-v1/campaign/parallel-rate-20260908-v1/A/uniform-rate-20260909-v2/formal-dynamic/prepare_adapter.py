"""Bind one uniform-v2 Alpaca observation to fresh native capacity ownership."""
import copy,hashlib,sys
from pathlib import Path
F=Path(__file__).resolve().parent;H=F.parent/'calibration-supplement';U=F.parents[1]/'uniform-rate-20260909-v1'
sys.path.insert(0,str(U/'dynamic-producer'))
import fresh_support as f

def prepare_dispatch(*,rows,**kwargs):
 f.need(len(rows)==1 and rows[0]['system']=='pdblend' and rows[0]['dataset']=='alpaca' and rows[0]['model']=='14b' and rows[0]['node']=='Anew20260909','one new uniform-v2 Alpaca cell required')
 sys.path.insert(0,str(F));verifier=f.load(F/'verify.py','newA_uniform2_cell_verifier')
 parent_ref=kwargs['qualification'];proof=verifier.verify(parent_ref);parent=f.checked(parent_ref);binding=f.checked(proof['binding'])
 row=rows[0];root=Path(kwargs['out']).resolve().parent/'dynamic-cell';f.need(not root.exists(),'fresh cell owner required')
 measurement=Path(kwargs['out']).resolve().parent/'measurement';owner='uniformcap'+hashlib.sha256(row['cell_id'].encode()).hexdigest()[:12]
 runtime=str(root/'runtime');inventory=str(root/'inventory.json');journal=str(root/'unused-controller.jsonl')
 config=f.read(binding['configs']['alpaca']);capacity=f.read(config['capacity_binding_path'])
 capacity.update(runtime_dir=runtime,owner_id=owner,max_creations=8);capref=f.save(root/'capacity-binding.json',capacity)
 config.update(journal=journal,capacity_binding_path=capref['path'],capacity_binding_sha256=capref['sha256'],capacity_inventory_path=inventory,measurement_window_protocol='per-dataset-slo-five-system-fixed-window-v1',arrival_window_s=100.)
 cfgref=f.save(root/'config.json',config);manifest=f.ref(f.ROOT/'common/token-evidence-v2/manifest.json');collector=f.checked(manifest)['collector'];equivalence=f.ref(F/'source-equivalence.json')
 files=dict(binding['files']);files.update(parent['files']);files.update(parent['source_files']);files.update({str(p):f.sha(p) for p in F.glob('*.py')})
 for r in [parent_ref,capref,cfgref,manifest,collector,equivalence,f.ref(H/'verify_v2.py')]:f.add(files,r)
 binding.update(configs=dict(alpaca=cfgref['path']),files=files,output=str(measurement),token_evidence_collector=collector,formal_source_equivalence=equivalence)
 bref=f.save(root/'binding.json',binding)
 qref=f.save(root/'qualified.json',dict(schema='new-A-dynamic-formal-cell-qualification-v2',parent_qualification=parent_ref,parent_validator=f.ref(H/'verify_v2.py'),node='Anew20260909',model='14b',dataset='alpaca',cell_id=row['cell_id'],binding=bref,config=cfgref,capacity=capref,measurement_output=str(measurement),owner_id=owner,runtime_dir=runtime,inventory_path=inventory,journal=journal,collector_manifest=manifest,collector=collector,source_equivalence=equivalence,files=files,source_files={str(F/'verify.py'):f.sha(F/'verify.py')}))
 kwargs.update(qualification=qref,qualification_validator=f.ref(F/'verify.py'),measurement_executor=f.ref(F/'dynamic_measurement.py'))
 shared=f.ROOT/'common/uniform-rate-20260909-v2';sys.path.insert(0,str(shared));prepare=f.load(shared/'prepare_release.py','newA_uniform2_original_prepare')
 return prepare.prepare(**kwargs)
