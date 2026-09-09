"""Small CPU contract tests. Synthetic publication fixtures exist only in tmp_path."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

S=importlib.util.spec_from_file_location('tested_main_barrier',Path(__file__).with_name('barrier.py'))
b=importlib.util.module_from_spec(S);S.loader.exec_module(b)

def fake_proof(model):
    path=b.facts().DEFAULT_SOURCES[model][0]
    source=b.read(path);rows=b.source_rows(source,model)
    groups=[]
    for system in b.SYSTEMS:
        ds=[['alpaca','sharegpt'],['longbench']] if model=='14b' and system=='distserve' else [list(b.DATASETS)]
        for selected in ds:
            groups.append(dict(system=system,datasets=selected,binding=f'/CPU-FIXTURE/{model}/{system}/{selected[0]}.json',
                binding_sha256='a'*64,terminal_invocations={'/CPU-FIXTURE/invocation.json':'b'*64}))
    records=[]
    for row in rows:
        g=next(g for g in groups if g['system']==row['system'] and row['dataset'] in g['datasets'])
        records.append(dict(row=row,checkpoint='/CPU-FIXTURE/'+row['cell_id']+'.json',checkpoint_sha256='c'*64,
            receipt_sha256='d'*64,config='/CPU-FIXTURE/config.json',config_sha256='e'*64,
            actual_config_sha256='f'*64,binding=g['binding'],binding_sha256=g['binding_sha256'],
            measurement_valid=True,work_complete=False,n_expected=row['n_requests'],good_requests=0,
            completed_work_requests=0,energy_j=999.,finished_s=b.DEADLINE-100))
    process=dict(hostname=b.HOSTS[model],no_live_serving_child=True,live_serving_children=[],observed_s=b.DEADLINE-100)
    evidence={str(path):b.SOURCE_SHA[model]}
    for g in groups:evidence.update(g['terminal_invocations'])
    for r in records:
        for key,hkey in (('checkpoint','checkpoint_sha256'),('binding','binding_sha256'),('config','config_sha256')):evidence[r[key]]=r[hkey]
        evidence[r['row']['trace']]=r['row']['trace_sha256']
    return dict(schema=1,kind='host-main-proof',model=model,hostname=b.HOSTS[model],protocol_id=b.PROTOCOL,
        deadline_s=b.DEADLINE,source_manifest=str(path),source_sha256=b.SOURCE_SHA[model],source_declaration=source,
        source_declaration_canonical_sha256=b.digest_json(source),baseline_systems=list(b.BASELINES),groups=groups,records=records,
        process_evidence=dict(before=process,after=copy.deepcopy(process)),created_s=b.DEADLINE-99,
        node_lease=dict(path=str(b.LEASE),exclusive=True,inherited=False),evidence_files=evidence)

def fake_release():
    models={m:fake_proof(m) for m in b.MODELS}
    return dict(schema=1,kind='global-main-release',protocol_id=b.PROTOCOL,deadline_s=b.DEADLINE,
        baseline_systems=list(b.BASELINES),models=models,coordinator_deep_verification=True,
        main_records=450,baseline_main_records=360,pdblend_main_records=90,created_s=b.DEADLINE-98,
        proof_refs={m:dict(sha256='1'*64,canonical_sha256=b.digest_json(p)) for m,p in models.items()})

def test_compact_explicit_trust_scope_retains_zero_good_incomplete(tmp_path):
    release=fake_release();p=tmp_path/'CPU-release.json';b.write_new(p,release)
    result=b.verify_release(p,b.sha(p))
    assert result['verified_scope']=='root_verified_release_sha_and_contract'
    assert release['models']['32b']['records'][0]['energy_j']==999.

@pytest.mark.parametrize('failure',['missing_model','missing_eco','missing_pdb','duplicate_record','scale_for_main',
    'wrong_deadline','hostname','source_sha','source_row','a_dist_layout','live_child','bad_binding','changed_embedded_proof'])
def test_partial_or_foreign_contract_is_rejected(tmp_path,failure):
    r=fake_release();p=r['models']['32b']
    if failure=='missing_model':r['models'].pop('7b')
    elif failure in ('missing_eco','missing_pdb'):
        system='ecoserve' if failure=='missing_eco' else 'pdblend'
        p['records']=[x for x in p['records'] if x['row']['system']!=system]
    elif failure=='duplicate_record':p['records'][-1]=copy.deepcopy(p['records'][0])
    elif failure=='scale_for_main':p['records'][0]['row']=dict(p['records'][0]['row'],phase='scale')
    elif failure=='wrong_deadline':r['deadline_s']+=1
    elif failure=='hostname':p['hostname']='CPU-wrong-host'
    elif failure=='source_sha':p['source_sha256']='0'*64
    elif failure=='source_row':p['records'][0]['row']=dict(p['records'][0]['row'],n_requests=1)
    elif failure=='a_dist_layout':
        p=r['models']['14b'];g=[g for g in p['groups'] if g['system']=='distserve'];g[0]['datasets']=list(b.DATASETS);p['groups'].remove(g[1])
    elif failure=='live_child':p['process_evidence']['after']['no_live_serving_child']=False
    elif failure=='bad_binding':p['records'][0]['binding_sha256']='0'*64
    else:p['created_s']-=1
    path=tmp_path/'CPU-invalid.json';b.write_new(path,r)
    with pytest.raises(ValueError):b.verify_release(path,b.sha(path))

def test_missing_or_unpinned_release_does_not_release(tmp_path):
    p=tmp_path/'missing.json'
    with pytest.raises(ValueError):b.verify_release(p,None)
    with pytest.raises(FileNotFoundError):b.verify_release(p,'a'*64)
    b.write_new(p,fake_release())
    with pytest.raises(ValueError):b.verify_release(p,'a'*64)

def test_synthetic_compact_proof_cannot_pass_deep_or_assemble(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'package_check',lambda:None)
    refs=[]
    for model in b.MODELS:
        p=tmp_path/(model+'.json');b.write_new(p,fake_proof(model));refs.append((p,b.sha(p)))
    with pytest.raises(FileNotFoundError):b.assemble_release(refs,tmp_path/'must-not-exist.json')
    assert not (tmp_path/'must-not-exist.json').exists()

def test_mapping_keeps_same_absolute_path_model_bytes_separate(tmp_path):
    a=tmp_path/'A.json';c=tmp_path/'C.json';a.write_text('{"model":"14b"}');c.write_text('{"model":"7b"}')
    original='/original/shared/model.json'
    assert b.Reader({original:str(a)}).read(original,b.sha(a))['model']=='14b'
    assert b.Reader({original:str(c)}).read(original,b.sha(c))['model']=='7b'
    with pytest.raises(ValueError):b.Reader({original:str(c)}).read(original,b.sha(a))

def test_no_overwrite(tmp_path):
    p=tmp_path/'immutable.json';b.write_new(p,{'old':1})
    with pytest.raises(FileExistsError):b.write_new(p,{'new':2})
    assert b.read(p)=={'old':1}

def test_expired_release_never_reopens_deadline(tmp_path,monkeypatch):
    p=tmp_path/'CPU-expired.json';b.write_new(p,fake_release())
    monkeypatch.setattr(b.time,'time',lambda:b.DEADLINE+1)
    with pytest.raises(ValueError,match='deadline elapsed'):b.verify_release(p,b.sha(p))

@pytest.mark.parametrize('failure',['live','wrong_source','scale','wrong_binding'])
def test_main_invocation_must_be_actual_bound_terminal(tmp_path,failure):
    p=tmp_path/'inv.json';v=dict(phase='main',system='mixed',selected_datasets=list(b.DATASETS),
        manifest_sha256=b.SOURCE_SHA['7b'],protocol_id=b.PROTOCOL,started_s=1,finished_s=2,complete=True,binding_sha256='a'*64)
    if failure=='live':v['finished_s']=None
    elif failure=='wrong_source':v['manifest_sha256']='0'*64
    elif failure=='scale':v['phase']='scale'
    else:v['binding_sha256']='0'*64
    b.write_new(p,v);g=dict(system='mixed',datasets=list(b.DATASETS),binding_sha256='a'*64,terminal_invocations={str(p):b.sha(p)})
    with pytest.raises(ValueError):b.verify_invocations(g,b.Reader(),b.SOURCE_SHA['7b'])

def test_process_scan_rejects_driver_and_child_but_not_idle_engine_or_supervisor(monkeypatch):
    class Proc:
        def __init__(self,pid,argv):self.parent=type('P',(),{'name':str(pid)})();self.data=b'\0'.join(x.encode() for x in argv)
        def read_bytes(self):return self.data
    entries=[Proc(10,['python3','-m','ecopadg.serving.engine']),
        Proc(11,['python3','/campaign/AC-baseline-main-first-sequence-v1/supervise.py']),
        Proc(12,['python3','/root/workspace/pdblend-next-v1/campaign/five-system-execution-v3/run.py','--run']),
        Proc(13,['python3','/root/workspace/pdblend-next-v1/campaign/five-system-execution-v3/child.py','/job.json'])]
    original=b.Path
    class Root:
        def glob(self,pattern):return entries
    monkeypatch.setattr(b,'Path',lambda p:Root() if p=='/proc' else original(p))
    scan=b.process_scan();assert not scan['no_live_serving_child'] and [x['pid'] for x in scan['live_serving_children']]==[12,13]
    entries[:]=entries[:2]
    assert b.process_scan()['no_live_serving_child']

def test_frozen_predicates_match_original_functions():
    import ast
    original=b.HERE.parent/'five-system-execution-v3/run.py'
    def function(path,name):
        source=path.read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
        return ast.get_source_segment(source,node)
    assert function(original,'barrier')==function(b.HERE/'native_barrier.frozen.py','barrier')
    assert b.sha(b.HERE/'point_evidence.frozen.py')==b.sha(b.HERE.parent/'five-system-results-v3/report.py')
