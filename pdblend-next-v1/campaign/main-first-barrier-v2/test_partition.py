"""CPU contracts: no fake real proof/release and no serving process."""
import copy
import json
from pathlib import Path
import pytest
import partition as p
import barrier as b
import prepare_draft as d

P=Path(__file__).resolve().parent
SOURCE=p.C/'five-system-fixed-window-v1/sources/C7B/manifest.json'
SOURCE_SHA=p.v1.SOURCE_SHA['7b']
BP=p.C/'C7B-baseline-main-first-v1/bindings/dynamollm-resident-resident/binding.json'
REF={'path':str(BP),'sha256':p.sha(BP)}
DRAFT=p.read(P/'draft-001/C-Dynamo-partition.draft.json')
VALIDATION=p.read(P/'draft-001/actual-old9-validation.json')
ROWS=[r for r in p.read(SOURCE)['cells'] if r['phase']=='main' and r['system']=='dynamollm']


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return {'path':str(path),'sha256':p.sha(path)}


def test_real_9_plus_future21_is_exact_not_ready():
    result=p.partition_contract(DRAFT['groups'],ROWS,require_bound=False)
    assert result==dict(cell_count=30,partitions=2,ready=False)
    with pytest.raises(ValueError,match='null/unready'):p.partition_contract(DRAFT['groups'],ROWS)
    assert len(DRAFT['groups'][0]['cell_ids'])==9 and len(DRAFT['groups'][1]['cell_ids'])==21


@pytest.mark.parametrize('kind',['overlap','missing','foreign','scale_row'])
def test_partition_never_double_counts_or_uses_scale(kind):
    g=copy.deepcopy(DRAFT['groups']);rows=copy.deepcopy(ROWS)
    if kind=='overlap':g[1]['cell_ids'][0]=g[0]['cell_ids'][0]
    if kind=='missing':g[1]['cell_ids'].pop()
    if kind=='foreign':g[1]['cell_ids'][0]='foreign-cell'
    if kind=='scale_row':rows[0]['phase']='scale'
    with pytest.raises(ValueError):p.partition_contract(g,rows,require_bound=False)


def test_partial30_cannot_be_full150():
    rows=p.v1.source_rows(p.read(SOURCE),'7b')
    with pytest.raises(ValueError):p.partition_contract(DRAFT['groups'],rows,require_bound=False)
    with pytest.raises(ValueError):b.proof_contract(DRAFT)


def test_actual_old9_remain_old_failed_invocation_success_prefix():
    reader=p.v1.Reader(d.PATH_MAP);failure=p.retained_failure(reader,DRAFT['retained_failure_refs'][0])
    group=DRAFT['groups'][0];compat,binding=p.compatibility(group,reader)
    assert compat['host']['kind']=='identical' and compat['host']['host']==str(p.OLD_HOST)
    for rec in VALIDATION['records']:
        p.v1.verify_record(rec,reader)
        got=p.execution(rec['row'],group,binding,reader,[failure],supplied=rec['execution'])
        assert got['invocation_succeeded'] is False and got['host_release']==str(p.OLD_HOST)
    reader.stable()
    assert failure['counts_as_completed'] is False and failure['full_operation_energy_j']>0
    assert len(failure['successful_original_checkpoints'])==9


def test_failed_invocation_without_bad_raw_cannot_qualify_prefix():
    r=VALIDATION['records'][0];g=DRAFT['groups'][0];binding=p.read(BP)
    with pytest.raises(ValueError,match='failed invocation needs'):
        p.execution(r['row'],g,binding,p.v1.Reader(d.PATH_MAP),[],supplied=r['execution'])


def test_old_cp_cannot_claim_new_host_or_invocation():
    r=VALIDATION['records'][0];g=DRAFT['groups'][0];reader=p.v1.Reader(d.PATH_MAP)
    failure=p.retained_failure(reader,DRAFT['retained_failure_refs'][0]);ex=copy.deepcopy(r['execution']);ex['host_release']=str(p.NEW_HOST)
    with pytest.raises(ValueError,match='compact per-cell execution/source'):
        p.execution(r['row'],g,p.read(BP),reader,[failure],supplied=ex)


def test_real_exact_single_yield_host_namespace():
    old=p.read(BP);new=copy.deepcopy(old);new['host_release']=str(p.NEW_HOST)
    for path,h in p.read(p.NEW_HOST/'manifest.json')['files'].items():new['files'][str(p.NEW_HOST/path)]=h
    result=p.host_contract(old,new,p.v1.Reader(d.PATH_MAP))
    assert result['kind']=='cooperative-admission-yield-v1' and result['changed_files']==[p.ADMISSION]
    assert result['manifest_sha256']==p.NEW_MANIFEST


def test_no_host_manifest_or_unreviewed_second_change():
    old=p.read(BP);new=copy.deepcopy(old);new['host_release']=str(p.NEW_HOST)
    with pytest.raises(ValueError,match='complete host source'):p.host_contract(old,new,p.v1.Reader(d.PATH_MAP))
    class Reader(p.v1.Reader):
        def read(self,path,expected=None):
            result=super().read(path,expected)
            if str(path)==str(p.NEW_HOST/'manifest.json'):
                result['files']['src/ecopadg/serving/runtime.py']='0'*64
            return result
    with pytest.raises(ValueError,match='only the exact reviewed'):p.host_contract(old,new,Reader(d.PATH_MAP))


def subset(tmp_path,mutation=None):
    group=copy.deepcopy(DRAFT['groups'][1]);binding=copy.deepcopy(p.read(BP));binding['host_release']=str(p.NEW_HOST)
    rows=[r for r in p.read(SOURCE)['cells'] if r['cell_id'] in group['cell_ids']]
    value=dict(model='7b',protocol_id=p.v1.PROTOCOL,parent_manifest=str(SOURCE),parent_manifest_sha256=SOURCE_SHA,cells=copy.deepcopy(rows))
    if mutation:mutation(value)
    ref=write(tmp_path/'subset.json',value);binding['files'][ref['path']]=ref['sha256'];group['execution_manifest']=ref
    return group,binding


def test_exact21_subset_keeps_original_source_and_rows(tmp_path):
    g,binding=subset(tmp_path)
    result=p.execution_source(g,binding,p.v1.Reader(),str(SOURCE),SOURCE_SHA)
    assert result['kind']=='exact-original-row-subset' and result['parent_manifest_sha256']==SOURCE_SHA
    assert len(result['original_cell_ids'])==21


@pytest.mark.parametrize('mutation',[lambda v:v['cells'].pop(),lambda v:v['cells'].append(v['cells'][0]),
    lambda v:v['cells'][0].update(slo_ttft_s=100),lambda v:v['cells'][0].update(n_requests=1),
    lambda v:v.update(parent_manifest_sha256='0'*64)])
def test_subset_cannot_change_work_slo_domain_or_parent(tmp_path,mutation):
    g,binding=subset(tmp_path,mutation)
    with pytest.raises(ValueError):p.execution_source(g,binding,p.v1.Reader(),str(SOURCE),SOURCE_SHA)


def test_repair_cannot_silently_rerun_full30(tmp_path):
    g,binding=subset(tmp_path);g.pop('execution_manifest')
    with pytest.raises(ValueError,match='whole-group rerun'):p.execution_source(g,binding,p.v1.Reader(),str(SOURCE),SOURCE_SHA)


def test_no_release_from_null_or_partial_proofs(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'package_check',lambda:None)
    out=tmp_path/'must-not-exist.json'
    with pytest.raises(ValueError):b.assemble_release([],out)
    assert not out.exists()
    with pytest.raises((ValueError,FileNotFoundError)):b.verify_release(out,'a'*64)
    assert not out.exists()


@pytest.mark.parametrize('change',[None,'prior','profile_bytes','budget'])
def test_actual_config_only_host_metadata_may_change(tmp_path,change):
    old=p.read(BP);new=copy.deepcopy(old);new['host_release']=str(p.NEW_HOST);new['output']=str(tmp_path/'new-results')
    for path,h in p.read(p.NEW_HOST/'manifest.json')['files'].items():new['files'][str(p.NEW_HOST/path)]=h
    for ds,path in old['configs'].items():
        config=copy.deepcopy(p.read(path));config['controller_source_release']=str(p.NEW_HOST)
        if change=='prior':config['prior_output_tokens']=999
        if change=='budget':config['max_num_batched_tokens']=16
        if change=='profile_bytes':new['files'][config['profiles']]='f'*64
        ref=write(tmp_path/'configs'/(ds+'.json'),config);new['configs'][ds]=ref['path'];new['files'][ref['path']]=ref['sha256']
    nr=write(tmp_path/'binding.json',new);g=dict(DRAFT['groups'][1],binding=nr)
    if change:
        with pytest.raises(ValueError):p.compatibility(g,p.v1.Reader(d.PATH_MAP))
    else:
        result,binding=p.compatibility(g,p.v1.Reader(d.PATH_MAP))
        assert result['host']['kind']=='cooperative-admission-yield-v1' and binding['instances']==old['instances']
