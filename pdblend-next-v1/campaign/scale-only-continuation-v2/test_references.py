"""CPU synthetic reference counterexamples plus immutable actual raw regression."""
import ast
import copy
import json
import sys
from pathlib import Path
import pytest
import contract as c
import reference_map as m


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    return dict(path=str(path),sha256=c.sha(path))


def fixture(tmp_path,monkeypatch):
    row=dict(cell_id='scale',reuse_main_cell_id='main',model='7b',system='dynamollm',dataset='sharegpt',
        phase='scale',slo_scale=.5,slo_ttft_s=2.5,slo_tpot_s=.075,trace_sha256='d'*64,seed=701,n_requests=164)
    main=dict(row=dict(row,cell_id='main',phase='main',slo_scale=1.,slo_ttft_s=5.,slo_tpot_s=.15),
        binding='/old/actual/binding.json',binding_sha256='a'*64,output='/old/actual/output',
        execution=dict(invocation='/old/actual/invocation.json',invocation_sha256='b'*64,execution_manifest=dict(path='/subset',sha256='c'*64)))
    source=write(tmp_path/'source.json',dict(cells=[row]))
    current=write(tmp_path/'fresh/binding.json',dict(host_release='host'))
    group=dict(id='dynamo',system='dynamollm',datasets=['sharegpt'],scale_binding=current)
    spec=write(tmp_path/'spec.json',dict(model='7b',source=source,groups=[group]))
    checked=dict(current={},group=group,rows=[row],main_records=[main])
    monkeypatch.setattr(c,'check_spec',lambda *a,**k:dict(groups=[copy.deepcopy(checked)]))
    monkeypatch.setattr(m,'package_check',lambda:None)
    value=m.build(spec['path'],spec['sha256'],tmp_path/'not-an-actual-release','e'*64,'dynamo')
    ref=write(tmp_path/'map.json',value)
    return row,source,current,spec,checked,value,ref


@pytest.mark.parametrize('field',['model','system','phase','slo_scale','slo_ttft_s','slo_tpot_s','trace_sha256','seed','n_requests'])
def test_requested_wrong_row_rejected(tmp_path,monkeypatch,field):
    row,source,current,_,_,_,ref=fixture(tmp_path,monkeypatch)
    bad=copy.deepcopy(row);bad[field]='wrong'
    with pytest.raises(ValueError):m.verify(ref['path'],ref['sha256'],source['path'],current['path'],'dynamollm',['sharegpt'],row=bad)


@pytest.mark.parametrize('where',['source','producer','invocation','execution_source','main_slo','main_trace','output'])
def test_falsified_map_cannot_relabel_actual_producer(tmp_path,monkeypatch,where):
    row,source,current,_,_,value,ref=fixture(tmp_path,monkeypatch)
    original=value['references']['scale']['main']
    if where=='source':value['source']['sha256']='f'*64
    if where=='producer':original['binding']='/new/binding-that-did-not-execute'
    if where=='invocation':original['execution']['invocation_sha256']='f'*64
    if where=='execution_source':original['execution']['execution_manifest']['sha256']='f'*64
    if where=='main_slo':original['row']['slo_ttft_s']=10
    if where=='main_trace':original['row']['trace_sha256']='f'*64
    if where=='output':original['output']='/new/output-with-no-main'
    ref=write(Path(ref['path']),value)
    with pytest.raises(ValueError):m.verify(ref['path'],ref['sha256'],source['path'],current['path'],'dynamollm',['sharegpt'],row=row)


def test_map_read_only_no_checkpoint_copy(tmp_path,monkeypatch):
    row,source,current,_,_,_,ref=fixture(tmp_path,monkeypatch)
    before={str(p):c.sha(p) for p in tmp_path.rglob('*') if p.is_file()}
    value=m.verify(ref['path'],ref['sha256'],source['path'],current['path'],'dynamollm',['sharegpt'],row=row)
    assert value['references']['scale']['main']['output']=='/old/actual/output'
    assert before=={str(p):c.sha(p) for p in tmp_path.rglob('*') if p.is_file()}
    assert not (Path(current['path']).parent/'checkpoints').exists()


def test_driver_hardware_and_work_functions_exact_original():
    functions=lambda p:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    original,now=functions(c.EXECUTOR),functions(c.DRIVER)
    assert set(original)==set(now)
    assert [name for name in original if original[name]!=now[name]]==['sweep','main']
    assert (c.HERE/'child.py').read_bytes()==(c.EXECUTOR.parent/'child.py').read_bytes()


def test_actual_old9_failed_prefix_still_identifies_original_source():
    doc=c.read(c.CAMPAIGN/'main-first-barrier-v2/draft-001/actual-old9-validation.json')
    draft=c.read(c.CAMPAIGN/'main-first-barrier-v2/draft-001/C-Dynamo-partition.draft.json')
    group=draft['groups'][0];reader=c.barrier.Reader(doc['local_path_maps'])
    failure=c.revision.retained_failure(reader,doc['retained_failure']['reference'])
    binding=c.revision.ref(reader,group['binding'])
    for record in doc['records']:
        c.barrier.verify_record(record,reader)
        actual=c.revision.execution(record['row'],group,binding,reader,[failure],supplied=record['execution'])
        assert actual['invocation_succeeded'] is False and actual['host_release']==str(c.revision.OLD_HOST)
    reader.stable()
    assert len(doc['records'])==9 and draft['groups'][1]['binding'] is None


def test_actual_original_fullsource_main_producer():
    bp=c.CAMPAIGN/'B32B-five-system100-v1/binding.pdblend.r2.json';binding=c.read(bp)
    cp=next(c.read(p) for p in (Path(binding['output'])/'checkpoints').glob('*.json')
        if c.read(p)['row']['phase']=='main' and c.read(p)['row']['dataset']=='sharegpt' and c.read(p)['row']['rate_rps']==2.)
    source=c.CAMPAIGN/'five-system-fixed-window-v1/sources/B32B/manifest.json'
    group=dict(binding=dict(path=str(bp),sha256=c.sha(bp)),execution_source=dict(reference=dict(path=str(source),sha256=c.sha(source))))
    reader=c.barrier.Reader();actual=c.revision.execution(cp['row'],group,binding,reader,[]);reader.stable()
    assert actual['invocation_succeeded'] and actual['execution_manifest']==group['execution_source']['reference']
    assert c.read(cp['receipt'])['summary']['work_complete'] is False


def test_preexisting_partition_module_cannot_redirect_verifier(monkeypatch):
    sentinel=object();monkeypatch.setitem(sys.modules,'partition',sentinel)
    loaded=c.load('isolated_scale_contract_test',c.HERE/'contract.py')
    assert sys.modules['partition'] is sentinel
    assert Path(loaded.revision.__file__).resolve()==c.CAMPAIGN/'main-first-barrier-v2/partition.py'


@pytest.mark.parametrize('bad',['subset_as_current','missing_explicit','missing_history','bad_history_sha','none'])
def test_repaired_scale_requires_full_actual_source_and_preserves_main_subset(tmp_path,bad):
    full=write(tmp_path/'full.json',dict(kind='CPU full fixture'))
    subset=write(tmp_path/'subset.json',dict(parent_manifest=full['path'],parent_manifest_sha256=full['sha256']))
    records=[dict(execution=dict(execution_manifest=x)) for x in (full,subset)]
    current=dict(execution_manifest=full,main_execution_manifests=[full,subset],files={x['path']:x['sha256'] for x in (full,subset)})
    if bad=='subset_as_current':current['execution_manifest']=subset
    if bad=='missing_explicit':current.pop('execution_manifest')
    if bad=='missing_history':current['main_execution_manifests']=[full]
    if bad=='bad_history_sha':current['files'][subset['path']]='f'*64
    if bad=='none':c.scale_source(current,full,records,repaired=True)
    else:
        with pytest.raises(ValueError):c.scale_source(current,full,records,repaired=True)
