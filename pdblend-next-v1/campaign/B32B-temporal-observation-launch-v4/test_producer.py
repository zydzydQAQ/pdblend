"""Six CPU counterexamples plus a split-invocation positive; no real proof."""
import json
from pathlib import Path
import pytest
import readiness_producer as p


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return p.sha(path)


@pytest.fixture
def data(tmp_path,monkeypatch):
    rows=[dict(cell_id=f'{s}-{i}',model='32b',system=s,dataset=('alpaca','sharegpt','longbench')[i%3],phase='main',slo_scale=1.)
        for s in p.SYSTEMS for i in range(30)]
    source=tmp_path/'source.json';digest=write(source,dict(model='32b',protocol_id=p.PROTOCOL,cells=rows));monkeypatch.setattr(p,'SOURCE_SHA',digest)
    proof=dict(model='32b',hostname=p.NODE,protocol_id=p.PROTOCOL,deadline_s=p.DEADLINE,source_manifest=str(source),source_sha256=digest,baseline_systems={})
    paths={}
    for group_index,system in enumerate(p.SYSTEMS):
        output=tmp_path/system;bp=output/'binding.json';binding=dict(model='32b',system=system,hostname=p.NODE,
            protocol_id=p.PROTOCOL,deadline_s=p.DEADLINE,output=str(output),files={str(source):digest})
        binding_sha=write(bp,binding);members=[r for r in rows if r['system']==system];headers=[]
        for i,row in enumerate(members):
            rp=output/'operations'/row['cell_id']/'receipt.json';rh=write(rp,dict(measurement_valid=True,child_stopped=True,child_pid=1000+group_index*30+i,started_s=20,finished_s=30))
            cp=output/'checkpoints'/(row['cell_id']+'.json');ch=write(cp,dict(row=row,receipt=str(rp),receipt_sha256=rh,measurement_valid=True,completed_s=31))
            headers.append(dict(cell_id=row['cell_id'],checkpoint=str(cp),checkpoint_sha256=ch,receipt_sha256=rh))
        a=output/'invocations/a.json';b=output/'invocations/b.json';skip=output/'invocations/skip.json'
        base=dict(phase='main',system=system,complete=True,started_s=10,finished_s=40,protocol_id=p.PROTOCOL,
            binding_sha256=binding_sha,manifest_sha256=digest,selected_datasets=['alpaca','sharegpt','longbench'])
        # Original Mixed's first prefix and resumed remaining prefix are each
        # actual execution. A later invocation which skips all CPs claims none.
        write(a,dict(base,pid=100+group_index*3,completed=[r['cell_id'] for r in members[:10]]))
        write(b,dict(base,pid=101+group_index*3,completed=[r['cell_id'] for r in members[10:]]))
        write(skip,dict(base,pid=102+group_index*3,completed=[],skipped=[r['cell_id'] for r in members]))
        paths[system]=(a,b,skip)
        proof['baseline_systems'][system]=dict(complete=True,completed=30,binding=str(bp),records=headers)
    pp=tmp_path/'cpu-fixture-not-real-proof.json';write(pp,proof)
    return pp,paths


def change(path,fn):
    value=json.loads(path.read_text());fn(value);write(path,value)


def test_split_invocations_and_skips_have90_unique_actual_producers(data):
    proof,paths=data;result=p.verify_main_producers(proof,live=lambda _:False)
    assert len(result['records'])==90 and result['actual_main_producers']==90
    assert all('/skip.json' not in r['invocation'] for r in result['records'])
    assert len({r['invocation'] for r in result['records'] if r['system']=='mixed'})==2


def test_wrong_binding_sha_rejected(data):
    proof,paths=data;change(paths['mixed'][0],lambda v:v.update(binding_sha256='0'*64))
    with pytest.raises(RuntimeError,match='binding/source'):p.verify_main_producers(proof,live=lambda _:False)


def test_wrong_execution_source_rejected(data):
    proof,paths=data;change(paths['mixed'][0],lambda v:v.update(manifest_sha256='0'*64))
    with pytest.raises(RuntimeError,match='binding/source'):p.verify_main_producers(proof,live=lambda _:False)


def test_skip_list_cannot_replace_missing_actual_completed(data):
    proof,paths=data;change(paths['mixed'][0],lambda v:v.update(completed=[]))
    with pytest.raises(RuntimeError,match='exactly one'):p.verify_main_producers(proof,live=lambda _:False)


def test_second_producer_not_chosen_by_better_result(data):
    proof,paths=data;change(paths['mixed'][2],lambda v:v.update(completed=['mixed-0']))
    with pytest.raises(RuntimeError,match='exactly one'):p.verify_main_producers(proof,live=lambda _:False)


def test_cp_outside_actual_invocation_time_rejected(data):
    proof,paths=data;change(paths['mixed'][0],lambda v:v.update(started_s=25))
    with pytest.raises(RuntimeError,match='interval'):p.verify_main_producers(proof,live=lambda _:False)


def test_dist_live_even_skip_only_invocation_cannot_release(data):
    proof,paths=data;change(paths['distserve'][2],lambda v:v.update(pid=420140))
    with pytest.raises(RuntimeError,match='main runner remains live'):
        p.verify_main_producers(proof,live=lambda pid:pid==420140)
