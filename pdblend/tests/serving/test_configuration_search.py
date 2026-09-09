from dataclasses import asdict,replace
import json

import pytest

from ecopadg.serving.configuration_search import calibration_shape,candidates
from ecopadg.serving import configuration_search
from ecopadg.serving.interconnect import InterconnectTopology
from ecopadg.serving.profiles import ProfilePoint,ProfileStore
from test_planner import system


def test_baseline_candidate_shapes_ignore_formal_and_development_workloads():
    corpus=dict(calibration=[dict(input_tokens=128,output_tokens=64)]*20,
                development=[dict(input_tokens=999999,output_tokens=999999)],formal={})
    shape=calibration_shape(corpus)
    assert shape['input_tokens']==128 and shape['output_tokens']==64
    planner,_,_=system()
    result=candidates(planner.profiles,planner.transfers,None,shape,{1:100000,2:100000},2,.1,1)
    assert result['distserve'] and result['mixed']
    assert {c['tp'] for c in result['mixed']}=={1}  # no synthetic TP2 measurement


def mixed_point(context,batch,prefill,iteration,source,error=0):
    return ProfilePoint('mixed',1,2520,128,context,batch,prefill,iteration,
                        200,30,error,1,source)


def mixed_candidates(points,ttft=.5,tpot=.1):
    return candidates(ProfileStore(points),[],None,dict(input_tokens=128,output_tokens=64),
                      {1:100000},ttft,tpot,0)['mixed']


def test_mixed_search_uses_online_short_context_prefill_even_when_decode_uses_old_bucket():
    fresh=mixed_point(129,1,1,.01,'fresh-prefill')
    old=mixed_point(640,1,.1,.01,'old-decode-context')
    assert not mixed_candidates([fresh,old])
    assert mixed_candidates([old])


@pytest.mark.parametrize('old_context',[256,640])
def test_mixed_search_cannot_replace_selected_decode_bucket_with_faster_old_point(old_context):
    prefill=mixed_point(129,1,.05,.01,'prefill')
    actual=mixed_point(256,4,.1,.2,'fresh-specific-decode')
    bypass=replace(actual,context_tokens=old_context,iteration_s=.01,source_sha256='old-faster')
    assert not mixed_candidates([prefill,actual,bypass])
    assert mixed_candidates([prefill,bypass])


def test_mixed_capacity_uses_two_selected_phase_buckets_and_their_own_envelopes():
    prefill=mixed_point(129,1,.2,.01,'prefill',error=.1)
    decode=mixed_point(512,4,.01,.03,'decode',error=.2)
    bypass=mixed_point(1024,4,.001,.001,'larger-faster')
    choices=mixed_candidates([prefill,decode,bypass])
    one=next(c for c in choices if c['instance_count']==1)
    assert one['batch']==4 and one['source_sha256']=='decode'
    assert one['prefill_source_sha256']=='prefill'
    assert one['capacity_rps']==pytest.approx(4/(4*.2*1.1+63*.03*1.2))


@pytest.fixture
def search_cli(tmp_path,monkeypatch):
    """Synthetic files exercise the real CPU CLI, never hardware evidence."""
    def save(name,value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return path
    planner,_,_=system()
    profiles=save('profiles.json',dict(schema=2,measurement='hardware',
        points=[asdict(p) for p in planner.profiles.points]))
    topology_path=tmp_path/'interconnect.txt'
    topology_path.write_text('\n'.join('GPU'+str(i)+' '+
        ' '.join('X' if i==j else 'PIX' for j in range(8)) for i in range(8)))
    topology=InterconnectTopology.parse(topology_path.read_text())
    transfer=replace(planner.transfers[0],source_gpus=(0,),target_gpus=(1,),
        interconnect_class='PIX',topology_sha256=topology.source_sha256)
    transfers=save('transfers.json',dict(certified=True,instant_power_costs_verified=True,
        receiver_transfer_energy_included=True,links=[asdict(transfer)]))
    capacity=save('capacity.json',dict(complete=True,errors=[],instances=[
        dict(spec=dict(tp=1),measured_kv_capacity=100000)]))
    for dataset in ('alpaca','sharegpt','longbench'):
        save('corpus/'+dataset+'.json',dict(calibration=[dict(input_tokens=128,output_tokens=64)]))
    manifest=save('manifest.json',dict(profiles=str(profiles),transfers=str(transfers),
        interconnect=str(topology_path),capacity_measurements=[str(capacity)],
        corpus=str(tmp_path/'corpus'),slo_ttft_s=2,slo_tpot_s=.1))
    out=tmp_path/'search.json'
    monkeypatch.setattr('sys.argv',['configuration_search','--manifest',str(manifest),'--out',str(out)])
    return transfers,out


@pytest.mark.parametrize('field',[
    'certified','instant_power_costs_verified','receiver_transfer_energy_included'])
@pytest.mark.parametrize('value',['missing',False,None,0,1,'true','false',[],{},[True]])
def test_search_cli_rejects_missing_false_and_nonboolean_transfer_certification(search_cli,field,value):
    path,out=search_cli;bundle=json.loads(path.read_text())
    if value=='missing':bundle.pop(field)
    else:bundle[field]=value
    path.write_text(json.dumps(bundle))
    with pytest.raises(ValueError,match=field):configuration_search.main()
    assert not out.exists()


def test_search_cli_accepts_only_explicit_receiver_inclusive_instant_certification(search_cli):
    _,out=search_cli
    configuration_search.main()
    result=json.loads(out.read_text())
    assert result['status']=='predicted_candidates_only'
    assert set(result['datasets'])=={'alpaca','sharegpt','longbench'}
    assert all(row['distserve'] and row['mixed'] for row in result['datasets'].values())
