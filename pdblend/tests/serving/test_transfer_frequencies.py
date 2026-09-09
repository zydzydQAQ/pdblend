from dataclasses import asdict,replace

import pytest
from ecopadg.serving.transfers import TransferCost,TransferStore
from ecopadg.serving.transfer_validation import merge_links
from ecopadg.serving.interconnect import InterconnectTopology
from ecopadg.serving.planner import JointPlanner
from ecopadg.serving.distserve import DistServeScheduler
from ecopadg.serving.profiles import ProfilePoint,ProfileStore
from ecopadg.serving.types import InstanceState,RequestBudget,RuntimeSnapshot


def costs():
    full=TransferCost(2,2,128,.12,3,'full',True,.08723,profile_batch=8,decode_frequency_mhz=2520)
    low=replace(full,seconds_upper=.23,incremental_j=2,import_seconds_upper=.18057,
                source_sha256='low',decode_frequency_mhz=900)
    return [low,full]


def test_exact_frequency_preserved_and_unmeasured_frequency_uses_independent_maxima():
    store=TransferStore(costs())
    def lookup(f): return store.lookup(2,2,(0,1),(2,3),128,8,f)
    assert lookup(2520).import_seconds_upper==.08723
    assert lookup(900).import_seconds_upper==.18057
    for frequency in (1500,2100,None):
        result=lookup(frequency)
        assert result.seconds_upper==.23 and result.import_seconds_upper==.18057
        assert result.incremental_j==3
        assert result.decode_frequency_mhz is None  # Do not invent a measured middle clock.


def test_bucket_placement_and_frequency_are_all_respected():
    low,full=costs()
    store=TransferStore([replace(full,max_input_tokens=2048,import_seconds_upper=.001),
                         low,full,replace(full,source_gpus=(6,7),target_gpus=(4,5),import_seconds_upper=9)])
    assert store.lookup(2,2,(0,1),(2,3),128,8,2520).import_seconds_upper==.08723
    assert store.lookup(2,2,(0,1),(2,3),128,9,2520) is None
    assert store.lookup(1,2,(0,),(2,3),128,8,2520) is None


def test_missing_closest_bucket_clock_cannot_borrow_faster_larger_bucket():
    low,full=costs()
    store=TransferStore([low,replace(full,max_input_tokens=2048)])
    result=store.lookup(2,2,(0,1),(2,3),128,8,2520)
    assert result.import_seconds_upper==.18057 and result.decode_frequency_mhz is None


def test_legacy_envelope_remains_conservative_and_import_compatible():
    from ecopadg.serving.planner import TransferCost as HistoricalImport
    assert HistoricalImport is TransferCost
    legacy=replace(costs()[0],decode_frequency_mhz=None)
    store=TransferStore([legacy])
    assert store.lookup(2,2,(0,1),(2,3),128,8,2520)==legacy


def fixture():
    profiles=ProfileStore([ProfilePoint('prefill',2,2520,128,129,8,.1,0,400,60,0,1,'p')]+
        [ProfilePoint('decode',2,f,128,1024,8,0,.05,300,60,0,1,'d') for f in (900,1500,2520)])
    old=RequestBudget('old',9,128,64,5,.1,emitted=2,first_token_s=10,last_token_s=10)
    request=RequestBudget('new',10,128,64,5,.1)
    p=InstanceState('p','prefill',2,(0,1),10,0,2520,20000,0,0)
    d=InstanceState('d','decode',2,(2,3),10,0,900,20000,1,0,requests=(old,),
        free_transfer_bytes=2**30,transfer_bytes_per_token=206848)
    return profiles,RuntimeSnapshot(1,10,(p,d)),request


def test_online_candidate_clock_reopens_only_full_frequency_admission():
    profiles,snapshot,request=fixture()
    planner=JointPlanner(profiles,costs())
    plans=planner.candidates(snapshot,request,10)
    assert len(plans)==1
    plan=plans[0]
    assert {action.frequency_mhz for action in plan.frequencies}=={2520}
    assert plan.routes[0].import_block_s==.08723
    predicted=planner.advance(snapshot,plan,request)
    target=predicted.instances[1]
    assert target.requests[-1].pending_import_s==.08723 and target.frequency_mhz==2520
    # The prior scalar worst-clock table wrongly excluded this same full-clock path.
    envelope=TransferStore(costs()).lookup(2,2,(0,1),(2,3),128,8)
    assert not JointPlanner(profiles,[envelope]).candidates(snapshot,request,10)


def test_distserve_and_pdblend_use_identical_full_frequency_transport_capability():
    profiles,snapshot,request=fixture()
    dist=DistServeScheduler(profiles,costs(),prefill_batch=8,decode_batch=8,clock_settle_s=0)
    fixed=JointPlanner(profiles,costs(),dvfs=False)
    baseline=dist.plan(snapshot,(request,),now=10)
    candidate=fixed.plan(snapshot,(request,),now=10)
    assert baseline.feasible and candidate.feasible
    assert baseline.routes==candidate.routes
    assert baseline.routes[0].import_block_s==.08723


def test_merge_keeps_repeated_placement_maximum_within_each_clock():
    topology=InterconnectTopology.parse('GPU0 X PIX\nGPU1 PIX X\n')
    full=replace(costs()[1],source_tp=1,target_tp=1,source_gpus=(0,),target_gpus=(1,),
        interconnect_class='PIX',topology_sha256=topology.source_sha256)
    low=replace(full,decode_frequency_mhz=900,import_seconds_upper=.18,incremental_j=2)
    repeated=replace(full,import_seconds_upper=.10,incremental_j=4,source_sha256='repeat')
    links,gaps=merge_links([(asdict(full),2520),(asdict(repeated),2520),(asdict(low),900)],topology)
    assert not gaps and len(links)==2
    values={r['decode_frequency_mhz']:r for r in links}
    assert values[2520]['import_seconds_upper']==.10 and values[2520]['incremental_j']==4
    assert values[900]['import_seconds_upper']==.18 and values[900]['incremental_j']==2
    with pytest.raises(ValueError,match='differs from its raw measurement'):
        merge_links([(asdict(full),900)],topology)
