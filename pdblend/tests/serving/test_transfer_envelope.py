from dataclasses import asdict,replace

from ecopadg.serving.interconnect import InterconnectTopology
from ecopadg.serving.transfer_validation import merge_links
from ecopadg.serving.profiling import hold_setup_limit
from test_planner import system


def test_low_clock_transfer_cannot_inherit_fast_endpoint_cost():
    topology=InterconnectTopology.parse('GPU0 X PIX\nGPU1 PIX X\n')
    planner,_,_=system()
    full=replace(planner.transfers[0],source_gpus=(0,),target_gpus=(1,),
        interconnect_class='PIX',topology_sha256=topology.source_sha256,
        seconds_upper=.1,import_seconds_upper=.02,incremental_j=3)
    low=replace(full,seconds_upper=.3,import_seconds_upper=.15,incremental_j=2)
    links,gaps=merge_links([(asdict(full),2520),(asdict(low),900)],topology)
    assert not gaps
    by_frequency={row['decode_frequency_mhz']:row for row in links}
    assert by_frequency[900]['seconds_upper']==.3
    assert by_frequency[900]['import_seconds_upper']==.15
    assert by_frequency[900]['incremental_j']==2
    assert by_frequency[2520]['seconds_upper']==.1
    assert by_frequency[2520]['import_seconds_upper']==.02
    assert by_frequency[2520]['incremental_j']==3
    links,gaps=merge_links([(asdict(full),2520)],topology)
    assert not links and gaps[0]['missing_decode_frequencies']==[900]


def test_long_held_batch_timeout_is_excluded_without_inventing_capacity():
    runs=[dict(skipped=False,frequency_mhz=900,input_tokens=7168,layout='mixed',
               batch=8,started_s=0,released_s=36)]
    limited=hold_setup_limit(runs,900,7168,32,'mixed')
    assert limited['skipped'] and limited['not_a_serving_capacity_measurement']
    assert hold_setup_limit(runs,900,7168,16,'mixed') is None
    assert hold_setup_limit(runs,1500,7168,32,'mixed') is None
