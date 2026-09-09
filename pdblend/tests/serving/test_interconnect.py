from dataclasses import replace

from ecopadg.serving.interconnect import InterconnectTopology
from ecopadg.serving.baselines import DistServeSearch
from test_planner import system


def test_search_uses_only_measured_physical_link_classes():
    text='GPU0 X PIX SYS SYS\nGPU1 PIX X SYS SYS\nGPU2 SYS SYS X PIX\nGPU3 SYS SYS PIX X\n'
    topology=InterconnectTopology.parse(text)
    assert topology.link_class((0,),(1,))=='PIX'
    assert topology.link_class((0,1),(2,3))=='SYS'
    planner,_,_=system()
    link=replace(planner.transfers[0],source_gpus=(0,),target_gpus=(1,),
                 interconnect_class='PIX',topology_sha256=topology.source_sha256)
    assert link.matches_placement((2,),(3,),topology)
    assert not link.matches_placement((0,),(2,),topology)
    search=DistServeSearch(planner.profiles,[link],gpu_count=4,topology=topology)
    choices=search.search(128,64,2,.1,1,{1:100000})
    assert choices
    assert any(c.gpus==((2,),(3,)) for c in choices)
    assert all(all(topology.link_class(p,d)=='PIX'
                   for p in c.gpus[:c.prefill_count] for d in c.gpus[c.prefill_count:]) for c in choices)
    legacy=DistServeSearch(planner.profiles,planner.transfers,gpu_count=4,topology=topology)
    assert not legacy.search(128,64,2,.1,1,{1:100000})
