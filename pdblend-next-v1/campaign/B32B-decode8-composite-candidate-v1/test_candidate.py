import json
from pathlib import Path
import pytest
from ecopadg.serving.profiles import ProfileStore,ProfilePoint
from ecopadg.serving.planner import JointPlanner
from ecopadg.serving.tails import TailModel
from ecopadg.serving.types import InstanceState,RequestBudget,RuntimeSnapshot
ROOT=Path(__file__).resolve().parent
def store():return ProfileStore.load(ROOT/'profiles.json')
@pytest.mark.parametrize('freq',[1500,2520])
def test_decode_role_is_not_relabelled(freq):
 s=store();assert s.lookup('mixed',2,freq,128,640,8) is None
 p=s.lookup_execution_phase('mixed',2,freq,128,640,8)
 assert p.role=='decode' and p.batch==8 and p.prefill_s==0
 assert s.lookup('mixed',2,freq,128,129,1).role=='mixed'
@pytest.mark.parametrize('args',[(2,1500,513,768,8),(2,1500,512,769,8),(2,1500,512,768,9),(1,1500,128,640,8),(2,1800,128,640,8)])
def test_uncovered_region_rejected(args):assert store().lookup_execution_phase('mixed',*args) is None
def test_default_without_explicit_flag_unchanged():
 s=store();plain=ProfileStore(s.points)
 assert plain.lookup_execution_phase('mixed',2,1500,128,640,8) is None
def test_old_covered_mixed_keeps_precedence():
 s=store();assert s.lookup_execution_phase('mixed',2,1500,128,640,4) is s.lookup('mixed',2,1500,128,640,4)
def test_only_two_decode_points_added():
 old=json.loads((ROOT.parent/'baseline-raw-mirror-v1/B/root/workspace/pdblend/new-results/campaigns/node-b-v9/quick32-v1/profiles.provisional-tp2.json').read_text())
 new=json.loads((ROOT/'profiles.json').read_text());assert new['points'][:-2]==old['points']
 assert all(p['role']=='decode' and p['samples']==3 for p in new['points'][-2:])
def test_planner_and_tail_use_phase_lookup():
 s=store();p=JointPlanner(s,allow_pd=False)
 reqs=tuple(RequestBudget(str(i),0,128,211,1,.1,emitted=10,first_token_s=0) for i in range(7))
 inst=InstanceState('x','mixed',2,(0,1),0,1,1500,100000,7,0,reqs)
 new=RequestBudget('new',0,128,211,1,.1)
 assert p.point(inst,new,1500,8).role=='decode'
 tail=TailModel(p,RuntimeSnapshot(1,0,(inst,)),0)
 assert tail.point(inst,new,1500,8).role=='decode'
 assert tail.tails['x'] is not None
