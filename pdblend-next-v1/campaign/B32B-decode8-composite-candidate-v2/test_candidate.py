import json
from pathlib import Path
import pytest
from ecopadg.serving.profiles import ProfileStore
ROOT=Path(__file__).resolve().parent
def store():return ProfileStore.load(ROOT/'profiles.json')
@pytest.mark.parametrize('frequency',[1500,2520])
def test_long_true_decode_and_short_points_preserved(frequency):
 s=store();assert s.lookup('mixed',2,frequency,4096,4352,8) is None
 d=s.lookup_execution_phase('mixed',2,frequency,4096,4352,8)
 assert d.role=='decode' and d.input_tokens==4096 and d.context_tokens==4352 and d.batch==8 and d.prefill_s==0
 short=s.lookup_execution_phase('mixed',2,frequency,128,640,8)
 prior=ProfileStore.load(ROOT.parent/'B32B-decode8-composite-candidate-v1/profiles.json')
 assert short==prior.lookup_execution_phase('mixed',2,frequency,128,640,8)
@pytest.mark.parametrize('args',[(2,1500,4097,4352,8),(2,1500,4096,4353,8),(2,1500,4096,4352,9),(1,1500,4096,4352,8)])
def test_unmeasured_domain_closed(args):assert store().lookup_execution_phase('mixed',*args) is None
def test_original_points_unchanged():
 a=json.loads((ROOT/'profiles.json').read_text());b=json.loads((ROOT.parent/'B32B-decode8-composite-candidate-v1/profiles.json').read_text())
 assert a['points'][:-2]==b['points'] and len(a['points'])==len(b['points'])+2
def test_default_host_behavior_remains_closed():
 s=store();plain=ProfileStore(s.points)
 assert plain.lookup_execution_phase('mixed',2,1500,4096,4352,8) is None
