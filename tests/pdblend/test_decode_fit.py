from pathlib import Path
import pytest
from pdblend.profile.decode_fit import fit_candidate,predict
from pdblend.profile.model import PerfModel

def test_hinge_candidate_is_continuous_and_monotone():
 rows=[dict(batch=b,effective_context_tokens=1024,context_tokens=1024,step_seconds=.02+.0001*b+.000001*b*b) for b in (1,4,8,16,32,64,128)]
 spec=fit_candidate(rows,'hinges4_64',100000);vals=[predict(spec,b,1024) for b in (1,4,8,16,32,64,128)]
 assert all(x>0 for x in vals) and all(a<=b for a,b in zip(vals,vals[1:]));assert abs(predict(spec,4,1024)-predict(spec,4.000001,1024))<1e-5

def test_segmented_candidate_is_continuous_and_monotone_at_all_knots():
 rows=[dict(batch=b,effective_context_tokens=2048,context_tokens=2048,step_seconds=.02+.0001*b+.000001*b*b) for b in (1,4,16,64,80,96,160)]
 spec=fit_candidate(rows,'segments4_64_80_96_ctx',100000)
 vals=[predict(spec,b,2048) for b in (1,4,16,64,80,96,160)]
 assert all(x>0 for x in vals) and all(a<=b for a,b in zip(vals,vals[1:]))
 assert all(abs(predict(spec,b-1e-6,2048)-predict(spec,b+1e-6,2048))<1e-4 for b in (4,64,80,96))

def test_override_round_trip_and_old_profile_equivalence(tmp_path):
 old=PerfModel.load(Path('results/v2/profile-7b/profile.json'));path=tmp_path/'old.json';old.save(path);loaded=PerfModel.load(path)
 for b in (1,4,32,128):assert loaded.step_seconds(b,1024,900)==pytest.approx(old.step_seconds(b,1024,900))
 assert loaded.decode_overrides=={}

def test_unsupported_override_is_explicit():
 p=PerfModel.load(Path('results/2026-09-22/decode900/round1/profile.json'));assert p.decode_supported(160,1024,900);assert p.decode_supported(160,20000,900) is False
