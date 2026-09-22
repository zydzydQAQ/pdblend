import json

from pdblend.control.planner import PlannerConfig, PoolPlanner, SLO
from pdblend.control.policies import get_policy
from pdblend.control.forecast import Forecast
from pdblend.profile.model import DecodePoint, PrefillPoint, StaticState, fit
from pdblend.profile.profiler import load_raw


def _static():
    return {"active_idle@2100": StaticState(70), "parked": StaticState(35, .03),
            "off": StaticState(35, 35)}


def test_unstable_b1_power_is_excluded_from_planner_fit():
    pre = [PrefillPoint(2100, n, .01 + n * .0001, 200) for n in (128, 512, 1024, 2048)]
    dec = [DecodePoint(2100, b, 1024, .02 + b * .0001, 180 + b * 2,
                      (150, 300, 180) if b == 1 else (180 + b * 2,) * 3)
           for b in (1, 4, 8, 16, 32)]
    model = fit(pre, dec, _static())
    q = model.quality["decode_power@2100"]
    assert q["excluded"] and q["excluded"][0]["batch"] == 1
    assert model.decode_power_w(16, 2100) < 230


def test_pdblend_empirical_floor_does_not_change_baseline_planner():
    model = fit(
        [PrefillPoint(2100, n, .01 + n * .0001, 200) for n in (128, 512, 1024, 2048)],
        [DecodePoint(2100, b, 1024, .02 + b * .0001, 180 + b * 2) for b in (1, 4, 8, 16, 32)],
        _static(), kv_capacity_tokens=200000)
    fc = Forecast(8.0, 0.0, 780.0, 1024.0, 300.0, 0, (780,), (300,))
    pd = PoolPlanner(model, get_policy("pdblend").planner_config(PlannerConfig(8, SLO(5, .15), freqs=(2100,))))
    baseline = PoolPlanner(model, PlannerConfig(8, SLO(5, .15), freqs=(2100,)))
    assert all(p.counts.get("M", 0) == 0 or p.counts.get("M", 0) >= 4 for p in pd.candidates(fc))
    assert any(p.counts.get("M", 0) in (1, 2, 3) for p in baseline.candidates(fc))


def test_profile_metadata_and_mixed_invalid_rows_round_trip(tmp_path):
    raw = dict(model="m", tp=1, freqs=[2100],
        prefill=[dict(freq_mhz=2100, input_tokens=128, seconds=.02, power_w=100)],
        decode=[dict(freq_mhz=2100, batch=1, context_tokens=256, step_seconds=.02, power_w=120)], mixed=[
        dict(freq_mhz=2100, batch=8, valid=False, invalid_reason="missing_base_step")],
        static={}, transfer=[], freq_switch_s=[], kv_capacity_tokens=1000,
        environment=dict(vllm="0.10.1.1", image_digest="sha256:test"))
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(raw))
    assert load_raw(path, "m", 1, 57344).freqs == (2100,)
