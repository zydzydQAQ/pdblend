import asyncio, json
from pdblend_baselines.ecoserve.mechanism_four import run
def test_four_driver_reports_inconclusive(tmp_path):
    out=tmp_path/'a.json'; result=asyncio.run(run({'instances':[],'eco_prefill_csv':str(tmp_path/'missing.csv'),'slo_ttft_s':1,'slo_tpot_s':1},{},str(out)))
    assert result['status']=='inconclusive'; assert result['missing_required_actions']; assert result['formal_eligible'] is False
