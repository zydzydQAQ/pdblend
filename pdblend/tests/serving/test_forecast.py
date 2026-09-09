from dataclasses import replace

from ecopadg.serving.forecast import forecast_roles
from ecopadg.serving.types import RequestBudget


def test_role_forecast_requires_history_and_never_reads_future_or_realized_output():
    budgets=[RequestBudget(str(i),float(i),128+i*100,64,5,.1,output_limit=512) for i in range(60)]
    assert forecast_roles(budgets,20,0) is None
    first=forecast_roles(budgets,60,0)
    assert first and first.observations==60 and 0<first.rate_lower_rps<1
    future=replace(budgets[-1],arrival_s=61,predicted_output=500)
    assert forecast_roles(budgets+[future],60,0)==first
    assert len(first.requests)==3 and all(r.arrival_s==60 and r.predicted_output==64 for r in first.requests)
    assert len({r.request_id for r in first.requests})==3
    assert forecast_roles(budgets,200,0) is None
