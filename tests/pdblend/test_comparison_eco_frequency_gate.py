"""Frequency invalidity cannot mask independent control or request corruption."""
import json
from pathlib import Path

import pytest

from pdblend.bench import comparison_ecoserve_acceptance as tp1
from pdblend.bench import comparison_ecoserve32_acceptance as tp2
from test_comparison_ecoserve import eco_fixture, mutate_events, save_change
from test_comparison_ecoserve32 import fixture as eco32_fixture


@pytest.mark.parametrize('topology',[1,2])
@pytest.mark.parametrize('problem',['none','outside_tolerance','missing','gap','nonfinite','boundary'])
def test_frequency_gate_preserves_original_tolerance_coverage_and_both_tp_ranks(tmp_path,monkeypatch,topology,problem):
    args=(eco_fixture if topology==1 else eco32_fixture)(tmp_path,monkeypatch)
    power=json.loads(Path(args['raw_refs']['power']['path']).read_text())
    rank=0 if topology==1 else 1
    if problem=='outside_tolerance':power['frequency_samples'][15][1][rank]=2489
    elif problem=='missing':power['frequency_samples']=[]
    elif problem=='gap':del power['frequency_samples'][15:18]
    elif problem=='nonfinite':power['frequency_samples'][15][1][rank]=None
    elif problem=='boundary':power['frequency_samples'][15][1][rank]=2490
    save_change(args,'power',power)
    result=(tp1 if topology==1 else tp2).audit_ecoserve_window(**args)
    if problem in ('none','boundary'):
        assert result['evidence_valid'] and not result['missing_gates'],result['gate_failures']
    else:
        assert not result['evidence_valid'] and not result['formal_eligible']
        assert result['missing_gates']==['eco.observed_active_frequency'],result['gate_failures']
        assert 'eco.raw_protocol_and_canonical_metrics' in result['checked_gates']


@pytest.mark.parametrize('topology',[1,2])
@pytest.mark.parametrize('problem',['http','admission','tokens'])
def test_frequency_and_protocol_failures_are_both_reported(tmp_path,monkeypatch,topology,problem):
    args=(eco_fixture if topology==1 else eco32_fixture)(tmp_path,monkeypatch)
    power=json.loads(Path(args['raw_refs']['power']['path']).read_text())
    power['frequency_samples'][15][1][0]=2430
    save_change(args,'power',power)
    def corrupt(rows):
        if problem=='http':
            next(r for r in rows if r['kind']=='eco_http_receipt')['response']['acknowledged']=False
        elif problem=='admission':
            next(r for r in rows if r['kind']=='eco_admission')['origin']='changed_policy'
        else:
            next(r for r in rows if r['kind']=='eco_native_sse')['payload']['token_ids']=[999]
    mutate_events(args,corrupt)
    result=(tp1 if topology==1 else tp2).audit_ecoserve_window(**args)
    assert set(result['missing_gates'])=={'eco.observed_active_frequency','eco.raw_protocol_and_canonical_metrics'}
    assert not result['evidence_valid'] and not result['formal_eligible']
