import csv
import json

import pytest

from pdblend.results.catalog import comparison_key, normalize, collect, write_csv


def test_export_never_promotes_functional_success_and_preserves_missing_metrics(tmp_path):
    path=tmp_path/'completion.json';path.write_text('{}')
    row=normalize(dict(status='passed',complete=True,formal_eligible=False,
        metrics=dict(output_tokens=100,good_output_tokens=80),energy_j=200),path=path,root=tmp_path)
    assert row['formal_eligible'] is False and row['model_id']==''
    assert row['j_per_token']==2 and row['j_per_good_token']==2.5
    assert row['energy_total_j']=='' and row['energy_recorded_j']==200
    assert row['energy_scope']=='recorded_interval_not_proven_full_lifecycle'
    assert row['ttft_p99_s']=='' and row['energy_cold_start_j']==''
    with pytest.raises(ValueError):comparison_key(row)


def test_pairing_requires_exact_model_trace_slo_and_common_measurement_identity():
    from pdblend.results.catalog import PAIR_FIELDS
    row={key:'same' for key in PAIR_FIELDS};row.update(formal_eligible=True,evidence_status='current',
        offered_rps=1,seed=701,duration_s=300,slo_ttft_s=5,slo_tpot_s=.15)
    for field in ('model_id','trace_sha256','seed','energy_protocol'):
        other=dict(row);other[field]=1701 if field=='seed' else 'different'
        assert comparison_key(row)!=comparison_key(other)
    assert comparison_key(row)==comparison_key(dict(row,duration_s='300.0',offered_rps='1.000'))
    with pytest.raises(ValueError):comparison_key(dict(row,evidence_status='raw_pruned'))


def test_pruned_historical_csv_survives_refresh_and_attempts_stay_separate(tmp_path):
    root=tmp_path/'results';root.mkdir()
    write_csv(root/'runs.csv',[dict(run_id='a',attempt_id='retry-1',evidence_status='raw_pruned',formal_eligible=False),
                             dict(run_id='b',attempt_id='retry-2',evidence_status='raw_pruned',formal_eligible=False)])
    rows,errors=collect(root)
    assert len(rows)==2 and not errors
    assert {r['attempt_id'] for r in rows}=={'retry-1','retry-2'}
