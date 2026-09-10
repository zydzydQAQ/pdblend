import copy
from types import SimpleNamespace
import pytest
from meter_evidence import attach


@pytest.mark.parametrize('original_valid', [True, False])
def test_only_evidence_added(original_valid):
    raw=dict(measurement_valid=original_valid,energy_j=12.5,measurement_start_s=1,
             measurement_end_s=2,artifacts={'power': 'sha'},power_evidence={'passed':True})
    before=copy.deepcopy(raw)
    hooks=SimpleNamespace(completed_artifacts=lambda *a: {'worker':'digest'},
                          sampler_references=lambda *a:[{'raw':'actual'}])
    result=attach(raw,SimpleNamespace(_directory='dir'),{}, {'path':'adapter'},hooks)
    assert all(result[k]==v for k,v in before.items() if k!='artifacts')
    assert result['artifacts']=={'power':'sha','worker':'digest'}
    assert result['isolated_samplers']==[{'raw':'actual'}]


def test_incomplete_worker_invalidates_preserving_original_numbers(tmp_path):
    def fail(*a): raise RuntimeError('missing footer')
    hooks=SimpleNamespace(completed_artifacts=fail)
    raw=dict(measurement_valid=True,energy_j=12.5,artifacts={})
    result=attach(raw,SimpleNamespace(_directory=tmp_path),{}, {},hooks)
    assert result['measurement_valid'] is False and result['energy_j']==12.5
    assert 'missing footer' in result['isolated_sampler_evidence_error']
