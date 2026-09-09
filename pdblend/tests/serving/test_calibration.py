import pytest

from ecopadg.serving.calibration import RateSearch, InvalidCalibrationEvidence


def result(attainment=1,complete=True):
    return dict(validity='ok' if complete else 'invalid_work',completed=64 if complete else 60,
                n_expected=64,slo_attainment=attainment)


def test_capacity_search_requires_complete_work_and_observed_infeasible_bracket():
    search=RateSearch(2)
    assert search.next_rate()==2
    assert search.record(2,result())
    assert search.next_rate()==4
    assert not search.record(4,result(.97))
    assert search.next_rate()==3
    assert not search.record(3,result(.98))
    assert search.next_rate()==2.5
    assert search.record(2.5,result())
    assert search.next_rate() is None
    assert not search.record(2.5,result(.95))  # longer independent confirmation fails
    assert search.lower==2 and search.upper==2.5


@pytest.mark.parametrize('invalid',[
    dict(validity='invalid_runtime',runtime_error='controller crashed'),
    dict(validity='invalid_power_source'),dict(validity='invalid_work',completed=60),
    dict(sampling_error='NVML read failed'),dict(runtime_error='writer failed'),
    dict(slo_attainment=float('nan')),dict(slo_attainment=None),dict(completed=0),
    dict(generated_tokens=127,expected_generated_tokens=128),
])
def test_invalid_evidence_never_creates_capacity_upper_or_allows_retry(invalid):
    search=RateSearch(1)
    assert search.record(1,result())
    with pytest.raises(InvalidCalibrationEvidence):search.record(2,dict(result(.9),**invalid))
    assert search.upper is None and search.lower==1
    assert search.observations[-1]['measurement_valid'] is False
    assert search.observations[-1]['feasible'] is None
    with pytest.raises(InvalidCalibrationEvidence):search.next_rate()
    with pytest.raises(InvalidCalibrationEvidence):search.record(1,result())


def admission_boundary(**changes):
    return dict(dict(validity='invalid_work',completed=63,n_expected=64,admission_rejections=1,
        slo_attainment=63/64,capacity_observation_valid=True,gpu_count=8,power_mode='instant',
        power_source_verified=True,runtime_error=None,sampling_error=None,
        generated_tokens=126,expected_generated_tokens=128),**changes)


def test_verified_admission_rejections_only_establish_infeasible_upper():
    search=RateSearch(1);assert search.record(1,result())
    assert not search.record(2,admission_boundary())
    assert search.lower==1 and search.upper==2
    assert search.observations[-1]['boundary_kind']=='explicit_admission_rejection'
    # A single rejected confirmation still fails equal-work delivery even if
    # its larger denominator leaves joint attainment above the 99% target.
    assert not search.record(1.5,admission_boundary(completed=255,n_expected=256,slo_attainment=255/256))
    assert search.lower==1 and search.upper==1.5


@pytest.mark.parametrize('change',[
    dict(capacity_observation_valid=False),dict(admission_rejections=0),dict(completed=62),
    dict(power_mode='average'),dict(power_source_verified=False),dict(gpu_count=4),
    dict(runtime_error='HTTP 500'),dict(sampling_error='counter stale'),dict(slo_attainment=None),
])
def test_unknown_or_unmeasured_rejection_cannot_be_capacity_evidence(change):
    search=RateSearch(1)
    with pytest.raises(InvalidCalibrationEvidence):search.record(1,admission_boundary(**change))
    assert search.upper is None and search.lower==0


def test_before_cell_restores_every_probe_and_confirmation_outside_serving_window(tmp_path,monkeypatch):
    import asyncio
    import json
    from types import SimpleNamespace
    from ecopadg.serving import calibration
    corpus=tmp_path/'corpus';corpus.mkdir()
    (corpus/'alpaca.json').write_text(json.dumps(dict(calibration=[dict(input_tokens=1,output_tokens=1)]*128)))
    config=tmp_path/'config.json';config.write_text(json.dumps(dict(strategy='dynamollm')))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(corpus=str(corpus),max_trials=2,
        entries=[dict(dataset='alpaca',config=str(config),initial_rate=1)])))
    monkeypatch.setattr(calibration,'freeze_files',lambda files:{'source':'unchanged'})
    monkeypatch.setattr(calibration,'make_trace',lambda records,rate,seed,**kwargs:dict(rate=rate))
    calls=[]
    async def before(options):calls.append(('prepare',options.out.name))
    async def cell(options):
        calls.append(('serve',options.out.name))
        rate=json.loads(options.trace.read_text())['rate']
        return dict(validity='ok',completed=128,n_expected=128,slo_attainment=1 if rate==1 else .9)
    monkeypatch.setattr(calibration,'run_cell',cell)
    answer=asyncio.run(calibration.calibrate(SimpleNamespace(manifest=manifest,out=tmp_path/'out'),before_cell=before))
    assert answer['passed']
    assert [kind for kind,name in calls]==['prepare','serve']*3
    assert all(calls[i][1]==calls[i+1][1] for i in range(0,len(calls),2))
    assert answer['results'][0]['infeasible_upper_observation']['rate']==2


@pytest.mark.parametrize('failure_kind',['returned-invalid','raised-exception'])
def test_calibration_persists_failure_and_never_confirms_after_bad_measurement(tmp_path,monkeypatch,failure_kind):
    import asyncio
    import json
    from types import SimpleNamespace
    from ecopadg.serving import calibration
    corpus=tmp_path/'corpus';corpus.mkdir()
    (corpus/'alpaca.json').write_text(json.dumps(dict(calibration=[dict(input_tokens=1,output_tokens=1)]*128)))
    config=tmp_path/'config.json';config.write_text(json.dumps(dict(strategy='dynamollm')))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(corpus=str(corpus),max_trials=7,
        entries=[dict(dataset='alpaca',config=str(config),initial_rate=1)])))
    monkeypatch.setattr(calibration,'freeze_files',lambda files:{'source':'unchanged'})
    monkeypatch.setattr(calibration,'make_trace',lambda records,rate,seed,**kwargs:dict(rate=rate))
    calls=[]
    async def cell(options):
        calls.append(options.out.name)
        if len(calls)==1:return result()
        if failure_kind=='raised-exception':raise RuntimeError('engine disconnected')
        return dict(result(),validity='invalid_runtime',runtime_error='controller failed')
    monkeypatch.setattr(calibration,'run_cell',cell)
    with pytest.raises((RuntimeError,InvalidCalibrationEvidence)):
        asyncio.run(calibration.calibrate(SimpleNamespace(manifest=manifest,out=tmp_path/'out')))
    answer=json.loads((tmp_path/'out/summary.json').read_text())
    assert len(calls)==2 and all('probe' in name for name in calls)
    assert not answer['passed'] and answer['failure']
    row=answer['results'][0]
    assert not row['passed'] and row['capacity_rps'] is None and row['confirmation'] is None
    assert row['infeasible_upper_rps'] is None and row['infeasible_upper_observation'] is None
