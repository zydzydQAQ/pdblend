import csv
import json

import pytest

from pdblend.results.catalog import PAIR_FIELDS
from pdblend.results.plot import comparison_groups
from pdblend.results.power_archive import write_power_archive, read_power_archive
from pdblend.results.profile_points import export
from pdblend.measure.power import trapezoid_energy


def test_profile_points_keep_models_and_training_holdout_separate(tmp_path):
    for system,model in [('pdblend','7B'),('dynamollm','32B')]:
        folder=tmp_path/system;folder.mkdir()
        data=dict(system=system,model_id=model,tp=2,pp=1)
        if system=='pdblend':
            data.update(training={'a':dict(point=dict(role='decode',freq_mhz=900,batch=4),
                repeats=[dict(repeat=0,samples_file='sample.json',samples_sha256='s',
                              summary=dict(step_seconds=.012,mixed_power_w=231.25))])},holdout={})
        else:data['points']=[dict(role='mixed',frequency_mhz=2520,batch=1,iteration_s=.04)]
        (folder/'raw.json').write_text(json.dumps(data))
    report=export(tmp_path)
    assert report['rows']==2 and not report['errors']
    rows=list(csv.DictReader((tmp_path/'profile_points.csv').open()))
    assert {(r['system'],r['model_id']) for r in rows}=={('pdblend','7B'),('dynamollm','32B')}
    assert {r['phase'] for r in rows}=={'training','unspecified'}
    assert all(r['formal_eligible']=='False' for r in rows)


def test_plot_rejects_cross_model_identity_and_ambiguous_attempts():
    base={key:key for key in PAIR_FIELDS}
    base.update(formal_eligible='True',evidence_status='current',offered_rps=1,
                seed=701,duration_s=300,slo_ttft_s=5,slo_tpot_s=.15)
    rows=[dict(base,system='mixed'),dict(base,system='pdblend',model_id='different')]
    groups,excluded=comparison_groups(rows,['mixed','pdblend'])
    assert not groups and len(excluded)==2
    rows[1]['model_id']=rows[0]['model_id']
    assert len(comparison_groups(rows,['mixed','pdblend'])[0])==1
    assert not comparison_groups(rows+[dict(rows[0])],['mixed','pdblend'])[0]
    assert not comparison_groups([dict(row,evidence_status='raw_pruned') for row in rows],['mixed','pdblend'])[0]


def test_power_exact_roundtrip_source_change_and_energy(tmp_path):
    fixed=dict(gpus=[0,1],mode=['instant']*2,source_id=['nvml']*2,field_id=[186]*2,
               scope_id=[0]*2,value_type=[1]*2)
    value=dict(gpu_uuids=['GPU-a','GPU-b'],samples=[[1.123456,[123.456,234.567]],[2.987654,[124.444,232.777]]],
        frequency_samples=[[1.123456,[900,1200]]],utilization_samples=[],error=None,
        power_metadata=[dict(fixed,t_s=1.123456,nvml_timestamp_us=[1123000,1124000],return_code=[0,0],
                             read_started_s=[1.1,1.11],read_finished_s=[1.12,1.123456]),
                        dict(fixed,source_id=['nvml-v2']*2,t_s=2.987654,return_code=[0,0])])
    path=tmp_path/'power.json'
    manifest=write_power_archive(path,value)
    assert 'samples' not in manifest and len(manifest['power_source_epochs'])==2
    restored=read_power_archive(path)
    assert restored==value
    assert trapezoid_energy(restored['samples'])==trapezoid_energy(value['samples'])
    raw=tmp_path/manifest['raw_path'];raw.write_bytes(raw.read_bytes()[:-5])
    with pytest.raises(ValueError,match='binding'):
        read_power_archive(path)
    old=tmp_path/'old.json';old.write_text(json.dumps(value))
    assert read_power_archive(old)==value
