import json
from pathlib import Path

import pytest

from ecopadg.serving.evidence import sha256
from ecopadg.serving.profile_reference import measured_unallocated_reference
from test_transfer_power import instant


def fixture(tmp_path):
    image='sha256:'+'a'*64
    added=dict(instance_id='m',tp=1,gpus=[0])
    kept=dict(instance_id='quiet-resident',tp=1,gpus=[1])
    manifest=dict(image=image,add=[added],keep=[kept],
        remove=[dict(instance_id='old'+str(g),tp=1,gpus=[g]) for g in range(8) if g!=1])
    prepared=dict(complete=True,errors=[],sampling_error=None,started_s=90.,finished_s=99.,
        manifest=manifest,instances=[dict(spec=added)])
    engine=Path(__file__).resolve().parents[2]/'src/ecopadg/serving/engine.py'
    provenance=[dict(instance_id='m',tp=1,cuda_visible_devices='0',image_id=image,
        model='/models/Qwen2.5-14B-Instruct',engine_version='0.9.2',
        source_files_at_import={str(engine):sha256(engine)})]
    power=[(100+j/10,[900,1000,30+j/10,15,16,17,18,19]) for j in range(81)]
    raw=dict(instant(power),complete=True,sampling_error=None,engine_provenance=provenance,
        topology=dict(mixed=dict(id='m',tp=1,gpus=[0])),runs=[
            dict(frequency_mhz=f,idle_start_s=100.25+2*j,idle_end_s=100.75+2*j)
            for j,f in enumerate((900,1500,2100,2520))])
    paths=dict(raw=str(tmp_path/'raw.json'),preparation=str(tmp_path/'startup.json'))
    write(paths,raw,prepared)
    return paths,raw,prepared


def write(paths,raw,prepared):
    Path(paths['raw']).write_text(json.dumps(raw))
    Path(paths['preparation']).write_text(json.dumps(prepared))


def test_reference_uses_completed_physical_layout_and_integrates_exact_idle_windows(tmp_path):
    paths,raw,prepared=fixture(tmp_path)
    # The quiet GPU1 is outside operator topology but explicitly remains
    # resident. Its high power must never enter the empty-GPU reference.
    originals={path:Path(path).read_bytes() for path in paths.values()}
    watts,evidence,artifacts=measured_unallocated_reference(paths)
    assert watts==pytest.approx(36.5)  # linear GPU2 mean on [106.25,106.75]
    assert evidence['free_gpus']==[2,3,4,5,6,7]
    assert evidence['resident_gpus']==[0,1]
    assert len(evidence['windows'])==4
    assert evidence['windows'][0]['watts_by_gpu']['2']==pytest.approx(30.5)
    assert all(set(w['watts_by_gpu'])==set(map(str,range(2,8))) for w in evidence['windows'])
    assert evidence['power_evidence']['power_source_verified']
    assert not evidence['formal_eligible'] and not evidence['measurement_changed']
    assert evidence['preparation_finished_s']==99
    assert originals=={path:Path(path).read_bytes() for path in paths.values()}
    assert artifacts[paths['raw']]==sha256(paths['raw'])
    assert artifacts[paths['preparation']]==sha256(paths['preparation'])
    assert any(path.endswith('/serving/profile_reference.py') for path in artifacts)
    assert any(path.endswith('/serving/engine.py') for path in artifacts)
    assert json.loads(json.dumps(evidence))==evidence


@pytest.mark.parametrize('defect', ['topology_id','topology_tp','topology_gpus','startup_missing',
    'startup_gpus','unknown_prior','overlapping_final','no_free','missing_keep'])
def test_reference_rejects_unproved_or_mismatched_physical_layout(tmp_path,defect):
    paths,raw,prepared=fixture(tmp_path)
    if defect=='topology_id':raw['topology']['mixed']['id']='unknown'
    elif defect=='topology_tp':raw['topology']['mixed']['tp']=2
    elif defect=='topology_gpus':raw['topology']['mixed']['gpus']=[2]
    elif defect=='startup_missing':prepared['instances']=[]
    elif defect=='startup_gpus':prepared['instances']=[dict(spec=dict(instance_id='m',tp=1,gpus=[2]))]
    elif defect=='unknown_prior':prepared['manifest']['remove'].pop()
    elif defect=='overlapping_final':prepared['manifest']['keep'][0]['gpus']=[0]
    elif defect=='missing_keep':prepared['manifest'].pop('keep')
    else:
        extra=[dict(instance_id='added'+str(g),tp=1,gpus=[g]) for g in range(2,8)]
        prepared['manifest']['add'].extend(extra)
        prepared['instances'].extend(dict(spec=x) for x in extra)
    write(paths,raw,prepared)
    with pytest.raises(ValueError):measured_unallocated_reference(paths)


@pytest.mark.parametrize('target', ['raw','preparation'])
@pytest.mark.parametrize('change', [dict(complete=False),dict(errors=['failure']),dict(sampling_error='failure')])
def test_reference_requires_both_measurements_complete(tmp_path,target,change):
    paths,raw,prepared=fixture(tmp_path)
    (raw if target=='raw' else prepared).update(change)
    write(paths,raw,prepared)
    with pytest.raises(ValueError,match='complete successful'):measured_unallocated_reference(paths)


@pytest.mark.parametrize('defect', ['missing','id','tp','devices','image','model','version','engine','duplicate'])
def test_reference_rejects_missing_stale_or_changed_engine_provenance(tmp_path,defect):
    paths,raw,prepared=fixture(tmp_path);record=raw['engine_provenance'][0]
    if defect=='missing':raw.pop('engine_provenance')
    elif defect=='duplicate':raw['engine_provenance'].append(dict(record))
    else:
        field,value={'id':('instance_id','unknown'),'tp':('tp',2),'devices':('cuda_visible_devices','2'),
            'image':('image_id','sha256:'+'c'*64),'model':('model','other'),
            'version':('engine_version','0.10.0'),'engine':('source_files_at_import',{'/x/serving/engine.py':'changed'})}[defect]
        record[field]=value
    write(paths,raw,prepared)
    with pytest.raises(ValueError,match='provenance'):measured_unallocated_reference(paths)


@pytest.mark.parametrize('defect', ['average','metadata_missing','metadata_stale','negative','nan',
    'nonmonotonic','short_row','empty','window_outside','window_before_prepare','window_empty','frequency_missing'])
def test_reference_requires_finite_instant_samples_and_contained_four_frequency_windows(tmp_path,defect):
    paths,raw,prepared=fixture(tmp_path)
    if defect=='average':raw['power_source']['mode']='average'
    elif defect=='metadata_missing':raw['power_metadata'].pop()
    elif defect=='metadata_stale':raw['power_metadata'][0]['nvml_timestamp_us'][2]-=1000000
    elif defect=='negative':raw['power_samples'][0][1][2]=-1
    elif defect=='nan':raw['power_samples'][0][1][2]=float('nan')
    elif defect=='nonmonotonic':raw['power_samples'][1]=raw['power_samples'][0]
    elif defect=='short_row':raw['power_samples'][0][1].pop()
    elif defect=='empty':raw['power_samples']=[]
    elif defect=='window_outside':raw['runs'][0]['idle_start_s']=99.5
    elif defect=='window_before_prepare':prepared['finished_s']=100.5
    elif defect=='window_empty':raw['runs'][0]['idle_end_s']=raw['runs'][0]['idle_start_s']
    else:raw['runs'].pop()
    write(paths,raw,prepared)
    with pytest.raises(ValueError):measured_unallocated_reference(paths)
