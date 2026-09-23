import json
from pathlib import Path

import pytest

# Integration checks require the experiment machine's immutable raw artifacts.
pytestmark = pytest.mark.historical

from pdblend.profile.model import PerfModel
from pdblend.profile.versions import VersionError, load_version

ROOT=Path(__file__).resolve().parents[2]
REGISTRY=ROOT/'results/2026-09-23/calibration-incremental-versions-v1/registry.json'


@pytest.fixture(scope='module')
def versions():
    data=json.loads(REGISTRY.read_text())
    return [(row,load_version(REGISTRY,row['version_id'],system='pdblend',model_id=row['model_id'],
                 tp=row['tp'],pp=1,usage='development')) for row in data['versions']]


@pytest.mark.parametrize('index',[0,1])
def test_actual_long_union_keeps_short_and_gaps(versions,index):
    row,loaded=versions[index];model=loaded.model
    original=PerfModel.load(row['evidence']['power_candidate']['path'])
    assert model.step_seconds(4,1024,1500)==original.step_seconds(4,1024,1500)
    assert model.decode_power_w(4,1500,ctx=1024)==original.decode_power_w(4,1500,ctx=1024)
    for batch in row['long_domain']['exact_batches']:
        assert model.step_seconds(batch,7000,1500)>0
        assert model.token_energy_j(batch,7000,1500)==model.step_seconds(batch,7000,1500)*model.decode_power_w(batch,1500,ctx=7000)/batch
        assert not model.decode_supported(batch,4800,1500)
        with pytest.raises(VersionError):model.step_seconds(batch,4800,1500)
    for batch in (2,3,6):
        assert not model.decode_supported(batch,7000,1500)
        with pytest.raises(VersionError):model.decode_power_w(batch,1500,ctx=7000)
    with pytest.raises(VersionError):model.step_seconds(1,8000,1500)
    assert loaded.profile_key['long_candidate_sha256']==row['evidence']['long_candidate']['sha256']


def test_actual_local_power_and_repaired_mixed_remain_bounded(versions):
    row,loaded=versions[2]
    assert row['component_kind']=='local_power_and_repaired_mixed'
    assert row['validated_power_batches']==[1,128]
    assert loaded.qualification['formal_eligible'] is False
    for b in (1,128):
        assert loaded.model.step_seconds(b,1200,1500)>0
        assert loaded.model.decode_power_w(b,1500,ctx=1200)>0
    with pytest.raises(VersionError):loaded.model.prefill_seconds(127,1500)
    with pytest.raises(VersionError):
        load_version(REGISTRY,row['version_id'],system='pdblend',model_id=row['model_id'],tp=4,pp=1,usage='formal')
