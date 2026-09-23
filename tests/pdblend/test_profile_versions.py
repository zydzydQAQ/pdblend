from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

# Integration checks require the experiment machine's immutable raw artifacts.
pytestmark = pytest.mark.historical

from pdblend.profile.model import PerfModel
from pdblend.profile.timing_calibration import TimingOverlay
from pdblend.profile.versions import VersionError, load_version

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT/'results/2026-09-23/calibration-versions-v1/registry.json'


@pytest.fixture(scope='module')
def rows():
    if not REGISTRY.is_file():
        pytest.skip('actual campaign registry unavailable')
    return json.loads(REGISTRY.read_text())['versions']


def load(row, registry=REGISTRY, **kwargs):
    args=dict(system=row['system'],model_id=row['model_id'],tp=row['tp'],pp=row['pp'],usage='development')
    args.update(kwargs)
    return load_version(registry,row['version_id'],**args)


def write_changed_registry(tmp_path,row,change):
    """Rebind a derived test record so deeper evidence gates are exercised."""
    row=deepcopy(row);change(row)
    body={k:v for k,v in row.items() if k not in ('version_id','execution')}
    digest=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]
    row['version_id']=f"{row['model_id']}-tp{row['tp']}-pp{row['pp']}-{digest}"
    registry=json.loads(REGISTRY.read_text());registry['versions']=[row]
    path=tmp_path/'registry.json';path.write_text(json.dumps(registry))
    return row,path


@pytest.mark.parametrize('index',[0,1])
def test_actual_versions_compose_power_timing_and_energy(rows,index):
    row=rows[index];loaded=load(row)
    base=PerfModel.load(row['evidence']['power_candidate']['path'])
    expected=base
    if index:
        expected=TimingOverlay(base,json.loads(Path(row['evidence']['timing_candidate']['path']).read_text()))
    step=expected.step_seconds(32,1200,1500)
    power=base.decode_power_w(32,1500,ctx=1200)
    assert loaded.model.step_seconds(32,1200,1500)==step
    assert loaded.model.decode_power_w(32,1500,ctx=1200)==power
    assert loaded.model.token_energy_j(32,1200,1500)==step*power/32
    if index:
        assert loaded.model.token_energy_j(32,1200,1500)>base.token_energy_j(32,1200,1500)
    manifest=loaded.manifest_fields()
    assert manifest['profile_key']['version_id']==row['version_id']
    assert manifest['profile_key']==loaded.model.profile_key
    assert manifest['calibration_qualification']['formal_eligible'] is False
    assert manifest['calibration_qualification']['consumer_loader_available'] is True


@pytest.mark.parametrize('mismatch',[dict(system='distserve'),dict(model_id='Qwen2.5-14B-Instruct'),dict(tp=2),dict(pp=2)])
def test_identity_cannot_fall_back(rows,mismatch):
    with pytest.raises(VersionError,match='identity mismatch'):
        load(rows[0],**mismatch)


def test_formal_and_implicit_latest_rejected(rows):
    with pytest.raises(VersionError,match='missing gates'):
        load(rows[0],usage='formal')
    with pytest.raises(VersionError,match='missing or ambiguous'):
        load_version(REGISTRY,'latest',system='pdblend',model_id=rows[0]['model_id'],tp=4,pp=1,usage='development')


def test_registry_tampering_and_failed_components_rejected(rows,tmp_path):
    data=json.loads(REGISTRY.read_text());data['versions'][0]['missing_gates']=[]
    path=tmp_path/'tampered.json';path.write_text(json.dumps(data))
    with pytest.raises(VersionError,match='content identity checksum'):
        load(rows[0],registry=path)
    row,path=write_changed_registry(tmp_path,rows[0],lambda r:r.update(power_passed=False))
    with pytest.raises(VersionError,match='components did not pass'):
        load(row,registry=path)


def test_candidate_sha_tampering_rejected(rows,tmp_path):
    original=Path(rows[0]['evidence']['power_candidate']['path'])
    changed=tmp_path/'candidate.json';changed.write_bytes(original.read_bytes()+b' ')
    def change(row):
        row['evidence']['power_candidate']['path']=str(changed)
    row,path=write_changed_registry(tmp_path,rows[0],change)
    with pytest.raises(VersionError,match='evidence checksum mismatch'):
        load(row,registry=path)


@pytest.mark.parametrize('index',[0,1])
def test_all_exposed_queries_reject_outside_domain(rows,index):
    model=load(rows[index]).model
    for call in (lambda:model.step_seconds(32,1200,1000),
                 lambda:model.step_seconds(32,10000,1500),
                 lambda:model.step_seconds(float('nan'),1200,1500),
                 lambda:model.decode_power_w(32,1500),
                 lambda:model.decode_power_w(32,1500,ctx=10000),
                 lambda:model.prefill_seconds(7169,1500),
                 lambda:model.prefill_power_w(512,1000),
                 lambda:model.prefill_marginal_seconds(7169,1500),
                 lambda:model.token_energy_j(0,1200,1500)):
        with pytest.raises(VersionError):
            call()
    assert model.decode_supported(32,10000,1500) is False
