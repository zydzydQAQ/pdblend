import json
from types import SimpleNamespace

import pytest
from ecopadg.serving.cell import verify_formal_config
from ecopadg.serving.evidence import freeze_files


def test_formal_cell_rejects_unfrozen_policy_and_slo_overrides(tmp_path):
    proof=tmp_path/'proof.json';proof.write_text('{}')
    profile=tmp_path/'profiles.json';profile.write_text(json.dumps(dict(
        frequency_commands_verified=True,heldout_calibration_complete=True,mixed_interference_measured=True,
        instant_prefill_calibration_complete=True,instant_heldout_calibration_complete=True,
        resident_idle_measured=True,
        certification_artifacts=freeze_files([proof]))))
    config=dict(strategy='pdblend-joint',slo_ttft_s=5,slo_tpot_s=.1,profiles=str(profile),dvfs=False,power_mode='instant')
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    args=SimpleNamespace(config=path,strategy=None,slo_ttft_s=None,slo_tpot_s=None)
    freeze=dict(files=freeze_files([path,profile,proof]),groups=dict(protocol=[str(path)],
        profiles=[str(profile),str(proof)]))
    verify_formal_config(args,config,freeze)
    args.strategy='pdblend-dynamic'
    with pytest.raises(ValueError,match='strategy override'):
        verify_formal_config(args,config,freeze)
    args.strategy=None;args.slo_tpot_s=.2
    with pytest.raises(ValueError,match='SLO override'):
        verify_formal_config(args,config,freeze)
    args.slo_tpot_s=None
    path.write_text(json.dumps(dict(config,allow_unprofiled_fallback=True)))
    with pytest.raises(ValueError,match='not in the frozen protocol'):
        verify_formal_config(args,config,freeze)


def test_profiles_without_frequency_evidence_cannot_enter_formal_run(tmp_path):
    profile=tmp_path/'profiles.json';profile.write_text(json.dumps(dict(
        heldout_calibration_complete=True,mixed_interference_measured=True)))
    config=dict(strategy='mixed_dvfs',slo_ttft_s=5,slo_tpot_s=.1,profiles=str(profile),power_mode='instant')
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    args=SimpleNamespace(config=path,strategy=None,slo_ttft_s=None,slo_tpot_s=None)
    freeze=dict(files=freeze_files([path,profile]),groups=dict(protocol=[str(path)],profiles=[str(profile)]))
    with pytest.raises(ValueError,match='clock evidence'):
        verify_formal_config(args,config,freeze)


@pytest.mark.parametrize('missing', ['instant_prefill_calibration_complete',
                                    'instant_heldout_calibration_complete'])
def test_legacy_average_profiles_cannot_enter_formal_run(tmp_path, missing):
    profile=tmp_path/'profiles.json'
    value=dict(frequency_commands_verified=True,heldout_calibration_complete=True,
        mixed_interference_measured=True,instant_prefill_calibration_complete=True,
        instant_heldout_calibration_complete=True)
    value.pop(missing)
    profile.write_text(json.dumps(value))
    config=dict(strategy='mixed',slo_ttft_s=5,slo_tpot_s=.1,profiles=str(profile),power_mode='instant')
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    args=SimpleNamespace(config=path,strategy=None,slo_ttft_s=None,slo_tpot_s=None)
    freeze=dict(files=freeze_files([path,profile]),groups=dict(protocol=[str(path)],profiles=[str(profile)]))
    with pytest.raises(ValueError,match='instant prefill and held-out'):
        verify_formal_config(args,config,freeze)


def test_formal_configuration_cannot_implicitly_change_its_frozen_power_mode(tmp_path):
    config=dict(strategy='mixed')
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    args=SimpleNamespace(config=path,strategy=None,slo_ttft_s=None,slo_tpot_s=None)
    freeze=dict(files=freeze_files([path]),groups=dict(protocol=[str(path)]))
    with pytest.raises(ValueError,match='explicitly require instant power'):
        verify_formal_config(args,config,freeze)


@pytest.mark.parametrize('receiver_proof',[None,False,1,'true',True])
def test_formal_transfer_freeze_requires_sender_and_receiver_energy(tmp_path,receiver_proof):
    proof=tmp_path/'raw.json';proof.write_text('{}')
    sources=freeze_files([proof])
    profile=tmp_path/'profiles.json';profile.write_text(json.dumps(dict(
        frequency_commands_verified=True,heldout_calibration_complete=True,mixed_interference_measured=True,
        instant_prefill_calibration_complete=True,instant_heldout_calibration_complete=True,
        resident_idle_measured=True,certification_artifacts=sources)))
    links=[dict(source_tp=1,target_tp=1,incremental_j=2)]
    transfer=tmp_path/'transfers.json'
    value=dict(certified=True,instant_power_costs_verified=True,engine_image='image',
               links=links,certification_artifacts=sources)
    if receiver_proof is not None:value['receiver_transfer_energy_included']=receiver_proof
    transfer.write_text(json.dumps(value))
    config=dict(strategy='distserve',profiles=str(profile),transfers=links,
                transfer_evidence=str(transfer),power_mode='instant')
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    args=SimpleNamespace(config=path,strategy=None,slo_ttft_s=None,slo_tpot_s=None)
    freeze=dict(files=freeze_files([path,profile,proof,transfer]),groups=dict(protocol=[str(path)],
        profiles=[str(profile),str(proof),str(transfer)]),identities=dict(engine_image='image'))
    if receiver_proof is True:verify_formal_config(args,config,freeze)
    else:
        with pytest.raises(ValueError,match='frozen hardware evidence'):
            verify_formal_config(args,config,freeze)
