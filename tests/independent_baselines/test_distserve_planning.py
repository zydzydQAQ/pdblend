import pytest
from pdblend_baselines.distserve.planning import select_placement


def simulate(config,rate):
    return {'ttft_s':[.1]*12,'tpot_s':[.01]*12}


def test_select_placement_uses_per_gpu_goodput_and_replicates_within_eight_cards():
    result=select_placement(layers=28,attention_heads=28,allowed_tps=(1,),
        supported_pairs={(1,1)},measured_pairs={(1,1)},simulator=simulate,
        rate_rps=3.,ttft_s=1.,tpot_s=.1,max_per_gpu_rate=1,epsilon=.1)
    assert result['selected']['config']==(1,1,1,1,1)
    assert result['selected']['replicas']==2
    assert result['selected']['total_gpu_count']==4
    assert result['complete_search_space'] is False  # PP is an explicit engine restriction.


def test_missing_supported_profile_blocks_placement_by_default():
    with pytest.raises(ValueError,match='missing profile'):
        select_placement(layers=28,attention_heads=28,allowed_tps=(1,2),
            supported_pairs={(1,1),(2,1)},measured_pairs={(1,1)},simulator=simulate,
            rate_rps=1,ttft_s=1,tpot_s=.1)


def test_overload_is_recorded_with_maximum_available_replicas():
    result=select_placement(layers=28,attention_heads=28,allowed_tps=(1,),
        supported_pairs={(1,1)},measured_pairs={(1,1)},simulator=simulate,
        rate_rps=100.,ttft_s=1.,tpot_s=.1,max_per_gpu_rate=1,epsilon=.1)
    assert result['selected']['replicas']==4
    assert result['selected']['predicted_capacity_shortfall'] is True
    assert result['gpu_qualified'] is False
