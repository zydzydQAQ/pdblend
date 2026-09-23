import pytest


def test_service_breakdown_preserves_cpu_wait_phases_without_calling_them_pure_wire():
    from pdblend_baselines.distserve.pipeline_service_profile import ServiceCalls
    timer=ServiceCalls(dict(rank=1,pp_rank=1,tp_rank=0))
    def work():
        timer.phase('prepare_input',lambda:None)
        timer.phase('pp_recv',lambda:None)
        timer.mark_model(dict(virtual_engine=1,is_prompt=False,input_tokens=2))
        assert timer.phase('model_runner',lambda:7)==7
        timer.phase('pp_send',lambda:None)
        return 42
    assert timer.around(work)==42
    row=timer.collect()[0]
    assert row['succeeded'] and row['is_model_call'] and row['virtual_engine']==1
    assert [r['phase'] for r in row['phases']]==['prepare_input','pp_recv','model_runner','pp_send']
    assert row['elapsed_s']>=sum(r['elapsed_s'] for r in row['phases'])
    assert row['recv_includes_upstream_wait'] is True and row['simulator_provider_eligible'] is False
    assert timer.collect()==[]


def test_failed_worker_phase_never_becomes_successful_service_sample():
    from pdblend_baselines.distserve.pipeline_service_profile import ServiceCalls
    timer=ServiceCalls(dict(rank=0))
    def fail():raise RuntimeError('communication failed')
    with pytest.raises(RuntimeError):timer.around(lambda:timer.phase('pp_send',fail))
    row=timer.collect()[0]
    assert not row['succeeded'] and not row['phases'][0]['succeeded']
