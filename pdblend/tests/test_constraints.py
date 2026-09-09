# -*- coding: utf-8 -*-
from __future__ import annotations

from ecopadg.constraints import (
    InstanceState, PoolView, RequestRecord, att_would_drop,
    check_constraints, constraint_reason, path_gross_j, pool_slo_ok,
)
from tests.conftest import SynthOpModel


def test_check_constraints_ttft():
    inst = InstanceState(idx=0, t_switch=0.0)
    req = RequestRecord(rid=1, arrival_time=0.0, output_len=10,
                        first_token_time=0.0, input_len=100)
    idx = check_constraints(
        [inst], req, s_ttft=5.0, s_tpot=0.1,
        prefill_time_fn=lambda n: 0.01 * n, now=0.0)
    assert idx == 0
    assert constraint_reason(
        inst, req, s_ttft=0.1, s_tpot=0.1,
        prefill_time_fn=lambda n: 1.0, now=0.0) == "ttft"


def test_pool_slo_ok_and_drop():
    view = PoolView(name="m", pending_prefill_s=0.2, tpot_slack_s=2.0)
    assert pool_slo_ok(view, 0.3, s_ttft=5.0)
    assert not pool_slo_ok(view, 5.0, s_ttft=5.0)
    assert att_would_drop(0.82, 0.99, 0.01)
    assert not att_would_drop(0.956, 0.960, 0.01)


def test_path_gross_j_pd_adds_kv(synth_opmodel):
    jm = path_gross_j(synth_opmodel, "mixed", 2000, 64, 0.0,
                      kv_xfer_s_per_tok=0.0)
    jp = path_gross_j(synth_opmodel, "pd", 2000, 64, 0.0,
                      kv_xfer_s_per_tok=1.0)
    assert jp > jm
